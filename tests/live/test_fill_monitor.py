"""app.live.fill_monitor — check_pending_fills's matching/classification is
pure enough to test directly against fake rows/executions (DB reads/writes
and get_clock are all faked). No network/DB (E-stage)."""

from __future__ import annotations

from contextlib import nullcontext
from datetime import datetime, timedelta, timezone

import pytest

from app.domain.models import BrokerExecution
from app.execution.ledger import Position
from app.live import alerts as alerts_module
from app.live import fill_monitor


class _FixedClock:
    def __init__(self, now: datetime):
        self._now = now

    def now(self, tz=None):
        return self._now if tz is None else self._now.astimezone(tz)


def _execution(order_no="ORD1", filled_qty=10.0, order_qty=10.0, cancelled=False, rejected=False) -> BrokerExecution:
    remaining = max(order_qty - filled_qty, 0.0)
    return BrokerExecution(
        order_date="20260813", order_no=order_no, code="005930", name="삼성전자", side="buy",
        order_qty=order_qty, filled_qty=filled_qty, remaining_qty=remaining,
        avg_price=70_000.0, filled_amount=filled_qty * 70_000.0, cancelled=cancelled, rejected=rejected,
    )


def _row(
    trade_id=1, ticker="005930.KS", side="buy", shares=10, order_no="ORD1", created_at=None,
    fill_unresolved_alerted_at=None,
) -> dict:
    return {
        "id": trade_id, "ticker": ticker, "side": side, "shares": shares,
        "price": 70_000.0, "created_at": created_at or datetime(2026, 8, 13, 9, 0, tzinfo=timezone.utc),
        "kis_order_no": order_no, "fill_unresolved_alerted_at": fill_unresolved_alerted_at,
    }


GIVE_UP = timedelta(minutes=30)
NOW = datetime(2026, 8, 13, 9, 20, tzinfo=timezone.utc)  # 20 min after the default row's created_at


@pytest.fixture(autouse=True)
def _clock(monkeypatch):
    monkeypatch.setattr(fill_monitor, "get_clock", lambda: _FixedClock(NOW))
    # filter_new_alerts' cooldown is keyed only by (ticker, kind), process-
    # global — without clearing it, an alert fired by an earlier test in
    # this file (same ticker/kind) suppresses a later test's alert too. Same
    # fix already applied in test_auto_trade_loop.py/test_alerts.py.
    alerts_module._last_fired_at.clear()


async def test_no_execution_found_young_order_neither_alerts_nor_marks_checked(monkeypatch):
    monkeypatch.setattr(fill_monitor, "_pending_fill_checks", lambda: _async_list([_row()]))
    marked = []
    monkeypatch.setattr(fill_monitor, "_mark_checked", lambda trade_id: _async_append(marked, trade_id))
    recorded = []
    monkeypatch.setattr(fill_monitor, "record_alert", lambda alert: _async_append(recorded, alert))

    await fill_monitor.check_pending_fills(domestic=[], overseas=[], give_up_after=GIVE_UP)

    assert recorded == []
    assert marked == []


async def test_no_execution_found_past_give_up_alerts_once_but_stays_re_examinable(monkeypatch):
    # 2026-08-27 fix: past give_up_after but nowhere near _PENDING_LOOKBACK
    # (2 days) — must alert once, but via _mark_unresolved_alerted, NOT
    # _mark_checked. Marking it fully checked here is exactly the bug that
    # let a genuinely-slow-but-real fill (MRVL order_no=0000043249) get
    # permanently excluded from every later re-check.
    old_row = _row(created_at=NOW - timedelta(hours=1))  # well past the 30-min give-up window
    monkeypatch.setattr(fill_monitor, "_pending_fill_checks", lambda: _async_list([old_row]))
    marked = []
    monkeypatch.setattr(fill_monitor, "_mark_checked", lambda trade_id: _async_append(marked, trade_id))
    alerted = []
    monkeypatch.setattr(fill_monitor, "_mark_unresolved_alerted", lambda trade_id: _async_append(alerted, trade_id))
    recorded = []
    monkeypatch.setattr(fill_monitor, "record_alert", lambda alert: _async_append(recorded, alert))

    await fill_monitor.check_pending_fills(domestic=[], overseas=[], give_up_after=GIVE_UP)

    assert len(recorded) == 1
    assert recorded[0].kind == "unconfirmed_fill"
    assert "no KIS execution record found" in recorded[0].message
    assert alerted == [1]
    assert marked == []  # must NOT be permanently excluded from re-checking


async def test_no_execution_found_already_alerted_keeps_re_checking_silently(monkeypatch):
    # The interim alert must fire at most once per order — a row already
    # marked fill_unresolved_alerted_at (from a prior pass) must neither
    # re-alert nor get marked/re-marked; it just stays pending for the next
    # pass, same as before this alert ever fired.
    old_row = _row(
        created_at=NOW - timedelta(hours=1),
        fill_unresolved_alerted_at=NOW - timedelta(minutes=30),
    )
    monkeypatch.setattr(fill_monitor, "_pending_fill_checks", lambda: _async_list([old_row]))
    marked = []
    monkeypatch.setattr(fill_monitor, "_mark_checked", lambda trade_id: _async_append(marked, trade_id))
    alerted = []
    monkeypatch.setattr(fill_monitor, "_mark_unresolved_alerted", lambda trade_id: _async_append(alerted, trade_id))
    recorded = []
    monkeypatch.setattr(fill_monitor, "record_alert", lambda alert: _async_append(recorded, alert))

    await fill_monitor.check_pending_fills(domestic=[], overseas=[], give_up_after=GIVE_UP)

    assert recorded == []
    assert marked == []
    assert alerted == []


async def test_no_execution_found_past_pending_lookback_gives_up_for_good(monkeypatch):
    # Once _PENDING_LOOKBACK (2 days) itself is reached with still no
    # matching record, the row IS finally marked checked — further waiting
    # has nothing left to gain, and this is the one case that should stop
    # being re-examined.
    ancient_row = _row(
        created_at=NOW - timedelta(days=3),
        fill_unresolved_alerted_at=NOW - timedelta(days=2, hours=12),
    )
    monkeypatch.setattr(fill_monitor, "_pending_fill_checks", lambda: _async_list([ancient_row]))
    marked = []
    monkeypatch.setattr(fill_monitor, "_mark_checked", lambda trade_id: _async_append(marked, trade_id))
    alerted = []
    monkeypatch.setattr(fill_monitor, "_mark_unresolved_alerted", lambda trade_id: _async_append(alerted, trade_id))
    recorded = []
    monkeypatch.setattr(fill_monitor, "record_alert", lambda alert: _async_append(recorded, alert))

    await fill_monitor.check_pending_fills(domestic=[], overseas=[], give_up_after=GIVE_UP)

    assert len(recorded) == 1
    assert recorded[0].kind == "unconfirmed_fill_final"
    assert marked == [1]
    assert alerted == []


async def test_cancelled_zero_fill_alerts_immediately_even_when_young(monkeypatch):
    # `cancelled=True` is what makes this immediate — see the new
    # `test_zero_fill_still_open_waits_before_alerting` for the same
    # execution WITHOUT that flag, which must now wait instead.
    monkeypatch.setattr(fill_monitor, "_pending_fill_checks", lambda: _async_list([_row()]))  # young (0 age)
    execution = _execution(filled_qty=0.0, cancelled=True)
    marked = []
    monkeypatch.setattr(fill_monitor, "_mark_checked", lambda trade_id: _async_append(marked, trade_id))
    recorded = []
    monkeypatch.setattr(fill_monitor, "record_alert", lambda alert: _async_append(recorded, alert))

    await fill_monitor.check_pending_fills(domestic=[execution], overseas=[], give_up_after=GIVE_UP)

    assert len(recorded) == 1
    assert recorded[0].kind == "unconfirmed_fill"
    assert "never filled" in recorded[0].message
    assert marked == [1]


async def test_partial_fill_still_open_waits_before_alerting(monkeypatch):
    monkeypatch.setattr(fill_monitor, "_pending_fill_checks", lambda: _async_list([_row()]))  # young
    execution = _execution(filled_qty=4.0, order_qty=10.0)
    marked = []
    monkeypatch.setattr(fill_monitor, "_mark_checked", lambda trade_id: _async_append(marked, trade_id))
    recorded = []
    monkeypatch.setattr(fill_monitor, "record_alert", lambda alert: _async_append(recorded, alert))

    await fill_monitor.check_pending_fills(domestic=[execution], overseas=[], give_up_after=GIVE_UP)

    assert recorded == []
    assert marked == []


async def test_partial_fill_still_open_alerts_after_give_up(monkeypatch):
    old_row = _row(created_at=NOW - timedelta(hours=1))
    monkeypatch.setattr(fill_monitor, "_pending_fill_checks", lambda: _async_list([old_row]))
    execution = _execution(filled_qty=4.0, order_qty=10.0)
    marked = []
    monkeypatch.setattr(fill_monitor, "_mark_checked", lambda trade_id: _async_append(marked, trade_id))
    recorded = []
    monkeypatch.setattr(fill_monitor, "record_alert", lambda alert: _async_append(recorded, alert))

    await fill_monitor.check_pending_fills(domestic=[execution], overseas=[], give_up_after=GIVE_UP)

    assert len(recorded) == 1
    assert "4/10" in recorded[0].message or "4" in recorded[0].message
    assert marked == [1]


async def test_zero_fill_not_cancelled_or_rejected_young_neither_alerts_nor_marks_checked(monkeypatch):
    # 2026-08-27: THE regression this fix exists for. A young overseas limit
    # order that simply hasn't been hit yet reports filled_qty=0 with the
    # full quantity still open and NO cancelled/rejected flag — it must be
    # left pending, exactly like a partial fill still working. Before the
    # fix this fired a false "never filled" alert AND marked the row checked
    # forever, so the real fill could never be reconciled afterwards.
    monkeypatch.setattr(fill_monitor, "_pending_fill_checks", lambda: _async_list([_row()]))  # young (0 age)
    execution = _execution(filled_qty=0.0, order_qty=10.0)  # remaining 10, no dead-order flag
    marked = []
    monkeypatch.setattr(fill_monitor, "_mark_checked", lambda trade_id: _async_append(marked, trade_id))
    alerted = []
    monkeypatch.setattr(fill_monitor, "_mark_unresolved_alerted", lambda trade_id: _async_append(alerted, trade_id))
    recorded = []
    monkeypatch.setattr(fill_monitor, "record_alert", lambda alert: _async_append(recorded, alert))

    await fill_monitor.check_pending_fills(domestic=[execution], overseas=[], give_up_after=GIVE_UP)

    assert recorded == []
    assert marked == []   # must stay re-examinable — the fill may still land
    assert alerted == []


async def test_zero_fill_still_open_alerts_after_give_up(monkeypatch):
    # Same shape, aged past give_up_after: now it IS worth telling a human
    # about, and the message must say 0 filled rather than claiming KIS
    # confirmed the order dead (it didn't).
    old_row = _row(created_at=NOW - timedelta(hours=1))
    monkeypatch.setattr(fill_monitor, "_pending_fill_checks", lambda: _async_list([old_row]))
    execution = _execution(filled_qty=0.0, order_qty=10.0)
    marked = []
    monkeypatch.setattr(fill_monitor, "_mark_checked", lambda trade_id: _async_append(marked, trade_id))
    recorded = []
    monkeypatch.setattr(fill_monitor, "record_alert", lambda alert: _async_append(recorded, alert))

    await fill_monitor.check_pending_fills(domestic=[execution], overseas=[], give_up_after=GIVE_UP)

    assert len(recorded) == 1
    assert recorded[0].kind == "unconfirmed_fill"
    assert "0/10 filled" in recorded[0].message
    assert "10 still open" in recorded[0].message
    assert "never filled" not in recorded[0].message  # NOT a confirmed dead order
    assert marked == [1]


async def test_rejected_order_alerts_immediately_even_when_young(monkeypatch):
    # `rejected` is the overseas equivalent of domestic cncl_yn (see
    # BrokerExecution.cancelled's docstring) — a rejected order never
    # reached the market and still reports its full quantity as 미체결, so
    # it must NOT be mistaken for a working order and must close out at once.
    monkeypatch.setattr(fill_monitor, "_pending_fill_checks", lambda: _async_list([_row(ticker="MRVL")]))
    execution = _execution(filled_qty=0.0, order_qty=10.0, rejected=True)
    marked = []
    monkeypatch.setattr(fill_monitor, "_mark_checked", lambda trade_id: _async_append(marked, trade_id))
    recorded = []
    monkeypatch.setattr(fill_monitor, "record_alert", lambda alert: _async_append(recorded, alert))

    await fill_monitor.check_pending_fills(domestic=[], overseas=[execution], give_up_after=GIVE_UP)

    assert len(recorded) == 1
    assert "never filled" in recorded[0].message
    assert "rejected" in recorded[0].message
    assert marked == [1]


async def test_zero_fill_with_nothing_left_open_is_treated_as_confirmed_dead(monkeypatch):
    # KIS says the order neither filled NOR has anything open: terminal, and
    # the one zero-fill shape that still closes out immediately without a
    # cancelled/rejected flag. Without this disjunct it would match neither
    # dead branch nor `remaining_qty > 0` and fall into the fully-filled
    # tail — silently marked checked with a DEBUG line reading "confirmed
    # filled (0 @ ...)", i.e. a phantom trade signed off as real.
    monkeypatch.setattr(fill_monitor, "_pending_fill_checks", lambda: _async_list([_row()]))  # young
    execution = _execution(filled_qty=0.0, order_qty=0.0)  # remaining 0, no flags
    marked = []
    monkeypatch.setattr(fill_monitor, "_mark_checked", lambda trade_id: _async_append(marked, trade_id))
    recorded = []
    monkeypatch.setattr(fill_monitor, "record_alert", lambda alert: _async_append(recorded, alert))

    await fill_monitor.check_pending_fills(domestic=[execution], overseas=[], give_up_after=GIVE_UP)

    assert len(recorded) == 1
    assert "never filled" in recorded[0].message
    assert marked == [1]


async def test_cancelled_after_partial_fill_reverses_only_the_unfilled_remainder(monkeypatch):
    # cncl_yn can sit on an order that partially executed before its
    # remainder was cancelled. Reversing the full recorded quantity there
    # would erase the shares that DID fill — a real position, with its
    # stop-loss, deleted from the ledger. Only the unfilled part is reversed.
    monkeypatch.setattr(fill_monitor.settings, "fill_monitor_auto_correct_enabled", True)
    monkeypatch.setattr(fill_monitor, "_corrections_today", lambda: _async_value(0))
    monkeypatch.setattr(fill_monitor, "_pending_fill_checks", lambda: _async_list([_row(shares=10)]))  # young
    execution = _execution(filled_qty=4.0, order_qty=10.0, cancelled=True)
    called = []

    async def fake_auto_correct(row, reversal_shares, resolution):
        called.append((reversal_shares, resolution))
        return True

    monkeypatch.setattr(fill_monitor, "_auto_correct", fake_auto_correct)
    monkeypatch.setattr(fill_monitor, "_mark_checked", lambda trade_id: _async_value(None))
    recorded = []
    monkeypatch.setattr(fill_monitor, "record_alert", lambda alert: _async_append(recorded, alert))

    await fill_monitor.check_pending_fills(domestic=[execution], overseas=[], give_up_after=GIVE_UP)

    assert called == [(6, "dead after partial fill 4/10")]  # 10 recorded - 4 filled
    assert "4/10" in recorded[0].message
    assert "cancelled" in recorded[0].message


async def test_cancelled_but_fully_filled_is_not_announced_as_dead(monkeypatch):
    # A cancel request that lost the race to the fill: the flag is set but
    # nothing is unfilled, so there is nothing to reverse and nothing to
    # warn about — it must resolve quietly like any other completed order.
    monkeypatch.setattr(fill_monitor, "_pending_fill_checks", lambda: _async_list([_row(shares=10)]))
    execution = _execution(filled_qty=10.0, order_qty=10.0, cancelled=True)
    marked = []
    monkeypatch.setattr(fill_monitor, "_mark_checked", lambda trade_id: _async_append(marked, trade_id))
    recorded = []
    monkeypatch.setattr(fill_monitor, "record_alert", lambda alert: _async_append(recorded, alert))

    await fill_monitor.check_pending_fills(domestic=[execution], overseas=[], give_up_after=GIVE_UP)

    assert recorded == []
    assert marked == [1]


async def test_fully_filled_marks_checked_without_alert(monkeypatch):
    monkeypatch.setattr(fill_monitor, "_pending_fill_checks", lambda: _async_list([_row()]))
    execution = _execution(filled_qty=10.0, order_qty=10.0)
    marked = []
    monkeypatch.setattr(fill_monitor, "_mark_checked", lambda trade_id: _async_append(marked, trade_id))
    recorded = []
    monkeypatch.setattr(fill_monitor, "record_alert", lambda alert: _async_append(recorded, alert))

    await fill_monitor.check_pending_fills(domestic=[execution], overseas=[], give_up_after=GIVE_UP)

    assert recorded == []
    assert marked == [1]


async def test_matches_by_order_no_exactly_not_by_position_in_list(monkeypatch):
    monkeypatch.setattr(fill_monitor, "_pending_fill_checks", lambda: _async_list([_row(order_no="ORD2")]))
    wrong = _execution(order_no="ORD1", filled_qty=10.0, order_qty=10.0)
    right = _execution(order_no="ORD2", filled_qty=10.0, order_qty=10.0)
    marked = []
    monkeypatch.setattr(fill_monitor, "_mark_checked", lambda trade_id: _async_append(marked, trade_id))
    recorded = []
    monkeypatch.setattr(fill_monitor, "record_alert", lambda alert: _async_append(recorded, alert))

    await fill_monitor.check_pending_fills(domestic=[wrong, right], overseas=[], give_up_after=GIVE_UP)

    assert recorded == []  # matched ORD2 (fully filled), not ORD1
    assert marked == [1]


async def test_matches_overseas_executions_too(monkeypatch):
    monkeypatch.setattr(fill_monitor, "_pending_fill_checks", lambda: _async_list([_row(ticker="AAPL")]))
    execution = _execution(filled_qty=10.0, order_qty=10.0)
    marked = []
    monkeypatch.setattr(fill_monitor, "_mark_checked", lambda trade_id: _async_append(marked, trade_id))
    recorded = []
    monkeypatch.setattr(fill_monitor, "record_alert", lambda alert: _async_append(recorded, alert))

    await fill_monitor.check_pending_fills(domestic=[], overseas=[execution], give_up_after=GIVE_UP)

    assert marked == [1]


# --- _same_order_no / overseas zero-padding mismatch (2026-08-27 fix) ------
#
# Real incident: `trades.kis_order_no` always stores KIS's zero-padded ODNO
# ('0000039147'), but the overseas execution-history TR returns it unpadded
# ('39147') — a plain `==` compare can never match an overseas order, so
# overseas fill confirmation silently never worked. See `_same_order_no`'s
# own docstring for the full incident (MRVL/SNDK genuine phantoms fell into
# the ambiguous "not found" branch instead of the confirmed-dead-order one;
# AXTI/GOOG/GOOGL/AMZN were false-positive "phantom" alerts for orders that
# had, in fact, fully filled).


def test_same_order_no_matches_across_zero_padding():
    assert fill_monitor._same_order_no("0000039147", "39147") is True
    assert fill_monitor._same_order_no("39147", "0000039147") is True


def test_same_order_no_matches_identical_padded_strings():
    assert fill_monitor._same_order_no("0000043249", "0000043249") is True


def test_same_order_no_rejects_off_by_one():
    assert fill_monitor._same_order_no("43249", "0000043250") is False


def test_same_order_no_never_matches_on_empty_values():
    # '' and '0000000000' must NOT both normalize to '' and compare equal —
    # that would mistake an unrelated order for this one's confirmation,
    # exactly the wrong failure direction (a real phantom trade reading as
    # "confirmed filled").
    assert fill_monitor._same_order_no("", "0000000000") is False
    assert fill_monitor._same_order_no(None, "0000043249") is False
    assert fill_monitor._same_order_no("0000043249", None) is False


# --- normalize_kis_order_no (extracted 2026-08-27 for app.audit.slippage_validation) ---


def test_normalize_kis_order_no_strips_leading_zeros():
    assert fill_monitor.normalize_kis_order_no("0000043249") == "43249"


def test_normalize_kis_order_no_leaves_unpadded_value_unchanged():
    assert fill_monitor.normalize_kis_order_no("43249") == "43249"


def test_normalize_kis_order_no_handles_none_and_empty():
    assert fill_monitor.normalize_kis_order_no(None) == ""
    assert fill_monitor.normalize_kis_order_no("") == ""


async def test_overseas_order_matches_despite_unpadded_kis_order_no(monkeypatch):
    # Integration-level regression: before this fix, this exact shape
    # (ledger's zero-padded kis_order_no vs. KIS's unpadded overseas
    # order_no) fell into the "no KIS execution record found" branch no
    # matter how the order actually resolved.
    row = _row(ticker="MRVL", order_no="0000039147")  # young — must not wait for give-up
    monkeypatch.setattr(fill_monitor, "_pending_fill_checks", lambda: _async_list([row]))
    # `rejected=True`, not just filled_qty=0: as of the 2026-08-27
    # confirmed-dead fix, zero fill alone is no longer a dead-order signal
    # (it is also a healthy resting limit order), and `rejected` — never
    # `cancelled` — is the field an OVERSEAS execution carries that answer
    # in. See BrokerExecution.cancelled's own docstring. What this test is
    # actually guarding is the zero-padding order-number MATCH reaching a
    # classified branch at all, not the classification itself.
    execution = _execution(order_no="39147", filled_qty=0.0, order_qty=56.0, rejected=True)
    marked = []
    monkeypatch.setattr(fill_monitor, "_mark_checked", lambda trade_id: _async_append(marked, trade_id))
    recorded = []
    monkeypatch.setattr(fill_monitor, "record_alert", lambda alert: _async_append(recorded, alert))

    await fill_monitor.check_pending_fills(domestic=[], overseas=[execution], give_up_after=GIVE_UP)

    # Matched immediately (no give-up wait) via the confirmed-dead-order
    # branch, not the ambiguous "not found" one.
    assert len(recorded) == 1
    assert "never filled" in recorded[0].message
    assert marked == [1]


async def test_not_found_alert_reports_execution_rows_seen_this_pass(monkeypatch):
    # Observability fix: the "not found" alert now says how many domestic/
    # overseas rows this pass actually saw — zero on the relevant side
    # points at a broken lookup, a nonzero count that still doesn't match
    # points at a genuine mismatch/lag instead.
    old_row = _row(created_at=NOW - timedelta(hours=1), order_no="0000099999")
    monkeypatch.setattr(fill_monitor, "_pending_fill_checks", lambda: _async_list([old_row]))
    monkeypatch.setattr(fill_monitor, "_mark_unresolved_alerted", lambda trade_id: _async_append([], trade_id))
    recorded = []
    monkeypatch.setattr(fill_monitor, "record_alert", lambda alert: _async_append(recorded, alert))
    other_overseas = _execution(order_no="11111", filled_qty=1.0, order_qty=1.0)

    await fill_monitor.check_pending_fills(domestic=[], overseas=[other_overseas], give_up_after=GIVE_UP)

    assert len(recorded) == 1
    assert "0 domestic" in recorded[0].message
    assert "1 overseas" in recorded[0].message


async def _async_list(value):
    return value


async def _async_append(lst, value):
    lst.append(value)


# ---------------------------------------------------------------------------
# fill_monitor_loop — skips a scope with no broker, never touches the ledger
# ---------------------------------------------------------------------------


async def test_fill_monitor_loop_skips_scope_without_broker(monkeypatch):
    import asyncio

    from app.config import account_mode_scope
    from app.user_context import ServicedScope, user_scope

    async def one_scope():
        return [ServicedScope("default", "paper")]

    async def no_broker():
        return None

    async def fail_if_called():
        raise AssertionError("must not fetch executions without a configured broker")

    monkeypatch.setattr(fill_monitor, "serviced_scopes", one_scope)
    monkeypatch.setattr("app.web.state.kis_broker", no_broker)
    monkeypatch.setattr("app.web.state.recent_executions", fail_if_called)

    task = asyncio.create_task(fill_monitor.fill_monitor_loop())
    await asyncio.sleep(0.05)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


async def test_fill_monitor_loop_skips_recent_executions_when_nothing_pending(monkeypatch):
    """FIX-4 (2026-09-03 id=1187 audit): a broker IS connected, but there are
    no pending trades to reconcile — `recent_executions()` (a real KIS
    execution-history ledger TR) must not fire at all in that case. Every
    quiet pass used to call it anyway, adding to the same EGW00215
    per-second quota pressure for no possible outcome."""
    import asyncio

    from app.user_context import ServicedScope

    async def one_scope():
        return [ServicedScope("default", "paper")]

    async def has_broker():
        return object()

    async def no_pending():
        return []

    async def fail_if_called():
        raise AssertionError("must not fetch executions when there is nothing pending to reconcile")

    monkeypatch.setattr(fill_monitor, "serviced_scopes", one_scope)
    monkeypatch.setattr("app.web.state.kis_broker", has_broker)
    monkeypatch.setattr("app.web.state.recent_executions", fail_if_called)
    monkeypatch.setattr(fill_monitor, "_pending_fill_checks", no_pending)

    task = asyncio.create_task(fill_monitor.fill_monitor_loop())
    await asyncio.sleep(0.05)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


async def test_fill_monitor_loop_still_fetches_executions_when_something_pending(monkeypatch):
    """Companion regression guard: with a pending row, the existing behavior
    (fetch executions, run check_pending_fills) must be unchanged."""
    import asyncio

    from app.user_context import ServicedScope

    async def one_scope():
        return [ServicedScope("default", "paper")]

    async def has_broker():
        return object()

    async def one_pending():
        return [_row()]

    calls = {"n": 0}

    async def fake_recent_executions():
        calls["n"] += 1
        return [], []

    async def fake_check_pending_fills(domestic, overseas, give_up_after, pending=None):
        assert pending is not None and len(pending) == 1

    monkeypatch.setattr(fill_monitor, "serviced_scopes", one_scope)
    monkeypatch.setattr("app.web.state.kis_broker", has_broker)
    monkeypatch.setattr("app.web.state.recent_executions", fake_recent_executions)
    monkeypatch.setattr(fill_monitor, "_pending_fill_checks", one_pending)
    monkeypatch.setattr(fill_monitor, "check_pending_fills", fake_check_pending_fills)

    task = asyncio.create_task(fill_monitor.fill_monitor_loop())
    await asyncio.sleep(0.05)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    assert calls["n"] == 1


# ---------------------------------------------------------------------------
# _pending_fill_checks / _mark_checked — DB degrades quietly, same
# fail-open convention as app.execution.reconciliation.
# ---------------------------------------------------------------------------


async def test_pending_fill_checks_degrades_on_db_error(monkeypatch):
    async def broken_get_pool():
        raise RuntimeError("pool unavailable")

    monkeypatch.setattr(fill_monitor, "get_pool", broken_get_pool)
    rows = await fill_monitor._pending_fill_checks()
    assert rows == []


async def test_mark_checked_degrades_on_db_error(monkeypatch):
    async def broken_get_pool():
        raise RuntimeError("pool unavailable")

    monkeypatch.setattr(fill_monitor, "get_pool", broken_get_pool)
    await fill_monitor._mark_checked(1)  # no raise = pass


# ---------------------------------------------------------------------------
# Stage 2 — _auto_correct's own safety gates, tested directly (not through
# check_pending_fills) so each gate's pass/fail branch is unambiguous.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _no_op_lock(monkeypatch):
    # market_lock is a real Postgres advisory lock outside unit-test scope —
    # same stub convention as tests/execution/test_ledger.py.
    monkeypatch.setattr(fill_monitor, "market_lock", lambda market: nullcontext())


async def test_auto_correct_skips_when_no_longer_most_recent_trade(monkeypatch):
    monkeypatch.setattr(fill_monitor, "_is_most_recent_trade", lambda ticker, trade_id: _async_value(False))
    called = []
    monkeypatch.setattr(fill_monitor, "reverse_pending_trade", lambda *a, **kw: _async_append(called, (a, kw)))

    corrected = await fill_monitor._auto_correct(_row(), reversal_shares=10, resolution="order never filled")

    assert corrected is False
    assert called == []


async def test_auto_correct_reverses_full_zero_fill_leaving_position_flat(monkeypatch):
    # get_position reflects CURRENT ledger state — i.e. it already includes
    # the phantom buy's own (wrong) effect. Reversing all 10 of it (a sell)
    # brings the 10-share current position back to exactly 0.
    monkeypatch.setattr(fill_monitor, "_is_most_recent_trade", lambda ticker, trade_id: _async_value(True))
    monkeypatch.setattr(fill_monitor, "get_position", lambda ticker: _async_value(Position(cash=1_000_000.0, shares=10)))
    calls = []
    monkeypatch.setattr(
        fill_monitor, "reverse_pending_trade",
        lambda ticker, side, shares, price, reason, prior_stop_loss=None: _async_append(
            calls, dict(ticker=ticker, side=side, shares=shares, price=price, reason=reason, prior_stop_loss=prior_stop_loss)
        ),
    )

    corrected = await fill_monitor._auto_correct(_row(shares=10), reversal_shares=10, resolution="order never filled")

    assert corrected is True
    assert len(calls) == 1
    assert calls[0]["shares"] == 10
    assert calls[0]["prior_stop_loss"] is None  # flat afterward — nothing to protect
    assert calls[0]["reason"].startswith(fill_monitor._CORRECTION_REASON)


async def test_auto_correct_carries_forward_prior_stop_loss_when_position_stays_open(monkeypatch):
    # A phantom BUY of 10 on top of an existing 5-share position: reversing
    # it (selling 10, clamped to the 5 actually held is NOT how this branch
    # goes — buy-side reversal is a sell, clamped by current holdings, so
    # position must have at least `reversal_shares` for the position to
    # stay open after). Use a held position of 20 so 10 reversed leaves 10.
    monkeypatch.setattr(fill_monitor, "_is_most_recent_trade", lambda ticker, trade_id: _async_value(True))
    monkeypatch.setattr(fill_monitor, "get_position", lambda ticker: _async_value(Position(cash=1_000_000.0, shares=20)))
    monkeypatch.setattr(fill_monitor, "_prior_stop_loss", lambda ticker, trade_id: _async_value((True, 65_000.0)))
    resolve_calls = []
    monkeypatch.setattr(
        fill_monitor, "resolve_stop_loss",
        lambda *a, **kw: resolve_calls.append((a, kw)) or None,
    )
    calls = []
    monkeypatch.setattr(
        fill_monitor, "reverse_pending_trade",
        lambda ticker, side, shares, price, reason, prior_stop_loss=None: _async_append(calls, prior_stop_loss),
    )

    corrected = await fill_monitor._auto_correct(_row(shares=10), reversal_shares=10, resolution="order never filled")

    assert corrected is True
    assert calls == [65_000.0]
    assert resolve_calls == []  # a prior row existed — no need to derive an ATR default


async def test_auto_correct_derives_atr_stop_when_no_prior_row_exists(monkeypatch):
    monkeypatch.setattr(fill_monitor, "_is_most_recent_trade", lambda ticker, trade_id: _async_value(True))
    monkeypatch.setattr(fill_monitor, "get_position", lambda ticker: _async_value(Position(cash=1_000_000.0, shares=20)))
    monkeypatch.setattr(fill_monitor, "_prior_stop_loss", lambda ticker, trade_id: _async_value((False, None)))
    monkeypatch.setattr(fill_monitor, "get_atr_pct", lambda ticker: _async_value(0.03))
    monkeypatch.setattr(fill_monitor, "resolve_stop_loss", lambda stop, price, shares_after, atr_pct: 66_000.0)
    calls = []
    monkeypatch.setattr(
        fill_monitor, "reverse_pending_trade",
        lambda ticker, side, shares, price, reason, prior_stop_loss=None: _async_append(calls, prior_stop_loss),
    )

    corrected = await fill_monitor._auto_correct(_row(shares=10), reversal_shares=10, resolution="order never filled")

    assert corrected is True
    assert calls == [66_000.0]


async def test_auto_correct_refuses_when_no_stop_loss_is_determinable(monkeypatch):
    # No prior row AND no ATR data (resolve_stop_loss degrades to None) — an
    # open position with literally no way to protect it. Must refuse rather
    # than restore it unprotected.
    monkeypatch.setattr(fill_monitor, "_is_most_recent_trade", lambda ticker, trade_id: _async_value(True))
    monkeypatch.setattr(fill_monitor, "get_position", lambda ticker: _async_value(Position(cash=1_000_000.0, shares=20)))
    monkeypatch.setattr(fill_monitor, "_prior_stop_loss", lambda ticker, trade_id: _async_value((False, None)))
    monkeypatch.setattr(fill_monitor, "get_atr_pct", lambda ticker: _async_value(None))
    monkeypatch.setattr(fill_monitor, "resolve_stop_loss", lambda stop, price, shares_after, atr_pct: None)
    called = []
    monkeypatch.setattr(fill_monitor, "reverse_pending_trade", lambda *a, **kw: _async_append(called, (a, kw)))

    corrected = await fill_monitor._auto_correct(_row(shares=10), reversal_shares=10, resolution="order never filled")

    assert corrected is False
    assert called == []


async def test_auto_correct_noop_when_reversal_shares_not_positive():
    corrected = await fill_monitor._auto_correct(_row(), reversal_shares=0, resolution="order never filled")
    assert corrected is False


# ---------------------------------------------------------------------------
# Stage 2 — check_pending_fills integration: feature flag, always-alert,
# and the circuit breaker.
# ---------------------------------------------------------------------------


async def test_check_pending_fills_never_auto_corrects_when_flag_is_off(monkeypatch):
    monkeypatch.setattr(fill_monitor.settings, "fill_monitor_auto_correct_enabled", False)
    monkeypatch.setattr(fill_monitor, "_pending_fill_checks", lambda: _async_list([_row()]))
    execution = _execution(filled_qty=0.0, cancelled=True)
    called = []
    monkeypatch.setattr(fill_monitor, "_auto_correct", lambda *a, **kw: _async_append(called, (a, kw)))
    monkeypatch.setattr(fill_monitor, "_mark_checked", lambda trade_id: _async_value(None))
    recorded = []
    monkeypatch.setattr(fill_monitor, "record_alert", lambda alert: _async_append(recorded, alert))

    await fill_monitor.check_pending_fills(domestic=[execution], overseas=[], give_up_after=GIVE_UP)

    assert called == []
    assert "자동보정" not in recorded[0].message


async def test_check_pending_fills_auto_corrects_zero_fill_when_enabled(monkeypatch):
    monkeypatch.setattr(fill_monitor.settings, "fill_monitor_auto_correct_enabled", True)
    monkeypatch.setattr(fill_monitor, "_corrections_today", lambda: _async_value(0))
    monkeypatch.setattr(fill_monitor, "_pending_fill_checks", lambda: _async_list([_row(shares=10)]))
    execution = _execution(filled_qty=0.0, cancelled=True)
    called = []

    async def fake_auto_correct(row, reversal_shares, resolution):
        called.append((row["id"], reversal_shares, resolution))
        return True

    monkeypatch.setattr(fill_monitor, "_auto_correct", fake_auto_correct)
    monkeypatch.setattr(fill_monitor, "_mark_checked", lambda trade_id: _async_value(None))
    recorded = []
    monkeypatch.setattr(fill_monitor, "record_alert", lambda alert: _async_append(recorded, alert))

    await fill_monitor.check_pending_fills(domestic=[execution], overseas=[], give_up_after=GIVE_UP)

    assert called == [(1, 10, "order never filled")]
    assert "자동보정" in recorded[0].message


async def test_check_pending_fills_auto_corrects_only_unfilled_remainder_on_partial(monkeypatch):
    old_row = _row(shares=10, created_at=NOW - timedelta(hours=1))
    monkeypatch.setattr(fill_monitor.settings, "fill_monitor_auto_correct_enabled", True)
    monkeypatch.setattr(fill_monitor, "_corrections_today", lambda: _async_value(0))
    monkeypatch.setattr(fill_monitor, "_pending_fill_checks", lambda: _async_list([old_row]))
    execution = _execution(filled_qty=4.0, order_qty=10.0)
    called = []

    async def fake_auto_correct(row, reversal_shares, resolution):
        called.append(reversal_shares)
        return True

    monkeypatch.setattr(fill_monitor, "_auto_correct", fake_auto_correct)
    monkeypatch.setattr(fill_monitor, "_mark_checked", lambda trade_id: _async_value(None))
    monkeypatch.setattr(fill_monitor, "record_alert", lambda alert: _async_value(None))

    await fill_monitor.check_pending_fills(domestic=[execution], overseas=[], give_up_after=GIVE_UP)

    assert called == [6]  # 10 recorded - 4 actually filled = 6 to reverse


async def test_circuit_breaker_disables_auto_correct_but_still_alerts(monkeypatch):
    monkeypatch.setattr(fill_monitor.settings, "fill_monitor_auto_correct_enabled", True)
    monkeypatch.setattr(fill_monitor.settings, "fill_monitor_max_corrections_per_day", 5)
    monkeypatch.setattr(fill_monitor, "_corrections_today", lambda: _async_value(5))  # at the cap
    monkeypatch.setattr(fill_monitor, "_pending_fill_checks", lambda: _async_list([_row()]))
    execution = _execution(filled_qty=0.0, cancelled=True)
    called = []
    monkeypatch.setattr(fill_monitor, "_auto_correct", lambda *a, **kw: _async_append(called, (a, kw)))
    monkeypatch.setattr(fill_monitor, "_mark_checked", lambda trade_id: _async_value(None))
    recorded = []
    monkeypatch.setattr(fill_monitor, "record_alert", lambda alert: _async_append(recorded, alert))

    await fill_monitor.check_pending_fills(domestic=[execution], overseas=[], give_up_after=GIVE_UP)

    assert called == []  # auto-correct never attempted
    kinds = [a.kind for a in recorded]
    assert "fill_monitor_circuit_breaker" in kinds  # but the breaker trip itself is still alerted
    assert "unconfirmed_fill" in kinds  # and the underlying discrepancy is still alerted too


async def _async_value(value):
    return value


# ---------------------------------------------------------------------------
# 2026-09-04 real-incident fix — a manual dashboard LIMIT order is now
# recorded to `trades` at submission (app.web.routers.orders), the same
# optimistic-record-then-mirror shape every other order path already uses.
# These two guard that the new row lands inside this module's EXISTING
# detect-only envelope, and that doing so does NOT smuggle in any
# auto-correction behavior along the way.
# ---------------------------------------------------------------------------


async def test_manual_limit_order_row_is_picked_up_by_pending_fill_checks(monkeypatch):
    """A trade row shaped exactly like the new manual-limit-order write
    (kis_order_no set, fill_checked_at NULL, mirror_disposition starting
    with 'submitted: manual dashboard limit order') must be selected by
    `_pending_fill_checks` — that SELECT is the entire mechanism that brings
    a manual limit order inside fill_monitor's alerting."""
    captured_query = {}

    class _FakeConn:
        async def fetch(self, query, *args):
            captured_query["query"] = query
            captured_query["args"] = args
            return [
                {
                    "id": 99, "ticker": "057680.KS", "side": "sell", "shares": 10,
                    "price": 5_000.0, "created_at": NOW - timedelta(minutes=5),
                    "kis_order_no": "0000012345", "fill_unresolved_alerted_at": None,
                }
            ]

    class _FakePool:
        def acquire(self):
            return nullcontext(_FakeConn())

    async def fake_get_pool():
        return _FakePool()

    monkeypatch.setattr(fill_monitor, "get_pool", fake_get_pool)

    rows = await fill_monitor._pending_fill_checks()

    assert len(rows) == 1
    assert rows[0]["ticker"] == "057680.KS"
    assert rows[0]["kis_order_no"] == "0000012345"
    # The SELECT itself filters on kis_order_no IS NOT NULL AND
    # fill_checked_at IS NULL — asserting the query text keeps this test
    # honest about WHY the fake row above is representative, rather than
    # just asserting the fake conn echoes back whatever it was given.
    assert "kis_order_no IS NOT NULL" in captured_query["query"]
    assert "fill_checked_at IS NULL" in captured_query["query"]


async def test_manual_limit_sell_cancelled_at_kis_alerts_without_correcting(monkeypatch):
    """Regression guard: with fill_monitor_auto_correct_enabled at its real
    default (False), a manual limit SELL that KIS reports as cancelled must
    be alerted (unconfirmed_fill) and marked checked — but reverse_pending_
    trade must NEVER be called. This is the proof that bringing the manual
    limit-order path inside fill_monitor's envelope stays detect-only, not
    a second auto-correction path smuggled in by this fix."""
    assert fill_monitor.settings.fill_monitor_auto_correct_enabled is False  # the real, untouched default

    row = _row(trade_id=99, ticker="057680.KS", side="sell", shares=10, order_no="0000012345")
    monkeypatch.setattr(fill_monitor, "_pending_fill_checks", lambda: _async_list([row]))
    execution = _execution(order_no="0000012345", filled_qty=0.0, order_qty=10.0, cancelled=True)

    marked = []
    monkeypatch.setattr(fill_monitor, "_mark_checked", lambda trade_id: _async_append(marked, trade_id))
    recorded = []
    monkeypatch.setattr(fill_monitor, "record_alert", lambda alert: _async_append(recorded, alert))
    reverse_calls = []
    monkeypatch.setattr(
        fill_monitor, "reverse_pending_trade", lambda *a, **kw: _async_append(reverse_calls, (a, kw))
    )

    await fill_monitor.check_pending_fills(domestic=[execution], overseas=[], give_up_after=GIVE_UP)

    assert len(recorded) == 1
    assert recorded[0].kind == "unconfirmed_fill"
    assert marked == [99]
    assert reverse_calls == []  # detect-only — no auto-correction smuggled in
