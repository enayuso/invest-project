"""app.audit.slippage_validation — broker + DB-backed ledger fetch are faked
(same pattern as tests/audit/test_reconciliation_report.py); exercises the
2-tier matching (order_no priority, (ticker,trade_date,side) 14-day window
fallback) and the same-day/delayed split in the aggregation + rendering."""

from __future__ import annotations

import statistics
from datetime import date, timedelta

import pytest

from app.audit import slippage_validation as sv
from app.domain.models.account import BrokerExecution
from app.execution.ledger import FORCED_EXIT_THRESHOLD_PRICE_NOTE_PREFIX


def _exec(
    order_date="20260801", order_no="1", code="005930", side="buy", filled_qty=10, avg_price=100.0,
) -> BrokerExecution:
    return BrokerExecution(
        order_date=order_date, order_no=order_no, code=code, name="", side=side,
        order_qty=filled_qty, filled_qty=filled_qty, remaining_qty=0, avg_price=avg_price,
        filled_amount=filled_qty * avg_price, fee_tax=0.0, cancelled=False,
    )


class FakeBroker:
    def __init__(self, domestic=None, overseas=None):
        self._domestic = domestic or []
        self._overseas = overseas or []

    async def fetch_daily_executions(self, start, end):
        return self._domestic

    async def fetch_overseas_executions(self, start, end):
        return self._overseas


async def _async_return(value):
    return value


def _ledger_row(
    ticker="005930.KS", trade_date=date(2026, 8, 1), side="Buy", shares=10, price=99.0, kis_order_no=None,
    adjustment_note=None,
):
    # NOTE: real `_fetch_ledger_trades` capitalizes `trades.side` ('buy' ->
    # 'Buy') before returning — this fake bypasses the DB layer, so it must
    # already hand back the same "Buy"/"Sell" convention the matching code
    # (and the KIS-side aggregation) expects.
    return {
        "ticker": ticker, "trade_date": trade_date, "side": side, "shares": shares, "price": price,
        "kis_order_no": kis_order_no, "adjustment_note": adjustment_note,
    }


def _patch(monkeypatch, broker, ledger_rows, get_broker_calls=None):
    def fake_get_broker(name, user_id=None, account_mode=None):
        if get_broker_calls is not None:
            get_broker_calls.append({"name": name, "user_id": user_id, "account_mode": account_mode})
        return _async_return(broker)

    monkeypatch.setattr(sv, "get_broker", fake_get_broker)

    async def fake_fetch(start, end, account_mode, user_id):
        return ledger_rows

    monkeypatch.setattr(sv, "_fetch_ledger_trades", fake_fetch)


async def test_raises_without_a_broker(monkeypatch):
    monkeypatch.setattr(sv, "get_broker", lambda name, user_id=None, account_mode=None: _async_return(None))
    with pytest.raises(RuntimeError, match="KIS broker not configured"):
        await sv.compute_slippage()


async def test_order_no_match_wins_over_window_match(monkeypatch):
    # Two KIS executions same ticker/side/day: one under the ledger's own
    # order_no (should be picked), one under a different order_no that would
    # also satisfy a window match (must NOT be picked / double-counted).
    exact = _exec(order_date="20260801", order_no="ORD-1", avg_price=101.0)
    decoy = _exec(order_date="20260801", order_no="ORD-2", avg_price=999.0)
    broker = FakeBroker(domestic=[exact, decoy])
    ledger = [_ledger_row(kis_order_no="ORD-1", price=100.0)]
    _patch(monkeypatch, broker, ledger)

    result = await sv.compute_slippage(start=date(2026, 8, 1), end=date(2026, 8, 1))

    assert len(result.matched) == 1
    row = result.matched[0]
    assert row.kis_price == 101.0
    assert row.fill_delay_days == 0
    # The decoy execution never got claimed by the ledger row, so it's KIS-only.
    assert ("005930.KS", date(2026, 8, 1), "Buy") in result.kis_only


async def test_delayed_match_excluded_from_main_stats(monkeypatch):
    # Ledger row recorded on 8/1 (e.g. stop-loss breakout-bar date), real
    # KIS fill landed on 8/5 — no kis_order_no (pre-migration / fallback path).
    late_fill = _exec(order_date="20260805", order_no="ORD-9", avg_price=90.0)
    broker = FakeBroker(domestic=[late_fill])
    ledger = [_ledger_row(trade_date=date(2026, 8, 1), price=100.0, kis_order_no=None)]
    _patch(monkeypatch, broker, ledger)

    result = await sv.compute_slippage(start=date(2026, 8, 1), end=date(2026, 8, 1))

    assert len(result.matched) == 1
    row = result.matched[0]
    assert row.fill_delay_days == 4

    rendered = sv.render_slippage(result)
    assert "지연체결" in rendered
    assert "당일 체결 거래가 없습니다" in rendered  # no same-day matches at all


async def test_beyond_max_delay_stays_ledger_only(monkeypatch):
    too_late = _exec(order_date="20260901", order_no="ORD-9", avg_price=90.0)  # 31 days later
    broker = FakeBroker(domestic=[too_late])
    ledger = [_ledger_row(trade_date=date(2026, 8, 1), price=100.0, kis_order_no=None)]
    _patch(monkeypatch, broker, ledger)

    result = await sv.compute_slippage(start=date(2026, 8, 1), end=date(2026, 8, 1))

    assert result.matched == []
    assert ("005930.KS", date(2026, 8, 1), "Buy") in result.ledger_only
    assert ("005930.KS", date(2026, 9, 1), "Buy") in result.kis_only


async def test_greedy_window_match_does_not_double_match_one_execution(monkeypatch):
    # One KIS execution, two fallback ledger rows (different dates) both
    # within its window — only the closer one should consume it; the other
    # must fall back to ledger_only rather than reusing the same fill.
    execution = _exec(order_date="20260803", order_no="ORD-1", avg_price=100.0, filled_qty=10)
    broker = FakeBroker(domestic=[execution])
    ledger = [
        _ledger_row(trade_date=date(2026, 8, 3), price=100.0, kis_order_no=None),  # exact date match
        _ledger_row(trade_date=date(2026, 8, 1), price=100.0, kis_order_no=None),  # 2 days off
    ]
    _patch(monkeypatch, broker, ledger)

    result = await sv.compute_slippage(start=date(2026, 8, 1), end=date(2026, 8, 3))

    # Greedy matching is chronological-order (same accepted heuristic as
    # app.execution.reconciliation), not globally optimal — so which of the
    # two candidate ledger rows wins isn't asserted here. What matters is
    # the single KIS execution is consumed exactly once: one match, one
    # ledger_only, never both ledger rows claiming the same fill.
    assert len(result.matched) == 1
    assert result.matched[0].kis_price == 100.0
    assert len(result.ledger_only) == 1
    assert result.kis_only == []


async def test_same_day_regression_case(monkeypatch):
    execution = _exec(order_date="20260801", order_no=None or "ORD-1", avg_price=105.0)
    broker = FakeBroker(domestic=[execution])
    ledger = [_ledger_row(trade_date=date(2026, 8, 1), price=100.0, kis_order_no="ORD-1")]
    _patch(monkeypatch, broker, ledger)

    result = await sv.compute_slippage(start=date(2026, 8, 1), end=date(2026, 8, 1))

    assert len(result.matched) == 1
    row = result.matched[0]
    assert row.fill_delay_days == 0
    assert row.slippage_pct == pytest.approx(0.05)

    rendered = sv.render_slippage(result)
    assert "대사 성공 1건" in rendered


def test_render_no_matches_at_all():
    result = sv.SlippageResult()
    rendered = sv.render_slippage(result)
    assert "당일 체결 거래가 없습니다" in rendered


# --- account_mode-aware caveat (2026-09-10: a real `--account-mode real` run
# printed the paper-account "이 수치는 하한선" caveat about its own real
# numbers — the line was hardcoded regardless of which mode was queried) ----


async def test_paper_report_keeps_the_lower_bound_caveat(monkeypatch):
    execution = _exec(order_date="20260801", order_no="ORD-1", avg_price=105.0)
    broker = FakeBroker(domestic=[execution])
    ledger = [_ledger_row(trade_date=date(2026, 8, 1), price=100.0, kis_order_no="ORD-1")]
    _patch(monkeypatch, broker, ledger)

    result = await sv.compute_slippage(start=date(2026, 8, 1), end=date(2026, 8, 1), account_mode="paper")

    assert result.account_mode == "paper"
    rendered = sv.render_slippage(result)
    assert "지금은 모의계좌 값이라 실계좌보다 관대할 수 있음" in rendered
    assert "실측치" not in rendered


async def test_real_report_does_not_claim_to_be_a_paper_lower_bound(monkeypatch):
    execution = _exec(order_date="20260801", order_no="ORD-1", avg_price=105.0)
    broker = FakeBroker(domestic=[execution])
    ledger = [_ledger_row(trade_date=date(2026, 8, 1), price=100.0, kis_order_no="ORD-1")]
    _patch(monkeypatch, broker, ledger)

    result = await sv.compute_slippage(start=date(2026, 8, 1), end=date(2026, 8, 1), account_mode="real")

    assert result.account_mode == "real"
    rendered = sv.render_slippage(result)
    assert "모의계좌 값이라 실계좌보다 관대할 수 있음" not in rendered
    assert "계좌 모드: real — 실측치." in rendered


# --- threshold-placeholder rows (047040.KS id=280 incident, 2026-08-27) ----


async def test_threshold_placeholder_row_is_excluded_from_the_mean(monkeypatch):
    # One clean row + one marked (threshold) row, both on the order_no path.
    # The marked row must be matched (see the next test) but must not pull
    # the reported mean toward its own fabricated slippage.
    clean = _exec(order_date="20260801", order_no="ORD-1", avg_price=101.0)  # +1% vs ledger 100.0
    threshold_exec = _exec(order_date="20260827", order_no="ORD-2", avg_price=21000.0)  # vs ledger's 19,729.24
    broker = FakeBroker(domestic=[clean, threshold_exec])
    ledger = [
        _ledger_row(trade_date=date(2026, 8, 1), price=100.0, kis_order_no="ORD-1"),
        _ledger_row(
            ticker="047040.KS", trade_date=date(2026, 8, 27), price=19729.24, kis_order_no="ORD-2",
            adjustment_note=f"{FORCED_EXIT_THRESHOLD_PRICE_NOTE_PREFIX} (no live tick) — recorded at the "
            "stop threshold 19,729.24, which is a threshold and NOT a traded price; exclude this row "
            "from slippage and return statistics",
        ),
    ]
    _patch(monkeypatch, broker, ledger)

    result = await sv.compute_slippage(start=date(2026, 8, 1), end=date(2026, 8, 27))

    assert len(result.matched) == 2
    flagged = [row for row in result.matched if row.price_is_threshold]
    assert len(flagged) == 1
    assert flagged[0].ticker == "047040.KS"

    rendered = sv.render_slippage(result)
    # Only the clean row's slippage (+1%) should drive the reported mean —
    # the threshold row's much larger gap must not be mixed in.
    assert "평균 슬리피지: +1.000%" in rendered


async def test_threshold_placeholder_row_still_consumes_its_kis_execution(monkeypatch):
    # A lone marked row with a kis_order_no must still consume its matching
    # KIS execution — the naive "just skip it" implementation would instead
    # leave the execution unclaimed and report it as kis_only (a phantom
    # "KIS fill with no ledger row" that doesn't actually exist).
    execution = _exec(order_date="20260827", order_no="ORD-2", avg_price=21000.0)
    broker = FakeBroker(domestic=[execution])
    ledger = [
        _ledger_row(
            ticker="047040.KS", trade_date=date(2026, 8, 27), price=19729.24, kis_order_no="ORD-2",
            adjustment_note=FORCED_EXIT_THRESHOLD_PRICE_NOTE_PREFIX,
        ),
    ]
    _patch(monkeypatch, broker, ledger)

    result = await sv.compute_slippage(start=date(2026, 8, 27), end=date(2026, 8, 27))

    assert result.kis_only == []
    assert result.ledger_only == []
    assert len(result.matched) == 1
    assert result.matched[0].price_is_threshold is True


async def test_threshold_placeholder_reported_in_its_own_section(monkeypatch):
    execution = _exec(order_date="20260827", order_no="ORD-2", avg_price=21000.0)
    broker = FakeBroker(domestic=[execution])
    ledger = [
        _ledger_row(
            ticker="047040.KS", trade_date=date(2026, 8, 27), price=19729.24, kis_order_no="ORD-2",
            adjustment_note=FORCED_EXIT_THRESHOLD_PRICE_NOTE_PREFIX,
        ),
    ]
    _patch(monkeypatch, broker, ledger)

    result = await sv.compute_slippage(start=date(2026, 8, 27), end=date(2026, 8, 27))
    rendered = sv.render_slippage(result)

    # When the only matched row is a threshold placeholder, `same_day` is
    # empty and the "no same-day rows" branch must not divide by zero.
    assert "임계값 기록행" in rendered
    assert "047040.KS" in rendered


async def test_threshold_taint_propagates_through_the_fallback_bucket(monkeypatch):
    # Two ledger rows bucketed into one (ticker, date, side) fallback key
    # (no kis_order_no on either) — only one of the two is marked. The
    # merged bucket must still come back tainted: an average that mixes a
    # threshold placeholder with a real price is neither.
    execution = _exec(order_date="20260827", order_no="ORD-9", code="047040", avg_price=21000.0, filled_qty=20)
    broker = FakeBroker(domestic=[execution])
    ledger = [
        _ledger_row(
            ticker="047040.KS", trade_date=date(2026, 8, 27), shares=10, price=19729.24, kis_order_no=None,
            adjustment_note=FORCED_EXIT_THRESHOLD_PRICE_NOTE_PREFIX,
        ),
        _ledger_row(
            ticker="047040.KS", trade_date=date(2026, 8, 27), shares=10, price=20000.0, kis_order_no=None,
            adjustment_note=None,
        ),
    ]
    _patch(monkeypatch, broker, ledger)

    result = await sv.compute_slippage(start=date(2026, 8, 27), end=date(2026, 8, 27))

    assert len(result.matched) == 1
    assert result.matched[0].price_is_threshold is True


# --- order-number zero-padding + priority-miss fallthrough (2026-08-27) ----


async def test_overseas_zero_padding_regression(monkeypatch):
    # trades.kis_order_no always stores the zero-padded acceptance-response
    # ODNO ("0000043249"); overseas execution history returns it unpadded
    # ("43249"). Before normalization this join silently missed every
    # overseas trade.
    execution = _exec(order_date="20260801", order_no="43249", code="AAPL", avg_price=101.0)
    broker = FakeBroker(overseas=[execution])
    ledger = [_ledger_row(ticker="AAPL", trade_date=date(2026, 8, 1), price=100.0, kis_order_no="0000043249")]
    _patch(monkeypatch, broker, ledger)

    result = await sv.compute_slippage(start=date(2026, 8, 1), end=date(2026, 8, 1))

    assert len(result.matched) == 1
    assert result.matched[0].kis_price == 101.0
    assert result.ledger_only == []
    assert result.kis_only == []


async def test_priority_path_miss_falls_through_to_window_match(monkeypatch):
    # Ledger row carries a kis_order_no with no matching KIS execution by
    # exact order_no, but there IS a same-day KIS execution for the same
    # (ticker, trade_date, side) that the window-fallback matcher can find.
    # Before this fix, a priority-path miss went straight to ledger_only.
    execution = _exec(order_date="20260801", order_no="OTHER-ORDER", avg_price=101.0)
    broker = FakeBroker(domestic=[execution])
    ledger = [_ledger_row(trade_date=date(2026, 8, 1), price=100.0, kis_order_no="NO-MATCH")]
    _patch(monkeypatch, broker, ledger)

    result = await sv.compute_slippage(start=date(2026, 8, 1), end=date(2026, 8, 1))

    assert result.ledger_only == []
    assert result.kis_only == []
    assert len(result.matched) == 1
    assert result.matched[0].kis_price == 101.0
    assert result.matched[0].fill_delay_days == 0


async def test_account_mode_passthrough_to_broker(monkeypatch):
    # compute_slippage must resolve the broker with the SAME account_mode
    # (and resolved user_id) it uses for the ledger query — otherwise the
    # report can silently compare one account's ledger against another
    # account's KIS executions.
    broker = FakeBroker()
    calls = []
    _patch(monkeypatch, broker, [], get_broker_calls=calls)

    await sv.compute_slippage(
        start=date(2026, 8, 1), end=date(2026, 8, 1), account_mode="real", user_id="u1",
    )

    assert len(calls) == 1
    assert calls[0]["account_mode"] == "real"
    assert calls[0]["user_id"] == "u1"


# --- Fix A: true median (not values[n//2]) ----------------------------------


async def test_median_is_true_median_not_upper_middle(monkeypatch):
    # Four same-day matched rows -> even-length list. The old `values[n//2]`
    # formula picks the upper-of-two-middle value; a true median averages
    # the two middle values. Slippage pcts here: -0.02, -0.01, +0.01, +0.02
    # (sorted) -> naive formula gives values[2] = +0.01, true median gives 0.0.
    execs = [
        _exec(order_date="20260801", order_no="ORD-1", code="000001", avg_price=98.0),   # -2%
        _exec(order_date="20260801", order_no="ORD-2", code="000002", avg_price=99.0),   # -1%
        _exec(order_date="20260801", order_no="ORD-3", code="000003", avg_price=101.0),  # +1%
        _exec(order_date="20260801", order_no="ORD-4", code="000004", avg_price=102.0),  # +2%
    ]
    broker = FakeBroker(domestic=execs)
    ledger = [
        _ledger_row(ticker="000001.KS", trade_date=date(2026, 8, 1), price=100.0, kis_order_no="ORD-1"),
        _ledger_row(ticker="000002.KS", trade_date=date(2026, 8, 1), price=100.0, kis_order_no="ORD-2"),
        _ledger_row(ticker="000003.KS", trade_date=date(2026, 8, 1), price=100.0, kis_order_no="ORD-3"),
        _ledger_row(ticker="000004.KS", trade_date=date(2026, 8, 1), price=100.0, kis_order_no="ORD-4"),
    ]
    _patch(monkeypatch, broker, ledger)

    result = await sv.compute_slippage(start=date(2026, 8, 1), end=date(2026, 8, 1))
    assert len(result.matched) == 4

    values = sorted(row.slippage_pct for row in result.matched)
    true_median = statistics.median(values)
    naive_median = values[len(values) // 2]
    assert true_median != naive_median  # sanity: the two formulas disagree here
    assert true_median == pytest.approx(0.0)

    rendered = sv.render_slippage(result)
    assert f"전체 중앙값: {true_median:+.3%}" in rendered
    assert f"전체 중앙값: {naive_median:+.3%}" not in rendered


async def test_per_market_summary_median_is_true_median(monkeypatch):
    # Same even-length-set trick, but all domestic so it exercises the
    # `_summary` closure's own median calculation.
    execs = [
        _exec(order_date="20260801", order_no="ORD-1", code="000001", avg_price=98.0),   # -2%
        _exec(order_date="20260801", order_no="ORD-2", code="000002", avg_price=99.0),   # -1%
        _exec(order_date="20260801", order_no="ORD-3", code="000003", avg_price=101.0),  # +1%
        _exec(order_date="20260801", order_no="ORD-4", code="000004", avg_price=102.0),  # +2%
    ]
    broker = FakeBroker(domestic=execs)
    ledger = [
        _ledger_row(ticker="000001.KS", trade_date=date(2026, 8, 1), price=100.0, kis_order_no="ORD-1"),
        _ledger_row(ticker="000002.KS", trade_date=date(2026, 8, 1), price=100.0, kis_order_no="ORD-2"),
        _ledger_row(ticker="000003.KS", trade_date=date(2026, 8, 1), price=100.0, kis_order_no="ORD-3"),
        _ledger_row(ticker="000004.KS", trade_date=date(2026, 8, 1), price=100.0, kis_order_no="ORD-4"),
    ]
    _patch(monkeypatch, broker, ledger)

    result = await sv.compute_slippage(start=date(2026, 8, 1), end=date(2026, 8, 1))
    rendered = sv.render_slippage(result)

    values = [row.slippage_pct for row in result.matched]
    true_median = statistics.median(values)
    assert f"중앙값 {true_median:+.3%}" in rendered


# --- Fix B: silent start-date clamping --------------------------------------


async def test_start_date_older_than_lookback_is_clamped_and_reported(monkeypatch):
    broker = FakeBroker()
    _patch(monkeypatch, broker, [])

    result = await sv.compute_slippage(start=date(2020, 1, 1), end=date(2020, 1, 1))

    assert result.start_clamped is True
    rendered = sv.render_slippage(result)
    assert "KIS 조회 한도" in rendered
    assert "90" in rendered


async def test_start_date_within_lookback_is_not_clamped(monkeypatch):
    broker = FakeBroker()
    _patch(monkeypatch, broker, [])

    recent_start = date.today() - timedelta(days=5)
    result = await sv.compute_slippage(start=recent_start, end=date.today())

    assert result.start_clamped is False
    rendered = sv.render_slippage(result)
    assert "KIS 조회 한도" not in rendered


# --- Fix C: malformed order_date must not crash the whole report -----------


async def test_malformed_order_date_is_skipped_not_fatal(monkeypatch):
    good = _exec(order_date="20260801", order_no="ORD-1", code="005930", avg_price=101.0)
    bad = _exec(order_date="", order_no="ORD-2", code="000660", avg_price=50.0)
    broker = FakeBroker(domestic=[good, bad])
    ledger = [
        _ledger_row(ticker="005930.KS", trade_date=date(2026, 8, 1), price=100.0, kis_order_no="ORD-1"),
    ]
    _patch(monkeypatch, broker, ledger)

    result = await sv.compute_slippage(start=date(2026, 8, 1), end=date(2026, 8, 1))

    assert len(result.matched) == 1
    assert result.matched[0].ticker == "005930.KS"
    assert result.kis_only == []
    assert result.ledger_only == []


# --- Fix D: reversal/mirror_disposition exclusions in the SQL WHERE --------


def test_fetch_ledger_trades_sql_excludes_reversal_and_unmirrored_rows():
    import inspect

    src = inspect.getsource(sv._fetch_ledger_trades)
    assert "reason NOT LIKE 'reversal:%'" in src
    assert "mirror_disposition IS NULL OR mirror_disposition LIKE 'submitted:%'" in src


# --- Fix E: price<=0 handling consistent between priority and fallback -----


async def test_priority_path_zero_price_recorded_as_unusable_not_lost(monkeypatch):
    execution = _exec(order_date="20260801", order_no="ORD-1", avg_price=101.0)
    broker = FakeBroker(domestic=[execution])
    ledger = [_ledger_row(trade_date=date(2026, 8, 1), price=0.0, kis_order_no="ORD-1")]
    _patch(monkeypatch, broker, ledger)

    result = await sv.compute_slippage(start=date(2026, 8, 1), end=date(2026, 8, 1))

    assert result.matched == []
    assert result.ledger_only == []
    assert result.kis_only == []
    assert ("005930.KS", date(2026, 8, 1), "Buy") in result.unusable_price


async def test_fallback_path_negative_price_recorded_as_unusable_not_lost(monkeypatch):
    execution = _exec(order_date="20260803", order_no="ORD-1", avg_price=100.0)
    broker = FakeBroker(domestic=[execution])
    ledger = [_ledger_row(trade_date=date(2026, 8, 3), price=-5.0, kis_order_no=None)]
    _patch(monkeypatch, broker, ledger)

    result = await sv.compute_slippage(start=date(2026, 8, 1), end=date(2026, 8, 3))

    assert result.matched == []
    assert result.ledger_only == []
    assert result.kis_only == []
    assert ("005930.KS", date(2026, 8, 3), "Buy") in result.unusable_price
