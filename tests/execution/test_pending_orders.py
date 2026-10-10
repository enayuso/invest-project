"""app.execution.pending_orders — asyncpg pool/connection faked (in-memory
table), no real DB/network."""

from __future__ import annotations

import asyncio
from datetime import date, datetime, timedelta, timezone

import pytest

from app.config import settings
from app.execution import pending_orders
from app.user_context import user_scope


class FakePendingOrdersConn:
    """In-memory stand-in for the pending_orders table."""

    def __init__(self):
        self.rows: dict[int, dict] = {}
        self._next_id = 1

    async def execute(self, query, *args):
        if "CREATE TABLE" in query or "ALTER TABLE" in query:
            return
        if "UPDATE pending_orders" in query:
            order_id, status, resolution, user_id = args
            row = self.rows.get(order_id)
            # The user_id predicate is enforced here on purpose: it is what
            # stops one user resolving another's pending order, and a fake
            # that ignored it would let that leak pass (Phase 4, plan §2.2).
            if row is not None and row["status"] == "pending" and row["user_id"] == user_id:
                row["status"] = status
                row["resolution"] = resolution
                row["resolved_at"] = datetime(2026, 7, 16, 12, 0, 0, tzinfo=timezone.utc)
            return
        raise AssertionError(f"unexpected query: {query}")

    async def fetchrow(self, query, *args):
        if "INSERT INTO pending_orders" in query:
            ticker, trade_date, side, shares, price, reason, account_mode, user_id, trade_id = args
            order_id = self._next_id
            self._next_id += 1
            self.rows[order_id] = {
                "id": order_id, "ticker": ticker, "trade_date": trade_date, "side": side,
                "shares": shares, "price": price, "reason": reason, "status": "pending",
                "resolution": None, "created_at": datetime.now(timezone.utc), "resolved_at": None,
                "account_mode": account_mode, "user_id": user_id, "trade_id": trade_id,
            }
            return {"id": order_id}
        if "SELECT * FROM pending_orders WHERE id" in query:
            order_id, account_mode, user_id = args
            row = self.rows.get(order_id)
            matches = row is not None and row["account_mode"] == account_mode and row["user_id"] == user_id
            return row if matches else None
        raise AssertionError(f"unexpected query: {query}")

    async def fetchval(self, query, *args):
        if "SELECT count(*) FROM pending_orders" in query:
            account_mode, user_id = args
            return sum(
                1 for r in self.rows.values()
                if r["status"] == "pending" and r["account_mode"] == account_mode and r["user_id"] == user_id
            )
        raise AssertionError(f"unexpected query: {query}")

    async def fetch(self, query, *args):
        if "WHERE status = 'pending' AND account_mode" in query:
            account_mode, user_id = args
            return [
                r for r in self.rows.values()
                if r["status"] == "pending" and r["account_mode"] == account_mode and r["user_id"] == user_id
            ]
        if "WHERE status != 'pending' AND account_mode" in query:
            account_mode, user_id, limit = args
            rows = [
                r for r in self.rows.values()
                if r["status"] != "pending" and r["account_mode"] == account_mode and r["user_id"] == user_id
            ]
            return rows[:limit]
        raise AssertionError(f"unexpected query: {query}")


class _AcquireCM:
    def __init__(self, conn):
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, *exc):
        return False


class FakePool:
    def __init__(self, conn):
        self._conn = conn

    def acquire(self):
        return _AcquireCM(self._conn)


async def _async_return(value):
    return value


@pytest.fixture
def fake_conn(monkeypatch):
    conn = FakePendingOrdersConn()
    monkeypatch.setattr(pending_orders, "_schema_ready", False)
    monkeypatch.setattr(pending_orders, "get_pool", lambda: _async_return(FakePool(conn)))
    # Fresh `_resolution_lock` per test — pytest-asyncio gives each test its
    # own event loop, but `asyncio.Lock` only binds to a loop on its SLOW
    # (actually-contended) acquire path, not on every acquire. Once one test
    # genuinely contends the lock (e.g. two gathered coroutines racing it),
    # the shared module-level lock binds to THAT test's loop; the next test
    # to genuinely contend it then hits "Lock ... is bound to a different
    # event loop" against the previous (already-closed) loop. Production is
    # unaffected (the real app runs on one persistent loop for its whole
    # lifetime) — this is purely test-isolation hygiene, same reasoning as
    # resetting `_schema_ready` above.
    monkeypatch.setattr(pending_orders, "_resolution_lock", asyncio.Lock())
    return conn


async def test_create_and_list_pending_order(fake_conn):
    order_id = await pending_orders.create_pending_order(
        "005930.KS", date(2026, 7, 16), "buy", 5, 150_000.0, reason="NEEDS_APPROVAL: over threshold"
    )
    assert order_id == 1
    listed = await pending_orders.list_pending_orders()
    assert len(listed) == 1
    assert listed[0].ticker == "005930.KS" and listed[0].status == "pending"


async def test_get_pending_order_returns_none_when_absent(fake_conn):
    assert await pending_orders.get_pending_order(999) is None


async def test_count_pending_reflects_only_pending_rows_for_the_current_scope(fake_conn, monkeypatch):
    assert await pending_orders.count_pending() == 0
    id1 = await pending_orders.create_pending_order(
        "005930.KS", date(2026, 7, 16), "buy", 5, 150_000.0, reason="test"
    )
    await pending_orders.create_pending_order(
        "AAPL", date(2026, 7, 16), "buy", 1, 200.0, reason="test"
    )
    assert await pending_orders.count_pending() == 2

    await pending_orders.reject_pending_order(id1)
    assert await pending_orders.count_pending() == 1  # resolved rows don't count

    monkeypatch.setattr(settings, "kis_account_mode", "real")
    assert await pending_orders.count_pending() == 0  # other account_mode invisible


async def test_reject_pending_order_marks_resolved(fake_conn, monkeypatch):
    captured = {}

    async def fake_reverse(ticker, side, shares, price, reason):
        captured.update(ticker=ticker, side=side, shares=shares, price=price, reason=reason)

    monkeypatch.setattr("app.execution.ledger.reverse_pending_trade", fake_reverse)

    order_id = await pending_orders.create_pending_order(
        "AAPL", date(2026, 7, 16), "buy", 1, 200.0, reason="test", trade_id=999,
    )
    await pending_orders.reject_pending_order(order_id, "changed my mind")
    order = await pending_orders.get_pending_order(order_id)
    assert order.status == "rejected"
    assert order.resolution == "changed my mind"
    # No longer listed as pending.
    assert await pending_orders.list_pending_orders() == []
    # The ledger's speculative entry for this order must be undone.
    assert captured == {
        "ticker": "AAPL", "side": "buy", "shares": 1, "price": 200.0, "reason": "changed my mind",
    }


async def test_reject_pending_order_skips_ledger_reversal_when_unlinked(fake_conn, monkeypatch):
    """Regression (2026-08-06 incident, 005930.KS): a pending order created
    with no `trade_id` (e.g. scripts/seed_test_pending_order.py, which never
    calls `record_trade`) has no speculative ledger entry to undo. Rejecting
    it must NOT call reverse_pending_trade — doing so used to clamp against
    whatever the ledger's CURRENT position happened to be and silently sell
    off real, unrelated shares. It should instead raise an alert flagging
    the anomaly."""
    reverse_calls = []

    async def should_not_be_called(*args, **kwargs):
        reverse_calls.append((args, kwargs))

    monkeypatch.setattr("app.execution.ledger.reverse_pending_trade", should_not_be_called)

    alerts = []

    async def fake_record_alert(alert):
        alerts.append(alert)

    monkeypatch.setattr(pending_orders, "record_alert", fake_record_alert)

    # No trade_id given — exactly what the dev seed script does.
    order_id = await pending_orders.create_pending_order(
        "005930.KS", date(2026, 8, 6), "buy", 5, 150_000.0, reason="[TEST DATA] seeded"
    )
    await pending_orders.reject_pending_order(order_id, "rejected by user")

    assert reverse_calls == []  # nothing reversed — no ledger entry existed for this order
    order = await pending_orders.get_pending_order(order_id)
    assert order.status == "rejected"
    assert len(alerts) == 1
    assert alerts[0].ticker == "005930.KS"
    assert alerts[0].kind == "pending_order_unlinked"


async def test_reject_pending_order_still_reverses_when_linked(fake_conn, monkeypatch):
    """The normal SEMI_AUTO path — rebalance_to_target always passes the
    trade_id of the speculative row it just wrote, so rejecting must still
    reverse it exactly as before."""
    captured = {}

    async def fake_reverse(ticker, side, shares, price, reason):
        captured.update(ticker=ticker, side=side, shares=shares, price=price, reason=reason)

    monkeypatch.setattr("app.execution.ledger.reverse_pending_trade", fake_reverse)

    order_id = await pending_orders.create_pending_order(
        "005930.KS", date(2026, 8, 6), "buy", 5, 150_000.0, reason="Buy", trade_id=42,
    )
    await pending_orders.reject_pending_order(order_id, "rejected by user")

    assert captured == {
        "ticker": "005930.KS", "side": "buy", "shares": 5, "price": 150_000.0, "reason": "rejected by user",
    }


async def test_sweep_expires_unlinked_pending_order_without_reversing(fake_conn, monkeypatch):
    reverse_calls = []

    async def should_not_be_called(*args, **kwargs):
        reverse_calls.append((args, kwargs))

    monkeypatch.setattr("app.execution.ledger.reverse_pending_trade", should_not_be_called)
    monkeypatch.setattr(pending_orders, "record_alert", lambda alert: _async_return(None))

    order_id = await pending_orders.create_pending_order(
        "AAPL", date(2026, 7, 16), "buy", 1, 200.0, reason="test"  # no trade_id
    )
    fake_conn.rows[order_id]["created_at"] = (
        datetime.now(timezone.utc) - timedelta(hours=settings.pending_order_ttl_hours + 1)
    )

    listed = await pending_orders.list_pending_orders()
    assert listed == []  # no longer pending
    assert reverse_calls == []

    order = await pending_orders.get_pending_order(order_id)
    assert order.status == "expired"


async def test_approve_failed_skips_ledger_reversal_when_unlinked(fake_conn, monkeypatch):
    reverse_calls = []

    async def should_not_be_called(*args, **kwargs):
        reverse_calls.append((args, kwargs))

    monkeypatch.setattr("app.execution.ledger.reverse_pending_trade", should_not_be_called)
    monkeypatch.setattr(pending_orders, "record_alert", lambda alert: _async_return(None))

    order_id = await pending_orders.create_pending_order(
        "005930.KS", date(2026, 8, 6), "buy", 5, 150_000.0, reason="test"  # no trade_id
    )

    async def fake_mirror(*args, **kwargs):
        return "BLOCKED: insufficient cash"

    monkeypatch.setattr("app.execution.kis_bridge.mirror_trade_to_kis", fake_mirror)

    disposition = await pending_orders.approve_pending_order(order_id)
    assert disposition == "BLOCKED: insufficient cash"
    assert reverse_calls == []
    order = await pending_orders.get_pending_order(order_id)
    assert order.status == "approve_failed"


async def test_reject_pending_order_raises_when_already_resolved(fake_conn):
    order_id = await pending_orders.create_pending_order(
        "AAPL", date(2026, 7, 16), "buy", 1, 200.0, reason="test"
    )
    await pending_orders.reject_pending_order(order_id)
    with pytest.raises(ValueError, match="not found or already resolved"):
        await pending_orders.reject_pending_order(order_id)


async def test_reject_pending_order_raises_when_never_existed(fake_conn):
    with pytest.raises(ValueError, match="not found or already resolved"):
        await pending_orders.reject_pending_order(12345)


async def test_approve_pending_order_calls_mirror_with_approved_true(fake_conn, monkeypatch):
    order_id = await pending_orders.create_pending_order(
        "005930.KS", date(2026, 7, 16), "buy", 5, 150_000.0, reason="NEEDS_APPROVAL: over threshold"
    )
    captured = {}

    async def fake_mirror(ticker, trade_date, side, shares, price, *, broker=None, approved=False, trade_id=None):
        captured.update(
            ticker=ticker, side=side, shares=shares, price=price, broker=broker, approved=approved,
            trade_id=trade_id,
        )
        return "submitted: mock order"

    monkeypatch.setattr("app.execution.kis_bridge.mirror_trade_to_kis", fake_mirror)

    disposition = await pending_orders.approve_pending_order(order_id, broker="fake-broker")
    assert disposition == "submitted: mock order"
    assert captured == {
        "ticker": "005930.KS", "side": "buy", "shares": 5, "price": 150_000.0,
        "broker": "fake-broker", "approved": True,
        # This order was created without a trade_id (line ~301 above), so
        # approve_pending_order forwards order.trade_id = None through to
        # mirror_trade_to_kis — pins that forwarding, not just the
        # pre-existing fields.
        "trade_id": None,
    }
    order = await pending_orders.get_pending_order(order_id)
    assert order.status == "approved"
    assert order.resolution == "submitted: mock order"


async def test_approve_pending_order_marks_approve_failed_when_still_blocked(fake_conn, monkeypatch):
    # Re-running the gate chain at approval time can still block (e.g. cash
    # dried up in the meantime) — that's a legitimate outcome, not a bug.
    order_id = await pending_orders.create_pending_order(
        "005930.KS", date(2026, 7, 16), "buy", 5, 150_000.0, reason="NEEDS_APPROVAL: over threshold"
    )

    async def fake_mirror(*args, **kwargs):
        return "BLOCKED: insufficient cash"

    monkeypatch.setattr("app.execution.kis_bridge.mirror_trade_to_kis", fake_mirror)

    disposition = await pending_orders.approve_pending_order(order_id)
    assert disposition == "BLOCKED: insufficient cash"
    order = await pending_orders.get_pending_order(order_id)
    assert order.status == "approve_failed"


async def test_approve_pending_order_reverses_the_ledger_entry_when_blocked(fake_conn, monkeypatch):
    # Real incident: a VI-halt false positive blocked an approval attempt,
    # and the phantom "bought" entry rebalance_to_target wrote when the
    # order was first proposed stayed in the ledger forever — a later
    # cycle then "sold" shares that were never actually bought. Must be
    # reversed here, exactly like reject/expire.
    captured = {}

    async def fake_reverse(ticker, side, shares, price, reason):
        captured.update(ticker=ticker, side=side, shares=shares, price=price, reason=reason)

    monkeypatch.setattr("app.execution.ledger.reverse_pending_trade", fake_reverse)

    order_id = await pending_orders.create_pending_order(
        "065420.KS", date(2026, 7, 23), "buy", 31, 214.0, reason="NEEDS_APPROVAL: over threshold", trade_id=999,
    )

    async def fake_mirror(*args, **kwargs):
        return "BLOCKED: 065420.KS is currently under a VI (변동성완화장치) halt"

    monkeypatch.setattr("app.execution.kis_bridge.mirror_trade_to_kis", fake_mirror)

    disposition = await pending_orders.approve_pending_order(order_id)

    assert disposition.startswith("BLOCKED")
    order = await pending_orders.get_pending_order(order_id)
    assert order.status == "approve_failed"
    assert captured == {
        "ticker": "065420.KS", "side": "buy", "shares": 31, "price": 214.0, "reason": disposition,
    }


async def test_approve_pending_order_raises_when_already_resolved(fake_conn, monkeypatch):
    order_id = await pending_orders.create_pending_order(
        "AAPL", date(2026, 7, 16), "buy", 1, 200.0, reason="test"
    )
    await pending_orders.reject_pending_order(order_id)
    with pytest.raises(ValueError, match="not found or already resolved"):
        await pending_orders.approve_pending_order(order_id)


async def test_concurrent_approve_and_reject_only_resolves_once(fake_conn, monkeypatch):
    """A double-click (or an approve/reject race) must not let both sides'
    real-world action fire — _resolution_lock serializes the whole
    check-act-resolve sequence, so the loser sees status != "pending"
    before doing anything, not after."""
    order_id = await pending_orders.create_pending_order(
        "005930.KS", date(2026, 7, 16), "buy", 5, 150_000.0, reason="test", trade_id=999,
    )
    calls = {"mirror": 0, "reverse": 0}

    async def fake_mirror(*args, **kwargs):
        calls["mirror"] += 1
        await asyncio.sleep(0)  # yield, so a racing task without the lock could interleave here
        return "submitted: mock"

    async def fake_reverse(*args, **kwargs):
        calls["reverse"] += 1
        await asyncio.sleep(0)

    monkeypatch.setattr("app.execution.kis_bridge.mirror_trade_to_kis", fake_mirror)
    monkeypatch.setattr("app.execution.ledger.reverse_pending_trade", fake_reverse)

    results = await asyncio.gather(
        pending_orders.approve_pending_order(order_id),
        pending_orders.reject_pending_order(order_id),
        return_exceptions=True,
    )

    assert calls["mirror"] + calls["reverse"] == 1, f"both sides fired: {calls}"
    errors = [r for r in results if isinstance(r, ValueError)]
    assert len(errors) == 1


# ---------------------------------------------------------------------------
# _sweep_expired (TTL)
# ---------------------------------------------------------------------------


async def test_list_pending_orders_sweep_does_not_race_a_concurrent_approval(fake_conn, monkeypatch):
    """Regression (2026-08-06): `list_pending_orders`'s own `_sweep_expired()`
    call used to run OUTSIDE `_resolution_lock` — the only sweep call site
    that wasn't already covered by approve/reject_pending_order acquiring
    the lock first. A dashboard poll landing here at the same instant a
    human approves an order that's *just* past its TTL could interleave:
    the sweep reverses the ledger and marks the row 'expired' while approve
    (having already read it as 'pending' moments earlier) goes on to submit
    a REAL order to KIS, then silently no-ops its own
    `UPDATE ... WHERE status = 'pending'` — a live KIS order with a ledger
    that says it was reversed. Locking list_pending_orders's sweep the same
    way approve/reject's already-locked get_pending_order is means only one
    of {mirror, reverse} can ever fire for the same order."""
    order_id = await pending_orders.create_pending_order(
        "005930.KS", date(2026, 7, 16), "buy", 5, 150_000.0, reason="test", trade_id=999,
    )
    fake_conn.rows[order_id]["created_at"] = (
        datetime.now(timezone.utc) - timedelta(hours=settings.pending_order_ttl_hours + 1)
    )
    calls = {"mirror": 0, "reverse": 0}

    async def fake_mirror(*args, **kwargs):
        calls["mirror"] += 1
        await asyncio.sleep(0)  # yield, so a racing task without the lock could interleave here
        return "submitted: mock"

    async def fake_reverse(*args, **kwargs):
        calls["reverse"] += 1
        await asyncio.sleep(0)

    monkeypatch.setattr("app.execution.kis_bridge.mirror_trade_to_kis", fake_mirror)
    monkeypatch.setattr("app.execution.ledger.reverse_pending_trade", fake_reverse)

    results = await asyncio.gather(
        pending_orders.approve_pending_order(order_id),
        pending_orders.list_pending_orders(),
        return_exceptions=True,
    )

    assert calls["mirror"] + calls["reverse"] == 1, f"both sides fired: {calls}"
    # Whichever side lost must see the effect of whichever side won, not a
    # half-applied state: approve raises ValueError if the sweep beat it to
    # the punch; list_pending_orders simply returns [] either way (approved
    # or expired, neither is 'pending' anymore).
    approve_result, listed = results
    if isinstance(approve_result, ValueError):
        assert calls["reverse"] == 1
    else:
        assert calls["mirror"] == 1
    assert listed == []


async def test_sweep_expires_stale_pending_order(fake_conn, monkeypatch):
    captured = {}

    async def fake_reverse(ticker, side, shares, price, reason):
        captured.update(ticker=ticker, side=side, shares=shares, price=price, reason=reason)

    monkeypatch.setattr("app.execution.ledger.reverse_pending_trade", fake_reverse)

    order_id = await pending_orders.create_pending_order(
        "AAPL", date(2026, 7, 16), "buy", 1, 200.0, reason="test", trade_id=999,
    )
    fake_conn.rows[order_id]["created_at"] = (
        datetime.now(timezone.utc) - timedelta(hours=settings.pending_order_ttl_hours + 1)
    )

    listed = await pending_orders.list_pending_orders()
    assert listed == []  # no longer pending

    order = await pending_orders.get_pending_order(order_id)
    assert order.status == "expired"
    assert order.resolution.startswith("expired: no decision within")
    assert captured == {
        "ticker": "AAPL", "side": "buy", "shares": 1, "price": 200.0, "reason": order.resolution,
    }


async def test_sweep_leaves_fresh_pending_order_untouched(fake_conn):
    order_id = await pending_orders.create_pending_order(
        "AAPL", date(2026, 7, 16), "buy", 1, 200.0, reason="test"
    )
    listed = await pending_orders.list_pending_orders()
    assert len(listed) == 1
    assert listed[0].id == order_id
    assert listed[0].status == "pending"


async def test_approve_raises_when_expired_between_listing_and_approval(fake_conn, monkeypatch):
    async def fake_reverse(*args, **kwargs):
        pass

    monkeypatch.setattr("app.execution.ledger.reverse_pending_trade", fake_reverse)

    order_id = await pending_orders.create_pending_order(
        "AAPL", date(2026, 7, 16), "buy", 1, 200.0, reason="test"
    )
    fake_conn.rows[order_id]["created_at"] = (
        datetime.now(timezone.utc) - timedelta(hours=settings.pending_order_ttl_hours + 1)
    )

    # approve_pending_order calls get_pending_order first, which sweeps —
    # this order is already 'expired' by the time the status check runs.
    with pytest.raises(ValueError, match="not found or already resolved"):
        await pending_orders.approve_pending_order(order_id)


# ---------------------------------------------------------------------------
# account_mode scoping (2026-08-04) — see project memory:
# user-account-mode-db-separation-design. The concrete risk this closes: a
# SEMI_AUTO proposal left unresolved while paper-testing must never be
# approvable (submitting a REAL order) after switching kis_account_mode to
# 'real' and restarting.
# ---------------------------------------------------------------------------


async def test_create_pending_order_stamps_current_account_mode(fake_conn, monkeypatch):
    monkeypatch.setattr(settings, "kis_account_mode", "real")
    order_id = await pending_orders.create_pending_order(
        "AAPL", date(2026, 7, 16), "buy", 1, 200.0, reason="test"
    )
    assert fake_conn.rows[order_id]["account_mode"] == "real"


async def test_list_pending_orders_does_not_see_the_other_account_modes_order(fake_conn, monkeypatch):
    await pending_orders.create_pending_order("AAPL", date(2026, 7, 16), "buy", 1, 200.0, reason="paper order")
    monkeypatch.setattr(settings, "kis_account_mode", "real")
    assert await pending_orders.list_pending_orders() == []


async def test_get_pending_order_returns_none_for_a_cross_mode_order(fake_conn, monkeypatch):
    order_id = await pending_orders.create_pending_order(
        "AAPL", date(2026, 7, 16), "buy", 1, 200.0, reason="paper order"
    )
    monkeypatch.setattr(settings, "kis_account_mode", "real")
    assert await pending_orders.get_pending_order(order_id) is None


async def test_approve_pending_order_cannot_touch_a_stale_paper_order_after_switching_to_real(
    fake_conn, monkeypatch
):
    # The concrete incident this test guards against: a paper-era proposal
    # sitting unresolved gets "approved" after the process restarts under
    # 'real' — this must raise, not submit a real order.
    order_id = await pending_orders.create_pending_order(
        "AAPL", date(2026, 7, 16), "buy", 1, 200.0, reason="paper order"
    )

    async def should_not_be_called(*args, **kwargs):
        raise AssertionError("must never reach mirror_trade_to_kis for a cross-mode order")

    monkeypatch.setattr("app.execution.kis_bridge.mirror_trade_to_kis", should_not_be_called)
    monkeypatch.setattr(settings, "kis_account_mode", "real")

    with pytest.raises(ValueError, match="not found or already resolved"):
        await pending_orders.approve_pending_order(order_id)


async def test_list_resolved_orders_does_not_see_the_other_account_modes_order(fake_conn, monkeypatch):
    order_id = await pending_orders.create_pending_order(
        "AAPL", date(2026, 7, 16), "buy", 1, 200.0, reason="paper order"
    )
    await pending_orders.reject_pending_order(order_id)
    monkeypatch.setattr(settings, "kis_account_mode", "real")
    assert await pending_orders.list_resolved_orders() == []


async def test_sweep_expired_only_touches_the_current_account_modes_orders(fake_conn, monkeypatch):
    # A stale 'real' order must not get swept (and its ledger entry
    # reversed under 'paper's tag) by a process currently running 'paper'.
    async def should_not_be_called(*args, **kwargs):
        raise AssertionError("must not reverse the other mode's ledger entry")

    monkeypatch.setattr("app.execution.ledger.reverse_pending_trade", should_not_be_called)

    monkeypatch.setattr(settings, "kis_account_mode", "real")
    real_order_id = await pending_orders.create_pending_order(
        "AAPL", date(2026, 7, 16), "buy", 1, 200.0, reason="real order", trade_id=999,
    )
    fake_conn.rows[real_order_id]["created_at"] = (
        datetime.now(timezone.utc) - timedelta(hours=settings.pending_order_ttl_hours + 1)
    )

    monkeypatch.setattr(settings, "kis_account_mode", "paper")
    assert await pending_orders.list_pending_orders() == []  # paper has nothing pending — sweep must not touch 'real'
    assert fake_conn.rows[real_order_id]["status"] == "pending"  # untouched


# ---------------------------------------------------------------------------
# Per-user scoping (multi-user migration Phase 4, plan §2.2): "Approval queue
# shared — user A can approve/reject user B's pending order." Approving does
# not just change a row: it submits a real order into that account.
# ---------------------------------------------------------------------------


async def test_pending_queue_is_per_user(fake_conn):
    with user_scope("u-alice"):
        await pending_orders.create_pending_order(
            "005930.KS", date(2026, 7, 16), "buy", 1, 70000.0, "alice's idea"
        )
    with user_scope("u-bob"):
        assert await pending_orders.list_pending_orders() == []
    with user_scope("u-alice"):
        assert len(await pending_orders.list_pending_orders()) == 1


async def test_one_user_cannot_fetch_anothers_pending_order(fake_conn):
    with user_scope("u-alice"):
        order_id = await pending_orders.create_pending_order(
            "005930.KS", date(2026, 7, 16), "buy", 1, 70000.0, "alice's idea"
        )
    with user_scope("u-bob"):
        assert await pending_orders.get_pending_order(order_id) is None


async def test_one_user_cannot_approve_anothers_pending_order(fake_conn):
    """The severe one: approving submits a real order using the OWNER's
    credentials, into the owner's account."""
    with user_scope("u-alice"):
        order_id = await pending_orders.create_pending_order(
            "005930.KS", date(2026, 7, 16), "buy", 1, 70000.0, "alice's idea"
        )
    with user_scope("u-bob"):
        with pytest.raises(Exception):
            await pending_orders.approve_pending_order(order_id)

    # Alice's order is untouched and still actionable by her.
    assert fake_conn.rows[order_id]["status"] == "pending"


async def test_one_user_cannot_reject_anothers_pending_order(fake_conn):
    with user_scope("u-alice"):
        order_id = await pending_orders.create_pending_order(
            "005930.KS", date(2026, 7, 16), "buy", 1, 70000.0, "alice's idea"
        )
    with user_scope("u-bob"):
        try:
            await pending_orders.reject_pending_order(order_id, "not mine")
        except Exception:
            pass
    assert fake_conn.rows[order_id]["status"] == "pending"


async def test_create_stamps_the_acting_user(fake_conn):
    with user_scope("u-alice"):
        order_id = await pending_orders.create_pending_order(
            "005930.KS", date(2026, 7, 16), "buy", 1, 70000.0, "x"
        )
    assert fake_conn.rows[order_id]["user_id"] == "u-alice"


async def test_resolved_history_is_per_user(fake_conn):
    with user_scope("u-alice"):
        order_id = await pending_orders.create_pending_order(
            "005930.KS", date(2026, 7, 16), "buy", 1, 70000.0, "x"
        )
        await pending_orders.reject_pending_order(order_id, "changed my mind")
        assert len(await pending_orders.list_resolved_orders()) == 1
    with user_scope("u-bob"):
        assert await pending_orders.list_resolved_orders() == []


async def test_sweep_expired_only_touches_the_current_users_orders(fake_conn, monkeypatch):
    # A stale order created by another user must not get swept (and its
    # ledger entry reversed) by a process currently running as this user.
    async def should_not_be_called(*args, **kwargs):
        raise AssertionError("must not reverse another user's ledger entry")

    monkeypatch.setattr("app.execution.ledger.reverse_pending_trade", should_not_be_called)

    with user_scope("u-alice"):
        alice_order_id = await pending_orders.create_pending_order(
            "005930.KS", date(2026, 7, 16), "buy", 1, 70000.0, "alice's order", trade_id=999,
        )
    fake_conn.rows[alice_order_id]["created_at"] = (
        datetime.now(timezone.utc) - timedelta(hours=settings.pending_order_ttl_hours + 1)
    )

    with user_scope("u-bob"):
        assert await pending_orders.list_pending_orders() == []  # bob has nothing pending — sweep must not touch alice's
    assert fake_conn.rows[alice_order_id]["status"] == "pending"  # untouched
