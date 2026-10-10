"""app.execution.safety_switch — kill switch + per-market daily max-loss
circuit breaker. asyncpg pool/connection faked (kill switch: single row;
daily loss: market-keyed rows, same convention as
app.execution.portfolio_risk's own per-market high-water-mark state),
real_account_snapshot faked — no real DB/network/KIS call (agent-
architecture-roadmap stage 2)."""

from __future__ import annotations

from datetime import date, datetime, timezone

import asyncpg
import pytest

import app.clock as clock_module
from app.clock import FakeClock
from app.config import settings
from app.execution import safety_switch
from app.user_context import get_current_user_id, user_scope

# The identity every non-scoped test runs under (settings.user_id, 'default').
_WHO = get_current_user_id()
from app.execution.portfolio_risk import AccountUnavailable, PortfolioSnapshot


class FakeSafetyConn:
    """In-memory stand-in for execution_safety_state (one row per user) +
    daily_loss_state (one row per (market, account_mode, user_id)).

    Both gained user_id in the multi-user migration Phase 4. The fake keys on
    it too, deliberately: if it kept collapsing users onto one row, a
    cross-user leak would pass here and only show up against real Postgres."""

    def __init__(self):
        self.kill_switch_rows: dict[str, dict] = {}
        self.daily_loss_rows: dict[tuple[str, str, str], dict] = {}

    # Back-compat for tests written before the kill switch was per-user.
    @property
    def kill_switch_row(self):
        return self.kill_switch_rows.get(get_current_user_id())

    @kill_switch_row.setter
    def kill_switch_row(self, value):
        if value is None:
            self.kill_switch_rows.pop(get_current_user_id(), None)
        else:
            self.kill_switch_rows[get_current_user_id()] = value

    async def execute(self, query, *args):
        if "CREATE TABLE" in query or "ALTER TABLE" in query or "CREATE UNIQUE INDEX" in query:
            return
        if "kill_switch_engaged = TRUE" in query:
            reason, user_id = args
            self.kill_switch_rows.setdefault(user_id, {}).update(
                kill_switch_engaged=True, kill_switch_reason=reason
            )
        elif "kill_switch_engaged = FALSE" in query:
            (user_id,) = args
            self.kill_switch_rows.setdefault(user_id, {}).update(
                kill_switch_engaged=False, kill_switch_reason=None
            )
        elif "INSERT INTO daily_loss_state" in query:
            market, account_mode, user_id, day, start_equity = args
            self.daily_loss_rows[(market, account_mode, user_id)] = {
                "daily_loss_day": day, "daily_loss_start_equity": start_equity,
            }
        elif query.strip().startswith("DELETE"):
            self.daily_loss_rows.pop(args, None)  # args = (market, account_mode, user_id)
        else:
            raise AssertionError(f"unexpected query: {query}")

    async def fetchrow(self, query, *args):
        if "execution_safety_state" in query:
            (user_id,) = args
            row = self.kill_switch_rows.get(user_id)
            if row is None:
                return None
            return {
                "kill_switch_engaged": row.get("kill_switch_engaged", False),
                "kill_switch_reason": row.get("kill_switch_reason"),
            }
        if "daily_loss_state" in query:
            key = args  # (market, account_mode, user_id)
            return self.daily_loss_rows.get(key)
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


@pytest.fixture
def fake_conn(monkeypatch):
    conn = FakeSafetyConn()
    monkeypatch.setattr(safety_switch, "_schema_ready", False)
    monkeypatch.setattr(safety_switch, "get_pool", lambda: _async_return(FakePool(conn)))
    return conn


async def _async_return(value):
    return value


def _snapshot(total: float) -> PortfolioSnapshot:
    return PortfolioSnapshot(per_ticker_value={}, total_value=total)


def _fake_real_account_snapshot(monkeypatch, values: list[float]):
    """values: sequence of totals returned by successive calls, keyed by
    call order (not by market) — matches the old test style where each
    test drives one market's baseline across two calls."""
    it = iter(values)

    async def fake(market, **kw):
        return _snapshot(next(it))

    # daily_loss_block uses real_account_snapshot (a gate); daily_loss_status
    # uses cached_account_snapshot (reporting). Patch both to the same fake.
    monkeypatch.setattr(safety_switch, "real_account_snapshot", fake)
    monkeypatch.setattr(safety_switch, "cached_account_snapshot", fake)


@pytest.fixture(autouse=True)
def _reset_settings(monkeypatch):
    monkeypatch.setattr(settings, "kis_safety_fail_open_when_unknown", False)
    monkeypatch.setattr(settings, "kis_daily_max_loss_pct", 0.05)
    monkeypatch.setattr(settings, "trade_allowlist_domestic", [])
    monkeypatch.setattr(settings, "trade_allowlist_overseas", ["AAPL"])


# ---------------------------------------------------------------------------
# kill switch
# ---------------------------------------------------------------------------


async def test_kill_switch_status_defaults_to_disengaged(fake_conn):
    assert await safety_switch.kill_switch_status() == (False, None, False)


async def test_engage_then_status_reports_engaged_with_reason(fake_conn):
    await safety_switch.engage_kill_switch("market crash, manual halt")
    assert await safety_switch.kill_switch_status() == (True, "market crash, manual halt", False)


async def test_disengage_clears_reason(fake_conn):
    await safety_switch.engage_kill_switch("test")
    await safety_switch.disengage_kill_switch()
    assert await safety_switch.kill_switch_status() == (False, None, False)


async def test_kill_switch_status_fails_closed_on_db_error(monkeypatch):
    async def broken_get_pool():
        raise OSError("db down")

    monkeypatch.setattr(safety_switch, "get_pool", broken_get_pool)
    monkeypatch.setattr(safety_switch, "_schema_ready", False)
    engaged, reason, state_unknown = await safety_switch.kill_switch_status()
    assert engaged is True
    assert "unreadable" in reason
    assert state_unknown is True


async def test_kill_switch_status_fails_closed_on_non_postgres_error(monkeypatch):
    # asyncpg.InterfaceError (pool closing / bad state) is NOT a subclass of
    # asyncpg.PostgresError — the old (OSError, PostgresError) except clause
    # let this kind of failure crash the caller instead of failing closed.
    async def broken_get_pool():
        raise asyncpg.InterfaceError("pool is closing")

    monkeypatch.setattr(safety_switch, "get_pool", broken_get_pool)
    monkeypatch.setattr(safety_switch, "_schema_ready", False)
    engaged, reason, state_unknown = await safety_switch.kill_switch_status()
    assert engaged is True
    assert "unreadable" in reason
    assert state_unknown is True


async def test_kill_switch_status_fails_closed_on_arbitrary_exception(monkeypatch):
    # A malformed database_url raises plain ValueError, not an asyncpg type
    # at all — must still fail closed, not propagate.
    async def broken_get_pool():
        raise ValueError("invalid database_url")

    monkeypatch.setattr(safety_switch, "get_pool", broken_get_pool)
    monkeypatch.setattr(safety_switch, "_schema_ready", False)
    engaged, reason, state_unknown = await safety_switch.kill_switch_status()
    assert engaged is True
    assert "unreadable" in reason
    assert state_unknown is True


async def test_kill_switch_status_fails_open_when_configured(monkeypatch):
    async def broken_get_pool():
        raise asyncpg.PostgresError("db down")

    monkeypatch.setattr(safety_switch, "get_pool", broken_get_pool)
    monkeypatch.setattr(safety_switch, "_schema_ready", False)
    monkeypatch.setattr(settings, "kis_safety_fail_open_when_unknown", True)
    assert await safety_switch.kill_switch_status() == (False, None, True)


# ---------------------------------------------------------------------------
# daily loss breaker (per market — AAPL is overseas)
# ---------------------------------------------------------------------------


async def test_daily_loss_block_never_blocks_sells(fake_conn, monkeypatch):
    async def fake_snapshot(market):
        raise AssertionError("sells should return before touching the snapshot")

    monkeypatch.setattr(safety_switch, "real_account_snapshot", fake_snapshot)
    assert await safety_switch.daily_loss_block("sell", "AAPL") is None


async def test_daily_loss_block_first_check_of_day_sets_baseline_no_block(fake_conn, monkeypatch):
    _fake_real_account_snapshot(monkeypatch, [100_000.0])
    assert await safety_switch.daily_loss_block("buy", "AAPL") is None
    assert fake_conn.daily_loss_rows[("overseas", "paper", _WHO)]["daily_loss_start_equity"] == 100_000.0


async def test_daily_loss_block_blocks_once_threshold_crossed(fake_conn, monkeypatch):
    _fake_real_account_snapshot(monkeypatch, [100_000.0, 94_000.0])  # -6% > 5% threshold
    await safety_switch.daily_loss_block("buy", "AAPL")  # sets baseline
    result = await safety_switch.daily_loss_block("buy", "AAPL")
    assert result is not None
    assert "BLOCKED" in result and "daily max-loss" in result


async def test_daily_loss_block_allows_when_under_threshold(fake_conn, monkeypatch):
    _fake_real_account_snapshot(monkeypatch, [100_000.0, 97_000.0])  # -3% < 5% threshold
    await safety_switch.daily_loss_block("buy", "AAPL")
    assert await safety_switch.daily_loss_block("buy", "AAPL") is None


async def test_daily_loss_baseline_resets_on_new_day(fake_conn, monkeypatch):
    fake_conn.daily_loss_rows[("overseas", "paper", _WHO)] = {
        "daily_loss_day": date(2026, 7, 1), "daily_loss_start_equity": 100_000.0,
    }
    _fake_real_account_snapshot(monkeypatch, [80_000.0])  # would be -20% against yesterday's baseline
    # 2026-07-02 15:30 UTC = 2026-07-02 11:30 ET — still July 2 in New York.
    fake_clock = FakeClock(datetime(2026, 7, 2, 15, 30, tzinfo=timezone.utc))
    monkeypatch.setattr(safety_switch, "get_clock", lambda: fake_clock)
    # local_today (used for the actual baseline date) now lives in app.clock
    # and reads app.clock's OWN get_clock(), not safety_switch's imported
    # binding of it — both must be patched to the same FakeClock.
    monkeypatch.setattr(clock_module, "get_clock", lambda: fake_clock)
    result = await safety_switch.daily_loss_block("buy", "AAPL")
    assert result is None  # today's own value becomes the new baseline
    assert fake_conn.daily_loss_rows[("overseas", "paper", _WHO)]["daily_loss_start_equity"] == 80_000.0
    assert fake_conn.daily_loss_rows[("overseas", "paper", _WHO)]["daily_loss_day"] == date(2026, 7, 2)


async def test_daily_loss_baseline_scopes_to_current_account_mode(fake_conn, monkeypatch):
    # A paper-era baseline must never apply to real trading (or vice versa)
    # — same class of bug as portfolio_drawdown_state's HWM, just lower
    # severity since a stale baseline self-corrects the next calendar day
    # regardless.
    fake_conn.daily_loss_rows[("overseas", "paper", _WHO)] = {
        "daily_loss_day": date(2026, 7, 2), "daily_loss_start_equity": 100_000.0,
    }
    monkeypatch.setattr(safety_switch, "get_clock", lambda: FakeClock(datetime(2026, 7, 2, 15, 30, tzinfo=timezone.utc)))
    monkeypatch.setattr(settings, "kis_account_mode", "real")
    _fake_real_account_snapshot(monkeypatch, [500_000.0])  # real account, totally different scale
    result = await safety_switch.daily_loss_block("buy", "AAPL")
    assert result is None  # fresh baseline for 'real', not a huge "gain" against paper's 100,000
    assert fake_conn.daily_loss_rows[("overseas", "real", _WHO)]["daily_loss_start_equity"] == 500_000.0
    assert fake_conn.daily_loss_rows[("overseas", "paper", _WHO)]["daily_loss_start_equity"] == 100_000.0  # untouched


async def test_reset_daily_loss_baseline_only_clears_the_current_account_mode(fake_conn, monkeypatch):
    fake_conn.daily_loss_rows[("overseas", "paper", _WHO)] = {
        "daily_loss_day": date(2026, 7, 2), "daily_loss_start_equity": 100_000.0,
    }
    fake_conn.daily_loss_rows[("overseas", "real", _WHO)] = {
        "daily_loss_day": date(2026, 7, 2), "daily_loss_start_equity": 500_000.0,
    }
    monkeypatch.setattr(settings, "kis_account_mode", "paper")
    await safety_switch.reset_daily_loss_baseline("overseas")
    assert ("overseas", "paper", _WHO) not in fake_conn.daily_loss_rows
    assert ("overseas", "real", _WHO) in fake_conn.daily_loss_rows  # untouched


async def test_daily_loss_baseline_uses_each_markets_own_local_midnight(fake_conn, monkeypatch):
    # 2026-07-02 15:30 UTC is already 2026-07-03 00:30 KST (a new KST day
    # just started) but still 2026-07-02 11:30 ET (mid-session, previous
    # day hasn't rolled over yet) — proves domestic and overseas reset on
    # their OWN local midnight, not a shared KST-anchored one.
    fixed_instant = datetime(2026, 7, 2, 15, 30, tzinfo=timezone.utc)
    # local_today (aliased here as safety_switch._local_today) lives in
    # app.clock and reads app.clock's own get_clock(), not safety_switch's
    # imported binding of it.
    monkeypatch.setattr(clock_module, "get_clock", lambda: FakeClock(fixed_instant))
    assert safety_switch._local_today("domestic") == date(2026, 7, 3)
    assert safety_switch._local_today("overseas") == date(2026, 7, 2)


async def test_daily_loss_block_fails_closed_unconditionally_on_account_unavailable(fake_conn, monkeypatch):
    # AccountUnavailable is ALWAYS fail-closed, regardless of
    # kis_safety_fail_open_when_unknown — unlike the DB-only failure below.
    async def broken_snapshot(market):
        raise AccountUnavailable("broker not connected")

    monkeypatch.setattr(safety_switch, "real_account_snapshot", broken_snapshot)
    monkeypatch.setattr(settings, "kis_safety_fail_open_when_unknown", True)
    result = await safety_switch.daily_loss_block("buy", "AAPL")
    assert result is not None and "BLOCKED" in result and "unavailable" in result


async def test_daily_loss_block_fails_closed_on_db_failure(fake_conn, monkeypatch):
    _fake_real_account_snapshot(monkeypatch, [100_000.0])

    async def broken_ready_pool():
        raise OSError("db down")

    monkeypatch.setattr(safety_switch, "_ready_pool", broken_ready_pool)
    result = await safety_switch.daily_loss_block("buy", "AAPL")
    assert result is not None and "BLOCKED" in result and "unreadable" in result


async def test_daily_loss_block_fails_open_when_configured(fake_conn, monkeypatch):
    _fake_real_account_snapshot(monkeypatch, [100_000.0])

    async def broken_ready_pool():
        raise OSError("db down")

    monkeypatch.setattr(safety_switch, "_ready_pool", broken_ready_pool)
    monkeypatch.setattr(settings, "kis_safety_fail_open_when_unknown", True)
    assert await safety_switch.daily_loss_block("buy", "AAPL") is None


# ---------------------------------------------------------------------------
# daily_loss_status / safety_gate_block
# ---------------------------------------------------------------------------


async def test_daily_loss_status_reports_unavailable_on_failure(monkeypatch, fake_conn):
    async def broken_snapshot(market, **kw):
        raise RuntimeError("down")

    monkeypatch.setattr(safety_switch, "cached_account_snapshot", broken_snapshot)
    result = await safety_switch.daily_loss_status("overseas")
    assert result == {"available": False}


async def test_daily_loss_status_reports_breach_flag(fake_conn, monkeypatch):
    _fake_real_account_snapshot(monkeypatch, [100_000.0, 90_000.0])
    await safety_switch.daily_loss_status("overseas")  # sets baseline at 100,000
    result = await safety_switch.daily_loss_status("overseas")
    assert result["available"] is True
    assert result["breached"] is True
    assert result["loss_fraction"] == pytest.approx(0.10)


async def test_safety_gate_block_kill_switch_takes_priority_over_daily_loss(fake_conn, monkeypatch):
    await safety_switch.engage_kill_switch("emergency")

    async def should_not_be_called(market):
        raise AssertionError("daily-loss check should be short-circuited by the kill switch")

    monkeypatch.setattr(safety_switch, "real_account_snapshot", should_not_be_called)
    result = await safety_switch.safety_gate_block("buy", "AAPL")
    assert result is not None and "kill switch" in result


async def test_safety_gate_block_falls_through_to_daily_loss_when_not_engaged(fake_conn, monkeypatch):
    _fake_real_account_snapshot(monkeypatch, [100_000.0])
    assert await safety_switch.safety_gate_block("buy", "AAPL") is None


# ---------------- per-user scoping (multi-user migration Phase 4) ----------------
#
# Plan §2.2: "User A's kill switch blocks user B's orders; A can also switch
# B's back off." A safety control that reaches into someone else's account is
# worse than none, because both people believe it is theirs.


async def test_kill_switch_is_per_user(fake_conn):
    with user_scope("u-alice"):
        await safety_switch.engage_kill_switch("alice panicked")
        assert await safety_switch.kill_switch_status() == (True, "alice panicked", False)

    # Bob never engaged anything. His orders must not be blocked by Alice's.
    with user_scope("u-bob"):
        assert await safety_switch.kill_switch_status() == (False, None, False)


async def test_disengaging_one_users_kill_switch_leaves_the_other_engaged(fake_conn):
    """The other half of the bug: A could clear B's panic button."""
    with user_scope("u-alice"):
        await safety_switch.engage_kill_switch("alice panicked")
    with user_scope("u-bob"):
        await safety_switch.engage_kill_switch("bob panicked")
        await safety_switch.disengage_kill_switch()
        assert await safety_switch.kill_switch_status() == (False, None, False)

    with user_scope("u-alice"):
        engaged, reason, _ = await safety_switch.kill_switch_status()
        assert engaged is True and reason == "alice panicked"


async def test_daily_loss_baseline_is_per_user(fake_conn, monkeypatch):
    """Two users on one shared equity baseline means one person's drawdown
    trips the other's breaker — and on a completely different account size."""
    monkeypatch.setattr(settings, "kis_account_mode", "paper")
    today = date(2026, 7, 2)

    with user_scope("u-alice"):
        await safety_switch._daily_loss_baseline("overseas", today, 100_000.0)
    with user_scope("u-bob"):
        await safety_switch._daily_loss_baseline("overseas", today, 5_000.0)

    assert fake_conn.daily_loss_rows[("overseas", "paper", "u-alice")]["daily_loss_start_equity"] == 100_000.0
    assert fake_conn.daily_loss_rows[("overseas", "paper", "u-bob")]["daily_loss_start_equity"] == 5_000.0


async def test_resetting_one_users_baseline_leaves_the_other(fake_conn, monkeypatch):
    monkeypatch.setattr(settings, "kis_account_mode", "paper")
    fake_conn.daily_loss_rows[("overseas", "paper", "u-alice")] = {
        "daily_loss_day": date(2026, 7, 2), "daily_loss_start_equity": 100_000.0,
    }
    fake_conn.daily_loss_rows[("overseas", "paper", "u-bob")] = {
        "daily_loss_day": date(2026, 7, 2), "daily_loss_start_equity": 5_000.0,
    }
    with user_scope("u-alice"):
        await safety_switch.reset_daily_loss_baseline("overseas")

    assert ("overseas", "paper", "u-alice") not in fake_conn.daily_loss_rows
    assert ("overseas", "paper", "u-bob") in fake_conn.daily_loss_rows
