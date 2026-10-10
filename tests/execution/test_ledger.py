"""app.execution.ledger — degrade-quietly-on-DB-failure regression tests.
No real DB; only exercises the fail-open except clauses (item 13 of the
codebase review: asyncpg.InterfaceError wasn't caught by the old
(OSError, PostgresError) clause)."""

from __future__ import annotations

import asyncio
import math
import time
from contextlib import nullcontext
from datetime import date, datetime, timedelta, timezone
from datetime import time as dtime
from zoneinfo import ZoneInfo

import asyncpg
import pytest

from app.clock import FakeClock, LiveClock, set_clock
from app.config import ExecutionMode, backtest_scope, settings
from app.execution import ledger, portfolio_risk
from app.state import PortfolioRating


@pytest.fixture(autouse=True)
def _reset_ledger_schema_ready(monkeypatch):
    monkeypatch.setattr(ledger, "_schema_ready", False)


@pytest.fixture(autouse=True)
def _reset_exit_failure_state(monkeypatch):
    # force_exit_position's failure-cooldown state (_exit_failure_state) is
    # process-local, in-memory, and keyed by ticker only (no user/account_mode
    # scoping — see its own comment) — several tests below reuse "AAPL" and a
    # failed exit in one leaves cooldown state that silently swallows the
    # next test's otherwise-successful exit attempt.
    monkeypatch.setattr(ledger, "_exit_failure_state", {})
    # Same leakage risk for filter_new_alerts' own (ticker, kind) cooldown
    # (app.live.alerts._last_fired_at) — several tests below assert on
    # exactly how many "AAPL" alerts of a given kind fired.
    from app.live import alerts as _alerts_module

    monkeypatch.setattr(_alerts_module, "_last_fired_at", {})


async def test_get_position_degrades_on_interface_error(monkeypatch):
    async def broken_ready_pool():
        raise asyncpg.InterfaceError("pool is closing")

    monkeypatch.setattr(ledger, "_ready_pool", broken_ready_pool)
    position = await ledger.get_position("AAPL")
    assert position.cash == settings.paper_starting_cash
    assert position.shares == 0


async def test_get_position_degrades_on_os_error(monkeypatch):
    async def broken_ready_pool():
        raise OSError("connection refused")

    monkeypatch.setattr(ledger, "_ready_pool", broken_ready_pool)
    position = await ledger.get_position("AAPL")
    assert position.cash == settings.paper_starting_cash


async def test_get_last_trade_degrades_on_interface_error(monkeypatch):
    async def broken_ready_pool():
        raise asyncpg.InterfaceError("pool is closing")

    monkeypatch.setattr(ledger, "_ready_pool", broken_ready_pool)
    assert await ledger.get_last_trade("AAPL") is None


async def test_get_open_stop_loss_degrades_on_interface_error(monkeypatch):
    async def broken_ready_pool():
        raise asyncpg.InterfaceError("pool is closing")

    monkeypatch.setattr(ledger, "_ready_pool", broken_ready_pool)
    assert await ledger.get_open_stop_loss("AAPL") is None


async def test_record_trade_degrades_on_interface_error(monkeypatch):
    async def broken_ready_pool():
        raise asyncpg.InterfaceError("pool is closing")

    async def fake_get_position(ticker):
        return ledger.Position(cash=1000.0, shares=0)

    monkeypatch.setattr(ledger, "_ready_pool", broken_ready_pool)
    monkeypatch.setattr(ledger, "get_position", fake_get_position)
    # Must not raise — logs and returns quietly, same as every other
    # degrade-on-failure function in this module.
    await ledger.record_trade("AAPL", date(2026, 7, 16), "buy", 1, 100.0, "test")


# ---------------------------------------------------------------------------
# rebalance_to_target's execution_mode branching
# ---------------------------------------------------------------------------


@pytest.fixture
def _rebalance_stubs(monkeypatch):
    """Minimal setup that reaches rebalance_to_target's mode branch without
    exercising the 1%-risk-cap/portfolio-cap machinery: a SELL (target
    shares < current shares) skips both, since they only apply when
    increasing exposure.

    Also stubs `market_lock` to a plain no-op (`nullcontext`) — this
    module's own docstring promises "No real DB", but rebalance_to_target's
    live path (portfolio_cap_override=None, true for every call in this
    file) always acquires the REAL Postgres advisory lock otherwise. That
    was a latent gap (these mode-branch/reversal tests don't care about
    locking at all), harmless when the lock was uncontended, but it made
    them silently depend on real external DB state — see
    test_market_lock_prevents_two_concurrent_buys_from_jointly_breaching_the_bucket_cap
    below for the one test that DOES want to exercise real serialization
    (it stubs market_lock itself, with a real asyncio.Lock, for that
    purpose).

    Also stubs `real_shares_held`/`real_equity_native` to `None` (2026-08-19)
    — both were added to rebalance_to_target's live path in the 2026-08-14
    equity-basis-mismatch fix, after these tests were written, and were never
    backfilled here. Left unmocked, a live (`as_of=None`) call falls through
    to the REAL KIS broker whenever one happens to be connected (true on any
    machine with real credentials in .env.local), producing nondeterministic
    real-network results instead of the deterministic internal-ledger
    fallback (`position.market_value(price)`) these tests were written
    against. `None` from both reproduces that fallback with no network I/O."""

    async def fake_get_position(ticker):
        return ledger.Position(cash=0.0, shares=10)

    async def fake_record_trade(*args, **kwargs):
        return None

    async def fake_real_shares_held(ticker):
        return None

    async def fake_real_equity_native(market):
        return None

    monkeypatch.setattr(ledger, "get_position", fake_get_position)
    monkeypatch.setattr(ledger, "record_trade", fake_record_trade)
    monkeypatch.setattr(ledger, "market_lock", lambda market: nullcontext())
    monkeypatch.setattr(ledger, "real_shares_held", fake_real_shares_held)
    monkeypatch.setattr(ledger, "real_equity_native", fake_real_equity_native)


async def test_rebalance_manual_mode_never_mirrors(monkeypatch, _rebalance_stubs):
    monkeypatch.setattr(settings, "execution_mode", ExecutionMode.MANUAL)

    async def should_not_be_called(*args, **kwargs):
        raise AssertionError("MANUAL mode must never call mirror_trade_to_kis")

    monkeypatch.setattr(ledger, "mirror_trade_to_kis", should_not_be_called)

    result = await ledger.rebalance_to_target("AAPL", date(2026, 7, 16), 100.0, 0.5, reason="test")
    assert result.side == "sell" and result.shares == 5


async def test_rebalance_semi_auto_mode_parks_a_pending_order(monkeypatch, _rebalance_stubs):
    monkeypatch.setattr(settings, "execution_mode", ExecutionMode.SEMI_AUTO)

    async def should_not_be_called(*args, **kwargs):
        raise AssertionError("SEMI_AUTO mode must not call mirror_trade_to_kis directly")

    monkeypatch.setattr(ledger, "mirror_trade_to_kis", should_not_be_called)

    captured = {}

    async def fake_create_pending_order(ticker, trade_date, side, shares, price, reason, trade_id=None):
        captured.update(ticker=ticker, side=side, shares=shares, price=price, reason=reason, trade_id=trade_id)
        return 42

    monkeypatch.setattr("app.execution.pending_orders.create_pending_order", fake_create_pending_order)

    result = await ledger.rebalance_to_target("AAPL", date(2026, 7, 16), 100.0, 0.5, reason="test reason")
    assert result.side == "sell" and result.shares == 5
    assert captured == {
        "ticker": "AAPL", "side": "sell", "shares": 5, "price": 100.0, "reason": "test reason",
        # _rebalance_stubs's fake_record_trade returns None, so the trade_id
        # rebalance_to_target forwards into create_pending_order is None here —
        # this also pins the wiring itself (create_pending_order must receive
        # whatever record_trade returned), not just the four original fields.
        "trade_id": None,
    }


async def test_rebalance_full_auto_mode_calls_mirror_directly(monkeypatch, _rebalance_stubs):
    monkeypatch.setattr(settings, "execution_mode", ExecutionMode.FULL_AUTO)
    captured = {}

    async def fake_mirror(ticker, trade_date, side, shares, price, assumed_current_shares=None, trade_id=None):
        captured.update(ticker=ticker, side=side, shares=shares, price=price)
        return "submitted: mock"

    monkeypatch.setattr(ledger, "mirror_trade_to_kis", fake_mirror)

    result = await ledger.rebalance_to_target("AAPL", date(2026, 7, 16), 100.0, 0.5, reason="test")
    assert result.side == "sell" and result.shares == 5


# market_lock now wraps the real-balance read + 1%-risk-rule sizing too,
# not just the portfolio-cap block (2026-08-06) — without this, two
# processes racing the same market could each size against the same
# pre-trade snapshot even though the portfolio-cap step inside the lock
# saw a fresh one, since `delta` was still computed against the STALE
# pre-lock current_shares. This test doesn't use _rebalance_stubs (which
# stubs market_lock to a no-op) specifically so it can assert on real
# lock-vs-read ordering.
async def test_rebalance_reads_real_balance_after_acquiring_the_lock_not_before(monkeypatch):
    events = []

    class _TrackingLock:
        def __init__(self, market):
            self.market = market

        async def __aenter__(self):
            events.append("lock_acquired")
            return self

        async def __aexit__(self, *exc):
            events.append("lock_released")
            return False

    async def fake_get_position(ticker):
        events.append("get_position")
        return ledger.Position(cash=0.0, shares=10)

    async def fake_real_shares_held(ticker):
        events.append("real_shares_held")
        return None

    async def fake_real_equity_native(market):
        events.append("real_equity_native")
        return None

    async def fake_record_trade(*args, **kwargs):
        return None

    async def fake_mirror(ticker, trade_date, side, shares, price, assumed_current_shares=None, trade_id=None):
        return "submitted: mock"

    monkeypatch.setattr(settings, "execution_mode", ExecutionMode.FULL_AUTO)
    monkeypatch.setattr(ledger, "market_lock", lambda market: _TrackingLock(market))
    monkeypatch.setattr(ledger, "get_position", fake_get_position)
    monkeypatch.setattr(ledger, "real_shares_held", fake_real_shares_held)
    monkeypatch.setattr(ledger, "real_equity_native", fake_real_equity_native)
    monkeypatch.setattr(ledger, "record_trade", fake_record_trade)
    monkeypatch.setattr(ledger, "mirror_trade_to_kis", fake_mirror)

    await ledger.rebalance_to_target("AAPL", date(2026, 7, 16), 100.0, 0.5, reason="test")

    assert events.index("lock_acquired") < events.index("get_position")
    assert events.index("lock_acquired") < events.index("real_shares_held")
    assert events.index("lock_acquired") < events.index("real_equity_native")
    assert events[-1] == "lock_released"


async def test_rebalance_balance_fetch_cache_spans_the_whole_lock_hold(monkeypatch):
    # 2026-09-01: cached_balance_fetches was a narrow inner scope around
    # sizing only, so mirror_trade_to_kis's own real_shares_held (via
    # _portfolio_cap_block) re-hit KIS. It now spans the whole hold — verify
    # the contextvar is still open when mirror runs.
    cache_at_mirror = {}

    async def fake_get_position(ticker):
        return ledger.Position(cash=0.0, shares=10)

    async def fake_real_shares_held(ticker):
        return None

    async def fake_real_equity_native(market):
        return None

    async def fake_record_trade(*args, **kwargs):
        return None

    async def fake_mirror(ticker, trade_date, side, shares, price, assumed_current_shares=None, trade_id=None):
        cache_at_mirror["open"] = portfolio_risk._balance_fetch_cache.get() is not None
        return "submitted: mock"

    monkeypatch.setattr(settings, "execution_mode", ExecutionMode.FULL_AUTO)
    monkeypatch.setattr(ledger, "get_position", fake_get_position)
    monkeypatch.setattr(ledger, "real_shares_held", fake_real_shares_held)
    monkeypatch.setattr(ledger, "real_equity_native", fake_real_equity_native)
    monkeypatch.setattr(ledger, "record_trade", fake_record_trade)
    monkeypatch.setattr(ledger, "mirror_trade_to_kis", fake_mirror)

    await ledger.rebalance_to_target("AAPL", date(2026, 7, 16), 100.0, 0.5, reason="test")
    assert cache_at_mirror.get("open") is True


async def test_rebalance_opens_the_balance_cache_enforcement_safe_when_live(monkeypatch):
    """FIX-1 (2026-09-03 id=1187): the live path (`portfolio_cap_override`
    is None) must open `cached_balance_fetches(enforcement_safe=True)` — the
    one condition under which `real_account_snapshot` may reuse a balance
    already fetched into this hold instead of re-asking the broker."""
    seen = {}

    async def fake_get_position(ticker):
        return ledger.Position(cash=0.0, shares=10)

    async def fake_real_shares_held(ticker):
        return None

    async def fake_real_equity_native(market):
        return None

    async def fake_record_trade(*args, **kwargs):
        return None

    async def fake_mirror(ticker, trade_date, side, shares, price, assumed_current_shares=None, trade_id=None):
        seen["enforcement_safe"] = portfolio_risk._balance_fetch_enforcement_safe.get()
        return "submitted: mock"

    monkeypatch.setattr(settings, "execution_mode", ExecutionMode.FULL_AUTO)
    monkeypatch.setattr(ledger, "get_position", fake_get_position)
    monkeypatch.setattr(ledger, "real_shares_held", fake_real_shares_held)
    monkeypatch.setattr(ledger, "real_equity_native", fake_real_equity_native)
    monkeypatch.setattr(ledger, "record_trade", fake_record_trade)
    monkeypatch.setattr(ledger, "mirror_trade_to_kis", fake_mirror)

    await ledger.rebalance_to_target("AAPL", date(2026, 7, 16), 100.0, 0.5, reason="test")
    assert seen.get("enforcement_safe") is True


async def test_rebalance_end_to_end_fetches_domestic_balance_once_per_lock_hold(monkeypatch):
    """FIX-1 end-to-end regression (2026-09-03 id=1187): within ONE real
    market_lock hold, `real_shares_held` (sizing) and
    `apply_live_portfolio_caps` -> `real_account_snapshot` must collapse to
    a SINGLE `fetch_balance` call — the second, previously-unavoidable call
    is exactly what tripped KIS's EGW00215 per-second ledger limit and
    fail-closed-blocked a real BUY (088350.KS, decision id=1187). Exercises
    the real `real_shares_held`/`apply_live_portfolio_caps`/
    `real_account_snapshot` chain against a fake broker — only
    `mirror_trade_to_kis` (order submission, a separate concern) and
    correlation clustering (network price history, irrelevant here) are
    stubbed."""
    from app.domain.models import BrokerBalance, BrokerPosition
    from app.web import state
    from conftest import broker_stub

    class _CountingDomesticBroker:
        def __init__(self):
            self.domestic_calls = 0

        async def fetch_balance(self):
            self.domestic_calls += 1
            return BrokerBalance(
                cash_deposit=1_000_000.0,
                d2_cash_deposit=1_000_000.0,
                total_eval_amount=0.0,
                total_pnl_amount=0.0,
                purchase_amount=0.0,
                positions=[],
            )

    broker = _CountingDomesticBroker()

    async def fake_get_position(ticker):
        return ledger.Position(cash=0.0, shares=0)

    async def fake_record_trade(*args, **kwargs):
        return None

    async def fake_mirror(ticker, trade_date, side, shares, price, assumed_current_shares=None, trade_id=None):
        return "submitted: mock"

    async def no_clusters(tickers, as_of=None, lookback_days=None):
        return {}

    monkeypatch.setattr(settings, "execution_mode", ExecutionMode.MANUAL)
    monkeypatch.setattr(state, "kis_broker", broker_stub(broker))
    monkeypatch.setattr(ledger, "get_position", fake_get_position)
    monkeypatch.setattr(ledger, "record_trade", fake_record_trade)
    monkeypatch.setattr(ledger, "mirror_trade_to_kis", fake_mirror)
    monkeypatch.setattr(portfolio_risk, "compute_correlation_clusters", no_clusters)
    monkeypatch.setattr(ledger, "market_lock", lambda market: nullcontext())

    result = await ledger.rebalance_to_target(
        "005930.KS", date(2026, 9, 3), 70_000.0, 1.0, reason="test buy", stop_loss=63_000.0,
        current_atr_pct=0.02,
    )

    assert result.side == "buy"
    assert broker.domestic_calls == 1  # would be 2 without FIX-1


async def test_rebalance_end_to_end_fetches_overseas_present_balance_once_per_lock_hold(monkeypatch):
    """Overseas twin of the domestic FIX-1 regression above — the exact same
    duplicate-fetch shape (id=1187, EGW00215) survived on the overseas leg
    until the 2026-09-08 follow-up (app.execution.portfolio_risk.
    real_equity_native_detailed's overseas branch calling the broker
    directly instead of the cached `_cached_fetch_overseas_present_balance`
    helper real_account_snapshot's own overseas branch already used) — the
    only end-to-end coverage of this chain was domestic-only, which is
    exactly why the gap survived 5 days undetected."""
    from app.domain.models import BrokerBalance, BrokerPosition, OverseasCash
    from app.web import state
    from conftest import broker_stub

    class _CountingOverseasBroker:
        def __init__(self):
            self.overseas_calls = 0
            self.present_balance_calls = 0

        async def fetch_overseas_balance(self):
            self.overseas_calls += 1
            return BrokerBalance(
                cash_deposit=0.0, total_eval_amount=0.0, total_pnl_amount=0.0,
                purchase_amount=0.0, positions=[],
            )

        async def fetch_overseas_present_balance(self, currency="USD"):
            self.present_balance_calls += 1
            return OverseasCash(currency=currency, deposit=100_000.0, orderable=100_000.0, eval_amount=0.0, matched=True)

    broker = _CountingOverseasBroker()

    async def fake_get_position(ticker):
        return ledger.Position(cash=0.0, shares=0)

    async def fake_record_trade(*args, **kwargs):
        return None

    async def fake_mirror(ticker, trade_date, side, shares, price, assumed_current_shares=None, trade_id=None):
        return "submitted: mock"

    async def no_clusters(tickers, as_of=None, lookback_days=None):
        return {}

    monkeypatch.setattr(settings, "execution_mode", ExecutionMode.MANUAL)
    monkeypatch.setattr(state, "kis_broker", broker_stub(broker))
    monkeypatch.setattr(ledger, "get_position", fake_get_position)
    monkeypatch.setattr(ledger, "record_trade", fake_record_trade)
    monkeypatch.setattr(ledger, "mirror_trade_to_kis", fake_mirror)
    monkeypatch.setattr(portfolio_risk, "compute_correlation_clusters", no_clusters)
    monkeypatch.setattr(ledger, "market_lock", lambda market: nullcontext())

    result = await ledger.rebalance_to_target(
        "AAPL", date(2026, 9, 8), 100.0, 1.0, reason="test buy", stop_loss=90.0, current_atr_pct=0.02,
    )

    assert result.side == "buy"
    assert broker.overseas_calls == 1  # would be 2 without FIX-1
    assert broker.present_balance_calls == 1  # would be 2 without the 2026-09-08 follow-up fix


async def test_rebalance_reversal_carries_forward_the_pre_trade_open_stop(monkeypatch, _rebalance_stubs):
    # A BLOCKED/FAILED mirror reverses this trade — the reversal must
    # restore whatever stop protected the PRE-trade position (captured via
    # get_open_stop_loss before record_trade overwrote the latest row), not
    # write stop_loss=None onto a position that's still open. Confirmed
    # live 2026-08-06: 055550.KS and AMZN both left with an open position
    # and no stop after a reversed trade.
    monkeypatch.setattr(settings, "execution_mode", ExecutionMode.FULL_AUTO)

    async def fake_mirror(ticker, trade_date, side, shares, price, assumed_current_shares=None, trade_id=None):
        return "BLOCKED: circuit-breaker state unknown"

    async def fake_get_open_stop_loss(ticker):
        return date(2026, 7, 10), 92.0  # the stop that applied BEFORE this (now-reversed) trade

    captured_reverse = {}

    async def fake_reverse_pending_trade(ticker, side, shares, price, reason, prior_stop_loss=None):
        captured_reverse.update(prior_stop_loss=prior_stop_loss)
        return 10  # back to _rebalance_stubs's fake_get_position shares=10 (full reversal)

    monkeypatch.setattr(ledger, "mirror_trade_to_kis", fake_mirror)
    monkeypatch.setattr(ledger, "get_open_stop_loss", fake_get_open_stop_loss)
    monkeypatch.setattr(ledger, "reverse_pending_trade", fake_reverse_pending_trade)

    await ledger.rebalance_to_target("AAPL", date(2026, 7, 16), 100.0, 0.5, reason="test")

    assert captured_reverse["prior_stop_loss"] == 92.0


# result.shares_after/action_label must reflect the ACTUAL post-reversal
# position, not the target this trade was aiming for but never reached —
# before this fix, a BLOCKED/reversed BUY still reported
# shares_after=target_shares and action_label="Buy"/"Increase" (derived
# from that same wrong target), even though the ledger had just reversed
# the trade back to the pre-trade position. That wrong result flowed into
# the dashboard and, via the stored decision narrative, into future
# decisions' retrieved past context, both asserting a trade that never
# actually happened.
async def test_rebalance_reversal_reports_the_reverted_position_not_the_missed_target(monkeypatch, _rebalance_stubs):
    monkeypatch.setattr(settings, "execution_mode", ExecutionMode.FULL_AUTO)

    async def fake_mirror(ticker, trade_date, side, shares, price, assumed_current_shares=None, trade_id=None):
        return "BLOCKED: circuit-breaker state unknown"

    async def fake_reverse_pending_trade(ticker, side, shares, price, reason, prior_stop_loss=None):
        return 10  # _rebalance_stubs's fake_get_position shares=10 (full reversal, back to pre-trade)

    monkeypatch.setattr(ledger, "mirror_trade_to_kis", fake_mirror)
    monkeypatch.setattr(ledger, "reverse_pending_trade", fake_reverse_pending_trade)

    result = await ledger.rebalance_to_target("AAPL", date(2026, 7, 16), 100.0, 0.5, reason="test")

    assert result.side is None
    assert result.shares == 0
    assert result.shares_after == 10
    assert result.action_label == "Hold"
    assert result.stop_loss is None


async def test_rebalance_reversal_passes_none_when_no_prior_open_stop_existed(monkeypatch, _rebalance_stubs):
    monkeypatch.setattr(settings, "execution_mode", ExecutionMode.FULL_AUTO)

    async def fake_mirror(ticker, trade_date, side, shares, price, assumed_current_shares=None, trade_id=None):
        return "BLOCKED: circuit-breaker state unknown"

    async def fake_get_open_stop_loss(ticker):
        return None  # flat before this trade, or no stop was ever set

    captured_reverse = {}

    async def fake_reverse_pending_trade(ticker, side, shares, price, reason, prior_stop_loss=None):
        captured_reverse.update(prior_stop_loss=prior_stop_loss)
        return 10  # back to _rebalance_stubs's fake_get_position shares=10 (full reversal)

    monkeypatch.setattr(ledger, "mirror_trade_to_kis", fake_mirror)
    monkeypatch.setattr(ledger, "get_open_stop_loss", fake_get_open_stop_loss)
    monkeypatch.setattr(ledger, "reverse_pending_trade", fake_reverse_pending_trade)

    await ledger.rebalance_to_target("AAPL", date(2026, 7, 16), 100.0, 0.5, reason="test")

    assert captured_reverse["prior_stop_loss"] is None


async def test_rebalance_prefers_real_shares_held_over_internal_ledger_when_live(monkeypatch):
    """Real incident: this ledger had never recorded owning a ticker (bought
    outside this system) while the real KIS account held 10 shares. A SELL
    decision computed target_shares=0 against the internal ledger's own
    (wrong) shares=0, so delta was 0 and the sell silently never fired.
    Preferring real_shares_held's 10 fixes the delta -> an actual sell."""
    monkeypatch.setattr(settings, "execution_mode", ExecutionMode.MANUAL)

    async def fake_get_position(ticker):
        return ledger.Position(cash=0.0, shares=0)  # internal ledger: never bought this

    async def fake_real_shares_held(ticker):
        return 10  # real KIS account: actually holds 10

    captured = {}

    async def fake_record_trade(*args, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(ledger, "get_position", fake_get_position)
    monkeypatch.setattr(ledger, "real_shares_held", fake_real_shares_held)
    monkeypatch.setattr(ledger, "record_trade", fake_record_trade)

    result = await ledger.rebalance_to_target("003280.KS", date(2026, 7, 23), 2000.0, 0.0, reason="test sell")

    assert result.side == "sell" and result.shares == 10
    assert captured["base_shares"] == 10


async def test_rebalance_ignores_real_shares_held_during_backtest(monkeypatch):
    """as_of set (a backtest replay) must never touch the real broker — the
    internal ledger's own shares stays authoritative, same as before this
    fix, since there's no "real account state" for a historical date."""
    monkeypatch.setattr(settings, "execution_mode", ExecutionMode.MANUAL)

    async def fake_get_position(ticker):
        return ledger.Position(cash=0.0, shares=0)

    async def should_not_be_called(*args, **kwargs):
        raise AssertionError("backtest must never call real_shares_held/real_equity_native")

    async def fake_record_trade(*args, **kwargs):
        return None

    monkeypatch.setattr(ledger, "get_position", fake_get_position)
    monkeypatch.setattr(ledger, "real_shares_held", should_not_be_called)
    monkeypatch.setattr(ledger, "real_equity_native", should_not_be_called)
    monkeypatch.setattr(ledger, "record_trade", fake_record_trade)

    result = await ledger.rebalance_to_target(
        "003280.KS", date(2026, 5, 1), 2000.0, 0.0, reason="test sell", as_of=date(2026, 5, 1),
    )
    assert result.side is None  # 0 target vs 0 internal shares -> still a no-op in backtests


async def test_rebalance_falls_back_to_internal_ledger_when_real_lookup_unavailable(monkeypatch):
    """Broker not connected / KIS call failed -> real_shares_held returns
    None, distinct from "genuinely 0 shares" -- must fall back to the
    internal ledger rather than treating None as 0."""
    monkeypatch.setattr(settings, "execution_mode", ExecutionMode.MANUAL)

    async def fake_get_position(ticker):
        return ledger.Position(cash=0.0, shares=10)

    async def fake_real_shares_held(ticker):
        return None

    captured = {}

    async def fake_record_trade(*args, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(ledger, "get_position", fake_get_position)
    monkeypatch.setattr(ledger, "real_shares_held", fake_real_shares_held)
    monkeypatch.setattr(ledger, "record_trade", fake_record_trade)

    result = await ledger.rebalance_to_target("AAPL", date(2026, 7, 23), 100.0, 0.0, reason="test sell")

    assert result.side == "sell" and result.shares == 10
    assert captured["base_shares"] == 10


async def test_rebalance_prefers_real_equity_native_over_internal_ledger_when_live(monkeypatch):
    """Real incident: the internal ledger's per-ticker cash had drifted to
    ~₩147M for one ticker alone (after enough simulated trades) while the
    REAL aggregate domestic account was worth ~$20K total — sizing off the
    fictional ₩147M produced a wildly oversized target that the (already
    real-balance-based) portfolio concentration cap then slammed back down
    to near the existing position, blocking the BUY. Preferring
    real_equity_native's real (small) figure keeps the target in scale
    with what the concentration cap itself checks against."""
    monkeypatch.setattr(settings, "execution_mode", ExecutionMode.MANUAL)

    async def fake_get_position(ticker):
        return ledger.Position(cash=147_000_000.0, shares=0)  # internal ledger: fictional, huge

    async def fake_real_shares_held(ticker):
        return 0

    async def fake_real_equity_native(market):
        return 2_000_000.0  # real KIS account: much smaller

    async def fake_apply_live_portfolio_caps(ticker, target_shares, price, current_shares):
        return target_shares, None  # isolate equity's effect on target_shares, not the cap itself

    captured = {}

    async def fake_record_trade(*args, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(ledger, "get_position", fake_get_position)
    monkeypatch.setattr(ledger, "real_shares_held", fake_real_shares_held)
    monkeypatch.setattr(ledger, "real_equity_native", fake_real_equity_native)
    monkeypatch.setattr(ledger, "apply_live_portfolio_caps", fake_apply_live_portfolio_caps)
    monkeypatch.setattr(ledger, "record_trade", fake_record_trade)

    # target_fraction=1.0 (Buy), price=2000 -> against the real ₩2M equity
    # this is 1,000 shares; against the fictional ₩147M it would be 73,500.
    # current_atr_pct given explicitly (2026-09-02 ATR gate): this test
    # starts flat (current_shares=0 via real_shares_held=0), and a missing
    # ATR now refuses any new entry outright — orthogonal to what this test
    # actually checks (equity-source selection).
    result = await ledger.rebalance_to_target(
        "373220.KS", date(2026, 7, 23), 2000.0, 1.0, reason="test buy", stop_loss=1800.0,
        current_atr_pct=0.02,
    )

    assert result.side == "buy"
    assert result.shares < 2000  # nowhere near the fictional-equity-scale figure
    assert captured["base_shares"] == 0


async def test_rebalance_falls_back_to_internal_equity_when_real_lookup_unavailable(monkeypatch):
    """Broker not connected / KIS call failed -> real_equity_native returns
    None -- must fall back to the internal ledger's own cash+shares,
    exactly like before this fix, rather than treating None as 0 equity
    (which would zero every target)."""
    monkeypatch.setattr(settings, "execution_mode", ExecutionMode.MANUAL)

    async def fake_get_position(ticker):
        return ledger.Position(cash=100_000.0, shares=0)

    async def fake_real_shares_held(ticker):
        return 0

    async def fake_real_equity_native(market):
        return None

    async def fake_apply_live_portfolio_caps(ticker, target_shares, price, current_shares):
        return target_shares, None  # isolate equity's fallback from the portfolio-cap gate itself

    captured = {}

    async def fake_record_trade(*args, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(ledger, "get_position", fake_get_position)
    monkeypatch.setattr(ledger, "real_shares_held", fake_real_shares_held)
    monkeypatch.setattr(ledger, "real_equity_native", fake_real_equity_native)
    monkeypatch.setattr(ledger, "apply_live_portfolio_caps", fake_apply_live_portfolio_caps)
    monkeypatch.setattr(ledger, "record_trade", fake_record_trade)

    # current_atr_pct given explicitly (2026-09-02 ATR gate): this test starts
    # flat (current_shares=0), and a missing ATR would now refuse the entry
    # outright — orthogonal to what this test actually checks (equity fallback).
    result = await ledger.rebalance_to_target(
        "AAPL", date(2026, 7, 23), 100.0, 1.0, reason="test buy", stop_loss=90.0,
        current_atr_pct=0.02,
    )

    # Same result as pre-fix: target_shares = floor(100_000 * 1.0 / 100) = 1000,
    # then the 1% risk cap (max_dollar_risk = 100_000 * 1% = 1000, risk_per_share
    # = 100-90 = 10) caps the add to 100 shares — proves the fallback used the
    # internal ledger's 100_000 equity, not a None-as-zero misread (which
    # would have produced 0 shares here, not 100).
    assert result.side == "buy" and result.shares == 100


async def test_rebalance_surfaces_equity_fallback_via_note_and_alert(monkeypatch):
    """Fix #3 (2026-08-14): before this, a live real_equity_native failure
    was completely silent from rebalance_to_target's side — only a
    logger.warning buried in portfolio_risk, nothing on the trade row, no
    alert. Now it must show up both in adjustment_note (survives on the
    trade row a human reviews) and as a persisted Alert (survives even a
    Hold/no-op cycle that produces no trade row at all)."""
    from app.live import alerts as alerts_module

    # filter_new_alerts' cooldown (`_last_fired_at`) is in-memory, module-
    # global, and keyed only by (ticker, kind) — shared across every test in
    # this process. Another test earlier in this run using the same "AAPL"
    # ticker could otherwise suppress this test's alert, since
    # time.monotonic() never resets between tests. Reset it here rather
    # than pick an ever-more-exotic unique ticker per test.
    monkeypatch.setattr(alerts_module, "_last_fired_at", {})
    monkeypatch.setattr(settings, "execution_mode", ExecutionMode.MANUAL)

    async def fake_get_position(ticker):
        return ledger.Position(cash=100_000.0, shares=0)

    async def fake_real_shares_held(ticker):
        return 0

    async def fake_real_equity_native(market):
        return None  # broker unavailable -> falls back to the internal ledger

    async def fake_apply_live_portfolio_caps(ticker, target_shares, price, current_shares):
        return target_shares, None

    captured_alerts = []

    async def fake_record_alert(alert):
        captured_alerts.append(alert)

    async def fake_record_trade(*args, **kwargs):
        return None

    monkeypatch.setattr(ledger, "get_position", fake_get_position)
    monkeypatch.setattr(ledger, "real_shares_held", fake_real_shares_held)
    monkeypatch.setattr(ledger, "real_equity_native", fake_real_equity_native)
    monkeypatch.setattr(ledger, "apply_live_portfolio_caps", fake_apply_live_portfolio_caps)
    monkeypatch.setattr(ledger, "record_trade", fake_record_trade)
    monkeypatch.setattr(ledger, "record_alert", fake_record_alert)

    # current_atr_pct given explicitly (2026-09-02 ATR gate) — see the same
    # note on test_rebalance_falls_back_to_internal_equity_when_real_lookup_
    # unavailable above.
    result = await ledger.rebalance_to_target(
        "AAPL", date(2026, 7, 23), 100.0, 1.0, reason="test buy", stop_loss=90.0,
        current_atr_pct=0.02,
    )

    assert result.adjustment_note is not None
    assert "fell back to the internal ledger" in result.adjustment_note
    assert len(captured_alerts) == 1
    assert captured_alerts[0].kind == "equity_basis_fallback"
    assert "AAPL" in captured_alerts[0].message


async def test_rebalance_does_not_alert_equity_fallback_during_backtest(monkeypatch):
    """Backtest always uses the internal ledger by design — that's not a
    fallback/failure, so neither the note nor the alert should fire. Mirrors
    test_rebalance_ignores_real_shares_held_during_backtest's shape (a
    target that's a no-op against the internal ledger, so no portfolio-cap
    machinery needs exercising)."""

    async def fake_get_position(ticker):
        return ledger.Position(cash=0.0, shares=0)

    async def should_not_be_called(*args, **kwargs):
        raise AssertionError("backtest must never call real_shares_held/real_equity_native")

    captured_alerts = []

    async def fake_record_alert(alert):
        captured_alerts.append(alert)

    async def fake_record_trade(*args, **kwargs):
        return None

    monkeypatch.setattr(ledger, "get_position", fake_get_position)
    monkeypatch.setattr(ledger, "real_shares_held", should_not_be_called)
    monkeypatch.setattr(ledger, "real_equity_native", should_not_be_called)
    monkeypatch.setattr(ledger, "record_trade", fake_record_trade)
    monkeypatch.setattr(ledger, "record_alert", fake_record_alert)

    result = await ledger.rebalance_to_target(
        "003280.KS", date(2026, 5, 1), 2000.0, 0.0, reason="test sell", as_of=date(2026, 5, 1),
    )

    assert not captured_alerts
    assert result.adjustment_note is None or "fell back to the internal ledger" not in result.adjustment_note


# ---------------------------------------------------------------------------
# resolve_equity_basis' real-account refusal (2026-08-31). Until now this
# function was only ever exercised indirectly, through rebalance_to_target —
# which meant the one branch that matters most (real mode + live + no broker
# balance => raise instead of fall back) had no test pinning the exception's
# own contract: which of the four input combinations raise, and what
# `.ticker`/`.market`/`.failure` carry for the caller building an alert out of
# them. The three non-raising combinations are pinned alongside it precisely
# because a fail-closed guard that fires too widely is worse than none: it
# would silently stop every paper trade and every backtest.
# 실계좌+라이브에서만 예외가 나야 함 — 나머지 세 조합은 기존 폴백을 유지해야 하므로 함께 고정.
# ---------------------------------------------------------------------------


async def test_resolve_equity_basis_real_success_returns_real_account(monkeypatch):
    """The happy path is mode-independent: a real broker figure wins over the
    internal ledger in paper mode too (paper KIS accounts are real accounts
    with fake money — their balance is still the honest sizing basis)."""
    monkeypatch.setattr(settings, "kis_account_mode", "real")

    async def fake_real_equity_native(market):
        return 4242.0

    monkeypatch.setattr(ledger, "real_equity_native", fake_real_equity_native)

    basis = await ledger.resolve_equity_basis(
        "AAPL", 100.0, as_of=None, position=ledger.Position(cash=1000.0, shares=5)
    )

    assert basis == ledger.EquityBasis(4242.0, "real_account")


async def test_resolve_equity_basis_paper_mode_falls_back_to_internal_ledger(monkeypatch):
    """Paper mode keeps the pre-2026-08-31 fallback unchanged. There is no
    real position to contradict the internal ledger's numbers there, so
    raising would only stop paper trading for no safety gain."""
    monkeypatch.setattr(settings, "kis_account_mode", "paper")

    async def fake_real_equity_native(market):
        return None

    async def should_not_be_called(market):
        raise AssertionError("paper mode must not reach the failure-detail lookup")

    monkeypatch.setattr(ledger, "real_equity_native", fake_real_equity_native)
    monkeypatch.setattr(ledger, "real_equity_native_detailed", should_not_be_called)

    basis = await ledger.resolve_equity_basis(
        "AAPL", 100.0, as_of=None, position=ledger.Position(cash=1000.0, shares=5)
    )

    # 1000 cash + 5 shares * 100 = 1500 (Position.market_value)
    assert basis == ledger.EquityBasis(1500.0, "internal_ledger")


async def test_resolve_equity_basis_backtest_always_internal_ledger_even_in_real_mode(monkeypatch):
    """The `as_of` check sits ABOVE the account-mode check on purpose: a
    historical replay has no real account to contradict, so real mode must
    not make a backtest raise (or even reach the broker). This is what makes
    rebalance_to_target's `else` branch provably unable to raise
    EquityBasisUnavailable, which is why only its live branch is wrapped."""
    monkeypatch.setattr(settings, "kis_account_mode", "real")

    async def should_not_be_called(market):
        raise AssertionError("a backtest must never reach the real broker")

    monkeypatch.setattr(ledger, "real_equity_native", should_not_be_called)
    monkeypatch.setattr(ledger, "real_equity_native_detailed", should_not_be_called)

    basis = await ledger.resolve_equity_basis(
        "AAPL", 100.0, as_of=date(2026, 5, 1), position=ledger.Position(cash=1000.0, shares=5)
    )

    assert basis == ledger.EquityBasis(1500.0, "internal_ledger")


async def test_resolve_equity_basis_real_mode_live_raises_when_broker_unavailable(monkeypatch):
    """The whole point of the change: rather than size a REAL account against
    the internal ledger's `starting_cash_for` seed (100,000 USD default =
    ~₩137M of fiction), refuse outright. The three attributes are asserted
    because rebalance_to_target and _risk_cap_context both build their
    user-facing text out of them rather than re-parsing the message."""
    monkeypatch.setattr(settings, "kis_account_mode", "real")

    async def fake_real_equity_native(market):
        return None

    async def fake_real_equity_native_detailed(market):
        return portfolio_risk.EquityLookup(None, "broker_unavailable")

    monkeypatch.setattr(ledger, "real_equity_native", fake_real_equity_native)
    monkeypatch.setattr(ledger, "real_equity_native_detailed", fake_real_equity_native_detailed)

    with pytest.raises(ledger.EquityBasisUnavailable) as excinfo:
        await ledger.resolve_equity_basis(
            "AAPL", 100.0, as_of=None, position=ledger.Position(cash=1000.0, shares=5)
        )

    assert excinfo.value.ticker == "AAPL"
    assert excinfo.value.market == "overseas"
    assert excinfo.value.failure == "broker_unavailable"
    assert "broker_unavailable" in str(excinfo.value)


async def test_resolve_equity_basis_still_raises_when_the_detail_lookup_itself_blows_up(monkeypatch):
    """real_equity_native_detailed is called ONLY to name the failure, and its
    own try/except doesn't cover the `state.kis_broker()` resolution above the
    KIS calls. If that raises, the refusal must still stand with an unnamed
    reason — swapping EquityBasisUnavailable for an unrelated traceback would
    escape rebalance_to_target's handler and abort the pipeline run, the exact
    failure shape (action_label permanently NULL) this whole change exists to
    prevent.
    실패 사유 조회가 터져도 본 예외를 덮으면 안 됨 — 사유만 'unknown'으로."""
    monkeypatch.setattr(settings, "kis_account_mode", "real")

    async def fake_real_equity_native(market):
        return None

    async def exploding_detailed(market):
        raise RuntimeError("kis_broker() resolution failed")

    monkeypatch.setattr(ledger, "real_equity_native", fake_real_equity_native)
    monkeypatch.setattr(ledger, "real_equity_native_detailed", exploding_detailed)

    with pytest.raises(ledger.EquityBasisUnavailable) as excinfo:
        await ledger.resolve_equity_basis(
            "AAPL", 100.0, as_of=None, position=ledger.Position(cash=1000.0, shares=5)
        )

    assert excinfo.value.failure is None
    assert "unknown" in str(excinfo.value)


async def test_rebalance_skips_trade_when_real_equity_unavailable_in_real_mode(monkeypatch):
    """End-to-end counterpart of the raise above: the refusal must surface as
    a clean no-trade Hold, NOT as an exception escaping into
    app.orchestrator.pipeline (which has no handler for it — an escape aborts
    the run after the decision row is already stored, leaving action_label
    permanently NULL and no execution note). record_trade is stubbed to raise
    so a regression that resumes sizing against the internal ledger fails
    loudly here instead of quietly writing a fictional-equity trade.
    실계좌 잔고 조회 실패 시 예외를 올리지 않고 Hold로 반환 + 장부 기록 없음."""
    monkeypatch.setattr(settings, "kis_account_mode", "real")
    monkeypatch.setattr(settings, "execution_mode", ExecutionMode.MANUAL)

    async def fake_get_position(ticker):
        return ledger.Position(cash=100_000.0, shares=0)

    async def fake_real_shares_held(ticker):
        return 7  # the REAL held quantity, deliberately != the ledger's 0

    async def fake_real_equity_native(market):
        return None

    async def fake_real_equity_native_detailed(market):
        return portfolio_risk.EquityLookup(None, "broker_unavailable")

    async def should_not_be_called(*args, **kwargs):
        raise AssertionError("a skipped cycle must never write a trade row")

    captured_alerts = []

    async def fake_record_alert(alert):
        captured_alerts.append(alert)

    monkeypatch.setattr(ledger, "get_position", fake_get_position)
    monkeypatch.setattr(ledger, "real_shares_held", fake_real_shares_held)
    monkeypatch.setattr(ledger, "real_equity_native", fake_real_equity_native)
    monkeypatch.setattr(ledger, "real_equity_native_detailed", fake_real_equity_native_detailed)
    monkeypatch.setattr(ledger, "record_trade", should_not_be_called)
    monkeypatch.setattr(ledger, "mirror_trade_to_kis", should_not_be_called)
    monkeypatch.setattr(ledger, "record_alert", fake_record_alert)
    monkeypatch.setattr(ledger, "market_lock", lambda market: nullcontext())

    result = await ledger.rebalance_to_target(
        "AAPL", date(2026, 7, 23), 100.0, 1.0, reason="test buy", stop_loss=90.0,
    )

    assert result.side is None
    assert result.shares == 0
    # 7, not the internal ledger's 0: current_shares was already corrected to
    # the real held quantity before resolve_equity_basis raised, and that is
    # the honest "position unchanged" figure.
    assert result.shares_after == 7
    # No ATR-derived stop may be reported for a cycle declared un-sizable.
    assert result.stop_loss is None
    assert result.action_label == "Hold"
    assert result.adjustment_note is not None
    assert "broker_unavailable" in result.adjustment_note
    assert "7 shares" in result.adjustment_note

    # The alert is the only durable trace: a skipped cycle produces no trade
    # row for adjustment_note to live on.
    assert len(captured_alerts) == 1
    assert captured_alerts[0].kind == "equity_basis_unavailable"
    assert "AAPL" in captured_alerts[0].message


async def test_rebalance_paper_mode_is_untouched_by_the_real_mode_refusal(monkeypatch):
    """Guard on the blast radius of the try/except above: with the SAME broker
    outage, paper mode must still size a real trade off the internal ledger
    and must NOT emit the skip alert. Deliberately narrower than
    test_rebalance_surfaces_equity_fallback_via_note_and_alert (which already
    pins the fallback note/alert content) — this one asserts only the two
    things that change if the wrap ever starts catching in paper mode.
    페이퍼 모드에서는 기존 폴백 동작이 그대로여야 함."""
    monkeypatch.setattr(settings, "kis_account_mode", "paper")
    monkeypatch.setattr(settings, "execution_mode", ExecutionMode.MANUAL)

    async def fake_get_position(ticker):
        return ledger.Position(cash=100_000.0, shares=0)

    async def fake_real_shares_held(ticker):
        return 0

    async def fake_real_equity_native(market):
        return None

    async def fake_apply_live_portfolio_caps(ticker, target_shares, price, current_shares):
        return target_shares, None

    captured_alerts = []

    async def fake_record_alert(alert):
        captured_alerts.append(alert)

    async def fake_record_trade(*args, **kwargs):
        return None

    monkeypatch.setattr(ledger, "get_position", fake_get_position)
    monkeypatch.setattr(ledger, "real_shares_held", fake_real_shares_held)
    monkeypatch.setattr(ledger, "real_equity_native", fake_real_equity_native)
    monkeypatch.setattr(ledger, "apply_live_portfolio_caps", fake_apply_live_portfolio_caps)
    monkeypatch.setattr(ledger, "record_trade", fake_record_trade)
    monkeypatch.setattr(ledger, "record_alert", fake_record_alert)
    monkeypatch.setattr(ledger, "market_lock", lambda market: nullcontext())

    # current_atr_pct given explicitly (2026-09-02 ATR gate) — see the note
    # on test_rebalance_falls_back_to_internal_equity_when_real_lookup_
    # unavailable above.
    result = await ledger.rebalance_to_target(
        "AAPL", date(2026, 7, 23), 100.0, 1.0, reason="test buy", stop_loss=90.0,
        current_atr_pct=0.02,
    )

    # Same 100 shares as test_rebalance_falls_back_to_internal_equity_when_
    # real_lookup_unavailable: the 1% risk cap on 100,000 internal equity.
    assert result.side == "buy" and result.shares == 100
    assert [a.kind for a in captured_alerts] == ["equity_basis_fallback"]


async def test_rebalance_skip_alert_respects_the_per_ticker_cooldown(monkeypatch):
    """An extended broker outage must not emit one alert per cycle per ticker.
    filter_new_alerts' (account_mode, ticker, kind) cooldown is deliberately
    left REAL here — the autouse _reset_exit_failure_state fixture clears
    _last_fired_at at the start of each test, so the two calls below share one
    fresh cooldown window (settings.alert_cooldown_seconds = 300s against
    time.monotonic(), so the second call is inside it by construction).
    장기 장애 시 사이클마다 알림이 쏟아지지 않도록 쿨다운 확인."""
    monkeypatch.setattr(settings, "kis_account_mode", "real")
    monkeypatch.setattr(settings, "execution_mode", ExecutionMode.MANUAL)

    async def fake_get_position(ticker):
        return ledger.Position(cash=100_000.0, shares=0)

    async def fake_real_shares_held(ticker):
        return 7

    async def fake_real_equity_native(market):
        return None

    async def fake_real_equity_native_detailed(market):
        return portfolio_risk.EquityLookup(None, "broker_unavailable")

    captured_alerts = []

    async def fake_record_alert(alert):
        captured_alerts.append(alert)

    monkeypatch.setattr(ledger, "get_position", fake_get_position)
    monkeypatch.setattr(ledger, "real_shares_held", fake_real_shares_held)
    monkeypatch.setattr(ledger, "real_equity_native", fake_real_equity_native)
    monkeypatch.setattr(ledger, "real_equity_native_detailed", fake_real_equity_native_detailed)
    monkeypatch.setattr(ledger, "record_alert", fake_record_alert)
    monkeypatch.setattr(ledger, "market_lock", lambda market: nullcontext())

    for _ in range(2):
        result = await ledger.rebalance_to_target(
            "AAPL", date(2026, 7, 23), 100.0, 1.0, reason="test buy", stop_loss=90.0,
        )
        # Every cycle still skips and still carries the note — only the ALERT
        # is deduped. Suppressing the note too would hide the reason on the
        # second and later cycles.
        assert result.side is None
        assert "broker_unavailable" in result.adjustment_note

    assert len(captured_alerts) == 1


# ---------------------------------------------------------------------------
# portfolio_risk_unavailable_block alert — FIX-5, 2026-09-03 id=1187 audit.
# A fail-CLOSED portfolio-risk block (apply_live_portfolio_caps couldn't even
# read the real account snapshot) means the AI's BUY never executed at all,
# not a routine concentration/correlation/cash-floor trim — must raise its
# own alert, distinct from an ordinary cap note which must NOT.
# ---------------------------------------------------------------------------


async def test_rebalance_raises_alert_when_portfolio_risk_check_is_unavailable(monkeypatch):
    monkeypatch.setattr(settings, "execution_mode", ExecutionMode.MANUAL)

    async def fake_get_position(ticker):
        return ledger.Position(cash=0.0, shares=0)

    async def fake_real_shares_held(ticker):
        return 0

    async def fake_real_equity_native(market):
        return 100_000.0

    async def fake_apply_live_portfolio_caps(ticker, target_shares, price, current_shares):
        note = (
            f"{portfolio_risk.PORTFOLIO_RISK_UNAVAILABLE_NOTE_PREFIX} (KIS domestic balance inquiry "
            f"failed: boom) — blocked the exposure increase to {target_shares} shares; held at the "
            f"existing {current_shares} instead."
        )
        return current_shares, note

    async def fake_record_trade(*args, **kwargs):
        return None

    captured_alerts = []

    async def fake_record_alert(alert):
        captured_alerts.append(alert)

    monkeypatch.setattr(ledger, "get_position", fake_get_position)
    monkeypatch.setattr(ledger, "real_shares_held", fake_real_shares_held)
    monkeypatch.setattr(ledger, "real_equity_native", fake_real_equity_native)
    monkeypatch.setattr(ledger, "apply_live_portfolio_caps", fake_apply_live_portfolio_caps)
    monkeypatch.setattr(ledger, "record_trade", fake_record_trade)
    monkeypatch.setattr(ledger, "record_alert", fake_record_alert)
    monkeypatch.setattr(ledger, "market_lock", lambda market: nullcontext())

    result = await ledger.rebalance_to_target(
        "AAPL", date(2026, 7, 23), 100.0, 1.0, reason="test buy", stop_loss=90.0,
        current_atr_pct=0.02,
    )

    assert result.side is None  # blocked back to the existing (flat) position
    assert [a.kind for a in captured_alerts] == ["portfolio_risk_unavailable_block"]
    assert "AAPL" in captured_alerts[0].message
    assert portfolio_risk.PORTFOLIO_RISK_UNAVAILABLE_NOTE_PREFIX in captured_alerts[0].message


async def test_rebalance_does_not_raise_alert_for_an_ordinary_cap_note(monkeypatch):
    """Regression guard: a routine concentration/correlation/cash-floor
    reduction (which still increases exposure, just by less) must NOT raise
    the fail-closed alert — only the specific "risk check unavailable" note
    prefix does."""
    monkeypatch.setattr(settings, "execution_mode", ExecutionMode.MANUAL)

    async def fake_get_position(ticker):
        return ledger.Position(cash=0.0, shares=0)

    async def fake_real_shares_held(ticker):
        return 0

    async def fake_real_equity_native(market):
        return 100_000.0

    async def fake_apply_live_portfolio_caps(ticker, target_shares, price, current_shares):
        return current_shares + 1, "Portfolio concentration cap: reduced the add from 100 to 1 shares."

    async def fake_record_trade(*args, **kwargs):
        return None

    captured_alerts = []

    async def fake_record_alert(alert):
        captured_alerts.append(alert)

    monkeypatch.setattr(ledger, "get_position", fake_get_position)
    monkeypatch.setattr(ledger, "real_shares_held", fake_real_shares_held)
    monkeypatch.setattr(ledger, "real_equity_native", fake_real_equity_native)
    monkeypatch.setattr(ledger, "apply_live_portfolio_caps", fake_apply_live_portfolio_caps)
    monkeypatch.setattr(ledger, "record_trade", fake_record_trade)
    monkeypatch.setattr(ledger, "record_alert", fake_record_alert)
    monkeypatch.setattr(ledger, "market_lock", lambda market: nullcontext())

    await ledger.rebalance_to_target(
        "AAPL", date(2026, 7, 23), 100.0, 1.0, reason="test buy", stop_loss=90.0,
        current_atr_pct=0.02,
    )

    assert captured_alerts == []


async def test_rebalance_direction_guard_clamps_underweight_that_would_otherwise_buy(monkeypatch):
    """Real incident (memory: live-rebalance-equity-basis-mismatch-2026-08-14):
    GOOG, 2026-08-13 — an Underweight rating whose target_fraction was set
    against the internal ledger's (small) per-ticker equity, but whose actual
    sizing here used real_equity_native's much larger real-account figure,
    computed to an INCREASE (109 shares) despite the position already being
    above that fraction of the smaller basis (73 shares) — a SELL rating
    that would have executed as a 36-share BUY. KIS happened to reject that
    specific order; this guard doesn't rely on that happening again."""
    monkeypatch.setattr(settings, "execution_mode", ExecutionMode.MANUAL)

    async def fake_get_position(ticker):
        return ledger.Position(cash=0.0, shares=73)  # currently holds 73

    async def fake_real_shares_held(ticker):
        return 73

    async def fake_real_equity_native(market):
        return 248_000.0  # real account equity — much larger than the internal basis

    async def fake_apply_live_portfolio_caps(ticker, target_shares, price, current_shares):
        return target_shares, None  # isolate the direction guard from the portfolio-cap gate

    captured = {}

    async def fake_record_trade(*args, **kwargs):
        captured.update(kwargs)
        return None

    monkeypatch.setattr(ledger, "get_position", fake_get_position)
    monkeypatch.setattr(ledger, "real_shares_held", fake_real_shares_held)
    monkeypatch.setattr(ledger, "real_equity_native", fake_real_equity_native)
    monkeypatch.setattr(ledger, "apply_live_portfolio_caps", fake_apply_live_portfolio_caps)
    monkeypatch.setattr(ledger, "record_trade", fake_record_trade)

    # 15% of the real $248,000 equity at $340.55/share -> floor(37,200/340.55)
    # = 109 shares, above the current 73 -> without the guard this prices out
    # as a BUY, exactly inverting the Underweight rating's intent to reduce.
    result = await ledger.rebalance_to_target(
        "GOOG", date(2026, 8, 13), 340.55, 0.15, reason="test underweight",
        stop_loss=300.0, rating=PortfolioRating.UNDERWEIGHT,
    )

    assert result.side is None  # clamped to current_shares -> no-op, not a buy
    assert result.shares_after == 73
    assert result.adjustment_note is not None and "Direction guard" in result.adjustment_note


async def test_rebalance_direction_guard_does_not_touch_legitimate_buy_overweight(monkeypatch):
    """The guard must only fire for Underweight/Sell — a Buy/Overweight
    rating legitimately increasing exposure must be unaffected."""
    monkeypatch.setattr(settings, "execution_mode", ExecutionMode.MANUAL)

    async def fake_get_position(ticker):
        return ledger.Position(cash=0.0, shares=10)

    async def fake_real_shares_held(ticker):
        return 10

    async def fake_real_equity_native(market):
        return 248_000.0

    async def fake_apply_live_portfolio_caps(ticker, target_shares, price, current_shares):
        return target_shares, None

    async def fake_record_trade(*args, **kwargs):
        return None

    monkeypatch.setattr(ledger, "get_position", fake_get_position)
    monkeypatch.setattr(ledger, "real_shares_held", fake_real_shares_held)
    monkeypatch.setattr(ledger, "real_equity_native", fake_real_equity_native)
    monkeypatch.setattr(ledger, "apply_live_portfolio_caps", fake_apply_live_portfolio_caps)
    monkeypatch.setattr(ledger, "record_trade", fake_record_trade)

    result = await ledger.rebalance_to_target(
        # Tight stop (340.0, 0.55 below price) so the unrelated 1%-risk-rule
        # cap (app.execution.ledger's own per-trade risk bound, further up
        # this function) doesn't also shrink the add — this test isolates
        # the direction guard, not that separate cap.
        "GOOG", date(2026, 8, 13), 340.55, 0.15, reason="test overweight",
        stop_loss=340.0, rating=PortfolioRating.OVERWEIGHT,
    )

    assert result.side == "buy"
    assert result.shares_after == 109


# ---------------------------------------------------------------------------
# check_stops_for_ticker — reads the trailing-stop state (read-only) as the
# hourly backstop for app.live.exit_monitor's own real-time ratcheting.
# ---------------------------------------------------------------------------


async def test_check_stops_for_ticker_uses_plain_pm_stop_when_no_trailing_state(monkeypatch):
    fixed = datetime(2026, 7, 25, 8, 0, tzinfo=ZoneInfo("Asia/Seoul"))
    set_clock(FakeClock(fixed))
    try:
        async def fake_get_open_stop_loss(ticker):
            return date(2026, 7, 20), 90.0

        async def fake_entry_time(ticker):
            return datetime(2026, 7, 20, 9, 0, tzinfo=timezone.utc), 100.0, 90.0

        async def fake_get_current_state(ticker, entry_time):
            return None  # exit_monitor never established trailing state for this streak

        async def fake_daily_lows(ticker, start, end):
            return {date(2026, 7, 24): 95.0}

        captured = {}

        async def fake_check_and_apply(ticker, daily_lows, stop_loss, reason, *, record_as_of=None):
            captured["stop_loss"] = stop_loss
            captured["record_as_of"] = record_as_of
            return None

        monkeypatch.setattr(ledger, "get_open_stop_loss", fake_get_open_stop_loss)
        monkeypatch.setattr(ledger, "get_open_stop_loss_entry_time", fake_entry_time)
        monkeypatch.setattr(ledger.trailing_stop, "get_current_state", fake_get_current_state)
        monkeypatch.setattr(ledger, "get_daily_low_series", fake_daily_lows)
        monkeypatch.setattr(ledger, "check_and_apply_stop_loss", fake_check_and_apply)

        await ledger.check_stops_for_ticker("AAPL")
        assert captured["stop_loss"] == 90.0  # the plain PM stop, unchanged
        assert captured["record_as_of"] == date(2026, 7, 25)  # today, not the historical breach day
    finally:
        set_clock(LiveClock())


async def test_check_stops_for_ticker_walks_the_ratchet_day_by_day_when_trailing_state_exists(monkeypatch):
    """Fix (2026-08-14, memory: live-trailing-stop-retroactive-breach-2026-08-14):
    this used to compute ONE floor from TODAY's high-water (150) and test
    every past day against it — always looser (higher, thus easier to
    breach) than what was actually in force on any earlier day, since
    high-water only ratchets up. Now it must delegate to
    check_and_apply_trailing_stop (the already-correct day-by-day engine
    the backtest replay has always used), seeded from entry_price (100),
    NOT from the live high-water (150) — seeding from the live value would
    just reintroduce the same retroactive-application bug one level down.

    Amended 2026-08-27: the live tick high-water is no longer written into
    `daily_highs[today]` at all — it is handed to the walk as its own
    `live_high_water`/`live_high_water_day` pair, so that a tick-level high
    the 5-minute bars never show cannot fail those bars' reconciliation
    against the daily bar and knock today back onto the very same-day
    high-before-low rule the 047040.KS fix removed."""
    fixed = datetime(2026, 7, 25, 8, 0, tzinfo=ZoneInfo("Asia/Seoul"))
    set_clock(FakeClock(fixed))
    try:
        entry_time = datetime(2026, 7, 20, 9, 0, tzinfo=timezone.utc)

        async def fake_get_open_stop_loss(ticker):
            return date(2026, 7, 20), 90.0

        async def fake_entry_time(ticker):
            return entry_time, 100.0, 90.0

        async def fake_get_current_state(ticker, et):
            assert et == entry_time
            return 150.0, 0.1  # high_water, trailing_pct

        async def fake_daily_lows(ticker, start, end):
            return {date(2026, 7, 24): 140.0}

        async def fake_daily_highs(ticker, start, end):
            return {date(2026, 7, 24): 145.0}  # today's own daily-bar high not yet at 150

        captured = {}

        async def fake_check_and_apply_trailing(
            ticker, daily_highs, daily_lows, pm_stop_loss, entry_price, high_water, trailing_pct, reason,
            *, record_as_of=None, intraday_bars=None, adjudicated_through=None, entry_time=None,
            live_high_water=None, live_high_water_day=None,
        ):
            captured.update(
                daily_highs=dict(daily_highs), pm_stop_loss=pm_stop_loss, entry_price=entry_price,
                high_water=high_water, trailing_pct=trailing_pct, record_as_of=record_as_of,
                intraday_bars=intraday_bars, adjudicated_through=adjudicated_through,
                entry_time=entry_time, live_high_water=live_high_water,
                live_high_water_day=live_high_water_day,
            )
            return None, high_water

        async def fake_intraday_bars(_ticker, _day):
            return []

        async def fake_get_adjudicated_through(_ticker, _entry_time, _pm_stop):
            return None

        async def fake_record_adjudicated_through(*_args, **_kwargs):
            return None

        monkeypatch.setattr(ledger, "get_open_stop_loss", fake_get_open_stop_loss)
        monkeypatch.setattr(ledger, "get_open_stop_loss_entry_time", fake_entry_time)
        monkeypatch.setattr(ledger.trailing_stop, "get_current_state", fake_get_current_state)
        monkeypatch.setattr(ledger, "get_daily_low_series", fake_daily_lows)
        monkeypatch.setattr(ledger, "get_daily_high_series", fake_daily_highs)
        monkeypatch.setattr(ledger, "check_and_apply_trailing_stop", fake_check_and_apply_trailing)
        monkeypatch.setattr(ledger, "get_intraday_bars", fake_intraday_bars)
        monkeypatch.setattr(ledger.trailing_stop, "get_adjudicated_through", fake_get_adjudicated_through)
        monkeypatch.setattr(
            ledger.trailing_stop, "record_adjudicated_through", fake_record_adjudicated_through
        )

        await ledger.check_stops_for_ticker("AAPL")

        assert captured["high_water"] == 100.0  # seeded from entry_price, not the live 150
        assert captured["entry_price"] == 100.0
        assert captured["pm_stop_loss"] == 90.0
        assert captured["trailing_pct"] == 0.1
        assert captured["record_as_of"] == date(2026, 7, 25)
        # 2026-08-27: the live high-water (150) is NO LONGER written into
        # daily_highs[today] — it travels as its own parameter instead.
        assert date(2026, 7, 25) not in captured["daily_highs"]
        assert captured["live_high_water"] == 150.0
        assert captured["live_high_water_day"] == date(2026, 7, 25)
        assert captured["daily_highs"][date(2026, 7, 24)] == 145.0  # still untouched
        assert captured["entry_time"] == entry_time
        assert captured["adjudicated_through"] is None
    finally:
        set_clock(LiveClock())


# ---------------------------------------------------------------------------
# check_and_apply_stop_loss — real KIS shares preferred over internal ledger
# ---------------------------------------------------------------------------


async def test_check_and_apply_stop_loss_prefers_real_shares_held_over_internal_ledger(monkeypatch):
    """Same drift risk rebalance_to_target already guards against (see
    test_rebalance_prefers_real_shares_held_over_internal_ledger_when_live):
    after a paper account reset, the internal ledger can say a different
    quantity than the real KIS account actually holds. A forced stop-loss
    exit must sell the REAL quantity, not the internal ledger's stale one —
    selling the wrong size both mis-records the ledger and mirrors a bad
    size to KIS."""

    async def fake_get_position(ticker):
        return ledger.Position(cash=0.0, shares=3)  # internal ledger: stale, says 3

    async def fake_real_shares_held(ticker):
        return 10  # real KIS account: actually holds 10

    captured_record = {}
    captured_mirror = {}

    async def fake_record_trade(ticker, trade_date, side, shares, price, reason, **kwargs):
        captured_record.update(shares=shares, **kwargs)

    async def fake_mirror_trade_to_kis(ticker, trade_date, side, shares, price, assumed_current_shares, approved=False, trade_id=None):
        captured_mirror.update(shares=shares, assumed_current_shares=assumed_current_shares)
        return "mirrored"

    async def fake_record_alert(alert):
        return None

    monkeypatch.setattr(ledger, "get_position", fake_get_position)
    monkeypatch.setattr(ledger, "real_shares_held", fake_real_shares_held)
    monkeypatch.setattr(ledger, "record_trade", fake_record_trade)
    monkeypatch.setattr(ledger, "mirror_trade_to_kis", fake_mirror_trade_to_kis)
    monkeypatch.setattr(ledger, "record_alert", fake_record_alert)

    breach_day = date(2026, 7, 24)
    result = await ledger.check_and_apply_stop_loss(
        "005930.KS", {breach_day: 9.0}, stop_loss=10.0, reason="test breach"
    )

    assert result.ok
    assert result.breach_date == breach_day
    assert captured_record["shares"] == 10
    assert captured_record["base_shares"] == 10
    assert captured_mirror["shares"] == 10
    assert captured_mirror["assumed_current_shares"] == 10


async def test_check_and_apply_stop_loss_falls_back_to_internal_ledger_when_real_lookup_unavailable(monkeypatch):
    """Broker not connected / KIS call failed -> real_shares_held returns
    None, distinct from "genuinely 0 shares" -- must fall back to the
    internal ledger's own shares, exactly like rebalance_to_target's
    identical fallback."""

    async def fake_get_position(ticker):
        return ledger.Position(cash=0.0, shares=7)

    async def fake_real_shares_held(ticker):
        return None

    captured_record = {}

    async def fake_record_trade(ticker, trade_date, side, shares, price, reason, **kwargs):
        captured_record.update(shares=shares, **kwargs)

    async def fake_mirror_trade_to_kis(ticker, trade_date, side, shares, price, assumed_current_shares, approved=False, trade_id=None):
        return "mirrored"

    async def fake_record_alert(alert):
        return None

    monkeypatch.setattr(ledger, "get_position", fake_get_position)
    monkeypatch.setattr(ledger, "real_shares_held", fake_real_shares_held)
    monkeypatch.setattr(ledger, "record_trade", fake_record_trade)
    monkeypatch.setattr(ledger, "mirror_trade_to_kis", fake_mirror_trade_to_kis)
    monkeypatch.setattr(ledger, "record_alert", fake_record_alert)

    breach_day = date(2026, 7, 24)
    result = await ledger.check_and_apply_stop_loss(
        "AAPL", {breach_day: 9.0}, stop_loss=10.0, reason="test breach"
    )

    assert result.ok
    assert result.breach_date == breach_day
    assert captured_record["shares"] == 7
    assert captured_record["base_shares"] == 7


# local_today(market), not get_clock().today() (KST): mirror_trade_to_kis's
# own today-only gate compares against THIS market's local calendar date.
# Confirmed live 2026-08-06: an overseas forced-exit mirrored with the KST
# date was rejected as "not today" for ~5 of every 6.5 hours of the US
# regular session (KST has already rolled to the next day), and — because
# that rejection was worded "skipped:" rather than "BLOCKED:" — never
# reversed: the ledger recorded the position closed while KIS still held
# it, with the stop-loss row consumed and no further monitoring.
async def test_check_and_apply_stop_loss_mirrors_with_the_markets_own_local_date(monkeypatch):
    # KST 2026-07-25 08:00 is still 2026-07-24 19:00 in US/Eastern — the
    # exact KST-ahead-of-ET mismatch window this fix targets.
    fixed = datetime(2026, 7, 25, 8, 0, tzinfo=ZoneInfo("Asia/Seoul"))
    set_clock(FakeClock(fixed))
    try:

        async def fake_get_position(ticker):
            return ledger.Position(cash=0.0, shares=10)

        async def fake_real_shares_held(ticker):
            return None

        captured_mirror = {}

        async def fake_mirror_trade_to_kis(ticker, trade_date, side, shares, price, assumed_current_shares, approved=False, trade_id=None):
            captured_mirror["trade_date"] = trade_date
            return "mirrored"

        async def fake_record_trade(ticker, trade_date, side, shares, price, reason, **kwargs):
            return None

        async def fake_record_alert(alert):
            return None

        monkeypatch.setattr(ledger, "get_position", fake_get_position)
        monkeypatch.setattr(ledger, "real_shares_held", fake_real_shares_held)
        monkeypatch.setattr(ledger, "record_trade", fake_record_trade)
        monkeypatch.setattr(ledger, "mirror_trade_to_kis", fake_mirror_trade_to_kis)
        monkeypatch.setattr(ledger, "record_alert", fake_record_alert)

        breach_day = date(2026, 7, 24)
        await ledger.check_and_apply_stop_loss("AAPL", {breach_day: 9.0}, stop_loss=10.0, reason="test breach")

        assert captured_mirror["trade_date"] == date(2026, 7, 24)  # ET date, not the KST date (7/25)
    finally:
        set_clock(LiveClock())


async def test_check_and_apply_stop_loss_reversal_carries_forward_the_triggering_stop(monkeypatch):
    # If the forced-exit mirror never reaches KIS (BLOCKED/FAILED), the
    # reversal restores the exact pre-exit position — still protected by
    # whatever stop was actually open on this position BEFORE this exit's
    # own record_trade call overwrote the latest row (get_open_stop_loss,
    # captured up front in _force_exit_locked) — not exit_price, which
    # would re-arm the stop at the exact price the exit just failed to
    # sell at and guarantee an immediate re-breach (2026-08-19 AXTI retry
    # storm). In this test the pre-exit stop happens to equal exit_price
    # (10.0), so the assertion looks unchanged, but it's now sourced from
    # get_open_stop_loss rather than the literal stop_loss/exit_price arg.
    async def fake_get_position(ticker):
        return ledger.Position(cash=0.0, shares=10)

    async def fake_real_shares_held(ticker):
        return None

    async def fake_get_open_stop_loss(ticker):
        return (date(2026, 7, 20), 10.0)

    async def fake_mirror_trade_to_kis(ticker, trade_date, side, shares, price, assumed_current_shares, approved=False, trade_id=None):
        return "BLOCKED: circuit-breaker state unknown"

    async def fake_record_trade(ticker, trade_date, side, shares, price, reason, **kwargs):
        return None

    async def fake_record_alert(alert):
        return None

    captured_reverse = {}

    async def fake_reverse_pending_trade(ticker, side, shares, price, reason, prior_stop_loss=None):
        captured_reverse.update(prior_stop_loss=prior_stop_loss)
        return 10  # unused by check_and_apply_stop_loss's own caller, just a valid int

    monkeypatch.setattr(ledger, "get_position", fake_get_position)
    monkeypatch.setattr(ledger, "real_shares_held", fake_real_shares_held)
    monkeypatch.setattr(ledger, "get_open_stop_loss", fake_get_open_stop_loss)
    monkeypatch.setattr(ledger, "record_trade", fake_record_trade)
    monkeypatch.setattr(ledger, "mirror_trade_to_kis", fake_mirror_trade_to_kis)
    monkeypatch.setattr(ledger, "record_alert", fake_record_alert)
    monkeypatch.setattr(ledger, "reverse_pending_trade", fake_reverse_pending_trade)

    breach_day = date(2026, 7, 24)
    result = await ledger.check_and_apply_stop_loss("AAPL", {breach_day: 9.0}, stop_loss=10.0, reason="test breach")

    assert captured_reverse["prior_stop_loss"] == 10.0
    assert result.status == "failed"


async def test_check_and_apply_stop_loss_no_position_is_a_noop(monkeypatch):
    """Flat in both the real account and the internal ledger -> nothing to
    exit, no trade/mirror/alert calls at all."""

    async def fake_get_position(ticker):
        return ledger.Position(cash=0.0, shares=0)

    async def fake_real_shares_held(ticker):
        return 0

    async def should_not_be_called(*args, **kwargs):
        raise AssertionError("no position -> record_trade/mirror must never be called")

    monkeypatch.setattr(ledger, "get_position", fake_get_position)
    monkeypatch.setattr(ledger, "real_shares_held", fake_real_shares_held)
    monkeypatch.setattr(ledger, "record_trade", should_not_be_called)
    monkeypatch.setattr(ledger, "mirror_trade_to_kis", should_not_be_called)

    result = await ledger.check_and_apply_stop_loss(
        "AAPL", {date(2026, 7, 24): 9.0}, stop_loss=10.0, reason="test breach"
    )
    assert result is None


# ---------------------------------------------------------------------------
# check_and_apply_trailing_stop — the backtest-replay day-by-day ratchet
# (app.backtest.harness's own in-memory counterpart to app.live.
# exit_monitor's live ratchet).
# ---------------------------------------------------------------------------


def _stub_flat_exit(monkeypatch):
    """force_exit_position always reports a real exit — these tests care
    about WHICH day/price check_and_apply_trailing_stop picks, not the
    execution machinery itself (already covered by force_exit_position's
    own tests)."""
    captured = {}

    async def fake_force_exit(ticker, exit_date, exit_price, reason, *, record_as_of=None):
        captured.update(
            ticker=ticker, exit_date=exit_date, exit_price=exit_price, reason=reason,
            record_as_of=record_as_of,
        )
        return ledger.ForcedExitOutcome(status="exited", shares=1, disposition="mirrored")

    monkeypatch.setattr(ledger, "force_exit_position", fake_force_exit)
    return captured


async def test_trailing_stop_gap_check_never_breached_ratchets_high_water_only(monkeypatch):
    captured = _stub_flat_exit(monkeypatch)
    highs = {date(2026, 7, 17): 110.0, date(2026, 7, 18): 120.0}
    lows = {date(2026, 7, 17): 105.0, date(2026, 7, 18): 115.0}  # never near the floor
    breach_day, high_water = await ledger.check_and_apply_trailing_stop(
        "AAPL", highs, lows, pm_stop_loss=80.0, entry_price=100.0,
        high_water=100.0, trailing_pct=0.1, reason="test",
    )
    assert breach_day is None
    assert high_water == 120.0  # ratcheted up from the gap's own highs
    assert captured == {}  # never called


async def test_trailing_stop_gap_check_breaches_on_pullback_after_runup(monkeypatch):
    captured = _stub_flat_exit(monkeypatch)
    # Day 1: runs up to 150 (high). Day 2: pulls back, low of 130 — below
    # the trailing floor from day 1's high (150 * 0.9 = 135), even though
    # the plain PM stop (80.0) is nowhere close.
    highs = {date(2026, 7, 17): 150.0, date(2026, 7, 18): 132.0}
    lows = {date(2026, 7, 17): 148.0, date(2026, 7, 18): 130.0}
    outcome, high_water = await ledger.check_and_apply_trailing_stop(
        "AAPL", highs, lows, pm_stop_loss=80.0, entry_price=100.0,
        high_water=100.0, trailing_pct=0.1, reason="test",
    )
    assert outcome.breach_date == date(2026, 7, 18)
    assert captured["exit_price"] == pytest.approx(135.0)  # 150 * (1 - 0.1)
    assert high_water == 150.0


async def test_trailing_stop_gap_check_same_day_high_ratchets_before_low_check(monkeypatch):
    # A single day whose own high is high enough to activate/tighten the
    # trailing floor past what that SAME day's low then breaches. As of
    # 2026-08-27 this pins the FALLBACK rule, not the primary one: with no
    # intraday bars passed (every backtest call, and any live day whose bars
    # are missing or don't reconcile), the day's high is still treated as
    # having already happened when its low is checked. Passing bars for this
    # same day flips the verdict — see
    # test_trailing_stop_intraday_bars_clear_the_047040_shaped_false_breach.
    # This test must keep passing unchanged: it is the proof that
    # app.backtest.harness's numbers did not move.
    captured = _stub_flat_exit(monkeypatch)
    highs = {date(2026, 7, 17): 200.0}
    lows = {date(2026, 7, 17): 175.0}  # below 200*(1-0.1)=180, above entry/pm stop
    outcome, high_water = await ledger.check_and_apply_trailing_stop(
        "AAPL", highs, lows, pm_stop_loss=80.0, entry_price=100.0,
        high_water=100.0, trailing_pct=0.1, reason="test",
    )
    assert outcome.breach_date == date(2026, 7, 17)
    assert captured["exit_price"] == pytest.approx(180.0)
    assert high_water == 200.0


async def test_trailing_stop_gap_check_never_picks_a_floor_looser_than_pm_stop(monkeypatch):
    captured = _stub_flat_exit(monkeypatch)
    # High-water only slightly above entry, wide trailing_pct -> the
    # trailing floor would be LOOSER than the PM's own stop; max() must
    # still pick the PM's stop as the breach threshold.
    highs = {date(2026, 7, 17): 105.0}
    lows = {date(2026, 7, 17): 96.0}  # above the PM stop (95), below a hypothetical loose trailing floor
    breach_day, _ = await ledger.check_and_apply_trailing_stop(
        "AAPL", highs, lows, pm_stop_loss=95.0, entry_price=100.0,
        high_water=100.0, trailing_pct=0.5, reason="test",
    )
    assert breach_day is None  # 96 > 95 (PM stop) even though 96 < 52.5 (loose trailing floor)


async def test_trailing_stop_gap_check_returns_none_shares_when_already_flat(monkeypatch):
    async def fake_force_exit(ticker, exit_date, exit_price, reason, *, record_as_of=None):
        return ledger.ForcedExitOutcome(status="already_flat")

    monkeypatch.setattr(ledger, "force_exit_position", fake_force_exit)
    highs = {date(2026, 7, 17): 150.0}
    lows = {date(2026, 7, 17): 100.0}
    outcome, _ = await ledger.check_and_apply_trailing_stop(
        "AAPL", highs, lows, pm_stop_loss=80.0, entry_price=100.0,
        high_water=100.0, trailing_pct=0.1, reason="test",
    )
    assert outcome is None  # a real breach happened, but nothing to exit


# ---------------------------------------------------------------------------
# get_open_stop_loss_entry_time — the tick-monitor's entry-timestamp read
# ---------------------------------------------------------------------------


def _fake_trade_row(*, created_at, price, stop_loss, shares_after, row_id=1, reason="AI Overweight decision"):
    # reason deliberately NOT a forced-exit row (STOP_LOSS_REASON_PREFIX) —
    # these tests exercise get_open_stop_loss_entry_time's own shape, not
    # the provisional-forced-exit race guard (see the dedicated
    # _select_effective_trade_row tests below for that).
    return {
        "trade_date": created_at.date(), "stop_loss": stop_loss, "shares_after": shares_after,
        "reason": reason, "mirror_disposition": "submitted: order 1", "created_at": created_at,
        "id": row_id, "price": price,
    }


async def test_get_open_stop_loss_entry_time_returns_created_at_and_entry_price(monkeypatch):
    entry_time = datetime(2026, 7, 24, 9, 5, 30, tzinfo=timezone.utc)

    class _FakeConn:
        async def fetch(self, query, *args):
            return [_fake_trade_row(created_at=entry_time, price=95.0, stop_loss=10.0, shares_after=5)]

    class _FakePool:
        def acquire(self):
            return _AcquireCtx(_FakeConn())

    async def fake_ready_pool():
        return _FakePool()

    monkeypatch.setattr(ledger, "_ready_pool", fake_ready_pool)
    result = await ledger.get_open_stop_loss_entry_time("AAPL")
    assert result == (entry_time, 95.0, 10.0)


async def test_get_open_stop_loss_entry_time_none_when_flat(monkeypatch):
    class _FakeConn:
        async def fetch(self, query, *args):
            return [_fake_trade_row(
                created_at=datetime.now(timezone.utc), price=95.0, stop_loss=10.0, shares_after=0,
            )]

    class _FakePool:
        def acquire(self):
            return _AcquireCtx(_FakeConn())

    async def fake_ready_pool():
        return _FakePool()

    monkeypatch.setattr(ledger, "_ready_pool", fake_ready_pool)
    assert await ledger.get_open_stop_loss_entry_time("AAPL") is None


class _AcquireCtx:
    def __init__(self, conn):
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, *exc_info):
        return False


# ---------------------------------------------------------------------------
# force_exit_position — per-ticker lock (2026-08-10, added for app.live.
# exit_monitor: the hourly daily-bar loop and the new 5s tick loop can both
# reach for the same ticker close together)
# ---------------------------------------------------------------------------


async def test_force_exit_position_serializes_concurrent_calls_for_the_same_ticker(monkeypatch):
    """Two callers racing force_exit_position for the SAME ticker (the
    hourly stop_loss_loop and the tick-based exit monitor) must not both
    pass the shares>0 check before either's record_trade lands — that would
    double-sell. Simulated by making the fake get_position/real_shares_held
    yield to the event loop (so a real race would interleave without the
    lock) and asserting record_trade only ever fires once."""
    remaining_shares = 10
    record_trade_calls = 0

    async def fake_get_position(ticker):
        return ledger.Position(cash=0.0, shares=remaining_shares)

    async def fake_real_shares_held(ticker):
        await asyncio.sleep(0)  # yield — where an unlocked race would interleave
        return remaining_shares

    async def fake_record_trade(ticker, trade_date, side, shares, price, reason, **kwargs):
        nonlocal remaining_shares, record_trade_calls
        record_trade_calls += 1
        remaining_shares = 0
        return 1

    async def fake_mirror_trade_to_kis(ticker, trade_date, side, shares, price, assumed_current_shares, approved=False, trade_id=None):
        return "mirrored"

    async def fake_record_alert(alert):
        return None

    monkeypatch.setattr(ledger, "get_position", fake_get_position)
    monkeypatch.setattr(ledger, "real_shares_held", fake_real_shares_held)
    monkeypatch.setattr(ledger, "record_trade", fake_record_trade)
    monkeypatch.setattr(ledger, "mirror_trade_to_kis", fake_mirror_trade_to_kis)
    monkeypatch.setattr(ledger, "record_alert", fake_record_alert)

    results = await asyncio.gather(
        ledger.force_exit_position("AAPL", date(2026, 7, 24), 9.0, "reason A"),
        ledger.force_exit_position("AAPL", date(2026, 7, 24), 9.0, "reason B"),
    )

    assert record_trade_calls == 1
    statuses = sorted(r.status for r in results)
    assert statuses == ["already_flat", "exited"]
    exited = next(r for r in results if r.status == "exited")
    assert exited.shares == 10


# ---------------------------------------------------------------------------
# force_exit_position — failure cooldown (2026-08-19, AXTI's 17-attempt,
# 43-minute stop-loss retry storm: the breach condition stays true for as
# long as the exit order keeps failing, so without a cooldown the same
# doomed order gets resubmitted on every tick forever).
# ---------------------------------------------------------------------------


def _stub_failing_exit(monkeypatch, *, disposition: str):
    """A position that always exists (10 shares) whose KIS mirror always
    fails with `disposition` — get_open_stop_loss is left unmocked (returns
    None via its own fail-open path with no real DB), so the reversal's
    prior_stop_loss ends up None; irrelevant to what these tests check."""

    async def fake_get_position(ticker):
        return ledger.Position(cash=0.0, shares=10)

    async def fake_real_shares_held(ticker):
        return None

    async def fake_mirror_trade_to_kis(ticker, trade_date, side, shares, price, assumed_current_shares, approved=False, trade_id=None):
        return disposition

    async def fake_record_trade(ticker, trade_date, side, shares, price, reason, **kwargs):
        return None

    reverse_calls = []

    async def fake_reverse_pending_trade(ticker, side, shares, price, reason, prior_stop_loss=None):
        reverse_calls.append(prior_stop_loss)
        return shares

    alerts = []

    async def fake_record_alert(alert):
        alerts.append(alert)

    monkeypatch.setattr(ledger, "get_position", fake_get_position)
    monkeypatch.setattr(ledger, "real_shares_held", fake_real_shares_held)
    monkeypatch.setattr(ledger, "record_trade", fake_record_trade)
    monkeypatch.setattr(ledger, "mirror_trade_to_kis", fake_mirror_trade_to_kis)
    monkeypatch.setattr(ledger, "reverse_pending_trade", fake_reverse_pending_trade)
    monkeypatch.setattr(ledger, "record_alert", fake_record_alert)
    return reverse_calls, alerts


async def test_force_exit_position_suppresses_immediate_retry_after_a_transient_failure(monkeypatch):
    reverse_calls, alerts = _stub_failing_exit(
        monkeypatch, disposition="FAILED: [EGW00201 transient] rate limited"
    )

    first = await ledger.force_exit_position("AAPL", date(2026, 7, 24), 9.0, "reason")
    assert first.status == "failed"
    assert len(reverse_calls) == 1

    # Immediately retrying (as the next 5s tick would) must be suppressed —
    # NOT attempt another doomed mirror call.
    second = await ledger.force_exit_position("AAPL", date(2026, 7, 24), 9.0, "reason")
    assert second.status == "cooldown"
    assert len(reverse_calls) == 1  # no second mirror/reversal attempt

    # Cooldown expiring lets the next attempt through again. Captures the
    # real time ONCE before patching — `ledger.time` and this test module's
    # `time` are the same module object, so a lambda that calls
    # `time.monotonic()` after patching it would recurse into itself.
    base = time.monotonic()
    monkeypatch.setattr(ledger.time, "monotonic", lambda: base + 120)
    third = await ledger.force_exit_position("AAPL", date(2026, 7, 24), 9.0, "reason")
    assert third.status == "failed"
    assert len(reverse_calls) == 2


async def test_force_exit_position_permanent_failure_gets_the_longest_backoff_immediately(monkeypatch):
    """A malformed-request failure (IGW00013-style, tagged 'permanent' by
    kis_bridge) is doomed to fail again with the exact same payload no
    matter how soon it's retried — so unlike a transient failure, it jumps
    straight to the long backoff on the FIRST failure, not just after
    several consecutive ones."""
    _stub_failing_exit(monkeypatch, disposition="FAILED: [IGW00013 permanent] bad ORD_QTY")

    first = await ledger.force_exit_position("AAPL", date(2026, 7, 24), 9.0, "reason")
    assert first.status == "failed"

    # Still suppressed a full minute later (the shortest transient backoff
    # would have already expired by now). Base captured before patching —
    # see the sibling test's comment for why.
    base = time.monotonic()
    monkeypatch.setattr(ledger.time, "monotonic", lambda: base + 65)
    second = await ledger.force_exit_position("AAPL", date(2026, 7, 24), 9.0, "reason")
    assert second.status == "cooldown"


async def test_force_exit_position_failed_alert_uses_a_distinct_kind_and_flags_position_unprotected(monkeypatch):
    _, alerts = _stub_failing_exit(monkeypatch, disposition="FAILED: [IGW00013 permanent] bad ORD_QTY")

    await ledger.force_exit_position("AAPL", date(2026, 7, 24), 9.0, "reason")

    assert len(alerts) == 1
    assert alerts[0].kind == "stop_loss_failed"
    assert "sold" not in alerts[0].message
    assert "UNPROTECTED" in alerts[0].message


async def test_force_exit_position_escalates_alert_after_repeated_failures(monkeypatch):
    _, alerts = _stub_failing_exit(monkeypatch, disposition="FAILED: [EGW00201 transient] rate limited")
    base = time.monotonic()

    for i in range(3):
        await ledger.force_exit_position("AAPL", date(2026, 7, 24), 9.0, "reason")
        # jump past whatever backoff was just set so the next call proceeds
        monkeypatch.setattr(ledger.time, "monotonic", lambda i=i: base + 3700 * (i + 1))

    assert len(alerts) == 3
    assert "CRITICAL" in alerts[2].message
    assert "CRITICAL" not in alerts[0].message


async def test_force_exit_position_success_clears_the_failure_streak(monkeypatch):
    call_count = 0

    async def fake_get_position(ticker):
        return ledger.Position(cash=0.0, shares=10)

    async def fake_real_shares_held(ticker):
        return None

    async def fake_mirror_trade_to_kis(ticker, trade_date, side, shares, price, assumed_current_shares, approved=False, trade_id=None):
        nonlocal call_count
        call_count += 1
        return "FAILED: [EGW00201 transient] rate limited" if call_count == 1 else "submitted: ok"

    async def fake_record_trade(ticker, trade_date, side, shares, price, reason, **kwargs):
        return None

    async def fake_reverse_pending_trade(ticker, side, shares, price, reason, prior_stop_loss=None):
        return shares

    async def fake_record_alert(alert):
        return None

    monkeypatch.setattr(ledger, "get_position", fake_get_position)
    monkeypatch.setattr(ledger, "real_shares_held", fake_real_shares_held)
    monkeypatch.setattr(ledger, "record_trade", fake_record_trade)
    monkeypatch.setattr(ledger, "mirror_trade_to_kis", fake_mirror_trade_to_kis)
    monkeypatch.setattr(ledger, "reverse_pending_trade", fake_reverse_pending_trade)
    monkeypatch.setattr(ledger, "record_alert", fake_record_alert)

    first = await ledger.force_exit_position("AAPL", date(2026, 7, 24), 9.0, "reason")
    assert first.status == "failed"

    base = time.monotonic()
    monkeypatch.setattr(ledger.time, "monotonic", lambda: base + 120)
    second = await ledger.force_exit_position("AAPL", date(2026, 7, 24), 9.0, "reason")
    assert second.status == "exited"

    # The streak cleared on success — a THIRD call right away (no time
    # jump) must not be suppressed by cooldown.
    third = await ledger.force_exit_position("AAPL", date(2026, 7, 24), 9.0, "reason")
    assert third.status != "cooldown"


# ---------------------------------------------------------------------------
# market_lock — serializes the live decide-and-commit sequence per market
# ---------------------------------------------------------------------------


async def test_market_lock_prevents_two_concurrent_buys_from_jointly_breaching_the_bucket_cap(monkeypatch):
    """Two same-bucket tickers (005930.KS, 000660.KS — both domestic, same
    correlation cluster) analyzed concurrently. Each one ALONE stays under
    the 45% correlation-bucket cap, but the SECOND one to commit must see
    the FIRST one's already-recorded value in the bucket total — otherwise
    both would fetch the same pre-trade snapshot and both pass, jointly
    breaching the cap. `fake_real_account_snapshot`'s `asyncio.sleep(0)` is
    a deliberate yield point (same technique as
    tests/execution/test_kis_bridge.py's RacyBroker) — without
    `portfolio_risk.market_lock` actually serializing this section, the two
    `rebalance_to_target` calls could interleave right there and both read
    the bucket as empty.

    `market_lock` itself is now a Postgres advisory lock (2026-08-04, cross-
    process) rather than a plain asyncio.Lock — its own SQL/degrade-quietly
    mechanics are tested in tests/execution/test_portfolio_risk.py. This
    test only needs to prove rebalance_to_target actually serializes on
    WHATEVER `market_lock` returns, so it stands in a plain asyncio.Lock
    here rather than exercising real Postgres."""
    monkeypatch.setattr(settings, "execution_mode", ExecutionMode.MANUAL)  # skip mirror/pending-order plumbing
    # Same lock instance regardless of market — both tickers here are
    # domestic anyway, but a fresh Lock() per call would give the two
    # concurrent rebalance_to_target calls independent locks and prove nothing.
    shared_lock = asyncio.Lock()
    monkeypatch.setattr(ledger, "market_lock", lambda market: shared_lock)
    monkeypatch.setattr(settings, "portfolio_correlation_bucket_max_pct", 0.45)
    monkeypatch.setattr(settings, "portfolio_concentration_max_pct", 0.25)

    async def fake_get_position(ticker):
        return ledger.Position(cash=0.0, shares=0)

    async def fake_real_shares_held(ticker):
        return 0

    async def fake_real_equity_native(market):
        return 30_000.0  # target_shares = floor(30_000 * 1.0 / 10.0) = 3000

    per_ticker_value: dict[str, float] = {}

    async def fake_real_account_snapshot(market):
        # Captured BEFORE the yield, not after: simulates "the balance is
        # read, then something else happens before this trade actually
        # commits" — the real race window. Capturing after the yield would
        # accidentally always observe whichever trade committed first,
        # passing even without `market_lock` actually serializing anything.
        captured = dict(per_ticker_value)
        await asyncio.sleep(0)  # yield point — without market_lock, the other call interleaves here
        return portfolio_risk.PortfolioSnapshot(per_ticker_value=captured, total_value=100_000.0)

    async def fake_compute_correlation_clusters(tickers, as_of=None, lookback_days=None):
        return {"005930.KS": 1, "000660.KS": 1}  # same bucket

    async def fake_current_drawdown_fraction(market, total_value):
        return 0.0

    async def fake_record_trade(ticker, trade_date, side, shares, price, reason, **kwargs):
        # Simulates the trade landing in the real account before the lock
        # releases — record_trade runs INSIDE the locked section, same as
        # the real mirror_trade_to_kis call that follows it.
        per_ticker_value[ticker] = shares * price

    monkeypatch.setattr(ledger, "get_position", fake_get_position)
    monkeypatch.setattr(ledger, "real_shares_held", fake_real_shares_held)
    monkeypatch.setattr(ledger, "real_equity_native", fake_real_equity_native)
    monkeypatch.setattr(ledger, "record_trade", fake_record_trade)
    monkeypatch.setattr(portfolio_risk, "real_account_snapshot", fake_real_account_snapshot)
    monkeypatch.setattr(portfolio_risk, "compute_correlation_clusters", fake_compute_correlation_clusters)
    monkeypatch.setattr(portfolio_risk, "current_drawdown_fraction", fake_current_drawdown_fraction)

    results = await asyncio.gather(
        # current_atr_pct given explicitly (2026-09-02 ATR gate): both start
        # flat, and a missing ATR would now refuse the entry outright. Small
        # enough (0.001) that resolve_stop_loss's floor (price*atr*multiple=
        # 0.025) stays below the given stop distance (0.1) and doesn't widen
        # it, preserving this test's risk-cap arithmetic unchanged.
        ledger.rebalance_to_target(
            "005930.KS", date(2026, 7, 28), 10.0, 1.0, reason="buy A", stop_loss=9.9, current_atr_pct=0.001,
        ),
        ledger.rebalance_to_target(
            "000660.KS", date(2026, 7, 28), 10.0, 1.0, reason="buy B", stop_loss=9.9, current_atr_pct=0.001,
        ),
    )

    shares = sorted(r.shares for r in results)
    # Whichever ticker committed first gets the full, uncapped 3000 shares
    # (its own bucket_frac is 30_000/130_000 = 23.1%, under the 45% cap).
    # The second one MUST see the first one's 30_000 already in the bucket
    # and get capped down — proving it read a fresh, not stale, snapshot.
    assert shares[1] == 3000
    assert shares[0] < 3000


async def test_record_trade_base_shares_override_prevents_a_stale_shares_after(monkeypatch):
    """Without base_shares, record_trade re-derives shares_after from its
    own get_position lookup — exactly the stale internal-ledger read that
    caused the bug this override exists to prevent. With base_shares=10
    (the real quantity), a sell of 10 must leave shares_after=0, not -10."""

    async def fake_get_position(ticker):
        return ledger.Position(cash=0.0, shares=0)  # internal ledger still thinks 0

    captured = {}

    async def fake_ready_pool():
        class _FakeConn:
            async def fetchrow(self, query, *args):
                captured["shares_after"] = args[6]
                return {"id": 1}

            def acquire(self):
                return self

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

        return _FakeConn()

    monkeypatch.setattr(ledger, "get_position", fake_get_position)
    monkeypatch.setattr(ledger, "_ready_pool", fake_ready_pool)

    await ledger.record_trade("003280.KS", date(2026, 7, 23), "sell", 10, 2000.0, "test", base_shares=10)

    assert captured["shares_after"] == 0


async def test_rebalance_persists_the_1pct_risk_cap_note_on_the_trade_row(monkeypatch):
    """RebalanceResult.adjustment_note must reach record_trade so the capped
    reason is queryable straight off the trades row, not only buried inside
    memory_entries.decision_text via append_execution_note (which is skipped
    whenever decision_id is None, e.g. every backtest/no-decision caller)."""
    monkeypatch.setattr(settings, "execution_mode", ExecutionMode.MANUAL)
    monkeypatch.setattr(settings, "sizing_max_risk_pct_of_equity", 0.001)

    async def fake_get_position(ticker):
        return ledger.Position(cash=100_000.0, shares=0)

    async def fake_apply_live_portfolio_caps(ticker, target_shares, price, current_shares):
        return target_shares, None  # no further reduction from portfolio caps

    captured = {}

    async def fake_record_trade(*args, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(ledger, "get_position", fake_get_position)
    monkeypatch.setattr(ledger, "apply_live_portfolio_caps", fake_apply_live_portfolio_caps)
    monkeypatch.setattr(ledger, "record_trade", fake_record_trade)

    # current_atr_pct given explicitly (2026-09-02 ATR gate): this test starts
    # flat, and a missing ATR would now refuse the entry outright before the
    # 1% risk cap this test is actually checking ever runs.
    result = await ledger.rebalance_to_target(
        "AAPL", date(2026, 7, 16), 100.0, 1.0, reason="test", stop_loss=90.0,
        current_atr_pct=0.02,
    )

    assert result.adjustment_note is not None
    assert "1% risk cap reduced" in result.adjustment_note
    assert captured["adjustment_note"] == result.adjustment_note


async def test_rebalance_notes_when_the_atr_floor_widens_the_ais_stop_loss(monkeypatch):
    # Real incident (2026-08-20, 950160.KS/코오롱티슈진): the Portfolio
    # Manager's own decision_text said "18,500 KRW stop-loss" (price 22,000,
    # ATR% ~16.7%), but resolve_stop_loss's 2.5x ATR floor silently widened
    # the actually-persisted trades.stop_loss to 12,794.64 (42% below price
    # vs. the AI's intended 16%) with no trace anywhere the AI's reasoning
    # was recorded — unlike the risk-cap/portfolio-cap notes just above,
    # which DO surface. This is the fix: rebalance_to_target must now note
    # the divergence too.
    monkeypatch.setattr(settings, "execution_mode", ExecutionMode.MANUAL)

    async def fake_get_position(ticker):
        return ledger.Position(cash=10_000_000.0, shares=0)

    async def fake_apply_live_portfolio_caps(ticker, target_shares, price, current_shares):
        return target_shares, None

    captured = {}

    async def fake_record_trade(*args, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(ledger, "get_position", fake_get_position)
    monkeypatch.setattr(ledger, "apply_live_portfolio_caps", fake_apply_live_portfolio_caps)
    monkeypatch.setattr(ledger, "record_trade", fake_record_trade)

    result = await ledger.rebalance_to_target(
        "950160.KS", date(2026, 8, 20), 22_000.0, 0.01, reason="test",
        stop_loss=18_500.0, current_atr_pct=0.167,
    )

    expected_floor = 22_000.0 - 22_000.0 * 0.167 * settings.sizing_min_stop_atr_multiple
    assert result.stop_loss == pytest.approx(expected_floor)
    assert result.stop_loss != 18_500.0
    assert result.adjustment_note is not None
    assert "Stop-loss adjusted by code-level validation from the AI's 18,500.00" in result.adjustment_note
    assert f"{expected_floor:,.2f}" in result.adjustment_note
    assert captured["adjustment_note"] == result.adjustment_note


async def test_rebalance_does_not_note_a_stop_loss_that_already_clears_the_atr_floor(monkeypatch):
    monkeypatch.setattr(settings, "execution_mode", ExecutionMode.MANUAL)

    async def fake_get_position(ticker):
        return ledger.Position(cash=10_000_000.0, shares=0)

    async def fake_apply_live_portfolio_caps(ticker, target_shares, price, current_shares):
        return target_shares, None

    async def fake_record_trade(*args, **kwargs):
        return None

    monkeypatch.setattr(ledger, "get_position", fake_get_position)
    monkeypatch.setattr(ledger, "apply_live_portfolio_caps", fake_apply_live_portfolio_caps)
    monkeypatch.setattr(ledger, "record_trade", fake_record_trade)

    # 5% ATR, 2.5x floor -> 12.5% min distance; a 20%-below stop already
    # clears it, so resolve_stop_loss returns it unchanged and no note
    # about the stop-loss should appear.
    result = await ledger.rebalance_to_target(
        "AAPL", date(2026, 7, 16), 100.0, 0.01, reason="test", stop_loss=80.0, current_atr_pct=0.05,
    )

    assert result.stop_loss == 80.0
    assert result.adjustment_note is None or "Stop-loss adjusted" not in result.adjustment_note


# ---------------------------------------------------------------------------
# reverse_pending_trade
# ---------------------------------------------------------------------------


def _capture_record_trade(monkeypatch, captured):
    async def fake_record_trade(
        ticker, trade_date, side, shares, price, reason, stop_loss=None, decision_id=None,
        transaction_cost_override=None,
    ):
        captured.update(
            ticker=ticker, side=side, shares=shares, price=price, reason=reason,
            stop_loss=stop_loss, decision_id=decision_id,
        )

    monkeypatch.setattr(ledger, "record_trade", fake_record_trade)


async def test_reverse_pending_trade_undoes_a_buy_with_a_sell(monkeypatch):
    async def fake_get_position(ticker):
        return ledger.Position(cash=0.0, shares=10)

    monkeypatch.setattr(ledger, "get_position", fake_get_position)
    captured = {}
    _capture_record_trade(monkeypatch, captured)

    await ledger.reverse_pending_trade("AAPL", "buy", 10, 100.0, "rejected by user")

    assert captured == {
        "ticker": "AAPL", "side": "sell", "shares": 10, "price": 100.0,
        "reason": "reversal: rejected by user", "stop_loss": None, "decision_id": None,
    }


async def test_reverse_pending_trade_undoes_a_sell_with_a_buy(monkeypatch):
    async def fake_get_position(ticker):
        return ledger.Position(cash=0.0, shares=0)

    monkeypatch.setattr(ledger, "get_position", fake_get_position)
    captured = {}
    _capture_record_trade(monkeypatch, captured)

    await ledger.reverse_pending_trade("AAPL", "sell", 5, 100.0, "expired: no decision within 12.0h")

    assert captured["side"] == "buy"
    assert captured["shares"] == 5
    assert captured["reason"] == "reversal: expired: no decision within 12.0h"


async def test_reverse_pending_trade_clamps_at_zero_when_shares_already_reduced(monkeypatch):
    """Some of the phantom buy's shares were legitimately sold in the
    interim (e.g. a stop-loss exit) — reversing the full original amount
    would push shares_after negative. Must clamp to what's still on the
    books instead."""

    async def fake_get_position(ticker):
        return ledger.Position(cash=0.0, shares=3)

    monkeypatch.setattr(ledger, "get_position", fake_get_position)
    captured = {}
    _capture_record_trade(monkeypatch, captured)

    await ledger.reverse_pending_trade("AAPL", "buy", 10, 100.0, "rejected by user")

    assert captured["side"] == "sell"
    assert captured["shares"] == 3


# prior_stop_loss (2026-08-06 fix): a reversal used to always write
# stop_loss=None, so a still-open position whose speculative trade got
# reversed silently lost stop-loss monitoring forever — get_open_stop_loss
# reads only the most recent row. Confirmed live: 055550.KS (19 held) and
# AMZN (100 held) both left with an open position and stop_loss IS NULL on
# their latest row after a reversed trade.
async def test_reverse_pending_trade_carries_forward_prior_stop_when_position_stays_open(monkeypatch):
    # Undoing a phantom SELL (side="sell") restores shares via a buy — the
    # position stays open at 19, still protected by whatever stop applied
    # before the phantom sell.
    async def fake_get_position(ticker):
        return ledger.Position(cash=0.0, shares=9)

    monkeypatch.setattr(ledger, "get_position", fake_get_position)
    captured = {}
    _capture_record_trade(monkeypatch, captured)

    await ledger.reverse_pending_trade(
        "055550.KS", "sell", 10, 50_000.0, "BLOCKED: circuit-breaker state unknown", prior_stop_loss=45_000.0
    )

    assert captured["side"] == "buy"
    assert captured["stop_loss"] == 45_000.0


async def test_reverse_pending_trade_forces_stop_to_none_when_reversal_leaves_position_flat(monkeypatch):
    # Even if the caller passes a prior_stop_loss, a reversal that fully
    # flattens the position must not carry a stop forward onto a flat row.
    async def fake_get_position(ticker):
        return ledger.Position(cash=0.0, shares=10)

    monkeypatch.setattr(ledger, "get_position", fake_get_position)
    captured = {}
    _capture_record_trade(monkeypatch, captured)

    await ledger.reverse_pending_trade("AAPL", "buy", 10, 100.0, "rejected by user", prior_stop_loss=90.0)

    assert captured["side"] == "sell"
    assert captured["stop_loss"] is None


async def test_reverse_pending_trade_noop_when_already_flat(monkeypatch):
    async def fake_get_position(ticker):
        return ledger.Position(cash=0.0, shares=0)

    monkeypatch.setattr(ledger, "get_position", fake_get_position)

    async def should_not_be_called(*args, **kwargs):
        raise AssertionError("nothing left to reverse — record_trade must not be called")

    monkeypatch.setattr(ledger, "record_trade", should_not_be_called)

    await ledger.reverse_pending_trade("AAPL", "buy", 10, 100.0, "rejected by user")


# ---------------------------------------------------------------------------
# force_exit_position — cross-path stop_loss==exit_price race
# (2026-08-20/21, suspected but unconfirmed: [[HANDOFF-2026-08-20-D]] /
# [[record-reversal-bug-evidence-strengthened-2026-08-21]]). SSH access to
# the server that would show the real trades-table history around the
# 035420.KS/GOOG/AXTI incidents stayed blocked all session, so this
# reproduces the MECHANISM the memory flagged as suspicious from first
# principles rather than confirming those specific incidents.
# ---------------------------------------------------------------------------


async def test_concurrent_reader_no_longer_observes_the_about_to_be_reversed_exit_price_as_the_open_stop(
    monkeypatch,
):
    """Was `test_concurrent_reader_can_observe_...` (2026-08-21) — this is
    that SAME race, with the fix (`_is_provisional_forced_exit` /
    `_select_effective_trade_row`, see `get_open_stop_loss`'s own
    docstring) now in place, asserting the race no longer poisons the read.

    Unlike the original version, `get_open_stop_loss` itself is NOT
    monkeypatched — only the lower-level `_recent_trade_rows` (the DB
    query) is faked, backed by a mutable row list that mimics `trades`, so
    the real `_select_effective_trade_row`/`_is_provisional_forced_exit`
    logic actually runs against it. `record_trade` is faked to append to
    that same list, mirroring how the real INSERT ... RETURNING id behaves.

    force_exit_position's own reversal is correct in isolation (see the
    prior_open_stop tests above and around line 935) — it captures the true
    prior stop BEFORE record_trade, and restores exactly that if the KIS
    mirror never lands. But that correctness only covers what happens
    INSIDE force_exit_position's own per-ticker lock (`_force_exit_locks`).
    Between its own record_trade (which writes stop_loss=exit_price — the
    price the sell was attempted at) and the reverse_pending_trade call that
    later undoes it, `trades`'s latest row for this ticker genuinely has
    stop_loss==exit_price for the width of that window. Nothing serializes
    that window against a caller on a DIFFERENT lock: rebalance_to_target
    holds `market_lock(market)` (one lock per MARKET, shared across every
    ticker in it — see market-lock-hold-time-investigation), never
    `_force_exit_locks` (one lock per TICKER). A rebalance_to_target call
    for this same ticker, sizing its own possible reversal via
    `get_open_stop_loss`, can land inside that window — but now reads PAST
    the provisional row to the true prior stop instead of the poisoned
    exit_price, because that row's `mirror_disposition` is still NULL
    (not yet confirmed) and its `reason` carries STOP_LOSS_REASON_PREFIX."""
    ticker = "GOOG"
    exit_price = 338.28
    true_prior_stop = 330.0
    now = datetime(2026, 8, 20, 12, 37, tzinfo=timezone.utc)
    rows = [
        {
            "trade_date": date(2026, 8, 15), "stop_loss": true_prior_stop, "shares_after": 10,
            "reason": "AI Overweight decision", "mirror_disposition": "submitted: order 123",
            "created_at": now - timedelta(days=5), "id": 100, "price": 300.0,
        },
    ]
    next_id = [101]
    mirror_may_return = asyncio.Event()
    observed = {}

    async def fake_get_position(t):
        return ledger.Position(cash=0.0, shares=rows[-1]["shares_after"])

    async def fake_real_shares_held(t):
        return rows[-1]["shares_after"]

    async def fake_recent_trade_rows(t, limit=3):
        return list(reversed(rows[-limit:]))

    async def fake_record_trade(
        t, trade_date, side, shares, price, reason, stop_loss=None, decision_id=None,
        adjustment_note=None, base_shares=None, transaction_cost_override=None,
        mirror_disposition=None, kis_order_no=None,
    ):
        prev_shares = rows[-1]["shares_after"]
        shares_after = prev_shares + shares if side == "buy" else prev_shares - shares
        row_id = next_id[0]
        next_id[0] += 1
        rows.append({
            "trade_date": trade_date, "stop_loss": stop_loss, "shares_after": shares_after,
            "reason": reason, "mirror_disposition": mirror_disposition, "created_at": now,
            "id": row_id, "price": price,
        })
        return row_id

    async def fake_mirror_trade_to_kis(*args, **kwargs):
        # Simulates the KIS round-trip taking long enough for a concurrent
        # reader (rebalance_to_target's own prior_open_stop capture, in
        # production) to land inside the window this test is probing.
        await mirror_may_return.wait()
        return "BLOCKED: market not open (pre-market rejection)"

    async def fake_record_mirror_disposition(trade_id, disposition):
        for row in rows:
            if row["id"] == trade_id:
                row["mirror_disposition"] = disposition

    monkeypatch.setattr(ledger, "get_position", fake_get_position)
    monkeypatch.setattr(ledger, "real_shares_held", fake_real_shares_held)
    monkeypatch.setattr(ledger, "_recent_trade_rows", fake_recent_trade_rows)
    monkeypatch.setattr(ledger, "record_trade", fake_record_trade)
    monkeypatch.setattr(ledger, "mirror_trade_to_kis", fake_mirror_trade_to_kis)
    monkeypatch.setattr(ledger, "_record_mirror_disposition", fake_record_mirror_disposition)
    set_clock(FakeClock(now))
    try:
        async def concurrent_reader():
            # Waits until force_exit_position's record_trade above has
            # definitely landed (a stop_loss==exit_price row now exists),
            # then reads it through the REAL get_open_stop_loss — via a
            # completely different lock in production, so nothing here
            # blocks it from landing mid-window.
            while rows[-1]["stop_loss"] != exit_price:
                await asyncio.sleep(0)
            observed["stop_loss"] = (await ledger.get_open_stop_loss(ticker))[1]
            mirror_may_return.set()

        await asyncio.gather(
            ledger.force_exit_position(ticker, date(2026, 8, 20), exit_price, "gap down"),
            concurrent_reader(),
        )
    finally:
        set_clock(LiveClock())

    # Fixed: the concurrent reader sees PAST the not-yet-confirmed
    # forced-exit row to the true prior stop, not the exit_price it's
    # about to be reversed away from.
    assert observed["stop_loss"] == true_prior_stop
    assert observed["stop_loss"] != exit_price
    # force_exit_position's own bookkeeping is unaffected by the fix — it
    # still restores the true prior stop once the reversal lands.
    assert rows[-1]["stop_loss"] == true_prior_stop
    assert rows[-1]["shares_after"] == 10  # sell + compensating buy nets back to the original position


# ---------------------------------------------------------------------------
# _is_provisional_forced_exit / _select_effective_trade_row
# ---------------------------------------------------------------------------


def _trade_row(
    *, reason="AI Overweight decision", mirror_disposition="submitted: order 1", age_seconds=0.0,
    shares_after=10, stop_loss=90.0, row_id=1, price=100.0,
):
    now = datetime(2026, 8, 20, 12, 37, tzinfo=timezone.utc)
    return {
        "trade_date": date(2026, 8, 20), "stop_loss": stop_loss, "shares_after": shares_after,
        "reason": reason, "mirror_disposition": mirror_disposition,
        "created_at": now - timedelta(seconds=age_seconds), "id": row_id, "price": price,
    }, now


def test_effective_trade_row_skips_an_in_flight_forced_exit_row_for_the_row_below_it(monkeypatch):
    monkeypatch.setattr(ledger, "is_backtest_context", lambda: False)
    in_flight, now = _trade_row(
        reason=f"{ledger.STOP_LOSS_REASON_PREFIX}gap down", mirror_disposition=None,
        age_seconds=2.0, shares_after=0, stop_loss=338.28, row_id=2,
    )
    prior, _ = _trade_row(shares_after=10, stop_loss=330.0, row_id=1, age_seconds=999.0)

    result = ledger._select_effective_trade_row("GOOG", [in_flight, prior], now=now)

    assert result["id"] == 1
    assert result["stop_loss"] == 330.0


def test_effective_trade_row_skips_a_blocked_not_yet_reversed_forced_exit_row(monkeypatch):
    monkeypatch.setattr(ledger, "is_backtest_context", lambda: False)
    blocked, now = _trade_row(
        reason=f"{ledger.STOP_LOSS_REASON_PREFIX}gap down",
        mirror_disposition="BLOCKED: market not open (pre-market rejection)",
        age_seconds=1.0, shares_after=0, stop_loss=338.28, row_id=2,
    )
    prior, _ = _trade_row(shares_after=10, stop_loss=330.0, row_id=1, age_seconds=999.0)

    result = ledger._select_effective_trade_row("GOOG", [blocked, prior], now=now)

    assert result["id"] == 1


def test_effective_trade_row_trusts_a_confirmed_submitted_exit(monkeypatch):
    monkeypatch.setattr(ledger, "is_backtest_context", lambda: False)
    confirmed, now = _trade_row(
        reason=f"{ledger.STOP_LOSS_REASON_PREFIX}gap down", mirror_disposition="submitted: order 42",
        age_seconds=1.0, shares_after=0, stop_loss=338.28, row_id=2,
    )
    prior, _ = _trade_row(shares_after=10, stop_loss=330.0, row_id=1, age_seconds=999.0)

    # A confirmed exit means the position is genuinely flat now — the
    # ticker's real state, not a stale "still open" reading from the row
    # underneath it.
    assert ledger._select_effective_trade_row("GOOG", [confirmed, prior], now=now) is None


def test_effective_trade_row_stops_trusting_an_unconfirmed_row_past_the_window(monkeypatch):
    monkeypatch.setattr(ledger, "is_backtest_context", lambda: False)
    stale_unconfirmed, now = _trade_row(
        reason=f"{ledger.STOP_LOSS_REASON_PREFIX}gap down", mirror_disposition=None,
        age_seconds=ledger._PROVISIONAL_EXIT_WINDOW_SECONDS + 1, shares_after=0, stop_loss=338.28, row_id=2,
    )
    prior, _ = _trade_row(shares_after=10, stop_loss=330.0, row_id=1, age_seconds=999.0)

    # Past the bounded window, a NULL mirror_disposition (e.g.
    # _record_mirror_disposition itself failed) is trusted as-is rather
    # than skipped forever — otherwise a genuinely-closed position would
    # read as open indefinitely. Trusting it means treating shares_after=0
    # on THIS row as authoritative (flat), not falling through to `prior`.
    result = ledger._select_effective_trade_row("GOOG", [stale_unconfirmed, prior], now=now)
    assert result is None


def test_effective_trade_row_ignores_a_non_forced_exit_row_regardless_of_disposition(monkeypatch):
    monkeypatch.setattr(ledger, "is_backtest_context", lambda: False)
    # An ordinary AI rebalance row, not a forced exit — never provisional,
    # even with a NULL/fresh disposition.
    ordinary, now = _trade_row(
        reason="AI Overweight decision", mirror_disposition=None, age_seconds=0.5, shares_after=10,
        stop_loss=330.0, row_id=1,
    )

    result = ledger._select_effective_trade_row("GOOG", [ordinary], now=now)

    assert result["id"] == 1
    assert result["stop_loss"] == 330.0


def test_effective_trade_row_never_skips_in_a_backtest(monkeypatch):
    monkeypatch.setattr(ledger, "is_backtest_context", lambda: True)
    in_flight, now = _trade_row(
        reason=f"{ledger.STOP_LOSS_REASON_PREFIX}gap down", mirror_disposition=None,
        age_seconds=2.0, shares_after=0, stop_loss=338.28, row_id=2,
    )
    prior, _ = _trade_row(shares_after=10, stop_loss=330.0, row_id=1, age_seconds=999.0)

    # A backtest replay has no concurrent caller to race against — the
    # provisional-row guard must be a no-op there.
    result = ledger._select_effective_trade_row("GOOG", [in_flight, prior], now=now)
    assert result is None  # shares_after<=0 on the (unskipped) top row -> flat


# ---------------------------------------------------------------------------
# resolve_stop_loss
# ---------------------------------------------------------------------------


def test_resolve_stop_loss_returns_none_when_flat():
    assert ledger.resolve_stop_loss(90.0, 100.0, shares_after=0) is None


def test_resolve_stop_loss_passes_through_a_reasonable_value():
    assert ledger.resolve_stop_loss(90.0, 100.0, shares_after=10) == 90.0


def test_resolve_stop_loss_derives_default_when_missing_and_atr_available():
    # 2.5x (sizing_min_stop_atr_multiple default) * 0.05 ATR% * 100 price = 12.5
    assert ledger.resolve_stop_loss(None, 100.0, shares_after=10, current_atr_pct=0.05) == pytest.approx(87.5)


def test_resolve_stop_loss_leaves_unset_when_missing_and_no_atr():
    assert ledger.resolve_stop_loss(None, 100.0, shares_after=10) is None


def test_resolve_stop_loss_widens_a_too_tight_stop_to_the_atr_floor():
    # Floor distance = 2.5 * 0.05 * 100 = 12.5 -> floored stop = 87.5.
    # Proposed 95.0 (5.0 distance) is tighter than that floor.
    assert ledger.resolve_stop_loss(95.0, 100.0, shares_after=10, current_atr_pct=0.05) == pytest.approx(87.5)


def test_resolve_stop_loss_discards_a_hallucinated_price_anchored_stop():
    # Real incident: Market Analyst skipped its price-lookup tool call and
    # anchored the whole decision to a hallucinated price — the "stop"
    # (71,500) was ~73% below the REAL fill price (268,000), nothing like
    # a real risk-managed stop (default sizing_max_stop_distance_pct=0.50).
    # No ATR data here to isolate the ceiling from the floor logic — must
    # still discard the value (leaving it unset) rather than pass through
    # a nonsensical 73%-away "stop".
    assert ledger.resolve_stop_loss(71_500.0, 268_000.0, shares_after=10) is None


def test_resolve_stop_loss_discards_unreasonably_wide_stop_in_favor_of_atr_default():
    result = ledger.resolve_stop_loss(71_500.0, 268_000.0, shares_after=10, current_atr_pct=0.0886)
    # Floor distance = 2.5 * 0.0886 * 268000 ~= 59,367 -> default ~= 208,633,
    # nowhere near the discarded 71,500.
    assert result == pytest.approx(268_000.0 - 2.5 * 0.0886 * 268_000.0)
    assert result > 200_000.0


def test_resolve_stop_loss_ceiling_does_not_reject_a_normally_wide_but_real_stop():
    # A volatile small-cap can legitimately have a wide-ish stop — the
    # ceiling (50% of price by default) must not reject anything short of
    # genuinely broken values.
    assert ledger.resolve_stop_loss(60.0, 100.0, shares_after=10) == 60.0


def test_resolve_stop_loss_backfilled_ai_stop_still_widens_to_the_same_atr_floor():
    """Counterfactual regression for decision id=950 (950160.KS,
    2026-08-21): app.state._recover_fields_from_rendered_header now
    recovers the AI's own stop-loss (18,870) from executive_summary
    instead of leaving the `stop_loss` field None — but on this high-
    volatility ticker (~18.2% ATR), 18,870 is still tighter than the 2.5x
    ATR floor, so resolve_stop_loss widens it to the exact same value the
    pre-fix None case fell back to. The fix's value here is data fidelity/
    observability (the ledger note now reads "AI's 18,870 widened to the
    floor" instead of "no usable stop from the AI"), not a changed
    execution outcome for THIS specific decision — a lower-volatility
    ticker where the AI's own stop already clears the floor would see an
    actual execution change instead."""
    price, atr_pct = 20_370.0, 0.182
    pre_fix_default = ledger.resolve_stop_loss(None, price, shares_after=33, current_atr_pct=atr_pct)
    post_fix_result = ledger.resolve_stop_loss(18_870.0, price, shares_after=33, current_atr_pct=atr_pct)
    assert post_fix_result == pytest.approx(pre_fix_default)
    assert post_fix_result == pytest.approx(price - price * atr_pct * 2.5)


# ---------------------------------------------------------------------------
# account_mode scoping (2026-08-04) — see project memory:
# user-account-mode-db-separation-design. Every trades reader/writer must
# scope to settings.kis_account_mode so a paper-era row never gets read (or
# a real-era row overwritten) by the other mode.
# ---------------------------------------------------------------------------


class _RecordingConn:
    """Minimal asyncpg-shaped fake that just records every call for
    assertion — no result rows needed for these tests (they only check
    what was SENT, not what a query returns)."""

    def __init__(self, fetchrow_result=None, fetch_result=None):
        self.fetchrow_result = fetchrow_result
        self.fetch_result = fetch_result if fetch_result is not None else []
        self.calls: list[tuple] = []

    async def fetchrow(self, query, *args):
        self.calls.append((query, args))
        return self.fetchrow_result

    async def fetch(self, query, *args):
        self.calls.append((query, args))
        return self.fetch_result

    async def execute(self, query, *args):
        self.calls.append((query, args))

    def acquire(self):
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


@pytest.fixture
def _fake_pool(monkeypatch):
    conn = _RecordingConn()

    async def fake_ready_pool():
        return conn

    monkeypatch.setattr(ledger, "_ready_pool", fake_ready_pool)
    return conn


async def test_get_position_scopes_to_current_account_mode(monkeypatch, _fake_pool):
    monkeypatch.setattr(settings, "kis_account_mode", "real")
    await ledger.get_position("AAPL")
    query, args = _fake_pool.calls[0]
    assert "account_mode = $2" in query
    assert "user_id = $3" in query
    assert args == ("AAPL", "real", "default", False)


async def test_get_position_for_backtest_overrides_ambient_context(monkeypatch, _fake_pool):
    # reporting.render calls this AFTER backtest_scope() has already exited,
    # so the ambient is_backtest_context() would resolve to False (live) —
    # for_backtest=True must override it, matching starting_cash_for's
    # existing convention.
    monkeypatch.setattr(settings, "kis_account_mode", "real")
    await ledger.get_position("AAPL", for_backtest=True)
    _query, args = _fake_pool.calls[0]
    assert args == ("AAPL", "real", "default", True)


async def test_get_last_trade_scopes_to_current_account_mode(monkeypatch, _fake_pool):
    monkeypatch.setattr(settings, "kis_account_mode", "paper")
    await ledger.get_last_trade("AAPL")
    query, args = _fake_pool.calls[0]
    assert "account_mode = $2" in query
    assert "user_id = $3" in query
    assert args == ("AAPL", "paper", "default", False)


async def test_get_open_stop_loss_scopes_to_current_account_mode(monkeypatch, _fake_pool):
    monkeypatch.setattr(settings, "kis_account_mode", "real")
    await ledger.get_open_stop_loss("AAPL")
    query, args = _fake_pool.calls[0]
    assert "account_mode = $2" in query
    assert "user_id = $3" in query
    assert args == ("AAPL", "real", "default", False, 3)


async def test_record_trade_stamps_current_account_mode(monkeypatch, _fake_pool):
    async def fake_get_position(ticker):
        return ledger.Position(cash=1000.0, shares=0)

    monkeypatch.setattr(ledger, "get_position", fake_get_position)
    monkeypatch.setattr(settings, "kis_account_mode", "real")

    await ledger.record_trade("AAPL", date(2026, 8, 4), "buy", 1, 100.0, "test")

    query, args = _fake_pool.calls[0]
    assert "account_mode" in query
    assert args[-4] == "real"  # account_mode, then mirror_disposition, is_backtest, kis_order_no follow


# ---------------------------------------------------------------------------
# clear_backtest_positions — cross-run backtest isolation
# ---------------------------------------------------------------------------


async def test_clear_backtest_positions_deletes_only_is_backtest_rows_for_the_given_tickers(monkeypatch, _fake_pool):
    monkeypatch.setattr(settings, "kis_account_mode", "paper")
    await ledger.clear_backtest_positions(["AAPL", "MSFT"])

    query, args = _fake_pool.calls[0]
    assert "DELETE FROM trades" in query
    assert "is_backtest = true" in query
    assert args == (["AAPL", "MSFT"], "paper", "default")


async def test_clear_backtest_positions_is_a_noop_for_an_empty_ticker_list(monkeypatch, _fake_pool):
    await ledger.clear_backtest_positions([])
    assert _fake_pool.calls == []  # never touches the DB — nothing to clear


# transaction_cost_override (2026-08-06): reverse_pending_trade passes a
# negative refund so a reversed trade doesn't cost two fees (one going in,
# one coming back out) for a net position/cash change of zero shares.
async def test_record_trade_transaction_cost_override_replaces_the_usual_bps_fee(monkeypatch, _fake_pool):
    async def fake_get_position(ticker):
        return ledger.Position(cash=1000.0, shares=10)

    monkeypatch.setattr(ledger, "get_position", fake_get_position)
    monkeypatch.setattr(settings, "transaction_cost_bps", 5.0)  # would normally charge 5.0 on this trade

    await ledger.record_trade(
        "AAPL", date(2026, 8, 6), "sell", 10, 100.0, "test", transaction_cost_override=-5.0
    )

    _query, args = _fake_pool.calls[0]
    cash_after = args[5]
    # sell: cash_after = cash + notional - transaction_cost = 1000 + 1000 - (-5.0) = 2005.0
    # (the override REFUNDS a fee rather than charging one, per its own sign)
    assert cash_after == pytest.approx(2005.0)


async def test_reverse_pending_trade_leaves_cash_unchanged_after_a_full_round_trip(monkeypatch, _fake_pool):
    # The end-to-end property this exists for: a BUY that gets fully
    # reversed must leave cash back where it started, not down by 2x the fee.
    monkeypatch.setattr(settings, "transaction_cost_bps", 10.0)
    monkeypatch.setattr(settings, "transaction_cost_bps_overseas", 10.0)

    async def fake_get_position(ticker):
        return ledger.Position(cash=1000.0, shares=0)

    monkeypatch.setattr(ledger, "get_position", fake_get_position)
    await ledger.record_trade("AAPL", date(2026, 8, 6), "buy", 10, 100.0, "original")
    _query, buy_args = _fake_pool.calls[0]
    cash_after_buy = buy_args[5]
    assert cash_after_buy == pytest.approx(1000.0 - 1000.0 - 1.0)  # notional 1000, fee 1.0 (10bps)

    async def fake_get_position_after_buy(ticker):
        return ledger.Position(cash=cash_after_buy, shares=10)

    monkeypatch.setattr(ledger, "get_position", fake_get_position_after_buy)
    await ledger.reverse_pending_trade("AAPL", "buy", 10, 100.0, "BLOCKED: test")
    _query, reversal_args = _fake_pool.calls[1]
    cash_after_reversal = reversal_args[5]
    assert cash_after_reversal == pytest.approx(1000.0)  # back to the pre-trade cash exactly


# sell_tax_bps_domestic (2026-08-21): Korea's 증권거래세 went up to a flat
# 20bp on both KOSPI and KOSDAQ effective 2026-01-01, charged on SELLS only.
# record_trade previously charged the symmetric transaction_cost_bps on both
# sides, understating the real domestic round-trip cost by ~20bp.
async def test_record_trade_charges_commission_plus_tax_on_domestic_sell(monkeypatch, _fake_pool):
    async def fake_get_position(ticker):
        return ledger.Position(cash=1000.0, shares=10)

    monkeypatch.setattr(ledger, "get_position", fake_get_position)
    monkeypatch.setattr(settings, "transaction_cost_bps", 5.0)
    monkeypatch.setattr(settings, "sell_tax_bps_domestic", 20.0)

    await ledger.record_trade("005930.KS", date(2026, 8, 21), "sell", 10, 100.0, "test")

    _query, args = _fake_pool.calls[0]
    cash_after = args[5]
    # notional=1000, fee = 1000 * (5+20)/10_000 = 2.5
    # sell: cash_after = cash + notional - fee = 1000 + 1000 - 2.5 = 1997.5
    assert cash_after == pytest.approx(1997.5)


async def test_record_trade_charges_commission_only_on_domestic_buy(monkeypatch, _fake_pool):
    async def fake_get_position(ticker):
        return ledger.Position(cash=1000.0, shares=0)

    monkeypatch.setattr(ledger, "get_position", fake_get_position)
    monkeypatch.setattr(settings, "transaction_cost_bps", 5.0)
    monkeypatch.setattr(settings, "sell_tax_bps_domestic", 20.0)

    await ledger.record_trade("005930.KS", date(2026, 8, 21), "buy", 10, 100.0, "test")

    _query, args = _fake_pool.calls[0]
    cash_after = args[5]
    # notional=1000, fee = 1000 * 5/10_000 = 0.5 (no tax on a buy)
    # buy: cash_after = cash - notional - fee = 1000 - 1000 - 0.5 = -0.5
    assert cash_after == pytest.approx(-0.5)


async def test_record_trade_overseas_sell_is_not_charged_the_korean_tax(monkeypatch, _fake_pool):
    async def fake_get_position(ticker):
        return ledger.Position(cash=1000.0, shares=10)

    monkeypatch.setattr(ledger, "get_position", fake_get_position)
    monkeypatch.setattr(settings, "transaction_cost_bps", 5.0)
    monkeypatch.setattr(settings, "transaction_cost_bps_overseas", 5.0)
    monkeypatch.setattr(settings, "sell_tax_bps_domestic", 20.0)

    await ledger.record_trade("AAPL", date(2026, 8, 21), "sell", 10, 100.0, "test")

    _query, args = _fake_pool.calls[0]
    cash_after = args[5]
    # notional=1000, fee = 1000 * 5/10_000 = 0.5 — no ".KS" suffix, no tax
    assert cash_after == pytest.approx(1999.5)


async def test_record_trade_overseas_buy_is_commission_only(monkeypatch, _fake_pool):
    async def fake_get_position(ticker):
        return ledger.Position(cash=1000.0, shares=0)

    monkeypatch.setattr(ledger, "get_position", fake_get_position)
    monkeypatch.setattr(settings, "transaction_cost_bps", 5.0)
    monkeypatch.setattr(settings, "transaction_cost_bps_overseas", 5.0)
    monkeypatch.setattr(settings, "sell_tax_bps_domestic", 20.0)

    await ledger.record_trade("AAPL", date(2026, 8, 21), "buy", 10, 100.0, "test")

    _query, args = _fake_pool.calls[0]
    cash_after = args[5]
    assert cash_after == pytest.approx(-0.5)


async def test_record_trade_overseas_uses_its_own_commission_rate(monkeypatch, _fake_pool):
    """2026-09-08 fee/tax audit: before the market split, every overseas
    trade was silently charged the domestic commission rate — pins that
    ledger.record_trade now reads transaction_cost_bps_overseas for a
    non-.KS ticker, not transaction_cost_bps."""
    async def fake_get_position(ticker):
        return ledger.Position(cash=1000.0, shares=0)

    monkeypatch.setattr(ledger, "get_position", fake_get_position)
    monkeypatch.setattr(settings, "transaction_cost_bps", 5.0)
    monkeypatch.setattr(settings, "transaction_cost_bps_overseas", 25.0)
    monkeypatch.setattr(settings, "sell_tax_bps_domestic", 20.0)

    await ledger.record_trade("AAPL", date(2026, 8, 21), "buy", 10, 100.0, "test")

    _query, args = _fake_pool.calls[0]
    cash_after = args[5]
    # notional=1000, fee = 1000 * 25/10_000 = 2.5 (overseas rate, not the 5bp domestic one)
    assert cash_after == pytest.approx(-2.5)


async def test_sell_tax_bps_domestic_zero_reproduces_the_old_symmetric_behavior(monkeypatch, _fake_pool):
    # Escape hatch / backward-compat pin: with the tax knob at 0, the new
    # asymmetric model collapses back to the pre-2026-08-21 symmetric cost.
    async def fake_get_position(ticker):
        return ledger.Position(cash=1000.0, shares=10)

    monkeypatch.setattr(ledger, "get_position", fake_get_position)
    monkeypatch.setattr(settings, "transaction_cost_bps", 5.0)
    monkeypatch.setattr(settings, "sell_tax_bps_domestic", 0.0)

    await ledger.record_trade("005930.KS", date(2026, 8, 21), "sell", 10, 100.0, "test")

    _query, args = _fake_pool.calls[0]
    cash_after = args[5]
    # notional=1000, fee = 1000 * 5/10_000 = 0.5 — same as a symmetric-only buy/sell
    assert cash_after == pytest.approx(1999.5)


async def test_reverse_pending_trade_refunds_tax_when_reversing_a_domestic_sell(monkeypatch, _fake_pool):
    # A reversed domestic SELL was charged commission + the 20bp tax on the
    # way in — the refund must restore both, or cash drifts by the tax on
    # every reversed domestic sell.
    monkeypatch.setattr(settings, "transaction_cost_bps", 5.0)
    monkeypatch.setattr(settings, "sell_tax_bps_domestic", 20.0)

    async def fake_get_position(ticker):
        return ledger.Position(cash=1000.0, shares=10)

    monkeypatch.setattr(ledger, "get_position", fake_get_position)
    await ledger.record_trade("005930.KS", date(2026, 8, 21), "sell", 10, 100.0, "original")
    _query, sell_args = _fake_pool.calls[0]
    cash_after_sell = sell_args[5]
    assert cash_after_sell == pytest.approx(1000.0 + 1000.0 - 2.5)  # fee = 1000 * 25/10_000

    async def fake_get_position_after_sell(ticker):
        return ledger.Position(cash=cash_after_sell, shares=0)

    monkeypatch.setattr(ledger, "get_position", fake_get_position_after_sell)
    await ledger.reverse_pending_trade("005930.KS", "sell", 10, 100.0, "BLOCKED: test")
    _query, reversal_args = _fake_pool.calls[1]
    cash_after_reversal = reversal_args[5]
    assert cash_after_reversal == pytest.approx(1000.0)  # back to the pre-trade cash exactly


async def test_reverse_pending_trade_refunds_commission_only_for_a_domestic_buy_and_charges_no_tax(
    monkeypatch, _fake_pool
):
    # The reversal leg of an undone BUY is itself a "sell" (reversed_side),
    # but it must NOT be charged the 20bp domestic sell tax — the refund
    # (and the reversal's own zero-cost row) must use the ORIGINAL side
    # ("buy"), which was never taxed.
    monkeypatch.setattr(settings, "transaction_cost_bps", 5.0)
    monkeypatch.setattr(settings, "sell_tax_bps_domestic", 20.0)

    async def fake_get_position(ticker):
        return ledger.Position(cash=1000.0, shares=0)

    monkeypatch.setattr(ledger, "get_position", fake_get_position)
    await ledger.record_trade("005930.KS", date(2026, 8, 21), "buy", 10, 100.0, "original")
    _query, buy_args = _fake_pool.calls[0]
    cash_after_buy = buy_args[5]
    assert cash_after_buy == pytest.approx(1000.0 - 1000.0 - 0.5)  # fee = 1000 * 5/10_000, no tax

    async def fake_get_position_after_buy(ticker):
        return ledger.Position(cash=cash_after_buy, shares=10)

    monkeypatch.setattr(ledger, "get_position", fake_get_position_after_buy)
    await ledger.reverse_pending_trade("005930.KS", "buy", 10, 100.0, "BLOCKED: test")
    _query, reversal_args = _fake_pool.calls[1]
    cash_after_reversal = reversal_args[5]
    # Back to the pre-trade cash exactly — if the reversal wrongly taxed
    # this "sell"-shaped leg, this would be short by the 20bp tax.
    assert cash_after_reversal == pytest.approx(1000.0)


async def test_get_trade_history_scopes_to_current_account_mode_for_all_tickers(monkeypatch, _fake_pool):
    monkeypatch.setattr(settings, "kis_account_mode", "paper")
    await ledger.get_trade_history()
    query, args = _fake_pool.calls[0]
    assert "account_mode = $1" in query
    assert args[0] == "paper"


async def test_get_trade_history_scopes_to_current_account_mode_for_one_ticker(monkeypatch, _fake_pool):
    monkeypatch.setattr(settings, "kis_account_mode", "real")
    await ledger.get_trade_history("AAPL")
    query, args = _fake_pool.calls[0]
    assert "account_mode = $2" in query
    assert args[:2] == ("AAPL", "real")


async def test_paper_and_real_do_not_see_each_others_position(monkeypatch, _fake_pool):
    # The concrete bug this closes: a paper-era row showing a ticker "flat"
    # must not make a real-mode caller think the same — real-mode's query
    # only ever matches real-mode rows, so a paper-only ticker looks
    # unopened (fresh starting position) from the real side, not "flat".
    async def fake_starting_cash_for(ticker, *, for_backtest=None):
        return 100_000.0

    _fake_pool.fetchrow_result = None  # no row for this (ticker, account_mode) pair
    monkeypatch.setattr(ledger, "starting_cash_for", fake_starting_cash_for)
    monkeypatch.setattr(settings, "kis_account_mode", "real")
    position = await ledger.get_position("AAPL")
    assert position.shares == 0
    _, args = _fake_pool.calls[0]
    assert args[1] == "real"


# ---------------------------------------------------------------------------
# check_and_apply_trailing_stop — intraday (5-minute) ordering adjudication,
# the 2026-08-27 P0 fix. The pre-fix walk ratcheted high_water from a day's
# HIGH and then tested that SAME day's LOW against the resulting floor, with
# no evidence the high preceded the low; on 2026-08-27 that force-liquidated
# 047040.KS at +11% (day low 18,760 at 09:00 KST, day high 21,950 at 09:30,
# floor 19,729.24, lowest post-09:30 low 19,800). See
# check_and_apply_trailing_stop's own docstring for the full incident.
#
# Shared fixture arithmetic for every test below:
#   entry_price 18_940, pm stop 15_065.06, trailing_pct 0.101173
#   activation  = 18_940 * 1.04        = 19_697.60
#   floor(hw)   = max(pm, hw * 0.898827) once hw >= activation, else pm
#   floor(21_950) = 21_950 * 0.898827  = 19_729.25  (the incident's floor)
#   floor(19_000) -> hw below activation -> pm 15_065.06
# ---------------------------------------------------------------------------

_ENTRY = 18_940.0
_PM = 15_065.06
_PCT = 0.101173
_KEEP = 1 - _PCT  # 0.898827


@pytest.fixture
def _fixed_activation(monkeypatch):
    """Pin trailing_stop_activation_pct so these fixtures' arithmetic is not
    at the mercy of a settings/risk_rules change (0.04 is also the current
    default, so this changes nothing today — it just makes the numbers in
    the comments above provably the ones under test)."""
    monkeypatch.setattr(
        ledger.trailing_stop.risk_rules, "rule",
        lambda key: {"trailing_stop_activation_pct": 0.04}[key],
    )


def _bar(hhmm: str, high: float, low: float, day=date(2026, 8, 27)) -> ledger.IntradayBar:
    start = datetime.combine(day, dtime(int(hhmm[:2]), int(hhmm[2:])), tzinfo=timezone.utc)
    return ledger.IntradayBar(start=start, end=start + timedelta(minutes=5), high=high, low=low)


async def test_trailing_stop_intraday_bars_clear_the_047040_shaped_false_breach(
    monkeypatch, _fixed_activation
):
    """THE regression test. Real 5-minute ordering for 047040.KS on
    2026-08-27 (UTC == KST-9): 00:00 bar is the day's low 18,760; 00:30 bar
    is the day's high 21,950; 01:50 bar's low 19,800 is the lowest price at
    any point after the high — and 19,800 > floor(21,950) = 19,729.25. The
    floor was never breached once it existed, so this must NOT exit."""
    captured = _stub_flat_exit(monkeypatch)
    highs = {date(2026, 8, 26): 19_100.0, date(2026, 8, 27): 21_950.0}
    lows = {date(2026, 8, 27): 18_760.0}  # entry day's low excluded, as the live caller does
    bars = {date(2026, 8, 27): [
        _bar("0000", 19_000.0, 18_760.0),
        _bar("0030", 21_950.0, 21_000.0),
        _bar("0150", 21_100.0, 19_800.0),
    ]}
    outcome, high_water = await ledger.check_and_apply_trailing_stop(
        "047040.KS", highs, lows, pm_stop_loss=_PM, entry_price=_ENTRY,
        high_water=_ENTRY, trailing_pct=_PCT, reason="test",
        intraday_bars=bars,
        entry_time=datetime(2026, 8, 26, 3, 36, tzinfo=timezone.utc),
        live_high_water=21_850.0,
        live_high_water_day=date(2026, 8, 27),
    )
    assert outcome is None
    assert captured == {}  # force_exit_position never called
    assert high_water == 21_950.0  # the day's daily high is still folded in


async def test_trailing_stop_without_intraday_bars_still_reproduces_the_047040_false_breach(
    monkeypatch, _fixed_activation
):
    """The same fixtures with no intraday evidence reproduce the incident
    exactly — documenting inside the suite both what the false exit was and
    that the conservative fallback branch is genuinely unchanged."""
    captured = _stub_flat_exit(monkeypatch)
    highs = {date(2026, 8, 26): 19_100.0, date(2026, 8, 27): 21_950.0}
    lows = {date(2026, 8, 27): 18_760.0}
    outcome, high_water = await ledger.check_and_apply_trailing_stop(
        "047040.KS", highs, lows, pm_stop_loss=_PM, entry_price=_ENTRY,
        high_water=_ENTRY, trailing_pct=_PCT, reason="test",
    )
    assert outcome.breach_date == date(2026, 8, 27)
    assert captured["exit_price"] == pytest.approx(21_950.0 * _KEEP)  # 19,729.25
    assert high_water == 21_950.0


async def test_trailing_stop_intraday_bars_still_exit_when_the_crash_follows_the_high(
    monkeypatch, _fixed_activation
):
    """The must-not-miss case, and the reason "lag the high by one session"
    was rejected: a rally to a new high FOLLOWED by a real crash through the
    floor that new high produced. 00:30 ratchets hw to 21,950 (floor
    19,729.25); 01:50's low of 19,000 is under it. Bar-by-bar adjudication
    is strictly ordered, so it catches exactly this."""
    captured = _stub_flat_exit(monkeypatch)
    day = date(2026, 8, 27)
    highs = {day: 21_950.0}
    lows = {day: 19_000.0}
    bars = {day: [_bar("0030", 21_950.0, 21_000.0), _bar("0150", 21_100.0, 19_000.0)]}
    outcome, _ = await ledger.check_and_apply_trailing_stop(
        "047040.KS", highs, lows, pm_stop_loss=_PM, entry_price=_ENTRY,
        high_water=19_100.0, trailing_pct=_PCT, reason="test", intraday_bars=bars,
    )
    assert outcome.breach_date == day
    assert captured["exit_price"] == pytest.approx(21_950.0 * _KEEP, rel=1e-6)
    assert "intraday bar" in captured["reason"]


async def test_trailing_stop_intraday_path_exits_on_a_low_below_the_floor_already_in_force(
    monkeypatch, _fixed_activation
):
    """Step (1), the ordering-independent pre-test — and the whole reason the
    0.5% reconciliation tolerance is safe. Entering the day hw is 25,000, so
    the floor already in force is 25,000 * 0.898827 = 22,470.68. The daily
    bar's low is 22,400 (under it — a breach under EVERY possible intraday
    ordering), but the bar set's lowest low is 22,500, which is inside the
    tolerance band (22,500 <= 22,400 * 1.005 = 22,512) and above every floor
    the bar scan itself would build. Bar-by-bar adjudication alone would
    therefore clear this day; step (1) must not."""
    captured = _stub_flat_exit(monkeypatch)
    day = date(2026, 8, 27)
    highs = {day: 25_000.0}
    lows = {day: 22_400.0}
    # Bar highs stay under 25,000 so the running floor never moves off
    # 22,470.68, and both bar lows sit above it.
    bars = {day: [_bar("0000", 24_900.0, 22_500.0), _bar("0030", 24_950.0, 22_600.0)]}
    outcome, _ = await ledger.check_and_apply_trailing_stop(
        "047040.KS", highs, lows, pm_stop_loss=_PM, entry_price=_ENTRY,
        high_water=25_000.0, trailing_pct=_PCT, reason="test", intraday_bars=bars,
    )
    assert outcome.breach_date == day
    assert captured["exit_price"] == pytest.approx(25_000.0 * _KEEP)  # 22,470.68
    assert "floor already in force at session start" in captured["reason"]


async def test_trailing_stop_ignores_intraday_bars_that_never_reach_the_days_low(
    monkeypatch, _fixed_activation
):
    """A bar set whose lowest low never reaches the day's real low has a hole
    in it exactly where a breach could hide (19,500 vs a daily low of 18,000
    — 8% short, far past the 0.5% tolerance), so the day falls back to the
    conservative daily rule rather than being adjudicated off partial data."""
    captured = _stub_flat_exit(monkeypatch)
    day = date(2026, 8, 27)
    highs = {day: 21_950.0}
    lows = {day: 18_000.0}
    bars = {day: [_bar("0000", 21_950.0, 19_500.0)]}
    outcome, _ = await ledger.check_and_apply_trailing_stop(
        "047040.KS", highs, lows, pm_stop_loss=_PM, entry_price=_ENTRY,
        high_water=_ENTRY, trailing_pct=_PCT, reason="test", intraday_bars=bars,
    )
    assert outcome.breach_date == day
    assert captured["exit_price"] == pytest.approx(21_950.0 * _KEEP)
    assert "intraday bar" not in captured["reason"]
    assert "session start" not in captured["reason"]


async def test_trailing_stop_ignores_intraday_bars_that_never_reach_the_days_high(
    monkeypatch, _fixed_activation
):
    """Mirror of the low-side check: a bar set whose highest high falls more
    than 0.5% short of the day's daily high (20,000 vs 21,950) is describing
    a different — truncated — session, so the coarse rule is the safe
    answer even though its low side reconciles fine."""
    captured = _stub_flat_exit(monkeypatch)
    day = date(2026, 8, 27)
    highs = {day: 21_950.0}
    lows = {day: 18_000.0}
    bars = {day: [_bar("0000", 20_000.0, 18_000.0)]}
    outcome, _ = await ledger.check_and_apply_trailing_stop(
        "047040.KS", highs, lows, pm_stop_loss=_PM, entry_price=_ENTRY,
        high_water=_ENTRY, trailing_pct=_PCT, reason="test", intraday_bars=bars,
    )
    assert outcome.breach_date == day
    assert captured["exit_price"] == pytest.approx(21_950.0 * _KEEP)
    assert "intraday bar" not in captured["reason"]
    assert "session start" not in captured["reason"]


async def test_trailing_stop_intraday_path_carries_the_daily_high_forward(
    monkeypatch, _fixed_activation
):
    """An understated high-water understates every LATER day's floor, which
    is the missed-breach direction — so the bar path must still fold the
    day's DAILY high in at the end of the scan.

    Day 1's bars top out at 21,900 (reconciles: 21,900 >= 21,950 * 0.995 =
    21,840.25) while its daily bar says 21,950. Day 2 has no bars and a low
    of 19,700, which sits between floor(21,900) = 19,684.31 and
    floor(21,950) = 19,729.25 — so it breaches if and only if the daily high
    was folded in."""
    captured = _stub_flat_exit(monkeypatch)
    d1, d2 = date(2026, 8, 26), date(2026, 8, 27)
    highs = {d1: 21_950.0, d2: 19_800.0}
    lows = {d2: 19_700.0}
    bars = {d1: [_bar("0000", 21_900.0, 21_500.0, day=d1)]}
    outcome, _ = await ledger.check_and_apply_trailing_stop(
        "047040.KS", highs, lows, pm_stop_loss=_PM, entry_price=_ENTRY,
        high_water=_ENTRY, trailing_pct=_PCT, reason="test", intraday_bars=bars,
    )
    assert outcome.breach_date == d2
    assert captured["exit_price"] == pytest.approx(21_950.0 * _KEEP)  # 19,729.25, not 19,684.31


async def test_trailing_stop_live_high_water_folds_into_the_last_bar_only(
    monkeypatch, _fixed_activation
):
    """A live tick high-water certainly happened at or before now, so it
    belongs at the END of that day's bar scan — where it can tighten a later
    floor but not retroactively tighten an earlier bar's. Here the tick HWM
    (21,950) is far above anything the 5-minute bars show (19,100): folded
    into the LAST bar it makes that bar's own low of 18,760 a real breach."""
    captured = _stub_flat_exit(monkeypatch)
    day = date(2026, 8, 27)
    highs = {day: 19_100.0}
    lows = {day: 18_760.0}
    bars = {day: [_bar("0000", 19_000.0, 18_900.0), _bar("0030", 19_100.0, 18_760.0)]}
    outcome, _ = await ledger.check_and_apply_trailing_stop(
        "047040.KS", highs, lows, pm_stop_loss=_PM, entry_price=_ENTRY,
        high_water=_ENTRY, trailing_pct=_PCT, reason="test", intraday_bars=bars,
        live_high_water=21_950.0, live_high_water_day=day,
    )
    assert outcome.breach_date == day
    assert captured["exit_price"] == pytest.approx(21_950.0 * _KEEP)


async def test_trailing_stop_live_high_water_is_confined_to_its_own_day(
    monkeypatch, _fixed_activation
):
    """Companion to the test above: the identical fixtures with the live
    high-water tagged to a DIFFERENT day must not touch this day's floors at
    all (hw stays 19,100, below activation, so the floor is the pm stop and
    18,760 clears it comfortably)."""
    captured = _stub_flat_exit(monkeypatch)
    day = date(2026, 8, 27)
    highs = {day: 19_100.0}
    lows = {day: 18_760.0}
    bars = {day: [_bar("0000", 19_000.0, 18_900.0), _bar("0030", 19_100.0, 18_760.0)]}
    outcome, high_water = await ledger.check_and_apply_trailing_stop(
        "047040.KS", highs, lows, pm_stop_loss=_PM, entry_price=_ENTRY,
        high_water=_ENTRY, trailing_pct=_PCT, reason="test", intraday_bars=bars,
        live_high_water=21_950.0, live_high_water_day=day + timedelta(days=1),
    )
    assert outcome is None
    assert captured == {}
    assert high_water == 19_100.0


async def test_trailing_stop_adjudicated_days_ratchet_highs_but_are_never_low_tested(
    monkeypatch, _fixed_activation
):
    """The `adjudicated_through` watermark. Day 1 WOULD breach under the
    conservative rule (low 18,760 vs floor(21,950) = 19,729.25) but was
    already settled by an earlier pass, so its low is never re-tested — that
    is what stops one yfinance hiccup from re-manufacturing an
    already-cleared false liquidation a day late.

    Its HIGH is still ratcheted, which is mandatory: day 2's low of 19,700
    breaches floor(21,950) = 19,729.25 but not floor(19,800) = 17,796.78, so
    it can only exit if day 1's high made it into the high-water mark."""
    captured = _stub_flat_exit(monkeypatch)
    d1, d2 = date(2026, 8, 26), date(2026, 8, 27)
    highs = {d1: 21_950.0, d2: 19_800.0}
    lows = {d1: 18_760.0, d2: 19_700.0}
    outcome, _ = await ledger.check_and_apply_trailing_stop(
        "047040.KS", highs, lows, pm_stop_loss=_PM, entry_price=_ENTRY,
        high_water=_ENTRY, trailing_pct=_PCT, reason="test",
        adjudicated_through=d1,
    )
    assert outcome.breach_date == d2  # NOT d1 — d1 was settled
    assert captured["exit_price"] == pytest.approx(21_950.0 * _KEEP)


async def test_trailing_stop_entry_day_bars_before_entry_do_not_inflate_the_high_water(
    monkeypatch, _fixed_activation
):
    """The second, independent false-liquidation generator: a PRE-entry spike
    on the entry day was never this position's high-water
    (`update_high_water_mark` baselines at max(entry_price, current_price)
    from entry onward), so folding the entry day's 25,000 daily high in
    over-tightens every later floor.

    Entry at 03:36Z. The 00:00 bar (high 25,000) is dropped; only the 04:00
    bar (high 19,000) counts, and the entry day's daily high is not folded.
    Day 2's low of 20,000 then clears both the pm stop and floor(19,000)
    (= pm, since 19,000 is below the 19,697.60 activation)."""
    captured = _stub_flat_exit(monkeypatch)
    entry_day, next_day = date(2026, 8, 26), date(2026, 8, 27)
    entry_time = datetime(2026, 8, 26, 3, 36, tzinfo=timezone.utc)
    highs = {entry_day: 25_000.0}
    lows = {next_day: 20_000.0}
    bars = {entry_day: [
        _bar("0000", 25_000.0, 24_000.0, day=entry_day),
        _bar("0400", 19_000.0, 18_900.0, day=entry_day),
    ]}
    outcome, high_water = await ledger.check_and_apply_trailing_stop(
        "047040.KS", highs, lows, pm_stop_loss=_PM, entry_price=_ENTRY,
        high_water=_ENTRY, trailing_pct=_PCT, reason="test",
        intraday_bars=bars, entry_time=entry_time,
    )
    assert outcome is None
    assert captured == {}
    assert high_water == 19_000.0  # NOT 25,000


async def test_trailing_stop_entry_day_without_entry_time_still_inflates_the_high_water(
    monkeypatch, _fixed_activation
):
    """Sibling of the test above, pinning the backtest path: with no
    `entry_time` there is no entry day, so every bar counts and the 25,000
    pre-entry spike DOES inflate the high-water — floor(25,000) = 22,470.68,
    which the 04:00 bar's own low of 18,900 is under. The replay has no
    entry timestamp to pass and its numbers must not move."""
    captured = _stub_flat_exit(monkeypatch)
    entry_day, next_day = date(2026, 8, 26), date(2026, 8, 27)
    highs = {entry_day: 25_000.0}
    lows = {next_day: 20_000.0}
    bars = {entry_day: [
        _bar("0000", 25_000.0, 24_000.0, day=entry_day),
        _bar("0400", 19_000.0, 18_900.0, day=entry_day),
    ]}
    outcome, _ = await ledger.check_and_apply_trailing_stop(
        "047040.KS", highs, lows, pm_stop_loss=_PM, entry_price=_ENTRY,
        high_water=_ENTRY, trailing_pct=_PCT, reason="test", intraday_bars=bars,
    )
    assert outcome is not None
    assert captured["exit_price"] == pytest.approx(25_000.0 * _KEEP)


# The pre-fix algorithm, verbatim, as an expectation table: for each case the
# (breach_date, exit_price, final_high_water) the 2026-08-26 code produced.
# Called with NO keyword-only arguments, exactly as app.backtest.harness does.
_PRE_FIX_WALK_CASES = [
    # never breached — pure ratchet
    ({date(2026, 7, 17): 110.0, date(2026, 7, 18): 120.0},
     {date(2026, 7, 17): 105.0, date(2026, 7, 18): 115.0},
     100.0, 80.0, 0.1, None, None, 120.0),
    # pullback the day AFTER the run-up: floor 150*0.9 = 135
    ({date(2026, 7, 17): 150.0, date(2026, 7, 18): 132.0},
     {date(2026, 7, 17): 148.0, date(2026, 7, 18): 130.0},
     100.0, 80.0, 0.1, date(2026, 7, 18), 135.0, 150.0),
    # SAME day high-then-low — the branch the 2026-08-27 fix leaves standing
    # whenever no intraday evidence exists: floor 200*0.9 = 180
    ({date(2026, 7, 17): 200.0}, {date(2026, 7, 17): 175.0},
     100.0, 80.0, 0.1, date(2026, 7, 17), 180.0, 200.0),
    # trailing floor looser than the pm stop -> max() keeps the pm stop
    ({date(2026, 7, 17): 105.0}, {date(2026, 7, 17): 96.0},
     100.0, 95.0, 0.5, None, None, 105.0),
    # high-water below activation (100*1.04 = 104) -> the floor IS the pm stop
    ({date(2026, 7, 17): 101.0}, {date(2026, 7, 17): 79.0},
     100.0, 80.0, 0.1, date(2026, 7, 17), 80.0, 101.0),
    # a high-only day, then a breach two days later off the retained high:
    # floor 160*0.9 = 144, and day 7/18's low of 140 is under it
    ({date(2026, 7, 16): 130.0, date(2026, 7, 17): 160.0, date(2026, 7, 18): 150.0},
     {date(2026, 7, 17): 150.0, date(2026, 7, 18): 140.0},
     100.0, 80.0, 0.1, date(2026, 7, 18), 144.0, 160.0),
]


@pytest.mark.parametrize(
    "highs,lows,seed_hw,pm,pct,want_breach,want_price,want_hw", _PRE_FIX_WALK_CASES
)
async def test_trailing_stop_defaults_are_identical_to_the_pre_fix_walk(
    monkeypatch, _fixed_activation, highs, lows, seed_hw, pm, pct,
    want_breach, want_price, want_hw,
):
    """The explicit backtest-identity guard for the 2026-08-27 fix. Every new
    capability is behind a keyword-only parameter defaulting to None; with
    them all absent the iteration set, the floors, the breach day, the exit
    price and the final high-water must all be exactly what the pre-fix
    loop produced. app.backtest.harness passes none of them, and its results
    feed the 4-stage gate criteria — see check_and_apply_trailing_stop's own
    docstring for why silently moving them is unacceptable."""
    captured = _stub_flat_exit(monkeypatch)
    outcome, high_water = await ledger.check_and_apply_trailing_stop(
        "AAPL", highs, lows, pm_stop_loss=pm, entry_price=100.0,
        high_water=seed_hw, trailing_pct=pct, reason="test",
    )
    if want_breach is None:
        assert outcome is None
        assert captured == {}
    else:
        assert outcome.breach_date == want_breach
        assert captured["exit_price"] == pytest.approx(want_price)
        # the pre-fix detail string carried no basis suffix
        assert "intraday bar" not in captured["reason"]
        assert "session start" not in captured["reason"]
    assert high_water == pytest.approx(want_hw)


# ---------------------------------------------------------------------------
# check_stops_for_ticker — the intraday-bar fetch and the adjudicated_through
# watermark wiring (2026-08-27).
# ---------------------------------------------------------------------------


def _stub_backstop_wiring(monkeypatch, *, trade_date, daily_lows, daily_highs,
                          adjudicated_through, trailing_state=(150.0, 0.1),
                          walk_result=None):
    """Everything check_stops_for_ticker touches, faked. Returns
    (captured_kwargs, fetched_days, recorded) for the assertions."""
    entry_time = datetime(2026, 7, 20, 9, 0, tzinfo=timezone.utc)
    captured: dict = {}
    fetched: list = []
    recorded: list = []

    async def fake_get_open_stop_loss(_ticker):
        return trade_date, 90.0

    async def fake_entry_time(_ticker):
        return entry_time, 100.0, 90.0

    async def fake_get_current_state(_ticker, _et):
        return trailing_state

    async def fake_daily_lows(_ticker, _start, _end):
        return dict(daily_lows)

    async def fake_daily_highs(_ticker, _start, _end):
        return dict(daily_highs)

    async def fake_intraday_bars(_ticker, day):
        fetched.append(day)
        return [_bar("0000", 100.0, 99.0, day=day)]

    async def fake_get_adjudicated_through(_ticker, _et, _pm):
        captured["adjudicated_read_pm"] = _pm
        return adjudicated_through

    async def fake_record(_ticker, et, through, pm):
        recorded.append({"entry_time": et, "through": through, "pm_stop_loss": pm})

    async def fake_walk(ticker, highs, lows, pm_stop_loss, entry_price, high_water,
                        trailing_pct, reason, *, record_as_of=None, intraday_bars=None,
                        adjudicated_through=None, entry_time=None, live_high_water=None,
                        live_high_water_day=None):
        captured.update(
            daily_highs=dict(highs), daily_lows=dict(lows), intraday_bars=intraday_bars,
            adjudicated_through=adjudicated_through, entry_time=entry_time,
            live_high_water=live_high_water, live_high_water_day=live_high_water_day,
        )
        return walk_result, high_water

    monkeypatch.setattr(ledger, "get_open_stop_loss", fake_get_open_stop_loss)
    monkeypatch.setattr(ledger, "get_open_stop_loss_entry_time", fake_entry_time)
    monkeypatch.setattr(ledger.trailing_stop, "get_current_state", fake_get_current_state)
    monkeypatch.setattr(ledger, "get_daily_low_series", fake_daily_lows)
    monkeypatch.setattr(ledger, "get_daily_high_series", fake_daily_highs)
    monkeypatch.setattr(ledger, "get_intraday_bars", fake_intraday_bars)
    monkeypatch.setattr(ledger.trailing_stop, "get_adjudicated_through", fake_get_adjudicated_through)
    monkeypatch.setattr(ledger.trailing_stop, "record_adjudicated_through", fake_record)
    monkeypatch.setattr(ledger, "check_and_apply_trailing_stop", fake_walk)
    return captured, fetched, recorded, entry_time


async def test_check_stops_for_ticker_passes_intraday_bars_and_the_watermark_to_the_walk(monkeypatch):
    set_clock(FakeClock(datetime(2026, 7, 25, 8, 0, tzinfo=ZoneInfo("Asia/Seoul"))))
    try:
        captured, fetched, _recorded, entry_time = _stub_backstop_wiring(
            monkeypatch,
            trade_date=date(2026, 7, 20),
            daily_lows={date(2026, 7, 22): 95.0, date(2026, 7, 23): 96.0, date(2026, 7, 24): 97.0},
            daily_highs={date(2026, 7, 22): 145.0, date(2026, 7, 23): 146.0, date(2026, 7, 24): 147.0},
            adjudicated_through=date(2026, 7, 23),
        )
        await ledger.check_stops_for_ticker("AAPL")
        # Settled days (<= 2026-07-23) are never low-tested, so fetching
        # their bars would be pure vendor load.
        assert fetched == [date(2026, 7, 24), date(2026, 7, 25)]
        assert captured["adjudicated_through"] == date(2026, 7, 23)
        assert set(captured["intraday_bars"]) == {date(2026, 7, 24), date(2026, 7, 25)}
        assert captured["entry_time"] == entry_time
        assert captured["adjudicated_read_pm"] == 90.0
    finally:
        set_clock(LiveClock())


async def test_check_stops_for_ticker_records_the_watermark_only_for_closed_days_after_a_clean_pass(
    monkeypatch
):
    """Never today: marking a still-open session "adjudicated" at 09:15
    would blind the backstop to a 14:00 crash for the rest of the day."""
    set_clock(FakeClock(datetime(2026, 7, 25, 8, 0, tzinfo=ZoneInfo("Asia/Seoul"))))
    try:
        _captured, _fetched, recorded, entry_time = _stub_backstop_wiring(
            monkeypatch,
            trade_date=date(2026, 7, 20),
            daily_lows={date(2026, 7, 24): 97.0, date(2026, 7, 25): 98.0},
            daily_highs={},
            adjudicated_through=None,
            walk_result=None,
        )
        await ledger.check_stops_for_ticker("AAPL")
        assert recorded == [
            {"entry_time": entry_time, "through": date(2026, 7, 24), "pm_stop_loss": 90.0}
        ]
    finally:
        set_clock(LiveClock())


async def test_check_stops_for_ticker_does_not_record_a_watermark_when_a_breach_was_found(monkeypatch):
    """A breach whose exit FAILED must keep re-firing on every pass until it
    lands — which is exactly what not writing a watermark preserves."""
    set_clock(FakeClock(datetime(2026, 7, 25, 8, 0, tzinfo=ZoneInfo("Asia/Seoul"))))
    try:
        breach = ledger.ForcedExitOutcome(
            status="failed", shares=1, disposition="FAILED: [X permanent] nope",
            breach_date=date(2026, 7, 24),
        )
        _captured, _fetched, recorded, _entry_time = _stub_backstop_wiring(
            monkeypatch,
            trade_date=date(2026, 7, 20),
            daily_lows={date(2026, 7, 24): 97.0},
            daily_highs={},
            adjudicated_through=None,
            walk_result=breach,
        )
        await ledger.check_stops_for_ticker("AAPL")
        assert recorded == []
    finally:
        set_clock(LiveClock())


async def test_check_stops_for_ticker_skips_intraday_fetch_when_the_kill_switch_is_off(monkeypatch):
    set_clock(FakeClock(datetime(2026, 7, 25, 8, 0, tzinfo=ZoneInfo("Asia/Seoul"))))
    try:
        monkeypatch.setattr(
            ledger.settings, "stop_loss_backstop_intraday_adjudication", False
        )
        captured, fetched, _recorded, _entry_time = _stub_backstop_wiring(
            monkeypatch,
            trade_date=date(2026, 7, 20),
            daily_lows={date(2026, 7, 24): 97.0},
            daily_highs={date(2026, 7, 24): 147.0},
            adjudicated_through=None,
        )
        await ledger.check_stops_for_ticker("AAPL")
        assert fetched == []  # no vendor calls at all
        assert captured["intraday_bars"] is None  # -> the walk runs the pre-fix rule
    finally:
        set_clock(LiveClock())


async def test_check_stops_for_ticker_caps_the_intraday_fetch_at_the_newest_days(monkeypatch):
    """A position whose last decision is months old would otherwise try to
    fetch a 100-day walk; yfinance's 5-minute retention is ~60 days anyway,
    so the older tail was always going to be conservative-only."""
    set_clock(FakeClock(datetime(2026, 7, 25, 8, 0, tzinfo=ZoneInfo("Asia/Seoul"))))
    try:
        days = [date(2026, 7, 1) + timedelta(days=i) for i in range(25)]  # 7/1 .. 7/25
        captured, fetched, _recorded, _entry_time = _stub_backstop_wiring(
            monkeypatch,
            trade_date=date(2026, 6, 30),
            daily_lows={d: 95.0 for d in days},
            daily_highs={},
            adjudicated_through=None,
        )
        await ledger.check_stops_for_ticker("AAPL")
        assert len(fetched) == ledger._MAX_INTRADAY_ADJUDICATION_DAYS
        assert fetched == days[-ledger._MAX_INTRADAY_ADJUDICATION_DAYS:]
        assert fetched[-1] == date(2026, 7, 25)  # today included
        assert captured["intraday_bars"] is not None
    finally:
        set_clock(LiveClock())


async def test_check_stops_for_ticker_plain_stop_path_touches_no_intraday_or_watermark_state(
    monkeypatch
):
    """The non-trailing branch tests each day's low against a STATIC pm stop
    that no day's high can move, so there is no high-before-low ordering to
    get wrong — and a real static-stop breach on a past day SHOULD keep
    re-firing until the exit lands, so it gets no watermark either."""
    set_clock(FakeClock(datetime(2026, 7, 25, 8, 0, tzinfo=ZoneInfo("Asia/Seoul"))))
    try:
        calls = []

        async def fake_get_open_stop_loss(_ticker):
            return date(2026, 7, 20), 90.0

        async def fake_entry_time(_ticker):
            return datetime(2026, 7, 20, 9, 0, tzinfo=timezone.utc), 100.0, 90.0

        async def fake_get_current_state(_ticker, _et):
            return None

        async def fake_daily_lows(_ticker, _s, _e):
            return {date(2026, 7, 24): 95.0}

        async def fake_plain(ticker, lows, stop_loss, reason, *, record_as_of=None):
            return None

        def _forbid(name):
            async def _inner(*_a, **_k):
                calls.append(name)
                return None
            return _inner

        monkeypatch.setattr(ledger, "get_open_stop_loss", fake_get_open_stop_loss)
        monkeypatch.setattr(ledger, "get_open_stop_loss_entry_time", fake_entry_time)
        monkeypatch.setattr(ledger.trailing_stop, "get_current_state", fake_get_current_state)
        monkeypatch.setattr(ledger, "get_daily_low_series", fake_daily_lows)
        monkeypatch.setattr(ledger, "check_and_apply_stop_loss", fake_plain)
        monkeypatch.setattr(ledger, "get_intraday_bars", _forbid("get_intraday_bars"))
        monkeypatch.setattr(
            ledger.trailing_stop, "get_adjudicated_through", _forbid("get_adjudicated_through")
        )
        monkeypatch.setattr(
            ledger.trailing_stop, "record_adjudicated_through", _forbid("record_adjudicated_through")
        )

        await ledger.check_stops_for_ticker("AAPL")
        assert calls == []
    finally:
        set_clock(LiveClock())


# ---------------------------------------------------------------------------
# force_exit_position — recorded fill price vs. stop threshold
# (2026-08-27, 047040.KS)
#
# id=280 recorded `price = stop_loss = 19729.244343797276` for a KRX MARKET
# sell: that number is the TRAILING-STOP FLOOR, not a price anything traded
# at. The real fill was ~21,000, so the ledger booked 1,578,339 KRW against
# an actual ~1,680,000 — ~101,660 KRW (~6.4%) of a real sale missing, and
# `cash_after` (a running snapshot every later row is computed against)
# carried the shortfall forward. `exit_price` stays the ORDER price; the
# recorded price is resolved separately.
# ---------------------------------------------------------------------------

_ID280_FLOOR = 19729.244343797276
_ID280_REASON = "stop-loss monitor — breach 2026-08-27: low 18,760.00 <= floor 19,729.24"


def _forced_exit_harness(monkeypatch, *, mirror="submitted: KIS market sell 047040 x80", shares=80):
    """Wires force_exit_position's collaborators and returns the capture
    dicts. `freshest_tick` is deliberately NOT patched here — each test
    below sets it, since that is the thing under test."""
    captured_record = {}
    captured_mirror = {}
    captured_reverse = {}
    alerts = []

    async def fake_get_position(ticker):
        return ledger.Position(cash=0.0, shares=shares)

    async def fake_real_shares_held(ticker):
        return shares

    async def fake_get_open_stop_loss(ticker):
        return (date(2026, 8, 20), _ID280_FLOOR)

    async def fake_record_trade(ticker, trade_date, side, shares_, price, reason, **kwargs):
        captured_record.update(
            ticker=ticker, trade_date=trade_date, side=side, shares=shares_, price=price,
            reason=reason, **kwargs,
        )
        return 280

    async def fake_mirror_trade_to_kis(ticker, trade_date, side, shares_, price, assumed_current_shares, approved=False, trade_id=None):
        captured_mirror.update(price=price, shares=shares_)
        return mirror

    async def fake_reverse_pending_trade(ticker, side, shares_, price, reason, prior_stop_loss=None):
        captured_reverse.update(price=price, shares=shares_, prior_stop_loss=prior_stop_loss)
        return 281

    async def fake_record_mirror_disposition(trade_id, disposition):
        return None

    async def fake_record_alert(alert):
        alerts.append(alert)

    monkeypatch.setattr(ledger, "get_position", fake_get_position)
    monkeypatch.setattr(ledger, "real_shares_held", fake_real_shares_held)
    monkeypatch.setattr(ledger, "get_open_stop_loss", fake_get_open_stop_loss)
    monkeypatch.setattr(ledger, "record_trade", fake_record_trade)
    monkeypatch.setattr(ledger, "mirror_trade_to_kis", fake_mirror_trade_to_kis)
    monkeypatch.setattr(ledger, "reverse_pending_trade", fake_reverse_pending_trade)
    monkeypatch.setattr(ledger, "_record_mirror_disposition", fake_record_mirror_disposition)
    monkeypatch.setattr(ledger, "record_alert", fake_record_alert)
    monkeypatch.setattr(ledger, "filter_new_alerts", lambda given: given)
    return captured_record, captured_mirror, captured_reverse, alerts


async def _force_exit_047040():
    return await ledger.force_exit_position(
        "047040.KS", date(2026, 8, 27), _ID280_FLOOR, _ID280_REASON,
        record_as_of=date(2026, 8, 27),
    )


def _tick_aged(monkeypatch, price, age_seconds):
    received_at = datetime.now(timezone.utc) - timedelta(seconds=age_seconds)
    monkeypatch.setattr(ledger, "freshest_tick", lambda ticker: (price, received_at))


async def test_forced_exit_records_the_live_tick_not_the_trailing_floor(monkeypatch):
    """The 047040.KS regression test."""
    captured_record, captured_mirror, _, _ = _forced_exit_harness(monkeypatch)
    _tick_aged(monkeypatch, 21000.0, 5.0)

    outcome = await _force_exit_047040()

    assert outcome.status == "exited"
    assert captured_record["price"] == 21000.0
    assert captured_record["stop_loss"] == pytest.approx(_ID280_FLOOR)
    assert captured_record["price"] != captured_record["stop_loss"]
    # 21,000 x 80 = 1,680,000 KRW, against the 1,578,339.5 the old behavior
    # would have booked (19,729.244343797276 x 80) — the ~101,660 KRW gap.
    assert 21000.0 * 80 == 1_680_000
    assert _ID280_FLOOR * 80 == pytest.approx(1_578_339.5, abs=0.5)
    assert "live tick" in captured_record["adjustment_note"]
    # The ORDER price is untouched: still the threshold every safety gate
    # in mirror_trade_to_kis was already sizing against.
    assert captured_mirror["price"] == pytest.approx(_ID280_FLOOR)


async def test_forced_exit_falls_back_to_the_threshold_and_flags_the_row_when_no_live_price_exists(monkeypatch):
    captured_record, captured_mirror, _, _ = _forced_exit_harness(monkeypatch)
    monkeypatch.setattr(ledger, "freshest_tick", lambda ticker: None)

    await _force_exit_047040()

    assert captured_record["price"] == _ID280_FLOOR
    assert captured_record["adjustment_note"].startswith(
        "forced-exit price basis: NO live price available"
    )
    assert "NOT a traded price" in captured_record["adjustment_note"]
    assert captured_mirror["price"] == pytest.approx(_ID280_FLOOR)
    # The exact prefix two downstream consumers (app.audit.slippage_validation,
    # app.orchestrator.pipeline's _position_context) match on — see
    # FORCED_EXIT_THRESHOLD_PRICE_NOTE_PREFIX's own comment for why this must
    # stay a literal, not just a substring the test happens to check.
    assert captured_record["adjustment_note"].startswith(
        ledger.FORCED_EXIT_THRESHOLD_PRICE_NOTE_PREFIX
    )
    assert ledger.is_threshold_placeholder_note(captured_record["adjustment_note"])


def test_is_threshold_placeholder_note_rejects_real_fill_notes():
    # None / empty / a genuine fill note (live tick or freshest cached tick)
    # must never be mistaken for a threshold placeholder — only the exact
    # marker prefix (or a backfilled note built from it) counts.
    assert ledger.is_threshold_placeholder_note(None) is False
    assert ledger.is_threshold_placeholder_note("") is False
    assert ledger.is_threshold_placeholder_note(
        "forced-exit price basis: live tick 21000.00 observed by the breach detector (stop threshold 19729.24)"
    ) is False
    assert ledger.is_threshold_placeholder_note(
        "forced-exit price basis: freshest live tick 21000.00 (5s old) at order time (stop threshold 19729.24)"
    ) is False
    assert ledger.is_threshold_placeholder_note(
        "1% risk cap reduced the intended add from 241 to 53 shares"
    ) is False
    # A backfilled historical row appends row-specific evidence AFTER the
    # marker (see the 047040.KS id=280 backfill) — must still match.
    assert ledger.is_threshold_placeholder_note(
        f"{ledger.FORCED_EXIT_THRESHOLD_PRICE_NOTE_PREFIX} (recorded before the 2026-08-27 fix) — "
        "recorded at the stop threshold 19,729.24, which is a threshold and NOT a traded price; "
        "exclude this row from slippage and return statistics"
    ) is True


async def test_forced_exit_rejects_a_stale_tick_as_the_fill_price(monkeypatch):
    captured_record, _, _, _ = _forced_exit_harness(monkeypatch)
    _tick_aged(monkeypatch, 21000.0, 1801.0)

    await _force_exit_047040()

    assert captured_record["price"] == _ID280_FLOOR
    assert "1,801s old" in captured_record["adjustment_note"]
    assert "> 900s" in captured_record["adjustment_note"]


async def test_forced_exit_never_raises_when_the_price_lookup_blows_up(monkeypatch):
    captured_record, _, _, _ = _forced_exit_harness(monkeypatch)

    def boom(ticker):
        raise RuntimeError("websocket feed exploded")

    monkeypatch.setattr(ledger, "freshest_tick", boom)

    outcome = await _force_exit_047040()

    assert outcome.status == "exited"
    assert captured_record["price"] == _ID280_FLOOR
    assert "live price lookup raised" in captured_record["adjustment_note"]


async def test_forced_exit_reversal_undoes_the_exact_cash_it_recorded(monkeypatch):
    """The cash-integrity guard: record_trade and reverse_pending_trade both
    derive cash (and the bps-of-notional fee) from the price they are given,
    so a mirror that never reaches KIS must reverse at the SAME price it
    recorded at, or cash is permanently off by the difference."""
    captured_record, _, captured_reverse, _ = _forced_exit_harness(
        monkeypatch, mirror="BLOCKED: not on the trade allowlist"
    )
    _tick_aged(monkeypatch, 21000.0, 5.0)

    outcome = await _force_exit_047040()

    assert outcome.status == "failed"
    assert captured_record["price"] == 21000.0
    assert captured_reverse["price"] == captured_record["price"] == 21000.0


async def test_forced_exit_in_a_backtest_records_the_threshold_and_no_note(monkeypatch):
    """Byte-identity guard for the replay path: no cache is even consulted."""
    captured_record, _, _, _ = _forced_exit_harness(monkeypatch)
    calls = []

    def counting_tick(ticker):
        calls.append(ticker)
        return (21000.0, datetime.now(timezone.utc))

    monkeypatch.setattr(ledger, "freshest_tick", counting_tick)

    with backtest_scope():
        await _force_exit_047040()

    assert captured_record["price"] == _ID280_FLOOR
    assert captured_record["stop_loss"] == _ID280_FLOOR
    assert captured_record["adjustment_note"] is None
    assert calls == []


async def test_forced_exit_success_alert_quotes_the_floor_and_the_fill_separately(monkeypatch):
    _, _, _, alerts = _forced_exit_harness(monkeypatch)
    _tick_aged(monkeypatch, 21000.0, 5.0)

    await _force_exit_047040()

    assert len(alerts) == 1
    assert "stop-loss (19,729.24) breached" in alerts[0].message
    assert "at ~21,000.00" in alerts[0].message


async def test_forced_exit_alert_omits_the_fill_clause_when_it_equals_the_threshold(monkeypatch):
    _, _, _, alerts = _forced_exit_harness(monkeypatch)
    monkeypatch.setattr(ledger, "freshest_tick", lambda ticker: None)

    await _force_exit_047040()

    assert len(alerts) == 1
    assert "at ~" not in alerts[0].message
    assert "stop-loss (19,729.24) breached" in alerts[0].message


# ---------------------------------------------------------------------------
# starting_cash_for's backtest/live split (2026-08-31). One knob
# (paper_starting_cash) had two irreconcilable jobs — a backtest denominator
# that must be large enough to size meaningful trades, and a live internal-
# ledger fallback that must sit near the real deposit so a failed balance
# lookup doesn't narrate fictional equity. .env.dev set it to ~146 USD for the
# second job, which floored every server-side backtest to 0 shares / all-Hold.
# These tests pin that the two constants are now actually independent, and
# that the discriminator is is_backtest_context() rather than anything the
# caller has to remember to pass.
# 백테스트 기준자본과 실계좌 폴백 기준자본이 서로 독립인지 고정.
# ---------------------------------------------------------------------------


@pytest.fixture
def _usd_no_fx(monkeypatch):
    """Neutralise the currency leg so these tests assert on the constant
    selection alone — a USD ticker at a 1.0 rate makes the conversion an
    identity, with no network lookup."""

    async def fake_get_ticker_currency(ticker):
        return "USD"

    async def fake_get_usd_fx_rate(currency):
        return 1.0

    monkeypatch.setattr(ledger, "get_ticker_currency", fake_get_ticker_currency)
    monkeypatch.setattr(ledger, "get_usd_fx_rate", fake_get_usd_fx_rate)


async def test_starting_cash_for_uses_the_backtest_knob_inside_backtest_scope(monkeypatch, _usd_no_fx):
    monkeypatch.setattr(settings, "backtest_starting_cash", 50_000.0)
    monkeypatch.setattr(settings, "paper_starting_cash", 200.0)

    with backtest_scope():
        assert await ledger.starting_cash_for("AAPL") == 50_000.0


async def test_starting_cash_for_uses_the_paper_knob_outside_backtest_scope(monkeypatch, _usd_no_fx):
    """The live/paper fallback must NOT silently inherit the backtest's large
    round number — that is precisely the fictional-equity figure
    resolve_equity_basis now refuses to size a real account against."""
    monkeypatch.setattr(settings, "backtest_starting_cash", 50_000.0)
    monkeypatch.setattr(settings, "paper_starting_cash", 200.0)

    assert await ledger.starting_cash_for("AAPL") == 200.0


async def test_starting_cash_for_explicit_override_beats_the_ambient_context(monkeypatch, _usd_no_fx):
    """app.backtest.reporting computes a replay's return AFTER the harness's
    backtest_scope() has already exited, so ambient detection would divide by
    the LIVE constant — a 685x-wrong ledger return under .env.dev, silently.
    `for_backtest=True` is that caller's escape hatch.
    스코프 밖에서 리포팅할 때를 위한 명시적 오버라이드."""
    monkeypatch.setattr(settings, "backtest_starting_cash", 50_000.0)
    monkeypatch.setattr(settings, "paper_starting_cash", 200.0)

    assert await ledger.starting_cash_for("AAPL", for_backtest=True) == 50_000.0
    # ...and the mirror image, so the override is proven to be honoured in
    # both directions rather than merely agreeing with the ambient default.
    with backtest_scope():
        assert await ledger.starting_cash_for("AAPL", for_backtest=False) == 200.0


# ---------------------------------------------------------------------------
# rating vs action_label gap notes — 2026-09-01 audit #10. A directional
# rating whose sizing floored to no position change used to leave only a
# logger.info; it now returns an adjustment_note the pipeline persists.
# ---------------------------------------------------------------------------


@pytest.fixture
def _gap_stubs(monkeypatch):
    """Reach the delta==0 branch deterministically: real balance reads return
    None so equity falls back to position.market_value(price), and the caller
    controls the position via the returned closure."""
    pos = {"cash": 200_000.0, "shares": 0}

    async def fake_get_position(ticker):
        return ledger.Position(cash=pos["cash"], shares=pos["shares"])

    async def _none(*a, **k):
        return None

    monkeypatch.setattr(ledger, "get_position", fake_get_position)
    monkeypatch.setattr(ledger, "record_trade", _none)
    monkeypatch.setattr(ledger, "market_lock", lambda market: nullcontext())
    monkeypatch.setattr(ledger, "real_shares_held", _none)
    monkeypatch.setattr(ledger, "real_equity_native", _none)
    return pos


async def test_buy_floors_to_zero_from_flat_produces_gap_note(_gap_stubs, _usd_no_fx):
    # equity 200,000, price 1,431,500, fraction 0.5 -> floor(100000/1431500) = 0
    r = await ledger.rebalance_to_target(
        "AAPL", date(2026, 7, 16), 1_431_500.0, 0.5, reason="t", rating=PortfolioRating.BUY, as_of=date(2026, 7, 16),
    )
    assert r.action_label == "Hold"
    assert r.adjustment_note.startswith(ledger.RATING_ACTION_GAP_NOTE_PREFIX)
    assert "floor(100,000 / 1,431,500.00) = 0 shares" in r.adjustment_note and "stays flat at 0 shares" in r.adjustment_note
    # The floor genuinely IS 0 here (no cap involved), so the sentence stating
    # that arithmetic is the true explanation. 진짜 배분 floor가 0인 경우.
    assert "does not cover one share at 1,431,500.00" in r.adjustment_note


async def test_buy_from_held_position_that_adds_nothing_produces_gap_note(_gap_stubs, _usd_no_fx):
    _gap_stubs["cash"] = 0.0
    _gap_stubs["shares"] = 5
    # equity = 5 * 100 = 500; fraction 1.0 -> floor(500/100) = 5 == current
    r = await ledger.rebalance_to_target(
        "AAPL", date(2026, 7, 16), 100.0, 1.0, reason="t", rating=PortfolioRating.BUY, as_of=date(2026, 7, 16),
    )
    assert r.action_label == "Hold"
    assert "already held" in r.adjustment_note
    assert "stays flat" not in r.adjustment_note


async def test_sell_on_flat_position_produces_gap_note(_gap_stubs, _usd_no_fx):
    r = await ledger.rebalance_to_target(
        "AAPL", date(2026, 7, 16), 100.0, 0.0, reason="t", rating=PortfolioRating.SELL, as_of=date(2026, 7, 16),
    )
    assert r.action_label == "Hold"
    assert "already flat at 0 shares" in r.adjustment_note


async def test_underweight_target_equals_current_produces_gap_note(_gap_stubs, _usd_no_fx):
    _gap_stubs["cash"] = 0.0
    _gap_stubs["shares"] = 10
    # equity 1000, fraction 1.0, price 100 -> target 10 == current
    r = await ledger.rebalance_to_target(
        "AAPL", date(2026, 7, 16), 100.0, 1.0, reason="t", rating=PortfolioRating.UNDERWEIGHT, as_of=date(2026, 7, 16),
    )
    assert "no shares were sold" in r.adjustment_note


async def test_genuine_hold_rating_produces_no_gap_note(_gap_stubs, _usd_no_fx):
    r = await ledger.rebalance_to_target(
        "AAPL", date(2026, 7, 16), 100.0, None, reason="t", rating=PortfolioRating.HOLD, as_of=date(2026, 7, 16),
    )
    assert r.adjustment_note is None


async def test_rating_none_produces_no_gap_note(_gap_stubs, _usd_no_fx):
    r = await ledger.rebalance_to_target(
        "AAPL", date(2026, 7, 16), 1_431_500.0, 0.5, reason="t", as_of=date(2026, 7, 16),
    )
    assert ledger.RATING_ACTION_GAP_NOTE_PREFIX not in (r.adjustment_note or "")


async def test_low_confidence_buy_returns_gap_note_on_no_target_path(_gap_stubs, _usd_no_fx):
    r = await ledger.rebalance_to_target(
        "AAPL", date(2026, 7, 16), 100.0, None, reason="t", rating=PortfolioRating.BUY, as_of=date(2026, 7, 16),
    )
    assert r.action_label == "Hold"
    assert r.adjustment_note.startswith(ledger.RATING_ACTION_GAP_NOTE_PREFIX)
    assert "no target allocation at all" in r.adjustment_note


# ---------------------------------------------------------------------------
# 2026-09-03 audit #4, decision id=1187 (088350.KS): the delta==0 gap note
# stacked a FALSE third sentence — "673,745 allocation is less than the
# 6,500 price of a single share, so floor(...) = 0" — when the real floor
# was 103 and the caps (1% risk cap, then a portfolio EGW00215 fail-closed
# block) took it to 0. The note must only state the floor arithmetic when
# the PRE-cap floor and the post-cap count agree; otherwise the caps that
# shrank it have each already appended their own note.
# ---------------------------------------------------------------------------


async def test_capped_to_zero_buy_does_not_claim_the_allocation_floored_to_zero(_gap_stubs, _usd_no_fx, monkeypatch):
    _gap_stubs["cash"] = 2_000_000.0
    _gap_stubs["shares"] = 0

    async def fake_portfolio_caps(ticker, target_shares, price, current_shares):
        return 0, (
            "Portfolio risk check unavailable (EGW00215) — blocked the exposure increase to "
            "103 shares; held at the existing 0 instead."
        )

    monkeypatch.setattr(ledger, "apply_live_portfolio_caps", fake_portfolio_caps)

    r = await ledger.rebalance_to_target(
        "088350.KS", date(2026, 7, 16), 6_500.0, 0.3369, reason="t",
        rating=PortfolioRating.BUY, stop_loss=6_000.0, current_atr_pct=0.02, as_of=date(2026, 7, 16),
    )
    assert r.action_label == "Hold"
    assert r.shares == 0
    assert ledger.RATING_ACTION_GAP_NOTE_PREFIX in r.adjustment_note
    # The pre-cap floor (103) is stated; the false "= 0 shares" / "less than
    # the price of a single share" claim is gone.
    assert "= 103 shares" in r.adjustment_note
    assert "does not cover one share" not in r.adjustment_note
    assert "= 0 shares" not in r.adjustment_note
    # The cap that actually shrank it still speaks for itself.
    assert "blocked the exposure increase" in r.adjustment_note


async def test_risk_cap_to_zero_buy_states_the_true_floor(_gap_stubs, _usd_no_fx, monkeypatch):
    _gap_stubs["cash"] = 200_000.0
    _gap_stubs["shares"] = 0
    # 1% risk cap alone clamps the whole add away (max_incremental_shares == 0);
    # portfolio caps left as a pass-through so this isolates the risk cap.
    monkeypatch.setattr(ledger.executability, "risk_capped_shares", lambda *a, **k: 0)

    async def passthrough_caps(ticker, target_shares, price, current_shares):
        return target_shares, None

    monkeypatch.setattr(ledger, "apply_live_portfolio_caps", passthrough_caps)

    r = await ledger.rebalance_to_target(
        "AAPL", date(2026, 7, 16), 100.0, 0.5, reason="t", rating=PortfolioRating.BUY,
        stop_loss=95.0, current_atr_pct=0.02, as_of=date(2026, 7, 16),
    )
    assert r.action_label == "Hold"
    assert "1% risk cap" in r.adjustment_note
    assert "does not cover one share" not in r.adjustment_note
    # pre-cap floor was floor(100,000 / 100) = 1000
    assert "= 1000 shares" in r.adjustment_note


async def test_capped_buy_onto_held_position_states_the_precap_floor(_gap_stubs, _usd_no_fx, monkeypatch):
    _gap_stubs["cash"] = 1_500.0
    _gap_stubs["shares"] = 5
    # equity = 1500 + 5*100 = 2000; floor(2000 * 1.0 / 100) = 20 raw, a
    # portfolio cap clamps it back to the 5 already held.
    monkeypatch.setattr(ledger.executability, "risk_capped_shares", lambda *a, **k: 999_999)

    async def fake_portfolio_caps(ticker, target_shares, price, current_shares):
        return 5, "portfolio concentration cap held the add at 5 shares"

    monkeypatch.setattr(ledger, "apply_live_portfolio_caps", fake_portfolio_caps)

    r = await ledger.rebalance_to_target(
        "AAPL", date(2026, 7, 16), 100.0, 1.0, reason="t", rating=PortfolioRating.BUY,
        stop_loss=95.0, current_atr_pct=0.02, as_of=date(2026, 7, 16),
    )
    assert "= 20 shares" in r.adjustment_note
    assert "= 5 shares" not in r.adjustment_note
    assert "already held" in r.adjustment_note


async def test_direction_guard_clamp_on_flat_sell_still_says_nothing_to_sell(_gap_stubs, _usd_no_fx):
    _gap_stubs["cash"] = 200_000.0
    _gap_stubs["shares"] = 0
    # Sell rating, but sizing computes a positive target on a flat position —
    # the direction guard clamps it, and branch 1 ("nothing to sell") wins.
    r = await ledger.rebalance_to_target(
        "AAPL", date(2026, 7, 16), 100.0, 0.5, reason="t", rating=PortfolioRating.SELL,
        current_atr_pct=0.02, as_of=date(2026, 7, 16),
    )
    assert "already flat at 0 shares" in r.adjustment_note
    assert "Direction guard" in r.adjustment_note


async def test_every_capped_gap_note_is_preceded_by_a_cause_note(_gap_stubs, _usd_no_fx, monkeypatch):
    """Invariant: whenever the delta==0 gap note fires with a shrunk target,
    the block that shrank it has ALREADY appended its causal note earlier in
    the string. 캡 원인 노트가 gap 노트보다 항상 앞선다."""
    PREFIX = ledger.RATING_ACTION_GAP_NOTE_PREFIX

    async def passthrough_caps(ticker, target_shares, price, current_shares):
        return target_shares, None

    # --- no-stop block (held position so the ATR new-entry gate is exempt) ---
    _gap_stubs["cash"] = 1_500.0
    _gap_stubs["shares"] = 5
    monkeypatch.setattr(ledger.executability, "risk_capped_shares", lambda *a, **k: 999_999)
    monkeypatch.setattr(ledger, "apply_live_portfolio_caps", passthrough_caps)
    r = await ledger.rebalance_to_target(
        "AAPL", date(2026, 7, 16), 100.0, 1.0, reason="t", rating=PortfolioRating.BUY,
        stop_loss=None, current_atr_pct=None, as_of=date(2026, 7, 16),
    )
    assert r.adjustment_note.index("No resolvable stop-loss") < r.adjustment_note.index(PREFIX)

    # --- 1% risk cap ---
    _gap_stubs["cash"] = 200_000.0
    _gap_stubs["shares"] = 0
    monkeypatch.setattr(ledger.executability, "risk_capped_shares", lambda *a, **k: 0)
    r = await ledger.rebalance_to_target(
        "AAPL", date(2026, 7, 16), 100.0, 0.5, reason="t", rating=PortfolioRating.BUY,
        stop_loss=95.0, current_atr_pct=0.02, as_of=date(2026, 7, 16),
    )
    assert r.adjustment_note.index("1% risk cap") < r.adjustment_note.index(PREFIX)

    # --- portfolio cap ---
    monkeypatch.setattr(ledger.executability, "risk_capped_shares", lambda *a, **k: 999_999)

    async def zero_caps(ticker, target_shares, price, current_shares):
        return 0, "portfolio concentration cap blocked the exposure increase"

    monkeypatch.setattr(ledger, "apply_live_portfolio_caps", zero_caps)
    r = await ledger.rebalance_to_target(
        "AAPL", date(2026, 7, 16), 100.0, 0.5, reason="t", rating=PortfolioRating.BUY,
        stop_loss=95.0, current_atr_pct=0.02, as_of=date(2026, 7, 16),
    )
    assert r.adjustment_note.index("portfolio concentration cap") < r.adjustment_note.index(PREFIX)

    # --- per-order value cap ---
    monkeypatch.setattr(ledger, "apply_live_portfolio_caps", passthrough_caps)
    monkeypatch.setattr(ledger, "route_ticker", lambda t: ("005930", "KRX"))
    monkeypatch.setattr(ledger, "max_order_value_cap", lambda code: (50.0, "KRW"))
    r = await ledger.rebalance_to_target(
        "AAPL", date(2026, 7, 16), 100.0, 0.5, reason="t", rating=PortfolioRating.BUY,
        stop_loss=95.0, current_atr_pct=0.02, as_of=date(2026, 7, 16),
    )
    assert r.adjustment_note.index("per-order value cap") < r.adjustment_note.index(PREFIX)


async def test_gap_note_contains_no_cap_label_substrings_and_no_bracket(_gap_stubs, _usd_no_fx, monkeypatch):
    from app.backtest.portfolio_harness import _CAP_LABELS

    PREFIX = ledger.RATING_ACTION_GAP_NOTE_PREFIX

    async def zero_caps(ticker, target_shares, price, current_shares):
        return 0, "portfolio concentration cap blocked the exposure increase"

    notes = []
    for rating, frac, cash, shares, price in [
        (PortfolioRating.BUY, 0.5, 200_000.0, 0, 1_431_500.0),
        (PortfolioRating.BUY, 1.0, 0.0, 5, 100.0),
        (PortfolioRating.SELL, 0.0, 200_000.0, 0, 100.0),
        (PortfolioRating.UNDERWEIGHT, 1.0, 0.0, 10, 100.0),
        (PortfolioRating.BUY, None, 200_000.0, 0, 100.0),
    ]:
        _gap_stubs["cash"] = cash
        _gap_stubs["shares"] = shares
        r = await ledger.rebalance_to_target("AAPL", date(2026, 7, 16), price, frac, reason="t", rating=rating, as_of=date(2026, 7, 16))
        notes.append((r.adjustment_note or "", False))

    # ...and the two capped-to-zero shapes, where adjustment_note now
    # legitimately contains a real cap note — the _CAP_LABELS / ']' invariant
    # is asserted on the gap-note slice only.
    monkeypatch.setattr(ledger.executability, "risk_capped_shares", lambda *a, **k: 0)
    _gap_stubs["cash"] = 200_000.0
    _gap_stubs["shares"] = 0
    r = await ledger.rebalance_to_target(
        "AAPL", date(2026, 7, 16), 100.0, 0.5, reason="t", rating=PortfolioRating.BUY,
        stop_loss=95.0, current_atr_pct=0.02, as_of=date(2026, 7, 16),
    )
    notes.append((r.adjustment_note or "", True))

    monkeypatch.setattr(ledger.executability, "risk_capped_shares", lambda *a, **k: 999_999)
    monkeypatch.setattr(ledger, "apply_live_portfolio_caps", zero_caps)
    r = await ledger.rebalance_to_target(
        "AAPL", date(2026, 7, 16), 100.0, 0.5, reason="t", rating=PortfolioRating.BUY,
        stop_loss=95.0, current_atr_pct=0.02, as_of=date(2026, 7, 16),
    )
    notes.append((r.adjustment_note or "", True))

    for note, capped in notes:
        slice_ = note[note.index(PREFIX):] if capped else note
        assert "]" not in slice_
        for label in _CAP_LABELS:
            assert label.lower() not in slice_.lower()


async def test_reversed_directional_order_that_nets_zero_produces_gap_note(monkeypatch, _rebalance_stubs):
    """2026-09-01 audit #10 P1: a BUY whose order was BLOCKED and fully
    reversed nets to no position change (action_label Hold). The decision
    row should carry a note saying so, even though the trades row already
    has the mirror disposition."""
    monkeypatch.setattr(settings, "execution_mode", ExecutionMode.FULL_AUTO)

    async def fake_mirror(ticker, trade_date, side, shares, price, assumed_current_shares=None, trade_id=None):
        return "BLOCKED: circuit-breaker state unknown"

    async def fake_reverse(ticker, side, shares, price, reason, prior_stop_loss=None):
        return 10  # back to _rebalance_stubs's pre-trade shares

    async def fake_portfolio_caps(ticker, target_shares, price, current_shares):
        return target_shares, None  # let the add through untouched

    monkeypatch.setattr(ledger, "mirror_trade_to_kis", fake_mirror)
    monkeypatch.setattr(ledger, "reverse_pending_trade", fake_reverse)
    monkeypatch.setattr(ledger, "apply_live_portfolio_caps", fake_portfolio_caps)

    # stop_loss present so the 1% cap lets a 1-share add through to the mirror
    # (equity 1000, risk/share 10, budget 10 -> +1 share), rather than the
    # no-stop guard collapsing target to current and hitting the delta==0 path.
    r = await ledger.rebalance_to_target(
        "AAPL", date(2026, 7, 16), 100.0, 2.0, reason="t", stop_loss=90.0,
        rating=PortfolioRating.BUY,
    )
    assert r.action_label == "Hold"
    assert r.shares == 0 and r.shares_after == 10
    assert r.adjustment_note is not None
    assert ledger.RATING_ACTION_GAP_NOTE_PREFIX in r.adjustment_note
    assert "never reached the broker" in r.adjustment_note


# ---------------------------------------------------------------------------
# 2026-09-01 audit #6: app.domain.executability's arithmetic kernels were
# extracted FROM this module's own inline formulas (allocation cap, ATR stop
# floor, 1% risk cap), not reimplemented, specifically so
# app.live.auto_trade_loop's pre-ranking affordability filter can never
# silently drift from the sizing this module actually enforces on a real
# trade. These tests are the drift guard: if a future change to either side
# stops matching the other, one of these breaks.
# ---------------------------------------------------------------------------


async def test_executability_parity_with_rebalance_to_target(_gap_stubs, _usd_no_fx):
    """`app.domain.executability.is_unexecutable`'s verdict must agree with
    whether `rebalance_to_target` actually adds any shares, across a grid of
    (equity, price, atr) — flat starting position, target_fraction=1.0 (BUY's
    own base allocation) on both sides, portfolio caps disabled
    (DISABLED_PORTFOLIO_CAPS — a live/backtest-cap network call has nothing
    to do with what this test is checking), no AI-proposed stop-loss (so
    `resolve_stop_loss` derives its default from the same ATR floor the
    filter assumes as its "best case" stop distance).

    cash is set equal to equity in `_gap_stubs`, so with target_fraction=1.0
    the cash cap and the allocation cap are numerically identical
    (floor(cash/price) == floor(equity*1.0/price)) — this isolates the
    parity check to the allocation/risk kernels the ledger and the filter
    actually share, rather than also re-deriving the cash-affordability
    logic that only the filter has (the ledger has no cash-cap concept of
    its own; app.execution.kis_bridge gates cash separately, at order-mirror
    time, which this test doesn't reach).

    ATR grid deliberately excludes 0/None: with no ATR and no AI stop, the
    ledger's "no resolvable stop -> block the increase entirely" guard and
    the filter's "no ATR -> risk cap uncapped" reading diverge BY DESIGN
    (see app.domain.executability.min_stop_distance's docstring and the
    audit #6 design's §2.2) — the filter is deliberately more permissive
    there (an occasional wasted analysis, never a wrongly-dropped
    candidate), not a bug this test should pin as equal."""
    from app.domain.executability import ExecutabilityInputs, SizingLimits, is_unexecutable

    grid = [
        (equity, price, atr)
        for equity in (200_000.0, 1_000_000.0, 10_000_000.0)
        for price in (500.0, 12_000.0, 61_000.0, 1_431_500.0)
        for atr in (0.005, 0.02, 0.05, 0.10, 0.30)
    ]
    for equity, price, atr in grid:
        _gap_stubs["cash"] = equity
        _gap_stubs["shares"] = 0
        result = await ledger.rebalance_to_target(
            "AAPL", date(2026, 7, 16), price, 1.0, reason="parity",
            current_atr_pct=atr, portfolio_cap_override=portfolio_risk.DISABLED_PORTFOLIO_CAPS,
            rating=PortfolioRating.BUY, as_of=date(2026, 7, 16),
        )
        verdict_unexecutable = is_unexecutable(
            ExecutabilityInputs(price=price, atr_pct=atr, equity=equity, cash=equity),
            SizingLimits(max_allocation_fraction=1.0, max_risk_pct_of_equity=0.01, min_stop_atr_multiple=2.5),
        )
        assert verdict_unexecutable == (result.shares == 0), (
            f"parity mismatch at equity={equity} price={price} atr={atr}: "
            f"is_unexecutable={verdict_unexecutable} but ledger sized {result.shares} shares"
        )


async def test_executability_kernels_match_the_extracted_inline_formulas():
    """Pins the exact arithmetic against literals re-derived independently
    of the extraction — guards the refactor itself (app.execution.ledger
    calling into app.domain.executability) against a transcription error,
    separate from the black-box parity test above."""
    from app.domain.executability import allocation_target_shares, min_stop_distance, risk_capped_shares

    # allocation_target_shares == floor(equity * target_fraction / price)
    assert allocation_target_shares(953_412.0, 0.37, 8_100.0) == math.floor(953_412.0 * 0.37 / 8_100.0)

    # min_stop_distance == price * atr_pct * multiple, clamped to price * 0.95
    assert min_stop_distance(45_300.0, 0.031, 2.5) == 45_300.0 * 0.031 * 2.5
    assert min_stop_distance(45_300.0, 0.60, 2.5) == 45_300.0 * 0.95  # clamp engaged

    # risk_capped_shares == max(floor(equity * max_risk_pct / risk_per_share), 0)
    assert risk_capped_shares(1_500_000.0, 733.25, 0.01) == max(
        math.floor(1_500_000.0 * 0.01 / 733.25), 0
    )


# ---------------------------------------------------------------------------
# 2026-09-02 fail-closed ATR gate (decision id=1169, 950260.KS): a NEW entry
# is refused this cycle when app.domain.executability.blocks_new_entry_
# without_atr says the ticker has no usable 14-day ATR. See that function's
# and app.config.settings.auto_trade_require_atr_for_new_entry's own
# docstrings for the full incident and rationale.
# ---------------------------------------------------------------------------


@pytest.fixture
def _atr_gate_flat_stubs(monkeypatch):
    """A flat position (internal ledger AND real broker agree: 0 shares),
    live mode — the shape under which the gate is meant to fire."""

    async def fake_get_position(ticker):
        return ledger.Position(cash=1_000_000.0, shares=0)

    async def fake_real_shares_held(ticker):
        return 0

    async def fake_real_equity_native(market):
        return 1_000_000.0

    async def fake_apply_live_portfolio_caps(ticker, target_shares, price, current_shares):
        return target_shares, None

    captured = {}

    async def fake_record_trade(*args, **kwargs):
        captured.update(kwargs)

    captured_alerts = []

    async def fake_record_alert(alert):
        captured_alerts.append(alert)

    monkeypatch.setattr(ledger, "get_position", fake_get_position)
    monkeypatch.setattr(ledger, "real_shares_held", fake_real_shares_held)
    monkeypatch.setattr(ledger, "real_equity_native", fake_real_equity_native)
    monkeypatch.setattr(ledger, "apply_live_portfolio_caps", fake_apply_live_portfolio_caps)
    monkeypatch.setattr(ledger, "record_trade", fake_record_trade)
    monkeypatch.setattr(ledger, "record_alert", fake_record_alert)
    monkeypatch.setattr(ledger, "market_lock", lambda market: nullcontext())
    return captured, captured_alerts


async def test_atr_gate_refuses_new_entry_when_atr_is_none(monkeypatch, _atr_gate_flat_stubs):
    monkeypatch.setattr(settings, "execution_mode", ExecutionMode.MANUAL)
    _captured, captured_alerts = _atr_gate_flat_stubs

    result = await ledger.rebalance_to_target(
        "950260.KS", date(2026, 9, 2), 100.0, 1.0, reason="test buy",
        current_atr_pct=None, rating=PortfolioRating.BUY,
    )

    assert result.side is None
    assert result.shares == 0
    assert result.shares_after == 0
    assert result.action_label == "Hold"
    assert result.adjustment_note is not None
    assert result.adjustment_note.startswith("No new position opened this cycle")
    assert len(captured_alerts) == 1
    assert captured_alerts[0].kind == "atr_unavailable_new_entry"
    assert "950260.KS" in captured_alerts[0].message


async def test_atr_gate_refuses_new_entry_when_atr_is_zero(monkeypatch, _atr_gate_flat_stubs):
    monkeypatch.setattr(settings, "execution_mode", ExecutionMode.MANUAL)
    _captured, captured_alerts = _atr_gate_flat_stubs

    result = await ledger.rebalance_to_target(
        "950260.KS", date(2026, 9, 2), 100.0, 1.0, reason="test buy",
        current_atr_pct=0.0, rating=PortfolioRating.BUY,
    )

    assert result.side is None and result.shares == 0
    assert result.adjustment_note.startswith("No new position opened this cycle")
    assert len(captured_alerts) == 1


async def test_atr_gate_note_contains_no_bracket(monkeypatch, _atr_gate_flat_stubs):
    # Same convention as RATING_ACTION_GAP_NOTE_PREFIX just above: the audit
    # SQL / signals.js extract a persisted '[EXECUTION NOTE: ...]' with a
    # ']'-terminated match, so this note must never contain ']'.
    monkeypatch.setattr(settings, "execution_mode", ExecutionMode.MANUAL)
    _captured, _alerts = _atr_gate_flat_stubs

    result = await ledger.rebalance_to_target(
        "950260.KS", date(2026, 9, 2), 100.0, 1.0, reason="test buy",
        current_atr_pct=None, rating=PortfolioRating.BUY,
    )
    assert "]" not in result.adjustment_note


async def test_atr_gate_respects_the_kill_switch(monkeypatch, _atr_gate_flat_stubs):
    monkeypatch.setattr(settings, "execution_mode", ExecutionMode.MANUAL)
    monkeypatch.setattr(settings, "auto_trade_require_atr_for_new_entry", False)
    _captured, captured_alerts = _atr_gate_flat_stubs

    # stop_loss given explicitly: with no ATR and no AI stop, the SEPARATE
    # "no resolvable stop-loss" guard (unaffected by this switch) would block
    # the increase regardless, which would falsely look like the ATR gate
    # still firing. A given stop isolates this test to the ATR gate alone.
    result = await ledger.rebalance_to_target(
        "950260.KS", date(2026, 9, 2), 100.0, 1.0, reason="test buy",
        current_atr_pct=None, stop_loss=90.0, rating=PortfolioRating.BUY,
    )

    # Restores exact pre-fix behavior: sized off the base (unscaled)
    # allocation, with the given stop passed through unvalidated (no ATR
    # floor exists to check it against).
    assert result.side == "buy"
    assert result.shares > 0
    assert not captured_alerts


async def test_atr_gate_allows_adding_to_an_existing_held_position_without_atr(monkeypatch):
    # Held (current_shares=5) -> the gate is about NEW exposure only; a held
    # position keeps its existing stop/trailing-stop machinery, never forced
    # to sell/reduce/hold for a missing ATR.
    monkeypatch.setattr(settings, "execution_mode", ExecutionMode.MANUAL)

    async def fake_get_position(ticker):
        return ledger.Position(cash=0.0, shares=5)

    async def fake_real_shares_held(ticker):
        return 5

    async def fake_real_equity_native(market):
        return 1_000_000.0

    async def fake_apply_live_portfolio_caps(ticker, target_shares, price, current_shares):
        return target_shares, None

    async def fake_record_trade(*args, **kwargs):
        return None

    monkeypatch.setattr(ledger, "get_position", fake_get_position)
    monkeypatch.setattr(ledger, "real_shares_held", fake_real_shares_held)
    monkeypatch.setattr(ledger, "real_equity_native", fake_real_equity_native)
    monkeypatch.setattr(ledger, "apply_live_portfolio_caps", fake_apply_live_portfolio_caps)
    monkeypatch.setattr(ledger, "record_trade", fake_record_trade)
    monkeypatch.setattr(ledger, "market_lock", lambda market: nullcontext())

    result = await ledger.rebalance_to_target(
        "950260.KS", date(2026, 9, 2), 100.0, 1.0, reason="test add",
        current_atr_pct=None, rating=PortfolioRating.BUY,
    )

    # Adds beyond the existing 5 shares without being refused by the ATR gate
    # (may still be shaped by the "no resolvable stop" guard elsewhere, which
    # this test does not exercise — stop_loss is left unset and no ATR floor
    # exists either, so the increase-guard below may cap it to 0 too; the
    # point pinned here is specifically that the ATR gate itself never fires
    # for a held ticker).
    assert result.adjustment_note is None or "No new position opened this cycle" not in (
        result.adjustment_note or ""
    )


async def test_atr_gate_allows_sell_to_flat_without_atr(monkeypatch, _atr_gate_flat_stubs):
    # target_fraction=0.0 from a flat position -> target_shares=0, so
    # `target_shares > 0` is false and the gate never engages regardless of
    # ATR — this is a no-op Sell-on-flat, not a new entry.
    monkeypatch.setattr(settings, "execution_mode", ExecutionMode.MANUAL)
    _captured, captured_alerts = _atr_gate_flat_stubs

    result = await ledger.rebalance_to_target(
        "950260.KS", date(2026, 9, 2), 100.0, 0.0, reason="test sell",
        current_atr_pct=None, rating=PortfolioRating.SELL,
    )

    assert result.adjustment_note is None or "No new position opened this cycle" not in (
        result.adjustment_note or ""
    )
    assert not captured_alerts


async def test_atr_gate_uses_real_shares_held_not_the_internal_ledger(monkeypatch):
    # The internal ledger thinks this ticker is flat (0 shares), but the real
    # broker already holds it — the gate must NOT treat this as a new entry
    # (see rebalance_to_target's own docstring on real_shares_held).
    monkeypatch.setattr(settings, "execution_mode", ExecutionMode.MANUAL)

    async def fake_get_position(ticker):
        return ledger.Position(cash=1_000_000.0, shares=0)  # internal ledger: never bought this

    async def fake_real_shares_held(ticker):
        return 3  # real broker: already holds 3

    async def fake_real_equity_native(market):
        return 1_000_000.0

    async def fake_apply_live_portfolio_caps(ticker, target_shares, price, current_shares):
        return target_shares, None

    captured_alerts = []

    async def fake_record_alert(alert):
        captured_alerts.append(alert)

    async def fake_record_trade(*args, **kwargs):
        return None

    monkeypatch.setattr(ledger, "get_position", fake_get_position)
    monkeypatch.setattr(ledger, "real_shares_held", fake_real_shares_held)
    monkeypatch.setattr(ledger, "real_equity_native", fake_real_equity_native)
    monkeypatch.setattr(ledger, "apply_live_portfolio_caps", fake_apply_live_portfolio_caps)
    monkeypatch.setattr(ledger, "record_trade", fake_record_trade)
    monkeypatch.setattr(ledger, "record_alert", fake_record_alert)
    monkeypatch.setattr(ledger, "market_lock", lambda market: nullcontext())

    result = await ledger.rebalance_to_target(
        "950260.KS", date(2026, 9, 2), 100.0, 1.0, reason="test add",
        current_atr_pct=None, rating=PortfolioRating.BUY,
    )

    assert not captured_alerts
    assert result.adjustment_note is None or "No new position opened this cycle" not in (
        result.adjustment_note or ""
    )


async def test_atr_gate_applies_in_backtest_mode_too(monkeypatch, _gap_stubs, _usd_no_fx):
    # as_of set (backtest) -- the gate must still refuse a new entry lacking
    # ATR, same as live. _gap_stubs starts flat (shares=0).
    result = await ledger.rebalance_to_target(
        "950260.KS", date(2026, 9, 2), 100.0, 1.0, reason="test buy",
        current_atr_pct=None, rating=PortfolioRating.BUY, as_of=date(2026, 9, 2),
    )
    assert result.side is None
    assert result.action_label == "Hold"
    assert result.adjustment_note is not None
    assert result.adjustment_note.startswith("No new position opened this cycle")
