"""app.execution.portfolio_risk — degrade-quietly-on-DB-failure regression
tests (item 13 of the codebase review: asyncpg.InterfaceError wasn't caught
by the old (OSError, PostgresError) clause), plus ticker_market/
real_account_snapshot (real KIS balance, no ledger) and per-market
high-water-mark isolation. No real DB/network — broker and pool faked."""

from __future__ import annotations

import asyncio

import asyncpg
import pytest

from app.domain.models import BrokerBalance, BrokerPosition, OverseasCash
from app.execution import ledger as ledger_module
from app.execution import portfolio_risk
from app.execution.ledger import Position
from app.user_context import get_current_user_id, user_scope

_WHO = get_current_user_id()
from app.web import state
from conftest import broker_stub


@pytest.fixture(autouse=True)
def _reset_schema_ready(monkeypatch):
    monkeypatch.setattr(portfolio_risk, "_schema_ready", False)


@pytest.fixture(autouse=True)
def _clear_broker(monkeypatch):
    monkeypatch.setattr(state, "kis_broker", broker_stub(None))


async def test_get_high_water_mark_degrades_on_interface_error(monkeypatch):
    async def broken_ready_pool():
        raise asyncpg.InterfaceError("pool is closing")

    monkeypatch.setattr(portfolio_risk, "_ready_pool", broken_ready_pool)
    assert await portfolio_risk._get_high_water_mark("domestic") is None


async def test_get_high_water_mark_degrades_on_os_error(monkeypatch):
    async def broken_ready_pool():
        raise OSError("connection refused")

    monkeypatch.setattr(portfolio_risk, "_ready_pool", broken_ready_pool)
    assert await portfolio_risk._get_high_water_mark("overseas") is None


async def test_update_high_water_mark_degrades_on_interface_error(monkeypatch):
    async def broken_ready_pool():
        raise asyncpg.InterfaceError("pool is closing")

    monkeypatch.setattr(portfolio_risk, "_ready_pool", broken_ready_pool)
    # Must not raise.
    await portfolio_risk._update_high_water_mark("domestic", 100_000.0)


# ---------------------------------------------------------------------------
# ticker_market
# ---------------------------------------------------------------------------


def test_ticker_market_classifies_ks_suffix_as_domestic():
    assert portfolio_risk.ticker_market("005930.KS") == "domestic"


def test_ticker_market_classifies_bare_symbol_as_overseas():
    assert portfolio_risk.ticker_market("AAPL") == "overseas"


def test_benchmark_for_market_returns_kospi_for_domestic():
    # Must never be SPY — see the function's own comment for why scoring a
    # domestic decision against the US index makes alpha uninterpretable.
    assert portfolio_risk.benchmark_for_market("domestic") == "^KS11"


def test_benchmark_for_market_returns_spy_for_overseas():
    assert portfolio_risk.benchmark_for_market("overseas") == "SPY"


# ---------------------------------------------------------------------------
# real_account_snapshot
# ---------------------------------------------------------------------------


class FakeRealBalanceBroker:
    def __init__(self, domestic_fail=False, overseas_fail=False, unmatched_overseas_cash=False):
        self.domestic_fail = domestic_fail
        self.overseas_fail = overseas_fail
        self.unmatched_overseas_cash = unmatched_overseas_cash

    async def fetch_balance(self):
        if self.domestic_fail:
            raise RuntimeError("KIS domestic balance inquiry failed")
        return BrokerBalance(
            cash_deposit=1_000_000.0,
            # real_account_snapshot/real_equity_native sum d2_cash_deposit
            # (not cash_deposit) for domestic — see their own docstrings.
            d2_cash_deposit=1_000_000.0,
            total_eval_amount=2_000_000.0,
            total_pnl_amount=0.0,
            purchase_amount=0.0,
            positions=[
                BrokerPosition(
                    code="005930", name="삼성전자", quantity=10,
                    avg_price=70_000.0, current_price=70_000.0, eval_amount=700_000.0,
                    pnl_amount=0.0, pnl_pct=0.0,
                )
            ],
        )

    async def fetch_overseas_balance(self):
        if self.overseas_fail:
            raise RuntimeError("KIS overseas balance inquiry failed")
        return BrokerBalance(
            cash_deposit=0.0,
            total_eval_amount=1_500.0,
            total_pnl_amount=0.0,
            purchase_amount=0.0,
            positions=[
                BrokerPosition(
                    code="AAPL", name="Apple Inc.", quantity=5,
                    avg_price=300.0, current_price=300.0, eval_amount=1_500.0,
                    pnl_amount=0.0, pnl_pct=0.0,
                )
            ],
        )

    async def fetch_overseas_present_balance(self, currency="USD"):
        if self.unmatched_overseas_cash:
            # No output2 row matched the requested currency — every field
            # is a zeroed placeholder (see OverseasCash's own docstring).
            return OverseasCash(currency=currency, deposit=0.0, orderable=0.0, eval_amount=0.0, matched=False)
        return OverseasCash(currency=currency, deposit=500.0, orderable=500.0, eval_amount=1_500.0, matched=True)


async def test_real_account_snapshot_raises_when_broker_not_connected():
    with pytest.raises(portfolio_risk.AccountUnavailable):
        await portfolio_risk.real_account_snapshot("domestic")


async def test_real_account_snapshot_domestic_stays_in_native_krw(monkeypatch):
    """Domestic and overseas snapshots are never combined (see this module's
    docstring), so there's no cross-currency sum that would need a common
    unit — real_account_snapshot keeps domestic in native KRW rather than
    converting to USD. get_usd_fx_rate must not even be called for this."""
    monkeypatch.setattr(state, "kis_broker", broker_stub(FakeRealBalanceBroker()))

    async def unexpected_fx_rate(currency):
        raise AssertionError("domestic snapshot must not convert currency")

    monkeypatch.setattr(portfolio_risk, "get_usd_fx_rate", unexpected_fx_rate)
    snap = await portfolio_risk.real_account_snapshot("domestic")
    assert snap.per_ticker_value == {"005930.KS": 700_000.0}
    assert snap.total_value == pytest.approx(3_000_000.0)  # 1,000,000 cash + 2,000,000 eval


async def test_real_account_snapshot_overseas_includes_present_balance_cash(monkeypatch):
    monkeypatch.setattr(state, "kis_broker", broker_stub(FakeRealBalanceBroker()))
    snap = await portfolio_risk.real_account_snapshot("overseas")
    assert snap.per_ticker_value == {"AAPL": 1_500.0}
    assert snap.total_value == pytest.approx(2_000.0)  # 500 present-balance cash + 1,500 eval


# cash.matched=False means no output2 row matched USD — deposit is then a
# zeroed placeholder, not a confirmed zero balance. Treating that
# placeholder as real cash used to silently drop total_value to
# securities-only, which safety_switch.daily_loss_block then compared
# against today's (correctly matched) baseline and reported as a huge
# loss, tripping the daily-loss breaker on a data gap rather than an
# actual loss. Must fail closed (AccountUnavailable) like every other read
# failure in this function, not fail open on a placeholder.
async def test_real_account_snapshot_overseas_raises_when_cash_unmatched(monkeypatch):
    monkeypatch.setattr(state, "kis_broker", broker_stub(FakeRealBalanceBroker(unmatched_overseas_cash=True)))
    with pytest.raises(portfolio_risk.AccountUnavailable):
        await portfolio_risk.real_account_snapshot("overseas")


async def test_real_account_snapshot_wraps_broker_failure_as_account_unavailable(monkeypatch):
    monkeypatch.setattr(state, "kis_broker", broker_stub(FakeRealBalanceBroker(domestic_fail=True)))
    with pytest.raises(portfolio_risk.AccountUnavailable):
        await portfolio_risk.real_account_snapshot("domestic")


# ---------------------------------------------------------------------------
# cached_account_snapshot — read-only/narrative TTL cache (2026-09-01) in
# front of real_account_snapshot. Fixes the EGW00215 burst from N concurrent
# narrative callers; the enforcement path stays uncached.
# ---------------------------------------------------------------------------


class _CountingBalanceBroker(FakeRealBalanceBroker):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.balance_calls = 0

    async def fetch_balance(self):
        self.balance_calls += 1
        return await super().fetch_balance()


async def test_cached_account_snapshot_collapses_concurrent_callers(monkeypatch):
    broker = _CountingBalanceBroker()
    monkeypatch.setattr(state, "kis_broker", broker_stub(broker))
    results = await asyncio.gather(*(
        portfolio_risk.cached_account_snapshot("domestic") for _ in range(8)
    ))
    assert broker.balance_calls == 1
    assert all(r is results[0] for r in results)


async def test_cached_account_snapshot_refetches_after_ttl(monkeypatch):
    broker = _CountingBalanceBroker()
    monkeypatch.setattr(state, "kis_broker", broker_stub(broker))
    await portfolio_risk.cached_account_snapshot("domestic", max_age=100.0)
    await portfolio_risk.cached_account_snapshot("domestic", max_age=0.0)  # force-expire
    assert broker.balance_calls == 2


async def test_cached_account_snapshot_never_caches_a_failure(monkeypatch):
    broker = _CountingBalanceBroker(domestic_fail=True)
    monkeypatch.setattr(state, "kis_broker", broker_stub(broker))
    for _ in range(2):
        with pytest.raises(portfolio_risk.AccountUnavailable):
            await portfolio_risk.cached_account_snapshot("domestic")
    assert broker.balance_calls == 2  # every call retried the broker


async def test_cached_account_snapshot_keys_by_market(monkeypatch):
    broker = _CountingBalanceBroker()
    monkeypatch.setattr(state, "kis_broker", broker_stub(broker))
    await portfolio_risk.cached_account_snapshot("domestic")
    await portfolio_risk.cached_account_snapshot("overseas")
    assert broker.balance_calls == 1  # domestic; overseas uses fetch_overseas_balance


async def test_real_account_snapshot_stays_uncached_by_the_read_cache(monkeypatch):
    # Enforcement-path regression guard: filling the read cache must not
    # short-circuit real_account_snapshot's own fetch.
    broker = _CountingBalanceBroker()
    monkeypatch.setattr(state, "kis_broker", broker_stub(broker))
    await portfolio_risk.cached_account_snapshot("domestic")   # call 1
    await portfolio_risk.real_account_snapshot("domestic")     # call 2 — fresh
    await portfolio_risk.real_account_snapshot("domestic")     # call 3 — fresh
    assert broker.balance_calls == 3


# ---------------------------------------------------------------------------
# cached_balance_fetches — request-scoped cache (2026-08-06) collapsing
# real_shares_held + real_equity_native's independent fetch_balance/
# fetch_overseas_balance calls into one when both run back-to-back for the
# same ticker (ledger.rebalance_to_target's own use). Deliberately does NOT
# touch real_account_snapshot — see cached_balance_fetches's own docstring,
# and cached_account_snapshots further below for the separate,
# lock-scoped cache that DOES dedupe that one.
# ---------------------------------------------------------------------------


class CountingBalanceBroker(FakeRealBalanceBroker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.domestic_calls = 0
        self.overseas_calls = 0
        self.present_balance_calls = 0

    async def fetch_balance(self):
        self.domestic_calls += 1
        return await super().fetch_balance()

    async def fetch_overseas_balance(self):
        self.overseas_calls += 1
        return await super().fetch_overseas_balance()

    async def fetch_overseas_present_balance(self, currency="USD"):
        self.present_balance_calls += 1
        return await super().fetch_overseas_present_balance(currency)


async def test_real_equity_native_overseas_includes_present_balance_cash(monkeypatch):
    monkeypatch.setattr(state, "kis_broker", broker_stub(FakeRealBalanceBroker()))
    equity = await portfolio_risk.real_equity_native("overseas")
    assert equity == pytest.approx(2_000.0)  # 500 present-balance cash + 1,500 eval


async def test_real_equity_native_overseas_falls_back_to_none_when_cash_unmatched(monkeypatch):
    # Same fail-closed reasoning as real_account_snapshot's identical
    # check — matched=False means deposit is a zeroed placeholder, not a
    # real balance. real_equity_native's own contract on any failure is
    # None (caller falls back to the internal ledger), not a wrong number.
    monkeypatch.setattr(state, "kis_broker", broker_stub(FakeRealBalanceBroker(unmatched_overseas_cash=True)))
    assert await portfolio_risk.real_equity_native("overseas") is None


async def test_cached_balance_fetches_collapses_domestic_calls(monkeypatch):
    broker = CountingBalanceBroker()
    monkeypatch.setattr(state, "kis_broker", broker_stub(broker))

    async with portfolio_risk.cached_balance_fetches():
        shares = await portfolio_risk.real_shares_held("005930.KS")
        equity = await portfolio_risk.real_equity_native("domestic")

    assert shares == 10
    assert equity == pytest.approx(3_000_000.0)  # d2_cash_deposit + total_eval_amount
    assert broker.domestic_calls == 1  # would be 2 without the cache


async def test_cached_balance_fetches_collapses_overseas_calls(monkeypatch):
    broker = CountingBalanceBroker()
    monkeypatch.setattr(state, "kis_broker", broker_stub(broker))

    async with portfolio_risk.cached_balance_fetches():
        shares = await portfolio_risk.real_shares_held("AAPL")
        equity = await portfolio_risk.real_equity_native("overseas")

    assert shares == 5
    assert equity == pytest.approx(2_000.0)  # present-balance cash + total_eval_amount
    assert broker.overseas_calls == 1  # would be 2 without the cache
    # Only real_equity_native is exercised in THIS test, so there's nothing
    # else in this particular scope to merge against — NOT because
    # present-balance has just one caller overall. real_account_snapshot's
    # overseas branch calls the same TR too (see the enforcement_safe tests
    # below) — see 2026-09-08's EGW00215 follow-up (real_equity_native_
    # detailed) for the incident this wrong assumption once masked.
    assert broker.present_balance_calls == 1


async def test_without_cache_scope_each_call_hits_the_broker_independently(monkeypatch):
    """Regression guard the other way: outside `cached_balance_fetches`,
    both functions must still each do their own fresh fetch - the cache
    only ever applies inside an explicitly-opened scope. The other 3 call
    sites of real_shares_held (kis_bridge.py, ledger.py's forced-exit path,
    pipeline.py) never open this scope, so they must be completely
    unaffected."""
    broker = CountingBalanceBroker()
    monkeypatch.setattr(state, "kis_broker", broker_stub(broker))

    await portfolio_risk.real_shares_held("005930.KS")
    await portfolio_risk.real_equity_native("domestic")

    assert broker.domestic_calls == 2  # no cache scope open - no merging


async def test_cached_balance_fetches_scope_does_not_leak_across_calls(monkeypatch):
    """One `cached_balance_fetches()` block must not silently cache across
    to a LATER, separate block - each is its own fresh window."""
    broker = CountingBalanceBroker()
    monkeypatch.setattr(state, "kis_broker", broker_stub(broker))

    async with portfolio_risk.cached_balance_fetches():
        await portfolio_risk.real_shares_held("005930.KS")
    async with portfolio_risk.cached_balance_fetches():
        await portfolio_risk.real_shares_held("005930.KS")

    assert broker.domestic_calls == 2  # each scope fetched fresh, no cross-scope reuse


# ---------------------------------------------------------------------------
# cached_balance_fetches(enforcement_safe=True) — FIX-1, 2026-09-03 id=1187
# audit: within one REAL market_lock hold, real_account_snapshot (and hence
# apply_live_portfolio_caps) may now reuse a balance real_shares_held already
# fetched into this same scope, instead of always re-asking the broker — that
# second, avoidable ask is what tripped KIS's EGW00215 per-second ledger
# limit and fail-closed-blocked a BUY. Must stay OFF by default (enforcement_
# safe=False) for every scope not opened by a held market_lock — see the
# regression guard below.
# ---------------------------------------------------------------------------


async def test_enforcement_safe_scope_lets_real_account_snapshot_reuse_domestic_balance(monkeypatch):
    broker = CountingBalanceBroker()
    monkeypatch.setattr(state, "kis_broker", broker_stub(broker))

    async with portfolio_risk.cached_balance_fetches(enforcement_safe=True):
        await portfolio_risk.real_shares_held("005930.KS")  # primes the cache
        snap = await portfolio_risk.real_account_snapshot("domestic")

    assert snap.total_value == pytest.approx(3_000_000.0)
    assert broker.domestic_calls == 1  # would be 2 without enforcement_safe reuse


async def test_enforcement_safe_scope_lets_real_account_snapshot_reuse_overseas_balance(monkeypatch):
    broker = CountingBalanceBroker()
    monkeypatch.setattr(state, "kis_broker", broker_stub(broker))

    async with portfolio_risk.cached_balance_fetches(enforcement_safe=True):
        await portfolio_risk.real_shares_held("AAPL")  # primes fetch_overseas_balance's cache entry
        snap = await portfolio_risk.real_account_snapshot("overseas")

    assert snap.total_value == pytest.approx(2_000.0)
    assert broker.overseas_calls == 1  # would be 2 without enforcement_safe reuse
    assert broker.present_balance_calls == 1  # real_account_snapshot's own present-balance fetch, now also cached


async def test_enforcement_safe_defaults_to_false_real_account_snapshot_stays_fresh(monkeypatch):
    """Regression guard for the 09ef687 safety property this fix must not
    break: a `cached_balance_fetches()` scope opened WITHOUT
    `enforcement_safe=True` (e.g. app.orchestrator.pipeline's narrative-only
    scope) must still make real_account_snapshot fetch fresh every time,
    even though the balance cache is populated and would otherwise satisfy
    the request — reusing a pre-market_lock balance in an ENFORCEMENT read
    is exactly the cross-ticker race market_lock exists to close."""
    broker = CountingBalanceBroker()
    monkeypatch.setattr(state, "kis_broker", broker_stub(broker))

    async with portfolio_risk.cached_balance_fetches():  # enforcement_safe defaults False
        await portfolio_risk.real_shares_held("005930.KS")  # populates the cache
        await portfolio_risk.real_account_snapshot("domestic")
        await portfolio_risk.real_account_snapshot("domestic")

    assert broker.domestic_calls == 3  # real_shares_held once, real_account_snapshot fresh each time


async def test_enforcement_safe_does_not_cache_a_failure(monkeypatch):
    broker = CountingBalanceBroker(domestic_fail=True)
    monkeypatch.setattr(state, "kis_broker", broker_stub(broker))

    async with portfolio_risk.cached_balance_fetches(enforcement_safe=True):
        with pytest.raises(portfolio_risk.AccountUnavailable):
            await portfolio_risk.real_account_snapshot("domestic")
        broker.domestic_fail = False  # KIS recovers mid-scope
        snap = await portfolio_risk.real_account_snapshot("domestic")

    assert snap.total_value == pytest.approx(3_000_000.0)
    assert broker.domestic_calls == 2  # retried for real, not served a cached failure


async def test_enforcement_safe_scope_does_not_leak_across_holds(monkeypatch):
    broker = CountingBalanceBroker()
    monkeypatch.setattr(state, "kis_broker", broker_stub(broker))

    async with portfolio_risk.cached_balance_fetches(enforcement_safe=True):
        await portfolio_risk.real_shares_held("005930.KS")
        await portfolio_risk.real_account_snapshot("domestic")
    async with portfolio_risk.cached_balance_fetches(enforcement_safe=True):
        await portfolio_risk.real_account_snapshot("domestic")

    assert broker.domestic_calls == 2  # each hold fetched fresh, no cross-hold reuse


async def test_ledger_inquiry_count_tracks_only_real_broker_calls(monkeypatch):
    """FIX-6, 2026-09-03 id=1187 audit — verification counter for one
    market_lock hold's actual ledger inquiries (cache hits excluded)."""
    broker = CountingBalanceBroker()
    monkeypatch.setattr(state, "kis_broker", broker_stub(broker))

    async with portfolio_risk.cached_balance_fetches(enforcement_safe=True):
        assert portfolio_risk.ledger_inquiry_count() == 0
        await portfolio_risk.real_shares_held("005930.KS")  # 1 real call
        assert portfolio_risk.ledger_inquiry_count() == 1
        await portfolio_risk.real_account_snapshot("domestic")  # cache hit, no bump
        assert portfolio_risk.ledger_inquiry_count() == 1

    # Counter resets for a fresh scope.
    async with portfolio_risk.cached_balance_fetches(enforcement_safe=True):
        assert portfolio_risk.ledger_inquiry_count() == 0


# ---------------------------------------------------------------------------
# 2026-09-08 EGW00215 follow-up (id=1187 FIX-1 closed the domestic leg;
# the overseas leg had the identical duplicate-fetch shape until now) —
# real_equity_native's overseas branch used to call
# broker.fetch_overseas_present_balance directly instead of the cached
# sibling real_account_snapshot's own overseas branch already used, so one
# rebalance_to_target hold made the present-balance TR call twice (once via
# each function) even under enforcement_safe=True.
# ---------------------------------------------------------------------------


async def test_enforcement_safe_scope_collapses_overseas_present_balance_across_equity_and_snapshot(monkeypatch):
    """Core regression: real_equity_native (via resolve_equity_basis) and
    real_account_snapshot (via apply_live_portfolio_caps) both run inside
    ONE rebalance_to_target market_lock hold, in that order — see
    ledger.rebalance_to_target's own call sequence."""
    broker = CountingBalanceBroker()
    monkeypatch.setattr(state, "kis_broker", broker_stub(broker))

    async with portfolio_risk.cached_balance_fetches(enforcement_safe=True):
        await portfolio_risk.real_shares_held("AAPL")
        equity = await portfolio_risk.real_equity_native("overseas")
        snap = await portfolio_risk.real_account_snapshot("overseas")

    assert equity == pytest.approx(2_000.0)
    assert snap.total_value == pytest.approx(2_000.0)
    assert broker.overseas_calls == 1  # would be 2 without the cache
    assert broker.present_balance_calls == 1  # would be 2 without this fix — the actual bug


async def test_overseas_equity_basis_and_snapshot_total_agree_in_one_hold(monkeypatch):
    """Consistency check, not just a call-count pin: without the fix, a fill
    landing between the two present-balance calls could leave sizing
    (real_equity_native) and the concentration cap (real_account_snapshot)
    reading two different account states inside what should be one atomic
    market_lock view."""
    broker = CountingBalanceBroker()
    monkeypatch.setattr(state, "kis_broker", broker_stub(broker))

    async with portfolio_risk.cached_balance_fetches(enforcement_safe=True):
        equity = await portfolio_risk.real_equity_native("overseas")
        snap = await portfolio_risk.real_account_snapshot("overseas")

    assert equity == snap.total_value


async def test_ledger_inquiry_count_matches_real_wire_calls_overseas(monkeypatch):
    """Guards against the exact blind spot that let the duplicate survive
    undetected: the counter must equal the broker's own actual call tally,
    not just "whatever the counter happens to read" — a call site that
    forgets to route through a _cached_fetch_* helper bumps neither."""
    broker = CountingBalanceBroker()
    monkeypatch.setattr(state, "kis_broker", broker_stub(broker))

    async with portfolio_risk.cached_balance_fetches(enforcement_safe=True):
        await portfolio_risk.real_shares_held("AAPL")
        await portfolio_risk.real_equity_native("overseas")
        await portfolio_risk.real_account_snapshot("overseas")
        wire_calls = broker.overseas_calls + broker.present_balance_calls
        assert portfolio_risk.ledger_inquiry_count() == wire_calls == 2


async def test_without_scope_overseas_present_balance_still_fresh_each_call(monkeypatch):
    """Safety property this fix must not break: outside a
    cached_balance_fetches(enforcement_safe=True) scope, real_equity_native
    must keep fetching present-balance fresh every call."""
    broker = CountingBalanceBroker()
    monkeypatch.setattr(state, "kis_broker", broker_stub(broker))

    await portfolio_risk.real_equity_native("overseas")
    await portfolio_risk.real_equity_native("overseas")

    assert broker.present_balance_calls == 2


# ---------------------------------------------------------------------------
# cached_account_snapshots — the SEPARATE, market_lock-scoped cache for whole
# real_account_snapshot results (2026-08-10). ledger.rebalance_to_target
# reaches real_account_snapshot twice inside one lock hold
# (apply_live_portfolio_caps, then safety_switch.daily_loss_block via
# mirror_trade_to_kis); on the paper account's 1-req/sec throttle the
# duplicate added ~1s domestic / ~2s overseas of LOCK HOLD against a 10s
# contention budget. Safe only because the scope opens AFTER the lock is
# won — see the cache's own comment.
# ---------------------------------------------------------------------------


async def test_cached_account_snapshots_collapses_repeated_domestic_snapshots(monkeypatch):
    broker = CountingBalanceBroker()
    monkeypatch.setattr(state, "kis_broker", broker_stub(broker))

    async with portfolio_risk.cached_account_snapshots():
        first = await portfolio_risk.real_account_snapshot("domestic")
        second = await portfolio_risk.real_account_snapshot("domestic")

    assert first.total_value == second.total_value
    assert broker.domestic_calls == 1  # would be 2 without the cache


async def test_cached_account_snapshots_collapses_both_overseas_trs(monkeypatch):
    # Overseas needs TWO TRs per snapshot (balance + present-balance), so the
    # duplicate cost this removes is twice that of domestic.
    broker = CountingBalanceBroker()
    monkeypatch.setattr(state, "kis_broker", broker_stub(broker))

    async with portfolio_risk.cached_account_snapshots():
        await portfolio_risk.real_account_snapshot("overseas")
        await portfolio_risk.real_account_snapshot("overseas")

    assert broker.overseas_calls == 1
    assert broker.present_balance_calls == 1


async def test_cached_account_snapshots_keeps_markets_separate(monkeypatch):
    # Keyed per market — a domestic snapshot must never be served for an
    # overseas request, which would blend two books this module never mixes.
    broker = CountingBalanceBroker()
    monkeypatch.setattr(state, "kis_broker", broker_stub(broker))

    async with portfolio_risk.cached_account_snapshots():
        domestic = await portfolio_risk.real_account_snapshot("domestic")
        overseas = await portfolio_risk.real_account_snapshot("overseas")

    assert domestic.total_value == pytest.approx(3_000_000.0)  # KRW
    assert overseas.total_value == pytest.approx(2_000.0)  # USD
    assert broker.domestic_calls == 1
    assert broker.overseas_calls == 1


async def test_real_account_snapshot_unaffected_without_the_scope(monkeypatch):
    """Every OTHER caller (place_manual_order's gates, force_exit_position,
    the periodic loops) never opens this scope and must keep fetching fresh
    — the dedupe applies only inside one held market_lock."""
    broker = CountingBalanceBroker()
    monkeypatch.setattr(state, "kis_broker", broker_stub(broker))

    await portfolio_risk.real_account_snapshot("domestic")
    await portfolio_risk.real_account_snapshot("domestic")

    assert broker.domestic_calls == 2  # no scope open — no merging


async def test_cached_account_snapshots_does_not_cache_a_failure(monkeypatch):
    """AccountUnavailable is fail-closed per call and must never be pinned
    for the rest of the scope: a transient KIS failure followed by a
    recovery has to be able to succeed, not keep raising because the first
    attempt's outcome got cached."""
    broker = CountingBalanceBroker(domestic_fail=True)
    monkeypatch.setattr(state, "kis_broker", broker_stub(broker))

    async with portfolio_risk.cached_account_snapshots():
        with pytest.raises(portfolio_risk.AccountUnavailable):
            await portfolio_risk.real_account_snapshot("domestic")
        broker.domestic_fail = False  # KIS recovers mid-scope
        snap = await portfolio_risk.real_account_snapshot("domestic")

    assert snap.total_value == pytest.approx(3_000_000.0)
    assert broker.domestic_calls == 2  # retried for real, not served a cached failure


async def test_cached_account_snapshots_scope_does_not_leak_across_holds(monkeypatch):
    """Two separate lock holds are two separate windows — the second must
    re-read, or it would miss whatever the first hold's own order changed."""
    broker = CountingBalanceBroker()
    monkeypatch.setattr(state, "kis_broker", broker_stub(broker))

    async with portfolio_risk.cached_account_snapshots():
        await portfolio_risk.real_account_snapshot("domestic")
    async with portfolio_risk.cached_account_snapshots():
        await portfolio_risk.real_account_snapshot("domestic")

    assert broker.domestic_calls == 2


# ---------------------------------------------------------------------------
# per-market high-water-mark isolation
# ---------------------------------------------------------------------------


class _AcquireCM:
    def __init__(self, conn):
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, *exc):
        return False


class FakeHwmConn:
    def __init__(self):
        # keyed (market, account_mode, user_id) — user_id joined the unique
        # index in the multi-user migration Phase 4.
        self.rows: dict[tuple[str, str, str], float] = {}

    async def execute(self, query, *args):
        if query.strip().startswith("DELETE"):
            self.rows.pop(args, None)  # args = (market, account_mode, user_id)
            return
        market, account_mode, user_id, total = args  # INSERT ... ON CONFLICT
        key = (market, account_mode, user_id)
        self.rows[key] = max(self.rows.get(key, total), total)

    async def fetchrow(self, query, *args):
        key = args  # (market, account_mode, user_id)
        if key not in self.rows:
            return None
        return {"high_water_mark": self.rows[key]}


class FakeHwmPool:
    def __init__(self, conn):
        self._conn = conn

    def acquire(self):
        return _AcquireCM(self._conn)


@pytest.fixture
def fake_hwm_conn(monkeypatch):
    conn = FakeHwmConn()

    async def fake_ready_pool():
        return FakeHwmPool(conn)

    monkeypatch.setattr(portfolio_risk, "_ready_pool", fake_ready_pool)
    return conn


async def test_current_drawdown_fraction_tracks_each_market_independently(fake_hwm_conn):
    # Domestic sets a high-water mark of 10,000, then drops 20% — should trip.
    await portfolio_risk.current_drawdown_fraction("domestic", 10_000.0)
    domestic_frac = await portfolio_risk.current_drawdown_fraction("domestic", 8_000.0)
    assert domestic_frac == pytest.approx(0.2)

    # Overseas has never been recorded — first call is always a "new high",
    # 0.0 drawdown, and must not be affected by domestic's HWM above.
    overseas_frac = await portfolio_risk.current_drawdown_fraction("overseas", 500.0)
    assert overseas_frac == 0.0


async def test_current_drawdown_fraction_tracks_each_account_mode_independently(fake_hwm_conn, monkeypatch):
    # 2026-08-04: a high-water mark built up during paper testing must never
    # apply to real trading (or vice versa) — the concrete bug this closes:
    # switching kis_account_mode used to inherit whatever peak the OTHER
    # mode had reached, tripping the drawdown breaker for real trading from
    # its very first live snapshot for no real reason.
    monkeypatch.setattr(portfolio_risk.settings, "kis_account_mode", "paper")
    await portfolio_risk.current_drawdown_fraction("domestic", 10_000.0)
    paper_frac = await portfolio_risk.current_drawdown_fraction("domestic", 8_000.0)
    assert paper_frac == pytest.approx(0.2)

    # Real has never been recorded under this same market — must be a fresh
    # baseline (0.0 drawdown) even though paper's HWM for "domestic" is 10,000
    # and this real total (500) is far below it.
    monkeypatch.setattr(portfolio_risk.settings, "kis_account_mode", "real")
    real_frac = await portfolio_risk.current_drawdown_fraction("domestic", 500.0)
    assert real_frac == 0.0


async def test_reset_high_water_mark_only_clears_the_current_account_mode(fake_hwm_conn, monkeypatch):
    monkeypatch.setattr(portfolio_risk.settings, "kis_account_mode", "paper")
    await portfolio_risk.current_drawdown_fraction("domestic", 10_000.0)
    monkeypatch.setattr(portfolio_risk.settings, "kis_account_mode", "real")
    await portfolio_risk.current_drawdown_fraction("domestic", 5_000.0)

    monkeypatch.setattr(portfolio_risk.settings, "kis_account_mode", "paper")
    await portfolio_risk.reset_high_water_mark("domestic")

    # Paper's HWM is gone — a much lower total is now a fresh high, not a drawdown.
    assert await portfolio_risk.current_drawdown_fraction("domestic", 1_000.0) == 0.0

    # Real's HWM (5,000) must be untouched by paper's reset.
    monkeypatch.setattr(portfolio_risk.settings, "kis_account_mode", "real")
    real_frac = await portfolio_risk.current_drawdown_fraction("domestic", 2_500.0)
    assert real_frac == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# market_lock — Postgres advisory lock (2026-08-04, cross-process
# replacement for the old in-memory asyncio.Lock). See project memory:
# user-account-mode-db-separation-design.
# ---------------------------------------------------------------------------


class FakeAdvisoryConn:
    def __init__(self, lock_results: list[bool] | None = None):
        self.calls: list[tuple] = []  # execute() calls - only pg_advisory_unlock now
        self.fetchval_calls: list[tuple] = []  # fetchval() calls - pg_try_advisory_lock
        self._lock_results = iter(lock_results if lock_results is not None else [True] * 100)

    async def execute(self, query, *args):
        self.calls.append((query, args))

    async def fetchval(self, query, *args):
        self.fetchval_calls.append((query, args))
        return next(self._lock_results)


class FakeAdvisoryPool:
    def __init__(self, conn):
        self._conn = conn
        self.released: list = []

    async def acquire(self):
        return self._conn

    async def release(self, conn):
        self.released.append(conn)


@pytest.fixture
def fake_advisory_conn(monkeypatch):
    conn = FakeAdvisoryConn()
    pool = FakeAdvisoryPool(conn)

    async def fake_ready_pool():
        return pool

    monkeypatch.setattr(portfolio_risk, "_ready_pool", fake_ready_pool)
    conn.pool = pool  # for release-assertion convenience
    return conn


async def test_market_lock_acquires_and_releases_the_advisory_lock(fake_advisory_conn, monkeypatch):
    monkeypatch.setattr(portfolio_risk.settings, "kis_account_mode", "paper")

    async with portfolio_risk.market_lock("domestic"):
        assert len(fake_advisory_conn.fetchval_calls) == 1
        lock_query, lock_args = fake_advisory_conn.fetchval_calls[0]
        assert "pg_try_advisory_lock" in lock_query
        expected = portfolio_risk._lock_key2("domestic", "paper", get_current_user_id())
        assert lock_args == (portfolio_risk._ADVISORY_LOCK_NAMESPACE, expected)

    unlock_query, unlock_args = fake_advisory_conn.calls[0]
    assert "pg_advisory_unlock" in unlock_query
    assert unlock_args == fake_advisory_conn.fetchval_calls[0][1]
    assert fake_advisory_conn.pool.released == [fake_advisory_conn]


async def test_market_lock_uses_a_distinct_key_per_market_and_account_mode(monkeypatch):
    seen_keys = set()
    for market, mode in [("domestic", "paper"), ("domestic", "real"), ("overseas", "paper"), ("overseas", "real")]:
        conn = FakeAdvisoryConn()
        pool = FakeAdvisoryPool(conn)

        async def fake_ready_pool(pool=pool):
            return pool

        monkeypatch.setattr(portfolio_risk, "_ready_pool", fake_ready_pool)
        monkeypatch.setattr(portfolio_risk.settings, "kis_account_mode", mode)
        async with portfolio_risk.market_lock(market):
            pass
        _, (_, key2) = conn.fetchval_calls[0]
        seen_keys.add(key2)
    assert len(seen_keys) == 4  # no two of the four combinations collide


async def test_market_lock_degrades_quietly_when_db_unreachable(monkeypatch):
    async def broken_ready_pool():
        raise OSError("connection refused")

    monkeypatch.setattr(portfolio_risk, "_ready_pool", broken_ready_pool)
    async with portfolio_risk.market_lock("domestic"):
        pass  # must not raise — degrades to no cross-process serialization


async def test_market_lock_raises_and_releases_connection_when_contended(monkeypatch):
    """Real incident this fixes (2026-08-06): pg_advisory_lock (blocking)
    combined with the 125s database_command_timeout_seconds meant a
    contended lock silently fell into the fail-open branch after up to two
    minutes, giving up the exact cross-process serialization it exists for.
    pg_try_advisory_lock (non-blocking) polled against its own short
    portfolio_risk_lock_wait_seconds budget must instead raise
    MarketLockContended - and still return the connection to the pool,
    not leak it."""
    monkeypatch.setattr(portfolio_risk.settings, "kis_account_mode", "paper")
    monkeypatch.setattr(portfolio_risk.settings, "portfolio_risk_lock_wait_seconds", 0.05)
    conn = FakeAdvisoryConn(lock_results=[False] * 100)  # never acquires
    pool = FakeAdvisoryPool(conn)

    async def fake_ready_pool():
        return pool

    monkeypatch.setattr(portfolio_risk, "_ready_pool", fake_ready_pool)

    with pytest.raises(portfolio_risk.MarketLockContended):
        async with portfolio_risk.market_lock("domestic"):
            pass

    assert pool.released == [conn]  # returned to the pool, not leaked
    assert conn.calls == []  # never held the lock, so no pg_advisory_unlock


async def test_market_lock_releases_connection_when_lock_query_errors(monkeypatch):
    """Regression: previously the connection was only released once the
    lock was actually held - a DB error DURING acquisition (after the pool
    hand-out but before the lock query succeeds) leaked the connection.
    With database_pool_max_size=5, 5 such leaks stalled every trade in this
    market+account_mode permanently."""
    monkeypatch.setattr(portfolio_risk.settings, "kis_account_mode", "paper")

    class BrokenFetchvalConn(FakeAdvisoryConn):
        async def fetchval(self, query, *args):
            raise OSError("connection reset")

    conn = BrokenFetchvalConn()
    pool = FakeAdvisoryPool(conn)

    async def fake_ready_pool():
        return pool

    monkeypatch.setattr(portfolio_risk, "_ready_pool", fake_ready_pool)

    async with portfolio_risk.market_lock("domestic"):
        pass  # must not raise - degrades to no cross-process serialization

    assert pool.released == [conn]  # the actual fix under test


# ---------------------------------------------------------------------------
# apply_live_portfolio_caps — fail-closed on AccountUnavailable
# ---------------------------------------------------------------------------


async def test_apply_live_portfolio_caps_blocks_exposure_increase_when_account_unavailable(monkeypatch):
    async def broken_snapshot(market):
        raise portfolio_risk.AccountUnavailable("broker not connected")

    monkeypatch.setattr(portfolio_risk, "real_account_snapshot", broken_snapshot)
    target_shares, note = await portfolio_risk.apply_live_portfolio_caps("005930.KS", 100, 70_000.0, 50)
    assert target_shares == 50  # held at the existing position, not increased
    assert note is not None and "unavailable" in note


async def test_apply_live_portfolio_caps_allows_exposure_decrease_when_account_unavailable(monkeypatch):
    async def broken_snapshot(market):
        raise portfolio_risk.AccountUnavailable("broker not connected")

    monkeypatch.setattr(portfolio_risk, "real_account_snapshot", broken_snapshot)
    # A SELL (target below current) must still pass straight through — only
    # exposure-INCREASING trades are fail-closed.
    target_shares, note = await portfolio_risk.apply_live_portfolio_caps("005930.KS", 20, 70_000.0, 50)
    assert target_shares == 20
    assert note is None


async def test_apply_live_portfolio_caps_keeps_domestic_price_in_native_krw(monkeypatch):
    """Regression test for a real bug: this function used to convert the
    ticker's native KRW price to USD before comparing it against
    real_account_snapshot's totals — but since that snapshot is now ALSO
    native KRW for domestic (never combined with overseas), converting only
    `price` (or only the snapshot) is exactly the ~1,400x unit mismatch that
    made almost any domestic increase look like it "reaches ~100% of
    aggregate equity" and get capped back to the existing position.
    FakeRealBalanceBroker: domestic total_value=3,000,000 KRW, existing
    005930.KS=700,000 KRW (10 shares @ 70,000). A modest 10->12 share
    increase (well within a 30% cap, entirely in KRW) must be ALLOWED."""
    monkeypatch.setattr(state, "kis_broker", broker_stub(FakeRealBalanceBroker()))

    async def unexpected_fx_rate(currency):
        raise AssertionError("native-KRW comparison must not need an FX rate")

    async def no_drawdown(market, total_value):
        return 0.0  # isolate the concentration-cap path being tested

    monkeypatch.setattr(portfolio_risk, "get_usd_fx_rate", unexpected_fx_rate)
    monkeypatch.setattr(portfolio_risk, "current_drawdown_fraction", no_drawdown)
    monkeypatch.setattr(portfolio_risk.settings, "portfolio_concentration_max_pct", 0.30)

    target_shares, note = await portfolio_risk.apply_live_portfolio_caps("005930.KS", 12, 70_000.0, 10)

    # projected (KRW) = 12 * 70_000 = 840_000; other_total = 3_000_000-700_000 = 2_300_000
    # concentration = 840_000 / (2_300_000+840_000) = 26.75% <= 30% cap -> uncapped.
    # A KRW/USD unit mismatch here would have computed ~99.97% and capped to 10.
    assert target_shares == 12
    assert note is None


async def test_apply_live_portfolio_caps_uses_regime_aware_floor_for_domestic(monkeypatch):
    """apply_live_portfolio_caps must resolve the Faber-signal-aware cash
    floor (app.execution.market_regime) for a domestic ticker and pass it
    into apply_portfolio_caps, rather than always using the static
    settings.portfolio_cash_floor_pct."""
    monkeypatch.setattr(state, "kis_broker", broker_stub(FakeRealBalanceBroker()))

    async def no_drawdown(market, total_value):
        return 0.0

    async def risk_off_floor():
        return 0.35  # far above the static default (0.20) — proves this path was used

    monkeypatch.setattr(portfolio_risk, "current_drawdown_fraction", no_drawdown)
    monkeypatch.setattr(portfolio_risk.settings, "portfolio_concentration_max_pct", 1.0)
    monkeypatch.setattr(portfolio_risk.settings, "portfolio_correlation_bucket_max_pct", 1.0)
    monkeypatch.setattr(portfolio_risk.settings, "portfolio_cash_floor_pct", 0.20)

    from app.execution import market_regime
    monkeypatch.setattr(market_regime, "domestic_cash_floor_pct", risk_off_floor)

    # FakeRealBalanceBroker domestic: total_value=3,000,000, existing
    # 005930.KS=700,000 -> cash_before=2,300,000. Buying to 30 shares would
    # leave projected_cash=900,000 — under the static 20% floor
    # (cash_floor_value=600,000) that's NOT capped at all (30 stays 30), but
    # under the mocked 35% Risk-Off floor (cash_floor_value=1,050,000,
    # max_incremental=1,250,000 -> +17 shares -> 27 total) it IS capped.
    # Getting exactly 27 (not 30) proves the regime-aware value was used.
    target_shares, note = await portfolio_risk.apply_live_portfolio_caps("005930.KS", 30, 70_000.0, 10)
    assert target_shares == 27
    assert note is not None and "cash floor" in note.lower()


async def test_apply_live_portfolio_caps_uses_overseas_regime_floor_not_domestic(monkeypatch):
    """Overseas has its own Faber signal (SPY-based) — must call
    market_regime.overseas_cash_floor_pct(), never the domestic one."""
    monkeypatch.setattr(state, "kis_broker", broker_stub(FakeRealBalanceBroker()))

    async def no_drawdown(market, total_value):
        return 0.0

    async def fail_if_called():
        raise AssertionError("overseas must consult its OWN regime signal, not the domestic one")

    async def risk_off_floor():
        return 0.35  # far above the static default (0.20) — proves this path was used

    monkeypatch.setattr(portfolio_risk, "current_drawdown_fraction", no_drawdown)
    monkeypatch.setattr(portfolio_risk.settings, "portfolio_concentration_max_pct", 1.0)
    monkeypatch.setattr(portfolio_risk.settings, "portfolio_correlation_bucket_max_pct", 1.0)
    monkeypatch.setattr(portfolio_risk.settings, "portfolio_cash_floor_pct", 0.20)
    monkeypatch.setattr(portfolio_risk.settings, "trade_allowlist_overseas", ["AAPL"])

    async def no_network_clusters(tickers, as_of=None):
        return {}

    monkeypatch.setattr(portfolio_risk, "compute_correlation_clusters", no_network_clusters)

    from app.execution import market_regime
    monkeypatch.setattr(market_regime, "domestic_cash_floor_pct", fail_if_called)
    monkeypatch.setattr(market_regime, "overseas_cash_floor_pct", risk_off_floor)

    # FakeRealBalanceBroker overseas: total_value=2,000, existing AAPL=1,500
    # -> cash_before=500. Buying to 6 shares leaves projected_cash=200,
    # under BOTH the static 20% floor (400) and the mocked 35% floor (700)
    # here, so this alone can't distinguish them — the real proof is that
    # `fail_if_called` (wired to the domestic function) never fires.
    target_shares, note = await portfolio_risk.apply_live_portfolio_caps("AAPL", 6, 300.0, 5)
    assert target_shares == 5
    assert note is not None and "cash floor" in note.lower()


# ---------------------------------------------------------------------------
# apply_portfolio_caps — max open positions
# ---------------------------------------------------------------------------


def test_max_open_positions_blocks_new_entry_at_cap(monkeypatch):
    monkeypatch.setattr(portfolio_risk.settings, "max_open_positions_domestic", 2)
    monkeypatch.setattr(portfolio_risk.settings, "portfolio_cash_floor_pct", 0.0)
    monkeypatch.setattr(portfolio_risk.settings, "portfolio_concentration_max_pct", 1.0)
    monkeypatch.setattr(portfolio_risk.settings, "portfolio_correlation_bucket_max_pct", 1.0)
    # 2 tickers already open (nonzero value), cap is 2 -> a brand-new third
    # ticker (position_shares_before=0) must be blocked entirely.
    snapshot = portfolio_risk.PortfolioSnapshot(
        per_ticker_value={"005930.KS": 100_000.0, "000660.KS": 100_000.0}, total_value=1_000_000.0
    )
    target_shares, note = portfolio_risk.apply_portfolio_caps(
        "035420.KS", 10, 1_000.0, 0, snapshot, clusters={}, current_drawdown_frac=0.0
    )
    assert target_shares == 0
    assert note is not None and "max open positions" in note.lower()


def test_max_open_positions_allows_adding_to_an_existing_position(monkeypatch):
    monkeypatch.setattr(portfolio_risk.settings, "max_open_positions_domestic", 2)
    monkeypatch.setattr(portfolio_risk.settings, "portfolio_cash_floor_pct", 0.0)
    monkeypatch.setattr(portfolio_risk.settings, "portfolio_concentration_max_pct", 1.0)
    monkeypatch.setattr(portfolio_risk.settings, "portfolio_correlation_bucket_max_pct", 1.0)
    # Already at the cap (2 open), but this ticker IS one of the two —
    # adding to it must never be blocked by this cap.
    snapshot = portfolio_risk.PortfolioSnapshot(
        per_ticker_value={"005930.KS": 100_000.0, "000660.KS": 100_000.0}, total_value=1_000_000.0
    )
    target_shares, note = portfolio_risk.apply_portfolio_caps(
        "005930.KS", 150, 1_000.0, 100, snapshot, clusters={}, current_drawdown_frac=0.0
    )
    assert target_shares == 150
    assert note is None


def test_max_open_positions_allows_new_entry_under_cap(monkeypatch):
    monkeypatch.setattr(portfolio_risk.settings, "max_open_positions_domestic", 3)
    monkeypatch.setattr(portfolio_risk.settings, "portfolio_cash_floor_pct", 0.0)
    monkeypatch.setattr(portfolio_risk.settings, "portfolio_concentration_max_pct", 1.0)
    monkeypatch.setattr(portfolio_risk.settings, "portfolio_correlation_bucket_max_pct", 1.0)
    snapshot = portfolio_risk.PortfolioSnapshot(
        per_ticker_value={"005930.KS": 100_000.0, "000660.KS": 100_000.0}, total_value=1_000_000.0
    )
    target_shares, note = portfolio_risk.apply_portfolio_caps(
        "035420.KS", 10, 1_000.0, 0, snapshot, clusters={}, current_drawdown_frac=0.0
    )
    assert target_shares == 10
    assert note is None


def test_max_open_positions_uses_overseas_setting_for_overseas_ticker(monkeypatch):
    monkeypatch.setattr(portfolio_risk.settings, "max_open_positions_domestic", 100)  # must NOT be used
    monkeypatch.setattr(portfolio_risk.settings, "max_open_positions_overseas", 2)
    monkeypatch.setattr(portfolio_risk.settings, "portfolio_cash_floor_pct", 0.0)
    monkeypatch.setattr(portfolio_risk.settings, "portfolio_concentration_max_pct", 1.0)
    monkeypatch.setattr(portfolio_risk.settings, "portfolio_correlation_bucket_max_pct", 1.0)
    snapshot = portfolio_risk.PortfolioSnapshot(
        per_ticker_value={"AAPL": 1_500.0, "MSFT": 1_500.0}, total_value=10_000.0
    )
    target_shares, note = portfolio_risk.apply_portfolio_caps(
        "GOOGL", 5, 300.0, 0, snapshot, clusters={}, current_drawdown_frac=0.0
    )
    assert target_shares == 0  # blocked by the overseas cap (2), not the domestic one (100)
    assert note is not None and "max open positions" in note.lower()


def test_max_open_positions_never_blocks_a_sell(monkeypatch):
    monkeypatch.setattr(portfolio_risk.settings, "max_open_positions_domestic", 1)
    snapshot = portfolio_risk.PortfolioSnapshot(
        per_ticker_value={"005930.KS": 100_000.0}, total_value=1_000_000.0
    )
    target_shares, note = portfolio_risk.apply_portfolio_caps(
        "005930.KS", 50, 1_000.0, 100, snapshot, clusters={}, current_drawdown_frac=0.0
    )
    assert target_shares == 50
    assert note is None


# ---------------------------------------------------------------------------
# apply_portfolio_caps — cash floor
# ---------------------------------------------------------------------------


def test_cash_floor_reduces_buy_that_would_breach_it(monkeypatch):
    monkeypatch.setattr(portfolio_risk.settings, "portfolio_cash_floor_pct", 0.20)
    monkeypatch.setattr(portfolio_risk.settings, "portfolio_concentration_max_pct", 1.0)
    monkeypatch.setattr(portfolio_risk.settings, "portfolio_correlation_bucket_max_pct", 1.0)
    # total_value=1,000,000; existing ticker value=100,000 -> cash=900,000 (90%).
    # Buying up to 800 shares @1,000 = 800,000 would leave cash=100,000 (10%),
    # under the 20% floor -> must be capped so cash stays at exactly 200,000 (20%).
    snapshot = portfolio_risk.PortfolioSnapshot(
        per_ticker_value={"005930.KS": 100_000.0}, total_value=1_000_000.0
    )
    target_shares, note = portfolio_risk.apply_portfolio_caps(
        "005930.KS", 800, 1_000.0, 100, snapshot, clusters={}, current_drawdown_frac=0.0
    )
    # max_incremental = cash_before(900,000) - floor_value(200,000) = 700,000 -> +700 shares -> 800 total
    assert target_shares == 800
    assert note is None


def test_cash_floor_caps_buy_below_floor(monkeypatch):
    monkeypatch.setattr(portfolio_risk.settings, "portfolio_cash_floor_pct", 0.20)
    monkeypatch.setattr(portfolio_risk.settings, "portfolio_concentration_max_pct", 1.0)
    monkeypatch.setattr(portfolio_risk.settings, "portfolio_correlation_bucket_max_pct", 1.0)
    # Same account as above, but the target (1,200 shares = 1,200,000) would
    # push cash to -100,000 (deeply negative) — must be capped to keep cash
    # at the 20% floor (200,000), same 800-share cap as the test above.
    snapshot = portfolio_risk.PortfolioSnapshot(
        per_ticker_value={"005930.KS": 100_000.0}, total_value=1_000_000.0
    )
    target_shares, note = portfolio_risk.apply_portfolio_caps(
        "005930.KS", 1_200, 1_000.0, 100, snapshot, clusters={}, current_drawdown_frac=0.0
    )
    assert target_shares == 800
    assert note is not None and "cash floor" in note.lower()


def test_cash_floor_does_not_affect_a_sell(monkeypatch):
    monkeypatch.setattr(portfolio_risk.settings, "portfolio_cash_floor_pct", 0.20)
    snapshot = portfolio_risk.PortfolioSnapshot(
        per_ticker_value={"005930.KS": 100_000.0}, total_value=1_000_000.0
    )
    # target (50) below position_shares_before (100) — exposure-decreasing,
    # must pass straight through regardless of the cash floor.
    target_shares, note = portfolio_risk.apply_portfolio_caps(
        "005930.KS", 50, 1_000.0, 100, snapshot, clusters={}, current_drawdown_frac=0.0
    )
    assert target_shares == 50
    assert note is None


# ---------------------------------------------------------------------------
# _log_cap_reduction — diagnostic logging added 2026-08-14 after the
# 98.1%-Hold baseline backtest turned out to be caused by two silent
# portfolio-cap bugs. These tests are about the LOG OUTPUT, not the returned
# `target_shares`/`note` (already covered by the tests above) — `caplog`,
# same convention as tests/agents/test_debater.py.
# ---------------------------------------------------------------------------


def test_full_wipe_cap_logs_at_warning(monkeypatch, caplog):
    """A cash-floor cap that eats the ENTIRE intended increase (capped
    target == position_shares_before, i.e. the increment is reduced all the
    way to 0) must log at WARNING — this is the exact shape the
    baseline-backtest bug produced (22 shares -> 0)."""
    monkeypatch.setattr(portfolio_risk.settings, "portfolio_concentration_max_pct", 1.0)
    monkeypatch.setattr(portfolio_risk.settings, "portfolio_correlation_bucket_max_pct", 1.0)
    monkeypatch.setattr(portfolio_risk.settings, "portfolio_cash_floor_pct", 0.9)
    # cash_before = 1,000,000 - 100,000 = 900,000, already exactly at the 90%
    # floor (900,000) — any further buy pushes cash below it, so the entire
    # intended increase (100 -> 200 shares) is capped back to the existing 100.
    snapshot = portfolio_risk.PortfolioSnapshot(
        per_ticker_value={"005930.KS": 100_000.0}, total_value=1_000_000.0
    )
    with caplog.at_level("INFO", logger="app.execution.portfolio_risk"):
        target_shares, note = portfolio_risk.apply_portfolio_caps(
            "005930.KS", 200, 1_000.0, 100, snapshot, clusters={}, current_drawdown_frac=0.0
        )
    assert target_shares == 100
    assert note is not None
    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert any("cash floor cap reduced target 200 -> 100" in r.message for r in warnings)


def test_partial_reduction_cap_logs_at_info(monkeypatch, caplog):
    """A cap that merely shrinks (not wipes) the increase logs at INFO, not
    WARNING — same distinction the "wipe" test above proves the other way."""
    monkeypatch.setattr(portfolio_risk.settings, "portfolio_cash_floor_pct", 0.20)
    monkeypatch.setattr(portfolio_risk.settings, "portfolio_concentration_max_pct", 1.0)
    monkeypatch.setattr(portfolio_risk.settings, "portfolio_correlation_bucket_max_pct", 1.0)
    snapshot = portfolio_risk.PortfolioSnapshot(
        per_ticker_value={"005930.KS": 100_000.0}, total_value=1_000_000.0
    )
    with caplog.at_level("INFO", logger="app.execution.portfolio_risk"):
        target_shares, note = portfolio_risk.apply_portfolio_caps(
            "005930.KS", 1_200, 1_000.0, 100, snapshot, clusters={}, current_drawdown_frac=0.0
        )
    assert target_shares == 800  # partial reduction, still > position_shares_before (100)
    assert note is not None
    cap_records = [r for r in caplog.records if "cash floor cap reduced target" in r.message]
    assert len(cap_records) == 1
    assert cap_records[0].levelname == "INFO"


def test_bug1_signature_cash_counted_as_position_value(monkeypatch, caplog):
    """Bug 1 signature: position_shares_before == 0 (flat) but the
    snapshot's own per_ticker_value entry for this ticker is already > 0 —
    the exact shape that made a 0-share position read as "50% of the
    portfolio" in the baseline-backtest bug. Must fire a dedicated WARNING
    distinct from the ordinary cap-reduction log line."""
    monkeypatch.setattr(portfolio_risk.settings, "portfolio_concentration_max_pct", 0.01)
    monkeypatch.setattr(portfolio_risk.settings, "portfolio_correlation_bucket_max_pct", 1.0)
    monkeypatch.setattr(portfolio_risk.settings, "portfolio_cash_floor_pct", 0.0)
    # Flat (0 shares) but the snapshot already attributes this ticker 500,000
    # of value — should never happen if cash/position accounting is correct.
    snapshot = portfolio_risk.PortfolioSnapshot(
        per_ticker_value={"005930.KS": 500_000.0}, total_value=1_000_000.0
    )
    with caplog.at_level("WARNING", logger="app.execution.portfolio_risk"):
        portfolio_risk.apply_portfolio_caps(
            "005930.KS", 22, 1_000.0, 0, snapshot, clusters={}, current_drawdown_frac=0.0
        )
    assert any(
        "SUSPECTED BUG (cash counted as position value)" in r.message for r in caplog.records
    )


def test_bug2_signature_currency_unit_mismatch(monkeypatch, caplog):
    """Bug 2 signature: a KRW-scale price against a USD-scale (or otherwise
    tiny) snapshot total makes the projected value dwarf the whole
    portfolio — the exact shape of the KRW/USD unit mismatch that also
    zeroed out targets in the baseline-backtest bug. Must fire a dedicated
    WARNING distinct from the ordinary cap-reduction log line."""
    monkeypatch.setattr(portfolio_risk.settings, "portfolio_concentration_max_pct", 0.30)
    monkeypatch.setattr(portfolio_risk.settings, "portfolio_correlation_bucket_max_pct", 1.0)
    monkeypatch.setattr(portfolio_risk.settings, "portfolio_cash_floor_pct", 0.0)
    # snapshot.total_value is USD-scale ($1,000) while price is KRW-scale
    # (₩70,000) — 22 shares alone is ₩1,540,000, wildly >20x the $1,000 total.
    snapshot = portfolio_risk.PortfolioSnapshot(per_ticker_value={}, total_value=1_000.0)
    with caplog.at_level("WARNING", logger="app.execution.portfolio_risk"):
        portfolio_risk.apply_portfolio_caps(
            "005930.KS", 22, 70_000.0, 0, snapshot, clusters={}, current_drawdown_frac=0.0
        )
    assert any(
        "SUSPECTED BUG (currency-unit mismatch)" in r.message for r in caplog.records
    )


def test_no_cap_triggered_logs_nothing(monkeypatch, caplog):
    """When no cap actually binds, none of the new diagnostic log lines
    (cap-reduction or either bug signature) should appear at all."""
    monkeypatch.setattr(portfolio_risk.settings, "portfolio_concentration_max_pct", 1.0)
    monkeypatch.setattr(portfolio_risk.settings, "portfolio_correlation_bucket_max_pct", 1.0)
    monkeypatch.setattr(portfolio_risk.settings, "portfolio_cash_floor_pct", 0.0)
    snapshot = portfolio_risk.PortfolioSnapshot(
        per_ticker_value={"005930.KS": 100_000.0}, total_value=1_000_000.0
    )
    with caplog.at_level("INFO", logger="app.execution.portfolio_risk"):
        target_shares, note = portfolio_risk.apply_portfolio_caps(
            "005930.KS", 150, 1_000.0, 100, snapshot, clusters={}, current_drawdown_frac=0.0
        )
    assert target_shares == 150
    assert note is None
    assert caplog.records == []


# ---------------------------------------------------------------------------
# portfolio_snapshot — bug 1 fix (2026-08-14): cash must never be counted as
# a ticker's own position value. per_ticker_value holds POSITION value only
# (shares * price); cash is summed across every ticker and folded into
# total_value exactly once. See _ticker_usd_position_and_cash's docstring.
# ---------------------------------------------------------------------------


def _patch_snapshot_deps(
    monkeypatch,
    positions: dict[str, Position],
    prices: dict[str, float],
    currencies: dict[str, str] | None = None,
    fx_rates: dict[str, float] | None = None,
    raise_for: set[str] | None = None,
):
    currencies = currencies or {}
    fx_rates = fx_rates or {}
    raise_for = raise_for or set()

    async def fake_get_position(ticker):
        if ticker in raise_for:
            raise RuntimeError(f"simulated price/DB failure for {ticker}")
        return positions[ticker]

    async def fake_get_last_close(ticker, as_of=None):
        return prices[ticker]

    async def fake_get_ticker_currency(ticker):
        return currencies.get(ticker, "USD")

    async def fake_get_usd_fx_rate(currency):
        return fx_rates.get(currency, 1.0)

    monkeypatch.setattr(ledger_module, "get_position", fake_get_position)
    monkeypatch.setattr(portfolio_risk, "get_last_close", fake_get_last_close)
    monkeypatch.setattr(portfolio_risk, "get_ticker_currency", fake_get_ticker_currency)
    monkeypatch.setattr(portfolio_risk, "get_usd_fx_rate", fake_get_usd_fx_rate)


async def test_portfolio_snapshot_all_flat_reports_zero_position_value(monkeypatch):
    """Core regression: two tickers, both 0 shares / $100,000 cash each. The
    old bug reported each ticker's per_ticker_value as its own cash (i.e.
    "50% of the book" for a flat position) — must now be exactly 0."""
    _patch_snapshot_deps(
        monkeypatch,
        positions={"AAPL": Position(cash=100_000.0, shares=0), "MSFT": Position(cash=100_000.0, shares=0)},
        prices={"AAPL": 200.0, "MSFT": 400.0},
    )
    snap = await portfolio_risk.portfolio_snapshot(["AAPL", "MSFT"])
    assert snap.per_ticker_value == {"AAPL": 0.0, "MSFT": 0.0}
    assert snap.total_value == pytest.approx(200_000.0)


async def test_portfolio_snapshot_total_value_unchanged_by_the_fix(monkeypatch):
    """Mixed state (one ticker holding shares + cash, one all-cash) — the
    grand total must come out identical to the old (buggy) sum-per-ticker
    total, since Σ(cash+pos) == Σpos + Σcash algebraically. Only the
    per-ticker attribution changes, not total_value."""
    _patch_snapshot_deps(
        monkeypatch,
        positions={"AAPL": Position(cash=50_000.0, shares=10), "MSFT": Position(cash=100_000.0, shares=0)},
        prices={"AAPL": 50.0, "MSFT": 400.0},
    )
    snap = await portfolio_risk.portfolio_snapshot(["AAPL", "MSFT"])
    assert snap.per_ticker_value == {"AAPL": pytest.approx(500.0), "MSFT": 0.0}
    # Old (buggy) total: AAPL market_value(50,000+500) + MSFT market_value(100,000+0) = 150,500.
    # New total: sum(position_usd)=500 + sum(cash_usd)=150,000 = 150,500 — identical.
    assert snap.total_value == pytest.approx(150_500.0)


async def test_portfolio_snapshot_unblocks_cash_floor_for_flat_tickers(monkeypatch):
    """Before the fix, cash_before = total_value - sum(per_ticker_value) was
    definitionally 0 for every backtest snapshot (cash was already folded
    into every ticker's own bucket), so ANY buy tripped the cash floor and
    got zeroed. With the fix, cash_before correctly reflects the real
    uninvested cash and a normal-sized buy is no longer wiped to 0."""
    _patch_snapshot_deps(
        monkeypatch,
        positions={"AAPL": Position(cash=50_000.0, shares=10), "MSFT": Position(cash=100_000.0, shares=0)},
        prices={"AAPL": 50.0, "MSFT": 400.0},
    )
    monkeypatch.setattr(portfolio_risk.settings, "portfolio_cash_floor_pct", 0.20)
    monkeypatch.setattr(portfolio_risk.settings, "portfolio_concentration_max_pct", 1.0)
    monkeypatch.setattr(portfolio_risk.settings, "portfolio_correlation_bucket_max_pct", 1.0)
    monkeypatch.setattr(portfolio_risk.settings, "max_open_positions_overseas", 100)

    snap = await portfolio_risk.portfolio_snapshot(["AAPL", "MSFT"])
    # Buying 22 MSFT shares @ $400 = $8,800 — trivial relative to the
    # $150,000 aggregate cash. Under the old bug this was always reduced to 0
    # (position_shares_before stays 0) because cash_before was always 0.
    target_shares, note = portfolio_risk.apply_portfolio_caps(
        "MSFT", 22, 400.0, 0, snap, clusters={}, current_drawdown_frac=0.0
    )
    assert target_shares == 22
    assert note is None


async def test_portfolio_snapshot_max_open_does_not_count_flat_tickers(monkeypatch):
    """`apply_portfolio_caps`'s max-open-positions cap counts tickers with
    per_ticker_value > 0 as "open". Before the fix, an all-cash (0-share)
    ticker still had a nonzero per_ticker_value (its own cash), so it was
    wrongly counted as an open position. With the fix, all-flat tickers
    contribute 0 open positions, so a brand-new entry is not blocked."""
    _patch_snapshot_deps(
        monkeypatch,
        positions={
            "AAPL": Position(cash=100_000.0, shares=0),
            "MSFT": Position(cash=100_000.0, shares=0),
            "GOOGL": Position(cash=100_000.0, shares=0),
        },
        prices={"AAPL": 200.0, "MSFT": 400.0, "GOOGL": 150.0},
    )
    monkeypatch.setattr(portfolio_risk.settings, "max_open_positions_overseas", 1)
    monkeypatch.setattr(portfolio_risk.settings, "portfolio_cash_floor_pct", 0.0)
    monkeypatch.setattr(portfolio_risk.settings, "portfolio_concentration_max_pct", 1.0)
    monkeypatch.setattr(portfolio_risk.settings, "portfolio_correlation_bucket_max_pct", 1.0)

    snap = await portfolio_risk.portfolio_snapshot(["AAPL", "MSFT", "GOOGL"])
    target_shares, note = portfolio_risk.apply_portfolio_caps(
        "GOOGL", 10, 150.0, 0, snap, clusters={}, current_drawdown_frac=0.0
    )
    assert target_shares == 10  # not blocked — 0 tickers actually "open" (all flat)
    assert note is None


async def test_portfolio_snapshot_normalizes_krw_cash_to_usd(monkeypatch):
    """FX regression guard (past bug: 1,400x KRW/USD unit mismatch). A
    domestic (KRW) ticker's cash must be divided by the fx_rate before
    landing in total_value, not added in raw KRW."""
    _patch_snapshot_deps(
        monkeypatch,
        positions={"005930.KS": Position(cash=1_400_000.0, shares=0), "AAPL": Position(cash=1_000.0, shares=0)},
        prices={"005930.KS": 70_000.0, "AAPL": 200.0},
        currencies={"005930.KS": "KRW", "AAPL": "USD"},
        fx_rates={"KRW": 1_400.0, "USD": 1.0},
    )
    snap = await portfolio_risk.portfolio_snapshot(["005930.KS", "AAPL"])
    # 1,400,000 KRW / 1,400 = $1,000 (not 1,400,000 raw) + $1,000 AAPL cash.
    assert snap.total_value == pytest.approx(2_000.0)
    assert snap.per_ticker_value == {"005930.KS": 0.0, "AAPL": 0.0}


async def test_portfolio_snapshot_price_lookup_failure_zeroes_ticker_entirely(monkeypatch):
    """Existing degrade-quietly convention: if valuing one ticker raises
    (price/DB lookup failure), that ticker contributes 0 to BOTH
    per_ticker_value and total_value's cash component — not just
    per_ticker_value — so a failed lookup can't silently inject phantom
    cash into the aggregate total either."""
    _patch_snapshot_deps(
        monkeypatch,
        positions={"AAPL": Position(cash=50_000.0, shares=10), "MSFT": Position(cash=100_000.0, shares=0)},
        prices={"AAPL": 50.0, "MSFT": 400.0},
        raise_for={"MSFT"},
    )
    snap = await portfolio_risk.portfolio_snapshot(["AAPL", "MSFT"])
    assert snap.per_ticker_value == {"AAPL": pytest.approx(500.0), "MSFT": 0.0}
    assert snap.total_value == pytest.approx(50_500.0)  # AAPL's cash+pos only; MSFT contributes nothing


# ---------------------------------------------------------------------------
# apply_portfolio_caps / apply_backtest_portfolio_caps — `price_fx_rate`
# (2026-08-14 bug 2 fix). `portfolio_snapshot` (backtest) returns a
# USD-normalized total, but the backtest harness's `price` for a domestic
# ticker is still raw KRW (yfinance close, no conversion) — apply_portfolio_
# caps used to compare that raw KRW price straight against the USD total,
# looking ~1,400x too concentrated and wiping otherwise-valid targets to 0.
# This is the exact "22 shares -> 0" shape from the 98.1%-Hold baseline
# backtest. See app.execution.portfolio_risk.apply_portfolio_caps's own
# `price_fx_rate` docstring for the arithmetic.
# ---------------------------------------------------------------------------


def test_apply_portfolio_caps_converts_krw_price_against_usd_snapshot(monkeypatch):
    """Regression: 22 shares @ KRW 677,000 against a $100,000 USD-normalized
    snapshot with no other positions must pass through UNCHANGED once the
    KRW price is correctly reconciled via price_fx_rate=1400.0. Under the
    pre-fix behavior (raw price compared straight against the USD total),
    projected_value = 22 * 677,000 = 14,894,000 against total_value=100,000
    looked like >99% concentration and wiped this to 0."""
    monkeypatch.setattr(portfolio_risk.settings, "portfolio_concentration_max_pct", 0.30)
    monkeypatch.setattr(portfolio_risk.settings, "portfolio_correlation_bucket_max_pct", 1.0)
    monkeypatch.setattr(portfolio_risk.settings, "portfolio_cash_floor_pct", 0.0)
    snapshot = portfolio_risk.PortfolioSnapshot(per_ticker_value={}, total_value=100_000.0)

    target_shares, note = portfolio_risk.apply_portfolio_caps(
        "005930.KS", 22, 677_000.0, 0, snapshot, clusters={}, current_drawdown_frac=0.0,
        price_fx_rate=1400.0,
    )

    # unit_price = 677,000 / 1400 = 483.57; projected = 22 * 483.57 = 10,638.6;
    # other_total = 100,000 (this ticker not yet in the snapshot) ->
    # concentration = 10,638.6 / 110,638.6 ~= 9.6% <= 30% cap -> uncapped.
    assert target_shares == 22
    assert note is None


def test_apply_portfolio_caps_default_fx_rate_matches_old_live_behavior(monkeypatch):
    """price_fx_rate left at its default (1.0) must behave exactly like the
    pre-change function — same-currency price and snapshot, no conversion —
    covering every existing live caller (apply_live_portfolio_caps never
    passes price_fx_rate)."""
    monkeypatch.setattr(portfolio_risk.settings, "portfolio_concentration_max_pct", 0.30)
    monkeypatch.setattr(portfolio_risk.settings, "portfolio_correlation_bucket_max_pct", 1.0)
    monkeypatch.setattr(portfolio_risk.settings, "portfolio_cash_floor_pct", 0.0)
    snapshot = portfolio_risk.PortfolioSnapshot(
        per_ticker_value={"005930.KS": 700_000.0}, total_value=3_000_000.0
    )

    target_shares, note = portfolio_risk.apply_portfolio_caps(
        "005930.KS", 12, 70_000.0, 10, snapshot, clusters={}, current_drawdown_frac=0.0
    )

    # Same math as test_apply_live_portfolio_caps_keeps_domestic_price_in_native_krw:
    # projected = 12 * 70,000 = 840,000; other_total = 3,000,000-700,000 = 2,300,000
    # concentration = 840,000 / 3,140,000 ~= 26.75% <= 30% -> uncapped.
    assert target_shares == 12
    assert note is None


def test_apply_portfolio_caps_mixed_krw_usd_concentration_matches_usd_share(monkeypatch):
    """Two tickers in one USD-normalized backtest snapshot — AAPL (USD,
    rate 1.0) and 005930.KS (KRW, rate 1400) — each ticker's concentration
    check must reflect its REAL USD-converted share of the book, not its
    raw native-currency price mismatched against the USD total."""
    monkeypatch.setattr(portfolio_risk.settings, "portfolio_correlation_bucket_max_pct", 1.0)
    monkeypatch.setattr(portfolio_risk.settings, "portfolio_cash_floor_pct", 0.0)
    monkeypatch.setattr(portfolio_risk.settings, "portfolio_concentration_max_pct", 0.15)

    # AAPL already holds $12,000 (60 sh @ $200); 005930.KS already holds
    # $10,000-equivalent (KRW 14,000,000 @ rate 1400, e.g. 20 sh @ KRW 700,000).
    # total_value = 12,000 + 10,000 + 8,000 cash = 30,000.
    snapshot = portfolio_risk.PortfolioSnapshot(
        per_ticker_value={"AAPL": 12_000.0, "005930.KS": 10_000.0}, total_value=30_000.0
    )

    # Adding 5 AAPL shares @ $200 = $1,000 -> projected = 13,000 / (18,000+13,000)
    # = 41.9% > 15% cap -> must be capped.
    aapl_target, aapl_note = portfolio_risk.apply_portfolio_caps(
        "AAPL", 65, 200.0, 60, snapshot, clusters={}, current_drawdown_frac=0.0, price_fx_rate=1.0,
    )
    assert aapl_target < 65
    assert aapl_note is not None and "concentration" in aapl_note.lower()

    # Adding 5 005930.KS shares @ KRW 700,000 = $2,500 (at rate 1400) ->
    # projected = 12,500 / (20,000+12,500) = 38.5% > 15% cap -> must be capped,
    # and by roughly the same USD magnitude as the AAPL case above (not a
    # ~1,400x-inflated wipe to the existing 20 shares).
    krw_target, krw_note = portfolio_risk.apply_portfolio_caps(
        "005930.KS", 25, 700_000.0, 20, snapshot, clusters={}, current_drawdown_frac=0.0, price_fx_rate=1400.0,
    )
    assert krw_target < 25
    assert krw_target == 20  # exact floor of max_value/unit_price — not the pre-fix bug's ~99% wipe shape
    assert krw_note is not None and "concentration" in krw_note.lower()


# ---------------------------------------------------------------------------
# apply_backtest_portfolio_caps — async wrapper resolving `price_fx_rate`
# from the ticker's own currency, symmetric to apply_live_portfolio_caps.
# ---------------------------------------------------------------------------


async def test_apply_backtest_portfolio_caps_resolves_fx_rate_from_ticker_currency(monkeypatch):
    """The wrapper must look up 005930.KS's currency (KRW) and USD rate
    (1400.0), then pass exactly that rate through to apply_portfolio_caps —
    proven here by observing the actual (uncapped) outcome, which only comes
    out right if the correct rate reached the pure function."""
    async def fake_get_ticker_currency(ticker):
        assert ticker == "005930.KS"
        return "KRW"

    async def fake_get_usd_fx_rate(currency):
        assert currency == "KRW"
        return 1400.0

    monkeypatch.setattr(portfolio_risk, "get_ticker_currency", fake_get_ticker_currency)
    monkeypatch.setattr(portfolio_risk, "get_usd_fx_rate", fake_get_usd_fx_rate)
    monkeypatch.setattr(portfolio_risk.settings, "portfolio_concentration_max_pct", 0.30)
    monkeypatch.setattr(portfolio_risk.settings, "portfolio_correlation_bucket_max_pct", 1.0)
    monkeypatch.setattr(portfolio_risk.settings, "portfolio_cash_floor_pct", 0.0)

    snapshot = portfolio_risk.PortfolioSnapshot(per_ticker_value={}, total_value=100_000.0)
    target_shares, note = await portfolio_risk.apply_backtest_portfolio_caps(
        "005930.KS", 22, 677_000.0, 0, snapshot, clusters={}, current_drawdown_frac=0.0,
    )

    # Same arithmetic as test_apply_portfolio_caps_converts_krw_price_against_usd_snapshot —
    # only reachable if the wrapper actually forwarded fx_rate=1400.0.
    assert target_shares == 22
    assert note is None


async def test_apply_backtest_portfolio_caps_wrong_fx_rate_would_wipe_to_zero(monkeypatch):
    """Sanity check on the test above: if the wrapper's fx lookup were
    skipped or wrong (e.g. left at 1.0), the same 22-share KRW buy WOULD be
    wiped — confirming the previous test's pass is actually meaningful and
    not just a permissive cap."""
    async def fake_get_ticker_currency(ticker):
        return "USD"  # wrong — must be treated as no conversion, reproducing the bug

    async def fake_get_usd_fx_rate(currency):
        return 1.0

    monkeypatch.setattr(portfolio_risk, "get_ticker_currency", fake_get_ticker_currency)
    monkeypatch.setattr(portfolio_risk, "get_usd_fx_rate", fake_get_usd_fx_rate)
    monkeypatch.setattr(portfolio_risk.settings, "portfolio_concentration_max_pct", 0.30)
    monkeypatch.setattr(portfolio_risk.settings, "portfolio_correlation_bucket_max_pct", 1.0)
    monkeypatch.setattr(portfolio_risk.settings, "portfolio_cash_floor_pct", 0.0)

    snapshot = portfolio_risk.PortfolioSnapshot(per_ticker_value={}, total_value=100_000.0)
    target_shares, note = await portfolio_risk.apply_backtest_portfolio_caps(
        "005930.KS", 22, 677_000.0, 0, snapshot, clusters={}, current_drawdown_frac=0.0,
    )
    assert target_shares == 0
    assert note is not None and "concentration" in note.lower()


# ---------------------------------------------------------------------------
# overseas_trading_enabled backstop — verify_overseas_flat_if_disabled /
# overseas_flag_contradicted (app.config.overseas_trading_enabled's own
# comment). THE critical invariant here is that the flag never reaches the
# enforcement path (real_account_snapshot/daily_loss_block) — it only
# suppresses BACKGROUND polling elsewhere (app.web.state, auto_trade_loop).
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_overseas_flag_latch(monkeypatch):
    monkeypatch.setattr(portfolio_risk, "_overseas_flag_contradicted_flag", False)
    monkeypatch.setattr(portfolio_risk, "_overseas_flag_last_verified_date", None)


async def test_real_account_snapshot_overseas_still_queries_when_flag_disabled(monkeypatch):
    """THE critical invariant: overseas_trading_enabled=False must never
    reach real_account_snapshot — the enforcement path always queries the
    real overseas ledger, exactly as if the flag were on."""
    broker = CountingBalanceBroker()
    monkeypatch.setattr(state, "kis_broker", broker_stub(broker))
    monkeypatch.setattr(portfolio_risk.settings, "overseas_trading_enabled", False)

    snap = await portfolio_risk.real_account_snapshot("overseas")

    assert broker.overseas_calls == 1
    assert broker.present_balance_calls == 1
    assert snap.per_ticker_value == {"AAPL": 1_500.0}
    assert snap.total_value == pytest.approx(2_000.0)


async def test_daily_loss_block_overseas_unaffected_by_flag(monkeypatch):
    from app.execution import safety_switch

    broker = CountingBalanceBroker()
    monkeypatch.setattr(state, "kis_broker", broker_stub(broker))
    monkeypatch.setattr(portfolio_risk.settings, "overseas_trading_enabled", False)

    async def fake_baseline(market, today, current_value):
        return 0.0  # baseline<=0 -> daily_loss_block returns None quickly, still after querying real_account_snapshot

    monkeypatch.setattr(safety_switch, "_daily_loss_baseline", fake_baseline)

    result = await safety_switch.daily_loss_block("buy", "AAPL")

    assert result is None
    assert broker.overseas_calls == 1  # real_account_snapshot ran normally, flag never consulted


class _FlatOverseasBroker:
    """Empty overseas ledger + zero USD deposit — the "assumption holds"
    case for verify_overseas_flat_if_disabled."""

    def __init__(self):
        self.overseas_balance_calls = 0
        self.present_balance_calls = 0

    async def fetch_overseas_balance(self):
        self.overseas_balance_calls += 1
        return BrokerBalance(cash_deposit=0.0, total_eval_amount=0.0, total_pnl_amount=0.0, purchase_amount=0.0, positions=[])

    async def fetch_overseas_present_balance(self, currency="USD"):
        self.present_balance_calls += 1
        return OverseasCash(currency=currency, deposit=0.0, orderable=0.0, eval_amount=0.0, matched=True)


class _PositionedOverseasBroker(_FlatOverseasBroker):
    async def fetch_overseas_balance(self):
        self.overseas_balance_calls += 1
        return BrokerBalance(
            cash_deposit=0.0, total_eval_amount=1_500.0, total_pnl_amount=0.0, purchase_amount=0.0,
            positions=[
                BrokerPosition(
                    code="AAPL", name="Apple Inc.", quantity=5,
                    avg_price=300.0, current_price=300.0, eval_amount=1_500.0,
                    pnl_amount=0.0, pnl_pct=0.0,
                )
            ],
        )


class _NonZeroDepositOverseasBroker(_FlatOverseasBroker):
    async def fetch_overseas_present_balance(self, currency="USD"):
        self.present_balance_calls += 1
        return OverseasCash(currency=currency, deposit=250.0, orderable=250.0, eval_amount=0.0, matched=True)


class _FailingOverseasBroker(_FlatOverseasBroker):
    async def fetch_overseas_balance(self):
        self.overseas_balance_calls += 1
        raise RuntimeError("KIS overseas balance inquiry failed")


def _captured_alerts(monkeypatch):
    from app.live import alerts as alerts_module

    captured = []

    async def fake_record_alert(alert):
        captured.append(alert)

    monkeypatch.setattr(alerts_module, "record_alert", fake_record_alert)
    return captured


async def test_verify_overseas_flat_latches_on_position(monkeypatch, caplog):
    broker = _PositionedOverseasBroker()
    monkeypatch.setattr(state, "kis_broker", broker_stub(broker))
    monkeypatch.setattr(portfolio_risk.settings, "overseas_trading_enabled", False)
    captured = _captured_alerts(monkeypatch)

    with caplog.at_level("CRITICAL", logger="app.execution.portfolio_risk"):
        await portfolio_risk.verify_overseas_flat_if_disabled()

    assert portfolio_risk.overseas_flag_contradicted() is True
    assert any(r.levelname == "CRITICAL" for r in caplog.records)
    assert len(captured) == 1
    assert captured[0].kind == "overseas_exposure_while_disabled"


async def test_verify_overseas_flat_latches_on_nonzero_deposit(monkeypatch, caplog):
    broker = _NonZeroDepositOverseasBroker()
    monkeypatch.setattr(state, "kis_broker", broker_stub(broker))
    monkeypatch.setattr(portfolio_risk.settings, "overseas_trading_enabled", False)
    captured = _captured_alerts(monkeypatch)

    with caplog.at_level("CRITICAL", logger="app.execution.portfolio_risk"):
        await portfolio_risk.verify_overseas_flat_if_disabled()

    assert portfolio_risk.overseas_flag_contradicted() is True
    assert any(r.levelname == "CRITICAL" for r in caplog.records)
    assert len(captured) == 1
    assert captured[0].kind == "overseas_exposure_while_disabled"


async def test_verify_overseas_flat_latches_on_probe_failure(monkeypatch):
    broker = _FailingOverseasBroker()
    monkeypatch.setattr(state, "kis_broker", broker_stub(broker))
    monkeypatch.setattr(portfolio_risk.settings, "overseas_trading_enabled", False)
    _captured_alerts(monkeypatch)

    await portfolio_risk.verify_overseas_flat_if_disabled()  # must not raise

    # Fail-safe direction: could not verify => do not keep suppressing.
    assert portfolio_risk.overseas_flag_contradicted() is True


async def test_verify_overseas_flat_noop_when_enabled(monkeypatch):
    broker = _PositionedOverseasBroker()  # would trip the latch if the probe ran at all
    monkeypatch.setattr(state, "kis_broker", broker_stub(broker))
    monkeypatch.setattr(portfolio_risk.settings, "overseas_trading_enabled", True)

    await portfolio_risk.verify_overseas_flat_if_disabled()

    assert broker.overseas_balance_calls == 0
    assert broker.present_balance_calls == 0
    assert portfolio_risk.overseas_flag_contradicted() is False


async def test_latched_flag_restores_overseas_polling(monkeypatch):
    """Integration between the latch and app.web.state — once tripped,
    _kis_account_snapshot_build must query overseas again despite the flag
    still being off."""
    from app.web import state as web_state

    broker = CountingBalanceBroker()  # has both fetch_balance and fetch_overseas_balance
    monkeypatch.setattr(web_state, "kis_broker", broker_stub(broker))
    monkeypatch.setattr(portfolio_risk.settings, "overseas_trading_enabled", False)
    monkeypatch.setattr(portfolio_risk, "_overseas_flag_contradicted_flag", True)  # already tripped

    account, _positions = await web_state._kis_account_snapshot_build()

    assert broker.overseas_calls == 1
    assert broker.present_balance_calls == 1
    assert account["overseas_enabled"] is True


async def test_verify_overseas_flat_runs_at_most_once_per_day(monkeypatch):
    broker = _FlatOverseasBroker()
    monkeypatch.setattr(state, "kis_broker", broker_stub(broker))
    monkeypatch.setattr(portfolio_risk.settings, "overseas_trading_enabled", False)
    _captured_alerts(monkeypatch)

    await portfolio_risk.verify_overseas_flat_if_disabled()
    await portfolio_risk.verify_overseas_flat_if_disabled()

    assert broker.overseas_balance_calls == 1
    assert broker.present_balance_calls == 1
    assert portfolio_risk.overseas_flag_contradicted() is False
