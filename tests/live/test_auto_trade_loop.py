"""app.live.auto_trade_loop — run_pipeline/krx_top_n/overseas_top_n/
state._kis_broker/market-hours all faked, no network/LLM. The loop itself
runs forever, so tests call one cycle's body worth of logic directly rather
than running auto_trade_loop() itself (same reasoning as app.live.poller's
own tests, which test poll_once, not run_poller_loop)."""

from __future__ import annotations

import asyncio
from datetime import date, datetime, tzinfo

import pytest

import app.clock as clock_module
from app.chat.screening import MomentumScore, ScreeningPass
from app.config import ExecutionMode, account_mode_scope, settings
from app.live import auto_trade_loop
from app.user_context import ServicedScope, user_scope
from app.web import trade_allowlist_store
from app.live.store import Tick
from conftest import broker_stub


def _as_screening_pass(fake_top_candidates):
    """Adapts a `top_candidates`-shaped fake (returns a ranked list[str], or
    raises) into a `screening_pass`-shaped one. auto_trade_loop now calls
    screening_pass directly (2026-09-08, §4-1 live screening instrumentation)
    so it can record the whole scored cross-section alongside the ranked
    picks, not just the ranked tickers `top_candidates` used to return. These
    cycle-level tests only care about which tickers survive to `final`, so a
    descending fake score per returned ticker reproduces the old ranked-list
    behavior exactly."""

    async def _screening_pass(universe, top_n=5, lookback_days=10, **kwargs):
        tickers = await fake_top_candidates(universe, top_n=top_n, lookback_days=lookback_days, **kwargs)
        scored = {
            t: MomentumScore(raw_pct=0.0, atr_pct=None, score=float(len(tickers) - i))
            for i, t in enumerate(tickers)
        }
        ranked = [(t, scored[t].score) for t in tickers]
        return ScreeningPass(scored=scored, ranked=ranked, score_key="score")

    return _screening_pass


class _FixedClock:
    def __init__(self, today: date):
        self._today = today

    def today(self) -> date:
        return self._today

    def now(self, tz: tzinfo | None = None) -> datetime:
        # Pre-labeled with the requested tz (not converted from another
        # one), so local_today(any market) reads back exactly `_today`
        # regardless of which market's timezone it asks for — these tests
        # care about a single deterministic date, not the KST-vs-ET
        # distinction _local_today/local_today exists to make.
        return datetime(self._today.year, self._today.month, self._today.day, 12, 0, tzinfo=tz)


def _set_trade_allowlist(monkeypatch, tickers):
    domestic = [t for t in tickers if t.endswith(".KS")]
    overseas = [t for t in tickers if not t.endswith(".KS")]
    monkeypatch.setattr(settings, "trade_allowlist_domestic", domestic)
    monkeypatch.setattr(settings, "trade_allowlist_overseas", overseas)


@pytest.fixture(autouse=True)
def _reset_settings(monkeypatch):
    _set_trade_allowlist(monkeypatch, ["AAPL"])
    monkeypatch.setattr(settings, "auto_trade_scan_count_domestic", 30)
    monkeypatch.setattr(settings, "auto_trade_scan_count_overseas", 30)
    monkeypatch.setattr(settings, "auto_trade_max_tickers_domestic", 3)
    monkeypatch.setattr(settings, "auto_trade_max_tickers_overseas", 2)
    # Fixed at 1 (no overfetch) so existing top_n/final-count assertions in
    # this file don't need to account for the multiplier — the dedicated
    # overfetch tests below override this explicitly to exercise it.
    monkeypatch.setattr(settings, "auto_trade_candidate_overfetch_multiplier", 1)
    # 2026-09-01 audit #6's purchasability filter defaults to False here so
    # every existing test in this file keeps its pre-#6 candidate list
    # without needing to stub cached_account_snapshot — it currently degrades
    # safely on its own (AccountUnavailable, no broker configured in tests),
    # but pinning it explicitly means that safety net isn't what these
    # unrelated tests are silently depending on. The dedicated tests below
    # turn it back on and stub the account snapshot themselves.
    monkeypatch.setattr(settings, "auto_trade_skip_unexecutable_candidates", False)
    monkeypatch.setattr(settings, "auto_trade_interval_minutes", 0.001)
    monkeypatch.setattr(settings, "max_open_positions_domestic", 15)
    monkeypatch.setattr(settings, "max_open_positions_overseas", 15)
    # Both markets "open" by default — dedicated tests override to check closed-market behavior.
    monkeypatch.setattr(auto_trade_loop, "is_domestic_market_open", lambda: True)
    monkeypatch.setattr(auto_trade_loop, "is_overseas_market_open", lambda: True)
    # Status and the semi-auto first-cycle flag are per (user, mode) now
    # (Phase 6) — a single module-level bool meant the second serviced user
    # never got its one SEMI_AUTO run, and a single status meant every user's
    # dashboard showed whichever cycle ran last. Clearing both scope maps is
    # the equivalent of the old two monkeypatches.
    # 상태/플래그가 스코프별이 되었으므로 맵 전체를 비움.
    auto_trade_loop.reset_status_scopes()
    # Process-local dedup for the overnight gap-risk warning — same reason
    # test_alerts.py clears app.live.alerts._last_fired_at: a dict keyed
    # only by ticker (no test isolation) would leak "already warned today"
    # state across tests that happen to reuse the same ticker symbol.
    auto_trade_loop._gap_warned_on.clear()
    # Phase 6 made the loop resolve its serviced scopes from Postgres and load
    # each scope's allowlist. Same reasoning as the two stubs below: unstubbed,
    # every test in this file would query the REAL dev Postgres, and the first
    # one to build a pool leaves it bound to a closed event loop for the rest.
    # Fixed single scope = the pre-Phase-6 behaviour these tests were written
    # against, with the identity now explicit.
    # 스텁 없이는 실제 Postgres를 조회하고, 첫 테스트가 만든 풀이 이후 테스트의
    # 이벤트 루프에서 죽음.
    async def _one_scope():
        return [ServicedScope(settings.user_id, settings.kis_account_mode)]

    async def _already_loaded(user_id, account_mode):
        return None

    monkeypatch.setattr(auto_trade_loop, "serviced_scopes", _one_scope)
    monkeypatch.setattr(trade_allowlist_store, "ensure_loaded", _already_loaded)
    # Default-safe stubs for the two candidate-pipeline filters that
    # otherwise reach a real DB/broker (get_last_decision_time,
    # kis_account_snapshot) — without these, an unmocked test silently
    # queries the REAL production Postgres for whatever tickers it happens
    # to use, making results depend on live account state instead of the
    # test's own fixtures. "never analyzed"/"no real broker" (the safest,
    # most permissive defaults) unless a test explicitly overrides them.
    async def _never_analyzed(ticker):
        return None

    async def _no_real_positions():
        return None, {}

    async def _no_cash_capped(tickers):
        return []

    monkeypatch.setattr(auto_trade_loop, "get_last_decision_time", _never_analyzed)
    monkeypatch.setattr("app.web.state.kis_account_snapshot", _no_real_positions)
    monkeypatch.setattr(auto_trade_loop, "tickers_cash_floor_capped", _no_cash_capped)


def test_split_trade_allowlist_separates_by_ks_suffix(monkeypatch):
    _set_trade_allowlist(monkeypatch, ["AAPL", "005930.KS", "MSFT", "000660.KS"])
    domestic, overseas = auto_trade_loop._split_trade_allowlist()
    assert domestic == ["005930.KS", "000660.KS"]
    assert overseas == ["AAPL", "MSFT"]


def test_split_trade_allowlist_drops_malformed_entries(monkeypatch):
    # A malformed entry predating app.web.state's format check (added
    # before it existed) must not keep getting pooled into every cycle's
    # own candidates — that's how one bad ticker self-perpetuated as a
    # recurring "AI signal" instead of just sitting inert.
    _set_trade_allowlist(monkeypatch, ["AAPL", "066430.KS,0P0000CBYQ,198463", "005930.KS"])
    domestic, overseas = auto_trade_loop._split_trade_allowlist()
    assert domestic == ["005930.KS"]
    assert overseas == ["AAPL"]


# ---------------------------------------------------------------------------
# _domestic_candidates / _overseas_candidates — never pooled together
# ---------------------------------------------------------------------------


async def test_domestic_candidates_pools_trade_allowlist_and_krx_candidates(monkeypatch):
    _set_trade_allowlist(monkeypatch, ["AAPL", "005930.KS"])
    monkeypatch.setattr(settings, "auto_trade_max_tickers_domestic", 10)
    captured = {}

    async def fake_krx_top_n(broker, count, *, max_price=None):
        return ["005930.KS", "000660.KS"]

    async def fake_top_candidates(universe, top_n=5, lookback_days=10, **kwargs):
        captured["universe"] = universe
        captured["top_n"] = top_n
        return universe

    monkeypatch.setattr(auto_trade_loop, "krx_top_n", fake_krx_top_n)
    monkeypatch.setattr(auto_trade_loop, "screening_pass", _as_screening_pass(fake_top_candidates))
    result = await auto_trade_loop._domestic_candidates(broker=None)
    assert captured["universe"] == ["005930.KS", "000660.KS"]  # deduped; AAPL never enters the domestic pool
    assert captured["top_n"] == 10
    assert result == ["005930.KS", "000660.KS"]


async def test_domestic_candidates_pools_market_cap_candidates_too(monkeypatch):
    # 2026-07-30: krx_market_cap_top_n is a second, stable discovery axis
    # alongside krx_top_n's turnover-driven one — added specifically so a
    # cycle isn't limited to just that day's momentum leaders (see
    # app.live.universe.krx_market_cap_top_n's own docstring for the
    # "everything already analyzed today" incident this addresses).
    _set_trade_allowlist(monkeypatch, ["AAPL"])
    monkeypatch.setattr(settings, "auto_trade_max_tickers_domestic", 10)
    captured = {}

    async def fake_krx_top_n(broker, count, *, max_price=None):
        return ["005930.KS"]

    async def fake_krx_market_cap_top_n(broker, count, *, max_price=None):
        return ["000660.KS"]

    async def fake_top_candidates(universe, top_n=5, lookback_days=10, **kwargs):
        captured["universe"] = universe
        return universe

    monkeypatch.setattr(auto_trade_loop, "krx_top_n", fake_krx_top_n)
    monkeypatch.setattr(auto_trade_loop, "krx_market_cap_top_n", fake_krx_market_cap_top_n)
    monkeypatch.setattr(auto_trade_loop, "screening_pass", _as_screening_pass(fake_top_candidates))
    result = await auto_trade_loop._domestic_candidates(broker=None)
    assert captured["universe"] == ["005930.KS", "000660.KS"]
    assert result == ["005930.KS", "000660.KS"]


async def test_domestic_candidates_tags_each_ticker_with_every_admitting_axis(monkeypatch):
    """§4-4 (2026-09-08): a ticker returned by more than one discovery axis
    (turnover + market-cap here) must be tagged with EVERY axis that
    admitted it, not just one — see _axis_map's own docstring for why
    collapsing to a single label would misattribute the overlap."""
    _set_trade_allowlist(monkeypatch, [])
    monkeypatch.setattr(settings, "screening_snapshot_log_enabled", True)

    async def fake_krx_top_n(broker, count, *, max_price=None):
        return ["005930.KS", "000660.KS"]

    async def fake_krx_market_cap_top_n(broker, count, *, max_price=None):
        return ["000660.KS"]  # overlaps with krx_top_n

    async def fake_top_candidates(universe, top_n=5, lookback_days=10, **kwargs):
        return universe

    captured = {}

    async def fake_record_screening_snapshot(ctx, *, axes_by_ticker, **kwargs):
        captured["axes_by_ticker"] = axes_by_ticker
        return 0

    monkeypatch.setattr(auto_trade_loop, "krx_top_n", fake_krx_top_n)
    monkeypatch.setattr(auto_trade_loop, "krx_market_cap_top_n", fake_krx_market_cap_top_n)
    monkeypatch.setattr(auto_trade_loop, "screening_pass", _as_screening_pass(fake_top_candidates))
    import app.live.screening_log as screening_log_module
    monkeypatch.setattr(screening_log_module, "record_screening_snapshot", fake_record_screening_snapshot)

    await auto_trade_loop._domestic_candidates(broker=None)

    assert captured["axes_by_ticker"]["005930.KS"] == ["krx_top_n"]
    assert captured["axes_by_ticker"]["000660.KS"] == ["krx_top_n", "market_cap_top_n"]


async def test_domestic_candidates_skips_new_axes_when_toggled_off(monkeypatch):
    # 2026-09-02 (screening-improvement-research #3): both new discovery
    # axes default OFF. This is the default-off safety guarantee — merging
    # this feature must not call either new axis function, let alone change
    # the resulting universe, until someone explicitly flips a toggle.
    _set_trade_allowlist(monkeypatch, ["AAPL"])
    monkeypatch.setattr(settings, "auto_trade_universe_fluctuation_losers_domestic", False)
    monkeypatch.setattr(settings, "auto_trade_universe_disparity_domestic", False)
    monkeypatch.setattr(settings, "auto_trade_max_tickers_domestic", 10)
    captured = {}

    async def fake_krx_top_n(broker, count, *, max_price=None):
        return ["005930.KS"]

    async def fake_krx_market_cap_top_n(broker, count, *, max_price=None):
        return ["000660.KS"]

    async def should_not_be_called(*args, **kwargs):
        raise AssertionError("a toggled-off discovery axis must not be called")

    async def fake_top_candidates(universe, top_n=5, lookback_days=10, **kwargs):
        captured["universe"] = universe
        return universe

    monkeypatch.setattr(auto_trade_loop, "krx_top_n", fake_krx_top_n)
    monkeypatch.setattr(auto_trade_loop, "krx_market_cap_top_n", fake_krx_market_cap_top_n)
    monkeypatch.setattr(auto_trade_loop, "krx_fluctuation_losers_n", should_not_be_called)
    monkeypatch.setattr(auto_trade_loop, "krx_disparity_low_n", should_not_be_called)
    monkeypatch.setattr(auto_trade_loop, "screening_pass", _as_screening_pass(fake_top_candidates))
    result = await auto_trade_loop._domestic_candidates(broker=None)
    assert captured["universe"] == ["005930.KS", "000660.KS"]
    assert result == ["005930.KS", "000660.KS"]


# ---------------------------------------------------------------------------
# _domestic_price_ceiling / max_price threading (2026-09-02,
# screening-improvement-research #3 follow-up) — same pattern as
# test_domestic_candidates_skips_new_axes_when_toggled_off above.
# ---------------------------------------------------------------------------


async def test_domestic_candidates_sends_no_price_ceiling_when_toggled_off(monkeypatch):
    _set_trade_allowlist(monkeypatch, ["AAPL"])
    monkeypatch.setattr(settings, "auto_trade_universe_price_ceiling_domestic", False)
    monkeypatch.setattr(settings, "auto_trade_universe_fluctuation_losers_domestic", True)
    monkeypatch.setattr(settings, "auto_trade_universe_disparity_domestic", True)
    captured = {}

    async def fail_if_called(market, **kwargs):
        raise AssertionError("cached_account_snapshot must not be called when the toggle is off")

    async def fake_krx_top_n(broker, count, *, max_price=None):
        captured["trading_value"] = max_price
        return []

    async def fake_krx_market_cap_top_n(broker, count, *, max_price=None):
        captured["market_cap"] = max_price
        return []

    async def fake_krx_fluctuation_losers_n(broker, count, *, min_volume=0, max_price=None):
        captured["fluctuation_losers"] = max_price
        return []

    async def fake_krx_disparity_low_n(broker, count, *, window_days=20, min_volume=0, max_price=None):
        captured["disparity_low"] = max_price
        return []

    monkeypatch.setattr("app.execution.portfolio_risk.cached_account_snapshot", fail_if_called)
    monkeypatch.setattr(auto_trade_loop, "krx_top_n", fake_krx_top_n)
    monkeypatch.setattr(auto_trade_loop, "krx_market_cap_top_n", fake_krx_market_cap_top_n)
    monkeypatch.setattr(auto_trade_loop, "krx_fluctuation_losers_n", fake_krx_fluctuation_losers_n)
    monkeypatch.setattr(auto_trade_loop, "krx_disparity_low_n", fake_krx_disparity_low_n)

    await auto_trade_loop._domestic_candidates(broker=None)
    assert captured == {
        "trading_value": None, "market_cap": None, "fluctuation_losers": None, "disparity_low": None,
    }


async def test_domestic_candidates_passes_computed_ceiling_to_all_four_axes(monkeypatch):
    _set_trade_allowlist(monkeypatch, ["AAPL"])
    monkeypatch.setattr(settings, "auto_trade_universe_price_ceiling_domestic", True)
    monkeypatch.setattr(settings, "auto_trade_universe_price_ceiling_slack_multiple", 1.0)
    monkeypatch.setattr(settings, "auto_trade_universe_fluctuation_losers_domestic", True)
    monkeypatch.setattr(settings, "auto_trade_universe_disparity_domestic", True)
    captured = {}

    async def fake_snapshot(market):
        return _fake_portfolio_snapshot({}, 2_000_000.0)

    async def fake_krx_top_n(broker, count, *, max_price=None):
        captured["trading_value"] = max_price
        return []

    async def fake_krx_market_cap_top_n(broker, count, *, max_price=None):
        captured["market_cap"] = max_price
        return []

    async def fake_krx_fluctuation_losers_n(broker, count, *, min_volume=0, max_price=None):
        captured["fluctuation_losers"] = max_price
        return []

    async def fake_krx_disparity_low_n(broker, count, *, window_days=20, min_volume=0, max_price=None):
        captured["disparity_low"] = max_price
        return []

    monkeypatch.setattr("app.execution.portfolio_risk.real_account_snapshot", fake_snapshot)
    monkeypatch.setattr(auto_trade_loop, "krx_top_n", fake_krx_top_n)
    monkeypatch.setattr(auto_trade_loop, "krx_market_cap_top_n", fake_krx_market_cap_top_n)
    monkeypatch.setattr(auto_trade_loop, "krx_fluctuation_losers_n", fake_krx_fluctuation_losers_n)
    monkeypatch.setattr(auto_trade_loop, "krx_disparity_low_n", fake_krx_disparity_low_n)

    await auto_trade_loop._domestic_candidates(broker=None)
    # Shipped risk defaults (sizing_max_risk_pct_of_equity=0.01,
    # sizing_target_atr_pct=0.02, sizing_min_stop_atr_multiple=2.5) at
    # slack=1.0: equity*0.01/(0.02*2.5) == 2,000,000*0.01/0.05 == 400,000.
    for axis, value in captured.items():
        assert value == pytest.approx(400_000.0), axis


async def test_domestic_candidates_keeps_no_ceiling_when_account_unavailable(monkeypatch, caplog):
    import logging

    from app.execution.portfolio_risk import AccountUnavailable

    _set_trade_allowlist(monkeypatch, ["AAPL"])
    monkeypatch.setattr(settings, "auto_trade_universe_price_ceiling_domestic", True)
    captured = {}

    async def fake_snapshot(market):
        raise AccountUnavailable("no broker configured")

    async def fake_krx_top_n(broker, count, *, max_price=None):
        captured["trading_value"] = max_price
        return ["005930.KS"]

    async def fake_top_candidates(universe, top_n=5, lookback_days=10, **kwargs):
        return universe

    monkeypatch.setattr("app.execution.portfolio_risk.real_account_snapshot", fake_snapshot)
    monkeypatch.setattr(auto_trade_loop, "krx_top_n", fake_krx_top_n)
    monkeypatch.setattr(auto_trade_loop, "screening_pass", _as_screening_pass(fake_top_candidates))

    with caplog.at_level(logging.WARNING):
        result = await auto_trade_loop._domestic_candidates(broker=None)
    assert captured["trading_value"] is None
    assert result == ["005930.KS"]  # universe unchanged, cycle doesn't raise
    assert any("NO price ceiling" in msg for msg in caplog.messages)


async def test_domestic_candidates_keeps_no_ceiling_when_equity_is_zero(monkeypatch):
    _set_trade_allowlist(monkeypatch, ["AAPL"])
    monkeypatch.setattr(settings, "auto_trade_universe_price_ceiling_domestic", True)
    captured = {}

    async def fake_snapshot(market):
        return _fake_portfolio_snapshot({}, 0.0)

    async def fake_krx_top_n(broker, count, *, max_price=None):
        captured["trading_value"] = max_price
        return ["005930.KS"]

    async def fake_top_candidates(universe, top_n=5, lookback_days=10, **kwargs):
        return universe

    monkeypatch.setattr("app.execution.portfolio_risk.real_account_snapshot", fake_snapshot)
    monkeypatch.setattr(auto_trade_loop, "krx_top_n", fake_krx_top_n)
    monkeypatch.setattr(auto_trade_loop, "screening_pass", _as_screening_pass(fake_top_candidates))

    result = await auto_trade_loop._domestic_candidates(broker=None)
    assert captured["trading_value"] is None
    assert result == ["005930.KS"]


async def test_domestic_candidates_skips_ceiling_when_it_falls_below_the_floor(monkeypatch, caplog):
    import logging

    _set_trade_allowlist(monkeypatch, ["AAPL"])
    monkeypatch.setattr(settings, "auto_trade_universe_price_ceiling_domestic", True)
    captured = {}

    async def fake_snapshot(market):
        # equity=cash=1,000: risk ceiling floor(1,000*0.01/0.05*1.5)=300,
        # well below the default 10,000 KRW floor.
        return _fake_portfolio_snapshot({}, 1_000.0)

    async def fake_krx_top_n(broker, count, *, max_price=None):
        captured["trading_value"] = max_price
        return ["005930.KS"]

    async def fake_top_candidates(universe, top_n=5, lookback_days=10, **kwargs):
        return universe

    monkeypatch.setattr("app.execution.portfolio_risk.real_account_snapshot", fake_snapshot)
    monkeypatch.setattr(auto_trade_loop, "krx_top_n", fake_krx_top_n)
    monkeypatch.setattr(auto_trade_loop, "screening_pass", _as_screening_pass(fake_top_candidates))

    with caplog.at_level(logging.WARNING):
        result = await auto_trade_loop._domestic_candidates(broker=None)
    assert captured["trading_value"] is None
    assert result == ["005930.KS"]
    assert any("below the configured floor" in msg for msg in caplog.messages)


async def test_domestic_candidates_sends_no_ceiling_for_a_large_account(monkeypatch):
    _set_trade_allowlist(monkeypatch, ["AAPL"])
    monkeypatch.setattr(settings, "auto_trade_universe_price_ceiling_domestic", True)
    captured = {}

    async def fake_snapshot(market):
        # equity=cash=1,000,000,000: risk ceiling
        # floor(1e9*0.01/0.05*1.5)=300,000,000, well above the default
        # 2,000,000 KRW ceiling — a well-funded account clears every KRX
        # share price already, so no filter is sent at all.
        return _fake_portfolio_snapshot({}, 1_000_000_000.0)

    async def fake_krx_top_n(broker, count, *, max_price=None):
        captured["trading_value"] = max_price
        return ["005930.KS"]

    async def fake_top_candidates(universe, top_n=5, lookback_days=10, **kwargs):
        return universe

    monkeypatch.setattr("app.execution.portfolio_risk.real_account_snapshot", fake_snapshot)
    monkeypatch.setattr(auto_trade_loop, "krx_top_n", fake_krx_top_n)
    monkeypatch.setattr(auto_trade_loop, "screening_pass", _as_screening_pass(fake_top_candidates))

    result = await auto_trade_loop._domestic_candidates(broker=None)
    assert captured["trading_value"] is None
    assert result == ["005930.KS"]


async def test_domestic_candidates_ceiling_honors_slack_multiple(monkeypatch):
    _set_trade_allowlist(monkeypatch, ["AAPL"])
    monkeypatch.setattr(settings, "auto_trade_universe_price_ceiling_domestic", True)
    monkeypatch.setattr(settings, "auto_trade_universe_price_ceiling_slack_multiple", 1.5)
    captured = {}

    async def fake_snapshot(market):
        return _fake_portfolio_snapshot({}, 2_000_000.0)

    async def fake_krx_top_n(broker, count, *, max_price=None):
        captured["trading_value"] = max_price
        return []

    async def fake_top_candidates(universe, top_n=5, lookback_days=10, **kwargs):
        return universe

    monkeypatch.setattr("app.execution.portfolio_risk.real_account_snapshot", fake_snapshot)
    monkeypatch.setattr(auto_trade_loop, "krx_top_n", fake_krx_top_n)
    monkeypatch.setattr(auto_trade_loop, "screening_pass", _as_screening_pass(fake_top_candidates))

    await auto_trade_loop._domestic_candidates(broker=None)
    # Same fixture as the slack=1.0 test above, scaled by 1.5x: 400,000 * 1.5.
    assert captured["trading_value"] == pytest.approx(600_000.0)


async def test_price_ceiling_does_not_change_the_atr_fail_closed_gate(monkeypatch):
    # Pins that the price-ceiling toggle never touches _drop_unexecutable's
    # own logic — the same candidate set is dropped whether or not the
    # upstream discovery-time ceiling is enabled, since _drop_unexecutable
    # never reads that setting at all.
    monkeypatch.setattr(settings, "auto_trade_skip_unexecutable_candidates", True)
    monkeypatch.setattr(settings, "auto_trade_require_atr_for_new_entry", True)

    async def fake_snapshot(market):
        return _fake_portfolio_snapshot({}, 200_000.0)

    async def fake_get_last_close(ticker, as_of=None):
        return 12_000.0

    async def failing_get_atr_pct(ticker, as_of=None):
        raise RuntimeError("ATR unavailable")

    monkeypatch.setattr("app.execution.portfolio_risk.real_account_snapshot", fake_snapshot)
    monkeypatch.setattr("app.dataflows.kis_provider.get_last_close", fake_get_last_close)
    monkeypatch.setattr("app.dataflows.kis_provider.get_atr_pct", failing_get_atr_pct)

    monkeypatch.setattr(settings, "auto_trade_universe_price_ceiling_domestic", False)
    result_without_ceiling = await auto_trade_loop._drop_unexecutable(["005930.KS"], "domestic")

    monkeypatch.setattr(settings, "auto_trade_universe_price_ceiling_domestic", True)
    result_with_ceiling = await auto_trade_loop._drop_unexecutable(["005930.KS"], "domestic")

    assert result_without_ceiling == result_with_ceiling == []


async def test_overseas_candidates_never_computes_a_price_ceiling(monkeypatch):
    # Domestic-only feature: the overseas path must never even call
    # _domestic_price_ceiling, let alone read a real account snapshot for it.
    monkeypatch.setattr(settings, "auto_trade_universe_price_ceiling_domestic", True)

    async def fail_if_called():
        raise AssertionError("_domestic_price_ceiling must not be called from the overseas path")

    async def fake_overseas_top_n(broker, count):
        return []

    async def fake_overseas_market_cap_top_n(broker, count):
        return []

    async def fake_top_candidates(universe, top_n=5, lookback_days=10, **kwargs):
        return universe

    monkeypatch.setattr(auto_trade_loop, "_domestic_price_ceiling", fail_if_called)
    monkeypatch.setattr(auto_trade_loop, "overseas_top_n", fake_overseas_top_n)
    monkeypatch.setattr(auto_trade_loop, "overseas_market_cap_top_n", fake_overseas_market_cap_top_n)
    monkeypatch.setattr(auto_trade_loop, "screening_pass", _as_screening_pass(fake_top_candidates))

    result = await auto_trade_loop._overseas_candidates(broker=None)
    assert result == ["AAPL"]  # trade_allowlist passthrough — no exception raised


async def test_domestic_candidates_pools_fluctuation_losers_when_enabled(monkeypatch):
    _set_trade_allowlist(monkeypatch, ["AAPL"])
    monkeypatch.setattr(settings, "auto_trade_universe_fluctuation_losers_domestic", True)
    monkeypatch.setattr(settings, "auto_trade_universe_disparity_domestic", False)
    monkeypatch.setattr(settings, "auto_trade_max_tickers_domestic", 10)
    captured = {}

    async def fake_krx_top_n(broker, count, *, max_price=None):
        return ["005930.KS"]

    async def fake_krx_market_cap_top_n(broker, count, *, max_price=None):
        return []

    async def fake_krx_fluctuation_losers_n(broker, count, *, min_volume=0, max_price=None):
        return ["123450.KS"]

    async def fake_top_candidates(universe, top_n=5, lookback_days=10, **kwargs):
        captured["universe"] = universe
        return universe

    monkeypatch.setattr(auto_trade_loop, "krx_top_n", fake_krx_top_n)
    monkeypatch.setattr(auto_trade_loop, "krx_market_cap_top_n", fake_krx_market_cap_top_n)
    monkeypatch.setattr(auto_trade_loop, "krx_fluctuation_losers_n", fake_krx_fluctuation_losers_n)
    monkeypatch.setattr(auto_trade_loop, "screening_pass", _as_screening_pass(fake_top_candidates))
    result = await auto_trade_loop._domestic_candidates(broker=None)
    assert captured["universe"] == ["005930.KS", "123450.KS"]
    assert result == ["005930.KS", "123450.KS"]


async def test_domestic_candidates_pools_disparity_candidates_when_enabled(monkeypatch):
    _set_trade_allowlist(monkeypatch, ["AAPL"])
    monkeypatch.setattr(settings, "auto_trade_universe_fluctuation_losers_domestic", False)
    monkeypatch.setattr(settings, "auto_trade_universe_disparity_domestic", True)
    monkeypatch.setattr(settings, "auto_trade_max_tickers_domestic", 10)
    captured = {}

    async def fake_krx_top_n(broker, count, *, max_price=None):
        return ["005930.KS"]

    async def fake_krx_market_cap_top_n(broker, count, *, max_price=None):
        return []

    async def fake_krx_disparity_low_n(broker, count, *, window_days=20, min_volume=0, max_price=None):
        return ["067310.KS"]

    async def fake_top_candidates(universe, top_n=5, lookback_days=10, **kwargs):
        captured["universe"] = universe
        return universe

    monkeypatch.setattr(auto_trade_loop, "krx_top_n", fake_krx_top_n)
    monkeypatch.setattr(auto_trade_loop, "krx_market_cap_top_n", fake_krx_market_cap_top_n)
    monkeypatch.setattr(auto_trade_loop, "krx_disparity_low_n", fake_krx_disparity_low_n)
    monkeypatch.setattr(auto_trade_loop, "screening_pass", _as_screening_pass(fake_top_candidates))
    result = await auto_trade_loop._domestic_candidates(broker=None)
    assert captured["universe"] == ["005930.KS", "067310.KS"]
    assert result == ["005930.KS", "067310.KS"]


async def test_domestic_candidates_dedupes_across_all_four_axes(monkeypatch):
    _set_trade_allowlist(monkeypatch, [])
    monkeypatch.setattr(settings, "auto_trade_universe_fluctuation_losers_domestic", True)
    monkeypatch.setattr(settings, "auto_trade_universe_disparity_domestic", True)
    monkeypatch.setattr(settings, "auto_trade_max_tickers_domestic", 10)
    captured = {}

    async def fake_krx_top_n(broker, count, *, max_price=None):
        return ["005930.KS", "000660.KS"]

    async def fake_krx_market_cap_top_n(broker, count, *, max_price=None):
        return ["000660.KS"]  # overlaps krx_top_n

    async def fake_krx_fluctuation_losers_n(broker, count, *, min_volume=0, max_price=None):
        return ["005930.KS"]  # overlaps krx_top_n too

    async def fake_krx_disparity_low_n(broker, count, *, window_days=20, min_volume=0, max_price=None):
        return ["123450.KS"]  # genuinely new

    async def fake_top_candidates(universe, top_n=5, lookback_days=10, **kwargs):
        captured["universe"] = universe
        return universe

    monkeypatch.setattr(auto_trade_loop, "krx_top_n", fake_krx_top_n)
    monkeypatch.setattr(auto_trade_loop, "krx_market_cap_top_n", fake_krx_market_cap_top_n)
    monkeypatch.setattr(auto_trade_loop, "krx_fluctuation_losers_n", fake_krx_fluctuation_losers_n)
    monkeypatch.setattr(auto_trade_loop, "krx_disparity_low_n", fake_krx_disparity_low_n)
    monkeypatch.setattr(auto_trade_loop, "screening_pass", _as_screening_pass(fake_top_candidates))
    result = await auto_trade_loop._domestic_candidates(broker=None)
    assert captured["universe"] == ["005930.KS", "000660.KS", "123450.KS"]
    assert result == ["005930.KS", "000660.KS", "123450.KS"]


async def test_domestic_candidates_new_axes_respect_exclude(monkeypatch):
    _set_trade_allowlist(monkeypatch, [])
    monkeypatch.setattr(settings, "auto_trade_universe_fluctuation_losers_domestic", True)
    monkeypatch.setattr(settings, "auto_trade_universe_disparity_domestic", True)
    monkeypatch.setattr(settings, "auto_trade_max_tickers_domestic", 10)
    captured = {}

    async def fake_krx_top_n(broker, count, *, max_price=None):
        return []

    async def fake_krx_market_cap_top_n(broker, count, *, max_price=None):
        return []

    async def fake_krx_fluctuation_losers_n(broker, count, *, min_volume=0, max_price=None):
        return ["123450.KS"]

    async def fake_krx_disparity_low_n(broker, count, *, window_days=20, min_volume=0, max_price=None):
        return ["067310.KS"]

    async def fake_top_candidates(universe, top_n=5, lookback_days=10, **kwargs):
        captured["universe"] = universe
        return universe

    monkeypatch.setattr(auto_trade_loop, "krx_top_n", fake_krx_top_n)
    monkeypatch.setattr(auto_trade_loop, "krx_market_cap_top_n", fake_krx_market_cap_top_n)
    monkeypatch.setattr(auto_trade_loop, "krx_fluctuation_losers_n", fake_krx_fluctuation_losers_n)
    monkeypatch.setattr(auto_trade_loop, "krx_disparity_low_n", fake_krx_disparity_low_n)
    monkeypatch.setattr(auto_trade_loop, "screening_pass", _as_screening_pass(fake_top_candidates))
    result = await auto_trade_loop._domestic_candidates(broker=None, exclude=frozenset({"123450.KS"}))
    assert "123450.KS" not in captured["universe"]
    assert captured["universe"] == ["067310.KS"]
    assert "123450.KS" not in result


async def test_domestic_candidates_excludes_given_tickers_before_ranking(monkeypatch):
    # Filtering happens BEFORE top_candidates ranks — a ticker already
    # analyzed today shouldn't just get truncated off an already-small
    # top-N, it should never occupy a ranking slot at all.
    _set_trade_allowlist(monkeypatch, ["AAPL", "005930.KS"])
    monkeypatch.setattr(settings, "auto_trade_max_tickers_domestic", 10)
    captured = {}

    async def fake_krx_top_n(broker, count, *, max_price=None):
        return ["005930.KS", "000660.KS", "035420.KS"]

    async def fake_top_candidates(universe, top_n=5, lookback_days=10, **kwargs):
        captured["universe"] = universe
        return universe

    monkeypatch.setattr(auto_trade_loop, "krx_top_n", fake_krx_top_n)
    monkeypatch.setattr(auto_trade_loop, "screening_pass", _as_screening_pass(fake_top_candidates))
    result = await auto_trade_loop._domestic_candidates(broker=None, exclude=frozenset({"005930.KS"}))
    assert "005930.KS" not in captured["universe"]
    assert captured["universe"] == ["000660.KS", "035420.KS"]
    assert "005930.KS" not in result


async def test_domestic_candidates_funnel_log_reports_each_stage(monkeypatch, caplog):
    # Log-only instrumentation (2026-08-18): lets a data-source failure
    # (empty universe) be told apart from a genuinely quiet market
    # (nonzero universe, everything filtered out downstream) — see
    # HANDOFF context for the "or" in the old empty-final-list log that
    # couldn't distinguish the two. Asserts the funnel line's numbers only;
    # must never change what _domestic_candidates actually returns.
    _set_trade_allowlist(monkeypatch, ["005930.KS"])

    async def fake_krx_top_n(broker, count, *, max_price=None):
        return ["000660.KS"]

    async def fake_top_candidates(universe, top_n=5, lookback_days=10, **kwargs):
        return universe  # momentum ranking passes everything through

    monkeypatch.setattr(auto_trade_loop, "krx_top_n", fake_krx_top_n)
    monkeypatch.setattr(auto_trade_loop, "screening_pass", _as_screening_pass(fake_top_candidates))

    with caplog.at_level("INFO", logger="app.live.auto_trade_loop"):
        result = await auto_trade_loop._domestic_candidates(broker=None)

    assert result == ["005930.KS", "000660.KS"]  # unchanged behavior
    funnel_records = [r for r in caplog.records if "funnel" in r.message]
    assert len(funnel_records) == 1
    message = funnel_records[0].message
    assert "domestic" in message
    assert "universe=2" in message
    assert "eligible=2" in message
    assert "momentum_top=2" in message
    assert "after_filters=2" in message
    assert "+held=0" in message
    assert "+cash_freed=0" in message
    assert "final=2" in message
    # Universe was nonzero — no data-source-failure warning expected.
    assert not any("candidate universe is empty" in r.message for r in caplog.records)


async def test_domestic_candidates_warns_when_universe_itself_is_empty(monkeypatch, caplog):
    # The empty-universe case is the one the old end-of-cycle log couldn't
    # tell apart from "quiet market, everything filtered downstream" —
    # this dedicated warning is the actual fix.
    _set_trade_allowlist(monkeypatch, [])

    async def fake_krx_top_n(broker, count, *, max_price=None):
        return []

    async def fake_top_candidates(universe, top_n=5, lookback_days=10, **kwargs):
        return universe

    monkeypatch.setattr(auto_trade_loop, "krx_top_n", fake_krx_top_n)
    monkeypatch.setattr(auto_trade_loop, "screening_pass", _as_screening_pass(fake_top_candidates))

    with caplog.at_level("INFO", logger="app.live.auto_trade_loop"):
        result = await auto_trade_loop._domestic_candidates(broker=None)

    assert result == []  # unchanged behavior
    assert any(
        "domestic" in r.message and "candidate universe is empty" in r.message and r.levelname == "WARNING"
        for r in caplog.records
    )


async def test_domestic_candidates_no_universe_warning_when_only_downstream_filtered(monkeypatch, caplog):
    # Nonzero universe with everything filtered out downstream must NOT
    # trigger the empty-universe warning — that's the "genuinely quiet
    # market" case this fix is explicitly trying to keep distinguishable.
    _set_trade_allowlist(monkeypatch, ["005930.KS"])

    async def fake_krx_top_n(broker, count, *, max_price=None):
        return []

    async def fake_top_candidates(universe, top_n=5, lookback_days=10, **kwargs):
        return []  # momentum ranking filters everything out

    monkeypatch.setattr(auto_trade_loop, "krx_top_n", fake_krx_top_n)
    monkeypatch.setattr(auto_trade_loop, "screening_pass", _as_screening_pass(fake_top_candidates))

    with caplog.at_level("INFO", logger="app.live.auto_trade_loop"):
        result = await auto_trade_loop._domestic_candidates(broker=None)

    assert result == []  # unchanged behavior
    funnel_records = [r for r in caplog.records if "funnel" in r.message]
    assert len(funnel_records) == 1
    assert "universe=1" in funnel_records[0].message
    assert "eligible=1" in funnel_records[0].message  # eligible, just filtered out downstream by ranking
    assert "momentum_top=0" in funnel_records[0].message
    assert not any("candidate universe is empty" in r.message for r in caplog.records)


async def test_held_needing_review_flags_ticker_near_its_stop_loss(monkeypatch):
    async def fake_snapshot():
        return {"cash_deposit": 1.0}, {"005930.KS": {"quantity": 5}}

    async def fake_get_open_stop_loss(ticker):
        return (1, 95.0)

    monkeypatch.setattr("app.web.state.kis_account_snapshot", fake_snapshot)
    monkeypatch.setattr(auto_trade_loop, "get_open_stop_loss", fake_get_open_stop_loss)
    monkeypatch.setattr(auto_trade_loop, "latest", lambda ticker: Tick(price=96.0, volume=1, timestamp=None))
    monkeypatch.setattr(settings, "auto_trade_held_review_buffer_pct", 0.05)

    result = await auto_trade_loop._held_needing_review(["005930.KS"], "domestic")
    assert result == ["005930.KS"]


async def test_held_needing_review_uses_real_kis_shares_not_local_ledger(monkeypatch):
    # 2026-07-29 fix: a position bought outside this system (manual KIS app
    # trade, or predating this tracking) was never recorded in the local
    # ledger's `trades` table — get_position would read that back as
    # shares=0 and this ticker would never qualify no matter how close its
    # price got to a stop-loss. Real KIS shares must be what decides "held."
    async def fake_snapshot():
        return {"cash_deposit": 1.0}, {"005930.KS": {"quantity": 3}}  # real position, never in the local ledger

    async def fake_get_open_stop_loss(ticker):
        return (1, 95.0)

    monkeypatch.setattr("app.web.state.kis_account_snapshot", fake_snapshot)
    monkeypatch.setattr(auto_trade_loop, "get_open_stop_loss", fake_get_open_stop_loss)
    monkeypatch.setattr(auto_trade_loop, "latest", lambda ticker: Tick(price=96.0, volume=1, timestamp=None))
    monkeypatch.setattr(settings, "auto_trade_held_review_buffer_pct", 0.05)

    result = await auto_trade_loop._held_needing_review(["005930.KS"], "domestic")
    assert result == ["005930.KS"]


async def test_held_needing_review_skips_ticker_with_zero_real_shares(monkeypatch):
    async def fake_snapshot():
        return {"cash_deposit": 1.0}, {}  # no real position

    async def fail_if_called(ticker):
        raise AssertionError("must not even look up a stop-loss for a ticker with no real shares")

    monkeypatch.setattr("app.web.state.kis_account_snapshot", fake_snapshot)
    monkeypatch.setattr(auto_trade_loop, "get_open_stop_loss", fail_if_called)

    result = await auto_trade_loop._held_needing_review(["005930.KS"], "domestic")
    assert result == []


async def test_held_needing_review_skips_ticker_with_no_recorded_stop_loss(monkeypatch):
    async def fake_snapshot():
        return {"cash_deposit": 1.0}, {"005930.KS": {"quantity": 5}}

    async def fake_get_open_stop_loss(ticker):
        return None

    monkeypatch.setattr("app.web.state.kis_account_snapshot", fake_snapshot)
    monkeypatch.setattr(auto_trade_loop, "get_open_stop_loss", fake_get_open_stop_loss)

    result = await auto_trade_loop._held_needing_review(["005930.KS"], "domestic")
    assert result == []


async def test_held_needing_review_warns_when_kis_snapshot_unavailable(monkeypatch, caplog):
    # 2026-08-13 fix: kis_account_snapshot() collapses "no broker configured"
    # and "the balance call failed" into the same (None, {}); before this
    # fix, that first element was discarded entirely and the resulting
    # "skip every ticker's risk review this cycle" was invisible.
    async def fake_snapshot():
        return None, {}

    monkeypatch.setattr("app.web.state.kis_account_snapshot", fake_snapshot)

    with caplog.at_level("WARNING", logger="app.live.auto_trade_loop"):
        result = await auto_trade_loop._held_needing_review(["005930.KS"], "domestic")

    assert result == []  # behavior unchanged — this is an observability fix only
    assert any(
        "KIS account snapshot unavailable" in r.message and "_held_needing_review" in r.message
        for r in caplog.records
    )


async def test_held_needing_review_no_warning_when_kis_snapshot_is_genuinely_empty(monkeypatch, caplog):
    # A real "flat account, KIS reachable" snapshot must NOT be confused
    # with a failed one — summary must be a real (non-None) dict here.
    async def fake_snapshot():
        return {"cash_deposit": 1.0}, {}

    monkeypatch.setattr("app.web.state.kis_account_snapshot", fake_snapshot)

    with caplog.at_level("WARNING", logger="app.live.auto_trade_loop"):
        result = await auto_trade_loop._held_needing_review(["005930.KS"], "domestic")

    assert result == []
    assert not any("KIS account snapshot unavailable" in r.message for r in caplog.records)


async def test_held_needing_review_skips_ticker_price_far_from_stop_loss(monkeypatch):
    async def fake_snapshot():
        return {"cash_deposit": 1.0}, {"005930.KS": {"quantity": 5}}

    async def fake_get_open_stop_loss(ticker):
        return (1, 50.0)

    monkeypatch.setattr("app.web.state.kis_account_snapshot", fake_snapshot)
    monkeypatch.setattr(auto_trade_loop, "get_open_stop_loss", fake_get_open_stop_loss)
    monkeypatch.setattr(auto_trade_loop, "latest", lambda ticker: Tick(price=200.0, volume=1, timestamp=None))
    monkeypatch.setattr(settings, "auto_trade_held_review_buffer_pct", 0.05)

    result = await auto_trade_loop._held_needing_review(["005930.KS"], "domestic")
    assert result == []


async def test_held_needing_review_flags_top_loss_tickers_by_amount_not_pct(monkeypatch):
    # OR condition (2026-08-03): a held ticker also qualifies via its
    # unrealized loss AMOUNT ranking, independent of stop-loss proximity —
    # here no ticker has a recorded stop at all, so only the loss signal
    # can flag anything.
    async def fake_snapshot():
        return {"cash_deposit": 1.0}, {
            "005930.KS": {"quantity": 5, "pnl_amount": -500.0},  # biggest loss
            "000660.KS": {"quantity": 3, "pnl_amount": -50.0},   # smaller loss
            "035420.KS": {"quantity": 2, "pnl_amount": -10.0},   # smallest loss
        }

    async def fake_get_open_stop_loss(ticker):
        return None  # no stop recorded for any of them

    monkeypatch.setattr("app.web.state.kis_account_snapshot", fake_snapshot)
    monkeypatch.setattr(auto_trade_loop, "get_open_stop_loss", fake_get_open_stop_loss)
    monkeypatch.setattr(settings, "auto_trade_top_loss_review_count", 2)

    result = await auto_trade_loop._held_needing_review(["005930.KS", "000660.KS", "035420.KS"], "domestic")
    assert result == ["005930.KS", "000660.KS"]  # two largest losses, not the smallest


async def test_held_needing_review_ignores_gains_for_loss_ranking(monkeypatch):
    async def fake_snapshot():
        return {"cash_deposit": 1.0}, {"005930.KS": {"quantity": 5, "pnl_amount": 500.0}}

    async def fake_get_open_stop_loss(ticker):
        return None

    monkeypatch.setattr("app.web.state.kis_account_snapshot", fake_snapshot)
    monkeypatch.setattr(auto_trade_loop, "get_open_stop_loss", fake_get_open_stop_loss)
    monkeypatch.setattr(settings, "auto_trade_top_loss_review_count", 5)

    result = await auto_trade_loop._held_needing_review(["005930.KS"], "domestic")
    assert result == []


async def test_held_needing_review_unions_stop_proximity_and_loss_signals(monkeypatch):
    # A ticker can qualify via EITHER signal — flagged here purely on stop
    # proximity (tiny loss, would never make the top-loss cut) alongside
    # one flagged purely on loss amount (nowhere near its stop).
    async def fake_snapshot():
        return {"cash_deposit": 1.0}, {
            "005930.KS": {"quantity": 5, "pnl_amount": -1.0},     # near its stop
            "000660.KS": {"quantity": 3, "pnl_amount": -9000.0},  # far from its stop, big loss
        }

    async def fake_get_open_stop_loss(ticker):
        return (1, 95.0) if ticker == "005930.KS" else (1, 50.0)

    def fake_latest(ticker):
        return Tick(price=96.0, volume=1, timestamp=None) if ticker == "005930.KS" else Tick(price=200.0, volume=1, timestamp=None)

    monkeypatch.setattr("app.web.state.kis_account_snapshot", fake_snapshot)
    monkeypatch.setattr(auto_trade_loop, "get_open_stop_loss", fake_get_open_stop_loss)
    monkeypatch.setattr(auto_trade_loop, "latest", fake_latest)
    monkeypatch.setattr(settings, "auto_trade_held_review_buffer_pct", 0.05)
    monkeypatch.setattr(settings, "auto_trade_top_loss_review_count", 1)

    result = await auto_trade_loop._held_needing_review(["005930.KS", "000660.KS"], "domestic")
    assert set(result) == {"005930.KS", "000660.KS"}


async def test_held_needing_review_throttles_flagged_ticker_within_min_interval(monkeypatch):
    # 2026-08-03: a ticker sitting near its stop-loss for several
    # consecutive days used to get a full re-underwrite every single cycle
    # — held_review_min_interval_days now throttles that, separately from
    # (and much shorter than) rebalance_interval_days.
    async def fake_snapshot():
        return {"cash_deposit": 1.0}, {"005930.KS": {"quantity": 5}}

    async def fake_get_open_stop_loss(ticker):
        return (1, 95.0)

    async def fake_last_decision_time(ticker):
        return datetime(2026, 8, 3, 1, 0, 0)  # analyzed earlier today

    fixed_clock = _FixedClock(date(2026, 8, 3))
    monkeypatch.setattr(auto_trade_loop, "get_clock", lambda: fixed_clock)
    # local_today (used by _held_needing_review/_drop_recently_rebalanced's
    # shared _drop_analyzed_within) reads app.clock's OWN get_clock(), not
    # auto_trade_loop's imported binding of it.
    monkeypatch.setattr(clock_module, "get_clock", lambda: fixed_clock)
    monkeypatch.setattr("app.web.state.kis_account_snapshot", fake_snapshot)
    monkeypatch.setattr(auto_trade_loop, "get_open_stop_loss", fake_get_open_stop_loss)
    monkeypatch.setattr(auto_trade_loop, "latest", lambda ticker: Tick(price=96.0, volume=1, timestamp=None))
    monkeypatch.setattr(settings, "auto_trade_held_review_buffer_pct", 0.05)
    monkeypatch.setattr(settings, "held_review_min_interval_days", 1)
    monkeypatch.setattr(auto_trade_loop, "get_last_decision_time", fake_last_decision_time)

    result = await auto_trade_loop._held_needing_review(["005930.KS"], "domestic")
    assert result == []


async def test_held_needing_review_reflags_ticker_past_min_interval(monkeypatch):
    async def fake_snapshot():
        return {"cash_deposit": 1.0}, {"005930.KS": {"quantity": 5}}

    async def fake_get_open_stop_loss(ticker):
        return (1, 95.0)

    async def fake_last_decision_time(ticker):
        return datetime(2026, 8, 1, 1, 0, 0)  # 2 days ago — past the 1-day interval

    fixed_clock = _FixedClock(date(2026, 8, 3))
    monkeypatch.setattr(auto_trade_loop, "get_clock", lambda: fixed_clock)
    # local_today (used by _held_needing_review/_drop_recently_rebalanced's
    # shared _drop_analyzed_within) reads app.clock's OWN get_clock(), not
    # auto_trade_loop's imported binding of it.
    monkeypatch.setattr(clock_module, "get_clock", lambda: fixed_clock)
    monkeypatch.setattr("app.web.state.kis_account_snapshot", fake_snapshot)
    monkeypatch.setattr(auto_trade_loop, "get_open_stop_loss", fake_get_open_stop_loss)
    monkeypatch.setattr(auto_trade_loop, "latest", lambda ticker: Tick(price=96.0, volume=1, timestamp=None))
    monkeypatch.setattr(settings, "auto_trade_held_review_buffer_pct", 0.05)
    monkeypatch.setattr(settings, "held_review_min_interval_days", 1)
    monkeypatch.setattr(auto_trade_loop, "get_last_decision_time", fake_last_decision_time)

    result = await auto_trade_loop._held_needing_review(["005930.KS"], "domestic")
    assert result == ["005930.KS"]


async def test_held_needing_review_never_analyzed_ticker_bypasses_min_interval(monkeypatch):
    async def fake_snapshot():
        return {"cash_deposit": 1.0}, {"005930.KS": {"quantity": 5}}

    async def fake_get_open_stop_loss(ticker):
        return (1, 95.0)

    async def fake_last_decision_time(ticker):
        return None  # never analyzed — always eligible

    fixed_clock = _FixedClock(date(2026, 8, 3))
    monkeypatch.setattr(auto_trade_loop, "get_clock", lambda: fixed_clock)
    # local_today (used by _held_needing_review/_drop_recently_rebalanced's
    # shared _drop_analyzed_within) reads app.clock's OWN get_clock(), not
    # auto_trade_loop's imported binding of it.
    monkeypatch.setattr(clock_module, "get_clock", lambda: fixed_clock)
    monkeypatch.setattr("app.web.state.kis_account_snapshot", fake_snapshot)
    monkeypatch.setattr(auto_trade_loop, "get_open_stop_loss", fake_get_open_stop_loss)
    monkeypatch.setattr(auto_trade_loop, "latest", lambda ticker: Tick(price=96.0, volume=1, timestamp=None))
    monkeypatch.setattr(settings, "auto_trade_held_review_buffer_pct", 0.05)
    monkeypatch.setattr(settings, "held_review_min_interval_days", 1)
    monkeypatch.setattr(auto_trade_loop, "get_last_decision_time", fake_last_decision_time)

    result = await auto_trade_loop._held_needing_review(["005930.KS"], "domestic")
    assert result == ["005930.KS"]


async def test_warn_overnight_gap_risk_noop_while_market_open(monkeypatch):
    monkeypatch.setattr(auto_trade_loop, "is_domestic_market_open", lambda: True)

    async def fail_if_called():
        raise AssertionError("must not read the account while the market is still open")

    monkeypatch.setattr("app.web.state.kis_account_snapshot", fail_if_called)

    await auto_trade_loop._warn_overnight_gap_risk(["005930.KS"], "domestic")  # no raise = pass


async def test_warn_overnight_gap_risk_fires_for_ticker_near_stop(monkeypatch):
    recorded = []

    async def fake_snapshot():
        return {"cash_deposit": 1.0}, {"005930.KS": {"quantity": 5}}

    async def fake_get_open_stop_loss(ticker):
        return (1, 95.0)

    async def fake_record_alert(alert):
        recorded.append(alert)

    monkeypatch.setattr(auto_trade_loop, "is_domestic_market_open", lambda: False)
    monkeypatch.setattr("app.web.state.kis_account_snapshot", fake_snapshot)
    monkeypatch.setattr(auto_trade_loop, "get_open_stop_loss", fake_get_open_stop_loss)
    monkeypatch.setattr(auto_trade_loop, "latest", lambda ticker: Tick(price=96.0, volume=1, timestamp=None))
    monkeypatch.setattr(auto_trade_loop, "record_alert", fake_record_alert)
    monkeypatch.setattr(settings, "overnight_gap_warning_buffer_pct", 0.08)

    await auto_trade_loop._warn_overnight_gap_risk(["005930.KS"], "domestic")

    assert len(recorded) == 1
    assert recorded[0].kind == "overnight_gap_risk"
    assert recorded[0].ticker == "005930.KS"


async def test_warn_overnight_gap_risk_skips_ticker_far_from_stop(monkeypatch):
    async def fake_snapshot():
        return {"cash_deposit": 1.0}, {"005930.KS": {"quantity": 5}}

    async def fake_get_open_stop_loss(ticker):
        return (1, 95.0)

    async def fail_if_called(alert):
        raise AssertionError("must not fire an alert this far from the stop")

    monkeypatch.setattr(auto_trade_loop, "is_domestic_market_open", lambda: False)
    monkeypatch.setattr("app.web.state.kis_account_snapshot", fake_snapshot)
    monkeypatch.setattr(auto_trade_loop, "get_open_stop_loss", fake_get_open_stop_loss)
    monkeypatch.setattr(auto_trade_loop, "latest", lambda ticker: Tick(price=150.0, volume=1, timestamp=None))
    monkeypatch.setattr(auto_trade_loop, "record_alert", fail_if_called)
    monkeypatch.setattr(settings, "overnight_gap_warning_buffer_pct", 0.08)

    await auto_trade_loop._warn_overnight_gap_risk(["005930.KS"], "domestic")


async def test_warn_overnight_gap_risk_skips_ticker_with_zero_real_shares(monkeypatch):
    async def fake_snapshot():
        return {"cash_deposit": 1.0}, {}  # no real position

    async def fail_if_called(ticker):
        raise AssertionError("must not even look up a stop-loss for a ticker with no real shares")

    monkeypatch.setattr(auto_trade_loop, "is_domestic_market_open", lambda: False)
    monkeypatch.setattr("app.web.state.kis_account_snapshot", fake_snapshot)
    monkeypatch.setattr(auto_trade_loop, "get_open_stop_loss", fail_if_called)

    await auto_trade_loop._warn_overnight_gap_risk(["005930.KS"], "domestic")  # no raise = pass


async def test_warn_overnight_gap_risk_warns_when_kis_snapshot_unavailable(monkeypatch, caplog):
    async def fake_snapshot():
        return None, {}

    monkeypatch.setattr(auto_trade_loop, "is_domestic_market_open", lambda: False)
    monkeypatch.setattr("app.web.state.kis_account_snapshot", fake_snapshot)

    with caplog.at_level("WARNING", logger="app.live.auto_trade_loop"):
        await auto_trade_loop._warn_overnight_gap_risk(["005930.KS"], "domestic")

    assert any(
        "KIS account snapshot unavailable" in r.message and "_warn_overnight_gap_risk" in r.message
        for r in caplog.records
    )


async def test_warn_overnight_gap_risk_dedups_within_same_day(monkeypatch):
    recorded = []

    async def fake_snapshot():
        return {"cash_deposit": 1.0}, {"005930.KS": {"quantity": 5}}

    async def fake_get_open_stop_loss(ticker):
        return (1, 95.0)

    async def fake_record_alert(alert):
        recorded.append(alert)

    fixed_clock = _FixedClock(date(2026, 8, 12))
    monkeypatch.setattr(auto_trade_loop, "get_clock", lambda: fixed_clock)
    monkeypatch.setattr(clock_module, "get_clock", lambda: fixed_clock)
    monkeypatch.setattr(auto_trade_loop, "is_domestic_market_open", lambda: False)
    monkeypatch.setattr("app.web.state.kis_account_snapshot", fake_snapshot)
    monkeypatch.setattr(auto_trade_loop, "get_open_stop_loss", fake_get_open_stop_loss)
    monkeypatch.setattr(auto_trade_loop, "latest", lambda ticker: Tick(price=96.0, volume=1, timestamp=None))
    monkeypatch.setattr(auto_trade_loop, "record_alert", fake_record_alert)
    monkeypatch.setattr(settings, "overnight_gap_warning_buffer_pct", 0.08)

    await auto_trade_loop._warn_overnight_gap_risk(["005930.KS"], "domestic")
    await auto_trade_loop._warn_overnight_gap_risk(["005930.KS"], "domestic")  # same closed-market cycle, later poll

    assert len(recorded) == 1  # not re-fired within the same calendar day


async def test_warn_overnight_gap_risk_dedup_is_per_account_mode(monkeypatch):
    """A paper-mode gap warning must not suppress the real-money one for the
    same ticker on the same calendar day — the mode is a runtime-switchable
    contextvar (app.config.current_account_mode)."""
    recorded = []

    async def fake_snapshot():
        return {"cash_deposit": 1.0}, {"005930.KS": {"quantity": 5}}

    async def fake_get_open_stop_loss(ticker):
        return (1, 95.0)

    async def fake_record_alert(alert):
        recorded.append(alert)

    fixed_clock = _FixedClock(date(2026, 8, 12))
    monkeypatch.setattr(auto_trade_loop, "get_clock", lambda: fixed_clock)
    monkeypatch.setattr(clock_module, "get_clock", lambda: fixed_clock)
    monkeypatch.setattr(auto_trade_loop, "is_domestic_market_open", lambda: False)
    monkeypatch.setattr("app.web.state.kis_account_snapshot", fake_snapshot)
    monkeypatch.setattr(auto_trade_loop, "get_open_stop_loss", fake_get_open_stop_loss)
    monkeypatch.setattr(auto_trade_loop, "latest", lambda ticker: Tick(price=96.0, volume=1, timestamp=None))
    monkeypatch.setattr(auto_trade_loop, "record_alert", fake_record_alert)
    monkeypatch.setattr(settings, "overnight_gap_warning_buffer_pct", 0.08)

    with account_mode_scope("paper"):
        await auto_trade_loop._warn_overnight_gap_risk(["005930.KS"], "domestic")
    with account_mode_scope("real"):
        await auto_trade_loop._warn_overnight_gap_risk(["005930.KS"], "domestic")  # same day, different mode

    assert len(recorded) == 2  # real-money warning is not swallowed by paper's cooldown


async def test_eligible_universe_keeps_never_analyzed_ticker(monkeypatch):
    async def fake_last_decision_time(ticker):
        return None  # never analyzed — always eligible, regardless of the interval

    monkeypatch.setattr(auto_trade_loop, "get_last_decision_time", fake_last_decision_time)
    result = await auto_trade_loop._eligible_universe(["005930.KS"], [], "domestic", max_open=99)
    assert result == ["005930.KS"]


async def test_eligible_universe_drops_ticker_within_rebalance_interval(monkeypatch):
    fixed_clock = _FixedClock(date(2026, 8, 3))
    monkeypatch.setattr(auto_trade_loop, "get_clock", lambda: fixed_clock)
    # local_today (used by _held_needing_review/_eligible_universe's shared
    # _partition_analyzed_within) reads app.clock's OWN get_clock(), not
    # auto_trade_loop's imported binding of it.
    monkeypatch.setattr(clock_module, "get_clock", lambda: fixed_clock)
    monkeypatch.setattr(settings, "rebalance_interval_days", 5)

    async def fake_last_decision_time(ticker):
        return datetime(2026, 8, 1, 3, 0, 0)  # 2 days ago — inside the 5-day interval

    monkeypatch.setattr(auto_trade_loop, "get_last_decision_time", fake_last_decision_time)
    result = await auto_trade_loop._eligible_universe(["005930.KS"], [], "domestic", max_open=99)
    assert result == []


async def test_eligible_universe_keeps_ticker_past_rebalance_interval(monkeypatch):
    fixed_clock = _FixedClock(date(2026, 8, 3))
    monkeypatch.setattr(auto_trade_loop, "get_clock", lambda: fixed_clock)
    # local_today (used by _held_needing_review/_eligible_universe's shared
    # _partition_analyzed_within) reads app.clock's OWN get_clock(), not
    # auto_trade_loop's imported binding of it.
    monkeypatch.setattr(clock_module, "get_clock", lambda: fixed_clock)
    monkeypatch.setattr(settings, "rebalance_interval_days", 5)

    async def fake_last_decision_time(ticker):
        return datetime(2026, 7, 27, 3, 0, 0)  # 7 days ago — past the 5-day interval

    monkeypatch.setattr(auto_trade_loop, "get_last_decision_time", fake_last_decision_time)
    result = await auto_trade_loop._eligible_universe(["005930.KS"], [], "domestic", max_open=99)
    assert result == ["005930.KS"]


async def test_eligible_universe_drops_tickers_over_max_open_positions_cap(monkeypatch):
    # The two eligibility checks are both applied here now (rebalance
    # throttle + max-open-positions) — this covers the second one, which
    # used to be _drop_new_positions_over_cap applied downstream of ranking.
    async def fake_snapshot():
        return {"cash_deposit": 1.0}, {"005930.KS": {"quantity": 5}}  # one already-open position

    monkeypatch.setattr("app.web.state.kis_account_snapshot", fake_snapshot)
    result = await auto_trade_loop._eligible_universe(
        ["005930.KS", "000660.KS"], ["005930.KS"], "domestic", max_open=1
    )
    assert result == ["005930.KS"]  # already-held ticker kept, new candidate dropped — cap is full


async def test_eligible_universe_warns_when_entire_universe_is_locked_out(monkeypatch, caplog):
    # 2026-08-26 regression test: universe > 0 but nothing survives means
    # every candidate is throttled/capped, not a quiet market — must warn,
    # with the unlock date, so this can't go unnoticed again.
    fixed_clock = _FixedClock(date(2026, 8, 26))
    monkeypatch.setattr(auto_trade_loop, "get_clock", lambda: fixed_clock)
    monkeypatch.setattr(clock_module, "get_clock", lambda: fixed_clock)
    monkeypatch.setattr(settings, "rebalance_interval_days", 5)

    async def fake_last_decision_time(ticker):
        return datetime(2026, 8, 25, 3, 0, 0)  # analyzed yesterday — locked until 2026-08-30

    monkeypatch.setattr(auto_trade_loop, "get_last_decision_time", fake_last_decision_time)

    with caplog.at_level("WARNING", logger="app.live.auto_trade_loop"):
        result = await auto_trade_loop._eligible_universe(["MRVL", "NVDA"], [], "overseas", max_open=99)

    assert result == []
    assert any(
        "overseas" in r.message and "2 ticker(s)" in r.message and "2026-08-30" in r.message
        and r.levelname == "WARNING"
        for r in caplog.records
    )


async def test_eligible_universe_no_warning_on_genuinely_empty_universe(monkeypatch, caplog):
    # An empty universe (data-source failure, handled by the caller's own
    # "candidate universe is empty" warning) must not ALSO trigger the
    # starvation warning — that warning means "something was locked out,"
    # which is a different, more specific claim than "nothing was fetched."
    with caplog.at_level("WARNING", logger="app.live.auto_trade_loop"):
        result = await auto_trade_loop._eligible_universe([], [], "domestic", max_open=99)

    assert result == []
    assert not any("ineligible this cycle" in r.message for r in caplog.records)


async def test_eligible_universe_detail_returns_locked_unlock_dates(monkeypatch):
    """2026-09-09 §0단계 계측: _eligible_universe_detail's second return value
    must be the same locked-ticker -> unlock-date map _partition_analyzed_
    within produces — app.live.screening_log needs this to tell "locked by
    the 5-day throttle" apart from "locked by the max_open cap"."""
    fixed_clock = _FixedClock(date(2026, 8, 26))
    monkeypatch.setattr(auto_trade_loop, "get_clock", lambda: fixed_clock)
    monkeypatch.setattr(clock_module, "get_clock", lambda: fixed_clock)
    monkeypatch.setattr(settings, "rebalance_interval_days", 5)

    async def fake_last_decision_time(ticker):
        return datetime(2026, 8, 25, 3, 0, 0)  # locked until 2026-08-30

    monkeypatch.setattr(auto_trade_loop, "get_last_decision_time", fake_last_decision_time)
    eligible, locked = await auto_trade_loop._eligible_universe_detail(
        ["005930.KS"], [], "domestic", max_open=99
    )
    assert eligible == []
    assert locked == {"005930.KS": date(2026, 8, 30)}


async def test_eligible_universe_wrapper_matches_detail_survivors(monkeypatch):
    """The thin wrapper must return exactly _eligible_universe_detail's first
    element — same idiom as _drop_analyzed_within/_partition_analyzed_within."""
    async def fake_last_decision_time(ticker):
        return None

    monkeypatch.setattr(auto_trade_loop, "get_last_decision_time", fake_last_decision_time)
    detail_eligible, _locked = await auto_trade_loop._eligible_universe_detail(
        ["005930.KS", "000660.KS"], [], "domestic", max_open=99
    )
    wrapper_eligible = await auto_trade_loop._eligible_universe(
        ["005930.KS", "000660.KS"], [], "domestic", max_open=99
    )
    assert wrapper_eligible == detail_eligible == ["005930.KS", "000660.KS"]


async def test_select_market_candidates_final_identical_regardless_of_observational_scoring(monkeypatch):
    """2026-09-09 §0단계 계측 core regression: whether
    screening_snapshot_observational_scoring_enabled is on or off, `final`
    (the actual trading candidates) must come out byte-identical. A locked-
    out ticker is given a huge observe-only score — if it ever leaked into
    the real ranking, it would dominate `final`; it must not."""
    monkeypatch.setattr(settings, "auto_trade_max_tickers_domestic", 2)
    monkeypatch.setattr(settings, "auto_trade_candidate_overfetch_multiplier", 1)
    fixed_clock = _FixedClock(date(2026, 8, 26))
    monkeypatch.setattr(auto_trade_loop, "get_clock", lambda: fixed_clock)
    monkeypatch.setattr(clock_module, "get_clock", lambda: fixed_clock)
    monkeypatch.setattr(settings, "rebalance_interval_days", 5)

    locked = {"999999.KS"}  # would otherwise dominate the ranking if it leaked in

    async def fake_last_decision_time(ticker):
        return datetime(2026, 8, 25, 3, 0, 0) if ticker in locked else None

    async def fake_krx_top_n(broker, count, *, max_price=None):
        return ["999999.KS", "005930.KS", "000660.KS"]

    async def fake_screening_pass(universe, top_n=5, lookback_days=10, observe_only=None, **kwargs):
        # Real ranking: descending score by input order, exactly like
        # _as_screening_pass above. observe_only tickers get a score that
        # would trivially top the ranking if it ever reached `ranked`.
        scored = {
            t: MomentumScore(raw_pct=0.0, atr_pct=None, score=float(len(universe) - i))
            for i, t in enumerate(universe)
        }
        ranked = [(t, scored[t].score) for t in universe[:top_n]]
        observed = {t: MomentumScore(raw_pct=0.0, atr_pct=None, score=9999.0) for t in (observe_only or [])}
        return ScreeningPass(scored=scored, ranked=ranked, score_key="score", observed=observed)

    monkeypatch.setattr(auto_trade_loop, "get_last_decision_time", fake_last_decision_time)
    monkeypatch.setattr(auto_trade_loop, "krx_top_n", fake_krx_top_n)
    monkeypatch.setattr(auto_trade_loop, "screening_pass", fake_screening_pass)

    monkeypatch.setattr(settings, "screening_snapshot_observational_scoring_enabled", True)
    result_on = await auto_trade_loop._domestic_candidates(broker=None)

    monkeypatch.setattr(settings, "screening_snapshot_observational_scoring_enabled", False)
    result_off = await auto_trade_loop._domestic_candidates(broker=None)

    assert result_on == result_off == ["005930.KS", "000660.KS"]  # locked ticker never wins a slot


async def test_select_market_candidates_passes_observed_and_rebalance_locked_to_snapshot(monkeypatch):
    """The observe_only set handed to screening_pass must be exactly
    universe - executable, and the resulting ScreeningPass.observed /
    _eligible_universe_detail's locked map must both reach
    record_screening_snapshot unchanged."""
    monkeypatch.setattr(settings, "auto_trade_max_tickers_domestic", 2)
    monkeypatch.setattr(settings, "auto_trade_candidate_overfetch_multiplier", 1)
    monkeypatch.setattr(settings, "screening_snapshot_observational_scoring_enabled", True)
    fixed_clock = _FixedClock(date(2026, 8, 26))
    monkeypatch.setattr(auto_trade_loop, "get_clock", lambda: fixed_clock)
    monkeypatch.setattr(clock_module, "get_clock", lambda: fixed_clock)
    monkeypatch.setattr(settings, "rebalance_interval_days", 5)

    async def fake_last_decision_time(ticker):
        return datetime(2026, 8, 25, 3, 0, 0) if ticker == "999999.KS" else None

    async def fake_krx_top_n(broker, count, *, max_price=None):
        return ["999999.KS", "005930.KS", "000660.KS"]

    seen_observe_only: list[str] = []

    async def fake_screening_pass(universe, top_n=5, lookback_days=10, observe_only=None, **kwargs):
        seen_observe_only.extend(observe_only or [])
        scored = {t: MomentumScore(raw_pct=0.0, atr_pct=None, score=1.0) for t in universe}
        ranked = [(t, scored[t].score) for t in universe[:top_n]]
        observed = {t: MomentumScore(raw_pct=0.0, atr_pct=None, score=1.0) for t in (observe_only or [])}
        return ScreeningPass(scored=scored, ranked=ranked, score_key="score", observed=observed)

    captured: dict = {}

    async def fake_record_screening_snapshot(ctx, *, observed=None, rebalance_locked=None, **kwargs):
        captured["observed"] = observed
        captured["rebalance_locked"] = rebalance_locked
        return 0

    monkeypatch.setattr(auto_trade_loop, "get_last_decision_time", fake_last_decision_time)
    monkeypatch.setattr(auto_trade_loop, "krx_top_n", fake_krx_top_n)
    monkeypatch.setattr(auto_trade_loop, "screening_pass", fake_screening_pass)
    import app.live.screening_log as screening_log_module
    monkeypatch.setattr(screening_log_module, "record_screening_snapshot", fake_record_screening_snapshot)

    await auto_trade_loop._domestic_candidates(broker=None)

    assert seen_observe_only == ["999999.KS"]
    assert set(captured["observed"]) == {"999999.KS"}
    assert captured["rebalance_locked"] == {"999999.KS": date(2026, 8, 30)}


async def test_select_market_candidates_no_observe_only_when_flag_disabled(monkeypatch):
    """screening_snapshot_observational_scoring_enabled=False must pass an
    empty observe_only — the killswitch fully disables the extra scoring,
    not just the recording of it."""
    monkeypatch.setattr(settings, "auto_trade_max_tickers_domestic", 2)
    monkeypatch.setattr(settings, "auto_trade_candidate_overfetch_multiplier", 1)
    monkeypatch.setattr(settings, "screening_snapshot_observational_scoring_enabled", False)
    fixed_clock = _FixedClock(date(2026, 8, 26))
    monkeypatch.setattr(auto_trade_loop, "get_clock", lambda: fixed_clock)
    monkeypatch.setattr(clock_module, "get_clock", lambda: fixed_clock)
    monkeypatch.setattr(settings, "rebalance_interval_days", 5)

    async def fake_last_decision_time(ticker):
        return datetime(2026, 8, 25, 3, 0, 0) if ticker == "999999.KS" else None

    async def fake_krx_top_n(broker, count, *, max_price=None):
        return ["999999.KS", "005930.KS", "000660.KS"]

    seen_observe_only: list[str] = []

    async def fake_screening_pass(universe, top_n=5, lookback_days=10, observe_only=None, **kwargs):
        seen_observe_only.extend(observe_only or [])
        scored = {t: MomentumScore(raw_pct=0.0, atr_pct=None, score=1.0) for t in universe}
        ranked = [(t, scored[t].score) for t in universe[:top_n]]
        return ScreeningPass(scored=scored, ranked=ranked, score_key="score")

    monkeypatch.setattr(auto_trade_loop, "get_last_decision_time", fake_last_decision_time)
    monkeypatch.setattr(auto_trade_loop, "krx_top_n", fake_krx_top_n)
    monkeypatch.setattr(auto_trade_loop, "screening_pass", fake_screening_pass)

    await auto_trade_loop._domestic_candidates(broker=None)

    assert seen_observe_only == []


async def test_select_market_candidates_starvation_does_not_lock_out_lower_ranked_names(monkeypatch):
    # The actual bug this fix targets: locked top-ranked names must not
    # occupy ranking slots that a lower-ranked, still-eligible name could
    # otherwise fill. Before this fix, ranking (top_n=max_tickers) happened
    # BEFORE the rebalance throttle, so the top 2 locked names alone would
    # starve out MSFT/GOOGL every single cycle.
    monkeypatch.setattr(settings, "auto_trade_max_tickers_domestic", 2)
    monkeypatch.setattr(settings, "auto_trade_candidate_overfetch_multiplier", 1)
    monkeypatch.setattr(settings, "rebalance_interval_days", 5)
    fixed_clock = _FixedClock(date(2026, 8, 26))
    monkeypatch.setattr(auto_trade_loop, "get_clock", lambda: fixed_clock)
    monkeypatch.setattr(clock_module, "get_clock", lambda: fixed_clock)

    locked_until = {"005930.KS", "000660.KS"}  # the two top-ranked names, both locked

    async def fake_last_decision_time(ticker):
        return datetime(2026, 8, 25, 3, 0, 0) if ticker in locked_until else None

    async def fake_top_candidates(universe, top_n=5, lookback_days=10, **kwargs):
        # Ranking preserves the given (already-filtered) order — a real
        # momentum ranker would rank the SAME locked names highest every
        # cycle, which is exactly the failure mode this test guards against;
        # passing them straight through here after eligibility filtering
        # proves they never got a chance to occupy a slot in the first place.
        return universe[:top_n]

    async def fake_krx_top_n(broker, count, *, max_price=None):
        return ["005930.KS", "000660.KS", "035420.KS", "051910.KS"]

    monkeypatch.setattr(auto_trade_loop, "get_last_decision_time", fake_last_decision_time)
    monkeypatch.setattr(auto_trade_loop, "krx_top_n", fake_krx_top_n)
    monkeypatch.setattr(auto_trade_loop, "screening_pass", _as_screening_pass(fake_top_candidates))

    result = await auto_trade_loop._domestic_candidates(broker=None)
    assert result == ["035420.KS", "051910.KS"]  # locked leaders never occupy a slot


async def test_select_market_candidates_overfetch_backfills_when_a_leader_is_illiquid(monkeypatch):
    # The overfetch multiplier's whole purpose: an execution-quality filter
    # (illiquid spread here) dropping the TOP-ranked name must be backfilled
    # from a rank that would NOT have been reached without the overfetch —
    # 4 candidates, max_tickers=2: without overfetch (multiplier=1),
    # ranking would only ever see the top 2 ("005930.KS", "000660.KS"), and
    # dropping the leader would permanently leave just 1 ticker for the
    # cycle's 2-slot budget (the same "sticky filter behind a hard cut"
    # shape as the rebalance-throttle starvation bug this whole change
    # fixes, just for a different, non-sticky filter). With
    # multiplier=2, ranking sees all 4, so the leader's drop is backfilled
    # by the 3rd-ranked name instead of leaving a slot empty.
    monkeypatch.setattr(settings, "auto_trade_max_tickers_domestic", 2)
    monkeypatch.setattr(settings, "auto_trade_candidate_overfetch_multiplier", 2)

    async def fake_krx_top_n(broker, count, *, max_price=None):
        return ["005930.KS", "000660.KS", "035420.KS", "051910.KS"]

    async def fake_top_candidates(universe, top_n=5, lookback_days=10, **kwargs):
        return universe[:top_n]  # preserves rank order — first 2 would be the WITHOUT-overfetch pool

    async def fake_drop_illiquid(broker, tickers):
        return [t for t in tickers if t != "005930.KS"]  # the top-ranked name is illiquid

    monkeypatch.setattr(auto_trade_loop, "krx_top_n", fake_krx_top_n)
    monkeypatch.setattr(auto_trade_loop, "screening_pass", _as_screening_pass(fake_top_candidates))
    monkeypatch.setattr(auto_trade_loop, "_drop_illiquid", fake_drop_illiquid)

    result = await auto_trade_loop._domestic_candidates(broker=None)
    # Still 2 tickers, backfilled by the 3rd-ranked name ("035420.KS") —
    # never reachable without the overfetch, since top_n=2 alone would
    # have excluded it from ranking entirely.
    assert result == ["000660.KS", "035420.KS"]


async def test_select_market_candidates_momentum_branch_is_capped_at_max_tickers(monkeypatch):
    # The overfetched ranking pool must still be cut back down to
    # max_tickers once the execution-quality filters have had their turn —
    # overfetch only exists to backfill a dropped slot, not to spend more
    # budget than configured.
    monkeypatch.setattr(settings, "auto_trade_max_tickers_domestic", 2)
    monkeypatch.setattr(settings, "auto_trade_candidate_overfetch_multiplier", 2)

    async def fake_krx_top_n(broker, count, *, max_price=None):
        return ["005930.KS", "000660.KS", "035420.KS", "051910.KS"]

    async def fake_top_candidates(universe, top_n=5, lookback_days=10, **kwargs):
        return universe[:top_n]  # all 4 survive ranking/illiquidity untouched

    monkeypatch.setattr(auto_trade_loop, "krx_top_n", fake_krx_top_n)
    monkeypatch.setattr(auto_trade_loop, "screening_pass", _as_screening_pass(fake_top_candidates))

    result = await auto_trade_loop._domestic_candidates(broker=None)
    assert result == ["005930.KS", "000660.KS"]  # exactly max_tickers, not the whole overfetched pool


async def test_select_market_candidates_forced_reviews_are_not_squeezed_out_by_momentum_cap(monkeypatch):
    # A held ticker forced in via _held_needing_review must survive the
    # max_tickers cut even when the momentum pool alone already fills it —
    # a risk signal is never subject to the momentum budget.
    monkeypatch.setattr(settings, "auto_trade_max_tickers_domestic", 1)
    monkeypatch.setattr(settings, "auto_trade_candidate_overfetch_multiplier", 1)

    async def fake_krx_top_n(broker, count, *, max_price=None):
        return ["005930.KS", "000660.KS"]

    async def fake_top_candidates(universe, top_n=5, lookback_days=10, **kwargs):
        # "035420.KS" is in the eligible universe (it's on the trade
        # allowlist, prepended into `universe`) but is excluded here to
        # simulate it not ranking on momentum merits alone — it only
        # reaches the final result via the forced held-review path, which
        # is exactly what this test needs to isolate.
        non_held = [t for t in universe if t != "035420.KS"]
        return non_held[:top_n]

    async def fake_snapshot():
        return {"cash_deposit": 1.0}, {"035420.KS": {"quantity": 5}}

    async def fake_get_open_stop_loss(ticker):
        return (1, 95.0)

    monkeypatch.setattr(auto_trade_loop, "krx_top_n", fake_krx_top_n)
    monkeypatch.setattr(auto_trade_loop, "screening_pass", _as_screening_pass(fake_top_candidates))
    monkeypatch.setattr("app.web.state.kis_account_snapshot", fake_snapshot)
    monkeypatch.setattr(auto_trade_loop, "get_open_stop_loss", fake_get_open_stop_loss)
    monkeypatch.setattr(auto_trade_loop, "latest", lambda ticker: Tick(price=96.0, volume=1, timestamp=None))
    monkeypatch.setattr(settings, "auto_trade_held_review_buffer_pct", 0.05)
    _set_trade_allowlist(monkeypatch, ["035420.KS"])

    result = await auto_trade_loop._domestic_candidates(broker=None)
    assert "035420.KS" in result  # forced risk-review ticker survives
    assert len(result) == 2  # 1 momentum slot (max_tickers=1) + the forced review, not squeezed to 1


# ---------------------------------------------------------------------------
# _drop_unexecutable — 2026-09-01 audit #6: drop a candidate that can't be
# sized to even 1 share against the real account, BEFORE momentum ranking.
# ---------------------------------------------------------------------------


def _fake_portfolio_snapshot(per_ticker_value, total_value):
    from app.execution.portfolio_risk import PortfolioSnapshot

    return PortfolioSnapshot(per_ticker_value=per_ticker_value, total_value=total_value)


def _stub_quotes(monkeypatch, prices: dict, atrs: dict):
    async def fake_get_last_close(ticker, as_of=None):
        return prices[ticker]

    async def fake_get_atr_pct(ticker, as_of=None):
        return atrs.get(ticker)

    monkeypatch.setattr("app.dataflows.kis_provider.get_last_close", fake_get_last_close)
    monkeypatch.setattr("app.dataflows.kis_provider.get_atr_pct", fake_get_atr_pct)


async def test_drop_unexecutable_removes_only_unaffordable_and_keeps_order(monkeypatch):
    monkeypatch.setattr(settings, "auto_trade_skip_unexecutable_candidates", True)

    async def fake_snapshot(market):
        return _fake_portfolio_snapshot({}, 200_000.0)

    monkeypatch.setattr("app.execution.portfolio_risk.real_account_snapshot", fake_snapshot)
    # 005930.KS: high ATR -> risk-capped to 0 shares (the audit's own sample).
    # 000660.KS: affordable at every cap.
    _stub_quotes(
        monkeypatch,
        prices={"005930.KS": 61_000.0, "000660.KS": 12_000.0},
        atrs={"005930.KS": 0.05, "000660.KS": 0.02},
    )

    result = await auto_trade_loop._drop_unexecutable(["005930.KS", "000660.KS"], "domestic")
    assert result == ["000660.KS"]  # order preserved, only the unaffordable one dropped


async def test_drop_unexecutable_kill_switch_skips_account_lookup_entirely(monkeypatch):
    monkeypatch.setattr(settings, "auto_trade_skip_unexecutable_candidates", False)

    async def fail_if_called(market):
        raise AssertionError("cached_account_snapshot must not be called when the flag is off")

    monkeypatch.setattr("app.execution.portfolio_risk.real_account_snapshot", fail_if_called)

    result = await auto_trade_loop._drop_unexecutable(["005930.KS"], "domestic")
    assert result == ["005930.KS"]


async def test_drop_unexecutable_fails_closed_when_account_unavailable(monkeypatch, caplog):
    import logging

    from app.execution.portfolio_risk import AccountUnavailable

    monkeypatch.setattr(settings, "auto_trade_skip_unexecutable_candidates", True)

    async def fake_snapshot(market):
        raise AccountUnavailable("no broker configured")

    monkeypatch.setattr("app.execution.portfolio_risk.real_account_snapshot", fake_snapshot)

    with caplog.at_level(logging.WARNING):
        result = await auto_trade_loop._drop_unexecutable(["005930.KS", "000660.KS"], "domestic")
    assert result == ["005930.KS", "000660.KS"]  # never computes affordability against a fabricated balance
    assert any("skipping the purchasability filter" in msg for msg in caplog.messages)


async def test_drop_unexecutable_keeps_all_when_equity_non_positive(monkeypatch):
    monkeypatch.setattr(settings, "auto_trade_skip_unexecutable_candidates", True)

    async def fake_snapshot(market):
        return _fake_portfolio_snapshot({}, 0.0)

    monkeypatch.setattr("app.execution.portfolio_risk.real_account_snapshot", fake_snapshot)

    result = await auto_trade_loop._drop_unexecutable(["005930.KS"], "domestic")
    assert result == ["005930.KS"]


async def test_drop_unexecutable_keeps_ticker_on_price_lookup_failure(monkeypatch):
    monkeypatch.setattr(settings, "auto_trade_skip_unexecutable_candidates", True)

    async def fake_snapshot(market):
        return _fake_portfolio_snapshot({}, 200_000.0)

    async def failing_get_last_close(ticker, as_of=None):
        raise RuntimeError("quote unavailable")

    async def fake_get_atr_pct(ticker, as_of=None):
        return 0.02

    monkeypatch.setattr("app.execution.portfolio_risk.real_account_snapshot", fake_snapshot)
    monkeypatch.setattr("app.dataflows.kis_provider.get_last_close", failing_get_last_close)
    monkeypatch.setattr("app.dataflows.kis_provider.get_atr_pct", fake_get_atr_pct)

    result = await auto_trade_loop._drop_unexecutable(["005930.KS"], "domestic")
    assert result == ["005930.KS"]  # a failed quote never causes a drop


async def test_drop_unexecutable_drops_on_missing_atr_by_default(monkeypatch):
    # 2026-09-02 (decision id=1169, 950260.KS): a ticker whose ATR lookup
    # fails (or returns None/0/negative) is now DROPPED by default, the one
    # deliberate exception to "never stricter than the real gate" — see
    # blocks_new_entry_without_atr / auto_trade_require_atr_for_new_entry.
    # Before this fix, missing ATR was treated as "risk cap uncapped" and
    # the candidate was kept; that let a candidate through that
    # app.execution.ledger.rebalance_to_target would then refuse anyway.
    monkeypatch.setattr(settings, "auto_trade_skip_unexecutable_candidates", True)
    monkeypatch.setattr(settings, "auto_trade_require_atr_for_new_entry", True)

    async def fake_snapshot(market):
        return _fake_portfolio_snapshot({}, 200_000.0)

    async def fake_get_last_close(ticker, as_of=None):
        return 12_000.0  # affordable on allocation/cash alone

    async def failing_get_atr_pct(ticker, as_of=None):
        raise RuntimeError("ATR unavailable")

    monkeypatch.setattr("app.execution.portfolio_risk.real_account_snapshot", fake_snapshot)
    monkeypatch.setattr("app.dataflows.kis_provider.get_last_close", fake_get_last_close)
    monkeypatch.setattr("app.dataflows.kis_provider.get_atr_pct", failing_get_atr_pct)

    result = await auto_trade_loop._drop_unexecutable(["005930.KS"], "domestic")
    assert result == []


async def test_drop_unexecutable_keeps_missing_atr_ticker_when_kill_switch_off(monkeypatch):
    # Same setup as above, but with auto_trade_require_atr_for_new_entry=False
    # (restoring pre-2026-09-02 behavior exactly): missing ATR goes back to
    # "risk cap uncapped," never a drop — cash/allocation caps still apply.
    monkeypatch.setattr(settings, "auto_trade_skip_unexecutable_candidates", True)
    monkeypatch.setattr(settings, "auto_trade_require_atr_for_new_entry", False)

    async def fake_snapshot(market):
        return _fake_portfolio_snapshot({}, 200_000.0)

    async def fake_get_last_close(ticker, as_of=None):
        return 12_000.0  # affordable on allocation/cash alone

    async def failing_get_atr_pct(ticker, as_of=None):
        raise RuntimeError("ATR unavailable")

    monkeypatch.setattr("app.execution.portfolio_risk.real_account_snapshot", fake_snapshot)
    monkeypatch.setattr("app.dataflows.kis_provider.get_last_close", fake_get_last_close)
    monkeypatch.setattr("app.dataflows.kis_provider.get_atr_pct", failing_get_atr_pct)

    result = await auto_trade_loop._drop_unexecutable(["005930.KS"], "domestic")
    assert result == ["005930.KS"]


async def test_drop_unexecutable_exempts_currently_held_ticker(monkeypatch):
    monkeypatch.setattr(settings, "auto_trade_skip_unexecutable_candidates", True)

    async def fake_snapshot(market):
        # Held at a value that alone proves it's in the account; price below
        # is set absurdly high so it would otherwise be dropped.
        return _fake_portfolio_snapshot({"005930.KS": 61_000.0}, 200_000.0)

    monkeypatch.setattr("app.execution.portfolio_risk.real_account_snapshot", fake_snapshot)
    _stub_quotes(monkeypatch, prices={"005930.KS": 9_999_999.0}, atrs={"005930.KS": 0.10})

    result = await auto_trade_loop._drop_unexecutable(["005930.KS"], "domestic")
    assert result == ["005930.KS"]  # held tickers are exempt regardless of price


async def test_drop_unexecutable_does_not_exempt_allowlist_membership(monkeypatch):
    # Being on the trade allowlist doesn't make an unaffordable ticker
    # affordable — only a held position or forced risk-review does.
    monkeypatch.setattr(settings, "auto_trade_skip_unexecutable_candidates", True)

    async def fake_snapshot(market):
        return _fake_portfolio_snapshot({}, 200_000.0)

    monkeypatch.setattr("app.execution.portfolio_risk.real_account_snapshot", fake_snapshot)
    _stub_quotes(monkeypatch, prices={"005930.KS": 61_000.0}, atrs={"005930.KS": 0.05})

    result = await auto_trade_loop._drop_unexecutable(["005930.KS"], "domestic")
    assert result == []


# --- through _select_market_candidates (E2E placement/exemption checks) ----


async def test_select_market_candidates_purchasability_filter_runs_before_ranking(monkeypatch):
    # Starvation regression guard (audit #6 design §6): the top-ranked names
    # are unaffordable and there is NO overfetch — without placement before
    # ranking, this would leave the budget slots empty rather than
    # backfilling from the affordable names ranked lower.
    monkeypatch.setattr(settings, "auto_trade_max_tickers_domestic", 2)
    monkeypatch.setattr(settings, "auto_trade_candidate_overfetch_multiplier", 1)
    monkeypatch.setattr(settings, "auto_trade_skip_unexecutable_candidates", True)

    async def fake_krx_top_n(broker, count, *, max_price=None):
        return ["005930.KS", "000660.KS", "035420.KS", "051910.KS"]

    async def fake_top_candidates(universe, top_n=5, lookback_days=10, **kwargs):
        return universe[:top_n]  # preserves order

    async def fake_snapshot(market):
        return _fake_portfolio_snapshot({}, 200_000.0)

    # First two (would-be top-ranked) are unaffordable; last two are cheap.
    monkeypatch.setattr("app.execution.portfolio_risk.real_account_snapshot", fake_snapshot)
    _stub_quotes(
        monkeypatch,
        prices={"005930.KS": 9_999_999.0, "000660.KS": 9_999_999.0, "035420.KS": 10_000.0, "051910.KS": 10_000.0},
        atrs={"005930.KS": 0.02, "000660.KS": 0.02, "035420.KS": 0.02, "051910.KS": 0.02},
    )
    monkeypatch.setattr(auto_trade_loop, "krx_top_n", fake_krx_top_n)
    monkeypatch.setattr(auto_trade_loop, "screening_pass", _as_screening_pass(fake_top_candidates))

    result = await auto_trade_loop._domestic_candidates(broker=None)
    assert result == ["035420.KS", "051910.KS"]  # backfilled with NO overfetch needed


async def test_select_market_candidates_forced_review_ticker_bypasses_purchasability_filter(monkeypatch):
    # A held-review ticker merges in AFTER ranking (_select_market_candidates),
    # so it never passes through _drop_unexecutable at all — a risk signal
    # is never squeezed out on affordability grounds either.
    monkeypatch.setattr(settings, "auto_trade_max_tickers_domestic", 1)
    monkeypatch.setattr(settings, "auto_trade_candidate_overfetch_multiplier", 1)
    monkeypatch.setattr(settings, "auto_trade_skip_unexecutable_candidates", True)
    monkeypatch.setattr(settings, "auto_trade_held_review_buffer_pct", 0.05)
    _set_trade_allowlist(monkeypatch, ["035420.KS"])

    async def fake_krx_top_n(broker, count, *, max_price=None):
        return ["005930.KS"]

    async def fake_top_candidates(universe, top_n=5, lookback_days=10, **kwargs):
        non_held = [t for t in universe if t != "035420.KS"]
        return non_held[:top_n]

    async def fake_account_snapshot(market):
        return _fake_portfolio_snapshot({}, 200_000.0)

    async def fake_kis_account_snapshot():
        return {"cash_deposit": 1.0}, {"035420.KS": {"quantity": 5}}

    async def fake_get_open_stop_loss(ticker):
        return (1, 95.0)

    monkeypatch.setattr("app.execution.portfolio_risk.real_account_snapshot", fake_account_snapshot)
    # The held-review ticker is priced far beyond what the account could
    # ever afford — if it passed through _drop_unexecutable, it would be
    # dropped. It must survive anyway, via the forced-review path.
    _stub_quotes(monkeypatch, prices={"005930.KS": 61_000.0, "035420.KS": 9_999_999.0}, atrs={"005930.KS": 0.02, "035420.KS": 0.10})
    monkeypatch.setattr(auto_trade_loop, "krx_top_n", fake_krx_top_n)
    monkeypatch.setattr(auto_trade_loop, "screening_pass", _as_screening_pass(fake_top_candidates))
    monkeypatch.setattr("app.web.state.kis_account_snapshot", fake_kis_account_snapshot)
    monkeypatch.setattr(auto_trade_loop, "get_open_stop_loss", fake_get_open_stop_loss)
    monkeypatch.setattr(auto_trade_loop, "latest", lambda ticker: Tick(price=96.0, volume=1, timestamp=None))

    result = await auto_trade_loop._domestic_candidates(broker=None)
    assert "035420.KS" in result  # forced risk-review ticker survives despite being unaffordable


async def test_select_market_candidates_alerts_when_purchasability_filter_empties_universe(monkeypatch, caplog):
    import logging

    from app.live import alerts as alerts_module

    alerts_module._last_fired_at.clear()  # isolate from any (ticker, kind) fired by another test
    monkeypatch.setattr(settings, "auto_trade_skip_unexecutable_candidates", True)

    async def fake_krx_top_n(broker, count, *, max_price=None):
        return ["005930.KS", "000660.KS"]

    async def fake_top_candidates(universe, top_n=5, lookback_days=10, **kwargs):
        return universe[:top_n]

    async def fake_snapshot(market):
        return _fake_portfolio_snapshot({}, 200_000.0)

    recorded = []

    async def fake_record_alert(alert):
        recorded.append(alert)

    monkeypatch.setattr("app.execution.portfolio_risk.real_account_snapshot", fake_snapshot)
    _stub_quotes(
        monkeypatch,
        prices={"005930.KS": 9_999_999.0, "000660.KS": 9_999_999.0},
        atrs={"005930.KS": 0.05, "000660.KS": 0.05},
    )
    monkeypatch.setattr(auto_trade_loop, "krx_top_n", fake_krx_top_n)
    monkeypatch.setattr(auto_trade_loop, "screening_pass", _as_screening_pass(fake_top_candidates))
    monkeypatch.setattr(auto_trade_loop, "record_alert", fake_record_alert)

    with caplog.at_level(logging.WARNING):
        result = await auto_trade_loop._domestic_candidates(broker=None)

    assert result == []
    assert any("purchasability filter dropped every eligible candidate" in msg for msg in caplog.messages)
    assert len(recorded) == 1
    assert recorded[0].ticker == "SYSTEM"
    assert recorded[0].kind == "account_too_small_domestic"


async def test_select_market_candidates_no_alert_when_universe_itself_was_empty(monkeypatch, caplog):
    # _eligible_universe already warns for this case (universe > 0, eligible
    # == 0) — the purchasability alert must not ALSO fire and double up the
    # diagnosis when there was nothing eligible to filter in the first place.
    import logging

    monkeypatch.setattr(settings, "auto_trade_skip_unexecutable_candidates", True)
    monkeypatch.setattr(settings, "rebalance_interval_days", 5)
    # Unrelated to this test (which is about _drop_unexecutable's alert
    # dedup, not the price ceiling) — disabled explicitly so the
    # fail_if_called guard below stays scoped to the ONE account lookup
    # this test actually cares about, not _domestic_price_ceiling's own
    # legitimate (and separate) cached_account_snapshot call.
    monkeypatch.setattr(settings, "auto_trade_universe_price_ceiling_domestic", False)

    async def fake_krx_top_n(broker, count, *, max_price=None):
        return ["005930.KS"]

    async def fake_last_decision_time(ticker):
        return datetime(2026, 8, 25, 3, 0, 0)  # locked out by the rebalance throttle

    fixed_clock = _FixedClock(date(2026, 8, 26))

    recorded = []

    async def fake_record_alert(alert):
        recorded.append(alert)

    async def fail_if_called(market):
        raise AssertionError("no account lookup should happen when eligible is already empty")

    monkeypatch.setattr(auto_trade_loop, "get_clock", lambda: fixed_clock)
    monkeypatch.setattr(clock_module, "get_clock", lambda: fixed_clock)
    monkeypatch.setattr(auto_trade_loop, "get_last_decision_time", fake_last_decision_time)
    monkeypatch.setattr(auto_trade_loop, "krx_top_n", fake_krx_top_n)
    monkeypatch.setattr(auto_trade_loop, "record_alert", fake_record_alert)
    monkeypatch.setattr("app.execution.portfolio_risk.real_account_snapshot", fail_if_called)

    with caplog.at_level(logging.WARNING):
        result = await auto_trade_loop._domestic_candidates(broker=None)

    assert result == []
    assert not any("purchasability filter dropped every eligible candidate" in msg for msg in caplog.messages)
    assert recorded == []


# ---------------------------------------------------------------------------
# _cash_freed_up_for_review — bypasses rebalance_interval_days when a
# ticker's last decision was capped purely by the portfolio cash floor and
# there's now headroom above it again
# ---------------------------------------------------------------------------


def _fake_snapshot(per_ticker_value, total_value):
    from app.execution.portfolio_risk import PortfolioSnapshot

    return PortfolioSnapshot(per_ticker_value=per_ticker_value, total_value=total_value)


async def test_cash_freed_up_for_review_noop_when_no_ticker_was_cash_capped(monkeypatch):
    async def fake_capped(tickers):
        return []

    monkeypatch.setattr(auto_trade_loop, "tickers_cash_floor_capped", fake_capped)
    result = await auto_trade_loop._cash_freed_up_for_review(["005930.KS"], "domestic")
    assert result == []


async def test_cash_freed_up_for_review_bypasses_when_headroom_exists(monkeypatch):
    async def fake_capped(tickers):
        return ["005930.KS"]

    async def fake_snapshot(market):
        # total 100, held 70 -> 30% cash, well above a 20% floor.
        return _fake_snapshot({"005930.KS": 70.0}, 100.0)

    async def fake_floor_pct():
        return 0.20

    async def fake_last_decision_time(ticker):
        return None  # never analyzed under _drop_analyzed_within's own throttle -> eligible

    monkeypatch.setattr(auto_trade_loop, "tickers_cash_floor_capped", fake_capped)
    monkeypatch.setattr("app.execution.portfolio_risk.real_account_snapshot", fake_snapshot)
    monkeypatch.setattr("app.execution.market_regime.domestic_cash_floor_pct", fake_floor_pct)
    monkeypatch.setattr(auto_trade_loop, "get_last_decision_time", fake_last_decision_time)

    result = await auto_trade_loop._cash_freed_up_for_review(["005930.KS"], "domestic")
    assert result == ["005930.KS"]


async def test_cash_freed_up_for_review_noop_when_still_no_headroom(monkeypatch):
    async def fake_capped(tickers):
        return ["005930.KS"]

    async def fake_snapshot(market):
        # total 100, held 85 -> 15% cash, still below a 20% floor.
        return _fake_snapshot({"005930.KS": 85.0}, 100.0)

    async def fake_floor_pct():
        return 0.20

    monkeypatch.setattr(auto_trade_loop, "tickers_cash_floor_capped", fake_capped)
    monkeypatch.setattr("app.execution.portfolio_risk.real_account_snapshot", fake_snapshot)
    monkeypatch.setattr("app.execution.market_regime.domestic_cash_floor_pct", fake_floor_pct)

    result = await auto_trade_loop._cash_freed_up_for_review(["005930.KS"], "domestic")
    assert result == []


async def test_cash_freed_up_for_review_degrades_when_account_unavailable(monkeypatch):
    from app.execution.portfolio_risk import AccountUnavailable

    async def fake_capped(tickers):
        return ["005930.KS"]

    async def fake_snapshot(market):
        raise AccountUnavailable("KIS broker not connected")

    monkeypatch.setattr(auto_trade_loop, "tickers_cash_floor_capped", fake_capped)
    monkeypatch.setattr("app.execution.portfolio_risk.real_account_snapshot", fake_snapshot)

    result = await auto_trade_loop._cash_freed_up_for_review(["005930.KS"], "domestic")
    assert result == []


async def test_cash_freed_up_for_review_still_throttled_by_held_review_interval(monkeypatch):
    fixed_clock = _FixedClock(date(2026, 8, 3))
    monkeypatch.setattr(auto_trade_loop, "get_clock", lambda: fixed_clock)
    monkeypatch.setattr(clock_module, "get_clock", lambda: fixed_clock)
    monkeypatch.setattr(settings, "held_review_min_interval_days", 1)

    async def fake_capped(tickers):
        return ["005930.KS"]

    async def fake_snapshot(market):
        return _fake_snapshot({"005930.KS": 70.0}, 100.0)

    async def fake_floor_pct():
        return 0.20

    async def fake_last_decision_time(ticker):
        return datetime(2026, 8, 3, 3, 0, 0)  # analyzed earlier today — inside the 1-day interval

    monkeypatch.setattr(auto_trade_loop, "tickers_cash_floor_capped", fake_capped)
    monkeypatch.setattr("app.execution.portfolio_risk.real_account_snapshot", fake_snapshot)
    monkeypatch.setattr("app.execution.market_regime.domestic_cash_floor_pct", fake_floor_pct)
    monkeypatch.setattr(auto_trade_loop, "get_last_decision_time", fake_last_decision_time)

    result = await auto_trade_loop._cash_freed_up_for_review(["005930.KS"], "domestic")
    assert result == []  # headroom exists, but still re-analyzed too recently


async def test_drop_new_positions_over_cap_passes_through_when_under_cap(monkeypatch):
    async def fake_snapshot():
        return {"cash_deposit": 1.0}, {"005930.KS": {"quantity": 5}}  # 1 open position

    monkeypatch.setattr("app.web.state.kis_account_snapshot", fake_snapshot)

    result = await auto_trade_loop._drop_new_positions_over_cap(
        ["000660.KS", "035420.KS"], trade_allowlist=["005930.KS"], max_open=2
    )
    assert result == ["000660.KS", "035420.KS"]


async def test_drop_new_positions_over_cap_drops_new_tickers_at_cap(monkeypatch):
    async def fake_snapshot():
        return {"cash_deposit": 1.0}, {
            "005930.KS": {"quantity": 5},
            "000660.KS": {"quantity": 3},
        }  # 2 open positions, at the cap

    monkeypatch.setattr("app.web.state.kis_account_snapshot", fake_snapshot)

    result = await auto_trade_loop._drop_new_positions_over_cap(
        ["000660.KS", "035420.KS"], trade_allowlist=["005930.KS", "000660.KS"], max_open=2
    )
    assert result == ["000660.KS"]  # already-held ticker kept, brand-new one dropped


async def test_drop_new_positions_over_cap_ignores_zero_quantity_trade_allowlist_entries(monkeypatch):
    # A trade_allowlist entry with no real shares (never bought, or fully sold)
    # must not count toward the open-position total.
    async def fake_snapshot():
        return {"cash_deposit": 1.0}, {"005930.KS": {"quantity": 0}}

    monkeypatch.setattr("app.web.state.kis_account_snapshot", fake_snapshot)

    result = await auto_trade_loop._drop_new_positions_over_cap(
        ["035420.KS"], trade_allowlist=["005930.KS"], max_open=1
    )
    assert result == ["035420.KS"]  # 0 open positions, well under the cap of 1


async def test_domestic_candidates_held_review_exempt_from_exclude(monkeypatch):
    # A held position near its stop-loss still gets reviewed even if
    # already analyzed today — the exclude filter is scoped to the
    # momentum pool, not this risk-driven forced-include.
    _set_trade_allowlist(monkeypatch, ["005930.KS"])

    async def fake_krx_top_n(broker, count, *, max_price=None):
        return []

    async def fake_held_needing_review(tickers, market):
        return ["005930.KS"]

    monkeypatch.setattr(auto_trade_loop, "krx_top_n", fake_krx_top_n)
    monkeypatch.setattr(auto_trade_loop, "_held_needing_review", fake_held_needing_review)
    result = await auto_trade_loop._domestic_candidates(broker=None, exclude=frozenset({"005930.KS"}))
    assert result == ["005930.KS"]


async def test_drop_execution_blocked_removes_flagged_tickers(monkeypatch):
    async def fake_pre_screening_blockers(broker, krx_code, ticker):
        return ["BLOCKED: VI"] if ticker == "005930.KS" else []

    monkeypatch.setattr(auto_trade_loop, "pre_screening_blockers", fake_pre_screening_blockers)
    result = await auto_trade_loop._drop_execution_blocked(None, ["005930.KS", "000660.KS"])
    assert result == ["000660.KS"]


async def test_drop_execution_blocked_keeps_candidate_when_check_errors(monkeypatch):
    async def raising_pre_screening_blockers(broker, krx_code, ticker):
        raise RuntimeError("KIS call failed")

    monkeypatch.setattr(auto_trade_loop, "pre_screening_blockers", raising_pre_screening_blockers)
    result = await auto_trade_loop._drop_execution_blocked(None, ["005930.KS"])
    assert result == ["005930.KS"]  # best-effort — never stricter than the real gate


async def test_drop_execution_blocked_passes_the_real_krx_code_not_a_string_slice(monkeypatch):
    # The code handed to KIS must come from krx_code_from_ticker, not from
    # chopping the last 3 characters off the ticker.
    seen = {}

    async def fake_pre_screening_blockers(broker, krx_code, ticker):
        seen[ticker] = krx_code
        return []

    monkeypatch.setattr(auto_trade_loop, "pre_screening_blockers", fake_pre_screening_blockers)
    await auto_trade_loop._drop_execution_blocked(None, ["005930.KS"])
    assert seen == {"005930.KS": "005930"}


async def test_drop_execution_blocked_skips_the_check_for_a_ticker_with_no_krx_code(monkeypatch):
    """A US symbol reaching this domestic-only filter must be kept unchecked,
    never asked about under a mangled code. `ticker[:-3]` turned "AAPL" into
    "A" and asked KIS about a completely different instrument; the guard now
    declines to answer instead. Kept (not dropped) because this filter is
    documented as never stricter than the real order-time gate."""
    called = []

    async def fake_pre_screening_blockers(broker, krx_code, ticker):
        called.append(ticker)
        return ["BLOCKED: whatever this code resolved to"]

    monkeypatch.setattr(auto_trade_loop, "pre_screening_blockers", fake_pre_screening_blockers)
    result = await auto_trade_loop._drop_execution_blocked(None, ["AAPL", "005930.KS"])

    assert called == ["005930.KS"]  # AAPL never reached the KIS check at all
    assert "AAPL" in result  # ...and survived as a candidate


async def test_drop_designated_removes_flagged_tickers(monkeypatch):
    async def fake_designation_block(broker, krx_code, ticker):
        return "BLOCKED: short overheat" if ticker == "005930.KS" else None

    monkeypatch.setattr(auto_trade_loop, "designation_block", fake_designation_block)
    result = await auto_trade_loop._drop_designated(None, ["005930.KS", "000660.KS"])
    assert result == ["000660.KS"]


async def test_drop_designated_keeps_candidate_when_check_errors(monkeypatch):
    async def raising_designation_block(broker, krx_code, ticker):
        raise RuntimeError("KIS call failed")

    monkeypatch.setattr(auto_trade_loop, "designation_block", raising_designation_block)
    result = await auto_trade_loop._drop_designated(None, ["005930.KS"])
    assert result == ["005930.KS"]  # best-effort — never stricter than the real gate


async def test_drop_designated_skips_the_check_for_a_ticker_with_no_krx_code(monkeypatch):
    called = []

    async def fake_designation_block(broker, krx_code, ticker):
        called.append(ticker)
        return "BLOCKED: whatever"

    monkeypatch.setattr(auto_trade_loop, "designation_block", fake_designation_block)
    result = await auto_trade_loop._drop_designated(None, ["AAPL", "005930.KS"])

    assert called == ["005930.KS"]  # AAPL never reached the KIS check at all
    assert "AAPL" in result


async def test_drop_illiquid_removes_wide_spread_tickers(monkeypatch):
    async def fake_spread_blocker(broker, ticker):
        return "BLOCKED: too wide" if ticker == "005930.KS" else None

    monkeypatch.setattr(auto_trade_loop, "spread_blocker", fake_spread_blocker)
    result = await auto_trade_loop._drop_illiquid(None, ["005930.KS", "000660.KS"])
    assert result == ["000660.KS"]


async def test_drop_illiquid_keeps_candidate_when_check_errors(monkeypatch):
    async def raising_spread_blocker(broker, ticker):
        raise RuntimeError("KIS call failed")

    monkeypatch.setattr(auto_trade_loop, "spread_blocker", raising_spread_blocker)
    result = await auto_trade_loop._drop_illiquid(None, ["005930.KS"])
    assert result == ["005930.KS"]  # best-effort — never stricter than the real gate


async def test_domestic_candidates_held_review_exempt_from_illiquid_filter(monkeypatch):
    # A held position drifting toward its stop-loss is a risk-driven forced
    # re-review — a wide spread (often the same stress event pushing it
    # toward the stop) must not silently drop it from this cycle's
    # candidates. See ai-cycle-audit-2026-08-10's "손절근접 종목이 유동성
    # 필터에 탈락" finding.
    _set_trade_allowlist(monkeypatch, ["005930.KS"])

    async def fake_krx_top_n(broker, count, *, max_price=None):
        return []

    async def fake_held_needing_review(tickers, market):
        return ["005930.KS"]

    async def fake_spread_blocker(broker, ticker):
        return "BLOCKED: too wide"  # every ticker looks illiquid

    monkeypatch.setattr(auto_trade_loop, "krx_top_n", fake_krx_top_n)
    monkeypatch.setattr(auto_trade_loop, "_held_needing_review", fake_held_needing_review)
    monkeypatch.setattr(auto_trade_loop, "spread_blocker", fake_spread_blocker)
    result = await auto_trade_loop._domestic_candidates(broker=None)
    assert result == ["005930.KS"]


async def test_domestic_candidates_non_held_ticker_still_dropped_by_illiquid_filter(monkeypatch):
    # The exemption above is scoped to held_review specifically — an
    # ordinary momentum candidate with a wide spread must still be dropped.
    _set_trade_allowlist(monkeypatch, [])
    monkeypatch.setattr(settings, "auto_trade_max_tickers_domestic", 10)

    async def fake_krx_top_n(broker, count, *, max_price=None):
        return ["005930.KS"]

    async def fake_top_candidates(universe, top_n=5, lookback_days=10, **kwargs):
        return universe

    async def fake_spread_blocker(broker, ticker):
        return "BLOCKED: too wide"

    monkeypatch.setattr(auto_trade_loop, "krx_top_n", fake_krx_top_n)
    monkeypatch.setattr(auto_trade_loop, "screening_pass", _as_screening_pass(fake_top_candidates))
    monkeypatch.setattr(auto_trade_loop, "spread_blocker", fake_spread_blocker)
    result = await auto_trade_loop._domestic_candidates(broker=None)
    assert result == []


async def test_overseas_candidates_held_review_exempt_from_illiquid_filter(monkeypatch):
    # Overseas counterpart of test_domestic_candidates_held_review_exempt_
    # from_illiquid_filter.
    _set_trade_allowlist(monkeypatch, ["AAPL"])

    async def fake_overseas_top_n(broker, count):
        return []

    async def fake_held_needing_review(tickers, market):
        return ["AAPL"]

    async def fake_spread_blocker(broker, ticker):
        return "BLOCKED: too wide"

    monkeypatch.setattr(auto_trade_loop, "overseas_top_n", fake_overseas_top_n)
    monkeypatch.setattr(auto_trade_loop, "_held_needing_review", fake_held_needing_review)
    monkeypatch.setattr(auto_trade_loop, "spread_blocker", fake_spread_blocker)
    result = await auto_trade_loop._overseas_candidates(broker=None)
    assert result == ["AAPL"]


async def test_overseas_candidates_excludes_illiquid_ticker(monkeypatch):
    _set_trade_allowlist(monkeypatch, [])
    monkeypatch.setattr(settings, "auto_trade_max_tickers_overseas", 10)

    async def fake_overseas_top_n(broker, count):
        return ["AAPL", "MSFT"]

    async def fake_top_candidates(universe, top_n=5, lookback_days=10, **kwargs):
        return universe

    async def fake_spread_blocker(broker, ticker):
        return "BLOCKED: too wide" if ticker == "AAPL" else None

    monkeypatch.setattr(auto_trade_loop, "overseas_top_n", fake_overseas_top_n)
    monkeypatch.setattr(auto_trade_loop, "screening_pass", _as_screening_pass(fake_top_candidates))
    monkeypatch.setattr(auto_trade_loop, "spread_blocker", fake_spread_blocker)
    result = await auto_trade_loop._overseas_candidates(broker=None)
    assert result == ["MSFT"]


async def test_domestic_candidates_excludes_execution_blocked_ticker(monkeypatch):
    _set_trade_allowlist(monkeypatch, [])
    monkeypatch.setattr(settings, "auto_trade_max_tickers_domestic", 10)

    async def fake_krx_top_n(broker, count, *, max_price=None):
        return ["005930.KS", "000660.KS"]

    async def fake_top_candidates(universe, top_n=5, lookback_days=10, **kwargs):
        return universe

    async def fake_pre_screening_blockers(broker, krx_code, ticker):
        return ["BLOCKED: trading-halted"] if ticker == "005930.KS" else []

    monkeypatch.setattr(auto_trade_loop, "krx_top_n", fake_krx_top_n)
    monkeypatch.setattr(auto_trade_loop, "screening_pass", _as_screening_pass(fake_top_candidates))
    monkeypatch.setattr(auto_trade_loop, "pre_screening_blockers", fake_pre_screening_blockers)
    result = await auto_trade_loop._domestic_candidates(broker=None)
    assert result == ["000660.KS"]


async def test_domestic_candidates_excludes_designated_ticker(monkeypatch):
    # §4-5 (2026-09-08): a plain momentum candidate carrying a KRX
    # designation (market-warning/short-overheat/liquidation) must be
    # dropped before it burns a full LLM analysis.
    _set_trade_allowlist(monkeypatch, [])
    monkeypatch.setattr(settings, "auto_trade_max_tickers_domestic", 10)

    async def fake_krx_top_n(broker, count, *, max_price=None):
        return ["005930.KS", "000660.KS"]

    async def fake_top_candidates(universe, top_n=5, lookback_days=10, **kwargs):
        return universe

    async def fake_designation_block(broker, krx_code, ticker):
        return "BLOCKED: short overheat" if ticker == "005930.KS" else None

    monkeypatch.setattr(auto_trade_loop, "krx_top_n", fake_krx_top_n)
    monkeypatch.setattr(auto_trade_loop, "screening_pass", _as_screening_pass(fake_top_candidates))
    monkeypatch.setattr(auto_trade_loop, "designation_block", fake_designation_block)
    result = await auto_trade_loop._domestic_candidates(broker=None)
    assert result == ["000660.KS"]


async def test_domestic_candidates_held_review_exempt_from_designation_filter(monkeypatch):
    # §4-5 (2026-09-08): the whole reason _drop_designated is a SEPARATE
    # filter from pre_screening_blockers — a position just designated
    # administrative/market-warned is exactly the moment the forced
    # held-position review exists to catch, not a candidate to silently
    # drop before the AI ever sees it.
    _set_trade_allowlist(monkeypatch, ["005930.KS"])

    async def fake_krx_top_n(broker, count, *, max_price=None):
        return []

    async def fake_held_needing_review(tickers, market):
        return ["005930.KS"]

    async def fake_designation_block(broker, krx_code, ticker):
        return "BLOCKED: market-warning"  # every ticker looks designated

    monkeypatch.setattr(auto_trade_loop, "krx_top_n", fake_krx_top_n)
    monkeypatch.setattr(auto_trade_loop, "_held_needing_review", fake_held_needing_review)
    monkeypatch.setattr(auto_trade_loop, "designation_block", fake_designation_block)
    result = await auto_trade_loop._domestic_candidates(broker=None)
    assert result == ["005930.KS"]


async def test_domestic_candidates_non_held_ticker_still_dropped_by_designation_filter(monkeypatch):
    # The exemption above is scoped to the forced risk-review set
    # specifically — an ordinary momentum candidate must still be dropped.
    _set_trade_allowlist(monkeypatch, [])
    monkeypatch.setattr(settings, "auto_trade_max_tickers_domestic", 10)

    async def fake_krx_top_n(broker, count, *, max_price=None):
        return ["005930.KS"]

    async def fake_top_candidates(universe, top_n=5, lookback_days=10, **kwargs):
        return universe

    async def fake_designation_block(broker, krx_code, ticker):
        return "BLOCKED: market-warning"

    monkeypatch.setattr(auto_trade_loop, "krx_top_n", fake_krx_top_n)
    monkeypatch.setattr(auto_trade_loop, "screening_pass", _as_screening_pass(fake_top_candidates))
    monkeypatch.setattr(auto_trade_loop, "designation_block", fake_designation_block)
    result = await auto_trade_loop._domestic_candidates(broker=None)
    assert result == []


class _FakeMarketFeed:
    MAX_REGISTRATIONS = 40

    def __init__(self, *, accept: bool = True):
        self.subscribed = []
        self.cb_only_calls = []
        self._accept = accept

    async def add_code(self, code, *, orderbook: bool = False, cb_only: bool = False):
        self.subscribed.append(code)
        self.cb_only_calls.append(cb_only)
        return self._accept


async def test_ensure_subscribed_requests_every_domestic_code(monkeypatch):
    from app.web import state

    fake_feed = _FakeMarketFeed()
    monkeypatch.setattr(state, "_market_feed", fake_feed)
    await auto_trade_loop._ensure_subscribed(["005930.KS", "000660.KS", "AAPL"])
    # AAPL has no KRX code — skipped, not an error.
    assert fake_feed.subscribed == ["005930", "000660"]
    # Candidates only need the circuit-breaker answer, not a live trade tick
    # (2026-08-28) — every pre-subscribe must ask for the 1TR cb_only tier,
    # never the full 2TR tier (that would re-exhaust the registration budget
    # every cycle, the exact incident this tier exists to fix).
    assert fake_feed.cb_only_calls == [True, True]


async def test_ensure_subscribed_noop_when_feed_not_running(monkeypatch):
    from app.web import state

    monkeypatch.setattr(state, "_market_feed", None)
    await auto_trade_loop._ensure_subscribed(["005930.KS"])  # must not raise


async def test_ensure_subscribed_registers_candidate_even_with_no_feed_yet(monkeypatch):
    """2026-09-04 (3rd recurrence — see state._candidate_cb_codes's own
    comment): note_candidate_cb_code must run BEFORE the `_market_feed is
    None` early return, so a candidate seen on a cycle with no feed object
    yet is still remembered for the next ensure_market_feed() build."""
    from app.web import state

    monkeypatch.setattr(state, "_market_feed", None)
    state._candidate_cb_codes.clear()

    await auto_trade_loop._ensure_subscribed(["005930.KS"])  # must not raise

    assert "005930" in state._candidate_cb_codes


async def test_ensure_subscribed_degrades_on_add_code_failure(monkeypatch):
    from app.web import state

    class _FailingFeed:
        async def add_code(self, code, *, orderbook: bool = False, cb_only: bool = False):
            raise RuntimeError("KIS WS send failed")

    monkeypatch.setattr(state, "_market_feed", _FailingFeed())
    await auto_trade_loop._ensure_subscribed(["005930.KS"])  # must not raise


async def test_ensure_subscribed_alerts_when_registration_budget_exhausted(monkeypatch):
    """2026-08-28 domestic AI audit: a plain logger.warning here (2026-08-05)
    went unnoticed for three weeks until it was traced to 6 of 11 same-day
    Buy decisions being blocked at execution — a cap-out here must be
    surfaced through alerts, not just the log."""
    from app.web import state

    async def _no_rebuild(reason):
        return False

    monkeypatch.setattr(state, "rebuild_market_feed_for_capacity", _no_rebuild)
    fake_feed = _FakeMarketFeed(accept=False)
    monkeypatch.setattr(state, "_market_feed", fake_feed)
    recorded = []

    async def fake_record_alert(alert):
        recorded.append(alert)

    monkeypatch.setattr(auto_trade_loop, "record_alert", fake_record_alert)

    await auto_trade_loop._ensure_subscribed(["005930.KS"])

    assert len(recorded) == 1
    assert recorded[0].ticker == "005930.KS"
    assert recorded[0].kind == "realtime_subscription_capped"


async def test_ensure_subscribed_keeps_candidate_registered_when_add_code_returns_false(monkeypatch):
    """2026-09-04: budget exhaustion must not evict the candidate from the
    registry — it may still win a cb-only slot on a future zone rebuild once
    something else ages out. The existing realtime_subscription_capped alert
    behavior (tested above) must be unchanged."""
    from app.web import state

    # filter_new_alerts' cooldown (app.live.alerts._last_fired_at) is
    # in-memory, module-global state with no per-test isolation (same pattern
    # test_unrequested_cancelled_error_alerts_on_exhaustion already guards
    # against above) — test_ensure_subscribed_alerts_when_registration_budget_exhausted
    # fires the exact same (005930.KS, "realtime_subscription_capped") key,
    # so without clearing it here this alert would be silently swallowed as
    # "already fired" whenever that test runs first in the same session.
    from app.live import alerts as alerts_module

    monkeypatch.setattr(alerts_module, "_last_fired_at", {})

    async def _no_rebuild(reason):
        return False

    monkeypatch.setattr(state, "rebuild_market_feed_for_capacity", _no_rebuild)
    fake_feed = _FakeMarketFeed(accept=False)
    monkeypatch.setattr(state, "_market_feed", fake_feed)
    state._candidate_cb_codes.clear()
    recorded = []

    async def fake_record_alert(alert):
        recorded.append(alert)

    monkeypatch.setattr(auto_trade_loop, "record_alert", fake_record_alert)

    await auto_trade_loop._ensure_subscribed(["005930.KS"])

    assert "005930" in state._candidate_cb_codes
    assert len(recorded) == 1
    assert recorded[0].kind == "realtime_subscription_capped"


class _RebuildableFeed(_FakeMarketFeed):
    """add_code rejects every code in `reject`."""

    def __init__(self, reject):
        super().__init__()
        self.reject = set(reject)

    async def add_code(self, code, *, orderbook: bool = False, cb_only: bool = False):
        self.subscribed.append(code)
        return code not in self.reject


async def _capacity_rebuild_setup(monkeypatch):
    from app.live import alerts as alerts_module
    from app.web import state

    monkeypatch.setattr(alerts_module, "_last_fired_at", {})
    # note_candidate_cb_code writes this module-global registry — isolate it so
    # candidates here don't leak into later _prioritized_krx_codes tests.
    monkeypatch.setattr(state, "_candidate_cb_codes", {})
    recorded = []

    async def fake_record_alert(alert):
        recorded.append(alert)

    monkeypatch.setattr(auto_trade_loop, "record_alert", fake_record_alert)
    return state, recorded


async def test_ensure_subscribed_rebuilds_once_and_no_alert_when_new_feed_fits(monkeypatch):
    state, recorded = await _capacity_rebuild_setup(monkeypatch)
    old_feed = _RebuildableFeed(reject={"005930", "000660"})
    new_feed = _RebuildableFeed(reject=set())
    monkeypatch.setattr(state, "_market_feed", old_feed)
    reasons = []

    async def fake_rebuild(reason):
        reasons.append(reason)
        state._market_feed = new_feed  # what stop+ensure does
        return True

    monkeypatch.setattr(state, "rebuild_market_feed_for_capacity", fake_rebuild)

    await auto_trade_loop._ensure_subscribed(["005930.KS", "000660.KS", "035420.KS"])

    assert len(reasons) == 1  # one rebuild per call, not one per capped code
    assert new_feed.subscribed == ["005930", "000660"]  # retried on the NEW feed
    assert recorded == []


async def test_ensure_subscribed_alerts_only_codes_still_missing_after_rebuild(monkeypatch):
    state, recorded = await _capacity_rebuild_setup(monkeypatch)
    old_feed = _RebuildableFeed(reject={"005930", "000660"})
    new_feed = _RebuildableFeed(reject={"000660"})
    monkeypatch.setattr(state, "_market_feed", old_feed)

    async def fake_rebuild(reason):
        state._market_feed = new_feed
        return True

    monkeypatch.setattr(state, "rebuild_market_feed_for_capacity", fake_rebuild)

    await auto_trade_loop._ensure_subscribed(["005930.KS", "000660.KS"])

    assert [a.ticker for a in recorded] == ["000660.KS"]
    assert recorded[0].kind == "realtime_subscription_capped"


async def test_ensure_subscribed_alerts_when_rebuild_skipped_by_interval(monkeypatch):
    state, recorded = await _capacity_rebuild_setup(monkeypatch)
    feed = _RebuildableFeed(reject={"005930"})
    monkeypatch.setattr(state, "_market_feed", feed)

    async def skipped(reason):
        return False

    monkeypatch.setattr(state, "rebuild_market_feed_for_capacity", skipped)

    await auto_trade_loop._ensure_subscribed(["005930.KS"])

    assert [a.ticker for a in recorded] == ["005930.KS"]


async def test_ensure_subscribed_no_rebuild_when_nothing_capped(monkeypatch):
    state, recorded = await _capacity_rebuild_setup(monkeypatch)
    monkeypatch.setattr(state, "_market_feed", _FakeMarketFeed(accept=True))
    reasons = []

    async def fake_rebuild(reason):
        reasons.append(reason)
        return True

    monkeypatch.setattr(state, "rebuild_market_feed_for_capacity", fake_rebuild)

    await auto_trade_loop._ensure_subscribed(["005930.KS"])

    assert reasons == [] and recorded == []


async def test_ensure_subscribed_no_rebuild_when_feed_none(monkeypatch):
    state, recorded = await _capacity_rebuild_setup(monkeypatch)
    monkeypatch.setattr(state, "_market_feed", None)
    reasons = []

    async def fake_rebuild(reason):
        reasons.append(reason)
        return True

    monkeypatch.setattr(state, "rebuild_market_feed_for_capacity", fake_rebuild)

    await auto_trade_loop._ensure_subscribed(["005930.KS"])

    assert reasons == [] and recorded == []


async def test_domestic_candidates_pre_subscribes_selected_candidates(monkeypatch):
    # Real incident (2026-07-30): a candidate's BUY got blocked at execution
    # because its subscription request only went out after the decision was
    # made, ~9 minutes after candidate selection — not enough lead time for
    # KIS to acknowledge it. Subscribing at selection time instead gives the
    # whole analysis pipeline's runtime as lead time.
    from app.web import state

    _set_trade_allowlist(monkeypatch, [])
    monkeypatch.setattr(settings, "auto_trade_max_tickers_domestic", 10)
    fake_feed = _FakeMarketFeed()
    monkeypatch.setattr(state, "_market_feed", fake_feed)

    async def fake_krx_top_n(broker, count, *, max_price=None):
        return ["005930.KS", "000660.KS"]

    async def fake_top_candidates(universe, top_n=5, lookback_days=10, **kwargs):
        return universe

    monkeypatch.setattr(auto_trade_loop, "krx_top_n", fake_krx_top_n)
    monkeypatch.setattr(auto_trade_loop, "screening_pass", _as_screening_pass(fake_top_candidates))
    await auto_trade_loop._domestic_candidates(broker=None)
    assert set(fake_feed.subscribed) == {"005930", "000660"}


async def test_domestic_candidates_uses_domestic_lookback_setting(monkeypatch):
    monkeypatch.setattr(settings, "momentum_lookback_days_domestic", 15)
    captured = {}

    async def fake_krx_top_n(broker, count, *, max_price=None):
        return []

    async def fake_top_candidates(universe, top_n=5, lookback_days=10, **kwargs):
        captured["lookback_days"] = lookback_days
        return []

    monkeypatch.setattr(auto_trade_loop, "krx_top_n", fake_krx_top_n)
    monkeypatch.setattr(auto_trade_loop, "screening_pass", _as_screening_pass(fake_top_candidates))
    await auto_trade_loop._domestic_candidates(broker=None)
    assert captured["lookback_days"] == 15


async def test_domestic_candidates_keeps_volatility_adjustment_and_dedup_on(monkeypatch):
    # screening-improvement-quality-review: domestic stayed direction-
    # unstable across 3 out-of-sample periods (no consistent signal either
    # way), so both stages stay on there — only overseas got turned off.
    captured = {}

    async def fake_krx_top_n(broker, count, *, max_price=None):
        return []

    async def fake_top_candidates(universe, top_n=5, lookback_days=10, **kwargs):
        captured["use_volatility_adjustment"] = kwargs["use_volatility_adjustment"]
        captured["use_correlation_dedup"] = kwargs["use_correlation_dedup"]
        return []

    monkeypatch.setattr(auto_trade_loop, "krx_top_n", fake_krx_top_n)
    monkeypatch.setattr(auto_trade_loop, "screening_pass", _as_screening_pass(fake_top_candidates))
    await auto_trade_loop._domestic_candidates(broker=None)
    assert captured["use_volatility_adjustment"] is True
    assert captured["use_correlation_dedup"] is True


async def test_domestic_candidates_returns_empty_when_krx_market_closed(monkeypatch):
    monkeypatch.setattr(auto_trade_loop, "is_domestic_market_open", lambda: False)

    async def should_not_be_called(*args, **kwargs):
        raise AssertionError("must not scan/rank when KRX is closed")

    monkeypatch.setattr(auto_trade_loop, "krx_top_n", should_not_be_called)
    monkeypatch.setattr(auto_trade_loop, "screening_pass", _as_screening_pass(should_not_be_called))
    assert await auto_trade_loop._domestic_candidates(broker=None) == []


async def test_overseas_candidates_pools_trade_allowlist_and_overseas_candidates(monkeypatch):
    _set_trade_allowlist(monkeypatch, ["AAPL", "005930.KS"])
    monkeypatch.setattr(settings, "auto_trade_max_tickers_overseas", 10)
    captured = {}

    async def fake_overseas_top_n(broker, count):
        return ["AAPL", "MSFT"]

    async def fake_top_candidates(universe, top_n=5, lookback_days=10, **kwargs):
        captured["universe"] = universe
        captured["top_n"] = top_n
        return universe

    monkeypatch.setattr(auto_trade_loop, "overseas_top_n", fake_overseas_top_n)
    monkeypatch.setattr(auto_trade_loop, "screening_pass", _as_screening_pass(fake_top_candidates))
    result = await auto_trade_loop._overseas_candidates(broker=None)
    assert captured["universe"] == ["AAPL", "MSFT"]  # deduped; 005930.KS never enters the overseas pool
    assert captured["top_n"] == 10
    assert result == ["AAPL", "MSFT"]


async def test_overseas_candidates_pools_market_cap_candidates_too(monkeypatch):
    # Overseas counterpart of test_domestic_candidates_pools_market_cap_
    # candidates_too — the broker method/TR existed since 2026-07-30 but
    # was never wired into a discovery feed here until now.
    _set_trade_allowlist(monkeypatch, ["005930.KS"])
    monkeypatch.setattr(settings, "auto_trade_max_tickers_overseas", 10)
    captured = {}

    async def fake_overseas_top_n(broker, count):
        return ["AAPL"]

    async def fake_overseas_market_cap_top_n(broker, count):
        return ["MSFT"]

    async def fake_top_candidates(universe, top_n=5, lookback_days=10, **kwargs):
        captured["universe"] = universe
        return universe

    monkeypatch.setattr(auto_trade_loop, "overseas_top_n", fake_overseas_top_n)
    monkeypatch.setattr(auto_trade_loop, "overseas_market_cap_top_n", fake_overseas_market_cap_top_n)
    monkeypatch.setattr(auto_trade_loop, "screening_pass", _as_screening_pass(fake_top_candidates))
    result = await auto_trade_loop._overseas_candidates(broker=None)
    assert captured["universe"] == ["AAPL", "MSFT"]
    assert result == ["AAPL", "MSFT"]


async def test_overseas_candidates_turns_off_volatility_adjustment_and_dedup(monkeypatch):
    # screening-improvement-quality-review: 3/3 independent out-of-sample
    # periods found both stages consistently hurt the overseas pool vs the
    # plain-momentum baseline — turned off here as a result (2026-07-29).
    captured = {}

    async def fake_overseas_top_n(broker, count):
        return []

    async def fake_top_candidates(universe, top_n=5, lookback_days=10, **kwargs):
        captured["use_volatility_adjustment"] = kwargs["use_volatility_adjustment"]
        captured["use_correlation_dedup"] = kwargs["use_correlation_dedup"]
        return []

    monkeypatch.setattr(auto_trade_loop, "overseas_top_n", fake_overseas_top_n)
    monkeypatch.setattr(auto_trade_loop, "screening_pass", _as_screening_pass(fake_top_candidates))
    await auto_trade_loop._overseas_candidates(broker=None)
    assert captured["use_volatility_adjustment"] is False
    assert captured["use_correlation_dedup"] is False


async def test_overseas_candidates_uses_overseas_lookback_setting(monkeypatch):
    monkeypatch.setattr(settings, "momentum_lookback_days_overseas", 20)
    captured = {}

    async def fake_overseas_top_n(broker, count):
        return []

    async def fake_top_candidates(universe, top_n=5, lookback_days=10, **kwargs):
        captured["lookback_days"] = lookback_days
        return []

    monkeypatch.setattr(auto_trade_loop, "overseas_top_n", fake_overseas_top_n)
    monkeypatch.setattr(auto_trade_loop, "screening_pass", _as_screening_pass(fake_top_candidates))
    await auto_trade_loop._overseas_candidates(broker=None)
    assert captured["lookback_days"] == 20


async def test_overseas_candidates_returns_empty_when_us_market_closed(monkeypatch):
    monkeypatch.setattr(auto_trade_loop, "is_overseas_market_open", lambda: False)

    async def should_not_be_called(*args, **kwargs):
        raise AssertionError("must not scan/rank when the US market is closed")

    monkeypatch.setattr(auto_trade_loop, "overseas_top_n", should_not_be_called)
    monkeypatch.setattr(auto_trade_loop, "screening_pass", _as_screening_pass(should_not_be_called))
    assert await auto_trade_loop._overseas_candidates(broker=None) == []


async def test_overseas_candidates_empty_when_disabled_even_if_market_open(monkeypatch):
    """Hard scope switch (app.config.overseas_trading_enabled) — must return
    [] before even checking market hours, and never reach the scan/rank
    calls, regardless of whether the US market happens to be open."""
    monkeypatch.setattr(settings, "overseas_trading_enabled", False)
    monkeypatch.setattr(auto_trade_loop, "is_overseas_market_open", lambda: True)

    async def should_not_be_called(*args, **kwargs):
        raise AssertionError("must not scan/rank when overseas_trading_enabled=False")

    monkeypatch.setattr(auto_trade_loop, "overseas_top_n", should_not_be_called)
    monkeypatch.setattr(auto_trade_loop, "overseas_market_cap_top_n", should_not_be_called)
    monkeypatch.setattr(auto_trade_loop, "screening_pass", _as_screening_pass(should_not_be_called))
    assert await auto_trade_loop._overseas_candidates(broker=None) == []


# ---------------------------------------------------------------------------
# auto_trade_loop's while-True cycle
# ---------------------------------------------------------------------------


class _StopLoop(Exception):
    pass


async def _run_one_cycle(monkeypatch):
    """Runs exactly one iteration of auto_trade_loop's while-True body by
    making asyncio.sleep raise, breaking the loop right after one pass."""

    async def fake_sleep(seconds):
        raise _StopLoop()

    monkeypatch.setattr(auto_trade_loop.asyncio, "sleep", fake_sleep)
    try:
        await auto_trade_loop.auto_trade_loop()
    except _StopLoop:
        pass


@pytest.fixture(autouse=True)
def _identity_candidates(monkeypatch):
    """Cycle-level tests below aren't testing the per-market selection
    policy itself (see the dedicated tests above) — stub both discovery
    feeds empty and top_candidates to a plain pass-through, so they don't
    need real price/broker data.

    Also stubs tickers_analyzed_on (real Postgres, unmocked, otherwise) —
    every SEMI_AUTO/FULL_AUTO path through run_cycle_now passes
    exclude_already_analyzed_today=True, which calls it for real. This is
    a shared dev DB actually used for real testing sessions, not an
    isolated test fixture — a ticker genuinely analyzed today (e.g.
    005930.KS during a live FULL_AUTO test run earlier the same day) would
    silently get excluded here too, making a cycle-level test's outcome
    depend on whatever real analysis happened to run earlier that day.
    Confirmed via reproduction: this exact flakiness pre-dates and is
    unrelated to any single change — see tickers_analyzed_on's own
    real-DB read returning today's actual analyzed set when this fixture
    didn't yet stub it.

    2026-08-10: also stubs the four real I/O calls run_cycle_now reaches
    BEFORE any of the above (state.kis_broker, market_regime's two
    refreshes, resolve_all_due, reconcile_trade_allowlist) — none of them
    were stubbed anywhere in this file except by a handful of individual
    tests for their own narrow purpose. Unstubbed, each is a real KIS/
    Postgres call; run_cycle_now's own try/except around the market_regime/
    resolve_all_due calls only guards against an exception, not a hang, and
    get_broker's single shared (name, user, mode) cache/lock means whichever
    test reaches an unstubbed state.kis_broker() FIRST in a given pytest
    process blocks every test after it in the same run, forever, if that
    call never returns (confirmed live in this sandbox — see project notes).
    A test below that needs a specific broker/reconciled-ticker value still
    overrides these with its own monkeypatch.setattr AFTER this fixture
    runs, same as it already does for krx_top_n/overseas_top_n above.

    2026-09-08 (§4-1): also disables screening_snapshot_log_enabled — both
    app.live.screening_log.record_screening_snapshot and
    app.audit.screening_ic.resolve_due_forward_returns check this flag
    before touching the pool at all, so this is enough to keep the new live
    instrumentation from making a real Postgres connection attempt in every
    cycle-level test here (would otherwise re-create the exact 'first
    unstubbed real-I/O call in the process hangs every test after it' trap
    this fixture's whole 2026-08-10 stubbing already exists to avoid)."""
    from app.web import state

    monkeypatch.setattr(settings, "screening_snapshot_log_enabled", False)

    async def fake_krx_top_n(broker, count, *, max_price=None):
        return []

    async def fake_overseas_top_n(broker, count):
        return []

    async def fake_top_candidates(universe, top_n=5, lookback_days=10, **kwargs):
        return universe[:top_n]

    async def fake_tickers_analyzed_on(trade_date):
        return set()

    async def fake_regime_refresh(*args, **kwargs):
        return None

    async def fake_resolve_all_due(*args, **kwargs):
        return None

    async def fake_reconcile_trade_allowlist(*args, **kwargs):
        return []

    monkeypatch.setattr(auto_trade_loop, "krx_top_n", fake_krx_top_n)
    monkeypatch.setattr(auto_trade_loop, "overseas_top_n", fake_overseas_top_n)
    monkeypatch.setattr(auto_trade_loop, "screening_pass", _as_screening_pass(fake_top_candidates))
    monkeypatch.setattr(auto_trade_loop, "tickers_analyzed_on", fake_tickers_analyzed_on)
    monkeypatch.setattr(state, "kis_broker", broker_stub(None))
    monkeypatch.setattr("app.execution.market_regime.refresh_domestic_regime", fake_regime_refresh)
    monkeypatch.setattr("app.execution.market_regime.refresh_overseas_regime", fake_regime_refresh)
    monkeypatch.setattr("app.memory.auto_resolve.resolve_all_due", fake_resolve_all_due)
    monkeypatch.setattr(state, "reconcile_trade_allowlist", fake_reconcile_trade_allowlist)


async def test_manual_mode_skips_analysis_entirely(monkeypatch):
    monkeypatch.setattr(settings, "execution_mode", ExecutionMode.MANUAL)

    async def should_not_be_called(*args, **kwargs):
        raise AssertionError("MANUAL mode must not call run_pipeline")

    monkeypatch.setattr(auto_trade_loop, "run_pipeline", should_not_be_called)
    monkeypatch.setattr(auto_trade_loop, "krx_top_n", should_not_be_called)
    monkeypatch.setattr(auto_trade_loop, "overseas_top_n", should_not_be_called)

    await _run_one_cycle(monkeypatch)


async def test_semi_auto_mode_runs_pipeline_for_both_sides(monkeypatch):
    monkeypatch.setattr(settings, "execution_mode", ExecutionMode.SEMI_AUTO)
    _set_trade_allowlist(monkeypatch, ["AAPL", "005930.KS"])
    calls = []

    async def fake_run_pipeline(ticker, trade_date, force_refresh=False, on_stage=None):
        calls.append(ticker)

    monkeypatch.setattr(auto_trade_loop, "run_pipeline", fake_run_pipeline)

    await _run_one_cycle(monkeypatch)
    assert set(calls) == {"AAPL", "005930.KS"}  # domestic + overseas both ran


async def test_cycle_force_includes_newly_reconciled_tickers_in_analysis(monkeypatch):
    # 2026-07-29: a real KIS holding discovered by reconcile_trade_allowlist
    # this cycle (bought outside this system) must get a full pipeline run
    # right away — it won't reliably surface via momentum ranking (it's not
    # a fresh candidate) or the ledger-based held-review (it was never
    # traded through this system's ledger), so without this force-include
    # it could go unanalyzed (and thus stop-loss-less) indefinitely.
    monkeypatch.setattr(settings, "execution_mode", ExecutionMode.SEMI_AUTO)
    _set_trade_allowlist(monkeypatch, [])
    calls = []

    async def fake_run_pipeline(ticker, trade_date, force_refresh=False, on_stage=None):
        calls.append(ticker)

    async def fake_reconcile():
        return ["005930.KS"]

    monkeypatch.setattr(auto_trade_loop, "run_pipeline", fake_run_pipeline)
    monkeypatch.setattr("app.web.state.reconcile_trade_allowlist", fake_reconcile)

    await _run_one_cycle(monkeypatch)

    assert "005930.KS" in calls


async def test_cycle_does_not_double_analyze_a_reconciled_ticker_already_a_candidate(monkeypatch):
    monkeypatch.setattr(settings, "execution_mode", ExecutionMode.SEMI_AUTO)
    _set_trade_allowlist(monkeypatch, ["005930.KS"])
    calls = []

    async def fake_run_pipeline(ticker, trade_date, force_refresh=False, on_stage=None):
        calls.append(ticker)

    async def fake_reconcile():
        return ["005930.KS"]  # same ticker the normal candidate pool already includes

    monkeypatch.setattr(auto_trade_loop, "run_pipeline", fake_run_pipeline)
    monkeypatch.setattr("app.web.state.reconcile_trade_allowlist", fake_reconcile)

    await _run_one_cycle(monkeypatch)

    assert calls.count("005930.KS") == 1


async def test_semi_auto_second_cycle_does_not_rerun(monkeypatch):
    # 2026-07-21: SEMI_AUTO auto-runs once, then only the manual "다시 분석"
    # trigger (not the periodic timer) starts another cycle.
    monkeypatch.setattr(settings, "execution_mode", ExecutionMode.SEMI_AUTO)
    calls = []

    async def fake_run_pipeline(ticker, trade_date, force_refresh=False, on_stage=None):
        calls.append(ticker)

    monkeypatch.setattr(auto_trade_loop, "run_pipeline", fake_run_pipeline)

    await _run_one_cycle(monkeypatch)
    assert calls == ["AAPL"]
    assert auto_trade_loop._status_scope() in auto_trade_loop._semi_auto_done
    assert auto_trade_loop.get_status().next_cycle_at is None

    await _run_one_cycle(monkeypatch)  # second pass through the while-loop body
    assert calls == ["AAPL"]  # unchanged — no second run
    assert auto_trade_loop.get_status().phase == "waiting_for_manual"


async def test_semi_auto_periodic_pass_does_not_clobber_an_in_flight_manual_cycle(monkeypatch):
    # The concurrency bug this guards against: the periodic SEMI_AUTO branch
    # used to unconditionally force phase="waiting_for_manual" on every
    # interval — including while a manual "다시 분석" cycle (a separate
    # asyncio.create_task from app.web.routers.live, not this loop) was
    # still mid-flight. That cleared check_manual_trigger_allowed's "a cycle
    # is already running" guard, letting a second click start a concurrent
    # cycle over the same tickers.
    monkeypatch.setattr(settings, "execution_mode", ExecutionMode.SEMI_AUTO)
    auto_trade_loop._semi_auto_done.add(auto_trade_loop._status_scope())  # already ran its one auto-cycle

    async def should_not_be_called(*args, **kwargs):
        raise AssertionError("SEMI_AUTO's periodic pass must not start a second cycle")

    monkeypatch.setattr(auto_trade_loop, "run_pipeline", should_not_be_called)
    auto_trade_loop.get_status().phase = "analyzing"  # a manual cycle is in flight

    await _run_one_cycle(monkeypatch)
    assert auto_trade_loop.get_status().phase == "analyzing"  # left alone, not overwritten


async def test_full_auto_mode_runs_pipeline_for_candidates(monkeypatch):
    monkeypatch.setattr(settings, "execution_mode", ExecutionMode.FULL_AUTO)
    calls = []

    async def fake_run_pipeline(ticker, trade_date, force_refresh=False, on_stage=None):
        calls.append(ticker)

    monkeypatch.setattr(auto_trade_loop, "run_pipeline", fake_run_pipeline)

    await _run_one_cycle(monkeypatch)
    assert calls == ["AAPL"]


async def test_market_closed_side_is_skipped_but_the_other_still_runs(monkeypatch):
    monkeypatch.setattr(settings, "execution_mode", ExecutionMode.FULL_AUTO)
    _set_trade_allowlist(monkeypatch, ["AAPL", "005930.KS"])
    monkeypatch.setattr(auto_trade_loop, "is_domestic_market_open", lambda: False)
    calls = []

    async def fake_run_pipeline(ticker, trade_date, force_refresh=False, on_stage=None):
        calls.append(ticker)

    monkeypatch.setattr(auto_trade_loop, "run_pipeline", fake_run_pipeline)

    await _run_one_cycle(monkeypatch)
    assert calls == ["AAPL"]  # 005930.KS skipped — KRX closed


async def test_one_ticker_failure_does_not_stop_the_rest(monkeypatch):
    monkeypatch.setattr(settings, "execution_mode", ExecutionMode.FULL_AUTO)
    _set_trade_allowlist(monkeypatch, ["AAPL", "MSFT"])
    calls = []

    async def fake_run_pipeline(ticker, trade_date, force_refresh=False, on_stage=None):
        calls.append(ticker)
        if ticker == "AAPL":
            raise RuntimeError("LLM backend timeout")

    monkeypatch.setattr(auto_trade_loop, "run_pipeline", fake_run_pipeline)

    await _run_one_cycle(monkeypatch)
    # Order-independent: candidates now analyze with bounded concurrency
    # (asyncio.gather), not strictly one at a time, so which one's call
    # lands first isn't guaranteed the way a plain sequential for-loop was.
    assert set(calls) == {"AAPL", "MSFT"}  # MSFT still ran despite AAPL's failure


async def test_pipeline_failure_exhausting_retries_records_a_system_alert(monkeypatch):
    # ai-engine-24h-operation-analysis P0-2: exhausting every retry on a
    # ticker used to be a log line and nothing else — this is the fix.
    monkeypatch.setattr(settings, "execution_mode", ExecutionMode.FULL_AUTO)

    async def fake_run_pipeline(ticker, trade_date, force_refresh=False, on_stage=None):
        raise RuntimeError("LLM backend timeout")

    monkeypatch.setattr(auto_trade_loop, "run_pipeline", fake_run_pipeline)

    recorded = []

    async def fake_record_alert(alert):
        recorded.append(alert)

    monkeypatch.setattr(auto_trade_loop, "record_alert", fake_record_alert)

    with user_scope(settings.user_id), account_mode_scope(settings.kis_account_mode):
        auto_trade_loop._retry_failures_for_scope()["AAPL"] = auto_trade_loop._MAX_CONSECUTIVE_RETRY_ATTEMPTS - 1

    await _run_one_cycle(monkeypatch)

    assert len(recorded) == 1
    assert recorded[0].kind == "pipeline_failure"
    assert recorded[0].ticker == "AAPL"


async def test_pipeline_failure_before_retries_exhausted_records_no_alert(monkeypatch):
    # Not every failure is alert-worthy — only giving up entirely is. A
    # ticker that will be force-retried next cycle stays a log line only.
    monkeypatch.setattr(settings, "execution_mode", ExecutionMode.FULL_AUTO)

    async def fake_run_pipeline(ticker, trade_date, force_refresh=False, on_stage=None):
        raise RuntimeError("LLM backend timeout")

    monkeypatch.setattr(auto_trade_loop, "run_pipeline", fake_run_pipeline)

    recorded = []

    async def fake_record_alert(alert):
        recorded.append(alert)

    monkeypatch.setattr(auto_trade_loop, "record_alert", fake_record_alert)

    await _run_one_cycle(monkeypatch)  # first-ever failure for AAPL — attempts=1, well under the max

    assert recorded == []


async def test_sibling_ticker_survives_an_unrequested_cancelled_error(monkeypatch):
    # id=377/id=949 shape, but at the gather level: a CancelledError that
    # merely surfaced through one ticker's task (no .cancel() call anywhere
    # here, so Task.cancelling() is genuinely 0 — not a real cancellation)
    # must not collaterally cancel every OTHER ticker analyzing concurrently
    # in the same cycle. Before this fix, asyncio.gather(return_exceptions=
    # False)'s default teardown would have cancelled MSFT's in-flight
    # analysis too, purely because AAPL's task raised a BaseException the
    # old `except Exception:` never saw.
    monkeypatch.setattr(settings, "execution_mode", ExecutionMode.FULL_AUTO)
    _set_trade_allowlist(monkeypatch, ["AAPL", "MSFT"])
    calls = []

    async def fake_run_pipeline(ticker, trade_date, force_refresh=False, on_stage=None):
        calls.append(ticker)
        if ticker == "AAPL":
            raise asyncio.CancelledError()

    monkeypatch.setattr(auto_trade_loop, "run_pipeline", fake_run_pipeline)

    await _run_one_cycle(monkeypatch)

    assert set(calls) == {"AAPL", "MSFT"}  # MSFT still ran despite AAPL's CancelledError


async def test_unrequested_cancelled_error_feeds_the_retry_path(monkeypatch):
    monkeypatch.setattr(settings, "execution_mode", ExecutionMode.FULL_AUTO)

    async def fake_run_pipeline(ticker, trade_date, force_refresh=False, on_stage=None):
        raise asyncio.CancelledError()

    monkeypatch.setattr(auto_trade_loop, "run_pipeline", fake_run_pipeline)

    await _run_one_cycle(monkeypatch)

    with user_scope(settings.user_id), account_mode_scope(settings.kis_account_mode):
        assert auto_trade_loop._retry_failures_for_scope()["AAPL"] == 1


async def test_unrequested_cancelled_error_alerts_on_exhaustion(monkeypatch):
    monkeypatch.setattr(settings, "execution_mode", ExecutionMode.FULL_AUTO)
    # filter_new_alerts' cooldown (app.live.alerts._last_fired_at) is
    # in-memory, module-global state with no per-test isolation (same
    # pattern test_ledger.py already guards against) — this file's own
    # test_pipeline_failure_exhausting_retries_records_a_system_alert fires
    # the exact same (AAPL, "pipeline_failure") key, and this suite's fixed
    # clock never advances, so without clearing it here this alert would be
    # silently swallowed as "already fired" whenever that test runs first.
    from app.live import alerts as alerts_module

    monkeypatch.setattr(alerts_module, "_last_fired_at", {})

    async def fake_run_pipeline(ticker, trade_date, force_refresh=False, on_stage=None):
        raise asyncio.CancelledError()

    monkeypatch.setattr(auto_trade_loop, "run_pipeline", fake_run_pipeline)

    recorded = []

    async def fake_record_alert(alert):
        recorded.append(alert)

    monkeypatch.setattr(auto_trade_loop, "record_alert", fake_record_alert)

    with user_scope(settings.user_id), account_mode_scope(settings.kis_account_mode):
        auto_trade_loop._retry_failures_for_scope()["AAPL"] = auto_trade_loop._MAX_CONSECUTIVE_RETRY_ATTEMPTS - 1

    await _run_one_cycle(monkeypatch)

    assert len(recorded) == 1
    assert recorded[0].kind == "pipeline_failure"
    assert recorded[0].ticker == "AAPL"
    with user_scope(settings.user_id), account_mode_scope(settings.kis_account_mode):
        assert "AAPL" not in auto_trade_loop._retry_failures_for_scope()  # cleared, not left dangling


async def test_real_task_cancellation_propagates_and_records_nothing(monkeypatch):
    # A genuine outer cancellation (e.g. app shutdown cancelling the whole
    # auto_trade_loop task) must still tear the cycle down and must NOT be
    # misclassified as a per-ticker failure just because it arrives as the
    # same CancelledError type as the unrequested case above. Cancels the
    # real task running run_cycle_now from outside (an actual .cancel()
    # call, unlike the tests above) so Task.cancelling() genuinely reports
    # a real cancellation for AAPL's in-flight _analyze_one child task too.
    monkeypatch.setattr(settings, "execution_mode", ExecutionMode.FULL_AUTO)
    started = asyncio.Event()

    async def fake_run_pipeline(ticker, trade_date, force_refresh=False, on_stage=None):
        started.set()
        await asyncio.Event().wait()  # blocks until this task is cancelled from outside

    monkeypatch.setattr(auto_trade_loop, "run_pipeline", fake_run_pipeline)

    recorded = []

    async def fake_record_alert(alert):
        recorded.append(alert)

    monkeypatch.setattr(auto_trade_loop, "record_alert", fake_record_alert)

    task = asyncio.create_task(auto_trade_loop.run_cycle_now())
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert recorded == []
    with user_scope(settings.user_id), account_mode_scope(settings.kis_account_mode):
        assert "AAPL" not in auto_trade_loop._retry_failures_for_scope()


async def test_cycle_exception_records_a_system_alert(monkeypatch):
    # Same P0-2 fix, for a whole-cycle failure (e.g. the allowlist store's
    # DB call) rather than a single ticker's pipeline.
    async def failing_ensure_loaded(user_id, account_mode):
        raise RuntimeError("DB unreachable")

    monkeypatch.setattr(trade_allowlist_store, "ensure_loaded", failing_ensure_loaded)

    recorded = []

    async def fake_record_alert(alert):
        recorded.append(alert)

    monkeypatch.setattr(auto_trade_loop, "record_alert", fake_record_alert)

    await _run_one_cycle(monkeypatch)

    assert len(recorded) == 1
    assert recorded[0].kind == "cycle_failure"
    assert recorded[0].ticker == "SYSTEM"
    assert "DB unreachable" in recorded[0].message


async def test_uses_current_broker_from_state(monkeypatch):
    monkeypatch.setattr(settings, "execution_mode", ExecutionMode.FULL_AUTO)
    captured = {}

    async def fake_krx_top_n(broker, count, *, max_price=None):
        captured["broker"] = broker
        return []

    async def fake_run_pipeline(ticker, trade_date, force_refresh=False, on_stage=None):
        return None

    from app.web import state

    monkeypatch.setattr(state, "kis_broker", broker_stub("fake-broker-instance"))
    monkeypatch.setattr(auto_trade_loop, "krx_top_n", fake_krx_top_n)
    monkeypatch.setattr(auto_trade_loop, "run_pipeline", fake_run_pipeline)

    await _run_one_cycle(monkeypatch)
    assert captured["broker"] == "fake-broker-instance"


# ---------------------------------------------------------------------------
# check_manual_trigger_allowed / run_cycle_now (the "다시 분석" button path)
# ---------------------------------------------------------------------------


def test_manual_trigger_rejected_outside_semi_auto(monkeypatch):
    monkeypatch.setattr(settings, "execution_mode", ExecutionMode.FULL_AUTO)
    with pytest.raises(auto_trade_loop.CooldownActive):
        auto_trade_loop.check_manual_trigger_allowed()


def test_manual_trigger_rejected_while_cycle_running(monkeypatch):
    monkeypatch.setattr(settings, "execution_mode", ExecutionMode.SEMI_AUTO)
    auto_trade_loop.get_status().phase = "analyzing"
    with pytest.raises(auto_trade_loop.CooldownActive):
        auto_trade_loop.check_manual_trigger_allowed()


def test_manual_trigger_rejected_within_cooldown(monkeypatch):
    from app.clock import get_clock

    monkeypatch.setattr(settings, "execution_mode", ExecutionMode.SEMI_AUTO)
    auto_trade_loop.get_status().cycle_finished_at = get_clock().now()
    with pytest.raises(auto_trade_loop.CooldownActive):
        auto_trade_loop.check_manual_trigger_allowed()


def test_manual_trigger_allowed_after_cooldown_elapses(monkeypatch):
    from datetime import timedelta

    from app.clock import get_clock

    monkeypatch.setattr(settings, "execution_mode", ExecutionMode.SEMI_AUTO)
    auto_trade_loop.get_status().cycle_finished_at = (
        get_clock().now() - timedelta(seconds=auto_trade_loop.MANUAL_TRIGGER_COOLDOWN_SECONDS + 1)
    )
    auto_trade_loop.check_manual_trigger_allowed()  # does not raise


def test_manual_trigger_allowed_when_never_run_before(monkeypatch):
    monkeypatch.setattr(settings, "execution_mode", ExecutionMode.SEMI_AUTO)
    assert auto_trade_loop.get_status().cycle_finished_at is None
    auto_trade_loop.check_manual_trigger_allowed()  # does not raise


def test_manual_trigger_reserves_the_slot_synchronously(monkeypatch):
    # The TOCTOU bug this guards against: run_cycle_now doesn't set
    # phase="scanning" until after several awaited calls (state.kis_broker(),
    # market_regime refreshes, resolve_all_due(), tickers_analyzed_on() x2) —
    # a multi-second window in which a second click's check still saw the
    # OLD phase and passed too, starting a concurrent second cycle. This
    # check must claim the slot itself, synchronously, so a back-to-back
    # second call is rejected immediately without waiting on any of that.
    monkeypatch.setattr(settings, "execution_mode", ExecutionMode.SEMI_AUTO)
    auto_trade_loop.check_manual_trigger_allowed()  # first click: allowed
    assert auto_trade_loop.get_status().phase == "scanning"
    with pytest.raises(auto_trade_loop.CooldownActive):
        auto_trade_loop.check_manual_trigger_allowed()  # second click, same instant: rejected


async def test_run_cycle_now_passes_exclude_set_when_requested(monkeypatch):
    monkeypatch.setattr(settings, "execution_mode", ExecutionMode.SEMI_AUTO)
    captured = {}

    async def fake_tickers_analyzed_on(trade_date):
        return {"AAPL"}

    async def fake_domestic(broker, exclude=frozenset()):
        captured["domestic_exclude"] = exclude
        return []

    async def fake_overseas(broker, exclude=frozenset()):
        captured["overseas_exclude"] = exclude
        return []

    monkeypatch.setattr(auto_trade_loop, "tickers_analyzed_on", fake_tickers_analyzed_on)
    monkeypatch.setattr(auto_trade_loop, "_domestic_candidates", fake_domestic)
    monkeypatch.setattr(auto_trade_loop, "_overseas_candidates", fake_overseas)

    await auto_trade_loop.run_cycle_now(exclude_already_analyzed_today=True)
    assert captured["domestic_exclude"] == frozenset({"AAPL"})
    assert captured["overseas_exclude"] == frozenset({"AAPL"})


async def test_run_cycle_now_excludes_nothing_by_default(monkeypatch):
    captured = {}

    async def fake_domestic(broker, exclude=frozenset()):
        captured["domestic_exclude"] = exclude
        return []

    async def fake_overseas(broker, exclude=frozenset()):
        return []

    monkeypatch.setattr(auto_trade_loop, "_domestic_candidates", fake_domestic)
    monkeypatch.setattr(auto_trade_loop, "_overseas_candidates", fake_overseas)

    await auto_trade_loop.run_cycle_now()
    assert captured["domestic_exclude"] == frozenset()


async def test_run_cycle_now_resets_phase_on_candidate_building_failure(monkeypatch):
    # 2026-08-10 fix: an exception anywhere between phase="scanning" and the
    # analysis gather (candidate building, reconciliation, etc. — none of
    # which are individually guarded the way each ticker's own pipeline call
    # is) used to leave _status.phase stuck at "scanning"/"analyzing"
    # forever, since nothing reset it on the way out. In SEMI_AUTO that
    # permanently wedges check_manual_trigger_allowed's "a cycle is already
    # running" guard, blocking every future "다시 분석" click.
    # (state.kis_broker/market_regime/resolve_all_due/reconcile_trade_
    # allowlist are all stubbed by the module's autouse _identity_candidates
    # fixture — see its own docstring for why this test used to hang
    # without them.)
    async def failing_domestic(broker, exclude=frozenset()):
        raise RuntimeError("KIS balance inquiry failed")

    monkeypatch.setattr(auto_trade_loop, "_domestic_candidates", failing_domestic)

    with pytest.raises(RuntimeError):
        await auto_trade_loop.run_cycle_now()

    status = auto_trade_loop.get_status()
    assert status.phase == "idle"
    assert status.current_tickers == set()
    assert status.cycle_finished_at is not None


async def test_run_cycle_now_resets_phase_when_the_prologue_itself_fails(monkeypatch):
    # Second pass of the same fix (2026-08-10): the first one only wrapped
    # candidate building onward, leaving the prologue — state.kis_broker(),
    # which does real I/O and can raise — OUTSIDE the guard. Only the manual
    # trigger reaches that gap, and it is the one path that can't tolerate
    # it: check_manual_trigger_allowed claims phase="scanning" itself,
    # synchronously, so a broker failure here left the slot claimed with no
    # cycle running and nothing left to release it — every later click then
    # failed the "already running" guard permanently, until a restart.
    from app.web import state as web_state

    async def failing_broker():
        raise RuntimeError("KIS auth failed")

    monkeypatch.setattr(settings, "execution_mode", ExecutionMode.SEMI_AUTO)
    monkeypatch.setattr(web_state, "kis_broker", failing_broker)

    auto_trade_loop.check_manual_trigger_allowed()  # claims the slot, as the endpoint does
    assert auto_trade_loop.get_status().phase == "scanning"

    with pytest.raises(RuntimeError):
        await auto_trade_loop.run_cycle_now()

    assert auto_trade_loop.get_status().phase == "idle"

    # The slot is genuinely released: the next click is refused by the
    # ordinary post-cycle cooldown, NOT by the "a cycle is already running"
    # wedge — that distinction is the whole point of this test.
    with pytest.raises(auto_trade_loop.CooldownActive) as exc_info:
        auto_trade_loop.check_manual_trigger_allowed()
    assert "cooldown active" in str(exc_info.value)


async def test_run_cycle_now_skips_ticker_whose_market_closed_while_queued(monkeypatch):
    # 2026-07-30 real incident: a cycle started 1 minute before KRX close
    # was still mid-analysis ~19 minutes after close (bounded concurrency +
    # a ~9-min-per-ticker LLM pipeline), and the resulting BUY only got
    # rejected at the KIS order step ("모의투자 장종료입니다") after the full
    # pipeline had already run for nothing. _analyze_one now re-checks the
    # market right before spending that cost.
    async def fake_domestic(broker, exclude=frozenset()):
        return ["005930.KS"]

    async def fake_overseas(broker, exclude=frozenset()):
        return []

    calls = []

    async def fake_run_pipeline(ticker, trade_date, force_refresh=False, on_stage=None):
        calls.append(ticker)

    monkeypatch.setattr(auto_trade_loop, "_domestic_candidates", fake_domestic)
    monkeypatch.setattr(auto_trade_loop, "_overseas_candidates", fake_overseas)
    monkeypatch.setattr(auto_trade_loop, "run_pipeline", fake_run_pipeline)
    # Candidate selection itself already saw the market open (that's how
    # 005930.KS got selected above) — this monkeypatch models the market
    # closing in the time between candidate selection and this ticker's
    # turn under the semaphore.
    monkeypatch.setattr(auto_trade_loop, "is_domestic_market_open", lambda: False)

    await auto_trade_loop.run_cycle_now()
    assert calls == []


async def test_run_cycle_now_still_analyzes_overseas_ticker_when_only_domestic_market_closed(monkeypatch):
    async def fake_domestic(broker, exclude=frozenset()):
        return []

    async def fake_overseas(broker, exclude=frozenset()):
        return ["AAPL"]

    calls = []

    async def fake_run_pipeline(ticker, trade_date, force_refresh=False, on_stage=None):
        calls.append(ticker)

    monkeypatch.setattr(auto_trade_loop, "_domestic_candidates", fake_domestic)
    monkeypatch.setattr(auto_trade_loop, "_overseas_candidates", fake_overseas)
    monkeypatch.setattr(auto_trade_loop, "run_pipeline", fake_run_pipeline)
    monkeypatch.setattr(auto_trade_loop, "is_domestic_market_open", lambda: False)

    await auto_trade_loop.run_cycle_now()
    assert calls == ["AAPL"]


async def test_run_cycle_now_threads_force_refresh_into_run_pipeline(monkeypatch):
    # AI-engine efficiency audit (2026-07-27): the manual "다시 분석" trigger's
    # force_refresh must reach run_pipeline for every candidate ticker, not
    # just be accepted and dropped.
    captured = {}

    async def fake_domestic(broker, exclude=frozenset()):
        return ["005930.KS"]

    async def fake_overseas(broker, exclude=frozenset()):
        return []

    async def fake_run_pipeline(ticker, trade_date, force_refresh=False, on_stage=None):
        captured["force_refresh"] = force_refresh

    monkeypatch.setattr(auto_trade_loop, "_domestic_candidates", fake_domestic)
    monkeypatch.setattr(auto_trade_loop, "_overseas_candidates", fake_overseas)
    monkeypatch.setattr(auto_trade_loop, "run_pipeline", fake_run_pipeline)

    await auto_trade_loop.run_cycle_now(force_refresh=True)
    assert captured["force_refresh"] is True


# ---------------------------------------------------------------------------
# _retry_failures_for_scope — per-(user, account_mode), not process-global.
# Was a single module-level dict until this fix: one user's failed ticker
# got force-appended into every OTHER user's very next cycle (bypassing
# their own allowlist/momentum ranking), and that user's own successful
# retry then silently cancelled the original user's retry guarantee.
# ---------------------------------------------------------------------------


def test_retry_failures_are_isolated_per_scope():
    with user_scope("alice"), account_mode_scope("paper"):
        auto_trade_loop._retry_failures_for_scope()["NVDA"] = 1

    with user_scope("bob"), account_mode_scope("paper"):
        assert "NVDA" not in auto_trade_loop._retry_failures_for_scope()

    with user_scope("alice"), account_mode_scope("paper"):
        assert auto_trade_loop._retry_failures_for_scope()["NVDA"] == 1


def test_reset_status_scopes_clears_retry_failures_too():
    with user_scope("alice"), account_mode_scope("paper"):
        auto_trade_loop._retry_failures_for_scope()["NVDA"] = 1
    auto_trade_loop.reset_status_scopes()
    with user_scope("alice"), account_mode_scope("paper"):
        assert auto_trade_loop._retry_failures_for_scope() == {}


# ---------------------------------------------------------------------------
# Bounded concurrency (settings.auto_trade_max_concurrent_analyses) — see
# this module's own docstring for why overlapping analysis is now safe
# (portfolio-cap enforcement moved to a per-market lock at commit time).
# ---------------------------------------------------------------------------


async def test_run_cycle_now_honors_the_concurrency_cap(monkeypatch):
    monkeypatch.setattr(settings, "auto_trade_max_concurrent_analyses", 2)

    async def fake_domestic(broker, exclude=frozenset()):
        return ["005930.KS", "000660.KS", "035420.KS"]

    async def fake_overseas(broker, exclude=frozenset()):
        return []

    monkeypatch.setattr(auto_trade_loop, "_domestic_candidates", fake_domestic)
    monkeypatch.setattr(auto_trade_loop, "_overseas_candidates", fake_overseas)

    in_flight = 0
    max_in_flight = 0

    async def fake_run_pipeline(ticker, trade_date, force_refresh=False, on_stage=None):
        nonlocal in_flight, max_in_flight
        in_flight += 1
        max_in_flight = max(max_in_flight, in_flight)
        await asyncio.sleep(0)  # yield point — lets the other candidates actually overlap
        in_flight -= 1

    monkeypatch.setattr(auto_trade_loop, "run_pipeline", fake_run_pipeline)

    await auto_trade_loop.run_cycle_now()

    assert max_in_flight == 2  # never more than the cap, even with 3 candidates queued


async def test_run_cycle_now_allows_full_concurrency_when_cap_exceeds_candidate_count(monkeypatch):
    monkeypatch.setattr(settings, "auto_trade_max_concurrent_analyses", 5)

    async def fake_domestic(broker, exclude=frozenset()):
        return ["005930.KS", "000660.KS"]

    async def fake_overseas(broker, exclude=frozenset()):
        return []

    monkeypatch.setattr(auto_trade_loop, "_domestic_candidates", fake_domestic)
    monkeypatch.setattr(auto_trade_loop, "_overseas_candidates", fake_overseas)

    in_flight = 0
    max_in_flight = 0

    async def fake_run_pipeline(ticker, trade_date, force_refresh=False, on_stage=None):
        nonlocal in_flight, max_in_flight
        in_flight += 1
        max_in_flight = max(max_in_flight, in_flight)
        await asyncio.sleep(0)
        in_flight -= 1

    monkeypatch.setattr(auto_trade_loop, "run_pipeline", fake_run_pipeline)

    await auto_trade_loop.run_cycle_now()

    assert max_in_flight == 2  # both candidates ran concurrently, cap never binding


async def test_status_current_tickers_reflects_every_in_flight_ticker(monkeypatch):
    monkeypatch.setattr(settings, "auto_trade_max_concurrent_analyses", 2)

    async def fake_domestic(broker, exclude=frozenset()):
        return ["005930.KS", "000660.KS"]

    async def fake_overseas(broker, exclude=frozenset()):
        return []

    monkeypatch.setattr(auto_trade_loop, "_domestic_candidates", fake_domestic)
    monkeypatch.setattr(auto_trade_loop, "_overseas_candidates", fake_overseas)

    observed_max = 0

    async def fake_run_pipeline(ticker, trade_date, force_refresh=False, on_stage=None):
        nonlocal observed_max
        await asyncio.sleep(0)
        observed_max = max(observed_max, len(auto_trade_loop.get_status().current_tickers))

    monkeypatch.setattr(auto_trade_loop, "run_pipeline", fake_run_pipeline)

    await auto_trade_loop.run_cycle_now()

    assert observed_max == 2
    assert auto_trade_loop.get_status().current_tickers == set()  # cleared once the cycle finishes


async def test_status_current_tickers_discards_a_ticker_even_when_its_pipeline_fails(monkeypatch):
    async def fake_domestic(broker, exclude=frozenset()):
        return ["005930.KS"]

    async def fake_overseas(broker, exclude=frozenset()):
        return []

    monkeypatch.setattr(auto_trade_loop, "_domestic_candidates", fake_domestic)
    monkeypatch.setattr(auto_trade_loop, "_overseas_candidates", fake_overseas)

    async def fake_run_pipeline(ticker, trade_date, force_refresh=False, on_stage=None):
        raise RuntimeError("LLM backend timeout")

    monkeypatch.setattr(auto_trade_loop, "run_pipeline", fake_run_pipeline)

    await auto_trade_loop.run_cycle_now()  # must not raise
    assert auto_trade_loop.get_status().current_tickers == set()


def test_forget_ticker_clears_gap_warned_on():
    auto_trade_loop._gap_warned_on[(settings.kis_account_mode, "005930.KS")] = date(2026, 8, 13)
    auto_trade_loop.forget_ticker("005930.KS")
    assert (settings.kis_account_mode, "005930.KS") not in auto_trade_loop._gap_warned_on
