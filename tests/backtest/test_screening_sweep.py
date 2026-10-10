"""app.backtest.screening_sweep — LLM-free, _scored_candidates/get_price_series/
_forward_return faked, no network. Variant-based two-phase structure (Phase A
network, Phase B pure) — the call-count tests pin the "fetch once, not once
per variant" property."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from app.chat.screening import MomentumScore
from app.domain.models.market import Candle
from app.config import settings
from app.backtest import screening_sweep


def _score(raw_pct: float, atr_pct: float = 0.02) -> MomentumScore:
    return MomentumScore(raw_pct=raw_pct, atr_pct=atr_pct, score=raw_pct / atr_pct)


# --- _forward_return (signature unchanged; now sourced from kis_provider) --

async def test_forward_return_computes_buy_and_hold_over_holding_window(monkeypatch):
    async def fake_price_series(ticker, start, end):
        return {date(2026, 7, 1): 100.0, date(2026, 7, 11): 110.0}

    monkeypatch.setattr(screening_sweep, "get_price_series", fake_price_series)
    result = await screening_sweep._forward_return("AAPL", date(2026, 7, 1), holding_days=10)
    assert result == pytest.approx(0.10)


async def test_forward_return_none_when_no_price_history(monkeypatch):
    async def fake_price_series(ticker, start, end):
        raise ValueError("no price history")

    monkeypatch.setattr(screening_sweep, "get_price_series", fake_price_series)
    assert await screening_sweep._forward_return("AAPL", date(2026, 7, 1), holding_days=10) is None


async def test_forward_return_none_when_prices_empty(monkeypatch):
    async def fake_price_series(ticker, start, end):
        return {}

    monkeypatch.setattr(screening_sweep, "get_price_series", fake_price_series)
    assert await screening_sweep._forward_return("AAPL", date(2026, 7, 1), holding_days=10) is None


# --- _forward_return: data-gap guard (2026-09-11 fix) ---------------------
#
# price_on_or_before falls back to the closest date STRICTLY BEFORE the
# target when the target itself is missing -- correct for an ordinary
# weekend/holiday gap, but silently returns a truncated-window "return" when
# the feed's history simply doesn't reach the target end date (recent
# listing, delisted, or the sweep run before the target date has happened).


async def test_forward_return_computes_normally_when_data_reaches_target_end(monkeypatch):
    # target_end = 2026-07-11; latest price is exactly on target_end -> no gap.
    async def fake_price_series(ticker, start, end):
        return {date(2026, 7, 1): 100.0, date(2026, 7, 11): 110.0}

    monkeypatch.setattr(screening_sweep, "get_price_series", fake_price_series)
    result = await screening_sweep._forward_return("AAPL", date(2026, 7, 1), holding_days=10)
    assert result == pytest.approx(0.10)


async def test_forward_return_none_when_data_ends_well_before_target_window(monkeypatch):
    # target_end = 2026-07-11, but the latest available price is 2026-06-25
    # -- a 16-day gap, far past _FORWARD_RETURN_DATA_GAP_TOLERANCE_DAYS (4).
    # Without the fix this would silently compute a truncated-window return
    # instead of reporting the data doesn't cover the requested window.
    async def fake_price_series(ticker, start, end):
        return {date(2026, 7, 1): 100.0, date(2026, 6, 25): 90.0}

    monkeypatch.setattr(screening_sweep, "get_price_series", fake_price_series)
    result = await screening_sweep._forward_return("AAPL", date(2026, 7, 1), holding_days=10)
    assert result is None


async def test_forward_return_normal_weekend_gap_still_computes(monkeypatch):
    # target_end (2026-07-11, a Saturday) has no price; the latest price is
    # the preceding Friday (2026-07-10) -- a 1-day gap, well under the
    # tolerance, so this must NOT be rejected: price_on_or_before's normal,
    # intended weekend fallback should still resolve it.
    assert date(2026, 7, 11).weekday() == 5  # confirm it's actually a Saturday
    async def fake_price_series(ticker, start, end):
        return {date(2026, 7, 1): 100.0, date(2026, 7, 10): 108.0}

    monkeypatch.setattr(screening_sweep, "get_price_series", fake_price_series)
    result = await screening_sweep._forward_return("AAPL", date(2026, 7, 1), holding_days=10)
    assert result == pytest.approx(0.08)


# --- shared fake wiring --------------------------------------------------

def _wire(monkeypatch, *, scored_counter=None, forward_counter=None, forward_map=None):
    universe_scores = {"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0}
    forward_map = forward_map or {"A": 0.40, "B": 0.30, "C": 0.20, "D": 0.10}

    async def fake_scored(universe, lookback_days, as_of=None):
        if scored_counter is not None:
            scored_counter.append((lookback_days, as_of))
        return {t: _score(universe_scores[t]) for t in universe}

    async def fake_forward(ticker, trade_date, holding_days):
        if forward_counter is not None:
            forward_counter.append((ticker, trade_date))
        return forward_map[ticker]

    monkeypatch.setattr(screening_sweep, "_scored_candidates", fake_scored)
    monkeypatch.setattr(screening_sweep, "_forward_return", fake_forward)


DATES = [date(2026, 7, 1), date(2026, 7, 8)]
UNIVERSE = ["A", "B", "C", "D"]


# case 44 — 3 variants, one lookback -> scored once per date, not per variant
async def test_scored_candidates_called_once_per_date_not_per_variant(monkeypatch):
    calls: list = []
    _wire(monkeypatch, scored_counter=calls)
    variants = [
        screening_sweep.Variant("a", lookback_days=10, use_correlation_dedup=False),
        screening_sweep.Variant("b", lookback_days=10, use_volatility_adjustment=False, use_correlation_dedup=False),
        screening_sweep.Variant("c", lookback_days=10, use_correlation_dedup=False, sign=-1),
    ]
    await screening_sweep.sweep(UNIVERSE, DATES, variants, top_n=2, holding_days=10, cadence_days=5)
    assert len(calls) == len(DATES)


# case 45 — forward return fetched once per (ticker, date), not per variant
async def test_forward_return_called_once_per_ticker_date(monkeypatch):
    calls: list = []
    _wire(monkeypatch, forward_counter=calls)
    variants = [
        screening_sweep.Variant("a", lookback_days=10, use_correlation_dedup=False),
        screening_sweep.Variant("b", lookback_days=10, use_correlation_dedup=False, sign=-1),
    ]
    await screening_sweep.sweep(UNIVERSE, DATES, variants, top_n=2, holding_days=10, cadence_days=5)
    assert len(calls) == len(DATES) * len(UNIVERSE)


# case 46 — sign=-1 reverses the pick set
async def test_sign_negative_reverses_picks(monkeypatch):
    _wire(monkeypatch)
    pos = screening_sweep.Variant("pos", lookback_days=10, use_volatility_adjustment=False, use_correlation_dedup=False)
    neg = screening_sweep.Variant("neg", lookback_days=10, use_volatility_adjustment=False, use_correlation_dedup=False, sign=-1)
    results = await screening_sweep.sweep(UNIVERSE, DATES, [pos, neg], top_n=2, holding_days=10, cadence_days=5)
    by_label = {r.variant.label: r for r in results}
    assert by_label["pos"].observations[0].picks == ["D", "C"]
    assert by_label["neg"].observations[0].picks == ["A", "B"]


# case 52 — mean_forward_return equals the old pooled-mean formula
async def test_mean_forward_return_matches_old_pooled_formula(monkeypatch):
    _wire(monkeypatch)
    v = screening_sweep.Variant("v", lookback_days=10, use_volatility_adjustment=False, use_correlation_dedup=False)
    results = await screening_sweep.sweep(UNIVERSE, DATES, [v], top_n=2, holding_days=10, cadence_days=5)

    # old formula: pool every usable pick forward return across every date, mean.
    picks_each_date = ["D", "C"]  # top 2 by raw_pct
    fwd = {"A": 0.40, "B": 0.30, "C": 0.20, "D": 0.10}
    pooled = [fwd[t] for _ in DATES for t in picks_each_date]
    assert results[0].mean_forward_return == pytest.approx(sum(pooled) / len(pooled))


async def test_pool_size_formula_matches_live_screening_expression():
    # MUST match app.chat.screening._ranked_and_deduped's pool_size line.
    for n, top_n in [(3, 2), (10, 4), (4, 4), (100, 5)]:
        assert screening_sweep._pool_size(n, top_n) == min(
            n, top_n * settings.screening_correlation_pool_multiplier
        )


async def test_correlation_dedup_uses_clusters_and_caches_them(monkeypatch):
    _wire(monkeypatch)
    cluster_calls: list = []

    async def fake_clusters(tickers, as_of=None):
        cluster_calls.append((tuple(tickers), as_of))
        return {t: 0 for t in tickers}  # all one cluster -> only the top survives

    monkeypatch.setattr(screening_sweep, "compute_correlation_clusters", fake_clusters)
    v = screening_sweep.Variant("d", lookback_days=10, use_volatility_adjustment=False, use_correlation_dedup=True)
    results = await screening_sweep.sweep(UNIVERSE, DATES, [v, v], top_n=2, holding_days=10, cadence_days=5)

    # all in one cluster -> single pick survives dedup
    assert results[0].observations[0].picks == ["D"]
    # cached by (frozenset(pool), date): 2 dates, 2 identical variants -> 2 calls
    assert len(cluster_calls) == len(DATES)


async def test_turnover_is_none_on_first_date_then_computed(monkeypatch):
    _wire(monkeypatch)
    v = screening_sweep.Variant("v", lookback_days=10, use_volatility_adjustment=False, use_correlation_dedup=False)
    results = await screening_sweep.sweep(UNIVERSE, DATES, [v], top_n=2, holding_days=10, cadence_days=5)
    obs = results[0].observations
    assert obs[0].turnover is None
    assert obs[1].turnover == pytest.approx(0.0)  # same picks both dates


async def test_n_eff_deflates_with_holding_overlap(monkeypatch):
    _wire(monkeypatch)
    v = screening_sweep.Variant("v", lookback_days=10, use_correlation_dedup=False)
    results = await screening_sweep.sweep(
        UNIVERSE, DATES, [v], top_n=2, holding_days=20, cadence_days=5
    )
    # 2 dates, ceil(20/5)=4 overlap -> 0.5
    assert results[0].n_eff == pytest.approx(0.5)


def test_render_runs_and_includes_header_and_guide():
    v = screening_sweep.Variant("lb10", lookback_days=10)
    result = screening_sweep.VariantResult(variant=v, n_dates=5, mean_excess_return=0.01)
    out = screening_sweep.render(
        [result], universe=["A.KS", "B.KS"], cadence_days=5, holding_days=10, seed=0, bootstrap_iters=10_000
    )
    assert "cadence=5d" in out
    assert "lb10" in out
    assert "INTERPRETATION GUIDE" in out


def test_variants_from_args_rejects_both_options():
    import argparse

    ns = argparse.Namespace(variants_json="x.json", lookback_options="5,10")
    with pytest.raises(SystemExit):
        screening_sweep._variants_from_args(ns)


async def test_universe_from_live_axes_rejects_unknown_axis():
    # Pure validation — must raise before touching get_broker/network at all.
    with pytest.raises(SystemExit):
        await screening_sweep._universe_from_live_axes(
            ["not_a_real_axis"], 30, min_volume=0, disparity_window_days=20
        )


async def test_universe_from_live_axes_threads_max_price_to_every_axis(monkeypatch):
    # 2026-09-02 (screening-improvement-research #3 follow-up,
    # --universe-max-price) — the sweep's A/B lever for
    # app.config.auto_trade_universe_price_ceiling_domestic must reach all
    # four axis functions, not just some of them.
    captured = {}

    async def fake_get_broker(name):
        return "fake-broker"

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

    import app.brokers as brokers_module
    import app.live.universe as universe_module

    monkeypatch.setattr(brokers_module, "get_broker", fake_get_broker)
    monkeypatch.setattr(universe_module, "krx_top_n", fake_krx_top_n)
    monkeypatch.setattr(universe_module, "krx_market_cap_top_n", fake_krx_market_cap_top_n)
    monkeypatch.setattr(universe_module, "krx_fluctuation_losers_n", fake_krx_fluctuation_losers_n)
    monkeypatch.setattr(universe_module, "krx_disparity_low_n", fake_krx_disparity_low_n)

    await screening_sweep._universe_from_live_axes(
        list(screening_sweep._UNIVERSE_AXES), 30, min_volume=0, disparity_window_days=20, max_price=400_000,
    )
    assert captured == {
        "trading_value": 400_000,
        "market_cap": 400_000,
        "fluctuation_losers": 400_000,
        "disparity_low": 400_000,
    }


async def test_universe_from_live_axes_default_max_price_is_none(monkeypatch):
    captured = {}

    async def fake_get_broker(name):
        return "fake-broker"

    async def fake_krx_top_n(broker, count, *, max_price=None):
        captured["max_price"] = max_price
        return []

    import app.brokers as brokers_module
    import app.live.universe as universe_module

    monkeypatch.setattr(brokers_module, "get_broker", fake_get_broker)
    monkeypatch.setattr(universe_module, "krx_top_n", fake_krx_top_n)

    await screening_sweep._universe_from_live_axes(
        ["trading_value"], 30, min_volume=0, disparity_window_days=20,
    )
    assert captured["max_price"] is None


def test_sweep_cli_rejects_tickers_and_axes_together():
    import argparse

    ns = argparse.Namespace(universe_axes="trading_value", tickers="AAPL")
    with pytest.raises(SystemExit):
        screening_sweep._check_universe_source_args(ns)


def test_variants_from_args_default_lookback_grid():
    import argparse

    ns = argparse.Namespace(variants_json=None, lookback_options=None)
    variants = screening_sweep._variants_from_args(ns)
    assert [v.lookback_days for v in variants] == [5, 10, 15, 20]
    assert [v.label for v in variants] == ["lb5", "lb10", "lb15", "lb20"]


# --- Phase A candle_run_cache scope (7ebb91d follow-up) ------------------

def _make_candle(day: date, px: float) -> Candle:
    return Candle(day=day.strftime("%Y%m%d"), open=px, high=px + 1.0, low=px - 1.0, close=px, volume=1000)


class _CountingCandleBroker:
    def __init__(self):
        self.calls = []

    async def fetch_daily_candles(self, code, start, end):
        self.calls.append((code, start, end))
        s = date(int(start[:4]), int(start[4:6]), int(start[6:]))
        e = date(int(end[:4]), int(end[4:6]), int(end[6:]))
        out, day, px = [], s, 100.0
        while day <= e:
            px += 0.5
            out.append(_make_candle(day, px))
            day += timedelta(days=1)
        return out


async def test_phase_a_fetches_candles_once_per_ticker_end_not_once_per_lookback(monkeypatch):
    from app.dataflows import kis_provider
    broker = _CountingCandleBroker()

    async def _get_broker():
        return broker

    monkeypatch.setattr(kis_provider, "_get_broker", _get_broker)
    tickers = ["005930.KS", "000660.KS"]
    dates = [date(2026, 7, 1), date(2026, 7, 8)]
    lookbacks = [5, 10, 20, 60]
    await screening_sweep._phase_a(tickers, dates, lookbacks, holding_days=10)
    keys = {(c, e) for c, _s, e in broker.calls}
    assert len(keys) == len(tickers) * len(dates) * 2
    assert len(broker.calls) == len(keys)  # each key fetched once
    assert len(broker.calls) == 8  # not 2*2*(4*2+1)=36


async def test_phase_a_runs_inside_a_candle_run_cache_scope(monkeypatch):
    from app.dataflows import kis_provider
    seen = []

    async def fake_scored(universe, lookback_days, as_of=None):
        seen.append(kis_provider._candle_run_cache.get() is not None)
        return {t: _score(1.0) for t in universe}

    async def fake_forward(ticker, trade_date, holding_days):
        seen.append(kis_provider._candle_run_cache.get() is not None)
        return 0.0

    monkeypatch.setattr(screening_sweep, "_scored_candidates", fake_scored)
    monkeypatch.setattr(screening_sweep, "_forward_return", fake_forward)
    await screening_sweep._phase_a(["A"], [date(2026, 7, 1)], [10], holding_days=10)
    assert seen and all(seen)
    assert kis_provider._candle_run_cache.get() is None


# --- residual momentum (screening-improvement-research #4, 2026-09-02) ----

def test_residual_max_lookback_matches_kis_provider_constant():
    # MUST match app.dataflows.kis_provider._MAX_KIS_CANDLE_RANGE_DAYS —
    # see screening_sweep._RESIDUAL_MAX_LOOKBACK_DAYS's own comment for why.
    from app.dataflows import kis_provider

    assert screening_sweep._RESIDUAL_MAX_LOOKBACK_DAYS == kis_provider._MAX_KIS_CANDLE_RANGE_DAYS


def test_total_mode_score_matches_precomputed_momentum_score():
    v_adj = screening_sweep.Variant("v", lookback_days=10)
    v_noadj = screening_sweep.Variant("v2", lookback_days=10, use_volatility_adjustment=False)
    for raw_pct, atr_pct in [(0.05, 0.02), (-0.03, 0.01), (0.10, None), (0.0, 0.05)]:
        s = MomentumScore(
            raw_pct=raw_pct, atr_pct=atr_pct,
            score=screening_sweep.risk_adjusted_score(raw_pct, atr_pct),
        )
        assert screening_sweep._variant_score(s, v_adj, None) == pytest.approx(s.score)
        assert screening_sweep._variant_score(s, v_noadj, None) == pytest.approx(s.raw_pct)


async def test_benchmark_fetched_once_per_date_not_once_per_ticker(monkeypatch):
    _wire(monkeypatch)
    calls: list = []

    async def fake_benchmark_series(market, end, max_lookback_days):
        calls.append((market, end, max_lookback_days))
        return {end - timedelta(days=i): 100.0 + i for i in range(3)}

    monkeypatch.setattr(screening_sweep, "_benchmark_series", fake_benchmark_series)
    variants = [
        screening_sweep.Variant("r1", lookback_days=5, use_correlation_dedup=False, momentum_mode="residual"),
        screening_sweep.Variant("r2", lookback_days=10, use_correlation_dedup=False, momentum_mode="residual"),
        screening_sweep.Variant("r3", lookback_days=20, use_correlation_dedup=False, momentum_mode="residual"),
    ]
    await screening_sweep.sweep(UNIVERSE, DATES, variants, top_n=2, holding_days=10, cadence_days=5)
    # UNIVERSE ("A".."D") are all plain-letter tickers -> ticker_market resolves
    # them all to "overseas" -> one market -> exactly one call per date.
    assert len(calls) == len(DATES)
    for _market, _end, max_lb in calls:
        assert max_lb == 20  # max(distinct_lookbacks), fetched once, sliced per-lookback client-side


async def test_no_benchmark_fetch_when_no_residual_variant(monkeypatch):
    _wire(monkeypatch)

    async def fail_benchmark_series(*args, **kwargs):
        raise AssertionError("_benchmark_series must not be called when no variant is residual")

    monkeypatch.setattr(screening_sweep, "_benchmark_series", fail_benchmark_series)
    v = screening_sweep.Variant("total", lookback_days=10, use_correlation_dedup=False)
    await screening_sweep.sweep(UNIVERSE, DATES, [v], top_n=2, holding_days=10, cadence_days=5)


async def test_benchmark_failure_degrades_that_date_to_total_momentum(monkeypatch):
    _wire(monkeypatch)

    async def fake_benchmark_series(market, end, max_lookback_days):
        if end == DATES[0]:
            return None
        return {end - timedelta(days=i): 100.0 + i for i in range(3)}

    monkeypatch.setattr(screening_sweep, "_benchmark_series", fake_benchmark_series)
    total_v = screening_sweep.Variant("total", lookback_days=10, use_correlation_dedup=False)
    resid_v = screening_sweep.Variant(
        "resid", lookback_days=10, use_correlation_dedup=False, momentum_mode="residual"
    )
    results = await screening_sweep.sweep(
        UNIVERSE, DATES, [total_v, resid_v], top_n=2, holding_days=10, cadence_days=5
    )
    by_label = {r.variant.label: r for r in results}

    # Date 0's benchmark fetch failed -> residual degrades to total for that date.
    assert by_label["resid"].observations[0].picks == by_label["total"].observations[0].picks
    assert len(by_label["resid"].observations) == len(DATES)  # date not excluded
    assert by_label["resid"].observations[0].residual_fallbacks == len(UNIVERSE)
    # Date 1's benchmark fetch succeeded -> no fallback there.
    assert by_label["resid"].observations[1].residual_fallbacks == 0
    assert by_label["resid"].residual_fallback_dates == 1


async def test_variants_from_args_parses_and_validates_momentum_mode(tmp_path):
    import argparse
    import json as json_mod

    # default (unset) -> "total"
    ns = argparse.Namespace(variants_json=None, lookback_options="10")
    variants = screening_sweep._variants_from_args(ns)
    assert variants[0].momentum_mode == "total"

    # explicit "residual" parses via the JSON path
    good_path = tmp_path / "variants.json"
    good_path.write_text(
        json_mod.dumps(
            [
                {"label": "a", "lookback_days": 10},
                {"label": "b", "lookback_days": 10, "momentum_mode": "residual"},
            ]
        ),
        encoding="utf-8",
    )
    ns_good = argparse.Namespace(variants_json=str(good_path), lookback_options=None)
    variants = screening_sweep._variants_from_args(ns_good)
    assert variants[0].momentum_mode == "total"
    assert variants[1].momentum_mode == "residual"

    # unknown value -> SystemExit
    bad_path = tmp_path / "bad.json"
    bad_path.write_text(
        json_mod.dumps([{"label": "c", "lookback_days": 10, "momentum_mode": "reversal"}]),
        encoding="utf-8",
    )
    ns_bad = argparse.Namespace(variants_json=str(bad_path), lookback_options=None)
    with pytest.raises(SystemExit):
        screening_sweep._variants_from_args(ns_bad)


async def test_benchmark_pct_per_lookback_slices_one_series(monkeypatch):
    from app.chat.screening import pct_change

    d = date(2026, 7, 10)
    series = {d: 100.0, d - timedelta(days=3): 90.0, d - timedelta(days=10): 80.0}

    async def fake_benchmark_series(market, end, max_lookback_days):
        return dict(series)

    monkeypatch.setattr(screening_sweep, "_benchmark_series", fake_benchmark_series)
    result = await screening_sweep._benchmark_pct_by_lookback({"domestic"}, d, [1, 5])

    window5 = {day: px for day, px in series.items() if d - timedelta(days=5) <= day <= d}
    assert result[("domestic", 5, d)] == pytest.approx(pct_change(window5))
    # lb=1's window contains only `d` itself (1 bar) -> pct_change can't compute -> None.
    assert result[("domestic", 1, d)] is None


async def test_mixed_universe_resolves_benchmark_per_market(monkeypatch):
    seen_markets: set = set()

    async def fake_benchmark_series(market, end, max_lookback_days):
        seen_markets.add(market)
        if market == "domestic":
            return {end - timedelta(days=1): 100.0, end: 150.0}  # pct_change = +0.50
        return {end - timedelta(days=1): 100.0, end: 100.0}  # pct_change = 0.0

    async def fake_scored(universe, lookback_days, as_of=None):
        return {t: _score(1.0) for t in universe}

    async def fake_forward(ticker, trade_date, holding_days):
        return 0.05

    monkeypatch.setattr(screening_sweep, "_benchmark_series", fake_benchmark_series)
    monkeypatch.setattr(screening_sweep, "_scored_candidates", fake_scored)
    monkeypatch.setattr(screening_sweep, "_forward_return", fake_forward)

    v = screening_sweep.Variant(
        "resid", lookback_days=10, use_volatility_adjustment=False,
        use_correlation_dedup=False, momentum_mode="residual",
    )
    universe = ["005930.KS", "AAPL"]
    results = await screening_sweep.sweep(
        universe, [date(2026, 7, 1)], [v], top_n=1, holding_days=10, cadence_days=5
    )
    assert seen_markets == {"domestic", "overseas"}
    # Both tickers share raw_pct=1.0; 005930.KS (domestic) gets the +0.50
    # benchmark subtracted (residual 0.50), AAPL (overseas) gets 0.0
    # subtracted (residual 1.0) -> AAPL ranks first only if its lookup used
    # the OVERSEAS value, not the domestic one.
    assert results[0].observations[0].picks == ["AAPL"]


async def test_residual_lookback_above_cap_raises_before_any_network(monkeypatch):
    async def fail_phase_a(*args, **kwargs):
        raise AssertionError("_phase_a must not be called — the cap check must happen first")

    monkeypatch.setattr(screening_sweep, "_phase_a", fail_phase_a)
    v = screening_sweep.Variant(
        "resid_big",
        lookback_days=screening_sweep._RESIDUAL_MAX_LOOKBACK_DAYS + 1,
        momentum_mode="residual",
    )
    with pytest.raises(SystemExit):
        await screening_sweep.sweep(["A"], DATES, [v], top_n=2, holding_days=10, cadence_days=5)


async def test_dedup_drop_rate_computed_from_pre_dedup_pool(monkeypatch):
    _wire(monkeypatch)

    async def fake_clusters(tickers, as_of=None):
        return {t: 0 for t in tickers}  # all one cluster -> only the top-ranked survives

    monkeypatch.setattr(screening_sweep, "compute_correlation_clusters", fake_clusters)

    v = screening_sweep.Variant("d", lookback_days=10, use_volatility_adjustment=False, use_correlation_dedup=True)
    results = await screening_sweep.sweep(UNIVERSE, DATES, [v], top_n=2, holding_days=10, cadence_days=5)
    obs = results[0].observations[0]
    # pre-dedup top 2 by raw_pct: D, C. post-dedup (all one cluster): D only.
    # overlap=1 -> drop_rate = 1 - 1/2 = 0.5
    assert obs.picks == ["D"]
    assert obs.dedup_drop_rate == pytest.approx(0.5)

    v2 = screening_sweep.Variant(
        "nodedup", lookback_days=10, use_volatility_adjustment=False, use_correlation_dedup=False
    )
    results2 = await screening_sweep.sweep(UNIVERSE, DATES, [v2], top_n=2, holding_days=10, cadence_days=5)
    assert results2[0].observations[0].dedup_drop_rate is None


async def test_dedup_drop_rate_is_zero_not_fabricated_on_a_thin_universe_day(monkeypatch):
    # 2026-09-11 fix: on a day where fewer tickers are scored than top_n,
    # _pool_size caps the pool below top_n so dedupe_correlated never runs
    # (`if len(pool) > top_n` is False) -- pre_dedup_top and post_dedup_top
    # are then the IDENTICAL list. The old code divided the overlap by the
    # fixed top_n regardless, so `overlap == len(pre_dedup_top) < top_n`
    # produced a fabricated positive "drop rate" on a day nothing was
    # actually dropped for correlation. Dividing by len(pre_dedup_top)
    # instead makes this correctly read 0.0.
    async def fake_scored(universe, lookback_days, as_of=None):
        return {"A": _score(1.0)}  # only 1 ticker scored

    async def fake_forward(ticker, trade_date, holding_days):
        return 0.05

    async def fake_clusters(tickers, as_of=None):
        raise AssertionError("compute_correlation_clusters must not run when the pool is <= top_n")

    monkeypatch.setattr(screening_sweep, "_scored_candidates", fake_scored)
    monkeypatch.setattr(screening_sweep, "_forward_return", fake_forward)
    monkeypatch.setattr(screening_sweep, "compute_correlation_clusters", fake_clusters)

    v = screening_sweep.Variant("d", lookback_days=10, use_volatility_adjustment=False, use_correlation_dedup=True)
    results = await screening_sweep.sweep(["A"], [DATES[0]], [v], top_n=2, holding_days=10, cadence_days=5)
    obs = results[0].observations[0]
    assert obs.picks == ["A"]
    assert obs.dedup_drop_rate == pytest.approx(0.0)


# --- excess-return CI weighting (2026-09-11 fix) --------------------------
#
# mean_excess_return is a plain mean over ONE value per date
# (o.excess_return, itself already mean(that date's picks) - mean(that
# date's universe)) -- every date weighted equally regardless of how many
# picks it had. The old excess_blocks built, per date, a LIST of that
# date's raw per-pick demeaned returns; block_bootstrap_ci then pools every
# value across resampled blocks and means the pool, which implicitly
# weights each date by its OWN pick count -- a different, pick-weighted
# statistic than the date-weighted mean_excess_return it's supposed to
# describe. The fix makes excess_blocks hold one already-averaged value per
# date (wrapped in a single-element block), matching ic_blocks' own
# convention immediately below it.


async def test_excess_blocks_are_one_value_per_date_matching_ic_blocks_convention(monkeypatch):
    # Two dates with DIFFERENT pick counts (2 vs 1) -- the exact condition
    # under which the old pick-weighted pooling diverged from the reported
    # date-weighted mean_excess_return.
    d1, d2 = DATES
    scored_cache = {
        (10, d1): {t: _score(v) for t, v in {"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0}.items()},
        (10, d2): {"A": _score(1.0)},  # thin day: only 1 ticker scored -> 1 pick, not 2
    }
    forward_cache = {
        d1: {"A": 0.40, "B": 0.30, "C": 0.20, "D": 0.10},
        d2: {"A": 0.05},
    }
    v = screening_sweep.Variant("v", lookback_days=10, use_volatility_adjustment=False, use_correlation_dedup=False)
    observations = await screening_sweep._variant_observations(
        v, [d1, d2], scored_cache, forward_cache, top_n=2, cluster_cache={},
    )
    assert len(observations) == 2
    assert len(observations[0].pick_forward_returns) == 2  # full day: 2 picks
    assert len(observations[1].pick_forward_returns) == 1  # thin day: 1 pick

    captured_blocks: list[list[list[float]]] = []
    real_ci = screening_sweep.factor_metrics.block_bootstrap_ci

    def spy_ci(blocks, n_bootstrap=10_000, confidence=0.95, seed=None):
        captured_blocks.append(blocks)
        return real_ci(blocks, n_bootstrap=n_bootstrap, confidence=confidence, seed=seed)

    monkeypatch.setattr(screening_sweep.factor_metrics, "block_bootstrap_ci", spy_ci)
    result = screening_sweep._summarise_variant(
        v, observations, cadence_days=5, holding_days=10, worth_bar=0.03, bootstrap_iters=500, seed=0,
    )

    # First call (excess_return_ci95, evaluated before rank_ic_ci95 in the
    # VariantResult(...) construction) must be excess_blocks: one block per
    # date, each with exactly ONE element (that date's own excess_return) --
    # NOT the old per-pick-count-sized blocks.
    excess_blocks_seen = captured_blocks[0]
    assert len(excess_blocks_seen) == len(observations) == 2
    assert all(len(b) == 1 for b in excess_blocks_seen)
    expected = [o.excess_return for o in observations]
    assert [b[0] for b in excess_blocks_seen] == pytest.approx(expected)

    # The CI now describes the SAME statistic as the point estimate: for
    # this 2-date synthetic case (block_bootstrap_ci needs >=3 blocks, so
    # with only 2 real observations here it degrades to None) the property
    # that matters is the block *construction*, already asserted above.
    assert result.excess_return_ci95 is None  # < 3 blocks -> block_bootstrap_ci's own guard


async def test_excess_return_ci_brackets_mean_when_pick_counts_vary(monkeypatch):
    # A real (>=3 dates) case demonstrating the restored property: the CI's
    # midpoint tracks mean_excess_return even when per-date pick counts
    # differ, which the old pick-weighted pooling could violate.
    dates = [date(2026, 7, 1), date(2026, 7, 8), date(2026, 7, 15)]
    scored_cache = {
        (10, dates[0]): {t: _score(v) for t, v in {"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0}.items()},
        (10, dates[1]): {"A": _score(1.0)},  # thin: 1 pick
        (10, dates[2]): {t: _score(v) for t, v in {"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0}.items()},
    }
    forward_cache = {
        dates[0]: {"A": 0.40, "B": 0.30, "C": 0.20, "D": 0.10},
        dates[1]: {"A": -0.30},
        dates[2]: {"A": 0.40, "B": 0.30, "C": 0.20, "D": 0.10},
    }
    v = screening_sweep.Variant("v", lookback_days=10, use_volatility_adjustment=False, use_correlation_dedup=False)
    observations = await screening_sweep._variant_observations(
        v, dates, scored_cache, forward_cache, top_n=2, cluster_cache={},
    )
    result = screening_sweep._summarise_variant(
        v, observations, cadence_days=5, holding_days=10, worth_bar=0.03, bootstrap_iters=5000, seed=0,
    )
    assert result.excess_return_ci95 is not None
    lo, hi = result.excess_return_ci95
    assert lo <= result.mean_excess_return <= hi


def test_render_includes_residual_fallback_and_dedup_columns():
    v = screening_sweep.Variant("lb10", lookback_days=10)
    result = screening_sweep.VariantResult(
        variant=v, n_dates=5, mean_excess_return=0.01,
        residual_fallback_dates=2, mean_dedup_drop_rate=0.25,
    )
    out = screening_sweep.render(
        [result], universe=["A.KS", "B.KS"], cadence_days=5, holding_days=10, seed=0, bootstrap_iters=10_000
    )
    assert "resid_fb" in out
    assert "dedup_drop" in out
    assert "2/5" in out
    assert "25%" in out
    # 2/5 = 40% > 20% threshold -> hard-rule warning fires.
    assert "degraded to total momentum on 40%" in out


async def test_clustering_failure_warns_and_negative_caches(monkeypatch, capsys):
    # 2026-09-11 fix: compute_correlation_clusters raising used to be
    # silently swallowed as {} (dedupe_correlated({}) drops nothing), making
    # the dedup=on arm behave exactly like dedup=off with no warning.
    _wire(monkeypatch)
    calls = []

    async def failing_clusters(tickers, as_of=None):
        calls.append((tuple(tickers), as_of))
        raise RuntimeError("rate limited")

    monkeypatch.setattr(screening_sweep, "compute_correlation_clusters", failing_clusters)

    v = screening_sweep.Variant("d", lookback_days=10, use_volatility_adjustment=False, use_correlation_dedup=True)
    # Two identical variants over the same dates -> same (pool, date) cache key twice.
    results = await screening_sweep.sweep(UNIVERSE, DATES, [v, v], top_n=2, holding_days=10, cadence_days=5)

    captured = capsys.readouterr()
    assert "correlation clustering failed" in captured.out
    # Negative-cached: only ONE call per distinct date, not one per variant.
    assert len(calls) == len(DATES)
    obs = results[0].observations[0]
    assert obs.dedup_degraded is True
    assert obs.dedup_drop_rate is None
    # Un-deduped picks still returned (behavior unchanged on failure) --
    # pre_dedup_top by raw_pct is D, C.
    assert obs.picks == ["D", "C"]


async def test_empty_clusters_is_not_degraded(monkeypatch):
    _wire(monkeypatch)

    async def empty_clusters(tickers, as_of=None):
        return {}

    monkeypatch.setattr(screening_sweep, "compute_correlation_clusters", empty_clusters)
    v = screening_sweep.Variant("d", lookback_days=10, use_volatility_adjustment=False, use_correlation_dedup=True)
    results = await screening_sweep.sweep(UNIVERSE, DATES, [v], top_n=2, holding_days=10, cadence_days=5)
    obs = results[0].observations[0]
    assert obs.dedup_degraded is False
    assert obs.dedup_drop_rate == pytest.approx(0.0)


def test_render_flags_dedup_degraded_dates():
    v = screening_sweep.Variant("lb10", lookback_days=10)
    result = screening_sweep.VariantResult(
        variant=v, n_dates=3, mean_excess_return=0.01, dedup_degraded_dates=2,
    )
    out = screening_sweep.render(
        [result], universe=["A.KS", "B.KS"], cadence_days=5, holding_days=10, seed=0, bootstrap_iters=10_000
    )
    assert "dedup_deg" in out
    assert "2/3" in out
    assert "behaved as dedup=off, not a clean A/B" in out

    result_clean = screening_sweep.VariantResult(
        variant=v, n_dates=3, mean_excess_return=0.01, dedup_degraded_dates=0,
    )
    out_clean = screening_sweep.render(
        [result_clean], universe=["A.KS", "B.KS"], cadence_days=5, holding_days=10, seed=0, bootstrap_iters=10_000
    )
    assert "behaved as dedup=off" not in out_clean


# --- --dump-json (T1, 2026-09-30) ----------------------------------------

import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402

DUMP_TICKERS = ["A", "B", "C", "D"]
DUMP_CAL = [date(2026, 7, 1), date(2026, 7, 8), date(2026, 7, 15)]


def _wire_dump(monkeypatch, *, fail_scored_on=None):
    """Fakes for the whole _main() path: calendar, scoring (lookback-dependent
    so signed-score checks are meaningful), forward detail, git, clustering."""
    base = {"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0}

    async def fake_calendar(ticker, start, end):
        return {d: 100.0 for d in DUMP_CAL}

    async def fake_scored(universe, lookback_days, as_of=None):
        if fail_scored_on is not None and as_of == fail_scored_on:
            raise RuntimeError("boom")
        return {t: _score(base[t] * lookback_days / 10, 0.02 + 0.001 * i) for i, t in enumerate(universe)}

    async def fake_detail(ticker, trade_date, holding_days):
        if ticker == "D":  # truncation-guard analogue: no data reaching the window
            return None, None, None
        r = {"A": 0.04, "B": -0.02, "C": 0.10}[ticker] + 0.001 * trade_date.day
        return r, trade_date, trade_date + timedelta(days=2)

    async def fake_forward(ticker, trade_date, holding_days):
        return (await fake_detail(ticker, trade_date, holding_days))[0]

    async def fake_clusters(tickers, as_of=None):
        return {}

    monkeypatch.setattr(screening_sweep, "_calendar_price_series", fake_calendar)
    monkeypatch.setattr(screening_sweep, "_scored_candidates", fake_scored)
    monkeypatch.setattr(screening_sweep, "_forward_detail", fake_detail)
    monkeypatch.setattr(screening_sweep, "_forward_return", fake_forward)
    monkeypatch.setattr(screening_sweep, "compute_correlation_clusters", fake_clusters)
    monkeypatch.setattr(screening_sweep, "_git_info", lambda: ("deadbeef", False))


def _argv(tmp_path, *extra):
    vj = tmp_path / "variants.json"
    vj.write_text(
        json.dumps([
            {"label": "pos", "lookback_days": 10, "sign": 1},
            {"label": "neg", "lookback_days": 10, "sign": -1},
            {"label": "pos20", "lookback_days": 20, "sign": 1},
        ]),
        encoding="utf-8",
    )
    return [
        "screening_sweep", "--start", "2026-07-01", "--end", "2026-07-15", "--tickers", ",".join(DUMP_TICKERS),
        "--variants-json", str(vj), "--top-n", "2", "--holding-days", "4", "--cadence-days", "1",
        "--bootstrap-iters", "50", *extra,
    ]


async def test_dump_json_off_vs_on_stdout_identical(monkeypatch, tmp_path, capsys):
    _wire_dump(monkeypatch)
    monkeypatch.setattr(sys, "argv", _argv(tmp_path))
    await screening_sweep._main()
    off = capsys.readouterr().out

    monkeypatch.setattr(sys, "argv", _argv(tmp_path, "--dump-json", str(tmp_path / "sub" / "dump.json")))
    await screening_sweep._main()
    on = capsys.readouterr().out
    assert off == on
    assert (tmp_path / "sub" / "dump.json").exists()  # parent dir created


async def test_dump_json_contents_match_in_memory_results(monkeypatch, tmp_path):
    _wire_dump(monkeypatch)
    captured = {}
    real_sweep = screening_sweep.sweep

    async def spy_sweep(*a, **k):
        captured["results"] = await real_sweep(*a, **k)
        return captured["results"]

    monkeypatch.setattr(screening_sweep, "sweep", spy_sweep)
    path = tmp_path / "dump.json"
    monkeypatch.setattr(sys, "argv", _argv(tmp_path, "--dump-json", str(path)))
    await screening_sweep._main()

    raw = path.read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf")
    dump = json.loads(raw.decode("utf-8"))
    assert dump["schema"] == "screening-sweep-dump-v1"
    assert dump["status"] == "complete"
    assert dump["git_head"] == "deadbeef" and dump["git_dirty"] is False
    assert dump["trade_dates"] == [d.isoformat() for d in DUMP_CAL]
    assert dump["universe"] == DUMP_TICKERS
    assert dump["holding_days"] == 4 and dump["cadence_days"] == 1 and dump["top_n"] == 2
    assert dump["args"]["dump_json"] == str(path)

    # scores: every scored ticker for every (lookback, date), unsigned
    for lb in (10, 20):
        for d in DUMP_CAL:
            sc = dump["scores"][f"{lb}|{d.isoformat()}"]
            assert set(sc) == set(DUMP_TICKERS)
            assert sc["A"]["raw_pct"] == pytest.approx(1.0 * lb / 10)
            assert sc["A"]["atr_pct"] == pytest.approx(0.02)

    # forward: every universe ticker, with exit_price_date; guard-None ticker -> nulls
    for d in DUMP_CAL:
        fw = dump["forward"][d.isoformat()]
        assert set(fw) == set(DUMP_TICKERS)
        assert fw["A"]["exit_price_date"] == (d + timedelta(days=2)).isoformat()
        assert fw["A"]["entry_date"] == d.isoformat()
        assert fw["D"] == {"fwd_return": None, "entry_date": None, "exit_price_date": None}

    # variants: observations equal the in-memory ones; signed scores = sign * unsigned
    results = captured["results"]
    assert [v["variant"]["label"] for v in dump["variants"]] == ["pos", "neg", "pos20"]
    for res, vd in zip(results, dump["variants"]):
        assert vd["variant"]["sign"] == res.variant.sign
        assert vd["summary"]["n_dates"] == res.n_dates
        assert "observations" not in vd["summary"]
        for o in res.observations:
            od = vd["observations"][o.trade_date.isoformat()]
            assert od["picks"] == o.picks
            assert list(od["pick_forward_returns"].values()) == pytest.approx(o.pick_forward_returns)
            assert set(od["pick_forward_returns"]) <= set(o.picks)
            assert od["universe_forward_returns"] == pytest.approx(o.universe_forward_returns)
            for key in ("excess_return", "rank_ic", "quantile_spread", "turnover", "dedup_drop_rate"):
                assert od[key] == (pytest.approx(getattr(o, key)) if getattr(o, key) is not None else None)
            assert od["residual_fallbacks"] == o.residual_fallbacks
            assert od["dedup_degraded"] == o.dedup_degraded
            assert set(od["signed_scores"]) == set(DUMP_TICKERS)
    pos, neg, pos20 = dump["variants"]
    for d in DUMP_CAL:
        key = d.isoformat()
        sp, sn = pos["observations"][key]["signed_scores"], neg["observations"][key]["signed_scores"]
        for t in DUMP_TICKERS:
            assert sn[t] == pytest.approx(-sp[t])
            raw = dump["scores"][f"10|{key}"][t]
            assert sp[t] == pytest.approx(raw["raw_pct"] / raw["atr_pct"])  # risk_adjusted_score for vol-adj variants
    assert dump["counters"]["forward_null"] == len(DUMP_CAL)  # ticker D each date


async def test_dump_json_incomplete_on_failure_reraises_original(monkeypatch, tmp_path):
    _wire_dump(monkeypatch, fail_scored_on=DUMP_CAL[1])
    path = tmp_path / "dump.json"
    monkeypatch.setattr(sys, "argv", _argv(tmp_path, "--dump-json", str(path)))
    with pytest.raises(RuntimeError, match="boom"):
        await screening_sweep._main()
    dump = json.loads(path.read_text(encoding="utf-8"))
    assert dump["status"] == "incomplete"
    assert dump["trade_dates"] == [DUMP_CAL[0].isoformat()]
    assert dump["trade_dates_requested"] == [d.isoformat() for d in DUMP_CAL]
    assert f"10|{DUMP_CAL[1].isoformat()}" not in dump["scores"]
    assert dump["variants"] == []


async def test_dump_write_failure_does_not_mask_original_exception(monkeypatch, tmp_path):
    _wire_dump(monkeypatch, fail_scored_on=DUMP_CAL[1])

    def broken_write(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(screening_sweep, "_write_dump", broken_write)
    monkeypatch.setattr(sys, "argv", _argv(tmp_path, "--dump-json", str(tmp_path / "d.json")))
    with pytest.raises(RuntimeError, match="boom"):
        await screening_sweep._main()


async def test_dump_write_failure_on_success_exits_nonzero(monkeypatch, tmp_path, capsys):
    _wire_dump(monkeypatch)

    def broken_write(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(screening_sweep, "_write_dump", broken_write)
    monkeypatch.setattr(sys, "argv", _argv(tmp_path, "--dump-json", str(tmp_path / "d.json")))
    with pytest.raises(SystemExit) as ei:
        await screening_sweep._main()
    assert "disk full" in str(ei.value)
    assert "Sweeping" in capsys.readouterr().out  # sweep output was still printed


def test_dump_serialises_nan_and_inf_as_null(tmp_path):
    c = screening_sweep._DumpCollector(meta={"started_at": None})
    c.scores["10|2026-07-01"] = {"A": {"raw_pct": math.nan, "atr_pct": math.inf}}
    c.completed_dates.append(date(2026, 7, 1))
    path = tmp_path / "nan.json"
    screening_sweep._write_dump(str(path), c, status="complete")
    text = path.read_text(encoding="utf-8")
    assert "NaN" not in text and "Infinity" not in text
    assert json.loads(text)["scores"]["10|2026-07-01"]["A"] == {"raw_pct": None, "atr_pct": None}
    assert not list(tmp_path.glob("*.tmp"))  # atomic write leaves no tmp file


async def test_forward_detail_reports_price_dates_and_guard_returns_all_none(monkeypatch):
    async def ok_prices(ticker, start, end):
        return {date(2026, 7, 1): 100.0, date(2026, 7, 10): 108.0}

    monkeypatch.setattr(screening_sweep, "get_price_series", ok_prices)
    ret, entry, exit_ = await screening_sweep._forward_detail("A", date(2026, 7, 1), 10)
    assert ret == pytest.approx(0.08) and entry == date(2026, 7, 1) and exit_ == date(2026, 7, 10)
    assert await screening_sweep._forward_return("A", date(2026, 7, 1), 10) == ret  # wrapper unchanged

    async def truncated(ticker, start, end):
        return {date(2026, 7, 1): 100.0, date(2026, 6, 25): 90.0}

    monkeypatch.setattr(screening_sweep, "get_price_series", truncated)
    assert await screening_sweep._forward_detail("A", date(2026, 7, 1), 10) == (None, None, None)
    assert await screening_sweep._forward_return("A", date(2026, 7, 1), 10) is None
