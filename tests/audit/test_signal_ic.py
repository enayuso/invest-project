"""app.audit.signal_ic — LLM-free, _scored_candidates/_forward_return
faked, no network (Qlib PoC: see qlib-poc-decision-2026-08-21 project
memory)."""

from __future__ import annotations

from datetime import date

import pytest

from app.audit import signal_ic
from app.chat.screening import MomentumScore


def _score(raw_pct: float, atr_pct: float = 0.02) -> MomentumScore:
    return MomentumScore(raw_pct=raw_pct, atr_pct=atr_pct, score=raw_pct / atr_pct)


async def test_daily_ic_is_none_when_no_candidates_scored(monkeypatch):
    async def fake_scored(universe, lookback_days, as_of=None):
        return {}

    monkeypatch.setattr(signal_ic, "_scored_candidates", fake_scored)
    result = await signal_ic._daily_ic(["AAPL"], date(2026, 7, 1), 10, 10)
    assert result == signal_ic.DailyIC(date(2026, 7, 1), 0, None, None)


async def test_daily_ic_below_min_cross_section_returns_none(monkeypatch):
    async def fake_scored(universe, lookback_days, as_of=None):
        return {t: _score(0.01 * i) for i, t in enumerate(universe)}

    async def fake_forward(ticker, trade_date, holding_days):
        return 0.01

    # Only 3 tickers < _MIN_CROSS_SECTION (5) — even with usable forward
    # returns for all of them, the cross-section is too thin to trust.
    monkeypatch.setattr(signal_ic, "_scored_candidates", fake_scored)
    monkeypatch.setattr(signal_ic, "_forward_return", fake_forward)
    result = await signal_ic._daily_ic(["A", "B", "C"], date(2026, 7, 1), 10, 10)
    assert result.rank_ic is None
    assert result.n == 3


async def test_daily_ic_drops_tickers_with_no_forward_return(monkeypatch):
    universe = ["A", "B", "C", "D", "E", "F"]

    async def fake_scored(universe, lookback_days, as_of=None):
        return {t: _score(0.01 * i) for i, t in enumerate(universe)}

    async def fake_forward(ticker, trade_date, holding_days):
        return None if ticker == "F" else 0.01 * universe.index(ticker)  # perfectly monotonic with score

    monkeypatch.setattr(signal_ic, "_scored_candidates", fake_scored)
    monkeypatch.setattr(signal_ic, "_forward_return", fake_forward)
    result = await signal_ic._daily_ic(universe, date(2026, 7, 1), 10, 10)
    assert result.n == 5  # F dropped, not counted


async def test_daily_ic_detects_a_perfect_positive_rank_relationship(monkeypatch):
    universe = ["A", "B", "C", "D", "E", "F"]

    async def fake_scored(universe, lookback_days, as_of=None):
        # Deliberately non-linear in raw_pct so Pearson IC and RankIC can
        # differ, but monotonic — RankIC must still read as a perfect 1.0.
        return {t: _score(0.01 * (i**2)) for i, t in enumerate(universe)}

    async def fake_forward(ticker, trade_date, holding_days):
        return 0.01 * universe.index(ticker)  # strictly increasing with rank

    monkeypatch.setattr(signal_ic, "_scored_candidates", fake_scored)
    monkeypatch.setattr(signal_ic, "_forward_return", fake_forward)
    result = await signal_ic._daily_ic(universe, date(2026, 7, 1), 10, 10)
    assert result.rank_ic == pytest.approx(1.0)


async def test_daily_ic_detects_a_perfect_negative_rank_relationship(monkeypatch):
    universe = ["A", "B", "C", "D", "E", "F"]

    async def fake_scored(universe, lookback_days, as_of=None):
        return {t: _score(0.01 * i) for i, t in enumerate(universe)}

    async def fake_forward(ticker, trade_date, holding_days):
        return -0.01 * universe.index(ticker)  # inversely ranked vs score

    monkeypatch.setattr(signal_ic, "_scored_candidates", fake_scored)
    monkeypatch.setattr(signal_ic, "_forward_return", fake_forward)
    result = await signal_ic._daily_ic(universe, date(2026, 7, 1), 10, 10)
    assert result.rank_ic == pytest.approx(-1.0)


async def test_daily_ic_uses_raw_pct_when_volatility_adjustment_disabled(monkeypatch):
    universe = ["A", "B", "C", "D", "E", "F"]

    async def fake_scored(universe, lookback_days, as_of=None):
        # score (raw/atr) ranks oppositely to raw_pct alone, by construction.
        return {
            "A": MomentumScore(raw_pct=0.05, atr_pct=0.50, score=0.10),
            "B": MomentumScore(raw_pct=0.04, atr_pct=0.20, score=0.20),
            "C": MomentumScore(raw_pct=0.03, atr_pct=0.10, score=0.30),
            "D": MomentumScore(raw_pct=0.02, atr_pct=0.05, score=0.40),
            "E": MomentumScore(raw_pct=0.01, atr_pct=0.02, score=0.50),
            "F": MomentumScore(raw_pct=0.00, atr_pct=0.01, score=0.60),
        }

    async def fake_forward(ticker, trade_date, holding_days):
        # Forward return increases with raw_pct rank (A best), the opposite
        # of `score`'s rank (F best) — so raw_pct should show +1.0 RankIC
        # and score should show -1.0.
        return {"A": 0.05, "B": 0.04, "C": 0.03, "D": 0.02, "E": 0.01, "F": 0.00}[ticker]

    monkeypatch.setattr(signal_ic, "_scored_candidates", fake_scored)
    monkeypatch.setattr(signal_ic, "_forward_return", fake_forward)

    raw_result = await signal_ic._daily_ic(universe, date(2026, 7, 1), 10, 10, use_volatility_adjustment=False)
    adjusted_result = await signal_ic._daily_ic(universe, date(2026, 7, 1), 10, 10, use_volatility_adjustment=True)
    assert raw_result.rank_ic == pytest.approx(1.0)
    assert adjusted_result.rank_ic == pytest.approx(-1.0)


def test_spearman_returns_none_for_zero_variance_input():
    # Every candidate scored identically -- Spearman is undefined here, not
    # 0.0: a degenerate 0.0 would silently drag summarize()'s average
    # toward zero instead of being excluded like a thin cross-section is.
    assert signal_ic._spearman([0.5, 0.5, 0.5, 0.5], [0.1, 0.2, 0.3, 0.4]) is None


def test_summarize_excludes_zero_variance_dates_from_the_average_like_thin_ones():
    zero_variance_date = signal_ic.DailyIC(date(2026, 7, 3), 10, 0.2, None)
    daily_with_degenerate_date = [
        signal_ic.DailyIC(date(2026, 7, 1), 10, 0.2, 0.3),
        signal_ic.DailyIC(date(2026, 7, 2), 10, 0.0, 0.1),
        zero_variance_date,
    ]
    daily_without_it = daily_with_degenerate_date[:2]

    with_degenerate = signal_ic.summarize(daily_with_degenerate_date)
    without_it = signal_ic.summarize(daily_without_it)

    assert with_degenerate.n_dates_scored == 2
    assert with_degenerate.mean_rank_ic == pytest.approx(without_it.mean_rank_ic)
    assert with_degenerate.rank_icir == pytest.approx(without_it.rank_icir)


def test_summarize_excludes_thin_cross_section_dates_from_the_average():
    daily = [
        signal_ic.DailyIC(date(2026, 7, 1), 10, 0.2, 0.3),
        signal_ic.DailyIC(date(2026, 7, 2), 10, 0.0, 0.1),
        signal_ic.DailyIC(date(2026, 7, 3), 2, None, None),  # too thin, excluded
    ]
    summary = signal_ic.summarize(daily)
    assert summary.n_dates_scored == 2
    assert summary.mean_ic == pytest.approx(0.1)
    assert summary.mean_rank_ic == pytest.approx(0.2)


def test_summarize_computes_rank_icir_as_mean_over_std():
    daily = [
        signal_ic.DailyIC(date(2026, 7, 1), 10, 0.1, 0.4),
        signal_ic.DailyIC(date(2026, 7, 2), 10, 0.1, 0.2),
        signal_ic.DailyIC(date(2026, 7, 3), 10, 0.1, 0.3),
    ]
    summary = signal_ic.summarize(daily)
    import statistics

    expected_mean = statistics.mean([0.4, 0.2, 0.3])
    expected_std = statistics.stdev([0.4, 0.2, 0.3])
    assert summary.rank_icir == pytest.approx(expected_mean / expected_std)


def test_summarize_rank_icir_is_none_when_ic_is_constant_across_dates():
    # std == 0 -- a "consistent" edge that's actually a degenerate,
    # data-starved case (e.g. every date landed the same handful of
    # tickers) -- must not raise a ZeroDivisionError or report infinity.
    daily = [
        signal_ic.DailyIC(date(2026, 7, 1), 10, 0.1, 0.2),
        signal_ic.DailyIC(date(2026, 7, 2), 10, 0.1, 0.2),
    ]
    summary = signal_ic.summarize(daily)
    assert summary.rank_icir is None


def test_summarize_is_all_none_when_every_date_was_too_thin():
    daily = [signal_ic.DailyIC(date(2026, 7, 1), 2, None, None)]
    summary = signal_ic.summarize(daily)
    assert summary.n_dates_scored == 0
    assert summary.mean_ic is None
    assert summary.mean_rank_ic is None
    assert summary.rank_icir is None


async def test_measure_signal_ic_summarizes_across_all_trade_dates(monkeypatch):
    universe = ["A", "B", "C", "D", "E", "F"]
    dates = [date(2026, 7, 1), date(2026, 7, 8)]

    async def fake_scored(universe, lookback_days, as_of=None):
        return {t: _score(0.01 * i) for i, t in enumerate(universe)}

    async def fake_forward(ticker, trade_date, holding_days):
        return 0.01 * universe.index(ticker)

    monkeypatch.setattr(signal_ic, "_scored_candidates", fake_scored)
    monkeypatch.setattr(signal_ic, "_forward_return", fake_forward)
    summary = await signal_ic.measure_signal_ic(universe, dates, lookback_days=10, holding_days=10)
    assert summary.n_dates_scored == 2
    assert summary.mean_rank_ic == pytest.approx(1.0)
