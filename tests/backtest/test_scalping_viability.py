"""app.backtest.scalping_viability — round-trip extraction, exclusion
accounting, clustered vs naive confidence intervals, and the VIABLE/NOT
VIABLE/INCONCLUSIVE verdict logic. Built entirely from hand-crafted
`RoundTrip` fixtures (and a few `DayResult`/`ScalpingTrade` fixtures for the
extraction test) — no cache/DB needed, per this module's own "pure
functions" docstring."""

from __future__ import annotations

import random
import statistics
from datetime import date, timedelta

import pytest

from app.backtest.fill_simulation import SimulatedFill
from app.backtest.scalping_harness import DayResult, ScalpingTrade
from app.backtest.scalping_viability import (
    RoundTrip,
    Verdict,
    VIABLE_MIN_TICKER_DAYS,
    VIABLE_MIN_TRADES,
    _max_drawdown_bp,
    bootstrap_ci_clustered,
    bootstrap_ci_naive,
    build_report,
    extract_round_trips,
    split_excluded,
)
from app.config import settings


def _fill(side: str, executed_price: float, filled_time: str = "093030", gap_inflated: bool = False) -> SimulatedFill:
    return SimulatedFill(
        signal_time="093000",
        requested_latency_seconds=30,
        filled_time=filled_time,
        fill_price=executed_price,
        slippage_bps=0.0,
        side=side,
        executed_price=executed_price,
        realized_latency_seconds=30,
        gap_inflated=gap_inflated,
    )


def _rt(
    ticker: str, trade_date: date, net_bp: float, gap_inflated: bool = False, entry_filled_time: str = "093000",
) -> RoundTrip:
    return RoundTrip(
        ticker=ticker, trade_date=trade_date, entry_price=100.0,
        exit_price=100.0 * (1 + net_bp / 10_000), net_bp=net_bp,
        gap_inflated=gap_inflated, holding_minutes=5.0,
        entry_filled_time=entry_filled_time,
    )


# --- extraction ------------------------------------------------------------


def test_extract_round_trips_matches_compute_return_pct_per_leg(monkeypatch):
    monkeypatch.setattr(settings, "transaction_cost_bps", 5.0)
    monkeypatch.setattr(settings, "sell_tax_bps_domestic", 20.0)

    trades = [
        ScalpingTrade(side="buy", signal_time="093000", fill=_fill("buy", 100.0)),
        ScalpingTrade(side="sell", signal_time="093100", fill=_fill("sell", 100.0, filled_time="093130")),
    ]
    day_result = DayResult(ticker="005930.KS", trade_date=date(2026, 7, 1), trades=trades, return_pct=None)

    round_trips = extract_round_trips(day_result)
    assert len(round_trips) == 1
    rt = round_trips[0]

    buy_fee_rate = 5.0 / 10_000
    sell_fee_rate = 25.0 / 10_000  # commission + domestic sell tax, same as scalping_harness
    expected_mult = (100.0 * (1 - sell_fee_rate)) / (100.0 * (1 + buy_fee_rate))
    assert rt.net_bp == pytest.approx((expected_mult - 1) * 10_000)
    assert rt.gap_inflated is False


def test_extract_round_trips_flags_gap_inflated_if_either_leg_is():
    trades = [
        ScalpingTrade(side="buy", signal_time="093000", fill=_fill("buy", 100.0, gap_inflated=True)),
        ScalpingTrade(side="sell", signal_time="093100", fill=_fill("sell", 101.0)),
    ]
    day_result = DayResult(ticker="AAPL", trade_date=date(2026, 7, 1), trades=trades, return_pct=None)

    round_trips = extract_round_trips(day_result)
    assert round_trips[0].gap_inflated is True


def test_extract_round_trips_populates_entry_filled_time_from_the_buy_legs_fill():
    trades = [
        ScalpingTrade(side="buy", signal_time="093000", fill=_fill("buy", 100.0, filled_time="093030")),
        ScalpingTrade(side="sell", signal_time="093100", fill=_fill("sell", 101.0, filled_time="093130")),
    ]
    day_result = DayResult(ticker="AAPL", trade_date=date(2026, 7, 1), trades=trades, return_pct=None)
    rt = extract_round_trips(day_result)[0]
    assert rt.entry_filled_time == "093030"  # the buy leg's fill time, not its signal_time (093000)


def test_holding_minutes_uses_fill_to_fill_not_signal_to_fill(monkeypatch):
    # 2026-09-11 fix: entry used signal_time (before its own fill latency),
    # overstating every hold by however long the entry itself took to fill.
    monkeypatch.setattr(settings, "transaction_cost_bps", 0.0)
    monkeypatch.setattr(settings, "sell_tax_bps_domestic", 0.0)
    trades = [
        # signal at 09:01:00, but the entry itself didn't fill until 09:02:00
        # (60s of latency) -- the true hold starts at the FILL, not the signal.
        ScalpingTrade(side="buy", signal_time="090100", fill=_fill("buy", 100.0, filled_time="090200")),
        ScalpingTrade(side="sell", signal_time="091200", fill=_fill("sell", 101.0, filled_time="091200")),
    ]
    day_result = DayResult(ticker="AAPL", trade_date=date(2026, 7, 1), trades=trades, return_pct=None)
    rt = extract_round_trips(day_result)[0]
    # fill-to-fill: 09:12:00 - 09:02:00 = 10.0 minutes (NOT 11.0, which the
    # old signal-to-fill definition, 09:12:00 - 09:01:00, would have given).
    assert rt.holding_minutes == pytest.approx(10.0)


def test_holding_minutes_zero_latency_entry_matches_either_definition(monkeypatch):
    monkeypatch.setattr(settings, "transaction_cost_bps", 0.0)
    monkeypatch.setattr(settings, "sell_tax_bps_domestic", 0.0)
    trades = [
        ScalpingTrade(side="buy", signal_time="090100", fill=_fill("buy", 100.0, filled_time="090100")),
        ScalpingTrade(side="sell", signal_time="091100", fill=_fill("sell", 101.0, filled_time="091100")),
    ]
    day_result = DayResult(ticker="AAPL", trade_date=date(2026, 7, 1), trades=trades, return_pct=None)
    rt = extract_round_trips(day_result)[0]
    assert rt.holding_minutes == pytest.approx(10.0)


# --- max drawdown ordering -----------------------------------------------


def test_max_drawdown_uses_entry_fill_time_not_holding_duration():
    # Same trade_date, three round trips whose TRUE chronological order
    # (by entry_filled_time) differs from a holding-duration sort:
    #   A: enters 09:15, held  5 min, +500bp
    #   B: enters 10:00, held 120 min, -1000bp
    #   C: enters 14:00, held  10 min, +500bp
    # True order A,B,C. A holding-duration sort (old bug) would put C before
    # B (10 &lt; 120), giving a different, wrong equity path.
    d = date(2026, 7, 1)
    a = _rt("005930.KS", d, 500.0, entry_filled_time="091500")
    b = _rt("005930.KS", d, -1000.0, entry_filled_time="100000")
    c = _rt("005930.KS", d, 500.0, entry_filled_time="140000")

    true_order_dd = _max_drawdown_bp([a, b, c])
    # Invariant to input list order/shuffling -- the function must sort itself.
    assert _max_drawdown_bp([c, a, b]) == pytest.approx(true_order_dd)
    assert _max_drawdown_bp([b, c, a]) == pytest.approx(true_order_dd)

    # Compute the TRUE chronological equity path by hand: A -> peak 1.05,
    # B -> equity 1.05*0.9=0.945, drawdown=(1.05-0.945)/1.05=0.10 -> 1000bp.
    # C doesn't raise the peak past 1.05*1.05=1.1025... actually recompute:
    equity = 1.0
    peak = 1.0
    expected_dd = 0.0
    for rt in (a, b, c):
        equity *= 1 + rt.net_bp / 10_000
        peak = max(peak, equity)
        expected_dd = max(expected_dd, (peak - equity) / peak)
    assert true_order_dd == pytest.approx(expected_dd * 10_000)


# --- exclusion ---------------------------------------------------------


def test_gap_inflated_legs_are_excluded_and_counted_separately():
    d = date(2026, 7, 1)
    clean = [_rt("005930.KS", d, 50.0) for _ in range(5)]
    flagged = [_rt("005930.KS", d, -500.0, gap_inflated=True) for _ in range(2)]  # extreme outlier if included

    included, excluded = split_excluded(clean + flagged)
    assert len(included) == 5
    assert len(excluded) == 2
    assert all(rt.net_bp == 50.0 for rt in included)  # the flagged outliers never leak into `included`


# --- confidence intervals ---------------------------------------------------


def _make_ticker_days(n_days: int, n_trades_per_day: int, trade_fn) -> list[RoundTrip]:
    round_trips = []
    for day_offset in range(n_days):
        d = date(2026, 7, 1) + timedelta(days=day_offset)
        for i in range(n_trades_per_day):
            round_trips.append(_rt("005930.KS", d, trade_fn(day_offset, i)))
    return round_trips


def test_clustered_bootstrap_ci_lower_bound_is_positive_for_uniformly_positive_trades():
    round_trips = _make_ticker_days(30, 10, lambda d, i: 50.0)
    lo, hi = bootstrap_ci_clustered(round_trips, n_bootstrap=2000, seed=42)
    assert lo > 0
    assert hi > 0


def test_clustered_bootstrap_ci_contains_zero_for_symmetric_noise_around_zero():
    rng = random.Random(7)
    round_trips = _make_ticker_days(60, 8, lambda d, i: rng.uniform(-30, 30))
    lo, hi = bootstrap_ci_clustered(round_trips, n_bootstrap=2000, seed=1)
    assert lo < 0 < hi


def test_clustered_ci_is_wider_than_naive_ci_when_trades_are_correlated_within_days():
    # Each ticker-day shares one common day-level shock (+/-100bp) plus
    # small per-trade noise — trades within a day move together, which is
    # exactly the correlation the clustered bootstrap exists to account for
    # and the naive per-trade CI ignores.
    rng = random.Random(3)
    round_trips = []
    for day_offset in range(40):
        d = date(2026, 7, 1) + timedelta(days=day_offset)
        day_shock = rng.choice([-100.0, 100.0])
        for _ in range(10):
            round_trips.append(_rt("005930.KS", d, day_shock + rng.uniform(-5, 5)))

    naive_lo, naive_hi = bootstrap_ci_naive(round_trips)
    clustered_lo, clustered_hi = bootstrap_ci_clustered(round_trips, n_bootstrap=3000, seed=5)

    naive_width = naive_hi - naive_lo
    clustered_width = clustered_hi - clustered_lo
    assert clustered_width > naive_width


# --- verdict -------------------------------------------------------------


def _good_looking_fixture(n_ticker_days: int, n_trades_per_day: int) -> list[RoundTrip]:
    """All trades at a strongly positive, low-variance net_bp — "looks good"
    on every axis except sample size."""
    round_trips = []
    tickers = [f"T{i}.KS" for i in range(15)]
    for day_offset in range(n_ticker_days):
        d = date(2026, 7, 1) + timedelta(days=day_offset)
        ticker = tickers[day_offset % len(tickers)]
        for _ in range(n_trades_per_day):
            round_trips.append(_rt(ticker, d, 50.0))
    return round_trips


def test_verdict_is_not_viable_when_n_trades_below_minimum():
    # 50 ticker-days * 2 trades/day = 100 trades, well under VIABLE_MIN_TRADES.
    round_trips = _good_looking_fixture(n_ticker_days=50, n_trades_per_day=2)
    assert len(round_trips) < VIABLE_MIN_TRADES

    report = build_report(round_trips, excluded_short_day_count=0, n_tickers_tested=15, n_bootstrap=500, seed=1)
    assert report.verdict == Verdict.NOT_VIABLE
    assert "trades" in report.verdict_reason


def test_verdict_is_inconclusive_when_gap_excluded_fraction_exceeds_10pct_regardless_of_ci():
    d = date(2026, 7, 1)
    tickers = [f"T{i}.KS" for i in range(15)]
    included = []
    for i, ticker in enumerate(tickers * 40):  # plenty of trades/ticker-days to otherwise pass the CI check
        included.append(_rt(ticker, date(2026, 7, 1) + timedelta(days=i % 20), 50.0))
    gap_excluded = [_rt(tickers[0], d, 50.0, gap_inflated=True) for _ in range(len(included) // 5)]  # >10%

    report = build_report(
        included + gap_excluded, excluded_short_day_count=0, n_tickers_tested=15, n_bootstrap=500, seed=1,
    )
    assert report.verdict == Verdict.INCONCLUSIVE
    assert "gap" in report.verdict_reason
