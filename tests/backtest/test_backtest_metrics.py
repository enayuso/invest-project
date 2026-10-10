"""Backtest metrics — pure numeric functions (A-stage, no network/DB)."""

from __future__ import annotations

from datetime import date

import pytest

from app.backtest.metrics import (
    BacktestRecord,
    RiskMetrics,
    _mean,
    annualized_sharpe,
    compute_buy_and_hold_return,
    compute_ledger_period_returns,
    compute_period_returns,
    compute_strategy_return,
    cvar,
    max_drawdown,
    price_on_or_before,
    sharpe_ratio,
    sortino_ratio,
)
from app.state import PortfolioDecision, PortfolioRating


def _decision(rating: PortfolioRating) -> PortfolioDecision:
    return PortfolioDecision(rating=rating, executive_summary="s", investment_thesis="t")


def _record(d: date, rating: PortfolioRating, close: float) -> BacktestRecord:
    return BacktestRecord(trade_date=d, decision=_decision(rating), close=close)


def test_risk_metrics_is_empty_true_when_all_none():
    assert RiskMetrics(None, None, None, None).is_empty() is True


def test_risk_metrics_is_empty_false_when_any_set():
    assert RiskMetrics(1.0, None, None, None).is_empty() is False


def test_compute_period_returns_uses_prior_decision_exposure():
    records = [
        _record(date(2026, 1, 1), PortfolioRating.BUY, 100.0),
        _record(date(2026, 1, 2), PortfolioRating.HOLD, 110.0),
    ]
    # BUY exposure = 1.0 * (110-100)/100 = 0.10
    assert compute_period_returns(records) == pytest.approx([0.10])


def test_compute_period_returns_short_exposure_for_sell():
    records = [
        _record(date(2026, 1, 1), PortfolioRating.SELL, 100.0),
        _record(date(2026, 1, 2), PortfolioRating.HOLD, 90.0),
    ]
    # SELL exposure = -1.0 * (90-100)/100 = 0.10
    assert compute_period_returns(records) == pytest.approx([0.10])


def test_compute_strategy_return_compounds_periods():
    records = [
        _record(date(2026, 1, 1), PortfolioRating.BUY, 100.0),
        _record(date(2026, 1, 2), PortfolioRating.BUY, 110.0),
        _record(date(2026, 1, 3), PortfolioRating.BUY, 121.0),
    ]
    assert compute_strategy_return(records) == pytest.approx(0.21)


def test_compute_ledger_period_returns_skips_zero_equity():
    assert compute_ledger_period_returns([100.0, 0.0, 50.0]) == pytest.approx([-1.0])


def test_price_on_or_before_exact_match():
    prices = {date(2026, 1, 1): 100.0, date(2026, 1, 3): 105.0}
    assert price_on_or_before(prices, date(2026, 1, 1)) == 100.0


def test_price_on_or_before_falls_back_to_earlier_date():
    prices = {date(2026, 1, 1): 100.0, date(2026, 1, 3): 105.0}
    assert price_on_or_before(prices, date(2026, 1, 2)) == 100.0


def test_price_on_or_before_raises_when_no_earlier_price():
    prices = {date(2026, 1, 5): 100.0}
    with pytest.raises(KeyError):
        price_on_or_before(prices, date(2026, 1, 1))


def test_compute_buy_and_hold_return():
    prices = {date(2026, 1, 1): 100.0, date(2026, 1, 10): 120.0}
    result = compute_buy_and_hold_return(prices, date(2026, 1, 1), date(2026, 1, 10))
    assert result == pytest.approx(0.20)


def test_sharpe_ratio_none_when_fewer_than_two_periods():
    assert sharpe_ratio([0.01]) is None


def test_sharpe_ratio_none_when_zero_variance():
    assert sharpe_ratio([0.01, 0.01, 0.01]) is None


def test_sharpe_ratio_computed_for_varying_returns():
    result = sharpe_ratio([0.02, -0.01, 0.03])
    assert result is not None


def test_sortino_ratio_none_when_no_downside_periods():
    assert sortino_ratio([0.01, 0.02, 0.03]) is None


def test_sortino_ratio_computed_when_downside_exists():
    result = sortino_ratio([0.02, -0.01, 0.03])
    assert result is not None


def test_annualized_sharpe_scales_the_period_sharpe_by_sqrt_periods_per_year():
    returns = [0.02, -0.01, 0.03]
    period_sharpe = sharpe_ratio(returns)
    # cadence_days=1 (daily) -> 252 periods/year, the textbook sqrt(252) factor.
    assert annualized_sharpe(returns, cadence_days=1) == pytest.approx(period_sharpe * (252**0.5))


def test_annualized_sharpe_uses_fewer_periods_per_year_for_a_wider_cadence():
    returns = [0.02, -0.01, 0.03]
    period_sharpe = sharpe_ratio(returns)
    # cadence_days=7 (weekly) -> 252/7 periods/year, not the daily factor.
    assert annualized_sharpe(returns, cadence_days=7) == pytest.approx(period_sharpe * ((252 / 7) ** 0.5))


def test_annualized_sharpe_none_when_the_underlying_sharpe_is_none():
    assert annualized_sharpe([0.01], cadence_days=1) is None  # too few periods
    assert annualized_sharpe([0.01, 0.01], cadence_days=1) is None  # zero variance


def test_annualized_sharpe_none_for_a_non_positive_cadence():
    assert annualized_sharpe([0.02, -0.01, 0.03], cadence_days=0) is None


def test_max_drawdown_none_for_empty_returns():
    assert max_drawdown([]) is None


def test_max_drawdown_computes_peak_to_trough():
    # equity: 1.0 -> 1.10 -> 0.99 (peak 1.10, trough 0.99: -10%)
    result = max_drawdown([0.10, -0.10])
    assert result == pytest.approx(-0.10)


def test_cvar_none_when_not_enough_tail_periods():
    assert cvar([], alpha=0.05) is None


def test_cvar_averages_worst_tail():
    # alpha chosen so n_tail = max(1, round(10*0.2)) = 2 -> a genuine
    # 2-period average, not the single-tail-value case (see the
    # n_tail < 2 guard added 2026-09-11 and its own dedicated tests below).
    returns = [0.05, -0.20, 0.01, -0.10, 0.02, 0.03, -0.01, 0.04, -0.02, 0.06]
    assert cvar(returns, alpha=0.2) == pytest.approx((-0.20 + -0.10) / 2)


# --- cvar: n_tail < 2 guard (2026-09-11 fix) ------------------------------
#
# Bug: n_tail = max(1, round(len(returns) * alpha)) is pinned at exactly 1
# for every len(returns) from 1 through 29 at the project's only actually-
# used alpha=0.05 (round()'s banker's-rounding first pushes it to 2 at
# len(returns)==30: round(29*0.05)==round(1.45)==1,
# round(30*0.05)==round(1.5)==2). The old guard (`len(returns) < n_tail`)
# only caught the empty-list case (0 < 1) -- for every realistic small/
# medium backtest length up to 29 periods, "CVaR" silently WAS just
# min(returns), presented under the same label/formatting as a genuine
# multi-period tail average.


def test_cvar_none_at_29_periods_alpha_05_single_tail_value_case():
    # n_tail = max(1, round(29*0.05)) = max(1, round(1.45)) = 1 -- exactly
    # the degenerate "average of one value" case this fix rejects.
    returns = [float(i) / 100 for i in range(29)]
    assert round(29 * 0.05) == 1  # pin the claimed round() boundary
    assert cvar(returns) is None


def test_cvar_real_two_period_average_at_30_periods_alpha_05():
    # n_tail = max(1, round(30*0.05)) = max(1, round(1.5)) = 2 -- the
    # natural threshold where a genuine >=2-period average first appears.
    assert round(30 * 0.05) == 2  # pin the claimed round() boundary
    returns = [-0.30, -0.20] + [0.01] * 28
    assert cvar(returns) == pytest.approx((-0.30 + -0.20) / 2)


def test_cvar_typical_size_unchanged_by_the_guard():
    # 100 returns, alpha=0.05 -> n_tail = 5, already >= 2 pre-fix -- same
    # numeric result as the pre-fix formula, guard is a no-op here.
    returns = sorted(float(i) / 1000 for i in range(-50, 50))
    assert cvar(returns) == pytest.approx(_mean(returns[:5]))
