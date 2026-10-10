"""app.backtest.reporting — render/render_portfolio, ledger calls faked, no
DB (F-stage)."""

from __future__ import annotations

from datetime import date

import pytest

from app.backtest import reporting
from app.backtest.metrics import BacktestRecord, BacktestResult, RiskMetrics
from app.backtest.portfolio_harness import PortfolioBacktestResult
from app.execution.ledger import Position
from app.state import PortfolioDecision, PortfolioRating


def _decision(rating=PortfolioRating.BUY) -> PortfolioDecision:
    return PortfolioDecision(rating=rating, executive_summary="s", investment_thesis="t")


def _record(d: date, close: float, rating=PortfolioRating.BUY) -> BacktestRecord:
    return BacktestRecord(trade_date=d, decision=_decision(rating), close=close)


def _empty_risk() -> RiskMetrics:
    return RiskMetrics(sharpe=None, sortino=None, max_drawdown=None, cvar_95=None)


def _full_risk() -> RiskMetrics:
    return RiskMetrics(sharpe=1.2, sortino=1.5, max_drawdown=-0.1, cvar_95=-0.05)


@pytest.fixture(autouse=True)
def _stub_ledger(monkeypatch):
    captured: dict = {}

    async def fake_get_position(ticker, *, for_backtest=None):
        captured["get_position_for_backtest"] = for_backtest
        return Position(cash=500.0, shares=10)

    async def fake_starting_cash_for(ticker, *, for_backtest=None):
        captured["starting_cash_for_for_backtest"] = for_backtest
        return 100_000.0

    monkeypatch.setattr(reporting, "get_position", fake_get_position)
    monkeypatch.setattr(reporting, "starting_cash_for", fake_starting_cash_for)
    return captured


async def test_render_includes_decisions_and_comparison_section():
    result = BacktestResult(
        ticker="AAPL", benchmark_ticker="SPY",
        records=[_record(date(2026, 7, 15), 150.0), _record(date(2026, 7, 16), 155.0, PortfolioRating.HOLD)],
        strategy_return=0.05, buy_and_hold_return=0.033, benchmark_return=0.01,
        exposure_risk_metrics=_empty_risk(), ledger_risk_metrics=_full_risk(),
    )
    text = await reporting.render(result)
    assert "# Backtest — AAPL vs SPY" in text
    assert "2026-07-15: Buy (close=150.00)" in text
    assert "2026-07-16: Hold (close=155.00)" in text
    assert "Strategy return (exposure proxy): +5.00%" in text


async def test_render_shows_insufficient_data_message_for_empty_risk_metrics():
    result = BacktestResult(
        ticker="AAPL", benchmark_ticker="SPY", records=[_record(date(2026, 7, 15), 150.0)],
        strategy_return=0.0, buy_and_hold_return=0.0, benchmark_return=0.0,
        exposure_risk_metrics=_empty_risk(), ledger_risk_metrics=_empty_risk(),
    )
    text = await reporting.render(result)
    assert text.count("(Insufficient data to compute risk metrics — need at least 2 periods.)") == 2


async def test_render_computes_ledger_return_against_actual_starting_cash():
    result = BacktestResult(
        ticker="AAPL", benchmark_ticker="SPY", records=[_record(date(2026, 7, 15), 150.0)],
        strategy_return=0.0, buy_and_hold_return=0.0, benchmark_return=0.0,
        exposure_risk_metrics=_empty_risk(), ledger_risk_metrics=_empty_risk(),
    )
    text = await reporting.render(result)
    # position: cash=500, shares=10, marked at 150 -> final_equity = 2000
    # ledger_return = (2000 - 100000) / 100000 = -0.98
    assert "Ledger return (real position sizing + stop-loss): -98.00%" in text


async def test_ledger_snapshot_returns_position_cash_and_return(_stub_ledger):
    # position: cash=500, shares=10, marked at 150 -> final_equity = 2000;
    # starting_cash=100000 -> ledger_return = (2000-100000)/100000 = -0.98
    result = BacktestResult(
        ticker="AAPL", benchmark_ticker="SPY", records=[_record(date(2026, 7, 15), 150.0)],
        strategy_return=0.0, buy_and_hold_return=0.0, benchmark_return=0.0,
        exposure_risk_metrics=_empty_risk(), ledger_risk_metrics=_empty_risk(),
    )
    position, starting_cash, ledger_return = await reporting.ledger_snapshot(result)
    assert position.cash == 500.0
    assert position.shares == 10
    assert starting_cash == 100_000.0
    assert ledger_return == pytest.approx(-0.98)


async def test_ledger_snapshot_returns_none_when_starting_cash_is_zero(monkeypatch):
    async def fake_get_position(ticker, *, for_backtest=None):
        return Position(cash=500.0, shares=10)

    async def fake_starting_cash_for(ticker, *, for_backtest=None):
        return 0.0

    monkeypatch.setattr(reporting, "get_position", fake_get_position)
    monkeypatch.setattr(reporting, "starting_cash_for", fake_starting_cash_for)
    result = BacktestResult(
        ticker="AAPL", benchmark_ticker="SPY", records=[_record(date(2026, 7, 15), 150.0)],
        strategy_return=0.0, buy_and_hold_return=0.0, benchmark_return=0.0,
        exposure_risk_metrics=_empty_risk(), ledger_risk_metrics=_empty_risk(),
    )
    _, starting_cash, ledger_return = await reporting.ledger_snapshot(result)
    assert starting_cash == 0.0
    assert ledger_return is None


async def test_render_reads_the_backtest_book_not_the_live_position(_stub_ledger):
    # reporting.render runs AFTER the harness's backtest_scope() has already
    # exited, so an implicit for_backtest (None -> is_backtest_context()) would
    # resolve to the LIVE account's position row instead of the replay's own.
    result = BacktestResult(
        ticker="AAPL", benchmark_ticker="SPY", records=[_record(date(2026, 7, 15), 150.0)],
        strategy_return=0.0, buy_and_hold_return=0.0, benchmark_return=0.0,
        exposure_risk_metrics=_empty_risk(), ledger_risk_metrics=_empty_risk(),
    )
    await reporting.render(result)
    assert _stub_ledger["get_position_for_backtest"] is True
    assert _stub_ledger["starting_cash_for_for_backtest"] is True


async def test_render_portfolio_includes_aggregate_and_per_ticker_sections():
    per_ticker_result = BacktestResult(
        ticker="AAPL", benchmark_ticker="SPY", records=[_record(date(2026, 7, 15), 150.0)],
        strategy_return=0.02, buy_and_hold_return=0.01, benchmark_return=0.005,
        exposure_risk_metrics=_empty_risk(), ledger_risk_metrics=_empty_risk(),
    )
    portfolio_result = PortfolioBacktestResult(
        tickers=["AAPL"], benchmark_ticker="SPY", trade_dates=[date(2026, 7, 15), date(2026, 7, 16)],
        per_ticker_results={"AAPL": per_ticker_result}, portfolio_equity_curve=[100_000.0, 102_000.0],
        portfolio_risk_metrics=_empty_risk(), starting_total_equity_usd=100_000.0,
        final_total_equity_usd=102_000.0, portfolio_return=0.02, portfolio_benchmark_return=0.01,
        win_rate=0.6, mean_alpha_return=0.015, n_resolved_decisions=5,
        cap_trigger_counts={"portfolio concentration cap": 2},
    )
    text = await reporting.render_portfolio(portfolio_result)
    assert "# Portfolio Backtest — AAPL vs SPY" in text
    assert "Portfolio return (real ledger, risk-capped, net of commission/tax, gross of slippage): +2.00%" in text
    assert "- portfolio concentration cap: 2 decision(s)" in text
    assert "# Per-Ticker Detail" in text
    assert "# Backtest — AAPL vs SPY" in text  # embedded per-ticker render()


async def test_render_portfolio_flags_tickers_never_screened_in():
    # A screened run (candidate_selector given) can leave some `tickers`
    # entries with no per_ticker_results row at all — must be named
    # explicitly, not just silently missing from "# Per-Ticker Detail".
    per_ticker_result = BacktestResult(
        ticker="AAPL", benchmark_ticker="SPY", records=[_record(date(2026, 7, 15), 150.0)],
        strategy_return=0.02, buy_and_hold_return=0.01, benchmark_return=0.005,
        exposure_risk_metrics=_empty_risk(), ledger_risk_metrics=_empty_risk(),
    )
    portfolio_result = PortfolioBacktestResult(
        tickers=["AAPL", "000660.KS"], benchmark_ticker="SPY",
        trade_dates=[date(2026, 7, 15), date(2026, 7, 16)],
        per_ticker_results={"AAPL": per_ticker_result}, portfolio_equity_curve=[100_000.0, 102_000.0],
        portfolio_risk_metrics=_empty_risk(), starting_total_equity_usd=100_000.0,
        final_total_equity_usd=102_000.0, portfolio_return=0.02, portfolio_benchmark_return=0.01,
        win_rate=0.6, mean_alpha_return=0.015, n_resolved_decisions=5, cap_trigger_counts={},
    )
    text = await reporting.render_portfolio(portfolio_result)
    assert "Never screened in on any date (no records to report): 000660.KS" in text
    assert "# Backtest — AAPL vs SPY" in text  # AAPL's own detail still renders fine


async def test_render_portfolio_shows_no_caps_triggered_message_when_empty():
    per_ticker_result = BacktestResult(
        ticker="AAPL", benchmark_ticker="SPY", records=[_record(date(2026, 7, 15), 150.0)],
        strategy_return=0.0, buy_and_hold_return=0.0, benchmark_return=0.0,
        exposure_risk_metrics=_empty_risk(), ledger_risk_metrics=_empty_risk(),
    )
    portfolio_result = PortfolioBacktestResult(
        tickers=["AAPL"], benchmark_ticker="SPY", trade_dates=[date(2026, 7, 15), date(2026, 7, 16)],
        per_ticker_results={"AAPL": per_ticker_result}, portfolio_equity_curve=[100_000.0],
        portfolio_risk_metrics=_empty_risk(), starting_total_equity_usd=100_000.0,
        final_total_equity_usd=100_000.0, portfolio_return=0.0, portfolio_benchmark_return=0.0,
        win_rate=None, mean_alpha_return=None, n_resolved_decisions=0, cap_trigger_counts={},
    )
    text = await reporting.render_portfolio(portfolio_result)
    assert "(None of the per-ticker or portfolio-level caps reduced any decision during this run.)" in text
    assert "Win rate: N/A (no resolved decisions)" in text
