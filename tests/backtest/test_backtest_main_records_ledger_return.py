"""app.backtest.__main__ — the 2026-09-11 fix for return_pct getting filled
with the exposure-proxy return (BacktestResult.strategy_return) instead of
the real ledger return (app.backtest.reporting.ledger_snapshot), the same
"actual, not hypothetical" rule record_experiment's own docstring states."""

from __future__ import annotations

from datetime import date

import pytest

from app.backtest import __main__ as backtest_main
from app.backtest.metrics import BacktestRecord, BacktestResult, RiskMetrics
from app.execution.ledger import Position
from app.state import PortfolioDecision, PortfolioRating


def _empty_risk() -> RiskMetrics:
    return RiskMetrics(sharpe=None, sortino=None, max_drawdown=None, cvar_95=None)


async def test_main_records_ledger_return_not_the_exposure_proxy(monkeypatch):
    result = BacktestResult(
        ticker="AAPL", benchmark_ticker="SPY",
        records=[BacktestRecord(
            trade_date=date(2026, 7, 15),
            decision=PortfolioDecision(rating=PortfolioRating.BUY, executive_summary="s", investment_thesis="t"),
            close=150.0,
        )],
        strategy_return=0.42, buy_and_hold_return=0.1, benchmark_return=0.05,
        exposure_risk_metrics=_empty_risk(), ledger_risk_metrics=_empty_risk(),
    )

    async def fake_run_backtest(**kwargs):
        return result

    async def fake_render(r):
        return "rendered"

    async def fake_ledger_snapshot(r):
        return Position(cash=0.0, shares=1), 100.0, 0.07

    captured = {}

    async def fake_record_experiment(**kwargs):
        captured.update(kwargs)
        return 1

    monkeypatch.setattr(backtest_main, "run_backtest", fake_run_backtest)
    monkeypatch.setattr(backtest_main.reporting, "render", fake_render)
    monkeypatch.setattr(backtest_main.reporting, "ledger_snapshot", fake_ledger_snapshot)
    monkeypatch.setattr(backtest_main, "record_experiment", fake_record_experiment)
    monkeypatch.setattr(
        backtest_main, "_parse_args",
        lambda: argparse_namespace(
            ticker="AAPL", start="2026-01-01", end="2026-07-15", random=False,
            cadence_days=5, benchmark="SPY", label=None, log_file=None,
        ),
    )

    await backtest_main._main()

    assert captured["return_pct"] == 0.07  # the REAL ledger return, never the 0.42 exposure proxy
    assert captured["exposure_return_pct"] == 0.42


def argparse_namespace(**kwargs):
    import argparse

    return argparse.Namespace(**kwargs)
