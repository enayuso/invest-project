"""Renders a BacktestResult to a readable summary.

The primary/secondary risk-metrics split below (real ledger vs. exposure
proxy) mirrors the GIPS (Global Investment Performance Standards, the
standard most asset managers report performance under) requirement that
simulated/model/hypothetical performance be clearly and separately
disclosed from actual results, never blended into one figure.
"""

from __future__ import annotations

from app.backtest.metrics import BacktestResult, RiskMetrics
from app.backtest.portfolio_harness import PortfolioBacktestResult
from app.execution.ledger import Position, get_position, starting_cash_for


async def ledger_snapshot(result: BacktestResult) -> tuple[Position, float, float | None]:
    """(position, starting_cash, ledger_return) — the REAL, executed
    result: actual ATR-scaled position sizing and stop-loss exits, as
    opposed to `result.strategy_return`'s exposure-proxy signal-quality
    number (see this module's own GIPS-style actual-vs-hypothetical
    disclosure discipline). Callers outside `render` (e.g. `__main__.py`,
    to fill `record_experiment`'s `return_pct` with the actual return
    rather than the proxy) should call this directly instead of
    re-deriving it.

    for_backtest=True on BOTH calls below, not the ambient
    is_backtest_context(): every caller of this module renders/records a
    replay's result, but all of them (backtest/__main__.py,
    portfolio_harness._main) run AFTER the harness's backtest_scope() has
    already exited. Left implicit, get_position would read the LIVE/real
    account's position row for this ticker (not the replay's), and
    starting_cash_for would divide by the LIVE starting cash — 100,000 vs
    146 under .env.dev — producing a nonsensical +193,561%-style figure.

    `ledger_return` is None if starting_cash is 0 (division would be
    undefined) — not expected in practice, but guarded rather than raising."""
    position = await get_position(result.ticker, for_backtest=True)
    last_close = result.records[-1].close
    final_equity = position.market_value(last_close)
    starting_cash = await starting_cash_for(result.ticker, for_backtest=True)
    ledger_return = (final_equity - starting_cash) / starting_cash if starting_cash else None
    return position, starting_cash, ledger_return


def _risk_lines(rm: RiskMetrics) -> list[str]:
    if rm.is_empty():
        return ["(Insufficient data to compute risk metrics — need at least 2 periods.)"]
    return [
        f"Sharpe ratio:  {rm.sharpe:+.3f}" if rm.sharpe is not None else "Sharpe ratio:  N/A",
        f"Sortino ratio: {rm.sortino:+.3f}" if rm.sortino is not None else "Sortino ratio: N/A",
        f"Max drawdown:  {rm.max_drawdown:+.2%}" if rm.max_drawdown is not None else "Max drawdown:  N/A",
        f"CVaR (95%):    {rm.cvar_95:+.2%}" if rm.cvar_95 is not None else "CVaR (95%):    N/A",
    ]


async def render(result: BacktestResult) -> str:
    lines = [f"# Backtest — {result.ticker} vs {result.benchmark_ticker}", "", "## Decisions"]
    for record in result.records:
        lines.append(f"- {record.trade_date.isoformat()}: {record.decision.rating.value} (close={record.close:.2f})")

    lines += [
        "",
        "## Comparison",
        f"Strategy return (exposure proxy): {result.strategy_return:+.2%}",
        f"Buy-and-hold ({result.ticker}):          {result.buy_and_hold_return:+.2%}",
        f"Benchmark ({result.benchmark_ticker}) buy-and-hold:     {result.benchmark_return:+.2%}",
    ]

    lines += [
        "",
        "## Risk Metrics — Actual (Real Ledger) (primary; per decision period, not "
        "annualized; reflects actual ATR-scaled, risk-capped position sizing and "
        "stop-loss exits)",
    ]
    lines += _risk_lines(result.ledger_risk_metrics)

    lines += [
        "",
        "## Risk Metrics — Hypothetical (Exposure Proxy) (GIPS-style supplemental "
        "disclosure only; ignores real position sizing and stop-loss — signal "
        "quality, not an executed strategy's risk)",
    ]
    lines += _risk_lines(result.exposure_risk_metrics)

    position, starting_cash, ledger_return = await ledger_snapshot(result)
    last_close = result.records[-1].close
    lines += [
        "",
        "## Execution — Paper Ledger",
        f"Starting cash: {starting_cash:,.2f}",
        f"Final position: cash={position.cash:,.2f}  shares={position.shares}  "
        f"(marked at {last_close:.2f})",
        "Ledger return (real position sizing + stop-loss): "
        + (f"{ledger_return:+.2%}" if ledger_return is not None else "N/A"),
    ]

    return "\n".join(lines)


async def render_portfolio(result: PortfolioBacktestResult) -> str:
    """Multi-ticker counterpart to `render` — an aggregate portfolio-level
    section (the new part: equity curve, risk-capped return, which of the
    concentration/correlation-bucket/drawdown caps actually fired) followed
    by each ticker's own full `render()` report unchanged, so per-ticker
    detail isn't lost inside the aggregate view. Same GIPS-style actual-vs-
    hypothetical disclosure discipline as `render` throughout."""
    lines = [
        f"# Portfolio Backtest — {', '.join(result.tickers)} vs {result.benchmark_ticker}",
        "",
        f"Trade dates: {result.trade_dates[0].isoformat()} to {result.trade_dates[-1].isoformat()} "
        f"({len(result.trade_dates)} decision dates, {result.n_resolved_decisions} resolved decisions "
        f"across all tickers)",
        "",
        "## Portfolio-Level Comparison",
        f"Starting aggregate equity (USD-equivalent): {result.starting_total_equity_usd:,.2f}",
        f"Final aggregate equity (USD-equivalent):    {result.final_total_equity_usd:,.2f}",
        # "net of transaction costs" is accurate (commission + domestic sell
        # tax ARE charged, app.execution.costs) but reads as "net of trading
        # friction" — this fill assumes exact-close execution, zero
        # slippage, on every leg. 2026-09-08 fee/tax/slippage audit found
        # that gap undocumented here specifically; see app.audit.
        # slippage_validation for the (separately reported) real-fill-vs-
        # ledger-price gap this return does NOT include.
        # "수수료/세금 반영"은 사실이나 슬리피지(체결가 괴리)는 반영 안 됨 — 종가 정확체결 가정.
        f"Portfolio return (real ledger, risk-capped, net of commission/tax, gross of slippage): "
        f"{result.portfolio_return:+.2%}",
        f"Benchmark ({result.benchmark_ticker}) buy-and-hold:                     "
        f"{result.portfolio_benchmark_return:+.2%}",
        f"Win rate (direction-aware — Buy/Overweight need alpha>0, Sell/Underweight "
        f"need alpha<0 (succeeded by underperforming); Hold excluded, held/flat "
        f"unknown in a replay): {result.win_rate:.1%} "
        f"(n={result.n_scored_decisions}, unscored Holds={result.n_unscored_holds})"
        if result.win_rate is not None
        else "Win rate: N/A (no resolved decisions)",
        f"Mean alpha per decision: {result.mean_alpha_return:+.2%}"
        if result.mean_alpha_return is not None
        else "Mean alpha per decision: N/A",
    ]

    lines += [
        "",
        "## Portfolio Risk Metrics — Actual (Real Aggregate Ledger) (primary; per decision-date "
        "period across the whole book, not annualized; reflects real per-ticker position sizing, "
        "the per-ticker 1% risk cap, AND the portfolio-level concentration/correlation-bucket/"
        "drawdown caps)",
    ]
    lines += _risk_lines(result.portfolio_risk_metrics)

    lines += ["", "## Portfolio-Level Risk Caps — Trigger Counts (this run)"]
    if result.cap_trigger_counts:
        for label, count in sorted(result.cap_trigger_counts.items(), key=lambda kv: -kv[1]):
            lines.append(f"- {label}: {count} decision(s)")
    else:
        lines.append("(None of the per-ticker or portfolio-level caps reduced any decision during this run.)")

    lines += ["", "---", "", "# Per-Ticker Detail", ""]
    never_screened_in = [t for t in result.tickers if t not in result.per_ticker_results]
    if never_screened_in:
        # Reachable when run_portfolio_backtest was given a candidate_selector
        # (e.g. screening_candidate_selector) that never once picked one of
        # `tickers` across the whole window — no records to render, so no
        # silent gap: named explicitly rather than just missing from the list.
        lines.append(
            f"Never screened in on any date (no records to report): {', '.join(never_screened_in)}"
        )
        lines.append("")
    for ticker in result.tickers:
        if ticker not in result.per_ticker_results:
            continue
        lines.append(await render(result.per_ticker_results[ticker]))
        lines.append("")

    return "\n".join(lines)