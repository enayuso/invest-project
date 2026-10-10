"""Renders the final pipeline state to console text and a saved markdown
report — plain file write, no report-storage system beyond that."""

from __future__ import annotations

from pathlib import Path

from app.execution.ledger import get_position
from app.state import (
    PipelineState,
    render_portfolio_decision,
    render_research_plan,
    render_trader_proposal,
)

REPORTS_DIR = Path.home() / ".invest-project" / "reports"


async def render(state: PipelineState) -> str:
    parts = [f"# Trading Decision — {state.ticker} ({state.trade_date.isoformat()})", "", "## Analyst Reports"]
    for name, text in state.reports.items():
        parts += [f"### {name}", text, ""]

    parts += ["## Bull/Bear Debate", state.research_debate.transcript(), ""]
    if state.research_plan:
        parts += ["## Research Manager", render_research_plan(state.research_plan), ""]
    if state.trader_proposal:
        parts += ["## Trader", render_trader_proposal(state.trader_proposal), ""]
    parts += ["## Risk Debate", state.risk_debate.transcript(), ""]
    if state.final_decision:
        parts += ["## Portfolio Manager — Final Decision", render_portfolio_decision(state.final_decision), ""]

    position = await get_position(state.ticker)
    parts += ["## Execution — Paper Position", f"Cash: {position.cash:,.2f}  |  Shares: {position.shares}", ""]

    return "\n".join(parts)


async def save(state: PipelineState) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / f"{state.ticker}_{state.trade_date.isoformat()}.md"
    path.write_text(await render(state), encoding="utf-8")
    return path
