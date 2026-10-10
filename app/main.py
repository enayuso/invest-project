"""Manual CLI trigger for the Phase-1 pipeline.

    python -m app.main --ticker AAPL --date 2026-07-08

If --date is omitted, the trade date defaults to the ticker's OWN market
calendar (KST for .KS tickers, US Eastern otherwise) rather than the
server's wall clock — see _main()'s comment for why.
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import date

from app import reporting
from app.clock import local_today
from app.config import settings
from app.execution import risk_rules, runtime_settings
from app.execution.portfolio_risk import ticker_market
from app.orchestrator.pipeline import run_pipeline
from app.web import trade_allowlist_store


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Phase-1 trading-agent pipeline for one ticker.")
    parser.add_argument("--ticker", default=settings.default_ticker)
    parser.add_argument(
        "--date", default=None,
        help="ISO date (YYYY-MM-DD). Defaults to the ticker's own market calendar "
        "(local_today) as of now, not the server's wall-clock date.",
    )
    return parser.parse_args()


async def _main() -> None:
    args = _parse_args()
    # local_today(ticker_market(...)), not get_clock().today() — see
    # app.chat.router's deep_dive branch for the full reasoning. An explicit
    # --date is still honored exactly as given; this only changes the default
    # an operator gets when they omit it.
    # 티커 자신의 시장 달력 사용 — --date 미지정 시에만 적용, 명시 시 그대로 사용.
    trade_date = date.fromisoformat(args.date) if args.date else local_today(ticker_market(args.ticker))
    # This process never runs app.web.state's lifespan, so run_pipeline
    # below — which can actually rebalance/mirror a real trade, not just
    # analyze — would otherwise use whatever risk_rules/runtime_settings/
    # trade_allowlist sit in .env.local/config defaults, NOT whatever the
    # live dashboard has actually been set to. Same class of bug already
    # fixed for the backtest CLI (see app.execution.ledger.rebalance_to_target's
    # own docstring, "Confirmed live 2026-07-24") — a manual `python -m
    # app.main` run diverging from what a human looking at the dashboard
    # believes is configured is exactly that failure mode, one process over.
    await risk_rules.load()
    await runtime_settings.load()
    await trade_allowlist_store.load()
    state = await run_pipeline(args.ticker, trade_date, decision_trigger="manual")
    print(await reporting.render(state))
    path = await reporting.save(state)
    print(f"\nSaved report to {path}")


if __name__ == "__main__":
    asyncio.run(_main())
