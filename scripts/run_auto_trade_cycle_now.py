"""Dev-only utility: manually trigger one real auto_trade_loop cycle right
now instead of waiting for the scheduled interval — for testing SEMI_AUTO/
FULL_AUTO end to end (real pipeline, real execution_mode branching) without
waiting for auto_trade_interval_minutes or for the market to still be open
when the schedule happens to fire.

Respects market hours exactly like the real loop does
(app.live.market_hours) by default — a closed market contributes no
candidates from that side, same as production. Pass --force to bypass that
check (this process only — the running dashboard's own market-hours checks
are untouched) when you want to test outside trading hours. --force only
gets you through OUR gate, though: KIS's own servers still enforce real
trading-session windows, so a SEMI_AUTO approval you actually confirm may
still be rejected at submission time if the market is genuinely closed —
this is for exercising the AI judgment + approval-popup path, not for
proving a real fill outside market hours.

Only the per-side candidate count is overridden (default 3 for whichever
side is open, via the COUNT arg) rather than production's
auto_trade_max_tickers_domestic/_overseas split (3/2) — this only affects
THIS process's in-memory settings, not the running dashboard's.

This runs the REAL multi-agent pipeline for every candidate found — not a
stub like seed_test_pending_order.py. Expect real vLLM cost/latency
(seconds to minutes per ticker), and in SEMI_AUTO a real approval-modal
popup with a genuine AI-produced recommendation once a candidate resolves
to a Buy/Sell/Overweight/Underweight rating.

Usage: python -m scripts.run_auto_trade_cycle_now [COUNT] [--force]
Must run in the same environment (.env / DB) as the actual dashboard.
"""

from __future__ import annotations

import argparse
import asyncio
import logging

import app.live.auto_trade_loop as auto_trade_loop
from app.brokers import get_broker
from app.clock import local_today
from app.config import settings
from app.execution.portfolio_risk import ticker_market
from app.orchestrator.pipeline import run_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("count", nargs="?", type=int, default=3)
    parser.add_argument(
        "--force", action="store_true",
        help="bypass the market-hours check for candidate discovery (this process only)",
    )
    args = parser.parse_args()

    settings.auto_trade_max_tickers_domestic = args.count
    settings.auto_trade_max_tickers_overseas = args.count

    if args.force:
        # Patched on the auto_trade_loop module itself, not app.live.market_hours
        # — that module did `from ... import is_domestic_market_open, ...`, so
        # the name it actually calls lives in its own namespace.
        auto_trade_loop.is_domestic_market_open = lambda: True
        auto_trade_loop.is_overseas_market_open = lambda: True
        print("--force: bypassing this app's market-hours check (KIS itself may still refuse")
        print("an actual order submission outside real trading hours)")

    broker = await get_broker("kis")
    tickers: list[str] = []

    if auto_trade_loop.is_domestic_market_open():
        domestic = await auto_trade_loop._domestic_candidates(broker)
        print(f"domestic candidates: {domestic or '(none passed screening)'}")
        tickers += domestic
    else:
        print("domestic market closed - skipping domestic candidates")

    if auto_trade_loop.is_overseas_market_open():
        overseas = await auto_trade_loop._overseas_candidates(broker)
        print(f"overseas candidates: {overseas or '(none passed screening)'}")
        tickers += overseas
    else:
        print("overseas market closed - skipping overseas candidates")

    if not tickers:
        print("no candidates this run (both markets closed, or nothing passed screening)")
        return

    print(f"execution_mode={settings.execution_mode.value} - running the full pipeline for: {tickers}")
    for ticker in tickers:
        print(f"--- {ticker} ---")
        try:
            # local_today(ticker_market(...)), not a single date computed once
            # for the whole batch — see app.chat.router's deep_dive branch for
            # the full reasoning. This loop mixes domestic and overseas
            # candidates in one run, so a shared date is wrong for whichever
            # side isn't KST-native.
            # 티커별 시장 달력 사용 — 배치 전체에 날짜 하나를 재사용하면 안 됨.
            today = local_today(ticker_market(ticker))
            state = await run_pipeline(ticker, today, decision_trigger="manual")
            if state.final_decision is None:
                print(f"{ticker}: no decision produced")
            else:
                print(f"{ticker}: rating={state.final_decision.rating.value}")
        except Exception as exc:
            print(f"{ticker}: FAILED - {exc}")

    print("If any ticker resulted in a Buy/Sell/Overweight/Underweight rating and")
    print("execution_mode=SEMI_AUTO, check the dashboard's Pending-Approval popup")
    print("(or GET /api/pending-orders) for a real AI-produced recommendation.")


if __name__ == "__main__":
    asyncio.run(main())
