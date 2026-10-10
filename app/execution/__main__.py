"""CLI for manual stop-loss checks and account-state maintenance.

    python -m app.execution check-stops --ticker AAPL
    python -m app.execution reset-account-state --market domestic
"""

import argparse
import asyncio
import sys
if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
from app.execution import risk_rules, runtime_settings
from app.execution.ledger import check_stops_for_ticker
from app.execution.portfolio_risk import reset_high_water_mark
from app.execution.safety_switch import reset_daily_loss_baseline
from app.web import trade_allowlist_store


async def _main() -> None:
    parser = argparse.ArgumentParser(description="Check a live position's stop-loss, or reset account-derived safety state.")
    parser.add_argument("command", choices=["check-stops", "reset-account-state"])
    parser.add_argument("--ticker", required=False)
    parser.add_argument("--market", required=False, choices=["domestic", "overseas"])
    args = parser.parse_args()

    # This process never runs app.web.state's lifespan, so without these it
    # reads whatever risk_rules/runtime_settings/trade_allowlist sit in
    # .env.local/config defaults, NOT whatever the live dashboard has
    # actually been set to — same class of bug already fixed for the
    # backtest CLI (see app.execution.ledger.rebalance_to_target's own
    # docstring, "Confirmed live 2026-07-24"): a stray CLI process reading
    # stale config diverges from what a human looking at the dashboard
    # believes is configured. check-stops in particular re-checks a live
    # stop-loss, which should honor whatever risk thresholds are actually
    # active right now, not whatever .env.local happened to say at deploy
    # time.
    await risk_rules.load()
    await runtime_settings.load()
    await trade_allowlist_store.load()

    if args.command == "check-stops":
        if not args.ticker:
            parser.error("check-stops requires --ticker")
        print(await check_stops_for_ticker(args.ticker))
    else:
        if not args.market:
            parser.error("reset-account-state requires --market")
        # Run right after manually resetting the KIS paper account balance —
        # clears the persisted drawdown high-water mark (which never lowers
        # on its own and would otherwise block all new buys against the
        # pre-reset high forever) and today's daily-loss baseline.
        # 모의계좌를 실제로 리셋한 직후에만 실행 — 자동 실행 금지.
        await reset_high_water_mark(args.market)
        await reset_daily_loss_baseline(args.market)
        print(f"{args.market}: high-water mark and daily-loss baseline reset.")


if __name__ == "__main__":
    asyncio.run(_main())