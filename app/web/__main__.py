"""CLI for administrative trade_allowlist changes.

    python -m app.web reseed-trade-allowlist --market domestic --tickers 005930.KS,000660.KS
    python -m app.web reseed-trade-allowlist --market overseas --tickers AAPL,MSFT --dry-run
"""

import argparse
import asyncio
import sys

if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from app.config import settings
from app.web import trade_allowlist_store
from app.web.trade_allowlist_admin import reseed_trade_allowlist


async def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Bulk-reseed one market's trade_allowlist through the same validated add/remove path the dashboard uses."
    )
    parser.add_argument("command", choices=["reseed-trade-allowlist"])
    parser.add_argument("--market", required=True, choices=["domestic", "overseas"])
    parser.add_argument("--tickers", required=True, help="comma-separated target list — replaces the market's current allowlist exactly")
    parser.add_argument("--dry-run", action="store_true", help="print the planned add/remove diff without applying it")
    args = parser.parse_args()

    desired = [t.strip() for t in args.tickers.split(",") if t.strip()]

    if args.dry_run:
        await trade_allowlist_store.load()
        current = set(settings.trade_allowlist_for_market(args.market))
        desired_set = set(desired)
        print(f"market: {args.market}")
        print(f"  would add:    {sorted(desired_set - current)}")
        print(f"  would remove: {sorted(current - desired_set)}")
        print(f"  unchanged:    {sorted(current & desired_set)}")
        return

    result = await reseed_trade_allowlist(args.market, desired)
    print(f"market: {result['market']}")
    print(f"  added:    {result['added']}")
    print(f"  removed:  {result['removed']}")
    print(f"  kept:     {result['kept']}")
    if result["rejected"]:
        print("  rejected (left off the list, see reason):")
        for ticker, reason in result["rejected"].items():
            print(f"    {ticker}: {reason}")


if __name__ == "__main__":
    asyncio.run(_main())
