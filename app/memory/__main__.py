"""Manual CLI for resolving pending memory entries for live (non-backtest)
runs, scoped to a single ticker.

As of 2026-08-03, this is no longer the only way pending live decisions get
resolved — app.live.auto_trade_loop now calls app.memory.auto_resolve's
`resolve_all_due()` every cycle, which walks EVERY ticker with something due
automatically (see that module's own docstring for why: there was no
evidence this CLI had ever actually been run in practice, meaning the
reflection feedback loop was likely never running live at all). This CLI
remains for a manual, single-ticker, on-demand resolution — e.g. to force
one ticker's resolution ahead of the automatic scheduler, or to resolve one
without waiting for the next cycle.

    python -m app.memory resolve --ticker AAPL

`--benchmark`, if omitted, resolves per-ticker (KOSPI for a domestic ticker,
SPY for overseas) — see `app.memory.auto_resolve.resolve_ticker`'s own
docstring. Pass it explicitly only to force a specific comparison.

    python -m app.memory backfill-horizon
    python -m app.memory backfill-horizon --all
    python -m app.memory backfill-horizon --ticker AAPL

Fills the fixed-trading-day-horizon columns (Stage 1, F6, 2026-08-13 — see
app.memory.db's `horizon_*` column comments) on already-resolved rows still
missing them. Default (no flags) bounds the scan to
`settings.memory_horizon_backfill_window_days`; `--all` removes that bound
entirely. This is a measurement-layer pass, not a decision-making one — it
never changes `status` or any existing raw_return/alpha_return/holding_days.
"""

from __future__ import annotations

import argparse
import asyncio

from app.config import settings
from app.memory.auto_resolve import backfill_horizon_returns, resolve_ticker


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resolve pending memory entries once enough time has passed.")
    parser.add_argument("command", choices=["resolve", "backfill-horizon"])
    parser.add_argument("--ticker", default=None, help="resolve: required. backfill-horizon: optional, limits to one ticker.")
    parser.add_argument(
        "--benchmark", default=None,
        help="resolve only. Explicit override; omit to auto-select per ticker's market (KOSPI/SPY).",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="backfill-horizon only: scan every trade_date, ignoring memory_horizon_backfill_window_days.",
    )
    args = parser.parse_args()
    if args.command == "resolve" and not args.ticker:
        parser.error("resolve requires --ticker")
    return args


async def _main() -> None:
    args = _parse_args()
    if args.command == "backfill-horizon":
        window_days = None if args.all else settings.memory_horizon_backfill_window_days
        count = await backfill_horizon_returns(window_days=window_days, ticker=args.ticker)
        print(f"Backfilled {count} horizon return{'s' if count != 1 else ''}.")
        return
    count = await resolve_ticker(args.ticker, args.benchmark)
    print(f"Resolved {count} pending entr{'y' if count == 1 else 'ies'} for {args.ticker}.")


if __name__ == "__main__":
    asyncio.run(_main())
