"""Manual CLI trigger for the Phase-2 backtest harness.

    python -m app.backtest --ticker AAPL --start 2026-01-01 --end 2026-07-01 \
        --cadence-days 5 --benchmark SPY

    python -m app.backtest --random --cadence-days 7   # random trade_allowlist ticker + date range
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import date

from app.backtest import reporting
from app.backtest.experiments import record_experiment
from app.backtest.harness import (
    configure_cli_logging,
    default_log_path,
    log_progress,
    pick_random_ticker_and_range,
    run_backtest,
)

# Windows ties stdout/stderr to the OS console codepage (cp949 on a Korean
# locale) unless told otherwise, so any em dash or Korean text in the final
# report (app.backtest.reporting.render) raises UnicodeEncodeError right at
# the last print — after the whole backtest replay (every historical date,
# every LLM call) already succeeded and was saved to the DB. Confirmed live:
# a 2-ticker run exited 1 on exactly this, even though every decision it
# made was intact in memory_entries. Same fix already applied to
# app/audit/__main__.py; ports it here too.
if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay the trading-agent pipeline across historical dates.")
    parser.add_argument("--ticker")
    parser.add_argument("--start", help="YYYY-MM-DD")
    parser.add_argument("--end", help="YYYY-MM-DD")
    parser.add_argument(
        "--random", action="store_true",
        help="Pick a random trade_allowlist ticker and a random past date range instead of --ticker/--start/--end.",
    )
    parser.add_argument("--cadence-days", type=int, default=5)
    parser.add_argument("--benchmark", default="SPY")
    parser.add_argument("--label", default=None, help="Optional tag for this run, e.g. 'prompt-v2' (see app.audit experiments).")
    parser.add_argument(
        "--log-file", default=None,
        help="Where warning-level log noise goes (see configure_cli_logging). Default: "
        "logs/domestic/backtest.log or logs/overseas/backtest.log, by the resolved ticker's "
        "market — app.backtest.parallel overrides this per-ticker so concurrent runs don't interleave.",
    )
    args = parser.parse_args()
    if args.random:
        if args.ticker or args.start or args.end:
            parser.error("--random cannot be combined with --ticker/--start/--end")
    elif not (args.ticker and args.start and args.end):
        parser.error("--ticker, --start, and --end are required unless --random is given")
    return args


async def _main() -> None:
    args = _parse_args()
    if args.random:
        # Refresh settings.trade_allowlist from Postgres before picking —
        # this standalone CLI never goes through app.web.state's startup
        # (the only place that normally calls this), so without it,
        # --random would only ever pick from .env.local's static seed list
        # instead of the currently-held/watched tickers the live system has
        # actually accumulated since then. Best-effort/no-op if the store is
        # unreachable (see trade_allowlist_store.load's own docstring) — falls
        # back to whatever settings already has, same as before this existed.
        from app.web import trade_allowlist_store

        await trade_allowlist_store.load()
        ticker, start, end = pick_random_ticker_and_range()
    else:
        ticker, start, end = args.ticker, date.fromisoformat(args.start), date.fromisoformat(args.end)
    configure_cli_logging(args.log_file if args.log_file is not None else default_log_path(ticker))
    if args.random:
        log_progress("random_pick", ticker=ticker, start=start, end=end)
    result = await run_backtest(
        ticker=ticker,
        start=start,
        end=end,
        cadence_days=args.cadence_days,
        benchmark_ticker=args.benchmark,
    )
    print(await reporting.render(result))

    try:
        # return_pct is the REAL ledger return (same one reporting.render
        # just printed), not result.strategy_return — that's the exposure-
        # proxy signal-quality number (ignores real sizing/stop-loss), kept
        # separately in exposure_return_pct so it isn't lost (useful for a
        # signal-level A/B that never moves the ledger, e.g. a trailing-
        # stop sweep) but never mixed into the column portfolio_harness
        # fills with its own genuinely-actual portfolio_return.
        _, _, ledger_return = await reporting.ledger_snapshot(result)
        exp_id = await record_experiment(
            kind="single",
            tickers=[ticker],
            benchmark_ticker=args.benchmark,
            start=start,
            end=end,
            cadence_days=args.cadence_days,
            n_dates=len(result.records),
            return_pct=ledger_return,
            exposure_return_pct=result.strategy_return,
            benchmark_return_pct=result.benchmark_return,
            risk_metrics=result.ledger_risk_metrics,
            label=args.label,
        )
        log_progress("experiment_recorded", id=exp_id)
    except Exception as exc:
        # Bookkeeping, not the point of running a backtest — a DB hiccup here
        # shouldn't make the run itself look like it failed.
        log_progress("experiment_record_failed", error=str(exc))


if __name__ == "__main__":
    asyncio.run(_main())
