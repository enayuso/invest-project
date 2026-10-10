"""One-off correction: 132 resolved live KRX (`.KS`) decisions
(trade_date 2026-07-22 through 2026-07-30) are still scored against SPY
instead of KOSPI (`^KS11`).

`app/execution/portfolio_risk.py`'s `benchmark_for_market` (added
2026-08-12, project memory benchmark-mismatch-fixed.md) already fixed the
LIVE resolution path going forward and backfilled 144 rows that were
already resolved at that time. These 132 rows resolved with the old SPY
default before that fix landed but were somehow not included in that
backfill — discovered 2026-08-20 while investigating a self-audit "Hold
(held)" win-rate anomaly (project memory
hold-scoring-gap-and-hold-held-defect-2026-08-20.md), where this turned out
to be a bigger, more urgent distortion than the thing that prompted the
investigation (39% of the entire scorable live audit was affected).

Same surgical convention as the 2026-08-12 backfill: ONLY `alpha_return` /
`horizon_alpha_return` / `benchmark_ticker` are corrected. `raw_return`
(the ticker's own return — doesn't depend on the benchmark, already
correct), `holding_days`, `horizon_raw_return`/`horizon_trading_days`/
`horizon_eval_date`, and the LLM `reflection` text are left untouched —
recomputing those would require re-running resolution (and an LLM call)
for no benefit, and a stale reflection mentioning "SPY" is a cosmetic
issue, not a numeric one.

Recomputes against the EXACT same dates already on record — `eval_date =
trade_date + holding_days` for the variable window, `horizon_eval_date`
as-is for the fixed window — so this is purely a benchmark substitution,
not a re-resolution with a different window.

Usage:
    python -m scripts.fix_krx_spy_benchmark_contamination --dry-run
    python -m scripts.fix_krx_spy_benchmark_contamination --apply
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import math
from datetime import date, timedelta

from app.dataflows.yfinance import get_price_series
from app.memory.db import get_pool

logger = logging.getLogger(__name__)

_BENCHMARK = "^KS11"

_SELECT_TARGETS = """
    SELECT id, ticker, trade_date, raw_return, holding_days,
           horizon_raw_return, horizon_trading_days, horizon_eval_date
    FROM memory_entries
    WHERE ticker LIKE '%.KS' AND benchmark_ticker = 'SPY'
      AND is_backtest = false AND status = 'resolved'
    ORDER BY id
"""


def _usable(value: float | None) -> bool:
    return value is not None and math.isfinite(value)


async def _run(apply: bool) -> None:
    pool = await get_pool()
    async with pool.acquire() as conn:
        targets = await conn.fetch(_SELECT_TARGETS)
        print(f"{len(targets)} candidate rows (ticker LIKE '%.KS', benchmark_ticker='SPY', "
              f"is_backtest=false, status='resolved')")
        if not targets:
            return

        min_date = min(row["trade_date"] for row in targets)
        max_date = max(
            max(row["trade_date"] + timedelta(days=row["holding_days"]), row["horizon_eval_date"])
            for row in targets
        )
        print(f"fetching {_BENCHMARK} prices for [{min_date}, {max_date}] once, reused for every row")
        kospi = await get_price_series(_BENCHMARK, min_date, max_date)

        results: list[tuple[int, float | None, float | None]] = []
        skipped = 0
        for row in targets:
            trade_date: date = row["trade_date"]
            eval_date = trade_date + timedelta(days=row["holding_days"])
            horizon_eval_date: date = row["horizon_eval_date"]

            kospi_entry = kospi.get(trade_date)
            kospi_exit = kospi.get(eval_date)
            kospi_horizon_exit = kospi.get(horizon_eval_date)

            if not (_usable(kospi_entry) and _usable(kospi_exit)):
                print(f"  id={row['id']:>5}  {row['ticker']:<12}  SKIPPED — no usable "
                      f"{_BENCHMARK} close for {trade_date} and/or {eval_date}")
                skipped += 1
                continue

            new_alpha = row["raw_return"] - (kospi_exit - kospi_entry) / kospi_entry

            new_horizon_alpha = None
            if row["horizon_raw_return"] is not None:
                if _usable(kospi_horizon_exit):
                    new_horizon_alpha = row["horizon_raw_return"] - (kospi_horizon_exit - kospi_entry) / kospi_entry
                else:
                    print(f"  id={row['id']:>5}  {row['ticker']:<12}  horizon_alpha_return left unchanged — "
                          f"no usable {_BENCHMARK} close for {horizon_eval_date}")

            results.append((row["id"], new_alpha, new_horizon_alpha))
            horizon_note = f", horizon_alpha={new_horizon_alpha:+.4f}" if new_horizon_alpha is not None else ""
            print(f"  id={row['id']:>5}  {row['ticker']:<12}  alpha={new_alpha:+.4f}{horizon_note}")

        print(f"\nsummary: {len(results)} rows to correct, {skipped} skipped (no usable benchmark price)")

        if not apply:
            print("\n--dry-run: no writes made. Re-run with --apply to write the corrections.")
            return

        async with conn.transaction():
            for decision_id, new_alpha, new_horizon_alpha in results:
                if new_horizon_alpha is not None:
                    await conn.execute(
                        "UPDATE memory_entries SET alpha_return = $1, horizon_alpha_return = $2, "
                        "benchmark_ticker = $3 WHERE id = $4",
                        new_alpha, new_horizon_alpha, _BENCHMARK, decision_id,
                    )
                else:
                    await conn.execute(
                        "UPDATE memory_entries SET alpha_return = $1, benchmark_ticker = $2 WHERE id = $3",
                        new_alpha, _BENCHMARK, decision_id,
                    )
        print(f"\napplied: corrected {len(results)} rows.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true", help="Report what would be written, without writing.")
    group.add_argument("--apply", action="store_true", help="Write the corrections for real.")
    args = parser.parse_args()
    asyncio.run(_run(apply=args.apply))


if __name__ == "__main__":
    main()
