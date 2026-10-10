"""One-off backfill: reconstruct `memory_entries.held_at_decision_inferred`
for live decisions that predate the `held_at_decision` column (added
2026-08-13) from the internal `trades` ledger.

Not a general-purpose or repeatable tool — it targets a fixed, closed set of
177 rows (created 2026-07-22 through 2026-08-13, `is_backtest = false`,
`held_at_decision IS NULL`) discovered during a 2026-08-20 self-audit
investigation. A distinct batch of 47 rows in the same date range turned out
to be backtest-replay contamination (mislabeled `is_backtest = false`) and
were retagged `is_backtest = true` separately, NOT touched by this script —
see project memory HANDOFF-2026-08-20-* for the full investigation.

Reconstruction rule, validated against the 144 rows where `held_at_decision`
was independently observed from the live broker balance (97.2% agreement,
140/144 — see app.memory.db's `held_at_decision_inferred` column comment for
why the 4 disagreements don't change the recommendation):

    For a decision row (ticker, account_mode, user_id, created_at), find the
    most recent `trades` row for the SAME ticker/account_mode/user_id/
    is_backtest=false with `trades.created_at < memory_entries.created_at`
    (ordered by created_at DESC, id DESC as a tiebreak) and read
    `shares_after > 0`. No prior trade => flat => False.

Ordering by `created_at`, not `trade_date`, is deliberate and was the one
thing that mattered in validation: 29 ticker-days in this window have more
than one trade (same-day round trips, up to 36 same-day trades for one
ticker) — a `trade_date`-only comparison can't tell which side of a same-day
flip a given decision landed on, `created_at` can.

Never writes `held_at_decision` itself — only the separate `_inferred`
column. Always run --dry-run first.

Usage:
    python -m scripts.backfill_held_at_decision --dry-run
    python -m scripts.backfill_held_at_decision --apply
"""

from __future__ import annotations

import argparse
import asyncio
import logging

from app.memory.db import get_pool

logger = logging.getLogger(__name__)

_SELECT_TARGETS = """
    SELECT id, ticker, account_mode, user_id, created_at
    FROM memory_entries
    WHERE is_backtest = false
      AND held_at_decision IS NULL
      AND held_at_decision_inferred IS NULL
      AND (created_at::date - trade_date) <= 3
    ORDER BY id
"""

# The reconstruction query, run per row rather than as one big correlated
# UPDATE — this script runs once, correctness/auditability of the per-row
# trail matters far more than a single round trip.
_LOOKUP_LAST_TRADE = """
    SELECT shares_after
    FROM trades
    WHERE ticker = $1 AND account_mode = $2 AND user_id = $3 AND is_backtest = false
      AND created_at < $4
    ORDER BY created_at DESC, id DESC
    LIMIT 1
"""


async def _run(apply: bool) -> None:
    pool = await get_pool()
    async with pool.acquire() as conn:
        targets = await conn.fetch(_SELECT_TARGETS)
        print(f"{len(targets)} candidate rows (is_backtest=false, held_at_decision IS NULL, "
              f"held_at_decision_inferred IS NULL, created_at within 3 days of trade_date)")

        results: list[tuple[int, bool]] = []
        for row in targets:
            last_trade = await conn.fetchrow(
                _LOOKUP_LAST_TRADE, row["ticker"], row["account_mode"], row["user_id"], row["created_at"],
            )
            inferred_held = bool(last_trade and last_trade["shares_after"] > 0)
            results.append((row["id"], inferred_held))
            basis = "no prior trade" if last_trade is None else f"shares_after={last_trade['shares_after']}"
            status = "held" if inferred_held else "flat"
            print(f"  id={row['id']:>5}  {row['ticker']:<12}  {status:<4}  ({basis})")

        held_n = sum(1 for _, h in results if h)
        print(f"\nsummary: {len(results)} rows -> held={held_n}, flat={len(results) - held_n}")

        if not apply:
            print("\n--dry-run: no writes made. Re-run with --apply to write held_at_decision_inferred.")
            return

        async with conn.transaction():
            for decision_id, inferred_held in results:
                await conn.execute(
                    "UPDATE memory_entries SET held_at_decision_inferred = $1 WHERE id = $2",
                    inferred_held, decision_id,
                )
        print(f"\napplied: wrote held_at_decision_inferred for {len(results)} rows.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true", help="Report what would be written, without writing.")
    group.add_argument("--apply", action="store_true", help="Write held_at_decision_inferred for real.")
    args = parser.parse_args()
    asyncio.run(_run(apply=args.apply))


if __name__ == "__main__":
    main()
