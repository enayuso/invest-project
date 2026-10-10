"""Aggregates `backtest_runs` rows produced by `app.backtest.atr_sweep` —
one row per (ATR candidate x group), averaged here into one line per ATR
candidate so the sweep's actual question ("which multiplier wins on
risk-adjusted return") can be answered without manually reading N rows.

    python -m app.backtest.atr_sweep_report --label-prefix trailing-sweep
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections import defaultdict

from app.backtest.atr_sweep import parse_atr_multipliers
from app.backtest.experiments import list_experiments

if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


def _atr_value_from_label(label: str, prefix: str) -> str | None:
    marker = f"{prefix}-atr"
    if not label.startswith(marker):
        return None
    return label[len(marker):]


async def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label-prefix", required=True)
    parser.add_argument("--limit", type=int, default=200, help="How many recent backtest_runs rows to scan.")
    parser.add_argument(
        "--expect-atr-multipliers", default=None,
        help="Same --atr-multipliers value the sweep was run with — when given, flags any candidate "
        "with zero rows as MISSING rather than letting it silently vanish from the table (e.g. atr_sweep.py "
        "exited nonzero for it, or it fell outside --limit).",
    )
    parser.add_argument(
        "--expect-groups", type=int, default=None,
        help="Expected row count per candidate (parallel_portfolio's --num-groups) — with --expect-atr-multipliers, "
        "flags a candidate with fewer rows than this as PARTIAL.",
    )
    args = parser.parse_args()

    # 2026-09-11 fix: scope the SQL-level LIMIT to this sweep's own label
    # prefix, not the whole backtest_runs table. Before this, --limit bounded
    # the most recent rows across EVERY experiment ever recorded — if enough
    # unrelated runs (other sweeps, ad-hoc invocations, scheduled jobs) were
    # recorded since this ATR sweep ran, this sweep's own older rows could
    # fall outside the window and silently vanish from `by_atr` with no
    # indication the cause was an unrelated global LIMIT rather than the
    # sweep itself failing. The Python-side _atr_value_from_label filter
    # below still runs unchanged — this only changes what set of rows
    # --limit is applied against.
    # 이 스윕의 label prefix로 SQL 단계에서 미리 필터링 — limit이 전체
    # 테이블이 아니라 이 스윕의 최근 행에만 적용되도록.
    experiments = await list_experiments(limit=args.limit, label_prefix=f"{args.label_prefix}-atr")
    by_atr: dict[str, list] = defaultdict(list)
    for exp in experiments:
        atr_value = _atr_value_from_label(exp.label or "", args.label_prefix)
        if atr_value is not None:
            by_atr[atr_value].append(exp)

    expected = parse_atr_multipliers(args.expect_atr_multipliers) if args.expect_atr_multipliers else None
    missing = [v for v in expected if v not in by_atr] if expected else []

    if not by_atr:
        print(f"No backtest_runs found with label prefix '{args.label_prefix}-atr...' in the last {args.limit} rows.")
        for v in missing:
            print(f"{v:>10}  MISSING — zero backtest_runs rows (candidate failed or was never run)")
        return

    def _mean(values: list[float]) -> float | None:
        present = [v for v in values if v is not None]
        return sum(present) / len(present) if present else None

    print(f"{'ATR mult.':>10} {'groups':>7} {'return':>9} {'sharpe':>8} {'sortino':>8} {'maxDD':>8} {'CVaR95':>8}")
    print("-" * 62)
    for atr_value in sorted(by_atr, key=lambda v: float(v)):
        rows = by_atr[atr_value]
        ret = _mean([r.return_pct for r in rows])
        sharpe = _mean([r.sharpe for r in rows])
        sortino = _mean([r.sortino for r in rows])
        max_dd = _mean([r.max_drawdown for r in rows])
        cvar = _mean([r.cvar_95 for r in rows])
        partial_note = ""
        if args.expect_groups is not None and len(rows) < args.expect_groups:
            partial_note = f"  PARTIAL — {len(rows)}/{args.expect_groups} groups"
        print(
            f"{atr_value:>10} {len(rows):>7} "
            f"{(f'{ret:+.2%}' if ret is not None else 'n/a'):>9} "
            f"{(f'{sharpe:+.2f}' if sharpe is not None else 'n/a'):>8} "
            f"{(f'{sortino:+.2f}' if sortino is not None else 'n/a'):>8} "
            f"{(f'{max_dd:+.2%}' if max_dd is not None else 'n/a'):>8} "
            f"{(f'{cvar:+.2%}' if cvar is not None else 'n/a'):>8}"
            f"{partial_note}"
        )
    for v in missing:
        print(f"{v:>10}  MISSING — zero backtest_runs rows (candidate failed or was never run)")
    print(
        "\nNote: each row averages across this candidate's disjoint ticker groups (see "
        "parallel_portfolio's docstring for why portfolio-level caps only apply within a group, "
        "not across the whole sweep universe)."
    )


if __name__ == "__main__":
    asyncio.run(_main())
