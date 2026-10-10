"""Trailing-stop parameter sweep — runs `app.backtest.parallel_portfolio`
once per `--atr-multipliers` candidate, each launch getting its own
`TRAILING_STOP_ATR_MULTIPLIER` environment override so every child process's
`risk_rules.rule("trailing_stop_atr_multiplier")` (which falls back to
`settings.trailing_stop_atr_multiplier`, itself sourced from the environment —
see app.execution.risk_rules.rule) picks up that candidate's value without
touching any other trailing-stop code path.

One candidate value's groups still run in parallel (same fidelity tradeoff as
parallel_portfolio itself — see that module's docstring); candidates
themselves run SEQUENTIALLY, not fanned out further, so the shared paper
ledger and KIS rate-limit throttle only ever see one candidate's load at a
time (two candidates' subprocesses touching the same ledger concurrently
would double-count trades against it).

Every child run already self-records to `backtest_runs` via
`record_experiment` (portfolio_harness's own behavior) tagged with
`--label`; this script sets a shared label prefix per candidate
(`{label-prefix}-atr{value}`) so results can be compared afterwards with
`python -m app.backtest.atr_sweep_report --label-prefix ...` or directly via
`app.backtest.experiments.list_experiments`.

    python -m app.backtest.atr_sweep --tickers 005930.KS,000660.KS,AAPL,MSFT,035420.KS,NVDA,005490.KS,GOOGL,051910.KS,AMZN,006400.KS,META \
        --num-groups 2 --start 2026-05-01 --end 2026-08-08 --screened --top-n 3 \
        --atr-multipliers 1.0,1.5,1.75,2.25 --label-prefix trailing-sweep
"""

from __future__ import annotations

import argparse
import asyncio
import os
import subprocess
import sys
import time

logger_prefix = "[atr_sweep]"

if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


def parse_atr_multipliers(raw: str) -> list[str]:
    """Splits/validates --atr-multipliers BEFORE any subprocess is spawned
    — without this, a malformed entry (e.g. "1.0,abc") only fails deep
    inside every spawned child (settings validation), reported by the
    launcher as an ordinary per-group failure rather than a clear upfront
    error. Returns the ORIGINAL textual tokens (not floats): labels are
    built as f"{prefix}-atr{token}" and atr_sweep_report matches on that
    exact text, so "1.50" must not be normalized to "1.5"."""
    values = [c.strip() for c in raw.split(",") if c.strip()]
    if not values:
        raise ValueError("--atr-multipliers must list at least one value")
    for v in values:
        try:
            parsed = float(v)
        except ValueError:
            raise ValueError(f"--atr-multipliers entry {v!r} is not a number") from None
        if not (parsed > 0) or parsed != parsed or parsed in (float("inf"), float("-inf")):
            raise ValueError(f"--atr-multipliers entry {v!r} must be a positive finite number")
    return values


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tickers", required=True, help="Comma-separated universe, passed through unchanged to parallel_portfolio.")
    parser.add_argument("--num-groups", type=int, required=True)
    parser.add_argument("--start", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD")
    parser.add_argument("--cadence-days", type=int, default=5)
    parser.add_argument("--benchmark", default="SPY")
    parser.add_argument("--screened", action="store_true")
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--lookback-days", type=int, default=10)
    parser.add_argument(
        "--atr-multipliers", required=True,
        help="Comma-separated TRAILING_STOP_ATR_MULTIPLIER candidates, e.g. '1.0,1.5,1.75,2.25'.",
    )
    parser.add_argument("--label-prefix", required=True, help="Each candidate's runs are labeled '{prefix}-atr{value}'.")
    args = parser.parse_args()
    try:
        args.atr_multipliers_parsed = parse_atr_multipliers(args.atr_multipliers)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def _run_one_candidate(atr_value: str, args: argparse.Namespace) -> tuple[str, int, float]:
    label = f"{args.label_prefix}-atr{atr_value}"
    cmd = [
        sys.executable, "-m", "app.backtest.parallel_portfolio",
        "--tickers", args.tickers,
        "--num-groups", str(args.num_groups),
        "--start", args.start, "--end", args.end,
        "--cadence-days", str(args.cadence_days), "--benchmark", args.benchmark,
        "--label", label,
    ]
    if args.screened:
        cmd += ["--screened", "--top-n", str(args.top_n), "--lookback-days", str(args.lookback_days)]
    env = {**os.environ, "TRAILING_STOP_ATR_MULTIPLIER": atr_value}

    print(f"{logger_prefix} candidate ATR={atr_value} -> label={label!r}: launching {' '.join(cmd)}")
    start_t = time.monotonic()
    result = subprocess.run(cmd, env=env)
    elapsed = time.monotonic() - start_t
    print(f"{logger_prefix} candidate ATR={atr_value} finished in {elapsed:.1f}s (exit {result.returncode})")
    if result.returncode != 0:
        # parallel_portfolio/parallel now propagate a nonzero exit whenever
        # ANY of their groups failed — surfaced loudly here since this
        # sweep's stdout log is the artifact a human actually reads, and a
        # failed candidate means backtest_runs rows for it are missing or
        # incomplete (some/all groups never called record_experiment).
        print(
            f"{logger_prefix} *** CANDIDATE FAILED *** ATR={atr_value} label={label!r} exit "
            f"{result.returncode} — backtest_runs rows for this candidate are missing or "
            "incomplete, treat atr_sweep_report output as unusable for it",
            file=sys.stderr,
        )
    return label, result.returncode, elapsed


def main() -> None:
    args = _parse_args()
    candidates = args.atr_multipliers_parsed

    print(f"{logger_prefix} sweeping {len(candidates)} ATR multiplier candidate(s): {candidates}")
    outcomes = [_run_one_candidate(value, args) for value in candidates]

    print(f"\n{logger_prefix} sweep summary:")
    for label, code, elapsed in outcomes:
        status = "OK" if code == 0 else f"FAILED (exit {code})"
        print(f"  {label:<30} {status:<16} {elapsed:7.1f}s")
    print(
        f"\n{logger_prefix} each candidate's per-group results were recorded to backtest_runs "
        f"under labels '{args.label_prefix}-atr<value>' — compare with "
        f"`python -m app.backtest.atr_sweep_report --label-prefix {args.label_prefix}`"
    )

    failed = [label for label, code, _ in outcomes if code != 0]
    if failed:
        print(f"{logger_prefix} {len(failed)} of {len(outcomes)} candidate(s) FAILED: {failed}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
