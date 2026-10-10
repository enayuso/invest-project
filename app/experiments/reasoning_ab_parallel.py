"""Parallel launcher for `app.experiments.reasoning_ab` — process-per-
ticker-group, same shape as `app.backtest.parallel_portfolio` (which
itself follows `app.backtest.parallel`'s "reuse process-level isolation
for free" reasoning).

Why NOT `asyncio.gather` inside `reasoning_ab.py` itself (rejected design,
2026-09-08 Opus review — do not revisit without re-deriving these):

- `settings.research_manager_reasoning` (the A/B arm itself) is a plain
  attribute on the `Settings` singleton, not a ContextVar — unlike
  `_is_backtest`/`_current_account_mode` right next to it in `app/config.py`.
  Two `run_pipeline` calls in flight at once would both read whichever
  value happens to be installed when their Research Manager step fires,
  silently measuring nothing while still writing a confidently-wrong
  `"research_manager_reasoning"` field into each result record.
- `language_ab._no_retrieved_memory`'s restore-on-exit would race the same
  way: one run's `finally` can restore `memory_n_same/_cross` while a
  sibling is still mid-pipeline, which trips `reasoning_ab.py`'s own
  `isolation_violation` tripwire and aborts the WHOLE sweep instead of
  just the offending run.
- `app.llm.metrics.drain()` empties a process-global ring buffer — one
  run's `drain()` steals a concurrent sibling's already-recorded token/
  latency data.

A separate OS process per ticker group sidesteps all three for free: each
child is a plain, unmodified, sequential `reasoning_ab` invocation, so
every existing invariant inside it (isolation checks, retry/resume, error
files) holds completely unchanged. `reasoning_ab.py` itself needs zero
changes for this to work.

KIS/LLM budget scaling: each child's own throttle only knows about its
own calls, so the group's combined request rate must be divided the same
way `app.backtest.parallel`/`parallel_portfolio` already do — see
`_run_one_group`'s own comment for the exact env vars.

MANDATORY calibration before committing to a full sweep: nobody has
measured this specific vLLM deployment's behavior under concurrent
pipeline runs (the last such measurement, `auto_trade_max_concurrent_
analyses`'s own comment history in app/config.py, is against a since-
replaced model and predates the 2026-08-24 latency-collapse incident).
Run a tiny pilot first and compare per-run `elapsed_seconds` against a
sequential baseline before launching the real sweep:

    # baseline (sequential, 1 ticker x 1 date x 1 repeat x 2 arms = 2 runs)
    python -m app.experiments.reasoning_ab --tickers 005930.KS --dates 2026-07-01 --repeats 1

    # K=2 pilot (2 tickers, same shape = 4 runs across 2 processes)
    python -m app.experiments.reasoning_ab_parallel --num-groups 2 \\
        --tickers 005930.KS,AAPL --dates 2026-07-01 --repeats 1

If per-run elapsed_seconds stays roughly flat vs. the baseline, the
endpoint has headroom — proceed (K=2, optionally try K=3). If it inflates
by ~1.8x or more, or degrades WITHIN one child's run (the 08-24 queue-
backlog signature), vLLM is already saturated at concurrency 1 and
parallelizing buys ~0 wall-clock — fall back to plain `reasoning_ab.py`
run overnight instead. Record whichever result you get in `app.config.
Settings.auto_trade_max_concurrent_analyses`'s own comment history, per
that setting's standing "don't change this without recording measurements"
instruction — this sweep is exactly the kind of measurement it asks for.

    python -m app.experiments.reasoning_ab_parallel \\
        --num-groups 2 --tickers 005930.KS,AAPL,MSFT,NVDA,000660.KS,035420.KS,051910.KS,GOOGL,META,TSLA \\
        --dates 2026-07-01,2026-07-15 --repeats 3
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import os
import sys
import time
from pathlib import Path

from app.config import settings

# Same Windows console-codepage fix as parallel_portfolio.py/audit/__main__.py
# — this module's own docstring/help text and summary output use em dashes.
if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

logger_prefix = "[reasoning_ab_parallel]"


def pick_groups(tickers_arg: str, num_groups: int) -> list[list[str]]:
    """Round-robin partition of `--tickers` into `num_groups` DISJOINT
    lists — a ticker's 12 runs (2 dates x 3 repeats x 2 arms, at the
    reasoning_ab.py defaults) must never be split across two concurrent
    children: `clear_backtest_positions([ticker])`/`clear_backtest_memory(
    [ticker])` at the start of each of that ticker's runs would delete a
    sibling child's still-in-flight rows for the SAME ticker (same
    "duplicate ticker corrupts the shared paper ledger" rule as
    app.backtest.parallel.pick_batch_tickers / parallel_portfolio.pick_groups
    — enforced here, not just documented, by rejecting a duplicate
    outright)."""
    tickers = [t.strip() for t in tickers_arg.split(",") if t.strip()]
    if len(set(tickers)) != len(tickers):
        dupes = sorted({t for t in tickers if tickers.count(t) > 1})
        raise ValueError(
            f"duplicate ticker(s): {dupes} — the same ticker must never appear in two groups "
            "(each group's clear_backtest_positions/clear_backtest_memory would delete a sibling "
            "group's in-flight rows for that ticker)"
        )
    if num_groups < 1:
        raise ValueError(f"--num-groups must be >= 1, got {num_groups}")
    if num_groups > len(tickers):
        raise ValueError(f"--num-groups {num_groups} exceeds distinct ticker count ({len(tickers)})")

    groups: list[list[str]] = [[] for _ in range(num_groups)]
    for i, ticker in enumerate(tickers):
        groups[i % num_groups].append(ticker)
    return [g for g in groups if g]


def _group_key(group_index: int) -> str:
    return f"group{group_index}"


def _stdout_log_path(out_dir: Path, group_index: int) -> Path:
    """<out-dir>/group<N>.log — inside the SHARED --out-dir, not a separate
    logs/ path. parallel_portfolio.py's own docstring records a real
    2026-08-12 incident where two unrelated runs' group logs collided on a
    shared non-run-scoped path and one silently truncated the other;
    out_dir here is already unique per sweep (timestamped, or user-given),
    so nesting the logs inside it avoids that class of collision for free."""
    return out_dir / f"{_group_key(group_index)}.log"


async def _run_one_group(
    group_index: int, group: list[str], args: argparse.Namespace, out_dir: Path, num_groups: int,
) -> tuple[int, list[str], int, float]:
    """Launches one plain, unmodified `python -m app.experiments.reasoning_ab`
    child process for `group`. No subprocess-level retry (unlike
    parallel_portfolio._run_one_group's `_GROUP_SUBPROCESS_MAX_ATTEMPTS`) —
    that retry exists there because a crashed portfolio_harness run loses
    hours of in-memory date-by-date replay progress. reasoning_ab persists
    each run to its own JSON immediately (reasoning_ab.py's own atomic
    write), so recovery from a crashed child is simply re-running the
    PARENT with --resume: already-written result files are skipped and
    only the missing runs re-execute — simpler and strictly better than a
    blind whole-subprocess restart here."""
    cmd = [
        sys.executable, "-m", "app.experiments.reasoning_ab",
        "--tickers", ",".join(group),
        "--dates", args.dates,
        "--language", args.language,
        "--repeats", str(args.repeats),
        "--out-dir", str(out_dir),
        "--attempts", str(args.attempts),
        "--retry-delay", str(args.retry_delay),
    ]
    if args.resume:
        cmd.append("--resume")

    env = {
        **os.environ,
        # Each child's BaseHttpClient throttle only sees its OWN calls, so
        # K children each independently honoring "1 req/sec of MY calls"
        # would jointly exceed the KIS paper account's zero-margin cap
        # (EGW00201 — see app.config.settings' own kis_min_request_interval_
        # paper comment, and app.backtest.parallel's identical scaling for
        # the same reason). Backtests only ever run against paper, so only
        # the _PAPER var needs scaling.
        "KIS_MIN_REQUEST_INTERVAL_PAPER": str(settings.kis_min_request_interval_paper * num_groups),
        # Each child gets its own PriorityLimiter(llm_max_concurrent_
        # requests) (app.llm.client) — K children would otherwise put up to
        # K * llm_max_concurrent_requests concurrent requests on the single
        # self-hosted vLLM instance. Dividing keeps the AGGREGATE ceiling
        # equal to what one sequential reasoning_ab run already produces —
        # this launcher changes how many PIPELINES run at once, not the
        # total request ceiling the endpoint has been living with.
        "LLM_MAX_CONCURRENT_REQUESTS": str(max(1, settings.llm_max_concurrent_requests // num_groups)),
    }

    start_t = time.monotonic()
    log_path = _stdout_log_path(out_dir, group_index)
    with open(log_path, "w", encoding="utf-8") as f:
        proc = await asyncio.create_subprocess_exec(*cmd, stdout=f, stderr=asyncio.subprocess.STDOUT, env=env)
        returncode = await proc.wait()
    elapsed = time.monotonic() - start_t
    return group_index, group, returncode, elapsed


async def _progress_monitor(out_dir: Path, total: int, interval: float) -> None:
    """Prints a global rollup every `interval` seconds by COUNTING RESULT
    FILES in `out_dir`, not by tracking children's own output — children
    finish runs out of order and each writes to its own separate log, so a
    parent-side counter driven by child stdout would be both interleaved
    and wrong. Counting files is order-independent and correctly accounts
    for --resume-skipped runs with no extra bookkeeping. Runs until
    cancelled by the caller once the sweep finishes."""
    start = time.monotonic()
    while True:
        await asyncio.sleep(interval)
        errors = list(out_dir.glob("*_error.json"))
        done = [p for p in out_dir.glob("*.json") if not p.name.endswith("_error.json")]
        elapsed = time.monotonic() - start
        eta_h = (elapsed / len(done) * (total - len(done)) / 3600) if done else float("nan")
        print(
            f"{logger_prefix} progress {len(done)}/{total} done, {len(errors)} failed, "
            f"elapsed={elapsed / 3600:.1f}h eta={eta_h:.1f}h",
            flush=True,
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tickers", required=True, help="Comma-separated ticker universe, partitioned into groups.")
    parser.add_argument(
        "--num-groups", type=int, required=True,
        help="How many disjoint ticker groups to run concurrently, each as its own process. "
             "Start at 2 and run the calibration pilot in this module's own docstring before going "
             "higher — this endpoint's concurrent-request behavior has not been measured since the "
             "last model swap. Hard-recommended cap: 3.",
    )
    parser.add_argument("--dates", required=True, help="Comma-separated YYYY-MM-DD as_of dates, passed through unchanged.")
    parser.add_argument("--language", default="en", help="Passed through unchanged — see reasoning_ab.py's own docstring.")
    parser.add_argument("--repeats", type=int, default=3, help="Passed through unchanged.")
    parser.add_argument(
        "--out-dir", default=None,
        help="Shared across all groups (defaults to results/reasoning_ab/<UTC timestamp>/) — "
             "must be the SAME directory for every child, or the sweep splits across separate "
             "directories and app.experiments.language_ab_report can't aggregate it in one pass.",
    )
    parser.add_argument("--resume", action="store_true", help="Passed through unchanged to every child.")
    parser.add_argument("--attempts", type=int, default=3, help="Passed through unchanged.")
    parser.add_argument("--retry-delay", type=float, default=60.0, help="Passed through unchanged.")
    parser.add_argument(
        "--progress-interval", type=float, default=120.0,
        help="Seconds between global progress rollups (default 120).",
    )
    return parser.parse_args()


async def _main() -> None:
    args = _parse_args()
    try:
        groups = pick_groups(args.tickers, args.num_groups)
    except ValueError as exc:
        raise SystemExit(str(exc)) from None

    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        from datetime import datetime

        out_dir = Path("results") / "reasoning_ab" / datetime.now().strftime("%Y%m%dT%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    dates = [d.strip() for d in args.dates.split(",") if d.strip()]
    total = sum(len(group) * len(dates) * args.repeats * 2 for group in groups)  # 2 = fixed arm count

    print(f"{logger_prefix} launching {len(groups)} group(s), {total} run(s) total -> {out_dir}")
    for i, group in enumerate(groups):
        print(f"  {_group_key(i)}: {group}")

    monitor = asyncio.create_task(_progress_monitor(out_dir, total, args.progress_interval))
    try:
        results = await asyncio.gather(
            *(_run_one_group(i, group, args, out_dir, len(groups)) for i, group in enumerate(groups))
        )
    finally:
        monitor.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await monitor

    print(f"\n{logger_prefix} summary:")
    for group_index, group, code, elapsed in sorted(results, key=lambda r: r[0]):
        status = "OK" if code == 0 else f"FAILED (exit {code})"
        print(
            f"  {_group_key(group_index):<10} {status:<16} {elapsed / 3600:5.2f}h  "
            f"tickers={group}  -> {_stdout_log_path(out_dir, group_index)}"
        )
    print(
        f"\n{logger_prefix} Aggregate with: python -m app.experiments.language_ab_report --dir {out_dir}  "
        "(reads the arm label from the 'language' field — see reasoning_ab.py's own docstring)."
    )


if __name__ == "__main__":
    asyncio.run(_main())
