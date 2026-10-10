"""Parallel backtest launcher — runs several single-ticker backtests as
separate OS processes concurrently.

Empirically confirmed (2026-07-23) this is worth doing: 3 tickers run
concurrently against the shared vLLM server finished within ~8 seconds of
each other at ~7 minutes total, vs. ~20+ minutes if the server only served
one request at a time — the server does handle concurrent load well.

Each ticker gets its own OS process (not `asyncio.gather` within one
process) specifically to reuse this project's existing process-level
isolation for free: in-process globals this codebase already has (the
DB pool in app.memory.db, the cached broker in app.dataflows.kis_provider,
...) are each per-process, so separate processes never contend on them —
no new isolation code needed here.

Tickers within one batch MUST be distinct: running the SAME ticker twice
concurrently corrupts the shared paper ledger (`app.execution.ledger`'s
`trades` table has no per-run isolation — see that module and
app.backtest.harness's own docstrings). This refuses to launch a batch
with a duplicate rather than silently letting that happen.

    python -m app.backtest.parallel --tickers AAPL,MSFT,NVDA --start 2026-05-01 --end 2026-07-01 --cadence-days 30
    python -m app.backtest.parallel --random-n 3 --start 2026-05-01 --end 2026-07-01 --cadence-days 30
"""

from __future__ import annotations

import argparse
import asyncio
import os
import random
import sys
import time

from collections.abc import Iterable

from app.backtest.harness import default_log_path
from app.config import settings


def pick_batch_tickers(tickers_arg: str | None, random_n: int | None) -> list[str]:
    """Resolves --tickers/--random-n into the distinct ticker list to run.
    Raises ValueError on a duplicate (explicit --tickers) or an impossible
    --random-n (larger than the de-duplicated trade_allowlist) — a batch that
    can't safely run shouldn't launch any subprocess at all, not fail
    halfway through."""
    if tickers_arg:
        tickers = [t.strip() for t in tickers_arg.split(",") if t.strip()]
    else:
        pool = list(dict.fromkeys(settings.trade_allowlist))  # de-dup, keep order
        if random_n > len(pool):
            raise ValueError(f"--random-n {random_n} exceeds distinct trade_allowlist size ({len(pool)})")
        tickers = random.sample(pool, random_n)

    if len(set(tickers)) != len(tickers):
        dupes = sorted({t for t in tickers if tickers.count(t) > 1})
        raise ValueError(
            f"duplicate ticker(s) in batch: {dupes} — running the same ticker concurrently "
            "corrupts the shared paper ledger, refusing to launch"
        )
    return tickers


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tickers", help="Comma-separated tickers to run in parallel (must be distinct).")
    parser.add_argument(
        "--random-n", type=int, help="Pick this many distinct random trade_allowlist tickers instead of --tickers."
    )
    parser.add_argument("--start", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD")
    parser.add_argument("--cadence-days", type=int, default=5)
    parser.add_argument("--benchmark", default="SPY")
    parser.add_argument("--label", default=None)
    args = parser.parse_args()
    if bool(args.tickers) == bool(args.random_n):
        parser.error("exactly one of --tickers or --random-n is required")
    return args


def overall_exit_code(codes: Iterable[int]) -> int:
    """1 if ANY child exited nonzero (including a negative signal code),
    else 0. Consumers (app.backtest.atr_sweep, subprocessing this launcher
    per ATR candidate) judge a whole launcher run purely by its exit code —
    without this, `_main` always exited 0 regardless of how many groups
    crashed, so a candidate whose every group failed was silently reported
    "OK" upstream."""
    return 1 if any(code != 0 for code in codes) else 0


def _stdout_log_path(ticker: str) -> str:
    """logs/domestic/parallel_<ticker>.log or logs/overseas/... — same
    domestic/overseas split as the warning-log file below, so a mixed
    KR/US batch doesn't dump every file flat into the project root."""
    return default_log_path(ticker, f"parallel_{ticker.replace('.', '_')}.log")


async def _run_one(ticker: str, args: argparse.Namespace, batch_size: int) -> tuple[str, int, float]:
    """Launches one `python -m app.backtest` child process for `ticker`,
    its stdout+stderr captured to a per-ticker file (never inherited
    directly — several processes writing straight to this parent's stdout
    at once would interleave into unreadable garbage).

    `batch_size` scales KIS_MIN_REQUEST_INTERVAL_PAPER for this child so the
    WHOLE batch's combined request rate against the shared KIS account
    stays ~1/sec, not `batch_size`/sec — each child's own KISBroker
    throttle (see app.config.Settings.kis_min_request_interval) only
    knows about its own calls, so `batch_size` separate processes each
    independently honoring "1 req/sec of MY OWN calls" can otherwise
    jointly exceed KIS's real per-account limit (confirmed live,
    2026-07-24: EGW00201 during an unscaled 2-ticker run). Backtests only
    ever run against the paper account (see kis_account_mode's own
    docstring), so the _PAPER env var is the only one that needs scaling
    here — the _real field keeps its own default, untouched."""
    cmd = [
        sys.executable, "-m", "app.backtest",
        "--ticker", ticker, "--start", args.start, "--end", args.end,
        "--cadence-days", str(args.cadence_days), "--benchmark", args.benchmark,
        "--log-file", default_log_path(ticker, f"backtest_{ticker.replace('.', '_')}.log"),
    ]
    if args.label:
        cmd += ["--label", args.label]
    env = {**os.environ, "KIS_MIN_REQUEST_INTERVAL_PAPER": str(1.0 * batch_size)}

    start_t = time.monotonic()
    with open(_stdout_log_path(ticker), "w", encoding="utf-8") as f:
        proc = await asyncio.create_subprocess_exec(*cmd, stdout=f, stderr=asyncio.subprocess.STDOUT, env=env)
        returncode = await proc.wait()
    elapsed = time.monotonic() - start_t
    return ticker, returncode, elapsed


async def _main() -> int:
    args = _parse_args()
    if not args.tickers:
        # Refresh from Postgres before falling back to the trade_allowlist — see
        # app.backtest.__main__'s identical call for why. Skipped when
        # --tickers is given explicitly since it wins outright either way.
        from app.web import trade_allowlist_store

        await trade_allowlist_store.load()
    try:
        tickers = pick_batch_tickers(args.tickers, args.random_n)
    except ValueError as exc:
        raise SystemExit(str(exc)) from None

    print(f"[parallel] launching {len(tickers)} backtests: {tickers}")
    results = await asyncio.gather(*(_run_one(t, args, len(tickers)) for t in tickers))

    print("\n[parallel] summary:")
    for ticker, code, elapsed in results:
        status = "OK" if code == 0 else f"FAILED (exit {code})"
        print(f"  {ticker:<12} {status:<16} {elapsed:6.1f}s  -> {_stdout_log_path(ticker)}")

    exit_code = overall_exit_code(code for _, code, _ in results)
    if exit_code:
        failed = [ticker for ticker, code, _ in results if code != 0]
        print(f"[parallel] {len(failed)} of {len(results)} backtest(s) FAILED: {failed}", file=sys.stderr)
    return exit_code


if __name__ == "__main__":
    sys.exit(asyncio.run(_main()))
