"""Parallel launcher for `app.backtest.portfolio_harness` — same process-per-
batch-item shape as `app.backtest.parallel` (single-ticker), but each item
here is a whole DISJOINT GROUP of tickers run through the multi-ticker
portfolio harness, not one ticker through the single-ticker one.

Why groups, not "just parallelize the ticker loop inside one
portfolio_harness run": that loop's whole point is coordinating PORTFOLIO-
LEVEL risk caps (concentration/correlation-bucket/drawdown) across every
ticker's shared book at each date — each ticker's turn re-fetches a fresh
snapshot reflecting every earlier ticker's trades THAT SAME DATE, then
commits, sequentially. Running that loop's iterations concurrently would
recreate the exact live race this project already found and fixed (two
tickers in flight at once each seeing the other's pre-trade balance and
jointly approving exposure neither alone would — see
app.live.auto_trade_loop's own module docstring). So the safe unit of
parallelism here is a GROUP of tickers with its own independent book, run
as its own OS process — same "reuse process-level isolation for free"
reasoning app.backtest.parallel already uses, one level up.

Real tradeoff this introduces: portfolio-level caps are only enforced
WITHIN each group, not across the whole universe — 3 groups of 10 tickers
each means concentration/correlation caps see three separate 10-ticker
books, not one 30-ticker book. Fewer, larger groups gets closer to a single
run's fidelity at the cost of less parallelism; more, smaller groups
parallelizes better but fragments the risk-cap view. Choose --num-groups
accordingly.

    python -m app.backtest.parallel_portfolio --tickers AAPL,MSFT,NVDA,005930.KS,000660.KS,035420.KS \
        --num-groups 3 --start 2026-01-01 --end 2026-04-08 --screened --top-n 3
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time

from app.backtest.harness import default_log_path
from app.backtest.parallel import overall_exit_code
from app.config import settings

# Windows ties stdout to the OS console codepage (cp949 on a Korean locale)
# unless told otherwise — this module's own docstring/help text and summary
# output contain em dashes, which would otherwise raise UnicodeEncodeError.
# Same fix already applied to app/backtest/__main__.py and app/audit/__main__.py.
if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

logger_prefix = "[parallel_portfolio]"


def pick_groups(tickers_arg: str | None, num_groups: int, trade_allowlist: list[str]) -> list[list[str]]:
    """Resolves --tickers (or the configured trade_allowlist) into `num_groups`
    disjoint, round-robin-partitioned ticker lists. Raises ValueError on a
    duplicate ticker (same "refuse to launch a batch that would corrupt the
    shared paper ledger" principle as app.backtest.parallel.pick_batch_
    tickers) or an impossible --num-groups (more groups than distinct
    tickers, or < 1)."""
    tickers = [t.strip() for t in tickers_arg.split(",") if t.strip()] if tickers_arg else list(trade_allowlist)
    if len(set(tickers)) != len(tickers):
        dupes = sorted({t for t in tickers if tickers.count(t) > 1})
        raise ValueError(
            f"duplicate ticker(s): {dupes} — the same ticker must never appear in two groups "
            "(each group runs as its own process against the shared paper ledger)"
        )
    if num_groups < 1:
        raise ValueError(f"--num-groups must be >= 1, got {num_groups}")
    if num_groups > len(tickers):
        raise ValueError(f"--num-groups {num_groups} exceeds distinct ticker count ({len(tickers)})")

    groups: list[list[str]] = [[] for _ in range(num_groups)]
    for i, ticker in enumerate(tickers):
        groups[i % num_groups].append(ticker)
    return [g for g in groups if g]  # a ticker count not evenly divisible can leave no trailing empty groups anyway


def _group_key(group_index: int) -> str:
    return f"group{group_index}"


def _stdout_log_path(label: str | None, group_index: int) -> str:
    """logs/domestic/parallel_portfolio_<run_tag>_group<N>.log — groups mix
    domestic/overseas freely, so this always lands under domestic/
    (default_log_path needs a ticker-shaped key to route; the group key
    itself isn't a real ticker, this just reuses the same path-building
    helper).

    `run_tag` (the CLI's own --label, or this process's pid if none was
    given) makes the path unique PER INVOCATION, not just per group index —
    without it, two separate parallel_portfolio runs sharing the same
    --num-groups count (e.g. one domestic, one overseas, launched back to
    back) both resolve to plain group0.log/group1.log and collide.
    Confirmed happening in practice (2026-08-12): the overseas run's own
    group0 log open (mode "w" on its first attempt) silently truncated the
    still-in-progress domestic run's group0 log, so `tail` on that path
    showed overseas lines under what looked like the domestic group's
    file — genuinely confusing, though it never touched either run's
    actual replay state (app.backtest.checkpoint keys on label+tickers,
    entirely separate from this log path)."""
    run_tag = label or f"pid{os.getpid()}"
    return default_log_path("005930.KS", f"parallel_portfolio_{run_tag}_{_group_key(group_index)}.log")


# Coarse safety net ON TOP OF harness._run_pipeline_with_retry's own
# per-date in-process retry — that one absorbs an LLM outage without losing
# in-memory progress; this one is the fallback for anything IT doesn't
# catch (e.g. a crash outside the pipeline call itself, or an outage that
# outlasts even its budget). A restarted subprocess always starts flat
# regardless of how far the dead one got — portfolio_harness's own
# clear_backtest_positions call at startup wipes whatever the dead attempt
# left behind (see that function's docstring: exactly the "don't inherit a
# prior run's ending position" property this needs). Kept small (2 total
# attempts): unlike the in-process retry, one of these attempts re-walks
# every trade_date from scratch, so it's an expensive last resort, not a
# first line of defense.
_GROUP_SUBPROCESS_MAX_ATTEMPTS = 2
_GROUP_SUBPROCESS_RETRY_DELAY_SECONDS = 300.0


async def _run_one_group(
    group_index: int, group: list[str], args: argparse.Namespace, batch_size: int
) -> tuple[int, list[str], int, float]:
    """Launches one `python -m app.backtest.portfolio_harness` child process
    for `group`, its stdout+stderr captured to a per-group file (never
    inherited directly — several processes writing straight to this
    parent's stdout at once would interleave into unreadable garbage).
    Retries the WHOLE subprocess (see `_GROUP_SUBPROCESS_MAX_ATTEMPTS`'s own
    comment) if it exits non-zero, rather than reporting the group FAILED
    on the first crash.

    `batch_size` scales KIS_MIN_REQUEST_INTERVAL_PAPER for this child the
    same way app.backtest.parallel._run_one does — the whole batch's
    combined request rate against the shared KIS account must stay ~1/sec,
    not `batch_size`/sec, since each child's own throttle only knows about
    its own calls. Backtests only ever run against the paper account, so
    the _PAPER env var is the only one that needs scaling here."""
    cmd = [
        sys.executable, "-m", "app.backtest.portfolio_harness",
        "--tickers", ",".join(group),
        "--start", args.start, "--end", args.end,
        "--cadence-days", str(args.cadence_days), "--benchmark", args.benchmark,
    ]
    if args.label:
        cmd += ["--label", args.label]
    if args.screened:
        cmd += ["--screened", "--top-n", str(args.top_n), "--lookback-days", str(args.lookback_days)]
        if args.no_volatility_adjustment:
            cmd += ["--no-volatility-adjustment"]
        if args.no_correlation_dedup:
            cmd += ["--no-correlation-dedup"]
    env = {**os.environ, "KIS_MIN_REQUEST_INTERVAL_PAPER": str(1.0 * batch_size)}

    start_t = time.monotonic()
    log_path = _stdout_log_path(args.label, group_index)
    for attempt in range(1, _GROUP_SUBPROCESS_MAX_ATTEMPTS + 1):
        # "w" only on the first attempt — a retry appends to the same file
        # (mode "a") so the log shows the full crash-then-retry history in
        # one place instead of the previous attempt's output being
        # silently overwritten.
        with open(log_path, "w" if attempt == 1 else "a", encoding="utf-8") as f:
            if attempt > 1:
                f.write(f"\n{logger_prefix} retry attempt {attempt}/{_GROUP_SUBPROCESS_MAX_ATTEMPTS}\n")
            proc = await asyncio.create_subprocess_exec(*cmd, stdout=f, stderr=asyncio.subprocess.STDOUT, env=env)
            returncode = await proc.wait()
        if returncode == 0 or attempt == _GROUP_SUBPROCESS_MAX_ATTEMPTS:
            break
        print(
            f"{logger_prefix} {_group_key(group_index)} exited {returncode} "
            f"(attempt {attempt}/{_GROUP_SUBPROCESS_MAX_ATTEMPTS}) — retrying in "
            f"{_GROUP_SUBPROCESS_RETRY_DELAY_SECONDS:.0f}s"
        )
        await asyncio.sleep(_GROUP_SUBPROCESS_RETRY_DELAY_SECONDS)
    elapsed = time.monotonic() - start_t
    return group_index, group, returncode, elapsed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tickers", help="Comma-separated tickers to partition into groups (default: the configured trade_allowlist).")
    parser.add_argument("--num-groups", type=int, required=True, help="How many disjoint groups to run concurrently.")
    parser.add_argument("--start", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD")
    parser.add_argument("--cadence-days", type=int, default=5)
    parser.add_argument("--benchmark", default="SPY")
    parser.add_argument("--label", default=None)
    parser.add_argument("--screened", action="store_true", help="Pass --screened through to each group's portfolio_harness run.")
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--lookback-days", type=int, default=10)
    parser.add_argument("--no-volatility-adjustment", action="store_true")
    parser.add_argument("--no-correlation-dedup", action="store_true")
    return parser.parse_args()


async def _main() -> int:
    args = _parse_args()
    if not args.tickers:
        # Refresh from Postgres before falling back to the trade_allowlist — see
        # app.backtest.__main__'s identical call for why. Skipped when
        # --tickers is given explicitly since it wins outright either way.
        from app.web import trade_allowlist_store

        await trade_allowlist_store.load()
    try:
        groups = pick_groups(args.tickers, args.num_groups, settings.trade_allowlist)
    except ValueError as exc:
        raise SystemExit(str(exc)) from None

    print(f"{logger_prefix} launching {len(groups)} portfolio-backtest group(s):")
    for i, group in enumerate(groups):
        print(f"  {_group_key(i)}: {group}")

    results = await asyncio.gather(*(_run_one_group(i, group, args, len(groups)) for i, group in enumerate(groups)))

    print(f"\n{logger_prefix} summary:")
    for group_index, group, code, elapsed in sorted(results, key=lambda r: r[0]):
        status = "OK" if code == 0 else f"FAILED (exit {code})"
        print(
            f"  {_group_key(group_index):<10} {status:<16} {elapsed:6.1f}s  "
            f"tickers={group}  -> {_stdout_log_path(args.label, group_index)}"
        )
    print(
        f"\n{logger_prefix} NOTE: portfolio-level caps (concentration/correlation-bucket/drawdown) were "
        f"enforced WITHIN each group's own book, not across all {sum(len(g) for g in groups)} tickers at once "
        "— see this module's own docstring for the tradeoff."
    )

    exit_code = overall_exit_code(code for _, _, code, _ in results)
    if exit_code:
        failed = [_group_key(i) for i, _, code, _ in results if code != 0]
        print(
            f"{logger_prefix} {len(failed)} of {len(results)} group(s) FAILED: {failed} "
            "— no backtest_runs rows were recorded for them",
            file=sys.stderr,
        )
    return exit_code


if __name__ == "__main__":
    sys.exit(asyncio.run(_main()))
