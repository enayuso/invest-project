"""Candle-pattern prompt-fix (commit `2f11d73`, 2026-09-14) mechanism A/B
runner — measures whether the candle-pattern prohibition clause appended to
`app.dataflows.session_state.intraday_narration_note` actually reaches the
pipeline and changes `session_close_guard` violations, via a paired
before/after comparison of the SAME live (ticker, date) input.

This is a MECHANISM validator, not an effect measurement. The guard this
targets (`app.agents.session_close_guard.scan_session_close_claims`) can only
ever fire while `app.dataflows.session_state.session_in_progress` is True —
that predicate is False for any past date by construction ("과거 날짜는 항상
False(백테스트 안전)", session_state.py's own module docstring) — so this can
ONLY be measured LIVE, during KRX/US regular-session hours, with today's date.
A run after the market has closed silently produces zeros in both arms; see
`--allow-closed-market` below, which exists only to unblock a dry-run/debug
session, not to sanction a real sweep outside market hours.

Base rate context (2026-09-14 audit of 141 live violations): 0.51 violations
per decision, 72% in the `completed_candle` tier the fix specifically targets.
A 4-run pilot is therefore expected to produce on the order of ~2 violations
TOTAL, and a 0/0 split is a fully acceptable, uninformative-by-design outcome
— do not read anything into small counts either way.

## Arm construction

`intraday_narration_note` is imported exactly once into production code:
`app.orchestrator.pipeline` does `from app.dataflows.session_state import
intraday_narration_note` (a `from X import Y` binding), then calls it at
`session_note = intraday_narration_note(ticker, trade_date)`. Patching
`app.dataflows.session_state.intraday_narration_note` would leave the
pipeline module's own local binding untouched, so this patches
`app.orchestrator.pipeline.intraday_narration_note` directly (see
`_candle_arm` below) and asserts the patch stuck before ever calling
`run_pipeline`.

  - arm `after` (post-fix, current code): calls the real function unchanged.
  - arm `before` (pre-fix reconstruction): calls the real function, then
    strips `_CANDLE_CLAUSE` out of the returned string. If the clause is
    expected (note non-empty) but absent, this RAISES rather than silently
    returning the un-stripped note — a future edit to the note's wording must
    break this experiment loudly, not quietly measure nothing forever.
  - Both arms pass an empty note through unchanged (session not in progress
    -> no fabricated note, in either arm).

## Reused isolation machinery

Same per-attempt clears, atomic writes, `--resume`, and interleaved/rotated
run order as `app.experiments.reasoning_ab`/`app.experiments.language_ab` —
see those modules' own docstrings for the rationale each piece exists for.
This module imports `_no_retrieved_memory`/`_state_to_dict`/
`_summarize_llm_metrics`/`degraded_report_names` from `language_ab` rather
than duplicating them.

`run_pipeline` call shape: matched to the LIVE path
(`app.live.auto_trade_loop`), which calls `run_pipeline(ticker, trade_date,
force_refresh=force_refresh, on_stage=...)` with NO `as_of` — i.e. live mode,
not backtest-as_of mode. This is deliberate, not an oversight: the guard this
experiment measures can only fire while `session_in_progress` is True, which
is live-only by construction (see module docstring above), so reproducing
`reasoning_ab`/`language_ab`'s `as_of=trade_date` shape would ALWAYS score
zero in both arms — that shape runs the pipeline as a historical backtest,
where `session_in_progress` is force-False regardless of wall clock. `trade_date`
is still `app.clock.local_today(market)` (matching auto_trade_loop's own
`today_domestic`/`today_overseas`), and `backtest_scope()` is still entered
around the call so ledger/memory writes land tagged `is_backtest=true` and
never touch the live ledger — that boundary is orthogonal to `as_of`.

The MEMORY half of that sentence only became true on 2026-09-15: until then
`app.orchestrator.pipeline` derived the row's `is_backtest` from `as_of is
not None` alone and ignored `backtest_scope()`, so this module's first 36
runs were stored as live decisions (retagged by hand afterwards). The
pipeline now honours the scope — do not "simplify" it back, and do not
assume a future harness that skips `backtest_scope()` is safe merely because
it writes no orders.

    python -m app.experiments.candle_ab \\
        --tickers 067290.KS,024060.KS --repeats 1

Aggregate manually — this pilot is meant to be read record-by-record, not
through language_ab_report (whose rollups are keyed for the language/reasoning
axes, not this one)."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Any

import app.orchestrator.pipeline as pipeline_module
from app.config import AIProfile, backtest_scope
from app.execution.ledger import clear_backtest_positions
from app.experiments.language_ab import (
    _no_retrieved_memory,
    _state_to_dict,
    _summarize_llm_metrics,
    degraded_report_names,
)
from app.dataflows.session_state import intraday_narration_note as _real_intraday_narration_note
from app.live.market_hours import is_market_open
from app.llm import metrics as llm_metrics
from app.memory.store import clear_backtest_memory
from app.orchestrator.pipeline import run_pipeline
from app.state import PipelineState

if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

logger = logging.getLogger(__name__)
logger_prefix = "[candle_ab]"

ARM_BEFORE = "before"
ARM_AFTER = "after"
_ARMS = (ARM_BEFORE, ARM_AFTER)

# Byte-identical to the clause appended to intraday_narration_note's return
# value by commit 2f11d73 (2026-09-14) — copied from
# app/dataflows/session_state.py rather than retyped from memory, so a stray
# whitespace/wording drift here can't silently make the `before` wrapper's
# str.replace a no-op (which would make `before` and `after` identical and
# the pilot's mechanism check pass for the wrong reason).
_CANDLE_CLAUSE = (
    " A candlestick pattern cannot be read from a bar that has not finished "
    "forming, so do not name one or reason from one today: no shooting star, "
    "doji, hanging man, marubozu, engulfing, long upper or lower wick or "
    "shadow, and no 'distribution pattern' or 'accumulation pattern' read off "
    "today's price action — every one of those asserts a completed bar. The "
    "distance between the latest price and today's high or low so far is not "
    "a wick and is not evidence of rejection, distribution, or accumulation; "
    "it is simply where the stock has traded up to now, and the remainder of "
    "the session can still erase or extend it. Describe what has happened so "
    "far in plain terms instead."
)

# `[<tier>]` marker written by
# app.agents.session_close_guard.scan_session_close_claims, e.g.
# "AgentName: [completed_candle] 'span' vs SESSION-IN-PROGRESS ground truth".
_TIER_MARKER_RE = re.compile(r"\[([a-z_]+)\]")


def tier_counts(violations: list[str]) -> dict[str, int]:
    """`{tier: count}` parsed from each violation's leading `[<tier>]`
    marker — `completed_candle` is the primary metric (the fix specifically
    targeted that tier; see this module's docstring), `closed_assertion` the
    other tier `scan_session_close_claims` can emit. A violation string with
    no recognizable marker is silently uncounted here (never observed in
    practice — scan_session_close_claims always prefixes one of exactly two
    literal tier names) rather than raising, since this is read-only
    reporting, not a correctness gate."""
    counts: dict[str, int] = {}
    for v in violations:
        m = _TIER_MARKER_RE.search(v)
        if not m:
            continue
        tier = m.group(1)
        counts[tier] = counts.get(tier, 0) + 1
    return counts


def _before_note(ticker: str, day: date) -> str:
    """Reconstructs the PRE-fix note: calls the real (post-fix) function,
    then strips `_CANDLE_CLAUSE` back out. Raises if the note is non-empty
    (session genuinely in progress) but does not contain the clause —
    silently returning the unmodified note would make this arm identical to
    `after` and the pilot's core mechanism check (item 3 in the spec) would
    pass for the wrong reason. Returns "" unchanged when the real function
    returns "" (session not in progress) — never fabricates a note."""
    note = _real_intraday_narration_note(ticker, day)
    if not note:
        return ""
    if _CANDLE_CLAUSE not in note:
        raise AssertionError(
            f"candle_ab: _CANDLE_CLAUSE not found in a non-empty note for {ticker} {day} — "
            "app.dataflows.session_state.intraday_narration_note's wording has drifted since "
            "this experiment's _CANDLE_CLAUSE constant was copied from it. Update the constant "
            "(never silently fall back to returning the note unmodified — see this function's "
            "own docstring)."
        )
    return note.replace(_CANDLE_CLAUSE, "")


def _after_note(ticker: str, day: date) -> str:
    """The real, current (post-fix) function, called unchanged."""
    return _real_intraday_narration_note(ticker, day)


@contextmanager
def _candle_arm(arm: str, captured: dict[str, str]) -> Iterator[None]:
    """Patches `app.orchestrator.pipeline.intraday_narration_note` (the
    pipeline MODULE'S OWN `from X import Y` binding — see this module's
    docstring for why patching `app.dataflows.session_state`'s copy would not
    affect the pipeline at all) to a wrapper around the arm's note function,
    asserts the patch landed, and restores the original binding
    unconditionally in `finally` — same unconditional-restore shape as
    `reasoning_ab._reasoning_arm`, for the same reason: a leaked patch would
    silently contaminate every later run of an unattended multi-run sweep.

    `captured["note"]` is set to whatever the wrapper actually returned to
    the pipeline on this call — the mechanism proof
    (`note_has_candle_clause`/`note_chars`) is recorded from THIS, not from a
    note recomputed after the fact (the market could open/close between the
    call and the caller reading it back)."""
    base = _before_note if arm == ARM_BEFORE else _after_note

    def _wrapped(ticker: str, day: date) -> str:
        note = base(ticker, day)
        captured["note"] = note
        return note

    original = pipeline_module.intraday_narration_note
    pipeline_module.intraday_narration_note = _wrapped
    assert pipeline_module.intraday_narration_note is _wrapped, (
        "candle_ab: patching app.orchestrator.pipeline.intraday_narration_note did not stick"
    )
    try:
        yield
    finally:
        pipeline_module.intraday_narration_note = original


def _run_filename(ticker: str, trade_date: date, arm: str, repeat: int) -> str:
    return f"{ticker}_{trade_date.isoformat()}_{arm}_r{repeat}.json"


async def _run_pipeline_with_retry(
    ticker: str, trade_date: date, arm: str, attempts: int, retry_delay: float,
) -> tuple[PipelineState, int, int, list[str], dict[str, Any], int, str]:
    """Same retry/isolation shape as `reasoning_ab._run_pipeline_with_retry`,
    with the candle-note arm as the varying axis and NO `as_of` passed to
    `run_pipeline` (see this module's docstring for why — the guard under
    test only fires live). Captures the note the wrapper actually returned
    (`note` in the return tuple) for the caller to record — the mechanism
    proof (`note_has_candle_clause`/`note_chars`) needs the note as SEEN by
    the pipeline on the attempt that was kept, not a note recomputed after
    the fact (the market could have opened/closed between attempts)."""
    captured_note: dict[str, str] = {}

    for attempt in range(1, attempts + 1):
        llm_metrics.drain()  # discard any prior attempt's records before this one starts
        captured_note.clear()
        try:
            await clear_backtest_positions([ticker])
            await clear_backtest_memory([ticker])
            with (
                _no_retrieved_memory() as (original_n_same, original_n_cross),
                _candle_arm(arm, captured_note),
                backtest_scope(),
            ):
                # NOTE: no `as_of=` — reproduces the LIVE call shape
                # (app.live.auto_trade_loop), which is the only regime
                # `session_in_progress` can ever be True in. See module
                # docstring's "run_pipeline call shape" section.
                state = await run_pipeline(
                    ticker, trade_date, profile=AIProfile.PRODUCTION, force_refresh=True,
                )
        except Exception:
            if attempt == attempts:
                raise
            logger.warning(
                "%s %s %s: pipeline call failed (attempt %d/%d) — retrying in %.0fs",
                ticker, trade_date, arm, attempt, attempts, retry_delay, exc_info=True,
            )
            await asyncio.sleep(retry_delay)
            continue

        degraded = degraded_report_names(state.reports)
        if not degraded or attempt == attempts:
            llm_summary = _summarize_llm_metrics(llm_metrics.drain())
            return (
                state, original_n_same, original_n_cross, degraded, llm_summary, attempt,
                captured_note.get("note", ""),
            )
        logger.warning(
            "%s %s %s: analyst report(s) degraded to placeholders %s (attempt %d/%d) — retrying in %.0fs",
            ticker, trade_date, arm, degraded, attempt, attempts, retry_delay,
        )
        await asyncio.sleep(retry_delay)
    raise AssertionError("unreachable")  # loop always returns or raises on the last attempt


async def _run_one(
    ticker: str,
    trade_date: date,
    arm: str,
    repeat: int,
    out_dir: Path,
    run_idx: int,
    total: int,
    resume: bool,
    attempts: int,
    retry_delay: float,
) -> None:
    out_path = out_dir / _run_filename(ticker, trade_date, arm, repeat)
    if resume and out_path.exists():
        print(
            f"{logger_prefix} run {run_idx}/{total} ticker={ticker} date={trade_date} "
            f"arm={arm} rep={repeat} skipped (--resume, already done)"
        )
        return

    print(
        f"{logger_prefix} run {run_idx}/{total} ticker={ticker} date={trade_date} "
        f"arm={arm} rep={repeat} starting..."
    )
    started_at = datetime.now()
    start = time.monotonic()
    try:
        (
            state, original_n_same, original_n_cross, degraded, llm_summary, attempts_used, note,
        ) = await _run_pipeline_with_retry(ticker, trade_date, arm, attempts, retry_delay)
    except Exception as exc:
        elapsed = time.monotonic() - start
        error_path = out_dir / f"{ticker}_{trade_date.isoformat()}_{arm}_r{repeat}_error.json"
        error_path.write_text(
            json.dumps(
                {
                    "ticker": ticker,
                    "date": trade_date.isoformat(),
                    "arm": arm,
                    "repeat": repeat,
                    "started_at": started_at.isoformat(),
                    "elapsed_seconds": elapsed,
                    "error": repr(exc),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(
            f"{logger_prefix} run {run_idx}/{total} ticker={ticker} date={trade_date} "
            f"arm={arm} rep={repeat} FAILED elapsed={elapsed:.0f}s error={exc!r}"
        )
        return

    elapsed = time.monotonic() - start
    finished_at = datetime.now()

    isolation_violation = bool(state.past_context.strip())

    if degraded:
        print(
            f"{logger_prefix} WARNING run {run_idx}/{total} ticker={ticker} date={trade_date} "
            f"arm={arm} rep={repeat}: DEGRADED — {degraded} fell back to placeholder reports "
            f"after {attempts} attempt(s). This run saw less data than a clean one and must be "
            f"excluded from the arm comparison."
        )

    violations = list(state.session_close_violations)
    by_tier = tier_counts(violations)

    record = {
        "ticker": ticker,
        "arm": arm,
        "repeat": repeat,
        "trade_date": trade_date.isoformat(),
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "wall_seconds": elapsed,
        "attempts_used": attempts_used,
        "isolation_violation": isolation_violation,
        "degraded": bool(degraded),
        "degraded_reports": degraded,
        "force_refresh": True,
        "memory_n_same_override": 0,
        "memory_n_same_production": original_n_same,
        "memory_n_cross_override": 0,
        "memory_n_cross_production": original_n_cross,
        "llm_attempts_used": attempts_used,
        **llm_summary,
        **_state_to_dict(state),
        # Mechanism proof (see module docstring) — recorded from the note the
        # patched wrapper actually returned during this run's kept attempt,
        # not recomputed after the fact.
        "note_injected": bool(note),
        "note_chars": len(note),
        "note_has_candle_clause": _CANDLE_CLAUSE in note,
        "session_close_violations": violations,
        "session_close_by_tier": by_tier,
    }
    tmp_path = out_path.with_suffix(out_path.suffix + f".tmp{os.getpid()}")
    tmp_path.write_text(json.dumps(record, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    os.replace(tmp_path, out_path)
    if isolation_violation:
        raise RuntimeError(
            f"memory isolation FAILED for {ticker} {trade_date} arm={arm} rep={repeat} — "
            f"past_context was non-empty despite memory_n_same=memory_n_cross=0. Result written to "
            f"{out_path}; see app.experiments.language_ab._no_retrieved_memory's docstring before "
            "re-running."
        )

    decision = state.final_decision
    rating = decision.rating.value if decision is not None else "N/A"
    print(
        f"{logger_prefix} run {run_idx}/{total} ticker={ticker} date={trade_date} "
        f"arm={arm} rep={repeat} done rating={rating} note_injected={bool(note)} "
        f"note_has_candle_clause={_CANDLE_CLAUSE in note} by_tier={by_tier} elapsed={elapsed:.0f}s"
    )


def _plan_runs(tickers: list[str], dates: list[date], repeats: int) -> list[tuple[str, date, str, int]]:
    """Same interleaved/rotated ordering as `reasoning_ab._plan_runs` — see
    that function's docstring for the elapsed-time/server-health confound
    this avoids. Produces `len(tickers) * len(dates) * repeats * 2` entries."""
    runs: list[tuple[str, date, str, int]] = []
    for ticker in tickers:
        for trade_date in dates:
            for repeat in range(1, repeats + 1):
                offset = (repeat - 1) % len(_ARMS)
                ordered = _ARMS[offset:] + _ARMS[:offset]
                runs.extend((ticker, trade_date, arm, repeat) for arm in ordered)
    return runs


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tickers", required=True, help="Comma-separated ticker universe.")
    parser.add_argument(
        "--dates", default=None,
        help="Comma-separated YYYY-MM-DD trade dates. Defaults to today (local, per-market) — "
             "the guard under test can only fire for today's date while that market is open; "
             "a past date always scores zero in both arms.",
    )
    parser.add_argument("--repeats", type=int, default=1, help="Independent repeats per (ticker, date, arm).")
    parser.add_argument(
        "--out-dir", default=None,
        help="Defaults to results/candle_ab/<UTC timestamp>/ if not given.",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip any (ticker, date, arm, repeat) whose result file already exists in --out-dir.",
    )
    parser.add_argument(
        "--attempts", type=int, default=3,
        help="Attempts per run, counting both raised errors and degraded (placeholder) analyst "
             "reports as retriable. Default 3.",
    )
    parser.add_argument(
        "--retry-delay", type=float, default=60.0,
        help="Seconds between attempts. Default 60 — short deliberately, see reasoning_ab's own "
             "docstring for why settings.backtest_pipeline_retry_delay_seconds (600) would idle a "
             "short pilot for the better part of an hour.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the planned runs and exit without calling run_pipeline or touching the DB.",
    )
    parser.add_argument(
        "--allow-closed-market", action="store_true",
        help="Bypass the market-hours guard. Exists ONLY so a dry-run/debug session outside "
             "market hours isn't blocked — a real sweep run with this flag after the close will "
             "silently produce zeros in both arms, which is worse than not running at all.",
    )
    args = parser.parse_args()
    if args.attempts < 1:
        parser.error(f"--attempts must be >= 1, got {args.attempts}")
    return args


def _default_dates(tickers: list[str]) -> list[date]:
    """Today's date on EACH ticker's own market calendar (domestic vs
    overseas) — matching `app.live.auto_trade_loop`'s own
    `today_domestic`/`today_overseas` split, since a single shared date would
    be wrong for a mixed domestic/overseas ticker list. Only used when
    `--dates` is omitted; when given explicitly, the same date list applies
    to every ticker exactly like `reasoning_ab`/`language_ab`."""
    from app.clock import local_today

    markets = {"domestic" if t.endswith(".KS") else "overseas" for t in tickers}
    return sorted({local_today(m) for m in markets})


async def _main() -> None:
    args = _parse_args()
    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    if args.dates:
        dates = [date.fromisoformat(d.strip()) for d in args.dates.split(",") if d.strip()]
    else:
        dates = _default_dates(tickers)
    repeats = args.repeats

    market_open_any = any(is_market_open(t) for t in tickers)
    print(f"{logger_prefix} market currently open for at least one requested ticker: {market_open_any}")
    if not market_open_any and not args.allow_closed_market and not args.dry_run:
        print(
            f"{logger_prefix} REFUSING to run: the market is closed for every requested ticker right "
            "now, and session_in_progress() is force-False for any past date, so a sweep run now "
            "would silently produce zero session_close_violations in BOTH arms — indistinguishable "
            "from a real null result. Re-run during KRX (09:00-15:30 KST) or US (09:30-16:00 ET) "
            "regular hours, or pass --allow-closed-market for a dry-run/debug session only."
        )
        sys.exit(1)

    runs = _plan_runs(tickers, dates, repeats)
    total = len(runs)

    if args.dry_run:
        print(f"{logger_prefix} DRY RUN — {total} run(s) planned, nothing executed:")
        for run_idx, (ticker, trade_date, arm, repeat) in enumerate(runs, start=1):
            print(f"  {run_idx}/{total}: ticker={ticker} date={trade_date} arm={arm} rep={repeat}")
        if not market_open_any:
            print(
                f"{logger_prefix} NOTE: market is closed right now — a real (non-dry-run) sweep "
                "would be refused unless --allow-closed-market is also passed."
            )
        return

    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = Path("results") / "candle_ab" / datetime.now().strftime("%Y%m%dT%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"{logger_prefix} {total} run(s) queued -> {out_dir}")

    for run_idx, (ticker, trade_date, arm, repeat) in enumerate(runs, start=1):
        await _run_one(
            ticker, trade_date, arm, repeat, out_dir, run_idx, total, args.resume,
            args.attempts, args.retry_delay,
        )

    print(f"{logger_prefix} done. Results in {out_dir}")


if __name__ == "__main__":
    asyncio.run(_main())
