"""Research Manager reasoning=True vs reasoning=False A/B runner (2026-09-07,
AI-engine step-0 latency audit — priority item 3, [[ai-engine-step0-latency-
measurement-2026-09-07]]).

Reuses `app.experiments.language_ab`'s isolation machinery (per-run ledger/
memory clears, `_no_retrieved_memory`, `_state_to_dict`, `_summarize_llm_metrics`,
`degraded_report_names`) wholesale rather than duplicating it, but varies
`settings.research_manager_reasoning` instead of `language` — `language` here
is FIXED across both arms (passed straight through to `run_pipeline`
unchanged) so it can never re-confound this comparison; `app.experiments.
language_ab` already measured the language axis separately.

This is explicitly a QUALITY-BET investigation, not a pure latency win (see
`settings.research_manager_reasoning`'s own comment): the Research Manager's
rating is one of the two independent votes
`app.orchestrator.pipeline._consensus_override_tier` (#205 gate) reads, so a
rating-distribution shift between arms changes that gate's fire rate in a way
no unit test catches. Aggregate results with `app.experiments.language_ab_report`
(pass `pre_gate=False` for the POST-gate rating comparison that is this
sweep's primary metric) — its rollups are keyed by `record["language"]`, so
each record here writes the arm label ("reason_on"/"reason_off") into that
same field as a deliberate alias; the REAL, fixed pipeline language is kept
separately in `record["pipeline_language"]` so it is never lost or confused
with the arm.

Known noise floor before running this (see [[language-ab-experiment-
conclusion-2026-08-18]]): PRE-gate rating self-inconsistency across repeats
of the SAME (ticker, date) input was measured at ~80% (en) / ~60% (ko) — a
120-run sweep (10 tickers x 2 dates x 3 repeats x 2 arms) can only reliably
detect a LARGE shift (roughly >=20-25pp of POST-gate Hold rate). Treat this
as a tripwire for "did turning reasoning off break the anti-hedging
behavior badly", not a proof of equivalence between the two arms.

    python -m app.experiments.reasoning_ab \\
        --tickers 005930.KS,AAPL --dates 2026-07-01,2026-07-15 --repeats 3

Aggregate afterwards with:
    python -m app.experiments.language_ab_report --dir <out-dir>
(reading "language" as the arm label — see above).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Any

from app.config import AIProfile, backtest_scope, settings
from app.execution.ledger import clear_backtest_positions
from app.experiments.language_ab import (
    _no_retrieved_memory,
    _state_to_dict,
    _summarize_llm_metrics,
    degraded_report_names,
)
from app.llm import metrics as llm_metrics
from app.memory.store import clear_backtest_memory
from app.orchestrator.pipeline import run_pipeline
from app.state import PipelineState

if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

logger = logging.getLogger(__name__)
logger_prefix = "[reasoning_ab]"

ARM_ON = "reason_on"
ARM_OFF = "reason_off"
_ARMS = (ARM_ON, ARM_OFF)


@contextmanager
def _reasoning_arm(on: bool) -> Iterator[bool]:
    """Temporarily overrides `settings.research_manager_reasoning`, yielding
    the original value, and always restores it in a `finally` — same shape as
    `language_ab._no_retrieved_memory`. A leaked override (e.g. an exception
    skipping the restore) would silently contaminate every later run in an
    unattended multi-hour sweep, which is exactly why the restore is
    unconditional rather than only happening at the end of a clean run."""
    original = settings.research_manager_reasoning
    settings.research_manager_reasoning = on
    try:
        yield original
    finally:
        settings.research_manager_reasoning = original


def _run_filename(ticker: str, trade_date: date, arm: str, repeat: int) -> str:
    return f"{ticker}_{trade_date.isoformat()}_{arm}_r{repeat}.json"


async def _run_pipeline_with_retry(
    ticker: str, trade_date: date, arm: str, language: str, attempts: int, retry_delay: float,
) -> tuple[PipelineState, int, int, list[str], dict[str, Any], int]:
    """Same retry/isolation shape as `language_ab._run_pipeline_with_retry`,
    with `arm` (reasoning on/off) as the varying axis instead of `language`.
    Returns `(state, original_memory_n_same, original_memory_n_cross,
    degraded_analyst_names, llm_metrics_summary, attempts_used)` — see that
    function's own docstring for what each element is for."""
    for attempt in range(1, attempts + 1):
        llm_metrics.drain()  # discard any prior attempt's records before this one starts
        try:
            await clear_backtest_positions([ticker])
            await clear_backtest_memory([ticker])
            with (
                _no_retrieved_memory() as (original_n_same, original_n_cross),
                _reasoning_arm(arm == ARM_ON),
                backtest_scope(),
            ):
                state = await run_pipeline(
                    ticker, trade_date, as_of=trade_date, language=language, profile=AIProfile.PRODUCTION,
                    force_refresh=True,
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
            return state, original_n_same, original_n_cross, degraded, llm_summary, attempt
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
    language: str,
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
        state, original_n_same, original_n_cross, degraded, llm_summary, attempts_used = await _run_pipeline_with_retry(
            ticker, trade_date, arm, language, attempts, retry_delay
        )
    except Exception as exc:
        elapsed = time.monotonic() - start
        error_path = out_dir / f"{ticker}_{trade_date.isoformat()}_{arm}_r{repeat}_error.json"
        error_path.write_text(
            json.dumps(
                {
                    "ticker": ticker,
                    "date": trade_date.isoformat(),
                    "arm": arm,
                    "pipeline_language": language,
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

    # Isolation tripwire — same reasoning as language_ab._run_one's own
    # comment: with the per-run clears plus _no_retrieved_memory zeroing both
    # retrieval slots, past_context is STRUCTURALLY guaranteed empty on every
    # run. A non-empty value means the override itself regressed, not a
    # benign data-dependent exception — fail fast after writing the record.
    isolation_violation = bool(state.past_context.strip())

    if degraded:
        print(
            f"{logger_prefix} WARNING run {run_idx}/{total} ticker={ticker} date={trade_date} "
            f"arm={arm} rep={repeat}: DEGRADED — {degraded} fell back to placeholder reports "
            f"after {attempts} attempt(s). This run saw less data than a clean one and must be "
            f"excluded from the arm comparison."
        )

    record = {
        "ticker": ticker,
        "date": trade_date.isoformat(),
        # Written into the SAME field name app.experiments.language_ab_report
        # groups/keys by ("language") so its existing rollups (e.g.
        # variance_by_language) work UNCHANGED on these records — read
        # "reason_on"/"reason_off" wherever that report prints a language
        # code. See this module's own docstring.
        "language": arm,
        "arm": arm,
        "research_manager_reasoning": arm == ARM_ON,
        # The REAL, fixed pipeline language — kept separate so it is never
        # lost or confused with the arm alias above.
        "pipeline_language": language,
        "repeat": repeat,
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "elapsed_seconds": elapsed,
        "isolation_violation": isolation_violation,
        # Not comparable across arms when non-empty — see degraded_report_names.
        "degraded": bool(degraded),
        "degraded_reports": degraded,
        "force_refresh": True,
        # Provenance: this run did NOT use the production same-/cross-ticker
        # memory settings. See language_ab._no_retrieved_memory.
        "memory_n_same_override": 0,
        "memory_n_same_production": original_n_same,
        "memory_n_cross_override": 0,
        "memory_n_cross_production": original_n_cross,
        "llm_attempts_used": attempts_used,
        **llm_summary,
        **_state_to_dict(state),
        # Added on top of language_ab's own _state_to_dict output — the #205
        # consensus-override gate record this sweep's go/no-go metric (b)
        # needs (tier fired / gated), and which language_ab's dump omits
        # since that experiment never varies a signal the gate reads.
        "consensus_override": dict(state.consensus_override) if state.consensus_override else {},
    }
    # Atomic write — same reasoning as language_ab._run_one's own comment:
    # --resume treats out_path.exists() as "done", so a write that dies
    # partway must never leave a truncated file at the final name.
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
    confidence = f"{decision.confidence:.2f}" if decision is not None and decision.confidence is not None else "N/A"
    print(
        f"{logger_prefix} run {run_idx}/{total} ticker={ticker} date={trade_date} "
        f"arm={arm} rep={repeat} done rating={rating} conf={confidence} elapsed={elapsed:.0f}s"
    )


def _plan_runs(tickers: list[str], dates: list[date], repeats: int) -> list[tuple[str, date, str, int]]:
    """Run order with the two arms INTERLEAVED, cyclically rotated on each
    repeat (rep1 reason_on,reason_off / rep2 reason_off,reason_on / ...) —
    same elapsed-time/server-health confound rationale as
    `language_ab._plan_runs`'s own docstring, just fixed at exactly 2 arms
    instead of an arbitrary language list."""
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
    parser.add_argument("--dates", required=True, help="Comma-separated YYYY-MM-DD as_of dates.")
    parser.add_argument(
        "--language", default="en",
        help="Pipeline language, FIXED across both arms (default en) — this experiment varies "
             "reasoning, not language; varying both here would re-confound a comparison "
             "app.experiments.language_ab already measured separately.",
    )
    parser.add_argument("--repeats", type=int, default=3, help="Independent repeats per (ticker, date, arm).")
    parser.add_argument(
        "--out-dir", default=None,
        help="Defaults to results/reasoning_ab/<UTC timestamp>/ if not given.",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip any (ticker, date, arm, repeat) whose result file already exists in --out-dir.",
    )
    parser.add_argument(
        "--attempts", type=int, default=3,
        help="Attempts per run, counting both raised errors and degraded (placeholder) analyst "
             "reports as retriable. Default 3, NOT settings.backtest_pipeline_retry_attempts (5).",
    )
    parser.add_argument(
        "--retry-delay", type=float, default=60.0,
        help="Seconds between attempts. Default 60, NOT settings.backtest_pipeline_retry_delay_seconds "
             "(600) — that budget is for unattended multi-hour backtests and would idle a short "
             "pilot for the better part of an hour.",
    )
    args = parser.parse_args()
    if args.attempts < 1:
        parser.error(f"--attempts must be >= 1, got {args.attempts}")
    return args


async def _main() -> None:
    args = _parse_args()
    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    dates = [date.fromisoformat(d.strip()) for d in args.dates.split(",") if d.strip()]
    repeats = args.repeats

    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = Path("results") / "reasoning_ab" / datetime.now().strftime("%Y%m%dT%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = _plan_runs(tickers, dates, repeats)
    total = len(runs)
    print(f"{logger_prefix} {total} run(s) queued -> {out_dir} (pipeline language fixed at {args.language!r})")

    for run_idx, (ticker, trade_date, arm, repeat) in enumerate(runs, start=1):
        await _run_one(
            ticker, trade_date, arm, args.language, repeat, out_dir, run_idx, total, args.resume,
            args.attempts, args.retry_delay,
        )

    print(
        f"{logger_prefix} done. Aggregate with: python -m app.experiments.language_ab_report "
        f"--dir {out_dir}  (reads the arm label from the 'language' field — see module docstring)"
    )


if __name__ == "__main__":
    asyncio.run(_main())
