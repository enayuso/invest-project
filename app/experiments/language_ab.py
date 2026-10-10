"""Korean-vs-English LLM judgment quality A/B runner.

Calls `app.orchestrator.pipeline.run_pipeline` directly, one (ticker, date,
language, repeat) combination at a time — it does NOT reuse
`app.backtest.harness.run_backtest`/`app.backtest.portfolio_harness.
run_portfolio_backtest`. That's a deliberate choice, not an oversight:

  - The harness replays gap trailing-stops, calls `reflect()` (an extra LLM
    call whose OWN output is language-dependent — a second confound stacked
    on top of the one this experiment is trying to isolate), resolves
    outcomes, and builds an equity curve. None of that is needed to compare
    two languages' judgment quality on the SAME (ticker, date) input, and
    every one of those extra steps is itself a place language could leak
    into the comparison.
  - Threading `language` through the harness would mean changing
    `run_backtest` -> `_process_ticker_date` -> `_run_pipeline_with_retry`'s
    shared signatures — production code `app.backtest.portfolio_harness`
    also calls into. Not worth touching for a research script.
  - The harness deliberately lets run N+1 of a replay learn from run N's own
    memory_entries (sequential, "the replay learns from itself" — see
    `app.memory.store.get_similar_context`'s `live_only_clause`). This
    experiment wants the OPPOSITE: every run must be independent of every
    other run, Korean and English included, or a language's results would
    partly reflect what the OTHER language's earlier run wrote to memory,
    not that language's own judgment quality.

Isolation, per run, in this exact order:

    1. `clear_backtest_positions([ticker])` — wipes `trades` so the run
       starts flat (see that function's own docstring).
    2. `clear_backtest_memory([ticker])` — wipes THIS EXPERIMENT'S OWN
       `is_backtest=true` `memory_entries` rows so a same-ticker/date Korean
       run followed by an English run can't "read" the Korean run's own
       decision text as a retrieved past case. This does NOT, on its own,
       stop `get_similar_context` (which does NOT filter `is_backtest` rows
       when `as_of` is set — see that function's own docstring) from
       retrieving a ticker's REAL, pre-existing LIVE (`is_backtest=false`)
       trading history — `_no_retrieved_memory()` below (zeroing
       `memory_n_same` too, not just `memory_n_cross`) is what actually
       closes that gap; see its own docstring for the confirmed 16/40
       stage-1 runs this fixed.
    3. `run_pipeline(..., as_of=trade_date, ...)` inside `backtest_scope()`.
       `as_of` is NOT optional here: without it `run_pipeline` runs LIVE,
       which can place a real KIS paper order — this experiment must never
       do that as a side effect of comparing prose.

Each run's full `PipelineState` is written to its own JSON file the moment
it finishes (`<ticker>_<date>_<lang>_r<repeat>.json` in `--out-dir`) — the
file's existence IS the checkpoint, so `--resume` after a kill (see project
memory: server-power-schedule-blocker — this box loses power) just means
"skip any (ticker, date, language, repeat) whose file is already there."
A run that raises is recorded as `..._error.json` instead and the runner
moves on to the next one; one bad run must never take down an unattended
multi-hour sweep.

    python -m app.experiments.language_ab \\
        --tickers 005930.KS,AAPL --dates 2026-07-01,2026-07-15 \\
        --languages ko,en --repeats 3

Aggregate the results afterwards with `app.experiments.language_ab_report`.
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
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

from app.config import AIProfile, backtest_scope, settings
from app.execution.ledger import clear_backtest_positions
from app.llm import metrics as llm_metrics
from app.memory.store import clear_backtest_memory
from app.orchestrator.pipeline import run_pipeline
from app.state import PipelineState

if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

logger = logging.getLogger(__name__)
logger_prefix = "[language_ab]"


def _run_filename(ticker: str, trade_date: date, language: str, repeat: int) -> str:
    return f"{ticker}_{trade_date.isoformat()}_{language}_r{repeat}.json"


def _to_jsonable(value: Any) -> Any:
    """Recursively turns a pipeline state field into something `json.dumps`
    accepts: pydantic models (`ResearchPlan`, `TraderProposal`,
    `PortfolioDecision`/`FlatPortfolioDecision`) via `model_dump(mode="json")`
    (handles nested enums/None on its own), dataclasses (`DebateState`,
    `DebateTurn`) via `dataclasses.asdict`, everything else passed through
    as-is for `json.dumps`'s own `default=str` fallback to handle."""
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    return value


def _state_to_dict(state: PipelineState) -> dict[str, Any]:
    """Everything the spec calls for: the full transcript (both debates),
    every analyst report, the trader proposal, the final decision's every
    field, `past_context` (the isolation-verification field — see module
    docstring), `decision_id`, and `rebalance_adjustment_note`."""
    return {
        "instrument_context": state.instrument_context,
        "reports": dict(state.reports),
        "research_debate": _to_jsonable(state.research_debate),
        "research_plan": _to_jsonable(state.research_plan),
        "trader_proposal": _to_jsonable(state.trader_proposal),
        "risk_debate": _to_jsonable(state.risk_debate),
        "past_context": state.past_context,
        "final_decision": _to_jsonable(state.final_decision),
        "decision_id": state.decision_id,
        "rebalance_adjustment_note": state.rebalance_adjustment_note,
    }


# `app.orchestrator.pipeline` degrades a failed analyst to a placeholder
# report rather than aborting the run (pipeline.py's `any_failed` branch), so
# a run whose data fetches timed out still completes and still produces a
# final decision — it just produced it from two real reports and two stubs.
# For this experiment that is worse than a crash: if the Korean arm got four
# real analyst reports and the English arm got two, the comparison measures
# data availability, not language. Runs that hit this are retried and, if
# they degrade again, recorded as `degraded` for the report to exclude.
#
# Matching on the placeholder's literal text is deliberate. The pipeline
# builds it as an f-string with no marker of its own, and it is hardcoded
# KOREAN regardless of the run's `language` — so this same substring
# identifies a degraded report in both arms.
_DEGRADED_REPORT_MARKER = "리포트를 생성하지 못했습니다"


def degraded_report_names(reports: dict[str, str]) -> list[str]:
    """Analyst names whose report is a failure placeholder, not real analysis."""
    return sorted(name for name, text in reports.items() if _DEGRADED_REPORT_MARKER in text)


@contextmanager
def _no_retrieved_memory() -> Iterator[tuple[int, int]]:
    """Temporarily sets BOTH `settings.memory_n_same` and `memory_n_cross` to
    0, yielding the originals `(n_same, n_cross)`, and always restores them.

    `clear_backtest_memory([ticker])` is NOT enough on its own, in TWO
    separate ways this run must guard against:

    1. Cross-ticker: `app.memory.store.get_similar_context` runs three
       queries, and two of them (`cross_rows_same_market`,
       `cross_rows_other_market`) select on `ticker != $1` — OTHER tickers'
       rows, which a per-ticker clear by definition cannot touch. In
       backtest mode those queries don't filter `is_backtest` either (see
       that function's `live_only_clause`), so they retrieve leftover replay
       memory from every backtest this database has ever run.

       Measured on this database (2026-08-14): 218 retrievable
       (`status='resolved' AND NOT flagged_bad`) `is_backtest` rows across 17
       tickers — 005930.KS 23, 000660.KS 22, 373220.KS 21, 005380.KS 21, and
       so on. With `memory_n_cross` at its default 3, a domestic pilot
       ticker fills all three cross slots on every single run. The pilot
       ticker 009540.KS isn't in that set at all, so its own per-ticker
       clear removes nothing while the cross queries still deliver three
       retrieved cases.

       That is fatal to THIS experiment specifically, because every one of
       those stored entries is Korean prose. Leaving the cross queries on
       means the English arm reasons in English while reading Korean past
       lessons, and the Korean arm reads Korean lessons under a Korean
       instruction — an asymmetry that penalises the English arm for
       reasons that have nothing to do with the model's English reasoning
       ability. A "Korean wins" result would be uninterpretable.

    2. Same-ticker: an EARLIER version of this contextmanager left
       `memory_n_same` alone, reasoning that `clear_backtest_memory([ticker])`
       already handled same-ticker retrieval — WRONG for any ticker with
       real LIVE trading history, since that clear only ever deletes
       `is_backtest=true` rows, never the ticker's actual `is_backtest=false`
       history, which `get_similar_context`'s same-ticker query retrieves
       unfiltered under `as_of` exactly like the cross-ticker queries above.
       Confirmed live (2026-08-14, stage-1 run): 16/40 runs came back with
       non-empty `past_context` — every single run for 005930.KS and AAPL
       (both have real live trading history in this database), both dates,
       both languages, both repeats. The other 3 tickers (005490.KS,
       055550.KS, XOM) have no live history, so they were clean by
       coincidence, not because the isolation actually held.

    Zeroing both slots (rather than DELETEing real trading history, which is
    destructive and irreversible) removes both confounds identically for
    every arm and leaves the database untouched — `get_similar_context`'s
    three queries all return nothing regardless of `is_backtest` or what
    real history exists for that ticker.

    Consequence worth relying on: with this active, `past_context` must come
    back empty on every run, structurally, not by luck. `_run_one` treats a
    non-empty value as an isolation failure, not as an expected case.
    """
    orig_same, orig_cross = settings.memory_n_same, settings.memory_n_cross
    settings.memory_n_same = 0
    settings.memory_n_cross = 0
    try:
        yield orig_same, orig_cross
    finally:
        settings.memory_n_same = orig_same
        settings.memory_n_cross = orig_cross


def _summarize_llm_metrics(records: list[llm_metrics.LLMCallMetric]) -> dict[str, Any]:
    """Rolls up one attempt's drained `LLMCallMetric` records into the fields
    written to the result JSON (module spec Part A2). `total_tokens` etc. can
    be `None` per-record when the backend didn't report usage, so every sum
    below treats `None` as 0 rather than letting one unreported call blank
    out the whole run's total.

    `llm_metrics_truncated` is a *possible*-loss flag, not a confirmed one:
    it fires when this attempt alone drained >= the buffer's cap
    (`app.llm.metrics._MAX_RECORDS`, 2000), which is the only case `drain()`
    itself cannot rule out eviction for (see its own docstring) — a single
    attempt would have to make 2000+ LLM calls, astronomically more than a
    normal pipeline run, but the flag exists so that scenario is visible
    instead of silently under-counted.
    """
    by_label: dict[str, dict[str, int]] = {}
    for m in records:
        entry = by_label.setdefault(m.label, {"calls": 0, "total_tokens": 0})
        entry["calls"] += 1
        entry["total_tokens"] += m.total_tokens or 0
    return {
        "llm_calls": len(records),
        "llm_prompt_tokens": sum(m.prompt_tokens or 0 for m in records),
        "llm_completion_tokens": sum(m.completion_tokens or 0 for m in records),
        "llm_total_tokens": sum(m.total_tokens or 0 for m in records),
        "llm_fallback_calls": sum(1 for m in records if m.fallback_used),
        # Same pattern as llm_fallback_calls above — see LLMCallMetric.
        # finish_reason's own comment for why a truncated response otherwise
        # passes downstream indistinguishable from a complete one.
        "llm_truncated_calls": sum(1 for m in records if m.finish_reason == "length"),
        "llm_by_label": by_label,
        "llm_metrics_truncated": len(records) >= llm_metrics.buffer_capacity(),
    }


async def _run_pipeline_with_retry(
    ticker: str, trade_date: date, language: str, attempts: int, retry_delay: float
) -> tuple[PipelineState, int, int, list[str], dict[str, Any], int]:
    """Same retry shape as `app.backtest.harness._run_pipeline_with_retry`
    (reusing `settings.backtest_pipeline_retry_attempts`/
    `backtest_pipeline_retry_delay_seconds` — see that function's own
    docstring for why a vLLM outage needs a retry budget this long), but
    NOT that function itself: this one threads `language` through and
    re-runs the isolation clears (step 1/2 of the module docstring's
    ordering) on every attempt, since a failed attempt may have partially
    written state a retry must not inherit either.

    `attempts`/`retry_delay` come from the CLI rather than
    `settings.backtest_pipeline_retry_*` (5 x 600s). That budget is tuned for
    an unattended multi-hour backtest riding out a vLLM outage; applied here
    it means a single bad run can idle a 20-minute pilot for 50 minutes doing
    nothing, which is how the first pilot attempt was spent.

    A DEGRADED result (see `_DEGRADED_REPORT_MARKER`) is retried the same way
    a raised exception is — the pipeline swallows analyst failures instead of
    raising, so without this check a run starved of data would be recorded as
    a clean success. If the last attempt still degrades, the state is returned
    with the offending analyst names so the caller can record them; throwing
    the run away entirely would lose the transcript that explains why.

    Returns `(state, original_memory_n_same, original_memory_n_cross,
    degraded_analyst_names, llm_metrics_summary, attempts_used)`. The
    second/third elements are provenance for the result JSON, so a later
    reader can confirm the override in `_no_retrieved_memory` was actually
    active for this run and see what values it would otherwise have used.

    `llm_metrics_summary`/`attempts_used` are Part A of the token-vs-retry
    investigation (see module memory: the ko slowdown could be more retries
    or more tokens per call, and elapsed time alone can't tell them apart).
    `app.llm.metrics.drain()` is called at the START of every attempt —
    discarding whatever a PRIOR failed/degraded attempt already recorded —
    so that a retried run's cost never gets counted against the language.
    `drain()` empties the module-level ring buffer `app.llm.metrics` — this
    is safe ONLY because this module always runs as its own standalone
    `python -m app.experiments.language_ab` process, never imported into the
    dashboard process. If it ever were, this drain() would steal whatever
    `/metrics` was about to report from unrelated production traffic — do
    not import this module into a long-lived process without addressing
    that first.
    Only the attempt that actually gets returned (a clean success, or the
    last attempt if every attempt degraded) has its post-call `drain()`
    kept; that is deliberately "the cost of the attempt that succeeded," not
    "the cost of the whole retry budget" — see `_summarize_llm_metrics`."""
    for attempt in range(1, attempts + 1):
        llm_metrics.drain()  # discard any prior attempt's records before this one starts
        try:
            await clear_backtest_positions([ticker])
            await clear_backtest_memory([ticker])
            # _no_retrieved_memory closes both the other-ticker retrieval path
            # AND the same-ticker real-history path a per-ticker clear cannot
            # reach — see its own docstring.
            with _no_retrieved_memory() as (original_n_same, original_n_cross), backtest_scope():
                state = await run_pipeline(
                    ticker, trade_date, as_of=trade_date, language=language, profile=AIProfile.PRODUCTION,
                    # Bypasses pipeline.py's 15-minute in-process analyst-report
                    # cache, which is keyed (ticker, date, as_of, language) —
                    # exactly this experiment's repeat key. Without it, repeats
                    # 2 and 3 of a language reuse repeat 1's analyst reports, so
                    # the measured run-to-run variance covers only the debate
                    # and decision stages. That variance IS the noise floor
                    # every ko-vs-en difference has to clear, and understating
                    # it makes differences look more significant than they are.
                    force_refresh=True,
                )
        except Exception:
            if attempt == attempts:
                raise
            logger.warning(
                "%s %s %s: pipeline call failed (attempt %d/%d) — retrying in %.0fs",
                ticker, trade_date, language, attempt, attempts, retry_delay, exc_info=True,
            )
            await asyncio.sleep(retry_delay)
            continue

        degraded = degraded_report_names(state.reports)
        if not degraded or attempt == attempts:
            llm_summary = _summarize_llm_metrics(llm_metrics.drain())
            return state, original_n_same, original_n_cross, degraded, llm_summary, attempt
        logger.warning(
            "%s %s %s: analyst report(s) degraded to placeholders %s (attempt %d/%d) — retrying in %.0fs",
            ticker, trade_date, language, degraded, attempt, attempts, retry_delay,
        )
        await asyncio.sleep(retry_delay)
    raise AssertionError("unreachable")  # loop always returns or raises on the last attempt


async def _run_one(
    ticker: str,
    trade_date: date,
    language: str,
    repeat: int,
    out_dir: Path,
    run_idx: int,
    total: int,
    resume: bool,
    attempts: int,
    retry_delay: float,
) -> None:
    out_path = out_dir / _run_filename(ticker, trade_date, language, repeat)
    if resume and out_path.exists():
        print(
            f"{logger_prefix} run {run_idx}/{total} ticker={ticker} date={trade_date} "
            f"lang={language} rep={repeat} skipped (--resume, already done)"
        )
        return

    print(
        f"{logger_prefix} run {run_idx}/{total} ticker={ticker} date={trade_date} "
        f"lang={language} rep={repeat} starting..."
    )
    started_at = datetime.now()
    start = time.monotonic()
    try:
        state, original_n_same, original_n_cross, degraded, llm_summary, attempts_used = await _run_pipeline_with_retry(
            ticker, trade_date, language, attempts, retry_delay
        )
    except Exception as exc:
        elapsed = time.monotonic() - start
        error_path = out_dir / f"{ticker}_{trade_date.isoformat()}_{language}_r{repeat}_error.json"
        error_path.write_text(
            json.dumps(
                {
                    "ticker": ticker,
                    "date": trade_date.isoformat(),
                    "language": language,
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
            f"lang={language} rep={repeat} FAILED elapsed={elapsed:.0f}s error={exc!r}"
        )
        return

    elapsed = time.monotonic() - start
    finished_at = datetime.now()

    # Isolation tripwire. With the per-run clears (module docstring, steps
    # 1-2) closing the same-ticker path and _no_retrieved_memory zeroing BOTH
    # memory_n_same and memory_n_cross, there is no remaining route by which
    # get_similar_context can return anything — past_context is now
    # STRUCTURALLY guaranteed empty on every run (zeroed retrieval slots
    # return nothing regardless of what real history exists), not merely
    # "expected" empty. A non-empty value therefore means the override itself
    # regressed (e.g. a future edit re-widens memory_n_same/_cross without
    # updating this experiment), not a benign data-dependent exception —
    # fail fast (raise below, after writing the record so the transcript
    # survives for diagnosis) rather than silently burning the rest of an
    # unattended multi-hour sweep on incomparable runs, which is exactly
    # what happened to 16/40 stage-1 runs before this fix existed.
    isolation_violation = bool(state.past_context.strip())

    if degraded:
        print(
            f"{logger_prefix} WARNING run {run_idx}/{total} ticker={ticker} date={trade_date} "
            f"lang={language} rep={repeat}: DEGRADED — {degraded} fell back to placeholder reports "
            f"after {attempts} attempt(s). This run saw less data than a clean one and must be "
            f"excluded from the language comparison."
        )

    record = {
        "ticker": ticker,
        "date": trade_date.isoformat(),
        "language": language,
        "repeat": repeat,
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "elapsed_seconds": elapsed,
        "isolation_violation": isolation_violation,
        # Not comparable across arms when non-empty — see degraded_report_names.
        "degraded": bool(degraded),
        "degraded_reports": degraded,
        # Recorded because it is a precondition of the repeat-variance metric,
        # not an incidental setting — see the force_refresh comment above.
        "force_refresh": True,
        # Provenance: this run did NOT use the production same-/cross-ticker
        # memory settings. Recorded so a later reader can't mistake these
        # results for a production-config run — see _no_retrieved_memory.
        "memory_n_same_override": 0,
        "memory_n_same_production": original_n_same,
        "memory_n_cross_override": 0,
        "memory_n_cross_production": original_n_cross,
        # Part A — the successful attempt's own LLM call cost, so a later
        # reader can tell "more retries" from "more tokens per call" apart
        # instead of just seeing one slower elapsed_seconds. See
        # _summarize_llm_metrics and _run_pipeline_with_retry's docstring.
        "llm_attempts_used": attempts_used,
        **llm_summary,
        **_state_to_dict(state),
    }
    # Atomic write: --resume (line 348) treats out_path.exists() as "done",
    # so a write that dies partway (power loss, kill -9 mid multi-hour sweep)
    # must never leave a truncated file at the final name — that would get
    # permanently skipped on resume and crash the whole report (load_records
    # has no per-file error isolation). Write to a same-directory temp file
    # and os.replace it in — replace is atomic on both POSIX and Windows
    # (unlike os.rename, which POSIX guarantees atomic but Windows does not).
    tmp_path = out_path.with_suffix(out_path.suffix + f".tmp{os.getpid()}")
    tmp_path.write_text(json.dumps(record, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    os.replace(tmp_path, out_path)
    if isolation_violation:
        # Record is on disk (diagnosable) — now abort the whole sweep rather
        # than continue producing more incomparable runs. See the tripwire
        # comment above for why this is a regression, not a benign case.
        raise RuntimeError(
            f"memory isolation FAILED for {ticker} {trade_date} lang={language} rep={repeat} — "
            f"past_context was non-empty despite memory_n_same=memory_n_cross=0. Result written to "
            f"{out_path}; see _no_retrieved_memory's docstring before re-running."
        )

    decision = state.final_decision
    rating = decision.rating.value if decision is not None else "N/A"
    confidence = f"{decision.confidence:.2f}" if decision is not None and decision.confidence is not None else "N/A"
    print(
        f"{logger_prefix} run {run_idx}/{total} ticker={ticker} date={trade_date} "
        f"lang={language} rep={repeat} done rating={rating} conf={confidence} elapsed={elapsed:.0f}s"
    )


def _plan_runs(
    tickers: list[str], dates: list[date], languages: list[str], repeats: int
) -> list[tuple[str, date, str, int]]:
    """Run order with the languages INTERLEAVED, cyclically rotated on each
    repeat (2 langs: rep1 ko,en / rep2 en,ko / rep3 ko,en / ...; 3 langs:
    rep1 a,b,c / rep2 b,c,a / rep3 c,a,b / ...).

    The obvious nesting — all of one language's repeats, then all of the
    other's — silently confounds language with elapsed time. This box loses
    power and degrades under sustained load (project memory:
    server-power-schedule-blocker), and the vLLM endpoint has its own
    connection-instability history (llm-model-swap-gpt-oss-issue), so a
    language that always runs first gets a cold, healthy server and a
    language that always runs last gets a hot, possibly degrading one. Any
    drift over the session would then be read as a language effect — the
    elapsed-time metric most of all. Interleaving spreads both arms evenly
    across the session; the cyclic rotation (not a plain reverse) removes
    the residual "ko is always the first of the pair" bias for 2 languages
    AND the "the middle language never moves" gap a reverse leaves for 3+.

    Still not perfectly balanced when `repeats` isn't a multiple of
    `len(languages)` — e.g. the default `--repeats 3` with 2 languages puts
    ko first on 2 of 3 repeats, en on 1 — that residual is a property of
    deterministic rotation over a count that doesn't divide evenly, not a
    bug; use a `--repeats` that's a multiple of the language count to
    eliminate it entirely.

    Result filenames encode (ticker, date, language, repeat), so `--resume`
    is unaffected by this ordering.
    """
    runs: list[tuple[str, date, str, int]] = []
    for ticker in tickers:
        for trade_date in dates:
            for repeat in range(1, repeats + 1):
                offset = (repeat - 1) % len(languages)
                ordered = languages[offset:] + languages[:offset]
                runs.extend((ticker, trade_date, language, repeat) for language in ordered)
    return runs


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tickers", required=True, help="Comma-separated ticker universe.")
    parser.add_argument("--dates", required=True, help="Comma-separated YYYY-MM-DD as_of dates.")
    parser.add_argument("--languages", default="ko,en", help="Comma-separated language codes (default ko,en).")
    parser.add_argument("--repeats", type=int, default=3, help="Independent repeats per (ticker, date, language).")
    parser.add_argument(
        "--out-dir", default=None,
        help="Defaults to results/language_ab/<UTC timestamp>/ if not given.",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip any (ticker, date, language, repeat) whose result file already exists in --out-dir.",
    )
    parser.add_argument(
        "--attempts", type=int, default=3,
        help="Attempts per run, counting both raised errors and degraded (placeholder) analyst "
             "reports as retriable. Default 3, NOT settings.backtest_pipeline_retry_attempts (5).",
    )
    parser.add_argument(
        "--retry-delay", type=float, default=60.0,
        help="Seconds between attempts. Default 60, NOT settings.backtest_pipeline_retry_delay_seconds "
             "(600) — that budget is for unattended multi-hour backtests and would idle a short pilot "
             "for the better part of an hour.",
    )
    args = parser.parse_args()
    if args.attempts < 1:
        # _run_pipeline_with_retry's loop is range(1, attempts + 1) — 0 or
        # negative makes it a no-op, falling through to a bare
        # `raise AssertionError("unreachable")` that reads like an internal
        # bug rather than a CLI usage error.
        parser.error(f"--attempts must be >= 1, got {args.attempts}")
    return args


async def _main() -> None:
    args = _parse_args()
    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    dates = [date.fromisoformat(d.strip()) for d in args.dates.split(",") if d.strip()]
    languages = [lang.strip() for lang in args.languages.split(",") if lang.strip()]
    repeats = args.repeats

    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = Path("results") / "language_ab" / datetime.now().strftime("%Y%m%dT%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = _plan_runs(tickers, dates, languages, repeats)
    total = len(runs)
    print(f"{logger_prefix} {total} run(s) queued -> {out_dir}")

    for run_idx, (ticker, trade_date, language, repeat) in enumerate(runs, start=1):
        await _run_one(
            ticker, trade_date, language, repeat, out_dir, run_idx, total, args.resume,
            args.attempts, args.retry_delay,
        )

    print(f"{logger_prefix} done. Aggregate with: python -m app.experiments.language_ab_report --dir {out_dir}")


if __name__ == "__main__":
    asyncio.run(_main())
