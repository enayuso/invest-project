"""Market Analyst long-horizon-block prompt-fix (2026-09-16, engine audit
follow-up — priority item 1) mechanism A/B runner. Measures whether
unconditionally injecting a 52-week weekly-bar price-context block (see
`app.dataflows.long_horizon`) actually changes what the Market Analyst
writes about multi-week/multi-month price action, via a paired before/after
comparison of the SAME (ticker, as_of) input.

## Why A/B, not a retroactive rescan (see [[engine-audit-2026-09-15]])

This is a DATA-SHAPE change, not a factual-error removal like the
billion/eok label fix — the report text itself changes shape between arms,
so a stored transcript can't be rescored after the fact the way a fixed
guard regex can be. Unlike `app.experiments.candle_ab`'s guard (which can
only fire while `session_in_progress` is True, live-only by construction),
this fix has no such live-only gate: the long-horizon block is built from
weekly bars and injected unconditionally regardless of session state, so —
like `reasoning_ab`/`language_ab` and UNLIKE `candle_ab` — this runs in
BACKTEST mode (`as_of=trade_date`, real historical dates, any time of day).

## Arm construction

`get_long_horizon_context` is imported exactly once into production code:
`app.agents.market_analyst` does `from app.dataflows.yfinance import ...
get_long_horizon_context ...` (a `from X import Y` binding), then `analyze()`
calls the bare module-global name at call time. Patching
`app.dataflows.yfinance.get_long_horizon_context` (the original provider)
would leave market_analyst's own binding untouched — same trap `candle_ab`'s
docstring warns about for `intraday_narration_note` — so this patches
`app.agents.market_analyst.get_long_horizon_context` directly (see
`_horizon_arm` below) and asserts the patch stuck before ever calling
`run_pipeline`.

  - arm `after` (post-fix, current code): wraps the real function, calls it
    unchanged, and captures whatever it returned.
  - arm `before` (pre-fix reconstruction): wraps a stub that always returns
    "" (no long-horizon data at all — the state before this fix existed),
    AND strips `_LONG_HORIZON_CLAUSE` out of
    `app.agents.market_analyst._BASE_SYSTEM_PROMPT` for the duration of the
    call — that prompt sentence is plain module data referenced by its own
    module's code (not a `from X import Y` binding), so patching the
    attribute directly is sufficient, no `pipeline_module`-style redirection
    needed. Raises if the clause is expected but absent, same
    "a future wording edit must break this loudly" contract as
    `candle_ab._before_note`.

Both arms wrap the SAME real (or stubbed) call so the block actually seen by
the Market Analyst in each arm is captured verbatim (`captured["block"]`) —
metrics are computed from what the model was actually given, not from a
value recomputed after the fact.

## Metrics

M1/M2 (primary — see this module's `long_horizon_claim_metrics`): sentences
in the Market Analyst's report that reference a period longer than one month
(52-week, N-month, 개월, year-to-date, ...) are extracted, then checked for
whether a number in that sentence matches (within tolerance) a number that
actually appears in the injected long-horizon block — "grounded" — and for
hedge words ('approximately'/'roughly'/'likely'/'추정'/'about') — "hedged".
Expectation per the design: arm `before` should show ~0 grounded claims
(there is nothing to ground against) and arm `after` should show most
long-horizon claims grounded and unhedged. If that contrast does not appear,
the injection mechanism itself is broken — read the failure as a plumbing
bug, not as "the fix has no effect" (same discipline candle_ab's own
docstring calls for on `note_has_candle_clause`).

M3 (tripwire only, NOT a verdict metric): paired rating change rate. Known
noise floor from `language_ab` (~80% en / ~60% ko self-inconsistency across
repeats of the SAME input) means a run of this size can only ever catch a
LARGE, badly-broken shift — never read a small M3 delta as either a pass or
a fail on its own.

M4: per-decision token delta between arms (the new block plus the
get_indicators/get_investor_flow tool-list removal are expected to roughly
offset — see the market_analyst commit message).

M5: paired `technical_claim_violations`/`unit_scale_violations` counts —
existing guards, not new ones (10월 `guard_alpha` 전까지 신규 가드 추가
중단 권고, [[fix-without-verification-loop-2026-09-15]]). If the new block's
wording trips either guard more than the `before` arm, that is a prompt-
wording bug in the block/clause text to fix, not a guard bug.

Reuses `app.experiments.language_ab`'s isolation machinery wholesale — see
that module's own docstrings for why each piece exists. `language` is FIXED
across both arms (this experiment varies the long-horizon block, not
language).

    python -m app.experiments.horizon_ab \\
        --tickers 005930.KS,000660.KS,AAPL --dates 2026-08-12,2026-08-26,2026-09-09 --repeats 1

No dedicated aggregator script — read results record-by-record (`long_horizon_*`
fields) the same way `candle_ab`'s own docstring recommends for its pilot,
rather than through `language_ab_report` (whose rollups are keyed for the
language/reasoning axes, not this one)."""

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

import app.agents.market_analyst as market_analyst
from app.config import AIProfile, backtest_scope
from app.execution.ledger import clear_backtest_positions
from app.experiments.language_ab import (
    _no_retrieved_memory,
    _state_to_dict,
    _summarize_llm_metrics,
    degraded_report_names,
)
from app.dataflows.yfinance import get_long_horizon_context as _real_get_long_horizon_context
from app.llm import metrics as llm_metrics
from app.memory.store import clear_backtest_memory
from app.orchestrator.pipeline import run_pipeline
from app.state import PipelineState

if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

logger = logging.getLogger(__name__)
logger_prefix = "[horizon_ab]"

ARM_BEFORE = "before"
ARM_AFTER = "after"
_ARMS = (ARM_BEFORE, ARM_AFTER)

# Byte-identical to the sentence appended to `_BASE_SYSTEM_PROMPT` by the
# 2026-09-16 market_analyst commit — copied from app/agents/market_analyst.py
# rather than retyped from memory, so a stray whitespace/wording drift there
# can't silently make the `before` arm's str.replace a no-op (which would
# make `before` and `after` prompts identical and this pilot's mechanism
# check pass for the wrong reason).
_LONG_HORIZON_CLAUSE = (
    " A longer-horizon price-context block is provided below, built from "
    "weekly bars covering up to the last 52 weeks. It is the ONLY multi-week "
    "history you have — the daily OHLCV table from get_stock_data covers "
    "just the last 10 sessions and cannot support any statement about a "
    "multi-week or multi-month move. Use the long-horizon block for the "
    "52-week range, distance from the 52-week high or low, and the 1/3/6/12-"
    "month trend, and quote its figures as given. Do not infer a 52-week "
    "extreme, a multi-month trend, a support or resistance level, or a "
    "moving-average crossover history from the 10-session daily table, and "
    "do not state such a figure hedged with 'approximately', 'roughly', "
    "'likely' or '추정'. If the block says NOT AVAILABLE, say plainly that no "
    "longer-horizon price context was available this cycle and confine your "
    "read to the last 10 sessions and the verified indicator values."
)

# Sentences containing one of these are candidate multi-week/multi-month
# price claims — the ONLY kind of claim the long-horizon block can possibly
# ground (a same-day or few-day move needs no 52-week context at all).
_LONG_HORIZON_KEYWORD_RE = re.compile(
    r"52[- ]week|52주|\b[0-9]+[- ]month\b|[0-9]+개월|year-to-date|multi-month|multi-week",
    re.IGNORECASE,
)
_HEDGE_WORDS = ("approximately", "roughly", "likely", "추정", "about")
_NUMBER_RE = re.compile(r"[-+]?[\d,]+(?:\.\d+)?")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _numbers(text: str) -> list[float]:
    out = []
    for m in _NUMBER_RE.findall(text):
        try:
            out.append(float(m.replace(",", "")))
        except ValueError:  # pragma: no cover - regex only matches numeric-shaped strings
            continue
    return out


def long_horizon_claim_metrics(report: str, block: str) -> dict[str, int]:
    """M1/M2 — see this module's docstring. `block` is the exact text the
    Market Analyst actually received this run (empty string in arm
    `before`, the real 52-week block in arm `after`); a claim sentence is
    "grounded" if at least one of its numbers matches a number in `block`
    within a tolerance of 0.5% of the block value (or 0.05 absolute for
    small/percentage figures, so a rounding difference like "58.7%" vs
    "58.70%" still counts)."""
    block_numbers = _numbers(block)
    sentences = [s for s in _SENTENCE_SPLIT_RE.split(report) if _LONG_HORIZON_KEYWORD_RE.search(s)]
    grounded = 0
    hedged = 0
    for s in sentences:
        claim_numbers = _numbers(s)
        if any(
            any(abs(n - bn) <= max(0.05, abs(bn) * 0.005) for bn in block_numbers)
            for n in claim_numbers
        ):
            grounded += 1
        if any(h in s.lower() for h in _HEDGE_WORDS):
            hedged += 1
    return {
        "long_horizon_claim_sentences": len(sentences),
        "long_horizon_grounded_sentences": grounded,
        "long_horizon_hedged_sentences": hedged,
    }


async def _after_long_horizon(ticker: str, as_of: date | None = None) -> str:
    """The real, current (post-fix) function, called unchanged. Calls the
    module-level `_real_get_long_horizon_context` reference captured at
    import time (NOT `market_analyst.get_long_horizon_context`, which by the
    time this runs has already been overwritten by `_horizon_arm`'s own
    wrapper — calling that would recurse into itself)."""
    return await _real_get_long_horizon_context(ticker, as_of=as_of)


async def _before_long_horizon(ticker: str, as_of: date | None = None) -> str:
    """Pre-fix reconstruction: no long-horizon data existed at all."""
    return ""


@contextmanager
def _horizon_arm(arm: str, captured: dict[str, Any]) -> Iterator[None]:
    """Patches `app.agents.market_analyst.get_long_horizon_context` (the
    module's OWN `from X import Y` binding — see this module's docstring for
    why patching `app.dataflows.yfinance`'s copy would not affect
    market_analyst at all) to a wrapper that captures whatever the arm's
    function actually returned, and — for arm `before` only — also strips
    `_LONG_HORIZON_CLAUSE` from `_BASE_SYSTEM_PROMPT` (plain module data,
    not an import binding, so patching the attribute directly is enough).
    Restores both unconditionally in `finally`, same reasoning as
    `candle_ab._candle_arm`/`reasoning_ab._reasoning_arm`: a leaked patch
    would silently contaminate every later run of an unattended sweep."""
    base = _before_long_horizon if arm == ARM_BEFORE else _after_long_horizon

    async def _wrapped(ticker: str, as_of: date | None = None) -> str:
        block = await base(ticker, as_of=as_of)
        captured["block"] = block
        return block

    original_fn = market_analyst.get_long_horizon_context
    original_prompt = market_analyst._BASE_SYSTEM_PROMPT
    market_analyst.get_long_horizon_context = _wrapped
    if arm == ARM_BEFORE:
        if _LONG_HORIZON_CLAUSE not in original_prompt:
            raise AssertionError(
                "horizon_ab: _LONG_HORIZON_CLAUSE not found in _BASE_SYSTEM_PROMPT — "
                "app.agents.market_analyst's wording has drifted since this experiment's "
                "_LONG_HORIZON_CLAUSE constant was copied from it. Update the constant (never "
                "silently fall back to leaving the prompt unmodified — see this function's own "
                "docstring)."
            )
        market_analyst._BASE_SYSTEM_PROMPT = original_prompt.replace(_LONG_HORIZON_CLAUSE, "")
    assert market_analyst.get_long_horizon_context is _wrapped, (
        "horizon_ab: patching app.agents.market_analyst.get_long_horizon_context did not stick"
    )
    try:
        yield
    finally:
        market_analyst.get_long_horizon_context = original_fn
        market_analyst._BASE_SYSTEM_PROMPT = original_prompt


def _run_filename(ticker: str, trade_date: date, arm: str, repeat: int) -> str:
    return f"{ticker}_{trade_date.isoformat()}_{arm}_r{repeat}.json"


async def _run_pipeline_with_retry(
    ticker: str, trade_date: date, arm: str, language: str, attempts: int, retry_delay: float,
) -> tuple[PipelineState, int, int, list[str], dict[str, Any], int, str]:
    """Same retry/isolation shape as `reasoning_ab._run_pipeline_with_retry`,
    with the long-horizon arm as the varying axis. Captures the block the
    wrapper actually returned (`block` in the return tuple) for the caller
    to record — the mechanism proof (M1/M2 above) needs the block as SEEN by
    the Market Analyst on the attempt that was kept, not one recomputed
    after the fact."""
    captured: dict[str, Any] = {}

    for attempt in range(1, attempts + 1):
        llm_metrics.drain()  # discard any prior attempt's records before this one starts
        captured.clear()
        try:
            await clear_backtest_positions([ticker])
            await clear_backtest_memory([ticker])
            with (
                _no_retrieved_memory() as (original_n_same, original_n_cross),
                _horizon_arm(arm, captured),
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
            return (
                state, original_n_same, original_n_cross, degraded, llm_summary, attempt,
                captured.get("block", ""),
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
        (
            state, original_n_same, original_n_cross, degraded, llm_summary, attempts_used, block,
        ) = await _run_pipeline_with_retry(ticker, trade_date, arm, language, attempts, retry_delay)
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

    market_report = state.reports.get("Market Analyst", "")
    claim_metrics = long_horizon_claim_metrics(market_report, block)

    record = {
        "ticker": ticker,
        "arm": arm,
        "repeat": repeat,
        "trade_date": trade_date.isoformat(),
        "pipeline_language": language,
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
        # Mechanism proof (M1/M2, see module docstring) — recorded from the
        # block the patched wrapper actually returned during this run's kept
        # attempt, not recomputed after the fact.
        "long_horizon_block": block,
        "long_horizon_block_injected": bool(block),
        **claim_metrics,
        # M5 tripwire — EXISTING guards, no new guard added by this change.
        "technical_claim_violations": list(state.technical_claim_violations),
        "unit_scale_violations": list(state.unit_scale_violations),
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
        f"arm={arm} rep={repeat} done rating={rating} block_injected={bool(block)} "
        f"claims={claim_metrics['long_horizon_claim_sentences']} "
        f"grounded={claim_metrics['long_horizon_grounded_sentences']} "
        f"hedged={claim_metrics['long_horizon_hedged_sentences']} elapsed={elapsed:.0f}s"
    )


def _plan_runs(tickers: list[str], dates: list[date], repeats: int) -> list[tuple[str, date, str, int]]:
    """Same interleaved/rotated ordering as `reasoning_ab._plan_runs` — see
    that function's docstring for the elapsed-time/server-health confound
    this avoids."""
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
        "--language", default="ko",
        help="Pipeline language, FIXED across both arms (default ko) — this experiment varies "
             "the long-horizon block, not language.",
    )
    parser.add_argument("--repeats", type=int, default=1, help="Independent repeats per (ticker, date, arm).")
    parser.add_argument(
        "--out-dir", default=None,
        help="Defaults to results/horizon_ab/<UTC timestamp>/ if not given.",
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
    args = parser.parse_args()
    if args.attempts < 1:
        parser.error(f"--attempts must be >= 1, got {args.attempts}")
    return args


async def _main() -> None:
    args = _parse_args()
    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    dates = [date.fromisoformat(d.strip()) for d in args.dates.split(",") if d.strip()]
    repeats = args.repeats

    runs = _plan_runs(tickers, dates, repeats)
    total = len(runs)

    if args.dry_run:
        print(f"{logger_prefix} DRY RUN — {total} run(s) planned, nothing executed:")
        for run_idx, (ticker, trade_date, arm, repeat) in enumerate(runs, start=1):
            print(f"  {run_idx}/{total}: ticker={ticker} date={trade_date} arm={arm} rep={repeat}")
        return

    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = Path("results") / "horizon_ab" / datetime.now().strftime("%Y%m%dT%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"{logger_prefix} {total} run(s) queued -> {out_dir} (pipeline language fixed at {args.language!r})")

    for run_idx, (ticker, trade_date, arm, repeat) in enumerate(runs, start=1):
        await _run_one(
            ticker, trade_date, arm, args.language, repeat, out_dir, run_idx, total, args.resume,
            args.attempts, args.retry_delay,
        )

    print(f"{logger_prefix} done. Results in {out_dir} — read record-by-record (see module docstring).")


if __name__ == "__main__":
    asyncio.run(_main())
