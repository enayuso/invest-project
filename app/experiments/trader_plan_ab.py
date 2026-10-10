"""Trader/risk-debate research-plan-visibility A/B (2026-09-16, engine audit
follow-up — priority item 3, the "Underweight collapses to full exit"
problem). Measures whether feeding the Research Manager's actual plan to the
Trader (via its own prompt's rating-scale note) and to the risk debate (via
`risk_debators.build_all`'s new `research_plan_note`) actually reduces how
often a Research-Manager Underweight (a 25% trim) gets described, and acted
on, as a full exit — via a paired before/after comparison of the SAME
(ticker, as_of) input.

## Why this problem, and why A/B (see [[engine-audit-2026-09-15]])

`app.state.TraderAction` is a 3-value direction (Buy/Hold/Sell) — by design,
not an oversight (see `app.audit.stats._RATING_TO_DIRECTION`, which buckets
the Portfolio Manager's own 5-level rating the same way) — and it never
reaches position-sizing arithmetic (`app.execution.sizing` reads only the
Portfolio Manager's 5-level rating). The real defect is a CONTEXT LOSS: once
the Research Manager's plan passes through the Trader, the risk debate never
sees it directly (only the Trader's 3-value proposal), and the Portfolio
Manager sees only a one-word summary of it
(`app.orchestrator.pipeline._desk_consensus_note`). Measured live (807
decisions, 2026-08-13..09-16, see this experiment's companion memory
[[instruction-contradiction-fixes-2026-09-16]] and the design doc it
summarizes): when the Research Manager concluded Underweight and the
Trader's action collapsed to Sell (which happened 61% of the time —
statistically indistinguishable from a real Sell, p=1.00), 46% of those
Trader proposals described a FULL EXIT in prose, and the Portfolio Manager's
own final Sell rate on that path was 5.0% vs. 0.0% when the Trader stayed at
Hold (p=0.013) — the label collapse measurably propagates into a stronger
exit than the plan intended.

This is a DATA-SHAPE/PROMPT change (the risk debate's context literally
grows a new section, the Trader's system prompt grows a new paragraph), not
a factual-error removal — so, like `horizon_ab`, this needs a live paired
run, not a retroactive rescan of stored transcripts. It runs in BACKTEST
mode (`as_of=trade_date`) for the same reason `horizon_ab`/`reasoning_ab` do:
nothing measured here depends on `session_in_progress`.

## Arm construction

Three changes are bundled as one arm (they were designed and shipped
together, and isolating one from the mechanism this measures would just
re-confound M1 with which of the three actually mattered — a follow-up A/B
can split them if this run's result calls for it):

  1. `app.agents.trader.SYSTEM_PROMPT` gained a paragraph naming the plan's
     5-level target-allocation scale and explicitly telling the Trader its
     own `action` field is direction-only. Plain module data (not an import
     binding) — patched directly, same as `horizon_ab`'s
     `market_analyst._BASE_SYSTEM_PROMPT` patch.
  2. `app.agents.risk_debators.build_all` gained a `research_plan_note`
     kwarg that the pipeline now always passes (the Research Manager's
     rendered plan). `pipeline.py` calls `risk_debators.build_all(...)` as a
     module-qualified reference, so patching
     `app.agents.risk_debators.build_all` itself (not the pipeline module)
     is what actually intercepts it — arm `before` wraps the real function
     and drops `research_plan_note` before delegating.
  3. `app.orchestrator.pipeline._desk_consensus_note` now appends the plan's
     target-allocation text after the Research Manager's rating (e.g.
     "Underweight (a real ~25% position, not an exit)"). This is a small
     addition inside a function body, not a module-level constant, so arm
     `before` wraps the real function and regex-strips the parenthetical
     back out — same "reconstruct the old text, raise loudly if the
     mechanism to strip has drifted" contract `candle_ab._before_note` uses,
     except here the mechanism is "the parenthetical must have been present
     to strip" rather than a fixed clause string (the parenthetical's exact
     text depends on which rating the Research Manager happened to pick this
     run).

Each patch is asserted to have stuck before `run_pipeline` is ever called,
and all three are restored unconditionally in `finally` — see
`horizon_ab._horizon_arm`'s own docstring for why a leaked patch would
silently contaminate every later run of an unattended sweep.

## Metrics

M1 (primary — see `trader_plan_metrics`): for runs where the Research
Manager concluded Underweight, whether the Trader's own text
(`position_sizing` + `reasoning`) reads as a full exit (liquidate/close the
position/0 shares/exit entirely) vs. a trim (trim/25%/reduce by/scale
back/partial). Live baseline (`before`, measured pre-fix): 46% full-exit
phrasing among Underweight-that-collapsed-to-Sell proposals. Target: a
material drop toward the ~5% trim-framing baseline already seen for those
proposals that DID phrase it as a trim.

M2 (secondary, NOT a verdict metric on its own — small-sample tripwire):
whether the Portfolio Manager's final rating matches the Research Manager's
Underweight (rather than escalating to Sell). Live baseline: 8.8%
(16/181, RM=Underweight -> PM=Underweight). A `horizon_ab`-sized sweep
(~40 pairs) cannot reliably move a rate this rare into significance — read
it as a direction check, not a pass/fail gate, and never as a confirmed
result from this sample alone.

M3: paired final-rating change rate overall — tripwire only, same
~60-80% self-inconsistency noise floor `language_ab` measured.

M4: per-decision token delta (both new prompt sections add tokens; no
offsetting removal was designed for this fix, unlike `horizon_ab`'s tool-list
trim — expect a real, small increase and confirm it stays small).

M5: paired `technical_claim_violations`/`unit_scale_violations`/
`sizing_cap_misuse_violations` counts — EXISTING guards, no new guard added
by this fix (10월 `guard_alpha` 전까지 신규 가드 추가 중단 권고,
[[fix-without-verification-loop-2026-09-15]]).

## Sample selection

Unlike `horizon_ab` (any ticker/date works), this experiment is only
informative on inputs where the Research Manager is likely to conclude
Underweight — running on an unstratified universe would waste most calls on
Buy/Hold/Overweight outcomes with nothing to measure. Pass `--tickers`/
`--dates` drawn from the live (ticker, trade_date) pairs already known to
have produced RM=Underweight (see the design doc's SQL query); this module
does not query the database itself, to keep it usable standalone in a
network-isolated backtest environment.

    python -m app.experiments.trader_plan_ab \\
        --tickers 950160.KS,005380.KS --dates 2026-08-12,2026-08-26 --repeats 1 --screen-first

No dedicated aggregator — read results record-by-record (`trader_plan_*`
fields), same convention as `horizon_ab`'s own docstring.

## `--screen-first` (added 2026-09-17)

A pilot run on 8 (ticker, trade_date) pairs drawn from live decisions
historically labeled RM=Underweight found 0/16 runs (both arms, all 8 pairs)
actually landed on RM=Underweight under fresh regeneration (backtest mode,
`force_refresh=True`, memory cleared) — LLM regeneration is not deterministic
here, so even a pair that WAS Underweight live does not reliably reproduce it.
Since M1 (see `trader_plan_metrics`) is only meaningful on the
`rm_underweight=True` subset, running the `after` arm unconditionally for a
unit whose `before` arm didn't land on Underweight just spends a real LLM
call on a run this experiment can't use.

`--screen-first` fixes that by handling each (ticker, trade_date, repeat)
unit as a mini-sequence rather than two independent items from `_plan_runs`'s
static, pre-interleaved list: run (or resume) `before` first — always first,
regardless of `_plan_runs`'s own before/after rotation, since none of
`_trader_plan_arm`'s three patches touch `research_manager` or how the RM's
own recommendation is produced, only the Trader/risk-debate/desk-note layer
downstream of it (see `_trader_plan_arm`'s docstring) — then only run `after`
if `before` actually concluded Underweight. A `before` run that errors or
lands on anything else causes `after` to be skipped entirely for that unit:
no `run_pipeline` call, no result or error file written for it, just a
one-line SCREENED OUT message naming the `rm_rating` that was actually
returned. `--resume` still works: an existing `before` result file is read
back off disk (not re-run) to decide whether to run `after`, and an existing
`after` file is skipped per the existing `--resume` logic without needing to
re-check anything. Default is off — both arms always run, in `_plan_runs`'s
existing order, byte-identical to before this flag existed."""

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

import app.agents.risk_debators as risk_debators
import app.agents.trader as trader
import app.orchestrator.pipeline as pipeline_module
from app.config import AIProfile, backtest_scope
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
logger_prefix = "[trader_plan_ab]"

ARM_BEFORE = "before"
ARM_AFTER = "after"
_ARMS = (ARM_BEFORE, ARM_AFTER)

# Byte-identical to the paragraph inserted into app/agents/trader.py's
# SYSTEM_PROMPT by the 2026-09-16 commit — copied rather than retyped so a
# stray wording drift there can't silently make the `before` arm's
# str.replace a no-op.
_TRADER_SCALE_CLAUSE = (
    "What the plan's recommendation actually targets: Buy=100% of full-conviction "
    "allocation, Overweight=75%, Hold=no change from the current position, "
    "Underweight=25% (a real, non-zero target position — NOT 'mostly exit' and NOT "
    "0%), Sell=0% (a full exit). Your own `action` field only has three values "
    "(Buy/Hold/Sell) — it captures DIRECTION, not the plan's target size. When the "
    "plan says Underweight, your action is Sell (the only decrease value available "
    "to you), but that does not make the plan's intent a full exit: describe the "
    "target in `position_sizing`/`reasoning` as a trim toward the plan's real "
    "target level, not as 'full exit', 'liquidate', 'close the position', or a move "
    "to 0 shares. Reserve those full-exit phrasings for when the plan itself says "
    "Sell (0%). Getting the direction right but overstating the size is a real "
    "execution error, not a harmless simplification.\n\n"
)

# The set of parenthetical target-allocation phrases
# `pipeline._desk_consensus_note` can append after the Research Manager's
# rating name — used by arm `before` to strip them back out. Order matters
# (Overweight/Underweight before Hold/Sell/Buy would be fine here since
# these are matched as a single alternation, not nested).
_RM_TARGET_PARENS_RE = re.compile(
    r" \((?:100%|75%|no change from the current position|"
    r"a real ~25% position, not an exit|a full exit)\)"
)

_FULL_EXIT_RE = re.compile(
    r"full exit|liquidat|close (?:the|this) position|exit (?:the|this) position entirely|"
    r"0 shares|zero shares|exit entirely|exit(?:ing)? (?:the|this) (?:entire )?position",
    re.IGNORECASE,
)
_TRIM_RE = re.compile(
    r"\btrim\b|25%|reduce (?:the )?position by|partial (?:reduce|exit|profit)|scale (?:down|back)",
    re.IGNORECASE,
)


def trader_plan_metrics(rm_rating: str | None, trader_text: str) -> dict[str, Any]:
    """M1 — see this module's docstring. `trader_text` is the Trader's
    `position_sizing` + `reasoning` text concatenated (the same fields the
    live measurement scanned). Only meaningful when `rm_rating ==
    "Underweight"`; the caller still calls this for every run so the record
    always carries the same fields, but the aggregate M1 rate must be
    computed over the `rm_underweight=True` subset only."""
    full_exit = bool(_FULL_EXIT_RE.search(trader_text))
    trim = bool(_TRIM_RE.search(trader_text))
    return {
        "rm_underweight": rm_rating == "Underweight",
        "trader_full_exit_phrasing": full_exit,
        "trader_trim_phrasing": trim,
    }


@contextmanager
def _trader_plan_arm(arm: str) -> Iterator[None]:
    """Patches all three 2026-09-16 changes together (see module docstring
    for why they're bundled) and restores them unconditionally in `finally`.

    `trader.SYSTEM_PROMPT` is plain module data — patched directly, same as
    `horizon_ab`'s `_BASE_SYSTEM_PROMPT` patch.

    `risk_debators.build_all` and `pipeline._desk_consensus_note` are
    functions patched with wrappers so arm `after` behaves identically to
    production and arm `before` reconstructs the pre-fix behavior from the
    CURRENT code rather than a frozen copy — same "one source of truth"
    reasoning `candle_ab`/`horizon_ab` use."""
    original_trader_prompt = trader.SYSTEM_PROMPT
    original_build_all = risk_debators.build_all
    original_desk_note = pipeline_module._desk_consensus_note

    if arm == ARM_BEFORE:
        if _TRADER_SCALE_CLAUSE not in original_trader_prompt:
            raise AssertionError(
                "trader_plan_ab: _TRADER_SCALE_CLAUSE not found in trader.SYSTEM_PROMPT — "
                "app.agents.trader's wording has drifted since this experiment's constant was "
                "copied from it. Update the constant (never silently fall back to leaving the "
                "prompt unmodified — see this function's own docstring)."
            )
        trader.SYSTEM_PROMPT = original_trader_prompt.replace(_TRADER_SCALE_CLAUSE, "")

        def _build_all_before(*args: Any, **kwargs: Any) -> Any:
            kwargs["research_plan_note"] = ""
            return original_build_all(*args, **kwargs)

        def _desk_note_before(*args: Any, **kwargs: Any) -> str:
            note = original_desk_note(*args, **kwargs)
            if "concluded" in note and not _RM_TARGET_PARENS_RE.search(note):
                raise AssertionError(
                    "trader_plan_ab: expected a target-allocation parenthetical after "
                    "'concluded <rating>' in _desk_consensus_note's output but found none — "
                    "app.orchestrator.pipeline's wording has drifted since this experiment's "
                    "_RM_TARGET_PARENS_RE was written against it. Update the regex (never "
                    "silently fall back to leaving the note unmodified)."
                )
            return _RM_TARGET_PARENS_RE.sub("", note)

        risk_debators.build_all = _build_all_before
        pipeline_module._desk_consensus_note = _desk_note_before
    else:
        risk_debators.build_all = original_build_all
        pipeline_module._desk_consensus_note = original_desk_note

    assert risk_debators.build_all is not original_build_all or arm == ARM_AFTER
    try:
        yield
    finally:
        trader.SYSTEM_PROMPT = original_trader_prompt
        risk_debators.build_all = original_build_all
        pipeline_module._desk_consensus_note = original_desk_note


def _run_filename(ticker: str, trade_date: date, arm: str, repeat: int) -> str:
    return f"{ticker}_{trade_date.isoformat()}_{arm}_r{repeat}.json"


async def _run_pipeline_with_retry(
    ticker: str, trade_date: date, arm: str, language: str, attempts: int, retry_delay: float,
) -> tuple[PipelineState, int, int, list[str], dict[str, Any], int]:
    """Same retry/isolation shape as `horizon_ab._run_pipeline_with_retry`,
    with the trader-plan arm as the varying axis."""
    for attempt in range(1, attempts + 1):
        llm_metrics.drain()  # discard any prior attempt's records before this one starts
        try:
            await clear_backtest_positions([ticker])
            await clear_backtest_memory([ticker])
            with (
                _no_retrieved_memory() as (original_n_same, original_n_cross),
                _trader_plan_arm(arm),
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
) -> dict[str, Any] | None:
    """Returns the run's result record (parsed back from disk if `--resume`
    skipped an already-done run, otherwise the freshly written record), or
    `None` if the run errored. `_main` reads `record["rm_underweight"]` off
    this return value under `--screen-first` to decide whether to run the
    `after` arm for the same unit — see module docstring's 2026-09-17
    addendum. The only caller is `_main`; no other module or test relies on
    this having returned `None` unconditionally."""
    out_path = out_dir / _run_filename(ticker, trade_date, arm, repeat)
    if resume and out_path.exists():
        print(
            f"{logger_prefix} run {run_idx}/{total} ticker={ticker} date={trade_date} "
            f"arm={arm} rep={repeat} skipped (--resume, already done)"
        )
        return json.loads(out_path.read_text(encoding="utf-8"))

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
                    "ticker": ticker, "date": trade_date.isoformat(), "arm": arm, "repeat": repeat,
                    "started_at": started_at.isoformat(), "elapsed_seconds": elapsed, "error": repr(exc),
                },
                ensure_ascii=False, indent=2,
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

    rm_rating = state.research_plan.recommendation.value if state.research_plan else None
    trader_action = state.trader_proposal.action.value if state.trader_proposal else None
    pm_rating = state.final_decision.rating.value if state.final_decision else None
    trader_text = ""
    if state.trader_proposal is not None:
        trader_text = " ".join(
            p for p in (state.trader_proposal.position_sizing, state.trader_proposal.reasoning) if p
        )
    plan_metrics = trader_plan_metrics(rm_rating, trader_text)

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
        # Mechanism proof (M1) — see module docstring.
        "rm_rating": rm_rating,
        "trader_action": trader_action,
        "pm_rating": pm_rating,
        "rm_pm_rating_match": rm_rating is not None and rm_rating == pm_rating,
        **plan_metrics,
        # M5 tripwire — EXISTING guards, no new guard added by this change.
        "technical_claim_violations": list(state.technical_claim_violations),
        "unit_scale_violations": list(state.unit_scale_violations),
        "sizing_cap_misuse_violations": list(state.sizing_cap_misuse_violations),
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

    print(
        f"{logger_prefix} run {run_idx}/{total} ticker={ticker} date={trade_date} "
        f"arm={arm} rep={repeat} done rm={rm_rating} trader={trader_action} pm={pm_rating} "
        f"full_exit_phrasing={plan_metrics['trader_full_exit_phrasing']} elapsed={elapsed:.0f}s"
    )
    return record


def _plan_runs(tickers: list[str], dates: list[date], repeats: int) -> list[tuple[str, date, str, int]]:
    """Same interleaved/rotated ordering as `horizon_ab._plan_runs`. Used
    only when `--screen-first` is NOT set — see `_plan_screen_first_units`
    for the runtime-decided alternative (module docstring's 2026-09-17
    addendum)."""
    runs: list[tuple[str, date, str, int]] = []
    for ticker in tickers:
        for trade_date in dates:
            for repeat in range(1, repeats + 1):
                offset = (repeat - 1) % len(_ARMS)
                ordered = _ARMS[offset:] + _ARMS[:offset]
                runs.extend((ticker, trade_date, arm, repeat) for arm in ordered)
    return runs


def _plan_screen_first_units(tickers: list[str], dates: list[date], repeats: int) -> list[tuple[str, date, int]]:
    """One unit per (ticker, trade_date, repeat) — under `--screen-first`
    the arm to run next is a runtime decision (whether the `before` arm
    landed on RM=Underweight), not something `_plan_runs`'s static list can
    encode. `before` always runs first for every unit regardless of
    `_plan_runs`'s before/after rotation, because none of `_trader_plan_arm`'s
    three patches touch `research_manager` or how the RM's own recommendation
    is produced — only the Trader/risk-debate/desk-note layer downstream of
    it — so which arm runs first cannot affect whether `before` reproduces
    Underweight (see module docstring's 2026-09-17 addendum)."""
    units: list[tuple[str, date, int]] = []
    for ticker in tickers:
        for trade_date in dates:
            for repeat in range(1, repeats + 1):
                units.append((ticker, trade_date, repeat))
    return units


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--tickers", required=True,
        help="Comma-separated ticker universe — draw these from (ticker, trade_date) pairs "
             "already known to have produced RM=Underweight (see module docstring's SQL note); "
             "an unstratified universe wastes most calls on outcomes this experiment can't use.",
    )
    parser.add_argument("--dates", required=True, help="Comma-separated YYYY-MM-DD as_of dates.")
    parser.add_argument(
        "--language", default="ko",
        help="Pipeline language, FIXED across both arms (default ko).",
    )
    parser.add_argument("--repeats", type=int, default=1, help="Independent repeats per (ticker, date, arm).")
    parser.add_argument(
        "--out-dir", default=None,
        help="Defaults to results/trader_plan_ab/<UTC timestamp>/ if not given.",
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
        help="Seconds between attempts. Default 60.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the planned runs and exit without calling run_pipeline or touching the DB.",
    )
    parser.add_argument(
        "--screen-first", action="store_true",
        help="Run the 'before' arm first for each (ticker, date, repeat) unit and only run "
             "'after' if that 'before' run actually landed on RM=Underweight (see module "
             "docstring's 2026-09-17 addendum — a fresh-regeneration pilot found 0/16 runs "
             "reproduced RM=Underweight from pairs believed to reproduce it, so running 'after' "
             "unconditionally wastes calls on units this experiment can't measure). Default off "
             "(both arms always run, in `_plan_runs`'s existing interleaved order) for full "
             "backward compatibility.",
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

    if args.screen_first:
        units = _plan_screen_first_units(tickers, dates, repeats)
        # Upper bound — see module docstring's 2026-09-17 addendum. The
        # actual number of runs executed will be lower whenever a `before`
        # run doesn't land on RM=Underweight (or errors), since its paired
        # `after` run is then skipped entirely.
        total = len(units) * len(_ARMS)

        if args.dry_run:
            print(
                f"{logger_prefix} DRY RUN (--screen-first) — up to {total} run(s) planned (actual "
                f"count is lower: each unit's 'after' run only happens if 'before' lands on "
                f"RM=Underweight, which cannot be known in advance), nothing executed:"
            )
            run_idx = 0
            for ticker, trade_date, repeat in units:
                run_idx += 1
                print(f"  {run_idx}/{total}: ticker={ticker} date={trade_date} arm={ARM_BEFORE} rep={repeat}")
                run_idx += 1
                print(
                    f"  {run_idx}/{total}: ticker={ticker} date={trade_date} arm={ARM_AFTER} rep={repeat} "
                    f"(conditional on the 'before' run above landing on RM=Underweight)"
                )
            return

        if args.out_dir:
            out_dir = Path(args.out_dir)
        else:
            out_dir = Path("results") / "trader_plan_ab" / datetime.now().strftime("%Y%m%dT%H%M%S")
        out_dir.mkdir(parents=True, exist_ok=True)

        print(
            f"{logger_prefix} up to {total} run(s) queued -> {out_dir} (pipeline language fixed at "
            f"{args.language!r}, --screen-first: 'after' runs skipped whenever 'before' doesn't land "
            "on RM=Underweight)"
        )

        run_idx = 0
        for ticker, trade_date, repeat in units:
            run_idx += 1
            before_record = await _run_one(
                ticker, trade_date, ARM_BEFORE, args.language, repeat, out_dir, run_idx, total,
                args.resume, args.attempts, args.retry_delay,
            )
            rm_underweight = bool(before_record and before_record.get("rm_underweight"))
            if rm_underweight:
                run_idx += 1
                await _run_one(
                    ticker, trade_date, ARM_AFTER, args.language, repeat, out_dir, run_idx, total,
                    args.resume, args.attempts, args.retry_delay,
                )
            else:
                rm_rating = before_record.get("rm_rating") if before_record else None
                skipped_idx = run_idx + 1
                print(
                    f"{logger_prefix} run {skipped_idx}/{total} ticker={ticker} date={trade_date} "
                    f"arm={ARM_AFTER} rep={repeat}: SCREENED OUT (before arm concluded "
                    f"rm_rating={rm_rating!r}, not Underweight) — skipping after arm"
                )

        print(f"{logger_prefix} done. Results in {out_dir} — read record-by-record (see module docstring).")
        return

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
        out_dir = Path("results") / "trader_plan_ab" / datetime.now().strftime("%Y%m%dT%H%M%S")
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
