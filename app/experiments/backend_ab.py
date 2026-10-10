"""Qwen/vLLM-vs-Claude/Anthropic backend A/B runner — 0단계 of the Claude-
backend measurement plan (see project memory: the plan is to compare
hallucination-guard violation rates between this desk's current Qwen/vLLM
setup and a Claude-driven one, before any decision about routing real traffic
to Claude).

*** THIS SCRIPT SPENDS REAL MONEY when its Claude arm runs — every claude-arm
run makes a real `AsyncAnthropic` call against `settings.anthropic_api_key`.
It must NEVER be imported or executed as a side effect of any test, pipeline
import, or other experiment script — the module deliberately does nothing
network/API-related at import time (see the provider-swap section below), and
the ONLY legitimate invocation is an explicit
`python -m app.experiments.backend_ab ...` typed by a human. A human (not an
agent session) must run a small pilot first —

    python -m app.experiments.backend_ab --tickers 005930.KS --dates 2026-07-01 --repeats 1

— to get one real `usd_cost_estimate`, multiply it by the planned Claude-run
count (`tickers x dates x repeats`) and by `--attempts` for the worst case,
and approve/re-scope the full sweep on that number BEFORE running it.
`usd_cost_estimate` is only an approximation of the eventual bill, not the
bill itself: prompt caching folds cached-read/cache-creation tokens into the
reported `prompt_tokens` at different effective per-token rates than a fresh
prompt token, and this script prices every prompt token at the same flat
$5/MTok regardless of which kind it was.

Structural template: this file clones `app.experiments.reasoning_ab`'s shape
(retry loop, atomic JSON write, error-JSON-on-exception, --resume, isolation
tripwire, interleaved run plan) file-for-file wherever the shape doesn't need
to change for this experiment's own axis (backend, not reasoning/language).
Reuses `app.experiments.language_ab`'s isolation machinery
(`_no_retrieved_memory`, `_state_to_dict`, `_summarize_llm_metrics`,
`degraded_report_names`) wholesale rather than duplicating it, exactly as
`reasoning_ab.py` does.

## The provider-swap problem this module exists to solve

`from app.llm import llm_client` (used by every one of the ~9 importer
modules below) binds a NEW NAME in the importing module's OWN namespace to
whatever object `app.llm.llm_client` pointed to AT IMPORT TIME. Reassigning
`app.llm.llm_client` later does not change what an already-imported module
sees when it references its own local `llm_client` name — every one of those
modules' namespace entries has to be patched individually for a Claude-arm
run to actually call Claude instead of silently keeps calling the real Qwen
backend with no error, which would invalidate this whole measurement without
telling anyone. See `_backend_arm` below for the exact mechanism, and
`tests/experiments/test_backend_ab.py` for the tests proving it actually
patches every target and always restores them, including across an
exception.

## Guard-violation scan

`_guard_violation_counts` reads `dataclasses.fields(PipelineState)` at call
time and keeps every field whose name ends with `_violations` or `_flags` —
it NEVER hardcodes that field-name list, so a future guard field is picked up
automatically without an edit here. A separate test pins the CURRENT
authoritative set of names so silent drift (a field renamed/added/removed) is
caught rather than silently changing what this script measures.

    python -m app.experiments.backend_ab \\
        --tickers 005930.KS,AAPL --dates 2026-07-01,2026-07-15 --repeats 3

Aggregate afterwards with:
    python -m app.experiments.backend_ab_report --dir <out-dir>
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
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

import app.llm  # the package itself, not `from app.llm import llm_client` — see module docstring.
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

# ---------------------------------------------------------------------------
# Eager import of every module that does `from app.llm import llm_client`
# (per the grep this batch's spec required — see the task's own list), PLUS
# app.orchestrator.pipeline (run_pipeline's home, imported above already).
#
# This MUST happen here, at backend_ab.py's own import time, before any arm
# is ever entered. If one of these modules were instead imported lazily for
# the first time from inside a Claude-arm `with _backend_arm(...)` block,
# its own `from app.llm import llm_client` statement would bind straight to
# the ALREADY-PATCHED (Claude) client — and _backend_arm's exit only restores
# modules it patched on ENTRY (the ones already in sys.modules at that point),
# so a module that only starts existing partway through the arm would never
# be un-patched, leaking the Claude client into every later qwen-arm run for
# the rest of the process. Importing everything up front means every one of
# these modules is already in sys.modules, with its own `llm_client` name
# already bound to the real qwen-arm object, before `_backend_arm` ever runs
# its patch loop.
import app.agents.debater  # noqa: E402,F401
import app.agents.portfolio_manager  # noqa: E402,F401
import app.agents.research_manager  # noqa: E402,F401
import app.agents.sentiment_analyst  # noqa: E402,F401
import app.agents.tool_loop  # noqa: E402,F401
import app.agents.trader  # noqa: E402,F401
import app.chat.router  # noqa: E402,F401
import app.memory.reflection  # noqa: E402,F401
import app.orchestrator.debate  # noqa: E402,F401
import app.orchestrator.pipeline  # noqa: E402,F401

if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

logger = logging.getLogger(__name__)
logger_prefix = "[backend_ab]"

ARM_CLAUDE = "claude"
ARM_QWEN = "qwen"
_ARMS = (ARM_CLAUDE, ARM_QWEN)

# The one qwen-arm client object used for EVERY qwen-arm run, for the whole
# process lifetime. Never rebuilt: `LLMClient.__init__` snapshots
# `settings.llm_provider` at construction time (see app/llm/client.py), so a
# freshly built instance made after `settings.llm_provider` had been flipped
# to "anthropic" for a Claude-arm run would be permanently wrong even after
# the setting is restored.
_ORIGINAL_LLM_CLIENT = app.llm.llm_client

# Every module name that actually holds an `llm_client` attribute bound via
# `from app.llm import llm_client` — the exact grep list from the module
# docstring/task spec — plus "app.llm" itself (the module that owns the
# canonical `llm_client` singleton `_build_llm_client()` produces).
_REQUIRED_PATCH_TARGETS: frozenset[str] = frozenset({
    "app.llm",
    "app.agents.tool_loop",
    "app.agents.debater",
    "app.agents.research_manager",
    "app.agents.trader",
    "app.agents.portfolio_manager",
    "app.agents.sentiment_analyst",
    "app.orchestrator.debate",
    "app.memory.reflection",
    "app.chat.router",
})

# Lazily constructed, cached AnthropicLLMClient — see _claude_client().
_claude_client_instance: object | None = None


def _claude_client():
    """Lazily builds (once) and caches the real `AnthropicLLMClient` used for
    every Claude-arm run in this process — repeated `_backend_arm(ARM_CLAUDE)`
    entries across a sweep reuse the SAME client/connection-pool/limiter
    rather than constructing a fresh one per run. `AnthropicLLMClient` is
    imported here, not at module level, so merely importing `backend_ab.py`
    (e.g. from a test, or `--dry-run`) never touches the `anthropic` package
    or constructs a client that would need a real API key — mirrors
    `app.llm.__init__._build_llm_client`'s own lazy-import doctrine."""
    global _claude_client_instance
    if _claude_client_instance is None:
        from app.llm.anthropic_client import AnthropicLLMClient

        _claude_client_instance = AnthropicLLMClient()
    return _claude_client_instance


@contextmanager
def _backend_arm(arm: str) -> Iterator[None]:
    """Temporarily patches every module in `_REQUIRED_PATCH_TARGETS` (plus
    any other `app.*` module discovered holding either known client object —
    see the identity scan below) so its own `llm_client` name resolves to the
    arm's client, flips `settings.llm_provider` to match, and unconditionally
    restores everything on exit — same "a leaked override would silently
    contaminate every later run in an unattended sweep" doctrine as
    `reasoning_ab._reasoning_arm`, just patching module attributes instead of
    one settings field.

    Discovery for the actual patch loop is by OBJECT IDENTITY, not a
    hardcoded name list: any `app.*` module whose current `llm_client`
    attribute `is` the "other" arm's client object gets patched to the
    target. This is deliberately more defensive than name-only — it patches
    any module holding either known client object even if it was missed by
    the grep this file's `_REQUIRED_PATCH_TARGETS` was built from. The
    mandatory post-patch assertion below still checks the REQUIRED set
    explicitly, so a genuinely missing target still raises loudly instead of
    only being "defensively" covered.
    """
    target = _ORIGINAL_LLM_CLIENT if arm == ARM_QWEN else _claude_client()
    other = _claude_client_instance if arm == ARM_QWEN else _ORIGINAL_LLM_CLIENT
    # `other` for the claude arm is always `_ORIGINAL_LLM_CLIENT` (never
    # None); for the qwen arm it's whatever the cached claude client is (may
    # be None if no claude-arm run has happened yet this process — in which
    # case nothing will ever match its identity, harmlessly).

    patched: list[tuple[Any, object]] = []
    for name, mod in list(sys.modules.items()):
        if not (name == "app.llm" or name.startswith("app.")):
            continue
        current = getattr(mod, "llm_client", None)
        if current is None:
            continue
        if current is target:
            continue  # already pointing at the target — nothing to patch/restore here
        if current is other or (arm == ARM_CLAUDE and current is _ORIGINAL_LLM_CLIENT):
            patched.append((mod, current))
            setattr(mod, "llm_client", target)

    _MISSING = object()
    previous_default_instance = _MISSING
    if hasattr(app.llm, "_llm_instances") and isinstance(app.llm._llm_instances, dict):
        previous_default_instance = app.llm._llm_instances.get("default", _MISSING)
        app.llm._llm_instances["default"] = target

    original_provider = settings.llm_provider
    settings.llm_provider = "anthropic" if arm == ARM_CLAUDE else _qwen_provider_name()

    try:
        # Mandatory post-patch assertion — runs on every single arm entry,
        # not just in tests. Anything reachable through
        # `from app.llm import llm_client` MUST resolve to `target` before a
        # single pipeline call happens under this arm.
        bad = []
        for name in _REQUIRED_PATCH_TARGETS:
            mod = sys.modules.get(name)
            if mod is None or getattr(mod, "llm_client", None) is not target:
                held = None if mod is None else getattr(mod, "llm_client", "<missing>")
                bad.append(f"{name} (module {'not imported' if mod is None else 'holds ' + repr(held)})")
        if bad:
            raise RuntimeError(
                f"_backend_arm({arm!r}): failed to patch every required module before running — "
                f"{', '.join(bad)}. This would silently run the WRONG backend for this arm; see "
                "app.experiments.backend_ab's module docstring for the provider-swap mechanism."
            )
        yield
    finally:
        for mod, previous in patched:
            setattr(mod, "llm_client", previous)
        if hasattr(app.llm, "_llm_instances") and isinstance(app.llm._llm_instances, dict):
            if previous_default_instance is _MISSING:
                app.llm._llm_instances.pop("default", None)
            else:
                app.llm._llm_instances["default"] = previous_default_instance
        settings.llm_provider = original_provider

        # Belt-and-braces re-scan: force-reset ANY app.* module still holding
        # the Claude client object back to the real qwen client — closes the
        # "a module was imported for the first time during the Claude arm,
        # was never in `patched` because it didn't exist in sys.modules on
        # entry, and so was never restored above" hole completely.
        if _claude_client_instance is not None:
            for name, mod in list(sys.modules.items()):
                if not (name == "app.llm" or name.startswith("app.")):
                    continue
                if getattr(mod, "llm_client", None) is _claude_client_instance:
                    setattr(mod, "llm_client", _ORIGINAL_LLM_CLIENT)


def _qwen_provider_name() -> str:
    """The qwen arm's own configured provider ("vllm" or "gateway"), read off
    `_ORIGINAL_LLM_CLIENT` itself rather than assumed — `LLMClient` records
    the provider it was built with as `self._provider` (see app/llm/client.py)
    at construction time. Falls back to "vllm" if that attribute is somehow
    absent (should not happen for any real `LLMClient` instance) — nothing
    inside `run_pipeline`'s call graph re-reads `settings.llm_provider` at
    runtime (only `app.llm._build_llm_client`, at construction time, does),
    so the exact value used here mid-arm is cosmetic/provenance only; what
    matters is that it is restored to `original_provider` on exit."""
    return getattr(_ORIGINAL_LLM_CLIENT, "_provider", "vllm")


def _run_filename(ticker: str, trade_date: date, arm: str, repeat: int) -> str:
    return f"{ticker}_{trade_date.isoformat()}_{arm}_r{repeat}.json"


async def _run_pipeline_with_retry(
    ticker: str, trade_date: date, arm: str, language: str, attempts: int, retry_delay: float,
) -> tuple[PipelineState, int, int, list[str], dict[str, Any], int]:
    """Same retry/isolation shape as `reasoning_ab._run_pipeline_with_retry`,
    with `arm` (claude/qwen backend) swapped in for reasoning as the varying
    axis via `_backend_arm` instead of `_reasoning_arm`."""
    for attempt in range(1, attempts + 1):
        llm_metrics.drain()  # discard any prior attempt's records before this one starts
        try:
            await clear_backtest_positions([ticker])
            await clear_backtest_memory([ticker])
            with (
                _no_retrieved_memory() as (original_n_same, original_n_cross),
                _backend_arm(arm),
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


def _guard_violation_counts(state: PipelineState) -> dict[str, int]:
    """Dynamic guard-violation scan — NEVER hardcode the field-name list
    here. Keeps every `dataclasses.fields(PipelineState)` entry whose name
    ends with "_violations" or "_flags" (the exact same suffix rule
    `app.audit.guard_alpha._guard_keys` uses over a stored transcript dict),
    reads it off `state`, and records `len(value)` (0 for an empty/falsy
    value) — works identically for a `list[str]` field and a `list[dict]`
    field like `injection_flags`. A future guard field is picked up
    automatically the moment it's added to `PipelineState`; a test in
    `tests/experiments/test_backend_ab.py` pins the CURRENT count so a silent
    rename/add/remove is caught rather than silently changing what this
    script measures."""
    counts: dict[str, int] = {}
    for f in dataclasses.fields(state):
        if f.name.endswith("_violations") or f.name.endswith("_flags"):
            value = getattr(state, f.name)
            counts[f.name] = len(value) if value else 0
    return counts


class _ClaudeCostTracker:
    """Running mean / cumulative total of Claude-arm `usd_cost_estimate`
    values seen so far in this invocation — passed into `_run_one` so it can
    print a cost estimate BEFORE a Claude-arm run starts (based on every
    prior Claude-arm run's actual cost) and the caller (`_main`) can check
    the cumulative total against `--max-claude-usd` after each one."""

    def __init__(self) -> None:
        self.count = 0
        self.total_usd = 0.0

    def running_mean(self) -> float | None:
        return self.total_usd / self.count if self.count else None

    def record(self, usd: float) -> None:
        self.count += 1
        self.total_usd += usd


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
    cost_tracker: _ClaudeCostTracker,
) -> None:
    out_path = out_dir / _run_filename(ticker, trade_date, arm, repeat)
    if resume and out_path.exists():
        print(
            f"{logger_prefix} run {run_idx}/{total} ticker={ticker} date={trade_date} "
            f"arm={arm} rep={repeat} skipped (--resume, already done)"
        )
        return

    if arm == ARM_CLAUDE:
        mean = cost_tracker.running_mean()
        if mean is None:
            print(
                f"{logger_prefix} run {run_idx}/{total} ticker={ticker} date={trade_date} "
                f"arm={arm} rep={repeat}: FIRST RUN — per-run cost is UNKNOWN until this run completes"
            )
        else:
            print(
                f"{logger_prefix} run {run_idx}/{total} ticker={ticker} date={trade_date} "
                f"arm={arm} rep={repeat}: running mean cost so far ${mean:.4f}/run "
                f"({cost_tracker.count} claude run(s) so far, ${cost_tracker.total_usd:.4f} total)"
            )

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

    guard_violations = _guard_violation_counts(state)
    decision = state.final_decision
    provider_settings: dict[str, Any] = {"llm_provider": settings.llm_provider}
    if arm == ARM_CLAUDE:
        provider_settings.update(
            anthropic_model=settings.anthropic_model,
            anthropic_effort=settings.anthropic_effort,
            anthropic_effort_low=settings.anthropic_effort_low,
            anthropic_thinking_display=settings.anthropic_thinking_display,
            anthropic_prompt_cache=settings.anthropic_prompt_cache,
        )

    record: dict[str, Any] = {
        **llm_summary,
        **_state_to_dict(state),
        "ticker": ticker,
        "date": trade_date.isoformat(),
        "arm": arm,
        "repeat": repeat,
        "language": language,
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "wall_seconds": elapsed,
        "attempts_used": attempts_used,
        "isolation_violation": isolation_violation,
        "degraded": bool(degraded),
        "degraded_reports": degraded,
        "state_degraded_analysts": list(state.degraded_analysts),
        "reasoning_sanitizations": list(state.reasoning_sanitizations),
        "provider_settings": provider_settings,
        "guard_violations": guard_violations,
        "guard_violations_total": sum(guard_violations.values()),
        "final_rating": decision.rating.value if decision is not None else None,
        "confidence": decision.confidence if decision is not None else None,
        "trader_action": (
            state.trader_proposal.action.value if state.trader_proposal is not None else None
        ),
        "consensus_override": dict(state.consensus_override) if state.consensus_override else {},
        "force_refresh": True,
        "memory_n_same_override": 0,
        "memory_n_same_production": original_n_same,
        "memory_n_cross_override": 0,
        "memory_n_cross_production": original_n_cross,
        "llm_attempts_used": attempts_used,
    }

    if arm == ARM_CLAUDE:
        prompt_tokens = llm_summary.get("llm_prompt_tokens", 0) or 0
        completion_tokens = llm_summary.get("llm_completion_tokens", 0) or 0
        usd_cost_estimate = prompt_tokens * 5 / 1e6 + completion_tokens * 25 / 1e6
        record["usd_cost_estimate"] = usd_cost_estimate
        record["pricing_note"] = (
            "Approximation only, priced at a flat $5/MTok prompt + $25/MTok completion — NOT a "
            "bill. Prompt caching folds cached-read/cache-creation tokens into prompt_tokens at "
            "different effective rates than a fresh prompt token, so this overstates cost whenever "
            "the cache actually helped."
        )
        cost_tracker.record(usd_cost_estimate)

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

    rating = decision.rating.value if decision is not None else "N/A"
    confidence = f"{decision.confidence:.2f}" if decision is not None and decision.confidence is not None else "N/A"
    cost_note = f" cost=${record['usd_cost_estimate']:.4f}" if arm == ARM_CLAUDE else ""
    print(
        f"{logger_prefix} run {run_idx}/{total} ticker={ticker} date={trade_date} "
        f"arm={arm} rep={repeat} done rating={rating} conf={confidence} elapsed={elapsed:.0f}s"
        f"{cost_note} guard_violations_total={record['guard_violations_total']}"
    )


def _plan_runs(tickers: list[str], dates: list[date], repeats: int) -> list[tuple[str, date, str, int]]:
    """Run order with the two arms INTERLEAVED, cyclically rotated on each
    repeat — identical shape/rationale to `reasoning_ab._plan_runs`, just with
    `_ARMS = (ARM_CLAUDE, ARM_QWEN)` instead of `(ARM_ON, ARM_OFF)`."""
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
    parser.add_argument("--language", default="en", help="Pipeline language, fixed across both arms (default en).")
    parser.add_argument("--repeats", type=int, default=3, help="Independent repeats per (ticker, date, arm).")
    parser.add_argument(
        "--out-dir", default=None,
        help="Defaults to results/backend_ab/<UTC timestamp>/ if not given.",
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
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the planned run list/counts per arm and exit WITHOUT importing/constructing "
             "anything Anthropic-related and without executing any run.",
    )
    parser.add_argument(
        "--max-claude-usd", type=float, default=5.0,
        help="Cumulative circuit breaker (default $5.00): after each Claude-arm run, if the running "
             "total usd_cost_estimate across this invocation exceeds this, abort the sweep.",
    )
    args = parser.parse_args()
    if args.attempts < 1:
        parser.error(f"--attempts must be >= 1, got {args.attempts}")
    return args


def _print_plan(runs: list[tuple[str, date, str, int]], out_dir: Path) -> None:
    total = len(runs)
    claude_count = sum(1 for _, _, arm, _ in runs if arm == ARM_CLAUDE)
    qwen_count = total - claude_count
    print(f"{logger_prefix} {total} run(s) planned -> {out_dir}  (claude={claude_count}, qwen={qwen_count})")
    print(f"{logger_prefix} cost per run unknown until the pilot (see module docstring)")


async def _main() -> None:
    args = _parse_args()
    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    dates = [date.fromisoformat(d.strip()) for d in args.dates.split(",") if d.strip()]
    repeats = args.repeats

    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = Path("results") / "backend_ab" / datetime.now().strftime("%Y%m%dT%H%M%S")

    runs = _plan_runs(tickers, dates, repeats)

    if args.dry_run:
        # No mkdir, no Anthropic import/construction, no run executed —
        # `_print_plan` and everything above it is pure/local computation.
        _print_plan(runs, out_dir)
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    _print_plan(runs, out_dir)

    has_claude_run = any(arm == ARM_CLAUDE for _, _, arm, _ in runs)
    if has_claude_run and not settings.anthropic_api_key:
        print(
            f"{logger_prefix} ERROR: this plan includes {sum(1 for _, _, a, _ in runs if a == ARM_CLAUDE)} "
            "claude-arm run(s) but settings.anthropic_api_key is not set — refusing to start. Set "
            "ANTHROPIC_API_KEY (see app.config.Settings) before running the claude arm."
        )
        sys.exit(1)

    cost_tracker = _ClaudeCostTracker()
    total = len(runs)
    for run_idx, (ticker, trade_date, arm, repeat) in enumerate(runs, start=1):
        await _run_one(
            ticker, trade_date, arm, args.language, repeat, out_dir, run_idx, total, args.resume,
            args.attempts, args.retry_delay, cost_tracker,
        )
        if arm == ARM_CLAUDE and cost_tracker.total_usd > args.max_claude_usd:
            print(
                f"{logger_prefix} ABORTING sweep: cumulative claude usd_cost_estimate "
                f"${cost_tracker.total_usd:.4f} exceeded --max-claude-usd {args.max_claude_usd:.2f} "
                f"after run {run_idx}/{total}. Results so far are in {out_dir}. To continue past this "
                f"cap, re-run with --resume --out-dir {out_dir} --max-claude-usd <higher value>."
            )
            return

    print(
        f"{logger_prefix} done. Aggregate with: python -m app.experiments.backend_ab_report --dir {out_dir}"
    )


if __name__ == "__main__":
    asyncio.run(_main())
