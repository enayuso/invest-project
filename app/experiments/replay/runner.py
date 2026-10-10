"""Downstream node composition + run loop of the replay harness (spec A.6, A.7, D).

Composes the REAL agent code (`research_manager.synthesize`, `trader.propose`,
`risk_debators.build_all` + `run_debate`, `portfolio_manager.decide`, the four gates)
in the same order and with the same arguments as `pipeline._run_pipeline_impl`
(~lines 2094-2444), fed by `contract.ReplayInputs`. Function-local literals of the
original (numeric turn guard, `_guard_fields`) are re-implemented here; the pinned
source hashes in `provenance.py` are what keeps this copy honest.
운영 파이프라인의 RM→Trader→Risk→PM→게이트 구간을 같은 순서·인자로 재조합한다.

Out of scope (stage 3-4): the ten post-PM sweep guards and the PM contradiction alerts —
they never change the rating and lack `verified_snapshot`/`kr_gt` here; case JSON records
`"sweeps": "not_run"`.
"""

from __future__ import annotations

import asyncio
import contextvars
import copy
import enum
import hashlib
import json
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Mapping

from app.agents import portfolio_manager, research_manager, risk_debators, trader
from app.agents.numeric_guard import enforce_numeric_guards
from app.clock import LiveClock
from app.config import account_mode_scope, backtest_scope, settings
from app.execution import runtime_settings
from app.experiments.replay import safety
from app.experiments.replay.contract import ReplayInputs, atomic_write_text
from app.experiments.replay.provenance import NODES
from app.llm import llm_client
from app.llm import metrics as llm_metrics
from app.orchestrator import pipeline
from app.orchestrator.debate import run_debate
from app.state import (
    DebateState,
    PortfolioDecision,
    ResearchPlan,
    TraderProposal,
    render_research_plan,
)

CASE_SCHEMA = "replay-case-v1"
BULL = frozenset({"Buy", "Overweight"})
EXIT_OK, EXIT_CONFIG, EXIT_SAFETY, EXIT_PAUSED, EXIT_INCOMPLETE = 0, 2, 3, 75, 76


def utc_now_iso() -> str:
    return LiveClock().now(timezone.utc).isoformat()


# --------------------------------------------------------------------------- call instrumentation


@dataclass
class CaseCalls:
    """Per-case LLM call capture (ContextVar-scoped, so concurrent cases never mix)."""

    records: list[dict[str, Any]] = field(default_factory=list)  # one per `metrics.record`
    outer: list[dict[str, Any]] = field(default_factory=list)  # one per complete/structured_complete call


_CASE_CALLS: contextvars.ContextVar[CaseCalls | None] = contextvars.ContextVar("_replay_case_calls", default=None)
_PENDING_SHA: contextvars.ContextVar[str | None] = contextvars.ContextVar("_replay_pending_sha", default=None)


def prompt_sha256(messages: Any) -> str:
    return hashlib.sha256(
        json.dumps(messages, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    ).hexdigest()


class Instrumentation:
    """Wraps `app.llm.metrics.record` and the `llm_client` methods (instance attributes) so each
    call is attributed to the running case. Install ONCE per process, AFTER any test fake."""

    def __init__(self) -> None:
        self._orig_record: Callable[..., Any] | None = None
        self._had_attr: dict[str, bool] = {}
        self._orig_methods: dict[str, Any] = {}
        self.installed = False

    def install(self) -> None:
        if self.installed:
            return
        self._orig_record = llm_metrics.record
        orig_record = self._orig_record

        def record(**kwargs: Any) -> Any:
            calls = _CASE_CALLS.get()
            if calls is not None:
                calls.records.append({
                    "label": kwargs.get("label"), "structured": bool(kwargs.get("structured")),
                    "fallback_used": bool(kwargs.get("fallback_used")),
                    "finish_reason": kwargs.get("finish_reason"),
                    "prompt_tokens": kwargs.get("prompt_tokens"),
                    "completion_tokens": kwargs.get("completion_tokens"),
                    "total_tokens": kwargs.get("total_tokens"),
                    "latency_seconds": kwargs.get("latency_seconds"),
                    "max_tokens": kwargs.get("max_tokens"), "prompt_sha256": _PENDING_SHA.get(),
                })
            return orig_record(**kwargs)

        llm_metrics.record = record  # type: ignore[assignment]
        for name, structured in (("complete", False), ("structured_complete", True)):
            self._had_attr[name] = name in vars(llm_client)
            orig = getattr(llm_client, name)
            self._orig_methods[name] = vars(llm_client).get(name)
            setattr(llm_client, name, self._wrap(orig, structured))
        self.installed = True

    @staticmethod
    def _wrap(orig: Callable[..., Awaitable[Any]], structured: bool) -> Callable[..., Awaitable[Any]]:
        async def wrapper(messages: Any, *args: Any, **kwargs: Any) -> Any:
            sha = prompt_sha256(messages)
            calls = _CASE_CALLS.get()
            if calls is not None:
                calls.outer.append({"label": kwargs.get("label", ""), "structured": structured, "prompt_sha256": sha})
            token = _PENDING_SHA.set(sha)
            try:
                return await orig(messages, *args, **kwargs)
            finally:
                _PENDING_SHA.reset(token)

        return wrapper

    def uninstall(self) -> None:
        if not self.installed:
            return
        llm_metrics.record = self._orig_record  # type: ignore[assignment]
        for name in ("complete", "structured_complete"):
            if self._had_attr.get(name):
                setattr(llm_client, name, self._orig_methods[name])
            else:
                llm_client.__dict__.pop(name, None)
        self.installed = False


# --------------------------------------------------------------------------- guard ledger


@dataclass
class GuardLedger:
    """What `pipeline`'s `state.unit_scale_*` / `period_mismatch_*` collect; recorded per node
    as a delta so a forked run can rebuild the running totals from cached nodes."""

    unit_scale_violations: list[str] = field(default_factory=list)
    unit_scale_scans: dict[str, Any] = field(default_factory=dict)
    period_mismatch_violations: list[str] = field(default_factory=list)
    period_mismatch_scans: dict[str, Any] = field(default_factory=dict)

    def snapshot(self) -> dict[str, Any]:
        return {
            "uv": len(self.unit_scale_violations), "pv": len(self.period_mismatch_violations),
            "us": copy.deepcopy(self.unit_scale_scans), "ps": copy.deepcopy(self.period_mismatch_scans),
        }

    def delta(self, snap: dict[str, Any]) -> dict[str, Any]:
        return {
            "unit_scale_violations": list(self.unit_scale_violations[snap["uv"]:]),
            "period_mismatch_violations": list(self.period_mismatch_violations[snap["pv"]:]),
            "unit_scale_scans": {k: v for k, v in self.unit_scale_scans.items() if snap["us"].get(k) != v},
            "period_mismatch_scans": {k: v for k, v in self.period_mismatch_scans.items() if snap["ps"].get(k) != v},
        }

    def apply(self, delta: Mapping[str, Any]) -> None:
        self.unit_scale_violations.extend(delta.get("unit_scale_violations", []))
        self.period_mismatch_violations.extend(delta.get("period_mismatch_violations", []))
        self.unit_scale_scans.update(delta.get("unit_scale_scans", {}))
        self.period_mismatch_scans.update(delta.get("period_mismatch_scans", {}))


def numeric_turn_guard(stage: str, inp: ReplayInputs, ledger: GuardLedger) -> Callable[[str, str, int], str]:
    """Copy of `pipeline._numeric_turn_guard` (~line 1985) with `kr_gt=kr_news_gt=""`."""

    def guard(content: str, speaker: str, round_num: int) -> str:
        if not content:
            return content
        guarded, desc = enforce_numeric_guards(
            content, node=f"{stage}[r{round_num} {speaker}]",
            kr_block=f"{inp.kr_gt}\n\n{inp.kr_risk_extra}" if inp.kr_risk_extra else inp.kr_gt,
            language=inp.language, usd_check=True, soft_eok_block=inp.kr_news_gt,
            scans_out=ledger.unit_scale_scans,
            period_out=ledger.period_mismatch_violations,
            period_scans_out=ledger.period_mismatch_scans,
        )
        if desc:
            ledger.unit_scale_violations.extend(desc)
        return guarded

    return guard


def guard_fields(obj: Any, fields: list[str], node_prefix: str, inp: ReplayInputs, ledger: GuardLedger, **kw: Any) -> dict[str, str]:
    """Copy of `pipeline._guard_fields` (~line 2019): returns the changed fields for `model_copy`."""
    updates: dict[str, str] = {}
    for f in fields:
        val = getattr(obj, f, None)
        if not isinstance(val, str) or not val:
            continue
        guarded, desc = enforce_numeric_guards(
            val, node=f"{node_prefix}.{f}", kr_block=inp.kr_gt, language=inp.language,
            usd_check=True, soft_eok_block=inp.kr_news_gt, banner=False, scans_out=ledger.unit_scale_scans,
            period_out=ledger.period_mismatch_violations, period_scans_out=ledger.period_mismatch_scans, **kw,
        )
        if desc:
            ledger.unit_scale_violations.extend(desc)
        if guarded != val:
            updates[f] = guarded
    return updates


def dump_model(obj: Any) -> dict[str, Any]:
    """Warning-free pydantic dump (gate `model_copy` can leave a Flat* enum member in a field
    typed with the held enum, which `model_dump(mode="json")` would warn about)."""
    out: dict[str, Any] = {}
    for name in type(obj).model_fields:
        value = getattr(obj, name)
        out[name] = value.value if isinstance(value, enum.Enum) else value
    return out


# --------------------------------------------------------------------------- case state / nodes


class ForkError(Exception):
    """A cached (baseline) node cannot be restored for this case."""


class PersistError(Exception):
    """Writing a case result failed (never classified as a network error)."""


class CaseNodeError(Exception):
    def __init__(self, node: str, original: BaseException):
        super().__init__(f"{node}: {original!r}")
        self.node = node
        self.original = original


@dataclass
class CaseState:
    inp: ReplayInputs
    resolved_profile: Any
    ledger: GuardLedger
    calls: CaseCalls
    san_box: list[str]
    inj_box: list[dict]
    plan: ResearchPlan | None = None
    proposal: TraderProposal | None = None
    risk_state: DebateState | None = None
    decision: PortfolioDecision | None = None  # raw PM output, before the gates
    pm_note: str = ""
    nodes: dict[str, dict[str, Any]] = field(default_factory=dict)


def _bullish(x: str | None) -> bool | None:
    return None if x is None else x in BULL


async def run_rm(st: CaseState) -> dict[str, Any]:
    inp = st.inp
    plan = await research_manager.synthesize(
        inp.reports_dict(), inp.research_debate(), language=inp.language, identity=inp.identity_note,
        held=inp.held, session_note=inp.session_note, period_note=inp.period_note,
        position_context=inp.position_context,
    )
    updates = guard_fields(plan, ["rationale", "strategic_actions"], "research_plan", inp, st.ledger)
    if updates:
        plan = plan.model_copy(update=updates)
    st.plan = plan
    return {"output": dump_model(plan), "guard_changed_fields": sorted(updates)}


async def run_trader(st: CaseState) -> dict[str, Any]:
    inp = st.inp
    proposal = await trader.propose(
        inp.reports_dict(), inp.research_debate(), st.plan, inp.position_context,
        language=inp.language, profile=st.resolved_profile, identity=inp.identity_note, held=inp.held,
        session_note=inp.session_note, period_note=inp.period_note,
    )
    updates = guard_fields(proposal, ["reasoning", "position_sizing"], "trader_proposal", inp, st.ledger)
    if updates:
        proposal = proposal.model_copy(update=updates)
    st.proposal = proposal
    return {"output": dump_model(proposal), "guard_changed_fields": sorted(updates)}


def _neutral_stances(state: DebateState) -> dict[str, str | None]:
    neutrals = [t for t in state.turns if t.speaker == "Neutral"]
    first = next((t for t in neutrals if t.round == 1), None)
    return {
        "r1": risk_debators._neutral_verdict(first.content)[0] if first else None,
        "final": risk_debators._neutral_verdict(neutrals[-1].content)[0] if neutrals else None,
    }


async def run_risk_debate(st: CaseState) -> dict[str, Any]:
    inp = st.inp
    speakers = risk_debators.build_all(
        inp.reports_dict(), st.proposal, inp.risk_extra, language=inp.language,
        identity=inp.identity_note, held=inp.held, session_note=inp.session_note, period_note=inp.period_note,
        sizing_context=inp.debate_sizing_context,
        research_plan_note=render_research_plan(st.plan) if st.plan else "",
    )
    state = await run_debate(
        state=DebateState(parties=["Aggressive", "Conservative", "Neutral"]),
        speakers=speakers,
        max_rounds=settings.max_risk_rounds,
        on_round=None,
        language=inp.language,
        check_convergence=False,
        parallel_first_round=settings.debate_parallel_first_round,
        turn_guard=numeric_turn_guard("risk_debate", inp, st.ledger),
    )
    st.risk_state = state
    return {
        "turns": [{"speaker": t.speaker, "content": t.content, "round": t.round} for t in state.turns],
        "rounds_completed": state.rounds_completed,
        "illegal_ratings": risk_debators.check_illegal_ratings(state, held=inp.held),
        "neutral_reversal": risk_debators.check_neutral_reversal(state),
        "stances": _neutral_stances(state),
    }


async def run_pm(st: CaseState) -> dict[str, Any]:
    inp = st.inp
    note = pipeline._desk_consensus_note(st.plan, st.proposal, inp.kr_risk_extra, inp.debate_sizing_context)
    decision = await portfolio_manager.decide(
        st.proposal, st.risk_state, inp.past_context, inp.position_context, inp.volatility_context,
        inp.risk_cap_context, inp.portfolio_risk_context, inp.execution_status_context,
        sibling_context=inp.sibling_context, language=inp.language, profile=st.resolved_profile,
        held=inp.held, identity=inp.identity_note, session_note=inp.session_note,
        period_note=inp.period_note, degradation_note=pipeline._degradation_note(list(inp.degraded_analysts)),
        desk_consensus_note=note,
    )
    st.decision, st.pm_note = decision, note
    return {"desk_consensus_note": note, "pre_gate_output": dump_model(decision)}


async def run_gates(st: CaseState) -> dict[str, Any]:
    """Deterministic gates in the live order (`pipeline.py` ~2378-2444). The unjustified-exit
    gate is a no-op for a flat ticker (`if not held`, ~line 353) so it is skipped and `{}` recorded."""
    inp = st.inp
    assert not inp.held, "replay range is flat tickers only"
    degraded = list(inp.degraded_analysts)
    record: dict[str, Any] = {}
    d = pipeline._apply_consensus_override_gate(st.decision, st.plan, st.proposal, record_out=record)
    before_hysteresis = d.rating.value
    d = pipeline._apply_hysteresis_gate(d, inp.held)
    hysteresis_applied = d.rating.value != before_hysteresis and d.executive_summary.startswith("[Hysteresis gate:")
    d = pipeline._apply_degradation_notice(d, degraded)
    updates = guard_fields(d, ["executive_summary", "investment_thesis"], "portfolio_manager", inp, st.ledger)
    updates.update(guard_fields(
        d, ["card_reason_ko", "card_reason_en"], "portfolio_manager", inp, st.ledger, inline_note=False,
    ))
    if updates:
        d = d.model_copy(update=updates)
    return {
        "consensus_override": record,
        "unjustified_exit_gate": {},
        "hysteresis_applied": hysteresis_applied,
        "hysteresis_from": before_hysteresis if hysteresis_applied else None,
        "degradation_applied": bool(degraded),
        "pm_guard_changed_fields": sorted(updates),
        "final_output": dump_model(d),
    }


NODE_FUNCS: dict[str, Callable[[CaseState], Awaitable[dict[str, Any]]]] = {
    "rm": run_rm, "trader": run_trader, "risk_debate": run_risk_debate, "pm": run_pm, "gates": run_gates,
}


def harness_node_digests() -> dict[str, str]:
    """Source digest of each harness node function (part of the node fingerprint)."""
    import inspect

    return {
        node: hashlib.sha256(inspect.getsource(fn).encode("utf-8")).hexdigest()[:16]
        for node, fn in NODE_FUNCS.items()
    }


def _restore_node(st: CaseState, name: str, block: Mapping[str, Any]) -> dict[str, Any]:
    """Rebuild the objects a later node needs from a baseline node block (no LLM call)."""
    if name == "rm":
        st.plan = ResearchPlan.model_validate(block["output"])
    elif name == "trader":
        st.proposal = TraderProposal.model_validate(block["output"])
    elif name == "risk_debate":
        from app.state import DebateTurn

        state = DebateState(parties=["Aggressive", "Conservative", "Neutral"])
        state.turns = [DebateTurn(speaker=t["speaker"], content=t["content"], round=t["round"]) for t in block["turns"]]
        state.rounds_completed = block.get("rounds_completed", max((t["round"] for t in block["turns"]), default=0))
        st.risk_state = state
    elif name == "pm":
        st.decision = PortfolioDecision.model_validate(block["pre_gate_output"])
        st.pm_note = block["desk_consensus_note"]
    else:
        raise ForkError(f"node {name} cannot be restored from cache")
    st.ledger.apply(block.get("guard_delta", {}))
    st.san_box.extend(block.get("sanitizations", []))
    st.inj_box.extend(block.get("injection_flags", []))
    cached = copy.deepcopy(dict(block))
    cached.update({
        "source": "baseline_cache", "baseline_wall_s": block.get("wall_s"),
        "wall_s": 0.0, "calls": [], "tokens": 0,
    })
    return cached


async def _run_live_node(st: CaseState, name: str) -> dict[str, Any]:
    ledger_snap = st.ledger.snapshot()
    calls_before, san_before, inj_before = len(st.calls.records), len(st.san_box), len(st.inj_box)
    started = time.monotonic()
    try:
        block = await NODE_FUNCS[name](st)
    except Exception as exc:  # ReplaySafetyViolation (BaseException) passes through untouched
        raise CaseNodeError(name, exc) from exc
    node_calls = st.calls.records[calls_before:]
    block.update({
        "source": "llm", "wall_s": round(time.monotonic() - started, 3), "calls": node_calls,
        "tokens": sum(c.get("total_tokens") or 0 for c in node_calls),
        "guard_delta": st.ledger.delta(ledger_snap),
        "sanitizations": list(st.san_box[san_before:]), "injection_flags": list(st.inj_box[inj_before:]),
    })
    return block


def _summary(inp: ReplayInputs, nodes: Mapping[str, Any]) -> dict[str, Any]:
    rm = nodes["rm"]["output"]["recommendation"]
    tr = nodes["trader"]["output"]["action"]
    neutral = nodes["risk_debate"]["stances"]["final"]
    final_out = nodes["gates"]["final_output"]
    pre_gate = nodes["pm"]["pre_gate_output"]["rating"]
    return {
        "rm": rm, "trader": tr, "neutral_final": neutral, "pm_pre_gate": pre_gate,
        "final": final_out["rating"], "confidence": final_out.get("confidence"),
        "rm_bullish": _bullish(rm), "final_bullish": _bullish(final_out["rating"]),
        "neutral_bullish": _bullish(neutral),
    }


def _calls_summary(calls: CaseCalls) -> dict[str, Any]:
    by_label: dict[str, dict[str, Any]] = {}
    for rec in calls.records:
        entry = by_label.setdefault(rec["label"] or "", {"n": 0, "total_tokens": 0, "latency_sum": 0.0})
        entry["n"] += 1
        entry["total_tokens"] += rec.get("total_tokens") or 0
        entry["latency_sum"] += rec.get("latency_seconds") or 0.0
    sha_by_label: dict[str, list[str]] = {}
    for rec in calls.outer:
        sha_by_label.setdefault(rec["label"] or "", []).append(rec["prompt_sha256"])
    return {
        "n": len(calls.records),
        "n_structured": sum(1 for r in calls.records if r["structured"]),
        "n_fallback": sum(1 for r in calls.records if r["structured"] and r["fallback_used"]),
        "n_length": sum(1 for r in calls.records if r["finish_reason"] == "length"),
        "by_label": by_label,
        "prompt_sha256_by_label": sha_by_label,
    }


async def run_case(
    inp: ReplayInputs,
    *,
    run_id: str = "",
    arm: str = "baseline",
    repeat: int = 1,
    variant: str = "baseline",
    fork_at: str = "rm",
    baseline_case: Mapping[str, Any] | None = None,
    attempt: int = 1,
) -> dict[str, Any]:
    """One replay of one case. Nodes before `fork_at` are restored from `baseline_case`;
    the rest run against the LLM. Raises `CaseNodeError` (node + original) on failure."""
    if fork_at not in NODES:
        raise ValueError(f"unknown fork_at {fork_at!r}")
    started_at = utc_now_iso()
    started = time.monotonic()
    with (
        account_mode_scope("real"), backtest_scope(),
        llm_metrics.track_run_tokens() as token_box,
        llm_metrics.track_run_sanitizations() as san_box,
        llm_metrics.track_run_injection_flags() as inj_box,
    ):
        calls = CaseCalls()
        ctx_token = _CASE_CALLS.set(calls)
        try:
            resolved_profile = runtime_settings.rule("ai_profile")
            if resolved_profile != settings.ai_profile:
                raise RuntimeError("resolved ai_profile differs from the pinned setting")
            st = CaseState(inp, resolved_profile, GuardLedger(), calls, san_box, inj_box)
            fork_index = NODES.index(fork_at)
            if fork_index > 0:
                if baseline_case is None or baseline_case.get("status") != "ok":
                    raise ForkError("baseline case missing or not ok")
                if baseline_case.get("input_hash") != inp.input_hash():
                    raise ForkError("input_hash differs from the baseline case")
            for index, name in enumerate(NODES):
                if index < fork_index:
                    st.nodes[name] = _restore_node(st, name, baseline_case["nodes"][name])  # type: ignore[index]
                else:
                    st.nodes[name] = await _run_live_node(st, name)
            tokens_total = token_box[0]
        finally:
            _CASE_CALLS.reset(ctx_token)
    return {
        "schema": CASE_SCHEMA, "case_id": inp.case_id, "memory_id": inp.memory_id, "ticker": inp.ticker,
        "trade_date": inp.trade_date.isoformat(), "era": inp.era, "split": inp.split, "run_id": run_id,
        "arm": arm, "repeat": repeat, "variant": variant, "fork_at": fork_at,
        "input_hash": inp.input_hash(), "contract_version": "rm-start-v1",
        "covariates": inp.covariates(), "contract_checks": inp.contract_checks,
        "nodes": st.nodes, "sweeps": "not_run",
        "sanitizations": list(san_box), "injection_flags": list(inj_box),
        "summary": _summary(inp, st.nodes), "stored": inp.stored.as_dict(),
        "tokens_total": tokens_total, "calls_summary": _calls_summary(calls),
        "wall_s": round(time.monotonic() - started, 3), "status": "ok", "error": None,
        "attempt": attempt, "started_at": started_at, "finished_at": utc_now_iso(),
    }


# --------------------------------------------------------------------------- run loop


NETWORK_ERROR_NAMES = frozenset({"APIConnectionError", "APITimeoutError", "ConnectError", "ReadTimeout"})


def is_network_error(exc: BaseException) -> bool:
    """True if ANY link of the exception chain is a connectivity failure. `structured_complete`
    ends with `raise ValueError(...) from last_error`, so a dead vLLM surfaces as a ValueError whose
    `__cause__` is the SDK's APIConnectionError; checking only the outer type would never trip the
    breaker (review M1). Follows `__cause__`/`__context__` (seen-set guards cycles) and unwraps
    `CaseNodeError.original`."""
    seen: set[int] = set()
    stack: list[BaseException | None] = [exc]
    while stack:
        cur = stack.pop()
        if cur is None or id(cur) in seen:
            continue
        seen.add(id(cur))
        if isinstance(cur, CaseNodeError):
            stack.append(cur.original)
        if isinstance(cur, (OSError, TimeoutError, asyncio.TimeoutError)):
            return True
        if any(cls.__name__ in NETWORK_ERROR_NAMES for cls in type(cur).__mro__):
            return True
        stack.append(cur.__cause__)
        stack.append(cur.__context__)
    return False


@dataclass
class RunConfig:
    run_dir: Path
    run_id: str
    arm: str = "baseline"
    repeat: int = 1
    variant: str = "baseline"
    fork_at: str = "rm"
    k: int = 1
    case_timeout: float = 1800.0
    max_attempts: int = 2
    max_consecutive_failures: int = 3
    allow_market_hours: bool = False
    heartbeat_s: float = 15.0
    now_fn: Callable[[], datetime] = safety.live_now_kst
    baseline_reader: Callable[[str], dict[str, Any] | None] | None = None
    case_fn: Callable[..., Awaitable[dict[str, Any]]] | None = None  # test seam; default run_case
    echo: Callable[[str], None] | None = print


@dataclass
class RunOutcome:
    status: str
    exit_code: int
    done: int = 0
    errors: int = 0
    skipped_ok: int = 0
    gave_up: int = 0
    stop_reason: str | None = None
    remaining: int = 0  # cases without an ok result when the loop ended
    retryable: int = 0  # of those, how many a rerun of the same command would retry


class ManifestMismatch(Exception):
    """Resuming into a directory whose manifest key differs (exit code 2)."""


def _ok_path(run_dir: Path, case_id: str) -> Path:
    return run_dir / "cases" / f"{case_id}.json"


def _error_paths(run_dir: Path, case_id: str) -> list[Path]:
    return sorted((run_dir / "errors").glob(f"{case_id}.a*.json")) if (run_dir / "errors").exists() else []


def read_ok_case(run_dir: Path, case_id: str) -> dict[str, Any] | None:
    path = _ok_path(run_dir, case_id)
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if data.get("status") == "ok" else None


def write_json(path: Path, obj: Any) -> None:
    atomic_write_text(path, json.dumps(obj, indent=1, ensure_ascii=False, default=str) + "\n")


class RunSession:
    """Owns one run directory: manifest, progress log, checkpointed case files, the loop."""

    def __init__(self, cfg: RunConfig, manifest_base: dict[str, Any]):
        self.cfg = cfg
        self.manifest: dict[str, Any] = dict(manifest_base)
        self.counts = {"done": 0, "error": 0, "skipped_ok": 0, "gave_up": 0}
        self.in_flight = 0
        self._stop_reason: str | None = None
        self._network_streak = 0
        self._safety_hit = False
        self._tasks: list[asyncio.Task] = []
        self._trips_seen = 0

    # -- manifest ---------------------------------------------------------
    @property
    def manifest_path(self) -> Path:
        return self.cfg.run_dir / "manifest.json"

    def load_or_init_manifest(self) -> None:
        self.cfg.run_dir.mkdir(parents=True, exist_ok=True)
        if self.manifest_path.exists():
            existing = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            for key in ("base_key", "arm", "variant", "run_id"):
                if existing.get(key) != self.manifest.get(key):
                    raise ManifestMismatch(f"manifest {key} differs: {existing.get(key)!r} != {self.manifest.get(key)!r}")
            self.manifest["resumed"] = int(existing.get("resumed", 0)) + 1
            self.manifest["started_at"] = existing.get("started_at", self.manifest.get("started_at"))
            self.manifest["windows"] = list(existing.get("windows", []))
        else:
            self.manifest.setdefault("started_at", utc_now_iso())
            self.manifest["windows"] = []
        # One window per harness PROCESS: rows created inside one are the harness's own (hard FAIL
        # in the DB invariants); a resumed run must not cover the pause between two processes.
        now = utc_now_iso()
        self.manifest["windows"].append({"start": now, "end": now})
        self._trips_seen = safety.trip_count()
        self.write_manifest("running")

    def write_manifest(self, status: str | None = None, *, best_effort: bool = False) -> None:
        """`best_effort` (heartbeats, per-case updates): an OSError (AV/indexer lock, full disk) is
        reported and swallowed so a transient file problem cannot kill the whole run (review minor 1)."""
        if status is not None:
            self.manifest["status"] = status
        in_flight_llm = getattr(getattr(llm_client, "_limiter", None), "_in_flight", None)
        ts = utc_now_iso()
        self.manifest["heartbeat"] = {
            "ts": ts, "in_flight": self.in_flight, "llm_in_flight": in_flight_llm,
            "done": self.counts["done"], "error": self.counts["error"],
        }
        if self.manifest.get("windows"):
            self.manifest["windows"][-1]["end"] = ts
        self.manifest["safety_trips"] = safety.trip_count()
        try:
            write_json(self.manifest_path, self.manifest)
        except OSError as exc:
            if not best_effort:
                raise
            self.progress(f"[warn] manifest write failed, continuing: {exc}")

    def progress(self, line: str) -> None:
        try:
            with open(self.cfg.run_dir / "progress.log", "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except OSError:
            pass  # the console echo below still shows the line
        if self.cfg.echo:
            self.cfg.echo(line)

    # -- one case ---------------------------------------------------------
    async def _execute(self, inp: ReplayInputs, attempt: int, sem: asyncio.Semaphore) -> None:
        cfg = self.cfg
        case_fn = cfg.case_fn or run_case
        started = time.monotonic()
        self.in_flight += 1
        try:
            baseline_case = cfg.baseline_reader(inp.case_id) if cfg.baseline_reader else None
            result = await asyncio.wait_for(
                case_fn(
                    inp, run_id=cfg.run_id, arm=cfg.arm, repeat=cfg.repeat, variant=cfg.variant,
                    fork_at=cfg.fork_at, baseline_case=baseline_case, attempt=attempt,
                ),
                timeout=cfg.case_timeout,
            )
            if self._trips_grew():  # a blocked call was swallowed inside the case: do not trust/persist it
                raise safety.ReplaySafetyViolation("blocked-call counter grew during the case")
            try:
                write_json(_ok_path(cfg.run_dir, inp.case_id), result)
            except OSError as exc:  # a disk problem is not a connectivity problem
                raise PersistError(f"cannot write the case result: {exc}") from exc
            self.counts["done"] += 1
            self._network_streak = 0
            s = result["summary"]
            self.progress(
                f"{inp.case_id} ok rm={s['rm']} final={s['final']} wall={result['wall_s']}s tokens={result['tokens_total']}"
            )
        except safety.ReplaySafetyViolation as exc:
            self._on_safety(inp, attempt, exc, started)
        except Exception as exc:  # noqa: BLE001 - a case failure must not stop the run
            if self._trips_grew():  # a swallowed block trip may be the real cause of this failure
                self._on_safety(inp, attempt, exc, started)
            else:
                node = exc.node if isinstance(exc, CaseNodeError) else None
                network = not isinstance(exc, PersistError) and is_network_error(exc)
                self._record_error(inp, attempt, exc, started, node=node, network=network)
                self.counts["error"] += 1
                if network:
                    self._network_streak += 1
                    if self._network_streak >= cfg.max_consecutive_failures:
                        self._stop_reason = self._stop_reason or "paused_network"
                else:
                    self._network_streak = 0
                self.progress(
                    f"{inp.case_id} error attempt={attempt}{' (network: attempt not consumed)' if network else ''} "
                    f"{type(exc).__name__}: {str(exc)[:160]}"
                )
        finally:
            self.in_flight -= 1
            sem.release()
            self.write_manifest(best_effort=True)

    def _trips_grew(self) -> bool:
        return safety.trip_count() > self._trips_seen

    def _on_safety(self, inp: ReplayInputs, attempt: int, exc: BaseException, started: float) -> None:
        self._safety_hit = True
        self._stop_reason = "aborted_safety"
        self._trips_seen = safety.trip_count()
        self._record_error(inp, attempt, exc, started, node="safety")
        self.progress(f"{inp.case_id} SAFETY VIOLATION {type(exc).__name__}: {exc}")
        current = asyncio.current_task()
        for other in self._tasks:  # a blocked side effect stops the whole run
            if other is not current and not other.done():
                other.cancel()

    def _record_error(
        self, inp: ReplayInputs, attempt: int, exc: BaseException, started: float, node: str | None,
        network: bool = False,
    ) -> None:
        original = exc.original if isinstance(exc, CaseNodeError) else exc
        payload = {
            "schema": CASE_SCHEMA, "case_id": inp.case_id, "status": "error", "attempt": attempt,
            "run_id": self.cfg.run_id, "input_hash": inp.input_hash(), "network": network,
            "error": {
                "type": type(original).__name__, "message": str(original)[:2000], "node": node,
                "traceback": "".join(traceback.format_exception(type(original), original, original.__traceback__))[-6000:],
            },
            "wall_s": round(time.monotonic() - started, 3), "finished_at": utc_now_iso(),
        }
        # Connectivity failures are logged as `.net<ns>.json`: `_assignable` counts only `.a<n>.json`,
        # so a dead vLLM never uses up a case's attempt budget (review M1).
        name = f"{inp.case_id}.net{time.time_ns()}.json" if network else f"{inp.case_id}.a{attempt}.json"
        write_json(self.cfg.run_dir / "errors" / name, payload)

    # -- the loop ---------------------------------------------------------
    def _assignable(self, ordered: list[ReplayInputs]) -> list[tuple[ReplayInputs, int]]:
        """Skip finished cases; retry failed ones while attempts < max_attempts."""
        out: list[tuple[ReplayInputs, int]] = []
        for inp in ordered:
            if read_ok_case(self.cfg.run_dir, inp.case_id) is not None:
                self.counts["skipped_ok"] += 1
                continue
            attempts = len(_error_paths(self.cfg.run_dir, inp.case_id))
            if attempts >= self.cfg.max_attempts:
                self.counts["gave_up"] += 1
                continue
            out.append((inp, attempts + 1))
        return out

    async def run(self, cases: list[ReplayInputs]) -> RunOutcome:
        cfg = self.cfg
        self.load_or_init_manifest()
        ordered = sorted(cases, key=lambda c: (c.trade_date, c.ticker))
        queue = self._assignable(ordered)
        sem = asyncio.Semaphore(cfg.k)
        tasks = self._tasks
        heartbeat = asyncio.create_task(self._heartbeat_loop())
        status = "done"
        unexpected: list[BaseException] = []
        try:
            for inp, attempt in queue:
                await sem.acquire()
                if self._stop_reason:
                    sem.release()
                    break
                if not cfg.allow_market_hours and safety.market_window_open(cfg.now_fn()):
                    self._stop_reason = "paused_window"
                    sem.release()
                    break
                tasks.append(asyncio.create_task(self._execute(inp, attempt, sem)))
            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                unexpected = [
                    r for r in results if isinstance(r, BaseException) and not isinstance(r, asyncio.CancelledError)
                ]
        finally:
            heartbeat.cancel()
            try:
                await heartbeat
            except asyncio.CancelledError:
                pass
        if unexpected and not self._safety_hit:
            self.write_manifest("aborted_error")
            raise unexpected[0]
        if self._stop_reason:
            status = self._stop_reason
        exit_code = {"done": EXIT_OK, "aborted_safety": EXIT_SAFETY}.get(status, EXIT_PAUSED)
        remaining = [c for c in ordered if read_ok_case(cfg.run_dir, c.case_id) is None]
        retryable = [c for c in remaining if len(_error_paths(cfg.run_dir, c.case_id)) < cfg.max_attempts]
        if status == "done" and remaining:
            exit_code = EXIT_INCOMPLETE  # 0 only when no case is left without an ok result
            self.progress(
                f"{len(remaining)} cases in error ({len(retryable)} retryable, {len(remaining) - len(retryable)} out of "
                "attempts) - rerun the same command to retry"
            )
        self.manifest["incomplete"] = {"remaining": len(remaining), "retryable": len(retryable)}
        self.manifest["finished_at"] = utc_now_iso()
        self.write_manifest(status)
        return RunOutcome(
            status=status, exit_code=exit_code, done=self.counts["done"], errors=self.counts["error"],
            skipped_ok=self.counts["skipped_ok"], gave_up=self.counts["gave_up"], stop_reason=self._stop_reason,
            remaining=len(remaining), retryable=len(retryable),
        )

    async def _heartbeat_loop(self) -> None:
        while True:
            await asyncio.sleep(self.cfg.heartbeat_s)
            self.write_manifest(best_effort=True)


def stdout_utf8() -> None:
    """Windows console codepage fix (same as `reasoning_ab_parallel`)."""
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
        sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
