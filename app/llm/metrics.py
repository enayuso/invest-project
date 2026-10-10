"""In-process LLM call observability — records who called the model, how
many tokens it cost, and how long it took, so agent-necessity/cost decisions
(see the agent-architecture-roadmap memory) have real data behind them
instead of a hand count of call sites.

Deliberately NOT a per-agent concern: every app.llm.client call funnels
through this one module (the single place LLMClient itself calls into), so
no agent file needs its own logging/timing code.
LLM 호출 관측 — 각 에이전트 코드에 흩어 넣지 않고 llm_client 한 곳에서만 기록
"""

from __future__ import annotations

import contextlib
import contextvars
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime

from app.clock import get_clock

_MAX_RECORDS = 2000  # bounded ring buffer — a long-running dashboard process must not leak memory


@dataclass
class LLMCallMetric:
    timestamp: datetime
    label: str  # caller identity, e.g. "market_analyst", "bull_researcher", "" if uninstrumented
    model: str
    reasoning: bool
    structured: bool  # complete() vs structured_complete()
    # True if this call is structured_complete's unconstrained-completion
    # fallback (native response_format returned nothing usable) — see
    # app.llm.client.LLMClient.structured_complete's docstring. A rising
    # rate for a given model is the signal that model's guided-decoding
    # support needs attention, exactly what _extract_json's own comment
    # anticipated before this field existed to actually measure it.
    # 네이티브 구조화 출력 실패로 폴백(스키마 텍스트지시+일반완성)을 탔는지 여부
    fallback_used: bool
    latency_seconds: float
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    # Why vLLM stopped generating — `"length"` means the response hit
    # llm_max_output_tokens (app.config) and was TRUNCATED. That's the only
    # signal that distinguishes a truncated response from a genuinely
    # malformed/empty one: a truncated structured_complete call gets
    # misdiagnosed as "failed to produce valid schema" (it's actually just
    # cut off mid-JSON), and a truncated complete() report passes downstream
    # indistinguishable from a complete one. None when absent from the API
    # response (e.g. the fallback fake response some tests construct).
    finish_reason: str | None = None
    # The effective per-call ceiling actually sent as `max_tokens` — usually
    # `settings.llm_max_output_tokens`, but a caller can override it lower
    # (2026-09-07: app.agents.debater's round-1 debate-turn budget,
    # `settings.debate_round1_max_output_tokens`). Lets a `finish_reason ==
    # "length"` truncation be attributed to the ceiling that actually applied
    # to THIS call, not always the global default.
    max_tokens: int | None = None


@dataclass
class LabelSummary:
    label: str
    call_count: int
    total_tokens: int
    mean_latency_seconds: float


_records: deque[LLMCallMetric] = deque(maxlen=_MAX_RECORDS)

# Per-run token accumulation, scoped via contextvars the same way
# app.dataflows.kis_provider.candle_run_cache scopes its own per-run cache —
# a plain module-level counter would double-count/race across the several
# ticker analyses app.live.auto_trade_loop runs concurrently
# (auto_trade_max_concurrent_analyses); a contextvar set inside one
# asyncio.Task is invisible to sibling tasks, which is exactly the isolation
# a per-ticker "how many tokens did THIS decision cost" figure needs.
_current_run_tokens: contextvars.ContextVar[list[int] | None] = contextvars.ContextVar(
    "_current_run_tokens", default=None
)


@contextlib.contextmanager
def track_run_tokens() -> Iterator[list[int]]:
    """Wrap one `app.orchestrator.pipeline.run_pipeline()` call in this to
    have every `record()` call made inside it (directly or through any
    `await`) add its `total_tokens` into the yielded single-element list —
    read `box[0]` at any point, including after the context exits, to get
    the running total for that one ticker/decision. Calls made outside any
    `track_run_tokens()` scope (nothing set, or a different task's scope)
    are simply not counted here, same as before this existed — the
    unscoped `_records` ring buffer this feeds `record()` from is
    unaffected either way."""
    box = [0]
    token = _current_run_tokens.set(box)
    try:
        yield box
    finally:
        _current_run_tokens.reset(token)


def current_run_tokens() -> int | None:
    """The running total for the innermost enclosing `track_run_tokens()`
    scope, or `None` if called outside one — a caller nested inside that
    scope (e.g. `app.orchestrator.pipeline`'s `store_pending` call, itself
    inside `run_pipeline`'s `track_run_tokens()`) reads the total-so-far
    without needing to exit the context first."""
    box = _current_run_tokens.get()
    return box[0] if box is not None else None


# Per-run sanitization log — same contextvar-per-Task isolation as
# _current_run_tokens. Two event families share this list, namespaced by
# entry prefix (see track_run_sanitizations' docstring):
#   1. Reasoning-leak strips. 2026-09-01, decision id=1156: the self-hosted
#      model published ~330 lines of its own deliberation as a risk-debate
#      turn (ending in a literal "</think>") and it CHANGED THE DECISION.
#      `app.llm.client._strip_reasoning_leak` removes tagged leaks; this
#      records that it happened so a later audit can ask "how often, on
#      which node, did the model leak" without re-reading transcripts.
#   2. Injection-banner events (`injection_banner:*`). Also 2026-09-01, root
#      cause #9, decision id=1157/id=1158:
#      `app.agents.untrusted_content.redact_injection_flag` classifies what
#      it found in the text a model wrote AFTER an INJECTION_SUSPECTED
#      banner — never strips it, only annotates and records the tier here.
_current_run_sanitizations: contextvars.ContextVar[list[str] | None] = contextvars.ContextVar(
    "_current_run_sanitizations", default=None
)


@contextlib.contextmanager
def track_run_sanitizations() -> Iterator[list[str]]:
    """Wrap one `run_pipeline()` call in this (alongside `track_run_tokens`)
    so every `note_sanitization(...)` made inside it appends to the yielded
    list. Two entry shapes share this list:
      - reasoning-leak strips: `"risk_conservative: think_block"` (also
        `untagged_suspected`/`declined_too_short`, which are recorded but are
        NOT strips — see `app.llm.client._strip_reasoning_leak`).
      - injection-banner events: `"risk_neutral: injection_banner:<tier>"`,
        `<tier>` one of `empty`/`rest_kept`/`rest_leak_suspected`/
        `rest_self_reversal`/`rest_self_reversal_weak` — see
        `app.agents.untrusted_content.redact_injection_flag`.
    Read after the context exits for the decision record; grep either family
    with its distinct prefix (`injection_banner:` vs. everything else)."""
    box: list[str] = []
    token = _current_run_sanitizations.set(box)
    try:
        yield box
    finally:
        _current_run_sanitizations.reset(token)


def current_run_sanitizations() -> list[str] | None:
    box = _current_run_sanitizations.get()
    return list(box) if box is not None else None


def note_sanitization(entry: str) -> None:
    box = _current_run_sanitizations.get()
    if box is not None:
        box.append(entry)


# Per-run injection-flag reasons — same contextvar-per-Task isolation as
# _current_run_sanitizations. Kept as a SEPARATE list, not more entries in
# _current_run_sanitizations, on purpose: that list is `list[str]` of
# parseable `"node: kind"` entries, shared between reasoning-leak strips and
# injection-banner tier records, and the 2026-09-04 injection-banner
# false-positive measurement depended entirely on being able to grep and
# split those entries mechanically. Folding a model's free-text one-sentence
# reason (arbitrary prose, newlines, colons, the marker string itself) into
# that same list would destroy that parseability. So the reason travels here
# as `list[dict]` — `{"agent", "tier", "reason"}` per flagged report —
# pairing 1:1 with the `injection_banner:<tier>` entry the sanitizations list
# already carries for the same report.
# 인젝션 자기신고 사유는 파싱 가능한 sanitizations 리스트를 오염시키지 않도록 별도 보관
_current_run_injection_flags: contextvars.ContextVar[list[dict] | None] = contextvars.ContextVar(
    "_current_run_injection_flags", default=None
)


@contextlib.contextmanager
def track_run_injection_flags() -> Iterator[list[dict]]:
    """Wrap one `run_pipeline()` call in this (alongside `track_run_tokens`
    and `track_run_sanitizations`) so every `note_injection_flag(...)` made
    inside it appends to the yielded list. Each entry is
    `{"agent": str, "tier": str, "reason": str}` — one per report that
    opened with an `INJECTION_SUSPECTED:` line, where `tier` is the same
    terminal tier string (`empty` / `rest_kept` / `rest_leak_suspected` /
    `rest_self_reversal`) recorded as `"<agent>: injection_banner:<tier>"` in
    `track_run_sanitizations`'s list, and `reason` is the model's own
    one-sentence explanation (stripped, truncated — see
    `app.agents.untrusted_content._FLAG_REASON_MAX`). Read after the context
    exits for the decision record. Separate from the sanitizations list so a
    free-text reason cannot break that list's `"node: kind"` parseability —
    see `_current_run_injection_flags`."""
    box: list[dict] = []
    token = _current_run_injection_flags.set(box)
    try:
        yield box
    finally:
        _current_run_injection_flags.reset(token)


def current_run_injection_flags() -> list[dict] | None:
    box = _current_run_injection_flags.get()
    return list(box) if box is not None else None


def note_injection_flag(entry: dict) -> None:
    box = _current_run_injection_flags.get()
    if box is not None:
        box.append(entry)


def record(
    *,
    label: str,
    model: str,
    reasoning: bool,
    structured: bool,
    latency_seconds: float,
    prompt_tokens: int | None,
    completion_tokens: int | None,
    total_tokens: int | None,
    fallback_used: bool = False,
    finish_reason: str | None = None,
    max_tokens: int | None = None,
) -> None:
    _records.append(
        LLMCallMetric(
            timestamp=get_clock().now(),
            label=label,
            model=model,
            reasoning=reasoning,
            structured=structured,
            fallback_used=fallback_used,
            latency_seconds=latency_seconds,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            finish_reason=finish_reason,
            max_tokens=max_tokens,
        )
    )
    box = _current_run_tokens.get()
    if box is not None and total_tokens:
        box[0] += total_tokens


def recent(n: int = 50) -> list[LLMCallMetric]:
    """Newest first."""
    return list(_records)[-n:][::-1]


def summary() -> list[LabelSummary]:
    """One row per label, sorted by total token usage (the costliest callers
    first) — the direct input to the agent-slimming decision the roadmap's
    stage 1 depends on."""
    by_label: dict[str, list[LLMCallMetric]] = {}
    for m in _records:
        by_label.setdefault(m.label, []).append(m)

    rows = [
        LabelSummary(
            label=label,
            call_count=len(metrics),
            total_tokens=sum(m.total_tokens or 0 for m in metrics),
            mean_latency_seconds=sum(m.latency_seconds for m in metrics) / len(metrics),
        )
        for label, metrics in by_label.items()
    ]
    return sorted(rows, key=lambda r: r.total_tokens, reverse=True)


def clear() -> None:
    """Test-only reset — production code never needs to clear this."""
    _records.clear()


def buffer_capacity() -> int:
    """Public accessor for the ring buffer's cap (`_MAX_RECORDS`) — callers
    that drain a bounded window (see `drain()`) need this to detect the one
    case draining itself can't rule out: a single window producing >= this
    many records, which could mean earlier records in that same window were
    evicted before the drain ever ran."""
    return _MAX_RECORDS


def drain() -> list[LLMCallMetric]:
    """Returns every record currently buffered, in insertion order, and empties
    the buffer — a snapshot-and-reset in one atomic step.

    Why drain() and not "filter records by timestamp since T" for a caller
    that wants just the metrics from one bounded window (e.g. one pipeline
    run):

    1. `_records` is a `deque(maxlen=_MAX_RECORDS)` (2000). A caller polling a
       long-lived process (e.g. app.experiments.language_ab, dozens of calls
       per run across 50+ runs in one process) can have its window's own
       records silently evicted by later, unrelated calls before it ever
       reads them — the deque does this with no warning, no exception,
       nothing to filter against because the record is simply gone. A
       drain-per-window discipline (drain at the start of each window, drain
       again to collect it) means a window's records can only be lost by
       exceeding 2000 calls WITHIN that one window, not by unrelated activity
       between windows.
    2. `record()` timestamps with `get_clock().now()`, and callers that run
       inside `app.config.backtest_scope()` (this experiment does, per
       ticker/date `as_of`) can have that clock pinned to a fixed historical
       instant rather than wall-clock time — see `app.clock.get_clock`. Two
       different attempts of the same backtest run can therefore report the
       IDENTICAL timestamp, or a timestamp that doesn't move forward between
       them, making "since T" filtering unreliable exactly when it matters
       most (retries).

    Draining sidesteps both: correctness depends only on buffer capacity
    within one window, never on clock behavior or eviction order.
    """
    records = list(_records)
    _records.clear()
    return records
