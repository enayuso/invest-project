"""Shared bounded tool-calling loop for the three tool-using analysts.

Each analyst has its own system prompt and tool set, but the mechanics —
call the model, execute any tool calls, feed results back, repeat until the
model stops calling tools or the iteration cap is hit — are identical, so
they're factored out once here rather than duplicated per analyst.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from datetime import date
import functools

from app.agents.untrusted_content import DATE_GROUNDING, INJECTION_WARNING, redact_injection_flag, wrap_untrusted
from app.cancellation import task_was_actually_cancelled
from app.config import settings
from app.llm import llm_client
from app.llm.language import DEFAULT_LANGUAGE

logger = logging.getLogger(__name__)

ToolFn = Callable[..., Awaitable[str]]


def bind_as_of(tools_map: dict[str, ToolFn], as_of: date | None) -> dict[str, ToolFn]:
    """{name: fn} -> {name: fn bound to as_of} — every tool-using analyst
    binds `as_of` into its dispatch closures the same way (never exposed to
    the LLM as a tool parameter, so backtesting can't request future data by
    asking for it), so the functools.partial(..., as_of=as_of) boilerplate
    is shared instead of repeated per analyst's dispatch dict.
    as_of를 각 도구에 바인딩(백테스트용, LLM에는 노출 안 함) — 3개 애널리스트 공용"""
    return {name: functools.partial(fn, as_of=as_of) for name, fn in tools_map.items()}


def ticker_label(ticker: str, company_name: str | None) -> str:
    """"{company_name} ({ticker})" when a display name is known, else the
    bare ticker — grounds the prompt with the real company identity instead
    of leaving the model to guess one from the symbol alone (a live
    transcript for 373220.KS, LG Energy Solution, once hallucinated "SK
    Hynix Inc." from the bare ticker, which then propagated through the
    rest of that decision's reasoning)."""
    return f"{company_name} ({ticker})" if company_name else ticker


# Grounding the label in the user prompt (ticker_label above) wasn't enough on
# its own: 086790.KS (하나금융지주) was mislabeled by the Market Analyst as
# "KB금융" (2026-07-30) and then as "현대차제강" (2026-08-03) even though the
# correct name was passed in every time — the model overrode a verified label
# with its own training-data guess for that ticker code. Every analyst that
# takes `company_name` must also append this to its SYSTEM_PROMPT so the model
# is told, not just shown, to trust it.
IDENTITY_GROUNDING_NOTE = (
    "\n\nThe company name given alongside the ticker (if any) has already been "
    "verified against the exchange's own ticker-code registry — treat it as "
    "ground truth. Do not restate, question, or substitute a different company "
    "name for that ticker based on your own training data; your own recollection "
    "of what a ticker code maps to can be wrong or stale, and the supplied name "
    "always overrides it. Reproduce that company name verbatim, exactly as "
    "supplied, in the first sentence of your report — including if it is written "
    "in a different script from the rest of your report. Do not translate it, "
    "transliterate it, romanize it, or replace it with the name of a parent, "
    "subsidiary, or affiliated company. If you want to give an English rendering "
    "for readability, put it in parentheses after the verbatim name on first "
    "mention, e.g. \"우리금융지주 (Woori Financial Group) (316140.KS)\" — the "
    "supplied form must still appear."
)

_FORCE_FINAL_PROMPT = (
    "You have reached the tool-call limit for this report. Give your final "
    "report now, using only the information already gathered."
)

async def run_tool_loop(
    *,
    system_prompt: str,
    user_prompt: str,
    tools: list[dict[str, Any]],
    dispatch: dict[str, ToolFn],
    agent_name: str,
    trusted_tools: frozenset[str] = frozenset(),
    language: str = DEFAULT_LANGUAGE,
) -> str:
    """`trusted_tools` names tools whose results are deterministic/verified
    vendor data (prices, financial statements, computed indicators) rather
    than genuinely externally-authored free text — their results are
    returned as-is, since wrapping them in the same "possibly-manipulated
    third-party content" framing used for real external text (news,
    insider-transaction descriptions) would contradict a system prompt that
    tells the model to treat that specific tool's output as authoritative.
    Everything not explicitly listed is wrapped, so the safe default for a
    caller that passes nothing is "wrap everything.\""""
    # 2026-08-05: confirmed live (196170.KS, Market Analyst) that this
    # "skip non-essential tools" instruction alone leaves a gap — the model
    # skipped get_investor_flow, then instead of omitting the topic, wrote a
    # full paragraph of generic/hypothetical supply-demand analysis "as if"
    # it had called the tool, complete with a leaked note-to-self admitting
    # it was a placeholder ("실제 리포트 작성 시에는 해당 툴의 출력값을 인용하여
    # 구체화해야 합니다"). Functionally identical to DATE_GROUNDING's own
    # "hedged prose instead of the real tool result" failure, just triggered
    # by a genuinely-skipped tool call instead of a mis-dated one — same fix
    # shape: name the concrete alternative (omit/note-skipped) so there's
    # never a plausible-sounding gap to fill with invented analysis.
    budget_note = (
        f"You have up to {settings.max_tool_iterations} tool-calling rounds for this report — "
        "call only the most decision-relevant tools, not everything available. If you skip a "
        "tool, skip (or briefly note as 'not checked this cycle') the topic it would have "
        "covered too — never write a hypothetical or generic paragraph describing what that "
        "analysis would probably say. A speculative 'framework' written in place of an actual "
        "tool result is exactly as misleading as fabricating the number itself."
    )
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": f"{system_prompt}\n\n{budget_note}\n\n{DATE_GROUNDING}\n\n{INJECTION_WARNING}"},
        {"role": "user", "content": user_prompt},
    ]

    for _ in range(settings.max_tool_iterations):
        response = await llm_client.complete(messages, tools=tools, reasoning=False, label=agent_name, language=language)
        if not response.tool_calls:
            return redact_injection_flag(response.content, agent_name=agent_name)

        messages.append(
            {
                "role": "assistant",
                "content": response.content,
                "tool_calls": [
                    {
                        "id": call["id"],
                        "type": "function",
                        "function": {"name": call["name"], "arguments": json.dumps(call["arguments"])},
                    }
                    for call in response.tool_calls
                ],
            }
        )
        # Executed concurrently, not one-at-a-time — a single model turn
        # commonly requests several independent tool calls (e.g. price data
        # + indicators + a verified snapshot), and awaiting them in a plain
        # for-loop serialized their I/O for no reason (none of these calls
        # depend on each other's result). `asyncio.gather` preserves the
        # request order in its results, so messages are still appended in
        # the same order the model asked for them.
        # 한 턴에 여러 도구 호출을 병렬 실행 — 서로 독립적이라 순차 실행할 이유 없음.
        #
        # Deliberately NOT `return_exceptions=True`: `_call_one` is now total
        # (a spurious CancelledError degrades internally, see its own
        # comment) — the only thing that can still propagate out of it is a
        # GENUINE outer cancellation, which must abort this loop, not get
        # zipped into a `{"role": "tool", ...}` message as if it were content.
        results = await asyncio.gather(*(_call_one(call, dispatch, trusted_tools) for call in response.tool_calls))
        for call, wrapped_result in zip(response.tool_calls, results):
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": wrapped_result})

    messages.append({"role": "user", "content": _FORCE_FINAL_PROMPT})
    final = await llm_client.complete(messages, reasoning=False, label=agent_name, language=language)
    return redact_injection_flag(final.content, agent_name=agent_name)


async def _call_one(call: dict[str, Any], dispatch: dict[str, ToolFn], trusted_tools: frozenset[str]) -> str:
    """One tool call -> its message content, error-normalized to a plain
    string (never raised except for a genuine outer cancellation, which must
    not be swallowed) so a batch of these can run under `asyncio.gather`
    without one failure cancelling its siblings."""
    fn = dispatch.get(call["name"])
    if fn is None:
        # Internal error, not fetched content — never wrapped.
        return f"Error: unknown tool {call['name']!r}"
    try:
        result = await fn(**call["arguments"])
    except asyncio.CancelledError:
        # CancelledError subclasses BaseException, not Exception, so the
        # handler below never saw it — despite this function's own docstring
        # promising it never raises. It propagated out of `run_tool_loop`'s
        # `asyncio.gather` (aborting the sibling tool calls of the same model
        # turn mid-flight), escaped `analyze()` un-normalized, and landed in
        # `pipeline.py`'s analyst-result classifier, whose
        # `isinstance(result, Exception)` test returned False for it — so the
        # exception OBJECT was passed to `enforce_identity` as if it were
        # report text, raising `TypeError: normalize() argument 2 must be
        # str, not CancelledError`. That discarded the entire decision along
        # with the other three analysts' already-paid-for LLM work, and left
        # no trace: degraded_analysts stayed empty and analysts_failed
        # recorded 0.
        #
        # A real source of spurious cancellations reachable from exactly
        # these tools was fixed upstream in `0235444` (shielding the shared
        # candle-fetch cache, which the price/ATR tools go through); this is
        # the consumer-side hole the same class of failure came through, and
        # the analyst-gather twin of the context-gather guard added in
        # `9896bce` (id=377/id=949).
        #
        # `task_was_actually_cancelled()` makes the same distinction the
        # orchestrator already makes for the same reason: gather wraps each
        # of these coroutines in its own child Task, and a genuine outer
        # cancellation propagates a real cancellation request down to that
        # child — so cancelling() > 0 here means shutdown/teardown, which is
        # not ours to swallow. Anything else is a CancelledError that merely
        # surfaced through an unrelated internal wait, and degrades to an
        # error string exactly like any other tool failure.
        if task_was_actually_cancelled():
            raise
        logger.warning(
            "tool %s hit an unrequested CancelledError — degrading to an error "
            "string rather than aborting this analyst and its sibling tool "
            "calls (see id=377/id=949, 0235444)",
            call["name"],
        )
        return f"Error calling {call['name']}: unrequested cancellation"
    except Exception as exc:  # noqa: BLE001 - surface to the model, don't crash the analyst
        # Also an internal error, not fetched content — never wrapped,
        # unlike a genuine successful tool result below.
        return f"Error calling {call['name']}: {exc}"
    return result if call["name"] in trusted_tools else wrap_untrusted(call["name"], result)
