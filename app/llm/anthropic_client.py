"""Native Anthropic (Claude) Messages API adapter — the "anthropic"
`llm_provider` (app.config), added 2026-09-14 as 0단계 of the Claude-backend
measurement plan (see project memory: the plan is to compare hallucination-
guard violation rates between this desk's current Qwen/vLLM setup and a
Claude-driven one). This batch is PLUMBING ONLY — nothing routes real
traffic here yet; the default `llm_provider` stays "vllm" and every existing
deployment is unaffected until someone deliberately flips it.

Anthropic's Messages API is NOT OpenAI-compatible — different request shape
(a top-level `system` param instead of a `role: "system"` message, `thinking`/
`output_config` instead of `chat_template_kwargs`/`response_format`, tools
shaped as `{"name", "description", "input_schema"}` instead of
`{"type": "function", "function": {...}}`, content returned as a list of
typed blocks instead of a single `message.content` string) and a different
response shape (`stop_reason` instead of `finish_reason`, a `usage` object
with different field names, a native `stop_reason == "refusal"` value with
no OpenAI equivalent). This module owns that entire translation so every
other caller keeps working against the same app.domain.ports.llm_port.LLMPort
Protocol (`complete()` / `structured_complete()` -> LLMMessage) regardless of
which backend is selected.

`complete()` also translates app.agents.tool_loop's OpenAI-shaped multi-turn
tool-calling history (`role: "tool"` messages, an assistant message carrying
a `tool_calls` list with JSON-string `arguments`) into Anthropic's expected
`tool_use`/`tool_result` content-block turns — see `_to_anthropic_messages`.
The one non-obvious rule baked into that translation: Anthropic requires ALL
`tool_result` blocks answering one assistant turn's `tool_use` blocks to be
combined into a SINGLE `role: "user"` message, not emitted as separate
messages the way `tool_loop.py` appends them one per tool call — so
`_to_anthropic_messages` buffers consecutive `role: "tool"` entries and
flushes them as one merged message per run. It also deliberately calls
`json.loads()` on each tool call's `arguments` (a JSON string, since it
round-tripped through `tool_loop.py`'s own `json.dumps()`), which is the
mirror image of `_tool_calls_from_content` below — that method must NEVER
`json.loads()` a live response's `tool_use.input`, since Anthropic already
hands that back as a parsed dict. These two directions are asymmetric on
purpose, not an inconsistency.

`complete()` separately accommodates `tool_loop.py`'s force-final call
(after the iteration cap, with no `tools=` argument but a message history
that still carries earlier tool_use/tool_result turns): Anthropic rejects a
request whose history contains tool blocks when `tools` is empty/omitted, so
`complete()` synthesizes minimal, never-actually-callable stub tool schemas
for every tool name found in the translated history and pairs them with
`tool_choice={"type": "none"}`, structurally blocking any call while still
satisfying the API's history-validation requirement.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, TypeVar

from anthropic import (
    APIConnectionError,
    APIStatusError,
    AsyncAnthropic,
    AuthenticationError,
    PermissionDeniedError,
    RateLimitError,
)
from pydantic import BaseModel

from app.config import is_backtest_context, settings
from app.domain.models.llm import LLMMessage
from app.llm import metrics as llm_metrics
from app.llm.client import _FALLBACK_MAX_ATTEMPTS, _FALLBACK_RETRY_DELAY_SECONDS
from app.llm.language import DEFAULT_LANGUAGE, language_instruction
from app.llm.shared import (
    PriorityLimiter,
    _clean_json_payload,
    _contains_hanja,
    _looks_degenerate,
    _strip_reasoning_leak,
)

__all__ = ["AnthropicLLMClient"]

T = TypeVar("T", bound=BaseModel)

logger = logging.getLogger(__name__)

# OpenAI-compatible finish_reason value this project's downstream code
# already keys off (see app.llm.client._TRUNCATED_FINISH_REASON /
# LLMCallMetric.finish_reason) — Claude's own `stop_reason` for the same
# condition is "max_tokens"; _record_metrics below maps one to the other so
# every existing "finish_reason == 'length'" check keeps working unchanged
# regardless of which backend produced the record.
_TRUNCATED_FINISH_REASON = "length"

# Errors that mean "you are out of budget / not authorized", not "transient
# glitch" — same doctrine as app.llm.client._is_quota_exhausted (there,
# APIStatusError.status_code in {402, 429}; here, the SDK already gives us
# dedicated exception classes for the equivalent conditions). structured_
# complete must NOT route these into the unconstrained-completion fallback —
# firing a second, differently-shaped request at an endpoint that just
# rejected us for quota/auth reasons only wastes time and still fails.
_QUOTA_EXCEPTIONS: tuple[type[BaseException], ...] = (RateLimitError, AuthenticationError, PermissionDeniedError)


def _is_quota_exhausted(exc: BaseException) -> bool:
    return isinstance(exc, _QUOTA_EXCEPTIONS)


class _AnthropicRefusal(ValueError):
    """Raised when `message.stop_reason == "refusal"` — a genuine policy
    refusal (Claude's streaming classifiers intervened), not a broken
    response-format/guided-decoding failure. structured_complete must not
    treat this as "try the unconstrained fallback" (a differently-shaped
    request routes around a broken JSON mode, it does not route around a
    safety refusal), and complete() must not pass through whatever partial
    body came back as if it were a normal answer. Never swallowed anywhere
    in this module — it always propagates to the caller."""


class AnthropicLLMClient:
    """`app.domain.ports.llm_port.LLMPort`-satisfying adapter over the native
    `anthropic` SDK (`AsyncAnthropic`). See module docstring for the overall
    shape-translation responsibilities and the known tool_loop scope gap."""

    def __init__(self) -> None:
        self._client = AsyncAnthropic(
            api_key=settings.anthropic_api_key,
            timeout=settings.anthropic_timeout_seconds,
            max_retries=settings.anthropic_max_retries,
        )
        self._limiter = PriorityLimiter(settings.llm_max_concurrent_requests)
        # One-time DEBUG note (not a WARNING — every existing call site
        # passes the default 0.0 anyway, so this is expected, not a sign of
        # misconfiguration) that a non-default `temperature` was ignored —
        # see complete()/structured_complete()'s own comment for why Claude
        # never receives it at all.
        self._warned_temperature = False

    @staticmethod
    def _current_priority() -> str:
        return "low" if is_backtest_context() else "high"

    @staticmethod
    def _thinking_and_effort(reasoning: bool) -> tuple[dict[str, Any], str]:
        """`thinking` stays `{"type": "adaptive"}` regardless of `reasoning`
        — only the `output_config.effort` level changes
        (settings.anthropic_effort / anthropic_effort_low). `thinking:
        {"type": "disabled"}` is deliberately never used on Opus 5: it has
        two known failure modes — it can emit a tool call as visible text
        instead of a proper `tool_use` block, and it can leak
        `<thinking>`-style tags into the response body — so "no reasoning"
        is expressed as adaptive-thinking-at-low-effort instead of disabling
        thinking outright. See app.config.Settings.anthropic_effort."""
        thinking = {"type": "adaptive", "display": settings.anthropic_thinking_display}
        effort = settings.anthropic_effort if reasoning else settings.anthropic_effort_low
        return thinking, effort

    @staticmethod
    def _split_system_and_messages(
        messages: list[dict[str, Any]], language: str
    ) -> tuple[str, list[dict[str, Any]]]:
        """Anthropic's Messages API takes `system` as a separate top-level
        param, not a `role: "system"` message — this pulls every system
        entry out of `messages`, joins them (defensively; callers on this
        desk send exactly one), and appends `language_instruction(language)`
        with the same "\\n\\n" separator style
        `LLMClient._with_language_instruction` uses today. Returns
        (system_text, remaining_messages); remaining_messages excludes every
        system entry.

        Hard invariant, not expected to actually fire (every current call
        site already satisfies it): the first remaining message must have
        role "user" — never silently pass a malformed request to the API."""
        system_parts = [m["content"] for m in messages if m.get("role") == "system"]
        remaining = [m for m in messages if m.get("role") != "system"]
        instruction = language_instruction(language)
        if system_parts:
            system_text = "\n\n".join([*system_parts, instruction])
        else:
            system_text = instruction
        if not remaining or remaining[0].get("role") != "user":
            raise ValueError(
                "AnthropicLLMClient: expected the first non-system message to have "
                f"role='user' after removing system messages, got {remaining[0].get('role')!r}"
                if remaining
                else "AnthropicLLMClient: no messages remained after removing system messages"
            )
        return system_text, remaining

    @staticmethod
    def _system_param(system_text: str) -> str | list[dict[str, Any]]:
        """Bare string, or (if settings.anthropic_prompt_cache) a
        single-block content list with one `cache_control: {"type":
        "ephemeral"}` breakpoint at the end of the system block — every
        call's system text is the same per-agent prompt plus the language
        instruction, so caching it cuts repeat-call cost/latency with no
        change to the response itself."""
        if not settings.anthropic_prompt_cache:
            return system_text
        return [{"type": "text", "text": system_text, "cache_control": {"type": "ephemeral"}}]

    @staticmethod
    def _convert_tools(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
        """OpenAI shape (`{"type": "function", "function": {"name",
        "description", "parameters"}}`, see app.agents.tool_loop) -> Anthropic
        shape (`{"name", "description", "input_schema"}`)."""
        if not tools:
            return None
        converted = []
        for tool in tools:
            fn = tool["function"]
            converted.append(
                {
                    "name": fn["name"],
                    "description": fn.get("description", ""),
                    "input_schema": fn["parameters"],
                }
            )
        return converted

    @staticmethod
    def _to_anthropic_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """OpenAI-shaped tool-calling history (app.agents.tool_loop) ->
        Anthropic content-block turns. Input is the `remaining` list from
        _split_system_and_messages (system entries already removed).

        Single pass over `messages`:
          - plain `{"role": "user", "content": str}` passes through unchanged.
          - plain `{"role": "assistant", "content": str}` with no (or empty/
            falsy) `tool_calls` passes through unchanged too — this shape
            also occurs outside tool_loop (e.g. portfolio_manager.py's
            retry-with-plain-assistant-turn feeding structured_complete), so
            it must round-trip byte-identical rather than being rejected.
          - `{"role": "assistant", "content": str, "tool_calls": [...]}`
            becomes ONE assistant message whose content is a list: an
            optional leading text block (only if `content` is truthy — never
            a spurious empty one) followed by one `tool_use` block per
            tool_calls entry, in order.
          - a RUN of consecutive `{"role": "tool", ...}` entries is buffered
            and flushed as exactly ONE `{"role": "user", "content": [...]}`
            message of `tool_result` blocks (one run -> one message,
            regardless of how many tool entries it contains) the moment a
            non-tool message is hit, or at the end of the list.
        """
        out: list[dict[str, Any]] = []
        pending_results: list[dict[str, Any]] = []
        # Whether the last message appended to `out` (not the pending
        # buffer) was an assistant turn carrying tool_use blocks — used only
        # to detect an orphaned tool message (a "role": "tool" entry with no
        # preceding assistant-tool_calls turn at all).
        last_appended_is_tool_use_turn = False

        def _flush() -> None:
            nonlocal pending_results
            if pending_results:
                out.append({"role": "user", "content": pending_results})
                pending_results = []

        for idx, msg in enumerate(messages):
            if msg.get("role") == "tool":
                if not pending_results and not last_appended_is_tool_use_turn:
                    raise ValueError(
                        "AnthropicLLMClient: orphaned tool message at index "
                        f"{idx} (tool_call_id={msg.get('tool_call_id')!r}) has no "
                        "preceding assistant tool_use turn"
                    )
                try:
                    tool_call_id = msg["tool_call_id"]
                except KeyError as exc:
                    raise ValueError(
                        f"AnthropicLLMClient: message at index {idx} (role='tool') "
                        "is missing 'tool_call_id'"
                    ) from exc
                pending_results.append(
                    {"type": "tool_result", "tool_use_id": tool_call_id, "content": msg.get("content")}
                )
                continue

            # Non-tool message: flush any buffered tool-result run first,
            # then process this message.
            _flush()

            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                content_blocks: list[dict[str, Any]] = []
                if msg.get("content"):
                    content_blocks.append({"type": "text", "text": msg["content"]})
                for call_idx, call in enumerate(msg["tool_calls"]):
                    try:
                        call_id = call["id"]
                        function = call["function"]
                        name = function["name"]
                        args_str = function["arguments"]
                    except KeyError as exc:
                        raise ValueError(
                            "AnthropicLLMClient: tool_calls entry at message index "
                            f"{idx}, call index {call_idx} is missing key {exc}"
                        ) from exc
                    try:
                        arguments = json.loads(args_str)
                    except json.JSONDecodeError as exc:
                        raise ValueError(
                            f"AnthropicLLMClient: malformed tool_calls arguments for "
                            f"tool={name!r} id={call_id!r}: {args_str[:120]!r}"
                        ) from exc
                    content_blocks.append(
                        {"type": "tool_use", "id": call_id, "name": name, "input": arguments}
                    )
                out.append({"role": "assistant", "content": content_blocks})
                last_appended_is_tool_use_turn = True
            else:
                out.append(msg)
                last_appended_is_tool_use_turn = False

        _flush()
        return out

    @staticmethod
    def _tool_use_names_in_history(messages: list[dict[str, Any]]) -> set[str]:
        """Distinct tool names found in any `tool_use` content block across
        an already-translated (Anthropic-shaped) message list — used by
        complete() to synthesize stub tool schemas for tool_loop's
        force-final call (see module docstring)."""
        names: set[str] = set()
        for msg in messages:
            content = msg.get("content")
            if msg.get("role") == "assistant" and isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_use":
                        names.add(block["name"])
        return names

    @staticmethod
    def _tool_calls_from_content(content: list[Any]) -> list[dict[str, Any]]:
        """Anthropic `tool_use` content blocks -> the same
        `[{"id", "name", "arguments"}]` shape LLMClient.complete() already
        produces for the vLLM/gateway path. `block.input` is ALREADY a
        parsed dict (see anthropic.types.tool_use_block.ToolUseBlock) — never
        `json.loads()` it, that would fail since it is not a JSON string."""
        return [
            {"id": block.id, "name": block.name, "arguments": block.input}
            for block in content
            if getattr(block, "type", None) == "tool_use"
        ]

    @staticmethod
    def _text_from_content(content: list[Any]) -> str:
        return "".join(block.text for block in content if getattr(block, "type", None) == "text")

    @staticmethod
    def _reasoning_from_content(content: list[Any]) -> str | None:
        """Claude's chain-of-thought comes back as separate `thinking`
        content blocks (never folded into `content`/text blocks) — captured
        here into LLMMessage.reasoning_content, mirroring how LLMClient
        surfaces vLLM's `reasoning_content`/`reasoning` fields."""
        parts = [block.thinking for block in content if getattr(block, "type", None) == "thinking"]
        return "\n\n".join(parts) if parts else None

    async def _send(
        self,
        *,
        system: str | list[dict[str, Any]],
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        thinking: dict[str, Any],
        output_config: dict[str, Any],
        max_tokens: int,
        tool_choice: dict[str, Any] | None = None,
    ) -> Any:
        """One streamed Messages API call -> the assembled final `Message`.
        Always streams (`client.messages.stream(...).get_final_message()`),
        for EVERY call, not just long ones — avoids any HTTP timeout risk
        from combining high effort + adaptive thinking + a large max_tokens
        ceiling; individual stream events are never processed, only the
        assembled final message.

        `tool_choice` only matters when `tools` is truthy: omit it (the
        default) for today's unchanged `{"type": "auto"}` behavior at every
        existing call site; pass an explicit value (e.g. `{"type": "none"}`,
        used by complete()'s force-final stub-tools accommodation — see
        module docstring) to override it for that one call.

        Hard-fails on `stop_reason == "refusal"` (see _AnthropicRefusal) —
        never passes through a refusal's partial/empty body as a normal
        answer."""
        kwargs: dict[str, Any] = {
            "model": settings.anthropic_model,
            "system": system,
            "messages": messages,
            "max_tokens": max_tokens,
            "thinking": thinking,
            "output_config": output_config,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice if tool_choice is not None else {"type": "auto"}
        async with self._client.messages.stream(**kwargs) as stream:
            message = await stream.get_final_message()
        if message.stop_reason == "refusal":
            details = message.stop_details
            raise _AnthropicRefusal(
                f"anthropic message refused (category={getattr(details, 'category', None)!r}, "
                f"explanation={getattr(details, 'explanation', None)!r})"
            )
        return message

    def _record_metrics(
        self,
        message: Any,
        *,
        label: str,
        reasoning: bool,
        structured: bool,
        start: float,
        fallback_used: bool = False,
        max_tokens: int | None = None,
    ) -> None:
        """Same field mapping/semantics as LLMClient._record_metrics (see
        app.llm.client), populated from Anthropic's response shape instead of
        an OpenAI-compatible one: `usage.input_tokens` plus any cached-token
        counts make up `prompt_tokens` (Claude bills/reports those
        separately, unlike the OpenAI-compatible `usage.prompt_tokens`
        total), and Claude's `stop_reason` maps to `finish_reason` — "max_
        tokens" becomes "length" (see _TRUNCATED_FINISH_REASON) so the
        existing truncation-warning branch below, and every downstream
        reader of LLMCallMetric.finish_reason, fires identically regardless
        of backend; every other stop_reason ("end_turn", "tool_use",
        "refusal", "pause_turn", ...) passes through unchanged."""
        effective_max_tokens = max_tokens if max_tokens is not None else settings.llm_max_output_tokens
        usage = message.usage
        prompt_tokens = (
            usage.input_tokens
            + (getattr(usage, "cache_read_input_tokens", None) or 0)
            + (getattr(usage, "cache_creation_input_tokens", None) or 0)
        )
        completion_tokens = usage.output_tokens
        total_tokens = prompt_tokens + completion_tokens
        stop_reason = message.stop_reason
        finish_reason = _TRUNCATED_FINISH_REASON if stop_reason == "max_tokens" else stop_reason
        llm_metrics.record(
            label=label,
            model=settings.anthropic_model,
            reasoning=reasoning,
            structured=structured,
            fallback_used=fallback_used,
            latency_seconds=time.monotonic() - start,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            finish_reason=finish_reason,
            max_tokens=effective_max_tokens,
        )
        if finish_reason == _TRUNCATED_FINISH_REASON:
            logger.warning(
                "llm call (anthropic) label=%s hit the max_tokens ceiling (stop_reason=%s, max_tokens=%s, "
                "structured=%s, fallback=%s) — the response is TRUNCATED",
                label, stop_reason, effective_max_tokens, structured, fallback_used,
            )

    def _maybe_warn_temperature(self, temperature: float, *, label: str) -> None:
        # Claude Opus 5 rejects `temperature` with a 400 in some
        # configurations combined with thinking, and the vLLM/gateway
        # "different temperature" retry trick is meaningless here anyway
        # (Claude isn't greedy-decoded the same way) — so `temperature` is
        # accepted in every public signature for interface compatibility
        # with LLMPort, but never sent in the request at all.
        if temperature != 0.0 and not self._warned_temperature:
            logger.debug(
                "AnthropicLLMClient label=%s: temperature=%s ignored — this backend never sends "
                "temperature to the Messages API (see complete()'s docstring)",
                label, temperature,
            )
            self._warned_temperature = True

    async def _retry_once_if_hanja(self, result: T, raw: str, *, label: str, retry_call: Any) -> T:
        """Same soft-retry doctrine as LLMClient._retry_once_if_hanja: one
        reissue of the IDENTICAL request (Claude's sampling isn't purely
        greedy, so a reissue can produce a different result) if the retry
        itself fails, keep the already-valid original rather than losing it."""
        if not _contains_hanja(raw):
            return result
        logger.warning(
            "structured_complete (anthropic) label=%s: Chinese-character (hanja) contamination "
            "detected — retrying once (same request)",
            label,
        )
        try:
            retried_result, _ = await retry_call()
            return retried_result
        except Exception as exc:  # noqa: BLE001 - keep the already-valid original rather than fail the call
            logger.warning(
                "structured_complete (anthropic) label=%s: hanja retry attempt itself failed (%s) — "
                "keeping original result",
                label, exc,
            )
            return result

    async def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        reasoning: bool = False,
        temperature: float = 0.0,
        label: str = "",
        language: str = DEFAULT_LANGUAGE,
        max_tokens: int | None = None,
    ) -> LLMMessage:
        self._maybe_warn_temperature(temperature, label=label)
        system_text, remaining = self._split_system_and_messages(messages, language)
        remaining = self._to_anthropic_messages(remaining)
        system_param = self._system_param(system_text)
        effective_max_tokens = max_tokens if max_tokens is not None else settings.llm_max_output_tokens
        thinking, effort = self._thinking_and_effort(reasoning)
        anthropic_tools = self._convert_tools(tools)

        # tool_loop.py's force-final call (after the iteration cap) passes no
        # `tools=` at all, but `remaining` still carries tool_use/tool_result
        # blocks from earlier loop iterations — Anthropic requires `tools` to
        # be present whenever the message history contains tool blocks, or
        # the request is rejected. When that happens, synthesize minimal stub
        # schemas (name-only, never actually callable) purely to satisfy
        # history validation, and force `tool_choice: none` so the model is
        # structurally prevented from calling anything this turn — matching
        # tool_loop's actual intent for this call ("give your final report
        # now, no more tool calls"). Computed once here (not inside _call())
        # so every retry attempt sees the identical request.
        tool_choice: dict[str, Any] | None = None
        if anthropic_tools is None:
            history_tool_names = self._tool_use_names_in_history(remaining)
            if history_tool_names:
                anthropic_tools = [
                    {"name": n, "description": "", "input_schema": {"type": "object"}}
                    for n in sorted(history_tool_names)
                ]
                tool_choice = {"type": "none"}

        async def _call() -> LLMMessage:
            start = time.monotonic()
            message = await self._send(
                system=system_param,
                messages=remaining,
                tools=anthropic_tools,
                thinking=thinking,
                output_config={"effort": effort},
                max_tokens=effective_max_tokens,
                tool_choice=tool_choice,
            )
            self._record_metrics(
                message, label=label, reasoning=reasoning, structured=False, start=start,
                max_tokens=effective_max_tokens,
            )
            tool_calls = self._tool_calls_from_content(message.content)
            raw_content = self._text_from_content(message.content)
            if tool_calls:
                sanitized_content, sanitized_kind = raw_content, None
            else:
                sanitized_content, sanitized_kind = _strip_reasoning_leak(raw_content, label=label)
            if sanitized_kind is not None:
                llm_metrics.note_sanitization(f"{label or 'unlabeled'}: {sanitized_kind}")
            return LLMMessage(
                content=sanitized_content,
                reasoning_content=self._reasoning_from_content(message.content),
                tool_calls=tool_calls,
            )

        # Same one-acquire-per-logical-call shape as LLMClient.complete —
        # see PriorityLimiter's own docstring.
        await self._limiter.acquire(self._current_priority())
        try:
            result = await _call()

            # Empty-response check: HARD fail, same doctrine as LLMClient —
            # not wrapped in try/except, so a genuine failure on retry
            # propagates rather than being swallowed.
            if not result.tool_calls and not result.content.strip():
                logger.warning(
                    "llm complete (anthropic): empty content with no tool calls (label=%s) — "
                    "retrying once (same request)",
                    label,
                )
                result = await _call()
                if not result.tool_calls and not result.content.strip():
                    raise ValueError(
                        f"llm complete (anthropic) label={label!r}: model returned empty content with "
                        "no tool calls twice in a row — treating as a genuine failure rather than "
                        "caching a blank result"
                    )
            elif not result.tool_calls and _looks_degenerate(result.content):
                # Soft retry — the retry attempt itself failing must not
                # turn an already-working response into a hard failure.
                logger.warning(
                    "llm complete (anthropic): degenerate repeated-paragraph output detected "
                    "(label=%s) — retrying once (same request)",
                    label,
                )
                try:
                    result = await _call()
                except Exception as exc:  # noqa: BLE001 - keep the already-valid original
                    logger.warning(
                        "llm complete (anthropic) label=%s: degenerate-output retry itself failed "
                        "(%s) — keeping original result",
                        label, exc,
                    )
            elif not result.tool_calls and language == "ko" and _contains_hanja(result.content):
                logger.warning(
                    "llm complete (anthropic): Chinese-character (hanja) contamination detected in "
                    "Korean output (label=%s) — retrying once (same request)",
                    label,
                )
                try:
                    result = await _call()
                except Exception as exc:  # noqa: BLE001 - keep the already-valid original
                    logger.warning(
                        "llm complete (anthropic) label=%s: hanja retry attempt itself failed (%s) — "
                        "keeping original result",
                        label, exc,
                    )
            return result
        finally:
            await self._limiter.release()

    async def structured_complete(
        self,
        messages: list[dict[str, Any]],
        schema: type[T],
        *,
        reasoning: bool = False,
        temperature: float = 0.0,
        label: str = "",
        language: str = DEFAULT_LANGUAGE,
    ) -> T:
        """Native `output_config.format` (json_schema of `schema`) first; if
        the server returns nothing usable, falls back to an unconstrained
        completion with the schema spelled out in the prompt instead — same
        two-tier doctrine as LLMClient.structured_complete (see its
        docstring for the original vLLM incidents this pattern guards
        against). Re-raises immediately (no fallback) on a quota/auth error
        or a genuine policy refusal — see _is_quota_exhausted /
        _AnthropicRefusal. Fails loud if the fallback ALSO doesn't validate."""
        self._maybe_warn_temperature(temperature, label=label)
        system_text, remaining = self._split_system_and_messages(messages, language)
        system_param = self._system_param(system_text)
        thinking, effort = self._thinking_and_effort(reasoning)
        max_tokens = settings.llm_max_output_tokens

        async def _native_call() -> tuple[T, str]:
            start = time.monotonic()
            message = await self._send(
                system=system_param,
                messages=remaining,
                tools=None,
                thinking=thinking,
                output_config={"effort": effort, "format": {"type": "json_schema", "schema": schema.model_json_schema()}},
                max_tokens=max_tokens,
            )
            self._record_metrics(
                message, label=label, reasoning=reasoning, structured=True, start=start, fallback_used=False,
                max_tokens=max_tokens,
            )
            raw = self._text_from_content(message.content)
            cleaned = _clean_json_payload(raw, label=label, path="native")
            return schema.model_validate_json(cleaned), raw

        await self._limiter.acquire(self._current_priority())
        try:
            try:
                result, raw = await _native_call()
            except _AnthropicRefusal:
                raise
            except _QUOTA_EXCEPTIONS as exc:
                logger.warning(
                    "structured_complete (anthropic) label=%s: native request hit a quota/auth error "
                    "(%s) — NOT falling back; re-raising",
                    label, exc,
                )
                raise
            except Exception as exc:  # noqa: BLE001 - deliberately broad: any parse/validation/request failure falls back
                logger.warning(
                    "structured_complete (anthropic) label=%s: native output_config request produced no "
                    "valid %s (%s) — falling back to unconstrained completion",
                    label, schema.__name__, exc,
                )
                return await self._structured_complete_fallback(
                    system_param, remaining, schema, thinking=thinking, effort=effort, max_tokens=max_tokens,
                    reasoning=reasoning, label=label, native_error=exc,
                )
            return await self._retry_once_if_hanja(result, raw, label=label, retry_call=_native_call)
        finally:
            await self._limiter.release()

    async def _structured_complete_fallback(
        self,
        system_param: str | list[dict[str, Any]],
        remaining: list[dict[str, Any]],
        schema: type[T],
        *,
        thinking: dict[str, Any],
        effort: str,
        max_tokens: int,
        reasoning: bool,
        label: str,
        native_error: Exception,
    ) -> T:
        """No `output_config.format` constraint at all — the schema is
        spelled out as plain-text instructions instead, same fallback shape
        as LLMClient._structured_complete_fallback. Reuses that module's own
        `_FALLBACK_MAX_ATTEMPTS`/`_FALLBACK_RETRY_DELAY_SECONDS` constants
        rather than redefining them."""
        schema_instruction = {
            "role": "user",
            "content": (
                f"Your previous response could not be parsed as valid JSON matching the required "
                f"schema. Respond with ONLY a single JSON object matching this schema — no markdown, "
                f"no commentary, no extra text before or after it:\n{json.dumps(schema.model_json_schema())}"
            ),
        }
        fallback_messages = [*remaining, schema_instruction]
        last_error: Exception = native_error

        async def _fallback_call() -> tuple[T, str]:
            start = time.monotonic()
            message = await self._send(
                system=system_param,
                messages=fallback_messages,
                tools=None,
                thinking=thinking,
                output_config={"effort": effort},
                max_tokens=max_tokens,
            )
            self._record_metrics(
                message, label=label, reasoning=reasoning, structured=True, start=start, fallback_used=True,
                max_tokens=max_tokens,
            )
            raw = self._text_from_content(message.content)
            cleaned = _clean_json_payload(raw, label=label, path="fallback")
            return schema.model_validate_json(cleaned), raw

        for attempt in range(1, _FALLBACK_MAX_ATTEMPTS + 1):
            try:
                result, raw = await _fallback_call()
            except _AnthropicRefusal:
                raise
            except _QUOTA_EXCEPTIONS:
                raise
            except (APIStatusError, APIConnectionError) as exc:
                last_error = exc
                logger.warning(
                    "structured_complete (anthropic) label=%s: fallback request itself failed on "
                    "attempt %d/%d (%s)",
                    label, attempt, _FALLBACK_MAX_ATTEMPTS, exc,
                )
                if attempt < _FALLBACK_MAX_ATTEMPTS:
                    await asyncio.sleep(_FALLBACK_RETRY_DELAY_SECONDS)
                continue
            except Exception as fallback_error:  # noqa: BLE001 - retry loop needs the broad catch too
                last_error = fallback_error
                logger.warning(
                    "structured_complete (anthropic) label=%s: fallback produced no valid %s on "
                    "attempt %d/%d (%s)",
                    label, schema.__name__, attempt, _FALLBACK_MAX_ATTEMPTS, fallback_error,
                )
                if attempt < _FALLBACK_MAX_ATTEMPTS:
                    await asyncio.sleep(_FALLBACK_RETRY_DELAY_SECONDS)
                continue

            return await self._retry_once_if_hanja(result, raw, label=label, retry_call=_fallback_call)

        raise ValueError(
            f"structured_complete (anthropic) label={label!r}: both native output_config and the "
            f"unconstrained fallback (after {_FALLBACK_MAX_ATTEMPTS} attempts) failed to produce a "
            f"valid {schema.__name__} — native error: {native_error!r}, last fallback error: {last_error!r}"
        ) from last_error
