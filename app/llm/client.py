"""Thin async wrapper around one OpenAI-compatible chat-completions endpoint.

Two interchangeable backends, selected by `settings.llm_provider`:

  * "vllm" (default) — the self-hosted endpoint this project was built
    against (`settings.llm_base_url`, no API key). Reasoning is toggled per
    call via Qwen3's `enable_thinking` chat-template kwarg instead of
    switching models, per the architecture spec's "toggle reasoning per
    role instead of deploying a second model" decision.
  * "gateway" — an OpenAI-compatible proxy that fans requests out to
    multiple upstream providers with its own fallback/load-balancing
    (`settings.llm_gateway_base_url` + `settings.llm_api_key`; reference
    target OmniRoute, run locally, its dashboard picks the upstreams and
    routing — `settings.llm_model` carries the gateway's model/combo/
    strategy name). A gateway does not understand `enable_thinking`, and
    the vLLM-only sampling extra `repetition_penalty` is dropped (not every
    upstream accepts it); reasoning is requested through the OpenAI-standard
    `reasoning_effort` field, which the gateway translates per upstream.
    See project memory: openrouter-integration-idea-2026-08-27.

`complete()` / `structured_complete()` themselves are provider-agnostic —
the only per-vendor difference lives in `_reasoning_extra_body()` and the
endpoint/credentials resolved in `__init__`. The vendor-agnostic response
post-processing helpers (JSON cleanup, degenerate/hanja/reasoning-leak
detection, the priority concurrency limiter) now live in app.llm.shared.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Awaitable, Callable, TypeVar

from openai import APIConnectionError, APIStatusError, AsyncOpenAI
from pydantic import BaseModel

from app.config import is_backtest_context, settings
from app.domain.models.llm import LLMMessage
from app.llm import metrics as llm_metrics
from app.llm.request_capture import observe_first_native_parse, observe_first_native_request, observe_first_native_result
from app.llm.leak_markers import UNTAGGED_REASONING_MARKERS as _UNTAGGED_REASONING_MARKERS
from app.llm.leak_markers import UNTAGGED_REASONING_MIN_MARKERS as _UNTAGGED_REASONING_MIN_MARKERS
from app.llm.language import DEFAULT_LANGUAGE, language_instruction
from app.llm.shared import (
    _ANY_CLOSE_RE,
    _ANY_OPEN_RE,
    _DEGENERATE_MIN_PARAGRAPH_LEN,
    _DEGENERATE_MIN_REPEATS,
    _HANJA_RE,
    _LOW_STARVATION_LIMIT,
    _MIN_SANITIZED_BODY_CHARS,
    _OPEN_TO_END_RE,
    _PRIORITY_RANK,
    _PriorityWaiter,
    _THINK_BLOCK_RE,
    _THROUGH_LAST_CLOSE_RE,
    _clean_json_payload,
    _contains_hanja,
    _extract_json,
    _looks_degenerate,
    _strip_reasoning_leak,
    PriorityLimiter,
)

# Re-exported for backward compatibility — every existing call site imports
# this as `from app.llm.client import LLMMessage`; the dataclass itself now
# lives in app.domain.models.llm so app.domain.ports.llm_port can reference
# it without importing this concrete adapter module (same split as
# BrokerBalance et al. living in app.domain.models.account, not app.brokers).
__all__ = ["LLMMessage", "LLMClient", "llm_client"]

T = TypeVar("T", bound=BaseModel)

logger = logging.getLogger(__name__)


# Extra attempts for structured_complete's fallback call specifically (on top
# of AsyncOpenAI's own max_retries=5, which already exhausted its retries
# against the identical native request by the time we get here) — a real
# 2026-07-28 incident hit an intermittent HTTP 500 from a gpt-oss-20b vLLM
# endpoint that persisted through all 5 SDK-level retries of the NATIVE
# (response_format-constrained) request, then succeeded once retried against
# the fallback's differently-shaped (no response_format) request. Kept small
# and separate from max_retries: retrying the exact same failing request
# shape more times wastes time on a failure mode the fallback's different
# shape is specifically meant to route around.
# 폴백 호출 전용 추가 재시도 — SDK의 max_retries=5는 이미 네이티브 요청에서 소진됨.
_FALLBACK_MAX_ATTEMPTS = 2
_FALLBACK_RETRY_DELAY_SECONDS = 2.0

# 2026-07-30: greedy decoding (temperature=0.0, this project's default
# everywhere) has no built-in mechanism to escape a locally-repeating pattern
# once one starts — confirmed live (fundamentals_analyst, 042660.KS) as a
# ~4000-token response that was the same one-sentence paragraph copy-pasted
# ~100 times, with no financial content at all. `repetition_penalty` (a vLLM
# sampling extension, same extra_body mechanism as chat_template_kwargs
# below) directly discourages the token-repeat pattern that causes this;
# 1.15 is a commonly-cited value that curbs repetition without meaningfully
# distorting normal output (much higher starts degrading coherence).
# Belt-and-suspenders: _looks_degenerate()/_retry_temperature below catch it
# even if repetition_penalty doesn't fully prevent it (see complete()).
_REPETITION_PENALTY = 1.15

# 2026-08-31: measured against a real Research-Manager-sized prompt (1574
# prompt tokens, max_tokens=16000) on the new reasoning model
# (Qwen3.8-Flash-Next-NVFP4, served via SGLang) — without this, a
# reasoning=True call burns the ENTIRE max_tokens budget on the hidden
# reasoning_content chain-of-thought and emits 0 body characters
# (388.8s, 16000/16000 completion tokens all reasoning, finish_reason=length).
# That empty body then fails FlatResearchPlan validation in
# structured_complete ("strategic_actions ... must not be empty"), falls back
# to unconstrained completion, and the fallback — same settings — repeats the
# same 0-body loop. Adding reasoning_effort="low" cut this to 60.2s with a
# real 6095-char body (finish_reason=stop). Also measured safe against the
# still-deployed vLLM model (Qwen3.6-35B-A3B-NVFP4): it accepts and silently
# ignores this field (HTTP 200, no error; output essentially unchanged —
# 41.2s/7180 chars vs 39.1s/7266 chars), so it is sent unconditionally
# whenever reasoning=True, no per-model branch needed.
# 리즈닝 모델(Qwen3.8-Flash-Next, SGLang)이 max_tokens 전부를 사고에만 쓰고 본문
# 0자로 잘리는 것을 실측 확인 — "low" 추가로 388.8초/0자 → 60.2초/6095자.
# 기존 vLLM 모델(Qwen3.6)은 이 값을 에러 없이 무시함이 실측 확인돼 조건 분기 없이 전송.
_REASONING_EFFORT = "low"
_DEGENERATE_RETRY_TEMPERATURE = 0.4

# OpenAI-compatible finish_reason value meaning "hit max_tokens, cut off
# mid-generation" — see LLMCallMetric.finish_reason / _record_metrics.
_TRUNCATED_FINISH_REASON = "length"

# HTTP statuses that mean "you are out of budget", not "transient glitch":
# 429 = rate/quota limit hit, 402 = payment required / credits exhausted
# (both common on free upstream tiers reached through a gateway).
# structured_complete must NOT route these into its unconstrained-completion
# fallback — the fallback is for a backend whose guided decoding is broken,
# and firing 2 more requests at an endpoint that just rate-limited us only
# burns the limit faster and still fails. Let them propagate so the caller
# sees the real cause.
# 예산 소진성 오류(429/402)는 폴백 대상이 아님 — 재시도가 한도만 더 태움.
_QUOTA_EXHAUSTED_STATUSES = frozenset({402, 429})


def _is_quota_exhausted(exc: BaseException) -> bool:
    return isinstance(exc, APIStatusError) and exc.status_code in _QUOTA_EXHAUSTED_STATUSES


# Every LLM call (chat answers, analyst reports, debate, final decision prose,
# AND structured-output calls — structured_complete previously skipped this
# entirely, which is why research/trader/portfolio-manager/sentiment output
# could come back in English regardless of what language the user asked in)
# goes through _with_language_instruction. `language` ("ko"/"en") is threaded
# down from the user's own question (app.llm.language.detect_language) through
# the whole pipeline, not hardcoded — see agent-architecture-roadmap memory.


class LLMClient:
    def __init__(self) -> None:
        base_url, api_key, max_retries = self._resolve_endpoint()
        # max_retries bumped from the SDK default (2) for "vllm" — a long
        # backtest makes enough calls that a single transient read-timeout to
        # a remote endpoint becomes likely over a multi-hour run, and losing
        # the whole run (with no checkpointing) to one blip is expensive.
        # Retries only help with transient failures, not a genuinely stuck
        # server — timeout is what bounds a single request that's genuinely
        # stuck (the SDK's own default is 10 minutes, which combined with 5
        # retries could block one deep_dive for 50+ minutes with no timeout
        # set at all). "gateway" uses fewer retries — see _resolve_endpoint.
        self._provider = settings.llm_provider
        self._client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            max_retries=max_retries,
            timeout=settings.llm_timeout_seconds,
        )
        self._limiter = PriorityLimiter(settings.llm_max_concurrent_requests)

    @staticmethod
    def _resolve_endpoint() -> tuple[str, str, int]:
        """(base_url, api_key, max_retries) for the configured provider.

        "gateway" caps retries at 2 (vs 5): the gateway does its own
        millisecond-scale upstream failover, and when an upstream free tier
        returns 429 the SDK retrying it 5 times with back-off just delays
        the inevitable failure while spending more of the same limit. A
        genuine transient 5xx still gets two shots."""
        if settings.llm_provider == "gateway":
            # The boot-time model_validator already guaranteed llm_api_key is
            # set for this provider — the `or` is only a belt-and-suspenders
            # against a hand-built LLMClient in a test bypassing Settings.
            return settings.llm_gateway_base_url, settings.llm_api_key or "not-needed", 2
        return settings.llm_base_url, "not-needed", 5

    @staticmethod
    def _current_priority() -> str:
        return "low" if is_backtest_context() else "high"

    def _reasoning_extra_body(self, reasoning: bool) -> dict[str, Any]:
        """`extra_body` fragment carrying the ONE piece of per-vendor request
        shape this project has hit in practice: how reasoning is toggled,
        plus any vendor-only sampling extras. It does not attempt to solve
        structured-output reliability differences, which are a model-behavior
        problem no request-shape hook can fix.

        "vllm": Qwen3's `enable_thinking` chat-template kwarg, plus
        `repetition_penalty` (a vLLM sampling extension — see _REPETITION_
        PENALTY; kept here because this dict is already every call site's
        only vLLM-specific extra_body seam). When reasoning=True it also
        sends `reasoning_effort` (see _REASONING_EFFORT): measured 2026-08-31
        against the new reasoning model (Qwen3.8-Flash-Next-NVFP4, SGLang) —
        without it, a reasoning call burns all of max_tokens on hidden
        reasoning_content and returns an empty body, which cascades into a
        structured_complete validation failure and an identically-empty
        fallback. Sent unconditionally (no per-model branch) because the
        currently-deployed vLLM model (Qwen3.6) was confirmed to accept and
        silently ignore this field with no behavior change. Omitted when
        reasoning=False since enable_thinking already disables reasoning
        there and there is no failure mode to guard against.

        "gateway": the gateway doesn't understand `enable_thinking` and its
        upstreams differ in which sampling params they accept, so both vLLM
        extras are dropped — `_looks_degenerate` stays as the (now sole)
        runaway-repetition backstop. Reasoning is requested only when a call
        actually asks for it, via the OpenAI-standard `reasoning_effort`
        field (the gateway translates it per upstream). reasoning=False
        sends nothing: the gateway's own thinking-budget config plus
        complete()'s existing empty/degenerate-output retries cover a model
        that over-reasons.
        모델별 reasoning 토글 흡수 지점 — gateway는 벤더 확장 제거, reasoning=True일
        때만 표준 reasoning_effort 전송."""
        if self._provider == "gateway":
            if reasoning:
                return {"reasoning_effort": settings.llm_gateway_reasoning_effort}
            return {}
        body: dict[str, Any] = {
            "chat_template_kwargs": {"enable_thinking": reasoning},
            "repetition_penalty": _REPETITION_PENALTY,
        }
        if reasoning:
            body["reasoning_effort"] = _REASONING_EFFORT
        return body

    @staticmethod
    def _with_language_instruction(messages: list[dict[str, Any]], language: str) -> list[dict[str, Any]]:
        instruction = language_instruction(language)
        result = list(messages)
        if result and result[0].get("role") == "system":
            result[0] = {**result[0], "content": f"{result[0]['content']}\n\n{instruction}"}
        else:
            result.insert(0, {"role": "system", "content": instruction})
        return result

    async def _retry_once_if_hanja(
        self,
        result: T,
        raw: str,
        *,
        language: str,
        label: str,
        retry_call: Callable[[], Awaitable[tuple[T, str]]],
    ) -> T:
        """Shared by structured_complete's native and fallback success paths
        — a structured field (e.g. the Portfolio Manager's
        executive_summary) can just as easily pick up Chinese-character
        contamination as complete()'s free-form prose does (see
        _contains_hanja/project memory: llm-korean-hanja-mixing-issue), but
        neither path checked language before this, only schema validity.
        Same belt-and-suspenders shape as complete()'s own hanja retry: one
        retry at a different temperature, same request shape. If the retry
        itself fails (request error or validation error), keep the
        already-valid original — this is a language-quality issue, not the
        fail-loud kind structured_complete reserves for a decision that
        never validated at all.

        Checks for hanja regardless of `language` (not gated to language ==
        "ko" as it used to be): every pipeline call now runs in English by
        deliberate decision (app.llm.language.DEFAULT_LANGUAGE), but
        app.state.PortfolioDecision.card_reason_ko is a Korean field
        embedded inside that English-language response — on an English
        response the only Korean content is that field, so any hanja
        appearing anywhere in `raw` is contamination by definition, not
        legitimate content, and would otherwise go completely unguarded.
        This stays safe to broaden because the retry is soft (see above): at
        most one extra call, and the original valid result survives either
        way, so this can never turn into a hard failure for a real decision."""
        if not _contains_hanja(raw):
            return result
        logger.warning(
            "structured_complete label=%s: Chinese-character (hanja) contamination detected in Korean "
            "output — retrying at temperature=%s",
            label, _DEGENERATE_RETRY_TEMPERATURE,
        )
        try:
            retried_result, _ = await retry_call()
            return retried_result
        except Exception as exc:  # noqa: BLE001 - keep the already-valid original rather than fail the call
            logger.warning(
                "structured_complete label=%s: hanja retry attempt itself failed (%s) — keeping original result",
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
        """`max_tokens`, if given, overrides `settings.llm_max_output_tokens`
        for this call only (2026-09-07: app.agents.debater passes a tighter
        round-1 debate-turn budget, `settings.debate_round1_max_output_tokens` —
        see its own comment for why only round 1, never the final round).
        `None` (the default) keeps today's behavior — every other caller of
        `complete()` is unaffected."""
        prepared_messages = self._with_language_instruction(messages, language)
        effective_max_tokens = max_tokens if max_tokens is not None else settings.llm_max_output_tokens
        # Exposes the last response's finish_reason to the empty-response
        # error message below (2.2) without adding a field to LLMMessage
        # itself (a domain model — app.domain.models.llm — that every other
        # caller shouldn't need to know about transport-level metadata).
        last_finish_reason: dict[str, str | None] = {"value": None}

        async def _call(effective_temperature: float) -> LLMMessage:
            kwargs: dict[str, Any] = {
                "model": settings.llm_model,
                "messages": prepared_messages,
                "temperature": effective_temperature,
                "extra_body": self._reasoning_extra_body(reasoning),
                "max_tokens": effective_max_tokens,
            }
            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "auto"

            start = time.monotonic()
            response = await self._client.chat.completions.create(**kwargs)
            # Metrics are recorded for every response received, valid or
            # not — an empty-choices response is itself a data point
            # (finish_reason, latency) worth keeping, not just a failure
            # to discard silently.
            self._record_metrics(
                response, label=label, reasoning=reasoning, structured=False, start=start,
                max_tokens=effective_max_tokens,
            )
            if not getattr(response, "choices", None):
                # Not retried: a 200 with no choices at all is a server-side
                # anomaly, not something a temperature change fixes — unlike
                # structured_complete's equivalent access (which is already
                # inside a broad except and falls back), complete() has no
                # fallback path to fall back TO, so this must fail loud
                # immediately rather than crash later on choices[0].
                raise ValueError(f"llm complete label={label!r}: response contained no choices")
            last_finish_reason["value"] = getattr(response.choices[0], "finish_reason", None)
            message = response.choices[0].message
            tool_calls = []
            for call in message.tool_calls or []:
                try:
                    arguments = json.loads(call.function.arguments)
                except json.JSONDecodeError:
                    # Skip, don't default to {} — a tool with all-optional
                    # arguments would otherwise silently run with fabricated
                    # defaults instead of the args the model actually meant.
                    # Never calling the tool with malformed args is safer
                    # than calling it with wrong ones. If every tool_calls
                    # entry is dropped this way, tool_calls ends up empty
                    # here — the empty-response guard below (in the caller)
                    # then handles that the same way as a bare empty string.
                    logger.warning(
                        "llm complete label=%s: tool call %r (%s) had malformed JSON arguments — skipping: %r",
                        label, getattr(call, "id", None), getattr(call.function, "name", None),
                        call.function.arguments[:120],
                    )
                    continue
                tool_calls.append({"id": call.id, "name": call.function.name, "arguments": arguments})
            # Strip a reasoning leak from the body BEFORE it becomes an
            # LLMMessage — one seam covers complete()'s four call sites and
            # every retry (2026-09-01, id=1156: a risk-debate turn published
            # ~330 lines of its own deliberation and it changed the decision;
            # structured_complete is already immune via _extract_json). Skip
            # entirely when tool_calls is non-empty — same as the empty-body
            # and _looks_degenerate guards. Cannot turn non-empty into empty
            # (see _strip_reasoning_leak's invariant).
            raw_content = message.content or ""
            if tool_calls:
                sanitized_content, sanitized_kind = raw_content, None
            else:
                sanitized_content, sanitized_kind = _strip_reasoning_leak(raw_content, label=label)
            if sanitized_kind is not None:
                llm_metrics.note_sanitization(f"{label or 'unlabeled'}: {sanitized_kind}")
            return LLMMessage(
                # vLLM exposes reasoning as `reasoning_content`; some gateway
                # upstreams as `reasoning`. Neither is load-bearing downstream
                # (metrics/log only), but capture whichever the backend used.
                content=sanitized_content,
                reasoning_content=(
                    getattr(message, "reasoning_content", None) or getattr(message, "reasoning", None)
                ),
                tool_calls=tool_calls,
            )

        # Priority admission: one acquire/release per logical `complete()`
        # call, covering every internal retry below as a single unit — see
        # PriorityLimiter's own docstring for why (live calls skip the line
        # ahead of a backtest/analysis burst sharing this one endpoint).
        # acquire() is deliberately OUTSIDE the try/finally below: if it
        # raises (e.g. cancelled while parked), we never held the slot in a
        # way that needs releasing (acquire()'s own cancellation handler
        # already cleans up), so entering `finally` here would double-release.
        await self._limiter.acquire(self._current_priority())
        try:
            result = await _call(temperature)

            # Empty-response check comes FIRST: an empty string trips
            # neither _looks_degenerate nor _contains_hanja, so if this were
            # an elif alongside them it would never fire. Unlike those two,
            # this is not soft — structured_complete already treats an empty
            # native response as an explicit failure (2026-07-28 gpt-oss
            # incident), and complete() silently returning "" used to let a
            # blank analyst report pass app.orchestrator.pipeline's
            # degraded-report check and get cached as a "clean" run for its
            # full TTL. A tool-call turn's content is normally empty/minimal
            # by design (tool_loop.py consumes tool_calls, not content), so
            # that shape is explicitly exempted.
            if not result.tool_calls and not result.content.strip():
                logger.warning(
                    "llm complete: empty content with no tool calls (label=%s, finish_reason=%s) — "
                    "retrying at temperature=%s",
                    label, last_finish_reason["value"], _DEGENERATE_RETRY_TEMPERATURE,
                )
                result = await _call(_DEGENERATE_RETRY_TEMPERATURE)
                if not result.tool_calls and not result.content.strip():
                    raise ValueError(
                        f"llm complete label={label!r}: model returned empty content with no tool calls "
                        f"twice in a row (finish_reason={last_finish_reason['value']!r}) — treating as a "
                        "genuine failure rather than caching a blank result"
                    )
            # Only a plain-text answer can be degenerate in this sense — a
            # tool-call response's `content` is normally empty/minimal, so
            # there's nothing meaningful to detect here. One retry at a higher
            # temperature (still deterministic-ish, not the model's max) is the
            # same "different request shape routes around a stuck failure mode"
            # logic structured_complete's fallback already uses, just applied to
            # a decoding-loop failure instead of a response_format failure.
            #
            # Both retries below are SOFT, same doctrine as
            # _retry_hanja_if_needed's structured_complete equivalent: if the
            # retry attempt itself raises, keep the already-valid original
            # rather than losing it — a quality-improvement retry must never
            # turn a working response into a hard failure. (CancelledError is
            # BaseException, not Exception, so a genuine cancellation during
            # the retry still propagates rather than being swallowed here.)
            elif not result.tool_calls and _looks_degenerate(result.content):
                logger.warning(
                    "llm complete: degenerate repeated-paragraph output detected (label=%s) — retrying at temperature=%s",
                    label, _DEGENERATE_RETRY_TEMPERATURE,
                )
                try:
                    result = await _call(_DEGENERATE_RETRY_TEMPERATURE)
                except Exception as exc:  # noqa: BLE001 - keep the already-valid original
                    logger.warning(
                        "llm complete label=%s: degenerate-output retry itself failed (%s) — keeping original result",
                        label, exc,
                    )
            elif not result.tool_calls and language == "ko" and _contains_hanja(result.content):
                # Only "ko" calls — NOT the language-independent scope
                # _retry_hanja_if_needed uses for structured_complete. That
                # broadening exists specifically for
                # app.state.PortfolioDecision.card_reason_ko, a Korean field
                # embedded in an otherwise-English structured response; every
                # card_reason_ko-producing call goes through
                # structured_complete (app.agents.portfolio_manager), never
                # through complete(), so complete() has no equivalent
                # "Korean field hiding inside an English response" case to
                # guard — an "en" complete() call legitimately contains no
                # Hangul either, so there's no "this script shouldn't appear
                # here" signal to check for it outside "ko".
                logger.warning(
                    "llm complete: Chinese-character (hanja) contamination detected in Korean output "
                    "(label=%s) — retrying at temperature=%s",
                    label, _DEGENERATE_RETRY_TEMPERATURE,
                )
                try:
                    result = await _call(_DEGENERATE_RETRY_TEMPERATURE)
                except Exception as exc:  # noqa: BLE001 - keep the already-valid original
                    logger.warning(
                        "llm complete label=%s: hanja retry attempt itself failed (%s) — keeping original result",
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
        """Native `response_format` structured output first; if the server
        returns nothing usable, falls back to an unconstrained completion
        with the schema spelled out in the prompt instead. Real incident
        this guards against (2026-07-28): swapping to a gpt-oss-20b vLLM
        endpoint made every `response_format` call return an EMPTY
        `message.content` (confirmed server-side bug in that backend's
        guided-decoding/Harmony-format handling, not this project's own
        schema — see project memory: agents-prompt-quality-review's LLM-swap
        discussion), while the exact same messages with NO response_format
        constraint got a normal answer from the same model. The two failure
        shapes that trigger this: empty content, or content that doesn't
        validate against `schema` even after `_extract_json`'s cleanup.
        Also falls back on a request-level failure (HTTP 5xx / connection
        error) from the native call itself, not just a bad response — real
        incident (2026-07-28): the same gpt-oss-20b endpoint intermittently
        returned HTTP 500 for the response_format-constrained request,
        persisting through AsyncOpenAI's own max_retries=5 against that
        identical request shape, but succeeded once retried against the
        fallback's differently-shaped (unconstrained) request instead — see
        _FALLBACK_MAX_ATTEMPTS's comment.

        Fails loud (re-raises) if the fallback ALSO doesn't validate — never
        silently substitutes a default/empty value for a trading decision.
        구조화 출력이 깨지면(빈 응답/요청 자체 실패 등) 프롬프트에 스키마를 명시한
        일반 완성으로 폴백 — 그것도 실패하면 조용히 넘어가지 않고 에러로 표면화."""
        prepared_messages = self._with_language_instruction(messages, language)
        native_call_index = 0

        async def _native_call(effective_temperature: float) -> tuple[T, str]:
            nonlocal native_call_index
            native_call_index += 1
            start = time.monotonic()
            request_kwargs = {
                "model": settings.llm_model,
                "messages": prepared_messages,
                "temperature": effective_temperature,
                "extra_body": self._reasoning_extra_body(reasoning),
                "max_tokens": settings.llm_max_output_tokens,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema.__name__,
                        "schema": schema.model_json_schema(),
                        "strict": True,
                    },
                },
            }
            capture_id = observe_first_native_request(
                request_kwargs, label=label, first=native_call_index == 1,
                provider=self._provider, sdk_max_retries=self._client.max_retries,
                timeout_seconds=self._client.timeout,
            )
            try:
                response = await self._client.chat.completions.create(**request_kwargs)
            except BaseException as exc:
                observe_first_native_result(capture_id, error=exc, elapsed_seconds=time.monotonic() - start)
                raise
            observe_first_native_result(capture_id, response=response, elapsed_seconds=time.monotonic() - start)
            self._record_metrics(
                response, label=label, reasoning=reasoning, structured=True, start=start, fallback_used=False
            )
            raw = response.choices[0].message.content or ""
            try:
                cleaned = _clean_json_payload(raw, label=label, path="native")
                result = schema.model_validate_json(cleaned)
            except Exception as exc:
                observe_first_native_parse(capture_id, error=exc)
                raise
            observe_first_native_parse(capture_id)
            return result, raw

        # Same one-acquire-per-logical-call shape as complete() — covers the
        # fallback delegation below too, since that's an internal
        # continuation of this same call, not a separate public entry point.
        await self._limiter.acquire(self._current_priority())
        try:
            try:
                result, raw = await _native_call(temperature)
            except (APIStatusError, APIConnectionError) as exc:
                if _is_quota_exhausted(exc):
                    # Not a broken-guided-decoding failure — the fallback
                    # would just fire more requests at a limit we already
                    # hit. Surface the real cause instead.
                    logger.warning(
                        "structured_complete label=%s: native request hit a quota/rate limit (%s) — "
                        "NOT falling back; re-raising",
                        label, exc,
                    )
                    raise
                logger.warning(
                    "structured_complete label=%s: native response_format request itself failed (%s) — "
                    "falling back to unconstrained completion",
                    label, exc,
                )
                return await self._structured_complete_fallback(
                    prepared_messages, schema, reasoning=reasoning, temperature=temperature, label=label,
                    language=language, native_error=exc,
                )
            except Exception as exc:  # noqa: BLE001 - deliberately broad: any parse/validation failure falls back
                logger.warning(
                    "structured_complete label=%s: native response_format produced no valid %s (%s) — "
                    "falling back to unconstrained completion",
                    label, schema.__name__, exc,
                )
                return await self._structured_complete_fallback(
                    prepared_messages, schema, reasoning=reasoning, temperature=temperature, label=label,
                    language=language, native_error=exc,
                )
            return await self._retry_once_if_hanja(
                result, raw, language=language, label=label,
                retry_call=lambda: _native_call(_DEGENERATE_RETRY_TEMPERATURE),
            )
        finally:
            await self._limiter.release()

    async def _structured_complete_fallback(
        self,
        prepared_messages: list[dict[str, Any]],
        schema: type[T],
        *,
        reasoning: bool,
        temperature: float,
        label: str,
        language: str,
        native_error: Exception,
    ) -> T:
        """Fallback for `structured_complete`: no `response_format` constraint
        at all (the code path already proven to work against a backend whose
        guided-decoding integration is broken — see `structured_complete`'s
        docstring), with the schema spelled out as plain-text instructions
        instead. Retries up to `_FALLBACK_MAX_ATTEMPTS` times on a
        request-level failure (HTTP 5xx/connection error) with a short delay
        — separate from AsyncOpenAI's own max_retries, which was already
        exhausted against the NATIVE request's shape by the time we get
        here (see _FALLBACK_MAX_ATTEMPTS's own comment). Raises (does not
        swallow) if every attempt fails, whether by request error or by
        producing content that still doesn't validate — a trading decision
        must never silently become a default/empty value."""
        schema_instruction = {
            "role": "user",
            "content": (
                f"Your previous response could not be parsed as valid JSON matching the required "
                f"schema. Respond with ONLY a single JSON object matching this schema — no markdown, "
                f"no commentary, no extra text before or after it:\n{json.dumps(schema.model_json_schema())}"
            ),
        }
        fallback_messages = [*prepared_messages, schema_instruction]
        last_error: Exception = native_error

        async def _fallback_call(effective_temperature: float) -> tuple[T, str]:
            start = time.monotonic()
            response = await self._client.chat.completions.create(
                model=settings.llm_model,
                messages=fallback_messages,
                temperature=effective_temperature,
                extra_body=self._reasoning_extra_body(reasoning),
                max_tokens=settings.llm_max_output_tokens,
            )
            self._record_metrics(
                response, label=label, reasoning=reasoning, structured=True, start=start, fallback_used=True
            )
            raw = response.choices[0].message.content or ""
            cleaned = _clean_json_payload(raw, label=label, path="fallback")
            return schema.model_validate_json(cleaned), raw

        for attempt in range(1, _FALLBACK_MAX_ATTEMPTS + 1):
            try:
                result, raw = await _fallback_call(temperature)
            except (APIStatusError, APIConnectionError) as exc:
                last_error = exc
                logger.warning(
                    "structured_complete label=%s: fallback request itself failed on attempt %d/%d (%s)",
                    label, attempt, _FALLBACK_MAX_ATTEMPTS, exc,
                )
                if attempt < _FALLBACK_MAX_ATTEMPTS:
                    await asyncio.sleep(_FALLBACK_RETRY_DELAY_SECONDS)
                continue
            except Exception as fallback_error:  # noqa: BLE001 - retry loop needs the broad catch too
                last_error = fallback_error
                logger.warning(
                    "structured_complete label=%s: fallback produced no valid %s on attempt %d/%d (%s)",
                    label, schema.__name__, attempt, _FALLBACK_MAX_ATTEMPTS, fallback_error,
                )
                if attempt < _FALLBACK_MAX_ATTEMPTS:
                    await asyncio.sleep(_FALLBACK_RETRY_DELAY_SECONDS)
                continue

            return await self._retry_once_if_hanja(
                result, raw, language=language, label=label,
                retry_call=lambda: _fallback_call(_DEGENERATE_RETRY_TEMPERATURE),
            )

        raise ValueError(
            f"structured_complete label={label!r}: both native response_format and the unconstrained "
            f"fallback (after {_FALLBACK_MAX_ATTEMPTS} attempts) failed to produce a valid "
            f"{schema.__name__} — native error: {native_error!r}, last fallback error: {last_error!r}"
        ) from last_error

    def _record_metrics(
        self, response: Any, *, label: str, reasoning: bool, structured: bool, start: float,
        fallback_used: bool = False, max_tokens: int | None = None,
    ) -> None:
        # `max_tokens` is the ceiling THIS call actually used — passed
        # explicitly by complete() (which may have overridden the global
        # default, e.g. app.agents.debater's round-1 debate-turn budget);
        # structured_complete's two call sites don't override it, so `None`
        # here correctly falls back to the global default, same value they
        # always used.
        effective_max_tokens = max_tokens if max_tokens is not None else settings.llm_max_output_tokens
        usage = getattr(response, "usage", None)
        choices = getattr(response, "choices", None) or []
        finish_reason = getattr(choices[0], "finish_reason", None) if choices else None
        llm_metrics.record(
            label=label,
            model=settings.llm_model,
            reasoning=reasoning,
            structured=structured,
            fallback_used=fallback_used,
            latency_seconds=time.monotonic() - start,
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
            total_tokens=getattr(usage, "total_tokens", None),
            finish_reason=finish_reason,
            max_tokens=effective_max_tokens,
        )
        if finish_reason == _TRUNCATED_FINISH_REASON:
            # Single choke point for all three call shapes (complete(),
            # structured_complete()'s native path, and its fallback path) —
            # see LLMCallMetric.finish_reason's own comment for why this
            # matters (a truncated structured_complete misdiagnoses as
            # "invalid schema", a truncated complete() report passes as
            # complete). Deliberately log-only: retrying at a different
            # temperature doesn't un-truncate anything, and llm_max_output_
            # tokens is a config decision, not something this call layer
            # should second-guess. Logs `effective_max_tokens`, not the
            # global setting, so a round-1 debate-turn truncation is reported
            # against the ceiling that actually applied (2026-09-07) instead
            # of always blaming the (much higher) global default.
            logger.warning(
                "llm call label=%s hit the max_tokens ceiling (finish_reason=%s, max_tokens=%s, "
                "structured=%s, fallback=%s) — the response is TRUNCATED",
                label, finish_reason, effective_max_tokens, structured, fallback_used,
            )


llm_client = LLMClient()
