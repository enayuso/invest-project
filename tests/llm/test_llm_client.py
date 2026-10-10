"""app.llm.client.LLMClient — the OpenAI SDK call itself mocked, no network.
Verifies both the pre-existing Korean-instruction/parsing behavior and the
new metrics recording (agent-architecture-roadmap stage 0)."""

from __future__ import annotations

from dataclasses import dataclass, field

import httpx
import pytest
from openai import APIConnectionError, APIStatusError
from pydantic import BaseModel

from app.llm import client as llm_client_module
from app.llm import metrics
from app.llm.client import LLMClient, _DEGENERATE_RETRY_TEMPERATURE, _contains_hanja, _extract_json, _looks_degenerate


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch):
    # The fallback retry loop's inter-attempt delay would otherwise add real
    # wall-clock seconds to every test that exercises a retry.
    async def instant_sleep(seconds):
        return None

    monkeypatch.setattr(llm_client_module.asyncio, "sleep", instant_sleep)


class _FakeToolCallFn:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class _FakeToolCall:
    def __init__(self, id, name, arguments):
        self.id = id
        self.function = _FakeToolCallFn(name, arguments)


class _FakeMessage:
    def __init__(self, content="", tool_calls=None, reasoning_content=None):
        self.content = content
        self.tool_calls = tool_calls or []
        self.reasoning_content = reasoning_content


@dataclass
class _FakeUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass
class _FakeChoice:
    message: _FakeMessage
    finish_reason: str | None = None


@dataclass
class _FakeResponse:
    choices: list = field(default_factory=list)
    usage: _FakeUsage | None = None


class _SampleSchema(BaseModel):
    value: str


@pytest.fixture(autouse=True)
def _clean_metrics():
    metrics.clear()
    yield
    metrics.clear()


@pytest.fixture
def client():
    return LLMClient()


async def test_complete_parses_content_and_tool_calls(client, monkeypatch):
    response = _FakeResponse(
        choices=[_FakeChoice(_FakeMessage(
            content="hello", tool_calls=[_FakeToolCall("call1", "get_price", '{"ticker": "AAPL"}')],
        ))],
        usage=_FakeUsage(10, 5, 15),
    )

    async def fake_create(**kwargs):
        return response

    monkeypatch.setattr(client._client.chat.completions, "create", fake_create)
    result = await client.complete([{"role": "user", "content": "hi"}], label="test_agent")

    assert result.content == "hello"
    assert result.tool_calls == [{"id": "call1", "name": "get_price", "arguments": {"ticker": "AAPL"}}]


async def test_complete_injects_english_instruction_by_default(client, monkeypatch):
    captured = {}

    async def fake_create(**kwargs):
        captured["messages"] = kwargs["messages"]
        return _FakeResponse(choices=[_FakeChoice(_FakeMessage(content="ok"))], usage=_FakeUsage(1, 1, 2))

    monkeypatch.setattr(client._client.chat.completions, "create", fake_create)
    await client.complete(
        [{"role": "system", "content": "You are an analyst."}, {"role": "user", "content": "hi"}]
    )
    assert "Always respond in English only" in captured["messages"][0]["content"]
    assert "You are an analyst." in captured["messages"][0]["content"]


async def test_complete_injects_english_instruction_when_language_is_en(client, monkeypatch):
    captured = {}

    async def fake_create(**kwargs):
        captured["messages"] = kwargs["messages"]
        return _FakeResponse(choices=[_FakeChoice(_FakeMessage(content="ok"))], usage=_FakeUsage(1, 1, 2))

    monkeypatch.setattr(client._client.chat.completions, "create", fake_create)
    await client.complete([{"role": "user", "content": "hi"}], language="en")
    assert "Always respond in English only" in captured["messages"][0]["content"]
    assert "한국어" not in captured["messages"][0]["content"]


async def test_structured_complete_also_applies_the_language_instruction(client, monkeypatch):
    # Previously structured_complete skipped the language instruction
    # entirely (a real bug — research/trader/portfolio-manager/sentiment
    # output could come back in whatever language the raw system prompt's
    # own language defaulted to, regardless of what the user asked in).
    captured = {}

    async def fake_create(**kwargs):
        captured["messages"] = kwargs["messages"]
        return _FakeResponse(choices=[_FakeChoice(_FakeMessage(content='{"value": "x"}'))], usage=_FakeUsage(1, 1, 2))

    monkeypatch.setattr(client._client.chat.completions, "create", fake_create)
    await client.structured_complete(
        [{"role": "system", "content": "You are the Portfolio Manager."}, {"role": "user", "content": "decide"}],
        _SampleSchema,
        language="en",
    )
    assert "Always respond in English only" in captured["messages"][0]["content"]


async def test_structured_complete_defaults_to_english_instruction(client, monkeypatch):
    captured = {}

    async def fake_create(**kwargs):
        captured["messages"] = kwargs["messages"]
        return _FakeResponse(choices=[_FakeChoice(_FakeMessage(content='{"value": "x"}'))], usage=_FakeUsage(1, 1, 2))

    monkeypatch.setattr(client._client.chat.completions, "create", fake_create)
    await client.structured_complete([{"role": "user", "content": "decide"}], _SampleSchema)
    assert "Always respond in English only" in captured["messages"][0]["content"]


async def test_complete_records_metrics_with_label(client, monkeypatch):
    async def fake_create(**kwargs):
        return _FakeResponse(choices=[_FakeChoice(_FakeMessage(content="ok"))], usage=_FakeUsage(20, 10, 30))

    monkeypatch.setattr(client._client.chat.completions, "create", fake_create)
    await client.complete([{"role": "user", "content": "hi"}], label="market_analyst", reasoning=True)

    recorded = metrics.recent(1)[0]
    assert recorded.label == "market_analyst"
    assert recorded.reasoning is True
    assert recorded.structured is False
    assert recorded.total_tokens == 30
    assert recorded.latency_seconds >= 0.0


async def test_complete_records_finish_reason(client, monkeypatch):
    async def fake_create(**kwargs):
        return _FakeResponse(
            choices=[_FakeChoice(_FakeMessage(content="ok"), finish_reason="stop")], usage=_FakeUsage(1, 1, 2)
        )

    monkeypatch.setattr(client._client.chat.completions, "create", fake_create)
    await client.complete([{"role": "user", "content": "hi"}], label="x")
    assert metrics.recent(1)[0].finish_reason == "stop"


async def test_record_metrics_finish_reason_is_none_when_absent(client, monkeypatch):
    async def fake_create(**kwargs):
        return _FakeResponse(choices=[_FakeChoice(_FakeMessage(content="ok"))], usage=_FakeUsage(1, 1, 2))

    monkeypatch.setattr(client._client.chat.completions, "create", fake_create)
    await client.complete([{"role": "user", "content": "hi"}], label="x")
    assert metrics.recent(1)[0].finish_reason is None


async def test_complete_logs_a_truncation_warning_when_length_capped(client, monkeypatch, caplog):
    import logging

    async def fake_create(**kwargs):
        return _FakeResponse(
            choices=[_FakeChoice(_FakeMessage(content="cut off mid-sent"), finish_reason="length")],
            usage=_FakeUsage(16000, 0, 16000),
        )

    monkeypatch.setattr(client._client.chat.completions, "create", fake_create)
    with caplog.at_level(logging.WARNING):
        await client.complete([{"role": "user", "content": "hi"}], label="market_analyst")
    assert any("TRUNCATED" in r.message for r in caplog.records)


async def test_complete_no_truncation_warning_on_a_normal_stop(client, monkeypatch, caplog):
    import logging

    async def fake_create(**kwargs):
        return _FakeResponse(
            choices=[_FakeChoice(_FakeMessage(content="ok"), finish_reason="stop")], usage=_FakeUsage(1, 1, 2)
        )

    monkeypatch.setattr(client._client.chat.completions, "create", fake_create)
    with caplog.at_level(logging.WARNING):
        await client.complete([{"role": "user", "content": "hi"}], label="x")
    assert not any("TRUNCATED" in r.message for r in caplog.records)


# ---------------- per-call max_tokens override (2026-09-07, AI-engine
# step-0 latency audit — round-1 debate-turn budget) ----------------


async def test_complete_uses_global_max_tokens_when_no_override_given(client, monkeypatch):
    captured = {}

    async def fake_create(**kwargs):
        captured.update(kwargs)
        return _FakeResponse(choices=[_FakeChoice(_FakeMessage(content="ok"))], usage=_FakeUsage(1, 1, 2))

    monkeypatch.setattr(client._client.chat.completions, "create", fake_create)
    from app.config import settings

    await client.complete([{"role": "user", "content": "hi"}], label="x")
    assert captured["max_tokens"] == settings.llm_max_output_tokens
    assert metrics.recent(1)[0].max_tokens == settings.llm_max_output_tokens


async def test_complete_max_tokens_override_reaches_the_request(client, monkeypatch):
    captured = {}

    async def fake_create(**kwargs):
        captured.update(kwargs)
        return _FakeResponse(choices=[_FakeChoice(_FakeMessage(content="ok"))], usage=_FakeUsage(1, 1, 2))

    monkeypatch.setattr(client._client.chat.completions, "create", fake_create)
    await client.complete([{"role": "user", "content": "hi"}], label="risk_neutral", max_tokens=1100)
    assert captured["max_tokens"] == 1100
    assert metrics.recent(1)[0].max_tokens == 1100


async def test_complete_truncation_warning_logs_the_overridden_ceiling_not_the_global_one(client, monkeypatch, caplog):
    import logging

    async def fake_create(**kwargs):
        return _FakeResponse(
            choices=[_FakeChoice(_FakeMessage(content="cut off"), finish_reason="length")],
            usage=_FakeUsage(500, 1100, 1600),
        )

    monkeypatch.setattr(client._client.chat.completions, "create", fake_create)
    with caplog.at_level(logging.WARNING):
        await client.complete([{"role": "user", "content": "hi"}], label="risk_neutral", max_tokens=1100)
    truncation_records = [r.message for r in caplog.records if "TRUNCATED" in r.message]
    assert len(truncation_records) == 1
    assert "max_tokens=1100" in truncation_records[0]
    assert "max_tokens=16000" not in truncation_records[0]


async def test_complete_defaults_label_to_empty_string_when_omitted(client, monkeypatch):
    async def fake_create(**kwargs):
        return _FakeResponse(choices=[_FakeChoice(_FakeMessage(content="ok"))], usage=_FakeUsage(1, 1, 2))

    monkeypatch.setattr(client._client.chat.completions, "create", fake_create)
    await client.complete([{"role": "user", "content": "hi"}])
    assert metrics.recent(1)[0].label == ""


async def test_structured_complete_parses_schema_and_records_metrics(client, monkeypatch):
    async def fake_create(**kwargs):
        return _FakeResponse(
            choices=[_FakeChoice(_FakeMessage(content='{"value": "buy"}'))],
            usage=_FakeUsage(5, 3, 8),
        )

    monkeypatch.setattr(client._client.chat.completions, "create", fake_create)
    result = await client.structured_complete(
        [{"role": "user", "content": "classify"}], _SampleSchema, label="research_manager"
    )

    assert result.value == "buy"
    recorded = metrics.recent(1)[0]
    assert recorded.label == "research_manager"
    assert recorded.structured is True
    assert recorded.total_tokens == 8


async def test_structured_complete_strips_markdown_noise_before_json(client, monkeypatch, caplog):
    async def fake_create(**kwargs):
        return _FakeResponse(
            choices=[_FakeChoice(_FakeMessage(content='\n\n#{"value": "buy"}'))],
            usage=_FakeUsage(5, 3, 8),
        )

    monkeypatch.setattr(client._client.chat.completions, "create", fake_create)
    with caplog.at_level("WARNING"):
        result = await client.structured_complete(
            [{"role": "user", "content": "classify"}], _SampleSchema, label="research_manager"
        )

    assert result.value == "buy"
    assert any("stripped non-JSON wrapper" in r.message for r in caplog.records)


async def test_structured_complete_no_warning_when_already_clean_json(client, monkeypatch, caplog):
    async def fake_create(**kwargs):
        return _FakeResponse(
            choices=[_FakeChoice(_FakeMessage(content='{"value": "buy"}'))],
            usage=_FakeUsage(5, 3, 8),
        )

    monkeypatch.setattr(client._client.chat.completions, "create", fake_create)
    with caplog.at_level("WARNING"):
        await client.structured_complete([{"role": "user", "content": "classify"}], _SampleSchema)

    assert not any("stripped non-JSON wrapper" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# structured_complete fallback — native response_format returns nothing
# usable (real incident 2026-07-28: a gpt-oss-20b vLLM endpoint's guided-
# decoding integration returned empty content for every response_format
# call, while the identical messages with no response_format worked fine)
# ---------------------------------------------------------------------------


async def test_structured_complete_success_never_calls_fallback(client, monkeypatch):
    calls = []

    async def fake_create(**kwargs):
        calls.append(kwargs)
        return _FakeResponse(choices=[_FakeChoice(_FakeMessage(content='{"value": "buy"}'))], usage=_FakeUsage(5, 3, 8))

    monkeypatch.setattr(client._client.chat.completions, "create", fake_create)
    result = await client.structured_complete([{"role": "user", "content": "classify"}], _SampleSchema)

    assert result.value == "buy"
    assert len(calls) == 1  # no fallback call made
    assert "response_format" in calls[0]


async def test_structured_complete_falls_back_when_native_returns_empty_content(client, monkeypatch, caplog):
    calls = []
    responses = [
        _FakeResponse(choices=[_FakeChoice(_FakeMessage(content=None))], usage=_FakeUsage(5, 3, 8)),  # native: empty
        _FakeResponse(choices=[_FakeChoice(_FakeMessage(content='{"value": "sell"}'))], usage=_FakeUsage(6, 4, 10)),  # fallback: valid
    ]

    async def fake_create(**kwargs):
        calls.append(kwargs)
        return responses.pop(0)

    monkeypatch.setattr(client._client.chat.completions, "create", fake_create)
    with caplog.at_level("WARNING"):
        result = await client.structured_complete(
            [{"role": "user", "content": "classify"}], _SampleSchema, label="research_manager"
        )

    assert result.value == "sell"
    assert len(calls) == 2
    assert "response_format" in calls[0]  # native attempt used response_format
    assert "response_format" not in calls[1]  # fallback does not
    assert any("falling back to unconstrained completion" in r.message for r in caplog.records)

    recorded = metrics.recent(2)
    assert recorded[0].fallback_used is True  # newest first
    assert recorded[1].fallback_used is False


async def test_structured_complete_fallback_logs_when_it_strips_wrapper_noise(client, monkeypatch, caplog):
    """2026-08-26 regression: the fallback path used to have no equivalent
    of the native path's "stripped non-JSON wrapper" warning — exactly the
    path most worth watching, since it means guided-decoding wasn't even
    attempted."""
    responses = [
        _FakeResponse(choices=[_FakeChoice(_FakeMessage(content=None))], usage=_FakeUsage(5, 3, 8)),  # native: empty
        _FakeResponse(
            choices=[_FakeChoice(_FakeMessage(content='\n\n#{"value": "sell"}'))], usage=_FakeUsage(6, 4, 10)
        ),  # fallback: wrapped

    ]

    async def fake_create(**kwargs):
        return responses.pop(0)

    monkeypatch.setattr(client._client.chat.completions, "create", fake_create)
    with caplog.at_level("WARNING"):
        result = await client.structured_complete(
            [{"role": "user", "content": "classify"}], _SampleSchema, label="research_manager"
        )

    assert result.value == "sell"
    stripped_logs = [r.message for r in caplog.records if "stripped non-JSON wrapper" in r.message]
    assert len(stripped_logs) == 1
    assert "fallback" in stripped_logs[0]


async def test_structured_complete_falls_back_when_native_content_fails_validation(client, monkeypatch):
    responses = [
        _FakeResponse(choices=[_FakeChoice(_FakeMessage(content="{}"))], usage=_FakeUsage(5, 3, 8)),  # missing required field
        _FakeResponse(choices=[_FakeChoice(_FakeMessage(content='{"value": "hold"}'))], usage=_FakeUsage(6, 4, 10)),
    ]

    async def fake_create(**kwargs):
        return responses.pop(0)

    monkeypatch.setattr(client._client.chat.completions, "create", fake_create)
    result = await client.structured_complete([{"role": "user", "content": "classify"}], _SampleSchema)

    assert result.value == "hold"


async def test_structured_complete_raises_when_native_and_fallback_both_fail(client, monkeypatch):
    # 1 native attempt + _FALLBACK_MAX_ATTEMPTS (2) fallback attempts, all empty.
    responses = [
        _FakeResponse(choices=[_FakeChoice(_FakeMessage(content=None))], usage=_FakeUsage(5, 3, 8)),
        _FakeResponse(choices=[_FakeChoice(_FakeMessage(content=None))], usage=_FakeUsage(5, 3, 8)),
        _FakeResponse(choices=[_FakeChoice(_FakeMessage(content=None))], usage=_FakeUsage(5, 3, 8)),
    ]

    async def fake_create(**kwargs):
        return responses.pop(0)

    monkeypatch.setattr(client._client.chat.completions, "create", fake_create)
    with pytest.raises(ValueError, match="both native response_format and the unconstrained"):
        await client.structured_complete([{"role": "user", "content": "classify"}], _SampleSchema, label="trader")


async def test_structured_complete_falls_back_on_native_request_error(client, monkeypatch):
    """Real incident (2026-07-28): the native response_format request itself
    raised an HTTP 500 (via the openai SDK's APIStatusError) — not a bad
    response, a failed request — and the fallback's differently-shaped
    request succeeded."""
    request = httpx.Request("POST", "http://test")
    native_error = APIStatusError(
        "Internal Server Error", response=httpx.Response(500, request=request), body=None
    )
    calls = []

    async def fake_create(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise native_error
        return _FakeResponse(choices=[_FakeChoice(_FakeMessage(content='{"value": "buy"}'))], usage=_FakeUsage(5, 3, 8))

    monkeypatch.setattr(client._client.chat.completions, "create", fake_create)
    result = await client.structured_complete([{"role": "user", "content": "classify"}], _SampleSchema)

    assert result.value == "buy"
    assert len(calls) == 2
    assert "response_format" not in calls[1]


async def test_structured_complete_retries_fallback_on_request_error_then_succeeds(client, monkeypatch):
    request = httpx.Request("POST", "http://test")
    native_error = APIConnectionError(request=request)
    fallback_error = APIStatusError(
        "Internal Server Error", response=httpx.Response(500, request=request), body=None
    )
    calls = []

    async def fake_create(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise native_error
        if len(calls) == 2:
            raise fallback_error  # fallback attempt 1/2 also fails
        return _FakeResponse(choices=[_FakeChoice(_FakeMessage(content='{"value": "hold"}'))], usage=_FakeUsage(5, 3, 8))

    monkeypatch.setattr(client._client.chat.completions, "create", fake_create)
    result = await client.structured_complete([{"role": "user", "content": "classify"}], _SampleSchema)

    assert result.value == "hold"
    assert len(calls) == 3  # native + 2 fallback attempts


async def test_structured_complete_raises_when_fallback_request_fails_every_attempt(client, monkeypatch):
    request = httpx.Request("POST", "http://test")

    async def fake_create(**kwargs):
        raise APIStatusError("Internal Server Error", response=httpx.Response(500, request=request), body=None)

    monkeypatch.setattr(client._client.chat.completions, "create", fake_create)
    with pytest.raises(ValueError, match="both native response_format and the unconstrained"):
        await client.structured_complete([{"role": "user", "content": "classify"}], _SampleSchema)


async def test_record_metrics_handles_missing_usage(client, monkeypatch):
    async def fake_create(**kwargs):
        return _FakeResponse(choices=[_FakeChoice(_FakeMessage(content="ok"))], usage=None)

    monkeypatch.setattr(client._client.chat.completions, "create", fake_create)
    await client.complete([{"role": "user", "content": "hi"}], label="x")
    recorded = metrics.recent(1)[0]
    assert recorded.total_tokens is None
    assert recorded.prompt_tokens is None


# ---------------------------------------------------------------------------
# _looks_degenerate / complete()'s degenerate-repetition retry — 2026-07-30:
# greedy decoding (temperature=0.0 everywhere in this project) has no
# built-in escape from a stuck repeating pattern. Real incident: a
# fundamentals_analyst run for 042660.KS came back as the same one-sentence
# paragraph pasted ~100 times, with zero actual financial content.
# ---------------------------------------------------------------------------


def test_looks_degenerate_true_for_many_repeated_long_paragraphs():
    paragraph = "한화오션은 조선업 경기 회복과 수주 호조에 힘입어 실적이 크게 개선되고 있습니다."
    content = "\n\n".join([paragraph] * 5)
    assert _looks_degenerate(content) is True


def test_looks_degenerate_false_for_normal_varied_report():
    content = "\n\n".join([
        "1. 재무 건전성: 매출 299,332억원으로 성장했습니다.",
        "2. 밸류에이션: PER 12.03배, PBR 1.96배입니다.",
        "3. 결론: 매수 의견을 유지합니다.",
    ])
    assert _looks_degenerate(content) is False


def test_looks_degenerate_false_for_short_repeated_bullet_markers():
    # Short repeated fragments ("- ", "**Buy**") are normal markdown/bullet
    # structure, not a decoding loop — only long paragraphs count.
    content = "\n\n".join(["- Buy"] * 10)
    assert _looks_degenerate(content) is False


def test_looks_degenerate_false_below_the_repeat_threshold():
    paragraph = "한화오션은 조선업 경기 회복과 수주 호조에 힘입어 실적이 크게 개선되고 있습니다."
    content = "\n\n".join([paragraph] * 3)  # below _DEGENERATE_MIN_REPEATS (4)
    assert _looks_degenerate(content) is False


def test_looks_degenerate_true_for_repeated_single_lines_without_blank_lines():
    # 2026-08-26: split on every newline, not just "\n\n" — a loop that
    # repeats one line back-to-back (no blank line between repeats) used to
    # go completely undetected.
    line = "한화오션은 조선업 경기 회복과 수주 호조에 힘입어 실적이 크게 개선되고 있습니다."
    content = "\n".join([line] * 5)
    assert _looks_degenerate(content) is True


def test_looks_degenerate_false_for_a_normal_bulleted_report():
    lines = [f"{i}. 항목 {i}에 대한 서로 다른 분석 내용이 여기에 들어갑니다." for i in range(20)]
    assert _looks_degenerate("\n".join(lines)) is False


def test_looks_degenerate_false_for_short_repeated_table_rows():
    # Short repeated lines (under _DEGENERATE_MIN_PARAGRAPH_LEN) are normal
    # table/markdown structure, not a decoding loop.
    content = "\n".join(["| - | - |"] * 10)
    assert _looks_degenerate(content) is False


# 2026-09-16: live decision id=1514 (Fundamentals Analyst, 379800.KS) came
# back as a 103,087-char run-on word chain with only two newlines in the
# entire runaway section — the paragraph check above never had a "block" to
# compare, so it missed a report that actually did lock into a repeating
# ~76-word cycle partway through. See _looks_degenerate's docstring.


def test_looks_degenerate_true_for_repeating_word_cycle_with_no_newlines():
    # Same shape as id=1514's tail: a run-on chain of distinct words that
    # eventually locks into an exact repeating cycle, all on one line.
    prefix = "widening vastifying extending comprehensivizing thoroughfying exhaustivizing meticulously "
    cycle = "spreading widening vastifying extending comprehensivizing thoroughfying exhaustivizing meticulously careful cautious prudent judicious wise intelligent smart clever ingenious inventive creative innovative imaginative "
    content = prefix + (cycle * 5)
    assert _looks_degenerate(content) is True


def test_looks_degenerate_false_for_a_long_run_on_report_with_no_repeats():
    # A long single-paragraph (no newline) report where every word run is
    # distinct must not be flagged — only an actual repeating cycle should be.
    words = [f"항목{i}에대한서로다른내용" for i in range(400)]
    assert _looks_degenerate(" ".join(words)) is False


def test_looks_degenerate_false_for_normal_length_run_on_report():
    # Shorter than the n-gram window's minimum word count entirely — must
    # short-circuit to False rather than erroring on an empty/short scan.
    content = "이 보고서는 짧고 정상적인 내용만 담고 있습니다 반복되는 구문은 없습니다"
    assert _looks_degenerate(content) is False


async def test_complete_retries_at_higher_temperature_when_output_is_degenerate(client, monkeypatch):
    paragraph = "한화오션은 조선업 경기 회복과 수주 호조에 힘입어 실적이 크게 개선되고 있습니다."
    degenerate_content = "\n\n".join([paragraph] * 5)
    calls = []

    async def fake_create(**kwargs):
        calls.append(kwargs["temperature"])
        if len(calls) == 1:
            return _FakeResponse(choices=[_FakeChoice(_FakeMessage(content=degenerate_content))], usage=_FakeUsage(1, 1, 2))
        return _FakeResponse(choices=[_FakeChoice(_FakeMessage(content="정상적인 분석 결과입니다."))], usage=_FakeUsage(1, 1, 2))

    monkeypatch.setattr(client._client.chat.completions, "create", fake_create)
    result = await client.complete([{"role": "user", "content": "hi"}], label="test_agent")

    assert len(calls) == 2
    assert calls[0] == 0.0
    assert calls[1] == _DEGENERATE_RETRY_TEMPERATURE
    assert result.content == "정상적인 분석 결과입니다."


async def test_complete_does_not_retry_normal_output(client, monkeypatch):
    calls = []

    async def fake_create(**kwargs):
        calls.append(1)
        return _FakeResponse(choices=[_FakeChoice(_FakeMessage(content="정상적인 분석 결과입니다."))], usage=_FakeUsage(1, 1, 2))

    monkeypatch.setattr(client._client.chat.completions, "create", fake_create)
    await client.complete([{"role": "user", "content": "hi"}], label="test_agent")
    assert len(calls) == 1


# ---------------------------------------------------------------------------
# 2026-08-26: complete() used to silently return "" on an empty response —
# unlike structured_complete, which already treats empty content as an
# explicit failure. A blank analyst report used to pass
# app.orchestrator.pipeline's degraded-report check and get cached as
# "clean" for its full TTL.
# ---------------------------------------------------------------------------


async def test_complete_retries_when_content_is_empty(client, monkeypatch):
    calls = []

    async def fake_create(**kwargs):
        calls.append(kwargs["temperature"])
        content = "" if len(calls) == 1 else "ok"
        return _FakeResponse(choices=[_FakeChoice(_FakeMessage(content=content))], usage=_FakeUsage(1, 1, 2))

    monkeypatch.setattr(client._client.chat.completions, "create", fake_create)
    result = await client.complete([{"role": "user", "content": "hi"}], label="test_agent")

    assert len(calls) == 2
    assert calls[1] == _DEGENERATE_RETRY_TEMPERATURE
    assert result.content == "ok"


async def test_complete_retries_when_content_is_whitespace_only(client, monkeypatch):
    calls = []

    async def fake_create(**kwargs):
        calls.append(1)
        content = "   \n\n  " if len(calls) == 1 else "ok"
        return _FakeResponse(choices=[_FakeChoice(_FakeMessage(content=content))], usage=_FakeUsage(1, 1, 2))

    monkeypatch.setattr(client._client.chat.completions, "create", fake_create)
    result = await client.complete([{"role": "user", "content": "hi"}], label="test_agent")
    assert len(calls) == 2
    assert result.content == "ok"


async def test_complete_raises_when_content_is_empty_twice(client, monkeypatch):
    async def fake_create(**kwargs):
        return _FakeResponse(
            choices=[_FakeChoice(_FakeMessage(content=""), finish_reason="length")], usage=_FakeUsage(1, 1, 2)
        )

    monkeypatch.setattr(client._client.chat.completions, "create", fake_create)
    with pytest.raises(ValueError, match="test_agent"):
        await client.complete([{"role": "user", "content": "hi"}], label="test_agent")


async def test_complete_does_not_treat_a_tool_call_turn_as_empty(client, monkeypatch):
    calls = []

    async def fake_create(**kwargs):
        calls.append(1)
        return _FakeResponse(choices=[_FakeChoice(_FakeMessage(
            content="", tool_calls=[_FakeToolCall("call1", "get_price", '{"ticker": "AAPL"}')],
        ))], usage=_FakeUsage(1, 1, 2))

    monkeypatch.setattr(client._client.chat.completions, "create", fake_create)
    result = await client.complete([{"role": "user", "content": "hi"}], label="test_agent")
    assert len(calls) == 1
    assert result.tool_calls


async def test_complete_holds_one_limiter_slot_across_the_empty_retry(client, monkeypatch):
    slot_states = []

    async def fake_create(**kwargs):
        slot_states.append(client._limiter._in_flight)
        content = "" if len(slot_states) == 1 else "ok"
        return _FakeResponse(choices=[_FakeChoice(_FakeMessage(content=content))], usage=_FakeUsage(1, 1, 2))

    monkeypatch.setattr(client._client.chat.completions, "create", fake_create)
    await client.complete([{"role": "user", "content": "hi"}], label="test_agent")
    assert slot_states == [1, 1]  # same single acquire covers both attempts
    assert client._limiter._in_flight == 0  # released after


# ---------------------------------------------------------------------------
# 2026-08-26: complete()'s empty-choices / malformed-tool-call-JSON guards.
# ---------------------------------------------------------------------------


async def test_complete_raises_on_empty_choices(client, monkeypatch):
    async def fake_create(**kwargs):
        return _FakeResponse(choices=[], usage=_FakeUsage(1, 1, 2))

    monkeypatch.setattr(client._client.chat.completions, "create", fake_create)
    with pytest.raises(ValueError, match="no choices"):
        await client.complete([{"role": "user", "content": "hi"}], label="test_agent")


async def test_complete_skips_tool_call_with_malformed_arguments_json(client, monkeypatch):
    async def fake_create(**kwargs):
        return _FakeResponse(choices=[_FakeChoice(_FakeMessage(
            content="",
            tool_calls=[
                _FakeToolCall("bad", "get_price", '{"ticker": '),  # malformed
                _FakeToolCall("good", "get_price", '{"ticker": "AAPL"}'),
            ],
        ))], usage=_FakeUsage(1, 1, 2))

    monkeypatch.setattr(client._client.chat.completions, "create", fake_create)
    result = await client.complete([{"role": "user", "content": "hi"}], label="test_agent")
    assert result.tool_calls == [{"id": "good", "name": "get_price", "arguments": {"ticker": "AAPL"}}]


async def test_complete_with_all_tool_calls_malformed_falls_through_to_empty_content_guard(client, monkeypatch):
    calls = []

    async def fake_create(**kwargs):
        calls.append(1)
        if len(calls) == 1:
            return _FakeResponse(choices=[_FakeChoice(_FakeMessage(
                content="", tool_calls=[_FakeToolCall("bad", "get_price", "not json")],
            ))], usage=_FakeUsage(1, 1, 2))
        return _FakeResponse(choices=[_FakeChoice(_FakeMessage(content="ok"))], usage=_FakeUsage(1, 1, 2))

    monkeypatch.setattr(client._client.chat.completions, "create", fake_create)
    result = await client.complete([{"role": "user", "content": "hi"}], label="test_agent")
    assert len(calls) == 2  # the malformed-only tool_calls response triggered the empty-content retry
    assert result.content == "ok"


# ---------------------------------------------------------------------------
# 2026-08-26: degenerate/hanja retries must be soft — if the retry attempt
# itself fails, keep the original (already-valid) result rather than losing
# it, same doctrine as structured_complete's hanja retry.
# ---------------------------------------------------------------------------


async def test_complete_keeps_original_when_degenerate_retry_itself_fails(client, monkeypatch):
    paragraph = "한화오션은 조선업 경기 회복과 수주 호조에 힘입어 실적이 크게 개선되고 있습니다."
    degenerate_content = "\n\n".join([paragraph] * 5)
    calls = []

    async def fake_create(**kwargs):
        calls.append(1)
        if len(calls) == 1:
            return _FakeResponse(choices=[_FakeChoice(_FakeMessage(content=degenerate_content))], usage=_FakeUsage(1, 1, 2))
        raise APIConnectionError(request=httpx.Request("POST", "http://x"))

    monkeypatch.setattr(client._client.chat.completions, "create", fake_create)
    result = await client.complete([{"role": "user", "content": "hi"}], label="test_agent")
    assert len(calls) == 2
    assert result.content == degenerate_content  # original kept, not lost


async def test_complete_keeps_original_when_hanja_retry_itself_fails(client, monkeypatch):
    contaminated = "현재 가격대는 价位 근처입니다."
    calls = []

    async def fake_create(**kwargs):
        calls.append(1)
        if len(calls) == 1:
            return _FakeResponse(choices=[_FakeChoice(_FakeMessage(content=contaminated))], usage=_FakeUsage(1, 1, 2))
        raise APIConnectionError(request=httpx.Request("POST", "http://x"))

    monkeypatch.setattr(client._client.chat.completions, "create", fake_create)
    result = await client.complete([{"role": "user", "content": "hi"}], label="test_agent", language="ko")
    assert len(calls) == 2
    assert result.content == contaminated  # original kept, not lost


async def test_complete_does_not_retry_tool_call_response_even_if_content_repeats(client, monkeypatch):
    # A tool-call response's content is normally empty/minimal — the
    # degenerate check is skipped entirely when tool_calls are present so a
    # legitimate tool-call response is never second-guessed on this basis.
    paragraph = "한화오션은 조선업 경기 회복과 수주 호조에 힘입어 실적이 크게 개선되고 있습니다."
    degenerate_content = "\n\n".join([paragraph] * 5)
    calls = []

    async def fake_create(**kwargs):
        calls.append(1)
        return _FakeResponse(choices=[_FakeChoice(_FakeMessage(
            content=degenerate_content, tool_calls=[_FakeToolCall("call1", "get_price", '{"ticker": "AAPL"}')],
        ))], usage=_FakeUsage(1, 1, 2))

    monkeypatch.setattr(client._client.chat.completions, "create", fake_create)
    await client.complete([{"role": "user", "content": "hi"}], label="test_agent")
    assert len(calls) == 1


def test_reasoning_extra_body_includes_repetition_penalty(client):
    extra_body = client._reasoning_extra_body(reasoning=False)
    assert extra_body["repetition_penalty"] == pytest.approx(1.15)


# ---------------------------------------------------------------------------
# _contains_hanja / complete()'s hanja-contamination retry — 2026-08-05:
# live-decision audit found Chinese-character mixing escalating past the
# single-word slips app.llm.language's own anti-hanja examples were added
# for, up to an entire technical-indicators section written in Chinese.
# ---------------------------------------------------------------------------


def test_contains_hanja_true_for_chinese_characters():
    assert _contains_hanja("최적의 투자 판단은 最优입니다.") is True


def test_contains_hanja_false_for_pure_korean():
    assert _contains_hanja("현재 등급은 매수이며 가격대는 안정적입니다.") is False


def test_contains_hanja_false_for_english_and_numbers():
    assert _contains_hanja("RSI(14) is 55.2, MACD is -26984.42, ticker 005930.KS") is False


def test_contains_hanja_true_for_compatibility_ideographs():
    # U+F900-U+FAFF (CJK Compatibility Ideographs) — exactly the block
    # Korean-reading-compatibility hanja live in, missing from the
    # original range before 2026-08-26.
    assert _contains_hanja("豈 test") is True


def test_contains_hanja_true_for_extension_b():
    assert _contains_hanja("\U00020000 test") is True


async def test_complete_retries_when_korean_output_contains_hanja(client, monkeypatch):
    contaminated = "현재 가격대는 价位 근처입니다."
    clean = "현재 가격대는 적정 근처입니다."
    calls = []

    async def fake_create(**kwargs):
        calls.append(kwargs["temperature"])
        content = contaminated if len(calls) == 1 else clean
        return _FakeResponse(choices=[_FakeChoice(_FakeMessage(content=content))], usage=_FakeUsage(1, 1, 2))

    monkeypatch.setattr(client._client.chat.completions, "create", fake_create)
    result = await client.complete([{"role": "user", "content": "hi"}], label="test_agent", language="ko")

    assert len(calls) == 2
    assert calls[0] == 0.0
    assert calls[1] == _DEGENERATE_RETRY_TEMPERATURE
    assert result.content == clean


async def test_complete_does_not_check_hanja_for_english_language(client, monkeypatch):
    # No equivalent "this script shouldn't appear" signal for "en" — a
    # legitimate English answer might reasonably quote a Chinese company
    # name or term, so this check is Korean-only.
    contaminated = "The price target is 价位 near resistance."
    calls = []

    async def fake_create(**kwargs):
        calls.append(1)
        return _FakeResponse(choices=[_FakeChoice(_FakeMessage(content=contaminated))], usage=_FakeUsage(1, 1, 2))

    monkeypatch.setattr(client._client.chat.completions, "create", fake_create)
    await client.complete([{"role": "user", "content": "hi"}], label="test_agent", language="en")
    assert len(calls) == 1


async def test_complete_does_not_retry_tool_call_response_for_hanja(client, monkeypatch):
    contaminated = "현재 가격대는 价位 근처입니다."
    calls = []

    async def fake_create(**kwargs):
        calls.append(1)
        return _FakeResponse(choices=[_FakeChoice(_FakeMessage(
            content=contaminated, tool_calls=[_FakeToolCall("call1", "get_price", '{"ticker": "AAPL"}')],
        ))], usage=_FakeUsage(1, 1, 2))

    monkeypatch.setattr(client._client.chat.completions, "create", fake_create)
    await client.complete([{"role": "user", "content": "hi"}], label="test_agent", language="ko")
    assert len(calls) == 1


# ---------------------------------------------------------------------------
# structured_complete's own hanja-contamination retry — 2026-08-05 follow-up:
# complete() got the runtime hanja detect+retry, but structured_complete's
# native and fallback paths only ever checked schema validity, never
# language, so a structured field (executive_summary etc.) could still come
# back contaminated. See project memory: llm-korean-hanja-mixing-issue.
# ---------------------------------------------------------------------------


async def test_structured_complete_retries_native_when_korean_output_contains_hanja(client, monkeypatch):
    contaminated = '{"value": "가격대는 价位 근처"}'
    clean = '{"value": "가격대는 적정 근처"}'
    calls = []

    async def fake_create(**kwargs):
        calls.append(kwargs["temperature"])
        content = contaminated if len(calls) == 1 else clean
        return _FakeResponse(choices=[_FakeChoice(_FakeMessage(content=content))], usage=_FakeUsage(1, 1, 2))

    monkeypatch.setattr(client._client.chat.completions, "create", fake_create)
    result = await client.structured_complete(
        [{"role": "user", "content": "classify"}], _SampleSchema, label="pm", language="ko"
    )

    assert len(calls) == 2
    assert calls[0] == 0.0
    assert calls[1] == _DEGENERATE_RETRY_TEMPERATURE
    assert result.value == "가격대는 적정 근처"


async def test_structured_complete_checks_hanja_for_english_language_too(client, monkeypatch):
    # 2026-08-19: the hanja retry gate used to short-circuit on
    # `language != "ko"`, which meant an English-language call (now every
    # pipeline call, per app.llm.language.DEFAULT_LANGUAGE) went completely
    # unguarded. That matters because app.state.PortfolioDecision.
    # card_reason_ko is a Korean field embedded inside an otherwise-English
    # response — hanja anywhere in that response is contamination by
    # definition. The gate now checks regardless of `language`.
    contaminated = '{"value": "target 价位"}'
    clean = '{"value": "target price"}'
    responses = [
        _FakeResponse(choices=[_FakeChoice(_FakeMessage(content=contaminated))], usage=_FakeUsage(1, 1, 2)),
        _FakeResponse(choices=[_FakeChoice(_FakeMessage(content=clean))], usage=_FakeUsage(1, 1, 2)),
    ]
    calls = []

    async def fake_create(**kwargs):
        calls.append(kwargs["temperature"])
        return responses.pop(0)

    monkeypatch.setattr(client._client.chat.completions, "create", fake_create)
    result = await client.structured_complete(
        [{"role": "user", "content": "classify"}], _SampleSchema, label="pm", language="en"
    )

    assert len(calls) == 2  # the hanja retry fired even though language="en"
    assert result.value == "target price"


async def test_structured_complete_keeps_original_when_hanja_retry_fails_to_validate(client, monkeypatch, caplog):
    contaminated = '{"value": "가격대는 价位 근처"}'
    invalid = "{}"  # missing required field on retry
    calls = []

    async def fake_create(**kwargs):
        calls.append(kwargs["temperature"])
        content = contaminated if len(calls) == 1 else invalid
        return _FakeResponse(choices=[_FakeChoice(_FakeMessage(content=content))], usage=_FakeUsage(1, 1, 2))

    monkeypatch.setattr(client._client.chat.completions, "create", fake_create)
    with caplog.at_level("WARNING"):
        result = await client.structured_complete(
            [{"role": "user", "content": "classify"}], _SampleSchema, label="pm", language="ko"
        )

    assert len(calls) == 2
    assert result.value == "가격대는 价位 근처"  # kept the original, already-valid result
    assert any("hanja retry attempt itself failed" in r.message for r in caplog.records)


async def test_structured_complete_retries_fallback_when_korean_output_contains_hanja(client, monkeypatch):
    # native returns empty content -> falls back; fallback's first attempt
    # validates but is hanja-contaminated -> one extra hanja retry call.
    contaminated = '{"value": "가격대는 价位 근처"}'
    clean = '{"value": "가격대는 적정 근처"}'
    responses = [
        _FakeResponse(choices=[_FakeChoice(_FakeMessage(content=None))], usage=_FakeUsage(1, 1, 2)),  # native: empty
        _FakeResponse(choices=[_FakeChoice(_FakeMessage(content=contaminated))], usage=_FakeUsage(1, 1, 2)),  # fallback attempt 1
        _FakeResponse(choices=[_FakeChoice(_FakeMessage(content=clean))], usage=_FakeUsage(1, 1, 2)),  # hanja retry
    ]
    calls = []

    async def fake_create(**kwargs):
        calls.append(kwargs)
        return responses.pop(0)

    monkeypatch.setattr(client._client.chat.completions, "create", fake_create)
    result = await client.structured_complete(
        [{"role": "user", "content": "classify"}], _SampleSchema, label="pm", language="ko"
    )

    assert len(calls) == 3
    assert result.value == "가격대는 적정 근처"


# ---------------------------------------------------------------------------
# _extract_json — 2026-08-26: a second guided-decoding artifact observed live
# against nvidia/Qwen3.6-35B-A3B-NVFP4 (doubled leading '{'), on top of the
# pre-existing markdown-wrapper stripping. See app/llm/client.py docstring.
# ---------------------------------------------------------------------------


def test_extract_json_noop_for_already_clean_json():
    assert _extract_json('{"a": 1}') == '{"a": 1}'


def test_extract_json_strips_markdown_wrapper_noise():
    assert _extract_json('\n\n#{"a": 1}') == '{"a": 1}'
    assert _extract_json('\n\n**{"a": 1}**') == '{"a": 1}'


def test_extract_json_fixes_doubled_leading_brace():
    raw = '\n\n{{"action":"Hold","reasoning":"test"}'
    assert _extract_json(raw) == '{"action":"Hold","reasoning":"test"}'


def test_extract_json_leaves_unfixable_imbalance_untouched():
    # opens=2, closes=0 — count-based trimming would strip 2 leading '{'
    # and leave a non-JSON fragment, so the parse-and-verify step must
    # reject the trim and fall through to returning the original string.
    raw = '{{"action":"Hold", "reasoning": "unterminated'
    assert _extract_json(raw) == raw


def test_extract_json_noop_for_balanced_nested_object():
    raw = '{"a": {"nested": 1}}'
    assert _extract_json(raw) == raw


# ---------------------------------------------------------------------------
# PriorityLimiter wiring (ai-engine-24h-operation-analysis P0-1 ③, 2026-08-11)
# ---------------------------------------------------------------------------


async def test_complete_acquires_and_releases_the_priority_limiter(client, monkeypatch):
    events = []
    real_acquire = client._limiter.acquire
    real_release = client._limiter.release

    async def tracking_acquire(priority):
        events.append(("acquire", priority))
        await real_acquire(priority)

    async def tracking_release():
        events.append(("release",))
        await real_release()

    monkeypatch.setattr(client._limiter, "acquire", tracking_acquire)
    monkeypatch.setattr(client._limiter, "release", tracking_release)

    async def fake_create(**kwargs):
        return _FakeResponse(choices=[_FakeChoice(_FakeMessage(content="ok"))], usage=_FakeUsage(1, 1, 2))

    monkeypatch.setattr(client._client.chat.completions, "create", fake_create)
    await client.complete([{"role": "user", "content": "hi"}])

    assert events == [("acquire", "high"), ("release",)]  # live by default — no backtest_scope active


async def test_complete_uses_low_priority_inside_backtest_scope(client, monkeypatch):
    from app.config import backtest_scope

    seen_priority = []
    real_acquire = client._limiter.acquire

    async def tracking_acquire(priority):
        seen_priority.append(priority)
        await real_acquire(priority)

    monkeypatch.setattr(client._limiter, "acquire", tracking_acquire)

    async def fake_create(**kwargs):
        return _FakeResponse(choices=[_FakeChoice(_FakeMessage(content="ok"))], usage=_FakeUsage(1, 1, 2))

    monkeypatch.setattr(client._client.chat.completions, "create", fake_create)
    with backtest_scope():
        await client.complete([{"role": "user", "content": "hi"}])

    assert seen_priority == ["low"]


async def test_structured_complete_releases_the_limiter_even_when_it_raises(client, monkeypatch):
    # Both the native call and the fallback fail to validate -> re-raises
    # (structured_complete's own fail-loud contract) — the limiter slot
    # must still be released, or one bad response permanently eats a slot.
    async def fake_create(**kwargs):
        return _FakeResponse(choices=[_FakeChoice(_FakeMessage(content="not json"))], usage=_FakeUsage(1, 1, 2))

    monkeypatch.setattr(client._client.chat.completions, "create", fake_create)

    with pytest.raises(Exception):
        await client.structured_complete([{"role": "user", "content": "classify"}], _SampleSchema)

    assert client._limiter._in_flight == 0


async def test_complete_calls_hold_the_slot_for_the_whole_retry_not_per_attempt(client, monkeypatch):
    # complete()'s internal degenerate/hanja retries are one logical call —
    # acquire/release must bracket the whole thing once, not fire per
    # underlying chat.completions.create attempt.
    paragraph = "한화오션은 조선업 경기 회복과 수주 호조에 힘입어 실적이 크게 개선되고 있습니다."
    degenerate_content = "\n\n".join([paragraph] * 5)
    in_flight_per_attempt = []

    async def fake_create(**kwargs):
        in_flight_per_attempt.append(client._limiter._in_flight)
        if len(in_flight_per_attempt) == 1:
            return _FakeResponse(choices=[_FakeChoice(_FakeMessage(content=degenerate_content))], usage=_FakeUsage(1, 1, 2))
        return _FakeResponse(choices=[_FakeChoice(_FakeMessage(content="정상적인 분석 결과입니다."))], usage=_FakeUsage(1, 1, 2))

    monkeypatch.setattr(client._client.chat.completions, "create", fake_create)
    await client.complete([{"role": "user", "content": "hi"}], label="test_agent")

    # Both attempts saw exactly one slot held (never released and
    # re-acquired between the two internal calls).
    assert in_flight_per_attempt == [1, 1]
    assert client._limiter._in_flight == 0  # released once the whole call finished

# ---------------------------------------------------------------------------
# "gateway" provider path (openrouter-integration-idea-2026-08-27) — an
# OpenAI-compatible proxy (reference: OmniRoute). The self-hosted "vllm"
# path is untouched; only endpoint resolution and the per-vendor
# reasoning/sampling seam differ.
# ---------------------------------------------------------------------------


@pytest.fixture
def gateway_client(monkeypatch):
    monkeypatch.setattr(llm_client_module.settings, "llm_provider", "gateway")
    monkeypatch.setattr(llm_client_module.settings, "llm_api_key", "test-key")
    monkeypatch.setattr(llm_client_module.settings, "llm_gateway_base_url", "http://localhost:20128/v1")
    monkeypatch.setattr(llm_client_module.settings, "llm_gateway_reasoning_effort", "low")
    return LLMClient()


def test_resolve_endpoint_vllm_default():
    base_url, api_key, max_retries = LLMClient._resolve_endpoint()
    assert api_key == "not-needed"
    assert max_retries == 5


def test_resolve_endpoint_gateway(monkeypatch):
    monkeypatch.setattr(llm_client_module.settings, "llm_provider", "gateway")
    monkeypatch.setattr(llm_client_module.settings, "llm_api_key", "gw-xxx")
    monkeypatch.setattr(llm_client_module.settings, "llm_gateway_base_url", "http://localhost:20128/v1")
    base_url, api_key, max_retries = LLMClient._resolve_endpoint()
    assert base_url == "http://localhost:20128/v1"
    assert api_key == "gw-xxx"
    assert max_retries == 2


def test_gateway_client_uses_configured_base_url_and_key(gateway_client):
    assert str(gateway_client._client.base_url).rstrip("/") == "http://localhost:20128/v1"
    assert gateway_client._client.api_key == "test-key"
    assert gateway_client._client.max_retries == 2


def test_gateway_reasoning_extra_body_sends_nothing_when_reasoning_off(gateway_client):
    body = gateway_client._reasoning_extra_body(reasoning=False)
    assert body == {}
    assert "chat_template_kwargs" not in body
    assert "repetition_penalty" not in body


def test_gateway_reasoning_extra_body_sends_effort_when_reasoning_on(gateway_client):
    body = gateway_client._reasoning_extra_body(reasoning=True)
    assert body == {"reasoning_effort": "low"}


def test_vllm_reasoning_extra_body_includes_effort_when_reasoning_on(client):
    # 2026-08-31: measured against a real Research-Manager-sized prompt on the
    # new reasoning model (Qwen3.8-Flash-Next-NVFP4, SGLang) — without
    # reasoning_effort, a reasoning=True call burns all of max_tokens=16000 on
    # hidden reasoning_content and returns a 0-char body (388.8s,
    # finish_reason=length), which cascades into a structured_complete
    # validation failure and an identically-empty fallback. Adding
    # reasoning_effort="low" cut this to 60.2s with a real 6095-char body
    # (finish_reason=stop). Sent unconditionally (no per-model branch)
    # because the still-deployed vLLM model (Qwen3.6) was confirmed to
    # accept and silently ignore this field with no behavior change.
    body = client._reasoning_extra_body(reasoning=True)
    assert body["chat_template_kwargs"] == {"enable_thinking": True}
    assert body["repetition_penalty"] == pytest.approx(1.15)
    assert body["reasoning_effort"] == "low"


def test_vllm_reasoning_extra_body_omits_effort_when_reasoning_off(client):
    # reasoning=False already disables thinking via enable_thinking; there is
    # no empty-body failure mode to guard against, so reasoning_effort is not
    # sent (avoids an unnecessary change to the request shape).
    body = client._reasoning_extra_body(reasoning=False)
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert "reasoning_effort" not in body


async def test_complete_captures_gateway_reasoning_field(gateway_client, monkeypatch):
    class _MsgWithReasoning:
        content = "answer"
        tool_calls = []
        reasoning = "chain of thought"

    async def fake_create(**kwargs):
        return _FakeResponse(choices=[_FakeChoice(_MsgWithReasoning())], usage=_FakeUsage(1, 1, 2))

    monkeypatch.setattr(gateway_client._client.chat.completions, "create", fake_create)
    result = await gateway_client.complete([{"role": "user", "content": "hi"}], label="t")
    assert result.reasoning_content == "chain of thought"


async def test_structured_complete_reraises_on_rate_limit_without_fallback(gateway_client, monkeypatch):
    calls = []

    async def fake_create(**kwargs):
        calls.append(kwargs)
        raise APIStatusError(
            "rate limited",
            response=httpx.Response(429, request=httpx.Request("POST", "http://x")),
            body=None,
        )

    monkeypatch.setattr(gateway_client._client.chat.completions, "create", fake_create)
    with pytest.raises(APIStatusError):
        await gateway_client.structured_complete([{"role": "user", "content": "c"}], _SampleSchema)
    # Only the native attempt — no unconstrained fallback requests fired.
    assert len(calls) == 1
    assert gateway_client._limiter._in_flight == 0


# ---------------- reasoning-leak sanitizer (2026-09-01, id=1156) ----------------

from app.llm.client import _MIN_SANITIZED_BODY_CHARS, _strip_reasoning_leak  # noqa: E402

_LONG = "This is a genuine finished argument that is comfortably over the minimum body length. " * 4


def test_strip_reasoning_leak_removes_a_well_formed_block():
    body, kind = _strip_reasoning_leak(f"<think>lots of deliberation here</think>\n\n{_LONG}", label="t")
    assert kind == "think_block"
    assert "<think>" not in body and "deliberation" not in body
    assert _LONG.strip() in body


def test_strip_reasoning_leak_two_blocks_in_one_response():
    body, kind = _strip_reasoning_leak(f"<think>a</think>{_LONG}<think>b</think>{_LONG}", label="t")
    assert kind == "think_block"
    assert "<think>" not in body


def test_strip_reasoning_leak_stray_trailing_close_tag_id1156_shape():
    leak = "Wait, let me reconsider. " * 40
    body, kind = _strip_reasoning_leak(f"{leak}</think>\n\n{_LONG}", label="risk_conservative")
    assert kind == "through_close_tag"
    assert "Wait, let me reconsider" not in body
    assert _LONG.strip() in body


def test_strip_reasoning_leak_unclosed_open_tag():
    body, kind = _strip_reasoning_leak(f"{_LONG}\n<think>and now I ramble forever with no close", label="t")
    assert kind == "open_to_end"
    assert "I ramble forever" not in body
    assert _LONG.strip() in body


def test_strip_reasoning_leak_clean_response_is_byte_identical():
    text = f"{_LONG} It contains < and > and 50% and the word wait once, plus **bold**."
    assert _strip_reasoning_leak(text, label="t") == (text, None)


def test_strip_reasoning_leak_entirely_reasoning_returns_original_declined():
    text = "<think>" + ("only reasoning here " * 30) + "</think>"
    body, kind = _strip_reasoning_leak(text, label="t")
    assert body == text  # ORIGINAL, unchanged — the 2026-08-31 regression guard
    assert kind == "declined_too_short"


def test_strip_reasoning_leak_invariant_never_empties_a_nonempty_body():
    for text in (
        "<think>x</think>", "<think>unclosed", "answer</think>", "  ",
        f"<think>{'y' * 500}</think>", f"<think>a</think>{_LONG}", "< think >z</ think >",
    ):
        out = _strip_reasoning_leak(text, label="t")[0]
        if text.strip():
            assert out.strip(), text


def test_strip_reasoning_leak_untagged_is_detect_only():
    frag = "Wait, let me re-read the prompt. My role is to be the Conservative debater. " + _LONG
    body, kind = _strip_reasoning_leak(frag, label="t")
    assert body == frag  # unchanged
    assert kind == "untagged_suspected"
    # one marker only -> not flagged
    assert _strip_reasoning_leak("Actually, wait — " + _LONG, label="t")[1] is None


def test_strip_reasoning_leak_disabled_by_flag(monkeypatch):
    monkeypatch.setattr(llm_client_module.settings, "llm_strip_reasoning_leak", False)
    text = f"<think>x</think>{_LONG}"
    assert _strip_reasoning_leak(text, label="t") == (text, None)


async def test_complete_strips_a_reasoning_leak_and_records_it(client, monkeypatch):
    with metrics.track_run_sanitizations() as box:
        async def fake_create(**kwargs):
            return _FakeResponse(
                choices=[_FakeChoice(_FakeMessage(content=f"<think>secret</think>\n\n{_LONG}"))],
                usage=_FakeUsage(1, 1, 2),
            )

        monkeypatch.setattr(client._client.chat.completions, "create", fake_create)
        result = await client.complete([{"role": "user", "content": "hi"}], label="risk_conservative")

    assert "<think>" not in result.content
    assert box == ["risk_conservative: think_block"]


async def test_complete_empty_response_guard_unchanged_by_sanitizer(client, monkeypatch):
    # A body that is ENTIRELY reasoning: _strip_reasoning_leak returns the
    # original (declined_too_short), so the body stays non-empty and the
    # empty-response guard does NOT fire — exactly as before this change.
    text = "<think>" + ("all reasoning " * 40) + "</think>"

    async def fake_create(**kwargs):
        return _FakeResponse(choices=[_FakeChoice(_FakeMessage(content=text))], usage=_FakeUsage(1, 1, 2))

    monkeypatch.setattr(client._client.chat.completions, "create", fake_create)
    result = await client.complete([{"role": "user", "content": "hi"}], label="t")
    assert result.content == text  # not emptied, no ValueError
