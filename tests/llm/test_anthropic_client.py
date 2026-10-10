"""app.llm.anthropic_client.AnthropicLLMClient — the native `anthropic` SDK
call itself mocked end-to-end (`client._client.messages.stream` swapped for a
fake that never touches the network), verifying the OpenAI<->Anthropic shape
translation and the same retry/fallback/hard-fail doctrines
app.llm.client.LLMClient already has for the vLLM/gateway path.

No test in this file can reach api.anthropic.com: AsyncAnthropic itself is
constructed (cheap, no I/O), but its `.messages.stream` method is always
replaced by `_FakeMessagesResource.stream` before any `complete()`/
`structured_complete()` call runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest
from anthropic import AuthenticationError, RateLimitError
from pydantic import BaseModel

from app.config import settings
from app.llm import anthropic_client as anthropic_client_module
from app.llm import metrics
from app.llm.anthropic_client import AnthropicLLMClient, _AnthropicRefusal


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch):
    # The fallback retry loop's inter-attempt delay would otherwise add real
    # wall-clock seconds to any test exercising a retry.
    async def instant_sleep(seconds):
        return None

    monkeypatch.setattr(anthropic_client_module.asyncio, "sleep", instant_sleep)


@pytest.fixture(autouse=True)
def _clean_metrics():
    metrics.clear()
    yield
    metrics.clear()


@pytest.fixture(autouse=True)
def _anthropic_api_key(monkeypatch):
    # AnthropicLLMClient.__init__ reads settings.anthropic_api_key directly —
    # give it a harmless placeholder so client construction never depends on
    # a real environment variable being set.
    monkeypatch.setattr(settings, "anthropic_api_key", "test-key")


# --- fakes: Anthropic Messages API response shapes, no network -------------


@dataclass
class _FakeTextBlock:
    text: str
    type: str = "text"


@dataclass
class _FakeThinkingBlock:
    thinking: str
    signature: str = "sig"
    type: str = "thinking"


@dataclass
class _FakeToolUseBlock:
    id: str
    name: str
    input: dict
    type: str = "tool_use"


@dataclass
class _FakeUsage:
    input_tokens: int
    output_tokens: int
    cache_read_input_tokens: int | None = None
    cache_creation_input_tokens: int | None = None


@dataclass
class _FakeStopDetails:
    category: str | None = None
    explanation: str | None = None
    type: str = "refusal"


@dataclass
class _FakeMessage:
    content: list = field(default_factory=list)
    usage: _FakeUsage = field(default_factory=lambda: _FakeUsage(10, 5))
    stop_reason: str | None = "end_turn"
    stop_details: _FakeStopDetails | None = None


class _FakeStreamCM:
    """Async context manager standing in for AsyncMessageStreamManager —
    `__aenter__` returns something with an async `get_final_message()`,
    exactly the shape `AnthropicLLMClient._send` drives."""

    def __init__(self, outcome):
        self._outcome = outcome  # a _FakeMessage, or an Exception instance to raise

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False

    async def get_final_message(self):
        if isinstance(self._outcome, BaseException):
            raise self._outcome
        return self._outcome


class _FakeMessagesResource:
    """Stands in for `client._client.messages` — `.stream(**kwargs)` pops the
    next pre-programmed outcome off a queue and records the kwargs it was
    called with, so tests can assert on the translated request shape."""

    def __init__(self, outcomes):
        self._outcomes = list(outcomes)
        self.calls: list[dict] = []

    def stream(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeStreamCM(self._outcomes.pop(0))


def _install(client: AnthropicLLMClient, *outcomes) -> _FakeMessagesResource:
    fake = _FakeMessagesResource(outcomes)
    monkeypatch_messages(client, fake)
    return fake


def monkeypatch_messages(client: AnthropicLLMClient, fake: _FakeMessagesResource) -> None:
    client._client.messages.stream = fake.stream  # type: ignore[method-assign]


@pytest.fixture
def client():
    return AnthropicLLMClient()


class _SampleSchema(BaseModel):
    value: str


# --- system-message splitting -----------------------------------------------


def test_split_system_and_messages_appends_language_instruction():
    messages = [
        {"role": "system", "content": "You are a helpful analyst."},
        {"role": "user", "content": "Analyze AAPL"},
    ]
    system_text, remaining = AnthropicLLMClient._split_system_and_messages(messages, "en")
    assert system_text.startswith("You are a helpful analyst.")
    assert "Always respond in English only" in system_text
    assert "\n\n" in system_text
    assert remaining == [{"role": "user", "content": "Analyze AAPL"}]


def test_split_system_and_messages_no_system_message_is_instruction_alone():
    messages = [{"role": "user", "content": "hi"}]
    system_text, remaining = AnthropicLLMClient._split_system_and_messages(messages, "ko")
    assert "한국어로만 답변" in system_text
    assert remaining == messages


def test_split_system_and_messages_requires_user_first():
    with pytest.raises(ValueError):
        AnthropicLLMClient._split_system_and_messages(
            [{"role": "system", "content": "sys"}, {"role": "assistant", "content": "oops"}], "en"
        )
    with pytest.raises(ValueError):
        AnthropicLLMClient._split_system_and_messages([{"role": "system", "content": "sys"}], "en")


# --- tool schema round-trip --------------------------------------------------


def test_convert_tools_openai_to_anthropic_shape():
    openai_tools = [
        {
            "type": "function",
            "function": {
                "name": "get_price",
                "description": "fetch a price",
                "parameters": {"type": "object", "properties": {"ticker": {"type": "string"}}},
            },
        }
    ]
    converted = AnthropicLLMClient._convert_tools(openai_tools)
    assert converted == [
        {
            "name": "get_price",
            "description": "fetch a price",
            "input_schema": {"type": "object", "properties": {"ticker": {"type": "string"}}},
        }
    ]


def test_convert_tools_empty_is_none():
    assert AnthropicLLMClient._convert_tools(None) is None
    assert AnthropicLLMClient._convert_tools([]) is None


def test_tool_calls_from_content_does_not_json_loads_input(monkeypatch):
    # block.input is ALREADY a parsed dict — patch json.loads to blow up if
    # anything in this path ever calls it, proving the conversion never does.
    import json as json_module

    def _forbidden(*_a, **_kw):
        raise AssertionError("json.loads must never be called on an already-parsed tool_use input")

    monkeypatch.setattr(json_module, "loads", _forbidden)
    block = _FakeToolUseBlock(id="call1", name="get_price", input={"ticker": "AAPL"})
    calls = AnthropicLLMClient._tool_calls_from_content([block])
    assert calls == [{"id": "call1", "name": "get_price", "arguments": {"ticker": "AAPL"}}]


# --- tool_loop history translation (_to_anthropic_messages) -----------------


def test_to_anthropic_messages_two_round_tool_loop():
    messages = [
        {"role": "user", "content": "Analyze AAPL"},
        {
            "role": "assistant",
            "content": "Let me check prices and news.",
            "tool_calls": [
                {"id": "call1", "type": "function", "function": {"name": "get_price", "arguments": '{"ticker": "AAPL"}'}},
                {"id": "call2", "type": "function", "function": {"name": "get_news", "arguments": '{"ticker": "AAPL"}'}},
            ],
        },
        {"role": "tool", "tool_call_id": "call1", "content": "price: 200"},
        {"role": "tool", "tool_call_id": "call2", "content": "news: nothing notable"},
        {
            "role": "assistant",
            "content": "Now checking flow.",
            "tool_calls": [
                {"id": "call3", "type": "function", "function": {"name": "get_flow", "arguments": '{"ticker": "AAPL"}'}},
            ],
        },
        {"role": "tool", "tool_call_id": "call3", "content": "flow: neutral"},
        {"role": "user", "content": "You have reached the tool-call limit for this report."},
    ]
    result = AnthropicLLMClient._to_anthropic_messages(messages)
    assert result == [
        {"role": "user", "content": "Analyze AAPL"},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Let me check prices and news."},
                {"type": "tool_use", "id": "call1", "name": "get_price", "input": {"ticker": "AAPL"}},
                {"type": "tool_use", "id": "call2", "name": "get_news", "input": {"ticker": "AAPL"}},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "call1", "content": "price: 200"},
                {"type": "tool_result", "tool_use_id": "call2", "content": "news: nothing notable"},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Now checking flow."},
                {"type": "tool_use", "id": "call3", "name": "get_flow", "input": {"ticker": "AAPL"}},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "call3", "content": "flow: neutral"},
            ],
        },
        {"role": "user", "content": "You have reached the tool-call limit for this report."},
    ]


def test_to_anthropic_messages_merges_parallel_results_into_one_message():
    messages = [
        {"role": "user", "content": "go"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "a", "type": "function", "function": {"name": "t1", "arguments": "{}"}},
                {"id": "b", "type": "function", "function": {"name": "t2", "arguments": "{}"}},
                {"id": "c", "type": "function", "function": {"name": "t3", "arguments": "{}"}},
            ],
        },
        {"role": "tool", "tool_call_id": "a", "content": "ra"},
        {"role": "tool", "tool_call_id": "b", "content": "rb"},
        {"role": "tool", "tool_call_id": "c", "content": "rc"},
    ]
    result = AnthropicLLMClient._to_anthropic_messages(messages)
    tool_result_messages = [m for m in result if m["role"] == "user" and isinstance(m["content"], list)]
    assert len(tool_result_messages) == 1
    merged = tool_result_messages[0]
    assert len(merged["content"]) == 3
    assert [b["tool_use_id"] for b in merged["content"]] == ["a", "b", "c"]


def test_to_anthropic_messages_empty_assistant_content_has_no_text_block():
    messages = [
        {"role": "user", "content": "go"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "a", "type": "function", "function": {"name": "t1", "arguments": "{}"}},
            ],
        },
    ]
    result = AnthropicLLMClient._to_anthropic_messages(messages)
    assistant_msg = result[1]
    assert assistant_msg["role"] == "assistant"
    assert not any(block.get("type") == "text" for block in assistant_msg["content"])
    assert len(assistant_msg["content"]) == 1
    assert assistant_msg["content"][0]["type"] == "tool_use"


def test_to_anthropic_messages_malformed_arguments_raises():
    messages = [
        {"role": "user", "content": "go"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "a", "type": "function", "function": {"name": "get_price", "arguments": "{not valid json"}},
            ],
        },
    ]
    with pytest.raises(ValueError, match="get_price"):
        AnthropicLLMClient._to_anthropic_messages(messages)


def test_to_anthropic_messages_orphaned_tool_message_raises():
    messages = [
        {"role": "user", "content": "go"},
        {"role": "user", "content": "again"},
        {"role": "tool", "tool_call_id": "orphan", "content": "oops"},
    ]
    with pytest.raises(ValueError):
        AnthropicLLMClient._to_anthropic_messages(messages)


def test_to_anthropic_messages_plain_assistant_passes_through():
    messages = [
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": "here is my plain retry answer"},
        {"role": "user", "content": "please fix the JSON"},
    ]
    result = AnthropicLLMClient._to_anthropic_messages(messages)
    assert result == messages


def test_to_anthropic_messages_plain_user_only_unchanged():
    messages = [
        {"role": "user", "content": "hello"},
    ]
    result = AnthropicLLMClient._to_anthropic_messages(messages)
    assert result == messages


# --- complete() wiring of the translation + force-final stub tools ---------


async def test_complete_translates_tool_history(client):
    tool_history = [
        {"role": "user", "content": "go"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "a", "type": "function", "function": {"name": "get_price", "arguments": '{"ticker": "AAPL"}'}},
            ],
        },
        {"role": "tool", "tool_call_id": "a", "content": "price: 200"},
    ]
    message = _FakeMessage(content=[_FakeTextBlock("final report")], stop_reason="end_turn")
    fake = _install(client, message)
    await client.complete(tool_history, tools=[{"type": "function", "function": {"name": "get_price", "parameters": {}}}], label="t")
    sent_messages = fake.calls[0]["messages"]
    assert sent_messages == [
        {"role": "user", "content": "go"},
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "a", "name": "get_price", "input": {"ticker": "AAPL"}}],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "a", "content": "price: 200"}],
        },
    ]


async def test_complete_echoes_stub_tools_when_history_has_tool_use_and_tools_is_none(client):
    tool_history = [
        {"role": "user", "content": "go"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "a", "type": "function", "function": {"name": "get_price", "arguments": "{}"}},
                {"id": "b", "type": "function", "function": {"name": "get_news", "arguments": "{}"}},
            ],
        },
        {"role": "tool", "tool_call_id": "a", "content": "price: 200"},
        {"role": "tool", "tool_call_id": "b", "content": "news: none"},
        {"role": "user", "content": "You have reached the tool-call limit for this report."},
    ]
    message = _FakeMessage(content=[_FakeTextBlock("final report")], stop_reason="end_turn")
    fake = _install(client, message)
    await client.complete(tool_history, tools=None, label="t")
    sent_kwargs = fake.calls[0]
    assert sent_kwargs["tools"] == [
        {"name": "get_news", "description": "", "input_schema": {"type": "object"}},
        {"name": "get_price", "description": "", "input_schema": {"type": "object"}},
    ]
    assert sent_kwargs["tool_choice"] == {"type": "none"}


async def test_structured_complete_does_not_translate(client, monkeypatch):
    called = False
    original = AnthropicLLMClient._to_anthropic_messages

    def _spy(messages):
        nonlocal called
        called = True
        return original(messages)

    monkeypatch.setattr(AnthropicLLMClient, "_to_anthropic_messages", staticmethod(_spy))
    message = _FakeMessage(content=[_FakeTextBlock('{"value": "ok"}')], stop_reason="end_turn")
    _install(client, message)
    plain_history = [
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": "plain retry"},
        {"role": "user", "content": "again"},
    ]
    result = await client.structured_complete(plain_history, _SampleSchema, label="t")
    assert result.value == "ok"
    assert called is False


# --- stop_reason -> finish_reason mapping -----------------------------------


async def test_complete_maps_max_tokens_stop_reason_to_length(client):
    message = _FakeMessage(content=[_FakeTextBlock("hello")], stop_reason="max_tokens")
    _install(client, message)
    result = await client.complete([{"role": "user", "content": "hi"}], label="t")
    assert result.content == "hello"
    recent = metrics.recent(1)
    assert recent[0].finish_reason == "length"


async def test_complete_passes_through_other_stop_reasons(client):
    message = _FakeMessage(content=[_FakeTextBlock("hello")], stop_reason="end_turn")
    _install(client, message)
    await client.complete([{"role": "user", "content": "hi"}], label="t")
    assert metrics.recent(1)[0].finish_reason == "end_turn"


# --- empty response --------------------------------------------------------


async def test_complete_empty_twice_raises(client):
    empty = _FakeMessage(content=[_FakeTextBlock("")], stop_reason="end_turn")
    _install(client, empty, empty)
    with pytest.raises(ValueError, match="empty content"):
        await client.complete([{"role": "user", "content": "hi"}], label="t")


async def test_complete_empty_then_recovers(client):
    empty = _FakeMessage(content=[_FakeTextBlock("")], stop_reason="end_turn")
    recovered = _FakeMessage(content=[_FakeTextBlock("real content here")], stop_reason="end_turn")
    _install(client, empty, recovered)
    result = await client.complete([{"role": "user", "content": "hi"}], label="t")
    assert result.content == "real content here"


# --- refusal -----------------------------------------------------------------


async def test_complete_hard_fails_on_refusal(client):
    refusal = _FakeMessage(
        content=[], stop_reason="refusal", stop_details=_FakeStopDetails(category="general_harms", explanation="no")
    )
    _install(client, refusal)
    with pytest.raises(_AnthropicRefusal):
        await client.complete([{"role": "user", "content": "hi"}], label="t")


# --- structured_complete: native success ------------------------------------


async def test_structured_complete_native_success(client):
    message = _FakeMessage(content=[_FakeTextBlock('{"value": "ok"}')], stop_reason="end_turn")
    fake = _install(client, message)
    result = await client.structured_complete([{"role": "user", "content": "go"}], _SampleSchema, label="t")
    assert result.value == "ok"
    # native call requested a JSON-schema output format
    assert fake.calls[0]["output_config"]["format"]["type"] == "json_schema"


# --- structured_complete: native failure -> fallback success ----------------


async def test_structured_complete_native_fails_then_fallback_succeeds(client):
    broken = _FakeMessage(content=[_FakeTextBlock("not json at all")], stop_reason="end_turn")
    fixed = _FakeMessage(content=[_FakeTextBlock('{"value": "fixed"}')], stop_reason="end_turn")
    fake = _install(client, broken, fixed)
    result = await client.structured_complete([{"role": "user", "content": "go"}], _SampleSchema, label="t")
    assert result.value == "fixed"
    assert len(fake.calls) == 2
    # fallback call must NOT carry a format constraint
    assert "format" not in fake.calls[1]["output_config"]


async def test_structured_complete_fallback_also_fails_raises(client):
    broken = _FakeMessage(content=[_FakeTextBlock("nope")], stop_reason="end_turn")
    still_broken = _FakeMessage(content=[_FakeTextBlock("still nope")], stop_reason="end_turn")
    _install(client, broken, still_broken, still_broken)  # native + _FALLBACK_MAX_ATTEMPTS(2)
    with pytest.raises(ValueError, match="failed to produce a valid"):
        await client.structured_complete([{"role": "user", "content": "go"}], _SampleSchema, label="t")


# --- quota/auth errors never trigger the fallback ---------------------------


async def test_structured_complete_rate_limit_does_not_fall_back(client):
    import httpx

    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    err = RateLimitError("rate limited", response=httpx.Response(429, request=request), body=None)
    fake = _install(client, err)
    with pytest.raises(RateLimitError):
        await client.structured_complete([{"role": "user", "content": "go"}], _SampleSchema, label="t")
    assert len(fake.calls) == 1  # never attempted the fallback


async def test_structured_complete_auth_error_does_not_fall_back(client):
    import httpx

    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    err = AuthenticationError("bad key", response=httpx.Response(401, request=request), body=None)
    fake = _install(client, err)
    with pytest.raises(AuthenticationError):
        await client.structured_complete([{"role": "user", "content": "go"}], _SampleSchema, label="t")
    assert len(fake.calls) == 1


# --- degenerate / hanja soft retries -----------------------------------------


async def test_complete_degenerate_output_retries_once(client):
    degenerate_paragraph = "This is a long repeated paragraph that keeps coming back. " * 1
    degenerate_text = "\n\n".join([degenerate_paragraph] * 5)
    degenerate = _FakeMessage(content=[_FakeTextBlock(degenerate_text)], stop_reason="end_turn")
    clean = _FakeMessage(content=[_FakeTextBlock("a perfectly normal, non-repeating report")], stop_reason="end_turn")
    _install(client, degenerate, clean)
    result = await client.complete([{"role": "user", "content": "hi"}], label="t")
    assert result.content == "a perfectly normal, non-repeating report"


async def test_complete_degenerate_retry_failure_keeps_original(client):
    degenerate_paragraph = "This is a long repeated paragraph that keeps coming back. " * 1
    degenerate_text = "\n\n".join([degenerate_paragraph] * 5)
    degenerate = _FakeMessage(content=[_FakeTextBlock(degenerate_text)], stop_reason="end_turn")
    _install(client, degenerate, RuntimeError("network blip"))
    result = await client.complete([{"role": "user", "content": "hi"}], label="t")
    assert result.content == degenerate_text  # kept the original, soft retry failure swallowed


async def test_complete_hanja_in_korean_output_retries_once(client):
    hanja_text = "이것은 등급이 评级 로 오염된 문장입니다"
    clean_text = "이것은 순한글로만 작성된 문장입니다"
    _install(
        client,
        _FakeMessage(content=[_FakeTextBlock(hanja_text)], stop_reason="end_turn"),
        _FakeMessage(content=[_FakeTextBlock(clean_text)], stop_reason="end_turn"),
    )
    result = await client.complete([{"role": "user", "content": "hi"}], label="t", language="ko")
    assert result.content == clean_text


async def test_structured_complete_hanja_retries_once_soft(client):
    hanja_json = '{"value": "评级"}'
    clean_json = '{"value": "clean"}'
    _install(
        client,
        _FakeMessage(content=[_FakeTextBlock(hanja_json)], stop_reason="end_turn"),
        _FakeMessage(content=[_FakeTextBlock(clean_json)], stop_reason="end_turn"),
    )
    result = await client.structured_complete([{"role": "user", "content": "go"}], _SampleSchema, label="t")
    assert result.value == "clean"


async def test_structured_complete_hanja_retry_failure_keeps_original(client):
    hanja_json = '{"value": "评级"}'
    _install(client, _FakeMessage(content=[_FakeTextBlock(hanja_json)], stop_reason="end_turn"), RuntimeError("blip"))
    result = await client.structured_complete([{"role": "user", "content": "go"}], _SampleSchema, label="t")
    assert result.value == "评级"  # original kept — hanja retry is soft
