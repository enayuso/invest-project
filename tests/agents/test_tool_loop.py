"""run_tool_loop / bind_as_of / ticker_label — llm_client mocked, no network (C-stage)."""

from __future__ import annotations

import asyncio
from datetime import date

import pytest

from app.agents import tool_loop
from app.config import settings
from app.llm.client import LLMMessage


def test_ticker_label_with_company_name():
    assert tool_loop.ticker_label("373220.KS", "LG Energy Solution") == "LG Energy Solution (373220.KS)"


def test_ticker_label_without_company_name():
    assert tool_loop.ticker_label("AAPL", None) == "AAPL"


async def test_bind_as_of_partially_applies_every_tool():
    calls = []

    async def tool(ticker: str, as_of=None):
        calls.append((ticker, as_of))
        return "ok"

    bound = tool_loop.bind_as_of({"t": tool}, date(2026, 1, 1))
    await bound["t"](ticker="AAPL")
    assert calls == [("AAPL", date(2026, 1, 1))]


class FakeLLM:
    """Scripted sequence of LLMMessage responses, returned in order per call to complete()."""

    def __init__(self, responses: list[LLMMessage]):
        self._responses = list(responses)
        self.calls: list[dict] = []

    async def complete(self, messages, *, tools=None, reasoning=False, temperature=0.0, label="", language="ko", max_tokens=None):
        self.calls.append({"messages": messages, "tools": tools})
        return self._responses.pop(0)


@pytest.fixture(autouse=True)
def _no_iteration_cap_surprises(monkeypatch):
    monkeypatch.setattr(settings, "max_tool_iterations", 4)


async def test_returns_immediately_when_no_tool_calls(monkeypatch):
    fake = FakeLLM([LLMMessage(content="final report, no tools needed")])
    monkeypatch.setattr(tool_loop, "llm_client", fake)
    result = await tool_loop.run_tool_loop(
        system_prompt="sys", user_prompt="user", tools=[], dispatch={}, agent_name="Test Agent",
    )
    assert result == "final report, no tools needed"
    assert len(fake.calls) == 1


async def test_executes_tool_call_and_wraps_untrusted_result(monkeypatch):
    async def get_news(ticker: str) -> str:
        return "breaking news headline"

    fake = FakeLLM([
        LLMMessage(content="", tool_calls=[{"id": "call1", "name": "get_news", "arguments": {"ticker": "AAPL"}}]),
        LLMMessage(content="final report using the news"),
    ])
    monkeypatch.setattr(tool_loop, "llm_client", fake)
    result = await tool_loop.run_tool_loop(
        system_prompt="sys", user_prompt="user", tools=[], dispatch={"get_news": get_news}, agent_name="Test Agent",
    )
    assert result == "final report using the news"
    tool_message = fake.calls[1]["messages"][-1]
    assert tool_message["role"] == "tool"
    assert "<external_data source=\"get_news\">" in tool_message["content"]
    assert "breaking news headline" in tool_message["content"]


async def test_trusted_tool_result_is_not_wrapped(monkeypatch):
    async def get_stock_data(ticker: str) -> str:
        return "OHLCV data"

    fake = FakeLLM([
        LLMMessage(content="", tool_calls=[{"id": "call1", "name": "get_stock_data", "arguments": {"ticker": "AAPL"}}]),
        LLMMessage(content="final report"),
    ])
    monkeypatch.setattr(tool_loop, "llm_client", fake)
    await tool_loop.run_tool_loop(
        system_prompt="sys", user_prompt="user", tools=[], dispatch={"get_stock_data": get_stock_data},
        agent_name="Test Agent", trusted_tools=frozenset({"get_stock_data"}),
    )
    tool_message = fake.calls[1]["messages"][-1]
    assert tool_message["content"] == "OHLCV data"


async def test_unknown_tool_call_reports_error_without_crashing(monkeypatch):
    fake = FakeLLM([
        LLMMessage(content="", tool_calls=[{"id": "call1", "name": "nonexistent_tool", "arguments": {}}]),
        LLMMessage(content="final report"),
    ])
    monkeypatch.setattr(tool_loop, "llm_client", fake)
    result = await tool_loop.run_tool_loop(
        system_prompt="sys", user_prompt="user", tools=[], dispatch={}, agent_name="Test Agent",
    )
    assert result == "final report"
    tool_message = fake.calls[1]["messages"][-1]
    assert tool_message["content"] == "Error: unknown tool 'nonexistent_tool'"


async def test_tool_exception_is_surfaced_to_the_model_not_raised(monkeypatch):
    async def broken_tool(ticker: str) -> str:
        raise RuntimeError("upstream API down")

    fake = FakeLLM([
        LLMMessage(content="", tool_calls=[{"id": "call1", "name": "broken_tool", "arguments": {"ticker": "AAPL"}}]),
        LLMMessage(content="final report despite the error"),
    ])
    monkeypatch.setattr(tool_loop, "llm_client", fake)
    result = await tool_loop.run_tool_loop(
        system_prompt="sys", user_prompt="user", tools=[], dispatch={"broken_tool": broken_tool}, agent_name="Test Agent",
    )
    assert result == "final report despite the error"
    tool_message = fake.calls[1]["messages"][-1]
    assert "Error calling broken_tool: upstream API down" in tool_message["content"]


async def test_spurious_tool_cancellation_degrades_to_an_error_string(monkeypatch):
    """2026-08-25: asyncio.CancelledError subclasses BaseException, not
    Exception, so the old `except Exception` in _call_one never saw it —
    despite this module's own contract that a tool call is always
    error-normalized to a plain string. A cancellation that merely surfaces
    through an unrelated internal wait (not a real outer cancellation of
    THIS task) must degrade the same as any other tool failure."""
    async def spuriously_cancelled_tool(ticker: str) -> str:
        raise asyncio.CancelledError()

    fake = FakeLLM([
        LLMMessage(
            content="", tool_calls=[{"id": "call1", "name": "spuriously_cancelled_tool", "arguments": {"ticker": "AAPL"}}],
        ),
        LLMMessage(content="final report despite the cancellation"),
    ])
    monkeypatch.setattr(tool_loop, "llm_client", fake)
    monkeypatch.setattr(tool_loop, "task_was_actually_cancelled", lambda: False)
    result = await tool_loop.run_tool_loop(
        system_prompt="sys", user_prompt="user", tools=[],
        dispatch={"spuriously_cancelled_tool": spuriously_cancelled_tool}, agent_name="Test Agent",
    )
    assert result == "final report despite the cancellation"
    tool_message = fake.calls[1]["messages"][-1]
    assert "unrequested cancellation" in tool_message["content"]


async def test_spurious_tool_cancellation_does_not_abort_sibling_tool_calls(monkeypatch):
    """The concrete loss the fix prevents: one tool call in a model turn
    hitting a spurious CancelledError used to propagate out of
    asyncio.gather and abort every sibling tool call mid-flight, discarding
    real results the model was waiting on."""
    sibling_ran = False

    async def spuriously_cancelled_tool(ticker: str) -> str:
        raise asyncio.CancelledError()

    async def sibling_tool(ticker: str) -> str:
        nonlocal sibling_ran
        await asyncio.sleep(0.01)
        sibling_ran = True
        return "sibling result"

    fake = FakeLLM([
        LLMMessage(
            content="",
            tool_calls=[
                {"id": "call1", "name": "spuriously_cancelled_tool", "arguments": {"ticker": "AAPL"}},
                {"id": "call2", "name": "sibling_tool", "arguments": {"ticker": "AAPL"}},
            ],
        ),
        LLMMessage(content="final report"),
    ])
    monkeypatch.setattr(tool_loop, "llm_client", fake)
    monkeypatch.setattr(tool_loop, "task_was_actually_cancelled", lambda: False)
    await tool_loop.run_tool_loop(
        system_prompt="sys", user_prompt="user", tools=[],
        dispatch={"spuriously_cancelled_tool": spuriously_cancelled_tool, "sibling_tool": sibling_tool},
        agent_name="Test Agent",
    )
    assert sibling_ran is True
    messages_by_call_id = {m["tool_call_id"]: m["content"] for m in fake.calls[1]["messages"] if m.get("role") == "tool"}
    assert "sibling result" in messages_by_call_id["call2"]


async def test_genuine_cancellation_propagates_out_of_the_tool_loop(monkeypatch):
    """A real outer cancellation (this task actually being told to stop —
    task_was_actually_cancelled() returning True) must not be swallowed as a
    data-quality degradation; it must tear the task down."""
    async def slow_tool(ticker: str) -> str:
        await asyncio.sleep(10)
        return "never reached"

    fake = FakeLLM([
        LLMMessage(content="", tool_calls=[{"id": "call1", "name": "slow_tool", "arguments": {"ticker": "AAPL"}}]),
    ])
    monkeypatch.setattr(tool_loop, "llm_client", fake)

    task = asyncio.ensure_future(tool_loop.run_tool_loop(
        system_prompt="sys", user_prompt="user", tools=[], dispatch={"slow_tool": slow_tool}, agent_name="Test Agent",
    ))
    await asyncio.sleep(0)  # let it start and reach the slow_tool's sleep
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_system_prompt_states_the_tool_call_budget(monkeypatch):
    monkeypatch.setattr(settings, "max_tool_iterations", 4)
    fake = FakeLLM([LLMMessage(content="final report")])
    monkeypatch.setattr(tool_loop, "llm_client", fake)

    await tool_loop.run_tool_loop(
        system_prompt="sys", user_prompt="user", tools=[], dispatch={}, agent_name="Test Agent",
    )

    system_message = fake.calls[0]["messages"][0]["content"]
    assert "up to 4 tool-calling rounds" in system_message


async def test_system_prompt_forbids_speculative_prose_for_a_skipped_tool(monkeypatch):
    # 2026-08-05: a skipped tool (get_investor_flow) previously got papered
    # over with a hypothetical "framework" paragraph instead of being
    # omitted/noted as skipped — the budget note must explicitly rule that
    # out, not just say "skip non-essential tools."
    fake = FakeLLM([LLMMessage(content="final report")])
    monkeypatch.setattr(tool_loop, "llm_client", fake)

    await tool_loop.run_tool_loop(
        system_prompt="sys", user_prompt="user", tools=[], dispatch={}, agent_name="Test Agent",
    )

    system_message = fake.calls[0]["messages"][0]["content"]
    assert "never write a hypothetical or generic paragraph" in system_message
    assert "not checked this cycle" in system_message


async def test_multiple_tool_calls_in_one_turn_run_concurrently(monkeypatch):
    """A single model turn can request several independent tool calls — they
    must not be serialized, since none of them depend on each other's
    result. Proven the same way tests/execution/test_ledger.py's
    market_lock test proves concurrency: each fake tool yields via
    asyncio.sleep(0) before returning, so a sequential for-loop would finish
    them in strict start-order, while asyncio.gather lets a later-started
    call finish before an earlier one if it yields less."""
    order: list[str] = []

    async def slow_tool(ticker: str) -> str:
        order.append("slow-start")
        await asyncio.sleep(0.02)
        order.append("slow-end")
        return "slow result"

    async def fast_tool(ticker: str) -> str:
        order.append("fast-start")
        order.append("fast-end")
        return "fast result"

    fake = FakeLLM([
        LLMMessage(
            content="",
            tool_calls=[
                {"id": "c1", "name": "slow_tool", "arguments": {"ticker": "AAPL"}},
                {"id": "c2", "name": "fast_tool", "arguments": {"ticker": "AAPL"}},
            ],
        ),
        LLMMessage(content="final report"),
    ])
    monkeypatch.setattr(tool_loop, "llm_client", fake)

    await tool_loop.run_tool_loop(
        system_prompt="sys", user_prompt="user", tools=[],
        dispatch={"slow_tool": slow_tool, "fast_tool": fast_tool}, agent_name="Test Agent",
    )

    # If these ran sequentially, fast-start/fast-end could only appear AFTER
    # slow-end (slow_tool is awaited to completion first). Concurrently,
    # fast_tool finishes while slow_tool is still asleep.
    assert order.index("fast-end") < order.index("slow-end")
    # Results still land in the messages in the original request order,
    # regardless of which finished first.
    tool_messages = [m for m in fake.calls[1]["messages"] if m["role"] == "tool"]
    assert tool_messages[0]["tool_call_id"] == "c1"
    assert "slow result" in tool_messages[0]["content"]
    assert tool_messages[1]["tool_call_id"] == "c2"
    assert "fast result" in tool_messages[1]["content"]


async def test_iteration_cap_forces_final_answer(monkeypatch):
    monkeypatch.setattr(settings, "max_tool_iterations", 2)

    async def get_news(ticker: str) -> str:
        return "news"

    # Two tool-calling responses (uses up both allowed iterations), then the
    # forced-final call (a third llm_client.complete invocation without tools).
    fake = FakeLLM([
        LLMMessage(content="", tool_calls=[{"id": "c1", "name": "get_news", "arguments": {"ticker": "AAPL"}}]),
        LLMMessage(content="", tool_calls=[{"id": "c2", "name": "get_news", "arguments": {"ticker": "AAPL"}}]),
        LLMMessage(content="forced final report"),
    ])
    monkeypatch.setattr(tool_loop, "llm_client", fake)
    result = await tool_loop.run_tool_loop(
        system_prompt="sys", user_prompt="user", tools=[], dispatch={"get_news": get_news}, agent_name="Test Agent",
    )
    assert result == "forced final report"
    assert len(fake.calls) == 3
    assert fake.calls[-1]["messages"][-1]["content"] == tool_loop._FORCE_FINAL_PROMPT
