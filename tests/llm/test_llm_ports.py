"""LLM Port abstraction verification — no new backend is actually integrated
here; this proves the STRUCTURE already supports one: a completely
independent, from-scratch LLM client (no LLMClient inheritance, no OpenAI
SDK) satisfies LLMPort, and the agent call sites that matter
(tool_loop.run_tool_loop, debater.make_speaker, portfolio_manager.decide)
work against it unchanged. Same scope/spirit as
tests/execution/test_broker_ports.py's IndependentSecondBroker proof."""

from __future__ import annotations

from app.agents import debater, portfolio_manager, tool_loop
from app.domain.models.llm import LLMMessage
from app.domain.ports import LLMPort
from app.llm.client import LLMClient
from app.state import DebateState, PortfolioDecision, PortfolioRating, TraderAction, TraderProposal


class IndependentSecondLLMClient:
    """A stand-in for a genuinely different LLM backend's adapter —
    deliberately shares no code, base class, or SDK with LLMClient (no
    OpenAI client, no HTTP). Returns pre-scripted responses in call order,
    same convention as the FakeLLM/FakeStructuredLLM test doubles already
    used throughout tests/agents/*.py — only exists to prove LLMPort is
    satisfiable by more than one implementation, and that the agent call
    sites don't secretly depend on anything OpenAI/vLLM-specific beyond
    this contract."""

    def __init__(self, complete_responses=None, structured_responses=None):
        self._complete_responses = list(complete_responses or [])
        self._structured_responses = list(structured_responses or [])
        self.calls: list[dict] = []

    async def complete(self, messages, *, tools=None, reasoning=False, temperature=0.0, label="", language="ko", max_tokens=None):
        self.calls.append({"messages": messages, "tools": tools, "kind": "complete"})
        return self._complete_responses.pop(0)

    async def structured_complete(self, messages, schema, *, reasoning=False, temperature=0.0, label="", language="ko"):
        self.calls.append({"messages": messages, "schema": schema, "kind": "structured"})
        return self._structured_responses.pop(0)


def test_llm_client_satisfies_llm_port():
    assert isinstance(LLMClient(), LLMPort)


def test_independent_second_llm_client_satisfies_llm_port():
    assert isinstance(IndependentSecondLLMClient(), LLMPort)


async def test_tool_loop_works_against_a_non_llmclient_backend(monkeypatch):
    fake = IndependentSecondLLMClient(complete_responses=[LLMMessage(content="final report")])
    monkeypatch.setattr(tool_loop, "llm_client", fake)

    result = await tool_loop.run_tool_loop(
        system_prompt="sys", user_prompt="user", tools=[], dispatch={}, agent_name="Test Agent",
    )

    assert result == "final report"


async def test_debater_works_against_a_non_llmclient_backend(monkeypatch):
    fake = IndependentSecondLLMClient(complete_responses=[LLMMessage(content="a rebuttal")])
    monkeypatch.setattr(debater, "llm_client", fake)

    speak = debater.make_speaker("You are a debator.", "context text")
    result = await speak(DebateState(parties=["Bull", "Bear"]), "Bull")

    assert result == "a rebuttal"


async def test_portfolio_manager_works_against_a_non_llmclient_backend(monkeypatch):
    decision = PortfolioDecision(rating=PortfolioRating.HOLD, executive_summary="s", investment_thesis="t")
    fake = IndependentSecondLLMClient(structured_responses=[decision])
    monkeypatch.setattr(portfolio_manager, "llm_client", fake)
    proposal = TraderProposal(action=TraderAction.HOLD, reasoning="wait")

    result = await portfolio_manager.decide(proposal, DebateState(parties=["Aggressive", "Conservative", "Neutral"]))

    assert result is decision
