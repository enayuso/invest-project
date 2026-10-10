"""app.agents.portfolio_manager.decide's `desk_consensus_note` param
(2026-09-04 PM consensus-override incident, id=1212/1217): a non-empty note
is placed directly above "Trader's proposal:" in the user prompt; an empty
one leaves the prompt byte-identical to before the param existed. No network
(C-stage)."""

from __future__ import annotations

import app.agents.portfolio_manager as portfolio_manager
from app.state import (
    DebateState,
    PortfolioDecision,
    PortfolioRating,
    TraderAction,
    TraderProposal,
)


class _OneShotLLM:
    def __init__(self, result):
        self._result = result
        self.calls: list[dict] = []

    async def structured_complete(self, messages, schema, *, reasoning=False, temperature=0.0, label="", language="ko"):
        self.calls.append({"messages": messages, "schema": schema})
        return self._result


def _decision() -> PortfolioDecision:
    return PortfolioDecision(
        rating=PortfolioRating.HOLD,
        executive_summary="s",
        investment_thesis="t",
        confidence=0.7,
        card_reason_ko="ko",
        card_reason_en="en",
    )


def _proposal() -> TraderProposal:
    return TraderProposal(action=TraderAction.HOLD, reasoning="reasoning")


async def _prompt_for(monkeypatch, **kwargs) -> str:
    fake = _OneShotLLM(_decision())
    monkeypatch.setattr(portfolio_manager, "llm_client", fake)
    await portfolio_manager.decide(_proposal(), DebateState(parties=["Aggressive", "Conservative", "Neutral"]), **kwargs)
    return fake.calls[0]["messages"][1]["content"]


async def test_consensus_note_appears_above_the_trader_proposal(monkeypatch):
    prompt = await _prompt_for(monkeypatch, desk_consensus_note="DESK CONSENSUS MARKER TEXT")
    assert "DESK CONSENSUS MARKER TEXT" in prompt
    assert prompt.index("DESK CONSENSUS MARKER TEXT") < prompt.index("Trader's proposal:")


async def test_empty_consensus_note_leaves_prompt_byte_identical(monkeypatch):
    baseline = await _prompt_for(monkeypatch)
    with_empty = await _prompt_for(monkeypatch, desk_consensus_note="")
    assert baseline == with_empty
