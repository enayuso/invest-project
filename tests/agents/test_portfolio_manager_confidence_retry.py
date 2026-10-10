"""app.agents.portfolio_manager's confidence retry (2026-08-19, widened
2026-08-26) — a missing `confidence` on any non-Hold rating
(`portfolio_manager._needs_confidence`) triggers exactly one extra
structured_complete call asking the model to re-state the same decision with
confidence populated. This is what should have surfaced the measured live
pattern much earlier: 9 of 9 hysteresis-gate downgrades since 2026-08-13 were
caused by a missing confidence field on Buy/Sell, not a genuine low-conviction
number, and a matching gap silently skipped Overweight/Underweight/held-Buy
trades via app.execution.sizing — see app.state.is_state_flipping_rating
(the narrower predicate the hysteresis gate itself still uses) and
app.orchestrator.pipeline._apply_hysteresis_gate. No network (C-stage)."""

from __future__ import annotations

import logging

import app.agents.portfolio_manager as portfolio_manager
from app.state import (
    DebateState,
    FlatPortfolioDecision,
    FlatPortfolioRating,
    PortfolioDecision,
    PortfolioRating,
    TraderAction,
    TraderProposal,
)


class SequencedStructuredLLM:
    """Returns one result (or raises one exception) per structured_complete
    call, in order — lets a test script "first call omits confidence, retry
    call provides/omits/raises" without a real LLM."""

    def __init__(self, results):
        self._results = list(results)
        self.calls: list[dict] = []

    async def structured_complete(self, messages, schema, *, reasoning=False, temperature=0.0, label="", language="ko"):
        self.calls.append({"messages": messages, "schema": schema})
        result = self._results.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result


def _debate() -> DebateState:
    return DebateState(parties=["Aggressive", "Conservative", "Neutral"])


def _proposal(action: TraderAction = TraderAction.SELL) -> TraderProposal:
    return TraderProposal(action=action, reasoning="reasoning")


async def test_held_sell_missing_confidence_triggers_one_retry_and_uses_result(monkeypatch):
    first = PortfolioDecision(
        rating=PortfolioRating.SELL, executive_summary="s", investment_thesis="t", confidence=None,
        card_reason_ko="ko reason", card_reason_en="en reason",
    )
    retried = PortfolioDecision(
        rating=PortfolioRating.SELL, executive_summary="s2", investment_thesis="t2", confidence=0.8,
        card_reason_ko="ko reason", card_reason_en="en reason",
    )
    fake = SequencedStructuredLLM([first, retried])
    monkeypatch.setattr(portfolio_manager, "llm_client", fake)

    result = await portfolio_manager.decide(_proposal(), _debate(), held=True)

    assert len(fake.calls) == 2
    assert fake.calls[1]["schema"] is portfolio_manager._ConfidenceFill
    # only confidence is grafted from the retry — every other field (rating,
    # executive_summary, ...) must come from the original decision, unchanged.
    assert result.confidence == 0.8
    assert result.rating is PortfolioRating.SELL
    assert result.executive_summary == "s"
    # the retry appends the original decision as an assistant turn (2026-08-25
    # — see portfolio_manager._retry_if_confidence_missing's own comment on
    # the gemma-4 incident this fixes) plus exactly one user instruction.
    retry_messages = fake.calls[1]["messages"]
    assert retry_messages[:2] == fake.calls[0]["messages"]
    assert len(retry_messages) == len(fake.calls[0]["messages"]) + 2
    assert retry_messages[-2]["role"] == "assistant"
    assert "confidence" in retry_messages[-1]["content"].lower()


async def test_retry_with_different_rating_does_not_flip_original_decision(monkeypatch):
    """Regression pin (updated 2026-08-25): the retry prompt asks the model
    to return the same decision unchanged except for confidence, but nothing
    enforces that. If the retry comes back with a different rating, that
    confidence was never actually attached to `decision.rating` — it's
    evidence about a decision that was never made — so the WHOLE retry is
    discarded, not just partially trusted. (Before 2026-08-25 this test
    pinned the opposite: grafting the differently-rated retry's confidence
    anyway. That was itself a bug — see
    portfolio_manager._retry_if_confidence_missing's own comment.)"""
    first = PortfolioDecision(
        rating=PortfolioRating.SELL,
        executive_summary="orig summary",
        investment_thesis="orig thesis",
        stop_loss=90.0,
        confidence=None,
        card_reason_ko="ko reason", card_reason_en="en reason",
    )
    retried = PortfolioDecision(
        rating=PortfolioRating.BUY,
        executive_summary="flipped summary",
        investment_thesis="flipped thesis",
        stop_loss=120.0,
        confidence=0.9,
        card_reason_ko="ko reason", card_reason_en="en reason",
    )
    fake = SequencedStructuredLLM([first, retried])
    monkeypatch.setattr(portfolio_manager, "llm_client", fake)

    result = await portfolio_manager.decide(_proposal(), _debate(), held=True)

    assert len(fake.calls) == 2
    assert result.rating is PortfolioRating.SELL  # original rating preserved, not flipped to BUY
    assert result.confidence is None  # discarded entirely, not grafted from a different rating
    assert result.investment_thesis == "orig thesis"  # every other field stays the original's
    assert result.stop_loss == 90.0


async def test_held_hold_missing_confidence_triggers_no_retry(monkeypatch):
    decision = PortfolioDecision(
        rating=PortfolioRating.HOLD, executive_summary="s", investment_thesis="t", confidence=None,
        card_reason_ko="ko reason", card_reason_en="en reason",
    )
    fake = SequencedStructuredLLM([decision])
    monkeypatch.setattr(portfolio_manager, "llm_client", fake)

    result = await portfolio_manager.decide(_proposal(TraderAction.HOLD), _debate(), held=True)

    assert len(fake.calls) == 1
    assert result.confidence is None
    assert result.rating is PortfolioRating.HOLD


async def test_retry_still_missing_confidence_keeps_original_decision(monkeypatch):
    first = PortfolioDecision(
        rating=PortfolioRating.SELL, executive_summary="orig", investment_thesis="t", confidence=None,
        card_reason_ko="ko reason", card_reason_en="en reason",
    )
    retried = PortfolioDecision(
        rating=PortfolioRating.SELL, executive_summary="still missing", investment_thesis="t2", confidence=None,
        card_reason_ko="ko reason", card_reason_en="en reason",
    )
    fake = SequencedStructuredLLM([first, retried])
    monkeypatch.setattr(portfolio_manager, "llm_client", fake)

    result = await portfolio_manager.decide(_proposal(), _debate(), held=True)

    assert len(fake.calls) == 2
    assert result.confidence is None
    assert result.executive_summary == "orig"  # original decision kept, not the retried one


async def test_retry_that_raises_is_swallowed_and_original_decision_returned(monkeypatch, caplog):
    first = PortfolioDecision(
        rating=PortfolioRating.SELL, executive_summary="orig", investment_thesis="t", confidence=None,
        card_reason_ko="ko reason", card_reason_en="en reason",
    )
    fake = SequencedStructuredLLM([first, RuntimeError("boom")])
    monkeypatch.setattr(portfolio_manager, "llm_client", fake)

    with caplog.at_level(logging.WARNING):
        result = await portfolio_manager.decide(_proposal(), _debate(), held=True)

    assert len(fake.calls) == 2
    assert result.confidence is None
    assert result.executive_summary == "orig"
    assert "confidence retry" in caplog.text


async def test_flat_buy_missing_confidence_triggers_retry_before_normalization(monkeypatch):
    # stop_loss is set on both fixtures (unlike this test's other fields) so
    # the stop_loss retry (portfolio_manager._retry_if_stop_loss_missing)
    # does not also fire here — this test is only about the confidence
    # retry chain; see test_flat_buy_missing_stop_loss_triggers_retry_before_normalization
    # for that one in isolation.
    first = FlatPortfolioDecision(
        rating=FlatPortfolioRating.BUY, executive_summary="s", investment_thesis="t",
        stop_loss=90.0, confidence=None,
        card_reason_ko="ko reason", card_reason_en="en reason",
    )
    retried = FlatPortfolioDecision(
        rating=FlatPortfolioRating.BUY, executive_summary="s2", investment_thesis="t2",
        stop_loss=90.0, confidence=0.7,
        card_reason_ko="ko reason", card_reason_en="en reason",
    )
    fake = SequencedStructuredLLM([first, retried])
    monkeypatch.setattr(portfolio_manager, "llm_client", fake)

    result = await portfolio_manager.decide(_proposal(TraderAction.BUY), _debate(), held=False)

    assert len(fake.calls) == 2
    # narrow schema is shared across the flat/held split (2026-09-07) — no
    # more FlatPortfolioDecision-vs-PortfolioDecision distinction on retries.
    assert fake.calls[1]["schema"] is portfolio_manager._ConfidenceFill
    assert isinstance(result, PortfolioDecision)
    assert result.confidence == 0.7
    assert result.executive_summary == "s"


async def test_flat_hold_missing_confidence_triggers_no_retry(monkeypatch):
    flat_decision = FlatPortfolioDecision(
        rating=FlatPortfolioRating.HOLD, executive_summary="s", investment_thesis="t", confidence=None,
        card_reason_ko="ko reason", card_reason_en="en reason",
    )
    fake = SequencedStructuredLLM([flat_decision])
    monkeypatch.setattr(portfolio_manager, "llm_client", fake)

    result = await portfolio_manager.decide(_proposal(TraderAction.HOLD), _debate(), held=False)

    assert len(fake.calls) == 1
    assert result.confidence is None


async def test_held_underweight_missing_confidence_triggers_retry_and_grafts_result(monkeypatch):
    first = PortfolioDecision(
        rating=PortfolioRating.UNDERWEIGHT, executive_summary="orig", investment_thesis="t", confidence=None,
        card_reason_ko="ko reason", card_reason_en="en reason",
    )
    retried = PortfolioDecision(
        rating=PortfolioRating.UNDERWEIGHT, executive_summary="s2", investment_thesis="t2", confidence=0.7,
        card_reason_ko="ko reason", card_reason_en="en reason",
    )
    fake = SequencedStructuredLLM([first, retried])
    monkeypatch.setattr(portfolio_manager, "llm_client", fake)

    result = await portfolio_manager.decide(_proposal(TraderAction.SELL), _debate(), held=True)

    assert len(fake.calls) == 2
    assert result.confidence == 0.7
    assert result.executive_summary == "orig"  # only confidence grafted


async def test_held_overweight_missing_confidence_triggers_retry_and_grafts_result(monkeypatch):
    first = PortfolioDecision(
        rating=PortfolioRating.OVERWEIGHT, executive_summary="orig", investment_thesis="t",
        confidence=None, stop_loss=90.0,
        card_reason_ko="ko reason", card_reason_en="en reason",
    )
    retried = PortfolioDecision(
        rating=PortfolioRating.OVERWEIGHT, executive_summary="s2", investment_thesis="t2",
        confidence=0.8, stop_loss=90.0,
        card_reason_ko="ko reason", card_reason_en="en reason",
    )
    fake = SequencedStructuredLLM([first, retried])
    monkeypatch.setattr(portfolio_manager, "llm_client", fake)

    result = await portfolio_manager.decide(_proposal(TraderAction.BUY), _debate(), held=True)

    assert len(fake.calls) == 2
    assert result.confidence == 0.8
    assert result.executive_summary == "orig"


async def test_held_buy_missing_confidence_triggers_retry(monkeypatch):
    # Buy while already held is not state-flipping (the position already
    # exists), but still has a non-None base allocation in sizing — the gap
    # `_needs_confidence` closes alongside Underweight/Overweight.
    first = PortfolioDecision(
        rating=PortfolioRating.BUY, executive_summary="orig", investment_thesis="t",
        confidence=None, stop_loss=90.0,
        card_reason_ko="ko reason", card_reason_en="en reason",
    )
    retried = PortfolioDecision(
        rating=PortfolioRating.BUY, executive_summary="s2", investment_thesis="t2",
        confidence=0.9, stop_loss=90.0,
        card_reason_ko="ko reason", card_reason_en="en reason",
    )
    fake = SequencedStructuredLLM([first, retried])
    monkeypatch.setattr(portfolio_manager, "llm_client", fake)

    result = await portfolio_manager.decide(_proposal(TraderAction.BUY), _debate(), held=True)

    assert len(fake.calls) == 2
    assert result.confidence == 0.9


# ---------------------------------------------------------------------------
# Flat-decision normalization must carry every shared field forward, not just
# the ones listed by name — 2026-08-20 incident: card_reason_ko/en were added
# to the shared field set (app.state._PortfolioDecisionFields) but the flat
# normalization site only listed fields one by one, so every flat ticker's
# card_reason silently dropped to None regardless of what the model answered.
# ---------------------------------------------------------------------------


async def test_flat_decision_normalization_carries_card_reason_fields(monkeypatch):
    flat_decision = FlatPortfolioDecision(
        rating=FlatPortfolioRating.HOLD, executive_summary="s", investment_thesis="t", confidence=0.8,
        card_reason_ko="한글 이유", card_reason_en="english reason",
    )
    fake = SequencedStructuredLLM([flat_decision])
    monkeypatch.setattr(portfolio_manager, "llm_client", fake)

    result = await portfolio_manager.decide(_proposal(TraderAction.HOLD), _debate(), held=False)

    assert len(fake.calls) == 1  # no retry — everything was already filled
    assert isinstance(result, PortfolioDecision)
    assert result.card_reason_ko == "한글 이유"
    assert result.card_reason_en == "english reason"


async def test_flat_normalization_preserves_every_shared_field(monkeypatch):
    flat_decision = FlatPortfolioDecision(
        rating=FlatPortfolioRating.BUY,
        executive_summary="summary",
        investment_thesis="thesis",
        price_target=123.45,
        time_horizon="3 months",
        stop_loss=90.0,
        confidence=0.75,
        card_reason_ko="ko",
        card_reason_en="en",
    )
    fake = SequencedStructuredLLM([flat_decision])
    monkeypatch.setattr(portfolio_manager, "llm_client", fake)

    result = await portfolio_manager.decide(_proposal(TraderAction.BUY), _debate(), held=False)

    assert isinstance(result, PortfolioDecision)
    assert result.rating is PortfolioRating.BUY
    assert result.price_target == 123.45
    assert result.time_horizon == "3 months"
    assert result.stop_loss == 90.0
    assert result.confidence == 0.75
    assert result.card_reason_ko == "ko"
    assert result.card_reason_en == "en"


# ---------------------------------------------------------------------------
# card_reason retry — mirrors the confidence retry above, but fires for
# every rating (not just state-flipping ones) since a display reason is
# expected on Hold/Sell too.
# ---------------------------------------------------------------------------


async def test_missing_card_reason_triggers_one_retry_and_grafts_both(monkeypatch):
    first = PortfolioDecision(
        rating=PortfolioRating.SELL, executive_summary="orig", investment_thesis="t", confidence=0.9,
        card_reason_ko=None, card_reason_en=None,
    )
    retried = PortfolioDecision(
        rating=PortfolioRating.BUY, executive_summary="different", investment_thesis="different", confidence=0.1,
        card_reason_ko="ko filled", card_reason_en="en filled",
    )
    fake = SequencedStructuredLLM([first, retried])
    monkeypatch.setattr(portfolio_manager, "llm_client", fake)

    result = await portfolio_manager.decide(_proposal(), _debate(), held=True)

    assert len(fake.calls) == 2  # confidence retry did not fire (0.9 already set)
    assert result.card_reason_ko == "ko filled"
    assert result.card_reason_en == "en filled"
    # every non-display field stays the original's
    assert result.rating is PortfolioRating.SELL
    assert result.executive_summary == "orig"
    assert result.confidence == 0.9


async def test_card_reason_retry_does_not_adopt_other_retried_fields(monkeypatch):
    first = PortfolioDecision(
        rating=PortfolioRating.SELL, executive_summary="orig", investment_thesis="t",
        stop_loss=90.0, confidence=0.9, card_reason_ko=None, card_reason_en=None,
    )
    retried = PortfolioDecision(
        rating=PortfolioRating.BUY, executive_summary="flipped", investment_thesis="flipped",
        stop_loss=200.0, confidence=0.1, card_reason_ko="ko filled", card_reason_en="en filled",
    )
    fake = SequencedStructuredLLM([first, retried])
    monkeypatch.setattr(portfolio_manager, "llm_client", fake)

    result = await portfolio_manager.decide(_proposal(), _debate(), held=True)

    assert result.rating is PortfolioRating.SELL
    assert result.stop_loss == 90.0
    assert result.confidence == 0.9
    assert result.card_reason_ko == "ko filled"
    assert result.card_reason_en == "en filled"


async def test_card_reason_retry_grafts_only_the_missing_side(monkeypatch):
    first = PortfolioDecision(
        rating=PortfolioRating.HOLD, executive_summary="s", investment_thesis="t", confidence=0.6,
        card_reason_ko=None, card_reason_en="already here",
    )
    retried = PortfolioDecision(
        rating=PortfolioRating.HOLD, executive_summary="s", investment_thesis="t", confidence=0.6,
        card_reason_ko="new ko", card_reason_en="should not be used",
    )
    fake = SequencedStructuredLLM([first, retried])
    monkeypatch.setattr(portfolio_manager, "llm_client", fake)

    result = await portfolio_manager.decide(_proposal(TraderAction.HOLD), _debate(), held=True)

    assert result.card_reason_ko == "new ko"
    assert result.card_reason_en == "already here"  # untouched, kept from the original


async def test_placeholder_card_reason_counts_as_missing(monkeypatch):
    first = PortfolioDecision(
        rating=PortfolioRating.HOLD, executive_summary="s", investment_thesis="t", confidence=0.6,
        card_reason_ko="   ", card_reason_en="N/A",
    )
    retried = PortfolioDecision(
        rating=PortfolioRating.HOLD, executive_summary="s", investment_thesis="t", confidence=0.6,
        card_reason_ko="real ko", card_reason_en="real en",
    )
    fake = SequencedStructuredLLM([first, retried])
    monkeypatch.setattr(portfolio_manager, "llm_client", fake)

    result = await portfolio_manager.decide(_proposal(TraderAction.HOLD), _debate(), held=True)

    assert len(fake.calls) == 2
    assert result.card_reason_ko == "real ko"
    assert result.card_reason_en == "real en"


async def test_card_reason_retry_that_raises_keeps_original_decision(monkeypatch, caplog):
    first = PortfolioDecision(
        rating=PortfolioRating.HOLD, executive_summary="orig", investment_thesis="t", confidence=0.6,
        card_reason_ko=None, card_reason_en=None,
    )
    fake = SequencedStructuredLLM([first, RuntimeError("boom")])
    monkeypatch.setattr(portfolio_manager, "llm_client", fake)

    with caplog.at_level(logging.WARNING):
        result = await portfolio_manager.decide(_proposal(TraderAction.HOLD), _debate(), held=True)

    assert len(fake.calls) == 2
    assert result.card_reason_ko is None
    assert result.card_reason_en is None
    assert result.executive_summary == "orig"
    assert "card_reason retry" in caplog.text


async def test_card_reason_retry_still_missing_keeps_original(monkeypatch):
    first = PortfolioDecision(
        rating=PortfolioRating.HOLD, executive_summary="orig", investment_thesis="t", confidence=0.6,
        card_reason_ko=None, card_reason_en=None,
    )
    retried = PortfolioDecision(
        rating=PortfolioRating.HOLD, executive_summary="s2", investment_thesis="t2", confidence=0.6,
        card_reason_ko=None, card_reason_en=None,
    )
    fake = SequencedStructuredLLM([first, retried])
    monkeypatch.setattr(portfolio_manager, "llm_client", fake)

    result = await portfolio_manager.decide(_proposal(TraderAction.HOLD), _debate(), held=True)

    assert len(fake.calls) == 2
    assert result.card_reason_ko is None
    assert result.card_reason_en is None
    assert result.executive_summary == "orig"


async def test_placeholder_card_reason_is_cleared_when_retry_also_degenerate(monkeypatch):
    first = PortfolioDecision(
        rating=PortfolioRating.HOLD, executive_summary="orig", investment_thesis="t", confidence=0.6,
        card_reason_ko="N/A", card_reason_en="N/A",
    )
    retried = PortfolioDecision(
        rating=PortfolioRating.HOLD, executive_summary="s2", investment_thesis="t2", confidence=0.6,
        card_reason_ko="none", card_reason_en=None,
    )
    fake = SequencedStructuredLLM([first, retried])
    monkeypatch.setattr(portfolio_manager, "llm_client", fake)

    result = await portfolio_manager.decide(_proposal(TraderAction.HOLD), _debate(), held=True)

    assert len(fake.calls) == 2
    assert result.card_reason_ko is None
    assert result.card_reason_en is None
    # a placeholder that couldn't be recovered clears to None (falls back to
    # the card's own executive_summary display), never renders "N/A" verbatim
    assert result.rating == PortfolioRating.HOLD
    assert result.confidence == 0.6
    assert result.executive_summary == "orig"


async def test_placeholder_card_reason_is_cleared_when_retry_raises(monkeypatch, caplog):
    first = PortfolioDecision(
        rating=PortfolioRating.HOLD, executive_summary="orig", investment_thesis="t", confidence=0.6,
        card_reason_ko="N/A", card_reason_en="  ",
    )
    fake = SequencedStructuredLLM([first, RuntimeError("boom")])
    monkeypatch.setattr(portfolio_manager, "llm_client", fake)

    with caplog.at_level(logging.WARNING):
        result = await portfolio_manager.decide(_proposal(TraderAction.HOLD), _debate(), held=True)

    assert len(fake.calls) == 2
    assert result.card_reason_ko is None
    assert result.card_reason_en is None
    assert result.executive_summary == "orig"
    assert "card_reason retry" in caplog.text


async def test_placeholder_on_one_side_cleared_while_other_side_recovers(monkeypatch):
    first = PortfolioDecision(
        rating=PortfolioRating.HOLD, executive_summary="orig", investment_thesis="t", confidence=0.6,
        card_reason_ko="tbd", card_reason_en="real en",
    )
    retried = PortfolioDecision(
        rating=PortfolioRating.HOLD, executive_summary="s2", investment_thesis="t2", confidence=0.6,
        card_reason_ko="none", card_reason_en="ignored",
    )
    fake = SequencedStructuredLLM([first, retried])
    monkeypatch.setattr(portfolio_manager, "llm_client", fake)

    result = await portfolio_manager.decide(_proposal(TraderAction.HOLD), _debate(), held=True)

    assert result.card_reason_ko is None
    assert result.card_reason_en == "real en"  # untouched, never overwritten by the retry


async def test_confidence_retry_also_grafts_card_reason_without_extra_call(monkeypatch):
    first = PortfolioDecision(
        rating=PortfolioRating.SELL, executive_summary="orig", investment_thesis="t", confidence=None,
        card_reason_ko=None, card_reason_en=None,
    )
    # Same rating as `first` — this test is about the card_reason cost-saving
    # graft, not the rating-mismatch guard (see
    # test_retry_with_different_rating_does_not_flip_original_decision for that).
    retried = PortfolioDecision(
        rating=PortfolioRating.SELL, executive_summary="different", investment_thesis="different", confidence=0.8,
        card_reason_ko="ko filled", card_reason_en="en filled",
    )
    fake = SequencedStructuredLLM([first, retried])
    monkeypatch.setattr(portfolio_manager, "llm_client", fake)

    result = await portfolio_manager.decide(_proposal(), _debate(), held=True)

    assert len(fake.calls) == 2  # NOT 3 — the confidence retry's own response supplied both
    assert result.confidence == 0.8
    assert result.card_reason_ko == "ko filled"
    assert result.card_reason_en == "en filled"
    assert result.rating is PortfolioRating.SELL  # non-display fields still from the original


async def test_flat_buy_missing_both_triggers_retries_in_order(monkeypatch):
    # stop_loss is set on both fixtures so the stop_loss retry does not also
    # fire here — see test_flat_buy_missing_confidence_triggers_retry_before_normalization
    # for why, and test_decide_chains_confidence_then_stop_loss_then_card_reason_retries
    # for a case where all three fire in sequence.
    first = FlatPortfolioDecision(
        rating=FlatPortfolioRating.BUY, executive_summary="s", investment_thesis="t",
        stop_loss=90.0, confidence=None,
        card_reason_ko=None, card_reason_en=None,
    )
    retried = FlatPortfolioDecision(
        rating=FlatPortfolioRating.BUY, executive_summary="s2", investment_thesis="t2",
        stop_loss=90.0, confidence=0.7,
        card_reason_ko="ko filled", card_reason_en="en filled",
    )
    fake = SequencedStructuredLLM([first, retried])
    monkeypatch.setattr(portfolio_manager, "llm_client", fake)

    result = await portfolio_manager.decide(_proposal(TraderAction.BUY), _debate(), held=False)

    assert len(fake.calls) == 2
    assert isinstance(result, PortfolioDecision)
    assert result.confidence == 0.7
    assert result.card_reason_ko == "ko filled"
    assert result.card_reason_en == "en filled"


# ---------------------------------------------------------------------------
# stop_loss retry — mirrors the confidence retry above, but scoped to
# Buy/Overweight only (the ratings that open or add to a position; see
# portfolio_manager._needs_stop_loss). Real incidents: 2026-08-20 950160.KS,
# 2026-08-25 MRVL, 2026-08-26 002990.KS — all Buy/Overweight, all with the
# stop-loss stated in prose but left blank in the structured field, silently
# substituted by app.execution.ledger's ATR-floor fallback.
# ---------------------------------------------------------------------------


async def test_buy_with_stop_loss_already_set_triggers_no_retry(monkeypatch):
    decision = PortfolioDecision(
        rating=PortfolioRating.BUY, executive_summary="s", investment_thesis="t",
        stop_loss=90.0, confidence=0.8, card_reason_ko="ko reason", card_reason_en="en reason",
    )
    fake = SequencedStructuredLLM([decision])
    monkeypatch.setattr(portfolio_manager, "llm_client", fake)

    result = await portfolio_manager.decide(_proposal(TraderAction.BUY), _debate(), held=True)

    assert len(fake.calls) == 1  # no retry at all — stop_loss already present
    assert result.stop_loss == 90.0


async def test_held_hold_missing_stop_loss_triggers_no_retry(monkeypatch):
    decision = PortfolioDecision(
        rating=PortfolioRating.HOLD, executive_summary="s", investment_thesis="t",
        stop_loss=None, confidence=0.6, card_reason_ko="ko reason", card_reason_en="en reason",
    )
    fake = SequencedStructuredLLM([decision])
    monkeypatch.setattr(portfolio_manager, "llm_client", fake)

    result = await portfolio_manager.decide(_proposal(TraderAction.HOLD), _debate(), held=True)

    assert len(fake.calls) == 1  # scope guard — Hold never needs a stop_loss
    assert result.stop_loss is None


async def test_held_sell_missing_stop_loss_triggers_no_retry(monkeypatch):
    decision = PortfolioDecision(
        rating=PortfolioRating.SELL, executive_summary="s", investment_thesis="t",
        stop_loss=None, confidence=0.8, card_reason_ko="ko reason", card_reason_en="en reason",
    )
    fake = SequencedStructuredLLM([decision])
    monkeypatch.setattr(portfolio_manager, "llm_client", fake)

    result = await portfolio_manager.decide(_proposal(), _debate(), held=True)

    assert len(fake.calls) == 1  # scope guard — Sell exits, nothing left for a stop to protect
    assert result.stop_loss is None


async def test_held_underweight_missing_stop_loss_triggers_no_retry(monkeypatch):
    decision = PortfolioDecision(
        rating=PortfolioRating.UNDERWEIGHT, executive_summary="s", investment_thesis="t",
        stop_loss=None, confidence=0.6, card_reason_ko="ko reason", card_reason_en="en reason",
    )
    fake = SequencedStructuredLLM([decision])
    monkeypatch.setattr(portfolio_manager, "llm_client", fake)

    result = await portfolio_manager.decide(_proposal(TraderAction.HOLD), _debate(), held=True)

    assert len(fake.calls) == 1  # scope guard — reduces but doesn't add exposure
    assert result.stop_loss is None


async def test_held_buy_missing_stop_loss_triggers_retry_and_grafts_result(monkeypatch):
    first = PortfolioDecision(
        rating=PortfolioRating.BUY, executive_summary="orig", investment_thesis="t",
        stop_loss=None, confidence=0.8, card_reason_ko="ko reason", card_reason_en="en reason",
    )
    retried = PortfolioDecision(
        rating=PortfolioRating.BUY, executive_summary="s2", investment_thesis="t2",
        stop_loss=232.08, confidence=0.9, card_reason_ko="ko reason", card_reason_en="en reason",
    )
    fake = SequencedStructuredLLM([first, retried])
    monkeypatch.setattr(portfolio_manager, "llm_client", fake)

    result = await portfolio_manager.decide(_proposal(TraderAction.BUY), _debate(), held=True)

    assert len(fake.calls) == 2
    assert fake.calls[1]["schema"] is portfolio_manager._StopLossFill
    # only stop_loss is grafted from the retry — every other field comes
    # from the original decision, unchanged.
    assert result.stop_loss == 232.08
    assert result.rating is PortfolioRating.BUY
    assert result.executive_summary == "orig"
    assert result.confidence == 0.8
    retry_messages = fake.calls[1]["messages"]
    assert retry_messages[:2] == fake.calls[0]["messages"]
    assert retry_messages[-2]["role"] == "assistant"
    assert "stop_loss" in retry_messages[-1]["content"].lower()


async def test_held_overweight_missing_stop_loss_triggers_retry_and_grafts_result(monkeypatch):
    first = PortfolioDecision(
        rating=PortfolioRating.OVERWEIGHT, executive_summary="orig", investment_thesis="t",
        stop_loss=None, confidence=0.8, card_reason_ko="ko reason", card_reason_en="en reason",
    )
    retried = PortfolioDecision(
        rating=PortfolioRating.OVERWEIGHT, executive_summary="s2", investment_thesis="t2",
        stop_loss=14000.0, confidence=0.9, card_reason_ko="ko reason", card_reason_en="en reason",
    )
    fake = SequencedStructuredLLM([first, retried])
    monkeypatch.setattr(portfolio_manager, "llm_client", fake)

    result = await portfolio_manager.decide(_proposal(TraderAction.BUY), _debate(), held=True)

    assert len(fake.calls) == 2
    assert result.stop_loss == 14000.0
    assert result.rating is PortfolioRating.OVERWEIGHT
    assert result.executive_summary == "orig"


async def test_stop_loss_retry_still_missing_keeps_original_decision(monkeypatch):
    first = PortfolioDecision(
        rating=PortfolioRating.BUY, executive_summary="orig", investment_thesis="t",
        stop_loss=None, confidence=0.8, card_reason_ko="ko reason", card_reason_en="en reason",
    )
    retried = PortfolioDecision(
        rating=PortfolioRating.BUY, executive_summary="still missing", investment_thesis="t2",
        stop_loss=None, confidence=0.8, card_reason_ko="ko reason", card_reason_en="en reason",
    )
    fake = SequencedStructuredLLM([first, retried])
    monkeypatch.setattr(portfolio_manager, "llm_client", fake)

    result = await portfolio_manager.decide(_proposal(TraderAction.BUY), _debate(), held=True)

    assert len(fake.calls) == 2
    assert result.stop_loss is None
    assert result.executive_summary == "orig"  # original decision kept, not the retried one


async def test_stop_loss_retry_with_different_rating_does_not_flip_original_decision(monkeypatch):
    first = PortfolioDecision(
        rating=PortfolioRating.BUY, executive_summary="orig summary", investment_thesis="orig thesis",
        stop_loss=None, confidence=0.8, card_reason_ko="ko reason", card_reason_en="en reason",
    )
    retried = PortfolioDecision(
        rating=PortfolioRating.SELL, executive_summary="flipped summary", investment_thesis="flipped thesis",
        stop_loss=120.0, confidence=0.9, card_reason_ko="ko reason", card_reason_en="en reason",
    )
    fake = SequencedStructuredLLM([first, retried])
    monkeypatch.setattr(portfolio_manager, "llm_client", fake)

    result = await portfolio_manager.decide(_proposal(TraderAction.BUY), _debate(), held=True)

    assert len(fake.calls) == 2
    assert result.rating is PortfolioRating.BUY  # original rating preserved, not flipped to SELL
    assert result.stop_loss is None  # discarded entirely, not grafted from a different rating
    assert result.investment_thesis == "orig thesis"


async def test_stop_loss_retry_that_raises_is_swallowed_and_original_decision_returned(monkeypatch, caplog):
    first = PortfolioDecision(
        rating=PortfolioRating.BUY, executive_summary="orig", investment_thesis="t",
        stop_loss=None, confidence=0.8, card_reason_ko="ko reason", card_reason_en="en reason",
    )
    fake = SequencedStructuredLLM([first, RuntimeError("boom")])
    monkeypatch.setattr(portfolio_manager, "llm_client", fake)

    with caplog.at_level(logging.WARNING):
        result = await portfolio_manager.decide(_proposal(TraderAction.BUY), _debate(), held=True)

    assert len(fake.calls) == 2
    assert result.stop_loss is None
    assert result.executive_summary == "orig"
    assert "stop_loss retry" in caplog.text


async def test_flat_buy_missing_stop_loss_triggers_retry_before_normalization(monkeypatch):
    first = FlatPortfolioDecision(
        rating=FlatPortfolioRating.BUY, executive_summary="s", investment_thesis="t",
        stop_loss=None, confidence=0.7, card_reason_ko="ko reason", card_reason_en="en reason",
    )
    retried = FlatPortfolioDecision(
        rating=FlatPortfolioRating.BUY, executive_summary="s2", investment_thesis="t2",
        stop_loss=18500.0, confidence=0.7, card_reason_ko="ko reason", card_reason_en="en reason",
    )
    fake = SequencedStructuredLLM([first, retried])
    monkeypatch.setattr(portfolio_manager, "llm_client", fake)

    result = await portfolio_manager.decide(_proposal(TraderAction.BUY), _debate(), held=False)

    assert len(fake.calls) == 2
    assert fake.calls[1]["schema"] is portfolio_manager._StopLossFill
    assert isinstance(result, PortfolioDecision)
    assert result.stop_loss == 18500.0
    assert result.executive_summary == "s"


async def test_decide_chains_confidence_then_stop_loss_then_card_reason_retries(monkeypatch):
    """Integration-level pin: all three retries can fire in sequence for one
    decision, in the order wired into decide() — confidence, then stop_loss,
    then card_reason — each a separate structured_complete call since none of
    the earlier retries' responses happen to fill the later missing fields.
    Uses a flat Buy (held=False) since that's the shape `decide()`'s
    normalization path (FlatPortfolioDecision -> PortfolioDecision) exercises
    alongside the retry chain; a held Buy would chain identically."""
    first = FlatPortfolioDecision(
        rating=FlatPortfolioRating.BUY, executive_summary="orig", investment_thesis="t",
        stop_loss=None, confidence=None, card_reason_ko=None, card_reason_en=None,
    )
    confidence_retried = FlatPortfolioDecision(
        rating=FlatPortfolioRating.BUY, executive_summary="orig", investment_thesis="t",
        stop_loss=None, confidence=0.8, card_reason_ko=None, card_reason_en=None,
    )
    stop_loss_retried = FlatPortfolioDecision(
        rating=FlatPortfolioRating.BUY, executive_summary="orig", investment_thesis="t",
        stop_loss=232.08, confidence=0.8, card_reason_ko=None, card_reason_en=None,
    )
    card_reason_retried = FlatPortfolioDecision(
        rating=FlatPortfolioRating.BUY, executive_summary="orig", investment_thesis="t",
        stop_loss=232.08, confidence=0.8, card_reason_ko="ko filled", card_reason_en="en filled",
    )
    fake = SequencedStructuredLLM([first, confidence_retried, stop_loss_retried, card_reason_retried])
    monkeypatch.setattr(portfolio_manager, "llm_client", fake)

    result = await portfolio_manager.decide(_proposal(TraderAction.BUY), _debate(), held=False)

    assert len(fake.calls) == 4  # original + confidence retry + stop_loss retry + card_reason retry
    assert isinstance(result, PortfolioDecision)
    assert result.confidence == 0.8
    assert result.stop_loss == 232.08
    assert result.card_reason_ko == "ko filled"
    assert result.card_reason_en == "en filled"


def test_system_prompt_mentions_both_card_reason_fields():
    assert "card_reason_ko" in portfolio_manager.SYSTEM_PROMPT
    assert "card_reason_en" in portfolio_manager.SYSTEM_PROMPT


def test_system_prompt_instructs_execution_status_overrides_halt_claims():
    # 2026-09-07, id=1238 (351320.KS): a stale news headline's false "currently
    # halted" claim propagated uncorrected through the Research Manager, the
    # risk debate, and into this exact PM's final decision text, even though
    # execution_status_context already said the opposite this same cycle.
    assert (
        "the only authoritative source for whether the ticker is halted, under a VI, "
        "or under a circuit-breaker right now" in portfolio_manager.SYSTEM_PROMPT
    )
    assert "we cannot act until trading resumes" in portfolio_manager.SYSTEM_PROMPT


async def test_decide_includes_sibling_context_block_when_present(monkeypatch):
    fake = SequencedStructuredLLM(
        [PortfolioDecision(rating=PortfolioRating.HOLD, executive_summary="s", investment_thesis="t", confidence=0.5)]
    )
    monkeypatch.setattr(portfolio_manager, "llm_client", fake)

    await portfolio_manager.decide(
        _proposal(), _debate(), held=True,
        sibling_context="Earlier decisions on THIS SAME ticker today (...):\n\n[00:21 | Sell]\nWHY: quick ratio 0%",
    )

    prompt = fake.calls[0]["messages"][1]["content"]
    assert "Earlier decisions on THIS SAME ticker today" in prompt
    assert "quick ratio 0%" in prompt


async def test_decide_omits_sibling_context_block_when_absent(monkeypatch):
    # The common case (no same-day sibling) must produce a prompt with no
    # trace of the feature — no placeholder line, unlike every other
    # context block in this prompt.
    fake = SequencedStructuredLLM(
        [PortfolioDecision(rating=PortfolioRating.HOLD, executive_summary="s", investment_thesis="t", confidence=0.5)]
    )
    monkeypatch.setattr(portfolio_manager, "llm_client", fake)

    await portfolio_manager.decide(_proposal(), _debate(), held=True)

    prompt = fake.calls[0]["messages"][1]["content"]
    assert "Earlier decisions on THIS SAME ticker today" not in prompt
    assert "Same-ticker" not in prompt


# ---------------------------------------------------------------------------
# degradation_note (2026-09-03 decision id=1179) — the PM must be told, before
# it decides, when an analyst report failed to generate this cycle. Empty
# note => prompt byte-identical to before the parameter existed.
# ---------------------------------------------------------------------------


async def test_decide_prompt_is_unchanged_without_a_degradation_note(monkeypatch):
    decision = PortfolioDecision(
        rating=PortfolioRating.HOLD, executive_summary="s", investment_thesis="t", confidence=0.5,
    )
    fake_a = SequencedStructuredLLM([decision])
    fake_b = SequencedStructuredLLM([decision])

    monkeypatch.setattr(portfolio_manager, "llm_client", fake_a)
    await portfolio_manager.decide(_proposal(), _debate(), held=True)

    monkeypatch.setattr(portfolio_manager, "llm_client", fake_b)
    await portfolio_manager.decide(_proposal(), _debate(), held=True, degradation_note="")

    assert fake_a.calls[0]["messages"] == fake_b.calls[0]["messages"]


# ---------------------------------------------------------------------------
# Narrow retry schemas (2026-09-07, AI-engine step-0 latency audit) — pins
# for the schema classes themselves, not just decide()'s behavior through
# them. SequencedStructuredLLM ignores `schema` entirely (returns whatever
# fixture object was queued), so these are the only tests that would catch
# field creep back into the retry transport shapes, or the narrow-schema
# selection logic picking the wrong class.
# ---------------------------------------------------------------------------


def test_confidence_fill_schema_has_only_rating_and_confidence():
    assert set(portfolio_manager._ConfidenceFill.model_fields) == {"rating", "confidence"}


def test_confidence_card_reason_fill_schema_adds_only_the_two_display_fields():
    assert set(portfolio_manager._ConfidenceCardReasonFill.model_fields) == {
        "rating", "confidence", "card_reason_ko", "card_reason_en",
    }


def test_stop_loss_fill_schema_has_only_rating_and_stop_loss():
    assert set(portfolio_manager._StopLossFill.model_fields) == {"rating", "stop_loss"}


def test_card_reason_fill_schema_has_only_the_two_display_fields_no_rating():
    # No `rating` field — this retry never checked rating agreement even
    # under the old full-schema form (display-only strings).
    assert set(portfolio_manager._CardReasonFill.model_fields) == {"card_reason_ko", "card_reason_en"}


def test_confidence_card_reason_fill_descriptions_match_card_reason_fill():
    # 2026-09-07 audit: _ConfidenceCardReasonFill was added by #221 days after
    # _CardReasonFill, and its card_reason_ko/en descriptions silently
    # dropped the "Example style: ..." exemplars and the "under ~40 words"
    # cap the older schema carries — copy-paste drift between two schemas
    # asking for the same two fields. Pinned so the two can never diverge
    # silently again; a deliberate future difference should update this test
    # with a comment explaining why, not just let it fail.
    for field in ("card_reason_ko", "card_reason_en"):
        assert (
            portfolio_manager._ConfidenceCardReasonFill.model_fields[field].description
            == portfolio_manager._CardReasonFill.model_fields[field].description
        )


async def test_confidence_retry_uses_combined_schema_when_card_reason_also_missing(monkeypatch):
    first = PortfolioDecision(
        rating=PortfolioRating.SELL, executive_summary="orig", investment_thesis="t", confidence=None,
        card_reason_ko=None, card_reason_en=None,
    )
    retried = PortfolioDecision(
        rating=PortfolioRating.SELL, executive_summary="s2", investment_thesis="t2", confidence=0.8,
        card_reason_ko="ko filled", card_reason_en="en filled",
    )
    fake = SequencedStructuredLLM([first, retried])
    monkeypatch.setattr(portfolio_manager, "llm_client", fake)

    await portfolio_manager.decide(_proposal(), _debate(), held=True)

    assert fake.calls[1]["schema"] is portfolio_manager._ConfidenceCardReasonFill
    assert "card_reason" in fake.calls[1]["messages"][-1]["content"]


async def test_confidence_retry_uses_plain_schema_when_card_reason_already_present(monkeypatch):
    first = PortfolioDecision(
        rating=PortfolioRating.SELL, executive_summary="orig", investment_thesis="t", confidence=None,
        card_reason_ko="already here", card_reason_en="already here",
    )
    retried = PortfolioDecision(
        rating=PortfolioRating.SELL, executive_summary="s2", investment_thesis="t2", confidence=0.8,
        card_reason_ko="already here", card_reason_en="already here",
    )
    fake = SequencedStructuredLLM([first, retried])
    monkeypatch.setattr(portfolio_manager, "llm_client", fake)

    await portfolio_manager.decide(_proposal(), _debate(), held=True)

    assert fake.calls[1]["schema"] is portfolio_manager._ConfidenceFill


async def test_confidence_retry_flat_decision_grafts_across_enum_classes(monkeypatch):
    """Regression pin: the narrow schema echoes `rating` as a plain
    `PortfolioRating`, even for a flat decision whose own `rating` is a
    `FlatPortfolioRating` — a different enum class with equal-looking
    members. A naive `retried.rating != decision.rating` (identity/equality
    across enum classes, which is always False for different classes even
    with the same `.value`) would silently discard every flat retry. This
    pins the fix: comparison must be by `.value`, so a same-value echo from
    the narrow schema still grafts."""
    first = FlatPortfolioDecision(
        rating=FlatPortfolioRating.BUY, executive_summary="orig", investment_thesis="t",
        stop_loss=90.0, confidence=None, card_reason_ko="ko", card_reason_en="en",
    )
    # The fake stands in for the narrow schema's real response shape: `rating`
    # comes back as a `PortfolioRating` (the narrow schema's declared type),
    # not a `FlatPortfolioRating`, even though the original decision was flat.
    retried = PortfolioDecision(
        rating=PortfolioRating.BUY, executive_summary="s2", investment_thesis="t2",
        stop_loss=90.0, confidence=0.7, card_reason_ko="ko", card_reason_en="en",
    )
    fake = SequencedStructuredLLM([first, retried])
    monkeypatch.setattr(portfolio_manager, "llm_client", fake)

    result = await portfolio_manager.decide(_proposal(TraderAction.BUY), _debate(), held=False)

    assert result.confidence == 0.7  # grafted, not discarded as a "different rating"


def test_confidence_retry_instructions_do_not_ask_to_reemit_the_whole_decision():
    # The old full-schema instruction said "re-emit that exact same JSON
    # object again, field for field identical" — meaningless (and
    # grammar-contradicting) once the retry schema only has 2-4 fields.
    for message in (
        portfolio_manager._MISSING_CONFIDENCE_RETRY_MESSAGE,
        portfolio_manager._MISSING_CONFIDENCE_AND_CARD_REASON_RETRY_MESSAGE,
        portfolio_manager._MISSING_STOP_LOSS_RETRY_MESSAGE,
        portfolio_manager._MISSING_CARD_REASON_RETRY_MESSAGE,
    ):
        assert "field for field identical" not in message


async def test_decide_prompt_includes_the_degradation_note(monkeypatch):
    decision = PortfolioDecision(
        rating=PortfolioRating.HOLD, executive_summary="s", investment_thesis="t", confidence=0.5,
    )
    fake = SequencedStructuredLLM([decision])
    monkeypatch.setattr(portfolio_manager, "llm_client", fake)

    await portfolio_manager.decide(
        _proposal(), _debate(), held=True,
        period_note="PERIOD_NOTE_MARKER",
        degradation_note="DEGRADATION_NOTE_MARKER",
    )

    prompt = fake.calls[0]["messages"][1]["content"]
    assert "DEGRADATION_NOTE_MARKER" in prompt
    # after period_note, before the trader proposal
    assert prompt.index("PERIOD_NOTE_MARKER") < prompt.index("DEGRADATION_NOTE_MARKER")
    assert prompt.index("DEGRADATION_NOTE_MARKER") < prompt.index("Trader's proposal:")
