"""app.state's pydantic model validators — no network/DB (A-stage)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.state import (
    FlatPortfolioDecision,
    PortfolioDecision,
    PortfolioRating,
    is_blank_card_reason,
    render_portfolio_decision,
)


def test_flat_and_held_decision_share_the_same_field_set():
    # app.agents.portfolio_manager.decide's flat-decision normalization
    # unpacks FlatPortfolioDecision.model_dump() straight into
    # PortfolioDecision(**shared) rather than listing fields by name — that
    # only works because the two classes share every field except `rating`
    # (both subclass app.state._PortfolioDecisionFields). If a field is ever
    # added to only one side, pydantic's default extra="ignore" would let
    # model_dump() silently drop it rather than raise, so this pin exists to
    # fail loudly the moment the two field sets diverge (this is exactly how
    # card_reason_ko/card_reason_en went missing for every flat ticker on
    # 2026-08-19 until the normalization site was fixed to unpack shared
    # fields wholesale instead of naming them one by one).
    assert set(FlatPortfolioDecision.model_fields) == set(PortfolioDecision.model_fields)


# ---------------------------------------------------------------------------
# _strip_rendered_header (executive_summary) — 2026-08-19 fix for the model
# occasionally writing the whole rendered decision format INSIDE
# executive_summary, then the renderer wrapping it again (real example:
# decision id 930).
# ---------------------------------------------------------------------------

_DOUBLED_HEADER_EXAMPLE = (
    "**Rating**: Hold\n\n"
    "**Executive Summary**: **Rating**: Hold\n\n"
    "**Executive Summary**: \n"
    "The Final Synthesis decisively validates the thesis and the position is held."
)


def test_strip_rendered_header_collapses_the_real_doubled_example():
    decision = PortfolioDecision(
        rating=PortfolioRating.HOLD,
        executive_summary=_DOUBLED_HEADER_EXAMPLE,
        investment_thesis="t",
    )
    assert decision.executive_summary == (
        "The Final Synthesis decisively validates the thesis and the position is held."
    )


def test_strip_rendered_header_leaves_a_clean_summary_untouched():
    clean = "Hold on strong fundamentals. Watch for earnings miss. Maintain position, review in 3 months."
    decision = PortfolioDecision(
        rating=PortfolioRating.HOLD,
        executive_summary=clean,
        investment_thesis="t",
    )
    assert decision.executive_summary == clean


def test_strip_rendered_header_only_fails_validation_when_nothing_but_a_header():
    with pytest.raises(ValidationError):
        PortfolioDecision(
            rating=PortfolioRating.HOLD,
            executive_summary="**Rating**: Hold\n\n**Executive Summary**:",
            investment_thesis="t",
        )


def test_strip_rendered_header_ignores_marker_not_at_the_start():
    text = "Solid quarter overall. See **Rating**: Hold for the prior call. Maintain the position."
    decision = PortfolioDecision(
        rating=PortfolioRating.HOLD,
        executive_summary=text,
        investment_thesis="t",
    )
    assert decision.executive_summary == text


# ---------------------------------------------------------------------------
# _recover_fields_from_rendered_header (executive_summary) — 2026-08-21 fix
# for the same failure mode as _strip_rendered_header above, but with
# different leading markers AND actual data loss: the model left the real
# `stop_loss`/`confidence` fields empty and wrote their values as rendered
# header lines at the start of executive_summary instead (real example:
# decision id 950, 950160.KS — the AI's own stated 18,870 stop-loss never
# reached `resolve_stop_loss`, which fell back to an ATR-floor stop).
# ---------------------------------------------------------------------------

_STOP_LOSS_HEADER_EXAMPLE = (
    "**Stop Loss**: 18,870\n"
    "**Confidence**: 0.65\n\n"
    "Although rated Buy, this re-entry will execute as a tightly constrained "
    "trial position given the recent force-exit."
)


def test_recover_fields_collapses_the_real_stop_loss_header_example():
    decision = PortfolioDecision(
        rating=PortfolioRating.BUY,
        executive_summary=_STOP_LOSS_HEADER_EXAMPLE,
        investment_thesis="t",
    )
    assert decision.executive_summary == (
        "Although rated Buy, this re-entry will execute as a tightly constrained "
        "trial position given the recent force-exit."
    )
    assert decision.stop_loss == pytest.approx(18870.0)
    assert decision.confidence == pytest.approx(0.65)


def test_recover_fields_never_overwrites_a_value_already_set_on_its_own_field():
    decision = PortfolioDecision(
        rating=PortfolioRating.BUY,
        executive_summary=_STOP_LOSS_HEADER_EXAMPLE,
        investment_thesis="t",
        stop_loss=15_000.0,
    )
    assert decision.stop_loss == pytest.approx(15_000.0)
    # Still stripped from the prose even though this field wasn't touched --
    # leftover header noise is worth cleaning up regardless.
    assert "**Stop Loss**" not in decision.executive_summary


def test_recover_fields_ignores_a_marker_not_at_the_start():
    text = "This looks fine. Stop Loss: 5000 is what a trader might casually mention mid-sentence."
    decision = PortfolioDecision(
        rating=PortfolioRating.BUY,
        executive_summary=text,
        investment_thesis="t",
    )
    assert decision.executive_summary == text
    assert decision.stop_loss is None


def test_recover_fields_strips_but_does_not_backfill_an_ambiguous_range():
    text = "**Stop Loss**: 18,000-19,000\n\nBody text here."
    decision = PortfolioDecision(
        rating=PortfolioRating.BUY,
        executive_summary=text,
        investment_thesis="t",
    )
    assert decision.executive_summary == "Body text here."
    assert decision.stop_loss is None  # too ambiguous to trust -- falls through to the ATR floor as before


def test_recover_fields_accepts_a_percentage_confidence():
    text = "**Confidence**: 65%\n\nBody text here."
    decision = PortfolioDecision(
        rating=PortfolioRating.BUY,
        executive_summary=text,
        investment_thesis="t",
    )
    assert decision.confidence == pytest.approx(0.65)


def test_recover_fields_never_backfills_rating_from_a_leading_header():
    # **Rating** is deliberately excluded from the whitelist -- handled
    # separately by _strip_rendered_header with no backfill by design,
    # since rating is direction-deciding data that must never be inferred
    # from prose.
    decision = PortfolioDecision(
        rating=PortfolioRating.BUY,
        executive_summary=_DOUBLED_HEADER_EXAMPLE,
        investment_thesis="t",
    )
    assert decision.rating == PortfolioRating.BUY


def test_recover_fields_is_idempotent_across_flat_to_held_normalization():
    # app.agents.portfolio_manager's flat-decision normalization re-runs
    # this model's validators a second time via PortfolioDecision(**shared)
    # (see test_flat_and_held_decision_share_the_same_field_set above) --
    # a second pass over already-recovered fields/already-stripped prose
    # must be a no-op, not a re-parse or a second (wrong) recovery attempt.
    first = PortfolioDecision(
        rating=PortfolioRating.BUY,
        executive_summary=_STOP_LOSS_HEADER_EXAMPLE,
        investment_thesis="t",
    )
    second = PortfolioDecision(**first.model_dump())
    assert second.executive_summary == first.executive_summary
    assert second.stop_loss == first.stop_loss
    assert second.confidence == first.confidence


# ---------------------------------------------------------------------------
# card_reason_ko / card_reason_en — optional dashboard-card fields, added
# alongside executive_summary. Must stay optional: structured_complete
# re-raises if its unconstrained fallback also fails schema validation, so a
# required field here could hard-fail a real trading decision over a
# display string (see app.state's own comment on these fields).
# ---------------------------------------------------------------------------


def test_card_reason_fields_default_to_none_and_still_validate():
    decision = PortfolioDecision(
        rating=PortfolioRating.HOLD,
        executive_summary="Hold on strong fundamentals. Watch for earnings miss. Maintain position.",
        investment_thesis="t",
    )
    assert decision.card_reason_ko is None
    assert decision.card_reason_en is None


@pytest.mark.parametrize(
    "value",
    [None, "", "  ", "N/A", "n/a", "NONE", "Null", "-", "tbd", "TBD", "\t\n"],
)
def test_is_blank_card_reason_true_for_placeholders(value):
    assert is_blank_card_reason(value) is True


@pytest.mark.parametrize(
    "value",
    ["외국인 5거래일 연속 순매수", "RSI 74", "Foreign investors net-bought", "N/A watch list"],
)
def test_is_blank_card_reason_false_for_real_text(value):
    assert is_blank_card_reason(value) is False


def test_render_portfolio_decision_excludes_card_reason_fields():
    decision = PortfolioDecision(
        rating=PortfolioRating.HOLD,
        executive_summary="Hold on strong fundamentals. Watch for earnings miss. Maintain position.",
        investment_thesis="t",
        card_reason_ko="외국인 5거래일 연속 순매수, HBM 모멘텀 유효.",
        card_reason_en="Foreign investors net-bought for 5 straight sessions.",
    )
    rendered = render_portfolio_decision(decision)
    assert "외국인" not in rendered
    assert "Foreign investors net-bought" not in rendered
