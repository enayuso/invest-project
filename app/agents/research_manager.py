"""Research Manager — deep-reasoning judgment node that resolves the
bull/bear debate into a structured investment plan. Anti-hedging: a well-run
debate makes both sides sound compelling by design, so "both sides argued
well" is not itself evidence of genuine balance — even a narrow lean after
the rebuttals should produce a directional stance, not Hold (see
audit-driven fix below).

Fixed 2026-07-29 after a DB audit of app.audit found Hold dominating
resolved decisions (91/172 overall, and specifically ~57/58 of domestic
decisions once TEST_AGGRESSIVE-mode runs — which explicitly bias toward
action — were excluded from the count). Inspecting stored full_transcript
rows showed the actual mechanism: the old wording ("reserve Hold for
situations where the evidence on both sides is genuinely balanced") asks
the model to judge whether the debate LOOKED balanced, and a competent
bull/bear debate almost always looks balanced because that's what both
debators are instructed to produce — so the Hold criterion was satisfied
far more often than intended. The fix reframes the question away from
"does this look balanced" toward "after the rebuttals, is there still any
lean at all" — Hold is for the absence of a lean, not the presence of two
good arguments.
"""

from __future__ import annotations

from app.agents.debater import render_reports
from app.agents.untrusted_content import downstream_injection_warning_for_field, redact_injection_flag
from app.config import settings
from app.llm import llm_client
from app.llm.language import DEFAULT_LANGUAGE
from app.state import DebateState, FlatResearchPlan, PortfolioRating, ResearchPlan

SYSTEM_PROMPT = (
    "You are the Research Manager. Review the bull/bear debate below and produce "
    "an investment recommendation. Weigh which side's arguments actually held up "
    "under rebuttal, not just volume of points. Do not let a side win just because "
    "it cites Wall Street analyst consensus or institutional positioning — that is "
    "one input among many, not a substitute for sound underlying reasoning; a side "
    "that merely restates consensus without engaging the fundamentals/technicals/news "
    "should not out-weigh one that argues from the evidence directly, even if that "
    "evidence disagrees with consensus.\n\n"
    "On committing to a stance: a well-run debate produces strong arguments on BOTH "
    "sides by design — bull and bear are each instructed to make the best case they "
    "can, so 'both sides argued well' is the normal outcome of a healthy debate, not "
    "evidence that the true picture is balanced. Do not default to Hold just because "
    "you can see the merit in both cases. Instead, ask: after the rebuttals, does "
    "either side's case hold up even slightly better? A 55/45 lean is still a lean — "
    "commit to that side (Buy/Overweight if bullish, Underweight/Sell if bearish), "
    "scaled by conviction rather than washed out to Hold. Reserve Hold for the "
    "genuinely rare case where, after weighing the rebuttals, you cannot form any "
    "lean at all — not as a default when the debate was simply close.\n\n"
    "Mandate constraints: this account is LONG-ONLY with NO leverage — no short "
    "selling, no margin, no options, futures, or other derivatives, and no instrument "
    "other than the common stock of this ticker. 'Underweight' and 'Sell' mean "
    "reducing or exiting shares the account already holds; they never mean selling "
    "shares it does not own. Never propose initiating, opening, or adding to a short "
    "position, and never propose a hedge. If your bearish case can only be expressed "
    "as a short, the correct answer is to exit/reduce an existing position, or to "
    "stay out if there is none. (2026-08-28 audit: a recommendation to 'initiate a "
    "short position' in 010170.KS was unexecutable and had to be discarded "
    "downstream.)\n\n"
    "If two of the reports below state materially different magnitudes for the same "
    "quantity — e.g. one calls a contract roughly 20% of annual revenue and another "
    "calls it several times annual revenue — resolve the conflict against the "
    "exchange-reported financial figures and state plainly which one is wrong. Do not "
    "restate one figure and leave the other standing, and do not split the difference. "
    "A news- or headline-sourced amount is the suspect side when it disagrees with the "
    "exchange's own statements. (2026-09-07 audit: a headline's 6.6억원 contract, "
    "reported upstream as \"₩6.6 billion\", became \"nearly five times annualized "
    "revenue\" in a debate turn — a 10x carry-through nobody reconciled.)\n\n"
    f"{downstream_injection_warning_for_field('rationale')}"
)

# Appended only when `held` is False — a flat ticker's recommendation is
# already structurally limited to Buy/Hold (see FlatResearchPlan), but
# telling the model why up front keeps its rationale from arguing toward
# "Underweight"/"Sell"/a short it then can't actually output. The explicit
# "Hold is a committed answer here" framing matters: SYSTEM_PROMPT above
# spends a whole paragraph warning against a wishy-washy default-Hold, and
# without this a flat ticker's Hold could read the same way and push the
# model toward a forced Buy just to avoid it.
_FLAT_SUFFIX = (
    "\n\nThis ticker is currently FLAT (no shares held) — your recommendation can only "
    "be Buy or Hold, not Overweight/Underweight/Sell, since there is no existing "
    "position to add to, trim, or exit. Crucially, this does NOT mean you must be "
    "bullish: a bearish conclusion on a flat ticker is expressed as Hold, which here "
    "means 'stay out / do not enter' — a fully committed, directional answer, not the "
    "wishy-washy Hold the paragraph above warns you off. Only the entry decision is "
    "yours; there is no exit to decide."
)


# 2026-09-16 (engine audit follow-up, item 2): appended only when `held` is
# True. The five-level rating ladder's actual target-allocation meaning
# already exists in both sibling nodes (risk_debators._HELD_RATING_SCALE_NOTE,
# portfolio_manager's SYSTEM_PROMPT "Buy=100%, Overweight=75%..." paragraph)
# but never reached this node — which is the one that PICKS the level, and
# whose pick is then read literally by app.orchestrator.pipeline's
# _consensus_override_tier (bucket membership) and _unjustified_exit_tier
# (the exact string "Underweight"). Worded for this node rather than
# imported from risk_debators: there the rating is a conclusion someone else
# executes, here it is this node's own output field. Deliberately avoids the
# literal word "FLAT" — tests/agents/test_judgment_nodes.py asserts it never
# appears in the held-branch system prompt.
_HELD_RATING_SCALE_NOTE = (
    "\n\nWhat each rating actually commits the account to: the five levels are not a "
    "vague direction, they map to fixed target allocations the execution code applies "
    "automatically — Buy = 100% of full-conviction allocation, Overweight = 75%, Hold = "
    "no change from the current position, Underweight = 25% (a real, non-zero target "
    "position — NOT 'mostly exit' and NOT 0%), Sell = 0% (a full exit). If you mean "
    "'exit this position entirely', the rating is Sell, not Underweight. If you mean "
    "'trim to a smaller but still meaningful holding', the rating is Underweight, not "
    "Sell. Choose the level whose target allocation matches the position you actually "
    "want held after this decision, and write your rationale to match that percentage "
    "rather than what the label might suggest in isolation."
)


def _system_prompt(held: bool) -> str:
    return SYSTEM_PROMPT + (_HELD_RATING_SCALE_NOTE if held else _FLAT_SUFFIX)


async def synthesize(
    reports: dict[str, str],
    debate: DebateState,
    *,
    language: str = DEFAULT_LANGUAGE,
    identity: str = "",
    held: bool = True,
    session_note: str = "",
    period_note: str = "",
    position_context: str = "",
) -> ResearchPlan:
    """`held`: whether the ticker currently has a non-zero position — False
    (flat) structurally restricts the LLM's recommendation to Buy/Hold via
    `FlatResearchPlan` (see app.state's docstring for the 2026-08-28 bug
    this fixes — 33/45 same-day decisions had the Research Manager
    recommending Sell, or even 'initiate a short position', on flat
    tickers), then normalizes the result back into a regular `ResearchPlan`
    so every caller downstream keeps seeing the one familiar type regardless
    of which schema was used — same pattern
    `app.agents.portfolio_manager.decide` already uses for
    PortfolioDecision/FlatPortfolioDecision.

    `position_context` (`app.orchestrator.pipeline._position_context`'s
    rendered text — "Currently holding N shares…" / "Currently flat…") is
    the same block the Trader and the Portfolio Manager already receive (see
    `app.agents.trader.propose`'s own `position_context` docstring). It
    reached every downstream node except this one, even though the pipeline
    deliberately moved the `_position_context` fetch ABOVE this call in
    2026-08-28 so the Research Manager would not reason about position
    blind (2026-09-16 fix). `held` alone says only zero-or-not; the size,
    average cost and the stop already on record are what make the
    difference between Underweight (trim to 25%) and Sell (full exit) a
    real choice rather than a label guess — and that choice is read
    literally by `app.orchestrator.pipeline._unjustified_exit_tier`. This
    block does NOT carry the portfolio WEIGHT fraction that gate compares
    against (`_current_weight`, computed from `PortfolioSnapshot` only after
    the PM has decided) — that remains a PM-side check."""
    reports_text = render_reports(reports, identity=identity)
    session_block = f"{session_note}\n\n" if session_note else ""
    period_block = f"{period_note}\n\n" if period_note else ""
    position_block = f"Current position:\n{position_context}\n\n" if position_context else ""
    prompt = (
        f"{session_block}{period_block}{position_block}Analyst reports:\n{reports_text}\n\n"
        f"Bull/Bear debate transcript:\n{debate.transcript()}"
    )
    messages = [{"role": "system", "content": _system_prompt(held)}, {"role": "user", "content": prompt}]
    # settings.research_manager_reasoning (2026-09-07, AI-engine step-0 latency
    # audit) — True (default) is this call's behavior since it was written;
    # see the setting's own comment for why turning it off is a quality bet,
    # not a pure latency win, and must go through an A/B sweep
    # (app.experiments.reasoning_ab) before the default ever changes.
    if held:
        plan = await llm_client.structured_complete(
            messages,
            ResearchPlan,
            reasoning=settings.research_manager_reasoning,
            label="research_manager",
            language=language,
        )
        plan.rationale = redact_injection_flag(plan.rationale, agent_name="Research Manager")
        return plan
    flat_plan = await llm_client.structured_complete(
        messages,
        FlatResearchPlan,
        reasoning=settings.research_manager_reasoning,
        label="research_manager",
        language=language,
    )
    # ResearchPlan and FlatResearchPlan share every field except
    # `recommendation` (app.state._ResearchPlanFields) — unpacked wholesale
    # rather than listed field-by-field, same reasoning as
    # portfolio_manager.decide's identical normalization (see its own
    # comment for the 2026-08-19 commit 3b96efa incident this shape exists
    # to prevent).
    shared = flat_plan.model_dump(exclude={"recommendation", "rationale"})
    return ResearchPlan(
        recommendation=PortfolioRating(flat_plan.recommendation.value),
        # The only field that's transformed rather than passed through, so
        # it's excluded from the dump above and set explicitly here.
        rationale=redact_injection_flag(flat_plan.rationale, agent_name="Research Manager"),
        **shared,
    )
