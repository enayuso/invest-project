"""Trader — turns the Research Manager's plan into a concrete transaction
proposal. Explicitly given the four analyst reports and the full debate
history: TradingAgents' trader prompt claims to anchor on "the analysts'
reports" but those reports are never actually injected into its context —
this is the fix for that prompt/context mismatch.

Fixed 2026-07-29 after a live WDC decision showed the Trader quietly
overturning the Research Manager's Underweight into Hold — not because of
any new evidence, but because the old prompt's "surface real disagreement
instead of smoothing over it" told the Trader to re-weigh the same bull/bear
conflict the Research Manager had already resolved. Its own stated reasoning
re-derived nearly the same points (stale-consensus fundamentals vs. real
technical/insider-selling weakness) research_manager.py had just used to
justify Underweight, then landed on Hold anyway — the same "both sides
argued well -> Hold" collapse research_manager.py's own 2026-07-29 fix
targeted, just recreated one node downstream. The Research Manager's plan is
now the Trader's default; deviating requires naming a genuinely new point
the debate raised that the plan didn't address, not re-litigating a
conflict that was already weighed.
"""

from __future__ import annotations

from app.agents.debater import render_reports
from app.agents.untrusted_content import downstream_injection_warning_for_field, redact_injection_flag
from app.config import AIProfile
from app.llm import llm_client
from app.llm.language import DEFAULT_LANGUAGE
from app.state import (
    DebateState,
    FlatTraderProposal,
    ResearchPlan,
    TraderAction,
    TraderProposal,
    render_research_plan,
)

SYSTEM_PROMPT = (
    "You are the Trader. Turn the Research Manager's investment plan into a "
    "concrete transaction proposal, anchored in the analysts' reports and the "
    "bull/bear debate below — not just the plan in isolation.\n\n"
    "The Research Manager already weighed the bull/bear tension and committed to "
    "a stance — treat that conclusion as your starting point, not something to "
    "re-litigate from scratch. Re-deriving the same conflict the plan already "
    "resolved (the same bull points vs. the same bear points) and landing on a "
    "softer/more neutral action than the plan is not 'surfacing disagreement' — "
    "it's silently re-introducing the exact indecision the Research Manager step "
    "exists to resolve, one node later. Only deviate from the plan's direction "
    "when the debate transcript below raises something genuinely NEW that the "
    "plan's rationale never addressed (e.g. a rebuttal point made after the plan "
    "was written, or a specific fact the plan's rationale overlooked) — in that "
    "case, name that new point explicitly in your reasoning. If you can't name "
    "one, follow the plan's direction (adjusting only size/entry/stop for "
    "execution mechanics), don't soften it toward Hold.\n\n"
    "What the plan's recommendation actually targets: Buy=100% of full-conviction "
    "allocation, Overweight=75%, Hold=no change from the current position, "
    "Underweight=25% (a real, non-zero target position — NOT 'mostly exit' and NOT "
    "0%), Sell=0% (a full exit). Your own `action` field only has three values "
    "(Buy/Hold/Sell) — it captures DIRECTION, not the plan's target size. When the "
    "plan says Underweight, your action is Sell (the only decrease value available "
    "to you), but that does not make the plan's intent a full exit: describe the "
    "target in `position_sizing`/`reasoning` as a trim toward the plan's real "
    "target level, not as 'full exit', 'liquidate', 'close the position', or a move "
    "to 0 shares. Reserve those full-exit phrasings for when the plan itself says "
    "Sell (0%). Getting the direction right but overstating the size is a real "
    "execution error, not a harmless simplification.\n\n"
    "Mandate constraints: this account is LONG-ONLY with NO leverage — no short "
    "selling, no margin, no options, futures, or other derivatives, and no instrument "
    "other than the common stock of this ticker. 'Sell' means reducing or exiting "
    "shares the account already holds; it never means selling shares it does not "
    "own. Never propose initiating, opening, or adding to a short position, and "
    "never propose a hedge. If the plan's bearish case can only be expressed as a "
    "short, the correct answer is to exit/reduce an existing position, or to stay "
    "out if there is none. (2026-08-28 audit: a recommendation to 'initiate a "
    "short position' in 010170.KS was unexecutable and had to be discarded "
    "downstream.)\n\n"
    f"{downstream_injection_warning_for_field('reasoning')}"
)

# AIProfile suffixes — persona/verbosity only, appended to SYSTEM_PROMPT above.
# See app.config.AIProfile: never touches RiskManager/Broker/Approval.
# 프로필별 페르소나 suffix — 판단 성향만 바꿈, 리스크/브로커/승인에는 영향 없음.
_PROFILE_SUFFIXES: dict[AIProfile, str] = {
    AIProfile.PRODUCTION: "",
    AIProfile.TEST_AGGRESSIVE: (
        "\n\nTEST_AGGRESSIVE mode: this run exists to exercise the order pipeline end-to-end, "
        "not to produce investment-quality calls. Minimize Hold — commit to whichever "
        "directional actions are actually available to you below (a flat ticker only offers "
        "Buy or Hold, and its Hold means 'stay out') whenever the evidence gives even a modest "
        "lean, so downstream approval/execution paths actually see varied order events."
    ),
    AIProfile.DEBUG: (
        "\n\nDEBUG mode: write `reasoning` with maximum detail — name every indicator/report you "
        "weighed, state your confidence level explicitly, and walk through the reasoning steps, "
        "not just the conclusion."
    ),
}

# Appended only when `held` is False — a flat ticker's action is already
# structurally limited to Buy/Hold (see FlatTraderProposal), but telling the
# model why up front keeps its reasoning from arguing toward a Sell it then
# can't actually output. Same "Hold is a committed answer here" framing as
# research_manager._FLAT_SUFFIX, for the same reason.
_FLAT_SUFFIX = (
    "\n\nThis ticker is currently FLAT (no shares held) — your action can only be Buy "
    "or Hold, not Sell, since there are no shares to sell and this system never opens "
    "short positions. If the plan's direction is bearish, Hold is the correct, fully "
    "committed answer — it does not mean you failed to follow the plan."
)


def _system_prompt(profile: AIProfile, held: bool) -> str:
    return SYSTEM_PROMPT + _PROFILE_SUFFIXES[profile] + ("" if held else _FLAT_SUFFIX)


async def propose(
    reports: dict[str, str],
    debate: DebateState,
    plan: ResearchPlan,
    position_context: str = "",
    *,
    language: str = DEFAULT_LANGUAGE,
    profile: AIProfile = AIProfile.PRODUCTION,
    identity: str = "",
    held: bool = True,
    session_note: str = "",
    period_note: str = "",
) -> TraderProposal:
    """`position_context` (app.orchestrator.pipeline._position_context's
    rendered text — "Currently holding N shares..." / "Currently flat...")
    grounds the proposal in the REAL current position, same as the
    Portfolio Manager already gets. Without it the Trader's `reasoning`/
    `position_sizing` prose has no way to know whether this is a fresh
    entry or an add-to/trim of an existing position, and could describe
    one while the actual position is the other — the code-level sizing
    (app.execution.ledger.rebalance_to_target) is unaffected either way
    (it always uses the real figure), but the proposal text itself would
    read wrong, the same "code right, prose wrong" pattern already seen
    with ATR-based sizing (see _atr_context's docstring).

    `held`: whether the ticker currently has a non-zero position — False
    (flat) structurally restricts the LLM's action to Buy/Hold via
    `FlatTraderProposal` (see app.state's docstring for the 2026-08-28 bug
    this fixes — 20 of 45 same-day decisions had the Trader proposing Sell
    on a flat ticker), then normalizes the result back into a regular
    `TraderProposal` so every caller downstream keeps seeing the one
    familiar type regardless of which schema was used — same pattern
    `app.agents.research_manager.synthesize`/
    `app.agents.portfolio_manager.decide` already use for their own
    held/flat schema pairs."""
    reports_text = render_reports(reports, identity=identity)
    session_block = f"{session_note}\n\n" if session_note else ""
    period_block = f"{period_note}\n\n" if period_note else ""
    position_block = f"Current position:\n{position_context}\n\n" if position_context else ""
    prompt = (
        f"{session_block}"
        f"{period_block}"
        f"{position_block}"
        f"Analyst reports:\n{reports_text}\n\n"
        f"Bull/Bear debate transcript:\n{debate.transcript()}\n\n"
        f"Research Manager's plan:\n{render_research_plan(plan)}"
    )
    messages = [{"role": "system", "content": _system_prompt(profile, held)}, {"role": "user", "content": prompt}]
    if held:
        proposal = await llm_client.structured_complete(
            messages,
            TraderProposal,
            reasoning=False,
            label="trader",
            language=language,
        )
        proposal.reasoning = redact_injection_flag(proposal.reasoning, agent_name="Trader")
        return proposal
    flat_proposal = await llm_client.structured_complete(
        messages,
        FlatTraderProposal,
        reasoning=False,
        label="trader",
        language=language,
    )
    # TraderProposal and FlatTraderProposal share every field except
    # `action` (app.state._TraderProposalFields) — unpacked wholesale rather
    # than listed field-by-field, same reasoning as
    # research_manager.synthesize/portfolio_manager.decide's identical
    # normalization (see either's own comment for the 2026-08-19 commit
    # 3b96efa incident this shape exists to prevent).
    shared = flat_proposal.model_dump(exclude={"action", "reasoning"})
    return TraderProposal(
        action=TraderAction(flat_proposal.action.value),
        # The only field that's transformed rather than passed through, so
        # it's excluded from the dump above and set explicitly here.
        reasoning=redact_injection_flag(flat_proposal.reasoning, agent_name="Trader"),
        **shared,
    )
