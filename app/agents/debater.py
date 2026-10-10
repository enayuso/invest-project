"""Shared debater factory used by the bull/bear researchers and the three
risk debators. Each is structurally identical: given the analyst reports (or
trader proposal, for risk) and every other debator's latest turn, produce a
rebuttal — rebutting specific points, not reciting one's own data. Only the
stance/system-prompt text differs per role.
"""

from __future__ import annotations

from app.agents.untrusted_content import DOWNSTREAM_INJECTION_WARNING, redact_injection_flag
from app.config import settings
from app.llm import llm_client
from app.llm.language import DEFAULT_LANGUAGE
from app.state import DebateState, SpeakerFn

# Byte-identical to the instruction every turn used before 2026-09-01 —
# still used whenever a real prior turn exists to rebut (the common case:
# every turn but the very first speaker in round 1). Keeping this string
# untouched, rather than editing it to also cover the no-prior-turn case,
# is what confines this fix's behavior change to opening turns only.
_REBUTTAL_INSTRUCTION = (
    "Give your argument, rebutting the specific points made above where "
    "applicable. Do not simply recite your own data if it's already been "
    "covered — engage directly with what was just said. If one of the "
    "points made against you is genuinely strong, acknowledge it "
    "explicitly rather than ignoring it, then explain why your overall "
    "position still holds despite it — or, if it genuinely doesn't, say "
    "so. A rebuttal that never concedes anything reads as scripted, not "
    "reasoned."
)

# 2026-09-01, TradingAgents v0.4.0 #1176 comparison: the instruction above is
# unconditional — it tells a debator to "rebut the specific points made
# above" and "engage with what was just said" even when `others_text` is the
# "(no one else has spoken yet)" placeholder, i.e. the FIRST speaker in round
# 1 of either debate this factory serves (Bull in the bull/bear debate,
# Aggressive in the risk debate — `debate_parallel_first_round=False` by
# default, so round 1 is a sequential chain and exactly one party per debate
# opens blind). A model given nothing to rebut but told to rebut invents what
# the other side would have said and argues against THAT — and the invented
# argument then sits in the transcript as if someone actually said it, read
# by Research Manager / Portfolio Manager as real content. Same failure
# shape TradingAgents fixed in their opening-turn prompt.
#
# This is the opener's actual task instead: state a position from scratch,
# with the same steelmanning discipline `bull_researcher`/`bear_researcher`'s
# own SYSTEM_PROMPT already applies at that layer — anchored explicitly to
# "found in the reports themselves" (never to an imagined opponent), which is
# why that phrasing is reused verbatim here rather than invented fresh.
_OPENING_INSTRUCTION = (
    "No one else has spoken yet this round, so there is nothing above to "
    "rebut. Do not imagine what another debator might argue and respond to "
    "that — an invented argument would be read by every later participant "
    "as something someone actually said. Instead, state your position from "
    "scratch, built from the evidence in the context above, laid out as a "
    "clear, ordered case. Before concluding, acknowledge the single "
    "strongest point against your own position found in the reports "
    "themselves, then explain why it doesn't change your conclusion — "
    "ignoring the best counter-evidence is evasive, not persuasive."
)


# 2026-09-15 engine audit. `state.latest_except(party)` excludes the
# speaker's OWN turns (see app.state.DebateState.latest_except), so from
# round 2 on every debator argues without seeing the position it took in
# round 1. For the risk-debate Neutral that is a plain contradiction: its own
# prompt says "if you already stated a verdict in an earlier round of this
# debate ... If there is no such fact, say that your earlier verdict stands
# and restate it" (app.agents.risk_debators) — an instruction to compare
# against a turn the node cannot see, in a 3-party sequential debate where
# Neutral speaks last and therefore sees only the other two parties' round-2
# turns. `check_neutral_reversal`'s measured `new_fact_cited == 0/10` is what
# a node that cannot see "then" looks like, not disobedience; and the
# Portfolio Manager promotes that same final Neutral turn to
# "=== FINAL SYNTHESIS ===" as the sole basis for its rating, stop-loss and
# sizing (app.agents.portfolio_manager._render_risk_debate).
#
# Supplied as DATA, not as another instruction: the instruction already
# exists downstream, and this desk's 2026-09-15 억원/billion fix is its own
# evidence that handing a model the missing material beats restating the
# rule at it. "not new evidence" is explicit because the Neutral prompt's
# reversal test turns on what a LATER turn introduced — a debator's own
# prior words must never be read as the new fact that licenses a reversal.
_OWN_TURN_HEADER = (
    "Your own most recent turn in this debate (you wrote this — it is not "
    "another debator's argument, and it is not new evidence):"
)


def _current_round(state: DebateState) -> int:
    """The round about to be spoken in, derived from `state` rather than
    threaded as a separate parameter — `app.orchestrator.debate.run_debate`
    sets `state.rounds_completed = round_num` only AFTER every party's turn
    for that round has been appended (both the sequential and the
    `parallel_first_round` path — see its own `for round_num in
    range(1, max_rounds + 1)` loop), so a speaker called DURING round N
    always observes `rounds_completed == N - 1`. Pinned by
    tests/orchestrator/test_debate.py."""
    return state.rounds_completed + 1


def make_speaker(system_prompt: str, context: str, *, label: str = "", language: str = DEFAULT_LANGUAGE) -> SpeakerFn:
    full_system_prompt = f"{system_prompt}\n\n{DOWNSTREAM_INJECTION_WARNING}"

    async def speak(state: DebateState, party: str) -> str:
        others = state.latest_except(party)
        others_text = (
            "\n\n".join(f"[{t.speaker}]: {t.content}" for t in others)
            if others
            else "(no one else has spoken yet)"
        )
        instruction = (
            _REBUTTAL_INSTRUCTION
            if others or not settings.debate_opening_turn_prompt
            else _OPENING_INSTRUCTION
        )
        # Chronological order: what this speaker already argued comes before
        # what the others have since said, so "the specific new fact a later
        # turn introduced" (the Neutral prompt's reversal test) reads as an
        # actual before/after rather than two undated blocks. Absent in round
        # 1 — `latest_by` returns None and no block is rendered, so an opening
        # turn is byte-identical to before this change.
        own = state.latest_by(party) if settings.debate_own_previous_turn else None
        own_block = f"{_OWN_TURN_HEADER}\n{own}\n\n" if own else ""
        prompt = (
            f"{context}\n\n"
            f"{own_block}"
            f"Other debators' latest arguments:\n{others_text}\n\n"
            f"{instruction}"
        )
        # Round-1 debate-turn budget (2026-09-07, AI-engine step-0 latency
        # audit): caps ONLY the opening/pre-rebuttal round's completion
        # tokens — never the final round a risk-debate Neutral turn is
        # promoted to "FINAL SYNTHESIS" from
        # (app.agents.portfolio_manager._render_risk_debate) or that the
        # #205 consensus-override gate and
        # app.agents.risk_debators.check_neutral_reversal's FINAL-round
        # parse depend on. `0` (or any round but 1) disables the override
        # entirely — see settings.debate_round1_max_output_tokens's own
        # comment for the chosen value and its trade-offs.
        budget = settings.debate_round1_max_output_tokens
        max_tokens = budget if (budget > 0 and _current_round(state) == 1) else None
        response = await llm_client.complete(
            [{"role": "system", "content": full_system_prompt}, {"role": "user", "content": prompt}],
            reasoning=False,
            label=label,
            language=language,
            max_tokens=max_tokens,
        )
        return redact_injection_flag(response.content, agent_name=label or "debater")

    return speak


def render_reports(reports: dict[str, str], *, identity: str = "") -> str:
    """`identity`, if given, is a verified ticker-name anchor (see
    app.orchestrator.pipeline's `identity_note`) prefixed ahead of the
    reports themselves — so a debater told "don't dismiss content over a
    name mismatch" reads that instruction before, not after, any report
    that might actually contain a mismatched name (the 2026-08-20 316140.KS
    incident: the Bear Researcher dismissed real news because the
    Fundamentals report named the wrong company)."""
    body = "\n\n".join(f"### {name}\n{text}" for name, text in reports.items())
    return f"{identity}\n\n{body}" if identity else body
