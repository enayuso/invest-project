"""Generic bounded N-party debate loop with convergence-aware early stop.

Used for both the bull/bear stage (2 parties) and the risk debate stage (3
parties). Each party sees every other party's latest turn, not just the
immediately-prior speaker. After each full round, a small structured LLM
judgment call decides whether the round added any new substance — per
[[feedback_language_agnostic]] this is a model judgment, not a keyword or
similarity heuristic, so it stays correct in any language. The loop exits
early on convergence instead of always running the full fixed schedule.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable

from app.llm import llm_client
from app.llm.language import DEFAULT_LANGUAGE
from app.state import DebateConvergence, DebateState, DebateTurn, RoundCallback, SpeakerFn


async def run_debate(
    *,
    state: DebateState,
    speakers: dict[str, SpeakerFn],
    max_rounds: int,
    on_round: RoundCallback | None = None,
    language: str = DEFAULT_LANGUAGE,
    check_convergence: bool = True,
    parallel_first_round: bool = False,
    turn_guard: Callable[[str, str, int], str] | None = None,
) -> DebateState:
    """`check_convergence=False` skips the extra `debate_convergence` LLM
    call after every non-final round and always runs the full `max_rounds`
    schedule. Both shipped debates set it False and both for measured
    reasons. Bull/Bear (`a66d19c`): flagged "converged" only 4% of the time
    over 200 live decisions (2026-08-21) — one call per decision bought an
    ~0.08-call expected saving. Risk debate (2026-08-25): its apparent 57.5%
    convergence rate turned out to be an artifact — at max_rounds=2 the check
    could only ever run after round 1, where the "prior rounds" it compares
    against are empty, so the question had no correct affirmative answer. See
    the `2 <= round_num` guard below.

    `parallel_first_round=True` (2026-08-31 structural profiling, both
    debates measured at 56.3% of pipeline wall-clock) runs round 1's parties
    concurrently instead of one-at-a-time. This is a real semantic change,
    not just a speed-up: today, round 1 is ALREADY a sequential chain —
    `debater.make_speaker`'s `state.latest_except(party)` means the second
    (and third) party in round 1 sees the first party's round-1 argument and
    rebuts it, even though nothing calls it a "rebuttal round" until round 2.
    Running round 1 in parallel makes every party open blind (matching what
    the FIRST speaker already experiences today), turning round 1 into
    independent opening statements; round 2+ is untouched and still gives
    every party the full set of round-1 turns to rebut, so the debate still
    ends in a fully cross-examined final round. Gated by
    `settings.debate_parallel_first_round` (default False) — ship this only
    after an explicit A/B check that round-1 quality doesn't regress without
    the early rebuttal, per [[qwen38-flash-next-vs-qwen36-comparison-2026-08-28]]'s
    structural-improvement item ①.

    `turn_guard(content, speaker, round)`, if given, is applied to each turn's
    content BEFORE the `DebateTurn` is appended to `state.turns` — which is
    the only correct place for it: `debater.make_speaker`'s
    `state.latest_except(party)` hands the STORED turn to the next speaker,
    and `portfolio_manager._render_risk_debate` promotes the final turn, so a
    guard that ran after `run_debate` returned would see corruption that has
    already propagated. Applied on both the sequential and the
    `parallel_first_round` path. Default `None` is a strict no-op.
    2026-09-01: decision id=1157's risk debate published "-785.2 billion KRW"
    (10x wrong) in a round-1 turn that then fed round 2 and the PM."""
    def _guarded(content: str, party: str, round_num: int) -> str:
        return turn_guard(content, party, round_num) if turn_guard is not None else content

    for round_num in range(1, max_rounds + 1):
        if on_round is not None:
            await on_round(round_num)
        if parallel_first_round and round_num == 1:
            contents = await asyncio.gather(*(speakers[party](state, party) for party in state.parties))
            for party, content in zip(state.parties, contents):
                state.turns.append(DebateTurn(
                    speaker=party, content=_guarded(content, party, round_num), round=round_num
                ))
        else:
            for party in state.parties:
                content = await speakers[party](state, party)
                state.turns.append(DebateTurn(
                    speaker=party, content=_guarded(content, party, round_num), round=round_num
                ))
        state.rounds_completed = round_num

        # `2 <=` is load-bearing, not defensive. `_check_convergence` compares
        # the latest round against `prior = [t for t in state.turns if t.round
        # < state.rounds_completed]`, which after round 1 is EMPTY — the model
        # was asked "did round 1 add anything beyond (none)?", a question with
        # no correct `converged=True` answer. It nonetheless answered True
        # 57.5% of the time on the live risk debate (200 decisions,
        # 2026-08-21), truncating a 3-party debate to one round: Aggressive
        # argued with zero prior context, Conservative rebutted once, and
        # Neutral — whose final-round turn `portfolio_manager.
        # _render_risk_debate` promotes to "=== FINAL SYNTHESIS ===" — never
        # revised after hearing the rebuttal. That 57.5% measured the vacuous
        # question, not real convergence.
        #
        # Both shipped debates run max_rounds=2 (app.config), so this guard
        # means the check cannot fire at all today. It stays so that raising
        # max_risk_rounds to 3+ re-enables a MEANINGFUL check automatically
        # rather than silently restoring the vacuous one. Same conclusion the
        # Bull/Bear debate reached in `a66d19c`, arrived at for a second,
        # independent reason.
        if check_convergence and 2 <= round_num < max_rounds:
            convergence = await _check_convergence(state, language=language)
            if convergence.converged:
                state.converged = True
                break

    return state


async def _check_convergence(state: DebateState, *, language: str = DEFAULT_LANGUAGE) -> DebateConvergence:
    latest_round = [t for t in state.turns if t.round == state.rounds_completed]
    prior = [t for t in state.turns if t.round < state.rounds_completed]
    prompt = (
        "Debate so far (prior rounds):\n"
        f"{_render(prior) or '(none)'}\n\n"
        "Most recent round:\n"
        f"{_render(latest_round)}\n\n"
        # Phrased in terms of the field's own semantics rather than as a
        # question whose natural "yes" means converged=False. The previous
        # wording ("Did the most recent round introduce any materially new
        # argument...?") was inverted relative to `DebateConvergence.
        # converged`'s description in app.state, and the model sees the
        # question text far more prominently than the schema field's
        # description — an independent way to get an implausible convergence
        # rate in EITHER direction, on top of the empty-prior defect the
        # `2 <=` guard above fixes.
        "Set converged=true ONLY if the most recent round merely restates the "
        "prior rounds — no materially new argument, rebuttal, evidence, or "
        "concession beyond what was already said. Set converged=false if the "
        "most recent round added any new substance. If the prior-rounds "
        "section above is empty, there is nothing for the latest round to "
        "have converged with: set converged=false."
    )
    return await llm_client.structured_complete(
        [{"role": "user", "content": prompt}],
        DebateConvergence,
        reasoning=False,
        label="debate_convergence",
        language=language,
    )


def _render(turns: list[DebateTurn]) -> str:
    return "\n\n".join(f"[{t.speaker}]\n{t.content}" for t in turns)
