"""run_debate — llm_client (convergence check) mocked, no network (D-stage)."""

from __future__ import annotations

from app.orchestrator import debate as debate_mod
from app.state import DebateConvergence, DebateState


def _speaker(label: str):
    calls = []

    async def speak(state: DebateState, party: str) -> str:
        calls.append((party, len(state.turns)))
        return f"{party} says round {state.rounds_completed + 1}"

    speak.calls = calls  # type: ignore[attr-defined]
    return speak


class FakeConvergenceLLM:
    def __init__(self, verdicts: list[bool]):
        self._verdicts = list(verdicts)
        self.calls = 0

    async def structured_complete(self, messages, schema, *, reasoning=False, temperature=0.0, label="", language="ko"):
        self.calls += 1
        return DebateConvergence(converged=self._verdicts.pop(0), reason="test")


async def test_runs_full_schedule_when_never_converged(monkeypatch):
    # 2026-08-25: the convergence check can no longer fire after round 1 —
    # `prior` (turns from before the round just completed) is empty at that
    # point, so the question would have no correct affirmative answer (see
    # run_debate's `2 <= round_num` guard). max_rounds=3 here so there's a
    # round 2 for the check to meaningfully run after.
    fake_llm = FakeConvergenceLLM([False])  # only called after round 2 (not after the final round)
    monkeypatch.setattr(debate_mod, "llm_client", fake_llm)
    state = DebateState(parties=["Bull", "Bear"])
    speakers = {"Bull": _speaker("Bull"), "Bear": _speaker("Bear")}

    result = await debate_mod.run_debate(state=state, speakers=speakers, max_rounds=3)

    assert result.rounds_completed == 3
    assert result.converged is False
    assert len(result.turns) == 6  # 2 parties x 3 rounds
    assert fake_llm.calls == 1  # only after round 2 — round 1 has an empty prior, the final round never checks


async def test_convergence_never_checked_after_round_one(monkeypatch):
    """Regression pin for the 2026-08-25 fix: round 1's "prior rounds" is
    always empty, so a convergence check there is answering a vacuous
    question — verified live to answer True 57.5% of the time on the risk
    debate despite that. Asserts both that the LLM is never called after
    round 1, and that when it IS called (after round 2+), the prompt it
    receives is never the vacuous "(none)" shape."""
    fake_llm = FakeConvergenceLLM([False, False])
    monkeypatch.setattr(debate_mod, "llm_client", fake_llm)
    prompts: list[str] = []

    class RecordingLLM:
        async def structured_complete(self, messages, schema, *, reasoning=False, temperature=0.0, label="", language="ko"):
            prompts.append(messages[0]["content"])
            return DebateConvergence(converged=False, reason="test")

    monkeypatch.setattr(debate_mod, "llm_client", RecordingLLM())
    state = DebateState(parties=["Bull", "Bear"])
    speakers = {"Bull": _speaker("Bull"), "Bear": _speaker("Bear")}

    await debate_mod.run_debate(state=state, speakers=speakers, max_rounds=4)

    assert len(prompts) == 2  # after round 2 and round 3 only — never after round 1, never after the final round
    for prompt in prompts:
        assert "(none)" not in prompt


async def test_two_round_debate_never_calls_convergence_llm(monkeypatch):
    """At the shipped max_rounds=2 (both debates), the convergence check can
    never fire at all — even with check_convergence=True — since round 1 is
    the only round it could possibly run after, and that's exactly the
    vacuous-question case the 2026-08-25 guard excludes."""
    def fail_if_called(*args, **kwargs):
        raise AssertionError("convergence must never be checked when max_rounds=2")

    monkeypatch.setattr(
        debate_mod, "llm_client", type("X", (), {"structured_complete": staticmethod(fail_if_called)})()
    )
    state = DebateState(parties=["Bull", "Bear"])
    speakers = {"Bull": _speaker("Bull"), "Bear": _speaker("Bear")}

    result = await debate_mod.run_debate(state=state, speakers=speakers, max_rounds=2, check_convergence=True)

    assert result.rounds_completed == 2
    assert result.converged is False


async def test_convergence_prompt_states_the_converged_field_semantics(monkeypatch):
    """Guards against a future rewrite silently re-inverting the polarity
    between the prompt QUESTION and DebateConvergence.converged's own
    description (app.state) — the model sees the question text, not the
    schema field's docstring."""
    captured = {}

    class RecordingLLM:
        async def structured_complete(self, messages, schema, *, reasoning=False, temperature=0.0, label="", language="ko"):
            captured["prompt"] = messages[0]["content"]
            return DebateConvergence(converged=False, reason="test")

    monkeypatch.setattr(debate_mod, "llm_client", RecordingLLM())
    state = DebateState(parties=["Bull", "Bear"])
    speakers = {"Bull": _speaker("Bull"), "Bear": _speaker("Bear")}

    await debate_mod.run_debate(state=state, speakers=speakers, max_rounds=3)

    assert "Set converged=true ONLY if" in captured["prompt"]


async def test_stops_early_on_convergence(monkeypatch):
    # Converges after round 2 (the earliest round the check can meaningfully
    # run, per the 2026-08-25 guard) rather than round 1.
    fake_llm = FakeConvergenceLLM([True])
    monkeypatch.setattr(debate_mod, "llm_client", fake_llm)
    state = DebateState(parties=["Bull", "Bear"])
    speakers = {"Bull": _speaker("Bull"), "Bear": _speaker("Bear")}

    result = await debate_mod.run_debate(state=state, speakers=speakers, max_rounds=3)

    assert result.rounds_completed == 2
    assert result.converged is True
    assert len(result.turns) == 4  # rounds 1 and 2 ran; round 3 skipped on convergence


async def test_on_round_callback_invoked_per_round(monkeypatch):
    fake_llm = FakeConvergenceLLM([False])
    monkeypatch.setattr(debate_mod, "llm_client", fake_llm)
    state = DebateState(parties=["Aggressive", "Conservative", "Neutral"])
    speakers = {name: _speaker(name) for name in state.parties}
    seen_rounds = []

    async def on_round(n: int) -> None:
        seen_rounds.append(n)

    await debate_mod.run_debate(state=state, speakers=speakers, max_rounds=2, on_round=on_round)

    assert seen_rounds == [1, 2]


async def test_check_convergence_false_never_calls_llm_and_always_runs_full_schedule(monkeypatch):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("check_convergence=False must never call the model")

    monkeypatch.setattr(debate_mod, "llm_client", type("X", (), {"structured_complete": staticmethod(fail_if_called)})())
    state = DebateState(parties=["Bull", "Bear"])
    speakers = {"Bull": _speaker("Bull"), "Bear": _speaker("Bear")}

    result = await debate_mod.run_debate(
        state=state, speakers=speakers, max_rounds=2, check_convergence=False
    )

    assert result.rounds_completed == 2
    assert result.converged is False
    assert len(result.turns) == 4  # both rounds ran — no early exit is possible without a convergence check


async def test_parallel_first_round_gives_every_party_an_empty_state():
    """With parallel_first_round=True, round 1 must be independent opening
    statements — every party sees zero prior turns (len(state.turns) == 0),
    not just the party that happens to speak first. Round 2 is untouched:
    normal sequential rebuttal against round 1's now-complete turns."""
    state = DebateState(parties=["Bull", "Bear"])
    bull = _speaker("Bull")
    bear = _speaker("Bear")

    result = await debate_mod.run_debate(
        state=state,
        speakers={"Bull": bull, "Bear": bear},
        max_rounds=2,
        check_convergence=False,
        parallel_first_round=True,
    )

    assert bull.calls[0] == ("Bull", 0)
    assert bear.calls[0] == ("Bear", 0)  # not 1 — Bear did NOT see Bull's round-1 turn
    assert result.rounds_completed == 2
    assert len(result.turns) == 4
    assert [t.round for t in result.turns] == [1, 1, 2, 2]
    assert [t.speaker for t in result.turns] == ["Bull", "Bear", "Bull", "Bear"]


async def test_parallel_first_round_false_keeps_sequential_chain():
    """Default behavior (parallel_first_round=False) is unchanged: the
    second party in round 1 already sees the first party's round-1 turn."""
    state = DebateState(parties=["Bull", "Bear"])
    bull = _speaker("Bull")
    bear = _speaker("Bear")

    await debate_mod.run_debate(
        state=state,
        speakers={"Bull": bull, "Bear": bear},
        max_rounds=1,
        check_convergence=False,
    )

    assert bull.calls == [("Bull", 0)]
    assert bear.calls == [("Bear", 1)]  # Bear saw Bull's round-1 turn already appended


async def test_each_party_speaks_once_per_round_in_order():
    state = DebateState(parties=["Bull", "Bear"])
    bull = _speaker("Bull")
    bear = _speaker("Bear")

    async def instant_convergence(messages, schema, *, reasoning=False, temperature=0.0, label="", language="ko"):
        return DebateConvergence(converged=True, reason="done")

    class Inline:
        structured_complete = staticmethod(instant_convergence)

    import app.orchestrator.debate as debate_mod2
    orig = debate_mod2.llm_client
    debate_mod2.llm_client = Inline()
    try:
        result = await debate_mod2.run_debate(state=state, speakers={"Bull": bull, "Bear": bear}, max_rounds=3)
    finally:
        debate_mod2.llm_client = orig

    # Converges after round 2, not round 1 — round 1's convergence check no
    # longer runs at all (see the 2026-08-25 `2 <= round_num` guard).
    assert [t.speaker for t in result.turns] == ["Bull", "Bear", "Bull", "Bear"]
    assert result.converged is True


# ---------------- rounds_completed sequencing invariant (2026-09-07) ----------------
#
# app.agents.debater._current_round derives "which round is this speaker
# about to speak in" as `state.rounds_completed + 1` — used to apply the
# round-1 debate-turn length budget ONLY to round 1, never the final round.
# That derivation is only correct because rounds_completed is set to N AFTER
# every party's round-N turn is appended, in BOTH the sequential and the
# parallel_first_round path. These pin that invariant directly so a future
# reordering of run_debate's loop body breaks a test here, not silently in
# the length-budget logic.


def _round_observer():
    observed = []

    async def speak(state: DebateState, party: str) -> str:
        observed.append(state.rounds_completed + 1)  # what debater._current_round would compute
        return f"{party} turn"

    speak.observed = observed  # type: ignore[attr-defined]
    return speak


async def test_rounds_completed_plus_one_matches_the_actual_round_sequential():
    state = DebateState(parties=["Bull", "Bear"])
    bull = _round_observer()
    bear = _round_observer()

    await debate_mod.run_debate(
        state=state, speakers={"Bull": bull, "Bear": bear}, max_rounds=3, check_convergence=False,
    )

    assert bull.observed == [1, 2, 3]
    assert bear.observed == [1, 2, 3]


async def test_rounds_completed_plus_one_matches_the_actual_round_parallel_first_round():
    state = DebateState(parties=["Bull", "Bear"])
    bull = _round_observer()
    bear = _round_observer()

    await debate_mod.run_debate(
        state=state, speakers={"Bull": bull, "Bear": bear}, max_rounds=2,
        check_convergence=False, parallel_first_round=True,
    )

    assert bull.observed == [1, 2]
    assert bear.observed == [1, 2]
