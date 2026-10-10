"""make_speaker/render_reports plus the bull/bear/risk debator builders that
sit on top of them — llm_client mocked, no network (C-stage)."""

from __future__ import annotations

import app.agents.bear_researcher as bear_researcher
import app.agents.bull_researcher as bull_researcher
import app.agents.risk_debators as risk_debators
from app.agents import debater
from app.llm.client import LLMMessage
from app.state import DebateState, DebateTurn, TraderAction, TraderProposal


class FakeLLM:
    def __init__(self, content: str = "a rebuttal"):
        self._content = content
        self.calls: list[dict] = []

    async def complete(self, messages, *, tools=None, reasoning=False, temperature=0.0, label="", language="ko", max_tokens=None):
        self.calls.append({"messages": messages, "max_tokens": max_tokens})
        return LLMMessage(content=self._content)


def test_render_reports_joins_named_sections():
    result = debater.render_reports({"Market": "technical view", "Fundamentals": "financial view"})
    assert "### Market\ntechnical view" in result
    assert "### Fundamentals\nfinancial view" in result


def test_render_reports_prefixes_identity_note_when_given():
    result = debater.render_reports({"Market": "technical view"}, identity="Ticker anchor text")
    assert result.startswith("Ticker anchor text")
    assert "### Market\ntechnical view" in result


def test_render_reports_omits_identity_block_when_not_given():
    result = debater.render_reports({"Market": "technical view"})
    assert result == "### Market\ntechnical view"


async def test_make_speaker_first_round_notes_no_prior_speech(monkeypatch):
    fake = FakeLLM()
    monkeypatch.setattr(debater, "llm_client", fake)
    speak = debater.make_speaker("You are a debator.", "context text")
    state = DebateState(parties=["Bull", "Bear"])
    result = await speak(state, "Bull")
    assert result == "a rebuttal"
    prompt = fake.calls[0]["messages"][1]["content"]
    assert "(no one else has spoken yet)" in prompt
    assert "context text" in prompt


async def test_make_speaker_includes_other_parties_latest_turn(monkeypatch):
    fake = FakeLLM()
    monkeypatch.setattr(debater, "llm_client", fake)
    speak = debater.make_speaker("You are a debator.", "context text")
    state = DebateState(parties=["Bull", "Bear"])
    state.turns.append(DebateTurn(speaker="Bear", content="bear's opening argument", round=1))
    await speak(state, "Bull")
    prompt = fake.calls[0]["messages"][1]["content"]
    assert "[Bear]: bear's opening argument" in prompt


async def test_make_speaker_prompt_asks_to_acknowledge_strong_counterpoints(monkeypatch):
    # Rebuttal-path assertion — needs a real prior turn, since the opening
    # (no-prior-turn) path now uses a different instruction (see the
    # opening-turn tests below).
    fake = FakeLLM()
    monkeypatch.setattr(debater, "llm_client", fake)
    speak = debater.make_speaker("You are a debator.", "context text")
    state = DebateState(parties=["Bull", "Bear"])
    state.turns.append(DebateTurn(speaker="Bear", content="bear's opening argument", round=1))
    await speak(state, "Bull")
    prompt = fake.calls[0]["messages"][1]["content"]
    assert "acknowledge it explicitly rather than ignoring it" in prompt


# ---------------- opening-turn instruction (2026-09-01, TradingAgents v0.4.0
# #1176 comparison) ----------------
#
# The first speaker in round 1 of either debate this factory serves (Bull in
# bull/bear, Aggressive in risk — debate_parallel_first_round=False by
# default) used to receive the rebuttal instruction with nothing to rebut,
# and would invent an opposing argument to respond to. These pin the fix:
# the opening turn drops every rebuttal-specific phrase, forbids inventing
# an opponent, still requires steelmanning from the reports, and every OTHER
# turn (any turn with a real prior turn to read) is byte-identical to before.


async def test_opening_turn_does_not_ask_for_rebuttal(monkeypatch):
    fake = FakeLLM()
    monkeypatch.setattr(debater, "llm_client", fake)
    speak = debater.make_speaker("You are a debator.", "context text")
    await speak(DebateState(parties=["Bull", "Bear"]), "Bull")
    prompt = fake.calls[0]["messages"][1]["content"]
    assert "rebutting the specific points made above" not in prompt
    assert "engage directly with what was just said" not in prompt
    assert "points made against you" not in prompt


async def test_opening_turn_forbids_inventing_an_opponent(monkeypatch):
    fake = FakeLLM()
    monkeypatch.setattr(debater, "llm_client", fake)
    speak = debater.make_speaker("You are a debator.", "context text")
    await speak(DebateState(parties=["Aggressive", "Conservative", "Neutral"]), "Aggressive")
    prompt = fake.calls[0]["messages"][1]["content"]
    assert "Do not imagine what another debator might argue" in prompt
    assert "nothing above to rebut" in prompt


async def test_opening_turn_still_requires_steelmanning_from_the_reports(monkeypatch):
    fake = FakeLLM()
    monkeypatch.setattr(debater, "llm_client", fake)
    speak = debater.make_speaker("You are a debator.", "context text")
    await speak(DebateState(parties=["Bull", "Bear"]), "Bull")
    prompt = fake.calls[0]["messages"][1]["content"]
    assert "acknowledge the single strongest point against your own position found in the reports themselves" in prompt


async def test_rebuttal_instruction_is_byte_identical_when_others_exist(monkeypatch):
    fake = FakeLLM()
    monkeypatch.setattr(debater, "llm_client", fake)
    speak = debater.make_speaker("You are a debator.", "context text")
    state = DebateState(parties=["Bull", "Bear"])
    state.turns.append(DebateTurn(speaker="Bear", content="bear's opening argument", round=1))
    await speak(state, "Bull")
    prompt = fake.calls[0]["messages"][1]["content"]
    assert debater._REBUTTAL_INSTRUCTION in prompt
    assert debater._OPENING_INSTRUCTION not in prompt


async def test_second_and_third_speaker_in_round_one_still_get_the_rebuttal_prompt(monkeypatch):
    fake = FakeLLM()
    monkeypatch.setattr(debater, "llm_client", fake)
    speak = debater.make_speaker("You are a debator.", "context text")
    state = DebateState(parties=["Aggressive", "Conservative", "Neutral"])
    state.turns.append(DebateTurn(speaker="Aggressive", content="aggressive's opening argument", round=1))
    await speak(state, "Conservative")
    prompt = fake.calls[0]["messages"][1]["content"]
    assert debater._REBUTTAL_INSTRUCTION in prompt


async def test_round_two_speaker_sees_its_own_round_one_turn(monkeypatch):
    """The 2026-09-15 audit's headline defect: `latest_except` hides a
    speaker's own turns, so the risk-debate Neutral was told to keep or
    reverse "your earlier verdict" while unable to see it."""
    fake = FakeLLM()
    monkeypatch.setattr(debater, "llm_client", fake)
    speak = debater.make_speaker("You are a debator.", "context text")
    state = DebateState(parties=["Aggressive", "Conservative", "Neutral"])
    state.turns.append(DebateTurn(speaker="Neutral", content="neutral r1 verdict: Hold", round=1))
    state.turns.append(DebateTurn(speaker="Aggressive", content="aggressive r2", round=2))
    state.turns.append(DebateTurn(speaker="Conservative", content="conservative r2", round=2))
    state.rounds_completed = 1
    await speak(state, "Neutral")
    prompt = fake.calls[0]["messages"][1]["content"]
    assert debater._OWN_TURN_HEADER in prompt
    assert "neutral r1 verdict: Hold" in prompt
    # Own turn precedes the others', so the reversal test reads as before/after.
    assert prompt.index(debater._OWN_TURN_HEADER) < prompt.index("Other debators' latest arguments:")


async def test_own_turn_block_absent_in_round_one(monkeypatch):
    fake = FakeLLM()
    monkeypatch.setattr(debater, "llm_client", fake)
    speak = debater.make_speaker("You are a debator.", "context text")
    state = DebateState(parties=["Aggressive", "Conservative", "Neutral"])
    state.turns.append(DebateTurn(speaker="Aggressive", content="aggressive's opening argument", round=1))
    await speak(state, "Conservative")
    prompt = fake.calls[0]["messages"][1]["content"]
    assert debater._OWN_TURN_HEADER not in prompt


async def test_own_turn_kill_switch_restores_pre_fix_prompt(monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "debate_own_previous_turn", False)
    fake = FakeLLM()
    monkeypatch.setattr(debater, "llm_client", fake)
    speak = debater.make_speaker("You are a debator.", "context text")
    state = DebateState(parties=["Bull", "Bear"])
    state.turns.append(DebateTurn(speaker="Bull", content="bull r1", round=1))
    state.turns.append(DebateTurn(speaker="Bear", content="bear r1", round=1))
    state.rounds_completed = 1
    await speak(state, "Bull")
    prompt = fake.calls[0]["messages"][1]["content"]
    assert debater._OWN_TURN_HEADER not in prompt
    assert "bull r1" not in prompt
    assert "[Bear]: bear r1" in prompt


async def test_kill_switch_restores_pre_fix_opening_prompt(monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "debate_opening_turn_prompt", False)
    fake = FakeLLM()
    monkeypatch.setattr(debater, "llm_client", fake)
    speak = debater.make_speaker("You are a debator.", "context text")
    await speak(DebateState(parties=["Bull", "Bear"]), "Bull")
    prompt = fake.calls[0]["messages"][1]["content"]
    assert debater._REBUTTAL_INSTRUCTION in prompt
    assert debater._OPENING_INSTRUCTION not in prompt


async def test_bull_researcher_build_includes_reports_and_stance(monkeypatch):
    fake = FakeLLM("bull case")
    monkeypatch.setattr(debater, "llm_client", fake)
    speak = bull_researcher.build({"Market": "uptrend"})
    result = await speak(DebateState(parties=["Bull", "Bear"]), "Bull")
    assert result == "bull case"
    system_message = fake.calls[0]["messages"][0]["content"]
    assert bull_researcher.SYSTEM_PROMPT in system_message
    assert "uptrend" in fake.calls[0]["messages"][1]["content"]


async def test_bear_researcher_build_includes_reports_and_stance(monkeypatch):
    fake = FakeLLM("bear case")
    monkeypatch.setattr(debater, "llm_client", fake)
    speak = bear_researcher.build({"Market": "downtrend"})
    result = await speak(DebateState(parties=["Bull", "Bear"]), "Bear")
    assert result == "bear case"
    assert bear_researcher.SYSTEM_PROMPT in fake.calls[0]["messages"][0]["content"]


def test_bull_researcher_system_prompt_requires_steelmanning_and_multiple_reports():
    assert "triangulating multiple independent signals" in bull_researcher.SYSTEM_PROMPT
    assert "acknowledge the single strongest point against your own position" in bull_researcher.SYSTEM_PROMPT


def test_bear_researcher_system_prompt_requires_steelmanning_and_multiple_reports():
    assert "triangulating multiple independent signals" in bear_researcher.SYSTEM_PROMPT
    assert "acknowledge the single strongest point against your own position" in bear_researcher.SYSTEM_PROMPT


async def test_make_speaker_appends_injection_warning_to_system_prompt(monkeypatch):
    fake = FakeLLM()
    monkeypatch.setattr(debater, "llm_client", fake)
    speak = debater.make_speaker("You are a debator.", "context text")
    await speak(DebateState(parties=["Bull", "Bear"]), "Bull")
    system_message = fake.calls[0]["messages"][0]["content"]
    assert "You are a debator." in system_message
    assert "INJECTION_SUSPECTED" in system_message


async def test_make_speaker_logs_when_response_flags_injection(monkeypatch, caplog):
    fake = FakeLLM("INJECTION_SUSPECTED: looked like an embedded command\nActual argument here.")
    monkeypatch.setattr(debater, "llm_client", fake)
    speak = debater.make_speaker("You are a debator.", "context text", label="bull_researcher")
    with caplog.at_level("WARNING"):
        await speak(DebateState(parties=["Bull", "Bear"]), "Bull")
    assert any("bull_researcher flagged possible prompt injection" in r.message for r in caplog.records)


async def test_risk_debators_build_all_returns_three_stances(monkeypatch):
    fake = FakeLLM("risk take")
    monkeypatch.setattr(debater, "llm_client", fake)
    proposal = TraderProposal(action=TraderAction.BUY, reasoning="strong case")
    speakers = risk_debators.build_all({"Market": "uptrend"}, proposal)
    assert set(speakers.keys()) == {"Aggressive", "Conservative", "Neutral"}
    result = await speakers["Aggressive"](DebateState(parties=list(speakers.keys())), "Aggressive")
    assert result == "risk take"


async def test_risk_debators_build_all_appends_extra_context(monkeypatch):
    fake = FakeLLM("risk take")
    monkeypatch.setattr(debater, "llm_client", fake)
    proposal = TraderProposal(action=TraderAction.HOLD, reasoning="wait and see")
    speakers = risk_debators.build_all({}, proposal, extra_context="Korean short-selling signal: high")
    await speakers["Neutral"](DebateState(parties=list(speakers.keys())), "Neutral")
    prompt = fake.calls[0]["messages"][1]["content"]
    assert "Korean short-selling signal: high" in prompt


async def test_risk_debators_build_all_injects_research_plan_note_before_trader_proposal(monkeypatch):
    """2026-09-16 fix (engine audit follow-up item 3): the risk debate
    previously saw only the Trader's proposal, whose 3-value `action` cannot
    distinguish the Research Manager's Underweight (a 25% trim) from Sell (a
    full exit) — measured live, that collapse propagates into the Portfolio
    Manager's own final rating (see pipeline._desk_consensus_note's matching
    comment). Feeding the debate the RM's own rendered plan directly closes
    that gap."""
    fake = FakeLLM("risk take")
    monkeypatch.setattr(debater, "llm_client", fake)
    proposal = TraderProposal(action=TraderAction.SELL, reasoning="exiting per plan")
    speakers = risk_debators.build_all(
        {}, proposal, research_plan_note="**Recommendation**: Underweight\n\n**Rationale**: trim, not exit",
    )
    await speakers["Neutral"](DebateState(parties=list(speakers.keys())), "Neutral")
    prompt = fake.calls[0]["messages"][1]["content"]
    assert "Research Manager's plan:" in prompt
    assert "**Recommendation**: Underweight" in prompt
    assert prompt.index("Research Manager's plan:") < prompt.index("Trader's proposal:")


async def test_risk_debators_build_all_omits_research_plan_note_when_not_given(monkeypatch):
    fake = FakeLLM("risk take")
    monkeypatch.setattr(debater, "llm_client", fake)
    proposal = TraderProposal(action=TraderAction.HOLD, reasoning="wait and see")
    speakers = risk_debators.build_all({}, proposal)
    await speakers["Neutral"](DebateState(parties=list(speakers.keys())), "Neutral")
    prompt = fake.calls[0]["messages"][1]["content"]
    assert "Research Manager's plan:" not in prompt


def test_risk_debators_neutral_stance_requires_judging_not_averaging():
    # 2026-08-12: Neutral used to be told to "propose a concrete middle
    # path" — a self-audit found this structurally converging on Hold (see
    # risk_debators.py's own module comment). Neutral now must judge which
    # side's evidence is stronger and adopt that side's level, only landing
    # in between for a genuine, justified tie.
    neutral_stance = risk_debators._STANCES["Neutral"]
    assert "JUDGE, not average" in neutral_stance
    assert "adopt THAT side's" in neutral_stance
    assert "Genuine ties are rare, not the norm" in neutral_stance


def test_risk_debators_aggressive_and_conservative_require_grounded_concrete_adjustments():
    # Symmetry check: Neutral was sharpened to require concrete, evidence-
    # grounded reasoning — Aggressive/Conservative must match that rigor
    # rather than staying generic personality descriptions.
    aggressive_stance = risk_debators._STANCES["Aggressive"]
    conservative_stance = risk_debators._STANCES["Conservative"]
    assert "Ground your conviction in specific evidence" in aggressive_stance
    assert "concrete position-size/stop-loss adjustment" in aggressive_stance
    assert "Ground your caution in specific risks" in conservative_stance
    assert "concrete position-size/stop-loss adjustment" in conservative_stance


# ---------------- rating-scale anchoring (2026-08-05) ----------------
#
# A live-decision audit found Neutral's synthesis landing on a freeform
# compromise ("reduce by 50%", "partial profit-taking") with no exact match
# among the Portfolio Manager's five fixed rating levels — the PM forced the
# conclusion into a rating anyway, but its Executive Summary kept quoting the
# debate's own freeform language, describing an action the system never
# executed. All three stances now anchor to the same five levels the PM must
# choose from, so whatever conclusion Neutral reaches is already executable.


def test_all_three_stances_state_the_five_fixed_rating_levels():
    for stance_name in ("Aggressive", "Conservative", "Neutral"):
        prompt = risk_debators._STANCES[stance_name]
        assert "Sell (0%" in prompt
        assert "Underweight (25%" in prompt
        assert "Hold (no change" in prompt
        assert "Overweight (75%" in prompt
        assert "Buy" in prompt and "100%" in prompt


def test_all_three_stances_reject_freeform_percentage_framing():
    for stance_name in ("Aggressive", "Conservative", "Neutral"):
        prompt = risk_debators._STANCES[stance_name]
        assert "not a custom percentage" in prompt
        assert "partial reduce/add" in prompt


def test_neutral_stance_requires_naming_one_of_the_five_levels_explicitly():
    neutral_stance = risk_debators._STANCES["Neutral"]
    assert "authoritative synthesis" in neutral_stance
    assert "state your conclusion as exactly one of the five levels above" in neutral_stance


async def test_risk_debators_prompt_includes_rating_scale_note(monkeypatch):
    fake = FakeLLM("risk take")
    monkeypatch.setattr(debater, "llm_client", fake)
    proposal = TraderProposal(action=TraderAction.BUY, reasoning="strong case")
    speakers = risk_debators.build_all({"Market": "uptrend"}, proposal)
    await speakers["Neutral"](DebateState(parties=list(speakers.keys())), "Neutral")
    system_message = fake.calls[0]["messages"][0]["content"]
    assert "not continuous" in system_message
    assert "five fixed levels" in system_message


# ---------------- flat-ticker rating menu (2026-09-01, id=1156/1157/1158) ----------------


def test_flat_stances_offer_only_buy_and_hold():
    for name, prompt in risk_debators._stances(held=False).items():
        assert "can only act on TWO levels here: Buy" in prompt
        assert "Overweight/Underweight/Sell, since there is no existing position" in prompt
        assert "Hold means stay out, Sell does not exist here" in prompt


def test_held_stances_are_byte_identical_to_the_module_constant():
    assert risk_debators._stances(held=True) == risk_debators._STANCES


def test_flat_clause_is_shared_verbatim_across_the_three_sibling_nodes():
    import app.agents.portfolio_manager as pm
    import app.agents.research_manager as rm

    clause = "Overweight/Underweight/Sell, since there is no existing "
    assert clause in risk_debators._FLAT_RATING_SCALE_NOTE
    assert clause in rm._FLAT_SUFFIX
    assert clause in pm._FLAT_SUFFIX


def test_flat_conservative_disclaims_smaller_sizing():
    assert "'smaller sizing' is not available to you as a lever" in risk_debators._stances(held=False)["Conservative"]


async def test_build_all_flat_reaches_the_model(monkeypatch):
    fake = FakeLLM("risk take")
    monkeypatch.setattr(debater, "llm_client", fake)
    proposal = TraderProposal(action=TraderAction.BUY, reasoning="strong case")
    speakers = risk_debators.build_all({"Market": "uptrend"}, proposal, held=False)
    await speakers["Aggressive"](DebateState(parties=list(speakers.keys())), "Aggressive")
    assert "can only act on TWO levels here" in fake.calls[0]["messages"][0]["content"]


# ---------------- sizing-mechanics preamble (2026-09-02, id=1170, 498400.KS) ----------------
#
# The risk debate previously argued sizing blind to the real cap: Conservative
# asserted "we cannot reduce position size (it's all-or-nothing)" and Neutral's
# deciding turn adopted it as verified fact. build_all's new sizing_context
# kwarg threads the pipeline's pre-computed risk_cap_context/volatility_context
# into the shared context so the debate argues against the real constraint.


async def test_build_all_sizing_context_reaches_every_stance(monkeypatch):
    fake = FakeLLM("risk take")
    monkeypatch.setattr(debater, "llm_client", fake)
    proposal = TraderProposal(action=TraderAction.BUY, reasoning="strong case")
    speakers = risk_debators.build_all(
        {"Market": "uptrend"}, proposal,
        sizing_context="RISK CAP: ~32 new shares at a 3% stop, ~9 at a 10% stop.",
    )
    for name in ("Aggressive", "Conservative", "Neutral"):
        await speakers[name](DebateState(parties=list(speakers.keys())), name)
    for call in fake.calls:
        prompt = call["messages"][1]["content"]
        assert "RISK CAP: ~32 new shares at a 3% stop, ~9 at a 10% stop." in prompt
        assert "How the code will actually size this trade." in prompt


async def test_build_all_without_sizing_context_is_byte_identical_to_before(monkeypatch):
    # Backward compatibility: any caller (older test/backtest harness) that
    # doesn't pass sizing_context must get the exact same prompt as before
    # this change — sizing_context defaults to "" and is a no-op when empty.
    fake = FakeLLM("risk take")
    monkeypatch.setattr(debater, "llm_client", fake)
    proposal = TraderProposal(action=TraderAction.BUY, reasoning="strong case")

    speakers_without_kwarg = risk_debators.build_all({"Market": "uptrend"}, proposal)
    await speakers_without_kwarg["Neutral"](DebateState(parties=["Aggressive", "Conservative", "Neutral"]), "Neutral")
    prompt_without_kwarg = fake.calls[-1]["messages"][1]["content"]

    speakers_with_empty = risk_debators.build_all({"Market": "uptrend"}, proposal, sizing_context="")
    await speakers_with_empty["Neutral"](DebateState(parties=["Aggressive", "Conservative", "Neutral"]), "Neutral")
    prompt_with_empty = fake.calls[-1]["messages"][1]["content"]

    assert prompt_without_kwarg == prompt_with_empty
    assert "How the code will actually size this trade." not in prompt_without_kwarg


def test_sizing_mechanics_preamble_does_not_favor_a_direction():
    # A cheap keyword guard, not a semantic proof: the preamble must not read
    # as an instruction to buy or to avoid buying.
    preamble = risk_debators._sizing_mechanics_preamble(held=True)
    lowered = preamble.lower()
    for banned in ("you should buy", "you should avoid", "you should sell", "you should hold"):
        assert banned not in lowered
    assert "not an argument for any rating" in preamble


def test_sizing_mechanics_preamble_flat_vs_held():
    held_preamble = risk_debators._sizing_mechanics_preamble(held=True)
    flat_preamble = risk_debators._sizing_mechanics_preamble(held=False)
    assert "RATING menu is two-valued" not in held_preamble
    assert "RATING menu is two-valued" in flat_preamble
    # The flat note is appended, not substituted — the shared body still appears.
    assert "How the code will actually size this trade." in flat_preamble


def test_sizing_mechanics_preamble_has_no_injection_banner_markers():
    # Cheap regression guard: this session's audit found 6 false-positive
    # injection flags, all originating from the pipeline's own imperative-
    # shaped ALL-CAPS correction banners embedded in report bodies. The new
    # preamble is deliberately worded to avoid that shape.
    for preamble in (
        risk_debators._sizing_mechanics_preamble(held=True),
        risk_debators._sizing_mechanics_preamble(held=False),
    ):
        for marker in ("WARNING", "IMPORTANT", "CRITICAL", "ATTENTION"):
            assert marker not in preamble


# ---------------- consensus-override / execution-mechanics guardrails (2026-09-04) ----------------


def test_neutral_stance_rejects_execution_mechanics_as_evidence():
    for held in (True, False):
        neutral = risk_debators._stances(held=held)["Neutral"]
        assert "Execution mechanics are not directional evidence" in neutral
        assert "only by naming the specific new fact" in neutral


def test_flat_neutral_stance_still_has_two_level_menu():
    neutral = risk_debators._stances(held=False)["Neutral"]
    assert "Buy or Hold" in neutral


def test_sizing_mechanics_preamble_has_scope_paragraph_and_keeps_194_anchor():
    preamble = risk_debators._SIZING_MECHANICS_PREAMBLE
    assert "Scope of this block" in preamble
    assert "all-or-nothing" in preamble


def _debate_with(turns):
    ds = DebateState(parties=["Aggressive", "Conservative", "Neutral"])
    for i, (speaker, content) in enumerate(turns):
        ds.turns.append(DebateTurn(speaker=speaker, content=content, round=1 + i // 3))
    return ds


def test_check_illegal_ratings_empty_for_held_ticker():
    ds = _debate_with([("Neutral", "CONCLUSION: Underweight (25%)")])
    assert risk_debators.check_illegal_ratings(ds, held=True) == []


def test_check_illegal_ratings_flags_a_flat_declaration():
    ds = _debate_with([("Neutral", "My conclusion: Underweight (25%) given the risk.")])
    found = risk_debators.check_illegal_ratings(ds, held=False)
    assert len(found) == 1 and "Neutral" in found[0]


def test_check_illegal_ratings_id1158_shape():
    ds = _debate_with([
        ("Aggressive", "Final Decision: Sell (0%)"),
        ("Conservative", "CONCLUSION: Sell (0%)"),
        ("Neutral", "CONCLUSION: Underweight (25%)"),
    ])
    assert len(risk_debators.check_illegal_ratings(ds, held=False)) == 3


def test_check_illegal_ratings_no_false_positive_on_a_disclaimer():
    ds = _debate_with([("Conservative", "Underweight (25%): Nonsensical for a non-held position.")])
    assert risk_debators.check_illegal_ratings(ds, held=False) == []


def test_check_illegal_ratings_clean_flat_debate():
    ds = _debate_with([("Neutral", "CONCLUSION: Buy (100%) with a stop at 128,000.")])
    assert risk_debators.check_illegal_ratings(ds, held=False) == []


def _debate_with_rounds(turns):
    """Like `_debate_with` but the caller controls `round` explicitly — needed
    for check_neutral_reversal tests, which compare round 1 against the final
    round rather than treating every 3 turns as one round."""
    ds = DebateState(parties=["Aggressive", "Conservative", "Neutral"])
    for speaker, content, round_ in turns:
        ds.turns.append(DebateTurn(speaker=speaker, content=content, round=round_))
    ds.rounds_completed = max(r for _, _, r in turns)
    return ds


def test_neutral_reversal_records_uncited_flip():
    ds = _debate_with_rounds([
        ("Neutral", "Weighing both sides. **Final Decision: Hold**", 1),
        (
            "Neutral",
            "On reflection, the risk cap allows for a meaningful position size.\n\n"
            "**Final Decision:**\n**Buy**\n**Stop-Loss:** 1,950 KRW",
            2,
        ),
    ])
    record = risk_debators.check_neutral_reversal(ds)
    assert record["r1"] == "Hold"
    assert record["final"] == "Buy"
    assert record["reversed"] is True
    assert record["new_fact_cited"] is False
    assert "risk cap" in record["mechanics_terms"]
    assert record["final_round"] == 2


def test_neutral_reversal_parses_the_id1158_level_label():
    # 2026-09-09: id=1158's r2 (the founding incident) wrote
    # "**Final Decision:**\n\n**Level: Buy (100%)**" — "LEVEL" was missing
    # from _VERDICT_LABEL so this reversal went unrecorded.
    ds = _debate_with_rounds([
        ("Neutral", "The Conservative case is better supported.\n**Final Decision: Sell (0%)**", 1),
        ("Neutral", "The Aggressive case now wins.\n\n**Final Decision:**\n\n**Level: Buy (100%)**", 2),
    ])
    record = risk_debators.check_neutral_reversal(ds)
    assert record["r1"] == "Sell"
    assert record["final"] == "Buy"
    assert record["reversed"] is True


def test_neutral_reversal_empty_when_verdict_unparsed():
    ds = _debate_with_rounds([
        ("Neutral", "Both sides make reasonable points and the evidence is mixed.", 1),
        ("Neutral", "**Final Decision: Hold**", 2),
    ])
    record = risk_debators.check_neutral_reversal(ds)
    assert record["r1"] is None
    assert record["reversed"] is False


def test_neutral_reversal_stable_verdict_not_flagged():
    ds = _debate_with_rounds([
        ("Neutral", "**Final Decision: Hold**", 1),
        (
            "Neutral",
            "Sticking with my earlier call given the allocation and risk profile. "
            "**Final Decision: Hold**",
            2,
        ),
    ])
    record = risk_debators.check_neutral_reversal(ds)
    assert record["reversed"] is False
    assert "allocation" in record["mechanics_terms"]


def test_neutral_reversal_last_declaration_wins():
    ds = _debate_with_rounds([
        ("Neutral", "**Final Decision: Hold**", 1),
        (
            "Neutral",
            "Note that the Aggressive case argues Buy (100%) on momentum grounds, "
            "but I disagree. **Final Decision: Hold**",
            2,
        ),
    ])
    record = risk_debators.check_neutral_reversal(ds)
    assert record["final"] == "Hold"


def test_neutral_reversal_new_fact_citation_recorded():
    ds = _debate_with_rounds([
        ("Neutral", "**Final Decision: Sell**", 1),
        (
            "Neutral",
            "There is a new fact introduced in round 2: the disclosed ₩31,500 swing low. "
            "**Final Decision: Overweight**",
            2,
        ),
    ])
    record = risk_debators.check_neutral_reversal(ds)
    assert record["reversed"] is True
    assert record["new_fact_cited"] is True


def test_neutral_reversal_later_turn_argument_cited_when_debater_named():
    # 2026-09-14: the clause's OTHER permitted reversal route — naming a
    # later-turn argument, not just a new fact. id=1282-style phrasing.
    ds = _debate_with_rounds([
        ("Neutral", "**Final Decision: Hold**", 1),
        (
            "Neutral",
            "As the Conservative debater correctly calculated, a stop at "
            "₩27,800 is virtually guaranteed to be triggered. "
            "**Final Decision: Sell**",
            2,
        ),
    ])
    record = risk_debators.check_neutral_reversal(ds)
    assert record["reversed"] is True
    assert record["new_fact_cited"] is False
    assert record["later_turn_argument_cited"] is True
    assert "Conservative" in record["later_turn_argument_span"]


def test_neutral_reversal_later_turn_argument_not_cited_without_named_debater():
    ds = _debate_with_rounds([
        ("Neutral", "**Final Decision: Hold**", 1),
        (
            "Neutral",
            "On further reflection this position deserves more weight. "
            "**Final Decision: Buy**",
            2,
        ),
    ])
    record = risk_debators.check_neutral_reversal(ds)
    assert record["later_turn_argument_cited"] is False
    assert record["later_turn_argument_span"] == ""


def test_neutral_reversal_basis_new_fact():
    # 2026-09-17: reversal_basis derives from the two booleans, fact takes
    # precedence over argument when both would be true.
    ds = _debate_with_rounds([
        ("Neutral", "**Final Decision: Sell**", 1),
        (
            "Neutral",
            "There is a new fact introduced in round 2: the disclosed ₩31,500 swing low. "
            "**Final Decision: Overweight**",
            2,
        ),
    ])
    record = risk_debators.check_neutral_reversal(ds)
    assert record["reversal_basis"] == "new_fact"


def test_neutral_reversal_basis_later_turn_argument_logs_info_not_warning(caplog):
    # id=1539-shaped case: no new fact, but a later-turn argument was named.
    # This is a PERMITTED route under f79213e's clause and must not log at
    # WARNING (that was the bug: id=1539 was misread as a compliance failure
    # from new_fact_cited alone).
    ds = _debate_with_rounds([
        ("Neutral", "**Final Decision: Hold**", 1),
        (
            "Neutral",
            "As the Conservative debater correctly calculated, a stop at "
            "₩27,800 is virtually guaranteed to be triggered. "
            "**Final Decision: Sell**",
            2,
        ),
    ])
    with caplog.at_level("INFO", logger=risk_debators.__name__):
        record = risk_debators.check_neutral_reversal(ds)
    assert record["reversal_basis"] == "later_turn_argument"
    assert not any(r.levelname == "WARNING" for r in caplog.records)
    info_records = [r for r in caplog.records if r.levelname == "INFO"]
    assert any("Neutral reversed" in r.message for r in info_records)


def test_neutral_reversal_basis_neither_logs_warning(caplog):
    ds = _debate_with_rounds([
        ("Neutral", "**Final Decision: Hold**", 1),
        (
            "Neutral",
            "On reflection, the risk cap allows for a meaningful position size.\n\n"
            "**Final Decision:**\n**Buy**\n**Stop-Loss:** 1,950 KRW",
            2,
        ),
    ])
    with caplog.at_level("INFO", logger=risk_debators.__name__):
        record = risk_debators.check_neutral_reversal(ds)
    assert record["reversal_basis"] == "neither"
    warning_records = [r for r in caplog.records if r.levelname == "WARNING"]
    assert len(warning_records) == 1
    assert "Neutral reversed" in warning_records[0].message


def test_neutral_reversal_basis_empty_when_not_reversed():
    ds = _debate_with_rounds([
        ("Neutral", "**Final Decision: Hold**", 1),
        (
            "Neutral",
            "Sticking with my earlier call given the allocation and risk profile. "
            "**Final Decision: Hold**",
            2,
        ),
    ])
    record = risk_debators.check_neutral_reversal(ds)
    assert record["reversed"] is False
    assert record["reversal_basis"] == ""


def test_neutral_reversal_empty_without_neutral_turns():
    ds = _debate_with_rounds([
        ("Aggressive", "**Final Decision: Buy**", 1),
        ("Conservative", "**Final Decision: Hold**", 1),
    ])
    assert risk_debators.check_neutral_reversal(ds) == {}


def test_neutral_reversal_matches_position_sizing_morphological_variant():
    # id=1281 (live, 2026-09-14): "a false sense of security provided by small
    # position sizing" was missed because the old tuple only had "position
    # size", which is not a substring of "position sizing".
    ds = _debate_with_rounds([
        ("Neutral", "**Final Decision: Hold**", 1),
        (
            "Neutral",
            "There is a false sense of security provided by small position sizing. "
            "**Final Decision: Buy**",
            2,
        ),
    ])
    record = risk_debators.check_neutral_reversal(ds)
    assert "position sizing" in record["mechanics_terms"]


def test_neutral_reversal_negligible_position_matches_but_bare_volume_does_not():
    # id=1268 (live, 2026-09-14): bare "negligible" false-positived on
    # "this volume is negligible against the backdrop of ... intraday range",
    # which is about trading volume, not position sizing. "negligible
    # position" keeps the genuine sizing hits (id=1249/1255/1267) without it.
    ds = _debate_with_rounds([
        ("Neutral", "**Final Decision: Hold**", 1),
        (
            "Neutral",
            "This volume is negligible against the backdrop of a stock trading "
            "in the middle of its intraday range. **Final Decision: Buy**",
            2,
        ),
    ])
    record = risk_debators.check_neutral_reversal(ds)
    assert "negligible position" not in record["mechanics_terms"]

    ds2 = _debate_with_rounds([
        ("Neutral", "**Final Decision: Hold**", 1),
        (
            "Neutral",
            "This is a negligible position size relative to the portfolio. "
            "**Final Decision: Buy**",
            2,
        ),
    ])
    record2 = risk_debators.check_neutral_reversal(ds2)
    assert "negligible position" in record2["mechanics_terms"]


def test_neutral_reversal_compliant_execution_mechanics_quote_does_not_match():
    # id=1251 (live, 2026-09-14): "execution mechanics" fires hardest when a
    # COMPLIANT debater quotes the clause back before rejecting it, not when
    # it is misused as the debater's own argument. Dropped from the tuple
    # entirely, so no term should be recorded here.
    ds = _debate_with_rounds([
        ("Neutral", "**Final Decision: Hold**", 1),
        (
            "Neutral",
            "As instructed, I must separate execution mechanics from "
            "directional evidence. **Final Decision: Hold**",
            2,
        ),
    ])
    record = risk_debators.check_neutral_reversal(ds)
    assert record["mechanics_terms"] == []


def test_neutral_reversal_coverage_keys_present_for_short_turn():
    content = "**Final Decision: Hold**"
    ds = _debate_with_rounds([
        ("Neutral", content, 1),
        ("Neutral", content, 2),
    ])
    record = risk_debators.check_neutral_reversal(ds)
    assert record["final_turn_chars"] == len(content)
    # Shorter than the 1200-char tail window: scanned == full length.
    assert record["mechanics_scanned_chars"] == len(content)


def test_neutral_reversal_coverage_keys_reflect_truncation_for_long_turn():
    long_content = ("Filler prose. " * 200) + "**Final Decision: Hold**"
    assert len(long_content) > risk_debators._MECHANICS_TAIL_CHARS
    ds = _debate_with_rounds([
        ("Neutral", "**Final Decision: Hold**", 1),
        ("Neutral", long_content, 2),
    ])
    record = risk_debators.check_neutral_reversal(ds)
    assert record["final_turn_chars"] == len(long_content)
    assert record["mechanics_scanned_chars"] == risk_debators._MECHANICS_TAIL_CHARS


# ---------------- round-1 debate-turn length budget (2026-09-07, AI-engine
# step-0 latency audit) ----------------
#
# Caps completion tokens ONLY for round 1 (the opening/pre-rebuttal
# argument), via a `max_tokens` override threaded through to
# llm_client.complete() — the FINAL round (whichever a risk-debate Neutral
# turn is promoted to "FINAL SYNTHESIS" from, portfolio_manager.
# _render_risk_debate) always keeps the global settings.llm_max_output_tokens
# ceiling, untouched, so #205's gate calibration and check_neutral_reversal's
# FINAL-round parse see exactly the same completions as before this change.


def test_current_round_before_any_turn_is_round_one():
    state = DebateState(parties=["Bull", "Bear"])
    assert debater._current_round(state) == 1


def test_current_round_after_round_one_completes_is_round_two():
    # Mirrors run_debate's own sequencing: rounds_completed is set to N only
    # AFTER round N's turns are all appended — see debater._current_round's
    # own docstring, which cites this exact invariant.
    state = DebateState(parties=["Bull", "Bear"], rounds_completed=1)
    assert debater._current_round(state) == 2


async def test_speak_round_one_uses_the_round1_budget(monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "debate_round1_max_output_tokens", 1100)
    fake = FakeLLM()
    monkeypatch.setattr(debater, "llm_client", fake)
    speak = debater.make_speaker("You are a debator.", "context text")
    state = DebateState(parties=["Bull", "Bear"])  # rounds_completed=0 -> round 1
    await speak(state, "Bull")
    assert fake.calls[0]["max_tokens"] == 1100


async def test_speak_round_two_uses_no_override(monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "debate_round1_max_output_tokens", 1100)
    fake = FakeLLM()
    monkeypatch.setattr(debater, "llm_client", fake)
    speak = debater.make_speaker("You are a debator.", "context text")
    state = DebateState(parties=["Bull", "Bear"], rounds_completed=1)  # round 2
    state.turns.append(DebateTurn(speaker="Bear", content="bear's round-1 argument", round=1))
    await speak(state, "Bull")
    assert fake.calls[0]["max_tokens"] is None


async def test_speak_kill_switch_disables_the_round1_budget(monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "debate_round1_max_output_tokens", 0)
    fake = FakeLLM()
    monkeypatch.setattr(debater, "llm_client", fake)
    speak = debater.make_speaker("You are a debator.", "context text")
    state = DebateState(parties=["Bull", "Bear"])  # round 1
    await speak(state, "Bull")
    assert fake.calls[0]["max_tokens"] is None


async def test_risk_debators_round_one_turns_all_get_the_budget(monkeypatch):
    # Pins design decision #2 (both debates share the budget) for the
    # risk-debate side specifically — all three stances, round 1.
    from app.config import settings

    monkeypatch.setattr(settings, "debate_round1_max_output_tokens", 1100)
    fake = FakeLLM("risk take")
    monkeypatch.setattr(debater, "llm_client", fake)
    proposal = TraderProposal(action=TraderAction.BUY, reasoning="strong case")
    speakers = risk_debators.build_all({"Market": "uptrend"}, proposal)
    state = DebateState(parties=list(speakers.keys()))
    for name in ("Aggressive", "Conservative", "Neutral"):
        await speakers[name](state, name)
    assert all(call["max_tokens"] == 1100 for call in fake.calls)
