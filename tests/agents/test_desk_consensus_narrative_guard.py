"""app.agents.desk_consensus_narrative_guard — no network, no LLM (A-stage).

Real incident this guards against (2026-09-09, id=1352, 028050.KS/삼성E&A):
`_desk_consensus_note` told the PM, verbatim, that the Research Manager
concluded Underweight and the Trader proposed Sell — and the PM's
executive_summary wrote "The desk unanimously recommends HOLD, and I
concur." This module is a detect-only recorder, mirroring
app.agents.halt_status_guard — see its own module docstring for the full
scope and reasoning.
"""

from __future__ import annotations

from app.agents.desk_consensus_narrative_guard import scan_desk_narrative


def test_flags_id1352_false_unanimity_claim():
    text = (
        "The desk unanimously recommends HOLD, and I concur. While the corrected "
        "fundamentals (operating profit +36.4% YoY) and the clean breakout above "
        "the 50,000 KRW psychological barrier support a constructive stance..."
    )
    findings = scan_desk_narrative(text, rm="Underweight", trader="Sell", pm_rating="Hold", agent_name="portfolio_manager")
    assert len(findings) == 1
    assert "RM=Underweight" in findings[0] and "Trader=Sell" in findings[0] and "PM=Hold" in findings[0]


def test_does_not_flag_genuine_unanimity():
    text = "The desk unanimously recommends HOLD, and I concur with that assessment."
    assert scan_desk_narrative(text, rm="Hold", trader="Hold", pm_rating="Hold", agent_name="portfolio_manager") == []


def test_does_not_flag_negated_claim():
    text = "The desk is not unanimous here — the Research Manager and the Trader diverge, and I side with the Trader."
    assert scan_desk_narrative(text, rm="Underweight", trader="Sell", pm_rating="Hold", agent_name="portfolio_manager") == []


def test_does_not_flag_bare_consensus_block_reference():
    # "consensus" alone describes the injected block's own heading
    # ("DESK CONSENSUS BEFORE THE RISK DEBATE"), not an agreement claim.
    text = "Per the desk consensus block above, the Research Manager concluded Underweight."
    assert scan_desk_narrative(text, rm="Underweight", trader="Sell", pm_rating="Hold", agent_name="portfolio_manager") == []


def test_flags_agree_direction_mismatch_even_without_named_level():
    text = "All three desk members agree on the path forward here."
    findings = scan_desk_narrative(text, rm="Buy", trader="Hold", pm_rating="Hold", agent_name="portfolio_manager")
    assert len(findings) == 1


def test_no_scan_when_research_plan_missing():
    text = "The desk unanimously recommends HOLD."
    assert scan_desk_narrative(text, rm="", trader="Sell", pm_rating="Hold", agent_name="portfolio_manager") == []


def test_no_scan_when_trader_proposal_missing():
    text = "The desk unanimously recommends HOLD."
    assert scan_desk_narrative(text, rm="Underweight", trader="", pm_rating="Hold", agent_name="portfolio_manager") == []


def test_dedups_repeated_span():
    span = "The desk unanimously recommends HOLD, and I concur"
    text = f"{span}. {span}."
    findings = scan_desk_narrative(text, rm="Underweight", trader="Sell", pm_rating="Hold", agent_name="portfolio_manager")
    assert len(findings) == 1


def test_does_not_flag_id1407_rating_token_in_next_sentence():
    # 2026-09-14, decision id=1407: RM=Hold/Trader=Hold/PM=Hold, so the
    # unanimity claim is TRUE — but the flat 90-char lookahead reached past
    # the period into "examined a Buy case" and read "Buy" as the claimed
    # level. The lookahead is now bounded by the end of the sentence.
    text = (
        "The Research Manager and Trader both proposed Hold, matching my decision. "
        "The desk is unanimously aligned. The risk debate thoroughly examined a Buy "
        "case centered on a technical washout, but the Final Synthesis correctly "
        "dismantled it by highlighting that micro-cap illiquidity and confirmed "
        "foreign distribution destroy the theoretical stop-loss mechanics."
    )
    assert scan_desk_narrative(text, rm="Hold", trader="Hold", pm_rating="Hold", agent_name="portfolio_manager") == []


def test_flags_wrong_level_named_in_same_sentence():
    text = "The desk unanimously recommends Buy."
    findings = scan_desk_narrative(text, rm="Hold", trader="Hold", pm_rating="Hold", agent_name="portfolio_manager")
    assert len(findings) == 1
    assert "(Buy)" in findings[0]
