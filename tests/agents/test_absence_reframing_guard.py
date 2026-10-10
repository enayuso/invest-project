"""app.agents.absence_reframing_guard — no network, no LLM (A-stage).

Real incidents this guards against (2026-09-09 domestic AI-decision audit,
ids 1349/1350/1351/1352/1358/1359): missing data (insider activity, investor
flow, analyst coverage) reframed as a specific positive/reassuring signal.
This module is a detect-only recorder, mirroring
app.agents.halt_status_guard.scan_halt_claims — see its own module docstring
for the full scope and the same-sample contrast (correct) usages that must
never fire.
"""

from __future__ import annotations

from app.agents.absence_reframing_guard import scan_absence_reframing


def test_flags_insider_silence_as_confidence_id_1349():
    text = (
        "the absence of insider selling suggests management does not believe the current price "
        "represents fair value, reducing the likelihood of a fraudulent pump-and-dump exit."
    )
    findings = scan_absence_reframing(text, agent_name="portfolio_manager")
    assert len(findings) == 1
    assert "portfolio_manager" in findings[0]


def test_flags_insider_silence_as_confidence_id_1350():
    text = (
        "In the absence of reported insider sales... The lack of such noise supports the view "
        "that the current price appreciation is organic ... however, silence here is not a red flag."
    )
    findings = scan_absence_reframing(text, agent_name="News + Macro Analyst")
    assert len(findings) == 1


def test_flags_insider_silence_as_confidence_id_1352():
    text = (
        "In the absence of conflicting insider selling signals... We assume neutral-to-positive "
        "insider stance given the prevailing buy-side pressure."
    )
    findings = scan_absence_reframing(text, agent_name="risk_aggressive")
    assert len(findings) == 1


def test_flags_insider_silence_as_confidence_id_1358():
    text = (
        "There are no public disclosures of insider buying/selling reported... Absence of "
        "negative insider signals reduces downside risk from management dumping shares."
    )
    findings = scan_absence_reframing(text, agent_name="Fundamentals Analyst")
    assert len(findings) == 1


def test_flags_missing_flow_as_proxy_id_1350():
    text = (
        "detailed individual/foreign/institutional net-buy flow data was not retrieved for this "
        "cycle due to scope constraints. In the absence of granular flow data, the price action "
        "itself serves as the primary proxy for sentiment: ... buyers are actively absorbing "
        "selling pressure."
    )
    findings = scan_absence_reframing(text, agent_name="Market Analyst")
    assert len(findings) == 1
    assert "price-action proxy" in findings[0]


def test_flags_missing_flow_as_proxy_id_1352():
    text = (
        "Specific daily net-flow figures for foreign vs. domestic institutions were not "
        "retrieved in this cycle due to prioritization of core price and indicator validation, "
        "but the structural setup implies healthy absorption."
    )
    findings = scan_absence_reframing(text, agent_name="risk_conservative")
    assert len(findings) == 1
    assert "price-action proxy" in findings[0]


def test_flags_coverage_absence_as_opportunity_id_1350():
    text = (
        "'No Analyst Coverage/Liquidity Risk': Lack of coverage reduces information asymmetry "
        "for insiders but increases opportunity for undervalued discovery."
    )
    findings = scan_absence_reframing(text, agent_name="Research Manager")
    assert len(findings) == 1
    assert "bullish opportunity" in findings[0]


def test_flags_coverage_absence_as_bullish_id_1351():
    text = (
        "The lack of analyst coverage (a bearish point for long-term holders) is a bullish point "
        "for short-term traders, as it prevents institutional suppression and allows retail "
        "momentum to drive price action unchecked."
    )
    findings = scan_absence_reframing(text, agent_name="risk_aggressive")
    assert len(findings) == 1
    assert "bullish opportunity" in findings[0]


def test_flags_coverage_absence_as_short_squeeze_id_1359():
    text = (
        "The absence of heavy institutional coverage often correlates with shorter floats. ... "
        "shorts may be getting squeezed, fueling further upside."
    )
    findings = scan_absence_reframing(text, agent_name="portfolio_manager")
    assert len(findings) == 1
    assert "bullish opportunity" in findings[0]


def test_does_not_flag_the_id_1349_rebuttal():
    text = (
        "The bull argues the lack of insider *selling* is positive. In micro-caps, silence is "
        "not affirmation; it is indifference."
    )
    assert scan_absence_reframing(text, agent_name="risk_conservative") == []


def test_does_not_flag_correct_coverage_as_risk_signal_1():
    text = "Lack of analyst coverage means little external scrutiny or guidance for investors."
    assert scan_absence_reframing(text, agent_name="Fundamentals Analyst") == []


def test_does_not_flag_correct_coverage_as_risk_signal_2():
    text = "Zero analyst coverage means no external validation of the financial narrative."
    assert scan_absence_reframing(text, agent_name="Fundamentals Analyst") == []


def test_does_not_flag_coverage_dismissed_as_dangerous():
    text = (
        "The bull dismisses the lack of analyst coverage as a 'contrarian entry' opportunity. "
        "This is dangerous."
    )
    assert scan_absence_reframing(text, agent_name="risk_conservative") == []


def test_empty_text_returns_empty():
    assert scan_absence_reframing("", agent_name="portfolio_manager") == []


def test_dedups_the_same_span_within_one_call():
    span = (
        "the absence of insider selling suggests management does not believe the current price "
        "represents fair value, reducing the likelihood of a fraudulent pump-and-dump exit"
    )
    text = f"{span}. {span}."
    findings = scan_absence_reframing(text, agent_name="portfolio_manager")
    assert len(findings) == 1


def test_templates_are_independent_and_label_flows_into_finding():
    # Template 2 alone (missing flow -> price-action proxy) must not carry
    # Template 1's or Template 3's label text.
    text = (
        "detailed individual/foreign/institutional net-buy flow data was not retrieved for this "
        "cycle due to scope constraints. In the absence of granular flow data, the price action "
        "itself serves as the primary proxy for sentiment: ... buyers are actively absorbing "
        "selling pressure."
    )
    findings = scan_absence_reframing(text, agent_name="Market Analyst")
    assert len(findings) == 1
    assert "price-action proxy" in findings[0]
    assert "management confidence" not in findings[0]
    assert "bullish opportunity" not in findings[0]
