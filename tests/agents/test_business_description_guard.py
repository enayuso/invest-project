"""app.agents.business_description_guard — no network, no LLM (A-stage).

Real incident this guards against (2026-09-07, id=1237, 294630.KS/서남): after
identity_gate corrected a mangled name ("Seonam Pharmaceutical Co., Ltd."),
the same report's Risk Factors section went on to describe the company as
"a pharmaceutical firm" — contradicting the report's own correct sector
classification stated elsewhere in the same document. This module is a
detect-only recorder, mirroring app.agents.technical_claim_guard /
app.agents.halt_status_guard — see either's own module docstring for the
full "detect-only, not a gate" pattern this repeats.
"""

from __future__ import annotations

from app.agents.identity_gate import UNNAMED, ForeignListingName
from app.agents.business_description_guard import scan_business_description


def test_flags_id_1237_pharmaceutical_self_description():
    offending_name = "Analysis Report: Seonam Pharmaceutical Co., Ltd."
    company_name = "서남"
    text = (
        "Risk Factors: As a pharmaceutical firm, regulatory changes or healthcare "
        "policy adjustments could impact longer-term prospects."
    )
    findings = scan_business_description(
        text, offending_name=offending_name, company_name=company_name, agent_name="Market Analyst"
    )
    assert len(findings) == 1
    assert "Market Analyst" in findings[0]
    assert "서남" in findings[0]


def test_does_not_flag_hi_steel():
    """Critical amnesty-map test: 하이스틸 legitimately contains 스틸, so a
    correctly-worded "As a steel producer" self-description must NOT be
    flagged even though "steel" is a sector keyword."""
    offending_name = "Haisteel Co., Ltd."
    company_name = "하이스틸"
    text = "As a steel producer, the company benefits from stable domestic demand."
    assert scan_business_description(
        text, offending_name=offending_name, company_name=company_name, agent_name="Market Analyst"
    ) == []


def test_does_not_flag_jw_pharmaceutical():
    """JW신약 contains 신약 (new-drug/pharma concept), so a pharmaceutical
    self-description is legitimate and must not be flagged."""
    offending_name = "JW Pharmaceutical Corp."
    company_name = "JW신약"
    text = "As a pharmaceutical firm, the company focuses on generic drug manufacturing."
    assert scan_business_description(
        text, offending_name=offending_name, company_name=company_name, agent_name="Market Analyst"
    ) == []


def test_does_not_flag_geumho_electric():
    """금호전기 legitimately contains 전기 (electric)."""
    offending_name = "Geumho Electric Co., Ltd."
    company_name = "금호전기"
    text = "As an electric equipment maker, the company supplies components to industrial clients."
    assert scan_business_description(
        text, offending_name=offending_name, company_name=company_name, agent_name="Market Analyst"
    ) == []


def test_returns_empty_for_unnamed_sentinel():
    text = "As a pharmaceutical firm, regulatory changes could impact prospects."
    assert scan_business_description(
        text, offending_name=UNNAMED, company_name="서남", agent_name="Market Analyst"
    ) == []


def test_returns_empty_for_foreign_listing_offender():
    offending_name = ForeignListingName("Appen Ltd. (APPN) — foreign listing", "APPN")
    text = "As a pharmaceutical firm, regulatory changes could impact prospects."
    assert scan_business_description(
        text, offending_name=offending_name, company_name="앱튼", agent_name="Market Analyst"
    ) == []


def test_industry_word_present_but_not_self_description_shape():
    """A mere sector-word mention with no self-description shape — must not
    be flagged; the guard requires an "as/is/being a(n) <keyword> firm/
    company/..." shape, not just the word appearing anywhere."""
    offending_name = "Seonam Pharmaceutical Co., Ltd."
    company_name = "서남"
    text = "The pharmaceutical sector rallied today on positive regulatory news."
    assert scan_business_description(
        text, offending_name=offending_name, company_name=company_name, agent_name="Market Analyst"
    ) == []


def test_scan_is_noop_when_offending_name_empty():
    text = "As a pharmaceutical firm, regulatory changes could impact prospects."
    assert scan_business_description(
        text, offending_name="", company_name="서남", agent_name="Market Analyst"
    ) == []


def test_dedups_repeated_span():
    offending_name = "Seonam Pharmaceutical Co., Ltd."
    company_name = "서남"
    span = "As a pharmaceutical firm, regulatory changes could impact prospects"
    text = f"{span}. {span}."
    findings = scan_business_description(
        text, offending_name=offending_name, company_name=company_name, agent_name="Market Analyst"
    )
    assert len(findings) == 1


def test_still_fires_after_identity_gate_has_already_scrubbed_the_wrong_name():
    # 2026-09-11: identity_gate.scrub_offending_name now replaces recurring
    # wrong-name mentions in the report BODY (see identity_gate.py), but
    # scan_business_description takes `offending_name` as a separate,
    # unmodified parameter used only to look up the sector keyword — it does
    # not re-derive the keyword from the (possibly already-scrubbed) report
    # text. So a report whose wrong-name mentions were already scrubbed to
    # the correct name must still trip this guard on the self-description
    # sentence, proving no call-order change is needed anywhere in
    # pipeline.py (see this module's own module docstring / the 2026-09-11
    # identity_gate scrub work for the background).
    offending_name = "Seonam Pharmaceutical Co., Ltd."
    company_name = "서남"
    # Body text as it would look AFTER scrub_offending_name already replaced
    # every "Seonam Pharmaceutical" mention with the correct company name —
    # the self-description sentence itself is untouched by the scrub (it
    # never named the company, only described what it does).
    text = (
        f'{company_name} [name corrected from "Seonam Pharmaceutical Co., Ltd."] posted stable results. '
        "Risk Factors: As a pharmaceutical firm, regulatory changes or healthcare "
        "policy adjustments could impact longer-term prospects."
    )
    findings = scan_business_description(
        text, offending_name=offending_name, company_name=company_name, agent_name="Market Analyst"
    )
    assert len(findings) == 1
    assert "Market Analyst" in findings[0]
    assert company_name in findings[0]
