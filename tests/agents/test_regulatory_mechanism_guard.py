"""app.agents.regulatory_mechanism_guard — no network, no LLM (A-stage).

Real incident this guards against (id=1380): a confirmed delinquent-filer
determination was narrated as mechanically triggering a cascade of specific
downstream consequences (automated compliance protocols: trading halts,
index exclusion, forced liquidation by institutional funds) — none of which
came from any tool output supplied this cycle. This module is a sibling of
`app.agents.halt_status_guard`, not a modification of it — see its own
module docstring for why the two guards' trigger shapes deliberately do not
overlap.
"""

from __future__ import annotations

from app.agents.regulatory_mechanism_guard import scan_regulatory_mechanism

_ALL_FIELDS = frozenset(
    {"execution_status_context", "kr_risk_extra", "verified_snapshot", "kr_gt", "kr_news_gt", "kr_financials_recency"}
)
_NO_FIELDS: frozenset[str] = frozenset()


def test_flags_id_1380_shape():
    text = (
        "A confirmed delinquent-filer determination for this issuer triggers automated compliance "
        "protocols: trading halts, exclusion from benchmark indices, and forced liquidation by "
        "institutional funds bound by fiduciary or ESG screens."
    )
    findings = scan_regulatory_mechanism(text, supplied_fields=_NO_FIELDS, agent_name="news_macro_analyst")
    assert findings
    assert all("news_macro_analyst" in f for f in findings)


def test_does_not_flag_hedged_eventual_suspension():
    text = "Regulators noted that a designation could eventually lead to a suspension."
    assert scan_regulatory_mechanism(text, supplied_fields=_NO_FIELDS, agent_name="News + Macro Analyst") == []


def test_does_not_flag_bare_designation_fact():
    text = "KRX designated the stock 불성실공시법인 on Sept 9."
    assert scan_regulatory_mechanism(text, supplied_fields=_NO_FIELDS, agent_name="News + Macro Analyst") == []


def test_halt_claim_defers_to_halt_status_guard_when_execution_status_supplied():
    text = "The delinquent-filer designation automatically triggers trading halts for this issuer."
    assert scan_regulatory_mechanism(text, supplied_fields=_ALL_FIELDS, agent_name="portfolio_manager") == []


def test_halt_claim_flagged_when_execution_status_not_supplied():
    text = "The delinquent-filer designation automatically triggers trading halts for this issuer."
    findings = scan_regulatory_mechanism(text, supplied_fields=_NO_FIELDS, agent_name="portfolio_manager")
    assert len(findings) == 1


def test_index_exclusion_flagged_unconditionally():
    text = "The designation automatically mandates exclusion from benchmark indices for this issuer."
    findings = scan_regulatory_mechanism(text, supplied_fields=_ALL_FIELDS, agent_name="risk_conservative")
    assert len(findings) == 1


def test_dedups_repeated_sentence():
    span = (
        "The delinquent-filer status automatically triggers forced liquidation by institutional funds"
    )
    text = f"{span}. {span}."
    findings = scan_regulatory_mechanism(text, supplied_fields=_NO_FIELDS, agent_name="portfolio_manager")
    assert len(findings) == 1


def test_flags_korean_variant():
    text = "불성실공시법인 지정은 자동으로 강제 청산과 지수 편출을 의무적으로 발동한다."
    findings = scan_regulatory_mechanism(text, supplied_fields=_NO_FIELDS, agent_name="Research Manager")
    assert findings
