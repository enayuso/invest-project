"""app.agents.staleness_narrative_guard — no network, no LLM (A-stage).

Real incident this guards against (2026-09-09 domestic AI decision audit,
id=1351, 198440.KS, Kangdong C&L): the Fundamentals Analyst's report opened
"As of September 9, 2026, Kangdong C&L presents a complex fundamental
picture characterized by significant operational deterioration compared to
its peak years." — presenting 21-month-stale FY2024 data as the company's
CURRENT condition, with zero staleness acknowledgment anywhere in the
report, despite `app.dataflows.kis_provider.get_kr_financials()` already
injecting a staleness warning into the financial-statements text itself.
Contrast id=1353 (088280.KS), same 21-month staleness, whose report
correctly disclosed the true data period throughout and must never be
flagged. This module is a detect-only recorder, the same class as
`app.agents.halt_status_guard.scan_halt_claims` and
`app.agents.technical_claim_guard.scan_technical_claims` — see this module's
own docstring for the full scope.
"""

from __future__ import annotations

from app.agents.staleness_narrative_guard import scan_staleness_narrative

_STALE_RECENCY = {
    "months_stale": 21,
    "latest_label": "202412 (FY, 12 months)",
    "latest_period": "202412",
    "interim": False,
}

_ID_1351_SENTENCE = (
    "As of September 9, 2026, Kangdong C&L presents a complex fundamental "
    "picture characterized by significant operational deterioration "
    "compared to its peak years."
)


def test_flags_the_id_1351_shape():
    findings = scan_staleness_narrative(
        _ID_1351_SENTENCE, recency=_STALE_RECENCY, agent_name="Fundamentals Analyst"
    )
    assert len(findings) == 1
    assert "Fundamentals Analyst" in findings[0]
    assert _ID_1351_SENTENCE.rstrip(".") in findings[0]


def test_does_not_flag_below_the_staleness_threshold():
    recency = dict(_STALE_RECENCY, months_stale=6)
    assert scan_staleness_narrative(_ID_1351_SENTENCE, recency=recency, agent_name="Fundamentals Analyst") == []


def test_does_not_flag_when_recency_is_empty_or_months_stale_is_none():
    assert scan_staleness_narrative(_ID_1351_SENTENCE, recency={}, agent_name="Fundamentals Analyst") == []
    assert (
        scan_staleness_narrative(
            _ID_1351_SENTENCE, recency={"months_stale": None}, agent_name="Fundamentals Analyst"
        )
        == []
    )


def test_does_not_flag_the_id_1353_shape():
    text = (
        "**As of:** September 9, 2026\n"
        "**Source Data Period:** Fiscal Year Ended December 31, 2024 (reported by KIS/KSX)\n\n"
        "### Executive Summary\n"
        "Sonnix (088280.KS) presents a fundamentally distressed profile characterized by "
        "persistent unprofitability."
    )
    assert scan_staleness_narrative(text, recency=_STALE_RECENCY, agent_name="Fundamentals Analyst") == []


def test_does_not_flag_a_report_that_cites_its_own_stale_period_as_the_as_of_clause():
    text = (
        "As of the latest audited period (FY2024), the company generated modest revenues of "
        "17.0 billion KRW but posted significant losses."
    )
    assert scan_staleness_narrative(text, recency=_STALE_RECENCY, agent_name="Fundamentals Analyst") == []


def test_dedups_the_same_span_within_one_call():
    text = f"{_ID_1351_SENTENCE} Some middle filler text. {_ID_1351_SENTENCE}"
    findings = scan_staleness_narrative(text, recency=_STALE_RECENCY, agent_name="Fundamentals Analyst")
    assert len(findings) == 1


def test_empty_report_returns_empty():
    assert scan_staleness_narrative("", recency=_STALE_RECENCY, agent_name="Fundamentals Analyst") == []


def test_current_tense_verb_must_be_reasonably_close_to_the_as_of_clause():
    # 150 filler chars (well past _VERB_WINDOW=100) between the "as of"
    # clause and the nearest present-tense verb — the verb must not count.
    filler = "x" * 150
    text = f"As of September 9, 2026, {filler} presents a distressed profile due to declining margins."
    assert scan_staleness_narrative(text, recency=_STALE_RECENCY, agent_name="Fundamentals Analyst") == []
