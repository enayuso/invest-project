"""app.agents.refusal_guard — no network, no LLM (A-stage).

Real incident this guards against (measured 2026-09-14 over 1,008 stored
decisions): the production model writes an "As an AI model, I do not have
access to real-time financial data... I cannot provide specific financial
details" preamble instead of analyzing, wrongly treating the real analysis
date it was handed as beyond its training cut-off. Fundamentals Analyst
21/1008 (2.1%; 1.5% live, 13.9% backtest), Market Analyst 1, News + Macro
Analyst 1 (LIVE), Sentiment Analyst 0. The DATE_GROUNDING system-prompt
clause has told the model not to do this since 2026-08-04 and it still
happens, so this module is a detect-only recorder, mirroring
app.agents.technical_claim_guard — see its module docstring for the scope.

Every positive fixture below is a VERBATIM snippet pulled from the
production database on 2026-09-14; they are the ground truth this must catch.
"""

from __future__ import annotations

from app.agents.refusal_guard import scan_refusal

# --- verbatim production refusals (2026-09-14) ------------------------------
_MSFT_BACKTEST = (
    "**Important Note:** As an AI model, I do not have access to real-time financial data, "
    "future dates, or live market feeds. Therefore, I cannot provide specific financial "
    "metrics, analyst recommendation"
)
_AAPL_BACKTEST = (
    "**Important Note:** As an AI model, I do not have access to real-time financial data, "
    "future projections beyond my training cut-off (which ends prior to July 2026), or live "
    "market feeds. Therefore, I"
)
_KR_LIVE_NEWS = (
    "te:** The requested analysis date is **September 4, 2026**. As an AI model trained prior "
    "to this date, I cannot verify future events or possess real-time data for September 2026. "
    "The following analysi"
)
_NVDA_BACKTEST_A = (
    "**Important Note:** As an AI model, I cannot predict future events or provide specific "
    "financial forecasts beyond my training data cutoff. My knowledge does not extend to "
    "providing specific financial"
)
_NVDA_BACKTEST_B = (
    "**Important Note:** As an AI model, I do not have access to future data, specifically "
    "regarding NVIDIA's financial performance or stock status beyond my last update. "
    "Therefore, I cannot provide a comp"
)

# A full analyst body, long enough to push a matched disclaimer over the
# 400-char label threshold into refusal_hedged.
_LONG_BODY = (
    " The following analysis is based on the supplied verified snapshot and the exchange-"
    "reported financials provided in this cycle. Revenue for the most recent reported quarter "
    "was 74.1 trillion KRW, up 12.4% year over year, with an operating margin of 18.2%. The "
    "balance sheet carries a net cash position, and the quick ratio stands at 1.84. Valuation "
    "sits at 14.2x trailing earnings against a five-year median of 16.0x. Risks include "
    "memory-price cyclicality and customer concentration. Overall the setup is constructive "
    "into the next reporting period."
)


def test_flags_msft_backtest_refusal():
    """Verbatim production replay (MSFT, backtest). Two separate sentences
    each carry a marker ('as an ai model', then 'cannot provide specific
    financial'), so two findings are correct here — the guard dedups per
    sentence span, not per report."""
    findings = scan_refusal(_MSFT_BACKTEST, agent_name="Fundamentals Analyst")
    assert len(findings) == 2
    assert all(f.startswith("Fundamentals Analyst: refusal_total — ") for f in findings)


def test_flags_aapl_backtest_refusal():
    """Verbatim production replay (AAPL, backtest) — 'as an ai model' and
    'training cut-off' both fire inside one sentence; one finding, not two."""
    findings = scan_refusal(_AAPL_BACKTEST, agent_name="Fundamentals Analyst")
    assert len(findings) == 1


def test_flags_nvda_cannot_provide_specific_financial():
    """Verbatim production replay (NVDA, backtest) — also exercises the
    'cannot provide specific financial' pattern."""
    findings = scan_refusal(_NVDA_BACKTEST_A, agent_name="Fundamentals Analyst")
    assert findings
    assert all("Fundamentals Analyst" in f for f in findings)


def test_flags_nvda_my_last_update():
    """Verbatim production replay (NVDA, backtest) — 'my last update'."""
    findings = scan_refusal(_NVDA_BACKTEST_B, agent_name="Fundamentals Analyst")
    assert findings


def test_flags_live_kr_news_macro_refusal():
    """Verbatim production replay (015760.KS, LIVE, News + Macro Analyst) —
    the one live non-Fundamentals case, which is why the guard covers all
    four analysts rather than only the Fundamentals Analyst."""
    findings = scan_refusal(_KR_LIVE_NEWS, agent_name="News + Macro Analyst")
    assert len(findings) == 1
    assert "News + Macro Analyst" in findings[0]


def test_live_kr_news_macro_with_analysis_is_hedged():
    """That same live case continued into a real report — with enough
    trailing analysis it must be labelled refusal_hedged, not
    refusal_total."""
    findings = scan_refusal(_KR_LIVE_NEWS + _LONG_BODY, agent_name="News + Macro Analyst")
    assert len(findings) == 1
    assert "refusal_hedged" in findings[0]


def test_tier_total_when_nothing_follows():
    """Same marker, nothing after the disclaimer sentence => refusal_total."""
    findings = scan_refusal(
        "**Important Note:** As an AI model, I cannot provide this analysis.",
        agent_name="Fundamentals Analyst",
    )
    assert len(findings) == 1
    assert "refusal_total" in findings[0]


def test_does_not_flag_legitimate_data_quality_caveat():
    """Critical false-positive guard — live reports frequently open with
    '**Important Note:**' carrying a legitimate data-quality caveat. The
    guard must key on the five refusal patterns ONLY, never on the opener."""
    text = (
        "**Important Note:** The reported quick ratio is 0.00%, which appears to be a data "
        "availability issue rather than a genuine liquidity position." + _LONG_BODY
    )
    assert scan_refusal(text, agent_name="Fundamentals Analyst") == []


def test_does_not_flag_ordinary_clean_report():
    text = (
        "Samsung Electronics reported revenue of 74.1 trillion KRW for the quarter, up 12.4% "
        "year over year, with an operating margin of 18.2% and a quick ratio of 1.84. "
        "Valuation is 14.2x trailing earnings versus a five-year median of 16.0x."
    )
    assert scan_refusal(text, agent_name="Market Analyst") == []


def test_empty_text_returns_empty():
    assert scan_refusal("", agent_name="Sentiment Analyst") == []


def test_tier_is_one_per_report_not_per_span():
    """The tier is per REPORT, from the first refusal marker — a report that
    disclaims early, analyses at length, then drops a second marker near the
    very end must label BOTH findings refusal_hedged. Computing the tier per
    span would label the trailing one refusal_total, i.e. one report giving
    two contradictory answers to the question the tier exists to answer."""
    text = _AAPL_BACKTEST + _LONG_BODY + " As an AI assistant, I note this caveat again."
    findings = scan_refusal(text, agent_name="Fundamentals Analyst")
    assert len(findings) >= 2
    assert all("refusal_hedged" in f for f in findings)
    assert not any("refusal_total" in f for f in findings)
