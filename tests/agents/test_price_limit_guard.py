"""app.agents.price_limit_guard — no network, no LLM (A-stage).

Real incident this guards against (2026-09-08, id=1313, 082850.KS): the
Market Analyst invented a full intraday narrative — "session high of 10,000
KRW to a low of 5,000 KRW, a staggering 50% drawdown... settling at 6,170
KRW" (+37.5% vs the correctly-stated previous close of 4,485 KRW) — with no
VERIFIED SNAPSHOT line anywhere in the report, and that fabrication
propagated uncorrected into the risk debate and the Portfolio Manager's own
executive summary. This module is a detect-only recorder, mirroring
app.agents.halt_status_guard.scan_halt_claims — see its own module docstring
for the full scope and the 800-report calibration behind it.
"""

from __future__ import annotations

from app.agents.price_limit_guard import scan_price_limit_claims


def test_flags_intraday_drawdown_over_limit():
    text = (
        "The stock fell from a session high of 10,000 KRW to a low of 5,000 KRW"
        "—a staggering 50% drawdown—before settling at 6,170 KRW."
    )
    findings = scan_price_limit_claims(text, agent_name="Market Analyst")
    assert len(findings) == 1
    assert "Market Analyst" in findings[0]
    assert "50%" in findings[0]


def test_flags_same_day_close_change_over_limit():
    text = "Change: +37.5% (from previous close of 4,485 KRW), ending the day with a net gain."
    findings = scan_price_limit_claims(text, agent_name="Market Analyst")
    assert len(findings) == 1


def test_flags_daily_limit_up_claim():
    text = "Notably, the stock hit the KOSDAQ daily limit-up (+33% roughly, though specific limits vary)."
    findings = scan_price_limit_claims(text, agent_name="News + Macro Analyst")
    assert len(findings) == 1


def test_does_not_flag_within_band_intraday_move():
    text = "The stock is down roughly 12.7% from its session high of 4,315 KRW."
    assert scan_price_limit_claims(text, agent_name="News + Macro Analyst") == []


def test_does_not_flag_exact_boundary_move():
    # 30.0% is the legitimate limit-up/limit-down boundary itself, not a
    # violation — only strictly above the tolerance-adjusted threshold fires.
    text = "The stock hit its daily limit-up, closing at +30.0% intraday."
    assert scan_price_limit_claims(text, agent_name="Market Analyst") == []


def test_does_not_flag_multiyear_comparison():
    text = "This follows a massive surge in 2024 (+60.70%) and an explosive recovery in 2023 (+207.89%)."
    assert scan_price_limit_claims(text, agent_name="Fundamentals Analyst") == []


def test_does_not_flag_fundamentals_ratio():
    text = "Borrowing dependency (interest-bearing debt / total assets) rose slightly to 30.71% in H1 2026."
    assert scan_price_limit_claims(text, agent_name="Fundamentals Analyst") == []


def test_does_not_flag_range_position_phrasing():
    text = (
        "The stock is trading in the bottom third of its daily range "
        "(approximately 32.3% of the way up from the low)."
    )
    assert scan_price_limit_claims(text, agent_name="Market Analyst") == []


# --- 2026-09-09 audit: two false-positive framings ---

def test_does_not_flag_op_profit_abbreviation():
    # id=1356 (두올): "-52% Op Profit" is a P&L line item, not a price move —
    # "operating profit" was already excluded, "Op Profit" was not.
    text = (
        "The market is reacting to real earnings decay (-52% Op Profit) and to "
        "real selling pressure with a complete reversal of intraday gains."
    )
    assert scan_price_limit_claims(text, agent_name="risk_neutral") == []


def test_does_not_flag_distance_between_low_and_high():
    # id=1360: "89% of the distance between its session low and high" is a
    # position WITHIN today's range, not a same-session CHANGE.
    text = "The current price of 232,000 is approximately 89% of the distance between its session low and high."
    assert scan_price_limit_claims(text, agent_name="Market Analyst") == []


def test_does_not_flag_monthly_headline_move():
    text = "While the stock experienced a massive surge (+29.88%) yesterday driven by specific product announcements."
    # Below threshold in magnitude regardless — kept as a within-band regression case.
    assert scan_price_limit_claims(text, agent_name="News + Macro Analyst") == []


# --- 2026-09-17 audit: retracement-fraction phrasing (id=1542) ---

def test_does_not_flag_retracement_fraction_of_gains():
    # id=1542 (432980.KS): "50% of its intraday gains" is the FRACTION of the
    # day's gain given back, not a same-session percentage CHANGE — the same
    # sentence's own actual move was 12.8%, well under the ±30% band.
    text = "The stock failed to hold even 50% of its intraday gains, though it is still up 12.8% on the day."
    assert scan_price_limit_claims(text, agent_name="risk_conservative") == []


def test_does_not_flag_retracement_fraction_of_losses():
    text = "The stock recovered 40% of its session losses by the close."
    assert scan_price_limit_claims(text, agent_name="risk_conservative") == []


def test_still_flags_staggering_drawdown_near_session_high_low_after_exclusion():
    # Regression guard: the new "of its/the gains/losses" exclusion must not
    # widen far enough to also swallow the original id=1313 true positive.
    text = (
        "The stock fell from a session high of 10,000 KRW to a low of 5,000 KRW"
        "—a staggering 50% drawdown—before settling at 6,170 KRW."
    )
    findings = scan_price_limit_claims(text, agent_name="Market Analyst")
    assert len(findings) == 1
    assert "50%" in findings[0]


def test_no_scan_for_empty_text():
    assert scan_price_limit_claims("", agent_name="Market Analyst") == []


def test_dedups_repeated_span():
    span = "a staggering 50% intraday drawdown before settling at 6,170 KRW"
    text = f"{span}. {span}."
    findings = scan_price_limit_claims(text, agent_name="Market Analyst")
    assert len(findings) == 1
