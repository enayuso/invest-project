"""app.agents.technical_claim_guard — no network, no LLM (A-stage).

Real incident this guards against (2026-09-07, 21-day audit, 255 domestic
Market Analyst reports): 35.7% had no real indicator value at all (the model
skipped get_indicators, not a data-availability gap), ~2.4% of which were
papered over with a fabricated hedged value ("Estimated RSI: ~35-40") instead
of an honest "unavailable" disclosure; a similar hedged moving-average
pattern showed up in ~5/255. Separately, id=1237 restated the KOSPI index's
own day-change as this stock's "contribution to the KOSPI index". This module
is a detect-only recorder, mirroring app.agents.halt_status_guard — see its
own module docstring for the full scope.
"""

from __future__ import annotations

from app.agents.technical_claim_guard import scan_technical_claims

_NOT_AVAILABLE = (
    "NOT AVAILABLE this cycle (insufficient price history or a vendor error). "
    "No RSI, moving-average or MACD value exists for this ticker right now."
)
_REAL_BLOCK = (
    "Technical indicators for 005930.KS (source: KIS):\n"
    "SMA(20)=71000.00  SMA(50)=69500.00  RSI(14)=41.2  MACD=120.50 (signal=110.00 hist=10.50)"
)


def test_flags_hedged_rsi_against_not_available():
    """id=1238-shaped replay: a hedged RSI claim with nothing to check it
    against (indicator block explicitly NOT AVAILABLE)."""
    text = "Estimated RSI: ~35-40 (oversold territory approaching extreme lows)."
    findings = scan_technical_claims(text, indicator_block=_NOT_AVAILABLE, agent_name="Market Analyst")
    assert len(findings) == 1
    assert "Market Analyst" in findings[0]


def test_flags_hedged_rsi_vs_true_value():
    text = "RSI stands at approximately 28.5, suggesting oversold conditions."
    findings = scan_technical_claims(text, indicator_block=_REAL_BLOCK, agent_name="Market Analyst")
    assert len(findings) == 1
    assert "28.5" in findings[0]
    assert "41.2" in findings[0]


def test_does_not_flag_unhedged_rsi_matching_block():
    """Critical false-positive guard — the healthy majority case: the report
    quotes the real value with no hedge word."""
    text = "RSI(14) is 41.2, in neutral territory, neither overbought nor oversold."
    assert scan_technical_claims(text, indicator_block=_REAL_BLOCK, agent_name="Market Analyst") == []


def test_does_not_flag_korean_unhedged_rsi():
    text = "RSI(14)는 41.2로 중립적인 구간에 위치해 있습니다."
    assert scan_technical_claims(text, indicator_block=_REAL_BLOCK, agent_name="Market Analyst") == []


def test_flags_hedged_ma_crossover():
    """id=1238 replay."""
    text = "Short-Term MA (e.g., 5-day): Likely crossed below longer-term averages."
    findings = scan_technical_claims(text, indicator_block=_NOT_AVAILABLE, agent_name="Market Analyst")
    assert len(findings) == 1


def test_flags_korean_hedged_ma():
    """id=843-shaped replay."""
    text = "20일 이동평균선(약 47~48만원 대 추정)에서 지지받고 있는 것으로 보인다."
    findings = scan_technical_claims(text, indicator_block=_NOT_AVAILABLE, agent_name="Market Analyst")
    assert len(findings) == 1


def test_flags_index_contribution_claim():
    """id=1237 replay."""
    text = "The stock showed a +3.50% contribution to the broader KOSPI index today."
    findings = scan_technical_claims(text, indicator_block=_REAL_BLOCK, agent_name="Market Analyst")
    assert len(findings) == 1
    assert "index-contribution" in findings[0]


def test_does_not_flag_legitimate_regime_quoting():
    """Critical false-positive guard — must not fire on ordinary regime-line
    quoting of the KOSPI/KOSDAQ index's own change."""
    text = "KOSPI: 6,902.44 (+215.23, +3.22%) was broadly higher today, lifting sentiment across the board."
    assert scan_technical_claims(text, indicator_block=_REAL_BLOCK, agent_name="Market Analyst") == []


def test_does_not_flag_honest_unavailable_disclosure():
    text = "Specific SMA/RSI/MACD values were not retrieved in this cycle; reasoning from price action alone."
    assert scan_technical_claims(text, indicator_block=_NOT_AVAILABLE, agent_name="Market Analyst") == []


def test_dedups_repeated_span():
    span = "Estimated RSI: ~35-40 approaching oversold territory"
    text = f"{span}. {span}."
    findings = scan_technical_claims(text, indicator_block=_NOT_AVAILABLE, agent_name="Market Analyst")
    assert len(findings) == 1


def test_empty_text_returns_empty():
    assert scan_technical_claims("", indicator_block=_NOT_AVAILABLE, agent_name="Market Analyst") == []
