"""app.agents.sizing_cap_misuse_guard — no network, no LLM (A-stage).

Real incident this guards against (2026-09-09, id=1358, 047040.KS/대우건설):
the risk debate's Neutral turn wrote "Since the position size is effectively
negligible regardless of the rating (due to the risk cap) ... the optimal
choice is to remove the exposure entirely", and the PM's investment_thesis
concluded "the 8.0% ATR compresses any theoretical position to a trivial
share count ... Exiting eliminates exposure to this asymmetric risk
profile." A real Sell of 5 held shares executed minutes later. The prompt
says in five places the added-share cap "only ever reduces shares being
ADDED, never forces a sale of shares already held". This module is a
detect-only recorder — see its own module docstring for scope and
reasoning.
"""

from __future__ import annotations

from app.agents.sizing_cap_misuse_guard import scan_sizing_cap_misuse


def test_flags_id1358_neutral_remove_exposure_from_risk_cap():
    text = (
        "Since the position size is effectively negligible regardless of the rating "
        "(due to the risk cap), and the fundamental/technical outlook is negative, "
        "the optimal choice is to remove the exposure entirely."
    )
    findings = scan_sizing_cap_misuse(text, agent_name="risk_neutral", held=True)
    assert len(findings) == 1
    assert "sizing cap cited as exit rationale" in findings[0]


def test_flags_id1358_pm_thesis_atr_compression_exit():
    text = (
        "The mechanical sizing constraints further validate this view: the 8.0% ATR "
        "compresses any theoretical position to a trivial share count, meaning the "
        "risk-adjusted return is negative. Exiting eliminates exposure to this "
        "asymmetric risk profile."
    )
    findings = scan_sizing_cap_misuse(text, agent_name="portfolio_manager", held=True)
    assert len(findings) == 1


def test_flags_id1358_conservative_zero_exposure_argument():
    text = (
        "The Mechanical Argument for Zero Exposure\n"
        "Both opposing sides fail to recognize that the volatility-adjusted sizing "
        "makes any position economically irrational."
    )
    assert len(scan_sizing_cap_misuse(text, agent_name="risk_conservative", held=True)) == 1


def test_no_scan_when_flat():
    text = (
        "Since the position size is effectively negligible regardless of the rating "
        "(due to the risk cap), the optimal choice is to remove the exposure entirely."
    )
    assert scan_sizing_cap_misuse(text, agent_name="risk_neutral", held=False) == []


def test_does_not_flag_correct_add_sizing_description():
    # The cap genuinely limits a NEW purchase — naming it here is correct.
    text = (
        "The combination of the wide stop and the hard risk cap restricts the new "
        "purchase to barely 5 shares, so a fresh entry is not worthwhile."
    )
    assert scan_sizing_cap_misuse(text, agent_name="risk_neutral", held=True) == []


def test_does_not_flag_prompt_restatement():
    for text in (
        "This can shrink the add on top of the volatility-based scaling above; it "
        "never forces a sale of shares already held.",
        "These portfolio-level caps can only further reduce an exposure increase, "
        "and only ever reduces shares being added; they never force a sale of shares already held.",
        "Position-size mechanics are never a reason to trade, and the risk-cap "
        "framework is never a reason to sell.",
    ):
        assert scan_sizing_cap_misuse(text, agent_name="portfolio_manager", held=True) == []


def test_does_not_flag_cap_mention_without_exit_conclusion():
    text = (
        "The 14-day ATR of 8.0% scales an Overweight allocation down to 19% of equity, "
        "and the risk cap on added shares limits the entry further."
    )
    assert scan_sizing_cap_misuse(text, agent_name="risk_aggressive", held=True) == []


def test_does_not_flag_exit_reasoning_unrelated_to_cap():
    text = (
        "The failed breakout and the 266% debt-to-equity ratio justify a full exit "
        "here; the fundamentals have deteriorated past the point of holding."
    )
    assert scan_sizing_cap_misuse(text, agent_name="portfolio_manager", held=True) == []


def test_dedups_repeated_span():
    span = (
        "Given the risk cap, the optimal choice is to liquidate the position entirely"
    )
    text = f"{span}. {span}."
    assert len(scan_sizing_cap_misuse(text, agent_name="risk_neutral", held=True)) == 1


def test_empty_text():
    assert scan_sizing_cap_misuse("", agent_name="x", held=True) == []
