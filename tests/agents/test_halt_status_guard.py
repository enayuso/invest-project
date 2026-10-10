"""app.agents.halt_status_guard — no network, no LLM (A-stage).

Real incident this guards against (2026-09-07, id=1238, 351320.KS): the News
+ Macro Analyst asserted a currently-active trading halt from a several-days-
old headline, and that false claim propagated uncorrected through the
Research Manager, the risk debate, and the Portfolio Manager's final decision
text — even though execution_status_context already said, that same cycle,
that no halt/VI/circuit-breaker was active. This module is a detect-only
recorder, mirroring app.agents.risk_debators.check_neutral_reversal — see its
own module docstring for the full scope and the 21-day live measurement
behind the calibration.
"""

from __future__ import annotations

from app.agents.halt_status_guard import scan_halt_claims

_CONFIRMED = (
    "CONFIRMED by the broker just now: no trading halt, no VI, and no "
    "circuit-breaker is currently active for this ticker. This is live broker "
    "data and it overrides any news item or analyst report that says this stock "
    "is halted or suspended — such an item is historical or a warning about a "
    "possible future halt, never a statement about right now."
)
_COULD_NOT_VERIFY = (
    "Trading-halt status could not be fully verified this cycle "
    "(checks that ran: circuit_breaker, vi). Do not assert either that this "
    "ticker is halted or that it is definitely clear."
)
_BLOCKED = "WARNING: this ticker is currently blocked from any exposure-increasing trade at execution time — BLOCKED: ..."


def test_flags_active_halt_claim_against_confirmed_clear_status():
    text = (
        "Given an active regulatory trading halt creating unmanageable binary gap risk, "
        "we recommend zero exposure."
    )
    findings = scan_halt_claims(text, exec_status=_CONFIRMED, agent_name="portfolio_manager")
    assert len(findings) == 1
    assert "portfolio_manager" in findings[0]


def test_flags_halt_lifts_claim():
    text = "Given the suspension, exposure will remain at zero until the halt lifts."
    findings = scan_halt_claims(text, exec_status=_CONFIRMED, agent_name="risk_conservative")
    assert len(findings) == 1


def test_does_not_flag_dated_historical_mention():
    text = "Note: a preliminary suspension was announced on Sept 4, ahead of the KRX review."
    assert scan_halt_claims(text, exec_status=_CONFIRMED, agent_name="News + Macro Analyst") == []


def test_does_not_flag_warning_framing():
    text = "The KRX issued a preliminary warning of possible trading suspension pending review."
    assert scan_halt_claims(text, exec_status=_CONFIRMED, agent_name="News + Macro Analyst") == []


def test_does_not_flag_already_lifted_framing():
    text = "A release confirmed the trading suspension lifted on August 31; shares resumed normal trading."
    assert scan_halt_claims(text, exec_status=_CONFIRMED, agent_name="News + Macro Analyst") == []


def test_does_not_flag_future_scenario_framing():
    text = (
        "In the hypothetical event the KRX were to impose a trading halt, exposure would be cut "
        "to zero immediately."
    )
    assert scan_halt_claims(text, exec_status=_CONFIRMED, agent_name="risk_aggressive") == []


def test_no_scan_when_status_not_confirmed_clear():
    text = "This stock has an active regulatory trading halt right now."
    assert scan_halt_claims(text, exec_status=_COULD_NOT_VERIFY, agent_name="portfolio_manager") == []
    assert scan_halt_claims(text, exec_status=_BLOCKED, agent_name="portfolio_manager") == []


def test_no_scan_for_empty_exec_status():
    text = "This stock has an active regulatory trading halt right now."
    assert scan_halt_claims(text, exec_status="", agent_name="portfolio_manager") == []


def test_dedups_repeated_span():
    span = "an active regulatory trading halt creating unmanageable binary gap risk"
    text = f"{span}. {span}."
    findings = scan_halt_claims(text, exec_status=_CONFIRMED, agent_name="portfolio_manager")
    assert len(findings) == 1


# --- 2026-09-09 audit: negation / conditional / verification framing
#     (~8/12 domestic decisions tripped the guard on a correctly-negated line) ---

import pytest  # noqa: E402


@pytest.mark.parametrize("text", [
    # negated ("no active" / "not currently subject to" / "no ... reported")
    "I checked the VI status; no active VI halt is currently in effect, so orders flow freely.",
    "The stock is not currently subject to a VI halt, so standard limit-up/limit-down rules apply.",
    "No active Volatility Interruption (VI) halt is reported for this ticker.",
    "get_kr_vi_status confirmed there is no active VI mechanism halting trading right now.",
    # conditional ("if the market is halted")
    "A stop-loss is useless if the market is halted or if liquidity vanishes.",
    # verification / meta ("a check was performed", "traders should verify")
    "A check was performed regarding any active VI halts.",
    "Traders should verify if the stock is currently under a VI halt before executing orders.",
    "Our checks did not include verification of whether the stock is currently subject to a VI mechanism.",
    # past-aftermath ("post-halt")
    "The post-halt repricing was chaotic and the leveraged positioning quickly unwound.",
    # designation-notice / future-risk framing
    "The KRX Investment Caution notice (issued Sept 2) designates the stock, introducing immediate suspension risk.",
    # 2026-09-10 audit, id=1369 — negation-to-term distance 56 chars, exercises the widened _NEG_LEFT window
    "checks confirm that 028050.KS is **not** currently subject to a Volatility Interruption (VI) mechanism.",
])
def test_does_not_flag_negated_conditional_or_verification_framing(text):
    assert scan_halt_claims(text, exec_status=_CONFIRMED, agent_name="Market Analyst") == []


def test_still_flags_a_present_tense_assertion_with_an_unrelated_negation_later():
    # id=1238b shape — a real active-halt claim whose sentence happens to
    # contain an unrelated "not" ("orders will not fill") must still fire.
    text = "There is currently an active trading halt on this ticker and orders will not fill."
    assert len(scan_halt_claims(text, exec_status=_CONFIRMED, agent_name="portfolio_manager")) == 1
