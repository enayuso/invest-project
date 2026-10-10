"""app.agents.verified_price_guard — no network, no LLM (A-stage).

Real incident this guards against (2026-09-08, id=1314, 033790.KS/피노): the
News + Macro Analyst — handed the same real verified snapshot as the Market
Analyst — wrote "Verified Close (Sept 6): 1,330 KRW" while the real snapshot
put the stock around 9,120-9,760 KRW that session, then built an entire
bearish thesis on a fabricated ~1,300 KRW price universe. This module is a
detect-only recorder, mirroring app.agents.halt_status_guard.scan_halt_claims
— see its own module docstring for the full scope and the 250-decision
calibration behind it.
"""

from __future__ import annotations

from app.agents.verified_price_guard import scan_verified_price_claims

_SNAPSHOT_CLOSED = "VERIFIED SNAPSHOT for 033790.KS as of 2026-09-05: close=9120.00 high=9760.00 low=9050.00 volume=120000"
_SNAPSHOT_IN_PROGRESS = (
    "VERIFIED SNAPSHOT for 033790.KS as of 2026-09-08 (SESSION IN PROGRESS — not a closed daily "
    "bar): last=9120.00 session_high=9760.00 session_low=9120.00 volume_so_far=45000"
)


def test_flags_fabricated_verified_close():
    text = "Verified Close (Sept 6): 1,330 KRW (-2.22%)."
    findings = scan_verified_price_claims(text, verified_snapshot=_SNAPSHOT_CLOSED, agent_name="News + Macro Analyst")
    assert len(findings) == 1
    assert "News + Macro Analyst" in findings[0]
    assert "9,120" in findings[0] or "9120" in findings[0]


def test_flags_fabricated_verified_price_against_in_progress_snapshot():
    text = "Verified Price: 1,290 KRW represents a break below the psychological level."
    findings = scan_verified_price_claims(
        text, verified_snapshot=_SNAPSHOT_IN_PROGRESS, agent_name="News + Macro Analyst"
    )
    assert len(findings) == 1


def test_does_not_flag_matching_verified_snapshot():
    text = "The verified snapshot shows the stock trading at 9,120 KRW, near its session low."
    assert scan_verified_price_claims(text, verified_snapshot=_SNAPSHOT_CLOSED, agent_name="Market Analyst") == []


def test_does_not_flag_within_tolerance_drift():
    # ~20% off the real snapshot -- within a plausible multi-session drift,
    # well under the tolerance an actual 7x-off fabrication like id=1314 blows past.
    text = "Verified Close (Sept 6): 7,300 KRW."
    assert scan_verified_price_claims(text, verified_snapshot=_SNAPSHOT_CLOSED, agent_name="News + Macro Analyst") == []


def test_does_not_flag_unrelated_verified_mention():
    text = "The verified financial statements show no material change to the balance sheet."
    assert scan_verified_price_claims(text, verified_snapshot=_SNAPSHOT_CLOSED, agent_name="Fundamentals Analyst") == []


def test_no_scan_when_snapshot_unparseable():
    text = "Verified Close (Sept 6): 1,330 KRW."
    assert scan_verified_price_claims(text, verified_snapshot="", agent_name="News + Macro Analyst") == []
    assert scan_verified_price_claims(text, verified_snapshot="no data here", agent_name="News + Macro Analyst") == []


def test_no_scan_for_empty_text():
    assert scan_verified_price_claims("", verified_snapshot=_SNAPSHOT_CLOSED, agent_name="News + Macro Analyst") == []


def test_dedups_repeated_span():
    span = "Verified Close (Sept 6): 1,330 KRW represents a break below the psychological level"
    text = f"{span}. {span}."
    findings = scan_verified_price_claims(text, verified_snapshot=_SNAPSHOT_CLOSED, agent_name="News + Macro Analyst")
    assert len(findings) == 1
