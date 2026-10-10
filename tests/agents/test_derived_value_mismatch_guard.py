"""app.agents.derived_value_mismatch_guard — no network, no LLM (A-stage).

Real incidents this guards against (2026-09-09 domestic AI-decision audit,
P1 item ⑤ "산술 오류 다발" / arithmetic errors): a stated DERIVED numeric
value (PER, stop-loss risk distance %) inconsistent with its own two nearby
GROUND-value citations (EPS+price, stop+price). This is a NUMERIC-DERIVATION
check, not a phrasing-pattern guard — see the module's own docstring for the
full scope, both incidents' exact quotes/arithmetic, and why a
self-correcting rebuttal still fires.
"""

from __future__ import annotations

from app.agents.derived_value_mismatch_guard import scan_derived_value_mismatches


def test_flags_the_id_1357_per_mismatch():
    text = (
        "**Price/Earnings Ratio (PER):** **~2,728x**. This extremely high multiple reflects "
        "the tiny EPS (₩62) relative to the stock price (~₩13,640)."
    )
    findings = scan_derived_value_mismatches(text, agent_name="Fundamentals Analyst")
    assert len(findings) == 1
    assert "220.0" in findings[0]


def test_flags_the_id_1353_stop_distance_mismatch():
    text = (
        "The central pillar of the Buy case is a stop-loss at **2,400 KRW**. The proponent "
        "claims this represents a ~3.5% risk distance from the current price of 2,815 KRW. "
        "This is a severe calculation error."
    )
    findings = scan_derived_value_mismatches(text, agent_name="risk_bear")
    assert len(findings) == 1
    assert "14.7" in findings[0]


def test_does_not_flag_a_correct_per_statement():
    text = "PER of 10.0x. EPS (₩1,000) relative to the stock price (₩10,000)."
    assert scan_derived_value_mismatches(text, agent_name="Fundamentals Analyst") == []


def test_does_not_flag_a_correct_stop_distance_statement():
    text = (
        "a stop-loss at 9,000 KRW. This represents a ~10% risk distance from the current "
        "price of 10,000 KRW."
    )
    assert scan_derived_value_mismatches(text, agent_name="risk_bear") == []


def test_does_not_flag_per_without_nearby_eps_or_price():
    # Real id=1348 sentence shape — PER alone, correctly not flaggable since
    # there is nothing nearby to check it against.
    text = "The stock trades at a PER of 75.94x, a significant premium for the sector."
    assert scan_derived_value_mismatches(text, agent_name="Fundamentals Analyst") == []


def test_does_not_flag_stop_distance_without_nearby_prices():
    text = "The stop-loss represents a ~10% risk distance, a reasonably tight level for this setup."
    assert scan_derived_value_mismatches(text, agent_name="risk_bear") == []


def test_empty_text_returns_empty():
    assert scan_derived_value_mismatches("", agent_name="portfolio_manager") == []


def test_dedups_within_one_call():
    # The id=1357 incident appearing twice, separated (and bracketed) by
    # generous same-character filler so each occurrence's +/-200-char window
    # lands on an identical span (filler is indistinguishable from itself at
    # any offset) without the two occurrences' windows overlapping each
    # other or being clipped asymmetrically by the string's own boundaries —
    # this exercises the guard's window-based dedup directly, rather than
    # relying on the two occurrences happening to sit close enough together
    # to produce byte-identical windows by accident.
    incident = (
        "**Price/Earnings Ratio (PER):** **~2,728x**. This extremely high multiple reflects "
        "the tiny EPS (₩62) relative to the stock price (~₩13,640)."
    )
    fill = "Z" * 300
    text = fill + incident + fill + fill + incident + fill
    findings = scan_derived_value_mismatches(text, agent_name="Fundamentals Analyst")
    assert len(findings) == 1


# --- 2026-09-14 live audit: false positive when a passage discusses TWO bases ---
def test_does_not_flag_the_id_1406_trailing_vs_forward_eps_passage():
    """id=1406 (Fundamentals Analyst): the report explicitly distinguished
    trailing from forward EPS and did BOTH calculations correctly
    (71,800 / 1,147 = 62.6x). The guard paired the stated trailing PER with
    the FORWARD EPS (5,601) — the first `_EPS_RE` match in the window — and
    reported a mismatch against arithmetic the analyst never performed.
    Any candidate combination reconciling the stated value means there is no
    arithmetic error to record."""
    text = (
        "*   FY 2025 EPS: ₩1,147. Price ₩71,800 → PE ~62.6x.\n"
        "*   H1 2026 EPS: ₩5,601. If we annualize this, EPS could be ~₩11,200. At ₩71,800 "
        "price, Forward PE would be ~6.4x.\n"
        "*   **Crucial Distinction:** The displayed PER of 62.6x is almost certainly based on "
        "**Trailing Twelve Months (TTM)** or **Last Fiscal Year (FY 2025)** earnings."
    )
    assert scan_derived_value_mismatches(text, agent_name="Fundamentals Analyst") == []


def test_does_not_flag_stop_distance_when_one_of_two_stop_levels_reconciles():
    # Check 2's analogue of the id=1406 shape: two stop levels named in the
    # same passage, the stated distance derived from the second one.
    text = (
        "The original plan used a stop-loss at 9,500 KRW, but the bull case now proposes a "
        "tightened stop-loss at 9,000 KRW. That represents a ~10% risk distance from the "
        "current price of 10,000 KRW."
    )
    assert scan_derived_value_mismatches(text, agent_name="risk_bull") == []
