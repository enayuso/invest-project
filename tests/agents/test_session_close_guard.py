"""app.agents.session_close_guard — no network, no LLM (A-stage).

Real incident this guards against (2026-09-09, id=1358, 047040.KS/대우건설):
the risk debate's round-2 turns wrote "closed near the session low ...
textbook distribution pattern" at 10:16 KST, 46 minutes into the session —
the stated #1 reason the Neutral verdict flipped Underweight -> Sell, which
executed against a real position minutes later. This module is a
detect-only recorder, mirroring app.agents.halt_status_guard — see its own
module docstring for the full scope and reasoning.
"""

from __future__ import annotations

from app.agents.session_close_guard import scan_session_close_claims

# The real injected marker text's shape (session_state.intraday_narration_note) —
# reconstructed literally rather than calling the function, so this test does
# not depend on market-hours/clock state.
_SESSION_NOTE = (
    "SESSION IN PROGRESS: 047040.KS's regular session is still open as of "
    "2026-09-09. Today has no open, no close, and no daily range yet. "
    "Any figure for today is a live snapshot: the latest trade price is NOT "
    "the open and NOT a close, and today's high/low bound only the part of "
    "the session traded so far. Do not write that the stock opened, closed, "
    "finished, settled, or concluded trading today, do not call today's range "
    "the daily or closing range, and do not place the latest price 'near the "
    "low/high of the day' as if the day were over. Say 'so far today' or 'as "
    "of now' instead. This applies to your own prose and to any earlier "
    "report or transcript quoted below that says otherwise."
)


def test_flags_id1358_closed_assertion_span():
    # Literal span from id=1358's risk_neutral round-2 turn. Matches BOTH
    # families (a completion verb AND a completed-candle "session low" read)
    # — by design these are measured independently, so both fire.
    text = (
        "The stock surged 6% to 21,300 KRW but closed near the session low "
        "(20,100 KRW), rejecting the breakout."
    )
    findings = scan_session_close_claims(text, session_note=_SESSION_NOTE, agent_name="risk_neutral")
    assert len(findings) == 2
    assert all("risk_neutral" in f for f in findings)
    assert {"closed_assertion", "completed_candle"} == {f.split("[")[1].split("]")[0] for f in findings}


def test_flags_id1358_distribution_pattern_span():
    # Literal span from id=1358's risk_conservative round-2 turn.
    text = (
        "This is a textbook distribution pattern—early buying followed by heavy "
        "selling pressure that pushes the price back to the lower third of the range."
    )
    findings = scan_session_close_claims(text, session_note=_SESSION_NOTE, agent_name="risk_conservative")
    assert len(findings) == 1
    assert "completed_candle" in findings[0]


def test_flags_id1358_opened_high_but_closed_span():
    # Also matches both families — see test_flags_id1358_closed_assertion_span.
    text = "On September 9, 2026, the stock opened higher, peaked at 21,300 KRW, but closed near the session low at 20,100 KRW."
    findings = scan_session_close_claims(text, session_note=_SESSION_NOTE, agent_name="portfolio_manager")
    assert len(findings) == 2


def test_does_not_flag_note_quoted_back_verbatim():
    # The note itself contains "closed"/"finished"/"settled"/"concluded trading" —
    # a node quoting its own instruction must never count as a violation of it.
    assert scan_session_close_claims(_SESSION_NOTE, session_note=_SESSION_NOTE, agent_name="market_analyst") == []


def test_does_not_flag_prior_session_close():
    text = "The previous close of 20,300 KRW sits just above today's opening print."
    assert scan_session_close_claims(text, session_note=_SESSION_NOTE, agent_name="market_analyst") == []


def test_does_not_flag_close_as_adverb():
    text = "Price is trading close to the 20-day moving average, a level to watch."
    assert scan_session_close_claims(text, session_note=_SESSION_NOTE, agent_name="market_analyst") == []


def test_does_not_flag_conditional_future_close():
    text = "If it closes below 19,800 KRW today, we would consider that a bearish signal."
    assert scan_session_close_claims(text, session_note=_SESSION_NOTE, agent_name="risk_aggressive") == []


def test_does_not_flag_correctly_qualified_intraday_framing():
    text = "So far today the stock has traded between 20,100 and 21,300 KRW, with no daily close yet."
    assert scan_session_close_claims(text, session_note=_SESSION_NOTE, agent_name="market_analyst") == []


def test_does_not_flag_negated_claim():
    text = "The stock has not closed yet — today's session remains open with no final print."
    assert scan_session_close_claims(text, session_note=_SESSION_NOTE, agent_name="market_analyst") == []


def test_no_scan_for_empty_session_note():
    text = "The stock closed near the session low, a textbook distribution pattern."
    assert scan_session_close_claims(text, session_note="", agent_name="portfolio_manager") == []


def test_dedups_repeated_span():
    # "concluded trading" matches only Family A, so this isolates the dedup
    # behavior from the both-families overlap exercised above.
    span = "the stock concluded trading at 20,100 KRW today"
    text = f"{span}. {span}."
    findings = scan_session_close_claims(text, session_note=_SESSION_NOTE, agent_name="portfolio_manager")
    assert len(findings) == 1


def test_does_not_flag_through_the_close():
    # 2026-09-10 audit, id=1365, Market Analyst: forward continuation
    # ("all the way until closing time"), not an assertion the close happened.
    text = (
        "This allows the current price discovery process to continue "
        "uninterrupted through the close"
    )
    assert scan_session_close_claims(text, session_note=_SESSION_NOTE, agent_name="market_analyst") == []


def test_does_not_flag_morning_session():
    # 2026-09-10 audit, id=1366, risk debate turn: KRX has no formal AM/PM
    # session split — colloquial for "so far this morning", same shape as
    # the already-exempt "so far today".
    text = "It closed the morning session near the midpoint of its range."
    assert scan_session_close_claims(text, session_note=_SESSION_NOTE, agent_name="risk_neutral") == []


def test_does_not_flag_monitor_the_close_conditional():
    # 2026-09-10 audit, id=1371, News + Macro Analyst: "if" precedes "close"
    # here, not the reverse, so this is forward/conditional framing.
    text = (
        "Monitor the close: if the stock holds above 1,000 KRW with "
        "stabilizing volume, the bullish trend remains intact."
    )
    assert scan_session_close_claims(text, session_note=_SESSION_NOTE, agent_name="news_macro_analyst") == []


def test_does_not_flag_hedged_despite_distribution_patterns():
    # 2026-09-10 audit, id=1371, Bear debater: referencing a risk the Bull is
    # (in the Bear's telling) dismissing, not directly asserting the pattern
    # occurred — "despite X" frames X as a hedged caveat.
    text = (
        "**Conclusion**\nThe bull case bets on a miracle: that 아이씨에이치 can "
        "avoid dilution, execute complex supply chain integrations instantly, "
        "and maintain its technical uptrend despite distribution patterns."
    )
    assert scan_session_close_claims(text, session_note=_SESSION_NOTE, agent_name="research_bear") == []


def test_still_flags_unhedged_candle_pattern_claim_without_despite():
    # 2026-09-10 audit, id=1370: an unhedged, present-tense candle-completion
    # claim while the session is open is the SAME shape as the id=1358
    # incident this guard exists for — explicitly OUT OF SCOPE for the
    # 2026-09-10 false-positive fixes above. Must keep firing.
    text = (
        "This long upper wick, occurring in a thinly covered small-cap with no "
        "institutional anchor, is a textbook speculative blow-off."
    )
    findings = scan_session_close_claims(text, session_note=_SESSION_NOTE, agent_name="Research Manager")
    assert len(findings) == 1
    assert "completed_candle" in findings[0]


def test_flags_id1387_pm_paraphrased_distribution_without_pattern_word():
    # 2026-09-11 audit, id=1387 (217820.KS): the PM's own investment_thesis
    # paraphrased "textbook distribution pattern" without the word "pattern"
    # itself ("shows classic distribution as shares dumped..."). Confirmed a
    # plain regex-vocabulary miss, not an exemption-window suppression — the
    # two risk debaters' near-identical "distribution pattern" phrasing in
    # the SAME decision WAS already caught.
    text = (
        "the intraday price action shows classic distribution as shares "
        "dumped from 3,300 KRW to 2,805 KRW."
    )
    findings = scan_session_close_claims(text, session_note=_SESSION_NOTE, agent_name="portfolio_manager")
    assert findings
    assert any("completed_candle" in f for f in findings)


def test_does_not_flag_id1435_past_dated_close_month_name():
    # 2026-09-15 live false positive, decision id=1435 (067290.KS/JW신약):
    # "Sept 10"/"Sept 11" are genuinely PAST trading days relative to the
    # session note's own today (2026-09-15) — a correct factual statement,
    # not a claim that today's still-open session has closed.
    note = (
        "SESSION IN PROGRESS: 067290.KS's regular session is still open as of "
        "2026-09-15. Today has no open, no close, and no daily range yet."
    )
    text = "The stock closed at 3,620 on Sept 10 and 3,515 on Sept 11"
    assert scan_session_close_claims(text, session_note=note, agent_name="market_analyst") == []


def test_iso_and_month_name_past_dates_are_equivalent():
    # Pins the whole fix: the SAME factual claim, written as an ISO date vs a
    # month-name date, must produce the SAME (clean) result — a future change
    # must not let these two spellings diverge again.
    note = (
        "SESSION IN PROGRESS: 067290.KS's regular session is still open as of "
        "2026-09-15. Today has no open, no close, and no daily range yet."
    )
    iso_text = "The stock closed at 3,515 on 2026-09-11"
    month_text = "The stock closed at 3,515 on September 11"
    iso_findings = scan_session_close_claims(iso_text, session_note=note, agent_name="market_analyst")
    month_findings = scan_session_close_claims(month_text, session_note=note, agent_name="market_analyst")
    assert iso_findings == [] == month_findings


def test_does_not_flag_full_month_name_past_date():
    note = (
        "SESSION IN PROGRESS: 067290.KS's regular session is still open as of "
        "2026-09-15. Today has no open, no close, and no daily range yet."
    )
    text = "The stock closed at 3,515 on September 11"
    assert scan_session_close_claims(text, session_note=note, agent_name="market_analyst") == []


def test_does_not_flag_korean_past_date():
    note = (
        "SESSION IN PROGRESS: 067290.KS's regular session is still open as of "
        "2026-09-15. Today has no open, no close, and no daily range yet."
    )
    text = "9월 11일 종가 3,515원"
    assert scan_session_close_claims(text, session_note=note, agent_name="market_analyst") == []


def test_still_flags_today_month_name_date():
    # TIGHTENING: today's own date, written as a month name, must NOT exempt.
    note = (
        "SESSION IN PROGRESS: 067290.KS's regular session is still open as of "
        "2026-09-15. Today has no open, no close, and no daily range yet."
    )
    text = "The stock closed at 3,990 on September 15"
    findings = scan_session_close_claims(text, session_note=note, agent_name="market_analyst")
    assert findings
    assert "closed_assertion" in findings[0]


def test_still_flags_today_iso_date():
    # TIGHTENING vs the pre-2026-09-15 behavior: today's own ISO date used to
    # wrongly exempt a same-day claim; it must not anymore.
    note = (
        "SESSION IN PROGRESS: 067290.KS's regular session is still open as of "
        "2026-09-15. Today has no open, no close, and no daily range yet."
    )
    text = "The stock closed at 3,990 on 2026-09-15"
    findings = scan_session_close_claims(text, session_note=note, agent_name="market_analyst")
    assert findings
    assert "closed_assertion" in findings[0]


def test_unparseable_session_note_falls_back_safely():
    # No parseable "as of YYYY-MM-DD" — must not crash, and must not become
    # noisier than the pre-fix behavior (any recognized date still exempts).
    note = "SESSION IN PROGRESS: regular session is still open right now."
    text = "The stock closed at 3,620 on Sept 10 and 3,515 on Sept 11"
    assert scan_session_close_claims(text, session_note=note, agent_name="market_analyst") == []


def test_does_not_flag_closing_above_a_moving_average():
    # 2026-09-10 audit, id=1376 (080220.KS): "Closing above the SMA(20)
    # confirms short-term momentum is positive" describes price LEVEL
    # relative to a moving-average benchmark, not an end-of-session event.
    text = (
        "*   **Bullish Signal:** Closing above the SMA(20) confirms "
        "short-term momentum is positive"
    )
    assert scan_session_close_claims(text, session_note=_SESSION_NOTE, agent_name="market_analyst") == []
