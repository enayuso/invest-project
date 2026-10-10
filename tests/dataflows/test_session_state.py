"""app.dataflows.session_state — session_in_progress/format_verified_snapshot/
live_session_footnote, pure wall-clock math against a FakeClock (A-stage),
same convention as tests/live/test_market_hours.py."""

from __future__ import annotations

from datetime import date, datetime
from zoneinfo import ZoneInfo

import pytest

from app.clock import FakeClock, LiveClock, set_clock
from app.dataflows.session_state import (
    format_verified_snapshot,
    intraday_narration_note,
    live_session_footnote,
    session_in_progress,
)

_KR_TZ = ZoneInfo("Asia/Seoul")
_US_TZ = ZoneInfo("America/New_York")


@pytest.fixture(autouse=True)
def restore_clock():
    yield
    set_clock(LiveClock())


# --- session_in_progress ---------------------------------------------------


def test_us_ticker_in_progress_during_regular_session():
    set_clock(FakeClock(datetime(2026, 8, 26, 9, 42, tzinfo=_US_TZ)))
    assert session_in_progress("MRVL", date(2026, 8, 26)) is True


def test_us_ticker_not_in_progress_after_close():
    set_clock(FakeClock(datetime(2026, 8, 26, 16, 30, tzinfo=_US_TZ)))
    assert session_in_progress("MRVL", date(2026, 8, 26)) is False


def test_us_ticker_in_progress_when_checked_from_kst_past_midnight():
    # 2026-08-27 01:00 KST == 2026-08-26 12:00 ET — the US regular session
    # is still running, but a naive get_clock().today() (KST) would already
    # read "2026-08-27", off by one from the still-open US session's own
    # calendar date. This is exactly why session_in_progress must use
    # local_today("overseas"), not a single shared KST date — see that
    # function's own docstring.
    set_clock(FakeClock(datetime(2026, 8, 27, 1, 0, tzinfo=_KR_TZ)))
    assert session_in_progress("MRVL", date(2026, 8, 26)) is True


def test_kr_ticker_in_progress_during_regular_session():
    set_clock(FakeClock(datetime(2026, 8, 26, 10, 0, tzinfo=_KR_TZ)))
    assert session_in_progress("005930.KS", date(2026, 8, 26)) is True


def test_kr_ticker_not_in_progress_after_close():
    set_clock(FakeClock(datetime(2026, 8, 26, 16, 0, tzinfo=_KR_TZ)))
    assert session_in_progress("005930.KS", date(2026, 8, 26)) is False


def test_past_row_never_in_progress_even_during_market_hours():
    # Proves the backtest path (as_of always a past date) is unaffected
    # without needing its own branch: a historical row's date can never
    # equal today's, market open or not.
    set_clock(FakeClock(datetime(2026, 8, 26, 9, 42, tzinfo=_US_TZ)))
    assert session_in_progress("MRVL", date(2026, 7, 10)) is False


def test_us_ticker_in_progress_when_caller_passes_kst_calendar_date():
    # The FIX 1 case (id=1157 shape): app.chat.router / chat_decisions pass
    # get_clock().today() — a KST date — as trade_date for BOTH markets, not
    # local_today(market). At ET 12:00 the US session is open but KST has
    # already rolled to the next calendar day (ET+13h crosses KST midnight
    # for ET >= 11:00), so a KST-dated row_date is one day AHEAD of
    # local_today("overseas"). The old strict `==` test failed here and
    # silently produced no in-progress note; `>=` must still catch it.
    set_clock(FakeClock(datetime(2026, 8, 26, 12, 0, tzinfo=_US_TZ)))
    assert session_in_progress("MRVL", date(2026, 8, 27)) is True


def test_not_in_progress_on_a_weekend():
    # 2026-08-29 is a Saturday.
    set_clock(FakeClock(datetime(2026, 8, 29, 10, 0, tzinfo=_US_TZ)))
    assert session_in_progress("MRVL", date(2026, 8, 29)) is False


# --- format_verified_snapshot -----------------------------------------------


def test_format_verified_snapshot_closed_session_matches_legacy_format():
    # Lock-in: the closed-session FIRST line must stay byte-identical to the
    # format every existing test/parser/script already depends on; the
    # price-position line is a new SECOND line that must not disturb it.
    set_clock(FakeClock(datetime(2026, 8, 26, 16, 30, tzinfo=_US_TZ)))
    result = format_verified_snapshot(
        "AAPL", date(2026, 7, 10), close=100.0, high=105.0, low=95.0, volume=1000, source="KIS"
    )
    lines = result.split("\n")
    assert lines[0] == (
        "VERIFIED SNAPSHOT for AAPL as of 2026-07-10: "
        "close=100.00 high=105.00 low=95.00 volume=1000"
    )
    assert lines[1] == (
        "Position in the day's range: close is 5.3% above the low and "
        "4.8% below the high — the MIDDLE third of the day's range."
    )
    # 2026-09-10 dataflows audit stage 2: source is now always a final line.
    assert lines[2] == "source: KIS"


def test_format_verified_snapshot_price_position_in_progress_bottom_third():
    set_clock(FakeClock(datetime(2026, 8, 26, 9, 42, tzinfo=_US_TZ)))
    result = format_verified_snapshot(
        "MRVL", date(2026, 8, 26), close=2390.0, high=2795.0, low=2210.0, volume=100, source="KIS"
    )
    assert "the BOTTOM third of the range traded so far today" in result
    assert "the TOP third of the range traded" not in result
    # Anchored on the preceding word, not a bare "8.1%"/"14.5%" substring: the
    # pre-2026-09-11 output rendered these as "+8.1%"/"-14.5%", and a bare
    # substring match would still pass against that double-negative wording,
    # which is the exact regression this assertion exists to catch.
    assert "last is 8.1% above session_low" in result
    assert "and 14.5% below session_high" in result
    assert "-14.5%" not in result
    # prose-only: the chat-router / seed-script regexes key on `close=`/`last=`
    assert "close=" not in result.split("\n", 1)[1]


def test_format_verified_snapshot_price_position_omitted_when_high_equals_low():
    set_clock(FakeClock(datetime(2026, 8, 26, 16, 30, tzinfo=_US_TZ)))
    result = format_verified_snapshot(
        "AAPL", date(2026, 7, 10), close=100.0, high=100.0, low=100.0, volume=1000, source="KIS"
    )
    assert result == (
        "VERIFIED SNAPSHOT for AAPL as of 2026-07-10: "
        "close=100.00 high=100.00 low=100.00 volume=1000\n"
        "source: KIS"
    )


def test_format_verified_snapshot_in_progress_session_uses_last_labels():
    set_clock(FakeClock(datetime(2026, 8, 26, 9, 42, tzinfo=_US_TZ)))
    result = format_verified_snapshot(
        "MRVL", date(2026, 8, 26), close=237.04, high=240.77, low=236.79, volume=769562, source="KIS"
    )
    assert "SESSION IN PROGRESS" in result
    assert "last=237.04" in result
    assert "session_high=240.77" in result
    assert "session_low=236.79" in result
    assert "volume_so_far=769562" in result
    # The closed-session field names must never appear alongside the
    # in-progress ones — that's exactly the ambiguity a model could
    # pattern-match past.
    assert "close=" not in result
    assert " high=" not in result
    assert " low=" not in result
    assert " volume=" not in result


# --- format_verified_snapshot: source marker (2026-09-10 dataflows audit,
# stage 2/3 — neither vendor's output said which one produced it) ----------


def test_format_verified_snapshot_closed_session_yfinance_source_no_krx_caveat():
    set_clock(FakeClock(datetime(2026, 8, 26, 16, 30, tzinfo=_US_TZ)))
    result = format_verified_snapshot(
        "AAPL", date(2026, 7, 10), close=100.0, high=105.0, low=95.0, volume=1000, source="yfinance"
    )
    assert result.split("\n")[-1] == "source: yfinance"
    assert "NOT the exchange-official" not in result


def test_format_verified_snapshot_krx_ticker_yfinance_source_has_caveat():
    set_clock(FakeClock(datetime(2026, 8, 26, 16, 30, tzinfo=_US_TZ)))
    result = format_verified_snapshot(
        "005930.KS", date(2026, 7, 10), close=100.0, high=105.0, low=95.0, volume=1000,
        source="yfinance",
    )
    last_line = result.split("\n")[-1]
    assert last_line.startswith("source: yfinance")
    assert "NOT the exchange-official KIS feed" in last_line
    assert "~2%" in last_line


def test_format_verified_snapshot_krx_ticker_kis_source_no_caveat():
    set_clock(FakeClock(datetime(2026, 8, 26, 16, 30, tzinfo=_US_TZ)))
    result = format_verified_snapshot(
        "005930.KS", date(2026, 7, 10), close=100.0, high=105.0, low=95.0, volume=1000, source="KIS"
    )
    assert result.split("\n")[-1] == "source: KIS"


def test_format_verified_snapshot_in_progress_session_source_line_appended():
    set_clock(FakeClock(datetime(2026, 8, 26, 9, 42, tzinfo=_US_TZ)))
    result = format_verified_snapshot(
        "MRVL", date(2026, 8, 26), close=237.04, high=240.77, low=236.79, volume=769562,
        source="KIS",
    )
    assert result.split("\n")[-1] == "source: KIS"
    # existing property must still hold with the new line appended
    assert "close=" not in result.split("\n", 1)[1]


def test_format_verified_snapshot_in_progress_krx_yfinance_caveat_present():
    set_clock(FakeClock(datetime(2026, 8, 26, 10, 0, tzinfo=_KR_TZ)))
    result = format_verified_snapshot(
        "005930.KS", date(2026, 8, 26), close=100.0, high=105.0, low=95.0, volume=1000,
        source="yfinance",
    )
    assert "SESSION IN PROGRESS" in result
    assert "NOT the exchange-official KIS feed" in result.split("\n")[-1]


# --- round-trip: downstream parsers must still extract the same values
# after the trailing source line is appended (the constraint that motivated
# "append, don't modify" in the first place) --------------------------------


def test_snapshot_parsers_unaffected_by_trailing_source_line():
    from app.agents.verified_price_guard import _GROUND_TRUTH_RE
    from app.chat.router import _SNAPSHOT_RE

    set_clock(FakeClock(datetime(2026, 8, 26, 16, 30, tzinfo=_US_TZ)))
    closed = format_verified_snapshot(
        "AAPL", date(2026, 7, 10), close=100.0, high=105.0, low=95.0, volume=1000, source="KIS"
    )
    m = _SNAPSHOT_RE.search(closed)
    assert m and m["close"] == "100.00" and m["volume"] == "1000"
    gt = _GROUND_TRUTH_RE.search(closed)
    assert gt and gt.group(1) == "100.00"

    set_clock(FakeClock(datetime(2026, 8, 26, 9, 42, tzinfo=_US_TZ)))
    live = format_verified_snapshot(
        "MRVL", date(2026, 8, 26), close=237.04, high=240.77, low=236.79, volume=769562,
        source="KIS",
    )
    m2 = _SNAPSHOT_RE.search(live)
    assert m2 and m2["close"] == "237.04" and m2["live"]
    gt2 = _GROUND_TRUTH_RE.search(live)
    assert gt2 and gt2.group(1) == "237.04"

    set_clock(FakeClock(datetime(2026, 8, 26, 16, 30, tzinfo=_US_TZ)))
    krx_caveat = format_verified_snapshot(
        "005930.KS", date(2026, 7, 10), close=71000.0, high=72000.0, low=70000.0, volume=500,
        source="yfinance",
    )
    m3 = _SNAPSHOT_RE.search(krx_caveat)
    assert m3 and m3["close"] == "71000.00"
    gt3 = _GROUND_TRUTH_RE.search(krx_caveat)
    assert gt3 and gt3.group(1) == "71000.00"


# --- live_session_footnote ---------------------------------------------------


def test_live_session_footnote_empty_when_session_closed():
    set_clock(FakeClock(datetime(2026, 8, 26, 16, 30, tzinfo=_US_TZ)))
    assert live_session_footnote("MRVL", date(2026, 8, 26)) == ""


def test_live_session_footnote_present_when_session_open():
    set_clock(FakeClock(datetime(2026, 8, 26, 9, 42, tzinfo=_US_TZ)))
    note = live_session_footnote("MRVL", date(2026, 8, 26))
    assert "STILL-OPEN" in note
    assert "2026-08-26" in note


# --- intraday_narration_note (2026-09-01, id=1157) -------------------------


def test_intraday_narration_note_empty_after_close():
    set_clock(FakeClock(datetime(2026, 8, 26, 16, 30, tzinfo=_US_TZ)))
    assert intraday_narration_note("MRVL", date(2026, 8, 26)) == ""


def test_intraday_narration_note_empty_for_a_past_row_during_market_hours():
    set_clock(FakeClock(datetime(2026, 8, 26, 10, 0, tzinfo=_KR_TZ)))
    assert intraday_narration_note("005930.KS", date(2026, 7, 10)) == ""


def test_intraday_narration_note_present_during_an_open_kr_session():
    set_clock(FakeClock(datetime(2026, 9, 1, 10, 30, tzinfo=_KR_TZ)))
    note = intraday_narration_note("005930.KS", date(2026, 9, 1))
    assert "SESSION IN PROGRESS" in note
    assert "2026-09-01" in note
    for forbidden in ("no open, no close", "NOT the open", "concluded trading", "so far today"):
        assert forbidden in note
    # the correction clause — the PM only ever sees the transcript, not the raw data
    assert "any earlier report or transcript quoted below" in note


# --- candle-pattern clause (2026-09-14, 141-violation audit) ---------------


def test_intraday_narration_note_forbids_candlestick_pattern_vocabulary():
    set_clock(FakeClock(datetime(2026, 9, 1, 10, 30, tzinfo=_KR_TZ)))
    note = intraday_narration_note("005930.KS", date(2026, 9, 1))
    for forbidden in (
        "shooting star",
        "doji",
        "hanging man",
        "marubozu",
        "engulfing",
        "wick",
        "shadow",
        "distribution pattern",
        "accumulation pattern",
    ):
        assert forbidden in note


def test_intraday_narration_note_still_empty_after_close_with_candle_clause_added():
    # Pins the empty-string invariant that keeps closed-session/backtest
    # prompts byte-identical — must still hold with the new clause appended.
    set_clock(FakeClock(datetime(2026, 8, 26, 16, 30, tzinfo=_US_TZ)))
    assert intraday_narration_note("MRVL", date(2026, 8, 26)) == ""
