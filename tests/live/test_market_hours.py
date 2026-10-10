"""is_market_open — pure wall-clock math against a FakeClock (A-stage)."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from app.clock import FakeClock, LiveClock, set_clock
from app.live import market_hours
from app.live.market_hours import (
    is_domestic_market_open,
    is_market_open,
    is_overseas_market_open,
    latest_domestic_signal_card_reset_at,
    latest_overseas_signal_card_reset_at,
)

_KR_TZ = ZoneInfo("Asia/Seoul")
_US_TZ = ZoneInfo("America/New_York")


@pytest.fixture(autouse=True)
def restore_clock():
    yield
    set_clock(LiveClock())


def test_kr_ticker_open_during_regular_session():
    # 2026-07-16 is a Thursday.
    set_clock(FakeClock(datetime(2026, 7, 16, 10, 0, tzinfo=_KR_TZ)))
    assert is_market_open("005930.KS") is True


def test_kr_ticker_closed_before_open():
    set_clock(FakeClock(datetime(2026, 7, 16, 8, 59, tzinfo=_KR_TZ)))
    assert is_market_open("005930.KS") is False


def test_kr_ticker_closed_after_close():
    set_clock(FakeClock(datetime(2026, 7, 16, 15, 31, tzinfo=_KR_TZ)))
    assert is_market_open("005930.KS") is False


def test_us_ticker_open_during_regular_session():
    set_clock(FakeClock(datetime(2026, 7, 16, 10, 0, tzinfo=_US_TZ)))
    assert is_market_open("AAPL") is True


def test_us_ticker_closed_outside_session():
    set_clock(FakeClock(datetime(2026, 7, 16, 20, 0, tzinfo=_US_TZ)))
    assert is_market_open("AAPL") is False


def test_weekend_closed_regardless_of_time_of_day():
    # 2026-07-18 is a Saturday.
    set_clock(FakeClock(datetime(2026, 7, 18, 10, 0, tzinfo=_US_TZ)))
    assert is_market_open("AAPL") is False
    assert is_market_open("005930.KS") is False


# ---------------------------------------------------------------------------
# Holiday-awareness (2026-08-08) — the real bug this closes: is_market_open
# used to answer "open" on every public holiday, weekday or not, because it
# only ever checked weekday + time-of-day. Confirmed live before the fix on
# 성탄절/한글날/삼일절 (KRX) and Thanksgiving/Good Friday (NYSE).
# ---------------------------------------------------------------------------


def test_kr_ticker_closed_on_christmas_2026_a_weekday():
    # 2026-12-25 is a Friday — a plain weekday+time check would say "open".
    set_clock(FakeClock(datetime(2026, 12, 25, 10, 0, tzinfo=_KR_TZ)))
    assert is_market_open("005930.KS") is False


def test_kr_ticker_closed_on_hangul_day_2026():
    # 2026-10-09 is a Friday.
    set_clock(FakeClock(datetime(2026, 10, 9, 10, 0, tzinfo=_KR_TZ)))
    assert is_market_open("005930.KS") is False


def test_kr_ticker_closed_on_liberation_day_substitute_2026():
    # 2026-08-15 (광복절) is a Saturday, already caught by the weekend check —
    # the substitute holiday is 2026-08-17 (Monday), which weekend-only logic
    # would have missed.
    set_clock(FakeClock(datetime(2026, 8, 17, 10, 0, tzinfo=_KR_TZ)))
    assert is_market_open("005930.KS") is False


def test_kr_ticker_still_open_on_an_ordinary_weekday():
    # Same session/time as the holiday tests above, one week later — the
    # holiday check must not accidentally block every day near a holiday.
    set_clock(FakeClock(datetime(2026, 10, 16, 10, 0, tzinfo=_KR_TZ)))
    assert is_market_open("005930.KS") is True


def test_us_ticker_closed_on_thanksgiving_2026():
    # 2026-11-26 is a Thursday.
    set_clock(FakeClock(datetime(2026, 11, 26, 10, 0, tzinfo=_US_TZ)))
    assert is_market_open("AAPL") is False


def test_us_ticker_closed_on_good_friday_2026():
    set_clock(FakeClock(datetime(2026, 4, 3, 10, 0, tzinfo=_US_TZ)))
    assert is_market_open("AAPL") is False


def test_us_ticker_still_open_on_an_ordinary_weekday():
    set_clock(FakeClock(datetime(2026, 11, 19, 10, 0, tzinfo=_US_TZ)))
    assert is_market_open("AAPL") is True


def test_kr_holiday_check_does_not_apply_to_us_tickers():
    # 2026-10-09 (한글날) is a KRX holiday only — NYSE trades normally.
    set_clock(FakeClock(datetime(2026, 10, 9, 10, 0, tzinfo=_US_TZ)))
    assert is_market_open("AAPL") is True


def test_us_holiday_check_does_not_apply_to_kr_tickers():
    # Thanksgiving is a US holiday only — KRX trades normally that day.
    set_clock(FakeClock(datetime(2026, 11, 26, 10, 0, tzinfo=_KR_TZ)))
    assert is_market_open("005930.KS") is True


def test_is_domestic_market_open_matches_kr_session():
    set_clock(FakeClock(datetime(2026, 7, 16, 10, 0, tzinfo=_KR_TZ)))
    assert is_domestic_market_open() is True
    set_clock(FakeClock(datetime(2026, 7, 16, 8, 59, tzinfo=_KR_TZ)))
    assert is_domestic_market_open() is False


def test_is_overseas_market_open_matches_us_session():
    set_clock(FakeClock(datetime(2026, 7, 16, 10, 0, tzinfo=_US_TZ)))
    assert is_overseas_market_open() is True
    set_clock(FakeClock(datetime(2026, 7, 16, 20, 0, tzinfo=_US_TZ)))
    assert is_overseas_market_open() is False


# ---------------------------------------------------------------------------
# latest_domestic_signal_card_reset_at — KRX-open-minus-30-minutes, KST
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _assume_every_weekday_is_a_trading_day(monkeypatch):
    # Every test in this section is about the weekday/boundary arithmetic
    # itself, not the holiday calendar — default every date to "open" (a
    # real network/DB call otherwise) so these behave exactly like the
    # pre-holiday-aware version unless a test explicitly overrides this.
    async def _always_open(day):
        return True

    monkeypatch.setattr(market_hours, "is_market_open_on", _always_open)


async def test_reset_boundary_is_todays_when_after_it():
    set_clock(FakeClock(datetime(2026, 7, 16, 14, 0, tzinfo=_KR_TZ)))  # afternoon
    reset_at = await latest_domestic_signal_card_reset_at()
    assert reset_at == datetime(2026, 7, 16, 8, 30, tzinfo=_KR_TZ)


async def test_reset_boundary_is_yesterdays_when_before_it_today():
    set_clock(FakeClock(datetime(2026, 7, 16, 7, 0, tzinfo=_KR_TZ)))  # before 08:30 today
    reset_at = await latest_domestic_signal_card_reset_at()
    assert reset_at == datetime(2026, 7, 15, 8, 30, tzinfo=_KR_TZ)


async def test_reset_boundary_exactly_at_boundary_counts_as_todays():
    set_clock(FakeClock(datetime(2026, 7, 16, 8, 30, tzinfo=_KR_TZ)))
    reset_at = await latest_domestic_signal_card_reset_at()
    assert reset_at == datetime(2026, 7, 16, 8, 30, tzinfo=_KR_TZ)


async def test_reset_boundary_on_monday_morning_rolls_back_to_friday_not_sunday():
    # 2026-07-20 is a Monday; before Monday's own boundary, the real prior
    # session is Friday 2026-07-17, not Sunday 2026-07-19.
    set_clock(FakeClock(datetime(2026, 7, 20, 7, 0, tzinfo=_KR_TZ)))
    reset_at = await latest_domestic_signal_card_reset_at()
    assert reset_at == datetime(2026, 7, 17, 8, 30, tzinfo=_KR_TZ)


async def test_reset_boundary_on_saturday_rolls_back_to_friday():
    # 2026-07-18 is a Saturday — no session that day, so even "today's own
    # boundary" (which now >= it) must still roll back to Friday.
    set_clock(FakeClock(datetime(2026, 7, 18, 10, 0, tzinfo=_KR_TZ)))
    reset_at = await latest_domestic_signal_card_reset_at()
    assert reset_at == datetime(2026, 7, 17, 8, 30, tzinfo=_KR_TZ)


async def test_reset_boundary_on_sunday_rolls_back_to_friday():
    set_clock(FakeClock(datetime(2026, 7, 19, 10, 0, tzinfo=_KR_TZ)))
    reset_at = await latest_domestic_signal_card_reset_at()
    assert reset_at == datetime(2026, 7, 17, 8, 30, tzinfo=_KR_TZ)


async def test_reset_boundary_rolls_back_past_a_public_holiday(monkeypatch):
    # 2026-07-16 (Thursday) is a made-up holiday here — the real prior
    # trading day is Wednesday 2026-07-15.
    async def _closed_on_2026_07_16(day):
        return day != date(2026, 7, 16)

    monkeypatch.setattr(market_hours, "is_market_open_on", _closed_on_2026_07_16)
    set_clock(FakeClock(datetime(2026, 7, 16, 20, 0, tzinfo=_KR_TZ)))
    reset_at = await latest_domestic_signal_card_reset_at()
    assert reset_at == datetime(2026, 7, 15, 8, 30, tzinfo=_KR_TZ)


async def test_reset_boundary_rolls_back_through_a_multi_day_holiday_stretch(monkeypatch):
    # A 설날-style 3-day closure (Mon-Wed) — the real prior trading day is
    # the preceding Friday, not just one day back.
    closed_dates = {date(2026, 7, 13), date(2026, 7, 14), date(2026, 7, 15)}

    async def _closed_on_stretch(day):
        return day not in closed_dates

    monkeypatch.setattr(market_hours, "is_market_open_on", _closed_on_stretch)
    set_clock(FakeClock(datetime(2026, 7, 15, 20, 0, tzinfo=_KR_TZ)))
    reset_at = await latest_domestic_signal_card_reset_at()
    assert reset_at == datetime(2026, 7, 10, 8, 30, tzinfo=_KR_TZ)


async def test_domestic_reset_boundary_ignores_a_us_only_holiday():
    # Split 2026-08-20: the domestic boundary no longer rejects a day for a
    # US-only holiday — that cross-market coupling was the direct cause of
    # the 2026-08-19 incident (overseas cards wiped by a KRX-based boundary
    # hours before the US market opened). 2026-11-26 (Thu, KST) is
    # Thanksgiving on the US side but an ordinary KRX trading day, so the
    # domestic boundary must land on its own date now, not roll back a day.
    set_clock(FakeClock(datetime(2026, 11, 27, 20, 0, tzinfo=_KR_TZ)))
    reset_at = await latest_domestic_signal_card_reset_at()
    assert reset_at == datetime(2026, 11, 27, 8, 30, tzinfo=_KR_TZ)


# ---------------------------------------------------------------------------
# latest_overseas_signal_card_reset_at — US-open-minus-30-minutes, ET
# (added 2026-08-20 — see latest_domestic_signal_card_reset_at's own
# docstring for the incident this split fixes)
# ---------------------------------------------------------------------------


async def test_overseas_reset_boundary_is_todays_when_after_it():
    set_clock(FakeClock(datetime(2026, 8, 20, 12, 0, tzinfo=_US_TZ)))
    reset_at = await latest_overseas_signal_card_reset_at()
    assert reset_at == datetime(2026, 8, 20, 9, 0, tzinfo=_US_TZ)


async def test_overseas_reset_boundary_is_yesterdays_when_before_it_today():
    set_clock(FakeClock(datetime(2026, 8, 20, 8, 0, tzinfo=_US_TZ)))
    reset_at = await latest_overseas_signal_card_reset_at()
    assert reset_at == datetime(2026, 8, 19, 9, 0, tzinfo=_US_TZ)


async def test_overseas_reset_boundary_exactly_at_boundary_counts_as_todays():
    set_clock(FakeClock(datetime(2026, 8, 20, 9, 0, tzinfo=_US_TZ)))
    reset_at = await latest_overseas_signal_card_reset_at()
    assert reset_at == datetime(2026, 8, 20, 9, 0, tzinfo=_US_TZ)


async def test_overseas_reset_boundary_on_monday_morning_rolls_back_to_friday_not_sunday():
    # 2026-08-24 is a Monday; before Monday's own boundary, the real prior
    # session is Friday 2026-08-21, not Sunday 2026-08-23.
    set_clock(FakeClock(datetime(2026, 8, 24, 8, 0, tzinfo=_US_TZ)))
    reset_at = await latest_overseas_signal_card_reset_at()
    assert reset_at == datetime(2026, 8, 21, 9, 0, tzinfo=_US_TZ)


async def test_overseas_reset_boundary_on_saturday_rolls_back_to_friday():
    set_clock(FakeClock(datetime(2026, 8, 22, 12, 0, tzinfo=_US_TZ)))
    reset_at = await latest_overseas_signal_card_reset_at()
    assert reset_at == datetime(2026, 8, 21, 9, 0, tzinfo=_US_TZ)


async def test_overseas_reset_boundary_rolls_back_past_thanksgiving():
    set_clock(FakeClock(datetime(2026, 11, 26, 12, 0, tzinfo=_US_TZ)))  # Thanksgiving
    reset_at = await latest_overseas_signal_card_reset_at()
    assert reset_at == datetime(2026, 11, 25, 9, 0, tzinfo=_US_TZ)


async def test_overseas_reset_boundary_rolls_back_past_good_friday():
    set_clock(FakeClock(datetime(2026, 4, 3, 12, 0, tzinfo=_US_TZ)))  # Good Friday
    reset_at = await latest_overseas_signal_card_reset_at()
    assert reset_at == datetime(2026, 4, 2, 9, 0, tzinfo=_US_TZ)


async def test_overseas_reset_boundary_ignores_a_kr_only_holiday_and_never_touches_kis(monkeypatch):
    # 한글날 (2026-10-09) is a KRX-only holiday — must not affect the
    # overseas boundary at all, and (unlike the domestic path) must never
    # even attempt the KIS-backed is_market_open_on call: the overseas
    # calendar is a pure static table with no I/O.
    async def _should_not_be_called(day):
        raise AssertionError("overseas reset boundary must never touch KIS/DB")

    monkeypatch.setattr(market_hours, "is_market_open_on", _should_not_be_called)
    set_clock(FakeClock(datetime(2026, 10, 9, 12, 0, tzinfo=_US_TZ)))
    reset_at = await latest_overseas_signal_card_reset_at()
    assert reset_at == datetime(2026, 10, 9, 9, 0, tzinfo=_US_TZ)


async def test_overseas_reset_boundary_est_offset():
    set_clock(FakeClock(datetime(2026, 1, 15, 12, 0, tzinfo=_US_TZ)))
    reset_at = await latest_overseas_signal_card_reset_at()
    assert reset_at == datetime(2026, 1, 15, 9, 0, tzinfo=_US_TZ)
    assert reset_at.utcoffset() == timedelta(hours=-5)


async def test_overseas_reset_boundary_across_a_dst_transition():
    # DST ends 2026-11-01; the Monday after (2026-11-02) is EST (-5h), while
    # the prior Friday session (2026-10-30) it rolls back to was EDT (-4h).
    # Wall-clock arithmetic (not UTC/epoch math) must still land on 09:00
    # local for the Friday, at the FRIDAY's own offset.
    set_clock(FakeClock(datetime(2026, 11, 2, 8, 0, tzinfo=_US_TZ)))
    reset_at = await latest_overseas_signal_card_reset_at()
    assert reset_at == datetime(2026, 10, 30, 9, 0, tzinfo=_US_TZ)
    assert reset_at.utcoffset() == timedelta(hours=-4)


async def test_overseas_reset_boundary_incident_regression():
    # 2026-08-19 incident: 5 real overseas decisions filed 22:46-22:58 KST
    # (well within the still-open US regular session, 09:00-16:00 ET) must
    # NOT be dropped by the overseas boundary the next KST morning, even
    # though the shared old boundary (08:30 KST) had already passed.
    set_clock(FakeClock(datetime(2026, 8, 20, 9, 0, tzinfo=_KR_TZ)))
    overseas_reset_at = await latest_overseas_signal_card_reset_at()
    domestic_reset_at = await latest_domestic_signal_card_reset_at()
    assert overseas_reset_at == datetime(2026, 8, 19, 22, 0, tzinfo=_KR_TZ)  # = 2026-08-19 09:00 EDT
    decision_created_at = datetime(2026, 8, 19, 22, 46, tzinfo=_KR_TZ)
    assert not (decision_created_at < overseas_reset_at)  # survives the overseas filter
    assert domestic_reset_at == datetime(2026, 8, 20, 8, 30, tzinfo=_KR_TZ)
