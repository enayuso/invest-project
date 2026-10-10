"""is_us_market_holiday — static NYSE rule table, no network/DB dependency."""

from __future__ import annotations

from datetime import date

from app.live.us_holidays import is_us_market_holiday


def test_fixed_date_holiday_christmas():
    assert is_us_market_holiday(date(2026, 12, 25)) is True


def test_nth_weekday_holiday_thanksgiving():
    # 4th Thursday of November 2026.
    assert is_us_market_holiday(date(2026, 11, 26)) is True
    assert is_us_market_holiday(date(2026, 11, 25)) is False


def test_good_friday_derived_from_easter():
    # Easter 2026 is 2026-04-05; Good Friday is two days before.
    assert is_us_market_holiday(date(2026, 4, 3)) is True


def test_juneteenth():
    assert is_us_market_holiday(date(2026, 6, 19)) is True


def test_fixed_holiday_observed_on_preceding_friday_when_it_falls_on_saturday():
    # Independence Day 2026-07-04 is a Saturday; NYSE observes it Friday 07-03.
    assert is_us_market_holiday(date(2026, 7, 3)) is True
    assert is_us_market_holiday(date(2026, 7, 4)) is False


def test_new_years_day_observed_on_preceding_friday_when_it_falls_on_saturday():
    # 2028-01-01 is a Saturday; NYSE observes it Friday 2027-12-31 — the
    # cross-year lookup this exercises is the one part of this module that
    # isn't a one-line rule.
    assert is_us_market_holiday(date(2027, 12, 31)) is True
    assert is_us_market_holiday(date(2028, 1, 1)) is False


def test_ordinary_weekday_is_not_a_holiday():
    assert is_us_market_holiday(date(2026, 7, 16)) is False
