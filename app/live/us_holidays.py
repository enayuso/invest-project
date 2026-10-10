"""Static NYSE holiday calendar — no KIS API call or DB cache, unlike
app.live.kr_holidays. Unlike 설날/추석 (lunar, requires an authoritative
source), US market holidays are almost entirely fixed by calendar rule
(nth-weekday-of-month, or a fixed date with a Saturday/Sunday
observed-shift), so a small rule table is simpler and more reliable than
KIS's 해외결제일자조회 TR (CTOS5011R) — that TR returns settlement dates per
country/market, not an explicit open/closed flag, and its exact semantics
for "was this date itself a holiday" were never confirmed live.

Does not model NYSE early-close days (e.g. the day after Thanksgiving) —
those are still full trading days, just shorter, and irrelevant to a
per-date open/closed check.
"""

from __future__ import annotations

from datetime import date, timedelta


def _nth_weekday_of_month(year: int, month: int, weekday: int, n: int) -> date:
    """weekday: Monday=0 ... Sunday=6. n=1 for the first such weekday in the
    month, n=2 for the second, etc."""
    d = date(year, month, 1)
    d += timedelta(days=(weekday - d.weekday()) % 7)
    return d + timedelta(weeks=n - 1)


def _last_weekday_of_month(year: int, month: int, weekday: int) -> date:
    next_month = date(year + 1, 1, 1) if month == 12 else date(year, month + 1, 1)
    d = next_month - timedelta(days=1)
    return d - timedelta(days=(d.weekday() - weekday) % 7)


def _easter_sunday(year: int) -> date:
    """Anonymous Gregorian algorithm (Meeus/Jones/Butcher) — needed for Good
    Friday, the one NYSE holiday that isn't a fixed date or nth-weekday."""
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = (h + l - 7 * m + 114) % 31 + 1
    return date(year, month, day)


def _observed(day: date) -> date:
    """NYSE shifts a holiday landing on Saturday to the preceding Friday,
    and on Sunday to the following Monday."""
    if day.weekday() == 5:
        return day - timedelta(days=1)
    if day.weekday() == 6:
        return day + timedelta(days=1)
    return day


def _nyse_holidays(year: int) -> set[date]:
    good_friday = _easter_sunday(year) - timedelta(days=2)
    return {
        _observed(date(year, 1, 1)),  # New Year's Day
        _nth_weekday_of_month(year, 1, 0, 3),  # MLK Day
        _nth_weekday_of_month(year, 2, 0, 3),  # Presidents Day
        good_friday,
        _last_weekday_of_month(year, 5, 0),  # Memorial Day
        _observed(date(year, 6, 19)),  # Juneteenth (NYSE holiday since 2022)
        _observed(date(year, 7, 4)),  # Independence Day
        _nth_weekday_of_month(year, 9, 0, 1),  # Labor Day
        _nth_weekday_of_month(year, 11, 3, 4),  # Thanksgiving
        _observed(date(year, 12, 25)),  # Christmas Day
    }


def is_us_market_holiday(day: date) -> bool:
    """Whether `day` is a NYSE holiday. Weekends are not included here —
    same convention as app.live.kr_holidays.is_market_open_on, whose
    weekend check lives in the caller."""
    holidays = _nyse_holidays(day.year)
    if day.month == 12:
        # New Year's Day observed-Friday (when Jan 1 falls on a Saturday)
        # lands on Dec 31 of the prior year — check next year's table too.
        holidays |= _nyse_holidays(day.year + 1)
    return day in holidays
