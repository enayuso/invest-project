"""app.dataflows.long_horizon — pure formatting/math, no network (A-stage)."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from app.dataflows.long_horizon import WeeklyBar, format_long_horizon_block


def _bars(n: int, base: float = 100.0, *, start: date = date(2025, 9, 1)) -> list[WeeklyBar]:
    return [
        WeeklyBar(start + timedelta(weeks=i), high=base + i + 1, low=base + i - 1, close=base + i)
        for i in range(n)
    ]


def test_raises_below_minimum_bars():
    with pytest.raises(ValueError, match="need at least 12"):
        format_long_horizon_block("AAPL", _bars(5), source="yfinance")


def test_reports_high_low_from_correct_weeks():
    bars = _bars(53)  # closes/highs/lows monotonically increasing -> last bar is the max, first is the min
    result = format_long_horizon_block("AAPL", bars, source="yfinance")
    assert f"52-week high: {bars[-1].high:,.0f}" in result
    assert f"week ending {bars[-1].week_end.isoformat()}" in result
    assert f"52-week low: {bars[0].low:,.0f}" in result
    assert f"week ending {bars[0].week_end.isoformat()}" in result


def test_pct_off_high_and_above_low_are_signed_correctly():
    # Flat series except the last bar dips below the rest -> high is earlier,
    # low is the latest close, so pct-off-high is negative and pct-above-low is 0.
    bars = [WeeklyBar(date(2025, 9, 1) + timedelta(weeks=i), high=110.0, low=90.0, close=100.0) for i in range(12)]
    bars[-1] = WeeklyBar(bars[-1].week_end, high=95.0, low=80.0, close=80.0)
    result = format_long_horizon_block("AAPL", bars, source="yfinance")
    assert "-27.3% from the 12-week high" in result  # 80/110 - 1
    assert "+0.0% above the 12-week low" in result  # 80/80 - 1
    assert "NOT a 52-week one" in result


def test_trailing_returns_present_for_full_52_weeks():
    bars = _bars(53)
    result = format_long_horizon_block("AAPL", bars, source="yfinance")
    line = next(ln for ln in result.splitlines() if "trailing returns" in ln)
    assert "1M " in line and "3M " in line and "6M " in line and "12M " in line
    assert "n/a" not in line


def test_trailing_returns_show_na_when_history_too_short_for_that_window():
    # 20 weekly bars: 1M/3M computable, 6M/12M are not (need 26/52 weeks back).
    bars = _bars(20)
    result = format_long_horizon_block("AAPL", bars, source="yfinance")
    line = next(ln for ln in result.splitlines() if "trailing returns" in ln)
    assert "6M n/a" in line
    assert "12M n/a" in line
    assert "1M n/a" not in line


def test_recent_weekly_closes_capped_at_13():
    bars = _bars(53)
    result = format_long_horizon_block("AAPL", bars, source="yfinance")
    closes_line = next(ln for ln in result.splitlines() if "weekly closes, oldest to newest" in ln)
    assert "(last 13):" in closes_line
    assert len(closes_line.split(":")[-1].split()) == 13


def test_source_label_is_reflected_in_header():
    bars = _bars(12)
    assert "source: KIS" in format_long_horizon_block("005930.KS", bars, source="KIS")
    assert "source: yfinance" in format_long_horizon_block("AAPL", bars, source="yfinance")


def test_usd_ticker_keeps_cents():
    """2026-09-16 fix: every price used to render with `:,.0f` regardless of
    market — fine for whole-won KRX prices, but a $3.20-$3.60 USD ticker
    collapsed to "3 .. 4" with no cents at all."""
    bars = [
        WeeklyBar(date(2025, 9, 1) + timedelta(weeks=i), high=3.60, low=3.20, close=3.47)
        for i in range(12)
    ]
    result = format_long_horizon_block("PLUG", bars, source="yfinance")
    assert "high: 3.60" in result and "low: 3.20" in result
    assert "latest weekly close: 3.47" in result
    assert "3.47 3.47" in result  # the recent-closes line, not "3 3"


def test_krx_ticker_renders_whole_won():
    bars = _bars(12, base=70000.0)
    result = format_long_horizon_block("005930.KS", bars, source="KIS")
    assert "70,011.00" not in result
    assert ".00" not in result


def test_short_window_labels_actual_span_not_52_weeks():
    result = format_long_horizon_block("AAPL", _bars(20), source="yfinance")
    assert "20-week high:" in result and "20-week low:" in result
    assert "52-week" in result  # only inside the note
    assert "52-week high:" not in result and "52-week low:" not in result
    assert "NOT a 52-week one" in result


def test_full_window_keeps_52_week_label_and_no_note():
    result = format_long_horizon_block("AAPL", _bars(52), source="yfinance")
    assert "52-week high:" in result
    assert "note: only" not in result


def test_output_never_contains_forbidden_verified_or_authoritative_wording():
    """`app.dataflows.session_state`'s `verified_price_guard` anchors on the
    literal word "verified"/"authoritative" — this block is deliberately
    plain data, not another "verified"-labeled source, so those words must
    never appear here (the caller's own prompt wraps it as authoritative,
    this function must not duplicate that claim inside the data itself)."""
    result = format_long_horizon_block("AAPL", _bars(53), source="yfinance").lower()
    assert "verified" not in result
    assert "authoritative" not in result
