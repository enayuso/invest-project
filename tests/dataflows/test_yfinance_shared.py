"""app.dataflows.yfinance.shared — pure date/retry helpers, no network (B-stage)."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from app.dataflows.yfinance.shared import (
    _filter_filing_lag,
    _filter_rows_by_date,
    _period_to_days,
    _retry_to_thread,
)


def test_period_to_days_variants():
    assert _period_to_days("6mo") == 180
    assert _period_to_days("1y") == 365
    assert _period_to_days("30d") == 30


def test_period_to_days_rejects_unknown_suffix():
    with pytest.raises(ValueError, match="unsupported period string"):
        _period_to_days("6w")


def test_filter_filing_lag_drops_periods_inside_lag_window():
    cols = pd.to_datetime(["2026-01-01", "2026-06-01"])
    df = pd.DataFrame([[1, 2]], columns=cols)
    # as_of 2026-07-16, lag=90d -> cutoff ~2026-04-17: only 2026-01-01 survives.
    result = _filter_filing_lag(df, date(2026, 7, 16))
    assert list(result.columns) == [cols[0]]


def test_filter_filing_lag_raises_when_nothing_survives():
    cols = pd.to_datetime(["2026-07-01"])
    df = pd.DataFrame([[1]], columns=cols)
    with pytest.raises(ValueError, match="no filed periods available"):
        _filter_filing_lag(df, date(2026, 7, 16))


def test_filter_rows_by_date_keeps_on_or_before_cutoff():
    df = pd.DataFrame({"val": [1, 2, 3]}, index=pd.to_datetime(["2026-01-01", "2026-06-01", "2026-08-01"]))
    result = _filter_rows_by_date(df, df.index, date(2026, 7, 1))
    assert list(result["val"]) == [1, 2]


def test_filter_rows_by_date_normalizes_tz_aware_dates():
    idx = pd.to_datetime(["2026-01-01", "2026-08-01"]).tz_localize("UTC")
    df = pd.DataFrame({"val": [1, 2]}, index=idx)
    result = _filter_rows_by_date(df, df.index, date(2026, 7, 1))
    assert list(result["val"]) == [1]


async def test_retry_to_thread_returns_on_first_success():
    calls = []

    def fn():
        calls.append(1)
        return "ok"

    result = await _retry_to_thread(fn, attempts=3, base_delay=0.01)
    assert result == "ok"
    assert len(calls) == 1


async def test_retry_to_thread_retries_then_succeeds():
    calls = []

    def fn():
        calls.append(1)
        if len(calls) < 2:
            raise ValueError("transient")
        return "ok"

    result = await _retry_to_thread(fn, attempts=3, base_delay=0.01)
    assert result == "ok"
    assert len(calls) == 2


async def test_retry_to_thread_raises_after_exhausting_attempts():
    def fn():
        raise ValueError("permanent")

    with pytest.raises(ValueError, match="permanent"):
        await _retry_to_thread(fn, attempts=2, base_delay=0.01)
