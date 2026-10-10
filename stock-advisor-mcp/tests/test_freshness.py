from datetime import date

import pandas as pd
import pytest

from stock_advisor_mcp import freshness


def _fake_bars(dates: list[str]) -> pd.DataFrame:
    idx = pd.to_datetime(dates)
    return pd.DataFrame({"Close": [100.0] * len(dates)}, index=idx)


def setup_function(_):
    freshness.reset_reference_cache_for_tests()


def test_get_latest_trading_day_reads_last_bar_date():
    def fetcher(symbol):
        assert symbol == freshness.REFERENCE_SYMBOL
        return _fake_bars(["2026-09-18", "2026-09-21"])

    day = freshness.get_latest_trading_day(fetcher)
    assert day == date(2026, 9, 21)


def test_get_latest_trading_day_is_cached_across_calls():
    calls = []

    def fetcher(symbol):
        calls.append(symbol)
        return _fake_bars(["2026-09-21"])

    freshness.get_latest_trading_day(fetcher)
    freshness.get_latest_trading_day(fetcher)
    assert len(calls) == 1  # second call must be served from cache, no re-fetch


def test_get_latest_trading_day_raises_on_empty_reference_data():
    def fetcher(symbol):
        return _fake_bars([])

    with pytest.raises(freshness.NoDataError):
        freshness.get_latest_trading_day(fetcher)


def test_assert_fresh_passes_when_dates_match():
    freshness.assert_fresh(date(2026, 9, 21), date(2026, 9, 21), symbol="005930.KS")


def test_assert_fresh_raises_when_stale():
    with pytest.raises(freshness.StaleDataError) as excinfo:
        freshness.assert_fresh(date(2026, 9, 18), date(2026, 9, 21), symbol="005930.KQ")
    assert "005930.KQ" in str(excinfo.value)


def test_assert_fresh_does_not_raise_for_future_bar():
    # Defensive: a bar "newer" than the reference should not be treated as stale.
    freshness.assert_fresh(date(2026, 9, 22), date(2026, 9, 21), symbol="005930.KS")
