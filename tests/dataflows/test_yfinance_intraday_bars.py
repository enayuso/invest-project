"""app.dataflows.yfinance.market_data.get_intraday_bars — the 5-minute
intraday feed the trailing-stop backstop uses to resolve a day's real
high/low ORDERING (2026-08-27; see app.execution.ledger.
check_and_apply_trailing_stop's docstring for the 047040.KS incident that
made this necessary). yf.Ticker mocked via monkeypatch, no real network.

Every failure mode here must degrade to `[]`, never to an exception: the
caller's correct response to "no intraday data" is to fall back to its
coarser daily-bar rule, and a vendor blip must not abort a safety check.
"""

from __future__ import annotations

import time
from datetime import date, datetime, timedelta, timezone

import pandas as pd
import pytest
import yfinance as yf

from app.dataflows.yfinance import market_data


class _FakeTicker:
    def __init__(self, df: pd.DataFrame | None = None, raises: Exception | None = None):
        self._df = df if df is not None else pd.DataFrame()
        self._raises = raises

    def history(self, *args, **kwargs):
        if self._raises is not None:
            raise self._raises
        return self._df


@pytest.fixture(autouse=True)
def _clear_intraday_cache(monkeypatch):
    """The cache is module-global and keyed by (ticker, day) — without this
    one test's bars leak into the next and the call counters below become
    meaningless."""
    monkeypatch.setattr(market_data, "_intraday_cache", {})


@pytest.fixture
def _no_retry_sleep(monkeypatch):
    """_retry_to_thread backs off 0.5s then 1.0s before giving up; the
    failure tests care about the fail-open outcome, not the wall clock."""
    import app.dataflows.yfinance.shared as shared

    async def _instant(_seconds):
        return None

    monkeypatch.setattr(shared.asyncio, "sleep", _instant)


def _frame(times: list[str], highs: list[float], lows: list[float], tz="Asia/Seoul") -> pd.DataFrame:
    idx = pd.to_datetime(times)
    if tz is not None:
        idx = idx.tz_localize(tz)
    return pd.DataFrame({"High": highs, "Low": lows}, index=idx)


def _install(monkeypatch, ticker_obj, counter: list | None = None):
    def _factory(_symbol):
        if counter is not None:
            counter.append(1)
        return ticker_obj

    monkeypatch.setattr(yf, "Ticker", _factory)


async def test_get_intraday_bars_normalizes_to_utc_and_sets_bar_end(monkeypatch):
    # KST 09:00 / 09:30 / 10:50 — the 047040.KS session, deliberately fed
    # out of order to pin the sort.
    df = _frame(
        ["2026-08-27 09:30", "2026-08-27 09:00", "2026-08-27 10:50"],
        [21_950.0, 19_000.0, 21_100.0],
        [21_000.0, 18_760.0, 19_800.0],
    )
    _install(monkeypatch, _FakeTicker(df))

    bars = await market_data.get_intraday_bars("047040.KS", date(2026, 8, 27))

    assert len(bars) == 3
    assert [b.start for b in bars] == sorted(b.start for b in bars)  # oldest first
    for bar in bars:
        assert bar.start.tzinfo is not None
        assert bar.start.utcoffset() == timedelta(0)  # normalized to UTC
        assert bar.end - bar.start == timedelta(minutes=market_data._INTRADAY_BAR_MINUTES)
    # KST 09:00 == 00:00 UTC, and that bar carries the session's real low
    assert bars[0].start == datetime(2026, 8, 27, 0, 0, tzinfo=timezone.utc)
    assert bars[0].low == 18_760.0
    assert bars[1].high == 21_950.0  # the 09:30 high, thirty minutes LATER


async def test_get_intraday_bars_discards_a_naive_index(monkeypatch):
    """A naive index cannot be aligned to a position's entry timestamp at
    all, so the whole day is discarded rather than guessed at — the caller
    then falls back to its daily-bar rule."""
    df = _frame(["2026-08-27 09:00", "2026-08-27 09:05"], [100.0, 101.0], [99.0, 99.5], tz=None)
    _install(monkeypatch, _FakeTicker(df))

    assert await market_data.get_intraday_bars("047040.KS", date(2026, 8, 27)) == []


async def test_get_intraday_bars_returns_empty_on_a_vendor_failure(monkeypatch, _no_retry_sleep):
    _install(monkeypatch, _FakeTicker(raises=RuntimeError("429 Too Many Requests")))

    # Must not propagate — see this module's own docstring.
    assert await market_data.get_intraday_bars("047040.KS", date(2026, 8, 27)) == []


async def test_get_intraday_bars_returns_empty_on_an_empty_download(monkeypatch):
    _install(monkeypatch, _FakeTicker(pd.DataFrame()))

    assert await market_data.get_intraday_bars("047040.KS", date(2026, 8, 27)) == []


async def test_get_intraday_bars_skips_nan_rows(monkeypatch):
    df = _frame(
        ["2026-08-27 09:00", "2026-08-27 09:05", "2026-08-27 09:10"],
        [100.0, float("nan"), 102.0],
        [99.0, 98.0, float("nan")],
    )
    _install(monkeypatch, _FakeTicker(df))

    bars = await market_data.get_intraday_bars("047040.KS", date(2026, 8, 27))
    assert len(bars) == 1
    assert (bars[0].high, bars[0].low) == (100.0, 99.0)


async def test_get_intraday_bars_caches_a_closed_session_and_short_caches_a_live_one(monkeypatch):
    """A closed session's bars are immutable, so re-downloading them on every
    hourly backstop pass is pure yfinance rate-limit pressure. Today's
    still-running session gets the short TTL instead, so a later pass
    genuinely re-reads it."""
    # --- a long-closed session: the long TTL ---
    closed_day = date(2026, 8, 27)
    calls: list = []
    _install(monkeypatch, _FakeTicker(_frame(["2026-08-27 09:00"], [100.0], [99.0])), calls)

    first = await market_data.get_intraday_bars("047040.KS", closed_day)
    assert len(first) == 1 and len(calls) == 1
    expiry = market_data._intraday_cache[("047040.KS", closed_day)][0]
    assert expiry - time.monotonic() > market_data._INTRADAY_CACHE_SHORT_TTL_SECONDS

    second = await market_data.get_intraday_bars("047040.KS", closed_day)
    assert second == first
    assert len(calls) == 1  # served from cache, no second download

    # --- a session whose last bar is still fresh: the short TTL ---
    now = datetime.now(timezone.utc)
    live_day = now.date()
    live_calls: list = []
    live_df = pd.DataFrame(
        {"High": [100.0], "Low": [99.0]},
        index=pd.DatetimeIndex([now.replace(tzinfo=timezone.utc)]),
    )
    _install(monkeypatch, _FakeTicker(live_df), live_calls)

    live = await market_data.get_intraday_bars("005930.KS", live_day)
    assert len(live) == 1 and len(live_calls) == 1
    live_expiry = market_data._intraday_cache[("005930.KS", live_day)][0]
    assert live_expiry - time.monotonic() <= market_data._INTRADAY_CACHE_SHORT_TTL_SECONDS


async def test_get_intraday_bars_evicts_when_the_cache_is_full(monkeypatch):
    """The bound exists so an unattended daemon walking many (ticker, day)
    pairs cannot grow this dict without limit."""
    monkeypatch.setattr(market_data, "_INTRADAY_CACHE_MAX_ENTRIES", 2)
    _install(monkeypatch, _FakeTicker(_frame(["2026-08-27 09:00"], [100.0], [99.0])))

    for offset in range(4):
        await market_data.get_intraday_bars(f"T{offset}.KS", date(2026, 8, 27))

    assert len(market_data._intraday_cache) <= 2
