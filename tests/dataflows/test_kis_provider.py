"""KIS-first / yfinance-fallback dataflow provider (B-stage). FakeBroker
stands in for app.brokers.KISBroker; no network, no real yfinance calls."""

from __future__ import annotations

import asyncio
import logging
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest
import talib

from app.clock import FakeClock, LiveClock, set_clock
from app.dataflows import kis_provider
from app.dataflows.indicators import atr_pct as _compute_atr_pct
from app.domain.models import Candle, NewsTitle, StockStatus
from app.domain.models.market import (
    BalanceSheet,
    FinancialRatio,
    IncomeStatement,
    ProgramTrade,
    ShortSaleDay,
    StockValuation,
)


_US_TZ = ZoneInfo("America/New_York")


@pytest.fixture(autouse=True)
def restore_clock():
    yield
    set_clock(LiveClock())


def _returns(broker):
    """Stand-in for kis_provider._get_broker, which became a coroutine
    function in the multi-user migration (Phase 2) — credentials come from
    Postgres now, so resolving a broker is I/O rather than an env read."""
    async def _get_broker():
        return broker

    return _get_broker


class FakeBroker:
    def __init__(
        self, candles: list[Candle] | None = None, raise_exc: Exception | None = None,
        overseas_candles: list[Candle] | None = None, news: list | None = None,
        program_trade: list[ProgramTrade] | None = None,
        daily_short_sale: list | None = None,
        income_statement: list[IncomeStatement] | None = None,
        balance_sheet: list[BalanceSheet] | None = None,
        financial_ratio: list[FinancialRatio] | None = None,
        stability_ratio: list | None = None,
        profitability_ratio: list | None = None,
        growth_ratio_detail: list | None = None,
        other_major_ratios: list | None = None,
        valuation: StockValuation | None = None,
        invest_opinion: list | None = None,
        weekly_candles: list[Candle] | None = None,
        weekly_overseas_candles: list[Candle] | None = None,
    ):
        self._candles = candles or []
        self._raise = raise_exc
        self._overseas_candles = overseas_candles if overseas_candles is not None else self._candles
        self._weekly_candles = weekly_candles if weekly_candles is not None else self._candles
        self._weekly_overseas_candles = (
            weekly_overseas_candles if weekly_overseas_candles is not None else self._overseas_candles
        )
        self._news = news or []
        self._program_trade = program_trade or []
        self._daily_short_sale = daily_short_sale or []
        self._income_statement = income_statement or []
        self._balance_sheet = balance_sheet or []
        self._financial_ratio = financial_ratio or []
        self._stability_ratio = stability_ratio or []
        self._profitability_ratio = profitability_ratio or []
        self._growth_ratio_detail = growth_ratio_detail or []
        self._other_major_ratios = other_major_ratios or []
        self._valuation = valuation
        self._invest_opinion = invest_opinion or []

    async def fetch_daily_candles(self, code, start, end, period="D"):
        if self._raise is not None:
            raise self._raise
        return self._weekly_candles if period == "W" else self._candles

    async def fetch_overseas_candles(self, symbol, end="", exchange="NAS", period="D"):
        if self._raise is not None:
            raise self._raise
        return self._weekly_overseas_candles if period == "W" else self._overseas_candles

    async def fetch_overseas_news_titles(self, symbol, exchange="", nation=""):
        if self._raise is not None:
            raise self._raise
        return self._news

    async def fetch_daily_short_sale(self, code, start, end):
        return self._daily_short_sale

    async def fetch_program_trade(self, code):
        return self._program_trade

    async def fetch_kr_income_statement(self, code, div_cls="0"):
        return self._income_statement

    async def fetch_kr_balance_sheet(self, code, div_cls="0"):
        return self._balance_sheet

    async def fetch_kr_financial_ratio(self, code, div_cls="0"):
        return self._financial_ratio

    async def fetch_kr_stability_ratio(self, code, div_cls="0"):
        return self._stability_ratio

    async def fetch_kr_profitability_ratio(self, code, div_cls="0"):
        return self._profitability_ratio

    async def fetch_kr_growth_ratio_detail(self, code, div_cls="0"):
        return self._growth_ratio_detail

    async def fetch_kr_other_major_ratios(self, code, div_cls="0"):
        return self._other_major_ratios

    async def fetch_valuation(self, code):
        if self._valuation is None:
            raise RuntimeError("no valuation configured")
        return self._valuation

    async def fetch_invest_opinion(self, code, start, end):
        return self._invest_opinion


def _candle(day: str, close: float, high: float | None = None, low: float | None = None) -> Candle:
    return Candle(day=day, open=close, high=high or close, low=low or close, close=close, volume=1000)


def test_is_krx_true_for_ks_suffixed_numeric_ticker():
    assert kis_provider._is_krx("005930.KS") is True


def test_is_krx_false_for_non_ks_ticker():
    assert kis_provider._is_krx("AAPL") is False


def test_is_krx_false_for_ks_suffix_without_numeric_code():
    assert kis_provider._is_krx("ABC.KS") is False


def test_is_us_true_for_plain_uppercase_ticker():
    assert kis_provider._is_us("AAPL") is True
    assert kis_provider._is_us("F") is True  # single-letter tickers exist (Ford)


def test_is_us_false_for_krx_ticker():
    assert kis_provider._is_us("005930.KS") is False


def test_is_us_false_for_dotted_or_longer_symbol():
    assert kis_provider._is_us("0700.HK") is False
    assert kis_provider._is_us("TOOLONG") is False


def test_krx_code_strips_suffix():
    assert kis_provider._krx_code("005930.KS") == "005930"


def test_krx_code_raises_for_non_krx_ticker():
    with pytest.raises(ValueError, match="is not a KRX ticker"):
        kis_provider._krx_code("AAPL")


def test_candle_date_parses_yyyymmdd():
    assert kis_provider._candle_date("20260716") == date(2026, 7, 16)


async def test_get_last_close_uses_kis_for_krx_ticker(monkeypatch):
    fake = FakeBroker(candles=[_candle("20260715", 100.0), _candle("20260716", 105.0)])
    monkeypatch.setattr(kis_provider, "_get_broker", _returns(fake))

    async def _yf_should_not_be_called(*a, **k):
        raise AssertionError("yfinance fallback should not be called when KIS succeeds")

    monkeypatch.setattr(kis_provider, "_yf_last_close", _yf_should_not_be_called)
    result = await kis_provider.get_last_close("005930.KS")
    assert result == 105.0


async def test_get_last_close_falls_back_to_yfinance_on_kis_failure(monkeypatch):
    async def _broken_broker():
        raise RuntimeError("KIS not configured")

    monkeypatch.setattr(kis_provider, "_get_broker", _broken_broker)

    async def fake_yf_last_close(ticker, as_of=None):
        return 999.0

    monkeypatch.setattr(kis_provider, "_yf_last_close", fake_yf_last_close)
    result = await kis_provider.get_last_close("005930.KS")
    assert result == 999.0


async def test_get_last_close_unsupported_ticker_never_calls_kis(monkeypatch):
    async def _should_not_be_called():
        raise AssertionError("KIS broker should not be looked up for a ticker matching neither market")

    monkeypatch.setattr(kis_provider, "_get_broker", _should_not_be_called)

    async def fake_yf_last_close(ticker, as_of=None):
        return 42.0

    monkeypatch.setattr(kis_provider, "_yf_last_close", fake_yf_last_close)
    result = await kis_provider.get_last_close("0700.HK")
    assert result == 42.0


async def test_get_last_close_uses_kis_for_us_ticker(monkeypatch):
    fake = FakeBroker(overseas_candles=[_candle("20260722", 200.0), _candle("20260723", 205.0)])
    monkeypatch.setattr(kis_provider, "_get_broker", _returns(fake))

    async def _yf_should_not_be_called(*a, **k):
        raise AssertionError("yfinance fallback should not be called when KIS succeeds")

    monkeypatch.setattr(kis_provider, "_yf_last_close", _yf_should_not_be_called)
    result = await kis_provider.get_last_close("AAPL", as_of=date(2026, 7, 23))
    assert result == 205.0


async def test_get_atr_pct_falls_back_when_too_few_candles(monkeypatch):
    # Fewer than window(14)+1 rows -> falls back to yfinance rather than
    # computing a rolling ATR off a too-short KIS candle series.
    fake = FakeBroker(candles=[_candle("2026071%d" % i, 100.0 + i) for i in range(5)])
    monkeypatch.setattr(kis_provider, "_get_broker", _returns(fake))

    async def fake_yf_atr(ticker, as_of=None):
        return 0.02

    monkeypatch.setattr(kis_provider, "_yf_atr_pct", fake_yf_atr)
    result = await kis_provider.get_atr_pct("005930.KS")
    assert result == 0.02


async def test_get_atr_pct_uses_kis_for_us_ticker(monkeypatch):
    as_of = date(2026, 7, 23)
    candles = [_candle((as_of - timedelta(days=i)).strftime("%Y%m%d"), 100.0 + i) for i in range(20, 0, -1)]
    fake = FakeBroker(overseas_candles=candles)
    monkeypatch.setattr(kis_provider, "_get_broker", _returns(fake))

    async def _yf_should_not_be_called(*a, **k):
        raise AssertionError("yfinance fallback should not be called when KIS succeeds")

    monkeypatch.setattr(kis_provider, "_yf_atr_pct", _yf_should_not_be_called)
    result = await kis_provider.get_atr_pct("AAPL", as_of=as_of)
    assert result > 0


async def test_kis_and_yfinance_atr_paths_agree_after_the_2026_08_21_window_widening(monkeypatch):
    """ATR TA-Lib PoC, commit 3/3 (see qlib-poc-decision-2026-08-21 project
    memory): the KIS path (~68 bars, commit 2's _MAX_KIS_CANDLE_RANGE_DAYS
    window) and the yfinance fallback path (~125 bars, "6mo") now feed
    talib.ATR's Wilder smoothing enough history to be near-converged on
    the same underlying price pattern — before commit 2 widened both
    windows to comparable depth, the two paths would have disagreed by
    several percent on the same ticker on the same day (Wilder, unlike
    the old SMA, keeps drifting toward its converged value with more
    history — see market_data's own convergence test)."""
    vals = [
        100, 102, 101, 103, 102, 104, 103, 105, 104, 106, 105, 107, 106, 108,
        107, 109, 108, 110, 109, 111, 110, 112, 111, 113, 112, 114, 113, 115, 114, 116,
    ]
    kis_n, yf_n = 68, 125
    kis_prices = (vals * ((kis_n // len(vals)) + 1))[:kis_n]
    yf_prices = (vals * ((yf_n // len(vals)) + 1))[:yf_n]

    base = date(2026, 5, 1)
    candles = [
        _candle((base + timedelta(days=i)).strftime("%Y%m%d"), p, p + 1.0, p - 1.0)
        for i, p in enumerate(kis_prices)
    ]
    fake = FakeBroker(candles=candles)
    monkeypatch.setattr(kis_provider, "_get_broker", _returns(fake))

    async def _yf_should_not_be_called(*a, **k):
        raise AssertionError("yfinance fallback should not be called when KIS succeeds")

    monkeypatch.setattr(kis_provider, "_yf_atr_pct", _yf_should_not_be_called)
    kis_result = await kis_provider.get_atr_pct("005930.KS")

    yf_close = pd.Series(yf_prices, dtype=float)
    yf_expected = _compute_atr_pct(yf_close + 1.0, yf_close - 1.0, yf_close, window=14)

    # 10%, not a tighter bound: this fixture's period-30 repeat creates its
    # own seam artifacts, so exact convergence isn't guaranteed — the point
    # is the OLD windows (21 vs 31 bars) disagreed by ~57% on this same
    # fixture, and the widened ones land within 10%.
    assert kis_result == pytest.approx(yf_expected, rel=0.10)


# ---------------------------------------------------------------------------
# get_atr_pct — 2026-09-11 still-forming-bar fix: during a live session the
# newest KIS candle is TODAY'S STILL-FORMING bar (its high/low cover only
# the part of the day traded so far), which pulled Wilder's ATR down a
# measured median 5.5%% low. The fix drops that candle from the series fed
# to talib.ATR but keeps its close as the denominator (the live price an
# order would actually fill at) — see kis_provider.get_atr_pct's own
# 2026-09-11 comment for the full rationale, including why the alternative
# (fall back to the previous close) was measured worse on gap days.
# ---------------------------------------------------------------------------


def _atr_candles(n: int, *, last_day: str, start: date) -> list:
    prices = [100.0 + i * 0.7 for i in range(n)]
    candles = [
        _candle((start + timedelta(days=i)).strftime("%Y%m%d"), p, p + 1.5, p - 1.5)
        for i, p in enumerate(prices)
    ]
    # The still-forming bar gets a much TIGHTER high/low than every completed
    # bar (a partial day's range genuinely is narrower than a full one) so
    # its True Range differs sharply from the rest of the series — otherwise,
    # on a fixture with a constant per-bar True Range, Wilder's ATR is
    # already fully converged and dropping one more identical bar changes
    # nothing, which would make this fixture unable to distinguish the fix
    # from the old behaviour at all.
    candles[-1] = _candle(last_day, prices[-1], prices[-1] + 0.05, prices[-1] - 0.05)
    return candles


def _talib_atr_pct(candles: list, denominator: float) -> float:
    highs = np.array([c.high for c in candles], dtype=float)
    lows = np.array([c.low for c in candles], dtype=float)
    closes = np.array([c.close for c in candles], dtype=float)
    atr = talib.ATR(highs, lows, closes, timeperiod=14)[-1]
    return float(atr / denominator)


async def test_get_atr_pct_krx_in_progress_session_excludes_forming_bar(monkeypatch):
    # KST 10:00 is inside the KRX regular session (09:00-15:30) — the newest
    # candle (dated "today", 2026-09-11) must be judged still-forming.
    set_clock(FakeClock(datetime(2026, 9, 11, 10, 0, tzinfo=_KST)))
    candles = _atr_candles(20, last_day="20260911", start=date(2026, 8, 1))
    fake = FakeBroker(candles=candles)
    monkeypatch.setattr(kis_provider, "_get_broker", _returns(fake))

    async def _yf_should_not_be_called(*a, **k):
        raise AssertionError("yfinance fallback should not be called when KIS succeeds")

    monkeypatch.setattr(kis_provider, "_yf_atr_pct", _yf_should_not_be_called)
    result = await kis_provider.get_atr_pct("005930.KS")

    # Expected: ATR from the first N-1 (completed) bars only, divided by the
    # still-forming candle's own close (the live price) — computed from
    # talib directly, not hardcoded.
    expected = _talib_atr_pct(candles[:-1], candles[-1].close)
    assert result == pytest.approx(expected)

    # Must NOT match what the OLD code produced: ATR over ALL bars (including
    # the still-forming one) divided by the same last close.
    old_value = _talib_atr_pct(candles, candles[-1].close)
    assert result != pytest.approx(old_value)


async def test_get_atr_pct_krx_outside_market_hours_is_byte_identical_to_before(monkeypatch):
    # KST 16:00 is after the KRX close (15:30) — even though the newest
    # candle is still dated "today", the session is not in progress, so the
    # output must be exactly the pre-fix computation (all bars, no trimming).
    set_clock(FakeClock(datetime(2026, 9, 11, 16, 0, tzinfo=_KST)))
    candles = _atr_candles(20, last_day="20260911", start=date(2026, 8, 1))
    fake = FakeBroker(candles=candles)
    monkeypatch.setattr(kis_provider, "_get_broker", _returns(fake))

    async def _yf_should_not_be_called(*a, **k):
        raise AssertionError("yfinance fallback should not be called when KIS succeeds")

    monkeypatch.setattr(kis_provider, "_yf_atr_pct", _yf_should_not_be_called)
    result = await kis_provider.get_atr_pct("005930.KS")

    expected = _talib_atr_pct(candles, candles[-1].close)
    assert result == pytest.approx(expected)


async def test_get_atr_pct_past_as_of_date_is_unaffected_even_during_market_hours(monkeypatch):
    # A backtest as_of in the past must never be treated as in-progress,
    # regardless of what the wall clock says right now.
    set_clock(FakeClock(datetime(2026, 9, 11, 10, 0, tzinfo=_KST)))
    as_of = date(2026, 8, 20)
    candles = _atr_candles(20, last_day=as_of.strftime("%Y%m%d"), start=date(2026, 7, 1))
    fake = FakeBroker(candles=candles)
    monkeypatch.setattr(kis_provider, "_get_broker", _returns(fake))

    async def _yf_should_not_be_called(*a, **k):
        raise AssertionError("yfinance fallback should not be called when KIS succeeds")

    monkeypatch.setattr(kis_provider, "_yf_atr_pct", _yf_should_not_be_called)
    result = await kis_provider.get_atr_pct("005930.KS", as_of=as_of)

    expected = _talib_atr_pct(candles, candles[-1].close)
    assert result == pytest.approx(expected)


async def test_get_atr_pct_in_progress_with_exactly_15_raw_candles_falls_back(monkeypatch):
    # 15 raw candles = 14 completed + 1 forming. After the forming bar is
    # dropped, only 14 remain — one short of window(14)+1=15 needed for a
    # real ATR — so this must fall through to yfinance, not compute from 14.
    set_clock(FakeClock(datetime(2026, 9, 11, 10, 0, tzinfo=_KST)))
    candles = _atr_candles(15, last_day="20260911", start=date(2026, 8, 20))
    fake = FakeBroker(candles=candles)
    monkeypatch.setattr(kis_provider, "_get_broker", _returns(fake))

    async def fake_yf_atr(ticker, as_of=None):
        return 0.031

    monkeypatch.setattr(kis_provider, "_yf_atr_pct", fake_yf_atr)
    result = await kis_provider.get_atr_pct("005930.KS")
    assert result == 0.031


async def test_get_atr_pct_in_progress_with_exactly_16_raw_candles_computes(monkeypatch):
    # The boundary's other side: 16 raw candles = 15 completed + 1 forming ->
    # exactly enough (15) after the drop to compute a real ATR from KIS.
    set_clock(FakeClock(datetime(2026, 9, 11, 10, 0, tzinfo=_KST)))
    candles = _atr_candles(16, last_day="20260911", start=date(2026, 8, 20))
    fake = FakeBroker(candles=candles)
    monkeypatch.setattr(kis_provider, "_get_broker", _returns(fake))

    async def _yf_should_not_be_called(*a, **k):
        raise AssertionError("yfinance fallback should not be called when KIS succeeds")

    monkeypatch.setattr(kis_provider, "_yf_atr_pct", _yf_should_not_be_called)
    result = await kis_provider.get_atr_pct("005930.KS")

    expected = _talib_atr_pct(candles[:-1], candles[-1].close)
    assert result == pytest.approx(expected)


async def test_get_price_series_falls_back_for_wide_date_range(monkeypatch):
    async def _should_not_be_called():
        raise AssertionError("KIS should be skipped when range exceeds the candle cap")

    monkeypatch.setattr(kis_provider, "_get_broker", _should_not_be_called)

    async def fake_yf_price_series(ticker, start, end):
        return {date(2026, 1, 1): 100.0}

    monkeypatch.setattr(kis_provider, "_yf_price_series", fake_yf_price_series)
    result = await kis_provider.get_price_series("005930.KS", date(2026, 1, 1), date(2027, 6, 1))
    assert result == {date(2026, 1, 1): 100.0}


async def test_get_price_series_uses_kis_within_range_cap(monkeypatch):
    fake = FakeBroker(candles=[_candle("20260715", 100.0), _candle("20260716", 105.0)])
    monkeypatch.setattr(kis_provider, "_get_broker", _returns(fake))
    result = await kis_provider.get_price_series("005930.KS", date(2026, 7, 15), date(2026, 7, 16))
    assert result == {date(2026, 7, 15): 100.0, date(2026, 7, 16): 105.0}


async def test_get_daily_low_series_uses_kis_lows(monkeypatch):
    fake = FakeBroker(candles=[_candle("20260716", close=105.0, low=101.0)])
    monkeypatch.setattr(kis_provider, "_get_broker", _returns(fake))
    result = await kis_provider.get_daily_low_series("005930.KS", date(2026, 7, 16), date(2026, 7, 16))
    assert result == {date(2026, 7, 16): 101.0}


async def test_get_price_series_uses_kis_for_us_ticker(monkeypatch):
    fake = FakeBroker(overseas_candles=[_candle("20260715", 100.0), _candle("20260716", 105.0)])
    monkeypatch.setattr(kis_provider, "_get_broker", _returns(fake))
    result = await kis_provider.get_price_series("AAPL", date(2026, 7, 15), date(2026, 7, 16))
    assert result == {date(2026, 7, 15): 100.0, date(2026, 7, 16): 105.0}


async def test_get_price_series_filters_overseas_candles_to_requested_range(monkeypatch):
    # fetch_overseas_candles has no `start` param — KIS returns up to ~100
    # rows ending at `end` regardless, so the dispatcher must filter client-side.
    fake = FakeBroker(overseas_candles=[
        _candle("20260710", 90.0), _candle("20260715", 100.0), _candle("20260716", 105.0),
    ])
    monkeypatch.setattr(kis_provider, "_get_broker", _returns(fake))
    result = await kis_provider.get_price_series("AAPL", date(2026, 7, 15), date(2026, 7, 16))
    assert result == {date(2026, 7, 15): 100.0, date(2026, 7, 16): 105.0}


async def test_get_daily_low_series_uses_kis_for_us_ticker(monkeypatch):
    fake = FakeBroker(overseas_candles=[_candle("20260716", close=205.0, low=201.0)])
    monkeypatch.setattr(kis_provider, "_get_broker", _returns(fake))
    result = await kis_provider.get_daily_low_series("AAPL", date(2026, 7, 16), date(2026, 7, 16))
    assert result == {date(2026, 7, 16): 201.0}


async def test_get_daily_high_series_uses_kis_highs(monkeypatch):
    fake = FakeBroker(candles=[_candle("20260716", close=105.0, high=109.0)])
    monkeypatch.setattr(kis_provider, "_get_broker", _returns(fake))
    result = await kis_provider.get_daily_high_series("005930.KS", date(2026, 7, 16), date(2026, 7, 16))
    assert result == {date(2026, 7, 16): 109.0}


async def test_get_daily_high_series_uses_kis_for_us_ticker(monkeypatch):
    fake = FakeBroker(overseas_candles=[_candle("20260716", close=205.0, high=209.0)])
    monkeypatch.setattr(kis_provider, "_get_broker", _returns(fake))
    result = await kis_provider.get_daily_high_series("AAPL", date(2026, 7, 16), date(2026, 7, 16))
    assert result == {date(2026, 7, 16): 209.0}


async def test_get_daily_high_series_falls_back_to_yfinance_on_kis_failure(monkeypatch):
    class _FailingBroker(FakeBroker):
        async def fetch_daily_candles(self, *args, **kwargs):
            raise RuntimeError("KIS candle fetch failed")

    monkeypatch.setattr(kis_provider, "_get_broker", _returns(_FailingBroker()))

    async def fake_yf_high_series(ticker, start, end):
        return {date(2026, 7, 16): 999.0}

    monkeypatch.setattr(kis_provider, "_yf_daily_high_series", fake_yf_high_series)
    result = await kis_provider.get_daily_high_series("005930.KS", date(2026, 7, 16), date(2026, 7, 16))
    assert result == {date(2026, 7, 16): 999.0}


class _FakeDisplayInfoBroker:
    def __init__(
        self, is_real, candles=None, korean_name="", raise_status_exc=None,
        overseas_korean_name="", raise_overseas_name_exc=None, security_group_code="",
    ):
        self._is_real = is_real
        self._candles = candles if candles is not None else [_candle("20260716", 105.0)]
        self._korean_name = korean_name
        self._raise_status_exc = raise_status_exc
        self._overseas_korean_name = overseas_korean_name
        self._raise_overseas_name_exc = raise_overseas_name_exc
        self._security_group_code = security_group_code

    async def fetch_stock_status(self, code):
        if self._raise_status_exc is not None:
            raise self._raise_status_exc
        return StockStatus(
            code=code, trading_halted=False, administrative_issue=False, nxt_trading_halted=False,
            korean_name=self._korean_name, english_name="", security_group_code=self._security_group_code,
        )

    async def fetch_overseas_company_name(self, symbol, prdt_type_cd="512"):
        if self._raise_overseas_name_exc is not None:
            raise self._raise_overseas_name_exc
        return self._overseas_korean_name, ""

    async def fetch_daily_candles(self, code, start, end):
        return self._candles


async def _no_op_ensure_loaded():
    return None


async def test_get_ticker_display_info_uses_master_file_name_regardless_of_account_mode(monkeypatch):
    """Master file is checked before any KIS TR or account-mode branching —
    it should win even on a paper account, unlike the old yfinance-default
    behavior this replaces."""
    from app.dataflows.yfinance.market_data import TickerDisplayInfo

    fake = _FakeDisplayInfoBroker(is_real=False)
    monkeypatch.setattr(kis_provider, "_get_broker", _returns(fake))
    monkeypatch.setattr(kis_provider.krx_master, "ensure_loaded", _no_op_ensure_loaded)
    monkeypatch.setattr(kis_provider.krx_master, "get", lambda ticker: "삼성E&A" if ticker == "028050.KS" else None)

    async def fake_yf_display_info(ticker):
        return TickerDisplayInfo(symbol=ticker, name="Hyundai Motor Co", previous_close=100.0, currency="USD")

    monkeypatch.setattr(kis_provider, "_yf_display_info", fake_yf_display_info)
    result = await kis_provider.get_ticker_display_info("028050.KS")
    assert result.name == "삼성E&A"


async def test_get_ticker_display_info_keeps_master_file_name_when_yfinance_raises(monkeypatch):
    """A yfinance failure (network/rate-limit, not just a bad name) must not
    take the already-resolved master-file name down with it — the
    2026-07-29 gap this replaces (dashboard showed bare ticker codes for
    005380.KS etc. despite the master file having the right answer)."""
    fake = _FakeDisplayInfoBroker(is_real=False)
    monkeypatch.setattr(kis_provider, "_get_broker", _returns(fake))
    monkeypatch.setattr(kis_provider.krx_master, "ensure_loaded", _no_op_ensure_loaded)
    monkeypatch.setattr(kis_provider.krx_master, "get", lambda ticker: "현대차" if ticker == "005380.KS" else None)

    async def raising_yf_display_info(ticker):
        raise RuntimeError("yfinance rate-limited")

    monkeypatch.setattr(kis_provider, "_yf_display_info", raising_yf_display_info)
    result = await kis_provider.get_ticker_display_info("005380.KS")
    assert result.name == "현대차"
    assert result.currency == "KRW"


async def test_get_ticker_display_info_falls_back_to_bare_ticker_when_both_sources_fail(monkeypatch):
    """No master-file entry (e.g. a same-day new listing) and yfinance also
    down: same "bare ticker" outcome as before this change — no new gap for
    that specific double failure, just one fewer single point of failure.
    previous_close still comes from KIS candles independently (unaffected
    by the yfinance failure), so it's asserted separately below rather
    than assumed to stay at the placeholder here."""
    fake = _FakeDisplayInfoBroker(is_real=False)
    monkeypatch.setattr(kis_provider, "_get_broker", _returns(fake))
    monkeypatch.setattr(kis_provider.krx_master, "ensure_loaded", _no_op_ensure_loaded)
    monkeypatch.setattr(kis_provider.krx_master, "get", lambda ticker: None)

    async def raising_yf_display_info(ticker):
        raise RuntimeError("yfinance rate-limited")

    monkeypatch.setattr(kis_provider, "_yf_display_info", raising_yf_display_info)
    result = await kis_provider.get_ticker_display_info("999999.KS")
    assert result.name == "999999.KS"


async def test_get_ticker_display_info_keeps_placeholder_previous_close_when_kis_candles_also_fail(monkeypatch):
    """The fully-degraded case: yfinance down AND KIS candles unavailable
    (e.g. broker misconfigured) — previous_close stays the 0.0 placeholder
    rather than raising, same "never block on a display-info failure"
    contract as before."""
    fake = _FakeDisplayInfoBroker(is_real=False, raise_status_exc=RuntimeError("candles down"))
    fake._candles = None

    async def raising_fetch_daily_candles(code, start, end):
        raise RuntimeError("candles down")

    fake.fetch_daily_candles = raising_fetch_daily_candles
    monkeypatch.setattr(kis_provider, "_get_broker", _returns(fake))
    monkeypatch.setattr(kis_provider.krx_master, "ensure_loaded", _no_op_ensure_loaded)
    monkeypatch.setattr(kis_provider.krx_master, "get", lambda ticker: None)

    async def raising_yf_display_info(ticker):
        raise RuntimeError("yfinance rate-limited")

    monkeypatch.setattr(kis_provider, "_yf_display_info", raising_yf_display_info)
    result = await kis_provider.get_ticker_display_info("999999.KS")
    assert result.name == "999999.KS"
    assert result.previous_close == 0.0


async def test_get_ticker_display_info_reraises_when_yfinance_fails_for_unsupported_ticker(monkeypatch):
    """A ticker matching neither KRX nor US shape has no fallback source —
    the original raise-through behavior is preserved for that case."""
    async def raising_yf_display_info(ticker):
        raise RuntimeError("yfinance failure")

    monkeypatch.setattr(kis_provider, "_yf_display_info", raising_yf_display_info)
    with pytest.raises(RuntimeError):
        await kis_provider.get_ticker_display_info("not-a-real-ticker")


async def test_get_ticker_display_info_uses_kis_name_on_real_account(monkeypatch):
    from app.dataflows.yfinance.market_data import TickerDisplayInfo

    fake = _FakeDisplayInfoBroker(is_real=True, korean_name="삼성전자")
    monkeypatch.setattr(kis_provider, "_get_broker", _returns(fake))
    monkeypatch.setattr(kis_provider.krx_master, "ensure_loaded", _no_op_ensure_loaded)
    monkeypatch.setattr(kis_provider.krx_master, "get", lambda ticker: None)

    async def fake_yf_display_info(ticker):
        return TickerDisplayInfo(symbol=ticker, name="Samsung Electronics Co Ltd", previous_close=100.0, currency="USD")

    monkeypatch.setattr(kis_provider, "_yf_display_info", fake_yf_display_info)
    result = await kis_provider.get_ticker_display_info("005930.KS")
    assert result.name == "삼성전자"
    assert result.previous_close == 105.0 and result.currency == "KRW"


async def test_get_ticker_display_info_overrides_mutualfund_quote_type_with_kis_security_group(monkeypatch):
    """Root-cause fix (2026-09-03): yfinance misreported 228340.KS (동양파일)
    and 142760.KS (모아라이프플러스), both ordinary KOSDAQ stocks, as
    quoteType "MUTUALFUND". On a real account, CTPF1002R's scty_grp_id_cd
    ("ST" = 주권/ordinary stock) is authoritative and must override
    yfinance's bad value so app.domain.leverage_filter.is_unsupported_
    product_type doesn't block a legitimate BUY."""
    from app.dataflows.yfinance.market_data import TickerDisplayInfo

    fake = _FakeDisplayInfoBroker(is_real=True, security_group_code="ST")
    monkeypatch.setattr(kis_provider, "_get_broker", _returns(fake))
    monkeypatch.setattr(kis_provider.krx_master, "ensure_loaded", _no_op_ensure_loaded)
    monkeypatch.setattr(kis_provider.krx_master, "get", lambda ticker: "동양파일" if ticker == "228340.KS" else None)

    async def fake_yf_display_info(ticker):
        return TickerDisplayInfo(
            symbol=ticker, name="Tongyang Pile Inc", previous_close=5000.0, currency="KRW",
            quote_type="MUTUALFUND",
        )

    monkeypatch.setattr(kis_provider, "_yf_display_info", fake_yf_display_info)
    result = await kis_provider.get_ticker_display_info("228340.KS")
    assert result.quote_type == "EQUITY"


async def test_get_ticker_display_info_maps_kis_etf_security_group_to_etf_quote_type(monkeypatch):
    from app.dataflows.yfinance.market_data import TickerDisplayInfo

    fake = _FakeDisplayInfoBroker(is_real=True, security_group_code="EF")
    monkeypatch.setattr(kis_provider, "_get_broker", _returns(fake))
    monkeypatch.setattr(kis_provider.krx_master, "ensure_loaded", _no_op_ensure_loaded)
    monkeypatch.setattr(kis_provider.krx_master, "get", lambda ticker: None)

    async def fake_yf_display_info(ticker):
        return TickerDisplayInfo(symbol=ticker, name="Some ETF", previous_close=10000.0, currency="KRW", quote_type="")

    monkeypatch.setattr(kis_provider, "_yf_display_info", fake_yf_display_info)
    result = await kis_provider.get_ticker_display_info("069500.KS")
    assert result.quote_type == "ETF"


async def test_get_ticker_display_info_passes_through_an_unmapped_kis_security_group(monkeypatch):
    """A genuine mutual fund (KIS security group "MF") isn't in the
    ST/EF/FE whitelist, so it passes through unmapped rather than being
    coerced into EQUITY/ETF — is_unsupported_product_type still blocks it,
    correctly this time (verified against the real product type, not
    yfinance's guess)."""
    from app.dataflows.yfinance.market_data import TickerDisplayInfo

    fake = _FakeDisplayInfoBroker(is_real=True, security_group_code="MF")
    monkeypatch.setattr(kis_provider, "_get_broker", _returns(fake))
    monkeypatch.setattr(kis_provider.krx_master, "ensure_loaded", _no_op_ensure_loaded)
    monkeypatch.setattr(kis_provider.krx_master, "get", lambda ticker: None)

    async def fake_yf_display_info(ticker):
        return TickerDisplayInfo(symbol=ticker, name="Actual Mutual Fund", previous_close=1000.0, currency="KRW", quote_type="")

    monkeypatch.setattr(kis_provider, "_yf_display_info", fake_yf_display_info)
    result = await kis_provider.get_ticker_display_info("000001.KS")
    assert result.quote_type == "MF"


async def test_get_ticker_display_info_keeps_yfinance_quote_type_on_paper_account(monkeypatch):
    """No KIS TR override on paper (EGW02006 for the whole TR) — quote_type
    stays whatever yfinance returned, same as before this fix."""
    from app.dataflows.yfinance.market_data import TickerDisplayInfo

    fake = _FakeDisplayInfoBroker(is_real=False, security_group_code="ST")
    monkeypatch.setattr(kis_provider, "_get_broker", _returns(fake))
    monkeypatch.setattr(kis_provider.krx_master, "ensure_loaded", _no_op_ensure_loaded)
    monkeypatch.setattr(kis_provider.krx_master, "get", lambda ticker: None)

    async def fake_yf_display_info(ticker):
        return TickerDisplayInfo(
            symbol=ticker, name="Tongyang Pile Inc", previous_close=5000.0, currency="KRW",
            quote_type="MUTUALFUND",
        )

    monkeypatch.setattr(kis_provider, "_yf_display_info", fake_yf_display_info)
    result = await kis_provider.get_ticker_display_info("228340.KS")
    assert result.quote_type == "MUTUALFUND"


async def test_get_ticker_display_info_keeps_yfinance_name_on_master_file_and_paper_miss(monkeypatch):
    """When the master file has no entry (e.g. a same-day new listing) and
    the account is paper (so the KIS TR override never fires), yfinance's
    value is what's left."""
    from app.dataflows.yfinance.market_data import TickerDisplayInfo

    fake = _FakeDisplayInfoBroker(is_real=False, korean_name="삼성전자")
    monkeypatch.setattr(kis_provider, "_get_broker", _returns(fake))
    monkeypatch.setattr(kis_provider.krx_master, "ensure_loaded", _no_op_ensure_loaded)
    monkeypatch.setattr(kis_provider.krx_master, "get", lambda ticker: None)

    async def fake_yf_display_info(ticker):
        return TickerDisplayInfo(symbol=ticker, name="Samsung Electronics Co Ltd", previous_close=100.0, currency="USD")

    monkeypatch.setattr(kis_provider, "_yf_display_info", fake_yf_display_info)
    result = await kis_provider.get_ticker_display_info("005930.KS")
    assert result.name == "Samsung Electronics Co Ltd"
    assert result.previous_close == 105.0 and result.currency == "KRW"


async def test_get_ticker_display_info_falls_back_to_yfinance_name_on_status_failure(monkeypatch):
    from app.dataflows.yfinance.market_data import TickerDisplayInfo

    fake = _FakeDisplayInfoBroker(is_real=True, raise_status_exc=RuntimeError("boom"))
    monkeypatch.setattr(kis_provider, "_get_broker", _returns(fake))
    monkeypatch.setattr(kis_provider.krx_master, "ensure_loaded", _no_op_ensure_loaded)
    monkeypatch.setattr(kis_provider.krx_master, "get", lambda ticker: None)

    async def fake_yf_display_info(ticker):
        return TickerDisplayInfo(symbol=ticker, name="Samsung Electronics Co Ltd", previous_close=100.0, currency="USD")

    monkeypatch.setattr(kis_provider, "_yf_display_info", fake_yf_display_info)
    result = await kis_provider.get_ticker_display_info("005930.KS")
    assert result.name == "Samsung Electronics Co Ltd"


async def test_get_ticker_display_info_preserves_quote_type(monkeypatch):
    from app.dataflows.yfinance.market_data import TickerDisplayInfo

    fake = _FakeDisplayInfoBroker(is_real=True, korean_name="삼성전자")
    monkeypatch.setattr(kis_provider, "_get_broker", _returns(fake))
    monkeypatch.setattr(kis_provider.krx_master, "ensure_loaded", _no_op_ensure_loaded)
    monkeypatch.setattr(kis_provider.krx_master, "get", lambda ticker: None)

    async def fake_yf_display_info(ticker):
        return TickerDisplayInfo(
            symbol=ticker, name="Samsung Electronics Co Ltd", previous_close=100.0, currency="USD", quote_type="EQUITY"
        )

    monkeypatch.setattr(kis_provider, "_yf_display_info", fake_yf_display_info)
    result = await kis_provider.get_ticker_display_info("005930.KS")
    assert result.quote_type == "EQUITY"


async def test_get_ticker_display_info_prefers_yfinance_english_name_for_us_ticker_over_master_file(monkeypatch):
    """Reversed 2026-08-20 (identity-misattribution fix): overseas_master's
    value is a KOREAN TRANSLITERATION (nasmst.cod col 6), not the master
    file's own native-script name the way krx_master's is for a KRX code.
    Handing that transliteration to an English-output LLM as the "verified"
    label was the confirmed cause of the QQQ->"인베스코 QQK 트러스트" /
    PANW->"팽탈 앨토 네트웍스" corruption — see kis_provider.get_ticker_display_
    info's own docstring/inline comment. yfinance's own English longName is
    the correct ground truth for a US ticker and must win, mirroring (not
    matching) the KRX master-file-wins test above."""
    from app.dataflows.yfinance.market_data import TickerDisplayInfo

    fake = _FakeDisplayInfoBroker(is_real=False)
    monkeypatch.setattr(kis_provider, "_get_broker", _returns(fake))
    monkeypatch.setattr(kis_provider.overseas_master, "ensure_loaded", _no_op_ensure_loaded)
    monkeypatch.setattr(kis_provider.overseas_master, "get", lambda ticker: "애플" if ticker == "AAPL" else None)

    async def fake_yf_display_info(ticker):
        return TickerDisplayInfo(symbol=ticker, name="Apple Inc.", previous_close=200.0, currency="USD")

    monkeypatch.setattr(kis_provider, "_yf_display_info", fake_yf_display_info)
    result = await kis_provider.get_ticker_display_info("AAPL")
    assert result.name == "Apple Inc."


async def test_get_ticker_display_info_keeps_yfinance_english_name_for_us_ticker_on_real_account(monkeypatch):
    """Same 2026-08-20 fix: fetch_overseas_company_name also returns a
    Korean transliteration (CTPF1702R), so a real account must not let it
    overwrite yfinance's already-correct English name either — only fill in
    when no name is known yet at all (see the `if not name or name ==
    ticker` guard in kis_provider.get_ticker_display_info)."""
    from app.dataflows.yfinance.market_data import TickerDisplayInfo

    fake = _FakeDisplayInfoBroker(is_real=True, overseas_korean_name="애플")
    monkeypatch.setattr(kis_provider, "_get_broker", _returns(fake))
    monkeypatch.setattr(kis_provider.overseas_master, "ensure_loaded", _no_op_ensure_loaded)
    monkeypatch.setattr(kis_provider.overseas_master, "get", lambda ticker: None)

    async def fake_yf_display_info(ticker):
        return TickerDisplayInfo(symbol=ticker, name="Apple Inc.", previous_close=200.0, currency="USD")

    monkeypatch.setattr(kis_provider, "_yf_display_info", fake_yf_display_info)
    result = await kis_provider.get_ticker_display_info("AAPL")
    assert result.name == "Apple Inc."
    # Unlike KRX, previous_close/currency are out of scope here — stay yfinance's own.
    assert result.previous_close == 200.0 and result.currency == "USD"


async def test_get_ticker_display_info_uses_kis_korean_name_for_us_ticker_when_no_name_known(monkeypatch):
    """The real-account KIS lookup is still useful as a last resort — when
    neither overseas_master nor yfinance produced anything (`name` is empty
    or still the bare ticker), filling in the Korean name beats showing no
    name at all."""
    from app.dataflows.yfinance.market_data import TickerDisplayInfo

    fake = _FakeDisplayInfoBroker(is_real=True, overseas_korean_name="애플")
    monkeypatch.setattr(kis_provider, "_get_broker", _returns(fake))
    monkeypatch.setattr(kis_provider.overseas_master, "ensure_loaded", _no_op_ensure_loaded)
    monkeypatch.setattr(kis_provider.overseas_master, "get", lambda ticker: None)

    async def fake_yf_display_info(ticker):
        return TickerDisplayInfo(symbol=ticker, name="AAPL", previous_close=200.0, currency="USD")

    monkeypatch.setattr(kis_provider, "_yf_display_info", fake_yf_display_info)
    result = await kis_provider.get_ticker_display_info("AAPL")
    assert result.name == "애플"


async def test_get_ticker_display_info_keeps_yfinance_name_for_us_ticker_on_master_file_and_paper_miss(monkeypatch):
    from app.dataflows.yfinance.market_data import TickerDisplayInfo

    fake = _FakeDisplayInfoBroker(is_real=False, overseas_korean_name="애플")
    monkeypatch.setattr(kis_provider, "_get_broker", _returns(fake))
    monkeypatch.setattr(kis_provider.overseas_master, "ensure_loaded", _no_op_ensure_loaded)
    monkeypatch.setattr(kis_provider.overseas_master, "get", lambda ticker: None)

    async def fake_yf_display_info(ticker):
        return TickerDisplayInfo(symbol=ticker, name="Apple Inc.", previous_close=200.0, currency="USD")

    monkeypatch.setattr(kis_provider, "_yf_display_info", fake_yf_display_info)
    result = await kis_provider.get_ticker_display_info("AAPL")
    assert result.name == "Apple Inc."


async def test_get_ticker_display_info_falls_back_for_us_ticker_on_name_lookup_failure(monkeypatch):
    from app.dataflows.yfinance.market_data import TickerDisplayInfo

    fake = _FakeDisplayInfoBroker(is_real=True, raise_overseas_name_exc=RuntimeError("boom"))
    monkeypatch.setattr(kis_provider, "_get_broker", _returns(fake))
    monkeypatch.setattr(kis_provider.overseas_master, "ensure_loaded", _no_op_ensure_loaded)
    monkeypatch.setattr(kis_provider.overseas_master, "get", lambda ticker: None)

    async def fake_yf_display_info(ticker):
        return TickerDisplayInfo(symbol=ticker, name="Apple Inc.", previous_close=200.0, currency="USD")

    monkeypatch.setattr(kis_provider, "_yf_display_info", fake_yf_display_info)
    result = await kis_provider.get_ticker_display_info("AAPL")
    assert result.name == "Apple Inc."


async def test_get_ticker_display_info_unsupported_ticker_returns_yfinance_info_untouched(monkeypatch):
    async def _should_not_be_called():
        raise AssertionError("KIS broker should not be looked up for a ticker matching neither market")

    monkeypatch.setattr(kis_provider, "_get_broker", _should_not_be_called)

    from app.dataflows.yfinance.market_data import TickerDisplayInfo

    async def fake_yf_display_info(ticker):
        return TickerDisplayInfo(symbol=ticker, name="Some Fund", previous_close=1.0, currency="HKD")

    monkeypatch.setattr(kis_provider, "_yf_display_info", fake_yf_display_info)
    result = await kis_provider.get_ticker_display_info("0700.HK")
    assert result.name == "Some Fund"


async def test_get_investor_flow_returns_not_supported_for_non_krx():
    result = await kis_provider.get_investor_flow("AAPL")
    assert "한국(KRX) 종목만" in result


async def test_get_investor_flow_returns_not_configured_message(monkeypatch):
    async def _broken_broker():
        raise RuntimeError("not configured")

    monkeypatch.setattr(kis_provider, "_get_broker", _broken_broker)
    result = await kis_provider.get_investor_flow("005930.KS")
    assert "KIS 미설정" in result


# ---------------------------------------------------------------------------
# VendorRouter-registered KIS-first providers (fundamental_data,
# analyst_consensus, news_data, short_interest, balance_sheet) — real
# incidents this closes off: 218150.KS (yfinance quoteType MUTUALFUND, a
# broken shortName) and 064260.KS (yfinance's longName resolving to a
# completely unrelated company) both happened on domestic tickers, so
# domestic fundamentals/news/etc. should never reach yfinance at all now.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "wrapper_name,delegate_name",
    [
        ("_kis_fundamentals", "get_kr_financials"),
        ("_kis_analyst_consensus", "get_kr_analyst_opinion"),
        ("_kis_news", "get_kr_news"),
        ("_kis_short_interest", "get_kr_risk_signals"),
    ],
)
async def test_kis_wrapper_raises_for_non_krx_ticker(monkeypatch, wrapper_name, delegate_name):
    # 0700.HK matches neither the KRX (.KS) nor the US (1-5 uppercase
    # letters) shape — AAPL no longer works here now that _kis_news also
    # accepts US-ticker-shaped symbols (see module docstring).
    async def should_not_be_called(*args, **kwargs):
        raise AssertionError(f"{delegate_name} should not be called for an unsupported ticker")

    monkeypatch.setattr(kis_provider, delegate_name, should_not_be_called)
    wrapper = getattr(kis_provider, wrapper_name)
    with pytest.raises(ValueError, match="is not a KRX"):
        await wrapper("0700.HK")


@pytest.mark.parametrize(
    "wrapper_name,delegate_name",
    [
        ("_kis_fundamentals", "get_kr_financials"),
        ("_kis_analyst_consensus", "get_kr_analyst_opinion"),
        ("_kis_news", "get_kr_news"),
        ("_kis_short_interest", "get_kr_risk_signals"),
    ],
)
async def test_kis_wrapper_delegates_for_krx_ticker(monkeypatch, wrapper_name, delegate_name):
    async def fake_delegate(ticker, as_of=None):
        return f"KIS data for {ticker}"

    monkeypatch.setattr(kis_provider, delegate_name, fake_delegate)
    wrapper = getattr(kis_provider, wrapper_name)
    assert await wrapper("005930.KS") == "KIS data for 005930.KS"


# ---------------------------------------------------------------------------
# get_kr_risk_signals's program-trade line — 2026-07-30: net_buy_amount is
# raw KRW (unlike the other KIS financials fields already in 억원), and a
# live decision (042660.KS) reported "4,500억원" for an actual ~456억원
# figure — a 10x error despite the number being comma-formatted with no
# scientific notation involved. Fixed by converting to 억원 here so the
# model no longer has to do that division itself.
# ---------------------------------------------------------------------------


async def test_get_kr_risk_signals_converts_program_trade_amount_to_eok(monkeypatch):
    fake = FakeBroker(program_trade=[ProgramTrade(slot="151818", net_buy_qty=574310, net_buy_amount=45613179100.0)])
    monkeypatch.setattr(kis_provider, "_get_broker", _returns(fake))
    result = await kis_provider.get_kr_risk_signals("042660.KS")
    assert "456.1억원" in result
    assert "45,613,179,100" not in result


# ---------------------------------------------------------------------------
# get_kr_risk_signals's [Short-selling] block — 2026-09-01, decision id=1156
# (267260.KS/HD현대일렉트릭): `_num` coerces a missing KIS short-sale field
# to 0.0, and `fetch_daily_short_sale` builds a row whenever the date is
# present, so a not-yet-finalized current-day row arrived as all-zeros and
# the PM read the 0.00% short ratio as "liquidity has dried up" (confidence
# 90%, #1 rationale). Same failure class as _pct_or_missing / id=785.
# ---------------------------------------------------------------------------

_KST = ZoneInfo("Asia/Seoul")


def _short(day, close, vol, ratio, accum):
    return ShortSaleDay(day=day, close=close, short_volume=vol, short_ratio=ratio, accum_short_ratio=accum)


def _short_rows(result: str) -> list[str]:
    return [ln for ln in result.splitlines() if ln[:8].isdigit()]


async def test_short_selling_real_zero_day_still_renders_zero(monkeypatch):
    # A day with zero short-sale executions is a real 0.00%, NOT missing —
    # the regression guard against a blanket 0.0 -> missing substitution.
    fake = FakeBroker(daily_short_sale=[_short("20260828", 12000.0, 0, 0.0, 0.0)])
    monkeypatch.setattr(kis_provider, "_get_broker", _returns(fake))
    result = await kis_provider.get_kr_risk_signals("005930.KS", as_of=date(2026, 8, 31))
    row = _short_rows(result)[0]
    assert "0.00" in row
    assert "missing" not in row


async def test_short_selling_impossible_zero_ratio_is_missing(monkeypatch):
    # short_volume > 0 but ratio == 0.0 is arithmetically impossible -> missing.
    fake = FakeBroker(daily_short_sale=[_short("20260828", 12000.0, 330953, 0.0, 0.0)])
    monkeypatch.setattr(kis_provider, "_get_broker", _returns(fake))
    result = await kis_provider.get_kr_risk_signals("005930.KS", as_of=date(2026, 8, 31))
    block = result.split("[Short-selling]")[1]
    assert "missing" in block
    assert "330,953" in block  # the volume itself is real, kept


async def test_short_selling_accum_field_follows_the_same_rule(monkeypatch):
    fake = FakeBroker(daily_short_sale=[_short("20260828", 12000.0, 5000, 1.04, 0.0)])
    monkeypatch.setattr(kis_provider, "_get_broker", _returns(fake))
    result = await kis_provider.get_kr_risk_signals("005930.KS", as_of=date(2026, 8, 31))
    block = result.split("[Short-selling]")[1].split("[Program")[0]
    assert "1.04" in block  # short_ratio kept
    assert "missing" in block  # accum flagged


async def test_short_selling_row_with_zero_close_is_all_missing(monkeypatch):
    fake = FakeBroker(daily_short_sale=[_short("20260828", 0.0, 0, 0.0, 0.0)])
    monkeypatch.setattr(kis_provider, "_get_broker", _returns(fake))
    result = await kis_provider.get_kr_risk_signals("005930.KS", as_of=date(2026, 8, 31))
    rows = _short_rows(result)
    assert len(rows) == 1
    assert rows[0].count("missing") == 4


async def test_short_selling_current_day_row_is_labelled_unfinalized(monkeypatch):
    set_clock(FakeClock(datetime(2026, 9, 1, 10, 30, tzinfo=_KST)))
    fake = FakeBroker(daily_short_sale=[
        _short("20260901", 12000.0, 0, 0.0, 0.0),
        _short("20260829", 12000.0, 5000, 1.04, 3.40),
    ])
    monkeypatch.setattr(kis_provider, "_get_broker", _returns(fake))
    result = await kis_provider.get_kr_risk_signals("005930.KS")
    lines = {ln[:8]: ln for ln in result.splitlines() if ln[:8].isdigit()}
    assert "not yet finalized" in lines["20260901"]
    assert "not yet finalized" not in lines["20260829"]


async def test_short_selling_id1156_shape_no_bare_zero_pct(monkeypatch):
    set_clock(FakeClock(datetime(2026, 9, 1, 10, 30, tzinfo=_KST)))
    fake = FakeBroker(daily_short_sale=[_short("20260901", 12000.0, 88000, 0.0, 0.0)])
    monkeypatch.setattr(kis_provider, "_get_broker", _returns(fake))
    result = await kis_provider.get_kr_risk_signals("005930.KS")
    row = [ln for ln in result.splitlines() if ln.startswith("20260901")][0]
    assert "0.00" not in row
    assert "missing" in row and "not yet finalized" in row


async def test_short_selling_normal_series_unchanged(monkeypatch):
    fake = FakeBroker(daily_short_sale=[
        _short("20260828", 12000.0, 5000, 1.04, 3.43),
        _short("20260827", 11900.0, 4200, 0.87, 3.21),
    ])
    monkeypatch.setattr(kis_provider, "_get_broker", _returns(fake))
    result = await kis_provider.get_kr_risk_signals("005930.KS", as_of=date(2026, 8, 31))
    rows = _short_rows(result)
    assert not any("missing" in r for r in rows)
    joined = "\n".join(rows)
    for want in ("12,000", "5,000", "1.04", "3.43", "0.87", "3.21"):
        assert want in joined
    # column alignment preserved: every data row is the same width
    assert len({len(r.rstrip()) for r in rows}) == 1


async def test_short_selling_empty_series_message_unchanged(monkeypatch):
    fake = FakeBroker(daily_short_sale=[])
    monkeypatch.setattr(kis_provider, "_get_broker", _returns(fake))
    result = await kis_provider.get_kr_risk_signals("005930.KS", as_of=date(2026, 8, 31))
    assert "[Short-selling] no recent short-sale data" in result


async def test_short_selling_non_krx_ticker_unchanged(monkeypatch):
    result = await kis_provider.get_kr_risk_signals("AAPL")
    assert "한국(KRX) 종목만" in result


async def test_short_selling_header_explains_missing_and_unfinalized(monkeypatch):
    fake = FakeBroker(daily_short_sale=[_short("20260828", 12000.0, 5000, 1.04, 3.43)])
    monkeypatch.setattr(kis_provider, "_get_broker", _returns(fake))
    result = await kis_provider.get_kr_risk_signals("005930.KS", as_of=date(2026, 8, 31))
    assert "`missing`" in result and "NOT zero" in result
    assert "partial/unfinalized" in result
    assert "SELL-side pressure" in result


# ---------------------------------------------------------------------------
# get_kr_risk_signals's two bonus TRs must not abort the whole tool — mirrors
# get_kr_analyst_opinion's fetch_investor_estimate ("estimate is a bonus,
# never fail the whole tool for it"). Both fetches are awaited unguarded
# AFTER the analysts and the Bull/Bear debate have run (pipeline.py:1883),
# so one EGW00201 on a decorative TR used to discard a completed multi-agent
# analysis.
# ---------------------------------------------------------------------------


async def test_short_sale_fetch_failure_degrades_to_no_data_message(monkeypatch, caplog):
    class _FailingShortSaleBroker(FakeBroker):
        async def fetch_daily_short_sale(self, *args, **kwargs):
            raise RuntimeError("EGW00201 rate limited")

    fake = _FailingShortSaleBroker(program_trade=[
        ProgramTrade(slot="151818", net_buy_qty=574310, net_buy_amount=45613179100.0)
    ])
    monkeypatch.setattr(kis_provider, "_get_broker", _returns(fake))
    # No as_of: defaults to get_clock().today(), same as
    # test_get_kr_risk_signals_converts_program_trade_amount_to_eok — that's
    # what makes the program-trade branch's `as_of is None` gate fire so the
    # program line below is actually exercised alongside the failing fetch.
    with caplog.at_level(logging.WARNING):
        result = await kis_provider.get_kr_risk_signals("005930.KS")
    # degrades to the same message an empty (but successful) response gets —
    # the pipeline keeps running instead of the whole tool call raising.
    assert "[Short-selling] no recent short-sale data" in result
    # the program-trade block (an independent fetch) is unaffected
    assert "456.1억원" in result


async def test_program_trade_fetch_failure_omits_program_line_only(monkeypatch, caplog):
    class _FailingProgramBroker(FakeBroker):
        async def fetch_program_trade(self, *args, **kwargs):
            raise RuntimeError("EGW00201 rate limited")

    # Fixed far-past day so `s.day <= end_str` holds against real "today"
    # regardless of when this test runs (no as_of override needed to
    # exercise the `as_of is None` program-trade gate — see the sibling
    # test above).
    fake = _FailingProgramBroker(daily_short_sale=[_short("20200102", 12000.0, 5000, 1.04, 3.43)])
    monkeypatch.setattr(kis_provider, "_get_broker", _returns(fake))
    with caplog.at_level(logging.WARNING):
        result = await kis_provider.get_kr_risk_signals("005930.KS")
    # the short-sale block (an independent fetch) is unaffected
    assert "1.04" in result
    # program-trade line is simply absent, not an error/crash
    assert "[Program trading]" not in result


# ---------------------------------------------------------------------------
# get_kr_financials — 2026-07-30: income statement / balance sheet / ratio
# sections used a bare "period | v1 | v2 | v3" row with column names only
# stated once in a header far above. A live reproduction (009540.KS)
# confirmed the model then swaps columns ACROSS sections — writing a
# financial-ratio table's EPS/BPS value into the income-statement
# narrative as if it were revenue/net income. Fixed by labeling every
# value inline (name=value), matching the [Valuation] section's existing
# (and never observed to be confused) convention.
# ---------------------------------------------------------------------------


async def test_get_kr_financials_labels_income_statement_values_inline(monkeypatch):
    fake = FakeBroker(income_statement=[
        IncomeStatement(period="202312", revenue=212_962, gross_profit=10_480, operating_profit=2_823, net_income=1_449),
    ])
    monkeypatch.setattr(kis_provider, "_get_broker", _returns(fake))
    result = await kis_provider.get_kr_financials("009540.KS")
    # revenue (212,962억) is at/above 1조, so a 조원 equivalent is spelled
    # out — see _eok's docstring for why (a live reproduction showed the
    # model misreads the 3-digit comma group as the 조 boundary otherwise).
    # operating_profit/net_income stay 억원-only below the 1조 threshold.
    assert "revenue=212,962억원(21.30조원)" in result
    assert "operating_profit=2,823억원" in result
    assert "operating_profit=2,823억원(" not in result
    assert "net_income=1,449억원" in result


async def test_get_kr_financials_all_zero_income_row_renders_not_reported(monkeypatch):
    # 2026-09-15, decision id=1433 / 004310.KS/현대약품: `_num` coerced every
    # missing KIS income-statement field to 0.0, and the Fundamentals
    # Analyst reported "zero revenue, zero gross profit, zero operating
    # profit, zero net income" and concluded "dormant shell company" for a
    # firm whose actual FY2025 revenue was ₩191.8B. This pins the fix: an
    # all-zero row (the exact shape KIS returned for that ticker) must
    # render every item as "not reported", with no `0억원` token anywhere.
    fake = FakeBroker(income_statement=[
        IncomeStatement(period="202512", revenue=0.0, gross_profit=0.0, operating_profit=0.0, net_income=0.0),
    ])
    monkeypatch.setattr(kis_provider, "_get_broker", _returns(fake))
    result = await kis_provider.get_kr_financials("004310.KS")
    assert "revenue=not reported" in result
    assert "gross_profit=not reported" in result
    assert "operating_profit=not reported" in result
    assert "net_income=not reported" in result
    assert "0억원" not in result


async def test_get_kr_financials_all_zero_balance_row_renders_not_reported(monkeypatch):
    fake = FakeBroker(balance_sheet=[
        BalanceSheet(
            period="202512", total_assets=0.0, total_liabilities=0.0,
            total_equity=0.0, current_assets=0.0, current_liabilities=0.0,
        ),
    ])
    monkeypatch.setattr(kis_provider, "_get_broker", _returns(fake))
    result = await kis_provider.get_kr_financials("004310.KS")
    assert "total_assets=not reported" in result
    assert "total_liabilities=not reported" in result
    assert "total_equity=not reported" in result
    assert "0억원" not in result


async def test_get_kr_financials_genuine_zero_revenue_spac_row_still_renders(monkeypatch):
    # Row-gating control (SPAC / pre-revenue biotech shape): revenue=0.0 is a
    # GENUINE reported zero here because operating_profit/net_income are
    # real non-zero values (listing/admin costs), not all-zero like the
    # 004310.KS incident row. `revenue` must still render as a real _eok
    # value, never "not reported" — this is the test that pins the
    # row-gating rule (revenue is exempt only when its siblings are ALSO
    # all exactly 0.0).
    fake = FakeBroker(income_statement=[
        IncomeStatement(period="202512", revenue=0.0, gross_profit=0.0, operating_profit=-50.0, net_income=-48.0),
    ])
    monkeypatch.setattr(kis_provider, "_get_broker", _returns(fake))
    result = await kis_provider.get_kr_financials("999999.KS")
    assert "revenue=0억원 (= 0.0 billion KRW)" in result
    assert "revenue=not reported" not in result
    assert "operating_profit=-50억원" in result
    assert "net_income=-48억원" in result


async def test_get_kr_financials_reported_revenue_with_zero_siblings_still_renders(monkeypatch):
    # Pins the "never hide a reported revenue figure" half of the rule: KIS
    # supplied a real revenue but the three sibling fields all came back as
    # the missing-coerced 0.0. Before this fix the row-gate looked only at
    # the siblings, so this shape was wrongly rendered as "not reported"
    # even though revenue itself was genuinely reported.
    fake = FakeBroker(income_statement=[
        IncomeStatement(period="202512", revenue=1_000.0, gross_profit=0.0, operating_profit=0.0, net_income=0.0),
    ])
    monkeypatch.setattr(kis_provider, "_get_broker", _returns(fake))
    result = await kis_provider.get_kr_financials("999999.KS")
    assert "revenue=1,000억원" in result
    assert "revenue=not reported" not in result


async def test_get_kr_financials_market_cap_zero_renders_not_reported(monkeypatch):
    fake = FakeBroker(valuation=StockValuation(
        code="004310", price=10_000, per=8.0, pbr=0.9, eps=1_200, bps=11_000, market_cap=0.0,
    ))
    monkeypatch.setattr(kis_provider, "_get_broker", _returns(fake))
    result = await kis_provider.get_kr_financials("004310.KS")
    assert "Market cap not reported" in result


async def test_get_kr_financials_balance_sheet_section_unchanged_by_unit_key_extraction(monkeypatch):
    # 2026-09-11: the [Balance sheet] section's unit-key prose was extracted
    # to the module-level `_EOK_TABLE_UNIT_KEY` constant (shared with
    # `_kis_balance_sheet` and `get_kr_earnings_estimate`). This pins the
    # exact resulting text so a future edit to the shared constant can't
    # silently change this section's wording without a test failure.
    fake = FakeBroker(balance_sheet=[
        BalanceSheet(
            period="202312", total_assets=100_000, total_liabilities=40_000,
            total_equity=60_000, current_assets=50_000, current_liabilities=20_000,
        ),
    ])
    monkeypatch.setattr(kis_provider, "_get_broker", _returns(fake))
    result = await kis_provider.get_kr_financials("009540.KS")
    expected_header = (
        "\n[Balance sheet] (억원, with billion-KRW and trillion-KRW equivalents "
        "already computed and given in parentheses — quote one of the supplied "
        "numbers as-is, never re-derive either yourself. Unit key: 1억원 = 0.1 "
        "billion KRW = 0.0001 trillion KRW — do NOT write an 억원 figure's digits "
        "followed by \"billion\")\n"
    )
    assert expected_header in result
    assert "total_assets=100,000억원(10.00조원)" in result
    assert "total_liabilities=40,000억원(4.00조원)" in result
    assert "total_equity=60,000억원(6.00조원)" in result


async def test_get_kr_financials_market_cap_gets_jo_equivalent_above_threshold(monkeypatch):
    fake = FakeBroker(valuation=StockValuation(
        code="009540", price=372_500, per=12.03, pbr=1.96, eps=30_639, bps=187_883, market_cap=260_799,
    ))
    monkeypatch.setattr(kis_provider, "_get_broker", _returns(fake))
    result = await kis_provider.get_kr_financials("009540.KS")
    assert "Market cap 260,799억원(26.08조원)" in result


async def test_get_kr_financials_labels_ratio_values_inline_with_correct_units(monkeypatch):
    fake = FakeBroker(financial_ratio=[
        FinancialRatio(
            period="202312", revenue_growth=23.09, op_profit_growth=0.0, net_income_growth=0.0,
            roe=2.26, eps=3_133, bps=140_052, debt_ratio=160.65,
        ),
    ])
    monkeypatch.setattr(kis_provider, "_get_broker", _returns(fake))
    result = await kis_provider.get_kr_financials("009540.KS")
    assert "ROE=2.26%" in result
    assert "EPS=3,133원" in result
    assert "BPS=140,052원" in result
    assert "debt_to_equity_ratio=160.65% (부채비율 = liabilities÷EQUITY)" in result
    # 2026-09-01 fix (item #4): the bare `debt_ratio=` key form must be gone
    # entirely — it's what let the model read this as an assets-denominated
    # "debt ratio" instead of KIS's equity-denominated 부채비율.
    assert "debt_ratio=" not in result
    # The exact bug reproduced live: these values must never appear
    # unlabeled/bare where they could be read as an income-statement or
    # balance-sheet figure instead.
    assert "| 3,133 |" not in result
    assert "| 140,052 |" not in result


async def test_get_kr_financials_ratio_header_explains_the_debt_ratio_denominator(monkeypatch):
    # 2026-09-01 real-account incident, 066970.KS: the Fundamentals report
    # described KIS's 부채비율 (a >100%-capable equity-denominated ratio) as
    # "372.75% (Liabilities/Assets)", and the Portfolio Manager repeated it
    # downstream as "a 372.75% debt-to-assets ratio" (decision id=1121) — a
    # liabilities/assets ratio cannot exceed 100% by definition. The header
    # must ground the model against exactly this misreading.
    fake = FakeBroker(financial_ratio=[
        FinancialRatio(
            period="202606", revenue_growth=0.0, op_profit_growth=0.0, net_income_growth=0.0,
            roe=-42.1, eps=-1200, bps=2850, debt_ratio=372.75,
        ),
    ])
    monkeypatch.setattr(kis_provider, "_get_broker", _returns(fake))
    result = await kis_provider.get_kr_financials("066970.KS")
    assert "total EQUITY" in result
    assert "ABOVE 100%" in result and "normal" in result
    # The header must explicitly forbid the assets-denominated gloss — it
    # necessarily names the banned phrases once, inside the prohibition
    # itself ('Never describe this field as "debt-to-assets"...').
    assert 'never describe this field as "debt-to-assets"' in result.lower()
    assert '"liabilities/assets"' in result.lower()
    assert "borrowing_dependency" in result


async def test_get_kr_financials_includes_profitability_ratio_section(monkeypatch):
    from app.domain.models.market import ProfitabilityRatio

    fake = FakeBroker(profitability_ratio=[
        ProfitabilityRatio(
            period="202603", return_on_capital=14.65, return_on_equity=19.16,
            net_margin=35.28, gross_margin=61.19,
        ),
    ])
    monkeypatch.setattr(kis_provider, "_get_broker", _returns(fake))
    result = await kis_provider.get_kr_financials("009540.KS")
    assert "[Profitability ratios]" in result
    assert "gross_margin=61.19%" in result
    assert "net_margin=35.28%" in result
    assert "return_on_capital=14.65%" in result
    assert "return_on_equity=19.16%" in result
    # return_on_capital (총자본순이익율) must not be confused with the Western
    # ROIC — see the header's 2026-09-01 clarification.
    assert "NOT the Western" in result


async def test_get_kr_financials_includes_stability_ratio_section(monkeypatch):
    from app.domain.models.market import StabilityRatio

    fake = FakeBroker(stability_ratio=[
        StabilityRatio(period="202603", borrowing_dependency=4.44, current_ratio=253.91, quick_ratio=0.0),
    ])
    monkeypatch.setattr(kis_provider, "_get_broker", _returns(fake))
    result = await kis_provider.get_kr_financials("009540.KS")
    assert "[Stability ratios]" in result
    assert "current_ratio=253.91%" in result
    assert "borrowing_dependency=4.44%" in result
    # borrowing_dependency is the ASSETS-denominated leverage metric — see
    # the header's 2026-09-01 clarification distinguishing it from
    # debt_to_equity_ratio.
    assert "TOTAL ASSETS" in result


async def test_get_kr_financials_stability_ratio_zero_renders_as_not_reported(monkeypatch):
    from app.domain.models.market import StabilityRatio

    fake = FakeBroker(stability_ratio=[
        StabilityRatio(period="202606", borrowing_dependency=0.0, current_ratio=89.49, quick_ratio=0.0),
    ])
    monkeypatch.setattr(kis_provider, "_get_broker", _returns(fake))
    result = await kis_provider.get_kr_financials("002990.KS")
    # current_ratio is a real, present value — rendered normally
    assert "current_ratio=89.49%" in result
    # quick_ratio was omitted by KIS (coerced to 0.0 by _num) — a listed
    # going concern cannot have an exact-zero quick ratio, so this must not
    # render as a literal "quick_ratio=0.00%"
    assert "quick_ratio=not reported" in result
    assert "quick_ratio=0.00%" not in result
    # borrowing_dependency=0.0 is left alone — a genuinely debt-free company
    # can report a real zero there
    assert "borrowing_dependency=0.00%" in result


async def test_get_kr_financials_borrowing_dependency_zero_is_a_real_value_not_missing(monkeypatch):
    from app.domain.models.market import StabilityRatio

    # All three [Stability ratios] fields at exactly 0.0 — same coerced-zero
    # shape `_num` produces whether KIS omitted the field or the company
    # genuinely reports zero. current_ratio/quick_ratio must still go
    # through `_pct_or_missing` (a listed going concern cannot have an
    # exact-zero current/quick ratio, so 0.0 there means "KIS omitted it").
    # borrowing_dependency is deliberately exempt from that helper: a
    # genuinely debt-free company (no interest-bearing liabilities) really
    # can report borrowing_dependency=0.00%, so collapsing it to "not
    # reported" would erase a true, meaningful reading. If a future refactor
    # sweeps borrowing_dependency into `_pct_or_missing` for consistency,
    # this test must fail.
    fake = FakeBroker(stability_ratio=[
        StabilityRatio(period="202606", borrowing_dependency=0.0, current_ratio=0.0, quick_ratio=0.0),
    ])
    monkeypatch.setattr(kis_provider, "_get_broker", _returns(fake))
    result = await kis_provider.get_kr_financials("002990.KS")
    assert "current_ratio=not reported" in result
    assert "quick_ratio=not reported" in result
    assert "borrowing_dependency=0.00%" in result
    assert "borrowing_dependency=not reported" not in result


async def test_get_kr_financials_includes_growth_ratio_detail_section(monkeypatch):
    from app.domain.models.market import GrowthRatioDetail

    fake = FakeBroker(growth_ratio_detail=[
        GrowthRatioDetail(period="202603", equity_growth=19.68, asset_growth=22.65),
    ])
    monkeypatch.setattr(kis_provider, "_get_broker", _returns(fake))
    result = await kis_provider.get_kr_financials("009540.KS")
    assert "[Growth ratios, additional]" in result
    assert "equity_growth=19.68%" in result
    assert "asset_growth=22.65%" in result


async def test_get_kr_financials_ratio_zero_fields_render_as_not_reported(monkeypatch):
    # 2026-09-10 widened scope of the `_pct_or_missing` fix (originally
    # [Stability ratios] only, see the 002990.KS/금호건설 id=785 incident
    # above) to [Financial ratios]: a missing KIS field coerced to 0.0 by
    # `_num` must not render as a confident "0.00%"/"0원" here either — a
    # listed going concern cannot have an exact-zero ROE/EPS/BPS/부채비율.
    fake = FakeBroker(financial_ratio=[
        FinancialRatio(
            period="202606", revenue_growth=0.0, op_profit_growth=0.0, net_income_growth=0.0,
            roe=0.0, eps=0.0, bps=0.0, debt_ratio=0.0,
        ),
    ])
    monkeypatch.setattr(kis_provider, "_get_broker", _returns(fake))
    result = await kis_provider.get_kr_financials("009540.KS")
    assert "ROE=not reported" in result
    assert "EPS=not reported" in result
    assert "BPS=not reported" in result
    assert "debt_to_equity_ratio=not reported" in result
    assert "revenue_growth=not reported" in result
    assert "net_income_growth=not reported" in result
    assert "0.00%" not in result
    assert "0원" not in result


async def test_get_kr_financials_ratio_nonzero_fields_still_render_the_number(monkeypatch):
    # Companion to the zero-fields test above — the fix must not turn a
    # real, present value into "not reported" too.
    fake = FakeBroker(financial_ratio=[
        FinancialRatio(
            period="202606", revenue_growth=1.5, op_profit_growth=1.5, net_income_growth=-2.0,
            roe=8.4, eps=1200, bps=15000, debt_ratio=45.0,
        ),
    ])
    monkeypatch.setattr(kis_provider, "_get_broker", _returns(fake))
    result = await kis_provider.get_kr_financials("009540.KS")
    assert "ROE=8.40%" in result
    assert "EPS=1,200원" in result
    assert "BPS=15,000원" in result
    assert "debt_to_equity_ratio=45.00%" in result
    assert "revenue_growth=1.50%" in result
    assert "net_income_growth=-2.00%" in result


async def test_get_kr_financials_profitability_ratio_zero_fields_render_as_not_reported(monkeypatch):
    from app.domain.models.market import ProfitabilityRatio

    fake = FakeBroker(profitability_ratio=[
        ProfitabilityRatio(
            period="202603", return_on_capital=0.0, return_on_equity=0.0,
            net_margin=0.0, gross_margin=0.0,
        ),
    ])
    monkeypatch.setattr(kis_provider, "_get_broker", _returns(fake))
    result = await kis_provider.get_kr_financials("009540.KS")
    assert "gross_margin=not reported" in result
    assert "net_margin=not reported" in result
    assert "return_on_capital=not reported" in result
    assert "return_on_equity=not reported" in result


async def test_get_kr_financials_growth_ratio_detail_zero_fields_render_as_not_reported(monkeypatch):
    from app.domain.models.market import GrowthRatioDetail

    fake = FakeBroker(growth_ratio_detail=[
        GrowthRatioDetail(period="202603", equity_growth=0.0, asset_growth=0.0),
    ])
    monkeypatch.setattr(kis_provider, "_get_broker", _returns(fake))
    result = await kis_provider.get_kr_financials("009540.KS")
    assert "equity_growth=not reported" in result
    assert "asset_growth=not reported" in result


async def test_get_kr_financials_includes_other_major_ratios_section(monkeypatch):
    from app.domain.models.market import OtherMajorRatio

    fake = FakeBroker(other_major_ratios=[
        OtherMajorRatio(period="202603", payout_ratio=0.0, eva=0.0, ebitda=695_994.0, ev_ebitda=6.88),
    ])
    monkeypatch.setattr(kis_provider, "_get_broker", _returns(fake))
    result = await kis_provider.get_kr_financials("009540.KS")
    assert "[Other key ratios]" in result
    assert "EBITDA=695,994억원" in result
    assert "EV/EBITDA=6.88x" in result
    assert "payout_ratio=0.00%" in result


async def test_kis_news_dispatches_to_overseas_news_for_us_ticker(monkeypatch):
    async def fake_overseas_news(ticker, as_of=None):
        return f"overseas news for {ticker}"

    monkeypatch.setattr(kis_provider, "get_overseas_news", fake_overseas_news)
    assert await kis_provider._kis_news("AAPL") == "overseas news for AAPL"


async def test_get_overseas_news_returns_not_supported_for_non_us_ticker():
    result = await kis_provider.get_overseas_news("005930.KS")
    assert "미국(US) 종목만" in result


async def test_get_overseas_news_returns_not_configured_message(monkeypatch):
    async def _broken_broker():
        raise RuntimeError("not configured")

    monkeypatch.setattr(kis_provider, "_get_broker", _broken_broker)
    result = await kis_provider.get_overseas_news("AAPL")
    assert "KIS 미설정" in result


async def test_get_overseas_news_formats_items(monkeypatch):
    fake = FakeBroker(news=[
        NewsTitle(day="20260722", time="154957", title="애플, 특허 분쟁 판결 유지", source="연합미국"),
    ])
    monkeypatch.setattr(kis_provider, "_get_broker", _returns(fake))
    result = await kis_provider.get_overseas_news("AAPL", as_of=date(2026, 7, 23))
    assert "애플, 특허 분쟁 판결 유지" in result and "source: KIS" in result


async def test_get_overseas_news_filters_by_as_of(monkeypatch):
    fake = FakeBroker(news=[NewsTitle(day="20260801", time="090000", title="future news", source="x")])
    monkeypatch.setattr(kis_provider, "_get_broker", _returns(fake))
    result = await kis_provider.get_overseas_news("AAPL", as_of=date(2026, 7, 23))
    assert "최근 뉴스 없음" in result


async def test_get_overseas_news_keeps_undated_item_when_live(monkeypatch):
    # 2026-09-11: undated (day="") headline is always the TR's newest item —
    # a live query (as_of=None) must keep it, same as before this fix.
    fake = FakeBroker(news=[NewsTitle(day="", time="", title="breaking now", source="x")])
    monkeypatch.setattr(kis_provider, "_get_broker", _returns(fake))
    result = await kis_provider.get_overseas_news("AAPL")
    assert "breaking now" in result


async def test_get_overseas_news_drops_undated_item_when_backtesting(monkeypatch):
    # 2026-09-11: same undated item, but as_of is strictly in the past — the
    # undated headline must be dropped so it can't leak future news into a
    # historical backtest prompt.
    set_clock(FakeClock(datetime(2026, 8, 26, 9, 42, tzinfo=_US_TZ)))
    fake = FakeBroker(news=[NewsTitle(day="", time="", title="breaking now", source="x")])
    monkeypatch.setattr(kis_provider, "_get_broker", _returns(fake))
    result = await kis_provider.get_overseas_news("AAPL", as_of=date(2026, 7, 23))
    assert "최근 뉴스 없음" in result


async def test_get_overseas_news_dated_item_still_filtered_by_cutoff(monkeypatch):
    # Dated-item filtering (n.day <= cutoff) is unchanged by this fix.
    fake = FakeBroker(news=[
        NewsTitle(day="20260722", time="154957", title="past news", source="x"),
        NewsTitle(day="20260801", time="090000", title="future news", source="x"),
    ])
    monkeypatch.setattr(kis_provider, "_get_broker", _returns(fake))
    result = await kis_provider.get_overseas_news("AAPL", as_of=date(2026, 7, 23))
    assert "past news" in result
    assert "future news" not in result


async def test_kis_balance_sheet_formats_recent_periods(monkeypatch):
    class _Row:
        def __init__(self, period):
            self.period = period
            self.total_assets = 100.0
            self.total_liabilities = 40.0
            self.total_equity = 60.0
            self.current_assets = 50.0
            self.current_liabilities = 20.0

    class _FakeFinanceBroker:
        async def fetch_kr_balance_sheet(self, code):
            return [_Row("202603"), _Row("202512")]

    monkeypatch.setattr(kis_provider, "_get_broker", _returns(_FakeFinanceBroker()))
    result = await kis_provider._kis_balance_sheet("005930.KS", as_of=date(2026, 7, 23))
    assert "202603" in result and "202512" in result
    assert "source: KIS" in result


async def test_kis_balance_sheet_raises_when_no_rows(monkeypatch):
    class _FakeFinanceBroker:
        async def fetch_kr_balance_sheet(self, code):
            return []

    monkeypatch.setattr(kis_provider, "_get_broker", _returns(_FakeFinanceBroker()))
    with pytest.raises(RuntimeError, match="no balance sheet rows"):
        await kis_provider._kis_balance_sheet("005930.KS")


async def test_kis_balance_sheet_raises_for_non_krx_ticker(monkeypatch):
    def should_not_be_called():
        raise AssertionError("KIS broker should not be looked up for a non-KRX ticker")

    monkeypatch.setattr(kis_provider, "_get_broker", should_not_be_called)
    with pytest.raises(ValueError, match="is not a KRX ticker"):
        await kis_provider._kis_balance_sheet("AAPL")


async def test_kis_balance_sheet_carries_unit_key_and_billion_equivalents(monkeypatch):
    # 2026-09-11 dataflows audit stage 1 (A): this used to render a bare
    # fixed-width table with no unit token at all. Now every field goes
    # through `_eok`, so each one must carry a billion-KRW equivalent and
    # the shared unit-key prose must appear once.
    class _Row:
        def __init__(self, period):
            self.period = period
            self.total_assets = 100.0
            self.total_liabilities = 40.0
            self.total_equity = 60.0
            self.current_assets = 50.0
            self.current_liabilities = 20.0

    class _FakeFinanceBroker:
        async def fetch_kr_balance_sheet(self, code):
            return [_Row("202603")]

    monkeypatch.setattr(kis_provider, "_get_broker", _returns(_FakeFinanceBroker()))
    result = await kis_provider._kis_balance_sheet("005930.KS", as_of=date(2026, 7, 23))
    assert "Unit key: 1억원 = 0.1 billion KRW" in result
    assert "total_assets=100억원 (= 10.0 billion KRW)" in result
    assert "total_liabilities=40억원 (= 4.0 billion KRW)" in result
    assert "total_equity=60억원 (= 6.0 billion KRW)" in result
    assert "current_assets=50억원 (= 5.0 billion KRW)" in result
    assert "current_liabilities=20억원 (= 2.0 billion KRW)" in result


async def test_kis_balance_sheet_renders_jo_branch_above_threshold(monkeypatch):
    # A value >= 10,000억 (1조) must go through _eok's >=1조 branch, which
    # additionally spells out the 조원/trillion-KRW equivalent.
    class _Row:
        def __init__(self, period):
            self.period = period
            self.total_assets = 39_045.0  # 3.90조원
            self.total_liabilities = 15_000.0
            self.total_equity = 24_045.0
            self.current_assets = 12_000.0
            self.current_liabilities = 8_000.0

    class _FakeFinanceBroker:
        async def fetch_kr_balance_sheet(self, code):
            return [_Row("202603")]

    monkeypatch.setattr(kis_provider, "_get_broker", _returns(_FakeFinanceBroker()))
    result = await kis_provider._kis_balance_sheet("005930.KS", as_of=date(2026, 7, 23))
    assert "total_assets=39,045억원(3.90조원)" in result
    assert "3,904.5 billion KRW" in result and "3.90 trillion KRW" in result


async def test_kis_balance_sheet_all_zero_row_renders_not_reported(monkeypatch):
    # Sibling of the `get_kr_financials` all-zero-balance-row regression
    # test (id=1433 / 004310.KS shape) for this narrower balance-sheet-only
    # view, which shares the same `_eok_or_missing` helper.
    class _Row:
        def __init__(self, period):
            self.period = period
            self.total_assets = 0.0
            self.total_liabilities = 0.0
            self.total_equity = 0.0
            self.current_assets = 0.0
            self.current_liabilities = 0.0

    class _FakeFinanceBroker:
        async def fetch_kr_balance_sheet(self, code):
            return [_Row("202603")]

    monkeypatch.setattr(kis_provider, "_get_broker", _returns(_FakeFinanceBroker()))
    result = await kis_provider._kis_balance_sheet("005930.KS", as_of=date(2026, 7, 23))
    assert "total_assets=not reported" in result
    assert "total_liabilities=not reported" in result
    assert "total_equity=not reported" in result
    assert "current_assets=not reported" in result
    assert "current_liabilities=not reported" in result
    assert "0억원" not in result


async def test_kis_indicators_raises_for_unsupported_ticker(monkeypatch):
    def should_not_be_called():
        raise AssertionError("KIS broker should not be looked up for a ticker matching neither market")

    monkeypatch.setattr(kis_provider, "_get_broker", should_not_be_called)
    with pytest.raises(ValueError, match="is not a KRX or US ticker"):
        await kis_provider._kis_indicators("0700.HK")


async def test_kis_indicators_computes_sma_rsi_macd_for_us_ticker(monkeypatch):
    as_of = date(2026, 7, 23)
    candles = [_candle((as_of - timedelta(days=i)).strftime("%Y%m%d"), 100.0 + i) for i in range(60, 0, -1)]
    fake = FakeBroker(overseas_candles=candles)
    monkeypatch.setattr(kis_provider, "_get_broker", _returns(fake))
    result = await kis_provider._kis_indicators("AAPL", as_of=as_of)
    assert "source: KIS" in result
    assert "SMA(20)=" in result and "SMA(50)=" in result
    assert "signal=" in result and "hist=" in result


async def test_kis_indicators_raises_when_too_few_candles(monkeypatch):
    fake = FakeBroker(candles=[_candle("2026%02d01" % m, 100.0 + m) for m in range(1, 3)])
    monkeypatch.setattr(kis_provider, "_get_broker", _returns(fake))
    with pytest.raises(RuntimeError, match="too few candles"):
        await kis_provider._kis_indicators("005930.KS")


async def test_kis_indicators_partial_block_below_51_candles(monkeypatch):
    """2026-09-08 fix (id=1315, 387690.KS): a candle count that supports
    RSI(14)/SMA(20) but not SMA(50)/MACD must render the ones it can,
    'n/a' for the rest -- not raise and discard everything the way this
    used to below a flat 51-candle floor."""
    candles = [_candle(f"202601{d:02d}" if d <= 31 else f"202602{d - 31:02d}", 100.0 + d) for d in range(1, 21)]
    fake = FakeBroker(candles=candles)
    monkeypatch.setattr(kis_provider, "_get_broker", _returns(fake))
    result = await kis_provider._kis_indicators("005930.KS")
    assert "source: KIS" in result
    assert "SMA(20)=" in result and "n/a" not in result.split("SMA(20)=")[1].split()[0]
    assert "RSI(14)=" in result and "n/a" not in result.split("RSI(14)=")[1].split()[0]
    assert "SMA(50)=n/a" in result
    assert "MACD=n/a" in result


async def test_kis_indicators_still_raises_below_rsi_floor(monkeypatch):
    """14 candles: one short of RSI(14)'s own 15-candle minimum -- nothing
    could render at all, so this must still raise (unchanged floor case,
    distinct from the old flat-51 threshold this replaces)."""
    candles = [_candle(f"202601{d:02d}", 100.0 + d) for d in range(1, 15)]
    fake = FakeBroker(candles=candles)
    monkeypatch.setattr(kis_provider, "_get_broker", _returns(fake))
    with pytest.raises(RuntimeError, match="too few candles"):
        await kis_provider._kis_indicators("005930.KS")


async def test_kis_indicators_computes_sma_rsi_macd(monkeypatch):
    candles = [_candle(f"202601{d:02d}" if d <= 31 else f"202602{d - 31:02d}", 100.0 + d) for d in range(1, 61)]
    fake = FakeBroker(candles=candles)
    monkeypatch.setattr(kis_provider, "_get_broker", _returns(fake))
    result = await kis_provider._kis_indicators("005930.KS")
    assert "source: KIS" in result
    assert "SMA(20)=" in result and "SMA(50)=" in result and "RSI(14)=" in result and "MACD=" in result
    assert "signal=" in result and "hist=" in result


async def test_kis_verified_snapshot_raises_for_unsupported_ticker(monkeypatch):
    def should_not_be_called():
        raise AssertionError("KIS broker should not be looked up for a ticker matching neither market")

    monkeypatch.setattr(kis_provider, "_get_broker", should_not_be_called)
    with pytest.raises(ValueError, match="is not a KRX or US ticker"):
        await kis_provider._kis_verified_snapshot("0700.HK")


async def test_kis_verified_snapshot_uses_kis_for_us_ticker(monkeypatch):
    fake = FakeBroker(overseas_candles=[_candle("20260722", 200.0, high=210.0, low=195.0)])
    monkeypatch.setattr(kis_provider, "_get_broker", _returns(fake))
    result = await kis_provider._kis_verified_snapshot("AAPL", as_of=date(2026, 7, 23))
    assert "VERIFIED SNAPSHOT for AAPL" in result and "close=200.00" in result
    assert "source: KIS" in result


async def test_kis_verified_snapshot_labels_in_progress_session_as_live(monkeypatch):
    # Regression test for the MRVL incident — see
    # app.dataflows.session_state's module docstring. _kis_verified_snapshot
    # is the vendor router's FIRST-choice provider for verified_snapshot, so
    # this fix mattered more here than on the yfinance fallback.
    set_clock(FakeClock(datetime(2026, 8, 26, 9, 42, tzinfo=_US_TZ)))
    fake = FakeBroker(overseas_candles=[_candle("20260826", 237.04, high=240.77, low=236.79)])
    monkeypatch.setattr(kis_provider, "_get_broker", _returns(fake))
    result = await kis_provider._kis_verified_snapshot("MRVL")
    assert "SESSION IN PROGRESS" in result
    assert "last=237.04" in result
    assert "close=" not in result
    assert "source: KIS" in result


async def test_kis_verified_snapshot_labels_closed_session_as_close(monkeypatch):
    # Same calendar day, but the clock reads after the US close.
    set_clock(FakeClock(datetime(2026, 8, 26, 16, 30, tzinfo=_US_TZ)))
    fake = FakeBroker(overseas_candles=[_candle("20260826", 245.11, high=245.17, low=236.79)])
    monkeypatch.setattr(kis_provider, "_get_broker", _returns(fake))
    result = await kis_provider._kis_verified_snapshot("MRVL")
    assert "SESSION IN PROGRESS" not in result
    assert "close=245.11" in result
    assert "source: KIS" in result


async def test_kis_stock_data_raises_for_unsupported_ticker(monkeypatch):
    def should_not_be_called():
        raise AssertionError("KIS broker should not be looked up for a ticker matching neither market")

    monkeypatch.setattr(kis_provider, "_get_broker", should_not_be_called)
    with pytest.raises(ValueError, match="is not a KRX or US ticker"):
        await kis_provider._kis_stock_data("0700.HK")


async def test_kis_stock_data_uses_kis_for_us_ticker(monkeypatch):
    fake = FakeBroker(overseas_candles=[_candle("20260722", 200.0), _candle("20260723", 205.0)])
    monkeypatch.setattr(kis_provider, "_get_broker", _returns(fake))
    result = await kis_provider._kis_stock_data("AAPL", as_of=date(2026, 7, 23))
    assert "source: KIS" in result and "AAPL" in result


async def test_kis_stock_data_appends_footnote_when_session_in_progress(monkeypatch):
    set_clock(FakeClock(datetime(2026, 8, 26, 9, 42, tzinfo=_US_TZ)))
    fake = FakeBroker(overseas_candles=[_candle("20260824", 240.0), _candle("20260826", 237.04)])
    monkeypatch.setattr(kis_provider, "_get_broker", _returns(fake))
    result = await kis_provider._kis_stock_data("MRVL")
    assert "STILL-OPEN" in result


async def test_kis_stock_data_no_footnote_when_session_closed(monkeypatch):
    set_clock(FakeClock(datetime(2026, 8, 26, 16, 30, tzinfo=_US_TZ)))
    fake = FakeBroker(overseas_candles=[_candle("20260824", 240.0), _candle("20260826", 245.11)])
    monkeypatch.setattr(kis_provider, "_get_broker", _returns(fake))
    result = await kis_provider._kis_stock_data("MRVL")
    assert "STILL-OPEN" not in result


def _weekly_candles(n: int, base: float = 100.0) -> list[Candle]:
    """`n` weekly bars, oldest to newest, with a mild upward drift so
    52-week-high/low and trailing-return math has something to distinguish."""
    start = date(2025, 9, 1)
    return [
        _candle((start + timedelta(weeks=i)).strftime("%Y%m%d"), base + i, high=base + i + 1, low=base + i - 1)
        for i in range(n)
    ]


async def test_kis_long_horizon_raises_for_unsupported_ticker(monkeypatch):
    def should_not_be_called():
        raise AssertionError("KIS broker should not be looked up for a ticker matching neither market")

    monkeypatch.setattr(kis_provider, "_get_broker", should_not_be_called)
    with pytest.raises(ValueError, match="is not a KRX or US ticker"):
        await kis_provider._kis_long_horizon("0700.HK")


async def test_kis_long_horizon_uses_weekly_candles_for_krx_ticker(monkeypatch):
    fake = FakeBroker(weekly_candles=_weekly_candles(53))
    monkeypatch.setattr(kis_provider, "_get_broker", _returns(fake))
    result = await kis_provider._kis_long_horizon("005930.KS", as_of=date(2026, 9, 15))
    assert "Longer-horizon price context for 005930.KS (53 weekly bars, source: KIS):" in result
    assert "52-week high:" in result and "52-week low:" in result
    assert "trailing returns from weekly closes:" in result


async def test_kis_long_horizon_uses_weekly_candles_for_us_ticker(monkeypatch):
    fake = FakeBroker(weekly_overseas_candles=_weekly_candles(53))
    monkeypatch.setattr(kis_provider, "_get_broker", _returns(fake))
    result = await kis_provider._kis_long_horizon("AAPL", as_of=date(2026, 9, 15))
    assert "Longer-horizon price context for AAPL (53 weekly bars, source: KIS):" in result


async def test_kis_long_horizon_raises_below_minimum_bars(monkeypatch):
    fake = FakeBroker(weekly_candles=_weekly_candles(5))
    monkeypatch.setattr(kis_provider, "_get_broker", _returns(fake))
    with pytest.raises(ValueError, match="need at least 12"):
        await kis_provider._kis_long_horizon("005930.KS", as_of=date(2026, 9, 15))


async def test_kis_long_horizon_raises_when_broker_returns_nothing(monkeypatch):
    fake = FakeBroker(weekly_candles=[])
    monkeypatch.setattr(kis_provider, "_get_broker", _returns(fake))
    with pytest.raises(RuntimeError, match="no weekly candles"):
        await kis_provider._kis_long_horizon("005930.KS", as_of=date(2026, 9, 15))


async def test_get_kr_earnings_estimate_returns_not_supported_for_non_krx():
    result = await kis_provider.get_kr_earnings_estimate("AAPL")
    assert "한국(KRX) 종목만" in result


async def test_get_kr_earnings_estimate_skips_for_backtest_date(monkeypatch):
    result = await kis_provider.get_kr_earnings_estimate("005930.KS", as_of=date(2020, 1, 1))
    assert "백테스트" in result


async def test_get_kr_earnings_estimate_formats_rows(monkeypatch):
    class _Row:
        def __init__(self, period):
            self.period = period
            self.revenue = 2589355.0
            self.revenue_growth = -1.4
            self.operating_profit = 65670.0
            self.operating_profit_growth = -8.5
            self.net_income = 144734.0
            self.net_income_growth = -7.4

    class _FakeFinanceBroker:
        async def fetch_kr_earnings_estimate(self, code):
            return [_Row("2023.12"), _Row("2024.12E")]

    monkeypatch.setattr(kis_provider, "_get_broker", _returns(_FakeFinanceBroker()))
    result = await kis_provider.get_kr_earnings_estimate("005930.KS")
    assert "2023.12" in result and "2024.12E" in result and "source: KIS" in result


async def test_get_kr_earnings_estimate_reports_no_data(monkeypatch):
    class _FakeFinanceBroker:
        async def fetch_kr_earnings_estimate(self, code):
            return []

    monkeypatch.setattr(kis_provider, "_get_broker", _returns(_FakeFinanceBroker()))
    result = await kis_provider.get_kr_earnings_estimate("005930.KS")
    assert "데이터 없음" in result


async def test_get_kr_earnings_estimate_money_columns_get_equivalents_growth_columns_stay_plain(monkeypatch):
    # 2026-09-11 dataflows audit stage 1 (A): the old caption said "억원
    # 단위" but supplied no equivalents — a milder version of the
    # _kis_balance_sheet bug. Only revenue/operating_profit/net_income are
    # 억원 money and must go through _eok; the three *_growth columns are
    # already percentages and must NOT be wrapped in _eok's output.
    class _Row:
        def __init__(self, period):
            self.period = period
            self.revenue = 2589355.0
            self.revenue_growth = 12.34
            self.operating_profit = 65670.0
            self.operating_profit_growth = -8.5
            self.net_income = 144734.0
            self.net_income_growth = -7.4

    class _FakeFinanceBroker:
        async def fetch_kr_earnings_estimate(self, code):
            return [_Row("2024.12E")]

    monkeypatch.setattr(kis_provider, "_get_broker", _returns(_FakeFinanceBroker()))
    result = await kis_provider.get_kr_earnings_estimate("005930.KS")
    assert "Unit key: 1억원 = 0.1 billion KRW" in result
    assert "revenue=2,589,355억원" in result and "billion KRW" in result
    assert "operating_profit=65,670억원" in result
    assert "net_income=144,734억원" in result
    # growth columns: plain percentages, never routed through _eok
    assert "revenue_growth=12.34%" in result
    assert "operating_profit_growth=-8.50%" in result
    assert "net_income_growth=-7.40%" in result


async def test_get_kr_earnings_estimate_all_zero_row_renders_not_reported(monkeypatch):
    # Sibling of the id=1433 / 004310.KS `get_kr_financials` regression test
    # for the consensus-estimate row shape, which has no `gross_profit`
    # field — `_estimate_revenue_or_missing` gates on operating_profit/
    # net_income only.
    class _Row:
        def __init__(self, period):
            self.period = period
            self.revenue = 0.0
            self.revenue_growth = 0.0
            self.operating_profit = 0.0
            self.operating_profit_growth = 0.0
            self.net_income = 0.0
            self.net_income_growth = 0.0

    class _FakeFinanceBroker:
        async def fetch_kr_earnings_estimate(self, code):
            return [_Row("2025.12E")]

    monkeypatch.setattr(kis_provider, "_get_broker", _returns(_FakeFinanceBroker()))
    result = await kis_provider.get_kr_earnings_estimate("005930.KS")
    assert "revenue=not reported" in result
    assert "operating_profit=not reported" in result
    assert "net_income=not reported" in result
    assert "0억원" not in result


async def test_get_kr_earnings_estimate_genuine_zero_revenue_still_renders(monkeypatch):
    # Row-gating control for the consensus-estimate row: revenue=0.0 with a
    # real non-zero operating_profit/net_income (a pre-revenue-year
    # estimate) is a genuine zero, not a missing row — must render through
    # the normal _eok, never "not reported".
    class _Row:
        def __init__(self, period):
            self.period = period
            self.revenue = 0.0
            self.revenue_growth = 0.0
            self.operating_profit = -120.0
            self.operating_profit_growth = -5.0
            self.net_income = -110.0
            self.net_income_growth = -4.0

    class _FakeFinanceBroker:
        async def fetch_kr_earnings_estimate(self, code):
            return [_Row("2025.12E")]

    monkeypatch.setattr(kis_provider, "_get_broker", _returns(_FakeFinanceBroker()))
    result = await kis_provider.get_kr_earnings_estimate("005930.KS")
    assert "revenue=0억원 (= 0.0 billion KRW)" in result
    assert "revenue=not reported" not in result
    assert "operating_profit=-120억원" in result
    assert "net_income=-110억원" in result
    assert "revenue_growth=12.34%(" not in result  # not _eok-wrapped
    assert "'E' suffix = estimated period" in result


async def test_get_kr_earnings_estimate_reported_revenue_with_zero_siblings_still_renders(monkeypatch):
    # Sibling of the get_kr_financials pin above, for the consensus-estimate
    # row shape (no gross_profit field): a real reported revenue with both
    # siblings at the missing-coerced 0.0 must still render through _eok,
    # never be hidden behind "not reported".
    class _Row:
        def __init__(self, period):
            self.period = period
            self.revenue = 1_000.0
            self.revenue_growth = 0.0
            self.operating_profit = 0.0
            self.operating_profit_growth = 0.0
            self.net_income = 0.0
            self.net_income_growth = 0.0

    class _FakeFinanceBroker:
        async def fetch_kr_earnings_estimate(self, code):
            return [_Row("2025.12E")]

    monkeypatch.setattr(kis_provider, "_get_broker", _returns(_FakeFinanceBroker()))
    result = await kis_provider.get_kr_earnings_estimate("005930.KS")
    assert "revenue=1,000억원" in result
    assert "revenue=not reported" not in result


async def test_cashflow_insider_ownership_categories_are_yfinance_only():
    # No KIS wrapper is registered for cashflow/insider_data/
    # institutional_ownership (KIS has no equivalent TR at all) — both
    # domestic and overseas tickers go straight to yfinance's own
    # registration from app.dataflows.yfinance.research.
    from app.dataflows.router import router
    from app.dataflows.yfinance.research import _yf_cashflow, _yf_insider_txns, _yf_institutional_ownership

    assert router._categories["cashflow"].providers == [_yf_cashflow]
    assert router._categories["insider_data"].providers == [_yf_insider_txns]
    assert router._categories["institutional_ownership"].providers == [_yf_institutional_ownership]


async def test_earnings_estimates_category_prefers_kis_for_domestic_ticker(monkeypatch):
    from app.dataflows.router import Category, router

    async def fake_kis(ticker, as_of=None):
        return f"KIS earnings estimates for {ticker}"

    async def yf_should_not_be_called(*args, **kwargs):
        raise AssertionError("yfinance should not be reached when KIS succeeds")

    monkeypatch.setitem(
        router._categories,
        "earnings_estimates",
        Category("earnings_estimates", [fake_kis, yf_should_not_be_called], True),
    )
    result = await router.call("earnings_estimates", ticker="005930.KS", as_of=None)
    assert result == "KIS earnings estimates for 005930.KS"


async def test_earnings_estimates_category_falls_back_to_yfinance_for_overseas_ticker(monkeypatch):
    from app.dataflows.router import Category, router

    async def fake_yf(ticker, as_of=None):
        return f"yfinance earnings estimates for {ticker}"

    monkeypatch.setitem(
        router._categories,
        "earnings_estimates",
        Category("earnings_estimates", [kis_provider._kis_earnings_estimates, fake_yf], True),
    )
    result = await router.call("earnings_estimates", ticker="AAPL", as_of=None)
    assert result == "yfinance earnings estimates for AAPL"


async def test_technical_indicators_category_prefers_kis_for_domestic_ticker(monkeypatch):
    from app.dataflows.router import Category, router

    async def fake_kis_indicators(ticker, as_of=None):
        return f"KIS indicators for {ticker}"

    async def yf_should_not_be_called(*args, **kwargs):
        raise AssertionError("yfinance should not be reached when KIS succeeds")

    monkeypatch.setitem(
        router._categories,
        "technical_indicators",
        Category("technical_indicators", [fake_kis_indicators, yf_should_not_be_called], False),
    )
    result = await router.call("technical_indicators", ticker="005930.KS", as_of=None)
    assert result == "KIS indicators for 005930.KS"


async def test_technical_indicators_category_falls_back_to_yfinance_for_unsupported_ticker(monkeypatch):
    from app.dataflows.router import Category, router

    async def fake_yf_indicators(ticker, as_of=None):
        return f"yfinance indicators for {ticker}"

    monkeypatch.setitem(
        router._categories,
        "technical_indicators",
        Category("technical_indicators", [kis_provider._kis_indicators, fake_yf_indicators], False),
    )
    result = await router.call("technical_indicators", ticker="0700.HK", as_of=None)
    assert result == "yfinance indicators for 0700.HK"


async def test_fundamental_data_category_prefers_kis_for_domestic_ticker(monkeypatch):
    from app.dataflows.router import Category, router

    async def fake_kis_fundamentals(ticker, as_of=None):
        return f"KIS fundamentals for {ticker}"

    async def yf_should_not_be_called(*args, **kwargs):
        raise AssertionError("yfinance should not be reached when KIS succeeds")

    # monkeypatch.setitem reverts the dict entry after the test — router is a
    # module-level singleton shared across the whole test session, so
    # mutating it via plain router.register() here would leak into every
    # later test/usage in the same process.
    monkeypatch.setitem(
        router._categories, "fundamental_data", Category("fundamental_data", [fake_kis_fundamentals, yf_should_not_be_called], False)
    )
    result = await router.call("fundamental_data", ticker="005930.KS", as_of=None)
    assert result == "KIS fundamentals for 005930.KS"


async def test_fundamental_data_category_falls_back_to_yfinance_for_overseas_ticker(monkeypatch):
    from app.dataflows.router import Category, router

    async def fake_yf_fundamentals(ticker, as_of=None):
        return f"yfinance fundamentals for {ticker}"

    monkeypatch.setitem(
        router._categories,
        "fundamental_data",
        Category("fundamental_data", [kis_provider._kis_fundamentals, fake_yf_fundamentals], False),
    )
    result = await router.call("fundamental_data", ticker="AAPL", as_of=None)
    assert result == "yfinance fundamentals for AAPL"


# ---------------- get_kr_sector_breadth ----------------


class _FakeSector:
    def __init__(self, name, change_pct, code=None):
        self.name = name
        self.change_pct = change_pct
        # None means "let _SectorBroker assign a realistic 00xx/10xx code below" —
        # real KIS rows always carry a market-prefixed code, so a fixture default
        # of `code=name` (Korean text, no prefix) would silently defeat any
        # prefix-based market verification and hide the exact bug this module fixes.
        self.code = code


class _SectorBroker:
    def __init__(self, kospi=None, kosdaq=None, raise_exc=None, sector_name=None, sector_name_exc=None):
        self._kospi = self._with_codes(kospi or [], "00")
        self._kosdaq = self._with_codes(kosdaq or [], "10")
        self._raise = raise_exc
        self._sector_name = sector_name
        self._sector_name_exc = sector_name_exc

    @staticmethod
    def _with_codes(sectors, prefix):
        for i, s in enumerate(sectors):
            if s.code is None:
                s.code = f"{prefix}{i:02d}"
        return sectors

    async def fetch_sector_indices(self, market="0001"):
        if self._raise is not None:
            raise self._raise
        return self._kospi if market == "0001" else self._kosdaq

    async def fetch_sector_name(self, code):
        if self._sector_name_exc is not None:
            raise self._sector_name_exc
        return self._sector_name


async def test_get_kr_sector_breadth_ranks_top_and_bottom_movers(monkeypatch):
    broker = _SectorBroker(
        kospi=[_FakeSector("전기·전자", 3.0), _FakeSector("건설", -2.0), _FakeSector("화학", 0.5)],
        kosdaq=[_FakeSector("유통", 1.5)],
    )
    monkeypatch.setattr(kis_provider, "_get_broker", _returns(broker))
    result = await kis_provider.get_kr_sector_breadth()
    assert "전기·전자 +3.00%" in result
    assert "건설 -2.00%" in result


async def test_get_kr_sector_breadth_combines_kospi_and_kosdaq(monkeypatch):
    broker = _SectorBroker(kospi=[_FakeSector("전기·전자", 1.0)], kosdaq=[_FakeSector("유통", 2.0)])
    monkeypatch.setattr(kis_provider, "_get_broker", _returns(broker))
    result = await kis_provider.get_kr_sector_breadth()
    assert "전기·전자" in result and "유통" in result


async def test_get_kr_sector_breadth_skipped_for_backtest_date(monkeypatch):
    async def should_not_be_called():
        raise AssertionError("must not call the broker for a historical as_of date")

    monkeypatch.setattr(kis_provider, "_get_broker", should_not_be_called)
    result = await kis_provider.get_kr_sector_breadth(as_of=date(2020, 1, 1))
    assert "과거 시점" in result


async def test_get_kr_sector_breadth_no_broker_configured(monkeypatch):
    async def _raise():
        raise RuntimeError("KIS not configured")

    monkeypatch.setattr(kis_provider, "_get_broker", _raise)
    result = await kis_provider.get_kr_sector_breadth()
    assert "KIS 미설정" in result


async def test_get_kr_sector_breadth_degrades_on_fetch_failure(monkeypatch):
    broker = _SectorBroker(raise_exc=RuntimeError("boom"))
    monkeypatch.setattr(kis_provider, "_get_broker", _returns(broker))
    result = await kis_provider.get_kr_sector_breadth()
    assert "데이터 없음" in result


# --- (a)/(d) reframed breadth + injected own-sector line -------------------


async def test_get_kr_sector_breadth_no_ticker_has_whole_market_framing_and_footer(monkeypatch):
    broker = _SectorBroker(kospi=[_FakeSector("전기·전자", 3.0), _FakeSector("건설", -2.0)])
    monkeypatch.setattr(kis_provider, "_get_broker", _returns(broker))
    result = await kis_provider.get_kr_sector_breadth()
    assert "WHOLE-MARKET ranking" in result
    assert "Do not state or imply" in result
    assert "KRX industry sector:" not in result  # no injected per-stock line
    assert "could not be determined this cycle" not in result


async def test_get_kr_sector_breadth_injects_own_sector_line_with_rank(monkeypatch):
    broker = _SectorBroker(
        kospi=[_FakeSector("전기·전자", 3.0), _FakeSector("보험", 1.2), _FakeSector("건설", -2.0)],
        sector_name=("보험", "KOSPI200"),
    )
    monkeypatch.setattr(kis_provider, "_get_broker", _returns(broker))
    result = await kis_provider.get_kr_sector_breadth(ticker="088350.KS")
    assert "This stock's own KRX industry sector: 보험 (KOSPI200), +1.20% today — ranked 2 of 3" in result


async def test_get_kr_sector_breadth_own_sector_not_in_ranking(monkeypatch):
    broker = _SectorBroker(
        kospi=[_FakeSector("전기·전자", 3.0), _FakeSector("건설", -2.0)],
        kosdaq=[_FakeSector("유통", 2.0)],  # normal KOSDAQ coverage — this sector is just absent
        sector_name=("기타제조", "KOSDAQ"),
    )
    monkeypatch.setattr(kis_provider, "_get_broker", _returns(broker))
    result = await kis_provider.get_kr_sector_breadth(ticker="026040.KS")
    assert "This stock's own KRX industry sector: 기타제조 (KOSDAQ). Its index is NOT in the ranking" in result
    assert "today —" not in result.split("기타제조 (KOSDAQ)")[1].split("\n")[0]


async def test_get_kr_sector_breadth_own_sector_lookup_failure_is_non_fatal(monkeypatch):
    broker = _SectorBroker(
        kospi=[_FakeSector("전기·전자", 3.0), _FakeSector("건설", -2.0)],
        sector_name_exc=RuntimeError("boom"),
    )
    monkeypatch.setattr(kis_provider, "_get_broker", _returns(broker))
    result = await kis_provider.get_kr_sector_breadth(ticker="088350.KS")
    assert "could not be determined this cycle" in result
    assert "전기·전자 +3.00%" in result  # ranking still rendered


async def test_get_kr_sector_breadth_dedupes_by_code(monkeypatch):
    broker = _SectorBroker(
        kospi=[_FakeSector("보험", 1.2, code="0025")],
        kosdaq=[_FakeSector("보험", 9.9, code="0025")],  # same code, must be dropped
    )
    monkeypatch.setattr(kis_provider, "_get_broker", _returns(broker))
    result = await kis_provider.get_kr_sector_breadth()
    assert result.count("보험") == 1


async def test_get_kr_sector_breadth_backtest_date_does_no_sector_lookup(monkeypatch):
    async def should_not_be_called():
        raise AssertionError("no broker call for a historical as_of")

    monkeypatch.setattr(kis_provider, "_get_broker", should_not_be_called)
    result = await kis_provider.get_kr_sector_breadth(as_of=date(2020, 1, 1), ticker="088350.KS")
    assert "과거 시점" in result


# --- own-sector line: byte-identical single/zero match + collision disambiguation ---

_SINGLE_MATCH_OWN_LINE = (
    "This stock's own KRX industry sector: 보험 (KOSPI200), +1.20% today — ranked 2 of 3 "
    "in the ranking above. This line, and only this line, is the authoritative sector for "
    "the analyzed stock."
)
_ZERO_MATCH_OWN_LINE = (
    "This stock's own KRX industry sector: 기타제조 (KOSDAQ). Its index is NOT in the "
    "ranking above, so no move is known for this stock's own sector — the ranking above "
    "tells you nothing about it."
)


async def test_own_sector_line_single_match_is_byte_identical(monkeypatch):
    broker = _SectorBroker(
        kospi=[_FakeSector("전기·전자", 3.0), _FakeSector("보험", 1.2), _FakeSector("건설", -2.0)],
        sector_name=("보험", "KOSPI200"),
    )
    monkeypatch.setattr(kis_provider, "_get_broker", _returns(broker))
    result = await kis_provider.get_kr_sector_breadth(ticker="088350.KS")
    assert _SINGLE_MATCH_OWN_LINE in result


async def test_own_sector_line_zero_match_is_byte_identical(monkeypatch):
    broker = _SectorBroker(
        kospi=[_FakeSector("전기·전자", 3.0), _FakeSector("건설", -2.0)],
        kosdaq=[_FakeSector("유통", 2.0)],  # normal KOSDAQ coverage — this sector is just absent
        sector_name=("기타제조", "KOSDAQ"),
    )
    monkeypatch.setattr(kis_provider, "_get_broker", _returns(broker))
    result = await kis_provider.get_kr_sector_breadth(ticker="026040.KS")
    assert _ZERO_MATCH_OWN_LINE in result


def _collision_broker(market_string):
    return _SectorBroker(
        kospi=[_FakeSector("보험", 2.0, code="0025"), _FakeSector("전기·전자", 3.0, code="0027")],
        kosdaq=[_FakeSector("보험", 8.0, code="1025")],
        sector_name=("보험", market_string),
    )


async def test_own_sector_line_collision_resolves_to_kosdaq(monkeypatch):
    monkeypatch.setattr(kis_provider, "_get_broker", _returns(_collision_broker("KOSDAQ")))
    result = await kis_provider.get_kr_sector_breadth(ticker="088350.KS")
    assert "보험 (KOSDAQ), +8.00% today" in result
    assert "보험 (KOSDAQ), +2.00% today" not in result


async def test_own_sector_line_collision_resolves_to_kospi(monkeypatch):
    monkeypatch.setattr(kis_provider, "_get_broker", _returns(_collision_broker("KOSPI")))
    result = await kis_provider.get_kr_sector_breadth(ticker="088350.KS")
    assert "보험 (KOSPI), +2.00% today" in result
    assert "보험 (KOSPI), +8.00% today" not in result


async def test_own_sector_line_collision_unrecognized_market_is_ambiguous(monkeypatch, caplog):
    monkeypatch.setattr(kis_provider, "_get_broker", _returns(_collision_broker("KONEX")))
    with caplog.at_level(logging.WARNING, logger="app.dataflows.kis_provider"):
        result = await kis_provider.get_kr_sector_breadth(ticker="088350.KS")
    assert "published as an index in more than one market" in result
    assert "No move is attributed to this stock's own sector" in result
    assert "is the authoritative sector for the analyzed stock" not in result
    assert "today — ranked" not in result
    assert any("own sector ambiguous" in r.getMessage() for r in caplog.records)


async def test_own_sector_line_coverage_collapse_does_not_misattribute_other_market(monkeypatch):
    """Regression for the 2026-09-09 재발: a KOSDAQ stock's sector name (e.g. '금속')
    only survives in KOSPI's ranking this cycle because KOSDAQ's own sector-index
    fetch came back empty (a real, observed intermittent KIS coverage gap — see
    fetch_sector_indices's docstring). A single leftover KOSPI-name match must NOT
    be trusted as "the authoritative sector" for a KOSDAQ stock — that was exactly
    the pre-fix bug (2026-09-03 앱튼 사건과 같은 클래스, 2026-09-09 라이브에서
    162300.KS/'금속'으로 재확인됨)."""
    broker = _SectorBroker(
        kospi=[_FakeSector("금속", 3.0)],
        kosdaq=[],  # this cycle's KOSDAQ fetch collapsed to nothing
        sector_name=("금속", "KOSDAQ"),
    )
    monkeypatch.setattr(kis_provider, "_get_broker", _returns(broker))
    result = await kis_provider.get_kr_sector_breadth(ticker="162300.KS")
    assert "금속 (KOSDAQ), +3.00% today" not in result  # must not borrow KOSPI's move
    assert "is the authoritative sector for the analyzed stock" not in result
    assert "ZERO sector-index rows for this stock's own market" in result


# ---------------- get_kr_vi_status ----------------


class _FakeVIStatus:
    def __init__(self, code, triggered_at, released_at, vi_price=70000.0):
        self.code = code
        self.triggered_at = triggered_at
        self.released_at = released_at
        self.vi_price = vi_price


class _VIBroker:
    def __init__(self, statuses=None, raise_exc=None):
        self._statuses = statuses or []
        self._raise = raise_exc

    async def fetch_vi_status(self, *, code=""):
        if self._raise is not None:
            raise self._raise
        return self._statuses


async def test_get_kr_vi_status_returns_not_supported_for_non_krx():
    result = await kis_provider.get_kr_vi_status("AAPL")
    assert "한국(KRX) 종목만" in result


async def test_get_kr_vi_status_skipped_for_backtest_date():
    result = await kis_provider.get_kr_vi_status("005930.KS", as_of=date(2020, 1, 1))
    assert "과거 시점" in result


async def test_get_kr_vi_status_not_configured_message(monkeypatch):
    async def _raise():
        raise RuntimeError("not configured")

    monkeypatch.setattr(kis_provider, "_get_broker", _raise)
    result = await kis_provider.get_kr_vi_status("005930.KS")
    assert "KIS 미설정" in result


async def test_get_kr_vi_status_no_history_today(monkeypatch):
    broker = _VIBroker(statuses=[])
    monkeypatch.setattr(kis_provider, "_get_broker", _returns(broker))
    result = await kis_provider.get_kr_vi_status("005930.KS")
    assert "발동 이력 없음" in result


async def test_get_kr_vi_status_currently_active(monkeypatch):
    broker = _VIBroker(statuses=[_FakeVIStatus("005930", "101500", "")])
    monkeypatch.setattr(kis_provider, "_get_broker", _returns(broker))
    result = await kis_provider.get_kr_vi_status("005930.KS")
    assert "VI 발동 중" in result
    assert "체결되지 않을 수 있음" in result


async def test_get_kr_vi_status_released(monkeypatch):
    broker = _VIBroker(statuses=[_FakeVIStatus("005930", "101500", "102000")])
    monkeypatch.setattr(kis_provider, "_get_broker", _returns(broker))
    result = await kis_provider.get_kr_vi_status("005930.KS")
    assert "현재는 해제됨" in result


async def test_get_kr_vi_status_degrades_on_fetch_failure(monkeypatch):
    broker = _VIBroker(raise_exc=RuntimeError("boom"))
    monkeypatch.setattr(kis_provider, "_get_broker", _returns(broker))
    result = await kis_provider.get_kr_vi_status("005930.KS")
    assert "조회 실패" in result


# ---------------- get_overseas_market_indices ----------------


class _FakeIndexQuote:
    def __init__(self, name, value, change_pct):
        self.name = name
        self.value = value
        self.change_pct = change_pct


class _OverseasIndexBroker:
    def __init__(self, quotes_by_code=None, raise_exc=None):
        self._quotes_by_code = quotes_by_code or {}
        self._raise = raise_exc

    async def fetch_overseas_index_quote(self, symbol, name, start, end):
        if self._raise is not None:
            raise self._raise
        return self._quotes_by_code[symbol]


async def test_get_overseas_market_indices_reports_all_three(monkeypatch):
    broker = _OverseasIndexBroker(quotes_by_code={
        ".DJI": _FakeIndexQuote("Dow Jones", 53178.41, 1.32),
        "SPX": _FakeIndexQuote("S&P 500", 7600.50, 0.60),
        "NDX": _FakeIndexQuote("Nasdaq 100", 28776.80, -0.10),
    })
    monkeypatch.setattr(kis_provider, "_get_broker", _returns(broker))
    result = await kis_provider.get_overseas_market_indices()
    assert "Dow Jones: 53,178.41 (+1.32%)" in result
    assert "S&P 500: 7,600.50 (+0.60%)" in result
    assert "Nasdaq 100: 28,776.80 (-0.10%)" in result


async def test_get_overseas_market_indices_skipped_for_backtest_date(monkeypatch):
    def should_not_be_called():
        raise AssertionError("must not call the broker for a historical as_of date")

    monkeypatch.setattr(kis_provider, "_get_broker", should_not_be_called)
    result = await kis_provider.get_overseas_market_indices(as_of=date(2020, 1, 1))
    assert "과거 시점" in result


async def test_get_overseas_market_indices_not_configured_message(monkeypatch):
    async def _raise():
        raise RuntimeError("not configured")

    monkeypatch.setattr(kis_provider, "_get_broker", _raise)
    result = await kis_provider.get_overseas_market_indices()
    assert "KIS 미설정" in result


async def test_get_overseas_market_indices_degrades_on_fetch_failure(monkeypatch):
    broker = _OverseasIndexBroker(raise_exc=RuntimeError("boom"))
    monkeypatch.setattr(kis_provider, "_get_broker", _returns(broker))
    result = await kis_provider.get_overseas_market_indices()
    assert "가져오지 못했습니다" in result


# ---------------------------------------------------------------------------
# _period_labels — 2026-08-19: KIS's "annual" financial-statement TRs
# (div_cls="0") actually return one leading interim row ahead of the FY rows.
# Rendered with no marker, the model divided an interim row by a full-year
# row and reported a fabricated "-63% revenue collapse" for 316140.KS
# (Woori) that drove a full liquidation recommendation. See docstring.
# ---------------------------------------------------------------------------


def test_period_labels_woori_shape_one_interim_three_fy():
    periods = ["202603", "202512", "202412", "202312"]
    labels = kis_provider._period_labels(periods)
    assert labels["202603"] == "202603 (interim, 3 months — NOT a full year)"
    assert labels["202512"] == "202512 (FY, 12 months)"
    assert labels["202412"] == "202412 (FY, 12 months)"
    assert labels["202312"] == "202312 (FY, 12 months)"


def test_period_labels_june_interim_yields_six_months():
    periods = ["202606", "202512", "202412", "202312"]
    labels = kis_provider._period_labels(periods)
    assert labels["202606"] == "202606 (interim, 6 months — NOT a full year)"
    assert labels["202512"] == "202512 (FY, 12 months)"


def test_period_labels_all_annual_labels_everything_fy():
    periods = ["202512", "202412", "202312"]
    labels = kis_provider._period_labels(periods)
    assert all(v.endswith("(FY, 12 months)") for v in labels.values())


def test_period_labels_no_guess_when_modal_month_occurs_once():
    periods = ["202603", "202512"]
    assert kis_provider._period_labels(periods) == {}


def test_period_labels_empty_input_returns_empty_dict():
    assert kis_provider._period_labels([]) == {}


def test_period_labels_count_tie_favors_earliest_month_when_sorted():
    # 2-2 tie between "03" and "12" months. Callers must pass periods sorted
    # ascending (get_kr_financials does this via `sorted({...})`, not
    # `list({...})`, specifically so this tie-break is deterministic instead
    # of depending on PYTHONHASHSEED). With the oldest period first, "12"
    # wins the tie — the correct answer, since the interim row is always the
    # most recent period, so the older "12" rows are the reliable FY anchor.
    periods = sorted(["202603", "202503", "202512", "202412"])
    labels = kis_provider._period_labels(periods)
    assert labels["202512"] == "202512 (FY, 12 months)"
    assert labels["202412"] == "202412 (FY, 12 months)"
    assert labels["202603"] == "202603 (interim, 3 months — NOT a full year)"
    assert labels["202503"] == "202503 (interim, 3 months — NOT a full year)"


# ---------------------------------------------------------------------------
# get_kr_financials — interim/FY marker actually reaches the rendered
# [Income statement] text, plus the header's redirect-to-revenue_growth
# instruction.
# ---------------------------------------------------------------------------


async def test_get_kr_financials_marks_interim_row_in_income_statement(monkeypatch):
    fake = FakeBroker(income_statement=[
        IncomeStatement(period="202603", revenue=152_433, gross_profit=1, operating_profit=1, net_income=1),
        IncomeStatement(period="202512", revenue=415_484, gross_profit=1, operating_profit=1, net_income=1),
        IncomeStatement(period="202412", revenue=466_167, gross_profit=1, operating_profit=1, net_income=1),
        IncomeStatement(period="202312", revenue=419_863, gross_profit=1, operating_profit=1, net_income=1),
    ])
    monkeypatch.setattr(kis_provider, "_get_broker", _returns(fake))
    result = await kis_provider.get_kr_financials("316140.KS")
    assert "202603 (interim, 3 months — NOT a full year): revenue=152,433억원" in result
    assert "202512 (FY, 12 months): revenue=415,484억원" in result
    assert "Never compute a growth rate, decline, or \"collapse\"" in result
    assert "revenue_growth" in result and "net_income_growth" in result


def _kia_inc():
    return [
        IncomeStatement(period="202606", revenue=625_389, gross_profit=1, operating_profit=48_337, net_income=41_579),
        IncomeStatement(period="202512", revenue=1_141_400, gross_profit=1, operating_profit=90_781, net_income=75_500),
        IncomeStatement(period="202412", revenue=1_074_500, gross_profit=1, operating_profit=126_671, net_income=97_800),
        IncomeStatement(period="202312", revenue=998_000, gross_profit=1, operating_profit=1, net_income=1),
    ]


def _kia_rat():
    return [
        FinancialRatio(period="202606", revenue_growth=2.87, op_profit_growth=6.51, net_income_growth=-10.79,
                       roe=11.61, eps=1, bps=1, debt_ratio=65.14),
        FinancialRatio(period="202512", revenue_growth=6.23, op_profit_growth=-8.0, net_income_growth=-22.72,
                       roe=12.92, eps=1, bps=1, debt_ratio=61.76),
    ]


async def test_interim_context_none_when_no_interim_row():
    labels = kis_provider._period_labels(["202512", "202412", "202312"])
    assert kis_provider._interim_context(labels, _kia_inc()[1:], _kia_rat()) is None


async def test_interim_context_none_when_labels_empty():
    assert kis_provider._interim_context({}, _kia_inc(), _kia_rat()) is None


async def test_interim_context_kia_shape():
    labels = kis_provider._period_labels(sorted(s.period for s in _kia_inc()))
    ctx = kis_provider._interim_context(labels, _kia_inc(), _kia_rat())
    assert ctx is not None
    assert ctx.period == "202606" and ctx.months == 6
    assert ctx.fy_periods == ("202512", "202412", "202312")
    assert ctx.op_profit_growth == 6.51 and ctx.net_income_growth == -10.79


async def test_interim_context_zero_growth_is_not_reported():
    labels = kis_provider._period_labels(sorted(s.period for s in _kia_inc()))
    rat = [FinancialRatio(period="202606", revenue_growth=0.0, op_profit_growth=0.0, net_income_growth=0.0,
                          roe=1, eps=1, bps=1, debt_ratio=1)]
    ctx = kis_provider._interim_context(labels, _kia_inc(), rat)
    assert ctx.revenue_growth is None and ctx.op_profit_growth is None
    assert kis_provider._interim_growth_sentence(ctx) == ""


async def test_get_kr_financials_period_warning_names_both_rows(monkeypatch):
    fake = FakeBroker(income_statement=_kia_inc(), financial_ratio=_kia_rat())
    monkeypatch.setattr(kis_provider, "_get_broker", _returns(fake))
    result = await kis_provider.get_kr_financials("000270.KS")
    assert "PERIOD WARNING" in result
    warn = result.split("PERIOD WARNING")[1]
    assert "202606" in warn and "202512" in warn and "6-MONTH interim" in warn
    assert "operating profit +6.51%, net income -10.79%" in warn


async def test_get_kr_financials_income_rows_carry_operating_profit_growth(monkeypatch):
    fake = FakeBroker(income_statement=_kia_inc(), financial_ratio=_kia_rat())
    monkeypatch.setattr(kis_provider, "_get_broker", _returns(fake))
    result = await kis_provider.get_kr_financials("000270.KS")
    assert "operating_profit_growth=+6.51%" in result
    assert "net_income_growth=-10.79%" in result


async def test_get_kr_financials_no_interim_has_no_period_warning(monkeypatch):
    fake = FakeBroker(income_statement=_kia_inc()[1:], financial_ratio=_kia_rat()[1:])
    monkeypatch.setattr(kis_provider, "_get_broker", _returns(fake))
    result = await kis_provider.get_kr_financials("000270.KS")
    assert "PERIOD WARNING" not in result


async def test_period_work_prints_only_percentages_never_new_eok(monkeypatch):
    fake = FakeBroker(income_statement=_kia_inc(), financial_ratio=_kia_rat())
    monkeypatch.setattr(kis_provider, "_get_broker", _returns(fake))
    result = await kis_provider.get_kr_financials("000270.KS")
    # the two things this fix adds — the per-row growth line and the PERIOD
    # WARNING block — must never contain a 억원 figure (it would poison
    # unit_scale_guard's ground-truth set).
    for line in result.splitlines():
        if "exchange-reported YoY" in line or "PERIOD WARNING" in line or line.strip().startswith("The only correct"):
            assert "억원" not in line, line


async def test_period_note_out_populated_only_with_interim(monkeypatch):
    fake = FakeBroker(income_statement=_kia_inc(), financial_ratio=_kia_rat())
    monkeypatch.setattr(kis_provider, "_get_broker", _returns(fake))
    out: list = []
    await kis_provider.get_kr_financials("000270.KS", period_note_out=out)
    assert len(out) == 1 and "INTERIM-PERIOD WARNING" in out[0]
    assert "any earlier report, research plan, or debate transcript" in out[0]

    fake2 = FakeBroker(income_statement=_kia_inc()[1:], financial_ratio=_kia_rat()[1:])
    monkeypatch.setattr(kis_provider, "_get_broker", _returns(fake2))
    out2: list = []
    await kis_provider.get_kr_financials("000270.KS", period_note_out=out2)
    assert out2 == []


def test_interim_period_note_scopes_out_margins_and_like_for_like():
    # 2026-09-04: the warning must not be usable to dismiss a margin-based or
    # same-period YoY deterioration.
    ctx = kis_provider._InterimContext(
        period="202606", months=6, fy_periods=("202512", "202412"),
        revenue_growth=2.87, op_profit_growth=6.51, net_income_growth=-10.79,
    )
    note = kis_provider.interim_period_note(ctx)
    assert "a margin that fell year-over-year fell for real" in note
    assert kis_provider.interim_period_note(None) == ""


async def test_get_kr_financials_renders_current_price_in_valuation(monkeypatch):
    fake = FakeBroker(valuation=StockValuation(
        code="207940", price=1_531_000, per=50.0, pbr=8.0, eps=30_000, bps=180_000, market_cap=1_000_000,
    ))
    monkeypatch.setattr(kis_provider, "_get_broker", _returns(fake))
    result = await kis_provider.get_kr_financials("207940.KS")
    assert "Current price 1,531,000원" in result


# ---------------------------------------------------------------------------
# get_kr_financials — FINANCIAL-DATA RECENCY warning (2026-09-07 domestic
# audit #6, id=1237 / 294630.KS: KIS's annual series ended at FY2024, 20
# months before the analysis date, and the desk read the 2024 figures as the
# company's current condition with no signal anywhere).
# ---------------------------------------------------------------------------


def _seonam_inc():
    # all FY, newest 202412 — the id=1237 shape
    return [
        IncomeStatement(period="202412", revenue=14, gross_profit=-9, operating_profit=-45, net_income=-39),
        IncomeStatement(period="202312", revenue=64, gross_profit=8, operating_profit=-31, net_income=-113),
        IncomeStatement(period="202212", revenue=64, gross_profit=-2, operating_profit=-24, net_income=-21),
        IncomeStatement(period="202112", revenue=11, gross_profit=-14, operating_profit=-36, net_income=-40),
    ]


async def test_get_kr_financials_warns_when_latest_period_is_stale(monkeypatch):
    set_clock(FakeClock(datetime(2026, 9, 7, 10, 0, tzinfo=_KST)))
    fake = FakeBroker(income_statement=_seonam_inc())
    monkeypatch.setattr(kis_provider, "_get_broker", _returns(fake))
    result = await kis_provider.get_kr_financials("294630.KS")
    assert "FINANCIAL-DATA RECENCY" in result
    recency = result.split("FINANCIAL-DATA RECENCY")[1]
    assert "202412 (FY, 12 months)" in recency
    # (2026-2024)*12 + (9-12) = 24 - 3 = 21 months since the FY2024 close
    assert "roughly 21 months" in result


async def test_get_kr_financials_no_staleness_note_when_fresh(monkeypatch):
    """Byte-identical to today's output for a company whose newest FY closed
    within the statutory filing window — the 'healthy path unchanged' fence."""
    set_clock(FakeClock(datetime(2026, 9, 7, 10, 0, tzinfo=_KST)))
    inc = [
        IncomeStatement(period="202512", revenue=100, gross_profit=10, operating_profit=5, net_income=3),
        IncomeStatement(period="202412", revenue=90, gross_profit=9, operating_profit=4, net_income=2),
    ]
    fake = FakeBroker(income_statement=inc)
    monkeypatch.setattr(kis_provider, "_get_broker", _returns(fake))
    result = await kis_provider.get_kr_financials("005930.KS")
    assert "FINANCIAL-DATA RECENCY" not in result
    assert "RECENCY" not in result


async def test_get_kr_financials_staleness_boundary(monkeypatch):
    # 11 months -> silent, 12 -> tier 1, 24 -> tier 2
    def _one_fy(period):
        return [IncomeStatement(period=period, revenue=10, gross_profit=1, operating_profit=1, net_income=1)]

    set_clock(FakeClock(datetime(2026, 9, 7, 10, 0, tzinfo=_KST)))
    for period, expect_note, expect_tier2 in (
        ("202510", False, False),   # 11 months
        ("202509", True, False),    # 12 months
        ("202410", True, False),    # 23 months -> tier 1 only
        ("202409", True, True),     # 24 months -> tier 2
    ):
        fake = FakeBroker(income_statement=_one_fy(period))
        monkeypatch.setattr(kis_provider, "_get_broker", _returns(fake))
        result = await kis_provider.get_kr_financials("005930.KS")
        assert ("FINANCIAL-DATA RECENCY" in result) is expect_note, period
        assert ("material concern" in result) is expect_tier2, period


async def test_staleness_note_carries_no_eok_or_growth_or_income_row(monkeypatch):
    set_clock(FakeClock(datetime(2026, 9, 7, 10, 0, tzinfo=_KST)))
    fake = FakeBroker(income_statement=_seonam_inc())
    monkeypatch.setattr(kis_provider, "_get_broker", _returns(fake))
    result = await kis_provider.get_kr_financials("294630.KS")
    note = "FINANCIAL-DATA RECENCY" + result.split("FINANCIAL-DATA RECENCY")[1].split("\n")[0]
    assert "억" not in note
    assert "billion" not in note and "trillion" not in note
    assert "_growth=" not in note
    assert "%" not in note


async def test_period_note_out_gets_staleness_note(monkeypatch):
    set_clock(FakeClock(datetime(2026, 9, 7, 10, 0, tzinfo=_KST)))
    fake = FakeBroker(income_statement=_seonam_inc())
    monkeypatch.setattr(kis_provider, "_get_broker", _returns(fake))
    out: list = []
    recency: dict = {}
    await kis_provider.get_kr_financials("294630.KS", period_note_out=out, recency_out=recency)
    assert len(out) == 1 and "FINANCIAL-DATA RECENCY" in out[0]
    assert "INTERIM-PERIOD WARNING" not in out[0]  # no interim row
    assert recency["latest_period"] == "202412"
    assert recency["months_stale"] == 21 and recency["interim"] is False


async def test_period_note_out_gets_both_when_stale_and_interim(monkeypatch):
    # newest row a 2-year-stale interim
    set_clock(FakeClock(datetime(2026, 9, 7, 10, 0, tzinfo=_KST)))
    inc = [
        IncomeStatement(period="202406", revenue=5, gross_profit=1, operating_profit=1, net_income=1),
        IncomeStatement(period="202312", revenue=10, gross_profit=1, operating_profit=1, net_income=1),
        IncomeStatement(period="202212", revenue=10, gross_profit=1, operating_profit=1, net_income=1),
    ]
    rat = [FinancialRatio(period="202406", revenue_growth=1.0, op_profit_growth=0.0, net_income_growth=0.0,
                          roe=1, eps=1, bps=1, debt_ratio=1)]
    fake = FakeBroker(income_statement=inc, financial_ratio=rat)
    monkeypatch.setattr(kis_provider, "_get_broker", _returns(fake))
    out: list = []
    await kis_provider.get_kr_financials("000270.KS", period_note_out=out)
    assert len(out) == 1
    assert "INTERIM-PERIOD WARNING" in out[0] and "FINANCIAL-DATA RECENCY" in out[0]


async def test_kr_financials_recency_helper_handles_unparseable_period():
    r = kis_provider.kr_financials_recency("", "", date(2026, 9, 7))
    assert r["months_stale"] is None


async def test_get_kr_disclosures_empty_result_names_its_scope(monkeypatch):
    class _NoDivBroker(FakeBroker):
        async def fetch_dividend_schedule(self, code, start, end):
            return []

    set_clock(FakeClock(datetime(2026, 9, 7, 10, 0, tzinfo=_KST)))
    monkeypatch.setattr(kis_provider, "_get_broker", _returns(_NoDivBroker()))
    result = await kis_provider.get_kr_disclosures("294630.KS")
    assert "CASH DIVIDEND SCHEDULE ONLY" in result
    assert "not a filings or disclosure feed" in result
    assert "fiscal" not in result.lower()


# ---------------------------------------------------------------------------
# kr_amount_unit_key — 2026-09-07 domestic audit #3 (id=1237 / 294630.KS:
# a headline's "6.6억원" contract reported as "₩6.6 billion").
# ---------------------------------------------------------------------------


def test_kr_amount_unit_key_converts_headline_eok():
    news = (
        "[20260904] 서남 수주공시 - 초전도 선재 공급 계약 6.6억원 (매출액대비 19.56 %)\n"
        "[20260904] 서남, 6.62억원 규모 공급계약 체결\n"
        "[20260904] 서남, 6.6억 규모 공급계약"
    )
    key = kis_provider.kr_amount_unit_key(news)
    assert "KRW UNIT KEY" in key
    assert "660 million KRW" in key and "0.66 billion KRW" in key
    assert "662 million KRW" in key
    # 6.6억 (no 원) is caught and deduped with 6.6억원
    assert key.count("660 million KRW") == 1


def test_kr_amount_unit_key_empty_without_amounts():
    assert kis_provider.kr_amount_unit_key("서남, 계약 소식에 강세") == ""


def test_kr_amount_unit_key_preserves_fractional_eok_not_rounded():
    # _eok would render 6.6억 as "7억원"; this gloss must keep the 0.66
    key = kis_provider.kr_amount_unit_key("공급계약 6.6억원 체결")
    assert "7억" not in key
    assert "0.66 billion KRW" in key


def test_kr_amount_unit_key_handles_jo_and_large_amounts():
    key = kis_provider.kr_amount_unit_key("삼성전자, 12조원 규모 투자 / 어떤 계약 5,000억원")
    assert "12 trillion KRW" in key and "12,000 billion KRW" in key  # 12조
    assert "500 billion KRW" in key  # 5,000억 = 500 billion


def test_kr_amount_unit_key_drops_implausible_magnitude():
    # 9,999,999억 = ~1000조, obviously not a real headline amount -> skipped
    key = kis_provider.kr_amount_unit_key("루머: 9999999억원 규모설")
    assert key == ""


def test_kr_amount_unit_key_percent_context_is_not_a_candidate():
    # "매출액대비 19.56 %" carries no 억/조 token — must not be picked up
    key = kis_provider.kr_amount_unit_key("계약 (매출액대비 19.56 %)")
    assert key == ""


# ---------------------------------------------------------------------------
# get_kr_analyst_opinion — PrevClose column renamed to CloseThen (display
# only) plus a clarifying note that it is not today's current price.
# ---------------------------------------------------------------------------


async def test_get_kr_analyst_opinion_renames_prevclose_column_and_clarifies(monkeypatch):
    from app.domain.models.market import InvestmentOpinion

    fake = FakeBroker(invest_opinion=[
        InvestmentOpinion(day="20260709", opinion="매수", prev_opinion="매수", target_price=40_000, prev_close=30_700),
    ])
    monkeypatch.setattr(kis_provider, "_get_broker", _returns(fake))
    result = await kis_provider.get_kr_analyst_opinion("316140.KS")
    assert "CloseThen" in result
    assert "PrevClose" not in result
    assert "NOT today's current price" in result
    assert "[Valuation]" in result


# ---------------------------------------------------------------------------
# _eok — 2026-08-19: an English-language report translated 173,363억원 as
# "173,363 billion won" (=173 trillion, a 10x mislabel) because only the
# Korean unit string was rendered. An English gloss is now appended.
# ---------------------------------------------------------------------------


async def test_eok_above_jo_threshold_includes_korean_and_english_units(monkeypatch):
    from app.domain.models.market import OtherMajorRatio

    fake = FakeBroker(other_major_ratios=[
        OtherMajorRatio(period="202512", payout_ratio=0.0, eva=0.0, ebitda=173_363.0, ev_ebitda=1.0),
    ])
    monkeypatch.setattr(kis_provider, "_get_broker", _returns(fake))
    result = await kis_provider.get_kr_financials("316140.KS")
    assert "173,363억원(17.34조원)" in result
    assert "17.34 trillion KRW" in result


async def test_eok_above_jo_threshold_also_includes_billion_equivalent(monkeypatch):
    """2026-08-28 addendum: before this, the >=1조 branch gave ONLY a
    trillion-KRW gloss — a model that reached for "billion" anyway (English
    financial writing's default for figures this large) had no supplied
    answer to quote and derived one itself, every time carrying the 억원
    mantissa straight over into "billion" (a 10x error, 70 reports/20
    tickers in one day's live audit). Giving billion unconditionally, same
    as the <1조 branch, removes that derivation entirely."""
    from app.domain.models.market import OtherMajorRatio

    fake = FakeBroker(other_major_ratios=[
        OtherMajorRatio(period="202512", payout_ratio=0.0, eva=0.0, ebitda=173_363.0, ev_ebitda=1.0),
    ])
    monkeypatch.setattr(kis_provider, "_get_broker", _returns(fake))
    result = await kis_provider.get_kr_financials("316140.KS")
    assert "17,336.3 billion KRW" in result


async def test_eok_below_jo_threshold_includes_korean_and_english_units(monkeypatch):
    from app.domain.models.market import OtherMajorRatio

    fake = FakeBroker(other_major_ratios=[
        OtherMajorRatio(period="202512", payout_ratio=0.0, eva=0.0, ebitda=4_692.0, ev_ebitda=1.0),
    ])
    monkeypatch.setattr(kis_provider, "_get_broker", _returns(fake))
    result = await kis_provider.get_kr_financials("316140.KS")
    assert "4,692억원" in result
    assert "469.2 billion KRW" in result


# ---------------------------------------------------------------------------
# candle_run_cache / _cached_fetch (2026-08-25 — a cancelled waiter used to
# poison the shared fetch for every other caller in the same run scope, see
# id=377/id=949 in app.orchestrator.pipeline's docstrings)
# ---------------------------------------------------------------------------


class _CountingFetch:
    """A fetch_fn stand-in that counts calls and can be told to sleep, fail
    once, or hang until released — independent of FakeBroker/get_last_close
    so these tests exercise `_cached_fetch`/`candle_run_cache` directly at
    the exact layer the bug lives in."""

    def __init__(self, result=None, fail_times: int = 0):
        self.calls = 0
        self._result = result if result is not None else ["candles"]
        self._fail_times = fail_times
        self.release = asyncio.Event()

    async def __call__(self):
        self.calls += 1
        if self._fail_times > 0:
            self._fail_times -= 1
            raise RuntimeError("KIS timeout")
        await self.release.wait()
        return self._result


async def test_candle_run_cache_coalesces_concurrent_callers_onto_one_fetch():
    with kis_provider.candle_run_cache() as _:
        cache = kis_provider._candle_run_cache.get()
        fetch = _CountingFetch()
        fetch.release.set()
        results = await asyncio.gather(*[
            kis_provider._cached_fetch(cache, ("k",), fetch) for _ in range(5)
        ])
        assert results == [["candles"]] * 5
        assert fetch.calls == 1


async def test_a_cancelled_waiter_does_not_poison_the_shared_candle_fetch():
    """The regression test for the id=377/id=949-shaped bug: caller A gets
    cancelled mid-await, caller B (in the same cache scope) must still get
    real data, not a CancelledError from a `.get()` that never asked for
    anything."""
    with kis_provider.candle_run_cache() as _:
        cache = kis_provider._candle_run_cache.get()
        fetch = _CountingFetch()

        task_a = asyncio.ensure_future(kis_provider._cached_fetch(cache, ("k",), fetch))
        await asyncio.sleep(0)  # let A create and start awaiting the shared task
        task_a.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task_a

        fetch.release.set()
        result_b = await kis_provider._cached_fetch(cache, ("k",), fetch)
        assert result_b == ["candles"]
        assert fetch.calls == 1  # B reused A's shared task, not a second fetch


async def test_a_cancelled_shared_fetch_is_evicted_so_the_next_caller_refetches():
    with kis_provider.candle_run_cache() as _:
        cache = kis_provider._candle_run_cache.get()
        fetch = _CountingFetch()
        task = asyncio.ensure_future(fetch())
        cache[("k",)] = task
        task.add_done_callback(lambda t: kis_provider._evict_if_unusable(t, cache=cache, key=("k",)))
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.sleep(0)  # let the done-callback run
        assert ("k",) not in cache

        fetch2 = _CountingFetch()
        fetch2.release.set()
        result = await kis_provider._cached_fetch(cache, ("k",), fetch2)
        assert result == ["candles"]
        assert fetch2.calls == 1


async def test_a_failed_fetch_is_not_cached_for_the_rest_of_the_run():
    with kis_provider.candle_run_cache() as _:
        cache = kis_provider._candle_run_cache.get()
        fetch = _CountingFetch(fail_times=1)
        fetch.release.set()
        with pytest.raises(RuntimeError):
            await kis_provider._cached_fetch(cache, ("k",), fetch)
        result = await kis_provider._cached_fetch(cache, ("k",), fetch)
        assert result == ["candles"]
        assert fetch.calls == 2


async def test_candle_run_cache_cancels_an_in_flight_fetch_on_scope_exit():
    fetch = _CountingFetch()
    with kis_provider.candle_run_cache() as _:
        cache = kis_provider._candle_run_cache.get()
        task = asyncio.ensure_future(kis_provider._cached_fetch(cache, ("k",), fetch))
        await asyncio.sleep(0)
        outstanding = list(cache.values())[0]
    with pytest.raises(asyncio.CancelledError):
        await task
    assert outstanding.cancelled()


# ---------------------------------------------------------------------------
# get_kr_financials fan-out concurrency gate (2026-09-03 decision id=1179) —
# the 7 statement/ratio TRs + fetch_valuation must not all hit KIS at once.
# ---------------------------------------------------------------------------


class _ConcurrencyBroker:
    """Every fetch_kr_* / fetch_valuation call bumps a live counter around an
    await, so the test can observe the peak simultaneity."""

    def __init__(self, *, fail_ratio: bool = False):
        self.live = 0
        self.peak = 0
        self._fail_ratio = fail_ratio

    async def _track(self):
        self.live += 1
        self.peak = max(self.peak, self.live)
        for _ in range(5):
            await asyncio.sleep(0)
        self.live -= 1

    async def fetch_kr_income_statement(self, code, div_cls="0"):
        await self._track()
        return []

    async def fetch_kr_balance_sheet(self, code, div_cls="0"):
        await self._track()
        return []

    async def fetch_kr_financial_ratio(self, code, div_cls="0"):
        await self._track()
        if self._fail_ratio:
            from app.brokers.kis import KISBrokerError

            raise KISBrokerError('financial ratio for 443670 failed: HTTP 500 {"msg_cd":"EGW00201"}')
        return []

    async def fetch_kr_stability_ratio(self, code, div_cls="0"):
        await self._track()
        return []

    async def fetch_kr_profitability_ratio(self, code, div_cls="0"):
        await self._track()
        return []

    async def fetch_kr_growth_ratio_detail(self, code, div_cls="0"):
        await self._track()
        return []

    async def fetch_kr_other_major_ratios(self, code, div_cls="0"):
        await self._track()
        return []

    async def fetch_valuation(self, code):
        await self._track()
        raise RuntimeError("no valuation")  # bonus section — swallowed by get_kr_financials


async def test_get_kr_financials_limits_concurrent_kis_calls(monkeypatch):
    broker = _ConcurrencyBroker()
    monkeypatch.setattr(kis_provider, "_get_broker", _returns(broker))
    monkeypatch.setattr(
        kis_provider.settings, "kis_financials_fanout_concurrency", 3, raising=False
    )

    await kis_provider.get_kr_financials("443670.KS")

    assert broker.peak <= 3, broker.peak
    assert broker.peak > 1  # sanity: the gate isn't accidentally serializing everything


async def test_get_kr_financials_propagates_a_rate_limited_ratio_tr(monkeypatch):
    from app.brokers.kis import KISBrokerError

    broker = _ConcurrencyBroker(fail_ratio=True)
    monkeypatch.setattr(kis_provider, "_get_broker", _returns(broker))

    with pytest.raises(KISBrokerError):
        await kis_provider.get_kr_financials("443670.KS")


# ---------------------------------------------------------------------------
# get_kis_english_name — 2026-09-03, identity_gate audit #6 Problem 2
# (measurement-only: feeds the WEAK alias tier, never authoritative).
# ---------------------------------------------------------------------------


class _EnglishNameBroker:
    def __init__(self, is_real, english_name="", raise_status_exc=None):
        self._is_real = is_real
        self._english_name = english_name
        self._raise_status_exc = raise_status_exc

    async def fetch_stock_status(self, code):
        if self._raise_status_exc is not None:
            raise self._raise_status_exc
        return StockStatus(
            code=code, trading_halted=False, administrative_issue=False, nxt_trading_halted=False,
            korean_name="", english_name=self._english_name,
        )


async def test_get_kis_english_name_returns_the_tr_field_on_a_real_account(monkeypatch):
    broker = _EnglishNameBroker(is_real=True, english_name="Tongyang Pile Inc")
    monkeypatch.setattr(kis_provider, "_get_broker", _returns(broker))
    result = await kis_provider.get_kis_english_name("003580.KS")
    assert result == "Tongyang Pile Inc"


async def test_get_kis_english_name_returns_none_on_a_paper_account(monkeypatch):
    # A paper account gets EGW02006 for the whole CTPF1002R TR — never even
    # attempt the call, just degrade to None so the caller falls back to
    # yfinance-only weak aliases exactly as before this addition.
    broker = _EnglishNameBroker(is_real=False, english_name="should not be reached")
    monkeypatch.setattr(kis_provider, "_get_broker", _returns(broker))
    result = await kis_provider.get_kis_english_name("003580.KS")
    assert result is None


async def test_get_kis_english_name_returns_none_when_broker_lookup_fails(monkeypatch):
    async def _broken_broker():
        raise RuntimeError("no credentials")

    monkeypatch.setattr(kis_provider, "_get_broker", _broken_broker)
    result = await kis_provider.get_kis_english_name("003580.KS")
    assert result is None


async def test_get_kis_english_name_returns_none_when_the_tr_itself_fails(monkeypatch):
    broker = _EnglishNameBroker(is_real=True, raise_status_exc=RuntimeError("EGW00215"))
    monkeypatch.setattr(kis_provider, "_get_broker", _returns(broker))
    result = await kis_provider.get_kis_english_name("003580.KS")
    assert result is None


async def test_get_kis_english_name_returns_none_for_a_non_krx_ticker():
    result = await kis_provider.get_kis_english_name("AAPL")
    assert result is None


async def test_get_kis_english_name_returns_none_when_field_is_blank(monkeypatch):
    broker = _EnglishNameBroker(is_real=True, english_name="")
    monkeypatch.setattr(kis_provider, "_get_broker", _returns(broker))
    result = await kis_provider.get_kis_english_name("003580.KS")
    assert result is None


# ---------------- get_kr_sector_name ----------------
# 2026-09-09, P1 audit item #3 (id=1354, 273060.KS/와이즈버즈): a cheap,
# per-ticker-only counterpart to get_kr_sector_breadth's own-sector line —
# see the function's own docstring for the Fundamentals Analyst incident.


class _SectorNameBroker:
    def __init__(self, name_market=None, raise_exc=None):
        self._name_market = name_market
        self._raise = raise_exc

    async def fetch_sector_name(self, code):
        if self._raise is not None:
            raise self._raise
        return self._name_market


async def test_get_kr_sector_name_returns_authoritative_line(monkeypatch):
    broker = _SectorNameBroker(name_market=("광고", "KOSDAQ"))
    monkeypatch.setattr(kis_provider, "_get_broker", _returns(broker))
    result = await kis_provider.get_kr_sector_name("273060.KS")
    assert "광고 (KOSDAQ)" in result
    assert "not inferred from financial ratios" in result


async def test_get_kr_sector_name_strips_ticker_suffix(monkeypatch):
    seen_codes = []

    class _RecordingBroker:
        async def fetch_sector_name(self, code):
            seen_codes.append(code)
            return ("광고", "KOSDAQ")

    monkeypatch.setattr(kis_provider, "_get_broker", _returns(_RecordingBroker()))
    await kis_provider.get_kr_sector_name("273060.KS")
    assert seen_codes == ["273060"]


async def test_get_kr_sector_name_empty_when_name_blank(monkeypatch):
    broker = _SectorNameBroker(name_market=("", "KOSDAQ"))
    monkeypatch.setattr(kis_provider, "_get_broker", _returns(broker))
    result = await kis_provider.get_kr_sector_name("273060.KS")
    assert result == ""


async def test_get_kr_sector_name_empty_on_fetch_failure(monkeypatch):
    broker = _SectorNameBroker(raise_exc=RuntimeError("boom"))
    monkeypatch.setattr(kis_provider, "_get_broker", _returns(broker))
    result = await kis_provider.get_kr_sector_name("273060.KS")
    assert result == ""


async def test_get_kr_sector_name_empty_when_no_broker_configured(monkeypatch):
    async def _raise():
        raise RuntimeError("KIS not configured")

    monkeypatch.setattr(kis_provider, "_get_broker", _raise)
    result = await kis_provider.get_kr_sector_name("273060.KS")
    assert result == ""


async def test_get_kr_sector_name_skipped_for_backtest_date(monkeypatch):
    async def should_not_be_called():
        raise AssertionError("must not call the broker for a historical as_of date")

    monkeypatch.setattr(kis_provider, "_get_broker", should_not_be_called)
    result = await kis_provider.get_kr_sector_name("273060.KS", as_of=date(2020, 1, 1))
    assert result == ""
