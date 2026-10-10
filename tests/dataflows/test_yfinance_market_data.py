"""app.dataflows.yfinance.market_data — yf.Ticker mocked via monkeypatch,
no real network (B-stage)."""

from __future__ import annotations

import time
from datetime import date, datetime
from zoneinfo import ZoneInfo

import pandas as pd
import pytest
import talib
import yfinance as yf

from app.clock import FakeClock, LiveClock, set_clock
from app.dataflows.yfinance import market_data

_US_TZ = ZoneInfo("America/New_York")


@pytest.fixture(autouse=True)
def restore_clock():
    yield
    set_clock(LiveClock())


def _df(rows: dict, index: list[str]) -> pd.DataFrame:
    return pd.DataFrame(rows, index=pd.to_datetime(index))


class FakeFastInfo(dict):
    def __getattr__(self, name):
        return self[name]


class FakeTicker:
    def __init__(
        self,
        history_df: pd.DataFrame | None = None,
        fast_info: dict | None = None,
        info: dict | None = None,
    ):
        self._history_df = history_df if history_df is not None else pd.DataFrame()
        self.fast_info = FakeFastInfo(fast_info or {})
        self._info = info or {}

    def history(self, *args, **kwargs):
        return self._history_df

    def get_info(self):
        return self._info


@pytest.fixture(autouse=True)
def _clear_unknown_ticker_cache():
    """market_data._UNKNOWN_TICKER_CACHE is a module-level dict shared
    across calls (by design, for the TTL behavior under test below) — clear
    it before and after every test so one test's cached failure can't leak
    into an unrelated one."""
    market_data._UNKNOWN_TICKER_CACHE.clear()
    yield
    market_data._UNKNOWN_TICKER_CACHE.clear()


@pytest.fixture
def fake_ticker_factory(monkeypatch):
    holder = {}

    def _install(ticker_obj: FakeTicker):
        holder["obj"] = ticker_obj
        monkeypatch.setattr(yf, "Ticker", lambda symbol: holder["obj"])

    return _install


async def test_get_last_close_reads_final_row(fake_ticker_factory):
    df = _df({"Close": [100.0, 105.0]}, ["2026-07-14", "2026-07-15"])
    fake_ticker_factory(FakeTicker(history_df=df))
    result = await market_data.get_last_close("AAPL")
    assert result == 105.0


async def test_get_atr_pct_computes_from_history(fake_ticker_factory):
    idx = pd.date_range("2026-06-01", periods=20, freq="D")
    df = pd.DataFrame(
        {"High": [101.0] * 20, "Low": [99.0] * 20, "Close": [100.0] * 20},
        index=idx,
    )
    fake_ticker_factory(FakeTicker(history_df=df))
    result = await market_data.get_atr_pct("AAPL")
    assert result == pytest.approx(0.02, abs=0.001)


def test_atr_matches_talibs_wilder_smoothed_value_exactly():
    """Golden-value test (2026-08-21, ATR TA-Lib PoC, commit 3/3 — see
    qlib-poc-decision-2026-08-21 project memory): `_atr` must keep
    matching `talib.ATR` bit-for-bit, the exact value sizing/stop-loss
    code consumes as ground truth."""
    vals = [
        100, 102, 101, 103, 102, 104, 103, 105, 104, 106, 105, 107, 106, 108,
        107, 109, 108, 110, 109, 111, 110, 112, 111, 113, 112, 114, 113, 115, 114, 116,
    ]
    closes = pd.Series((vals * 3)[:70], dtype=float)
    df = pd.DataFrame({"High": closes + 1.0, "Low": closes - 1.0, "Close": closes})
    assert market_data._atr(df, window=14) == pytest.approx(0.02951215331541526)


def test_atr_has_no_cliff_when_a_volatility_spike_ages_out_of_the_window():
    """Documents WHY this changed, not just that it did (2026-08-21):
    the rolling-mean ATR this replaced is exactly as sensitive to the
    OLDEST bar leaving its window as to the newest one entering it, so a
    volatility spike that happened `window` days ago causes ATR%% to drop
    off a cliff on the day it ages out — with zero new market
    information. Measured on real data: 13-23 such single-day drops of
    >=10%% per ticker per 3 years (see this commit's own project memory).
    Wilder's exponential decay has no such cliff — this asserts the day-
    over-day ATR%% change never drops more than 10%% anywhere in a series
    built specifically to trigger the old cliff (one large spike, then
    quiet bars past the point the old SMA would have dropped it)."""
    n = 40
    close = pd.Series([100.0 + 0.3 * ((i % 3) - 1) for i in range(n)])
    high = close + 0.5
    low = close - 0.5
    spike_day = 10
    high.iloc[spike_day] += 8.0
    low.iloc[spike_day] -= 8.0
    df = pd.DataFrame({"High": high, "Low": low, "Close": close})

    atr_series = [market_data._atr(df.iloc[: i + 1], window=14) for i in range(14, n)]
    changes = [b / a - 1 for a, b in zip(atr_series, atr_series[1:]) if a and b]
    assert min(changes) > -0.10


def test_atr_converges_as_more_history_feeds_it():
    """Unlike the old rolling-mean ATR (window-length-invariant beyond its
    own window), Wilder's value keeps drifting toward its converged
    figure as more history feeds it — this is WHY get_atr_pct's own
    lookback window had to widen (commit 2) ahead of this formula switch:
    a too-short history gives a measurably different (less converged)
    ATR than a longer one on the exact same trailing bars."""
    idx = pd.date_range("2026-01-01", periods=90, freq="D")
    highs = [100.0 + (i % 7) for i in range(90)]
    lows = [95.0 + (i % 5) for i in range(90)]
    closes = [98.0 + (i % 6) for i in range(90)]
    df = pd.DataFrame({"High": highs, "Low": lows, "Close": closes}, index=idx)

    short = market_data._atr(df.tail(20), window=14)
    long = market_data._atr(df, window=14)  # same trailing bars, much more history feeding the average
    assert short != pytest.approx(long)  # they now genuinely differ...
    assert short == pytest.approx(long, rel=0.15)  # ...but not wildly, both near-converged


async def test_get_atr_pct_returns_none_not_nan_for_insufficient_history(fake_ticker_factory):
    """2026-08-21 fix (ATR TA-Lib PoC review): fewer than `window` (14) rows
    used to leave `_atr` returning a bare NaN float, which silently defeats
    every downstream `atr_pct <= 0` guard (NaN comparisons are always
    False) — see app.chat.screening.risk_adjusted_score's own docstring."""
    idx = pd.date_range("2026-06-01", periods=5, freq="D")
    df = pd.DataFrame({"High": [101.0] * 5, "Low": [99.0] * 5, "Close": [100.0] * 5}, index=idx)
    fake_ticker_factory(FakeTicker(history_df=df))
    result = await market_data.get_atr_pct("AAPL")
    assert result is None


# ---------------------------------------------------------------------------
# get_atr_pct — 2026-09-11 still-forming-bar fix, yfinance path. Mirrors
# kis_provider.get_atr_pct's own fix (see that function's comment for the
# full measured rationale): during a live session the newest yfinance row is
# TODAY'S STILL-FORMING bar, which understated Wilder's ATR — fixed by
# dropping that row from the series fed to talib.ATR while keeping its Close
# as the denominator.
# ---------------------------------------------------------------------------


def _atr_df(n: int, *, last_index: str) -> pd.DataFrame:
    idx = list(pd.date_range("2026-08-01", periods=n - 1, freq="D")) + [pd.Timestamp(last_index)]
    highs = [101.0 + i * 0.15 for i in range(n)]
    lows = [99.0 + i * 0.15 for i in range(n)]
    closes = [100.0 + i * 0.15 for i in range(n)]
    # The still-forming (final) row gets a much TIGHTER high/low than every
    # completed row — a partial day's range genuinely is narrower than a
    # full one — so its True Range differs sharply from the rest of the
    # series. On a fixture with a constant per-row True Range, Wilder's ATR
    # is already fully converged and dropping one more identical row changes
    # nothing, which would make the fixture unable to distinguish the fix
    # from the old behaviour.
    highs[-1] = closes[-1] + 0.05
    lows[-1] = closes[-1] - 0.05
    return pd.DataFrame({"High": highs, "Low": lows, "Close": closes}, index=pd.DatetimeIndex(idx))


def _talib_atr_pct(df: pd.DataFrame, denominator: float) -> float:
    atr = talib.ATR(
        df["High"].to_numpy(dtype=float), df["Low"].to_numpy(dtype=float), df["Close"].to_numpy(dtype=float),
        timeperiod=14,
    )[-1]
    return float(atr / denominator)


async def test_get_atr_pct_us_ticker_in_progress_session_excludes_forming_bar(fake_ticker_factory):
    # ET 12:00 is inside the US regular session (9:30-16:00) — the newest row
    # (dated "today", 2026-09-11) must be judged still-forming.
    set_clock(FakeClock(datetime(2026, 9, 11, 12, 0, tzinfo=_US_TZ)))
    df = _atr_df(20, last_index="2026-09-11")
    fake_ticker_factory(FakeTicker(history_df=df))
    result = await market_data.get_atr_pct("AAPL")

    trimmed = df.iloc[:-1]
    last_close = float(df.iloc[-1]["Close"])
    expected = _talib_atr_pct(trimmed, last_close)
    assert result == pytest.approx(expected)

    # Must NOT match the OLD code's value: ATR over ALL rows (including the
    # still-forming one) divided by the same last close.
    old_value = _talib_atr_pct(df, last_close)
    assert result != pytest.approx(old_value)


async def test_get_atr_pct_us_ticker_outside_market_hours_is_byte_identical_to_before(fake_ticker_factory):
    # ET 20:00 is after the US close (16:00) — even though the newest row is
    # still dated "today", the session is not in progress, so the output
    # must be exactly the pre-fix computation (all rows, no trimming).
    set_clock(FakeClock(datetime(2026, 9, 11, 20, 0, tzinfo=_US_TZ)))
    df = _atr_df(20, last_index="2026-09-11")
    fake_ticker_factory(FakeTicker(history_df=df))
    result = await market_data.get_atr_pct("AAPL")

    last_close = float(df.iloc[-1]["Close"])
    expected = _talib_atr_pct(df, last_close)
    assert result == pytest.approx(expected)


async def test_get_atr_pct_past_as_of_date_is_unaffected_even_during_market_hours(fake_ticker_factory):
    # A backtest as_of in the past must never be treated as in-progress,
    # regardless of what the wall clock says right now. _history's as_of
    # branch always ends with `end = as_of + 1 day`, so the last row lands on
    # as_of itself.
    set_clock(FakeClock(datetime(2026, 9, 11, 12, 0, tzinfo=_US_TZ)))
    as_of = date(2026, 8, 20)
    df = _atr_df(20, last_index=as_of.isoformat())
    fake_ticker_factory(FakeTicker(history_df=df))
    result = await market_data.get_atr_pct("AAPL", as_of=as_of)

    last_close = float(df.iloc[-1]["Close"])
    expected = _talib_atr_pct(df, last_close)
    assert result == pytest.approx(expected)


async def test_get_price_series_maps_dates_to_closes(fake_ticker_factory):
    df = _df({"Close": [100.0, 110.0]}, ["2026-07-01", "2026-07-02"])
    fake_ticker_factory(FakeTicker(history_df=df))
    result = await market_data.get_price_series("AAPL", date(2026, 7, 1), date(2026, 7, 2))
    assert result == {date(2026, 7, 1): 100.0, date(2026, 7, 2): 110.0}


async def test_get_price_series_raises_on_empty_history(fake_ticker_factory):
    fake_ticker_factory(FakeTicker(history_df=pd.DataFrame()))
    with pytest.raises(ValueError, match="no price history returned"):
        await market_data.get_price_series("AAPL", date(2026, 7, 1), date(2026, 7, 2))


async def test_get_price_series_caches_a_ticker_with_no_history_and_skips_retry(fake_ticker_factory):
    """2026-09-02: a ticker yfinance has never heard of used to burn the
    full 3-attempt exponential-backoff retry on EVERY call — a 12-month
    screening_sweep A/B run (49 trade dates) meant the same bad ticker
    retried ~1.5s of backoff, 49 times, drowning the sweep's own progress
    log in tracebacks. The second call within the TTL must skip _run()
    entirely (no new yfinance call, no sleep) and raise immediately."""
    calls = {"n": 0}

    class CountingTicker(FakeTicker):
        def history(self, *args, **kwargs):
            calls["n"] += 1
            return self._history_df

    fake_ticker_factory(CountingTicker(history_df=pd.DataFrame()))

    with pytest.raises(ValueError, match="no price history returned"):
        await market_data.get_price_series("BADTICKER", date(2026, 7, 1), date(2026, 7, 2))
    calls_after_first = calls["n"]
    assert calls_after_first == 3  # exhausted every _retry_to_thread attempt

    with pytest.raises(ValueError, match="cached failure, not retried"):
        await market_data.get_price_series("BADTICKER", date(2026, 7, 3), date(2026, 7, 4))
    assert calls["n"] == calls_after_first  # served from cache — no new yfinance call, no retry sleep


async def test_get_price_series_cache_expires_after_ttl(fake_ticker_factory, monkeypatch):
    """A cached failure must not become a permanent blacklist — a newly-
    listed ticker yfinance hasn't indexed yet can start returning real data
    later, and this is a shared module-level cache used by the always-on
    live trading process, not just a one-shot sweep."""
    calls = {"n": 0}

    class CountingTicker(FakeTicker):
        def history(self, *args, **kwargs):
            calls["n"] += 1
            return self._history_df

    fake_ticker_factory(CountingTicker(history_df=pd.DataFrame()))

    with pytest.raises(ValueError):
        await market_data.get_price_series("BADTICKER", date(2026, 7, 1), date(2026, 7, 2))
    calls_after_first = calls["n"]

    monkeypatch.setitem(
        market_data._UNKNOWN_TICKER_CACHE, "BADTICKER",
        time.monotonic() - market_data._UNKNOWN_TICKER_TTL_SECONDS - 1,
    )
    with pytest.raises(ValueError, match="no price history returned"):
        await market_data.get_price_series("BADTICKER", date(2026, 7, 3), date(2026, 7, 4))
    assert calls["n"] > calls_after_first  # TTL expired — retried for real, not served from cache


async def test_get_daily_low_series_maps_dates_to_lows(fake_ticker_factory):
    df = _df({"Low": [98.0, 108.0]}, ["2026-07-01", "2026-07-02"])
    fake_ticker_factory(FakeTicker(history_df=df))
    result = await market_data.get_daily_low_series("AAPL", date(2026, 7, 1), date(2026, 7, 2))
    assert result == {date(2026, 7, 1): 98.0, date(2026, 7, 2): 108.0}


async def test_get_daily_high_series_maps_dates_to_highs(fake_ticker_factory):
    df = _df({"High": [102.0, 115.0]}, ["2026-07-01", "2026-07-02"])
    fake_ticker_factory(FakeTicker(history_df=df))
    result = await market_data.get_daily_high_series("AAPL", date(2026, 7, 1), date(2026, 7, 2))
    assert result == {date(2026, 7, 1): 102.0, date(2026, 7, 2): 115.0}


async def test_get_daily_high_series_raises_when_history_empty(fake_ticker_factory):
    fake_ticker_factory(FakeTicker(history_df=_df({}, [])))
    with pytest.raises(ValueError):
        await market_data.get_daily_high_series("AAPL", date(2026, 7, 1), date(2026, 7, 2))


async def test_get_live_quote_uses_second_to_last_row_to_avoid_forming_bar(fake_ticker_factory):
    idx = pd.date_range("2026-07-16 09:30", periods=3, freq="min")
    df = pd.DataFrame({"Close": [100.0, 101.0, 102.0], "Volume": [500, 600, 0]}, index=idx)
    fake_ticker_factory(FakeTicker(history_df=df))
    quote = await market_data.get_live_quote("AAPL")
    assert quote.price == 101.0
    assert quote.volume == 600


async def test_get_intraday_series_excludes_final_forming_bar(fake_ticker_factory):
    idx = pd.date_range("2026-07-16 09:30", periods=3, freq="min")
    df = pd.DataFrame({"Close": [100.0, 101.0, 102.0], "Volume": [500, 600, 0]}, index=idx)
    fake_ticker_factory(FakeTicker(history_df=df))
    series = await market_data.get_intraday_series("AAPL")
    assert len(series) == 2
    assert series[-1].price == 101.0


async def test_get_ticker_currency_is_cached(fake_ticker_factory, monkeypatch):
    # AAPL, not a KRX ticker, so this actually exercises the network-backed
    # cache path rather than the .KS/.KQ short-circuit below.
    market_data._currency_cache.clear()
    fake_ticker_factory(FakeTicker(fast_info={"currency": "USD"}))
    first = await market_data.get_ticker_currency("AAPL")
    # Swap in a ticker that would error if called again — cache should prevent a second lookup.
    monkeypatch.setattr(yf, "Ticker", lambda symbol: (_ for _ in ()).throw(AssertionError("should be cached")))
    second = await market_data.get_ticker_currency("AAPL")
    assert first == second == "USD"


async def test_get_ticker_currency_short_circuits_krx_ks_without_network(monkeypatch):
    market_data._currency_cache.clear()
    monkeypatch.setattr(yf, "Ticker", lambda symbol: (_ for _ in ()).throw(AssertionError("no network call for KRX")))
    assert await market_data.get_ticker_currency("005930.KS") == "KRW"


async def test_get_ticker_currency_short_circuits_krx_kq_without_network(monkeypatch):
    market_data._currency_cache.clear()
    monkeypatch.setattr(yf, "Ticker", lambda symbol: (_ for _ in ()).throw(AssertionError("no network call for KRX")))
    assert await market_data.get_ticker_currency("123456.KQ") == "KRW"


async def test_get_ticker_currency_non_krx_still_goes_through_yfinance(fake_ticker_factory):
    market_data._currency_cache.clear()
    fake_ticker_factory(FakeTicker(fast_info={"currency": "USD"}))
    assert await market_data.get_ticker_currency("AAPL") == "USD"


async def test_get_ticker_currency_non_digit_ks_code_does_not_short_circuit(fake_ticker_factory):
    # Proves the digit check is load-bearing — a .KS ticker whose code isn't
    # all-digits must not be treated as KRX and must still hit yfinance.
    market_data._currency_cache.clear()
    fake_ticker_factory(FakeTicker(fast_info={"currency": "USD"}))
    assert await market_data.get_ticker_currency("FOO.KS") == "USD"


async def test_get_ticker_currency_raises_on_none_and_does_not_cache(fake_ticker_factory):
    market_data._currency_cache.clear()
    fake_ticker_factory(FakeTicker(fast_info={"currency": None}))
    with pytest.raises(ValueError):
        await market_data.get_ticker_currency("AAPL")
    assert "AAPL" not in market_data._currency_cache


async def test_get_ticker_currency_raises_on_malformed_value_and_does_not_cache(fake_ticker_factory):
    market_data._currency_cache.clear()
    for bad_value in ("usd", "US", ""):
        fake_ticker_factory(FakeTicker(fast_info={"currency": bad_value}))
        with pytest.raises(ValueError):
            await market_data.get_ticker_currency("AAPL")
        assert "AAPL" not in market_data._currency_cache


async def test_get_usd_fx_rate_returns_one_for_usd_without_network(monkeypatch):
    monkeypatch.setattr(yf, "Ticker", lambda symbol: (_ for _ in ()).throw(AssertionError("no call expected for USD")))
    assert await market_data.get_usd_fx_rate("USD") == 1.0


async def test_get_usd_fx_rate_fetches_and_caches_non_usd(fake_ticker_factory):
    market_data._fx_rate_cache.clear()
    df = _df({"Close": [1450.0, 1460.0]}, ["2026-07-15", "2026-07-16"])
    fake_ticker_factory(FakeTicker(history_df=df))
    rate = await market_data.get_usd_fx_rate("KRW")
    assert rate == 1460.0


async def test_get_usd_fx_rate_reuses_cache_within_ttl(fake_ticker_factory, monkeypatch):
    market_data._fx_rate_cache.clear()
    df = _df({"Close": [1460.0]}, ["2026-07-16"])
    fake_ticker_factory(FakeTicker(history_df=df))
    await market_data.get_usd_fx_rate("KRW")
    # A second call within the TTL must not hit the network again.
    monkeypatch.setattr(yf, "Ticker", lambda symbol: (_ for _ in ()).throw(AssertionError("should be cached")))
    assert await market_data.get_usd_fx_rate("KRW") == 1460.0


async def test_get_usd_fx_rate_refetches_after_ttl_expires(fake_ticker_factory, monkeypatch):
    # Regression: this cache used to live for the whole process — fine for a
    # short backtest run, but the live daemon can run for days with a stale
    # rate otherwise.
    market_data._fx_rate_cache.clear()
    market_data._fx_rate_cache["KRW"] = (1400.0, time.monotonic() - market_data._FX_RATE_TTL_SECONDS - 1)
    df = _df({"Close": [1500.0]}, ["2026-07-16"])
    fake_ticker_factory(FakeTicker(history_df=df))
    rate = await market_data.get_usd_fx_rate("KRW")
    assert rate == 1500.0


async def test_get_ticker_display_info_prefers_long_name(fake_ticker_factory):
    fake_ticker_factory(
        FakeTicker(
            fast_info={"previousClose": 150.0, "currency": "USD"},
            info={"longName": "Apple Inc.", "shortName": "Apple"},
        )
    )
    info = await market_data.get_ticker_display_info("AAPL")
    assert info.name == "Apple Inc."
    assert info.previous_close == 150.0
    assert info.currency == "USD"


async def test_get_ticker_display_info_falls_back_to_short_name(fake_ticker_factory):
    fake_ticker_factory(
        FakeTicker(fast_info={"previousClose": 100.0, "currency": "USD"}, info={"shortName": "Foo Co"})
    )
    info = await market_data.get_ticker_display_info("FOO")
    assert info.name == "Foo Co"


async def test_get_ticker_display_info_exposes_short_name_alongside_long_name(fake_ticker_factory):
    # 2026-08-28: short_name is a SEPARATE field from name (longName-or-
    # shortName) — app.orchestrator.pipeline feeds both into identity_gate's
    # weak-alias set, since a report can match one but not the other.
    fake_ticker_factory(
        FakeTicker(
            fast_info={"previousClose": 150.0, "currency": "KRW"},
            info={"longName": "Hyosung Heavy Industries Corporation", "shortName": "HYOSUNG HEAVY"},
        )
    )
    info = await market_data.get_ticker_display_info("298040.KS")
    assert info.name == "Hyosung Heavy Industries Corporation"
    assert info.short_name == "HYOSUNG HEAVY"


async def test_get_ticker_display_info_blanks_an_implausible_short_name(fake_ticker_factory):
    fake_ticker_factory(
        FakeTicker(
            fast_info={"previousClose": 150.0, "currency": "KRW"},
            info={"longName": "Hyosung Heavy Industries Corporation", "shortName": "298040.KS,0P,123456"},
        )
    )
    info = await market_data.get_ticker_display_info("298040.KS")
    assert info.short_name == ""


async def test_get_ticker_display_info_blanks_quote_type_for_an_implausible_short_name(fake_ticker_factory):
    # Real incident (2026-09-03): 228340.KS "동양파일" and 142760.KS
    # "모아라이프플러스" are both ordinary KOSDAQ stocks, but yfinance
    # returned quoteType "MUTUALFUND" for both. Unlike the 058610.KS/
    # 218150.KS incidents above, longName itself was fine here ("Tongyang
    # Pile Inc") — only shortName carried the garbled
    # "ticker,internal-id,price" tell. quote_type must still be blanked
    # (routing it through is_unsupported_product_type's "empty never
    # matches" convention) even though the plausible longName is kept.
    fake_ticker_factory(
        FakeTicker(
            fast_info={"previousClose": 5000.0, "currency": "KRW"},
            info={
                "longName": "Tongyang Pile Inc",
                "shortName": "228340.KS,0P00017R1H,41744",
                "quoteType": "MUTUALFUND",
            },
        )
    )
    info = await market_data.get_ticker_display_info("228340.KS")
    assert info.name == "Tongyang Pile Inc"
    assert info.quote_type == ""
    assert info.short_name == ""


async def test_get_ticker_display_info_exposes_quote_type(fake_ticker_factory):
    fake_ticker_factory(
        FakeTicker(
            fast_info={"previousClose": 150.0, "currency": "USD"},
            info={"longName": "Apple Inc.", "quoteType": "EQUITY"},
        )
    )
    info = await market_data.get_ticker_display_info("AAPL")
    assert info.quote_type == "EQUITY"


async def test_get_ticker_display_info_discards_a_comma_glued_implausible_name(fake_ticker_factory):
    # Real incident: yfinance's own shortName for 218150.KS (quoteType
    # MUTUALFUND) came back as "218150.KS,0P000168Z2,87763" - a ticker+
    # internal-id+price string glued together on Yahoo's own side. Falls
    # back to the bare ticker rather than propagating that into the card
    # title / AI prompt.
    #
    # Second real incident (2026-08-03, 058610.KS — an ordinary KOSDAQ
    # stock, not a fund): the same garbled-response pattern carried
    # quoteType "MUTUALFUND" too, which fed into
    # app.domain.leverage_filter.is_unsupported_product_type and blocked a
    # legitimate BUY. quote_type now gets discarded right alongside the
    # name — it came from the same untrustworthy response.
    fake_ticker_factory(
        FakeTicker(
            fast_info={"previousClose": 87763.0, "currency": "KRW"},
            info={"shortName": "218150.KS,0P000168Z2,87763", "quoteType": "MUTUALFUND"},
        )
    )
    info = await market_data.get_ticker_display_info("218150.KS")
    assert info.name == "218150.KS"
    assert info.quote_type == ""


async def test_get_ticker_display_info_accepts_a_plausible_name_with_no_comma(fake_ticker_factory):
    fake_ticker_factory(
        FakeTicker(
            fast_info={"previousClose": 18410.0, "currency": "KRW"},
            info={"shortName": "Shinhan Sol Ai Semiconductor Top2 Plus Etf", "quoteType": "ETF"},
        )
    )
    info = await market_data.get_ticker_display_info("0167A0.KS")
    assert info.name == "Shinhan Sol Ai Semiconductor Top2 Plus Etf"


async def test_get_ticker_display_info_accepts_a_real_name_with_one_comma(fake_ticker_factory):
    # Real incident (2026-08-04): AXTI's genuine legal name "AXT, Inc." was
    # discarded by the old "any comma = implausible" check, leaving the
    # ticker ungrounded — the model then guessed a wrong company
    # ("Axiomtek") from the bare symbol alone. A single comma before a
    # corporate suffix is an ordinary legal-name shape, not the
    # ticker+internal-id+price glued shape the guard exists to catch (that
    # shape always has two commas — see the MUTUALFUND tests above).
    fake_ticker_factory(
        FakeTicker(
            fast_info={"previousClose": 66.30, "currency": "USD"},
            info={"longName": "AXT, Inc.", "quoteType": "EQUITY"},
        )
    )
    info = await market_data.get_ticker_display_info("AXTI")
    assert info.name == "AXT, Inc."
