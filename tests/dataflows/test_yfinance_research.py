"""app.dataflows.yfinance.research — _rsi (pure) plus a few _yf_* providers
with yf.Ticker mocked via monkeypatch, no real network (B-stage)."""

from __future__ import annotations

from datetime import date, datetime
from zoneinfo import ZoneInfo

import pandas as pd
import pytest
import yfinance as yf

from app.clock import FakeClock, LiveClock, set_clock
from app.dataflows.yfinance.research import (
    _rsi,
    _yf_balance_sheet,
    _yf_cashflow,
    _yf_fundamentals,
    _yf_indicators,
    _yf_long_horizon,
    _yf_news,
    _yf_stock_data,
    _yf_verified_snapshot,
)

_US_TZ = ZoneInfo("America/New_York")


@pytest.fixture(autouse=True)
def restore_clock():
    yield
    set_clock(LiveClock())


class FakeTicker:
    def __init__(
        self,
        history_df: pd.DataFrame | None = None,
        info: dict | None = None,
        news: list | None = None,
        balance_sheet: pd.DataFrame | None = None,
        cashflow: pd.DataFrame | None = None,
    ):
        self._history_df = history_df if history_df is not None else pd.DataFrame()
        self._info = info or {}
        self.news = news or []
        self.balance_sheet = balance_sheet if balance_sheet is not None else pd.DataFrame()
        self.cashflow = cashflow if cashflow is not None else pd.DataFrame()

    def history(self, *args, **kwargs):
        return self._history_df

    def get_info(self):
        return self._info


@pytest.fixture
def install_ticker(monkeypatch):
    def _install(ticker_obj: FakeTicker):
        monkeypatch.setattr(yf, "Ticker", lambda symbol: ticker_obj)

    return _install


def test_rsi_mostly_gains_is_high():
    # A single small dip keeps the rolling loss average non-zero (an
    # unbroken uptrend divides by a zero loss average and yields NaN).
    values = [100 + i for i in range(19)] + [117.5]
    closes = pd.Series(values, dtype=float)
    assert _rsi(closes) > 80.0


def test_rsi_all_losses_approaches_zero():
    closes = pd.Series([100 - i for i in range(20)], dtype=float)
    assert _rsi(closes) == pytest.approx(0.0, abs=0.5)


def test_rsi_matches_talibs_wilder_smoothed_value_exactly():
    """Golden-value test (2026-08-21, TA-Lib OSS PoC — see
    oss-candidates-license-review-2026-08-21 project memory): `_rsi` must
    keep matching `talib.RSI` bit-for-bit, not just "high" or "low" — this
    is the exact value an AI analyst prompt quotes as ground truth, and the
    two direction-only tests above wouldn't catch a regression to a
    different (if still plausible-looking) formula."""
    closes = pd.Series([100 + i for i in range(19)] + [117.5], dtype=float)
    assert _rsi(closes) == pytest.approx(96.2962962962963)


def test_rsi_diverges_from_the_old_cutler_variant_it_replaced():
    """Documents WHY this changed, not just that it did: the formula this
    replaced (simple rolling-mean gain/loss, a.k.a. Cutler's RSI) gives a
    materially different number from Wilder's smoothing (what TA-Lib, and
    every other platform an analyst would compare against, actually
    means by "RSI") on the same closes — this was a real, silent
    correctness gap, not a cosmetic library swap."""

    def _cutler_rsi(closes: pd.Series, window: int = 14) -> float:
        delta = closes.diff()
        gain = delta.clip(lower=0).rolling(window).mean()
        loss = (-delta.clip(upper=0)).rolling(window).mean()
        rs = gain / loss.replace(0, float("nan"))
        return float((100 - (100 / (1 + rs))).iloc[-1])

    # A choppy up-and-down series (not a clean monotonic run) — this is
    # where Wilder's recursive smoothing and Cutler's flat rolling mean
    # actually pull apart; a near-monotonic fixture happens to keep them
    # within a hair of each other for the first `window`-ish bars.
    vals = [
        100, 102, 101, 103, 102, 104, 103, 105, 104, 106, 105, 107, 106, 108,
        107, 109, 108, 110, 109, 111, 110, 112, 111, 113, 112, 114, 113, 115, 114, 116,
    ]
    closes = pd.Series(vals, dtype=float)
    assert _rsi(closes) != pytest.approx(_cutler_rsi(closes), abs=1.0)


# ---------------------------------------------------------------------------
# _yf_indicators — no coverage existed before 2026-08-21 (TA-Lib PoC
# extension: SMA/MACD moved to the shared app.dataflows.indicators module).
# ---------------------------------------------------------------------------


async def test_yf_indicators_reports_sma_rsi_macd_with_signal_and_histogram(install_ticker):
    vals = [
        100, 102, 101, 103, 102, 104, 103, 105, 104, 106, 105, 107, 106, 108,
        107, 109, 108, 110, 109, 111, 110, 112, 111, 113, 112, 114, 113, 115, 114, 116,
    ]
    closes = (vals * 3)[:70]
    idx = pd.date_range("2026-01-01", periods=len(closes), freq="D")
    df = pd.DataFrame({"Close": closes}, index=idx)
    install_ticker(FakeTicker(history_df=df))
    result = await _yf_indicators("AAPL")
    assert result.startswith("Technical indicators for AAPL:\n")
    assert "SMA(20)=" in result and "SMA(50)=" in result and "RSI(14)=" in result
    # MACD now carries signal/histogram alongside the line (2026-08-21) —
    # a bare "MACD=x.xx" number is uninterpretable on its own; the
    # crossover (line vs signal) is the actual signal an analyst reads.
    assert "MACD=" in result and "signal=" in result and "hist=" in result
    assert "nan" not in result.lower()


async def test_yf_indicators_renders_short_history_as_na_not_nan(install_ticker):
    idx = pd.date_range("2026-01-01", periods=10, freq="D")
    df = pd.DataFrame({"Close": [100.0 + i for i in range(10)]}, index=idx)
    install_ticker(FakeTicker(history_df=df))
    result = await _yf_indicators("AAPL")
    assert "nan" not in result.lower()
    assert "SMA(50)=n/a" in result


async def test_yf_verified_snapshot_formats_last_row(install_ticker):
    idx = pd.date_range("2026-07-10", periods=3, freq="D")
    df = pd.DataFrame(
        {"Open": [1, 2, 3], "High": [105.0, 106.0, 107.0], "Low": [95.0, 96.0, 97.0],
         "Close": [100.0, 101.0, 102.0], "Volume": [1000, 1100, 1200]},
        index=idx,
    )
    install_ticker(FakeTicker(history_df=df))
    result = await _yf_verified_snapshot("AAPL")
    assert "VERIFIED SNAPSHOT for AAPL" in result
    assert "close=102.00" in result
    assert "volume=1200" in result
    # 2026-09-10 dataflows audit stage 2: non-KRX yfinance gets a plain
    # marker, no ~2%-off caveat (that error was measured on KRX only).
    assert result.split("\n")[-1] == "source: yfinance"


async def test_yf_verified_snapshot_labels_in_progress_session_as_live(install_ticker):
    # Regression test for the MRVL incident (2026-08-26 09:42 ET, 12 minutes
    # after open) — see app.dataflows.session_state's module docstring. The
    # last row's date matches "today" under a FakeClock set to mid-session,
    # so this must come back labeled `last=`, not `close=`.
    set_clock(FakeClock(datetime(2026, 8, 26, 9, 42, tzinfo=_US_TZ)))
    idx = pd.date_range("2026-08-24", periods=3, freq="D")
    df = pd.DataFrame(
        {"Open": [240.0, 229.22, 240.10], "High": [251.73, 231.70, 240.77],
         "Low": [233.28, 220.42, 236.79], "Close": [237.04, 229.29, 237.04],
         "Volume": [25526600, 18791400, 769562]},
        index=idx,
    )
    install_ticker(FakeTicker(history_df=df))
    result = await _yf_verified_snapshot("MRVL")
    assert "SESSION IN PROGRESS" in result
    assert "last=237.04" in result
    assert "close=" not in result
    assert result.split("\n")[-1] == "source: yfinance"


async def test_yf_verified_snapshot_labels_closed_session_as_close(install_ticker):
    # Same data/date as above, but the FakeClock is set to after the US
    # close — must fall back to the ordinary `close=` label.
    set_clock(FakeClock(datetime(2026, 8, 26, 16, 30, tzinfo=_US_TZ)))
    idx = pd.date_range("2026-08-24", periods=3, freq="D")
    df = pd.DataFrame(
        {"Open": [240.0, 229.22, 240.10], "High": [251.73, 231.70, 245.17],
         "Low": [233.28, 220.42, 236.79], "Close": [237.04, 229.29, 245.11],
         "Volume": [25526600, 18791400, 15919680]},
        index=idx,
    )
    install_ticker(FakeTicker(history_df=df))
    result = await _yf_verified_snapshot("MRVL")
    assert "SESSION IN PROGRESS" not in result
    assert "close=245.11" in result


async def test_yf_stock_data_appends_footnote_when_session_in_progress(install_ticker):
    set_clock(FakeClock(datetime(2026, 8, 26, 9, 42, tzinfo=_US_TZ)))
    idx = pd.date_range("2026-08-24", periods=3, freq="D")
    df = pd.DataFrame(
        {"Open": [240.0, 229.22, 240.10], "High": [251.73, 231.70, 240.77],
         "Low": [233.28, 220.42, 236.79], "Close": [237.04, 229.29, 237.04],
         "Volume": [25526600, 18791400, 769562]},
        index=idx,
    )
    install_ticker(FakeTicker(history_df=df))
    result = await _yf_stock_data("MRVL")
    assert "STILL-OPEN" in result


async def test_yf_stock_data_no_footnote_when_session_closed(install_ticker):
    set_clock(FakeClock(datetime(2026, 8, 26, 16, 30, tzinfo=_US_TZ)))
    idx = pd.date_range("2026-08-24", periods=3, freq="D")
    df = pd.DataFrame(
        {"Open": [240.0, 229.22, 240.10], "High": [251.73, 231.70, 245.17],
         "Low": [233.28, 220.42, 236.79], "Close": [237.04, 229.29, 245.11],
         "Volume": [25526600, 18791400, 15919680]},
        index=idx,
    )
    install_ticker(FakeTicker(history_df=df))
    result = await _yf_stock_data("MRVL")
    assert "STILL-OPEN" not in result


async def test_yf_long_horizon_reports_52_week_range(install_ticker):
    idx = pd.date_range("2025-09-01", periods=380, freq="D")
    closes = [100.0 + i * 0.05 for i in range(380)]
    df = pd.DataFrame(
        {
            "Open": closes, "High": [c + 1 for c in closes], "Low": [c - 1 for c in closes],
            "Close": closes, "Volume": [1_000_000] * 380,
        },
        index=idx,
    )
    install_ticker(FakeTicker(history_df=df))
    result = await _yf_long_horizon("AAPL", as_of=date(2026, 9, 15))
    assert "Longer-horizon price context for AAPL" in result
    assert "source: yfinance" in result
    assert "52-week high:" in result and "52-week low:" in result
    assert "trailing returns from weekly closes:" in result


async def test_yf_long_horizon_raises_on_empty_history(install_ticker):
    install_ticker(FakeTicker(history_df=pd.DataFrame()))
    with pytest.raises(ValueError, match="no price history"):
        await _yf_long_horizon("AAPL")


async def test_yf_long_horizon_never_labels_a_bar_after_as_of(install_ticker):
    """2026-09-16 fix: W-FRI resample labels each bin by its right edge (the
    upcoming Friday), including the still-in-progress final week — an
    as_of-bounded backtest for a Wednesday used to print a week_end two days
    in the future. week_end must now be the last actual trading day the bar
    contains, never later than as_of."""
    as_of = date(2026, 9, 16)  # a Wednesday: the naive W-FRI right edge is 2026-09-18
    idx = pd.date_range("2025-09-01", periods=380, freq="D")
    idx = idx[idx <= pd.Timestamp(as_of)]
    closes = [100.0 + i * 0.05 for i in range(len(idx))]
    df = pd.DataFrame(
        {"Open": closes, "High": [c + 1 for c in closes], "Low": [c - 1 for c in closes],
         "Close": closes, "Volume": [1_000_000] * len(idx)},
        index=idx,
    )
    install_ticker(FakeTicker(history_df=df))
    result = await _yf_long_horizon("AAPL", as_of=as_of)
    window_line = next(ln for ln in result.splitlines() if "window:" in ln)
    last_labeled = date.fromisoformat(window_line.split("..")[-1].strip())
    assert last_labeled <= as_of


async def test_yf_fundamentals_live_mode_includes_valuation_keys(install_ticker):
    install_ticker(FakeTicker(
        info={
            "longName": "Apple Inc.", "trailingPE": 30.5, "marketCap": 3_000_000_000,
            "totalDebt": 98_660_000_000, "totalCash": 62_400_000_000,
        },
    ))
    result = await _yf_fundamentals("AAPL")
    # longName is deliberately never surfaced — see _yf_fundamentals's
    # docstring (086790.KS cross-symbol mismatch); the caller already gets a
    # verified name via ticker_label, so this tool must not offer a second,
    # occasionally-wrong one.
    assert "longName" not in result
    assert "trailingPE: 30.5" in result
    # totalDebt/totalCash are deliberately never surfaced either — see
    # _yf_fundamentals's docstring (get_info() vs get_balance_sheet mismatch,
    # confirmed live on AAPL/HOOD/AMGN/TXN/INTC — 2026-08-25 ids 963-997).
    assert "totalDebt" not in result
    assert "totalCash" not in result
    assert "get_balance_sheet" in result


async def test_yf_fundamentals_backtest_mode_omits_valuation_ratios(install_ticker):
    install_ticker(FakeTicker(info={"longName": "Apple Inc.", "sector": "Technology", "trailingPE": 30.5}))
    result = await _yf_fundamentals("AAPL", as_of=date(2026, 1, 1))
    assert "longName" not in result
    assert "sector: Technology" in result
    assert "trailingPE" not in result
    assert "not available point-in-time" in result


# ---------------------------------------------------------------------------
# _yf_balance_sheet / _yf_cashflow — 2026-07-30: pandas' default to_string()
# switches large floats to scientific notation (e.g. "4.422051e+12"), which
# a live reproduction (009540.KS) confirmed the LLM reliably mis-scales by
# 10x-100x when converting to prose. Regression coverage: large real-world-
# scale values must render as plain comma-separated integers, never in
# scientific-notation form ("e+" never appears).
# ---------------------------------------------------------------------------


async def test_yf_balance_sheet_formats_large_values_without_scientific_notation(install_ticker):
    idx = pd.date_range("2025-12-31", periods=1, freq="D")
    bs = pd.DataFrame({idx[0]: [4_220_490_000_000.0]}, index=["Total Assets"])
    install_ticker(FakeTicker(balance_sheet=bs))
    result = await _yf_balance_sheet("009540.KS")
    assert "e+" not in result
    # Pre-scaled to a T/B/M/K suffix (see _scaled_unit_str) rather than a
    # raw comma-separated integer — the model reads "4.220T" directly
    # instead of doing the trillion-conversion arithmetic itself.
    assert "4.220T" in result
    assert "4,220,490,000,000" not in result


async def test_yf_cashflow_formats_large_values_without_scientific_notation(install_ticker):
    idx = pd.date_range("2025-12-31", periods=1, freq="D")
    cf = pd.DataFrame({idx[0]: [4_422_051_000_000.0]}, index=["Operating Cash Flow"])
    install_ticker(FakeTicker(cashflow=cf))
    result = await _yf_cashflow("009540.KS")
    assert "e+" not in result
    assert "4.422T" in result
    assert "4,422,051,000,000" not in result


@pytest.mark.parametrize(
    "value,expected",
    [
        (4_220_490_000_000.0, "4.220T"),
        (456_131_791_000.0, "456.132B"),
        (5_500_000.0, "5.500M"),
        (12_500.0, "12.500K"),
        (999.0, "999"),
        (-4_220_490_000_000.0, "-4.220T"),
    ],
)
def test_scaled_unit_str_picks_correct_suffix(value, expected):
    from app.dataflows.yfinance.research import _scaled_unit_str

    assert _scaled_unit_str(value) == expected


def test_scaled_unit_str_passes_nan_through():
    from app.dataflows.yfinance.research import _scaled_unit_str

    assert _scaled_unit_str(float("nan")) == "NaN"


async def test_yf_balance_sheet_raises_on_empty(install_ticker):
    install_ticker(FakeTicker(balance_sheet=pd.DataFrame()))
    with pytest.raises(ValueError):
        await _yf_balance_sheet("AAPL")


async def test_yf_cashflow_raises_on_empty(install_ticker):
    install_ticker(FakeTicker(cashflow=pd.DataFrame()))
    with pytest.raises(ValueError):
        await _yf_cashflow("AAPL")


async def test_yf_news_live_mode_lists_recent_items(install_ticker):
    install_ticker(FakeTicker(news=[{"content": {"title": "Big news", "pubDate": "2026-07-16"}}]))
    result = await _yf_news("AAPL")
    assert "Recent news for AAPL:" in result
    assert "Big news" in result


async def test_yf_news_backtest_mode_returns_unavailable_message(install_ticker):
    install_ticker(FakeTicker(news=[{"content": {"title": "Should not appear"}}]))
    result = await _yf_news("AAPL", as_of=date(2026, 1, 1))
    assert "unavailable in point-in-time backtest mode" in result
    assert "Should not appear" not in result


async def test_yf_news_no_items_returns_no_news_message(install_ticker):
    install_ticker(FakeTicker(news=[]))
    result = await _yf_news("AAPL")
    assert result == "No recent news found for AAPL."
