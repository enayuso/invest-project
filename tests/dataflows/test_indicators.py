"""app.dataflows.indicators — shared RSI/SMA/MACD math (2026-08-21 TA-Lib
PoC extension, see oss-candidates-license-review-2026-08-21 project
memory). `rsi` itself is covered by tests/dataflows/test_yfinance_research.py
(kept there since `research._rsi` re-exports it) — this file covers the
newer `sma`/`macd`/`format_indicator_line` additions."""

from __future__ import annotations

import pandas as pd
import pytest

from app.dataflows import indicators

# A choppy (not monotonic) oscillation, tiled and sliced to length —
# monotonic fixtures let an EMA-seeding difference hide for many bars (see
# `macd`'s own docstring on why a real disagreement needs an oscillating
# series to show up quickly).
_OSCILLATION = [
    100, 102, 101, 103, 102, 104, 103, 105, 104, 106, 105, 107, 106, 108,
    107, 109, 108, 110, 109, 111, 110, 112, 111, 113, 112, 114, 113, 115, 114, 116,
]


def _closes(n: int) -> pd.Series:
    return pd.Series((_OSCILLATION * ((n // len(_OSCILLATION)) + 1))[:n], dtype=float)


def test_sma_matches_the_plain_windowed_mean():
    closes = _closes(60)
    assert indicators.sma(closes, 20) == pytest.approx(closes.tail(20).mean())
    assert indicators.sma(closes, 50) == pytest.approx(closes.tail(50).mean())


def test_sma_is_none_when_history_is_shorter_than_the_window():
    assert indicators.sma(_closes(40), 50) is None


def test_macd_golden_value_on_a_70_bar_oscillation():
    """Locks in the exact TA-Lib MACD/signal/histogram for a fixed
    fixture — a regression here means the formula changed, not just that
    a "plausible" number moved."""
    macd_line, signal, hist = indicators.macd(_closes(70))
    assert macd_line == pytest.approx(-1.2628014691856748)
    assert signal == pytest.approx(-0.8522088942987023)
    assert hist == pytest.approx(-0.41059257488697254)
    assert hist == pytest.approx(macd_line - signal)


def test_macd_is_none_before_the_signal_lines_warmup():
    # slow(26) + signal(9) - 1 = 34 bars needed; 30 isn't enough.
    assert indicators.macd(_closes(30)) == (None, None, None)


def test_macd_becomes_defined_at_exactly_its_warmup_length():
    macd_line, signal, hist = indicators.macd(_closes(34))
    assert macd_line is not None
    assert signal is not None
    assert hist is not None


def test_macd_diverges_from_the_old_ewm_line_only_calculation_on_a_short_history():
    """Documents WHY this changed: the formula this replaced (`ewm(adjust=
    False)` seeded from the first close) and TA-Lib's SMA-seeded EMA agree
    to a fraction of a cent on a long history, but disagree by a real,
    `.2f`-visible amount on `_kis_indicators`'s ~65-70-bar window — see
    `macd`'s own docstring for the measured ~10-40 KRW gap this represents
    at KRW price scale."""
    closes = _closes(70) * 700  # KRW-like price scale (~70,000)

    def _old_macd_line(closes: pd.Series) -> float:
        ema12 = closes.ewm(span=12, adjust=False).mean()
        ema26 = closes.ewm(span=26, adjust=False).mean()
        return float((ema12 - ema26).iloc[-1])

    new_macd_line, _, _ = indicators.macd(closes)
    assert new_macd_line != pytest.approx(_old_macd_line(closes), abs=1.0)


def test_format_indicator_line_matches_the_established_prose_shape():
    result = indicators.format_indicator_line("AAPL", _closes(70))
    assert result.startswith("Technical indicators for AAPL:\n")
    assert "SMA(20)=" in result and "SMA(50)=" in result
    assert "RSI(14)=" in result
    assert "MACD=" in result and "signal=" in result and "hist=" in result


def test_format_indicator_line_includes_the_source_suffix_when_given():
    result = indicators.format_indicator_line("005930.KS", _closes(70), source_suffix="source: KIS")
    assert result.startswith("Technical indicators for 005930.KS (source: KIS):\n")


def test_format_indicator_line_renders_not_enough_history_as_na_not_nan():
    # 25 bars: enough for SMA(20)/RSI(14), but short of both SMA(50) and
    # MACD's 34-bar warm-up (see test_macd_is_none_before_the_signal_lines_
    # warmup) — so both come back None/NaN and must render as "n/a".
    result = indicators.format_indicator_line("AAPL", _closes(25))
    assert "nan" not in result.lower()
    assert "SMA(50)=n/a" in result
    assert "MACD=n/a (signal=n/a hist=n/a)" in result


# --- atr_pct's `denominator_close` (2026-09-11, still-forming-bar ATR fix) --


def _atr_ohlc(n: int) -> tuple[pd.Series, pd.Series, pd.Series]:
    close = _closes(n)
    return close + 1.0, close - 1.0, close


def test_atr_pct_denominator_close_overrides_the_divisor():
    high, low, close = _atr_ohlc(40)
    default_result = indicators.atr_pct(high, low, close, window=14)
    custom_result = indicators.atr_pct(high, low, close, window=14, denominator_close=50.0)
    assert default_result is not None and custom_result is not None
    # differ from each other (denominator actually changed the ratio)...
    assert custom_result != pytest.approx(default_result)
    # ...and the ATR itself (the numerator) is untouched by the override —
    # recovering it from the default result and re-dividing by the custom
    # denominator must reproduce the custom result exactly.
    atr_value = default_result * close.iloc[-1]
    assert custom_result == pytest.approx(atr_value / 50.0)


def test_atr_pct_without_denominator_close_is_unchanged():
    high, low, close = _atr_ohlc(40)
    assert indicators.atr_pct(high, low, close, window=14) == indicators.atr_pct(
        high, low, close, window=14, denominator_close=None
    )
