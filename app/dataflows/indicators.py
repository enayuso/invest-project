"""Vendor-neutral technical-indicator math (TA-Lib-backed), shared by
`app.dataflows.yfinance.research` and `app.dataflows.kis_provider` so the
KRX and US/yfinance data paths compute RSI/SMA/MACD identically instead of
each carrying its own inline formula. Lives here (not inside the yfinance
package) so kis_provider — a KRX-first data source — doesn't have to
depend on a US-vendor package just to reuse a pure-math helper; before
this module existed, `_rsi` lived in `app.dataflows.yfinance.research` and
`kis_provider` imported it under an alias (`_compute_rsi`), which was a
dependency pointing the wrong way for a KIS-first module.

Every function here takes a `pandas.Series` of closes and returns plain
`float`/`None` (never NaN) — callers format for display, this module only
computes."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import talib


def _closes_array(closes: pd.Series) -> np.ndarray:
    # talib needs a contiguous float64 array, not a pandas Series.
    return closes.to_numpy(dtype=float)


def rsi(closes: pd.Series, window: int = 14) -> float:
    """Wilder-smoothed RSI via TA-Lib (2026-08-21, OSS PoC — see
    oss-candidates-license-review-2026-08-21 project memory). Replaces a
    homegrown simple-rolling-mean variant (a.k.a. Cutler's RSI) this used
    before: every other platform an analyst (human or this system's own
    LLM agents) would compare against — TradingView, Bloomberg, the
    textbook definition — reports Wilder's exponential smoothing instead,
    so the old value silently disagreed with "RSI 70" read anywhere else,
    on every ticker, all the time."""
    return float(talib.RSI(_closes_array(closes), timeperiod=window)[-1])


def sma(closes: pd.Series, window: int) -> float | None:
    """Simple moving average. `None` (not NaN) when there isn't enough
    history yet — pandas' own `.rolling(window).mean()` agreed with
    `talib.SMA` to ~1e-14 in practice (both are a plain windowed mean),
    so this switch is about having ONE shared implementation for SMA/RSI/
    MACD, not a numeric correction the way RSI's was."""
    value = talib.SMA(_closes_array(closes), timeperiod=window)[-1]
    return None if np.isnan(value) else float(value)


def macd(
    closes: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[float | None, float | None, float | None]:
    """MACD line, signal line, and histogram (line - signal) — replaces a
    homegrown `ewm(adjust=False)` line-only calculation (2026-08-21, TA-Lib
    OSS PoC, see oss-candidates-license-review-2026-08-21 project memory).

    The old code already used `adjust=False`, avoiding pandas' `adjust=True`
    trap, but its EMA seeds from the single first close; `talib.MACD` seeds
    each EMA from a plain SMA over its first `timeperiod` bars. That seed
    difference decays exponentially, so on a long history (yfinance's own
    `_yf_indicators`, ~125 daily bars) the two agree to a fraction of a
    cent — but on `_kis_indicators`'s shorter ~65-70-bar window
    (`_MAX_KIS_CANDLE_RANGE_DAYS=100` calendar days), it's real: measured
    ~10-40 KRW on a ~70,000 KRW stock, large enough to move the `.2f`
    figure an AI analyst prompt quotes. Narrowing that gap further would
    mean widening the KIS candle window (a separate, unstarted change, not
    part of this PoC) — this function just makes both paths use the same
    (now industry-standard) formula going forward.

    Returns `(None, None, None)` when there's too little history for the
    signal line to be defined yet (`talib.MACD`'s own warm-up: the MACD
    line needs `slow` bars, the signal line needs `slow + signal - 1` —
    33 bars before any value is non-NaN at the defaults). Both callers
    already guard SMA(50)'s own 50-bar minimum, which covers this."""
    macd_line, signal_line, hist = talib.MACD(
        _closes_array(closes), fastperiod=fast, slowperiod=slow, signalperiod=signal
    )
    latest = (macd_line[-1], signal_line[-1], hist[-1])
    return tuple(None if np.isnan(v) else float(v) for v in latest)


def atr_pct(
    high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14, *, denominator_close: float | None = None
) -> float | None:
    """Average True Range as a fraction of the latest close, via TA-Lib's
    Wilder smoothing (2026-08-21, ATR TA-Lib PoC — see
    qlib-poc-decision-2026-08-21 project memory) — replaces a homegrown
    simple-rolling-mean True Range that both `app.execution.sizing`'s
    volatility-adjusted position sizing and `app.execution.ledger.
    resolve_stop_loss`'s stop-loss floor relied on.

    Not a formula-standardization change like RSI's (Wilder IS the
    original 1978 ATR definition; the old rolling mean was the non-
    standard one, same category of drift as RSI's Cutler variant) — the
    real motivation is a measured defect: a rolling-mean ATR is exactly
    as sensitive to the OLDEST bar leaving its window as to the newest
    one entering it, so a volatility spike ages out and ATR%% can fall
    off a cliff (measured: 13-23 such >=10%% single-day drops per ticker
    per 3 years on real data) with no new market information at all —
    the exact "stop-loss distance changes for no reason" failure mode
    this codebase already has an incident history with (see
    live-trailing-stop-retroactive-breach-2026-08-14 project memory).
    Wilder's exponential decay has no such cliff (0-1 such drops/ticker/
    3yr in the same measurement). Mean level is preserved (measured mean
    ATR%% ratio 1.0047, median 0.9985) — this is a noise reduction, not a
    level shift, which is why `sizing_min_stop_atr_multiple` (2.5x),
    `trailing_stop_atr_multiplier` (1.25x), and `sizing_target_atr_pct`
    (0.02) are deliberately left untouched here.

    Unlike a rolling mean, Wilder's value keeps drifting toward its
    converged figure as more history feeds it (measured ~7%% off at 16
    bars, ~0.2%% by 60) — callers on a short history should widen their
    window rather than trust an early value the way they safely could
    with the old SMA (see `app.dataflows.kis_provider.get_atr_pct`'s own
    comment on why its window was widened ahead of this function landing).

    `None` (never NaN) when there's less than `window + 1` bars — same
    contract as `sma`/`macd` above, and the same reason
    `app.chat.screening.risk_adjusted_score` needs it (a NaN silently
    defeats its `<= 0` fallback check).

    `denominator_close` (2026-09-11, still-forming-bar ATR fix — see
    `app.dataflows.kis_provider.get_atr_pct`'s own comment): keyword-only,
    defaults to `None`, in which case behaviour is EXACTLY as before this
    parameter existed (`close.iloc[-1]` is the divisor). A caller that has
    already dropped today's still-forming bar from `high`/`low`/`close` (so
    the ATR itself is computed only from completed sessions) still wants the
    ratio expressed against the price an order would actually fill at right
    now, not against the last COMPLETED close — which, mid-session, can
    already be a stale, gapped-away price. When given, `denominator_close`
    replaces `close.iloc[-1]` as the divisor, with the same `!= 0 -> 0.0`
    guard applied to it."""
    atr = talib.ATR(_closes_array(high), _closes_array(low), _closes_array(close), timeperiod=window)[-1]
    if np.isnan(atr):
        return None
    latest_close = close.iloc[-1] if denominator_close is None else denominator_close
    return float(atr / latest_close) if latest_close != 0 else 0.0


def format_indicator_line(ticker: str, closes: pd.Series, *, source_suffix: str = "") -> str:
    """The `SMA(20)=.. SMA(50)=.. RSI(14)=.. MACD=.. (signal=.. hist=..)`
    prose both `_yf_indicators` and `_kis_indicators` return — shared so
    the two data paths can't drift into different formats or formulas
    again the way MACD's inline duplication let them. `source_suffix`
    (e.g. "source: KIS") reproduces `_kis_indicators`' distinguishing
    header; empty for the yfinance path, unchanged from before."""

    def fmt(value: float | None, decimals: int = 2) -> str:
        # `rsi()` keeps its pre-existing signature (always a float, never
        # None) for backward compatibility with direct callers/tests, so a
        # too-short history surfaces as NaN there rather than None — both
        # spellings of "no value yet" must render the same "n/a" here.
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return "n/a"
        return f"{value:.{decimals}f}"

    sma20 = sma(closes, 20)
    sma50 = sma(closes, 50)
    rsi14 = rsi(closes)
    macd_line, macd_signal, macd_hist = macd(closes)
    suffix = f" ({source_suffix})" if source_suffix else ""
    return (
        f"Technical indicators for {ticker}{suffix}:\n"
        f"SMA(20)={fmt(sma20)}  SMA(50)={fmt(sma50)}  RSI(14)={fmt(rsi14, 1)}  "
        f"MACD={fmt(macd_line)} (signal={fmt(macd_signal)} hist={fmt(macd_hist)})"
    )
