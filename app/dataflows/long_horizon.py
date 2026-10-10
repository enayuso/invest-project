"""Vendor-neutral 52-week price-context formatting, shared by
`app.dataflows.yfinance.research` and `app.dataflows.kis_provider` so the
KRX and US/yfinance data paths render the same math from the same weekly-bar
shape — same reason `app.dataflows.indicators` exists (RSI/SMA/MACD must not
drift between vendors).

2026-09-16 (engine audit follow-up): Market Analyst's `get_stock_data` tool
schema advertised `lookback_days: default 180` but both data paths silently
rendered only the last 10 daily sessions regardless of that value — the
model was asked to write "3-month trend" / "52-week high" style prose with
no data behind it. This module supplies that missing longer-horizon context
from weekly bars (not by widening the daily table, which stays at 10 sessions
— see `app.experiments.candle_ab` for the in-flight daily-table A/B this must
not disturb)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class WeeklyBar:
    week_end: date
    high: float
    low: float
    close: float


# Below this, "52-week high/low" is not a meaningful claim (a newly-listed or
# soon-to-be-delisted ticker) — the caller degrades to its own NOT AVAILABLE
# fallback instead of labeling a 3-bar range a 52-week one.
_MIN_BARS = 12

# At/above this, the window really does span a year and "52-week" is a true
# label; below it, the block names its own actual span instead (2026-09-16
# fix — a 20-bar window used to be hardcoded as "52-week high/low" anyway,
# telling a recently-listed ticker's ~4-month high to the model as a 52-week
# extreme, right below a prompt instruction to quote these figures as given).
_FULL_WINDOW_BARS = 52

# (label, weeks-back) pairs for the trailing-return line, evaluated against
# weekly closes. A period with no bar that far back is rendered "n/a" rather
# than skipped, so the line's shape never changes across tickers.
_TRAILING_RETURNS = (("1M", 4), ("3M", 13), ("6M", 26), ("12M", 52))

_RECENT_WEEKS_SHOWN = 13


def _price_decimals(ticker: str) -> int:
    """KRX quotes are whole won — a 2-decimal rendering there is noise.
    Every other market this repo trades (US) is quoted in dollars and
    cents, where a blanket `:,.0f` destroyed the figure outright: a $3.47
    close rendered as "3", and a $3.20..$3.60 52-week range collapsed to
    "3 .. 4" (reproduced 2026-09-16). Duplicated here rather than imported
    from app.dataflows.kis_provider/yfinance.market_data's own `_is_krx*`
    helpers — this module is imported BY both of them, so importing either
    back would be circular — same convention as those modules' own copies."""
    for suffix in (".KS", ".KQ"):
        if ticker.endswith(suffix) and ticker[: -len(suffix)].isdigit():
            return 0
    return 2


def format_long_horizon_block(ticker: str, bars: list[WeeklyBar], *, source: str) -> str:
    """`bars` must be oldest-to-newest, at most 53 entries (the current/most
    recent week plus up to 52 prior). Raises `ValueError` below `_MIN_BARS`
    bars — callers catch this the same way they already catch a too-short
    daily history from `get_indicators`/`get_atr_pct`, degrading only this
    one block rather than the whole analyst report."""
    if len(bars) < _MIN_BARS:
        raise ValueError(f"{ticker}: only {len(bars)} weekly bars, need at least {_MIN_BARS} for long-horizon context")

    dp = _price_decimals(ticker)
    full_window = len(bars) >= _FULL_WINDOW_BARS
    range_label = "52-week" if full_window else f"{len(bars)}-week"
    short_window_note = (
        ""
        if full_window
        else (
            f"\n  note: only {len(bars)} weekly bars exist for this ticker (recent listing or "
            f"thin history) — the high/low above is a {len(bars)}-week extreme, NOT a 52-week one. "
            f"Do not describe it as a 52-week high, low, or range."
        )
    )

    # w52_high/w52_low are the WINDOW extreme — a true 52-week extreme only
    # when `full_window` (see range_label above).
    w52_high = max(b.high for b in bars)
    w52_high_week = next(b.week_end for b in bars if b.high == w52_high)
    w52_low = min(b.low for b in bars)
    w52_low_week = next(b.week_end for b in bars if b.low == w52_low)
    last_close = bars[-1].close

    pct_off_high = (last_close / w52_high - 1) * 100
    pct_above_low = (last_close / w52_low - 1) * 100

    return_parts = []
    for label, weeks_back in _TRAILING_RETURNS:
        idx = len(bars) - 1 - weeks_back
        if idx < 0:
            return_parts.append(f"{label} n/a")
            continue
        base_close = bars[idx].close
        pct = (last_close / base_close - 1) * 100 if base_close else None
        return_parts.append(f"{label} {pct:+.1f}%" if pct is not None else f"{label} n/a")

    recent = bars[-_RECENT_WEEKS_SHOWN:]
    recent_closes = " ".join(f"{b.close:,.{dp}f}" for b in recent)

    return (
        f"Longer-horizon price context for {ticker} ({len(bars)} weekly bars, source: {source}):\n"
        f"  window: {bars[0].week_end.isoformat()} .. {bars[-1].week_end.isoformat()}\n"
        f"  {range_label} high: {w52_high:,.{dp}f} (week ending {w52_high_week.isoformat()})\n"
        f"  {range_label} low: {w52_low:,.{dp}f} (week ending {w52_low_week.isoformat()})\n"
        f"  latest weekly close: {last_close:,.{dp}f} ({pct_off_high:+.1f}% from the {range_label} high, "
        f"{pct_above_low:+.1f}% above the {range_label} low)\n"
        f"  trailing returns from weekly closes: {' | '.join(return_parts)}\n"
        f"  weekly closes, oldest to newest (last {len(recent)}): {recent_closes}"
        f"{short_window_note}"
    )
