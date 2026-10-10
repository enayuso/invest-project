"""Shared helpers for the yfinance provider package — used by both
`research.py` (LLM-agent research providers) and `market_data.py` (internal
deterministic data feeds).

Every public function accepts an optional `as_of` date for point-in-time
backtesting (Phase 2). `as_of=None` (the default) means "live" — behaves
exactly as Phase 1. When set, each provider bounds itself to what was
actually knowable on that date:

- Price/technicals: `history(end=as_of)` — straightforward date truncation.
- Balance sheet / cashflow: yfinance exposes fiscal *period-end* dates, not
  actual public *filing* dates, so a period isn't included unless
  `period_end + _FILING_LAG_DAYS <= as_of` — a conservative stand-in for the
  real SEC filing lag (10-Qs/10-Ks aren't public the instant the quarter ends).
- Insider transactions: filtered by the transaction's own `Start Date`.
- `get_fundamentals`'s valuation ratios (P/E, market cap, div yield, ...) are
  computed from *today's* price/share count by yfinance and cannot be
  rebound to a past date at all — in backtest mode these are dropped
  entirely rather than silently reported as if they applied historically.
- `get_news` has no historical archive in yfinance — in backtest mode it
  returns an explicit "unavailable" string rather than leaking current
  headlines into a historical decision.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from datetime import date, timedelta
from typing import TypeVar

import pandas as pd
import yfinance as yf

logger = logging.getLogger("app.dataflows.yfinance")

_FILING_LAG_DAYS = 90
_HOLDINGS_LAG_DAYS = 45  # 13F filings are due 45 days after quarter-end

_CANDLE_CFG = {"D": ("1d", "1y"), "W": ("1wk", "2y"), "M": ("1mo", "10y"), "Y": ("3mo", "max")}

T = TypeVar("T")


async def _retry_to_thread(fn: Callable[[], T], *, attempts: int = 3, base_delay: float = 0.5) -> T:
    """Runs a sync yfinance call in a worker thread with exponential-backoff
    retry — yfinance has no retry of its own and rate-limits aggressively
    (HTTP 429), so a single transient blip previously failed the whole call
    immediately. The final attempt's exception propagates unchanged if every
    retry is exhausted; callers' existing except/ValueError handling is
    unaffected either way.
    yfinance 호출 재시도(지수 백오프) — 마지막 시도 실패시 원래 예외 그대로 전파"""
    last_exc: Exception | None = None
    for attempt in range(attempts):
        try:
            return await asyncio.to_thread(fn)
        except Exception as exc:  # noqa: BLE001 - retried broadly, yfinance raises varied types
            last_exc = exc
            if attempt < attempts - 1:
                delay = base_delay * (2**attempt)
                logger.warning(
                    "yfinance call failed (attempt %d/%d), retrying in %.1fs: %s",
                    attempt + 1, attempts, delay, exc,
                )
                await asyncio.sleep(delay)
    assert last_exc is not None
    raise last_exc


def _period_to_days(period: str) -> int:
    if period.endswith("mo"):
        return int(period[:-2]) * 30
    if period.endswith("y"):
        return int(period[:-1]) * 365
    if period.endswith("d"):
        return int(period[:-1])
    raise ValueError(f"unsupported period string: {period!r}")


def _history(ticker: str, period: str = "6mo", as_of: date | None = None) -> pd.DataFrame:
    ticker_obj = yf.Ticker(ticker)
    if as_of is None:
        data = ticker_obj.history(period=period)
    else:
        start = as_of - timedelta(days=_period_to_days(period))
        end = as_of + timedelta(days=1)  # yfinance's `end` is exclusive
        data = ticker_obj.history(start=start.isoformat(), end=end.isoformat())
    if data.empty:
        raise ValueError(f"no price history returned for {ticker!r}")
    return data


def _filter_filing_lag(df: pd.DataFrame, as_of: date) -> pd.DataFrame:
    cutoff = as_of - timedelta(days=_FILING_LAG_DAYS)
    cols = [c for c in df.columns if c.date() <= cutoff]
    if not cols:
        raise ValueError(f"no filed periods available as of {as_of} (filing-lag buffer applied)")
    return df[cols]


def _filter_rows_by_date(df: pd.DataFrame, dates, cutoff: date) -> pd.DataFrame:
    """Keeps only rows whose (already-extracted) date is on or before
    `cutoff` — shared by every provider here with real per-row dates to
    bound against `as_of` (upgrades/downgrades, institutional holdings,
    earnings history). `dates` may be a column, an index, tz-aware or not —
    normalized to a naive `DatetimeIndex` first so the comparison is
    consistent either way."""
    normalized = pd.DatetimeIndex(pd.to_datetime(dates))
    if normalized.tz is not None:
        normalized = normalized.tz_localize(None)
    return df[normalized <= pd.Timestamp(cutoff)]
