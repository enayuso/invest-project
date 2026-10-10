"""Thin, throttled wrapper around yfinance.

Everything that talks to yfinance goes through here so that:
  - there is exactly one place enforcing the "short sleep between calls,
    no retries on 429/404" rule from spec S:3-3,
  - `resolver.py` / `market_data.py` / `indicators.py` / `fundamentals.py`
    can be unit-tested by monkeypatching this module instead of mocking
    yfinance's internals directly.

No caching lives here -- that is layered on top by callers via
`cache.py`, at the granularity each tool needs (a snapshot call and a
history call for the same symbol have different natural TTLs even
though both go through this module).
"""

from __future__ import annotations

import threading
import time

import pandas as pd
import yfinance as yf

_MIN_GAP_SECONDS = 0.4
_lock = threading.Lock()
_last_call_at: float = 0.0


def _throttle() -> None:
    global _last_call_at
    with _lock:
        wait = _MIN_GAP_SECONDS - (time.monotonic() - _last_call_at)
        if wait > 0:
            time.sleep(wait)
        _last_call_at = time.monotonic()


def get_info(yf_symbol: str) -> dict:
    _throttle()
    return yf.Ticker(yf_symbol).info or {}


def get_history(yf_symbol: str, period: str = "5d", interval: str = "1d") -> pd.DataFrame:
    _throttle()
    return yf.Ticker(yf_symbol).history(period=period, interval=interval)


def get_financials(yf_symbol: str) -> pd.DataFrame:
    _throttle()
    return yf.Ticker(yf_symbol).financials


def get_quarterly_financials(yf_symbol: str) -> pd.DataFrame:
    _throttle()
    return yf.Ticker(yf_symbol).quarterly_financials


def get_balance_sheet(yf_symbol: str) -> pd.DataFrame:
    _throttle()
    return yf.Ticker(yf_symbol).balance_sheet
