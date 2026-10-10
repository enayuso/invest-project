"""yfinance-backed data providers, split by responsibility:

- `research.py` — LLM-agent research providers, router-registered, return
  prose `str` for analyst prompts.
- `market_data.py` — internal deterministic data feeds used directly by
  `execution/`, `live/`, `backtest/`, `chat/` (not router-routed).
- `shared.py` — helpers/constants used by both.

This package `__init__` re-exports every name that was previously
importable from the flat `app.dataflows.yfinance_provider` module, so
existing call sites keep working unchanged. Importing this package also
triggers `research`'s `router.register(...)` calls (import side effects).
"""

from __future__ import annotations

from . import market_data, research
from .market_data import (
    IntradayBar,
    LiveQuote,
    TickerDisplayInfo,
    _atr,
    _series_to_points,
    get_atr_pct,
    get_daily_high_series,
    get_daily_low_series,
    get_daily_open_series,
    get_earnings_trend,
    get_intraday_bars,
    get_intraday_series,
    get_last_close,
    get_live_quote,
    get_news_items,
    get_ohlcv_candles,
    get_price_series,
    get_ticker_currency,
    get_ticker_display_info,
    get_usd_fx_rate,
)
from .research import (
    _rsi,
    _yf_analyst_consensus,
    _yf_balance_sheet,
    _yf_cashflow,
    _yf_earnings_estimates,
    _yf_fundamentals,
    _yf_indicators,
    _yf_insider_txns,
    _yf_institutional_ownership,
    _yf_news,
    _yf_short_interest,
    _yf_stock_data,
    _yf_verified_snapshot,
    get_analyst_consensus,
    get_balance_sheet,
    get_cashflow,
    get_earnings_estimates,
    get_fundamentals,
    get_indicators,
    get_insider_txns,
    get_institutional_ownership,
    get_long_horizon_context,
    get_news,
    get_short_interest,
    get_stock_data,
    get_verified_snapshot,
    _yf_long_horizon,
)
from .shared import (
    _FILING_LAG_DAYS,
    _HOLDINGS_LAG_DAYS,
    _CANDLE_CFG,
    _filter_filing_lag,
    _filter_rows_by_date,
    _history,
    _period_to_days,
)

__all__ = [
    # dataclasses
    "IntradayBar",
    "LiveQuote",
    "TickerDisplayInfo",
    # research.py — router-registered LLM-facing providers
    "get_stock_data",
    "get_long_horizon_context",
    "get_indicators",
    "get_verified_snapshot",
    "get_fundamentals",
    "get_balance_sheet",
    "get_cashflow",
    "get_insider_txns",
    "get_analyst_consensus",
    "get_institutional_ownership",
    "get_short_interest",
    "get_earnings_estimates",
    "get_news",
    "_yf_stock_data",
    "_yf_long_horizon",
    "_yf_indicators",
    "_yf_verified_snapshot",
    "_yf_fundamentals",
    "_yf_balance_sheet",
    "_yf_cashflow",
    "_yf_insider_txns",
    "_yf_analyst_consensus",
    "_yf_institutional_ownership",
    "_yf_short_interest",
    "_yf_earnings_estimates",
    "_yf_news",
    "_rsi",
    # market_data.py — internal deterministic feeds
    "get_ohlcv_candles",
    "get_news_items",
    "get_earnings_trend",
    "get_price_series",
    "get_last_close",
    "get_atr_pct",
    "get_live_quote",
    "get_intraday_bars",
    "get_intraday_series",
    "get_ticker_currency",
    "get_usd_fx_rate",
    "get_ticker_display_info",
    "get_daily_low_series",
    "get_daily_high_series",
    "get_daily_open_series",
    "_series_to_points",
    "_atr",
    # shared.py — helpers/constants
    "_period_to_days",
    "_history",
    "_filter_filing_lag",
    "_filter_rows_by_date",
    "_FILING_LAG_DAYS",
    "_HOLDINGS_LAG_DAYS",
    "_CANDLE_CFG",
    # submodules
    "research",
    "market_data",
]
