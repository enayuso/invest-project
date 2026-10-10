"""Shared "what session are we in, which bars are completed" logic.

Used by both `market_data.py` (price snapshot/history) and
`indicators.py` (technical indicators must also exclude the in-progress
bar -- spec S:2-4: "진행 중인 오늘 봉은 계산에서 제외... 실계좌 시스템의
'순위가 스캔 시각의 함수' 결함과 같은 함정"). Centralizing this means
both tools can never disagree about which bar counts as "completed".
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

import pandas as pd

from . import cache, freshness, yf_gateway
from .market_hours import KST, decide_session_status

_BARS_TTL_SECONDS = 5 * 60
_INFO_TTL_SECONDS = 5 * 60


def _now_kst() -> datetime:
    return datetime.now(KST)


def _reference_history_fetcher(symbol: str) -> pd.DataFrame:
    return yf_gateway.get_history(symbol, period="5d", interval="1d")


def latest_trading_day() -> date:
    return freshness.get_latest_trading_day(_reference_history_fetcher)


def fetch_daily_bars(yf_symbol: str, period: str) -> pd.DataFrame:
    return cache.cached(
        "daily_bars",
        f"{yf_symbol}:{period}",
        _BARS_TTL_SECONDS,
        lambda: yf_gateway.get_history(yf_symbol, period=period, interval="1d"),
    )


def fetch_info(yf_symbol: str) -> dict:
    return cache.cached(
        "info",
        yf_symbol,
        _INFO_TTL_SECONDS,
        lambda: yf_gateway.get_info(yf_symbol),
    )


@dataclass
class SessionView:
    bars: pd.DataFrame
    completed_bars: pd.DataFrame
    session_status: str  # "in_progress" | "closed"
    market_state_raw: str | None
    session_basis: str = ""


def get_session_view(yf_symbol: str, *, period: str) -> SessionView:
    bars = fetch_daily_bars(yf_symbol, period)
    if bars is None or len(bars) == 0:
        raise freshness.NoDataError(
            f"{yf_symbol}: yfinance returned zero price rows. This usually means the wrong "
            "exchange suffix was used, the symbol is delisted, or it does not exist on yfinance."
        )

    last_bar_date = bars.index[-1].date()
    freshness.assert_fresh(last_bar_date, latest_trading_day(), symbol=yf_symbol)

    info = fetch_info(yf_symbol)
    market_state = info.get("marketState")
    session_status, session_basis = decide_session_status(last_bar_date, _now_kst(), market_state)

    completed_bars = bars.iloc[:-1] if session_status == "in_progress" else bars

    return SessionView(
        bars=bars,
        completed_bars=completed_bars,
        session_status=session_status,
        market_state_raw=market_state,
        session_basis=session_basis,
    )
