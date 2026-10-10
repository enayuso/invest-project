"""Tiny leaf module: an in-memory ticker -> Korean-name cache, populated by
app.live.universe.krx_top_n (which already gets hts_kor_isnm from KIS's
volume-ranking TR for its own leverage-filter check) and read by
app.web.state.get_display_name so a scanned-candidate KRX ticker shows its
real Korean name instead of falling back to yfinance's English legal name
(e.g. "hyungji Elite Co., Ltd." instead of "삼성전자").

Deliberately its own module rather than living in either universe.py or
state.py: state.py already imports app.live.auto_trade_loop (which imports
app.live.universe), so having universe.py import app.web.state back would
be a circular import. Both sides depend on this small leaf instead —
neither needs to import the other.
"""

from __future__ import annotations

_cache: dict[str, str] = {}


def remember(ticker: str, name: str) -> None:
    if ticker and name:
        _cache[ticker] = name


def get(ticker: str) -> str | None:
    return _cache.get(ticker)
