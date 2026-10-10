"""Shared ticker-symbol format guard for web routes that forward a raw path
param straight into a yfinance/KIS call.
라우터가 받은 티커 문자열을 외부 API로 넘기기 전 형식만 가볍게 검증."""

from __future__ import annotations

import re

from fastapi import HTTPException

# Permissive on purpose — real yfinance symbols aren't just "AAPL"-shaped:
# KRX suffix (005930.KS), indices (^GSPC), futures (GC=F), FX (KRW=X). This
# only rejects clearly-not-a-ticker input (wrong charset, unreasonable
# length) before it's forwarded straight into a yfinance/KIS call.
_TICKER_RE = re.compile(r"^[A-Z0-9.\-^=]{1,20}$")


def is_valid_ticker_format(ticker: str) -> bool:
    """Framework-agnostic half of the same check `validate_ticker` (below)
    uses for the HTTP route — shared with `app.web.state`'s auto-add path
    (AI BUY hook / krx_top_n-sourced trade-allowlist growth), which has no
    HTTPException to raise and previously did no format checking at all.
    Real incident: a comma-containing, 20+ char value (looked like several
    KIS fields glued together) reached `settings.trade_allowlist_domestic`/
    `_overseas` this way, self-perpetuated every cycle (trade allowlist
    entries are pooled back into auto_trade_loop's own candidates), and was
    misrouted by ticker_market's `.endswith(".KS")` check (that string
    doesn't end in `.KS`) into being treated as an overseas ticker for
    portfolio-cap purposes."""
    return bool(_TICKER_RE.match(ticker.strip().upper()))


def validate_ticker(ticker: str) -> str:
    upper = ticker.strip().upper()
    if not is_valid_ticker_format(upper):
        raise HTTPException(422, f"'{ticker}' doesn't look like a valid ticker symbol")
    return upper
