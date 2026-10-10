"""US-ticker (NASDAQ) symbol -> Korean company name, from KIS's own daily
master-file snapshot (nasmst.cod), downloaded from the same public KRX/KIS
distribution mirror as `app.live.krx_master`'s domestic file. Same
motivation and same no-auth/no-account-mode-distinction property — see
that module's docstring for the yfinance cross-symbol-confusion incidents
this class of fix addresses.

Scoped to NASDAQ only (`nas`), matching this project's current overseas
trading scope everywhere else (order routing defaults to
`OVRS_EXCG_CD="NASD"`, see app.brokers.kis.overseas.order/account and
app.domain.ports.broker_port's `exchange` defaults) — NYSE/AMEX and the
non-US markets KIS also publishes (Shanghai, Shenzhen, Tokyo, Hong Kong,
Hanoi, Ho Chi Minh) aren't covered here since nothing in this codebase
trades them yet. Add another market's .cod file the same way this one
was added if that scope ever expands.

Row format: tab-separated text, cp949-encoded, no header row, 24 fixed
columns per stock (verified against the reference parser at
github.com/koreainvestment/open-trading-api/blob/main/stocks_info/
overseas_stock_code.py on 2026-07-29) — column 4 is the trading symbol,
column 6 is the Korean name. Confirmed against known tickers on
2026-07-29 (AAPL->애플, TSLA->테슬라, NVDA->엔비디아), all 5,216 NASDAQ rows
parsed cleanly (no malformed lines, no duplicate symbols).
"""

from __future__ import annotations

import asyncio
import io
import logging
import time
import zipfile

import httpx

logger = logging.getLogger(__name__)

_URL = "https://new.real.download.dws.co.kr/common/master/nasmst.cod.zip"
_SYMBOL_COLUMN = 4
_KOREAN_NAME_COLUMN = 6
_EXPECTED_COLUMNS = 24

_cache: dict[str, str] = {}
_loaded = False
_lock = asyncio.Lock()
# See app.live.krx_master's identical constants for the reasoning — this
# module has the exact same race and permanent-latch-on-failure fix applied.
_last_failed_at: float | None = None
_RETRY_BACKOFF_SECONDS = 300.0


def _parse(text: str) -> dict[str, str]:
    names: dict[str, str] = {}
    for row in text.splitlines():
        parts = row.split("\t")
        if len(parts) != _EXPECTED_COLUMNS:
            continue  # format drift (KIS's own reference script warns of this) — skip rather than misparse
        symbol = parts[_SYMBOL_COLUMN].strip()
        name = parts[_KOREAN_NAME_COLUMN].strip()
        if symbol and name:
            names[symbol] = name
    return names


async def _download() -> dict[str, str]:
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(_URL)
        resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        raw = zf.read(zf.namelist()[0])
    return _parse(raw.decode("cp949"))


async def ensure_loaded() -> None:
    """Same once-per-process, no-op-after-first-successful-load,
    backoff-and-retry-on-failure contract as `app.live.krx_master.
    ensure_loaded` — see that docstring for the full reasoning, including
    why `_loaded` must only flip to True after the download actually
    succeeds (not before it starts, which raced concurrent callers into
    silently seeing an empty cache) and why a failure retries after
    `_RETRY_BACKOFF_SECONDS` rather than latching for the process's life."""
    global _loaded, _last_failed_at
    if _loaded:
        return
    async with _lock:
        if _loaded:
            return
        if _last_failed_at is not None and time.monotonic() - _last_failed_at < _RETRY_BACKOFF_SECONDS:
            return
        try:
            names = await _download()
        except Exception:
            logger.warning(
                "NASDAQ master-file download failed, falling back to other name sources", exc_info=True
            )
            _last_failed_at = time.monotonic()
            return
        _cache.update(names)
        _loaded = True
        _last_failed_at = None


def get(ticker: str) -> str | None:
    """`ticker` like 'AAPL' (this project's US tickers are bare symbols,
    no suffix — see kis_provider._is_us). None if not loaded yet,
    download failed, or the symbol isn't in the NASDAQ snapshot (e.g. an
    NYSE/AMEX-listed ticker, out of scope per this module's docstring)."""
    return _cache.get(ticker)


def search(query: str, limit: int = 8) -> list[tuple[str, str]]:
    """Same reverse-lookup rationale as `app.live.krx_master.search` —
    matches the Korean name (e.g. "애플" finds AAPL) or the bare symbol
    itself, case-insensitively. Returns (symbol, name) pairs."""
    if not query:
        return []
    q = query.strip().lower()
    results = [
        (symbol, name) for symbol, name in _cache.items()
        if q in name.lower() or q in symbol.lower()
    ]
    return results[:limit]
