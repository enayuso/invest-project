"""KOSPI/KOSDAQ ticker -> Korean company name from KRX's own daily
code->name master-file snapshot (kospi_code.mst / kosdaq_code.mst),
downloaded from KIS's public distribution mirror. No app-key/token and
no paper/real account distinction — unlike the CTPF1002R TR this
substitutes for (rejected outright on a paper account with EGW02006,
see app.orchestrator.pipeline._company_identity's docstring), the master
file is a plain HTTPS download of an exchange-published snapshot.

Exists because yfinance's longName/shortName has shown real cross-symbol
confusion for an exact KRX code, which a plausibility check on the name
string alone can't catch (it looks like a real company, just the wrong
one): 064260.KS (다날) came back "에코프로비엠" (2026-07-23 incident, see
app.web.state.get_display_name's docstring), and 028050.KS got argued
over as "Hyundai Motor" instead of 삼성E&A mid-debate during the
2026-07-28 FULL_AUTO test, burning several risk-debate cycles on mutual
"prompt injection suspected" flags before anyone caught that the
grounding name itself was wrong. KRX's own snapshot can't suffer that —
it's an exact code->name table straight from the exchange, not a
same-ish-sounding-company guess.

Row format: fixed-width text, cp949-encoded, one stock per line. Every
row ends in a 228-character trailing block of numeric/flag fields (그룹
코드, 시가총액규모 등) that this module has no use for and never parses.
Everything before that block is 단축코드(9자)+표준코드(12자)+한글명(나머지) —
confirmed by downloading both files and manually checking known codes
on each market (005930->삼성전자, 028050->삼성E&A, 005380->현대차 on KOSPI;
064260->다날, 247540->에코프로비엠 on KOSDAQ) on 2026-07-29.
"""

from __future__ import annotations

import asyncio
import io
import logging
import time
import zipfile

import httpx

logger = logging.getLogger(__name__)

_URLS = {
    "kospi": "https://new.real.download.dws.co.kr/common/master/kospi_code.mst.zip",
    "kosdaq": "https://new.real.download.dws.co.kr/common/master/kosdaq_code.mst.zip",
}
_FLAG_SUFFIX_LEN = 228

_cache: dict[str, str] = {}
_loaded = False
_lock = asyncio.Lock()
# Same backoff pattern as app.memory.db's pool bootstrap (_last_failure_at),
# just a longer window: a download failure here is far more likely a
# sustained mirror outage than a one-off DB hiccup (confirmed live —
# new.real.download.dws.co.kr timed out entirely in one of four runs while
# auditing this module), and every lookup calls ensure_loaded(), so without
# a backoff a down mirror means every single ticker lookup blocks behind
# the same 15s HTTPS timeout.
_last_failed_at: float | None = None
_RETRY_BACKOFF_SECONDS = 300.0


def _parse(text: str) -> dict[str, str]:
    names: dict[str, str] = {}
    for row in text.splitlines():
        if len(row) <= _FLAG_SUFFIX_LEN + 21:
            continue  # malformed/short row — not a real stock record
        head = row[: len(row) - _FLAG_SUFFIX_LEN]
        code = head[:9].strip()
        name = head[21:].strip()
        if code and name:
            names[code] = name
    return names


async def _download(market: str) -> dict[str, str]:
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(_URLS[market])
        resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        raw = zf.read(zf.namelist()[0])
    return _parse(raw.decode("cp949"))


async def ensure_loaded() -> None:
    """Downloads both master files once per process and populates the
    in-memory cache — a daily exchange snapshot, not intraday data, so
    one load per process lifetime is enough. Safe to call on every
    lookup: a no-op after the first successful load. On download failure
    (network issue, KRX mirror down) leaves the cache empty and logs a
    warning rather than raising, so callers fall through to their next
    name source (yfinance) exactly as if this module didn't exist — but
    retries after `_RETRY_BACKOFF_SECONDS`, rather than latching the
    failure for the rest of the process's life (a genuinely down mirror
    used to mean every subsequent lookup silently degraded to yfinance
    forever, which is exactly the wrong-company-name failure mode this
    module exists to prevent — see the module docstring's two real
    incidents).

    `_loaded` is set to True only AFTER a successful load, not before the
    download starts — setting it early raced every concurrent caller: caller
    A marks it loaded and starts a 15s download, caller B (analyzed under
    `auto_trade_max_concurrent_analyses` concurrency) sees `_loaded=True`
    moments later and returns immediately with an EMPTY cache, silently
    falling through to yfinance for that lookup even though the download
    would have succeeded a few seconds later."""
    global _loaded, _last_failed_at
    if _loaded:
        return
    async with _lock:
        if _loaded:
            return
        if _last_failed_at is not None and time.monotonic() - _last_failed_at < _RETRY_BACKOFF_SECONDS:
            return  # still in backoff since the last failure — don't hammer a down mirror
        try:
            kospi, kosdaq = await asyncio.gather(_download("kospi"), _download("kosdaq"))
        except Exception:
            logger.warning("KRX master-file download failed, falling back to other name sources", exc_info=True)
            _last_failed_at = time.monotonic()
            return
        _cache.update(kospi)
        _cache.update(kosdaq)
        _loaded = True
        _last_failed_at = None


def get(ticker: str) -> str | None:
    """`ticker` like '028050.KS'. None if not loaded yet, download
    failed, or the code isn't in either snapshot (e.g. a same-day new
    listing not yet in the daily file)."""
    if not ticker.endswith(".KS"):
        return None
    return _cache.get(ticker[:-3])


def search(query: str, limit: int = 8) -> list[tuple[str, str]]:
    """Reverse lookup for the name-search box (2026-07-30) — `handleGlobalSearch`
    only recognized an exact ticker code before this, so a user who didn't
    already know a code (the common case) had no way to find one. `query`
    matched case-insensitively as a substring of the Korean name; also
    matches a bare/partial code so "0059" still finds 삼성전자. Empty list
    if the master file hasn't loaded (download failed, or `ensure_loaded`
    was never awaited) — same "degrade to nothing, don't raise" contract
    as `get`. Returns (code, name) pairs, not `.KS`-suffixed tickers — the
    caller appends that, matching `get`'s own input convention."""
    if not query:
        return []
    q = query.strip().lower()
    results = [
        (code, name) for code, name in _cache.items()
        if q in name.lower() or q in code
    ]
    return results[:limit]
