"""KRX (Korea Exchange) code -> Korean name -> market lookup.

Downloads the two unauthenticated KRX master-file snapshots (KOSPI and
KOSDAQ) and answers "which market is this 6-digit code listed on?". This
is a fresh, standalone reimplementation for stock-advisor-mcp -- it does
NOT import anything from the `invest-project` trading
repository. It was written after *reading* that repo's
`app/live/krx_master.py` as a reference for the file format (fixed-width
text, cp949-encoded, no authentication, distributed from KRX's own
download mirror), then reimplemented from scratch here.

One important difference from the reference implementation: the
reference merges both markets into a single `code -> name` dict, which
throws away *which file the code came from*. That is fine for a
name-lookup helper, but wrong for us: the whole point of this module is
that **market membership is derived from which master file the code
appears in** (spec S:2-1 / hard constraint #1 in the task brief) so we
can pick the correct yfinance suffix (`.KS` for KOSPI, `.KQ` for
KOSDAQ). So this module keeps two separate dicts and never merges them.

Caching: an in-memory cache for the life of the process, backed by a
same-day on-disk JSON cache (spec S:3-3 "마스터파일: 하루 1회 다운로드 후
로컬 캐시") so a fresh CLI/MCP process started later the same day does
not re-download. The cache file records the KST calendar date it was
fetched on; a process started on a later KST date re-downloads.
"""

from __future__ import annotations

import io
import json
import logging
import time
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path

import httpx

from .errors import AdvisorError

logger = logging.getLogger(__name__)

_KST = timezone(timedelta(hours=9))

_URLS = {
    "KOSPI": "https://new.real.download.dws.co.kr/common/master/kospi_code.mst.zip",
    "KOSDAQ": "https://new.real.download.dws.co.kr/common/master/kosdaq_code.mst.zip",
}

# Every row ends in a long fixed-width block of numeric/flag fields this
# module has no use for. Confirmed empirically (2026-09-21) against known
# codes on both markets: 005930->삼성전자, 028050->삼성E&A, 005380->현대차
# (KOSPI); 036540->SFA반도체, 247540->에코프로비엠 (KOSDAQ);
# 379800->KODEX 미국S&P500 (KOSPI -- Korean ETFs list on the KOSPI board
# even when the underlying is a KOSDAQ-style small stock; do not assume
# ETF == KOSDAQ).
_FLAG_SUFFIX_LEN = 228

_CACHE_DIR = Path(__file__).resolve().parent.parent.parent / ".cache"


class KrxMasterError(AdvisorError):
    """Raised when the KRX master files cannot be downloaded/parsed.

    Callers must NOT fall back to guessing a yfinance suffix when this is
    raised -- that guess is exactly the failure mode (silent stale data,
    wrong company name) this module exists to prevent.
    """


@dataclass(frozen=True)
class MasterLookupResult:
    code: str
    market: str  # "KOSPI" | "KOSDAQ"
    name_ko: str


def _parse(text: str) -> dict[str, str]:
    names: dict[str, str] = {}
    for row in text.splitlines():
        if len(row) <= _FLAG_SUFFIX_LEN + 21:
            continue  # malformed/short row -- not a real stock record
        head = row[: len(row) - _FLAG_SUFFIX_LEN]
        code = head[:9].strip()
        name = head[21:].strip()
        if code and name:
            names[code] = name
    return names


def _download(market: str) -> dict[str, str]:
    with httpx.Client(timeout=15.0) as client:
        resp = client.get(_URLS[market])
        resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        raw = zf.read(zf.namelist()[0])
    return _parse(raw.decode("cp949"))


def _today_kst() -> str:
    return datetime.now(_KST).strftime("%Y-%m-%d")


def _cache_path(market: str) -> Path:
    return _CACHE_DIR / f"{market.lower()}_master.json"


def _load_disk_cache(market: str) -> dict[str, str] | None:
    path = _cache_path(market)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if payload.get("fetched_on_kst") != _today_kst():
        return None  # stale disk cache (different KST calendar day)
    codes = payload.get("codes")
    if not isinstance(codes, dict):
        return None
    return codes


def _save_disk_cache(market: str, codes: dict[str, str]) -> None:
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _cache_path(market).write_text(
            json.dumps({"fetched_on_kst": _today_kst(), "codes": codes}, ensure_ascii=False),
            encoding="utf-8",
        )
    except OSError:
        logger.warning("Could not write KRX master-file disk cache for %s", market, exc_info=True)


class KrxMaster:
    """Holds the two per-market code->name dicts, loaded once per process."""

    def __init__(self) -> None:
        self._by_market: dict[str, dict[str, str]] | None = None

    def ensure_loaded(self) -> None:
        if self._by_market is not None:
            return
        by_market: dict[str, dict[str, str]] = {}
        for market in _URLS:
            cached = _load_disk_cache(market)
            if cached is not None:
                by_market[market] = cached
                continue
            try:
                codes = _download(market)
            except Exception as exc:  # network error, bad zip, bad encoding, etc.
                raise KrxMasterError(
                    f"KRX {market} master-file download/parse failed -- refusing to fall back "
                    "to guessing a yfinance suffix"
                ) from exc
            if not codes:
                raise KrxMasterError(f"KRX {market} master file parsed to zero rows")
            _save_disk_cache(market, codes)
            by_market[market] = codes
        self._by_market = by_market

    def lookup_code(self, code: str) -> MasterLookupResult | None:
        """`code` is a bare 6-digit KRX code (no suffix). None if not found
        in either market's snapshot."""
        self.ensure_loaded()
        assert self._by_market is not None
        for market, codes in self._by_market.items():
            name = codes.get(code)
            if name is not None:
                return MasterLookupResult(code=code, market=market, name_ko=name)
        return None

    def search_name(self, query: str, limit: int = 8) -> list[MasterLookupResult]:
        """Case-insensitive substring search over Korean names, across both
        markets. Used when `resolve_ticker` is given a company name instead
        of a code."""
        self.ensure_loaded()
        assert self._by_market is not None
        q = query.strip().lower()
        if not q:
            return []
        results: list[MasterLookupResult] = []
        for market, codes in self._by_market.items():
            for code, name in codes.items():
                if q in name.lower():
                    results.append(MasterLookupResult(code=code, market=market, name_ko=name))
        return results[:limit]


# Module-level singleton -- one KRX master snapshot per process, exactly
# like the reference module's contract ("one load per process lifetime is
# enough" -- this is a daily exchange snapshot, not intraday data).
_singleton = KrxMaster()


def get_master() -> KrxMaster:
    return _singleton
