"""FRED macro data provider — optional category.

Free API key: https://fred.stlouisfed.org/docs/api/api_key.html. Without a
key configured, this raises immediately and the router's "optional" handling
degrades the macro section quietly rather than failing the whole
News+Macro analyst report.

Accepts an optional `as_of` date for point-in-time backtesting: FRED's API
natively supports vintage (as-it-was-known-then) queries via
`observation_end`/`realtime_end`, so macro data can be correctly bounded to
a historical date without any approximation.
"""

from __future__ import annotations

from datetime import date

import httpx

from app.config import settings
from app.dataflows.router import router

_BASE = "https://api.stlouisfed.org/fred/series/observations"

# Series worth surfacing for a general macro backdrop: fed funds rate, CPI,
# unemployment rate, real GDP.
#
# 2026-09-11: `GDP` was previously mapped here under the "Real GDP" label,
# but FRED series `GDP` is Gross Domestic Product in CURRENT (nominal)
# dollars — real GDP is series `GDPC1`. The News+Macro analyst was computing
# quarter-over-quarter growth off a nominal series while believing it was
# real, overstating growth by the GDP deflator (and in an inflationary
# regime, capable of showing "growth" during a real contraction). Fixed by
# switching the series ID to `GDPC1`, which makes the existing "Real GDP"
# label accurate rather than relabeling to "nominal" — the surrounding
# intent was always real GDP, so the series ID was the thing that was wrong.
# 이 라벨은 항상 "실질GDP"였는데 시리즈ID가 명목GDP(GDP)를 가리키고 있었음 —
# GDPC1로 교체해 라벨과 데이터를 일치시킴 (라벨을 명목으로 바꾸지 않음).
#
# Each entry is (label, unit) so the rendered line always carries a unit —
# see the render loop below for why a hardcoded-but-base-year-agnostic unit
# string was chosen over fetching FRED's `series` metadata endpoint.
_SERIES = {
    "FEDFUNDS": ("Fed Funds Rate", "percent per annum"),
    "CPIAUCSL": (
        "CPI (All Urban Consumers)",
        "index, 1982-84=100 (an index LEVEL, not an inflation rate)",
    ),
    "UNRATE": ("Unemployment Rate", "percent"),
    "GDPC1": ("Real GDP", "billions of chained dollars, seasonally adjusted annual rate"),
}


async def _fred_macro_indicators(as_of: date | None = None) -> str:
    if not settings.fred_api_key:
        raise RuntimeError("FRED_API_KEY not configured")

    async with httpx.AsyncClient(timeout=10.0) as client:
        lines = []
        for series_id, (label, unit) in _SERIES.items():
            params = {
                "series_id": series_id,
                "api_key": settings.fred_api_key,
                "file_type": "json",
                "sort_order": "desc",
                "limit": 1,
            }
            if as_of is not None:
                params["observation_end"] = as_of.isoformat()
                params["realtime_end"] = as_of.isoformat()
            try:
                resp = await client.get(_BASE, params=params)
                resp.raise_for_status()
            except httpx.HTTPError as exc:
                # Never let the raw httpx exception propagate: it embeds the
                # full request URL — including `api_key` — in its own
                # __str__. Unsanitized, that would leak the key into
                # `router.py`'s vendor-failure warning log, into the
                # News+Macro analyst's own tool-result text on degrade (this
                # category is `optional=True`), and from there into a stored
                # transcript served back out by the unauthenticated
                # `GET /api/decision/{id}` endpoint. `from None` also drops
                # the exception chain so the raw error can't resurface via a
                # future `exc_info=True` traceback either.
                raise RuntimeError(f"FRED request failed for {series_id}: {type(exc).__name__}") from None
            obs = resp.json().get("observations", [])
            if obs:
                # Every series carries an explicit unit so a bare number
                # (e.g. a CPI index level) can't be misread as a rate or a
                # dollar figure. FRED's `series` metadata endpoint exposes a
                # `units` field that would stay correct across FRED's
                # periodic chained-dollar rebenchmarking automatically, but
                # that's an extra API call per series for an `optional=True`
                # macro-context category — a hardcoded, base-year-agnostic
                # unit string (no year baked into the GDP unit) is the
                # better tradeoff here. Recorded so a future reader doesn't
                # assume the metadata call was simply overlooked.
                # 단위 표기 없이 숫자만 내보내던 버그 수정 — 시리즈별 단위 문자열 추가.
                lines.append(f"{label}: {obs[0]['value']} {unit} (as of {obs[0]['date']})")
        if not lines:
            raise RuntimeError("FRED returned no observations")
        return "Macro indicators (FRED):\n" + "\n".join(lines)


router.register("macro_data", [_fred_macro_indicators], optional=True)


async def get_macro_indicators(as_of: date | None = None) -> str:
    return await router.call("macro_data", as_of=as_of)
