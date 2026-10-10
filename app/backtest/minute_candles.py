"""Historical 1-minute candle collector + Postgres cache for scalping
backtests (Phase 1 of docs/scalping-backtest-infra-plan.md).

Uses TR FHKST03010230 (주식일별분봉조회) — confirmed live on the paper
account (scripts/check_daily_minute_chart_tr.py, 2026-07-27) to return
past-date minute candles, unlike the intraday-only FHKST03010200 that
app.brokers.kis.domestic.quote.fetch_minute_candles wraps. One call returns
at most 120 rows, newest-first, so a full KRX session (~09:00-15:30, ~390
minutes) takes several calls paged backward by FID_INPUT_HOUR_1.

Cached in its own table rather than app.memory (that module's docstring
scopes it to the decision/reflection store) — same double-checked-locking
schema-bootstrap pattern as app.execution.pending_orders/safety_switch,
sharing the one Postgres pool via app.memory.db.get_pool().
"""

from __future__ import annotations

import asyncio
import logging
import math
from dataclasses import dataclass
from datetime import date

import asyncpg

from app.brokers.kis.const import KISBrokerError
from app.memory.db import get_pool

_TR_ID = "FHKST03010230"  # 주식일별분봉조회 [국내주식-213]
_PATH = "/uapi/domestic-stock/v1/quotations/inquire-time-dailychartprice"

# KRX regular session bounds. Paging stops once a page's earliest row is at
# or before market open — data before that (pre-market) isn't needed for
# scalping-hours backtests and isn't worth the extra round trip.
_MARKET_OPEN_HOUR = "090000"
_FIRST_PAGE_HOUR = "153000"  # after the 15:20 regular close + a small margin
_MAX_PAGES = 6  # safety cap: ~390 minutes / 120 per page ~= 4, +2 margin

_SCHEMA = """
CREATE TABLE IF NOT EXISTS minute_candles (
    ticker      TEXT NOT NULL,
    trade_date  DATE NOT NULL,
    time        TEXT NOT NULL,  -- HHMMSS, stck_cntg_hour
    open        DOUBLE PRECISION NOT NULL,
    high        DOUBLE PRECISION NOT NULL,
    low         DOUBLE PRECISION NOT NULL,
    close       DOUBLE PRECISION NOT NULL,
    volume      BIGINT NOT NULL,
    PRIMARY KEY (ticker, trade_date, time)
);
"""

_schema_ready = False
_schema_lock = asyncio.Lock()

logger = logging.getLogger(__name__)


async def _ready_pool() -> asyncpg.Pool:
    """Double-checked locking around the first-time DDL, same pattern as
    app.execution.pending_orders._ready_pool / safety_switch._ready_pool."""
    global _schema_ready
    pool = await get_pool()
    if _schema_ready:
        return pool
    async with _schema_lock:
        if not _schema_ready:
            async with pool.acquire() as conn:
                await conn.execute(_SCHEMA)
            _schema_ready = True
    return pool


@dataclass
class HistoricalMinuteCandle:
    time: str  # HHMMSS
    open: float
    high: float
    low: float
    close: float
    volume: int


def _price(value: object) -> float | None:
    """`None` for anything that isn't a real, positive price — a blank/
    malformed KIS field must never become a silent `0.0`, since a 0.0 open/
    high/low/close would be cached PERMANENTLY (see fetch_minute_day_from_kis)
    and corrupt every future reader of that day (a zero denominator in
    app.backtest.scalping_viability's return math, a fabricated -100% bar in
    app.backtest.gap_reversion, a fabricated fill price in
    app.backtest.fill_simulation.simulate_fill)."""
    try:
        f = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f) or f <= 0:
        return None
    return f


def _volume(value: object) -> int:
    """0 IS a legitimate value here (a minute with no trades) — unlike a
    price field, so this keeps the old soft-default-to-0 behavior."""
    try:
        v = int(float(value))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0
    return max(v, 0)


async def get_cached_minute_day(ticker: str, trade_date: date) -> list[HistoricalMinuteCandle] | None:
    """None if this (ticker, trade_date) has never been fetched — distinct
    from an empty list, which would mean "fetched, market was a holiday /
    had no trades" and should NOT be re-fetched on every call."""
    pool = await _ready_pool()
    rows = await pool.fetch(
        "SELECT time, open, high, low, close, volume FROM minute_candles "
        "WHERE ticker = $1 AND trade_date = $2 ORDER BY time",
        ticker, trade_date,
    )
    if not rows:
        return None
    return [
        HistoricalMinuteCandle(
            time=r["time"], open=r["open"], high=r["high"], low=r["low"],
            close=r["close"], volume=r["volume"],
        )
        for r in rows
    ]


async def _store_minute_day(ticker: str, trade_date: date, candles: list[HistoricalMinuteCandle]) -> None:
    pool = await _ready_pool()
    async with pool.acquire() as conn:
        async with conn.transaction():
            # Mark the day as fetched even if the market had no trades
            # (holiday) by writing nothing — get_cached_minute_day would
            # then see an empty result and treat it as "not fetched" and
            # re-fetch forever. A sentinel row would need a NULLable OHLCV
            # shape; simpler to have the caller record "fetched, empty" via
            # a separate marker table if that case matters later — not
            # needed yet since backfill callers skip non-trading days
            # up front (app.live.market_hours-style weekday/holiday check).
            await conn.executemany(
                "INSERT INTO minute_candles (ticker, trade_date, time, open, high, low, close, volume) "
                "VALUES ($1, $2, $3, $4, $5, $6, $7, $8) "
                "ON CONFLICT (ticker, trade_date, time) DO NOTHING",
                [
                    (ticker, trade_date, c.time, c.open, c.high, c.low, c.close, c.volume)
                    for c in candles
                ],
            )


async def fetch_minute_day_from_kis(broker, code: str, trade_date: date) -> list[HistoricalMinuteCandle]:
    """Pages FHKST03010230 backward from _FIRST_PAGE_HOUR to _MARKET_OPEN_HOUR
    for one (code, trade_date), deduping by time. Does not touch the cache —
    callers that want caching should use fetch_and_cache_minute_day instead.
    Raises KISBrokerError if the account/TR rejects the request (e.g. an
    unsupported account mode) rather than silently returning an empty list,
    so a rejection isn't mistaken for "no trades that day"."""
    date_str = trade_date.strftime("%Y%m%d")
    by_time: dict[str, HistoricalMinuteCandle] = {}
    next_hour = _FIRST_PAGE_HOUR

    for _ in range(_MAX_PAGES):
        resp = await broker._request(
            "GET", _PATH, _TR_ID,
            params={
                "FID_COND_MRKT_DIV_CODE": "J",
                "FID_INPUT_ISCD": code,
                "FID_INPUT_HOUR_1": next_hour,
                "FID_INPUT_DATE_1": date_str,
                "FID_PW_DATA_INCU_YN": "N",
                "FID_FAKE_TICK_INCU_YN": "",
            },
        )
        body = broker._checked_body(resp, f"historical minute candles for {code} on {date_str}")
        rows = body.get("output2", [])
        if not rows:
            break

        page_times = []
        for row in rows:
            t = row.get("stck_cntg_hour", "")
            if not t:
                continue
            page_times.append(t)  # counts toward paging progress regardless of the bar's own validity below

            o, h, l, c = (_price(row.get(k)) for k in ("stck_oprc", "stck_hgpr", "stck_lwpr", "stck_prpr"))
            if None in (o, h, l, c) or not (h >= max(o, c) and l <= min(o, c) and h >= l):
                logger.warning(
                    "minute_candles: dropping malformed bar code=%s date=%s time=%s raw=%r",
                    code, date_str, t,
                    {k: row.get(k) for k in ("stck_oprc", "stck_hgpr", "stck_lwpr", "stck_prpr", "cntg_vol")},
                )
                continue
            by_time[t] = HistoricalMinuteCandle(
                time=t, open=o, high=h, low=l, close=c, volume=_volume(row.get("cntg_vol")),
            )

        earliest = min(page_times) if page_times else next_hour
        if earliest <= _MARKET_OPEN_HOUR or earliest >= next_hour:
            # earliest >= next_hour means the page made no backward progress
            # (e.g. a short/holiday session) — stop rather than loop forever.
            break
        # KIS's "as of this hour" semantics are inclusive of the boundary row
        # in practice (observed 2026-07-27 verification calls), so paging
        # from the exact earliest time re-returns it — harmless since
        # by_time/ON CONFLICT DO NOTHING both dedupe on exact time match.
        next_hour = earliest

    return [by_time[t] for t in sorted(by_time)]


async def fetch_and_cache_minute_day(broker, ticker: str, code: str, trade_date: date) -> list[HistoricalMinuteCandle]:
    """Cache-first: returns the cached day if present, otherwise fetches from
    KIS via fetch_minute_day_from_kis and stores the result (even if empty,
    for now callers should avoid requesting non-trading days at all — see
    _store_minute_day's docstring on why an empty day isn't persisted as
    "fetched")."""
    cached = await get_cached_minute_day(ticker, trade_date)
    if cached is not None:
        return cached
    candles = await fetch_minute_day_from_kis(broker, code, trade_date)
    if candles:
        await _store_minute_day(ticker, trade_date, candles)
    return candles
