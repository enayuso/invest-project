"""Forward-collects live H0STASP0 (실시간호가) snapshots to Postgres for
future scalping-strategy backtesting research (2026-08-24).

KIS provides no historical orderbook endpoint — only forward-collected data
is possible, so starting this now (however small the orderbook zone is
today) has real value that waiting does not. This is deliberately separate
from `app.backtest.minute_candles`'s KIS-fetch-on-demand cache: there is
nothing to fetch on demand here, only whatever RealtimeFeed happened to have
in memory when this ran.

Only logs codes actually in `RealtimeFeed.orderbook_codes` (see
app.brokers.kis.domestic.websocket and app.web.state._prioritized_krx_codes's
_ORDERBOOK_ZONE_SIZE) — the trailing, currently-unheld tail of the KRX
subscribe list that trades trade-tick (H0STCNT0) coverage for orderbook
(H0STASP0) coverage within the same 41-registration cap. A code NOT in
orderbook_codes never had its orderbook TR subscribed in the first place, so
anything sitting in `feed.orderbook` for it is either absent or a leftover
from some earlier session where it briefly was in the zone — logging that
would silently mix data collected under a different subscription regime into
this table.

Same schema-bootstrap convention as app.backtest.minute_candles: a
module-level `_SCHEMA`, double-checked-locking bootstrap via
app.memory.db.get_pool(), raw asyncpg — no ORM. Degrades quietly like
app.memory.store: this is research logging, not a trading path, so a DB
hiccup here must never affect live trading.
"""

from __future__ import annotations

import asyncio
import json
import logging

import asyncpg

from app.brokers.kis.domestic.websocket import RealtimeFeed
from app.config import settings
from app.live import kis_market_status
from app.memory.db import get_pool

logger = logging.getLogger("app.live.orderbook_log")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS orderbook_ticks (
    ticker      TEXT NOT NULL,
    captured_at TIMESTAMPTZ NOT NULL,
    quote_time  TEXT NOT NULL,  -- HHMMSS, the feed's own OrderBook.time
    bids        JSONB NOT NULL,
    asks        JSONB NOT NULL,
    PRIMARY KEY (ticker, captured_at)
);
"""

_schema_ready = False
_schema_lock = asyncio.Lock()


async def _ready_pool() -> asyncpg.Pool:
    """Double-checked locking around the first-time DDL, same pattern as
    app.backtest.minute_candles._ready_pool."""
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


async def log_orderbook_snapshot(feed: RealtimeFeed) -> int:
    """Inserts one row per orderbook-zone code with a present entry in
    `feed.orderbook`. `captured_at` is `now()` in SQL (not app-side) so every
    row in one batch shares exactly one timestamp, consistent across codes.

    `ON CONFLICT (ticker, captured_at) DO NOTHING` (matching minute_candles's
    own dedup idiom) means two snapshots of the same ticker landing in the
    same second collide on the PK and the second is silently dropped — an
    accepted, documented limitation at this write cadence
    (orderbook_log_interval_seconds), not a bug to chase down here. A future
    need for sub-second resolution is a one-line PK/column change (add a
    sequence or switch captured_at to a finer type), not a redesign.

    Returns the number of rows actually inserted, so the caller can log a
    count. Degrades quietly on any failure (a DB hiccup must never affect
    live trading, same posture as app.memory.store) — logs a warning and
    returns 0 rather than raising."""
    try:
        pool = await _ready_pool()
        inserted = 0
        async with pool.acquire() as conn:
            for code in feed.orderbook_codes:
                ob = feed.orderbook.get(code)
                if ob is None:
                    continue
                # feed.orderbook_codes holds raw 6-digit KRX codes (what
                # RealtimeFeed subscribes with — see krx_code_from_ticker's
                # docstring: "005930.KS" -> "005930"), but every OTHER
                # research table this data is meant to be joined against
                # (app.backtest.minute_candles, and the scalping_signals/
                # scalping_viability tooling built on top of it) keys on the
                # ".KS"-suffixed ticker. Storing the raw code here would
                # silently make this table unjoinable by ticker without a
                # caller remembering to strip/append the suffix — append it
                # once, here, so `orderbook_ticks.ticker` matches every other
                # domestic research table's convention.
                ticker = f"{code}.KS"
                status = await conn.execute(
                    "INSERT INTO orderbook_ticks (ticker, captured_at, quote_time, bids, asks) "
                    "VALUES ($1, now(), $2, $3, $4) "
                    "ON CONFLICT (ticker, captured_at) DO NOTHING",
                    ticker, ob.time, json.dumps(ob.bids), json.dumps(ob.asks),
                )
                # asyncpg's execute() returns a command tag like "INSERT 0 1"
                # (0 rows) or "INSERT 0 0" (conflict, skipped) — the trailing
                # integer is the actual row count, which ON CONFLICT DO
                # NOTHING makes not always 1.
                try:
                    inserted += int(status.rsplit(" ", 1)[-1])
                except (ValueError, AttributeError):
                    pass
        return inserted
    except Exception:  # noqa: BLE001 - research logging must never break the pipeline
        logger.warning("orderbook snapshot logging failed — will retry next interval", exc_info=True)
        return 0


async def run_orderbook_logger() -> None:
    """Background loop, started alongside the other daemon loops
    (app.web.state's lifespan). A no-op pass (feed not up yet, or its
    orderbook zone is currently empty) is cheap and expected — most sessions
    spend some time with zero unheld tail codes (see _trailing_orderbook_zone)."""
    logger.info("orderbook logger starting — interval=%ss", settings.orderbook_log_interval_seconds)
    while True:
        try:
            feed = kis_market_status.get_active_feed()
            if feed is not None and feed.orderbook_codes:
                count = await log_orderbook_snapshot(feed)
                if count:
                    logger.debug("orderbook logger: wrote %d row(s)", count)
        except Exception:
            # Same posture as the other background loops (poller, stop_loss) —
            # never let one bad pass kill the loop for the rest of the process.
            logger.exception("orderbook logger pass failed — will retry next interval")
        await asyncio.sleep(settings.orderbook_log_interval_seconds)
