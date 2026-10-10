"""In-memory live-tick cache — the dashboard's chart data source.

The in-memory deque is still the hot path every read goes through (no DB
round-trip per chart render). At startup, each ticker's buffer is backfilled
two ways, tried in order: (1) `seed_from_intraday_series` pulls the whole
trading day's 1-minute bars straight from yfinance — Yahoo has been tracking
the full session regardless of whether our own poller was running, so this
alone gives a complete chart from market open even on a first-ever process
start; (2) `load_snapshot` restores our own periodic Postgres snapshot as a
fallback for whatever (1) doesn't cover (e.g. yfinance temporarily
unreachable). Both only fill in if nothing's recorded yet this process, so
neither ever clobbers fresher live data. The Postgres snapshot itself is a
periodic backup, not a second source of truth: only one row per ticker,
overwritten each snapshot, not an ever-growing tick log — that cost still
isn't worth it for what this is. Everything here degrades quietly if its
data source isn't reachable, same convention as app.live.alerts /
app.memory.store.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections import deque
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger("app.live.store")

_MAX_TICKS_PER_TICKER = 500  # ~a trading day at 1-minute resolution

_SCHEMA = """
CREATE TABLE IF NOT EXISTS live_ticks_snapshot (
    ticker     TEXT PRIMARY KEY,
    ticks      JSONB NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""

# `persist_snapshot`/`load_snapshot` run concurrently across the whole
# trade allowlist (startup fan-out, and every poll cycle) — bare `CREATE TABLE IF
# NOT EXISTS` isn't safe under real concurrency (the existence check and the
# create aren't atomic across sessions), so this gates it to run exactly
# once per process instead of once per concurrent caller.
_schema_ready = False
_schema_lock = asyncio.Lock()


async def _ensure_schema(conn) -> None:
    global _schema_ready
    if _schema_ready:
        return
    async with _schema_lock:
        if _schema_ready:
            return
        await conn.execute(_SCHEMA)
        _schema_ready = True


@dataclass
class Tick:
    price: float
    volume: int
    timestamp: datetime


_ticks: dict[str, deque[Tick]] = {}


def record_tick(ticker: str, tick: Tick) -> None:
    _ticks.setdefault(ticker, deque(maxlen=_MAX_TICKS_PER_TICKER)).append(tick)


def get_ticks(ticker: str) -> list[Tick]:
    return list(_ticks.get(ticker, ()))


def latest(ticker: str) -> Tick | None:
    bucket = _ticks.get(ticker)
    return bucket[-1] if bucket else None


def _to_json(tick: Tick) -> dict:
    return {"price": tick.price, "volume": tick.volume, "timestamp": tick.timestamp.isoformat()}


def _from_json(d: dict) -> Tick:
    return Tick(price=d["price"], volume=d["volume"], timestamp=datetime.fromisoformat(d["timestamp"]))


async def seed_from_intraday_series(ticker: str) -> None:
    """Backfills from yfinance's own full-day intraday series — see the
    module docstring. Tried before `load_snapshot` at startup since it's the
    more complete, authoritative source when it's available."""
    if _ticks.get(ticker):
        return
    try:
        from app.dataflows.yfinance import get_intraday_series  # avoids a hard dependency at module load

        quotes = await get_intraday_series(ticker)
    except Exception:
        logger.warning("could not backfill intraday series for %s", ticker, exc_info=True)
        return
    if quotes:
        _ticks[ticker] = deque(
            (Tick(price=q.price, volume=q.volume, timestamp=q.timestamp) for q in quotes[-_MAX_TICKS_PER_TICKER:]),
            maxlen=_MAX_TICKS_PER_TICKER,
        )


async def persist_snapshot(ticker: str) -> None:
    try:
        from app.memory.db import get_pool  # local import: avoids a hard dependency at module load

        pool = await get_pool()
        payload = json.dumps([_to_json(t) for t in get_ticks(ticker)])
        async with pool.acquire() as conn:
            await _ensure_schema(conn)
            await conn.execute(
                """
                INSERT INTO live_ticks_snapshot (ticker, ticks, updated_at)
                VALUES ($1, $2::jsonb, now())
                ON CONFLICT (ticker) DO UPDATE SET ticks = $2::jsonb, updated_at = now()
                """,
                ticker,
                payload,
            )
    except Exception as exc:
        # No exc_info (2026-08-06) — fires on the same DB-pool-contention
        # hiccup as app.execution.safety_switch's fail-closed reads (11x in
        # one day), and the message already names the exception.
        logger.warning("could not persist tick snapshot for %s — continuing without it: %s", ticker, exc)


async def forget_ticker(ticker: str) -> None:
    """Drops every trace of `ticker` from this module — the in-memory tick
    buffer AND its persisted snapshot row. Called from
    `app.web.state.remove_ticker_from_trade_allowlist` (2026-08-13) when a
    ticker leaves the trade allowlist.

    Without this, re-adding the same ticker later would silently resurrect
    stale history two different ways: `_ticks` is keyed by ticker and never
    expires on its own, so the old in-memory deque would still be sitting
    there (`seed_from_intraday_series`/`load_snapshot` both no-op once
    `_ticks.get(ticker)` is truthy, so neither would ever refresh it); and
    even if the process restarted in between, `load_snapshot` would
    re-hydrate the exact same stale Postgres row. Either way, the chart and
    every alert threshold that reads `get_ticks`/`latest` would present
    ticks from whenever the ticker was last watched as if they were fresh,
    with no visible sign that the buffer predates the re-add.
    """
    _ticks.pop(ticker, None)
    try:
        from app.memory.db import get_pool  # local import: avoids a hard dependency at module load

        pool = await get_pool()
        async with pool.acquire() as conn:
            await _ensure_schema(conn)
            await conn.execute("DELETE FROM live_ticks_snapshot WHERE ticker = $1", ticker)
    except Exception:
        logger.warning("could not delete persisted tick snapshot for %s — continuing", ticker, exc_info=True)


async def load_snapshot(ticker: str) -> None:
    """Restores the last-persisted snapshot on startup so charts don't start
    blank/flat after a restart. Only fills in if this ticker has no
    in-memory ticks yet, so it can never clobber fresher live data."""
    if _ticks.get(ticker):
        return
    try:
        from app.memory.db import get_pool

        pool = await get_pool()
        async with pool.acquire() as conn:
            await _ensure_schema(conn)
            row = await conn.fetchrow("SELECT ticks FROM live_ticks_snapshot WHERE ticker = $1", ticker)
        if row is None:
            return
        restored = [_from_json(d) for d in json.loads(row["ticks"])]
        if restored:
            _ticks[ticker] = deque(restored[-_MAX_TICKS_PER_TICKER:], maxlen=_MAX_TICKS_PER_TICKER)
    except Exception:
        logger.warning("could not restore tick snapshot for %s — starting empty", ticker, exc_info=True)
