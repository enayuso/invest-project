"""Deterministic, code-only alert thresholds — no LLM judgment involved,
same principle as the execution ledger's risk controls (app.execution.sizing):
anything that must reliably fire belongs in code, not a model's discretion.

Two trigger conditions, checked on every poll:
- price moved more than `settings.alert_pct_threshold` over the last
  `settings.alert_lookback_ticks` polls
- current volume is more than `settings.alert_volume_multiplier` times the
  average of the same lookback window

`check_thresholds` is pure and synchronous (no I/O) so it's cheaply
unit-testable; persistence is a separate async step that degrades quietly
if Postgres isn't reachable, same convention as app.memory.store /
app.execution.ledger.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import datetime

from app.config import current_account_mode, settings
from app.live.store import get_ticks
from app.user_context import get_current_user_id

logger = logging.getLogger("app.live.alerts")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS alerts (
    id           SERIAL PRIMARY KEY,
    ticker       TEXT NOT NULL,
    kind         TEXT NOT NULL,
    message      TEXT NOT NULL,
    triggered_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Multi-instance seam (agent-architecture-roadmap stage 4) — see
-- app.memory.db's identical column for the rationale.
ALTER TABLE alerts ADD COLUMN IF NOT EXISTS user_id TEXT NOT NULL DEFAULT 'default';

-- Same axis every other execution/memory table scopes on (ledger.py,
-- safety_switch.py, etc.) — without it, a paper-mode price move and a
-- real-mode one for the same ticker are indistinguishable in the alert feed.
ALTER TABLE alerts ADD COLUMN IF NOT EXISTS account_mode TEXT NOT NULL DEFAULT 'paper';

-- Ties every row to a registered person — same reasoning and same
-- pg_constraint-guarded re-run as app.execution.ledger's identical trades
-- FK. Safe here because this module's _ensure_schema only ever runs via a
-- pool from app.memory.db.get_pool(), whose own _SCHEMA (creating `users`
-- and its 'default' seed row) always executes first on that pool.
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'alerts_user_id_fkey') THEN
        ALTER TABLE alerts ADD CONSTRAINT alerts_user_id_fkey
            FOREIGN KEY (user_id) REFERENCES users(user_id);
    END IF;
END $$;

-- Retention is enforced by `prune_old_alerts` (a boot-time DELETE), not a
-- partial index or a trigger. This index matches recent_alerts' query shape
-- (WHERE user_id, account_mode ORDER BY triggered_at DESC) so that lookup
-- stays an index scan instead of a growing seq scan. The prune's DELETE
-- filters on triggered_at alone and deliberately runs as a seq scan — at
-- once-a-day frequency on a small table that is cheaper than the write cost
-- an extra index would add to every INSERT.
CREATE INDEX IF NOT EXISTS idx_alerts_user_account_triggered
    ON alerts (user_id, account_mode, triggered_at DESC);
"""

# Boot-time prune horizon. Do NOT lower below ~30 days: slippage analysis and
# the re-entry-prose validation pass routinely look a month or more back
# (forced-exit-threshold-price-contamination-fix, 2026-08-27). A module
# constant, not a settings field — this is housekeeping with no per-deploy
# reason to tune (same posture as app.daemon._LOG_BACKUP_DAYS), and the single
# in-house deployment has no environment-specific variation. Boot-time-only is
# enough because the server loses power nightly and on weekends
# (server-power-schedule-blocker), so this process restarts most days anyway.
_ALERT_RETENTION_DAYS = 90

# Unlike `app.live.store`'s analogous tables, this used to run `CREATE TABLE
# IF NOT EXISTS` unconditionally on every single call — harmless once the
# table exists, but the *first* concurrent pair (two trade allowlist tickers both
# alerting in the same poll tick, via `run_poller_loop`'s `asyncio.gather`)
# could race the non-atomic existence-check-then-create, raise, and get
# silently swallowed by the blanket `except Exception` below, dropping that
# alert. Same double-checked-locking fix as `app.live.store._ensure_schema`.
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
class Alert:
    ticker: str
    kind: str
    message: str
    triggered_at: datetime


def check_thresholds(ticker: str) -> list[Alert]:
    window = get_ticks(ticker)[-(settings.alert_lookback_ticks + 1) :]
    if len(window) < 2:
        return []

    latest_tick, baseline = window[-1], window[0]
    alerts: list[Alert] = []

    if baseline.price:
        pct_move = (latest_tick.price - baseline.price) / baseline.price
        if abs(pct_move) >= settings.alert_pct_threshold:
            alerts.append(
                Alert(
                    ticker=ticker,
                    kind="price_move",
                    message=(
                        f"{ticker} moved {pct_move:+.2%} over the last "
                        f"{settings.alert_lookback_ticks} polls (now {latest_tick.price:.2f})"
                    ),
                    triggered_at=latest_tick.timestamp,
                )
            )

    prior = window[:-1]
    avg_volume = sum(t.volume for t in prior) / len(prior)
    if avg_volume and latest_tick.volume >= settings.alert_volume_multiplier * avg_volume:
        alerts.append(
            Alert(
                ticker=ticker,
                kind="volume_spike",
                message=(
                    f"{ticker} volume spiked to {latest_tick.volume:,} "
                    f"(avg over window: {avg_volume:,.0f})"
                ),
                triggered_at=latest_tick.timestamp,
            )
        )

    return alerts


# (account_mode, ticker, kind) -> monotonic time last fired. In-memory only,
# like app.live.store's tick cache — a cooldown is a rate limit on THIS
# process's emission, not a fact that needs to survive a restart.
# account_mode is part of the key (2026-08-25) because it is NOT constant for
# the process: the mode is a contextvar (app.config.current_account_mode) that
# the dashboard can flip at runtime, so a paper-mode alert for a ticker could
# suppress the real-money alert for the same (ticker, kind) for up to
# `alert_cooldown_seconds` — including a stop-loss-adjacent price move.
_last_fired_at: dict[tuple[str, str, str], float] = {}


def forget_ticker(ticker: str) -> None:
    """Drops every cooldown entry keyed to `ticker` (both alert kinds) —
    called from `app.web.state.remove_ticker_from_trade_allowlist`
    (2026-08-13) when a ticker leaves the trade allowlist. Without this, an
    alert that fired shortly before removal would keep suppressing a
    re-fire on the SAME (ticker, kind) for up to `alert_cooldown_seconds`
    even after the ticker is re-added — even though `app.live.store.
    forget_ticker` has already cleared the tick window `check_thresholds`
    would use to decide whether a fresh alert is warranted, so the
    suppression would be enforcing a stale cooldown against what is
    effectively a brand-new observation window. Clears the entry under
    every account_mode."""
    # k[1] is the ticker; every account_mode's entry for it is dropped —
    # over-clearing a cooldown can only cause an extra alert, never a
    # suppressed one.
    for key in [k for k in _last_fired_at if k[1] == ticker]:
        _last_fired_at.pop(key, None)


def filter_new_alerts(alerts: list[Alert]) -> list[Alert]:
    """Drops any alert whose (ticker, kind) fired within
    `settings.alert_cooldown_seconds` — without this, a price move or volume
    spike that stays past threshold for several consecutive polls re-emits an
    identical alert every single poll, burying anything else in the feed.
    Call this on `check_thresholds`' output before logging/persisting it;
    `check_thresholds` itself stays pure/synchronous (no mutable state) for
    its own unit tests."""
    now = time.monotonic()
    fresh: list[Alert] = []
    for alert in alerts:
        key = (current_account_mode(), alert.ticker, alert.kind)
        last = _last_fired_at.get(key)
        if last is not None and now - last < settings.alert_cooldown_seconds:
            continue
        _last_fired_at[key] = now
        fresh.append(alert)
    return fresh


async def record_alert(alert: Alert) -> None:
    try:
        from app.memory.db import get_pool  # local import: avoids a hard dependency at module load

        pool = await get_pool()
        async with pool.acquire() as conn:
            await _ensure_schema(conn)
            await conn.execute(
                "INSERT INTO alerts (ticker, kind, message, triggered_at, user_id, account_mode) "
                "VALUES ($1, $2, $3, $4, $5, $6)",
                alert.ticker,
                alert.kind,
                alert.message,
                alert.triggered_at,
                get_current_user_id(),
                current_account_mode(),
            )
    except Exception:
        logger.warning("could not persist alert for %s — continuing without it", alert.ticker, exc_info=True)


async def prune_old_alerts() -> int:
    """Deletes alert rows older than `_ALERT_RETENTION_DAYS` and returns the
    count removed. Run once at startup (app.web.state.lifespan) as a
    backgrounded task — global across all users/modes. Degrades quietly like
    every other DB path in this module: a failure logs and returns 0, never
    blocks boot."""
    try:
        from app.memory.db import get_pool  # local import: avoids a hard dependency at module load

        pool = await get_pool()
        async with pool.acquire() as conn:
            await _ensure_schema(conn)
            result = await conn.execute(
                "DELETE FROM alerts WHERE triggered_at < now() - ($1 || ' days')::interval",
                str(_ALERT_RETENTION_DAYS),
            )
        # asyncpg's execute() returns the command tag, e.g. "DELETE 42".
        deleted = int(result.split()[-1]) if result.startswith("DELETE") else 0
        if deleted:
            logger.info("pruned %d alert row(s) older than %d days", deleted, _ALERT_RETENTION_DAYS)
        return deleted
    except Exception:
        logger.warning("could not prune old alerts — continuing", exc_info=True)
        return 0


async def recent_alerts(limit: int = 50) -> list[Alert]:
    try:
        from app.memory.db import get_pool

        pool = await get_pool()
        async with pool.acquire() as conn:
            await _ensure_schema(conn)
            rows = await conn.fetch(
                "SELECT ticker, kind, message, triggered_at FROM alerts "
                "WHERE user_id = $1 AND account_mode = $2 ORDER BY triggered_at DESC LIMIT $3",
                get_current_user_id(),
                current_account_mode(),
                limit,
            )
        return [Alert(r["ticker"], r["kind"], r["message"], r["triggered_at"]) for r in rows]
    except Exception as exc:
        # No exc_info (2026-08-06) — fires on the same DB-pool-contention
        # hiccup as app.execution.safety_switch's fail-closed reads, and a
        # full traceback per occurrence was drowning out the rest of the
        # log for a failure this one-line message already names.
        logger.warning("could not read alert history — returning empty: %s", exc)
        return []
