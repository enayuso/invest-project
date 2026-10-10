"""Kill switch + daily max-loss circuit breaker — the first gate in
kis_bridge.py's chain, kept in its own module (agent-architecture-roadmap
stage 2) so kis_bridge.py doesn't keep growing every time a new portfolio-
wide safety mechanism is added.

Kill switch is one manual panic button per USER — global across that person's
markets and tickers, but never across people (multi-user migration Phase 4).
It used to be a single row for the whole process, which meant one user's
panic button blocked everyone's orders and any of them could switch it back
off. Daily max-loss is account-balance derived, so it's tracked independently
per user per market (domestic/overseas never combined) — a US-only bad day
never halts KRX buys and vice versa. See app.execution.portfolio_risk's module
docstring for the domestic/overseas separation principle this module follows.
킬스위치는 '사용자별' 전체 정지 — 예전엔 프로세스 전역이라 한 사람이 건 스위치가
모두를 막고 아무나 끌 수 있었음. 일일손실은 사용자·시장별로 독립 추적.

Persisted in Postgres (same asyncpg pool as app.memory.db, same
double-checked-locking schema bootstrap as app.execution.portfolio_risk) so
an engaged kill switch or a mid-day loss halt survives a process restart —
an emergency stop that quietly resets itself on the next deploy isn't one.
킬스위치·일일손실 서킷브레이커 — kis_bridge와 분리된 별도 모듈, 재시작해도 유지
"""

from __future__ import annotations

import asyncio
import logging
import sys
from datetime import date

import asyncpg

from app.clock import get_clock
from app.clock import local_today as _local_today
from app.config import current_account_mode, settings
from app.execution import _schema_bootstrap
from app.execution.portfolio_risk import (
    AccountUnavailable,
    Market,
    cached_account_snapshot,
    real_account_snapshot,
    ticker_market,
)
from app.memory.db import get_pool
from app.user_context import get_current_user_id
from app.execution import risk_rules

logger = logging.getLogger(__name__)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS execution_safety_state (
    id                      INTEGER PRIMARY KEY DEFAULT 1,
    kill_switch_engaged     BOOLEAN NOT NULL DEFAULT FALSE,
    kill_switch_reason      TEXT,
    updated_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT single_row CHECK (id = 1)
);

-- Replaces the old single daily_loss_day/daily_loss_start_equity columns
-- above (which used to be blended across the whole trade allowlist) — a new
-- table since the key shape changes (single row -> market-keyed), same
-- convention as portfolio_risk.py's portfolio_drawdown_state. The old
-- columns are simply left unused (harmless, no migration needed for a
-- personal single-user system).
CREATE TABLE IF NOT EXISTS daily_loss_state (
    market                  TEXT NOT NULL CHECK (market IN ('domestic', 'overseas')),
    daily_loss_day          DATE NOT NULL,
    daily_loss_start_equity DOUBLE PRECISION NOT NULL,
    updated_at              TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Paper/real account separation (2026-08-04 — see project memory:
-- user-account-mode-db-separation-design, priority-3 "여유 있을 때" item).
-- Before this column, a paper-era daily-loss baseline (a completely
-- different equity scale from a real account) would carry straight over
-- the moment kis_account_mode flips to 'real' — same class of bug as
-- portfolio_drawdown_state's HWM, just lower severity here since a stale
-- baseline self-corrects the next calendar day regardless (this table's
-- own existing docstring already noted that). Same unique-index-instead-
-- of-ADD-CONSTRAINT approach as portfolio_drawdown_state, for the same
-- reason: Postgres has no `ADD CONSTRAINT IF NOT EXISTS` and this schema
-- block re-runs on every process start.
ALTER TABLE daily_loss_state DROP CONSTRAINT IF EXISTS daily_loss_state_pkey;
ALTER TABLE daily_loss_state ADD COLUMN IF NOT EXISTS account_mode TEXT NOT NULL DEFAULT 'paper'
    CHECK (account_mode IN ('paper', 'real'));

-- Per-user separation (multi-user migration Phase 4, plan §2.2).
--
-- The kill switch was ONE row for the whole process (`id = 1`, enforced by a
-- CHECK). With several users that means user A's kill switch blocks user B's
-- orders, and A can switch B's back off — a safety control that reaches into
-- someone else's account is worse than no safety control, because both people
-- believe it is theirs.
-- 킬스위치가 전역이면 A가 건 스위치로 B의 주문이 막히고, A가 B의 것을 끌 수도
-- 있음 — 내 것이라고 믿는 안전장치가 남의 계좌에 작용하는 상태.
--
-- The single-row CHECK and the primary key both have to go: `id` defaults to
-- 1, so a second user's insert would collide on it. Uniqueness moves to
-- user_id. Existing rows backfill to 'default', matching how account_mode was
-- backfilled to 'paper'.
ALTER TABLE execution_safety_state DROP CONSTRAINT IF EXISTS single_row;
ALTER TABLE execution_safety_state DROP CONSTRAINT IF EXISTS execution_safety_state_pkey;
ALTER TABLE execution_safety_state ADD COLUMN IF NOT EXISTS user_id TEXT NOT NULL DEFAULT 'default';
CREATE UNIQUE INDEX IF NOT EXISTS execution_safety_state_user_uidx
    ON execution_safety_state (user_id);

-- The daily-loss breaker is the same story one level down: one baseline per
-- (market, account_mode) meant every user's losses were measured against, and
-- tripped by, a single shared equity curve.
ALTER TABLE daily_loss_state ADD COLUMN IF NOT EXISTS user_id TEXT NOT NULL DEFAULT 'default';
DROP INDEX IF EXISTS daily_loss_state_market_mode_uidx;
CREATE UNIQUE INDEX IF NOT EXISTS daily_loss_state_market_mode_user_uidx
    ON daily_loss_state (market, account_mode, user_id);
"""

_schema_ready = False
_schema_lock = asyncio.Lock()


async def _ready_pool() -> asyncpg.Pool:
    """Double-checked locking around the first-time DDL — shared with the
    other app.execution modules via _schema_bootstrap."""
    return await _schema_bootstrap.ready_pool(sys.modules[__name__], _SCHEMA)


async def _row() -> asyncpg.Record | None:
    pool = await _ready_pool()
    async with pool.acquire() as conn:
        return await conn.fetchrow(
            "SELECT * FROM execution_safety_state WHERE user_id = $1", get_current_user_id()
        )


async def kill_switch_status() -> tuple[bool, str | None, bool]:
    """(engaged, reason, state_unknown). Fail-closed: a DB failure blocks
    (engaged=True) unless kis_safety_fail_open_when_unknown is explicitly
    set — same policy as the circuit-breaker gate's fail-closed-on-unknown
    default. `state_unknown` distinguishes "the switch was actually read as
    on" from "we couldn't read the switch at all", so callers don't report
    a DB hiccup to a human as if someone had thrown the panic switch."""
    try:
        row = await _row()
    except Exception as exc:  # noqa: BLE001 - any failure to read must fail closed, not just
        # (OSError, asyncpg.PostgresError) — a bad database_url, a pool in a
        # closing/contended state (asyncpg.InterfaceError, which is NOT a
        # PostgresError subclass), or any other read failure all mean the
        # same thing here: this gate's state is unknown, and unknown must
        # block, not crash the caller. Matches daily_loss_block's identical
        # broad-except policy just below, which this previously didn't.
        # No exc_info here (2026-08-06) — this fires on every DB-pool-
        # contention hiccup (dozens of times in a single busy day, same
        # asyncpg TimeoutError shape each time) and a full traceback per
        # occurrence was drowning out the rest of the log for a failure
        # mode this message already identifies precisely.
        logger.warning("safety_switch kill_switch_status failed: %s", exc)
        if settings.kis_safety_fail_open_when_unknown:
            return False, None, True
        return True, "safety state unreadable — failing closed", True
    if row is None:
        return False, None, False
    return row["kill_switch_engaged"], row["kill_switch_reason"], False


async def engage_kill_switch(reason: str) -> None:
    pool = await _ready_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO execution_safety_state (id, user_id, kill_switch_engaged, kill_switch_reason, updated_at)
            VALUES (1, $2, TRUE, $1, now())
            ON CONFLICT (user_id) DO UPDATE
            SET kill_switch_engaged = TRUE, kill_switch_reason = $1, updated_at = now()
            """,
            reason, get_current_user_id(),
        )
    logger.warning("KILL SWITCH ENGAGED for user_id=%s: %s", get_current_user_id(), reason)


async def disengage_kill_switch() -> None:
    pool = await _ready_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO execution_safety_state (id, user_id, kill_switch_engaged, kill_switch_reason, updated_at)
            VALUES (1, $1, FALSE, NULL, now())
            ON CONFLICT (user_id) DO UPDATE
            SET kill_switch_engaged = FALSE, kill_switch_reason = NULL, updated_at = now()
            """,
            get_current_user_id(),
        )
    logger.warning("kill switch disengaged for user_id=%s", get_current_user_id())


async def _daily_loss_baseline(market: Market, today: date, current_total: float) -> float:
    """The equity baseline today's loss is measured against, for THIS
    market only — today's own value at the first check of the day if one
    hasn't been recorded yet, else the already-persisted one. Resets
    automatically on a new day, independently per market.

    Scoped to the CURRENT account_mode (2026-08-04) — a paper-era baseline
    (a completely different equity scale) must never apply to real trading
    or vice versa, same reasoning as portfolio_risk's HWM."""
    pool = await _ready_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT daily_loss_day, daily_loss_start_equity FROM daily_loss_state "
            "WHERE market = $1 AND account_mode = $2 AND user_id = $3",
            market, current_account_mode(), get_current_user_id(),
        )
        if row is not None and row["daily_loss_day"] == today:
            return row["daily_loss_start_equity"]

        await conn.execute(
            """
            INSERT INTO daily_loss_state (market, account_mode, user_id, daily_loss_day, daily_loss_start_equity, updated_at)
            VALUES ($1, $2, $3, $4, $5, now())
            ON CONFLICT (market, account_mode, user_id) DO UPDATE
            SET daily_loss_day = $4, daily_loss_start_equity = $5, updated_at = now()
            """,
            market, current_account_mode(), get_current_user_id(), today, current_total,
        )
    return current_total


async def reset_daily_loss_baseline(market: Market) -> None:
    """Deletes today's persisted daily-loss baseline for `market` under the
    CURRENT account_mode — same explicit-only convention as
    portfolio_risk.reset_high_water_mark, bundled into the same CLI reset
    command for convenience even though this one self-corrects on the next
    calendar day regardless. Scoped to account_mode (2026-08-04) so
    resetting paper's baseline can never touch real's row or vice versa.
    모의계좌 리셋 직후 수동 실행 전용."""
    pool = await _ready_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            "DELETE FROM daily_loss_state WHERE market = $1 AND account_mode = $2 AND user_id = $3",
            market, current_account_mode(), get_current_user_id(),
        )
    logger.warning("safety_switch: daily-loss baseline reset for %s (%s)", market, current_account_mode())


async def daily_loss_status(market: Market) -> dict:
    """Today's baseline/current equity and loss fraction for THIS market —
    never raises, degrades to an 'unavailable' shape on failure (this is a
    reporting endpoint for the dashboard, not a gate, so it stays
    fail-open/degrade-quietly for any failure including AccountUnavailable,
    same as before)."""
    try:
        # Reporting endpoint, not a gate (docstring) — the short read-only
        # cache is fine. daily_loss_block below deliberately keeps
        # real_account_snapshot: it IS a gate.
        snapshot = await cached_account_snapshot(market)
        baseline = await _daily_loss_baseline(market, _local_today(market), snapshot.total_value)
    except Exception as exc:  # noqa: BLE001 - reporting endpoint, never fail the whole page over this
        logger.warning("safety_switch daily_loss_status(%s) failed: %s", market, exc)  # see kill_switch_status above for why no exc_info
        return {"available": False}
    loss_frac = (baseline - snapshot.total_value) / baseline if baseline > 0 else 0.0
    return {
        "available": True,
        # Native per-market currency (KRW domestic, USD overseas) — the
        # dashboard only ever reads loss_fraction/threshold (both unitless),
        # never these two directly, so the field names dropped their
        # earlier "_usd" suffix now that domestic no longer holds a
        # USD-converted number under it.
        "start_equity": baseline,
        "current_equity": snapshot.total_value,
        "loss_fraction": loss_frac,
        "threshold": risk_rules.rule("kis_daily_max_loss_pct"),
        "breached": loss_frac >= risk_rules.rule("kis_daily_max_loss_pct"),
    }


async def daily_loss_block(side: str, ticker: str) -> str | None:
    """A disposition string if today's aggregate loss (for `ticker`'s own
    market — domestic/overseas never combined) has crossed the configured
    threshold and this order would increase exposure (a BUY), else None.
    Sells always pass — reducing exposure is never blocked, same philosophy
    as portfolio_risk's other portfolio-wide caps.

    Two distinct failure modes, both fail-closed but for different reasons:
    the real account snapshot being unavailable (AccountUnavailable) is
    ALWAYS fail-closed, unconditionally — we can't verify real exposure at
    all, same policy as portfolio_risk.py's own AccountUnavailable handling.
    A Postgres-only failure reading/writing the baseline row is a lesser,
    pre-existing failure mode that stays governed by the
    kis_safety_fail_open_when_unknown flag, unchanged from before.
    실제 계좌 조회 실패는 무조건 fail-closed, DB(baseline 저장소) 실패는 기존 플래그 그대로."""
    if side != "buy":
        return None
    market = ticker_market(ticker)
    try:
        snapshot = await real_account_snapshot(market)
    except AccountUnavailable as exc:
        return f"BLOCKED: {market} account snapshot unavailable ({exc}) — failing closed"

    try:
        baseline = await _daily_loss_baseline(market, _local_today(market), snapshot.total_value)
    except Exception as exc:  # noqa: BLE001 - see fail-closed policy below
        logger.warning("safety_switch daily_loss_block(%s) failed, treating as unknown: %s", market, exc)  # see kill_switch_status above for why no exc_info
        if settings.kis_safety_fail_open_when_unknown:
            return None
        return "BLOCKED: daily loss state unreadable — failing closed (kis_safety_fail_open_when_unknown=false)"

    if baseline <= 0:
        return None
    loss_frac = (baseline - snapshot.total_value) / baseline
    if loss_frac >= risk_rules.rule("kis_daily_max_loss_pct"):
        return (
            f"BLOCKED: today's {market} loss ({loss_frac:.1%}) has reached the daily max-loss "
            f"breaker ({risk_rules.rule('kis_daily_max_loss_pct'):.1%}) — new buys are halted until the next trading day"
        )
    return None


async def safety_gate_block(side: str, ticker: str) -> str | None:
    """Combined kill-switch + daily-loss check — the very first gate both
    kis_bridge.py entry points call, before any per-ticker check. Kill
    switch is account-wide; daily-loss is scoped to `ticker`'s own market."""
    engaged, reason, state_unknown = await kill_switch_status()
    if engaged:
        if state_unknown:
            return f"BLOCKED: kill-switch state unreadable ({reason or 'DB failure'}) — failing closed"
        return f"BLOCKED: kill switch engaged ({reason or 'no reason recorded'})"
    return await daily_loss_block(side, ticker)
