"""Paper execution ledger: turns a target allocation into an actual
simulated trade, position-sized and stop-loss enforced in code instead of
described in prose. Reuses the Postgres+pgvector instance from Phase 3
(`app/memory/db.py`) rather than a second database — one standalone service,
two purposes.

Every public function degrades quietly (log + no-op) if Postgres isn't
reachable — same convention as `app/memory/store.py` — so a DB outage never
crashes decision-making, though it does mean a trade silently wasn't
recorded (logged at warning level).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import math
import re
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass, replace as _dataclasses_replace
from datetime import date, datetime, timezone
from typing import Literal, NamedTuple

import asyncpg

from app.clock import get_clock, local_today
from app.config import ExecutionMode, current_account_mode, is_backtest_context, settings
from app.domain import executability
from app.execution import _schema_bootstrap, runtime_settings, trailing_stop
from app.execution import costs
from app.dataflows.kis_provider import get_daily_high_series, get_daily_low_series
from app.dataflows.yfinance import IntradayBar, get_intraday_bars, get_ticker_currency, get_usd_fx_rate
from app.execution.kis_bridge import max_order_value_cap, mirror_trade_to_kis, route_ticker
from app.live.alerts import Alert, filter_new_alerts, record_alert
from app.live.last_price import freshest_tick
from app.execution.portfolio_risk import (
    PORTFOLIO_RISK_UNAVAILABLE_NOTE_PREFIX,
    PortfolioSnapshot,
    apply_backtest_portfolio_caps,
    apply_live_portfolio_caps,
    cached_account_snapshots,
    cached_balance_fetches,
    ledger_inquiry_count,
    market_lock,
    real_equity_native,
    real_equity_native_detailed,
    real_shares_held,
    ticker_market,
)
from app.execution.sizing import rebalance_action_label
from app.memory.db import get_pool
from app.state import PortfolioRating
from app.user_context import get_current_user_id
from app.execution import risk_rules

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS trades (
    id           SERIAL PRIMARY KEY,
    ticker       TEXT NOT NULL,
    trade_date   DATE NOT NULL,
    side         TEXT NOT NULL,
    shares       INTEGER NOT NULL,
    price        DOUBLE PRECISION NOT NULL,
    cash_after   DOUBLE PRECISION NOT NULL,
    shares_after INTEGER NOT NULL,
    stop_loss    DOUBLE PRECISION,
    reason       TEXT NOT NULL,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- The table above predates this column, so it's an ALTER rather than part
-- of the CREATE — links a trade back to the memory_entries decision that
-- produced it (NULL for stop-loss-triggered exits and manual dashboard
-- orders, neither of which is tied to a specific decision). Without this,
-- the dashboard could only join a decision to a trade by (ticker,
-- trade_date), which collides whenever the same ticker trades more than
-- once on the same day (e.g. a rebalance trade plus a same-day stop-loss
-- exit) and showed the wrong trade next to a decision's rating.
ALTER TABLE trades ADD COLUMN IF NOT EXISTS decision_id INTEGER;

-- Multi-instance seam (agent-architecture-roadmap stage 4) — see
-- app.memory.db's identical column for the rationale; same "tag now while
-- cheap, not yet an enforced boundary" caveat applies here too.
ALTER TABLE trades ADD COLUMN IF NOT EXISTS user_id TEXT NOT NULL DEFAULT 'default';

-- RebalanceResult.adjustment_note (1%-risk-cap / portfolio-cap / no-stop
-- guard text) previously only reached persistence via memory.append_execution_note,
-- which is skipped whenever decision_id is None and buries the note as free
-- text inside memory_entries.decision_text either way -- not queryable per
-- trade. Storing it directly on the row makes "which trades got capped, and
-- by how much" a plain column read instead of a decision_text string-search.
-- 리스크캡이 주문을 축소시킨 이유를 trades 행에 직접 저장 (decision_id 없어도 유지됨).
ALTER TABLE trades ADD COLUMN IF NOT EXISTS adjustment_note TEXT;

-- Paper/real account separation (2026-08-04 — see project memory:
-- user-account-mode-db-separation-design). Before this column, every mode
-- shared one ledger, so get_open_stop_loss/get_position/get_last_trade
-- would trust whichever mode's row happened to be most recent for a
-- ticker — e.g. a paper-era row showing a ticker "flat" would make
-- check_stops_for_ticker skip a real position's stop-loss monitoring
-- entirely, never even reaching the real-balance check that would have
-- caught it. Backfilled to 'paper' since every row recorded before this
-- migration was written while kis_account_mode defaulted to 'paper' (this
-- project has not run in 'real' mode yet). Every read/write below now
-- scopes to current_account_mode() so paper and real never see each
-- other's rows — same principle already applied to KIS credentials
-- themselves (fully separate app_key/secret/base_url per mode).
ALTER TABLE trades ADD COLUMN IF NOT EXISTS account_mode TEXT NOT NULL DEFAULT 'paper'
    CHECK (account_mode IN ('paper', 'real'));

-- Ties every row to a registered person (app.memory.db's `users` table,
-- seeded with 'default' before this ever runs — see that module's _SCHEMA
-- ordering comment) instead of trusting user_id to be a typo-free free
-- string forever. ADD CONSTRAINT has no IF NOT EXISTS form in Postgres, so
-- this guards re-runs itself via pg_constraint instead.
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'trades_user_id_fkey') THEN
        ALTER TABLE trades ADD CONSTRAINT trades_user_id_fkey
            FOREIGN KEY (user_id) REFERENCES users(user_id);
    END IF;
END $$;

-- What actually happened to this trade's KIS submission — "submitted: ..."
-- (mirror_trade_to_kis actually sent it), "skipped: ..."/"PENDING_USER_APPROVAL: ..."
-- (deliberately never sent — MANUAL mode, backtest replay, or still awaiting
-- SEMI_AUTO approval), or NULL (not yet known — the brief window between this
-- row's INSERT and the follow-up UPDATE once the disposition is known, or a
-- reversal row, which never gets one at all). app.execution.reconciliation
-- reads this to tell an actual crash (never reached KIS) apart from a trade
-- that was never SUPPOSED to reach KIS in the first place — before this
-- column existed, every MANUAL/dry-run/still-pending row looked identical to
-- a genuine "possible crash before the KIS mirror completed" to reconciliation,
-- since nothing on `trades` recorded which case it was.
ALTER TABLE trades ADD COLUMN IF NOT EXISTS mirror_disposition TEXT;

-- KIS's own order number for this trade's submission, if it reached KIS at
-- all (app.execution.kis_bridge.mirror_trade_to_kis / place_manual_order) —
-- 2026-08-13, app.live.fill_monitor's exact-match key. Before this, the
-- only trace of the order number was embedded inside the mirror_disposition
-- TEXT ("submitted: ...; order_no=..."), unusable for a real join against
-- KIS's own execution-history TR (BrokerExecution.order_no) — confirming
-- what actually happened to a submitted order meant guessing by
-- (ticker, date, side) instead, which is exactly what mismatched two real
-- delayed-settlement stop-loss exits against nothing and mis-flagged them
-- as crashes (project memory: order-fill-confirmation-gap-2026-08-13).
ALTER TABLE trades ADD COLUMN IF NOT EXISTS kis_order_no TEXT;

-- When app.live.fill_monitor last resolved this order's actual fill outcome
-- against KIS — NULL means still unresolved. Never touches
-- cash/shares/stop_loss — this column is bookkeeping for that loop's own
-- idempotency (so a resolved order isn't re-examined every pass), not a
-- correction mechanism. See that module's own docstring for the detect-
-- only boundary this stays inside (stage 1 of two — stage 2, auto-
-- correcting the ledger once KIS gives a definitive answer, is a
-- deliberate follow-up, not done here).
--
-- 2026-08-27 fix: this used to ALSO get set the first time no matching KIS
-- execution was found at all (the "ambiguous, might just be lagging" case),
-- which permanently stopped that order from ever being looked at again —
-- even though a later pass, within the same _PENDING_LOOKBACK window, could
-- easily have gotten KIS's actual definitive answer by then (confirmed live:
-- both of MRVL's last two orders hit this exact case, one of which turned
-- out to have genuinely filled). This column now means ONLY "genuinely
-- resolved" (a real fill/no-fill answer, or the final _PENDING_LOOKBACK
-- give-up) — see `fill_unresolved_alerted_at` below for the interim
-- "already alerted once, still watching" state that used to be conflated
-- with this one.
ALTER TABLE trades ADD COLUMN IF NOT EXISTS fill_checked_at TIMESTAMPTZ;

-- First time app.live.fill_monitor alerted "no KIS execution record found"
-- for this order without yet giving up on it — separate from
-- `fill_checked_at` specifically so that alert doesn't stop the loop from
-- re-examining the order on every later pass (see that column's own
-- 2026-08-27 comment for the incident this fixes). NULL means either never
-- alerted, or already past this interim stage (resolved one way or the
-- other, `fill_checked_at` is set instead). Used purely to suppress a
-- repeat of the SAME interim alert every remaining pass within
-- _PENDING_LOOKBACK — never read by anything else, and never touches
-- cash/shares/stop_loss.
ALTER TABLE trades ADD COLUMN IF NOT EXISTS fill_unresolved_alerted_at TIMESTAMPTZ;

-- Historical-replay flag (2026-08-10 — see project memory:
-- backtest-live-db-shared-contamination). Before this column, `trades` had
-- no way to tell a backtest replay's rows apart from real paper/real
-- trades — every read below (get_position, get_open_stop_loss, ...) is
-- scoped to account_mode/user_id but NOT to backtest-vs-live, so a
-- backtest run and the live daemon sharing the same DATABASE_URL (true for
-- this project's local dev + deployed server, both pointing at the same
-- Postgres instance) silently interleaved: a backtest replaying a past
-- date could read/extend a real live position, and live's own
-- get_position/get_open_stop_loss/screening "already held" check could
-- pick up a backtest run's speculative rows as if they were real holdings.
-- Confirmed live 2026-08-10: 9 of 12 trailing-stop-sweep tickers already
-- had contaminated rows, including a same-day live BLOCKED-by-allowlist
-- event sitting next to backtest-only reason text
-- ("gap check before ...", only ever written by app.backtest.harness).
-- Every read/write below now scopes to is_backtest_context() the same way
-- account_mode is already scoped, so a backtest replay only ever sees (and
-- only ever writes) its own is_backtest=true rows, and live only ever sees
-- is_backtest=false ones.
ALTER TABLE trades ADD COLUMN IF NOT EXISTS is_backtest BOOLEAN NOT NULL DEFAULT false;
"""

_schema_ready = False
_schema_lock = asyncio.Lock()

# Stable, grep-able `trades.reason` prefixes — written at `_force_exit_locked`
# and `reverse_pending_trade` below, read back by `get_last_trade` (2026-08-14)
# so a caller (app.orchestrator.pipeline._position_context) can tell a
# FORCED stop-loss exit apart from the AI's own Sell decision, and a
# reversed/never-executed trade apart from both, without re-parsing free
# text. Defined once here rather than re-literaled at each read/write site.
STOP_LOSS_REASON_PREFIX = "stop-loss triggered: "
REVERSAL_REASON_PREFIX = "reversal: "


@dataclass
class Position:
    cash: float
    shares: int

    def market_value(self, price: float) -> float:
        return self.cash + self.shares * price


@dataclass
class RebalanceResult:
    """What `rebalance_to_target` actually did, for callers that need to
    reconcile it against a pre-computed narrative (see
    `app.orchestrator.pipeline`) — the risk cap and the no-resolvable-stop
    guard can both silently reduce the trade below what a rating/target
    allocation alone implied, and a caller with no visibility into that
    can't correct a decision's stored narrative to match."""

    side: str | None  # None if no trade was made (Hold, or delta rounded to 0)
    shares: int  # shares actually bought/sold this call, 0 if no trade
    shares_after: int  # resulting position size
    stop_loss: float | None
    # Buy/Sell/Increase/Decrease/Hold — see rebalance_action_label's own
    # docstring for why this is separate from the AI's PortfolioRating.
    action_label: str
    adjustment_note: str | None = None  # set when the risk cap or no-stop guard changed the outcome


async def _ready_pool() -> asyncpg.Pool:
    """Double-checked locking around the first-time DDL — without it,
    concurrent first callers (the stop-loss monitoring loop and a live
    request both touching the ledger at startup, for instance) could race
    `CREATE TABLE IF NOT EXISTS`, which isn't atomic across sessions, and
    raise a duplicate-object error that this function's callers then treat
    as "Postgres unreachable" and silently fall back to a fresh position.
    Locking logic shared via _schema_bootstrap — see that module's own
    docstring for why `_schema_ready`/`_schema_lock` stay module-level
    globals here rather than moving into the shared helper."""
    return await _schema_bootstrap.ready_pool(sys.modules[__name__], _SCHEMA)


async def clear_backtest_positions(tickers: list[str]) -> None:
    """Deletes every `is_backtest=true` row for `tickers` (current
    account_mode/user_id), so a fresh replay always starts flat instead of
    silently inheriting whatever an EARLIER backtest run left behind.

    `is_backtest` (see the column's own schema comment) only tells live and
    backtest rows apart — it does NOT tell two different backtest runs apart
    from each other. Without this, `get_position`'s "most recent row for
    this ticker" logic makes run N+1 of a repeated/swept backtest (e.g.
    `app.backtest.atr_sweep`, which replays the SAME tickers over the SAME
    date range once per parameter candidate, back to back) start from
    wherever run N's replay ended, not flat — defeating the entire point of
    a parameter sweep, which is comparing candidates from an identical
    starting point. See project memory: backtest-live-db-shared-
    contamination's "known residual risk" note.

    Called by `run_backtest`/`run_portfolio_backtest` for their own
    tickers, right before entering `backtest_scope()` — every backtest
    entrypoint gets this for free, not just the sweep. Deliberately does
    NOT degrade quietly on a DB failure (unlike every other function in
    this module): a silent failure here would let exactly the
    cross-contamination this function exists to prevent slip back in
    unnoticed, so a broken run should fail loudly instead of producing
    quietly-wrong comparison results."""
    if not tickers:
        return
    pool = await _ready_pool()
    async with pool.acquire() as conn:
        deleted = await conn.fetch(
            "DELETE FROM trades WHERE ticker = ANY($1) AND account_mode = $2 AND user_id = $3 "
            "AND is_backtest = true RETURNING id",
            tickers,
            current_account_mode(),
            get_current_user_id(),
        )
    if deleted:
        logger.info("clear_backtest_positions: removed %d stale backtest row(s) for %s", len(deleted), tickers)


async def starting_cash_for(ticker: str, *, for_backtest: bool | None = None) -> float:
    """The cash a fresh (no ledger rows) position starts from, converted into
    the ticker's own quote currency.

    WHICH USD constant it scales depends on the context, because the two uses
    are irreconcilable: a historical replay wants a large round denominator
    (`settings.backtest_starting_cash`) so sizing produces meaningful trades,
    while the live/paper internal ledger must sit near the real deposit
    (`settings.paper_starting_cash`) so an empty row or a failed balance lookup
    never narrates a fictional multi-hundred-million-won equity to sizing and
    the five LLM nodes. `is_backtest_context()` is the discriminator.

    `for_backtest` overrides that detection, and exists for one specific caller
    shape: code that reports on a replay from OUTSIDE `backtest_scope()`
    (`app.backtest.reporting`, invoked after the harness's scope has already
    exited). Without the override it would divide a replay's final equity by
    the LIVE constant — a 685x-wrong ledger return under .env.dev, silently.

    The FX conversion itself: without it, sizing a position as `equity / price`
    compares a USD cash figure against a price quoted in a completely different
    currency's units — the exact bug that made every Buy signal on a KRW ticker
    floor to 0 shares (100,000 / ~250,000 < 1). Falls back to the raw USD figure
    on any lookup failure — better to size in the wrong currency's scale than to
    stop trading entirely over a transient FX/currency lookup error, same
    "degrade, don't crash" convention as the rest of this module."""
    if for_backtest is None:
        for_backtest = is_backtest_context()
    base = settings.backtest_starting_cash if for_backtest else settings.paper_starting_cash
    try:
        currency = await get_ticker_currency(ticker)
        fx_rate = await get_usd_fx_rate(currency)
        return base * fx_rate
    except Exception as exc:
        logger.warning("currency/FX lookup failed for %s, using raw USD starting cash: %s", ticker, exc)
        return base


async def get_position(ticker: str, *, for_backtest: bool | None = None) -> Position:
    """Current position for `ticker` under the ambient account scope.

    `for_backtest` overrides the ambient `is_backtest_context()` for callers
    that read a replay's book from OUTSIDE `backtest_scope()` — e.g.
    `app.backtest.reporting`, which renders after the harness's scope has
    already exited. Without it, such a call resolves to the LIVE/real
    account's row for the same ticker instead of the backtest's own row.
    Same convention as `starting_cash_for`'s `for_backtest` parameter.
    """
    if for_backtest is None:
        for_backtest = is_backtest_context()
    try:
        pool = await _ready_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT cash_after, shares_after FROM trades "
                "WHERE ticker = $1 AND account_mode = $2 AND user_id = $3 AND is_backtest = $4 "
                "ORDER BY trade_date DESC, id DESC LIMIT 1",
                ticker,
                current_account_mode(),
                get_current_user_id(),
                for_backtest,
            )
    except Exception as exc:  # noqa: BLE001 - (OSError, PostgresError) missed
        # asyncpg.InterfaceError (pool closing/contended — not a PostgresError
        # subclass) and any other read failure; this module's existing
        # convention is to degrade quietly on any DB failure, not just those two.
        # PostgresError만으로는 못 잡는 InterfaceError 등도 조용히 무시하도록 확장.
        logger.warning("execution get_position failed, assuming starting position: %s", exc)
        # starting_cash_for(ticker), not the raw USD settings.paper_starting_cash
        # — this exact currency-scale bug (a KRW ticker's target flooring to 0
        # shares against a USD-scale cash figure) is what starting_cash_for
        # exists to prevent; the no-row branch just below already gets this
        # right, this except-branch didn't.
        return Position(cash=await starting_cash_for(ticker, for_backtest=for_backtest), shares=0)

    if row is None:
        return Position(cash=await starting_cash_for(ticker, for_backtest=for_backtest), shares=0)
    return Position(cash=row["cash_after"], shares=row["shares_after"])


class LastTrade(NamedTuple):
    """Positionally identical (indices 0-2) to the plain
    `(trade_date, side, price)` tuple this used to return, so existing
    positional reads (`last_trade[1]`, `last_trade[2]`) and `trade_date, side,
    price = last_trade`-style unpacking of the first three fields both still
    work — but a caller that unpacks all four now needs `.reason` (a bare
    3-tuple would raise on that, which is the point: it forces call sites
    that assumed the old shape to be updated deliberately rather than
    silently mis-indexing)."""

    trade_date: date
    side: str
    price: float
    reason: str | None
    # `trades.adjustment_note` (2026-08-27). Defaulted so every existing
    # positional construction of this NamedTuple still type-checks — appended
    # LAST for the same reason `reason` was: indices 0-3 stay put. Read by
    # `app.orchestrator.pipeline._position_context` (via
    # `is_threshold_placeholder_note` below) to tell a forced exit recorded at
    # a real fill price apart from one recorded at the stop THRESHOLD instead
    # (see `_resolve_forced_exit_price`'s own docstring — the 047040.KS
    # id=280 incident this exists to stop from silently repeating in the
    # Portfolio Manager's re-entry narrative).
    adjustment_note: str | None = None


async def get_last_trade(ticker: str) -> LastTrade | None:
    """Date, side, price, and reason of the most recent trade — used to give
    the Portfolio Manager re-entry context (e.g. "flat since selling at $X on
    date Y") instead of reasoning about a position with no memory of how it
    got there. `None` if this ticker has never traded (in the CURRENT
    account_mode and by the CURRENT user — see get_position's identical scoping).

    `reason` (2026-08-14) lets a caller (app.orchestrator.pipeline.
    _position_context) tell a FORCED stop-loss exit (`STOP_LOSS_REASON_PREFIX`)
    apart from the AI's own Sell decision (the rendered portfolio-decision
    narrative) apart from a reversed/never-executed trade
    (`REVERSAL_REASON_PREFIX`) — real incident this fixes: 005930, stopped
    out at 198,714 on 2026-07-29, re-bought 21h later at 214,500 (+7.94%)
    with the Portfolio Manager having no idea the prior exit was a forced
    safety stop rather than its own considered Sell, or that it was now
    paying a meaningfully higher price to get back in. See memory:
    ai-reentry-context-2026-08-14."""
    try:
        pool = await _ready_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT trade_date, side, price, reason, adjustment_note FROM trades "
                "WHERE ticker = $1 AND account_mode = $2 AND user_id = $3 AND is_backtest = $4 "
                "ORDER BY trade_date DESC, id DESC LIMIT 1",
                ticker,
                current_account_mode(),
                get_current_user_id(),
                is_backtest_context(),
            )
    except Exception as exc:  # noqa: BLE001 - same broadened fail-open policy as get_position above
        logger.warning("execution get_last_trade failed: %s", exc)
        return None

    if row is None:
        return None
    return LastTrade(row["trade_date"], row["side"], row["price"], row["reason"], row["adjustment_note"])


# How long a forced-exit row (STOP_LOSS_REASON_PREFIX) is treated as
# "not yet confirmed" by _is_provisional_forced_exit below — bounded so a
# `_record_mirror_disposition` that silently never lands (a second DB
# failure right after the first) can't leave a genuinely-closed position
# reading as open forever. See that function's own docstring for the full
# race this guards against.
_PROVISIONAL_EXIT_WINDOW_SECONDS = 300.0


async def _recent_trade_rows(ticker: str, limit: int = 3) -> list[asyncpg.Record]:
    """The last `limit` rows for `ticker` (current account_mode/user/
    backtest scope), newest first — shared by `get_open_stop_loss` and
    `get_open_stop_loss_entry_time` so both see the same race guard below.
    Callers are responsible for the fail-open try/except (this raises on a
    DB failure, same as any other query here)."""
    pool = await _ready_pool()
    async with pool.acquire() as conn:
        return await conn.fetch(
            "SELECT trade_date, stop_loss, shares_after, reason, mirror_disposition, created_at, id, price "
            "FROM trades WHERE ticker = $1 AND account_mode = $2 AND user_id = $3 AND is_backtest = $4 "
            "ORDER BY trade_date DESC, id DESC LIMIT $5",
            ticker,
            current_account_mode(),
            get_current_user_id(),
            is_backtest_context(),
            limit,
        )


def _is_provisional_forced_exit(row: asyncpg.Record, *, now: datetime) -> bool:
    """True for a forced-exit row (`STOP_LOSS_REASON_PREFIX`) whose KIS
    mirror outcome isn't confirmed yet (`mirror_disposition IS NULL` — the
    brief window between `record_trade` and the follow-up
    `_record_mirror_disposition`) or was BLOCKED/FAILED (about to be
    undone by `reverse_pending_trade`, which always follows immediately in
    the SAME coroutine) — and young enough that this is plausibly still
    that in-flight window rather than a permanently-unconfirmed old row.

    Exists because `force_exit_position` writes `stop_loss=exit_price` on
    this row BEFORE it knows whether the mirror will reach KIS
    (`_force_exit_locked`'s own comment explains why: the ledger record
    happens first so a mirror failure leaves a visible reconciliation gap
    instead of an unrecorded live order) — for the width of that window,
    this row genuinely says "the open stop is the exact price that just
    failed to sell". `_force_exit_locks` (a per-ticker asyncio.Lock)
    protects `force_exit_position` from itself, but NOTHING serializes
    that window against `rebalance_to_target` (a completely different,
    market-scoped `market_lock`) reading `get_open_stop_loss` for this same
    ticker in the meantime — see record-reversal-bug-evidence-strengthened
    project memory. Skipping this row is always the SAFE direction to be
    wrong in: worst case a since-reversed row gets skipped needlessly and
    the row below it (the reversal itself, or an older still-open position)
    is used instead — never a fabricated stop-loss VALUE, only a choice of
    which real row to trust.

    Never triggers in a backtest — a replay has no concurrent caller to
    race against, and `created_at` there is wall-clock (not `as_of`), so
    age-based skipping would just be noise."""
    if is_backtest_context():
        return False
    reason = row["reason"] or ""
    if not reason.startswith(STOP_LOSS_REASON_PREFIX):
        return False
    disposition = row["mirror_disposition"]
    if disposition is not None and not _mirror_never_reached_kis(disposition):
        return False  # confirmed submitted/skipped — this IS the real outcome, trust it
    age = (now - row["created_at"]).total_seconds()
    return age <= _PROVISIONAL_EXIT_WINDOW_SECONDS


def _select_effective_trade_row(ticker: str, rows: list[asyncpg.Record], *, now: datetime) -> asyncpg.Record | None:
    """The newest row in `rows` that isn't a provisional forced-exit (see
    above), or `None` if there is no such row or it shows the position
    flat/unset — the shared decision `get_open_stop_loss` and
    `get_open_stop_loss_entry_time` build their own return shape from."""
    for row in rows:
        if _is_provisional_forced_exit(row, now=now):
            logger.warning(
                "%s: skipping provisional forced-exit trade id=%s (mirror_disposition=%r, age=%.1fs) as the "
                "open stop — race guard, see _is_provisional_forced_exit",
                ticker, row["id"], row["mirror_disposition"], (now - row["created_at"]).total_seconds(),
            )
            continue
        if row["shares_after"] <= 0 or row["stop_loss"] is None:
            return None
        return row
    return None


async def get_open_stop_loss(ticker: str) -> tuple[date, float] | None:
    """The trade date and stop-loss of the most recent open (non-flat)
    position, for the live `check-stops` CLI to know what to check against
    and from which date forward. `None` if flat or unset.

    Scoped to the CURRENT account_mode (2026-08-04) — before this, a
    paper-era row showing a ticker as flat/closed would make this return
    None even after switching to real and actually holding that ticker for
    real, and check_stops_for_ticker short-circuits on None before ever
    checking the real KIS balance — silently leaving a real position with
    no automated stop-loss monitoring at all. See project memory:
    user-account-mode-db-separation-design.

    Skips a not-yet-confirmed forced-exit row via `_select_effective_trade_row`
    (2026-08-21) — see `_is_provisional_forced_exit`'s own docstring."""
    try:
        rows = await _recent_trade_rows(ticker)
    except Exception as exc:  # noqa: BLE001 - same broadened fail-open policy as get_position above
        logger.warning("execution get_open_stop_loss failed: %s", exc)
        return None

    row = _select_effective_trade_row(ticker, rows, now=get_clock().now(timezone.utc))
    if row is None:
        return None
    return row["trade_date"], row["stop_loss"]


async def get_open_stop_loss_entry_time(ticker: str) -> tuple[datetime, float, float] | None:
    """Like `get_open_stop_loss`, but returns the precise entry TIMESTAMP
    (`trades.created_at`) and the entry FILL PRICE alongside the stop-loss
    — for `app.live.exit_monitor`'s tick-based check (2026-08-10), which
    must trigger even ON the entry day itself but only against ticks
    strictly AFTER the moment of entry: a same-day intraday dip that
    happened BEFORE this position opened is not a breach of ITS stop.
    `check_and_apply_stop_loss`'s own daily-bar path stays on trade_date
    (`get_open_stop_loss`) since it already excludes the entry date's own
    daily low entirely (see `check_stops_for_ticker`) — a coarser,
    date-only exclusion that's correct at daily granularity but not
    precise enough for a tick-level check on the entry day itself.

    The entry price (2026-08-10, trailing stop) is this same row's `price`
    — the fill price `app.execution.trailing_stop` measures a position's
    run-up against and its activation threshold is a percentage of.

    Skips a not-yet-confirmed forced-exit row via `_select_effective_trade_row`
    (2026-08-21) — see `_is_provisional_forced_exit`'s own docstring."""
    try:
        rows = await _recent_trade_rows(ticker)
    except Exception as exc:  # noqa: BLE001 - same broadened fail-open policy as get_open_stop_loss
        logger.warning("execution get_open_stop_loss_entry_time failed: %s", exc)
        return None

    row = _select_effective_trade_row(ticker, rows, now=get_clock().now(timezone.utc))
    if row is None:
        return None
    return row["created_at"], row["price"], row["stop_loss"]


async def record_trade(
    ticker: str,
    trade_date: date,
    side: str,
    shares: int,
    price: float,
    reason: str,
    stop_loss: float | None = None,
    decision_id: int | None = None,
    adjustment_note: str | None = None,
    base_shares: int | None = None,
    transaction_cost_override: float | None = None,
    mirror_disposition: str | None = None,
    kis_order_no: str | None = None,
) -> int | None:
    """`base_shares`, if given, overrides `position.shares` as the starting
    point for `shares_after` — cash accounting still comes from the
    internal ledger (`position.cash`, unaffected). Needed when a caller
    (rebalance_to_target, live only) computed `shares` against the REAL
    KIS-held quantity rather than this internal ledger's own — without
    this, shares_after would be derived from a stale/wrong base and could
    go negative (see rebalance_to_target's own docstring on this).

    `transaction_cost_override`, if given, replaces the usual bps-of-
    notional fee below — used by `reverse_pending_trade` to pass a
    NEGATIVE value (refunding the original trade's fee) rather than
    charging a second one for a trade that never actually reached KIS. See
    that function's own docstring for why.

    `mirror_disposition`, if already known at call time (e.g. the manual
    dashboard order path, which only calls this after `place_manual_order`
    already succeeded), is stored directly. Most callers don't know it yet —
    rebalance_to_target/check_and_apply_stop_loss compute their disposition
    AFTER this call (mirror_trade_to_kis needs the ledger row to exist
    first) — those instead capture this function's returned id and call
    `_record_mirror_disposition` once the disposition is known. See that
    column's own schema comment for why app.execution.reconciliation needs
    it. Returns the inserted row's id, or None on a DB failure (same
    degrade-quietly contract as everything else here — callers that can't
    get an id simply skip the follow-up update).

    `kis_order_no`, same "known upfront on the manual path, unknown until
    after this call on every automated path" split as `mirror_disposition`
    — automated callers instead capture this function's returned id and
    call `_record_kis_order_no` once mirror_trade_to_kis returns an
    OrderResult. See app.live.fill_monitor and the column's own schema
    comment."""
    position = await get_position(ticker)
    starting_shares = base_shares if base_shares is not None else position.shares
    notional = shares * price
    # Asymmetric cost model: broker commission (`transaction_cost_bps`) on
    # both sides, plus Korea's sell-only Securities Transaction Tax
    # (`sell_tax_bps_domestic`, domestic `.KS` sells only) — see
    # `app.execution.costs.transaction_cost`'s own docstring and the
    # setting's comment for the 2026-01-01 rate. Every fill before the
    # symmetric-only version of this was frictionless at the exact close,
    # which overstates achievable real-world returns; the tax split above
    # closed that gap for domestic sells specifically. Shared with
    # `app.backtest.scalping_harness` (2026-08-21) so the live ledger and
    # the intraday backtest can never disagree on this formula — see that
    # module's own docstring. Deducted from cash_after, not `price` itself,
    # so the ledger's stored fill price keeps meaning "actual close" and
    # isn't confused with a synthetic slippage-adjusted price on
    # dashboard/audit displays.
    transaction_cost = (
        transaction_cost_override
        if transaction_cost_override is not None
        else costs.transaction_cost(ticker, side, notional)
    )
    if side == "buy":
        cash_after = position.cash - notional - transaction_cost
        shares_after = starting_shares + shares
    else:
        cash_after = position.cash + notional - transaction_cost
        shares_after = starting_shares - shares

    try:
        pool = await _ready_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO trades
                    (ticker, trade_date, side, shares, price, cash_after, shares_after, stop_loss, reason, decision_id, user_id, adjustment_note, account_mode, mirror_disposition, is_backtest, kis_order_no)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16)
                RETURNING id
                """,
                ticker,
                trade_date,
                side,
                shares,
                price,
                cash_after,
                shares_after,
                stop_loss,
                reason,
                decision_id,
                get_current_user_id(),
                adjustment_note,
                current_account_mode(),
                mirror_disposition,
                is_backtest_context(),
                kis_order_no,
            )
    except Exception as exc:  # noqa: BLE001 - same broadened fail-open policy as get_position above
        logger.warning("execution record_trade failed, continuing without it: %s", exc)
        return None
    return row["id"] if row is not None else None


async def _record_mirror_disposition(trade_id: int | None, disposition: str) -> None:
    """Best-effort follow-up write recording what happened to a just-recorded
    trade's KIS submission, once that's known (see `record_trade`'s own
    docstring for why this is a separate step for most callers). A no-op if
    `trade_id` is None (the original `record_trade` call already failed and
    logged a warning — nothing to attach this to)."""
    if trade_id is None:
        return
    try:
        pool = await _ready_pool()
        async with pool.acquire() as conn:
            await conn.execute("UPDATE trades SET mirror_disposition = $1 WHERE id = $2", disposition, trade_id)
    except Exception as exc:  # noqa: BLE001 - same broadened fail-open policy as record_trade above
        logger.warning("execution _record_mirror_disposition failed, continuing without it: %s", exc)


async def _record_kis_order_no(trade_id: int | None, order_no: str) -> None:
    """Best-effort follow-up write, same shape and reasoning as
    `_record_mirror_disposition` above — called from
    app.execution.kis_bridge.mirror_trade_to_kis right after KIS accepts an
    order and returns its order number, so app.live.fill_monitor has an
    exact key to later confirm the ACTUAL fill against
    (BrokerExecution.order_no from KIS's own execution-history TR). A no-op
    if `trade_id` is None or `order_no` is falsy (KIS didn't actually
    return one — nothing to record)."""
    if trade_id is None or not order_no:
        return
    try:
        pool = await _ready_pool()
        async with pool.acquire() as conn:
            await conn.execute("UPDATE trades SET kis_order_no = $1 WHERE id = $2", order_no, trade_id)
    except Exception as exc:  # noqa: BLE001 - same broadened fail-open policy as record_trade above
        logger.warning("execution _record_kis_order_no failed, continuing without it: %s", exc)


def _mirror_never_reached_kis(disposition: str) -> bool:
    """True for a `mirror_trade_to_kis` disposition meaning this specific
    trade will never be submitted — "BLOCKED: ..." (a permanent gate:
    allowlist, value cap, safety switch, ...) or "FAILED: ..." (credentials
    missing, an exception during submission). NOT true for "skipped: ..."
    (KIS_EXECUTION_ENABLED=false or KIS_EXECUTION_DRY_RUN=true — the ledger
    is deliberately the sole source of truth in those modes, nothing to
    reconcile against) or "PENDING_USER_APPROVAL: ...`/"submitted: ..."
    (still live or already real — reversed elsewhere, or not reversible)."""
    return disposition.startswith("BLOCKED:") or disposition.startswith("FAILED:")


_FAILURE_MARKER = re.compile(r"^FAILED: \[(?P<code>\S+) (?P<kind>permanent|transient)\]")


def failure_is_permanent(disposition: str) -> bool:
    """True if `disposition` means resubmitting the identical trade cannot
    ever succeed — a permanent gate (`BLOCKED:`, e.g. allowlist/value cap)
    or a mirror failure `kis_bridge.mirror_trade_to_kis` itself tagged
    permanent (`FAILED: [<msg_cd> permanent] ...`, see that module's
    exception handler and `app.brokers.kis.const.is_permanent_refusal`).
    A `FAILED:` disposition with no such marker (older format, or a path
    that doesn't set msg_cd) is conservatively treated as NOT permanent —
    i.e. still retryable — so an unrecognized shape can only ever cost an
    extra retry, never suppress a needed one. Added 2026-08-19 alongside
    force_exit_position's failure cooldown, after AXTI's stop-loss exit
    resubmitted the same doomed (IGW00013) order 17 times in 43 minutes."""
    if disposition.startswith("BLOCKED:"):
        return True
    match = _FAILURE_MARKER.match(disposition)
    return bool(match and match.group("kind") == "permanent")


async def reverse_pending_trade(
    ticker: str, side: str, shares: int, price: float, reason: str, prior_stop_loss: float | None = None
) -> int:
    """Returns the resulting share count after the reversal (equal to the
    pre-reversal position's shares — a no-op reversal, see below, returns
    that unchanged) — callers that report the trade's outcome (e.g.
    `rebalance_to_target`'s own `RebalanceResult`) need this instead of
    whatever pre-reversal target they were about to report, since the
    reversal means that target was never actually reached.

    Writes a compensating trade undoing a pending order's speculative
    ledger entry — `rebalance_to_target` records its trade BEFORE checking
    `execution_mode`, so a SEMI_AUTO/approval-gated order already has a
    `trades` row the moment it's proposed, even though it never actually
    reached KIS. If that order is later rejected or expires unresolved
    (`app.execution.pending_orders`), `get_position` must stop reflecting it
    as if it had happened — otherwise every later decision sizes itself off
    a position that increasingly diverges from the real account the longer
    unresolved orders pile up (same root cause as the ledger-drift incident
    `app.orchestrator.pipeline._position_context`'s real-balance preference
    already patches around, just for the AI's own sizing math this time).

    Appends a new row (this file's append-only convention — see
    `TradeEntry`), never edits/deletes the original: a dashboard should show
    "bought 10, then reversed 10", not silently rewritten history. No KIS
    call — a pending order by construction never reached the broker, so
    there is nothing on that side to undo; only `record_trade` runs.

    Flips `side` (undoing a buy is a sell of the same shares, and vice
    versa). The sell-direction case (undoing a phantom buy) is clamped to
    `min(shares, position.shares)` — this is a long-only book, and if some
    of those shares were legitimately sold in the interim, reversing the
    full original amount would push `shares_after` negative, corrupting
    every downstream reader that treats `shares_after <= 0` as "flat"
    (`get_open_stop_loss`, `check_and_apply_stop_loss`). The buy-direction
    case (undoing a phantom sell) has no symmetric oversell risk, so it
    always reverses the full amount. A no-op (nothing left to reverse)
    logs and returns rather than writing a zero-share row.

    `decision_id=None` on the reversal row: it isn't a new AI decision (a
    correction of a phantom one), so it stays unlinked. `reason` is prefixed
    `"reversal: "` so the dashboard's trade history reads as a correction,
    not a fresh trade, while keeping the caller's own resolution string
    (e.g. "rejected by user") for full audit context.

    `prior_stop_loss`, if given, is the stop-loss that was protecting the
    position BEFORE the now-reversed phantom trade — written onto this
    compensating row only if the position remains open (`shares_after > 0`)
    after undoing it; forced to `None` if the reversal leaves the position
    flat, since a flat position needs no stop regardless of what the caller
    passed. Without this, every reversal wrote `stop_loss=None`
    unconditionally, and `get_open_stop_loss` reads only the MOST RECENT
    row — so a still-open position whose speculative trade got reversed
    (BLOCKED/FAILED mirror) silently lost stop-loss monitoring forever,
    until some unrelated future trade happened to write a fresh stop.
    Confirmed live 2026-08-06: two held positions (055550.KS, AMZN) with an
    open position and `stop_loss IS NULL` on their latest row, both from a
    reversed trade. The caller is responsible for passing the correct
    prior value — `check_and_apply_stop_loss` passes the stop that just
    triggered (the reversal restores the exact pre-exit position, still
    protected by that same stop); `rebalance_to_target` passes whatever
    `get_open_stop_loss` returned immediately before recording the
    now-reversed trade."""
    position = await get_position(ticker)
    reversed_side = "sell" if side == "buy" else "buy"
    clamped_shares = min(shares, position.shares) if reversed_side == "sell" else shares
    if clamped_shares <= 0:
        logger.info(
            "reverse_pending_trade %s: nothing to reverse (position already %d)", ticker, position.shares
        )
        return position.shares
    resulting_shares = (
        position.shares - clamped_shares if reversed_side == "sell" else position.shares + clamped_shares
    )
    stop_loss_after = prior_stop_loss if resulting_shares > 0 else None
    # Surfaces (never corrects — see this function's own docstring on
    # trusting the caller's prior_stop_loss) the exact "stop_loss==fill
    # price" shape the record-reversal-bug-evidence-strengthened project
    # memory flagged as suspicious in live incidents (035420.KS/GOOG/AXTI,
    # 2026-08-20) — a `prior_stop_loss` that happens to equal THIS
    # reversal's own `price` would mean whatever caller supplied it (most
    # plausibly `rebalance_to_target`, racing `force_exit_position` for
    # this same ticker via a different lock — see
    # `_is_provisional_forced_exit`) captured a stop that was itself about
    # to be reversed, not a genuine prior value. Doesn't change
    # `stop_loss_after` — this codebase's convention is to degrade/report,
    # never fabricate, and the caller may simply be legitimately reversing
    # AT that exact price by coincidence.
    if stop_loss_after is not None and abs(stop_loss_after - price) <= max(0.01, price * 1e-6):
        logger.warning(
            "%s: reverse_pending_trade's prior_stop_loss (%.4f) matches this reversal's own price (%.4f) — "
            "possible stop-loss provenance race, see _is_provisional_forced_exit",
            ticker, stop_loss_after, price,
        )
        for fresh in filter_new_alerts(
            [Alert(
                ticker=ticker,
                kind="suspect_stop_provenance",
                message=(
                    f"{ticker}: a reversed trade's restored stop-loss ({stop_loss_after:,.2f}) matches the "
                    f"exact price it was reversed at ({price:,.2f}) — possible race with a concurrent "
                    f"force_exit_position/rebalance_to_target for this ticker. Reason: {reason}"
                ),
                triggered_at=get_clock().now(),
            )]
        ):
            await record_alert(fresh)
    # Refund the fee the original (now-undone) trade charged on these same
    # clamped_shares/price, and charge nothing for this reversal itself —
    # without this, record_trade's usual bps-of-notional fee applied HERE
    # TOO, so a trade that never reached KIS still cost two fees (one going
    # in, one coming back out) for a net position/cash change of zero
    # shares. Proportional to clamped_shares, not the original trade's full
    # `shares` — a PARTIAL reversal (some shares already legitimately sold
    # in the interim, see this function's own clamp above) should only
    # refund the fee on the portion actually being undone.
    #
    # Uses `side` (the ORIGINAL trade's side), not `reversed_side` — the
    # refund must match what was actually charged going in. A now-undone
    # domestic SELL was charged commission + the 20bp Securities Transaction
    # Tax, so the refund must include the tax too, even though the
    # compensating row below is itself a "buy". Getting this backwards would
    # silently under- or over-refund cash on every reversed domestic sell.
    # The reversal row still charges nothing regardless of its own
    # `reversed_side` — `transaction_cost_override` replaces record_trade's
    # normal cost computation entirely, so `costs.transaction_cost` never
    # runs a second time for this leg.
    refund = costs.transaction_cost(ticker, side, clamped_shares * price)
    await record_trade(
        ticker, get_clock().today(), reversed_side, clamped_shares, price,
        reason=f"{REVERSAL_REASON_PREFIX}{reason}", stop_loss=stop_loss_after, decision_id=None,
        transaction_cost_override=-refund,
    )
    return resulting_shares


def resolve_stop_loss(
    stop_loss: float | None,
    price: float,
    shares_after: int,
    current_atr_pct: float | None = None,
) -> float | None:
    """Validates/floors the Portfolio Manager's chosen stop-loss in code,
    the same "enforced in code, not trusted from prose" treatment
    `app.execution.sizing.target_allocation_for` already gives position size.

    - No open position after this trade (`shares_after <= 0`): no stop needed.
    - Missing, non-positive, at/above the fill price, or unreasonably far
      below it (more than `sizing_max_stop_distance_pct` of price — see
      that setting's own docstring for the incident this catches: an
      LLM-hallucinated price anchoring the whole decision, not a real
      risk-managed stop) — all treated as equally invalid on this
      long-only book. A default is derived from the ATR floor below
      instead of leaving the position with zero protection (or a
      nonsensical one) over a bad LLM value, unless no ATR data is
      available either, in which case it's left unset rather than guessed
      (same "degrade, don't fabricate" convention as the rest of this
      module).
    - Otherwise, floored at `sizing_min_stop_atr_multiple` x the current
      ATR% distance from price (2.5x by default — see the setting's
      docstring for the Turtle Trading / Chandelier Exit grounding), so a
      too-tight stop can't be whipsawed out by ordinary volatility. Never
      widened past that floor, and never narrowed by the ceiling above
      either — both are one-directional guards, not a target.
    """
    if shares_after <= 0:
        return None

    # 2026-09-01 (audit #6): extracted to app.domain.executability.min_stop_distance
    # so app.live.auto_trade_loop's pre-ranking affordability filter shares this
    # EXACT floor (including the 0.95-of-price clamp for extreme volatility)
    # instead of a second, driftable copy.
    min_distance = executability.min_stop_distance(
        price, current_atr_pct, risk_rules.rule("sizing_min_stop_atr_multiple")
    )

    max_distance = price * risk_rules.rule("sizing_max_stop_distance_pct")
    unreasonably_wide = stop_loss is not None and stop_loss > 0 and (price - stop_loss) > max_distance

    if stop_loss is None or stop_loss <= 0 or stop_loss >= price or unreasonably_wide:
        if min_distance is None:
            logger.warning(
                "stop-loss missing/invalid (%s) and no ATR data available — leaving unset", stop_loss
            )
            return None
        default_stop = price - min_distance
        if unreasonably_wide:
            logger.warning(
                "stop-loss %.4f is %.1f%% below price %.4f (max allowed %.1f%%) — likely anchored to a "
                "wrong/hallucinated price, not a real stop; discarded in favor of ATR default %.4f",
                stop_loss, (price - stop_loss) / price * 100, price,
                risk_rules.rule("sizing_max_stop_distance_pct") * 100, default_stop,
            )
        else:
            logger.warning(
                "stop-loss missing/invalid (%s) — derived default %.4f from ATR floor", stop_loss, default_stop
            )
        return default_stop

    if min_distance is not None and stop_loss > price - min_distance:
        floored = price - min_distance
        logger.info("stop-loss %.4f tighter than ATR floor — widened to %.4f", stop_loss, floored)
        return floored

    return stop_loss


class EquityBasisUnavailable(Exception):
    """Raised by `resolve_equity_basis` when the account is REAL, the call is
    live (`as_of is None`), and the real broker equity can't be read. There is
    deliberately no fallback in that case: `get_position` returns
    `Position(cash=await starting_cash_for(ticker), shares=0)` for an empty
    ledger row OR any DB error, and `starting_cash_for` is
    `settings.paper_starting_cash * fx_rate` — with the 100,000 USD default
    that is ~137,000,000 KRW of FICTIONAL equity, which then feeds the 1%
    position-sizing rule, the stop-distance math, and the risk-cap narrative
    shown to five LLM nodes against a real account worth a few hundred
    thousand won. `insufficient_cash_block` and `over_position_pct` both fail
    OPEN on the same broker error, so nothing downstream catches it either.
    Paper mode and backtest (`as_of is not None`) keep the internal-ledger
    fallback — there is no real account to contradict there.

    `ticker`, `market` and `failure` (`"broker_unavailable"`/`"broker_error"`/
    `None`) are attached so a caller can build an alert without re-parsing the
    message.
    실계좌 + 라이브에서 실제 잔고 조회가 실패하면 내부 장부의 가상 시드머니로
    대체하지 않고 예외를 던져 fail-closed 처리 — 주문 자체를 포기시킴."""

    def __init__(self, message: str, *, ticker: str, market: str, failure: str | None) -> None:
        super().__init__(message)
        self.ticker = ticker
        self.market = market
        self.failure = failure


class EquityBasis(NamedTuple):
    equity: float
    source: Literal["real_account", "internal_ledger"]


async def resolve_equity_basis(
    ticker: str,
    price: float,
    *,
    as_of: date | None,
    position: Position | None = None,
) -> EquityBasis:
    """The ONE place "what equity backs this ticker's sizing" is decided —
    both the Portfolio Manager's pre-decision narrative
    (`app.orchestrator.pipeline._risk_cap_context`) and the actual order
    sizing below (`rebalance_to_target`) call this instead of each computing
    equity their own way, closing the gap that let an Underweight/Sell
    rating's target_fraction be set against one equity figure while the real
    order sized against another several times larger (memory:
    live-rebalance-equity-basis-mismatch-2026-08-14 — GOOG, 2026-08-13:
    internal ~$69K vs real ~$248K, a 3.6x gap that flipped a 30-share SELL
    into a 36-share BUY; the direction guard in `rebalance_to_target` below
    stops the worst outcome of that gap, this closes the gap itself).

    Backtest (`as_of` set): always the internal ledger's own
    `position.market_value(price)` — there is no real account for a
    historical replay, exactly as every other real_*-preferring path in this
    module already treats `as_of`.

    Live (`as_of is None`): prefers `real_equity_native(ticker_market(ticker))`
    (the REAL aggregate KIS broker account equity — see that function's own
    docstring for the ~₩147M-vs-$20K incident it exists to prevent) over the
    internal ledger's `position.market_value(price)`. `source` tells a caller
    which one actually backed the number, e.g. so the narrative can flag it
    when it fell back.

    What happens when the broker call returns `None` depends on the account
    mode, and this is the one place that distinction is made:

      * paper (`current_account_mode() != "real"`): falls back to the internal
        ledger, same convention as everywhere else `real_equity_native`/
        `real_shares_held` are used. Nothing real is at stake.
      * real: raises `EquityBasisUnavailable` — NO fallback. The internal
        ledger's equity for an empty/erroring row is
        `settings.paper_starting_cash * fx_rate` (see `starting_cash_for`),
        i.e. a fictional ~₩137M against a real account worth a fraction of
        that, and it would flow straight into the 1% risk rule, the stop
        distance and the risk-cap narrative five LLM nodes read. See
        `EquityBasisUnavailable`'s own docstring for why nothing downstream
        catches it (`insufficient_cash_block`/`over_position_pct` both fail
        OPEN on the same broker error).

    The account-mode check is deliberately AFTER the `as_of` check: a
    historical replay has no real account to contradict, so backtest keeps the
    internal ledger unconditionally, in real mode too.

    `real_equity_native_detailed` is re-called (not used as the primary call)
    only on that refuse path, purely to name the failure in the exception. It
    is not the primary call because every existing caller and test binds to the
    plain-float `real_equity_native`. The duplicate costs nothing when the
    broker simply isn't connected (that branch returns before any fetch) and at
    most one repeated failing balance TR otherwise — paid only on a path that
    is already abandoning the trade.

    `position`, if already fetched by the caller, is reused instead of a
    second `get_position` DB round-trip for the same ticker — `rebalance_to_target`
    always has one already; `_risk_cap_context` doesn't, so it fetches fresh.

    Raises:
        EquityBasisUnavailable: real account mode, live, and the real broker
            equity could not be read."""
    if position is None:
        position = await get_position(ticker)
    if as_of is not None:
        return EquityBasis(position.market_value(price), "internal_ledger")
    market = ticker_market(ticker)
    real_equity = await real_equity_native(market)
    if real_equity is not None:
        return EquityBasis(real_equity, "real_account")
    if current_account_mode() == "real":
        try:
            failure = (await real_equity_native_detailed(market)).failure
        except Exception:
            # real_equity_native_detailed's own try/except covers the KIS
            # calls but NOT the `state.kis_broker()` resolution above them.
            # This second call exists only to name the failure — it must never
            # replace EquityBasisUnavailable with some unrelated traceback.
            # 실패 사유 문자열을 얻기 위한 보조 호출 — 여기서 난 예외로 본 예외를 덮지 않음.
            failure = None
        message = (
            f"{ticker}: real-account mode and the real broker equity could not be read "
            f"({failure or 'unknown'}) — refusing to size against the internal ledger, whose "
            f"balance is a fictional paper_starting_cash default whenever the real balance is "
            f"missing. No equity basis for this cycle."
        )
        logger.error(message)
        raise EquityBasisUnavailable(message, ticker=ticker, market=market, failure=failure)
    return EquityBasis(position.market_value(price), "internal_ledger")


# Stable, greppable marker for "the AI's rating asked for a change in exposure
# and the position did not change" — the audit CLI
# (app.audit.stats.compute_rating_action_gap) groups on it and the dashboard
# signal card (_executionNotesFrom in signals.js) surfaces the sentence after
# it. 2026-09-01, first real-account KRX cycle: 6 of 41 decisions were rated
# Buy with action_label Hold (target sized below one share on a 200,000 KRW
# account) and 5 of them left no explanation anywhere — only a logger.info.
# NEVER put a ']' in an adjustment_note: the audit SQL and signals.js both
# extract the persisted '[EXECUTION NOTE: ...]' with a ']'-terminated match.
# The wording deliberately avoids every app.backtest.portfolio_harness._CAP_LABELS
# substring so the backtest cap tally stays accurate.
RATING_ACTION_GAP_NOTE_PREFIX = "No position change despite a directional rating"

_INCREASE_INTENT_RATINGS = (PortfolioRating.BUY, PortfolioRating.OVERWEIGHT)
_DECREASE_INTENT_RATINGS = (PortfolioRating.UNDERWEIGHT, PortfolioRating.SELL)


def _log_ledger_inquiries_on_exit(ticker: str) -> None:
    """FIX-6 (2026-09-03 id=1187 audit): logs how many times the broker's
    real ledger/balance TR was actually called (cache hits excluded — see
    `app.execution.portfolio_risk.ledger_inquiry_count`) during one
    `market_lock` hold of `rebalance_to_target`, for verifying FIX-1 closed
    the duplicate-fetch gap (should read 1 for a domestic hold, 2 for
    overseas — balance + present-balance). Pushed via `AsyncExitStack.
    callback` as the LAST context manager entered in `rebalance_to_target`'s
    `_lock_stack` so it runs FIRST on unwind — i.e. before
    `cached_balance_fetches`' own `finally` resets the counter back to 0.

    The "2 for overseas" expectation only became TRUE as of the 2026-09-08
    follow-up fix in `app.execution.portfolio_risk.real_equity_native_
    detailed` — before that, its overseas branch called
    `broker.fetch_overseas_present_balance` directly instead of the cached
    helper `real_account_snapshot` a few lines away already used, so one
    overseas hold actually made 3 real present-balance-shaped wire calls
    (balance + present-balance via real_equity_native_detailed + present-
    balance again via real_account_snapshot) while this counter, which
    never saw the un-cached call, still read 2 — the exact EGW00215-id=1187
    duplicate-fetch shape FIX-1 was meant to close, just on the overseas
    leg, and invisible to this very instrumentation because the missed call
    site never bumped `_bump_ledger_inquiry_count()`. Do not trust this
    counter alone to rule out a duplicate real fetch — a new call site added
    without routing through one of the `_cached_fetch_*` helpers in
    `portfolio_risk.py` will make the same class of mistake silently."""
    logger.info("%s: ledger inquiries this market_lock hold: %d", ticker, ledger_inquiry_count())


async def rebalance_to_target(
    ticker: str,
    trade_date: date,
    price: float,
    target_fraction: float | None,
    reason: str,
    stop_loss: float | None = None,
    current_atr_pct: float | None = None,
    portfolio_cap_override: tuple[PortfolioSnapshot, dict[str, int], float] | None = None,
    decision_id: int | None = None,
    as_of: date | None = None,
    rating: PortfolioRating | None = None,
) -> RebalanceResult:
    """`portfolio_cap_override`, if given, is (snapshot, correlation_clusters,
    drawdown_fraction) precomputed by a caller that already has this
    information for the whole date/trade allowlist (the multi-ticker backtest
    harness) — skips the live fetch inside `apply_live_portfolio_caps` and
    calls `apply_backtest_portfolio_caps` with it instead, which resolves
    this ticker's FX rate (native `price` currency vs. this USD-normalized
    `snapshot`) before delegating to the pure `apply_portfolio_caps`. `None`
    (the default, and every live caller's path) fetches live.

    `decision_id`, if given, is recorded on the resulting trade row so the
    dashboard can link a decision to the exact trade it produced instead of
    guessing by (ticker, trade_date) — which collides when the same ticker
    trades more than once on the same day (e.g. a same-day stop-loss exit
    on top of this rebalance).

    `as_of` (LIVE ONLY, i.e. `as_of is None`): the direction/sizing math
    below (`current_shares`, everywhere it stands in for the older
    `position.shares`) prefers the REAL KIS-held quantity over this internal
    ledger's own `position.shares` — same correction `_position_context`
    already applies to the narrative (app.orchestrator.pipeline), now
    applied to the actual trade decision. Real incident this fixes: a SELL
    decision on a ticker this ledger had never recorded owning (bought
    outside this system, or from before the ledger tracked it) computed
    target_shares=0 against an internal shares=0 too, so delta was 0 and
    the sell silently never happened — even though the real account held
    shares and the AI correctly said to exit them.

    Equity (used for the 1% risk rule and ATR-scaled BUY sizing) is resolved
    by `resolve_equity_basis` — the REAL aggregate account equity
    (`real_equity_native`) when live and available, the internal ledger's own
    cash+shares otherwise (always, during backtest — no "real account"
    exists for a historical replay). Real incident this fixes: the internal
    ledger's per-ticker cash had drifted, after enough simulated trades, to
    ~₩147M for one ticker alone while the REAL aggregate domestic account
    was worth ~$20K total — every target was wildly oversized on that
    fictional basis and then slammed back down to near the existing position
    by the (already real-balance-based) portfolio concentration cap, a
    self-defeating combination that blocked nearly every BUY. See memory:
    kis-integration-status.

    `app.orchestrator.pipeline._risk_cap_context` (the narrative shown to the
    LLM before it picks target_fraction, for the same ticker/cycle) now calls
    the same `resolve_equity_basis` — see that function's own docstring for
    why the two call sites used to disagree and the real incident
    (live-rebalance-equity-basis-mismatch-2026-08-14, GOOG 2026-08-13) that
    exposed it.

    `rating`, if given (always given by the live pipeline; only omitted by
    older/test callers), is a direction-guard backstop for whatever gap
    remains even with both call sites unified — the two calls still happen
    ~9 LLM calls apart, and `real_equity_native` can independently succeed at
    one and fail at the other in that window. When that residual gap flips
    an Underweight/Sell rating's intended decrease into a computed increase,
    this clamps target_shares down to current_shares rather than execute
    the increase."""
    if target_fraction is None:
        # Hold - no rebalancing trade, no target at all to compare against.
        #
        # NOT only Hold, despite the line above: app.execution.sizing.
        # target_allocation_for also returns None for a Buy/Overweight/
        # Underweight whose stated confidence lands in the lowest band —
        # including an OMITTED confidence, scored there on purpose. That is a
        # directional rating that comes out as action_label "Hold" with no
        # explanation anywhere, the same undiagnosable shape as the delta == 0
        # branch below (2026-09-01, audit #10). Leave a note when the rating
        # was directional. `current_shares` is not known here (this returns
        # before the position read), so the note can't state it.
        skip_note = None
        if rating is not None and rating is not PortfolioRating.HOLD:
            skip_note = (
                f"{RATING_ACTION_GAP_NOTE_PREFIX}: the {rating.value} rating produced no target "
                f"allocation at all — app.execution.sizing.target_allocation_for skips the trade when "
                f"the model's stated confidence falls in the lowest band (an omitted confidence is "
                f"scored there too). No order was sized and the position is unchanged."
            )
            logger.info("%s: %s", ticker, skip_note)
        return RebalanceResult(
            side=None, shares=0, shares_after=0, stop_loss=None,
            action_label="Hold", adjustment_note=skip_note,
        )

    # `lock_cm` serializes EVERYTHING below, from the very first real-balance
    # read through the actual KIS mirror, ONE MARKET+ACCOUNT_MODE AT A TIME —
    # live only (`portfolio_cap_override is None`, which today only happens
    # on the live path; backtests always pass a precomputed override, see
    # this function's own docstring). A backtest never touches `market_lock`
    # at all — `nullcontext()` is a plain no-op for that path, same as
    # before.
    #
    # Widened (2026-08-06) to cover current_shares/equity and the 1%-risk-
    # rule sizing above the portfolio-cap block, not just that block —
    # those used to run BEFORE the lock, so two processes (the exact
    # scenario this lock exists for) could each read the same pre-trade
    # snapshot, and even though the portfolio-cap step inside the lock DID
    # see a fresh balance, `delta = target_shares - current_shares` was
    # still computed against the stale pre-lock current_shares: process A
    # buys 10 first, releases the lock; process B's fresh in-lock snapshot
    # now reflects A's 10, but B's own `current_shares` was captured before
    # any of this ran, so B still computes delta against 0 and orders its
    # own full target on top of A's — a combined position roughly double
    # either one's intended size. Widening the lock to cover the read
    # closes that gap at its source instead of only at the portfolio-cap
    # step. 라이브에서만 시장별 락으로 잔고조회~KIS 반영까지 전체를 직렬화 — 동시분석 시
    # 두 종목/두 프로세스가 서로의 매매 반영 전 잔고를 보고 각자 목표치를 계산하는 레이스 방지.
    lock_cm = market_lock(ticker_market(ticker)) if portfolio_cap_override is None else nullcontext()
    # `cached_account_snapshots` is opened INSIDE the lock, never around it —
    # that ordering is the whole basis of its correctness (see its own
    # comment in portfolio_risk). It dedupes the two `real_account_snapshot`
    # calls this one hold makes: `apply_live_portfolio_caps` below, then
    # `mirror_trade_to_kis` -> safety_switch.daily_loss_block moments later.
    # Nothing changes the KIS balance in between — `record_trade` writes the
    # internal ledger only, and the real `submit_order` happens after both —
    # so the second call was re-asking KIS a question it had just answered,
    # at ~1s (domestic) / ~2s (overseas) of extra lock hold on the paper
    # account's 1-req/sec throttle. Backtests get a no-op scope: they pass
    # portfolio_cap_override and never reach real_account_snapshot at all.
    # 락 안에서만 여는 스냅샷 캐시 — 같은 락 구간에서 두 번 조회하던 실계좌 잔고를 1회로.
    #
    # `cached_balance_fetches` now spans the WHOLE lock hold too (was a narrow
    # inner scope around sizing only). Same correctness basis as
    # `cached_account_snapshots`: opened AFTER the lock is held and closed
    # before release, so no other trade in this market+account_mode can change
    # the KIS balance during the hold (`record_trade` writes the internal
    # ledger only; `submit_order` comes after every reader). This removes one
    # domestic balance inquiry — `mirror_trade_to_kis` -> _portfolio_cap_block
    # (kis_bridge.py) calls `real_shares_held` a second time, moments after
    # sizing just asked KIS the same question, and it used to fall outside the
    # old inner scope. It still does NOT cover `apply_live_portfolio_caps`,
    # which reads via `real_account_snapshot`/`_snapshot_cache`, not
    # `_balance_fetch_cache` — the two caches stay separate. The sell-ceiling
    # gate is a deliberate exception (kis_bridge._held_domestic_quantity keeps
    # a fresh fetch — see its comment).
    # 잔고 캐시 스코프를 락 구간 전체로 확대 — 근거는 스냅샷 캐시와 동일(락 획득 후 개시).
    #
    # `enforcement_safe=portfolio_cap_override is None` (FIX-1, 2026-09-03
    # id=1187): only when this hold is a REAL `market_lock` (live, not a
    # backtest's `nullcontext`) is it safe for `real_account_snapshot` inside
    # `apply_live_portfolio_caps` to reuse a balance `real_shares_held` above
    # already fetched into this same scope — see
    # `app.execution.portfolio_risk._balance_fetch_enforcement_safe`'s own
    # comment. Built with `AsyncExitStack` rather than a plain `async with
    # a, b, c:` tuple SOLELY so `_log_ledger_inquiries_on_exit` below can be
    # pushed as the innermost (last-entered, first-exited) context manager —
    # it must read `ledger_inquiry_count()` before `cached_balance_fetches`'
    # own `finally` resets the counter, which a plain `async with` tuple
    # cannot sequence (its `__aexit__`s unwind in the same LIFO order, but
    # there is nowhere to hook in between them without one). The body below
    # is otherwise UNCHANGED — same indentation, same statements — this is
    # purely a different way to enter the same three scopes.
    _lock_stack = contextlib.AsyncExitStack()
    async with _lock_stack:
        await _lock_stack.enter_async_context(lock_cm)
        await _lock_stack.enter_async_context(cached_account_snapshots())
        await _lock_stack.enter_async_context(
            cached_balance_fetches(enforcement_safe=portfolio_cap_override is None)
        )
        _lock_stack.callback(_log_ledger_inquiries_on_exit, ticker)
        position = await get_position(ticker)
        current_shares = position.shares
        if as_of is None:
            # EquityBasisUnavailable = real account mode, live, and the real
            # broker equity could not be read. resolve_equity_basis refuses to
            # fall back to the internal ledger there (its ~₩137M
            # paper_starting_cash default against a real account worth a
            # fraction of that — see its own docstring), so there is simply no
            # equity basis for this cycle. Caught here rather than left to
            # propagate: everything below this point (the 1% risk rule, the
            # ATR-scaled target, the portfolio caps, record_trade,
            # mirror_trade_to_kis) is sizing math that needs an equity figure,
            # and the caller (app.orchestrator.pipeline, ~line 1455) has no
            # handler for this — an escape would abort the whole pipeline run
            # after the decision row was already stored, leaving action_label
            # permanently NULL and no execution note, the exact undiagnosable
            # shape of the id=377/id=949 incidents. A clean "no trade" result
            # keeps the decision row honest instead. Only this live branch is
            # wrapped: the `else` below is backtest (`as_of is not None`), and
            # resolve_equity_basis returns before its account-mode check there,
            # so it provably cannot raise this.
            # 실계좌에서 실제 잔고를 못 읽으면 가상 잔고로 사이징하지 않고 이번 사이클
            # 매매를 건너뜀 — 예외를 파이프라인까지 올리지 않고 Hold 결과로 반환.
            try:
                real_shares = await real_shares_held(ticker)
                if real_shares is not None:
                    current_shares = real_shares
                equity, equity_source = await resolve_equity_basis(
                    ticker, price, as_of=as_of, position=position
                )
            except EquityBasisUnavailable as exc:
                # current_shares is whatever was resolved before the raise —
                # the REAL held quantity when real_shares_held answered (it
                # runs three lines above and never raises this), the internal
                # ledger's own otherwise. Either way it's the honest
                # "position unchanged" figure for shares_after.
                skip_note = (
                    f"No trade this cycle: real-account mode and the real broker balance was "
                    f"unavailable ({exc.failure or 'unknown'}) — sizing was skipped rather than "
                    f"computed against the internal ledger's fictional paper balance. Position "
                    f"left unchanged at {current_shares} shares."
                )
                # warning, not error: resolve_equity_basis already logged this
                # exact condition at ERROR once, right before raising.
                logger.warning("%s: %s", ticker, skip_note)
                # Same reasoning as the equity_basis_fallback alert below — a
                # skipped cycle produces NO trade row, so adjustment_note alone
                # would leave no trace anywhere that this ticker went untraded
                # because of a broker outage. Distinct `kind`, so its
                # per-(account_mode, ticker, kind) cooldown is independent of
                # the fallback alert's and neither can suppress the other; the
                # cooldown still caps an extended outage to one alert per
                # ticker per settings.alert_cooldown_seconds instead of one
                # per cycle.
                for alert in filter_new_alerts(
                    [
                        Alert(
                            ticker=exc.ticker,
                            kind="equity_basis_unavailable",
                            message=(
                                f"{exc.ticker} trade SKIPPED this cycle — real-account mode and the "
                                f"real broker equity could not be read ({exc.failure or 'unknown'}). "
                                f"No order was sized; the position is unchanged at {current_shares} "
                                f"shares. Check the KIS broker connection."
                            ),
                            triggered_at=get_clock().now(),
                        )
                    ]
                ):
                    await record_alert(alert)
                # stop_loss=None deliberately: the delta==0 path below reports
                # `resolved_stop_loss`, which is derived from the target share
                # count this branch never computed. Reporting an ATR-derived
                # stop for a cycle we just declared un-sizable would be
                # fabricating one.
                return RebalanceResult(
                    side=None,
                    shares=0,
                    shares_after=current_shares,
                    stop_loss=None,
                    action_label="Hold",  # == rebalance_action_label(current_shares, current_shares)
                    adjustment_note=skip_note,
                )
        else:
            equity, equity_source = await resolve_equity_basis(ticker, price, as_of=as_of, position=position)
        # 2026-09-01 (audit #6): shares this kernel with app.domain.executability's
        # pre-ranking affordability filter — see that module's docstring.
        target_shares = executability.allocation_target_shares(equity, target_fraction, price)
        # The PRE-CAP allocation floor, retained because five separate blocks
        # below mutate `target_shares` in place (direction guard, no-stop
        # block, 1% risk cap, portfolio caps, per-order value cap) and the
        # delta == 0 gap note must not attribute their reductions to the
        # allocation arithmetic. 2026-09-03 audit #4, decision id=1187
        # (088350.KS): the note claimed a 673,745 allocation was "less than
        # the 6,500.00 price of a single share, so floor(...) = 0" — the
        # floor was 103; the caps took it to 0. Every mutator appends its own
        # note, so `allocation_floor_shares != target_shares` is an exact,
        # note-independent test for "something above already explained this".
        # 캡 이전 배분 floor 값 보존 — 캡이 줄인 걸 배분 산술 탓으로 돌리지 않게.
        allocation_floor_shares = target_shares
        adjustment_note = None

        # Equity-fallback visibility (2026-08-14) — before this, a live
        # broker balance failure inside resolve_equity_basis was completely
        # silent from here: only a logger.warning deep in
        # portfolio_risk.real_equity_native, nothing on the trade row, no
        # alert. A trade whose adjustment_note stays empty (or a Hold that
        # never trades at all) left no trace that this cycle sized off the
        # internal ledger instead of the real account — exactly the
        # condition that let the GOOG incident's mismatch go unnoticed
        # until an after-the-fact audit (memory:
        # live-rebalance-equity-basis-mismatch-2026-08-14). Live only
        # (`as_of is None`; backtest always uses the internal ledger by
        # design, not as a fallback, so it's not news there). The alert
        # (not just the note) exists because adjustment_note only survives
        # on an actual trade row — a Hold decision or a target that happens
        # to match current_shares produces neither a trade nor a note, and
        # would otherwise leave zero record that the fallback happened.
        # filter_new_alerts caps this to one per (ticker, "equity_basis_fallback")
        # per settings.alert_cooldown_seconds — an extended real broker
        # outage would otherwise re-fire this on every ticker, every cycle,
        # burying the stop-loss/price/volume alerts this same feed carries.
        if as_of is None and equity_source == "internal_ledger":
            adjustment_note = (
                f"Equity basis fell back to the internal ledger ({equity:,.0f}) for this cycle — the "
                f"real broker balance was unavailable; sizing may not reflect the real account."
            )
            logger.warning("%s: %s", ticker, adjustment_note)
            for alert in filter_new_alerts(
                [
                    Alert(
                        ticker=ticker,
                        kind="equity_basis_fallback",
                        message=(
                            f"{ticker} sized off the internal ledger ({equity:,.0f}) instead of the real "
                            f"broker account this cycle — balance inquiry unavailable. Target sizing may not "
                            f"reflect real account equity."
                        ),
                        triggered_at=get_clock().now(),
                    )
                ]
            ):
                await record_alert(alert)

        # Direction guard (2026-08-14) — the AI's Underweight/Sell rating
        # already means "reduce exposure"; if the *sizing* math nonetheless
        # comes out above current_shares (an increase), that's not a
        # legitimate edge case. Originally this caught `equity` disagreeing
        # with itself between the risk-cap narrative
        # (app.orchestrator.pipeline._risk_cap_context, shown to the LLM
        # before it picks target_fraction) and this actual sizing call, since
        # the narrative always used the internal ledger's market_value while
        # this used real_equity_native when available — now both call the
        # same resolve_equity_basis, closing that specific gap. Kept as a
        # backstop regardless: the two calls still happen ~9 LLM calls apart
        # (real fills can land in between) and real_equity_native can still
        # independently succeed at one call and fail at the other. Clamp to
        # current_shares — never bounce sub-, never let a decrease flip into
        # an increase — rather than trust a target that already contradicts
        # the rating that produced it. Real incident: GOOG 2026-08-13 (memory:
        # live-rebalance-equity-basis-mismatch-2026-08-14) — an Underweight
        # call meant to cut 73 shares to 43 instead sized to 109 (real-
        # account equity ~3.6x the internal ledger's for that ticker),
        # i.e. a 36-share BUY where the rating said SELL ~30. KIS happened to
        # reject that specific order; nothing here should depend on that
        # happening again. Only Buy/Overweight/Hold are meant to ever
        # increase a position, so this only fires for Underweight/Sell.
        if rating in (PortfolioRating.UNDERWEIGHT, PortfolioRating.SELL) and target_shares > current_shares:
            direction_note = (
                f"Direction guard: {rating.value} rating implies reducing exposure, but sizing computed "
                f"target {target_shares} shares above the current {current_shares} — clamped to "
                f"{current_shares} instead of letting a decrease flip into an increase."
            )
            logger.warning("%s: %s", ticker, direction_note)
            target_shares = current_shares
            # Appended, not overwritten — this can fire in the same cycle as
            # the equity-fallback note just above (that combination is
            # exactly the GOOG incident's shape: real equity unavailable ->
            # internal-ledger fallback -> the resulting target computes as
            # an increase against an Underweight/Sell rating), and losing
            # the fallback note here would hide the actual root cause behind
            # the symptom this guard caught.
            adjustment_note = f"{adjustment_note} {direction_note}" if adjustment_note else direction_note

        # 2026-09-02 fail-closed gate (decision id=1169, 950260.KS, first
        # corporate full_auto KRX cycle). NEW ENTRIES ONLY (current_shares<=0):
        # without a usable 14-day ATR, app.execution.sizing.
        # vol_adjusted_allocation_for keeps the rating's full unscaled
        # allocation AND resolve_stop_loss just below has no floor to enforce
        # or validate the LLM's stop against — the 1% risk cap that follows
        # derives risk_per_share from that same unvalidated stop, so it isn't
        # a backstop either. Checked here, after target_shares/current_shares
        # are final (the real-shares-held correction above already ran) and
        # before any of that sizing/stop machinery executes. See
        # app.domain.executability.blocks_new_entry_without_atr's own
        # docstring for why a HELD position is deliberately exempt.
        if (
            settings.auto_trade_require_atr_for_new_entry
            and target_shares > 0
            and executability.blocks_new_entry_without_atr(current_atr_pct, current_shares=current_shares)
        ):
            skip_note = (
                f"No new position opened this cycle: 14-day ATR is unavailable for this ticker "
                f"(value: {current_atr_pct!r}), so neither the volatility scale-down "
                f"(app.execution.sizing.vol_adjusted_allocation_for) nor the ATR stop-loss floor "
                f"(app.execution.ledger.resolve_stop_loss) could be applied — a new entry would "
                f"have been sized at the unscaled base allocation with an unvalidated stop. "
                f"Refused the {target_shares}-share entry; the position stays flat at "
                f"{current_shares} shares. Reason for the missing ATR is unknown here: too little "
                f"price history (a recent listing), a data-provider failure, or a degenerate "
                f"close all look identical at this point. Set "
                f"auto_trade_require_atr_for_new_entry=False to restore the previous behavior."
            )
            logger.warning("%s: %s", ticker, skip_note)
            for alert in filter_new_alerts([Alert(
                ticker=ticker, kind="atr_unavailable_new_entry",
                message=(
                    f"{ticker} NEW ENTRY REFUSED this cycle — no usable 14-day ATR (value: "
                    f"{current_atr_pct!r}). Volatility-scaled sizing and the ATR stop-loss floor "
                    f"both disable when ATR is missing, so the entry would have been full-size "
                    f"with an unvalidated stop. No order was sized; the position is unchanged at "
                    f"{current_shares} shares. WHY the ATR is missing is NOT known from here — a "
                    f"recent listing with under 15 candles, a KIS/yfinance outage, and a zero "
                    f"close are indistinguishable at this point. Check whether this ticker is "
                    f"newly listed or the price feed is degraded before treating this as routine."
                ),
                triggered_at=get_clock().now(),
            )]):
                await record_alert(alert)
            return RebalanceResult(
                side=None,
                shares=0,
                shares_after=current_shares,
                stop_loss=None,
                action_label="Hold",
                adjustment_note=skip_note,
            )

        resolved_stop_loss = resolve_stop_loss(stop_loss, price, target_shares, current_atr_pct)

        # The "1% rule" (risk_rules.rule("sizing_max_risk_pct_of_equity")): bound the
        # dollar risk this trade ADDS, not the whole resulting position's risk —
        # a wider stop on a later decision must never force-liquidate shares
        # already held under an earlier one. Only applies when increasing
        # exposure; reducing/flattening a position never needs capping.
        #
        # 2026-09-10: this cap structurally dominates app.execution.sizing.
        # target_allocation_for's rating/confidence-scaled result for any
        # ATR >= sizing_target_atr_pct (2%) — exactly 5x tighter in the
        # smooth-scaling 2-8% ATR band, wider beyond it. See that module's
        # own extended docstring for the precise arithmetic and the
        # 2026-09-10 live-account measurement (2.6x-6.0x observed across 8
        # real trades). Intentional, not a bug — documented here so a
        # future reader doesn't "fix" what is a risk-cap-dominated design.
        if target_shares > current_shares:
            if resolved_stop_loss is None:
                # No resolvable stop distance (missing LLM value and no ATR data
                # to derive a default from) — there's nothing to bound risk
                # against, so refuse to add exposure rather than let an
                # unbounded, unprotected position through. Holding/reducing
                # still needs no stop, so this only blocks increases.
                no_stop_note = (
                    f"No resolvable stop-loss (missing value and no ATR data) — blocked the exposure "
                    f"increase to {target_shares} shares; held at the existing {current_shares} instead."
                )
                logger.warning("%s: %s", ticker, no_stop_note)
                # Appended (2026-08-14), not overwritten — preserves an
                # equity-fallback/direction-guard note from above rather
                # than silently dropping it whenever this cap ALSO applies
                # the same cycle.
                adjustment_note = f"{adjustment_note} {no_stop_note}" if adjustment_note else no_stop_note
                target_shares = max(current_shares, 0)
            else:
                risk_per_share = price - resolved_stop_loss
                # 2026-09-01 (audit #6): shares this kernel with
                # app.domain.executability's pre-ranking affordability filter.
                # None (risk_per_share <= 0) means "uncapped by this rule" —
                # same meaning as the old `if risk_per_share > 0:` guard this
                # replaces.
                max_incremental_shares = executability.risk_capped_shares(
                    equity, risk_per_share, risk_rules.rule("sizing_max_risk_pct_of_equity")
                )
                if max_incremental_shares is not None:
                    incremental_shares = target_shares - current_shares
                    if max_incremental_shares < incremental_shares:
                        capped_target = current_shares + max_incremental_shares
                        risk_cap_note = (
                            f"1% risk cap reduced the intended add from {incremental_shares} to "
                            f"{max_incremental_shares} shares (target {target_shares} -> {capped_target}) to keep "
                            f"this trade's stop-loss risk within {risk_rules.rule('sizing_max_risk_pct_of_equity'):.1%} of equity."
                        )
                        logger.info("%s: %s", ticker, risk_cap_note)
                        # Appended (2026-08-14) — same reasoning as no_stop_note above.
                        adjustment_note = f"{adjustment_note} {risk_cap_note}" if adjustment_note else risk_cap_note
                        target_shares = capped_target
                        # Only re-resolve here: shares_after actually changed and
                        # resolved_stop_loss was non-None going in, so this is the
                        # one case where the outcome (e.g. now-flat) can differ.
                        resolved_stop_loss = resolve_stop_loss(stop_loss, price, target_shares, current_atr_pct)

        # Portfolio-level caps (concentration, correlation-bucket, drawdown
        # breaker — app.execution.portfolio_risk) apply AFTER the per-ticker 1%
        # rule above, only ever further reducing target_shares, never loosening
        # it. Still inside `lock_cm` — see this function's own comment above
        # for why the read this whole block sizes against must be too.
        if target_shares > current_shares:
            pre_portfolio_cap_target = target_shares
            if portfolio_cap_override is not None:
                snapshot, clusters, drawdown_frac = portfolio_cap_override
                target_shares, portfolio_note = await apply_backtest_portfolio_caps(
                    ticker, target_shares, price, current_shares, snapshot, clusters, drawdown_frac
                )
            else:
                target_shares, portfolio_note = await apply_live_portfolio_caps(
                    ticker, target_shares, price, current_shares
                )
                # FIX-5 (2026-09-03 id=1187 audit): a fail-CLOSED portfolio-
                # risk block (the real account snapshot itself couldn't be
                # read — see PORTFOLIO_RISK_UNAVAILABLE_NOTE_PREFIX's own
                # comment) means the AI's decision never executed at all,
                # not just a routine concentration/correlation/cash-floor
                # trim — and until now that was visible only in this
                # ticker's own adjustment_note/log line, with nothing
                # raising it anywhere an operator would see it in aggregate
                # (unlike equity_basis_unavailable just above, which already
                # gets one). Distinct `kind` so its own per-(ticker, kind)
                # cooldown is independent of every other alert here.
                if portfolio_note is not None and portfolio_note.startswith(
                    PORTFOLIO_RISK_UNAVAILABLE_NOTE_PREFIX
                ):
                    for alert in filter_new_alerts(
                        [
                            Alert(
                                ticker=ticker,
                                kind="portfolio_risk_unavailable_block",
                                message=(
                                    f"{ticker} BUY blocked this cycle — {portfolio_note} "
                                    f"(target {pre_portfolio_cap_target}, held at {current_shares})."
                                ),
                                triggered_at=get_clock().now(),
                            )
                        ]
                    ):
                        await record_alert(alert)
            if portfolio_note is not None:
                adjustment_note = f"{adjustment_note} {portfolio_note}" if adjustment_note else portfolio_note
            if target_shares != pre_portfolio_cap_target:
                resolved_stop_loss = resolve_stop_loss(stop_loss, price, target_shares, current_atr_pct)

            # Per-order value cap (KIS_MAX_ORDER_KRW/_USD) — same "further
            # reduce, never loosen" chain as the 1% risk cap and the
            # portfolio caps above it, applied last so it wins over whatever
            # they left the increment at. Live only (portfolio_cap_override
            # is None; a backtest never reaches kis_bridge.mirror_trade_to_kis
            # at all, so pre-shrinking to this cap there would just be an
            # unrealistic extra constraint with nothing to gain by matching
            # it). Real incident this fixes (2026-08-03): a BUY sized to pass
            # every portfolio cap still got hard-BLOCKED at the KIS-mirror
            # step by this same cap, wasting the whole decision instead of
            # executing at a smaller, still-legitimate size —
            # kis_bridge.over_value_cap stays in place as a defense-in-depth
            # backstop (see its own docstring), this just avoids relying on
            # it for the common case.
            if portfolio_cap_override is None:
                incremental_shares = target_shares - current_shares
                if incremental_shares > 0 and price > 0:
                    krx_code, _ = route_ticker(ticker)
                    cap, currency = max_order_value_cap(krx_code)
                    max_order_shares = math.floor(cap / price)
                    if incremental_shares > max_order_shares:
                        capped_target = current_shares + max_order_shares
                        value_cap_note = (
                            f"Max per-order value cap reduced the intended add from {incremental_shares} to "
                            f"{max_order_shares} shares (target {target_shares} -> {capped_target}) to stay "
                            f"within the {cap:,.2f} {currency} per-order cap."
                        )
                        logger.info("%s: %s", ticker, value_cap_note)
                        adjustment_note = (
                            f"{adjustment_note} {value_cap_note}" if adjustment_note else value_cap_note
                        )
                        target_shares = capped_target
                        resolved_stop_loss = resolve_stop_loss(stop_loss, price, target_shares, current_atr_pct)

        # Surfaces resolve_stop_loss's own override so a diverging stop-loss
        # doesn't stay invisible outside logger.warning/.info lines nobody
        # reviews. Real incident (2026-08-20, 950160.KS): the Portfolio
        # Manager's own decision_text said "18,500 KRW stop-loss", but the
        # ATR floor (this ticker's ATR% was ~16.7%, so 2.5x = ~42% of price)
        # silently widened the actually-persisted trades.stop_loss to
        # 12,794.64 — a real, ~26-percentage-point-wider stop than the AI's
        # own reasoning described, with zero trace anywhere the AI's prose
        # was recorded (unlike the risk-cap/portfolio-cap share reductions
        # just above, which do get a note). Checked once here, after every
        # resolve_stop_loss call above has settled, rather than at each call
        # site — the floor's distance from price doesn't depend on
        # target_shares, so re-checking per call would only risk duplicate
        # notes for what is, in the end, one single divergence.
        if resolved_stop_loss is not None:
            if stop_loss is not None and stop_loss > 0 and abs(resolved_stop_loss - stop_loss) > 0.01:
                stop_note = (
                    f"Stop-loss adjusted by code-level validation from the AI's {stop_loss:,.2f} to "
                    f"{resolved_stop_loss:,.2f} (ATR floor / sanity check — see resolve_stop_loss)."
                )
                adjustment_note = f"{adjustment_note} {stop_note}" if adjustment_note else stop_note
            elif stop_loss is None or stop_loss <= 0:
                stop_note = (
                    f"No usable stop-loss from the AI ({stop_loss}) — derived {resolved_stop_loss:,.2f} "
                    f"from the ATR floor instead."
                )
                adjustment_note = f"{adjustment_note} {stop_note}" if adjustment_note else stop_note

        delta = target_shares - current_shares
        if delta == 0:
            # A BUY/OVERWEIGHT decision whose target share count floors to the
            # existing position (a high-priced ticker whose allocation
            # fraction rounds down to no new shares), or a SELL/UNDERWEIGHT
            # that sized to exactly what's already held, used to leave only a
            # logger.info nobody reads — "the AI said BUY but nothing showed
            # up" was undiagnosable after the fact. 2026-09-01, audit #10:
            # 6 of 41 real-account decisions had rating Buy / action_label
            # Hold and 5 left no explanation anywhere. A directional rating
            # here now gets an adjustment_note (the pipeline persists it as an
            # [EXECUTION NOTE], and the audit CLI / dashboard read it). Still
            # info, not warning: this is an expected outcome of the sizing
            # math on a small account, not a fault — the fix for the
            # underlying "unbuyable name reached the pipeline" is the
            # affordability filter (audit #6).
            if rating is not None and rating is not PortfolioRating.HOLD:
                allocation_value = equity * target_fraction
                # `allocation_floor_shares` is the PRE-cap floor; `target_shares`
                # here is post-cap. Only state the floor arithmetic when the two
                # agree — otherwise the sentence describes a number the caps
                # already overrode (2026-09-03 audit #4). When they disagree, the
                # cap that shrank the order has ALREADY appended its own note
                # saying so, so this adds only the fact none of those notes can
                # state: the shrink went all the way to no change at all. Never
                # a second causal claim, and never a cap's name — the backtest
                # tally (app.backtest.portfolio_harness._CAP_LABELS) substring-
                # matches this same string.
                # 캡이 이미 인과를 설명함 — 여기선 "0까지 줄었다"만, 인과 재주장 금지.
                sized_to = (
                    f"floor({allocation_value:,.0f} / {price:,.2f}) = {allocation_floor_shares} shares "
                    f"({target_fraction:.2%} of {equity:,.0f} equity)"
                )
                if rating in _DECREASE_INTENT_RATINGS and current_shares == 0:
                    gap_note = (
                        f"{RATING_ACTION_GAP_NOTE_PREFIX}: the {rating.value} rating implies reducing "
                        f"exposure, but the position is already flat at 0 shares — there was nothing to sell."
                    )
                elif allocation_floor_shares != target_shares:
                    gap_note = (
                        f"{RATING_ACTION_GAP_NOTE_PREFIX}: the {rating.value} rating sized to {sized_to}, "
                        f"and the reductions noted above brought that target down to the {current_shares} "
                        f"share(s) already held — no order was placed and the position is unchanged at "
                        f"{current_shares} shares."
                    )
                elif rating in _INCREASE_INTENT_RATINGS and current_shares == 0 and price > 0:
                    gap_note = (
                        f"{RATING_ACTION_GAP_NOTE_PREFIX}: the {rating.value} rating's target allocation "
                        f"({target_fraction:.2%} of {equity:,.0f} equity = {allocation_value:,.0f}) does not "
                        f"cover one share at {price:,.2f}, so "
                        f"floor({allocation_value:,.0f} / {price:,.2f}) = 0 shares — nothing reduced it. "
                        f"Nothing was bought; the position stays flat at 0 shares."
                    )
                elif rating in _INCREASE_INTENT_RATINGS:
                    gap_note = (
                        f"{RATING_ACTION_GAP_NOTE_PREFIX}: the {rating.value} rating sized to {sized_to}, "
                        f"which is exactly the {current_shares} share(s) already held — no shares were "
                        f"added and no order was placed."
                    )
                else:
                    gap_note = (
                        f"{RATING_ACTION_GAP_NOTE_PREFIX}: the {rating.value} rating sized to {sized_to}, "
                        f"which is exactly the {current_shares} share(s) already held — no shares were "
                        f"sold and no order was placed."
                    )
                # Appended, never overwritten — same convention as every note
                # above. When an upstream cap already clamped target_shares
                # down to current_shares, BOTH are wanted: the caps say what
                # shrank it, this says the shrink reached zero net change — no
                # duplicated causal claim.
                adjustment_note = f"{adjustment_note} {gap_note}" if adjustment_note else gap_note
            logger.info(
                "%s: rebalance target (%d shares, %.4f fraction) matches the existing position — no trade%s",
                ticker, target_shares, target_fraction, f" ({adjustment_note})" if adjustment_note else "",
            )
            return RebalanceResult(
                side=None, shares=0, shares_after=current_shares, stop_loss=resolved_stop_loss,
                action_label=rebalance_action_label(current_shares, target_shares),
                adjustment_note=adjustment_note,
            )

        side = "buy" if delta > 0 else "sell"
        # Overwritten below only if the mirror never reaches KIS and gets
        # reversed — the default (unreversed) outcome is exactly the
        # target this whole function just computed.
        final_shares_after = target_shares
        # Captured BEFORE record_trade below overwrites the latest row —
        # this is the stop still protecting the PRE-trade position, which
        # is what a reversal (this trade never actually reaching KIS) must
        # restore. See reverse_pending_trade's own docstring. Skipped
        # entirely for a backtest replay (as_of is not None): that branch
        # below always short-circuits to "skipped: backtest replay" before
        # ever reaching the reversal path, so the query's result would just
        # be discarded — not worth an extra DB round-trip per backtest trade.
        prior_open_stop = await get_open_stop_loss(ticker) if as_of is None else None
        trade_id = await record_trade(
            ticker, trade_date, side, abs(delta), price, reason,
            stop_loss=resolved_stop_loss, decision_id=decision_id, adjustment_note=adjustment_note,
            base_shares=current_shares,
        )
        # execution_mode gates the AI pipeline's own auto-trading path only —
        # never place_manual_order (dashboard button) or check_and_apply_stop_loss's
        # forced-exit sell (a risk-reducing safety mechanism, not an "AI trading
        # decision"; see ExecutionMode's own docstring for why that one always
        # runs regardless of mode). Ledger record above happens in every mode —
        # this is the paper-portfolio's own bookkeeping, independent of whether
        # KIS is mirrored at all.
        #
        # Backtest (as_of is not None) skips this whole branch outright, rather
        # than relying on execution_mode or mirror_trade_to_kis's own today-only
        # gate to keep it harmless — those aren't actually reachable-proof for a
        # backtest CLI process specifically: a fresh `python -m app.backtest`
        # process never calls app.web.state's lifespan (the only place
        # runtime_settings.load() is called), so it reads whatever
        # execution_mode happens to sit in the config file/.env, NOT whatever
        # the live dashboard has actually been set to. Confirmed live
        # (2026-07-24): with EXECUTION_MODE=semi_auto in .env.local but the
        # live dashboard actually running FULL_AUTO, a batch of backtest runs
        # still went through the SEMI_AUTO branch below and created real,
        # dashboard-visible pending orders dated in the past — exactly what
        # app.execution.pending_orders's own docstring already assumed could
        # never happen ("backtests never create one"). A capped, mode-agnostic
        # check here is correct regardless of which execution_mode a stray
        # process happens to read.
        if as_of is not None:
            disposition = "skipped: backtest replay — no live order/approval routing"
        else:
            current_execution_mode = runtime_settings.rule("execution_mode")
            if current_execution_mode is ExecutionMode.MANUAL:
                disposition = "skipped: MANUAL execution mode — AI auto-trading path disabled, use the dashboard order ticket"
            elif current_execution_mode is ExecutionMode.SEMI_AUTO:
                from app.execution.pending_orders import create_pending_order

                # trade_id links this proposal back to the speculative row
                # just written above — see create_pending_order's own
                # docstring and the pending_orders.trade_id column comment
                # for why _reject_or_expire needs this to tell a real
                # proposal apart from one created with no ledger entry at
                # all (e.g. scripts/seed_test_pending_order.py).
                pending_id = await create_pending_order(
                    ticker, trade_date, side, abs(delta), price, reason=reason, trade_id=trade_id
                )
                disposition = f"PENDING_USER_APPROVAL: SEMI_AUTO mode holds every AI order for approval (pending_order_id={pending_id})"
            else:
                # Ledger record first, KIS mirror second — a mirror failure leaves a
                # visible reconciliation gap instead of an unrecorded live order.
                disposition = await mirror_trade_to_kis(
                    ticker, trade_date, side, abs(delta), price, assumed_current_shares=current_shares,
                    trade_id=trade_id,
                )
                if _mirror_never_reached_kis(disposition):
                    # Ledger recorded this trade above (before the mirror
                    # attempt — see this branch's own comment) on the
                    # assumption it would reach KIS. It didn't, permanently,
                    # so undo that speculative entry now rather than leaving
                    # the ledger showing a position/cash change that never
                    # actually happened on the real (paper) account. Real
                    # incident (2026-08-03): a BUY blocked by the value cap
                    # left the ledger showing shares held that KIS itself
                    # never received. prior_stop_loss restores whatever stop
                    # protected the pre-trade position (captured above,
                    # before this trade's own record_trade call overwrote
                    # the latest row) — see reverse_pending_trade's docstring.
                    #
                    # Capture the ACTUAL resulting share count rather than
                    # keep reporting target_shares below — before this fix,
                    # a reversed trade's RebalanceResult still claimed
                    # shares_after=target_shares and an action_label derived
                    # from it (e.g. "Buy"), even though the position was
                    # right back where it started. That wrong result flows
                    # into the dashboard and (via the stored decision
                    # narrative) into future decisions' retrieved past
                    # context, both asserting a trade the ledger itself had
                    # just reversed one line above.
                    final_shares_after = await reverse_pending_trade(
                        ticker, side, abs(delta), price, reason=disposition,
                        prior_stop_loss=prior_open_stop[1] if prior_open_stop else None,
                    )
        await _record_mirror_disposition(trade_id, disposition)
        logger.info("KIS mirror (%s %s x%d): %s", side, ticker, abs(delta), disposition)
        if final_shares_after == target_shares:
            # Unreversed (the common case) — nothing to reconcile.
            actual_side, actual_shares, actual_stop_loss = side, abs(delta), resolved_stop_loss
        else:
            # Reversed (fully or partially, see reverse_pending_trade's own
            # clamp) — report what the ledger actually ended up holding,
            # not the target this trade was aiming for but never reached.
            actual_shares = abs(final_shares_after - current_shares)
            actual_side = None if actual_shares == 0 else ("buy" if final_shares_after > current_shares else "sell")
            actual_stop_loss = (prior_open_stop[1] if prior_open_stop else None) if actual_shares > 0 else None
            if actual_shares == 0 and rating is not None and rating is not PortfolioRating.HOLD:
                # A directional rating whose order was fully reversed nets to
                # no position change and action_label "Hold" — the same
                # rating/action gap as the delta == 0 branch, but reached a
                # different way (2026-09-01, audit #10 P1). The disposition
                # itself is recorded on the trades row by
                # _record_mirror_disposition above; this note goes ONLY on
                # the decision row (adjustment_note was already passed to
                # record_trade earlier), so a reader of the stored decision
                # sees why nothing moved.
                reversal_note = (
                    f"{RATING_ACTION_GAP_NOTE_PREFIX}: the {rating.value} rating's order for "
                    f"{abs(delta)} share(s) never reached the broker ({disposition}) and the speculative "
                    f"ledger entry was reversed — the position is back at {current_shares} share(s)."
                )
                adjustment_note = f"{adjustment_note} {reversal_note}" if adjustment_note else reversal_note
        return RebalanceResult(
            side=actual_side, shares=actual_shares, shares_after=final_shares_after, stop_loss=actual_stop_loss,
            action_label=rebalance_action_label(current_shares, final_shares_after),
            adjustment_note=adjustment_note,
        )


# Per-ticker guard around force_exit_position's whole body (2026-08-10,
# added alongside app.live.exit_monitor) — this function now has TWO
# callers that can legitimately fire for the same ticker close together:
# the hourly daily-bar stop_loss_loop and the new 5s tick-based exit
# monitor. Both call get_position/real_shares_held, see shares > 0, and
# only THEN write record_trade — without serializing the two, both could
# pass that check before either's write lands, double-selling the same
# position. An in-memory per-ticker asyncio.Lock is enough (both callers
# always run in this same process — unlike rebalance_to_target's
# market_lock, which also has to serialize against OTHER processes/
# deployments sharing the same DB, this doesn't).
#
# 2026-08-21: do NOT "fix" the resulting asymmetry (this lock doesn't
# serialize against rebalance_to_target's market_lock at all, even for the
# SAME ticker) by having force_exit_position also acquire market_lock.
# That was considered and rejected — market_lock is scoped to the whole
# MARKET, so a forced safety exit would have to wait for an unrelated
# ticker's rebalance to finish (up to ~portfolio_risk_lock_wait_seconds,
# see market-lock-hold-time-investigation) or fail outright on
# MarketLockContended — turning a safety mechanism into one that can be
# delayed or blocked by unrelated activity, exactly backwards for a stop-
# loss. Narrowing market_lock itself to per-ticker isn't right either —
# the invariants it protects (concentration/correlation-bucket/drawdown
# caps) are portfolio-wide, not per-ticker. The actual risk this asymmetry
# creates — a concurrent reader observing a forced-exit row's stop_loss
# before it's confirmed or reversed — is handled instead at the READ side:
# see `_is_provisional_forced_exit` and record-reversal-bug-evidence-
# strengthened-2026-08-21 project memory for the full design rationale.
_force_exit_locks: dict[str, asyncio.Lock] = {}


@dataclass(frozen=True)
class ForcedExitOutcome:
    """Result of one force_exit_position attempt. Added 2026-08-19: before
    this, the function returned only `int | None`, so "N shares sold" and
    "N shares' worth of exit was ATTEMPTED but KIS rejected the order,
    position still open" were indistinguishable to every caller —
    `app.live.exit_monitor` logged the latter as "sold N shares" at DEBUG,
    which is how a 43-minute, 17-attempt stop-loss failure loop on AXTI went
    unnoticed (2026-08-19)."""

    status: str  # "exited" | "already_flat" | "failed" | "cooldown"
    shares: int = 0  # actually submitted (only set when status == "exited")
    disposition: str = ""  # mirror_trade_to_kis's disposition, "" if never attempted
    detail: str = ""  # human-readable detail for logs/alerts
    breach_date: date | None = None  # filled in by check_and_apply_*

    @property
    def ok(self) -> bool:
        return self.status == "exited"


# Per-ticker state for the failure-cooldown below (process-local memory is
# enough — same reasoning as _force_exit_locks just above: both callers of
# force_exit_position run in this one process).
@dataclass
class _ExitFailureState:
    consecutive: int = 0
    until: float = 0.0  # time.monotonic() deadline
    last_disposition: str = ""


_exit_failure_state: dict[str, _ExitFailureState] = {}

# Backoff by consecutive-failure count (1st, 2nd, 3rd, 4th+) for a
# TRANSIENT failure — the condition (price below stop) stays true the
# whole time an order keeps failing, so without this a broken order every
# retries every ~5s tick forever. 2026-08-19: AXTI's stop-loss failed 17
# times in 43 minutes (~2.5min average gap, no backoff at all).
_EXIT_BACKOFF_SECONDS = (60.0, 300.0, 900.0, 3600.0)
# A PERMANENT failure (malformed request, see kis_bridge.is_permanent_refusal)
# means the exact same payload will fail again no matter how soon it's
# retried, so it goes straight to the longest backoff from the first
# failure — but still just 1 hour, not forever: a fix landing (or the
# broker recovering) should let a still-breached stop resume trying within
# a bounded window rather than staying silently suppressed all day.
_EXIT_PERMANENT_BACKOFF_SECONDS = 3600.0
# Alert severity escalates once a ticker has failed this many times in a row.
_EXIT_FAILURE_ALERT_ESCALATION = 3

# How old the freshest cached tick may be and still be recorded as a forced
# exit's fill price. Deliberately MUCH looser than
# `settings.exit_monitor_max_tick_age_seconds` (30s), and the difference is
# not an oversight: that gate answers "is this price current enough to SELL
# on", where stale data must read as unknown rather than safe, while this one
# answers "is this price a better description of what we just sold at than a
# threshold nothing ever traded at". A 10-minute-old real print is a far
# better answer to the second question than the stop floor, and it can never
# cause a trade — it is only ever written down after the order has already
# gone out. 900s also comfortably absorbs a 60s overseas poller hiccup or a
# brief domestic websocket reconnect while staying far shorter than any
# session, so an accepted price is always from the same session as the order.
_FORCED_EXIT_FILL_PRICE_MAX_TICK_AGE_SECONDS = 900.0

# The exact, stable prefix of the `trades.adjustment_note` a forced exit gets
# when no live price was knowable and the STOP THRESHOLD had to be recorded in
# `trades.price` instead. A literal, not an f-string: two downstream
# consumers match on it (app.audit.slippage_validation, app.orchestrator.
# pipeline's _position_context via `is_threshold_placeholder_note` below),
# and it is also the marker BACKFILLED onto the pre-2026-08-27 daily-bar
# forced exits whose price column holds a threshold for the same reason (the
# daily-bar backstop always passed the threshold as exit_price before this
# function existed — only the tick monitor ever passed a real fill).
# Changing this string silently un-hides those fabricated prices in the
# slippage report and in five LLM nodes' re-entry narrative — treat it as a
# wire format, not prose.
FORCED_EXIT_THRESHOLD_PRICE_NOTE_PREFIX = "forced-exit price basis: NO live price available"


def is_threshold_placeholder_note(note: str | None) -> bool:
    """True when `trades.adjustment_note` says this row's `price` is a stop
    THRESHOLD, not an observed fill — either written live by
    `_resolve_forced_exit_price`'s last-resort branch, or backfilled onto a
    pre-2026-08-27 daily-bar forced exit (see
    FORCED_EXIT_THRESHOLD_PRICE_NOTE_PREFIX's own comment). `startswith`, not
    `in`: a backfilled note appends row-specific evidence AFTER the marker,
    and a substring test would also fire on a note that merely quotes it."""
    return note is not None and note.startswith(FORCED_EXIT_THRESHOLD_PRICE_NOTE_PREFIX)


def _resolve_forced_exit_price(
    ticker: str, *, order_price: float, floor: float, fill_price: float | None
) -> tuple[float, str | None]:
    """(price to record in `trades.price`, provenance note for
    `trades.adjustment_note`) for a forced exit.

    THE 047040.KS INCIDENT, 2026-08-27, paper account — why this exists.
    A trailing-stop forced exit recorded `trades` id=280:
    `047040.KS sell x80 price=19729.244343797276 stop_loss=19729.244343797276`,
    reason `"stop-loss triggered: stop-loss monitor — breach 2026-08-27: low
    18,760.00 <= floor 19,729.24 …"`. 19,729.24 is the TRAILING-STOP FLOOR —
    a threshold, not a price anything ever traded at. The order was a KRX
    MARKET sell submitted 09:49 KST; the 09:45 5-minute bar was
    O 21,350 / H 21,400 / L 20,850 / C 21,000 and the 09:50 bar opened
    20,950, so the real fill was ~21,000. Recorded proceeds
    19,729.24 x 80 = 1,578,339 KRW against an actual ~1,680,000 — the ledger
    understated the sale by ~101,660 KRW (~6.4%) and recorded the trade's
    return as +4.17% instead of ~+10.9% against the 18,940 entry. Nothing in
    the row hinted at the real number: `mirror_disposition`'s
    "~1,578,340 KRW" is just floor x 80, derived from the same threshold.
    (The exit itself was ALSO spurious — a separate defect fixed the same day
    in `b503d5e`, see `check_and_apply_trailing_stop`'s own docstring — but
    the two are independent: a CORRECT forced exit recorded the same false
    price, and would have kept doing so.)

    Resolution order, and the reasoning for each step:

    1. `fill_price` supplied by the caller — `app.live.exit_monitor` already
       holds the breaching tick and has already applied its own freshness
       gate to it. Trusted as-is: re-reading the cache here would give the
       same number a few seconds later and make the tick monitor's own
       behavior non-deterministic in tests.
    2. `is_backtest_context()` short-circuits to `order_price` with NO note,
       BEFORE any cache read. A replay has no tick cache, and its ledger rows
       feed the 4-stage gate criteria — see `check_and_apply_trailing_stop`'s
       docstring on why replay output must stay byte-identical, and project
       memory for the 2026-08-21 cost-model change that made earlier
       backtests incomparable. Ordered before step 1 in the code below only
       because no backtest caller passes `fill_price` anyway; either order is
       equivalent, and both are byte-identical for a replay.
    3. The freshest cached tick (`app.live.last_price.freshest_tick`), if
       within `_FORCED_EXIT_FILL_PRICE_MAX_TICK_AGE_SECONDS`. This is the
       right answer even when `exit_date` is a PAST gap day: the daily-bar
       backstop discovers a breach that happened days ago, but the market
       order it sends executes NOW, at today's price. The ledger already
       applies exactly this reasoning to the DATE (`record_as_of=today`,
       2026-08-14) — this applies it to the PRICE. Had this existed on
       2026-08-27, 047040.KS would have recorded ~21,000: `trailing_stop_state`
       shows the websocket feed was live and ratcheting throughout that
       session, so a fresh domestic tick was certainly in the cache.
    4. Otherwise `order_price` (the pre-2026-08-27 behavior) WITH a note
       saying so in as many words. Falling back to the threshold is the only
       option that cannot fail, cannot block, and cannot invent a number —
       but a threshold recorded silently is exactly the bug this function
       exists to end, so the row is labelled instead. Never raises: a price
       lookup must never delay, block, or break a safety exit (see the
       broad `except` below), and any failure lands here.

    Does NOT change the ORDER — see `_force_exit_locked`'s call to
    `mirror_trade_to_kis`, which keeps receiving `order_price` untouched."""
    if is_backtest_context():
        return order_price, None
    if fill_price is not None and fill_price > 0:
        return fill_price, (
            f"forced-exit price basis: live tick {fill_price:,.2f} observed by the breach detector "
            f"(stop threshold {floor:,.2f})"
        )
    why = "no live tick cached for this ticker"
    try:
        snapshot = freshest_tick(ticker)
        if snapshot is not None:
            tick_price, received_at = snapshot
            age = (get_clock().now(timezone.utc) - received_at).total_seconds()
            if tick_price > 0 and age <= _FORCED_EXIT_FILL_PRICE_MAX_TICK_AGE_SECONDS:
                return tick_price, (
                    f"forced-exit price basis: freshest live tick {tick_price:,.2f} ({age:,.0f}s old) at "
                    f"order time (stop threshold {floor:,.2f})"
                )
            why = (
                f"cached tick {age:,.0f}s old (> {_FORCED_EXIT_FILL_PRICE_MAX_TICK_AGE_SECONDS:,.0f}s)"
                if tick_price > 0
                else "cached tick has a non-positive price"
            )
    except Exception:  # noqa: BLE001 - a price LOOKUP must never delay or break a safety EXIT
        logger.warning(
            "%s: freshest_tick failed while recording a forced exit — recording the stop threshold "
            "instead and flagging the row", ticker, exc_info=True,
        )
        why = "live price lookup raised"
    logger.warning(
        "%s: forced exit has no usable live price (%s) — recording the stop threshold %.4f as the trade "
        "price; this row is NOT a traded price and is flagged in adjustment_note", ticker, why, order_price,
    )
    return order_price, (
        f"{FORCED_EXIT_THRESHOLD_PRICE_NOTE_PREFIX} ({why}) — recorded at the stop threshold "
        f"{floor:,.2f}, which is a threshold and NOT a traded price; exclude this row from slippage and "
        f"return statistics"
    )


async def force_exit_position(
    ticker: str,
    exit_date: date,
    exit_price: float,
    reason: str,
    *,
    record_as_of: date | None = None,
    fill_price: float | None = None,
    breach_floor: float | None = None,
) -> ForcedExitOutcome:
    """Sells the entire real position to flat, bypassing
    execution_mode/SEMI_AUTO approval (a forced safety exit must never wait
    on a human click) — the shared path BOTH the daily-bar stop-loss check
    (`check_and_apply_stop_loss`, below) and the tick-based real-time exit
    monitor (`app.live.exit_monitor`) call once a breach is confirmed.
    Returns a `ForcedExitOutcome` — see that dataclass's own docstring for
    why a bare `int | None` used to hide a failed exit as a successful one.

    `record_as_of`, if given, is the `trades.trade_date` this exit gets
    recorded under, OVERRIDING `exit_date` for that one column only —
    everything else (KIS mirror timing, the exit price) still uses
    `exit_date`/today as before. Added 2026-08-14: the live daily-bar
    backstop (`check_stops_for_ticker`) walks PAST days looking for the
    first one whose real low breached the stop, so `exit_date` there is
    that historical breach day — recording the resulting `trades` row
    under that SAME past date (the old, only behavior) buried the sell
    under weeks of newer trades on any `trade_date DESC` view, and worse,
    `get_open_stop_loss`/`get_open_stop_loss_entry_time` themselves sort by
    `trade_date DESC, id DESC` — a back-dated sell can sort BELOW a newer
    buy and the position reads as still open with its stop already
    consumed (same class of bug already documented on the KIS-mirror side,
    see this function's own git history). Live callers now pass
    `record_as_of=today` so the row lands where it actually happened;
    backtest callers (`app.backtest.harness`) omit it, so a replay's own
    trade_date stays the (correct, since it's a historical replay) breach
    day, unchanged.

    THREE PRICES, not one (2026-08-27). `exit_price` used to be spent three
    ways at once — the ORDER price handed to `mirror_trade_to_kis`, the
    `trades.price` recorded for the sale, and the `trades.stop_loss`
    threshold — and no caller could ever get all three right, because the
    two callers hold DIFFERENT numbers. The daily-bar backstop
    (`check_and_apply_stop_loss` / `check_and_apply_trailing_stop`) passes
    the breached THRESHOLD, so `trades.price` recorded a number nothing had
    traded at; the tick monitor (`app.live.exit_monitor`) passes the
    breaching TICK, so `trades.stop_loss` recorded a price instead of the
    floor. Both halves are now separable:

    - `exit_price` — UNCHANGED in meaning and unchanged in every value any
      caller passes: the ORDER price, forwarded verbatim to
      `mirror_trade_to_kis`. It is the overseas limit price and the input to
      that function's value-cap/approval/cash gates, so nothing here may
      move it. Also still the default for the two below.
    - `fill_price` — what actually goes in `trades.price` (and therefore
      what `record_trade` computes `cash_after`/`transaction_cost` from,
      and what the reversal branch below refunds against). None means
      "resolve it here" — see `_resolve_forced_exit_price`, which prefers
      the freshest cached tick, records WHY in `trades.adjustment_note`, and
      falls back to `exit_price` (the pre-2026-08-27 behavior) if no live
      price is knowable. That fallback is the one branch that can still
      write a threshold into a price column, and it is now labelled as such
      in the row itself rather than being indistinguishable from a real fill.
    - `breach_floor` — what goes in `trades.stop_loss`: the threshold that
      was actually breached. None keeps the old `stop_loss=exit_price`.

    A live forced exit's fill price is TODAY's price even when `exit_date`
    is a past gap day: the backstop discovers a breach that happened days
    ago, but the market order it sends fills NOW. This is the same argument
    `record_as_of` already makes for the DATE, applied to the PRICE.

    Every one of these arguments defaults to None, and with them all None
    this function behaves exactly as it did before 2026-08-27 — which is
    what `app.backtest.harness`'s replay path gets (it reaches here only
    through `check_and_apply_stop_loss`/`check_and_apply_trailing_stop`,
    neither of whose signatures changed), reinforced by
    `_resolve_forced_exit_price`'s own `is_backtest_context()` short-circuit.
    The incident that forced all of this — 047040.KS, id=280, ~101,660 KRW
    of a real sale missing from the ledger — is written out in full in
    `_resolve_forced_exit_price`'s docstring.

    Same "real KIS quantity over internal ledger" preference already
    applied in rebalance_to_target, for the same reason: after a paper
    account reset (or any other ledger/real-balance drift), the internal
    `trades` row can say "holding N shares" when the real account holds
    something else entirely, and forcing an exit for the WRONG quantity
    both mis-records the ledger and mirrors a bad size to KIS.
    `real_shares_held` returns None (not 0) when no live broker is
    connected — always the case for a backtest process — so this falls
    back to the internal ledger there exactly as before.
    리셋 등으로 내부 원장과 실제 잔고가 어긋나도 실제 KIS 보유수량 기준으로 청산."""
    lock = _force_exit_locks.setdefault(ticker, asyncio.Lock())
    async with lock:
        state = _exit_failure_state.get(ticker)
        if state is not None and time.monotonic() < state.until:
            remaining = state.until - time.monotonic()
            # INFO, not debug: for as long as this suppression window is in
            # effect, the position is sitting past its stop with nothing
            # protecting it — worth seeing in a normal log scan, not just
            # when someone goes looking for a problem.
            logger.info(
                "%s: forced exit suppressed for another %.0fs after %d consecutive failures "
                "(last: %s) — position is still open BELOW its stop",
                ticker, remaining, state.consecutive, state.last_disposition,
            )
            return ForcedExitOutcome(
                status="cooldown",
                disposition=state.last_disposition,
                detail=f"retry suppressed for {remaining:.0f}s after {state.consecutive} consecutive failures",
            )
        return await _force_exit_locked(
            ticker, exit_date, exit_price, reason,
            record_as_of=record_as_of, fill_price=fill_price, breach_floor=breach_floor,
        )


async def _force_exit_locked(
    ticker: str,
    exit_date: date,
    exit_price: float,
    reason: str,
    *,
    record_as_of: date | None = None,
    fill_price: float | None = None,
    breach_floor: float | None = None,
) -> ForcedExitOutcome:
    position = await get_position(ticker)
    real_shares = await real_shares_held(ticker)
    shares_to_exit = real_shares if real_shares is not None else position.shares
    if shares_to_exit <= 0:
        _exit_failure_state.pop(ticker, None)  # flat again — any prior failure streak is moot
        return ForcedExitOutcome(status="already_flat")

    # Captured BEFORE record_trade below writes a new latest row, so a
    # failed mirror can restore the stop that was actually protecting this
    # position going in — see the reversal branch further down for why.
    prior_open_stop = await get_open_stop_loss(ticker)

    # The threshold that was actually breached (2026-08-14: this column used
    # to be NULL, so a human reviewing the trade had no number to check the
    # exit against without reading code/logs). Defaults to `exit_price`,
    # which IS the floor for both daily-bar callers — only the tick monitor
    # needs to say otherwise, and now can.
    effective_floor = breach_floor if breach_floor is not None else exit_price
    # Resolved BEFORE record_trade and reused by the reversal branch below:
    # `reverse_pending_trade` recomputes cash from the price it is given, so
    # recording at one price and reversing at another would leave cash
    # permanently off by the difference on every never-reached-KIS exit.
    # Same number, both legs, always.
    recorded_price, price_basis = _resolve_forced_exit_price(
        ticker, order_price=exit_price, floor=effective_floor, fill_price=fill_price
    )
    trade_id = await record_trade(
        ticker, record_as_of or exit_date, "sell", shares_to_exit, recorded_price,
        f"{STOP_LOSS_REASON_PREFIX}{reason}",
        stop_loss=effective_floor,
        base_shares=shares_to_exit,
        # Provenance for the price just recorded — free on this path (no
        # forced-exit row has ever set adjustment_note) and the only way a
        # later reader can tell a real fill price from the threshold
        # fallback. NULL in a backtest, keeping replay rows byte-identical.
        adjustment_note=price_basis,
    )
    # Mirrored with today's date, not exit_date: the breach was discovered
    # now, so the KIS position (if any) is closed now — and the today-only
    # gate would drop a past-dated mirror anyway (only matters for the
    # daily-bar caller, whose exit_date can be a past trading day; the
    # tick-based caller's exit_date is always today already).
    # local_today(ticker_market(ticker)), not get_clock().today() —
    # mirror_trade_to_kis's own today-only gate compares against THIS
    # market's local calendar date (KST domestic / ET overseas), not the
    # shared KST-based get_clock().today(). Using the wrong one here meant
    # every overseas stop-loss exit filed between 11:00-16:00 ET (KST has
    # already rolled to the next calendar day) was rejected as "not today"
    # — with no reversal, since that gate's own disposition was "skipped:",
    # not "BLOCKED:" (see _mirror_never_reached_kis and kis_bridge.py's own
    # fix below). Net effect before this fix: the ledger recorded the
    # position closed, KIS still held it, and the stop-loss row was
    # consumed, so check_stops_for_ticker never re-fired — an unprotected
    # real position with no further monitoring.
    # approved=True: bypasses requires_approval's value-threshold hold
    # inside mirror_trade_to_kis — a SEPARATE mechanism from the
    # execution_mode/SEMI_AUTO approval-queue bypass described above (that
    # one is structural, since this function never goes through
    # rebalance_to_target's mode branching at all). Without this, a
    # stop-loss exit large enough to cross kis_approval_threshold_krw/usd
    # got diverted into PENDING_USER_APPROVAL — an automated safety exit
    # waiting on a human click contradicts the whole point of an automatic
    # stop-loss, and is exactly the "bypasses approval by design" intent
    # just described, just not actually enforced against this specific
    # gate until now.
    disposition = await mirror_trade_to_kis(
        ticker, local_today(ticker_market(ticker)), "sell", shares_to_exit, exit_price,
        assumed_current_shares=shares_to_exit, approved=True, trade_id=trade_id,
    )
    if _mirror_never_reached_kis(disposition):
        # Same reconciliation as rebalance_to_target's FULL_AUTO branch —
        # the forced-exit sell was recorded on the ledger above before we
        # knew the mirror would never reach KIS. Restore the stop that was
        # ACTUALLY protecting this position beforehand (prior_open_stop,
        # captured above), not prior_stop_loss=exit_price (the old
        # behavior) — re-arming the stop at the exact price that just
        # failed to sell guarantees an immediate re-breach on the next
        # tick, which is what turned a single failed order into AXTI's
        # 17-attempt, 43-minute retry storm on 2026-08-19. See
        # reverse_pending_trade's own docstring for why the compensating
        # row must carry a stop at all rather than leaving it None.
        await reverse_pending_trade(
            # `recorded_price`, NOT `exit_price` — this must undo the exact
            # cash movement `record_trade` above made (notional AND the
            # bps-of-notional fee `transaction_cost_override` refunds), and
            # that was computed from `recorded_price`. Reversing at a
            # different price would leave the position correctly restored
            # but the cash permanently wrong.
            ticker, "sell", shares_to_exit, recorded_price, reason=disposition,
            prior_stop_loss=prior_open_stop[1] if prior_open_stop else None,
        )
        state = _exit_failure_state.setdefault(ticker, _ExitFailureState())
        state.consecutive += 1
        state.last_disposition = disposition
        permanent = failure_is_permanent(disposition)
        backoff = (
            _EXIT_PERMANENT_BACKOFF_SECONDS if permanent
            else _EXIT_BACKOFF_SECONDS[min(state.consecutive - 1, len(_EXIT_BACKOFF_SECONDS) - 1)]
        )
        state.until = time.monotonic() + backoff
        outcome = ForcedExitOutcome(status="failed", disposition=disposition)
    else:
        _exit_failure_state.pop(ticker, None)  # success (or a dry-run/disabled skip) clears any streak
        state = None
        outcome = ForcedExitOutcome(status="exited", shares=shares_to_exit, disposition=disposition)
    await _record_mirror_disposition(trade_id, disposition)
    logger.info("KIS mirror (stop-loss sell %s x%d): %s", ticker, shares_to_exit, disposition)
    # Forced-exit sells bypass execution_mode/SEMI_AUTO approval by design
    # (see ExecutionMode's docstring) -> that's the one path where the user
    # could otherwise learn about a sell only by noticing their position is
    # gone. Surface it through the same alerts feed price/volume alerts
    # already use.
    if outcome.ok:
        alert = Alert(
            ticker=ticker,
            kind="stop_loss",
            # `effective_floor` in the threshold slot and the fill price
            # alongside it — an alert that quoted one number for both read
            # as "sold at the stop", which for a market order it never is.
            # The clause is omitted when the two are equal, which is every
            # backtest row and every threshold-fallback row, so those
            # messages are unchanged.
            message=(
                f"{ticker} stop-loss ({effective_floor:,.2f}) breached on {exit_date} — "
                f"sold {shares_to_exit} shares"
                + (f" at ~{recorded_price:,.2f}" if recorded_price != effective_floor else "")
                + f". Reason: {reason}. KIS: {disposition}"
            ),
            triggered_at=get_clock().now(),
        )
    else:
        # Separate `kind` from the success alert above (2026-08-19) — reusing
        # "stop_loss" would let filter_new_alerts' per-(ticker, kind) cooldown
        # suppress a genuine failure alert right after (or right before) a
        # genuine success one for the same ticker, and vice versa.
        escalated = state is not None and state.consecutive >= _EXIT_FAILURE_ALERT_ESCALATION
        alert = Alert(
            ticker=ticker,
            kind="stop_loss_failed",
            message=(
                f"{'CRITICAL — ' if escalated else ''}{ticker} stop-loss ({effective_floor:,.2f}) breached on "
                f"{exit_date} but the exit order did NOT reach KIS"
                + (f" ({state.consecutive} consecutive failures)" if state is not None else "")
                + f". POSITION IS STILL OPEN AND UNPROTECTED — {shares_to_exit} share(s). "
                f"Reason: {reason}. KIS: {disposition}"
            ),
            triggered_at=get_clock().now(),
        )
    for fresh in filter_new_alerts([alert]):
        await record_alert(fresh)
    return outcome


# How far an intraday bar set may disagree with the SAME day's daily bar on
# that day's own high/low before the bar set is distrusted and the day falls
# back to the conservative daily rule. 0.5%: in the 047040.KS incident the
# two feeds agreed to the won (yfinance's 09:00 5-minute low == the 18,760.00
# daily low the false exit cited), so a disagreement past half a percent
# means they are describing different sessions — a truncated/gapped intraday
# download, an unadjusted split — and the coarse rule, not the bars, is then
# the safe answer.
_INTRADAY_EXTREME_TOLERANCE = 0.005
# Ceiling on intraday fetches per backstop pass per ticker, newest days
# first. Normally 1-2 (the watermark absorbs everything older), but a
# position whose last decision is months old would otherwise try to fetch a
# 100-day walk; yfinance's own 5-minute retention is ~60 days anyway, so the
# older tail was always going to be conservative-only.
_MAX_INTRADAY_ADJUDICATION_DAYS = 10


async def check_and_apply_stop_loss(
    ticker: str,
    daily_lows: dict[date, float],
    stop_loss: float,
    reason: str,
    *,
    record_as_of: date | None = None,
) -> ForcedExitOutcome | None:
    """Walks `daily_lows` in date order and force-exits (sells to flat) the
    first day the low breaches `stop_loss`, via `force_exit_position`.
    Returns the resulting `ForcedExitOutcome` (with `.breach_date` set), or
    `None` if never breached. `outcome.ok` distinguishes an actual sell from
    a failed/suppressed exit attempt — see `ForcedExitOutcome`'s own
    docstring; callers that only care about "did a breach happen" (the
    backtest replay) can keep treating a non-None return as one, same as
    before this returned a bare date.

    `record_as_of` is forwarded to `force_exit_position` unchanged — see
    that function's own docstring. `None` (the default, every backtest
    caller's path) records the trade under the historical breach day
    itself, correct for a replay; a live caller (`check_stops_for_ticker`)
    passes today's date instead, so the resulting `trades` row reflects
    when this actually executed, not the past day it was found to have
    breached."""
    for day in sorted(daily_lows):
        if daily_lows[day] <= stop_loss:
            detail = (
                f"{reason} — breach {day}: low {daily_lows[day]:,.2f} <= stop {stop_loss:,.2f}"
            )
            outcome = await force_exit_position(ticker, day, stop_loss, detail, record_as_of=record_as_of)
            if outcome.status == "already_flat":
                return None
            return _dataclasses_replace(outcome, breach_date=day)
    return None


async def check_and_apply_trailing_stop(
    ticker: str,
    daily_highs: dict[date, float],
    daily_lows: dict[date, float],
    pm_stop_loss: float,
    entry_price: float,
    high_water: float,
    trailing_pct: float,
    reason: str,
    *,
    record_as_of: date | None = None,
    intraday_bars: dict[date, list[IntradayBar]] | None = None,
    adjudicated_through: date | None = None,
    entry_time: datetime | None = None,
    live_high_water: float | None = None,
    live_high_water_day: date | None = None,
) -> tuple[ForcedExitOutcome | None, float]:
    """Trailing-stop counterpart to `check_and_apply_stop_loss`. Originally
    backtest-only (`app.backtest.harness` — see that module's own docstring
    for why it keeps this state in memory rather than in this module's
    live `trailing_stop_state` table); now also the live daily-bar backstop's
    engine (`check_stops_for_ticker`, 2026-08-14) — see that function's own
    docstring for why it used to compute ONE static floor from TODAY's
    high-water and apply it retroactively to every past day instead of
    walking the ratchet day by day like this function always has (memory:
    live-trailing-stop-retroactive-breach-2026-08-14).

    Walks days in date order, ratcheting `high_water` UP only (never down —
    same one-directional ratchet as `trailing_stop.update_high_water_mark`).
    HOW each day's low is tested depends on what evidence that day has:

    (3) NO intraday evidence — the original rule, unchanged, and the only
    branch a backtest replay ever reaches: fold the day's own HIGH into
    `high_water`, then test that SAME day's LOW against the resulting
    `effective_stop_loss`. Within a real session the high and low can occur
    in either order; assuming the high came first is the conservative
    choice, since it can only make the exit trigger EARLIER (a tighter
    floor) or equal, never later. That is the right answer with daily bars
    and nothing else — you cannot know the order, and erring toward exiting
    protects capital.

    (2) WITH a trusted intraday bar set for that day (`intraday_bars`, live
    path only) — scan the bars in real time order: ratchet from each bar's
    high, then test THAT bar's low against the floor that high produced.
    This is the identical computation `app.live.exit_monitor` performs per
    tick, at 5-minute instead of 5-second resolution, so it cannot miss any
    breach the tick monitor would have caught: every bar's low is tested
    against exactly the floor that was genuinely in force when that low
    happened, no sub-interval of the session goes unscanned, and the
    conservative high-before-low assumption survives inside each single
    5-minute bar. Every breach it declines relative to (3) is a pair whose
    high occurred AFTER the low — i.e. not a breach at all.

    (1) Before scanning bars, one ordering-independent test: a low at or
    below the floor ALREADY in force entering the day (built from strictly
    earlier days' highs) breached under every possible ordering, so it needs
    no evidence. This is what covers a real low sitting inside the
    tolerance band above the lowest bar low.

    A bar set is trusted for a day only if it independently reproduces that
    day's own daily high and low within `_INTRADAY_EXTREME_TOLERANCE` — a
    set that never reaches the day's real low has a hole in it exactly where
    a breach could hide, and a day that fails reconciliation falls back to
    (3) rather than being adjudicated off partial data. When bars ARE used,
    the day's daily high is still folded into `high_water` at the end of the
    scan: an understated high-water understates every LATER day's floor,
    which is the missed-breach direction, and that is the one direction this
    function must never move in. (Exception: the entry day, whose daily high
    may be a PRE-entry high this position never actually had — see
    `entry_time` below.)

    WHY (2)/(1) exist — the 047040.KS (대우건설) incident, 2026-08-27,
    paper account. Entry 2026-08-26 12:36 KST @ 18,940 x 80 shares, pm stop
    15,065.06, trailing_pct 10.1173%. On 2026-08-27 this walk force-
    liquidated the whole position: "breach 2026-08-27: low 18,760.00 <=
    floor 19,729.24 (pm 15,065.06, HWM 21,950.00, trail 10.12%)". The real
    5-minute session (KST): the day's LOW of 18,760 was the 09:00 bar; the
    day's HIGH of 21,950 — the high that produced the 19,729.24 floor — was
    the 09:30 bar, THIRTY MINUTES LATER; and the lowest low at any point
    after 09:30 was 19,800 (10:50), which is 70.76 KRW ABOVE that floor. The
    floor was never breached once it existed. Reaching 18,760 would have
    needed HWM >= 18,760/0.898827 = 20,871, and 047040 had never traded
    there before 09:25 that morning (prior day's high: 19,760). The market
    sell went out at 09:49 with the price around 21,000 — the position was
    up ~+11% when it was liquidated as a stop-loss casualty.
    `trailing_stop_state` showed high_water_price=21,850 updated
    2026-08-27 00:49:19Z (siblings as late as 02:35Z), so
    `app.live.exit_monitor` was up, ratcheting, and correctly did NOT fire:
    the tick monitor was right and this backstop manufactured a breach the
    ticks never saw.

    Same defect family as the cross-day version already fixed on
    2026-08-14 (memory: live-trailing-stop-retroactive-breach-2026-08-14,
    042660.KS and 000810.KS) — that fix removed today's high-water from PAST
    dates and deliberately left the same-day case standing. Note what the
    naive repair would NOT have fixed: skipping (or lagging) today's low
    only postpones this by one day, because tomorrow today becomes a past
    day and (3) re-manufactures the identical breach — and a lagged high
    opens a real MISSED-breach hole (rally to a new high, then genuinely
    crash through the NEW floor the same session) that (2) catches and a lag
    would not. That is why the fix resolves the ordering rather than
    avoiding it, and why `adjudicated_through` persists the result.

    `intraday_bars`: {day: bars, oldest-first, from
    `app.dataflows.yfinance.get_intraday_bars`}. `adjudicated_through`: days
    at or below it have their HIGHS ratcheted but their LOWS never tested —
    already settled by an earlier pass (see `trailing_stop`'s own schema
    comment for why the verdict is immutable within a streak, and why the
    watermark is refused when the pm stop has moved). `entry_time`: bars at
    or before the moment of entry are dropped from the entry day's scan, and
    the entry day's daily high is then NOT folded in — a pre-entry spike was
    never this position's high-water (`update_high_water_mark` baselines at
    `max(entry_price, current_price)` from entry onward), and folding it in
    over-tightens every later floor, which is a second, independent
    generator of exactly the false liquidation above. `entry_time` is UTC
    while `day` is market-local; the two agree for every market this project
    trades (KRX 00:00-06:30 UTC, US 13:30-20:00 UTC — neither straddles a
    UTC date boundary) and `get_intraday_bars` normalizes bars to UTC, so
    the comparison needs no timezone table. `live_high_water` /
    `live_high_water_day`: folded into the LAST bar of that day's scan (a
    tick high certainly happened at or before now) and into `high_water`
    after it — the same "a stale live value self-heals rather than
    understating today's floor" property the caller used to get by writing
    `daily_highs[today]` directly.

    EVERY keyword-only argument above defaults to None, and with them all
    None this function is byte-identical to its pre-2026-08-27 behavior —
    same days walked, same floors, same exit prices, same detail strings.
    That is deliberate and load-bearing: `app.backtest.harness` (the only
    other caller) has no tick monitor, no live state, and no intraday feed,
    its whole premise is walking historical daily bars, and its results feed
    the 4-stage gate criteria — silently changing them would destroy
    cross-run comparability, the same harm project memory records from the
    2026-08-21 cost-model change. The live path passes the arguments; the
    replay does not, and its numbers do not move.

    Returns `(outcome_or_None, the final high_water after this call)` — the
    caller threads the second value into the NEXT gap's call as its own
    `high_water` input, the same way it already threads `prev` forward for
    the plain stop-loss gap check. See `check_and_apply_stop_loss`'s own
    docstring for what changed about the first element (`ForcedExitOutcome`
    instead of a bare date).

    `record_as_of`: see `check_and_apply_stop_loss`'s identical parameter."""
    bars_by_day = intraday_bars or {}
    entry_day = entry_time.date() if entry_time is not None else None

    def _floor(hw: float) -> float:
        return trailing_stop.effective_stop_loss(pm_stop_loss, entry_price, hw, trailing_pct)

    async def _exit(day: date, floor: float, low: float, hw: float, basis: str):
        detail = (
            f"{reason} — breach {day}: low {low:,.2f} <= floor {floor:,.2f} "
            f"(pm {pm_stop_loss:,.2f}, HWM {hw:,.2f}, trail {trailing_pct:.2%}{basis})"
        )
        outcome = await force_exit_position(ticker, day, floor, detail, record_as_of=record_as_of)
        if outcome.status == "already_flat":
            return None
        return _dataclasses_replace(outcome, breach_date=day)

    for day in sorted(set(daily_highs) | set(daily_lows) | set(bars_by_day)):
        day_high = daily_highs.get(day)
        day_low = daily_lows.get(day)
        settled = adjudicated_through is not None and day <= adjudicated_through
        live_today = live_high_water is not None and live_high_water_day == day

        # A bar set is trusted for this day only if it independently
        # reproduces that day's OWN high and low (within
        # _INTRADAY_EXTREME_TOLERANCE) — a set that never reaches the day's
        # real low has a hole in it exactly where a breach could hide. For
        # the entry day, reconciliation uses the UNFILTERED set: the day's
        # daily bar covers the whole session including before entry, so
        # validating the post-entry subset against it would fail every time.
        bars_all = bars_by_day.get(day) or []
        usable_bars = bool(bars_all) and (
            (day_low is None or min(b.low for b in bars_all) <= day_low * (1 + _INTRADAY_EXTREME_TOLERANCE))
            and (day_high is None or max(b.high for b in bars_all) >= day_high * (1 - _INTRADAY_EXTREME_TOLERANCE))
        )

        if usable_bars:
            # (1) The floor ALREADY in force entering this day, built from
            # strictly earlier days' highs only. A low at or below this
            # breached under every possible intraday ordering, so this test
            # needs no evidence at all — it is what covers a real low that
            # sits inside the tolerance band above the lowest bar low.
            if day_low is not None and not settled:
                floor_in = _floor(high_water)
                if day_low <= floor_in:
                    return await _exit(
                        day, floor_in, day_low, high_water, ", floor already in force at session start"
                    ), high_water

            # (2) Bar-by-bar, in real time order: ratchet from each bar's
            # high, then test THAT bar's low against the floor that high
            # produced — the same computation app.live.exit_monitor performs
            # per tick, at 5-minute instead of 5-second resolution.
            bars_scan = bars_all
            if entry_day is not None and day == entry_day:
                bars_scan = [b for b in bars_all if b.end > entry_time]
            last = len(bars_scan) - 1
            for index, bar in enumerate(bars_scan):
                bar_high = bar.high
                if live_today and index == last:
                    bar_high = max(bar_high, live_high_water)
                high_water = max(high_water, bar_high)
                if settled:
                    continue
                floor = _floor(high_water)
                if bar.low <= floor:
                    return await _exit(
                        day, floor, bar.low, high_water, f", intraday bar {bar.start:%H:%M}Z"
                    ), high_water
            # Never let the bar path leave the high-water BELOW what the
            # conservative path would have carried forward — an understated
            # high-water understates every later day's floor, which is the
            # missed-breach direction. Skipped on the entry day, where the
            # daily high may be a PRE-entry high this position never had.
            if day_high is not None and day != entry_day:
                high_water = max(high_water, day_high)
            if live_today:
                high_water = max(high_water, live_high_water)
            continue

        # (3) No usable intraday data for this day: the pre-2026-08-27 rule,
        # unchanged — fold the day's high, then test the day's low against
        # the resulting floor. With every keyword-only argument above left at
        # its default this is the ONLY branch that runs, which is what keeps
        # app.backtest.harness byte-identical.
        if day_high is not None:
            high_water = max(high_water, day_high)
        if live_today:
            high_water = max(high_water, live_high_water)
        if day_low is None or settled:
            continue
        floor = _floor(high_water)
        if day_low <= floor:
            return await _exit(day, floor, day_low, high_water, ""), high_water

    return None, high_water


async def check_stops_for_ticker(ticker: str, reason: str = "stop-loss monitor") -> str:
    """Single entry point for 'is the current live position's stop-loss
    breached as of today' — shared by the manual CLI (`app/execution
    __main__.py`) and the daemon's continuous monitoring loop
    (`app/daemon.py`). This is the BACKSTOP for `app.live.exit_monitor`'s
    real-time tick-based check (a process outage, a websocket gap, the
    daily server power-cycle restart, etc.) — see project memory:
    server-power-schedule-blocker.

    Reads the CURRENT trailing stop state (2026-08-10) if any, but — fixed
    2026-08-14 — walks the ratchet day by day (`check_and_apply_trailing_stop`,
    the same engine the backtest replay has always used) rather than
    computing ONE floor from TODAY's high-water and testing every past day
    against it. That single-floor version was a real bug, not a
    simplification: `effective_stop_loss` is monotonic in high-water, and
    high-water only ever ratchets UP over a holding streak, so TODAY's
    floor is always >= whatever was actually in force on any earlier day —
    a LOOSER (higher) threshold applied retroactively, which can only ever
    make the breach test fire MORE easily on past days than it correctly
    should have (never less — this bug cannot cause a MISSED breach, only
    a SPURIOUS one). Confirmed in production: 042660.KS (2026-08-13 entry)
    was force-liquidated citing "breached on 2026-08-04" at a moment its
    real price was at a fresh HIGH — a full day-by-day replay through the
    live low/high series shows it never actually breached its true
    trailing floor on any day it was held. See memory:
    live-trailing-stop-retroactive-breach-2026-08-14 for the full
    reproduction and blast-radius review (both 042660.KS and 000810.KS
    were spuriously liquidated for real paper-account money — see that
    memory for the manual-review decision this still needs).

    Fixed again 2026-08-27, the same defect one resolution finer: walking
    day by day was necessary but not sufficient, because within a single day
    this still ratcheted from that day's HIGH before testing that day's LOW.
    047040.KS (대우건설) was force-liquidated at +11% on a low that occurred
    THIRTY MINUTES BEFORE the high whose floor it supposedly breached (the
    day's low 18,760 at 09:00, the day's high 21,950 at 09:30, floor
    19,729.24, and the lowest low after 09:30 was 19,800 — 70.76 KRW clear
    of the floor). The tick monitor was demonstrably up and correctly silent
    throughout. Full numbers and blast radius:
    `check_and_apply_trailing_stop`'s own docstring.

    This pass therefore fetches 5-minute intraday bars
    (`app.dataflows.yfinance.get_intraday_bars` — yfinance, not KIS: the KIS
    past-date intraday TR is domestic-only and costs 4-6 throttled calls per
    ticker-day) for the days it has not already settled, and hands them to
    the walk, which resolves each day's true ordering and per-day falls back
    to the old conservative rule wherever bars are missing or don't
    reconcile against that day's daily bar. Every closed day the walk clears
    is then recorded in `trailing_stop.record_adjudicated_through`, so a
    later yfinance outage cannot re-litigate an already-correctly-cleared
    day with the coarse rule and re-manufacture the same false liquidation a
    day late. `settings.stop_loss_backstop_intraday_adjudication` is the
    kill switch back to the old rule (over-eager, but never blind).

    Deliberately NOT gated on whether `app.live.exit_monitor` "had coverage"
    for a day: nothing in this system records coverage.
    `trailing_stop_state.updated_at` is a single last-write timestamp, so
    "it was alive at 02:35Z" is no evidence at all about 00:49Z, and a
    websocket gap over exactly the crash minute is the failure this backstop
    exists for. Trading a spurious-exit bug for a blind-spot bug is the one
    trade this code must never make. Intraday bars give the same answer
    continuous ticks would, from a feed whose completeness does not depend
    on our own process being healthy, and they can be reconciled against the
    daily bar — which is why they replace that idea rather than complement
    it.

    The plain (non-trailing) branch below needs none of this and gets none:
    `check_and_apply_stop_loss` tests each day's low against a STATIC pm
    stop that no day's high can move, so there is no high-before-low
    ordering to get wrong — if the low reached the stop, the stop was
    breached whenever that low happened. It also gets no watermark: a real
    static-stop breach on a past day SHOULD keep re-firing until the exit
    actually lands.

    Seeds the day-by-day walk from `entry_price` (never from the live
    `trailing_stop_state.high_water_price`, which is TODAY's highest-ever
    value and reintroducing it as the seed would just reintroduce the same
    bug one level down) — identical to how `app.backtest.harness`
    re-baselines `trailing_high_water` on a fresh position. The entry day's
    HIGH stays in scope (a same-day rally legitimately ratchets the floor
    from the moment of entry — exit_monitor already ratchets from
    entry-day ticks); only the entry day's LOW is excluded, same as
    before (an intraday dip before/at entry isn't a breach of a stop set
    for days after). The live `trailing_stop_state.high_water_price` IS
    still used, but narrowly — folded in as TODAY's own high only (a
    tick-level high that already occurred today really was in force by
    now), via `max()` against today's daily-bar high so a stale/
    under-ratcheted live value after a restart self-heals rather than
    understating today's floor.

    Records the resulting `trades` row under TODAY's date
    (`record_as_of=today`), not the historical breach day — see
    `force_exit_position`'s own docstring for why leaving that as the
    breach day made a real sell invisible/misordered on a
    `trade_date DESC` view."""
    open_stop = await get_open_stop_loss(ticker)
    if open_stop is None:
        return f"No open stop-loss on record for {ticker} (flat, or none was set)."

    trade_date, pm_stop_loss = open_stop
    today = get_clock().today()
    if trade_date >= today:
        return f"No new trading days since the last decision for {ticker} yet."

    daily_lows = await get_daily_low_series(ticker, trade_date, today)
    # Exclude the entry date's own low — `get_daily_low_series` is inclusive
    # of `trade_date`, and an intraday dip on the very day the position was
    # opened isn't a real breach of a stop set for the days AFTER entry.
    # Same filter the backtest replay already applies at harness.py's gap
    # stop-loss check.
    daily_lows = {d: low for d, low in daily_lows.items() if d > trade_date}

    entry_info = await get_open_stop_loss_entry_time(ticker)
    trailing_state = None
    entry_price = None
    if entry_info is not None:
        entry_time, entry_price, _ = entry_info
        trailing_state = await trailing_stop.get_current_state(ticker, entry_time)

    if trailing_state is None:
        outcome = await check_and_apply_stop_loss(
            ticker, daily_lows, pm_stop_loss, reason=reason, record_as_of=today
        )
    else:
        live_high_water, trailing_pct = trailing_state
        daily_highs = await get_daily_high_series(ticker, trade_date, today)
        # Live tick high-water is NO LONGER folded into daily_highs[today]
        # (pre-2026-08-27 it was) — it is passed to the walk separately. Two
        # reasons: a tick-level high the 5-minute bars never show would fail
        # the bar set's reconciliation against the daily bar and knock today
        # back onto the very rule being fixed; and inside the walk it belongs
        # at the END of today's bars, where it can only tighten a later
        # floor, not retroactively tighten an earlier one. The fallback
        # branch still folds it in BEFORE testing today's low, so the
        # stale-live-value self-healing described above is unchanged.
        walk_days = sorted(set(daily_highs) | set(daily_lows) | {today})
        intraday_bars: dict[date, list[IntradayBar]] = {}
        adjudicated_through = await trailing_stop.get_adjudicated_through(
            ticker, entry_time, pm_stop_loss
        )
        if settings.stop_loss_backstop_intraday_adjudication:
            # Days at/below the watermark are never low-tested, so their
            # ordering is irrelevant and fetching bars for them would be
            # pure vendor load; their highs come from the daily bar, which
            # is the conservative (>=) value, so skipping them cannot
            # understate any later floor.
            needed = [d for d in walk_days if adjudicated_through is None or d > adjudicated_through]
            for day in needed[-_MAX_INTRADAY_ADJUDICATION_DAYS:]:
                bars = await get_intraday_bars(ticker, day)
                if bars:
                    intraday_bars[day] = bars
        outcome, _ = await check_and_apply_trailing_stop(
            ticker, daily_highs, daily_lows, pm_stop_loss, entry_price, entry_price, trailing_pct,
            reason=reason, record_as_of=today,
            intraday_bars=intraday_bars or None,
            adjudicated_through=adjudicated_through,
            entry_time=entry_time,
            live_high_water=live_high_water,
            live_high_water_day=today,
        )
        if outcome is None:
            # Only CLOSED sessions whose low was actually in the series we
            # just tested — never today (still open: a 14:00 crash must stay
            # testable for the rest of the day), never the entry day (its low
            # is excluded from the walk entirely, so nothing about it was
            # adjudicated), and never on a breach (see
            # record_adjudicated_through's own docstring).
            closed = [d for d in daily_lows if d < today]
            if closed:
                await trailing_stop.record_adjudicated_through(
                    ticker, entry_time, max(closed), pm_stop_loss
                )

    if outcome is None:
        return f"{ticker}: stop-loss not breached through {today}."
    if outcome.ok:
        return f"{ticker}: stop-loss triggered (breach on {outcome.breach_date}) — position closed."
    if outcome.status == "cooldown":
        return (
            f"{ticker}: STOP-LOSS BREACHED (on {outcome.breach_date}) but the exit is in failure "
            f"cooldown — POSITION STILL OPEN AND UNPROTECTED ({outcome.detail})."
        )
    return (
        f"{ticker}: STOP-LOSS BREACHED (on {outcome.breach_date}) but the exit order FAILED — "
        f"POSITION STILL OPEN AND UNPROTECTED. KIS: {outcome.disposition}"
    )


@dataclass
class TradeEntry:
    """One row from `trades`, for display purposes (e.g. the dashboard) —
    not used by the pipeline itself."""

    ticker: str
    trade_date: date
    side: str
    shares: int
    price: float
    cash_after: float
    shares_after: int
    stop_loss: float | None
    reason: str
    decision_id: int | None = None


async def get_trade_history(ticker: str | None = None, limit: int = 100) -> list[TradeEntry]:
    """Most recent trades, for on-demand reporting — NOT used by the
    pipeline, so unlike every other function in this file, this
    deliberately does not degrade quietly on a DB failure (same
    reasoning as `app/audit/stats.py` and `app/memory/store.get_recent_entries`).

    Currently unused: its only caller, `app/dashboard` (a pre-live-dashboard
    static HTML report generator), was removed 2026-08-26 as unmaintained
    dead code (see that removal's commit) — its bugs (Hold silently mixed
    into a "Buy/Sell only" win rate, backtest rows unscoped, etc.) were
    cheaper to delete than fix, since app.web's live dashboard already
    covers the same ground correctly. Left in place rather than deleted:
    the account_mode/is_backtest scoping below is exactly right for a
    future on-demand trade-history reader, if one gets built.

    Scoped to the CURRENT account_mode and user, same as every other reader
    in this module — a dashboard showing paper and real fills blended
    together (or one person's trades in another's history) would misrepresent
    both. Always excludes is_backtest rows regardless of is_backtest_context()
    — unlike the pipeline-facing readers above, this is dashboard-only and
    a backtest replay's own trades belong in its own report, not mixed into
    "what actually happened" history (same reasoning as
    app.memory.store.get_recent_entries's include_backtest default)."""
    pool = await _ready_pool()
    async with pool.acquire() as conn:
        if ticker is None:
            rows = await conn.fetch(
                "SELECT * FROM trades WHERE account_mode = $1 AND user_id = $2 AND is_backtest = false "
                "ORDER BY trade_date DESC, id DESC LIMIT $3",
                current_account_mode(),
                get_current_user_id(),
                limit,
            )
        else:
            rows = await conn.fetch(
                "SELECT * FROM trades WHERE ticker = $1 AND account_mode = $2 AND user_id = $3 AND is_backtest = false "
                "ORDER BY trade_date DESC, id DESC LIMIT $4",
                ticker,
                current_account_mode(),
                get_current_user_id(),
                limit,
            )

    return [
        TradeEntry(
            ticker=row["ticker"],
            trade_date=row["trade_date"],
            side=row["side"],
            shares=row["shares"],
            price=row["price"],
            cash_after=row["cash_after"],
            shares_after=row["shares_after"],
            stop_loss=row["stop_loss"],
            reason=row["reason"],
            decision_id=row["decision_id"],
        )
        for row in rows
    ]