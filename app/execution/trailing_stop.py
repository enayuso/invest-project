"""Trailing stop-loss — ratchets a profitable position's own stop-loss UP
(never down) as its high-water price rises.

Why this exists: the PM's own `stop_loss` only ever changes when a real
trade is recorded (see `ledger.resolve_stop_loss`/`rebalance_to_target`) —
between scheduled rebalances (days apart), a position that has run up
significantly sits with the SAME stop it was assigned at entry, giving back
however much of that run-up before the next scheduled re-underwrite. This
module adds a second, continuously-updating floor underneath the PM's own
stop, without ever loosening it.

The entire safety argument is one line of math (`effective_stop_loss`
below): `max(pm_stop_loss, trailing_floor)`. Because it's a max(), this can
only ever move the effective exit EARLIER than the PM's plain stop, never
later — there is no configuration of this module's own parameters that
makes a position LESS protected than before it existed.

State model: one row per (ticker, account_mode, user_id), keyed to the
CURRENT open position's entry timestamp (`ledger.get_open_stop_loss_entry_
time`'s own `entry_time` — the latest trade row's `created_at`). A single
UPSERT (see `update_high_water_mark`) both ratchets the high-water price up
within a holding streak and re-baselines it when `entry_time` changes:

- A HOLD that changes nothing writes no new `trades` row (see
  `rebalance_to_target`'s `delta == 0` early return) -> `entry_time` stays
  the same -> the existing high-water mark and its `trailing_pct` (frozen
  at that streak's start, not recomputed every call) both persist and keep
  ratcheting.
- An add-on (BUY while already holding) DOES write a new row -> `entry_time`
  changes -> this re-baselines the high-water mark to the add-on's own
  fill price and recomputes `trailing_pct` fresh. This is a deliberate
  choice, not an oversight: it mirrors the EXISTING precedent that
  `resolve_stop_loss` already re-underwrites the PM's own stop-loss fresh
  on every real trade, including add-ons — a trailing baseline that didn't
  do the same would be inconsistent with how this system already treats
  "the latest trade re-commits the position's risk parameters."
- A forced exit to flat writes a row with `shares_after = 0`, which makes
  `get_open_stop_loss_entry_time` return `None` — nothing reads this
  module's stale row for that ticker again until a NEW trade opens a new
  streak with a different `entry_time`, at which point the UPSERT's own
  entry_time-mismatch branch resets it. No explicit delete-on-exit is
  needed for correctness.
"""

from __future__ import annotations

import asyncio
import sys
from datetime import date, datetime

import asyncpg

from app.execution import _schema_bootstrap, risk_rules
from app.config import current_account_mode
from app.memory.db import get_pool
from app.user_context import get_current_user_id

_SCHEMA = """
CREATE TABLE IF NOT EXISTS trailing_stop_state (
    ticker           TEXT NOT NULL,
    account_mode     TEXT NOT NULL,
    user_id          TEXT NOT NULL REFERENCES users(user_id),
    entry_time       TIMESTAMPTZ NOT NULL,
    high_water_price DOUBLE PRECISION NOT NULL,
    trailing_pct     DOUBLE PRECISION NOT NULL,
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (ticker, account_mode, user_id)
);

-- 2026-08-27 — how far the daily-bar backstop (ledger.check_stops_for_ticker)
-- has already ADJUDICATED this holding streak's daily lows, i.e. the latest
-- CLOSED session whose low was tested and cleared against the floors that
-- were genuinely in force during it. Days at or below this are never
-- low-tested again by that walk (their highs are still ratcheted into the
-- high-water mark, which is mandatory — an unratcheted high understates
-- every later floor).
--
-- Exists because the fix for the same-day high-before-low false breach is
-- only correct while the intraday bars that resolved the ordering are
-- fetchable. Without this column, one yfinance hiccup on any LATER pass
-- reverts an already-correctly-cleared day to the coarse conservative rule
-- and re-manufactures the exact false liquidation the fix removed, one day
-- late — that "the naive fix only postpones it" trap is the whole reason
-- this is a persisted watermark and not an in-process memo. See
-- ledger.check_and_apply_trailing_stop for the 047040.KS incident itself.
--
-- Reset to NULL, not carried, whenever the streak resets (see
-- update_high_water_mark's entry_time-mismatch branch): a re-entry or add-on
-- re-baselines entry_price/trailing_pct, so every earlier day's verdict was
-- computed against parameters that no longer apply and must be re-walked
-- from scratch. Never advanced to today or to any still-open session —
-- marking today "adjudicated" at 09:15 would blind the backstop to a 14:00
-- crash for the rest of the day.
ALTER TABLE trailing_stop_state ADD COLUMN IF NOT EXISTS adjudicated_through DATE;

-- The pm stop_loss `adjudicated_through` was computed under. A day's verdict
-- is immutable within a streak ONLY as long as the PM's own (static) stop
-- doesn't move underneath it — a raised pm stop can turn a previously
-- cleared day into a real breach, and skipping that day would be a MISSED
-- breach, the one failure mode strictly worse than the bug being fixed.
-- Today that cannot happen (a stop only ever changes by writing a new trades
-- row, which changes entry_time, which resets this row), so this column is
-- not fixing a live bug — it turns that argument from a comment somebody has
-- to keep re-verifying into an enforced precondition: get_adjudicated_through
-- returns NULL on any mismatch, and the walk re-tests everything.
ALTER TABLE trailing_stop_state ADD COLUMN IF NOT EXISTS adjudicated_pm_stop DOUBLE PRECISION;
"""

_schema_ready = False
_schema_lock = asyncio.Lock()


async def _ready_pool() -> asyncpg.Pool:
    """Double-checked locking around the first-time DDL, shared via
    app.execution._schema_bootstrap — same pattern as every other
    app.execution module's own `_ready_pool`."""
    return await _schema_bootstrap.ready_pool(sys.modules[__name__], _SCHEMA)


def compute_trailing_pct(atr_pct: float | None) -> float:
    """The trailing distance (fraction of high-water price) for a NEW
    holding streak — ATR-scaled when available, a flat fallback otherwise.
    Never returns None: an un-computed trailing distance would silently
    disable ratcheting instead of degrading to a wider-but-present one
    (same "always leave a number behind, just a more conservative one"
    convention `resolve_stop_loss`'s own ATR-unavailable path uses)."""
    if atr_pct is None or atr_pct <= 0:
        return risk_rules.rule("trailing_stop_fallback_pct")
    return risk_rules.rule("trailing_stop_atr_multiplier") * atr_pct


def effective_stop_loss(
    pm_stop_loss: float, entry_price: float, high_water_price: float, trailing_pct: float
) -> float:
    """max(PM's own stop, trailing floor) — see this module's own docstring
    for why the max() is the whole safety argument. The trailing floor only
    applies once `high_water_price` has risen at least
    `trailing_stop_activation_pct` above `entry_price`; below that, this
    returns `pm_stop_loss` unchanged (see that setting's own config
    comment — there is usually nothing worth protecting yet, and the max()
    would pick the PM's stop anyway)."""
    activation_price = entry_price * (1 + risk_rules.rule("trailing_stop_activation_pct"))
    if high_water_price < activation_price:
        return pm_stop_loss
    trailing_floor = high_water_price * (1 - trailing_pct)
    return max(pm_stop_loss, trailing_floor)


async def update_high_water_mark(
    ticker: str, entry_time: datetime, entry_price: float, current_price: float, atr_pct: float | None
) -> tuple[float, float]:
    """One atomic UPSERT: ratchets this ticker's high-water price up to
    `current_price` if it's a new high WITHIN the current holding streak
    (`entry_time` matches the stored row), or re-baselines to a fresh
    streak if it doesn't (or no row exists yet) — see this module's own
    docstring for why an `entry_time` mismatch is the correct reset
    signal. Returns (high_water_price, trailing_pct) as stored after this
    call, for the caller to feed into `effective_stop_loss`.

    `trailing_pct` is deliberately NOT recomputed every call within the
    same streak — only when the streak resets — so a position's trailing
    distance stays stable for its own analysis, the same "computed once at
    the decision that opened this position" treatment the PM's own
    stop_loss already gets, rather than drifting as ATR moves intraday."""
    pool = await _ready_pool()
    candidate_high = max(entry_price, current_price)
    trailing_pct = compute_trailing_pct(atr_pct)
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO trailing_stop_state
                (ticker, account_mode, user_id, entry_time, high_water_price, trailing_pct)
            VALUES ($1, $2, $3, $4, $5, $6)
            ON CONFLICT (ticker, account_mode, user_id) DO UPDATE SET
                entry_time = EXCLUDED.entry_time,
                high_water_price = CASE
                    WHEN trailing_stop_state.entry_time = EXCLUDED.entry_time
                    THEN GREATEST(trailing_stop_state.high_water_price, EXCLUDED.high_water_price)
                    ELSE EXCLUDED.high_water_price
                END,
                trailing_pct = CASE
                    WHEN trailing_stop_state.entry_time = EXCLUDED.entry_time
                    THEN trailing_stop_state.trailing_pct
                    ELSE EXCLUDED.trailing_pct
                END,
                adjudicated_through = CASE
                    WHEN trailing_stop_state.entry_time = EXCLUDED.entry_time
                    THEN trailing_stop_state.adjudicated_through
                    ELSE NULL
                END,
                adjudicated_pm_stop = CASE
                    WHEN trailing_stop_state.entry_time = EXCLUDED.entry_time
                    THEN trailing_stop_state.adjudicated_pm_stop
                    ELSE NULL
                END,
                updated_at = now()
            RETURNING high_water_price, trailing_pct
            """,
            ticker, current_account_mode(), get_current_user_id(), entry_time, candidate_high, trailing_pct,
        )
    return row["high_water_price"], row["trailing_pct"]


async def get_current_state(ticker: str, entry_time: datetime) -> tuple[float, float] | None:
    """Read-only counterpart to `update_high_water_mark`, for a caller that
    only wants to KNOW the current effective stop without also feeding it a
    fresh price — the hourly daily-bar backstop (`ledger.check_stops_for_
    ticker`) uses this rather than the update path: it only has a day's
    LOW, not a continuous stream of prices to ratchet a high-water mark
    from, and ratcheting off a single daily low would be actively wrong
    (a low is never a new HIGH). `None` if no state exists yet for this
    exact `entry_time` — the caller falls back to the PM's plain stop_loss
    in that case (see effective_stop_loss's own docstring: below
    activation, or with no trailing state at all, is the same outcome)."""
    pool = await _ready_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT high_water_price, trailing_pct FROM trailing_stop_state "
            "WHERE ticker = $1 AND account_mode = $2 AND user_id = $3 AND entry_time = $4",
            ticker, current_account_mode(), get_current_user_id(), entry_time,
        )
    if row is None:
        return None
    return row["high_water_price"], row["trailing_pct"]


async def get_adjudicated_through(ticker: str, entry_time: datetime, pm_stop_loss: float) -> date | None:
    """The watermark described in this module's `adjudicated_through` schema
    comment, for THIS streak and THIS pm stop only — `None` if no row exists,
    the row belongs to a different streak, nothing has been adjudicated yet,
    or the pm stop has moved since (in which case every earlier verdict is
    stale and the caller must re-test everything, the fail-safe direction).

    Deliberately a separate read rather than three more return values on
    `get_current_state`: that function is on app.live.exit_monitor's 5-second
    path, which has no use for a daily-bar watermark, and widening its tuple
    would churn every one of its call sites and tests for nothing. One extra
    query per ticker per HOUR is not a cost worth optimizing."""
    pool = await _ready_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT adjudicated_through, adjudicated_pm_stop FROM trailing_stop_state "
            "WHERE ticker = $1 AND account_mode = $2 AND user_id = $3 AND entry_time = $4",
            ticker, current_account_mode(), get_current_user_id(), entry_time,
        )
    if row is None or row["adjudicated_through"] is None:
        return None
    if row["adjudicated_pm_stop"] is None or row["adjudicated_pm_stop"] != pm_stop_loss:
        return None
    return row["adjudicated_through"]


async def record_adjudicated_through(
    ticker: str, entry_time: datetime, through: date, pm_stop_loss: float
) -> None:
    """Advances this streak's watermark to `through`. Monotonic — the WHERE
    clause refuses to move it BACKWARD, so a pass that happened to walk a
    shorter window (a fresh `get_open_stop_loss` trade_date, a partial daily
    series) can never un-adjudicate days that were already settled.

    Writes nothing when no row matches this exact `entry_time`: either the
    streak reset mid-pass (an add-on/re-entry landed between the walk and
    this write, so the walk's verdicts are already void) or exit_monitor has
    never established trailing state at all. Both are correctly no-ops, not
    errors — a missing watermark only ever costs a redundant re-walk.

    Only ever called after a walk that found NO breach: a breach that exited
    leaves the position flat (nothing to watermark) and a breach whose exit
    FAILED must keep re-firing on every pass until it lands, which is exactly
    what NOT writing a watermark preserves."""
    pool = await _ready_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE trailing_stop_state SET adjudicated_through = $5, adjudicated_pm_stop = $6 "
            "WHERE ticker = $1 AND account_mode = $2 AND user_id = $3 AND entry_time = $4 "
            "AND (adjudicated_through IS NULL OR adjudicated_through < $5)",
            ticker, current_account_mode(), get_current_user_id(), entry_time, through, pm_stop_loss,
        )
