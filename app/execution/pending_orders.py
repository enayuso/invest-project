"""Persisted queue for AI-proposed orders awaiting explicit human approval —
SEMI_AUTO mode's every proposal, plus any FULL_AUTO order requires_approval
holds (see app.config.ExecutionMode, kis_bridge.mirror_trade_to_kis).

Before this existed, a held order was just a disposition string logged and
discarded — the ledger's own paper position had already been recorded
regardless, so the ONLY record of "this KIS order never actually went out"
was a log line nobody was watching. Persisting it here means a human can
still see it and decide, and the resolution outcome (submitted/blocked/
rejected) is recorded rather than lost.

Approval re-runs kis_bridge.mirror_trade_to_kis's FULL current gate chain
(cash, VI, circuit-breaker, caps, ...) at approval time rather than trusting
whatever was true when the proposal was created — the market can move in
the time a human takes to look at a pending order.

Same asyncpg + double-checked-locking schema bootstrap as
app.execution.safety_switch/portfolio_risk.
대기 주문 영속화 — 예전엔 승인 대기 주문이 로그 한 줄로 사라졌음, 이제 사람이 확인/승인/거절 가능.

Paper/real account separation (2026-08-04 — see project memory:
user-account-mode-db-separation-design): every row is scoped to the
CURRENT account_mode. Before this column, a SEMI_AUTO proposal created
while paper-testing and left unresolved would still be sitting in
`pending_orders` after switching `kis_account_mode` to 'real' — and
`approve_pending_order` submits through WHATEVER broker is currently
connected, so approving that stale paper-era proposal after the switch
would send a real order built from paper-account context (paper balance,
paper position sizing) straight to real money. Scoping every read (list/
get) to the current account_mode means a cross-mode order is invisible
and `get_pending_order` can't find it at all, so `approve_pending_order`/
`reject_pending_order` raise "not found" instead of ever touching it.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone

import asyncpg

from app.config import current_account_mode, settings
from app.execution import _schema_bootstrap
from app.live.alerts import Alert, record_alert
from app.memory.db import get_pool
from app.user_context import get_current_user_id
from app.clock import get_clock

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS pending_orders (
    id           SERIAL PRIMARY KEY,
    ticker       TEXT NOT NULL,
    trade_date   DATE NOT NULL,
    side         TEXT NOT NULL,
    shares       INTEGER NOT NULL,
    price        DOUBLE PRECISION NOT NULL,
    reason       TEXT NOT NULL,
    status       TEXT NOT NULL DEFAULT 'pending',
    resolution   TEXT,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    resolved_at  TIMESTAMPTZ
);

-- Paper/real account separation (2026-08-04) — see module docstring.
-- No composite-key migration needed here (unlike trades/portfolio_
-- drawdown_state/daily_loss_state): `id` is already the sole PK, so this
-- is a plain filter column, not part of a uniqueness constraint.
-- Per-user separation (multi-user migration Phase 4, plan §2.2). Without it
-- the approval queue is shared: user A sees user B's proposed orders and can
-- approve or reject them, and approving submits a real order into B's account
-- using B's credentials. Backfilled to 'default' like account_mode was to
-- 'paper'.
-- 승인 대기열이 공유되면 A가 B의 주문을 승인/거부할 수 있고, 승인 시 B의 계좌로
-- 실제 주문이 나감.
ALTER TABLE pending_orders ADD COLUMN IF NOT EXISTS user_id TEXT NOT NULL DEFAULT 'default';
ALTER TABLE pending_orders ADD COLUMN IF NOT EXISTS account_mode TEXT NOT NULL DEFAULT 'paper'
    CHECK (account_mode IN ('paper', 'real'));

-- Multi-instance seam (agent-architecture-roadmap stage 4) — same pattern
-- as app.execution.ledger's `trades.user_id` (see that module's own
-- comment for the full rationale). This table was scoped to account_mode
-- alone until now; ties every row to a registered person the same way, so
-- a future multi-user deployment can't have one person's SEMI_AUTO
-- proposal show up in — or be approvable from — another person's
-- dashboard. `id` stays the sole PK (no composite-key migration needed,
-- same as the account_mode column above).
ALTER TABLE pending_orders ADD COLUMN IF NOT EXISTS user_id TEXT NOT NULL DEFAULT 'default';
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'pending_orders_user_id_fkey') THEN
        ALTER TABLE pending_orders ADD CONSTRAINT pending_orders_user_id_fkey
            FOREIGN KEY (user_id) REFERENCES users(user_id);
    END IF;
END $$;

-- Links this proposal back to the `trades` row `rebalance_to_target`
-- speculatively recorded for it (see app.execution.ledger.record_trade's
-- own docstring on why the ledger writes BEFORE execution_mode routing
-- decides SEMI_AUTO). NULL means no such row exists — the only legitimate
-- way that happens is a proposal created OUTSIDE rebalance_to_target
-- (currently just scripts/seed_test_pending_order.py, a dev utility that
-- previews the approval popup without going through the AI pipeline/
-- ledger at all). `_reject_or_expire` uses this to tell the two cases
-- apart: a linked proposal has a real speculative position to undo on
-- reject/expire/approve_failed; an unlinked one does not, and reversing it
-- anyway means acting on a phantom trade the ledger never actually
-- recorded. Confirmed live (2026-08-06): a seeded test BUY pending order
-- for 005930.KS (no ledger entry of its own) was rejected, and
-- reverse_pending_trade's clamp (`min(shares, position.shares)`) silently
-- sold off 5 REAL shares from an unrelated, legitimately-held position
-- instead of correctly no-opping — the clamp has no way to tell "shares
-- this reversal is entitled to undo" from "shares that happen to be
-- sitting in the ledger right now" once there's no real trade row backing
-- the reversal at all. See project memory:
-- position-mismatch-unlinked-pending-order-fixed (or equivalent).
-- 대기주문이 record_trade 없이 생성된 경우(seed 스크립트 등) reject 시 실제 보유
-- 포지션을 잘못 청산하지 않도록, 짝이 되는 trades 행 id를 명시적으로 연결.
ALTER TABLE pending_orders ADD COLUMN IF NOT EXISTS trade_id INTEGER;
"""

_schema_ready = False
_schema_lock = asyncio.Lock()

# Serializes approve/reject against the SAME pending order (and, for
# simplicity at this system's single-user scale, against every other one
# too — same "one coarse lock beats a per-row one" tradeoff kis_bridge's
# own _order_lock already makes). Without this, a double-click (or an
# approve/reject race) could both pass the "is it still pending?" check
# before either writes back: approve would submit a REAL order to KIS at
# the same time reject reverses the ledger's speculative entry, leaving a
# live KIS order with no matching ledger record. _resolve()'s own
# `WHERE status = 'pending'` update is atomic, but only for the DB write —
# it can't undo a side effect (a real order already sent, a ledger entry
# already reversed) that happened before it.
# 승인/거절 동시 클릭 시 이중집행(진짜 주문 전송 + 장부 반전이 동시에)을 막는 락.
_resolution_lock = asyncio.Lock()


async def _ready_pool() -> asyncpg.Pool:
    """Double-checked locking around the first-time DDL — shared with the
    other app.execution modules via _schema_bootstrap."""
    return await _schema_bootstrap.ready_pool(sys.modules[__name__], _SCHEMA)


@dataclass
class PendingOrder:
    id: int
    ticker: str
    trade_date: date
    side: str
    shares: int
    price: float
    reason: str
    status: str
    resolution: str | None
    created_at: datetime
    resolved_at: datetime | None
    # None means no `trades` row was ever recorded for this proposal — see
    # the schema comment on `trade_id` above for why that matters to
    # `_reject_or_expire`. `.get` (not `row["trade_id"]`) so a fake test
    # connection's row dict need not always carry the key.
    trade_id: int | None = None


def _from_row(row: asyncpg.Record) -> PendingOrder:
    return PendingOrder(
        id=row["id"], ticker=row["ticker"], trade_date=row["trade_date"], side=row["side"],
        shares=row["shares"], price=row["price"], reason=row["reason"], status=row["status"],
        resolution=row["resolution"], created_at=row["created_at"], resolved_at=row["resolved_at"],
        trade_id=row["trade_id"] if "trade_id" in row.keys() else None,
    )


async def count_pending() -> int:
    """How many rows are currently status='pending' for the CURRENT
    account_mode + user_id — what `settings.pending_order_max_count`
    caps. Callers check this BEFORE `create_pending_order` (see
    kis_bridge.requires_approval's call site) rather than this function
    enforcing the cap itself, so a full queue produces an ordinary
    disposition string (`BLOCKED: ...`) through the same path every other
    gate already uses, instead of an exception this module's callers don't
    otherwise expect. Does NOT sweep first — an approval-tab view of the
    queue's current size, not a "how many WOULD remain after cleanup"
    estimate; the TTL sweep still runs on the next `list_pending_orders`/
    `get_pending_order` touch regardless."""
    pool = await _ready_pool()
    async with pool.acquire() as conn:
        return await conn.fetchval(
            "SELECT count(*) FROM pending_orders WHERE status = 'pending' AND account_mode = $1 AND user_id = $2",
            current_account_mode(), get_current_user_id(),
        )


async def create_pending_order(
    ticker: str, trade_date: date, side: str, shares: int, price: float, reason: str,
    trade_id: int | None = None,
) -> int:
    """Records a proposal awaiting approval. Returns its id. Stamped with
    the CURRENT account_mode (2026-08-04) and CURRENT user_id (2026-08-06,
    multi-instance seam — see module docstring) — see module docstring.

    `trade_id`, if given, is the id of the `trades` row `rebalance_to_target`
    speculatively recorded for this same proposal (its ONLY real caller
    today) — see the `trade_id` column's own schema comment for why
    `_reject_or_expire` needs this to tell a normal SEMI_AUTO proposal apart
    from one created directly against this table with no matching ledger
    entry (e.g. scripts/seed_test_pending_order.py). Left `None` by any
    caller that doesn't have one — reject/expire/approve_failed on such a
    proposal then correctly no-ops the ledger side instead of guessing."""
    pool = await _ready_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO pending_orders (ticker, trade_date, side, shares, price, reason, account_mode, user_id, trade_id)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            RETURNING id
            """,
            ticker, trade_date, side, shares, price, reason, current_account_mode(), get_current_user_id(), trade_id,
        )
    logger.info("pending order #%s created: %s %s x%d @ %.2f — %s", row["id"], side, ticker, shares, price, reason)
    return row["id"]


async def _sweep_expired() -> None:
    """Expires (status='expired') every still-pending row older than
    `settings.pending_order_ttl_hours`, reversing each one's ledger entry
    exactly like `reject_pending_order` does (via `_reject_or_expire`) —
    an order nobody ever approved/rejected must not sit forever, both
    because the approval-tab list would grow unbounded and because the
    frozen price/shares/stop-loss it carries gets further from current
    market reality (and the internal ledger's assumed position further from
    the real account) the longer it's left unresolved.

    Lazy sweep-on-touch, not a background job: called at the top of
    `list_pending_orders`/`get_pending_order` (the two read paths a human or
    the approve/reject flow actually uses), same "clean up whatever's stale
    on the next real touch" convention as
    `app.orchestrator.pipeline._store_cached_reports`. Real wall-clock time,
    not `app.clock.get_clock()` — pending orders are a live-only concept
    (backtests never create one; `mirror_trade_to_kis`'s today-only gate
    already excludes historical replay), so there's no fake-clock seam to
    respect here.

    Scoped to the CURRENT account_mode (2026-08-04) — sweeping the OTHER
    mode's stale orders here would call `reverse_pending_trade` (which
    stamps its reversal with whatever mode is running NOW) against a
    speculative entry that was originally recorded under the OTHER mode,
    corrupting that mode's own ledger bookkeeping the next time it runs.
    Also scoped to the CURRENT user_id (2026-08-06, multi-instance seam) —
    same reasoning, one axis over."""
    pool = await _ready_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT * FROM pending_orders WHERE status = 'pending' AND account_mode = $1 AND user_id = $2",
            current_account_mode(), get_current_user_id(),
        )

    cutoff = datetime.now(timezone.utc) - timedelta(hours=settings.pending_order_ttl_hours)
    for row in rows:
        if row["created_at"] < cutoff:
            order = _from_row(row)
            resolution = f"expired: no decision within {settings.pending_order_ttl_hours}h"
            await _reject_or_expire(order.id, order, "expired", resolution)


async def list_pending_orders() -> list[PendingOrder]:
    """Every order still awaiting a decision for the CURRENT account_mode,
    oldest first — a cross-mode order is invisible here (2026-08-04, see
    module docstring), not just filtered out of the display.

    `_sweep_expired()` here is lock-guarded (2026-08-06) — this is the only
    caller of it that ISN'T already inside `_resolution_lock`
    (approve/reject_pending_order acquire the lock before their own
    `get_pending_order` call triggers a sweep). Without this, a dashboard
    poll landing here at the same moment a human clicks "approve" on an
    order that's *just* past its TTL could interleave: this sweep reverses
    the ledger entry and marks the row 'expired' while approve, having
    already read it as 'pending' a moment earlier, goes on to submit a REAL
    order to KIS and then silently no-ops its own `UPDATE ... WHERE
    status = 'pending'` — leaving a real KIS order with a ledger that says
    it was reversed. The window is narrow (both have to land within the
    same TTL-boundary instant) but the failure mode (a live order with no
    matching ledger record) is exactly the class of bug this system's
    reconciliation exists to catch, so it doesn't need to be likely to be
    worth closing. Also scoped to the CURRENT user_id (2026-08-06,
    multi-instance seam) — same "cross-scope row is invisible" treatment as
    account_mode above."""
    async with _resolution_lock:
        await _sweep_expired()
    pool = await _ready_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT * FROM pending_orders WHERE status = 'pending' AND account_mode = $1 AND user_id = $2 "
            "ORDER BY created_at",
            current_account_mode(), get_current_user_id(),
        )
    return [_from_row(r) for r in rows]


async def list_resolved_orders(limit: int = 50) -> list[PendingOrder]:
    """Most-recently-resolved orders (approved/approve_failed/rejected) for
    the CURRENT account_mode, newest first — the approve/reject history the
    dashboard's pending-orders tab doesn't otherwise surface, since
    list_pending_orders only returns status='pending' rows. Nothing is ever
    deleted from this table; a resolved order is just a status change on
    the same row (see _resolve), so this is simply the other half of that
    same table. Also scoped to the CURRENT user_id (2026-08-06,
    multi-instance seam) — same reasoning as account_mode above."""
    pool = await _ready_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT * FROM pending_orders WHERE status != 'pending' AND account_mode = $1 AND user_id = $2 "
            "ORDER BY resolved_at DESC LIMIT $3",
            current_account_mode(),
            get_current_user_id(),
            limit,
        )
    return [_from_row(r) for r in rows]


async def get_pending_order(order_id: int) -> PendingOrder | None:
    """`None` if `order_id` doesn't exist OR belongs to the OTHER
    account_mode (2026-08-04, see module docstring) OR the OTHER user_id
    (2026-08-06, multi-instance seam) — the same "not found" result either
    way, so a stale cross-mode/cross-user order can never be approved/
    rejected through this lookup, not just hidden from the list views.

    Its own `_sweep_expired()` call is NOT lock-guarded here, unlike
    `list_pending_orders`'s — this function's only two callers today
    (`approve_pending_order`/`reject_pending_order`) already hold
    `_resolution_lock` before calling in, and `asyncio.Lock` isn't
    reentrant, so acquiring it again here would deadlock those callers. A
    future standalone caller (outside that lock) MUST wrap its own call to
    this function in `async with _resolution_lock:` itself — see
    `list_pending_orders`'s own docstring for the race this closes."""
    await _sweep_expired()
    pool = await _ready_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT * FROM pending_orders WHERE id = $1 AND account_mode = $2 AND user_id = $3",
            order_id, current_account_mode(), get_current_user_id(),
        )
    return _from_row(row) if row else None


async def _resolve(order_id: int, status: str, resolution: str) -> None:
    pool = await _ready_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            UPDATE pending_orders SET status = $2, resolution = $3, resolved_at = now()
            WHERE id = $1 AND status = 'pending' AND user_id = $4
            """,
            order_id, status, resolution, get_current_user_id(),
        )


async def approve_pending_order(order_id: int, *, broker=None) -> str:
    """Re-runs the full current gate chain (not the state from when this was
    created — the market can move while a human is deciding) via
    mirror_trade_to_kis(approved=True), then records the outcome. Raises
    ValueError if `order_id` isn't a still-pending order (already resolved,
    or never existed) — the caller decides how to surface that, same
    convention as risk_rules.save's ValueError-on-bad-input."""
    from app.execution.kis_bridge import mirror_trade_to_kis  # local import: avoid a circular import at module load

    async with _resolution_lock:
        order = await get_pending_order(order_id)
        if order is None or order.status != "pending":
            raise ValueError(f"pending order {order_id} not found or already resolved")
        disposition = await mirror_trade_to_kis(
            order.ticker, order.trade_date, order.side, order.shares, order.price, approved=True, broker=broker,
            trade_id=order.trade_id,
        )
        if disposition.startswith(("submitted", "dry-run")):
            await _resolve(order_id, "approved", disposition)
        else:
            # Approval was attempted but the mirror itself got blocked by a
            # gate re-checked at approval time (VI halt, insufficient cash,
            # circuit-breaker unknown, ...) — the trade never actually
            # reached KIS, so the ledger's speculative entry from when this
            # order was first proposed (rebalance_to_target writes it
            # immediately, before execution_mode is even checked) must be
            # undone too, exactly like a reject/expire. Real incident this
            # fixes: an order blocked here (this used to just mark
            # approve_failed and stop) left a phantom position in the
            # ledger that a LATER cycle then "sold" based on shares that
            # were never actually bought.
            await _reject_or_expire(order_id, order, "approve_failed", disposition)
        return disposition


async def _reject_or_expire(order_id: int, order: PendingOrder, status: str, resolution: str) -> None:
    """Shared by `reject_pending_order`, the TTL sweep (`_sweep_expired`),
    and `approve_pending_order`'s own failure branch — all three need the
    same "undo the ledger's speculative entry, then mark this row
    resolved" pair, since `rebalance_to_target` already wrote a `trades`
    row for this order the moment it was proposed (before `execution_mode`
    was even checked). Ledger reversal FIRST, status flip SECOND: if the
    process dies in between, a `pending` row with an already-reversed
    ledger entry is a safer failure mode than a `rejected`/`expired`/
    `approve_failed` row whose ledger was never actually fixed — the
    latter is undetectable without cross-referencing `trades` directly.

    Skips the ledger reversal entirely when `order.trade_id is None` —
    there is no speculative `trades` row to undo (see the `trade_id`
    column's own schema comment). Calling `reverse_pending_trade` anyway
    would fall back to clamping against whatever the ledger's CURRENT
    position happens to be, which may be entirely unrelated, real, shares
    (confirmed live 2026-08-06: rejecting a dev-seeded test pending order
    for 005930.KS — created directly via `create_pending_order`, never
    paired with a `record_trade` call — silently sold off 5 real,
    legitimately-held shares from an unrelated earlier trade instead of
    correctly doing nothing). A CRITICAL alert is raised instead, since a
    proposal existing here with no ledger backing at all is itself worth a
    human's attention even though there's nothing to reverse."""
    if order.trade_id is None:
        logger.warning(
            "pending order #%s (%s %s x%d) has no linked trades row — skipping ledger reversal "
            "(nothing to undo); resolving as %s only. This is expected for the dev seed script "
            "(scripts/seed_test_pending_order.py), a red flag for anything else.",
            order_id, order.side, order.ticker, order.shares, status,
        )
        await record_alert(
            Alert(
                ticker=order.ticker,
                kind="pending_order_unlinked",
                message=(
                    f"Pending order #{order_id} ({order.side} {order.ticker} x{order.shares}) resolved "
                    f"as {status} ({resolution}) with no linked ledger entry — nothing was reversed. "
                    "Expected for a dev-seeded test proposal; unexpected for anything else."
                ),
                triggered_at=get_clock().now(),
            )
        )
    else:
        from app.execution.ledger import reverse_pending_trade  # local import: ledger imports this module (circular)

        await reverse_pending_trade(order.ticker, order.side, order.shares, order.price, reason=resolution)
    await _resolve(order_id, status, resolution)


async def reject_pending_order(order_id: int, reason: str = "") -> None:
    async with _resolution_lock:
        order = await get_pending_order(order_id)
        if order is None or order.status != "pending":
            raise ValueError(f"pending order {order_id} not found or already resolved")
        await _reject_or_expire(order_id, order, "rejected", reason or "rejected by user")
