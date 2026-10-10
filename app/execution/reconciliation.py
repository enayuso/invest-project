# app/execution/reconciliation.py
# Detects (never corrects) a FULL_AUTO trade that was recorded on the local
# ledger but never actually reached KIS — the gap between
# app.execution.ledger.rebalance_to_target's record_trade() (line ~650) and
# mirror_trade_to_kis() (line ~693) call. A process kill in that gap leaves
# no trace the existing _mirror_never_reached_kis reversal can catch, since
# that reversal only runs if the process survives to see mirror_trade_to_kis's
# response. See project memory: force-kill-mid-trade-ledger-drift-risk.
#
# Detect-only by design: right after a crash, "KIS never got the order" and
# "KIS got it but the response just hasn't come back yet" are indistinguishable
# from here, so auto-reversing risks undoing a trade that actually went
# through. A human should reconcile against the real KIS balance before
# touching the ledger.
# 강제종료로 인한 원장/KIS 불일치 탐지 전용 — 자동수정 없음, 로그만.

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Literal

from app.config import current_account_mode
from app.domain.models import BrokerExecution
from app.execution.kis_bridge import route_ticker
from app.memory.db import get_pool
from app.user_context import get_current_user_id

logger = logging.getLogger(__name__)

Severity = Literal["CRITICAL", "WARNING"]

# Named so app.web.state's periodic reconciliation pass can call
# fetch_recent_local_trades directly (to skip the KIS execution-history
# fetch entirely when there's nothing local to check — see that module's
# own comment) while still sharing find_unmirrored_trades' original tuning:
# 48h catches a crash while it's still fresh, 2min grace_period skips a
# trade whose mirror_trade_to_kis call may still be in flight. Values
# unchanged from find_unmirrored_trades' own prior inline defaults.
DEFAULT_LOOKBACK = timedelta(hours=48)
DEFAULT_GRACE_PERIOD = timedelta(minutes=2)


@dataclass
class LocalTrade:
    id: int
    ticker: str
    trade_date: date
    side: str
    shares: int
    created_at: datetime
    reason: str


@dataclass
class TradeDiscrepancy:
    trade_id: int
    ticker: str
    trade_date: date
    side: str
    shares: int
    created_at: datetime
    severity: Severity
    note: str


# 2026-08-13: a self-audit found two real trades (000810.KS, 006340.KS)
# whose orders were accepted by KIS on one date but didn't actually fill
# until 5-8 days later (mirror_trade_to_kis's market order apparently
# queued rather than filling immediately — the "why" is still unexplained,
# but the fact is confirmed against raw KIS execution history). The
# exact-date match below flagged both CRITICAL — "no matching KIS
# execution found" — when the real executions existed all along, just
# under a later order_date. 14 days covers the worst delay actually
# observed with real margin; a fill that takes longer than that is rare
# enough that treating it as a genuine discrepancy (worth a human
# double-checking) rather than silently absorbing it is the safer default.
_MAX_FILL_DELAY_DAYS = 14


def _order_date_as_date(order_date: str):
    return date(int(order_date[:4]), int(order_date[4:6]), int(order_date[6:]))


def match_trades_against_kis(
    local_trades: list[LocalTrade],
    domestic_executions: list[BrokerExecution],
    overseas_executions: list[BrokerExecution],
) -> list[TradeDiscrepancy]:
    """Pure matching, no I/O — greedily pairs each local trade (oldest first)
    with the closest-quantity unconsumed KIS execution sharing its
    (code, side), filled on or up to `_MAX_FILL_DELAY_DAYS` AFTER the local
    trade's own recorded date (never before — a fill can't precede the order
    that produced it). No order_no to join on: the crash this detects
    happens before KIS ever returns one, so this is a heuristic match, not an
    exact key lookup. Among same-quantity candidates, the one closest in
    date wins — same-day still takes priority over a later date whenever
    both exist, so a genuinely same-day fill is never mismatched against a
    coincidentally same-quantity later trade in an active ticker.

    A local trade with no candidate at all is the crash scenario itself
    (CRITICAL). A matched-but-partially-filled candidate is normally
    WARNING — KIS accepted the order but didn't fill it in full — EXCEPT for
    a stop-loss forced exit (`reason` starts with "stop-loss triggered:"),
    which is always CRITICAL regardless of fill amount: the ledger already
    recorded the position as flat the moment the exit was submitted (see
    ledger.check_and_apply_stop_loss), so any unfilled/partially-filled
    remainder means the real KIS position may still be open and NO LONGER
    monitored for a stop breach (the ledger's open-stop row was consumed).
    Overseas stop-loss exits are always limit orders (KIS has no true
    immediate market order for overseas equities — see kis_bridge.py's
    mirror_trade_to_kis docstring), so an adverse gap can leave this order
    sitting unfilled indefinitely; this is the detection half of that risk,
    not a fix for it (still no auto-correction, same as every other
    discrepancy here). Fully filled matches aren't discrepancies and aren't
    returned."""
    remaining = list(domestic_executions) + list(overseas_executions)
    discrepancies: list[TradeDiscrepancy] = []

    for trade in sorted(local_trades, key=lambda t: t.created_at):
        is_stop_loss = trade.reason.startswith("stop-loss triggered:")
        krx_code, us_symbol = route_ticker(trade.ticker)
        code = krx_code or us_symbol
        candidates = [
            e for e in remaining
            if e.code == code and e.side == trade.side
            and trade.trade_date <= _order_date_as_date(e.order_date) <= trade.trade_date + timedelta(days=_MAX_FILL_DELAY_DAYS)
        ]
        if not candidates:
            note = (
                f"no matching KIS execution found for this ticker/side within "
                f"{_MAX_FILL_DELAY_DAYS} days of the recorded date — possible crash before the KIS mirror "
                "completed, or the order was never actually filled"
            )
            if is_stop_loss:
                note += " — this was a stop-loss exit: the real position may still be open and UNPROTECTED"
            discrepancies.append(TradeDiscrepancy(
                trade_id=trade.id, ticker=trade.ticker, trade_date=trade.trade_date,
                side=trade.side, shares=trade.shares, created_at=trade.created_at,
                severity="CRITICAL",
                note=note,
            ))
            continue

        best = min(
            candidates,
            key=lambda e: (abs(e.filled_qty - trade.shares), abs((_order_date_as_date(e.order_date) - trade.trade_date).days)),
        )
        remaining.remove(best)
        fill_delay_days = (_order_date_as_date(best.order_date) - trade.trade_date).days
        delay_note = f" (filled {fill_delay_days} day(s) after the recorded order date)" if fill_delay_days > 0 else ""
        if best.filled_qty < trade.shares:
            if is_stop_loss:
                discrepancies.append(TradeDiscrepancy(
                    trade_id=trade.id, ticker=trade.ticker, trade_date=trade.trade_date,
                    side=trade.side, shares=trade.shares, created_at=trade.created_at,
                    severity="CRITICAL",
                    note=(
                        f"stop-loss exit only filled {best.filled_qty}/{trade.shares} shares{delay_note} — the ledger "
                        "already marked this position flat and stopped monitoring it, but the real KIS "
                        "position may still hold the unfilled remainder, UNPROTECTED"
                    ),
                ))
            else:
                discrepancies.append(TradeDiscrepancy(
                    trade_id=trade.id, ticker=trade.ticker, trade_date=trade.trade_date,
                    side=trade.side, shares=trade.shares, created_at=trade.created_at,
                    severity="WARNING",
                    note=f"KIS filled only {best.filled_qty}/{trade.shares} shares{delay_note}",
                ))

    return discrepancies


async def fetch_recent_local_trades(lookback: timedelta, grace_period: timedelta) -> list[LocalTrade]:
    """Recent trades table rows eligible for reconciliation — excludes
    reversal rows (already-corrected phantom entries, see
    ledger.reverse_pending_trade), anything within `grace_period` of now
    (its mirror_trade_to_kis call may still be in flight), and any row whose
    `mirror_disposition` already says it was never meant to reach KIS
    (MANUAL mode, backtest replay, dry-run) or is still waiting on a human
    (SEMI_AUTO pending approval) — those aren't the "possible crash before
    the KIS mirror completed" this module exists to catch, and flagging them
    CRITICAL was a standing false-positive (see project memory:
    execution-audit-2026-08-06-fixes). `mirror_disposition IS NULL` is kept
    IN (not excluded): that's either a pre-migration row, or the narrow
    window between record_trade's INSERT and the follow-up
    _record_mirror_disposition UPDATE — a crash landing exactly there is
    itself part of the scenario this module is meant to catch, so staying
    conservative here is correct, not an oversight. Degrades to [] on any DB
    failure, same fail-open convention as app.execution.ledger."""
    try:
        pool = await get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, ticker, trade_date, side, shares, created_at, reason FROM trades
                WHERE account_mode = $1
                  AND user_id = $2
                  AND reason NOT LIKE 'reversal:%'
                  AND (mirror_disposition IS NULL OR mirror_disposition LIKE 'submitted:%')
                  AND created_at BETWEEN now() - $3::interval AND now() - $4::interval
                ORDER BY created_at ASC
                """,
                current_account_mode(),
                get_current_user_id(),
                lookback,
                grace_period,
            )
    except Exception as exc:  # noqa: BLE001 - same broadened fail-open policy as app.execution.ledger
        logger.warning("reconciliation: reading recent trades failed, skipping this pass: %s", exc)
        return []

    return [
        LocalTrade(
            id=row["id"], ticker=row["ticker"], trade_date=row["trade_date"],
            side=row["side"], shares=row["shares"], created_at=row["created_at"],
            reason=row["reason"],
        )
        for row in rows
    ]


async def find_unmirrored_trades(
    domestic_executions: list[BrokerExecution],
    overseas_executions: list[BrokerExecution],
    lookback: timedelta = DEFAULT_LOOKBACK,
    grace_period: timedelta = DEFAULT_GRACE_PERIOD,
) -> list[TradeDiscrepancy]:
    """Entry point: recent local trades vs the given KIS execution history
    (caller fetches these — typically app.web.state.recent_executions(),
    already cached/best-effort per market — so this module stays broker-
    agnostic and independently testable)."""
    local_trades = await fetch_recent_local_trades(lookback, grace_period)
    if not local_trades:
        return []
    return match_trades_against_kis(local_trades, domestic_executions, overseas_executions)
