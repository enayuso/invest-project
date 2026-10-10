"""On-demand, wide-window version of `app.execution.reconciliation.
find_unmirrored_trades` — that function's only existing caller
(`app.web.state._startup_trade_reconciliation`) runs once at boot with a
48-hour lookback, built to catch a process-kill-mid-mirror crash while it's
still fresh. It was never meant to answer "how many ledger trades over the
last N weeks were never actually confirmed at KIS" — the exact question
this session's slippage investigation (`app.audit.slippage_validation`)
needed answered after finding 3 "submitted: ..." trades (order ACCEPTED,
per `app.brokers.kis.domestic.order.submit_order`'s own log line, not
necessarily filled) with no matching KIS execution record, one of them
still a live 10-share phantom position weeks later.

Same "detect, never correct" boundary as reconciliation.py itself (see that
module's docstring for why: right after a submission, "KIS never got the
order" and "KIS got it but hasn't confirmed the fill yet" are
indistinguishable without an independent execution-history check like this
one). This is that independent check, runnable over an arbitrary historical
window instead of only the last 48 hours.

Read-only. No LLM calls, no ledger writes."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta, timezone

from app.brokers import get_broker
from app.execution.reconciliation import TradeDiscrepancy, fetch_recent_local_trades, match_trades_against_kis
from app.memory.db import get_pool

_KIS_LOOKBACK_LIMIT_DAYS = 90  # same TR cap app.audit.slippage_validation documents


@dataclass
class ReconciliationReport:
    discrepancies: list[TradeDiscrepancy] = field(default_factory=list)
    trades_checked: int = 0


async def compute_reconciliation(days: int = 30) -> ReconciliationReport:
    """`days`: how far back to look for local `trades` rows to check —
    capped to `_KIS_LOOKBACK_LIMIT_DAYS` since that's the hard limit on how
    far back KIS's own execution-history TR will answer at all (a trade
    older than that literally cannot be verified this way, matched or not
    — see app.execution.pnl's identical cap). Probes the DB directly before
    delegating to the (deliberately fail-open) `fetch_recent_local_trades`,
    so a real DB outage raises here instead of silently rendering as a
    clean "0 trades checked" report."""
    days = min(days, _KIS_LOOKBACK_LIMIT_DAYS)
    broker = await get_broker("kis")
    if broker is None:
        raise RuntimeError("KIS broker not configured — cannot fetch execution history")

    end = date.today()
    start = end - timedelta(days=days)
    start_str, end_str = start.strftime("%Y%m%d"), end.strftime("%Y%m%d")
    domestic_executions = await broker.fetch_daily_executions(start_str, end_str)
    overseas_executions = await broker.fetch_overseas_executions(start_str, end_str)

    # fetch_recent_local_trades is deliberately fail-open (see its own
    # docstring) for its original boot-time caller, which must never crash the
    # app on a transient DB hiccup. An on-demand report has the opposite
    # requirement — a swallowed DB failure must not render as "0 trades,
    # everything reconciles" (see app.audit.stats's own module docstring: "a DB
    # failure here should be visible to whoever ran the report, not silently
    # swallowed") — so probe the DB directly here and let a real failure raise,
    # instead of silently trusting a possibly-fail-opened [].
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.fetchval("SELECT 1")

    # grace_period=0: unlike the startup check (which must skip a trade
    # whose mirror_trade_to_kis call may still be mid-flight, seconds old),
    # a report run on demand has no such freshly-in-flight trade to worry
    # about excluding.
    local_trades = await fetch_recent_local_trades(timedelta(days=days), timedelta(seconds=0))
    discrepancies = match_trades_against_kis(local_trades, domestic_executions, overseas_executions)
    return ReconciliationReport(discrepancies=discrepancies, trades_checked=len(local_trades))


def render_reconciliation(r: ReconciliationReport) -> str:
    if r.trades_checked == 0:
        return "확인 대상 거래가 없습니다 (기간 안에 원장 거래 0건)."
    if not r.discrepancies:
        return f"확인한 거래 {r.trades_checked}건 전부 KIS 체결내역과 일치합니다."

    critical = [d for d in r.discrepancies if d.severity == "CRITICAL"]
    warning = [d for d in r.discrepancies if d.severity == "WARNING"]
    lines = [
        f"확인한 거래 {r.trades_checked}건 중 불일치 {len(r.discrepancies)}건 "
        f"(CRITICAL {len(critical)} / WARNING {len(warning)})",
        "",
    ]
    for label, group in (("CRITICAL", critical), ("WARNING", warning)):
        if not group:
            continue
        lines.append(f"[{label}]")
        for d in sorted(group, key=lambda x: x.created_at):
            lines.append(
                f"  trade_id={d.trade_id} {d.trade_date} {d.ticker} {d.side} x{d.shares} "
                f"(기록 {d.created_at.astimezone(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}) — {d.note}"
            )
        lines.append("")

    lines.append(
        "참고: 이건 탐지만 합니다 — 원장을 자동으로 고치지 않습니다. "
        "직접 체결 여부를 KIS에서 재확인 후 필요하면 사람이 원장을 보정해야 합니다."
    )
    return "\n".join(lines)
