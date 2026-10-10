"""Dev-only utility: seed a fake BUY pending order so the SEMI_AUTO approval
popup (app/web/static/index.html's showPendingApprovalModal) can be previewed
without waiting for the AI pipeline to actually produce a Buy rating — real
decisions on the current trade allowlist have all come back Sell/Hold so far.

Inserts directly into the same `pending_orders` table the running dashboard
reads, using the current price so it looks realistic. Appears in the
dashboard's 승인 대기 tab, toast card, and approval modal within one poll
cycle (~30s). NOT a real AI decision — reject or dismiss it once you've seen
the popup; approving it would submit a real (paper-account) order.

Deliberately calls `create_pending_order` WITHOUT a `trade_id` (no matching
`app.execution.ledger.record_trade` row exists for this fake proposal —
unlike the real `rebalance_to_target` path, which always writes one first).
Rejecting/expiring an order created this way is now a ledger no-op by
design (see pending_orders.py's `trade_id` column comment) — before that
fix (2026-08-06 incident, 005930.KS), rejecting a seeded order like this
one clamped-and-reversed against whatever the ledger's CURRENT position
happened to be, silently selling off real, unrelated shares.

Usage: python -m scripts.seed_test_pending_order [TICKER] [SHARES]
Must run in the same environment (.env / DB) as the actual dashboard.
"""

from __future__ import annotations

import asyncio
import re
import sys

from app.clock import get_clock
from app.config import settings
from app.dataflows.yfinance import get_verified_snapshot
from app.execution.pending_orders import create_pending_order

# Matches both a closed session's `close=` and a still-open session's
# `last=` (app.dataflows.session_state.format_verified_snapshot) — a script
# run during market hours would otherwise silently fail to parse the price.
_SNAPSHOT_RE = re.compile(r"(?:close|last)=([\d.]+)")


async def main() -> None:
    ticker = sys.argv[1] if len(sys.argv) > 1 else (settings.trade_allowlist[0] if settings.trade_allowlist else "AAPL")
    shares = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    snapshot = await get_verified_snapshot(ticker)
    match = _SNAPSHOT_RE.search(snapshot)
    if not match:
        raise RuntimeError(f"could not parse a price out of snapshot: {snapshot!r}")
    price = float(match[1])

    reason = (
        "**Rating**: Buy\n\n"
        "**Executive Summary**: [TEST DATA] Manually seeded to preview the approval "
        "popup — not a real AI decision. Illustrative entry near the current price "
        "with a defined stop.\n\n"
        f"**Price Target**: {price * 1.15:.2f}\n\n"
        "**Time Horizon**: 3-6 months\n\n"
        f"**Stop Loss**: {price * 0.92:.2f}"
    )

    order_id = await create_pending_order(ticker, get_clock().today(), "buy", shares, price, reason=reason)
    # Plain ASCII punctuation only in these prints: a cp949 Windows console
    # chokes on an em dash, same gotcha noted in kis_bridge.py's dispositions.
    print(f"created test pending BUY order id={order_id} ticker={ticker} shares={shares} price={price:.2f}")
    print("It'll show up in the dashboard's Pending-Approval tab / toast / popup within ~30s.")
    print("Reject or dismiss it once you've seen the popup - it isn't a real AI decision.")


if __name__ == "__main__":
    asyncio.run(main())
