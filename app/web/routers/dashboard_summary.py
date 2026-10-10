"""Aggregate endpoint for the dashboard's fixed-interval polls (Phase 6 of
the browser-slowness fix, see docs/... — portfolio, index, trades, open
orders, and the decisions list each of the first three separately
re-fetched on their own timer).

Composed by directly awaiting each existing route handler rather than
duplicating their logic — a FastAPI path function is just a plain async
function, nothing prevents calling it from another one. This keeps the
individual endpoints (/api/portfolio, /api/index, /api/trades,
/api/orders/open, /api/decisions) working exactly as before for any other
caller; this endpoint is purely an additional, thinner way to get the same
data in one round trip.

Does not reduce the number of underlying KIS calls by itself — each
handler still goes through the same caches (app.web.state.cached) as
before. The win is fewer browser<->server round trips, plus incidentally
deduplicating the decisions fetch that /api/portfolio's and /api/trades'
own frontend callers each did separately (see main.js's pollPortfolio/
pollTrades prior to migrating to this endpoint).
"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter

from app.web.routers.chat_decisions import decisions
from app.web.routers.market_data import index_quotes_endpoint
from app.web.routers.orders import open_orders_endpoint
from app.web.routers.portfolio import portfolio, trades

router = APIRouter()


@router.get("/api/dashboard-summary")
async def dashboard_summary() -> dict:
    portfolio_data, index_data, trades_data, open_orders_data, decisions_data = await asyncio.gather(
        portfolio(), index_quotes_endpoint(), trades(limit=100), open_orders_endpoint(), decisions(limit=100),
    )
    return {
        "portfolio": portfolio_data,
        "index": index_data,
        "trades": trades_data,
        "open_orders": open_orders_data,
        "decisions": decisions_data,
    }
