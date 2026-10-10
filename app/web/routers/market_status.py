"""Real-time market-operation status, orderbook, and fill notifications from
the WebSocket feed (app.brokers.kis.domestic.websocket.RealtimeFeed)."""

from __future__ import annotations

from fastapi import APIRouter, Query

from app.web import state

router = APIRouter()


@router.get("/api/market-status")
async def market_status_endpoint() -> dict:
    """Latest real-time market-operation state per KRX code from the
    WebSocket feed — trading-halt, VI, and the raw market-operation code
    (which carries circuit-breaker state; see app.live.kis_market_status for
    the interpretation layer app.execution.kis_bridge's gate actually uses).
    Empty until the feed receives its first frame (KRX trading hours only)."""
    if state._market_feed is None:
        return {}
    return {
        f"{code}.KS": {
            "trading_halted": s.trading_halted,
            "halt_reason": s.halt_reason,
            "market_op_code": s.market_op_code,
            "vi_code": s.vi_code,
            "stock_state_code": s.stock_state_code,
        }
        for code, s in state._market_feed.statuses.items()
    }


@router.get("/api/market-status/feed-subscriptions")
async def feed_subscriptions_endpoint() -> dict:
    """Diagnostic-only, read-only: which KRX codes the WS feed's subscribe
    list (`RealtimeFeed.codes`) currently covers, and whether the socket
    itself is connected — the two facts `is_circuit_breaker_active` actually
    keys off (see its docstring). `/api/market-status` alone can't answer
    this: it only shows codes that have received an event FRAME, and on a
    quiet day a correctly-subscribed code may never get one — so an empty
    `/api/market-status` doesn't mean "not subscribed." Temporary: added
    2026-07-30 to check whether a same-day `add_code()` call (from
    `reconcile_trade_allowlist`) actually landed, after a newly-reconciled
    ticker's BUY was blocked with "circuit breaker not connected." No
    behavior change to the feed/gate itself."""
    if state._market_feed is None:
        return {"connected": False, "codes": []}
    return {"connected": state._market_feed.connected, "codes": sorted(state._market_feed.codes)}


@router.get("/api/orderbook/{ticker}")
async def orderbook_endpoint(ticker: str) -> dict:
    """Live 10-level bid/ask orderbook (호가창) from the WebSocket feed for a
    KRX ticker. Empty until a frame arrives (KRX trading hours only)."""
    if state._market_feed is None:
        return {}
    code = ticker.upper()[:-3] if ticker.upper().endswith(".KS") else ticker.upper()
    ob = state._market_feed.orderbook.get(code)
    tick = state._market_feed.ticks.get(code)
    if ob is None and tick is None:
        return {}
    return {
        "ticker": f"{code}.KS",
        "time": ob.time if ob else (tick.time if tick else ""),
        "asks": ob.asks if ob else [],
        "bids": ob.bids if ob else [],
        "last": tick.price if tick else None,
        "change_pct": tick.change_pct if tick else None,
        "volume": tick.volume if tick else None,
    }


@router.get("/api/fills")
async def fills_endpoint(limit: int = Query(default=10, ge=0)) -> list[dict]:
    """Recent real-time fill notifications (체결통보) from the WebSocket
    feed, newest first. Empty until a 체결통보 subscription is active (needs
    an HTS ID configured) and an order actually fills during trading hours."""
    if state._market_feed is None:
        return []
    # `fills[-limit:]` is wrong for limit <= 0: -0 == 0, so limit=0 sliced
    # `fills[0:]` and returned the WHOLE buffer instead of nothing. `ge=0` at
    # the API boundary rejects a negative limit with a 422, and this explicit
    # check makes the literal zero-row request mean zero rows — kept in
    # addition to `ge=0` because this function is also called directly in
    # tests, bypassing FastAPI's own validation.
    # -0 == 0 이라 limit=0이 전체를 반환하던 슬라이스 버그.
    if limit == 0:
        return []
    fills = list(reversed(state._market_feed.fills[-limit:]))
    rows = [
        {"order_no": f.order_no, "ticker": f"{f.code}.KS", "name": f.name, "side": f.side,
         "order_qty": f.order_qty, "fill_qty": f.fill_qty, "fill_price": f.fill_price,
         "time": f.time, "status": f.status}
        for f in fills
    ]
    # KIS's own 체결통보 name field (cntg_isnm40) has been observed blank on
    # the paper account (real incident: 042660.KS fill showed the bare
    # ticker as its "name" in the trade-fill modal, since the frontend's
    # `f.name || f.ticker` fallback had nothing better to fall back to).
    # Same display-name cache every other panel already uses, so this
    # resolves to the same "한화오션" the portfolio table shows for the
    # same ticker — not a second, possibly-inconsistent name source.
    for row in rows:
        if not row["name"]:
            row["name"] = await state.get_display_name(row["ticker"])
    return rows
