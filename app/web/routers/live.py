"""Live tick/news/earnings/alerts endpoints for the dashboard, plus the
SSE tick-price stream."""

from __future__ import annotations

import asyncio
import json
import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.config import settings
from app.dataflows.yfinance import get_earnings_trend
from app.execution import runtime_settings
from app.execution.kis_bridge import route_ticker
from app.live import alerts as alerts_module
from app.live.auto_trade_loop import (
    MANUAL_TRIGGER_COOLDOWN_SECONDS,
    CooldownActive,
    check_manual_trigger_allowed,
    get_status as get_auto_trade_status,
    run_cycle_now,
)
from app.live.market_hours import (
    latest_domestic_signal_card_reset_at,
    latest_overseas_signal_card_reset_at,
)
from app.live.store import get_ticks, latest
from app.web import state
from app.web.ticker_validation import validate_ticker as _validate_ticker

logger = logging.getLogger("app.web")

router = APIRouter()


@router.get("/api/ticks/{ticker}")
async def ticks(ticker: str) -> list[dict]:
    try:
        return [
            {"price": t.price, "volume": t.volume, "timestamp": t.timestamp.isoformat()}
            for t in get_ticks(_validate_ticker(ticker))
        ]
    except HTTPException:
        raise
    except Exception:
        logger.warning("could not load ticks for %s", ticker, exc_info=True)
        return []


@router.get("/api/news/{ticker}")
async def news_endpoint(ticker: str) -> list[dict]:
    """KIS-only by policy (see app.live.poller's module docstring) — no
    yfinance fallback. Unlike yfinance's get_news_items, the KIS news TRs
    (domestic 국내주식-141 / overseas 해외주식-053) return title/source/time
    only, no article link or thumbnail — those fields are just absent from
    the response now rather than faked."""
    ticker = _validate_ticker(ticker)
    broker = await state.kis_broker()
    if broker is None:
        return []
    krx_code, us_symbol = route_ticker(ticker.upper())
    try:
        if krx_code is not None:
            return await broker.fetch_news(krx_code)
        if us_symbol is not None:
            return await broker.fetch_overseas_news(us_symbol)
        return []
    except HTTPException:
        raise
    except Exception:
        logger.warning("could not load news for %s", ticker, exc_info=True)
        return []


@router.get("/api/earnings-trend/{ticker}")
async def earnings_trend_endpoint(ticker: str) -> dict:
    try:
        return await get_earnings_trend(_validate_ticker(ticker))
    except HTTPException:
        raise
    except Exception:
        logger.warning("could not load earnings trend for %s", ticker, exc_info=True)
        return {}


@router.get("/api/alerts")
async def alerts_endpoint() -> list[dict]:
    rows = await alerts_module.recent_alerts()
    return [
        {
            "ticker": a.ticker,
            "kind": a.kind,
            "message": a.message,
            "triggered_at": a.triggered_at.isoformat(),
        }
        for a in rows
    ]


@router.get("/api/auto-trade/status")
async def auto_trade_status_endpoint() -> dict:
    """Dashboard visibility into app.live.auto_trade_loop — is it running,
    what's it doing right now, when's the next cycle. `enabled` is False
    whenever the loop's own background task was never started (MANUAL mode
    or auto_trade_enabled off) — that's a distinct, more definitive "off"
    signal than the loop's own `phase` field, which would otherwise just
    look identical to "idle, waiting for next cycle"."""
    s = get_auto_trade_status()
    current_execution_mode = runtime_settings.rule("execution_mode")
    _domestic_reset_at = await latest_domestic_signal_card_reset_at()
    current_tickers = sorted(s.current_tickers)
    # Name lookup, not just the bare ticker — a screened/momentum candidate
    # analyzed this cycle may never have been on the trade allowlist (see
    # app.live.auto_trade_loop's candidate lists), so the "분석 중..." card
    # otherwise has no name source at all and falls back to showing the bare
    # code the whole time it's analyzing. get_display_name already covers
    # any ticker, not just held/trade-allowlist ones — same source
    # /api/decisions uses once the analysis finishes.
    current_ticker_names = dict(
        zip(current_tickers, await asyncio.gather(*(state.get_display_name(t) for t in current_tickers)))
    ) if current_tickers else {}
    return {
        "enabled": settings.auto_trade_enabled and current_execution_mode.value != "manual",
        "execution_mode": current_execution_mode.value,
        "phase": s.phase,
        "current_tickers": current_tickers,
        "current_ticker_names": current_ticker_names,
        "current_stage": s.current_stage,
        "tickers_this_cycle": s.tickers_this_cycle,
        "cycle_started_at": s.cycle_started_at.isoformat() if s.cycle_started_at else None,
        "cycle_finished_at": s.cycle_finished_at.isoformat() if s.cycle_finished_at else None,
        "next_cycle_at": s.next_cycle_at.isoformat() if s.next_cycle_at else None,
        "manual_trigger_cooldown_seconds": MANUAL_TRIGGER_COOLDOWN_SECONDS,
        # The portfolio table marks a holding as "near its stop" using this
        # same fraction, so the highlight means "inside the band that forces
        # a pipeline re-review this cycle" rather than a display-only
        # threshold the frontend invented.
        "held_review_buffer_pct": settings.auto_trade_held_review_buffer_pct,
        # AI 시그널 card history clears at these boundaries — the frontend
        # drops any decision older than its own market's boundary instead of
        # showing yesterday's leftovers forever. Split per market 2026-08-20:
        # a single shared KRX-based value used to wipe US (overseas) cards
        # at 08:30 KST, hours before the US market had even opened (5 real
        # decisions filed 22:46-22:58 KST the night before vanished at the
        # very next KST morning boundary).
        "cards_reset_at_domestic": _domestic_reset_at.isoformat(),
        "cards_reset_at_overseas": (await latest_overseas_signal_card_reset_at()).isoformat(),
        # Deprecated: kept equal to the domestic value so a browser holding
        # cached pre-split frontend JS still filters domestic cards exactly
        # as before, instead of losing the filter entirely (new Date(undefined)
        # would be Invalid Date on the old code's single `cards_reset_at` read).
        "cards_reset_at": _domestic_reset_at.isoformat(),
    }


# Strong references to in-flight fire-and-forget cycles. asyncio only holds a
# WEAK reference to a running task, so a bare `asyncio.create_task(...)` whose
# result nobody keeps can be garbage-collected mid-run — the cycle then just
# stops, silently, part-way through (CPython's own asyncio docs call this out
# explicitly). Every other create_task in this codebase already retains its
# handle (app.web.state's lifespan tasks, chat_decisions' streaming task);
# this endpoint was the one that didn't.
# asyncio는 실행 중인 태스크를 약한 참조로만 잡음 — 반환값을 안 들고 있으면 사이클이
# 도중에 GC되어 조용히 중단될 수 있어서, 완료될 때까지 강한 참조를 유지함.
_analysis_tasks: set[asyncio.Task] = set()


def pending_analysis_tasks() -> tuple[asyncio.Task, ...]:
    """Snapshot of in-flight manual '다시 분석' cycles, for lifespan shutdown
    to cancel and await alongside the other background tasks — see
    app.web.state's lifespan docstring/comment for why this one needs the
    same treatment the other 8 background loops already get."""
    return tuple(_analysis_tasks)


def _log_analysis_task_result(task: asyncio.Task) -> None:
    """Surface a failed cycle. Nobody awaits these tasks, so without this an
    exception only ever appears as asyncio's own "Task exception was never
    retrieved" at GC time — detached from the request that started it and
    easy to miss entirely. Cancellation is normal on shutdown, not an error."""
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        logger.error("manual '다시 분석' cycle failed", exc_info=exc)


@router.post("/api/auto-trade/analyze-now")
async def trigger_auto_trade_analysis() -> dict:
    """SEMI_AUTO's "다시 분석" button. Synchronous cooldown/mode check so a
    disallowed click gets an immediate 429, not a request that silently
    waits on the whole cycle; the actual run is fire-and-forget (the
    dashboard already polls /api/auto-trade/status + /api/decisions to show
    progress, so this endpoint doesn't need to hold the connection open for
    the ~20-LLM-call pipeline to finish).

    Note `check_manual_trigger_allowed` has already claimed the "a cycle is
    running" slot (phase="scanning") by the time the task is spawned — that
    slot is released by `run_cycle_now`'s own try/finally, which is why that
    guard has to span its whole body rather than just the analysis part.
    See its own comment for the wedge this prevents."""
    try:
        check_manual_trigger_allowed()
    except CooldownActive as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    task = asyncio.create_task(run_cycle_now(exclude_already_analyzed_today=True, force_refresh=True))
    _analysis_tasks.add(task)
    task.add_done_callback(_analysis_tasks.discard)
    task.add_done_callback(_log_analysis_task_result)
    return {"status": "started"}


_STREAM_MAX_TICKS = 60  # ~5min at the 5s interval below
_STREAM_RECONNECT_MS = 200  # SSE `retry:` hint, so the ~5min reconnect is near-instant


@router.get("/api/stream")
async def stream() -> StreamingResponse:
    """SSE tick-price feed. Ends on its own after `_STREAM_MAX_TICKS` (~5min)
    instead of running forever — the frontend uses a native browser
    `EventSource` (index.html), which reconnects automatically on a closed
    connection. The `retry:` hint below makes that reconnect near-instant
    (default browser behavior is a ~3s wait), so the periodic
    disconnect/reconnect is not noticeable in normal use.

    The real reason: an unbounded `while True` stream never lets uvicorn's
    graceful-shutdown "wait for open connections to close" phase finish
    while any dashboard tab is open, which in turn means the app's lifespan
    shutdown (app.web.state's `finally:` block, which stops the KIS realtime
    WebSocket feed) never runs either — Ctrl+C just hangs, and on Windows a
    second Ctrl+C isn't reliably delivered to a busy asyncio process either
    (asyncio has no signal-handler support there at all). Bounding this
    connection's lifetime is what actually lets a normal Ctrl+C finish."""

    async def _events():
        yield f"retry: {_STREAM_RECONNECT_MS}\n\n"
        for _ in range(_STREAM_MAX_TICKS):
            payload = {
                ticker: (
                    {"price": q.price, "volume": q.volume, "timestamp": q.timestamp.isoformat()}
                    if (q := latest(ticker))
                    else None
                )
                for ticker in settings.trade_allowlist
            }
            yield f"data: {json.dumps(payload)}\n\n"
            await asyncio.sleep(5)

    return StreamingResponse(_events(), media_type="text/event-stream")
