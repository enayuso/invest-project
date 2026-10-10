"""Chat and one-click analysis endpoints (streamed multi-agent pipeline
progress), plus past-decision transcript/history lookups."""

from __future__ import annotations

import asyncio
import json
import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.chat.router import answer as chat_answer
from app.clock import local_today
from app.execution.portfolio_risk import ticker_market
from app.memory.store import get_recent_entries, get_transcript
from app.orchestrator.pipeline import run_pipeline
from app.state import render_portfolio_decision
from app.web import state
from app.web.ticker_validation import validate_ticker

logger = logging.getLogger("app.web")

router = APIRouter()

# asyncio only holds a running task by a weak reference — without keeping our
# own strong reference, a task can be silently GC'd mid-run if nothing else
# holds it (same issue app.web.routers.live's _analysis_tasks exists for).
# 2026-08-26: also fixes a real bug, not just a GC risk — chat/analyze's own
# `finally: task.cancel()` used to cancel this task the instant the SSE
# client disconnected, contradicting this module's own docstring ("a client
# disconnect can't wedge the pipeline mid-run") — disconnect WAS the wedge.
# Kept alive here instead; only lifespan shutdown (app.web.state) cancels
# these now, via pending_chat_tasks() below.
_chat_tasks: set[asyncio.Task] = set()


def pending_chat_tasks() -> tuple[asyncio.Task, ...]:
    """Snapshot of in-flight chat/analyze cycles, for lifespan shutdown to
    cancel and await alongside the other background tasks — same treatment
    app.web.routers.live.pending_analysis_tasks gets, and for the same
    reason (see that function's docstring)."""
    return tuple(_chat_tasks)


def _log_chat_task_result(task: asyncio.Task) -> None:
    """Surface a failure. Nobody awaits these tasks directly (the SSE
    generator only reads from the queue they post onto), so without this an
    exception would only ever appear as asyncio's own "Task exception was
    never retrieved" at GC time. In practice `_run()` below already wraps
    its own body in try/except and posts a `done` event either way, so this
    is a backstop for a bug in that handling itself, not the expected path.
    Cancellation is normal on shutdown, not an error."""
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        logger.error("chat/analyze background task failed", exc_info=exc)


class ChatRequest(BaseModel):
    question: str


@router.post("/api/chat")
async def chat(req: ChatRequest) -> StreamingResponse:
    """Streams progress-stage events (`{"type":"stage","label":...}`) while
    a `deep_dive` question works through the full multi-agent pipeline —
    that can take a couple of minutes, and a single static "Thinking…" for
    the whole span is a bad look. Ends with one `{"type":"done","answer":...}`
    event. The other three intents are fast enough that they'll typically
    only ever emit the final `done` event, no stages in between — that's
    fine, there's nothing worth narrating there.

    A background task (not the generator itself) runs the actual work and
    posts events onto a queue the generator relays — same shape as
    `/api/stream`'s poll loop, so a client disconnect can't wedge the
    pipeline mid-run waiting on a generator that's no longer being pulled.
    """

    async def _events():
        queue: asyncio.Queue[dict] = asyncio.Queue()

        async def on_stage(label: str) -> None:
            await queue.put({"type": "stage", "label": label})

        async def _run() -> None:
            try:
                try:
                    portfolio_context = await state.portfolio_context()
                except Exception:
                    logger.warning("portfolio context unavailable for chat", exc_info=True)
                    portfolio_context = None
                result = await chat_answer(
                    req.question,
                    state.trade_allowlist_names(),
                    on_stage=on_stage,
                    portfolio_context=portfolio_context,
                )
                text, decision_id = result.text, result.decision_id
            except Exception:
                logger.exception("chat request failed for question: %r", req.question)
                text, decision_id = "Sorry, something went wrong answering that — try rephrasing the question.", None
            await queue.put({"type": "done", "answer": text, "decision_id": decision_id})

        task = asyncio.create_task(_run())
        _chat_tasks.add(task)
        task.add_done_callback(_chat_tasks.discard)
        task.add_done_callback(_log_chat_task_result)
        while True:
            item = await queue.get()
            yield f"data: {json.dumps(item)}\n\n"
            if item["type"] == "done":
                break
        # No `finally: task.cancel()` here — see _chat_tasks' own comment for
        # why disconnect must not cancel `_run()` mid-pipeline. The generator
        # returning simply stops relaying events; `_run()` keeps going and
        # its `done` event (still written to the DB via chat_answer's
        # deep_dive -> run_pipeline path) is just never read by anyone.

    return StreamingResponse(_events(), media_type="text/event-stream")


@router.post("/api/analyze/{ticker}")
async def analyze(ticker: str) -> StreamingResponse:
    """One-click full multi-agent analysis for a ticker — same streaming shape
    as /api/chat (stage events, then a done event), but runs the pipeline
    directly instead of routing a natural-language question, so a dashboard
    button can trigger the flagship flow without the user typing anything.

    Failure events carry an `error_code` (not pre-rendered text) so the
    client's i18n layer renders the message in whichever language is active,
    instead of baking Korean-only strings into the API response."""
    # Ticker format check, matching live.py/trade_allowlist.py's existing convention.
    # 티커 형식 검증 — 다른 라우터와 동일한 validate_ticker 재사용
    ticker = validate_ticker(ticker)

    async def _events():
        queue: asyncio.Queue[dict] = asyncio.Queue()

        async def on_stage(label: str) -> None:
            await queue.put({"type": "stage", "label": label})

        async def _run() -> None:
            try:
                # local_today(ticker_market(...)), not get_clock().today() — see
                # app.chat.router's deep_dive branch for the full reasoning
                # (files a US ticker under a full session-ahead KST date otherwise).
                # 티커 자신의 시장 달력 사용 — app.chat.router 주석 참고.
                pipeline_state = await run_pipeline(
                    ticker, local_today(ticker_market(ticker)), on_stage=on_stage, decision_trigger="manual",
                )
                if pipeline_state.final_decision is None:
                    await queue.put({"type": "done", "error_code": "no_decision", "ticker": ticker})
                    return
                await queue.put({
                    "type": "done",
                    "decision_id": pipeline_state.decision_id,
                    "rating": pipeline_state.final_decision.rating.value,
                    "answer": render_portfolio_decision(pipeline_state.final_decision),
                })
            except Exception:
                logger.exception("analyze failed for %s", ticker)
                await queue.put({"type": "done", "error_code": "analysis_failed", "ticker": ticker})

        task = asyncio.create_task(_run())
        _chat_tasks.add(task)
        task.add_done_callback(_chat_tasks.discard)
        task.add_done_callback(_log_chat_task_result)
        while True:
            item = await queue.get()
            yield f"data: {json.dumps(item)}\n\n"
            if item["type"] == "done":
                break
        # See /api/chat's identical comment above — disconnect must not
        # cancel a still-running run_pipeline call.

    return StreamingResponse(_events(), media_type="text/event-stream")


@router.get("/api/decision/{decision_id}")
async def decision_transcript(decision_id: int) -> dict:
    """Full agent-by-agent trail (analyst reports, both debates, research
    plan, trader proposal) behind one past decision — how the chat's "see
    how the agents got here" link and the dashboard's decision history both
    let the user look past the final rating into the reasoning."""
    try:
        data = await get_transcript(decision_id)
    except Exception as exc:
        # get_transcript deliberately doesn't degrade quietly on a DB failure
        # (see its own docstring) — but a raw 500 here reaches the client as
        # a plain-text body, breaking any caller expecting JSON. Same
        # "translate to a real HTTP error" convention as the kill-switch fix
        # (confirmed live 2026-08-06: 17 unhandled DB-timeout crashes on this
        # endpoint's sibling /api/decisions in one day).
        raise HTTPException(502, f"failed to load decision transcript: {exc}") from exc
    if data is None:
        raise HTTPException(status_code=404, detail=f"no decision with id {decision_id}")
    return data


@router.get("/api/decisions")
async def decisions(limit: int = 20) -> list[dict]:
    try:
        entries = await get_recent_entries(limit=limit)
    except Exception as exc:
        raise HTTPException(502, f"failed to load recent decisions: {exc}") from exc
    return [
        {
            "id": e.id,
            "ticker": e.ticker,
            # Same unified name source as every other panel — works even
            # for a ticker no longer on the trade allowlist (previously showed
            # the bare ticker code with no name at all).
            "name": await state.get_display_name(e.ticker),
            "trade_date": e.trade_date.isoformat(),
            "rating": e.rating,
            "status": e.status,
            "raw_return": e.raw_return,
            "alpha_return": e.alpha_return,
            "reflection": e.reflection,
            # decision_text (the AI's Rating/Executive Summary/Price Target
            # markdown at decision time) — unlike reflection, which stays
            # null until a later decision resolves this one's outcome, this
            # is populated immediately, so the AI Signals card has something
            # to show before any hindsight exists.
            "decision_text": e.decision_text,
            "created_at": e.created_at.isoformat(),
            # Buy/Sell/Increase/Decrease/Hold — what the rebalance actually
            # did to the position (current vs post-cap target shares), null
            # if the rebalance hasn't run yet. Deliberately separate from
            # `rating` (the AI's own conviction judgment) since the two can
            # disagree — see app.execution.sizing.rebalance_action_label.
            "action_label": e.action_label,
            # Short, display-only "why" for the AI Signals card, one per UI
            # language — null for rows predating these columns or where the
            # model omitted them; the frontend falls back to decision_text.
            "card_reason_ko": e.card_reason_ko,
            "card_reason_en": e.card_reason_en,
            # Display-only LLM token-usage total for this decision — see
            # app.memory.db's total_tokens column comment. Null for rows
            # predating this column; the frontend hides the cost footer then.
            "total_tokens": e.total_tokens,
            # 'auto' (scheduled auto-trade cycle, including the 다시 분석
            # button's re-screen) or 'manual' (a human picked this exact
            # ticker — the 분석 버튼, a chat deep dive, or a CLI run). Null
            # for rows predating the column; the frontend treats anything
            # that is not literally 'manual' as auto, mirroring
            # app.audit.stats's `IS DISTINCT FROM 'manual'` predicate.
            "decision_trigger": e.decision_trigger,
        }
        for e in entries
    ]
