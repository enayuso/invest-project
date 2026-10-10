"""chat_decisions router — decision transcript/history lookups. memory.store
stubbed, no DB (E-stage). The two SSE streaming endpoints (/api/chat,
/api/analyze/{ticker}) drive a long-running background task per request and
are not covered here — the pipeline/chat logic they wrap is already tested
directly in test_pipeline_integration.py and app.chat's own tests."""

from __future__ import annotations

import asyncio
import logging
from datetime import date, datetime

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.web import state
from app.web.routers import chat_decisions


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(chat_decisions.router)
    return TestClient(app, base_url="http://127.0.0.1:8094")


async def test_decision_transcript_404_when_missing(client, monkeypatch):
    async def fake_get_transcript(decision_id):
        return None

    monkeypatch.setattr(chat_decisions, "get_transcript", fake_get_transcript)
    resp = client.get("/api/decision/999")
    assert resp.status_code == 404


async def test_decision_transcript_returns_data_when_found(client, monkeypatch):
    async def fake_get_transcript(decision_id):
        return {"reports": {"Market Analyst": "bullish"}, "research_plan": "Buy"}

    monkeypatch.setattr(chat_decisions, "get_transcript", fake_get_transcript)
    resp = client.get("/api/decision/42")
    assert resp.status_code == 200
    assert resp.json()["research_plan"] == "Buy"


async def test_decisions_list_uses_shared_display_name_source(client, monkeypatch):
    class _FakeEntry:
        id = 1
        ticker = "AAPL"
        trade_date = date(2026, 7, 15)
        rating = "Buy"
        status = "open"
        raw_return = 0.05
        alpha_return = 0.02
        reflection = None
        decision_text = "**Rating**: Buy\n\n**Executive Summary**: Strong momentum."
        created_at = datetime(2026, 7, 15, 9, 5)
        action_label = "Buy"
        card_reason_ko = "외국인 순매수 지속."
        card_reason_en = "Foreign net buying continues."
        total_tokens = 8642
        decision_trigger = "manual"

    async def fake_get_recent_entries(limit=20):
        return [_FakeEntry()]

    async def fake_get_display_name(ticker):
        return "Apple Inc."

    monkeypatch.setattr(chat_decisions, "get_recent_entries", fake_get_recent_entries)
    monkeypatch.setattr(state, "get_display_name", fake_get_display_name)

    resp = client.get("/api/decisions")
    assert resp.status_code == 200
    body = resp.json()
    assert body[0]["name"] == "Apple Inc."
    assert body[0]["trade_date"] == "2026-07-15"
    # card_reason_ko/card_reason_en — new dashboard-card "why" fields,
    # exposed verbatim so the frontend can pick the one matching UI language.
    assert body[0]["card_reason_ko"] == "외국인 순매수 지속."
    assert body[0]["card_reason_en"] == "Foreign net buying continues."
    assert body[0]["total_tokens"] == 8642
    assert body[0]["decision_trigger"] == "manual"


async def test_decisions_list_respects_limit_param(client, monkeypatch):
    captured = {}

    async def fake_get_recent_entries(limit=20):
        captured["limit"] = limit
        return []

    monkeypatch.setattr(chat_decisions, "get_recent_entries", fake_get_recent_entries)
    client.get("/api/decisions?limit=5")
    assert captured["limit"] == 5


# get_recent_entries/get_transcript deliberately don't degrade quietly on a
# DB failure (see their own docstrings) — but the router previously let that
# raw exception reach the client as a plain-text 500 body. Confirmed live
# 2026-08-06: 17 unhandled DB-timeout crashes on /api/decisions in one day,
# same class of bug already fixed for the kill switch endpoint.
async def test_decisions_list_db_failure_returns_502_json(client, monkeypatch):
    async def fake_get_recent_entries(limit=20):
        raise TimeoutError("db timeout")

    monkeypatch.setattr(chat_decisions, "get_recent_entries", fake_get_recent_entries)
    resp = client.get("/api/decisions")
    assert resp.status_code == 502
    assert "failed to load recent decisions" in resp.json()["detail"]


async def test_decision_transcript_db_failure_returns_502_json(client, monkeypatch):
    async def fake_get_transcript(decision_id):
        raise TimeoutError("db timeout")

    monkeypatch.setattr(chat_decisions, "get_transcript", fake_get_transcript)
    resp = client.get("/api/decision/42")
    assert resp.status_code == 502
    assert "failed to load decision transcript" in resp.json()["detail"]


async def test_analyze_rejects_malformed_ticker_before_running_pipeline(client, monkeypatch):
    async def should_not_be_called(*args, **kwargs):
        raise AssertionError("pipeline must not run for an invalid ticker")

    monkeypatch.setattr(chat_decisions, "run_pipeline", should_not_be_called)
    resp = client.post("/api/analyze/<script>alert(1)>")
    assert resp.status_code == 422


async def test_analyze_resolves_trade_date_from_the_ticker_own_market(client, monkeypatch):
    # Regression test for the dataflows-folder-audit stage-3 bug: the one-click
    # "analyze" button used to pass get_clock().today() (server KST date) for
    # every ticker. At KST 2026-09-11 10:00, ET is still 2026-09-10 21:00 — an
    # overseas ticker must file under the ET date, a .KS ticker under the KST
    # date.
    from zoneinfo import ZoneInfo

    from app.clock import FakeClock, LiveClock, set_clock

    try:
        set_clock(FakeClock(datetime(2026, 9, 11, 10, 0, tzinfo=ZoneInfo("Asia/Seoul"))))

        seen = {}

        async def fake_run_pipeline(ticker, trade_date, on_stage=None, decision_trigger="manual"):
            seen["trade_date"] = trade_date

            class FakeState:
                final_decision = None
                decision_id = None

            return FakeState()

        monkeypatch.setattr(chat_decisions, "run_pipeline", fake_run_pipeline)
        resp = client.post("/api/analyze/AAPL")
        assert resp.status_code == 200
        list(resp.iter_lines())  # drain the SSE stream so the background task completes
        assert seen["trade_date"] == date(2026, 9, 10)

        resp2 = client.post("/api/analyze/005930.KS")
        assert resp2.status_code == 200
        list(resp2.iter_lines())
        assert seen["trade_date"] == date(2026, 9, 11)
    finally:
        set_clock(LiveClock())


# ---------------------------------------------------------------------------
# Background task lifecycle (2026-08-26 fix, FINDING H): a chat/analyze SSE
# client disconnecting must not cancel the still-running pipeline task —
# only lifespan shutdown does that, via pending_chat_tasks(). These exercise
# the tracking helpers directly rather than through a real SSE round trip
# (the module docstring above deliberately keeps that out of scope here).
# ---------------------------------------------------------------------------


async def test_pending_chat_tasks_tracks_and_untracks_a_task():
    async def _noop():
        return None

    task = asyncio.create_task(_noop())
    chat_decisions._chat_tasks.add(task)
    task.add_done_callback(chat_decisions._chat_tasks.discard)
    assert task in chat_decisions.pending_chat_tasks()

    await task
    # add_done_callback runs on the next loop iteration, not synchronously
    # with the awaited task's own completion — yield once to let it fire.
    await asyncio.sleep(0)
    assert task not in chat_decisions.pending_chat_tasks()


async def test_log_chat_task_result_ignores_a_cancelled_task(caplog):
    async def _hangs_forever():
        await asyncio.sleep(3600)

    task = asyncio.create_task(_hangs_forever())
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    with caplog.at_level(logging.ERROR):
        chat_decisions._log_chat_task_result(task)
    assert not any(r.levelno >= logging.ERROR for r in caplog.records)


async def test_log_chat_task_result_logs_an_unexpected_exception(caplog):
    async def _boom():
        raise RuntimeError("unexpected bug in _run()'s own error handling")

    task = asyncio.create_task(_boom())
    with pytest.raises(RuntimeError):
        await task
    with caplog.at_level(logging.ERROR):
        chat_decisions._log_chat_task_result(task)
    assert any(r.levelno >= logging.ERROR for r in caplog.records)
