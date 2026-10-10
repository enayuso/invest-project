"""app.web.state's lifespan shutdown — verifies the bounded cancel-and-await
sequence in the `finally` block actually reaches every background task,
including the manual '다시 분석' cycles tracked in app.web.routers.live's
module-level _analysis_tasks set (added 2026-08-26: that task source used to
get cancelled by asyncio's own process-exit cleanup only, AFTER this block
had already torn down the broker connection pools a mid-flight cycle might
still be using).

Runs the real `lifespan` async generator rather than a no-op stand-in (see
test_app_cache_headers.py's `_noop_lifespan` for why other tests avoid this):
every startup call that would hit a DB, KIS, or a WebSocket is monkeypatched
to a fast no-op, and the 8 background loops are replaced with a coroutine
that sleeps forever, so cancellation is the only way any of them finish."""

from __future__ import annotations

import asyncio

import pytest

from app.web import state
from app.web.routers import chat_decisions, live


async def _forever() -> None:
    await asyncio.sleep(3600)


@pytest.fixture
def patched_lifespan(monkeypatch):
    """Stub out every real startup side effect `lifespan` performs, and
    replace its 8 background loops with a coroutine that only stops when
    cancelled — this isolates the test to the shutdown sequence itself."""
    async def _noop(*_a, **_kw) -> None:
        return None

    async def _noop_scopes() -> list:
        return []

    monkeypatch.setattr(state, "register_user", _noop)
    monkeypatch.setattr(state, "record_session_start", _noop)
    monkeypatch.setattr(state, "_verify_configured_kis_credentials", _noop)
    monkeypatch.setattr(state.risk_rules, "load", _noop)
    monkeypatch.setattr(state.runtime_settings, "load", _noop)
    monkeypatch.setattr(state, "serviced_scopes", _noop_scopes)
    monkeypatch.setattr(state.trade_allowlist_store, "ensure_loaded", _noop)
    monkeypatch.setattr(state, "ensure_market_feed", _noop)
    monkeypatch.setattr(state, "stop_market_feed", _noop)
    monkeypatch.setattr(state, "invalidate_broker_cache", _noop)

    monkeypatch.setattr(state, "_prefetch_startup_data", _forever)
    monkeypatch.setattr(state, "trade_reconciliation_loop", _forever)
    monkeypatch.setattr(state, "run_poller_loop", _forever)
    monkeypatch.setattr(state, "stop_loss_loop", _forever)
    monkeypatch.setattr(state, "exit_monitor_loop", _forever)
    monkeypatch.setattr(state, "fill_monitor_loop", _forever)
    monkeypatch.setattr(state, "auto_trade_loop", _forever)
    monkeypatch.setattr(state, "run_orderbook_logger", _forever)
    monkeypatch.setattr(state.settings, "auto_trade_enabled", False)
    monkeypatch.setattr(state.settings, "dashboard_graceful_shutdown_timeout_seconds", 5.0)


@pytest.mark.asyncio
async def test_shutdown_cancels_pending_manual_analysis_task(patched_lifespan):
    """A manual '다시 분석' cycle still in flight when the app shuts down must
    be cancelled and awaited by lifespan's finally block, exactly like the
    other 8 background loops — not left to asyncio's own process-exit
    cleanup, which only runs after stop_market_feed() has already torn down
    the broker connection pools it might still be using."""
    manual_task = asyncio.create_task(_forever())
    live._analysis_tasks.add(manual_task)
    manual_task.add_done_callback(live._analysis_tasks.discard)

    try:
        async with state.lifespan(None):
            assert manual_task in live.pending_analysis_tasks()
            assert not manual_task.done()

        # lifespan's finally block has now run to completion (bounded by
        # dashboard_graceful_shutdown_timeout_seconds) — the task must be
        # cancelled, not still running and merely GC-pending.
        assert manual_task.done()
        assert manual_task.cancelled()
        assert manual_task not in live.pending_analysis_tasks()
    finally:
        if not manual_task.done():
            manual_task.cancel()


@pytest.mark.asyncio
async def test_shutdown_cancels_pending_chat_task(patched_lifespan):
    """A chat/analyze SSE background task (app.web.routers.chat_decisions'
    _chat_tasks) still in flight when the app shuts down must be cancelled
    and awaited by lifespan's finally block — 2026-08-26 fix: these used to
    be cancelled by their own request handler's `finally: task.cancel()` the
    instant the SSE client disconnected, which was the actual bug (killing a
    still-running deep_dive mid-pipeline). Now only shutdown cancels them."""
    chat_task = asyncio.create_task(_forever())
    chat_decisions._chat_tasks.add(chat_task)
    chat_task.add_done_callback(chat_decisions._chat_tasks.discard)

    try:
        async with state.lifespan(None):
            assert chat_task in chat_decisions.pending_chat_tasks()
            assert not chat_task.done()

        assert chat_task.done()
        assert chat_task.cancelled()
        assert chat_task not in chat_decisions.pending_chat_tasks()
    finally:
        if not chat_task.done():
            chat_task.cancel()
