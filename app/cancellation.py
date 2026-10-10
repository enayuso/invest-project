# app/cancellation.py
# Dependency-free home for the "was THIS task actually asked to stop"
# classifier — moved out of app.orchestrator.pipeline (2026-08-25) so
# app.agents.tool_loop can use it too without creating an agents->orchestrator
# import cycle (pipeline.py already imports from app.agents). Same precedent
# as app.clock/app.account_mode: a small pure helper gets its own top-level
# module rather than living inside whichever caller happened to need it first.
# 순수 헬퍼라 자체 모듈로 분리 — app.agents가 app.orchestrator를 참조하는
# 순환 임포트를 피하기 위함.

from __future__ import annotations

import asyncio


def task_was_actually_cancelled() -> bool:
    """Distinguishes a real outer cancellation (someone called `.cancel()`
    on this task — shutdown, an enclosing gather/TaskGroup tearing down)
    from a `CancelledError` that merely surfaced through this task without
    it being the target — e.g. a connection-pool internal wait getting
    cancelled by something unrelated. `Task.cancelling()` (3.11+) counts
    outstanding cancellation requests against *this* task, so 0 means
    nothing actually asked it to stop. Shared by every id=377/id=949-shaped
    call site that needs the same classification instead of duplicating it:
    `app.orchestrator.pipeline` (the original id=377/id=949 incident),
    `app.live.auto_trade_loop`'s per-ticker `_analyze_one`, and
    `app.agents.tool_loop`'s per-tool-call cancellation guard."""
    task = asyncio.current_task()
    return task is not None and task.cancelling() > 0
