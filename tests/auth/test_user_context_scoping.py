"""The ContextVar identity seam (multi-user migration Phase 1).

The property that actually matters for one-process-many-users: two coroutines
running concurrently must each see their own identity, never each other's.
동시에 실행되는 두 요청이 서로의 신원을 보지 않는지가 핵심 검증 대상.
"""

from __future__ import annotations

import asyncio

from app import user_context
from app.config import settings


def test_falls_back_to_settings_when_unset():
    """Pre-migration behavior for everything outside a request — CLI,
    backtest, tests. This is what keeps those paths unchanged."""
    assert user_context.get_current_user_id() == settings.user_id


def test_user_scope_binds_and_restores():
    before = user_context.get_current_user_id()
    with user_context.user_scope("alice"):
        assert user_context.get_current_user_id() == "alice"
    assert user_context.get_current_user_id() == before


def test_user_scope_restores_on_exception():
    before = user_context.get_current_user_id()
    try:
        with user_context.user_scope("bob"):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert user_context.get_current_user_id() == before


def test_nested_scopes_restore_inner_first():
    with user_context.user_scope("outer"):
        with user_context.user_scope("inner"):
            assert user_context.get_current_user_id() == "inner"
        assert user_context.get_current_user_id() == "outer"


async def test_concurrent_tasks_do_not_leak_identity():
    """The multi-tenancy guarantee. Each task sets its own identity and then
    yields control repeatedly, so the scheduler interleaves them — a plain
    module-level global would fail this."""
    observed: dict[str, list[str]] = {}

    async def run_as(user_id: str) -> None:
        with user_context.user_scope(user_id):
            seen = []
            for _ in range(5):
                await asyncio.sleep(0)
                seen.append(user_context.get_current_user_id())
            observed[user_id] = seen

    await asyncio.gather(run_as("alice"), run_as("bob"), run_as("carol"))

    assert observed["alice"] == ["alice"] * 5
    assert observed["bob"] == ["bob"] * 5
    assert observed["carol"] == ["carol"] * 5


async def test_identity_does_not_escape_the_task_that_set_it():
    async def sets_identity() -> None:
        user_context.set_current_user_id("dave")

    before = user_context.get_current_user_id()
    await asyncio.create_task(sets_identity())
    # create_task copies the context, so the child's set() is invisible here.
    assert user_context.get_current_user_id() == before


def test_set_reset_pair():
    before = user_context.get_current_user_id()
    token = user_context.set_current_user_id("erin")
    assert user_context.get_current_user_id() == "erin"
    user_context.reset_current_user_id(token)
    assert user_context.get_current_user_id() == before
