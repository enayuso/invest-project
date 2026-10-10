# app/user_context.py
# Identity seam. Originally (agent-architecture-roadmap stage 4) this just
# returned settings.user_id — one process per person, no real login — with
# the stated intent that "adopting real auth later means changing this one
# function's body, not every call site." That is exactly what happened: as of
# the multi-user migration (docs/multi-user-credentials-plan.md Phase 1) the
# answer comes from a ContextVar that the web layer sets per REQUEST, so one
# process can serve many people concurrently.
#
# The fallback to settings.user_id is load-bearing, not vestigial. Everything
# that runs outside a request — the CLI entry points (app.main, app.backtest,
# app.memory, app.execution), the backtest harness, the test suite — has no
# request to inherit an identity from, and must keep behaving exactly as it
# did before. Those paths read the ContextVar's default, which is
# settings.user_id, which still defaults to 'default'.
#
# ContextVar (not a global, not a thread-local) is what makes this correct
# under asyncio: each request's handler and every coroutine it awaits sees its
# own value, with no leakage between concurrently-running requests, which a
# plain module global would not give us.
# 요청마다 ContextVar로 사용자 식별 — 요청 밖(CLI/백테스트/테스트)에서는
# 기존처럼 settings.user_id로 폴백하므로 동작이 바뀌지 않음.

from __future__ import annotations

import asyncio
import contextvars
import logging
import os
import socket
from collections.abc import Iterator
from typing import NamedTuple
from contextlib import contextmanager

import asyncpg

from app.config import settings
from app.memory.db import get_pool

logger = logging.getLogger(__name__)

DEFAULT_USER_ID = "default"

# None means "nothing set an identity for this context" — resolved lazily to
# settings.user_id by get_current_user_id() rather than being baked in as the
# default here, because settings.user_id can be reassigned at runtime (tests
# do exactly this) and a default captured at import time would go stale.
_current_user_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "current_user_id", default=None
)

# Purely forensic (see record_session_start's own docstring for why this
# can't PREVENT anything under a shared-OS-account deployment) — a trail of
# which user_id each process claimed, from where, and when, to check after
# the fact instead of trying to block impersonation up front (that would
# need real login; see project memory: user-account-mode-db-separation-design).
_SCHEMA = """
CREATE TABLE IF NOT EXISTS identity_claims (
    id         SERIAL PRIMARY KEY,
    user_id    TEXT NOT NULL REFERENCES users(user_id),
    hostname   TEXT NOT NULL,
    pid        INTEGER NOT NULL,
    started_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""

_schema_ready = False
_schema_lock = asyncio.Lock()


def get_current_user_id() -> str:
    """Whose data is this. Set per-request by the web layer; falls back to
    settings.user_id everywhere else (CLI, backtest, tests). Still a valid
    zero-arg FastAPI dependency, as the original design intended."""
    return _current_user_id.get() or settings.user_id


def set_current_user_id(user_id: str) -> contextvars.Token:
    """Bind an identity to the current context. Returns the token needed to
    restore the previous value — prefer `user_scope()` unless you genuinely
    need to set and reset across separate call sites (ASGI middleware does)."""
    return _current_user_id.set(user_id)


def reset_current_user_id(token: contextvars.Token) -> None:
    _current_user_id.reset(token)


@contextmanager
def user_scope(user_id: str) -> Iterator[str]:
    """Run a block as `user_id`, restoring the previous identity afterwards.

    This is what background work uses. app.web.state's stop_loss_loop /
    auto_trade_loop iterate over users and must re-bind identity per user per
    iteration — without it, every row those loops write would be stamped with
    whatever identity happened to be current, which outside a request is
    settings.user_id ('default') for every user alike.
    백그라운드 루프는 사용자마다 이 스코프 안에서 실행해야 함."""
    token = _current_user_id.set(user_id)
    try:
        yield user_id
    finally:
        _current_user_id.reset(token)


class ServicedScope(NamedTuple):
    """One (user, account_mode) the background loops should work on.

    A pair, not a bare user id: every store the loops touch is keyed by both,
    so a loop that binds only the identity writes into whatever mode the
    process happens to default to. Matches the key shape of
    `config._scoped_lists`, `risk_rules._overrides` and the persistence tables.
    사용자와 모드를 함께 다뤄야 함 — 모든 스토어가 두 값으로 키를 잡음."""

    user_id: str
    account_mode: str


async def serviced_scopes() -> list[ServicedScope]:
    """Whose behalf the background loops run, one pass per scope.

    Single-account deployment (2026-08-07 rollback — see
    docs/single-user-rollback-guide.md Phase 4): there is exactly one
    scope, `settings.user_id` at its resolved mode, whenever this deployment
    has KIS credentials configured for at least one mode. This used to be a
    `user_broker_credentials JOIN users` query that could return any number
    of scopes (the multi-user migration's Phase 6) — the KEEP here is the
    *shape* of the answer (`list[ServicedScope]`), not that query, so every
    caller (the poller, stop_loss_loop, auto_trade_loop, lifespan's allowlist
    restore, the real-time feed's scope pick) needed no changes at all.
    반환 타입(list[ServicedScope])은 유지 — 호출부는 그대로 두고 내용만 단일
    계정 기준으로 바꿈.

    **No configured credentials, or a DB failure reading the stored
    preference, both return an empty list — same deliberate choice as
    before.** An empty list means the loops idle for one interval and try
    again, which is a real cost (a missed stop-loss check), but guessing
    which mode to trade in is a worse one.
    자격증명 미설정이거나 선호 모드 조회가 실패하면 빈 목록 — 추측해서 도는
    것보다 한 주기 쉬는 편이 낫음.

    **This function returning an empty list is not an error — it will not
    raise or log at error level for the ordinary "not configured yet" case.**
    If the background loops (poller, stop-loss, auto-trade, real-time feed)
    all appear to be doing nothing, check the boot log for "serviced scopes:
    []" or "no KIS credentials configured" rather than assuming a crash —
    see docs/single-user-rollback-guide.md §5 invariant #1."""
    from app.account_mode import resolve_account_mode, stored_preference

    if not settings.configured_kis_modes():
        logger.info("no KIS credentials configured — background loops idle this interval")
        return []

    try:
        preferred = await stored_preference()
    except Exception:  # noqa: BLE001 - see docstring; idling beats guessing
        logger.warning(
            "could not read the stored account-mode preference; loops idle this interval",
            exc_info=True,
        )
        return []

    mode = await resolve_account_mode(settings.user_id, preferred)
    scopes = [ServicedScope(settings.user_id, mode)]
    logger.info("serviced scopes: %s", [f"{s.user_id}/{s.account_mode}" for s in scopes])
    return scopes


async def background_user_ids() -> list[str]:
    """Deprecated shim — the user ids of `serviced_scopes()`.

    Kept only because dropping a user's mode is exactly the bug Phase 6 had to
    fix; anything still calling this is binding an identity without a mode and
    should move to `serviced_scopes()`.
    모드가 빠지므로 신규 호출부에서는 serviced_scopes()를 쓸 것."""
    return [scope.user_id for scope in await serviced_scopes()]


async def _ready_pool() -> asyncpg.Pool:
    """Double-checked locking around the first-time DDL, same pattern as
    app.execution.ledger._ready_pool — `users` (referenced by the FK above)
    is created earlier, in app.memory.db's own _SCHEMA, which get_pool()
    always runs through first."""
    global _schema_ready
    pool = await get_pool()
    if _schema_ready:
        return pool
    async with _schema_lock:
        if not _schema_ready:
            async with pool.acquire() as conn:
                await conn.execute(_SCHEMA)
            _schema_ready = True
    return pool


async def register_user(user_id: str, display_name: str) -> None:
    """Idempotent upsert into the `users` table (app.memory.db._SCHEMA) —
    insert a new person or update an existing one's display name. Call once
    per person when wiring up their instance's config (e.g. from
    app.web.state.lifespan), not on every request; safe to call repeatedly."""
    pool = await _ready_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO users (user_id, display_name)
            VALUES ($1, $2)
            ON CONFLICT (user_id) DO UPDATE SET display_name = EXCLUDED.display_name
            """,
            user_id,
            display_name,
        )


async def record_session_start() -> None:
    """Best-effort audit trail, not an access control — logs which user_id
    THIS process claims to be, plus hostname/pid/timestamp, into
    `identity_claims`. With no real login (see this module's own docstring),
    nothing stops a person on a shared OS account from setting settings.user_id
    to someone else's id; this can only make that check-able after the fact
    (does the claim's timing/host line up with who was actually meant to be
    running that instance then), not prevent it up front. Degrades quietly
    (log + continue) on a DB failure — this must never block startup.
    Call once per process (e.g. app.web.state.lifespan), not per request."""
    try:
        pool = await _ready_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO identity_claims (user_id, hostname, pid) VALUES ($1, $2, $3)",
                get_current_user_id(),
                socket.gethostname(),
                os.getpid(),
            )
    except Exception as exc:  # noqa: BLE001 - forensic best-effort, same convention as app.live.alerts
        logger.warning("could not record identity claim, continuing without it: %s", exc)
