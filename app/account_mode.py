# app/account_mode.py
# Which account mode a person is operating in. The ContextVar plumbing lives in
# app.config (see current_account_mode there) purely to avoid an import cycle —
# Settings itself needs to read it; this module is the policy layer on top.
#
# Policy, per the 2026-08-06 product decision: **every user may use both 실전
# and 모의투자.** Real credentials are required at onboarding; paper credentials
# are optional. Which of the two is active is the user's own preference,
# recorded per user, so two people served by the same process can be in
# different modes at the same time — which is the whole reason
# current_account_mode() exists rather than one process-wide setting.
# 모든 사용자가 두 모드를 다 쓸 수 있고, 활성 모드는 사용자별 선호로 결정됨.
#
# Note on docs/safety-and-risk.md §5.1, which forbids exposing account-mode
# selection through the HTTP API: that rule was written when the dashboard was
# unauthenticated and single-user, so "anything on localhost could flip the
# whole system to real trading". Neither premise holds now — the API requires a
# session, and a preference only ever moves the person who set it. What §5.1
# was really protecting (nobody reaching real money without deliberately
# choosing it) is preserved differently here: a user only reaches real mode by
# entering real credentials, and switching is an explicit act. §5.1 needs
# updating to say that; see the migration plan.
# §5.1의 전제(무인증·단일 사용자)가 사라졌으므로 문서 갱신 필요.

from __future__ import annotations

import asyncio
import logging
import socket

import asyncpg

from app.config import settings

logger = logging.getLogger(__name__)

ACCOUNT_MODES = ("paper", "real")

# Audit trail for account-mode switches (2026-08-07 single-account rollback,
# see docs/single-user-rollback-guide.md §7-1/Phase 3.4). This
# deployment has no login any more, so `PUT /api/account-mode` is reachable
# by anyone who can reach the dashboard at all — the tradeoff accepted for
# keeping the mode toggle in the UI (docs/safety-and-risk.md §5.1's original
# concern, revived) is that every switch leaves a row here.
#
# `client_host` cannot identify a person by itself: everyone connects through
# the same SSH tunnel and shows up as 127.0.0.1 (see
# docs/single-user-rollback-guide.md Phase 3.4's own note). This
# table is an anchor to cross-reference against the server's own SSH access
# log by timestamp, not a standalone answer to "who did this" — same
# limitation as app.user_context.record_session_start's identity_claims,
# and for the same reason (no real login to check against).
# client_host만으로는 사람을 특정할 수 없음 — SSH 접속 로그와 시각을 대조하기
# 위한 앵커일 뿐, identity_claims와 같은 한계.
_SWITCH_SCHEMA = """
CREATE TABLE IF NOT EXISTS account_mode_switches (
    id           SERIAL PRIMARY KEY,
    user_id      TEXT NOT NULL REFERENCES users(user_id),
    from_mode    TEXT,
    to_mode      TEXT NOT NULL,
    hostname     TEXT NOT NULL,
    client_host  TEXT,
    switched_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""

_switch_schema_ready = False
_switch_schema_lock = asyncio.Lock()

# Preference order used when a person has NOT chosen: paper first, so someone
# who configured both and never picked is not silently placed on real money.
_UNCHOSEN_PREFERENCE = ("paper", "real")


def allowed_modes_for(email: str | None = None) -> tuple[str, ...]:
    """Which modes a person may configure. Both, for everyone.

    Kept as a function rather than inlining the constant because it is the one
    place a future restriction (per-plan, per-role) would go, and every caller
    already routes through it.
    현재는 모두 허용 — 향후 제한이 생기면 이 함수만 바꾸면 됨."""
    return ACCOUNT_MODES


def is_valid_mode(account_mode: str) -> bool:
    return account_mode in ACCOUNT_MODES


def resolve_preference(preferred: str | None, configured: set[str]) -> str:
    """The mode a person is in, given their stated preference and the modes
    they actually have credentials for.

    A preference is only honoured while it is backed by credentials. Someone
    who preferred 모의투자 and then deleted their paper credentials must not
    keep being scoped to a mode they can no longer authenticate — the ledger
    rows, allowlists and kill-switch state would all be filed under a mode
    whose broker cannot be built.
    자격증명이 없는 모드는 선호로 지정돼 있어도 활성화하지 않음."""
    if preferred and preferred in configured and is_valid_mode(preferred):
        return preferred
    for mode in _UNCHOSEN_PREFERENCE:
        if mode in configured:
            return mode
    # Nothing configured yet — this person is still onboarding. Resolve to
    # 'real', the REQUIRED mode, not to the safer-looking 'paper'. Nothing is
    # reachable in this state either way (require_credentials refuses every
    # gated route), so the only thing this choice affects is what that refusal
    # SAYS: "no real KIS credentials configured" is the actual next step, while
    # "no paper credentials configured" would send someone off to open a
    # 모의투자 account they were never required to have.
    # 미온보딩 상태에서는 필수 모드(실전)로 해석 — 어차피 접근은 막히므로,
    # 안내 문구가 올바른 다음 단계를 가리키게 하는 것이 유일한 차이.
    return "real"


async def resolve_account_mode(user_id: str, preferred: str | None) -> str:
    """`resolve_preference` against this deployment's actually-configured
    credentials.

    `user_id` stays in the signature (every caller passes one) but is unused
    now — single-account deployment, so "whose credentials" isn't a question
    any more; `settings.configured_kis_modes()` answers for the one account
    this process runs. Kept `async def` for the same reason as
    `broker_for_user`: no caller across the codebase needs to change.

    Reading configured_kis_modes() is a sync, in-process check (env fields,
    no I/O), so unlike the old DB-backed version this cannot itself fail —
    the try/except this used to need is gone.
    자격증명이 env로 옮겨가면서 조회가 실패할 일이 없어짐 — 예외 처리 불필요."""
    del user_id  # single-account deployment — see docstring
    return resolve_preference(preferred, settings.configured_kis_modes())


async def stored_preference() -> str | None:
    """This deployment's last-chosen account mode, or None if nobody has
    switched yet. Read from the `users` table's single row (one row per the
    2026-08-07 single-account rollback — see
    docs/single-user-rollback-guide.md Phase 3.1/4).

    This survives the login/session removal on purpose: the point of storing
    it was never "whose preference is this" (there is one account), it was
    "don't snap back to real mode on every restart" — a person switches to
    paper for testing, restarts the dashboard for a deploy, and should not
    silently find themselves back on the real account. `set_account_mode`
    (app/web/routers/credentials.py) is what writes this column; nothing
    downstream needs to change for this function to keep answering
    correctly.
    로그인이 사라져도 이 컬럼은 남는다 — 목적이 '누구의 선호'가 아니라
    '재시작할 때마다 실전으로 되돌아가지 않기'였기 때문."""
    from app.memory.db import get_pool

    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT preferred_account_mode FROM users WHERE user_id = $1", settings.user_id
        )
    return row["preferred_account_mode"] if row else None


async def set_stored_preference(account_mode: str) -> None:
    """Persist this deployment's chosen account mode — the write side of
    `stored_preference()`. Called by `PUT /api/account-mode`
    (app/web/routers/credentials.py) after confirming the mode is actually
    configured; this function does not re-check that, on purpose, so it stays
    a pure write and the caller owns the policy decision."""
    if not is_valid_mode(account_mode):
        raise ValueError(f"unknown account mode {account_mode!r}")
    from app.memory.db import get_pool

    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE users SET preferred_account_mode = $1 WHERE user_id = $2",
            account_mode, settings.user_id,
        )


async def _ready_switch_pool() -> asyncpg.Pool:
    """Double-checked locking around the first-time DDL — same pattern as
    app.user_context._ready_pool. `users` (the FK target) is created earlier
    by app.memory.db's own _SCHEMA, which get_pool() always runs first."""
    global _switch_schema_ready
    from app.memory.db import get_pool

    pool = await get_pool()
    if _switch_schema_ready:
        return pool
    async with _switch_schema_lock:
        if not _switch_schema_ready:
            async with pool.acquire() as conn:
                await conn.execute(_SWITCH_SCHEMA)
            _switch_schema_ready = True
    return pool


async def record_mode_switch(from_mode: str | None, to_mode: str, client_host: str | None) -> None:
    """Best-effort audit row for one account-mode switch — see
    `_SWITCH_SCHEMA`'s own comment on why this exists and what it cannot
    prove by itself.

    Best-effort, same convention as app.user_context.record_session_start:
    a failure here must never block the switch itself, and degrades to a
    warning log rather than a 500 the caller cannot act on.
    베스트에포트 — 기록 실패가 전환 자체를 막으면 안 됨."""
    try:
        pool = await _ready_switch_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO account_mode_switches (user_id, from_mode, to_mode, hostname, client_host) "
                "VALUES ($1, $2, $3, $4, $5)",
                settings.user_id, from_mode, to_mode, socket.gethostname(), client_host,
            )
    except Exception:  # noqa: BLE001 - forensic best-effort, see docstring
        logger.warning("could not record account-mode switch audit row", exc_info=True)
