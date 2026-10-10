"""Account-mode binding + the KIS-configured gate.

Replaces `app.auth.dependencies` (deleted 2026-08-07 single-account rollback,
see docs/single-user-rollback-guide.md Phase 3). That module bound a
per-request IDENTITY resolved from a session cookie; this deployment has no
login any more, so `app.user_context.get_current_user_id()` simply falls back
to `settings.user_id` for every request — nothing needs to bind it.

What still has to be bound per request is the account MODE (paper vs real):
it scopes every persistence key (ledger rows, allowlists, kill-switch state),
so it must be resolved before any handler runs, same as before. Unlike
identity, mode is not free to read on every request — `resolve_account_mode`
touches Postgres for the stored preference — so it is cached process-wide and
only invalidated by an actual mode switch (`invalidate_cached_mode`).
"""

from __future__ import annotations

import logging

from fastapi import HTTPException, Request, status

from app.config import current_account_mode, reset_current_account_mode, set_current_account_mode, settings

logger = logging.getLogger(__name__)

_cached_mode: str | None = None
# Bumped by every invalidation. A resolve that started before an invalidation
# must not write its now-stale result back into the cache: `stored_preference()`
# is a real Postgres read, and a read whose snapshot predates PUT
# /api/account-mode's COMMIT legitimately returns the PRE-switch value (READ
# COMMITTED). Because this cache has no TTL — `invalidate_cached_mode()` is the
# only thing that ever clears it, and it has already fired by then — such a
# write-back pinned the old mode process-wide until the next switch or a
# restart, silently making "effective_from: next request" a lie for every
# subsequent request, not just the one that raced.
# TTL이 없는 캐시라 뒤늦게 돌아온 예전 값이 한 번 덮어쓰면 다음 전환이나
# 재시작 전까지 영구히 고정됨.
_cache_generation: int = 0
# Bound: a switch racing every single attempt is not a realistic steady state
# (it would mean someone is flipping the account mode multiple times per
# request's own DB round trip), only a safety net against looping forever if
# that ever somehow happened.
_MAX_RESOLVE_ATTEMPTS = 5


async def bind_account_mode(request: Request, call_next):
    """Resolve and bind the active account mode for this request.

    Cached at module scope rather than re-resolved per request: unlike the
    old per-user session lookup (a cheap indexed row read), this reads the
    `users` table's stored preference and is not worth a DB round trip on
    every single request for a value that only changes when someone
    explicitly switches modes. `invalidate_cached_mode()` is what makes a
    switch actually take effect.
    요청마다 다시 조회하지 않음 — 모드는 명시적으로 전환할 때만 바뀌므로 캐시.

    A failure reading the stored preference degrades to
    `settings.kis_account_mode` (the deployment default) FOR THIS REQUEST
    ONLY — the failure is never cached, so the next request tries the real
    lookup again rather than being stuck on the fallback until a restart.
    Same "DB outage degrades to the configured default, does not take the
    request down" convention as risk_rules/runtime_settings/
    trade_allowlist_store; unlike those, this used to have no such guard
    (an unhandled exception here would 500 every single request, including
    static asset loads, until Postgres came back) — the old session-cookie
    middleware this replaces degraded the same way (an unresolvable session
    was just treated as "no session"), so this restores that property rather
    than introducing a new one.
    조회 실패는 이번 요청만 배포 기본값으로 처리하고 캐시하지 않음 — 다음
    요청은 다시 정식 조회를 시도함. DB 장애가 요청 전체를 500으로 죽이던
    문제를 막기 위한 것으로, 예전 세션 미들웨어도 조회 실패를 '세션 없음'으로
    우아하게 처리했던 것과 같은 성질을 되살린 것."""
    # Imported inside the function, not at module scope: app.account_mode's
    # `resolve_account_mode`/`stored_preference` are exactly what a test needs
    # to monkeypatch (same convention as app.user_context.serviced_scopes,
    # for the same reason) — a top-level `from ... import` here would bind
    # this module's OWN reference at import time, and patching
    # app.account_mode's attribute afterward would not reach it.
    # 모듈 최상단 import는 시점에 참조를 고정시켜 테스트에서 패치해도 반영되지
    # 않음 — serviced_scopes()와 같은 이유로 함수 내부 import를 씀.
    from app.account_mode import resolve_account_mode, stored_preference

    global _cached_mode
    mode = _cached_mode
    if mode is None:
        for attempt in range(_MAX_RESOLVE_ATTEMPTS):
            generation = _cache_generation  # captured BEFORE the awaits below
            try:
                mode = await resolve_account_mode(settings.user_id, await stored_preference())
            except Exception:  # noqa: BLE001 - see docstring; never take the request down over this
                logger.warning(
                    "could not read the stored account-mode preference; using the deployment "
                    "default for this request only",
                    exc_info=True,
                )
                mode = settings.kis_account_mode
                break
            if generation == _cache_generation:
                _cached_mode = mode
                break
            # A switch committed (and called invalidate_cached_mode(), bumping
            # the generation) while this read was in flight — `mode` is now
            # the PRE-switch value. Re-resolve rather than bind it: the switch
            # already committed its write to Postgres before invalidating, so
            # the next attempt's read is guaranteed to observe the new value
            # (or a LATER switch's, which is still more current than what this
            # attempt just read) — this request ends up bound to the CURRENT
            # mode instead of momentarily executing under the one that was
            # just switched away from.
            # 전환이 이 조회 도중에 커밋됨 — 재조회해서 최신 모드로 반영.
            logger.info(
                "account mode changed while resolving it for this request (generation %d -> %d, "
                "read %r) — re-resolving so this request binds the CURRENT mode",
                generation, _cache_generation, mode,
            )
        else:
            # Exhausted retries — not a realistic steady state (would mean the
            # mode was switched on every single one of this request's own DB
            # round trips), only a safety net against looping forever. Bind
            # the last read rather than the deployment default: it is still a
            # real (if possibly just-superseded) resolved mode, not a guess.
            logger.warning(
                "account-mode resolution kept racing a switch after %d attempts — "
                "binding the last value read (%r) rather than retrying forever",
                _MAX_RESOLVE_ATTEMPTS, mode,
            )
    token = set_current_account_mode(mode)
    try:
        return await call_next(request)
    finally:
        reset_current_account_mode(token)


def invalidate_cached_mode() -> None:
    """Drop the cached mode so the next request re-resolves it.

    Must be called whenever the stored preference changes (`PUT
    /api/account-mode`) — otherwise every request keeps binding the mode that
    was cached at boot (or at the last switch) regardless of what was just
    saved, and the switch response's "effective_from: next request" promise
    is a lie.
    전환 직후 반드시 호출 — 안 하면 다음 요청도 예전에 캐시된 모드로 바인딩됨.

    Also bumps `_cache_generation` (2026-08-26) — a bare `_cached_mode = None`
    could be undone by a read that started before this call and is still in
    flight (see `bind_account_mode`'s own comment): that read's eventual
    write-back would re-populate the cache with the stale pre-switch value,
    and since this cache has no TTL, nothing would ever clear it again short
    of another switch or a restart."""
    global _cached_mode, _cache_generation
    _cached_mode = None
    _cache_generation += 1


async def require_kis() -> None:
    """Route dependency for anything that needs a live KIS account behind it:
    412 when this deployment has no credentials configured for the active
    mode. Replaces `app.auth.dependencies.require_credentials` — there is no
    session to check any more (401 is gone), only whether KIS itself is
    usable.

    **The `ensure_market_feed()` call below is not incidental — do not
    delete it as leftover auth plumbing.** This is the earliest point a
    request is known to need a working KIS account, which is exactly what
    the real-time circuit-breaker/VI feed needs to authenticate. Skipping it
    leaves `kis_market_status` reporting every circuit-breaker check as
    "unknown", which `kis_bridge`'s gate 7 fails CLOSED on — every domestic
    order blocked, silently, with nothing in the order-rejection message
    pointing here. See docs/single-user-rollback-guide.md §5
    invariant #2.
    이 함수 안의 ensure_market_feed() 호출을 인증 잔재로 오인해 지우지 말 것 —
    지우면 서킷브레이커 상태가 '불명'이 되어 국내 주문이 전량 차단됨."""
    mode = current_account_mode()
    if mode not in settings.configured_kis_modes():
        raise HTTPException(
            status.HTTP_412_PRECONDITION_FAILED,
            f"no {mode} KIS credentials configured on this deployment — check the Jenkins "
            f"env for KIS_APP_KEY_{mode.upper()}/KIS_APP_SECRET_{mode.upper()}/KIS_ACCOUNT_8_{mode.upper()}",
        )

    # Imported inside the function on purpose: app.web.state imports most of
    # the application, so a module-level import here would close an import
    # cycle (this module ends up imported from app.web.app, which also wires
    # app.web.state).
    # 함수 내부 import — 모듈 수준으로 올리면 순환 import가 생김.
    from app.web.state import ensure_market_feed

    await ensure_market_feed()
