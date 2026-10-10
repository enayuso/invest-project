"""Account-mode status and switching.

Through 2026-08-06 this also owned KIS credential entry/removal (PUT/DELETE
/api/credentials/{mode}) — a fresh deployment had no in-app way to configure
KIS at all, so a screen for it lived here. The 2026-08-07 single-account
rollback (docs/single-user-rollback-guide.md Phase 2) removed that:
this deployment trades one company account and its KIS app key/secret/account
number arrive as env vars that Jenkins injects at deploy time
(Settings.kis_credentials — app/config.py), never typed into a browser. There
is nothing left to PUT or DELETE here.

What remains: telling the frontend which modes are configured (so it can grey
out a mode with no credentials behind it) and letting this deployment switch
which configured mode it operates in. Neither route requires a session any
more (no login — Phase 3 of the same rollback removed it) — reachable by
anyone who can reach the dashboard, i.e. anyone with an SSH tunnel to this
host. That tradeoff is what `record_mode_switch` below exists to offset; see
its own docstring.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel

from app.account_mode import (
    ACCOUNT_MODES,
    is_valid_mode,
    record_mode_switch,
    set_stored_preference,
    stored_preference,
)
from app.brokers import invalidate_broker_cache
from app.config import current_account_mode, settings
from app.web import state
from app.web.account_mode_middleware import invalidate_cached_mode

logger = logging.getLogger(__name__)

router = APIRouter()


class AccountModePayload(BaseModel):
    account_mode: str


@router.get("/api/credentials")
async def credential_status() -> dict:
    """What the frontend renders its account-mode picker from.

    Reading `configured_kis_modes()` is a sync, in-process check now (env
    fields, no ciphertext, no DB round trip) — replaces the old
    app.credentials.store.configured_modes(user_id) call, which could 503 on
    a DB outage. Nothing here can fail that way any more.
    자격증명이 env로 옮겨가면서 이 조회는 DB 왕복이 사라짐 — 실패할 일이 없음."""
    modes = settings.configured_kis_modes()
    # "ready" means the ACTIVE mode (not real specifically) has credentials
    # behind it. This inherited "real" in modes from the multi-user
    # onboarding gate, whose product policy was "실전 필수, 모의투자 선택"
    # for every ONBOARDING USER — real credentials as a precondition for
    # using the dashboard at all. That policy doesn't fit a deployment that
    # deliberately runs paper-only (every local dev instance, per
    # docs/single-user-rollback-guide.md §8.2: "로컬에는 모의계좌
    # 키만") — under the old check, a correctly-configured local dev
    # instance could never pass this gate. current_account_mode() already
    # resolves to paper first when only paper is configured
    # (account_mode._UNCHOSEN_PREFERENCE), so checking THAT mode's presence
    # in `modes` is both what actually matters (can this deployment build a
    # broker for the mode it's about to use) and what makes paper-only dev
    # work.
    # '실전 키가 있는가'가 아니라 '지금 활성 모드에 키가 있는가' — 모의만
    # 설정된 로컬 개발 환경도 정상적으로 통과해야 함.
    active_mode = current_account_mode()
    return {
        "active_mode": active_mode,
        "configured": sorted(modes),
        "ready": active_mode in modes,
        "allowed": list(ACCOUNT_MODES),
        "required": [],
        "optional": list(ACCOUNT_MODES),
        "preferred": await stored_preference(),
        # So the dashboard can warn plainly instead of every real order
        # failing with "estimated order value exceeds cap 0.00" — see
        # app.config._require_real_money_limits_on_real_account, which
        # refuses to boot in real mode without these, but this deployment
        # can still be running in paper mode with real credentials configured
        # and these limits simply never set.
        "real_limits_configured": all(
            getattr(settings, name) is not None
            for name in (
                "kis_max_order_krw_real", "kis_max_order_usd_real",
                "kis_approval_threshold_krw_real", "kis_approval_threshold_usd_real",
            )
        ),
    }


@router.put("/api/account-mode")
async def set_account_mode(payload: AccountModePayload, request: Request) -> dict:
    """Switch which account this deployment operates against.

    No session to check any more — anyone who can reach this dashboard (i.e.
    anyone with an SSH tunnel to this host, see
    docs/single-user-rollback-guide.md §0) can call this. Two things
    keep that from being "anyone can trade on the real account by accident":
    `is_valid_mode`/`configured_kis_modes` below still refuse to switch to a
    mode with no credentials behind it, and every switch is logged to
    `account_mode_switches` for after-the-fact review — see
    `record_mode_switch`'s own docstring on what that log can and cannot
    prove.
    로그인이 없어 세션 확인이 사라짐 — 자격증명 없는 모드로는 전환 불가 +
    모든 전환을 감사로그에 기록하는 것으로 상쇄."""
    if not is_valid_mode(payload.account_mode):
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"unknown account mode {payload.account_mode!r}, expected one of {list(ACCOUNT_MODES)}",
        )
    if payload.account_mode not in settings.configured_kis_modes():
        raise HTTPException(
            status.HTTP_412_PRECONDITION_FAILED,
            f"no {payload.account_mode} credentials configured on this deployment",
        )

    from_mode = current_account_mode()
    await set_stored_preference(payload.account_mode)
    await record_mode_switch(
        from_mode, payload.account_mode, request.client.host if request.client else None
    )

    # Order matters, same reasoning as the old credential-save path this
    # replaces: invalidate the cached mode + broker BEFORE tearing down and
    # rebuilding the real-time feed, so nothing downstream can observe a
    # half-switched state (new mode bound, old broker/feed still live).
    #
    # invalidate_cached_mode() makes the NEXT request re-resolve the mode
    # instead of reusing bind_account_mode's process-wide cache — without
    # it, "effective_from: next request" below would be a lie.
    #
    # invalidate_broker_cache() with no account_mode drops BOTH cached
    # broker instances for this account, not just the new mode's — the old
    # mode's is no longer the active one and holding it open serves nothing.
    #
    # The feed authenticated with the OLD mode's credentials and cannot be
    # re-pointed in place, so it is torn down here and rebuilt lazily by the
    # next call to require_kis() (orders.py/portfolio.py) — rebuilding it
    # eagerly here would mean this response blocks on a KIS token issuance.
    # 모드 캐시 무효화 → 브로커 캐시 폐기 → 피드 중단 순서 — 안 지키면 절반만
    # 전환된 상태(새 모드가 바인딩됐는데 예전 브로커/피드가 살아있음)가 관측될 수 있음.
    invalidate_cached_mode()
    await invalidate_broker_cache(settings.user_id)
    await state.stop_market_feed()

    return {"account_mode": payload.account_mode, "effective_from": "next request"}
