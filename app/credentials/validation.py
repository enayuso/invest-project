"""Prove a credential set actually works before storing it (Phase 3).

A typo'd app secret stored successfully is the worst outcome available here:
the save reports success, the onboarding gate clears, and the person only
discovers the problem later as a wall of "KIS call failed" warnings with
nothing pointing at the cause. Issuing a real token is the only check that
distinguishes "these credentials are right" from "these are 32 plausible
characters", so save goes through here first.
저장은 성공했는데 값이 틀린 경우가 가장 나쁨 — 온보딩은 통과하고 실패는 한참
뒤에 엉뚱한 곳에서 드러나므로, 저장 전에 실제 토큰 발급으로 검증함.

**The issued token is deliberately kept.** KIS rate-limits issuance to one per
minute per app key, so a validate-then-discard would burn the allowance and
leave the very next real call to trip the limit. Writing to
`token_cache_path_for(app_key)` — the same file the broker will read — means
validation warms the cache instead of competing with it, and re-saving the
same credentials inside the token's lifetime re-uses it rather than issuing
again.
검증용 토큰을 버리지 않고 브로커가 쓸 캐시 파일에 그대로 남김 — 발급이 앱키당
분당 1회로 제한되므로, 버리면 바로 다음 호출이 제한에 걸림.
"""

from __future__ import annotations

import logging

import httpx

from app.brokers.kis.auth import KISAuth
from app.brokers.kis.const import KISBrokerError, is_transient_token_failure, token_cache_path_for
from app.config import settings

logger = logging.getLogger(__name__)


class CredentialsRejected(Exception):
    """KIS would not issue a token for this app key / secret pair."""


class CredentialCheckUnavailable(Exception):
    """KIS could not be reached to check. Distinct from CredentialsRejected on
    purpose: one means "your keys are wrong", the other means "we don't know
    yet", and telling a user the first when it was the second sends them off
    to re-copy credentials that were fine all along.
    '틀렸다'와 '확인 불가'를 구분 — 후자를 전자로 표시하면 멀쩡한 값을 다시
    입력하게 만듦."""


async def verify_kis_credentials(
    app_key: str, app_secret: str, account_mode: str, *, transport: httpx.BaseTransport | None = None
) -> None:
    """Return normally if KIS accepts these credentials.

    Only the app key and secret are checkable this way — token issuance does
    not involve the account number, so a wrong `account_8` still passes here
    and surfaces on the first balance call instead. Worth knowing before
    trusting a green check too far.
    계좌번호는 토큰 발급에 쓰이지 않으므로 이 검증으로는 확인되지 않음.

    `transport` is a test-only seam (httpx.MockTransport) — left None in
    production, which gets httpx's normal transport unchanged.
    """
    base_url = settings.kis_real_base_url if account_mode == "real" else settings.kis_paper_base_url
    async with httpx.AsyncClient(timeout=15.0, transport=transport) as client:
        # KISAuth construction lives inside the `async with` (not before it)
        # so that if it — or token_cache_path_for(app_key) above it — ever
        # raises, the client this scope owns still gets closed.
        auth = KISAuth(
            app_key=app_key,
            app_secret=app_secret,
            base_url=base_url,
            token_cache_path=token_cache_path_for(app_key),
            client=client,
        )
        try:
            # _get_token, not _issue_token: it checks the cache first, so
            # re-saving unchanged credentials within the token's lifetime costs
            # no issuance at all.
            await auth._get_token()
        except KISBrokerError as exc:
            if is_transient_token_failure(exc):
                # KIS answers a 5xx outage, a token-issuance rate limit, and a
                # genuinely bad app key/secret with the exact same exception
                # type — the only way to tell them apart is msg_cd/status_code
                # (see is_transient_token_failure). Misclassifying this branch
                # as CredentialsRejected sends whoever's on call chasing a
                # typo in Jenkins Credentials that was never there.
                logger.warning(
                    "KIS could not be checked for %s credentials — transient "
                    "failure (msg_cd=%s, HTTP %s), not a credentials problem",
                    account_mode, exc.msg_cd or "-", exc.status_code, exc_info=True,
                )
                raise CredentialCheckUnavailable(str(exc)) from exc
            # KIS answers a bad key/secret with a 403 and its own error body,
            # which KISAuth has already truncated. Surfaced to the user as-is
            # because KIS's message ("유효하지 않은 AppKey") is more useful than
            # anything this layer could invent.
            logger.info("KIS rejected %s credentials during validation", account_mode)
            raise CredentialsRejected(str(exc)) from exc
        except httpx.HTTPError as exc:
            logger.warning("could not reach KIS to validate %s credentials", account_mode, exc_info=True)
            raise CredentialCheckUnavailable(str(exc)) from exc
        except Exception as exc:  # noqa: BLE001 — deliberate: see below
            # Anything not already classified above (e.g. resp.json() giving
            # a non-dict, or some future KISAuth change) must still resolve
            # to "couldn't check" rather than propagate and take the whole
            # boot (app.web.state.lifespan) down with it — the documented
            # contract of this function's two exception types is "Rejected
            # or Unavailable, never anything else". Full traceback goes to
            # the log; callers only see the summarized CredentialCheckUnavailable.
            logger.exception("unexpected error while checking %s credentials", account_mode)
            raise CredentialCheckUnavailable(f"unexpected error: {exc!r}") from exc
