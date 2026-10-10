"""KIS (한국투자증권) Open API client — paper (모의) by default, real (실전)
per user, real or paper by the credentials each account holds (see broker_for_user below).

`KISBroker` composes its ~35 domestic/overseas REST methods from mixins
under app/brokers/kis/domestic/ and app/brokers/kis/overseas/ (account,
order, quote, ranking, finance, info) on top of `KISHttpClient`
(app/brokers/kis/client.py), which owns the token-backed auth headers and
the throttled `_request`/`_checked_body` plumbing. This module is now just
the composition point plus a couple of module-level helpers.

Design notes:
- Async httpx to match the rest of the project (the KIS sample repo is
  sync `requests` and reads `~/KIS/config/kis_devlp.yaml`; we read credentials
  from Settings.kis_credentials() — env fields Jenkins injects at deploy
  time, see docs/single-user-rollback-guide.md §1/§3).
- Access tokens live 1 day and issuance is rate-limited (1/min), so the
  token is cached to a gitignored JSON file next to the project and reused
  across processes (daemon + CLI runs share it). One file per credential set,
  keyed by a hash of the app key — see `token_cache_path_for`; a single
  shared file would make two users invalidate each other's token and then
  both hit that 1/min limit on the retry.
- The paper server returns at most 20 positions per balance call; the
  CTX_AREA_FK100/NK100 continuation loop handles the rest.
"""

from __future__ import annotations

from app.brokers.kis.client import KISHttpClient
from app.brokers.kis.const import token_cache_path_for
from app.brokers.kis.domestic.account import DomesticAccountMixin
from app.brokers.kis.domestic.order import DomesticOrderMixin
from app.brokers.kis.domestic.quote import DomesticQuoteMixin
from app.brokers.kis.domestic.ranking import DomesticRankingMixin
from app.brokers.kis.domestic.finance import DomesticFinanceMixin
from app.brokers.kis.domestic.info import DomesticInfoMixin
from app.brokers.kis.overseas.account import OverseasAccountMixin
from app.brokers.kis.overseas.info import OverseasInfoMixin
from app.brokers.kis.overseas.order import OverseasOrderMixin
from app.brokers.kis.overseas.quote import OverseasQuoteMixin
from app.brokers.kis.overseas.ranking import OverseasRankingMixin

from app.config import current_account_mode, settings
from app.domain.models import KISCredentials


class KISBroker(
    KISHttpClient,
    DomesticAccountMixin,
    DomesticOrderMixin,
    DomesticQuoteMixin,
    DomesticRankingMixin,
    DomesticFinanceMixin,
    DomesticInfoMixin,
    OverseasAccountMixin,
    OverseasOrderMixin,
    OverseasQuoteMixin,
    OverseasRankingMixin,
    OverseasInfoMixin,
):
    def __init__(
        self,
        *,
        app_key: str,
        app_secret: str,
        account_8: str,
        account_prod: str = "01",
        **client_kwargs,
    ) -> None:
        """`client_kwargs` (base_url, token_cache_path, client,
        min_request_interval, account_min_request_interval,
        max_requests_per_second) pass straight through to
        KISHttpClient.__init__ — see there for defaults."""
        super().__init__(app_key=app_key, app_secret=app_secret, **client_kwargs)
        self._account_8 = account_8
        self._account_prod = account_prod


def krx_code_from_ticker(ticker: str) -> str | None:
    """"005930.KS" -> "005930", or None for anything else (a US symbol, an
    unrecognized suffix). Shared by app.execution.kis_bridge (real order
    routing) and app.live.kis_market_status (WebSocket subscriptions) —
    lives here, not in kis_bridge, so the "live" module doesn't have to
    depend on the "execution" module for a plain string-parsing utility."""
    if ticker.endswith(".KS") and ticker[:-3].isdigit():
        return ticker[:-3]
    return None


def broker_from_credentials(creds: KISCredentials) -> KISBroker:
    """The one place a credential set becomes a broker. Pure — no I/O, no
    settings lookup for anything secret — so it is trivially testable and has
    exactly one caller path that can pick the wrong account.

    `is_real` comes from the credentials' own account_mode, never from
    current_account_mode(). That distinction is the whole point: the mode
    is a property of the credentials that were loaded, so a real-account
    credential set cannot be driven with paper TR IDs (or vice versa) by a
    stale global. It is also what every mixin's self._tr(real_id, paper_id)
    reads off the instance.
    실전/모의 판정을 전역 설정이 아니라 불러온 자격증명 자체에서 가져옴 —
    설정이 어긋나 있어도 TR ID와 계좌가 갈라지지 않도록."""
    return KISBroker(
        app_key=creds.app_key,
        app_secret=creds.app_secret,
        account_8=creds.account_8,
        account_prod=creds.account_prod,
        base_url=settings.kis_real_base_url if creds.is_real else settings.kis_paper_base_url,
        is_real=creds.is_real,
        # Intervals keyed off creds.is_real, same source as base_url/is_real
        # above — not settings.kis_account_mode (env). A config/DB mismatch on
        # active mode used to be able to run a real process on paper timing
        # (memory: real-account-switch-checklist-2026-08-31 #2).
        # 요청 간격도 자격증명에서 — 설정 불일치로 실전이 모의 타이밍 쓰던 잠재버그 제거.
        min_request_interval=settings.kis_min_request_interval_for(creds.is_real),
        account_min_request_interval=settings.kis_account_inquiry_min_interval_for(creds.is_real),
        # Rolling 1s-window request cap (0 = disabled) — keyed off creds.is_real
        # like the two intervals above. The min interval alone bounds only the
        # mean rate; a busy event loop still bursts grants to the wire and trips
        # EGW00201 (2026-09-03 decision id=1179).
        max_requests_per_second=settings.kis_max_requests_per_second_for(creds.is_real),
        # Per credential set, not one shared file — see token_cache_path_for.
        token_cache_path=token_cache_path_for(creds.app_key),
    )


async def broker_for_user(user_id: str, account_mode: str | None = None) -> KISBroker | None:
    """Build this account's broker, or None when this deployment has not
    configured that mode. `user_id` is accepted (and stays part of the
    signature — `app.brokers.get_broker()` and every caller pass it) for a
    single-account deployment there is only ever one value, `settings.
    user_id`; kept `async def` even though the lookup itself is now sync
    (env fields, no I/O) so no caller across the codebase needs to change.

    None means "this deployment has no KIS credentials for this mode" and
    callers degrade quietly, same convention as before — see
    Settings.kis_credentials's own docstring on why it never falls back to
    the other mode's values.

    account_mode defaults to current_account_mode(), the per-request/context
    ContextVar (paper vs real — see app.account_mode). Callers that care
    should pass it explicitly rather than assume.

    Prefer `app.brokers.get_broker()` over calling this directly — it caches,
    and the cache is what makes the per-instance request throttle actually
    shared (see that module).
    """
    mode = account_mode or current_account_mode()
    creds = settings.kis_credentials(mode)
    if creds is None:
        return None
    return broker_from_credentials(creds)
