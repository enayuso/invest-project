"""Broker integrations: real (or paper) brokerage accounts behind one thin
interface, so the pipeline/ledger/dashboard never import a vendor SDK
directly. KIS paper trading is the first implementation (`kis/`); adding
another broker later means one new registry entry here, not edits at call
sites — see `get_broker()`.
"""

import asyncio
import logging

from app.brokers.kis.broker import (
    KISBroker,
    broker_for_user,
    broker_from_credentials,
    krx_code_from_ticker,
)
from app.brokers.kis.const import KISBrokerError

logger = logging.getLogger(__name__)

# Name -> async factory (user_id, account_mode) -> broker instance, or None
# when that user hasn't configured that broker.
# 브로커 이름 -> 인스턴스 팩토리 (신규 증권사는 이 딕셔너리에 한 줄 추가)
_BROKER_FACTORIES = {
    "kis": broker_for_user,
}

# (name, user_id, account_mode) -> cached instance (or None), populated lazily.
#
# Every broker instance owns its own request throttle (see
# BaseHttpClient._throttle — a per-instance asyncio.Lock + last-request-time),
# so constructing one per call — as this used to do before caching — silently
# defeated that throttle: each new instance started counting from zero, so
# concurrent callers (poller.py's per-ticker gather, the web routers,
# kis_bridge, kis_provider) never actually serialized against each other and
# could burst past KIS's per-second rate limit (EGW00201).
#
# The key gained user_id and account_mode in the multi-user migration
# (docs/multi-user-credentials-plan.md §4.1). That is NOT a weakening of the
# invariant above — it is the correct granularity. KIS rate-limits per app
# key, and different users have different app keys, therefore different
# buckets. Sharing one throttle across two users' keys would have throttled
# them against each other for no reason while still not protecting either.
# 스로틀은 앱키(=사용자)별 제한을 반영해야 하므로 캐시 키에 user_id를 포함하는
# 것이 오히려 정확함 — 사용자끼리 서로를 불필요하게 지연시키지 않음.
#
# No longer safe to keep forever the way the single-account version was: a
# user can now replace their own credentials at runtime, so a stale instance
# would keep using the old keys. `invalidate_broker_cache` is what the
# credential-save path calls.
_broker_instances: dict[tuple[str, str, str], KISBroker | None] = {}

# Guards construction, not lookup. Without it two coroutines that miss the
# cache for the same key both build an instance, one wins the dict slot, and
# the loser is handed back to its caller as a second live client with its own
# independent throttle — reintroducing exactly the burst this cache exists to
# prevent, but only under concurrency, which is when it matters.
# 캐시 미스가 동시에 나면 인스턴스가 두 개 생겨 스로틀이 갈라지므로 생성 구간을 잠금.
_construction_lock = asyncio.Lock()


async def get_broker(
    name: str = "kis",
    *,
    user_id: str | None = None,
    account_mode: str | None = None,
) -> KISBroker | None:
    """This user's broker, or None when they have not configured it.

    Async as of Phase 2 — credentials come from Postgres now, not from the
    process environment, so resolving one is I/O. Every call site was already
    inside a coroutine.

    `user_id` defaults to the current request's identity
    (app.user_context.get_current_user_id), which is the whole point of the
    Phase 1 seam: a route handler calls `await get_broker()` with no arguments
    and gets the logged-in person's broker, not the process's.
    인자 없이 호출하면 현재 요청 사용자의 브로커가 반환됨.

    Pass `user_id` explicitly only where there is no ambiguity about whose
    work it is and no bound identity to inherit — and prefer binding the
    identity with `user_scope` instead, so everything downstream agrees.
    """
    from app.config import current_account_mode, settings
    from app.user_context import get_current_user_id

    if name not in _BROKER_FACTORIES:
        raise ValueError(f"unknown broker: {name!r}, known: {list(_BROKER_FACTORIES)}")

    resolved_user = user_id or get_current_user_id()
    resolved_mode = account_mode or current_account_mode()
    key = (name, resolved_user, resolved_mode)

    # Fast path outside the lock: once populated, this is a plain dict read
    # and must not serialize every caller behind a single lock.
    if key in _broker_instances:
        return _broker_instances[key]

    async with _construction_lock:
        # Re-check: another coroutine may have populated it while we waited.
        if key not in _broker_instances:
            _broker_instances[key] = await _BROKER_FACTORIES[name](resolved_user, resolved_mode)
    return _broker_instances[key]


async def invalidate_broker_cache(
    user_id: str,
    account_mode: str | None = None,
    *,
    name: str = "kis",
) -> None:
    """Drop this user's cached broker(s) and close them.

    Must be called whenever their credentials change or are deleted —
    otherwise the cached instance keeps authenticating with the old app key
    until the process restarts, which for a user who just rotated a leaked
    key is the opposite of what they asked for.
    자격증명을 바꾸면 반드시 호출 — 안 하면 프로세스 재시작 전까지 예전 키를 계속 사용.

    Closing matters as much as evicting: each instance holds an httpx client
    with a live connection pool, so dropping the reference without aclose()
    leaks sockets every time somebody updates their keys.
    """
    doomed = [
        key for key in _broker_instances
        if key[0] == name and key[1] == user_id
        and (account_mode is None or key[2] == account_mode)
    ]
    for key in doomed:
        broker = _broker_instances.pop(key)
        if broker is None:
            continue
        try:
            await broker.aclose()
        except Exception:  # noqa: BLE001 - eviction must succeed even if teardown doesn't
            logger.warning("could not close evicted broker %s, dropping it anyway", key, exc_info=True)
    if doomed:
        logger.info("invalidated %d cached broker(s) for user_id=%s", len(doomed), user_id)

from app.domain.models import (
    BrokerPosition,
    BrokerBalance,
    BuyableInfo,
    SellableInfo,
    OrderResult,
    BrokerExecution,
    VIStatus,
    InvestorFlow,
    InvestorEstimate,
    IncomeStatement,
    BalanceSheet,
    FinancialRatio,
    EarningsEstimate,
    InvestmentOpinion,
    ShortSaleDay,
    ProgramTrade,
    NewsTitle,
    DividendInfo,
    OverseasCash,
    IndexQuote,
    RankedStock,
    SectorIndex,
    MinuteCandle,
    Candle,
    StockStatus,
    StockValuation,
)

__all__ = [
    "KISBroker",
    "KISBrokerError",
    "krx_code_from_ticker",
    "broker_for_user",
    "broker_from_credentials",
    "get_broker",
    "invalidate_broker_cache",

    "BrokerPosition",
    "BrokerBalance",
    "BuyableInfo",
    "SellableInfo",
    "OrderResult",
    "BrokerExecution",
    "VIStatus",
    "InvestorFlow",
    "InvestorEstimate",
    "IncomeStatement",
    "BalanceSheet",
    "FinancialRatio",
    "EarningsEstimate",
    "InvestmentOpinion",
    "ShortSaleDay",
    "ProgramTrade",
    "NewsTitle",
    "DividendInfo",
    "OverseasCash",
    "IndexQuote",
    "RankedStock",
    "SectorIndex",
    "MinuteCandle",
    "Candle",
    "StockStatus",
    "StockValuation",
]