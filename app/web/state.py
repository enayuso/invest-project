# app/web/state.py
# Shared mutable state + lifespan for the dashboard's FastAPI app, split out
# so every router module reads/writes it the same way: `from app.web import
# state` then `state.xxx` (module-attribute access, never a
# `from app.web.state import xxx` copy — the latter would freeze a
# stale snapshot instead of tracking what lifespan/handlers actually set).
# 대시보드 라우터 공용 상태 — 항상 `state.xxx`로 접근 (값 복사 금지)
#
# Note the broker is no longer part of "shared state" at all: it is per-user
# now, resolved on demand by `kis_broker()` below. Anything that is genuinely
# per-user must not live in this module as a global.

from __future__ import annotations

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import timedelta
from typing import NamedTuple

from fastapi import FastAPI

from app.brokers import KISBroker, get_broker, invalidate_broker_cache, krx_code_from_ticker
from app.brokers.kis.const import token_cache_dir_is_explicit, token_cache_file
from app.brokers.kis.domestic.websocket import PAPER_WS_URL, REAL_WS_URL, RealtimeFeed
from app.clock import get_clock
from app.config import account_mode_scope, current_account_mode, settings
from app.dataflows.kis_provider import get_ticker_display_info
from app.dataflows.yfinance import TickerDisplayInfo, get_usd_fx_rate
from app.execution import portfolio_risk, risk_rules, runtime_settings
from app.execution.ledger import check_stops_for_ticker, get_position
from app.execution.pending_orders import list_pending_orders
from app.execution.portfolio_risk import ticker_market
from app.execution.reconciliation import (
    DEFAULT_GRACE_PERIOD,
    DEFAULT_LOOKBACK,
    fetch_recent_local_trades,
    match_trades_against_kis,
)
from app.live import kis_market_status, ticker_names
from app.live.orderbook_log import run_orderbook_logger
from app.live.alerts import (
    Alert,
    filter_new_alerts,
    forget_ticker as forget_ticker_alerts,
    prune_old_alerts,
    record_alert,
)
from app.live.auto_trade_loop import auto_trade_loop, forget_ticker as forget_ticker_auto_trade_loop
from app.live.exit_monitor import exit_monitor_loop
from app.live.fill_monitor import fill_monitor_loop
from app.live.poller import forget_ticker as forget_ticker_poller, run_poller_loop
from app.live.store import forget_ticker as forget_ticker_store, latest, load_snapshot, seed_from_intraday_series
from app.credentials.validation import CredentialCheckUnavailable, CredentialsRejected, verify_kis_credentials
from app.user_context import (
    serviced_scopes,
    get_current_user_id,
    record_session_start,
    register_user,
    user_scope,
)
from app.web import trade_allowlist_store

logger = logging.getLogger("app.web")

async def kis_broker() -> KISBroker | None:
    """The KIS broker **for whoever this request belongs to**, or None when
    they haven't configured credentials — every consumer degrades quietly in
    that case (account tile hidden, rows fall back to ledger-only), matching
    the memory store's convention.

    This replaced a module-global `_kis_broker` that lifespan populated once
    at boot from the process's env credentials. That global was the single
    largest cross-user leak in the web layer: roughly forty call sites across
    six routers read it, so with two people logged in, *every* balance,
    position, order, valuation and ranking request would have been answered
    from whichever account the process started with — and POST /api/order
    would have placed the second user's orders into the first user's account.
    사용자별 조회가 아니라 프로세스 부팅 시점 계정을 모두가 공유하던 구조를 제거 —
    주문까지 남의 계좌로 나갈 수 있었음.

    Resolution goes through app.brokers.get_broker, which reads the current
    identity (app.user_context) and caches per (user, account_mode), so this
    is a dict lookup after the first call per user and callers can invoke it
    freely rather than threading a broker through their signatures.
    """
    return await get_broker("kis")

# Company name / previous-close per ticker — fetched once at startup (not
# per poll, this data barely changes intraday) so the dashboard can show
# "Apple Inc. +0.8%" instead of a bare symbol with no reference point.
_display_info: dict[str, TickerDisplayInfo] = {}
# Per-ticker lock so concurrent get_display_name() callers racing an empty
# _display_info entry (multiple dashboard panels rendering the same new
# candidate at once) await one shared yfinance fetch instead of each firing
# their own — same coalescing idea as `_cache_locks` above, just keyed
# forever instead of by TTL since a company name never changes.
_display_info_locks: dict[str, asyncio.Lock] = {}

# order_no -> what cancel_order/revise_order need, for still-open domestic
# orders placed in THIS process. The paper server has no 정정취소가능주문조회
# TR, so the cancel org number only ever comes from the submit response — an
# order placed in another session or before a restart therefore can't be
# cancelled from the UI (it shows cancelable=False). Domestic-only.
# Populated by /api/order, consumed by /api/order/cancel and /api/order/revise.
#
# Keyed by (user, account_mode), same per-scope pattern as `_status_by_scope`
# in app.live.auto_trade_loop: KIS order numbers are per-account daily
# sequences, so a paper-mode order_no and a real-mode order_no can collide
# ordinarily (not a security workaround) — a flat dict here would let a
# paper-mode order's org_no/share-count get replayed against the real broker
# after a runtime account-mode switch.
_cancelable_orders_by_scope: dict[tuple[str, str], dict[str, dict]] = {}


def _cancelable_orders_scope() -> tuple[str, str]:
    return (get_current_user_id(), current_account_mode())


def cancelable_orders() -> dict[str, dict]:
    """This request's own (user, account_mode) order-cancel registry.

    Returns a plain dict, so callers use it exactly like the old bare
    `_cancelable_orders` module dict (`[...] = `, `.get(...)`, `.pop(...)`,
    `in ...`) — only the lookup of *which* dict now depends on the caller's
    current account mode."""
    return _cancelable_orders_by_scope.setdefault(_cancelable_orders_scope(), {})

# Module-level so route handlers (market-status/orderbook/fills) can read the
# same running feed — kis_market_status.set_active_feed handles the CB-gate
# side separately; this is purely for dashboard display.
_market_feed: RealtimeFeed | None = None
_market_feed_task: asyncio.Task | None = None
# Serializes startup. Without it two concurrent first-requests both pass the
# `is None` check and open two WebSocket sessions on one app key — KIS caps
# real-time registrations per session, so the second silently halves the
# effective subscription budget while looking like it worked.
# 동시 최초 요청이 소켓을 두 개 열지 않도록 직렬화 — 앱키당 세션 등록 수 제한이
# 있어 중복 연결은 조용히 구독 한도를 갉아먹음.
_market_feed_lock = asyncio.Lock()

# Registration budget split (2026-08-28, domestic AI audit "안 B") — held
# positions + the highest-priority trade allowlist codes get the full 2TR
# (market-status + trade-tick) tier up to _CORE_FEED_CODES, and the next
# _CB_ONLY_ZONE_SIZE codes get RealtimeFeed's 1TR cb_only tier (market-status
# only — see cb_only_codes's own comment for why that's enough for the
# circuit-breaker gate). 12*2 + 16*1 = 40 = RealtimeFeed.MAX_REGISTRATIONS,
# raising CB coverage from the prior all-2TR MAX_CODES=20 codes to 28.
# Screening candidates dynamically pre-subscribed mid-session
# (app.live.auto_trade_loop._ensure_subscribed) claim cb_only slots out of
# whatever this initial split left free, on top of this boot-time zone.
_CORE_FEED_CODES = 12
_CB_ONLY_ZONE_SIZE = 16

# 2026-09-04 (3rd recurrence of the "candidate CB coverage lost mid-cycle"
# bug — #9, #173, and this one): _ensure_subscribed's ad-hoc
# `add_code(code, cb_only=True)` calls only ever mutated the live
# RealtimeFeed object in memory. reconcile_feed_tier_promotions's
# stop_market_feed()+ensure_market_feed() rebuild throws that object away and
# rebuilds cb_only_zone from _prioritized_krx_codes alone, which had no idea
# candidates existed — so a rebuild mid-cycle silently dropped every
# candidate's circuit-breaker coverage back to "unknown" again. This registry
# is the missing persistent record: note_candidate_cb_code registers a
# candidate the moment auto_trade_loop sees it (regardless of whether a feed
# object exists yet), and _prioritized_krx_codes now merges it back in on
# every (re)build, including the ones triggered by reconcile_feed_tier_promotions.
# 후보 종목의 서킷브레이커 구독이 피드 재구축(포지션 승격 시) 때마다 사라지던
# 3번째 재발 — 후보 등록을 메모리 객체가 아닌 이 모듈 레지스트리에 영구 기록해
# _prioritized_krx_codes가 매 재구축마다 다시 반영하도록 함.
_candidate_cb_codes: dict[str, float] = {}  # code -> monotonic() registration timestamp
_CANDIDATE_CB_TTL_SECONDS = 3600.0
_CANDIDATE_CB_RESERVE = 8


def note_candidate_cb_code(code: str) -> None:
    """Idempotently (re-)register `code` as a live screening candidate that
    needs circuit-breaker coverage, timestamped now. Called from
    app.live.auto_trade_loop._ensure_subscribed for every candidate in a
    cycle's combined list — before that function's `_market_feed is None`
    guard, so a candidate is remembered even on a cycle where no feed object
    exists yet, ready for the next `ensure_market_feed()` to pick up."""
    _candidate_cb_codes[code] = time.monotonic()


def _live_candidate_cb_codes() -> list[str]:
    """Returns still-live candidate codes, most-recently-registered first,
    evicting any entry older than `_CANDIDATE_CB_TTL_SECONDS` as a side
    effect (no separate cleanup timer needed — eviction just piggybacks on
    every call, which _prioritized_krx_codes makes at least once per
    reconcile_feed_tier_promotions pass)."""
    now = time.monotonic()
    expired = [code for code, ts in _candidate_cb_codes.items() if now - ts >= _CANDIDATE_CB_TTL_SECONDS]
    for code in expired:
        del _candidate_cb_codes[code]
    return [code for code, _ts in sorted(_candidate_cb_codes.items(), key=lambda item: item[1], reverse=True)]

# Trailing, currently-unheld codes of _prioritized_krx_codes's core (2TR)
# list that get orderbook coverage (H0STASP0) instead of trade-tick coverage
# (H0STCNT0) — 2026-08-24, forward-collecting 호가창 depth for future
# scalping-strategy backtesting research (KIS has no historical orderbook
# endpoint, so only data collected from here on has any value). User-approved
# tradeoff: these codes lose live trade-tick coverage and fall back to the
# existing 60s REST poller, same degrade already accepted for codes that
# don't fit under the core budget at all (see _prioritized_krx_codes's own
# docstring). Held positions and the rest of the core list are completely
# unaffected — see _held_krx_codes/_prioritized_krx_codes below. Computed
# over the core list only (never the cb_only tail) — the two zones are
# mutually exclusive 2TR-vs-1TR tiers for the same registration budget.
_ORDERBOOK_ZONE_SIZE = 5


async def ensure_market_feed() -> RealtimeFeed | None:
    """Start the real-time feed if it isn't running, and return it.

    Lazy as of Phase 3. It used to be built in `lifespan` from the process's
    env credentials, which cannot work once credentials are per-user: at boot
    nobody is logged in, so there is nothing to authenticate a WebSocket with.
    Calling it at boot is now just an optimistic attempt — it succeeds only if
    the serviced user already has credentials stored from a previous session,
    and otherwise costs nothing and starts the feed on first authenticated use
    instead.
    부팅 시점에는 로그인한 사람이 없어 인증할 자격증명이 없으므로 지연 생성 —
    부팅 때 호출하는 건 '이미 저장돼 있으면 바로 띄운다'는 낙관적 시도일 뿐.

    This construction site does NOT go through the broker factory — it needs
    the raw key/secret, not a broker (plan §4, "easy to miss") — so it does its
    own credential lookup.

    ONE feed, authenticated with the first serviced scope's key, subscribing
    the union of every scope's KRX codes (see `_prioritized_krx_codes`).
    Everything this feed carries for the order gates — trading halt, VI,
    circuit breaker, ticks, orderbook — is PUBLIC per-ticker data, so one
    connection legitimately serves everybody, and the 41-registration cap is
    spent once on the union instead of N times on overlapping lists.
    게이트가 쓰는 데이터는 전부 공개 시세라 연결 하나로 전원을 커버할 수 있음.

    **What is NOT covered: 체결통보 (fill notifications).** That subscription is
    keyed by HTS ID and only reports fills for the account whose key
    authenticated the socket, so on this single feed only the first serviced
    user gets live fill events; everyone else's fills surface through the
    existing REST reconciliation path instead. Per-user fill notifications need
    per-user sockets and are the remaining feed work — a display/notification
    gap, not a gate, which is why it does not block the loops.
    체결통보만은 앱키 소유자 계정 것만 오므로 나머지 사용자는 REST 경로로 확인됨 —
    게이트가 아니라 알림 기능의 공백.

    Starting with an empty `codes` list is fine and deliberate: RealtimeFeed's
    subscribe loop is a no-op over an empty list, so the socket is open and
    ready for `add_code()` the moment a KRX ticker is added. Before that was
    true, an empty-at-boot allowlist left the feed permanently None and a
    later-added KRX code had no live feed to subscribe on — its
    circuit-breaker status stayed "unknown" (fail-closed, blocking every
    trade) until the whole process restarted.
    """
    global _market_feed, _market_feed_task
    if _market_feed is not None:
        return _market_feed

    async with _market_feed_lock:
        if _market_feed is not None:  # filled while we waited
            return _market_feed

        scopes = await serviced_scopes()
        if not scopes:
            return None
        if len(scopes) > 1:
            logger.info(
                "realtime feed: %d serviced scopes share one connection (public market data); "
                "live 체결통보 covers %s/%s only, the rest reconcile over REST",
                len(scopes), scopes[0].user_id, scopes[0].account_mode,
            )
        serviced = [s.user_id for s in scopes]
        # Sync now (env fields, no I/O) — the try/except this used to need
        # around a DB read is gone; settings.kis_credentials() cannot itself
        # fail. See Settings.kis_credentials's own docstring.
        feed_credentials = settings.kis_credentials(current_account_mode())
        if feed_credentials is None:
            logger.info("KIS real-time feed not started (no credentials configured yet)")
            return None

        # is_real comes from the credentials themselves, not from
        # current_account_mode() — the feed must authenticate against the
        # same account the credentials belong to, and a stale global could
        # otherwise point a real-account key at the paper WebSocket.
        is_real = feed_credentials.is_real
        codes, orderbook_zone, cb_only_zone = await _prioritized_krx_codes()
        _market_feed = RealtimeFeed(
            app_key=feed_credentials.app_key,
            app_secret=feed_credentials.app_secret,
            codes=codes,
            base_url=settings.kis_real_base_url if is_real else settings.kis_paper_base_url,
            ws_url=REAL_WS_URL if is_real else PAPER_WS_URL,
            hts_id=feed_credentials.hts_id,  # None -> no 체결통보 subscription
            is_real=is_real,
            orderbook_codes=orderbook_zone,
            cb_only_codes=cb_only_zone,
        )
        _market_feed_task = asyncio.create_task(_market_feed.run())
        kis_market_status.set_active_feed(_market_feed)
        logger.info("KIS real-time feed started for user_id=%s (%s)", serviced[0], current_account_mode())
        return _market_feed


async def stop_market_feed() -> None:
    """Tear the feed down and allow a later `ensure_market_feed()` to build a
    fresh one — which is what a credential change needs: the running socket is
    authenticated with the old app key and cannot be re-pointed in place."""
    global _market_feed, _market_feed_task
    async with _market_feed_lock:
        if _market_feed is not None:
            try:
                await _market_feed.stop()
            except Exception:  # noqa: BLE001 - teardown must not block the caller
                logger.warning("could not cleanly stop the real-time feed", exc_info=True)
            kis_market_status.set_active_feed(None)
            _market_feed = None
        if _market_feed_task is not None:
            _market_feed_task.cancel()
            _market_feed_task = None

# Small async TTL cache for read-only KIS-backed endpoints. Every KIS call
# funnels through the broker's 1-req/sec throttle, so the dashboard's polls
# and on-demand refetches (chart period toggles, tab switches, refresh) would
# otherwise queue up behind each other. Caching serves repeats within the TTL
# and coalesces concurrent callers behind one lock (no stampede).
_cache: dict[str, tuple[float, object]] = {}
_cache_locks: dict[str, asyncio.Lock] = {}


def user_key(key: str) -> str:
    """Namespace a cache key with the current user.

    Use this for anything derived from a KIS *account* — balances, positions,
    fills, buyable amounts. Do NOT use it for public market data (candles,
    rankings, index quotes, sector indices): those are identical for everyone
    and sharing them across users is the entire point of the cache.
    계좌 유래 데이터에만 사용 — 시세/랭킹 등 공개 데이터는 공유가 맞음.

    Two keys in this module were bare literals holding per-account data
    ("account" and "executions", plan §2.3). With more than one user those
    served the first caller's balances and fills to everyone else for the
    length of the TTL — a cross-user data leak that no amount of correct
    per-user credential resolution upstream would have caught, because the
    request never reached the broker at all on a cache hit.
    캐시 적중 시 브로커까지 가지 않으므로, 상위에서 사용자별 자격증명을 아무리
    정확히 해도 이 유출은 막히지 않았음.

    Now also namespaced by account mode — paper/real switches at runtime via
    PUT /api/account-mode with no restart, so a user_id-only key served the
    other mode's cached account snapshot (wrong is_real flag, wrong
    positions) for up to the cache's TTL after a switch.
    """
    return f"{get_current_user_id()}:{current_account_mode()}:{key}"


async def cached(key: str, ttl: float, factory):
    now = time.monotonic()
    hit = _cache.get(key)
    if hit is not None and now - hit[0] < ttl:
        return hit[1]
    lock = _cache_locks.setdefault(key, asyncio.Lock())
    async with lock:
        hit = _cache.get(key)  # another caller may have filled it while we waited
        if hit is not None and time.monotonic() - hit[0] < ttl:
            return hit[1]
        value = await factory()
        _cache[key] = (time.monotonic(), value)
        return value


def invalidate_cache(*prefixes: str) -> None:
    """Drop every cache entry whose key starts with one of `prefixes`.

    Matching is on the FULL key, and per-account keys are now namespaced with
    the user (see `user_key`), so `invalidate_cache("account")` no longer
    matches this user's "u-abc123:account" entry. Callers busting per-account
    data must pass `user_key("account")`, not the bare literal — otherwise the
    call silently becomes a no-op and the dashboard serves stale balances for
    a full TTL after an order.
    사용자 네임스페이스가 붙은 키는 반드시 user_key()로 지워야 함 — 그냥 "account"를
    넘기면 아무것도 지워지지 않고 조용히 오래된 잔고가 표시됨."""
    for k in [k for k in _cache if any(k.startswith(p) for p in prefixes)]:
        _cache.pop(k, None)


async def recent_executions():
    """(domestic, overseas) KIS executions over the last week — cached 15s and
    shared by /api/trades and /api/orders/open, which both need this exact
    data and would otherwise each fetch it separately (4 throttled calls,
    doubled)."""
    broker = await kis_broker()
    if broker is None:
        return [], []

    async def _load():
        today = get_clock().today()
        start = (today - timedelta(days=7)).strftime("%Y%m%d")
        end = today.strftime("%Y%m%d")
        dom, ovs = [], []
        try:
            dom = await broker.fetch_daily_executions(start, end)
        except Exception:
            logger.warning("KIS domestic executions failed", exc_info=True)
        if settings.overseas_trading_enabled or portfolio_risk.overseas_flag_contradicted():
            try:
                ovs = await broker.fetch_overseas_executions(start, end)
            except Exception:
                logger.warning("KIS overseas executions failed", exc_info=True)
        return dom, ovs

    # 35s, not 15s: /api/orders/open polls this every 30s and /api/trades
    # every 60s (see app/web/static/js/trading.js's setInterval calls) — a
    # TTL shorter than the faster poller's own interval meant the cache
    # never actually got hit by either recurring poller, only by rarer
    # near-simultaneous calls. 35s (a bit above the 30s poller) means most
    # of that poller's calls now reuse this cache instead of re-querying
    # KIS, at the cost of executions data being up to ~60-70s old — fine
    # for this display list (order gates re-check live state independently,
    # they don't read this cache).
    return await cached(user_key("executions"), 35, _load)


async def load_display_info() -> None:
    for ticker in settings.trade_allowlist:
        try:
            _display_info[ticker] = await get_ticker_display_info(ticker)
        except Exception:
            logger.warning("could not load display info for %s", ticker, exc_info=True)


async def _prefetch_startup_data() -> None:
    """Backgrounded from lifespan (not awaited before yield) — display name/
    previous-close and intraday tick history are cosmetic and every consumer
    already degrades to a bare ticker/empty history until populated (see
    get_display_name, /api/trade-allowlist-info's `info else ticker` fallback), so
    there's no correctness reason to block the dashboard from becoming
    reachable on every trade allowlist ticker's KIS/yfinance round-trip finishing.
    포트 바인딩을 막지 않도록 백그라운드로 — 이름/틱히스토리는 표시용이라
    채워지기 전에도 이미 폴백 처리됨."""
    await load_display_info()
    await asyncio.gather(*(seed_ticker_history(ticker) for ticker in settings.trade_allowlist))


# (user_id, account_mode, trade_id) already alerted on — in-process dedup so
# trade_reconciliation_loop's periodic re-runs don't re-alert the SAME
# unresolved discrepancy every pass. Needed because app.live.alerts.
# filter_new_alerts' own cooldown (settings.alert_cooldown_seconds, default
# 300s) is shorter than this loop's interval (600s) and keyed by (account_mode,
# ticker, kind) — not trade_id — so it cannot tell two different unresolved
# trades on the same ticker apart, and would let this one re-fire every
# single pass for as long as the discrepancy stays unresolved (up to
# DEFAULT_LOOKBACK=48h of passes). Cleared only by a process restart —
# intentional: after a restart, one fresh alert for a still-unresolved
# discrepancy is correct, not noise.
_reported_discrepancies: set[tuple[str, str, int]] = set()


async def _reconciliation_pass() -> None:
    """One pass of trade/KIS reconciliation, run by
    `trade_reconciliation_loop` below (previously boot-time-only, see that
    function's own docstring for the periodic conversion) — detects (never
    corrects) a local `trades` row whose FULL_AUTO mirror to KIS may never
    have completed, e.g. a process kill between ledger.record_trade() and
    mirror_trade_to_kis(). See project memory:
    force-kill-mid-trade-ledger-drift-risk. Never auto-corrects — right
    after a crash, "KIS never got the order" and "KIS got it but the
    response just hasn't arrived yet" are indistinguishable from here, so
    auto-reversing risks undoing a trade that actually went through; a
    human should reconcile against the real KIS balance before touching the
    ledger.
    강제종료로 인한 원장/KIS 불일치를 주기적으로 탐지 — 자동수정 없음.

    2026-08-13: log-only used to mean a discrepancy was invisible unless
    someone was tailing this exact log at the exact moment it printed — a
    self-audit that day found 3 live "submitted: ..." trades (order
    ACCEPTED, never actually confirmed filled — see app.brokers.kis.
    domestic.order.submit_order's own "order accepted" log line, not
    "filled") that this exact check would have flagged CRITICAL, weeks
    earlier, with nobody around to see it: one left a 10-share phantom
    position open (008290.KS, ledger said held, KIS said 0), two others
    left the ledger's own cash pool overstated by real, uncorrected amounts
    (~1.23M and ~887K KRW — see the discovery session's own numbers). Now
    also persisted as a proper alert (app.live.alerts, same table/cooldown/
    dashboard feed as price_move/pipeline_failure/etc.) so it survives past
    the log line and a human actually sees it on the dashboard — still
    never touches the ledger itself.

    2026-09-07: fetches local trades FIRST and returns immediately if there
    are none — same "don't fire the KIS execution-history TR (EGW00215/
    EGW00201 pressure) on a pass with nothing to check" optimization
    app.live.fill_monitor.fill_monitor_loop already uses (its own FIX-4),
    now that this runs periodically rather than once. Also dedupes against
    `_reported_discrepancies` so a still-unresolved discrepancy alerts once,
    not once per pass — see that set's own comment."""
    if await kis_broker() is None:
        return
    local_trades = await fetch_recent_local_trades(DEFAULT_LOOKBACK, DEFAULT_GRACE_PERIOD)
    if not local_trades:
        return
    dom, ovs = await recent_executions()
    user_id, mode = get_current_user_id(), current_account_mode()
    fresh_alerts = []
    for discrepancy in match_trades_against_kis(local_trades, dom, ovs):
        key = (user_id, mode, discrepancy.trade_id)
        if key in _reported_discrepancies:
            logger.debug(
                "trade reconciliation: trade_id=%d still unresolved (already alerted this run)",
                discrepancy.trade_id,
            )
            continue
        _reported_discrepancies.add(key)
        log = logger.error if discrepancy.severity == "CRITICAL" else logger.warning
        log(
            "trade reconciliation %s: trade_id=%d %s %s x%d on %s (recorded %s) — %s",
            discrepancy.severity, discrepancy.trade_id, discrepancy.side, discrepancy.ticker,
            discrepancy.shares, discrepancy.trade_date, discrepancy.created_at, discrepancy.note,
        )
        fresh_alerts.append(
            Alert(
                ticker=discrepancy.ticker,
                kind="unmirrored_trade",
                message=(
                    f"[{discrepancy.severity}] {discrepancy.side} x{discrepancy.shares} on "
                    f"{discrepancy.trade_date} — {discrepancy.note}"
                ),
                triggered_at=get_clock().now(),
            )
        )
    for alert in filter_new_alerts(fresh_alerts):
        await record_alert(alert)


async def trade_reconciliation_loop() -> None:
    """Periodic wrapper around `_reconciliation_pass` — converts what was a
    boot-time-only check into a standing loop, so a ledger/KIS drift that
    appears mid-session doesn't sit undetected until the next restart (the
    gap a user explicitly flagged, 2026-09-07). Runs the first pass
    immediately (interval sleep is at the END of the loop, matching
    app.live.fill_monitor.fill_monitor_loop), then every
    `settings.trade_reconciliation_interval_seconds`.

    Scoped per (user, account_mode) like every other standing background
    loop here (stop_loss_loop, fill_monitor_loop, auto_trade_loop) — a
    single-shot boot check could get away with the ambient default scope,
    but a loop that runs for the life of the process must not silently keep
    reading/writing one user's ledger under another's identity (see
    stop_loss_loop's own docstring for the incident shape this guards
    against). `serviced_scopes()` itself is awaited outside the per-scope
    try — same reasoning as stop_loss_loop's identical guard: a transient
    DB-pool hiccup there must not propagate out of `while True` and
    silently stop reconciliation for the rest of the process (this task is
    only ever awaited at shutdown with return_exceptions=True, so an
    uncaught exception here would never be logged)."""
    interval_seconds = settings.trade_reconciliation_interval_seconds
    while True:
        try:
            scopes = await serviced_scopes()
        except Exception:
            logger.exception(
                "trade reconciliation pass could not resolve serviced scopes — will retry next interval"
            )
            scopes = []
        for scope in scopes:
            try:
                with user_scope(scope.user_id), account_mode_scope(scope.account_mode):
                    await _reconciliation_pass()
            except Exception:
                logger.exception(
                    "trade reconciliation pass failed for %s/%s — continuing with the other scopes",
                    scope.user_id, scope.account_mode,
                )
        await asyncio.sleep(interval_seconds)


async def get_display_name(ticker: str) -> str:
    """The single source of a ticker's display name across every panel
    (holdings, trade allowlist, decision history, trade history) — mostly the
    yfinance-sourced company name (get_ticker_display_info), never a KIS
    position's own (Korean-only) name field. Two reasons this one source
    was picked over KIS's: it covers ANY ticker, not just ones currently
    held or on the trade allowlist (a decision made for a ticker later removed
    from the trade allowlist previously showed no name at all, just the bare
    code), and it's the same string regardless of which panel/data path
    renders it.

    One exception (2026-07-21): a KRX candidate discovered via
    app.live.universe.krx_top_n (volume scan, not the fixed trade allowlist) has
    no Korean-friendly name from yfinance — it returns the English legal
    name (e.g. "hyungji Elite Co., Ltd." for 093240.KS), unreadable to a
    Korean user who'd recognize "삼성전자" instantly. krx_top_n already sees
    KIS's own hts_kor_isnm (used there for the leverage-filter check) and
    remembers it via app.live.ticker_names.

    ticker_names is checked FIRST, unconditionally (not just when
    `_display_info` doesn't have this ticker yet) — real incident
    (2026-07-23): 064260.KS's `_display_info` got cached early with a
    yfinance name for a completely different company (064260.KS is 다날;
    yfinance returned "에코프로비엠", itself a real company but the wrong
    one for this code — see app.dataflows.yfinance's own name-plausibility
    check, which can't catch a wrong-but-real-looking name). Once cached,
    that wrong name stuck forever under the old "only check ticker_names if
    _display_info is still empty" ordering, even after krx_top_n later
    learned the correct KIS-sourced name for the same ticker. KIS's
    hts_kor_isnm is authoritative for an exact KRX code — it can't suffer
    yfinance's cross-symbol confusion — so it should always win once known,
    regardless of cache order.

    `_display_info` is pre-populated for the trade allowlist at startup
    (`load_display_info`); for any other ticker (a past decision's now
    off-trade allowlist symbol) this fetches once and caches the result, same
    "lazy fetch, cache forever, name is time-invariant" convention as
    `get_ticker_currency` in dataflows/yfinance."""
    if ticker.endswith(".KS") and (kor_name := ticker_names.get(ticker)):
        return kor_name
    if ticker not in _display_info:
        lock = _display_info_locks.setdefault(ticker, asyncio.Lock())
        async with lock:
            # Re-check after acquiring: a racing caller may have already
            # populated this while we waited on the lock.
            if ticker not in _display_info:
                try:
                    _display_info[ticker] = await get_ticker_display_info(ticker)
                except Exception:
                    logger.warning("could not load display name for %s", ticker, exc_info=True)
                    return ticker
    return _display_info[ticker].name


async def stop_loss_loop() -> None:
    """Checks every trade allowlist ticker's stop-loss on a slower cadence than
    the minute poller — a real gap otherwise, since without this nothing
    ever enforces a stop-loss for the live dashboard between decisions
    (only the backtest harness and the manual CLI did before this).

    Runs one pass per serviced scope, inside BOTH `user_scope` and
    `account_mode_scope` — this loop calls check_stops_for_ticker, which reads
    positions and writes trades, and both sides filter on the current user, in
    the current mode. Outside a request there is nothing to inherit either from,
    so without the scopes every pass would read and write as settings.user_id in
    the deployment's default mode, regardless of whose stop-loss it was.
    사용자·모드 스코프를 함께 열고 순회 — 둘 중 하나라도 빠지면 남의 계좌 기준으로
    읽고 기록하게 됨.

    `ensure_loaded` is not optional here. The allowlist is resolved per
    (user, mode) and a scope is empty until it has been loaded, so skipping it
    would iterate zero tickers and check nobody's stop-loss — silently, since an
    empty allowlist is indistinguishable from "this user permits nothing".
    스코프는 load() 전까지 비어 있어, 생략하면 아무 종목도 확인하지 않고 조용히 넘어감.

    One scope's failure must not skip the rest: a broker outage for user A is
    not a reason to stop checking user B's stop-losses. The per-ticker try
    already covers ticker failures; the per-scope try covers a scope that cannot
    even be loaded.
    한 사용자의 실패가 다른 사용자의 손절 감시를 막지 않도록 스코프별로도 감쌈."""
    interval_seconds = settings.stop_loss_check_interval_hours * 3600
    while True:
        try:
            scopes = await serviced_scopes()
        except Exception:
            # `serviced_scopes()` is the ONE await in this loop body that no
            # inner try covers, and it genuinely raises on an ordinary DB-pool
            # hiccup (get_pool's documented backoff-window RuntimeError). Before
            # this, that propagated straight out of `while True` and stopped
            # stop-loss enforcement for the rest of the process — silently: this
            # task is only ever awaited at shutdown with return_exceptions=True,
            # so the exception was never logged or even retrieved. Identical
            # guard to app.live.poller.run_poller_loop's (live-audit-2026-08-08).
            # 이 한 줄만 try 밖이라, DB 일시장애 한 번에 손절 감시가 조용히
            # 프로세스가 끝날 때까지 영구 정지됐음.
            logger.exception("stop-loss pass could not resolve serviced scopes — will retry next interval")
            scopes = []
        for scope in scopes:
            try:
                with user_scope(scope.user_id), account_mode_scope(scope.account_mode):
                    await trade_allowlist_store.ensure_loaded(scope.user_id, scope.account_mode)
                    for ticker in settings.trade_allowlist:
                        try:
                            result = await check_stops_for_ticker(ticker)
                            logger.info(
                                "stop-loss check for %s (%s/%s): %s",
                                ticker, scope.user_id, scope.account_mode, result,
                            )
                        except Exception:
                            logger.exception(
                                "stop-loss check failed for %s (%s/%s) (will retry next interval)",
                                ticker, scope.user_id, scope.account_mode,
                            )
            except Exception:
                logger.exception(
                    "stop-loss pass failed for %s/%s — continuing with the other users",
                    scope.user_id, scope.account_mode,
                )
        await asyncio.sleep(interval_seconds)


async def seed_ticker_history(ticker: str) -> None:
    # yfinance's own full-day series first (most complete/authoritative —
    # see app.live.store's module docstring), Postgres snapshot as a
    # fallback for whatever that doesn't cover. Neither call clobbers
    # already-populated in-memory ticks.
    await seed_from_intraday_series(ticker)
    await load_snapshot(ticker)


class LeveragedTickerError(Exception):
    """Raised by add_ticker_to_trade_allowlist when `ticker` looks like a
    futures/leveraged/inverse product — not supported by this system's
    long-only, no-leverage scope."""


class UnresolvedTickerNameError(Exception):
    """Raised by add_ticker_to_trade_allowlist when no source (KRX/NASDAQ
    master file, yfinance) could resolve a real name for `ticker` —
    `get_ticker_display_info` then falls back to the bare ticker string
    itself. Refusing this rather than proceeding: `is_leveraged_or_inverse`
    correctly treats an unresolvable name as "no evidence either way" (an
    empty name matches no keyword), but that logic was being reached with
    the BARE TICKER standing in for a name, which reads as "definitely not
    leveraged" instead of "we have no idea" — confirmed reachable for any
    NYSE-listed ticker (SOXL, UPRO, TNA, ...), since the NASDAQ-only
    overseas master file has no entry for them."""


class UnsupportedProductTypeError(Exception):
    """Raised by add_ticker_to_trade_allowlist when `ticker` isn't a plain
    stock or ETF (see app.domain.leverage_filter.is_unsupported_product_type)
    — a mutual fund/ETN doesn't fit this system's fundamentals/news-driven
    analysis model, and real incident showed materially worse yfinance
    data coverage for one (218150.KS, quoteType MUTUALFUND)."""


class CashOrBondFundError(Exception):
    """Raised by add_ticker_to_trade_allowlist when `ticker` is a cash-like /
    bond / bond-mix fund (see app.domain.product_filter.is_cash_or_bond_fund)
    — no company earnings/news to analyze, and a near-zero blended ATR that
    misleads the sizing prompt. 2026-09-01: 481050.KS was auto-added this way
    after reaching a BUY-ish debate."""


class DerivativeIncomeFundError(Exception):
    """Raised by add_ticker_to_trade_allowlist when `ticker` is a covered-call
    / option-income fund (see
    app.domain.product_filter.is_derivative_income_fund) — 100% equity with a
    written-option overlay, so it is NOT a CashOrBondFundError (no bond
    sleeve), but still has no company to analyze and an option-driven payoff
    that momentum/valuation/ATR analysis reads backwards. 2026-09-02:
    498400.KS "KODEX 200타겟위클리커버드콜" burned a full ~17-LLM-call pass
    this way (audit id=1170)."""


class MalformedTickerError(Exception):
    """Raised by add_ticker_to_trade_allowlist/add_ticker_to_watchlist when
    `ticker` doesn't match the basic ticker-symbol shape (see
    app.web.ticker_validation.is_valid_ticker_format) — this used to be
    entirely unchecked here (only the dashboard's manual-add HTTP route
    validated), so a malformed value from the AI auto-add path (a
    krx_top_n-sourced candidate, or a trade_allowlist entry pooled back into
    a later cycle's own candidates) could reach settings.trade_allowlist
    with no check at all."""


async def add_ticker_to_trade_allowlist(ticker: str) -> tuple[str, TickerDisplayInfo | None]:
    """Add `ticker` to the live trade_allowlist (domestic or overseas half,
    picked by `ticker_market(ticker)`). Shared core behind
    `POST /api/trade-allowlist/add` and `app.orchestrator.pipeline`'s
    auto-add-on-BUY-decision hook — only the router further translates
    exceptions into HTTPException; a caller like the pipeline that wants
    best-effort behavior just catches Exception and logs. Reassigns the
    relevant `settings.trade_allowlist_domestic`/`_overseas` list (never
    mutates in place) so background pollers iterating it never see a
    half-updated list. Returns `("exists" | "added", display_info)` —
    display_info is `None` for "exists" if this ticker was added before
    display info was ever fetched for it (no extra network round-trip just
    to report a name for a no-op)."""
    from app.domain.leverage_filter import is_leveraged_or_inverse, is_unsupported_product_type
    from app.domain.product_filter import out_of_scope_reason
    from app.web.ticker_validation import is_valid_ticker_format

    if not is_valid_ticker_format(ticker):
        raise MalformedTickerError(f"'{ticker}' doesn't look like a valid ticker symbol")
    if ticker in settings.trade_allowlist:
        return "exists", _display_info.get(ticker)
    info = await get_ticker_display_info(ticker)
    if info.name == ticker:
        raise UnresolvedTickerNameError(
            f"could not resolve a real name for {ticker} from any source (KRX/NASDAQ master file, "
            "yfinance) — refusing to add it sight-unseen, since the leveraged/inverse filter can't "
            "clear a name it never actually got"
        )
    if is_leveraged_or_inverse(info.name):
        raise LeveragedTickerError(
            f"{info.name} looks like a futures/leveraged/inverse product — "
            "not supported by this system's long-only, no-leverage scope"
        )
    if is_unsupported_product_type(info.quote_type):
        raise UnsupportedProductTypeError(
            f"{ticker} is a {info.quote_type} (not a plain stock or ETF) — "
            "not supported by this system's fundamentals/news-driven analysis scope"
        )
    reason = out_of_scope_reason(info.name)
    if reason == "cash_or_bond_fund":
        raise CashOrBondFundError(
            f"{info.name} is a cash-like/bond fund — this system analyzes companies "
            "(earnings, news, valuation), which such a product has none of"
        )
    if reason == "derivative_income_fund":
        raise DerivativeIncomeFundError(
            f"{info.name} is a covered-call / option-income fund — its return is an "
            "option-driven payoff, not a company's (no earnings, news or valuation to analyze)"
        )
    _display_info[ticker] = info
    market = ticker_market(ticker)
    market_key = "trade_allowlist_domestic" if market == "domestic" else "trade_allowlist_overseas"
    if market == "domestic":
        settings.trade_allowlist_domestic = [*settings.trade_allowlist_domestic, ticker]
    else:
        settings.trade_allowlist_overseas = [*settings.trade_allowlist_overseas, ticker]
    # Keep trade_allowlist ⊆ watchlist (see app.config.Settings) an
    # invariant at runtime too, not just at boot — this is the one place
    # trade_allowlist grows after startup, so it's the one place that can
    # otherwise violate it.
    watchlist_grew = ticker not in settings.watchlist
    if watchlist_grew:
        settings.watchlist = [*settings.watchlist, ticker]
    await trade_allowlist_store.save(market_key, settings.trade_allowlist_for_market(market))
    if watchlist_grew:
        await trade_allowlist_store.save("watchlist", settings.watchlist)
    # Tell the running real-time feed to subscribe this code too — without
    # this, a ticker added after the feed's initial startup (dashboard add,
    # or the AI auto-add-on-BUY hook) stays permanently unsubscribed and
    # app.execution.kis_bridge's circuit-breaker gate reports its state as
    # unknown forever (fail-closed, blocks every trade) until the whole
    # process restarts. Best-effort: a failure here just means the next
    # reconnect picks it up instead (see RealtimeFeed.add_code's own
    # docstring) — never worse than the pre-existing permanent-unknown bug.
    # US tickers have no KRX code and aren't this feed's concern (krx_code
    # is None). `_market_feed` is only None when no KIS broker is configured
    # at all (lifespan now always starts it otherwise, even with an
    # empty-at-boot trade allowlist — see lifespan's own comment) — that remaining
    # case is a pre-existing, separate condition this doesn't change.
    if _market_feed is not None:
        krx_code = krx_code_from_ticker(ticker)
        if krx_code is not None:
            try:
                subscribed = await _market_feed.add_code(krx_code)
            except Exception:
                logger.warning("could not live-subscribe %s to the realtime feed", ticker, exc_info=True)
            else:
                if not subscribed:
                    # A trade allowlist addition (unlike a screening candidate
                    # pre-subscribe) is an explicit intent to trade this
                    # ticker — a cap-out here is worse than the candidate
                    # case, and RealtimeFeed.add_code's False return means
                    # circuit-breaker state will be permanently "unknown"
                    # (fail-closed) for it, same as the 2026-08-28 audit's
                    # 55%-of-Buys-blocked incident, just via this entrypoint
                    # instead of auto_trade_loop's. Surfaced via alerts (not
                    # just logger.warning) since the 2026-08-05 warning-only
                    # version of this went unnoticed until that audit.
                    for alert in filter_new_alerts([Alert(
                        ticker=ticker, kind="realtime_subscription_capped",
                        message=(
                            f"{ticker}: KIS 실시간 등록 {_market_feed.MAX_REGISTRATIONS}건 소진 — "
                            "서킷브레이커 상태가 unknown으로 남아 매매가 차단됩니다"
                        ),
                        triggered_at=get_clock().now(),
                    )]):
                        await record_alert(alert)
    try:
        await seed_ticker_history(ticker)
    except Exception:
        logger.warning("could not seed history for newly added %s", ticker, exc_info=True)
    return "added", info


async def add_ticker_to_watchlist(ticker: str) -> tuple[str, TickerDisplayInfo | None]:
    """Add `ticker` to `settings.watchlist` only — the light "관심종목"
    concept, with no implication of AI trading permission. Unlike
    add_ticker_to_trade_allowlist (the heavy, trading-permission path),
    this does NOT run the leverage/inverse filter, does NOT subscribe the
    realtime feed, and does NOT seed price history — those are
    trading-pipeline side effects that only make sense for a ticker the AI
    may actually trade. Returns `("exists" | "added", display_info)`, same
    shape as add_ticker_to_trade_allowlist, for a consistent router response
    either way."""
    from app.web.ticker_validation import is_valid_ticker_format

    if not is_valid_ticker_format(ticker):
        raise MalformedTickerError(f"'{ticker}' doesn't look like a valid ticker symbol")
    if ticker in settings.watchlist:
        return "exists", _display_info.get(ticker)
    info = await get_ticker_display_info(ticker)
    _display_info[ticker] = info
    settings.watchlist = [*settings.watchlist, ticker]
    await trade_allowlist_store.save("watchlist", settings.watchlist)
    return "added", info


async def remove_ticker_from_trade_allowlist(ticker: str) -> None:
    """Drop `ticker` from the trade allowlist (domestic or overseas half,
    picked by `ticker_market(ticker)`) and its cached display info — shared
    core behind `POST /api/trade-allowlist/remove`, persisted so the removal
    survives a restart too (otherwise a restart would silently bring back a
    ticker someone deliberately removed, straight from the stale `.env`
    default)."""
    market = ticker_market(ticker)
    if market == "domestic":
        settings.trade_allowlist_domestic = [t for t in settings.trade_allowlist_domestic if t != ticker]
    else:
        settings.trade_allowlist_overseas = [t for t in settings.trade_allowlist_overseas if t != ticker]
    _display_info.pop(ticker, None)
    # Clear every process-local cache keyed to this ticker (2026-08-13) —
    # `_display_info.pop` above only ever cleared the display-name cache;
    # the tick buffer (app.live.store), the cumulative-volume baseline
    # (app.live.poller), the alert cooldowns (app.live.alerts), and the
    # overnight-gap-warning cooldown (app.live.auto_trade_loop) were all
    # left untouched. Real, not theoretical, impact: re-adding the same
    # ticker later resurrected its OLD tick buffer intact (nothing else
    # ever clears `app.live.store._ticks`), so the chart and every alert
    # threshold that reads off it would use stale ticks from whenever the
    # ticker was last watched as if they were current. Each module exposes
    # its own `forget_ticker` rather than this function reaching into their
    # module-level dicts directly, so ownership of each cache's shape stays
    # with the module that defines it.
    await forget_ticker_store(ticker)
    forget_ticker_poller(ticker)
    forget_ticker_alerts(ticker)
    forget_ticker_auto_trade_loop(ticker)
    market_key = "trade_allowlist_domestic" if market == "domestic" else "trade_allowlist_overseas"
    await trade_allowlist_store.save(market_key, settings.trade_allowlist_for_market(market))
    # Keep the auto-added marker from outliving the membership it's about —
    # a stale marker for a ticker no longer even on the list is meaningless
    # (also enforced defensively at boot by
    # Settings._auto_added_subset_of_trade_allowlist, but no reason to wait
    # for a restart when we're already removing it here).
    await unmark_trade_allowlist_auto_added(ticker)


async def remove_ticker_from_watchlist(ticker: str) -> None:
    """Drop `ticker` from the watchlist — shared core behind
    `POST /api/watchlist/remove`. Caller must already have checked the
    `trade_allowlist ⊆ watchlist` invariant (see that router's own
    docstring); this function doesn't re-check it."""
    settings.watchlist = [t for t in settings.watchlist if t != ticker]
    await trade_allowlist_store.save("watchlist", settings.watchlist)


class HeldCheck(NamedTuple):
    """`ticker`를 trade_allowlist에서 빼도 안전한지 판단하는 데 필요한 사실만."""
    shares: int      # 확인된 보유수량. known=False면 의미 없음(0).
    known: bool      # False = 보유 여부를 확인하지 못함(KIS 장애). "보유 없음"이 아님.
    source: str      # "ledger" | "kis" | "both" | "none" | "unknown"


async def held_shares_for_removal(ticker: str) -> HeldCheck:
    """POST /api/trade-allowlist/remove의 보유-포지션 가드가 쓰는 판단 근거.

    보유 여부는 두 소스의 합집합으로 판단한다 — 원장(exit_monitor/손절 루프가
    실제로 보는 테이블)과 KIS 실계좌. 어느 한쪽이라도 보유를 보고하면 보호
    대상으로 취급한다: 원장에만 있는 포지션은 손절 감시 대상이고, KIS에만
    있는 포지션은 reconcile_trade_allowlist가 채워 넣어야 할 대상이다.

    known=False(KIS 장애로 조회 실패)는 "보유 없음"과 다르게 취급해야 한다 —
    reconcile_trade_allowlist가 이미 같은 판단을 내려놨다: 실보유를 못 읽는
    상태에서 "모름"을 "없음"으로 취급하면 대량 오삭제로 이어진다.
    KIS 미설정(broker is None)은 장애가 아니라 "이 배포엔 KIS가 없다"이므로
    원장 결과만으로 known=True 취급한다.

    원장 쪽 fail-open(get_position이 DB 실패를 조용히 0으로 반환하는 것,
    app/execution/ledger.py)은 이 함수의 범위 밖 — KIS가 정상이면 KIS 쪽이
    보유를 잡아주므로 구멍은 "DB와 KIS가 동시에 문제 있는" 경우뿐이다."""
    ledger_shares = 0
    try:
        ledger_shares = (await get_position(ticker)).shares
    except Exception:
        ledger_shares = 0

    try:
        broker = await kis_broker()
    except Exception:
        broker = None  # 자격증명 미설정/조회 실패 = "이 배포엔 KIS가 없다"로 취급

    if broker is None:
        return HeldCheck(ledger_shares, True, "ledger" if ledger_shares > 0 else "none")

    try:
        account, positions = await kis_account_snapshot()
    except Exception:
        account, positions = None, {}
    if account is None:
        return HeldCheck(ledger_shares, False, "unknown")  # 장애 = 모름

    if ticker_market(ticker) == "overseas" and not account.get("overseas_ok", True):
        # 해외 잔고 TR만 실패한 부분 스냅샷 — positions에 해외 종목이 통째로
        # 빠져 있으므로 "없음"이 아니라 "모름"이다. 원장이 보유를 확정해 주면
        # 그건 여전히 아는 사실이므로 known=True로 둔다(broker is None 분기와 동형).
        if ledger_shares > 0:
            return HeldCheck(ledger_shares, True, "ledger")
        return HeldCheck(0, False, "unknown")

    kis_shares = int(positions.get(ticker, {}).get("quantity", 0) or 0)
    shares = max(ledger_shares, kis_shares)
    if ledger_shares > 0 and kis_shares > 0:
        source = "both"
    elif kis_shares > 0:
        source = "kis"
    elif ledger_shares > 0:
        source = "ledger"
    else:
        source = "none"
    return HeldCheck(shares, True, source)


async def mark_trade_allowlist_auto_added(ticker: str) -> None:
    """Records that `ticker` reached `trade_allowlist` via an automatic
    mechanism (the AI BUY-decision hook, or the real-holdings
    reconciliation job) rather than an explicit user add — see
    `settings.trade_allowlist_auto_added`'s docstring for the add/prune
    contract this enables. Idempotent. Caller's responsibility to have
    already added `ticker` to trade_allowlist itself; this only sets the
    bookkeeping marker."""
    if ticker not in settings.trade_allowlist_auto_added:
        settings.trade_allowlist_auto_added = [*settings.trade_allowlist_auto_added, ticker]
        await trade_allowlist_store.save("trade_allowlist_auto_added", settings.trade_allowlist_auto_added)


async def unmark_trade_allowlist_auto_added(ticker: str) -> None:
    """Clears `ticker`'s auto-added marker — e.g. once a human explicitly
    confirms interest in it (a future promotion action), so it stops
    being eligible for automatic pruning. Idempotent."""
    if ticker in settings.trade_allowlist_auto_added:
        settings.trade_allowlist_auto_added = [t for t in settings.trade_allowlist_auto_added if t != ticker]
        await trade_allowlist_store.save("trade_allowlist_auto_added", settings.trade_allowlist_auto_added)


async def reconcile_trade_allowlist() -> list[str]:
    """Keeps `settings.trade_allowlist_domestic`/`_overseas` consistent
    with reality, in both directions:

    ADD — a real KIS holding that's off the allowlist (bought manually
    via the KIS app, or held from before this system tracked it) is
    invisible to `stop_loss_loop`, `app.live.auto_trade_loop`'s
    `_held_needing_review`, and the live price stream, and gets its own
    exit order unconditionally BLOCKED by `kis_bridge.py`'s
    `ticker not in settings.trade_allowlist` gate — a real position that
    nothing here would ever protect or let itself be sold. Any such
    ticker is added via `add_ticker_to_trade_allowlist` (so it also gets
    that function's display-info fetch + realtime-feed subscription) and
    marked `trade_allowlist_auto_added`.

    PRUNE — a ticker that only ever reached the allowlist through the AI
    BUY-decision hook (marked auto-added, see that field's docstring)
    but whose order was rejected, expired, or failed to submit never
    became a real position — left alone, it would sit on the allowlist
    forever, re-screened every cycle for no reason. Pruned here if it
    currently has zero real KIS shares AND no currently-open pending
    order (an order still awaiting approval must survive this pass, not
    get pruned out from under itself). A user-added ticker is never
    marked auto-added, so a legitimate "watching, haven't bought yet"
    candidate is never pruned just for having zero shares.

    Skips entirely (both directions) if the KIS account snapshot itself
    is unavailable (`kis_account_snapshot` returns `account=None` on any
    fetch failure or unconfigured broker) — with no reliable read on
    real positions, treating "couldn't tell" as "not held" would mass-
    prune every auto-added ticker on a transient KIS outage. The same
    reasoning applies per-leg (2026-08-26): if only the OVERSEAS balance
    TR failed, `account["overseas_ok"]` is False and every overseas
    ticker is skipped from the PRUNE pass too — a partial snapshot has
    no overseas holdings in `held` at all, so pruning off that would be
    mass-deleting on "couldn't tell", not "confirmed not held".

    Returns the tickers newly ADDED this call, for a caller that wants
    to force those onto this cycle's review list — a freshly discovered
    real holding almost certainly has no stop-loss on record yet (see
    app.live.auto_trade_loop's use of this). Best-effort throughout: one
    ticker's failure (a leveraged/unsupported product type from
    `add_ticker_to_trade_allowlist`, a DB hiccup) is logged and skipped
    rather than aborting the whole pass."""
    account, kis_positions = await kis_account_snapshot()
    if account is None:
        logger.info("reconcile_trade_allowlist: KIS account snapshot unavailable — skipping this cycle")
        return []
    held = {ticker for ticker, pos in kis_positions.items() if pos.get("quantity", 0) > 0}

    newly_added: list[str] = []
    for ticker in held:
        if ticker in settings.trade_allowlist:
            continue
        try:
            status, _ = await add_ticker_to_trade_allowlist(ticker)
        except Exception:
            logger.warning("reconcile_trade_allowlist: could not add held ticker %s", ticker, exc_info=True)
            continue
        if status == "added":
            await mark_trade_allowlist_auto_added(ticker)
            newly_added.append(ticker)

    try:
        pending_tickers = {order.ticker for order in await list_pending_orders()}
    except Exception:
        logger.warning("reconcile_trade_allowlist: could not read pending orders — skipping prune this cycle", exc_info=True)
        return newly_added

    overseas_ok = account.get("overseas_ok", True)
    for ticker in list(settings.trade_allowlist_auto_added):
        if ticker in held or ticker in pending_tickers:
            continue
        if not overseas_ok and ticker_market(ticker) == "overseas":
            # 부분 스냅샷에선 해외 보유가 통째로 안 보이므로 prune 판단 근거가
            # 없음 — 위 docstring의 '모름을 없음으로 취급하면 대량 오삭제'가
            # 전체 실패뿐 아니라 해외 TR만 실패해도 똑같이 성립한다.
            logger.info("reconcile_trade_allowlist: overseas balance unavailable — not pruning %s this cycle", ticker)
            continue
        try:
            await remove_ticker_from_trade_allowlist(ticker)
        except Exception:
            logger.warning("reconcile_trade_allowlist: could not prune %s", ticker, exc_info=True)

    return newly_added


def is_trade_allowlist_auto_added(ticker: str) -> bool:
    return ticker in settings.trade_allowlist_auto_added


def trade_allowlist_names() -> dict[str, str]:
    return {
        ticker: (info.name if (info := _display_info.get(ticker)) else ticker)
        for ticker in settings.trade_allowlist
    }


async def kis_account_snapshot() -> tuple[dict | None, dict[str, dict]]:
    """(account summary, KIS positions keyed by trade-allowlist-style ticker).

    KIS domestic codes are bare ("005930"); the trade allowlist uses yfinance's
    "005930.KS" form, so domestic positions are keyed with the suffix added
    while US positions key by their plain symbol. The KRW account summary
    stays domestic-only (mixing USD into those totals would need FX);
    overseas totals ride along as separate USD fields. Any failure of the
    DOMESTIC balance call — server maintenance window, expired token,
    network — returns (None, {}) so the dashboard keeps working on ledger
    data alone. A failure of the OVERSEAS leg alone is a different, PARTIAL
    failure: domestic positions are still returned, `positions` simply has
    no overseas entries, and `account["overseas_ok"]` is False — a consumer
    that reads a missing overseas ticker from `positions` as "confirmed not
    held" during that window is wrong (see held_shares_for_removal and
    reconcile_trade_allowlist, both fixed for exactly this in 2026-08-26).

    Cached ~8s: it makes three throttled KIS calls (domestic + overseas
    balance + present-balance), and both /api/portfolio and the chat's
    portfolio context ask for it, so an uncached call would cost ~3s each time.
    """
    if await kis_broker() is None:
        return None, {}
    # 35s, not 8s: /api/portfolio (the only caller) polls this every 30s
    # (main.js's pollPortfolio interval) — an 8s TTL was always expired by
    # the time the next poll fired, so this cache was never actually being
    # reused by its own recurring consumer. 35s (a bit above the 30s poll
    # interval) means most polls now hit cache instead of re-querying KIS,
    # same reasoning as the "executions" cache above.
    return await cached(user_key("account"), 35, _kis_account_snapshot_build)


async def _kis_account_snapshot_build() -> tuple[dict | None, dict[str, dict]]:
    broker = await kis_broker()
    if broker is None:
        return None, {}
    try:
        balance = await broker.fetch_balance()
    except Exception as exc:
        # No exc_info on any of the four calls in this function (2026-08-06)
        # — during a KIS/DB hiccup this whole account-panel build gets
        # retried on every poll, so a full traceback per attempt drowned out
        # the rest of the log (11+ times in one day for the overseas call
        # alone) for a failure this one-line message already names.
        logger.warning("KIS balance inquiry failed — account panel degraded: %s", exc)
        return None, {}
    positions = {
        f"{p.code}.KS": {
            "name": p.name,
            "quantity": p.quantity,
            "avg_price": p.avg_price,
            "current_price": p.current_price,
            "eval_amount": p.eval_amount,
            "pnl_amount": p.pnl_amount,
            "pnl_pct": p.pnl_pct,
        }
        for p in balance.positions
    }

    # Hard scope switch (settings.overseas_trading_enabled) — see that
    # setting's own comment. A latch trip (portfolio_risk.overseas_flag_
    # contradicted()) re-enables polling regardless of the flag: the probe
    # already found real overseas exposure, so continuing to skip these
    # calls would keep the dashboard/allowlist blind to a position that
    # demonstrably exists. When disabled and not latched, `overseas_ok` is
    # deliberately left False (NOT True) — "we did not ask" must never be
    # readable as "asked, and you hold nothing"; see that field's own
    # comment below and held_shares_for_removal/reconcile_trade_allowlist,
    # which already treat overseas_ok=False as "unknown, don't act".
    overseas_enabled = settings.overseas_trading_enabled or portfolio_risk.overseas_flag_contradicted()
    overseas_eval = overseas_pnl = overseas_purchase = 0.0
    overseas_count = 0
    overseas_ok = True
    if not overseas_enabled:
        overseas_ok = False
    else:
        try:
            overseas = await broker.fetch_overseas_balance()
            overseas_eval, overseas_pnl = overseas.total_eval_amount, overseas.total_pnl_amount
            overseas_purchase = overseas.purchase_amount
            overseas_count = len(overseas.positions)
            for p in overseas.positions:
                positions[p.code] = {
                    "name": p.name,
                    "quantity": p.quantity,
                    "avg_price": p.avg_price,
                    "current_price": p.current_price,
                    "eval_amount": p.eval_amount,
                    "pnl_amount": p.pnl_amount,
                    "pnl_pct": p.pnl_pct,
                }
        except Exception as exc:
            overseas_ok = False
            logger.warning("KIS overseas balance inquiry failed — domestic only: %s", exc)  # see fetch_balance above for why no exc_info

    # USD orderable cash + true foreign-currency deposit both come from the
    # present-balance TR (the overseas balance TR carries no deposit field);
    # 0 when unavailable/no USD. orderable = 외화출금가능금액1 (매수 여력) —
    # live-verified 2026-07-20 to read 0.000000 for every currency on this
    # account (not a bug: inspected the raw response directly, and the
    # WCRC_FRCR_DVSN_CD "01"/"02" basis switch made no difference either, so
    # there is no alternate call that surfaces a non-zero figure here).
    # Displayed as-is rather than floored to `deposit` — a display floor
    # would show a made-up "원화예수금" figure identical to 외화예수금 right
    # next to it, exactly the confusion reported and fixed here. The
    # order-gate's own floor (kis_bridge.insufficient_cash_block) is a
    # separate, deliberate exception for that one purpose, not this one.
    # foreign_cash_deposit = 외화예수금 (실제 외화예수금, orderable과 다름
    # — 예수금 있어도 출금가능액은 증거금 등으로 더 적을 수 있음).
    overseas_cash = 0.0
    overseas_foreign_cash_deposit = 0.0
    if overseas_enabled:
        try:
            present = await broker.fetch_overseas_present_balance("USD")
            overseas_cash = present.orderable
            overseas_foreign_cash_deposit = present.deposit
        except Exception as exc:
            logger.warning("KIS overseas present-balance inquiry failed: %s", exc)  # see fetch_balance above for why no exc_info

    # KRW per USD, for the dashboard's overseas-card currency toggle. Best-
    # effort: the toggle just doesn't work if yfinance's FX quote is down,
    # everything else on the card still renders in USD.
    usd_krw_rate = None
    try:
        usd_krw_rate = await get_usd_fx_rate("KRW")
    except Exception as exc:
        logger.warning("USD/KRW FX rate fetch failed: %s", exc)  # see fetch_balance above for why no exc_info

    account = {
        # Drives the header pill's 실전/모의 label — read from settings
        # rather than the broker instance so it's correct even before any
        # KIS call succeeds. See memory: KIS 실전투자 안전 규칙.
        "is_real": current_account_mode() == "real",
        # 총평가금액(cash + holdings) vs. total_eval_amount (holdings only) —
        # the hero tile shows both. Uses d2_cash_deposit (D+2예수금), not
        # cash_deposit (예수금총금액) — cash_deposit doesn't drop on a same-day
        # domestic buy until T+2 settlement, so cash_deposit+total_eval_amount
        # double-counts today's purchases (see the identical fix + real
        # incident writeup on app.execution.portfolio_risk.real_account_
        # snapshot). Note this now reads slightly lower than KIS's own app
        # would show for 총자산 on a same-day-trade day — KIS's own tot_evlu_amt
        # has the same dnca_tot_amt-based quirk, which is why 총자산 and D+2예수금
        # are conventionally shown as two separate lines there; this dashboard
        # intentionally bakes the D+2-accurate figure into the hero number
        # instead.
        "total_assets": balance.d2_cash_deposit + balance.total_eval_amount,
        "cash_deposit": balance.cash_deposit,
        "total_eval_amount": balance.total_eval_amount,
        "total_pnl_amount": balance.total_pnl_amount,
        "purchase_amount": balance.purchase_amount,
        # D+2예수금 (가수도정산금액) — domestic only.
        "d2_cash_deposit": balance.d2_cash_deposit,
        "domestic_count": len(balance.positions),
        "position_count": len(balance.positions) + overseas_count,
        # overseas (USD) block — mirrors the domestic fields
        "overseas_total_assets": overseas_eval + overseas_foreign_cash_deposit,
        "overseas_cash_deposit": overseas_cash,
        "overseas_foreign_cash_deposit": overseas_foreign_cash_deposit,
        "overseas_eval_amount": overseas_eval,
        "overseas_pnl_amount": overseas_pnl,
        "overseas_purchase_amount": overseas_purchase,
        "overseas_count": overseas_count,
        "usd_krw_rate": usd_krw_rate,
        # False = the overseas balance TR failed for this build, so `positions`
        # covers DOMESTIC ONLY. Distinct from overseas_count == 0 ("we asked and
        # you hold nothing overseas"): a consumer that reads a missing overseas
        # ticker as "not held" makes a destructive decision on data it never
        # actually got. Readers must use .get("overseas_ok", True) — a caller's
        # test double may not carry this key.
        # 조회 실패와 '보유 없음'을 구분하기 위한 필드.
        "overseas_ok": overseas_ok,
        # True when background overseas polling actually ran this build —
        # settings.overseas_trading_enabled, OR the safety-latch trip that
        # overrides it (see that flag's own comment). Distinct from
        # overseas_ok: this says whether we were ALLOWED to ask, overseas_ok
        # says whether the ask (if made) succeeded.
        "overseas_enabled": overseas_enabled,
    }
    return account, positions


async def portfolio_context() -> str | None:
    """Compact text snapshot of the KIS paper account and its positions, for
    the chat's general-intent system prompt. English on purpose — it's
    agent-facing context, and the model answers in whatever language the
    user asked in. None (never an error) when nothing is held and no
    account is configured. Falls back to the internal paper ledger only
    when the real KIS snapshot itself is unavailable — see the fallback
    loop's own comment below for why it doesn't also run when the real
    snapshot succeeds."""
    lines: list[str] = []
    account, kis_positions = await kis_account_snapshot()
    if account is not None:
        lines.append(
            "KIS paper account (KRW): deposit {:,.0f}; total value {:,.0f}; "
            "PnL {:+,.0f}; {} position(s)".format(
                account["cash_deposit"],
                account["total_eval_amount"],
                account["total_pnl_amount"],
                account["position_count"],
            )
        )
    for ticker, p in kis_positions.items():
        lines.append(
            "- {} {}: {} shares @ avg {:,.0f}, now {:,.0f}, PnL {:+,.0f} ({:+.2f}%) [KIS paper account]".format(
                ticker, p["name"], p["quantity"], p["avg_price"], p["current_price"],
                p["pnl_amount"], p["pnl_pct"],
            )
        )
    # Only fall back to the internal paper ledger when the real KIS snapshot
    # itself couldn't be read (`account is None` — broker not connected or
    # the query failed, which also means `kis_positions` is `{}`). When the
    # snapshot succeeds, `kis_positions` already covers every real holding
    # across both markets (see kis_account_snapshot's docstring), so any
    # ticker missing from it genuinely isn't held — showing it from the
    # internal ledger instead would surface a ghost position the AI can't
    # actually act on (e.g. stale shares left over after a paper account
    # reset; see memory: paper-account-reset-risks).
    # 실제 KIS 스냅샷 조회에 성공했으면 kis_positions가 이미 전체 보유종목을
    # 포함하므로, 내부 원장 fallback은 조회 자체가 실패했을 때만 사용.
    if account is None:
        for ticker in settings.trade_allowlist:
            if ticker in kis_positions:
                continue
            try:
                position = await get_position(ticker)
            except Exception:
                continue
            if position.shares <= 0:
                continue
            info = _display_info.get(ticker)
            tick = latest(ticker)
            price_part = f", last price {tick.price:,.2f}" if tick else ""
            lines.append(
                f"- {ticker} {info.name if info else ticker}: {position.shares} shares "
                f"[local paper ledger, {info.currency if info else '?'}]{price_part}"
            )
    return "\n".join(lines) if lines else None


async def _held_krx_codes(scopes: list) -> set[str]:
    """Every serviced scope's currently-held domestic KRX codes, union'd —
    extracted out of `_prioritized_krx_codes` (2026-08-24) so
    `reconcile_orderbook_zone_promotions` can ask the exact same "is this
    code actually held right now" question without duplicating the
    per-scope real_account_snapshot loop. Failures are per scope and
    degrade to "this scope's holdings don't get priority/promotion this
    round" rather than blocking every other scope's — same posture the
    original loop already had."""
    # Realtime-feed tiering advisory, not a gate — a 60s poller calls this
    # every pass, so the short read-only snapshot cache saves a per-scope
    # ledger inquiry each time (2026-09-01 EGW00215 pressure).
    from app.execution.portfolio_risk import cached_account_snapshot

    held_codes: set[str] = set()
    for scope in scopes:
        try:
            with user_scope(scope.user_id), account_mode_scope(scope.account_mode):
                # max_age=120.0 (2026-09-03 id=1187 audit, FIX-4): this is
                # feed-tiering priority only, not an enforcement read, and
                # `invalidate_account_read_cache()` already clears the cache
                # after every fill — so a 2-minute-stale snapshot can never
                # hide a newly-held position past the next real trade, while
                # cutting a per-poll ledger inquiry that otherwise added to
                # the same EGW00215 quota pressure the default TTL was
                # already too short to spare (see kis_account_read_cache_
                # seconds's own comment).
                snapshot = await cached_account_snapshot("domestic", max_age=120.0)
            held_codes |= {c for t in snapshot.per_ticker_value if (c := krx_code_from_ticker(t))}
        except Exception:
            logger.warning(
                "could not fetch %s/%s's real domestic positions for the realtime feed's "
                "held-code priority/promotion check — their holdings may lose their slot claim",
                scope.user_id, scope.account_mode, exc_info=True,
            )
    return held_codes


def _trailing_orderbook_zone(codes: list[str], held_codes: set[str]) -> frozenset[str]:
    """Walks `codes` from the end, collecting up to `_ORDERBOOK_ZONE_SIZE`
    codes NOT in `held_codes`, stopping early once that many are found. A
    held code is never eligible — held positions keep full trade-tick
    coverage unconditionally, non-negotiably (see _ORDERBOOK_ZONE_SIZE's own
    comment). If fewer than `_ORDERBOOK_ZONE_SIZE` non-held codes exist in
    `codes` at all (e.g. held positions fill most of the cap), the zone is
    just whatever is available — possibly empty — never padded out."""
    zone: list[str] = []
    for code in reversed(codes):
        if code in held_codes:
            continue
        zone.append(code)
        if len(zone) >= _ORDERBOOK_ZONE_SIZE:
            break
    return frozenset(zone)


async def _prioritized_krx_codes() -> tuple[list[str], frozenset[str], frozenset[str]]:
    """Domestic KRX codes for the real-time feed's initial subscribe list,
    held positions first, split into a core 2TR tier (up to _CORE_FEED_CODES)
    and a cb_only 1TR tier (up to _CB_ONLY_ZONE_SIZE more) — see
    _CORE_FEED_CODES's own comment for the registration-budget math — plus
    (2026-08-24) the trailing-unheld orderbook zone within the core list —
    see _ORDERBOOK_ZONE_SIZE's own comment for why that split exists. KIS
    caps combined real-time registrations at 41 per session (2026-04-20
    notice); RealtimeFeed.__post_init__ would also refuse anything past its
    own cap as a backstop, but a plain list-order truncation there would
    arbitrarily favor whichever tickers happen to sort first in the trade
    allowlist — a currently-held position needs real-time circuit-breaker/
    price coverage far more than an unheld screening candidate does (a
    candidate's only real-time need is deciding whether to buy it TODAY; the
    existing 60s REST poller already covers price, though NOT
    circuit-breaker — see cb_only_codes's own comment), so held positions get
    first claim on the limited slots, decided HERE before RealtimeFeed ever
    sees the list.

    Subscribes the UNION across every serviced scope, not just one scope's
    list (Phase 6). Circuit-breaker / VI / halt status is PUBLIC per-ticker
    data, so one feed can serve everybody — but only for codes it actually
    subscribed. Built from a single scope, every other user's tickers would
    have had no real-time coverage, and `kis_cb_fail_open_when_unknown`
    defaults to False, so their domestic orders would all fail closed on
    "circuit-breaker status unknown": safe, and a total trading outage for them.
    서킷브레이커/VI는 공개 데이터라 피드 하나로 전원을 커버할 수 있지만, 구독한
    종목에 대해서만 가능함 — 한 사용자 목록만 구독하면 나머지 사용자는 국내
    주문이 전부 fail-closed로 막힘.

    Held positions still get first claim on the 41 slots, collected per scope
    because whose position it is decides which account snapshot reports it.
    보유 종목 우선권은 유지하되 계좌별 조회가 필요하므로 스코프별로 수집.

    Returns (codes, orderbook_zone, cb_only_zone) for RealtimeFeed's matching
    constructor fields."""
    scopes = await serviced_scopes()
    union: dict[str, None] = {}
    for scope in scopes:
        try:
            with user_scope(scope.user_id), account_mode_scope(scope.account_mode):
                await trade_allowlist_store.ensure_loaded(scope.user_id, scope.account_mode)
                for ticker in settings.trade_allowlist:
                    if (code := krx_code_from_ticker(ticker)) is not None:
                        union.setdefault(code, None)
        except Exception:
            logger.exception(
                "could not read %s/%s's allowlist for the realtime feed — their tickers "
                "will have no circuit-breaker coverage this session",
                scope.user_id, scope.account_mode,
            )
    all_codes = list(union)
    held_codes = await _held_krx_codes(scopes)
    # dict.fromkeys, not a set, to keep trade allowlist order among same-priority codes.
    prioritized = list(dict.fromkeys([c for c in all_codes if c in held_codes] + [c for c in all_codes if c not in held_codes]))
    core = prioritized[:_CORE_FEED_CODES]
    allowlist_tail = prioritized[_CORE_FEED_CODES:]

    # Merge screening candidates (see note_candidate_cb_code) into the
    # cb-only zone alongside the allowlist tail — see _candidate_cb_codes's
    # own comment for why this exists. Priority order: up to
    # _CANDIDATE_CB_RESERVE reserved candidate slots first, then the
    # allowlist tail fills whatever is left, then — only if the reserve
    # wasn't fully used AND the allowlist tail didn't fill the zone —
    # overflow candidates beyond the reserve get the remaining room. A
    # candidate already present in `core` (e.g. it was also a held position)
    # is never duplicated into the cb-only zone.
    candidates = [c for c in _live_candidate_cb_codes() if c not in core]
    reserved_candidates = candidates[:_CANDIDATE_CB_RESERVE]
    overflow_candidates = candidates[_CANDIDATE_CB_RESERVE:]

    cb_tail: list[str] = list(reserved_candidates)
    remaining = _CB_ONLY_ZONE_SIZE - len(cb_tail)
    allowlist_used = [c for c in allowlist_tail if c not in cb_tail][:remaining]
    cb_tail.extend(allowlist_used)
    remaining = _CB_ONLY_ZONE_SIZE - len(cb_tail)
    if remaining > 0 and overflow_candidates:
        cb_tail.extend(c for c in overflow_candidates if c not in cb_tail)
        cb_tail = cb_tail[:_CB_ONLY_ZONE_SIZE]

    dropped_allowlist = [c for c in allowlist_tail if c not in cb_tail]
    dropped_candidates = [c for c in candidates if c not in cb_tail]
    if dropped_allowlist:
        logger.warning(
            "realtime feed: %d domestic KRX codes exceeds the %d-code (%d core 2TR + %d cb-only "
            "1TR) budget (KIS's 41 combined real-time registration limit) — subscribing held "
            "positions + trade allowlist codes by priority, dropping allowlist codes "
            "(REST-poller-only, NO circuit-breaker coverage from here): %s",
            len(all_codes), _CORE_FEED_CODES + _CB_ONLY_ZONE_SIZE, _CORE_FEED_CODES,
            _CB_ONLY_ZONE_SIZE, dropped_allowlist,
        )
    if dropped_candidates:
        logger.warning(
            "realtime feed: %d screening candidate codes could not fit the cb-only zone "
            "(reserve=%d, zone=%d) — dropping candidate codes (no circuit-breaker coverage "
            "until a future rebuild frees a slot): %s",
            len(candidates), _CANDIDATE_CB_RESERVE, _CB_ONLY_ZONE_SIZE, dropped_candidates,
        )
    return core + cb_tail, _trailing_orderbook_zone(core, held_codes), frozenset(cb_tail)


async def reconcile_feed_tier_promotions() -> None:
    """Called once per app.live.poller pass (cheap no-op in the common case)
    to catch a screening candidate that was sitting in the orderbook zone (no
    live trade-tick coverage) OR the cb_only zone (2026-08-28: no live trade
    tick AND no live price at all) and just got bought — a real held position
    MUST regain full (market-status + trade-tick) coverage, since
    app.live.exit_monitor needs live ticks to trigger its stop-loss check,
    and leaving a real position without that coverage for the rest of the
    session is unacceptable (2026-08-24, extended 2026-08-28 to cb_only —
    same-grade incident: a cb_only code that gets bought is exactly as
    exposed as an orderbook-zone one was, just via a different degraded tier).

    RealtimeFeed has no unsubscribe protocol (`_subscribe_msg`'s tr_type is
    hardcoded "1" — see its own comment), so there is no way to flip just
    this one code back to full coverage in place. Instead this does a full
    stop_market_feed() + ensure_market_feed() rebuild — the same already-
    tested teardown/reconnect path a credential change already uses (see
    stop_market_feed's docstring) — which recomputes zones fresh, so the
    promoted code naturally gets full coverage again and a different,
    still-unheld tail code may take its place in the new orderbook/cb_only
    zone. Deliberately not cleverer than this: a full reconnect for this rare
    event (a tail screening candidate actually getting bought) is an
    acceptable, already-proven-safe cost, and patching just one code in place
    would mean building a second, untested reconfiguration path for a live
    safety-gate feed."""
    if _market_feed is None:
        return
    demoted = _market_feed.orderbook_codes | _market_feed.cb_only_codes
    if not demoted:
        return
    scopes = await serviced_scopes()
    held_codes = await _held_krx_codes(scopes)
    promoted = demoted & held_codes
    if not promoted:
        return
    logger.warning(
        "realtime feed: %s in a degraded (orderbook/cb-only) zone just became a held position — "
        "rebuilding the feed to restore full trade-tick/circuit-breaker coverage",
        sorted(promoted),
    )
    await stop_market_feed()
    await ensure_market_feed()


# 2026-10-02: capacity rebuild (see rebuild_market_feed_for_capacity). The
# 10-minute floor keeps every-cycle candidate churn from turning into a
# reconnect loop; the 60s confirm window covers KIS holding the previous
# session ("ALREADY IN USE") while held-position ticks/CB are down.
# 용량 재구성 최소 간격 10분, 재연결 확인 60초.
_CAPACITY_REBUILD_MIN_INTERVAL_SECONDS = 600.0
_CAPACITY_REBUILD_CONFIRM_SECONDS = 60.0
_capacity_rebuild_last_at: float | None = None  # monotonic() of the last rebuild
_capacity_rebuild_confirm_task: asyncio.Task | None = None  # exposed so tests can await it


async def _confirm_feed_reconnected(feed: RealtimeFeed, started: float) -> None:
    """Background check after a capacity rebuild: polls `feed.connected` once a
    second for up to _CAPACITY_REBUILD_CONFIRM_SECONDS. Connected in time ->
    info log with elapsed seconds; otherwise error log + a
    realtime_feed_rebuild_slow alert (held-position live ticks and the CB gate
    stay down meanwhile, so a human needs to know). Never blocks the cycle."""
    try:
        while True:
            if feed.connected:
                logger.info(
                    "realtime feed rebuilt for capacity: connected after %.1fs",
                    time.monotonic() - started,
                )
                return
            if time.monotonic() - started >= _CAPACITY_REBUILD_CONFIRM_SECONDS:
                break
            await asyncio.sleep(1.0)
        logger.error(
            "realtime feed rebuilt for capacity but NOT connected after %.0fs — "
            "KIS may still hold the previous session (ALREADY IN USE); held-position "
            "real-time ticks and circuit-breaker state are unavailable",
            _CAPACITY_REBUILD_CONFIRM_SECONDS,
        )
        await record_alert(
            Alert(
                ticker="-",
                kind="realtime_feed_rebuild_slow",
                message=(
                    f"KIS 실시간 피드 재구성 후 {_CAPACITY_REBUILD_CONFIRM_SECONDS:.0f}초 안에 "
                    "연결되지 않았습니다 — 보유 종목 실시간 시세와 서킷브레이커 확인이 중단된 상태입니다"
                ),
                triggered_at=get_clock().now(),
            )
        )
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.warning("capacity rebuild connection check failed", exc_info=True)


async def rebuild_market_feed_for_capacity(reason: str) -> bool:
    """2026-10-02: the 40-registration budget is consumed intraday by
    candidates' add_code(cb_only=True) and RealtimeFeed has no unsubscribe
    protocol, so once full every later candidate reads CB unknown (fail-closed
    at execution). A full stop_market_feed()+ensure_market_feed() rebuild — the
    path reconcile_feed_tier_promotions already uses — re-applies the candidate
    TTL/reserve in _prioritized_krx_codes and frees the slots. Rate-limited to
    one rebuild per _CAPACITY_REBUILD_MIN_INTERVAL_SECONDS. Returns True only
    if a rebuild was performed. The reconnect check runs in the background."""
    global _capacity_rebuild_last_at, _capacity_rebuild_confirm_task
    feed = _market_feed
    if feed is None:
        return False
    now = time.monotonic()
    if (
        _capacity_rebuild_last_at is not None
        and now - _capacity_rebuild_last_at < _CAPACITY_REBUILD_MIN_INTERVAL_SECONDS
    ):
        logger.info(
            "realtime feed capacity rebuild skipped (last one %.0fs ago, min interval %.0fs): %s",
            now - _capacity_rebuild_last_at, _CAPACITY_REBUILD_MIN_INTERVAL_SECONDS, reason,
        )
        return False
    logger.warning(
        "realtime feed registration budget exhausted — rebuilding feed (%s); "
        "registrations before=%d/%d, capped codes=%d",
        reason, feed.registration_count(), feed.MAX_REGISTRATIONS, len(feed.capped_codes),
    )
    _capacity_rebuild_last_at = now
    await stop_market_feed()
    new_feed = await ensure_market_feed()
    if new_feed is not None:
        _capacity_rebuild_confirm_task = asyncio.create_task(
            _confirm_feed_reconnected(new_feed, time.monotonic())
        )
    return True


async def _verify_configured_kis_credentials() -> None:
    """Boot-time credential check (2026-08-07 single-account rollback — see
    docs/single-user-rollback-guide.md Phase 2).

    KIS keys arrive as env vars now, assembled by Jenkins at deploy time —
    there is no more onboarding screen that proves them against KIS before
    they can be used (app.credentials.validation used to gate PUT
    /api/credentials that way). A typo'd app secret would otherwise surface
    only as a wall of "KIS call failed" warnings once the market opens, with
    nothing pointing at the cause. Issuing a real token for every configured
    mode here — once, at boot — is the same proof the old onboarding screen
    required, just moved to the one place left that runs before anything
    else does.

    Logs and continues rather than refusing to boot: the dashboard itself
    (memory, portfolio history, chat) is useful even when KIS is temporarily
    unreachable, and every order-placing path is independently gated on
    working credentials regardless (require_kis, kis_bridge). Refusing to
    boot here would take down a dashboard that has nothing wrong with it
    over a KIS-side hiccup.
    부팅을 막지 않음 — KIS가 일시적으로 안 되더라도 대시보드 자체(메모리·이력·채팅)는
    쓸모 있고, 주문 경로는 어차피 별도로 자격증명을 확인함."""
    # KIS_TOKEN_CACHE_DIR is set deliberately (Docker volume mount, see
    # pipeline.sh) so one OAuth token survives redeploys. If that dir isn't
    # writable, every deploy silently re-issues a token — the exact thing the
    # volume exists to prevent, and KIS restricts frequent re-issuance. Probe
    # it once here: log loud + record an alert a human sees on the dashboard,
    # but don't fail the boot (this function's contract is log-and-continue).
    # 볼륨이 쓰기 불가면 배포마다 토큰 재발급 — 부팅은 막지 않고 에러 로그+알림만.
    if token_cache_dir_is_explicit():
        cache_dir = token_cache_file().parent
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            probe = cache_dir / f".probe.{os.getpid()}"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
        except OSError as exc:
            logger.error(
                "kis_token_cache_unwritable: KIS_TOKEN_CACHE_DIR=%s is not writable (%s) — "
                "every deploy will re-issue a KIS OAuth token, which KIS restricts",
                cache_dir, exc,
            )
            await record_alert(
                Alert(
                    ticker="-",
                    kind="kis_token_cache_unwritable",
                    message=(
                        f"KIS token cache dir {cache_dir} is not writable ({exc}) — "
                        "OAuth tokens will be re-issued on every deploy"
                    ),
                    triggered_at=get_clock().now(),
                )
            )
    for mode in sorted(settings.configured_kis_modes()):
        creds = settings.kis_credentials(mode)
        if creds is None:  # can't happen given configured_kis_modes(), but no reason to trust that here
            continue
        try:
            await verify_kis_credentials(creds.app_key, creds.app_secret, mode)
        except CredentialsRejected as exc:
            logger.error(
                "KIS rejected the configured %s credentials at boot — check Jenkins Credentials "
                "for KIS_APP_KEY_%s/KIS_APP_SECRET_%s: %s",
                mode, mode.upper(), mode.upper(), exc,
            )
        except CredentialCheckUnavailable as exc:
            logger.warning("could not reach KIS to verify %s credentials at boot: %s", mode, exc)
        else:
            logger.info("KIS %s credentials verified at boot", mode)


@asynccontextmanager
async def lifespan(_: FastAPI):
    global _market_feed
    # Must precede record_session_start() below: that INSERTs into
    # identity_claims, which FKs users(user_id) — on a fresh database there
    # is no users row yet without this. Single-account deployment (2026-08-07
    # rollback), so there is exactly one row to ensure.
    # identity_claims가 users를 FK로 참조하므로, 그 기록보다 먼저 실행돼야 함.
    await register_user(settings.user_id, settings.user_id)
    await record_session_start()  # forensic identity-claim trail — see app.user_context
    await _verify_configured_kis_credentials()
    # Both of these load EVERY (user, mode) scope into a keyed cache, so a
    # real-mode user's overrides survive a boot that happens under the
    # deployment default. Neither depends on a bound identity.
    await risk_rules.load()  # runtime risk-rule overrides → gates use them
    await runtime_settings.load()  # runtime ai_profile/execution_mode overrides
    # Restore trade allowlist/watchlist from a previous session BEFORE anything
    # below reads either list (history seeding, the feed's initial KRX
    # codes) — otherwise every ticker added during a running session (AI
    # auto-add-on-BUY included) is silently lost on every restart.
    #
    # One load per serviced scope (Phase 6). The lists are per (user, mode) and
    # a scope is empty until loaded, so the loops below would otherwise iterate
    # nothing for everybody except whichever scope happened to get loaded.
    #
    # `ensure_loaded`, not `load`, so the loops' own calls are no-ops rather
    # than re-running the migrations on every pass. Failures are per scope: one
    # user's unreadable allowlist must not stop the others from being restored,
    # and must not stop the process from booting at all.
    # 스코프별로 한 번씩 — 실패는 사용자 단위로 격리(부팅 자체를 막지 않음).
    for _scope in await serviced_scopes():
        try:
            await trade_allowlist_store.ensure_loaded(_scope.user_id, _scope.account_mode)
        except Exception:
            logger.exception(
                "could not restore the allowlist for %s/%s at startup — that user's loops "
                "will retry on their next pass", _scope.user_id, _scope.account_mode,
            )
    startup_prefetch_task = asyncio.create_task(_prefetch_startup_data())
    reconciliation_task = asyncio.create_task(trade_reconciliation_loop())
    # One-shot housekeeping: drop alert rows past the retention horizon. Boot
    # time only (see app.live.alerts._ALERT_RETENTION_DAYS) — backgrounded so a
    # slow DELETE never delays the dashboard coming up.
    alert_prune_task = asyncio.create_task(prune_old_alerts())
    poller_task = asyncio.create_task(run_poller_loop())
    stop_loss_task = asyncio.create_task(stop_loss_loop())
    # Tick-based real-time stop-loss check (app.live.exit_monitor) — runs
    # alongside, not instead of, stop_loss_task above: the hourly daily-bar
    # loop stays as the backstop for whatever this one's freshness gate
    # (a missing/stale tick) makes it skip.
    exit_monitor_task = asyncio.create_task(exit_monitor_loop())
    # Confirms a mirrored order's ACTUAL KIS fill (app.live.fill_monitor) —
    # detect-and-alert only, same "human decides" boundary as
    # trade_reconciliation_loop above, extended to an exact order_no
    # match instead of that one's (ticker, date, side) heuristic.
    fill_monitor_task = asyncio.create_task(fill_monitor_loop())
    # POC recommend/auto-trade loop — opt-in like kis_execution_enabled (see
    # settings.auto_trade_enabled's own docstring for why this one needs a
    # deliberate switch on top of execution_mode's default-MANUAL safety).
    auto_trade_task = asyncio.create_task(auto_trade_loop()) if settings.auto_trade_enabled else None
    # Forward-collects H0STASP0 orderbook snapshots for the codes RealtimeFeed
    # puts in its orderbook zone (2026-08-24) — research-only, no live safety
    # gate depends on it (see app.live.orderbook_log's module docstring), so
    # it follows the same cancel-at-shutdown pattern as the other background
    # loops rather than needing any special ordering relative to them.
    orderbook_logger_task = asyncio.create_task(run_orderbook_logger())

    # Real-time market-operation feed (trading-halt / VI / circuit-breaker),
    # plus trade ticks/orderbook/fill notifications — the one WebSocket channel
    # (app.brokers.kis.domestic.websocket.RealtimeFeed).
    #
    # No longer BUILT here (Phase 3): at boot nobody is logged in and
    # credentials are per-user, so there is nothing to authenticate a socket
    # with. This call is an optimistic attempt that succeeds only when the
    # serviced user already has credentials stored from a previous session;
    # otherwise ensure_market_feed() is a no-op and the feed starts on first
    # authenticated use instead. See its docstring.
    # 부팅 시점에는 인증할 자격증명이 없으므로 여기서는 '이미 저장돼 있으면 띄운다'만 시도.
    await ensure_market_feed()
    try:
        yield
    finally:
        startup_prefetch_task.cancel()
        reconciliation_task.cancel()
        alert_prune_task.cancel()
        poller_task.cancel()
        stop_loss_task.cancel()
        exit_monitor_task.cancel()
        fill_monitor_task.cancel()
        if auto_trade_task is not None:
            auto_trade_task.cancel()
        orderbook_logger_task.cancel()
        # Manual '다시 분석' cycles (app.web.routers.live's _analysis_tasks)
        # follow the same shutdown rules as the other background loops here —
        # local import to dodge the circular import (live.py imports `state`
        # at module top).
        from app.web.routers.live import pending_analysis_tasks

        manual_analysis_tasks = pending_analysis_tasks()
        for _task in manual_analysis_tasks:
            _task.cancel()
        # Chat/analyze SSE background tasks (app.web.routers.chat_decisions)
        # — same reasoning as manual_analysis_tasks just above, but these are
        # deliberately NOT cancelled by their own request handler anymore
        # (2026-08-26 fix: a client disconnect used to cancel a still-running
        # deep_dive mid-pipeline, contradicting that module's own docstring).
        # Shutdown is the only place that still cancels them.
        from app.web.routers.chat_decisions import pending_chat_tasks

        chat_tasks = pending_chat_tasks()
        for _task in chat_tasks:
            _task.cancel()
        await stop_market_feed()
        # Close every per-user broker the registry handed out, not just one:
        # each holds an httpx connection pool, and there is no longer a single
        # process broker to close. Read the serviced list here rather than at
        # startup — a user could have been added since.
        for _scope in await serviced_scopes():
            await invalidate_broker_cache(_scope.user_id)

        # cancel() above only requests cancellation — without waiting for it
        # here, lifespan returns while these tasks are still mid-unwind, so
        # "shutdown is done" was never actually true, just unobserved. Bounded
        # by the same timeout uvicorn itself waits for connections
        # (dashboard_graceful_shutdown_timeout_seconds) so this can't itself
        # become the thing that hangs shutdown if a task's cleanup misbehaves.
        # The feed task is not listed: stop_market_feed() above already
        # cancelled it and cleared the module reference, so there is nothing
        # left here to await. Manual analysis tasks are included for the same
        # reason as the rest — cancel() alone doesn't wait for the unwind.
        pending = [
            t for t in (
                startup_prefetch_task, reconciliation_task, alert_prune_task, poller_task, stop_loss_task,
                exit_monitor_task, fill_monitor_task, auto_trade_task, orderbook_logger_task,
                *manual_analysis_tasks, *chat_tasks,
            )
            if t is not None
        ]
        if pending:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*pending, return_exceptions=True),
                    timeout=settings.dashboard_graceful_shutdown_timeout_seconds,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "shutdown: %d background task(s) still running after %.1fs, proceeding anyway",
                    sum(1 for t in pending if not t.done()),
                    settings.dashboard_graceful_shutdown_timeout_seconds,
                )


@asynccontextmanager
async def capture_finalizing_lifespan(app: FastAPI):
    """Finalize the optional recorder even when normal lifespan cleanup fails."""
    # Finish capture after producers unwind. Never initialize a writer here.
    # 생산자 정리 후 기록만 마무리하며 새 writer는 만들지 않는다.
    try:
        async with lifespan(app):
            yield
    finally:
        from app.llm.request_capture import finalize_capture

        try:
            if not await asyncio.to_thread(finalize_capture):
                logger.warning("RM request capture finalization incomplete")
        except Exception:
            logger.exception("RM request capture finalization failed")
