"""Portfolio-level risk coordination — the counterpart to sizing.py's
per-ticker ATR scaling and ledger.py's per-ticker 1% risk cap, both of which
only ever look at one ticker's own book. Nothing upstream of this module
notices that all 10 trade_allowlist tickers could go to Buy simultaneously, that
AAPL/MSFT/NVDA/GOOGL/AMZN/TSLA are all highly-correlated US tech names, or
that the aggregate book is in a multi-ticker drawdown.

Four independent caps, in the same "reduce, never loosen or force-liquidate"
philosophy the existing 1% rule already uses:
  1. Concentration — no single ticker over `portfolio_concentration_max_pct`
     of aggregate equity (native per-market currency live; USD when a
     caller's snapshot spans both markets, e.g. the backtest trade_allowlist).
  2. Correlation-bucket exposure — no cluster of tickers correlated above
     `portfolio_correlation_bucket_threshold` over
     `portfolio_correlation_bucket_max_pct` combined.
  3. Drawdown breaker — once aggregate equity has fallen
     `portfolio_drawdown_breaker_pct` below its own running high-water mark,
     block any exposure-INCREASING trade entirely.
  4. Cash floor — blocks any exposure-INCREASING trade that would push
     aggregate invested value (across every ticker) above
     `1 - portfolio_cash_floor_pct` of aggregate equity. Unlike 1-2, which
     only stop money piling into one ticker or one correlated cluster,
     nothing else stops it piling into many uncorrelated tickers at once.

Each ticker still keeps its own independent, currency-converted cash pool
(see ledger.py::starting_cash_for) — retrofitting one true shared pool
across tickers would touch nearly the entire execution layer's currency
handling. These caps operate on a DERIVED aggregate view for risk
monitoring only; they cannot reallocate capital between books, only refuse
to grow one further.

The high-water mark (`portfolio_risk_state`) is strictly LIVE-only — the
backtest harness tracks its own local, in-memory high-water mark instead
and must never read or write this table. `trades` has no run-id/environment
column, so live and backtest trades already share the same rows keyed only
by ticker; letting a shared drawdown-breaker table cross that same boundary
would let a historical backtest inherit today's live equity state (or vice
versa) and trip the breaker for reasons that have nothing to do with that
backtest's own replay.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import logging
import sys
import time
import zlib
from dataclasses import dataclass
from datetime import date
from typing import Literal, NamedTuple

import asyncpg
import pandas as pd

from app.brokers.kis.broker import krx_code_from_ticker
from app.clock import get_clock, local_today
from app.config import current_account_mode, settings
from app.dataflows.kis_provider import get_last_close, get_price_series
from app.dataflows.yfinance import get_ticker_currency, get_usd_fx_rate
from app.domain.models.account import BrokerBalance, BrokerPosition, OverseasCash
from app.execution import _schema_bootstrap
from app.memory.db import get_pool
from app.user_context import get_current_user_id
from app.execution import risk_rules

Market = Literal["domestic", "overseas"]


def ticker_market(ticker: str) -> Market:
    """Same .KS-suffix convention as app.live.auto_trade_loop._split_trade_allowlist
    — shared here so portfolio_risk.py/safety_switch.py/kis_bridge.py never
    each reimplement the domestic/overseas split rule independently."""
    return "domestic" if ticker.endswith(".KS") else "overseas"


# yfinance ticker for KOSPI Composite — confirmed reachable (2026-08-12).
# Domestic scoring/alpha calculations must never fall back to `SPY`: with
# KOSPI down ~31% from its high while SPY sits near an all-time high (see
# project memory: stage-gate-criteria-draft's market-regime snapshot), a
# domestic decision scored against SPY reads as a large negative alpha
# almost regardless of whether the AI actually chose well — the benchmark,
# not the decision, dominates the number. Confirmed live: 207 already-
# resolved domestic memory_entries were all scored this way (app.memory.
# auto_resolve's SPY-only default), making the current win-rate/alpha
# figures for domestic tickers uninterpretable until re-scored.
_DOMESTIC_BENCHMARK_TICKER = "^KS11"
_OVERSEAS_BENCHMARK_TICKER = "SPY"


def benchmark_for_market(market: Market) -> str:
    """The correct-index counterpart to `ticker_market` — which benchmark a
    ticker in this market should be scored against. See
    `_DOMESTIC_BENCHMARK_TICKER`'s own comment for why this must never
    silently default to SPY for a domestic ticker."""
    return _DOMESTIC_BENCHMARK_TICKER if market == "domestic" else _OVERSEAS_BENCHMARK_TICKER


# One lock per (market, account_mode) — serializes the live decide-and-commit
# sequence (fresh real_account_snapshot -> apply_portfolio_caps -> record_trade
# -> KIS mirror) in app.execution.ledger.rebalance_to_target, so two tickers in
# the SAME market+account can never both fetch a snapshot before either's
# trade is reflected in it. Without this, concurrently-analyzed tickers (see
# app.live.auto_trade_loop's candidate parallelization) could each see the
# other's pre-trade balance and jointly approve exposure that, combined,
# breaches concentration/correlation/drawdown caps neither alone would have.
# Domestic and overseas never block each other, and neither does paper vs
# real — same independence as every other domestic/overseas/account_mode
# separation in this project. Only meaningful for the live path; backtests
# never touch this (they pass their own precomputed `portfolio_cap_override`
# and never call `apply_live_portfolio_caps`).
#
# A Postgres advisory lock (2026-08-04 — see project memory:
# user-account-mode-db-separation-design), not a plain asyncio.Lock — the
# in-memory version only ever serialized within ONE Python process. Two
# separate processes (a second `python -m app.daemon`, a stray duplicate
# launch, or eventually two people's instances sharing one account) each had
# their OWN lock that knew nothing about the other, so the exact race this
# lock exists to prevent could still happen across process boundaries —
# the one thing this lock is FOR. `pg_advisory_lock` is session-scoped (tied
# to one physical connection, not a transaction), so the connection is held
# for the lock's entire duration rather than the usual acquire-query-release
# pattern every other function in this module uses. Falls back to a no-op
# (never blocks) only if the DB itself is unreachable during acquisition,
# same degrade-quietly convention as everything else here — losing this
# lock reopens the old single-process race window, but the real-balance
# check it wraps is still the actual safety net, not this lock itself.
# CONTENDED (another process actively holds it) is different and
# deliberately NOT fail-open — see _MarketAdvisoryLock's own docstring and
# MarketLockContended below.
# 국내/해외+계좌모드별 별도 DB 레벨 락(Postgres advisory lock)으로 교체 — 기존
# asyncio.Lock은 프로세스 하나에서만 유효해서 여러 프로세스/컴퓨터 사이의 레이스를
# 못 막았음. 백테스트는 이 락을 아예 안 씀.
_ADVISORY_LOCK_NAMESPACE = 928374  # arbitrary constant; this module's only user of pg_advisory_lock
_LOCK_POLL_INTERVAL_SECONDS = 0.25


def _lock_key2(market: Market, account_mode: str, user_id: str) -> int:
    """The advisory lock's second key, now including the user (multi-user
    migration Phase 4, plan §2.2).

    It used to be a fixed four-entry table keyed (market, account_mode), which
    meant every user's order commits serialized against each other: one
    person's slow decide-and-commit blocked everyone else's, for no safety
    benefit at all — the invariant this lock protects (don't let two commits
    race on the same position) is per account, and different users have
    different accounts.
    예전엔 (시장, 계좌모드)만으로 키를 잡아 모든 사용자의 주문이 서로 직렬화됐음 —
    보호하려는 불변식은 계좌 단위라 사용자끼리 막을 이유가 없었음.

    CRC32 masked to 31 bits so it fits the int4 the two-argument
    `pg_advisory_lock(int, int)` form takes, and stays positive. A hash
    collision would make two unrelated users serialize against each other —
    the old behavior for that pair, so a slowdown, never a correctness
    failure. That is why a hash is acceptable here and would not be for
    anything that decides access.
    해시 충돌은 두 사용자가 불필요하게 직렬화되는 성능 문제일 뿐 정확성 문제는 아님.
    """
    return zlib.crc32(f"{market}:{account_mode}:{user_id}".encode("utf-8")) & 0x7FFFFFFF


class MarketLockContended(Exception):
    """Raised by `_MarketAdvisoryLock.__aenter__` when another process is
    still holding the same (market, account_mode, user) advisory lock after
    `settings.portfolio_risk_lock_wait_seconds` of polling — an explicit
    "block this trade decision" outcome, deliberately NOT the fail-open
    path used when the DB itself is unreachable (see this module's own
    top-of-file comment). Safe to let propagate: every live caller of
    `rebalance_to_target` already runs behind app.live.auto_trade_loop's
    per-ticker try/except-and-retry, so this costs one decision cycle for
    one ticker, not the daemon."""


class _MarketAdvisoryLock:
    """`async with market_lock(market):` — see the module-level comment
    above for why this holds a Postgres advisory lock instead of an
    in-memory asyncio.Lock.

    Uses `pg_try_advisory_lock` (non-blocking, returns immediately) polled
    in a short client-side loop rather than the blocking `pg_advisory_lock`
    this used before (2026-08-06) — `pg_advisory_lock` blocks server-side
    for as long as the contending session holds the lock, which after
    `database_command_timeout_seconds` was widened to 125s (see that
    setting's own comment) meant a contended lock could stall a live
    decision for up to two minutes before the client's own command timeout
    finally cancelled the wait and this fell into the fail-open branch below
    — silently giving up the exact serialization this lock exists for,
    right when it mattered most (two processes actually racing). Polling
    with a short, dedicated `portfolio_risk_lock_wait_seconds` budget makes
    that wait bounded and independent of the query timeout, and turns
    genuine contention into an explicit `MarketLockContended` (fail-closed)
    instead of a silent fail-open."""

    def __init__(self, market: Market):
        self._market = market
        self._pool: asyncpg.Pool | None = None
        self._conn: asyncpg.Connection | None = None
        self._key2: int | None = None

    async def __aenter__(self) -> "_MarketAdvisoryLock":
        self._key2 = _lock_key2(self._market, current_account_mode(), get_current_user_id())
        try:
            self._pool = await _ready_pool()
            conn = await self._pool.acquire()
        except Exception as exc:  # noqa: BLE001 - DB itself unreachable, fail-open (unchanged)
            logger.warning(
                "portfolio_risk: market_lock pool acquire failed for %s (%s) — proceeding WITHOUT "
                "cross-process serialization this one time: %s", self._market, current_account_mode(), exc,
            )
            self._conn = None
            return self

        # From here on, `conn` is checked out of the pool and MUST be
        # released on every exit path (previously only released once the
        # lock was actually held, leaking one connection per failed/
        # contended attempt — with database_pool_max_size=5, 5 leaks stalls
        # every trade in this market+account_mode permanently).
        try:
            deadline = asyncio.get_event_loop().time() + settings.portfolio_risk_lock_wait_seconds
            while True:
                acquired = await conn.fetchval(
                    "SELECT pg_try_advisory_lock($1, $2)", _ADVISORY_LOCK_NAMESPACE, self._key2
                )
                if acquired:
                    self._conn = conn  # only set once the lock is actually held
                    return self
                if asyncio.get_event_loop().time() >= deadline:
                    raise MarketLockContended(
                        f"{self._market}/{current_account_mode()} advisory lock still held by "
                        f"another process after {settings.portfolio_risk_lock_wait_seconds}s"
                    )
                await asyncio.sleep(_LOCK_POLL_INTERVAL_SECONDS)
        except MarketLockContended:
            await self._pool.release(conn)
            raise
        except Exception as exc:  # noqa: BLE001 - genuine DB error mid-poll, fail-open (unchanged)
            await self._pool.release(conn)
            logger.warning(
                "portfolio_risk: market_lock acquire failed for %s (%s) — proceeding WITHOUT cross-process "
                "serialization this one time: %s", self._market, current_account_mode(), exc,
            )
            self._conn = None
            return self

    async def __aexit__(self, *exc_info) -> bool:
        if self._conn is not None:
            try:
                await self._conn.execute("SELECT pg_advisory_unlock($1, $2)", _ADVISORY_LOCK_NAMESPACE, self._key2)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "portfolio_risk: market_lock release failed for %s (%s): %s",
                    self._market, current_account_mode(), exc,
                )
            finally:
                await self._pool.release(self._conn)
        return False


def market_lock(market: Market) -> _MarketAdvisoryLock:
    return _MarketAdvisoryLock(market)


# Module-level so `app.execution.ledger.rebalance_to_target` can detect a
# fail-closed portfolio-risk block by prefix (FIX-5, 2026-09-03 id=1187
# audit) and raise a `portfolio_risk_unavailable_block` alert for it — an
# outage-shaped block was otherwise indistinguishable from an ordinary
# concentration/correlation/cash-floor cap trip anywhere outside the log
# line, even though it means a BUY the AI decided on never executed at all.
PORTFOLIO_RISK_UNAVAILABLE_NOTE_PREFIX = "Portfolio risk check unavailable"


class AccountUnavailable(Exception):
    """Raised by real_account_snapshot() when the real KIS balance for a
    market can't be fetched — broker not connected, or the KIS call itself
    failed. Callers must fail closed (block exposure-increasing trades),
    never silently substitute a stale/default total.
    실제 잔고 조회 실패 시 예외로 알림 — 호출부는 반드시 fail-closed 처리."""

# `get_position` is imported lazily inside `_ticker_usd_value` rather than at
# module load: ledger.py calls INTO this module (apply_live_portfolio_caps),
# so importing ledger.py at the top here would be circular. Same "local
# import to avoid a hard dependency at module load" convention already used
# in app.live.store/app.live.alerts for the same reason.

logger = logging.getLogger(__name__)

# Replaces the old single-row portfolio_risk_state table (id=1, one blended
# domestic+overseas total) — a new table rather than an ALTER since the key
# shape changes entirely (single fixed id -> market-keyed). The old table is
# simply left unused; no migration needed for a personal single-user system,
# and its recorded high-water mark was against a different data source
# (the internal ledger, not the real KIS balance) anyway, so carrying it
# forward would be meaningless.
# 기존 단일행 테이블 대신 시장별 키를 쓰는 새 테이블 — 예전 값은 다른 데이터 소스(내부 장부)
# 기준이라 이어받을 의미가 없어 그냥 버림(개인용 시스템이라 마이그레이션 불필요).
_SCHEMA = """
CREATE TABLE IF NOT EXISTS portfolio_drawdown_state (
    market           TEXT NOT NULL CHECK (market IN ('domestic', 'overseas')),
    high_water_mark  DOUBLE PRECISION NOT NULL,
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Paper/real account separation (2026-08-04 — see project memory:
-- user-account-mode-db-separation-design). Before this column, the HWM was
-- keyed by market alone, so a peak reached during paper testing (a
-- completely different equity scale from a real account) would carry
-- straight over the moment kis_account_mode flips to 'real' — the real
-- account's actual total could look like a large drawdown against that
-- inflated paper peak from the very first live snapshot, blocking every
-- new domestic/overseas buy for no real reason (the exact mechanism
-- already known from the 2026-07-28 paper-reset incident, just triggered
-- by a mode switch instead of a KIS-side reset this time). Dropping the
-- single-column primary key in favor of a unique index on (market,
-- account_mode) rather than an ADD CONSTRAINT, since Postgres has no
-- `ADD CONSTRAINT IF NOT EXISTS` and this whole block re-runs on every
-- process start — a unique index is what ON CONFLICT actually needs
-- anyway, and CREATE UNIQUE INDEX IF NOT EXISTS is properly idempotent.
ALTER TABLE portfolio_drawdown_state DROP CONSTRAINT IF EXISTS portfolio_drawdown_state_pkey;
ALTER TABLE portfolio_drawdown_state ADD COLUMN IF NOT EXISTS account_mode TEXT NOT NULL DEFAULT 'paper'
    CHECK (account_mode IN ('paper', 'real'));

-- Per-user separation (multi-user migration Phase 4, plan §2.2). The
-- high-water mark is the reference point the drawdown breaker measures
-- against, so one shared HWM meant a user with a large account set the
-- benchmark that halted a user with a small one — and the GREATEST() upsert
-- below makes it ratchet, so the largest account's peak silently became
-- everyone's. Backfilled to 'default' like account_mode was to 'paper'.
-- HWM이 공유되면 큰 계좌의 고점이 작은 계좌의 손실 기준이 됨(GREATEST 갱신이라
-- 한 번 올라가면 내려오지도 않음).
ALTER TABLE portfolio_drawdown_state ADD COLUMN IF NOT EXISTS user_id TEXT NOT NULL DEFAULT 'default';
DROP INDEX IF EXISTS portfolio_drawdown_state_market_mode_uidx;
CREATE UNIQUE INDEX IF NOT EXISTS portfolio_drawdown_state_market_mode_user_uidx
    ON portfolio_drawdown_state (market, account_mode, user_id);
"""

_schema_ready = False
_schema_lock = asyncio.Lock()


async def _ready_pool() -> asyncpg.Pool:
    """Double-checked locking around the first-time DDL — shared with the
    other app.execution modules via _schema_bootstrap."""
    return await _schema_bootstrap.ready_pool(sys.modules[__name__], _SCHEMA)


@dataclass
class PortfolioSnapshot:
    """A point-in-time aggregate view across the whole trade_allowlist — NOT a
    claim that there's one shared cash pool (see module docstring). Currency
    depends on the caller: `real_account_snapshot` (live, always scoped to
    ONE market) keeps each market's own native currency (KRW for domestic,
    USD for overseas) since it never mixes the two; `portfolio_snapshot`
    (backtest, walks the WHOLE trade_allowlist which mixes both markets) still
    normalizes to USD since summing raw KRW+USD would be meaningless."""

    per_ticker_value: dict[str, float]
    total_value: float


# A `portfolio_cap_override` that can never bind — an infinite total means
# any real position is 0% of it, so concentration/bucket checks always pass,
# and 0.0 drawdown means the breaker never trips. For callers (single-ticker
# `run_backtest`, and `run_pipeline` when called with `as_of` set) that must
# never touch the live-only `portfolio_risk_state` table but also aren't yet
# wired to the multi-ticker harness's own local-state override.
DISABLED_PORTFOLIO_CAPS: tuple[PortfolioSnapshot, dict[str, int], float] = (
    PortfolioSnapshot(per_ticker_value={}, total_value=float("inf")),
    {},
    0.0,
)


async def _ticker_usd_position_and_cash(
    ticker: str, as_of: date | None, price_override: float | None = None
) -> tuple[float, float]:
    """Returns (position_usd, cash_usd) SEPARATELY — never pre-summed — so
    `portfolio_snapshot` can put position value in `per_ticker_value` and
    fold cash into `total_value` exactly once across the whole trade_allowlist.
    Bug 1 fix (2026-08-14): this used to return a single
    `position.market_value(price)` (= cash + shares*price) per ticker, which
    `portfolio_snapshot` then used directly as `per_ticker_value[ticker]` —
    so a ticker with 0 shares still reported its full cash balance as
    "position value", making a flat ticker look like it occupied 50% of the
    book (2 tickers, both flat) and, via `apply_portfolio_caps`'s cash-floor
    check, made `cash_before` come out to exactly 0 for every trade (see
    this module's top-of-file docstring and _log_cap_reduction's bug-1/bug-2
    comments)."""
    from app.execution.ledger import get_position  # local import — see module-level note above

    position = await get_position(ticker)
    price = price_override if price_override is not None else await get_last_close(ticker, as_of=as_of)
    currency = await get_ticker_currency(ticker)
    fx_rate = await get_usd_fx_rate(currency)  # units of `currency` per 1 USD
    position_usd = (position.shares * price) / fx_rate
    cash_usd = position.cash / fx_rate
    return position_usd, cash_usd


async def portfolio_snapshot(
    tickers: list[str],
    as_of: date | None = None,
    price_overrides: dict[str, float] | None = None,
) -> PortfolioSnapshot:
    """Aggregate USD-equivalent value of every given ticker's own book, from
    this app's internal simulated per-ticker ledger (ledger.py's `trades`
    table) — NOT the real KIS account balance (see `real_account_snapshot`
    below for that). BACKTEST-ONLY now: every live caller uses
    `real_account_snapshot` instead, since the internal ledger's own
    per-ticker starting cash has no relationship to what a real KIS account
    actually holds. Kept here because the backtest harness has no real
    broker to query and must simulate its own book from these ledger rows.
    백테스트 전용 — 라이브 경로는 전부 real_account_snapshot()을 씀(실제 KIS 잔고 기준).

    Each ticker's POSITION is always fetched fresh — never cached, since it
    must reflect intra-walk-updated state (a backtest deciding ticker 3 of
    10 on a given date needs to see tickers 1-2's just-recorded trades from
    that same date).

    `price_overrides`, if given, skips the `get_last_close` network fetch
    for any ticker present in it — for a fixed `as_of`, a ticker's close is
    already deterministic, so a caller that's walking trade dates with its
    own already-fetched price series (the multi-ticker backtest harness) can
    pass those straight in instead of re-fetching the same historical close
    from yfinance on every one of the many `portfolio_snapshot` calls made
    per date.

    `per_ticker_value` holds POSITION value only (shares * price), never
    cash — matching `real_account_snapshot`'s live convention (see that
    function's own per_ticker construction). Cash is summed across every
    ticker and folded into `total_value` exactly once, so
    `total_value == sum(position_usd) + sum(cash_usd)` — numerically
    identical to the old `sum(cash+pos per ticker)` total, just no longer
    double-counted into any single ticker's own bucket (bug 1 fix,
    2026-08-14 — see `_ticker_usd_position_and_cash`'s docstring)."""
    per_ticker: dict[str, float] = {}
    total_cash = 0.0
    for ticker in tickers:
        try:
            override = price_overrides.get(ticker) if price_overrides else None
            position_usd, cash_usd = await _ticker_usd_position_and_cash(ticker, as_of, price_override=override)
            per_ticker[ticker] = position_usd
            total_cash += cash_usd
        except Exception:
            logger.warning("portfolio_snapshot: could not value %s, treating as 0", ticker, exc_info=True)
            per_ticker[ticker] = 0.0
    return PortfolioSnapshot(per_ticker_value=per_ticker, total_value=sum(per_ticker.values()) + total_cash)


# Request-scoped cache (2026-08-06) for the raw fetch_balance/
# fetch_overseas_balance calls `real_shares_held`/`real_equity_native` each
# make independently — `ledger.rebalance_to_target` calls both, back-to-
# back, for the SAME ticker with no trade in between, collapsing what would
# be 2 separate KIS network round-trips into 1.
# `contextvars.ContextVar`, not a plain dict, so the cache is scoped to
# whichever call stack opened it (`cached_balance_fetches` below) and
# invisible to any other concurrent request — a Python default value
# argument or module-level dict would leak across concurrent tickers.
# (This comment used to say those two calls happen "BEFORE market_lock is
# ever acquired" — true when written, stale since the same day's later
# widening of the lock to cover the real-balance read; corrected 2026-08-10
# so the stated reason matches where the code actually runs.)
#
# Used by `real_account_snapshot` (and hence `apply_live_portfolio_caps`)
# ONLY when the scope was opened with `enforcement_safe=True` — i.e. only
# when the caller holds this market's `market_lock` for the scope's entire
# duration, so nothing else in this market+account_mode can change the real
# balance while the scope is open (see `_balance_fetch_enforcement_safe`'s
# own comment for the id=1187 incident this fixes, and `market_lock`'s
# docstring for why that guarantee holds). Outside such a scope, reusing a
# balance fetched before a market_lock was won would miss trades that landed
# while still queuing for it — the exact cross-ticker portfolio-cap race
# `market_lock` exists to close — so `real_account_snapshot` fetches fresh
# by default (`enforcement_safe` defaults to False). `real_position_detail`
# is narrative-only and always safe to reuse a same-scope cache hit from,
# enforcement_safe or not — see that function's own comment.
#
# For deduping `real_account_snapshot` specifically, see the separate
# `_snapshot_cache`/`cached_account_snapshots` below, which is safe only
# because it is opened AFTER the lock is held rather than around it.
_balance_fetch_cache: contextvars.ContextVar[dict[str, object] | None] = contextvars.ContextVar(
    "_balance_fetch_cache", default=None
)

# Set only while a `cached_balance_fetches(enforcement_safe=True)` scope is
# open — the single condition under which `real_account_snapshot` (and
# therefore `apply_live_portfolio_caps`/`safety_switch.daily_loss_block`) may
# reuse a balance already fetched earlier in this same scope instead of
# calling the broker again. True is only ever passed by
# `app.execution.ledger.rebalance_to_target`'s `market_lock` hold
# (`portfolio_cap_override is None`, i.e. live, real `market_lock` held) —
# see that call site's own comment. 2026-09-03 id=1187: within one
# `market_lock` hold, `real_shares_held` (via `_cached_fetch_balance`) had
# already fetched and cached the domestic balance, but
# `apply_live_portfolio_caps` -> `real_account_snapshot` ignored that cache
# and called `broker.fetch_balance()` again — the second call is what hit
# KIS's EGW00215 per-second ledger-inquiry limit and fail-closed-blocked the
# BUY. `enforcement_safe` makes that reuse explicit and opt-in rather than
# widening the existing narrow-scope cache silently (see the 456-462 comment
# above forbidding exactly that for callers outside a held market_lock).
_balance_fetch_enforcement_safe: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "_balance_fetch_enforcement_safe", default=False
)

# Counts actual broker calls (never cache hits) made while a
# `cached_balance_fetches()` scope is open — FIX-6, 2026-09-03 id=1187 audit.
# Reset to 0 at scope entry so it reports per-hold, not cumulative-per-task,
# counts; read via `ledger_inquiry_count()` by a caller still inside the
# scope (or by an inner context manager entered/exited before the scope's
# own reset — see app.execution.ledger's use via AsyncExitStack).
_ledger_inquiry_count: contextvars.ContextVar[int] = contextvars.ContextVar(
    "_ledger_inquiry_count", default=0
)


def _bump_ledger_inquiry_count() -> None:
    _ledger_inquiry_count.set(_ledger_inquiry_count.get() + 1)


def ledger_inquiry_count() -> int:
    """Broker-call count so far in the currently-open `cached_balance_fetches`
    scope (0 outside one). See that ContextVar's own comment."""
    return _ledger_inquiry_count.get()


@contextlib.asynccontextmanager
async def cached_balance_fetches(*, enforcement_safe: bool = False):
    """`async with cached_balance_fetches():` — opens the short-lived cache
    described above for the enclosed block. See that comment for exactly
    which calls participate and why the scope is deliberately this narrow.

    `enforcement_safe=True` additionally asserts that the caller holds this
    market's `market_lock` for the whole scope — the one condition under
    which `real_account_snapshot` (and hence `apply_live_portfolio_caps`/
    `safety_switch.daily_loss_block`) may reuse a balance an earlier reader
    in this same scope already fetched, instead of always re-asking the
    broker. Default False: a scope opened outside a held market_lock (e.g.
    `app.orchestrator.pipeline`'s narrative reads) still dedupes its OWN
    repeated fetches, but stays invisible to `real_account_snapshot`, which
    keeps fetching fresh — see `_balance_fetch_enforcement_safe`'s own
    comment for the id=1187 incident this distinction fixes."""
    token = _balance_fetch_cache.set({})
    safe_token = _balance_fetch_enforcement_safe.set(enforcement_safe)
    count_token = _ledger_inquiry_count.set(0)
    try:
        yield
    finally:
        _balance_fetch_cache.reset(token)
        _balance_fetch_enforcement_safe.reset(safe_token)
        _ledger_inquiry_count.reset(count_token)


# A SEPARATE cache from `_balance_fetch_cache` above, for whole
# `real_account_snapshot` results, and it exists under one strict condition:
# the scope must be opened only AFTER `market_lock` is held, and closed
# before it is released (`cached_account_snapshots`, opened by
# ledger.rebalance_to_target).
#
# That condition is what separates this from the widening the comment above
# forbids. The danger there is reusing a balance fetched BEFORE the lock:
# it would miss trades that landed while we were still queuing for it, which
# is precisely the cross-ticker race `market_lock` exists to close. A fetch
# made after the lock is won cannot miss any of those — and no other trade
# in this market+account_mode can land during the hold, because we are the
# one holding the lock.
#
# What it buys: `rebalance_to_target` reaches `real_account_snapshot` twice
# inside a single hold — once via `apply_live_portfolio_caps`, then again
# moments later via `mirror_trade_to_kis` -> safety_switch.daily_loss_block.
# Nothing between them changes the KIS-side balance (`record_trade` writes
# the internal ledger only; the actual `submit_order` comes after both), so
# the second fetch re-asks KIS a question it just answered. On the paper
# account's 1 req/sec throttle that duplicate costs ~1s domestic / ~2s
# overseas (overseas needs two TRs) of extra LOCK HOLD time, against a
# `portfolio_risk_lock_wait_seconds` budget of 10s after which a contending
# ticker's trade is dropped outright rather than delayed.
#
# Secondary benefit, arguably the more important one: the portfolio caps and
# the daily-loss gate now judge the SAME account state instead of two fetches
# that can disagree mid-decision.
#
# Known and accepted: `force_exit_position` (stop-loss exits) does NOT take
# `market_lock` — only a per-ticker asyncio.Lock — so a forced exit on a
# DIFFERENT ticker in this market can still land mid-hold and make the
# cached snapshot marginally stale. Harmless for both readers: an exit swaps
# position value for cash at roughly market price, leaving `total_value`
# (the only field either gate reads in aggregate) essentially unchanged.
#
# That harmlessness argument covers `total_value` only — it does NOT cover
# `trades.stop_loss` visibility for the SAME ticker. 2026-08-21: do not
# read this "known and accepted" as license to also skip `market_lock` for
# force_exit_position without checking `app.execution.ledger`'s own
# `_force_exit_locks` comment first — a live incident (record-reversal-bug-
# evidence-strengthened-2026-08-21 project memory) traced a stop_loss==
# fill_price race to exactly this asymmetry, for a caller
# (`rebalance_to_target`, same market_lock this file guards) reading the
# open stop-loss for a ticker `force_exit_position` was mid-reversal on.
# Fixed on the READ side (`ledger._is_provisional_forced_exit`), not by
# adding `market_lock` to force_exit_position — see that function's own
# comment for why a safety exit must never wait on or be blocked by a
# market-wide lock.
# 락을 잡은 뒤 열리는 스코프 전용 캐시 — 락 획득 이전 잔고를 재사용하는 것(위 주석이
# 금지하는 것)과 달리, 락 유지 중에는 다른 매매가 낄 수 없으므로 안전함.
_snapshot_cache: contextvars.ContextVar[dict[str, "PortfolioSnapshot"] | None] = contextvars.ContextVar(
    "_snapshot_cache", default=None
)


@contextlib.asynccontextmanager
async def cached_account_snapshots():
    """`async with cached_account_snapshots():` — collapses repeated
    `real_account_snapshot(market)` calls within one market_lock hold into a
    single KIS fetch per market. **Only ever open this inside a held
    `market_lock`** — see the comment above for why that is the entire basis
    of its correctness."""
    token = _snapshot_cache.set({})
    try:
        yield
    finally:
        _snapshot_cache.reset(token)


def _domestic_equity_total(balance: BrokerBalance) -> float:
    """d2_cash_deposit (D+2예수금/가수도정산금액), not cash_deposit
    (예수금총금액) — cash_deposit doesn't drop on a same-day domestic buy
    until T+2 settlement, so summing it with total_eval_amount double-counts
    today's purchases (that cash still shows as "available" AND the bought
    position's value is added on top). Confirmed live 2026-07-30: a
    ₩1,624,000 same-day buy inflated a total computed this way by
    ~₩1,573,600 — matches the purchase amount almost exactly, not a real
    gain. Shared by `real_account_snapshot`/`real_equity_native`'s domestic
    branches, which previously each reimplemented this line independently."""
    return balance.d2_cash_deposit + balance.total_eval_amount


def _overseas_equity_total(balance: BrokerBalance, cash: OverseasCash) -> float:
    """cash.deposit + total_eval_amount — raises AccountUnavailable if
    `cash.matched` is False, meaning the present-balance TR returned no
    matching USD row (a zeroed placeholder, not a confirmed zero balance —
    see OverseasCash's own docstring). Treating that placeholder as real
    cash previously reached daily_loss_block as a data-gap-shaped "huge
    loss" and tripped the breaker on nothing. Shared by
    `real_account_snapshot`/`real_equity_native`'s overseas branches, which
    previously each reimplemented this check independently."""
    if not cash.matched:
        raise AccountUnavailable("overseas present-balance TR returned no matching USD row")
    return cash.deposit + balance.total_eval_amount


async def _cached_fetch_balance(broker) -> BrokerBalance:
    cache = _balance_fetch_cache.get()
    if cache is None:
        _bump_ledger_inquiry_count()
        return await broker.fetch_balance()
    if "domestic" not in cache:
        _bump_ledger_inquiry_count()
        cache["domestic"] = await broker.fetch_balance()
    return cache["domestic"]


async def _cached_fetch_overseas_balance(broker) -> BrokerBalance:
    cache = _balance_fetch_cache.get()
    if cache is None:
        _bump_ledger_inquiry_count()
        return await broker.fetch_overseas_balance()
    if "overseas" not in cache:
        _bump_ledger_inquiry_count()
        cache["overseas"] = await broker.fetch_overseas_balance()
    return cache["overseas"]


async def _cached_fetch_overseas_present_balance(broker, currency: str) -> OverseasCash:
    """Sibling of `_cached_fetch_overseas_balance` for the separate
    present-balance TR (`fetch_overseas_present_balance`) that actually
    carries the USD cash figure — see `real_account_snapshot`'s overseas
    branch comment for why the two are fetched separately. Added alongside
    FIX-1 (2026-09-03, id=1187) so `real_account_snapshot` can reuse this
    call too, not just `fetch_overseas_balance`."""
    cache = _balance_fetch_cache.get()
    if cache is None:
        _bump_ledger_inquiry_count()
        return await broker.fetch_overseas_present_balance(currency)
    key = f"overseas_present:{currency}"
    if key not in cache:
        _bump_ledger_inquiry_count()
        cache[key] = await broker.fetch_overseas_present_balance(currency)
    return cache[key]


async def real_account_snapshot(market: Market) -> PortfolioSnapshot:
    """LIVE per-market aggregate, sourced from the REAL connected KIS account
    balance — kept in that market's OWN currency (KRW for domestic, USD for
    overseas), never converted or combined across markets (see module
    docstring's domestic/overseas separation principle). No FX round-trip
    needed since every caller of this snapshot (apply_live_portfolio_caps,
    safety_switch's daily-loss check) only ever compares it against another
    value from the SAME market — an earlier version converted domestic to
    USD here for a since-removed cross-market blended check, but every
    caller passes domestic tickers' prices in native KRW too, and forgetting
    that mismatched conversion once already caused a real bug (a KRW price
    compared against a USD total looked ~1,400x too concentrated). Raises
    AccountUnavailable if the broker isn't connected or the KIS call fails —
    callers must treat that as fail-closed, never fall back to some other
    total, since a wrong/stale total could silently under- or over-state
    real risk. 실제 KIS 잔고 기준, 국내/해외 절대 안 섞음 — 국내는 원화 그대로(환전 안 함),
    조회 실패시 예외로 호출부에 fail-closed를 강제.

    Served from `cached_account_snapshots()` when a caller has opened that
    scope — see its own comment for the strict "inside a held market_lock
    only" condition that makes reuse safe here. Outside such a scope
    (every other caller) this fetches fresh, exactly as before. A FAILURE is
    never cached: AccountUnavailable must stay fail-closed per call, not get
    pinned for the rest of the scope.

    Also reuses `_cached_fetch_balance`/`_cached_fetch_overseas_balance`/
    `_cached_fetch_overseas_present_balance` — but ONLY while
    `_balance_fetch_enforcement_safe` is set (a `cached_balance_fetches(
    enforcement_safe=True)` scope, opened only by a held `market_lock`); see
    that ContextVar's own comment for the id=1187 EGW00215 incident this
    fixes and why the reuse must stay conditional rather than unconditional."""
    from app.web import state  # local import — avoid circular import, see module-level note above

    cache = _snapshot_cache.get()
    if cache is not None and market in cache:
        return cache[market]

    broker = await state.kis_broker()

    if broker is None:
        raise AccountUnavailable(f"KIS broker not connected — cannot compute real {market} account snapshot")

    enforcement_safe = _balance_fetch_enforcement_safe.get()
    try:
        if market == "domestic":
            if enforcement_safe:
                balance = await _cached_fetch_balance(broker)
            else:
                _bump_ledger_inquiry_count()
                balance = await broker.fetch_balance()
            per_ticker = {f"{p.code}.KS": p.eval_amount for p in balance.positions}
            total = _domestic_equity_total(balance)
        else:
            # fetch_overseas_balance's own cash_deposit is always 0 (that TR
            # has no cash field) — fetch_overseas_present_balance is the one
            # that actually returns it, same two-call pattern already used
            # by app.web.state._kis_account_snapshot_build for the dashboard.
            if enforcement_safe:
                balance = await _cached_fetch_overseas_balance(broker)
                cash = await _cached_fetch_overseas_present_balance(broker, "USD")
            else:
                _bump_ledger_inquiry_count()
                balance = await broker.fetch_overseas_balance()
                _bump_ledger_inquiry_count()
                cash = await broker.fetch_overseas_present_balance("USD")
            per_ticker = {p.code: p.eval_amount for p in balance.positions}
            total = _overseas_equity_total(balance, cash)
    except AccountUnavailable:
        raise
    except Exception as exc:
        raise AccountUnavailable(f"KIS {market} balance inquiry failed: {exc}") from exc

    snapshot = PortfolioSnapshot(per_ticker_value=per_ticker, total_value=total)
    if cache is not None:
        cache[market] = snapshot
    return snapshot


# Backstop for settings.overseas_trading_enabled=False (see that flag's own
# comment in app/config.py) — a one-way, process-lifetime latch that undoes
# the flag's background-polling suppression the instant the assumption
# behind it ("this account never holds anything overseas") is contradicted.
#
# Deliberately module-level, plain globals (not a DB row): this must survive
# for the rest of THIS process's life regardless of which contextvar scope
# or user/account_mode is active when it trips — the flag itself is a
# single Jenkins-injected, deployment-wide constant, not a per-user/request
# setting, so the backstop that can override it is scoped the same way.
# Matches this module's existing convention for process-lifetime, non-DB
# state (see `_correlation_cache`/`_read_snapshot_cache` above).
_overseas_flag_contradicted_flag = False  # one-way; never reset except by a fresh test/process
_overseas_flag_last_verified_date: date | None = None  # KST calendar date of the last probe run


def overseas_flag_contradicted() -> bool:
    """True once `verify_overseas_flat_if_disabled` has found (or failed to
    rule out) real overseas exposure while the flag claimed there was none.
    Every background-polling skip gated on `settings.overseas_trading_enabled`
    must OR this in — see app.config.overseas_trading_enabled's own comment
    and this module's `real_account_snapshot`'s docstring block for why the
    flag alone is never trusted blindly once contradicted."""
    return _overseas_flag_contradicted_flag


async def verify_overseas_flat_if_disabled() -> None:
    """Once per KST calendar day (and effectively on daemon start, since the
    module-level date starts unset), when `settings.overseas_trading_enabled`
    is False, ask the overseas ledger ONCE whether the assertion still holds.

    Any position, or any non-zero USD deposit, is a CRITICAL log + an
    `overseas_exposure_while_disabled` alert + a one-way latch
    (`overseas_flag_contradicted`) that re-enables every gated path for the
    rest of the process. A probe FAILURE also trips the latch — fail-safe:
    could not verify, so do not keep suppressing.

    No-op immediately if the flag is True (don't burn a TR the enforcement
    path already covers unconditionally). No-op if already run today. Never
    raises — any failure inside the probe itself is caught and treated as a
    contradiction (see above), and this function is additionally expected to
    be called from within the caller's own try/except (belt-and-suspenders
    against anything in the wiring, e.g. import errors, that isn't the probe
    itself)."""
    global _overseas_flag_last_verified_date

    if settings.overseas_trading_enabled:
        return

    today = local_today("overseas")
    if _overseas_flag_last_verified_date == today:
        return
    _overseas_flag_last_verified_date = today

    from app.live.alerts import Alert, record_alert
    from app.web import state  # local import — see module-level note above

    contradicted = False
    detail = ""
    try:
        broker = await state.kis_broker()
        if broker is None:
            contradicted = True
            detail = "KIS broker not connected — could not verify the overseas account is flat"
        else:
            balance = await broker.fetch_overseas_balance()
            cash = await broker.fetch_overseas_present_balance("USD")
            if balance.positions:
                contradicted = True
                detail = (
                    f"overseas_trading_enabled=False but {len(balance.positions)} overseas "
                    f"position(s) found: {[p.code for p in balance.positions]}"
                )
            elif cash.matched and cash.deposit > 0:
                contradicted = True
                detail = (
                    f"overseas_trading_enabled=False but a non-zero USD deposit ({cash.deposit}) "
                    "was found in the overseas account"
                )
    except Exception as exc:  # noqa: BLE001 - a probe failure is itself a contradiction (fail-safe)
        contradicted = True
        detail = f"overseas flat-check probe failed — cannot verify the assumption: {exc}"

    if not contradicted:
        return

    global _overseas_flag_contradicted_flag
    _overseas_flag_contradicted_flag = True
    logger.critical(
        "portfolio_risk: overseas_trading_enabled=False was CONTRADICTED — %s. "
        "Background overseas polling is now re-enabled for the rest of this process.",
        detail,
    )
    await record_alert(
        Alert(
            ticker="OVERSEAS",
            kind="overseas_exposure_while_disabled",
            message=(
                f"overseas_trading_enabled=False but the daily flat-check found: {detail}. "
                "Background overseas polling has been re-enabled for the rest of this process — "
                "check the deployment's account scope assumption."
            ),
            triggered_at=get_clock().now(),
        )
    )


# A short process-wide TTL cache in FRONT of `real_account_snapshot`, for
# READ-ONLY / narrative callers ONLY. The safety property is the function
# NAME: enforcement callers (apply_live_portfolio_caps, daily_loss_block)
# keep calling `real_account_snapshot` directly and are wholly unaffected —
# a stale snapshot cannot reach a sizing/block decision by accident, only by
# a caller explicitly asking for `cached_account_snapshot`. The 441-536
# comment block forbids reusing a balance fetched BEFORE a market_lock in an
# ENFORCEMENT judgement; every consumer of this cache is outside the lock
# and its result decides no order size and no block. Single-flight per key so
# N concurrent narrative callers (the 2026-09-01 EGW00215 driver was two
# analyses' _portfolio_risk_context racing) collapse to one fetch. A failure
# is never cached.
# 서술·표시용 전용 짧은 캐시 — 집행 경로(real_account_snapshot)는 그대로 신선.
_read_snapshot_cache: dict[tuple[str, str, "Market"], tuple[float, PortfolioSnapshot]] = {}
_read_snapshot_locks: dict[tuple[str, str, "Market"], asyncio.Lock] = {}


async def cached_account_snapshot(market: Market, *, max_age: float | None = None) -> PortfolioSnapshot:
    """READ-ONLY / narrative callers only (see the comment above). Delegates
    to `real_account_snapshot` on a miss, single-flight per
    (user, account_mode, market). `AccountUnavailable` propagates every call
    — never cached."""
    if max_age is None:
        max_age = settings.kis_account_read_cache_seconds
    key = (get_current_user_id(), current_account_mode(), market)
    hit = _read_snapshot_cache.get(key)
    if hit is not None and time.monotonic() - hit[0] < max_age:
        return hit[1]
    lock = _read_snapshot_locks.setdefault(key, asyncio.Lock())
    async with lock:
        hit = _read_snapshot_cache.get(key)
        if hit is not None and time.monotonic() - hit[0] < max_age:
            return hit[1]
        snapshot = await real_account_snapshot(market)  # raises AccountUnavailable — not cached
        _read_snapshot_cache[key] = (time.monotonic(), snapshot)
        return snapshot


def invalidate_account_read_cache() -> None:
    """Drop every entry — called after a fill so a stale narrative snapshot
    isn't served for the rest of the TTL (see app.web.routers.orders). Also
    drops the per-key single-flight locks: clearing mid-flight at worst
    costs one duplicate fetch (the cache dict, not the lock, is the
    coordination point), and it keeps tests that swap event loops from
    reusing a lock bound to a closed one."""
    _read_snapshot_cache.clear()
    _read_snapshot_locks.clear()


async def real_shares_held(ticker: str) -> int | None:
    """Actual KIS-held quantity for a single ticker, LIVE ONLY — the
    per-ticker counterpart to `real_account_snapshot`'s aggregate view, same
    "read the real broker balance, not the internal simulated ledger"
    correction. Returns `None` (not 0) when the real balance can't be read
    (broker not connected / KIS call failed) so a caller can distinguish
    "genuinely flat" from "unknown" and fall back to the internal ledger
    instead of quietly asserting a wrong flat position. 실제 KIS 보유수량 —
    조회 실패시 0이 아니라 None을 반환해 "진짜 0주"와 "조회 불가"를 구분.

    The `int` return is guaranteed by `BrokerBalance.held_quantity` (both
    domestic and overseas positions carry an int `quantity` now) — no
    casting is done here on purpose. If an upstream parser regresses and
    leaks a float again, the final guard in overseas/order.py's
    submit_overseas_order should fail loudly rather than this function
    quietly flooring it, which would hide the regression again (see the
    2026-08-19 ORD_QTY="111.0" incident)."""
    from app.web import state  # local import — avoid circular import, see module-level note above

    broker = await state.kis_broker()

    if broker is None:
        return None

    market = ticker_market(ticker)
    try:
        if market == "domestic":
            krx_code = krx_code_from_ticker(ticker)
            balance = await _cached_fetch_balance(broker)
            return balance.held_quantity(krx_code)
        else:
            balance = await _cached_fetch_overseas_balance(broker)
            return balance.held_quantity(ticker)
    except Exception:
        logger.warning("real_shares_held: KIS balance inquiry failed for %s", ticker, exc_info=True)
        return None


async def real_position_detail(ticker: str) -> BrokerPosition | None:
    """Real KIS-reported avg cost/unrealized P&L for a single ticker, LIVE
    ONLY — sibling to `real_shares_held`'s "real broker over internal
    ledger" correction, but for fields the internal ledger has no concept
    of at all (it tracks cash+shares, not a weighted-average cost basis).
    KIS already computes `avg_price`/`pnl_amount`/`pnl_pct` server-side (see
    `app.domain.models.account.BrokerPosition`) — this just reads them,
    same fields `real_account_snapshot` already fetches but discards down
    to `eval_amount` only. Returns `None` when the broker isn't connected,
    the ticker isn't currently held, or the KIS call fails — narrative-only,
    so degrading rather than raising is fine here, same convention as
    `real_shares_held`. 평단가/미실현손익 — KIS가 이미 계산해서 주는 값을 그대로 노출."""
    from app.web import state  # local import — see module-level note above

    broker = await state.kis_broker()

    if broker is None:
        return None

    market = ticker_market(ticker)
    try:
        if market == "domestic":
            krx_code = krx_code_from_ticker(ticker)
            # Narrative-only — safe to reuse a same-scope cache hit
            # regardless of `enforcement_safe` (unlike `real_account_
            # snapshot`, this never feeds an enforcement decision). Falls
            # through to a fresh fetch automatically when no
            # `cached_balance_fetches()` scope is open at all (see
            # `_cached_fetch_balance`'s own None-cache branch).
            balance = await _cached_fetch_balance(broker)
            return next((p for p in balance.positions if p.code == krx_code), None)
        else:
            balance = await _cached_fetch_overseas_balance(broker)
            return next((p for p in balance.positions if p.code == ticker), None)
    except Exception:
        logger.warning("real_position_detail: KIS balance inquiry failed for %s", ticker, exc_info=True)
        return None


class EquityLookup(NamedTuple):
    equity: float | None
    failure: Literal["broker_unavailable", "broker_error"] | None


async def real_equity_native_detailed(market: Market) -> EquityLookup:
    """`real_equity_native`'s full result, including WHY it failed when it
    does — added (2026-08-14) so a caller sizing a live trade off the
    fallback (`app.execution.ledger.resolve_equity_basis`) can surface a
    reason a human can act on, not just a silent `None`. `"broker_unavailable"`
    (no KIS broker connected — a config/connection-state fact, not a
    transient error) used to not even get a log line; `"broker_error"` is the
    KIS-call-raised case the plain warning below already covered. See memory:
    live-rebalance-equity-basis-mismatch-2026-08-14 for why this distinction
    started mattering — the equity this function feeds now directly gates
    whether a trade sizes off the real account or the internal ledger."""
    from app.web import state  # local import — avoid circular import, see module-level note above

    broker = await state.kis_broker()

    if broker is None:
        logger.warning(
            "real_equity_native: no KIS broker connected (%s) — caller will fall back to the internal ledger",
            market,
        )
        return EquityLookup(None, "broker_unavailable")

    try:
        if market == "domestic":
            balance = await _cached_fetch_balance(broker)
            return EquityLookup(_domestic_equity_total(balance), None)
        else:
            balance = await _cached_fetch_overseas_balance(broker)
            # 2026-09-08 (EGW00215 follow-up to FIX-1/`4c11b24`, id=1187):
            # this used to call the broker directly instead of the cached
            # sibling `_cached_fetch_overseas_present_balance` FIX-1 itself
            # added — the exact same domestic-only gap FIX-1 closed, just on
            # the overseas leg. Inside one rebalance_to_target market_lock
            # hold, resolve_equity_basis (which calls this) and
            # apply_live_portfolio_caps (real_account_snapshot, a few lines
            # below in this module) both read the present-balance TR — this
            # duplicated the wire call AND could read two different account
            # states if a fill landed between the two calls, so the equity
            # basis sizing used and the concentration cap's total_value could
            # silently disagree within the same hold. Outside a
            # cached_balance_fetches(enforcement_safe=True) scope (every
            # non-rebalance caller) this helper still fetches fresh — see its
            # own docstring.
            cash = await _cached_fetch_overseas_present_balance(broker, "USD")
            return EquityLookup(_overseas_equity_total(balance, cash), None)
    except Exception:
        logger.warning("real_equity_native: KIS %s balance inquiry failed", market, exc_info=True)
        return EquityLookup(None, "broker_error")


async def real_equity_native(market: Market) -> float | None:
    """Real aggregate KIS account equity for `market`, in that market's OWN
    currency (KRW for domestic, USD for overseas) — same native-currency
    convention `real_account_snapshot`'s `total_value` now also uses; this
    function exists separately since a caller sizing a trade in the
    ticker's native price wants a plain float, not a full snapshot. Returns
    `None` (not 0 or a
    fictional default) when the real balance can't be read, so a caller
    can fall back to whatever it was already doing instead of sizing off
    a wrong zero. A thin wrapper over `real_equity_native_detailed` (which
    also reports WHY, for callers that need to surface that) — kept so
    every existing caller/test using this plain-float signature is
    unaffected.

    Real incident this exists for: `rebalance_to_target`'s target-share
    math used to size off this internal paper ledger's own per-ticker
    cash — which, after enough simulated trades, had drifted to ~₩147M
    for one ticker alone while the REAL aggregate domestic account was
    worth ~$20K total. Every resulting target was wildly oversized and
    then slammed back down to near-zero by the (already real-balance-
    based) portfolio concentration cap — a self-defeating combination
    that blocked nearly every BUY. 실제 계좌 규모 기준으로 목표수량을 계산하도록
    맞춰서, 내부 장부의 부풀려진 가상현금 때문에 캡이 항상 되돌리는 문제를 해소."""
    return (await real_equity_native_detailed(market)).equity


_correlation_cache: dict[tuple[str, ...], dict[str, int]] = {}


async def compute_correlation_clusters(
    tickers: list[str],
    as_of: date | None = None,
    lookback_days: int | None = None,
) -> dict[str, int]:
    """Real pairwise return-correlation clusters, computed from actual price
    history (pandas already a dependency) — not a hand-assigned "these are
    both tech" guess. COMPLETE-linkage: a ticker only joins an existing
    bucket if it correlates above `portfolio_correlation_bucket_threshold`
    with EVERY current member, not just one. Single-linkage/greedy union
    risks chaining unrelated names together through a correlated
    intermediary (A-B and B-C both correlated, A-C not, still collapsing
    all three into one bucket) — complete-linkage avoids that.

    Cached by (tickers, as_of) for the lifetime of the process: this is the
    expensive part (one yfinance history fetch per ticker), and correlation
    structure over a multi-month window doesn't meaningfully change within
    a single trade date's worth of decisions. Mirrors
    app.orchestrator.pipeline's `_company_name_cache` pattern.
    """
    lookback_days = lookback_days or risk_rules.rule("portfolio_correlation_lookback_days")
    cache_key = (tuple(sorted(tickers)), as_of)
    if cache_key in _correlation_cache:
        return _correlation_cache[cache_key]

    end = as_of or get_clock().today()
    start = date.fromordinal(end.toordinal() - lookback_days)

    series: dict[str, pd.Series] = {}
    fetch_failed = False
    for ticker in tickers:
        try:
            prices = await get_price_series(ticker, start, end)
            if len(prices) < 5:
                continue
            s = pd.Series(prices).sort_index()
            series[ticker] = s.pct_change().dropna()
        except Exception:
            fetch_failed = True
            logger.warning("compute_correlation_clusters: could not fetch history for %s", ticker, exc_info=True)

    if len(series) < 2:
        clusters = {ticker: i for i, ticker in enumerate(tickers)}
        # Don't cache a fetch-failure fallback as if it were a real "no correlation" result —
        # a transient outage would otherwise stick as the answer for the rest of `as_of`'s day.
        if not fetch_failed:
            _correlation_cache[cache_key] = clusters
        return clusters

    frame = pd.concat(series, axis=1)
    corr = frame.corr()

    threshold = risk_rules.rule("portfolio_correlation_bucket_threshold")
    clusters: dict[str, int] = {}
    buckets: list[list[str]] = []
    for ticker in tickers:
        if ticker not in corr.columns:
            clusters[ticker] = len(buckets)
            buckets.append([ticker])
            continue
        placed = False
        for bucket_idx, bucket in enumerate(buckets):
            members_with_data = [m for m in bucket if m in corr.columns]
            comparisons = [
                corr.loc[ticker, m] >= threshold for m in members_with_data if not pd.isna(corr.loc[ticker, m])
            ]
            # all([]) is vacuously True — without the `comparisons` guard, a ticker with zero
            # non-NaN correlations against a bucket would be placed in it as if fully confirmed.
            if comparisons and all(comparisons):
                clusters[ticker] = bucket_idx
                bucket.append(ticker)
                placed = True
                break
        if not placed:
            clusters[ticker] = len(buckets)
            buckets.append([ticker])

    _correlation_cache[cache_key] = clusters
    return clusters


def _log_cap_reduction(
    ticker: str,
    cap_name: str,
    target_before: int,
    target_after: int,
    position_shares_before: int,
    price: float,
    snapshot: PortfolioSnapshot,
    other_total: float,
    *,
    cap_ratio: float | None = None,
    cap_limit_pct: float | None = None,
    concentration_frac: float | None = None,
    price_fx_rate: float = 1.0,
) -> None:
    """Diagnostic-only logging for every `apply_portfolio_caps` reduction —
    added 2026-08-14 after the 98.1%-Hold baseline backtest turned out to be
    caused by two portfolio-cap bugs (cash counted as position value; a
    KRW/USD unit mismatch) that were invisible because (a) the concentration/
    bucket/cash-floor caps only ever logged at INFO, which the backtest
    harness's WARNING-level root logger silently dropped, and (b) even at
    INFO the log line carried no diagnostic values (total_value,
    per_ticker_value, price, ...) to tell a real cap trip apart from a bug.
    This function is purely additive: it never changes `target_shares` or
    the returned `note` string (still byte-identical — see
    `app.backtest.portfolio_harness._tally_cap_triggers`'s substring match
    and `tests/memory/test_memory_store.py`'s assertions on that exact
    text), only what gets written to this module's own logger channel.

    Level: WARNING when the cap ate the ENTIRE intended increase
    (`target_after <= position_shares_before`, including a full wipe to
    zero) — that is the shape the baseline-backtest bug produced (22 shares
    -> 0). A merely-reduced-but-still-positive increase logs at INFO.

    Two extra WARNING lines fire independently of the level above, one per
    known bug signature, so a bug trip is visually distinguishable from a
    genuine risk-cap trip in the logs alone:
      (a) `position_shares_before == 0` (flat) but this ticker's own
          `snapshot.per_ticker_value` entry is already > 0 — the "cash
          counted as position value" bug (bug 1): a flat ticker should show
          0 portfolio value, not some nonzero figure that then dominates the
          concentration math.
      (b) the pre-cap projected value is > 20x the whole snapshot's
          total_value, or (first entry) the concentration fraction itself is
          > 99% — the "KRW price compared against a USD/mixed-currency
          snapshot" bug (bug 2): a single position that size relative to the
          whole book is a unit mismatch, not a real allocation decision.
    Both are near-silent in live trading (a real account with that little
    cash left, or a real >2000% single-position swing, would be extremely
    unusual) — if either turns out unexpectedly noisy in practice, that is
    itself worth reporting rather than silencing.

    `price_fx_rate` (2026-08-14, bug 2 fix): units of `price`'s currency per
    1 USD, same convention as `get_usd_fx_rate` — used ONLY to convert the
    bug-2 detector's internal `projected_value` comparison into the same
    unit as `snapshot.total_value` (USD for the backtest's cross-market
    snapshot). The `price=` value actually written to the log line stays the
    native/raw value passed in, unconverted — a human reading the log wants
    the real per-share price, not a normalized one.
    """
    market = ticker_market(ticker)
    per_ticker_value = snapshot.per_ticker_value.get(ticker)
    cash = snapshot.total_value - sum(snapshot.per_ticker_value.values())
    wiped = target_after <= position_shares_before
    level = logging.WARNING if wiped else logging.INFO
    logger.log(
        level,
        "%s (%s): %s cap reduced target %d -> %d [position_shares_before=%d price=%r "
        "snapshot.total_value=%r per_ticker_value=%r other_total=%r cash=%r%s]",
        ticker, market, cap_name, target_before, target_after, position_shares_before, price,
        snapshot.total_value, per_ticker_value, other_total, cash,
        f" cap_ratio={cap_ratio!r} cap_limit_pct={cap_limit_pct!r}" if cap_ratio is not None else "",
    )

    projected_value = target_before * (price / price_fx_rate)
    if position_shares_before == 0 and (per_ticker_value or 0) > 0:
        logger.warning(
            "%s (%s): %s cap — SUSPECTED BUG (cash counted as position value): position_shares_before=0 "
            "but per_ticker_value=%r > 0 [snapshot.total_value=%r other_total=%r]",
            ticker, market, cap_name, per_ticker_value, snapshot.total_value, other_total,
        )
    if (snapshot.total_value > 0 and projected_value > 20 * snapshot.total_value) or (
        position_shares_before == 0 and concentration_frac is not None and concentration_frac > 0.99
    ):
        logger.warning(
            "%s (%s): %s cap — SUSPECTED BUG (currency-unit mismatch): projected_value=%r vs "
            "snapshot.total_value=%r [price=%r%s]",
            ticker, market, cap_name, projected_value, snapshot.total_value, price,
            f" concentration_frac={concentration_frac!r}" if concentration_frac is not None else "",
        )


def apply_portfolio_caps(
    ticker: str,
    target_shares: int,
    price: float,
    position_shares_before: int,
    snapshot: PortfolioSnapshot,
    clusters: dict[str, int],
    current_drawdown_frac: float,
    cash_floor_pct: float | None = None,
    *,
    price_fx_rate: float = 1.0,
) -> tuple[int, str | None]:
    """Pure, no I/O — both live code and the backtest harness call this with
    their own precomputed snapshot/clusters/drawdown fraction, so it's
    trivially unit-testable and never pays for a redundant fetch. Only ever
    reduces `target_shares` below what the per-ticker 1% risk cap already
    decided, mirroring RebalanceResult.adjustment_note's style for the
    returned reason string. `price` is in the CALLER's native currency for
    `ticker` (native per-market currency for the live single-market path;
    for the backtest's cross-market trade_allowlist snapshot, still whatever
    native currency the caller's price series is in — see `price_fx_rate`
    below for how that gets reconciled against `snapshot`, which is always
    USD-normalized there) — see PortfolioSnapshot's docstring.

    `price_fx_rate` (2026-08-14, bug 2 fix — units of `price`'s currency per
    1 USD, same convention as `get_usd_fx_rate`; default 1.0 preserves the
    old behavior for every live caller, where `price` and `snapshot` are
    already in the same native currency and no conversion is needed): all
    VALUE comparisons/arithmetic against `snapshot` (which may be in a
    different currency than `price`, e.g. the backtest's USD-normalized
    cross-market snapshot against a KRW `price`) are computed against
    `unit_price = price / price_fx_rate` instead of raw `price`. Plain
    share-count arithmetic (numerator and denominator both already in the
    same currency) is left on raw `price` and does not need this. Without
    this, a KRW price compared directly against a USD-normalized
    `snapshot.total_value` looked ~1,400x too concentrated and wiped
    otherwise-valid targets to 0 (the second of the two 98.1%-Hold
    baseline-backtest bugs, 2026-08-14).

    `cash_floor_pct`, if given, overrides `risk_rules.rule("portfolio_cash_floor_pct")`
    for the cash-floor check below — `apply_live_portfolio_caps` passes the
    market-regime-aware value from `app.execution.market_regime` here for
    the LIVE domestic path; every other caller (backtest harness, ledger's
    `portfolio_cap_override`) leaves this None and gets the static default,
    since a historical replay must never read today's actual market regime.

    Returns (possibly-reduced target_shares,
    note)."""
    unit_price = price / price_fx_rate  # `price`'s currency normalized into `snapshot`'s currency
    if target_shares <= position_shares_before:
        return target_shares, None  # not increasing exposure — nothing to cap

    # Computed once up front (doesn't depend on target_shares) so every cap
    # below — including the two full-block caps that return early — can pass
    # it to `_log_cap_reduction` for diagnostics.
    other_total = snapshot.total_value - snapshot.per_ticker_value.get(ticker, 0.0)

    if current_drawdown_frac >= risk_rules.rule("portfolio_drawdown_breaker_pct"):
        note = (
            f"Portfolio drawdown breaker: aggregate equity is {current_drawdown_frac:.1%} below its "
            f"high-water mark (limit {risk_rules.rule('portfolio_drawdown_breaker_pct'):.1%}) — blocked the exposure "
            f"increase to {target_shares} shares; held at the existing {position_shares_before} instead."
        )
        _log_cap_reduction(
            ticker, "drawdown breaker", target_shares, position_shares_before, position_shares_before,
            price, snapshot, other_total, cap_ratio=current_drawdown_frac,
            cap_limit_pct=risk_rules.rule("portfolio_drawdown_breaker_pct"), price_fx_rate=price_fx_rate,
        )
        return position_shares_before, note

    # Max open positions — only gates a FIRST-TIME entry (position_shares_
    # before == 0); adding to an already-held ticker never counts against
    # this cap, since it doesn't increase the number of distinct names held.
    # `snapshot.per_ticker_value` already reflects every currently-held
    # ticker in this market (see PortfolioSnapshot's docstring), so a count
    # of its nonzero entries IS the current open-position count.
    if position_shares_before == 0:
        max_open = (
            risk_rules.rule("max_open_positions_domestic") if ticker_market(ticker) == "domestic"
            else risk_rules.rule("max_open_positions_overseas")
        )
        open_count = sum(1 for v in snapshot.per_ticker_value.values() if v > 0)
        if open_count >= max_open:
            note = (
                f"Max open positions cap: {open_count} positions already open (limit {max_open}) — "
                f"blocked new entry into {ticker}."
            )
            _log_cap_reduction(
                ticker, "max open positions", target_shares, position_shares_before, position_shares_before,
                price, snapshot, other_total, cap_ratio=float(open_count), cap_limit_pct=float(max_open),
                price_fx_rate=price_fx_rate,
            )
            return position_shares_before, note

    note: str | None = None

    projected_value = target_shares * unit_price  # normalized into `snapshot`'s currency
    projected_total = other_total + projected_value
    if projected_total <= 0:
        return target_shares, None

    concentration_frac = projected_value / projected_total
    if concentration_frac > risk_rules.rule("portfolio_concentration_max_pct"):
        max_value = risk_rules.rule("portfolio_concentration_max_pct") * other_total / (
            1 - risk_rules.rule("portfolio_concentration_max_pct")
        )
        capped_shares = (
            max(int(max_value / unit_price), position_shares_before) if unit_price > 0 else position_shares_before
        )
        if capped_shares < target_shares:
            note = (
                f"Portfolio concentration cap: {ticker} would reach {concentration_frac:.1%} of aggregate "
                f"equity (limit {risk_rules.rule('portfolio_concentration_max_pct'):.1%}) — reduced the add from "
                f"{target_shares} to {capped_shares} shares."
            )
            _log_cap_reduction(
                ticker, "concentration", target_shares, capped_shares, position_shares_before,
                price, snapshot, other_total, cap_ratio=concentration_frac,
                cap_limit_pct=risk_rules.rule("portfolio_concentration_max_pct"),
                concentration_frac=concentration_frac, price_fx_rate=price_fx_rate,
            )
            target_shares = capped_shares

    bucket = clusters.get(ticker)
    if bucket is not None:
        bucket_tickers = [t for t, b in clusters.items() if b == bucket]
        bucket_other_total = sum(
            v for t, v in snapshot.per_ticker_value.items() if t in bucket_tickers and t != ticker
        )
        bucket_projected_value = target_shares * unit_price
        bucket_projected_total = other_total + bucket_projected_value  # denominator is still the whole portfolio
        bucket_frac = (
            (bucket_other_total + bucket_projected_value) / bucket_projected_total
            if bucket_projected_total > 0
            else 0.0
        )
        if bucket_frac > risk_rules.rule("portfolio_correlation_bucket_max_pct"):
            # Solve (bucket_other_total + x) / (other_total + x) <= max_pct for x:
            # x <= (max_pct * other_total - bucket_other_total) / (1 - max_pct).
            # Clamped at 0: if the bucket is already over the cap without any
            # help from this ticker (bucket_other_total alone too large), no
            # increase at all is allowed, not a negative one.
            max_bucket_new_value = max(
                (risk_rules.rule("portfolio_correlation_bucket_max_pct") * other_total - bucket_other_total)
                / (1 - risk_rules.rule("portfolio_correlation_bucket_max_pct")),
                0.0,
            )
            capped_shares = (
                max(int(max_bucket_new_value / unit_price), position_shares_before)
                if unit_price > 0
                else position_shares_before
            )
            if capped_shares < target_shares:
                note = (
                    f"Portfolio correlation-bucket cap: {ticker}'s bucket (correlated names: "
                    f"{', '.join(t for t in bucket_tickers if t != ticker) or 'none other'}) would reach "
                    f"{bucket_frac:.1%} of aggregate equity (limit {risk_rules.rule('portfolio_correlation_bucket_max_pct'):.1%}) "
                    f"— reduced the add from {target_shares} to {capped_shares} shares."
                )
                _log_cap_reduction(
                    ticker, "correlation bucket", target_shares, capped_shares, position_shares_before,
                    price, snapshot, other_total, cap_ratio=bucket_frac,
                    cap_limit_pct=risk_rules.rule("portfolio_correlation_bucket_max_pct"),
                    price_fx_rate=price_fx_rate,
                )
                target_shares = capped_shares

    # Cash floor: computed from the snapshot's OWN pre-trade cash (total_value
    # minus every ticker's current value, this ticker included) rather than
    # reusing `other_total`/`projected_total` above — those two already
    # assume the earlier caps' approximation of "everything else held fixed"
    # for concentration purposes, which is a different question from "how
    # much of the real account is actually cash after this specific trade."
    if unit_price > 0:
        effective_cash_floor_pct = (
            cash_floor_pct if cash_floor_pct is not None else risk_rules.rule("portfolio_cash_floor_pct")
        )
        cash_before = snapshot.total_value - sum(snapshot.per_ticker_value.values())
        incremental_purchase = (target_shares - position_shares_before) * unit_price
        projected_cash = cash_before - incremental_purchase
        cash_floor_value = effective_cash_floor_pct * snapshot.total_value
        if snapshot.total_value > 0 and projected_cash < cash_floor_value:
            max_incremental = max(cash_before - cash_floor_value, 0.0)
            capped_shares = max(position_shares_before + int(max_incremental / unit_price), position_shares_before)
            if capped_shares < target_shares:
                projected_cash_frac = projected_cash / snapshot.total_value
                note = (
                    f"Portfolio cash floor: this buy would leave only {projected_cash_frac:.1%} of aggregate "
                    f"equity in cash (floor {effective_cash_floor_pct:.1%}) — reduced the add from "
                    f"{target_shares} to {capped_shares} shares."
                )
                _log_cap_reduction(
                    ticker, "cash floor", target_shares, capped_shares, position_shares_before,
                    price, snapshot, other_total, cap_ratio=projected_cash_frac,
                    cap_limit_pct=effective_cash_floor_pct, price_fx_rate=price_fx_rate,
                )
                target_shares = capped_shares

    return target_shares, note


async def apply_backtest_portfolio_caps(
    ticker: str,
    target_shares: int,
    price: float,
    position_shares_before: int,
    snapshot: PortfolioSnapshot,
    clusters: dict[str, int],
    current_drawdown_frac: float,
    cash_floor_pct: float | None = None,
) -> tuple[int, str | None]:
    """Thin, I/O-performing wrapper around `apply_portfolio_caps` for the
    backtest harness — symmetric counterpart to `apply_live_portfolio_caps`
    below, but for the historical-replay path instead of the real account.

    `price` here is the ticker's own NATIVE currency, exactly as the
    multi-ticker backtest harness already carries it end-to-end (yfinance
    close, no conversion) — see `portfolio_harness.py`. `snapshot`, however,
    is USD-normalized (`portfolio_snapshot`'s own convention, needed because
    the backtest's trade_allowlist can span both KRW and USD tickers at
    once — see that function's docstring). Resolves the `price_fx_rate`
    `apply_portfolio_caps` needs to reconcile the two by looking up
    `ticker`'s quote currency and its USD rate, then delegates.

    2026-08-14 bug 2 fix: before this wrapper existed, callers passed native
    `price` straight into `apply_portfolio_caps` with no conversion at all,
    so a KRW price got compared directly against the USD-normalized
    snapshot — a domestic 22-share target looked ~1,400x too concentrated
    and was wiped to 0 every time. See `apply_portfolio_caps`'s own
    `price_fx_rate` docstring for the arithmetic this fixes."""
    currency = await get_ticker_currency(ticker)
    fx_rate = await get_usd_fx_rate(currency)  # units of `currency` per 1 USD
    return apply_portfolio_caps(
        ticker, target_shares, price, position_shares_before, snapshot, clusters, current_drawdown_frac,
        cash_floor_pct, price_fx_rate=fx_rate,
    )


async def _get_high_water_mark(market: Market) -> float | None:
    try:
        pool = await _ready_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT high_water_mark FROM portfolio_drawdown_state "
                "WHERE market = $1 AND account_mode = $2 AND user_id = $3",
                market, current_account_mode(), get_current_user_id(),
            )
    except Exception as exc:  # noqa: BLE001 - (OSError, PostgresError) missed
        # asyncpg.InterfaceError (pool closing/contended — not a PostgresError
        # subclass) and any other read failure; degrade quietly either way.
        # This is a DB-availability failure, distinct from (and handled
        # differently than) real_account_snapshot's AccountUnavailable —
        # a DB outage here fail-opens (treated as "no HWM yet", 0.0
        # drawdown), matching this function's existing convention, while an
        # unreachable KIS balance is the caller's problem to fail-closed on.
        # PostgresError만으로는 못 잡는 InterfaceError 등도 조용히 무시하도록 확장.
        logger.warning("portfolio_risk get_high_water_mark(%s) failed: %s", market, exc)
        return None
    return row["high_water_mark"] if row else None


async def _update_high_water_mark(market: Market, current_total: float) -> None:
    try:
        pool = await _ready_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO portfolio_drawdown_state (market, account_mode, user_id, high_water_mark, updated_at)
                VALUES ($1, $2, $3, $4, now())
                ON CONFLICT (market, account_mode, user_id) DO UPDATE
                SET high_water_mark = GREATEST(portfolio_drawdown_state.high_water_mark, $4), updated_at = now()
                """,
                market, current_account_mode(), get_current_user_id(), current_total,
            )
    except Exception as exc:  # noqa: BLE001 - same broadened fail-open policy as get_high_water_mark above
        logger.warning("portfolio_risk update_high_water_mark(%s) failed: %s", market, exc)


async def reset_high_water_mark(market: Market) -> None:
    """Deletes the persisted HWM row for `market` under the CURRENT
    account_mode so the next current_drawdown_fraction call treats the
    current real account total as a fresh baseline (0.0 drawdown) — call
    right after manually resetting the paper account balance, or right
    after switching kis_account_mode for the first time, never
    automatically (a real crash and a manual reset are indistinguishable
    from the total alone, so this must stay an explicit, user-triggered
    action). Scoped to account_mode (2026-08-04) so resetting paper's HWM
    can never touch real's row or vice versa. 모의계좌 리셋/계좌모드 전환 직후 수동 실행 전용."""
    pool = await _ready_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            "DELETE FROM portfolio_drawdown_state "
            "WHERE market = $1 AND account_mode = $2 AND user_id = $3",
            market, current_account_mode(), get_current_user_id(),
        )
    logger.warning(
        "portfolio_risk: high-water mark reset for %s (%s, user_id=%s)",
        market, current_account_mode(), get_current_user_id(),
    )


async def current_drawdown_fraction(market: Market, total_value: float) -> float:
    """Fraction the given market's aggregate total (its own native currency
    — see PortfolioSnapshot's docstring) sits below ITS OWN persisted
    (live-only) running high-water mark — 0.0 if this total IS a new high
    (and updates the persisted mark to match). Domestic and overseas each
    have their own independent HWM row (and currency) — a US drawdown never
    trips the KRX breaker or vice versa. Public specifically so
    `app.orchestrator.pipeline`'s context-builder can describe the same
    drawdown state to the Portfolio Manager that `apply_live_portfolio_caps`
    is about to enforce, without either duplicating this logic or the
    caller reaching into this module's private HWM functions directly."""
    hwm = await _get_high_water_mark(market)
    if hwm is None or total_value > hwm:
        await _update_high_water_mark(market, total_value)
        return 0.0
    return (hwm - total_value) / hwm if hwm > 0 else 0.0


async def apply_live_portfolio_caps(
    ticker: str,
    target_shares: int,
    price: float,
    position_shares_before: int,
) -> tuple[int, str | None]:
    """Thin, I/O-performing wrapper around `apply_portfolio_caps` for live
    trading — fetches the REAL current account snapshot (domestic/overseas
    scoped to whichever market `ticker` belongs to, see `ticker_market`),
    correlation clusters, and the persisted (live-only, per-market) high-
    water mark, then delegates to the pure function.

    Fail-CLOSED specifically when the real account snapshot itself is
    unavailable (`AccountUnavailable` — broker not connected, or the KIS
    call failed): we can't verify the real account's actual exposure at
    all in that case, so any exposure-INCREASING trade is blocked rather
    than silently let through — same treatment as an active drawdown
    breaker trip. Holds/reductions still pass straight through. This is a
    deliberate departure from the old ledger-backed snapshot's fail-open
    convention (a monitoring-fetch hiccup used to just skip the cap) —
    now that the snapshot IS the real account, "can't read it" means "we
    don't know if this account can safely take more risk," which must
    fail closed, not open.
    실제 잔고 조회 실패 시 fail-closed(노출 증가 거래 차단) — 예전 내부장부 기준의
    fail-open 관행과 다름, 실제 계좌 상태를 모르면 안전하게 막아야 함.

    Correlation clustering failure is a lesser-severity monitoring fetch
    (only refines the bucket cap, doesn't gate exposure by itself) and
    stays fail-open as before."""
    market = ticker_market(ticker)
    market_trade_allowlist = settings.trade_allowlist_for_market(market)

    try:
        snapshot = await real_account_snapshot(market)
    except AccountUnavailable as exc:
        if target_shares > position_shares_before:
            note = (
                f"{PORTFOLIO_RISK_UNAVAILABLE_NOTE_PREFIX} ({exc}) — blocked the exposure increase to "
                f"{target_shares} shares; held at the existing {position_shares_before} instead."
            )
            logger.warning("%s: %s", ticker, note)
            return position_shares_before, note
        return target_shares, None

    try:
        clusters = await compute_correlation_clusters(market_trade_allowlist, as_of=get_clock().today())
    except Exception:
        logger.warning("apply_live_portfolio_caps: correlation clustering failed, using no clusters", exc_info=True)
        clusters = {}

    drawdown_frac = await current_drawdown_fraction(market, snapshot.total_value)

    # Market-regime-aware cash floor (app.execution.market_regime) — KOSPI
    # for domestic, SPY for overseas. Best-effort: market_regime itself
    # already degrades to the static Risk-On floor on any failure, so this
    # can't newly fail here.
    from app.execution import market_regime  # local import — see module-level note above

    cash_floor_pct = (
        await market_regime.domestic_cash_floor_pct()
        if market == "domestic"
        else await market_regime.overseas_cash_floor_pct()
    )

    # `price` and `snapshot` are both in the ticker's own native currency
    # here (real_account_snapshot no longer converts domestic to USD — see
    # its docstring: domestic/overseas are never combined in this live,
    # single-market path, so there's no cross-currency sum that would need
    # a common unit, and skipping the conversion removes the FX round-trip
    # entirely rather than relying on every caller remembering to redo it
    # (an earlier version of this function forgot to, which is exactly the
    # unit-mismatch bug this comment used to describe).
    return apply_portfolio_caps(
        ticker, target_shares, price, position_shares_before, snapshot, clusters, drawdown_frac,
        cash_floor_pct=cash_floor_pct,
    )
