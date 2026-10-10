"""KIS quotations as the primary quote source for KRX tickers, and (for the
price/technical/news categories only — KIS's overseas-stock API has no
fundamentals/analyst/ownership/estimates TRs at all, confirmed by scanning
its full API list) for plain US-ticker-shaped symbols too.

Re-registers price/technical/news categories with a KIS-first chain (KIS,
then yfinance) — `router.register` replaces a category's provider list, so
this module must be imported after `app.dataflows.yfinance` (see the
package __init__; the from-import below guarantees the ordering anyway).

Tickers matching neither market and unconfigured KIS credentials raise
immediately, which the VendorRouter treats as "try the next provider": a
checkout without keys behaves exactly as before this module existed.
Point-in-time discipline for backtests is preserved by capping the candle
range at `as_of` server-side.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import dataclasses
import logging
import re
from datetime import date, timedelta

import pandas as pd

from app.brokers import KISBroker, get_broker
from app.clock import get_clock
from app.config import settings
from app.dataflows.router import router
from app.live import krx_master, overseas_master
from app.dataflows.yfinance import get_atr_pct as _yf_atr_pct
from app.dataflows.yfinance import get_daily_high_series as _yf_daily_high_series
from app.dataflows.yfinance import get_daily_low_series as _yf_daily_low_series
from app.dataflows.yfinance import get_daily_open_series as _yf_daily_open_series
from app.dataflows.yfinance import get_last_close as _yf_last_close
from app.dataflows.yfinance import get_price_series as _yf_price_series
from app.dataflows.indicators import atr_pct as _compute_atr_pct
from app.dataflows.indicators import format_indicator_line
from app.dataflows.long_horizon import WeeklyBar, format_long_horizon_block
from app.dataflows.session_state import format_verified_snapshot, live_session_footnote, session_in_progress
from app.dataflows.yfinance import get_ticker_display_info as _yf_display_info
from app.dataflows.yfinance.market_data import TickerDisplayInfo
from app.dataflows.yfinance.market_data import get_usd_fx_rate
from app.domain.models.market import EarningsEstimate
from app.domain.models.market import IncomeStatement
from app.dataflows.yfinance.research import (
    _yf_analyst_consensus,
    _yf_balance_sheet,
    _yf_earnings_estimates,
    _yf_fundamentals,
    _yf_indicators,
    _yf_long_horizon,
    _yf_news,
    _yf_short_interest,
    _yf_stock_data,
    _yf_verified_snapshot,
)
from app.live.market_hours import is_market_open

logger = logging.getLogger(__name__)

# Real yfinance closes for KRX names were observed off by ~2% from the
# exchange-official figure (same discrepancy _kis_verified_snapshot's
# docstring notes) — get_last_close/get_atr_pct/get_price_series/
# get_daily_low_series below go KIS-first for .KS tickers (and, via the
# overseas candle TR, US-ticker-shaped symbols too) for the same reason,
# falling back to yfinance on any failure or for unsupported tickers.
_MAX_KIS_CANDLE_RANGE_DAYS = 100  # stay under fetch_daily_candles's ~100-row cap; wider ranges use yfinance

async def _get_broker() -> KISBroker:
    # Looked up via the brokers registry (not a direct KISBroker import) so
    # this research/LLM-tool path and the execution path (kis_bridge.py)
    # both resolve "the KIS broker" the same way.
    # 레지스트리 경유로 조회 — 실행 경로(kis_bridge.py)와 동일한 방식으로 브로커 획득
    #
    # Async, and holding no cache of its own, as of the multi-user migration
    # Phase 2. It used to memoize the resolved broker in module globals
    # (`_broker` / `_broker_checked`), which was a SECOND process-wide cache
    # layered on top of app.brokers' own — harmless when one process meant one
    # account, and a cross-user data leak the moment it didn't: whoever
    # triggered the first KIS-backed research call would have had their broker
    # pinned for every other user in the process, for the process lifetime.
    # app.brokers.get_broker already caches, keyed by (name, user, mode), so
    # there is nothing lost by deleting the local one.
    # 이전의 모듈 전역 캐시를 제거 — 첫 호출자의 브로커가 프로세스 전체에 고정돼
    # 다른 사용자에게도 쓰이는 문제가 있었음(레지스트리가 이미 사용자별로 캐싱함).
    broker = await get_broker("kis")
    if broker is None:
        raise RuntimeError("KIS credentials not configured")
    return broker


# Bounds get_kr_financials's fan-out (7 statement/ratio TRs + valuation) so it
# doesn't burst the client throttle — the Fundamentals Analyst runs that call
# concurrently with get_kr_analyst_opinion, so the raw fan-out is ~9 KIS calls
# at once and it tripped EGW00201 (2026-09-03 decision id=1179). Rebuilt on the
# running loop rather than bound at import time: the CLI, daemon and test
# suites each run on a different loop, and an asyncio.Semaphore bound to a dead
# loop raises the moment it's awaited.
# 재무 팬아웃 동시성 게이트 — 실행 루프가 바뀌면 재생성(import 시점 바인딩 회피).
_kr_financials_gate: asyncio.Semaphore | None = None
_kr_financials_gate_loop: asyncio.AbstractEventLoop | None = None


def _get_kr_financials_gate() -> asyncio.Semaphore:
    global _kr_financials_gate, _kr_financials_gate_loop
    loop = asyncio.get_running_loop()
    if _kr_financials_gate is None or _kr_financials_gate_loop is not loop:
        _kr_financials_gate = asyncio.Semaphore(settings.kis_financials_fanout_concurrency)
        _kr_financials_gate_loop = loop
    return _kr_financials_gate


def _krx_code(ticker: str) -> str:
    if ticker.endswith(".KS") and ticker[:-3].isdigit():
        return ticker[:-3]
    raise ValueError(f"{ticker} is not a KRX ticker")


def _is_krx(ticker: str) -> bool:
    """Non-raising cousin of `_krx_code`, for callers that want to fall
    through to a yfinance path for non-KRX tickers instead of handling an
    exception."""
    return ticker.endswith(".KS") and ticker[:-3].isdigit()


# Same loose 1-5 uppercase letters shape app.execution.kis_bridge._US_SYMBOL
# uses for order routing — duplicated rather than imported to avoid a
# dataflows -> execution dependency (execution already imports this module
# the other way, at both module load time in ledger.py/portfolio_risk.py
# and via a deliberately-local import in kis_bridge.py to dodge the cycle).
_US_TICKER = re.compile(r"^[A-Z]{1,5}$")


def _is_us(ticker: str) -> bool:
    return bool(_US_TICKER.match(ticker))


def _candle_date(day: str) -> date:
    return date(int(day[:4]), int(day[4:6]), int(day[6:]))


# Per-analysis-run candle cache. One `run_pipeline()` call fans out to
# several independent callers that all want "recent daily candles for this
# ticker" at the same `end` (Market Analyst's get_stock_data/get_indicators/
# get_verified_snapshot tools, plus the pipeline's own get_atr_pct/
# get_last_close) — each asking for a different [start, end] sub-window of
# the same underlying series, with no shared cache, so a single ticker
# analysis fired off up to 5 separate KIS requests for overlapping data.
# `None` (the default, outside `candle_run_cache()`) keeps every caller's
# behavior byte-for-byte unchanged — the fetch goes out exactly as before,
# scoped to its own [start, end] with no widening. Only code that opts in
# by wrapping itself in `candle_run_cache()` (currently just
# `app.orchestrator.pipeline.run_pipeline`) gets the sharing.
# 분석 1회 실행 범위로만 캔들 캐시 — 여러 애널리스트/파이프라인이 같은 종목의
# 캔들을 각자 다시 가져오던 걸 한 번만 가져오도록(순수 중복 제거, 로직 불변).
_candle_run_cache: contextvars.ContextVar[dict | None] = contextvars.ContextVar("_candle_run_cache", default=None)


@contextlib.contextmanager
def candle_run_cache():
    """Scopes `_fetch_kis_candles` caching/coalescing to the wrapped block —
    wrap one `run_pipeline()` call in this, not a whole process lifetime, so
    two different analysis runs (possibly hours apart, possibly a live vs a
    backtest date) never share stale candle data.

    On exit, cancels any fetch still in flight (2026-08-25). Needed since
    `_cached_fetch` below now shields the shared task from its waiters'
    cancellations: a task nobody is waiting on any more would otherwise
    outlive the dict that referenced it and surface as "Task was destroyed
    but it is pending" at shutdown. Cancelling an already-finished task is a
    no-op, which is the normal case."""
    cache: dict = {}
    token = _candle_run_cache.set(cache)
    try:
        yield
    finally:
        _candle_run_cache.reset(token)
        for task in cache.values():
            task.cancel()


def _evict_if_unusable(finished: asyncio.Task, *, cache: dict, key: tuple) -> None:
    """Drops a cache entry whose shared fetch ended CANCELLED or FAILED, so
    the next caller in the same run scope starts a fresh one instead of
    re-awaiting a permanently-dead task. `cancelled()` is checked first
    because `.exception()` re-raises on a cancelled task — and calling
    `.exception()` at all is deliberate: it marks a failed task's exception
    retrieved, which is what keeps a coalesced-but-unawaited failure from
    logging "exception was never retrieved" at GC time. The identity check
    (`cache.get(key) is finished`) makes this safe if a replacement fetch
    for the same key was already stored."""
    if finished.cancelled() or finished.exception() is not None:
        if cache.get(key) is finished:
            del cache[key]


async def _cached_fetch(cache: dict, key: tuple, fetch_fn) -> list:
    """Coalesces concurrent misses for the same key onto one in-flight
    fetch — an `asyncio.Task` can be awaited more than once, so a second
    caller racing the first (e.g. two of Market Analyst's tool calls) just
    awaits the already-running task instead of firing its own request. Same
    coalescing idea as `app.web.state`'s `_display_info_locks`. Safe without
    an explicit lock: nothing awaits between the `cache.get` check and the
    `cache[key] =` write below, and asyncio only switches tasks at an
    `await`, so no other coroutine can observe the gap.

    `asyncio.shield`, not a bare `await` (fixed 2026-08-25): a bare await
    hands the AWAITING caller's cancellation straight to the SHARED task,
    which then sits in this dict permanently cancelled — and every later
    caller in the same `candle_run_cache()` scope re-raises that
    `CancelledError` from a `.get()` that never asked for anything. Because
    `CancelledError` subclasses BaseException, it also walks straight past
    the `except Exception:` yfinance fallbacks in `get_last_close`/
    `get_atr_pct`/`get_price_series` below, and past
    `app.orchestrator.pipeline`'s post-debate gather retry (which re-runs
    inside the SAME cache scope and so hits the same dead task) — turning
    one unrelated cancellation into a discarded decision after ~144k tokens
    of LLM work (see id=377/id=949 in that module's docstrings). Shielding
    keeps the fetch alive for the other callers and confines the
    cancellation to the caller that was actually cancelled; the done-
    callback above then evicts the entry if the fetch ITSELF ends up
    cancelled or failed, so a transient KIS error costs one retry instead of
    poisoning the whole run.
    한 호출자의 취소가 공유 캔들 fetch를 죽여 그 실행 전체를 오염시키던 문제 수정."""
    task = cache.get(key)
    if task is None:
        task = asyncio.ensure_future(fetch_fn())
        cache[key] = task
        task.add_done_callback(lambda t: _evict_if_unusable(t, cache=cache, key=key))
    return await asyncio.shield(task)


async def _fetch_kis_candles(ticker: str, start: date, end: date) -> list:
    """Dispatches to the domestic or overseas KIS candle TR depending on
    the ticker's market; raises ValueError for neither. The overseas TR
    (fetch_overseas_candles) has no `start` param of its own — it returns
    up to ~100 rows ending at `end` — so the [start, end] window is
    enforced client-side here, same effective contract as the domestic
    range-cap callers already rely on. Ticker-shape is checked before the
    broker lookup (not after), so an unsupported ticker never touches KIS
    credentials at all — same fast-fail invariant `_krx_code` gave callers
    before this dispatcher existed.

    When `candle_run_cache()` is active, always fetches the widest window
    any caller in this module uses (`_MAX_KIS_CANDLE_RANGE_DAYS`) and caches
    it per (ticker, end) — cheap, since that's already the row cap this
    module stays under — then filters down to this call's own [start, end]
    client-side, so every caller still sees exactly the window it asked
    for. Outside the cache scope, behavior is unchanged from before this
    cache existed: each call fetches exactly its own [start, end]."""
    cache = _candle_run_cache.get()
    if _is_krx(ticker):
        broker = await _get_broker()
        if cache is None:
            return await broker.fetch_daily_candles(ticker[:-3], start.strftime("%Y%m%d"), end.strftime("%Y%m%d"))
        key = ("KS", ticker, end)
        wide_start = end - timedelta(days=_MAX_KIS_CANDLE_RANGE_DAYS)
        full = await _cached_fetch(
            cache, key,
            lambda: broker.fetch_daily_candles(ticker[:-3], wide_start.strftime("%Y%m%d"), end.strftime("%Y%m%d")),
        )
        return [c for c in full if start <= _candle_date(c.day) <= end]
    if _is_us(ticker):
        broker = await _get_broker()
        if cache is None:
            candles = await broker.fetch_overseas_candles(ticker, end=end.strftime("%Y%m%d"))
        else:
            key = ("US", ticker, end)
            candles = await _cached_fetch(cache, key, lambda: broker.fetch_overseas_candles(ticker, end=end.strftime("%Y%m%d")))
        return [c for c in candles if start <= _candle_date(c.day) <= end]
    raise ValueError(f"{ticker} is not a KRX or US ticker")


async def _kis_verified_snapshot(ticker: str, as_of: date | None = None) -> str:
    """Same overall shape as `_yf_verified_snapshot` (both build through
    `format_verified_snapshot`), but from the exchange-official KIS daily bar
    — KRX closes from yfinance were observed off by ~2% during integration,
    which is exactly the class of error a "verified" snapshot exists to rule
    out. Also covers US-ticker-shaped symbols via the overseas candle TR —
    see module docstring.

    The two providers' text is no longer byte-identical end-to-end: each
    declares its own `source=` to `format_verified_snapshot`, which appends a
    trailing `source: KIS` / `source: yfinance` line (2026-09-10 dataflows
    audit, stage 2/3) so a transcript audit — or the model itself, for a
    fallback-served KRX ticker — can tell which vendor actually answered
    instead of the two being indistinguishable.

    This is the vendor router's FIRST-choice provider for `verified_snapshot`
    (registered ahead of `_yf_verified_snapshot`, which only runs as a
    fallback on KIS failure — see `router.register` below), so the
    still-in-progress-session mislabeling this fixes (see
    app.dataflows.session_state's module docstring for the MRVL incident)
    had to be fixed HERE, not just on the yfinance side — the yfinance fix
    alone would never even run for the common case. `format_verified_snapshot`
    is the single place both providers build this exact string, so the two
    can no longer drift apart on this."""
    end = as_of or get_clock().today()
    candles = await _fetch_kis_candles(ticker, end - timedelta(days=14), end)
    if not candles:
        raise RuntimeError(f"KIS returned no candles for {ticker}")
    last = candles[-1]
    return format_verified_snapshot(
        ticker, _candle_date(last.day),
        close=last.close, high=last.high, low=last.low, volume=last.volume,
        source="KIS",
    )


async def _kis_stock_data(ticker: str, as_of: date | None = None) -> str:
    """Last 10 sessions of OHLCV, mirroring `_yf_stock_data`'s framing. Only
    ~6 weeks are actually requested — the tail is all that's rendered, and
    the KIS endpoint caps one call at 100 rows anyway. Also covers
    US-ticker-shaped symbols via the overseas candle TR — see module
    docstring.

    2026-09-16: used to accept a `lookback_days` param and quote it in the
    returned label ("... of 180d lookback") even though only a 45-day window
    was ever requested and only the last 10 rows were ever rendered — a
    second, KIS-side instance of the dead/dishonest parameter `_yf_stock_data`
    had (see that function's own note). Dropped rather than honestly
    reduced to 45/10: nothing here ever varied it, and the caller that wants
    real multi-week context now has `get_long_horizon_context` for that
    instead of a wider version of this 10-row table (see `long_horizon.py`'s
    module docstring for why the table itself, not just the label, stays
    fixed at 10 sessions)."""
    end = as_of or get_clock().today()
    start = end - timedelta(days=45)
    candles = await _fetch_kis_candles(ticker, start, end)
    if not candles:
        raise RuntimeError(f"KIS returned no candles for {ticker}")
    tail = candles[-10:]
    header = f"{'Date':<12}{'Open':>10}{'High':>10}{'Low':>10}{'Close':>10}{'Volume':>14}"
    lines = [
        f"{c.day[:4]}-{c.day[4:6]}-{c.day[6:]:<4}"
        f"{c.open:>10.0f}{c.high:>10.0f}{c.low:>10.0f}{c.close:>10.0f}{c.volume:>14,}"
        for c in tail
    ]
    # live_session_footnote is a no-op (empty string) once as_of is set or
    # the session has closed — see its own docstring — so this stays
    # byte-identical to before for every already-passing caller/test.
    return (
        f"Daily OHLCV for {ticker} (last {len(tail)} sessions, source: KIS):\n"
        + "\n".join([header, *lines])
        + live_session_footnote(ticker, _candle_date(tail[-1].day))
    )


async def _fetch_kis_weekly_candles(ticker: str, end: date, weeks: int = 53) -> list:
    """Weekly OHLCV via the same KIS candle TRs `_fetch_kis_candles` uses,
    `period="W"` instead of the daily default — feeds `get_long_horizon_
    context`'s 52-week high/low and trailing-return figures (see
    `long_horizon.py`). A separate function rather than a `period` param on
    `_fetch_kis_candles`: that function's `candle_run_cache` keys
    (`("KS", ticker, end)` / `("US", ticker, end)`) are daily-only, and
    reusing them for weekly bars would silently mix the two granularities in
    one cache entry."""
    cache = _candle_run_cache.get()
    wide_start = end - timedelta(days=weeks * 7 + 14)
    if _is_krx(ticker):
        broker = await _get_broker()
        fetch = lambda: broker.fetch_daily_candles(  # noqa: E731
            ticker[:-3], wide_start.strftime("%Y%m%d"), end.strftime("%Y%m%d"), period="W"
        )
        if cache is None:
            candles = await fetch()
        else:
            candles = await _cached_fetch(cache, ("KS-W", ticker, end), fetch)
        return candles[-weeks:]
    if _is_us(ticker):
        broker = await _get_broker()
        fetch = lambda: broker.fetch_overseas_candles(ticker, end=end.strftime("%Y%m%d"), period="W")  # noqa: E731
        if cache is None:
            candles = await fetch()
        else:
            candles = await _cached_fetch(cache, ("US-W", ticker, end), fetch)
        return candles[-weeks:]
    raise ValueError(f"{ticker} is not a KRX or US ticker")


async def _kis_long_horizon(ticker: str, as_of: date | None = None) -> str:
    """52-week price context from KIS weekly bars — see `long_horizon.py`'s
    module docstring for why this exists (Market Analyst's daily table only
    ever covers the last 10 sessions, but analysts are still expected to
    speak to a 52-week range or a multi-month trend)."""
    end = as_of or get_clock().today()
    candles = await _fetch_kis_weekly_candles(ticker, end)
    if not candles:
        raise RuntimeError(f"KIS returned no weekly candles for {ticker}")
    bars = [WeeklyBar(_candle_date(c.day), c.high, c.low, c.close) for c in candles]
    return format_long_horizon_block(ticker, bars, source="KIS")


async def get_last_close(ticker: str, as_of: date | None = None) -> float:
    """KIS-first last close for KRX or US tickers, yfinance fallback
    otherwise or on any KIS failure. Same contract as yfinance's
    get_last_close (used by the execution layer for trade fills, and
    position sizing)."""
    if _is_krx(ticker) or _is_us(ticker):
        try:
            end = as_of or get_clock().today()
            candles = await _fetch_kis_candles(ticker, end - timedelta(days=14), end)
            if candles:
                return candles[-1].close
        except Exception:
            logger.warning("KIS get_last_close failed for %s, falling back to yfinance", ticker, exc_info=True)
    return await _yf_last_close(ticker, as_of=as_of)


async def get_atr_pct(ticker: str, as_of: date | None = None) -> float | None:
    """KIS-first 14-day ATR%% for KRX or US tickers, yfinance fallback
    otherwise or on any KIS failure. Used for volatility-adjusted position
    sizing. `None` (never NaN) if there isn't enough history yet — see
    `app.dataflows.indicators.atr_pct`'s own docstring.

    Calls `indicators.atr_pct` directly rather than going through
    `app.dataflows.yfinance.market_data` (2026-08-21) — this is a KRX-
    first module, so depending on a yfinance-specific module for shared
    math pointed the dependency the wrong way (same fix already applied
    to RSI/SMA/MACD; see indicators.py's own module docstring)."""
    if _is_krx(ticker) or _is_us(ticker):
        try:
            end = as_of or get_clock().today()
            # _MAX_KIS_CANDLE_RANGE_DAYS (not a narrower 45-day window, pre-
            # 2026-08-21): matches the widest window any other caller in this
            # module uses, so under candle_run_cache this reuses that same
            # cached fetch, and outside it costs the same one call regardless
            # of window width. Widened specifically for the switch to
            # talib.ATR's Wilder smoothing below — unlike this module's old
            # SMA, Wilder keeps drifting toward its converged value as more
            # history feeds it (measured: ~7% off true at 16 rows, ~0.2% by 60
            # rows), so a too-narrow window would make this KIS path
            # (previously ~31 rows) and the yfinance fallback path (previously
            # ~21 rows) disagree by several percent on the same ticker on the
            # same day.
            candles = await _fetch_kis_candles(ticker, end - timedelta(days=_MAX_KIS_CANDLE_RANGE_DAYS), end)
            # 2026-09-11 (live-ATR-understatement fix): during a live session
            # the last candle KIS returns is TODAY'S STILL-FORMING bar — its
            # high/low cover only the part of the day traded so far, so its
            # True Range is far too small and Wilder's ATR gets pulled down
            # with it. Measured on real KRX data (6 tickers, 2496
            # ticker-days, evaluated at the open): ATR% came out a median
            # 5.5% LOW versus the correct value. Both downstream consumers
            # move the wrong way TOGETHER from that: sizing.py divides
            # position size by ATR%, so the position gets BIGGER, and
            # ledger.resolve_stop_loss multiplies its floor by ATR%, so the
            # stop gets TIGHTER — a bigger position with a tighter stop, at
            # the same time, from the same understated number.
            #
            # Fix: compute the ATR from COMPLETED bars only (drop the
            # in-progress candle before it ever reaches talib.ATR), but keep
            # the LIVE trade price as the denominator rather than falling
            # back to the previous close. The previous-close alternative was
            # measured worse specifically on gap days (stale denominator):
            # median +5.5%, worst +28.4%, on overnight gaps above 5% — so
            # the live price stays the divisor even though the ATR numerator
            # itself no longer includes today.
            if candles:
                live_price = None
                if session_in_progress(ticker, _candle_date(candles[-1].day)):
                    live_price = candles[-1].close
                    candles = candles[:-1]
                if len(candles) >= 15:  # need window(14)+1 rows for a real ATR at the last row
                    high = pd.Series([c.high for c in candles])
                    low = pd.Series([c.low for c in candles])
                    close = pd.Series([c.close for c in candles])
                    if live_price is None:
                        return _compute_atr_pct(high, low, close, window=14)
                    return _compute_atr_pct(high, low, close, window=14, denominator_close=live_price)
        except Exception:
            logger.warning("KIS get_atr_pct failed for %s, falling back to yfinance", ticker, exc_info=True)
    return await _yf_atr_pct(ticker, as_of=as_of)


async def get_price_series(ticker: str, start: date, end: date) -> dict[date, float]:
    """KIS-first real close prices per trading day in [start, end] for KRX
    or US tickers, yfinance fallback for unsupported tickers, KIS failures,
    or ranges wide enough to risk the ~100-row candle cap silently
    truncating the series (a real correctness risk for a long backtest
    window that yfinance's own history() doesn't share)."""
    if (_is_krx(ticker) or _is_us(ticker)) and (end - start).days <= _MAX_KIS_CANDLE_RANGE_DAYS:
        try:
            candles = await _fetch_kis_candles(ticker, start, end)
            if candles:
                return {_candle_date(c.day): c.close for c in candles}
        except Exception:
            logger.warning("KIS get_price_series failed for %s, falling back to yfinance", ticker, exc_info=True)
    return await _yf_price_series(ticker, start, end)


async def get_long_range_price_series(ticker: str, start: date, end: date) -> dict[date, float]:
    """KIS closes over an ARBITRARILY wide [start, end] range, by walking it
    in <=_MAX_KIS_CANDLE_RANGE_DAYS chunks — unlike `get_price_series`, which
    falls back to yfinance whole-range the moment the window exceeds that
    cap. For a backtest replay, `get_price_series`'s cap means a 90-day run
    scores on KIS closes and a 120-day run scores on yfinance closes — the
    two differ by ~2%% on KRX names (see this module's other docstrings), so
    that split would make `backtest_runs` rows silently non-comparable
    across window lengths, the exact thing `record_experiment` exists to
    compare.

    ALL-OR-NOTHING: if any chunk fails or returns no candles, the WHOLE range
    falls back to yfinance rather than splicing a KIS segment onto a
    yfinance segment for the same ticker — a seam would fabricate a phantom
    return on the boundary date from the ~2%% source discrepancy."""
    if not (_is_krx(ticker) or _is_us(ticker)):
        return await _yf_price_series(ticker, start, end)
    merged: dict[date, float] = {}
    cursor = start
    try:
        while cursor <= end:
            chunk_end = min(cursor + timedelta(days=_MAX_KIS_CANDLE_RANGE_DAYS), end)
            candles = await _fetch_kis_candles(ticker, cursor, chunk_end)
            if not candles:
                raise RuntimeError(f"no KIS candles for {ticker} {cursor}..{chunk_end}")
            merged.update({_candle_date(c.day): c.close for c in candles})
            cursor = chunk_end + timedelta(days=1)
    except Exception:
        logger.warning(
            "KIS get_long_range_price_series failed for %s, falling back to yfinance for the whole range",
            ticker, exc_info=True,
        )
        return await _yf_price_series(ticker, start, end)
    return merged


async def get_daily_low_series(ticker: str, start: date, end: date) -> dict[date, float]:
    """KIS-first real daily lows per trading day in [start, end] for KRX or
    US tickers — same range-cap fallback reasoning as get_price_series."""
    if (_is_krx(ticker) or _is_us(ticker)) and (end - start).days <= _MAX_KIS_CANDLE_RANGE_DAYS:
        try:
            candles = await _fetch_kis_candles(ticker, start, end)
            if candles:
                return {_candle_date(c.day): c.low for c in candles}
        except Exception:
            logger.warning("KIS get_daily_low_series failed for %s, falling back to yfinance", ticker, exc_info=True)
    return await _yf_daily_low_series(ticker, start, end)


async def get_daily_high_series(ticker: str, start: date, end: date) -> dict[date, float]:
    """KIS-first real daily highs per trading day in [start, end] for KRX or
    US tickers — mirror of `get_daily_low_series`, for replaying a trailing
    stop's high-water mark through backtest history (2026-08-10)."""
    if (_is_krx(ticker) or _is_us(ticker)) and (end - start).days <= _MAX_KIS_CANDLE_RANGE_DAYS:
        try:
            candles = await _fetch_kis_candles(ticker, start, end)
            if candles:
                return {_candle_date(c.day): c.high for c in candles}
        except Exception:
            logger.warning("KIS get_daily_high_series failed for %s, falling back to yfinance", ticker, exc_info=True)
    return await _yf_daily_high_series(ticker, start, end)


async def get_daily_open_series(ticker: str, start: date, end: date) -> dict[date, float]:
    """KIS-first real daily opens per trading day in [start, end] for KRX or
    US tickers — used to determine the realizable fill for a stop order
    when the day's open already gaps through the stop level."""
    if (_is_krx(ticker) or _is_us(ticker)) and (end - start).days <= _MAX_KIS_CANDLE_RANGE_DAYS:
        try:
            candles = await _fetch_kis_candles(ticker, start, end)
            if candles:
                return {_candle_date(c.day): c.open for c in candles}
        except Exception:
            logger.warning("KIS get_daily_open_series failed for %s, falling back to yfinance", ticker, exc_info=True)
    return await _yf_daily_open_series(ticker, start, end)


# CTPF1002R's scty_grp_id_cd (증권그룹ID코드) -> this project's quote_type
# vocabulary ({"EQUITY","ETF"}, see app.domain.leverage_filter). ST(주권) and
# EF/FE(국내/해외ETF) are the only codes this project trades; anything else
# (MF=뮤추얼펀드, RT=리츠, EW=ELW, SC/IF=특수목적법인 등) is passed through
# unmapped so is_unsupported_product_type still blocks it — this table only
# needs to whitelist the supported codes, not enumerate every unsupported one.
_KRX_SECURITY_GROUP_TO_QUOTE_TYPE = {"ST": "EQUITY", "EF": "ETF", "FE": "ETF"}


async def get_ticker_display_info(ticker: str) -> TickerDisplayInfo:
    """Company name for a KRX ticker comes from `app.live.krx_master`'s
    KRX-published code->name snapshot first, and for a NASDAQ-shaped US
    ticker from `app.live.overseas_master`'s sibling snapshot (NASDAQ
    only — this project's whole overseas scope, see that module's
    docstring) — both resolved *before* and independently of the
    yfinance call below, so a yfinance failure can no longer take the
    name down with it (see the try/except around `_yf_display_info`).
    Neither master file needs a paper/real account distinction, unlike
    the KIS TRs below, and neither can suffer yfinance's cross-symbol
    confusion (a real but wrong company for an exact code: 064260.KS
    came back "에코프로비엠" instead of 다날, and 028050.KS got argued over
    as Hyundai Motor instead of 삼성E&A mid-debate during the 2026-07-28
    FULL_AUTO test — see krx_master's docstring). yfinance's longName/
    shortName is used only when the relevant master file has no entry
    for that symbol (a same-day new listing, an NYSE/AMEX-listed US
    ticker outside this project's NASDAQ-only master-file coverage, or
    the download itself failed) or the ticker matches neither shape.

    If the yfinance call itself raises (network/rate-limit failure, not
    just a bad name) for a KRX or US-shaped ticker, this no longer lets
    the exception propagate and wipe out an already-resolved KRX-master
    name along with it (the 2026-07-29 gap this replaces: a dashboard
    row showing the bare ticker code instead of any name at all, even
    though the master file had the right answer, because the whole
    function raised before ever reaching it). Instead it falls back to
    a placeholder TickerDisplayInfo (`previous_close=0.0`, `quote_type=
    ""`) carrying whatever name is available (master file, else the bare
    ticker). Confirmed safe to do (2026-07-29 code audit): no caller
    uses `previous_close`/`currency`/`quote_type` for order pricing,
    P&L, sizing, or stop-loss math — those all come from KIS's live
    account/position data independently. The one call site that turns
    a field from here into an allow/deny decision (`state.py`'s
    `add_ticker_to_trade_allowlist`, via `is_leveraged_or_inverse`/
    `is_unsupported_product_type`) matches on the *name*, and the
    krx_master Korean name it'd fall back to matches that filter's
    Korean keyword list (레버리지/인버스/선물) at least as reliably as
    yfinance's English longName — verified against KODEX 레버리지/
    200선물인버스2X/코스닥150레버리지/인버스. A ticker with neither a
    master-file nor yfinance name (KRX and yfinance both down/missing
    it) still ends up with the bare ticker as its name, same as before
    this change — no new gap for that combination, just one fewer
    failure mode (yfinance being the sole point of failure) than today.

    On a REAL account, KRX tickers additionally get a final check
    against `fetch_stock_status`'s underlying TR (CTPF1002R, 주식기본조회,
    prdt_abrv_name) — confirmed live against a real account (a paper
    account gets EGW02006, "모의투자 TR 이 아닙니다", for the whole TR), so
    this only ever fires once real trading is enabled. The same TR's
    scty_grp_id_cd (증권그룹ID코드) also overrides `quote_type` for KRX
    tickers on a real account (see `_KRX_SECURITY_GROUP_TO_QUOTE_TYPE`
    above) — this is the authoritative source, unlike yfinance's `quoteType`,
    which has been confirmed live to misreport plain KOSDAQ stocks as
    "MUTUALFUND" (228340.KS, 142760.KS, 2026-09-03) whenever `shortName`
    (not `longName`) carries the garbled `ticker,internal-id,price` pattern
    that only `get_ticker_display_info`'s own `name` plausibility check
    watches for. On paper accounts or when this TR fails, `quote_type`
    still falls back to yfinance's (name-plausibility-gated) value below.
    US-ticker-shaped
    symbols get the same real-account-only final check from
    `fetch_overseas_company_name` (CTPF1702R), though that TR's field
    names are assumed by analogy rather than live-verified — see that
    method's docstring. previous_close/currency are additionally
    overridden from KIS candles for KRX tickers only (currency is
    trivially "KRW"; previous_close avoids the ~2% yfinance/exchange
    discrepancy noted elsewhere in this module) — out of scope for US
    tickers here, so their previous_close/currency stay yfinance's own
    (or the placeholder's, on a yfinance failure)."""
    master_name: str | None = None
    if _is_krx(ticker):
        await krx_master.ensure_loaded()
        master_name = krx_master.get(ticker)
    elif _is_us(ticker):
        await overseas_master.ensure_loaded()
        master_name = overseas_master.get(ticker)
    try:
        info = await _yf_display_info(ticker)
    except Exception:
        if not (_is_krx(ticker) or _is_us(ticker)):
            raise  # no fallback source exists for a ticker of neither shape
        logger.warning(
            "yfinance display info failed for %s, falling back to master-file name + placeholder fields",
            ticker, exc_info=True,
        )
        info = TickerDisplayInfo(
            symbol=ticker,
            name=master_name or ticker,
            previous_close=0.0,
            currency="KRW" if _is_krx(ticker) else "USD",
        )
    if not (_is_krx(ticker) or _is_us(ticker)):
        return info
    # US tickers: prefer yfinance's own English longName/shortName over
    # overseas_master's value, which is a KOREAN TRANSLITERATION (nasmst.cod
    # col 6, e.g. QQQ->"인베스코 QQQ 트러스트", PANW->"팔로알토 네트웍스" — see
    # that module's docstring). Handing that Korean string to an
    # English-output LLM as a "verified" grounding label pushed it to
    # transliterate/mistranscribe it into a fake company name ("인베스코 QQK
    # 트러스트", "팽탈 앨토 네트웍스") — confirmed root cause of the 2026-08-20
    # ticker-identity-misattribution incident's class-B failures. KRX stays
    # master-first: krx_master cannot cross-symbol-confuse the way
    # yfinance's longName can for .KS codes (see this function's own
    # docstring above and app.dataflows.yfinance.research's 086790.KS case).
    name = info.name if _is_us(ticker) else (master_name or info.name)
    quote_type = info.quote_type
    try:
        broker = await _get_broker()
        if getattr(broker, "_is_real", False):
            if _is_krx(ticker):
                status = await broker.fetch_stock_status(ticker[:-3])
                if status.korean_name:
                    name = status.korean_name
                if status.security_group_code:
                    quote_type = _KRX_SECURITY_GROUP_TO_QUOTE_TYPE.get(
                        status.security_group_code, status.security_group_code
                    )
            else:
                # fetch_overseas_company_name also returns a Korean name
                # (same transliteration concern as above) — only use it to
                # fill in a name we don't already have, never to overwrite
                # yfinance's English name on a real account.
                if not name or name == ticker:
                    korean_name, _ = await broker.fetch_overseas_company_name(ticker)
                    if korean_name:
                        name = korean_name
    except Exception:
        logger.warning("KIS company name lookup failed for %s, using yfinance value", ticker, exc_info=True)
    if not _is_krx(ticker):
        return dataclasses.replace(info, name=name)
    try:
        broker = await _get_broker()
        end = get_clock().today()
        candles = await broker.fetch_daily_candles(
            ticker[:-3], (end - timedelta(days=14)).strftime("%Y%m%d"), end.strftime("%Y%m%d")
        )
        if not candles:
            return dataclasses.replace(info, name=name)
        last = candles[-1]
        # If the most recent row is today's still-forming session, "previous
        # close" means the one before it, not today's live/incomplete price.
        if _candle_date(last.day) == end and is_market_open(ticker) and len(candles) >= 2:
            last = candles[-2]
        return dataclasses.replace(info, name=name, previous_close=last.close, currency="KRW", quote_type=quote_type)
    except Exception:
        logger.warning("KIS previous_close failed for %s, using yfinance value", ticker, exc_info=True)
        return dataclasses.replace(info, name=name, quote_type=quote_type)


async def get_kis_english_name(ticker: str) -> str | None:
    """`prdt_eng_abrv_name` from CTPF1002R (주식기본조회) for a KRX code —
    the only AUTHORITATIVE English name source for a `.KS` ticker. Real
    account only (a paper account gets EGW02006 for the whole TR); returns
    None on a paper broker or any failure, so the caller degrades to the
    yfinance-only weak set exactly as before. 2026-09-03: measurement-only
    addition (identity_gate #6 Problem 2) — feeds into the WEAK alias tier
    for now, not authoritative, until a live cycle confirms KIS's romanization
    actually matches analyst-written names better than yfinance's does."""
    if not _is_krx(ticker):
        return None
    try:
        broker = await _get_broker()
        if not getattr(broker, "_is_real", False):
            return None
        status = await broker.fetch_stock_status(ticker[:-3])
    except Exception:
        logger.info("KIS english name lookup failed for %s", ticker, exc_info=True)
        return None
    return status.english_name or None


async def get_investor_flow(ticker: str, as_of: date | None = None) -> str:
    """종목별 수급 (개인/외국인/기관 순매수) — the Korean market's core
    supply-demand signal, which yfinance has no equivalent for. KRX-only:
    returns a plain "not supported" line for US tickers rather than raising,
    since this is a standalone agent tool with no fallback provider.

    Point-in-time safe for backtests: daily rows are filtered to `as_of`, and
    the intraday estimate (a current-day figure) is included only when the
    analysis date is actually today.
    """
    if not (ticker.endswith(".KS") and ticker[:-3].isdigit()):
        return f"수급(투자자 매매동향) 데이터는 한국(KRX) 종목만 제공됩니다 — {ticker} 미지원"
    try:
        broker = await _get_broker()
    except RuntimeError:
        return "수급 데이터를 가져올 수 없습니다 (KIS 미설정)"

    code = ticker[:-3]
    cutoff = (as_of or get_clock().today()).strftime("%Y%m%d")
    flows = [f for f in await broker.fetch_investor_trend(code) if f.day <= cutoff][:10]
    if not flows:
        return f"{ticker} 수급(투자자 매매동향) 데이터 없음"

    header = (
        f"{'Date':<10}{'Close':>10}{'Individual':>13}{'Foreign':>13}{'Institution':>14}"
        "   (net-buy shares; + = net buyer, - = net seller)"
    )
    lines = [header] + [
        f"{f.day:<10}{f.close:>10.0f}{f.individual_net:>13,}{f.foreign_net:>13,}{f.institution_net:>14,}"
        for f in flows
    ]
    if as_of is None or cutoff == get_clock().today().strftime("%Y%m%d"):
        try:
            est = await broker.fetch_investor_estimate(code)
        except Exception:  # estimate is a bonus, never fail the whole tool for it
            est = []
        if est:
            e = est[-1]
            lines.append(
                f"\nIntraday estimate (당일 가집계): foreign {e.foreign_net:+,} · "
                f"institution {e.institution_net:+,} · total {e.total_net:+,}"
            )
    return (
        f"Investor net-buy flow for {ticker} (source: KIS, last {len(flows)} sessions, "
        "most recent first):\n" + "\n".join(lines)
    )


async def get_kr_vi_status(ticker: str, as_of: date | None = None) -> str:
    """변동성완화장치(VI) 발동 현황 — whether `ticker` is under a VI
    (volatility interruption) halt right now, so the analyst can flag that
    an order may not fill even if its own signal is sound. Ticker-specific
    (unlike get_kr_sector_breadth's whole-market snapshot) — built on
    `KISBroker.fetch_vi_status`, which until now was only used as an
    execution-time order gate (app.execution.kis_bridge's `_vi_block`),
    never surfaced to an analyst as informational context.

    KRX-only, live snapshot only TR — VI only exists during a live session,
    so same "skip for a backtest date" gating as get_kr_market_regime/
    get_kr_sector_breadth.
    """
    if not (ticker.endswith(".KS") and ticker[:-3].isdigit()):
        return f"VI(변동성완화장치) 발동 현황은 한국(KRX) 종목만 제공됩니다 — {ticker} 미지원"
    if as_of is not None and as_of != get_clock().today():
        return "VI 발동 현황은 실시간 값이라 과거 시점 백테스트에선 제공하지 않습니다."
    try:
        broker = await _get_broker()
    except RuntimeError:
        return "VI 발동 현황을 가져올 수 없습니다 (KIS 미설정)"
    code = ticker[:-3]
    try:
        statuses = await broker.fetch_vi_status(code=code)
    except Exception:
        logger.warning("KIS VI status failed for %s", ticker, exc_info=True)
        return f"{ticker} VI 발동 현황 조회 실패"
    if not statuses:
        return f"{ticker}: 오늘 VI 발동 이력 없음"
    # A stock can trigger and release VI more than once in a session; the
    # last row with no released_at (해제시간, empty until lifted) is the
    # unresolved one, same distinction `is_vi_triggered` relies on.
    active = [s for s in statuses if not s.released_at]
    if active:
        s = active[-1]
        return (
            f"{ticker}: VI 발동 중 (트리거 {s.triggered_at}, 발동가 {s.vi_price:.0f}) "
            "— 현재 주문이 체결되지 않을 수 있음"
        )
    s = statuses[-1]
    return f"{ticker}: 오늘 VI 발동 이력 있음 (트리거 {s.triggered_at}, 해제 {s.released_at}) — 현재는 해제됨, 체결 가능"


def _period_labels(periods: list[str]) -> dict[str, str]:
    """Label each `YYYYMM` settlement period as annual (FY) or interim.

    KIS's "annual" financial-statement TRs (called with div_cls="0") are
    documented as an annual series but actually return one leading interim
    row followed by the full-year rows — confirmed live 2026-08-19 on
    316140.KS (Woori) and 207940.KS (Samsung Biologics): a 202603 interim
    row (Q1 2026, revenue=152,433) sits directly above 202512/202412/202312
    FY rows. Rendered with no marker, the model divided the interim row by
    the prior FY row and reported a fabricated "-63% revenue collapse" for
    Woori (152,433 / 415,484 = -63.3%, an exact match) that drove a full
    multi-agent debate to a liquidation recommendation. This function exists
    so callers can mark every row unambiguously instead.

    The fiscal-year-end month is inferred as the *most common* 2-digit month
    suffix across `periods` — the modal month is assumed to be the annual
    close, and any other month is interim relative to it. If that modal
    month doesn't occur at least twice, there isn't enough evidence to
    identify a fiscal-year pattern, so this returns `{}` (label nothing)
    rather than guess.

    Callers must pass `periods` in a deterministic order — sorted ascending
    by `YYYYMM` — because `max` below breaks a count tie by returning the
    *first*-encountered key, and iterating an unordered container (e.g. a
    plain `set`) makes that tie-break depend on PYTHONHASHSEED, silently
    flipping which month is called FY between runs. With periods sorted
    ascending, a tie resolves to the *earliest* period's month, which is the
    correct bias: the interim row is always the most recent one (see
    docstring above), so the older rows are the reliable fiscal-year anchor.
    """
    if not periods:
        return {}
    months = [p[-2:] for p in periods]
    counts: dict[str, int] = {}
    for m in months:
        counts[m] = counts.get(m, 0) + 1
    fy_month = max(counts, key=lambda m: counts[m])
    if counts[fy_month] < 2:
        return {}
    fy = int(fy_month)
    labels: dict[str, str] = {}
    for p in periods:
        p_month = int(p[-2:])
        if p_month == fy:
            labels[p] = f"{p} (FY, 12 months)"
        else:
            elapsed = ((p_month - fy - 1) % 12) + 1
            labels[p] = f"{p} (interim, {elapsed} months — NOT a full year)"
    return labels


@dataclasses.dataclass(frozen=True)
class _InterimContext:
    """The leading interim row and the full-year rows below it, plus the
    exchange's own YoY growth for that interim period. `None` for any growth
    field means KIS did not report it (a literal 0.00 is treated as missing,
    same as `_pct_or_missing` — 2026-08-26 id=785)."""
    period: str
    months: int
    fy_periods: tuple[str, ...]
    revenue_growth: float | None
    op_profit_growth: float | None
    net_income_growth: float | None

    def growths(self) -> list[tuple[str, float]]:
        return [
            (name, v) for name, v in (
                ("revenue", self.revenue_growth),
                ("operating profit", self.op_profit_growth),
                ("net income", self.net_income_growth),
            ) if v is not None
        ]


def _interim_context(labels: dict[str, str], inc: list, rat: list) -> "_InterimContext | None":
    """The interim/FY split for `get_kr_financials`'s downstream period note
    and inline warning. `None` when there's no fiscal-year pattern
    (`_period_labels` returned {}), no interim row, the interim row isn't the
    most recent, or there are no FY rows to be mis-divided against.

    2026-09-01, decision id=1158 (000270.KS/기아): every agent from the
    Fundamentals Analyst on compared FY2024's annual operating profit against
    H1 2026's half-year figure and called it a "three-period sequential
    decline" / "62% collapse". Annualized, H1 2026 was actually +6.5% vs
    FY2025. The exchange's own YoY figures (-10.79% net income for the H1
    period) were in `[Financial ratios]` and the model computed its own
    decline from the raw income-statement rows instead. Labeling the rows
    (`_period_labels`) had already been tried after the 2026-08-19 Woori
    incident and was not enough."""
    if not labels or not inc:
        return None
    interim = [p for p, lbl in labels.items() if "interim" in lbl]
    fy = [p for p, lbl in labels.items() if "(FY," in lbl]
    if not interim or not fy:
        return None
    latest = max(labels)
    if latest not in interim:
        return None  # interim row isn't the most recent — the mis-division shape doesn't apply
    months_m = re.search(r"interim, (\d+) months", labels[latest])
    rat_by_period = {getattr(r, "period", None): r for r in rat}
    row = rat_by_period.get(latest)

    def _g(attr: str) -> float | None:
        v = getattr(row, attr, None) if row is not None else None
        return None if v is None or v == 0.0 else float(v)

    return _InterimContext(
        period=latest,
        months=int(months_m.group(1)) if months_m else 6,
        fy_periods=tuple(sorted(fy, reverse=True)),
        revenue_growth=_g("revenue_growth"),
        op_profit_growth=_g("op_profit_growth"),
        net_income_growth=_g("net_income_growth"),
    )


def _interim_growth_sentence(ctx: _InterimContext) -> str:
    """'revenue +2.87%, operating profit +6.51%, net income -10.79%' — empty
    string when KIS reported none of the three."""
    parts = [f"{name} {v:+.2f}%" for name, v in ctx.growths()]
    return ", ".join(parts)


def _interim_warning_block(ctx: "_InterimContext | None") -> str:
    """The named-row warning that rides INSIDE the [Income statement] block
    (empty when there's no interim row). Unlike `_period_labels`'s per-row
    marker and the block's general rule, this points at the exact row pair —
    the 2026-08-19 Woori fix and its per-row label were not enough on their
    own (id=1158)."""
    if ctx is None:
        return ""
    fy_list = ", ".join(ctx.fy_periods)
    block = (
        f"\n  PERIOD WARNING — the top row ({ctx.period}) is a {ctx.months}-MONTH interim; the rows "
        f"below it ({fy_list}) are FULL YEARS. A {ctx.months}-month figure is smaller than a "
        f"12-month figure for that reason alone. Dividing {ctx.period} by {ctx.fy_periods[0]} (or "
        f"by any FY row) does NOT produce a decline, a slowdown, a collapse, or a growth rate — it "
        f"produces a meaningless number. If you are about to write that revenue or profit fell, "
        f"check first that you are not comparing {ctx.period} against a FY row."
    )
    growth = _interim_growth_sentence(ctx)
    if growth:
        block += (
            f" The only correct year-over-year read for {ctx.period} is the exchange's own, on that "
            f"row above: {growth}."
        )
    return block


def interim_period_note(ctx: "_InterimContext | None") -> str:
    """The downstream period-mismatch warning threaded to the Research
    Manager / Trader / risk debate / Portfolio Manager (2026-09-01 audit #5).
    Empty string when there's no leading interim row, so a prompt for a
    ticker with only full-year data (or a US ticker) stays byte-identical."""
    if ctx is None:
        return ""
    fy_list = ", ".join(ctx.fy_periods)
    note = (
        f"KOREAN INTERIM-PERIOD WARNING: this company's most recent exchange-reported financial "
        f"period ({ctx.period}) is a {ctx.months}-MONTH INTERIM, not a full year. Every figure for "
        f"it — revenue, operating profit, net income — covers part of a year and is therefore "
        f"smaller than a full-year figure by construction. A 'decline', 'collapse', 'slowdown' or "
        f"'-NN%' derived by comparing that interim period against a full fiscal year ({fy_list}) is "
        f"NOT a real deterioration; it is a part-year number set against a 12-month number."
    )
    growth = _interim_growth_sentence(ctx)
    if growth:
        note += (
            f" The exchange's own year-over-year figures for that interim period, against the same "
            f"period one year earlier, are: {growth}. Use those."
        )
    note += (
        " This warning is strictly about comparing an interim figure against a "
        "FULL-YEAR figure. It does not apply to a comparison of this interim period "
        "against the SAME period one year earlier, and it does not apply to margins, "
        "ratios, or any other percentage-of-revenue figure — those do not depend on how "
        "many months the period covers, so a margin that fell year-over-year fell for "
        "real. Do not cite this warning to dismiss a like-for-like or margin-based "
        "deterioration."
        " This applies to your own prose and to any earlier report, research plan, or debate "
        "transcript quoted below that says otherwise — if one of them describes a collapse or a "
        "multi-period sequential decline built on that comparison, treat it as an error and say so "
        "rather than carrying it forward."
    )
    return note


def _months_between(period_yyyymm: str, ref: date) -> int | None:
    """Whole calendar months from the close of `period_yyyymm` (a KIS
    'YYYYMM' settlement month) to `ref`. `None` if the period string isn't
    six digits."""
    if not (len(period_yyyymm) == 6 and period_yyyymm.isdigit()):
        return None
    py, pm = int(period_yyyymm[:4]), int(period_yyyymm[4:])
    return (ref.year - py) * 12 + (ref.month - pm)


def kr_financials_recency(latest_period: str, latest_label: str, ref: date) -> dict:
    """The facts about how current `get_kr_financials`' newest row is —
    recorded on `PipelineState.kr_financials_recency` so a batch audit can
    ask, in SQL, how many domestic decisions ran on financials a full fiscal
    year (or more) out of date. That number is otherwise invisible in every
    stored field. `months_stale` is `None` when the period string can't be
    parsed."""
    return {
        "latest_period": latest_period,
        "latest_label": latest_label,
        "months_stale": _months_between(latest_period, ref),
        "interim": "interim" in latest_label,
    }


def _staleness_note(recency: dict, ref: date) -> str:
    """The downstream RECENCY warning threaded to the Research Manager /
    Trader / risk debate / Portfolio Manager alongside `interim_period_note`.
    Empty string below `settings.kr_financials_staleness_warn_months`, so a
    healthy ticker's (or a US ticker's) prompt stays byte-identical — the
    same discipline `interim_period_note` uses.

    2026-09-07 domestic audit #6 (id=1237 / 294630.KS): KIS's annual series
    for the issuer ended at FY2024 — 20 months before the analysis date —
    and there was no signal anywhere that the figures were stale. The desk
    read "December 2024" numbers as the company's current financial
    condition and a second node invented a fresher-sounding fiscal year to
    match. The provider knows exactly how old its newest row is; this states
    it.

    Deliberately carries no `%`, no `억`/`억원`/`billion`/`trillion` token,
    and no `<item>=<n>` pair — so it can't be picked up as a figure by
    `unit_scale_guard._EOK_RE`, as a growth rate by
    `period_mismatch_guard._REAL_GROWTH_RE`, or as an income row by
    `period_mismatch_guard._INC_ROW_RE` (which also needs a line start it
    never gets)."""
    months = recency.get("months_stale")
    if months is None or months < settings.kr_financials_staleness_warn_months:
        return ""
    label = recency.get("latest_label") or recency.get("latest_period") or "?"
    note = (
        f"FINANCIAL-DATA RECENCY: as of {ref.isoformat()}, the most recent financial period the "
        f"exchange (KIS) reports for this company is {label}, which closed roughly {months} months "
        f"ago. No newer annual report and no newer interim figures are available from this source "
        f"for this company. Every figure below therefore describes the company as it stood at that "
        f"period, not as it stands now — cite them with their period attached, and do not present "
        f"them as the company's \"current\", \"latest\", \"recent\" or \"most recent quarter\" "
        f"financial condition, or use them to characterise where the business stands today. "
        f"Whatever has changed in the months since — including anything in this cycle's news — is "
        f"reflected nowhere in this data."
    )
    if months >= 2 * settings.kr_financials_staleness_warn_months:
        note += (
            f" A listed issuer with no financial filing reaching this data source in {months}+ "
            f"months is itself a material concern — a 관리종목 designation, an audit-opinion "
            f"problem, or a delisting review can all present this way — so weigh that possibility, "
            f"not only the staleness of the numbers."
        )
    return note


def _eok(value: float) -> str:
    """`value` is already in 억원 (KIS convention throughout this
    module). Comma-grouped numbers group every 3 digits (Western,
    "39,045"), but 조/억 group every 4 digits (10,000-based) — a live
    reproduction (2026-07-30, 009540.KS) confirmed the model reads a
    3-digit comma group as if it aligned with the 조 boundary, turning
    "39,045억원" (실제 3조 9,045억원) into "39조 45억원" — a 10x error,
    distinct from (and found after fixing) the scientific-notation and
    column-swap bugs. Spelling out the 조 conversion here removes that
    division from the model entirely for anything at or above 1조;
    below that, 억원 alone is unambiguous and left as-is.

    2026-08-19 addendum: once the pipeline's output language switched to
    English, the model had to translate 억원 itself and wrote
    "173,363억원" as "173,363 billion won" (=173 trillion KRW, a
    different 10x error from the one above). An explicit English gloss
    is now appended to both branches so the unit is unambiguous
    regardless of the report's output language, without removing the
    Korean rendering (still needed for Korean-language output).

    2026-08-28 addendum: the >=1조 branch used to give ONLY a trillion-KRW
    gloss, no billion-KRW one — so a model that reached for "billion"
    anyway (English financial writing defaults to billion for anything
    this large) had no supplied answer to quote and had to derive one
    itself, and every observed derivation just carried the 억원 mantissa
    straight over ("951,542억원" -> "951,542 billion KRW", a 10x error —
    the correct value is 95,154.2 billion). 70 reports across 20 tickers
    hit this in one day's live audit, including two that then concluded
    the (accurately-sourced) data must belong to a different, larger
    company. Giving both units unconditionally, same as the <1조 branch,
    removes the model's need to derive either one — see
    app.agents.unit_scale_guard for the sweep that catches whatever
    still gets derived wrong anyway.

    2026-09-11 addendum: hoisted from a nested closure inside
    `get_kr_financials` to module scope so `_kis_balance_sheet` and
    `get_kr_earnings_estimate` — two other renderers of KIS 억원 money
    figures that were previously shipping bare, equivalent-free numbers —
    could share this exact conversion instead of each growing (or
    skipping) their own copy. Pure function, no behavior change."""
    if abs(value) < 10_000:
        return f"{value:,.0f}억원 (= {value / 10:,.1f} billion KRW)"
    return (
        f"{value:,.0f}억원({value / 10_000:,.2f}조원) "
        f"(= {value / 10:,.1f} billion KRW = {value / 10_000:,.2f} trillion KRW)"
    )


def _eok_or_missing(value: float) -> str:
    """Sibling of `_pct_or_missing` / `_won_or_missing` (below, in
    `get_kr_financials`) for the absolute won-denominated 억원 figures `_eok`
    renders (income statement, balance sheet, valuation, EBITDA/EVA): `_num`
    (app/brokers/kis/parser.py) coerces a missing KIS field to 0.0 exactly
    the same way it does for a ratio or a per-share figure, and `_eok(0.0)`
    then renders that as a confident "0억원 (= 0.0 billion KRW)" —
    indistinguishable from a real, reported zero.

    Confirmed live (2026-09-15, decision id=1433 / 004310.KS/현대약품): every
    income-statement and balance-sheet field KIS omitted came back as 0.0,
    and the Fundamentals Analyst reported "zero revenue, zero gross profit,
    zero operating profit, zero net income for FY2022-FY2025" and "total
    assets, liabilities and equity all reported as zero", concluding the
    company was a "dormant shell company" with "zero economic activity" —
    a false premise the Portfolio Manager then wrote into the final
    decision ("four consecutive years of zero revenue and equity eliminates
    any intrinsic value floor"). Ground truth (yfinance): revenue ₩162.7B /
    ₩180.8B / ₩175.7B / ₩191.8B for FY2022-2025, total assets ₩163.2B — a
    normal operating pharmaceutical company. Two more verified cases in the
    21 days before that (003010.KS/혜인, 006380.KS/카프로) show the same
    shape. Same bug, same fix as `_pct_or_missing` / `_won_or_missing`: an
    exact 0 is unattainable for a listed going concern in these fields, so
    0.0 always means "KIS returned nothing for this period", never a real
    zero.

    `revenue` is deliberately NOT routed through this helper directly — see
    `_income_revenue_or_missing` / `_estimate_revenue_or_missing` below
    instead. A KOSDAQ SPAC (기업인수목적회사) or a pre-revenue biotech can
    genuinely report exactly 0 revenue for a full fiscal year, and that is
    real information the model should see; but such a company still has a
    non-zero operating loss (listing/admin costs), so its row is never ALL
    zero. An all-zero row (revenue AND its sibling money fields all exactly
    0.0, as in the 004310.KS case) means KIS returned nothing for that
    period. `total_assets`/`total_liabilities`/`total_equity`/
    `current_assets`/`current_liabilities`/`market_cap` carry no such
    exemption and always go through this helper directly. `ebitda`/`eva`
    also go through this helper unconditionally — a lower-confidence
    extension of the same argument (an exact 0 EBITDA/EVA is likewise not
    attainable for a going concern), though these were not the field
    observed failing live.

    Do NOT widen this to `ev_ebitda` (rendered as `:.2f}x`, not through
    `_eok` — a different kind of field, out of scope) or to any `*_growth`
    percentage (already `_pct_or_missing`'s / `_growth_line`'s territory) or
    to `borrowing_dependency` (deliberately exempt — see `_pct_or_missing`'s
    docstring)."""
    return "not reported" if value == 0.0 else _eok(value)


def _income_revenue_or_missing(s: IncomeStatement) -> str:
    """Row-gated revenue rendering for a `get_kr_financials` income-statement
    row (see `_eok_or_missing`'s docstring for the SPAC/pre-revenue
    exemption this preserves): `revenue` renders as "not reported" only when
    `revenue` itself, AND `gross_profit`, `operating_profit` AND
    `net_income`, are ALL exactly 0.0 for this same row — i.e. KIS returned
    nothing at all for the period, the 004310.KS shape. If `revenue` is
    non-zero it always renders through the normal `_eok`, even when every
    sibling field is 0.0 — a real reported revenue figure must never be
    hidden behind "not reported" just because KIS omitted the other fields."""
    if (
        s.revenue == 0.0
        and s.gross_profit == 0.0
        and s.operating_profit == 0.0
        and s.net_income == 0.0
    ):
        return "not reported"
    return _eok(s.revenue)


def _estimate_revenue_or_missing(r: EarningsEstimate) -> str:
    """`_income_revenue_or_missing`'s counterpart for a
    `get_kr_earnings_estimate` consensus row, which has no `gross_profit`
    field: `revenue` renders as "not reported" only when `revenue` itself,
    AND `operating_profit` AND `net_income`, are ALL exactly 0.0 for this
    same row. If `revenue` is non-zero it always renders through `_eok`,
    even when both siblings are 0.0 — a real reported revenue figure must
    never be hidden behind "not reported" just because KIS omitted the
    other fields."""
    if r.revenue == 0.0 and r.operating_profit == 0.0 and r.net_income == 0.0:
        return "not reported"
    return _eok(r.revenue)


# Shared unit-key prose for any per-period 억원 table that renders through
# `_eok` (`get_kr_financials`'s [Balance sheet] section, `_kis_balance_sheet`,
# `get_kr_earnings_estimate`'s money columns). Extracted 2026-09-11 so the
# three renderers can't drift into three slightly different unit
# explanations of the same underlying convention — see
# `format_verified_snapshot`'s docstring (app/dataflows/session_state.py)
# for this codebase's stated preference for one shared builder over
# "same text shape promised in prose" duplication.
# 억원 단위 안내문 중복 방지 — 세 렌더러가 동일 문구를 공유하도록 모듈 상수로 추출.
_EOK_TABLE_UNIT_KEY = (
    "억원, with billion-KRW and trillion-KRW equivalents "
    "already computed and given in parentheses — quote one of the supplied "
    "numbers as-is, never re-derive either yourself. Unit key: 1억원 = 0.1 "
    "billion KRW = 0.0001 trillion KRW — do NOT write an 억원 figure's digits "
    "followed by \"billion\""
)


async def get_kr_financials(
    ticker: str,
    as_of: date | None = None,
    period_note_out: list[str] | None = None,
    recency_out: dict | None = None,
) -> str:
    """국내 재무 (손익계산서·재무상태표·재무비율) — KIS reports Korean-company
    financials more completely than yfinance does. KRX-only (US tickers get a
    not-supported line). Point-in-time: periods are filtered to on/before
    `as_of`'s settlement month (approximate — ignores filing lag, but blocks
    obvious future periods in a backtest).

    `period_note_out`, if given, is appended with `interim_period_note(...)`
    and/or `_staleness_note(...)` (joined when both apply) — the pipeline
    threads that to every downstream node so a period-mismatch decline the
    Fundamentals report made anyway can still be caught (2026-09-01 audit
    #5, decision id=1158), and so a desk reading a full-year-stale figure as
    the company's current condition is warned (2026-09-07 audit #6, id=1237).
    Nothing is appended when the newest row is a fresh full year.

    `recency_out`, if given, is `.update()`d with `kr_financials_recency(...)`
    — `{latest_period, latest_label, months_stale, interim}` — recorded on
    `PipelineState.kr_financials_recency` for batch audit. Untouched (left as
    the caller's empty default) for a non-KRX ticker or when no periods came
    back."""
    if not (ticker.endswith(".KS") and ticker[:-3].isdigit()):
        return f"국내 재무 데이터는 한국(KRX) 종목만 제공됩니다 — {ticker} 미지원"
    try:
        broker = await _get_broker()
    except RuntimeError:
        return "국내 재무 데이터를 가져올 수 없습니다 (KIS 미설정)"

    code = ticker[:-3]
    ref_date = as_of or get_clock().today()
    cutoff = ref_date.strftime("%Y%m")
    # Cap concurrent KIS calls (7 here + fetch_valuation below) — see
    # _get_kr_financials_gate. A failed TR still propagates (asyncio.gather
    # without return_exceptions): one rate-limited ratio TR fails the whole
    # report, the analyst degrades, and Part B surfaces it — better than a
    # silently partial report.
    gate = _get_kr_financials_gate()

    async def _guarded(make):
        # `make` is a zero-arg coroutine factory, not a coroutine — so a
        # sibling failing fast (gather cancels the rest) can't leave an
        # un-awaited coroutine behind for the ones still queued on the gate.
        async with gate:
            return await make()

    inc, bal, rat, stab, prof, growth, other = await asyncio.gather(
        _guarded(lambda: broker.fetch_kr_income_statement(code)),
        _guarded(lambda: broker.fetch_kr_balance_sheet(code)),
        _guarded(lambda: broker.fetch_kr_financial_ratio(code)),
        _guarded(lambda: broker.fetch_kr_stability_ratio(code)),
        _guarded(lambda: broker.fetch_kr_profitability_ratio(code)),
        _guarded(lambda: broker.fetch_kr_growth_ratio_detail(code)),
        _guarded(lambda: broker.fetch_kr_other_major_ratios(code)),
    )
    inc = [x for x in inc if x.period <= cutoff][:4]
    bal = [x for x in bal if x.period <= cutoff][:4]
    rat = [x for x in rat if x.period <= cutoff][:4]
    stab = [x for x in stab if x.period <= cutoff][:4]
    prof = [x for x in prof if x.period <= cutoff][:4]
    growth = [x for x in growth if x.period <= cutoff][:4]
    other = [x for x in other if x.period <= cutoff][:4]
    # Valuation multiples (PER/PBR/market cap) are computed off *today's*
    # live price — same "current price/share count can't be rebound to a
    # past date" limitation get_fundamentals's docstring already notes for
    # yfinance, so this section is live-only (skipped for a backtest date).
    val = None
    if as_of is None or as_of == get_clock().today():
        try:
            val = await _guarded(lambda: broker.fetch_valuation(code))
        except Exception:  # noqa: BLE001 - valuation is a bonus section, never fail the whole call for it
            val = None
    if not (inc or bal or rat or stab or prof or growth or other or val):
        return f"{ticker} 국내 재무 데이터 없음"

    # Built once from the union of periods present across the fetched
    # tables — the tables share the same settlement periods in practice, but
    # `.get(period, period)` below falls back to the bare period string
    # rather than raising if one is somehow missing from this map.
    # `sorted`, not `list`, over the set: iterating a Python set has an order
    # that depends on PYTHONHASHSEED, and `_period_labels` below breaks a
    # count tie by picking the *first*-encountered month — so an unsorted
    # input let the same tie-input data get labelled two opposite ways
    # (which rows are "FY") across process runs. Sorting ascending makes the
    # input order (and therefore the label) deterministic.
    all_periods = sorted(
        {s.period for s in (*inc, *bal, *rat, *stab, *prof, *growth, *other)}
    )
    labels = _period_labels(all_periods)
    interim_ctx = _interim_context(labels, inc, rat)
    latest_period = all_periods[-1] if all_periods else ""
    recency = (
        kr_financials_recency(latest_period, labels.get(latest_period, latest_period), ref_date)
        if latest_period else {}
    )
    if recency and recency_out is not None:
        recency_out.update(recency)
    staleness_note = _staleness_note(recency, ref_date) if recency else ""
    if period_note_out is not None:
        _notes = [
            n for n in (
                interim_period_note(interim_ctx) if interim_ctx is not None else "",
                staleness_note,
            ) if n
        ]
        if _notes:
            period_note_out.append("\n\n".join(_notes))
    # Exchange-reported YoY growth per period, to print ON the income-statement
    # rows (2026-09-01 audit #5) instead of only in [Financial ratios] — the
    # model kept computing its own decline from the raw rows there.
    _growth_by_period = {getattr(r, "period", None): r for r in rat}

    def _growth_line(period: str) -> str:
        r = _growth_by_period.get(period)
        if r is None:
            return ""
        bits = []
        for label, attr in (
            ("revenue_growth", "revenue_growth"),
            ("operating_profit_growth", "op_profit_growth"),
            ("net_income_growth", "net_income_growth"),
        ):
            v = getattr(r, attr, None)
            bits.append(f"{label}={'not reported' if v is None or v == 0.0 else f'{v:+.2f}%'}")
        return "\n    | exchange-reported YoY for this period (vs the same period one year earlier, KIS): " + " ".join(bits)

    def _pct_or_missing(value: float) -> str:
        """`_num` (app/brokers/kis/parser.py) coerces a missing KIS field
        to 0.0, which this line then rendered as a literal "0.00%" —
        indistinguishable from a real zero. Confirmed live (2026-08-26,
        002990.KS/금호건설): KIS omitted `quck_rate` for the 202606 interim
        while reporting current_ratio=89.49%, and the Portfolio Manager
        (id=785) quoted "당좌비율 0.00%" as a "치명적인 유동성 취약성" in its
        Hold rationale. yfinance's own quickRatio for that same period is
        0.656 (its currentRatio 0.895 matches KIS's 89.49% exactly, so it
        is the same period) — the zero was missing data, not distress. A
        current/quick ratio of exactly zero is not attainable for a listed
        going concern, so 0.00 in these two fields is always "not
        reported". `borrowing_dependency` is left alone: a genuinely
        debt-free company really can report 0 there.

        2026-09-10 widened scope: hoisted out of the `[Stability ratios]`
        block (where this docstring originated) so `[Financial ratios]`,
        `[Profitability ratios]`, and `[Growth ratios, additional]` can
        share it too — those three had the identical raw-`{value:.2f}%`
        shape and the identical hazard, just never audited. `roe`,
        `debt_ratio`, `revenue_growth`, `net_income_growth`, `gross_margin`,
        `net_margin`, `return_on_capital`, `return_on_equity`,
        `equity_growth`, `asset_growth` all cannot legitimately be exactly
        0.00% for a listed operating company either, so the same "exact
        zero -> not reported" convention applies. `revenue_growth` /
        `net_income_growth` already used this convention in `_growth_line`
        above for the identical concept (KIS YoY growth) — leaving the
        ratio-block copies raw produced the observed self-contradiction of
        one report saying `revenue_growth=not reported` on the income-
        statement row and `revenue_growth=0.00%` eight lines later in
        [Financial ratios].

        Exemption rule (do not widen further without re-checking this):
        only apply this where an exact zero is unattainable for a going
        concern. `borrowing_dependency` stays exempt (above). Any
        dividend/payout field would need the same exemption for the
        opposite reason (a company that pays no dividend really is 0), but
        none of `roe`/`eps`/`bps`/`debt_ratio`/`revenue_growth`/
        `net_income_growth`/`gross_margin`/`net_margin`/
        `return_on_capital`/`return_on_equity`/`equity_growth`/
        `asset_growth` is a dividend/payout field, so no new exemption is
        needed here. `payout_ratio` (in `[Other key ratios]`) is a separate,
        untouched block."""
        return "not reported" if value == 0.0 else f"{value:.2f}%"

    def _won_or_missing(value: float) -> str:
        """Sibling of `_pct_or_missing` for won-denominated per-share
        integers (EPS/BPS): `_num` coercing a missing field to 0.0 renders
        as a confident "0원", and an exact 0 EPS/BPS is just as unattainable
        for a listed going concern as an exact 0.00% ratio is — same bug,
        same fix, different formatting (thousands-grouped integer won, not
        a percentage)."""
        return "not reported" if value == 0.0 else f"{value:,.0f}원"

    sections = [f"Korean financial statements for {ticker} (source: KIS, most recent periods first)"]
    if staleness_note:
        sections.append("\n" + staleness_note)
    if val:
        sections.append(
            "\n[Valuation] (live, source: KIS)\n"
            f"  Current price {val.price:,.0f}원 | PER {val.per:.2f} | PBR {val.pbr:.2f} | "
            f"EPS {val.eps:,.0f} | BPS {val.bps:,.0f} | Market cap {_eok_or_missing(val.market_cap)}"
        )
    # Every value below is labeled inline (name=value, same convention the
    # [Valuation] section above already uses) rather than a bare positional
    # "period | v1 | v2 | v3" row with the column names only named once in
    # a header far above. Confirmed live (2026-07-30, 009540.KS
    # reproduction) that the positional form gets columns swapped across
    # DIFFERENT sections — a ratio-table EPS/BPS value written into the
    # income-statement narrative as if it were revenue/net income. Four
    # periods x four tables is a lot of numbers to track back to one
    # header line; inline labels remove that tracking requirement entirely.
    if inc:
        sections.append(
            "\n[Income statement] (억원, with billion-KRW and trillion-KRW equivalents "
            "already computed and given in parentheses — quote one of the supplied "
            "numbers as-is, never re-derive either from the 억원 figure yourself. Unit "
            "key: 1억원 = 0.1 billion KRW = 0.0001 trillion KRW — do NOT write an 억원 "
            "figure's digits followed by \"billion\" (e.g. 89,859억원 is 8,985.9 billion "
            "KRW, not 89,859 billion). A row "
            "marked `(interim, N months)` covers only part of a fiscal year and is NOT "
            "comparable to a `(FY, 12 months)` row. Never compute a growth rate, decline, "
            "or \"collapse\" by dividing an interim row by a full-year row — that "
            "comparison is meaningless. For growth rates use the revenue_growth / "
            "net_income_growth fields in [Financial ratios], which the exchange computes "
            "against the correct comparable period. The exchange's own YoY growth for each "
            "period is printed on the row itself below — use it; do NOT compute your own growth "
            "rate, decline, or \"collapse\" by dividing one row's figure by another's.)\n"
            + "\n".join(
                f"  {labels.get(s.period, s.period)}: revenue={_income_revenue_or_missing(s)} "
                f"gross_profit={_eok_or_missing(s.gross_profit)} operating_profit={_eok_or_missing(s.operating_profit)} "
                f"net_income={_eok_or_missing(s.net_income)}{_growth_line(s.period)}"
                for s in inc
            )
            + _interim_warning_block(interim_ctx)
        )
    if bal:
        sections.append(
            f"\n[Balance sheet] ({_EOK_TABLE_UNIT_KEY})\n"
            + "\n".join(
                f"  {labels.get(s.period, s.period)}: total_assets={_eok_or_missing(s.total_assets)} "
                f"total_liabilities={_eok_or_missing(s.total_liabilities)} total_equity={_eok_or_missing(s.total_equity)}"
                for s in bal
            )
        )
    if rat:
        sections.append(
            "\n[Financial ratios] (EPS/BPS in 원 per share, not 억원 — everything else is a %. "
            "debt_to_equity_ratio is KIS's 부채비율 (`lblt_rate`) = total liabilities ÷ total "
            "EQUITY × 100 — the Korean-market convention. Readings ABOVE 100% are normal and "
            "expected (they only mean liabilities exceed book equity; Korean capital-intensive "
            "industrials routinely run 100~400%), so never call a >100% reading impossible, "
            "corrupted, or a data error. Never describe this field as \"debt-to-assets\", "
            "\"liabilities/assets\", \"debt/total assets\" or any other assets-denominated "
            "wording — a liabilities/assets ratio cannot exceed 100% by definition. If you want "
            "an assets-denominated leverage view, use borrowing_dependency in [Stability ratios] "
            "below, or divide total_liabilities by total_assets from [Balance sheet] above.)\n"
            + "\n".join(
                f"  {labels.get(s.period, s.period)}: ROE={_pct_or_missing(s.roe)} EPS={_won_or_missing(s.eps)} "
                f"BPS={_won_or_missing(s.bps)} debt_to_equity_ratio={_pct_or_missing(s.debt_ratio)} "
                "(부채비율 = liabilities÷EQUITY) "
                f"revenue_growth={_pct_or_missing(s.revenue_growth)} net_income_growth={_pct_or_missing(s.net_income_growth)}"
                for s in rat
            )
        )
    if prof:
        sections.append(
            "\n[Profitability ratios] (margins, not growth — everything is a %. "
            "return_on_capital is KIS's 총자본순이익율 = net income ÷ TOTAL capital (총자본, "
            "i.e. total assets) — it is NOT the Western \"return on invested capital\"/ROIC "
            "(NOPAT ÷ invested capital); do not call it ROIC. return_on_equity is 자기자본순이익율 "
            "= net income ÷ shareholders' equity, the same concept as the ROE field in "
            "[Financial ratios] above — a small difference between the two is a different KIS "
            "TR's rounding, not a contradiction.)\n"
            + "\n".join(
                f"  {labels.get(s.period, s.period)}: gross_margin={_pct_or_missing(s.gross_margin)} "
                f"net_margin={_pct_or_missing(s.net_margin)} return_on_capital={_pct_or_missing(s.return_on_capital)} "
                f"return_on_equity={_pct_or_missing(s.return_on_equity)}"
                for s in prof
            )
        )
    if stab:
        sections.append(
            "\n[Stability ratios] (liquidity/leverage, not the debt_to_equity_ratio already given "
            "above — everything is a %. borrowing_dependency is 차입금의존도 = interest-bearing "
            "debt ÷ TOTAL ASSETS × 100 — this is the ASSETS-denominated leverage metric, and by "
            "construction it cannot exceed 100%; do not confuse it with debt_to_equity_ratio "
            "above, which is equity-denominated and routinely does.)\n"
            + "\n".join(
                f"  {labels.get(s.period, s.period)}: current_ratio={_pct_or_missing(s.current_ratio)} "
                f"quick_ratio={_pct_or_missing(s.quick_ratio)} borrowing_dependency={s.borrowing_dependency:.2f}%"
                for s in stab
            )
        )
    if growth:
        sections.append(
            "\n[Growth ratios, additional] (revenue/operating-profit growth is already in [Financial ratios] "
            "above — this is equity/asset growth only, everything is a %)\n"
            + "\n".join(
                f"  {labels.get(s.period, s.period)}: equity_growth={_pct_or_missing(s.equity_growth)} "
                f"asset_growth={_pct_or_missing(s.asset_growth)}"
                for s in growth
            )
        )
    if other:
        sections.append(
            "\n[Other key ratios] (payout_ratio is a %; EVA/EBITDA are 억원 like the statements above "
            "— quote their supplied billion/trillion-KRW equivalent as-is, never re-derive it (1억원 = "
            "0.1 billion KRW = 0.0001 trillion KRW); ev_ebitda is a plain multiple, not a %. EVA/EBITDA "
            "already render as 'not reported' rather than a bare zero when KIS omitted the field (see "
            "_eok_or_missing) — for payout_ratio, an exact 0.00 may still mean 'not provided this "
            "period' rather than a confirmed zero, so say so if it looks that way rather than "
            "asserting it as a real zero)\n"
            + "\n".join(
                f"  {labels.get(s.period, s.period)}: payout_ratio={s.payout_ratio:.2f}% "
                f"EBITDA={_eok_or_missing(s.ebitda)} EV/EBITDA={s.ev_ebitda:.2f}x EVA={_eok_or_missing(s.eva)}"
                for s in other
            )
        )
    return "\n".join(sections)


async def get_kr_analyst_opinion(ticker: str, as_of: date | None = None) -> str:
    """국내 증권사 투자의견 — domestic brokerage opinions and target prices,
    which yfinance's consensus almost entirely misses for Korean names.
    KRX-only; point-in-time filtered to on/before `as_of`."""
    if not (ticker.endswith(".KS") and ticker[:-3].isdigit()):
        return f"국내 투자의견 데이터는 한국(KRX) 종목만 제공됩니다 — {ticker} 미지원"
    try:
        broker = await _get_broker()
    except RuntimeError:
        return "국내 투자의견 데이터를 가져올 수 없습니다 (KIS 미설정)"

    code = ticker[:-3]
    end = as_of or get_clock().today()
    start = end - timedelta(days=180)
    opinions = await broker.fetch_invest_opinion(
        code, start.strftime("%Y%m%d"), end.strftime("%Y%m%d")
    )
    opinions = [o for o in opinions if o.day <= end.strftime("%Y%m%d")]
    opinions.sort(key=lambda o: o.day, reverse=True)
    opinions = opinions[:12]
    if not opinions:
        return f"{ticker} 최근 6개월 국내 증권사 투자의견 없음"

    header = f"{'Date':<10}{'Opinion':<12}{'Prev':<12}{'TargetPrice':>13}{'CloseThen':>12}"
    lines = [header] + [
        f"{o.day:<10}{o.opinion:<12}{o.prev_opinion:<12}{o.target_price:>13,.0f}{o.prev_close:>12,.0f}"
        for o in opinions
    ]
    return (
        f"Domestic brokerage investment opinions for {ticker} (source: KIS, last "
        f"{len(opinions)}, most recent first):\n" + "\n".join(lines) + "\n"
        # Corrected 2026-09-11: the note previously claimed CloseThen was the
        # close ON that row's own Date. It is actually KIS field
        # stck_prdy_clpr (see prev_close mapping in
        # app/brokers/kis/domestic/info.py and app/domain/models/market.py),
        # which is the PREVIOUS trading day's close — i.e. the close of the
        # session immediately BEFORE that row's Date, last known as of that
        # opinion's publication.
        "Note: CloseThen is the closing price of the trading day immediately "
        "BEFORE that row's own Date (KIS field stck_prdy_clpr) — the last "
        "close known when that opinion was published — it is NOT today's "
        "current price. Any upside computed as target_price vs this column is "
        "therefore measured against the prior session's close, not the price "
        "on the opinion's own date. For the current price, use the "
        "`Current price` field in the [Valuation] section of the "
        "financial-statements block above."
    )


async def get_kr_risk_signals(ticker: str, as_of: date | None = None) -> str:
    """Korean risk-side signals for the risk debate: daily short-selling
    intensity and program-trade net-buy — both distinct pressures that
    yfinance doesn't surface for KRX names. KRX-only; short-sale rows are
    point-in-time filtered, and the (intraday) program figure is included
    only when the analysis date is today."""
    if not (ticker.endswith(".KS") and ticker[:-3].isdigit()):
        return f"국내 리스크 신호(공매도/프로그램)는 한국(KRX) 종목만 제공됩니다 — {ticker} 미지원"
    try:
        broker = await _get_broker()
    except RuntimeError:
        return "국내 리스크 신호를 가져올 수 없습니다 (KIS 미설정)"

    code = ticker[:-3]
    end = as_of or get_clock().today()
    end_str = end.strftime("%Y%m%d")
    today_str = get_clock().today().strftime("%Y%m%d")
    # Both TRs below are bonus context for the risk debate, fetched AFTER the
    # analysts and the Bull/Bear debate have already run (pipeline.py awaits
    # get_kr_risk_signals late in the run) — an unguarded EGW00201 (or any
    # other broker exception) on a decorative short-sale/program-trade TR
    # would discard a completed multi-agent analysis over data that is not
    # essential to it. `get_kr_analyst_opinion`'s sibling fetch
    # (fetch_investor_estimate, ~line 637) already states and applies this
    # exact policy ("estimate is a bonus, never fail the whole tool for
    # it") — same shape here, degrading to an empty list so the existing
    # `else`/`if program:` branches below render the same "no data" text
    # they already render for a genuinely-empty KIS response.
    # 보너스 성격의 공매도/프로그램매매 TR 실패로 전체 파이프라인이 죽지 않도록 방어.
    try:
        shorts = [
            s for s in await broker.fetch_daily_short_sale(
                code, (end - timedelta(days=30)).strftime("%Y%m%d"), end_str
            )
            if s.day <= end_str
        ]
    except Exception:
        logger.warning("KIS fetch_daily_short_sale failed for %s", ticker, exc_info=True)
        shorts = []
    shorts.sort(key=lambda s: s.day, reverse=True)
    shorts = shorts[:10]

    sections = [f"Korean risk signals for {ticker} (source: KIS)"]
    if shorts:
        def _short_cells(s) -> tuple[str, str, str, str]:
            """(close, short_vol, short_pct, accum_pct) already width-padded,
            with a missing KIS field shown as `missing` rather than 0.

            `_num` (app/brokers/kis/parser.py) coerces a missing KIS field to
            0.0, and `fetch_daily_short_sale` builds a ShortSaleDay whenever
            `stck_bsop_date` is present even if every other field is blank —
            so a not-yet-finalized current-day row arrives as all-zeros,
            indistinguishable from a real zero. Confirmed live (2026-09-01,
            267260.KS/HD현대일렉트릭, decision id=1156): the risk debate and
            the Portfolio Manager read the current day's 0.00% short ratio as
            "공매도 0% ⇒ 유동성 고갈" and made it the #1 rationale at
            confidence 90% — while, in the same packet, the Market Analyst
            said orders could execute normally and execution_status_context
            said "no halt / VI". Same module, same failure as `_pct_or_missing`
            above / 2026-08-26 decision id=785 (002990.KS `quck_rate`) — see
            [[debt-ratio-verified-quick-ratio-fix-2026-08-26]].

            Unlike current/quick ratio, a short-selling ratio of exactly 0.00%
            IS attainable (a day with zero short-sale executions — common for
            illiquid names or names on the short-sale ban list), so a blanket
            `0.0 -> missing` would be wrong here. Only two shapes are treated
            as missing: a row whose close is <= 0 (a listed name's close is
            never 0 — the whole row is unfilled), and a ratio that is exactly
            0.0 while `short_volume > 0` (arithmetically impossible: a nonzero
            short volume cannot be exactly 0% of the day's turnover, and the
            accumulated ratio cannot be below the day's own contribution). A
            0.0 ratio paired with `short_volume == 0` is a consistent real
            zero and is rendered as-is."""
            row_missing = s.close <= 0
            close = "missing" if row_missing else f"{s.close:,.0f}"
            vol = "missing" if row_missing else f"{s.short_volume:,}"

            def _pct(value: float) -> str:
                if row_missing or (s.short_volume > 0 and value == 0.0):
                    return "missing"
                return f"{value:.2f}"

            return (
                f"{close:>10}", f"{vol:>12}",
                f"{_pct(s.short_ratio):>9}", f"{_pct(s.accum_short_ratio):>13}",
            )

        header = f"{'Date':<10}{'Close':>10}{'ShortVol':>12}{'Short%':>9}{'AccumShort%':>13}"
        rows = [header]
        for s in shorts:
            c_close, c_vol, c_pct, c_accum = _short_cells(s)
            marker = "   <- today, not yet finalized" if s.day == today_str else ""
            rows.append(f"{s.day:<10}{c_close}{c_vol}{c_pct}{c_accum}{marker}")
        sections.append(
            "\n[Short-selling] (daily short volume, its % of the day's turnover, and accumulated %). "
            "`missing` means KIS did not report that field for that row — it is NOT zero. A row "
            "dated today may be partial/unfinalized (KRX publishes confirmed short-sale figures the "
            "next business day), so do not assert a 0 or a low value read from today's row as "
            "settled fact. A short-selling ratio is SELL-side pressure, not a liquidity or "
            "tradability signal — do not read a low/zero short ratio as 'liquidity has dried up' "
            "(for tradability, see execution_status_context and the Market Analyst).\n"
            + "\n".join(rows)
        )
    else:
        sections.append("\n[Short-selling] no recent short-sale data")

    if as_of is None or end_str == today_str:
        # Same bonus-TR policy as the short-sale fetch above.
        # 프로그램매매도 동일하게 보너스 TR — 실패해도 전체 분석은 계속 진행.
        try:
            program = await broker.fetch_program_trade(code)
        except Exception:
            logger.warning("KIS fetch_program_trade failed for %s", ticker, exc_info=True)
            program = []
        if program:
            # Corrected 2026-09-11: previously took program[0] on the assumption
            # that's the latest slot. fetch_program_trade builds ProgramTrade
            # with slot=bsop_hour (a zero-padded HHMMSS intraday time slot),
            # and KIS's row order for this TR is not documented or verified
            # anywhere in this repo — the sibling intraday-estimate path in
            # this module (get_kr_investor_flow, est[-1]) assumes the opposite
            # index, so the two can't both be right by construction. Selecting
            # by max slot string is correct regardless of which direction KIS
            # actually sorts. If all rows have an empty/missing slot, this
            # degrades to program[0], matching the pre-existing behavior.
            p = max(program, key=lambda t: t.slot)
            # net_buy_amount is raw KRW (not pre-divided like the KIS
            # financials fields elsewhere in this module, e.g. market_cap
            # already in 억원) — confirmed live (2026-07-30, 042660.KS) that
            # handing the model a raw won figure like +45,613,179,100 gets
            # it reported as "4,500억원" instead of the actual "456억원", a
            # 10x error, even though the number was already comma-formatted
            # with no scientific notation involved. Converting to 억원 here
            # (dividing by 1e8) matches this function's own existing
            # convention and removes that arithmetic step from the model.
            sections.append(
                f"\n[Program trading] latest net-buy: {p.net_buy_qty:+,} shares "
                f"({p.net_buy_amount / 1e8:+,.1f}억원)"
            )

    return "\n".join(sections)


# A money amount written the Korean way in a headline: digits (with an
# optional decimal / sign / comma group) then 억 or 조, optionally 원. The
# `(?<![\d.,])` guard stops it matching the tail of a larger number, and the
# leading digit requirement means "수억"/"억대"/"수십억" (no digits) never
# match. Percentages carry no 억/조 token, so a "매출액대비 19.56%" phrase is
# not a candidate.
_KR_HEADLINE_AMOUNT_RE = re.compile(r"(?<![\d.,])(-?\d+(?:,\d{3})*(?:\.\d+)?)\s*(억|조)\s*원?")


def _headline_won_gloss(eok: float) -> str:
    """`eok` in 억 units -> 'N million KRW (M billion KRW)', preserving the
    headline's own precision. Deliberately NOT `_eok` (which rounds to whole
    억 with `:,.0f` — it would turn a 6.6억 contract into '7억' and lose the
    very digit this gloss exists to get right)."""
    def _trim(x: float, dp: int) -> str:
        s = f"{x:,.{dp}f}"
        return s.rstrip("0").rstrip(".") if "." in s else s

    parts = [f"{_trim(eok * 100.0, 0)} million KRW", f"{_trim(eok / 10.0, 3)} billion KRW"]
    if abs(eok) >= 10_000.0:
        parts.append(f"{_trim(eok / 10_000.0, 3)} trillion KRW")
    return f"{parts[0]} ({', '.join(parts[1:])})"


def kr_amount_unit_key(news_text: str) -> str:
    """A trusted, mechanically-derived KRW unit key for the 억/조 money
    amounts written in `news_text` (a `get_kr_news` block). Empty string when
    the text names no such amount, so a caller that appends this leaves its
    prompt byte-identical otherwise.

    2026-09-07 domestic audit #3 (id=1237 / 294630.KS): a KIS headline —
    "...공급 계약 6.6억원 (매출액대비 19.56 %)" — was reported by the News +
    Macro Analyst as "a ₩6.6 billion supply contract" (6.6억원 is ₩0.66
    billion), and that 10x-inflated magnitude propagated through the whole
    desk, including a bear turn that made it "nearly five times its
    annualized revenue". Same 억원->billion carry-through `_eok` was built to
    remove for the Fundamentals block — but the News node never had an
    equivalent, because headlines arrive as raw third-party text.

    The block is meant to be appended by the caller AFTER the wrapped
    `<external_data>` news, as its OWN trusted line — it restates the
    headline's digits in another unit, it does not endorse the headline. The
    quoted amount is rebuilt from the parsed number + unit, never spliced
    from the source, so nothing an attacker writes reaches the block except a
    number."""
    seen: dict[float, str] = {}
    for m in _KR_HEADLINE_AMOUNT_RE.finditer(news_text):
        num = float(m.group(1).replace(",", ""))
        eok = num * (10_000.0 if m.group(2) == "조" else 1.0)
        if abs(eok) >= 1_000_000.0:  # >= 100조 in a headline — implausible, skip as noise
            continue
        seen.setdefault(eok, f"{m.group(1)}{m.group(2)}원")
    if not seen:
        return ""
    rows = sorted(seen.items(), key=lambda kv: abs(kv[0]), reverse=True)[:10]
    lines = "\n".join(f'  "{tok}" = {_headline_won_gloss(eok)}' for eok, tok in rows)
    return (
        "KRW UNIT KEY for the Korean headlines above (mechanical conversion of the 억/조 amounts "
        "written in those headlines — this restates their digits in another unit, it does NOT "
        "verify that any headline is true, and it is not a financial-statement figure):\n"
        f"{lines}\n"
        "Unit rule: 1억 = 100 million KRW = 0.1 billion KRW; 1조 = 1,000 billion KRW. An amount a "
        "Korean headline writes in 억 is a millions-scale figure unless it is at least 1,000억 — "
        'quote the converted value from this key; never carry an 억 amount\'s digits across into '
        'the word "billion".'
    )


async def get_kr_news(ticker: str, as_of: date | None = None) -> str:
    """종목 뉴스 — recent Korean-market headlines for a KRX ticker, filling
    yfinance's Korean-news gap. KRX-only; point-in-time filtered to on/before
    `as_of`.

    NOTE: the returned text contains third-party headlines — the classic
    prompt-injection surface. Callers feeding it to an LLM MUST wrap it with
    app.agents.untrusted_content.wrap_untrusted first (this provider stays
    agent-agnostic and does not wrap)."""
    if not (ticker.endswith(".KS") and ticker[:-3].isdigit()):
        return f"국내 뉴스는 한국(KRX) 종목만 제공됩니다 — {ticker} 미지원"
    try:
        broker = await _get_broker()
    except RuntimeError:
        return "국내 뉴스를 가져올 수 없습니다 (KIS 미설정)"

    code = ticker[:-3]
    cutoff = (as_of or get_clock().today()).strftime("%Y%m%d")
    # 2026-09-11: an undated headline (`n.day == ""`) is always the TR's newest
    # item, so keeping it unconditionally is fine for a live query (as_of is
    # None/today-or-later) but contaminates a backtest replay — it would be a
    # headline published AFTER the simulated `as_of` date fed into a historical
    # prompt as though it were already known. Only keep undated items when the
    # query is effectively live; live behaviour (as_of=None) is unchanged.
    keep_undated = as_of is None or as_of >= get_clock().today()
    items = [
        n for n in await broker.fetch_news_titles(code)
        if (n.day and n.day <= cutoff) or (not n.day and keep_undated)
    ][:15]
    if not items:
        return f"{ticker} 최근 뉴스 없음"
    lines = [
        f"[{n.day} {n.time}] ({n.source}) {n.title}" if n.day else f"({n.source}) {n.title}"
        for n in items
    ]
    return f"Recent Korean news headlines for {ticker} (source: KIS):\n" + "\n".join(lines)


async def get_overseas_news(ticker: str, as_of: date | None = None) -> str:
    """해외 종목 뉴스 (KIS가 한글로 번역해 제공) — `get_kr_news`의 해외판.
    US-ticker-shaped(1~5 uppercase letters) 심볼만 지원. 응답 envelope가
    `outblock1`이라는 것만 빼면 국내와 대칭(`fetch_overseas_news_titles`).
    TR 자체에 조회일자/페이지네이션 파라미터가 없어(항상 최신순 반환) 나머지와
    달리 KIS 서버 쪽 point-in-time 필터가 없다 — 반환된 항목의 day 필드로
    사후 필터만 가능.

    NOTE: 국내 뉴스와 마찬가지로 서드파티 원문이라 prompt-injection 표면 —
    LLM에 넣기 전 반드시 app.agents.untrusted_content.wrap_untrusted로
    감쌀 것(이 함수는 agent-agnostic이라 감싸지 않음)."""
    if not _is_us(ticker):
        return f"해외 뉴스는 미국(US) 종목만 제공됩니다 — {ticker} 미지원"
    try:
        broker = await _get_broker()
    except RuntimeError:
        return "해외 뉴스를 가져올 수 없습니다 (KIS 미설정)"

    cutoff = (as_of or get_clock().today()).strftime("%Y%m%d")
    # 2026-09-11: same undated-headline contamination guard as get_kr_news
    # above — an undated item is always the TR's newest, which is fine live
    # (as_of=None/today-or-later) but would leak future news into a backtest
    # replay's historical prompt. Live behaviour (as_of=None) is unchanged.
    keep_undated = as_of is None or as_of >= get_clock().today()
    items = [
        n for n in await broker.fetch_overseas_news_titles(ticker, exchange="NAS", nation="US")
        if (n.day and n.day <= cutoff) or (not n.day and keep_undated)
    ][:15]
    if not items:
        return f"{ticker} 최근 뉴스 없음"
    lines = [
        f"[{n.day} {n.time}] ({n.source}) {n.title}" if n.day else f"({n.source}) {n.title}"
        for n in items
    ]
    return f"Recent news headlines for {ticker} (source: KIS, Korean-translated):\n" + "\n".join(lines)


async def get_kr_disclosures(ticker: str, as_of: date | None = None) -> str:
    """예탁원 기업행위 (배당) 일정 — the KSD **cash-dividend schedule only**
    for a KRX ticker, which yfinance doesn't provide for Korean names.
    Structured data (dates/amounts), so trusted; KRX-only, point-in-time
    filtered.

    Despite the function name this is NOT a filings/공시 feed — KIS exposes
    no DART endpoint, and this returns dividend records and nothing else. An
    empty result therefore means "no dividend on the calendar", never "no
    disclosure / no filing / no contract". 2026-09-07 audit #6 (id=1237): a
    bare `record 20251231` line here was turned by the News + Macro Analyst
    into "no dividends for the fiscal year ending Dec 31, 2025" — a
    fiscal-year claim this feed cannot support. Both halves are labelled
    now, and news_macro_analyst's prompt says the block is a dividend
    schedule only."""
    if not (ticker.endswith(".KS") and ticker[:-3].isdigit()):
        return f"국내 기업행위(배당) 일정은 한국(KRX) 종목만 제공됩니다 — {ticker} 미지원"
    try:
        broker = await _get_broker()
    except RuntimeError:
        return "국내 기업행위 데이터를 가져올 수 없습니다 (KIS 미설정)"

    code = ticker[:-3]
    end = as_of or get_clock().today()
    start = end - timedelta(days=365)
    divs = await broker.fetch_dividend_schedule(
        code, start.strftime("%Y%m%d"), (end + timedelta(days=90)).strftime("%Y%m%d")
    )
    window = f"{start.strftime('%Y-%m-%d')}~{(end + timedelta(days=90)).strftime('%Y-%m-%d')}"
    if not divs:
        return (
            f"Corporate actions for {ticker} — CASH DIVIDEND SCHEDULE ONLY "
            f"(source: KIS/KSD; not a filings or disclosure feed), window {window}: "
            f"no dividend on the calendar."
        )
    lines = [
        f"record {d.record_date}: {d.kind} cash {d.cash_amount:,.0f} ({d.cash_rate:.2f}%), pay {d.pay_date}"
        for d in divs[:8]
    ]
    return (
        f"Corporate actions for {ticker} — CASH DIVIDEND SCHEDULE ONLY "
        f"(source: KIS/KSD; not a filings or disclosure feed), window {window}:\n"
        + "\n".join(lines)
    )


async def get_kr_macro_indicators(as_of: date | None = None) -> str:
    """국내 매크로 백드롭 — KOSPI/KOSDAQ 지수 현재가와 USD/KRW 환율.
    app.dataflows.fred_provider.get_macro_indicators(FRED, 미국 지표)의
    국내 대응물: News+Macro Analyst가 국내 종목을 분석할 때도 골라 쓸 수
    있는 실제 관련 있는 매크로 툴이 지금까지 하나도 없었음(연준 금리 등
    미국 지표뿐이었음) — 2026-08-03 논의 결과 추가.

    시장 전체 지표라 특정 티커에 종속되지 않음(get_kr_news/disclosures와
    달리 ticker 인자를 받지 않음) — news_macro_analyst.py가 .KS 종목
    분석일 때만 이 tool을 노출시켜 게이팅한다.

    지수 현재가 조회(fetch_index_price)는 오늘 시점 스냅샷만 제공하고
    과거 시점 조회를 지원하지 않으므로, get_kr_earnings_estimate와 같은
    이유로 backtest(as_of가 과거)에서는 제공하지 않는다."""
    if as_of is not None and as_of != get_clock().today():
        return "국내 매크로 지표(KOSPI/KOSDAQ/환율)는 과거 시점 백테스트에선 제공하지 않습니다."
    try:
        broker = await _get_broker()
    except RuntimeError:
        return "국내 매크로 지표를 가져올 수 없습니다 (KIS 미설정)"

    try:
        kospi, kosdaq = await asyncio.gather(
            broker.fetch_index_price("0001"), broker.fetch_index_price("1001")
        )
    except Exception as exc:  # noqa: BLE001 - degrade to partial/no data, never block the analyst
        return f"국내 지수 데이터를 가져오지 못했습니다: {type(exc).__name__}"

    try:
        usd_krw = await get_usd_fx_rate("KRW")
        fx_line = f"USD/KRW: {usd_krw:,.2f}"
    except Exception:  # noqa: BLE001 - FX is supplementary; index data alone is still useful
        fx_line = "USD/KRW: 조회 실패"

    return (
        "Macro indicators (KOSPI/KOSDAQ/FX, source: KIS/yfinance):\n"
        f"KOSPI: {kospi.value:,.2f} ({kospi.change_pct:+.2f}%)\n"
        f"KOSDAQ: {kosdaq.value:,.2f} ({kosdaq.change_pct:+.2f}%)\n"
        f"{fx_line}"
    )


# US index codes confirmed live against this TR (2026-08-04) — every other
# guessed variant (.SPX, .NDX, .IXIC, DJI without the dot, ...) silently
# returned an all-zero snapshot rather than an error. See
# OverseasInfoMixin.fetch_overseas_index_quote's own docstring.
_US_INDEX_CODES = (("Dow Jones", ".DJI"), ("S&P 500", "SPX"), ("Nasdaq 100", "NDX"))


async def get_overseas_market_indices(as_of: date | None = None) -> str:
    """미국 시장 전반 배경 — Dow/S&P500/Nasdaq100 현재가+등락률, KIS 전용
    (fetch_overseas_index_quote). app.dataflows.fred_provider.
    get_macro_indicators(FRED, 금리/실업률 등 거시경제지표)와는 목적이
    다른 별개 툴 — FRED는 미국 경제 전반의 상태를, 이건 국내의 KOSPI/
    KOSDAQ과 대칭으로 "지금 미국 주식시장 전체가 어느 방향인지"(시장
    breadth/추세)를 보여준다. news_macro_analyst.py가 해외 종목 분석일
    때 FRED와 나란히(대체가 아니라 추가로) 노출한다.

    지수 스냅샷은 조회 시점 기준만 제공하고 과거 시점 조회를 지원하지
    않으므로(get_kr_macro_indicators와 같은 이유), backtest(as_of가
    과거)에서는 제공하지 않는다."""
    if as_of is not None and as_of != get_clock().today():
        return "미국 주요 지수(Dow/S&P500/Nasdaq)는 과거 시점 백테스트에선 제공하지 않습니다."
    try:
        broker = await _get_broker()
    except RuntimeError:
        return "미국 주요 지수를 가져올 수 없습니다 (KIS 미설정)"

    today = get_clock().today()
    start = (today - timedelta(days=7)).strftime("%Y%m%d")
    end = today.strftime("%Y%m%d")
    try:
        quotes = await asyncio.gather(
            *(broker.fetch_overseas_index_quote(code, name, start, end) for name, code in _US_INDEX_CODES)
        )
    except Exception as exc:  # noqa: BLE001 - degrade to no data, never block the analyst
        return f"미국 주요 지수 데이터를 가져오지 못했습니다: {type(exc).__name__}"

    lines = [f"{q.name}: {q.value:,.2f} ({q.change_pct:+.2f}%)" for q in quotes]
    return "US market indices (Dow/S&P500/Nasdaq100, source: KIS):\n" + "\n".join(lines)


async def get_kr_earnings_estimate(ticker: str, as_of: date | None = None) -> str:
    """증권사 컨센서스 추정실적 (매출액/영업이익/순이익, 연도별 증가율 포함) —
    yfinance의 earnings_estimate는 미국 종목 중심이라 한국 종목 커버리지가
    약함. KRX-only. 컨센서스 추정치는 현재 시점 스냅샷이라 backtest(as_of가
    과거)에서는 제공하지 않는다 — get_kr_analyst_opinion의 목표주가와 같은
    이유."""
    if not (ticker.endswith(".KS") and ticker[:-3].isdigit()):
        return f"국내 추정실적(컨센서스) 데이터는 한국(KRX) 종목만 제공됩니다 — {ticker} 미지원"
    if as_of is not None and as_of != get_clock().today():
        return "국내 추정실적(컨센서스)은 과거 시점 백테스트에선 제공하지 않습니다."
    try:
        broker = await _get_broker()
    except RuntimeError:
        return "국내 추정실적 데이터를 가져올 수 없습니다 (KIS 미설정)"

    code = ticker[:-3]
    rows = await broker.fetch_kr_earnings_estimate(code)
    if not rows:
        return f"{ticker} 국내 추정실적(컨센서스) 데이터 없음"

    # 2026-09-11 unit-labeling fix (dataflows audit stage 1, item A): the old
    # fixed-width table's caption stated "억원 단위" but never supplied the
    # billion/trillion-KRW equivalents the Fundamentals Analyst prompt
    # promises are always given — a milder version of the `_kis_balance_sheet`
    # bug above (unit named, but no equivalent computed). Only revenue /
    # operating_profit / net_income are 억원 money and go through `_eok`; the
    # three *_growth columns are already percentages and are left as `:.2f`%
    # — do NOT wrap them in `_eok`, they are not 억원 figures.
    # 금액 3개 컬럼만 _eok 환산 추가, 증가율 3개 컬럼은 그대로 %로 유지.
    return (
        f"Consensus earnings estimates for {ticker} (source: KIS, {_EOK_TABLE_UNIT_KEY}, "
        "'E' suffix = estimated period):\n"
        + "\n".join(
            f"  {r.period}: revenue={_estimate_revenue_or_missing(r)} revenue_growth={r.revenue_growth:.2f}% "
            f"operating_profit={_eok_or_missing(r.operating_profit)} "
            f"operating_profit_growth={r.operating_profit_growth:.2f}% "
            f"net_income={_eok_or_missing(r.net_income)} net_income_growth={r.net_income_growth:.2f}%"
            for r in rows
        )
    )


async def get_kr_market_regime(as_of: date | None = None) -> str:
    """KOSPI/KOSDAQ current level and day change — the market-wide regime the
    analysts otherwise lack (they see only the single stock). Market-level, so
    not ticker-specific. The TR returns only the current index, so for a
    backtest date other than today this is skipped (would leak the future)."""
    if as_of is not None and as_of != get_clock().today():
        return "국내 시장지수(KOSPI/KOSDAQ) 실시간 값은 과거 시점 백테스트에선 제공하지 않습니다."
    try:
        broker = await _get_broker()
    except RuntimeError:
        return "국내 시장지수를 가져올 수 없습니다 (KIS 미설정)"
    lines = []
    for code in ("0001", "1001"):
        try:
            q = await broker.fetch_index_price(code)
            lines.append(f"{q.name}: {q.value:,.2f} ({q.change:+,.2f}, {q.change_pct:+.2f}%)")
        except Exception:  # noqa: BLE001
            continue
    if not lines:
        return "국내 시장지수 데이터 없음"
    return "Korean market regime (source: KIS):\n" + "\n".join(lines)


# fetch_sector_indices' TR bundles genuine KRX industry-sector indices in
# with a much larger set of derivative/thematic/commodity/bond index
# products (leveraged/inverse ETF-tracking indices, futures TWAP indices,
# Nikkei/Bloomberg/Solactive-branded commodity trackers, KEBI bond
# indices, WISE thematic indices, ...) — confirmed live, 2026-07-29, by
# dumping every (code, name) row for both markets. A simple leveraged/
# inverse keyword filter (app.domain.leverage_filter.is_leveraged_or_
# inverse) still let through VIX/commodity/Nikkei-futures/bond entries
# that dominated the top/bottom-5 ranking just as badly, since they're
# not leveraged by name even though they're not real economic sectors
# either. This allowlist is the actual KRX standard industry
# classification names observed live for both markets (KOSPI codes
# 0001-0030, KOSDAQ codes in the 1000s) — precise rather than trying to
# pattern-match an open-ended universe of derivative product names.
_KR_REAL_SECTOR_NAMES = frozenset({
    "종합", "대형주", "중형주", "소형주", "코스닥 소형주",
    "음식료·담배", "섬유·의류", "종이·목재", "화학", "제약", "비금속", "금속",
    "기계·장비", "전기·전자", "의료·정밀기기", "운송장비·부품", "유통", "전기·가스",
    "건설", "운송·창고", "통신", "금융", "증권", "보험", "일반서비스", "제조",
    "부동산", "IT 서비스", "오락·문화",
    # Standard-classification name the 2026-09 audit found fetch_sector_indices
    # returns legitimately (026040 -> '기타제조') but the allowlist dropped.
    "기타제조",
    # KOSDAQ leaf-sector name returned by FID_BLNG_CLS_CODE="3" (§섹터breadth
    # 2단계, 2026-09-09) — stable across every probed cycle but missing here.
    "출판·매체복제",
})


_SECTOR_BREADTH_FOOTER = (
    "(This ranking is market-wide. It does NOT tell you which sector the analyzed stock "
    'belongs to, and this system has no ticker->sector mapping beyond the single "This '
    'stock\'s own KRX industry sector" line above, if present. Do not state or imply that '
    "the analyzed stock belongs to, operates in, is a player in, is represented by, or is "
    "part of any sector named in this ranking, and do not call a sector's gain a tailwind "
    "for this stock on the basis of this ranking alone. Use it only to judge whether "
    "today's move is a broad market rotation or idiosyncratic.)"
)


async def get_kr_sector_breadth(as_of: date | None = None, ticker: str | None = None) -> str:
    """업종별 지수 등락 (sector breadth) — top/bottom-moving KOSPI+KOSDAQ
    sectors right now, so the Market Analyst can tell whether a stock's
    move is part of a broad sector rotation or an idiosyncratic one.
    Market-level like `get_kr_market_regime` (not ticker-specific — this
    codebase has no ticker->sector-code mapping yet, only a whole-market
    breadth view; `app/web/routers/market_data.py`'s `/api/sectors`
    endpoint is the only other caller of `fetch_sector_indices`, same
    unmapped shape). Live snapshot only TR, no historical query option —
    same "skip for a backtest date, don't leak today's values into the
    past" gating as `get_kr_market_regime`.

    Filtered to `_KR_REAL_SECTOR_NAMES` — see that constant's docstring
    for why an unfiltered ranking is actively misleading rather than just
    noisy (derivative index products move at 2-3x the real market and so
    dominate both ends of the ranking every single time)."""
    if as_of is not None and as_of != get_clock().today():
        return "국내 업종 지수 실시간 값은 과거 시점 백테스트에선 제공하지 않습니다."
    try:
        broker = await _get_broker()
    except RuntimeError:
        return "국내 업종 지수를 가져올 수 없습니다 (KIS 미설정)"
    sectors = []
    # Diagnostics only (2026-09-04) — the 28-vs-48 total swing means a whole
    # market's rows are non-deterministically absent and today nothing records
    # which market or why. No effect on the returned string.
    per_market_raw: dict[str, int] = {}
    per_market_filtered: dict[str, int] = {}
    for market in ("0001", "1001"):
        try:
            fetched = await broker.fetch_sector_indices(market)
        except Exception:
            logger.warning("KIS sector indices failed for market %s", market, exc_info=True)
            continue
        kept = [s for s in fetched if s.name in _KR_REAL_SECTOR_NAMES]
        per_market_raw[market] = len(fetched)
        per_market_filtered[market] = len(kept)
        if not kept:
            logger.warning(
                "sector breadth composition: market %s contributed ZERO rows after the "
                "_KR_REAL_SECTOR_NAMES filter (raw=%d) — this is the silent 28-vs-48 shortfall",
                market, len(fetched),
            )
        sectors.extend(fetched)
    sectors = [s for s in sectors if s.name in _KR_REAL_SECTOR_NAMES]
    _prefixes_by_name: dict[str, set] = {}
    for s in sectors:
        _prefixes_by_name.setdefault(s.name, set()).add((s.code or "")[:2])
    both_market_names = sum(1 for p in _prefixes_by_name.values() if "00" in p and "10" in p)
    # Dedupe by code (keep first) before ranking — fetch_sector_indices can
    # return the same industry twice when a continuation page re-serves an
    # overlapping chunk (see its own docstring).
    _seen_codes: set[str] = set()
    _deduped = []
    for s in sectors:
        if s.code in _seen_codes:
            continue
        _seen_codes.add(s.code)
        _deduped.append(s)
    sectors = _deduped
    if not sectors:
        return "국내 업종 지수 데이터 없음"
    ranked = sorted(sectors, key=lambda s: s.change_pct, reverse=True)
    logger.info(
        "sector breadth composition: kospi_raw=%s kospi_filtered=%s kosdaq_raw=%s "
        "kosdaq_filtered=%s names_in_both_markets=%d ranked=%d",
        per_market_raw.get("0001"), per_market_filtered.get("0001"),
        per_market_raw.get("1001"), per_market_filtered.get("1001"),
        both_market_names, len(ranked),
    )
    top = ranked[:5]
    bottom = ranked[-5:] if len(ranked) > 5 else []
    lines = [
        "Korean sector breadth (source: KIS, live snapshot) — a WHOLE-MARKET "
        "ranking of KRX industry indices, NOT the analyzed stock's own sector:"
    ]
    lines.append("Top gaining: " + ", ".join(f"{s.name} {s.change_pct:+.2f}%" for s in top))
    if bottom:
        lines.append("Top losing: " + ", ".join(f"{s.name} {s.change_pct:+.2f}%" for s in bottom))
    own_line = await _own_sector_line(ticker, ranked, per_market_filtered) if ticker else ""
    if own_line:
        lines.append(own_line)
    lines.append(_SECTOR_BREADTH_FOOTER)
    return "\n".join(lines)


async def _own_sector_line(
    ticker: str, ranked: list, per_market_filtered: dict[str, int] | None = None,
) -> str:
    """The single authoritative per-stock sector fact, injected AFTER the
    Top gaining/losing lines and BEFORE the footer. One of four verbatim
    forms (found & in ranking / found & not in ranking / this cycle's own
    market had zero coverage / undetermined). A fetch_sector_name failure
    must never fail the whole breadth call.

    2026-09-09: market verification now happens BEFORE any single-match
    short-circuit (previously a lone name match was trusted unconditionally,
    which — when KOSDAQ's per-cycle coverage collapses to a handful of
    sectors, see fetch_sector_indices's docstring — silently attributed the
    OTHER market's index move to this stock as "the authoritative sector",
    reproducing the 2026-09-03 앱튼 misattribution every time coverage
    dropped instead of only when a name collided across both markets."""
    code = ticker
    for suffix in (".KS", ".KQ"):
        if code.endswith(suffix):
            code = code[: -len(suffix)]
            break
    try:
        broker = await _get_broker()
        name, market = await broker.fetch_sector_name(code)
    except Exception:
        logger.warning("KIS fetch_sector_name failed for %s", ticker, exc_info=True)
        name, market = "", ""
    # Production values of rprs_mrkt_kor_name have never been observed — log the
    # raw string so the market-disambiguation branch below can be tuned.
    logger.info(
        "own sector resolve: ticker=%s sector_name=%r rprs_mrkt_kor_name=%r",
        ticker, name, market,
    )
    if not name:
        return (
            "This stock's own KRX industry sector could not be determined this cycle. "
            "Do not infer it from the ranking above."
        )
    # WHY: fetch_sector_name returns only an industry NAME, and _own_sector_line
    # matches on name alone. About 18 of the 30 _KR_REAL_SECTOR_NAMES are
    # published as an index in BOTH KOSPI (code 00…) and KOSDAQ (code 10…), so a
    # name-only match can pin this line — which declares itself "the
    # authoritative sector for the analyzed stock" — to the OTHER market's index
    # move. We disambiguate by the market string BEFORE trusting any match count,
    # and refuse to guess if we can't tell the market apart.
    # 업종명만으로 매칭하면 다른 시장 지수 등락을 이 종목 것으로 오귀속할 수 있어
    # (허용목록 30개 중 ~18개가 양시장 공통) — 시장 문자열로 먼저 좁히고 못 가리면 보류.
    matches = [s for s in ranked if s.name == name]
    market_lower = market.lower()
    if "kosdaq" in market_lower or "코스닥" in market:
        prefix, market_key = "10", "1001"
    elif "kospi" in market_lower or "코스피" in market:
        prefix, market_key = "00", "0001"
    else:
        prefix, market_key = None, None

    def _authoritative(match) -> str:
        rank = ranked.index(match) + 1
        return (
            f"This stock's own KRX industry sector: {name} ({market}), "
            f"{match.change_pct:+.2f}% today — ranked {rank} of {len(ranked)} in the ranking "
            f"above. This line, and only this line, is the authoritative sector for the "
            f"analyzed stock."
        )

    def _not_in_ranking() -> str:
        return (
            f"This stock's own KRX industry sector: {name} ({market}). Its index is NOT in the "
            f"ranking above, so no move is known for this stock's own sector — the ranking above "
            f"tells you nothing about it."
        )

    def _ambiguous(colliding) -> str:
        logger.warning(
            "own sector ambiguous: ticker=%s name=%r market=%r colliding_codes=%s",
            ticker, name, market, [s.code for s in colliding],
        )
        return (
            f"This stock's own KRX industry sector: {name} ({market}) is published as an index "
            f"in more than one market, and this cycle could not tell them apart. No move is "
            f"attributed to this stock's own sector, and the ranking above tells you nothing "
            f"about it."
        )

    if prefix is None:
        # Market string didn't resolve — can't verify against a code prefix, so
        # only trust an unambiguous single name match (same risk the old code
        # always took for every match count).
        if len(matches) == 1:
            return _authoritative(matches[0])
        if len(matches) == 0:
            return _not_in_ranking()
        return _ambiguous(matches)

    narrowed = [s for s in matches if (s.code or "").startswith(prefix)]
    if len(narrowed) == 1:
        return _authoritative(narrowed[0])
    if len(narrowed) == 0:
        if per_market_filtered is not None and per_market_filtered.get(market_key, 0) == 0:
            return (
                f"This stock's own KRX industry sector: {name} ({market}). This cycle's KIS "
                f"response had ZERO sector-index rows for this stock's own market (a known "
                f"intermittent KOSDAQ coverage gap), so no move is known for this stock's own "
                f"sector this cycle — do not infer it from the ranking above, and do not "
                f"substitute another market's index for it."
            )
        return _not_in_ranking()
    return _ambiguous(narrowed)


async def get_kr_sector_name(ticker: str, as_of: date | None = None) -> str:
    """The exchange-registered KRX industry classification for `ticker` alone
    (`fetch_sector_name`'s `bstp_kor_isnm`/`rprs_mrkt_kor_name`, same TR as
    fetch_price — no extra network cost class beyond what any KIS quote call
    already pays), WITHOUT the whole-market ranking `get_kr_sector_breadth`
    computes to produce its own `_own_sector_line` — this is for a caller
    that wants just the one authoritative per-stock fact, cheaply.

    2026-09-09, P1 audit item #3 (id=1354, 273060.KS/와이즈버즈): the
    Fundamentals Analyst has no access to this fact at all today, and
    inferred a business classification ("software/SaaS provider") purely
    from a 100% gross-margin/zero-COGS artifact of the company's OWN income
    statement (an advertising agency's KIS-reported cost-of-goods-sold
    happened to be reported as 0, with labor/service costs bucketed under
    SG&A instead — a reporting-classification quirk, not evidence of a
    software business model). Injecting the real KRX classification here
    gives that analyst something authoritative to check its own inference
    against, the same role `get_kr_sector_breadth`'s `_own_sector_line`
    already plays for the Market Analyst's sector-attribution claims.

    Live-only, same reasoning as `get_kr_sector_breadth`/
    `get_kr_macro_indicators`: `fetch_sector_name` is a live snapshot TR
    with no historical-query option, so a backtest date returns a note
    instead of a (potentially wrong, definitely un-auditable) live value.
    Best-effort: any fetch failure returns "" rather than raising, so a
    caller can simply skip the line when this is empty."""
    if as_of is not None and as_of != get_clock().today():
        return ""
    code = ticker
    for suffix in (".KS", ".KQ"):
        if code.endswith(suffix):
            code = code[: -len(suffix)]
            break
    try:
        broker = await _get_broker()
        name, market = await broker.fetch_sector_name(code)
    except Exception:
        logger.warning("get_kr_sector_name: fetch_sector_name failed for %s", ticker, exc_info=True)
        return ""
    if not name:
        return ""
    return (
        f"Exchange-registered KRX industry classification (authoritative, from KIS — "
        f"not inferred from financial ratios): {name} ({market})."
    )


# yfinance-only chains for the categories below were the last remaining path
# by which a domestic (.KS) ticker's data (fundamentals, analyst opinions,
# news, short-interest, balance sheet) still went through yfinance — real
# incidents this closes off: 218150.KS (yfinance quoteType MUTUALFUND, a
# broken shortName) and 064260.KS (yfinance's own longName resolving to a
# completely different, unrelated company, 에코프로비엠 instead of 다날) both
# happened on domestic tickers specifically. `_krx_code` raises for anything
# not KRX-shaped, which the VendorRouter treats as "try the next provider" —
# so a US ticker never even reaches these, same convention already
# established by `_kis_verified_snapshot`/`_kis_stock_data` above. Unlike
# get_last_close/get_atr_pct's inline try/except-then-yfinance pattern, these
# don't catch a KIS-side failure and retry yfinance themselves — a KIS
# outage surfaces as this category's own "미설정"/"없음" text rather than
# silently falling through to yfinance, which is the whole point: no
# domestic-ticker path through yfinance left unless the ticker genuinely
# isn't KRX-shaped.
async def _kis_fundamentals(ticker: str, as_of: date | None = None) -> str:
    _krx_code(ticker)  # raises for non-KRX -> VendorRouter tries yfinance next
    return await get_kr_financials(ticker, as_of=as_of)


async def _kis_analyst_consensus(ticker: str, as_of: date | None = None) -> str:
    _krx_code(ticker)
    return await get_kr_analyst_opinion(ticker, as_of=as_of)


async def _kis_news(ticker: str, as_of: date | None = None) -> str:
    """KRX tickers get domestic KIS news; US-ticker-shaped symbols get the
    overseas KIS news TR (Korean-translated headlines, confirmed working
    on a paper account) — see module docstring. Anything else raises so
    the VendorRouter falls through to yfinance."""
    if _is_krx(ticker):
        return await get_kr_news(ticker, as_of=as_of)
    if _is_us(ticker):
        return await get_overseas_news(ticker, as_of=as_of)
    raise ValueError(f"{ticker} is not a KRX or US ticker")


async def _kis_short_interest(ticker: str, as_of: date | None = None) -> str:
    _krx_code(ticker)
    return await get_kr_risk_signals(ticker, as_of=as_of)


async def _kis_indicators(ticker: str, as_of: date | None = None) -> str:
    """SMA(20)/SMA(50)/RSI(14)/MACD off KIS daily candles — same formulas and
    output format as `_yf_indicators` (both call the shared
    `app.dataflows.indicators.format_indicator_line`), so KRX tickers get
    technical signals computed from the exchange-official closes
    (yfinance's KRX closes were observed off by ~2%%, the same discrepancy
    noted elsewhere in this module) instead of yfinance's OHLCV. Also
    covers US-ticker-shaped symbols via the overseas candle TR — see
    module docstring.

    2026-09-08 fix (id=1315, 387690.KS/레메디): this used to require
    `len(candles) >= 51` — covering SMA(50)'s 50-bar need and MACD's
    signal-line warm-up (`slow + signal - 1` = 33 bars, see
    `app.dataflows.indicators.macd`'s own docstring) in one gate — and
    RAISE below that, discarding SMA(20)/RSI(14)/MACD too even though each
    needs far fewer bars (20/15/26). That incident: `get_atr_pct` (needs
    only 15 candles, `app.dataflows.indicators.atr_pct`'s own floor)
    succeeded (atr_pct=0.1198) on the SAME candle set this function threw
    away entirely, and the yfinance fallback (`_yf_indicators`) ALSO
    failed independently for this specific ticker (confirmed: yfinance
    carries zero rows for 387690.KS, "possibly delisted; no price data
    found" — a separate, likely-permanent gap on the fallback path this
    fix cannot address) — so the Market Analyst ended up with the
    all-caps NOT AVAILABLE fallback for all four indicators, when RSI(14)
    and SMA(20) were plausibly computable from the same KIS candles ATR
    used.

    The all-or-nothing framing was never necessary: `format_indicator_
    line`'s underlying math (`sma`/`rsi`/`macd` in this same package)
    already renders `n/a` per-field for whatever specific indicator lacks
    enough lookback (verified against raw `talib` calls on a short array —
    it returns NaN, never raises) — `_yf_indicators` already relies on
    exactly this graceful degradation with no floor at all. The guard now
    only rules out the case where NOTHING could render: 15 candles,
    matching RSI(14)'s own minimum (`app.dataflows.indicators.rsi`'s
    `window + 1`) — the smallest of the four requirements — and the same
    floor `get_atr_pct` already uses for the identical candle set.

    Deliberately NOT session-aware (unlike _kis_verified_snapshot/
    _kis_stock_data above) — see `_yf_indicators`'s identical note for the
    full reasoning: no per-row date label here for a model to misread as "the
    day closed," a live-reacting indicator matches what a real-time chart
    already shows, and marking this "in progress" while the verified
    snapshot's `last=` reflects the same current price would just trade one
    inconsistency for another."""
    end = as_of or get_clock().today()
    start = end - timedelta(days=_MAX_KIS_CANDLE_RANGE_DAYS)
    candles = await _fetch_kis_candles(ticker, start, end)
    if len(candles) < 15:  # RSI(14)'s own floor -- below this NOTHING here can render at all
        raise RuntimeError(f"KIS returned too few candles for {ticker} indicators")
    closes = pd.Series([c.close for c in candles])
    return format_indicator_line(ticker, closes, source_suffix="source: KIS")


async def _kis_balance_sheet(ticker: str, as_of: date | None = None) -> str:
    """Focused balance-sheet-only view (mirrors yfinance's own balance_sheet
    category, which is likewise not combined with income/ratio data) — a
    narrower slice of the same `fetch_kr_balance_sheet` call
    `get_kr_financials` already makes as part of its combined report.

    2026-09-11 unit-labeling fix (dataflows audit stage 1, item A): this used
    to render a fixed-width table of bare comma-grouped 억원 numbers with no
    unit token anywhere — the Fundamentals Analyst prompt
    (app/agents/fundamentals_analyst.py) promises every Korean figure is
    "supplied with its billion-KRW and trillion-KRW equivalents already
    computed", a promise `get_kr_financials`'s own [Balance sheet] section
    kept and this function alone did not. Switched to the same per-period
    `label=_eok(value)` line shape `get_kr_financials` already uses for the
    identical data: `_eok`'s output is a long multi-part string
    ("39,045억원(3.90조원) (= 3,904.5 billion KRW = 3.90 trillion KRW)"),
    fundamentally incompatible with fixed-width columns — and the table's
    alignment was never worth more than unit correctness, since
    `get_kr_financials` already made exactly this tradeoff for this same
    data.
    억원 단위/billion·조원 환산 없이 숫자만 내보내던 버그 수정 — get_kr_financials와
    동일한 _eok 렌더링 방식으로 통일."""
    code = _krx_code(ticker)
    broker = await _get_broker()
    cutoff = (as_of or get_clock().today()).strftime("%Y%m")
    rows = [r for r in await broker.fetch_kr_balance_sheet(code) if r.period <= cutoff][:4]
    if not rows:
        raise RuntimeError(f"KIS returned no balance sheet rows for {ticker}")
    return (
        f"Balance sheet for {ticker} (source: KIS, most recent {len(rows)} periods first) "
        f"({_EOK_TABLE_UNIT_KEY}):\n"
        + "\n".join(
            f"  {r.period}: total_assets={_eok_or_missing(r.total_assets)} "
            f"total_liabilities={_eok_or_missing(r.total_liabilities)} total_equity={_eok_or_missing(r.total_equity)} "
            f"current_assets={_eok_or_missing(r.current_assets)} current_liabilities={_eok_or_missing(r.current_liabilities)}"
            for r in rows
        )
    )


async def _kis_earnings_estimates(ticker: str, as_of: date | None = None) -> str:
    _krx_code(ticker)  # raises for non-KRX -> VendorRouter tries yfinance next
    return await get_kr_earnings_estimate(ticker, as_of=as_of)


router.register("verified_snapshot", [_kis_verified_snapshot, _yf_verified_snapshot])
router.register("core_stock", [_kis_stock_data, _yf_stock_data])
router.register("long_horizon", [_kis_long_horizon, _yf_long_horizon])
router.register("fundamental_data", [_kis_fundamentals, _yf_fundamentals])
router.register("analyst_consensus", [_kis_analyst_consensus, _yf_analyst_consensus], optional=True)
router.register("news_data", [_kis_news, _yf_news])
router.register("short_interest", [_kis_short_interest, _yf_short_interest], optional=True)
router.register("balance_sheet", [_kis_balance_sheet, _yf_balance_sheet])
router.register("technical_indicators", [_kis_indicators, _yf_indicators])
router.register("earnings_estimates", [_kis_earnings_estimates, _yf_earnings_estimates], optional=True)
# cashflow/insider_data/institutional_ownership: KIS has no equivalent TR at
# all (confirmed by scanning the full domestic + overseas API lists, and by
# a live real-account probe of the closest overseas candidate) — no KIS
# wrapper is registered for these, so they simply keep yfinance's own
# registration from app.dataflows.yfinance.research (both domestic and
# overseas tickers). Same yfinance-data-quality risk this module exists to
# avoid elsewhere (218150.KS/064260.KS-style misidentification) still
# applies here since there's no KIS data to cross-check against; accepted
# as an explicit tradeoff (some data > no data) rather than blocking output.
