"""KRX trading-day cache — the piece `app.live.market_hours.
latest_domestic_signal_card_reset_at` was missing to become holiday-aware (previously
weekend-aware only, see that function's own 2026-07-22 history).

Backed by a permanent per-date cache (once a date's open/closed status is
known it never changes, unlike app.execution.market_regime's per-CALENDAR-DAY
cache) — KIS's own docs ask that the underlying TR (chk-holiday) be called
at most once a day since it's tied to their ledger service, so caching
forever rather than daily is the safer side to err on. Best-effort: any
failure (KIS unreachable, TR rejected, DB down) degrades to "assume open"
— same as the pre-existing weekend-only behavior for every date this can't
resolve, and consistent with `latest_domestic_signal_card_reset_at`'s own docstring
that a stale boundary showing through a quiet holiday is harmless.

`is_market_open_on` above is fine for that cosmetic use — but it was never
safe to gate real trading on, and for a reason worse than "best-effort":
this project's paper account TR (chk-holiday) is confirmed to reject
EGW02006 unconditionally (see `_memory_cache`'s own comment below), so in
practice this function has *never once* returned a confirmed answer in this
deployment — `kr_market_holiday_cache` has zero rows. `is_kr_market_holiday`
below is the fix: a static table, same approach as `app.live.us_holidays`,
used directly by `app.live.market_hours.is_domestic_market_open` (the
actual trading gate) instead of the always-fails-open TR path.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import date

import asyncpg

from app.memory.db import get_pool

logger = logging.getLogger(__name__)

# Static KRX market-holiday table — unlike app.live.us_holidays, this can't
# be a computable rule: 설날/추석/부처님오신날 are lunar-calendar dates, and
# 대체공휴일 (substitute-holiday) assignment depends on which weekday the
# underlying holiday(s) land on in a given year. Cross-checked against two
# independent sources (calendarlabs.com/krx-market-holidays-YYYY,
# kstockguide.com/holidays) plus direct date/weekday arithmetic for the
# substitute-day placements, 2026-08-08. Includes the recurring year-end
# 폐장일 (Dec 31, KRX's own practice of closing the last trading day of the
# year and reopening late on Jan 2) alongside the government-designated
# 공휴일. Does NOT and cannot include ad-hoc 임시공휴일 (a one-off holiday
# the government can declare for an election or a special occasion, e.g.
# 2025's Jan 27 extended-Seollal designation, added below because it already
# happened, but a *future* one is unknowable in advance by definition — no
# static table can cover this, same limitation any such table has).
_KR_MARKET_HOLIDAYS: dict[int, frozenset[date]] = {
    2025: frozenset({
        date(2025, 1, 1),   # 신정
        date(2025, 1, 27),  # 임시공휴일 (설날 연휴 연장 지정)
        date(2025, 1, 28),  # 설날 전날
        date(2025, 1, 29),  # 설날
        date(2025, 1, 30),  # 설날 다음날
        date(2025, 3, 3),   # 삼일절 대체 (3/1이 토요일)
        date(2025, 5, 1),   # 근로자의 날
        date(2025, 5, 5),   # 어린이날/부처님오신날 (겹침)
        date(2025, 5, 6),   # 어린이날·부처님오신날 대체
        date(2025, 6, 6),   # 현충일
        date(2025, 8, 15),  # 광복절
        date(2025, 10, 3),  # 개천절
        date(2025, 10, 6),  # 추석 전날
        date(2025, 10, 7),  # 추석
        date(2025, 10, 8),  # 추석 다음날
        date(2025, 10, 9),  # 한글날
        date(2025, 12, 25),  # 성탄절
        date(2025, 12, 31),  # 연말 휴장(폐장일)
    }),
    2026: frozenset({
        date(2026, 1, 1),   # 신정
        date(2026, 2, 16),  # 설날 전날
        date(2026, 2, 17),  # 설날
        date(2026, 2, 18),  # 설날 다음날
        date(2026, 3, 2),   # 삼일절 대체 (3/1이 일요일)
        date(2026, 5, 1),   # 근로자의 날
        date(2026, 5, 5),   # 어린이날
        date(2026, 5, 25),  # 부처님오신날 대체 (5/24가 일요일)
        date(2026, 8, 17),  # 광복절 대체 (8/15가 토요일)
        date(2026, 9, 24),  # 추석 전날
        date(2026, 9, 25),  # 추석
        date(2026, 10, 5),  # 개천절 대체 (10/3이 토요일)
        date(2026, 10, 9),  # 한글날
        date(2026, 12, 25),  # 성탄절
        date(2026, 12, 31),  # 연말 휴장(폐장일)
    }),
    2027: frozenset({
        date(2027, 1, 1),   # 신정
        date(2027, 2, 8),   # 설날 연휴 대체 (설날 자체가 토/일에 걸침)
        date(2027, 3, 1),   # 삼일절
        date(2027, 5, 5),   # 어린이날
        date(2027, 5, 13),  # 부처님오신날
        date(2027, 9, 14),  # 추석 전날
        date(2027, 9, 15),  # 추석
        date(2027, 9, 16),  # 추석 다음날
        date(2027, 12, 31),  # 연말 휴장(폐장일)
        # 8/15(광복절)·10/3(개천절)·10/9(한글날)·12/25(성탄절)는 2027년에
        # 주말과 겹치는데, 대체공휴일 지정 여부를 소스 간 확인하지 못해
        # 이 표에는 넣지 않음 — 이미 주말 체크로 걸러지는 날짜라 실질적
        # 위험은 낮지만, 정확한 대체공휴일이면 이 표가 놓치는 셈이니
        # 2027년이 가까워지면 재확인 필요.
    }),
}

# 이 표가 다루는 연도를 벗어나면 예전처럼 "모름 → 거래일로 간주"로
# 폴백 — 표를 매년 갱신하지 않으면 그 해부터는 다시 예전과 같은
# 공휴일 무시 상태가 됨. 연도당 한 번만 경고해서 로그 폭탄을 피함.
_warned_years: set[int] = set()


def is_kr_market_holiday(day: date) -> bool:
    """Whether `day` is a KRX market holiday. Weekends are not included
    here — same convention as `is_market_open_on`/`app.live.us_holidays.
    is_us_market_holiday`, whose weekend check lives in the caller
    (`app.live.market_hours`).

    Falls back to `False` (not a holiday — "assume open", the same
    direction this whole module already degraded in before this table
    existed) for any year the static table doesn't cover, logging a
    warning once per missing year rather than raising or silently staying
    wrong forever. This can't be fixed by falling back to `is_market_open_
    on`'s KIS TR either — that path is confirmed to never return a real
    answer on this account (see the module docstring)."""
    if day.year not in _KR_MARKET_HOLIDAYS:
        if day.year not in _warned_years:
            _warned_years.add(day.year)
            logger.warning(
                "kr_holidays: no static holiday table for %d — treating every weekday in %d as a "
                "trading day until this table is extended (see app.live.kr_holidays._KR_MARKET_HOLIDAYS)",
                day.year, day.year,
            )
        return False
    return day in _KR_MARKET_HOLIDAYS[day.year]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS kr_market_holiday_cache (
    day        DATE PRIMARY KEY,
    is_open    BOOLEAN NOT NULL,
    checked_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""

_schema_ready = False
_schema_lock = asyncio.Lock()
# In-memory fast path — avoids a DB round-trip for a date already resolved
# earlier in this same process (the common case: the reset-boundary loop
# re-checks "today" on every dashboard poll).
_memory_cache: dict[date, bool] = {}


async def _ready_pool() -> asyncpg.Pool:
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


async def is_market_open_on(day: date) -> bool:
    """Whether KRX actually traded on `day` — memory cache, then DB cache,
    then (at most once ever per date) the live KIS TR. Degrades to True
    (assume open) if every layer fails, so a caller looping backwards
    through candidate dates (see latest_domestic_signal_card_reset_at) never spins
    forever on a date it can't resolve."""
    if day in _memory_cache:
        return _memory_cache[day]

    try:
        pool = await _ready_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow("SELECT is_open FROM kr_market_holiday_cache WHERE day = $1", day)
        if row is not None:
            _memory_cache[day] = row["is_open"]
            return row["is_open"]
    except Exception:
        logger.warning("kr_holidays: could not read cache for %s, falling through to a live check", day, exc_info=True)

    from app.web import state  # local import — avoid circular import, same reasoning as market_regime.py

    broker = await state.kis_broker()

    if broker is None:
        logger.info("kr_holidays: KIS broker not connected — assuming %s is a trading day", day)
        return True
    try:
        is_open = await broker.fetch_is_market_open_on(day.strftime("%Y%m%d"))
    except Exception as exc:
        # Not persisted to kr_market_holiday_cache — that table means
        # "confirmed by KIS", and a paper account's TR rejection isn't a
        # confirmed answer, just an assumption. Memory-only, so a process
        # restart gets one fresh attempt (cheap) instead of baking an
        # unconfirmed guess into the permanent record. Still worth caching
        # in-memory: a paper account's own docs confirm this TR rejects
        # unconditionally (EGW02006), so without this every poll of
        # latest_domestic_signal_card_reset_at re-hit KIS and re-logged this same
        # traceback for the rest of the process's life — confirmed live
        # 2026-08-04 (4 identical tracebacks in the first 20 minutes).
        # No exc_info (2026-08-06) — a paper account rejects this TR
        # unconditionally, so the traceback is always the exact same
        # KISBrokerError shape; the message already names it.
        logger.warning("kr_holidays: holiday check failed for %s — assuming it's a trading day: %s", day, exc)
        _memory_cache[day] = True
        return True

    _memory_cache[day] = is_open
    try:
        pool = await _ready_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO kr_market_holiday_cache (day, is_open) VALUES ($1, $2) "
                "ON CONFLICT (day) DO UPDATE SET is_open = $2, checked_at = now()",
                day, is_open,
            )
    except Exception:
        logger.warning("kr_holidays: could not persist cache entry for %s", day, exc_info=True)
    return is_open
