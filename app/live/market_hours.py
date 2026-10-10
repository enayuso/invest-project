"""Regular-session open/closed status per ticker — pure wall-clock math
plus a static holiday-calendar check, no extra API call.

Holiday-aware since 2026-08-08 (was weekday + time-of-day only before —
"NOT holiday-aware... acceptable for a trade allowlist status badge, not
for anything that gates a real trade" was this module's own prior warning,
while `app.live.auto_trade_loop` used it as exactly that gate). Confirmed
live: `is_domestic_market_open()`/`is_overseas_market_open()` answered
"open" on every 2026 KRX/NYSE holiday checked (성탄절, 한글날, 삼일절,
Thanksgiving, Good Friday). Does not model early closes or KRX half-days
(e.g. 수능일 late open) — see `app.live.us_holidays`'s own docstring for
the same limitation on the US side, and `app.live.kr_holidays`'s static
table for what it does/doesn't cover on the domestic side.

`latest_domestic_signal_card_reset_at`/`latest_overseas_signal_card_reset_at`
below have their own, separate holiday-awareness (via
`app.live.kr_holidays.is_market_open_on`, the async KIS-TR-backed path, and
the static `app.live.us_holidays` table respectively) — the domestic one
degrading to "assume open" on a lookup failure is harmless for a cosmetic
reset boundary, unlike the trading gate above.
"""

from __future__ import annotations

from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

from app.clock import get_clock
from app.live.kr_holidays import is_kr_market_holiday, is_market_open_on
from app.live.us_holidays import is_us_market_holiday

_US_TZ = ZoneInfo("America/New_York")
_US_OPEN, _US_CLOSE = time(9, 30), time(16, 0)

_KR_TZ = ZoneInfo("Asia/Seoul")
_KR_OPEN, _KR_CLOSE = time(9, 0), time(15, 30)

_SIGNAL_CARD_RESET_LEAD_MINUTES = 30


def is_market_open(ticker: str) -> bool:
    if ticker.endswith(".KS"):
        tz, open_t, close_t = _KR_TZ, _KR_OPEN, _KR_CLOSE
    else:
        tz, open_t, close_t = _US_TZ, _US_OPEN, _US_CLOSE

    now = get_clock().now(tz)
    if now.weekday() >= 5:  # Saturday/Sunday
        return False
    today = now.date()
    if ticker.endswith(".KS"):
        if is_kr_market_holiday(today):
            return False
    elif is_us_market_holiday(today):
        return False
    return open_t <= now.time() <= close_t


def is_domestic_market_open() -> bool:
    """KRX regular session — a market-level check (not per-ticker), for
    callers that decide whether to touch the domestic side at all (e.g.
    app.live.auto_trade_loop) rather than checking one specific ticker."""
    return is_market_open("KRX.KS")


def is_overseas_market_open() -> bool:
    """US regular session — same as is_domestic_market_open, for the
    overseas side."""
    return is_market_open("US")


async def _kr_trading_day(day) -> bool:
    # A thin wrapper, not a direct reference to is_market_open_on, so that
    # tests' monkeypatch.setattr(market_hours, "is_market_open_on", ...)
    # keeps working — looked up on the module at CALL time, not bound into
    # a default arg or closure at import time.
    return await is_market_open_on(day)


async def _us_trading_day(day) -> bool:
    return not is_us_market_holiday(day)  # static table, no I/O — always resolves


async def _latest_reset_boundary(tz: ZoneInfo, open_t: time, trading_day) -> datetime:
    now = get_clock().now(tz)
    reset_at = now.replace(hour=open_t.hour, minute=open_t.minute, second=0, microsecond=0)
    reset_at -= timedelta(minutes=_SIGNAL_CARD_RESET_LEAD_MINUTES)
    if now < reset_at:
        reset_at -= timedelta(days=1)
    while reset_at.weekday() >= 5 or not await trading_day(reset_at.date()):
        reset_at -= timedelta(days=1)
    return reset_at


async def latest_domestic_signal_card_reset_at() -> datetime:
    """The most recent KRX-open-minus-30-minutes boundary that has already
    passed, in KST — the dashboard's AI 시그널 card history clears at this
    point each day for DOMESTIC decisions (app.web.routers.live exposes it,
    the frontend drops any domestic decision older than it) rather than
    accumulating forever or resetting at an arbitrary midnight.

    Weekend-aware (rolls back to the prior weekday) since 2026-07-22 — a
    Monday morning before the open used to land on Sunday 08:30 instead of
    the real prior session, Friday 08:30. Public-holiday-aware since
    2026-08-04 too (설날/추석/신정 etc., via app.live.kr_holidays) — a stale
    boundary showing through a quiet holiday was harmless either way
    (nothing new to clear anyway), but now resolves to the real prior
    session instead. is_market_open_on degrades to "assume open" on any
    failure, so this can never loop indefinitely or block on a flaky KIS
    call — same degrade-quietly contract as the rest of this module.
    주말+공휴일 모두 이전 실제 거래일로 되돌아가도록 처리.

    Until 2026-08-20 this was a single value shared by both domestic and
    overseas signal cards, rejecting a candidate date if EITHER side's
    market was closed that day — that meant 5 US decisions filed
    2026-08-19 22:46-22:58 KST (well within the still-open US session)
    vanished from the dashboard the moment this boundary crossed
    2026-08-20 08:30 KST, hours before the US market had even opened. See
    latest_overseas_signal_card_reset_at for the now-separate US boundary;
    this function no longer looks at NYSE holidays at all."""
    return await _latest_reset_boundary(_KR_TZ, _KR_OPEN, _kr_trading_day)


async def latest_overseas_signal_card_reset_at() -> datetime:
    """US-regular-open-minus-30-minutes boundary that has already passed,
    in ET (NOT converted to KST — a caller/serializer should emit this
    ET-offset instant as-is; comparing instants doesn't require a shared
    timezone). Added 2026-08-20 as the overseas counterpart to
    latest_domestic_signal_card_reset_at — see that function's own
    docstring for the incident this split fixes.

    Weekend/NYSE-holiday-aware via the static app.live.us_holidays table
    (no KIS/DB round-trip, so this can never block or degrade). DST is
    handled implicitly: the wall-clock arithmetic above operates on a
    ZoneInfo-aware datetime, so "09:00 ET" always means the correct UTC
    instant on either side of a DST transition — 09:00 ET is never
    ambiguous or nonexistent since US transitions happen at 02:00 local."""
    return await _latest_reset_boundary(_US_TZ, _US_OPEN, _us_trading_day)
