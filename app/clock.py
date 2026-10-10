# app/clock.py
# The single point of wall-clock access — nowhere else should call
# datetime.now()/date.today() directly, so live/backtest/replay share
# identical code and tests can freeze time with a FakeClock.
# 유일한 실시각 접근점 — 다른 곳은 datetime.now()/date.today() 직접 호출 금지

from __future__ import annotations

from datetime import date, datetime, tzinfo
from typing import Literal, Protocol
from zoneinfo import ZoneInfo


class Clock(Protocol):
    def now(self, tz: tzinfo | None = None) -> datetime: ...
    def today(self) -> date: ...


class LiveClock:
    """Real wall-clock time — the default in every process."""

    def now(self, tz: tzinfo | None = None) -> datetime:
        return datetime.now(tz)

    def today(self) -> date:
        return date.today()


class FakeClock:
    """Fixed time for tests/backtests — never advances on its own.
    테스트/백테스트용 고정 시각 — 스스로 흐르지 않음"""

    def __init__(self, fixed: datetime | None = None) -> None:
        self._fixed = fixed or datetime(2000, 1, 1)

    def now(self, tz: tzinfo | None = None) -> datetime:
        return self._fixed.astimezone(tz) if tz is not None else self._fixed

    def today(self) -> date:
        return self._fixed.date()

    def set(self, when: datetime) -> None:
        self._fixed = when


_clock: Clock = LiveClock()


def get_clock() -> Clock:
    return _clock


def set_clock(new_clock: Clock) -> None:
    """Swap the process-wide clock — tests use this with a FakeClock, then
    restore LiveClock() (or the prior clock) afterward."""
    global _clock
    _clock = new_clock


# Duplicated (not imported) from app.execution.portfolio_risk.Market — that
# module already imports get_clock from here, so importing its Market back
# would be circular. Same two string values, so the two Literals are
# structurally interchangeable for type-checking purposes; nothing besides
# the type alias itself needs to be shared.
Market = Literal["domestic", "overseas"]

# Public (no underscore): app.brokers.kis.auth tags KIS's KST wall-clock
# token-expiry string with this so every comparison against get_clock().now()
# is tz-aware. Other modules that need Seoul time import it from here rather
# than re-deriving ZoneInfo("Asia/Seoul").
# 공개 이름 — KIS 토큰 만료(KST 벽시계) 태깅 등에서 공용으로 씀.
KR_TZ = ZoneInfo("Asia/Seoul")
_KR_TZ = KR_TZ  # backwards-compat alias for any older `from app.clock import _KR_TZ`
_US_TZ = ZoneInfo("America/New_York")


def local_today(market: Market) -> date:
    """The calendar date in THIS market's own timezone — KST for domestic,
    US Eastern for overseas. Both timezones' midnight falls outside that
    market's regular session (KRX 09:00-15:30 KST, NYSE/NASDAQ 9:30-16:00
    ET), so anything anchored to a market's own local_today (rather than a
    single shared get_clock().today()) always represents one uninterrupted
    trading day for that market, never split across an arbitrary KST-vs-ET
    boundary.

    Originally added 2026-08-04 as `_local_today` in app.execution.
    safety_switch (for the daily-loss baseline only); moved here 2026-08-05
    so app.live.auto_trade_loop's same-day dedup/rebalance-throttle and
    app.execution.kis_bridge's trade-date execution gate can share the
    identical definition instead of each comparing against plain KST —
    see project memory for the live incident (a GOOG/GOOGL/AVGO position
    bought right before KST midnight looked "not yet analyzed today" again
    minutes later, on the wrong side of a KST boundary that had nothing to
    do with the still-open NASDAQ session)."""
    return get_clock().now(KR_TZ if market == "domestic" else _US_TZ).date()
