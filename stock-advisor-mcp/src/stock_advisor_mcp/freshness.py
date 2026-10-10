"""Freshness guard (spec S:3-2): "ask the market, not a holiday calendar".

Step-1 measurement (see the spec's S:0 table) found that a wrong
yfinance suffix does not fail loudly -- it silently returns stale data
(prices frozen at the previous trading day, fundamentals all None, a
garbled name). A model handed that data has no way to know it is stale
and will confidently analyze days-old prices as if they were current.

The fix does not need a holiday calendar: a reference ticker
(005930.KS, chosen because Samsung Electronics is about as close to
"the market is definitely open when KRX is open" as KRX has) tells us
what "the latest trading day" is, and any other ticker whose last bar is
older than that is stale by definition, regardless of weekends or
holidays.
"""

from __future__ import annotations

from datetime import date

from .errors import AdvisorError

REFERENCE_SYMBOL = "005930.KS"


class StaleDataError(AdvisorError):
    """Raised when a ticker's most recent bar is older than the reference
    ticker's latest trading day. Per spec S:1 principle 3, this must be a
    loud error, never a silent pass-through or a warning field."""


class NoDataError(AdvisorError):
    """Raised when yfinance returns zero rows / no usable quote data for a
    symbol. This is spec S:2-3's "0행 반환 -> 에러" rule, generalized to
    every tool that reads price data."""


class _ReferenceCache:
    """Caches the reference ticker's latest trading day for the life of
    the process (spec S:3-2: "기준 종목 조회 결과는 프로세스 수명 동안 캐시")."""

    def __init__(self) -> None:
        self._latest_trading_day: date | None = None

    def get(self, *, history_fetcher) -> date:
        if self._latest_trading_day is None:
            bars = history_fetcher(REFERENCE_SYMBOL)
            if bars is None or len(bars) == 0:
                raise NoDataError(
                    f"Reference symbol {REFERENCE_SYMBOL} returned no data -- cannot "
                    "establish the latest trading day. Refusing to guess."
                )
            self._latest_trading_day = bars.index[-1].date()
        return self._latest_trading_day

    def reset(self) -> None:
        """Test-only: clear the cached reference day."""
        self._latest_trading_day = None


_cache = _ReferenceCache()


def get_latest_trading_day(history_fetcher) -> date:
    """`history_fetcher(symbol) -> DataFrame-like with a DatetimeIndex`,
    injected so this module and its tests never import yfinance directly."""
    return _cache.get(history_fetcher=history_fetcher)


def assert_fresh(last_bar_date: date, latest_trading_day: date, *, symbol: str) -> None:
    if last_bar_date < latest_trading_day:
        raise StaleDataError(
            f"{symbol}: last available bar is {last_bar_date.isoformat()}, but the market's "
            f"latest trading day (per reference {REFERENCE_SYMBOL}) is "
            f"{latest_trading_day.isoformat()}. This usually means the wrong exchange suffix "
            "was used (.KS vs .KQ) and yfinance silently returned stale/cross-market data. "
            "Refusing to serve it."
        )


def reset_reference_cache_for_tests() -> None:
    _cache.reset()
