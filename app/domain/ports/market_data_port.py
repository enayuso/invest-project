# app/domain/ports/market_data_port.py
# Quote/candle/ranking contract — the KIS-specific fundamentals/news/investor-
# flow methods stay out of this Protocol: they're a KIS value-add, not
# something every future broker is expected to offer.
# 시세/캔들/순위 계약 (KIS 전용 부가 데이터는 이 계약에 포함하지 않음)

from __future__ import annotations

from typing import Protocol, runtime_checkable

from app.domain.models import Candle, IndexQuote, MinuteCandle, RankedStock, SectorIndex, StockStatus


@runtime_checkable
class MarketDataPort(Protocol):
    """Price/candle/ranking lookups, domestic and overseas.
    국내/해외 시세·캔들·순위 조회 계약"""

    async def fetch_price(self, code: str) -> float: ...

    async def fetch_stock_status(self, code: str) -> StockStatus: ...

    async def fetch_daily_candles(
        self, code: str, start: str, end: str, period: str = "D"
    ) -> list[Candle]: ...

    async def fetch_minute_candles(self, code: str, end_hour: str = "") -> list[MinuteCandle]: ...

    async def fetch_index_price(self, code: str = "0001") -> IndexQuote: ...

    async def fetch_volume_rank(
        self, count: int = 30, *, max_price: float | None = None
    ) -> list[RankedStock]: ...

    async def fetch_trading_value_rank(
        self, count: int = 30, *, max_price: float | None = None
    ) -> list[RankedStock]: ...

    # 시가총액 상위 — a stable large-cap discovery axis, distinct from the
    # turnover-driven feeds above. Live since 2026-07-30 but missing from
    # this Protocol until 2026-09-02 (screening-improvement-research #3
    # follow-up, price-ceiling work) — a drift fixed here rather than
    # carried forward, since every other rank method on this port has one.
    async def fetch_market_cap_rank(
        self, count: int = 30, *, max_price: float | None = None
    ) -> list[RankedStock]: ...

    # `min_volume` (added 2026-09-02, screening-improvement-research #3)
    # was also missing from this Protocol until now — same drift as
    # fetch_market_cap_rank's absence above, fixed in the same pass.
    async def fetch_fluctuation_rank(
        self, rising: bool = True, count: int = 30, *, min_volume: int = 0, max_price: float | None = None
    ) -> list[RankedStock]: ...

    # 이격도 순위 — price as a % of its own N-day moving average (a third,
    # size-free discovery axis; see app.brokers.kis.domestic.ranking's
    # fetch_disparity_rank for the full rationale).
    async def fetch_disparity_rank(
        self, low: bool = True, window_days: int = 20, count: int = 30, *,
        min_volume: int = 0, max_price: float | None = None,
    ) -> list[RankedStock]: ...

    async def fetch_sector_indices(self, market: str = "0001") -> list[SectorIndex]: ...

    async def fetch_overseas_price(self, symbol: str, exchange: str = "NAS") -> float: ...

    async def fetch_overseas_trading_value_rank(
        self, exchange: str = "NAS", count: int = 30
    ) -> list[RankedStock]: ...

    async def fetch_overseas_candles(
        self, symbol: str, end: str = "", exchange: str = "NAS", period: str = "D"
    ) -> list[Candle]: ...
