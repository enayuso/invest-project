"""Broker abstraction verification (agent-architecture-roadmap stage 3) — no
new broker is actually integrated here; this proves the STRUCTURE already
supports one: a completely independent, from-scratch broker implementation
(no KISBroker inheritance, no KIS models) satisfies BrokerPort and
MarketDataPort, and the trading-critical safety-gate chain in
app.execution.kis_bridge works against it unchanged.
"""

from __future__ import annotations

from datetime import date

import pytest

from app.brokers.kis import KISBroker
from app.config import settings
from app.domain.models import (
    BrokerBalance,
    BrokerExecution,
    BrokerPosition,
    BuyableInfo,
    Candle,
    IndexQuote,
    MinuteCandle,
    OrderResult,
    OverseasCash,
    RankedStock,
    SectorIndex,
    SellableInfo,
    StockStatus,
)
from app.clock import local_today
from app.domain.ports import BrokerPort, MarketDataPort
from app.execution.kis_bridge import mirror_trade_to_kis, place_manual_order

TODAY = date.today()
# mirror_trade_to_kis's trade-date gate checks each market's own local date
# — an AAPL (overseas) call must use the US-Eastern-local date, not KST's,
# to look like "today" rather than a historical replay (see test_kis_bridge
# .py's TODAY_OVERSEAS for the same reasoning).
TODAY_OVERSEAS = local_today("overseas")


class IndependentSecondBroker:
    """A stand-in for a genuinely different second brokerage's adapter —
    deliberately shares no code, base class, or model with KISBroker. Only
    exists to prove BrokerPort/MarketDataPort are satisfiable by more than
    one implementation, and that kis_bridge.py's gate chain doesn't secretly
    depend on anything KIS-specific beyond these two contracts."""

    def __init__(self, held: int = 100):
        self.held = held
        self.orders: list[tuple] = []
        self.overseas_orders: list[tuple] = []
        self.closed = False

    # ---- BrokerPort ----
    async def fetch_balance(self) -> BrokerBalance:
        positions = (
            [BrokerPosition(
                code="005930", name="테스트전자", quantity=self.held,
                avg_price=0.0, current_price=0.0, eval_amount=0.0, pnl_amount=0.0, pnl_pct=0.0,
            )]
            if self.held
            else []
        )
        return BrokerBalance(
            cash_deposit=0.0, total_eval_amount=0.0, total_pnl_amount=0.0, purchase_amount=0.0, positions=positions,
        )

    async def fetch_buyable(self, code: str, price: float | None = None) -> BuyableInfo:
        return BuyableInfo(cash=1_000_000.0, max_qty=100, max_amount=1_000_000.0)

    async def fetch_sellable(self, code: str) -> SellableInfo:
        return SellableInfo(sellable_qty=self.held, holding_qty=self.held, already_sold_qty=0)

    async def submit_order(self, code: str, side: str, quantity: int, price: float | None = None) -> OrderResult:
        self.orders.append((code, side, quantity, price))
        return OrderResult(order_no="SB-0001", order_time="090000", message="ok", org_no="SB-ORG")

    async def cancel_order(
        self, order_no: str, org_no: str, quantity: int, price: float | None = None, ord_dvsn: str = "00"
    ) -> OrderResult:
        return OrderResult(order_no=order_no, order_time="090001", message="cancelled")

    async def revise_order(
        self, order_no: str, org_no: str, quantity: int, new_price: float, ord_dvsn: str = "00"
    ) -> OrderResult:
        return OrderResult(order_no=order_no, order_time="090002", message="revised")

    async def fetch_daily_executions(self, start: str, end: str) -> list[BrokerExecution]:
        return []

    async def fetch_overseas_balance(self, exchange: str = "NASD", currency: str = "USD") -> BrokerBalance:
        return BrokerBalance(cash_deposit=0.0, total_eval_amount=0.0, total_pnl_amount=0.0, purchase_amount=0.0)

    async def submit_overseas_order(
        self, code: str, side: str, quantity: int, price: float, exchange: str = "NASD"
    ) -> OrderResult:
        self.overseas_orders.append((code, side, quantity, price, exchange))
        return OrderResult(order_no="SB-2001", order_time="223000", message="ok")

    async def cancel_overseas_order(
        self, order_no: str, code: str, quantity: int, exchange: str = "NASD"
    ) -> OrderResult:
        return OrderResult(order_no=order_no, order_time="223001", message="cancelled")

    async def fetch_overseas_executions(self, start: str, end: str) -> list[BrokerExecution]:
        return []

    async def fetch_overseas_present_balance(self, currency: str = "USD") -> OverseasCash:
        return OverseasCash(currency=currency, deposit=1_000_000.0, orderable=1_000_000.0, eval_amount=0.0)

    # ---- MarketDataPort ----
    async def fetch_price(self, code: str) -> float:
        return 100.0

    async def fetch_stock_status(self, code: str) -> StockStatus:
        return StockStatus(code=code, trading_halted=False, administrative_issue=False, nxt_trading_halted=False)

    async def fetch_daily_candles(self, code: str, start: str, end: str, period: str = "D") -> list[Candle]:
        return []

    async def fetch_minute_candles(self, code: str, end_hour: str = "") -> list[MinuteCandle]:
        return []

    async def fetch_index_price(self, code: str = "0001") -> IndexQuote:
        return IndexQuote(code=code, name="TEST INDEX", value=1000.0, change=0.0, change_pct=0.0)

    async def fetch_volume_rank(self, count: int = 30, *, max_price: float | None = None) -> list[RankedStock]:
        return []

    async def fetch_trading_value_rank(
        self, count: int = 30, *, max_price: float | None = None
    ) -> list[RankedStock]:
        return []

    async def fetch_market_cap_rank(self, count: int = 30, *, max_price: float | None = None) -> list[RankedStock]:
        return []

    async def fetch_fluctuation_rank(
        self, rising: bool = True, count: int = 30, *, min_volume: int = 0, max_price: float | None = None
    ) -> list[RankedStock]:
        return []

    async def fetch_disparity_rank(
        self, low: bool = True, window_days: int = 20, count: int = 30, *,
        min_volume: int = 0, max_price: float | None = None,
    ) -> list[RankedStock]:
        return []

    async def fetch_sector_indices(self, market: str = "0001") -> list[SectorIndex]:
        return []

    async def fetch_overseas_price(self, symbol: str, exchange: str = "NAS") -> float:
        return 100.0

    async def fetch_overseas_candles(
        self, symbol: str, end: str = "", exchange: str = "NAS", period: str = "D"
    ) -> list[Candle]:
        return []

    async def fetch_overseas_trading_value_rank(self, exchange: str = "NAS", count: int = 30) -> list[RankedStock]:
        return []

    async def aclose(self) -> None:
        self.closed = True


def test_kis_broker_satisfies_broker_port():
    broker = KISBroker(
        app_key="x", app_secret="x", account_8="00000000", base_url="https://test",
    )
    assert isinstance(broker, BrokerPort)


def test_kis_broker_satisfies_market_data_port():
    broker = KISBroker(
        app_key="x", app_secret="x", account_8="00000000", base_url="https://test",
    )
    assert isinstance(broker, MarketDataPort)


def test_independent_second_broker_satisfies_broker_port():
    assert isinstance(IndependentSecondBroker(), BrokerPort)


def test_independent_second_broker_satisfies_market_data_port():
    assert isinstance(IndependentSecondBroker(), MarketDataPort)


@pytest.fixture
def live_settings(monkeypatch):
    monkeypatch.setattr(settings, "kis_execution_enabled", True)
    monkeypatch.setattr(settings, "kis_execution_dry_run", False)
    monkeypatch.setattr(settings, "trade_allowlist_domestic", ["005930.KS"])
    monkeypatch.setattr(settings, "trade_allowlist_overseas", ["AAPL"])
    monkeypatch.setattr(settings, "kis_max_order_krw", 5_000_000.0)


@pytest.fixture(autouse=True)
def _no_gate_side_effects(monkeypatch):
    import app.execution.kis_bridge as kis_bridge

    monkeypatch.setattr(kis_bridge.kis_market_status, "is_circuit_breaker_active", lambda code: False)

    async def _no_safety_block(side, ticker):
        return None

    monkeypatch.setattr(kis_bridge.safety_switch, "safety_gate_block", _no_safety_block)


async def test_gate_chain_submits_domestic_order_against_a_non_kis_broker(live_settings):
    """The entire kis_bridge.py safety-gate chain (routing, circuit-breaker,
    per-stock halt, sell-ceiling, position-weight) runs against a broker that
    shares zero code with KISBroker — proving it depends only on
    BrokerPort/MarketDataPort, not on anything KIS-specific."""
    broker = IndependentSecondBroker(held=10)
    result = await mirror_trade_to_kis("005930.KS", TODAY, "buy", 1, 70000, broker=broker)
    assert result.startswith("submitted")
    assert broker.orders == [("005930", "buy", 1, None)]


async def test_gate_chain_submits_overseas_order_against_a_non_kis_broker(live_settings):
    broker = IndependentSecondBroker()
    result = await mirror_trade_to_kis("AAPL", TODAY_OVERSEAS, "buy", 1, 315.5, broker=broker)
    assert result.startswith("submitted") and "NASDAQ" in result
    assert broker.overseas_orders == [("AAPL", "buy", 1, 315.5, "NASD")]


async def test_gate_chain_blocks_sell_over_held_against_a_non_kis_broker(live_settings):
    broker = IndependentSecondBroker(held=1)
    result = await place_manual_order("005930.KS", "sell", 5, reference_price=70000, broker=broker)
    assert result.status == "blocked" and "held 1" in result.message
    assert broker.orders == []


async def test_gate_chain_blocks_halted_stock_against_a_non_kis_broker(live_settings, monkeypatch):
    broker = IndependentSecondBroker(held=10)

    async def halted_status(code):
        return StockStatus(code=code, trading_halted=True, administrative_issue=False, nxt_trading_halted=False)

    monkeypatch.setattr(broker, "fetch_stock_status", halted_status)
    result = await mirror_trade_to_kis("005930.KS", TODAY, "buy", 1, 70000, broker=broker)
    assert result.startswith("BLOCKED") and "halted" in result
    assert broker.orders == []
