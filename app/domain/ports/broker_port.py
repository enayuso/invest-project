# app/domain/ports/broker_port.py
# Account/order contract — a Protocol, not a base class, so an existing
# broker class (e.g. KISBroker) satisfies it by matching method shapes,
# with no import-time coupling back to domain/.
# 계좌/주문 계약 (구조적 타이핑 — 상속 없이 시그니처만 맞으면 충족)

from __future__ import annotations

from typing import Protocol, runtime_checkable

from app.domain.models import (
    BrokerBalance,
    BrokerExecution,
    BuyableInfo,
    OrderResult,
    OverseasCash,
    SellableInfo,
)


@runtime_checkable
class BrokerPort(Protocol):
    """Account inquiry + order execution, domestic and overseas.
    계좌 조회 및 주문 실행 (국내/해외 공통 계약)"""

    async def fetch_balance(self) -> BrokerBalance: ...

    async def fetch_buyable(self, code: str, price: float | None = None) -> BuyableInfo: ...

    async def fetch_sellable(self, code: str) -> SellableInfo: ...

    async def submit_order(
        self, code: str, side: str, quantity: int, price: float | None = None
    ) -> OrderResult: ...

    async def cancel_order(
        self,
        order_no: str,
        org_no: str,
        quantity: int,
        price: float | None = None,
        ord_dvsn: str = "00",
    ) -> OrderResult: ...

    async def revise_order(
        self, order_no: str, org_no: str, quantity: int, new_price: float, ord_dvsn: str = "00"
    ) -> OrderResult: ...

    async def fetch_daily_executions(self, start: str, end: str) -> list[BrokerExecution]: ...

    async def fetch_overseas_balance(
        self, exchange: str = "NASD", currency: str = "USD"
    ) -> BrokerBalance: ...

    async def submit_overseas_order(
        self, code: str, side: str, quantity: int, price: float, exchange: str = "NASD"
    ) -> OrderResult: ...

    async def cancel_overseas_order(
        self, order_no: str, code: str, quantity: int, exchange: str = "NASD"
    ) -> OrderResult: ...

    async def fetch_overseas_executions(self, start: str, end: str) -> list[BrokerExecution]: ...

    async def fetch_overseas_present_balance(self, currency: str = "USD") -> OverseasCash: ...
