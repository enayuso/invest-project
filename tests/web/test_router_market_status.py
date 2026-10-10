"""market_status router — TestClient, state._market_feed stubbed, no real
WebSocket (E-stage)."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.web import state
from app.web.routers import market_status


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(market_status.router)
    return TestClient(app, base_url="http://127.0.0.1:8094")


@pytest.fixture(autouse=True)
def _clean():
    orig = state._market_feed
    yield
    state._market_feed = orig


@dataclass
class _Status:
    trading_halted: bool
    halt_reason: str
    market_op_code: str
    vi_code: str
    stock_state_code: str


@dataclass
class _OrderBookRow:
    time: str
    asks: list
    bids: list


@dataclass
class _Tick:
    time: str
    price: float
    change_pct: float
    volume: int


@dataclass
class _Fill:
    order_no: str
    code: str
    name: str
    side: str
    order_qty: int
    fill_qty: int
    fill_price: float
    time: str
    status: str


class _FakeFeed:
    def __init__(self):
        self.statuses = {}
        self.orderbook = {}
        self.ticks = {}
        self.fills = []


def test_market_status_empty_when_no_feed(client):
    state._market_feed = None
    assert client.get("/api/market-status").json() == {}


def test_market_status_reports_statuses_keyed_by_ks_ticker(client):
    feed = _FakeFeed()
    feed.statuses["005930"] = _Status(True, "circuit breaker", "174", "", "")
    state._market_feed = feed
    resp = client.get("/api/market-status")
    assert resp.json() == {
        "005930.KS": {
            "trading_halted": True, "halt_reason": "circuit breaker",
            "market_op_code": "174", "vi_code": "", "stock_state_code": "",
        }
    }


def test_orderbook_empty_when_no_feed(client):
    state._market_feed = None
    assert client.get("/api/orderbook/005930.KS").json() == {}


def test_orderbook_empty_when_no_data_for_code(client):
    state._market_feed = _FakeFeed()
    assert client.get("/api/orderbook/005930.KS").json() == {}


def test_orderbook_returns_book_and_tick_data(client):
    feed = _FakeFeed()
    feed.orderbook["005930"] = _OrderBookRow(time="09:00:00", asks=[1, 2], bids=[3, 4])
    feed.ticks["005930"] = _Tick(time="09:00:01", price=74000.0, change_pct=1.2, volume=1000)
    state._market_feed = feed
    resp = client.get("/api/orderbook/005930.KS")
    body = resp.json()
    assert body["ticker"] == "005930.KS"
    assert body["last"] == 74000.0
    assert body["asks"] == [1, 2]


def test_fills_empty_when_no_feed(client):
    state._market_feed = None
    assert client.get("/api/fills").json() == []


def test_fills_returns_newest_first(client):
    feed = _FakeFeed()
    feed.fills = [
        _Fill("1", "005930", "삼성전자", "buy", 10, 10, 74000.0, "09:00:00", "filled"),
        _Fill("2", "005930", "삼성전자", "sell", 5, 5, 75000.0, "09:05:00", "filled"),
    ]
    state._market_feed = feed
    resp = client.get("/api/fills")
    rows = resp.json()
    assert rows[0]["order_no"] == "2"
    assert rows[1]["order_no"] == "1"


async def test_fills_limit_zero_returns_no_rows():
    """2026-08-26: `fills[-limit:]` is wrong for limit <= 0 — Python's
    `-0 == 0`, so limit=0 sliced `fills[0:]` and returned the WHOLE buffer.
    Calls the function directly (as the original repro did), which bypasses
    FastAPI's own `Query(ge=0)` validation and confirms the fix isn't only
    at the HTTP boundary."""
    from app.web.routers import market_status

    feed = _FakeFeed()
    feed.fills = [_Fill("1", "005930", "삼성전자", "buy", 10, 10, 74000.0, "09:00:00", "filled")]
    state._market_feed = feed

    result = await market_status.fills_endpoint(limit=0)

    assert result == []


def test_fills_negative_limit_is_rejected_by_the_api(client):
    feed = _FakeFeed()
    feed.fills = [_Fill("1", "005930", "삼성전자", "buy", 10, 10, 74000.0, "09:00:00", "filled")]
    state._market_feed = feed

    resp = client.get("/api/fills?limit=-1")

    assert resp.status_code == 422


def test_fills_limit_larger_than_the_buffer_returns_everything(client):
    feed = _FakeFeed()
    feed.fills = [
        _Fill("1", "005930", "삼성전자", "buy", 10, 10, 74000.0, "09:00:00", "filled"),
        _Fill("2", "005930", "삼성전자", "sell", 5, 5, 75000.0, "09:05:00", "filled"),
    ]
    state._market_feed = feed

    resp = client.get("/api/fills?limit=1000")

    assert len(resp.json()) == 2


def test_fills_backfills_name_when_kis_sends_it_blank(client, monkeypatch):
    # Real incident (2026-08-03): KIS's own 체결통보 name field (cntg_isnm40)
    # came back blank for 042660.KS, and the trade-fill modal's own
    # `f.name || f.ticker` fallback had nothing better than the bare ticker
    # to show — same display-name cache every other panel uses should fill
    # this in instead.
    async def fake_get_display_name(ticker):
        assert ticker == "042660.KS"
        return "한화오션"

    monkeypatch.setattr(state, "get_display_name", fake_get_display_name)
    feed = _FakeFeed()
    feed.fills = [_Fill("3", "042660", "", "buy", 34, 34, 84400.0, "09:10:00", "filled")]
    state._market_feed = feed
    resp = client.get("/api/fills")
    assert resp.json()[0]["name"] == "한화오션"
