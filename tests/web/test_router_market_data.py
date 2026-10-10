"""market_data router — candles/volume-rank/index endpoints, KIS broker
stubbed, no network (E-stage)."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.web import state
from conftest import broker_stub
from app.web.routers import market_data


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(market_data.router)
    return TestClient(app, base_url="http://127.0.0.1:8094")


@pytest.fixture(autouse=True)
def _clean():
    state._cache.clear()
    orig = state.kis_broker
    yield
    state._cache.clear()
    state.kis_broker = orig
def test_candles_unknown_period_returns_empty_list(client):
    resp = client.get("/api/candles/AAPL?period=X")
    assert resp.status_code == 200
    assert resp.json() == []


class _FakeCandle:
    def __init__(self, day, o, h, l, c, v):
        self.day, self.open, self.high, self.low, self.close, self.volume = day, o, h, l, c, v


async def test_candles_krx_ticker_uses_kis_broker(client, monkeypatch):
    class FakeBroker:
        async def fetch_daily_candles(self, code, start, end, period="D"):
            return [_FakeCandle("20260716", 100, 105, 99, 104, 1000)]

    state.kis_broker = broker_stub(FakeBroker())
    resp = client.get("/api/candles/005930.KS")
    assert resp.status_code == 200
    body = resp.json()
    assert body == [{"day": "20260716", "open": 100, "high": 105, "low": 99, "close": 104, "volume": 1000}]


def test_candles_us_ticker_empty_when_no_broker(client):
    # KIS-only policy (no yfinance fallback for anything the frontend shows)
    # — no broker configured means no candles, not a yfinance substitute.
    state.kis_broker = broker_stub(None)
    resp = client.get("/api/candles/AAPL")
    assert resp.status_code == 200
    assert resp.json() == []


async def test_candles_us_ticker_yearly_period_always_empty(client):
    # KIS's overseas per-stock candle TR has no Y (연봉) option — confirmed
    # against the KIS API catalog, not a transient failure — so this stays
    # empty even with a working broker (see candles_endpoint's docstring).
    class FakeBroker:
        async def fetch_overseas_candles(self, symbol, period="D"):
            raise AssertionError("Y should never reach fetch_overseas_candles — filtered by _OVERSEAS_CANDLE_PERIODS")

    state.kis_broker = broker_stub(FakeBroker())
    resp = client.get("/api/candles/AAPL?period=Y")
    assert resp.status_code == 200
    assert resp.json() == []


def test_minute_candles_empty_when_no_broker(client):
    state.kis_broker = broker_stub(None)
    resp = client.get("/api/minute/005930.KS")
    assert resp.json() == []


def test_minute_candles_empty_for_non_krx_ticker(client):
    state.kis_broker = broker_stub(object())
    resp = client.get("/api/minute/AAPL")
    assert resp.json() == []


def test_volume_rank_empty_when_no_broker(client):
    state.kis_broker = broker_stub(None)
    resp = client.get("/api/volume-rank")
    assert resp.json() == []


class _FakeRankRow:
    def __init__(self, rank, code, name, price=1000.0, change_pct=1.0, volume=100):
        self.rank, self.code, self.name = rank, code, name
        self.price, self.change_pct, self.volume = price, change_pct, volume


def test_ticker_suggestions_filters_leveraged(client):
    class FakeBroker:
        async def fetch_trading_value_rank(self, count):
            return [
                _FakeRankRow(1, "122630", "KODEX 레버리지"),
                _FakeRankRow(2, "005930", "삼성전자"),
                _FakeRankRow(3, "252670", "KODEX 200선물인버스2X"),
                _FakeRankRow(4, "000660", "SK하이닉스"),
            ]

    state.kis_broker = broker_stub(FakeBroker())
    rows = client.get("/api/ticker-suggestions").json()
    names = [r["name"] for r in rows]
    assert names == ["삼성전자", "SK하이닉스"]
    assert rows[0] == {"symbol": "005930.KS", "name": "삼성전자", "market": "domestic",
                       "price": 1000.0, "change_pct": 1.0}


def test_ticker_suggestions_filters_cash_bond_funds(client):
    class FakeBroker:
        async def fetch_trading_value_rank(self, count):
            return [
                _FakeRankRow(1, "481050", "KODEX CD1년금리플러스액티브(합성)"),
                _FakeRankRow(2, "005930", "삼성전자"),
                _FakeRankRow(3, "439870", "KODEX 국고채30년액티브"),
                _FakeRankRow(4, "000660", "SK하이닉스"),
            ]

    state.kis_broker = broker_stub(FakeBroker())
    names = [r["name"] for r in client.get("/api/ticker-suggestions").json()]
    assert names == ["삼성전자", "SK하이닉스"]


def test_ticker_suggestions_empty_when_broker_none(client):
    state.kis_broker = broker_stub(None)
    assert client.get("/api/ticker-suggestions").json() == []


async def test_index_quotes_skips_failed_code_but_returns_others(client, monkeypatch):
    class _Quote:
        def __init__(self, code, name, value, change, change_pct):
            self.code, self.name, self.value = code, name, value
            self.change, self.change_pct = change, change_pct

    class FakeBroker:
        async def fetch_index_price(self, code):
            if code == "0001":
                return _Quote("0001", "KOSPI", 2500.0, 10.0, 0.4)
            raise RuntimeError("KOSDAQ unavailable")

    state.kis_broker = broker_stub(FakeBroker())
    resp = client.get("/api/index")
    body = resp.json()
    assert len(body) == 1
    assert body[0]["name"] == "KOSPI"


# ---------------------------------------------------------------------------
# valuation — 2026-08-26: a transient KIS failure used to cache {} for the
# full 300s TTL, blanking the trading tab's fundamentals row for 5 minutes
# even after KIS had already recovered.
# ---------------------------------------------------------------------------


class _FakeValuation:
    def __init__(self, market_cap=1_000_000.0, per=10.0, pbr=1.5, week52_high=None, week52_low=None):
        self.market_cap, self.per, self.pbr = market_cap, per, pbr
        self.week52_high, self.week52_low = week52_high, week52_low


def test_valuation_transient_failure_is_not_cached_and_the_next_call_retries(client):
    calls = 0

    class FlakyBroker:
        async def fetch_valuation(self, code):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("KIS timeout")
            return _FakeValuation()

    state.kis_broker = broker_stub(FlakyBroker())

    r1 = client.get("/api/valuation/005930.KS")
    assert r1.json() == {}

    r2 = client.get("/api/valuation/005930.KS")
    assert r2.json()["per"] == 10.0
    assert calls == 2  # the failed first call was never cached


def test_valuation_success_is_cached_within_the_ttl(client):
    calls = 0

    class FakeBroker:
        async def fetch_valuation(self, code):
            nonlocal calls
            calls += 1
            return _FakeValuation()

    state.kis_broker = broker_stub(FakeBroker())

    client.get("/api/valuation/005930.KS")
    client.get("/api/valuation/005930.KS")

    assert calls == 1  # the fix must not disable caching outright


def test_valuation_of_an_unroutable_ticker_returns_empty_and_is_cached(client):
    calls = 0

    class FakeBroker:
        async def fetch_valuation(self, code):
            nonlocal calls
            calls += 1
            return _FakeValuation()

    state.kis_broker = broker_stub(FakeBroker())

    r1 = client.get("/api/valuation/not-a-real-symbol-format-!!")
    r2 = client.get("/api/valuation/not-a-real-symbol-format-!!")

    assert r1.json() == {}
    assert r2.json() == {}
    assert calls == 0  # never routed to a broker call at all — confirmed empty, not a failure


def test_valuation_returns_empty_when_no_broker_is_configured(client):
    state.kis_broker = broker_stub(None)
    resp = client.get("/api/valuation/005930.KS")
    assert resp.json() == {}
