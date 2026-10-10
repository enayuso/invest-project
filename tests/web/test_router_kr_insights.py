"""kr_insights router — TestClient, KIS broker/route_ticker stubbed, no
network (E-stage)."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.web import state
from conftest import broker_stub
from app.web.routers import kr_insights


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(kr_insights.router)
    return TestClient(app, base_url="http://127.0.0.1:8094")


@pytest.fixture(autouse=True)
def _clean():
    state._cache.clear()
    orig = state.kis_broker
    yield
    state._cache.clear()
    state.kis_broker = orig
def test_unknown_kind_returns_404(client):
    state.kis_broker = broker_stub(object())
    resp = client.get("/api/kr/bogus-kind/005930.KS")
    assert resp.status_code == 404


def test_broker_not_configured_returns_503(client):
    state.kis_broker = broker_stub(None)
    resp = client.get("/api/kr/news/005930.KS")
    assert resp.status_code == 503


def test_non_krx_ticker_returns_unsupported(client):
    state.kis_broker = broker_stub(object())
    resp = client.get("/api/kr/news/AAPL")
    assert resp.status_code == 200
    assert resp.json() == {"unsupported": True}


class _FakeNewsItem:
    def __init__(self, day, time, source, title):
        self.day, self.time, self.source, self.title = day, time, source, title


async def test_news_kind_returns_structured_table(client, monkeypatch):
    class FakeBroker:
        async def fetch_news_titles(self, code):
            return [_FakeNewsItem("20260716", "0930", "연합뉴스", "삼성전자 실적 발표")]

    state.kis_broker = broker_stub(FakeBroker())
    resp = client.get("/api/kr/news/005930.KS")
    assert resp.status_code == 200
    body = resp.json()
    assert body["type"] == "news"
    assert body["items"][0]["title"] == "삼성전자 실적 발표"
    assert body["items"][0]["time"] == "07/16 09:30"


async def test_data_fetch_failure_returns_error_payload_not_500(client):
    class BrokenBroker:
        async def fetch_news_titles(self, code):
            raise RuntimeError("KIS timeout")

    state.kis_broker = broker_stub(BrokenBroker())
    resp = client.get("/api/kr/news/005930.KS")
    assert resp.status_code == 200
    assert "데이터를 불러오지 못했습니다" in resp.json()["error"]
