"""live router — ticks/news/earnings-trend/alerts endpoints, dependencies
stubbed, no network (E-stage). /api/stream is a long-running SSE loop and is
not exercised here."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.web import state
from conftest import broker_stub
from app.web.routers import live


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(live.router)
    return TestClient(app, base_url="http://127.0.0.1:8094")


@pytest.fixture(autouse=True)
def _clean_broker():
    orig = state.kis_broker
    yield
    state.kis_broker = orig
class _FakeTick:
    def __init__(self, price, volume, timestamp):
        self.price, self.volume, self.timestamp = price, volume, timestamp


def test_ticks_rejects_invalid_ticker_format(client):
    resp = client.get("/api/ticks/bad;ticker")
    assert resp.status_code == 422


def test_ticks_returns_formatted_rows(client, monkeypatch):
    ts = datetime(2026, 7, 16, 9, 30)
    monkeypatch.setattr(live, "get_ticks", lambda ticker: [_FakeTick(150.0, 1000, ts)])
    resp = client.get("/api/ticks/AAPL")
    assert resp.status_code == 200
    assert resp.json() == [{"price": 150.0, "volume": 1000, "timestamp": ts.isoformat()}]


def test_ticks_degrades_to_empty_list_on_unexpected_error(client, monkeypatch):
    def broken(ticker):
        raise RuntimeError("store unavailable")

    monkeypatch.setattr(live, "get_ticks", broken)
    resp = client.get("/api/ticks/AAPL")
    assert resp.status_code == 200
    assert resp.json() == []


def test_news_endpoint_empty_when_no_broker(client):
    # KIS-only policy (see app.live.poller's module docstring) — no broker
    # configured means no news, not a yfinance substitute.
    state.kis_broker = broker_stub(None)
    resp = client.get("/api/news/AAPL")
    assert resp.status_code == 200
    assert resp.json() == []


async def test_news_endpoint_returns_krx_items_via_kis(client):
    class FakeBroker:
        async def fetch_news(self, code):
            return [{"title": "headline", "publisher": "src", "published_at": "2026-07-23T10:00:00"}]

    state.kis_broker = broker_stub(FakeBroker())
    resp = client.get("/api/news/005930.KS")
    assert resp.status_code == 200
    assert resp.json() == [{"title": "headline", "publisher": "src", "published_at": "2026-07-23T10:00:00"}]


async def test_news_endpoint_returns_us_items_via_kis(client):
    class FakeBroker:
        async def fetch_overseas_news(self, symbol):
            return [{"title": "headline", "publisher": "src", "published_at": "2026-07-23T10:00:00"}]

    state.kis_broker = broker_stub(FakeBroker())
    resp = client.get("/api/news/AAPL")
    assert resp.status_code == 200
    assert resp.json() == [{"title": "headline", "publisher": "src", "published_at": "2026-07-23T10:00:00"}]


async def test_news_endpoint_degrades_to_empty_list_on_failure(client):
    class FakeBroker:
        async def fetch_overseas_news(self, symbol):
            raise RuntimeError("KIS down")

    state.kis_broker = broker_stub(FakeBroker())
    resp = client.get("/api/news/AAPL")
    assert resp.json() == []


async def test_earnings_trend_degrades_to_empty_dict_on_failure(client, monkeypatch):
    async def broken(ticker):
        raise RuntimeError("no data")

    monkeypatch.setattr(live, "get_earnings_trend", broken)
    resp = client.get("/api/earnings-trend/AAPL")
    assert resp.status_code == 200
    assert resp.json() == {}


def test_auto_trade_status_reports_enabled_and_phase(client, monkeypatch):
    class _FakeStatus:
        phase = "analyzing"
        current_tickers = {"AAPL", "005930.KS"}
        current_stage = {}
        tickers_this_cycle = ["AAPL", "005930.KS"]
        cycle_started_at = datetime(2026, 7, 21, 9, 0)
        cycle_finished_at = None
        next_cycle_at = datetime(2026, 7, 21, 9, 30)

    from app.config import ExecutionMode

    async def fake_get_display_name(ticker):
        return {"AAPL": "Apple Inc.", "005930.KS": "삼성전자"}[ticker]

    monkeypatch.setattr(live, "get_auto_trade_status", lambda: _FakeStatus())
    monkeypatch.setattr(live.settings, "auto_trade_enabled", True)
    monkeypatch.setattr(live.settings, "execution_mode", ExecutionMode.SEMI_AUTO)
    monkeypatch.setattr(live.state, "get_display_name", fake_get_display_name)
    resp = client.get("/api/auto-trade/status")
    assert resp.status_code == 200
    body = resp.json()
    assert body["enabled"] is True
    assert body["phase"] == "analyzing"
    assert body["current_tickers"] == ["005930.KS", "AAPL"]  # sorted
    # Analyzing-card name lookup (covers a screened candidate never on the
    # trade allowlist, which has no other name source while it's mid-cycle)
    # — see the endpoint's own comment.
    assert body["current_ticker_names"] == {"005930.KS": "삼성전자", "AAPL": "Apple Inc."}
    assert body["tickers_this_cycle"] == ["AAPL", "005930.KS"]
    assert body["cycle_started_at"] == datetime(2026, 7, 21, 9, 0).isoformat()
    assert body["cycle_finished_at"] is None
    assert body["next_cycle_at"] == datetime(2026, 7, 21, 9, 30).isoformat()


async def test_auto_trade_status_reports_split_reset_boundaries(client, monkeypatch):
    class _FakeStatus:
        phase = "idle"
        current_tickers = set()
        current_stage = {}
        tickers_this_cycle = []
        cycle_started_at = None
        cycle_finished_at = None
        next_cycle_at = None

    from app.config import ExecutionMode

    monkeypatch.setattr(live, "get_auto_trade_status", lambda: _FakeStatus())
    monkeypatch.setattr(live.settings, "auto_trade_enabled", True)
    monkeypatch.setattr(live.settings, "execution_mode", ExecutionMode.SEMI_AUTO)

    domestic_at = datetime(2026, 8, 20, 8, 30, tzinfo=ZoneInfo("Asia/Seoul"))
    overseas_at = datetime(2026, 8, 19, 9, 0, tzinfo=ZoneInfo("America/New_York"))

    async def fake_domestic():
        return domestic_at

    async def fake_overseas():
        return overseas_at

    monkeypatch.setattr(live, "latest_domestic_signal_card_reset_at", fake_domestic)
    monkeypatch.setattr(live, "latest_overseas_signal_card_reset_at", fake_overseas)

    resp = client.get("/api/auto-trade/status")
    body = resp.json()
    assert body["cards_reset_at_domestic"] == domestic_at.isoformat()
    assert body["cards_reset_at_overseas"] == overseas_at.isoformat()
    # Deprecated legacy key kept equal to the domestic value — a cached
    # pre-split frontend build must keep filtering domestic cards exactly
    # as before.
    assert body["cards_reset_at"] == body["cards_reset_at_domestic"]
    assert body["cards_reset_at_overseas"] != body["cards_reset_at_domestic"]
    # Pins the "don't convert overseas to KST" decision — the -04:00/-05:00
    # ET offset must survive into the JSON as-is.
    assert body["cards_reset_at_overseas"].endswith("-04:00")


def test_auto_trade_status_disabled_when_manual_mode(client, monkeypatch):
    class _FakeStatus:
        phase = "manual_skip"
        current_tickers = set()
        current_stage = {}
        tickers_this_cycle = []
        cycle_started_at = None
        cycle_finished_at = None
        next_cycle_at = None

    monkeypatch.setattr(live, "get_auto_trade_status", lambda: _FakeStatus())
    monkeypatch.setattr(live.settings, "auto_trade_enabled", True)
    from app.config import ExecutionMode
    monkeypatch.setattr(live.settings, "execution_mode", ExecutionMode.MANUAL)
    resp = client.get("/api/auto-trade/status")
    assert resp.json()["enabled"] is False


def test_analyze_now_starts_a_cycle_when_allowed(client, monkeypatch):
    calls = []

    # Records args the moment the endpoint calls run_cycle_now(...) — not
    # when the fire-and-forget asyncio.create_task(...) eventually runs the
    # resulting coroutine, which isn't guaranteed to happen before the
    # response (and this test's assertions) return.
    def fake_run_cycle_now(exclude_already_analyzed_today=False, force_refresh=False):
        calls.append((exclude_already_analyzed_today, force_refresh))

        async def _noop():
            return None

        return _noop()

    monkeypatch.setattr(live, "check_manual_trigger_allowed", lambda: None)
    monkeypatch.setattr(live, "run_cycle_now", fake_run_cycle_now)
    resp = client.post("/api/auto-trade/analyze-now")
    assert resp.status_code == 200
    assert resp.json() == {"status": "started"}
    # AI-engine efficiency audit (2026-07-27): the manual button must force a
    # fresh analysis, not silently reuse the 15-minute report cache.
    assert calls == [(True, True)]


def test_analyze_now_rejected_returns_429(client, monkeypatch):
    def fake_check():
        raise live.CooldownActive("cooldown active, 120s remaining")

    monkeypatch.setattr(live, "check_manual_trigger_allowed", fake_check)
    resp = client.post("/api/auto-trade/analyze-now")
    assert resp.status_code == 429
    assert "cooldown" in resp.json()["detail"]


async def test_alerts_endpoint_formats_recent_alerts(client, monkeypatch):
    class _FakeAlert:
        ticker = "AAPL"
        kind = "price_move"
        message = "AAPL up 3%"
        triggered_at = datetime(2026, 7, 16, 10, 0)

    async def fake_recent_alerts():
        return [_FakeAlert()]

    monkeypatch.setattr(live.alerts_module, "recent_alerts", fake_recent_alerts)
    resp = client.get("/api/alerts")
    assert resp.status_code == 200
    body = resp.json()
    assert body[0]["ticker"] == "AAPL"
    assert body[0]["triggered_at"] == datetime(2026, 7, 16, 10, 0).isoformat()
