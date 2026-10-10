"""portfolio router — TestClient, ledger/KIS dependencies stubbed, no
network/DB (E-stage)."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.clock import FakeClock
from app.config import settings
from app.live.store import Tick
from app.web import state
from conftest import broker_stub, bypass_kis_gate
from app.web.routers import portfolio


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(portfolio.router)
    bypass_kis_gate(app)
    return TestClient(app, base_url="http://127.0.0.1:8094")


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.setattr(settings, "trade_allowlist_domestic", [])
    monkeypatch.setattr(settings, "trade_allowlist_overseas", ["AAPL"])
    state._display_info.clear()
    original_broker = state.kis_broker
    yield
    state._display_info.clear()
    state.kis_broker = original_broker
async def test_portfolio_uses_kis_position_when_present(client, monkeypatch):
    async def fake_snapshot():
        return (
            {"cash_deposit": 1.0},
            {"AAPL": {"quantity": 5, "current_price": 200.0, "avg_price": 180.0,
                      "pnl_amount": 100.0, "pnl_pct": 11.1, "eval_amount": 1000.0}},
        )

    async def fake_get_display_name(ticker):
        return "Apple Inc."

    monkeypatch.setattr(state, "kis_account_snapshot", fake_snapshot)
    monkeypatch.setattr(state, "get_display_name", fake_get_display_name)
    monkeypatch.setattr(portfolio, "latest", lambda ticker: None)

    resp = client.get("/api/portfolio")
    assert resp.status_code == 200
    body = resp.json()
    row = body["rows"][0]
    assert row["venue"] == "kis"
    assert row["shares"] == 5
    assert row["current_price"] == 200.0


async def test_portfolio_no_kis_position_shows_zero_shares_no_ledger_fallback(client, monkeypatch):
    """The local paper-ledger fallback for a ticker KIS doesn't hold was
    deliberately removed (55eb188, phantom-holdings incident) — a flat
    ticker now shows 0 shares / no venue rather than the ledger's own
    bookkeeping, and nothing here even references the ledger anymore."""
    async def fake_snapshot():
        return None, {}

    async def fake_get_display_name(ticker):
        return "Apple Inc."

    monkeypatch.setattr(state, "kis_account_snapshot", fake_snapshot)
    monkeypatch.setattr(state, "get_display_name", fake_get_display_name)
    monkeypatch.setattr(portfolio, "latest", lambda ticker: None)

    resp = client.get("/api/portfolio")
    body = resp.json()
    row = body["rows"][0]
    assert row["venue"] is None
    assert row["shares"] == 0
    assert body["account"] is None


async def test_portfolio_venue_none_uses_live_quote_price_when_flat(client, monkeypatch):
    """A flat ticker (no KIS position) still shows a `current_price` from
    the live quote store — the one piece of `_position_and_price`'s old
    per-ticker behavior that survived the ledger removal."""
    async def fake_snapshot():
        return None, {}

    async def fake_get_display_name(ticker):
        return "Apple Inc."

    monkeypatch.setattr(state, "kis_account_snapshot", fake_snapshot)
    monkeypatch.setattr(state, "get_display_name", fake_get_display_name)
    monkeypatch.setattr(
        portfolio,
        "latest",
        lambda ticker: Tick(price=195.5, volume=100, timestamp=datetime(2026, 7, 29, 10, 0)),
    )

    resp = client.get("/api/portfolio")
    row = resp.json()["rows"][0]
    assert row["venue"] is None
    assert row["current_price"] == 195.5


async def test_portfolio_includes_a_real_holding_thats_off_the_trade_allowlist(client, monkeypatch):
    # Real incident: a ticker with real KIS shares (bought/held from before
    # this trade_allowlist run, or a SELL-only ticker that never qualifies for
    # the BUY-only auto-add hook) fell off settings.trade_allowlist — on
    # restart, the trade allowlist isn't DB-persisted — and was invisible
    # here even though KIS itself still reported the shares.
    async def fake_snapshot():
        return (
            {"cash_deposit": 1.0},
            {
                "AAPL": {"quantity": 2, "current_price": 200.0, "avg_price": 180.0,
                         "pnl_amount": 40.0, "pnl_pct": 11.1, "eval_amount": 400.0},
                "003280.KS": {"quantity": 10, "current_price": 2000.0, "avg_price": 1850.0,
                              "pnl_amount": 1500.0, "pnl_pct": 8.1, "eval_amount": 20000.0},
            },
        )

    async def fake_get_display_name(ticker):
        return "003280.KS" if ticker == "003280.KS" else "Apple Inc."

    monkeypatch.setattr(state, "kis_account_snapshot", fake_snapshot)
    monkeypatch.setattr(state, "get_display_name", fake_get_display_name)
    monkeypatch.setattr(portfolio, "latest", lambda ticker: None)

    resp = client.get("/api/portfolio")
    assert resp.status_code == 200
    rows = {row["ticker"]: row for row in resp.json()["rows"]}
    assert "003280.KS" in rows  # not on settings.trade_allowlist (["AAPL"] per _clean), still shown
    assert rows["003280.KS"]["shares"] == 10
    assert rows["003280.KS"]["venue"] == "kis"


async def test_trades_returns_empty_when_no_kis_broker(client, monkeypatch):
    """The local paper-ledger merge into /api/trades was deliberately
    removed (55eb188) — a ledger "trade" doesn't mean it ever reached KIS
    (see rebalance_to_target's "ledger record first, KIS mirror second"
    ordering), so it read as confusing noise next to real fills. With no
    KIS broker configured there's nothing left to report at all."""
    state.kis_broker = broker_stub(None)

    resp = client.get("/api/trades")
    assert resp.status_code == 200
    assert resp.json() == []


async def test_trades_lists_kis_executions_when_broker_present(client, monkeypatch):
    class FakeBroker:
        pass

    class FakeExecution:
        def __init__(self, code, order_no, order_date, filled_qty, order_qty):
            self.code = code
            self.order_no = order_no
            self.order_date = order_date
            self.side = "buy"
            self.avg_price = 150.0
            self.filled_qty = filled_qty
            self.order_qty = order_qty

    async def fake_recent_executions():
        domestic = [FakeExecution("005930", "1", "20260715", 2, 2)]
        overseas = []
        return domestic, overseas

    async def fake_get_display_name(ticker):
        return "삼성전자"

    monkeypatch.setattr(state, "recent_executions", fake_recent_executions)
    monkeypatch.setattr(state, "get_display_name", fake_get_display_name)
    state.kis_broker = broker_stub(FakeBroker())

    resp = client.get("/api/trades")
    assert resp.status_code == 200
    rows = resp.json()
    assert len(rows) == 1
    assert rows[0]["venue"] == "kis"
    assert rows[0]["status"] == "filled"
    assert rows[0]["ticker"] == "005930.KS"


async def test_trades_negative_limit_is_rejected_by_the_api(client, monkeypatch):
    """2026-08-26: `rows[:limit]` on a negative limit silently drops rows off
    the wrong end (e.g. limit=-1 returns all but the last row) instead of
    signaling a malformed request."""
    state.kis_broker = broker_stub(None)
    resp = client.get("/api/trades?limit=-1")
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# /api/pnl — realized P&L over a date range (app.execution.pnl)
# ---------------------------------------------------------------------------


class _FakePnlExecution:
    def __init__(self, code, order_no, order_date, side, filled_qty, avg_price, name="삼성전자", fee_tax=0.0):
        self.code = code
        self.order_no = order_no
        self.order_date = order_date
        self.side = side
        self.name = name
        self.order_qty = filled_qty
        self.filled_qty = filled_qty
        self.remaining_qty = 0
        self.avg_price = avg_price
        self.filled_amount = filled_qty * avg_price
        self.fee_tax = fee_tax


class _FakePnlBroker:
    def __init__(self, domestic=None, overseas=None):
        self._domestic = domestic or []
        self._overseas = overseas or []

    async def fetch_daily_executions(self, start, end):
        return self._domestic

    async def fetch_overseas_executions(self, start, end):
        return self._overseas


async def test_pnl_returns_empty_when_no_kis_broker(client, monkeypatch):
    state.kis_broker = broker_stub(None)
    resp = client.get("/api/pnl")
    assert resp.status_code == 200
    body = resp.json()
    assert body["rows"] == []
    assert body["summary"]["total_pnl_domestic"] == 0.0
    assert body["summary"]["win_rate"] is None


async def test_pnl_bad_start_date_returns_422_not_500(client, monkeypatch):
    """2026-08-26: date.fromisoformat raises ValueError on a malformed
    string, unguarded — a bad query param used to reach the client as an
    unhandled 500 instead of a 422."""
    state.kis_broker = broker_stub(None)
    resp = client.get("/api/pnl", params={"start": "not-a-date"})
    assert resp.status_code == 422


async def test_pnl_bad_end_date_returns_422_not_500(client, monkeypatch):
    state.kis_broker = broker_stub(None)
    resp = client.get("/api/pnl", params={"end": "not-a-date"})
    assert resp.status_code == 422


async def test_pnl_computes_realized_gain_for_a_round_trip(client, monkeypatch):
    async def fake_get_display_name(ticker):
        return "삼성전자"

    monkeypatch.setattr(state, "get_display_name", fake_get_display_name)
    monkeypatch.setattr(
        portfolio, "get_clock", lambda: FakeClock(datetime(2026, 8, 5, tzinfo=timezone.utc))
    )
    state.kis_broker = broker_stub(_FakePnlBroker(domestic=[
        _FakePnlExecution("005930", "1", "20260701", "buy", 10, 70_000.0),
        _FakePnlExecution("005930", "2", "20260710", "sell", 10, 75_000.0),
    ]))

    resp = client.get("/api/pnl", params={"start": "2026-07-01", "end": "2026-07-31"})
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["rows"]) == 1
    row = body["rows"][0]
    assert row["ticker"] == "005930.KS"
    assert row["buy_avg_price"] == 70_000.0
    assert row["sell_price"] == 75_000.0
    assert row["realized_pnl"] == 50_000.0
    assert body["summary"]["total_pnl_domestic"] == 50_000.0
    assert body["summary"]["win_count"] == 1
    assert body["start_clamped"] is False


async def test_pnl_clamps_start_older_than_the_lookback_cap(client, monkeypatch):
    monkeypatch.setattr(
        portfolio, "get_clock", lambda: FakeClock(datetime(2026, 8, 5, tzinfo=timezone.utc))
    )
    state.kis_broker = broker_stub(_FakePnlBroker())

    resp = client.get("/api/pnl", params={"start": "2025-01-01", "end": "2026-08-05"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["start_clamped"] is True
    assert body["start"] == "2026-05-07"  # 90 days back from 2026-08-05


async def test_pnl_defaults_to_a_thirty_day_window_ending_today(client, monkeypatch):
    monkeypatch.setattr(
        portfolio, "get_clock", lambda: FakeClock(datetime(2026, 8, 5, tzinfo=timezone.utc))
    )
    state.kis_broker = broker_stub(_FakePnlBroker())

    resp = client.get("/api/pnl")
    assert resp.status_code == 200
    body = resp.json()
    assert body["end"] == "2026-08-05"
    assert body["start"] == "2026-07-06"


async def test_pnl_market_filter_keeps_only_the_requested_market(client, monkeypatch):
    async def fake_get_display_name(ticker):
        return ticker

    monkeypatch.setattr(state, "get_display_name", fake_get_display_name)
    monkeypatch.setattr(
        portfolio, "get_clock", lambda: FakeClock(datetime(2026, 8, 5, tzinfo=timezone.utc))
    )
    state.kis_broker = broker_stub(_FakePnlBroker(
        domestic=[
            _FakePnlExecution("005930", "1", "20260701", "buy", 10, 70_000.0),
            _FakePnlExecution("005930", "2", "20260710", "sell", 10, 75_000.0),
        ],
        overseas=[
            _FakePnlExecution("AAPL", "1", "20260701", "buy", 5, 150.0),
            _FakePnlExecution("AAPL", "2", "20260710", "sell", 5, 160.0),
        ],
    ))

    resp = client.get("/api/pnl", params={"start": "2026-07-01", "end": "2026-07-31", "market": "overseas"})
    assert resp.status_code == 200
    rows = resp.json()["rows"]
    assert len(rows) == 1
    assert rows[0]["ticker"] == "AAPL"


# ---------------------------------------------------------------------------
# /api/pnl fee_measured / fee_unavailable_overseas (2026-09-08 fee/tax audit)
# — overseas execution rows never carry a real fee/tax figure at all (unlike
# domestic, which the enrichment block below can at least probe for), so
# these flags let the frontend tell "measured and zero" apart from "never
# measured" instead of silently presenting a gross number as net.


async def test_pnl_domestic_only_reports_fee_unavailable_overseas_false(client, monkeypatch):
    monkeypatch.setattr(
        portfolio, "get_clock", lambda: FakeClock(datetime(2026, 8, 5, tzinfo=timezone.utc))
    )
    state.kis_broker = broker_stub(_FakePnlBroker(domestic=[
        _FakePnlExecution("005930", "1", "20260701", "buy", 10, 70_000.0),
        _FakePnlExecution("005930", "2", "20260710", "sell", 10, 75_000.0),
    ]))

    resp = client.get("/api/pnl", params={"start": "2026-07-01", "end": "2026-07-31"})
    body = resp.json()
    assert body["fee_unavailable_overseas"] is False
    assert body["rows"][0]["fee_measured"] is True


async def test_pnl_overseas_trade_reports_fee_unavailable_overseas_true(client, monkeypatch):
    async def fake_get_display_name(ticker):
        return ticker

    monkeypatch.setattr(state, "get_display_name", fake_get_display_name)
    monkeypatch.setattr(
        portfolio, "get_clock", lambda: FakeClock(datetime(2026, 8, 5, tzinfo=timezone.utc))
    )
    state.kis_broker = broker_stub(_FakePnlBroker(overseas=[
        _FakePnlExecution("AAPL", "1", "20260701", "buy", 5, 150.0),
        _FakePnlExecution("AAPL", "2", "20260710", "sell", 5, 160.0),
    ]))

    resp = client.get("/api/pnl", params={"start": "2026-07-01", "end": "2026-07-31"})
    body = resp.json()
    assert body["fee_unavailable_overseas"] is True
    assert body["rows"][0]["fee_measured"] is False


# ---------------------------------------------------------------------------
# /api/pnl fee/tax enrichment (2026-09-07) — app.brokers.kis.domestic.order.
# fetch_domestic_fee_tax is an OPTIONAL broker method (getattr-guarded): a
# broker stub without it (every _FakePnlBroker above) must keep working
# exactly as before (already proven by the tests above still passing
# unmodified), and one that HAS it must be called with only the tickers that
# had a reported sell, and its unresolved-group count must surface as
# fee_partial.
# ---------------------------------------------------------------------------


class _FakePnlBrokerWithFeeProbe(_FakePnlBroker):
    def __init__(self, domestic=None, overseas=None, unresolved=0):
        super().__init__(domestic=domestic, overseas=overseas)
        self.probe_calls: list[dict] = []
        self._unresolved = unresolved

    async def fetch_domestic_fee_tax(self, executions, *, codes, max_groups, today):
        self.probe_calls.append({"codes": set(codes), "max_groups": max_groups, "today": today})
        return self._unresolved


async def test_pnl_calls_fee_probe_only_for_tickers_with_a_reported_sell(client, monkeypatch):
    async def fake_get_display_name(ticker):
        return ticker

    monkeypatch.setattr(state, "get_display_name", fake_get_display_name)
    monkeypatch.setattr(
        portfolio, "get_clock", lambda: FakeClock(datetime(2026, 8, 5, tzinfo=timezone.utc))
    )
    broker = _FakePnlBrokerWithFeeProbe(domestic=[
        _FakePnlExecution("005930", "1", "20260701", "buy", 10, 70_000.0),
        _FakePnlExecution("005930", "2", "20260710", "sell", 10, 75_000.0),
        # Bought but never sold in-window — must not trigger a probe for it.
        _FakePnlExecution("000660", "3", "20260705", "buy", 5, 100_000.0),
    ])
    state.kis_broker = broker_stub(broker)

    resp = client.get("/api/pnl", params={"start": "2026-07-01", "end": "2026-07-31"})
    assert resp.status_code == 200
    assert len(broker.probe_calls) == 1
    assert broker.probe_calls[0]["codes"] == {"005930"}
    assert broker.probe_calls[0]["today"] == "20260805"  # YYYYMMDD, matching KIS's own date format
    assert resp.json()["fee_partial"] is False


async def test_pnl_skips_fee_probe_when_no_domestic_sell_in_window(client, monkeypatch):
    monkeypatch.setattr(
        portfolio, "get_clock", lambda: FakeClock(datetime(2026, 8, 5, tzinfo=timezone.utc))
    )
    broker = _FakePnlBrokerWithFeeProbe(domestic=[
        _FakePnlExecution("005930", "1", "20260701", "buy", 10, 70_000.0),
    ])
    state.kis_broker = broker_stub(broker)

    resp = client.get("/api/pnl", params={"start": "2026-07-01", "end": "2026-07-31"})
    assert resp.status_code == 200
    assert broker.probe_calls == []  # nothing sold in-window — never worth a probe


async def test_pnl_surfaces_fee_partial_when_probe_reports_unresolved_groups(client, monkeypatch):
    async def fake_get_display_name(ticker):
        return ticker

    monkeypatch.setattr(state, "get_display_name", fake_get_display_name)
    monkeypatch.setattr(
        portfolio, "get_clock", lambda: FakeClock(datetime(2026, 8, 5, tzinfo=timezone.utc))
    )
    broker = _FakePnlBrokerWithFeeProbe(
        domestic=[
            _FakePnlExecution("005930", "1", "20260701", "buy", 10, 70_000.0),
            _FakePnlExecution("005930", "2", "20260710", "sell", 10, 75_000.0),
        ],
        unresolved=1,
    )
    state.kis_broker = broker_stub(broker)

    resp = client.get("/api/pnl", params={"start": "2026-07-01", "end": "2026-07-31"})
    assert resp.status_code == 200
    assert resp.json()["fee_partial"] is True


async def test_pnl_fee_partial_false_by_default_without_a_probe_capable_broker(client, monkeypatch):
    # Every existing _FakePnlBroker-based test above already proves this
    # path doesn't crash; this pins the actual response field.
    monkeypatch.setattr(
        portfolio, "get_clock", lambda: FakeClock(datetime(2026, 8, 5, tzinfo=timezone.utc))
    )
    state.kis_broker = broker_stub(_FakePnlBroker())
    resp = client.get("/api/pnl")
    assert resp.status_code == 200
    assert resp.json()["fee_partial"] is False


async def test_pnl_fee_probe_exception_sets_fee_partial_without_failing_the_request(client, monkeypatch):
    async def fake_get_display_name(ticker):
        return ticker

    monkeypatch.setattr(state, "get_display_name", fake_get_display_name)
    monkeypatch.setattr(
        portfolio, "get_clock", lambda: FakeClock(datetime(2026, 8, 5, tzinfo=timezone.utc))
    )

    class _ExplodingFeeBroker(_FakePnlBroker):
        async def fetch_domestic_fee_tax(self, executions, *, codes, max_groups, today):
            raise RuntimeError("KIS probe blew up")

    state.kis_broker = broker_stub(_ExplodingFeeBroker(domestic=[
        _FakePnlExecution("005930", "1", "20260701", "buy", 10, 70_000.0),
        _FakePnlExecution("005930", "2", "20260710", "sell", 10, 75_000.0),
    ]))

    resp = client.get("/api/pnl", params={"start": "2026-07-01", "end": "2026-07-31"})
    assert resp.status_code == 200  # the report itself must still succeed
    assert resp.json()["fee_partial"] is True
    assert len(resp.json()["rows"]) == 1
