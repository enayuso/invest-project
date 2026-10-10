"""app.live.last_price.freshest_tick — the freshest-known-price read shared
by the tick-based exit monitor (`app.live.exit_monitor`) and the forced-exit
ledger write (`app.execution.ledger.force_exit_position`, 2026-08-27).

These four cases used to be reachable only through
`check_exit_for_ticker`'s own branching (they lived on the private
`exit_monitor._live_price`); with the function extracted they are testable
directly. The point of every one of them is that the two markets read
DIFFERENT caches — domestic goes straight to the KIS websocket feed, never
`app.live.store`, because `app.live.poller` only samples that feed into the
store once per 60s (2026-08-10 fix)."""

from __future__ import annotations

from datetime import datetime, timezone

from app.live import last_price
from app.live.store import Tick


class _FakeTick:
    def __init__(self, price: float):
        self.price = price


class _FakeFeed:
    def __init__(self, ticks: dict, received_at: dict):
        self.ticks = ticks
        self._received_at = received_at

    def latest_tick_received_at(self, code):
        return self._received_at.get(code)


def test_domestic_reads_the_websocket_feed_not_the_store(monkeypatch):
    received_at = datetime.now(timezone.utc)
    feed = _FakeFeed({"005930": _FakeTick(21000.0)}, {"005930": received_at})
    monkeypatch.setattr(last_price.kis_market_status, "get_active_feed", lambda: feed)

    def should_not_be_called(ticker):
        raise AssertionError("domestic ticker must not read app.live.store")

    monkeypatch.setattr(last_price, "latest", should_not_be_called)

    assert last_price.freshest_tick("005930.KS") == (21000.0, received_at)


def test_domestic_with_no_active_feed_returns_none(monkeypatch):
    monkeypatch.setattr(last_price.kis_market_status, "get_active_feed", lambda: None)
    assert last_price.freshest_tick("005930.KS") is None


def test_overseas_reads_the_store(monkeypatch):
    stamped = datetime.now(timezone.utc)
    monkeypatch.setattr(
        last_price, "latest", lambda ticker: Tick(price=101.5, volume=0, timestamp=stamped)
    )
    assert last_price.freshest_tick("AAPL") == (101.5, stamped)


def test_unknown_ticker_returns_none(monkeypatch):
    monkeypatch.setattr(last_price.kis_market_status, "get_active_feed", lambda: None)
    monkeypatch.setattr(last_price, "latest", lambda ticker: None)
    assert last_price.freshest_tick("AAPL") is None
