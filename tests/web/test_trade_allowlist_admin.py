"""app.web.trade_allowlist_admin.reseed_trade_allowlist — bulk reseed reuses
the same validated add/remove path as the dashboard, no real DB/network."""

from __future__ import annotations

import pytest

from app.config import settings
from app.web import state, trade_allowlist_admin, trade_allowlist_store


class _FakeDisplayInfo:
    def __init__(self, name="Test Co.", previous_close=100.0, currency="KRW", quote_type="EQUITY"):
        self.name, self.previous_close, self.currency = name, previous_close, currency
        self.quote_type = quote_type


@pytest.fixture(autouse=True)
def _reseed_stubs(monkeypatch):
    monkeypatch.setattr(settings, "trade_allowlist_domestic", [])
    monkeypatch.setattr(settings, "trade_allowlist_overseas", [])
    monkeypatch.setattr(settings, "trade_allowlist_auto_added", [])
    monkeypatch.setattr(settings, "watchlist", [])
    monkeypatch.setattr(state, "_market_feed", None)

    async def fake_display_info(ticker):
        return _FakeDisplayInfo()

    async def fake_seed(ticker):
        return None

    async def fake_save(key, tickers):
        return None

    async def fake_load():
        return None  # no real DB — current settings values are the "persisted" state

    monkeypatch.setattr(state, "get_ticker_display_info", fake_display_info)
    monkeypatch.setattr(state, "seed_ticker_history", fake_seed)
    monkeypatch.setattr(state.trade_allowlist_store, "save", fake_save)
    monkeypatch.setattr(trade_allowlist_store, "load", fake_load)


async def test_reseed_adds_all_tickers_to_an_empty_market():
    result = await trade_allowlist_admin.reseed_trade_allowlist("domestic", ["005930.KS", "000660.KS"])

    assert result["added"] == ["000660.KS", "005930.KS"]
    assert result["removed"] == []
    assert result["rejected"] == {}
    assert set(settings.trade_allowlist_domestic) == {"005930.KS", "000660.KS"}


async def test_reseed_removes_tickers_not_in_the_new_list(monkeypatch):
    monkeypatch.setattr(settings, "trade_allowlist_domestic", ["005930.KS", "000660.KS"])

    result = await trade_allowlist_admin.reseed_trade_allowlist("domestic", ["005930.KS"])

    assert result["removed"] == ["000660.KS"]
    assert result["added"] == []
    assert result["kept"] == ["005930.KS"]
    assert settings.trade_allowlist_domestic == ["005930.KS"]


async def test_reseed_leaves_unchanged_tickers_alone(monkeypatch):
    monkeypatch.setattr(settings, "trade_allowlist_domestic", ["005930.KS"])

    result = await trade_allowlist_admin.reseed_trade_allowlist("domestic", ["005930.KS", "000660.KS"])

    assert result["kept"] == ["005930.KS"]
    assert result["added"] == ["000660.KS"]
    assert result["removed"] == []


async def test_reseed_reports_rejected_ticker_without_aborting_the_rest(monkeypatch):
    async def fake_display_info(ticker):
        if ticker == "BADTICKER!!":
            raise AssertionError("should never reach display-info lookup for a malformed ticker")
        return _FakeDisplayInfo()

    monkeypatch.setattr(state, "get_ticker_display_info", fake_display_info)

    result = await trade_allowlist_admin.reseed_trade_allowlist("domestic", ["005930.KS", "BADTICKER!!"])

    assert result["added"] == ["005930.KS"]
    assert "BADTICKER!!" in result["rejected"]
    assert settings.trade_allowlist_domestic == ["005930.KS"]


async def test_reseed_overseas_market_only_touches_overseas_list(monkeypatch):
    monkeypatch.setattr(settings, "trade_allowlist_domestic", ["005930.KS"])
    monkeypatch.setattr(settings, "trade_allowlist_overseas", ["TSLA"])

    result = await trade_allowlist_admin.reseed_trade_allowlist("overseas", ["AAPL"])

    assert result["added"] == ["AAPL"]
    assert result["removed"] == ["TSLA"]
    assert settings.trade_allowlist_overseas == ["AAPL"]
    assert settings.trade_allowlist_domestic == ["005930.KS"]  # untouched
