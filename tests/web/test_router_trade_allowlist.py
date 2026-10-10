"""trade_allowlist router — TestClient against a minimal app (just this
router), every dependency stubbed, no network/DB (E-stage)."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.config import settings
from app.web import state
from app.web.routers import trade_allowlist

# Captured before the autouse `_clean` fixture below ever runs (it stubs
# state.held_shares_for_removal for every test in this file) — needed by the
# one test that must exercise the REAL guard end-to-end.
_real_held_shares_for_removal = state.held_shares_for_removal


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(trade_allowlist.router)
    return TestClient(app, base_url="http://127.0.0.1:8094")


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.setattr(settings, "trade_allowlist_domestic", [])
    monkeypatch.setattr(settings, "trade_allowlist_overseas", ["AAPL"])
    state._display_info.clear()

    async def fake_save(key, tickers):
        return None

    # Without this, every add/remove test here would perform a REAL write
    # to the shared dev Postgres's trade_allowlist_state table, breaking
    # this file's own "no network/DB" contract (see its module docstring)
    # and polluting what the actual dashboard restores on its next restart.
    monkeypatch.setattr(state.trade_allowlist_store, "save", fake_save)

    # The remove-endpoint's held-position guard (state.held_shares_for_removal)
    # queries the ledger (Postgres) and the KIS account — same "no network/DB"
    # contract as above. Default: not held, confirmed — the guard passes
    # through, matching every existing remove test's expectation below.
    async def fake_held(ticker):
        return state.HeldCheck(0, True, "none")

    monkeypatch.setattr(state, "held_shares_for_removal", fake_held)
    yield
    state._display_info.clear()


class _FakeInfo:
    def __init__(self, name, previous_close=100.0, currency="USD", quote_type="EQUITY"):
        self.name, self.previous_close, self.currency = name, previous_close, currency
        self.quote_type = quote_type


def test_add_returns_exists_when_already_on_trade_allowlist(client):
    resp = client.post("/api/trade-allowlist/add", json={"ticker": "AAPL"})
    assert resp.status_code == 200
    # No display_info cached for AAPL in this test, so name falls back to
    # the bare ticker rather than paying for a network fetch on a no-op.
    assert resp.json() == {"status": "exists", "ticker": "AAPL", "name": "AAPL"}


async def test_manual_add_clears_the_auto_added_marker_for_a_ticker_already_on_the_list(client):
    """2026-08-26: an auto-added ticker (speculative — added on the AI's own
    BUY decision, per state.mark_trade_allowlist_auto_added's docstring)
    that a human then explicitly re-adds must stop being eligible for
    reconcile_trade_allowlist's automatic pruning. The interesting status
    here is "exists" (AAPL is already on the fixture's allowlist) — the
    marker must clear on THAT path, not just on a fresh "added"."""
    await state.mark_trade_allowlist_auto_added("AAPL")
    assert state.is_trade_allowlist_auto_added("AAPL") is True

    resp = client.post("/api/trade-allowlist/add", json={"ticker": "AAPL"})

    assert resp.status_code == 200
    assert resp.json()["status"] == "exists"
    assert state.is_trade_allowlist_auto_added("AAPL") is False


async def test_manual_add_of_a_brand_new_ticker_leaves_no_auto_added_marker(client, monkeypatch):
    async def fake_display_info(ticker):
        return _FakeInfo("Tesla Inc.")

    async def fake_seed(ticker):
        return None

    monkeypatch.setattr(state, "get_ticker_display_info", fake_display_info)
    monkeypatch.setattr(state, "seed_ticker_history", fake_seed)

    resp = client.post("/api/trade-allowlist/add", json={"ticker": "tsla"})

    assert resp.status_code == 200
    assert resp.json()["status"] == "added"
    assert state.is_trade_allowlist_auto_added("TSLA") is False


async def test_add_rejects_cash_or_bond_fund(client, monkeypatch):
    # 2026-09-01: 481050.KS "KODEX CD1년금리플러스액티브(합성)" was auto-added
    # after a BUY-ish debate. state.add_ticker_to_trade_allowlist now refuses
    # it and the router maps that to a structured 422.
    async def fake_display_info(ticker):
        return _FakeInfo("KODEX CD1년금리플러스액티브(합성)", currency="KRW")

    async def fake_seed(ticker):
        return None

    monkeypatch.setattr(state, "get_ticker_display_info", fake_display_info)
    monkeypatch.setattr(state, "seed_ticker_history", fake_seed)

    resp = client.post("/api/trade-allowlist/add", json={"ticker": "481050.KS"})

    assert resp.status_code == 422
    assert resp.json()["detail"]["error_code"] == "cash_or_bond_fund"
    assert "481050.KS" not in settings.trade_allowlist


async def test_a_rejected_add_leaves_the_auto_added_marker_in_place(client, monkeypatch):
    """Pins that unmark_trade_allowlist_auto_added is only reached PAST the
    try/except chain — a rejected add (leveraged, unresolvable, etc.) must
    not clear a marker that a genuinely-speculative auto-add still owns."""
    async def fake_add(ticker):
        raise state.LeveragedTickerError(f"{ticker} is a leveraged/inverse product")

    monkeypatch.setattr(state, "add_ticker_to_trade_allowlist", fake_add)
    monkeypatch.setattr(settings, "trade_allowlist_overseas", ["AAPL", "SOXL"])
    await state.mark_trade_allowlist_auto_added("SOXL")

    resp = client.post("/api/trade-allowlist/add", json={"ticker": "SOXL"})

    assert resp.status_code == 422
    assert state.is_trade_allowlist_auto_added("SOXL") is True


async def test_watchlist_add_does_not_clear_the_auto_added_marker(client, monkeypatch):
    """Policy pin: this fix is scoped to the TRADE allowlist. The watchlist
    is the lighter "관심종목" concept with no trading permission of its own,
    so a watchlist add is not the "human confirmed this trade" signal
    trade_allowlist_auto_added exists to track."""
    await state.mark_trade_allowlist_auto_added("AAPL")

    async def fake_display_info(ticker):
        return _FakeInfo("Apple Inc.")

    monkeypatch.setattr(state, "get_ticker_display_info", fake_display_info)

    resp = client.post("/api/watchlist/add", json={"ticker": "AAPL"})

    assert resp.status_code == 200
    assert state.is_trade_allowlist_auto_added("AAPL") is True


def test_add_rejects_empty_ticker(client):
    resp = client.post("/api/trade-allowlist/add", json={"ticker": "   "})
    assert resp.status_code == 400


def test_add_rejects_invalid_ticker_format(client):
    resp = client.post("/api/trade-allowlist/add", json={"ticker": "bad;ticker"})
    assert resp.status_code == 422


async def test_add_new_ticker_succeeds(client, monkeypatch):
    async def fake_display_info(ticker):
        return _FakeInfo("Tesla Inc.")

    async def fake_seed(ticker):
        return None

    monkeypatch.setattr(state, "get_ticker_display_info", fake_display_info)
    monkeypatch.setattr(state, "seed_ticker_history", fake_seed)

    resp = client.post("/api/trade-allowlist/add", json={"ticker": "tsla"})
    assert resp.status_code == 200
    assert resp.json() == {"status": "added", "ticker": "TSLA", "name": "Tesla Inc."}
    assert "TSLA" in settings.trade_allowlist


def test_add_unresolvable_ticker_returns_structured_422(client, monkeypatch):
    async def fake_display_info(ticker):
        raise RuntimeError("no data for ticker")

    monkeypatch.setattr(state, "get_ticker_display_info", fake_display_info)
    resp = client.post("/api/trade-allowlist/add", json={"ticker": "NOPE"})
    assert resp.status_code == 422
    assert resp.json()["detail"]["error_code"] == "ticker_not_found"


def test_remove_drops_ticker_and_display_info(client):
    state._display_info["AAPL"] = _FakeInfo("Apple Inc.")
    resp = client.post("/api/trade-allowlist/remove", json={"ticker": "AAPL"})
    assert resp.status_code == 200
    assert resp.json() == {"status": "removed", "ticker": "AAPL"}
    assert "AAPL" not in settings.trade_allowlist
    assert "AAPL" not in state._display_info


def test_remove_blocked_when_position_held(client, monkeypatch):
    async def fake_held(ticker):
        return state.HeldCheck(7, True, "ledger")

    monkeypatch.setattr(state, "held_shares_for_removal", fake_held)
    resp = client.post("/api/trade-allowlist/remove", json={"ticker": "AAPL"})
    assert resp.status_code == 409
    assert resp.json()["detail"]["error_code"] == "held_position_exists"
    assert resp.json()["detail"]["shares"] == 7
    assert "AAPL" in settings.trade_allowlist


def test_remove_blocked_when_holdings_unknown(client, monkeypatch):
    async def fake_held(ticker):
        return state.HeldCheck(0, False, "unknown")

    monkeypatch.setattr(state, "held_shares_for_removal", fake_held)
    resp = client.post("/api/trade-allowlist/remove", json={"ticker": "AAPL"})
    assert resp.status_code == 409
    assert resp.json()["detail"]["error_code"] == "holdings_unknown"
    assert "AAPL" in settings.trade_allowlist


def test_remove_force_bypasses_held_guard(client, monkeypatch):
    async def fake_held(ticker):
        return state.HeldCheck(7, True, "both")

    monkeypatch.setattr(state, "held_shares_for_removal", fake_held)
    resp = client.post("/api/trade-allowlist/remove", json={"ticker": "AAPL", "force": True})
    assert resp.status_code == 200
    assert resp.json() == {"status": "removed", "ticker": "AAPL"}
    assert "AAPL" not in settings.trade_allowlist


def test_remove_without_force_field_defaults_to_guarded(client, monkeypatch):
    async def fake_held(ticker):
        return state.HeldCheck(3, True, "kis")

    monkeypatch.setattr(state, "held_shares_for_removal", fake_held)
    resp = client.post("/api/trade-allowlist/remove", json={"ticker": "AAPL"})
    assert resp.status_code == 409
    assert "AAPL" in settings.trade_allowlist


def test_remove_returns_409_holdings_unknown_when_only_the_overseas_leg_failed(client, monkeypatch):
    """End-to-end through the REAL held_shares_for_removal (the tests above
    all stub it out, so none of them exercise this) — 2026-08-26: a partial
    KIS snapshot (only the overseas balance TR failed) used to let this
    guard pass through as "confirmed not held" instead of 409'ing."""
    async def fake_snapshot():
        return {"overseas_ok": False}, {}

    async def fake_get_position(ticker):
        class P:
            shares = 0

        return P()

    async def fake_broker():
        return object()

    monkeypatch.setattr(state, "held_shares_for_removal", _real_held_shares_for_removal)
    monkeypatch.setattr(state, "kis_account_snapshot", fake_snapshot)
    monkeypatch.setattr(state, "get_position", fake_get_position)
    monkeypatch.setattr(state, "kis_broker", fake_broker)

    resp = client.post("/api/trade-allowlist/remove", json={"ticker": "AAPL"})

    assert resp.status_code == 409
    assert resp.json()["detail"]["error_code"] == "holdings_unknown"
    assert "AAPL" in settings.trade_allowlist


def test_trade_allowlist_info_uses_display_info_when_available(client):
    state._display_info["AAPL"] = _FakeInfo("Apple Inc.", previous_close=150.0, currency="USD")
    resp = client.get("/api/trade-allowlist-info")
    assert resp.status_code == 200
    rows = resp.json()
    assert rows[0]["symbol"] == "AAPL"
    assert rows[0]["name"] == "Apple Inc."
    assert rows[0]["previous_close"] == 150.0


def test_trade_allowlist_info_falls_back_to_bare_ticker_when_no_display_info(client):
    resp = client.get("/api/trade-allowlist-info")
    rows = resp.json()
    assert rows[0]["symbol"] == "AAPL"
    assert rows[0]["name"] == "AAPL"
    assert rows[0]["previous_close"] is None


# ---- watchlist -------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_watchlist(monkeypatch):
    monkeypatch.setattr(settings, "watchlist", ["AAPL"])


def test_watchlist_add_returns_exists_when_already_on_watchlist(client):
    resp = client.post("/api/watchlist/add", json={"ticker": "AAPL"})
    assert resp.status_code == 200
    assert resp.json() == {"status": "exists", "ticker": "AAPL", "name": "AAPL"}


async def test_watchlist_add_new_ticker_does_not_apply_leverage_filter(client, monkeypatch):
    # The watchlist has no leverage/inverse rejection — unlike
    # trade_allowlist_add, this is just tracking interest, not a trading
    # decision.
    async def fake_display_info(ticker):
        return _FakeInfo("ProShares UltraPro Short QQQ")

    monkeypatch.setattr(state, "get_ticker_display_info", fake_display_info)
    resp = client.post("/api/watchlist/add", json={"ticker": "sqqq"})
    assert resp.status_code == 200
    assert resp.json()["status"] == "added"
    assert "SQQQ" in settings.watchlist


def test_watchlist_remove_blocked_while_on_trade_allowlist(client):
    monkeypatch_ticker = "AAPL"
    resp = client.post("/api/watchlist/remove", json={"ticker": monkeypatch_ticker})
    assert resp.status_code == 409
    assert resp.json()["detail"]["error_code"] == "still_on_trade_allowlist"
    assert "AAPL" in settings.watchlist


def test_watchlist_remove_succeeds_once_off_trade_allowlist(client, monkeypatch):
    monkeypatch.setattr(settings, "trade_allowlist_domestic", [])
    monkeypatch.setattr(settings, "trade_allowlist_overseas", [])
    monkeypatch.setattr(settings, "watchlist", ["AAPL"])
    resp = client.post("/api/watchlist/remove", json={"ticker": "AAPL"})
    assert resp.status_code == 200
    assert resp.json() == {"status": "removed", "ticker": "AAPL"}
    assert "AAPL" not in settings.watchlist


def test_watchlist_info_reports_trade_allowlist_membership(client):
    state._display_info["AAPL"] = _FakeInfo("Apple Inc.", previous_close=150.0, currency="USD")
    resp = client.get("/api/watchlist-info")
    rows = resp.json()
    assert rows[0]["symbol"] == "AAPL"
    assert rows[0]["on_trade_allowlist"] is True


# ---------------------------------------------------------------------------
# /api/search-tickers — name-or-code search backing the header search box
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _fake_masters(monkeypatch):
    from app.live import krx_master, overseas_master

    async def fake_ensure_loaded():
        return None

    monkeypatch.setattr(krx_master, "ensure_loaded", fake_ensure_loaded)
    monkeypatch.setattr(overseas_master, "ensure_loaded", fake_ensure_loaded)
    monkeypatch.setattr(krx_master, "_cache", {"005930": "삼성전자", "000660": "SK하이닉스"})
    monkeypatch.setattr(overseas_master, "_cache", {"AAPL": "애플", "MSFT": "마이크로소프트"})
    yield


def test_search_tickers_matches_domestic_name(client):
    resp = client.get("/api/search-tickers", params={"q": "삼성"})
    assert resp.status_code == 200
    assert resp.json() == [{"symbol": "005930.KS", "name": "삼성전자", "market": "domestic", "tradable": True}]


def test_search_tickers_matches_overseas_name(client):
    resp = client.get("/api/search-tickers", params={"q": "애플"})
    assert resp.json() == [{"symbol": "AAPL", "name": "애플", "market": "overseas", "tradable": True}]


def test_search_tickers_matches_bare_code(client):
    resp = client.get("/api/search-tickers", params={"q": "0059"})
    assert resp.json() == [{"symbol": "005930.KS", "name": "삼성전자", "market": "domestic", "tradable": True}]


def test_search_tickers_empty_query_returns_nothing(client):
    resp = client.get("/api/search-tickers", params={"q": "  "})
    assert resp.json() == []


def test_search_tickers_no_match_returns_empty_list(client):
    resp = client.get("/api/search-tickers", params={"q": "존재하지않는회사"})
    assert resp.json() == []


def test_search_tickers_caps_combined_results_at_limit(client, monkeypatch):
    from app.live import krx_master

    monkeypatch.setattr(krx_master, "_cache", {f"00{i}": f"삼성테스트{i}" for i in range(10)})
    resp = client.get("/api/search-tickers", params={"q": "삼성", "limit": 3})
    assert len(resp.json()) == 3


def test_search_tickers_ranks_exact_name_before_substring(client, monkeypatch):
    from app.live import krx_master

    # master-file order puts 삼성화재/삼성제약 ahead of 삼성전자 by code number —
    # ranking must still float the exact-tier / prefix names to the top.
    monkeypatch.setattr(krx_master, "_cache", {
        "000810": "삼성화재",
        "001360": "삼성제약",
        "005930": "삼성전자",
        "207940": "삼성바이오로직스",
    })
    names = [r["name"] for r in client.get("/api/search-tickers", params={"q": "삼성전자"}).json()]
    assert names[0] == "삼성전자"


def test_search_tickers_deprioritizes_preferred_shares(client, monkeypatch):
    from app.live import krx_master

    monkeypatch.setattr(krx_master, "_cache", {"005935": "삼성전자우", "005930": "삼성전자"})
    names = [r["name"] for r in client.get("/api/search-tickers", params={"q": "삼성전자"}).json()]
    assert names.index("삼성전자") < names.index("삼성전자우")


def test_search_tickers_marks_leveraged_not_tradable(client, monkeypatch):
    from app.live import krx_master

    monkeypatch.setattr(krx_master, "_cache", {"122630": "KODEX 레버리지", "069500": "KODEX 200"})
    rows = client.get("/api/search-tickers", params={"q": "KODEX"}).json()
    by_name = {r["name"]: r for r in rows}
    assert by_name["KODEX 레버리지"]["tradable"] is False
    assert by_name["KODEX 200"]["tradable"] is True
    # ...and the tradable one ranks first
    assert rows[0]["name"] == "KODEX 200"


def test_search_tickers_gives_overseas_slots_when_domestic_fills_limit(client, monkeypatch):
    from app.live import krx_master, overseas_master

    monkeypatch.setattr(krx_master, "_cache", {f"10{i:02d}": f"테스트전자{i}" for i in range(20)})
    monkeypatch.setattr(overseas_master, "_cache", {"TSLA": "테스트모터스"})
    rows = client.get("/api/search-tickers", params={"q": "테스트", "limit": 5}).json()
    assert any(r["market"] == "overseas" for r in rows)


def test_search_tickers_marks_cash_bond_fund_not_tradable(client, monkeypatch):
    from app.live import krx_master

    monkeypatch.setattr(krx_master, "_cache", {
        "481050": "KODEX CD1년금리플러스액티브(합성)",
        "069500": "KODEX 200",
    })
    rows = client.get("/api/search-tickers", params={"q": "KODEX"}).json()
    by_name = {r["name"]: r for r in rows}
    assert by_name["KODEX CD1년금리플러스액티브(합성)"]["tradable"] is False
    assert by_name["KODEX 200"]["tradable"] is True
    assert rows[0]["name"] == "KODEX 200"
