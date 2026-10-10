"""app.web.state — cache helper, display-name resolution, and the KIS account
snapshot builder. Every KIS/yfinance call is faked; no network/DB (E-stage)."""

from __future__ import annotations

import asyncio
import time

import pytest

from datetime import date, datetime

from app.config import account_mode_scope, current_account_mode, settings
from app.execution.reconciliation import TradeDiscrepancy
from app.live import ticker_names
from app.web import state
from conftest import broker_stub


@pytest.fixture(autouse=True)
def _clean_state():
    state._cache.clear()
    state._cache_locks.clear()
    state._display_info.clear()
    state._candidate_cb_codes.clear()
    orig_broker = state.kis_broker
    orig_feed = state._market_feed
    yield
    state._cache.clear()
    state._cache_locks.clear()
    state._display_info.clear()
    state._candidate_cb_codes.clear()
    state.kis_broker = orig_broker
    state._market_feed = orig_feed


# ---------------------------------------------------------------------------
# cached / invalidate_cache
# ---------------------------------------------------------------------------


async def test_cached_calls_factory_once_within_ttl():
    calls = []

    async def factory():
        calls.append(1)
        return "value"

    first = await state.cached("k", 60, factory)
    second = await state.cached("k", 60, factory)
    assert first == second == "value"
    assert len(calls) == 1


async def test_cached_refetches_after_ttl_expires(monkeypatch):
    calls = []

    async def factory():
        calls.append(1)
        return f"value-{len(calls)}"

    clock = {"t": 100.0}
    monkeypatch.setattr(state.time, "monotonic", lambda: clock["t"])
    first = await state.cached("k", 5, factory)
    clock["t"] = 200.0  # advance well past the 5s ttl before the second call
    second = await state.cached("k", 5, factory)
    assert first == "value-1"
    assert second == "value-2"


async def test_invalidate_cache_drops_matching_prefixes():
    state._cache["account"] = (0.0, "stale account")
    state._cache["executions"] = (0.0, "stale executions")
    state._cache["candles:AAPL:D"] = (0.0, "unrelated")
    state.invalidate_cache("account", "executions")
    assert "account" not in state._cache
    assert "executions" not in state._cache
    assert "candles:AAPL:D" in state._cache


# ---------------------------------------------------------------------------
# get_display_name / trade_allowlist_names
# ---------------------------------------------------------------------------


class _FakeInfo:
    def __init__(self, name):
        self.name = name


async def test_get_display_name_uses_prepopulated_info():
    state._display_info["AAPL"] = _FakeInfo("Apple Inc.")
    assert await state.get_display_name("AAPL") == "Apple Inc."


async def test_get_display_name_lazily_fetches_and_caches(monkeypatch):
    calls = []

    async def fake_display_info(ticker):
        calls.append(ticker)
        return _FakeInfo("Tesla Inc.")

    monkeypatch.setattr(state, "get_ticker_display_info", fake_display_info)
    first = await state.get_display_name("TSLA")
    second = await state.get_display_name("TSLA")
    assert first == second == "Tesla Inc."
    assert calls == ["TSLA"]  # only fetched once, then cached


async def test_get_display_name_falls_back_to_ticker_on_failure(monkeypatch):
    async def fake_display_info(ticker):
        raise RuntimeError("no data")

    monkeypatch.setattr(state, "get_ticker_display_info", fake_display_info)
    assert await state.get_display_name("XYZ") == "XYZ"


async def test_get_display_name_prefers_kis_korean_name_for_krx_candidate(monkeypatch):
    # 093240.KS discovered via krx_top_n (volume scan), not the fixed
    # trade allowlist — yfinance's unreadable English legal name shouldn't win.
    monkeypatch.setattr(ticker_names, "_cache", {"093240.KS": "형지엘리트"})

    async def fake_display_info(ticker):
        raise AssertionError("yfinance should not be called when a KIS name is cached")

    monkeypatch.setattr(state, "get_ticker_display_info", fake_display_info)
    assert await state.get_display_name("093240.KS") == "형지엘리트"


async def test_get_display_name_kis_cache_wins_even_over_a_prepopulated_trade_allowlist_entry(monkeypatch):
    # Real incident (2026-07-23): 064260.KS's _display_info got cached early
    # with yfinance's name for a completely different, real company
    # ("에코프로비엠" instead of the actual 다날) — under the old "KIS name only
    # checked if _display_info is still empty" ordering, that wrong name
    # stuck forever even after krx_top_n later learned the correct
    # KIS-sourced name for the same ticker. KIS's hts_kor_isnm is
    # authoritative for an exact KRX code (no cross-symbol confusion risk
    # the way yfinance has shown twice now), so it must win regardless of
    # cache order.
    state._display_info["005930.KS"] = _FakeInfo("Samsung Electronics")
    monkeypatch.setattr(ticker_names, "_cache", {"005930.KS": "삼성전자"})
    assert await state.get_display_name("005930.KS") == "삼성전자"


def test_trade_allowlist_names_uses_display_info_when_present(monkeypatch):
    monkeypatch.setattr(settings, "trade_allowlist_domestic", [])
    monkeypatch.setattr(settings, "trade_allowlist_overseas", ["AAPL", "TSLA"])
    state._display_info["AAPL"] = _FakeInfo("Apple Inc.")
    result = state.trade_allowlist_names()
    assert result == {"AAPL": "Apple Inc.", "TSLA": "TSLA"}


# ---------------------------------------------------------------------------
# kis_account_snapshot / _kis_account_snapshot_build
# ---------------------------------------------------------------------------


async def test_kis_account_snapshot_returns_none_when_broker_unconfigured():
    state.kis_broker = broker_stub(None)
    account, positions = await state.kis_account_snapshot()
    assert account is None
    assert positions == {}


class _FakeBalancePosition:
    def __init__(self, code, name, quantity, avg_price, current_price, eval_amount, pnl_amount, pnl_pct):
        self.code, self.name, self.quantity = code, name, quantity
        self.avg_price, self.current_price = avg_price, current_price
        self.eval_amount, self.pnl_amount, self.pnl_pct = eval_amount, pnl_amount, pnl_pct


class _FakeBalance:
    def __init__(self, positions):
        self.positions = positions
        self.cash_deposit = 1_000_000.0
        self.total_eval_amount = 500_000.0
        self.total_pnl_amount = 20_000.0
        self.purchase_amount = 480_000.0
        self.d2_cash_deposit = 950_000.0


class _FakeBroker:
    def __init__(self, balance, overseas_fails=True, present_fails=True):
        self._balance = balance
        self._overseas_fails = overseas_fails
        self._present_fails = present_fails

    async def fetch_balance(self):
        return self._balance

    async def fetch_overseas_balance(self):
        if self._overseas_fails:
            raise RuntimeError("overseas unavailable")
        raise NotImplementedError

    async def fetch_overseas_present_balance(self, currency):
        if self._present_fails:
            raise RuntimeError("present-balance unavailable")
        raise NotImplementedError


async def test_kis_account_snapshot_build_reports_domestic_positions_and_degrades_overseas(monkeypatch):
    position = _FakeBalancePosition("005930", "삼성전자", 10, 70000.0, 74000.0, 740000.0, 40000.0, 5.71)
    fake_broker = _FakeBroker(_FakeBalance([position]))
    state.kis_broker = broker_stub(fake_broker)

    async def fake_fx_rate(currency):
        return 1450.0

    monkeypatch.setattr(state, "get_usd_fx_rate", fake_fx_rate)
    monkeypatch.setattr(settings, "kis_account_mode", "paper")

    account, positions = await state._kis_account_snapshot_build()

    assert account["is_real"] is False
    # total_assets now sums d2_cash_deposit (950_000), not cash_deposit
    # (1_000_000) — see app.web.state's identical fix to
    # app.execution.portfolio_risk.real_account_snapshot.
    assert account["total_assets"] == pytest.approx(1_450_000.0)
    assert account["d2_cash_deposit"] == 950_000.0
    assert account["domestic_count"] == 1
    assert account["usd_krw_rate"] == 1450.0
    # Overseas balance failed -> degrades to zero, doesn't raise.
    assert account["overseas_eval_amount"] == 0.0
    assert positions["005930.KS"]["name"] == "삼성전자"
    # 2026-08-26: distinguishes "overseas TR failed" from "confirmed zero
    # overseas positions" — see held_shares_for_removal/reconcile_trade_
    # allowlist, both of which used to read this same overseas_fails=True
    # snapshot as "AAPL definitely not held" during an outage.
    assert account["overseas_ok"] is False


async def test_snapshot_build_reports_overseas_ok_true_when_the_account_holds_nothing_overseas(monkeypatch):
    position = _FakeBalancePosition("005930", "삼성전자", 10, 70000.0, 74000.0, 740000.0, 40000.0, 5.71)
    fake_broker = _FakeBroker(_FakeBalance([position]), overseas_fails=False, present_fails=False)

    class _EmptyOverseas:
        total_eval_amount = 0.0
        total_pnl_amount = 0.0
        purchase_amount = 0.0
        positions = []

    class _EmptyPresent:
        orderable = 0.0
        deposit = 0.0

    async def fake_overseas_balance():
        return _EmptyOverseas()

    async def fake_present_balance(currency):
        return _EmptyPresent()

    fake_broker.fetch_overseas_balance = fake_overseas_balance
    fake_broker.fetch_overseas_present_balance = fake_present_balance
    state.kis_broker = broker_stub(fake_broker)

    async def fake_fx_rate(currency):
        return 1450.0

    monkeypatch.setattr(state, "get_usd_fx_rate", fake_fx_rate)

    account, _positions = await state._kis_account_snapshot_build()

    # Genuinely queried and got zero — distinct from the TR having failed.
    assert account["overseas_ok"] is True
    assert account["overseas_count"] == 0


# ---------------------------------------------------------------------------
# overseas_trading_enabled (background-polling scope switch) — see
# app.config.overseas_trading_enabled's own comment for the domestic-only
# deployment this exists for.
# ---------------------------------------------------------------------------


class _CountingOverseasBroker:
    """Tracks whether the two overseas TRs were ever awaited, for asserting
    they are SKIPPED entirely (not just failed) when the flag is off."""

    def __init__(self, balance):
        self._balance = balance
        self.overseas_balance_calls = 0
        self.overseas_present_calls = 0

    async def fetch_balance(self):
        return self._balance

    async def fetch_overseas_balance(self):
        self.overseas_balance_calls += 1

        class _Empty:
            total_eval_amount = 0.0
            total_pnl_amount = 0.0
            purchase_amount = 0.0
            positions = []

        return _Empty()

    async def fetch_overseas_present_balance(self, currency):
        self.overseas_present_calls += 1

        class _EmptyPresent:
            orderable = 0.0
            deposit = 0.0

        return _EmptyPresent()


async def test_account_snapshot_skips_overseas_trs_when_disabled(monkeypatch):
    from app.execution import portfolio_risk

    position = _FakeBalancePosition("005930", "삼성전자", 10, 70000.0, 74000.0, 740000.0, 40000.0, 5.71)
    broker = _CountingOverseasBroker(_FakeBalance([position]))
    state.kis_broker = broker_stub(broker)
    monkeypatch.setattr(settings, "overseas_trading_enabled", False)
    monkeypatch.setattr(portfolio_risk, "_overseas_flag_contradicted_flag", False)

    account, _positions = await state._kis_account_snapshot_build()

    assert broker.overseas_balance_calls == 0
    assert broker.overseas_present_calls == 0
    assert account["overseas_enabled"] is False
    assert account["overseas_count"] == 0


async def test_account_snapshot_disabled_reports_overseas_unknown_not_flat(monkeypatch):
    from app.execution import portfolio_risk

    position = _FakeBalancePosition("005930", "삼성전자", 10, 70000.0, 74000.0, 740000.0, 40000.0, 5.71)
    broker = _CountingOverseasBroker(_FakeBalance([position]))
    state.kis_broker = broker_stub(broker)
    monkeypatch.setattr(settings, "overseas_trading_enabled", False)
    monkeypatch.setattr(portfolio_risk, "_overseas_flag_contradicted_flag", False)

    account, _positions = await state._kis_account_snapshot_build()

    # "we did not ask" must never be readable as "asked, and you hold
    # nothing" — overseas_ok stays False, same as a genuine TR failure.
    assert account["overseas_ok"] is False


async def test_account_snapshot_queries_overseas_when_enabled(monkeypatch):
    position = _FakeBalancePosition("005930", "삼성전자", 10, 70000.0, 74000.0, 740000.0, 40000.0, 5.71)
    broker = _CountingOverseasBroker(_FakeBalance([position]))
    state.kis_broker = broker_stub(broker)
    monkeypatch.setattr(settings, "overseas_trading_enabled", True)

    account, _positions = await state._kis_account_snapshot_build()

    assert broker.overseas_balance_calls == 1
    assert broker.overseas_present_calls == 1
    assert account["overseas_enabled"] is True
    assert account["overseas_ok"] is True


async def test_reconcile_trade_allowlist_does_not_prune_overseas_when_disabled(monkeypatch):
    """Regression guard: overseas_ok=False produced by the flag being OFF
    (not by a genuine TR failure) must be respected downstream exactly like
    the existing TR-failure case already is (see
    test_reconcile_does_not_prune_an_overseas_auto_added_ticker_when_the_
    overseas_leg_failed above) — a disabled background poll must never look
    like "confirmed not held" to the allowlist pruner. Exercises the real
    `_kis_account_snapshot_build` (not a stubbed snapshot) so the flag's own
    overseas_ok=False wiring is what's under test, not a hand-written double."""
    monkeypatch.setattr(settings, "trade_allowlist_overseas", ["AAPL"])
    await state.mark_trade_allowlist_auto_added("AAPL")

    position = _FakeBalancePosition("005930", "삼성전자", 10, 70000.0, 74000.0, 740000.0, 40000.0, 5.71)
    broker = _CountingOverseasBroker(_FakeBalance([position]))
    state.kis_broker = broker_stub(broker)
    monkeypatch.setattr(settings, "overseas_trading_enabled", False)

    async def fake_list_pending():
        return []

    monkeypatch.setattr(state, "list_pending_orders", fake_list_pending)

    await state.reconcile_trade_allowlist()

    assert "AAPL" in settings.trade_allowlist
    assert state.is_trade_allowlist_auto_added("AAPL") is True


async def test_kis_account_snapshot_build_returns_none_on_balance_failure():
    class BrokenBroker:
        async def fetch_balance(self):
            raise RuntimeError("token expired")

    state.kis_broker = broker_stub(BrokenBroker())
    account, positions = await state._kis_account_snapshot_build()
    assert account is None
    assert positions == {}


# ---------------------------------------------------------------------------
# kis_account_snapshot / recent_executions — cache must be scoped by account
# mode, not just by user (moved+fixed from the audit-scratch repros:
# tests/web/test_zzallow_scratch.py::test_repro_account_snapshot_cache_is_not_account_mode_scoped
# and tests/web/test_zzq_modestate_audit_scratch.py::test_F1a_.../test_F1b_...
# — a user can flip paper/real at runtime via PUT /api/account-mode with no
# restart, and state.user_key() used to namespace cache keys by user_id only,
# so the other mode's cached account snapshot/executions leaked through for
# up to the cache's TTL after a switch.)
# ---------------------------------------------------------------------------


class _PosPM:
    def __init__(self, code, quantity):
        self.code, self.name, self.quantity = code, code, quantity
        self.avg_price = self.current_price = self.eval_amount = 0.0
        self.pnl_amount = self.pnl_pct = 0.0


class _BalancePM:
    def __init__(self, positions):
        self.positions = positions
        self.cash_deposit = self.total_eval_amount = self.total_pnl_amount = 0.0
        self.purchase_amount = self.d2_cash_deposit = 0.0


class _BrokerPaperPM:
    async def fetch_balance(self):
        return _BalancePM([_PosPM("005930", 7)])

    async def fetch_overseas_balance(self):
        raise RuntimeError("n/a")

    async def fetch_overseas_present_balance(self, c):
        raise RuntimeError("n/a")


class _BrokerRealPM:
    async def fetch_balance(self):
        return _BalancePM([_PosPM("000660", 3)])

    async def fetch_overseas_balance(self):
        raise RuntimeError("n/a")

    async def fetch_overseas_present_balance(self, c):
        raise RuntimeError("n/a")


async def test_kis_account_snapshot_cache_is_scoped_by_account_mode(monkeypatch):
    async def fake_fx(c):
        return 1300.0

    monkeypatch.setattr(state, "get_usd_fx_rate", fake_fx)

    monkeypatch.setattr(state, "kis_broker", broker_stub(_BrokerPaperPM()))
    with account_mode_scope("paper"):
        _, paper_pos = await state.kis_account_snapshot()

    monkeypatch.setattr(state, "kis_broker", broker_stub(_BrokerRealPM()))
    with account_mode_scope("real"):
        _, real_pos = await state.kis_account_snapshot()

    assert list(paper_pos) == ["005930.KS"]
    assert list(real_pos) == ["000660.KS"], "real mode served the paper account's positions from cache"


async def test_kis_account_snapshot_build_is_called_again_after_a_mode_switch(monkeypatch):
    calls: list[str] = []

    async def fake_broker():
        return object()

    async def fake_build():
        mode = current_account_mode()
        calls.append(mode)
        return ({"is_real": mode == "real", "total_assets": 999.0 if mode == "real" else 1.0}, {})

    monkeypatch.setattr(state, "kis_broker", fake_broker)
    monkeypatch.setattr(state, "_kis_account_snapshot_build", fake_build)

    with account_mode_scope("paper"):
        paper_acct, _ = await state.kis_account_snapshot()
    with account_mode_scope("real"):
        real_acct, _ = await state.kis_account_snapshot()

    assert calls == ["paper", "real"]
    assert paper_acct["is_real"] is False
    assert real_acct["is_real"] is True
    assert real_acct["total_assets"] == 999.0


async def test_recent_executions_cache_is_scoped_by_account_mode(monkeypatch):
    class FakeBroker:
        def __init__(self, tag):
            self.tag = tag

        async def fetch_daily_executions(self, s, e):
            return [f"{self.tag}-domestic"]

        async def fetch_overseas_executions(self, s, e):
            return []

    async def fake_broker():
        return FakeBroker(current_account_mode())

    monkeypatch.setattr(state, "kis_broker", fake_broker)

    with account_mode_scope("paper"):
        p, _ = await state.recent_executions()
    with account_mode_scope("real"):
        r, _ = await state.recent_executions()

    assert p == ["paper-domestic"]
    assert r == ["real-domestic"], "real mode served the paper account's cached executions"


# ---------------------------------------------------------------------------
# portfolio_context
# ---------------------------------------------------------------------------


async def test_portfolio_context_returns_none_when_nothing_held(monkeypatch):
    monkeypatch.setattr(settings, "trade_allowlist_domestic", [])
    monkeypatch.setattr(settings, "trade_allowlist_overseas", [])

    async def fake_snapshot():
        return None, {}

    monkeypatch.setattr(state, "kis_account_snapshot", fake_snapshot)
    assert await state.portfolio_context() is None


async def test_portfolio_context_reports_kis_account_and_positions(monkeypatch):
    monkeypatch.setattr(settings, "trade_allowlist_domestic", [])
    monkeypatch.setattr(settings, "trade_allowlist_overseas", [])
    account = {
        "cash_deposit": 1_000_000.0, "total_eval_amount": 500_000.0,
        "total_pnl_amount": 20_000.0, "position_count": 1,
    }
    positions = {
        "005930.KS": {
            "name": "삼성전자", "quantity": 10, "avg_price": 70000.0, "current_price": 74000.0,
            "pnl_amount": 40000.0, "pnl_pct": 5.71,
        }
    }

    async def fake_snapshot():
        return account, positions

    monkeypatch.setattr(state, "kis_account_snapshot", fake_snapshot)
    result = await state.portfolio_context()
    assert "KIS paper account" in result
    assert "005930.KS 삼성전자" in result


async def test_portfolio_context_does_not_show_ghost_ledger_position_when_kis_snapshot_succeeds(monkeypatch):
    """Real incident this guards against: after a paper account reset, the
    real KIS snapshot correctly comes back empty, but the internal ledger
    can still say a ticker holds shares from before the reset. Since a
    successful KIS snapshot already covers every real holding (see
    kis_account_snapshot's docstring), that internal-ledger-only ticker must
    NOT be shown — it's a ghost position the AI can't actually act on."""
    monkeypatch.setattr(settings, "trade_allowlist_domestic", ["005930.KS"])
    monkeypatch.setattr(settings, "trade_allowlist_overseas", [])
    account = {
        "cash_deposit": 1_000_000.0, "total_eval_amount": 0.0,
        "total_pnl_amount": 0.0, "position_count": 0,
    }

    async def fake_snapshot():
        return account, {}  # real KIS snapshot succeeded: genuinely flat

    async def fake_get_position(ticker):
        class _Position:
            shares = 10  # stale internal ledger, left over from before reset
        return _Position()

    monkeypatch.setattr(state, "kis_account_snapshot", fake_snapshot)
    monkeypatch.setattr(state, "get_position", fake_get_position)
    result = await state.portfolio_context()
    assert result is not None
    assert "local paper ledger" not in result
    assert "005930.KS" not in result


async def test_portfolio_context_falls_back_to_ledger_when_kis_snapshot_unavailable(monkeypatch):
    """account is None -> the real snapshot itself couldn't be read (broker
    not connected / query failed), the one case where showing the internal
    ledger's best-effort figure is still better than nothing."""
    monkeypatch.setattr(settings, "trade_allowlist_domestic", ["005930.KS"])
    monkeypatch.setattr(settings, "trade_allowlist_overseas", [])

    async def fake_snapshot():
        return None, {}

    async def fake_get_position(ticker):
        class _Position:
            shares = 10
        return _Position()

    monkeypatch.setattr(state, "kis_account_snapshot", fake_snapshot)
    monkeypatch.setattr(state, "get_position", fake_get_position)
    result = await state.portfolio_context()
    assert result is not None
    assert "local paper ledger" in result
    assert "005930.KS" in result


# ---------------------------------------------------------------------------
# add_ticker_to_trade_allowlist — live-subscribing a newly-added KRX ticker to
# the real-time feed (circuit-breaker dynamic-subscription bug fix)
# ---------------------------------------------------------------------------


class _FakeDisplayInfo:
    def __init__(self, name="Test Co.", previous_close=100.0, currency="KRW", quote_type="EQUITY"):
        self.name, self.previous_close, self.currency = name, previous_close, currency
        self.quote_type = quote_type


class _FakeFeed:
    def __init__(self):
        self.added: list[str] = []

    async def add_code(self, code, *, orderbook: bool = False, cb_only: bool = False):
        self.added.append(code)
        return True


@pytest.fixture(autouse=True)
def _trade_allowlist_add_stubs(monkeypatch):
    monkeypatch.setattr(settings, "trade_allowlist_domestic", [])
    monkeypatch.setattr(settings, "trade_allowlist_overseas", [])
    monkeypatch.setattr(settings, "trade_allowlist_auto_added", [])

    async def fake_display_info(ticker):
        return _FakeDisplayInfo()

    async def fake_seed(ticker):
        return None

    async def fake_save(key, tickers):
        return None

    monkeypatch.setattr(state, "get_ticker_display_info", fake_display_info)
    monkeypatch.setattr(state, "seed_ticker_history", fake_seed)
    # Without this, every add/remove test below would perform a REAL write
    # to the shared dev Postgres's trade_allowlist_state table — which would
    # then get restored on the actual dashboard's next restart, polluting it
    # with test data. See trade_allowlist_store's own tests for real load/save
    # coverage.
    monkeypatch.setattr(state.trade_allowlist_store, "save", fake_save)


async def test_add_ticker_to_trade_allowlist_subscribes_krx_ticker_to_live_feed(monkeypatch):
    fake_feed = _FakeFeed()
    monkeypatch.setattr(state, "_market_feed", fake_feed)

    status, info = await state.add_ticker_to_trade_allowlist("005930.KS")

    assert status == "added"
    assert fake_feed.added == ["005930"]
    assert settings.trade_allowlist_domestic == ["005930.KS"]
    assert settings.trade_allowlist_overseas == []


async def test_add_ticker_to_trade_allowlist_skips_us_ticker(monkeypatch):
    fake_feed = _FakeFeed()
    monkeypatch.setattr(state, "_market_feed", fake_feed)

    status, info = await state.add_ticker_to_trade_allowlist("TSLA")

    assert status == "added"
    assert fake_feed.added == []  # no KRX code — not this feed's concern
    assert settings.trade_allowlist_overseas == ["TSLA"]
    assert settings.trade_allowlist_domestic == []


async def test_add_ticker_to_trade_allowlist_noop_when_no_feed_configured(monkeypatch):
    monkeypatch.setattr(state, "_market_feed", None)

    # Must not raise even though there's no feed to subscribe to.
    status, info = await state.add_ticker_to_trade_allowlist("005930.KS")
    assert status == "added"


async def test_add_ticker_to_trade_allowlist_degrades_quietly_when_feed_subscribe_fails(monkeypatch):
    class _BrokenFeed:
        async def add_code(self, code):
            raise RuntimeError("boom")

    monkeypatch.setattr(state, "_market_feed", _BrokenFeed())

    # Must not raise — the trade_allowlist add itself should still succeed.
    status, info = await state.add_ticker_to_trade_allowlist("005930.KS")
    assert status == "added"
    assert "005930.KS" in settings.trade_allowlist


async def test_add_ticker_to_trade_allowlist_rejects_malformed_ticker(monkeypatch):
    # Real incident: a comma-glued, 20+ char value (looked like several KIS
    # fields joined together) reached settings.trade_allowlist via the AI
    # auto-add path (pipeline.py), which has no HTTPException-raising
    # validate_ticker in front of it the way the dashboard's manual-add
    # route does. Must be rejected here too, before any network call.
    with pytest.raises(state.MalformedTickerError):
        await state.add_ticker_to_trade_allowlist("066430.KS,0P0000CBYQ,198463")
    assert settings.trade_allowlist == []


async def test_add_ticker_to_watchlist_rejects_malformed_ticker(monkeypatch):
    with pytest.raises(state.MalformedTickerError):
        await state.add_ticker_to_watchlist("066430.KS,0P0000CBYQ,198463")
    assert "066430.KS,0P0000CBYQ,198463" not in settings.watchlist


async def test_add_ticker_to_trade_allowlist_rejects_mutual_fund(monkeypatch):
    # Real incident: 218150.KS (yfinance quoteType MUTUALFUND) also came
    # with a broken shortName ("218150.KS,0P000168Z2,87763") - this checks
    # the product-type gate specifically, independent of that name defect.
    async def fake_display_info(ticker):
        return _FakeDisplayInfo(name="Some Fund", quote_type="MUTUALFUND")

    monkeypatch.setattr(state, "get_ticker_display_info", fake_display_info)
    with pytest.raises(state.UnsupportedProductTypeError):
        await state.add_ticker_to_trade_allowlist("218150.KS")
    assert settings.trade_allowlist == []


async def test_add_ticker_to_trade_allowlist_allows_etf(monkeypatch):
    async def fake_display_info(ticker):
        return _FakeDisplayInfo(name="Some ETF", quote_type="ETF")

    monkeypatch.setattr(state, "get_ticker_display_info", fake_display_info)
    status, info = await state.add_ticker_to_trade_allowlist("0167A0.KS")
    assert status == "added"
    assert "0167A0.KS" in settings.trade_allowlist


# ---------------------------------------------------------------------------
# trade_allowlist_auto_added marker — bookkeeping for reconcile_trade_allowlist's
# add/prune contract (2026-07-29)
# ---------------------------------------------------------------------------


async def test_mark_trade_allowlist_auto_added_is_idempotent():
    await state.mark_trade_allowlist_auto_added("005930.KS")
    await state.mark_trade_allowlist_auto_added("005930.KS")
    assert settings.trade_allowlist_auto_added == ["005930.KS"]


async def test_is_trade_allowlist_auto_added_reflects_marker_state():
    assert state.is_trade_allowlist_auto_added("005930.KS") is False
    await state.mark_trade_allowlist_auto_added("005930.KS")
    assert state.is_trade_allowlist_auto_added("005930.KS") is True


async def test_unmark_trade_allowlist_auto_added_is_idempotent():
    await state.mark_trade_allowlist_auto_added("005930.KS")
    await state.unmark_trade_allowlist_auto_added("005930.KS")
    await state.unmark_trade_allowlist_auto_added("005930.KS")  # must not raise on an absent ticker
    assert settings.trade_allowlist_auto_added == []


async def test_remove_ticker_from_trade_allowlist_clears_auto_added_marker(monkeypatch):
    monkeypatch.setattr(settings, "trade_allowlist_domestic", ["005930.KS"])
    await state.mark_trade_allowlist_auto_added("005930.KS")
    await state.remove_ticker_from_trade_allowlist("005930.KS")
    assert "005930.KS" not in settings.trade_allowlist
    assert state.is_trade_allowlist_auto_added("005930.KS") is False


async def test_remove_ticker_from_trade_allowlist_clears_live_process_caches(monkeypatch):
    # 2026-08-13 fix: removing a ticker used to only clear `_display_info`,
    # leaving app.live.store's tick buffer, app.live.poller's
    # cumulative-volume baseline, app.live.alerts' cooldowns, and
    # app.live.auto_trade_loop's gap-warning cooldown all still populated
    # under the old ticker's key. Re-adding the same ticker later would then
    # resurrect that stale state (most visibly: an old tick buffer, since
    # nothing else ever clears app.live.store._ticks) as if it were current.
    from app.live import alerts as live_alerts
    from app.live import auto_trade_loop as live_auto_trade_loop
    from app.live import poller as live_poller

    monkeypatch.setattr(settings, "trade_allowlist_domestic", ["005930.KS"])

    live_poller._last_cumulative_volume["005930"] = 1_000_000
    live_alerts._last_fired_at[(settings.kis_account_mode, "005930.KS", "price_move")] = 123.0
    live_auto_trade_loop._gap_warned_on[(settings.kis_account_mode, "005930.KS")] = date(2026, 8, 13)

    store_forget_calls = []

    async def fake_store_forget(ticker):
        store_forget_calls.append(ticker)

    monkeypatch.setattr(state, "forget_ticker_store", fake_store_forget)

    await state.remove_ticker_from_trade_allowlist("005930.KS")

    assert store_forget_calls == ["005930.KS"]  # app.live.store's tick buffer + snapshot row
    assert "005930" not in live_poller._last_cumulative_volume
    assert (settings.kis_account_mode, "005930.KS", "price_move") not in live_alerts._last_fired_at
    assert (settings.kis_account_mode, "005930.KS") not in live_auto_trade_loop._gap_warned_on


# ---------------------------------------------------------------------------
# held_shares_for_removal — 2026-08-26: a partial KIS snapshot (only the
# OVERSEAS balance TR failed) used to read as "AAPL confirmed not held"
# instead of "unknown", because account is not None (domestic succeeded) so
# the known=False branch never fired, and the missing overseas ticker just
# looked like a real zero. That let POST /api/trade-allowlist/remove skip
# its confirmation dialog and delete a still-held, now-unsellable US
# position, and separately let reconcile_trade_allowlist auto-prune it.
# ---------------------------------------------------------------------------


async def test_held_check_reports_unknown_for_a_us_ticker_when_the_overseas_leg_failed(monkeypatch):
    async def fake_snapshot():
        return {"overseas_ok": False}, {}  # domestic succeeded, overseas TR failed

    async def fake_get_position(ticker):
        class P:
            shares = 0

        return P()

    monkeypatch.setattr(state, "kis_account_snapshot", fake_snapshot)
    monkeypatch.setattr(state, "get_position", fake_get_position)
    state.kis_broker = broker_stub(object())

    result = await state.held_shares_for_removal("AAPL")

    assert result.known is False
    assert result.source == "unknown"


async def test_held_check_still_reports_a_domestic_ticker_normally_during_an_overseas_outage(monkeypatch):
    async def fake_snapshot():
        return {"overseas_ok": False}, {"005930.KS": {"quantity": 10}}

    async def fake_get_position(ticker):
        class P:
            shares = 0

        return P()

    monkeypatch.setattr(state, "kis_account_snapshot", fake_snapshot)
    monkeypatch.setattr(state, "get_position", fake_get_position)
    state.kis_broker = broker_stub(object())

    result = await state.held_shares_for_removal("005930.KS")

    assert result.known is True
    assert result.shares == 10
    assert result.source == "kis"


async def test_held_check_trusts_the_ledger_for_a_us_ticker_during_an_overseas_outage(monkeypatch):
    async def fake_snapshot():
        return {"overseas_ok": False}, {}

    async def fake_get_position(ticker):
        class P:
            shares = 7

        return P()

    monkeypatch.setattr(state, "kis_account_snapshot", fake_snapshot)
    monkeypatch.setattr(state, "get_position", fake_get_position)
    state.kis_broker = broker_stub(object())

    result = await state.held_shares_for_removal("AAPL")

    assert result.known is True
    assert result.shares == 7
    assert result.source == "ledger"


# ---------------------------------------------------------------------------
# reconcile_trade_allowlist — bidirectional sync against real KIS positions
# ---------------------------------------------------------------------------


async def test_reconcile_adds_held_ticker_not_on_allowlist(monkeypatch):
    async def fake_snapshot():
        return {"cash_deposit": 1.0}, {"005930.KS": {"quantity": 5}}

    async def fake_list_pending():
        return []

    monkeypatch.setattr(state, "kis_account_snapshot", fake_snapshot)
    monkeypatch.setattr(state, "list_pending_orders", fake_list_pending)

    newly_added = await state.reconcile_trade_allowlist()

    assert newly_added == ["005930.KS"]
    assert "005930.KS" in settings.trade_allowlist
    assert state.is_trade_allowlist_auto_added("005930.KS") is True


async def test_reconcile_skips_entirely_when_account_snapshot_unavailable(monkeypatch):
    monkeypatch.setattr(settings, "trade_allowlist_domestic", ["005930.KS"])
    await state.mark_trade_allowlist_auto_added("005930.KS")

    async def fake_snapshot():
        return None, {}

    async def fail_if_called():
        raise AssertionError("must not read pending orders when the snapshot itself failed")

    monkeypatch.setattr(state, "kis_account_snapshot", fake_snapshot)
    monkeypatch.setattr(state, "list_pending_orders", fail_if_called)

    result = await state.reconcile_trade_allowlist()

    assert result == []
    assert "005930.KS" in settings.trade_allowlist  # untouched, not mass-pruned on an outage


async def test_reconcile_prunes_auto_added_ticker_with_zero_shares_and_no_pending_order(monkeypatch):
    monkeypatch.setattr(settings, "trade_allowlist_domestic", ["005930.KS"])
    await state.mark_trade_allowlist_auto_added("005930.KS")

    async def fake_snapshot():
        return {"cash_deposit": 1.0}, {}  # no real position for it anymore

    async def fake_list_pending():
        return []

    monkeypatch.setattr(state, "kis_account_snapshot", fake_snapshot)
    monkeypatch.setattr(state, "list_pending_orders", fake_list_pending)

    await state.reconcile_trade_allowlist()

    assert "005930.KS" not in settings.trade_allowlist
    assert state.is_trade_allowlist_auto_added("005930.KS") is False


async def test_reconcile_does_not_prune_ticker_with_an_open_pending_order(monkeypatch):
    monkeypatch.setattr(settings, "trade_allowlist_domestic", ["005930.KS"])
    await state.mark_trade_allowlist_auto_added("005930.KS")

    class _FakeOrder:
        ticker = "005930.KS"

    async def fake_snapshot():
        return {"cash_deposit": 1.0}, {}

    async def fake_list_pending():
        return [_FakeOrder()]

    monkeypatch.setattr(state, "kis_account_snapshot", fake_snapshot)
    monkeypatch.setattr(state, "list_pending_orders", fake_list_pending)

    await state.reconcile_trade_allowlist()

    assert "005930.KS" in settings.trade_allowlist  # still awaiting approval — must survive this pass


async def test_reconcile_does_not_prune_an_overseas_auto_added_ticker_when_the_overseas_leg_failed(monkeypatch):
    """2026-08-26: a partial snapshot (overseas TR failed, domestic
    succeeded) has no overseas holdings in `held` at all — pruning AAPL from
    that is deleting on "couldn't tell", not "confirmed not held"."""
    monkeypatch.setattr(settings, "trade_allowlist_overseas", ["AAPL"])
    await state.mark_trade_allowlist_auto_added("AAPL")

    async def fake_snapshot():
        return {"cash_deposit": 1.0, "overseas_ok": False}, {}

    async def fake_list_pending():
        return []

    monkeypatch.setattr(state, "kis_account_snapshot", fake_snapshot)
    monkeypatch.setattr(state, "list_pending_orders", fake_list_pending)

    await state.reconcile_trade_allowlist()

    assert "AAPL" in settings.trade_allowlist
    assert state.is_trade_allowlist_auto_added("AAPL") is True


async def test_reconcile_still_prunes_a_domestic_auto_added_ticker_during_an_overseas_outage(monkeypatch):
    """The skip above must be scoped to overseas tickers only — a domestic
    prune candidate is unaffected by an overseas-only TR failure, since the
    domestic leg of the snapshot succeeded."""
    monkeypatch.setattr(settings, "trade_allowlist_domestic", ["005930.KS"])
    await state.mark_trade_allowlist_auto_added("005930.KS")

    async def fake_snapshot():
        return {"cash_deposit": 1.0, "overseas_ok": False}, {}

    async def fake_list_pending():
        return []

    monkeypatch.setattr(state, "kis_account_snapshot", fake_snapshot)
    monkeypatch.setattr(state, "list_pending_orders", fake_list_pending)

    await state.reconcile_trade_allowlist()

    assert "005930.KS" not in settings.trade_allowlist


async def test_reconcile_never_prunes_a_user_added_ticker_even_at_zero_shares(monkeypatch):
    # Not marked auto-added -> a legitimate "watching, haven't bought yet"
    # candidate, must never be pruned just for having zero shares.
    monkeypatch.setattr(settings, "trade_allowlist_domestic", ["005930.KS"])

    async def fake_snapshot():
        return {"cash_deposit": 1.0}, {}

    async def fake_list_pending():
        return []

    monkeypatch.setattr(state, "kis_account_snapshot", fake_snapshot)
    monkeypatch.setattr(state, "list_pending_orders", fake_list_pending)

    await state.reconcile_trade_allowlist()

    assert "005930.KS" in settings.trade_allowlist


async def test_reconcile_leaves_still_held_auto_added_ticker_alone(monkeypatch):
    monkeypatch.setattr(settings, "trade_allowlist_domestic", ["005930.KS"])
    await state.mark_trade_allowlist_auto_added("005930.KS")

    async def fake_snapshot():
        return {"cash_deposit": 1.0}, {"005930.KS": {"quantity": 5}}

    async def fake_list_pending():
        return []

    monkeypatch.setattr(state, "kis_account_snapshot", fake_snapshot)
    monkeypatch.setattr(state, "list_pending_orders", fake_list_pending)

    newly_added = await state.reconcile_trade_allowlist()

    assert newly_added == []  # already on the allowlist — not re-added
    assert "005930.KS" in settings.trade_allowlist


async def test_reconcile_does_not_prune_a_real_holding_using_a_stale_paper_snapshot(monkeypatch):
    """Moved+fixed from tests/web/test_zzq_modestate_audit_scratch.py::
    test_F1c_reconcile_prunes_a_real_holding_using_the_paper_snapshot.
    auto_trade_loop -> reconcile_trade_allowlist() reads
    kis_account_snapshot(); before the cache-scoping fix, within the cache's
    TTL after a paper->real switch it got the PAPER positions, and a real
    10-share holding with no matching paper position looked unheld and got
    pruned off the allowlist."""
    async def fake_broker():
        return object()

    paper_positions = {"000660.KS": {"quantity": 5}}
    real_positions = {"005930.KS": {"quantity": 10}}

    async def fake_build():
        mode = current_account_mode()
        pos = real_positions if mode == "real" else paper_positions
        return ({"is_real": mode == "real"}, pos)

    async def no_pending():
        return []

    removed: list[str] = []

    async def fake_remove(t):
        removed.append(t)

    async def fake_add(t):
        return "exists", None

    monkeypatch.setattr(state, "kis_broker", fake_broker)
    monkeypatch.setattr(state, "_kis_account_snapshot_build", fake_build)
    monkeypatch.setattr(state, "list_pending_orders", no_pending)
    monkeypatch.setattr(state, "remove_ticker_from_trade_allowlist", fake_remove)
    monkeypatch.setattr(state, "add_ticker_to_trade_allowlist", fake_add)

    async def fake_save(*a, **k):
        return None

    monkeypatch.setattr(state.trade_allowlist_store, "save", fake_save)

    # a dashboard poll warms the cache while still in paper mode
    with account_mode_scope("paper"):
        await state.kis_account_snapshot()
    # user switches to real; auto_trade_loop's next pass runs under the real
    # scope. The allowlist lists are themselves per-(user, mode) scoped, so
    # they're populated inside the real scope here -- this is the REAL
    # account's state.
    with account_mode_scope("real"):
        settings.trade_allowlist_domestic = ["005930.KS", "000660.KS"]
        settings.trade_allowlist_overseas = []
        settings.trade_allowlist_auto_added = ["005930.KS"]
        acct, pos = await state.kis_account_snapshot()
        assert pos == real_positions, "real mode must see the real broker's own positions, not the cached paper ones"
        await state.reconcile_trade_allowlist()

    assert removed == [], "a real 10-share holding must not be pruned using a stale paper-mode snapshot"


async def test_reconcile_continues_after_one_ticker_fails_to_add(monkeypatch):
    async def fake_snapshot():
        return {"cash_deposit": 1.0}, {"005930.KS": {"quantity": 5}, "AAPL": {"quantity": 3}}

    async def fake_list_pending():
        return []

    real_add = state.add_ticker_to_trade_allowlist

    async def flaky_add(ticker):
        if ticker == "005930.KS":
            raise RuntimeError("leveraged/unsupported product, or a DB hiccup")
        return await real_add(ticker)

    monkeypatch.setattr(state, "kis_account_snapshot", fake_snapshot)
    monkeypatch.setattr(state, "list_pending_orders", fake_list_pending)
    monkeypatch.setattr(state, "add_ticker_to_trade_allowlist", flaky_add)

    newly_added = await state.reconcile_trade_allowlist()

    assert newly_added == ["AAPL"]
    assert "AAPL" in settings.trade_allowlist
    assert "005930.KS" not in settings.trade_allowlist


# ---------------------------------------------------------------------------
# _prioritized_krx_codes — realtime feed's 41-combined-registration cap
# (2026-04-20 KIS notice), held positions prioritized over unheld candidates
# ---------------------------------------------------------------------------


async def test_prioritized_krx_codes_returns_everything_under_the_cap(monkeypatch):
    codes = [f"{100000 + i}.KS" for i in range(5)]
    monkeypatch.setattr(settings, "trade_allowlist_domestic", codes)
    monkeypatch.setattr(settings, "trade_allowlist_overseas", [])

    result, orderbook_zone, cb_only_zone = await state._prioritized_krx_codes()

    assert result == [f"{100000 + i}" for i in range(5)]
    assert len(result) <= state._CORE_FEED_CODES + state._CB_ONLY_ZONE_SIZE
    # Only 5 codes exist total, well under _CORE_FEED_CODES — all 5 land in
    # the core (2TR) tier, none spill into the cb_only tail.
    assert cb_only_zone == frozenset()
    # None are held (no account configured in tests), so every one of them
    # is eligible and the orderbook zone is the whole set — _trailing_
    # orderbook_zone never pads beyond what's actually available, but also
    # never withholds an eligible code below _ORDERBOOK_ZONE_SIZE.
    assert orderbook_zone == frozenset(f"{100000 + i}" for i in range(5))


async def test_prioritized_krx_codes_puts_held_positions_first_and_caps(monkeypatch):
    from app.execution.portfolio_risk import PortfolioSnapshot

    budget = state._CORE_FEED_CODES + state._CB_ONLY_ZONE_SIZE
    codes = [f"{100000 + i}.KS" for i in range(budget + 5)]
    monkeypatch.setattr(settings, "trade_allowlist_domestic", codes)
    monkeypatch.setattr(settings, "trade_allowlist_overseas", [])

    held = {"100003.KS": 1.0, "100010.KS": 1.0}  # both outside the cap by plain list order

    async def fake_snapshot(market):
        return PortfolioSnapshot(per_ticker_value=held, total_value=0.0)

    monkeypatch.setattr("app.execution.portfolio_risk.real_account_snapshot", fake_snapshot)

    result, orderbook_zone, cb_only_zone = await state._prioritized_krx_codes()

    assert len(result) == budget
    assert set(result[:2]) == {"100003", "100010"}  # held codes claimed the first (core) slots
    assert orderbook_zone.isdisjoint({"100003", "100010"})  # held codes never in the zone
    assert len(orderbook_zone) == state._ORDERBOOK_ZONE_SIZE
    # Everything past the core budget lands in the 1TR cb_only tail, not
    # dropped — that's the entire point of the tiered budget.
    assert len(cb_only_zone) == state._CB_ONLY_ZONE_SIZE
    assert cb_only_zone.isdisjoint(set(result[: state._CORE_FEED_CODES]))


async def test_prioritized_krx_codes_falls_back_to_trade_allowlist_order_when_snapshot_fails(monkeypatch):
    budget = state._CORE_FEED_CODES + state._CB_ONLY_ZONE_SIZE
    codes = [f"{100000 + i}.KS" for i in range(budget + 5)]
    monkeypatch.setattr(settings, "trade_allowlist_domestic", codes)
    monkeypatch.setattr(settings, "trade_allowlist_overseas", [])

    async def broken_snapshot(market):
        raise RuntimeError("KIS unreachable")

    monkeypatch.setattr("app.execution.portfolio_risk.real_account_snapshot", broken_snapshot)

    result, orderbook_zone, cb_only_zone = await state._prioritized_krx_codes()

    assert result == [f"{100000 + i}" for i in range(budget)]
    # snapshot failure means held_codes comes back empty, not that the zone
    # computation itself blows up — the trailing core codes (all "unheld"
    # absent better information) become the orderbook zone, and the codes
    # past the core budget become the cb_only tail.
    assert len(orderbook_zone) == state._ORDERBOOK_ZONE_SIZE
    assert len(cb_only_zone) == state._CB_ONLY_ZONE_SIZE


# ---------------------------------------------------------------------------
# _prioritized_krx_codes / orderbook zone edge cases (2026-08-24)
# ---------------------------------------------------------------------------


async def test_orderbook_zone_shrinks_when_held_positions_fill_most_slots(monkeypatch):
    from app.execution.portfolio_risk import PortfolioSnapshot

    codes = [f"{100000 + i}.KS" for i in range(state._CORE_FEED_CODES)]
    monkeypatch.setattr(settings, "trade_allowlist_domestic", codes)
    monkeypatch.setattr(settings, "trade_allowlist_overseas", [])

    # All but 2 of the core held, 2 unheld — the zone must have exactly 2,
    # not padded to _ORDERBOOK_ZONE_SIZE (5).
    held = {f"{100000 + i}.KS": 1.0 for i in range(state._CORE_FEED_CODES - 2)}

    async def fake_snapshot(market):
        return PortfolioSnapshot(per_ticker_value=held, total_value=0.0)

    monkeypatch.setattr("app.execution.portfolio_risk.real_account_snapshot", fake_snapshot)

    result, orderbook_zone, _cb_only_zone = await state._prioritized_krx_codes()

    assert len(orderbook_zone) == 2
    last_two = {f"{100000 + i}" for i in range(state._CORE_FEED_CODES - 2, state._CORE_FEED_CODES)}
    assert orderbook_zone == last_two


async def test_orderbook_zone_empty_when_every_slot_is_held(monkeypatch):
    from app.execution.portfolio_risk import PortfolioSnapshot

    codes = [f"{100000 + i}.KS" for i in range(state._CORE_FEED_CODES)]
    monkeypatch.setattr(settings, "trade_allowlist_domestic", codes)
    monkeypatch.setattr(settings, "trade_allowlist_overseas", [])

    held = {c: 1.0 for c in codes}

    async def fake_snapshot(market):
        return PortfolioSnapshot(per_ticker_value=held, total_value=0.0)

    monkeypatch.setattr("app.execution.portfolio_risk.real_account_snapshot", fake_snapshot)

    _result, orderbook_zone, _cb_only_zone = await state._prioritized_krx_codes()

    assert orderbook_zone == frozenset()


# ---------------------------------------------------------------------------
# reconcile_feed_tier_promotions (2026-08-24, extended to cb_only 2026-08-28)
# ---------------------------------------------------------------------------


async def test_reconcile_promotions_noop_when_no_feed(monkeypatch):
    monkeypatch.setattr(state, "_market_feed", None)
    calls: list[str] = []
    monkeypatch.setattr(state, "stop_market_feed", _record_call(calls, "stop"))
    monkeypatch.setattr(state, "ensure_market_feed", _record_call(calls, "ensure"))

    await state.reconcile_feed_tier_promotions()

    assert calls == []


async def test_reconcile_promotions_noop_when_zone_empty(monkeypatch):
    from app.brokers.kis.domestic.websocket import RealtimeFeed

    feed = RealtimeFeed(app_key="k", app_secret="s", codes=[])
    monkeypatch.setattr(state, "_market_feed", feed)
    calls: list[str] = []
    monkeypatch.setattr(state, "stop_market_feed", _record_call(calls, "stop"))
    monkeypatch.setattr(state, "ensure_market_feed", _record_call(calls, "ensure"))

    await state.reconcile_feed_tier_promotions()

    assert calls == []


async def test_reconcile_promotions_rebuilds_once_when_a_zone_code_becomes_held(monkeypatch):
    from app.brokers.kis.domestic.websocket import RealtimeFeed

    feed = RealtimeFeed(app_key="k", app_secret="s", codes=["100000"], orderbook_codes=frozenset({"100000"}))
    monkeypatch.setattr(state, "_market_feed", feed)

    async def fake_held(scopes):
        return {"100000"}

    monkeypatch.setattr(state, "_held_krx_codes", fake_held)
    calls: list[str] = []
    monkeypatch.setattr(state, "stop_market_feed", _record_call(calls, "stop"))
    monkeypatch.setattr(state, "ensure_market_feed", _record_call(calls, "ensure"))

    await state.reconcile_feed_tier_promotions()

    assert calls == ["stop", "ensure"]


async def test_reconcile_promotions_noop_when_zone_code_still_unheld(monkeypatch):
    from app.brokers.kis.domestic.websocket import RealtimeFeed

    feed = RealtimeFeed(app_key="k", app_secret="s", codes=["100000"], orderbook_codes=frozenset({"100000"}))
    monkeypatch.setattr(state, "_market_feed", feed)

    async def fake_held(scopes):
        return set()

    monkeypatch.setattr(state, "_held_krx_codes", fake_held)
    calls: list[str] = []
    monkeypatch.setattr(state, "stop_market_feed", _record_call(calls, "stop"))
    monkeypatch.setattr(state, "ensure_market_feed", _record_call(calls, "ensure"))

    await state.reconcile_feed_tier_promotions()

    assert calls == []


async def test_reconcile_promotions_rebuilds_once_when_a_cb_only_code_becomes_held(monkeypatch):
    """2026-08-28: a cb_only-tier code (1TR, no live trade tick) that just got
    bought is exactly as exposed as an orderbook-zone one was — same
    promotion path must catch it too."""
    from app.brokers.kis.domestic.websocket import RealtimeFeed

    feed = RealtimeFeed(app_key="k", app_secret="s", codes=["100000"], cb_only_codes=frozenset({"100000"}))
    monkeypatch.setattr(state, "_market_feed", feed)

    async def fake_held(scopes):
        return {"100000"}

    monkeypatch.setattr(state, "_held_krx_codes", fake_held)
    calls: list[str] = []
    monkeypatch.setattr(state, "stop_market_feed", _record_call(calls, "stop"))
    monkeypatch.setattr(state, "ensure_market_feed", _record_call(calls, "ensure"))

    await state.reconcile_feed_tier_promotions()

    assert calls == ["stop", "ensure"]


# ---------------------------------------------------------------------------
# rebuild_market_feed_for_capacity (2026-10-02, CB registration-capacity fix)
# ---------------------------------------------------------------------------


@pytest.fixture
def _capacity_state(monkeypatch):
    monkeypatch.setattr(state, "_capacity_rebuild_last_at", None)
    monkeypatch.setattr(state, "_capacity_rebuild_confirm_task", None)
    yield


async def test_capacity_rebuild_noop_when_no_feed(monkeypatch, _capacity_state):
    monkeypatch.setattr(state, "_market_feed", None)
    calls: list[str] = []
    monkeypatch.setattr(state, "stop_market_feed", _record_call(calls, "stop"))
    monkeypatch.setattr(state, "ensure_market_feed", _record_call(calls, "ensure"))
    assert await state.rebuild_market_feed_for_capacity("t") is False
    assert calls == []


def _patch_capacity_rebuild(monkeypatch, calls, *, new_feed):
    async def fake_ensure():
        calls.append("ensure")
        state._market_feed = new_feed
        return new_feed

    monkeypatch.setattr(state, "stop_market_feed", _record_call(calls, "stop"))
    monkeypatch.setattr(state, "ensure_market_feed", fake_ensure)


async def test_capacity_rebuild_runs_stop_then_ensure_and_respects_interval(monkeypatch, _capacity_state):
    from app.brokers.kis.domestic.websocket import RealtimeFeed

    old = RealtimeFeed(app_key="k", app_secret="s", codes=["005930"])
    new = RealtimeFeed(app_key="k", app_secret="s", codes=["005930"])
    new.connected = True
    monkeypatch.setattr(state, "_market_feed", old)
    calls: list[str] = []
    _patch_capacity_rebuild(monkeypatch, calls, new_feed=new)

    assert await state.rebuild_market_feed_for_capacity("first") is True
    assert calls == ["stop", "ensure"]
    await state._capacity_rebuild_confirm_task

    # second call right away: inside the 10-minute floor -> skipped
    assert await state.rebuild_market_feed_for_capacity("second") is False
    assert calls == ["stop", "ensure"]

    # past the floor -> allowed again
    monkeypatch.setattr(state, "_capacity_rebuild_last_at", time.monotonic() - 601)
    assert await state.rebuild_market_feed_for_capacity("third") is True
    assert calls == ["stop", "ensure", "stop", "ensure"]
    await state._capacity_rebuild_confirm_task


async def test_capacity_rebuild_confirmed_connected_raises_no_alert(monkeypatch, _capacity_state):
    from app.brokers.kis.domestic.websocket import RealtimeFeed

    new = RealtimeFeed(app_key="k", app_secret="s", codes=["005930"])
    monkeypatch.setattr(state, "_market_feed", RealtimeFeed(app_key="k", app_secret="s", codes=[]))
    _patch_capacity_rebuild(monkeypatch, [], new_feed=new)
    alerts = []

    async def fake_record_alert(alert):
        alerts.append(alert)

    monkeypatch.setattr(state, "record_alert", fake_record_alert)
    sleeps = []

    async def fake_sleep(sec):
        sleeps.append(sec)
        new.connected = len(sleeps) >= 3  # connects on the 3rd poll

    monkeypatch.setattr(state.asyncio, "sleep", fake_sleep)

    assert await state.rebuild_market_feed_for_capacity("t") is True
    await state._capacity_rebuild_confirm_task
    assert alerts == []
    assert len(sleeps) == 3


async def test_capacity_rebuild_not_connected_in_60s_alerts(monkeypatch, _capacity_state):
    from app.brokers.kis.domestic.websocket import RealtimeFeed

    new = RealtimeFeed(app_key="k", app_secret="s", codes=["005930"])  # never connects
    monkeypatch.setattr(state, "_market_feed", RealtimeFeed(app_key="k", app_secret="s", codes=[]))
    _patch_capacity_rebuild(monkeypatch, [], new_feed=new)
    alerts = []

    async def fake_record_alert(alert):
        alerts.append(alert)

    monkeypatch.setattr(state, "record_alert", fake_record_alert)
    fake_now = {"t": 1000.0}
    monkeypatch.setattr(state.time, "monotonic", lambda: fake_now["t"])

    async def fake_sleep(sec):
        fake_now["t"] += sec

    monkeypatch.setattr(state.asyncio, "sleep", fake_sleep)

    assert await state.rebuild_market_feed_for_capacity("t") is True
    await state._capacity_rebuild_confirm_task
    assert [a.kind for a in alerts] == ["realtime_feed_rebuild_slow"]


def _record_call(calls: list[str], label: str):
    async def _fn(*_args, **_kwargs):
        calls.append(label)
    return _fn


# ---------------------------------------------------------------------------
# note_candidate_cb_code / _live_candidate_cb_codes / candidate merge into
# _prioritized_krx_codes's cb-only zone (2026-09-04, 3rd recurrence of the
# "candidate CB coverage lost on feed rebuild" bug — #9, #173, and this one).
# See state._candidate_cb_codes's own comment for the full root-cause chain:
# _ensure_subscribed's ad-hoc add_code(cb_only=True) calls only mutated the
# live RealtimeFeed object, which reconcile_feed_tier_promotions's
# stop_market_feed()+ensure_market_feed() rebuild throws away — this registry
# is the missing persistent record that survives a rebuild.
# ---------------------------------------------------------------------------


async def test_note_candidate_cb_code_lands_in_cb_only_zone_not_core(monkeypatch):
    # A full core (12 unheld allowlist codes) so `result[:_CORE_FEED_CODES]`
    # unambiguously means "core" — with an empty/short allowlist, `result`
    # itself is shorter than _CORE_FEED_CODES and the candidate (in the
    # cb-only tail) would wrongly appear to be "in core" simply because
    # result[:12] spills past the (empty) core into the tail.
    codes = [f"{100000 + i}.KS" for i in range(state._CORE_FEED_CODES)]
    monkeypatch.setattr(settings, "trade_allowlist_domestic", codes)
    monkeypatch.setattr(settings, "trade_allowlist_overseas", [])

    state.note_candidate_cb_code("441270")

    result, _orderbook_zone, cb_only_zone = await state._prioritized_krx_codes()

    assert "441270" in cb_only_zone
    assert "441270" not in result[: state._CORE_FEED_CODES]


async def test_candidate_registry_survives_a_feed_rebuild(monkeypatch):
    """Regression reproduction for the 3rd recurrence: before this fix,
    _prioritized_krx_codes had no idea auto_trade_loop's ad-hoc candidate
    pre-subscriptions existed, so reconcile_feed_tier_promotions's
    stop_market_feed()+ensure_market_feed() rebuild — which recomputes the
    zones from _prioritized_krx_codes and constructs a brand-new RealtimeFeed
    from them, exactly as done here — silently dropped every candidate's
    circuit-breaker coverage. Fails on the pre-fix code (no
    note_candidate_cb_code / no merge in _prioritized_krx_codes existed, so
    none of these codes could ever appear in cb_only_zone at all); passes
    once _live_candidate_cb_codes is folded back into every zone rebuild."""
    from app.brokers.kis.domestic.websocket import RealtimeFeed

    monkeypatch.setattr(settings, "trade_allowlist_domestic", [])
    monkeypatch.setattr(settings, "trade_allowlist_overseas", [])

    for code in ("441270", "091990", "060310"):
        state.note_candidate_cb_code(code)

    # Simulate the exact rebuild reconcile_feed_tier_promotions triggers:
    # recompute the zones fresh and construct a brand-new RealtimeFeed from
    # them (the same constructor call ensure_market_feed makes).
    codes, orderbook_zone, cb_only_zone = await state._prioritized_krx_codes()
    new_feed = RealtimeFeed(
        app_key="k", app_secret="s", codes=codes,
        orderbook_codes=orderbook_zone, cb_only_codes=cb_only_zone,
    )

    assert {"441270", "091990", "060310"} <= new_feed.cb_only_codes


async def test_held_positions_still_own_core_even_with_candidates_registered(monkeypatch):
    from app.execution.portfolio_risk import PortfolioSnapshot

    codes = [f"{100000 + i}.KS" for i in range(state._CORE_FEED_CODES)]
    monkeypatch.setattr(settings, "trade_allowlist_domestic", codes)
    monkeypatch.setattr(settings, "trade_allowlist_overseas", [])
    held = {c: 1.0 for c in codes}

    async def fake_snapshot(market):
        return PortfolioSnapshot(per_ticker_value=held, total_value=0.0)

    monkeypatch.setattr("app.execution.portfolio_risk.real_account_snapshot", fake_snapshot)

    candidate_codes = [f"{900000 + i}" for i in range(8)]
    for code in candidate_codes:
        state.note_candidate_cb_code(code)

    result, _orderbook_zone, cb_only_zone = await state._prioritized_krx_codes()

    core = result[: state._CORE_FEED_CODES]
    assert set(core) == {c.removesuffix(".KS") for c in codes}
    assert not (set(candidate_codes) & set(core))
    assert set(candidate_codes) <= cb_only_zone


async def test_orderbook_zone_unaffected_by_candidate_registration(monkeypatch):
    from app.execution.portfolio_risk import PortfolioSnapshot

    codes = [f"{100000 + i}.KS" for i in range(state._CORE_FEED_CODES + 3)]
    monkeypatch.setattr(settings, "trade_allowlist_domestic", codes)
    monkeypatch.setattr(settings, "trade_allowlist_overseas", [])
    held = {f"{100000 + i}.KS": 1.0 for i in range(state._CORE_FEED_CODES - 2)}

    async def fake_snapshot(market):
        return PortfolioSnapshot(per_ticker_value=held, total_value=0.0)

    monkeypatch.setattr("app.execution.portfolio_risk.real_account_snapshot", fake_snapshot)

    _result, orderbook_zone_before, _cb_before = await state._prioritized_krx_codes()

    for code in ("900001", "900002"):
        state.note_candidate_cb_code(code)

    _result2, orderbook_zone_after, _cb_after = await state._prioritized_krx_codes()

    assert orderbook_zone_before == orderbook_zone_after


async def test_candidate_reserve_boundary_splits_cb_only_zone_8_and_8(monkeypatch):
    total = state._CORE_FEED_CODES + 20  # core fully filled + a 20-code allowlist tail
    codes = [f"{100000 + i}.KS" for i in range(total)]
    monkeypatch.setattr(settings, "trade_allowlist_domestic", codes)
    monkeypatch.setattr(settings, "trade_allowlist_overseas", [])

    candidate_codes = [f"{900000 + i}" for i in range(12)]
    for code in candidate_codes:
        state.note_candidate_cb_code(code)

    _result, _orderbook_zone, cb_only_zone = await state._prioritized_krx_codes()

    assert len(cb_only_zone) == state._CB_ONLY_ZONE_SIZE  # 16
    assert len(cb_only_zone & set(candidate_codes)) == state._CANDIDATE_CB_RESERVE  # 8
    allowlist_tail_codes = {f"{100000 + i}" for i in range(state._CORE_FEED_CODES, total)}
    assert len(cb_only_zone & allowlist_tail_codes) == 8


async def test_candidate_ttl_expires_and_is_evicted(monkeypatch):
    # Deliberately does NOT monkeypatch the global time.monotonic — this
    # module's `time` is the real stdlib module, and asyncio's own event-loop
    # scheduling (BaseEventLoop.time()) reads from the very same global
    # time.monotonic. Freezing it process-wide breaks any real async I/O this
    # test's `await`s touch (call_later-based timeouts/retries never fire,
    # since the frozen clock never reaches their computed deadline) — an
    # actual hang was observed here during review. Backdating the registry
    # entry directly, using one real `time.monotonic()` reading, gets the
    # same TTL-expiry behavior with zero interference with the event loop.
    monkeypatch.setattr(settings, "trade_allowlist_domestic", [])
    monkeypatch.setattr(settings, "trade_allowlist_overseas", [])

    state.note_candidate_cb_code("441270")
    state._candidate_cb_codes["441270"] = time.monotonic() - state._CANDIDATE_CB_TTL_SECONDS - 1

    _result, _orderbook_zone, cb_only_zone = await state._prioritized_krx_codes()

    assert "441270" not in cb_only_zone
    assert "441270" not in state._candidate_cb_codes


async def test_total_codes_never_exceed_max_registrations_even_with_many_candidates(monkeypatch):
    from app.brokers.kis.domestic.websocket import RealtimeFeed

    total_allowlist = state._CORE_FEED_CODES + state._CB_ONLY_ZONE_SIZE + 10
    codes = [f"{100000 + i}.KS" for i in range(total_allowlist)]
    monkeypatch.setattr(settings, "trade_allowlist_domestic", codes)
    monkeypatch.setattr(settings, "trade_allowlist_overseas", [])

    for i in range(30):
        state.note_candidate_cb_code(f"{900000 + i}")

    result, _orderbook_zone, _cb_only_zone = await state._prioritized_krx_codes()

    assert len(result) <= RealtimeFeed.MAX_REGISTRATIONS
    assert len(result) == state._CORE_FEED_CODES + state._CB_ONLY_ZONE_SIZE


async def test_candidate_promoted_to_held_moves_to_core_others_stay_cb_only(monkeypatch):
    """Mirrors reconcile_feed_tier_promotions's real-world trigger: candidate
    A gets bought (added to the trade allowlist + becomes a held position,
    same as add_ticker_to_trade_allowlist's existing behavior), while B and C
    are still bare screening candidates that never made the allowlist."""
    from app.execution.portfolio_risk import PortfolioSnapshot

    # Pad the allowlist out to a full core (12) so `result[:_CORE_FEED_CODES]`
    # unambiguously means "core" — with a 1-code allowlist, `result` itself
    # would be shorter than _CORE_FEED_CODES and result[:12] would spill past
    # the (mostly empty) core into the cb-only tail, wrongly reading as if
    # the candidates were "in core".
    allowlist = ["441270.KS"] + [f"{100000 + i}.KS" for i in range(state._CORE_FEED_CODES - 1)]
    monkeypatch.setattr(settings, "trade_allowlist_domestic", allowlist)
    monkeypatch.setattr(settings, "trade_allowlist_overseas", [])

    for code in ("441270", "091990", "060310"):
        state.note_candidate_cb_code(code)

    async def fake_snapshot(market):
        return PortfolioSnapshot(per_ticker_value={"441270.KS": 1.0}, total_value=0.0)

    monkeypatch.setattr("app.execution.portfolio_risk.real_account_snapshot", fake_snapshot)

    result, _orderbook_zone, cb_only_zone = await state._prioritized_krx_codes()

    core = result[: state._CORE_FEED_CODES]
    assert "441270" in core
    assert {"091990", "060310"} <= cb_only_zone
    assert not ({"091990", "060310"} & set(core))


# ---------------------------------------------------------------------------
# _reconciliation_pass / trade_reconciliation_loop — 2026-08-13: a
# discrepancy persists as a proper alert (not just a log line), so it
# survives past the moment it's printed. See app.execution.reconciliation
# for the underlying detect-never-correct contract this doesn't change.
# 2026-09-07: converted from a boot-time-only check into a periodic loop
# (trade_reconciliation_loop) — see that function's own docstring for why,
# and _reported_discrepancies' own comment for the in-process dedup this
# needed once re-running became routine instead of once-per-process.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_reconciliation_dedup():
    state._reported_discrepancies.clear()
    yield
    state._reported_discrepancies.clear()


def _discrepancy(trade_id=42, ticker="008290.KS", severity="CRITICAL"):
    return TradeDiscrepancy(
        trade_id=trade_id, ticker=ticker, trade_date=date(2026, 7, 22), side="buy", shares=10,
        created_at=datetime(2026, 7, 22, 9, 5), severity=severity,
        note="no matching KIS execution found for this ticker/side/date — possible crash before the KIS mirror completed",
    )


async def test_reconciliation_pass_records_an_alert_per_discrepancy(monkeypatch):
    state.kis_broker = broker_stub(object())  # any non-None broker

    async def fake_fetch_recent_local_trades(lookback, grace_period):
        return ["a local trade row — content unused by this fake"]

    async def fake_recent_executions():
        return [], []

    monkeypatch.setattr(state, "fetch_recent_local_trades", fake_fetch_recent_local_trades)
    monkeypatch.setattr(state, "recent_executions", fake_recent_executions)

    discrepancy = _discrepancy()

    def fake_match_trades_against_kis(local_trades, dom, ovs):
        return [discrepancy]

    monkeypatch.setattr(state, "match_trades_against_kis", fake_match_trades_against_kis)

    recorded = []

    async def fake_record_alert(alert):
        recorded.append(alert)

    monkeypatch.setattr(state, "record_alert", fake_record_alert)

    await state._reconciliation_pass()

    assert len(recorded) == 1
    assert recorded[0].kind == "unmirrored_trade"
    assert recorded[0].ticker == "008290.KS"
    assert "CRITICAL" in recorded[0].message
    assert "no matching KIS execution found" in recorded[0].message


async def test_reconciliation_pass_records_no_alert_when_nothing_found(monkeypatch):
    state.kis_broker = broker_stub(object())

    async def fake_fetch_recent_local_trades(lookback, grace_period):
        return ["a local trade row"]

    async def fake_recent_executions():
        return [], []

    monkeypatch.setattr(state, "fetch_recent_local_trades", fake_fetch_recent_local_trades)
    monkeypatch.setattr(state, "recent_executions", fake_recent_executions)

    def fake_match_trades_against_kis(local_trades, dom, ovs):
        return []

    monkeypatch.setattr(state, "match_trades_against_kis", fake_match_trades_against_kis)

    async def fail_if_called(alert):
        raise AssertionError("must not record an alert when there are no discrepancies")

    monkeypatch.setattr(state, "record_alert", fail_if_called)

    await state._reconciliation_pass()  # no raise = pass


async def test_reconciliation_pass_skips_entirely_without_a_broker(monkeypatch):
    state.kis_broker = broker_stub(None)

    async def fail_if_called(*a, **kw):
        raise AssertionError("must not fetch local trades or reconcile without a configured broker")

    monkeypatch.setattr(state, "fetch_recent_local_trades", fail_if_called)
    monkeypatch.setattr(state, "recent_executions", fail_if_called)

    await state._reconciliation_pass()  # no raise = pass


async def test_reconciliation_pass_skips_kis_call_when_no_local_trades(monkeypatch):
    # FIX-4-style optimization (2026-09-07, matches app.live.fill_monitor.
    # fill_monitor_loop's identical guard): a periodic pass with nothing
    # local to check must never fire the KIS execution-history TR — that
    # was the whole point of doing this before the loop existed only once.
    state.kis_broker = broker_stub(object())

    async def fake_fetch_recent_local_trades(lookback, grace_period):
        return []

    async def fail_if_called():
        raise AssertionError("must not fetch KIS executions when there are no local trades to check")

    monkeypatch.setattr(state, "fetch_recent_local_trades", fake_fetch_recent_local_trades)
    monkeypatch.setattr(state, "recent_executions", fail_if_called)

    await state._reconciliation_pass()  # no raise = pass


async def test_reconciliation_pass_does_not_realert_the_same_unresolved_discrepancy(monkeypatch):
    state.kis_broker = broker_stub(object())

    async def fake_fetch_recent_local_trades(lookback, grace_period):
        return ["a local trade row"]

    async def fake_recent_executions():
        return [], []

    monkeypatch.setattr(state, "fetch_recent_local_trades", fake_fetch_recent_local_trades)
    monkeypatch.setattr(state, "recent_executions", fake_recent_executions)
    # Isolates this test from app.live.alerts._last_fired_at's real,
    # process-wide (account_mode, ticker, kind) cooldown — this test is
    # about _reported_discrepancies' trade_id-level dedup specifically,
    # which runs BEFORE filter_new_alerts and must not be entangled with
    # that separate, ticker-keyed mechanism (see _reported_discrepancies'
    # own comment on why the two are not redundant).
    monkeypatch.setattr(state, "filter_new_alerts", lambda alerts: alerts)

    discrepancy = _discrepancy(trade_id=99)

    def fake_match_trades_against_kis(local_trades, dom, ovs):
        return [discrepancy]  # same unresolved discrepancy, every pass

    monkeypatch.setattr(state, "match_trades_against_kis", fake_match_trades_against_kis)

    recorded = []

    async def fake_record_alert(alert):
        recorded.append(alert)

    monkeypatch.setattr(state, "record_alert", fake_record_alert)

    await state._reconciliation_pass()
    await state._reconciliation_pass()
    await state._reconciliation_pass()

    assert len(recorded) == 1  # not 3 — filter_new_alerts' cooldown alone would have let this repeat


async def test_reconciliation_pass_realerts_a_different_trade_id(monkeypatch):
    state.kis_broker = broker_stub(object())

    async def fake_fetch_recent_local_trades(lookback, grace_period):
        return ["a local trade row"]

    async def fake_recent_executions():
        return [], []

    monkeypatch.setattr(state, "fetch_recent_local_trades", fake_fetch_recent_local_trades)
    monkeypatch.setattr(state, "recent_executions", fake_recent_executions)
    # Same isolation as the test above — see its comment.
    monkeypatch.setattr(state, "filter_new_alerts", lambda alerts: alerts)

    calls = {"n": 0}

    def fake_match_trades_against_kis(local_trades, dom, ovs):
        calls["n"] += 1
        return [_discrepancy(trade_id=calls["n"])]  # a NEW trade_id each pass

    monkeypatch.setattr(state, "match_trades_against_kis", fake_match_trades_against_kis)

    recorded = []

    async def fake_record_alert(alert):
        recorded.append(alert)

    monkeypatch.setattr(state, "record_alert", fake_record_alert)

    await state._reconciliation_pass()
    await state._reconciliation_pass()

    assert len(recorded) == 2  # different trade_id each time — dedup must not conflate them


# ---------------------------------------------------------------------------
# trade_reconciliation_loop
# ---------------------------------------------------------------------------


async def test_trade_reconciliation_loop_runs_once_per_scope_per_interval(monkeypatch):
    from app.user_context import ServicedScope

    monkeypatch.setattr(state.settings, "trade_reconciliation_interval_seconds", 0.0)
    monkeypatch.setattr(
        state, "serviced_scopes", _async_return([ServicedScope(user_id="u1", account_mode="real")])
    )

    calls = {"n": 0}

    async def fake_pass():
        calls["n"] += 1
        if calls["n"] >= 3:
            raise asyncio.CancelledError

    monkeypatch.setattr(state, "_reconciliation_pass", fake_pass)

    with pytest.raises(asyncio.CancelledError):
        await state.trade_reconciliation_loop()

    assert calls["n"] == 3


async def test_trade_reconciliation_loop_continues_past_a_failed_scope(monkeypatch):
    from app.user_context import ServicedScope

    monkeypatch.setattr(state.settings, "trade_reconciliation_interval_seconds", 0.0)
    monkeypatch.setattr(
        state,
        "serviced_scopes",
        _async_return([
            ServicedScope(user_id="broken", account_mode="real"),
            ServicedScope(user_id="fine", account_mode="real"),
        ]),
    )

    seen = []

    async def fake_pass():
        from app.user_context import get_current_user_id

        user_id = get_current_user_id()
        seen.append(user_id)
        if user_id == "broken":
            raise RuntimeError("boom")
        raise asyncio.CancelledError  # stop the loop once the healthy scope also ran

    monkeypatch.setattr(state, "_reconciliation_pass", fake_pass)

    with pytest.raises(asyncio.CancelledError):
        await state.trade_reconciliation_loop()

    assert seen == ["broken", "fine"]  # the failure did not skip the next scope


def _async_return(value):
    async def _fn():
        return value

    return _fn
