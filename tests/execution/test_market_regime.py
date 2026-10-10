"""app.execution.market_regime — Faber trend signal (pure calc), refresh
idempotency, and cash-floor resolution. No real DB/network/KIS — pool,
connection, and broker all faked."""

from __future__ import annotations

from datetime import date

import pytest

from app.domain.models import IndexDailyBar
from app.execution import market_regime
from app.web import state
from conftest import broker_stub


class _AcquireCM:
    def __init__(self, conn):
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, *exc):
        return False


class FakeConn:
    def __init__(self, fetchrow_result=None):
        self.fetchrow_result = fetchrow_result
        self.executed: list[tuple] = []
        self.queries: list[tuple] = []

    async def fetchrow(self, query, *args):
        self.queries.append((query, args))
        return self.fetchrow_result

    async def execute(self, query, *args):
        if "CREATE TABLE" in query:
            return
        self.executed.append((query, args))


class FakePool:
    def __init__(self, conn):
        self._conn = conn

    def acquire(self):
        return _AcquireCM(self._conn)


class FakeBroker:
    def __init__(self, bars):
        self._bars = bars

    async def fetch_index_daily_chart(self, code, start, end, period="D"):
        return self._bars


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    monkeypatch.setattr(market_regime, "_schema_ready", False)
    monkeypatch.setattr(state, "kis_broker", broker_stub(None))


def _bars(closes: list[float]) -> list[IndexDailyBar]:
    # Ascending fake dates, oldest first — _faber_signal must not assume order.
    return [IndexDailyBar(day=f"202601{str(i + 1).zfill(2)}", close=c) for i, c in enumerate(closes)]


# ---------------------------------------------------------------------------
# _faber_signal — pure
# ---------------------------------------------------------------------------


def test_faber_signal_none_when_fewer_bars_than_window():
    closes_by_day = [("20260101", 100.0), ("20260102", 101.0)]
    assert market_regime._faber_signal(closes_by_day, window=5) is None


def test_faber_signal_risk_on_when_price_above_sma():
    # 5 flat bars at 100, then a close well above -> above its own SMA -> Risk-On.
    closes_by_day = [(f"2026010{i}", 100.0) for i in range(1, 6)] + [("20260106", 110.0)]
    is_risk_off, latest, sma = market_regime._faber_signal(closes_by_day, window=5)
    assert is_risk_off is False
    assert latest == 110.0


def test_faber_signal_risk_off_when_price_below_sma():
    closes_by_day = [(f"2026010{i}", 100.0) for i in range(1, 6)] + [("20260106", 90.0)]
    is_risk_off, latest, sma = market_regime._faber_signal(closes_by_day, window=5)
    assert is_risk_off is True
    assert latest == 90.0


def test_faber_signal_sorts_by_day_regardless_of_input_order():
    # Deliberately out of order — the KIS TR doesn't guarantee an order.
    closes_by_day = [("20260106", 90.0), ("20260102", 100.0), ("20260101", 100.0),
                      ("20260104", 100.0), ("20260103", 100.0), ("20260105", 100.0)]
    is_risk_off, latest, sma = market_regime._faber_signal(closes_by_day, window=5)
    assert latest == 90.0  # the last-by-DATE close, not the last list element
    assert is_risk_off is True


# ---------------------------------------------------------------------------
# refresh_domestic_regime
# ---------------------------------------------------------------------------


async def test_refresh_skips_when_already_computed_today(monkeypatch):
    conn = FakeConn(fetchrow_result={"computed_on": date(2026, 8, 3)})
    monkeypatch.setattr(market_regime, "_ready_pool", lambda: _async_return(FakePool(conn)))
    monkeypatch.setattr(market_regime, "local_today", lambda market: date(2026, 8, 3))

    async def fail_if_called(*a, **k):
        raise AssertionError("must not fetch broker data when already refreshed today")

    monkeypatch.setattr(state, "kis_broker", broker_stub(type("B", (), {"fetch_index_daily_chart": fail_if_called})()))

    await market_regime.refresh_domestic_regime()
    assert conn.executed == []  # no INSERT/UPDATE happened


async def test_refresh_recomputes_when_forced_even_if_already_done_today(monkeypatch):
    conn = FakeConn(fetchrow_result={"computed_on": date(2026, 8, 3)})
    monkeypatch.setattr(market_regime, "_ready_pool", lambda: _async_return(FakePool(conn)))
    monkeypatch.setattr(market_regime, "local_today", lambda market: date(2026, 8, 3))
    monkeypatch.setattr(market_regime.settings, "faber_sma_window_days", 5)
    monkeypatch.setattr(state, "kis_broker", broker_stub(FakeBroker(_bars([100.0] * 4 + [110.0]))))

    await market_regime.refresh_domestic_regime(force=True)
    assert len(conn.executed) == 1
    _, args = conn.executed[0]
    assert args[0] == "domestic"
    assert args[1] is False  # is_risk_off — 110 above SMA(100) -> Risk-On


async def test_refresh_skips_quietly_when_broker_not_connected(monkeypatch):
    conn = FakeConn(fetchrow_result=None)
    monkeypatch.setattr(market_regime, "_ready_pool", lambda: _async_return(FakePool(conn)))
    monkeypatch.setattr(market_regime, "local_today", lambda market: date(2026, 8, 3))
    # state._kis_broker is None (autouse fixture) — must return quietly, not raise.
    await market_regime.refresh_domestic_regime()
    assert conn.executed == []


async def test_refresh_skips_quietly_on_insufficient_history(monkeypatch):
    conn = FakeConn(fetchrow_result=None)
    monkeypatch.setattr(market_regime, "_ready_pool", lambda: _async_return(FakePool(conn)))
    monkeypatch.setattr(market_regime, "local_today", lambda market: date(2026, 8, 3))
    monkeypatch.setattr(market_regime.settings, "faber_sma_window_days", 200)
    monkeypatch.setattr(state, "kis_broker", broker_stub(FakeBroker(_bars([100.0, 101.0]))))  # far fewer than 200

    await market_regime.refresh_domestic_regime()
    assert conn.executed == []


async def test_refresh_persists_risk_off_signal(monkeypatch):
    conn = FakeConn(fetchrow_result=None)
    monkeypatch.setattr(market_regime, "_ready_pool", lambda: _async_return(FakePool(conn)))
    monkeypatch.setattr(market_regime, "local_today", lambda market: date(2026, 8, 3))
    monkeypatch.setattr(market_regime.settings, "faber_sma_window_days", 5)
    monkeypatch.setattr(state, "kis_broker", broker_stub(FakeBroker(_bars([100.0] * 4 + [90.0]))))

    await market_regime.refresh_domestic_regime()
    assert len(conn.executed) == 1
    query, args = conn.executed[0]
    assert "INSERT INTO market_regime_state" in query
    assert args[0] == "domestic"
    assert args[1] is True  # is_risk_off


# ---------------------------------------------------------------------------
# refresh_overseas_regime — SPY via kis_provider.get_price_series, no broker
# ---------------------------------------------------------------------------


def _price_series(closes: list[float]) -> dict:
    return {date(2026, 1, i + 1): c for i, c in enumerate(closes)}


async def test_refresh_overseas_skips_when_already_computed_today(monkeypatch):
    conn = FakeConn(fetchrow_result={"computed_on": date(2026, 8, 3)})
    monkeypatch.setattr(market_regime, "_ready_pool", lambda: _async_return(FakePool(conn)))
    monkeypatch.setattr(market_regime, "local_today", lambda market: date(2026, 8, 3))

    async def fail_if_called(ticker, start, end):
        raise AssertionError("must not fetch SPY history when already refreshed today")

    import app.dataflows.kis_provider as kis_provider
    monkeypatch.setattr(kis_provider, "get_price_series", fail_if_called)

    await market_regime.refresh_overseas_regime()
    assert conn.executed == []


async def test_refresh_overseas_does_not_require_a_kis_broker(monkeypatch):
    # No state._kis_broker at all (autouse fixture sets it to None) — SPY
    # history comes from get_price_series (KIS-first/yfinance-fallback
    # internally), not a direct broker call, so this must still work.
    conn = FakeConn(fetchrow_result=None)
    monkeypatch.setattr(market_regime, "_ready_pool", lambda: _async_return(FakePool(conn)))
    monkeypatch.setattr(market_regime, "local_today", lambda market: date(2026, 8, 3))
    monkeypatch.setattr(market_regime.settings, "faber_sma_window_days", 5)

    import app.dataflows.kis_provider as kis_provider

    async def fake_get_price_series(ticker, start, end):
        assert ticker == "SPY"
        return _price_series([100.0] * 4 + [110.0])

    monkeypatch.setattr(kis_provider, "get_price_series", fake_get_price_series)

    await market_regime.refresh_overseas_regime()
    assert len(conn.executed) == 1
    _, args = conn.executed[0]
    assert args[0] == "overseas"
    assert args[1] is False  # is_risk_off — 110 above SMA(100) -> Risk-On


async def test_refresh_overseas_persists_risk_off_signal(monkeypatch):
    conn = FakeConn(fetchrow_result=None)
    monkeypatch.setattr(market_regime, "_ready_pool", lambda: _async_return(FakePool(conn)))
    monkeypatch.setattr(market_regime, "local_today", lambda market: date(2026, 8, 3))
    monkeypatch.setattr(market_regime.settings, "faber_sma_window_days", 5)

    import app.dataflows.kis_provider as kis_provider

    async def fake_get_price_series(ticker, start, end):
        return _price_series([100.0] * 4 + [90.0])

    monkeypatch.setattr(kis_provider, "get_price_series", fake_get_price_series)

    await market_regime.refresh_overseas_regime()
    query, args = conn.executed[0]
    assert "INSERT INTO market_regime_state" in query
    assert args[0] == "overseas"
    assert args[1] is True  # is_risk_off


async def test_refresh_overseas_skips_quietly_on_insufficient_history(monkeypatch):
    conn = FakeConn(fetchrow_result=None)
    monkeypatch.setattr(market_regime, "_ready_pool", lambda: _async_return(FakePool(conn)))
    monkeypatch.setattr(market_regime, "local_today", lambda market: date(2026, 8, 3))
    monkeypatch.setattr(market_regime.settings, "faber_sma_window_days", 200)

    import app.dataflows.kis_provider as kis_provider

    async def fake_get_price_series(ticker, start, end):
        return _price_series([100.0, 101.0])  # far fewer than 200

    monkeypatch.setattr(kis_provider, "get_price_series", fake_get_price_series)

    await market_regime.refresh_overseas_regime()
    assert conn.executed == []


async def test_refresh_overseas_skips_quietly_on_fetch_failure(monkeypatch):
    conn = FakeConn(fetchrow_result=None)
    monkeypatch.setattr(market_regime, "_ready_pool", lambda: _async_return(FakePool(conn)))
    monkeypatch.setattr(market_regime, "local_today", lambda market: date(2026, 8, 3))

    import app.dataflows.kis_provider as kis_provider

    async def broken_get_price_series(ticker, start, end):
        raise RuntimeError("network down")

    monkeypatch.setattr(kis_provider, "get_price_series", broken_get_price_series)

    await market_regime.refresh_overseas_regime()  # must not raise
    assert conn.executed == []


# ---------------------------------------------------------------------------
# domestic_cash_floor_pct
# ---------------------------------------------------------------------------


async def test_domestic_cash_floor_uses_risk_off_value_when_flagged(monkeypatch):
    conn = FakeConn(fetchrow_result={"is_risk_off": True})
    monkeypatch.setattr(market_regime, "_ready_pool", lambda: _async_return(FakePool(conn)))
    monkeypatch.setattr(market_regime.settings, "portfolio_cash_floor_pct", 0.20)
    monkeypatch.setattr(market_regime.settings, "portfolio_cash_floor_pct_risk_off", 0.35)

    assert await market_regime.domestic_cash_floor_pct() == 0.35


async def test_domestic_cash_floor_uses_risk_on_value_when_not_flagged(monkeypatch):
    conn = FakeConn(fetchrow_result={"is_risk_off": False})
    monkeypatch.setattr(market_regime, "_ready_pool", lambda: _async_return(FakePool(conn)))
    monkeypatch.setattr(market_regime.settings, "portfolio_cash_floor_pct", 0.20)
    monkeypatch.setattr(market_regime.settings, "portfolio_cash_floor_pct_risk_off", 0.35)

    assert await market_regime.domestic_cash_floor_pct() == 0.20


async def test_domestic_cash_floor_degrades_to_risk_on_value_when_no_signal_yet(monkeypatch):
    conn = FakeConn(fetchrow_result=None)
    monkeypatch.setattr(market_regime, "_ready_pool", lambda: _async_return(FakePool(conn)))
    monkeypatch.setattr(market_regime.settings, "portfolio_cash_floor_pct", 0.20)

    assert await market_regime.domestic_cash_floor_pct() == 0.20


async def test_domestic_cash_floor_degrades_to_risk_on_value_on_db_failure(monkeypatch):
    async def broken_ready_pool():
        raise RuntimeError("pool unavailable")

    monkeypatch.setattr(market_regime, "_ready_pool", broken_ready_pool)
    monkeypatch.setattr(market_regime.settings, "portfolio_cash_floor_pct", 0.20)

    assert await market_regime.domestic_cash_floor_pct() == 0.20


# ---------------------------------------------------------------------------
# overseas_cash_floor_pct
# ---------------------------------------------------------------------------


async def test_overseas_cash_floor_uses_risk_off_value_when_flagged(monkeypatch):
    conn = FakeConn(fetchrow_result={"is_risk_off": True})
    monkeypatch.setattr(market_regime, "_ready_pool", lambda: _async_return(FakePool(conn)))
    monkeypatch.setattr(market_regime.settings, "portfolio_cash_floor_pct", 0.20)
    monkeypatch.setattr(market_regime.settings, "portfolio_cash_floor_pct_risk_off", 0.35)

    assert await market_regime.overseas_cash_floor_pct() == 0.35


async def test_overseas_cash_floor_queries_the_overseas_row_not_domestic(monkeypatch):
    conn = FakeConn(fetchrow_result={"is_risk_off": True})
    monkeypatch.setattr(market_regime, "_ready_pool", lambda: _async_return(FakePool(conn)))
    monkeypatch.setattr(market_regime.settings, "portfolio_cash_floor_pct_risk_off", 0.35)

    await market_regime.overseas_cash_floor_pct()
    query, args = conn.queries[-1]
    assert args == ("overseas",)


async def test_overseas_cash_floor_degrades_to_risk_on_value_when_no_signal_yet(monkeypatch):
    conn = FakeConn(fetchrow_result=None)
    monkeypatch.setattr(market_regime, "_ready_pool", lambda: _async_return(FakePool(conn)))
    monkeypatch.setattr(market_regime.settings, "portfolio_cash_floor_pct", 0.20)

    assert await market_regime.overseas_cash_floor_pct() == 0.20


async def _async_return(value):
    return value
