"""app.audit.reconciliation_report — broker and the DB-backed local-trades
fetch are faked; match_trades_against_kis itself (pure) is already covered
directly in tests/execution/test_reconciliation.py, so these only exercise
this module's own composition/rendering (E-stage)."""

from __future__ import annotations

from datetime import date, datetime, timedelta

import pytest

from app.audit import reconciliation_report as rr
from app.execution.reconciliation import LocalTrade, TradeDiscrepancy


class FakeBroker:
    def __init__(self, domestic=None, overseas=None):
        self._domestic = domestic or []
        self._overseas = overseas or []

    async def fetch_daily_executions(self, start, end):
        return self._domestic

    async def fetch_overseas_executions(self, start, end):
        return self._overseas


async def _async_return(value):
    return value


class FakeConn:
    """Stands in for the asyncpg connection the new DB health probe uses."""

    def __init__(self, fail=False):
        self._fail = fail

    async def fetchval(self, query):
        if self._fail:
            raise RuntimeError("db unavailable")
        return 1


class FakeAcquireCtx:
    def __init__(self, conn):
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, *exc_info):
        return False


class FakePool:
    """Stands in for `app.memory.db.get_pool()`'s return value."""

    def __init__(self, fail=False):
        self._fail = fail

    def acquire(self):
        return FakeAcquireCtx(FakeConn(fail=self._fail))


def _patch_healthy_pool(monkeypatch):
    monkeypatch.setattr(rr, "get_pool", lambda: _async_return(FakePool()))


async def test_compute_reconciliation_raises_without_a_broker(monkeypatch):
    _patch_healthy_pool(monkeypatch)
    monkeypatch.setattr(rr, "get_broker", lambda name: _async_return(None))
    with pytest.raises(RuntimeError, match="KIS broker not configured"):
        await rr.compute_reconciliation()


async def test_compute_reconciliation_raises_on_db_probe_failure(monkeypatch):
    """A genuine DB outage must surface as a visible exception, not render
    downstream as a clean "0 trades checked" report (see the module
    docstring's rationale for the probe)."""
    monkeypatch.setattr(rr, "get_pool", lambda: _async_return(FakePool(fail=True)))
    monkeypatch.setattr(rr, "get_broker", lambda name: _async_return(FakeBroker()))

    async def fake_fetch(lookback, grace_period):
        return []

    monkeypatch.setattr(rr, "fetch_recent_local_trades", fake_fetch)
    with pytest.raises(RuntimeError, match="db unavailable"):
        await rr.compute_reconciliation()


async def test_compute_reconciliation_reports_no_trades_checked(monkeypatch):
    _patch_healthy_pool(monkeypatch)
    monkeypatch.setattr(rr, "get_broker", lambda name: _async_return(FakeBroker()))

    async def fake_fetch(lookback, grace_period):
        return []

    monkeypatch.setattr(rr, "fetch_recent_local_trades", fake_fetch)
    result = await rr.compute_reconciliation()
    assert result.trades_checked == 0
    assert result.discrepancies == []


async def test_compute_reconciliation_finds_a_ledger_only_trade(monkeypatch):
    _patch_healthy_pool(monkeypatch)
    monkeypatch.setattr(rr, "get_broker", lambda name: _async_return(FakeBroker()))  # no KIS executions at all

    local = LocalTrade(
        id=1, ticker="008290.KS", trade_date=date(2026, 7, 22), side="buy", shares=10,
        created_at=datetime(2026, 7, 22, 9, 5), reason="rebalance to target",
    )

    async def fake_fetch(lookback, grace_period):
        return [local]

    monkeypatch.setattr(rr, "fetch_recent_local_trades", fake_fetch)
    result = await rr.compute_reconciliation(days=30)

    assert result.trades_checked == 1
    assert len(result.discrepancies) == 1
    assert result.discrepancies[0].severity == "CRITICAL"
    assert result.discrepancies[0].ticker == "008290.KS"


async def test_compute_reconciliation_caps_days_to_kis_lookback_limit(monkeypatch):
    seen_lookback = {}

    _patch_healthy_pool(monkeypatch)
    monkeypatch.setattr(rr, "get_broker", lambda name: _async_return(FakeBroker()))

    async def fake_fetch(lookback, grace_period):
        seen_lookback["value"] = lookback
        return []

    monkeypatch.setattr(rr, "fetch_recent_local_trades", fake_fetch)
    await rr.compute_reconciliation(days=365)

    assert seen_lookback["value"] == timedelta(days=rr._KIS_LOOKBACK_LIMIT_DAYS)


def test_render_reconciliation_no_trades():
    result = rr.ReconciliationReport(discrepancies=[], trades_checked=0)
    assert "확인 대상 거래가 없습니다" in rr.render_reconciliation(result)


def test_render_reconciliation_all_clean():
    result = rr.ReconciliationReport(discrepancies=[], trades_checked=5)
    rendered = rr.render_reconciliation(result)
    assert "5건 전부" in rendered


def test_render_reconciliation_groups_by_severity():
    critical = TradeDiscrepancy(
        trade_id=1, ticker="008290.KS", trade_date=date(2026, 7, 22), side="buy", shares=10,
        created_at=datetime(2026, 7, 22, 9, 5), severity="CRITICAL", note="no matching KIS execution found",
    )
    warning = TradeDiscrepancy(
        trade_id=2, ticker="AAPL", trade_date=date(2026, 8, 1), side="sell", shares=5,
        created_at=datetime(2026, 8, 1, 10, 0), severity="WARNING", note="KIS filled only 3/5 shares",
    )
    result = rr.ReconciliationReport(discrepancies=[critical, warning], trades_checked=10)
    rendered = rr.render_reconciliation(result)

    assert "CRITICAL 1" in rendered
    assert "WARNING 1" in rendered
    assert "[CRITICAL]" in rendered
    assert "[WARNING]" in rendered
    assert "008290.KS" in rendered
    assert "AAPL" in rendered
    assert "자동으로 고치지 않습니다" in rendered
