"""app.backtest.experiments — asyncpg pool/connection faked, no real Postgres,
no real git subprocess (F-stage)."""

from __future__ import annotations

from datetime import date, datetime

import pytest

from app.backtest import experiments
from app.backtest.metrics import RiskMetrics


class FakeConn:
    def __init__(self, fetchrow_result=None, fetch_result=None):
        self.fetchrow_result = fetchrow_result
        self.fetch_result = fetch_result if fetch_result is not None else []
        self.queries: list[tuple] = []

    async def fetchrow(self, query, *args):
        self.queries.append((query, args))
        return self.fetchrow_result

    async def fetch(self, query, *args):
        self.queries.append((query, args))
        return self.fetch_result


class _AcquireCM:
    def __init__(self, conn):
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, *exc):
        return False


class FakePool:
    def __init__(self, conn):
        self._conn = conn

    def acquire(self):
        return _AcquireCM(self._conn)


async def _async_return(value):
    return value


@pytest.fixture
def fake_git(monkeypatch):
    monkeypatch.setattr(experiments, "current_git_commit", lambda: "abc1234")


async def test_record_experiment_inserts_and_returns_id(monkeypatch, fake_git):
    conn = FakeConn(fetchrow_result={"id": 42})
    monkeypatch.setattr(experiments, "get_pool", lambda: _async_return(FakePool(conn)))
    exp_id = await experiments.record_experiment(
        kind="single",
        tickers=["AAPL"],
        benchmark_ticker="SPY",
        start=date(2026, 4, 1),
        end=date(2026, 6, 1),
        cadence_days=7,
        n_dates=9,
        return_pct=0.05,
        benchmark_return_pct=0.03,
        risk_metrics=RiskMetrics(sharpe=1.2, sortino=1.5, max_drawdown=-0.1, cvar_95=-0.05),
        label="prompt-v2",
    )
    assert exp_id == 42
    query, args = conn.queries[0]
    assert "INSERT INTO backtest_runs" in query
    assert args[0] == "abc1234"  # git_commit
    assert args[1] == "prompt-v2"  # label
    assert args[3] == "AAPL"  # tickers, comma-joined
    assert len(args) == 18
    assert args[17] is None  # exposure_return_pct defaults to None


async def test_record_experiment_carries_exposure_return_pct(monkeypatch, fake_git):
    conn = FakeConn(fetchrow_result={"id": 1})
    monkeypatch.setattr(experiments, "get_pool", lambda: _async_return(FakePool(conn)))
    await experiments.record_experiment(
        kind="single",
        tickers=["AAPL"],
        benchmark_ticker="SPY",
        start=date(2026, 4, 1),
        end=date(2026, 6, 1),
        cadence_days=7,
        n_dates=9,
        return_pct=0.05,
        benchmark_return_pct=0.03,
        risk_metrics=RiskMetrics(sharpe=1.2, sortino=1.5, max_drawdown=-0.1, cvar_95=-0.05),
        exposure_return_pct=0.42,
    )
    _, args = conn.queries[0]
    assert args[17] == 0.42


async def test_record_experiment_joins_multiple_tickers(monkeypatch, fake_git):
    conn = FakeConn(fetchrow_result={"id": 1})
    monkeypatch.setattr(experiments, "get_pool", lambda: _async_return(FakePool(conn)))
    await experiments.record_experiment(
        kind="portfolio",
        tickers=["AAPL", "MSFT"],
        benchmark_ticker="SPY",
        start=date(2026, 4, 1),
        end=date(2026, 6, 1),
        cadence_days=7,
        n_dates=9,
        return_pct=0.05,
        benchmark_return_pct=0.03,
        risk_metrics=RiskMetrics(sharpe=None, sortino=None, max_drawdown=None, cvar_95=None),
    )
    _, args = conn.queries[0]
    assert args[3] == "AAPL,MSFT"


async def test_list_experiments_parses_rows(monkeypatch):
    row = {
        "id": 1, "created_at": datetime(2026, 7, 23, 10, 0), "git_commit": "abc1234", "label": "v2",
        "kind": "single", "tickers": "AAPL", "benchmark_ticker": "SPY",
        "start_date": date(2026, 4, 1), "end_date": date(2026, 6, 1), "cadence_days": 7, "n_dates": 9,
        "return_pct": 0.05, "exposure_return_pct": None, "benchmark_return_pct": 0.03,
        "sharpe": 1.2, "sortino": 1.5, "max_drawdown": -0.1, "cvar_95": -0.05,
        "win_rate": 0.6, "mean_alpha": 0.02,
    }
    conn = FakeConn(fetch_result=[row])
    monkeypatch.setattr(experiments, "get_pool", lambda: _async_return(FakePool(conn)))
    result = await experiments.list_experiments(limit=10)
    assert len(result) == 1
    assert result[0].id == 1 and result[0].git_commit == "abc1234"
    query, args = conn.queries[0]
    assert "ORDER BY created_at DESC" in query
    assert args == (10,)


# --- list_experiments label_prefix scoping (2026-09-11 fix) --------------
#
# Bug: `limit` bounded the most recent rows across the WHOLE backtest_runs
# table, not rows matching a caller's own sweep label prefix -- a caller
# (app.backtest.atr_sweep_report) scoping to one sweep could silently lose
# its own older rows to unrelated experiments recorded since.


async def test_list_experiments_no_prefix_behaves_as_before(monkeypatch):
    # Regression check for the existing caller (app.audit.__main__), which
    # must keep getting the unfiltered whole-table query.
    conn = FakeConn(fetch_result=[])
    monkeypatch.setattr(experiments, "get_pool", lambda: _async_return(FakePool(conn)))
    await experiments.list_experiments(limit=15)
    query, args = conn.queries[0]
    assert "WHERE" not in query
    assert args == (15,)


async def test_list_experiments_with_label_prefix_filters_at_sql_level(monkeypatch):
    conn = FakeConn(fetch_result=[])
    monkeypatch.setattr(experiments, "get_pool", lambda: _async_return(FakePool(conn)))
    await experiments.list_experiments(limit=5, label_prefix="trailing-sweep-atr")
    query, args = conn.queries[0]
    assert "WHERE label LIKE $2" in query
    assert args == (5, "trailing-sweep-atr%")


async def test_list_experiments_label_prefix_scopes_limit_to_matching_rows(monkeypatch):
    # Construct enough non-matching rows that an unfiltered call would
    # exceed `limit` before any matching row is reached -- the fix's SQL-
    # level WHERE means the fake DB layer (standing in for a real Postgres
    # WHERE+LIMIT) only ever returns matching rows here, confirming the
    # query the caller issues is scoped BEFORE limit, not filtered after.
    def _row(label: str) -> dict:
        return {
            "id": 1, "created_at": datetime(2026, 7, 23, 10, 0), "git_commit": "abc1234", "label": label,
            "kind": "single", "tickers": "AAPL", "benchmark_ticker": "SPY",
            "start_date": date(2026, 4, 1), "end_date": date(2026, 6, 1), "cadence_days": 7, "n_dates": 9,
            "return_pct": 0.05, "exposure_return_pct": None, "benchmark_return_pct": 0.03,
            "sharpe": 1.2, "sortino": 1.5, "max_drawdown": -0.1, "cvar_95": -0.05,
            "win_rate": 0.6, "mean_alpha": 0.02,
        }

    # Only matching rows are returned -- standing in for a real WHERE clause
    # having already excluded the unrelated experiments from the result set.
    matching_rows = [_row("trailing-sweep-atr1.5"), _row("trailing-sweep-atr2.0")]
    conn = FakeConn(fetch_result=matching_rows)
    monkeypatch.setattr(experiments, "get_pool", lambda: _async_return(FakePool(conn)))
    result = await experiments.list_experiments(limit=2, label_prefix="trailing-sweep-atr")
    assert len(result) == 2
    assert all(r.label.startswith("trailing-sweep-atr") for r in result)
    query, args = conn.queries[0]
    assert args == (2, "trailing-sweep-atr%")


def test_render_experiments_empty():
    assert "No experiments" in experiments.render_experiments([])


def test_render_experiments_shows_row():
    rec = experiments.ExperimentRecord(
        id=1, created_at=datetime(2026, 7, 23, 10, 0), git_commit="abc1234", label="prompt-v2",
        kind="single", tickers="AAPL", benchmark_ticker="SPY",
        start_date=date(2026, 4, 1), end_date=date(2026, 6, 1), cadence_days=7, n_dates=9,
        return_pct=0.05, exposure_return_pct=None, benchmark_return_pct=0.03,
        sharpe=1.2, sortino=1.5, max_drawdown=-0.1, cvar_95=-0.05,
        win_rate=0.6, mean_alpha=0.02,
    )
    result = experiments.render_experiments([rec])
    assert "abc1234" in result and "prompt-v2" in result and "AAPL" in result


def test_current_git_commit_returns_none_on_failure(monkeypatch):
    import subprocess

    def _raise(*a, **k):
        raise FileNotFoundError("no git")

    monkeypatch.setattr(subprocess, "run", _raise)
    assert experiments.current_git_commit() is None


def test_current_git_commit_prefers_the_env_var_over_the_subprocess(monkeypatch):
    """Same fix as its twin in app.memory.store (2026-08-13): inside the
    deployed container there is no .git and no git binary, so a containerized
    backtest tagged every run with a NULL commit."""
    def should_not_run(cmd, **kwargs):
        raise AssertionError("the baked-in commit must be used without shelling out to git")

    monkeypatch.setattr(experiments.settings, "git_commit", "cafe123")
    monkeypatch.setattr(experiments.subprocess, "run", should_not_run)

    assert experiments.current_git_commit() == "cafe123"


def test_current_git_commit_falls_back_to_git_when_env_var_is_unset(monkeypatch):
    class FakeCompletedProcess:
        stdout = "localdev\n"

    monkeypatch.setattr(experiments.settings, "git_commit", None)
    monkeypatch.setattr(experiments.subprocess, "run", lambda cmd, **kwargs: FakeCompletedProcess())

    assert experiments.current_git_commit() == "localdev"
