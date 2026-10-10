"""app.experiments.replay.safety — block stubs, invariants, market window (spec C, G).
No DB, no LLM, no network: the invariant queries are exercised with dict snapshots."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from app.experiments.replay import safety
from replay_fixtures import FakeReadonlyConn, install_db_block


@pytest.fixture(autouse=True)
def _block_db(monkeypatch):
    install_db_block(monkeypatch)


def kst(y, m, d, hh, mm):
    from zoneinfo import ZoneInfo

    return datetime(y, m, d, hh, mm, tzinfo=ZoneInfo("Asia/Seoul"))


def test_violation_is_baseexception():
    assert issubclass(safety.ReplaySafetyViolation, BaseException)
    assert not issubclass(safety.ReplaySafetyViolation, Exception)


def test_blocks_raise_and_log(monkeypatch, tmp_path):
    from app.brokers.base.http_client import BaseHttpClient
    from app.live import alerts
    from app.memory import db, store
    from app.orchestrator import pipeline

    trips_file = tmp_path / "safety_trips.jsonl"
    safety.reset_trips()
    safety.set_trips_path(trips_file)
    try:
        report = safety.install_blocks(setattr_fn=monkeypatch.setattr)
        assert report.missing == []
        # `from app.memory.store import store_pending` copies were replaced too
        assert "app.orchestrator.pipeline.store_pending" in report.aliases

        with pytest.raises(safety.ReplaySafetyViolation):
            store.store_pending("005930.KS", None, "s", "Hold", "d")
        with pytest.raises(safety.ReplaySafetyViolation):
            pipeline.store_pending("005930.KS", None, "s", "Hold", "d")  # by-name copy
        with pytest.raises(safety.ReplaySafetyViolation):
            db.get_pool()
        with pytest.raises(safety.ReplaySafetyViolation):
            alerts.record_alert(object())
        with pytest.raises(safety.ReplaySafetyViolation):
            BaseHttpClient()
        with pytest.raises(safety.ReplaySafetyViolation):
            pipeline.run_pipeline("005930.KS", None)
        with pytest.raises(safety.ReplaySafetyViolation):
            import asyncpg

            asyncpg.connect("postgresql://x")

        assert len(safety.trips()) == 7
        lines = trips_file.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 7 and json.loads(lines[0])["target"] == "app.memory.store.store_pending"
        # our own read-only connection keeps the ORIGINAL connect, never a stub
        assert not safety.is_block_stub(safety._ORIG_CONNECT)
        assert safety.is_block_stub(store.store_pending)
    finally:
        safety.set_trips_path(None)
        safety.reset_trips()


def test_install_is_idempotent_and_reports_missing(monkeypatch):
    safety.install_blocks(setattr_fn=monkeypatch.setattr)
    second = safety.install_blocks(setattr_fn=monkeypatch.setattr)
    assert second.installed == []  # everything already a stub
    with pytest.raises(RuntimeError, match="targets missing"):
        safety.install_blocks(
            setattr_fn=monkeypatch.setattr, targets=(("app.memory.store", "no_such_function"),)
        )
    report = safety.install_blocks(
        setattr_fn=monkeypatch.setattr, targets=(("app.memory.store", "no_such_function"),), strict=False
    )
    assert report.missing and report.installed == []


def test_stub_keeps_original_source_visible(monkeypatch):
    """The provenance hash of `_run_pipeline_impl` is taken via inspect.getsource — it must
    still see the original source after the block is installed."""
    import inspect

    from app.orchestrator import pipeline

    before = inspect.getsource(pipeline._run_pipeline_impl)
    safety.install_blocks(setattr_fn=monkeypatch.setattr)
    assert safety.is_block_stub(pipeline._run_pipeline_impl)
    assert inspect.getsource(pipeline._run_pipeline_impl) == before


def test_offhours():
    assert safety.market_window_open(kst(2026, 9, 22, 10, 0)) is True  # Tuesday
    assert safety.market_window_open(kst(2026, 9, 22, 8, 29)) is False
    assert safety.market_window_open(kst(2026, 9, 22, 8, 30)) is True
    assert safety.market_window_open(kst(2026, 9, 22, 15, 39)) is True
    assert safety.market_window_open(kst(2026, 9, 22, 15, 40)) is False
    assert safety.market_window_open(kst(2026, 9, 26, 10, 0)) is False  # Saturday
    with pytest.raises(safety.MarketWindowError):
        safety.assert_offhours(kst(2026, 9, 22, 10, 0))
    safety.assert_offhours(kst(2026, 9, 22, 15, 40))
    safety.assert_offhours(kst(2026, 9, 26, 10, 0))
    safety.assert_offhours(kst(2026, 9, 22, 10, 0), allow_market_hours=True)
    with pytest.raises(ValueError):
        safety.assert_offhours(datetime(2026, 9, 22, 10, 0))
    # UTC input is converted to KST: 2026-09-22T01:00Z == 10:00 KST
    assert safety.market_window_open(datetime(2026, 9, 22, 1, 0, tzinfo=timezone.utc)) is True


def _snap(**over):
    base = {
        "tables": {
            "memory_entries": {"max_id": 100, "count": 90, "max_at": "2026-09-29T01:00:00+00:00"},
            "trades": {"max_id": 50, "count": 50, "max_at": "2026-09-29T01:00:00+00:00"},
            "pending_orders": {"max_id": 7, "count": 7, "max_at": "2026-09-29T01:00:00+00:00"},
            "alerts": {"max_id": 900, "count": 900, "max_at": "2026-09-29T01:00:00+00:00"},
        },
        "digests": {"cases": "c", "risk_rules": "r", "runtime_settings": "s"},
        "new_rows": {},
    }
    for key, value in over.items():
        base[key] = value
    return base


def _copy(snap):
    return json.loads(json.dumps(snap))


def test_compare_invariants():
    tickers = {"005930.KS"}
    before = _snap()
    assert safety.compare_invariants(before, _copy(before), tickers).status == "PASS"

    after = _copy(before)
    after["tables"]["memory_entries"] = {"max_id": 101, "count": 91, "max_at": "2026-09-30T01:00:00+00:00"}
    after["new_rows"] = {"memory_entries": [{"id": 101, "ticker": "000660.KS", "kind": None, "at": "x"}]}
    res = safety.compare_invariants(before, after, tickers)
    # unknown provenance and no run windows: FAIL_CANDIDATE (human must confirm it is a live row)
    assert res.status == "FAIL_CANDIDATE" and res.candidates[0]["id"] == 101 and res.candidates[0]["ticker"] == "000660.KS"
    released = safety.compare_invariants(before, after, tickers, confirmed={"memory_entries": [101]})
    assert released.status == "PASS" and released.released[0]["id"] == 101

    after = _copy(before)
    after["tables"]["alerts"] = {"max_id": 901, "count": 901, "max_at": "2026-09-30T01:00:00+00:00"}
    after["new_rows"] = {"alerts": [{"id": 901, "ticker": "005930.KS", "kind": "price_move", "at": "x"}]}
    res = safety.compare_invariants(before, after, tickers)
    assert res.status == "WARN" and not res.fails and res.warns

    after["new_rows"] = {"alerts": [{"id": 901, "ticker": "005930.KS", "kind": "trade_allowlist_blocked", "at": "x"}]}
    assert safety.compare_invariants(before, after, tickers).status == "FAIL"
    after["new_rows"] = {"alerts": [{"id": 901, "ticker": "999999.KS", "kind": "trade_allowlist_blocked", "at": "x"}]}
    assert safety.compare_invariants(before, after, tickers).status == "WARN"  # not a case ticker

    after = _copy(before)
    after["digests"]["cases"] = "changed"
    assert safety.compare_invariants(before, after, tickers).status == "FAIL"

    after = _copy(before)
    after["tables"]["trades"]["count"] = 49
    res = safety.compare_invariants(before, after, tickers)
    assert res.status == "FAIL" and "decreased" in res.fails[0]

    after = _copy(before)
    after["digests"]["risk_rules"] = "other"
    assert safety.compare_invariants(before, after, tickers).status == "FAIL"
    after = _copy(before)
    after["digests"]["runtime_settings"] = "other"
    assert safety.compare_invariants(before, after, tickers).as_dict()["status"] == "FAIL"


async def test_snapshot_uses_readonly_transactions():
    conn = FakeReadonlyConn()
    snap = await safety.snapshot_invariants(conn, [1, 2, 3])
    assert set(snap["tables"]) == {"memory_entries", "trades", "pending_orders", "alerts"}
    assert snap["digests"] == {"risk_rules": "digest", "runtime_settings": "digest", "cases": "digest"}
    assert conn.tx_kwargs and all(kw == {"isolation": "repeatable_read", "readonly": True} for kw in conn.tx_kwargs)
    assert not any(s.lstrip().upper().startswith(("INSERT", "UPDATE", "DELETE", "CREATE", "ALTER")) for s in conn.sql)
    again = await safety.snapshot_invariants(conn, [1, 2, 3], since=snap)
    assert again["new_rows"]["alerts"] == []


def test_keep_awake_is_best_effort():
    assert safety.keep_awake(False) in (True, False)
    assert safety.keep_awake(False) in (True, False)


def test_verify_process_state_reports_limiter_mismatch():
    problems = safety.verify_process_state(k=999)
    assert any("limiter" in p for p in problems)


async def test_readonly_connection_is_bounded_and_read_only(monkeypatch):
    seen = {}

    async def fake_connect(dsn, **kwargs):
        seen["dsn"], seen["kwargs"] = dsn, kwargs
        return "conn"

    monkeypatch.setattr(safety, "_ORIG_CONNECT", fake_connect)
    assert await safety.open_readonly_connection("postgresql://x/y") == "conn"
    kw = seen["kwargs"]
    assert kw["timeout"] == 30.0 and kw["command_timeout"] == 120.0
    assert kw["server_settings"]["default_transaction_read_only"] == "on"
    assert kw["server_settings"]["statement_timeout"] == "120000"
    assert kw["server_settings"]["application_name"] == "replay_downstream_ro"


def test_trip_count_uses_the_on_disk_log(tmp_path):
    safety.reset_trips()
    path = tmp_path / "t" / "safety_trips.jsonl"
    path.parent.mkdir()
    path.write_text('{"a":1}\n{"a":2}\n', encoding="utf-8")
    safety.set_trips_path(path)
    try:
        assert safety.trip_count() == 2  # survives a resume (memory list is empty)
    finally:
        safety.set_trips_path(None)
        safety.reset_trips()
