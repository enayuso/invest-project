"""app.live.store.forget_ticker — clearing a ticker's tick buffer AND its
persisted snapshot row when it leaves the trade allowlist.

Real incident this guards against (2026-08-13): before `forget_ticker`
existed, nothing ever cleared `_ticks[ticker]` or its `live_ticks_snapshot`
row on removal. Re-adding the same ticker later would resurrect the OLD
buffer untouched (`seed_from_intraday_series`/`load_snapshot` both no-op
once `_ticks.get(ticker)` is truthy) or, after a restart, re-hydrate the
same stale Postgres row — either way presenting old ticks as current with
no visible sign they predate the re-add.
"""

from __future__ import annotations

from collections import deque
from datetime import datetime, timezone

import pytest

from app.live import store


class _AcquireCM:
    def __init__(self, conn):
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, *exc):
        return False


class FakeStoreConn:
    def __init__(self):
        self.schema_execute_count = 0
        self.deleted = []

    async def execute(self, query, *args):
        if "CREATE TABLE" in query:
            self.schema_execute_count += 1
            return
        if query.strip().startswith("DELETE"):
            self.deleted.append(args[0])
            return
        raise AssertionError(f"unexpected query in forget_ticker test: {query}")


class FakeStorePool:
    def __init__(self, conn):
        self._conn = conn

    def acquire(self):
        return _AcquireCM(self._conn)


@pytest.fixture
def fake_conn(monkeypatch):
    conn = FakeStoreConn()

    async def fake_get_pool():
        return FakeStorePool(conn)

    monkeypatch.setattr("app.memory.db.get_pool", fake_get_pool, raising=False)
    return conn


def setup_function(_fn):
    store._ticks.clear()
    store._schema_ready = False


def _tick(price=100.0):
    return store.Tick(price=price, volume=1, timestamp=datetime.now(timezone.utc))


async def test_forget_ticker_clears_in_memory_buffer(fake_conn):
    store._ticks["005930.KS"] = deque([_tick()], maxlen=500)
    await store.forget_ticker("005930.KS")
    assert store.get_ticks("005930.KS") == []
    assert store.latest("005930.KS") is None


async def test_forget_ticker_deletes_the_persisted_snapshot_row(fake_conn):
    store._ticks["005930.KS"] = deque([_tick()], maxlen=500)
    await store.forget_ticker("005930.KS")
    assert fake_conn.deleted == ["005930.KS"]


async def test_forget_ticker_leaves_other_tickers_alone(fake_conn):
    store._ticks["005930.KS"] = deque([_tick()], maxlen=500)
    store._ticks["000660.KS"] = deque([_tick(price=200.0)], maxlen=500)
    await store.forget_ticker("005930.KS")
    assert store.get_ticks("000660.KS") != []


async def test_forget_ticker_degrades_quietly_when_db_unreachable(monkeypatch):
    store._ticks["005930.KS"] = deque([_tick()], maxlen=500)

    async def broken_get_pool():
        raise RuntimeError("db unavailable (in backoff window)")

    monkeypatch.setattr("app.memory.db.get_pool", broken_get_pool, raising=False)
    await store.forget_ticker("005930.KS")  # must not raise
    assert store.get_ticks("005930.KS") == []  # memory side still cleared


async def test_re_add_after_forget_does_not_resurrect_stale_ticks(fake_conn, monkeypatch):
    # The end-to-end concern from the bug report: remove, then re-add (which
    # goes through seed_from_intraday_series) must not see the OLD buffer as
    # "already populated" and skip reseeding.
    store._ticks["005930.KS"] = deque([_tick(price=999.0)], maxlen=500)
    await store.forget_ticker("005930.KS")

    calls = []

    class _Quote:
        def __init__(self, price):
            self.price = price
            self.volume = 10
            self.timestamp = datetime.now(timezone.utc)

    async def fake_get_intraday_series(ticker):
        calls.append(ticker)
        return [_Quote(70000.0)]

    monkeypatch.setattr("app.dataflows.yfinance.get_intraday_series", fake_get_intraday_series, raising=False)

    await store.seed_from_intraday_series("005930.KS")

    assert calls == ["005930.KS"]  # reseed actually ran — not skipped as "already populated"
    assert store.latest("005930.KS").price == 70000.0  # fresh data, not the old 999.0
