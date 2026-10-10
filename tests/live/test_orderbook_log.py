"""app.live.orderbook_log — asyncpg pool/connection faked, no real Postgres,
no real KIS feed. Same fake-pool convention as tests/memory/test_memory_store.py."""

from __future__ import annotations

from app.brokers.kis.domestic.websocket import OrderBook, RealtimeFeed
from app.live import orderbook_log


class FakeConn:
    def __init__(self, fail: bool = False):
        self.fail = fail
        self.executed: list[tuple] = []

    async def execute(self, query, *args):
        if self.fail:
            raise RuntimeError("db down")
        self.executed.append((query, args))
        return "INSERT 0 1"


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


def _reset_schema_state(monkeypatch):
    # _ready_pool's double-checked-locking bootstrap only runs the DDL once
    # per process — force it to skip straight to "ready" so tests don't
    # depend on execution order or a real pool for the CREATE TABLE call.
    monkeypatch.setattr(orderbook_log, "_schema_ready", True)


async def test_log_orderbook_snapshot_inserts_one_row_per_zone_code_present(monkeypatch):
    _reset_schema_state(monkeypatch)
    conn = FakeConn()
    monkeypatch.setattr(orderbook_log, "get_pool", lambda: _async_return(FakePool(conn)))

    feed = RealtimeFeed(
        app_key="k", app_secret="s", codes=["005930", "000660"],
        orderbook_codes=frozenset({"005930", "000660"}),
    )
    feed.orderbook["005930"] = OrderBook(
        code="005930", time="101530",
        asks=[{"price": 70100.0, "qty": 10}], bids=[{"price": 70000.0, "qty": 5}],
    )
    feed.orderbook["000660"] = OrderBook(
        code="000660", time="101531",
        asks=[{"price": 150100.0, "qty": 3}], bids=[{"price": 150000.0, "qty": 2}],
    )

    count = await orderbook_log.log_orderbook_snapshot(feed)

    assert count == 2
    assert len(conn.executed) == 2
    # Stored as ".KS"-suffixed tickers, matching minute_candles/scalping_*'s
    # convention — not RealtimeFeed's own raw 6-digit KRX codes — so this
    # table stays joinable by ticker with the rest of the research tooling.
    tickers_inserted = {args[0] for _query, args in conn.executed}
    assert tickers_inserted == {"005930.KS", "000660.KS"}


async def test_log_orderbook_snapshot_skips_codes_not_in_orderbook_zone(monkeypatch):
    _reset_schema_state(monkeypatch)
    conn = FakeConn()
    monkeypatch.setattr(orderbook_log, "get_pool", lambda: _async_return(FakePool(conn)))

    # 005930 is on the normal trade-tick path (not in orderbook_codes) but
    # still has a stale/leftover entry in feed.orderbook — must NOT be
    # logged, since it was never actually subscribed to H0STASP0.
    feed = RealtimeFeed(
        app_key="k", app_secret="s", codes=["005930", "000660"],
        orderbook_codes=frozenset({"000660"}),
    )
    feed.orderbook["005930"] = OrderBook(code="005930", time="101530", asks=[], bids=[])
    feed.orderbook["000660"] = OrderBook(
        code="000660", time="101531",
        asks=[{"price": 150100.0, "qty": 3}], bids=[{"price": 150000.0, "qty": 2}],
    )

    count = await orderbook_log.log_orderbook_snapshot(feed)

    assert count == 1
    assert len(conn.executed) == 1
    assert conn.executed[0][1][0] == "000660.KS"


async def test_log_orderbook_snapshot_skips_zone_code_with_no_orderbook_entry_yet(monkeypatch):
    _reset_schema_state(monkeypatch)
    conn = FakeConn()
    monkeypatch.setattr(orderbook_log, "get_pool", lambda: _async_return(FakePool(conn)))

    feed = RealtimeFeed(
        app_key="k", app_secret="s", codes=["005930"],
        orderbook_codes=frozenset({"005930"}),
    )
    # No entry in feed.orderbook yet (just subscribed, no frame arrived).

    count = await orderbook_log.log_orderbook_snapshot(feed)

    assert count == 0
    assert conn.executed == []


async def test_log_orderbook_snapshot_degrades_to_zero_on_db_failure(monkeypatch):
    _reset_schema_state(monkeypatch)
    conn = FakeConn(fail=True)
    monkeypatch.setattr(orderbook_log, "get_pool", lambda: _async_return(FakePool(conn)))

    feed = RealtimeFeed(
        app_key="k", app_secret="s", codes=["005930"],
        orderbook_codes=frozenset({"005930"}),
    )
    feed.orderbook["005930"] = OrderBook(
        code="005930", time="101530",
        asks=[{"price": 70100.0, "qty": 10}], bids=[{"price": 70000.0, "qty": 5}],
    )

    count = await orderbook_log.log_orderbook_snapshot(feed)  # must not raise

    assert count == 0
