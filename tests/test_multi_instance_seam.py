"""agent-architecture-roadmap stage 4 — the user_id columns/stamp added to
memory_entries/trades/alerts, and the get_current_user_id() seam they share.
asyncpg pool/connection faked, no real DB (store_pending's own coverage
lives in test_memory_store.py; this focuses on what stage 4 itself added)."""

from __future__ import annotations

from datetime import date, datetime

import app.memory.db as memory_db
from app import user_context
from app.execution import ledger
from app.live import alerts


class FakeConn:
    def __init__(self):
        self.queries: list[tuple] = []

    async def execute(self, query, *args):
        self.queries.append((query, args))

    async def fetchrow(self, query, *args):
        self.queries.append((query, args))
        return None  # ledger.get_position's SELECT: no prior trades -> fresh position


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


def _insert_columns(query: str, args: tuple) -> dict:
    """Maps an `INSERT INTO table (col1, col2, ...) VALUES ($1, $2, ...)`
    query's positional args back to column names, so tests don't hardcode a
    fragile args[-N] index that silently drifts every time a new column
    (e.g. account_mode) is appended to the INSERT — confirmed broken in
    practice: this file's own args[-2]/args[-1] assumptions stopped
    matching once account_mode was added as trades' new last column."""
    columns_part = query.split("(", 1)[1].split(")", 1)[0]
    columns = [c.strip() for c in columns_part.split(",")]
    return dict(zip(columns, args))


def test_get_current_user_id_returns_the_fixed_default():
    assert user_context.get_current_user_id() == user_context.DEFAULT_USER_ID == "default"


async def test_record_trade_stamps_current_user_id(monkeypatch):
    conn = FakeConn()
    monkeypatch.setattr(ledger, "_ready_pool", lambda: _async_return(FakePool(conn)))

    async def fake_get_position(ticker):
        return ledger.Position(cash=100_000.0, shares=0)

    monkeypatch.setattr(ledger, "get_position", fake_get_position)

    await ledger.record_trade("AAPL", date(2026, 7, 16), "buy", 1, 150.0, reason="test")

    insert_query, args = next(q for q in conn.queries if "INSERT INTO trades" in q[0])
    assert "user_id" in insert_query
    assert _insert_columns(insert_query, args)["user_id"] == "default"


async def test_record_trade_uses_current_user_id_override(monkeypatch):
    conn = FakeConn()
    monkeypatch.setattr(ledger, "_ready_pool", lambda: _async_return(FakePool(conn)))

    async def fake_get_position(ticker):
        return ledger.Position(cash=100_000.0, shares=0)

    monkeypatch.setattr(ledger, "get_position", fake_get_position)
    monkeypatch.setattr(ledger, "get_current_user_id", lambda: "someone-else")

    await ledger.record_trade("AAPL", date(2026, 7, 16), "sell", 1, 150.0, reason="test")

    insert_query, args = next(q for q in conn.queries if "INSERT INTO trades" in q[0])
    assert _insert_columns(insert_query, args)["user_id"] == "someone-else"


async def test_record_alert_stamps_current_user_id(monkeypatch):
    conn = FakeConn()
    # record_alert imports get_pool locally from app.memory.db (avoids a
    # hard module-load dependency), so the real target module is patched.
    monkeypatch.setattr(memory_db, "get_pool", lambda: _async_return(FakePool(conn)))
    monkeypatch.setattr(alerts, "_schema_ready", True)  # skip DDL, focus on the insert itself

    alert = alerts.Alert(ticker="AAPL", kind="price_move", message="test alert", triggered_at=datetime(2026, 7, 16, 9, 0))
    await alerts.record_alert(alert)

    insert_query, args = conn.queries[0]
    assert "user_id" in insert_query
    assert _insert_columns(insert_query, args)["user_id"] == "default"
