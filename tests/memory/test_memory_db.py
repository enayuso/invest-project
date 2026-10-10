"""app.memory.db.get_pool — asyncpg.connect/create_pool faked, no real DB.
Covers the outer asyncio.wait_for backstop added after this project's shared
dev Postgres accepted TCP but never completed the wire protocol handshake,
which asyncpg's own per-call timeout/command_timeout didn't reliably bound
in practice (see get_pool's own docstring)."""

from __future__ import annotations

import asyncio
from datetime import date, timedelta

import pytest

from app.config import settings
from app.memory import db


@pytest.fixture(autouse=True)
def _reset_pool_state(monkeypatch):
    monkeypatch.setattr(db, "_pool", None)
    monkeypatch.setattr(db, "_pool_lock", asyncio.Lock())
    # Without this, one test's real failure (e.g. a signature mismatch
    # between a fake and the real asyncpg call) sets this module-global
    # timestamp, and every OTHER test in the same run then fails fast on
    # the generic "database unavailable (recent failure)" RuntimeError
    # instead of ever reaching its own actual behavior — masking the real
    # cause behind what looks like a DB outage. Confirmed live: this
    # exact scenario happened (2026-07-29) when _bootstrap_pool's real
    # asyncpg.connect/create_pool calls picked up a new `server_settings`
    # kwarg that the fakes below hadn't been updated for yet.
    monkeypatch.setattr(db, "_last_failure_at", None)
    monkeypatch.setattr(settings, "database_connect_timeout_seconds", 0.05)


class _FakeConn:
    def __init__(self):
        self.executed = []

    async def execute(self, query, *args):
        self.executed.append(query)

    async def close(self):
        pass


# ---------------------------------------------------------------------------
# _SCHEMA — vector + covering indexes (P2)
# ---------------------------------------------------------------------------


def test_schema_creates_hnsw_index_on_situation_embedding():
    # Before this, every similarity search in app.memory.store.
    # get_similar_context did a full sequential scan + sort over
    # memory_entries — cheap to add now while the table is small, expensive
    # to retrofit later under load. HNSW (not ivfflat) because this bootstrap
    # script also runs against a fresh/empty database and ivfflat wants to be
    # built after the table is populated.
    assert "hnsw" in db._SCHEMA.lower()
    assert "memory_entries_situation_embedding_hnsw_idx" in db._SCHEMA
    assert "vector_cosine_ops" in db._SCHEMA


def test_schema_creates_ticker_status_covering_index():
    # Backs get_similar_context's same_rows query (WHERE ticker = $1 AND
    # status = 'resolved' AND ...).
    assert "memory_entries_ticker_status_idx" in db._SCHEMA


# ---------------------------------------------------------------------------
# 2026-09-01 PIT fix — resolved_on column + exact backfill (no real DB here,
# same "assert on the _SCHEMA string" style as the index tests above).
# ---------------------------------------------------------------------------


def test_schema_adds_resolved_on_column():
    assert "ADD COLUMN IF NOT EXISTS resolved_on DATE" in db._SCHEMA


def test_schema_backfills_resolved_on_exactly_from_holding_days():
    # trade_date + holding_days reconstructs resolved_on EXACTLY (both
    # writers define holding_days as resolved_on - trade_date) — not an
    # approximation like trade_date + memory_min_holding_days would be.
    assert "SET resolved_on = trade_date + holding_days" in db._SCHEMA
    # Guarded so it's a no-op after the first boot and self-heals rows
    # written by an older image mid-deploy.
    assert "WHERE status = 'resolved' AND resolved_on IS NULL AND holding_days IS NOT NULL" in db._SCHEMA


@pytest.mark.parametrize(
    ("trade_date_", "holding_days", "expected"),
    [
        (date(2026, 7, 1), 5, date(2026, 7, 6)),
        (date(2026, 6, 1), 23, date(2026, 6, 24)),  # live-style variable window
        (date(2025, 12, 29), 7, date(2026, 1, 5)),  # month/year boundary crossing
    ],
)
def test_resolved_on_backfill_identity_holds_across_windows(trade_date_, holding_days, expected):
    """Pins the arithmetic identity the backfill SQL depends on —
    trade_date + holding_days == resolved_on — for the variable live window
    and across a month/year boundary, independent of any DB."""
    assert trade_date_ + timedelta(days=holding_days) == expected


async def test_get_pool_returns_pool_on_success(monkeypatch):
    fake_conn = _FakeConn()
    fake_pool = object()

    async def fake_connect(dsn, **kwargs):
        return fake_conn

    async def fake_create_pool(dsn, init=None, **kwargs):
        return fake_pool

    monkeypatch.setattr(db.asyncpg, "connect", fake_connect)
    monkeypatch.setattr(db.asyncpg, "create_pool", fake_create_pool)

    pool = await db.get_pool()
    assert pool is fake_pool
    assert fake_conn.executed  # schema DDL ran on the bootstrap connection


async def test_get_pool_caches_across_calls(monkeypatch):
    calls = {"connect": 0}
    fake_pool = object()

    async def fake_connect(dsn, **kwargs):
        calls["connect"] += 1
        return _FakeConn()

    async def fake_create_pool(dsn, init=None, **kwargs):
        return fake_pool

    monkeypatch.setattr(db.asyncpg, "connect", fake_connect)
    monkeypatch.setattr(db.asyncpg, "create_pool", fake_create_pool)

    first = await db.get_pool()
    second = await db.get_pool()
    assert first is second is fake_pool
    assert calls["connect"] == 1  # bootstrap only ran once


async def test_get_pool_raises_fast_when_connect_hangs(monkeypatch):
    # Regression: a DB that accepts the TCP connection but never completes
    # the handshake used to hang past asyncpg's own timeout in practice —
    # the outer wait_for must bound this regardless. The outer budget now
    # covers three phases (see get_pool's own comment) — pin
    # database_command_timeout_seconds down too so this test's sleep(10)
    # still exceeds the total budget instead of the default 125s command
    # timeout swamping it and making this test slow.
    monkeypatch.setattr(settings, "database_command_timeout_seconds", 0.05)

    async def hanging_connect(dsn, **kwargs):
        await asyncio.sleep(10)

    monkeypatch.setattr(db.asyncpg, "connect", hanging_connect)

    with pytest.raises(asyncio.TimeoutError):
        await db.get_pool()


async def test_get_pool_raises_fast_when_schema_execute_hangs(monkeypatch):
    # Connect succeeds quickly, but the actual query never returns —
    # a distinct hang point from the connect step itself. Same
    # command-timeout pin as the test above, for the same reason.
    monkeypatch.setattr(settings, "database_command_timeout_seconds", 0.05)

    class _HangingConn:
        async def execute(self, query, *args):
            await asyncio.sleep(10)

        async def close(self):
            pass

    async def fake_connect(dsn, **kwargs):
        return _HangingConn()

    monkeypatch.setattr(db.asyncpg, "connect", fake_connect)

    with pytest.raises(asyncio.TimeoutError):
        await db.get_pool()


async def test_get_pool_outer_timeout_covers_three_bootstrap_phases(monkeypatch):
    # Fix 4: the old budget (connect_timeout + 1) only covered roughly ONE
    # of _bootstrap_pool's three sequential phases (connect, schema execute,
    # pool connect) — verify the formula directly rather than only via the
    # hang-detection tests above.
    monkeypatch.setattr(settings, "database_connect_timeout_seconds", 3.0)
    monkeypatch.setattr(settings, "database_command_timeout_seconds", 20.0)
    captured = {}

    async def fake_wait_for(coro, timeout):
        captured["timeout"] = timeout
        coro.close()
        return object()

    monkeypatch.setattr(db.asyncio, "wait_for", fake_wait_for)

    await db.get_pool()
    assert captured["timeout"] == pytest.approx(3.0 * 2 + 20.0 + 1)


# command_timeout used to reuse database_connect_timeout_seconds (5s) for
# every query on every pooled connection — including the RESET asyncpg runs
# on every release() — which cut off perfectly healthy queries anytime they
# took a hair over 5s, even though Postgres's own statement_timeout (120s)
# was still happily willing to run them. Confirmed live 2026-08-06: this was
# the #1 cause of that day's kill_switch_status/daily_loss_status warning
# spam. Now a separate setting (database_command_timeout_seconds).
async def test_bootstrap_connect_uses_command_timeout_not_connect_timeout(monkeypatch):
    monkeypatch.setattr(settings, "database_command_timeout_seconds", 999.0)
    captured = {}

    async def fake_connect(dsn, **kwargs):
        captured.update(kwargs)
        return _FakeConn()

    async def fake_create_pool(dsn, init=None, **kwargs):
        return object()

    monkeypatch.setattr(db.asyncpg, "connect", fake_connect)
    monkeypatch.setattr(db.asyncpg, "create_pool", fake_create_pool)

    await db.get_pool()
    assert captured["command_timeout"] == 999.0
    assert captured["timeout"] == settings.database_connect_timeout_seconds  # unchanged, still the connect-phase bound


async def test_pool_creation_uses_command_timeout_not_connect_timeout(monkeypatch):
    monkeypatch.setattr(settings, "database_command_timeout_seconds", 999.0)
    captured = {}

    async def fake_connect(dsn, **kwargs):
        return _FakeConn()

    async def fake_create_pool(dsn, init=None, **kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(db.asyncpg, "connect", fake_connect)
    monkeypatch.setattr(db.asyncpg, "create_pool", fake_create_pool)

    await db.get_pool()
    assert captured["command_timeout"] == 999.0
    assert captured["timeout"] == settings.database_connect_timeout_seconds


async def test_get_pool_concurrent_callers_only_bootstrap_once(monkeypatch):
    calls = {"connect": 0}
    fake_pool = object()

    async def fake_connect(dsn, **kwargs):
        calls["connect"] += 1
        await asyncio.sleep(0)  # yield, widen the race window
        return _FakeConn()

    async def fake_create_pool(dsn, init=None, **kwargs):
        return fake_pool

    monkeypatch.setattr(db.asyncpg, "connect", fake_connect)
    monkeypatch.setattr(db.asyncpg, "create_pool", fake_create_pool)

    results = await asyncio.gather(*(db.get_pool() for _ in range(5)))
    assert all(p is fake_pool for p in results)
    assert calls["connect"] == 1
