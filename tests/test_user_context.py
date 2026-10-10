"""app.user_context.register_user/record_session_start — upsert/insert
against fake DB objects, no real Postgres (same convention as
tests/memory/test_memory_db.py)."""

from __future__ import annotations

import asyncio

import pytest

from app import user_context


@pytest.fixture(autouse=True)
def _reset_schema_state(monkeypatch):
    # _schema_ready is a module global (see app.execution.ledger's identical
    # pattern) — without resetting it, whichever test runs first "uses up"
    # the one-time _SCHEMA execution and every later test's conn.calls no
    # longer includes it, making assertions order-dependent.
    monkeypatch.setattr(user_context, "_schema_ready", False)
    monkeypatch.setattr(user_context, "_schema_lock", asyncio.Lock())


class _FakeConn:
    def __init__(self):
        self.calls: list[tuple[str, tuple]] = []

    async def execute(self, query, *args):
        self.calls.append((query, args))

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _FakePool:
    def __init__(self, conn):
        self._conn = conn

    def acquire(self):
        return self._conn


def test_get_current_user_id_defaults_to_default():
    # No real auth yet (see the module's own docstring) — an instance with
    # no USER_ID configured behaves exactly as before this setting existed.
    assert user_context.get_current_user_id() == user_context.DEFAULT_USER_ID


def test_get_current_user_id_reflects_configured_settings_user_id(monkeypatch):
    # One process per person, sharing one DB — each instance's own
    # settings.user_id (from its own .env.local) is what distinguishes them.
    monkeypatch.setattr(user_context.settings, "user_id", "chulsoo")
    assert user_context.get_current_user_id() == "chulsoo"


async def test_register_user_upserts_into_users_table(monkeypatch):
    conn = _FakeConn()

    async def fake_get_pool():
        return _FakePool(conn)

    monkeypatch.setattr(user_context, "get_pool", fake_get_pool)

    await user_context.register_user("chulsoo", "Chulsoo Kim")

    # calls[0] is the one-time schema bootstrap (_ready_pool), the actual
    # upsert is the last call.
    query, args = conn.calls[-1]
    assert "INSERT INTO users" in query
    assert "ON CONFLICT (user_id) DO UPDATE" in query
    assert args == ("chulsoo", "Chulsoo Kim")


async def test_record_session_start_inserts_identity_claim(monkeypatch):
    conn = _FakeConn()

    async def fake_get_pool():
        return _FakePool(conn)

    monkeypatch.setattr(user_context, "get_pool", fake_get_pool)
    monkeypatch.setattr(user_context.settings, "user_id", "chulsoo")
    monkeypatch.setattr(user_context.socket, "gethostname", lambda: "server4")
    monkeypatch.setattr(user_context.os, "getpid", lambda: 4242)

    await user_context.record_session_start()

    query, args = conn.calls[-1]
    assert "INSERT INTO identity_claims" in query
    assert args == ("chulsoo", "server4", 4242)


async def test_record_session_start_degrades_quietly_on_db_failure(monkeypatch):
    # Forensic-only — must never block process startup (see the function's
    # own docstring), unlike a real safety gate.
    async def broken_get_pool():
        raise RuntimeError("database unavailable")

    monkeypatch.setattr(user_context, "get_pool", broken_get_pool)

    await user_context.record_session_start()  # must not raise
