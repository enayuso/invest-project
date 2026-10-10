"""app.web.trade_allowlist_store — load/save round-trip, no real DB. Mirrors
tests/execution/test_runtime_settings.py's fake Postgres pool/conn pattern."""

from __future__ import annotations

import asyncio
import json

import pytest

from app.config import account_mode_scope, reset_scoped_lists, settings
from app.user_context import user_scope
from app.web import trade_allowlist_store


@pytest.fixture(autouse=True)
def _fresh_schema_guard(monkeypatch):
    # _schema_lock is bound to whatever event loop first acquires it, and
    # each test function gets its own loop (same reasoning as
    # test_runtime_settings.py's identical fixture).
    monkeypatch.setattr(trade_allowlist_store, "_schema_ready", False)
    monkeypatch.setattr(trade_allowlist_store, "_schema_lock", asyncio.Lock())
    # Deterministic default regardless of what another test file left behind
    # on the shared settings singleton — every persistence key in this
    # module is namespaced by this value (see trade_allowlist_store._scoped).
    monkeypatch.setattr(settings, "kis_account_mode", "paper")
    # The four ticker lists live in a module-level per-scope dict now, so a
    # test that writes one leaks into the next unless it is cleared.
    reset_scoped_lists()


class _AcquireCM:
    def __init__(self, conn):
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, *exc):
        return False


class FakeTradeAllowlistStoreConn:
    def __init__(self):
        self.schema_execute_count = 0
        self.rows = []
        self.saved = []
        self.deleted = []

    async def execute(self, query, *args):
        if "CREATE TABLE" in query:
            self.schema_execute_count += 1
            return
        if query.strip().startswith("DELETE"):
            self.deleted.append(args[0])
            return
        self.saved.append(args)  # INSERT ... ON CONFLICT

    async def fetch(self, query, *args):
        return self.rows


class FakeTradeAllowlistStorePool:
    def __init__(self, conn):
        self._conn = conn

    def acquire(self):
        return _AcquireCM(self._conn)


@pytest.fixture
def fake_conn(monkeypatch):
    conn = FakeTradeAllowlistStoreConn()

    async def fake_get_pool():
        return FakeTradeAllowlistStorePool(conn)

    monkeypatch.setattr("app.memory.db.get_pool", fake_get_pool, raising=False)
    return conn


async def test_ensure_schema_runs_ddl_only_once_across_repeated_loads(fake_conn):
    await trade_allowlist_store.load()
    await trade_allowlist_store.load()
    assert fake_conn.schema_execute_count == 1


async def test_load_restores_persisted_trade_allowlist_and_watchlist(fake_conn, monkeypatch):
    monkeypatch.setattr(settings, "trade_allowlist_domestic", ["AAPL"])  # will be overwritten
    monkeypatch.setattr(settings, "trade_allowlist_overseas", ["AAPL"])
    monkeypatch.setattr(settings, "watchlist", ["AAPL"])
    fake_conn.rows = [
        {"key": "default:paper:trade_allowlist_domestic", "tickers": json.dumps(["005930.KS", "373220.KS"])},
        {"key": "default:paper:trade_allowlist_overseas", "tickers": json.dumps(["AAPL"])},
        {"key": "default:paper:watchlist", "tickers": json.dumps(["005930.KS", "373220.KS", "AAPL"])},
    ]
    await trade_allowlist_store.load()
    assert settings.trade_allowlist_domestic == ["005930.KS", "373220.KS"]
    assert settings.trade_allowlist_overseas == ["AAPL"]
    assert settings.watchlist == ["005930.KS", "373220.KS", "AAPL"]


async def test_load_migrates_old_scoped_favorites_key_into_watchlist(fake_conn, monkeypatch):
    # favorites -> watchlist rename (2026-08-05): the OLD scoped key
    # ("default:paper:favorites") predates this rename — load() reissues its value
    # under the new scoped key ("default:paper:watchlist") once, then drops the old
    # row, same shape as the bare-key migration below.
    monkeypatch.setattr(settings, "watchlist", [])
    fake_conn.rows = [
        {"key": "default:paper:favorites", "tickers": json.dumps(["005930.KS", "AAPL"])},
    ]
    await trade_allowlist_store.load()
    assert settings.watchlist == ["005930.KS", "AAPL"]
    saved_keys = {args[0] for args in fake_conn.saved}
    assert saved_keys == {"default:paper:watchlist"}
    assert fake_conn.deleted == ["default:paper:favorites"]


async def test_load_skips_favorites_migration_when_watchlist_already_persisted(fake_conn, monkeypatch):
    # An old scoped favorites row lingering alongside an already-migrated
    # watchlist row (e.g. a second app instance racing the same DB) must
    # not re-migrate/overwrite.
    monkeypatch.setattr(settings, "watchlist", [])
    fake_conn.rows = [
        {"key": "default:paper:favorites", "tickers": json.dumps(["999999.KS"])},
        {"key": "default:paper:watchlist", "tickers": json.dumps(["005930.KS"])},
    ]
    await trade_allowlist_store.load()
    assert settings.watchlist == ["005930.KS"]
    assert fake_conn.deleted == []


async def test_load_restores_persisted_auto_added_marker(fake_conn, monkeypatch):
    monkeypatch.setattr(settings, "trade_allowlist_domestic", ["005930.KS"])
    monkeypatch.setattr(settings, "trade_allowlist_auto_added", [])
    fake_conn.rows = [
        {"key": "default:paper:trade_allowlist_domestic", "tickers": json.dumps(["005930.KS"])},
        {"key": "default:paper:trade_allowlist_auto_added", "tickers": json.dumps(["005930.KS"])},
    ]
    await trade_allowlist_store.load()
    assert settings.trade_allowlist_auto_added == ["005930.KS"]


async def test_load_migrates_bare_pre_scoping_keys_under_paper_mode(fake_conn, monkeypatch):
    # 2026-08-04: every row this project has ever written used the bare,
    # unscoped key names — without this migration, restarting under the
    # (unchanged) 'paper' mode would find nothing under the new scoped
    # names and silently reset trade_allowlist/watchlist/auto-added to the
    # `.env` defaults.
    monkeypatch.setattr(settings, "trade_allowlist_domestic", [])
    monkeypatch.setattr(settings, "trade_allowlist_auto_added", [])
    fake_conn.rows = [
        {"key": "trade_allowlist_domestic", "tickers": json.dumps(["005930.KS"])},
        {"key": "trade_allowlist_auto_added", "tickers": json.dumps(["005930.KS"])},
    ]
    await trade_allowlist_store.load()
    assert settings.trade_allowlist_domestic == ["005930.KS"]
    assert settings.trade_allowlist_auto_added == ["005930.KS"]
    saved_keys = {args[0] for args in fake_conn.saved}
    assert saved_keys == {"default:paper:trade_allowlist_domestic", "default:paper:trade_allowlist_auto_added"}
    assert sorted(fake_conn.deleted) == ["trade_allowlist_auto_added", "trade_allowlist_domestic"]


async def test_load_skips_bare_key_migration_when_scoped_already_persisted(fake_conn, monkeypatch):
    # A bare row lingering alongside an already-migrated scoped row (e.g. a
    # previous partial migration, or a stale leftover) must not overwrite
    # the scoped one or get deleted out from under a still-relevant read.
    monkeypatch.setattr(settings, "trade_allowlist_domestic", [])
    fake_conn.rows = [
        {"key": "trade_allowlist_domestic", "tickers": json.dumps(["999999.KS"])},
        {"key": "default:paper:trade_allowlist_domestic", "tickers": json.dumps(["005930.KS"])},
    ]
    await trade_allowlist_store.load()
    assert settings.trade_allowlist_domestic == ["005930.KS"]  # scoped row wins
    assert fake_conn.deleted == []
    assert fake_conn.saved == []


async def test_load_does_not_migrate_bare_keys_under_real_mode(fake_conn, monkeypatch):
    # The concrete leak this guard prevents: a bare row is ALWAYS paper's
    # data (this project has never run 'real'), so migrating it while
    # running under 'real' would hand paper's allowlist straight to real
    # money. A first-ever real run must start empty instead.
    monkeypatch.setattr(settings, "kis_account_mode", "real")
    monkeypatch.setattr(settings, "trade_allowlist_domestic", [])
    fake_conn.rows = [
        {"key": "trade_allowlist_domestic", "tickers": json.dumps(["005930.KS"])},
    ]
    await trade_allowlist_store.load()
    assert settings.trade_allowlist_domestic == []  # untouched .env default, not paper's list
    assert fake_conn.deleted == []
    assert fake_conn.saved == []


async def test_load_keeps_env_defaults_when_nothing_persisted(fake_conn, monkeypatch):
    monkeypatch.setattr(settings, "trade_allowlist_domestic", ["005930.KS"])
    fake_conn.rows = []
    await trade_allowlist_store.load()
    assert settings.trade_allowlist_domestic == ["005930.KS"]  # untouched


async def test_load_degrades_quietly_when_store_unreachable(monkeypatch):
    async def broken_get_pool():
        raise OSError("connection refused")

    monkeypatch.setattr("app.memory.db.get_pool", broken_get_pool, raising=False)
    monkeypatch.setattr(settings, "trade_allowlist_overseas", ["AAPL"])
    await trade_allowlist_store.load()  # must not raise
    assert settings.trade_allowlist_overseas == ["AAPL"]


async def test_save_persists_the_given_list_under_the_scoped_key(fake_conn):
    await trade_allowlist_store.save("trade_allowlist_domestic", ["005930.KS"])
    assert len(fake_conn.saved) == 1
    key, tickers_json = fake_conn.saved[0]
    assert key == "default:paper:trade_allowlist_domestic"
    assert json.loads(tickers_json) == ["005930.KS"]


async def test_save_degrades_quietly_when_store_unreachable(monkeypatch):
    async def broken_get_pool():
        raise OSError("connection refused")

    monkeypatch.setattr("app.memory.db.get_pool", broken_get_pool, raising=False)
    await trade_allowlist_store.save("watchlist", ["AAPL"])  # must not raise


# ---------------------------------------------------------------------------
# account_mode isolation (2026-08-04) — see project memory:
# user-account-mode-db-separation-design. A ticker the AI auto-added to the
# allowlist during paper testing must never surface as already-approved
# once kis_account_mode flips to 'real'.
# ---------------------------------------------------------------------------


async def test_save_scopes_to_real_when_account_mode_is_real(fake_conn, monkeypatch):
    monkeypatch.setattr(settings, "kis_account_mode", "real")
    await trade_allowlist_store.save("trade_allowlist_domestic", ["005930.KS"])
    key, _ = fake_conn.saved[0]
    assert key == "default:real:trade_allowlist_domestic"


async def test_load_does_not_see_the_other_account_modes_allowlist(fake_conn, monkeypatch):
    # A ticker only ever persisted under paper's namespace...
    monkeypatch.setattr(settings, "trade_allowlist_domestic", [])
    fake_conn.rows = [
        {"key": "default:paper:trade_allowlist_domestic", "tickers": json.dumps(["005930.KS"])},
    ]
    # ...must not restore into a real-mode process, which has no matching row.
    monkeypatch.setattr(settings, "kis_account_mode", "real")
    await trade_allowlist_store.load()
    assert settings.trade_allowlist_domestic == []  # untouched .env default, not paper's list


# ---------------------------------------------------------------------------
# Per-user isolation (multi-user migration Phase 4, plan §2.2).
#
# This is the list kis_bridge gates on to decide what may be traded at all,
# so a shared allowlist means adding a ticker to your own list silently
# authorizes it for everyone else's automated trading.
# ---------------------------------------------------------------------------


async def test_save_scopes_to_the_acting_user(fake_conn):
    with user_scope("u-alice"):
        await trade_allowlist_store.save("trade_allowlist_domestic", ["005930.KS"])
    key, _ = fake_conn.saved[0]
    assert key == "u-alice:paper:trade_allowlist_domestic"


async def test_load_does_not_see_another_users_allowlist(fake_conn, monkeypatch):
    monkeypatch.setattr(settings, "trade_allowlist_domestic", [])
    fake_conn.rows = [
        {"key": "u-alice:paper:trade_allowlist_domestic", "tickers": json.dumps(["005930.KS"])},
    ]
    with user_scope("u-bob"):
        await trade_allowlist_store.load()
    assert settings.trade_allowlist_domestic == []


async def test_two_users_persist_to_separate_rows(fake_conn):
    with user_scope("u-alice"):
        await trade_allowlist_store.save("watchlist", ["005930.KS"])
    with user_scope("u-bob"):
        await trade_allowlist_store.save("watchlist", ["AAPL"])
    assert {key for key, _ in fake_conn.saved} == {
        "u-alice:paper:watchlist", "u-bob:paper:watchlist",
    }


async def test_load_migrates_pre_user_scoping_rows_to_default(fake_conn, monkeypatch):
    """Every row written before Phase 4 belongs to the single user who was
    being served then. Without this, the first restart after the upgrade
    silently resets the allowlist to the .env defaults."""
    monkeypatch.setattr(settings, "trade_allowlist_domestic", [])
    fake_conn.rows = [
        {"key": "paper:trade_allowlist_domestic", "tickers": json.dumps(["005930.KS"])},
    ]
    await trade_allowlist_store.load()  # runs as 'default'

    assert settings.trade_allowlist_domestic == ["005930.KS"]
    assert "default:paper:trade_allowlist_domestic" in {key for key, _ in fake_conn.saved}
    assert fake_conn.deleted == ["paper:trade_allowlist_domestic"]


async def test_pre_user_scoping_rows_are_not_handed_to_a_new_user(fake_conn, monkeypatch):
    """The migration must be gated on actually BEING 'default'. Reissuing
    one-user-era data under whoever logs in first would hand a stranger the
    original owner's trading allowlist."""
    monkeypatch.setattr(settings, "trade_allowlist_domestic", [])
    fake_conn.rows = [
        {"key": "paper:trade_allowlist_domestic", "tickers": json.dumps(["005930.KS"])},
    ]
    with user_scope("u-newcomer"):
        await trade_allowlist_store.load()

    assert settings.trade_allowlist_domestic == []
    assert fake_conn.saved == []
    assert fake_conn.deleted == []


async def test_both_key_shapes_present_is_left_alone_and_warned(fake_conn, monkeypatch, caplog):
    """The stranded-row hazard. If a row already exists under the new shape,
    the legacy row must NOT be silently dropped, clobbered, or merged —
    merging is the worst option, because a ticker the user deliberately
    removed would come back and regain trading permission."""
    monkeypatch.setattr(settings, "trade_allowlist_domestic", [])
    fake_conn.rows = [
        {"key": "paper:trade_allowlist_domestic", "tickers": json.dumps(["005930.KS", "000660.KS"])},
        {"key": "default:paper:trade_allowlist_domestic", "tickers": json.dumps(["035420.KS"])},
    ]
    with caplog.at_level("WARNING"):
        await trade_allowlist_store.load()

    # Nothing written, nothing deleted.
    assert fake_conn.saved == []
    assert fake_conn.deleted == []
    # The new-shape row is what loads, unchanged.
    assert settings.trade_allowlist_domestic == ["035420.KS"]
    # And it is impossible to miss.
    assert any(
        "BOTH" in r.message and "paper:trade_allowlist_domestic" in r.getMessage()
        for r in caplog.records
    )


async def test_the_warning_repeats_rather_than_resolving_itself(fake_conn, monkeypatch, caplog):
    """A one-shot warning would scroll away and the ambiguity would persist
    unnoticed. It must fire every boot until someone fixes the data."""
    monkeypatch.setattr(settings, "trade_allowlist_domestic", [])
    fake_conn.rows = [
        {"key": "paper:trade_allowlist_domestic", "tickers": json.dumps(["005930.KS"])},
        {"key": "default:paper:trade_allowlist_domestic", "tickers": json.dumps(["035420.KS"])},
    ]
    with caplog.at_level("WARNING"):
        await trade_allowlist_store.load()
        first = len([r for r in caplog.records if "BOTH" in r.message])
        await trade_allowlist_store.load()
        second = len([r for r in caplog.records if "BOTH" in r.message])
    assert first == 1 and second == 2


async def test_a_clean_legacy_row_still_migrates(fake_conn, monkeypatch):
    """Guard against over-correcting: the ordinary upgrade path (legacy row
    present, no new-shape row) must still migrate as before."""
    monkeypatch.setattr(settings, "trade_allowlist_overseas", [])
    fake_conn.rows = [
        {"key": "paper:trade_allowlist_overseas", "tickers": json.dumps(["AAPL", "NVDA"])},
    ]
    await trade_allowlist_store.load()
    assert settings.trade_allowlist_overseas == ["AAPL", "NVDA"]
    assert fake_conn.deleted == ["paper:trade_allowlist_overseas"]


# ---------------------------------------------------------------------------
# Explicit load scope (2026-08-06).
#
# load() used to derive its scope from get_current_user_id()/
# current_account_mode(). At startup neither is bound — no user is logged in
# and no mode is chosen — so it silently resolved to the deployment default and
# a real-mode user's allowlist was never restored, with nothing saying so.
#
# It still restores only ONE scope, because the destination is the global
# settings lists. That limitation is Phase 5's remaining work; what these
# tests pin is that the scope is now chosen deliberately.
# ---------------------------------------------------------------------------


async def test_load_can_restore_a_scope_other_than_the_current_one(fake_conn, monkeypatch):
    """The actual fix: at startup the boot context is default/paper, but a
    real-mode user's rows must be reachable by naming them."""
    monkeypatch.setattr(settings, "trade_allowlist_domestic", [])
    fake_conn.rows = [
        {"key": "u-alice:real:trade_allowlist_domestic",
         "tickers": json.dumps(["005930.KS", "000660.KS"])},
    ]
    await trade_allowlist_store.load(user_id="u-alice", account_mode="real")
    # Read INSIDE the scope that was loaded. Reading from the ambient context
    # would see default/paper, which is the whole point of the change: a load
    # for one scope must not land in another.
    with user_scope("u-alice"), account_mode_scope("real"):
        assert settings.trade_allowlist_domestic == ["005930.KS", "000660.KS"]
    # ...and the ambient scope is untouched.
    assert settings.trade_allowlist_domestic == []


async def test_load_with_no_arguments_is_unchanged(fake_conn, monkeypatch):
    """Every existing call site passes nothing and must behave exactly as
    before — that is what makes this fix safe to land on its own."""
    monkeypatch.setattr(settings, "trade_allowlist_domestic", [])
    fake_conn.rows = [
        {"key": "default:paper:trade_allowlist_domestic", "tickers": json.dumps(["035420.KS"])},
    ]
    await trade_allowlist_store.load()
    assert settings.trade_allowlist_domestic == ["035420.KS"]


async def test_load_does_not_bleed_another_scopes_rows_in(fake_conn, monkeypatch):
    monkeypatch.setattr(settings, "trade_allowlist_domestic", [])
    # Both scopes visible to the query — this proves the KEY CONSTRUCTION
    # selects the right one, not that the SQL filter did it.
    fake_conn.rows = [
        {"key": "u-alice:paper:trade_allowlist_domestic", "tickers": json.dumps(["005930.KS"])},
        {"key": "u-alice:real:trade_allowlist_domestic", "tickers": json.dumps(["000660.KS"])},
    ]
    await trade_allowlist_store.load(user_id="u-alice", account_mode="real")
    with user_scope("u-alice"), account_mode_scope("real"):
        assert settings.trade_allowlist_domestic == ["000660.KS"]
    with user_scope("u-alice"), account_mode_scope("paper"):
        assert settings.trade_allowlist_domestic == []


async def test_migration_writes_back_to_the_scope_it_read(fake_conn, monkeypatch):
    """A partially-threaded scope would be worse than none: read one scope,
    migrate into another. The legacy row here belongs to paper, and the
    migration is gated on the DEFAULT user, so loading default/paper must
    migrate it under default/paper — not under whatever is current."""
    monkeypatch.setattr(settings, "trade_allowlist_domestic", [])
    fake_conn.rows = [
        {"key": "paper:trade_allowlist_domestic", "tickers": json.dumps(["005930.KS"])},
    ]
    await trade_allowlist_store.load(user_id="default", account_mode="paper")
    assert settings.trade_allowlist_domestic == ["005930.KS"]
    assert {key for key, _ in fake_conn.saved} == {"default:paper:trade_allowlist_domestic"}
    assert fake_conn.deleted == ["paper:trade_allowlist_domestic"]


async def test_save_can_target_an_explicit_scope(fake_conn):
    await trade_allowlist_store.save(
        "watchlist", ["AAPL"], user_id="u-alice", account_mode="real"
    )
    assert fake_conn.saved[0][0] == "u-alice:real:watchlist"
