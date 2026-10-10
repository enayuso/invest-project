"""runtime_settings resolver + safety-guard tests, no DB required. Mirrors
test_risk_rules.py's fake Postgres pool/conn pattern."""

from __future__ import annotations

import asyncio

import pytest

from app.config import AIProfile, ExecutionMode, settings
from app.execution import runtime_settings
from app.config import account_mode_scope, current_account_mode
from app.user_context import get_current_user_id, user_scope

# The scope every non-scoped test runs under: default identity, paper mode.
_SCOPE = (get_current_user_id(), "paper")


@pytest.fixture(autouse=True)
def clean_overrides(monkeypatch):
    monkeypatch.setattr(runtime_settings, "_overrides", {})


@pytest.fixture(autouse=True)
def _fresh_schema_guard(monkeypatch):
    # Same reasoning as test_risk_rules.py's identical fixture — _schema_lock
    # is bound to whatever event loop first acquires it, and each test
    # function gets its own loop.
    monkeypatch.setattr(runtime_settings, "_schema_ready", False)
    monkeypatch.setattr(runtime_settings, "_schema_lock", asyncio.Lock())


class _AcquireCM:
    def __init__(self, conn):
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, *exc):
        return False


class _NoopTransaction:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class FakeRuntimeSettingsConn:
    def __init__(self):
        self.schema_execute_count = 0
        # keyed by (key, account_mode, user_id) to match the composite unique
        # index — user_id joined it in the multi-user migration Phase 4, and
        # the fake has to enforce the same shape or a cross-user collision
        # would pass here and only fail against real Postgres.
        self.rows_by_mode: dict[tuple[str, str, str], dict] = {}
        self.inserted_account_modes: list[str] = []
        self.inserted_user_ids: list[str] = []

    async def execute(self, query, *args):
        if "CREATE TABLE" in query:
            self.schema_execute_count += 1
            return
        if "INSERT INTO runtime_settings" in query:
            key, value, account_mode, user_id = args
            self.rows_by_mode[(key, account_mode, user_id)] = {
                "key": key, "value": value, "user_id": user_id,
                "account_mode": account_mode,
            }
            self.inserted_account_modes.append(account_mode)
            self.inserted_user_ids.append(user_id)
            return
        return

    async def fetch(self, query, *args):
        # load() selects every (user, mode) scope now, so there is no
        # account_mode bind parameter to filter on. A fake that still filtered
        # would hide exactly the bug this change fixes.
        assert not args, f"unexpected bind params for a full-table select: {args}"
        return list(self.rows_by_mode.values())

    def transaction(self):
        return _NoopTransaction()

    @property
    def rows(self):
        """Back-compat alias for tests that seed rows before account_mode /
        user_id scoping existed — seeds under whatever account_mode `settings`
        currently resolves to, owned by the current identity."""
        return [row for (_, mode, _u), row in self.rows_by_mode.items() if mode == settings.kis_account_mode]

    @rows.setter
    def rows(self, value):
        for row in value:
            user_id = row.setdefault("user_id", get_current_user_id())
            mode = row.setdefault("account_mode", current_account_mode())
            self.rows_by_mode[(row["key"], mode, user_id)] = row


class FakeRuntimeSettingsPool:
    def __init__(self, conn):
        self._conn = conn

    def acquire(self):
        return _AcquireCM(self._conn)


@pytest.fixture
def fake_runtime_settings_conn(monkeypatch):
    conn = FakeRuntimeSettingsConn()

    async def fake_get_pool():
        return FakeRuntimeSettingsPool(conn)

    monkeypatch.setattr("app.memory.db.get_pool", fake_get_pool, raising=False)
    return conn


def test_rule_falls_back_to_settings(monkeypatch):
    monkeypatch.setattr(settings, "execution_mode", ExecutionMode.SEMI_AUTO)
    assert runtime_settings.rule("execution_mode") is ExecutionMode.SEMI_AUTO


def test_override_wins_over_settings(monkeypatch):
    monkeypatch.setattr(settings, "ai_profile", AIProfile.PRODUCTION)
    runtime_settings._overrides[_SCOPE] = {"ai_profile": AIProfile.TEST_AGGRESSIVE}
    assert runtime_settings.rule("ai_profile") is AIProfile.TEST_AGGRESSIVE


def test_current_reports_overridden_keys_and_read_only_account_mode(monkeypatch):
    monkeypatch.setattr(settings, "kis_account_mode", "paper")
    runtime_settings._overrides[_SCOPE] = {"execution_mode": ExecutionMode.FULL_AUTO}
    snap = runtime_settings.current()
    assert snap["values"]["execution_mode"] == "full_auto"
    assert "execution_mode" in snap["overridden"]
    assert snap["kis_account_mode"] == "paper"


def test_coerce_rejects_unknown_key():
    with pytest.raises(ValueError, match="unknown runtime setting"):
        runtime_settings._coerce("not_a_setting", "x")


def test_coerce_rejects_invalid_enum_value():
    with pytest.raises(ValueError, match="expects one of"):
        runtime_settings._coerce("ai_profile", "not_a_profile")


def test_coerce_accepts_valid_string():
    assert runtime_settings._coerce("execution_mode", "full_auto") is ExecutionMode.FULL_AUTO


async def test_ensure_schema_runs_ddl_only_once_across_repeated_loads(fake_runtime_settings_conn):
    await runtime_settings.load()
    await runtime_settings.load()
    assert fake_runtime_settings_conn.schema_execute_count == 1


async def test_load_restores_stored_override(fake_runtime_settings_conn):
    # asyncpg returns jsonb as raw JSON text — a string value comes back
    # quoted (e.g. '"full_auto"'), not the plain string. See load()'s own
    # comment for why this bit the first version of this function.
    fake_runtime_settings_conn.rows = [{"key": "execution_mode", "value": '"full_auto"'}]
    await runtime_settings.load()
    assert runtime_settings.rule("execution_mode") is ExecutionMode.FULL_AUTO


# load() must re-check the real-account/ai_profile guard, not just save() —
# without this, an override saved while ai_profile_live_ack=True survives a
# restart even after the operator flips ai_profile_live_ack back to False
# specifically to re-arm this safety check. save()'s own guard only ever
# ran once, at the moment the value was originally set.
async def test_load_drops_stored_ai_profile_override_if_the_ack_was_since_revoked(
    fake_runtime_settings_conn, monkeypatch
):
    monkeypatch.setattr(settings, "kis_account_mode", "real")
    monkeypatch.setattr(settings, "ai_profile_live_ack", False)
    fake_runtime_settings_conn.rows = [{"key": "ai_profile", "value": '"test_aggressive"'}]

    await runtime_settings.load()

    assert "ai_profile" not in runtime_settings._overrides
    assert runtime_settings.rule("ai_profile") is settings.ai_profile  # falls back to the config default


async def test_load_keeps_stored_ai_profile_override_while_the_ack_still_holds(
    fake_runtime_settings_conn, monkeypatch
):
    monkeypatch.setattr(settings, "kis_account_mode", "real")
    monkeypatch.setattr(settings, "ai_profile_live_ack", True)
    fake_runtime_settings_conn.rows = [{"key": "ai_profile", "value": '"test_aggressive"'}]

    await runtime_settings.load()

    assert runtime_settings.rule("ai_profile") is AIProfile.TEST_AGGRESSIVE


async def test_save_persists_and_updates_cache(fake_runtime_settings_conn, monkeypatch):
    monkeypatch.setattr(settings, "kis_account_mode", "paper")
    result = await runtime_settings.save({"execution_mode": "manual"})
    assert result["values"]["execution_mode"] == "manual"
    assert runtime_settings.rule("execution_mode") is ExecutionMode.MANUAL


# ---------------- real-account safety guard ----------------


async def test_save_blocks_non_production_profile_on_real_account_without_ack(monkeypatch):
    monkeypatch.setattr(settings, "kis_account_mode", "real")
    monkeypatch.setattr(settings, "ai_profile_live_ack", False)
    with pytest.raises(ValueError, match="ai_profile_live_ack"):
        await runtime_settings.save({"ai_profile": "test_aggressive"})


async def test_save_allows_non_production_profile_on_real_account_with_ack(fake_runtime_settings_conn, monkeypatch):
    monkeypatch.setattr(settings, "kis_account_mode", "real")
    monkeypatch.setattr(settings, "ai_profile_live_ack", True)
    result = await runtime_settings.save({"ai_profile": "debug"})
    assert result["values"]["ai_profile"] == "debug"


async def test_save_allows_production_profile_on_real_account_without_ack(fake_runtime_settings_conn, monkeypatch):
    monkeypatch.setattr(settings, "kis_account_mode", "real")
    monkeypatch.setattr(settings, "ai_profile_live_ack", False)
    result = await runtime_settings.save({"ai_profile": "production"})
    assert result["values"]["ai_profile"] == "production"


def test_validate_real_account_guard_noop_when_ai_profile_not_touched(monkeypatch):
    monkeypatch.setattr(settings, "kis_account_mode", "real")
    monkeypatch.setattr(settings, "ai_profile_live_ack", False)
    runtime_settings._validate_real_account_guard({"execution_mode": ExecutionMode.FULL_AUTO}, "real")  # must not raise


# ---------------- account_mode scoping (2026-08-05) ----------------
#
# Before this, a FULL_AUTO override saved while paper-testing would silently
# survive a switch to kis_account_mode='real' with zero re-validation —
# load() only restores from Postgres, it never re-runs
# _validate_real_account_guard. Scoping the table by account_mode means the
# real account simply never sees a paper-mode override at all.


async def test_save_scopes_the_insert_to_the_current_account_mode(fake_runtime_settings_conn, monkeypatch):
    monkeypatch.setattr(settings, "kis_account_mode", "paper")
    await runtime_settings.save({"execution_mode": "full_auto"})
    assert fake_runtime_settings_conn.inserted_account_modes == ["paper"]


async def test_switching_to_real_does_not_inherit_a_paper_mode_override(fake_runtime_settings_conn, monkeypatch):
    monkeypatch.setattr(settings, "kis_account_mode", "paper")
    await runtime_settings.save({"execution_mode": "full_auto"})

    # Simulate the process restarting into real mode.
    runtime_settings._overrides.clear()
    monkeypatch.setattr(settings, "kis_account_mode", "real")
    monkeypatch.setattr(settings, "execution_mode", ExecutionMode.MANUAL)
    await runtime_settings.load()

    assert "execution_mode" not in runtime_settings._overrides.get((get_current_user_id(), "real"), {})
    assert runtime_settings.rule("execution_mode") is ExecutionMode.MANUAL


async def test_real_mode_override_is_independent_of_paper_mode_override(fake_runtime_settings_conn, monkeypatch):
    monkeypatch.setattr(settings, "kis_account_mode", "paper")
    await runtime_settings.save({"execution_mode": "full_auto"})

    runtime_settings._overrides.clear()
    monkeypatch.setattr(settings, "kis_account_mode", "real")
    monkeypatch.setattr(settings, "ai_profile_live_ack", True)
    await runtime_settings.save({"execution_mode": "semi_auto"})

    assert runtime_settings.rule("execution_mode") is ExecutionMode.SEMI_AUTO
    who = get_current_user_id()
    assert set(fake_runtime_settings_conn.rows_by_mode.keys()) == {
        ("execution_mode", "paper", who),
        ("execution_mode", "real", who),
    }
    assert fake_runtime_settings_conn.rows_by_mode[("execution_mode", "paper", who)]["value"] == '"full_auto"'


# ---------------- per-user scoping (multi-user migration Phase 4) ----------------
#
# THE highest-severity item in docs/multi-user-credentials-plan.md §2.2. Before
# user_id joined the key, execution_mode was one value for the whole process:
# one person selecting FULL_AUTO put every other user's account into
# autonomous trading, using those users' own credentials, against their own
# real money. These tests exist to make that unrepeatable.


async def test_one_users_full_auto_does_not_touch_another(fake_runtime_settings_conn, monkeypatch):
    """The exact incident this column prevents."""
    monkeypatch.setattr(settings, "kis_account_mode", "paper")
    monkeypatch.setattr(settings, "execution_mode", ExecutionMode.MANUAL)

    with user_scope("u-alice"):
        await runtime_settings.save({"execution_mode": "full_auto"})
        assert runtime_settings.rule("execution_mode") is ExecutionMode.FULL_AUTO

    # Bob never opted in. He must still be MANUAL — the settings default —
    # not inherit Alice's automation.
    with user_scope("u-bob"):
        assert runtime_settings.rule("execution_mode") is ExecutionMode.MANUAL


async def test_each_user_keeps_their_own_override(fake_runtime_settings_conn, monkeypatch):
    monkeypatch.setattr(settings, "kis_account_mode", "paper")
    with user_scope("u-alice"):
        await runtime_settings.save({"execution_mode": "full_auto"})
    with user_scope("u-bob"):
        await runtime_settings.save({"execution_mode": "semi_auto"})

    with user_scope("u-alice"):
        assert runtime_settings.rule("execution_mode") is ExecutionMode.FULL_AUTO
    with user_scope("u-bob"):
        assert runtime_settings.rule("execution_mode") is ExecutionMode.SEMI_AUTO


async def test_save_stamps_the_acting_user(fake_runtime_settings_conn, monkeypatch):
    monkeypatch.setattr(settings, "kis_account_mode", "paper")
    with user_scope("u-alice"):
        await runtime_settings.save({"execution_mode": "full_auto"})
    assert fake_runtime_settings_conn.inserted_user_ids == ["u-alice"]


async def test_two_users_do_not_collide_on_one_row(fake_runtime_settings_conn, monkeypatch):
    """The unique index is (key, account_mode, user_id). If user_id were
    missing from it, Bob's save would overwrite Alice's row rather than
    creating his own — one stored value for everyone, which is the old bug
    wearing a new column."""
    monkeypatch.setattr(settings, "kis_account_mode", "paper")
    with user_scope("u-alice"):
        await runtime_settings.save({"execution_mode": "full_auto"})
    with user_scope("u-bob"):
        await runtime_settings.save({"execution_mode": "semi_auto"})
    assert set(fake_runtime_settings_conn.rows_by_mode) == {
        ("execution_mode", "paper", "u-alice"),
        ("execution_mode", "paper", "u-bob"),
    }


async def test_load_restores_every_users_override(fake_runtime_settings_conn, monkeypatch):
    """load() must populate ALL users, not just the serviced one: rule() is
    synchronous and cannot fetch a missing user at call time, so anyone absent
    from the cache silently falls back to the settings default."""
    monkeypatch.setattr(settings, "kis_account_mode", "paper")
    monkeypatch.setattr(settings, "execution_mode", ExecutionMode.MANUAL)
    fake_runtime_settings_conn.rows = [
        {"key": "execution_mode", "value": '"full_auto"', "user_id": "u-alice"},
        {"key": "execution_mode", "value": '"semi_auto"', "user_id": "u-bob"},
    ]
    await runtime_settings.load()

    assert runtime_settings.rule("execution_mode", "u-alice") is ExecutionMode.FULL_AUTO
    assert runtime_settings.rule("execution_mode", "u-bob") is ExecutionMode.SEMI_AUTO
    assert runtime_settings.rule("execution_mode", "u-carol") is ExecutionMode.MANUAL


def test_current_reports_only_the_asking_users_overrides(monkeypatch):
    monkeypatch.setattr(settings, "kis_account_mode", "paper")
    monkeypatch.setattr(settings, "execution_mode", ExecutionMode.MANUAL)
    monkeypatch.setattr(runtime_settings, "_overrides", {
        ("u-alice", "paper"): {"execution_mode": ExecutionMode.FULL_AUTO},
    })
    assert runtime_settings.current("u-alice")["overridden"] == ["execution_mode"]
    bob = runtime_settings.current("u-bob")
    assert bob["overridden"] == []
    assert bob["values"]["execution_mode"] == "manual"
    # Same user, other mode: Alice's paper FULL_AUTO must not answer a real
    # read. This is the leak that was found in risk_rules' cache.
    alice_real = runtime_settings.current("u-alice", "real")
    assert alice_real["overridden"] == []
    assert alice_real["values"]["execution_mode"] == "manual"


# ---------------------------------------------------------------------------
# Per-mode scoping of the cache, and loading every mode at startup.
#
# Two separate defects, fixed together (2026-08-06). Both were already found
# and fixed in risk_rules; these pin the same properties here.
#   1. _overrides keyed by user alone let a paper override answer a real read.
#   2. load() filtered WHERE account_mode = <mode current at startup>, but at
#      startup no user is bound, so that resolved to the deployment default and
#      every user on the other mode silently lost their overrides.
# ---------------------------------------------------------------------------


def test_paper_override_does_not_answer_a_real_read(monkeypatch):
    """execution_mode is exactly the value that must not cross this boundary:
    FULL_AUTO set while paper-testing must never drive a real-money account."""
    monkeypatch.setattr(settings, "execution_mode", ExecutionMode.MANUAL)
    monkeypatch.setattr(runtime_settings, "_overrides", {
        ("u-alice", "paper"): {"execution_mode": ExecutionMode.FULL_AUTO},
    })
    assert runtime_settings.rule("execution_mode", "u-alice", "paper") is ExecutionMode.FULL_AUTO
    assert runtime_settings.rule("execution_mode", "u-alice", "real") is ExecutionMode.MANUAL


async def test_load_restores_both_modes_for_the_same_user(fake_runtime_settings_conn, monkeypatch):
    monkeypatch.setattr(settings, "execution_mode", ExecutionMode.MANUAL)
    fake_runtime_settings_conn.rows = [
        {"key": "execution_mode", "value": '"full_auto"', "user_id": "u-alice", "account_mode": "paper"},
        {"key": "execution_mode", "value": '"semi_auto"', "user_id": "u-alice", "account_mode": "real"},
    ]
    await runtime_settings.load()
    assert runtime_settings.rule("execution_mode", "u-alice", "paper") is ExecutionMode.FULL_AUTO
    assert runtime_settings.rule("execution_mode", "u-alice", "real") is ExecutionMode.SEMI_AUTO


async def test_load_does_not_depend_on_the_mode_current_at_startup(fake_runtime_settings_conn, monkeypatch):
    """The actual regression. At startup no user is bound, so
    current_account_mode() returns the deployment default — a real-mode user's
    overrides must still be loaded."""
    monkeypatch.setattr(settings, "kis_account_mode", "paper")  # deployment default
    monkeypatch.setattr(settings, "execution_mode", ExecutionMode.MANUAL)
    fake_runtime_settings_conn.rows = [
        {"key": "execution_mode", "value": '"semi_auto"', "user_id": "u-real-only", "account_mode": "real"},
    ]
    await runtime_settings.load()
    assert runtime_settings.rule("execution_mode", "u-real-only", "real") is ExecutionMode.SEMI_AUTO


async def test_save_writes_to_the_acting_mode_and_caches_it_there(fake_runtime_settings_conn, monkeypatch):
    monkeypatch.setattr(settings, "ai_profile_live_ack", True)
    with user_scope("u-alice"), account_mode_scope("real"):
        await runtime_settings.save({"execution_mode": "semi_auto"})
    assert fake_runtime_settings_conn.inserted_account_modes == ["real"]
    assert ("u-alice", "real") in runtime_settings._overrides
    assert ("u-alice", "paper") not in runtime_settings._overrides


async def test_saving_one_mode_leaves_the_other_alone(fake_runtime_settings_conn, monkeypatch):
    monkeypatch.setattr(settings, "execution_mode", ExecutionMode.MANUAL)
    monkeypatch.setattr(settings, "ai_profile_live_ack", True)
    with user_scope("u-alice"), account_mode_scope("paper"):
        await runtime_settings.save({"execution_mode": "full_auto"})
    with user_scope("u-alice"), account_mode_scope("real"):
        assert runtime_settings.rule("execution_mode") is ExecutionMode.MANUAL
