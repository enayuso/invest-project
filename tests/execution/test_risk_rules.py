"""risk_rules resolver + validation tests (A-3), no DB required."""

from __future__ import annotations

import asyncio

import pytest

from app.config import settings
from app.execution import risk_rules
from app.user_context import get_current_user_id, user_scope

# The identity every non-scoped test runs under (settings.user_id, 'default').
_WHO = get_current_user_id()
# _overrides is keyed by (user_id, account_mode) since per-user account modes
# landed — these tests all run under the default mode.
_SCOPE = (_WHO, "paper")
from app.execution.kis_bridge import over_value_cap


@pytest.fixture(autouse=True)
def clean_overrides(monkeypatch):
    # Each test starts from an empty override cache; restored after.
    monkeypatch.setattr(risk_rules, "_overrides", {})


@pytest.fixture(autouse=True)
def _fresh_schema_guard(monkeypatch):
    # A fresh, un-triggered schema guard per test — _schema_lock is a
    # module-level asyncio.Lock bound to whatever event loop first acquires
    # it, and each test function gets its own loop (see kis_bridge._order_lock's
    # identical fixture in test_kis_bridge.py).
    monkeypatch.setattr(risk_rules, "_schema_ready", False)
    monkeypatch.setattr(risk_rules, "_schema_lock", asyncio.Lock())


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


class FakeRiskRulesConn:
    """Counts CREATE TABLE executions separately from data queries, so tests
    can assert the schema DDL only ever runs once per process."""

    def __init__(self):
        self.schema_execute_count = 0
        # keyed by (key, account_mode, user_id) to match the composite unique
        # index — user_id joined it in the multi-user migration Phase 4.
        self.rows_by_mode: dict[tuple[str, str, str], dict] = {}
        self.inserted_account_modes: list[str] = []
        self.inserted_user_ids: list[str] = []

    async def execute(self, query, *args):
        if "CREATE TABLE" in query:
            self.schema_execute_count += 1
            return
        if "INSERT INTO risk_rules" in query:
            key, value, account_mode, user_id = args
            self.rows_by_mode[(key, account_mode, user_id)] = {
                "key": key, "value": value, "user_id": user_id,
            }
            self.inserted_account_modes.append(account_mode)
            self.inserted_user_ids.append(user_id)
            return
        return

    async def fetch(self, query, *args):
        """`load()` now fetches EVERY mode in one query (one process serves
        both 실전 and 모의투자 users), so there is no account_mode argument to
        filter on. Each row carries its own mode instead — which the previous
        version dropped, and `load()`'s own try/except swallowed the resulting
        unpack error, leaving the cache silently empty rather than failing.
        모드 인자 없이 전체를 조회하므로 각 행이 자기 모드를 들고 있어야 함."""
        assert not args, f"load() no longer passes query args, got {args!r}"
        return [
            {**row, "account_mode": mode} for (_, mode, _u), row in self.rows_by_mode.items()
        ]

    def transaction(self):
        return _NoopTransaction()

    @property
    def rows(self):
        """Back-compat alias for tests that seed rows before account_mode
        scoping existed — seeds under whatever account_mode `settings`
        currently resolves to."""
        return [row for (_, mode, _u), row in self.rows_by_mode.items() if mode == settings.kis_account_mode]

    @rows.setter
    def rows(self, value):
        for row in value:
            user_id = row.setdefault("user_id", get_current_user_id())
            self.rows_by_mode[(row["key"], settings.kis_account_mode, user_id)] = row


class FakeRiskRulesPool:
    def __init__(self, conn):
        self._conn = conn

    def acquire(self):
        return _AcquireCM(self._conn)


@pytest.fixture
def fake_risk_rules_conn(monkeypatch):
    conn = FakeRiskRulesConn()

    async def fake_get_pool():
        return FakeRiskRulesPool(conn)

    monkeypatch.setattr("app.memory.db.get_pool", fake_get_pool, raising=False)
    return conn


async def test_ensure_schema_runs_ddl_only_once_across_repeated_loads(fake_risk_rules_conn):
    await risk_rules.load()
    await risk_rules.load()
    await risk_rules.load()
    assert fake_risk_rules_conn.schema_execute_count == 1


async def test_ensure_schema_runs_ddl_only_once_across_load_then_save(fake_risk_rules_conn):
    await risk_rules.load()
    await risk_rules.save({"kis_max_position_pct": 10.0})
    assert fake_risk_rules_conn.schema_execute_count == 1


def test_rule_falls_back_to_settings(monkeypatch):
    monkeypatch.setattr(settings, "kis_max_order_krw", 5_000_000.0)
    assert risk_rules.rule("kis_max_order_krw") == 5_000_000.0


def test_override_wins_over_settings(monkeypatch):
    monkeypatch.setattr(settings, "kis_max_order_krw", 5_000_000.0)
    risk_rules._overrides[_SCOPE] = {"kis_max_order_krw": 1_000_000.0}
    assert risk_rules.rule("kis_max_order_krw") == 1_000_000.0


def test_current_reports_overridden_keys():
    risk_rules._overrides[_SCOPE] = {"kis_max_position_pct": 10.0}
    snap = risk_rules.current()
    assert snap["values"]["kis_max_position_pct"] == 10.0
    assert "kis_max_position_pct" in snap["overridden"]


def test_coerce_rejects_unknown_key():
    with pytest.raises(ValueError, match="unknown risk rule"):
        risk_rules._coerce("not_a_rule", 1)


def test_coerce_rejects_negative():
    with pytest.raises(ValueError, match="non-negative"):
        risk_rules._coerce("kis_max_order_krw", -1)


def test_coerce_rejects_pct_over_100():
    with pytest.raises(ValueError, match="between 0 and 100"):
        risk_rules._coerce("kis_max_position_pct", 150)


def test_coerce_bool_from_string():
    assert risk_rules._coerce("kis_approval_required_enabled", "true") is True
    assert risk_rules._coerce("kis_approval_required_enabled", "false") is False


def test_coerce_bool_rejects_unrecognized_value():
    """Regression: this used to silently fall back to python's own
    `bool(raw)` truthiness for anything that wasn't a recognized string -
    "" -> False, any typo'd non-empty string (e.g. "disabled") -> True -
    for a safety-gate toggle like kis_approval_required_enabled. Must
    reject instead, matching runtime_settings._coerce's own sibling enum
    validation, which already refuses an unrecognized value outright."""
    with pytest.raises(ValueError, match="expects a boolean"):
        risk_rules._coerce("kis_approval_required_enabled", "disabled")
    with pytest.raises(ValueError, match="expects a boolean"):
        risk_rules._coerce("kis_approval_required_enabled", "")
    with pytest.raises(ValueError, match="expects a boolean"):
        risk_rules._coerce("kis_approval_required_enabled", None)
    with pytest.raises(ValueError, match="expects a boolean"):
        risk_rules._coerce("kis_approval_required_enabled", [])


def test_coerce_float_from_string(monkeypatch):
    monkeypatch.setattr(settings, "kis_max_order_krw", 5_000_000.0)
    assert risk_rules._coerce("kis_max_order_krw", "2000000") == 2_000_000.0


# ---------------- runtime override of an order-value cap/threshold may be
# raised by the user, bounded only by the deployment's absolute ceiling
# (kis_max_order_*_ceiling) — see _settings_ceiling/_MAX_VALUE. The stricter
# "may only ever tighten toward the boot value" policy this section used to
# test was superseded (multi-user migration: users own their own risk
# appetite; the deployment owns only the outer bound). ----------------


def test_coerce_accepts_order_cap_at_or_below_boot_ceiling(monkeypatch):
    monkeypatch.setattr(settings, "kis_max_order_krw", 5_000_000.0)
    assert risk_rules._coerce("kis_max_order_krw", 5_000_000.0) == 5_000_000.0
    assert risk_rules._coerce("kis_max_order_krw", 1_000_000.0) == 1_000_000.0


def test_coerce_does_not_ceiling_unrelated_float_keys(monkeypatch):
    # kis_max_position_pct isn't an order-value knob — its own 0-100 rule
    # applies (tested above), not the boot-ceiling one.
    monkeypatch.setattr(settings, "kis_max_position_pct", 25.0)
    assert risk_rules._coerce("kis_max_position_pct", 50.0) == 50.0


async def test_override_flows_into_the_value_cap_gate(monkeypatch):
    # The gate reads through risk_rules.rule(), so an override lowers the cap
    # without touching settings — a KRW order over the override is blocked.
    monkeypatch.setattr(settings, "kis_max_order_krw", 5_000_000.0)
    risk_rules._overrides[_SCOPE] = {"kis_max_order_krw": 500_000.0}
    blocked = over_value_cap("005930", 10, 100_000)  # 1,000,000 > 500,000 override
    assert blocked is not None and "cap" in blocked


# ---------------- cross-field validation: approval threshold vs value cap ----------------


def test_validate_cross_field_rejects_threshold_at_or_above_cap():
    with pytest.raises(ValueError, match="must be less than"):
        risk_rules._validate_cross_field({
            "kis_approval_threshold_krw": 5_000_000.0,
            "kis_max_order_krw": 5_000_000.0,
        })


def test_validate_cross_field_rejects_threshold_above_cap():
    with pytest.raises(ValueError, match="must be less than"):
        risk_rules._validate_cross_field({
            "kis_approval_threshold_krw": 6_000_000.0,
            "kis_max_order_krw": 5_000_000.0,
        })


def test_validate_cross_field_accepts_threshold_below_cap():
    risk_rules._validate_cross_field({
        "kis_approval_threshold_krw": 3_000_000.0,
        "kis_max_order_krw": 5_000_000.0,
    })  # must not raise


def test_validate_cross_field_checks_against_existing_value_when_only_one_side_saved(monkeypatch):
    monkeypatch.setattr(settings, "kis_max_order_usd", 1_000.0)
    # Saving only the threshold, above the existing (settings-default) cap.
    with pytest.raises(ValueError, match="kis_approval_threshold_usd"):
        risk_rules._validate_cross_field({"kis_approval_threshold_usd": 1_500.0})


def test_validate_cross_field_noop_when_neither_key_touched():
    risk_rules._validate_cross_field({"kis_max_position_pct": 10.0})  # must not raise


async def test_save_rejects_invalid_combination_before_touching_the_store(monkeypatch):
    async def should_not_be_called():
        raise AssertionError("save must validate before ever reaching the DB")

    # High enough that both values pass _coerce's own boot-ceiling check
    # (see the section below) — this test isolates _validate_cross_field's
    # rejection specifically, not the ceiling check.
    monkeypatch.setattr(settings, "kis_approval_threshold_krw", 10_000_000.0)
    monkeypatch.setattr(settings, "kis_max_order_krw", 10_000_000.0)
    monkeypatch.setattr("app.memory.db.get_pool", should_not_be_called, raising=False)
    with pytest.raises(ValueError, match="must be less than"):
        await risk_rules.save({"kis_approval_threshold_krw": 5_000_000.0, "kis_max_order_krw": 5_000_000.0})


# ---------------- account_mode scoping (2026-08-05) ----------------
#
# Before this, a relaxed order cap / disabled approval requirement set
# during paper testing would silently carry over the moment
# kis_account_mode flips to 'real', with no re-validation. Scoping the
# table by account_mode means the real account simply never sees a
# paper-mode override at all.


async def test_save_scopes_the_insert_to_the_current_account_mode(fake_risk_rules_conn, monkeypatch):
    monkeypatch.setattr(settings, "kis_account_mode", "paper")
    await risk_rules.save({"kis_max_position_pct": 50.0})
    assert fake_risk_rules_conn.inserted_account_modes == ["paper"]


async def test_switching_to_real_does_not_inherit_a_relaxed_paper_mode_cap(fake_risk_rules_conn, monkeypatch):
    monkeypatch.setattr(settings, "kis_account_mode", "paper")
    await risk_rules.save({"kis_max_position_pct": 50.0})  # relaxed for paper convenience

    # Simulate the process restarting into real mode.
    risk_rules._overrides.clear()
    monkeypatch.setattr(settings, "kis_account_mode", "real")
    monkeypatch.setattr(settings, "kis_max_position_pct", 10.0)
    await risk_rules.load()

    assert "kis_max_position_pct" not in risk_rules._overrides
    assert risk_rules.rule("kis_max_position_pct") == 10.0


async def test_real_mode_override_is_independent_of_paper_mode_override(fake_risk_rules_conn, monkeypatch):
    monkeypatch.setattr(settings, "kis_account_mode", "paper")
    await risk_rules.save({"kis_max_position_pct": 50.0})

    risk_rules._overrides.clear()
    monkeypatch.setattr(settings, "kis_account_mode", "real")
    # current() resolves every RULE_SPEC key, including the money caps that
    # now require an explicit real-mode value (app.config._require_real_
    # money_limits_on_real_account) — set them so this test can exercise
    # kis_max_position_pct in isolation without tripping that guard.
    for key in ("kis_max_order_krw", "kis_max_order_usd", "kis_approval_threshold_krw", "kis_approval_threshold_usd"):
        monkeypatch.setattr(settings, f"{key}_real", 1.0)
    await risk_rules.save({"kis_max_position_pct": 15.0})

    assert risk_rules.rule("kis_max_position_pct") == 15.0
    assert set(fake_risk_rules_conn.rows_by_mode.keys()) == {
        ("kis_max_position_pct", "paper", _WHO),
        ("kis_max_position_pct", "real", _WHO),
    }
    assert fake_risk_rules_conn.rows_by_mode[("kis_max_position_pct", "paper", _WHO)]["value"] == "50.0"


# ---------------------------------------------------------------------------
# Per-user scoping (multi-user migration Phase 4, plan §2.2): "Order value
# caps, position-% caps, approval thresholds all shared." A risk limit that
# someone else can raise on your behalf is not a limit.
# ---------------------------------------------------------------------------


async def test_one_users_relaxed_cap_does_not_relax_anothers(fake_risk_rules_conn, monkeypatch):
    monkeypatch.setattr(settings, "kis_max_order_krw", 100_000.0)
    with user_scope("u-alice"):
        await risk_rules.save({"kis_max_order_krw": 9_000_000.0})
        assert risk_rules.rule("kis_max_order_krw") == 9_000_000.0

    with user_scope("u-bob"):
        assert risk_rules.rule("kis_max_order_krw") == 100_000.0


async def test_each_user_keeps_their_own_override(fake_risk_rules_conn):
    with user_scope("u-alice"):
        await risk_rules.save({"kis_max_order_krw": 4_000_000.0})
    with user_scope("u-bob"):
        await risk_rules.save({"kis_max_order_krw": 5_000_000.0})
    with user_scope("u-alice"):
        assert risk_rules.rule("kis_max_order_krw") == 4_000_000.0
    with user_scope("u-bob"):
        assert risk_rules.rule("kis_max_order_krw") == 5_000_000.0


async def test_save_stamps_the_acting_user(fake_risk_rules_conn):
    with user_scope("u-alice"):
        await risk_rules.save({"kis_max_order_krw": 4_000_000.0})
    assert fake_risk_rules_conn.inserted_user_ids == ["u-alice"]


async def test_two_users_do_not_collide_on_one_row(fake_risk_rules_conn):
    with user_scope("u-alice"):
        await risk_rules.save({"kis_max_order_krw": 4_000_000.0})
    with user_scope("u-bob"):
        await risk_rules.save({"kis_max_order_krw": 5_000_000.0})
    assert set(fake_risk_rules_conn.rows_by_mode) == {
        ("kis_max_order_krw", "paper", "u-alice"),
        ("kis_max_order_krw", "paper", "u-bob"),
    }


async def test_load_restores_every_users_overrides(fake_risk_rules_conn, monkeypatch):
    monkeypatch.setattr(settings, "kis_max_order_krw", 100_000.0)
    # Also below both raised caps: the deployment default approval threshold
    # (3,000,000 KRW) would otherwise sit ABOVE alice's raised 1,000,000 cap,
    # tripping load()'s own threshold<cap cross-field guard (added after this
    # test) and dropping her override right back to defaults — a real
    # invariant, not a bug, but this test is about scope isolation, not that
    # guard, so keep the threshold out of its way.
    monkeypatch.setattr(settings, "kis_approval_threshold_krw", 10_000.0)
    fake_risk_rules_conn.rows = [
        {"key": "kis_max_order_krw", "value": 1_000_000.0, "user_id": "u-alice", "account_mode": "paper"},
        {"key": "kis_max_order_krw", "value": 2_000_000.0, "user_id": "u-bob", "account_mode": "paper"},
    ]
    await risk_rules.load()
    assert risk_rules.rule("kis_max_order_krw", "u-alice") == 1_000_000.0
    assert risk_rules.rule("kis_max_order_krw", "u-bob") == 2_000_000.0
    assert risk_rules.rule("kis_max_order_krw", "u-carol") == 100_000.0
