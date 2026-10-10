"""app.execution.trailing_stop — the trailing-stop ratchet. compute_trailing_
pct/effective_stop_loss are pure and tested directly; update_high_water_mark
is tested against a fake asyncpg pool/connection (no real Postgres)."""

from __future__ import annotations

from datetime import date, datetime, timezone

import pytest

from app.execution import trailing_stop


# ---------------------------------------------------------------------------
# compute_trailing_pct
# ---------------------------------------------------------------------------


def test_compute_trailing_pct_scales_with_atr(monkeypatch):
    monkeypatch.setattr(trailing_stop.risk_rules, "rule", lambda key: {"trailing_stop_atr_multiplier": 1.75}[key])
    assert trailing_stop.compute_trailing_pct(0.04) == pytest.approx(0.07)


def test_compute_trailing_pct_falls_back_when_atr_unavailable(monkeypatch):
    monkeypatch.setattr(trailing_stop.risk_rules, "rule", lambda key: {"trailing_stop_fallback_pct": 0.08}[key])
    assert trailing_stop.compute_trailing_pct(None) == 0.08


def test_compute_trailing_pct_falls_back_when_atr_zero_or_negative(monkeypatch):
    monkeypatch.setattr(trailing_stop.risk_rules, "rule", lambda key: {"trailing_stop_fallback_pct": 0.08}[key])
    assert trailing_stop.compute_trailing_pct(0.0) == 0.08
    assert trailing_stop.compute_trailing_pct(-0.01) == 0.08


# ---------------------------------------------------------------------------
# effective_stop_loss — the entire safety argument: max(pm_stop, trailing)
# ---------------------------------------------------------------------------


def test_effective_stop_loss_never_looser_than_pm_stop_before_activation(monkeypatch):
    monkeypatch.setattr(trailing_stop.risk_rules, "rule", lambda key: 0.04)
    # High-water barely above entry — below the activation threshold.
    result = trailing_stop.effective_stop_loss(
        pm_stop_loss=90.0, entry_price=100.0, high_water_price=101.0, trailing_pct=0.1,
    )
    assert result == 90.0


def test_effective_stop_loss_ratchets_up_once_activated(monkeypatch):
    monkeypatch.setattr(trailing_stop.risk_rules, "rule", lambda key: 0.04)
    # High-water well past activation (100 * 1.04 = 104); trailing floor
    # (120 * 0.9 = 108) is tighter than the PM's own stop (90).
    result = trailing_stop.effective_stop_loss(
        pm_stop_loss=90.0, entry_price=100.0, high_water_price=120.0, trailing_pct=0.1,
    )
    assert result == pytest.approx(108.0)


def test_effective_stop_loss_never_picks_a_looser_trailing_floor(monkeypatch):
    # Real incident this guards against structurally: if the trailing floor
    # ever computed LOOSER than the PM's stop (e.g. a very wide
    # trailing_pct), max() must still pick the PM's own (tighter) stop.
    monkeypatch.setattr(trailing_stop.risk_rules, "rule", lambda key: 0.04)
    result = trailing_stop.effective_stop_loss(
        pm_stop_loss=95.0, entry_price=100.0, high_water_price=105.0, trailing_pct=0.5,
    )
    assert result == 95.0  # trailing floor (105*0.5=52.5) is looser -> PM stop wins


# ---------------------------------------------------------------------------
# update_high_water_mark — fake pool/connection, no real Postgres
# ---------------------------------------------------------------------------


class _FakeConn:
    def __init__(self, existing_row=None):
        # {"entry_time":..., "high_water_price":..., "trailing_pct":...,
        #  and since 2026-08-27 "adjudicated_through"/"adjudicated_pm_stop"}
        self.existing = existing_row
        self.executed_ddl = False
        self.last_upsert_query = ""
        self.last_update_query = ""

    async def execute(self, query, *args):
        # The DDL bootstrap passes only the schema string; the 2026-08-27
        # watermark writer passes the UPDATE plus its six bind parameters.
        if not args:
            self.executed_ddl = True
            return "CREATE"
        self.last_update_query = query
        ticker, account_mode, user_id, entry_time, through, pm_stop = args
        row = self.existing
        if row is None or row["entry_time"] != entry_time:
            return "UPDATE 0"  # stale streak / no row: a correct no-op
        current = row.get("adjudicated_through")
        if current is not None and not current < through:
            return "UPDATE 0"  # the WHERE clause refuses to move it backward
        row["adjudicated_through"] = through
        row["adjudicated_pm_stop"] = pm_stop
        return "UPDATE 1"

    async def fetchrow(self, query, *args):
        if "adjudicated_through, adjudicated_pm_stop" in query:
            ticker, account_mode, user_id, entry_time = args
            if self.existing is None or self.existing["entry_time"] != entry_time:
                return None
            return {
                "adjudicated_through": self.existing.get("adjudicated_through"),
                "adjudicated_pm_stop": self.existing.get("adjudicated_pm_stop"),
            }
        self.last_upsert_query = query
        ticker, account_mode, user_id, entry_time, candidate_high, trailing_pct = args
        same_streak = self.existing is not None and self.existing["entry_time"] == entry_time
        if same_streak:
            high_water = max(self.existing["high_water_price"], candidate_high)
            pct = self.existing["trailing_pct"]
            through = self.existing.get("adjudicated_through")
            through_pm = self.existing.get("adjudicated_pm_stop")
        else:
            high_water = candidate_high
            pct = trailing_pct
            through = None  # the UPSERT's entry_time-mismatch branch
            through_pm = None
        self.existing = {
            "entry_time": entry_time, "high_water_price": high_water, "trailing_pct": pct,
            "adjudicated_through": through, "adjudicated_pm_stop": through_pm,
        }
        return {"high_water_price": high_water, "trailing_pct": pct}


class _AcquireCtx:
    def __init__(self, conn):
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, *exc_info):
        return False


class _FakePool:
    def __init__(self, conn):
        self._conn = conn

    def acquire(self):
        return _AcquireCtx(self._conn)


def _fake_ready_pool(conn):
    async def _inner():
        return _FakePool(conn)

    return _inner


async def test_update_high_water_mark_initializes_on_first_call(monkeypatch):
    conn = _FakeConn(existing_row=None)
    monkeypatch.setattr(trailing_stop, "_ready_pool", _fake_ready_pool(conn))
    monkeypatch.setattr(trailing_stop.risk_rules, "rule", lambda key: {"trailing_stop_fallback_pct": 0.08}[key])

    entry_time = datetime(2026, 7, 24, 9, 0, tzinfo=timezone.utc)
    high_water, pct = await trailing_stop.update_high_water_mark(
        "AAPL", entry_time, entry_price=100.0, current_price=100.0, atr_pct=None,
    )
    assert high_water == 100.0
    assert pct == 0.08


async def test_update_high_water_mark_ratchets_up_within_the_same_streak(monkeypatch):
    entry_time = datetime(2026, 7, 24, 9, 0, tzinfo=timezone.utc)
    conn = _FakeConn(existing_row={"entry_time": entry_time, "high_water_price": 105.0, "trailing_pct": 0.1})
    monkeypatch.setattr(trailing_stop, "_ready_pool", _fake_ready_pool(conn))
    monkeypatch.setattr(trailing_stop.risk_rules, "rule", lambda key: {"trailing_stop_fallback_pct": 0.08}[key])

    high_water, pct = await trailing_stop.update_high_water_mark(
        "AAPL", entry_time, entry_price=100.0, current_price=110.0, atr_pct=None,
    )
    assert high_water == 110.0
    assert pct == 0.1  # preserved from the streak's start, not recomputed


async def test_update_high_water_mark_never_lowers_within_the_same_streak(monkeypatch):
    entry_time = datetime(2026, 7, 24, 9, 0, tzinfo=timezone.utc)
    conn = _FakeConn(existing_row={"entry_time": entry_time, "high_water_price": 120.0, "trailing_pct": 0.1})
    monkeypatch.setattr(trailing_stop, "_ready_pool", _fake_ready_pool(conn))
    monkeypatch.setattr(trailing_stop.risk_rules, "rule", lambda key: {"trailing_stop_fallback_pct": 0.08}[key])

    # Current price has pulled back from the earlier high — must not lower it.
    high_water, pct = await trailing_stop.update_high_water_mark(
        "AAPL", entry_time, entry_price=100.0, current_price=115.0, atr_pct=None,
    )
    assert high_water == 120.0


async def test_update_high_water_mark_resets_on_a_new_entry_time(monkeypatch):
    # An add-on (or a reopened position after going flat) writes a new
    # trades row with a different entry_time — this re-baselines the
    # high-water mark and recomputes trailing_pct fresh, per this module's
    # own docstring on why that's consistent, not a bug.
    old_entry_time = datetime(2026, 7, 24, 9, 0, tzinfo=timezone.utc)
    new_entry_time = datetime(2026, 7, 28, 9, 0, tzinfo=timezone.utc)
    conn = _FakeConn(existing_row={"entry_time": old_entry_time, "high_water_price": 150.0, "trailing_pct": 0.1})
    monkeypatch.setattr(trailing_stop, "_ready_pool", _fake_ready_pool(conn))
    monkeypatch.setattr(trailing_stop.risk_rules, "rule", lambda key: {"trailing_stop_fallback_pct": 0.05}[key])

    high_water, pct = await trailing_stop.update_high_water_mark(
        "AAPL", new_entry_time, entry_price=90.0, current_price=90.0, atr_pct=None,
    )
    assert high_water == 90.0  # NOT 150.0 — old streak's high-water is stale
    assert pct == 0.05  # freshly recomputed, not the old streak's 0.1


# ---------------------------------------------------------------------------
# adjudicated_through — the daily-bar backstop's per-streak watermark
# (2026-08-27, the 047040.KS same-day high-before-low fix). See this module's
# own schema comment and ledger.check_and_apply_trailing_stop's docstring.
# ---------------------------------------------------------------------------


async def test_get_adjudicated_through_returns_the_watermark_for_a_matching_streak_and_stop(
    monkeypatch
):
    entry_time = datetime(2026, 8, 26, 3, 36, tzinfo=timezone.utc)
    conn = _FakeConn(existing_row={
        "entry_time": entry_time, "high_water_price": 21_950.0, "trailing_pct": 0.101173,
        "adjudicated_through": date(2026, 8, 26), "adjudicated_pm_stop": 15_065.06,
    })
    monkeypatch.setattr(trailing_stop, "_ready_pool", _fake_ready_pool(conn))

    got = await trailing_stop.get_adjudicated_through("047040.KS", entry_time, 15_065.06)
    assert got == date(2026, 8, 26)


async def test_get_adjudicated_through_returns_none_for_a_different_entry_time(monkeypatch):
    """A re-entry or add-on re-baselines entry_price/trailing_pct, so every
    earlier day's verdict was computed against parameters that no longer
    apply and the whole streak must be re-walked from scratch."""
    stored_entry_time = datetime(2026, 8, 26, 3, 36, tzinfo=timezone.utc)
    conn = _FakeConn(existing_row={
        "entry_time": stored_entry_time, "high_water_price": 21_950.0, "trailing_pct": 0.1,
        "adjudicated_through": date(2026, 8, 26), "adjudicated_pm_stop": 15_065.06,
    })
    monkeypatch.setattr(trailing_stop, "_ready_pool", _fake_ready_pool(conn))

    got = await trailing_stop.get_adjudicated_through(
        "047040.KS", datetime(2026, 8, 27, 1, 0, tzinfo=timezone.utc), 15_065.06
    )
    assert got is None


async def test_get_adjudicated_through_returns_none_when_the_pm_stop_has_moved(monkeypatch):
    """The enforced precondition. A day's verdict is immutable within a
    streak ONLY as long as the PM's own static stop doesn't move underneath
    it — a RAISED pm stop can turn a previously cleared day into a real
    breach, and skipping that day would be a MISSED breach, the one failure
    mode strictly worse than the spurious-exit bug this watermark serves.
    Today a stop can only change by writing a new trades row (which changes
    entry_time, which resets the row), so this is belt-and-braces rather
    than a live bug — but it makes the argument enforced instead of merely
    documented: on any mismatch the walk re-tests everything."""
    entry_time = datetime(2026, 8, 26, 3, 36, tzinfo=timezone.utc)
    conn = _FakeConn(existing_row={
        "entry_time": entry_time, "high_water_price": 21_950.0, "trailing_pct": 0.1,
        "adjudicated_through": date(2026, 8, 26), "adjudicated_pm_stop": 90.0,
    })
    monkeypatch.setattr(trailing_stop, "_ready_pool", _fake_ready_pool(conn))

    got = await trailing_stop.get_adjudicated_through("047040.KS", entry_time, 95.0)
    assert got is None


async def test_get_adjudicated_through_returns_none_when_nothing_was_adjudicated_yet(monkeypatch):
    entry_time = datetime(2026, 8, 26, 3, 36, tzinfo=timezone.utc)
    conn = _FakeConn(existing_row={
        "entry_time": entry_time, "high_water_price": 21_950.0, "trailing_pct": 0.1,
        "adjudicated_through": None, "adjudicated_pm_stop": None,
    })
    monkeypatch.setattr(trailing_stop, "_ready_pool", _fake_ready_pool(conn))

    assert await trailing_stop.get_adjudicated_through("047040.KS", entry_time, 90.0) is None


async def test_record_adjudicated_through_never_moves_the_watermark_backward(monkeypatch):
    """Monotonic: a pass that happened to walk a shorter window (a fresh
    get_open_stop_loss trade_date, a partial daily series) can never
    un-adjudicate days that were already settled."""
    entry_time = datetime(2026, 8, 26, 3, 36, tzinfo=timezone.utc)
    conn = _FakeConn(existing_row={
        "entry_time": entry_time, "high_water_price": 21_950.0, "trailing_pct": 0.1,
        "adjudicated_through": date(2026, 8, 26), "adjudicated_pm_stop": 15_065.06,
    })
    monkeypatch.setattr(trailing_stop, "_ready_pool", _fake_ready_pool(conn))

    await trailing_stop.record_adjudicated_through(
        "047040.KS", entry_time, date(2026, 8, 20), 15_065.06
    )
    # The refusal lives in the SQL's own WHERE clause, and the fake honours it.
    assert "adjudicated_through < $5" in conn.last_update_query
    assert conn.existing["adjudicated_through"] == date(2026, 8, 26)  # unmoved

    await trailing_stop.record_adjudicated_through(
        "047040.KS", entry_time, date(2026, 8, 27), 15_065.06
    )
    assert conn.existing["adjudicated_through"] == date(2026, 8, 27)  # forward is fine


async def test_record_adjudicated_through_is_a_noop_for_a_stale_entry_time(monkeypatch):
    """Either the streak reset mid-pass (so the walk's verdicts are already
    void) or exit_monitor never established trailing state at all. Both are
    correctly no-ops, not errors — a missing watermark only ever costs a
    redundant re-walk."""
    conn = _FakeConn(existing_row={
        "entry_time": datetime(2026, 8, 26, 3, 36, tzinfo=timezone.utc),
        "high_water_price": 21_950.0, "trailing_pct": 0.1,
        "adjudicated_through": None, "adjudicated_pm_stop": None,
    })
    monkeypatch.setattr(trailing_stop, "_ready_pool", _fake_ready_pool(conn))

    await trailing_stop.record_adjudicated_through(
        "047040.KS", datetime(2026, 8, 27, 1, 0, tzinfo=timezone.utc),
        date(2026, 8, 27), 15_065.06,
    )
    assert conn.existing["adjudicated_through"] is None  # nothing written, no exception


async def test_update_high_water_mark_resets_the_watermark_on_a_new_entry_time(monkeypatch):
    """Guards the reset semantics the re-entry/add-on scenario depends on:
    the UPSERT carries the watermark forward within a streak and nulls it
    the moment entry_time changes."""
    old_entry_time = datetime(2026, 7, 24, 9, 0, tzinfo=timezone.utc)
    conn = _FakeConn(existing_row={
        "entry_time": old_entry_time, "high_water_price": 150.0, "trailing_pct": 0.1,
        "adjudicated_through": date(2026, 7, 23), "adjudicated_pm_stop": 90.0,
    })
    monkeypatch.setattr(trailing_stop, "_ready_pool", _fake_ready_pool(conn))
    monkeypatch.setattr(trailing_stop.risk_rules, "rule", lambda key: {"trailing_stop_fallback_pct": 0.05}[key])

    # Same streak: preserved.
    await trailing_stop.update_high_water_mark(
        "AAPL", old_entry_time, entry_price=100.0, current_price=160.0, atr_pct=None,
    )
    assert conn.existing["adjudicated_through"] == date(2026, 7, 23)
    assert conn.existing["adjudicated_pm_stop"] == 90.0
    assert "adjudicated_through = CASE" in conn.last_upsert_query
    assert "ELSE NULL" in conn.last_upsert_query

    # New streak: reset.
    await trailing_stop.update_high_water_mark(
        "AAPL", datetime(2026, 7, 28, 9, 0, tzinfo=timezone.utc),
        entry_price=90.0, current_price=90.0, atr_pct=None,
    )
    assert conn.existing["adjudicated_through"] is None
    assert conn.existing["adjudicated_pm_stop"] is None
