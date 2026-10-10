"""app.backtest.checkpoint — asyncpg pool/connection faked, no real Postgres.
Same FakeConn/FakePool shape as test_experiments.py, extended with execute()
for the INSERT/DELETE this module also needs."""

from __future__ import annotations

from datetime import date, datetime

import pytest

from app.backtest import checkpoint
from app.backtest.metrics import BacktestRecord
from app.state import PortfolioDecision, PortfolioRating


@pytest.fixture(autouse=True)
def _fixed_git_commit(monkeypatch):
    # Deterministic across environments (this test file shouldn't depend on
    # what's actually checked out) and lets test_load_checkpoint_ignores_
    # mismatched_git_commit below control the "current" commit independently
    # of whatever save_checkpoint used.
    monkeypatch.setattr(checkpoint, "current_git_commit", lambda: "abc1234")


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


class FakeCheckpointConn:
    """In-memory stand-in for the `backtest_checkpoints` table — a plain
    dict keyed by label, since asyncpg's Record supports the same
    `row["col"]` access a dict does."""

    def __init__(self):
        self.rows: dict[str, dict] = {}
        self.raise_on_fetchrow = False
        self.raise_on_execute = False
        self._clock = 0  # fake monotonic updated_at, newest = highest

    async def fetchrow(self, query, *args):
        if self.raise_on_fetchrow:
            raise RuntimeError("connection lost")
        (label,) = args
        return self.rows.get(label)

    async def fetch(self, query, *args):
        return sorted(self.rows.values(), key=lambda r: r["updated_at"], reverse=True)

    async def execute(self, query, *args):
        if self.raise_on_execute:
            raise RuntimeError("connection lost")
        normalized = " ".join(query.split())
        if normalized.startswith("INSERT INTO backtest_checkpoints"):
            label, params, state, completed_date_index = args
            self._clock += 1
            self.rows[label] = {
                "label": label, "params": params, "state": state,
                "completed_date_index": completed_date_index, "updated_at": self._clock,
            }
        elif normalized.startswith("DELETE FROM backtest_checkpoints"):
            (label,) = args
            self.rows.pop(label, None)
        else:
            raise AssertionError(f"unexpected query: {query}")


async def _async_return(value):
    return value


def _decision(rating=PortfolioRating.BUY, stop_loss=90.0) -> PortfolioDecision:
    return PortfolioDecision(
        rating=rating,
        executive_summary="Momentum intact, entering here.",
        investment_thesis="Strong revenue growth and expanding margins support this rating.",
        price_target=120.0,
        time_horizon="3-6 months",
        stop_loss=stop_loss,
        confidence=0.8,
    )


def _record(trade_date, close, stop_loss=90.0, decision_id=None) -> BacktestRecord:
    return BacktestRecord(
        trade_date=trade_date,
        decision=_decision(stop_loss=stop_loss),
        close=close,
        adjustment_note="1% risk cap",
        trailing_high_water=95.0,
        trailing_entry_price=91.0,
        decision_id=decision_id,
    )


PARAMS = dict(
    tickers=["AAPL", "MSFT"], start=date(2026, 1, 1), end=date(2026, 6, 1), cadence_days=7, benchmark_ticker="SPY",
)


async def test_load_checkpoint_returns_none_without_label():
    result = await checkpoint.load_checkpoint(None, **PARAMS)
    assert result is None


async def test_save_checkpoint_noop_without_label():
    state = checkpoint.CheckpointState(
        completed_date_index=0, prev_records={}, per_ticker_records={}, per_ticker_equity_curves={},
        portfolio_equity_curve=[], running_high_water_mark=0.0,
    )
    # get_pool is deliberately left unpatched — a real call would try to
    # reach Postgres and fail; no exception here proves save_checkpoint
    # returned before ever touching the DB.
    await checkpoint.save_checkpoint(None, **PARAMS, state=state)


async def test_clear_checkpoint_noop_without_label():
    await checkpoint.clear_checkpoint(None)  # no raise = pass (get_pool unpatched)


async def test_load_checkpoint_returns_none_when_no_row(monkeypatch):
    conn = FakeCheckpointConn()
    monkeypatch.setattr(checkpoint, "get_pool", lambda: _async_return(FakePool(conn)))
    result = await checkpoint.load_checkpoint("my-run", **PARAMS)
    assert result is None


async def test_load_checkpoint_degrades_on_db_error(monkeypatch):
    conn = FakeCheckpointConn()
    conn.raise_on_fetchrow = True
    monkeypatch.setattr(checkpoint, "get_pool", lambda: _async_return(FakePool(conn)))
    result = await checkpoint.load_checkpoint("my-run", **PARAMS)
    assert result is None


async def test_save_checkpoint_degrades_on_db_error(monkeypatch):
    conn = FakeCheckpointConn()
    conn.raise_on_execute = True
    monkeypatch.setattr(checkpoint, "get_pool", lambda: _async_return(FakePool(conn)))
    state = checkpoint.CheckpointState(
        completed_date_index=0, prev_records={}, per_ticker_records={}, per_ticker_equity_curves={},
        portfolio_equity_curve=[], running_high_water_mark=0.0,
    )
    await checkpoint.save_checkpoint("my-run", **PARAMS, state=state)  # no raise = pass


async def test_save_then_load_roundtrips_the_full_state(monkeypatch):
    conn = FakeCheckpointConn()
    monkeypatch.setattr(checkpoint, "get_pool", lambda: _async_return(FakePool(conn)))

    aapl_record = _record(date(2026, 1, 8), close=101.5)
    state = checkpoint.CheckpointState(
        completed_date_index=2,
        prev_records={"AAPL": aapl_record},
        per_ticker_records={"AAPL": [_record(date(2026, 1, 1), close=100.0), aapl_record], "MSFT": []},
        per_ticker_equity_curves={"AAPL": [1000.0, 1015.0], "MSFT": [1000.0, 1000.0]},
        portfolio_equity_curve=[2000.0, 2015.0],
        running_high_water_mark=2015.0,
        ever_selected={"AAPL", "MSFT"},
        cap_trigger_counts={"1% per-ticker risk cap": 1},
        # Must match the autouse fixture's current_git_commit() so this
        # roundtrip actually resumes rather than being refused by load_
        # checkpoint's own resume gate (which compares against a FRESH
        # current_git_commit() call, by design — see checkpoint.py).
        git_commit="abc1234",
    )

    await checkpoint.save_checkpoint("my-run", **PARAMS, state=state)
    restored = await checkpoint.load_checkpoint("my-run", **PARAMS)

    assert restored is not None
    assert restored.completed_date_index == 2
    assert restored.prev_records["AAPL"].trade_date == date(2026, 1, 8)
    assert restored.prev_records["AAPL"].close == 101.5
    assert restored.prev_records["AAPL"].decision.rating == PortfolioRating.BUY
    assert restored.prev_records["AAPL"].decision.stop_loss == 90.0
    assert restored.prev_records["AAPL"].trailing_high_water == 95.0
    assert len(restored.per_ticker_records["AAPL"]) == 2
    assert restored.per_ticker_records["MSFT"] == []
    assert restored.per_ticker_equity_curves["AAPL"] == [1000.0, 1015.0]
    assert restored.portfolio_equity_curve == [2000.0, 2015.0]
    assert restored.running_high_water_mark == 2015.0
    assert restored.ever_selected == {"AAPL", "MSFT"}
    assert restored.cap_trigger_counts == {"1% per-ticker risk cap": 1}


async def test_load_checkpoint_ignores_mismatched_tickers(monkeypatch):
    conn = FakeCheckpointConn()
    monkeypatch.setattr(checkpoint, "get_pool", lambda: _async_return(FakePool(conn)))
    state = checkpoint.CheckpointState(
        completed_date_index=0, prev_records={}, per_ticker_records={}, per_ticker_equity_curves={},
        portfolio_equity_curve=[], running_high_water_mark=0.0,
    )
    await checkpoint.save_checkpoint("my-run", **PARAMS, state=state)

    mismatched = {**PARAMS, "tickers": ["AAPL", "NVDA"]}  # different ticker set, same label
    result = await checkpoint.load_checkpoint("my-run", **mismatched)
    assert result is None


async def test_load_checkpoint_ignores_mismatched_git_commit(monkeypatch):
    # The whole point of tagging the checkpoint with git_commit: a prompt/
    # pipeline change landed between the crash and the resume attempt must
    # never be silently spliced into one "comparable" backtest result.
    conn = FakeCheckpointConn()
    monkeypatch.setattr(checkpoint, "get_pool", lambda: _async_return(FakePool(conn)))
    monkeypatch.setattr(checkpoint, "current_git_commit", lambda: "abc1234")
    state = checkpoint.CheckpointState(
        completed_date_index=0, prev_records={}, per_ticker_records={}, per_ticker_equity_curves={},
        portfolio_equity_curve=[], running_high_water_mark=0.0,
    )
    await checkpoint.save_checkpoint("my-run", **PARAMS, state=state)

    monkeypatch.setattr(checkpoint, "current_git_commit", lambda: "def5678")  # code changed since the crash
    result = await checkpoint.load_checkpoint("my-run", **PARAMS)
    assert result is None


async def test_load_checkpoint_resumes_when_git_commit_unchanged(monkeypatch):
    conn = FakeCheckpointConn()
    monkeypatch.setattr(checkpoint, "get_pool", lambda: _async_return(FakePool(conn)))
    monkeypatch.setattr(checkpoint, "current_git_commit", lambda: "abc1234")
    state = checkpoint.CheckpointState(
        completed_date_index=1, prev_records={}, per_ticker_records={}, per_ticker_equity_curves={},
        portfolio_equity_curve=[], running_high_water_mark=0.0, git_commit="abc1234",
    )
    await checkpoint.save_checkpoint("my-run", **PARAMS, state=state)

    result = await checkpoint.load_checkpoint("my-run", **PARAMS)  # same commit, unchanged fixture
    assert result is not None
    assert result.completed_date_index == 1


async def test_load_checkpoint_ignores_mismatched_date_range(monkeypatch):
    conn = FakeCheckpointConn()
    monkeypatch.setattr(checkpoint, "get_pool", lambda: _async_return(FakePool(conn)))
    state = checkpoint.CheckpointState(
        completed_date_index=0, prev_records={}, per_ticker_records={}, per_ticker_equity_curves={},
        portfolio_equity_curve=[], running_high_water_mark=0.0,
    )
    await checkpoint.save_checkpoint("my-run", **PARAMS, state=state)

    mismatched = {**PARAMS, "end": date(2026, 7, 1)}
    result = await checkpoint.load_checkpoint("my-run", **mismatched)
    assert result is None


async def test_clear_checkpoint_deletes_the_row(monkeypatch):
    conn = FakeCheckpointConn()
    monkeypatch.setattr(checkpoint, "get_pool", lambda: _async_return(FakePool(conn)))
    state = checkpoint.CheckpointState(
        completed_date_index=0, prev_records={}, per_ticker_records={}, per_ticker_equity_curves={},
        portfolio_equity_curve=[], running_high_water_mark=0.0,
    )
    await checkpoint.save_checkpoint("my-run", **PARAMS, state=state)
    assert "my-run" in conn.rows

    await checkpoint.clear_checkpoint("my-run")
    assert "my-run" not in conn.rows

    result = await checkpoint.load_checkpoint("my-run", **PARAMS)
    assert result is None


async def test_save_checkpoint_carries_total_dates_through_roundtrip(monkeypatch):
    conn = FakeCheckpointConn()
    monkeypatch.setattr(checkpoint, "get_pool", lambda: _async_return(FakePool(conn)))
    state = checkpoint.CheckpointState(
        completed_date_index=4, prev_records={}, per_ticker_records={}, per_ticker_equity_curves={},
        portfolio_equity_curve=[1000.0, 1010.0], running_high_water_mark=1010.0, total_dates=15,
        git_commit="abc1234",
    )
    await checkpoint.save_checkpoint("my-run", **PARAMS, state=state)
    restored = await checkpoint.load_checkpoint("my-run", **PARAMS)
    assert restored.total_dates == 15


async def test_load_checkpoint_defaults_total_dates_to_zero_for_older_rows(monkeypatch):
    # A checkpoint written before this field existed has no "total_dates"
    # key in its stored state JSON at all — must degrade to 0 (unknown),
    # never raise a KeyError, so a listing can still show it (as "N/?").
    conn = FakeCheckpointConn()
    monkeypatch.setattr(checkpoint, "get_pool", lambda: _async_return(FakePool(conn)))
    state = checkpoint.CheckpointState(
        completed_date_index=0, prev_records={}, per_ticker_records={}, per_ticker_equity_curves={},
        portfolio_equity_curve=[], running_high_water_mark=0.0, git_commit="abc1234",
    )
    await checkpoint.save_checkpoint("my-run", **PARAMS, state=state)
    # Simulate an old row by stripping total_dates back out of the stored JSON.
    import json as _json

    stored = _json.loads(conn.rows["my-run"]["state"])
    del stored["total_dates"]
    conn.rows["my-run"]["state"] = _json.dumps(stored)

    restored = await checkpoint.load_checkpoint("my-run", **PARAMS)
    assert restored.total_dates == 0


async def test_save_checkpoint_carries_in_flight_fields_through_roundtrip(monkeypatch):
    conn = FakeCheckpointConn()
    monkeypatch.setattr(checkpoint, "get_pool", lambda: _async_return(FakePool(conn)))
    state = checkpoint.CheckpointState(
        completed_date_index=1, prev_records={}, per_ticker_records={}, per_ticker_equity_curves={},
        portfolio_equity_curve=[], running_high_water_mark=0.0,
        in_flight_date_index=2, in_flight_rotated_tickers=["MSFT", "AAPL"],
        completed_tickers_this_date={"MSFT"}, git_commit="abc1234",
    )
    await checkpoint.save_checkpoint("my-run", **PARAMS, state=state)
    restored = await checkpoint.load_checkpoint("my-run", **PARAMS)
    assert restored.in_flight_date_index == 2
    assert restored.in_flight_rotated_tickers == ["MSFT", "AAPL"]
    assert restored.completed_tickers_this_date == {"MSFT"}


async def test_load_checkpoint_defaults_in_flight_fields_for_older_rows(monkeypatch):
    # A checkpoint written before ticker-granular resume existed has none of
    # these three keys in its stored state JSON — must degrade to "no
    # in-flight date" (-1/[]/set()), never raise a KeyError.
    conn = FakeCheckpointConn()
    monkeypatch.setattr(checkpoint, "get_pool", lambda: _async_return(FakePool(conn)))
    state = checkpoint.CheckpointState(
        completed_date_index=0, prev_records={}, per_ticker_records={}, per_ticker_equity_curves={},
        portfolio_equity_curve=[], running_high_water_mark=0.0, git_commit="abc1234",
    )
    await checkpoint.save_checkpoint("my-run", **PARAMS, state=state)
    import json as _json

    stored = _json.loads(conn.rows["my-run"]["state"])
    del stored["in_flight_date_index"]
    del stored["in_flight_rotated_tickers"]
    del stored["completed_tickers_this_date"]
    conn.rows["my-run"]["state"] = _json.dumps(stored)

    restored = await checkpoint.load_checkpoint("my-run", **PARAMS)
    assert restored.in_flight_date_index == -1
    assert restored.in_flight_rotated_tickers == []
    assert restored.completed_tickers_this_date == set()


async def test_save_checkpoint_carries_starting_total_equity_usd_through_roundtrip(monkeypatch):
    conn = FakeCheckpointConn()
    monkeypatch.setattr(checkpoint, "get_pool", lambda: _async_return(FakePool(conn)))
    state = checkpoint.CheckpointState(
        completed_date_index=0, prev_records={}, per_ticker_records={}, per_ticker_equity_curves={},
        portfolio_equity_curve=[], running_high_water_mark=0.0, starting_total_equity_usd=100_073.0,
        git_commit="abc1234",
    )
    await checkpoint.save_checkpoint("my-run", **PARAMS, state=state)
    restored = await checkpoint.load_checkpoint("my-run", **PARAMS)
    assert restored.starting_total_equity_usd == 100_073.0


async def test_load_checkpoint_defaults_starting_total_equity_usd_to_zero_for_older_rows(monkeypatch):
    conn = FakeCheckpointConn()
    monkeypatch.setattr(checkpoint, "get_pool", lambda: _async_return(FakePool(conn)))
    state = checkpoint.CheckpointState(
        completed_date_index=0, prev_records={}, per_ticker_records={}, per_ticker_equity_curves={},
        portfolio_equity_curve=[], running_high_water_mark=0.0, git_commit="abc1234",
    )
    await checkpoint.save_checkpoint("my-run", **PARAMS, state=state)
    import json as _json

    stored = _json.loads(conn.rows["my-run"]["state"])
    del stored["starting_total_equity_usd"]
    conn.rows["my-run"]["state"] = _json.dumps(stored)

    restored = await checkpoint.load_checkpoint("my-run", **PARAMS)
    assert restored.starting_total_equity_usd == 0.0


def test_params_includes_the_git_commit_it_was_given():
    # Simple signature-change regression check: _params() no longer computes
    # git_commit itself (see its own docstring) — it must pass through
    # whatever the caller hands it, verbatim.
    params = checkpoint._params(**PARAMS, git_commit="abc123")
    assert params["git_commit"] == "abc123"


async def test_save_checkpoint_persists_git_commit_in_both_params_and_state(monkeypatch):
    # 2026-09-11 fix: git_commit must be frozen on CheckpointState and
    # written into BOTH the params JSONB (via _params, compared by
    # load_checkpoint's resume gate) and the state JSONB payload (round-
    # tripped back onto CheckpointState.git_commit) — not recomputed fresh
    # on every save, which was the bug (see CheckpointState.git_commit's
    # own docstring).
    conn = FakeCheckpointConn()
    monkeypatch.setattr(checkpoint, "get_pool", lambda: _async_return(FakePool(conn)))
    state = checkpoint.CheckpointState(
        completed_date_index=0, prev_records={}, per_ticker_records={}, per_ticker_equity_curves={},
        portfolio_equity_curve=[], running_high_water_mark=0.0, git_commit="frozen-at-run-start",
    )
    await checkpoint.save_checkpoint("my-run", **PARAMS, state=state)

    import json as _json

    stored_row = conn.rows["my-run"]
    assert _json.loads(stored_row["params"])["git_commit"] == "frozen-at-run-start"

    restored = await checkpoint.load_checkpoint("my-run", **PARAMS)
    # load_checkpoint's own resume gate compares against a FRESH
    # current_git_commit() (fixed to "abc1234" by the autouse fixture), which
    # doesn't match "frozen-at-run-start" — so this resume correctly refuses
    # (that mismatch IS the safety check). Read the state payload directly
    # to verify the round-trip instead.
    assert restored is None
    assert _json.loads(stored_row["state"])["git_commit"] == "frozen-at-run-start"


async def test_load_checkpoint_restores_git_commit_from_a_matching_checkpoint(monkeypatch):
    # Round-trip test: when the stored git_commit DOES match current HEAD
    # (the normal resume case), load_checkpoint must actually restore the
    # frozen value onto CheckpointState.git_commit, not just pass its own
    # gate check.
    conn = FakeCheckpointConn()
    monkeypatch.setattr(checkpoint, "get_pool", lambda: _async_return(FakePool(conn)))
    monkeypatch.setattr(checkpoint, "current_git_commit", lambda: "abc1234")
    state = checkpoint.CheckpointState(
        completed_date_index=0, prev_records={}, per_ticker_records={}, per_ticker_equity_curves={},
        portfolio_equity_curve=[], running_high_water_mark=0.0, git_commit="abc1234",
    )
    await checkpoint.save_checkpoint("my-run", **PARAMS, state=state)

    restored = await checkpoint.load_checkpoint("my-run", **PARAMS)
    assert restored is not None
    assert restored.git_commit == "abc1234"


async def test_load_checkpoint_defaults_git_commit_to_none_for_older_rows(monkeypatch):
    # A checkpoint saved by code that predates this field has no
    # "git_commit" key in its stored state JSON at all — must degrade to
    # None (matching CheckpointState.git_commit's own default) rather than
    # raising a KeyError, same backward-compat convention as every other
    # field in this file.
    conn = FakeCheckpointConn()
    monkeypatch.setattr(checkpoint, "get_pool", lambda: _async_return(FakePool(conn)))
    state = checkpoint.CheckpointState(
        completed_date_index=0, prev_records={}, per_ticker_records={}, per_ticker_equity_curves={},
        portfolio_equity_curve=[], running_high_water_mark=0.0, git_commit=None,
    )
    await checkpoint.save_checkpoint("my-run", **PARAMS, state=state)
    import json as _json

    stored = _json.loads(conn.rows["my-run"]["state"])
    del stored["git_commit"]
    conn.rows["my-run"]["state"] = _json.dumps(stored)
    # git_commit=None in both params and state means the stored params dict
    # already has "git_commit": None (from the save above, before we
    # stripped the STATE key) — that still matches a fresh current_git_
    # commit() of "abc1234"? No: params comparison requires exact equality,
    # so a None-vs-"abc1234" mismatch would refuse the resume. Force params'
    # own git_commit to match what load_checkpoint will freshly compute, to
    # isolate this test to the STATE payload's backward-compat degrade only.
    stored_params = _json.loads(conn.rows["my-run"]["params"])
    stored_params["git_commit"] = "abc1234"
    conn.rows["my-run"]["params"] = _json.dumps(stored_params)

    restored = await checkpoint.load_checkpoint("my-run", **PARAMS)
    assert restored is not None
    assert restored.git_commit is None


async def test_save_checkpoint_carries_decision_id_through_roundtrip(monkeypatch):
    # resolve_outcome now finds its DB row by decision_id, not by
    # (ticker, trade_date) — so this must survive a checkpoint round-trip
    # or harness.py's post-resume resolve has nothing to resolve against.
    conn = FakeCheckpointConn()
    monkeypatch.setattr(checkpoint, "get_pool", lambda: _async_return(FakePool(conn)))

    aapl_record = _record(date(2026, 1, 8), close=101.5, decision_id=4321)
    state = checkpoint.CheckpointState(
        completed_date_index=2,
        prev_records={"AAPL": aapl_record},
        per_ticker_records={"AAPL": [aapl_record], "MSFT": []},
        per_ticker_equity_curves={"AAPL": [1000.0, 1015.0], "MSFT": [1000.0, 1000.0]},
        portfolio_equity_curve=[2000.0, 2015.0],
        running_high_water_mark=2015.0,
        git_commit="abc1234",
    )

    await checkpoint.save_checkpoint("my-run", **PARAMS, state=state)
    restored = await checkpoint.load_checkpoint("my-run", **PARAMS)

    assert restored is not None
    assert restored.prev_records["AAPL"].decision_id == 4321


async def test_record_from_json_defaults_decision_id_to_none_for_older_rows():
    # A checkpoint saved before decision_id existed on BacktestRecord has no
    # "decision_id" key in its stored record payload at all — must degrade
    # to None (matching BacktestRecord's own default), never raise a
    # KeyError. harness.py already treats decision_id=None as "skip this
    # one resolve, don't crash" for exactly this case.
    payload = {
        "trade_date": "2026-01-08",
        "decision": _decision().model_dump(mode="json"),
        "close": 101.5,
        "adjustment_note": "1% risk cap",
        "trailing_high_water": 95.0,
        "trailing_entry_price": 91.0,
    }
    record = checkpoint._record_from_json(payload)
    assert record.decision_id is None


async def test_list_checkpoints_returns_progress_for_every_unfinished_run(monkeypatch):
    conn = FakeCheckpointConn()
    monkeypatch.setattr(checkpoint, "get_pool", lambda: _async_return(FakePool(conn)))

    state_a = checkpoint.CheckpointState(
        completed_date_index=4, prev_records={}, per_ticker_records={}, per_ticker_equity_curves={},
        portfolio_equity_curve=[1000.0, 1010.0], running_high_water_mark=1010.0, total_dates=15,
    )
    await checkpoint.save_checkpoint("run-a:AAPL,MSFT", **PARAMS, state=state_a)
    state_b = checkpoint.CheckpointState(
        completed_date_index=1, prev_records={}, per_ticker_records={}, per_ticker_equity_curves={},
        portfolio_equity_curve=[], running_high_water_mark=0.0, total_dates=10,
    )
    await checkpoint.save_checkpoint("run-b:AAPL,MSFT", **PARAMS, state=state_b)

    progress = await checkpoint.list_checkpoints()

    assert {p.label for p in progress} == {"run-a:AAPL,MSFT", "run-b:AAPL,MSFT"}
    # Most-recently-saved first — run-b saved second.
    assert progress[0].label == "run-b:AAPL,MSFT"
    by_label = {p.label: p for p in progress}
    assert by_label["run-a:AAPL,MSFT"].completed_date_index == 4
    assert by_label["run-a:AAPL,MSFT"].total_dates == 15
    assert by_label["run-a:AAPL,MSFT"].latest_portfolio_equity == 1010.0
    assert by_label["run-a:AAPL,MSFT"].tickers == sorted(PARAMS["tickers"])
    assert by_label["run-b:AAPL,MSFT"].latest_portfolio_equity is None  # empty equity curve


def test_render_checkpoints_empty_list():
    assert "No unfinished" in checkpoint.render_checkpoints([])


def test_render_checkpoints_formats_progress_fraction_and_equity():
    progress = [
        checkpoint.CheckpointProgress(
            label="baseline-domestic-20260812:005930.KS",
            tickers=["005930.KS"],
            completed_date_index=6,
            total_dates=15,
            latest_portfolio_equity=31234567.0,
            updated_at=datetime(2026, 8, 12, 11, 15),
        ),
        checkpoint.CheckpointProgress(
            label="baseline-overseas-20260812:AAPL",
            tickers=["AAPL"],
            completed_date_index=2,
            total_dates=0,  # unknown — older checkpoint
            latest_portfolio_equity=None,
            updated_at=datetime(2026, 8, 12, 10, 0),
        ),
    ]
    rendered = checkpoint.render_checkpoints(progress)
    assert "7/15" in rendered  # completed_date_index is 0-based, display is 1-based
    assert "31,234,567" in rendered
    assert "3/?" in rendered  # unknown total_dates
    assert "n/a" in rendered  # no equity yet
