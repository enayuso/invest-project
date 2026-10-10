"""app.memory.store — asyncpg pool/connection faked, no real Postgres, no
real embedding model (F-stage)."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime

import asyncpg
import pytest

from app.memory import store


class FakeConn:
    def __init__(self, fetchrow_result=None, fetch_result=None):
        self.fetchrow_result = fetchrow_result
        self.fetch_result = fetch_result if fetch_result is not None else []
        self.queries: list[tuple] = []

    async def fetchrow(self, query, *args):
        self.queries.append((query, args))
        return self.fetchrow_result

    async def fetch(self, query, *args):
        self.queries.append((query, args))
        return self.fetch_result

    async def execute(self, query, *args):
        self.queries.append((query, args))


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


@pytest.fixture
def fake_embed(monkeypatch):
    async def _embed(text):
        return [0.1, 0.2, 0.3]

    monkeypatch.setattr(store, "embed_async", _embed)


# ---------------------------------------------------------------------------
# store_pending
# ---------------------------------------------------------------------------


async def test_store_pending_returns_new_row_id(monkeypatch, fake_embed):
    conn = FakeConn(fetchrow_result={"id": 7})
    monkeypatch.setattr(store, "get_pool", lambda: _async_return(FakePool(conn)))

    result = await store.store_pending("AAPL", date(2026, 7, 16), "situation", "Buy", "decision text")

    assert result == 7
    query, args = conn.queries[0]
    assert args[0] == "AAPL"
    assert args[3] == [0.1, 0.2, 0.3]


async def test_store_pending_serializes_transcript_as_json(monkeypatch, fake_embed):
    conn = FakeConn(fetchrow_result={"id": 1})
    monkeypatch.setattr(store, "get_pool", lambda: _async_return(FakePool(conn)))

    await store.store_pending(
        "AAPL", date(2026, 7, 16), "situation", "Buy", "decision text", transcript={"reports": {"a": "b"}}
    )

    _, args = conn.queries[0]
    assert json.loads(args[6]) == {"reports": {"a": "b"}}
    # Stage 1 (2026-08-13) added `confidence`/`held_at_decision`, later tasks
    # added `card_reason_ko`/`card_reason_en`, `total_tokens`,
    # `decision_trigger`, and `analysts_failed` — all appended to the end of
    # the parameter list, shifting every negative index below.
    assert args[-13] == "default"  # user_id stamp (agent-architecture-roadmap stage 4)
    assert args[-12] is False  # is_backtest defaults to False when not given
    assert args[-11] is None  # trader_action defaults to None when not given
    assert args[-10] == "paper"  # account_mode stamp defaults to settings.kis_account_mode
    # git_commit/llm_model (2026-08-12) — see the dedicated version-tagging
    # tests below for their own coverage; here just confirms the two extra
    # positional slots exist in the expected place.
    assert args[-8] == store.settings.llm_model
    assert args[-7] is None  # confidence defaults to None when not given
    assert args[-6] is None  # held_at_decision defaults to None when not given
    assert args[-5] is None  # card_reason_ko defaults to None when not given
    assert args[-4] is None  # card_reason_en defaults to None when not given
    assert args[-3] is None  # total_tokens defaults to None when not given
    assert args[-2] == "auto"  # decision_trigger defaults to "auto" when not given
    assert args[-1] is None  # analysts_failed defaults to None when not given


async def test_store_pending_persists_the_trader_action(monkeypatch, fake_embed):
    conn = FakeConn(fetchrow_result={"id": 2})
    monkeypatch.setattr(store, "get_pool", lambda: _async_return(FakePool(conn)))

    await store.store_pending(
        "AAPL", date(2026, 7, 16), "situation", "Overweight", "decision text", trader_action="Buy",
    )

    _, args = conn.queries[0]
    assert args[-11] == "Buy"  # recorded separately from the final `rating` above


async def test_store_pending_stamps_current_account_mode(monkeypatch, fake_embed):
    conn = FakeConn(fetchrow_result={"id": 3})
    monkeypatch.setattr(store, "get_pool", lambda: _async_return(FakePool(conn)))
    monkeypatch.setattr(store.settings, "kis_account_mode", "real")

    await store.store_pending("AAPL", date(2026, 7, 16), "situation", "Buy", "decision text")

    _, args = conn.queries[0]
    assert args[-10] == "real"


async def test_store_pending_persists_confidence_and_held_at_decision(monkeypatch, fake_embed):
    # Stage 1 (2026-08-13) — F5: the value that actually scaled real capital
    # via app.execution.sizing.confidence_size_modifier, and the pre-decision
    # position state used to score Hold in the correct direction.
    conn = FakeConn(fetchrow_result={"id": 4})
    monkeypatch.setattr(store, "get_pool", lambda: _async_return(FakePool(conn)))

    await store.store_pending(
        "AAPL", date(2026, 7, 16), "situation", "Buy", "decision text",
        confidence=0.82, held_at_decision=True,
    )

    _, args = conn.queries[0]
    # confidence, held_at_decision sit before card_reason_ko/card_reason_en/
    # total_tokens/decision_trigger/analysts_failed in the positional
    # insert — see store_pending's INSERT statement.
    assert args[-7] == 0.82
    assert args[-6] is True


async def test_store_pending_persists_card_reason_fields(monkeypatch, fake_embed):
    conn = FakeConn(fetchrow_result={"id": 5})
    monkeypatch.setattr(store, "get_pool", lambda: _async_return(FakePool(conn)))

    await store.store_pending(
        "AAPL", date(2026, 7, 16), "situation", "Buy", "decision text",
        card_reason_ko="한글 이유", card_reason_en="english reason",
    )

    _, args = conn.queries[0]
    assert args[-5] == "한글 이유"
    assert args[-4] == "english reason"


async def test_store_pending_persists_total_tokens(monkeypatch, fake_embed):
    conn = FakeConn(fetchrow_result={"id": 6})
    monkeypatch.setattr(store, "get_pool", lambda: _async_return(FakePool(conn)))

    await store.store_pending(
        "AAPL", date(2026, 7, 16), "situation", "Buy", "decision text", total_tokens=12345,
    )

    _, args = conn.queries[0]
    assert args[-3] == 12345


async def test_store_pending_persists_manual_decision_trigger(monkeypatch, fake_embed):
    conn = FakeConn(fetchrow_result={"id": 7})
    monkeypatch.setattr(store, "get_pool", lambda: _async_return(FakePool(conn)))

    await store.store_pending(
        "AAPL", date(2026, 7, 16), "situation", "Buy", "decision text", decision_trigger="manual",
    )

    query, args = conn.queries[0]
    assert "decision_trigger" in query
    assert args[-2] == "manual"


async def test_store_pending_persists_analysts_failed(monkeypatch, fake_embed):
    conn = FakeConn(fetchrow_result={"id": 8})
    monkeypatch.setattr(store, "get_pool", lambda: _async_return(FakePool(conn)))

    await store.store_pending(
        "AAPL", date(2026, 7, 16), "situation", "Buy", "decision text", analysts_failed=1,
    )

    query, args = conn.queries[0]
    assert "analysts_failed" in query
    assert args[-1] == 1


async def test_store_pending_degrades_to_none_on_db_failure(monkeypatch, fake_embed):
    async def broken_get_pool():
        raise OSError("connection refused")

    monkeypatch.setattr(store, "get_pool", broken_get_pool)
    result = await store.store_pending("AAPL", date(2026, 7, 16), "situation", "Buy", "decision text")
    assert result is None


async def test_store_pending_degrades_on_postgres_error(monkeypatch, fake_embed):
    async def broken_get_pool():
        raise asyncpg.PostgresError("db error")

    monkeypatch.setattr(store, "get_pool", broken_get_pool)
    result = await store.store_pending("AAPL", date(2026, 7, 16), "situation", "Buy", "decision text")
    assert result is None


async def test_store_pending_degrades_on_interface_error(monkeypatch, fake_embed):
    # asyncpg.InterfaceError (pool closing/contended) is NOT a PostgresError
    # subclass — the old (OSError, PostgresError) except clause let this
    # crash the caller instead of degrading quietly like every other failure.
    async def broken_get_pool():
        raise asyncpg.InterfaceError("pool is closing")

    monkeypatch.setattr(store, "get_pool", broken_get_pool)
    result = await store.store_pending("AAPL", date(2026, 7, 16), "situation", "Buy", "decision text")
    assert result is None


# ---------------------------------------------------------------------------
# store_pending — decision version tagging (git_commit/llm_model, 2026-08-12)
#
# Without these, a model swap or a prompt change left no trace on any
# individual decision — only whole backtest runs were tagged
# (backtest_runs.git_commit). Both are derived automatically inside
# store_pending; no caller passes either. See app.memory.db's column
# comments and store_pending's own docstring for the fuller rationale.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_git_commit_cache(monkeypatch):
    """The cache is process-lifetime by design (see `_current_git_commit`'s
    own comment — re-running `git rev-parse` on every decision would be pure
    overhead), which is exactly what makes it a cross-test hazard: whichever
    test resolves it first would otherwise pin the value (or the failure)
    for every test that runs after it in the same pytest process. Reset
    before AND after so this file never leaks state either direction.

    `settings.git_commit` is forced to None as well: it now takes precedence
    over the subprocess (2026-08-13, so the deployed container can tag
    decisions at all), which would otherwise make the subprocess-path tests
    below depend on whether GIT_COMMIT happens to be set in the environment
    running pytest. The env-precedence behavior gets its own explicit tests.
    settings.git_commit이 subprocess보다 우선하므로, 환경변수 유무가 테스트
    결과를 바꾸지 않도록 여기서 명시적으로 비움."""
    monkeypatch.setattr(store.settings, "git_commit", None)
    store._cached_git_commit = None
    store._git_commit_resolved = False
    yield
    store._cached_git_commit = None
    store._git_commit_resolved = False


async def test_store_pending_records_the_current_llm_model(monkeypatch, fake_embed):
    conn = FakeConn(fetchrow_result={"id": 8})
    monkeypatch.setattr(store, "get_pool", lambda: _async_return(FakePool(conn)))
    monkeypatch.setattr(store.settings, "llm_model", "nvidia/Qwen3.6-35B-A3B-NVFP4")
    monkeypatch.setattr(store, "_current_git_commit", lambda: "abc1234")

    await store.store_pending("AAPL", date(2026, 7, 16), "situation", "Buy", "decision text")

    _, args = conn.queries[0]
    assert args[-8] == "nvidia/Qwen3.6-35B-A3B-NVFP4"
    assert args[-9] == "abc1234"


async def test_current_git_commit_caches_across_calls(monkeypatch):
    calls = []

    class FakeCompletedProcess:
        stdout = "deadbee\n"

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return FakeCompletedProcess()

    monkeypatch.setattr(store.subprocess, "run", fake_run)

    first = store._current_git_commit()
    second = store._current_git_commit()

    assert first == "deadbee"
    assert second == "deadbee"
    assert len(calls) == 1  # second call served from the cache, no re-exec


async def test_current_git_commit_caches_a_failure_as_none_without_retrying(monkeypatch):
    calls = []

    def failing_run(cmd, **kwargs):
        calls.append(cmd)
        raise FileNotFoundError("git not on PATH")

    monkeypatch.setattr(store.subprocess, "run", failing_run)

    assert store._current_git_commit() is None
    assert store._current_git_commit() is None
    assert len(calls) == 1  # the failure itself is cached, not retried


# ---------------------------------------------------------------------------
# get_transcript
# ---------------------------------------------------------------------------


async def test_get_transcript_returns_none_when_missing(monkeypatch):
    conn = FakeConn(fetchrow_result=None)
    monkeypatch.setattr(store, "get_pool", lambda: _async_return(FakePool(conn)))
    assert await store.get_transcript(999) is None


async def test_get_transcript_parses_full_transcript_json(monkeypatch):
    row = {
        "ticker": "AAPL", "trade_date": date(2026, 7, 16), "rating": "Buy",
        "decision_text": "decision", "full_transcript": json.dumps({"reports": {}}),
    }
    conn = FakeConn(fetchrow_result=row)
    monkeypatch.setattr(store, "get_pool", lambda: _async_return(FakePool(conn)))
    result = await store.get_transcript(1)
    assert result["ticker"] == "AAPL"
    assert result["trade_date"] == "2026-07-16"
    assert result["transcript"] == {"reports": {}}


async def test_get_transcript_none_transcript_when_column_empty(monkeypatch):
    row = {
        "ticker": "AAPL", "trade_date": date(2026, 7, 16), "rating": "Buy",
        "decision_text": "decision", "full_transcript": None,
    }
    conn = FakeConn(fetchrow_result=row)
    monkeypatch.setattr(store, "get_pool", lambda: _async_return(FakePool(conn)))
    result = await store.get_transcript(1)
    assert result["transcript"] is None


# ---------------------------------------------------------------------------
# resolve_outcome / flag_bad_entry / append_execution_note — degrade quietly
# ---------------------------------------------------------------------------


async def test_resolve_outcome_degrades_quietly_on_failure(monkeypatch):
    async def broken_get_pool():
        raise OSError("down")

    monkeypatch.setattr(store, "get_pool", broken_get_pool)
    # Must not raise.
    await store.resolve_outcome(
        42, "AAPL", date(2026, 7, 16), 0.05, 0.02, 3, "reflection text", "SPY", resolved_on=date(2026, 7, 21),
    )


async def test_resolve_outcome_executes_update_with_correct_args(monkeypatch):
    conn = FakeConn()
    monkeypatch.setattr(store, "get_pool", lambda: _async_return(FakePool(conn)))
    await store.resolve_outcome(
        42, "AAPL", date(2026, 7, 16), 0.05, 0.02, 3, "reflection text", "SPY", resolved_on=date(2026, 7, 21),
    )
    _, args = conn.queries[0]
    assert args == (42, "AAPL", date(2026, 7, 16), 0.05, 0.02, 3, "reflection text", "SPY", date(2026, 7, 21))


async def test_resolve_outcome_requires_resolved_on(monkeypatch):
    # 2026-09-01 PIT fix: no default — a caller that forgets it must fail
    # loudly, not silently fall back to a guessed or wall-clock date.
    with pytest.raises(TypeError):
        await store.resolve_outcome(42, "AAPL", date(2026, 7, 16), 0.05, 0.02, 3, "reflection text", "SPY")


async def test_resolve_outcome_scopes_the_update_by_id_not_natural_key(monkeypatch):
    """More than one pending row can share (ticker, trade_date) — e.g. a
    same-day Sell and Buy on the same ticker (id=949/950160.KS). The UPDATE
    must be scoped by `id`, not just (ticker, trade_date), or resolving one
    row's outcome would silently touch every row sharing that natural key."""
    conn = FakeConn()
    monkeypatch.setattr(store, "get_pool", lambda: _async_return(FakePool(conn)))
    await store.resolve_outcome(
        42, "AAPL", date(2026, 7, 16), 0.05, 0.02, 3, "reflection text", "SPY", resolved_on=date(2026, 7, 21),
    )
    query, args = conn.queries[0]
    assert "WHERE id = $1" in query
    assert args[0] == 42


async def test_resolve_outcome_records_which_benchmark_was_actually_used(monkeypatch):
    # 2026-08-12: the benchmark_ticker column existed since the original
    # schema but nothing ever wrote it, so it silently claimed its 'SPY'
    # default for every row regardless of what was actually used — see
    # project memory: benchmark-mismatch-fixed.
    conn = FakeConn()
    monkeypatch.setattr(store, "get_pool", lambda: _async_return(FakePool(conn)))
    await store.resolve_outcome(
        42, "005930.KS", date(2026, 7, 16), 0.05, 0.02, 3, "reflection text", "^KS11", resolved_on=date(2026, 7, 21),
    )
    query, args = conn.queries[0]
    assert "benchmark_ticker" in query
    assert args == (42, "005930.KS", date(2026, 7, 16), 0.05, 0.02, 3, "reflection text", "^KS11", date(2026, 7, 21))


async def test_resolve_outcome_writes_resolved_on(monkeypatch):
    conn = FakeConn()
    monkeypatch.setattr(store, "get_pool", lambda: _async_return(FakePool(conn)))
    await store.resolve_outcome(
        42, "AAPL", date(2026, 7, 16), 0.05, 0.02, 3, "reflection text", "SPY", resolved_on=date(2026, 7, 21),
    )
    query, args = conn.queries[0]
    assert "resolved_on = $9" in query
    assert args[8] == date(2026, 7, 21)


class _FakeSameDaySiblingConn:
    """Minimal in-memory simulation of the id-scoped UPDATE's real-world
    filtering (the other FakeConn above only RECORDS the query/args, it
    doesn't apply the WHERE clause) — used to prove two pending rows
    sharing the same (ticker, trade_date) don't cross-contaminate when
    resolved by id, the exact id=949/950160.KS shape documented in
    `get_same_day_sibling_context`'s own docstring."""

    def __init__(self, rows: dict[int, dict]):
        self.rows = rows

    async def execute(self, query, *args):
        assert "WHERE id = $1" in query
        decision_id = args[0]
        row = self.rows.get(decision_id)
        if row is None or row["status"] != "pending":
            return
        row["reflection"] = args[6]
        row["status"] = "resolved"


async def test_resolve_outcome_does_not_touch_sibling_row_sharing_ticker_and_trade_date(monkeypatch):
    trade_date = date(2026, 8, 21)
    rows = {
        949: {"ticker": "950160.KS", "trade_date": trade_date, "status": "pending", "reflection": None},
        950: {"ticker": "950160.KS", "trade_date": trade_date, "status": "pending", "reflection": None},
    }
    conn = _FakeSameDaySiblingConn(rows)
    monkeypatch.setattr(store, "get_pool", lambda: _async_return(FakePool(conn)))

    await store.resolve_outcome(
        949, "950160.KS", trade_date, 0.05, 0.02, 3, "sell reflection", "^KS11", resolved_on=trade_date,
    )

    assert rows[949]["status"] == "resolved"
    assert rows[949]["reflection"] == "sell reflection"
    # The sibling row (same ticker, same trade_date, different id) must be
    # untouched — this is exactly what a WHERE ticker=... AND trade_date=...
    # (no id) UPDATE would have gotten wrong.
    assert rows[950]["status"] == "pending"
    assert rows[950]["reflection"] is None


# ---------------------------------------------------------------------------
# set_horizon_outcome (Stage 1, F6, 2026-08-13)
# ---------------------------------------------------------------------------


async def test_set_horizon_outcome_executes_update_with_correct_args(monkeypatch):
    conn = FakeConn()
    monkeypatch.setattr(store, "get_pool", lambda: _async_return(FakePool(conn)))
    await store.set_horizon_outcome(42, 0.05, 0.02, 10, date(2026, 7, 30))
    query, args = conn.queries[0]
    assert "WHERE id = $1" in query
    assert "status" not in query  # must never touch status — only fills horizon_* columns
    assert args == (42, 0.05, 0.02, 10, date(2026, 7, 30))


async def test_set_horizon_outcome_refuses_a_nan_raw_return(monkeypatch):
    conn = FakeConn()
    monkeypatch.setattr(store, "get_pool", lambda: _async_return(FakePool(conn)))
    await store.set_horizon_outcome(42, float("nan"), 0.02, 10, date(2026, 7, 30))
    assert conn.queries == []  # UPDATE must never run


async def test_set_horizon_outcome_refuses_a_nan_alpha_return(monkeypatch):
    conn = FakeConn()
    monkeypatch.setattr(store, "get_pool", lambda: _async_return(FakePool(conn)))
    await store.set_horizon_outcome(42, 0.05, float("nan"), 10, date(2026, 7, 30))
    assert conn.queries == []


async def test_set_horizon_outcome_degrades_quietly_on_db_failure(monkeypatch):
    async def broken_get_pool():
        raise OSError("down")

    monkeypatch.setattr(store, "get_pool", broken_get_pool)
    # Must not raise.
    await store.set_horizon_outcome(42, 0.05, 0.02, 10, date(2026, 7, 30))


async def test_flag_bad_entry_degrades_quietly_on_failure(monkeypatch):
    async def broken_get_pool():
        raise OSError("down")

    monkeypatch.setattr(store, "get_pool", broken_get_pool)
    await store.flag_bad_entry(1, "hallucinated ticker identity")


async def test_append_execution_note_executes_with_wrapped_note(monkeypatch):
    conn = FakeConn()
    monkeypatch.setattr(store, "get_pool", lambda: _async_return(FakePool(conn)))
    await store.append_execution_note(1, "risk cap reduced size")
    _, args = conn.queries[0]
    assert "[EXECUTION NOTE: risk cap reduced size]" in args[1]


# ---------------------------------------------------------------------------
# get_similar_context
# ---------------------------------------------------------------------------


async def test_get_similar_context_returns_empty_string_when_no_rows(monkeypatch, fake_embed):
    conn = FakeConn(fetch_result=[])
    monkeypatch.setattr(store, "get_pool", lambda: _async_return(FakePool(conn)))
    result = await store.get_similar_context("AAPL", "situation")
    assert result == ""


async def test_get_similar_context_formats_same_and_cross_ticker_rows(monkeypatch, fake_embed):
    same_row = {
        "trade_date": date(2026, 7, 1), "rating": "Buy", "decision_text": "bought on momentum",
        "reflection": "worked out well", "raw_return": 0.05, "alpha_return": 0.02,
    }
    cross_row = {
        "ticker": "MSFT", "trade_date": date(2026, 6, 1), "rating": "Sell",
        "reflection": "sold too early", "alpha_return": -0.01,
    }

    class SequencedConn(FakeConn):
        def __init__(self):
            super().__init__()
            self._calls = 0

        async def fetch(self, query, *args):
            self._calls += 1
            return [same_row] if self._calls == 1 else [cross_row]

    monkeypatch.setattr(store, "get_pool", lambda: _async_return(FakePool(SequencedConn())))
    result = await store.get_similar_context("AAPL", "situation")
    assert "Past analyses of AAPL with a similar setup" in result
    assert "bought on momentum" in result
    assert "Similar situations from other tickers" in result
    assert "MSFT" in result


async def test_get_similar_context_degrades_to_empty_string_on_db_failure(monkeypatch, fake_embed):
    async def broken_get_pool():
        raise OSError("down")

    monkeypatch.setattr(store, "get_pool", broken_get_pool)
    result = await store.get_similar_context("AAPL", "situation")
    assert result == ""


async def test_get_similar_context_prioritizes_same_market_cross_rows(monkeypatch, fake_embed):
    # Domestic/overseas separation (2026-07-27): when same-market matches
    # already fill the n_cross quota, the other-market fallback query must
    # not even run.
    monkeypatch.setattr(store.settings, "memory_n_cross", 2)
    same_market_row = {
        "ticker": "MSFT", "trade_date": date(2026, 6, 1), "rating": "Sell",
        "reflection": "sold too early", "alpha_return": -0.01,
    }

    class Conn(FakeConn):
        async def fetch(self, query, *args):
            self.queries.append((query, args))
            if "ticker = $1 AND status" in query:
                return []
            if "NOT LIKE" in query:  # AAPL is overseas -> same-market clause is "NOT LIKE '%.KS'"
                return [same_market_row, same_market_row]
            raise AssertionError("other-market fallback must not run once same-market rows fill the quota")

    monkeypatch.setattr(store, "get_pool", lambda: _async_return(FakePool(Conn())))
    result = await store.get_similar_context("AAPL", "situation")
    assert "Similar situations from other tickers (same market)" in result
    assert "different market" not in result


async def test_get_similar_context_fills_remaining_slots_from_other_market(monkeypatch, fake_embed):
    # Domestic/overseas separation (2026-07-27): "같은 시장 우선 + 보조 허용" —
    # a cross-market match is only pulled in to fill slots same-market
    # matches didn't cover, and it's labeled distinctly.
    monkeypatch.setattr(store.settings, "memory_n_cross", 3)
    same_market_row = {
        "ticker": "MSFT", "trade_date": date(2026, 6, 1), "rating": "Sell",
        "reflection": "sold too early", "alpha_return": -0.01,
    }
    other_market_row = {
        "ticker": "005930.KS", "trade_date": date(2026, 6, 2), "rating": "Buy",
        "reflection": "worked out", "alpha_return": 0.03,
    }

    class Conn(FakeConn):
        async def fetch(self, query, *args):
            self.queries.append((query, args))
            if "ticker = $1 AND status" in query:
                return []
            if "NOT LIKE" in query:
                return [same_market_row]  # only 1, short of n_cross=3
            return [other_market_row]

    conn = Conn()
    monkeypatch.setattr(store, "get_pool", lambda: _async_return(FakePool(conn)))
    result = await store.get_similar_context("AAPL", "situation")
    assert "Similar situations from other tickers (same market)" in result
    assert "MSFT" in result
    assert "different market" in result
    assert "005930.KS" in result
    # the fallback query asked for exactly the remaining slots (3 - 1 = 2)
    assert conn.queries[2][1][2] == 2


async def test_get_similar_context_domestic_ticker_uses_ks_suffix_as_same_market(monkeypatch, fake_embed):
    monkeypatch.setattr(store.settings, "memory_n_cross", 1)
    same_market_row = {
        "ticker": "000660.KS", "trade_date": date(2026, 6, 1), "rating": "Buy",
        "reflection": "similar setup", "alpha_return": 0.01,
    }

    class Conn(FakeConn):
        async def fetch(self, query, *args):
            self.queries.append((query, args))
            if "ticker = $1 AND status" in query:
                return []
            if "NOT LIKE" in query:
                raise AssertionError("005930.KS is domestic — same-market clause should be 'LIKE', not 'NOT LIKE'")
            return [same_market_row]

    monkeypatch.setattr(store, "get_pool", lambda: _async_return(FakePool(Conn())))
    result = await store.get_similar_context("005930.KS", "situation")
    assert "Similar situations from other tickers (same market)" in result
    assert "000660.KS" in result


async def test_get_similar_context_passes_as_of_bound(monkeypatch, fake_embed):
    conn = FakeConn(fetch_result=[])
    monkeypatch.setattr(store, "get_pool", lambda: _async_return(FakePool(conn)))
    await store.get_similar_context("AAPL", "situation", as_of=date(2026, 1, 1))
    query, args = conn.queries[0]
    assert "user_id = $5" in query
    assert args[3] == date(2026, 1, 1)
    # $7 (the trailing similarity-distance-cutoff param, Fix 2) shifts these
    # two down from args[-2]/args[-1].
    assert args[-3] == "default"
    assert args[-2] == "paper"
    assert args[-1] == store.settings.memory_similarity_max_distance


async def test_get_similar_context_scopes_every_query_by_account_mode(monkeypatch, fake_embed):
    # Fix 1: once real trading starts, a paper-account query without this
    # filter would show the real-account Portfolio Manager's paper-account
    # outcomes as its own track record, or vice versa.
    conn = FakeConn(fetch_result=[])
    monkeypatch.setattr(store, "get_pool", lambda: _async_return(FakePool(conn)))
    monkeypatch.setattr(store.settings, "kis_account_mode", "real")

    await store.get_similar_context("AAPL", "situation")

    assert len(conn.queries) == 3
    for query, args in conn.queries:
        assert "account_mode = $6" in query
        # args[-1] is now the trailing similarity-distance-cutoff param
        # ($7, Fix 2) — account_mode shifted down to args[-2].
        assert args[-2] == "real"


async def test_get_similar_context_applies_similarity_distance_cutoff(monkeypatch, fake_embed):
    # Fix 2: without a cutoff, the top-N nearest neighbors were returned with
    # no floor on how similar "most similar" actually was — a sparse-history
    # ticker's "similar setup" match could be barely more related than a
    # random pair while still presented to the LLM as one. All three queries
    # must filter on the distance operator and bind settings.memory_similar
    # ity_max_distance as the trailing parameter.
    monkeypatch.setattr(store.settings, "memory_similarity_max_distance", 0.42)
    conn = FakeConn(fetch_result=[])
    monkeypatch.setattr(store, "get_pool", lambda: _async_return(FakePool(conn)))

    await store.get_similar_context("AAPL", "situation")

    assert len(conn.queries) == 3
    for query, args in conn.queries:
        assert "situation_embedding <=> $2 <= $7" in query
        assert args[-1] == 0.42


async def test_get_similar_context_excludes_null_and_nan_returns(monkeypatch, fake_embed):
    # Fix 3: a NULL/NaN raw_return or alpha_return on a 'resolved' row must
    # never reach the f-string formatting in _format_same_ticker/
    # _format_cross_ticker (which would raise TypeError, or format literal
    # "nan%" into the LLM prompt) — excluded at the SQL level instead.
    conn = FakeConn(fetch_result=[])
    monkeypatch.setattr(store, "get_pool", lambda: _async_return(FakePool(conn)))

    await store.get_similar_context("AAPL", "situation")

    same_query = conn.queries[0][0]
    assert "raw_return IS NOT NULL AND raw_return <> 'NaN'::float8" in same_query
    assert "alpha_return IS NOT NULL AND alpha_return <> 'NaN'::float8" in same_query
    for query, _args in conn.queries[1:]:
        assert "alpha_return IS NOT NULL AND alpha_return <> 'NaN'::float8" in query


async def test_get_similar_context_live_excludes_backtest_rows_in_every_query(monkeypatch, fake_embed):
    # 2026-08-04: a backtest run overlapping with live could otherwise leak
    # its fictional, historical-replay decisions into the live Portfolio
    # Manager's "past lessons" retrieval — see this function's own docstring.
    # fetch_result=[] for every call keeps all 3 queries (same-ticker, same-
    # market cross, other-market cross fallback) running, since an empty
    # same-market result never fills the n_cross quota.
    conn = FakeConn(fetch_result=[])
    monkeypatch.setattr(store, "get_pool", lambda: _async_return(FakePool(conn)))
    await store.get_similar_context("AAPL", "situation")  # as_of=None -> live
    assert len(conn.queries) == 3
    for query, _ in conn.queries:
        assert "NOT is_backtest" in query


async def test_get_similar_context_backtest_does_not_exclude_backtest_rows(monkeypatch, fake_embed):
    # A replay is expected to learn from its own earlier-round decisions in
    # the same run — as_of's temporal bound already keeps this sound.
    conn = FakeConn(fetch_result=[])
    monkeypatch.setattr(store, "get_pool", lambda: _async_return(FakePool(conn)))
    await store.get_similar_context("AAPL", "situation", as_of=date(2026, 1, 1))
    assert len(conn.queries) == 3
    for query, _ in conn.queries:
        assert "NOT is_backtest" not in query


# ---------------------------------------------------------------------------
# 2026-09-01 point-in-time fix — trade_date <= as_of alone let a replay of
# date T retrieve a row graded (status='resolved', alpha_return/reflection
# populated) AFTER T in simulated time, since grading happens
# `holding_days` (5 to 112+ calendar days live, exactly 1+ in a replay)
# after trade_date. resolved_on < as_of is the second bound this fix adds.
# ---------------------------------------------------------------------------


async def test_get_similar_context_backtest_bounds_on_resolution_date(monkeypatch, fake_embed):
    conn = FakeConn(fetch_result=[])
    monkeypatch.setattr(store, "get_pool", lambda: _async_return(FakePool(conn)))
    await store.get_similar_context("AAPL", "situation", as_of=date(2026, 1, 1))
    assert len(conn.queries) == 3
    for query, args in conn.queries:
        assert "resolved_on IS NOT NULL AND resolved_on < $4" in query
        # No new bind parameter — reuses $4 (as_of), already bound.
        assert args[3] == date(2026, 1, 1)


async def test_get_similar_context_live_has_no_resolution_bound(monkeypatch, fake_embed):
    conn = FakeConn(fetch_result=[])
    monkeypatch.setattr(store, "get_pool", lambda: _async_return(FakePool(conn)))
    await store.get_similar_context("AAPL", "situation")  # as_of=None -> live
    assert len(conn.queries) == 3
    for query, _ in conn.queries:
        assert "resolved_on" not in query


async def test_get_similar_context_live_query_text_unchanged_by_pit_fix(monkeypatch, fake_embed):
    # Proves live is untouched at the parameter-binding level too, not just
    # "doesn't mention resolved_on" — every existing live-mode assertion
    # (as_of bound position, account_mode scoping, similarity cutoff,
    # NaN exclusion, is_backtest exclusion) already re-runs unmodified
    # elsewhere in this file; this pins that the trade_date bound and the
    # account/user scoping are still textually adjacent (no clause spliced
    # between them when as_of is None).
    conn = FakeConn(fetch_result=[])
    monkeypatch.setattr(store, "get_pool", lambda: _async_return(FakePool(conn)))
    await store.get_similar_context("AAPL", "situation")  # as_of=None -> live
    for query, _ in conn.queries:
        trade_date_idx = query.index("trade_date <= $4)")
        user_id_idx = query.index("AND user_id = $5")
        between = query[trade_date_idx + len("trade_date <= $4)") : user_id_idx]
        assert between.strip() == ""


# ---------------------------------------------------------------------------
# get_same_day_sibling_context
# ---------------------------------------------------------------------------


def _sibling_row(**overrides):
    row = {
        "created_at": datetime(2026, 8, 21, 0, 21, 53, tzinfo=UTC),
        "rating": "Sell",
        "confidence": 0.90,
        "action_label": None,
        "decision_trigger": "auto",
        "card_reason_ko": None,
        "card_reason_en": "quick ratio 0%, going-concern risk flagged",
        "decision_text": "**Rating**: Sell\n\nlong decision text " + "x" * 500,
    }
    row.update(overrides)
    return row


async def test_get_same_day_sibling_context_surfaces_a_pending_row(monkeypatch):
    # id=949 (950160.KS) shape — a 'pending' row (no outcome graded yet)
    # must still surface here, unlike get_similar_context's resolved-only
    # filter. This function reads no `status` column at all.
    conn = FakeConn(fetch_result=[_sibling_row()])
    monkeypatch.setattr(store, "get_pool", lambda: _async_return(FakePool(conn)))

    result = await store.get_same_day_sibling_context("950160.KS", date(2026, 8, 21))

    assert "status" not in conn.queries[0][0]
    assert "Sell" in result
    assert "quick ratio 0%, going-concern risk flagged" in result
    assert "execution outcome not recorded" in result  # action_label was None


async def test_get_same_day_sibling_context_uses_action_label_when_present(monkeypatch):
    conn = FakeConn(fetch_result=[_sibling_row(action_label="Buy")])
    monkeypatch.setattr(store, "get_pool", lambda: _async_return(FakePool(conn)))

    result = await store.get_same_day_sibling_context("AAPL", date(2026, 8, 21))

    assert "execution outcome not recorded" not in result
    assert "Buy" in result


async def test_get_same_day_sibling_context_confidence_none_does_not_raise(monkeypatch):
    conn = FakeConn(fetch_result=[_sibling_row(confidence=None)])
    monkeypatch.setattr(store, "get_pool", lambda: _async_return(FakePool(conn)))

    result = await store.get_same_day_sibling_context("AAPL", date(2026, 8, 21))

    assert "confidence n/a" in result


async def test_get_same_day_sibling_context_falls_back_to_decision_text(monkeypatch):
    conn = FakeConn(fetch_result=[_sibling_row(card_reason_ko=None, card_reason_en=None)])
    monkeypatch.setattr(store, "get_pool", lambda: _async_return(FakePool(conn)))

    result = await store.get_same_day_sibling_context("AAPL", date(2026, 8, 21))

    assert "**Rating**: Sell" in result  # from decision_text, truncated to ~400 chars
    assert len(result) < len(_sibling_row()["decision_text"]) + 200


async def test_get_same_day_sibling_context_picks_language(monkeypatch):
    conn = FakeConn(fetch_result=[_sibling_row(card_reason_ko="퀵비율 0%, 존속위험")])
    monkeypatch.setattr(store, "get_pool", lambda: _async_return(FakePool(conn)))

    result_ko = await store.get_same_day_sibling_context("AAPL", date(2026, 8, 21), language="ko")
    result_en = await store.get_same_day_sibling_context("AAPL", date(2026, 8, 21), language="en")

    assert "퀵비율 0%, 존속위험" in result_ko
    assert "quick ratio 0%, going-concern risk flagged" in result_en


async def test_get_same_day_sibling_context_empty_when_no_rows(monkeypatch):
    conn = FakeConn(fetch_result=[])
    monkeypatch.setattr(store, "get_pool", lambda: _async_return(FakePool(conn)))

    assert await store.get_same_day_sibling_context("AAPL", date(2026, 8, 21)) == ""


async def test_get_same_day_sibling_context_degrades_to_empty_on_db_failure(monkeypatch):
    async def broken_get_pool():
        raise OSError("down")

    monkeypatch.setattr(store, "get_pool", broken_get_pool)
    assert await store.get_same_day_sibling_context("AAPL", date(2026, 8, 21)) == ""


async def test_get_same_day_sibling_context_backtest_returns_empty_without_touching_pool(monkeypatch):
    # Live-only — a historical replay has no intraday "earlier today", and
    # the language-A/B isolation harness requires every backtest field
    # besides past_context to stay untouched by live-only retrieval.
    def fail_if_called():
        raise AssertionError("must not touch the pool in backtest mode")

    monkeypatch.setattr(store, "get_pool", fail_if_called)

    result = await store.get_same_day_sibling_context("AAPL", date(2026, 8, 21), as_of=date(2026, 8, 21))

    assert result == ""


async def test_get_same_day_sibling_context_disabled_by_zero_config(monkeypatch):
    def fail_if_called():
        raise AssertionError("must not touch the pool when the feature is disabled")

    monkeypatch.setattr(store, "get_pool", fail_if_called)
    monkeypatch.setattr(store.settings, "memory_sibling_context_n", 0)

    result = await store.get_same_day_sibling_context("AAPL", date(2026, 8, 21))

    assert result == ""


async def test_get_same_day_sibling_context_scopes_query_by_ticker_date_and_account_mode(monkeypatch):
    conn = FakeConn(fetch_result=[])
    monkeypatch.setattr(store, "get_pool", lambda: _async_return(FakePool(conn)))

    await store.get_same_day_sibling_context("AAPL", date(2026, 8, 21))

    query, args = conn.queries[0]
    assert "ticker = $1 AND trade_date = $2" in query
    assert "account_mode = $4" in query
    assert "NOT is_backtest" in query and "NOT flagged_bad" in query
    assert args[0] == "AAPL"
    assert args[1] == date(2026, 8, 21)


# ---------------------------------------------------------------------------
# get_recent_entries
# ---------------------------------------------------------------------------


async def test_get_recent_entries_maps_rows_to_decision_entries(monkeypatch):
    row = {
        "id": 1, "ticker": "AAPL", "trade_date": date(2026, 7, 16), "rating": "Buy",
        "status": "resolved", "raw_return": 0.05, "alpha_return": 0.02, "reflection": "good",
        "decision_text": "**Rating**: Buy", "created_at": datetime(2026, 7, 16, 9, 5),
        "action_label": "Buy", "card_reason_ko": "이유", "card_reason_en": "reason",
        "total_tokens": 4321, "decision_trigger": "manual",
    }
    conn = FakeConn(fetch_result=[row])
    monkeypatch.setattr(store, "get_pool", lambda: _async_return(FakePool(conn)))
    entries = await store.get_recent_entries()
    assert len(entries) == 1
    assert entries[0].ticker == "AAPL"
    assert entries[0].rating == "Buy"
    assert entries[0].card_reason_ko == "이유"
    assert entries[0].card_reason_en == "reason"
    assert entries[0].total_tokens == 4321
    assert entries[0].decision_trigger == "manual"


async def test_tickers_analyzed_on_returns_distinct_tickers(monkeypatch):
    conn = FakeConn(fetch_result=[{"ticker": "AAPL"}, {"ticker": "005930.KS"}])
    monkeypatch.setattr(store, "get_pool", lambda: _async_return(FakePool(conn)))
    result = await store.tickers_analyzed_on(date(2026, 7, 21))
    assert result == {"AAPL", "005930.KS"}
    query, args = conn.queries[0]
    assert "WHERE trade_date = $1" in query
    assert "user_id = $2" in query
    assert args == (date(2026, 7, 21), "default")


async def test_tickers_with_pending_resolution_returns_distinct_tickers(monkeypatch):
    conn = FakeConn(fetch_result=[{"ticker": "AAPL"}, {"ticker": "005930.KS"}])
    monkeypatch.setattr(store, "get_pool", lambda: _async_return(FakePool(conn)))
    result = await store.tickers_with_pending_resolution(date(2026, 7, 21))
    assert result == {"AAPL", "005930.KS"}
    query, args = conn.queries[0]
    assert "status = 'pending'" in query
    assert "trade_date <= $1" in query
    assert "is_backtest = false" in query
    assert "user_id = $2" in query
    assert args == (date(2026, 7, 21), "default")


async def test_get_recent_entries_filters_by_ticker_when_given(monkeypatch):
    conn = FakeConn(fetch_result=[])
    monkeypatch.setattr(store, "get_pool", lambda: _async_return(FakePool(conn)))
    await store.get_recent_entries(ticker="AAPL", limit=10)
    query, args = conn.queries[0]
    assert "WHERE ticker = $1" in query
    assert "user_id = $4" in query
    assert args == ("AAPL", "paper", 10, "default")


async def test_get_recent_entries_scopes_to_current_account_mode(monkeypatch):
    # 2026-08-04: the dashboard has no legitimate reason to show the OTHER
    # account's recent decisions mixed in with this one's.
    conn = FakeConn(fetch_result=[])
    monkeypatch.setattr(store, "get_pool", lambda: _async_return(FakePool(conn)))
    monkeypatch.setattr(store.settings, "kis_account_mode", "real")
    await store.get_recent_entries()
    query, args = conn.queries[0]
    assert "account_mode = $1" in query
    assert "user_id = $3" in query
    assert args == ("real", 50, "default")


# ---------------------------------------------------------------------------
# get_last_decision_time
# ---------------------------------------------------------------------------


async def test_get_last_decision_time_returns_timestamp(monkeypatch):
    conn = FakeConn(fetchrow_result={"last_decision": datetime(2026, 7, 30, 3, 0, 0)})
    monkeypatch.setattr(store, "get_pool", lambda: _async_return(FakePool(conn)))
    result = await store.get_last_decision_time("005930.KS")
    assert result == datetime(2026, 7, 30, 3, 0, 0)
    query, args = conn.queries[0]
    assert "is_backtest = false" in query
    assert "user_id = $2" in query
    assert "account_mode = $3" in query
    assert args == ("005930.KS", "default", "paper")


async def test_get_last_decision_time_returns_none_for_never_analyzed_ticker(monkeypatch):
    conn = FakeConn(fetchrow_result={"last_decision": None})
    monkeypatch.setattr(store, "get_pool", lambda: _async_return(FakePool(conn)))
    result = await store.get_last_decision_time("005930.KS")
    assert result is None


async def test_get_last_decision_time_degrades_quietly_on_db_failure(monkeypatch):
    # A cost-saving throttle, not a safety gate — must return None (treat as
    # "never analyzed") rather than raise, unlike get_recent_entries/
    # tickers_analyzed_on above which deliberately do NOT degrade quietly.
    async def broken_get_pool():
        raise asyncpg.InterfaceError("pool is closing")

    monkeypatch.setattr(store, "get_pool", broken_get_pool)
    result = await store.get_last_decision_time("005930.KS")
    assert result is None


# ---------------------------------------------------------------------------
# tickers_cash_floor_capped
# ---------------------------------------------------------------------------


async def test_tickers_cash_floor_capped_returns_empty_for_empty_input():
    assert await store.tickers_cash_floor_capped([]) == []


async def test_tickers_cash_floor_capped_matches_the_marker_text(monkeypatch):
    conn = FakeConn(fetch_result=[
        {"ticker": "005930.KS", "decision_text": "some reasoning\n\n[EXECUTION NOTE: Portfolio cash floor: this buy would leave only 5% ...]"},
        {"ticker": "000660.KS", "decision_text": "some reasoning, decided Hold with no cap involved"},
    ])
    monkeypatch.setattr(store, "get_pool", lambda: _async_return(FakePool(conn)))
    result = await store.tickers_cash_floor_capped(["005930.KS", "000660.KS"])
    assert result == ["005930.KS"]
    query, args = conn.queries[0]
    assert "is_backtest = false" in query
    assert "user_id = $2" in query
    assert "account_mode = $3" in query
    assert args == (["005930.KS", "000660.KS"], "default", "paper")


async def test_tickers_cash_floor_capped_ignores_other_cap_reasons(monkeypatch):
    # A 1%-risk-cap or max-open-positions block means something other than
    # cash was the constraint — more cash wouldn't change the outcome, so
    # these must NOT match.
    conn = FakeConn(fetch_result=[
        {"ticker": "005930.KS", "decision_text": "[EXECUTION NOTE: 1% risk cap reduced the intended add ...]"},
        {"ticker": "000660.KS", "decision_text": "[EXECUTION NOTE: Max open positions cap: 15 positions already open ...]"},
    ])
    monkeypatch.setattr(store, "get_pool", lambda: _async_return(FakePool(conn)))
    result = await store.tickers_cash_floor_capped(["005930.KS", "000660.KS"])
    assert result == []


async def test_tickers_cash_floor_capped_degrades_quietly_on_db_failure(monkeypatch):
    async def broken_get_pool():
        raise asyncpg.InterfaceError("pool is closing")

    monkeypatch.setattr(store, "get_pool", broken_get_pool)
    result = await store.tickers_cash_floor_capped(["005930.KS"])
    assert result == []


# ---------------------------------------------------------------------------
# clear_backtest_memory — cross-run backtest memory isolation
# ---------------------------------------------------------------------------


async def test_clear_backtest_memory_deletes_only_is_backtest_rows_for_the_given_tickers(monkeypatch):
    conn = FakeConn(fetch_result=[{"id": 1}, {"id": 2}])
    monkeypatch.setattr(store, "get_pool", lambda: _async_return(FakePool(conn)))
    monkeypatch.setattr(store.settings, "kis_account_mode", "paper")

    await store.clear_backtest_memory(["AAPL", "MSFT"])

    query, args = conn.queries[0]
    assert "DELETE FROM memory_entries" in query
    assert "is_backtest = true" in query
    assert args == (["AAPL", "MSFT"], "paper", "default")


async def test_clear_backtest_memory_is_a_noop_for_an_empty_ticker_list(monkeypatch):
    conn = FakeConn()
    monkeypatch.setattr(store, "get_pool", lambda: _async_return(FakePool(conn)))

    await store.clear_backtest_memory([])

    assert conn.queries == []  # never touches the DB — nothing to clear


async def test_clear_backtest_memory_raises_loudly_on_db_failure(monkeypatch):
    # Unlike most functions in this module, a silent failure here would let
    # exactly the cross-run memory contamination this function exists to
    # prevent slip back in unnoticed — so it must NOT degrade quietly, same
    # as app.execution.ledger.clear_backtest_positions.
    async def broken_get_pool():
        raise asyncpg.InterfaceError("pool is closing")

    monkeypatch.setattr(store, "get_pool", broken_get_pool)

    with pytest.raises(asyncpg.InterfaceError):
        await store.clear_backtest_memory(["AAPL"])


async def _async_return(value):
    return value


# ---------------------------------------------------------------------------
# git_commit resolution — env var precedence (2026-08-13)
#
# The subprocess path CANNOT work in the deployed container: .dockerignore
# excludes .git and python:3.12-slim ships no git binary, so `git rev-parse`
# failed on every deployed decision and the failure was then cached for the
# life of the process. Confirmed against the live DB: 0 of 580 memory_entries
# rows carried a git_commit, while llm_model (resolved from config, not a
# subprocess) was populated normally. The Dockerfile now bakes GIT_COMMIT in
# at build time and this reads it first.
# ---------------------------------------------------------------------------


def test_current_git_commit_prefers_the_env_var_over_the_subprocess(monkeypatch):
    def should_not_run(cmd, **kwargs):
        raise AssertionError("the baked-in commit must be used without shelling out to git")

    monkeypatch.setattr(store.settings, "git_commit", "cafe123")
    monkeypatch.setattr(store.subprocess, "run", should_not_run)

    assert store._current_git_commit() == "cafe123"


def test_current_git_commit_falls_back_to_git_when_env_var_is_unset(monkeypatch):
    class FakeCompletedProcess:
        stdout = "localdev\n"

    monkeypatch.setattr(store.settings, "git_commit", None)
    monkeypatch.setattr(store.subprocess, "run", lambda cmd, **kwargs: FakeCompletedProcess())

    # A local checkout still self-identifies with no env setup at all.
    assert store._current_git_commit() == "localdev"


def test_current_git_commit_ignores_an_empty_env_var(monkeypatch):
    """Dockerfile's `ARG GIT_COMMIT=""` default means a plain `docker build`
    with no --build-arg yields an empty string, not an unset var — that must
    fall through to the subprocess rather than being tagged as a commit."""
    class FakeCompletedProcess:
        stdout = "fallback\n"

    monkeypatch.setattr(store.settings, "git_commit", "")
    monkeypatch.setattr(store.subprocess, "run", lambda cmd, **kwargs: FakeCompletedProcess())

    assert store._current_git_commit() == "fallback"
