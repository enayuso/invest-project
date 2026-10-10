"""app.live.screening_log — asyncpg pool/connection faked, no real Postgres.
Same fake-pool convention as tests/live/test_orderbook_log.py."""

from __future__ import annotations

from datetime import date, datetime, timezone

from app.chat.screening import MomentumScore
from app.config import settings
from app.live import screening_log
from app.live.screening_log import ScreeningSnapshotContext, record_screening_snapshot


class _TxnCM:
    async def __aenter__(self):
        return None

    async def __aexit__(self, *exc):
        return False


class FakeConn:
    def __init__(self, fail: bool = False):
        self.fail = fail
        self.executed: list[tuple] = []

    def transaction(self):
        return _TxnCM()

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
    monkeypatch.setattr(screening_log, "_schema_ready", True)


def _ctx(**overrides) -> ScreeningSnapshotContext:
    defaults = dict(
        market="domestic",
        captured_at=datetime(2026, 9, 8, 9, 30, tzinfo=timezone.utc),
        trade_date=date(2026, 9, 8),
        lookback_days=15,
        volatility_adjusted=True,
        correlation_dedup=True,
        top_n_requested=8,
        max_tickers=4,
    )
    defaults.update(overrides)
    return ScreeningSnapshotContext(**defaults)


async def test_disabled_flag_writes_nothing_and_never_touches_the_pool(monkeypatch):
    monkeypatch.setattr(settings, "screening_snapshot_log_enabled", False)

    def _boom():
        raise AssertionError("get_pool must not be called when the flag is off")

    monkeypatch.setattr(screening_log, "get_pool", _boom)

    count = await record_screening_snapshot(
        _ctx(),
        universe=["005930.KS"],
        eligible=["005930.KS"],
        executable=["005930.KS"],
        scored={"005930.KS": MomentumScore(raw_pct=0.05, atr_pct=0.02, score=2.5)},
        score_key="score",
        ranked=[("005930.KS", 2.5)],
        final=["005930.KS"],
        held_review=[],
        cash_freed=[],
        axes_by_ticker={"005930.KS": ["krx_top_n"]},
    )

    assert count == 0


async def test_records_one_row_per_universe_ticker_with_shared_captured_at(monkeypatch):
    monkeypatch.setattr(settings, "screening_snapshot_log_enabled", True)
    _reset_schema_state(monkeypatch)
    conn = FakeConn()
    monkeypatch.setattr(screening_log, "get_pool", lambda: _async_return(FakePool(conn)))

    scored = {
        "005930.KS": MomentumScore(raw_pct=0.05, atr_pct=0.02, score=2.5),
        "000660.KS": MomentumScore(raw_pct=0.03, atr_pct=0.01, score=3.0),
    }
    ctx = _ctx()
    count = await record_screening_snapshot(
        ctx,
        universe=["005930.KS", "000660.KS", "035420.KS"],
        eligible=["005930.KS", "000660.KS"],
        executable=["005930.KS", "000660.KS"],
        scored=scored,
        score_key="score",
        ranked=[("000660.KS", 3.0), ("005930.KS", 2.5)],
        final=["000660.KS"],
        held_review=[],
        cash_freed=[],
        axes_by_ticker={"005930.KS": ["krx_top_n"], "000660.KS": ["krx_top_n", "market_cap_top_n"]},
    )

    assert count == 3
    assert len(conn.executed) == 3
    captured_ats = {args[0] for _q, args in conn.executed}
    assert captured_ats == {ctx.captured_at}

    by_ticker = {args[3]: args for _q, args in conn.executed}
    # 035420.KS was never scored (dropped upstream) — still recorded, with
    # NULL score/atr_pct/rank and axes=[].
    unscored = by_ticker["035420.KS"]
    assert unscored[4] is None  # score
    assert unscored[10] is False  # scored flag
    assert unscored[14] == []  # axes

    # score_rank: 000660.KS (3.0) ranks 1, 005930.KS (2.5) ranks 2.
    assert by_ticker["000660.KS"][7] == 1
    assert by_ticker["005930.KS"][7] == 2
    # axes carried through for a multi-axis ticker.
    assert by_ticker["000660.KS"][14] == ["krx_top_n", "market_cap_top_n"]


async def test_selection_reason_priority_held_review_beats_cash_freed_and_momentum(monkeypatch):
    monkeypatch.setattr(settings, "screening_snapshot_log_enabled", True)
    _reset_schema_state(monkeypatch)
    conn = FakeConn()
    monkeypatch.setattr(screening_log, "get_pool", lambda: _async_return(FakePool(conn)))

    scored = {"005930.KS": MomentumScore(raw_pct=0.01, atr_pct=0.01, score=1.0)}
    await record_screening_snapshot(
        _ctx(),
        universe=["005930.KS"],
        eligible=[],
        executable=[],
        scored=scored,
        score_key="score",
        ranked=[],
        final=["005930.KS"],
        held_review=["005930.KS"],
        cash_freed=["005930.KS"],
        axes_by_ticker={},
    )

    row = conn.executed[0][1]
    assert row[13] == "held_review"  # selection_reason


async def test_score_rank_uses_the_score_key_the_ranking_actually_used(monkeypatch):
    monkeypatch.setattr(settings, "screening_snapshot_log_enabled", True)
    _reset_schema_state(monkeypatch)
    conn = FakeConn()
    monkeypatch.setattr(screening_log, "get_pool", lambda: _async_return(FakePool(conn)))

    # raw_pct order disagrees with score order — score_key="raw_pct" (overseas
    # convention) must rank by raw_pct, not the ATR-adjusted `score` field.
    scored = {
        "AAA": MomentumScore(raw_pct=0.10, atr_pct=0.20, score=0.5),
        "BBB": MomentumScore(raw_pct=0.05, atr_pct=0.01, score=5.0),
    }
    await record_screening_snapshot(
        _ctx(market="overseas", volatility_adjusted=False),
        universe=["AAA", "BBB"],
        eligible=["AAA", "BBB"],
        executable=["AAA", "BBB"],
        scored=scored,
        score_key="raw_pct",
        ranked=[("AAA", 0.10), ("BBB", 0.05)],
        final=[],
        held_review=[],
        cash_freed=[],
        axes_by_ticker={},
    )

    by_ticker = {args[3]: args for _q, args in conn.executed}
    assert by_ticker["AAA"][7] == 1
    assert by_ticker["BBB"][7] == 2
    assert by_ticker["AAA"][4] == 0.10  # score column holds raw_pct here


async def test_observed_ticker_recorded_with_score_observational_flag(monkeypatch):
    """2026-09-09 §0단계 계측: a ticker scored only via the observe-only pass
    must get score_observational=true, its score/raw_pct/atr_pct populated,
    but executable=false and in_ranked_pool/selected=false — the ranking
    never saw it."""
    monkeypatch.setattr(settings, "screening_snapshot_log_enabled", True)
    _reset_schema_state(monkeypatch)
    conn = FakeConn()
    monkeypatch.setattr(screening_log, "get_pool", lambda: _async_return(FakePool(conn)))

    scored = {"005930.KS": MomentumScore(raw_pct=0.01, atr_pct=0.01, score=1.0)}
    observed = {"LOCKED.KS": MomentumScore(raw_pct=0.20, atr_pct=0.05, score=4.0)}
    count = await record_screening_snapshot(
        _ctx(),
        universe=["005930.KS", "LOCKED.KS"],
        eligible=["005930.KS"],
        executable=["005930.KS"],
        scored=scored,
        score_key="score",
        ranked=[("005930.KS", 1.0)],
        final=["005930.KS"],
        held_review=[],
        cash_freed=[],
        axes_by_ticker={},
        observed=observed,
        rebalance_locked={"LOCKED.KS": date(2026, 9, 13)},
    )

    assert count == 2
    by_ticker = {args[3]: args for _q, args in conn.executed}
    locked_row = by_ticker["LOCKED.KS"]
    assert locked_row[4] == 4.0  # score, populated even though never ranked
    assert locked_row[10] is True  # scored flag: it DID have a score this cycle
    assert locked_row[11] is False  # in_ranked_pool
    assert locked_row[12] is False  # selected
    assert locked_row[23] is True  # score_observational
    assert locked_row[24] is True  # rebalance_locked
    assert locked_row[25] == date(2026, 9, 13)  # rebalance_unlock_date

    real_row = by_ticker["005930.KS"]
    assert real_row[23] is False  # score_observational — this one WAS really ranked
    assert real_row[24] is False


async def test_score_rank_spans_scored_and_observed_together(monkeypatch):
    """The counterfactual rank (score_rank) must place an observe-only
    ticker among the real ranking's own scored tickers — this is exactly
    the "would it have made top_n" question §0단계 계측 exists to answer."""
    monkeypatch.setattr(settings, "screening_snapshot_log_enabled", True)
    _reset_schema_state(monkeypatch)
    conn = FakeConn()
    monkeypatch.setattr(screening_log, "get_pool", lambda: _async_return(FakePool(conn)))

    scored = {"005930.KS": MomentumScore(raw_pct=0.01, atr_pct=0.01, score=1.0)}
    observed = {"LOCKED.KS": MomentumScore(raw_pct=0.20, atr_pct=0.05, score=9.0)}  # highest score
    await record_screening_snapshot(
        _ctx(),
        universe=["005930.KS", "LOCKED.KS"],
        eligible=["005930.KS"],
        executable=["005930.KS"],
        scored=scored,
        score_key="score",
        ranked=[("005930.KS", 1.0)],
        final=["005930.KS"],
        held_review=[],
        cash_freed=[],
        axes_by_ticker={},
        observed=observed,
    )

    by_ticker = {args[3]: args for _q, args in conn.executed}
    assert by_ticker["LOCKED.KS"][7] == 1  # score_rank — would have ranked #1
    assert by_ticker["005930.KS"][7] == 2


async def test_scored_wins_over_observed_on_overlap(monkeypatch):
    """A ticker that (incorrectly) appears in both `scored` and `observed`
    must use the real ranking's value and NOT be flagged observational —
    should never happen in practice (screening_pass dedupes), but the
    merge order must still prefer the real value if it ever does."""
    monkeypatch.setattr(settings, "screening_snapshot_log_enabled", True)
    _reset_schema_state(monkeypatch)
    conn = FakeConn()
    monkeypatch.setattr(screening_log, "get_pool", lambda: _async_return(FakePool(conn)))

    scored = {"005930.KS": MomentumScore(raw_pct=0.01, atr_pct=0.01, score=1.0)}
    observed = {"005930.KS": MomentumScore(raw_pct=0.99, atr_pct=0.99, score=99.0)}
    await record_screening_snapshot(
        _ctx(),
        universe=["005930.KS"],
        eligible=["005930.KS"],
        executable=["005930.KS"],
        scored=scored,
        score_key="score",
        ranked=[("005930.KS", 1.0)],
        final=["005930.KS"],
        held_review=[],
        cash_freed=[],
        axes_by_ticker={},
        observed=observed,
    )

    row = conn.executed[0][1]
    assert row[4] == 1.0  # scored's value wins, not observed's 99.0
    assert row[23] is False  # score_observational


async def test_omitting_observed_and_locked_args_matches_pre_existing_behavior(monkeypatch):
    """Regression: a caller that doesn't pass observed/rebalance_locked at
    all (every pre-2026-09-09 caller) must write byte-identical rows."""
    monkeypatch.setattr(settings, "screening_snapshot_log_enabled", True)
    _reset_schema_state(monkeypatch)
    conn = FakeConn()
    monkeypatch.setattr(screening_log, "get_pool", lambda: _async_return(FakePool(conn)))

    scored = {"005930.KS": MomentumScore(raw_pct=0.05, atr_pct=0.02, score=2.5)}
    await record_screening_snapshot(
        _ctx(),
        universe=["005930.KS"],
        eligible=["005930.KS"],
        executable=["005930.KS"],
        scored=scored,
        score_key="score",
        ranked=[("005930.KS", 2.5)],
        final=["005930.KS"],
        held_review=[],
        cash_freed=[],
        axes_by_ticker={},
    )

    row = conn.executed[0][1]
    assert row[23] is False
    assert row[24] is False
    assert row[25] is None


async def test_degrades_to_zero_on_db_failure(monkeypatch):
    monkeypatch.setattr(settings, "screening_snapshot_log_enabled", True)
    _reset_schema_state(monkeypatch)
    conn = FakeConn(fail=True)
    monkeypatch.setattr(screening_log, "get_pool", lambda: _async_return(FakePool(conn)))

    count = await record_screening_snapshot(
        _ctx(),
        universe=["005930.KS"],
        eligible=["005930.KS"],
        executable=["005930.KS"],
        scored={"005930.KS": MomentumScore(raw_pct=0.01, atr_pct=0.01, score=1.0)},
        score_key="score",
        ranked=[("005930.KS", 1.0)],
        final=["005930.KS"],
        held_review=[],
        cash_freed=[],
        axes_by_ticker={},
    )  # must not raise

    assert count == 0
