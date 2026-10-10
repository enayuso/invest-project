"""app.audit.screening_ic — asyncpg pool/connection faked, no real Postgres.
Same fake-pool convention as tests/audit/test_weekly.py."""

from __future__ import annotations

from datetime import date, datetime, timezone

import pytest

from app.audit import screening_ic
from app.config import settings


class FakeConn:
    def __init__(self, fetch_result=None):
        self.fetch_result = fetch_result if fetch_result is not None else []
        self.queries: list[tuple] = []

    async def fetch(self, query, *args):
        self.queries.append((query, args))
        return self.fetch_result

    async def execute(self, query, *args):
        self.queries.append((query, args))
        return "INSERT 0 1"

    def transaction(self):
        return _TxnCM()


class _TxnCM:
    async def __aenter__(self):
        return None

    async def __aexit__(self, *exc):
        return False


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
    monkeypatch.setattr(screening_ic, "_schema_ready", True)


async def test_resolve_due_forward_returns_noop_when_flag_disabled(monkeypatch):
    monkeypatch.setattr(settings, "screening_snapshot_log_enabled", False)

    def _boom():
        raise AssertionError("get_pool must not be called when the flag is off")

    monkeypatch.setattr(screening_ic, "get_pool", _boom)

    count = await screening_ic.resolve_due_forward_returns()

    assert count == 0


async def test_resolve_due_forward_returns_writes_a_row_per_due_pair(monkeypatch):
    monkeypatch.setattr(settings, "screening_snapshot_log_enabled", True)
    _reset_schema_state(monkeypatch)
    conn = FakeConn(fetch_result=[
        {"ticker": "005930.KS", "trade_date": date(2026, 8, 20)},
    ])
    monkeypatch.setattr(screening_ic, "get_pool", lambda: _async_return(FakePool(conn)))

    async def fake_forward_return(ticker, trade_date, holding_days):
        assert ticker == "005930.KS"
        assert trade_date == date(2026, 8, 20)
        return 0.04

    monkeypatch.setattr(screening_ic, "_forward_return", fake_forward_return)

    count = await screening_ic.resolve_due_forward_returns(holding_days=10)

    assert count == 1
    insert_calls = [q for q, _args in conn.queries if q.strip().upper().startswith("INSERT")]
    assert len(insert_calls) == 1


async def test_resolve_due_forward_returns_degrades_to_zero_on_failure(monkeypatch):
    monkeypatch.setattr(settings, "screening_snapshot_log_enabled", True)
    _reset_schema_state(monkeypatch)

    def _boom():
        raise RuntimeError("db down")

    monkeypatch.setattr(screening_ic, "get_pool", _boom)

    count = await screening_ic.resolve_due_forward_returns()  # must not raise

    assert count == 0


async def test_compute_screening_ic_matches_factor_metrics_rank_ic_directly(monkeypatch):
    """A minimal two-date fixture: compute_screening_ic's per-date RankIC
    must equal calling app.backtest.factor_metrics.rank_ic directly on the
    same (score, forward_return) pairs — this pins the join/aggregation
    logic against the canonical statistic it's supposed to reuse, not
    reimplement."""
    from app.backtest import factor_metrics

    day1_rows = [
        {"trade_date": date(2026, 8, 1), "ticker": "A", "score": 3.0, "selected": True,
         "selection_reason": "momentum", "lookback_days": 15, "volatility_adjusted": True,
         "correlation_dedup": True, "forward_return": 0.05, "executable": True,
         "score_observational": False, "rebalance_locked": False, "max_tickers": 4},
        {"trade_date": date(2026, 8, 1), "ticker": "B", "score": 2.0, "selected": False,
         "selection_reason": None, "lookback_days": 15, "volatility_adjusted": True,
         "correlation_dedup": True, "forward_return": 0.02, "executable": True,
         "score_observational": False, "rebalance_locked": False, "max_tickers": 4},
        {"trade_date": date(2026, 8, 1), "ticker": "C", "score": 1.0, "selected": False,
         "selection_reason": None, "lookback_days": 15, "volatility_adjusted": True,
         "correlation_dedup": True, "forward_return": -0.01, "executable": True,
         "score_observational": False, "rebalance_locked": False, "max_tickers": 4},
        {"trade_date": date(2026, 8, 1), "ticker": "D", "score": 0.0, "selected": False,
         "selection_reason": None, "lookback_days": 15, "volatility_adjusted": True,
         "correlation_dedup": True, "forward_return": -0.03, "executable": True,
         "score_observational": False, "rebalance_locked": False, "max_tickers": 4},
        {"trade_date": date(2026, 8, 1), "ticker": "E", "score": -1.0, "selected": False,
         "selection_reason": None, "lookback_days": 15, "volatility_adjusted": True,
         "correlation_dedup": True, "forward_return": -0.04, "executable": True,
         "score_observational": False, "rebalance_locked": False, "max_tickers": 4},
    ]
    conn = FakeConn(fetch_result=day1_rows)
    monkeypatch.setattr(screening_ic, "get_pool", lambda: _async_return(FakePool(conn)))

    report = await screening_ic.compute_screening_ic(
        "domestic", date(2026, 8, 1), date(2026, 8, 1),
        holding_days=10, user_id="default", account_mode="paper",
    )

    assert report.n_dates == 1
    scores = {r["ticker"]: r["score"] for r in day1_rows}
    forwards = {r["ticker"]: r["forward_return"] for r in day1_rows}
    expected_ic = factor_metrics.rank_ic(scores, forwards)
    assert report.per_date[0].rank_ic == expected_ic
    # The one 'momentum'-selected row (A) is the only pick.
    assert report.per_date[0].picks == ["A"]


async def test_compute_screening_ic_excludes_forced_review_picks_from_excess_return(monkeypatch):
    """held_review/cash_freed rows must never count as a screening 'pick' —
    they're forced risk-review entries the ranking didn't choose, and mixing
    them into excess-return would measure something other than screening
    performance (project memory screening-analysis-2026-09-08 §4-1)."""
    rows = [
        {"trade_date": date(2026, 8, 1), "ticker": "A", "score": 1.0, "selected": True,
         "selection_reason": "held_review", "lookback_days": 15, "volatility_adjusted": True,
         "correlation_dedup": True, "forward_return": -0.20, "executable": True,
         "score_observational": False, "rebalance_locked": False, "max_tickers": 4},
        {"trade_date": date(2026, 8, 1), "ticker": "B", "score": 2.0, "selected": True,
         "selection_reason": "momentum", "lookback_days": 15, "volatility_adjusted": True,
         "correlation_dedup": True, "forward_return": 0.05, "executable": True,
         "score_observational": False, "rebalance_locked": False, "max_tickers": 4},
        {"trade_date": date(2026, 8, 1), "ticker": "C", "score": 0.5, "selected": False,
         "selection_reason": None, "lookback_days": 15, "volatility_adjusted": True,
         "correlation_dedup": True, "forward_return": 0.01, "executable": True,
         "score_observational": False, "rebalance_locked": False, "max_tickers": 4},
    ]
    conn = FakeConn(fetch_result=rows)
    monkeypatch.setattr(screening_ic, "get_pool", lambda: _async_return(FakePool(conn)))

    report = await screening_ic.compute_screening_ic(
        "domestic", date(2026, 8, 1), date(2026, 8, 1), holding_days=10,
    )

    assert report.per_date[0].picks == ["B"]


async def test_compute_screening_ic_warns_when_settings_change_mid_range(monkeypatch):
    rows = [
        {"trade_date": date(2026, 8, 1), "ticker": "A", "score": 1.0, "selected": False,
         "selection_reason": None, "lookback_days": 15, "volatility_adjusted": True,
         "correlation_dedup": True, "forward_return": 0.01, "executable": True,
         "score_observational": False, "rebalance_locked": False, "max_tickers": 4},
        {"trade_date": date(2026, 8, 2), "ticker": "A", "score": 1.0, "selected": False,
         "selection_reason": None, "lookback_days": 20, "volatility_adjusted": True,
         "correlation_dedup": True, "forward_return": 0.01, "executable": True,
         "score_observational": False, "rebalance_locked": False, "max_tickers": 4},
    ]
    conn = FakeConn(fetch_result=rows)
    monkeypatch.setattr(screening_ic, "get_pool", lambda: _async_return(FakePool(conn)))

    report = await screening_ic.compute_screening_ic(
        "domestic", date(2026, 8, 1), date(2026, 8, 2), holding_days=10,
    )

    assert len(report.settings_variants) == 2
    rendered = screening_ic.render_screening_ic(report)
    assert "WARNING" in rendered


async def test_compute_screening_ic_no_lock_cohorts_when_nothing_locked(monkeypatch):
    """A range entirely of pre-§0단계 (or simply lock-free) rows must report
    lock_cohorts=None, not an empty-but-present stat block — the report
    renders '-' for this case rather than a misleadingly precise zero."""
    rows = [
        {"trade_date": date(2026, 8, 1), "ticker": "A", "score": 1.0, "selected": True,
         "selection_reason": "momentum", "lookback_days": 15, "volatility_adjusted": True,
         "correlation_dedup": True, "forward_return": 0.05, "executable": True,
         "score_observational": False, "rebalance_locked": False, "max_tickers": 4},
    ]
    conn = FakeConn(fetch_result=rows)
    monkeypatch.setattr(screening_ic, "get_pool", lambda: _async_return(FakePool(conn)))

    report = await screening_ic.compute_screening_ic(
        "domestic", date(2026, 8, 1), date(2026, 8, 1), holding_days=10,
    )

    assert report.lock_cohorts is None
    assert report.mixed_observational_coverage is False
    rendered = screening_ic.render_screening_ic(report)
    assert "lock cohorts: -" in rendered


async def test_compute_screening_ic_lock_cohorts_show_ranking_never_saw_locked_ticker(monkeypatch):
    """The core §0단계 measurement: a rebalance_locked ticker with the HIGHEST
    score of the day (would have topped the counterfactual top-1) must show
    up in mean_locked_in_counterfactual_top_n=1.0 and contribute to
    mean_fwd_locked, NOT to mean_fwd_picked/mean_fwd_passed — the ranking
    never had the chance to pick it."""
    rows = [
        # Highest score, but locked out — the ranking never saw this one.
        {"trade_date": date(2026, 8, 1), "ticker": "LOCKED", "score": 9.0, "selected": False,
         "selection_reason": None, "lookback_days": 15, "volatility_adjusted": True,
         "correlation_dedup": True, "forward_return": 0.10, "executable": False,
         "score_observational": True, "rebalance_locked": True, "max_tickers": 1},
        # Executable, ranked, and actually picked.
        {"trade_date": date(2026, 8, 1), "ticker": "PICKED", "score": 2.0, "selected": True,
         "selection_reason": "momentum", "lookback_days": 15, "volatility_adjusted": True,
         "correlation_dedup": True, "forward_return": 0.03, "executable": True,
         "score_observational": False, "rebalance_locked": False, "max_tickers": 1},
        # Executable, ranked, but not picked (budget cut).
        {"trade_date": date(2026, 8, 1), "ticker": "PASSED", "score": 1.0, "selected": False,
         "selection_reason": None, "lookback_days": 15, "volatility_adjusted": True,
         "correlation_dedup": True, "forward_return": -0.02, "executable": True,
         "score_observational": False, "rebalance_locked": False, "max_tickers": 1},
    ]
    conn = FakeConn(fetch_result=rows)
    monkeypatch.setattr(screening_ic, "get_pool", lambda: _async_return(FakePool(conn)))

    report = await screening_ic.compute_screening_ic(
        "domestic", date(2026, 8, 1), date(2026, 8, 1), holding_days=10,
    )

    assert report.per_date[0].n_observational == 1
    assert report.mixed_observational_coverage is False  # only one date, nothing to mix against
    lc = report.lock_cohorts
    assert lc is not None
    assert lc.mean_locked_in_counterfactual_top_n == 1.0
    assert lc.mean_fwd_locked == 0.10
    assert lc.mean_fwd_picked == 0.03
    assert lc.mean_fwd_passed == -0.02
    rendered = screening_ic.render_screening_ic(report)
    assert "lock cohorts" in rendered


def _axis_row(ticker, captured_at, axes, score=1.0, forward_return=None, trade_date=date(2026, 9, 1)):
    return {
        "trade_date": trade_date, "ticker": ticker, "captured_at": captured_at,
        "axes": axes, "score": score, "forward_return": forward_return,
    }


def test_compute_axis_cohorts_folds_axes_union_across_captures_not_double_counted():
    """Two captures of the SAME ticker on the SAME day, each naming a
    different single axis, must fold into ONE ticker-day whose axes set is
    the union of both — so it counts once (not twice) toward each axis's
    marginal cohort, and toward neither exclusive cohort (its folded axes
    set is {krx_top_n, market_cap_top_n}, not a singleton)."""
    rows = [
        _axis_row("A", captured_at=1, axes=["krx_top_n"], forward_return=0.05),
        _axis_row("A", captured_at=2, axes=["market_cap_top_n"], forward_return=0.05),
        # A second ticker on an UNRELATED axis, present only so the day has
        # more than one captured_at and the cross-sectional baseline isn't
        # degenerate — must not itself land in krx_top_n/market_cap_top_n.
        _axis_row("B", captured_at=1, axes=["unrelated_axis"], forward_return=-0.01),
    ]
    cohorts = screening_ic._compute_axis_cohorts(rows, holding_days=10)

    krx_marginal = [c for c in cohorts if c.axis == "krx_top_n" and c.mode == "marginal"]
    mcap_marginal = [c for c in cohorts if c.axis == "market_cap_top_n" and c.mode == "marginal"]
    # A has 2/2 captures that day (rate=1.0) -> persistent bucket for both axes.
    assert sum(c.n_ticker_days for c in krx_marginal if c.bucket == "persistent") == 1  # A only, not 2
    assert sum(c.n_ticker_days for c in mcap_marginal if c.bucket == "persistent") == 1
    krx_exclusive = [c for c in cohorts if c.axis == "krx_top_n" and c.mode == "exclusive"]
    # A's folded axes = {krx_top_n, market_cap_top_n} — never a singleton, and
    # nobody else carries krx_top_n at all, so the exclusive cohort is empty.
    assert krx_exclusive == []
    mcap_exclusive = [c for c in cohorts if c.axis == "market_cap_top_n" and c.mode == "exclusive"]
    assert mcap_exclusive == []  # nobody has axes == {market_cap_top_n} exactly


def test_compute_axis_cohorts_marginal_includes_overlap_exclusive_does_not():
    rows = [
        _axis_row("MULTI", captured_at=1, axes=["krx_top_n", "market_cap_top_n"], forward_return=0.02),
        _axis_row("SOLO", captured_at=1, axes=["krx_top_n"], forward_return=0.03),
    ]
    cohorts = screening_ic._compute_axis_cohorts(rows, holding_days=10)

    marginal_krx = next(c for c in cohorts if c.axis == "krx_top_n" and c.mode == "marginal" and c.bucket == "persistent")
    assert marginal_krx.n_ticker_days == 2  # both MULTI and SOLO carry krx_top_n
    exclusive_krx = next(c for c in cohorts if c.axis == "krx_top_n" and c.mode == "exclusive" and c.bucket == "persistent")
    assert exclusive_krx.n_ticker_days == 1  # only SOLO is exactly {krx_top_n}


def test_compute_axis_cohorts_appearance_rate_bucket_boundary_is_le_0_2():
    """10 captures that day. A ticker appearing in exactly 2 (rate=0.2) must
    land in `transient` (<=0.2); one appearing in 3 (rate=0.3) must land in
    `persistent` — the boundary is inclusive on the transient side."""
    rows = [_axis_row("BOUNDARY", captured_at=i, axes=["krx_top_n"], forward_return=0.01) for i in range(1, 3)]
    rows += [_axis_row("ABOVE", captured_at=i, axes=["krx_top_n"], forward_return=0.01) for i in range(1, 4)]
    # Pad out remaining captures of the day with an unrelated ticker so the
    # day has exactly 10 distinct captured_at values.
    rows += [_axis_row("FILLER", captured_at=i, axes=[]) for i in range(4, 11)]

    cohorts = screening_ic._compute_axis_cohorts(rows, holding_days=10)
    krx = [c for c in cohorts if c.axis == "krx_top_n" and c.mode == "marginal"]
    transient = next(c for c in krx if c.bucket == "transient")
    persistent = next(c for c in krx if c.bucket == "persistent")
    assert transient.n_ticker_days == 1  # BOUNDARY (2/10 = 0.2)
    assert persistent.n_ticker_days == 1  # ABOVE (3/10 = 0.3)


def test_compute_axis_cohorts_empty_forward_returns_reports_no_data_not_no_difference():
    """forward_returns is entirely empty right now (table populated from
    ~2026-09-23) — the population stats (n_dates/n_ticker_days) must still
    compute, but mean_excess/CI/RankIC must all be None (n_resolved=0), so
    the caller can render 'no data yet' instead of a false 'no difference'."""
    rows = [
        _axis_row("A", captured_at=1, axes=["krx_top_n"], forward_return=None),
        _axis_row("B", captured_at=1, axes=["krx_top_n"], forward_return=None),
    ]
    cohorts = screening_ic._compute_axis_cohorts(rows, holding_days=10)
    assert cohorts  # population is still measurable
    for c in cohorts:
        assert c.n_ticker_days > 0
        assert c.n_resolved == 0
        assert c.mean_excess is None
        assert c.excess_ci95 is None
        assert c.rank_ic is None


def test_compute_axis_cohorts_cross_sectional_baseline_is_same_day_mean_not_index():
    """mean_excess must be forward_return minus THAT DATE's cross-sectional
    mean over every scored ticker-day (not an index benchmark) — verified
    with hand-computed numbers: returns 0.10, 0.00, -0.10 -> baseline 0.0,
    so each one's excess equals its own raw return exactly."""
    rows = [
        _axis_row("A", captured_at=1, axes=["krx_top_n"], forward_return=0.10),
        _axis_row("B", captured_at=1, axes=["krx_top_n"], forward_return=0.00),
        _axis_row("C", captured_at=1, axes=["krx_top_n"], forward_return=-0.10),
    ]
    cohorts = screening_ic._compute_axis_cohorts(rows, holding_days=10)
    marginal_persistent = next(
        c for c in cohorts if c.axis == "krx_top_n" and c.mode == "marginal" and c.bucket == "persistent"
    )
    assert marginal_persistent.n_ticker_days == 3
    assert marginal_persistent.mean_excess == pytest.approx(0.0, abs=1e-9)


def test_compute_axis_cohorts_missing_columns_degrade_gracefully(monkeypatch):
    """Legacy/pre-existing test fixtures (and the FIRST query's own row
    shape) don't carry captured_at/axes at all — `.get()` must degrade this
    to 'no axis data' rather than raising, since compute_screening_ic's
    FakeConn test double returns the SAME fetch_result for both the
    first-cycle query and this module's second (axis) query."""
    rows = [
        {"trade_date": date(2026, 8, 1), "ticker": "A", "score": 1.0, "selected": False,
         "selection_reason": None, "lookback_days": 15, "volatility_adjusted": True,
         "correlation_dedup": True, "forward_return": 0.01, "executable": True,
         "score_observational": False, "rebalance_locked": False, "max_tickers": 4},
    ]
    cohorts = screening_ic._compute_axis_cohorts(rows, holding_days=10)
    assert cohorts == []


async def test_compute_screening_ic_wires_axis_cohorts_through_and_renders(monkeypatch):
    """Integration smoke test: axis/captured_at columns on the fetched rows
    reach report.axis_cohorts, and render_screening_ic includes the new
    section without raising."""
    rows = [
        _axis_row("A", captured_at=1, axes=["krx_top_n"], forward_return=None, score=1.0),
        _axis_row("B", captured_at=1, axes=["disparity_low"], forward_return=None, score=0.5),
    ]
    # The compute_screening_ic FIRST query additionally reads selected/
    # selection_reason/etc — absent here, `.get()`-free direct indexing
    # would raise, so this also proves the first-cycle path tolerates the
    # axis-only fixture the same way FakeConn's single fetch_result implies.
    for row in rows:
        row.update({
            "selected": False, "selection_reason": None, "lookback_days": 15,
            "volatility_adjusted": True, "correlation_dedup": True, "executable": True,
            "score_observational": False, "rebalance_locked": False, "max_tickers": 4,
        })
    conn = FakeConn(fetch_result=rows)
    monkeypatch.setattr(screening_ic, "get_pool", lambda: _async_return(FakePool(conn)))

    report = await screening_ic.compute_screening_ic(
        "domestic", date(2026, 9, 1), date(2026, 9, 1), holding_days=10,
    )

    assert len(report.axis_cohorts) > 0
    rendered = screening_ic.render_screening_ic(report)
    assert "axis / persistence cohorts" in rendered
    assert "데이터없음" in rendered  # forward_return is None for both rows here


async def test_render_axis_column_keeps_gap_for_long_axis_name(monkeypatch):
    """'fluctuation_losers' is exactly 18 chars, the old fixed column width, so it
    used to run straight into the mode column ('fluctuation_losersmarginal')."""
    rows = [
        _axis_row("A", captured_at=1, axes=["fluctuation_losers"], forward_return=None, score=1.0),
        _axis_row("B", captured_at=1, axes=["krx_top_n"], forward_return=None, score=0.5),
    ]
    for row in rows:
        row.update({
            "selected": False, "selection_reason": None, "lookback_days": 15,
            "volatility_adjusted": True, "correlation_dedup": True, "executable": True,
            "score_observational": False, "rebalance_locked": False, "max_tickers": 4,
        })
    conn = FakeConn(fetch_result=rows)
    monkeypatch.setattr(screening_ic, "get_pool", lambda: _async_return(FakePool(conn)))

    report = await screening_ic.compute_screening_ic(
        "domestic", date(2026, 9, 1), date(2026, 9, 1), holding_days=10,
    )
    rendered = screening_ic.render_screening_ic(report)

    assert "fluctuation_losersmarginal" not in rendered
    assert "fluctuation_losers  marginal" in rendered  # at least a 2-space gap
    header = next(line for line in rendered.splitlines() if "n_tdays" in line and "mode" in line)
    row_line = next(line for line in rendered.splitlines() if "fluctuation_losers" in line and "marginal" in line)
    assert header.index("mode") == row_line.index("marginal")  # header and rows stay aligned


async def test_compute_screening_ic_flags_mixed_observational_coverage(monkeypatch):
    """One date pre-deployment (n_observational=0), one date post-deployment
    (n_observational>0) in the same range must set
    mixed_observational_coverage=True and render a warning — pooling the two
    would average an executable-only cross-section with a whole-universe
    one as if they were the same factor."""
    rows = [
        {"trade_date": date(2026, 8, 1), "ticker": "A", "score": 1.0, "selected": True,
         "selection_reason": "momentum", "lookback_days": 15, "volatility_adjusted": True,
         "correlation_dedup": True, "forward_return": 0.01, "executable": True,
         "score_observational": False, "rebalance_locked": False, "max_tickers": 4},
        {"trade_date": date(2026, 8, 2), "ticker": "A", "score": 1.0, "selected": True,
         "selection_reason": "momentum", "lookback_days": 15, "volatility_adjusted": True,
         "correlation_dedup": True, "forward_return": 0.01, "executable": True,
         "score_observational": False, "rebalance_locked": False, "max_tickers": 4},
        {"trade_date": date(2026, 8, 2), "ticker": "B", "score": 0.5, "selected": False,
         "selection_reason": None, "lookback_days": 15, "volatility_adjusted": True,
         "correlation_dedup": True, "forward_return": -0.01, "executable": False,
         "score_observational": True, "rebalance_locked": True, "max_tickers": 4},
    ]
    conn = FakeConn(fetch_result=rows)
    monkeypatch.setattr(screening_ic, "get_pool", lambda: _async_return(FakePool(conn)))

    report = await screening_ic.compute_screening_ic(
        "domestic", date(2026, 8, 1), date(2026, 8, 2), holding_days=10,
    )

    assert report.mixed_observational_coverage is True
    rendered = screening_ic.render_screening_ic(report)
    assert "mixes dates" in rendered
