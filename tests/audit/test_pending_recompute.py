"""app.audit.pending_recompute + entry_paired's --recompute-pending mode
(2026-09-28). No Postgres, no network: the asyncpg pool is faked and
`get_price_series` is patched with dict-backed series."""

from __future__ import annotations

import math
import sys
from datetime import date, timedelta

import pytest

from app.audit import __main__ as audit_main
from app.audit import entry_paired, pending_recompute
from app.memory import auto_resolve

TODAY = date(2026, 9, 28)
MON = date(2026, 9, 14)  # decision date; not_before = +5d = Sat 09-19

# entry 100 -> 110 (raw +10%), benchmark 1000 -> 1010 (+1%): alpha = +9%
def _series(entry_price: float, exit_price: float, dates_after: list[date], entry: date = MON) -> dict:
    out = {entry: entry_price}
    out.update({d: exit_price for d in dates_after})
    return out


BENCH_DATES = [date(2026, 9, 18), date(2026, 9, 21), date(2026, 9, 22)]


def _bench(dates_after=None) -> dict:
    return _series(1000.0, 1010.0, BENCH_DATES if dates_after is None else dates_after)


class FakeTxn:
    def __init__(self, conn, kwargs):
        self._conn = conn
        conn.transaction_kwargs.append(kwargs)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class FakeConn:
    def __init__(self, baseline=None, pending=None, resolved_keys=None):
        self.baseline = baseline or []
        self.pending = pending or []
        self.resolved_keys = resolved_keys or []
        self.queries: list[tuple] = []
        self.transaction_kwargs: list[dict] = []

    def transaction(self, **kwargs):
        return FakeTxn(self, kwargs)

    async def fetch(self, query, *args):
        self.queries.append((query, args))
        if "status = 'pending'" in query:
            return self.pending
        if query.strip().startswith("SELECT DISTINCT ticker"):
            return [{"ticker": t, "trade_date": d} for t, d in self.resolved_keys]
        return self.baseline


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


def _pending(ticker, trade_date, rating="Hold", held=False):
    return {
        "ticker": ticker, "trade_date": trade_date, "rating": rating,
        "held_at_decision": held, "held_at_decision_inferred": None, "created_at": None,
    }


def _baseline(trade_date, rating, alpha, ticker="B", held=None):
    return {
        "trade_date": trade_date, "ticker": ticker, "rating": rating, "alpha_return": alpha,
        "held_at_decision": held, "held_at_decision_inferred": None,
    }


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    """Settled sessions everywhere, fixed KST 'today', no real user id."""
    for module in (pending_recompute, auto_resolve):
        monkeypatch.setattr(module, "session_in_progress", lambda t, d: False)
    monkeypatch.setattr(entry_paired, "local_today", lambda market: TODAY)
    monkeypatch.setattr(entry_paired, "get_current_user_id", lambda: "default")


def _patch_prices(monkeypatch, table: dict[str, dict], calls: list | None = None):
    async def fake(symbol, start, end):
        if calls is not None:
            calls.append(symbol)
        if symbol not in table:
            raise ValueError(f"no price history returned for {symbol!r}")
        return table[symbol]

    monkeypatch.setattr(pending_recompute, "get_price_series", fake)


def _patch_db(monkeypatch, conn):
    async def _pool():
        return FakePool(conn)

    monkeypatch.setattr(entry_paired, "get_pool", _pool)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def test_alternate_krx_symbol():
    assert pending_recompute.alternate_krx_symbol("950260.KS") == "950260.KQ"
    assert pending_recompute.alternate_krx_symbol("950260.KQ") == "950260.KS"
    assert pending_recompute.alternate_krx_symbol("AAPL") is None


async def test_ks_fails_kq_succeeds(monkeypatch):
    calls: list[str] = []
    _patch_prices(monkeypatch, {"^KS11": _bench(), "X.KQ": _series(100.0, 110.0, BENCH_DATES)}, calls)
    result = await pending_recompute.recompute_pending_alphas(
        [_pending("X.KS", MON)], basis="variable", today=TODAY,
    )
    assert len(result.rows) == 1 and not result.failures
    row = result.rows[0]
    assert row.ticker == "X.KS"  # stored ticker preserved
    assert row.price_symbol == "X.KQ"
    assert result.n_alt_symbol_tickers == 1 and result.n_tickers_fetched == 1
    assert row.alpha_return == pytest.approx(0.10 - 0.01)
    assert calls == ["^KS11", "X.KS", "X.KQ"]


async def test_both_suffixes_fail_counted(monkeypatch):
    _patch_prices(monkeypatch, {"^KS11": _bench()})
    result = await pending_recompute.recompute_pending_alphas(
        [_pending("Z.KS", MON)], basis="variable", today=TODAY,
    )
    assert not result.rows
    assert [f.reason for f in result.failures] == [pending_recompute.REASON_PRICE_UNAVAILABLE]


async def test_benchmark_unavailable(monkeypatch):
    _patch_prices(monkeypatch, {})
    result = await pending_recompute.recompute_pending_alphas(
        [_pending("X.KS", MON)], basis="variable", today=TODAY,
    )
    assert [f.reason for f in result.failures] == [pending_recompute.REASON_BENCHMARK_UNAVAILABLE]


async def test_eval_date_is_first_common_close_not_latest(monkeypatch):
    # Bars exist on Fri 09-18 (before not_before), Mon 09-21 and Tue 09-22.
    prices = {MON: 100.0, date(2026, 9, 18): 105.0, date(2026, 9, 21): 110.0, date(2026, 9, 22): 120.0}
    bench = {MON: 1000.0, date(2026, 9, 18): 1005.0, date(2026, 9, 21): 1010.0, date(2026, 9, 22): 1020.0}
    _patch_prices(monkeypatch, {"^KS11": bench, "X.KS": prices})
    result = await pending_recompute.recompute_pending_alphas(
        [_pending("X.KS", MON)], basis="variable", today=TODAY,
    )
    row = result.rows[0]
    assert row.eval_date == date(2026, 9, 21)
    assert row.holding_days == 7
    # Production's helper would have picked the LATEST close -- pin the difference.
    assert auto_resolve._common_eval_date("X.KS", "^KS11", prices, bench, date(2026, 9, 19)) == date(2026, 9, 22)


async def test_holiday_gap_and_benchmark_only_dates_skipped(monkeypatch):
    prices = {MON: 100.0, date(2026, 9, 22): 110.0}
    bench = {MON: 1000.0, date(2026, 9, 21): 1005.0, date(2026, 9, 22): 1010.0}  # 09-21 benchmark-only
    _patch_prices(monkeypatch, {"^KS11": bench, "X.KS": prices})
    result = await pending_recompute.recompute_pending_alphas(
        [_pending("X.KS", MON)], basis="variable", today=TODAY,
    )
    assert result.rows[0].eval_date == date(2026, 9, 22)


async def test_in_progress_bar_not_used_and_guard_gets_stored_ticker(monkeypatch):
    seen: list[tuple] = []

    def guard(ticker, d):
        seen.append((ticker, d))
        return ticker.endswith(".KS") and d == TODAY

    monkeypatch.setattr(pending_recompute, "session_in_progress", guard)
    prices = {MON: 100.0, TODAY: 110.0}
    bench = {MON: 1000.0, TODAY: 1010.0}
    _patch_prices(monkeypatch, {"^KS11": bench, "X.KQ": prices})  # only the .KQ symbol has bars
    result = await pending_recompute.recompute_pending_alphas(
        [_pending("X.KS", MON)], basis="variable", today=TODAY,
    )
    assert [f.reason for f in result.failures] == [pending_recompute.REASON_NOT_MATURED]
    assert ("X.KS", TODAY) in seen  # stored ticker, never the .KQ fetch symbol
    assert all(t != "X.KQ" for t, _ in seen)


async def test_no_entry_close(monkeypatch):
    bench = _bench()
    bench[MON] = math.nan
    _patch_prices(monkeypatch, {"^KS11": bench, "X.KS": _series(100.0, 110.0, BENCH_DATES)})
    result = await pending_recompute.recompute_pending_alphas(
        [_pending("X.KS", MON)], basis="variable", today=TODAY,
    )
    assert [f.reason for f in result.failures] == [pending_recompute.REASON_NO_ENTRY_CLOSE]


async def test_horizon_basis_uses_nth_trading_day(monkeypatch):
    all_days = [MON + timedelta(days=i) for i in range(0, 20)]
    days = [d for d in all_days if d.weekday() < 5]
    prices = {d: 100.0 + i for i, d in enumerate(days)}
    bench = {d: 1000.0 for d in days}
    _patch_prices(monkeypatch, {"^KS11": bench, "X.KS": prices})
    result = await pending_recompute.recompute_pending_alphas(
        [_pending("X.KS", MON)], basis="horizon", today=TODAY,
    )
    assert result.rows[0].eval_date == days[10]  # 10th trading day after entry
    # Too few bars -> not matured
    _patch_prices(monkeypatch, {"^KS11": _bench(), "X.KS": _series(100.0, 110.0, BENCH_DATES)})
    short = await pending_recompute.recompute_pending_alphas(
        [_pending("X.KS", MON)], basis="horizon", today=TODAY,
    )
    assert [f.reason for f in short.failures] == [pending_recompute.REASON_NOT_MATURED]


# ---------------------------------------------------------------------------
# entry_paired --recompute-pending wiring
# ---------------------------------------------------------------------------


def test_build_where_default_sql_unchanged():
    where, params = entry_paired._build_where("domestic", "live", "real", None, None)
    assert where == (
        "status = 'resolved' AND rating IN ('Buy', 'Hold') AND ticker LIKE '%.KS' AND is_backtest = false "
        "AND account_mode = $1 AND (analysts_failed IS NULL OR analysts_failed = 0) AND user_id = $2"
    )
    assert params == ["real", "default"]


def test_build_where_rejects_unknown_status():
    with pytest.raises(ValueError):
        entry_paired._build_where("domestic", "live", "real", None, None, status="resolved' OR '1'='1")


def _combo_conn():
    """Buy on 09-14 (baseline) + a pending Hold(flat) the same day => a new paired day."""
    baseline = [_baseline(MON, "Buy", 0.02, ticker="BUY.KS")]
    pending = [_pending("X.KS", MON)]
    return FakeConn(baseline=baseline, pending=pending)


async def test_compute_recompute_end_to_end_and_no_writes(monkeypatch):
    conn = _combo_conn()
    _patch_db(monkeypatch, conn)
    _patch_prices(monkeypatch, {"^KS11": _bench(), "X.KQ": _series(100.0, 110.0, BENCH_DATES)})
    report = await entry_paired.compute_entry_paired_alpha_with_recompute(
        market="domestic", scope="live", account_mode="real",
    )

    assert report.baseline.n_paired_days == 0  # Buy-only day
    assert report.combined.n_paired_days == 1  # recomputed Hold(flat) completes the pair
    assert report.combined.mean_daily_delta == pytest.approx(0.02 - 0.09)
    assert report.n_recomputed_rows == 1 and report.n_recomputed_hold_flat == 1
    assert report.n_alt_symbol_tickers == 1 and report.n_recomputed_alt_rows == 1
    assert report.holding_days_min == report.holding_days_max == 7
    # Read-only snapshot, and not one write statement anywhere.
    assert conn.transaction_kwargs == [{"isolation": "repeatable_read", "readonly": True}]
    for query, _ in conn.queries:
        assert not any(kw in query.upper() for kw in ("INSERT", "UPDATE", "DELETE"))
    assert not hasattr(conn, "execute")  # the fake offers no write entry point at all


def test_new_code_has_no_write_or_reflection_path():
    """Static freeze guard on CODE (not docstrings): neither module may import,
    name or call the production write/reflection functions."""
    import ast
    import inspect

    forbidden = {"resolve_outcome", "set_horizon_outcome", "reflect", "execute", "executemany"}
    for module in (pending_recompute, entry_paired):
        tree = ast.parse(inspect.getsource(module))
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                names.add(node.id)
            elif isinstance(node, ast.Attribute):
                names.add(node.attr)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                names.update(a.name.split(".")[-1] for a in node.names)
        assert not (names & forbidden), f"{module.__name__} references {sorted(names & forbidden)}"


async def test_resolved_sibling_shadows_pending(monkeypatch):
    conn = FakeConn(
        baseline=[_baseline(MON, "Buy", 0.02, ticker="BUY.KS")],
        pending=[_pending("X.KS", MON)],
        resolved_keys=[("X.KS", MON)],  # e.g. resolved sibling whose alpha was NaN -> absent from baseline
    )
    _patch_db(monkeypatch, conn)
    calls: list[str] = []
    _patch_prices(monkeypatch, {"^KS11": _bench()}, calls)
    report = await entry_paired.compute_entry_paired_alpha_with_recompute(
        market="domestic", scope="live", account_mode="real",
    )
    assert report.n_shadowed_by_resolved == 1
    assert report.n_recomputed_rows == 0
    assert report.combined.n_buy_rows == report.baseline.n_buy_rows
    assert "X.KS" not in calls and "X.KQ" not in calls  # never even fetched
    # The key set is taken BEFORE any alpha/NaN filter, so a resolved twin that fell out of the
    # baseline still shadows its pending row.
    keys_query = next(q for q, _ in conn.queries if q.strip().startswith("SELECT DISTINCT ticker"))
    assert "alpha_return" not in keys_query and "NaN" not in keys_query


async def test_pending_query_shape(monkeypatch):
    conn = _combo_conn()
    _patch_db(monkeypatch, conn)
    _patch_prices(monkeypatch, {"^KS11": _bench(), "X.KQ": _series(100.0, 110.0, BENCH_DATES)})
    await entry_paired.compute_entry_paired_alpha_with_recompute(
        market="domestic", scope="live", account_mode="real",
    )
    pending_q = next((q, a) for q, a in conn.queries if "status = 'pending'" in q)
    query, params = pending_q
    assert "DISTINCT ON (ticker, trade_date)" in query and "created_at DESC" in query
    assert "is_backtest = false" in query
    assert params[-1] == date(2026, 9, 23)  # today - memory_min_holding_days (5)
    assert f"trade_date <= ${len(params)}" in query  # placeholder number matches the appended cutoff


async def test_recomputed_hold_classification_counters(monkeypatch):
    conn = FakeConn(pending=[
        _pending("F.KS", MON, held=False),
        {"ticker": "H.KS", "trade_date": MON, "rating": "Hold", "held_at_decision": True,
         "held_at_decision_inferred": None, "created_at": None},
        {"ticker": "U.KS", "trade_date": MON, "rating": "Hold", "held_at_decision": None,
         "held_at_decision_inferred": None, "created_at": None},
        _pending("B.KS", MON, rating="Buy"),
    ])
    _patch_db(monkeypatch, conn)
    series = _series(100.0, 110.0, BENCH_DATES)
    _patch_prices(monkeypatch, {"^KS11": _bench(), **{f"{t}.KS": series for t in "FHUB"}})
    report = await entry_paired.compute_entry_paired_alpha_with_recompute(
        market="domestic", scope="live", account_mode="real",
    )
    assert (report.n_recomputed_hold_flat, report.n_recomputed_hold_held,
            report.n_recomputed_hold_unknown, report.n_recomputed_buy) == (1, 1, 1, 1)


async def test_not_matured_counted_separately_from_failures(monkeypatch):
    conn = FakeConn(pending=[_pending("X.KS", MON), _pending("Z.KS", MON)])
    _patch_db(monkeypatch, conn)
    _patch_prices(monkeypatch, {"^KS11": _bench(), "X.KS": {MON: 100.0}})  # X: no later bar; Z: no data
    report = await entry_paired.compute_entry_paired_alpha_with_recompute(
        market="domestic", scope="live", account_mode="real",
    )
    assert report.n_not_matured == 1
    assert report.n_recompute_failed == 1
    assert report.failed_tickers == ["Z.KS"]


async def test_combined_p_value_independent_of_row_order(monkeypatch):
    def run(baseline):
        conn = FakeConn(baseline=baseline, pending=[_pending("X.KS", MON)])
        _patch_db(monkeypatch, conn)
        _patch_prices(monkeypatch, {"^KS11": _bench(), "X.KS": _series(100.0, 110.0, BENCH_DATES)})
        return entry_paired.compute_entry_paired_alpha_with_recompute(
            market="domestic", scope="live", account_mode="real",
        )

    base = [_baseline(MON, "Buy", 0.02, "A.KS"), _baseline(MON, "Buy", -0.01, "C.KS"),
            _baseline(date(2026, 9, 15), "Buy", 0.03, "D.KS"), _baseline(date(2026, 9, 15), "Hold", 0.01, "E.KS", held=False)]
    r1 = await run(base)
    r2 = await run(list(reversed(base)))
    assert r1.combined.p_value == r2.combined.p_value
    assert r1.combined.mean_daily_delta == pytest.approx(r2.combined.mean_daily_delta)


async def test_rejects_bad_combinations():
    for kwargs, fragment in (
        ({"market": None, "scope": "live", "account_mode": "real"}, "domestic-only"),
        ({"market": "overseas", "scope": "live", "account_mode": "real"}, "domestic-only"),
        ({"market": "domestic", "scope": "backtest", "account_mode": "real"}, "scope='live'"),
        ({"market": "domestic", "scope": None, "account_mode": "real"}, "scope='live'"),  # baseline would pool backtest
        ({"market": "domestic", "scope": "live", "account_mode": None}, "explicit account_mode"),
    ):
        with pytest.raises(ValueError, match=fragment):
            await entry_paired.compute_entry_paired_alpha_with_recompute(**kwargs)


async def test_non_valueerror_ticker_fetch_is_fetch_error_not_missing_data(monkeypatch):
    async def fake(symbol, start, end):
        if symbol == "^KS11":
            return _bench()
        raise RuntimeError("429 Too Many Requests")

    monkeypatch.setattr(pending_recompute, "get_price_series", fake)
    result = await pending_recompute.recompute_pending_alphas(
        [_pending("X.KS", MON)], basis="variable", today=TODAY,
    )
    assert [f.reason for f in result.failures] == ["fetch_error:RuntimeError"]


async def test_benchmark_non_valueerror_is_reraised(monkeypatch):
    async def fake(symbol, start, end):
        raise RuntimeError("429 Too Many Requests")

    monkeypatch.setattr(pending_recompute, "get_price_series", fake)
    with pytest.raises(RuntimeError):
        await pending_recompute.recompute_pending_alphas(
            [_pending("X.KS", MON)], basis="variable", today=TODAY,
        )


async def test_fetch_error_counts_as_hard_failure_in_report(monkeypatch):
    conn = FakeConn(pending=[_pending("X.KS", MON)])
    _patch_db(monkeypatch, conn)

    async def fake(symbol, start, end):
        if symbol == "^KS11":
            return _bench()
        raise RuntimeError("boom")

    monkeypatch.setattr(pending_recompute, "get_price_series", fake)
    report = await entry_paired.compute_entry_paired_alpha_with_recompute(
        market="domestic", scope="live", account_mode="real",
    )
    assert report.n_recompute_failed == 1 and report.n_not_matured == 0
    assert report.failures_by_reason == {"fetch_error:RuntimeError": 1}
    assert report.failed_tickers == ["X.KS"]


def test_render_shows_both_columns_and_notice():
    conn_report = entry_paired.EntryPairedRecomputeReport(
        baseline=entry_paired._build_paired_report(
            [], basis="variable", market="domestic", scope="live", account_mode="real"),
        combined=entry_paired._build_paired_report(
            [], basis="variable", market="domestic", scope="live", account_mode="real"),
        eval_rule="rule", n_pending_past_cutoff=3, n_shadowed_by_resolved=1, n_recomputed_rows=2,
        n_recomputed_buy=0, n_recomputed_hold_flat=2, n_recomputed_hold_held=0, n_recomputed_hold_unknown=0,
        n_recompute_failed=0, n_not_matured=1, failures_by_reason={"not_matured": 1},
        failed_tickers=[], n_alt_symbol_tickers=3, n_tickers_fetched=4, n_recomputed_alt_rows=2, holding_days_min=5, holding_days_median=6.0, holding_days_max=7,
    )
    text = entry_paired.render_entry_paired_recompute(conn_report)
    assert "baseline(resolved only)" in text and "recomputed(resolved+recomputed)" in text
    assert "not written to DB, no reflection generated" in text
    assert "n_not_matured=1" in text
    assert "alt-symbol: tickers 3/4, recomputed rows 2/2" in text  # tickers and rows reported separately


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _argv(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["app.audit", *args])


def test_cli_rejects_flag_on_other_commands_and_bad_combos(monkeypatch):
    for args in (
        ("report", "--recompute-pending"),
        ("entry-paired", "--recompute-pending"),  # no --market
        ("entry-paired", "--recompute-pending", "--market", "overseas"),
        ("entry-paired", "--recompute-pending", "--market", "domestic"),  # no --scope
        ("entry-paired", "--recompute-pending", "--market", "domestic", "--scope", "backtest",
         "--account-mode", "real"),
        ("entry-paired", "--recompute-pending", "--market", "domestic", "--scope", "live"),  # no --account-mode
        ("entry-paired", "--recompute-pending", "--market", "domestic", "--scope", "live",
         "--account-mode", "real", "--start-date", "2026-09-01"),
    ):
        _argv(monkeypatch, *args)
        with pytest.raises(SystemExit):
            audit_main._parse_args()


async def test_cli_default_path_never_touches_recompute(monkeypatch, capsys):
    async def boom(**kwargs):
        raise AssertionError("recompute path must not run without the flag")

    class _Report:
        pass

    async def fake_compute(**kwargs):
        return _Report()

    monkeypatch.setattr(audit_main, "compute_entry_paired_alpha_with_recompute", boom)
    monkeypatch.setattr(audit_main, "compute_entry_paired_alpha", fake_compute)
    monkeypatch.setattr(audit_main, "render_entry_paired_alpha", lambda r: "DEFAULT-RENDER")
    _argv(monkeypatch, "entry-paired", "--market", "domestic", "--scope", "live", "--account-mode", "real")
    await audit_main._main()
    assert capsys.readouterr().out.strip() == "DEFAULT-RENDER"


async def test_cli_flag_routes_to_recompute(monkeypatch, capsys):
    seen = {}

    async def fake_recompute(**kwargs):
        seen.update(kwargs)
        return "REPORT"

    monkeypatch.setattr(audit_main, "compute_entry_paired_alpha_with_recompute", fake_recompute)
    monkeypatch.setattr(audit_main, "render_entry_paired_recompute", lambda r: f"RENDER:{r}")
    _argv(monkeypatch, "entry-paired", "--market", "domestic", "--scope", "live",
          "--account-mode", "real", "--recompute-pending")
    await audit_main._main()
    assert capsys.readouterr().out.strip() == "RENDER:REPORT"
    assert seen == {"market": "domestic", "scope": "live", "account_mode": "real", "basis": "variable"}


async def test_alt_symbol_counts_tickers_and_rows_separately(monkeypatch):
    # One KOSDAQ ticker with TWO pending rows: the early one matures, the late one does not.
    late = date(2026, 9, 21)
    prices = {MON: 100.0, late: 100.0, date(2026, 9, 22): 110.0, date(2026, 9, 18): 105.0}
    bench = {MON: 1000.0, late: 1000.0, date(2026, 9, 22): 1010.0, date(2026, 9, 18): 1005.0}
    _patch_prices(monkeypatch, {"^KS11": bench, "X.KQ": prices})  # only the .KQ symbol has bars
    result = await pending_recompute.recompute_pending_alphas(
        [_pending("X.KS", MON), _pending("X.KS", late)], basis="variable", today=TODAY,
    )
    assert result.n_alt_symbol_tickers == 1 and result.n_tickers_fetched == 1  # one ticker ...
    assert len(result.rows) == 1  # ... but only one of its two rows matured
    assert [f.reason for f in result.failures] == [pending_recompute.REASON_NOT_MATURED]
