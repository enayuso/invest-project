"""app.audit.pending_health (2026-09-29). No Postgres, no network by default;
`--probe` mode patches `pending_recompute.get_price_series` the same way
tests/audit/test_pending_recompute.py does."""

from __future__ import annotations

import sys
from datetime import date

import pytest

from app.audit import __main__ as audit_main
from app.audit import pending_health, pending_recompute


class FakeTxn:
    def __init__(self, conn, kwargs):
        conn.transaction_kwargs.append(kwargs)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class FakeConn:
    def __init__(self, daily_rows=None, last_resolved_rows=None, overdue_rows=None):
        self.daily_rows = daily_rows or []
        self.last_resolved_rows = last_resolved_rows or []
        self.overdue_rows = overdue_rows or []
        self.queries: list[tuple] = []
        self.transaction_kwargs: list[dict] = []

    def transaction(self, **kwargs):
        return FakeTxn(self, kwargs)

    async def fetch(self, query, *args):
        self.queries.append((query, args))
        if "GROUP BY 1, 2, 3" in query:
            return self.daily_rows
        if "max(resolved_on)" in query:
            return self.last_resolved_rows
        if "status = 'pending'" in query:
            return self.overdue_rows
        raise AssertionError(f"unexpected query: {query}")


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


def _daily(trade_date, market, mode, total, pending, eligible, overdue):
    return {
        "trade_date": trade_date, "market": market, "account_mode": mode,
        "n_total": total, "n_pending": pending, "n_eligible": eligible, "n_overdue_pending": overdue,
    }


def _last_resolved(market, mode, last):
    return {"market": market, "account_mode": mode, "last_resolved_on": last}


def _overdue_row(ticker, trade_date, mode, rating="Hold", held=False):
    return {
        "ticker": ticker, "trade_date": trade_date, "rating": rating,
        "held_at_decision": held, "held_at_decision_inferred": None, "account_mode": mode,
    }


DOM_TODAY = date(2026, 9, 29)
OVS_TODAY = date(2026, 9, 28)
DOM_CUTOFF = date(2026, 9, 24)  # DOM_TODAY - 5
OVS_CUTOFF = date(2026, 9, 23)


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    def fake_local_today(market):
        return DOM_TODAY if market == "domestic" else OVS_TODAY

    monkeypatch.setattr(pending_health, "local_today", fake_local_today)
    monkeypatch.setattr(pending_health, "get_current_user_id", lambda: "default")
    monkeypatch.setattr(pending_recompute, "session_in_progress", lambda t, d: False)


def _patch_db(monkeypatch, conn):
    async def _pool():
        return FakePool(conn)

    monkeypatch.setattr(pending_health, "get_pool", _pool)


def _patch_prices(monkeypatch, table: dict):
    async def fake(symbol, start, end):
        entry = table.get(symbol)
        if entry is None:
            raise ValueError(f"no price history returned for {symbol!r}")
        if isinstance(entry, Exception):
            raise entry
        return entry

    monkeypatch.setattr(pending_recompute, "get_price_series", fake)


# ---------------------------------------------------------------------------
# cutoffs / SQL shape
# ---------------------------------------------------------------------------


def test_overdue_cutoffs_are_per_market():
    cutoffs = pending_health.overdue_cutoffs()
    assert cutoffs == {"domestic": DOM_CUTOFF, "overseas": OVS_CUTOFF}


async def test_daily_query_binds_both_market_cutoffs_and_has_no_rating_filter(monkeypatch):
    conn = FakeConn()
    _patch_db(monkeypatch, conn)
    await pending_health.compute_pending_health(market="domestic", account_mode="real")
    query, params = next(q for q in conn.queries if "GROUP BY 1, 2, 3" in q[0])
    assert params[0] == DOM_CUTOFF and params[1] == OVS_CUTOFF
    assert "is_backtest = false" in query and "user_id =" in query
    assert "rating IN" not in query and "DISTINCT ON" not in query
    assert "ticker LIKE '%.KS'" in query  # market filter appended


async def test_market_and_mode_filters_use_correct_placeholder_numbers(monkeypatch):
    conn = FakeConn()
    _patch_db(monkeypatch, conn)
    await pending_health.compute_pending_health(market="overseas", account_mode="paper")
    query, params = next(q for q in conn.queries if "GROUP BY 1, 2, 3" in q[0])
    assert "ticker NOT LIKE '%.KS'" in query
    assert "account_mode = $4" in query and "user_id = $5" in query
    assert params[-2] == "paper"  # account_mode bound before user_id


def test_buckets_group_by_iso_week_and_partial_flag():
    rows = [
        _daily(date(2026, 9, 14), "domestic", "real", 20, 5, 20, 5),
        _daily(date(2026, 9, 18), "domestic", "real", 20, 0, 20, 0),
        _daily(date(2026, 9, 28), "domestic", "real", 10, 10, 0, 0),  # this week, not yet eligible
    ]
    cutoffs = {"domestic": DOM_CUTOFF, "overseas": OVS_CUTOFF}
    buckets = pending_health._build_buckets(rows, cutoffs=cutoffs)
    by_week = {b.week: b for b in buckets}
    assert by_week["2026-W38"].n_total == 40 and by_week["2026-W38"].n_eligible == 40
    assert by_week["2026-W38"].partial is False
    assert by_week["2026-W40"].partial is True  # week_start 2026-09-28 + 6 > cutoff 2026-09-24


def test_warn_triggers_at_threshold_not_below():
    cutoffs = {"domestic": DOM_CUTOFF, "overseas": OVS_CUTOFF}
    hit = pending_health._build_buckets(
        [_daily(date(2026, 9, 14), "domestic", "real", 125, 45, 125, 45)], cutoffs=cutoffs,
    )[0]
    miss = pending_health._build_buckets(
        [_daily(date(2026, 9, 14), "domestic", "real", 118, 7, 118, 7)], cutoffs=cutoffs,
    )[0]
    assert pending_health._bucket_warns(hit, warn_pct=15.0, min_eligible=30, stale_groups=set()) is True
    assert pending_health._bucket_warns(miss, warn_pct=15.0, min_eligible=30, stale_groups=set()) is False


def test_warn_suppressed_below_min_eligible_but_pct_still_shown():
    cutoffs = {"domestic": DOM_CUTOFF, "overseas": OVS_CUTOFF}
    b = pending_health._build_buckets(
        [_daily(date(2026, 9, 14), "domestic", "real", 20, 10, 20, 10)], cutoffs=cutoffs,
    )[0]
    assert b.overdue_pct == pytest.approx(50.0)
    assert pending_health._bucket_warns(b, warn_pct=15.0, min_eligible=30, stale_groups=set()) is False


def test_stale_group_detected_and_suppresses_week_warnings():
    daily = [_daily(date(2026, 8, 24), "domestic", "paper", 60, 53, 60, 53)]
    last_resolved = [_last_resolved("domestic", "paper", date(2026, 8, 31))]
    cutoffs = {"domestic": DOM_CUTOFF, "overseas": OVS_CUTOFF}
    freshness = pending_health._build_freshness(daily, last_resolved, cutoffs=cutoffs)
    f = freshness[0]
    assert f.stale is True and f.last_resolved_on == date(2026, 8, 31)
    stale_groups = {(x.market, x.account_mode) for x in freshness if x.stale}
    bucket = pending_health._build_buckets(daily, cutoffs=cutoffs)[0]
    assert pending_health._bucket_warns(bucket, warn_pct=15.0, min_eligible=30, stale_groups=stale_groups) is False


def test_freshness_not_stale_when_resolved_after_cutoff():
    daily = [_daily(date(2026, 9, 20), "domestic", "real", 50, 5, 50, 5)]
    last_resolved = [_last_resolved("domestic", "real", date(2026, 9, 28))]
    cutoffs = {"domestic": DOM_CUTOFF, "overseas": OVS_CUTOFF}
    f = pending_health._build_freshness(daily, last_resolved, cutoffs=cutoffs)[0]
    assert f.stale is False


async def test_single_readonly_snapshot_transaction(monkeypatch):
    conn = FakeConn()
    _patch_db(monkeypatch, conn)
    await pending_health.compute_pending_health(market="domestic", account_mode="real", probe=False)
    assert conn.transaction_kwargs == [{"isolation": "repeatable_read", "readonly": True}]


async def test_default_mode_makes_no_network_calls(monkeypatch):
    conn = FakeConn(daily_rows=[_daily(date(2026, 9, 14), "domestic", "real", 10, 2, 10, 2)])
    _patch_db(monkeypatch, conn)

    async def boom(*a, **k):
        raise AssertionError("no network call expected without --probe")

    monkeypatch.setattr(pending_recompute, "get_price_series", boom)
    report = await pending_health.compute_pending_health(market="domestic", account_mode="real", probe=False)
    assert report.probe is None


# ---------------------------------------------------------------------------
# probe
# ---------------------------------------------------------------------------


def _bench():
    return {date(2026, 9, 14): 1000.0, date(2026, 9, 21): 1010.0, date(2026, 9, 22): 1020.0}


async def test_probe_classifies_all_categories(monkeypatch):
    entry = date(2026, 9, 14)
    conn = FakeConn(overdue_rows=[
        _overdue_row("ALT.KS", entry, "real"),  # .KS fails, .KQ succeeds -> alt_suffix
        _overdue_row("STORED.KS", entry, "real"),  # .KS succeeds -> stored_symbol
        _overdue_row("SHORT.KS", entry, "real"),  # matures too soon -> not_matured
        _overdue_row("NOENTRY.KS", entry, "real"),  # no entry close -> no_entry_close
        _overdue_row("BOTHFAIL.KS", entry, "real"),  # both suffixes fail -> price_unavailable
        _overdue_row("BOOM.KS", entry, "real"),  # non-ValueError -> fetch_error:RuntimeError
    ])
    _patch_db(monkeypatch, conn)
    series = {entry: 100.0, date(2026, 9, 21): 110.0, date(2026, 9, 22): 120.0}
    no_entry = {date(2026, 9, 21): 110.0}
    short_series = {entry: 100.0}
    _patch_prices(monkeypatch, {
        "^KS11": _bench(),
        "ALT.KQ": series,
        "STORED.KS": series,
        "SHORT.KS": short_series,
        "NOENTRY.KS": no_entry,
        "BOOM.KS": RuntimeError("429"),
    })
    report = await pending_health.compute_pending_health(market="domestic", account_mode="real", probe=True)
    by_ticker = {row.ticker: row for row in report.probe.rows}
    assert by_ticker["ALT.KS"].category == pending_health.CATEGORY_ALT_SUFFIX
    assert by_ticker["ALT.KS"].price_symbol == "ALT.KQ"
    assert by_ticker["STORED.KS"].category == pending_health.CATEGORY_STORED_SYMBOL
    assert by_ticker["SHORT.KS"].category == pending_recompute.REASON_NOT_MATURED
    assert by_ticker["NOENTRY.KS"].category == pending_recompute.REASON_NO_ENTRY_CLOSE
    assert by_ticker["BOTHFAIL.KS"].category == pending_recompute.REASON_PRICE_UNAVAILABLE
    assert by_ticker["BOOM.KS"].category == "fetch_error:RuntimeError"


async def test_probe_groups_by_account_mode_without_key_collision(monkeypatch):
    entry = date(2026, 9, 14)
    conn = FakeConn(overdue_rows=[
        _overdue_row("005930.KS", entry, "paper"),
        _overdue_row("005930.KS", entry, "real"),
    ])
    _patch_db(monkeypatch, conn)
    _patch_prices(monkeypatch, {"^KS11": _bench(), "005930.KS": {entry: 100.0, date(2026, 9, 21): 110.0}})
    report = await pending_health.compute_pending_health(market="domestic", account_mode=None, probe=True)
    modes = {row.account_mode for row in report.probe.rows}
    assert modes == {"paper", "real"}
    assert len(report.probe.rows) == 2


async def test_probe_ignores_fallback_gate_but_reports_gate_covers(monkeypatch):
    entry_old, entry_new = date(2026, 9, 14), date(2026, 9, 22)
    conn = FakeConn(overdue_rows=[
        _overdue_row("ALT.KS", entry_old, "real"),
        _overdue_row("ALT.KS", entry_new, "real"),
    ])
    _patch_db(monkeypatch, conn)
    series = {entry_old: 100.0, entry_new: 100.0, date(2026, 9, 28): 110.0}
    _patch_prices(monkeypatch, {"^KS11": {entry_old: 1000.0, entry_new: 1000.0, date(2026, 9, 28): 1010.0}, "ALT.KQ": series})
    monkeypatch.setattr(pending_health.settings, "memory_krx_fallback_since", date(2026, 9, 20))
    try:
        report = await pending_health.compute_pending_health(market="domestic", account_mode="real", probe=True)
    finally:
        monkeypatch.setattr(pending_health.settings, "memory_krx_fallback_since", None)
    by_date = {row.trade_date: row for row in report.probe.rows}
    assert by_date[entry_old].gate_covers is False
    assert by_date[entry_new].gate_covers is True
    assert all(row.category == pending_health.CATEGORY_ALT_SUFFIX for row in report.probe.rows)  # fetched regardless of gate


async def test_probe_isolates_a_benchmark_fetch_failure_to_its_own_mode(monkeypatch):
    # A total benchmark outage (rate limit, network) inside recompute_pending_alphas raises rather
    # than returning per-row failures -- one mode's outage must not blank out the whole report,
    # including the DB-only weekly table already computed, or the other mode's classification.
    entry = date(2026, 9, 14)
    conn = FakeConn(
        daily_rows=[_daily(entry, "domestic", "real", 10, 2, 10, 2)],
        overdue_rows=[_overdue_row("BAD.KS", entry, "paper"), _overdue_row("GOOD.KS", entry, "real")],
    )
    _patch_db(monkeypatch, conn)

    async def fake(symbol, start, end):
        if symbol == "^KS11":
            raise RuntimeError("429 Too Many Requests")
        return {entry: 100.0, date(2026, 9, 21): 110.0}

    monkeypatch.setattr(pending_recompute, "get_price_series", fake)
    report = await pending_health.compute_pending_health(market="domestic", account_mode=None, probe=True)
    assert len(report.buckets) == 1  # DB-only table still rendered
    by_ticker = {row.ticker: row for row in report.probe.rows}
    assert by_ticker["BAD.KS"].category == "fetch_error:benchmark:RuntimeError"
    # Same shared benchmark, so both groups fail the same way here -- this proves the isolation
    # doesn't crash the whole report, not that different modes get different outcomes.
    assert by_ticker["GOOD.KS"].category == "fetch_error:benchmark:RuntimeError"


async def test_overdue_rows_query_has_no_rating_filter_or_dedup(monkeypatch):
    conn = FakeConn()
    _patch_db(monkeypatch, conn)
    await pending_health.compute_pending_health(market="domestic", account_mode="real", probe=True)
    query, params = next(q for q in conn.queries if "held_at_decision" in q[0])
    assert "rating IN" not in query and "DISTINCT ON" not in query
    assert "ORDER BY account_mode, trade_date, ticker" in query
    assert "held_at_decision" in query and "held_at_decision_inferred" in query
    assert params[0] == DOM_CUTOFF


async def test_probe_false_runs_exactly_two_queries_inside_one_transaction(monkeypatch):
    conn = FakeConn()
    _patch_db(monkeypatch, conn)
    await pending_health.compute_pending_health(market="domestic", account_mode="real", probe=False)
    assert len(conn.queries) == 2
    assert conn.transaction_kwargs == [{"isolation": "repeatable_read", "readonly": True}]


async def test_probe_true_runs_exactly_three_queries_inside_one_transaction(monkeypatch):
    conn = FakeConn()
    _patch_db(monkeypatch, conn)
    await pending_health.compute_pending_health(market="domestic", account_mode="real", probe=True)
    assert len(conn.queries) == 3
    assert conn.transaction_kwargs == [{"isolation": "repeatable_read", "readonly": True}]


async def test_probe_truncates_long_stored_symbol_listing(monkeypatch):
    entry = date(2026, 9, 14)
    rows = [_overdue_row(f"{100000 + i}.KS", entry, "real") for i in range(25)]
    conn = FakeConn(overdue_rows=rows)
    _patch_db(monkeypatch, conn)
    series = {entry: 100.0, date(2026, 9, 21): 110.0}
    table = {"^KS11": _bench()}
    for row in rows:
        table[row["ticker"]] = series
    _patch_prices(monkeypatch, table)
    report = await pending_health.compute_pending_health(market="domestic", account_mode="real", probe=True)
    text = pending_health.render_pending_health(report)
    assert "+5 more" in text


async def test_probe_rejects_non_domestic():
    with pytest.raises(ValueError):
        await pending_health.compute_pending_health(market=None, probe=True)
    with pytest.raises(ValueError):
        await pending_health.compute_pending_health(market="overseas", probe=True)


def test_no_write_or_reflection_path():
    import ast
    import inspect

    forbidden = {
        "resolve_outcome", "set_horizon_outcome", "reflect", "execute", "executemany",
        "_UNKNOWN_TICKER_CACHE",
    }
    tree = ast.parse(inspect.getsource(pending_health))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update(a.name.split(".")[-1] for a in node.names)
    assert not (names & forbidden), f"pending_health references {sorted(names & forbidden)}"
    assert "auto_resolve" not in names  # no import of the production write path


def test_render_contains_cutoffs_threshold_and_local_gate_caveat():
    report = pending_health.PendingHealthReport(
        market="domestic", account_mode="real", weeks=8, since=date(2026, 8, 4),
        cutoffs={"domestic": DOM_CUTOFF, "overseas": OVS_CUTOFF}, min_holding_days=5,
        warn_pct=15.0, min_eligible=30, krx_fallback_since_local=None,
        buckets=[pending_health.PendingHealthBucket(
            week="2026-W38", week_start=date(2026, 9, 14), market="domestic", account_mode="real",
            n_total=125, n_pending=45, n_eligible=125, n_overdue_pending=45, partial=False,
        )],
        freshness=[pending_health.ResolverFreshness(
            market="domestic", account_mode="real", last_resolved_on=date(2026, 9, 28),
            n_overdue_pending=45, stale=False,
        )],
    )
    text = pending_health.render_pending_health(report)
    assert "domestic<=2026-09-24" in text and "overseas<=2026-09-23" in text
    assert "LOCAL setting -- server value may differ" in text
    assert "None (OFF)" in text
    assert "!! WARNING 2026-W38 domestic/real" in text
    assert "resolver freshness" in text


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _argv(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["app.audit", *args])


def test_cli_rejects_bad_combinations(monkeypatch):
    for args in (
        ("report", "--probe"),
        ("report", "--warn-pct", "10"),
        ("pending-health", "--probe"),  # no --market
        ("pending-health", "--market", "overseas", "--probe"),
        ("pending-health", "--warn-pct", "0"),
        ("pending-health", "--warn-pct", "101"),
        ("pending-health", "--scope", "backtest"),
        ("pending-health", "--weeks", "0"),
        ("pending-health", "--ticker", "005930.KS"),
        ("pending-health", "--start-date", "2026-09-01"),
    ):
        _argv(monkeypatch, *args)
        with pytest.raises(SystemExit):
            audit_main._parse_args()


async def test_cli_defaults_and_routing(monkeypatch, capsys):
    seen = {}

    async def fake_compute(**kwargs):
        seen.update(kwargs)
        return "REPORT"

    monkeypatch.setattr(audit_main, "compute_pending_health", fake_compute)
    monkeypatch.setattr(audit_main, "render_pending_health", lambda r: f"RENDER:{r}")
    _argv(monkeypatch, "pending-health", "--market", "domestic", "--account-mode", "real")
    await audit_main._main()
    assert capsys.readouterr().out.strip() == "RENDER:REPORT"
    assert seen == {"market": "domestic", "account_mode": "real", "weeks": 8, "warn_pct": 15.0, "probe": False}


async def test_cli_custom_weeks_warn_probe(monkeypatch):
    seen = {}

    async def fake_compute(**kwargs):
        seen.update(kwargs)
        return "REPORT"

    monkeypatch.setattr(audit_main, "compute_pending_health", fake_compute)
    monkeypatch.setattr(audit_main, "render_pending_health", lambda r: "x")
    _argv(monkeypatch, "pending-health", "--market", "domestic", "--account-mode", "real",
          "--weeks", "4", "--warn-pct", "20", "--probe")
    await audit_main._main()
    assert seen == {"market": "domestic", "account_mode": "real", "weeks": 4, "warn_pct": 20.0, "probe": True}


async def test_screening_ic_weeks_default_unchanged(monkeypatch, capsys):
    from datetime import date as _date

    fixed_today = _date(2026, 9, 10)

    class _FixedClock:
        def today(self):
            return fixed_today

    seen = {}

    async def fake_compute_screening_ic(market, start, end, *, holding_days=None, account_mode=None):
        seen["start"] = start
        return {}

    import app.clock as clock_module
    monkeypatch.setattr(clock_module, "get_clock", lambda: _FixedClock())
    monkeypatch.setattr(audit_main, "compute_screening_ic", fake_compute_screening_ic)
    monkeypatch.setattr(audit_main, "render_screening_ic", lambda r: "x")
    _argv(monkeypatch, "screening-ic", "--market", "domestic")
    await audit_main._main()
    assert seen["start"] == fixed_today - __import__("datetime").timedelta(weeks=4)
