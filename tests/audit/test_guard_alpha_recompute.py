"""app.audit.guard_alpha's --recompute-pending mode (2026-09-29). No Postgres,
no network: the asyncpg pool is faked and `pending_recompute.get_price_series`
is patched with dict-backed series, same convention as
tests/audit/test_pending_recompute.py."""

from __future__ import annotations

import sys
from datetime import date, timedelta

import pytest

from app.audit import __main__ as audit_main
from app.audit import guard_alpha, pending_recompute

MON = date(2026, 9, 14)
TODAY = date(2026, 9, 28)
BENCH_DATES = [date(2026, 9, 18), date(2026, 9, 21), date(2026, 9, 22)]


def _series(entry_price: float, exit_price: float, dates_after=None, entry: date = MON) -> dict:
    out = {entry: entry_price}
    out.update({d: exit_price for d in (dates_after or BENCH_DATES)})
    return out


def _bench(dates_after=None) -> dict:
    return _series(1000.0, 1010.0, dates_after)


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


def _pending(ticker, trade_date, rating="Hold", held=False, transcript=None, gate_orig=None):
    return {
        "ticker": ticker, "trade_date": trade_date, "rating": rating,
        "held_at_decision": held, "held_at_decision_inferred": None,
        "full_transcript": transcript or {}, "created_at": None, "gate_orig_rating": gate_orig,
    }


def _baseline(trade_date, rating, signed_alpha, ticker="B", transcript=None, holding_days=7):
    return {
        "trade_date": trade_date, "ticker": ticker, "rating": rating,
        "full_transcript": transcript or {}, "holding_days": holding_days, "signed_alpha": signed_alpha,
    }


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for module in (pending_recompute,):
        monkeypatch.setattr(module, "session_in_progress", lambda t, d: False)
    monkeypatch.setattr(guard_alpha, "local_today", lambda market: TODAY)
    monkeypatch.setattr(guard_alpha, "get_current_user_id", lambda: "default")
    monkeypatch.setattr(guard_alpha, "_N_PERMUTATIONS", 200)


def _patch_prices(monkeypatch, table: dict[str, dict]):
    async def fake(symbol, start, end):
        if symbol not in table:
            raise ValueError(f"no price history returned for {symbol!r}")
        return table[symbol]

    monkeypatch.setattr(pending_recompute, "get_price_series", fake)


def _patch_db(monkeypatch, conn):
    async def _pool():
        return FakePool(conn)

    monkeypatch.setattr(guard_alpha, "get_pool", _pool)


# ---------------------------------------------------------------------------
# A: refactor -- _build_where / _validate_filters / _signed_alpha
# ---------------------------------------------------------------------------


def test_build_where_default_matches_legacy_sql():
    where, params, manual_excluded = guard_alpha._build_where(None, "domestic", "live", "real", None, None, None)
    assert where == (
        "status = 'resolved' AND ticker LIKE '%.KS' AND is_backtest = false AND account_mode = $1 "
        "AND decision_trigger IS DISTINCT FROM 'manual' "
        "AND (analysts_failed IS NULL OR analysts_failed = 0) AND user_id = $2"
    )
    assert params == ["real", "default"]
    assert manual_excluded is True


def test_build_where_rejects_unknown_status():
    with pytest.raises(ValueError):
        guard_alpha._build_where(None, None, None, None, None, None, None, status="resolved' OR '1'='1")


def test_build_where_manual_trigger():
    where, params, manual_excluded = guard_alpha._build_where(None, None, None, None, "manual", None, None)
    assert "decision_trigger = 'manual'" in where and "IS DISTINCT FROM" not in where
    assert manual_excluded is False


@pytest.mark.parametrize(
    "rating,gate,alpha,raw,held,held_inf,expected",
    [
        ("Hold", "Buy", 0.05, 0.02, None, None, 0.05),  # gate overrides rating branch
        ("Hold", "Sell", 0.05, 0.02, None, None, -0.05),
        ("Buy", None, 0.05, 0.02, None, None, 0.05),
        ("Overweight", None, 0.05, 0.02, None, None, 0.05),
        ("Sell", None, 0.05, 0.02, None, None, -0.05),
        ("Underweight", None, 0.05, 0.02, None, None, -0.05),
        ("Hold", None, 0.05, 0.02, True, None, 0.02),  # held -> raw
        ("Hold", None, 0.05, 0.02, False, None, -0.05),  # flat -> -alpha
        ("Hold", None, 0.05, 0.02, None, True, 0.02),  # falls back to inferred
        ("Hold", None, 0.05, 0.02, None, None, None),  # unknown held -> excluded
        ("Sell", None, None, 0.02, None, None, None),  # alpha missing -> None, not -None
        ("Hold", None, float("nan"), 0.02, False, None, None),  # NaN -> excluded
    ],
)
def test_signed_alpha_matches_sql_case_truth_table(rating, gate, alpha, raw, held, held_inf, expected):
    result = guard_alpha._signed_alpha(rating, gate, alpha, raw, held, held_inf)
    if expected is None:
        assert result is None
    else:
        assert result == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Statistical mechanics: power-gate crossing and BH m re-evaluation.
# Exercised directly through _build_report (no network) since it's the same
# function compute_guard_alpha_with_recompute calls on baseline vs combined
# rows -- this is what actually decides FINDING/NO_EFFECT/UNDERPOWERED.
# ---------------------------------------------------------------------------


def _synthetic_days(n: int, key: str, ticker_prefix: str):
    """n days, each with one violated + one clean row for `key`, alpha chosen
    so the delta is stable and non-zero."""
    rows = []
    for i in range(n):
        d = MON + timedelta(days=i)
        rows.append(_baseline(d, "Buy", 0.05, ticker=f"{ticker_prefix}V{i}", transcript={key: ["bad"]}))
        rows.append(_baseline(d, "Buy", 0.01, ticker=f"{ticker_prefix}C{i}", transcript={key: []}))
    return rows


def test_recompute_crosses_power_gate_flags_transition():
    key = "some_guard_violations"
    baseline_rows = _synthetic_days(19, key, "D")  # 19 paired days, 19 violated, 19 clean -> UNDERPOWERED
    baseline = guard_alpha._build_report(baseline_rows, basis="variable", manual_excluded=True)
    b_stats = next(s for s in baseline.key_stats if s.key == key)
    assert b_stats.verdict == "UNDERPOWERED"
    assert b_stats.n_paired_days == 19

    extra_day = MON + timedelta(days=19)
    combined_rows = baseline_rows + [
        _baseline(extra_day, "Buy", 0.05, ticker="D_V20", transcript={key: ["bad"]}),
        _baseline(extra_day, "Buy", 0.01, ticker="D_C20", transcript={key: []}),
    ]
    combined = guard_alpha._build_report(combined_rows, basis="variable", manual_excluded=True)
    c_stats = next(s for s in combined.key_stats if s.key == key)
    assert c_stats.n_paired_days == 20 and c_stats.n_violated == 20 and c_stats.n_clean == 20
    assert c_stats.verdict != "UNDERPOWERED"

    text = guard_alpha.render_guard_alpha_recompute(guard_alpha.GuardAlphaRecomputeReport(
        baseline=baseline, combined=combined, market="domestic", scope="live", account_mode="real",
        trigger=None, eval_rule="rule", n_pending_past_cutoff=1, n_shadowed_by_resolved=0,
        n_recomputed_rows=2, n_recomputed_unscorable=0, recomputed_rating_mix={"Buy": 2},
        recomputed_added={key: (1, 1)}, n_recompute_failed=0, n_not_matured=0, failures_by_reason={},
        failed_tickers=[], n_alt_symbol_tickers=0, n_tickers_fetched=2, n_recomputed_alt_rows=0,
        holding_days_min=7, holding_days_median=7.0, holding_days_max=7,
    ))
    assert "!! POWER GATE CROSSED ONLY WITH RECOMPUTED ROWS" in text


def _stats(key, p_value):
    return guard_alpha.GuardKeyStats(
        key=key, n_violated=25, n_clean=25, n_paired_days=25, paired_mean_delta=0.01,
        paired_ci95=(-0.01, 0.03), p_value=p_value, q_value=None,
        naive_mean_violated=0.01, naive_mean_clean=0.0, naive_delta=0.01,
    )


def test_bh_m_changes_q_of_an_unrelated_key_with_zero_added_rows():
    # key_b's own p-value is IDENTICAL in both runs (it gained zero recomputed rows) -- only a
    # third key ("brand_new", entirely from recomputed rows) is added, changing BH's m from 2 to 3.
    # A middle-ranked key's q depends on the full ordered set, not just its own p, so it must move
    # even though not a single row of ITS OWN cohort changed.
    p_a, p_b = 0.20, 0.03  # p_a large enough that q_b's binding constraint is its own candidate, not p_a's
    baseline_stats = [_stats("a_violations", p_a), _stats("b_violations", p_b)]
    guard_alpha._apply_bh_correction(baseline_stats)
    baseline_q_b = next(s for s in baseline_stats if s.key == "b_violations").q_value
    assert baseline_q_b == pytest.approx(2 * p_b)  # rank 1 of 2: p*m/rank = p_b*2/1

    combined_stats = [_stats("a_violations", p_a), _stats("b_violations", p_b), _stats("brand_new_violations", 0.001)]
    guard_alpha._apply_bh_correction(combined_stats)
    combined_q_b = next(s for s in combined_stats if s.key == "b_violations").q_value
    assert combined_q_b == pytest.approx(1.5 * p_b)  # now rank 2 of 3: p*m/rank = p_b*3/2

    assert baseline_q_b != combined_q_b  # b_violations gained 0 rows, but its q still moved via BH's m


# ---------------------------------------------------------------------------
# End-to-end wiring: fetch -> shadow -> recompute -> classify -> combine
# ---------------------------------------------------------------------------


async def test_pending_row_priced_via_alt_suffix_and_classified_by_own_transcript(monkeypatch):
    transcript = {"session_close_violations": ["bad"]}
    conn = FakeConn(pending=[_pending("X.KS", MON, transcript=transcript)])
    _patch_db(monkeypatch, conn)
    _patch_prices(monkeypatch, {"^KS11": _bench(), "X.KQ": _series(100.0, 110.0)})

    report = await guard_alpha.compute_guard_alpha_with_recompute(
        market="domestic", scope="live", account_mode="real",
    )
    assert report.n_recomputed_rows == 1 and report.n_recomputed_unscorable == 0
    assert report.n_alt_symbol_tickers == 1 and report.n_recomputed_alt_rows == 1
    key_stats = next(s for s in report.combined.key_stats if s.key == "session_close_violations")
    assert (key_stats.n_violated, key_stats.n_clean) == (1, 0)  # picked up from the row's OWN transcript
    assert report.recomputed_added == {"session_close_violations": (1, 0)}


async def test_resolved_sibling_shadows_pending(monkeypatch):
    conn = FakeConn(
        baseline=[_baseline(MON, "Buy", 0.02, ticker="BASE")],
        pending=[_pending("X.KS", MON)],
        resolved_keys=[("X.KS", MON)],
    )
    _patch_db(monkeypatch, conn)
    calls: list[str] = []

    async def fake(symbol, start, end):
        calls.append(symbol)
        if symbol == "^KS11":
            return _bench()
        raise ValueError("should never be fetched")

    monkeypatch.setattr(pending_recompute, "get_price_series", fake)
    report = await guard_alpha.compute_guard_alpha_with_recompute(
        market="domestic", scope="live", account_mode="real",
    )
    assert report.n_shadowed_by_resolved == 1 and report.n_recomputed_rows == 0
    assert "X.KS" not in calls and "X.KQ" not in calls


async def test_hold_unknown_held_state_is_unscorable_not_included(monkeypatch):
    conn = FakeConn(pending=[_pending("X.KS", MON, rating="Hold", held=None)])
    _patch_db(monkeypatch, conn)
    _patch_prices(monkeypatch, {"^KS11": _bench(), "X.KS": _series(100.0, 110.0)})
    report = await guard_alpha.compute_guard_alpha_with_recompute(
        market="domestic", scope="live", account_mode="real",
    )
    assert report.n_recomputed_unscorable == 1 and report.n_recomputed_rows == 0


async def test_readonly_snapshot_and_no_writes(monkeypatch):
    conn = FakeConn(pending=[_pending("X.KS", MON)])
    _patch_db(monkeypatch, conn)
    _patch_prices(monkeypatch, {"^KS11": _bench(), "X.KQ": _series(100.0, 110.0)})
    await guard_alpha.compute_guard_alpha_with_recompute(market="domestic", scope="live", account_mode="real")
    assert conn.transaction_kwargs == [{"isolation": "repeatable_read", "readonly": True}]
    for query, _ in conn.queries:
        assert not any(kw in query.upper() for kw in ("INSERT", "UPDATE", "DELETE"))


async def test_trigger_and_account_mode_propagate_to_all_three_queries(monkeypatch):
    conn = FakeConn(pending=[_pending("X.KS", MON)])
    _patch_db(monkeypatch, conn)
    _patch_prices(monkeypatch, {"^KS11": _bench(), "X.KQ": _series(100.0, 110.0)})
    await guard_alpha.compute_guard_alpha_with_recompute(
        market="domestic", scope="live", account_mode="real", trigger="manual",
    )
    assert len(conn.queries) == 3  # baseline, pending, resolved_keys
    for query, params in conn.queries:
        assert "decision_trigger = 'manual'" in query
        assert "account_mode = $" in query and "real" in params


async def test_rejects_bad_combinations():
    for kwargs in (
        {"market": None, "scope": "live", "account_mode": "real"},
        {"market": "overseas", "scope": "live", "account_mode": "real"},
        {"market": "domestic", "scope": "backtest", "account_mode": "real"},
        {"market": "domestic", "scope": None, "account_mode": "real"},
        {"market": "domestic", "scope": "live", "account_mode": None},
    ):
        with pytest.raises(ValueError):
            await guard_alpha.compute_guard_alpha_with_recompute(**kwargs)


def test_new_key_only_in_recomputed_rows_rendered_with_dash_baseline():
    baseline = guard_alpha._build_report([], basis="variable", manual_excluded=True)
    combined_rows = [
        _baseline(MON, "Buy", 0.05, ticker="A", transcript={"brand_new_violations": ["x"]}),
        _baseline(MON, "Buy", 0.01, ticker="B", transcript={"brand_new_violations": []}),
    ]
    combined = guard_alpha._build_report(combined_rows, basis="variable", manual_excluded=True)
    text = guard_alpha.render_guard_alpha_recompute(guard_alpha.GuardAlphaRecomputeReport(
        baseline=baseline, combined=combined, market="domestic", scope="live", account_mode="real",
        trigger=None, eval_rule="rule", n_pending_past_cutoff=2, n_shadowed_by_resolved=0,
        n_recomputed_rows=2, n_recomputed_unscorable=0, recomputed_rating_mix={"Buy": 2},
        recomputed_added={"brand_new_violations": (1, 1)}, n_recompute_failed=0, n_not_matured=0,
        failures_by_reason={}, failed_tickers=[], n_alt_symbol_tickers=0, n_tickers_fetched=2,
        n_recomputed_alt_rows=0, holding_days_min=7, holding_days_median=7.0, holding_days_max=7,
    ))
    assert "brand_new_violations" in text
    assert "(new key: only present in recomputed rows)" in text


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _argv(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["app.audit", *args])


def test_cli_recompute_pending_shared_with_guard_alpha(monkeypatch):
    for args in (
        ("guard-alpha", "--recompute-pending"),  # no --market
        ("guard-alpha", "--recompute-pending", "--market", "overseas", "--scope", "live", "--account-mode", "real"),
        ("guard-alpha", "--recompute-pending", "--market", "domestic", "--scope", "backtest", "--account-mode", "real"),
        ("guard-alpha", "--recompute-pending", "--market", "domestic", "--scope", "live"),  # no --account-mode
        ("guard-alpha", "--recompute-pending", "--market", "domestic", "--scope", "live",
         "--account-mode", "real", "--start-date", "2026-09-01"),
    ):
        _argv(monkeypatch, *args)
        with pytest.raises(SystemExit):
            audit_main._parse_args()
    # valid combination, including --trigger manual, must parse without error
    _argv(monkeypatch, "guard-alpha", "--recompute-pending", "--market", "domestic", "--scope", "live",
          "--account-mode", "real", "--trigger", "manual")
    audit_main._parse_args()


async def test_cli_guard_alpha_default_path_never_touches_recompute(monkeypatch, capsys):
    async def boom(**kwargs):
        raise AssertionError("recompute path must not run without the flag")

    async def fake_compute(*a, **kwargs):
        return "REPORT"

    monkeypatch.setattr(audit_main, "compute_guard_alpha_with_recompute", boom)
    monkeypatch.setattr(audit_main, "compute_guard_alpha", fake_compute)
    monkeypatch.setattr(audit_main, "render_guard_alpha", lambda r: "DEFAULT-RENDER")
    _argv(monkeypatch, "guard-alpha", "--scope", "live", "--account-mode", "real")
    await audit_main._main()
    assert capsys.readouterr().out.strip() == "DEFAULT-RENDER"


async def test_cli_guard_alpha_flag_routes_to_recompute(monkeypatch, capsys):
    seen = {}

    async def fake_recompute(**kwargs):
        seen.update(kwargs)
        return "REPORT"

    monkeypatch.setattr(audit_main, "compute_guard_alpha_with_recompute", fake_recompute)
    monkeypatch.setattr(audit_main, "render_guard_alpha_recompute", lambda r: f"RENDER:{r}")
    _argv(monkeypatch, "guard-alpha", "--market", "domestic", "--scope", "live",
          "--account-mode", "real", "--recompute-pending")
    await audit_main._main()
    assert capsys.readouterr().out.strip() == "RENDER:REPORT"
    assert seen == {"market": "domestic", "scope": "live", "account_mode": "real", "trigger": None, "basis": "variable"}
