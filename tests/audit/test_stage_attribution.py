"""app.audit.stage_attribution -- synthetic only: pure functions plus a faked
asyncpg pool (same convention as tests/audit/test_entry_paired.py and
tests/audit/test_guard_alpha_recompute.py). No Postgres, no network."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest

from app.audit import guard_alpha, pending_recompute, stage_attribution as sa
from app.backtest import factor_metrics

MON = date(2026, 9, 14)
TODAY = date(2026, 9, 28)
BENCH_DATES = [date(2026, 9, 18), date(2026, 9, 21), date(2026, 9, 22)]
D0 = date(2026, 9, 1)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _transcript(rm=None, neutral=None, extra=None):
    t: dict = {}
    if rm is not None:
        t["research_plan"] = f"**Recommendation**: {rm}\n\n**Rationale**: x"
    if neutral is not None:
        t["risk_debate"] = [{"speaker": "Neutral", "round": 1, "content": f"FINAL DECISION: {neutral}"}]
    t.update(extra or {})
    return t


def _row(
    ticker, trade_date, *, rm="Hold", trader=None, neutral=None, rating="Hold", alpha=None, status="resolved",
    held=False, extra=None, created_at=None, git_commit=None, decision_text=None,
):
    return {
        "id": 1, "ticker": ticker, "trade_date": trade_date, "created_at": created_at, "git_commit": git_commit,
        "status": status, "rating": rating, "trader_action": trader, "alpha_return": alpha,
        "held_at_decision": held, "held_at_decision_inferred": None,
        "full_transcript": _transcript(rm, neutral, extra), "decision_text": decision_text,
    }


def _report(rows, **kw):
    kw.setdefault("market", "domestic")
    kw.setdefault("account_mode", "real")
    return sa._build_report(rows, **kw)


def _day(i):
    return D0 + timedelta(days=i)


class FakeTxn:
    def __init__(self, conn, kwargs):
        conn.transaction_kwargs.append(kwargs)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class FakeConn:
    def __init__(self, rows=None, resolved_keys=None, mixed=0):
        self.rows = rows or []
        self.resolved_keys = resolved_keys or []
        self.mixed = mixed
        self.queries: list[tuple] = []
        self.transaction_kwargs: list[dict] = []

    def transaction(self, **kwargs):
        return FakeTxn(self, kwargs)

    async def fetch(self, query, *args):
        self.queries.append((query, args))
        if "HAVING" in query:
            return [{"n": self.mixed}]
        if "WITH latest" in query:
            return self.rows
        return [{"ticker": t, "trade_date": d} for t, d in self.resolved_keys]


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


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setattr(pending_recompute, "session_in_progress", lambda t, d: False)
    monkeypatch.setattr(sa, "local_today", lambda market: TODAY)
    monkeypatch.setattr(sa, "get_current_user_id", lambda: "default")
    monkeypatch.setattr(guard_alpha, "_N_PERMUTATIONS", 200)


def _patch_db(monkeypatch, conn):
    async def _pool():
        return FakePool(conn)

    monkeypatch.setattr(sa, "get_pool", _pool)


def _series(entry_price, exit_price, dates_after=None, entry=MON):
    out = {entry: entry_price}
    out.update({d: exit_price for d in (dates_after or BENCH_DATES)})
    return out


def _patch_prices(monkeypatch, table):
    async def fake(symbol, start, end):
        if symbol not in table:
            raise ValueError(f"no price history returned for {symbol!r}")
        return table[symbol]

    monkeypatch.setattr(pending_recompute, "get_price_series", fake)


# ---------------------------------------------------------------------------
# 1-6: extraction
# ---------------------------------------------------------------------------


def test_rm_extraction_markdown_dict_and_garbage():
    md = sa.extract_stage_labels({"rating": "Hold"}, {"research_plan": "**Recommendation**: Overweight\n\n**Rationale**: x"})
    assert md.rm == "Overweight" and md.rm_parsed and sa.stance(md.rm) == "bullish"
    dct = sa.extract_stage_labels({"rating": "Hold"}, {"research_plan": {"recommendation": "Underweight"}})
    assert dct.rm == "Underweight" and sa.stance(dct.rm) == "bearish"
    bad = sa.extract_stage_labels({"rating": "Hold"}, {"research_plan": "garbage"})
    assert bad.rm is None and bad.rm_parsed is False
    r = _report([_row("A.KS", D0, rm=None, extra={"research_plan": "garbage"})])
    assert r.n_rm_unparsed == 1


def test_normalize_level_and_stance_table():
    assert [sa.stance(x) for x in ("Buy", "Overweight", "Hold", "Underweight", "Sell")] == [
        "bullish", "bullish", "neutral", "bearish", "bearish",
    ]
    assert sa.normalize_level("buy") == "Buy"
    assert sa.normalize_level("Foo") is None and sa.normalize_level(None) is None
    assert sa.stance("Foo") is None and sa.stance(None) is None


def test_trader_column_markdown_fallback_and_mismatch():
    fb = sa.extract_stage_labels(
        {"rating": "Hold", "trader_action": None}, {"trader_proposal": "**Action**: Buy\n\n**Reasoning**: x"},
    )
    assert fb.trader == "Buy" and fb.trader_source == "markdown" and fb.trader_mismatch is False
    mm = sa.extract_stage_labels(
        {"rating": "Hold", "trader_action": "Hold"}, {"trader_proposal": "**Action**: Buy\n\n**Reasoning**: x"},
    )
    assert mm.trader == "Hold" and mm.trader_source == "column" and mm.trader_mismatch is True


def test_neutral_uses_last_round_verdict():
    debate = [
        {"speaker": "Neutral", "round": 1, "content": "FINAL DECISION: Buy"},
        {"speaker": "Aggressive", "round": 2, "content": "FINAL DECISION: Sell"},
        {"speaker": "Neutral", "round": 2, "content": "Final Verdict: Hold"},
    ]
    assert sa.extract_stage_labels({"rating": "Hold"}, {"risk_debate": debate}).neutral == "Hold"
    assert sa.extract_stage_labels({"rating": "Hold"}, {"risk_debate": debate[1:2]}).neutral is None
    nov = [{"speaker": "Neutral", "round": 1, "content": "no verdict at all"}]
    assert sa.extract_stage_labels({"rating": "Hold"}, {"risk_debate": nov}).neutral is None


_HYST_NOTE = (
    "[Hysteresis gate: downgraded from Buy to Hold — confidence (unset) below the 60% bar required "
    "to flip position state.]"
)


def _pm(row_extra, transcript):
    lab = sa.extract_stage_labels({"rating": "Hold", **row_extra}, transcript)
    return lab.pm_pre, lab.post_consensus, lab.final, lab


def test_pm_pre_recovery_cases():
    # consensus gated: structured record
    assert _pm({}, {"consensus_override": {"gated": True, "pm_rating": "Buy"}})[:3] == ("Buy", "Hold", "Hold")
    # consensus applied but passed
    assert _pm({"rating": "Buy"}, {"consensus_override": {"gated": False, "pm_rating": "Buy"}})[:3] == (
        "Buy", "Buy", "Buy",
    )
    # hysteresis note in thesis, consensus {}
    assert _pm({}, {"consensus_override": {}, "investment_thesis": "x\n\n" + _HYST_NOTE})[:3] == (
        "Buy", "Buy", "Hold",
    )
    # unjustified exit
    assert _pm({}, {"unjustified_exit_gate": {"gated": True}})[:3] == ("Sell", "Sell", "Hold")
    # garbled marker
    pm, post, _final, lab = _pm({}, {"investment_thesis": "[Hysteresis gate: garbled"})
    assert pm is None and post is None and lab.gate_parse_failed is True
    # legacy consensus note, no record
    legacy = "[Consensus override gate: downgraded from Overweight to Hold — the Research Manager ..."
    pm, post, _final, _lab = _pm({}, {"investment_thesis": legacy})
    assert pm == "Overweight" and post == "Hold"
    # decision_text only
    dt = "[Hysteresis gate: downgraded from Sell to Hold — below the confidence bar to flip position state.] x"
    assert _pm({"decision_text": dt}, {})[0] == "Sell"
    # record Buy vs note Overweight: record wins, mismatch counted
    pm, _post, _final, lab = _pm(
        {}, {"consensus_override": {"gated": True, "pm_rating": "Buy"}, "investment_thesis": legacy},
    )
    assert pm == "Buy" and lab.gate_record_note_mismatch is True
    # nothing: rating
    assert _pm({}, {})[:3] == ("Hold", "Hold", "Hold")
    # both markers at once
    both = "[Hysteresis gate: downgraded from Buy to Hold — x] [Consensus override gate: downgraded from Buy to Hold — y"
    assert _pm({}, {"investment_thesis": both})[3].gate_parse_failed is True


def test_holding_state():
    base = {"held_at_decision": None, "held_at_decision_inferred": None}
    assert sa._holding_state({**base, "held_at_decision": True}) == "held"
    assert sa._holding_state({**base, "held_at_decision_inferred": True}) == "held"
    assert sa._holding_state(base) == "unknown"
    assert sa._holding_state({**base, "held_at_decision": False}) == "flat"


# ---------------------------------------------------------------------------
# 7-8: paired statistics
# ---------------------------------------------------------------------------


def _tuples(day, ticker_prefix, values):
    return [(day, f"{ticker_prefix}{day}{i}", v) for i, v in enumerate(values)]


def test_paired_contrast_exact_values():
    d1, d2, d3, d4 = (date(2026, 9, i) for i in (1, 2, 3, 4))
    a = _tuples(d1, "a", [0.03, 0.01]) + _tuples(d2, "a", [0.05]) + _tuples(d3, "a", [-0.01])
    b = (
        _tuples(d1, "b", [0.00, -0.02, 0.02]) + _tuples(d2, "b", [0.01]) + _tuples(d3, "b", [0.01, 0.03])
        + _tuples(d4, "b", [0.1])
    )
    c = sa._paired_contrast("t", a, b)
    assert c.delta == pytest.approx(0.01)
    assert c.median_delta == pytest.approx(0.02)
    assert (c.n_paired_days, c.n_a_paired, c.n_b_paired, c.n_one_sided_days) == (3, 4, 6, 1)
    expected = factor_metrics.block_bootstrap_ci([[0.02], [0.04], [-0.03]], seed=0)
    assert c.ci95 == pytest.approx(expected)
    assert c.p_value is not None
    assert sa._verdict(
        n_paired_days=3, q=0.5, ci=c.ci95, delta=c.delta, n_a_paired=4, n_b_paired=6,
    ).verdict == "UNDERPOWERED"


def test_paired_contrast_uses_raw_alpha_not_sign_flipped():
    d = date(2026, 9, 1)
    c = sa._paired_contrast("t", [(d, "a", 0.00)], [(d, "b", -0.05)])
    assert c.delta == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# 9-10: cohorts
# ---------------------------------------------------------------------------


def test_transition_cells():
    assert sa._transition_cell("Buy", "Hold") == "blocked"
    assert sa._transition_cell("Buy", "Buy") == "passed"
    assert sa._transition_cell("Hold", "Hold") == "agreed_not"
    assert sa._transition_cell("Hold", "Buy") == "promoted"
    assert sa._transition_cell("Overweight", "Sell") == "blocked"
    assert sa._transition_cell("Buy", None) is None


def test_none_pm_pre_only_drops_pm_transitions():
    d = D0
    rows = [
        _row("A.KS", d, rm="Buy", trader="Hold", rating="Hold", alpha=0.01,
             extra={"investment_thesis": "[Hysteresis gate: garbled"}),
    ]
    r = _report(rows)
    flow = {(f.x, f.y): f for f in r.flow}
    assert flow[("RM", "Trader")].blocked == 1
    assert flow[("RM", "PM_pre")].n_both_known == 0
    assert flow[("Neutral", "PM_pre")].n_both_known == 0
    assert r.n_gate_parse_failed == 1


def test_identical_stage_is_not_tested_and_one_difference_is():
    rows = []
    for i in range(4):
        rows.append(_row(f"A{i}.KS", _day(i), rm="Buy", rating="Buy", alpha=0.02))
        rows.append(_row(f"B{i}.KS", _day(i), rm="Hold", rating="Hold", alpha=0.0))
    r = _report(rows)
    final = next(c for c in r.q1 if c.stages == ("Final",))
    pm_pre = next(c for c in r.q1 if c.stages == ("PM_pre",))
    assert final.identical_to in ("RM", "PM_pre") and final.p_value is None
    assert pm_pre.identical_to == "RM"
    # one row where Final differs from every earlier stage -> tested
    rows[0] = _row(
        "A0.KS", _day(0), rm="Buy", rating="Hold", alpha=0.02, extra={"investment_thesis": "x " + _HYST_NOTE},
    )
    r2 = _report(rows)
    final2 = next(c for c in r2.q1 if c.stages == ("Final",))
    assert final2.identical_to is None and final2.p_value is not None


# ---------------------------------------------------------------------------
# 11: integration
# ---------------------------------------------------------------------------


def _day_rows(n_days, a_alpha, b_alpha):
    rows = []
    for i in range(n_days):
        for j, v in enumerate(a_alpha(i) if callable(a_alpha) else a_alpha):
            rows.append(_row(f"A{i}x{j}.KS", _day(i), rm="Buy", alpha=v))
        for j, v in enumerate(b_alpha):
            rows.append(_row(f"B{i}x{j}.KS", _day(i), rm="Hold", alpha=v))
    return rows


def test_integration_finding_positive_material():
    r = _report(_day_rows(22, [0.02, 0.02], [0.0, 0.0, 0.0]))
    rm = r.q1[0]
    assert rm.delta == pytest.approx(0.02)
    assert rm.p_value == pytest.approx(1 / 201)
    assert rm.verdict == "FINDING" and rm.direction == "+" and rm.material is True
    assert rm.q_value is not None and rm.q_value < 0.05
    assert sa.interpret_q1(rm) == "discriminates in the correct direction"


def test_integration_no_effect_alternating_signs():
    r = _report(_day_rows(22, lambda i: [0.01 if i % 2 == 0 else -0.01], [0.0, 0.0]))
    rm = r.q1[0]
    assert rm.delta == pytest.approx(0.0, abs=1e-12)
    assert rm.verdict == "NO_EFFECT"
    assert "no discrimination detected" in sa.interpret_q1(rm)


# ---------------------------------------------------------------------------
# 12-13: verdict + interpretation
# ---------------------------------------------------------------------------


def _v(n=25, q=0.01, ci=(0.001, 0.02), delta=0.01, **kw):
    return sa._verdict(n_paired_days=n, q=q, ci=ci, delta=delta, **kw)


def test_verdict_rule_table():
    assert _v().verdict == "FINDING" and _v().direction == "+"
    assert _v(ci=(-0.001, 0.02)).verdict == "INCONCLUSIVE"
    assert _v(q=0.2, ci=(-0.01, 0.02)).verdict == "NO_EFFECT"
    assert _v(q=0.2, ci=(0.001, 0.02)).verdict == "INCONCLUSIVE"
    assert _v(n=19, q=0.001, ci=(0.01, 0.02)).verdict == "UNDERPOWERED"
    assert _v(q=None).verdict == "UNDERPOWERED"
    assert _v(n_a_paired=19, n_b_paired=40).verdict == "UNDERPOWERED"
    assert _v(delta=0.003).material is False and _v(delta=0.003).verdict == "FINDING"
    assert _v(delta_ex_extreme=-0.002).verdict == "INCONCLUSIVE"
    assert _v(delta_first_per_ticker=-0.002).verdict == "INCONCLUSIVE"
    assert _v(delta_ex_extreme=None).verdict == "INCONCLUSIVE"
    assert _v(coverage=0.38).verdict == "LOW_COVERAGE"
    assert _v(q=0.001, ci=(-0.02, -0.001), delta=-0.01).direction == "-"
    assert _v(ci95_5day_blocks=(-0.001, 0.02)).fragile is True


def _c(verdict, delta=0.0, material=True, direction=None):
    return sa.ContrastStats(
        name="x", verdict=verdict, delta=delta, material=material,
        direction=direction or ("+" if delta > 0 else "-"),
    )


def test_interpret_block_and_promotion():
    pos = _c("FINDING", 0.02)
    under = _c("UNDERPOWERED")
    assert sa.interpret_block(pos, under) == "DESTROYS"
    assert sa.interpret_block(pos, _c("FINDING", -0.01)) == "STRICT_BUT_INFORMATIVE"
    assert sa.interpret_block(_c("NO_EFFECT"), _c("FINDING", -0.02)) == "ADDS_VALUE"
    assert sa.interpret_block(under, _c("FINDING", -0.02)) == "NOT_DETERMINED"
    assert sa.interpret_block(_c("NO_EFFECT"), _c("NO_EFFECT")) == "NOT_DETERMINED"
    assert sa.interpret_promotion(pos) == "ADDS_VALUE"
    assert sa.interpret_promotion(_c("FINDING", -0.02)) == "DESTROYS"
    assert sa.interpret_promotion(_c("FINDING", 0.002, material=False)) == "NOT_DETERMINED"
    assert sa.interpret_promotion(under) == "NOT_DETERMINED"


# ---------------------------------------------------------------------------
# 14: render masking
# ---------------------------------------------------------------------------


def test_render_masks_underpowered_by_default():
    r = _report(_day_rows(3, [0.01], [0.0]))
    text = sa.render_stage_attribution(r)
    rm_line = next(line for line in text.splitlines() if line.startswith("Q1 RM"))
    assert "(masked)" in rm_line and "UNDERPOWERED" in rm_line
    assert "+1.0000%" not in text
    revealed = sa.render_stage_attribution(r, show_underpowered=True)
    rm_line2 = next(line for line in revealed.splitlines() if line.startswith("Q1 RM"))
    assert "PROVISIONAL" in rm_line2 and "+1.0000%" in rm_line2
    for out in (text, revealed):
        for line in out.splitlines():
            if "UNDERPOWERED" in line:
                assert not any(w in line for w in ("FINDING", "DESTROYS", "ADDS_VALUE"))


# ---------------------------------------------------------------------------
# 15: Q3 projection
# ---------------------------------------------------------------------------


def test_power_projection_values_and_never():
    dates = [_day(i) for i in range(16)]
    eligible = {dates[i] for i in range(0, 16, 2)}  # 8 of 16 -> 0.5
    p = sa._power_projection(decision_dates=dates, eligible_dates=eligible, n_paired_days=6, max_scored_date=dates[-1])
    assert p.elig_rate == pytest.approx(0.5) and p.realization == pytest.approx(0.75)
    assert p.backlog == 0 and p.extra_decision_days == 38
    never = sa._power_projection(decision_dates=dates, eligible_dates=set(), n_paired_days=0, max_scored_date=None)
    assert never.extra_decision_days is None
    done = sa._power_projection(decision_dates=dates, eligible_dates=eligible, n_paired_days=8, max_scored_date=dates[-1])
    assert done.realization == 1.0
    assert sa._power_projection(
        decision_dates=dates, eligible_dates=eligible, n_paired_days=8, max_scored_date=dates[-1], window=20,
    ).extra_decision_days == 24


# ---------------------------------------------------------------------------
# 16-17: SQL capture, argument validation
# ---------------------------------------------------------------------------


async def test_sql_is_read_only_trade_date_filtered_snapshot(monkeypatch):
    conn = FakeConn()
    _patch_db(monkeypatch, conn)
    await sa.compute_stage_attribution(
        market="domestic", account_mode="real", since=date(2026, 9, 18), until=date(2026, 9, 30),
    )
    assert conn.transaction_kwargs == [{"isolation": "repeatable_read", "readonly": True}]
    main_query, params = conn.queries[0]
    for needle in (
        "is_backtest = false", "ticker LIKE '%.KS'", "account_mode = $1",
        "decision_trigger IS DISTINCT FROM 'manual'", "trade_date >= $2", "trade_date <= $3",
        "(analysts_failed IS NULL OR analysts_failed = 0)", "user_id = $4", "DISTINCT ON (ticker, trade_date)",
        "horizon_alpha_return AS alpha_return",
    ):
        assert needle in main_query, needle
    assert "horizon_eval_date" not in main_query
    assert params == ("real", date(2026, 9, 18), date(2026, 9, 30), "default")
    for query, _ in conn.queries:
        assert not any(kw in query.upper() for kw in ("INSERT", "UPDATE", "DELETE"))


async def test_manual_trigger_and_variable_basis_sql(monkeypatch):
    conn = FakeConn()
    _patch_db(monkeypatch, conn)
    await sa.compute_stage_attribution(market="overseas", account_mode="paper", trigger="manual", basis="variable")
    query = conn.queries[0][0]
    assert "decision_trigger = 'manual'" in query and "ticker NOT LIKE '%.KS'" in query
    assert "alpha_return AS alpha_return" in query and "horizon" not in query


@pytest.mark.parametrize("kwargs", [
    dict(market=None, account_mode="real"), dict(market="domestic", account_mode=None),
    dict(market="both", account_mode="real"), dict(market="domestic", account_mode="real", basis="x"),
])
async def test_missing_or_bad_arguments_raise(kwargs):
    with pytest.raises(ValueError):
        await sa.compute_stage_attribution(**kwargs)
    with pytest.raises(ValueError):
        await sa.compute_stage_attribution_with_recompute(**kwargs)


async def test_recompute_rejects_overseas():
    with pytest.raises(ValueError):
        await sa.compute_stage_attribution_with_recompute(market="overseas", account_mode="real")


# ---------------------------------------------------------------------------
# 18: recompute
# ---------------------------------------------------------------------------


async def test_recompute_adds_pending_row_to_combined_only(monkeypatch):
    rows = [
        _row("B.KS", MON, rm="Buy", alpha=0.02),  # resolved baseline row
        _row("X.KS", MON, rm="Buy", status="pending", alpha=None),  # priced via X.KQ
        _row("Y.KS", MON, rm="Buy", status="pending", alpha=None),  # has a resolved sibling -> shadowed
        _row("H.KS", MON, rm="Buy", status="pending", alpha=None, held=True),  # held: skipped
    ]
    conn = FakeConn(rows=rows, resolved_keys=[("Y.KS", MON)])
    _patch_db(monkeypatch, conn)
    _patch_prices(monkeypatch, {"^KS11": _series(1000.0, 1010.0), "X.KQ": _series(100.0, 110.0)})
    rep = await sa.compute_stage_attribution_with_recompute(market="domestic", account_mode="real", basis="variable")
    assert rep.n_recomputed_rows == 1 and rep.n_recomputed_alt_rows == 1
    assert rep.n_shadowed_by_resolved == 1 and rep.n_skipped_not_flat == 1
    base_rm, comb_rm = rep.baseline.q1[0], rep.combined.q1[0]
    assert comb_rm.n_a == base_rm.n_a + 1
    assert rep.baseline.n_scored_flat == 1 and rep.combined.n_scored_flat == 2
    assert conn.transaction_kwargs == [{"isolation": "repeatable_read", "readonly": True}]
    text = sa.render_stage_attribution_recompute(rep)
    assert "baseline" in text and "recomputed" in text


# ---------------------------------------------------------------------------
# 19: mixed-status warning
# ---------------------------------------------------------------------------


async def test_mixed_status_keys_warn(monkeypatch):
    conn = FakeConn(rows=[_row("A.KS", D0, alpha=0.0)], mixed=2)
    _patch_db(monkeypatch, conn)
    rep = await sa.compute_stage_attribution(market="domestic", account_mode="real")
    assert rep.n_mixed_status_keys == 2
    assert "WARNING" in sa.render_stage_attribution(rep)
    clean = _report([_row("A.KS", D0, alpha=0.0)])
    assert "WARNING" not in sa.render_stage_attribution(clean)


# ---------------------------------------------------------------------------
# era split
# ---------------------------------------------------------------------------


def test_era_split_flags_heterogeneous_sign():
    boundary = datetime(2026, 9, 17, tzinfo=timezone.utc)
    rows = []
    for i in range(12):
        created = datetime(2026, 9, 10, tzinfo=timezone.utc) if i < 6 else datetime(2026, 9, 20, tzinfo=timezone.utc)
        sign = 1 if i < 6 else -1
        rows.append(_row(f"A{i}.KS", _day(i), rm="Buy", alpha=0.02 * sign, created_at=created, git_commit="aaaaaaaa1"))
        rows.append(_row(f"B{i}.KS", _day(i), rm="Hold", alpha=0.0, created_at=created, git_commit="bbbbbbbb2"))
    rep = _report(rows, era_boundary=boundary)
    rm_era = next(e for e in rep.era if e.name.startswith("Q1 RM"))
    assert rm_era.status == "ERA_HETEROGENEOUS" and rm_era.n_before == 6 and rm_era.n_after == 6
    assert rep.era_commits["before"] and rep.era_commits["after"]
    assert "ERA_HETEROGENEOUS" in sa.render_stage_attribution(rep)
