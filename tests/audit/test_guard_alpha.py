"""app.audit.guard_alpha — asyncpg pool/connection faked, no real Postgres.
Same fake-pool convention as tests/audit/test_screening_ic.py."""

from __future__ import annotations

import json
from datetime import date

from app.audit import guard_alpha


class FakeConn:
    def __init__(self, fetch_result=None):
        self.fetch_result = fetch_result if fetch_result is not None else []
        self.queries: list[tuple] = []

    async def fetch(self, query, *args):
        self.queries.append((query, args))
        return self.fetch_result


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


def _row(trade_date, ticker, rating, transcript, signed_alpha, holding_days=10):
    """Builds a row in the INTERNAL shape `_compute_key_stats` expects
    (post-`_parse_transcript`, as produced by `_build_report`) — distinct
    from the raw asyncpg row shape (`full_transcript`) that
    `compute_guard_alpha`/`_build_report` consume."""
    return {
        "trade_date": trade_date, "ticker": ticker, "rating": rating,
        "transcript": transcript, "holding_days": holding_days, "signed_alpha": signed_alpha,
    }


def _raw_row(trade_date, ticker, rating, transcript, signed_alpha, holding_days=10):
    """Raw asyncpg-row shape (`full_transcript`, possibly a JSON string) for
    tests that exercise `_build_report`/`compute_guard_alpha` directly."""
    return {
        "trade_date": trade_date, "ticker": ticker, "rating": rating,
        "full_transcript": transcript, "holding_days": holding_days, "signed_alpha": signed_alpha,
    }


# ---------------------------------------------------------------------------
# 1c: absent key must be EXCLUDED from that key's comparison, not counted
# as clean.
# ---------------------------------------------------------------------------

def test_absent_key_excludes_row_from_comparison():
    rows = [
        _row(date(2026, 9, 1), "A", "Buy", {"guard_x_violations": ["bad"]}, 0.05),
        _row(date(2026, 9, 1), "B", "Buy", {"guard_x_violations": []}, 0.02),
        # No guard_x key at all — must not be treated as "clean".
        _row(date(2026, 9, 1), "C", "Buy", {}, 0.01),
    ]
    stats = guard_alpha._compute_key_stats("guard_x_violations", rows)
    assert stats.n_violated == 1
    assert stats.n_clean == 1
    # n_violated + n_clean == 2, NOT 3 — row "C" (no guard_x key at all) is
    # excluded from the comparison entirely, not folded into "clean".
    assert stats.n_violated + stats.n_clean == 2


# ---------------------------------------------------------------------------
# violated/clean assignment on present-but-empty vs present-and-non-empty.
# ---------------------------------------------------------------------------

def test_present_empty_is_clean_present_nonempty_is_violated():
    rows = [
        _row(date(2026, 9, 1), "A", "Buy", {"g_flags": {"x": 1}}, 0.1),
        _row(date(2026, 9, 1), "B", "Buy", {"g_flags": {}}, 0.1),
        _row(date(2026, 9, 1), "C", "Buy", {"g_flags": None}, 0.1),
        _row(date(2026, 9, 1), "D", "Buy", {"g_flags": []}, 0.1),
    ]
    stats = guard_alpha._compute_key_stats("g_flags", rows)
    assert stats.n_violated == 1
    assert stats.n_clean == 3


# ---------------------------------------------------------------------------
# full_transcript as a JSON string is handled the same as a dict.
# ---------------------------------------------------------------------------

def test_transcript_as_json_string_parsed_same_as_dict():
    as_dict = {"guard_x_violations": ["bad"]}
    as_string = json.dumps(as_dict)
    assert guard_alpha._parse_transcript(as_dict) == guard_alpha._parse_transcript(as_string)
    assert guard_alpha._guard_keys(guard_alpha._parse_transcript(as_string)) == ["guard_x_violations"]


def test_transcript_none_and_garbage_string_yield_empty_dict():
    assert guard_alpha._parse_transcript(None) == {}
    assert guard_alpha._parse_transcript("not json") == {}
    assert guard_alpha._parse_transcript("[1,2,3]") == {}


# ---------------------------------------------------------------------------
# within-day pairing: a day with only violated rows (no clean) contributes
# no delta.
# ---------------------------------------------------------------------------

def test_day_with_only_violated_rows_contributes_no_delta():
    rows = [
        # Day 1: only violated — must not pair.
        _row(date(2026, 9, 1), "A", "Buy", {"g_violations": ["x"]}, 0.10),
        _row(date(2026, 9, 1), "B", "Buy", {"g_violations": ["x"]}, 0.20),
        # Day 2: both present — must pair.
        _row(date(2026, 9, 2), "C", "Buy", {"g_violations": ["x"]}, 0.05),
        _row(date(2026, 9, 2), "D", "Buy", {"g_violations": []}, 0.01),
    ]
    stats = guard_alpha._compute_key_stats("g_violations", rows)
    assert stats.n_paired_days == 1
    assert stats.paired_mean_delta is not None


# ---------------------------------------------------------------------------
# paired delta arithmetic on a small hand-computed fixture.
# ---------------------------------------------------------------------------

def test_paired_delta_hand_computed():
    rows = [
        # Day 1: violated mean = (0.10+0.20)/2=0.15, clean mean=0.05 -> delta=0.10
        _row(date(2026, 9, 1), "A", "Buy", {"g_violations": ["x"]}, 0.10),
        _row(date(2026, 9, 1), "B", "Buy", {"g_violations": ["x"]}, 0.20),
        _row(date(2026, 9, 1), "C", "Buy", {"g_violations": []}, 0.05),
        # Day 2: violated mean = 0.00, clean mean = 0.02 -> delta = -0.02
        _row(date(2026, 9, 2), "D", "Buy", {"g_violations": ["x"]}, 0.00),
        _row(date(2026, 9, 2), "E", "Buy", {"g_violations": []}, 0.02),
    ]
    stats = guard_alpha._compute_key_stats("g_violations", rows)
    assert stats.n_paired_days == 2
    expected_mean_delta = (0.10 + (-0.02)) / 2
    assert abs(stats.paired_mean_delta - expected_mean_delta) < 1e-9


# ---------------------------------------------------------------------------
# permutation p-value determinism under seed=0.
# ---------------------------------------------------------------------------

def test_permutation_p_value_deterministic_under_seed_zero():
    day_violated = [[0.10, 0.20], [0.00]]
    day_clean = [[0.05], [0.02, 0.03]]
    p1 = guard_alpha._permutation_p_value(day_violated, day_clean, observed=0.08)
    p2 = guard_alpha._permutation_p_value(day_violated, day_clean, observed=0.08)
    assert p1 == p2
    assert 0 < p1 <= 1


# ---------------------------------------------------------------------------
# BH: hand-computed small vector of p-values maps to expected q-values.
# ---------------------------------------------------------------------------

def test_bh_correction_hand_computed():
    # p = [0.01, 0.02, 0.03, 0.04, 0.50], m=5
    # sorted ascending, rank i=1..5:
    # q5 = 0.50*5/5 = 0.50
    # q4 = min(0.50, 0.04*5/4) = min(0.50, 0.05) = 0.05
    # q3 = min(0.05, 0.03*5/3) = min(0.05, 0.05) = 0.05
    # q2 = min(0.05, 0.02*5/2) = min(0.05, 0.05) = 0.05
    # q1 = min(0.05, 0.01*5/1) = min(0.05, 0.05) = 0.05
    ps = [0.01, 0.02, 0.03, 0.04, 0.50]
    key_stats = [
        guard_alpha.GuardKeyStats(
            key=f"k{i}", n_violated=25, n_clean=25, n_paired_days=25,
            paired_mean_delta=0.01, paired_ci95=(0.0, 0.02), p_value=p, q_value=None,
            naive_mean_violated=0.01, naive_mean_clean=0.0, naive_delta=0.01,
        )
        for i, p in enumerate(ps)
    ]
    guard_alpha._apply_bh_correction(key_stats)
    q_values = [round(s.q_value, 4) for s in key_stats]
    assert q_values == [0.05, 0.05, 0.05, 0.05, 0.5]


# ---------------------------------------------------------------------------
# verdict suppression: below-threshold fixture renders UNDERPOWERED and no
# conclusion-sounding text.
# ---------------------------------------------------------------------------

def test_underpowered_fixture_renders_marker_and_no_conclusion():
    rows = []
    # Only 3 paired days, well under _MIN_DAYS_FOR_VERDICT=20.
    for i in range(3):
        rows.append(_raw_row(date(2026, 9, 1 + i), f"V{i}", "Buy", {"g_violations": ["x"]}, 0.10))
        rows.append(_raw_row(date(2026, 9, 1 + i), f"C{i}", "Buy", {"g_violations": []}, 0.01))

    report = guard_alpha._build_report(rows, basis="variable", manual_excluded=True)
    # 1 raw guard key + the 3 always-on derived signals (none of which have
    # their source field in these transcripts at all, so all three are
    # entirely unpaired/excluded — see the derived-signal tests below).
    assert len(report.key_stats) == 1 + len(guard_alpha._DERIVED_SIGNALS)
    stats = next(s for s in report.key_stats if s.key == "g_violations")
    assert stats.verdict == "UNDERPOWERED"
    assert stats.underpowered is True

    rendered = guard_alpha.render_guard_alpha(report)
    assert "UNDERPOWERED" in rendered
    # Must not read as a conclusion.
    for banned in ("guard causes", "proven", "confirmed worse", "guard is bad"):
        assert banned not in rendered.lower()


# ---------------------------------------------------------------------------
# Fix 2: a well-powered key whose q-value shows no detectable effect must
# render as NO_EFFECT (a real finding — deprioritize), never as
# UNDERPOWERED/"keep collecting". This is the core of the fix: the old code
# conflated "too little data" with "measured, no effect".
# ---------------------------------------------------------------------------

def test_well_powered_large_q_renders_no_effect_not_underpowered():
    key_stats = [
        guard_alpha.GuardKeyStats(
            key="well_powered_key", n_violated=25, n_clean=25, n_paired_days=25,
            paired_mean_delta=0.001, paired_ci95=(-0.01, 0.01), p_value=0.9, q_value=None,
            naive_mean_violated=0.01, naive_mean_clean=0.009, naive_delta=0.001,
        ),
    ]
    guard_alpha._apply_bh_correction(key_stats)
    assert key_stats[0].q_value == 0.9

    # Re-run the same verdict logic _build_report applies, directly on this
    # single-key fixture (bypassing the DB row plumbing).
    stats_row = key_stats[0]
    well_powered = (
        stats_row.n_paired_days >= guard_alpha._MIN_DAYS_FOR_VERDICT
        and stats_row.n_violated >= guard_alpha._MIN_ROWS_PER_GROUP
        and stats_row.n_clean >= guard_alpha._MIN_ROWS_PER_GROUP
        and stats_row.q_value is not None
    )
    assert well_powered is True
    verdict = "FINDING" if stats_row.q_value < 0.05 else "NO_EFFECT"
    assert verdict == "NO_EFFECT"

    from dataclasses import replace
    final = replace(stats_row, verdict=verdict)
    assert final.verdict == "NO_EFFECT"
    assert final.underpowered is False

    report = guard_alpha.GuardAlphaReport(
        basis="variable", n_rows=50, n_days=25, start=None, end=None,
        manual_excluded=True, n_keys_discovered=1, n_keys_tested=1,
        holding_days_min=None, holding_days_max=None, holding_days_mean=None,
        key_stats=[final],
    )
    rendered = guard_alpha.render_guard_alpha(report)
    assert "NO_EFFECT" in rendered
    assert "well_powered_key" in rendered
    # The per-key note for this NO_EFFECT row must not tell the reader to
    # keep collecting data — that instruction is reserved for UNDERPOWERED.
    no_effect_note = next(line for line in rendered.splitlines() if line.strip().startswith("NO_EFFECT:"))
    assert "underpowered" not in no_effect_note.lower()
    assert "deprioritize" in no_effect_note.lower()


# ---------------------------------------------------------------------------
# Fix 1: n_keys_discovered (every guard key found in any transcript) vs
# n_keys_tested (only keys that produced a p-value, i.e. BH's m) must differ
# when some keys have no paired days, and the report must not conflate them.
# ---------------------------------------------------------------------------

def test_n_keys_discovered_vs_tested_differ_when_some_keys_unpaired():
    rows = [
        # guard_a: has both violated and clean rows -> testable.
        _raw_row(date(2026, 9, 1), "A", "Buy", {"guard_a_violations": ["x"]}, 0.05),
        _raw_row(date(2026, 9, 1), "B", "Buy", {"guard_a_violations": []}, 0.01),
        # guard_b: present in only one row, no clean counterpart anywhere ->
        # discovered but never testable (no p-value).
        _raw_row(date(2026, 9, 1), "C", "Buy", {"guard_b_violations": ["y"]}, 0.02),
    ]
    report = guard_alpha._build_report(rows, basis="variable", manual_excluded=True)
    assert report.n_keys_discovered == 2
    assert report.n_keys_tested == 1

    rendered = guard_alpha.render_guard_alpha(report)
    assert "keys discovered: 2" in rendered
    assert "keys actually tested (BH m" in rendered
    assert "keys actually tested (BH m, multiple-comparison count): 1" in rendered
    assert "corrected across the 1 keys that actually produced a p-value" in rendered
    assert "out of 2 keys discovered" in rendered


# ---------------------------------------------------------------------------
# renderer includes rating mix per group.
# ---------------------------------------------------------------------------

def test_renderer_includes_rating_mix_per_group():
    rows = [
        _raw_row(date(2026, 9, 1), "A", "Hold", {"g_violations": ["x"]}, 0.01),
        _raw_row(date(2026, 9, 1), "B", "Buy", {"g_violations": ["x"]}, 0.02),
        _raw_row(date(2026, 9, 1), "C", "Hold", {"g_violations": []}, 0.01),
    ]
    report = guard_alpha._build_report(rows, basis="variable", manual_excluded=True)
    rendered = guard_alpha.render_guard_alpha(report)
    assert "rating mix" in rendered
    assert "Hold" in rendered
    assert "Buy" in rendered


# ---------------------------------------------------------------------------
# derived signals: `derived:neutral_reversed` — three-state classification.
# ---------------------------------------------------------------------------

def test_derived_neutral_reversed_three_states():
    classify = guard_alpha._DERIVED_SIGNALS["derived:neutral_reversed"]
    # Absent -> None (excluded).
    assert classify({}) is None
    # Not a dict -> None (excluded).
    assert classify({"neutral_reversal": "not a dict"}) is None
    # Empty dict -> None (excluded) — the field never fired this cycle.
    assert classify({"neutral_reversal": {}}) is None
    # Present, reversed True -> True.
    assert classify({"neutral_reversal": {"reversed": True}}) is True
    # Present, reversed False -> False.
    assert classify({"neutral_reversal": {"reversed": False}}) is False


# ---------------------------------------------------------------------------
# derived signals: `derived:neutral_reversal_uncited` — three-state
# classification, with the reversal_basis fast path and the two-boolean
# fallback for older records, and the critical "missing must not read as
# neither" guard.
# ---------------------------------------------------------------------------

def test_derived_neutral_reversal_uncited_three_states():
    classify = guard_alpha._DERIVED_SIGNALS["derived:neutral_reversal_uncited"]
    # Absent / not reversed / empty -> None (not part of this contrast).
    assert classify({}) is None
    assert classify({"neutral_reversal": {}}) is None
    assert classify({"neutral_reversal": {"reversed": False}}) is None
    # reversal_basis present -> used directly.
    assert classify({"neutral_reversal": {"reversed": True, "reversal_basis": "neither"}}) is True
    assert classify({"neutral_reversal": {"reversed": True, "reversal_basis": "new_fact"}}) is False
    assert (
        classify({"neutral_reversal": {"reversed": True, "reversal_basis": "later_turn_argument"}})
        is False
    )
    # No reversal_basis, but both booleans present -> derive from them.
    assert classify({
        "neutral_reversal": {
            "reversed": True, "new_fact_cited": False, "later_turn_argument_cited": False,
        },
    }) is True
    assert classify({
        "neutral_reversal": {
            "reversed": True, "new_fact_cited": True, "later_turn_argument_cited": False,
        },
    }) is False
    assert classify({
        "neutral_reversal": {
            "reversed": True, "new_fact_cited": False, "later_turn_argument_cited": True,
        },
    }) is False


def test_derived_neutral_reversal_uncited_missing_basis_is_none_not_uncited():
    """CRITICAL regression guard (task spec): an old record that reversed but
    predates `later_turn_argument_cited` — and has no `reversal_basis` either
    — must classify as None (unjudgeable), never as True ("uncited"). Reading
    a missing field as "neither" is the exact bug this signal exists to
    avoid (id=1539, 2026-09-17)."""
    classify = guard_alpha._DERIVED_SIGNALS["derived:neutral_reversal_uncited"]
    assert classify({"neutral_reversal": {"reversed": True}}) is None
    assert classify({
        "neutral_reversal": {"reversed": True, "new_fact_cited": False},
    }) is None


# ---------------------------------------------------------------------------
# derived signals: `derived:consensus_override_gated` — three-state
# classification, with `{}` (gate did not apply) treated as a non-
# observation, not a clean one.
# ---------------------------------------------------------------------------

def test_derived_consensus_override_gated_three_states():
    classify = guard_alpha._DERIVED_SIGNALS["derived:consensus_override_gated"]
    assert classify({}) is None
    assert classify({"consensus_override": "not a dict"}) is None
    # {} means the gate did not apply — a non-observation, NOT clean.
    assert classify({"consensus_override": {}}) is None
    assert classify({"consensus_override": {"gated": True}}) is True
    assert classify({"consensus_override": {"gated": False}}) is False


# ---------------------------------------------------------------------------
# a None-classified row lands in NEITHER group when run through
# _compute_key_stats with a derived classifier.
# ---------------------------------------------------------------------------

def test_derived_signal_none_row_excluded_from_both_groups():
    classify = guard_alpha._DERIVED_SIGNALS["derived:consensus_override_gated"]
    rows = [
        _row(date(2026, 9, 1), "A", "Buy", {"consensus_override": {"gated": True}}, 0.05),
        _row(date(2026, 9, 1), "B", "Buy", {"consensus_override": {"gated": False}}, 0.01),
        # {} -> None -> excluded from both groups.
        _row(date(2026, 9, 1), "C", "Buy", {"consensus_override": {}}, 0.02),
        # absent -> None -> excluded from both groups.
        _row(date(2026, 9, 1), "D", "Buy", {}, 0.03),
    ]
    stats = guard_alpha._compute_key_stats(
        "derived:consensus_override_gated", rows, classify=classify,
    )
    assert stats.n_violated == 1
    assert stats.n_clean == 1
    assert stats.n_violated + stats.n_clean == 2


# ---------------------------------------------------------------------------
# a signal that CANNOT pair yields p_value=None and must NOT increase
# n_keys_tested (the BH m) — the key regression this whole task exists to
# prevent.
# ---------------------------------------------------------------------------

def test_unpairable_derived_signal_yields_none_pvalue_and_does_not_inflate_m():
    classify = guard_alpha._DERIVED_SIGNALS["derived:consensus_override_gated"]
    # 258 empty-dict rows ({} -> None/excluded) + 1 non-empty (gated=True) —
    # mirrors the real census in the task brief: 1 violated, 0 clean, so it
    # can never pair.
    rows = [
        _row(date(2026, 9, 1), f"E{i}", "Hold", {"consensus_override": {}}, 0.0)
        for i in range(258)
    ]
    rows.append(_row(date(2026, 9, 1), "V0", "Buy", {"consensus_override": {"gated": True}}, 0.10))
    stats = guard_alpha._compute_key_stats(
        "derived:consensus_override_gated", rows, classify=classify,
    )
    assert stats.n_violated == 1
    assert stats.n_clean == 0
    assert stats.p_value is None

    # Now push it through the full report path alongside a genuine,
    # well-behaved raw guard key, and confirm n_keys_tested (BH m) counts
    # only the raw key, not this unpairable derived signal.
    raw_rows = []
    for i in range(258):
        raw_rows.append(_raw_row(
            date(2026, 9, 1), f"E{i}", "Hold",
            {"consensus_override": {}, "g_violations": ["x"] if i % 2 == 0 else []}, 0.01 * (i % 3),
        ))
    raw_rows.append(_raw_row(
        date(2026, 9, 1), "V0", "Buy",
        {"consensus_override": {"gated": True}, "g_violations": []}, 0.10,
    ))
    report = guard_alpha._build_report(raw_rows, basis="variable", manual_excluded=True)
    override_stats = next(
        s for s in report.key_stats if s.key == "derived:consensus_override_gated"
    )
    assert override_stats.p_value is None
    # n_keys_tested must reflect only the keys/signals that actually
    # produced a p-value (the raw g_violations key, single-day so also
    # UNDERPOWERED but IS pairable/testable) — the unpairable derived signal
    # must not be counted into it.
    assert report.n_keys_tested == sum(
        1 for s in report.key_stats if s.key != "derived:consensus_override_gated"
        and s.p_value is not None
    )


# ---------------------------------------------------------------------------
# raw-key path must reproduce IDENTICAL behavior via the default classifier
# (classify=None) as before this task's refactor.
# ---------------------------------------------------------------------------

def test_raw_key_path_unchanged_via_default_classifier():
    rows = [
        _row(date(2026, 9, 1), "A", "Buy", {"guard_x_violations": ["bad"]}, 0.05),
        _row(date(2026, 9, 1), "B", "Buy", {"guard_x_violations": []}, 0.02),
        _row(date(2026, 9, 1), "C", "Buy", {}, 0.01),
    ]
    stats = guard_alpha._compute_key_stats("guard_x_violations", rows)
    assert stats.n_violated == 1
    assert stats.n_clean == 1
    assert stats.n_violated + stats.n_clean == 2


# ---------------------------------------------------------------------------
# n_derived_signals is reported separately from n_keys_discovered (raw guard
# keys only) and never folded into it.
# ---------------------------------------------------------------------------

def test_n_derived_signals_reported_separately_from_n_keys_discovered():
    rows = [
        _raw_row(date(2026, 9, 1), "A", "Buy", {"guard_a_violations": ["x"]}, 0.05),
        _raw_row(date(2026, 9, 1), "B", "Buy", {"guard_a_violations": []}, 0.01),
    ]
    report = guard_alpha._build_report(rows, basis="variable", manual_excluded=True)
    assert report.n_keys_discovered == 1
    assert report.n_derived_signals == len(guard_alpha._DERIVED_SIGNALS) == 3
    rendered = guard_alpha.render_guard_alpha(report)
    assert "derived signals registered: 3" in rendered
    assert "derived:neutral_reversed" in rendered
    assert "derived:consensus_override_gated" in rendered
    assert "derived:neutral_reversal_uncited" in rendered


# ---------------------------------------------------------------------------
# end-to-end wiring: compute_guard_alpha issues one query and builds a
# report from the fake pool's fetch result.
# ---------------------------------------------------------------------------

async def test_compute_guard_alpha_end_to_end(monkeypatch):
    rows = [
        _raw_row(date(2026, 9, 1), "A", "Buy", {"g_violations": ["x"]}, 0.05),
        _raw_row(date(2026, 9, 1), "B", "Buy", {"g_violations": []}, 0.01),
    ]
    conn = FakeConn(fetch_result=rows)
    monkeypatch.setattr(guard_alpha, "get_pool", lambda: _async_return(FakePool(conn)))
    monkeypatch.setattr(guard_alpha, "get_current_user_id", lambda: "default")

    report = await guard_alpha.compute_guard_alpha(scope="live", account_mode="real")

    assert report.n_rows == 2
    assert report.manual_excluded is True
    assert len(conn.queries) == 1
    query, params = conn.queries[0]
    assert "decision_trigger IS DISTINCT FROM 'manual'" in query
    assert "analysts_failed" in query
    assert "real" in params
