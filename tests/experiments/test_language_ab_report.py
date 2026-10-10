"""app.experiments.language_ab_report — every function under test here is
pure (dict in, value out); no files, no network, no LLM (F-stage), except the
load_records section below, which needs a real directory by definition."""

from __future__ import annotations

import json

from app.experiments import language_ab_report as report


# ---------------------------------------------------------------------------
# load_records — one corrupt/truncated file must not crash the whole report
# ---------------------------------------------------------------------------


def test_load_records_skips_unreadable_file_without_crashing(tmp_path, capsys):
    good = {"ticker": "AAPL", "date": "2026-07-01", "language": "ko"}
    (tmp_path / "AAPL_2026-07-01_ko_r1.json").write_text(json.dumps(good), encoding="utf-8")
    # Simulates a write interrupted mid-flush (power loss / kill -9) before
    # the atomic-write fix in language_ab.py's _run_one.
    (tmp_path / "AAPL_2026-07-02_ko_r1.json").write_text('{"ticker": "AAPL", "lang', encoding="utf-8")

    records, errors = report.load_records(tmp_path)

    assert records == [good]
    assert errors == []
    assert "skipping unreadable" in capsys.readouterr().err


def test_load_records_separates_errors_from_records(tmp_path):
    ok = {"ticker": "AAPL", "date": "2026-07-01", "language": "ko"}
    err = {"ticker": "AAPL", "language": "ko", "error": "boom"}
    (tmp_path / "AAPL_2026-07-01_ko_r1.json").write_text(json.dumps(ok), encoding="utf-8")
    (tmp_path / "AAPL_2026-07-01_ko_r1_error.json").write_text(json.dumps(err), encoding="utf-8")

    records, errors = report.load_records(tmp_path)

    assert records == [ok]
    assert errors == [err]


def test_load_records_skips_unrelated_json_file(tmp_path, capsys):
    """Regression: out_dir.glob('*.json') is unfiltered — a stray JSON file
    that isn't a run record must be skipped, not crash a downstream
    record["language"]/["ticker"]/["date"] lookup with a KeyError."""
    good = {"ticker": "AAPL", "date": "2026-07-01", "language": "ko"}
    (tmp_path / "AAPL_2026-07-01_ko_r1.json").write_text(json.dumps(good), encoding="utf-8")
    (tmp_path / "notes.json").write_text(json.dumps({"unrelated": "data"}), encoding="utf-8")

    records, errors = report.load_records(tmp_path)

    assert records == [good]
    assert errors == []
    assert "not a language_ab run record" in capsys.readouterr().err


def _record(
    ticker="AAPL",
    date="2026-07-01",
    language="ko",
    repeat=1,
    rating="Buy",
    confidence=0.6,
    stop_loss=None,
    price_target=None,
    entry_price=None,
    executive_summary="summary",
    investment_thesis="thesis",
    reports=None,
    isolation_violation=False,
    elapsed_seconds=100.0,
):
    return {
        "ticker": ticker,
        "date": date,
        "language": language,
        "repeat": repeat,
        "elapsed_seconds": elapsed_seconds,
        "isolation_violation": isolation_violation,
        "reports": reports or {},
        "research_debate": {"turns": []},
        "risk_debate": {"turns": []},
        "trader_proposal": {"entry_price": entry_price} if entry_price is not None else {},
        "research_plan": {},
        "final_decision": {
            "rating": rating,
            "confidence": confidence,
            "stop_loss": stop_loss,
            "price_target": price_target,
            "executive_summary": executive_summary,
            "investment_thesis": investment_thesis,
        },
    }


# ---------------------------------------------------------------------------
# hanja contamination
# ---------------------------------------------------------------------------


def test_record_has_hanja_true_when_a_report_text_contains_a_cjk_ideograph():
    rec = _record(reports={"market_analyst": "이 종목은 상승 여력이 있습니다 (現在)"})
    assert report.record_has_hanja(rec) is True


def test_record_has_hanja_false_for_pure_hangul_text():
    rec = _record(reports={"market_analyst": "이 종목은 상승 여력이 있습니다"})
    assert report.record_has_hanja(rec) is False


def test_hanja_contamination_rate_computes_fraction():
    clean = _record(reports={"market_analyst": "상승 여력 있음"})
    dirty = _record(reports={"market_analyst": "上昇 여력 있음"})
    assert report.hanja_contamination_rate([clean, dirty]) == 0.5


def test_hanja_contamination_rate_is_none_for_empty_input():
    assert report.hanja_contamination_rate([]) is None


def test_record_text_blob_includes_card_reason_fields():
    """FINDING B3 regression: card_reason_ko/card_reason_en (app/state.py,
    the dashboard signal card's short free-generated Korean/English prose,
    added after this experiment concluded) were silently absent from the
    hanja-contamination scan — a hanja slip could hide there undetected."""
    rec = _record()
    rec["final_decision"]["card_reason_ko"] = "現在 모멘텀 양호"
    assert "現在 모멘텀 양호" in report.record_text_blob(rec)
    assert report.record_has_hanja(rec) is True


# ---------------------------------------------------------------------------
# self-consistency violations
# ---------------------------------------------------------------------------


def test_self_consistency_flags_confidence_out_of_range():
    rec = _record(confidence=1.5)
    assert "confidence_out_of_range" in report.self_consistency_violations(rec)


def test_self_consistency_flags_stop_loss_at_or_above_entry_for_a_buy():
    rec = _record(rating="Buy", stop_loss=105.0, entry_price=100.0)
    assert "stop_loss_at_or_above_entry" in report.self_consistency_violations(rec)


def test_self_consistency_does_not_flag_a_stop_loss_below_entry():
    rec = _record(rating="Buy", stop_loss=95.0, entry_price=100.0)
    assert "stop_loss_at_or_above_entry" not in report.self_consistency_violations(rec)


def test_self_consistency_flags_rating_text_direction_mismatch():
    rec = _record(
        rating="Buy",
        executive_summary="This stock is a strong sell, expect a sharp decline and bearish downside.",
        investment_thesis="Sell now.",
    )
    assert "rating_text_direction_mismatch" in report.self_consistency_violations(rec)


def test_self_consistency_no_violations_for_a_coherent_buy():
    rec = _record(
        rating="Buy",
        stop_loss=90.0,
        entry_price=100.0,
        executive_summary="Bullish outlook, expect upside rally, recommend buy.",
        investment_thesis="Positive momentum supports a buy.",
    )
    assert report.self_consistency_violations(rec) == []


def test_self_consistency_returns_empty_when_no_final_decision():
    rec = _record()
    rec["final_decision"] = None
    assert report.self_consistency_violations(rec) == []


def test_hold_is_not_subject_to_the_stop_loss_at_or_above_entry_check():
    # Hold isn't a long entry — there's no position to protect with a stop,
    # so this check must not fire for it even when stop_loss >= entry_price.
    rec = _record(rating="Hold", stop_loss=105.0, entry_price=100.0)
    assert "stop_loss_at_or_above_entry" not in report.self_consistency_violations(rec)


# ---------------------------------------------------------------------------
# rating_claim_contradicts_field (explicit-claim detector, real ko_r2 bug)
# ---------------------------------------------------------------------------


def test_rating_claim_contradicts_field_catches_the_real_ko_r2_case():
    # Verbatim strings from
    # results/language_ab/pilot_009540_2026-07-15/009540.KS_2026-07-15_ko_r2.json
    # — rating="Hold" while the prose explicitly assigns Buy. This is the bug
    # `_rating_direction_violation` missed entirely (Hold always returned
    # False), and the reason this check exists.
    rec = _record(
        rating="Hold",
        confidence=None,
        executive_summary=(
            "현재 평지 상태를 기반으로 Buy 등급을 부여합니다. 14일 ATR 6.8%의 높은 변동성으로 "
            "인해 시스템 자동 스케일링이 적용되며, 실제 집행 가능한 최대 노출액은 총 자산의 "
            "30% 수준으로 제한됩니다."
        ),
        investment_thesis=(
            "트레이더의 매수 계획과 연구관리자의 Overweight 권고를 수용하되, 평지 진입 제약을 "
            "고려해 Buy로 확정합니다. 펀더멘털 측면에서 2026년 3분기 영업이익률 16.6%, ROE "
            "20.2%, EV/EBITDA 3.97x는 역사적 저평가 구간을 명확히 보여줍니다.\n\n"
            "[Hysteresis gate: downgraded from Buy to Hold — confidence (unset) below the 60% "
            "bar required to flip position state.]"
        ),
    )
    assert report.rating_claim_contradicts_field(rec) is True
    # NOTE: this record IS hysteresis-gated (its investment_thesis carries the
    # gate note), so as of Part B4 the contradiction is reclassified to
    # stale_prose_after_gate_override rather than counted as a language
    # self-consistency violation — see the dedicated Part B4 tests below for
    # why (the model didn't contradict itself; the gate silently overrode the
    # rating and left the pre-gate prose in place).
    violations = report.self_consistency_violations(rec)
    assert "stale_prose_after_gate_override" in violations
    assert "rating_claim_contradicts_field" not in violations


def test_rating_claim_contradicts_field_ignores_a_bare_mention_without_assignment():
    # "Buy 신호가 약해 관망합니다" mentions Buy but never assigns it — a false
    # positive here would be exactly the kind of noise the module docstring
    # warns against, so this must NOT be flagged.
    rec = _record(
        rating="Hold",
        executive_summary="Buy 신호가 약해 관망합니다.",
        investment_thesis="Buy를 고려했으나 기각하고 관망을 유지합니다.",
    )
    assert report.rating_claim_contradicts_field(rec) is False
    assert "rating_claim_contradicts_field" not in report.self_consistency_violations(rec)


def test_rating_claim_contradicts_field_catches_an_english_explicit_assignment():
    rec = _record(
        rating="Hold",
        executive_summary="Given the current setup, we assign a Buy rating to this position.",
        investment_thesis="",
    )
    assert report.rating_claim_contradicts_field(rec) is True


def test_rating_claim_contradicts_field_does_not_flag_a_matching_claim():
    rec = _record(
        rating="Buy",
        executive_summary="현재 상승 여력을 근거로 Buy 등급을 부여합니다.",
        investment_thesis="",
    )
    assert report.rating_claim_contradicts_field(rec) is False


def test_rating_claim_contradicts_field_false_when_no_rating_on_record():
    rec = _record(executive_summary="Buy 등급을 부여합니다.")
    rec["final_decision"]["rating"] = None
    assert report.rating_claim_contradicts_field(rec) is False


def test_rating_claim_contradicts_field_false_with_no_explicit_claim_at_all():
    rec = _record(rating="Hold", executive_summary="관망을 유지합니다.", investment_thesis="")
    assert report.rating_claim_contradicts_field(rec) is False


# ---------------------------------------------------------------------------
# KO claim regex boundary bug (module task fix #1): Python's \b is
# Unicode-aware, so Hangul counts as \w and the old `\b` boundary never
# matched a token directly followed by a Korean particle (no space) — the
# single most natural way Korean attaches a verb to an untranslated English
# token. These four are the exact reported blind-spot strings.
# ---------------------------------------------------------------------------


def test_rating_claim_detects_token_plus_ko_verb_with_no_space_between():
    # Sanity check the previously-working shape still works.
    rec = _record(rating="Hold", executive_summary="Buy 등급을 부여합니다.", investment_thesis="")
    assert report.rating_claim_contradicts_field(rec) is True


def test_rating_claim_detects_buy_ro_hwakjeong():
    rec = _record(rating="Hold", executive_summary="", investment_thesis="Buy로 확정합니다.")
    assert report.rating_claim_contradicts_field(rec) is True


def test_rating_claim_detects_sell_eul_gwongo():
    rec = _record(rating="Hold", executive_summary="Sell을 권고합니다.", investment_thesis="")
    assert report.rating_claim_contradicts_field(rec) is True


def test_rating_claim_detects_hold_ro_pandan():
    rec = _record(rating="Buy", executive_summary="Hold로 판단합니다.", investment_thesis="")
    assert report.rating_claim_contradicts_field(rec) is True


def test_rating_claim_ko_re_boundary_no_longer_uses_word_boundary():
    # Direct regression guard against re-introducing `\b` for the token
    # boundary — if someone reverts to `\b`, this assertion fails even
    # before the higher-level behavioral tests would catch it.
    assert "\\b(" not in report._RATING_CLAIM_KO_RE.pattern


# ---------------------------------------------------------------------------
# Negation guard — short lookahead only, documented limitation (module task)
# ---------------------------------------------------------------------------


def test_rating_claim_negation_guard_excludes_a_negated_recommendation():
    rec = _record(
        rating="Hold",
        executive_summary="Buy를 권고하지 않습니다.",
        investment_thesis="",
    )
    assert report.rating_claim_contradicts_field(rec) is False


def test_rating_claim_still_ignores_bare_mention_and_considered_then_rejected():
    # Pre-existing false-positive guards (precision-first design) must
    # still hold after the regex rewrite.
    rec = _record(
        rating="Hold",
        executive_summary="Buy 신호가 약해 관망합니다.",
        investment_thesis="Buy를 고려했으나 기각.",
    )
    assert report.rating_claim_contradicts_field(rec) is False


# ---------------------------------------------------------------------------
# magnitude outlier candidates
# ---------------------------------------------------------------------------


def test_magnitude_outlier_flags_a_10x_prose_number():
    rec = _record(stop_loss=1000.0, investment_thesis="The stop should sit around 100 per share.")
    candidates = report.magnitude_outlier_candidates(rec)
    assert any(c["field"] == "stop_loss" for c in candidates)


def test_magnitude_outlier_flags_a_one_tenth_prose_number():
    rec = _record(stop_loss=100.0, investment_thesis="The stop should sit around 1,000 per share.")
    candidates = report.magnitude_outlier_candidates(rec)
    assert any(c["field"] == "stop_loss" for c in candidates)


def test_magnitude_outlier_does_not_flag_a_matching_number():
    rec = _record(stop_loss=100.0, investment_thesis="The stop should sit around 100 per share.")
    candidates = report.magnitude_outlier_candidates(rec)
    assert candidates == []


def test_magnitude_outlier_returns_empty_with_no_structured_fields():
    rec = _record(stop_loss=None, price_target=None, investment_thesis="Some number 1000 appears here.")
    assert report.magnitude_outlier_candidates(rec) == []


# ---------------------------------------------------------------------------
# run-to-run variance (noise floor)
# ---------------------------------------------------------------------------


def test_variance_by_group_detects_rating_disagreement():
    records = [
        _record(repeat=1, rating="Buy"),
        _record(repeat=2, rating="Hold"),
    ]
    groups = report.variance_by_group(records)
    key = ("AAPL", "2026-07-01", "ko")
    assert groups[key]["rating_disagreement"] is True
    assert groups[key]["n_repeats"] == 2


def test_variance_by_group_no_disagreement_when_ratings_match():
    records = [
        _record(repeat=1, rating="Buy", confidence=0.5),
        _record(repeat=2, rating="Buy", confidence=0.7),
    ]
    groups = report.variance_by_group(records)
    key = ("AAPL", "2026-07-01", "ko")
    assert groups[key]["rating_disagreement"] is False
    assert groups[key]["confidence_stdev"] is not None


def test_variance_summary_by_language_excludes_groups_with_only_one_repeat():
    records = [_record(repeat=1)]
    summary = report.variance_summary_by_language(records)
    assert summary == {}


def test_variance_summary_by_language_aggregates_across_tickers():
    records = [
        _record(ticker="AAPL", repeat=1, rating="Buy"),
        _record(ticker="AAPL", repeat=2, rating="Buy"),
        _record(ticker="MSFT", repeat=1, rating="Buy"),
        _record(ticker="MSFT", repeat=2, rating="Sell"),
    ]
    summary = report.variance_summary_by_language(records)
    assert summary["ko"]["n_groups"] == 2
    assert summary["ko"]["rating_disagreement_rate"] == 0.5  # MSFT disagreed, AAPL didn't


def test_variance_by_group_post_gate_excludes_missing_rating_rather_than_counting_it():
    """FINDING B1 regression: pre_gate=False used to append final_decision's
    "rating" as-is, so a record with no usable rating (None) counted as a
    distinct rating value — two repeats that both agree ("Buy", "Buy") plus
    one with rating=None used to read as a 3-way disagreement."""
    records = [
        _record(repeat=1, rating="Buy"),
        _record(repeat=2, rating="Buy"),
        _record(repeat=3, rating=None),
    ]
    groups = report.variance_by_group(records, pre_gate=False)
    key = ("AAPL", "2026-07-01", "ko")
    assert groups[key]["rating_disagreement"] is False
    assert groups[key]["modal_agreement"] == 1.0
    assert groups[key]["n_ratings"] == 2
    assert groups[key]["n_repeats"] == 3  # raw record count is unaffected


def test_variance_summary_by_language_gates_on_usable_ratings_not_raw_repeats():
    """FINDING B2 regression: the >=2 noise-floor gate used to key off
    n_repeats (raw record count) instead of n_ratings (usable ratings after
    exclusion) — a group with 3 repeats but only 1 usable rating used to
    read as modal_agreement=1.0 ("perfectly stable"), a single observation
    misrepresented as a 3-way agreement."""
    records = [
        _record(ticker="AAA", repeat=1, rating="Buy"),
        _record(ticker="AAA", repeat=2, rating=None),
        _record(ticker="AAA", repeat=3, rating=None),
    ]
    summary = report.variance_summary_by_language(records, pre_gate=False)
    assert summary == {}  # only 1 usable rating — must not enter the noise-floor average at all


# ---------------------------------------------------------------------------
# modal agreement fraction (module task fix #2 — replaces the saturating
# binary disagreement metric as the primary stability number)
# ---------------------------------------------------------------------------


def test_modal_agreement_fraction_all_agree_is_1_0():
    assert report._modal_agreement_fraction(["Buy", "Buy", "Buy"]) == 1.0


def test_modal_agreement_fraction_one_outlier_of_three():
    assert report._modal_agreement_fraction(["Hold", "Buy", "Buy"]) == 2 / 3
    # Same fraction from a differently-shaped split — the whole point of
    # this metric vs. the old binary one: both of these read as 100%
    # disagreement under the old metric, but this one tells them apart from
    # a fully-split group below.
    assert report._modal_agreement_fraction(["Hold", "Hold", "Buy"]) == 2 / 3


def test_modal_agreement_fraction_fully_split_three_way():
    assert report._modal_agreement_fraction(["Hold", "Buy", "Sell"]) == 1 / 3


def test_modal_agreement_fraction_empty_is_none():
    assert report._modal_agreement_fraction([]) is None


def test_modal_agreement_fraction_tie_is_order_independent():
    # A 2-way tie ("Buy" x1, "Hold" x1): `Counter.most_common` picks one tied
    # value arbitrarily (stable insertion order), but the RETURNED FRACTION
    # is identical regardless of which tied value is picked, because both
    # tied values share the same top count by definition of "tied". This is
    # the documented tie-handling rule: ties don't need to be broken because
    # they can't change the numeric result.
    assert report._modal_agreement_fraction(["Buy", "Hold"]) == 0.5
    assert report._modal_agreement_fraction(["Hold", "Buy"]) == 0.5


def test_variance_by_group_reports_modal_agreement():
    records = [
        _record(repeat=1, rating="Hold"),
        _record(repeat=2, rating="Buy"),
        _record(repeat=3, rating="Buy"),
    ]
    groups = report.variance_by_group(records)
    key = ("AAPL", "2026-07-01", "ko")
    assert groups[key]["modal_agreement"] == 2 / 3
    # Legacy binary flag still present (backward compat) and still True here.
    assert groups[key]["rating_disagreement"] is True


def test_variance_summary_by_language_modal_agreement_rate_distinguishes_saturated_groups():
    # The exact scenario from the module task: "Hold,Buy,Buy" and
    # "Hold,Hold,Buy" both read as 100% under the old binary metric but are
    # NOT equally stable — modal_agreement_rate must show the same number
    # here (both 2/3) while still being meaningfully below 1.0, unlike the
    # binary rate which floors both cases at 100%.
    records = [
        _record(ticker="AAA", repeat=1, rating="Hold"),
        _record(ticker="AAA", repeat=2, rating="Buy"),
        _record(ticker="AAA", repeat=3, rating="Buy"),
        _record(ticker="BBB", repeat=1, rating="Hold"),
        _record(ticker="BBB", repeat=2, rating="Hold"),
        _record(ticker="BBB", repeat=3, rating="Buy"),
    ]
    summary = report.variance_summary_by_language(records)
    assert summary["ko"]["modal_agreement_rate"] == 2 / 3
    assert summary["ko"]["rating_disagreement_rate"] == 1.0  # old metric: both saturate to 100%


def test_variance_summary_by_language_avg_confidence_stdev_is_none_not_nan_when_no_confidence():
    records = [
        _record(repeat=1, rating="Buy", confidence=None),
        _record(repeat=2, rating="Buy", confidence=None),
    ]
    summary = report.variance_summary_by_language(records)
    assert summary["ko"]["avg_confidence_stdev"] is None


def test_render_report_prints_na_not_nan_for_missing_confidence_stdev():
    records = [
        _record(language="ko", repeat=1, rating="Buy", confidence=None),
        _record(language="ko", repeat=2, rating="Buy", confidence=None),
    ]
    out = report.render_report(records, errors=[])
    assert "nan" not in out.lower()
    assert "n/a (no confidence values)" in out


# ---------------------------------------------------------------------------
# cross-language disagreement (module task fix #2 — new metric)
# ---------------------------------------------------------------------------


def test_cross_language_disagreement_rate_flags_a_differing_cell():
    records = [
        _record(language="ko", ticker="X", date="2026-01-01", repeat=1, rating="Buy"),
        _record(language="en", ticker="X", date="2026-01-01", repeat=1, rating="Hold"),
    ]
    assert report.cross_language_disagreement_rate(records) == 1.0


def test_cross_language_disagreement_rate_zero_when_modal_ratings_match():
    records = [
        _record(language="ko", ticker="X", date="2026-01-01", repeat=1, rating="Buy"),
        _record(language="ko", ticker="X", date="2026-01-01", repeat=2, rating="Hold"),  # ko modal still Buy (tie->either, but here 1x1 tie... use 3 reps for clarity)
        _record(language="en", ticker="X", date="2026-01-01", repeat=1, rating="Buy"),
    ]
    # Make ko's modal unambiguous: Buy,Buy,Hold -> modal Buy.
    records[1]["final_decision"]["rating"] = "Buy"
    records.append(_record(language="ko", ticker="X", date="2026-01-01", repeat=3, rating="Hold"))
    assert report.cross_language_disagreement_rate(records) == 0.0


def test_cross_language_disagreement_rate_uses_pre_gate_rating():
    # ko's post-gate rating is Hold, but pre-gate (recovered from the
    # hysteresis note) it was Buy — the comparison must use the pre-gate
    # judgment per module task spec ("게이트 이전 rating 기준으로 계산해라").
    ko = _record(
        language="ko", ticker="X", date="2026-01-01", repeat=1,
        rating="Hold", confidence=None, investment_thesis="x" + _GATE_NOTE,
    )
    en = _record(language="en", ticker="X", date="2026-01-01", repeat=1, rating="Buy")
    assert report.cross_language_disagreement_rate([ko, en]) == 0.0


def test_cross_language_disagreement_rate_skips_cells_missing_one_language():
    records = [_record(language="ko", ticker="X", date="2026-01-01", repeat=1, rating="Buy")]
    assert report.cross_language_disagreement_rate(records) is None


def test_cross_language_disagreement_rate_averages_across_multiple_cells():
    records = [
        _record(language="ko", ticker="X", date="2026-01-01", repeat=1, rating="Buy"),
        _record(language="en", ticker="X", date="2026-01-01", repeat=1, rating="Buy"),  # agree
        _record(language="ko", ticker="Y", date="2026-01-01", repeat=1, rating="Sell"),
        _record(language="en", ticker="Y", date="2026-01-01", repeat=1, rating="Hold"),  # disagree
    ]
    assert report.cross_language_disagreement_rate(records) == 0.5


def test_render_report_includes_cross_language_disagreement_section():
    records = [
        _record(language="ko", ticker="X", date="2026-01-01", repeat=1, rating="Buy"),
        _record(language="en", ticker="X", date="2026-01-01", repeat=1, rating="Hold"),
    ]
    out = report.render_report(records, errors=[])
    assert "Cross-language disagreement" in out
    assert "ko vs en: 100.0%" in out


# ---------------------------------------------------------------------------
# cost
# ---------------------------------------------------------------------------


def test_avg_elapsed_by_language():
    records = [
        _record(language="ko", elapsed_seconds=100.0),
        _record(language="ko", elapsed_seconds=200.0),
        _record(language="en", elapsed_seconds=50.0),
    ]
    result = report.avg_elapsed_by_language(records)
    assert result["ko"] == 150.0
    assert result["en"] == 50.0


# ---------------------------------------------------------------------------
# isolation violations
# ---------------------------------------------------------------------------


def test_isolation_violation_records_filters_to_flagged_runs():
    clean = _record(isolation_violation=False)
    dirty = _record(isolation_violation=True)
    assert report.isolation_violation_records([clean, dirty]) == [dirty]


# ---------------------------------------------------------------------------
# render_report — smoke test only (exact formatting isn't the contract)
# ---------------------------------------------------------------------------


def test_render_report_runs_without_error_and_mentions_both_languages():
    records = [
        _record(language="ko", repeat=1),
        _record(language="ko", repeat=2),
        _record(language="en", repeat=1, executive_summary="Bullish.", investment_thesis="Buy."),
    ]
    text = report.render_report(records, errors=[])
    assert "ko" in text
    assert "en" in text
    assert "Isolation violations" in text


# ---------------------------------------------------------------------------
# Degraded-run exclusion (4th design fix)
# ---------------------------------------------------------------------------


def _rec(language, repeat, *, degraded=False, elapsed=100.0, ticker="AAPL", date_="2026-07-01"):
    return {
        "ticker": ticker,
        "date": date_,
        "language": language,
        "repeat": repeat,
        "elapsed_seconds": elapsed,
        "degraded": degraded,
        "degraded_reports": ["Fundamentals Analyst"] if degraded else [],
        "reports": {"Market Analyst": "clean"},
        "final_decision": {"rating": "BUY", "confidence": 0.6},
    }


def test_degraded_records_picks_out_only_the_degraded_ones():
    records = [_rec("ko", 1), _rec("ko", 2, degraded=True), _rec("en", 1)]
    assert [r["repeat"] for r in report.degraded_records(records)] == [2]


def test_render_report_excludes_degraded_runs_from_the_counts():
    records = [_rec("ko", 1), _rec("ko", 2, degraded=True), _rec("en", 1), _rec("en", 2)]
    out = report.render_report(records, errors=[])
    assert "3 analysed run(s), 1 degraded (excluded)" in out
    assert "EXCLUDED from every metric" in out


def test_render_report_keeps_a_degraded_runs_cost_out_of_the_average():
    # The degraded ko run is 10x slower; if it leaked into the mean, ko's
    # average would be dragged far above en's and read as a language cost.
    records = [_rec("ko", 1, elapsed=100.0), _rec("ko", 2, degraded=True, elapsed=1000.0), _rec("en", 1, elapsed=100.0)]
    out = report.render_report(records, errors=[])
    assert "ko: 100.0s" in out


def test_render_report_says_so_when_every_run_degraded():
    out = report.render_report([_rec("ko", 1, degraded=True)], errors=[])
    assert "No clean runs to analyse." in out


# ---------------------------------------------------------------------------
# pre_gate_rating (Part B1) — recovering the model's rating before
# _apply_hysteresis_gate silently downgraded it to Hold
# ---------------------------------------------------------------------------

# Verbatim note format from app/orchestrator/pipeline.py's
# _apply_hysteresis_gate (lines 103-106) — any test asserting against this
# string must stay byte-for-byte in sync with that function's f-string.
_GATE_NOTE = (
    "\n\n[Hysteresis gate: downgraded from Buy to Hold — confidence "
    "(unset) below the 60% bar required to flip position state.]"
)

# The actual ko_r2 pilot record's executive_summary, verbatim, from the same
# file — this is where the explicit "Buy 등급을 부여합니다" (space before the
# assignment noun) claim actually lives; "Buy로 확정합니다" further down in
# the thesis has no space before the Korean particle, which the KO claim
# regex's \b boundary does not match (word-boundary logic treats a Hangul
# syllable as a \w character in Python's Unicode-aware re). Both are real
# excerpts from the same record; using them together (as the real
# self_consistency_violations call does — it concatenates both fields) is
# what makes this a faithful regression test rather than a synthetic one.
_KO_R2_EXEC_SUMMARY = (
    "현재 평지 상태를 기반으로 Buy 등급을 부여합니다. 14일 ATR 6.8%의 높은 변동성으로 인해 시스템 "
    "자동 스케일링이 적용되며, 실제 집행 가능한 최대 노출액은 총 자산의 30% 수준으로 제한됩니다. "
    "또한, 추가 매수분의 달러 리스크가 총자산 1%(1,418,270원)를 초과하지 못하도록 코드가 강제하므로, "
    "손절가 310,000원 설정 시 허용되는 신규 매수 주식 수는 약 56주로 제한됩니다. 이는 변동성 "
    "스케일링보다 리스크 캡이 더 긴밀하게 작동함을 의미하며, 당초 제안된 100% 집중 매수 대신 "
    "체계적인 분할 진입과 엄격한 리스크 관리를 전제로 합니다."
)

# The actual ko_r2 pilot record's investment_thesis, verbatim, from
# results/language_ab/pilot_009540_2026-07-15/009540.KS_2026-07-15_ko_r2.json
# — a real-data regression test, not a synthetic approximation.
_KO_R2_THESIS = (
    "트레이더의 매수 계획과 연구관리자의 Overweight 권고를 수용하되, 평지 진입 제약을 고려해 "
    "Buy로 확정합니다. 펀더멘털 측면에서 2026년 3분기 영업이익률 16.6%, ROE 20.2%, EV/EBITDA "
    "3.97x는 역사적 저평가 구간을 명확히 보여줍니다. 향후 목표주장이 550,000~600,000원으로 "
    "책정될 정도로 상승 여력(70~90%)이 확보되었으며, 8월 13일 예정된 6,500원 배당은 하방 리스크를 "
    "흡수하는 안전마진 역할을 합니다. 조선업종의 구조적 수주 호조와 실적 실현 단계에서는 이러한 "
    "펀더멘털이 가격의 하단을 든든하게 받칩니다.\n\n"
    "기술적으로는 MACD 데드크스와 RSI의 중립선 방향성, 그리고 12~16%대의 쇼트 비중 증가가 단기 "
    "모멘텀을 위축시키고 있습니다. 그러나 이는 320,000~330,000원 구간에서의 지지선 테스트로 "
    "해석되며, FINAL SYNTHESIS에서 지적된 대로 기술적 약세가 장기 펀더멘털의 우위를 상쇄하기에는 "
    "부족합니다. Conservative 디베이터가 제기한 기술적 붕괴 우려는 리스크 대 보상 비율에서 "
    "기각됩니다. \n\n"
    "변동성 스케일링(Buy 시 30% 노출 제한)과 리스크 캡(신규 매수분 리스크 1% 이내)이 중복 "
    "적용되므로, 당초 100% 매수보다는 30% 수준의 제한적 진입이 먼저 이루어집니다. 310,000원 붕괴 "
    "시 손절 규칙을 철저히 준수하면, 320,000원 지지선 방어 성공 시 추가 매수로 포지션을 확대할 수 "
    "있는 구조를 갖춥니다. 현재 환경에서는 펀더멘털의 저평가 폭과 배당 안전마진이 단기 기술적 "
    "헤드윈을 충분히 커버한다고 판단됩니다.\n\n"
    "[Hysteresis gate: downgraded from Buy to Hold — confidence (unset) below the 60% bar required "
    "to flip position state.]"
)


def test_pre_gate_rating_returns_current_rating_when_no_note():
    rec = _record(rating="Hold", investment_thesis="Plain thesis, no gate note.")
    assert report.pre_gate_rating(rec) == ("Hold", False)


def test_pre_gate_rating_recovers_the_original_rating_from_the_note():
    rec = _record(rating="Hold", investment_thesis="Some reasoning." + _GATE_NOTE)
    assert report.pre_gate_rating(rec) == ("Buy", True)


def test_pre_gate_rating_raises_loudly_on_a_broken_note_format():
    # Marker present, but the rest of the note doesn't match the fixed
    # format — must NOT silently fall back to "no downgrade" (module spec:
    # a quiet failure here would corrupt every downstream metric).
    broken = "reasoning\n\n[Hysteresis gate: something changed about this format]"
    rec = _record(rating="Hold", investment_thesis=broken)
    import pytest as _pytest
    with _pytest.raises(report.HysteresisNoteParseError):
        report.pre_gate_rating(rec)


def test_pre_gate_rating_matches_the_real_ko_r2_record():
    rec = _record(rating="Hold", confidence=None, investment_thesis=_KO_R2_THESIS)
    assert report.pre_gate_rating(rec) == ("Buy", True)


def test_pre_gate_rating_safe_flags_parse_failure_without_raising():
    broken = "reasoning\n\n[Hysteresis gate: unexpected format here]"
    rec = _record(rating="Hold", investment_thesis=broken)
    rating, downgraded, parse_failed = report._pre_gate_rating_safe(rec)
    assert parse_failed is True
    assert downgraded is False  # not silently counted as "not downgraded" in the loud sense — see flag


def test_gate_note_parse_failures_collects_the_broken_records():
    broken = _record(rating="Hold", investment_thesis="x\n\n[Hysteresis gate: broken]")
    clean = _record(rating="Hold", investment_thesis="fine" + _GATE_NOTE)
    result = report.gate_note_parse_failures([broken, clean])
    assert result == [broken]


# ---------------------------------------------------------------------------
# pre_gate_rating — the desk-consensus override gate marker (2026-09-04,
# id=1212/1217 — app.orchestrator.pipeline._apply_consensus_override_gate)
# ---------------------------------------------------------------------------

_CONSENSUS_NOTE = (
    "\n\n[Consensus override gate: downgraded from Overweight to Hold — the Research "
    "Manager (Underweight) and the Trader (Sell) both declined more exposure, and confidence "
    "(unset) is below the 85% bar required to increase exposure against them.]"
)


def test_pre_gate_rating_recovers_the_original_rating_from_a_consensus_note():
    rec = _record(rating="Hold", investment_thesis="Some reasoning." + _CONSENSUS_NOTE)
    assert report.pre_gate_rating(rec) == ("Overweight", True)


def test_pre_gate_rating_raises_loudly_on_a_broken_consensus_note():
    broken = "reasoning\n\n[Consensus override gate: format changed]"
    rec = _record(rating="Hold", investment_thesis=broken)
    import pytest as _pytest

    with _pytest.raises(report.HysteresisNoteParseError):
        report.pre_gate_rating(rec)


def test_pre_gate_rating_safe_flags_a_broken_consensus_note():
    rec = _record(rating="Hold", investment_thesis="x\n\n[Consensus override gate: broken]")
    rating, downgraded, parse_failed = report._pre_gate_rating_safe(rec)
    assert parse_failed is True
    assert downgraded is False


# ---------------------------------------------------------------------------
# gate downgrade counts + confidence-missing rate (Part B3)
# ---------------------------------------------------------------------------


def test_gate_downgrade_stats_counts_downgrades_and_confidence_unset():
    gated_conf_unset = _record(language="ko", repeat=1, rating="Hold", confidence=None, investment_thesis="x" + _GATE_NOTE)
    not_gated = _record(language="ko", repeat=2, rating="Buy", confidence=0.8, investment_thesis="fine")
    stats = report.gate_downgrade_stats_by_language([gated_conf_unset, not_gated])
    assert stats["ko"]["n_runs"] == 2
    assert stats["ko"]["downgraded"] == 1
    assert stats["ko"]["downgraded_confidence_unset"] == 1


def test_confidence_missing_rate_by_language():
    records = [
        _record(language="ko", confidence=None),
        _record(language="ko", confidence=0.7),
        _record(language="en", confidence=0.5),
    ]
    rates = report.confidence_missing_rate_by_language(records)
    assert rates["ko"] == 0.5
    assert rates["en"] == 0.0


# ---------------------------------------------------------------------------
# stale_prose_after_gate_override reclassification (Part B4)
# ---------------------------------------------------------------------------


def test_gated_contradiction_is_reclassified_as_stale_prose_not_self_consistency():
    rec = _record(
        rating="Hold", confidence=None,
        executive_summary=_KO_R2_EXEC_SUMMARY, investment_thesis=_KO_R2_THESIS,
    )
    violations = report.self_consistency_violations(rec)
    assert "stale_prose_after_gate_override" in violations
    assert "rating_claim_contradicts_field" not in violations


def test_ungated_contradiction_still_counts_as_self_consistency_violation():
    # Same contradiction shape, but no hysteresis note — a genuine
    # self-consistency bug, must stay in the language-attributable bucket.
    rec = _record(
        rating="Hold",
        executive_summary="Buy 등급을 부여합니다.",
        investment_thesis="No gate note here at all.",
    )
    violations = report.self_consistency_violations(rec)
    assert "rating_claim_contradicts_field" in violations
    assert "stale_prose_after_gate_override" not in violations


def test_render_report_splits_stale_prose_from_language_self_consistency():
    ko_gated = _record(
        language="ko", repeat=1, rating="Hold", confidence=None,
        executive_summary=_KO_R2_EXEC_SUMMARY, investment_thesis=_KO_R2_THESIS,
    )
    text = report.render_report([ko_gated], errors=[])
    assert "stale_prose_after_gate_override" in text
    assert "language-attributable only" in text


# ---------------------------------------------------------------------------
# variance pre-gate vs post-gate (Part B2)
# ---------------------------------------------------------------------------


def test_variance_by_group_pre_gate_recovers_the_original_rating():
    records = [
        _record(repeat=1, rating="Buy", confidence=0.8, investment_thesis="fine"),
        _record(repeat=2, rating="Hold", confidence=None, investment_thesis="x" + _GATE_NOTE),  # was Buy pre-gate
    ]
    pre = report.variance_by_group(records, pre_gate=True)
    post = report.variance_by_group(records, pre_gate=False)
    key = ("AAPL", "2026-07-01", "ko")
    # Post-gate: Buy vs Hold -> disagreement. Pre-gate: Buy vs Buy -> agreement.
    assert post[key]["rating_disagreement"] is True
    assert pre[key]["rating_disagreement"] is False


def test_variance_by_group_pre_gate_excludes_unparseable_notes_rather_than_guessing():
    records = [
        _record(repeat=1, rating="Buy", confidence=0.8, investment_thesis="fine"),
        _record(repeat=2, rating="Hold", confidence=None, investment_thesis="x\n\n[Hysteresis gate: broken]"),
    ]
    pre = report.variance_by_group(records, pre_gate=True)
    key = ("AAPL", "2026-07-01", "ko")
    # Only the one parseable rating (Buy) contributes -> no disagreement signal
    # possible from a single rating, and the group must not crash.
    assert pre[key]["rating_disagreement"] is False


def test_variance_summary_by_language_matches_the_pilot_data_expectation():
    # Mirrors the real pilot outcome (results/language_ab/pilot_009540_2026-07-15):
    # ko pre-gate ratings are Hold, Buy(recovered), Buy -> disagreement.
    # en ratings are Hold, Hold, Hold -> no disagreement.
    ko_records = [
        _record(language="ko", repeat=1, rating="Hold", confidence=None, investment_thesis="plain hold"),
        _record(language="ko", repeat=2, rating="Hold", confidence=None, investment_thesis="x" + _GATE_NOTE),
        _record(language="ko", repeat=3, rating="Buy", confidence=0.82, investment_thesis="plain buy"),
    ]
    en_records = [_record(language="en", repeat=r, rating="Hold", confidence=0.6) for r in (1, 2, 3)]
    summary = report.variance_summary_by_language(ko_records + en_records, pre_gate=True)
    assert summary["ko"]["rating_disagreement_rate"] == 1.0
    assert summary["en"]["rating_disagreement_rate"] == 0.0


# ---------------------------------------------------------------------------
# LLM token metrics section (Part A3) — graceful "not instrumented" handling
# ---------------------------------------------------------------------------


def test_llm_metrics_present_is_false_when_no_record_has_the_field():
    records = [_record(language="ko"), _record(language="en")]
    assert report.llm_metrics_present(records) is False


def test_llm_metrics_present_is_true_when_at_least_one_record_has_it():
    with_llm = _record(language="ko")
    with_llm["llm_calls"] = 5
    assert report.llm_metrics_present([with_llm, _record(language="en")]) is True


def test_avg_llm_stats_by_language_none_safe_token_sums():
    rec = _record(language="ko")
    rec.update({
        "llm_calls": 2,
        "llm_prompt_tokens": 100,
        "llm_completion_tokens": None,  # one call reported no usage
        "llm_total_tokens": 100,
        "llm_fallback_calls": 1,
        "llm_by_label": {"trader": {"calls": 2, "total_tokens": 100}},
    })
    stats = report.avg_llm_stats_by_language([rec])
    assert stats["ko"]["avg_calls"] == 2
    assert stats["ko"]["avg_completion_tokens"] == 0  # None treated as 0, not skipped
    assert stats["ko"]["fallback_rate"] == 0.5


def test_avg_llm_stats_by_language_all_zero_calls_is_none_not_nan():
    """FINDING B4 regression: avg_tokens_per_call/fallback_rate returned
    float("nan") for a language whose runs all recorded 0 calls, unlike the
    sibling avg_confidence_stdev, which deliberately uses None (see its own
    docstring: a bare nan in the printed report reads like a calculation
    bug, not "nothing to compute")."""
    rec = _record(language="ko")
    rec.update({
        "llm_calls": 0, "llm_prompt_tokens": 0, "llm_completion_tokens": 0,
        "llm_total_tokens": 0, "llm_fallback_calls": 0,
    })
    stats = report.avg_llm_stats_by_language([rec])
    assert stats["ko"]["avg_tokens_per_call"] is None
    assert stats["ko"]["fallback_rate"] is None


def test_render_report_prints_na_not_nan_for_all_zero_call_language():
    rec = _record(language="ko")
    rec.update({
        "llm_calls": 0, "llm_prompt_tokens": 0, "llm_completion_tokens": 0,
        "llm_total_tokens": 0, "llm_fallback_calls": 0,
    })
    text = report.render_report([rec], errors=[])
    assert "nan" not in text.lower()
    assert "tok/call=n/a" in text
    assert "fallback_rate=n/a" in text


def test_render_report_says_not_instrumented_when_llm_fields_absent():
    records = [_record(language="ko"), _record(language="en")]
    text = report.render_report(records, errors=[])
    assert "not instrumented" in text
