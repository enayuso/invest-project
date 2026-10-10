"""app.agents.unit_scale_guard — no network, no LLM (A-stage).

Real incidents this guards against (2026-08-28 domestic AI audit): a
Fundamentals Analyst report restated a correctly-supplied 억원 figure as
"N billion KRW" 10x too large, because `_eok`'s >=1조 branch used to give
only a trillion-KRW gloss — see kis_provider._eok's own 2026-08-28 addendum.
"""

from __future__ import annotations

import pytest

from app.agents.unit_scale_guard import (
    UnitScaleViolation,
    check_unit_scale,
    enforce_unit_scale,
    extract_kr_eok_values,
    extract_kr_headline_eok_values,
    scan_unit_scale,
)


def test_extract_kr_eok_values_reads_every_eok_figure():
    block = "revenue=951,542억원(95.15조원) net_income=12,345억원"
    assert extract_kr_eok_values(block) == frozenset({951_542.0, 12_345.0})


def test_extract_kr_eok_values_empty_when_no_kr_data():
    assert extract_kr_eok_values("no korean figures here") == frozenset()


def test_check_unit_scale_passes_a_correct_billion_figure():
    # 951,542억원 = 95,154.2 billion KRW — the correct conversion.
    eok = frozenset({951_542.0})
    report = "Revenue reached 95,154.2 billion KRW this year."
    assert check_unit_scale(report, eok) == []


def test_check_unit_scale_flags_the_hyundai_incident_shape():
    # Real incident: 현대차 951,542억원 restated as "951,542 billion KRW"
    # (should be 95,154.2 billion) — the 억원 mantissa carried straight over.
    eok = frozenset({951_542.0})
    report = "Revenue ... (951,542 billion KRW annualized run-rate)"
    violations = check_unit_scale(report, eok)
    assert len(violations) == 1
    assert violations[0].kind == "billion_10x"
    assert violations[0].expected == "95,154.2 billion KRW"


def test_check_unit_scale_flags_the_doosan_self_inconsistent_shape():
    # Real incident: 두산에너빌리티 89,859억원 (= 8,985.9 billion = 8.9859
    # trillion) restated as "89,859 billion KRW (8.99 trillion KRW)" — the
    # two numbers in the same sentence already disagree by 10x, AND the
    # billion figure doesn't match the supplied 억원 mantissa/10 either.
    eok = frozenset({89_859.0})
    report = "Revenue reached 89,859 billion KRW (8.99 trillion KRW)"
    violations = check_unit_scale(report, eok)
    kinds = {v.kind for v in violations}
    assert "billion_10x" in kinds  # 89,859 matches the 억원 mantissa exactly


def test_check_unit_scale_flags_trillion_10x():
    # 951,542억원 = 95.1542 trillion KRW; writing 9.51542 trillion is a 10x
    # UNDER-statement of the same figure.
    eok = frozenset({951_542.0})
    report = "Revenue reached 9.51542 trillion KRW."
    violations = check_unit_scale(report, eok)
    assert len(violations) == 1
    assert violations[0].kind == "trillion_10x"


def test_check_unit_scale_no_false_positive_on_unrelated_large_number():
    # A billion-KRW figure that matches NEITHER a supplied 억원 mantissa NOR
    # any correct conversion of one must not be flagged — this guard only
    # catches a KNOWN mislabel, never guesses at an unrelated figure (e.g. a
    # price target, a peer's market cap mentioned for comparison).
    eok = frozenset({951_542.0})
    report = "The peer trades at 50,000 billion KRW market cap."
    assert check_unit_scale(report, eok) == []


def test_check_unit_scale_returns_nothing_when_no_kr_data_supplied():
    # Fail-open, same posture as identity_gate.check_identity when there is
    # nothing to check against — an empty ground-truth set must never be
    # treated as "everything is wrong".
    assert check_unit_scale("Revenue reached 999,999 billion KRW.", frozenset()) == []


def test_check_unit_scale_accept_first_avoids_a_coincidental_10x_overlap():
    # Two DIFFERENT supplied 억원 figures sit in an exact 10:1 ratio
    # (12,345 and 123,450) — a report correctly quoting 12,345's own true
    # billion value (1,234.5) must never be flagged just because 123,450/10
    # also equals 12,345.
    eok = frozenset({12_345.0, 123_450.0})
    report = "Segment revenue was 1,234.5 billion KRW."
    assert check_unit_scale(report, eok) == []


def test_enforce_unit_scale_corrects_the_span_and_adds_a_banner():
    kr_block = "revenue=951,542억원(95.15조원)"
    report = "Revenue ... (951,542 billion KRW annualized run-rate)"
    corrected, violations = enforce_unit_scale(report, kr_block, language="en", agent_name="Fundamentals Analyst")
    assert len(violations) == 1
    assert "95,154.2 billion KRW" in corrected
    assert '[unit corrected from "951,542 billion KRW"]' in corrected
    assert "951,542 billion KRW annualized" not in corrected  # the wrong span is gone, not just annotated
    assert "UNIT-SCALE CORRECTION" in corrected


def test_enforce_unit_scale_noop_when_clean():
    kr_block = "revenue=951,542억원(95.15조원)"
    report = "Revenue reached 95,154.2 billion KRW."
    corrected, violations = enforce_unit_scale(report, kr_block, language="en")
    assert corrected == report
    assert violations == []


def test_enforce_unit_scale_self_inconsistent_gets_a_note_not_a_replacement():
    # No supplied ground truth pins down which of the two disagreeing
    # numbers is right, so the text is left as-is (no replacement possible)
    # with only an advisory note appended.
    kr_block = "no matching eok figure here"
    report = "Revenue reached 50,000 billion KRW (4 trillion KRW)."
    corrected, violations = enforce_unit_scale(report, kr_block, language="en")
    assert violations  # flagged
    assert "50,000 billion KRW (4 trillion KRW)" in corrected  # left in place
    assert "self-inconsistent" in corrected


def test_enforce_unit_scale_banner_false_suppresses_the_trailing_flag_note():
    # 2026-09-09: a length-budgeted / structured field (banner=False) gets no
    # trailing advisory note, but the finding is still returned.
    report = "Revenue reached 50,000 billion KRW (4 trillion KRW)."
    with_note, v1 = enforce_unit_scale(report, "no eok here", language="en")
    no_note, v2 = enforce_unit_scale(report, "no eok here", language="en", banner=False)
    assert "self-inconsistent" in with_note
    assert "self-inconsistent" not in no_note
    assert report in no_note                 # text itself untouched
    assert v1 == v2 and v2                    # same finding, still reported


def test_enforce_unit_scale_reverse_order_handles_multiple_violations():
    # Two corrections in one report — must not corrupt offsets when applying
    # the second replacement after the first has already changed the string
    # length.
    kr_block = "a=951,542억원(95.15조원) b=12,345억원"
    report = "A reached 951,542 billion KRW. B reached 1,234.5 billion KRW is correct, but C claims 12,345 billion KRW."
    corrected, violations = enforce_unit_scale(report, kr_block, language="en")
    assert len(violations) == 2
    assert "95,154.2 billion KRW" in corrected
    assert "1,234.5 billion KRW is correct" in corrected  # untouched, was already right
    assert '[unit corrected from "12,345 billion KRW"]' in corrected


def test_enforce_unit_scale_korean_banner():
    kr_block = "revenue=951,542억원(95.15조원)"
    report = "매출은 951,542 billion KRW 입니다."
    corrected, violations = enforce_unit_scale(report, kr_block, language="ko")
    assert violations
    assert "단위 오류 정정" in corrected


def test_unit_scale_violation_is_frozen_dataclass():
    v = UnitScaleViolation(kind="billion_10x", written="x", expected="y", start=0, end=1)
    assert v.kind == "billion_10x"


# --- 2026-09-01 blind-spot fixes (decisions id=1156/1157/1158) ---

_KIA_EOK = frozenset(
    {625_389.0, 41_579.0, 126_671.0, 90_781.0, 48_337.0, 806_278.0, 1_068_294.0, 465_582.0, 646_892.0}
)

_KIA_SPANS = [
    ("H1 2026 revenue was ₩625,389 billion, up", "62,538.9 billion KRW"),
    ("H1 2026 net income was ₩41,579 billion overall", "4,157.9 billion KRW"),
    ("declining from ₩126,671 billion in FY2024", "12,667.1 billion KRW"),
    ("to ₩90,781 billion in FY2025", "9,078.1 billion KRW"),
    ("and ₩48,337 billion in H1 2026", "4,833.7 billion KRW"),
    ("from ₩806,278 billion in FY2023", "80,627.8 billion KRW"),
    ("total assets of ₩1,068,294 billion (H1 2026)", "106,829.4 billion KRW"),
    ("from ₩465,582 billion in FY2023", "46,558.2 billion KRW"),
    ("to ₩646,892 billion in H1 2026", "64,689.2 billion KRW"),
]


@pytest.mark.parametrize("text,expected", _KIA_SPANS)
def test_prefix_won_billion_is_caught(text, expected):
    # The 9/9 blind spot: `₩` written BEFORE the number, unit word followed by
    # a comma / " in " / "(" — invisible to the old trailing-currency regex.
    violations = check_unit_scale(text, _KIA_EOK)
    assert len(violations) == 1
    assert violations[0].kind == "billion_10x"
    assert violations[0].expected == expected


def test_all_nine_kia_spans_in_one_report():
    report = " ".join(t for t, _ in _KIA_SPANS)
    corrected, violations = enforce_unit_scale(report, "\n".join(f"x={int(e)}억원" for e in _KIA_EOK))
    assert len(violations) == 9
    for _, expected in _KIA_SPANS:
        assert f'{expected} [unit corrected from "' in corrected
    # idempotent: a second pass over corrected text re-flags nothing
    _, again = enforce_unit_scale(corrected, "\n".join(f"x={int(e)}억원" for e in _KIA_EOK))
    assert again == []


def test_approx_won_trillion_bare_T_is_caught_but_not_under_rel_tol():
    eok = frozenset({48_337.0})
    text = "operating profit is still positive at ~₩48T in H1"
    violations = check_unit_scale(text, eok)
    assert len(violations) == 1
    assert violations[0].kind == "trillion_10x"
    assert violations[0].expected == "4.83 trillion KRW"


def test_market_cap_tier_mention_not_flagged_magnitude_suspect():
    """2026-09-08 fix, id=1315 (387690.KS/레메디): 'This is common for
    micro-cap stocks (<₩100B market cap)' is a generic industry-tier
    definition, not a restatement of this company's own supplied figure —
    ₩100B happened to land inside magnitude_suspect's power-of-10 tolerance
    band against an unrelated ground-truth value."""
    report = (
        "This is common for micro-cap stocks (<₩100B market cap) due to low "
        "trading volume and lack of institutional demand."
    )
    assert check_unit_scale(report, frozenset({10_000.0})) == []


def test_market_cap_tier_mention_not_flagged_billion_10x_either():
    """The same exclusion must protect Tier 1 (auto-correcting) too, not
    only the kind this was first noticed on — a boilerplate tier
    definition can coincidentally land on an EXACT 10x match just as
    easily as an approximate one."""
    report = "This is common for micro-cap stocks (<₩100B market cap) due to low trading volume."
    assert check_unit_scale(report, frozenset({100.0})) == []


def test_comparator_without_market_cap_still_flagged():
    """Regression guard for the exclusion's own scope: a comparator word
    alone (no 'market cap' nearby) must NOT be excluded — 900270.KS
    (id=1259) discusses this company's own revenue/assets using hedged
    'under'/'over' language, a genuine anomaly worth flagging, not a
    boilerplate tier definition."""
    report = "The revenue base is relatively small (<120 billion KRW annually), making it susceptible to shocks."
    violations = check_unit_scale(report, frozenset({120.0}))
    assert [v.kind for v in violations] == ["billion_10x"]


def test_bare_B_suffix_behind_won_is_magnitude_suspect_only():
    eok = frozenset({48_337.0})
    for text in ("reported ₩48B in H1", "reported ₩48.3B in H1"):
        violations = check_unit_scale(text, eok)
        assert [v.kind for v in violations] == ["magnitude_suspect"]
        assert violations[0].expected is None
        corrected, _ = enforce_unit_scale(text, "x=48337억원")
        assert text.split()[1] in corrected  # number left in place


def test_jo_won_notation_correct_is_accepted_wrong_is_flagged():
    eok = frozenset({625_389.0})
    assert check_unit_scale("revenue of 62.54조원", eok) == []
    wrong = check_unit_scale("revenue of 625,389조원", eok)
    assert len(wrong) == 1 and wrong[0].kind == "trillion_10x"


def test_usd_trillion_is_opt_in_flag_only():
    text = "government earmarked $3 trillion for EV incentives"
    assert check_unit_scale(text, frozenset()) == []
    violations, _ = scan_unit_scale(text, frozenset(), usd_check=True)
    assert [v.kind for v in violations] == ["usd_unit_suspect"]
    assert violations[0].expected is None
    corrected, descriptions = enforce_unit_scale(text, "", usd_check=True)
    assert descriptions and "usd_unit_suspect" in descriptions[0]


def test_bare_B_T_without_currency_prefix_is_not_flagged():
    eok = frozenset({48.0})
    assert check_unit_scale("The company's Plan B and Tier T rollout continues", eok) == []
    assert check_unit_scale("Version 3T of the platform, a 48B parameter model", eok) == []


def test_prefix_form_written_correctly_is_not_flagged():
    eok = frozenset({951_542.0})
    assert check_unit_scale("H1 revenue was ₩95,154.2 billion", eok) == []
    assert check_unit_scale("revenue of ₩4.8T", frozenset({48_337.0})) == []


def test_inline_note_false_replaces_span_with_bare_value():
    corrected, violations = enforce_unit_scale(
        "카드 요약: ₩625,389 billion 매출", "x=625389억원", banner=False, inline_note=False,
    )
    assert "62,538.9 billion KRW" in corrected
    assert "[unit corrected from" not in corrected
    assert "UNIT-SCALE CORRECTION" not in corrected
    assert len(violations) == 1


def test_scan_record_positive_signal():
    report = " ".join(t for t, _ in _KIA_SPANS)
    violations, scan = scan_unit_scale(report, _KIA_EOK, node="Fundamentals Analyst")
    assert scan.candidates_examined == 9
    assert scan.corrected == 9
    assert scan.suspicious is False


def test_scan_record_flags_a_blind_spot():
    # candidates present, ground truth present, but nothing matches — the
    # exact shape id=1158 presented before this fix.
    report = "revenue ₩625,389 billion and profit ₩48,337 billion"
    violations, scan = scan_unit_scale(report, frozenset({111_111.0, 222_222.0}), node="x")
    assert scan.candidates_examined == 2
    assert scan.corrected == 0
    assert scan.unverified == 2
    assert scan.suspicious is True


def test_scan_record_clean_report_not_suspicious():
    _, scan = scan_unit_scale("no large figures here", frozenset({111.0}), node="x")
    assert scan.candidates_examined == 0
    assert scan.suspicious is False


def test_enforce_unit_scale_scan_out_is_populated():
    out: list = []
    enforce_unit_scale("revenue ₩625,389 billion", "x=625389억원", scan_out=out)
    assert len(out) == 1 and out[0].corrected == 1


# --- 2026-09-02 sign / floor / Unicode-minus fixes (live decision id=1168,
# 024060.KS) — two interdependent defects on the SAME sign-comparison bug,
# plus a previously-undocumented auto-correct floor made explicit and
# corroboration-gated, plus a Unicode-minus (U+2212) regex gap that would
# actively corrupt output once the floor fix ships alone. ---

def test_id1168_bear_3billion_inflation_is_caught_ascii():
    # Real incident: Bear wrote "-3 billion KRW" for a loss Fundamentals
    # correctly gave as "-3억 원 (-0.3 billion KRW)" — a real 10x inflation
    # the pre-fix sign-asymmetric comparison silently passed as `unverified`.
    eok = frozenset({-3.0})
    report = "The company generated an operating loss in FY2024 (-3 billion KRW)"
    violations = check_unit_scale(report, eok)
    assert len(violations) == 1
    assert violations[0].kind == "billion_10x"
    assert violations[0].expected == "-0.3 billion KRW"


def test_id1168_bear_3billion_inflation_is_caught_unicode_minus():
    # Same shape, but the model used the proper Unicode minus U+2212 (−)
    # instead of ASCII "-". Identical result, and the violation's `written`
    # span must capture the sign character (proof `_SIGN` is doing its job,
    # not just that `_to_float` happens to tolerate it downstream).
    eok = frozenset({-3.0})
    report = "The company generated an operating loss in FY2024 (−3 billion KRW)"
    violations = check_unit_scale(report, eok)
    assert len(violations) == 1
    assert violations[0].kind == "billion_10x"
    assert violations[0].expected == "-0.3 billion KRW"
    assert violations[0].written.startswith("−")


def test_id1168_unicode_minus_correction_has_no_double_minus_corruption():
    # THE safety-critical assertion: before this fix, `_NUM`'s ASCII-only
    # `-?` would fail to capture the leading U+2212 into the matched span,
    # so the replacement (freshly signed off `sign`) would be spliced in
    # right after the untouched "−", producing "−-0.3 billion KRW". `_SIGN`
    # must make the whole span — sign included — get replaced cleanly.
    kr_block = "loss=-3억원"
    report = "The company generated an operating loss in FY2024 (−3 billion KRW)"
    corrected, violations = enforce_unit_scale(report, kr_block, language="en")
    assert len(violations) == 1
    assert "-0.3 billion KRW" in corrected
    assert "−-" not in corrected  # no double-minus artifact
    # The corrected NUMBER itself must be clean: no Unicode minus anywhere in
    # the replacement value (the marker's verbatim quote of the ORIGINAL
    # written span, "−3 billion KRW", legitimately still contains it).
    assert '−0.3' not in corrected
    assert corrected.count("−") == 1  # only inside the "[unit corrected from ...]" quote


def test_id1168_fundamentals_chain_all_accepted_ascii():
    # The OTHER defect from the same incident: a correctly-stated
    # "-9억 원 (-0.9 billion KRW)" was FALSELY flagged as `magnitude_suspect`
    # because the sign bug blocked it from being *accepted*, so it fell
    # through to the power-of-10 ladder and coincidentally collided with an
    # unrelated positive ground-truth value. All three figures here are
    # correctly stated and must be silently accepted — candidates_examined
    # counts them, unverified must be zero (not merely "no violations").
    eok = frozenset({-6.0, -16.0, -9.0})
    report = (
        "deepening from -6억 원 (-0.6 billion KRW) in FY2022 to -16억 원 "
        "(-1.6 billion KRW) in FY2023 to -9억 원 (-0.9 billion KRW) in FY2024"
    )
    violations, scan = scan_unit_scale(report, eok)
    assert violations == []
    assert scan.candidates_examined == 3
    assert scan.unverified == 0


def test_id1168_fundamentals_chain_all_accepted_unicode_minus():
    # The exact shape observed live (id=1168 used the proper Unicode minus)
    # — pinned separately from the ASCII twin above per the design note that
    # this false-positived even on ASCII before the fix.
    eok = frozenset({-6.0, -16.0, -9.0})
    report = (
        "deepening from −6억 원 (−0.6 billion KRW) in FY2022 to −16억 원 "
        "(−1.6 billion KRW) in FY2023 to −9억 원 (−0.9 billion KRW) in FY2024"
    )
    violations, scan = scan_unit_scale(report, eok)
    assert violations == []
    assert scan.candidates_examined == 3
    assert scan.unverified == 0


def test_bare_suffix_B_KRW_billion_10x():
    # 176B/48T bare-suffix currency (currency AFTER the number) was not
    # detected at all before this fix (candidates_examined: 0). This is the
    # same shape as id=1168's "-3B KRW" span.
    eok = frozenset({-3.0})
    violations = check_unit_scale("-3B KRW", eok)
    assert len(violations) == 1
    assert violations[0].kind == "billion_10x"
    assert violations[0].expected == "-0.3 billion KRW"


def test_bare_suffix_B_KRW_genuine_10x_over():
    # 176억원 = 17.6 billion KRW true value; "176B KRW" carries the mantissa
    # straight through — a genuine 10x error.
    eok = frozenset({176.0})
    violations = check_unit_scale("176B KRW", eok)
    assert len(violations) == 1
    assert violations[0].kind == "billion_10x"
    assert violations[0].expected == "17.6 billion KRW"


def test_bare_suffix_T_KRW_genuine_10x():
    # Same shape as the existing "~₩48T" trillion test, but with currency
    # as a bare suffix instead of a ₩ prefix.
    eok = frozenset({48_337.0})
    violations = check_unit_scale("48T KRW", eok)
    assert len(violations) == 1
    assert violations[0].kind == "trillion_10x"
    assert violations[0].expected == "4.83 trillion KRW"


@pytest.mark.parametrize(
    "text",
    [
        "a 48B parameter model",
        "Series B funding",
        "Grade B rating",
        "Version 3T of the platform",
    ],
)
def test_bare_suffix_false_positive_guards(text):
    # No currency context (no ₩/KRW prefix, no trailing KRW/won/₩) — must
    # not be claimed as a candidate at all.
    _, scan = scan_unit_scale(text, frozenset({48.0}))
    assert scan.candidates_examined == 0


def test_prefix_form_billion_still_regresses_clean():
    # Regression pin, not new behavior: the existing ₩-prefix bare-B form
    # must still work unchanged after adding the suffix alternative.
    eok = frozenset({48_337.0})
    violations = check_unit_scale("reported ₩48B in H1", eok)
    assert [v.kind for v in violations] == ["magnitude_suspect"]


# --- 2026-09-15 eok_pair_10x (id=1432, 040300.KS/YTN) ---
# Real incident: a Fundamentals Analyst report stated its OWN 억원 source
# figure and its OWN billion-KRW conversion in the same sentence, off by
# exactly 10x, for all three of its headline P&L figures.

_YTN_REVENUE_2022 = "Revenue peaked in FY2022 at 1,522억 원 ($15.22B KRW)."
_YTN_REVENUE_2024 = "FY2024 revenue was 1,312억 원 ($13.12B KRW)."
_YTN_NET_INCOME_2021 = (
    "FY2021 saw a positive net income of 519억 원 ($5.19B KRW), a rare profitable year."
)


@pytest.mark.parametrize(
    "text,eok_val,expected",
    [
        (_YTN_REVENUE_2022, 1522.0, "152.2 billion KRW"),
        (_YTN_REVENUE_2024, 1312.0, "131.2 billion KRW"),
        (_YTN_NET_INCOME_2021, 519.0, "51.9 billion KRW"),
    ],
)
def test_eok_pair_10x_gated_auto_corrects_against_supplied_ground_truth(text, eok_val, expected):
    # All three of id=1432's real true-positive figures: the 억원 side
    # matches a SUPPLIED ground-truth value, so the correct billion value is
    # KNOWN (not guessed) and the violation is auto-corrected exactly like
    # billion_10x/trillion_10x.
    violations = check_unit_scale(text, frozenset({eok_val}))
    kinds = [v.kind for v in violations]
    assert "eok_pair_10x" in kinds
    v = next(v for v in violations if v.kind == "eok_pair_10x")
    assert v.expected == expected


@pytest.mark.parametrize(
    "text",
    [_YTN_REVENUE_2022, _YTN_REVENUE_2024, _YTN_NET_INCOME_2021],
)
def test_eok_pair_10x_ungated_is_flag_only(text):
    # Same shape, but the 억원 side does NOT match any supplied ground-truth
    # figure — the pair is still self-contradictory, but we cannot tell
    # which side the model got wrong, so nothing is auto-corrected.
    violations = check_unit_scale(text, frozenset({999_999.0}))
    v = next(v for v in violations if v.kind == "eok_pair_10x")
    assert v.expected is None


def test_eok_pair_10x_correct_conversion_not_flagged():
    text = "1,522억 원 (152.2 billion KRW) was the peak revenue."
    violations = check_unit_scale(text, frozenset({1522.0}))
    assert "eok_pair_10x" not in {v.kind for v in violations}


def test_eok_pair_10x_rounded_correct_conversion_not_flagged():
    # Rounding never trips this check — only an exact power-of-ten
    # discrepancy does.
    text = "1,522억 원 (약 152 billion KRW) was the peak revenue."
    violations = check_unit_scale(text, frozenset({1522.0}))
    assert "eok_pair_10x" not in {v.kind for v in violations}


def test_eok_pair_10x_non_power_of_ten_mismatch_not_flagged():
    # A wrong-but-not-10x-related figure proves the power-of-ten restriction
    # holds — this check does not guess at an arbitrary mismatch.
    text = "1,522억 원 (300 billion KRW) was the peak revenue."
    violations = check_unit_scale(text, frozenset({1522.0}))
    assert "eok_pair_10x" not in {v.kind for v in violations}


def test_eok_pair_10x_detects_space_and_bare_forms():
    # The reason for the new regex: `_EOK_RE` requires the tight "억원"
    # token and matches neither "억 원" (space) nor bare "억" (no 원 at all).
    spaced = check_unit_scale("1,522억 원 ($15.22B KRW)", frozenset({1522.0}))
    assert "eok_pair_10x" in {v.kind for v in spaced}
    bare = check_unit_scale("1,522억 ($15.22B KRW)", frozenset({1522.0}))
    assert "eok_pair_10x" in {v.kind for v in bare}


def test_eok_pair_10x_enforce_corrects_the_span_when_gated():
    corrected, violations = enforce_unit_scale(
        _YTN_REVENUE_2022, "revenue=1522억원", language="en", agent_name="Fundamentals Analyst",
    )
    assert any(v.startswith("eok_pair_10x") for v in violations)
    assert "152.2 billion KRW" in corrected
    assert '[unit corrected from "15.22B KRW"]' in corrected


def test_eok_pair_10x_enforce_leaves_text_in_place_when_ungated():
    corrected, violations = enforce_unit_scale(
        _YTN_REVENUE_2022, "no matching eok figure here", language="en",
    )
    assert any(v.startswith("eok_pair_10x") for v in violations)
    assert "$15.22B KRW" in corrected  # left in place, not rewritten
    assert "own billion-KRW conversion" in corrected  # the flag-only advisory note


def test_floor_boundary_above_floor_sign_agreeing_corrected():
    # 11억원 -> true billion value 1.1 >= _BILLION_CORRECT_FLOOR (1.0):
    # corrected on the bare mantissa match alone, no corroboration needed.
    eok = frozenset({11.0})
    violations = check_unit_scale("11 billion KRW", eok)
    assert len(violations) == 1
    assert violations[0].kind == "billion_10x"
    assert violations[0].expected == "1.1 billion KRW"


def test_floor_boundary_below_floor_sign_agreeing_now_corrected():
    # 9억원 -> true billion value 0.9 < _BILLION_CORRECT_FLOOR: this is the
    # EXACT case the old `e / 10 >= 1.0` guard refused outright. Sign
    # agreement (both positive) now supplies the corroboration the floor
    # used to demand unconditionally — key regression-fix pin.
    eok = frozenset({9.0})
    violations = check_unit_scale("9 billion KRW", eok)
    assert len(violations) == 1
    assert violations[0].kind == "billion_10x"
    assert violations[0].expected == "0.9 billion KRW"


def test_floor_boundary_below_floor_sign_disagreeing_stays_silent():
    # Below the floor, WITHOUT sign agreement: the module stays silent
    # rather than guessing — pins the corroboration gate itself.
    eok = frozenset({9.0})
    assert check_unit_scale("-9 billion KRW", eok) == []


def test_floor_boundary_above_floor_sign_disagreeing_still_corrected():
    # Above the floor, sign agreement is NOT required — a report stating a
    # figure with the opposite sign convention from a positive ground truth
    # is a legitimate sign-convention difference (see
    # tests/agents/test_numeric_guard.py's "-785.2 billion KRW" pin), not
    # evidence of anything, and correction must still proceed.
    eok = frozenset({11.0})
    violations = check_unit_scale("-11 billion KRW", eok)
    assert len(violations) == 1
    assert violations[0].kind == "billion_10x"
    assert violations[0].expected == "-1.1 billion KRW"


def test_correct_small_positive_billion_not_flagged():
    eok = frozenset({5.0})
    assert check_unit_scale("0.5 billion KRW", eok) == []


def test_correct_small_positive_billion_not_flagged_larger():
    eok = frozenset({900.0})
    assert check_unit_scale("90.0 billion KRW", eok) == []


def test_id1168_mixed_signs_in_one_report_only_the_wrong_one_flagged():
    # One correct negative, one incorrect negative (flagged), one correct
    # positive, all in the same report — exactly one violation.
    eok = frozenset({-9.0, -3.0, 900.0})
    report = (
        "Loss A: -0.9 billion KRW. Loss B: -3 billion KRW. "
        "Revenue: 90.0 billion KRW."
    )
    violations, scan = scan_unit_scale(report, eok)
    assert len(violations) == 1
    assert violations[0].kind == "billion_10x"
    assert violations[0].expected == "-0.3 billion KRW"
    assert scan.candidates_examined == 3
    assert scan.corrected == 1
    assert scan.unverified == 0


# --- 2026-09-03 currency-composition net (live decision id=1207, 270520.KS/앱튼,
# KOSDAQ bio). The Fundamentals Analyst report was, cover to cover, a report for
# a DIFFERENT company on a DIFFERENT exchange — Appen Ltd. (APPN), NASDAQ (ex-
# ASX): every figure USD-denominated, not one 원/억원/₩ figure, while the
# exchange handed the model a full 억원 block it made zero use of.
# `identity_gate` Detector C is the primary net; this is the numeric-layer net,
# opt-in via `currency_check` (Fundamentals Analyst only). ---

_ID_1207_REPORT = (
    "Appen Limited (NASDAQ: APPN) is a data-annotation vendor headquartered in "
    "Sydney that moved its primary listing from the ASX to the NASDAQ. Full-year "
    "revenue was $273.0 million, down from $389.4 million a year earlier, and "
    "management guided to $228.0 million for the year ahead. Gross profit was "
    "$95.1 million against $148.7 million previously, and the underlying EBITDA "
    "loss widened to $45.5 million from $8.5 million. Cash and equivalents were "
    "$52.0 million at year end alongside an undrawn facility of $17.0 million. "
    "The company raised $47.0 million in a placement priced at $1.85 per share. "
    "Operating cash outflow was $28.0 million, capitalised development spend was "
    "$19.0 million, and lease liabilities were $12.0 million. The average analyst "
    "price target is $1.20 versus a last close of $1.05. The largest institutional "
    "holder owns a $9.0 million position and disclosed short interest is about "
    "$6.0 million of the float. We would not add exposure here given persistent "
    "cash burn and the absence of a near-term catalyst for the shares to re-rate."
)  # exactly 18 USD-anchored figures, zero KRW figures

_GT_39 = frozenset(float(1000 + i) for i in range(39))
_KR_BLOCK_39 = "\n".join(f"line{i}={1000 + i}억원" for i in range(39))

# padding used to clear the 600-char premise gate in the smaller fixtures
_PAD = (
    " Additional discussion padding repeated so the report is realistically long "
    "enough to have declared its reporting currency somewhere in the body, well "
    "past the six-hundred-character premise gate for the currency-composition "
    "check, so each fixture exercises the intended branch and not a length gate."
)


def test_currency_inverted_flags_the_id_1207_shape():
    for language, marker in (("en", "CURRENCY-COMPOSITION ALERT"), ("ko", "통화 구성 이상")):
        corrected, descriptions = enforce_unit_scale(
            _ID_1207_REPORT, _KR_BLOCK_39, language=language,
            agent_name="Fundamentals Analyst", currency_check=True,
        )
        assert len(descriptions) == 1
        assert descriptions[0].startswith("currency_inverted:")
        assert "-> None" in descriptions[0]
        assert marker in corrected[:80]
        assert corrected.endswith(_ID_1207_REPORT)  # body untouched, banner prepended
        assert "$273.0 million" in corrected


def test_scan_carries_currency_composition():
    violations, scan = scan_unit_scale(
        _ID_1207_REPORT, _GT_39, currency_check=True, node="Fundamentals Analyst",
    )
    assert len(violations) == 1
    assert violations[0].kind == "currency_inverted"
    assert violations[0].expected is None
    assert scan.candidates_examined == 0
    assert scan.usd_figures == 18
    assert scan.krw_figures == 0
    assert scan.currency_inverted is True
    assert scan.suspicious is True
    d = scan.as_dict()
    assert d["usd_figures"] == 18
    assert d["krw_figures"] == 0
    assert d["currency_inverted"] == 1


def test_currency_inverted_is_node_prefixed_and_reaches_the_violation_list():
    from app.agents.numeric_guard import enforce_numeric_guards

    guarded, descriptions = enforce_numeric_guards(
        _ID_1207_REPORT, node="Fundamentals Analyst", kr_block=_KR_BLOCK_39,
        currency_check=True, language="en",
    )
    assert descriptions[0].startswith("Fundamentals Analyst: currency_inverted:")
    assert "CURRENCY-COMPOSITION ALERT" in guarded


def test_tier_b_catches_a_single_token_evasion():
    eok = frozenset(float(1000 + i) for i in range(6))
    usd10 = " ".join(f"item {i} was ${10 + i}.0 million." for i in range(10))
    fires = (
        "This report concerns a KRX-listed company but quotes almost everything "
        "in dollars. " + usd10 + " The only local figure mentioned in passing is "
        "₩12,300 per share." + _PAD
    )
    v1, _ = scan_unit_scale(fires, eok, currency_check=True)
    assert [v.kind for v in v1] == ["currency_inverted"]

    usd8 = " ".join(f"item {i} was ${10 + i}.0 million." for i in range(8))
    no_fire = (
        "KRX-listed company, mixed-currency discussion. " + usd8 + " Local anchors: "
        "₩12,300 per share and revenue near ₩5,000 last year." + _PAD
    )
    v2, _ = scan_unit_scale(no_fire, eok, currency_check=True)
    assert v2 == []


def test_no_flag_when_the_report_cites_a_us_peer_market_cap():
    eok = frozenset(float(1000 + i) for i in range(8))
    usd8 = " ".join(f"the US peer traded near ${20 + i} billion of equity value" for i in range(8))
    krw9 = " ".join(f"our company reported ₩{i + 1},000 in segment {i}" for i in range(9))
    report = "KRX name with a US peer comparison. " + usd8 + ". " + krw9 + "." + _PAD
    violations, scan = scan_unit_scale(report, eok, currency_check=True)
    assert violations == []
    assert scan.usd_figures == 8
    assert scan.krw_figures == 9


def test_no_flag_for_a_krw_exporter_quoting_usd_export_revenue():
    eok = frozenset(float(1000 + i) for i in range(6))
    usd7 = " ".join(f"export revenue in region {i} was ${5 + i}.0 million" for i in range(7))
    krw19 = " ".join(f"domestic segment {i} posted ₩{i + 1},000" for i in range(19))
    report = "KRX exporter reporting in won. " + krw19 + ". " + usd7 + "." + _PAD
    violations, _ = scan_unit_scale(report, eok, currency_check=True)
    assert violations == []


def test_no_flag_for_a_holdco_with_a_us_subsidiary():
    eok = frozenset(float(1000 + i) for i in range(6))
    usd6 = " ".join(f"the subsidiary booked ${5 + i}.0 million in area {i}" for i in range(6))
    report = (
        "Holdco is KRX-listed; its US subsidiary reports in dollars. " + usd6
        + " Parent-level revenue was ₩4,500 and the group posted 320억원 "
        "operating profit." + _PAD
    )
    violations, _ = scan_unit_scale(report, eok, currency_check=True)
    assert violations == []


def test_no_flag_for_an_etf_with_thin_ground_truth():
    eok = frozenset({50_000.0})
    usd5 = " ".join(f"the fund holds ${10 + i} billion of AAPL-like names" for i in range(5))
    report = "A KRX ETF tracking a US index. " + usd5 + "." + _PAD + _PAD
    violations, scan = scan_unit_scale(report, eok, currency_check=True)
    assert violations == []
    assert scan.usd_figures == 5  # counts still recorded
    assert scan.currency_inverted is False


def test_no_flag_on_a_short_degraded_stub():
    report = "Short degraded stub. $1 million $2 million $3 million. Analysis unavailable."
    assert len(report) < 600
    violations, scan = scan_unit_scale(report, _GT_39, currency_check=True)
    assert violations == []
    assert scan.usd_figures == 3


def test_currency_banner_is_idempotent():
    once, d1 = enforce_unit_scale(
        _ID_1207_REPORT, _KR_BLOCK_39, language="ko", currency_check=True,
    )
    assert len(d1) == 1
    twice, d2 = enforce_unit_scale(once, _KR_BLOCK_39, language="ko", currency_check=True)
    assert d2 == []
    assert twice == once
    assert twice.count("통화 구성 이상") == 1


def test_currency_banner_does_not_disarm_identity_detector_c():
    from app.agents.identity_gate import _KRW_AMOUNT_RE, _foreign_listing_offender
    from app.agents.unit_scale_guard import _CURRENCY_INVERTED_BANNER

    banner_ko = _CURRENCY_INVERTED_BANNER["ko"]
    banner_en = _CURRENCY_INVERTED_BANNER["en"]
    assert _KRW_AMOUNT_RE.search(banner_ko) is None
    assert _KRW_AMOUNT_RE.search(banner_en) is None

    corrected, _ = enforce_unit_scale(
        _ID_1207_REPORT, _KR_BLOCK_39, language="en", currency_check=True,
    )
    offender = _foreign_listing_offender(corrected, "270520.KS", frozenset({"앱튼"}))
    assert offender is not None
    assert type(offender).__name__ == "ForeignListingName"


# ---------------------------------------------------------------------------
# supplied_kr_block gate on the id=1158 "SUSPECTED BLIND SPOT" warning.
# The heuristic's premise is block-readership; only the Fundamentals Analyst
# is ever handed the 억원 block. Measured 88/88 false alarms 2026-08-29→09-04
# from nodes that never received it. The gate is log-only — see the
# behavioral-invariance test below.
# ---------------------------------------------------------------------------

# A blind-spot input: candidates present, ground truth present, nothing
# matches at 1x / 10x / the power-of-10 ladder → corrected == 0 and every
# candidate stays unverified.
_BLIND_SPOT_REPORT = "revenue ₩625,389 billion and profit ₩48,337 billion"
_BLIND_SPOT_KR_BLOCK = "a=111,111억원 b=222,222억원"


def test_blind_spot_warning_emitted_when_supplied_kr_block_true(caplog):
    with caplog.at_level("WARNING", logger="app.agents.unit_scale_guard"):
        enforce_unit_scale(
            _BLIND_SPOT_REPORT, _BLIND_SPOT_KR_BLOCK,
            agent_name="Fundamentals Analyst", supplied_kr_block=True,
        )
    assert "SUSPECTED BLIND SPOT" in caplog.text


def test_blind_spot_warning_suppressed_when_flag_omitted(caplog):
    with caplog.at_level("WARNING", logger="app.agents.unit_scale_guard"):
        enforce_unit_scale(
            _BLIND_SPOT_REPORT, _BLIND_SPOT_KR_BLOCK,
            agent_name="research_debate[r1 Bull]",
        )
    assert "SUSPECTED BLIND SPOT" not in caplog.text


def test_blind_spot_gate_is_log_only_behaviorally_invariant():
    """The flag changes exactly one thing — whether that one warning logs.
    Returned text, returned violation descriptions and the captured
    UnitScaleScan must be byte-for-byte identical either way."""
    scan_off: list = []
    scan_on: list = []
    text_off, desc_off = enforce_unit_scale(
        _BLIND_SPOT_REPORT, _BLIND_SPOT_KR_BLOCK,
        agent_name="x", supplied_kr_block=False, scan_out=scan_off,
    )
    text_on, desc_on = enforce_unit_scale(
        _BLIND_SPOT_REPORT, _BLIND_SPOT_KR_BLOCK,
        agent_name="x", supplied_kr_block=True, scan_out=scan_on,
    )
    assert text_off == text_on
    assert desc_off == desc_on
    assert scan_off[0] == scan_on[0]


def test_blind_spot_gate_threads_through_enforce_numeric_guards(caplog):
    from app.agents.numeric_guard import enforce_numeric_guards

    with caplog.at_level("WARNING", logger="app.agents.unit_scale_guard"):
        enforce_numeric_guards(
            _BLIND_SPOT_REPORT, node="Fundamentals Analyst",
            kr_block=_BLIND_SPOT_KR_BLOCK, supplied_kr_block=True,
        )
    assert "SUSPECTED BLIND SPOT" in caplog.text

    caplog.clear()
    with caplog.at_level("WARNING", logger="app.agents.unit_scale_guard"):
        enforce_numeric_guards(
            _BLIND_SPOT_REPORT, node="research_debate[r1 Bull]",
            kr_block=_BLIND_SPOT_KR_BLOCK,
        )
    assert "SUSPECTED BLIND SPOT" not in caplog.text


# --- 2026-09-07 audit #6: the FINANCIAL-DATA RECENCY note (kis_provider)
# must add no 억원 ground-truth value and trip no unit-scale violation.
def test_staleness_note_adds_no_eok_ground_truth():
    from datetime import date

    from app.dataflows.kis_provider import _staleness_note, kr_financials_recency

    note = _staleness_note(
        kr_financials_recency("202412", "202412 (FY, 12 months)", date(2026, 9, 7)),
        date(2026, 9, 7),
    )
    assert note
    assert extract_kr_eok_values(note) == frozenset()
    violations, scan = scan_unit_scale(note, frozenset(), node="Fundamentals Analyst")
    assert violations == []
    assert scan.candidates_examined == 0


# --- 2026-09-07 audit #3: news-headline 억 amounts as SOFT ground truth ---
_SEONAM_NEWS = "[20260904] 서남 수주공시 - 초전도 선재 공급 계약 6.6억원 (매출액대비 19.56 %)\n[20260904] 서남, 6.62억원 규모 공급계약 체결\n[20260904] 서남, 6.6억 규모 공급계약"


def test_extract_kr_headline_eok_values_reads_eok_and_jo():
    vals = extract_kr_headline_eok_values(_SEONAM_NEWS + "\n삼성전자, 12조원 투자 발표")
    assert 6.6 in vals and 6.62 in vals
    assert 120_000.0 in vals  # 12조 -> 120,000억


def test_extract_kr_headline_eok_values_empty_without_amounts():
    assert extract_kr_headline_eok_values("서남, 계약 소식에 강세") == frozenset()


def test_news_eok_10x_flagged_not_corrected():
    # the id=1237 shape: report says ₩6.6 billion, headline said 6.6억원
    report = "The desk notes a ₩6.6 billion supply contract announced Sept 4."
    soft = extract_kr_headline_eok_values(_SEONAM_NEWS)
    violations, scan = scan_unit_scale(report, frozenset(), soft_eok_values=soft, node="News + Macro Analyst")
    assert len(violations) == 1
    v = violations[0]
    assert v.kind == "news_eok_10x" and v.expected is None
    assert scan.news_eok_flagged == 1 and scan.soft_ground_truth_values >= 2
    # enforce leaves the text intact, appends a note, no banner
    guarded, desc = enforce_unit_scale(report, "", soft_eok_block=_SEONAM_NEWS, agent_name="News + Macro Analyst")
    assert "₩6.6 billion supply contract" in guarded
    assert "[unit corrected from" not in guarded
    assert "UNIT-SCALE CORRECTION" not in guarded
    assert "headline this cycle wrote in the 억 unit" in guarded


def test_news_eok_correct_conversion_is_accepted_not_flagged():
    report = "The desk notes a ₩0.66 billion supply contract (roughly ₩660 million)."
    soft = extract_kr_headline_eok_values(_SEONAM_NEWS)
    violations, scan = scan_unit_scale(report, frozenset(), soft_eok_values=soft, node="x")
    assert violations == []
    assert scan.news_eok_flagged == 0
    assert scan.unverified == 0


def test_hard_ground_truth_wins_over_soft():
    # a figure that IS a correct conversion of an exchange 억원 value is
    # accepted even though its mantissa also equals a headline 억 amount
    report = "Revenue reached 6.6 billion KRW last year."
    hard = frozenset({66.0})   # 66억원 -> 6.6 billion KRW, correct
    soft = frozenset({6.6})    # a headline that wrote "6.6억"
    violations, scan = scan_unit_scale(report, hard, soft_eok_values=soft, node="x")
    assert violations == []
    assert scan.news_eok_flagged == 0


def test_soft_values_never_reach_tier1_autocorrect():
    report = "A ₩6.6 billion deal."
    _, scan = scan_unit_scale(report, frozenset(), soft_eok_values=frozenset({6.6}), node="x")
    assert scan.corrected == 0


def test_news_eok_10x_upgrades_a_magnitude_suspect_on_the_same_span():
    # 2026-09-09, id=1358 / 047040.KS: a 4,134.3억원 redevelopment contract
    # (three outlets: 4,134.3억 / 4134억 / 4,134.27억) was written
    # "4,134.3 billion KRW". It collided with an unrelated exchange 억원 value
    # on magnitude_suspect's power-of-10 ladder AND matched the headline
    # amount digit for digit. The vague magnitude_suspect used to win the
    # span (it ran first, the soft pass skipped an already-flagged span) —
    # now the specific news_eok_10x supersedes it, and the RM's own corrected
    # value ("413.4 billion") in the same text reads as accepted, not flagged.
    report = "Daewoo E&C secured a 4,134.3 billion KRW contract for the Sanggye 2-district project."
    hard = frozenset({413_430.0})                    # collides at 10^2, not an exact 10x
    soft = frozenset({4134.0, 4134.27, 4134.3})      # the three headline renderings
    violations, scan = scan_unit_scale(report, hard, soft_eok_values=soft, node="News + Macro Analyst")
    assert [v.kind for v in violations] == ["news_eok_10x"]
    assert violations[0].written == "4,134.3 billion KRW" and violations[0].expected is None
    assert scan.news_eok_flagged == 1
    assert scan.flagged == 1  # not double-counted
    # without the headline ground truth it stays the vague label
    v2, _ = scan_unit_scale(report, hard, soft_eok_values=frozenset(), node="x")
    assert [v.kind for v in v2] == ["magnitude_suspect"]


def test_news_eok_headline_match_clears_a_coincidental_magnitude_suspect():
    # The RM's own corrected value ("413.4 billion KRW") is the correct
    # conversion of the 4,134.3억 headline; even if it also collides with an
    # unrelated exchange figure on the power-of-10 ladder, the headline match
    # is the better explanation and it reads as accepted, not flagged.
    report = "The Sanggye 2-district contract is worth 413.4 billion KRW."
    hard = frozenset({413_430.0})               # 413.4 collides at 10^3
    soft = frozenset({4134.3})
    violations, scan = scan_unit_scale(report, hard, soft_eok_values=soft, node="x")
    assert violations == []
    assert scan.news_eok_flagged == 0


def test_news_eok_upgrade_swaps_the_banner_note():
    report = "Daewoo E&C secured a 4,134.3 billion KRW contract."
    kr_block = "[Income statement]\nrevenue=413,430억원\n"   # collides at 10^2, not an exact 10x
    guarded, _ = enforce_unit_scale(
        report, kr_block,
        soft_eok_block="[20260909] 대우건설, 상계2구역 4,134.3억원 규모 시공권 확보",
        agent_name="News + Macro Analyst",
    )
    assert "headline this cycle wrote in the 억 unit" in guarded       # news_eok note
    assert "near power-of-10 off a supplied 억원" not in guarded        # magnitude_suspect note gone


def test_default_soft_empty_is_byte_identical():
    report = "A ₩6.6 billion deal and revenue of 89,859 billion KRW."
    a = scan_unit_scale(report, frozenset({898_590.0}))
    b = scan_unit_scale(report, frozenset({898_590.0}), soft_eok_values=frozenset())
    assert a[0] == b[0]
    assert a[1].as_dict() == b[1].as_dict()


@pytest.mark.parametrize("report,hard,soft", [
    # a genuine blind-spot shape (id=1158): large KRW figure, ground truth, nothing matches
    ("Revenue of 89,859 billion KRW and 12,345 billion KRW.", frozenset({777_777.0}), frozenset({89_859.0})),
    # a clean report
    ("Revenue reached 6.6 billion KRW.", frozenset({66.0}), frozenset({6.6})),
    # a news carry-through
    ("A 4,134.3 billion KRW contract.", frozenset({999_999.0}), frozenset({4134.3})),
])
def test_soft_ground_truth_can_only_lower_unverified_never_raise_suspicion(report, hard, soft):
    # The #211 lesson as a test: supplying the SOFT (news-headline) ground
    # truth must never turn a report SUSPICIOUS that was not, and must never
    # raise `unverified` — the blind-spot heuristic does not read the soft
    # fields at all, and the soft pass can only add `accepted`/`news_eok_flagged`
    # (so it can only clear suspicion / lower unverified, never the reverse).
    _, without = scan_unit_scale(report, hard, soft_eok_values=frozenset())
    _, with_soft = scan_unit_scale(report, hard, soft_eok_values=soft)
    assert without.suspicious or not with_soft.suspicious   # soft never ADDS suspicion
    assert with_soft.unverified <= without.unverified


def test_scan_dict_gains_soft_fields_without_disturbing_the_rest():
    _, scan = scan_unit_scale("nothing here", frozenset())
    d = scan.as_dict()
    assert d["soft_ground_truth_values"] == 0 and d["news_eok_flagged"] == 0
    for k in ("candidates_examined", "ground_truth_values", "corrected", "flagged", "unverified"):
        assert d[k] == 0


def test_news_eok_note_does_not_trip_identity_gate_krw_amount_re():
    from app.agents.identity_gate import _KRW_AMOUNT_RE
    from app.agents.unit_scale_guard import _NEWS_EOK_NOTE

    for text in _NEWS_EOK_NOTE.values():
        assert _KRW_AMOUNT_RE.search(text) is None, text


def test_magnitude_suspect_id1433_range_hyphen_hypothetical_not_flagged():
    # 2026-09-15 60-day scan false positive: id=1433, 004310.KS. "perhaps
    # 50-100 billion KRW if any assets existed" — a hypothetical valuation,
    # not a data claim, and the "-" in "50-100" is a range hyphen that
    # `_SIGN` misparsed as this candidate's own minus sign (a real bug: the
    # flagged figure was "-100 billion KRW", sign and all). eok=10000 makes
    # the power-of-10 ladder fire (av=100, k=1 -> 1000 == e/10) so this pins
    # the exemption, not merely "nothing collided".
    eok = frozenset({10_000.0})
    report = "a return to a rational valuation, perhaps 50-100 billion KRW if any assets existed"
    assert check_unit_scale(report, eok) == []


def test_magnitude_suspect_id1441_annualized_arithmetic_not_flagged():
    # 2026-09-15 60-day scan false positive: id=1441, 278650.KS. The model's
    # own derived arithmetic ("Annualizing this H1 figure suggests...") is
    # not a misstated data figure. eok=900 makes the ladder fire (av=9, k=1
    # -> 90 == e/10).
    eok = frozenset({900.0})
    report = "Annualizing this H1 figure suggests an operating loss of nearly 9 billion KRW."
    assert check_unit_scale(report, eok) == []


def test_magnitude_suspect_scenario_hypothetical_without_a_range_not_flagged():
    # A hypothetical/scenario figure with no range hyphen at all — pins the
    # `_SCENARIO_CTX_RE` exemption independent of the range-hyphen rule.
    # eok=3000 makes the ladder fire (av=30, k=1 -> 300 == e/10).
    eok = frozenset({3_000.0})
    report = "could reach 30 billion KRW in a recovery scenario"
    assert check_unit_scale(report, eok) == []


def test_magnitude_suspect_id1436_true_positive_still_flagged():
    # The one confirmed true positive from the 2026-09-15 60-day scan
    # sample: id=1436, 024060.KS, "the company generated negligible revenues
    # (~₩11.7 billion)" against a real FY2024 revenue of ₩116.85B. No range
    # hyphen, no hedge/scenario vocabulary nearby — this MUST survive the new
    # exemptions, or they have been widened into uselessness. eok=1170 makes
    # the ladder fire (av=11.7, k=1 -> 117.0 == e/10).
    eok = frozenset({1_170.0})
    report = "the company generated negligible revenues (~₩11.7 billion)"
    violations = check_unit_scale(report, eok)
    assert [v.kind for v in violations] == ["magnitude_suspect"]
    assert violations[0].expected is None


def test_magnitude_suspect_catches_10x_under_shifted_program_trading_figure():
    # 2026-09-11, id=1386 (006360.KS): ground truth was "+156.5억원"
    # (e=156.5). One debater correctly wrote "+15.65 billion KRW" (av=15.65
    # = e/10 — accepted, not flagged). A second debater then "corrected"
    # this to "+1.565 billion KRW" (av=1.565 = e/100), calling the correct
    # figure a "tenfold exaggeration" — a real 10x-UNDER-shifted write that,
    # before the k=1 fix, matched neither Tier 1's exact av≈e check nor the
    # old magnitude_suspect ladder (k in {2, 3, -2, -3}).
    eok = frozenset({156.5})
    correct = check_unit_scale("Program trading shows +15.65 billion KRW in buys.", eok)
    assert correct == []

    wrong = check_unit_scale("Program trading shows +1.565 billion KRW in buys.", eok)
    assert [v.kind for v in wrong] == ["magnitude_suspect"]
    assert wrong[0].expected is None


# --- 2026-09-11 addendum (id=1396, 003350.KS): a companion USD figure derived
# from a pre-correction (10x-wrong) KRW magnitude, sitting in the SAME
# sentence as a billion_10x/trillion_10x correction, is stale and must be
# redacted alongside the numeric fix — otherwise a corrected, authoritative-
# looking KRW figure sits right next to a USD figure that is still ~10x off. ---

def test_id1396_stale_companion_usd_is_redacted_alongside_the_correction():
    from app.agents.unit_scale_guard import _DERIVED_USD_STALE_NOTE

    eok = frozenset({3_732.0})  # true value: 373.2 billion KRW
    report = (
        "Revenue guidance was raised to 3,732 billion KRW (~$28 million USD) "
        "for the year."
    )
    corrected, descriptions = enforce_unit_scale(
        report, "x=3732억원", language="en", agent_name="Fundamentals Analyst",
    )
    assert "373.2 billion KRW" in corrected
    assert _DERIVED_USD_STALE_NOTE["en"] in corrected
    assert "$28" not in corrected
    assert any(d.startswith("derived_usd_stale:") for d in descriptions)

    _, scan = scan_unit_scale(report, eok)
    assert scan.derived_usd_stale == 1


def test_id1396_plausible_companion_usd_is_not_redacted():
    # $270 million against the CORRECTED 373.2 billion KRW implies a ~1,382
    # KRW/USD rate — squarely inside the real range, so this is a genuinely
    # unrelated, correct USD figure and must be left alone.
    eok = frozenset({3_732.0})
    report = (
        "Revenue guidance was raised to 3,732 billion KRW (~$270 million USD) "
        "for the year."
    )
    corrected, descriptions = enforce_unit_scale(
        report, "x=3732억원", language="en", agent_name="Fundamentals Analyst",
    )
    assert "373.2 billion KRW" in corrected
    assert "$270 million" in corrected
    assert not any(d.startswith("derived_usd_stale:") for d in descriptions)

    _, scan = scan_unit_scale(report, eok)
    assert scan.derived_usd_stale == 0


def test_id1396_usd_figure_in_a_different_sentence_is_left_alone():
    eok = frozenset({3_732.0})
    report = (
        "Revenue guidance was raised to 3,732 billion KRW for the year. "
        "Separately, the company disclosed a $28 million USD line of credit "
        "unrelated to revenue, filed with regulators last month for reference."
    )
    corrected, descriptions = enforce_unit_scale(
        report, "x=3732억원", language="en", agent_name="Fundamentals Analyst",
    )
    assert "373.2 billion KRW" in corrected
    assert "$28 million USD line of credit" in corrected
    assert not any(d.startswith("derived_usd_stale:") for d in descriptions)


def test_id1396_no_derived_usd_stale_when_no_billion_or_trillion_10x_at_all():
    # The KRW figure is already correct — no Tier-1 correction fires — so the
    # companion USD check, which only runs off a confirmed correction's
    # `expected_won`, must never fire either, regardless of any USD figures
    # present nearby.
    eok = frozenset({3_732.0})  # true value: 373.2 billion KRW
    report = (
        "Revenue guidance was raised to 373.2 billion KRW (~$28 million USD) "
        "for the year."
    )
    violations, scan = scan_unit_scale(report, eok)
    assert violations == []
    assert scan.derived_usd_stale == 0


def test_id1396_derived_usd_stale_note_has_no_currency_or_digit_tokens():
    # Same HARD CONSTRAINT as _CURRENCY_INVERTED_BANNER / _NEWS_EOK_NOTE: no
    # `$`, no digit, no won/KRW token — either would perturb identity_gate
    # Detector C's USD-figure/KRW-figure count over the whole report.
    from app.agents.identity_gate import _KRW_AMOUNT_RE
    from app.agents.unit_scale_guard import _DERIVED_USD_STALE_NOTE

    for text in _DERIVED_USD_STALE_NOTE.values():
        assert _KRW_AMOUNT_RE.search(text) is None, text
        assert "$" not in text
        assert not any(ch.isdigit() for ch in text)
        for token in ("KRW", "원", "won", "₩"):
            assert token not in text
