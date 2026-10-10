"""app.agents.ratio_label_guard — no network, no LLM (A-stage).

Real incident this guards against (2026-09-01 domestic AI audit, item #4,
first real-account KRX cycle, 066970.KS/엘앤에프): the Fundamentals Analyst
report described KIS's 부채비율 (total liabilities ÷ total EQUITY, can exceed
100%) as an assets-denominated ratio — "372.75% (Liabilities/Assets)" — and
the Portfolio Manager repeated it downstream as "a 372.75% debt-to-assets
ratio" (decision id=1121). A liabilities/assets ratio cannot exceed 100% by
definition, so this module needs no external ground truth: an above-100%
reading carrying assets-denominated wording is self-refuting on its face.
"""

from __future__ import annotations

import dataclasses

import pytest

from app.agents.ratio_label_guard import (
    RatioLabelViolation,
    check_ratio_labels,
    enforce_ratio_labels,
)


def test_check_ratio_labels_flags_the_066970_incident_shape():
    report = (
        "Debt Ratio remains elevated at 372.75% (Liabilities/Assets), though "
        "this metric is distorted by the small equity base."
    )
    violations = check_ratio_labels(report)
    assert len(violations) == 1
    v = violations[0]
    assert v.kind == "debt_ratio_denominator"
    assert "372.75" in v.written
    assert "Liabilities/Assets" in v.written
    assert "total EQUITY" in v.expected


def test_check_ratio_labels_flags_debt_to_assets_wording():
    report = "The company carries a 372.75% debt-to-assets ratio."
    violations = check_ratio_labels(report)
    assert len(violations) == 1


def test_check_ratio_labels_flags_denominator_first_word_order():
    report = "Its debt-to-assets ratio of 372.75% is a red flag."
    violations = check_ratio_labels(report)
    assert len(violations) == 1


def test_check_ratio_labels_does_not_flag_a_legitimate_sub_100_assets_ratio():
    # Critical false-positive guard: a genuinely assets-denominated ratio
    # below 100% (e.g. borrowing_dependency-shaped prose) is legitimate and
    # must never be flagged just because the wording says "liabilities/assets".
    report = "Liabilities/assets stands at 78.8%, within the sector norm."
    assert check_ratio_labels(report) == []


def test_check_ratio_labels_does_not_flag_a_correctly_labeled_equity_ratio():
    report = "부채비율은 372.75% (총부채 ÷ 자기자본)로 높다."
    assert check_ratio_labels(report) == []


def test_check_ratio_labels_does_not_flag_borrowing_dependency_shaped_prose():
    report = "Interest-bearing debt/total assets is 44.4%, moderate for the sector."
    assert check_ratio_labels(report) == []


def test_check_ratio_labels_dedups_a_span_matched_by_both_patterns():
    # Shaped so the pct-then-denom regex matches starting at the first "372.75%"
    # through "debt-to-assets", AND the denom-then-pct regex independently
    # matches "debt-to-assets ratio of 372.75%" (the second occurrence) —
    # the two matches' spans overlap at "debt-to-assets", so this must still
    # yield exactly one violation, not two.
    report = "The 372.75% debt-to-assets ratio of 372.75% is a red flag."
    violations = check_ratio_labels(report)
    assert len(violations) == 1


def test_enforce_ratio_labels_corrects_the_span_and_adds_a_banner():
    report = (
        "Debt Ratio remains elevated at 372.75% (Liabilities/Assets), though "
        "this metric is distorted by the small equity base."
    )
    corrected, violations = enforce_ratio_labels(report, language="en")
    assert len(violations) == 1
    assert "RATIO-LABEL CORRECTION" in corrected
    assert "(Liabilities/Assets)" not in corrected  # raw span gone, not left unreplaced
    assert '[label corrected from "' in corrected
    assert "total EQUITY" in corrected


def test_enforce_ratio_labels_noop_when_clean():
    report = "부채비율은 372.75% (총부채 ÷ 자기자본)로 높다."
    corrected, violations = enforce_ratio_labels(report, language="ko")
    assert corrected == report
    assert violations == []


def test_enforce_ratio_labels_korean_banner():
    report = "The company carries a 372.75% debt-to-assets ratio."
    corrected, violations = enforce_ratio_labels(report, language="ko")
    assert violations
    assert "지표 정의 정정" in corrected


def test_enforce_ratio_labels_multiple_violations_all_corrected():
    report = (
        "Debt Ratio remains elevated at 372.75% (Liabilities/Assets). "
        "Separately, the company carries a 288.4% debt-to-assets ratio for FY2025."
    )
    corrected, violations = enforce_ratio_labels(report, language="en")
    assert len(violations) == 2
    # The banner itself explains the "[label corrected from "..."]" markup
    # with that exact bracketed phrase, so count only in the corrected body
    # (after the banner's leading "\n\n"), not the whole string.
    body = corrected.split("\n\n", 1)[1]
    assert body.count('[label corrected from "') == len(violations)


def test_ratio_label_violation_is_frozen_dataclass():
    v = RatioLabelViolation(
        kind="debt_ratio_denominator", written="x", expected="y", start=0, end=1
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        v.kind = "other"
