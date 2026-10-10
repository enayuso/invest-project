"""app.experiments.backend_ab_report — pure computation over hand-built
record dicts (no filesystem, no DB). Fixture JSON files are only used in the
census/error-file test, matching the report's actual `_load_records` I/O
boundary."""

from __future__ import annotations

import json
from pathlib import Path

from app.experiments.backend_ab_report import ARM_CLAUDE, ARM_QWEN, compute_backend_ab_report, _load_records


def _record(
    ticker: str, trade_date: str, arm: str, repeat: int = 1, *,
    guard_violations_total: int = 0, guard_violations: dict | None = None,
    degraded: bool = False, isolation_violation: bool = False,
    llm_fallback_calls: int = 0, llm_truncated_calls: int = 0,
) -> dict:
    return {
        "ticker": ticker,
        "date": trade_date,
        "arm": arm,
        "repeat": repeat,
        "degraded": degraded,
        "isolation_violation": isolation_violation,
        "guard_violations_total": guard_violations_total,
        "guard_violations": guard_violations or {},
        "llm_fallback_calls": llm_fallback_calls,
        "llm_truncated_calls": llm_truncated_calls,
    }


def _make_pairs(n: int, *, claude_total: int, qwen_total: int, guard_field: str | None = None) -> list[dict]:
    records = []
    for i in range(n):
        ticker = f"T{i:03d}"
        date_str = f"2026-07-{(i % 28) + 1:02d}"
        gv_claude = {guard_field: claude_total} if guard_field else {}
        gv_qwen = {guard_field: qwen_total} if guard_field else {}
        records.append(_record(ticker, date_str, ARM_CLAUDE, guard_violations_total=claude_total, guard_violations=gv_claude))
        records.append(_record(ticker, date_str, ARM_QWEN, guard_violations_total=qwen_total, guard_violations=gv_qwen))
    return records


def test_clear_claude_wins():
    records = _make_pairs(20, claude_total=0, qwen_total=3)
    report = compute_backend_ab_report(records, min_pairs=20, min_field_fires=10)
    assert report.paired == 20
    assert report.verdict == "CLAUDE_WINS"


def test_clear_qwen_wins():
    records = _make_pairs(20, claude_total=3, qwen_total=0)
    report = compute_backend_ab_report(records, min_pairs=20, min_field_fires=10)
    assert report.paired == 20
    assert report.verdict == "QWEN_WINS"


def test_underpowered_via_too_few_pairs():
    records = _make_pairs(5, claude_total=0, qwen_total=3)
    report = compute_backend_ab_report(records, min_pairs=20, min_field_fires=10)
    assert report.paired == 5
    assert report.verdict == "UNDERPOWERED"


def test_underpowered_via_all_zero_violations():
    records = _make_pairs(20, claude_total=0, qwen_total=0)
    report = compute_backend_ab_report(records, min_pairs=20, min_field_fires=10)
    assert report.paired == 20
    assert report.verdict == "UNDERPOWERED"


def test_no_effect_noisy_deltas_centered_on_zero():
    records = []
    for i in range(20):
        ticker = f"T{i:03d}"
        date_str = f"2026-07-{(i % 28) + 1:02d}"
        if i % 2 == 0:
            claude_total, qwen_total = 1, 0
        else:
            claude_total, qwen_total = 0, 1
        records.append(_record(ticker, date_str, ARM_CLAUDE, guard_violations_total=claude_total))
        records.append(_record(ticker, date_str, ARM_QWEN, guard_violations_total=qwen_total))
    report = compute_backend_ab_report(records, min_pairs=20, min_field_fires=10)
    assert report.paired == 20
    assert report.mean_delta == 0.0
    assert report.verdict == "NO_EFFECT"


def test_mixed_claude_better_primary_but_truncated_worse():
    records = []
    for i in range(20):
        ticker = f"T{i:03d}"
        date_str = f"2026-07-{(i % 28) + 1:02d}"
        records.append(
            _record(ticker, date_str, ARM_CLAUDE, guard_violations_total=0, llm_truncated_calls=1)
        )
        records.append(
            _record(ticker, date_str, ARM_QWEN, guard_violations_total=3, llm_truncated_calls=0)
        )
    report = compute_backend_ab_report(records, min_pairs=20, min_field_fires=10)
    assert report.paired == 20
    assert report.mean_delta is not None and report.mean_delta < 0
    assert report.verdict == "MIXED"
    assert "llm_truncated_calls" in report.verdict_detail


def test_degraded_record_excluded_from_pairing_but_counted_in_census():
    records = _make_pairs(20, claude_total=0, qwen_total=3)
    # One extra (ticker, date, repeat) group where the claude arm degraded —
    # must show up in the claude census as completed+degraded, but must NOT
    # be counted among the paired groups.
    records.append(_record("EXTRA", "2026-08-01", ARM_CLAUDE, guard_violations_total=0, degraded=True))
    records.append(_record("EXTRA", "2026-08-01", ARM_QWEN, guard_violations_total=1))

    report = compute_backend_ab_report(records, min_pairs=20, min_field_fires=10)
    assert report.paired == 20  # the degraded pair did not get counted
    assert report.dropped_degraded == 1
    assert report.claude_census.completed == 21
    assert report.claude_census.degraded == 1
    assert report.qwen_census.completed == 21
    assert report.qwen_census.degraded == 0


def test_underpowered_per_field_row_prints_marker_not_number():
    records = _make_pairs(20, claude_total=0, qwen_total=3, guard_field="common_violations")
    # Add a rare field that only fires 5 times combined (< default min_field_fires=10).
    for i in range(3):
        records[2 * i]["guard_violations"]["rare_violations"] = 1  # claude side, 3 fires
    for i in range(3, 5):
        records[2 * i + 1]["guard_violations"]["rare_violations"] = 1  # qwen side, 2 more fires

    report = compute_backend_ab_report(records, min_pairs=20, min_field_fires=10)
    rare = next(s for s in report.field_stats if s.field_name == "rare_violations")
    assert rare.underpowered is True
    assert rare.n_fires == 5
    assert rare.mean_delta is None
    assert rare.p_value is None

    common = next(s for s in report.field_stats if s.field_name == "common_violations")
    assert common.underpowered is False
    assert common.p_value is not None


def test_load_records_skips_error_json_but_counts_it(tmp_path: Path):
    (tmp_path / "T1_2026-07-01_claude_r1.json").write_text(
        json.dumps(_record("T1", "2026-07-01", ARM_CLAUDE)), encoding="utf-8",
    )
    (tmp_path / "T1_2026-07-01_qwen_r1.json").write_text(
        json.dumps(_record("T1", "2026-07-01", ARM_QWEN)), encoding="utf-8",
    )
    (tmp_path / "T2_2026-07-01_claude_r1_error.json").write_text(
        json.dumps({"ticker": "T2", "date": "2026-07-01", "arm": ARM_CLAUDE, "repeat": 1, "error": "boom"}),
        encoding="utf-8",
    )

    records, error_counts = _load_records(tmp_path)
    assert len(records) == 2
    assert error_counts == {ARM_CLAUDE: 1}

    report = compute_backend_ab_report(records, error_counts, min_pairs=1, min_field_fires=1)
    assert report.claude_census.errors == 1
    assert report.qwen_census.errors == 0
