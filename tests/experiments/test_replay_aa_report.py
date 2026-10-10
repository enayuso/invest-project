"""app.experiments.replay.aa_report — A/A metrics, verdict order, listings (spec F, G).
Synthetic run data only: no DB, no LLM, no network."""

from __future__ import annotations

import copy
import json

import pytest

from app.experiments import replay
from app.experiments.replay import aa_report as aa
from replay_fixtures import install_db_block


@pytest.fixture(autouse=True)
def _block_db(monkeypatch):
    install_db_block(monkeypatch)


def rating(flag: int) -> str:
    return "Buy" if flag else "Hold"


def make_case(i, r_rm, r_final, s_rm, s_final, interim=True, neutral="Hold", s_neutral="Hold", attempt=1, sha="a" * 64):
    return {
        "schema": "replay-case-v1", "case_id": f"c{i}", "ticker": f"00000{i}.KS", "trade_date": "2026-09-22",
        "status": "ok", "attempt": attempt,
        "covariates": {"interim": interim, "kr_risk_neutralized": False, "session_meta": {"agree": True}},
        "contract_checks": {"desk_note_stored_match": True},
        "summary": {"rm": r_rm, "trader": "Hold", "neutral_final": neutral, "pm_pre_gate": r_final, "final": r_final, "confidence": 0.6},
        "stored": {"rm": s_rm, "trader": "Hold", "neutral_final": s_neutral, "pm_pre_gate": s_final, "final": s_final, "confidence": 0.5},
        "calls_summary": {"n": 9, "n_structured": 3, "n_fallback": 0, "n_length": 0, "prompt_sha256_by_label": {"research_manager": [sha]}},
        "nodes": {"gates": {"consensus_override": {}, "hysteresis_applied": False}, "rm": {"tokens": 5}},
        "tokens_total": 90, "wall_s": 12.0,
    }


SNAP = {
    "tables": {t: {"max_id": 1, "count": 1, "max_at": "x"} for t in ("memory_entries", "trades", "pending_orders", "alerts")},
    "digests": {"cases": "c", "risk_rules": "r", "runtime_settings": "s"}, "new_rows": {},
}


def make_run(cases, idx=1, **manifest):
    base = {
        "n_cases": len(cases), "safety_trips": 0, "status": "done",
        "invariants": {"start": copy.deepcopy(SNAP), "end": copy.deepcopy(SNAP)},
        "run_id": f"run{idx}", "repeat": idx, "base_key": "K" * 64, "variant": "baseline", "fork_at": "rm",
        "caseset": {"name": "cs", "sha256": "c" * 64}, "windows": [],
    }
    base.update(manifest)
    return aa.RunData(manifest=base, cases={c["case_id"]: c for c in cases})


def hand_example():
    r1_rm = [1, 1, 0, 0, 0, 0, 0, 0, 0, 1]
    r2_rm = [1, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    st_rm = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
    r1 = [make_case(i + 1, rating(a), rating(a), rating(s), rating(s)) for i, (a, s) in enumerate(zip(r1_rm, st_rm))]
    r2 = [make_case(i + 1, rating(b), rating(b), rating(s), rating(s)) for i, (b, s) in enumerate(zip(r2_rm, st_rm))]
    return make_run(r1, idx=1), make_run(r2, idx=2)


def row(report, rid, metric_part=""):
    matches = [r for r in report["rows"] if r["id"] == rid and metric_part in r["metric"]]
    assert matches, (rid, metric_part)
    return matches[0]


def test_aa_metrics_hand_example():
    r1, r2 = hand_example()
    report = aa.build_report(r1, r2)
    assert report["n_population"] == 10
    assert row(report, "AA-2a")["value"] == pytest.approx(0.10)  # d(R1,R2) RM
    assert row(report, "AA-2a")["verdict"] == "FAIL"              # 10% > 8%
    assert report["rm_vs_stored"]["d_vs_stored"] == pytest.approx(0.20)
    a3a = row(report, "AA-3a")
    assert a3a["value"] == pytest.approx(0.10) and a3a["verdict"] == "FAIL"  # 0.20 > 0.10 + 0.05
    a3b = row(report, "AA-3b")
    assert a3b["value"] == pytest.approx(0.0) and a3b["verdict"] == "PASS"
    assert report["rm_vs_stored"]["bullrate_run"] == pytest.approx(0.3) == report["rm_vs_stored"]["bullrate_stored"]
    assert (report["rm_vs_stored"]["b"], report["rm_vs_stored"]["c"]) == (1, 1)
    a3c = row(report, "AA-3c")
    assert a3c["value"] == 1.0 and a3c["verdict"] == "PASS"


def test_mcnemar_exact():
    assert aa.mcnemar_exact(0, 6) == pytest.approx(0.03125)
    assert aa.mcnemar_exact(1, 7) == pytest.approx(0.0703125)
    assert aa.mcnemar_exact(0, 0) == 1.0
    assert aa.mcnemar_exact(1, 1) == 1.0
    assert aa.mcnemar_exact(2, 2) == 1.0 and aa.mcnemar_exact(10, 0) < 0.005


def test_fisher_exact():
    assert aa.fisher_exact(1, 4, 0, 5) == pytest.approx(1.0)
    assert aa.fisher_exact(5, 0, 0, 5) == pytest.approx(2 / 252)  # 2 / C(10,5)
    assert 0 < aa.fisher_exact(3, 7, 9, 2) < 0.1


def test_interim_gap():
    interim = [True] * 5 + [False] * 4 + [None]
    r1_rm = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    stored = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]  # the only R1-vs-stored mismatch is case 3 (group A)
    r1 = make_run([make_case(i + 1, rating(a), rating(a), rating(s), rating(s), interim=g) for i, (a, s, g) in enumerate(zip(r1_rm, stored, interim))])
    r2 = make_run([make_case(i + 1, rating(a), rating(a), rating(s), rating(s), interim=g) for i, (a, s, g) in enumerate(zip(r1_rm, stored, interim))])
    report = aa.build_report(r1, r2)
    split = report["interim"]
    assert (split["n_A"], split["n_B"]) == (5, 5)  # B = False U None
    assert split["d_A"] == pytest.approx(0.20) and split["d_B"] == pytest.approx(0.0)
    assert split["gap"] == pytest.approx(0.20) and split["restore_period_note_needed"] is True
    assert row(report, "AA-3d")["verdict"] == "FAIL"
    assert split["three_way"]["none"]["n"] == 1 and split["three_way"]["false"]["n"] == 4
    assert 0 < split["fisher_p"] <= 1
    # None counts with False, never with True
    only_none = aa.interim_split(r1, ["c10"])
    assert only_none["n_A"] == 0 and only_none["n_B"] == 1


def test_overweight_is_bullish_and_neutral_none_excluded():
    assert aa.bull("Overweight") is True and aa.bull("Buy") is True
    assert aa.bull("Underweight") is False and aa.bull("Sell") is False and aa.bull("Hold") is False
    assert aa.bull(None) is None
    r1 = make_run([
        make_case(1, "Overweight", "Hold", "Hold", "Hold", neutral=None),
        make_case(2, "Hold", "Hold", "Hold", "Hold", neutral="Buy"),
        make_case(3, "Underweight", "Hold", "Hold", "Hold", neutral="Hold"),
    ])
    r2 = make_run([
        make_case(1, "Hold", "Hold", "Hold", "Hold", neutral="Hold"),
        make_case(2, "Hold", "Hold", "Hold", "Hold", neutral="Hold"),
        make_case(3, "Hold", "Hold", "Hold", "Hold", neutral="Hold"),
    ])
    report = aa.build_report(r1, r2)
    assert row(report, "AA-2a")["value"] == pytest.approx(1 / 3)  # Overweight vs Hold differs; Underweight vs Hold does not
    neutral = report["info_2c"]["neutral_final"]
    assert neutral["n"] == 2 and neutral["d_R1_R2"] == pytest.approx(0.5)  # case 1 (None) excluded
    assert row(report, "AA-2c", "neutral_final")["verdict"] == "INFO"


def test_changed_cases_listing():
    r1, r2 = hand_example()
    ids = aa.population(r1, r2)
    changed = {c["case_id"]: c["change_kind"] for c in aa.changed_cases(r1, r2, ids)}
    assert set(changed) == {"c2", "c3", "c10"}
    assert changed["c2"] == "R1≠R2 + R2≠S"
    assert "R1≠S" in changed["c3"] and "R1≠R2" not in changed["c3"]
    assert "R1≠S" in changed["c10"] and "R1≠R2" not in changed["c10"]
    csv_text = aa.changed_cases_csv(aa.changed_cases(r1, r2, ids))
    header = csv_text.splitlines()[0].split(",")
    assert header[:4] == ["case_id", "ticker", "trade_date", "interim"] and header[-1] == "change_kind"
    assert len(csv_text.splitlines()) == 4


def _clean_run(n=10):
    cases = [make_case(i, "Hold", "Hold", "Hold", "Hold") for i in range(1, n + 1)]
    return make_run(cases, idx=1), make_run(copy.deepcopy(cases), idx=2)


def test_verdict_pass_and_order():
    r1, r2 = _clean_run()
    report = aa.build_report(r1, r2)
    assert report["verdict"] == aa.VERDICT_PASS
    assert all(r["verdict"] != "FAIL" for r in report["rows"])

    # AA-2 only -> three repeats
    r2b = copy.deepcopy(r2)
    for c in list(r2b.cases.values())[:2]:
        c["summary"]["rm"] = "Buy"
    assert aa.build_report(r1, r2b)["verdict"].startswith("AA-2 only")  # also trips AA-3 if stored differs? no: stored is Hold
    # (R2 differs from R1 AND from stored -> AA-3 unaffected because AA-3 uses R1 vs stored)

    # AA-3 failing -> input-contract defect
    r1c = copy.deepcopy(r1)
    for c in list(r1c.cases.values())[:3]:
        c["summary"]["rm"] = "Buy"
    r2c = copy.deepcopy(r1c)
    assert aa.build_report(r1c, r2c)["verdict"] == aa.VERDICT_CONTRACT

    # AA-1 failing beats everything
    r2d = copy.deepcopy(r1c)
    r2d.manifest["safety_trips"] = 1
    assert aa.build_report(r1c, r2d)["verdict"] == aa.VERDICT_NOGO


def test_aa1_checks_individually():
    r1, r2 = _clean_run()
    base = aa.build_report(r1, r2)
    assert row(base, "AA-1c", "R1")["value"] == 1.0 and row(base, "AA-1f")["value"] == 1.0

    low = copy.deepcopy(r1)
    low.manifest["n_cases"] = 12  # 10/12 completed
    assert row(aa.build_report(low, r2), "AA-1c", "R1")["verdict"] == "FAIL"

    fb = copy.deepcopy(r1)
    for c in fb.cases.values():
        c["calls_summary"]["n_fallback"] = 1  # 10 of 30 structured calls
    assert row(aa.build_report(fb, r2), "AA-1d")["verdict"] == "FAIL"

    ln = copy.deepcopy(r1)
    next(iter(ln.cases.values()))["calls_summary"]["n_length"] = 9  # 9 / 180 = 5% of all calls
    assert row(aa.build_report(ln, r2), "AA-1e")["verdict"] == "FAIL"

    sha = copy.deepcopy(r2)
    next(iter(sha.cases.values()))["calls_summary"]["prompt_sha256_by_label"]["research_manager"] = ["b" * 64]
    rep = aa.build_report(r1, sha)
    assert row(rep, "AA-1f")["verdict"] == "FAIL" and rep["rm_prompt_sha_mismatch_cases"] == ["c1"]

    desk = copy.deepcopy(r1)
    next(iter(desk.cases.values()))["contract_checks"]["desk_note_stored_match"] = False
    assert row(aa.build_report(desk, r2), "AA-1g")["verdict"] == "FAIL"

    changed_db = copy.deepcopy(r2)
    changed_db.manifest["invariants"]["end"]["tables"]["memory_entries"]["count"] = 2
    changed_db.manifest["invariants"]["end"]["new_rows"] = {"memory_entries": [{"id": 9, "ticker": "x", "kind": None, "at": "y"}]}
    changed_db.manifest["invariants"]["end"]["tables"]["memory_entries"]["max_id"] = 9
    assert row(aa.build_report(r1, changed_db), "AA-1a")["verdict"] == "FAIL"

    missing = copy.deepcopy(r1)
    del missing.manifest["invariants"]
    assert row(aa.build_report(missing, r2), "AA-1a")["verdict"] == "FAIL"

    warn = copy.deepcopy(r2)
    warn.manifest["invariants"]["end"]["tables"]["alerts"]["count"] = 2
    warn.manifest["invariants"]["end"]["tables"]["alerts"]["max_id"] = 5
    warn.manifest["invariants"]["end"]["new_rows"] = {"alerts": [{"id": 5, "ticker": "z", "kind": "price_move", "at": "y"}]}
    rep = aa.build_report(r1, warn)
    assert row(rep, "AA-1a")["verdict"] == "PASS" and rep["invariants"]["warns"]  # WARNs are listed, not failed


def test_thresholds_match_the_preregistered_docstring():
    doc = replay.__doc__
    for line in (
        "| AA-1a | DB 불변식 R1 시작 → R2 종료 | FAIL 0(WARN은 목록 제시) |",
        "| AA-1b | safety trips | 0 |",
        "≥ 98%(첫 시도 성공률도 보고)",
        "**≤ 3% (보완 1: 라이브 기준값이 저장돼 있지 않아 절대 상한으로 대체.",
        "≤ 2 × 1.4% = 2.8%(1단계 `/metrics`)",
        "**(보완 2: 새 무결성 검사. 입력이 같으면 RM 요청 바이트가 같아야 함)**",
        "100%(지금 179/179)",
        "| AA-2a | d(R1,R2) RM | ≤ 8% |",
        "| AA-2b | d(R1,R2) final(게이트 후) | ≤ 3% |",
        "정보용 |",
        "≤ +5%p |",
        "| ≤ 5%p |",
        "p ≥ 0.05 |",
        "**B = interim False ∪ None**",
        "< 10%p, 아니면 `period_note` 복원 필요",
        "**(보완 3: None = 캐시 적중이라 라이브도 `period_note=\"\"` → B는 라이브 입력과 정확히 같은 대조군. 3분할도 함께 보고)**",
        "**AA-2만 실패 → arm당 3회(repeat 1..3) 설계로 전환**",
        "검정력 주의: B 그룹 n=39, A 140.",
    ):
        assert line in doc, line
    assert (aa.T_COMPLETION, aa.T_FALLBACK, aa.T_LENGTH) == (0.98, 0.03, 0.028)
    assert (aa.T_AA2A, aa.T_AA2B, aa.T_AA3A, aa.T_AA3B, aa.T_AA3C, aa.T_AA3D) == (0.08, 0.03, 0.05, 0.05, 0.05, 0.10)
    assert "freeze" in doc.lower() and "handoff" in doc.lower()


def test_write_and_load_roundtrip(tmp_path):
    r1, r2 = hand_example()
    for name, run in (("r1", r1), ("r2", r2)):
        run_dir = tmp_path / name
        (run_dir / "cases").mkdir(parents=True)
        (run_dir / "manifest.json").write_text(json.dumps(run.manifest), encoding="utf-8")
        for case in run.cases.values():
            (run_dir / "cases" / f"{case['case_id']}.json").write_text(json.dumps(case), encoding="utf-8")
        failed = dict(next(iter(run.cases.values())), case_id="zz_err", status="error")
        (run_dir / "cases" / "zz_err.json").write_text(json.dumps(failed), encoding="utf-8")
    report, paths = aa.make_report(tmp_path / "r1", tmp_path / "r2", tmp_path / "reports", "20260930T120000")
    assert report["n_population"] == 10  # the error-status file is not an ok case
    md = paths["md"].read_text(encoding="utf-8")
    assert "Verdict:" in md and "AA-3a" in md and "Changed cases: 3" in md
    assert json.loads(paths["json"].read_text(encoding="utf-8"))["n_changed_cases"] == 3
    assert paths["csv"].read_text(encoding="utf-8").splitlines()[0].startswith("case_id,")
    assert "c10" in paths["changed_md"].read_text(encoding="utf-8")
    assert paths["md"].parent.name == "aa_20260930T120000"


# ---------------------------------------------------------------- M4: pair validation


def broken_first(run, **changes):
    run = copy.deepcopy(run)
    run.manifest.update(changes)
    return run


def test_validate_pair_accepts_a_proper_pair_and_rejects_each_defect():
    r1, r2 = _clean_run()
    aa.validate_pair(r1, r2)

    same_dir_twice = copy.deepcopy(r1)
    with pytest.raises(aa.PairError, match="run_id must differ.*repeat must differ"):
        aa.validate_pair(r1, same_dir_twice)
    cases = {
        "base_key": broken_first(r2, base_key="X" * 64),
        "run_id": broken_first(r2, run_id="run1"),
        "repeat": broken_first(r2, repeat=1),
        "variant": broken_first(r2, variant="_test_pm_marker"),
        "fork_at": broken_first(r2, fork_at="pm"),
        "case set sha256": broken_first(r2, caseset={"name": "cs", "sha256": "d" * 64}),
        "status": broken_first(r2, status="paused_window"),
    }
    for needle, other in cases.items():
        with pytest.raises(aa.PairError, match=needle):
            aa.validate_pair(r1, other)
    with pytest.raises(aa.PairError, match="R1 status"):
        aa.validate_pair(broken_first(r1, status="aborted_safety"), r2)


def test_make_report_refuses_the_same_directory_twice(tmp_path):
    r1, _ = _clean_run()
    run_dir = tmp_path / "r1"
    (run_dir / "cases").mkdir(parents=True)
    (run_dir / "manifest.json").write_text(json.dumps(r1.manifest), encoding="utf-8")
    for case in r1.cases.values():
        (run_dir / "cases" / f"{case['case_id']}.json").write_text(json.dumps(case), encoding="utf-8")
    with pytest.raises(aa.PairError):
        aa.make_report(run_dir, run_dir, tmp_path / "reports", "s")


# ---------------------------------------------------------------- M2: new-row classification


def _windows(*spans):
    from datetime import datetime

    return [(datetime.fromisoformat(a), datetime.fromisoformat(b)) for a, b in spans]


WIN = _windows(("2026-09-30T16:00:00+09:00", "2026-09-30T20:00:00+09:00"))


def _new_row(id, at, is_backtest=False, ticker="005930.KS"):
    return {"id": id, "at": at, "ticker": ticker, "kind": None, "is_backtest": is_backtest,
            "account_mode": "real", "decision_trigger": "auto"}


def test_classify_new_row_rule():
    from app.experiments.replay import safety

    inside = _new_row(1, "2026-09-30T17:00:00+09:00")
    before = _new_row(2, "2026-09-30T10:00:00+09:00")
    after_end = _new_row(3, "2026-09-30T21:00:00+09:00")
    assert safety.classify_new_row(inside, WIN) == "harness_window"  # even when is_backtest=false
    assert safety.classify_new_row(before, WIN) == "auto_live"
    assert safety.classify_new_row(after_end, WIN) == "auto_live"
    assert safety.classify_new_row(_new_row(4, "2026-09-30T10:00:00+09:00", is_backtest=True), WIN) == "candidate"
    assert safety.classify_new_row(_new_row(5, "2026-09-30T10:00:00+09:00", is_backtest=None), WIN) == "candidate"
    assert safety.classify_new_row(_new_row(6, "not a time"), WIN) == "candidate"
    assert safety.classify_new_row(_new_row(7, "2026-09-30T02:00:00"), WIN) == "auto_live"  # naive => UTC, outside
    # window boundaries are inclusive
    assert safety.classify_new_row(_new_row(8, "2026-09-30T16:00:00+09:00"), WIN) == "harness_window"
    assert safety.classify_new_row(_new_row(9, "2026-09-30T20:00:00+09:00"), WIN) == "harness_window"
    assert safety.classify_new_row(before, []) == "auto_live"


def _snap_with_rows(table, rows):
    snap = copy.deepcopy(SNAP)
    t = snap["tables"][table]
    t["max_id"] += len(rows)
    t["count"] += len(rows)
    t["max_at"] = "2026-09-30T23:00:00+00:00"
    snap["new_rows"] = {table: rows}
    return snap


def test_compare_invariants_windows_and_confirmation():
    from app.experiments.replay import safety

    tickers = {"005930.KS"}
    live_before_run = _new_row(11, "2026-09-30T10:00:00+09:00")
    after = _snap_with_rows("memory_entries", [live_before_run])
    res = safety.compare_invariants(SNAP, after, tickers, run_windows=WIN)
    assert res.status == "WARN" and [r["id"] for r in res.auto_live] == [11] and not res.candidates

    # inside a harness window: a hard FAIL that no confirmation can release
    inside = _new_row(12, "2026-09-30T17:00:00+09:00")
    after = _snap_with_rows("trades", [inside])
    res = safety.compare_invariants(SNAP, after, tickers, run_windows=WIN, confirmed={"trades": [12]})
    assert res.status == "FAIL" and "INSIDE a harness run window" in res.fails[0] and not res.released

    # unknown provenance outside the windows: FAIL_CANDIDATE until the human confirms the id
    maybe = _new_row(13, "2026-09-30T10:00:00+09:00", is_backtest=None)
    after = _snap_with_rows("pending_orders", [maybe])
    res = safety.compare_invariants(SNAP, after, tickers, run_windows=WIN)
    assert res.status == "FAIL_CANDIDATE" and res.candidates[0]["table"] == "pending_orders"
    res = safety.compare_invariants(SNAP, after, tickers, run_windows=WIN, confirmed={"pending_orders": [13]})
    assert res.status == "PASS" and [r["id"] for r in res.released] == [13]
    # a confirmation for another id or table releases nothing
    res = safety.compare_invariants(SNAP, after, tickers, run_windows=WIN, confirmed={"pending_orders": [99], "trades": [13]})
    assert res.status == "FAIL_CANDIDATE"

    # rows that appeared but cannot be listed are a hard fail (cannot be classified)
    after = copy.deepcopy(SNAP)
    after["tables"]["memory_entries"]["count"] += 1
    assert safety.compare_invariants(SNAP, after, tickers, run_windows=WIN).status == "FAIL"


def _run_with_end_rows(cases, idx, table, rows, windows):
    run = make_run(cases, idx=idx)
    run.manifest["windows"] = windows
    run.manifest["invariants"]["end"] = _snap_with_rows(table, rows)
    return run


def test_aa1a_across_two_nights_with_live_rows_between_runs():
    """R1 and R2 straddle KRX hours: live rows written between the two harness windows are not a
    structural AA-1a failure; a row inside a window is."""
    cases = [make_case(i, "Hold", "Hold", "Hold", "Hold") for i in range(1, 4)]
    w1 = {"start": "2026-09-29T16:00:00+09:00", "end": "2026-09-29T20:00:00+09:00"}
    w2 = {"start": "2026-09-30T16:00:00+09:00", "end": "2026-09-30T20:00:00+09:00"}
    r1 = make_run(cases, idx=1)
    r1.manifest["windows"] = [w1]
    live_daytime = _new_row(21, "2026-09-30T10:30:00+09:00")
    r2 = _run_with_end_rows(copy.deepcopy(cases), 2, "memory_entries", [live_daytime], [w2])
    report = aa.build_report(r1, r2)
    assert row(report, "AA-1a")["verdict"] == "PASS" and report["invariants"]["status"] == "WARN"
    assert [r["id"] for r in report["invariants"]["auto_live"]] == [21]

    written_by_harness = _new_row(22, "2026-09-30T17:00:00+09:00")
    r2b = _run_with_end_rows(copy.deepcopy(cases), 2, "memory_entries", [written_by_harness], [w2])
    assert row(aa.build_report(r1, r2b), "AA-1a")["verdict"] == "FAIL"

    unknown = _new_row(23, "2026-09-30T10:30:00+09:00", is_backtest=None)
    r2c = _run_with_end_rows(copy.deepcopy(cases), 2, "memory_entries", [unknown], [w2])
    rep = aa.build_report(r1, r2c)
    assert row(rep, "AA-1a")["verdict"] == "FAIL" and rep["invariants"]["status"] == "FAIL_CANDIDATE"
    rep = aa.build_report(
        r1, r2c, confirmed={"memory_entries": [23]}, confirm_info={"path": "f.json", "ids": {"memory_entries": [23]}}
    )
    assert row(rep, "AA-1a")["verdict"] == "PASS" and rep["invariants"]["released"][0]["id"] == 23
    md = aa.render_markdown(rep, [])
    assert "released by the human" in md and "f.json" in md and "HTTP-call-level" in md


def test_confirm_file_loading(tmp_path):
    f = tmp_path / "confirm.json"
    f.write_text(json.dumps({"memory_entries": [1, 2], "trades": []}), encoding="utf-8")
    ids, info = aa.load_confirm_file(f)
    assert ids == {"memory_entries": [1, 2], "trades": []} and len(info["sha256"]) == 64
    f.write_text(json.dumps({"alerts": [1]}), encoding="utf-8")
    with pytest.raises(aa.PairError, match="unknown tables"):
        aa.load_confirm_file(f)
    f.write_text("[1]", encoding="utf-8")
    with pytest.raises(aa.PairError):
        aa.load_confirm_file(f)


def test_preregistered_aa1a_clarification_present():
    doc = replay.__doc__
    assert "| AA-1a | DB 불변식 R1 시작 → R2 종료 | FAIL 0(WARN은 목록 제시) |" in doc
    assert "AA-1a clarification" in doc and "nothing is loosened" in doc
