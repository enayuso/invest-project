"""app.experiments.replay_downstream (CLI) — import order, argument surface, pure helpers.
No DB, no LLM, no network: nothing here reaches `bootstrap` with a real pinned file."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from app.experiments import replay_downstream as cli
from replay_fixtures import install_db_block, make_row, make_transcript

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _block_db(monkeypatch):
    install_db_block(monkeypatch)


def test_cli_import_does_not_load_config():
    code = (
        "import sys, app.experiments.replay_downstream; "
        "assert 'app.config' not in sys.modules, 'app.config was imported at CLI import time'; "
        "assert 'app.orchestrator.pipeline' not in sys.modules"
    )
    proc = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


def test_bootstrap_applies_pinned_env_before_config_is_imported(tmp_path):
    """End-to-end order check in a fresh interpreter: after `pinned.apply_to_environ` a later
    `from app.config import settings` sees the pinned value and the harness K."""
    code = (
        "import sys\n"
        "from app.experiments.replay import pinned\n"
        "p = pinned.parse(pinned.default_document())\n"
        "doc = pinned.default_document(); doc['settings']['MAX_RISK_ROUNDS']['value'] = 3\n"
        "p = pinned.parse(doc)\n"
        "assert 'app.config' not in sys.modules\n"
        "pinned.apply_to_environ(p, k=5)\n"
        "from app.config import settings\n"
        "assert settings.max_risk_rounds == 3 and settings.llm_max_concurrent_requests == 5\n"
        "from app.llm import llm_client\n"
        "assert llm_client._limiter._max_concurrent == 5\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


def test_parser_accepts_the_documented_commands():
    parser = cli.build_parser()
    a = parser.parse_args(["extract", "--name", "aa_freeze_0929", "--era", "freeze", "--min-trade-date", "2026-09-17", "--max-trade-date", "2026-09-29"])
    assert a.command == "extract" and a.era == "freeze"
    a = parser.parse_args(["extract", "--name", "pilot12", "--from-caseset", "dev_0916", "--sample", "12", "--seed", "20261001", "--min-stored-rm-buy", "4"])
    assert (a.sample, a.seed, a.min_stored_rm_buy) == (12, 20261001, 4)
    a = parser.parse_args(["preflight", "--pinned", "P", "--caseset", "aa_freeze_0929", "--k", "4"])
    assert a.k == 4 and a.pinned == "P"
    a = parser.parse_args(["pilot", "--pinned", "P", "--caseset", "pilot12", "--ks", "2,4,6"])
    assert a.ks == "2,4,6"
    a = parser.parse_args([
        "run", "--pinned", "P", "--caseset", "aa_freeze_0929", "--arm", "baseline", "--repeat", "1", "--k", "4",
        "--baseline-run", "D", "--max-cases", "1", "--run-tag", "smoke", "--case-timeout", "1800", "--max-attempts", "2",
        "--allow-market-hours", "--allow-test-variant",
    ])
    assert (a.arm, a.repeat, a.max_cases, a.run_tag, a.allow_market_hours, a.allow_test_variant) == ("baseline", 1, 1, "smoke", True, True)
    a = parser.parse_args(["aa-report", "--r1", "A", "--r2", "B"])
    assert (a.r1, a.r2) == ("A", "B")
    assert parser.parse_args(["status", "--run", "D"]).run == "D"
    with pytest.raises(SystemExit):
        parser.parse_args(["run", "--caseset", "x", "--era", "freeze"])


def test_exit_codes(monkeypatch, tmp_path, capsys):
    assert cli.main(["run", "--pinned", str(tmp_path / "missing.json"), "--caseset", "x"]) == cli.EXIT_CONFIG
    assert "cannot read" in capsys.readouterr().err

    def raise_safety(args):
        from app.experiments.replay import safety

        raise safety.ReplaySafetyViolation("blocked")

    monkeypatch.setattr(cli, "cmd_status", raise_safety)
    assert cli.main(["status", "--run", str(tmp_path)]) == cli.EXIT_SAFETY

    def raise_window(args):
        from app.experiments.replay import safety

        raise safety.MarketWindowError("KRX hours")

    monkeypatch.setattr(cli, "cmd_status", raise_window)
    assert cli.main(["status", "--run", str(tmp_path)]) == cli.EXIT_CONFIG

    def raise_other(args):
        raise ZeroDivisionError("bug")

    monkeypatch.setattr(cli, "cmd_status", raise_other)
    with pytest.raises(ZeroDivisionError):
        cli.main(["status", "--run", str(tmp_path)])
    assert (cli.EXIT_OK, cli.EXIT_CONFIG, cli.EXIT_SAFETY, cli.EXIT_PAUSED, cli.EXIT_INCOMPLETE) == (0, 2, 3, 75, 76)
    assert "76" in cli.build_parser().format_help()


def test_status_command(tmp_path, capsys):
    (tmp_path / "cases").mkdir()
    (tmp_path / "cases" / "a.json").write_text("{}", encoding="utf-8")
    (tmp_path / "manifest.json").write_text(json.dumps({"run_id": "r", "status": "paused_window", "n_cases": 5, "k": 2, "safety_trips": 0}), encoding="utf-8")
    assert cli.main(["status", "--run", str(tmp_path)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "paused_window" and out["ok_files"] == 1 and out["error_files"] == 0


def test_parse_prometheus():
    text = "\n".join([
        "# HELP vllm:num_requests_running running",
        'vllm:num_requests_running{model_name="m"} 2.0',
        'vllm:num_requests_running{model_name="m2"} 1.0',
        "vllm:num_requests_waiting 0.0",
        "vllm:num_requests_running_total 99.0",
        'vllm:prompt_tokens_total{model_name="m"} 12345.0',
    ])
    out = cli.parse_prometheus(text, ["vllm:num_requests_running", "vllm:num_requests_waiting", "vllm:prompt_tokens_total"])
    assert out == {"vllm:num_requests_running": 3.0, "vllm:num_requests_waiting": 0.0, "vllm:prompt_tokens_total": 12345.0}


def _stats(errors=0, cph=10.0, rm_p90=10.0, foreign=0.0):
    return {"errors": errors, "cases_per_hour": cph, "rm_p90": rm_p90, "foreign_share": foreign}


def test_select_k_star_rule():
    # K=4 gives +50% over K=2, K=6 only +10% -> K*=4
    stats = {2: _stats(cph=10), 4: _stats(cph=15, rm_p90=15), 6: _stats(cph=16.5, rm_p90=20)}
    decision = cli.select_k_star(stats)
    assert decision["k_star"] == 4 and decision["shared"] is False
    # latency blows past 2.5x of K=2 -> stop at the lower K
    assert cli.select_k_star({2: _stats(cph=10), 4: _stats(cph=30, rm_p90=30)})["k_star"] == 2
    # errors at a higher K disqualify it
    assert cli.select_k_star({2: _stats(cph=10), 4: _stats(errors=1, cph=30)})["k_star"] == 2
    # errors at the smallest K -> no choice at all
    assert cli.select_k_star({2: _stats(errors=2)})["k_star"] is None
    # foreign load above 20% is reported as shared
    shared = cli.select_k_star({2: _stats(cph=10, foreign=0.3), 4: _stats(cph=15, rm_p90=12)})
    assert shared["shared"] is True and shared["k_star"] == 4
    assert cli.select_k_star({})["k_star"] is None


def test_sample_is_deterministic_and_respects_min_buy():
    from app.experiments.replay import contract

    records = []
    for i in range(1, 21):
        plan = "**Recommendation**: Buy\n\n**Rationale**: r" if i % 5 == 0 else "**Recommendation**: Hold\n\n**Rationale**: r"
        row = make_row(id=i, ticker=f"0000{i:02d}.KS", transcript=make_transcript(research_plan=plan))
        records.append(contract.record_from_row(row))
    first = cli._sample_cases(records, sample=8, seed=7, min_buy=3)
    again = cli._sample_cases(records, sample=8, seed=7, min_buy=3)
    assert [r["case_id"] for r in first] == [r["case_id"] for r in again] and len(first) == 8
    assert sum(contract.parse_stored_rm(r["transcript"]["research_plan"]) == "Buy" for r in first) >= 3
    assert [r["case_id"] for r in first] == sorted((r["case_id"] for r in first), key=lambda c: (
        next(r["trade_date"] for r in first if r["case_id"] == c), c.split("_")[0]))
    other = cli._sample_cases(records, sample=8, seed=8, min_buy=3)
    assert [r["case_id"] for r in other] != [r["case_id"] for r in first]
    with pytest.raises(cli.CliRefusal):
        cli._sample_cases(records, sample=8, seed=7, min_buy=5)  # only 4 Buy rows exist


def test_pilot_stats_and_report(tmp_path):
    run = tmp_path / "run"
    (run / "cases").mkdir(parents=True)
    for i, wall in enumerate((100.0, 200.0), start=1):
        case = {
            "wall_s": wall, "tokens_total": 1000,
            "nodes": {"rm": {"calls": [{"label": "research_manager", "latency_seconds": 30.0 * i}]}, "trader": {"calls": []}},
        }
        (run / "cases" / f"c{i}.json").write_text(json.dumps(case), encoding="utf-8")
    stats = cli._pilot_stats(run, wall=600.0, foreign=[0.0, 1.0, 0.0, 0.0], returncode=0)
    assert stats["cases"] == 2 and stats["cases_per_hour"] == pytest.approx(12.0)
    assert stats["foreign_share"] == 0.25 and stats["rm_p90"] == pytest.approx(57.0)
    assert stats["tokens_per_s"] == pytest.approx(2000 / 600)
    decision = cli.select_k_star({2: stats})
    text = cli.render_pilot_report({2: stats}, decision)
    assert "K\\* = 2" in text and "SHARED" in text
    assert cli.percentile([], 0.5) is None and cli.percentile([1.0, 3.0], 0.5) == 2.0


def test_default_pinned_path_is_under_gitignored_results():
    rel = cli.DEFAULT_PINNED.relative_to(cli.REPO_ROOT).as_posix()
    assert rel == "results/replay_downstream/pinned/server-2026-09-30.json"
    proc = subprocess.run(["git", "check-ignore", rel], cwd=REPO, capture_output=True, text=True)
    assert proc.returncode == 0, "results/ must stay gitignored (transcripts must never be committed)"


async def test_run_async_end_to_end_with_fakes_and_aa_report(monkeypatch, tmp_path, capsys):
    """Wiring test for everything the CLI adds around the runner: manifest content, DB invariant
    snapshots (fake read-only connection), resume, then `aa-report` over two runs. Fake LLM,
    no DB, no network."""
    import argparse

    from app.config import settings
    from app.experiments.replay import aa_report, pinned, runner, safety
    from app.llm import llm_client
    from replay_fixtures import FakeLLM, FakeReadonlyConn, desk_note_for, make_inputs

    fake = FakeLLM()
    monkeypatch.setattr(llm_client, "complete", fake.complete)
    monkeypatch.setattr(llm_client, "structured_complete", fake.structured_complete)
    instrumentation = runner.Instrumentation()
    instrumentation.install()

    async def open_ro(*args, **kwargs):
        return FakeReadonlyConn()

    monkeypatch.setattr(safety, "open_readonly_connection", open_ro)
    monkeypatch.setattr(cli, "REPORTS", tmp_path / "reports")
    try:
        inputs = [
            make_inputs(make_row(id=i, ticker=f"00000{i}.KS", transcript=make_transcript(desk_consensus_note=desk_note_for())))
            for i in (1, 2, 3)
        ]
        assert all(i.contract_checks["desk_note_stored_match"] is True for i in inputs)
        fake.tokens = {i.ticker: 10 for i in inputs}
        boot = cli.Boot(pinned.parse(pinned.default_document()), settings, [])
        prov = {
            "source_hashes": {"file:a": "1"}, "git": {"head": "abc", "pinned_files_dirty": False, "dirty_lines": []},
            "model": {"id": "m", "max_model_len": 1}, "base_key": "b" * 64, "caseset_sha256": "c" * 64,
        }

        def args(repeat):
            return argparse.Namespace(
                arm="baseline", baseline_run=None, allow_test_variant=False, allow_noop_variant=False, repeat=repeat,
                run_tag="t", k=2, caseset="cs", allow_market_hours=True, allow_dirty=False, accept_default_drift=False,
                case_timeout=60.0, max_attempts=2, max_consecutive_failures=3,
            )

        dirs = []
        for repeat in (1, 2):
            run_dir = tmp_path / f"run{repeat}"
            code = await cli._run_async(args(repeat), boot, inputs, prov, run_dir, f"run{repeat}")
            assert code == 0
            dirs.append(run_dir)
        manifest = json.loads((dirs[0] / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["status"] == "done" and manifest["n_cases"] == 3 and manifest["k"] == 2
        assert manifest["invariants"]["result"]["status"] == "PASS"
        assert manifest["invariants"]["start"]["tables"]["memory_entries"]["count"] == 5
        assert set(manifest["node_fingerprints"]) == set(runner.NODES) and manifest["variant_patches"] == []
        assert manifest["pinned"]["values"]["MAX_RISK_ROUNDS"] == 2 and manifest["flags"]["allow_market_hours"] is True
        assert manifest["safety_trips"] == 0 and manifest["heartbeat"]["done"] == 3
        assert len(list((dirs[0] / "cases").glob("*.json"))) == 3

        # resuming a finished run does no LLM work and keeps the first invariants snapshot
        before = len(fake.calls)
        assert await cli._run_async(args(1), boot, inputs, prov, dirs[0], "run1") == 0
        assert len(fake.calls) == before
        again = json.loads((dirs[0] / "manifest.json").read_text(encoding="utf-8"))
        assert again["resumed"] == 1 and again["invariants"]["start"] == manifest["invariants"]["start"]

        assert cli.main(["aa-report", "--r1", str(dirs[0]), "--r2", str(dirs[1])]) == 0
        out = capsys.readouterr().out
        assert "verdict: PASS" in out, out
        reports = list((tmp_path / "reports").glob("aa_*/aa_report.json"))
        assert len(reports) == 1
        report = json.loads(reports[0].read_text(encoding="utf-8"))
        assert report["n_population"] == 3 and report["verdict"] == aa_report.VERDICT_PASS
    finally:
        instrumentation.uninstall()


def _cases_with_commits(specs):
    from app.experiments.replay import contract

    records = []
    for i, (commit, era_ts) in enumerate(specs, start=1):
        row = make_row(
            id=i, ticker=f"0000{i:02d}.KS", created_at=era_ts, git_commit=commit,
            transcript=make_transcript(),
        )
        records.append(contract.record_from_row(row))
    return records


def test_apply_era_and_drift_uses_the_two_sided_rule_and_reports_exclusions():
    from datetime import datetime, timezone

    freeze = datetime(2026, 9, 22, 1, 0, tzinfo=timezone.utc)
    old = datetime(2026, 9, 10, 1, 0, tzinfo=timezone.utc)
    records = _cases_with_commits([("82dac90", freeze), ("9a787bc", freeze), ("aaaaaaa", freeze), ("", freeze), ("82dac90", old)])
    verdicts = {
        "82dac90": {"vs_baseline": False, "vs_head": True, "drift": False},
        "9a787bc": {"vs_baseline": True, "vs_head": False, "drift": False},
        "aaaaaaa": {"vs_baseline": True, "vs_head": True, "drift": True},
    }
    calls = []

    def classify(commit):
        calls.append(commit)
        return verdicts[commit]

    kept, excluded, by_commit = cli.apply_era_and_drift(records, [{"memory_id": 99, "reason": "held_at_decision_true"}], "freeze", classify)
    assert [r["memory_id"] for r in kept] == [1, 2]  # only the two non-drifted freeze commits
    assert sorted(calls) == ["82dac90", "9a787bc", "aaaaaaa"]  # classified once per commit, not per case
    reasons = cli._count(e["reason"] for e in excluded)
    assert reasons == {"held_at_decision_true": 1, "era_freeze_drift": 2, "era_pre_freeze": 1}
    assert by_commit["aaaaaaa"]["drift"] is True and by_commit[""]["note"] == "no git_commit"
    assert {r["era"] for r in records} == {"freeze", "freeze_drift", "pre_freeze"}

    # without --era filter nothing is excluded here but drifted cases are re-labelled
    records = _cases_with_commits([("aaaaaaa", freeze), ("82dac90", freeze)])
    kept, excluded, _ = cli.apply_era_and_drift(records, [], None, classify)
    assert len(kept) == 2 and excluded == [] and sorted(r["era"] for r in kept) == ["freeze", "freeze_drift"]


def test_pilot_stats_count_a_nonzero_child_exit_as_an_error(tmp_path):
    run = tmp_path / "run"
    (run / "cases").mkdir(parents=True)
    ok = cli._pilot_stats(run, wall=60.0, foreign=[], returncode=0)
    paused = cli._pilot_stats(run, wall=60.0, foreign=[], returncode=75)
    aborted = cli._pilot_stats(run, wall=60.0, foreign=[], returncode=3)
    assert ok["errors"] == 0 and paused["errors"] == 1 and aborted["errors"] == 1
    assert cli.select_k_star({2: paused})["k_star"] is None


def test_aa_report_command_rejects_an_invalid_pair_with_exit_2(tmp_path, capsys):
    run = tmp_path / "r"
    (run / "cases").mkdir(parents=True)
    (run / "manifest.json").write_text(json.dumps({"run_id": "r", "repeat": 1, "status": "done", "variant": "baseline", "fork_at": "rm", "base_key": "k", "caseset": {"sha256": "s"}}), encoding="utf-8")
    assert cli.main(["aa-report", "--r1", str(run), "--r2", str(run)]) == cli.EXIT_CONFIG
    assert "not a valid A/A pair" in capsys.readouterr().err
