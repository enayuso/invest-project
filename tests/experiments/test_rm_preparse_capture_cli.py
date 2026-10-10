"""Synthetic command lifecycle tests. 준비/수집/검사와 안전 관문 합성 시험."""

from __future__ import annotations

import asyncio
import json
import os
import socket
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import app.experiments.rm_preparse_capture as cli
from app.experiments.rm_diagnostics.boundary import OfflineIOViolation, offline_io, response_snapshot
from app.experiments.rm_diagnostics.evidence import Evidence, Refusal, canonical, verify_bundle
from test_rm_preparse_capture import deny_external, private_root, request, sdk_response

_REAL_POPEN = subprocess.Popen  # captured before the autouse fixture blocks subprocess


def snapshot():
    return {"tables": {name: {"max_id": 1, "count": 1, "max_at": None} for name in
                       ("memory_entries", "trades", "pending_orders", "alerts")},
            "digests": {"cases": "a" * 32, "runtime_settings": None, "risk_rules": None}, "new_rows": {}}


@pytest.fixture
def synthetic_cli(monkeypatch):
    calls = []
    db_calls = []
    status = {"invariants": "PASS", "before_error": False, "after_error": False, "close_error": False,
              "original_raises": None, "request_changes": None, "after_create": None, "market_open": False}

    async def original(**kwargs):
        calls.append(kwargs)
        if status["original_raises"] is not None:
            raise status["original_raises"]
        return sdk_response('Based{"recommendation":"Hold","rationale":"The model has synthesized reports.","strategic_actions":"Stay out."}')

    async def close():
        if status["close_error"]:
            raise OSError("synthetic close")

    client = SimpleNamespace(_client=SimpleNamespace(close=close))
    inp = SimpleNamespace(memory_id=1141, ticker="035420.KS")
    provenance = {"source_hashes": {"synthetic": "source"}, "input_sha256": "input"}

    async def invoke(inp, client, boundary):
        from app.experiments.rm_diagnostics.boundary import DiagnosticBoundaryStop

        boundary.original = original
        try:
            await boundary.create(**{**request(), **((status["request_changes"] or {}) if boundary.transmit else {})})
        except DiagnosticBoundaryStop:
            if boundary.transmit and status["after_create"] is not None:
                await status["after_create"]()
            return

    async def connection():
        async def closed():
            pass
        return SimpleNamespace(close=closed)

    async def snap(conn, ids, *, since=None):
        db_calls.append((list(ids), since is not None))
        if status["after_error" if since else "before_error"]:
            raise OSError("synthetic DB failure")
        return snapshot()

    def compare(before, after, tickers, *, run_windows):
        assert run_windows and run_windows[0][1] >= run_windows[0][0]
        return SimpleNamespace(status=status["invariants"], as_dict=lambda: {
            "status": status["invariants"], "new_rows": after["new_rows"]})

    from app.experiments.replay import safety as real_safety

    def offhours(*a, **k):
        if status["market_open"]:
            raise real_safety.MarketWindowError("synthetic")

    safety = SimpleNamespace(assert_offhours=offhours, live_now_kst=lambda: None,
        MarketWindowError=real_safety.MarketWindowError,
        open_readonly_connection=connection, snapshot_invariants=snap, compare_invariants=compare,
        parse_windows=real_safety.parse_windows, trip_count=lambda: 0)
    monkeypatch.setattr(cli, "bootstrap", lambda args: (inp, client, provenance, safety))
    monkeypatch.setattr(cli, "invoke_rm", invoke)
    monkeypatch.setattr(cli, "source_hashes", lambda: provenance["source_hashes"])
    monkeypatch.setattr(cli, "assert_no_competing_work", lambda: None)
    monkeypatch.setattr(cli, "assert_server_idle", lambda url: None)
    return SimpleNamespace(calls=calls, db_calls=db_calls, status=status, provenance=provenance)


def arguments(root: Path, **changes):
    values = dict(output_root=str(root), prepared="", coordination_confirmed=True,
                  retention_confirmed=True, allow_market_hours=False)
    values.update(changes)
    return SimpleNamespace(**values)


@pytest.mark.asyncio
async def test_prepare_capture_inspect_lifecycle(private_root, synthetic_cli):
    code, prep = await cli.prepare(arguments(private_root))
    assert code == 0 and not synthetic_cli.calls and not synthetic_cli.db_calls
    prepared = private_root / prep["run_id"]
    code, capture = await cli.capture(arguments(private_root, prepared=str(prepared)))
    assert code == 0 and len(synthetic_cli.calls) == 1
    assert synthetic_cli.db_calls == [([1141], False), ([1141], True)]
    captured = private_root / capture["run_id"]
    manifest = verify_bundle(captured)
    assert manifest["capture_complete"]
    assert manifest["operator_preflight"] == {"coordination_confirmed": True,
        "retention_confirmed": True, "allow_market_hours": False}
    original_bytes = (captured / "response.json").read_bytes()
    code, inspection = cli.inspect(arguments(private_root, run=str(captured)))
    assert code == 0 and inspection["schema_valid"]
    assert (captured / "response.json").read_bytes() == original_bytes


@pytest.mark.asyncio
@pytest.mark.parametrize("invariant", ["WARN", "FAIL", "FAIL_CANDIDATE"])
async def test_db_nonpass_keeps_response_but_cannot_pass(private_root, synthetic_cli, invariant):
    _, prep = await cli.prepare(arguments(private_root))
    synthetic_cli.status["invariants"] = invariant
    code, captured = await cli.capture(arguments(private_root, prepared=str(private_root / prep["run_id"])))
    assert code == cli.EXIT_INCOMPLETE and len(synthetic_cli.calls) == 1
    run = private_root / captured["run_id"]
    assert (run / "response.json").exists()
    assert not verify_bundle(run)["capture_complete"]
    with pytest.raises(Refusal, match="capture_incomplete"):
        cli.inspect(arguments(private_root, run=str(run)))


@pytest.mark.asyncio
async def test_failed_initial_db_check_prevents_request(private_root, synthetic_cli):
    _, prep = await cli.prepare(arguments(private_root))
    synthetic_cli.status["before_error"] = True
    with pytest.raises(OSError):
        await cli.capture(arguments(private_root, prepared=str(private_root / prep["run_id"])))
    assert not synthetic_cli.calls


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["after_error", "close_error"])
async def test_failed_cleanup_cannot_pass(private_root, synthetic_cli, failure):
    _, prep = await cli.prepare(arguments(private_root))
    synthetic_cli.status[failure] = True
    code, captured = await cli.capture(arguments(private_root, prepared=str(private_root / prep["run_id"])))
    assert code == cli.EXIT_INCOMPLETE
    assert not verify_bundle(private_root / captured["run_id"])["capture_complete"]


@pytest.mark.asyncio
@pytest.mark.parametrize("gate", ["coordination_confirmed", "retention_confirmed"])
async def test_operator_gates_refuse_before_bootstrap(private_root, synthetic_cli, gate):
    with pytest.raises(Refusal, match="operator_preflight_required"):
        await cli.capture(arguments(private_root, **{gate: False}))
    assert not synthetic_cli.calls and not synthetic_cli.db_calls


@pytest.mark.asyncio
async def test_request_and_provenance_drift_stop_before_transmit(private_root, synthetic_cli):
    _, prep = await cli.prepare(arguments(private_root))
    synthetic_cli.provenance["input_sha256"] = "changed"
    with pytest.raises(Refusal, match="provenance_drift"):
        await cli.capture(arguments(private_root, prepared=str(private_root / prep["run_id"])))
    assert not synthetic_cli.calls and not synthetic_cli.db_calls


@pytest.mark.asyncio
async def test_request_kwargs_drift_stops_before_transmit(private_root, synthetic_cli):
    # CLI path: kwargs differ from the prepared request fingerprint (expected_sha mismatch).
    _, prep = await cli.prepare(arguments(private_root))
    synthetic_cli.status["request_changes"] = {"temperature": 0.7}
    with pytest.raises(Refusal, match="request_drift"):
        await cli.capture(arguments(private_root, prepared=str(private_root / prep["run_id"])))
    assert not synthetic_cli.calls
    assert not any((p / "manifest.json").exists() for p in private_root.iterdir()
                   if p.is_dir() and p.name != prep["run_id"])


def run_prepared(root):
    """Sync helper: prepare via asyncio.run, then build capture argv for main()."""
    _, prep = asyncio.run(cli.prepare(arguments(root)))
    argv = ["capture", "--output-root", str(root), "--pinned", "x", "--caseset-dir", "x",
            "--prepared", str(root / prep["run_id"]), "--coordination-confirmed", "--retention-confirmed"]
    return prep, argv


def no_complete_run(root, prep):
    for path in root.iterdir():
        if path.is_dir() and path.name != prep["run_id"] and (path / "manifest.json").exists():
            assert not verify_bundle(path)["capture_complete"]


def test_deadline_exceeded_is_incomplete_without_extra_sdk_call(private_root, synthetic_cli, monkeypatch):
    async def hang():
        await asyncio.sleep(10)

    synthetic_cli.status["after_create"] = hang
    real_wait_for = asyncio.wait_for
    monkeypatch.setattr(cli.asyncio, "wait_for",
                        lambda aw, timeout: real_wait_for(aw, 0.05 if timeout == 600 else timeout))
    _, prep = asyncio.run(cli.prepare(arguments(private_root)))
    code, captured = asyncio.run(cli.capture(arguments(private_root, prepared=str(private_root / prep["run_id"]))))
    assert code == cli.EXIT_INCOMPLETE and captured["reason"] == "deadline_exceeded"
    assert len(synthetic_cli.calls) == 1
    manifest = verify_bundle(private_root / captured["run_id"])
    assert manifest["capture_complete"] is False and manifest["status"] == "incomplete"


@pytest.mark.asyncio
async def test_keyboard_interrupt_during_capture_never_completes(private_root, synthetic_cli):
    synthetic_cli.status["original_raises"] = KeyboardInterrupt()
    _, prep = await cli.prepare(arguments(private_root))
    with pytest.raises(KeyboardInterrupt):
        await cli.capture(arguments(private_root, prepared=str(private_root / prep["run_id"])))
    assert len(synthetic_cli.calls) == 1
    # no manifest is ever written for the interrupted run
    assert not any((p / "manifest.json").exists() for p in private_root.iterdir()
                   if p.is_dir() and p.name != prep["run_id"])


def test_replay_safety_violation_aborts_without_completion(private_root, synthetic_cli, capsys):
    from app.experiments.replay import safety as real_safety

    synthetic_cli.status["original_raises"] = real_safety.ReplaySafetyViolation()
    prep, argv = run_prepared(private_root)
    capsys.readouterr()
    assert cli.main(argv) == cli.EXIT_SAFETY
    assert json.loads(capsys.readouterr().out) == {"status": "aborted_safety", "reason": "blocked_side_effect"}
    assert len(synthetic_cli.calls) == 1
    no_complete_run(private_root, prep)


@pytest.mark.parametrize("failure", ["rejected_sensitive", "event_limit", "write_failed"])
def test_post_transmit_invariants_write_failure_is_incomplete_not_refused(
        private_root, synthetic_cli, monkeypatch, capsys, failure):
    write = Evidence.write

    def broken(self, name, value):
        if name == "invariants.json":
            raise Refusal(failure)
        return write(self, name, value)

    prep, argv = run_prepared(private_root)
    monkeypatch.setattr(Evidence, "write", broken)
    capsys.readouterr()
    assert cli.main(argv) == cli.EXIT_INCOMPLETE
    assert json.loads(capsys.readouterr().out) == {"status": "incomplete", "transmitted": True, "reason": failure}
    assert len(synthetic_cli.calls) == 1
    no_complete_run(private_root, prep)


def test_rm_not_stopping_after_transmit_is_incomplete(private_root, synthetic_cli, capsys):
    async def not_stopped():
        raise Refusal("rm_did_not_stop_at_boundary")

    synthetic_cli.status["after_create"] = not_stopped
    prep, argv = run_prepared(private_root)
    capsys.readouterr()
    assert cli.main(argv) == cli.EXIT_INCOMPLETE
    assert json.loads(capsys.readouterr().out) == {"status": "incomplete", "transmitted": True,
                                                    "reason": "rm_did_not_stop_at_boundary"}
    no_complete_run(private_root, prep)


def test_market_hours_refusal_is_pre_transmit_refused(private_root, synthetic_cli, capsys):
    synthetic_cli.status["market_open"] = True
    prep, argv = run_prepared(private_root)
    capsys.readouterr()
    assert cli.main(argv) == cli.EXIT_REFUSED
    assert json.loads(capsys.readouterr().out) == {"status": "refused", "reason": "market_window_open"}
    assert not synthetic_cli.calls and not synthetic_cli.db_calls
    assert [p.name for p in private_root.iterdir() if p.is_dir()] == [prep["run_id"]]


def test_pre_transmit_failure_stays_diagnostic_failed_not_transmitted(private_root, synthetic_cli, capsys):
    synthetic_cli.status["before_error"] = True
    prep, argv = run_prepared(private_root)
    capsys.readouterr()
    assert cli.main(argv) == cli.EXIT_INCOMPLETE
    assert json.loads(capsys.readouterr().out) == {"status": "incomplete", "reason": "diagnostic_failed"}
    assert not synthetic_cli.calls


# ---- competing-work gate (pure logic + real Windows process query) ----

def proc(pid, ppid, cmd, name="python.exe"):
    return {"pid": pid, "ppid": ppid, "name": name, "cmd": cmd}


def test_competing_processes_excludes_self_ancestors_and_checker_by_pid():
    capture_cmd = "python -m app.experiments.rm_preparse_capture capture --x"
    procs = [proc(1, 0, "init", "init"),
             proc(10, 1, capture_cmd),                       # venv launcher (ancestor)
             proc(11, 10, capture_cmd),                      # this process
             proc(12, 11, "powershell screening_sweep", "powershell.exe"),  # the checker
             proc(20, 1, "python c01_rerun_at_1541.py"),     # real competitor
             proc(21, 1, "python -m app.experiments.rm_preparse_capture capture"),  # other capture
             proc(22, 1, "node rm_preparse_capture capture", "node.exe"),  # not python: ignored
             proc(23, 1, None), proc(24, 1, "python replay_downstream pilot --a")]
    found = cli.competing_processes(procs, 11, exclude_pids=(12,))
    assert sorted(p["pid"] for p in found) == [20, 21, 24]
    # without the PID exclusions the launcher/checker would be false positives
    assert sorted(p["pid"] for p in cli.competing_processes(procs, 99)) == [10, 11, 12, 20, 21, 24]
    # an explicit parents mapping with a cycle terminates
    cyc = cli.competing_processes(procs, 11, parents={11: 10, 10: 11}, exclude_pids=(12,))
    assert [p["pid"] for p in cyc] == [20, 21, 24]


def test_competing_processes_posix_style_without_names():
    procs = [{"pid": 5, "ppid": 1, "name": None, "cmd": "python -m x.rm_preparse_capture capture"},
             {"pid": 6, "ppid": 1, "name": None, "cmd": "bash screening_sweep"}]
    assert [p["pid"] for p in cli.competing_processes(procs, 1)] == [5, 6]


@pytest.mark.parametrize("failure", ["oserror", "badjson", "notlisted"])
def test_process_check_fails_closed(monkeypatch, failure):
    def listed():
        if failure == "oserror":
            raise OSError("x")
        if failure == "badjson":
            raise json.JSONDecodeError("x", "", 0)
        return [{"pid": 1, "ppid": 0, "name": "a", "cmd": "b"}], 2

    monkeypatch.setattr(cli, "list_processes", listed)
    with pytest.raises(Refusal, match="process_check_unavailable"):
        cli.assert_no_competing_work()


@pytest.mark.skipif(os.name != "nt", reason="Windows PowerShell process query")
def test_real_windows_query_does_not_flag_self_parents_or_checker(monkeypatch):
    monkeypatch.setattr(subprocess, "Popen", _REAL_POPEN)
    processes, checker = cli.list_processes()
    by_pid = {p["pid"]: p for p in processes}
    assert os.getpid() in by_pid and checker in by_pid
    flagged = {p["pid"] for p in cli.competing_processes(processes, os.getpid(), exclude_pids=(checker,))}
    ancestors, node = set(), os.getpid()
    while node in by_pid and by_pid[node]["ppid"] not in ancestors | {0}:
        node = by_pid[node]["ppid"]
        ancestors.add(node)
    assert not ({os.getpid(), checker} | ancestors) & flagged
    # the checker's own command line carries no pattern, so it is safe even if not excluded
    assert not cli._ANY_WORK.search(by_pid[checker]["cmd"] or "")
    assert not cli._CAPTURE_WORK.search(by_pid[checker]["cmd"] or "")


@pytest.mark.skipif(os.name != "nt", reason="Windows PowerShell process query")
def test_real_windows_query_flags_a_real_competing_process(monkeypatch):
    monkeypatch.setattr(subprocess, "Popen", _REAL_POPEN)
    child = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(30)  # screening_sweep"])
    try:
        processes, checker = cli.list_processes()
        flagged = {p["pid"] for p in cli.competing_processes(processes, os.getpid(), exclude_pids=(checker,))}
        assert child.pid in flagged
        with pytest.raises(Refusal, match="competing_work"):
            cli.assert_no_competing_work()
    finally:
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(child.pid)], capture_output=True)
        child.wait(timeout=10)


def test_required_invariant_case_digest_and_tables():
    cli.validate_snapshot(snapshot())
    for mutate in (lambda x: x["digests"].update(cases=None), lambda x: x["tables"].pop("trades")):
        item = snapshot()
        mutate(item)
        with pytest.raises(Refusal, match="invariant_snapshot_incomplete"):
            cli.validate_snapshot(item)


@pytest.mark.parametrize("metrics", ["", "vllm:num_requests_running 0", "vllm:num_requests_running NaN\nvllm:num_requests_waiting 0",
    "vllm:num_requests_running 1\nvllm:num_requests_waiting 0"])
def test_idle_gate_requires_both_present_finite_zero(monkeypatch, metrics):
    from app.experiments import replay_downstream

    monkeypatch.setattr(replay_downstream, "http_get_text", lambda url: metrics)
    with pytest.raises(Refusal, match="server_not_idle"):
        cli.assert_server_idle("http://synthetic.invalid/v1")


def test_idle_gate_accepts_all_zero_samples(monkeypatch):
    from app.experiments import replay_downstream

    monkeypatch.setattr(replay_downstream, "http_get_text", lambda url:
        'vllm:num_requests_running{model_name="synthetic"} 0\nvllm:num_requests_waiting 0')
    cli.assert_server_idle("http://synthetic.invalid/v1")


def test_offline_io_is_not_swallowed_as_normal_boundary():
    from app.experiments.rm_diagnostics.boundary import DiagnosticBoundaryStop

    assert not issubclass(OfflineIOViolation, DiagnosticBoundaryStop)
    original = socket.getaddrinfo
    with offline_io(), pytest.raises(OfflineIOViolation):
        socket.getaddrinfo("synthetic.invalid", 80)
    assert socket.getaddrinfo is original


def test_cli_failure_does_not_print_untrusted_exception(monkeypatch, capsys):
    async def failed(args):
        raise ValueError("bearer should_never_appear")

    monkeypatch.setattr(cli, "prepare", failed)
    assert cli.main(["prepare", "--output-root", "synthetic", "--pinned", "synthetic",
                     "--caseset-dir", "synthetic"]) == cli.EXIT_INCOMPLETE
    assert "should_never_appear" not in capsys.readouterr().out


def test_capture_cli_confirmation_defaults_to_off():
    args = cli.parser().parse_args(["capture", "--prepared", "synthetic", "--output-root", "synthetic",
                                   "--pinned", "synthetic", "--caseset-dir", "synthetic"])
    assert not args.coordination_confirmed and not args.retention_confirmed and not args.allow_market_hours


def test_real_source_pins_and_local_git_head():
    assert len(cli.source_hashes()) >= 25
    assert len(cli.git_head()) == 40


def test_real_safety_blocks_cover_side_effects(monkeypatch):
    from app.experiments.replay import safety
    from app.memory import store
    from app.orchestrator import pipeline
    from app.execution import ledger

    safety.reset_trips()
    report = safety.install_blocks(setattr_fn=monkeypatch.setattr)
    assert len(report.installed) == len(safety.BLOCK_TARGETS) and not report.missing
    for blocked in (store.store_pending, pipeline.store_pending, ledger.record_trade):
        with pytest.raises(safety.ReplaySafetyViolation):
            blocked()
    assert safety.trip_count() == 3
    safety.reset_trips()


def test_production_source_drift_is_refused(monkeypatch):
    original = Path.read_bytes

    def changed(path):
        if path.as_posix().endswith("app/agents/research_manager.py"):
            return original(path) + b"\n# synthetic source change\n"
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", changed)
    with pytest.raises(Refusal, match="production_source_drift"):
        cli.source_hashes()
