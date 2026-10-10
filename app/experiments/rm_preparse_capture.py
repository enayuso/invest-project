"""RM first-native evidence CLI. 독립 진단: prepare/capture/inspect, 자동 재시도 없음."""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import math
import os
import re
import subprocess
import sys
from datetime import timezone
from importlib.metadata import version
from pathlib import Path
from urllib.parse import urlsplit

from app.experiments.replay import pinned
from app.experiments.rm_diagnostics.evidence import (
    Evidence, Refusal, canonical, digest, read_json, verify_bundle,
)
from app.experiments.rm_diagnostics.boundary import (
    Boundary, OfflineIOViolation, inspect_response, invoke_rm, offline_io, quiet_logs,
)

REPO = Path(__file__).resolve().parents[2]
CASE_ID = "035420.KS_20260901_m1141"
EXIT_REFUSED, EXIT_SAFETY, EXIT_INCOMPLETE = 2, 3, 76


def timestamps() -> dict[str, str]:
    from app.clock import KR_TZ, LiveClock

    now = LiveClock().now(timezone.utc)
    return {"utc": now.isoformat(), "kst": now.astimezone(KR_TZ).isoformat()}


def source_hashes() -> dict[str, str]:
    from app.experiments.replay.provenance import PINNED_SOURCE_HASHES

    paths = {key[5:] for key in PINNED_SOURCE_HASHES if key.startswith("file:")}
    paths |= {"app/config.py", "app/clock.py", "app/experiments/rm_preparse_capture.py",
              "app/experiments/replay/contract.py", "app/experiments/replay/pinned.py",
              "app/experiments/replay/safety.py", "app/experiments/replay/provenance.py"}
    paths |= {p.relative_to(REPO).as_posix() for p in (REPO / "app/experiments/rm_diagnostics").glob("*.py")}
    result = {}
    for rel in sorted(paths):
        data = (REPO / rel).read_bytes()
        expected = PINNED_SOURCE_HASHES.get("file:" + rel)
        if expected and digest(data.replace(b"\r\n", b"\n")) != expected:
            raise Refusal("production_source_drift")
        result[rel] = digest(data)
    return result


def git_head() -> str:
    # Read refs locally; prepare needs no git subprocess. 로컬 ref만 읽는다.
    directory = REPO / ".git"
    if directory.is_file():
        directory = (REPO / directory.read_text().strip().split(": ", 1)[1]).resolve()
    head = (directory / "HEAD").read_text().strip()
    if head.startswith("ref: "):
        ref = head[5:]
        common = directory
        if (directory / "commondir").exists():
            common = (directory / (directory / "commondir").read_text().strip()).resolve()
        if (common / ref).exists():
            head = (common / ref).read_text().strip()
        else:
            matches = [line.split()[0] for line in (common / "packed-refs").read_text().splitlines()
                       if line.endswith(" " + ref)]
            if len(matches) != 1:
                raise Refusal("git_head_unavailable")
            head = matches[0]
    if not re.fullmatch(r"[a-f0-9]{40}", head):
        raise Refusal("git_head_unavailable")
    return head


def bootstrap(args):
    if "app.config" in sys.modules:
        raise Refusal("config_already_imported")
    p = pinned.load(args.pinned)
    pinned.apply_to_environ(p, k=1)
    from app.config import Settings, settings
    from app.experiments.replay import contract, safety
    from app.llm import llm_client

    if (pinned.check_code_defaults(p, Settings.model_fields) or
            pinned.verify_against_settings(p, settings) or safety.verify_process_state(1)):
        raise Refusal("settings_drift")
    if settings.llm_provider != "vllm" or settings.rm_request_capture_enabled is not False:
        raise Refusal("unsupported_provider_or_capture_on")
    url = urlsplit(settings.llm_base_url)
    if url.username or url.password or url.query or url.fragment:
        raise Refusal("unsafe_endpoint")
    records, meta = contract.load_caseset(Path(args.caseset_dir), args.caseset)
    chosen = [r for r in records if r.get("case_id") == args.case_id]
    if len(chosen) != 1 or chosen[0]["columns"].get("llm_model") != p.values["LLM_MODEL"]:
        raise Refusal("case_selection_invalid")
    safety.reset_trips()
    safety.set_trips_path(None)
    report = safety.install_blocks()
    if report.missing or len(report.installed) != len(safety.BLOCK_TARGETS):
        raise Refusal("safety_blocks_incomplete")
    inp = contract.session_prepass([contract.build_inputs(chosen[0])])[0]
    provenance = {"case_id": inp.case_id, "memory_id": inp.memory_id, "ticker": inp.ticker,
        "input_contract": contract.INPUT_CONTRACT_VERSION, "input_sha256": inp.input_hash(),
        "caseset_sha256": meta["jsonl_sha256"],
        "caseset_meta_sha256": digest(canonical(meta)), "pinned_sha256": p.sha256,
        "pinned_values": p.keyed_values(), "source_hashes": source_hashes(), "git_head": git_head(),
        "python": sys.version.split()[0], "openai": version("openai"), "pydantic": version("pydantic"),
        "request_fidelity": "client_boundary_request", "response_fidelity": "sdk_decoded_response",
        "endpoint_alias": "vllm", "endpoint_sha256": digest(settings.llm_base_url.encode()),
        "sdk_max_retries": llm_client._client.max_retries,
        "sdk_timeout": settings.llm_timeout_seconds, "sdk_retry_visibility": "not_observed",
        "llm_deadline_seconds": 600, "safety_blocks": len(report.installed),
        "safety_aliases": len(report.aliases)}
    return inp, llm_client, provenance, safety


_ANY_WORK = re.compile(r"c01_rerun_at_1541|screening_sweep|replay_downstream\s+(run|pilot)")
_CAPTURE_WORK = re.compile(r"rm_preparse_capture\s+capture")
# Patterns live only in Python; the PowerShell script carries none, so the checker cannot match itself.
# 패턴은 파이썬에만 둔다. 조회용 PowerShell 명령줄에는 패턴이 없다.
_WINDOWS_LIST = ("[Console]::OutputEncoding=[Text.Encoding]::UTF8; "
                 "Get-CimInstance Win32_Process -ErrorAction Stop | "
                 "Select-Object @{n='pid';e={[int]$_.ProcessId}},@{n='ppid';e={[int]$_.ParentProcessId}},"
                 "@{n='name';e={$_.Name}},@{n='cmd';e={$_.CommandLine}} | ConvertTo-Json -Compress")


def competing_processes(processes: list[dict], self_pid: int, parents: dict | None = None,
                        exclude_pids=()) -> list[dict]:
    """Pure: processes (pid/ppid/name/cmd) -> competing ones. 순수 판정, PID 기반 제외.

    Excluded: this process, its launcher ancestors (parents chain), and exclude_pids (the checker).
    name=None skips the python-name condition (ps output has no separate name).
    """
    parents = parents if parents is not None else {p["pid"]: p["ppid"] for p in processes}
    excluded = {self_pid, *exclude_pids}
    node = self_pid
    while node in parents and parents[node] not in excluded and parents[node] != 0:
        node = parents[node]
        excluded.add(node)
    result = []
    for item in processes:
        cmd = item.get("cmd")
        if item["pid"] in excluded or not cmd:
            continue
        name = item.get("name")
        if _ANY_WORK.search(cmd) or (_CAPTURE_WORK.search(cmd) and (
                name is None or re.match(r"python", name, re.IGNORECASE))):
            result.append(item)
    return result


def list_processes() -> tuple[list[dict], int]:
    """Process snapshot and the query child's PID; raises on any failure. 조회 실패는 예외."""
    if os.name == "nt":
        command = ["powershell", "-NoProfile", "-NonInteractive", "-Command", _WINDOWS_LIST]
    else:
        command = ["ps", "-eo", "pid=,ppid=,args="]
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                            text=True, encoding="utf-8")
    try:
        out, _ = proc.communicate(timeout=30)
    except BaseException:
        proc.kill()
        proc.communicate()
        raise
    if proc.returncode != 0:
        raise ValueError("process_query_failed")
    if os.name == "nt":
        rows = json.loads(out)
        rows = [rows] if isinstance(rows, dict) else rows
        processes = [{"pid": int(r["pid"]), "ppid": int(r["ppid"]), "name": r.get("name") or "",
                      "cmd": r.get("cmd")} for r in rows]
    else:
        processes = []
        for line in out.splitlines():
            parts = line.split(None, 2)
            if len(parts) >= 2:
                processes.append({"pid": int(parts[0]), "ppid": int(parts[1]), "name": None,
                                  "cmd": parts[2] if len(parts) > 2 else ""})
    if not processes:
        raise ValueError("process_list_empty")
    return processes, proc.pid


def assert_no_competing_work() -> None:
    # Return only a verdict; process command lines never enter logs/evidence. Fail-closed on any query/parse failure.
    # 프로세스 명령줄은 보존하지 않는다. 조회/파싱 실패는 거부(fail-closed).
    try:
        processes, checker = list_processes()
        if os.getpid() not in {p["pid"] for p in processes}:
            raise ValueError("self_not_listed")
        competing = competing_processes(processes, os.getpid(), exclude_pids=(checker,))
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError):
        raise Refusal("process_check_unavailable") from None
    if competing:
        raise Refusal("competing_work")


def assert_server_idle(base_url: str) -> None:
    from app.experiments.replay_downstream import http_get_text

    names = ("vllm:num_requests_running", "vllm:num_requests_waiting")
    endpoint = base_url.removesuffix("/").removesuffix("/v1") + "/metrics"
    try:
        values = {name: [] for name in names}
        for line in http_get_text(endpoint).splitlines():
            for name in names:
                if re.match(re.escape(name) + r"(?:\{|\s)", line):
                    values[name].append(float(line.rsplit(None, 1)[1]))
        if any(not samples or any(not math.isfinite(v) or v != 0 for v in samples)
               for samples in values.values()):
            raise Refusal("server_not_idle")
    except Refusal:
        raise
    except Exception:
        raise Refusal("server_idle_unconfirmed") from None


def new_evidence(root: str) -> Evidence:
    return Evidence.create(Path(root), forbidden=(REPO, REPO.parent / "handoff"))


async def close_client(client) -> bool:
    try:
        await asyncio.wait_for(client._client.close(), timeout=10)
        return True
    except Exception:
        return False


def validate_snapshot(snapshot: dict) -> None:
    if (any(snapshot.get("tables", {}).get(table) is None
            for table in ("memory_entries", "trades", "pending_orders", "alerts")) or
            not re.fullmatch(r"[a-f0-9]{32}", snapshot.get("digests", {}).get("cases") or "") or
            not {"cases", "runtime_settings", "risk_rules"} <= set(snapshot.get("digests", {}))):
        raise Refusal("invariant_snapshot_incomplete")


async def prepare(args) -> tuple[int, dict]:
    inp, client, prov, safety = bootstrap(args)
    try:
        evidence = new_evidence(args.output_root)
        evidence.write("provenance.json", prov)
        boundary = Boundary(evidence)
        await invoke_rm(inp, client, boundary)
        if boundary.reason != "prepared" or safety.trip_count():
            raise Refusal(boundary.reason)
        if not await close_client(client):
            raise Refusal("cleanup_failed")
        manifest = evidence.finish(phase="prepare", status="prepared", capture_complete=False,
            sdk_calls_started=0, sdk_calls_completed=0, safety_trips=0, timestamps=timestamps())
        return 0, {"status": manifest["status"], "run_id": manifest["run_id"],
                   "request_sha256": manifest["files"]["request.json"]["canonical_sha256"]}
    finally:
        await close_client(client)


class PostTransmitFailure(Exception):
    """Failure after the SDK callable was entered; carries only a reason code. 전송 후 실패(사유 코드만)."""


async def capture(args) -> tuple[int, dict]:
    state: dict = {}
    try:
        return await _capture(args, state)
    except Exception as error:
        boundary = state.get("boundary")
        if boundary is not None and boundary.started:
            # SDK callable was entered: never present this as a pre-transmit refusal.
            raise PostTransmitFailure(str(error) if isinstance(error, Refusal) else "diagnostic_failed") from None
        raise


async def _capture(args, state: dict) -> tuple[int, dict]:
    if not args.coordination_confirmed or not args.retention_confirmed:
        raise Refusal("operator_preflight_required")
    inp, client, prov, safety = bootstrap(args)
    try:
        prepared = Path(args.prepared)
        previous = verify_bundle(prepared)
        if previous.get("phase") != "prepare" or previous.get("status") != "prepared":
            raise Refusal("not_prepared")
        if read_json(prepared / "provenance.json") != prov:
            raise Refusal("provenance_drift")
        try:
            safety.assert_offhours(safety.live_now_kst(), allow_market_hours=args.allow_market_hours)
        except safety.MarketWindowError:
            raise Refusal("market_window_open") from None  # raised before any transmission
        assert_no_competing_work()
        from app.config import settings

        assert_server_idle(settings.llm_base_url)
        evidence = new_evidence(args.output_root)
        evidence.write("provenance.json", prov)
        boundary = Boundary(evidence, transmit=True,
            expected_sha=previous["files"]["request.json"]["canonical_sha256"])
        state["boundary"] = boundary
        before = after = None
        result = None
        cleanup_ok = True
        window_start = timestamps()["utc"]
        try:
            conn = await asyncio.wait_for(safety.open_readonly_connection(), 45)
            try:
                before = await asyncio.wait_for(safety.snapshot_invariants(conn, [inp.memory_id]), 120)
                validate_snapshot(before)
            finally:
                await asyncio.wait_for(conn.close(), 10)
            try:
                await asyncio.wait_for(invoke_rm(inp, client, boundary), timeout=600)
            except TimeoutError:
                boundary.reason = "deadline_exceeded"
        finally:
            window_end = timestamps()["utc"]
            if before is not None:
                try:
                    conn = await asyncio.wait_for(safety.open_readonly_connection(), 45)
                    try:
                        after = await asyncio.wait_for(
                            safety.snapshot_invariants(conn, [inp.memory_id], since=before), 120)
                        validate_snapshot(after)
                    finally:
                        await asyncio.wait_for(conn.close(), 10)
                    windows = safety.parse_windows([{"start": window_start, "end": window_end}])
                    result = safety.compare_invariants(before, after, {inp.ticker}, run_windows=windows)
                except Exception:
                    cleanup_ok = False
            cleanup_ok = await close_client(client) and cleanup_ok
            invariant_status = result.status if result else "UNKNOWN"
            evidence.write("invariants.json", {"case_ids": [inp.memory_id], "before": before, "after": after,
                "status": invariant_status, "comparison": copy.deepcopy(result.as_dict()) if result else None,
                "safety_trips": safety.trip_count(), "cleanup_ok": cleanup_ok})
        if not boundary.started:
            # Rejected before the SDK callable: nothing was transmitted, so this is a refusal.
            raise Refusal(boundary.reason)
        complete = (boundary.reason == "response_saved" and boundary.entries == 1 and
                    invariant_status == "PASS" and not safety.trip_count() and cleanup_ok and
                    source_hashes() == prov["source_hashes"])
        manifest = evidence.finish(phase="capture", status="complete" if complete else "incomplete",
            capture_complete=complete, reason=boundary.reason, sdk_calls_started=boundary.started,
            sdk_calls_completed=boundary.completed, invariants_status=invariant_status,
            safety_trips=safety.trip_count(), timestamps=timestamps(),
            run_window={"start": window_start, "end": window_end},
            operator_preflight={"coordination_confirmed": args.coordination_confirmed,
                "retention_confirmed": args.retention_confirmed,
                "allow_market_hours": args.allow_market_hours},
            prepared_manifest_sha256=digest(canonical(previous)))
        return (0 if complete else EXIT_INCOMPLETE), {"status": manifest["status"],
            "run_id": manifest["run_id"], "reason": boundary.reason,
            "transmitted": bool(boundary.started), "invariants_status": invariant_status}
    finally:
        await close_client(client)


def inspect(args) -> tuple[int, dict]:
    original = Path(args.run)
    manifest = verify_bundle(original)
    if manifest.get("phase") != "capture" or not manifest.get("capture_complete"):
        raise Refusal("capture_incomplete")
    prov = read_json(original / "provenance.json")
    if source_hashes() != prov["source_hashes"]:
        raise Refusal("inspection_source_drift")
    response = read_json(original / "response.json")
    if digest(canonical(response["raw_response"])) != response["raw_response_canonical_sha256"]:
        raise Refusal("response_hash_mismatch")
    inspection = inspect_response(response)
    evidence = new_evidence(args.output_root)
    evidence.write("provenance.json", {"source_hashes": prov["source_hashes"],
        "parent_run_id": original.name, "parent_manifest_sha256": digest(canonical(manifest))})
    evidence.write("inspection.json", inspection)
    evidence.finish(phase="inspect", status="inspected", capture_complete=False,
                    schema_valid=inspection["schema_valid"], timestamps=timestamps())
    return 0, {"status": "inspected", "run_id": evidence.directory.name,
               "schema_valid": inspection["schema_valid"]}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    for command in ("prepare", "capture", "inspect"):
        p = commands.add_parser(command)
        p.add_argument("--output-root", required=True, help="existing local private directory")
        if command in ("prepare", "capture"):
            p.add_argument("--pinned", required=True)
            p.add_argument("--caseset-dir", required=True)
            p.add_argument("--caseset", default="pilot12")
            p.add_argument("--case-id", default=CASE_ID)
        if command == "capture":
            p.add_argument("--prepared", required=True)
            p.add_argument("--coordination-confirmed", action="store_true",
                help="operator checked CURRENT, c01 completion and no other experiments")
            p.add_argument("--retention-confirmed", action="store_true",
                help="operator approved private location, access and retention period")
            p.add_argument("--allow-market-hours", action="store_true",
                help="requires explicit approval for this run")
        if command == "inspect":
            p.add_argument("--run", required=True)
    return result


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    with quiet_logs():
        try:
            if args.command == "prepare":
                with offline_io():
                    code, summary = asyncio.run(prepare(args))
            elif args.command == "capture":
                code, summary = asyncio.run(capture(args))
            else:
                with offline_io():
                    code, summary = inspect(args)
            print(json.dumps(summary, ensure_ascii=False))
            return code
        except PostTransmitFailure as error:
            print(json.dumps({"status": "incomplete", "transmitted": True, "reason": str(error)}))
            return EXIT_INCOMPLETE
        except Refusal as error:
            print(json.dumps({"status": "refused", "reason": str(error)}))
            return EXIT_REFUSED
        except Exception:
            print('{"status":"incomplete","reason":"diagnostic_failed"}')
            return EXIT_INCOMPLETE
        except BaseException as error:
            if isinstance(error, OfflineIOViolation):
                print('{"status":"aborted_safety","reason":"offline_io_attempt"}')
                return EXIT_SAFETY
            if type(error).__name__ == "ReplaySafetyViolation":
                print('{"status":"aborted_safety","reason":"blocked_side_effect"}')
                return EXIT_SAFETY
            raise


if __name__ == "__main__":
    raise SystemExit(main())
