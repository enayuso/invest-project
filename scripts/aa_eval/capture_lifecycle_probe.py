"""Linux-only synthetic ASGI lifespan SIGTERM/SIGKILL probe; no live app."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.util
import json
import os
import re
import signal
import subprocess
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace


def _recorder(repo: Path, directory: Path):
    config = ModuleType("app.config")
    config.settings = SimpleNamespace(
        rm_request_capture_enabled=True, rm_request_capture_directory=str(directory),
        rm_request_capture_event_max_bytes=2 * 1024 * 1024,
        rm_request_capture_queue_max_bytes=64 * 1024 * 1024,
        rm_request_capture_session_max_bytes=512 * 1024 * 1024)
    sys.modules["app.config"] = config
    spec = importlib.util.spec_from_file_location("synthetic_lifecycle_recorder", repo / "app/llm/request_capture.py")
    recorder = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = recorder
    spec.loader.exec_module(recorder)
    return recorder


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validator(repo: Path):
    spec = importlib.util.spec_from_file_location("synthetic_lifecycle_validator", repo / "scripts/aa_eval/validate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


async def _child(directory: Path, mode: str):
    from fastapi import FastAPI
    import uvicorn

    recorder = _recorder(Path(__file__).resolve().parents[2], directory)
    ready_info: dict[str, str | int] = {}

    @asynccontextmanager
    async def lifespan(_app):
        with recorder.capture_research_manager(
                ticker="005930.KS", trade_date="2026-09-23", market="domestic",
                trigger="auto", held=False, session_in_progress=True, historical=False):
            capture_id = recorder.observe_first_native_request(
                {"model": "synthetic", "messages": [{"role": "user", "content": "synthetic"}],
                 "temperature": 0, "extra_body": {}, "max_tokens": 32,
                 "response_format": {"type": "json_schema", "json_schema": {
                     "name": "Synthetic", "schema": {"type": "object"}, "strict": True}}},
                label="research_manager", first=True, provider="vllm",
                sdk_max_retries=0, timeout_seconds=300)
        if capture_id is None:
            raise RuntimeError("synthetic request was not accepted")
        writer = recorder._writer
        writer.events.join()
        writer.control.join()
        request_path = directory / f"{capture_id}.request.json"
        ready_info.update({"session_id": writer.session_id, "capture_id": capture_id,
                           "request_sha256": _sha(request_path), "pending_captures": len(writer.pending_captures)})
        try:
            yield
        finally:
            if mode == "term":
                # Hold status I/O until finalization sees queued data and control.
                # 종료 기록기가 데이터·제어 대기를 관측할 때까지 상태 I/O를 잠시 잡는다.
                writer.status_lock.acquire()
                try:
                    recorder.observe_first_native_result(capture_id,
                        response={"choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}]},
                        elapsed_seconds=.001)
                    recorder.observe_first_native_parse(capture_id)
                    writer._wake_control()
                    pending_data = writer.events.unfinished_tasks
                    pending_control = writer.control.unfinished_tasks
                    future = asyncio.create_task(asyncio.to_thread(recorder.finalize_capture))
                    deadline = time.monotonic() + 3
                    while not writer.closing and time.monotonic() < deadline:
                        await asyncio.sleep(.005)
                    if not writer.closing:
                        raise RuntimeError("synthetic finalizer did not start")
                finally:
                    writer.status_lock.release()
                finalized = await future
                (directory / f"shutdown-{mode}.json").write_text(json.dumps({
                    "finalized": finalized, "pending_data": pending_data,
                    "pending_control": pending_control}), encoding="utf-8")

    app = FastAPI(lifespan=lifespan)
    config = uvicorn.Config(app, uds=str(directory / f"synthetic-{mode}.sock"),
                            lifespan="on", log_level="error")
    server = uvicorn.Server(config)

    async def announce_ready():
        while not server.started and not server.should_exit:
            await asyncio.sleep(.01)
        if server.started:
            (directory / f"ready-{mode}.json").write_text(json.dumps(ready_info), encoding="utf-8")

    watcher = asyncio.create_task(announce_ready())
    try:
        await server.serve()
    finally:
        watcher.cancel()


def _parent(directory: Path, timeout: float) -> dict:
    repo = Path(__file__).resolve().parents[2]
    if directory.exists() or repo in directory.parents:
        raise ValueError("probe directory must be new and outside the repository")
    directory.mkdir(parents=True)
    validator = _validator(repo)
    results = {}
    earlier_hashes: dict[str, str] = {}
    for mode, stop_signal in (("term", signal.SIGTERM), ("kill", signal.SIGKILL)):
        started = time.monotonic()
        child = subprocess.Popen([sys.executable, "-B", str(Path(__file__).resolve()),
                                  "_child", str(directory), mode],
                                 stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL,
                                 env={"PATH": os.environ.get("PATH", ""),
                                      "PYTHONNOUSERSITE": "1"})
        try:
            marker = directory / f"ready-{mode}.json"
            while not marker.exists():
                if child.poll() is not None or time.monotonic() - started > timeout:
                    raise RuntimeError(f"synthetic child did not become ready: {mode}")
                time.sleep(.05)
            ready = json.loads(marker.read_text(encoding="utf-8"))
            session_id = ready.get("session_id")
            capture_id = ready.get("capture_id")
            if (not isinstance(session_id, str) or not re.fullmatch(r"[0-9a-f]{32}", session_id)
                    or not isinstance(capture_id, str) or not re.fullmatch(r"[0-9a-f]{32}", capture_id)
                    or ready.get("pending_captures") != 1):
                raise RuntimeError("synthetic ready metadata invalid")
            request_path = directory / f"{capture_id}.request.json"
            if not request_path.is_file() or _sha(request_path) != ready.get("request_sha256"):
                raise RuntimeError("synthetic request evidence missing")
            os.kill(child.pid, stop_signal)
            child.wait(timeout=timeout)
        finally:
            if child.poll() is None:
                child.kill()
                child.wait(timeout=timeout)
        if child.returncode not in ((0, -signal.SIGTERM) if mode == "term" else (-signal.SIGKILL,)):
            raise RuntimeError(f"synthetic child exited unexpectedly: {mode}")
        status_paths = list(directory.glob(".capture-session-*.json"))
        if len(status_paths) != len(results) + 1 or len({path.name for path in status_paths}) != len(status_paths):
            raise RuntimeError("unexpected synthetic session count")
        status_path = directory / f".capture-session-{session_id}.json"
        if not status_path.is_file():
            raise RuntimeError("synthetic session status missing")
        status = json.loads(status_path.read_text(encoding="utf-8"))
        state = status.get("state")
        manifest = validator.build_manifest(directory)
        expected_count = 1 if mode == "term" else 2
        expected_complete = mode == "term"
        if (state != ("closed" if expected_complete else "open")
                or status.get("session_id") != session_id
                or status.get("eligible_attempts") != 1
                or status.get("recorded_events") != (3 if expected_complete else 1)
                or manifest["request_count"] != expected_count
                or manifest["complete_count"] != 1
                or manifest["session_complete"] != expected_complete):
            raise RuntimeError(f"unexpected synthetic session state: {mode}")
        capture_rows = manifest["captures"]
        session_rows = manifest["capture_sessions"]
        if (len(capture_rows) != expected_count or len(session_rows) != expected_count
                or manifest["orphan_results"] or manifest["rejected_count"] != 0
                or not any(row.get("capture_id") == capture_id
                           and row.get("session_id") == session_id
                           and row.get("complete") == expected_complete for row in capture_rows)
                or not any(row.get("session_id") == session_id
                           and row.get("complete") == expected_complete for row in session_rows)):
            raise RuntimeError(f"unexpected synthetic capture/session linkage: {mode}")
        expected_by_kind = ({"request": 1, "result": 1, "parse": 1, "rejected": 0}
                            if expected_complete else
                            {"request": 1, "result": 0, "parse": 0, "rejected": 0})
        if status.get("recorded_by_kind") != expected_by_kind or status.get("drops") != {}:
            raise RuntimeError(f"unexpected synthetic event ledger: {mode}")
        if expected_complete and (not (directory / f"{capture_id}.result.json").is_file()
                                  or not (directory / f"{capture_id}.parse.json").is_file()):
            raise RuntimeError("synthetic result or parse evidence missing")
        if expected_complete:
            shutdown_path = directory / "shutdown-term.json"
            if not shutdown_path.is_file():
                raise RuntimeError("synthetic shutdown evidence missing")
            shutdown = json.loads(shutdown_path.read_text(encoding="utf-8"))
            if (shutdown.get("finalized") is not True or shutdown.get("pending_data", 0) < 2
                    or shutdown.get("pending_control", 0) < 1):
                raise RuntimeError("synthetic pending queue contract failed")
        elif (directory / f"{capture_id}.result.json").exists() or (directory / f"{capture_id}.parse.json").exists():
            raise RuntimeError("SIGKILL unexpectedly completed the pending capture")
        current_hashes = {path.name: _sha(path) for path in directory.glob("*.json") if path.name.endswith(
            (".request.json", ".result.json", ".parse.json"))}
        current_hashes.update({path.name: _sha(path) for path in status_paths})
        if any(current_hashes.get(name) != digest for name, digest in earlier_hashes.items()):
            raise RuntimeError("earlier synthetic evidence changed after restart")
        earlier_hashes = current_hashes
        results[mode] = {"session_id": session_id, "state": state,
                         "elapsed_seconds": round(time.monotonic() - started, 3),
                         "recorded_events": status["recorded_events"],
                         "request_sha256": ready["request_sha256"],
                         "session_complete": manifest["session_complete"]}
    return {"synthetic_only": True, "probe_valid": True, "cases": results,
            "limitations": "tests isolated Uvicorn/ASGI over Unix socket, not production app or Docker stop"}


def main(argv=None):
    if sys.platform != "linux":
        raise SystemExit("Linux only; no process launched")
    if argv is None:
        argv = sys.argv[1:]
    if len(argv) == 3 and argv[0] == "_child":
        asyncio.run(_child(Path(argv[1]), argv[2]))
        return 0
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=30)
    args = parser.parse_args(argv)
    if args.timeout <= 0:
        parser.error("timeout must be positive")
    result = _parent(args.output.resolve(), args.timeout)
    (args.output / "lifecycle-summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
