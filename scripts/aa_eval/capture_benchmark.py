"""Offline synthetic RM capture latency/RSS probe; never import the live app."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import statistics
import subprocess
import sys
import time
import uuid
from pathlib import Path
from types import ModuleType, SimpleNamespace

CASES = ("ascii_8k", "ascii_128k", "ascii_near_limit",
         "korean_8k", "korean_128k", "korean_near_limit", "sdk_extra",
         "rejected_request", "rejected_response", "queue_saturation")
STAGES = ("request", "result", "parse")


def _deny_external(event, _args):
    if event in ("socket.connect", "socket.getaddrinfo", "subprocess.Popen", "os.system"):
        raise RuntimeError("external execution denied by synthetic capture probe")


def _load_recorder(repo: Path, output: Path):
    settings = SimpleNamespace(rm_request_capture_enabled=False,
        rm_request_capture_directory=str(output), rm_request_capture_event_max_bytes=2 * 1024 * 1024,
        rm_request_capture_queue_max_bytes=64 * 1024 * 1024,
        rm_request_capture_session_max_bytes=512 * 1024 * 1024)
    config = ModuleType("app.config")
    config.settings = settings
    sys.modules["app.config"] = config
    spec = importlib.util.spec_from_file_location("synthetic_rm_capture_probe", repo / "app/llm/request_capture.py")
    recorder = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = recorder
    spec.loader.exec_module(recorder)
    return recorder, settings


def _percentile(values: list[float], rank: float) -> float:
    values = sorted(values)
    return values[min(len(values) - 1, max(0, int((len(values) - 1) * rank)))]


def _stats(values: list[float]) -> dict:
    return {"samples": len(values), "p50_ms": statistics.median(values),
            "p95_ms": _percentile(values, .95), "max_ms": max(values)}


def _sdk_response(content: str, *, oversized_extra: str | None = None) -> ChatCompletion:
    """Construct the SDK's nested Pydantic types for the observer cost path."""
    from openai.types.chat import ChatCompletion

    # The recorder inspects model_extra only on SDK models with model_dump.
    # 기록기는 model_dump가 있는 SDK 모델의 중첩 추가 필드까지 검사한다.
    extras = {"top": {"nested_extension": content[:4096]},
              "choice": content[:1024], "message": content[:1024]}
    if oversized_extra is not None:
        extras[oversized_extra] = "x" * 2_300_000
    return ChatCompletion.model_validate({
        "id": "synthetic", "object": "chat.completion", "created": 1,
        "model": "synthetic", "top_extension": extras["top"],
        "choices": [{"index": 0, "finish_reason": "stop",
                     "choice_extension": extras["choice"],
                     "message": {"role": "assistant", "content": content,
                                 "message_extension": extras["message"]}}],
    })


def _payload(label: str):
    if label.startswith("ascii_") or label.startswith("korean_"):
        language, size = label.split("_", 1)
        unit = "x" if language == "ascii" else "한"
        count = {"8k": 8192, "128k": 131072, "near_limit": 1800000}[size]
        count //= len(unit.encode("utf-8"))
        request_body = unit * count
        response_body = unit * count
    elif label == "sdk_extra":
        request_body, response_body = "x" * 8192, "x" * 65536
    elif label == "rejected_request":
        request_body, response_body = "x" * 2300000, "{}"
    elif label == "rejected_response":
        request_body, response_body = "x" * 8192, "x" * 2300000
    else:
        raise ValueError("unknown synthetic case")
    request = {"model": "synthetic", "messages": [{"role": "user", "content": request_body}],
               "temperature": 0, "extra_body": {}, "max_tokens": 32,
               "response_format": {"type": "json_schema", "json_schema": {"name": "Synthetic",
                   "schema": {"type": "object"}, "strict": True}}}
    response = (_sdk_response(response_body) if label == "sdk_extra" else
                {"choices": [{"message": {"content": response_body}, "finish_reason": "stop"}]})
    return request, response, len(request_body.encode("utf-8")), len(response_body.encode("utf-8"))


def _observe(recorder, request: dict, response, stage: str, capture_id: str | None):
    if stage == "request":
        with recorder.capture_research_manager(ticker="005930.KS", trade_date="2026-09-23",
                market="domestic", trigger="auto", held=False,
                session_in_progress=True, historical=False):
            return recorder.observe_first_native_request(request, label="research_manager",
                first=True, provider="vllm", sdk_max_retries=0, timeout_seconds=300)
    if stage == "result":
        recorder.observe_first_native_result(capture_id, response=response, elapsed_seconds=0.001)
    else:
        recorder.observe_first_native_parse(capture_id,
            error=ValueError("synthetic") if isinstance(response, dict) and
            len(response["choices"][0]["message"]["content"]) > 2 * 1024 * 1024 else None)
    return capture_id


def _saturation(recorder, settings) -> dict:
    settings.rm_request_capture_enabled = True
    writer = recorder._get_writer()
    if writer is None:
        raise RuntimeError("synthetic writer failed to start")
    deadline = time.monotonic() + 5
    while not writer.status_path.exists() and time.monotonic() < deadline:
        time.sleep(.01)
    body = "x" * 1800000
    before = _memory()
    accepted = 0
    peak_queue = 0
    writer.status_lock.acquire()
    try:
        for index in range(40):
            event = {"schema_version": 1, "kind": "parse", "capture_id": uuid.uuid4().hex,
                     "captured_at_utc": "2026-09-23T00:00:00Z", "valid": True,
                     "synthetic_padding": body}
            if writer.submit(f"synthetic-{index}.parse.json", event):
                accepted += 1
            peak_queue = max(peak_queue, writer.queued_bytes)
            if writer.stopped_reason:
                break
    finally:
        writer.status_lock.release()
    at_peak = _memory()
    writer.events.join()
    writer.control.join()
    finalized = recorder.finalize_capture()
    after = _memory()
    valid = (1 <= accepted <= 33 and writer.stopped_reason == "queue_limit"
             and peak_queue <= 64 * 1024 * 1024 and writer.queued_bytes == 0
             and writer.drops.get("queue_limit", 0) >= 1 and finalized is False
             and _memory_available(before, at_peak, after))
    return {"probe_valid": valid, "accepted": accepted, "peak_queued_bytes": peak_queue,
            "queue_bytes_at_end": writer.queued_bytes, "drops": dict(writer.drops),
            "writer_finalized": finalized, "rss_before": before,
            "rss_after_submit": at_peak, "rss_after_drain": after}


def _case_run(label: str, output: Path, repetitions: int, warmup: int) -> dict:
    repo = Path(__file__).resolve().parents[2]
    sys.addaudithook(_deny_external)
    recorder, settings = _load_recorder(repo, output)
    if label == "queue_saturation":
        return _saturation(recorder, settings)
    request, response, request_bytes, response_bytes = _payload(label)
    with_inputs = _memory()
    settings.rm_request_capture_enabled = True
    startup_started = time.perf_counter_ns()
    writer = recorder._get_writer()
    if writer is None:
        raise RuntimeError("synthetic writer failed to start")
    deadline = time.monotonic() + 5
    while not writer.status_path.exists() and time.monotonic() < deadline:
        time.sleep(.01)
    if not writer.status_path.exists():
        raise RuntimeError("synthetic writer status missing")
    startup_ms = (time.perf_counter_ns() - startup_started) / 1_000_000
    idle_on_writer = _memory()
    settings.rm_request_capture_enabled = False
    samples = {mode: {stage: [] for stage in STAGES} for mode in ("off", "on")}
    accepted = 0
    for index in range(warmup + repetitions):
        for mode in (("off", "on") if index % 2 == 0 else ("on", "off")):
            settings.rm_request_capture_enabled = mode == "on"
            capture_id = None
            for stage in STAGES:
                started = time.perf_counter_ns()
                capture_id = _observe(recorder, request, response, stage, capture_id)
                if index >= warmup:
                    samples[mode][stage].append((time.perf_counter_ns() - started) / 1_000_000)
            if mode == "on" and capture_id is not None:
                accepted += 1
            if mode == "on" and recorder._writer is not None:
                recorder._writer.events.join()
                recorder._writer.control.join()
    before_finalize = _memory()
    finalized = recorder.finalize_capture()
    after_finalize = _memory()
    total = warmup + repetitions
    expected_accepted = 0 if label == "rejected_request" else total
    expected_finalized = label not in ("rejected_request", "rejected_response")
    drops = {} if writer is None else dict(writer.drops)
    expected_drop = ("request_limit" if label == "rejected_request" else
                     "response_limit" if label == "rejected_response" else None)
    expected_kinds = ({"request": 0, "result": 0, "parse": 0, "rejected": total}
                      if label == "rejected_request" else
                      {"request": total, "result": 0, "parse": total, "rejected": total}
                      if label == "rejected_response" else
                      {"request": total, "result": total, "parse": total, "rejected": 0})
    valid = (accepted == expected_accepted and finalized == expected_finalized
             and writer is not None and writer.queued_bytes == 0
             and writer.recorded_by_kind == expected_kinds
             and (drops.get(expected_drop, 0) == total if expected_drop else drops == {})
             and _memory_available(with_inputs, idle_on_writer,
                                   before_finalize, after_finalize))
    memory_delta = {
        "rss_idle_added_bytes": _nonnegative_delta(idle_on_writer["rss_bytes"], with_inputs["rss_bytes"]),
        "rss_before_finalize_added_bytes": _nonnegative_delta(before_finalize["rss_bytes"], with_inputs["rss_bytes"]),
        "peak_after_finalize_added_bytes": _nonnegative_delta(after_finalize["peak_rss_bytes"], with_inputs["peak_rss_bytes"]),
    }
    return {"probe_valid": valid, "request_utf8_bytes": request_bytes,
            "response_utf8_bytes": response_bytes, "accepted_on": accepted,
            "off": {stage: _stats(values) for stage, values in samples["off"].items()},
            "on": {stage: _stats(values) for stage, values in samples["on"].items()},
            "p95_added_ms": {stage: _percentile(samples["on"][stage], .95)
                             - _percentile(samples["off"][stage], .95) for stage in STAGES},
            "writer_finalized": finalized, "drops": drops,
            "recorded_by_kind": dict(writer.recorded_by_kind),
            "queue_bytes_at_end": writer.queued_bytes,
            "writer_startup_ms": startup_ms,
            "rss_with_inputs": with_inputs, "rss_idle_on_writer": idle_on_writer,
            "rss_before_finalize": before_finalize,
            "rss_after_finalize": after_finalize, "memory_delta": memory_delta}


def _memory_available(*readings: dict[str, int | None]) -> bool:
    return all(isinstance(value, int) and value > 0
               for reading in readings for value in reading.values())


def _nonnegative_delta(after: int | None, before: int | None) -> int | None:
    if after is None or before is None:
        return None
    return max(0, after - before)


def run(output: Path, repetitions: int, warmup: int) -> dict:
    repo = Path(__file__).resolve().parents[2]
    if output.exists() or repo == output or repo in output.parents:
        raise ValueError("probe output must be new and outside the repository")
    output.mkdir(parents=True)
    results = {}
    child_env = {"PATH": os.environ.get("PATH", ""), "PYTHONNOUSERSITE": "1"}
    if os.name == "nt":
        # Windows needs its system directory to initialize the SDK's asyncio dependency.
        # Windows에서는 SDK의 asyncio 의존성 초기화에 시스템 경로가 필요하다.
        for name in ("SystemRoot", "WINDIR"):
            if name in os.environ:
                child_env[name] = os.environ[name]
    for label in CASES:
        case_dir = output / label
        case_dir.mkdir()
        completed = subprocess.run([sys.executable, "-B", str(Path(__file__).resolve()),
                                    "_case", label, str(case_dir), str(repetitions), str(warmup)],
                                   stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, text=True, timeout=600,
                                   env=child_env)
        summary_path = case_dir / "case-summary.json"
        if completed.returncode != 0 or not summary_path.is_file():
            raise RuntimeError(f"synthetic case failed: {label} (exit {completed.returncode}): "
                               f"{completed.stderr[-500:]}")
        results[label] = json.loads(summary_path.read_text(encoding="utf-8"))
    return {"synthetic_only": True,
            "probe_valid": all(row["probe_valid"] for row in results.values()),
            "worker_processes": len(CASES), "cases": results,
            "limitations": "isolated synthetic processes; no live server, Docker, network, or storage fault"}


def _memory() -> dict[str, int | None]:
    try:
        import psutil
        info = psutil.Process(os.getpid()).memory_info()
        peak = getattr(info, "peak_wset", None)
        if peak is not None:
            return {"rss_bytes": info.rss, "peak_rss_bytes": peak}
    except ImportError:
        pass
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t),
                        ("PrivateUsage", ctypes.c_size_t)]

        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        get_process = ctypes.windll.kernel32.GetCurrentProcess
        get_process.restype = wintypes.HANDLE
        get_memory = ctypes.windll.psapi.GetProcessMemoryInfo
        get_memory.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        get_memory.restype = wintypes.BOOL
        if get_memory(get_process(), ctypes.byref(counters), counters.cb):
            return {"rss_bytes": counters.WorkingSetSize,
                    "peak_rss_bytes": counters.PeakWorkingSetSize}
    try:
        import resource
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        rss = None
        if sys.platform.startswith("linux"):
            try:
                rss = int(Path("/proc/self/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")
            except (OSError, ValueError, IndexError):
                pass
        return {"rss_bytes": rss,
                "peak_rss_bytes": value if sys.platform == "darwin" else value * 1024}
    except ImportError:
        return {"rss_bytes": None, "peak_rss_bytes": None}


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if len(argv) == 5 and argv[0] == "_case":
        label, destination, repetitions, warmup = argv[1:]
        if label not in CASES or int(repetitions) < 1 or int(warmup) < 0:
            raise ValueError("invalid synthetic child arguments")
        result = _case_run(label, Path(destination).resolve(), int(repetitions), int(warmup))
        (Path(destination) / "case-summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        return 0 if result["probe_valid"] else 2
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args(argv)
    if args.repetitions < 1 or args.warmup < 0:
        parser.error("repetitions must be positive and warmup nonnegative")
    destination = args.output.resolve()
    result = run(destination, args.repetitions, args.warmup)
    (destination / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"synthetic_only": True, "probe_valid": result["probe_valid"],
                      "case_names": list(result["cases"])}))
    return 0 if result["probe_valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
