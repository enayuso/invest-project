"""Default-off, fail-open observation of the first native RM request.

Files are private evaluation evidence, never application inputs. The SDK still
receives the original kwargs/object and keeps its own retry semantics.
"""

from __future__ import annotations

import contextvars
import atexit
import hashlib
import json
import logging
import os
import queue
import re
import threading
import time
import uuid
import math
from contextlib import contextmanager
from dataclasses import asdict, dataclass, fields, is_dataclass
from datetime import datetime, timezone
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Iterator

import httpx

from app.config import settings

_VERSION = "rm-first-native-v1"
_logger = logging.getLogger(__name__)
_sensitive_key = re.compile(r"^((?:proxy[_-])?authorization|(?:set[_-])?cookie|headers|extra_headers|api[_-]?key|app[_-]?key|secret|password|access[_-]?token|refresh[_-]?token|account([_-]?(id|number|no))?)$", re.I)
_sensitive_value = re.compile(
    r"(?i)(?:bearer\s+\S+|(?<![\w-])[\"']?(?:(?:proxy[_-])?authorization|(?:set[_-])?cookie)[\"']?\s*[:=]"
    r"|sk-[A-Za-z0-9_-]{12,}"
    r"|(?:https?://)[^\s/@:]+:[^\s/@]+@|\b(?:\d{2}-\d{4}-\d{4}|\d{3}-\d{2,4}-\d{4,6})\b"
    r"|[\"']?(?:password|api[_-]?key|app[_-]?key|secret|access[_-]?token|refresh[_-]?token|account(?:[_-]?(?:id|number|no))?)[\"']?\s*[:=]\s*[\"']?\S+)"
)
_MAX_NODES = 20000
_MAX_DEPTH = 64
_REQUEST_FIELDS = frozenset({"model", "messages", "temperature", "extra_body", "max_tokens", "response_format"})


@dataclass(frozen=True)
class _Scope:
    run_id: str
    ticker: str
    trade_date: str
    market: str
    trigger: str
    held: bool
    session_in_progress: bool


_scope: contextvars.ContextVar[_Scope | None] = contextvars.ContextVar("rm_capture_scope", default=None)


@contextmanager
def capture_research_manager(
    *, ticker: str, trade_date: str, market: str, trigger: str,
    held: bool, session_in_progress: bool, historical: bool,
) -> Iterator[None]:
    """Bind only the predeclared domestic, automatic, flat, intraday stratum."""
    try:
        active = (settings.rm_request_capture_enabled and market == "domestic" and trigger == "auto"
                  and not held and session_in_progress and not historical)
        scope = _Scope(uuid.uuid4().hex, ticker, trade_date, market, trigger, held, session_in_progress) if active else None
    except Exception:
        scope = None
    if scope is None:
        yield
        return
    token = _scope.set(scope)
    try:
        yield
    finally:
        _scope.reset(token)


def _timestamp() -> str:
    return datetime.fromtimestamp(time.time(), timezone.utc).isoformat()


def _canonical(data: Any) -> bytes:
    return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _ordered(data: Any) -> bytes:
    return json.dumps(data, ensure_ascii=False, sort_keys=False, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _preflight(data: Any, limit: int) -> None:
    """Reject oversized or pathological inputs before making a full JSON copy."""
    stack = [(data, 0)]
    seen: set[int] = set()
    budget = limit
    nodes = 0
    while stack:
        value, depth = stack.pop()
        nodes += 1
        if nodes > _MAX_NODES or depth > _MAX_DEPTH:
            raise ValueError("structure_limit")
        if isinstance(value, (dict, list, tuple)):
            identity = id(value)
            if identity in seen:
                raise ValueError("repeated_container")
            seen.add(identity)
            children = len(value) * (2 if isinstance(value, dict) else 1)
            if nodes + len(stack) + children > _MAX_NODES:
                raise ValueError("structure_limit")
            budget -= len(value) * 3 + 2
            if budget < 0:
                raise ValueError("input_limit")
            if isinstance(value, dict):
                stack.extend((item, depth + 1) for pair in value.items() for item in pair)
            else:
                stack.extend((item, depth + 1) for item in value)
        elif isinstance(value, str):
            budget -= len(value) + 2
        elif value is None or isinstance(value, (bool, int, float)):
            budget -= len(str(value))
        else:
            raise ValueError("unsupported_type")
        if budget < 0:
            raise ValueError("input_limit")


def _preflight_response(data: Any, limit: int) -> None:
    """Bound the SDK object graph before model_dump can allocate a JSON copy."""
    stack = [(data, 0)]
    seen: set[int] = set()
    budget = limit
    nodes = 0
    while stack:
        value, depth = stack.pop()
        nodes += 1
        if nodes > _MAX_NODES or depth > _MAX_DEPTH:
            raise ValueError("structure_limit")
        if isinstance(value, str):
            if len(value) + 2 > budget:
                raise ValueError("response_limit")
            budget -= 2 + sum(1 if ord(char) < 128 and ord(char) >= 32 and char not in ('"', '\\')
                              else (2 if char in ('"', '\\') else 6 if ord(char) < 32 else
                                    2 if ord(char) < 2048 else 3 if ord(char) < 65536 else 4)
                              for char in value)
        elif value is None or isinstance(value, (bool, int, float)):
            budget -= len(str(value))
        else:
            if isinstance(value, dict):
                if nodes + len(stack) + len(value) * 2 > _MAX_NODES:
                    raise ValueError("structure_limit")
                children = [item for pair in value.items() for item in pair]
            elif isinstance(value, (list, tuple)):
                if nodes + len(stack) + len(value) > _MAX_NODES:
                    raise ValueError("structure_limit")
                children = value
            elif is_dataclass(value):
                children = {field.name: getattr(value, field.name) for field in fields(value)}
                stack.append((children, depth + 1))
                continue
            elif hasattr(value, "model_dump"):
                children = vars(value)
                stack.append((children, depth + 1))
                extra = getattr(value, "model_extra", None)
                if extra is not None:
                    if not isinstance(extra, dict):
                        raise ValueError("unsupported_response_type")
                    stack.append((extra, depth + 1))
                continue
            elif hasattr(value, "__dict__"):
                children = vars(value)
                stack.append((children, depth + 1))
                continue
            else:
                raise ValueError("unsupported_response_type")
            identity = id(value)
            if identity in seen:
                raise ValueError("repeated_container")
            seen.add(identity)
            budget -= len(children) * 3 + 2
            if nodes + len(stack) + len(children) > _MAX_NODES:
                raise ValueError("structure_limit")
            stack.extend((item, depth + 1) for item in children)
        if budget < 0:
            raise ValueError("response_limit")


def _timeout_metadata(value: Any) -> dict[str, Any]:
    if value is None:
        return {"kind": "none"}
    if type(value) in (int, float) and math.isfinite(value) and value > 0:
        return {"kind": "scalar", "seconds": value}
    if isinstance(value, httpx.Timeout):
        phases = {name: getattr(value, name) for name in ("connect", "read", "write", "pool")}
        if all(item is None or (type(item) in (int, float) and math.isfinite(item) and item > 0)
               for item in phases.values()):
            return {"kind": "phases", **phases}
    raise ValueError("unsupported_timeout")


def _contains_sensitive(data: Any) -> bool:
    if isinstance(data, dict):
        return any(_sensitive_key.fullmatch(str(k)) or _contains_sensitive(v) for k, v in data.items())
    if isinstance(data, (list, tuple)):
        return any(_contains_sensitive(v) for v in data)
    return isinstance(data, str) and bool(_sensitive_value.search(data))


def _as_json(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if is_dataclass(value):
        return asdict(value)
    return value


def _open_private(path: Path):
    """Create capture evidence with owner-only file permissions."""
    # The capture directory is private; files must also remain private if moved.
    # 수집 폴더 밖으로 옮겨도 파일의 접근 범위를 유지한다.
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        return os.fdopen(descriptor, "wb")
    except Exception:
        os.close(descriptor)
        raise


@lru_cache(maxsize=1)
def _provenance() -> dict[str, Any]:
    repo = Path(__file__).resolve().parents[2]
    files = {
        "recorder": repo / "app/llm/request_capture.py",
        "client_parser": repo / "app/llm/client.py",
        "shared_parser": repo / "app/llm/shared.py",
        "schema_validators": repo / "app/state.py",
        "research_manager": repo / "app/agents/research_manager.py",
    }
    try:
        sdk_version: str | None = version("openai")
    except PackageNotFoundError:
        sdk_version = None
    return {
        "source_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in files.items()},
        "openai_sdk_version": sdk_version,
        "pydantic_version": version("pydantic"),
    }


class _Writer:
    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.events: queue.Queue[tuple[str, bytes]] = queue.Queue(maxsize=32)
        self.control: queue.Queue[tuple[str, bytes] | None] = queue.Queue(maxsize=256)
        self.lock = threading.Lock()
        self.status_lock = threading.Lock()
        self.session_id = uuid.uuid4().hex
        self.status_path = directory / f".capture-session-{self.session_id}.json"
        self.eligible_attempts = 0
        self.recorded_events = 0
        self.recorded_by_kind = {"request": 0, "result": 0, "parse": 0, "rejected": 0}
        self.marker_attempts = 0
        self.closed = False
        self.closing = False
        self.pending_captures: set[str] = set()
        self.pending_done = threading.Condition(self.lock)
        self.queued_bytes = 0
        self.session_bytes = 0
        self.drops: dict[str, int] = {}
        self.stopped_reason: str | None = None
        self.status_io_failed = False
        self.worker = threading.Thread(target=self._run, name="rm-request-capture", daemon=True)
        self.worker.start()
        atexit.register(self.close)

    def _write_status(self) -> bool:
        # Serialize snapshot through rename so an older closed state cannot win.
        # 스냅샷부터 교체까지 직렬화하여 오래된 완료 상태가 최신 누락을 덮지 못한다.
        with self.status_lock:
            with self.lock:
                body = {"schema_version": _VERSION, "kind": "capture_session", "session_id": self.session_id,
                        "state": "incomplete" if self.drops or self.stopped_reason else ("closed" if self.closed else "open"),
                        "stopped_reason": self.stopped_reason,
                        "eligible_attempts": self.eligible_attempts, "recorded_events": self.recorded_events,
                        "recorded_by_kind": dict(self.recorded_by_kind), "drops": dict(self.drops)}
            tmp = self.directory / f".{self.session_id}.{uuid.uuid4().hex}.tmp"
            try:
                with _open_private(tmp) as stream:
                    stream.write(_ordered(body))
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(tmp, self.status_path)
                return True
            except OSError:
                _logger.warning("RM request capture session status write failed")
                try:
                    tmp.unlink(missing_ok=True)
                except OSError:
                    pass
                return False

    def note_eligible(self) -> None:
        with self.lock:
            self.eligible_attempts += 1
        self._wake_control()

    def _wake_control(self) -> None:
        if self.status_io_failed:
            return
        try:
            self.control.put_nowait(None)
        except queue.Full:
            with self.lock:
                if self.stopped_reason is None:
                    self.stopped_reason = "control_queue_limit"
                    self.drops["control_queue_limit"] = self.drops.get("control_queue_limit", 0) + 1
                    _logger.warning("RM request capture control queue full; collection stopped")

    def close(self) -> bool:
        with self.lock:
            if self.closed:
                return not self.drops and self.stopped_reason is None
            if self.closing:
                return False
            self.closing = True
        deadline = time.monotonic() + 5
        drained = True
        with self.pending_done:
            while self.pending_captures:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    drained = False
                    break
                self.pending_done.wait(remaining)
        for pending in (self.events, self.control):
            with pending.all_tasks_done:
                while pending.unfinished_tasks:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        drained = False
                        break
                    pending.all_tasks_done.wait(remaining)
            if not drained:
                break
        if not drained:
            self._halt("drain_timeout")
            self._write_status()
            return False
        with self.lock:
            self.closed = True
        if not self._write_status():
            with self.lock:
                self.closed = False
            self.status_io_failed = True
            self._halt("status_unavailable")
            return False
        with self.lock:
            return not self.drops and self.stopped_reason is None

    def _drop(self, reason: str) -> None:
        with self.lock:
            # Closed sessions have a final denominator; late observers are outside it.
            # 종료 확정 뒤 관측은 확정된 분모와 상태 원장을 바꾸지 않는다.
            if self.closed:
                return
            count = self.drops.get(reason, 0) + 1
            self.drops[reason] = count
            closing = self.closing
        if reason != "capture_stopped" or count == 1:
            _logger.warning("RM request capture dropped an event: %s", reason)
        if closing:
            # A late producer must persist incompleteness even after close's last snapshot.
            # 종료 뒤의 누락도 다음 작업자 기회에 의존하지 않고 바로 기록한다.
            if not self.status_io_failed:
                self._write_status()
        else:
            self._wake_control()

    def _halt(self, reason: str) -> None:
        with self.lock:
            if self.stopped_reason is None:
                self.stopped_reason = reason
        self._drop(reason)

    def is_stopped(self) -> bool:
        with self.lock:
            return self.stopped_reason is not None or self.closing or self.closed

    def accepts_result(self, capture_id: str) -> bool:
        with self.lock:
            return not self.closed and self.stopped_reason is None and capture_id in self.pending_captures

    def reject(self, capture_id: str, stage: str, reason: str, *, run_id: str | None = None) -> None:
        with self.lock:
            if self.stopped_reason is not None or self.closing or self.closed or self.marker_attempts >= 1024:
                marker_allowed = False
            else:
                self.marker_attempts += 1
                marker_allowed = True
        if not marker_allowed:
            if not self.is_stopped():
                self._halt("control_budget_limit")
            return
        event = {
            "schema_version": _VERSION, "kind": "rejected", "capture_id": capture_id,
            "run_id": run_id, "capture_session_id": self.session_id,
            "captured_at_utc": _timestamp(), "stage": stage, "reason": reason,
        }
        name = f"{capture_id}.{stage}.rejected.json"
        encoded = _ordered(event)
        with self.lock:
            try:
                self.events.put_nowait((name, encoded))
            except queue.Full:
                pass
            else:
                self.queued_bytes += len(encoded)
                return
        try:
            self.control.put_nowait((name, encoded))
        except queue.Full:
            self._halt("control_queue_limit")

    def submit(self, name: str, event: dict[str, Any]) -> bool:
        capture_id = event.get("capture_id")
        kind = event.get("kind")
        if self.is_stopped() and not (kind in ("response", "error", "parse")
                                      and isinstance(capture_id, str) and self.accepts_result(capture_id)):
            self._drop("capture_stopped")
            return False
        event = {"capture_session_id": self.session_id, **event}
        try:
            _preflight(event, min(settings.rm_request_capture_event_max_bytes, 2 * 1024 * 1024))
        except (TypeError, ValueError, OverflowError):
            self._drop("event_limit")
            return False
        if _contains_sensitive(event):
            self._drop("sensitive_content")
            return False
        try:
            encoded = _ordered(event)
        except (TypeError, ValueError, OverflowError):
            self._drop("serialization_failed")
            return False
        if len(encoded) > min(settings.rm_request_capture_event_max_bytes, 2 * 1024 * 1024):
            self._drop("event_limit")
            return False
        with self.lock:
            if self.stopped_reason is not None:
                reason = "capture_stopped"
            elif self.closed or (self.closing and kind not in ("response", "error", "parse")):
                reason = "capture_stopped"
            elif self.closing and capture_id not in self.pending_captures:
                reason = "capture_stopped"
            elif self.queued_bytes + len(encoded) > min(settings.rm_request_capture_queue_max_bytes, 64 * 1024 * 1024):
                reason = "queue_limit"
            elif self.session_bytes + self.queued_bytes + len(encoded) > min(settings.rm_request_capture_session_max_bytes, 512 * 1024 * 1024):
                reason = "session_limit"
            else:
                try:
                    self.events.put_nowait((name, encoded))
                except queue.Full:
                    reason = "queue_limit"
                else:
                    self.queued_bytes += len(encoded)
                    if kind == "request" and isinstance(capture_id, str):
                        self.pending_captures.add(capture_id)
                    elif (kind in ("error", "parse") and isinstance(capture_id, str)
                          and capture_id in self.pending_captures):
                        self.pending_captures.remove(capture_id)
                        self.pending_done.notify_all()
                    return True
        if reason in ("queue_limit", "session_limit"):
            self._halt(reason)
        else:
            self._drop(reason)
        return False

    def _run(self) -> None:
        if not self._write_status():
            self.status_io_failed = True
            self._halt("status_unavailable")
        while True:
            control = True
            try:
                entry = self.control.get_nowait()
            except queue.Empty:
                control = False
                try:
                    entry = self.events.get(timeout=0.1)
                except queue.Empty:
                    continue
            tmp: Path | None = None
            try:
                if entry is None:
                    continue
                with self.lock:
                    halted = self.stopped_reason is not None
                if halted:
                    self._drop("capture_stopped")
                    continue
                name, encoded = entry
                tmp = self.directory / f".{name}.{uuid.uuid4().hex}.tmp"
                target = self.directory / name
                with _open_private(tmp) as stream:
                    stream.write(encoded)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(tmp, target)
                with self.lock:
                    if not control:
                        self.session_bytes += len(encoded)
                    self.recorded_events += 1
                    kind = "rejected" if name.endswith(".rejected.json") else name.rsplit(".", 2)[-2]
                    self.recorded_by_kind[kind] += 1
            except Exception:
                self._halt("write_failed")
                try:
                    if tmp is not None:
                        tmp.unlink(missing_ok=True)
                except OSError:
                    pass
            finally:
                try:
                    if not control and entry is not None:
                        with self.lock:
                            self.queued_bytes -= len(entry[1])
                    if not self.status_io_failed and not self._write_status():
                        self.status_io_failed = True
                        self._halt("status_unavailable")
                finally:
                    (self.control if control else self.events).task_done()


_writer_lock = threading.Lock()
_request_registration_lock = threading.Lock()
_writer: _Writer | None = None
_writer_shutdown = False


def _get_writer() -> _Writer | None:
    global _writer
    if _writer is not None:
        return _writer
    with _writer_lock:
        if _writer is not None:
            return _writer
        if _writer_shutdown:
            return None
        configured = settings.rm_request_capture_directory
        if not configured:
            _logger.warning("RM request capture disabled for this event: directory_not_configured")
            return None
        path = Path(configured)
        if not path.is_absolute() or path.is_symlink() or not path.is_dir() or path != path.resolve():
            _logger.warning("RM request capture disabled for this event: invalid_directory")
            return None
        path = path.resolve()
        repository = Path(__file__).resolve().parents[2]
        if repository == path or repository in path.parents or path == repository.parent:
            _logger.warning("RM request capture disabled for this event: source_directory")
            return None
        try:
            _writer = _Writer(path)
        except OSError:
            _logger.warning("RM request capture disabled for this event: session_status_unavailable")
            return None
        return _writer


def health_snapshot() -> dict[str, Any]:
    """Safe counts only; capture content and path are never exposed."""
    writer = _writer
    if writer is None:
        return {"active": False, "queued_bytes": 0, "session_bytes": 0,
                "eligible_attempts": 0, "recorded_events": 0, "drops": {}, "stopped_reason": None}
    with writer.lock:
        return {"active": writer.stopped_reason is None and not writer.closing and not writer.closed,
                "stopped_reason": writer.stopped_reason,
                "queued_bytes": writer.queued_bytes,
                "session_bytes": writer.session_bytes, "eligible_attempts": writer.eligible_attempts,
                "recorded_events": writer.recorded_events,
                "recorded_by_kind": dict(writer.recorded_by_kind), "drops": dict(writer.drops)}


def finalize_capture() -> bool:
    """Close an existing writer at application shutdown without creating one."""
    global _writer_shutdown
    # Complete request registration before publishing the shutdown barrier.
    # 요청 등록이 끝난 뒤 종료 경계를 공개해 분모를 확정한다.
    with _request_registration_lock:
        with _writer_lock:
            _writer_shutdown = True
            writer = _writer
    return True if writer is None else writer.close()


def observe_first_native_request(
    request_kwargs: dict[str, Any], *, label: str, first: bool,
    provider: str, sdk_max_retries: int, timeout_seconds: Any,
) -> str | None:
    # Enrollment and shutdown use one boundary, including request serialization.
    # 적격 등록과 종료가 같은 경계를 사용하며 요청 직렬화도 포함한다.
    with _request_registration_lock:
        if _writer_shutdown:
            return None
        return _observe_registered_request(
            request_kwargs, label=label, first=first, provider=provider,
            sdk_max_retries=sdk_max_retries, timeout_seconds=timeout_seconds,
        )


def _observe_registered_request(
    request_kwargs: dict[str, Any], *, label: str, first: bool,
    provider: str, sdk_max_retries: int, timeout_seconds: Any,
) -> str | None:
    scope = _scope.get()
    if not first or label != "research_manager" or scope is None or provider not in ("vllm", "gateway"):
        return None
    writer: _Writer | None = None
    capture_id: str | None = None
    try:
        writer = _get_writer()
        if writer is None:
            return None
        writer.note_eligible()
        if writer.is_stopped():
            writer._drop("capture_stopped")
            return None
        capture_id = uuid.uuid4().hex
        try:
            limit = min(settings.rm_request_capture_event_max_bytes, 2 * 1024 * 1024)
            _preflight(request_kwargs, limit)
            if not isinstance(request_kwargs, dict) or set(request_kwargs) != _REQUEST_FIELDS:
                writer._drop("request_fields_invalid")
                writer.reject(capture_id, "request", "request_fields_invalid", run_id=scope.run_id)
                return None
            ordered_request = _ordered(request_kwargs)
            if len(ordered_request) > limit:
                raise ValueError("request_limit")
            payload = json.loads(ordered_request)
        except (TypeError, ValueError, OverflowError, RecursionError):
            writer._drop("request_limit")
            writer.reject(capture_id, "request", "request_limit", run_id=scope.run_id)
            return None
        if _contains_sensitive(payload):
            writer._drop("sensitive_content")
            writer.reject(capture_id, "request", "sensitive_content", run_id=scope.run_id)
            return None
        try:
            sdk_timeout = _timeout_metadata(timeout_seconds)
        except ValueError:
            writer._drop("unsupported_timeout")
            writer.reject(capture_id, "request", "unsupported_timeout", run_id=scope.run_id)
            return None
        schema = payload["response_format"]["json_schema"]
        event = {
            "schema_version": _VERSION, "kind": "request", "capture_id": capture_id,
            "run_id": scope.run_id, "logical_call_id": capture_id,
            "node": "research_manager", "application_attempt_kind": "first_native",
            "captured_at_utc": _timestamp(), "source_decision_id": None,
            "market": scope.market, "trigger": scope.trigger, "held": scope.held,
            "session_in_progress": scope.session_in_progress,
            "trade_date": scope.trade_date, "ticker": scope.ticker,
            "provider": provider, "endpoint_alias": f"{provider}_primary",
            "sdk_retry_visibility": "not_observed",
            "sdk_max_retries": sdk_max_retries, "sdk_timeout": sdk_timeout,
            "latency_definition": "observer_start_to_sdk_result_including_request_capture",
            "capture_level": "client_boundary_request", "request_complete": True,
            "request": payload, "request_sha256": hashlib.sha256(_canonical(payload)).hexdigest(),
            "request_ordered_sha256": hashlib.sha256(ordered_request).hexdigest(),
            "messages_sha256": hashlib.sha256(_canonical(payload["messages"])).hexdigest(),
            "schema_sha256": hashlib.sha256(_canonical(schema)).hexdigest(),
            "provenance": _provenance(),
            "schema_name": schema["name"],
            "serving_provenance": None,
        }
        if writer.submit(f"{capture_id}.request.json", event):
            return capture_id
        writer.reject(capture_id, "request", "request_not_recorded", run_id=scope.run_id)
        return None
    except Exception:
        _logger.warning("RM request capture dropped an event: observer_failed")
        if writer is not None:
            writer._drop("observer_failed")
            if capture_id is not None:
                writer.reject(capture_id, "request", "observer_failed", run_id=scope.run_id)
        return None


def observe_first_native_result(
    capture_id: str | None, *, response: Any = None, error: BaseException | None = None,
    elapsed_seconds: float,
) -> None:
    if capture_id is None:
        return
    writer: _Writer | None = None
    try:
        writer = _get_writer()
        if writer is None:
            return
        if not writer.accepts_result(capture_id):
            writer._drop("capture_stopped")
            return
        if error is None:
            _preflight_response(response, min(settings.rm_request_capture_event_max_bytes, 2 * 1024 * 1024))
        raw_response = None if error is not None else _as_json(response)
        if error is None:
            _preflight(raw_response, min(settings.rm_request_capture_event_max_bytes, 2 * 1024 * 1024))
        event = {
            "schema_version": _VERSION, "kind": "error" if error is not None else "response",
            "capture_id": capture_id, "captured_at_utc": _timestamp(),
            "elapsed_seconds": elapsed_seconds,
            "latency_definition": "observer_start_to_sdk_result_including_request_capture",
            "raw_response": raw_response,
            "raw_response_sha256": None if error is not None else hashlib.sha256(_canonical(raw_response)).hexdigest(),
            "error_class": type(error).__name__ if error is not None else None,
            "response_complete": error is None,
        }
        if not writer.submit(f"{capture_id}.result.json", event):
            writer.reject(capture_id, "result", "result_not_recorded")
    except Exception:
        _logger.warning("RM request capture dropped an event: observer_failed")
        if writer is not None:
            writer._drop("response_limit" if error is None else "observer_failed")
            writer.reject(capture_id, "result", "response_not_recorded" if error is None else "observer_failed")


def observe_first_native_parse(capture_id: str | None, *, error: Exception | None = None) -> None:
    if capture_id is None:
        return
    writer: _Writer | None = None
    try:
        writer = _get_writer()
        if writer is None:
            return
        if not writer.accepts_result(capture_id):
            writer._drop("capture_stopped")
            return
        if not writer.submit(f"{capture_id}.parse.json", {
            "schema_version": _VERSION, "kind": "parse", "capture_id": capture_id,
            "captured_at_utc": _timestamp(), "valid": error is None,
            "error_class": type(error).__name__ if error is not None else None,
        }):
            writer.reject(capture_id, "parse", "parse_not_recorded")
    except Exception:
        _logger.warning("RM request capture dropped an event: observer_failed")
        if writer is not None:
            writer._drop("observer_failed")
            writer.reject(capture_id, "parse", "observer_failed")
