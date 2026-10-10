"""Synthetic boundary checks only: no model, broker, database or pipeline run."""

from __future__ import annotations

import asyncio
import json
import os
import queue
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import httpx
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion
from pydantic import BaseModel

from app.config import settings
from app.llm import request_capture as capture
from app.llm.client import LLMClient
from scripts.aa_eval.validate import build_manifest, main as validate_main


class _Plan(BaseModel):
    value: str


class _Response:
    def __init__(self) -> None:
        self.choices = [SimpleNamespace(message=SimpleNamespace(content='{"value":"ok"}'), finish_reason="stop")]
        self.usage = None

    def model_dump(self, *, mode: str) -> dict:
        return {"choices": [{"message": {"content": '{"value":"ok"}'}, "finish_reason": "stop"}], "usage": None}


@pytest.fixture
def enabled(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(settings, "rm_request_capture_enabled", True)
    monkeypatch.setattr(settings, "rm_request_capture_directory", str(tmp_path))
    monkeypatch.setattr(capture, "_writer", None)
    monkeypatch.setattr(capture, "_writer_shutdown", False)
    yield tmp_path
    writer = capture._writer
    if writer is not None:
        with writer.pending_done:
            writer.pending_captures.clear()
            writer.pending_done.notify_all()
        writer.close()
    monkeypatch.setattr(capture, "_writer", None)


def _scope():
    return capture.capture_research_manager(
        ticker="005930.KS", trade_date="2026-09-23", market="domestic",
        trigger="auto", held=False, session_in_progress=True, historical=False,
    )


def _kwargs(content: str = "safe") -> dict:
    return {"model": "fixture", "messages": [{"role": "user", "content": content}],
            "temperature": 0, "extra_body": {}, "max_tokens": 10,
            "response_format": {"type": "json_schema", "json_schema": {
                "name": "_Plan", "schema": _Plan.model_json_schema(), "strict": True}}}


def _record_request(kwargs: dict | None = None, timeout=10) -> str | None:
    with _scope():
        return capture.observe_first_native_request(
            kwargs or _kwargs(), label="research_manager", first=True,
            provider="vllm", sdk_max_retries=5, timeout_seconds=timeout)


def _record_complete() -> str:
    capture_id = _record_request()
    assert capture_id is not None
    capture.observe_first_native_result(capture_id, response=_Response(), elapsed_seconds=0.1)
    capture.observe_first_native_parse(capture_id)
    capture._writer.events.join()
    return capture_id


@pytest.mark.asyncio
async def test_first_native_capture_preserves_sdk_kwargs_and_result(enabled, monkeypatch):
    client = LLMClient()
    sent = []

    async def fake_create(**kwargs):
        sent.append(kwargs)
        return _Response()

    monkeypatch.setattr(client._client.chat.completions, "create", fake_create)
    monkeypatch.setattr(client, "_record_metrics", lambda *args, **kwargs: None)
    original = [{"role": "user", "content": "analysis"}]
    with _scope():
        output = await client.structured_complete(original, _Plan, label="research_manager", language="ko")
    assert output.value == "ok"
    assert original == [{"role": "user", "content": "analysis"}]
    assert len(sent) == 1
    capture._writer.events.join()
    request_path = next(enabled.glob("*.request.json"))
    event = json.loads(request_path.read_text(encoding="utf-8"))
    assert event["request"] == sent[0]
    assert "Korean" in str(event["request"]["messages"]) or "한국" in str(event["request"]["messages"])
    assert event["request"]["response_format"]["json_schema"]["name"] == "_Plan"
    assert event["request"]["extra_body"] == sent[0]["extra_body"]
    assert json.loads(next(enabled.glob("*.result.json")).read_text(encoding="utf-8"))["raw_response"] == _Response().model_dump(mode="json")
    assert json.loads(next(enabled.glob("*.parse.json")).read_text(encoding="utf-8"))["valid"] is True
    manifest = build_manifest(enabled)
    assert manifest["complete_count"] == 1
    assert manifest["ready_for_replay"] is False
    assert manifest["captures"][0]["problems"] == ["serving_provenance_missing", "replay_runtime_not_implemented"]


def test_secret_rejection_and_context_isolation(enabled):
    kwargs = {"model": "fixture", "messages": [{"role": "user", "content": "sk-ABCDEFGHIJKLMNOP"}],
              "temperature": 0, "extra_body": {}, "max_tokens": 10,
              "response_format": {"type": "json_schema", "json_schema": {"name": "_Plan", "schema": _Plan.model_json_schema(), "strict": True}}}
    with _scope():
        assert capture.observe_first_native_request(kwargs, label="research_manager", first=True,
                                                    provider="vllm", sdk_max_retries=5, timeout_seconds=10) is None
    capture._writer.events.join()
    assert not list(enabled.glob("*.request.json"))
    assert build_manifest(enabled)["rejected_count"] == 1
    assert capture.health_snapshot()["drops"]["sensitive_content"] == 1

    clean = {**kwargs, "messages": [{"role": "user", "content": "safe"}]}

    async def one():
        with _scope():
            await asyncio.sleep(0)
            return capture.observe_first_native_request(clean, label="research_manager", first=True,
                                                        provider="vllm", sdk_max_retries=5, timeout_seconds=10)

    async def pair():
        return await asyncio.gather(one(), one())

    ids = asyncio.run(pair())
    assert len(set(ids)) == 2
    capture._writer.events.join()
    events = [json.loads(path.read_text(encoding="utf-8")) for path in enabled.glob("*.request.json")]
    assert len({event["run_id"] for event in events}) == 2
    assert all(event["capture_id"] in ids for event in events)


def test_validator_rejects_missing_and_tampered_capture(enabled):
    kwargs = {"model": "fixture", "messages": [{"role": "user", "content": "safe"}],
              "temperature": 0, "extra_body": {}, "max_tokens": 10,
              "response_format": {"type": "json_schema", "json_schema": {"name": "_Plan", "schema": _Plan.model_json_schema(), "strict": True}}}
    with _scope():
        capture_id = capture.observe_first_native_request(kwargs, label="research_manager", first=True,
                                                          provider="vllm", sdk_max_retries=5, timeout_seconds=10)
    capture._writer.events.join()
    assert "response_missing" in build_manifest(enabled)["captures"][0]["problems"]
    event_path = enabled / f"{capture_id}.request.json"
    event = json.loads(event_path.read_text(encoding="utf-8"))
    event["node"] = "portfolio_manager"
    event["request"]["model"] = "tampered"
    event_path.write_text(json.dumps(event), encoding="utf-8")
    problems = build_manifest(enabled)["captures"][0]["problems"]
    assert "wrong_node_or_attempt" in problems
    assert "request_hash_mismatch" in problems
    event["schema_version"] = "unknown"
    event_path.write_text(json.dumps(event), encoding="utf-8")
    assert "unknown_schema_version" in build_manifest(enabled)["captures"][0]["problems"]
    manifest_path = enabled.parent / "manifest.json"
    assert validate_main([str(enabled), "--output", str(manifest_path)]) == 2
    assert manifest_path.is_file()


def test_native_error_is_recorded_without_error_text(enabled):
    kwargs = {"model": "fixture", "messages": [{"role": "user", "content": "safe"}],
              "temperature": 0, "extra_body": {}, "max_tokens": 10,
              "response_format": {"type": "json_schema", "json_schema": {"name": "_Plan", "schema": _Plan.model_json_schema(), "strict": True}}}
    with _scope():
        capture_id = capture.observe_first_native_request(kwargs, label="research_manager", first=True,
                                                          provider="vllm", sdk_max_retries=5, timeout_seconds=10)
        capture.observe_first_native_result(capture_id, error=RuntimeError("secret details"), elapsed_seconds=0.1)
    capture._writer.events.join()
    result = json.loads((enabled / f"{capture_id}.result.json").read_text(encoding="utf-8"))
    assert result["error_class"] == "RuntimeError"
    assert "secret details" not in json.dumps(result)
    assert build_manifest(enabled)["complete_count"] == 1


@pytest.mark.asyncio
async def test_recorder_failure_does_not_change_native_result(enabled, monkeypatch):
    client = LLMClient()
    async def fake_create(**kwargs):
        return _Response()
    monkeypatch.setattr(client._client.chat.completions, "create", fake_create)
    monkeypatch.setattr(client, "_record_metrics", lambda *args, **kwargs: None)
    def failed_writer():
        raise OSError("synthetic writer failure")
    monkeypatch.setattr(capture, "_get_writer", failed_writer)
    with _scope():
        result = await client.structured_complete([{"role": "user", "content": "safe"}],
                                                  _Plan, label="research_manager")
    assert result.value == "ok"


def test_only_first_native_and_declared_stratum_are_captured(enabled):
    kwargs = {"model": "fixture", "messages": [{"role": "user", "content": "safe"}],
              "temperature": 0, "extra_body": {}, "max_tokens": 10,
              "response_format": {"type": "json_schema", "json_schema": {"name": "_Plan", "schema": _Plan.model_json_schema(), "strict": True}}}
    with _scope():
        assert capture.observe_first_native_request(kwargs, label="research_manager", first=False,
                                                    provider="vllm", sdk_max_retries=5, timeout_seconds=10) is None
    with capture.capture_research_manager(ticker="005930.KS", trade_date="2026-09-23", market="domestic",
                                          trigger="manual", held=False, session_in_progress=True, historical=False):
        assert capture.observe_first_native_request(kwargs, label="research_manager", first=True,
                                                    provider="vllm", sdk_max_retries=5, timeout_seconds=10) is None
    assert not list(enabled.glob("*.request.json"))


@pytest.mark.asyncio
async def test_sdk_transport_body_matches_captured_generation_fields(enabled, monkeypatch):
    body = None

    def transport(request: httpx.Request) -> httpx.Response:
        nonlocal body
        body = json.loads(request.content)
        return httpx.Response(200, json={
            "id": "synthetic", "object": "chat.completion", "created": 1, "model": "fixture",
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": '{"value":"ok"}'}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        })

    client = LLMClient()
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(transport))
    client._client = AsyncOpenAI(api_key="fixture", base_url="http://synthetic.invalid/v1",
                                 http_client=http_client, max_retries=0)
    monkeypatch.setattr(client, "_record_metrics", lambda *args, **kwargs: None)
    try:
        with _scope():
            result = await client.structured_complete([{"role": "user", "content": "safe"}],
                                                      _Plan, label="research_manager")
    finally:
        await client._client.close()
    assert result.value == "ok"
    capture._writer.events.join()
    request = json.loads(next(enabled.glob("*.request.json")).read_text(encoding="utf-8"))["request"]
    assert body["model"] == request["model"]
    assert body["messages"] == request["messages"]
    assert body["temperature"] == request["temperature"]
    assert body["max_tokens"] == request["max_tokens"]
    assert body["response_format"] == request["response_format"]
    for key, value in request["extra_body"].items():
        assert body[key] == value


@pytest.mark.parametrize("content", [
    '{"Authorization":"Basic TEST_ONLY_NOT_A_REAL_CREDENTIAL"}',
    'Cookie: session=TEST_ONLY_NOT_A_REAL_CREDENTIAL',
    '{ "cOoKiE" : "session=TEST_ONLY_NOT_A_REAL_CREDENTIAL" }',
    'Authorization: Token TEST_ONLY_NOT_A_REAL_CREDENTIAL',
    '{"Authorization":"Digest TEST_ONLY_NOT_A_REAL_CREDENTIAL"}',
    'Proxy-Authorization: Synthetic TEST_ONLY_NOT_A_REAL_CREDENTIAL',
])
def test_header_secrets_are_rejected_in_request_and_response(enabled, content):
    assert _record_request(_kwargs(content)) is None
    capture._writer.events.join()
    assert not list(enabled.glob("*.request.json"))
    capture_id = _record_request()
    assert capture_id is not None
    response = {"choices": [{"message": {"content": content}, "finish_reason": "stop"}]}
    capture.observe_first_native_result(capture_id, response=response, elapsed_seconds=0.1)
    capture._writer.events.join()
    assert not (enabled / f"{capture_id}.result.json").exists()
    assert capture.health_snapshot()["drops"]["sensitive_content"] >= 2


def test_unexpected_header_kwargs_are_rejected_before_storage(enabled):
    kwargs = {**_kwargs(), "extra_headers": {"X-Synthetic": "TEST_ONLY_NOT_A_REAL_CREDENTIAL"}}
    assert _record_request(kwargs) is None
    capture._writer.events.join()
    assert not list(enabled.glob("*.request.json"))
    assert capture.health_snapshot()["drops"]["request_fields_invalid"] == 1


def _sdk_response(*, top_extra=None, message_extra=None):
    message = {"role": "assistant", "content": '{"value":"ok"}', **(message_extra or {})}
    return ChatCompletion.model_validate({
        "id": "synthetic", "object": "chat.completion", "created": 1, "model": "fixture",
        "choices": [{"index": 0, "finish_reason": "stop", "message": message}],
        **(top_extra or {}),
    })


@pytest.mark.parametrize("location", ["top", "nested"])
def test_sdk_extra_fields_are_bounded_before_model_dump(enabled, monkeypatch, location):
    response = (_sdk_response(top_extra={"vendor_trace": [0] * 25000}) if location == "top"
                else _sdk_response(message_extra={"vendor_trace": [0] * 25000}))
    capture_id = _record_request()
    assert capture_id is not None
    copied = []

    def tracked_copy(value):
        copied.append(True)
        return value.model_dump(mode="json")

    monkeypatch.setattr(capture, "_as_json", tracked_copy)
    capture.observe_first_native_result(capture_id, response=response, elapsed_seconds=0.1)
    capture._writer.events.join()
    assert copied == []
    assert not (enabled / f"{capture_id}.result.json").exists()
    assert capture.health_snapshot()["drops"]["response_limit"] == 1


def test_small_sdk_extra_field_still_captures(enabled):
    capture_id = _record_request()
    assert capture_id is not None
    capture.observe_first_native_result(
        capture_id, response=_sdk_response(top_extra={"vendor_trace": [0, 1]}), elapsed_seconds=0.1)
    capture.observe_first_native_parse(capture_id)
    capture._writer.events.join()
    assert (enabled / f"{capture_id}.result.json").exists()
    assert build_manifest(enabled)["complete_count"] == 1


def test_response_limit_precedes_model_dump_and_first_status_io_is_off_thread(enabled, monkeypatch):
    main_thread = threading.current_thread()
    main_fsync = []
    original_fsync = os.fsync

    def checked_fsync(fd):
        if threading.current_thread() is main_thread:
            main_fsync.append(True)
        return original_fsync(fd)

    monkeypatch.setattr(capture.os, "fsync", checked_fsync)
    capture_id = _record_request()
    assert capture_id is not None
    assert main_fsync == []

    class OversizedResponse:
        def __init__(self):
            self.payload = "X" * (8 * 1024 * 1024)

        def model_dump(self, *, mode):
            raise AssertionError("oversized response was copied")

    capture.observe_first_native_result(capture_id, response=OversizedResponse(), elapsed_seconds=0.1)
    capture._writer.events.join()
    assert not (enabled / f"{capture_id}.result.json").exists()
    assert capture.health_snapshot()["drops"]["response_limit"] == 1


def test_timeout_forms_are_explicit_and_unsupported_is_distinct(enabled):
    for timeout, expected in ((10, "scalar"), (httpx.Timeout(10, connect=2), "phases"), (None, "none")):
        capture_id = _record_request(timeout=timeout)
        assert capture_id is not None
        capture._writer.events.join()
        event = json.loads((enabled / f"{capture_id}.request.json").read_text(encoding="utf-8"))
        assert event["sdk_timeout"]["kind"] == expected
        row = next(row for row in build_manifest(enabled)["captures"] if row["capture_id"] == capture_id)
        assert "sdk_metadata_invalid" not in row["problems"]
        if expected == "phases":
            assert event["sdk_timeout"]["connect"] == 2
            assert event["sdk_timeout"]["read"] == 10
    assert _record_request(timeout=object()) is None
    capture._writer.events.join()
    assert capture.health_snapshot()["drops"]["unsupported_timeout"] == 1


def test_missing_response_fields_are_not_complete(enabled):
    capture_id = _record_complete()
    path = enabled / f"{capture_id}.result.json"
    event = json.loads(path.read_text(encoding="utf-8"))
    event["raw_response"]["choices"][0]["message"] = {}
    event["raw_response"]["choices"][0].pop("finish_reason")
    event["raw_response_sha256"] = capture.hashlib.sha256(capture._canonical(event["raw_response"])).hexdigest()
    path.write_text(json.dumps(event), encoding="utf-8")
    assert "response_incomplete" in build_manifest(enabled)["captures"][0]["problems"]
    event["raw_response"]["choices"][0] = {"message": {"content": None}, "finish_reason": "stop"}
    event["raw_response_sha256"] = capture.hashlib.sha256(capture._canonical(event["raw_response"])).hexdigest()
    path.write_text(json.dumps(event), encoding="utf-8")
    parse = enabled / f"{capture_id}.parse.json"
    parsed = json.loads(parse.read_text(encoding="utf-8"))
    parsed["valid"] = False
    parsed["error_class"] = "ParseFailure"
    parse.write_text(json.dumps(parsed), encoding="utf-8")
    assert build_manifest(enabled)["complete_count"] == 1


def test_surrogate_is_isolated_to_bad_capture(enabled):
    good_id = _record_complete()
    bad_id = _record_complete()
    path = enabled / f"{bad_id}.request.json"
    event = json.loads(path.read_text(encoding="utf-8"))
    event["ticker"] = "\ud800"
    path.write_text(json.dumps(event, ensure_ascii=True), encoding="utf-8")
    manifest = build_manifest(enabled)
    assert manifest["request_count"] == 2
    assert manifest["complete_count"] == 1
    assert next(row for row in manifest["captures"] if row["capture_id"] == bad_id)["problems"] == ["request_unreadable"]
    assert next(row for row in manifest["captures"] if row["capture_id"] == good_id)["complete"] is True


def test_surrogate_in_response_or_metadata_is_rejected_but_pair_is_valid(enabled):
    first = _record_complete()
    second = _record_complete()
    response = enabled / f"{second}.result.json"
    original = response.read_bytes()
    event = json.loads(original)
    event["raw_response"]["choices"][0]["message"]["content"] = "\ud800"
    response.write_text(json.dumps(event, ensure_ascii=True), encoding="utf-8")
    assert build_manifest(enabled)["complete_count"] == 1
    assert "response_unreadable" in next(row for row in build_manifest(enabled)["captures"]
                                        if row["capture_id"] == second)["problems"]
    response.write_bytes(original)
    request = enabled / f"{second}.request.json"
    event = json.loads(request.read_text(encoding="utf-8"))
    event["provenance"]["\ud800"] = "synthetic"
    request.write_text(json.dumps(event, ensure_ascii=True), encoding="utf-8")
    assert build_manifest(enabled)["complete_count"] == 1
    assert next(row for row in build_manifest(enabled)["captures"] if row["capture_id"] == first)["complete"]
    paired = enabled / "unicode-pair.json"
    paired.write_text('{"emoji":"\\ud83d\\ude00"}', encoding="utf-8")
    from scripts.aa_eval.validate import _load_event
    assert _load_event(paired)["emoji"] == "😀"


def test_writer_failure_stops_later_captures(enabled, monkeypatch):
    original_replace = os.replace
    failed = False

    def fail_first_request(src, dst):
        nonlocal failed
        if str(dst).endswith(".request.json") and not failed:
            failed = True
            raise OSError("synthetic disk failure")
        return original_replace(src, dst)

    monkeypatch.setattr(capture.os, "replace", fail_first_request)
    first = _record_request()
    assert first is not None
    capture._writer.events.join()
    assert failed
    assert _record_request() is None
    capture._writer.events.join()
    assert not list(enabled.glob("*.request.json"))
    snapshot = capture.health_snapshot()
    assert snapshot["eligible_attempts"] == 2
    assert snapshot["drops"]["write_failed"] == 1
    assert snapshot["drops"]["capture_stopped"] >= 1
    assert snapshot["stopped_reason"] == "write_failed"


def _stalled_writer(enabled, monkeypatch):
    original_start = threading.Thread.start
    monkeypatch.setattr(threading.Thread, "start", lambda self: None)
    try:
        writer = capture._Writer(enabled)
    finally:
        monkeypatch.setattr(threading.Thread, "start", original_start)
    monkeypatch.setattr(capture, "_writer", writer)
    return writer


def _await_drain(writer):
    deadline = time.monotonic() + 3
    while (writer.events.unfinished_tasks or writer.control.unfinished_tasks) and time.monotonic() < deadline:
        time.sleep(0.01)
    assert writer.events.unfinished_tasks == 0
    assert writer.control.unfinished_tasks == 0
    assert writer.queued_bytes == 0


def test_stopped_writer_drains_current_data_and_control_entries(enabled, monkeypatch):
    writer = _stalled_writer(enabled, monkeypatch)
    assert _record_request() is not None
    original_put = writer.events.put_nowait
    monkeypatch.setattr(writer.events, "put_nowait", lambda entry: (_ for _ in ()).throw(queue.Full))
    writer.reject("a" * 32, "request", "synthetic")
    monkeypatch.setattr(writer.events, "put_nowait", original_put)
    writer._halt("queue_limit")
    writer.worker.start()
    _await_drain(writer)
    assert writer.worker.is_alive()
    assert not list(enabled.glob("*.request.json"))


def test_failed_write_drains_different_sized_queued_events(enabled, monkeypatch):
    writer = _stalled_writer(enabled, monkeypatch)
    assert _record_request(_kwargs("short")) is not None
    assert _record_request(_kwargs("longer" * 100)) is not None
    original_replace = os.replace
    failed = []

    def fail_first_request(src, dst):
        if str(dst).endswith(".request.json") and not failed:
            failed.append(True)
            raise OSError("synthetic disk failure")
        return original_replace(src, dst)

    monkeypatch.setattr(capture.os, "replace", fail_first_request)
    writer.worker.start()
    _await_drain(writer)
    assert failed
    assert writer.worker.is_alive()
    assert writer.drops["write_failed"] == 1
    assert not list(enabled.glob("*.request.json"))


def test_initial_status_failure_drains_accepted_event(enabled, monkeypatch):
    writer = _stalled_writer(enabled, monkeypatch)
    assert _record_request() is not None
    original_status = writer._write_status
    failures = []

    def fail_first_status():
        if not failures:
            failures.append(True)
            return False
        return original_status()

    monkeypatch.setattr(writer, "_write_status", fail_first_status)
    writer.worker.start()
    _await_drain(writer)
    assert writer.worker.is_alive()
    assert writer.stopped_reason == "status_unavailable"
    assert not list(enabled.glob("*.request.json"))


def test_queue_limit_halts_before_next_request_copy(enabled, monkeypatch):
    monkeypatch.setattr(settings, "rm_request_capture_queue_max_bytes", 1)
    assert _record_request() is None
    assert capture._writer.is_stopped()

    class NoCopy(dict):
        def items(self):
            raise AssertionError("stopped recorder traversed request")

    assert _record_request(NoCopy(_kwargs())) is None
    snapshot = capture.health_snapshot()
    assert snapshot["eligible_attempts"] == 2
    assert snapshot["drops"]["queue_limit"] == 1


def test_control_queue_saturation_stops_collection(enabled, monkeypatch):
    writer = capture._get_writer()

    def full(_entry):
        raise queue.Full

    monkeypatch.setattr(writer.control, "put_nowait", full)
    writer.note_eligible()
    assert writer.is_stopped()
    assert capture.health_snapshot()["stopped_reason"] == "control_queue_limit"


def test_session_reconciles_all_event_files(enabled):
    capture_id = _record_complete()
    writer = capture._writer
    writer.close()
    result = enabled / f"{capture_id}.result.json"
    original = result.read_bytes()
    assert build_manifest(enabled)["session_complete"] is True
    result.unlink()
    assert build_manifest(enabled)["session_complete"] is False
    result.write_bytes(original)
    assert build_manifest(enabled)["session_complete"] is True
    parse = enabled / f"{capture_id}.parse.json"
    original_parse = parse.read_bytes()
    parse.unlink()
    assert build_manifest(enabled)["session_complete"] is False
    parse.write_bytes(original_parse)
    assert build_manifest(enabled)["session_complete"] is True
    orphan = enabled / f"{'f' * 32}.result.json"
    event = json.loads(original)
    event["capture_id"] = "f" * 32
    orphan.write_text(json.dumps(event), encoding="utf-8")
    assert build_manifest(enabled)["session_complete"] is False
    orphan.unlink()
    orphan_parse = enabled / f"{'f' * 32}.parse.json"
    parse_event = json.loads(original_parse)
    parse_event["capture_id"] = "f" * 32
    orphan_parse.write_text(json.dumps(parse_event), encoding="utf-8")
    assert build_manifest(enabled)["session_complete"] is False
    orphan_parse.unlink()
    rejected = enabled / f"{'e' * 32}.request.rejected.json"
    rejected.write_text(json.dumps({"schema_version": capture._VERSION, "kind": "rejected",
                                    "capture_id": "e" * 32, "capture_session_id": writer.session_id,
                                    "stage": "request"}), encoding="utf-8")
    assert build_manifest(enabled)["session_complete"] is False
    status_path = enabled / f".capture-session-{writer.session_id}.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status["recorded_by_kind"]["rejected"] = 1
    status["recorded_events"] += 1
    status_path.write_text(json.dumps(status), encoding="utf-8")
    assert build_manifest(enabled)["session_complete"] is False


def test_shutdown_closes_completed_session_and_stays_idempotent(enabled):
    capture_id = _record_complete()
    writer = capture._writer
    assert capture.finalize_capture() is True
    assert capture.finalize_capture() is True
    assert writer.closed is True
    assert capture.health_snapshot()["active"] is False
    assert build_manifest(enabled)["session_complete"] is True
    assert (enabled / f"{capture_id}.request.json").exists()


def test_shutdown_waits_for_inflight_result_and_parse(enabled):
    capture_id = _record_request()
    writer = capture._writer
    finished = []
    thread = threading.Thread(target=lambda: finished.append(capture.finalize_capture()))
    thread.start()
    deadline = time.monotonic() + 1
    while not writer.closing and time.monotonic() < deadline:
        time.sleep(0.01)
    assert writer.closing
    assert _record_request() is None
    capture.observe_first_native_result(capture_id, response=_Response(), elapsed_seconds=0.1)
    capture.observe_first_native_parse(capture_id)
    thread.join(timeout=2)
    assert finished == [True]
    assert writer.stopped_reason is None
    assert writer.queued_bytes == 0
    assert writer.eligible_attempts == 1
    assert writer.drops == {}
    assert build_manifest(enabled)["session_complete"] is True


def test_shutdown_preserves_inflight_result_and_parse(enabled):
    capture_id = _record_request()
    writer = capture._writer
    finished = []
    thread = threading.Thread(target=lambda: finished.append(capture.finalize_capture()))
    thread.start()
    deadline = time.monotonic() + 1
    while not writer.closing and time.monotonic() < deadline:
        time.sleep(0.01)
    assert writer.closing
    capture.observe_first_native_result(capture_id, response=_Response(), elapsed_seconds=0.1)
    capture.observe_first_native_parse(capture_id)
    thread.join(timeout=2)
    assert finished == [True]
    assert build_manifest(enabled)["session_complete"] is True


def test_shutdown_with_unfinished_request_stays_incomplete(enabled, monkeypatch):
    capture_id = _record_request()
    writer = capture._writer
    writer.events.join()
    clock = [0.0]
    monkeypatch.setattr(capture.time, "monotonic", lambda: clock[0])
    original_wait = writer.pending_done.wait

    def expire(_seconds):
        clock[0] = 6.0
        return original_wait(0)

    monkeypatch.setattr(writer.pending_done, "wait", expire)
    assert capture.finalize_capture() is False
    assert writer.stopped_reason == "drain_timeout"
    assert build_manifest(enabled)["session_complete"] is False
    assert (enabled / f"{capture_id}.request.json").exists()


def test_shutdown_status_failure_is_not_reported_as_closed(enabled, monkeypatch):
    _record_complete()
    writer = capture._writer
    original_write = writer._write_status
    monkeypatch.setattr(writer, "_write_status", lambda: False)
    assert capture.finalize_capture() is False
    monkeypatch.setattr(writer, "_write_status", original_write)
    assert writer.stopped_reason == "status_unavailable"
    assert build_manifest(enabled)["session_complete"] is False


def test_shutdown_before_first_request_blocks_late_writer(enabled):
    assert capture._writer is None
    assert capture.finalize_capture() is True
    assert _record_request() is None
    assert capture._writer is None
    assert list(enabled.iterdir()) == []


def test_shutdown_waits_for_request_registration_boundary(enabled, monkeypatch):
    entered = threading.Event()
    release = threading.Event()
    original_get = capture._get_writer

    def gated_get():
        if threading.current_thread().name == "Registrant":
            entered.set()
            assert release.wait(timeout=3)
        return original_get()

    monkeypatch.setattr(capture, "_get_writer", gated_get)
    registered = []
    requester = threading.Thread(target=lambda: registered.append(_record_request()),
                                 name="Registrant")
    requester.start()
    assert entered.wait(timeout=3)
    finished = []
    closer = threading.Thread(target=lambda: finished.append(capture.finalize_capture()))
    closer.start()
    assert capture._writer_shutdown is False
    release.set()
    requester.join(timeout=3)
    assert len(registered) == 1 and registered[0] is not None
    capture.observe_first_native_result(registered[0], response=_Response(), elapsed_seconds=0.1)
    capture.observe_first_native_parse(registered[0])
    closer.join(timeout=3)
    assert finished == [True]
    assert capture._writer.eligible_attempts == 1
    assert build_manifest(enabled)["session_complete"] is True


def test_closed_session_excludes_late_request_even_if_status_io_fails(enabled, monkeypatch):
    _record_complete()
    writer = capture._writer
    writer.control.join()
    blocked = threading.Event()
    release = threading.Event()
    original_open = capture._open_private

    def gated_open(path):
        if threading.current_thread().name == "CloseStatusProbe":
            blocked.set()
            assert release.wait(timeout=3)
        return original_open(path)

    monkeypatch.setattr(capture, "_open_private", gated_open)
    finished = []
    closer = threading.Thread(target=lambda: finished.append(capture.finalize_capture()),
                              name="CloseStatusProbe")
    closer.start()
    assert blocked.wait(timeout=3)
    late_result = []
    late_request = threading.Thread(target=lambda: late_result.append(_record_request()),
                                    name="LateRequestProbe")
    late_request.start()
    late_request.join(timeout=3)
    assert late_result == [None]
    release.set()
    closer.join(timeout=3)
    assert finished == [True]
    monkeypatch.setattr(capture, "_open_private", lambda _path: (_ for _ in ()).throw(OSError("synthetic")))
    assert _record_request() is None
    assert capture.finalize_capture() is True
    writer.control.join()
    status = json.loads(writer.status_path.read_text(encoding="utf-8"))
    assert status["state"] == "closed"
    assert status["eligible_attempts"] == 1
    assert status["drops"] == {}
    assert writer.eligible_attempts == 1
    assert writer.status_io_failed is False
    assert build_manifest(enabled)["session_complete"] is True
