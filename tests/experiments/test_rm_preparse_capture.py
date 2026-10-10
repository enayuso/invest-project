"""Synthetic SDK-boundary and evidence tests. 실제 LLM/DB/네트워크 접근 금지."""

from __future__ import annotations

import asyncio
import ctypes
import json
import inspect
import logging
import os
import socket
import subprocess
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion

from app.experiments.rm_diagnostics import evidence as storage
from app.experiments.rm_diagnostics.boundary import (
    Boundary, DiagnosticBoundaryStop, inspect_response, invoke_rm, quiet_logs, response_snapshot,
)
from app.experiments.rm_diagnostics.evidence import Evidence, Refusal, canonical, digest, verify_bundle
from replay_fixtures import install_db_block, make_inputs


@pytest.fixture(autouse=True)
def deny_external(monkeypatch):
    install_db_block(monkeypatch)
    from app.experiments.replay import safety

    def deny(*args, **kwargs):
        raise AssertionError("external access attempted in synthetic RM test")

    monkeypatch.setattr(safety, "_ORIG_CONNECT", deny)
    monkeypatch.setattr(socket, "getaddrinfo", deny)
    original_connect = socket.socket.connect

    def blocked_connect(sock, address):
        frame = inspect.currentframe().f_back
        if frame.f_code.co_name == "connect" and frame.f_code.co_filename.endswith("rm_diagnostics\\boundary.py"):
            frame = frame.f_back
        # Windows asyncio builds an internal socketpair over loopback.
        # Windows asyncio 자체 socketpair 생성만 허용하고 앱 연결은 차단한다.
        if (frame.f_code.co_filename == socket.__file__ and
                frame.f_code.co_name in ("_fallback_socketpair", "_socketpair", "socketpair") and
                address[0] in ("127.0.0.1", "::1")):
            return original_connect(sock, address)
        return deny(sock, address)

    monkeypatch.setattr(socket.socket, "connect", blocked_connect)
    monkeypatch.setattr(subprocess, "Popen", deny)
    monkeypatch.setattr(os, "system", deny)


def protect_windows(path: Path, *, public=False):
    # Change only the newly created synthetic directory. 합성 임시 폴더 DACL만 설정.
    from ctypes import wintypes as w

    adv = ctypes.WinDLL("advapi32", use_last_error=True)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = w.HANDLE
    adv.OpenProcessToken.argtypes = [w.HANDLE, w.DWORD, ctypes.POINTER(w.HANDLE)]
    adv.GetTokenInformation.argtypes = [w.HANDLE, ctypes.c_int, ctypes.c_void_p, w.DWORD, ctypes.POINTER(w.DWORD)]
    adv.ConvertSidToStringSidW.argtypes = [ctypes.c_void_p, ctypes.POINTER(w.LPWSTR)]
    adv.ConvertStringSecurityDescriptorToSecurityDescriptorW.argtypes = [w.LPCWSTR, w.DWORD,
        ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
    adv.SetFileSecurityW.argtypes = [w.LPCWSTR, w.DWORD, ctypes.c_void_p]
    kernel.LocalFree.argtypes = [ctypes.c_void_p]
    kernel.CloseHandle.argtypes = [w.HANDLE]
    token = w.HANDLE()
    assert adv.OpenProcessToken(kernel.GetCurrentProcess(), 8, ctypes.byref(token))
    descriptor = ctypes.c_void_p()
    sid = w.LPWSTR()
    try:
        needed = w.DWORD()
        adv.GetTokenInformation(token, 1, None, 0, ctypes.byref(needed))
        data = ctypes.create_string_buffer(needed.value)
        assert adv.GetTokenInformation(token, 1, data, needed, ctypes.byref(needed))
        assert adv.ConvertSidToStringSidW(ctypes.cast(data, ctypes.POINTER(ctypes.c_void_p))[0], ctypes.byref(sid))
        sddl = f"D:P(A;OICI;FA;;;{sid.value})(A;OICI;FA;;;SY)(A;OICI;FA;;;BA)"
        if public:
            sddl += "(A;OICI;FR;;;WD)"
        assert adv.ConvertStringSecurityDescriptorToSecurityDescriptorW(sddl, 1, ctypes.byref(descriptor), None)
        assert adv.SetFileSecurityW(str(path), 0x80000004, descriptor)
    finally:
        kernel.CloseHandle(token)
        kernel.LocalFree(ctypes.cast(sid, ctypes.c_void_p))
        kernel.LocalFree(descriptor)


@pytest.fixture
def private_root(tmp_path):
    if os.name == "nt":
        protect_windows(tmp_path)
    else:
        tmp_path.chmod(0o700)
    return tmp_path


def sdk_response(content="{}", *, content_present=True, reasoning="absent", empty_choices=False, extra=None):
    message = {"role": "assistant", "message_extension": {"unicode": "한글🙂"}}
    if content_present:
        message["content"] = content
    if reasoning != "absent":
        message["reasoning_content"] = reasoning
    if extra is not None:
        message["extra_blob"] = extra
    return ChatCompletion.model_validate({"id": "synthetic", "object": "chat.completion", "created": 1,
        "model": "synthetic", "top_extension": "top", "choices": [] if empty_choices else [
            {"index": 0, "finish_reason": "stop", "choice_extension": "choice", "message": message}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110}})


def request():
    return {"model": "synthetic", "messages": [{"role": "user", "content": "safe"}],
        "temperature": 0.0, "extra_body": {}, "max_tokens": 16000,
        "response_format": {"type": "json_schema", "json_schema": {
            "name": "FlatResearchPlan", "strict": True, "schema": {"type": "object"}}}}


@pytest.mark.asyncio
@pytest.mark.parametrize("payload", ["{}", "\n\nBased{\"rationale\":\"The model has synthesized\"}  ", " \n한글🙂\r\n"])
async def test_real_rm_request_and_response_stop_before_parsing(monkeypatch, private_root, payload):
    from app.agents import research_manager
    from app.llm.client import LLMClient
    client = LLMClient()
    await client._client.close()
    sent = []

    async def transport(req):
        sent.append(json.loads(req.content))
        return httpx.Response(200, json=sdk_response(payload, reasoning="returned reasoning").model_dump(mode="json"))

    client._client = AsyncOpenAI(api_key="synthetic", base_url="http://synthetic.invalid/v1",
        max_retries=5, http_client=httpx.AsyncClient(transport=httpx.MockTransport(transport)))
    monkeypatch.setattr(research_manager, "llm_client", client)
    evidence = Evidence.create(private_root)
    boundary = Boundary(evidence, transmit=True)
    original = client._client.chat.completions.create
    await invoke_rm(make_inputs(), client, boundary)
    assert boundary.reason == "response_saved"
    assert boundary.entries == boundary.started == boundary.completed == len(sent) == 1
    recorded = json.loads((evidence.directory / "request.json").read_text(encoding="utf-8"))
    # SDK flattens extra_body but receives the unchanged original kwargs.
    # extra_body 직렬화 규칙을 고려해 실제 전송과 SDK 경계 kwargs를 대조한다.
    sent_expected = {**recorded, **recorded["extra_body"]}
    del sent_expected["extra_body"]
    assert sent[0] == sent_expected
    assert recorded["response_format"]["json_schema"]["name"] == "FlatResearchPlan"
    from app.llm.language import language_instruction

    assert recorded["messages"][0]["content"].endswith(language_instruction("en"))
    stored = json.loads((evidence.directory / "response.json").read_text(encoding="utf-8"))
    assert stored["raw_response"]["choices"][0]["message"]["content"] == payload
    assert stored["raw_response"]["choices"][0]["message"]["reasoning_content"] == "returned reasoning"
    assert client._client.chat.completions.create == original
    assert client._limiter._in_flight == 0
    await client._client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ["sdk_error", "persist_failure", "timeout"])
async def test_real_rm_fault_propagation_releases_limiter_without_fallback(monkeypatch, private_root, fault):
    from app.agents import research_manager
    from app.llm.client import LLMClient

    client = LLMClient()
    await client._client.close()
    calls = []

    async def transport(req):
        calls.append(req)
        if fault == "sdk_error":
            return httpx.Response(500, json={"error": "synthetic upstream failure"})
        if fault == "timeout":
            await asyncio.sleep(10)
        return httpx.Response(200, json=sdk_response("{}").model_dump(mode="json"))

    client._client = AsyncOpenAI(api_key="synthetic", max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(transport)))
    monkeypatch.setattr(research_manager, "llm_client", client)
    e = Evidence.create(private_root)
    b = Boundary(e, transmit=True)
    if fault == "persist_failure":
        write = e.write

        def fail(name, value):
            if name == "response.json":
                raise Refusal("write_failed")
            write(name, value)

        monkeypatch.setattr(e, "write", fail)
    original = client._client.chat.completions.create
    if fault == "timeout":
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(invoke_rm(make_inputs(), client, b), timeout=0.05)
    else:
        await invoke_rm(make_inputs(), client, b)
    assert len(calls) == 1 and b.entries == 1
    assert b.reason == {"sdk_error": "sdk_error", "persist_failure": "write_failed", "timeout": "cancelled"}[fault]
    assert client._limiter._in_flight == 0 and client._client.chat.completions.create == original
    await client._client.close()


@pytest.mark.asyncio
async def test_prepare_does_not_call_sdk_and_restores_instance(private_root):
    calls = []

    async def real(**kwargs):
        calls.append(kwargs)
        raise AssertionError("prepare transmitted")

    resource = SimpleNamespace(create=real)
    client = SimpleNamespace(_client=SimpleNamespace(chat=SimpleNamespace(completions=resource)))
    b = Boundary(Evidence.create(private_root))
    with b.installed(client), pytest.raises(DiagnosticBoundaryStop):
        await resource.create(**request())
    assert b.reason == "prepared" and not calls and b.started == 0
    assert resource.create is real


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ["sdk", "write", "dump", "sensitive", "oversize", "drift", "shape"])
async def test_boundary_faults_never_return_response_or_retry(private_root, monkeypatch, fault):
    e = Evidence.create(private_root)
    calls = []

    async def original(**kwargs):
        calls.append(kwargs)
        if fault == "sdk":
            raise httpx.ConnectError("untrusted diagnostic text must not appear")
        if fault == "dump":
            return object()
        if fault == "sensitive":
            return sdk_response("bearer synthetic-sensitive-value")
        if fault == "oversize":
            return sdk_response(extra="한" * storage.EVENT_LIMIT)
        return sdk_response("{}")

    if fault == "write":
        write = e.write

        def broken(name, value):
            if name == "response.json":
                raise OSError("disk full")
            write(name, value)

        monkeypatch.setattr(e, "write", broken)
    b = Boundary(e, transmit=True, expected_sha="incorrect" if fault == "drift" else None)
    b.original = original
    kwargs = request()
    if fault == "shape":
        kwargs["stream"] = True
    with pytest.raises(DiagnosticBoundaryStop):
        await b.create(**kwargs)
    assert len(calls) == (0 if fault in ("drift", "shape") else 1)
    assert not (e.directory / "response.json").exists()
    assert b.reason != "response_saved"
    with pytest.raises(DiagnosticBoundaryStop):
        await b.create(**request())
    assert len(calls) <= 1 and b.reason == "second_sdk_call"


@pytest.mark.asyncio
async def test_sdk_internal_retry_distinct_from_boundary_call(private_root):
    attempts = []

    async def transport(req):
        attempts.append(req)
        if len(attempts) == 1:
            return httpx.Response(500, headers={"retry-after-ms": "1"}, json={"error": "synthetic"})
        return httpx.Response(200, json=sdk_response().model_dump(mode="json"))

    sdk = AsyncOpenAI(api_key="synthetic", max_retries=1,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(transport)))
    b = Boundary(Evidence.create(private_root), transmit=True)
    b.original = sdk.chat.completions.create
    with pytest.raises(DiagnosticBoundaryStop):
        await b.create(**request())
    assert len(attempts) == 2 and b.started == b.completed == b.entries == 1
    await sdk.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [asyncio.CancelledError, KeyboardInterrupt, SystemExit])
async def test_cancellation_is_not_diagnostic_success(private_root, error):
    async def original(**kwargs):
        raise error()

    b = Boundary(Evidence.create(private_root), transmit=True)
    b.original = original
    with pytest.raises(error):
        await b.create(**request())
    assert b.reason == "cancelled" and b.completed == 0


@pytest.mark.parametrize("content,present,reasoning,state", [
    (None, False, "absent", "missing"), (None, True, None, "null"),
    ("", True, "", "empty"), ("{}", True, "returned", "value")])
def test_presence_and_extra_preserved(content, present, reasoning, state):
    snap = response_snapshot(sdk_response(content, content_present=present, reasoning=reasoning))
    result = inspect_response(snap)
    assert result["content_state"] == state
    assert snap["raw_response"]["top_extension"] == "top"
    assert snap["raw_response"]["choices"][0]["choice_extension"] == "choice"
    assert snap["raw_response"]["choices"][0]["message"]["message_extension"] == {"unicode": "한글🙂"}
    assert result["reasoning_fields"]["reasoning_content"] == (
        "missing" if reasoning == "absent" else "null" if reasoning is None else "empty" if reasoning == "" else "value")


@pytest.mark.parametrize("wrapper", ["", "\n\nBased", "**"])
def test_offline_parsing_meta_and_wrapper_no_logs(wrapper, caplog):
    text = json.dumps({"recommendation": "Hold", "rationale": "The model has synthesized the provided reports.",
                       "strategic_actions": "Stay out."})
    snapshot = response_snapshot(sdk_response(wrapper + text))
    original = canonical(snapshot)
    with caplog.at_level(logging.WARNING):
        result = inspect_response(snapshot)
    assert result["schema_valid"] is True
    assert result["parsed"]["rationale"].startswith("The model")
    assert result["wrapper_changed"] == bool(wrapper)
    assert canonical(snapshot) == original and not caplog.records


@pytest.mark.parametrize("content", ["invalid", "", None, "{\"recommendation\":\"Sell\"}"])
def test_bad_json_is_retained_without_fallback(content):
    result = inspect_response(response_snapshot(sdk_response(content)))
    assert not result["schema_valid"] and result["error_class"]


def test_empty_choices_and_length_reason_are_evidence():
    assert not inspect_response(response_snapshot(sdk_response(empty_choices=True)))["schema_valid"]
    response = sdk_response("partial")
    response.choices[0].finish_reason = "length"
    assert response_snapshot(response)["raw_response"]["choices"][0]["finish_reason"] == "length"


def test_immutable_files_and_partial_run(private_root):
    e = Evidence.create(private_root)
    e.write("request.json", request())
    original = (e.directory / "request.json").read_bytes()
    with pytest.raises(Refusal, match="file_collision"):
        e.write("request.json", {"different": True})
    assert (e.directory / "request.json").read_bytes() == original
    with pytest.raises(Refusal):
        verify_bundle(e.directory)


@pytest.mark.parametrize("mutation", ["content", "missing", "orphan", "false_complete"])
def test_manifest_detects_corrupt_and_missing_evidence(private_root, mutation):
    e = Evidence.create(private_root)
    e.write("request.json", request())
    e.write("provenance.json", {"synthetic": True})
    e.finish(phase="prepare", status="prepared", capture_complete=False,
             sdk_calls_started=0, sdk_calls_completed=0)
    if mutation == "content":
        (e.directory / "request.json").write_bytes(canonical({"changed": True}))
    elif mutation == "missing":
        (e.directory / "request.json").unlink()
    elif mutation == "orphan":
        (e.directory / "orphan").write_text("extra")
    else:
        path = e.directory / "manifest.json"
        value = json.loads(path.read_bytes())
        value.update(phase="capture", capture_complete=True, sdk_calls_started=1, sdk_calls_completed=1,
                     invariants_status="PASS", safety_trips=0)
        path.write_bytes(canonical(value))
    with pytest.raises(Refusal):
        verify_bundle(e.directory)


def test_unicode_limit_sensitive_cycle_and_unsupported_types():
    with pytest.raises(Refusal):
        storage.checked_json({"text": "한" * 40}, limit=100)
    with pytest.raises(Refusal, match="rejected_sensitive"):
        storage.checked_json({"text": "bearer synthetic-sensitive"})
    for value in ({"nan": float("nan")}, {"object": object()}):
        with pytest.raises(Refusal):
            storage.checked_json(value)
    cycle = []
    cycle.append(cycle)
    with pytest.raises(Refusal):
        storage.checked_json(cycle)


def test_private_acl_real_pass_and_public_rejection(private_root):
    storage.verify_private(private_root)
    if os.name == "nt":
        protect_windows(private_root, public=True)
    else:
        private_root.chmod(0o755)
    with pytest.raises(Refusal, match="not_private"):
        storage.verify_private(private_root)


def test_source_relative_sync_and_forbidden_directories(private_root):
    with pytest.raises(Refusal):
        storage.verify_private(Path("relative"))
    with pytest.raises(Refusal, match="source_directory"):
        storage.verify_private(private_root, forbidden=(private_root,))
    shared = private_root / "OneDrive"
    shared.mkdir(mode=0o700)
    with pytest.raises(Refusal, match="shared_directory"):
        storage.verify_private(shared)
    (private_root / ".git").write_text("synthetic")
    with pytest.raises(Refusal, match="source_directory"):
        storage.verify_private(private_root)


def test_symlink_rejected(private_root):
    linked = private_root / "linked"
    try:
        linked.symlink_to(private_root, target_is_directory=True)
    except OSError:
        pytest.skip("OS does not permit symlink creation")
    with pytest.raises(Refusal, match="reparse_directory"):
        storage.verify_private(linked)


def test_reparse_flag_rejected_without_resolving_link(private_root, monkeypatch):
    real_lstat = Path.lstat

    def flagged(path):
        result = real_lstat(path)
        if path == private_root:
            return SimpleNamespace(st_mode=result.st_mode, st_file_attributes=0x400)
        return result

    monkeypatch.setattr(Path, "lstat", flagged)
    with pytest.raises(Refusal, match="reparse_directory"):
        storage.verify_private(private_root)


def test_quiet_logs_restores_previous_threshold():
    previous = logging.root.manager.disable
    with quiet_logs():
        assert logging.root.manager.disable == logging.CRITICAL
    assert logging.root.manager.disable == previous
