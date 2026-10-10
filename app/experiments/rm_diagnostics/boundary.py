"""Stop before app parsing, including on recorder failures. 파싱 직전 진단 종료."""

from __future__ import annotations

import asyncio
import logging
import inspect
import os
import socket
import subprocess
import time
from contextlib import contextmanager
from typing import Any

from .evidence import EVENT_LIMIT, Refusal, canonical, checked_json, digest

REQUEST_FIELDS = frozenset({"model", "messages", "temperature", "extra_body", "max_tokens", "response_format"})


class DiagnosticBoundaryStop(BaseException):
    """Cannot be mistaken for a parse failure and trigger fallback. 폴백 방지 종료."""


class OfflineIOViolation(BaseException):
    """Distinct from normal diagnostic completion. 오프라인 위반은 정상 종료와 구별."""


@contextmanager
def quiet_logs():
    previous = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    try:
        yield
    finally:
        logging.disable(previous)


@contextmanager
def offline_io():
    """Forbid network/process I/O during preparation and inspection. 오프라인 입출력 차단."""
    originals = [(socket, "getaddrinfo", socket.getaddrinfo),
                 (socket.socket, "connect", socket.socket.connect),
                 (socket.socket, "connect_ex", socket.socket.connect_ex),
                 (subprocess, "Popen", subprocess.Popen), (os, "system", os.system)]

    def denied(*args, **kwargs):
        raise OfflineIOViolation()

    original_connect = socket.socket.connect

    def connect(sock, address):
        frame = inspect.currentframe().f_back
        internal = (frame.f_code.co_filename == socket.__file__ and
                    frame.f_code.co_name in ("_fallback_socketpair", "_socketpair", "socketpair"))
        if internal and isinstance(address, tuple) and address[0] in ("127.0.0.1", "::1"):
            return original_connect(sock, address)
        raise OfflineIOViolation()

    try:
        for owner, name, _ in originals:
            setattr(owner, name, connect if owner is socket.socket and name == "connect" else denied)
        yield
    finally:
        for owner, name, original in originals:
            setattr(owner, name, original)


def response_snapshot(response: Any) -> dict[str, Any]:
    from app.llm.request_capture import _preflight_response

    try:
        _preflight_response(response, EVENT_LIMIT)
        raw = response.model_dump(mode="json")
        fields_set = {"top": sorted(response.model_fields_set), "choices": [
            {"choice": sorted(choice.model_fields_set), "message": sorted(choice.message.model_fields_set)}
            for choice in response.choices]}
        result = {"raw_response": raw, "model_fields_set": fields_set,
                  "raw_response_canonical_sha256": digest(canonical(raw))}
        checked_json(result)
        return result
    except Refusal:
        raise
    except Exception:
        raise Refusal("response_limit_or_type") from None


class Boundary:
    def __init__(self, evidence, *, transmit: bool = False, expected_sha: str | None = None):
        self.evidence, self.transmit, self.expected_sha = evidence, transmit, expected_sha
        self.entries = self.started = self.completed = 0
        self.reason = "not_started"
        self.original = None

    async def create(self, *args, **kwargs):
        self.entries += 1
        if self.entries != 1:
            self.reason = "second_sdk_call"
            raise DiagnosticBoundaryStop()
        try:
            if args or set(kwargs) != REQUEST_FIELDS:
                raise Refusal("request_fields_changed")
            shape = kwargs["response_format"]
            if (shape.get("type") != "json_schema" or
                    shape.get("json_schema", {}).get("name") != "FlatResearchPlan" or
                    shape.get("json_schema", {}).get("strict") is not True):
                raise Refusal("not_first_native_rm")
            request_sha = digest(checked_json(kwargs))
            if self.expected_sha is not None and request_sha != self.expected_sha:
                raise Refusal("request_drift")
            self.evidence.write("request.json", kwargs)
            if not self.transmit:
                self.reason = "prepared"
                raise DiagnosticBoundaryStop()
            self.started = 1
            started = time.monotonic()
            try:
                response = await self.original(**kwargs)
            except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
                self.reason = "cancelled"
                raise
            except Exception as error:
                code = getattr(error, "status_code", None)
                self.evidence.write("error.json", {"error_class": type(error).__name__,
                    "http_status": code if type(code) is int and 100 <= code <= 599 else None})
                self.reason = "sdk_error"
                raise DiagnosticBoundaryStop() from None
            self.completed = 1
            elapsed = time.monotonic() - started
            snapshot = response_snapshot(response)
            snapshot["elapsed_seconds"] = elapsed
            snapshot["elapsed_definition"] = "sdk_callable_wall_seconds_including_sdk_retries"
            self.evidence.write("response.json", snapshot)
            self.reason = "response_saved"
        except DiagnosticBoundaryStop:
            raise
        except Refusal as error:
            self.reason = str(error)
        except Exception:
            self.reason = "observer_failed"
        raise DiagnosticBoundaryStop() from None

    @contextmanager
    def installed(self, llm_client):
        resource = llm_client._client.chat.completions
        had = "create" in vars(resource)
        own = vars(resource).get("create")
        self.original = resource.create
        resource.create = self.create
        try:
            yield
        finally:
            if had:
                resource.create = own
            else:
                del resource.__dict__["create"]


async def invoke_rm(inp, client, boundary: Boundary) -> None:
    from app.agents import research_manager
    from app.config import account_mode_scope, backtest_scope

    if research_manager.llm_client is not client:
        raise Refusal("rm_client_binding_changed")
    with boundary.installed(client), quiet_logs(), account_mode_scope("real"), backtest_scope():
        try:
            await research_manager.synthesize(inp.reports_dict(), inp.research_debate(),
                language=inp.language, identity=inp.identity_note, held=inp.held,
                session_note=inp.session_note, period_note=inp.period_note,
                position_context=inp.position_context)
        except DiagnosticBoundaryStop:
            return
    raise Refusal("rm_did_not_stop_at_boundary")


def inspect_response(snapshot: dict[str, Any]) -> dict[str, Any]:
    from app.llm.shared import _clean_json_payload
    from app.state import FlatResearchPlan

    result = {"inspection_origin": "offline_reconstruction", "schema_valid": False,
              "raw_response_canonical_sha256": snapshot["raw_response_canonical_sha256"]}
    raw_response = snapshot["raw_response"]
    try:
        message = raw_response["choices"][0]["message"]
        fields = snapshot["model_fields_set"]["choices"][0]["message"]
        content = message.get("content")
        result["content_state"] = "missing" if "content" not in fields else (
            "null" if content is None else "empty" if content == "" else "value")
        result["reasoning_fields"] = {key: "missing" if key not in fields else (
            "null" if message.get(key) is None else "empty" if message.get(key) == "" else "value")
            for key in ("reasoning", "reasoning_content")}
        with quiet_logs():
            cleaned = _clean_json_payload(content or "", label="research_manager", path="native")
            result["cleaned_content"] = cleaned
            result["wrapper_changed"] = cleaned != (content or "")
            parsed = FlatResearchPlan.model_validate_json(cleaned)
        result.update(schema_valid=True, parsed=parsed.model_dump(mode="json"), error_class=None)
    except Exception as error:
        result["error_class"] = type(error).__name__
    checked_json(result)
    return result
