"""Offline lifecycle oracle checks; no process or signal is started."""

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/aa_eval/capture_lifecycle_probe.py"
spec = importlib.util.spec_from_file_location("synthetic_lifecycle_probe_test", SCRIPT)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def test_closed_status_without_result_and_parse_is_rejected(tmp_path, monkeypatch):
    session_id = "a" * 32
    capture_id = "b" * 32
    output = tmp_path / "new-probe"

    class FakeChild:
        pid = 1234
        returncode = None

        def __init__(self, *args, **kwargs):
            request = b"synthetic request"
            (output / f"{capture_id}.request.json").write_bytes(request)
            (output / "ready-term.json").write_text(json.dumps({
                "session_id": session_id, "capture_id": capture_id,
                "request_sha256": hashlib.sha256(request).hexdigest(),
                "pending_captures": 1}), encoding="utf-8")
            (output / f".capture-session-{session_id}.json").write_text(json.dumps({
                "session_id": session_id, "state": "closed", "eligible_attempts": 1,
                "recorded_events": 3,
                "recorded_by_kind": {"request": 1, "result": 1, "parse": 1, "rejected": 0},
                "drops": {}}), encoding="utf-8")

        def poll(self):
            return self.returncode

        def wait(self, timeout):
            self.returncode = 0

    monkeypatch.setattr(probe.subprocess, "Popen", FakeChild)
    monkeypatch.setattr(probe.signal, "SIGKILL", 9, raising=False)
    monkeypatch.setattr(probe.os, "kill", lambda *args: None)
    monkeypatch.setattr(probe, "_validator", lambda repo: SimpleNamespace(
        build_manifest=lambda directory: {"request_count": 1, "complete_count": 1,
                                           "session_complete": True,
                                           "captures": [{"capture_id": capture_id,
                                                         "session_id": session_id, "complete": True}],
                                           "capture_sessions": [{"session_id": session_id,
                                                                 "complete": True}],
                                           "orphan_results": [], "rejected_count": 0}))
    with pytest.raises(RuntimeError, match="result or parse evidence missing"):
        probe._parent(output, 1)
