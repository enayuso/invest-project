"""Read-only session summaries must never include captured request content."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.aa_eval.status import summarize


def _write_status(folder: Path, **changes):
    session_id = "a" * 32
    body = {"schema_version": "rm-first-native-v1", "kind": "capture_session",
            "session_id": session_id, "state": "closed", "stopped_reason": None,
            "eligible_attempts": 1, "recorded_events": 3,
            "recorded_by_kind": {"request": 1, "result": 1, "parse": 1, "rejected": 0},
            "drops": {}, "synthetic_raw_secret": "SYNTHETIC_MUST_NOT_APPEAR"}
    body.update(changes)
    path = folder / f".capture-session-{session_id}.json"
    path.write_text(json.dumps(body), encoding="utf-8")
    return path


def test_status_summary_whitelists_fields_and_does_not_read_request(tmp_path):
    _write_status(tmp_path)
    (tmp_path / f"{'b' * 32}.request.json").write_text("SYNTHETIC_MUST_NOT_APPEAR", encoding="utf-8")
    result = summarize(tmp_path)
    assert result["observer_status"] == "available"
    assert result["sessions"][0]["eligible_attempts"] == 1
    assert "SYNTHETIC_MUST_NOT_APPEAR" not in json.dumps(result)


def test_status_summary_treats_missing_and_malformed_as_unknown(tmp_path):
    assert summarize(tmp_path)["observer_status"] == "not_initialized_or_unavailable"
    path = _write_status(tmp_path, eligible_attempts="not a count")
    result = summarize(tmp_path)
    assert result["observer_status"] == "invalid"
    assert result["sessions"] == []
    assert result["invalid_session_files"] == 1
    path.write_bytes(b"x" * (2 * 1024 * 1024 + 1))
    assert summarize(tmp_path)["observer_status"] == "invalid"


def test_status_summary_rejects_false_closed_session(tmp_path):
    _write_status(tmp_path, drops={"write_failed": 1})
    assert summarize(tmp_path)["observer_status"] == "invalid"


def test_status_summary_rejects_symlink(tmp_path):
    real = tmp_path / "real.json"
    real.write_text("{}", encoding="utf-8")
    link = tmp_path / f".capture-session-{'a' * 32}.json"
    try:
        link.symlink_to(real)
    except (OSError, NotImplementedError):
        return
    assert summarize(tmp_path)["observer_status"] == "invalid"
