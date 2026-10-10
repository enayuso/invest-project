"""Summarize capture session status files without reading request content."""

from __future__ import annotations

import argparse
import json
import itertools
import re
from datetime import datetime, timezone
from pathlib import Path

try:
    from .validate import DROP_REASONS, EVENT_KINDS, VERSION, _load_event
except ImportError:  # Direct execution as a paired standalone artifact.
    from validate import DROP_REASONS, EVENT_KINDS, VERSION, _load_event

SESSION_NAME = re.compile(r"\.capture-session-([0-9a-f]{32})\.json\Z")
MAX_SESSIONS = 1024


def summarize(directory: Path) -> dict:
    """Return only whitelisted counts and states from bounded status files."""
    # Never import the live application or echo arbitrary file content.
    # 운영 앱을 import하거나 임의의 파일 내용을 반사하지 않는다.
    if not directory.is_dir() or directory.is_symlink():
        return {"observer_status": "directory_unavailable", "sessions": [],
                "invalid_session_files": 0}
    paths = sorted(itertools.islice(directory.glob(".capture-session-*.json"), MAX_SESSIONS + 1))
    if len(paths) > MAX_SESSIONS:
        return {"observer_status": "session_limit", "sessions": [],
                "invalid_session_files": None}
    rows: list[dict] = []
    invalid = 0
    for path in paths:
        match = SESSION_NAME.fullmatch(path.name)
        if match is None:
            invalid += 1
            continue
        if path.is_symlink():
            invalid += 1
            continue
        try:
            status = _load_event(path)
        except (OSError, ValueError, UnicodeError, RecursionError):
            invalid += 1
            continue
        if (not isinstance(status, dict) or status.get("schema_version") != VERSION
                or status.get("kind") != "capture_session" or status.get("session_id") != match.group(1)
                or status.get("state") not in ("open", "closed", "incomplete")
                or (status.get("stopped_reason") is not None
                    and (not isinstance(status.get("stopped_reason"), str)
                         or status["stopped_reason"] not in DROP_REASONS))
                or type(status.get("eligible_attempts")) is not int or status["eligible_attempts"] < 0
                or type(status.get("recorded_events")) is not int or status["recorded_events"] < 0
                or not isinstance(status.get("recorded_by_kind"), dict)
                or set(status["recorded_by_kind"]) != set(EVENT_KINDS)
                or any(type(v) is not int or v < 0 for v in status["recorded_by_kind"].values())
                or sum(status["recorded_by_kind"].values()) != status["recorded_events"]
                or not isinstance(status.get("drops"), dict)
                or any(k not in DROP_REASONS or type(v) is not int or v < 0
                       for k, v in status["drops"].items())
                or (status["state"] == "closed" and
                    (status["drops"] or status.get("stopped_reason") is not None))):
            invalid += 1
            continue
        rows.append({"session_id": match.group(1), "state": status["state"],
                     "eligible_attempts": status["eligible_attempts"],
                     "recorded_events": status["recorded_events"],
                     "recorded_by_kind": {k: status["recorded_by_kind"][k] for k in EVENT_KINDS},
                     "drops": {k: status["drops"][k] for k in sorted(status["drops"])},
                     "stopped_reason": status.get("stopped_reason")})
    state = "invalid" if invalid else "available" if rows else "not_initialized_or_unavailable"
    return {"observer_status": state, "sessions": rows, "invalid_session_files": invalid}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture_directory", type=Path)
    args = parser.parse_args(argv)
    result = {"observed_at_utc": datetime.now(timezone.utc).isoformat(),
              "container_status": "not_observed", **summarize(args.capture_directory)}
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result["observer_status"] == "available" else 2


if __name__ == "__main__":
    raise SystemExit(main())
