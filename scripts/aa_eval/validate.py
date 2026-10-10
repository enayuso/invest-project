"""Offline integrity and readiness manifest for RM first-native snapshots.

This module is stdlib-only. It never imports the live application, opens a
model endpoint, or runs the trading pipeline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any

VERSION = "rm-first-native-v1"
REQUIRED_KWARGS = {"model", "messages", "temperature", "extra_body", "max_tokens", "response_format"}
EVENT_LIMIT = 2 * 1024 * 1024
HEX32 = re.compile(r"[0-9a-f]{32}\Z")
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
TICKER = re.compile(r"[0-9]{6}\.(?:KS|KQ)\Z")
SOURCE_NAMES = ("recorder", "client_parser", "shared_parser", "schema_validators", "research_manager")
EVENT_KINDS = ("request", "result", "parse", "rejected")
DROP_REASONS = {"sensitive_content", "serialization_failed", "event_limit", "queue_limit",
                "session_limit", "write_failed", "control_queue_limit", "request_limit", "observer_failed",
                "drain_timeout", "control_budget_limit", "capture_stopped", "status_unavailable",
                "unsupported_timeout", "response_limit", "request_fields_invalid"}


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _sha(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def _pairs_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _reject_nonfinite(_: str) -> None:
    raise ValueError("non-finite JSON")


def _finite_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("non-finite JSON")
    return result


def _bounded_int(value: str) -> int:
    if len(value.lstrip("-")) > 18:
        raise ValueError("integer limit")
    return int(value)


def _safe_id(value: Any, pattern: re.Pattern[str]) -> bool:
    return isinstance(value, str) and pattern.fullmatch(value) is not None


def _safe_ref(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()[:16]


def _valid_time(value: Any) -> bool:
    if not isinstance(value, str) or len(value) > 40:
        return False
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return False
    return parsed.tzinfo is not None


def _bounded_tree(value: Any) -> None:
    stack = [(value, 0)]
    nodes = 0
    while stack:
        item, depth = stack.pop()
        nodes += 1
        if depth > 64 or nodes > 20000:
            raise ValueError("JSON structure limit")
        if isinstance(item, dict):
            if nodes + len(stack) + len(item) * 2 > 20000:
                raise ValueError("JSON structure limit")
            stack.extend((part, depth + 1) for pair in item.items() for part in pair)
        elif isinstance(item, list):
            if nodes + len(stack) + len(item) > 20000:
                raise ValueError("JSON structure limit")
            stack.extend((v, depth + 1) for v in item)
        elif isinstance(item, str) and any(0xD800 <= ord(char) <= 0xDFFF for char in item):
            raise ValueError("invalid Unicode scalar")


def _valid_timeout(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    kind = value.get("kind")
    if kind == "none":
        return set(value) == {"kind"}
    if kind == "scalar":
        seconds = value.get("seconds")
        return (set(value) == {"kind", "seconds"} and type(seconds) in (int, float)
                and math.isfinite(seconds) and seconds > 0)
    if kind == "phases":
        if set(value) != {"kind", "connect", "read", "write", "pool"}:
            return False
        return all(item is None or (type(item) in (int, float) and math.isfinite(item) and item > 0)
                   for name, item in value.items() if name != "kind")
    return False


def _load_event(path: Path) -> Any:
    if path.is_symlink() or path.stat().st_size > EVENT_LIMIT:
        raise ValueError("event file invalid")
    with path.open("rb") as stream:
        data = stream.read(EVENT_LIMIT + 1)
    if len(data) > EVENT_LIMIT:
        raise ValueError("event file invalid")
    event = json.loads(data, object_pairs_hook=_pairs_without_duplicates,
                       parse_constant=_reject_nonfinite, parse_float=_finite_float,
                       parse_int=_bounded_int)
    _bounded_tree(event)
    return event


def inspect_capture(request_path: Path) -> dict[str, Any]:
    raw_id = request_path.name.removesuffix(".request.json")
    capture_id = raw_id if _safe_id(raw_id, HEX32) else f"invalid-{_safe_ref(raw_id)}"
    problems: list[str] = []
    if capture_id != raw_id:
        problems.append("capture_id_invalid")
    try:
        event = _load_event(request_path)
    except (OSError, ValueError, UnicodeError, RecursionError):
        return {"capture_id": capture_id, "complete": False, "ready_for_replay": False,
                "problems": problems + ["request_unreadable"]}
    if not isinstance(event, dict):
        return {"capture_id": capture_id, "complete": False, "ready_for_replay": False,
                "problems": problems + ["request_not_object"]}
    request = event.get("request")
    if event.get("schema_version") != VERSION or event.get("kind") != "request":
        problems.append("unknown_schema_version")
    if event.get("capture_id") != raw_id or event.get("logical_call_id") != raw_id:
        problems.append("capture_id_mismatch")
    if not _safe_id(event.get("capture_session_id"), HEX32):
        problems.append("capture_session_missing")
    if event.get("node") != "research_manager" or event.get("application_attempt_kind") != "first_native":
        problems.append("wrong_node_or_attempt")
    if event.get("capture_level") != "client_boundary_request" or event.get("sdk_retry_visibility") != "not_observed":
        problems.append("capture_level_mismatch")
    if (event.get("market") != "domestic" or event.get("trigger") != "auto"
            or event.get("held") is not False or event.get("session_in_progress") is not True
            or event.get("request_complete") is not True):
        problems.append("outside_declared_stratum")
    try:
        valid_date = isinstance(event.get("trade_date"), str) and date.fromisoformat(event["trade_date"]).isoformat() == event["trade_date"]
    except ValueError:
        valid_date = False
    if (not _safe_id(event.get("run_id"), HEX32) or not valid_date
            or not isinstance(event.get("ticker"), str) or not TICKER.fullmatch(event["ticker"])):
        problems.append("linkage_invalid")
    if not _valid_time(event.get("captured_at_utc")):
        problems.append("request_timestamp_invalid")
    if (type(event.get("sdk_max_retries")) is not int or event["sdk_max_retries"] < 0
            or not _valid_timeout(event.get("sdk_timeout"))
            or event.get("latency_definition") != "observer_start_to_sdk_result_including_request_capture"):
        problems.append("sdk_metadata_invalid")
    if not isinstance(request, dict) or set(request) != REQUIRED_KWARGS:
        problems.append("request_fields_invalid")
    else:
        messages = request["messages"]
        if (not isinstance(request["model"], str) or not request["model"]
                or not isinstance(messages, list) or not messages
                or any(not isinstance(m, dict) or not isinstance(m.get("role"), str)
                       or not isinstance(m.get("content"), str) or not m["content"] for m in messages)
                or type(request["temperature"]) not in (int, float)
                or not math.isfinite(request["temperature"])
                or not isinstance(request["extra_body"], dict)
                or type(request["max_tokens"]) is not int or request["max_tokens"] <= 0):
            problems.append("request_values_invalid")
        if _sha(request) != event.get("request_sha256"):
            problems.append("request_hash_mismatch")
        if hashlib.sha256(json.dumps(request, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()).hexdigest() != event.get("request_ordered_sha256"):
            problems.append("request_order_mismatch")
        if _sha(messages) != event.get("messages_sha256"):
            problems.append("messages_hash_mismatch")
        fmt = request["response_format"]
        schema = fmt.get("json_schema") if isinstance(fmt, dict) else None
        if (not isinstance(schema, dict) or fmt.get("type") != "json_schema"
                or schema.get("strict") is not True or not isinstance(schema.get("name"), str)
                or not schema["name"] or not isinstance(schema.get("schema"), dict)
                or event.get("schema_name") != schema.get("name")):
            problems.append("schema_invalid")
        elif _sha(schema) != event.get("schema_sha256"):
            problems.append("schema_hash_mismatch")
    provenance = event.get("provenance")
    source_hashes = provenance.get("source_sha256") if isinstance(provenance, dict) else None
    if (not isinstance(source_hashes, dict) or not provenance.get("openai_sdk_version")
            or not isinstance(provenance.get("openai_sdk_version"), str)
            or not provenance.get("pydantic_version") or not isinstance(provenance.get("pydantic_version"), str)
            or any(not _safe_id(source_hashes.get(name), HEX64) for name in SOURCE_NAMES)):
        problems.append("source_provenance_missing")
    if event.get("provider") not in ("vllm", "gateway") or event.get("endpoint_alias") != f'{event.get("provider")}_primary':
        problems.append("provider_metadata_invalid")
    result_path = request_path.with_name(f"{raw_id}.result.json")
    try:
        result = _load_event(result_path)
    except FileNotFoundError:
        problems.append("response_missing")
        result = None
    except (OSError, ValueError, UnicodeError, RecursionError):
        problems.append("response_unreadable")
        result = None
    if result is None and "response_missing" not in problems and "response_unreadable" not in problems:
        problems.append("response_invalid")
    elif result is not None:
        if (not isinstance(result, dict) or result.get("capture_id") != raw_id
                or result.get("schema_version") != VERSION
                or result.get("capture_session_id") != event.get("capture_session_id")):
            problems.append("response_link_mismatch")
        else:
            if (not _valid_time(result.get("captured_at_utc"))
                    or type(result.get("elapsed_seconds")) not in (int, float)
                    or not math.isfinite(result["elapsed_seconds"]) or result["elapsed_seconds"] < 0
                    or result.get("latency_definition") != "observer_start_to_sdk_result_including_request_capture"):
                problems.append("result_metadata_invalid")
            if result.get("kind") == "response":
                raw = result.get("raw_response")
                choices = raw.get("choices") if isinstance(raw, dict) else None
                if (result.get("response_complete") is not True or result.get("error_class") is not None
                        or not isinstance(choices, list) or not choices
                        or not isinstance(choices[0], dict)
                        or not isinstance(choices[0].get("message"), dict)
                        or "content" not in choices[0]["message"]
                        or not isinstance(choices[0]["message"]["content"], (str, type(None)))
                        or "finish_reason" not in choices[0]
                        or not isinstance(choices[0]["finish_reason"], (str, type(None)))):
                    problems.append("response_incomplete")
                elif _sha(raw) != result.get("raw_response_sha256"):
                    problems.append("response_hash_mismatch")
                parse_path = request_path.with_name(f"{raw_id}.parse.json")
                try:
                    parsed = _load_event(parse_path)
                except FileNotFoundError:
                    problems.append("parse_missing")
                except (OSError, ValueError, UnicodeError, RecursionError):
                    problems.append("parse_unreadable")
                else:
                    if (not isinstance(parsed, dict) or parsed.get("schema_version") != VERSION
                            or parsed.get("kind") != "parse" or parsed.get("capture_id") != raw_id
                            or parsed.get("capture_session_id") != event.get("capture_session_id")
                            or not _valid_time(parsed.get("captured_at_utc"))
                            or type(parsed.get("valid")) is not bool
                            or (parsed["valid"] and parsed.get("error_class") is not None)
                            or (parsed["valid"] and (not isinstance(choices, list) or not choices
                                or not isinstance(choices[0], dict)
                                or not isinstance(choices[0].get("message"), dict)
                                or not isinstance(choices[0]["message"].get("content"), str)
                                or not choices[0]["message"]["content"]))
                            or (not parsed["valid"] and (not isinstance(parsed.get("error_class"), str)
                                                         or not parsed["error_class"]))):
                        problems.append("parse_invalid")
            elif result.get("kind") == "error":
                if (result.get("response_complete") is not False or not isinstance(result.get("error_class"), str)
                        or not result["error_class"] or result.get("raw_response") is not None
                        or result.get("raw_response_sha256") is not None
                        or request_path.with_name(f"{raw_id}.parse.json").exists()):
                    problems.append("error_incomplete")
            else:
                problems.append("result_kind_invalid")
    serving = event.get("serving_provenance")
    if not isinstance(serving, dict) or not all(serving.get(key) for key in (
        "model_revision", "tokenizer_revision", "chat_template_sha256", "server_revision", "generation_defaults_sha256",
    )):
        problems.append("serving_provenance_missing")
    problems.append("replay_runtime_not_implemented")
    complete = not [p for p in problems if p not in ("serving_provenance_missing", "replay_runtime_not_implemented")]
    return {"capture_id": capture_id,
            "session_id": event.get("capture_session_id") if _safe_id(event.get("capture_session_id"), HEX32) else None,
            "run_id": event.get("run_id") if _safe_id(event.get("run_id"), HEX32) else None,
            "request_sha256": event.get("request_sha256") if _safe_id(event.get("request_sha256"), HEX64) else None,
            "complete": complete, "ready_for_replay": False, "problems": problems}


def build_manifest(directory: Path) -> dict[str, Any]:
    captures = [inspect_capture(path) for path in sorted(directory.glob("*.request.json"))]
    requested = {path.name.removesuffix(".request.json") for path in directory.glob("*.request.json")}
    orphans = [name if _safe_id(name, HEX32) else f"invalid-{_safe_ref(name)}"
               for path in sorted(directory.glob("*.result.json"))
               if (name := path.name.removesuffix(".result.json")) not in requested]
    rejected = [f"rejected-{_safe_ref(path.name)}" for path in sorted(directory.glob("*.rejected.json"))]
    actual: dict[str, dict[str, int]] = {}
    inventory_valid = True
    for suffix, kind in ((".request.json", "request"), (".result.json", "result"),
                         (".parse.json", "parse"), (".rejected.json", "rejected")):
        for path in directory.glob(f"*{suffix}"):
            filename_id = path.name.removesuffix(suffix)
            if kind == "rejected":
                parts = filename_id.split(".")
                valid_name = len(parts) == 2 and _safe_id(parts[0], HEX32) and parts[1] in ("request", "result", "parse")
                capture_id = parts[0] if valid_name else None
            else:
                valid_name = _safe_id(filename_id, HEX32)
                capture_id = filename_id if valid_name else None
            try:
                event = _load_event(path)
            except (OSError, ValueError, UnicodeError, RecursionError):
                event = None
            if (not valid_name or not isinstance(event, dict) or event.get("schema_version") != VERSION
                    or event.get("capture_id") != capture_id
                    or not _safe_id(event.get("capture_session_id"), HEX32)
                    or event.get("kind") not in (("response", "error") if kind == "result" else (kind,))
                    or (kind == "rejected" and event.get("stage") != parts[1])
                    or (kind in ("result", "parse") and capture_id not in requested)
                    or (kind == "rejected" and parts[1] != "request" and capture_id not in requested)):
                inventory_valid = False
                continue
            session_id = event["capture_session_id"]
            counts = actual.setdefault(session_id, dict.fromkeys(EVENT_KINDS, 0))
            counts[kind] += 1
    sessions: list[dict[str, Any]] = []
    for path in sorted(directory.glob(".capture-session-*.json")):
        session_id = path.name.removeprefix(".capture-session-").removesuffix(".json")
        safe_session = session_id if _safe_id(session_id, HEX32) else f"invalid-{_safe_ref(session_id)}"
        try:
            status = _load_event(path)
        except (OSError, ValueError, UnicodeError, RecursionError):
            status = None
        valid_structure = (isinstance(status, dict) and status.get("schema_version") == VERSION
                 and status.get("kind") == "capture_session" and status.get("session_id") == session_id
                 and _safe_id(session_id, HEX32) and status.get("state") in ("open", "closed", "incomplete")
                 and (status.get("stopped_reason") is None or (isinstance(status.get("stopped_reason"), str)
                      and status["stopped_reason"] in DROP_REASONS))
                 and (status.get("stopped_reason") is None or status.get("state") == "incomplete")
                 and type(status.get("eligible_attempts")) is int and status["eligible_attempts"] >= 0
                 and type(status.get("recorded_events")) is int and status["recorded_events"] >= 0
                 and isinstance(status.get("recorded_by_kind"), dict)
                 and set(status["recorded_by_kind"]) == set(EVENT_KINDS)
                 and all(type(value) is int and value >= 0 for value in status["recorded_by_kind"].values())
                 and sum(status["recorded_by_kind"].values()) == status["recorded_events"]
                 and isinstance(status.get("drops"), dict)
                 and all(key in DROP_REASONS and type(value) is int and value >= 0
                         for key, value in status.get("drops", {}).items()))
        request_rows = sum(row.get("session_id") == session_id for row in captures)
        valid = (inventory_valid and valid_structure and status["state"] == "closed" and status["drops"] == {}
                 and status["eligible_attempts"] == request_rows
                 and status["recorded_by_kind"] == actual.get(session_id, dict.fromkeys(EVENT_KINDS, 0))
                 and status["recorded_by_kind"]["rejected"] == 0
                 and status["recorded_by_kind"]["request"] == request_rows
                 and all(row["complete"] for row in captures if row.get("session_id") == session_id))
        sessions.append({"session_id": safe_session, "complete": bool(valid),
                         "eligible_attempts": status["eligible_attempts"] if valid_structure else None,
                         "recorded_events": status["recorded_events"] if valid_structure else None,
                         "recorded_by_kind": status["recorded_by_kind"] if valid_structure else None,
                         "drops": status["drops"] if valid_structure else None})
    known_sessions = {row["session_id"] for row in sessions}
    session_complete = (bool(sessions) and inventory_valid and all(row["complete"] for row in sessions)
                        and set(actual) <= known_sessions
                        and all(row.get("session_id") in known_sessions for row in captures))
    return {
        "manifest_version": VERSION,
        "capture_level": "client_boundary_request",
        "captures": captures,
        "orphan_results": orphans,
        "rejected_events": rejected,
        "rejected_count": len(rejected),
        "capture_sessions": sessions,
        "session_complete": session_complete,
        "request_count": len(captures),
        "complete_count": sum(row["complete"] for row in captures),
        "ready_for_replay_count": sum(row["ready_for_replay"] for row in captures),
        "ready_for_replay": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture_directory", type=Path)
    parser.add_argument("--output", required=True, type=Path, help="New manifest JSON file; never overwrites")
    args = parser.parse_args(argv)
    if not args.capture_directory.is_dir() or args.output.exists():
        parser.error("capture directory must exist and output must be a new file")
    manifest = build_manifest(args.capture_directory)
    with args.output.open("xb") as stream:
        stream.write(canonical(manifest) + b"\n")
    print(json.dumps({"request_count": manifest["request_count"],
                      "complete_count": manifest["complete_count"],
                      "ready_for_replay_count": manifest["ready_for_replay_count"]}))
    return 0 if manifest["ready_for_replay"] else 2


if __name__ == "__main__":
    sys.exit(main())
