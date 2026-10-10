"""Pinned server settings for the offline replay harness (stdlib only).

Stdlib-only ON PURPOSE: `app.config.Settings` reads env + `.env.local` at
import time, and `LLMClient`/`PriorityLimiter` are built at import time too, so
the pinned values must reach `os.environ` BEFORE anything imports `app.config`
(spec A.2). This module therefore never imports `app.*`; the checks that need
the real `Settings` take it (or its `model_fields`) as an argument.
설정 고정 모듈 — app.config import 전에 os.environ에 값을 넣어야 해서 stdlib만 쓴다.

File format (`replay-pinned-settings-v1`):
    {"schema": ..., "settings": {KEY: {"value", "source", "config_line"}},
     "record_only": {KEY: {...}}, "notes": "..."}
`source` is one of server_env / server_env_inferred / code_default. A
`code_default` entry must equal `Settings.model_fields[name].default`
(otherwise "code default drift"), every entry must equal the imported
`settings` value, and the file must never hold a secret.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SCHEMA = "replay-pinned-settings-v1"
SOURCES = ("server_env", "server_env_inferred", "code_default")

# (kind, value, source, config_line) — spec section E table. The default document
# below is generated from this table, so the repo copy and the on-disk JSON cannot
# drift apart silently (a test compares them).
_ROWS: dict[str, tuple[str, Any, str, int | None]] = {
    "AI_PROFILE": ("str", "production", "server_env", 1674),
    "MAX_DEBATE_ROUNDS": ("int", 2, "server_env", 431),
    "MAX_RISK_ROUNDS": ("int", 2, "server_env", 432),
    "LLM_MODEL": ("str", "nvidia/Qwen3.6-35B-A3B-NVFP4", "server_env", 133),
    "LLM_PROVIDER": ("str", "vllm", "code_default", 148),
    "LLM_MAX_OUTPUT_TOKENS": ("int", 16000, "code_default", 253),
    "DEBATE_ROUND1_MAX_OUTPUT_TOKENS": ("int", 1800, "code_default", 352),
    "LLM_STRIP_REASONING_LEAK": ("bool", True, "code_default", 363),
    "RESEARCH_MANAGER_REASONING": ("bool", True, "code_default", 382),
    "INJECTION_FLAG_REST_GUARD": ("bool", True, "code_default", 396),
    "DEBATE_PARALLEL_FIRST_ROUND": ("bool", False, "code_default", 443),
    "DEBATE_OPENING_TURN_PROMPT": ("bool", True, "code_default", 455),
    "DEBATE_OWN_PREVIOUS_TURN": ("bool", True, "code_default", 467),
    "HYSTERESIS_MIN_CONFIDENCE_TO_FLIP": ("float", 0.6, "code_default", 1379),
    "CONSENSUS_OVERRIDE_MIN_CONFIDENCE": ("float", 0.75, "code_default", 1392),
    "CONSENSUS_OVERRIDE_MIN_CONFIDENCE_BEARISH": ("float", 0.85, "code_default", 1393),
    "UNJUSTIFIED_EXIT_GATE_ENABLED": ("bool", True, "code_default", 1827),
    "RM_REQUEST_CAPTURE_ENABLED": ("bool", False, "code_default", 257),
}
KEYED_KEYS: tuple[str, ...] = tuple(_ROWS)
KINDS: dict[str, str] = {k: v[0] for k, v in _ROWS.items()}

# Recorded, never applied or keyed (spec E): downstream unused or set by the harness.
RECORD_ONLY_ROWS: dict[str, tuple[Any, str, int | None]] = {
    "AUTO_TRADE_MAX_CONCURRENT_ANALYSES": (2, "server_env", 1037),
    "LLM_TIMEOUT_SECONDS": (300.0, "code_default", 240),
    "LLM_MAX_CONCURRENT_REQUESTS": ("harness_k", "harness", 408),
    "LLM_GATEWAY_REASONING_EFFORT": ("low", "code_default", 166),
}

_SECRET_HINT = re.compile(r"URL|KEY|TOKEN|PASSWORD|SECRET|DATABASE_", re.IGNORECASE)

# The harness must never run with request capture on (writes prompt bytes to disk).
MUST_BE_FALSE = ("RM_REQUEST_CAPTURE_ENABLED",)


class PinnedError(Exception):
    """Pinned-settings refusal — the CLI maps this to exit code 2."""


@dataclass(frozen=True)
class Pinned:
    values: dict[str, Any]  # KEY -> coerced value
    sources: dict[str, str]
    config_lines: dict[str, int | None]
    record_only: dict[str, Any]
    sha256: str
    path: str = ""
    notes: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    def keyed_values(self) -> dict[str, Any]:
        """Canonical, sorted values that enter the run's `base_key`."""
        return {k: self.values[k] for k in sorted(self.values)}


def default_document() -> dict[str, Any]:
    """The pinned document for the server as of 2026-09-30 (spec E table)."""
    settings = {
        key: {"value": value, "source": source, "config_line": line}
        for key, (_kind, value, source, line) in _ROWS.items()
    }
    record_only = {
        key: {"value": value, "source": source, "config_line": line}
        for key, (value, source, line) in RECORD_ONLY_ROWS.items()
    }
    return {
        "schema": SCHEMA,
        "notes": (
            "Server settings of 2026-09-30 for the offline replay harness. No secrets. "
            "LLM_MODEL was confirmed by the user as the server env value."
        ),
        "settings": settings,
        "record_only": record_only,
    }


def _coerce(key: str, value: Any) -> Any:
    kind = KINDS[key]
    if kind == "bool":
        if isinstance(value, bool):
            return value
        raise PinnedError(f"{key}: expected JSON boolean, got {value!r}")
    if kind == "int":
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        raise PinnedError(f"{key}: expected integer, got {value!r}")
    if kind == "float":
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        raise PinnedError(f"{key}: expected number, got {value!r}")
    if isinstance(value, str) and value:
        return value
    raise PinnedError(f"{key}: expected non-empty string, got {value!r}")


def parse(document: dict[str, Any], *, sha256: str = "", path: str = "") -> Pinned:
    """Validate a decoded document. Rules (spec E): schema, every required key present
    and non-null, no key outside the allow-list, secret-looking keys refused."""
    if not isinstance(document, dict) or document.get("schema") != SCHEMA:
        raise PinnedError(f"pinned settings: schema must be {SCHEMA!r}")
    section = document.get("settings")
    if not isinstance(section, dict):
        raise PinnedError("pinned settings: missing 'settings' object")
    record_section = document.get("record_only", {})
    if not isinstance(record_section, dict):
        raise PinnedError("pinned settings: 'record_only' must be an object")

    for key in list(section) + list(record_section):
        if _SECRET_HINT.search(key) and key not in KEYED_KEYS and key not in RECORD_ONLY_ROWS:
            raise PinnedError(f"pinned settings: secret-looking key refused: {key}")
    unknown = sorted(k for k in section if k not in KEYED_KEYS)
    if unknown:
        raise PinnedError(f"pinned settings: keys outside the allow-list refused: {unknown}")
    unknown_ro = sorted(k for k in record_section if k not in RECORD_ONLY_ROWS)
    if unknown_ro:
        raise PinnedError(f"pinned settings: record_only keys outside the allow-list refused: {unknown_ro}")
    missing = [k for k in KEYED_KEYS if k not in section]
    if missing:
        raise PinnedError(f"pinned settings: missing required keys: {missing}")

    values: dict[str, Any] = {}
    sources: dict[str, str] = {}
    lines: dict[str, int | None] = {}
    for key in KEYED_KEYS:
        entry = section[key]
        if not isinstance(entry, dict) or entry.get("value") is None:
            raise PinnedError(f"pinned settings: {key} has no value")
        source = entry.get("source")
        if source not in SOURCES:
            raise PinnedError(f"pinned settings: {key} source must be one of {SOURCES}, got {source!r}")
        values[key] = _coerce(key, entry["value"])
        sources[key] = source
        lines[key] = entry.get("config_line")
    for key in MUST_BE_FALSE:
        if values[key] is not False:
            raise PinnedError(f"pinned settings: {key} must be false for the replay harness")
    record_only = {k: (v.get("value") if isinstance(v, dict) else v) for k, v in record_section.items()}
    return Pinned(
        values=values, sources=sources, config_lines=lines, record_only=record_only,
        sha256=sha256, path=path, notes=str(document.get("notes", "")), raw=document,
    )


def load(path: str | os.PathLike[str]) -> Pinned:
    p = Path(path)
    try:
        data = p.read_bytes()
    except OSError as exc:
        raise PinnedError(f"pinned settings: cannot read {p}: {exc}") from exc
    try:
        document = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PinnedError(f"pinned settings: invalid JSON in {p}: {exc}") from exc
    return parse(document, sha256=hashlib.sha256(data).hexdigest(), path=str(p))


def env_string(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def apply_to_environ(pinned: Pinned, k: int, environ: dict[str, str] | None = None) -> dict[str, str]:
    """Put the pinned values (plus the harness concurrency K) into the environment.
    pydantic-settings gives env precedence over `.env.local`, so this wins."""
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise PinnedError(f"K must be a positive integer, got {k!r}")
    target = os.environ if environ is None else environ
    applied: dict[str, str] = {}
    for key in KEYED_KEYS:
        applied[key] = env_string(pinned.values[key])
    applied["LLM_MAX_CONCURRENT_REQUESTS"] = str(k)
    for key, value in applied.items():
        target[key] = value
    return applied


def check_code_defaults(pinned: Pinned, model_fields: Any) -> list[str]:
    """`code_default` entries must equal `Settings.model_fields[name].default`.
    Returns drift messages (empty = OK). `model_fields` is passed in so this module
    stays free of `app.config`."""
    drift: list[str] = []
    for key in KEYED_KEYS:
        if pinned.sources[key] != "code_default":
            continue
        field_info = model_fields.get(key.lower())
        if field_info is None:
            drift.append(f"{key}: no such Settings field")
            continue
        default = _normalize(field_info.default)
        if default != pinned.values[key]:
            drift.append(f"{key}: pinned code_default {pinned.values[key]!r} != Settings default {default!r}")
    return drift


def _normalize(value: Any) -> Any:
    return getattr(value, "value", value)  # Enum -> its value, everything else as is


def verify_against_settings(pinned: Pinned, settings: Any) -> list[str]:
    """Compare each pinned key with the imported `settings` object; mismatch list."""
    problems: list[str] = []
    for key in KEYED_KEYS:
        actual = _normalize(getattr(settings, key.lower(), None))
        expected = pinned.values[key]
        if isinstance(expected, float) and isinstance(actual, (int, float)) and not isinstance(actual, bool):
            actual = float(actual)
        if actual != expected:
            problems.append(f"{key}: pinned {expected!r} != settings {actual!r}")
    return problems
