"""Strict, side-effect-free contracts for the offline evaluation bundle."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from typing import Any


MAX_FILE_BYTES = 4 * 1024 * 1024
MAX_TOTAL_BYTES = 64 * 1024 * 1024
MAX_REFERENCES = 1024
MAX_TEXT_CODEPOINTS = 1_000_000
LABEL_SOURCE_VERSION = "claim-annotations-v1"

ALLOWED_LABELS = {
    "confirmed_error",
    "confirmed_valid",
    "ambiguous",
    "unassessable",
}
ALLOWED_REVIEW_STATUSES = {"draft_single_reviewer"}
ALLOWED_SPLITS = {
    "development_representative_unopened",
    "quarantine_prior_exposure",
    "validation_candidate_prior_exposure_unknown",
    "development_diagnostic",
}

_EVIDENCE_PATH_RE = re.compile(r"evidence/([1-9][0-9]*)\.json\Z")
_HEX_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_SAFE_IDENTIFIER_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}\Z")
_SAFE_REFERENCE_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/\[\]-]{0,159}\Z")


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    reference: str
    description: str

    def as_dict(self) -> dict[str, str]:
        reference = self.reference
        if _SAFE_REFERENCE_RE.fullmatch(reference) is None:
            reference = "input"
        return {
            "code": self.code,
            "reference": reference,
            "description": self.description,
        }


class ContractViolation(Exception):
    """Expected input-contract failure with a safe, non-sensitive message."""

    def __init__(self, code: str, reference: str, description: str) -> None:
        super().__init__(description)
        self.issue = ValidationIssue(code, reference, description)


@dataclass
class ValidationResult:
    profile_sha256: str
    input_file_sha256: dict[str, str]
    expected_counts: dict[str, int]
    actual_counts: dict[str, int]
    case_statuses: list[dict[str, Any]]
    scanner_source_sha256: str
    errors: list[ValidationIssue] = field(default_factory=list)


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    try:
        rendered = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ContractViolation(
            "INVALID_JSON", "payload", "JSON value cannot be canonicalized"
        ) from exc
    return rendered.encode("utf-8")


def canonical_json_sha256(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def _pairs_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ContractViolation(
                "DUPLICATE_KEY", "json", "JSON object contains a duplicate key"
            )
        result[key] = value
    return result


def _reject_constant(_: str) -> Any:
    raise ContractViolation(
        "INVALID_JSON", "json", "NaN and Infinity are not valid inputs"
    )


def strict_json_loads(raw: bytes, reference: str) -> Any:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ContractViolation(
            "INVALID_JSON", reference, "File is not valid UTF-8 JSON"
        ) from exc
    try:
        return json.loads(
            text,
            object_pairs_hook=_pairs_without_duplicates,
            parse_constant=_reject_constant,
        )
    except ContractViolation as exc:
        raise ContractViolation(
            exc.issue.code, reference, exc.issue.description
        ) from exc
    except (json.JSONDecodeError, RecursionError) as exc:
        raise ContractViolation(
            "INVALID_JSON", reference, "File is not structurally valid JSON"
        ) from exc


def require_dict(value: Any, reference: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ContractViolation("INVALID_TYPE", reference, "Expected a JSON object")
    return value


def require_list(value: Any, reference: str) -> list[Any]:
    if not isinstance(value, list):
        raise ContractViolation("INVALID_TYPE", reference, "Expected a JSON array")
    return value


def require_string(value: Any, reference: str, *, nonempty: bool = True) -> str:
    if not isinstance(value, str) or (nonempty and not value):
        raise ContractViolation("INVALID_TYPE", reference, "Expected a string")
    return value


def require_identifier(value: Any, reference: str) -> str:
    identifier = require_string(value, reference)
    if _SAFE_IDENTIFIER_RE.fullmatch(identifier) is None:
        raise ContractViolation("INVALID_IDENTIFIER", reference, "Identifier format is not allowed")
    return identifier


def require_bool(value: Any, reference: str) -> bool:
    if not isinstance(value, bool):
        raise ContractViolation("INVALID_TYPE", reference, "Expected a boolean")
    return value


def require_int(value: Any, reference: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContractViolation("INVALID_TYPE", reference, "Expected an integer")
    if minimum is not None and value < minimum:
        raise ContractViolation("INVALID_VALUE", reference, "Integer is below minimum")
    return value


def require_sha256(value: Any, reference: str) -> str:
    digest = require_string(value, reference)
    if _HEX_SHA256_RE.fullmatch(digest) is None:
        raise ContractViolation("INVALID_VALUE", reference, "Expected lowercase SHA-256")
    return digest


def require_unique_strings(value: Any, reference: str) -> list[str]:
    values = require_list(value, reference)
    result: list[str] = []
    seen: set[str] = set()
    for index, item in enumerate(values):
        string = require_string(item, f"{reference}[{index}]")
        if string in seen:
            raise ContractViolation(
                "DUPLICATE_ID", reference, "List contains a duplicate identifier"
            )
        seen.add(string)
        result.append(string)
    return result


def require_unique_identifiers(value: Any, reference: str) -> list[str]:
    values = require_list(value, reference)
    result: list[str] = []
    seen: set[str] = set()
    for index, item in enumerate(values):
        identifier = require_identifier(item, f"{reference}[{index}]")
        if identifier in seen:
            raise ContractViolation(
                "DUPLICATE_ID", reference, "List contains a duplicate identifier"
            )
        seen.add(identifier)
        result.append(identifier)
    return result


def validate_evidence_reference(value: Any, expected_id: int) -> str:
    safe_reference = f"evidence:{expected_id}"
    reference = require_string(value, safe_reference)
    if "\\" in reference or ":" in reference:
        raise ContractViolation(
            "PATH_REJECTED", safe_reference, "Evidence path is not a safe relative path"
        )
    pure = PurePosixPath(reference)
    if pure.is_absolute() or ".." in pure.parts or "." in pure.parts:
        raise ContractViolation(
            "PATH_REJECTED", safe_reference, "Evidence path is not a safe relative path"
        )
    match = _EVIDENCE_PATH_RE.fullmatch(reference)
    if match is None or int(match.group(1)) != expected_id:
        raise ContractViolation(
            "PATH_REJECTED", safe_reference, "Evidence path does not match its decision id"
        )
    return reference


def resolve_json_path(root: Any, path: Any, reference: str) -> str:
    parts = require_list(path, reference)
    if not parts:
        raise ContractViolation("INVALID_PATH", reference, "JSON path cannot be empty")
    value = root
    for index, part in enumerate(parts):
        part_ref = f"{reference}[{index}]"
        if isinstance(part, bool):
            raise ContractViolation("INVALID_PATH", part_ref, "Boolean path parts are forbidden")
        if isinstance(part, str):
            if not isinstance(value, dict) or part not in value:
                raise ContractViolation("REFERENCE_MISSING", reference, "JSON path is missing")
            value = value[part]
        elif isinstance(part, int) and part >= 0:
            if not isinstance(value, list) or part >= len(value):
                raise ContractViolation("REFERENCE_MISSING", reference, "JSON path is missing")
            value = value[part]
        else:
            raise ContractViolation(
                "INVALID_PATH", part_ref, "Path parts must be strings or nonnegative integers"
            )
    if not isinstance(value, str):
        raise ContractViolation("INVALID_TYPE", reference, "Referenced node must be text")
    if len(value) > MAX_TEXT_CODEPOINTS:
        raise ContractViolation("LIMIT_EXCEEDED", reference, "Referenced text is too large")
    return value


def path_has_reparse_component(path: Path) -> bool:
    """Return True for symlinks, junctions, or other Windows reparse points."""

    current = Path(path.anchor)
    for part in path.parts[1:]:
        current = current / part
        if not current.exists():
            continue
        if current.is_symlink():
            return True
        is_junction = getattr(current, "is_junction", None)
        if is_junction is not None and is_junction():
            return True
        attributes = getattr(current.lstat(), "st_file_attributes", 0)
        if attributes & 0x400:  # FILE_ATTRIBUTE_REPARSE_POINT
            return True
    return False


def require_safe_absolute_path(path: Path, reference: str, *, kind: str) -> Path:
    if not path.is_absolute():
        raise ContractViolation("PATH_REJECTED", reference, "Path must be absolute")
    if path_has_reparse_component(path):
        raise ContractViolation("PATH_REJECTED", reference, "Reparse paths are forbidden")
    if not path.exists():
        raise ContractViolation("REFERENCE_MISSING", reference, "Path does not exist")
    if kind == "file" and not path.is_file():
        raise ContractViolation("PATH_REJECTED", reference, "Path must identify a file")
    if kind == "directory" and not path.is_dir():
        raise ContractViolation("PATH_REJECTED", reference, "Path must identify a directory")
    return path.resolve(strict=True)


def read_limited_file(path: Path, reference: str) -> bytes:
    size = path.stat().st_size
    if size > MAX_FILE_BYTES:
        raise ContractViolation("LIMIT_EXCEEDED", reference, "File exceeds size limit")
    return path.read_bytes()
