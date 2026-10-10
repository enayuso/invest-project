"""Safe JSON-only reporting for offline evaluation validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import container_contract as cc
from bundle import LoadedBundle
from contracts import LABEL_SOURCE_VERSION, ValidationIssue, ValidationResult, require_safe_absolute_path
from isolation import IsolationStatus
from contracts import sha256_bytes


SCHEMA_VERSION = "offline-eval.validate.v1"
MANIFEST_SCHEMA = "offline-eval.scan-run.v2"
FINDINGS_SCHEMA = "offline-eval.findings.v2"
# design §15 G3: single source is container_contract.TOOL_SOURCE_FILES.
TOOL_SOURCE_FILES = cc.TOOL_SOURCE_FILES


def valid_report(result: ValidationResult) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "command": "validate",
        "validation_status": "valid",
        "profile_sha256": result.profile_sha256,
        "input_file_sha256": result.input_file_sha256,
        "scanner_source_sha256": result.scanner_source_sha256,
        "counts": {
            "expected": result.expected_counts,
            "actual": result.actual_counts,
        },
        "case_statuses": result.case_statuses,
        "errors": [issue.as_dict() for issue in result.errors],
        "engine_generation_ran": False,
        "scanner_ran": False,
    }


def error_report(
    command: str,
    issue: ValidationIssue,
    *,
    status: str = "invalid",
    scanner_ran: bool = False,
    incomplete_marker_status: str = "not_attempted",
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "command": command,
        "validation_status": status,
        "profile_sha256": None,
        "input_file_sha256": {},
        "scanner_source_sha256": None,
        "counts": {"expected": {}, "actual": {}},
        "case_statuses": [],
        "errors": [issue.as_dict()],
        "engine_generation_ran": False,
        "scanner_ran": scanner_ran,
        "incomplete_marker_status": incomplete_marker_status,
    }


def invalid_report(issue: ValidationIssue) -> dict[str, Any]:
    return error_report("validate", issue)


def internal_error_report() -> dict[str, Any]:
    return invalid_report(
        ValidationIssue(
            "INTERNAL_ERROR",
            "validator",
            "Unexpected validator failure; no input content is included",
        )
    )


def render_json(report: dict[str, Any]) -> str:
    return json.dumps(
        report,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _pretty_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _exclusive_write(path: Path, raw: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(raw)


def collect_tool_hashes(tool_directory: Path) -> dict[str, str]:
    directory = require_safe_absolute_path(tool_directory, "tool_directory", kind="directory")
    hashes: dict[str, str] = {}
    for filename in TOOL_SOURCE_FILES:
        path = require_safe_absolute_path(directory / filename, f"tool:{filename}", kind="file")
        hashes[filename] = sha256_bytes(path.read_bytes())
    return hashes


def build_findings_document(
    loaded: LoadedBundle,
    cases: list[dict[str, Any]],
    tool_file_sha256: dict[str, str],
) -> dict[str, Any]:
    validation = loaded.validation
    return {
        "schema_version": FINDINGS_SCHEMA,
        "profile_sha256": validation.profile_sha256,
        "scanner_source_sha256": validation.scanner_source_sha256,
        "label_source_sha256": validation.input_file_sha256["claim-annotations.json"],
        "label_source_version": LABEL_SOURCE_VERSION,
        "tool_file_sha256": dict(sorted(tool_file_sha256.items())),
        "candidate_case_ids": list(loaded.profile.get("candidate_case_ids", [])),
        "scope_counts": {
            "expected": validation.expected_counts,
            "actual": validation.actual_counts,
        },
        "scope_case_statuses": validation.case_statuses,
        "context_origin": loaded.profile.get("context_origin"),
        "replay_mode": loaded.profile.get("replay_mode"),
        "cases": cases,
        "engine_generation_ran": False,
        "historical_replay_certified": False,
        "operational_accuracy_estimate": None,
    }


def _scan_counts(cases: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "applicable_candidates": len(cases),
        "scanned": sum(case.get("execution_status") == "scanned" for case in cases),
        "n_a": sum(case.get("execution_status") != "scanned" for case in cases),
        "raw_findings": sum(len(case.get("raw_findings", [])) for case in cases),
        "unique_mappings": sum(
            candidate.get("mapping_status") == "unique"
            for case in cases
            for candidate in case.get("mapping_candidates", [])
        ),
        "ambiguous_mappings": sum(
            candidate.get("mapping_status") != "unique"
            for case in cases
            for candidate in case.get("mapping_candidates", [])
        ),
    }


def _summary_markdown(cases: list[dict[str, Any]]) -> str:
    counts = _scan_counts(cases)
    lines = [
        "# Offline session-close scan summary",
        "",
        "Controlled-context rescore only. This is not a certified historical replay or an operational accuracy estimate.",
        "",
        f"- candidates: {counts['applicable_candidates']}",
        f"- scanned: {counts['scanned']}",
        f"- N/A: {counts['n_a']}",
        f"- raw findings: {counts['raw_findings']}",
        f"- unique mapping candidates: {counts['unique_mappings']}",
        f"- ambiguous mapping candidates: {counts['ambiguous_mappings']}",
        "- real-claim precision/recall: N/A until hash-bound human review",
        "",
        "| case_id | decision_id | draft label | execution | findings | mapping |",
        "|---|---:|---|---|---:|---|",
    ]
    for case in sorted(cases, key=lambda item: item["case_id"]):
        mappings = case.get("mapping_candidates", [])
        mapping = ",".join(candidate["mapping_status"] for candidate in mappings) or "none"
        lines.append(
            f"| {case['case_id']} | {case['decision_id']} | {case['label_draft']} | "
            f"{case['execution_status']} | {len(case.get('raw_findings', []))} | {mapping} |"
        )
    return "\n".join(lines) + "\n"


def write_scan_artifacts(
    output_directory: Path,
    loaded: LoadedBundle,
    cases: list[dict[str, Any]],
    isolation: IsolationStatus,
    tool_file_sha256: dict[str, str],
    *,
    launch_nonce: str | None,
    python_version: str,
) -> dict[str, Any]:
    findings = build_findings_document(loaded, cases, tool_file_sha256)
    findings_raw = _pretty_bytes(findings)
    summary_raw = _summary_markdown(cases).encode("utf-8")
    _exclusive_write(output_directory / "findings.json", findings_raw)
    _exclusive_write(output_directory / "summary.md", summary_raw)
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "status": "complete",
        "command": "scan",
        "profile_sha256": loaded.validation.profile_sha256,
        "scanner_source_sha256": loaded.validation.scanner_source_sha256,
        "input_file_sha256": loaded.validation.input_file_sha256,
        "label_source_sha256": loaded.validation.input_file_sha256["claim-annotations.json"],
        "label_source_version": LABEL_SOURCE_VERSION,
        "tool_file_sha256": dict(sorted(tool_file_sha256.items())),
        "candidate_case_ids": list(loaded.profile.get("candidate_case_ids", [])),
        "scope_counts": {
            "expected": loaded.validation.expected_counts,
            "actual": loaded.validation.actual_counts,
        },
        "scope_case_statuses": loaded.validation.case_statuses,
        "findings_file_sha256": sha256_bytes(findings_raw),
        "summary_file_sha256": sha256_bytes(summary_raw),
        "counts": _scan_counts(cases),
        "isolation": isolation.as_dict(),
        "launch_nonce": launch_nonce,
        "python_version": python_version,
        "container_contract_sha256": tool_file_sha256.get("container_contract.py"),
        "context_origin": loaded.profile.get("context_origin"),
        "replay_mode": loaded.profile.get("replay_mode"),
        "engine_generation_ran": False,
        "scanner_ran": any(case.get("execution_status") in {"scanned", "execution_error"} for case in cases),
        "historical_replay_certified": False,
        "operational_accuracy_estimate": None,
        "real_claim_metrics_before_review": None,
        "limitations": list(cc.LIMITATIONS),
    }
    manifest_raw = _pretty_bytes(manifest)
    _exclusive_write(output_directory / "run-manifest.json", manifest_raw)
    return manifest


def write_incomplete_marker(
    output_directory: Path,
    issue: ValidationIssue,
    *,
    phase: str,
    scanner_ran: bool,
    tool_file_sha256: dict[str, str] | None,
) -> dict[str, Any]:
    marker = {
        "schema_version": MANIFEST_SCHEMA,
        "status": "incomplete",
        "command": "scan",
        "phase": phase,
        "errors": [issue.as_dict()],
        "tool_file_sha256": dict(sorted((tool_file_sha256 or {}).items())),
        "engine_generation_ran": False,
        "scanner_ran": scanner_ran,
        "historical_replay_certified": False,
        "operational_accuracy_estimate": None,
        "limitations": list(cc.LIMITATIONS),
    }
    _exclusive_write(output_directory / "run-incomplete.json", _pretty_bytes(marker))
    return marker


def scan_stdout_report(run_name: str, manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "offline-eval.scan.v1",
        "command": "scan",
        "status": "complete",
        "run_name": run_name,
        "counts": manifest["counts"],
        "profile_sha256": manifest["profile_sha256"],
        "scanner_source_sha256": manifest["scanner_source_sha256"],
        "engine_generation_ran": False,
        "scanner_ran": manifest["scanner_ran"],
        "historical_replay_certified": False,
        "operational_accuracy_estimate": None,
        "errors": [],
    }


def isolation_error_report(
    issue: ValidationIssue,
    *,
    scanner_ran: bool = False,
    isolation_assessment: dict[str, Any] | None = None,
    incomplete_marker_status: str = "not_attempted",
) -> dict[str, Any]:
    return {
        "schema_version": "offline-eval.scan.v1",
        "command": "scan",
        "status": "withheld",
        "counts": None,
        "engine_generation_ran": False,
        "scanner_ran": scanner_ran,
        "isolation_assessment": isolation_assessment,
        "incomplete_marker_status": incomplete_marker_status,
        "historical_replay_certified": False,
        "operational_accuracy_estimate": None,
        "errors": [issue.as_dict()],
    }
