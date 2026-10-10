"""Load and validate a Phase 2 evidence bundle without executing project code."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from contracts import (
    ALLOWED_LABELS,
    ALLOWED_REVIEW_STATUSES,
    ALLOWED_SPLITS,
    MAX_REFERENCES,
    MAX_TOTAL_BYTES,
    ContractViolation,
    ValidationResult,
    canonical_json_sha256,
    read_limited_file,
    require_bool,
    require_dict,
    require_int,
    require_identifier,
    require_list,
    require_safe_absolute_path,
    require_sha256,
    require_string,
    require_unique_identifiers,
    resolve_json_path,
    sha256_bytes,
    strict_json_loads,
    validate_evidence_reference,
)


PROFILE_SCHEMA = "phase3a.scope.v1"
BUNDLE_FILES = (
    "inventory.json",
    "claim-annotations.json",
    "split-manifest.json",
    "collection-summary.json",
)
FIXED_SCANNER_REFERENCE = "app/agents/session_close_guard.py"


@dataclass
class LoadedBundle:
    """Validated data retained in memory for the separately approved scanner."""

    validation: ValidationResult
    profile: dict[str, Any]
    inventory_by_id: dict[int, dict[str, Any]]
    evidence_by_id: dict[int, dict[str, Any]]
    claims_by_case: dict[str, dict[str, Any]]


def _read_json_file(path: Path, reference: str) -> tuple[bytes, Any]:
    raw = read_limited_file(path, reference)
    return raw, strict_json_loads(raw, reference)


def _require_count_map(value: Any, reference: str) -> dict[str, int]:
    raw = require_dict(value, reference)
    result: dict[str, int] = {}
    for index, (key, item) in enumerate(raw.items()):
        name = require_string(key, f"{reference}[{index}].key")
        result[name] = require_int(item, f"{reference}[{index}].value", minimum=0)
    return result


def _validate_profile(profile: dict[str, Any], bundle_name: str) -> tuple[list[str], list[str], dict[str, int]]:
    if profile.get("schema_version") != PROFILE_SCHEMA:
        raise ContractViolation("INVALID_ENUM", "profile", "Unsupported profile schema")
    if require_string(profile.get("source_bundle"), "profile.source_bundle") != bundle_name:
        raise ContractViolation("REFERENCE_MISMATCH", "profile.source_bundle", "Bundle name does not match profile")
    if profile.get("replay_mode") != "controlled_context_rescore":
        raise ContractViolation("INVALID_ENUM", "profile.replay_mode", "Unsupported replay mode")
    if profile.get("engine_generation_allowed") is not False:
        raise ContractViolation("INVALID_VALUE", "profile.engine_generation_allowed", "Engine generation must be disabled")
    if profile.get("holdout_scoring_allowed") is not False:
        raise ContractViolation("INVALID_VALUE", "profile.holdout_scoring_allowed", "Holdout scoring must be disabled")

    candidate_ids = require_unique_identifiers(profile.get("candidate_case_ids"), "profile.candidate_case_ids")
    out_of_scope_ids = require_unique_identifiers(
        profile.get("out_of_scope_clear_case_ids"), "profile.out_of_scope_clear_case_ids"
    )
    overlap = set(candidate_ids) & set(out_of_scope_ids)
    if overlap:
        raise ContractViolation("DUPLICATE_ID", "profile", "Scope case sets overlap")
    expected = _require_count_map(profile.get("expected_counts"), "profile.expected_counts")
    required_counts = {
        "unique_evidence",
        "source_claims",
        "candidates",
        "candidate_error_drafts",
        "candidate_valid_drafts",
        "out_of_scope_clear",
        "ambiguous",
        "unassessable",
    }
    if set(expected) != required_counts:
        raise ContractViolation("REFERENCE_MISMATCH", "profile.expected_counts", "Expected-count keys are incomplete")
    return candidate_ids, out_of_scope_ids, expected


def _validate_scanner(profile: dict[str, Any], repo_root: Path) -> str:
    scanner = require_dict(profile.get("scanner"), "profile.scanner")
    if scanner.get("name") != "session_close":
        raise ContractViolation("INVALID_ENUM", "profile.scanner.name", "Unsupported scanner")
    if scanner.get("source_reference") != FIXED_SCANNER_REFERENCE:
        raise ContractViolation("PATH_REJECTED", "profile.scanner.source_reference", "Scanner path is not fixed")
    expected_hash = require_sha256(scanner.get("source_bytes_sha256"), "profile.scanner.source_bytes_sha256")
    scanner_path = repo_root / Path(*FIXED_SCANNER_REFERENCE.split("/"))
    scanner_path = require_safe_absolute_path(scanner_path, FIXED_SCANNER_REFERENCE, kind="file")
    actual_hash = sha256_bytes(read_limited_file(scanner_path, FIXED_SCANNER_REFERENCE))
    if actual_hash != expected_hash:
        raise ContractViolation("FILE_HASH_MISMATCH", FIXED_SCANNER_REFERENCE, "Scanner source hash differs from profile")
    return actual_hash


def _load_inventory(value: Any) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
    rows = require_list(value, "inventory.json")
    by_id: dict[int, dict[str, Any]] = {}
    normalized: list[dict[str, Any]] = []
    for index, raw in enumerate(rows):
        row = require_dict(raw, f"inventory[{index}]")
        decision_id = require_int(row.get("id"), f"inventory[{index}].id", minimum=1)
        if decision_id in by_id:
            raise ContractViolation("DUPLICATE_ID", "inventory.json", "Inventory contains a duplicate id")
        validate_evidence_reference(row.get("evidence_path"), decision_id)
        require_sha256(row.get("evidence_sha256"), f"inventory[{index}].evidence_sha256")
        require_sha256(row.get("transcript_sha256"), f"inventory[{index}].transcript_sha256")
        require_string(row.get("ticker"), f"inventory[{index}].ticker")
        require_string(row.get("trade_date"), f"inventory[{index}].trade_date")
        require_bool(row.get("representative"), f"inventory[{index}].representative")
        require_bool(row.get("diagnostic"), f"inventory[{index}].diagnostic")
        by_id[decision_id] = row
        normalized.append(row)
    if len(rows) > MAX_REFERENCES:
        raise ContractViolation("LIMIT_EXCEEDED", "inventory.json", "Too many evidence references")
    return normalized, by_id


def _load_evidence(
    bundle_root: Path,
    inventory: list[dict[str, Any]],
    input_hashes: dict[str, str],
    total_bytes: int,
) -> tuple[dict[int, dict[str, Any]], int]:
    evidence: dict[int, dict[str, Any]] = {}
    root_resolved = bundle_root.resolve(strict=True)
    for row in inventory:
        decision_id = row["id"]
        relative = validate_evidence_reference(row["evidence_path"], decision_id)
        candidate = bundle_root / Path(*relative.split("/"))
        resolved = require_safe_absolute_path(candidate, relative, kind="file")
        if root_resolved not in resolved.parents:
            raise ContractViolation("PATH_REJECTED", relative, "Evidence file escapes bundle root")
        raw = read_limited_file(resolved, relative)
        total_bytes += len(raw)
        if total_bytes > MAX_TOTAL_BYTES:
            raise ContractViolation("LIMIT_EXCEEDED", "bundle", "Bundle exceeds total size limit")
        payload = require_dict(strict_json_loads(raw, relative), relative)
        if canonical_json_sha256(payload) != row["evidence_sha256"]:
            raise ContractViolation("PAYLOAD_HASH_MISMATCH", relative, "Evidence payload hash differs from inventory")
        transcript = require_dict(payload.get("full_transcript"), f"{relative}:full_transcript")
        if canonical_json_sha256(transcript) != row["transcript_sha256"]:
            raise ContractViolation("PAYLOAD_HASH_MISMATCH", relative, "Transcript hash differs from inventory")
        input_hashes[relative] = sha256_bytes(raw)
        evidence[decision_id] = payload
    return evidence, total_bytes


def _load_claims(
    value: Any,
    inventory_by_id: dict[int, dict[str, Any]],
    evidence: dict[int, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    top = require_dict(value, "claim-annotations.json")
    rows = require_list(top.get("items"), "claim-annotations.items")
    by_case: dict[str, dict[str, Any]] = {}
    diagnostic_ids: set[int] = set()
    for index, raw in enumerate(rows):
        row = require_dict(raw, f"claims[{index}]")
        case_id = require_identifier(row.get("case_id"), f"claims[{index}].case_id")
        if case_id in by_case:
            raise ContractViolation("DUPLICATE_ID", "claim-annotations.json", "Claims contain a duplicate case id")
        decision_id = require_int(row.get("decision_id"), f"claims[{index}].decision_id", minimum=1)
        if decision_id not in inventory_by_id:
            raise ContractViolation("REFERENCE_MISSING", case_id, "Claim decision is absent from inventory")
        inventory_row = inventory_by_id[decision_id]
        if validate_evidence_reference(row.get("evidence_path"), decision_id) != inventory_row["evidence_path"]:
            raise ContractViolation("REFERENCE_MISMATCH", case_id, "Claim evidence path differs from inventory")
        if require_sha256(row.get("evidence_sha256"), f"{case_id}.evidence_sha256") != inventory_row["evidence_sha256"]:
            raise ContractViolation("REFERENCE_MISMATCH", case_id, "Claim evidence hash differs from inventory")
        label = require_string(row.get("label"), f"{case_id}.label")
        if label not in ALLOWED_LABELS:
            raise ContractViolation("INVALID_ENUM", case_id, "Claim label is not allowed")
        review_status = require_string(row.get("review_status"), f"{case_id}.review_status")
        if review_status not in ALLOWED_REVIEW_STATUSES:
            raise ContractViolation("INVALID_ENUM", case_id, "Claim review status is not allowed")
        node = resolve_json_path(evidence[decision_id], row.get("path"), f"{case_id}.path")
        if sha256_bytes(node.encode("utf-8")) != require_sha256(row.get("node_sha256_utf8"), f"{case_id}.node_sha256_utf8"):
            raise ContractViolation("PAYLOAD_HASH_MISMATCH", case_id, "Claim node hash differs from evidence")
        start = require_int(row.get("span_start"), f"{case_id}.span_start", minimum=0)
        end = require_int(row.get("span_end_exclusive"), f"{case_id}.span_end_exclusive", minimum=0)
        needle = require_string(row.get("needle"), f"{case_id}.needle", nonempty=False)
        if end < start or end > len(node) or node[start:end] != needle:
            raise ContractViolation("CLAIM_SPAN_MISMATCH", case_id, "Claim span does not match evidence")
        by_case[case_id] = row
        diagnostic_ids.add(decision_id)

    expected_diagnostic = {decision_id for decision_id, row in inventory_by_id.items() if row["diagnostic"]}
    if diagnostic_ids != expected_diagnostic:
        raise ContractViolation("REFERENCE_MISMATCH", "claim-annotations.json", "Claim decisions differ from diagnostic inventory")
    return rows, by_case


def _duplicate_groups(inventory: list[dict[str, Any]], field: str) -> list[dict[str, Any]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for row in inventory:
        if field == "group_ticker_day":
            key = f"{row['ticker']}|{row['trade_date']}"
        else:
            key = row["transcript_sha256"]
        groups[key].append(row["id"])
    return [
        {"key": key, "ids": sorted(ids)}
        for key, ids in sorted(groups.items())
        if len(ids) > 1
    ]


def _load_splits(value: Any, inventory: list[dict[str, Any]], inventory_by_id: dict[int, dict[str, Any]]) -> dict[str, int]:
    top = require_dict(value, "split-manifest.json")
    rows = require_list(top.get("items"), "split-manifest.items")
    by_id: dict[int, dict[str, Any]] = {}
    counts: Counter[str] = Counter()
    for index, raw in enumerate(rows):
        row = require_dict(raw, f"split[{index}]")
        decision_id = require_int(row.get("id"), f"split[{index}].id", minimum=1)
        if decision_id in by_id:
            raise ContractViolation("DUPLICATE_ID", "split-manifest.json", "Split manifest contains a duplicate id")
        if decision_id not in inventory_by_id:
            raise ContractViolation("REFERENCE_MISSING", "split-manifest.json", "Split id is absent from inventory")
        split = require_string(row.get("split"), f"split[{index}].split")
        if split not in ALLOWED_SPLITS:
            raise ContractViolation("INVALID_ENUM", str(decision_id), "Split value is not allowed")
        inv = inventory_by_id[decision_id]
        expected_group = f"{inv['ticker']}|{inv['trade_date']}"
        if row.get("group_original") != str(decision_id) or row.get("group_ticker_day") != expected_group or row.get("group_transcript") != inv["transcript_sha256"]:
            raise ContractViolation("REFERENCE_MISMATCH", str(decision_id), "Split grouping fields differ from inventory")
        if row.get("representative") is not inv["representative"] or row.get("diagnostic") is not inv["diagnostic"]:
            raise ContractViolation("REFERENCE_MISMATCH", str(decision_id), "Split flags differ from inventory")
        by_id[decision_id] = row
        counts[split] += 1
    if set(by_id) != set(inventory_by_id):
        raise ContractViolation("REFERENCE_MISMATCH", "split-manifest.json", "Split ids differ from inventory ids")
    declared_counts = _require_count_map(top.get("counts"), "split-manifest.counts")
    if dict(counts) != declared_counts:
        raise ContractViolation("REFERENCE_MISMATCH", "split-manifest.counts", "Split counts do not match items")
    duplicates = require_dict(top.get("duplicate_groups"), "split-manifest.duplicate_groups")
    for field in ("group_ticker_day", "group_transcript"):
        expected = _duplicate_groups(inventory, field)
        declared = require_list(duplicates.get(field), f"split-manifest.duplicate_groups.{field}")
        if declared != expected:
            raise ContractViolation("REFERENCE_MISMATCH", f"duplicate_groups.{field}", "Duplicate groups differ from inventory")
    return dict(counts)


def _validate_collection_summary(value: Any, inventory: list[dict[str, Any]]) -> None:
    summary = require_dict(value, "collection-summary.json")
    representative_ids = {row["id"] for row in inventory if row["representative"]}
    diagnostic_ids = {row["id"] for row in inventory if row["diagnostic"]}
    checks = {
        "representative_n": len(representative_ids),
        "diagnostic_found_n": len(diagnostic_ids),
        "unique_evidence_n": len(inventory),
        "overlap_n": len(representative_ids & diagnostic_ids),
    }
    for key, expected in checks.items():
        if require_int(summary.get(key), f"collection-summary.{key}", minimum=0) != expected:
            raise ContractViolation("REFERENCE_MISMATCH", f"collection-summary.{key}", "Collection count differs from inventory")
    sampled: set[int] = set()
    for index, day in enumerate(require_list(summary.get("sample_days"), "collection-summary.sample_days")):
        day_obj = require_dict(day, f"sample_days[{index}]")
        ids = require_list(day_obj.get("ids"), f"sample_days[{index}].ids")
        if require_int(day_obj.get("n"), f"sample_days[{index}].n", minimum=0) != len(ids):
            raise ContractViolation("REFERENCE_MISMATCH", f"sample_days[{index}]", "Sample-day count differs from ids")
        for item in ids:
            sampled.add(require_int(item, f"sample_days[{index}].ids", minimum=1))
    if sampled != representative_ids:
        raise ContractViolation("REFERENCE_MISMATCH", "collection-summary.sample_days", "Sample ids differ from representative inventory")
    initial_counts = Counter(require_string(row.get("split_proposal"), "inventory.split_proposal") for row in inventory)
    if _require_count_map(summary.get("split_counts"), "collection-summary.split_counts") != dict(initial_counts):
        raise ContractViolation("REFERENCE_MISMATCH", "collection-summary.split_counts", "Collection snapshot split counts differ from inventory proposals")


def _classify_cases(
    claims: list[dict[str, Any]],
    candidate_ids: list[str],
    out_of_scope_ids: list[str],
    expected: dict[str, int],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    by_case = {row["case_id"]: row for row in claims}
    if not set(candidate_ids).issubset(by_case) or not set(out_of_scope_ids).issubset(by_case):
        raise ContractViolation("REFERENCE_MISSING", "profile", "Profile references an unknown claim case")
    statuses: list[dict[str, Any]] = []
    actual = {
        "source_claims": len(claims),
        "candidates": 0,
        "candidate_error_drafts": 0,
        "candidate_valid_drafts": 0,
        "out_of_scope_clear": 0,
        "ambiguous": 0,
        "unassessable": 0,
    }
    candidate_set = set(candidate_ids)
    out_set = set(out_of_scope_ids)
    for row in claims:
        case_id = row["case_id"]
        label = row["label"]
        if case_id in candidate_set:
            if label not in {"confirmed_error", "confirmed_valid"}:
                raise ContractViolation("INVALID_ENUM", case_id, "Candidate must have a clear draft label")
            status = "selected"
            reason = "phase3a_session_close_v1"
            actual["candidates"] += 1
            actual["candidate_error_drafts" if label == "confirmed_error" else "candidate_valid_drafts"] += 1
        elif case_id in out_set:
            if label not in {"confirmed_error", "confirmed_valid"}:
                raise ContractViolation("INVALID_ENUM", case_id, "Out-of-scope clear case lacks a clear label")
            status = "excluded"
            reason = "clear_label_outside_session_close_v1"
            actual["out_of_scope_clear"] += 1
        elif label == "ambiguous":
            status = "excluded"
            reason = "ambiguous_label"
            actual["ambiguous"] += 1
        elif label == "unassessable":
            status = "excluded"
            reason = "unassessable_label"
            actual["unassessable"] += 1
        else:
            raise ContractViolation("REFERENCE_MISMATCH", case_id, "Clear claim is not accounted for by profile")
        statuses.append(
            {
                "case_id": case_id,
                "decision_id": row["decision_id"],
                "status": status,
                "reason": reason,
            }
        )
    for key in (
        "source_claims",
        "candidates",
        "candidate_error_drafts",
        "candidate_valid_drafts",
        "out_of_scope_clear",
        "ambiguous",
        "unassessable",
    ):
        if actual[key] != expected[key]:
            raise ContractViolation("COUNT_MISMATCH", key, "Actual count differs from profile")
    return sorted(statuses, key=lambda item: item["case_id"]), actual


def load_validated_bundle(bundle_path: Path, profile_path: Path, repo_root: Path) -> LoadedBundle:
    bundle_root = require_safe_absolute_path(bundle_path, "bundle", kind="directory")
    profile_file = require_safe_absolute_path(profile_path, "profile", kind="file")
    repository = require_safe_absolute_path(repo_root, "repository", kind="directory")

    profile_raw, profile_value = _read_json_file(profile_file, "profile")
    profile = require_dict(profile_value, "profile")
    candidate_ids, out_of_scope_ids, expected = _validate_profile(profile, bundle_root.name)
    scanner_hash = _validate_scanner(profile, repository)

    declared_hashes = require_dict(profile.get("source_metadata_file_sha256"), "profile.source_metadata_file_sha256")
    if set(declared_hashes) != set(BUNDLE_FILES):
        raise ContractViolation("REFERENCE_MISMATCH", "profile.source_metadata_file_sha256", "Metadata hash list is incomplete")

    loaded: dict[str, Any] = {}
    input_hashes: dict[str, str] = {}
    total_bytes = 0
    for filename in BUNDLE_FILES:
        path = require_safe_absolute_path(bundle_root / filename, filename, kind="file")
        raw, value = _read_json_file(path, filename)
        total_bytes += len(raw)
        expected_hash = require_sha256(declared_hashes[filename], f"profile.hashes.{filename}")
        actual_hash = sha256_bytes(raw)
        if actual_hash != expected_hash:
            raise ContractViolation("FILE_HASH_MISMATCH", filename, "Metadata file hash differs from profile")
        input_hashes[filename] = actual_hash
        loaded[filename] = value

    inventory, inventory_by_id = _load_inventory(loaded["inventory.json"])
    if len(inventory) != expected["unique_evidence"]:
        raise ContractViolation("COUNT_MISMATCH", "unique_evidence", "Inventory count differs from profile")
    evidence, total_bytes = _load_evidence(bundle_root, inventory, input_hashes, total_bytes)
    if total_bytes > MAX_TOTAL_BYTES:
        raise ContractViolation("LIMIT_EXCEEDED", "bundle", "Bundle exceeds total size limit")
    claims, claims_by_case = _load_claims(loaded["claim-annotations.json"], inventory_by_id, evidence)
    split_counts = _load_splits(loaded["split-manifest.json"], inventory, inventory_by_id)
    _validate_collection_summary(loaded["collection-summary.json"], inventory)
    statuses, actual = _classify_cases(claims, candidate_ids, out_of_scope_ids, expected)
    actual["unique_evidence"] = len(inventory)
    for name, count in sorted(split_counts.items()):
        actual[f"split:{name}"] = count

    validation = ValidationResult(
        profile_sha256=sha256_bytes(profile_raw),
        input_file_sha256=dict(sorted(input_hashes.items())),
        expected_counts=expected,
        actual_counts=actual,
        case_statuses=statuses,
        scanner_source_sha256=scanner_hash,
    )
    return LoadedBundle(
        validation=validation,
        profile=profile,
        inventory_by_id=inventory_by_id,
        evidence_by_id=evidence,
        claims_by_case=claims_by_case,
    )


def validate_bundle(bundle_path: Path, profile_path: Path, repo_root: Path) -> ValidationResult:
    return load_validated_bundle(bundle_path, profile_path, repo_root).validation
