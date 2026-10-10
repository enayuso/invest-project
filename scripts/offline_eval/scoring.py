"""Review-bound scoring; scanner output alone never becomes ground truth."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from contracts import (
    ContractViolation,
    LABEL_SOURCE_VERSION,
    canonical_json_sha256,
    read_limited_file,
    require_bool,
    require_dict,
    require_identifier,
    require_int,
    require_list,
    require_safe_absolute_path,
    require_sha256,
    require_string,
    require_unique_identifiers,
    sha256_bytes,
    strict_json_loads,
)
import container_contract as cc
import report
from session_adapter import parse_finding_contract


REVIEW_SCHEMA = "offline-eval.hit-review.v2"
FINDINGS_SCHEMA = "offline-eval.findings.v2"
MANIFEST_SCHEMA = "offline-eval.scan-run.v2"
CLEAR_LABELS = {"confirmed_error", "confirmed_valid"}
EXECUTION_STATUSES = {"scanned", "not_applicable", "execution_error"}
MAPPING_STATUSES = {"unique", "not_found", "ambiguous_multiple_locations"}
FAMILIES = {"closed_assertion", "completed_candle"}
# design §15 G3: single source is container_contract.TOOL_SOURCE_FILES.
TOOL_SOURCE_FILES = set(cc.TOOL_SOURCE_FILES)
# G3: this scorer's own directory -- the running scoring.py's own tool folder,
# never a caller-supplied path.
TOOL_DIR = Path(__file__).resolve().parent
SCOPE_COUNT_KEYS = {
    "unique_evidence", "source_claims", "candidates",
    "candidate_error_drafts", "candidate_valid_drafts",
    "out_of_scope_clear", "ambiguous", "unassessable",
}
SPLIT_COUNT_KEYS = {
    "split:development_diagnostic",
    "split:development_representative_unopened",
    "split:quarantine_prior_exposure",
    "split:validation_candidate_prior_exposure_unknown",
}


def _current_tool_hashes() -> dict[str, str]:
    """G3: the scorer's own, currently-running tool source hashes. A module-level
    function (not inlined) so tests can patch it without touching real files on
    disk. No product bypass flag exists for this check."""

    return report.collect_tool_hashes(TOOL_DIR)


def _load_json(path: Path, reference: str) -> tuple[bytes, dict[str, Any]]:
    safe = require_safe_absolute_path(path, reference, kind="file")
    raw = read_limited_file(safe, reference)
    return raw, require_dict(strict_json_loads(raw, reference), reference)


def _percent(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else round(numerator / denominator, 6)


def _require_enum(value: Any, allowed: set[str], reference: str) -> str:
    normalized = require_string(value, reference)
    if normalized not in allowed:
        raise ContractViolation("INVALID_ENUM", reference, "Value is outside the allowed set")
    return normalized


def _validate_hash_map(value: Any, reference: str) -> dict[str, str]:
    raw = require_dict(value, reference)
    if set(raw) != TOOL_SOURCE_FILES:
        raise ContractViolation("REFERENCE_MISMATCH", reference, "Tool source hash set is incomplete")
    return {name: require_sha256(raw[name], f"{reference}.{name}") for name in sorted(raw)}


def _validate_scope(findings: dict[str, Any]) -> tuple[list[str], list[dict[str, Any]]]:
    candidate_ids = require_unique_identifiers(findings.get("candidate_case_ids"), "findings.candidate_case_ids")
    scope_counts = require_dict(findings.get("scope_counts"), "findings.scope_counts")
    expected = require_dict(scope_counts.get("expected"), "findings.scope_counts.expected")
    actual = require_dict(scope_counts.get("actual"), "findings.scope_counts.actual")
    if set(expected) != SCOPE_COUNT_KEYS:
        raise ContractViolation("REFERENCE_MISMATCH", "findings.scope_counts.expected", "Scope count key set is incomplete")
    if not SCOPE_COUNT_KEYS.issubset(actual) or not set(actual).issubset(SCOPE_COUNT_KEYS | SPLIT_COUNT_KEYS):
        raise ContractViolation("REFERENCE_MISMATCH", "findings.scope_counts.actual", "Scope count key set is not allowed")
    for key, expected_value in expected.items():
        count = require_int(expected_value, f"findings.scope_counts.expected.{key}", minimum=0)
        if require_int(actual.get(key), f"findings.scope_counts.actual.{key}", minimum=0) != count:
            raise ContractViolation("COUNT_MISMATCH", f"scope_counts.{key}", "Validated scope counts disagree")

    statuses: list[dict[str, Any]] = []
    seen: set[str] = set()
    selected: set[str] = set()
    reason_counts = {
        "clear_label_outside_session_close_v1": 0,
        "ambiguous_label": 0,
        "unassessable_label": 0,
    }
    allowed_reasons = {"phase3a_session_close_v1", *reason_counts}
    for index, raw_status in enumerate(require_list(findings.get("scope_case_statuses"), "findings.scope_case_statuses")):
        status = require_dict(raw_status, f"findings.scope_case_statuses[{index}]")
        case_id = require_identifier(status.get("case_id"), f"findings.scope_case_statuses[{index}].case_id")
        if case_id in seen:
            raise ContractViolation("DUPLICATE_ID", "findings.scope_case_statuses", "Scope contains duplicate case ids")
        seen.add(case_id)
        require_int(status.get("decision_id"), f"{case_id}.decision_id", minimum=1)
        state = _require_enum(status.get("status"), {"selected", "excluded"}, f"{case_id}.status")
        reason = _require_enum(status.get("reason"), allowed_reasons, f"{case_id}.reason")
        if state == "selected":
            if reason != "phase3a_session_close_v1":
                raise ContractViolation("REFERENCE_MISMATCH", case_id, "Selected scope reason is inconsistent")
            selected.add(case_id)
        else:
            if reason == "phase3a_session_close_v1":
                raise ContractViolation("REFERENCE_MISMATCH", case_id, "Excluded scope reason is inconsistent")
            reason_counts[reason] += 1
        statuses.append(status)
    if selected != set(candidate_ids):
        raise ContractViolation("REFERENCE_MISMATCH", "findings.candidate_case_ids", "Candidate set differs from scope statuses")
    derived = {
        "source_claims": len(statuses),
        "candidates": len(selected),
        "out_of_scope_clear": reason_counts["clear_label_outside_session_close_v1"],
        "ambiguous": reason_counts["ambiguous_label"],
        "unassessable": reason_counts["unassessable_label"],
    }
    for key, value in derived.items():
        if require_int(actual.get(key), f"findings.scope_counts.actual.{key}", minimum=0) != value:
            raise ContractViolation("COUNT_MISMATCH", f"scope_counts.{key}", "Scope status count is inconsistent")
    return candidate_ids, statuses


def _validate_mapping(case_id: str, mapping: dict[str, Any], raw_finding: str, expected_index: int) -> str:
    if require_int(mapping.get("finding_index"), f"{case_id}.finding_index", minimum=0) != expected_index:
        raise ContractViolation("REFERENCE_MISMATCH", case_id, "Finding mapping index is inconsistent")
    expected_hash = sha256_bytes(raw_finding.encode("utf-8"))
    if require_sha256(mapping.get("finding_sha256"), f"{case_id}.finding_sha256") != expected_hash:
        raise ContractViolation("FILE_HASH_MISMATCH", case_id, "Finding mapping hash is inconsistent")
    parsed_family, parsed_quote = parse_finding_contract(raw_finding, expected_index)
    family = _require_enum(mapping.get("family"), FAMILIES, f"{case_id}.family")
    if family != parsed_family:
        raise ContractViolation("REFERENCE_MISMATCH", case_id, "Finding family differs from raw scanner output")
    quote_sha256 = require_sha256(mapping.get("quote_sha256"), f"{case_id}.quote_sha256")
    if quote_sha256 != sha256_bytes(parsed_quote.encode("utf-8")):
        raise ContractViolation("FILE_HASH_MISMATCH", case_id, "Finding quote hash differs from raw scanner output")
    quote_codepoints = require_int(mapping.get("quote_codepoints"), f"{case_id}.quote_codepoints", minimum=0)
    if quote_codepoints != len(parsed_quote):
        raise ContractViolation("COUNT_MISMATCH", case_id, "Finding quote length differs from raw scanner output")
    mapping_status = _require_enum(mapping.get("mapping_status"), MAPPING_STATUSES, f"{case_id}.mapping_status")
    ranges = require_list(mapping.get("candidate_ranges"), f"{case_id}.candidate_ranges")
    for raw_range in ranges:
        candidate_range = require_dict(raw_range, f"{case_id}.candidate_range")
        start = require_int(candidate_range.get("start"), f"{case_id}.range.start", minimum=0)
        end = require_int(candidate_range.get("end_exclusive"), f"{case_id}.range.end", minimum=0)
        if end < start:
            raise ContractViolation("INVALID_VALUE", case_id, "Finding mapping range is reversed")
        if end - start != quote_codepoints:
            raise ContractViolation("REFERENCE_MISMATCH", case_id, "Finding mapping range length differs from quote")
    overlaps = mapping.get("candidate_overlaps_claim_span")
    if mapping_status == "unique":
        if len(ranges) != 1:
            raise ContractViolation("COUNT_MISMATCH", case_id, "Unique mapping must contain one range")
        require_bool(overlaps, f"{case_id}.candidate_overlaps_claim_span")
    elif overlaps is not None or (mapping_status == "not_found" and ranges) or (mapping_status == "ambiguous_multiple_locations" and len(ranges) < 2):
        raise ContractViolation("REFERENCE_MISMATCH", case_id, "Finding mapping status and ranges disagree")
    return family


def _validate_findings(manifest: dict[str, Any], findings: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    for key in ("profile_sha256", "scanner_source_sha256", "label_source_sha256"):
        value = require_sha256(findings.get(key), f"findings.{key}")
        if require_sha256(manifest.get(key), f"manifest.{key}") != value:
            raise ContractViolation("REFERENCE_MISMATCH", key, "Manifest and findings provenance disagree")
    input_hashes = require_dict(manifest.get("input_file_sha256"), "manifest.input_file_sha256")
    label_hash = require_sha256(input_hashes.get("claim-annotations.json"), "manifest.input_file_sha256.claim-annotations.json")
    if label_hash != findings["label_source_sha256"]:
        raise ContractViolation("REFERENCE_MISMATCH", "label_source_sha256", "Label source hash is inconsistent")
    if findings.get("label_source_version") != LABEL_SOURCE_VERSION or manifest.get("label_source_version") != LABEL_SOURCE_VERSION:
        raise ContractViolation("REFERENCE_MISMATCH", "label_source_version", "Label source version is inconsistent")
    tool_hashes = _validate_hash_map(findings.get("tool_file_sha256"), "findings.tool_file_sha256")
    if _validate_hash_map(manifest.get("tool_file_sha256"), "manifest.tool_file_sha256") != tool_hashes:
        raise ContractViolation("REFERENCE_MISMATCH", "tool_file_sha256", "Tool source hashes disagree")
    candidate_ids, scope_statuses = _validate_scope(findings)
    scope_by_case = {status["case_id"]: status for status in scope_statuses}
    if require_unique_identifiers(manifest.get("candidate_case_ids"), "manifest.candidate_case_ids") != candidate_ids:
        raise ContractViolation("REFERENCE_MISMATCH", "candidate_case_ids", "Manifest candidate order differs")
    if manifest.get("scope_counts") != findings.get("scope_counts") or manifest.get("scope_case_statuses") != scope_statuses:
        raise ContractViolation("REFERENCE_MISMATCH", "scope", "Manifest and findings scope disagree")
    if manifest.get("context_origin") != findings.get("context_origin") or manifest.get("replay_mode") != findings.get("replay_mode"):
        raise ContractViolation("REFERENCE_MISMATCH", "context", "Manifest and findings context disagree")

    by_case: dict[str, dict[str, Any]] = {}
    family_hits: dict[str, int] = {}
    for index, raw_case in enumerate(require_list(findings.get("cases"), "findings.cases")):
        case = require_dict(raw_case, f"findings.cases[{index}]")
        case_id = require_identifier(case.get("case_id"), f"findings.cases[{index}].case_id")
        if case_id in by_case:
            raise ContractViolation("DUPLICATE_ID", "findings.cases", "Findings contain duplicate case ids")
        require_int(case.get("decision_id"), f"{case_id}.decision_id", minimum=1)
        _require_enum(case.get("label_draft"), CLEAR_LABELS, f"{case_id}.label_draft")
        if case["decision_id"] != scope_by_case.get(case_id, {}).get("decision_id"):
            raise ContractViolation("REFERENCE_MISMATCH", case_id, "Case decision id differs from scope status")
        if case.get("context_origin") != findings.get("context_origin") or case.get("replay_mode") != findings.get("replay_mode"):
            raise ContractViolation("REFERENCE_MISMATCH", case_id, "Case context differs from findings document")
        require_sha256(case.get("input_node_sha256"), f"{case_id}.input_node_sha256")
        execution = _require_enum(case.get("execution_status"), EXECUTION_STATUSES, f"{case_id}.execution_status")
        raw_findings = [require_string(item, f"{case_id}.raw_findings") for item in require_list(case.get("raw_findings"), f"{case_id}.raw_findings")]
        case_hash = require_sha256(case.get("case_findings_sha256"), f"{case_id}.case_findings_sha256")
        if case_hash != canonical_json_sha256(raw_findings):
            raise ContractViolation("FILE_HASH_MISMATCH", case_id, "Case finding hash is inconsistent")
        mappings = [require_dict(item, f"{case_id}.mapping_candidates") for item in require_list(case.get("mapping_candidates"), f"{case_id}.mapping_candidates")]
        if len(mappings) != len(raw_findings):
            raise ContractViolation("COUNT_MISMATCH", case_id, "Finding and mapping counts differ")
        for finding_index, (mapping, raw_finding) in enumerate(zip(mappings, raw_findings)):
            family = _validate_mapping(case_id, mapping, raw_finding, finding_index)
            family_hits[family] = family_hits.get(family, 0) + 1
        note = case.get("session_note")
        note_hash = case.get("session_note_sha256")
        if note is None:
            if note_hash is not None:
                raise ContractViolation("REFERENCE_MISMATCH", case_id, "Session note hash exists without a note")
        else:
            note_text = require_string(note, f"{case_id}.session_note")
            if require_sha256(note_hash, f"{case_id}.session_note_sha256") != sha256_bytes(note_text.encode("utf-8")):
                raise ContractViolation("FILE_HASH_MISMATCH", case_id, "Session note hash is inconsistent")
        reason = case.get("n_a_reason")
        if execution == "scanned":
            if reason is not None or note is None:
                raise ContractViolation("REFERENCE_MISMATCH", case_id, "Scanned case state is inconsistent")
        else:
            allowed_reason = {"session_in_progress_not_exact_true", "session_metadata_missing_or_invalid"} if execution == "not_applicable" else {"scanner_execution_error"}
            _require_enum(reason, allowed_reason, f"{case_id}.n_a_reason")
            if raw_findings or mappings:
                raise ContractViolation("REFERENCE_MISMATCH", case_id, "Unscanned case cannot contain findings")
        by_case[case_id] = case
    if set(by_case) != set(candidate_ids):
        raise ContractViolation("REFERENCE_MISMATCH", "findings.cases", "Findings cases differ from candidate set")
    actual_scope_counts = require_dict(findings["scope_counts"].get("actual"), "findings.scope_counts.actual")
    label_counts = {
        "candidate_error_drafts": sum(case["label_draft"] == "confirmed_error" for case in by_case.values()),
        "candidate_valid_drafts": sum(case["label_draft"] == "confirmed_valid" for case in by_case.values()),
    }
    for key, value in label_counts.items():
        if require_int(actual_scope_counts.get(key), f"findings.scope_counts.actual.{key}", minimum=0) != value:
            raise ContractViolation("COUNT_MISMATCH", f"scope_counts.{key}", "Candidate label count is inconsistent")

    counts = {
        "applicable_candidates": len(by_case),
        "scanned": sum(case["execution_status"] == "scanned" for case in by_case.values()),
        "n_a": sum(case["execution_status"] != "scanned" for case in by_case.values()),
        "raw_findings": sum(len(case["raw_findings"]) for case in by_case.values()),
        "unique_mappings": sum(mapping["mapping_status"] == "unique" for case in by_case.values() for mapping in case["mapping_candidates"]),
        "ambiguous_mappings": sum(mapping["mapping_status"] != "unique" for case in by_case.values() for mapping in case["mapping_candidates"]),
    }
    manifest_counts = require_dict(manifest.get("counts"), "manifest.counts")
    for key, value in counts.items():
        if require_int(manifest_counts.get(key), f"manifest.counts.{key}", minimum=0) != value:
            raise ContractViolation("COUNT_MISMATCH", f"manifest.counts.{key}", "Manifest scan count is inconsistent")
    return by_case, family_hits


def _require_confirmed_isolation(manifest: dict[str, Any]) -> None:
    # S1 (F8): exact P1-P11 probe-id set, each individually passed, plus the
    # confirmed status/security-boundary strings -- single source in
    # container_contract so this can never drift from the launcher's own L8 check.
    if cc.validate_isolation_block(manifest.get("isolation")):
        raise ContractViolation(
            "ISOLATION_NOT_CONFIRMED", "run-manifest.json.isolation", "OS isolation was not confirmed for this run"
        )


def _require_valid_attestation(
    *, manifest: dict[str, Any], manifest_raw: bytes, tool_hashes: dict[str, str], launch_root: Path, run_name: str
) -> bytes:
    launch_root_safe = require_safe_absolute_path(launch_root, "launch_root", kind="directory")
    attestation_dir = launch_root_safe / run_name
    try:
        attestation_dir = require_safe_absolute_path(attestation_dir, "attestation_dir", kind="directory")
        attestation_raw, attestation = _load_json(
            attestation_dir / cc.ATTESTATION_FILENAME, cc.ATTESTATION_FILENAME
        )
    except ContractViolation as exc:
        raise ContractViolation(
            "ATTESTATION_MISSING", "attestation", "Launch attestation is missing or unreadable"
        ) from exc

    if attestation.get("schema_version") != cc.ATTESTATION_SCHEMA:
        raise ContractViolation("ATTESTATION_MISSING", "attestation.schema_version", "Attestation schema is unsupported")
    if attestation.get("run_name") != run_name:
        raise ContractViolation(
            "ATTESTATION_CONTRACT_MISMATCH", "attestation.run_name", "Attestation run name differs from the scored run"
        )
    if require_sha256(attestation.get("run_manifest_sha256"), "attestation.run_manifest_sha256") != sha256_bytes(manifest_raw):
        raise ContractViolation(
            "ATTESTATION_HASH_MISMATCH", "attestation.run_manifest_sha256", "Attestation manifest hash is stale"
        )
    attestation_nonce = require_string(attestation.get("launch_nonce"), "attestation.launch_nonce")
    if cc.NONCE_RE.fullmatch(attestation_nonce) is None or attestation_nonce != manifest.get("launch_nonce"):
        raise ContractViolation(
            "ATTESTATION_NONCE_MISMATCH", "attestation.launch_nonce", "Attestation nonce does not match the run manifest"
        )
    attestation_tool_hashes = _validate_hash_map(attestation.get("tool_file_sha256"), "attestation.tool_file_sha256")
    if attestation_tool_hashes != tool_hashes:
        raise ContractViolation(
            "ATTESTATION_TOOL_HASH_MISMATCH", "attestation.tool_file_sha256", "Attestation tool hashes differ from the run"
        )
    # G1: the launcher (host side) independently hashes the guard source file
    # before it is mounted; this must match what the container itself hashed
    # after reading the (possibly re-mounted) file, binding the two views of
    # the same file together.
    if require_sha256(attestation.get("guard_source_sha256"), "attestation.guard_source_sha256") != require_sha256(
        manifest.get("scanner_source_sha256"), "manifest.scanner_source_sha256"
    ):
        raise ContractViolation(
            "ATTESTATION_GUARD_HASH_MISMATCH",
            "attestation.guard_source_sha256",
            "Attestation guard source hash differs from the run manifest",
        )
    # S6 (F7): compare against the pinned contract constants, not just against
    # each other -- two stale-but-matching values must not pass.
    if attestation.get("image_ref") != cc.IMAGE_REF:
        raise ContractViolation(
            "ATTESTATION_CONTRACT_MISMATCH", "attestation.image_ref", "Attestation image reference is not the pinned contract"
        )
    if attestation.get("image_id") != cc.IMAGE_ID:
        raise ContractViolation(
            "ATTESTATION_CONTRACT_MISMATCH", "attestation.image_id", "Attestation image id is not the pinned contract"
        )
    if attestation.get("python_version") != cc.PYTHON_VERSION_STRING or manifest.get("python_version") != cc.PYTHON_VERSION_STRING:
        raise ContractViolation(
            "ATTESTATION_CONTRACT_MISMATCH", "attestation.python_version", "Python version is not the pinned contract value"
        )
    if attestation.get("container_contract_sha256") != tool_hashes.get("container_contract.py"):
        raise ContractViolation(
            "ATTESTATION_CONTRACT_MISMATCH",
            "attestation.container_contract_sha256",
            "Attestation contract hash differs from the tool hash set",
        )
    inspect_subset = require_dict(attestation.get("inspect_subset"), "attestation.inspect_subset")
    # S7: replay the same validator the launcher used, against the stored
    # subset. No Source in the subset (F6), so expected_sources is not checked here.
    if cc.validate_inspect(inspect_subset, nonce_hex=attestation_nonce, run_name=run_name, expected_sources=None):
        raise ContractViolation(
            "ATTESTATION_INSPECT_INVALID",
            "attestation.inspect_subset",
            "Attestation inspect data does not satisfy the container contract",
        )
    if require_int(attestation.get("container_exit_code"), "attestation.container_exit_code") != 0:
        raise ContractViolation(
            "CONTAINER_EXIT_NONZERO", "attestation.container_exit_code", "Container exited with a non-zero status"
        )
    return attestation_raw


def score_review(run_directory: Path, review_path: Path, output_root: Path, launch_root: Path) -> dict[str, Any]:
    run_dir = require_safe_absolute_path(run_directory, "run", kind="directory")
    root = require_safe_absolute_path(output_root, "output_root", kind="directory")
    if run_dir.parent != root:
        raise ContractViolation("PATH_REJECTED", "run", "Run must be a direct child of the fixed output root")
    review_file = require_safe_absolute_path(review_path, "review", kind="file")
    if review_file.parent != run_dir or review_file.name != "hit-review-v1.json":
        raise ContractViolation("PATH_REJECTED", "review", "Review file must be hit-review-v1.json inside the run folder")

    manifest_raw, manifest = _load_json(run_dir / "run-manifest.json", "run-manifest.json")
    findings_raw, findings = _load_json(run_dir / "findings.json", "findings.json")
    review_raw, review = _load_json(review_file, "hit-review-v1.json")
    if manifest.get("schema_version") != MANIFEST_SCHEMA or manifest.get("status") != "complete":
        raise ContractViolation("INVALID_ENUM", "run-manifest.json", "Run manifest is not complete")
    if findings.get("schema_version") != FINDINGS_SCHEMA:
        raise ContractViolation("INVALID_ENUM", "findings.json", "Findings schema is unsupported")
    if review.get("schema_version") != REVIEW_SCHEMA:
        raise ContractViolation("INVALID_ENUM", "hit-review-v1.json", "Review schema is unsupported")
    if sha256_bytes(findings_raw) != manifest.get("findings_file_sha256"):
        raise ContractViolation("FILE_HASH_MISMATCH", "findings.json", "Findings file hash differs from manifest")

    _require_confirmed_isolation(manifest)
    manifest_tool_hashes = _validate_hash_map(manifest.get("tool_file_sha256"), "manifest.tool_file_sha256")
    # G3 (N3): the run must have been produced by exactly the tool source this
    # scorer is itself running right now -- not merely by "some 9 files whose
    # hashes are internally consistent". A weakened-then-reverted tool would
    # otherwise still pass S5's cross-document check.
    if manifest_tool_hashes != _current_tool_hashes():
        raise ContractViolation(
            "TOOL_VERSION_MISMATCH",
            "manifest.tool_file_sha256",
            "Tool source does not match the running scorer",
        )
    attestation_raw = _require_valid_attestation(
        manifest=manifest,
        manifest_raw=manifest_raw,
        tool_hashes=manifest_tool_hashes,
        launch_root=launch_root,
        run_name=run_dir.name,
    )

    by_case, family_hits = _validate_findings(manifest, findings)

    source = require_dict(review.get("source"), "review.source")
    source_checks = {
        "run_manifest_sha256": sha256_bytes(manifest_raw),
        "findings_file_sha256": sha256_bytes(findings_raw),
        "profile_sha256": findings.get("profile_sha256"),
        "scanner_source_sha256": findings.get("scanner_source_sha256"),
        "label_source_sha256": findings.get("label_source_sha256"),
        "launch_attestation_sha256": sha256_bytes(attestation_raw),
    }
    for key, expected in source_checks.items():
        if require_sha256(source.get(key), f"review.source.{key}") != expected:
            raise ContractViolation("REVIEW_HASH_MISMATCH", f"review.source.{key}", "Review source hash is stale")
    if source.get("label_source_version") != LABEL_SOURCE_VERSION:
        raise ContractViolation("REVIEW_HASH_MISMATCH", "review.source.label_source_version", "Review label version is stale")

    reviews: dict[str, dict[str, Any]] = {}
    for index, raw_item in enumerate(require_list(review.get("items"), "review.items")):
        item = require_dict(raw_item, f"review.items[{index}]")
        case_id = require_identifier(item.get("case_id"), f"review.items[{index}].case_id")
        if case_id in reviews:
            raise ContractViolation("DUPLICATE_ID", "review.items", "Review contains duplicate case ids")
        if case_id not in by_case:
            raise ContractViolation("REFERENCE_MISSING", "review.items", "Review case is absent from findings")
        case = by_case[case_id]
        for key in ("input_node_sha256", "case_findings_sha256"):
            if require_sha256(item.get(key), f"{case_id}.{key}") != case.get(key):
                raise ContractViolation("REVIEW_HASH_MISMATCH", case_id, "Review case hash is stale")
        for key in ("profile_sha256", "scanner_source_sha256", "label_source_sha256"):
            if require_sha256(item.get(key), f"{case_id}.{key}") != findings.get(key):
                raise ContractViolation("REVIEW_HASH_MISMATCH", case_id, "Review provenance hash is stale")
        verdict = _require_enum(item.get("verdict"), {"hit", "no_hit", "ambiguous"}, f"{case_id}.verdict")
        raw_label_status = item.get("label_review_status")
        label_status = "missing" if raw_label_status is None else _require_enum(
            raw_label_status,
            {"approved", "pending", "ambiguous"},
            f"{case_id}.label_review_status",
        )
        require_string(item.get("rationale"), f"{case_id}.rationale")
        require_identifier(item.get("reviewer"), f"{case_id}.reviewer")
        raw_label_version = item.get("label_version")
        label_version = None if raw_label_version is None else require_identifier(
            raw_label_version,
            f"{case_id}.label_version",
        )
        final_label = item.get("final_label")
        if label_status == "approved":
            if label_version is None:
                raise ContractViolation("INVALID_TYPE", f"{case_id}.label_version", "Approved label requires a version")
            final_label = _require_enum(final_label, CLEAR_LABELS, f"{case_id}.final_label")
            if final_label != case["label_draft"]:
                require_string(item.get("label_change_rationale"), f"{case_id}.label_change_rationale")
                if label_version == LABEL_SOURCE_VERSION:
                    raise ContractViolation("REFERENCE_MISMATCH", case_id, "Changed label requires a new label version")
        elif final_label is not None:
            raise ContractViolation("REFERENCE_MISMATCH", case_id, "Unapproved label cannot contain a final label")
        indices: list[int] = []
        seen_indices: set[int] = set()
        for raw_index in require_list(item.get("finding_indices"), f"{case_id}.finding_indices"):
            finding_index = require_int(raw_index, f"{case_id}.finding_indices", minimum=0)
            if finding_index in seen_indices or finding_index >= len(case["raw_findings"]):
                raise ContractViolation("REFERENCE_MISSING", case_id, "Review finding index is invalid")
            seen_indices.add(finding_index)
            indices.append(finding_index)
        if verdict == "hit" and not indices:
            raise ContractViolation("REFERENCE_MISMATCH", case_id, "Hit verdict requires a finding index")
        if verdict == "no_hit" and indices:
            raise ContractViolation("REFERENCE_MISMATCH", case_id, "No-hit verdict cannot cite a finding")
        item["label_review_status"] = label_status
        item["final_label"] = final_label
        reviews[case_id] = item

    confusion = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    statuses: list[dict[str, Any]] = []
    for case_id in sorted(by_case):
        case = by_case[case_id]
        if case["execution_status"] != "scanned":
            statuses.append({"case_id": case_id, "status": "n_a", "reason": case["n_a_reason"]})
            continue
        item = reviews.get(case_id)
        if item is None:
            statuses.append({"case_id": case_id, "status": "n_a", "reason": "review_missing"})
            continue
        if item["label_review_status"] != "approved":
            statuses.append({"case_id": case_id, "status": "n_a", "reason": f"label_review_{item['label_review_status']}"})
            continue
        if item["verdict"] == "ambiguous":
            statuses.append({"case_id": case_id, "status": "n_a", "reason": "review_ambiguous"})
            continue
        detected = item["verdict"] == "hit"
        label = item["final_label"]
        bucket = ("tp" if detected else "fn") if label == "confirmed_error" else ("fp" if detected else "tn")
        confusion[bucket] += 1
        statuses.append({"case_id": case_id, "status": "scored", "bucket": bucket})

    scored = sum(confusion.values())
    return {
        "schema_version": "offline-eval.score.v1",
        "command": "score",
        "status": "complete",
        "run_manifest_sha256": sha256_bytes(manifest_raw),
        "findings_file_sha256": sha256_bytes(findings_raw),
        "review_file_sha256": sha256_bytes(review_raw),
        "counts": {
            "applicable_candidates": len(by_case), "scored": scored,
            "n_a": len(by_case) - scored, **confusion,
            "raw_family_hits": dict(sorted(family_hits.items())),
        },
        "metrics": {
            "precision": _percent(confusion["tp"], confusion["tp"] + confusion["fp"]),
            "recall": _percent(confusion["tp"], confusion["tp"] + confusion["fn"]),
        },
        "case_statuses": statuses,
        "historical_replay_certified": False,
        "operational_accuracy_estimate": None,
        "engine_generation_ran": False,
    }
