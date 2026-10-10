"""Separate-process smoke test because Python audit hooks cannot be removed."""

from pathlib import Path
import json
import sys


TEST_DIR = Path(__file__).resolve().parent
TOOL_DIR = TEST_DIR.parent
REPO_ROOT = TOOL_DIR.parents[1]
sys.path.insert(0, str(TOOL_DIR))

from bundle import FIXED_SCANNER_REFERENCE, LoadedBundle  # noqa: E402
from contracts import ValidationResult, sha256_bytes  # noqa: E402
from isolation import IsolationViolation, OsIsolationAssessment, RuntimePolicy, install_runtime_policy  # noqa: E402
from session_adapter import SESSION_NOTE_TEMPLATE, instantiate_guard, prepare_guard, scan_selected_cases  # noqa: E402


source = REPO_ROOT / Path(*FIXED_SCANNER_REFERENCE.split("/"))
prepared = prepare_guard(REPO_ROOT, sha256_bytes(source.read_bytes()))
policy = RuntimePolicy(scanner_filename=prepared.code.co_filename, output_directory=None)
assessment = OsIsolationAssessment("verification_unimplemented", None, "synthetic_test_only")
status = install_runtime_policy(policy, os_isolation=assessment)
scanner = instantiate_guard(prepared)
node = "The stock closed at 100 KRW today."
loaded = LoadedBundle(
    validation=ValidationResult("a" * 64, {"claim-annotations.json": "b" * 64}, {}, {}, [], prepared.source_sha256),
    profile={
        "candidate_case_ids": ["S01"],
        "session_note_template": SESSION_NOTE_TEMPLATE,
        "context_origin": "synthetic",
        "replay_mode": "controlled_context_rescore",
    },
    inventory_by_id={1: {"ticker": "SYNTHETIC", "trade_date": "2026-09-18", "session_in_progress": True}},
    evidence_by_id={1: {"decision_text": node}},
    claims_by_case={"S01": {
        "case_id": "S01", "decision_id": 1, "label": "confirmed_error",
        "path": ["decision_text"], "span_start": 10, "span_end_exclusive": 23,
    }},
)
cases = scan_selected_cases(loaded, scanner)
blocked = False
try:
    sys.audit("socket.connect", None, None)
except IsolationViolation:
    blocked = True
if len(cases) != 1 or len(cases[0]["raw_findings"]) != 1 or not blocked:
    raise SystemExit(1)
print(json.dumps({
    "audit_hook_installed": status.audit_hook_installed,
    "sentinel_blocked": status.sentinel_blocked,
    "network_event_blocked": blocked,
    "synthetic_findings": len(cases[0]["raw_findings"]),
    "os_level_sandbox_confirmed": status.os_level_sandbox_confirmed,
}, sort_keys=True))
