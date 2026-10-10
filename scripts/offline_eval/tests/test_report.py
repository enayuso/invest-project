from pathlib import Path
import json
import shutil
import sys
import tempfile
import unittest


TOOL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOL_DIR))

from bundle import LoadedBundle  # noqa: E402
from contracts import LABEL_SOURCE_VERSION, ValidationIssue, ValidationResult  # noqa: E402
from isolation import IsolationStatus  # noqa: E402
from report import invalid_report, render_json, valid_report, write_incomplete_marker, write_scan_artifacts  # noqa: E402


class ReportTests(unittest.TestCase):
    def test_valid_report_is_deterministic_and_marks_no_execution(self) -> None:
        result = ValidationResult(
            profile_sha256="a" * 64,
            input_file_sha256={"inventory.json": "b" * 64},
            expected_counts={"candidates": 1},
            actual_counts={"candidates": 1},
            case_statuses=[{"case_id": "C1", "decision_id": 1, "status": "selected", "reason": "phase3a_session_close_v1"}],
            scanner_source_sha256="c" * 64,
        )
        first = render_json(valid_report(result))
        second = render_json(valid_report(result))
        self.assertEqual(first, second)
        self.assertIn('"engine_generation_ran":false', first)
        self.assertIn('"scanner_ran":false', first)

    def test_invalid_report_does_not_contain_raw_text(self) -> None:
        marker = "SECRET_RAW_NEEDLE"
        report = invalid_report(ValidationIssue("INVALID_JSON", "evidence/1.json", "File is invalid"))
        rendered = render_json(report)
        self.assertNotIn(marker, rendered)
        self.assertEqual(report["case_statuses"], [])

    def test_report_contains_only_safe_case_fields(self) -> None:
        result = ValidationResult(
            profile_sha256="a" * 64,
            input_file_sha256={},
            expected_counts={},
            actual_counts={},
            case_statuses=[{"case_id": "C1", "decision_id": 1, "status": "selected", "reason": "phase3a_session_close_v1"}],
            scanner_source_sha256="b" * 64,
        )
        case = valid_report(result)["case_statuses"][0]
        self.assertEqual(set(case), {"case_id", "decision_id", "status", "reason"})

    def test_scan_artifacts_are_created_once_with_manifest_last(self) -> None:
        temp = Path(tempfile.mkdtemp(prefix="offline-report-"))
        try:
            validation = ValidationResult(
                profile_sha256="a" * 64,
                input_file_sha256={"claim-annotations.json": "b" * 64},
                expected_counts={},
                actual_counts={},
                case_statuses=[],
                scanner_source_sha256="c" * 64,
            )
            loaded = LoadedBundle(
                validation=validation,
                profile={"context_origin": "constructed", "replay_mode": "controlled_context_rescore"},
                inventory_by_id={},
                evidence_by_id={},
                claims_by_case={},
            )
            cases = [{
                "case_id": "C1",
                "decision_id": 1,
                "label_draft": "confirmed_error",
                "execution_status": "scanned",
                "n_a_reason": None,
                "raw_findings": ["raw finding retained locally"],
                "mapping_candidates": [],
                "input_node_sha256": "d" * 64,
                "case_findings_sha256": "e" * 64,
            }]
            isolation = IsolationStatus(True, True, True, "test_boundary")
            tool_hashes = {name: "f" * 64 for name in (
                "bundle.py", "cli.py", "container_contract.py", "contracts.py", "isolation.py",
                "launcher.py", "report.py", "scoring.py", "session_adapter.py",
            )}
            manifest = write_scan_artifacts(
                temp, loaded, cases, isolation, tool_hashes, launch_nonce="a" * 64, python_version="3.12.13"
            )
            self.assertEqual(
                {path.name for path in temp.iterdir()},
                {"findings.json", "summary.md", "run-manifest.json"},
            )
            stored = json.loads((temp / "run-manifest.json").read_text("utf-8"))
            findings = json.loads((temp / "findings.json").read_text("utf-8"))
            self.assertEqual(stored["findings_file_sha256"], manifest["findings_file_sha256"])
            self.assertEqual(stored["label_source_version"], LABEL_SOURCE_VERSION)
            self.assertEqual(findings["label_source_version"], LABEL_SOURCE_VERSION)
            with self.assertRaises(FileExistsError):
                write_scan_artifacts(
                    temp, loaded, cases, isolation, tool_hashes, launch_nonce="a" * 64, python_version="3.12.13"
                )
        finally:
            shutil.rmtree(temp)

    def test_incomplete_marker_records_phase_without_raw_input(self) -> None:
        temp = Path(tempfile.mkdtemp(prefix="offline-incomplete-"))
        try:
            marker = "../SECRET_RAW_NEEDLE"
            issue = ValidationIssue("PATH_REJECTED", marker, "Rejected input path")
            result = write_incomplete_marker(
                temp, issue, phase="artifact_write", scanner_ran=True,
                tool_file_sha256={"report.py": "a" * 64},
            )
            self.assertEqual(result["status"], "incomplete")
            self.assertTrue(result["scanner_ran"])
            self.assertNotIn(marker, (temp / "run-incomplete.json").read_text("utf-8"))
        finally:
            shutil.rmtree(temp)


if __name__ == "__main__":
    unittest.main()
