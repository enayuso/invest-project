from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
import io
import json
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch


TOOL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOL_DIR))

import cli  # noqa: E402
from contracts import ContractViolation  # noqa: E402
from isolation import IsolationViolation, OsIsolationAssessment  # noqa: E402


class CliFailureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = Path(tempfile.mkdtemp(prefix="offline-cli-"))
        self.output = self.temp / "run-01"
        self.output.mkdir()
        self.assessment = OsIsolationAssessment("confirmed", "synthetic", "test_only")

    def tearDown(self) -> None:
        shutil.rmtree(self.temp)

    def _run_scan_failure(self, failure: Exception, *, marker_failure: bool = False) -> tuple[int, dict[str, object]]:
        def fail_after_start(loaded, scanner, on_scanner_start=None):
            on_scanner_start()
            raise failure

        marker_effect = OSError("synthetic marker failure") if marker_failure else None
        captured = io.StringIO()
        patches = (
            patch.object(cli, "assess_os_isolation", return_value=self.assessment),
            patch.object(cli, "require_material_scan_isolation", return_value=self.assessment),
            patch.object(
                cli,
                "load_validated_bundle",
                return_value=SimpleNamespace(
                    validation=SimpleNamespace(scanner_source_sha256="a" * 64),
                ),
            ),
            patch.object(cli, "prepare_guard", return_value=SimpleNamespace(code=(lambda: None).__code__)),
            patch.object(cli, "collect_tool_hashes", return_value={"cli.py": "a" * 64}),
            patch.object(cli, "prepare_new_output_directory", return_value=self.output),
            patch.object(cli, "install_runtime_policy", return_value=object()),
            patch.object(cli, "instantiate_guard", return_value=object()),
            patch.object(cli, "scan_selected_cases", side_effect=fail_after_start),
        )
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6], patches[7], patches[8]:
            marker_patch = patch.object(cli, "write_incomplete_marker", side_effect=marker_effect) if marker_failure else None
            if marker_patch is None:
                with redirect_stdout(captured):
                    code = cli.main(["scan", "--bundle", "synthetic", "--profile", "synthetic", "--out", str(self.output)])
            else:
                with marker_patch, redirect_stdout(captured):
                    code = cli.main(["scan", "--bundle", "synthetic", "--profile", "synthetic", "--out", str(self.output)])
        return code, json.loads(captured.getvalue())

    def test_runtime_failures_preserve_scanner_state_and_marker(self) -> None:
        cases = (
            (ValueError("synthetic"), 3, "INTERNAL_ERROR"),
            (OSError("synthetic"), 3, "INTERNAL_ERROR"),
            (ContractViolation("SYNTHETIC_CONTRACT", "runtime", "Synthetic contract failure"), 2, "SYNTHETIC_CONTRACT"),
            (IsolationViolation("SYNTHETIC_ISOLATION", "runtime", "Synthetic isolation failure"), 4, "SYNTHETIC_ISOLATION"),
        )
        for failure, expected_exit, expected_error in cases:
            with self.subTest(kind=type(failure).__name__):
                if (self.output / "run-incomplete.json").exists():
                    (self.output / "run-incomplete.json").unlink()
                code, response = self._run_scan_failure(failure)
                self.assertEqual(code, expected_exit)
                self.assertTrue(response["scanner_ran"])
                self.assertEqual(response["errors"][0]["code"], expected_error)
                self.assertEqual(response["incomplete_marker_status"], "written")
                marker = json.loads((self.output / "run-incomplete.json").read_text("utf-8"))
                self.assertTrue(marker["scanner_ran"])
                self.assertEqual(marker["phase"], "scanner_execution")

    def test_marker_write_failure_is_visible_in_stdout(self) -> None:
        code, response = self._run_scan_failure(ValueError("synthetic"), marker_failure=True)
        self.assertEqual(code, 3)
        self.assertTrue(response["scanner_ran"])
        self.assertEqual(response["incomplete_marker_status"], "write_failed")

    def test_usage_error_does_not_create_a_marker(self) -> None:
        captured = io.StringIO()
        with redirect_stdout(captured):
            code = cli.main(["unknown-command"])
        response = json.loads(captured.getvalue())
        self.assertEqual(code, 2)
        self.assertFalse(response["scanner_ran"])
        self.assertEqual(response["incomplete_marker_status"], "not_attempted")


if __name__ == "__main__":
    unittest.main()
