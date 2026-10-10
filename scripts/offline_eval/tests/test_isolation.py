from pathlib import Path
import shutil
import sys
import tempfile
import unittest


TOOL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOL_DIR))

from contracts import ContractViolation  # noqa: E402
from isolation import (  # noqa: E402
    IsolationViolation,
    OsIsolationAssessment,
    RuntimePolicy,
    prepare_new_output_directory,
    require_material_scan_isolation,
    _raise_for_unconfirmed_assessment,
)


class IsolationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = Path(tempfile.mkdtemp(prefix="offline-isolation-"))

    def tearDown(self) -> None:
        shutil.rmtree(self.temp)

    def test_policy_blocks_network_and_non_output_files(self) -> None:
        output = self.temp / "output"
        output.mkdir()
        policy = RuntimePolicy(scanner_filename="fixed-guard.py", output_directory=output)
        with self.assertRaises(IsolationViolation):
            policy.handle_event("socket.connect", (None, None))
        with self.assertRaises(IsolationViolation):
            policy.handle_event("open", (str(self.temp / "outside.json"), "x", 0))
        policy.handle_event("open", (str(output / "result.json"), "x", 0))

    def test_only_new_direct_child_output_is_allowed(self) -> None:
        root = self.temp / "runs"
        output = prepare_new_output_directory(root / "run-01", root)
        self.assertTrue(output.is_dir())
        with self.assertRaises(ContractViolation):
            prepare_new_output_directory(root / "run-01", root)
        with self.assertRaises(ContractViolation):
            prepare_new_output_directory(self.temp / "elsewhere" / "run-02", root)

    def test_material_scan_is_withheld_without_os_attestation(self) -> None:
        with self.assertRaises(IsolationViolation) as raised:
            require_material_scan_isolation()
        self.assertEqual(raised.exception.issue.code, "OS_SANDBOX_VERIFICATION_UNIMPLEMENTED")

    def test_failed_verification_has_a_distinct_failure_code(self) -> None:
        assessment = OsIsolationAssessment("verification_failed", "synthetic", "sentinel_failed")
        with self.assertRaises(IsolationViolation) as raised:
            _raise_for_unconfirmed_assessment(assessment)
        self.assertEqual(raised.exception.issue.code, "OS_SANDBOX_VERIFICATION_FAILED")


if __name__ == "__main__":
    unittest.main()
