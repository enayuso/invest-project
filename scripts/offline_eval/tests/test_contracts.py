from pathlib import Path
import sys
import unittest


TOOL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOL_DIR))

from contracts import (  # noqa: E402
    ContractViolation,
    canonical_json_sha256,
    resolve_json_path,
    strict_json_loads,
    validate_evidence_reference,
)
from bundle import _require_count_map  # noqa: E402


class ContractTests(unittest.TestCase):
    def test_canonical_hash_ignores_object_key_order(self) -> None:
        self.assertEqual(canonical_json_sha256({"b": 2, "a": 1}), canonical_json_sha256({"a": 1, "b": 2}))

    def test_duplicate_json_key_is_rejected(self) -> None:
        with self.assertRaises(ContractViolation) as raised:
            strict_json_loads(b'{"a":1,"a":2}', "fixture.json")
        self.assertEqual(raised.exception.issue.code, "DUPLICATE_KEY")

    def test_nan_is_rejected(self) -> None:
        with self.assertRaises(ContractViolation) as raised:
            strict_json_loads(b'{"a":NaN}', "fixture.json")
        self.assertEqual(raised.exception.issue.code, "INVALID_JSON")

    def test_boolean_json_path_part_is_rejected(self) -> None:
        with self.assertRaises(ContractViolation) as raised:
            resolve_json_path({"a": ["text"]}, ["a", True], "case.path")
        self.assertEqual(raised.exception.issue.code, "INVALID_PATH")

    def test_evidence_path_must_match_id(self) -> None:
        with self.assertRaises(ContractViolation) as raised:
            validate_evidence_reference("evidence/2.json", 1)
        self.assertEqual(raised.exception.issue.code, "PATH_REJECTED")

    def test_evidence_path_traversal_is_rejected(self) -> None:
        marker = "../SECRET_RAW_NEEDLE"
        with self.assertRaises(ContractViolation) as raised:
            validate_evidence_reference(marker, 1)
        self.assertEqual(raised.exception.issue.code, "PATH_REJECTED")
        self.assertNotIn(marker, str(raised.exception.issue.as_dict()))

    def test_count_map_error_does_not_echo_external_key(self) -> None:
        marker = "SYNTHETIC_PRIVATE_MARKER"
        with self.assertRaises(ContractViolation) as raised:
            _require_count_map({marker: None}, "profile.expected_counts")
        self.assertNotIn(marker, str(raised.exception.issue.as_dict()))


if __name__ == "__main__":
    unittest.main()
