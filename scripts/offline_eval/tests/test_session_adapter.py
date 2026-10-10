from pathlib import Path
import sys
import unittest


TOOL_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = TOOL_DIR.parents[1]
sys.path.insert(0, str(TOOL_DIR))

from bundle import FIXED_SCANNER_REFERENCE, LoadedBundle  # noqa: E402
from contracts import ContractViolation, ValidationResult, sha256_bytes  # noqa: E402
from session_adapter import (  # noqa: E402
    SESSION_NOTE_TEMPLATE,
    _decode_bounded_repr_string,
    _parse_finding,
    instantiate_guard,
    prepare_guard,
    scan_selected_cases,
)


class SessionAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        source = REPO_ROOT / Path(*FIXED_SCANNER_REFERENCE.split("/"))
        cls.prepared = prepare_guard(REPO_ROOT, sha256_bytes(source.read_bytes()))
        cls.scanner = staticmethod(instantiate_guard(cls.prepared))

    def test_controlled_synthetic_baseline(self) -> None:
        note = SESSION_NOTE_TEMPLATE.format(ticker="SYNTHETIC", trade_date="2026-09-18")
        cases = {
            "S01": ("The stock closed at 100 KRW today.", note, 1),
            "S02": ("The previous close was 100 KRW.", note, 0),
            "S03": ("If it closes below 100 KRW today, we would reassess.", note, 0),
            "S04": (note, note, 0),
            "S05": ("The stock closed at 100 KRW today.", "", 0),
            "S06": ("The stock closed at 100 KRW on September 17.", note, 0),
            "S07": ("The stock closed at 100 KRW on September 18.", note, 1),
            "S08": ("The stock closed at 100 KRW today. A rebound is likely.", note, 0),
            "S09": (
                "Foreign investors were net sellers for five sessions. "
                "This distribution pattern describes investor flows, not a completed daily candle.",
                note,
                1,
            ),
        }
        for case_id, (text, session_note, expected) in cases.items():
            with self.subTest(case_id=case_id):
                findings = self.scanner(text=text, session_note=session_note, agent_name="offline_eval")
                self.assertEqual(len(findings), expected)

    def test_guard_hash_mismatch_is_rejected(self) -> None:
        with self.assertRaises(ContractViolation) as raised:
            prepare_guard(REPO_ROOT, "0" * 64)
        self.assertEqual(raised.exception.issue.code, "FILE_HASH_MISMATCH")

    def test_finding_mapping_is_candidate_only(self) -> None:
        raw = "offline_eval: [closed_assertion] 'closed at 100' vs SESSION-IN-PROGRESS ground truth"
        mapped = _parse_finding(raw, "The stock closed at 100 today.", 10, 23, 0)
        self.assertEqual(mapped["mapping_status"], "unique")
        self.assertTrue(mapped["candidate_overlaps_claim_span"])
        duplicated = _parse_finding(raw, "closed at 100 then closed at 100", 0, 13, 0)
        self.assertEqual(duplicated["mapping_status"], "ambiguous_multiple_locations")
        self.assertIsNone(duplicated["candidate_overlaps_claim_span"])

    def test_repr_decoder_accepts_only_canonical_bounded_strings(self) -> None:
        for value in ("plain", "한글", "quote ' and newline\n", "tab\tvalue", "\\path"):
            with self.subTest(value=value):
                self.assertEqual(_decode_bounded_repr_string(repr(value), "finding[0]"), value)
        for literal in ("r'raw'", "'''triple'''", "'unknown\\qescape'", "'raw\nline'", '"plain"'):
            with self.subTest(literal=literal):
                with self.assertRaises(ContractViolation):
                    _decode_bounded_repr_string(literal, "finding[0]")

    def test_missing_session_truth_is_not_a_true_negative(self) -> None:
        node = "The stock closed at 100 KRW today."
        claim = {
            "case_id": "C1",
            "decision_id": 1,
            "label": "confirmed_error",
            "path": ["decision_text"],
            "span_start": 10,
            "span_end_exclusive": 23,
        }
        validation = ValidationResult(
            profile_sha256="a" * 64,
            input_file_sha256={"claim-annotations.json": "b" * 64},
            expected_counts={},
            actual_counts={},
            case_statuses=[],
            scanner_source_sha256=self.prepared.source_sha256,
        )
        loaded = LoadedBundle(
            validation=validation,
            profile={
                "candidate_case_ids": ["C1"],
                "session_note_template": SESSION_NOTE_TEMPLATE,
                "context_origin": "constructed_from_stored_metadata",
                "replay_mode": "controlled_context_rescore",
            },
            inventory_by_id={1: {"ticker": "SYNTHETIC", "trade_date": "2026-09-18", "session_in_progress": None}},
            evidence_by_id={1: {"decision_text": node}},
            claims_by_case={"C1": claim},
        )
        starts: list[bool] = []
        result = scan_selected_cases(loaded, self.scanner, on_scanner_start=lambda: starts.append(True))[0]
        self.assertEqual(result["execution_status"], "not_applicable")
        self.assertEqual(result["n_a_reason"], "session_in_progress_not_exact_true")
        self.assertEqual(starts, [])


if __name__ == "__main__":
    unittest.main()
