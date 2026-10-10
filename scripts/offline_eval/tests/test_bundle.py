from pathlib import Path
import json
import shutil
import sys
import tempfile
import unittest


TOOL_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = TOOL_DIR.parents[1]
sys.path.insert(0, str(TOOL_DIR))

from bundle import FIXED_SCANNER_REFERENCE, validate_bundle  # noqa: E402
from contracts import ContractViolation, canonical_json_sha256, sha256_bytes  # noqa: E402


def dump(path: Path, value: object) -> bytes:
    raw = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return raw


class BundleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = Path(tempfile.mkdtemp(prefix="offline-eval-"))
        self.bundle = self.temp / "fixture-bundle"
        self.bundle.mkdir()
        transcript = {"reports": [{"body": "market closed"}]}
        payload = {"full_transcript": transcript, "decision_text": "market closed"}
        evidence_hash = canonical_json_sha256(payload)
        transcript_hash = canonical_json_sha256(transcript)
        inventory = [{
            "id": 1,
            "ticker": "000001.KS",
            "trade_date": "2026-09-21",
            "representative": True,
            "diagnostic": True,
            "split_proposal": "development_diagnostic",
            "evidence_path": "evidence/1.json",
            "evidence_sha256": evidence_hash,
            "transcript_sha256": transcript_hash,
        }]
        node = payload["decision_text"]
        labels = ["confirmed_error", "confirmed_valid", "ambiguous", "unassessable"]
        claims = []
        for index, label in enumerate(labels):
            claims.append({
                "case_id": f"C{index + 1}",
                "decision_id": 1,
                "path": ["decision_text"],
                "needle": node,
                "label": label,
                "review_status": "draft_single_reviewer",
                "span_start": 0,
                "span_end_exclusive": len(node),
                "node_sha256_utf8": sha256_bytes(node.encode("utf-8")),
                "evidence_path": "evidence/1.json",
                "evidence_sha256": evidence_hash,
            })
        split = {
            "items": [{
                "id": 1,
                "split": "development_diagnostic",
                "group_original": "1",
                "group_ticker_day": "000001.KS|2026-09-21",
                "group_transcript": transcript_hash,
                "representative": True,
                "diagnostic": True,
            }],
            "counts": {"development_diagnostic": 1},
            "duplicate_groups": {"group_ticker_day": [], "group_transcript": []},
        }
        summary = {
            "representative_n": 1,
            "diagnostic_found_n": 1,
            "unique_evidence_n": 1,
            "overlap_n": 1,
            "sample_days": [{"date": "2026-09-21", "N": 1, "n": 1, "ids": [1]}],
            "split_counts": {"development_diagnostic": 1},
        }
        files = {
            "inventory.json": inventory,
            "claim-annotations.json": {"items": claims},
            "split-manifest.json": split,
            "collection-summary.json": summary,
        }
        hashes = {name: sha256_bytes(dump(self.bundle / name, value)) for name, value in files.items()}
        dump(self.bundle / "evidence" / "1.json", payload)
        scanner_path = REPO_ROOT / Path(*FIXED_SCANNER_REFERENCE.split("/"))
        profile = {
            "schema_version": "phase3a.scope.v1",
            "source_bundle": "fixture-bundle",
            "replay_mode": "controlled_context_rescore",
            "source_metadata_file_sha256": hashes,
            "scanner": {
                "name": "session_close",
                "source_reference": FIXED_SCANNER_REFERENCE,
                "source_bytes_sha256": sha256_bytes(scanner_path.read_bytes()),
            },
            "candidate_case_ids": ["C1", "C2"],
            "out_of_scope_clear_case_ids": [],
            "expected_counts": {
                "unique_evidence": 1,
                "source_claims": 4,
                "candidates": 2,
                "candidate_error_drafts": 1,
                "candidate_valid_drafts": 1,
                "out_of_scope_clear": 0,
                "ambiguous": 1,
                "unassessable": 1,
            },
            "engine_generation_allowed": False,
            "holdout_scoring_allowed": False,
        }
        self.profile = self.temp / "profile.json"
        dump(self.profile, profile)

    def tearDown(self) -> None:
        shutil.rmtree(self.temp)

    def test_valid_fixture(self) -> None:
        result = validate_bundle(self.bundle, self.profile, REPO_ROOT)
        self.assertEqual(result.actual_counts["candidates"], 2)
        self.assertEqual(result.actual_counts["unique_evidence"], 1)
        self.assertEqual(len(result.case_statuses), 4)

    def test_metadata_hash_mutation_is_rejected(self) -> None:
        with (self.bundle / "inventory.json").open("ab") as handle:
            handle.write(b" ")
        with self.assertRaises(ContractViolation) as raised:
            validate_bundle(self.bundle, self.profile, REPO_ROOT)
        self.assertEqual(raised.exception.issue.code, "FILE_HASH_MISMATCH")

    def test_claim_span_mutation_is_rejected_after_rehash(self) -> None:
        claim_path = self.bundle / "claim-annotations.json"
        claims = json.loads(claim_path.read_text("utf-8"))
        claims["items"][0]["span_end_exclusive"] = 1
        raw = dump(claim_path, claims)
        profile = json.loads(self.profile.read_text("utf-8"))
        profile["source_metadata_file_sha256"]["claim-annotations.json"] = sha256_bytes(raw)
        dump(self.profile, profile)
        with self.assertRaises(ContractViolation) as raised:
            validate_bundle(self.bundle, self.profile, REPO_ROOT)
        self.assertEqual(raised.exception.issue.code, "CLAIM_SPAN_MISMATCH")


if __name__ == "__main__":
    unittest.main()
