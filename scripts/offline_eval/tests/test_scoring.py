from pathlib import Path
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


TOOL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOL_DIR))

import container_contract as cc  # noqa: E402
import scoring  # noqa: E402
from contracts import LABEL_SOURCE_VERSION, ContractViolation, canonical_json_sha256, sha256_bytes  # noqa: E402
from scoring import score_review  # noqa: E402


TOOL_NAMES = (
    "bundle.py", "cli.py", "container_contract.py", "contracts.py", "isolation.py",
    "launcher.py", "report.py", "scoring.py", "session_adapter.py",
)


def write_json(path: Path, value: object) -> bytes:
    raw = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
    path.write_bytes(raw)
    return raw


RUN_NAME = "run-01"


def valid_inspect_subset(nonce_hex: str, *, run_name: str = RUN_NAME) -> dict[str, object]:
    env = sorted(cc.BASE_IMAGE_ENV_ALLOWLIST) + [f"{cc.NONCE_ENV_NAME}={nonce_hex}"]
    mounts = [
        {"Type": "bind", "Destination": destination, "RW": rw}
        for destination, rw in cc.REQUIRED_MOUNT_DESTINATIONS.items()
    ]
    return {
        "Image": cc.IMAGE_ID,
        "HostConfig": {
            "NetworkMode": "none",
            "ReadonlyRootfs": True,
            "Privileged": False,
            "CapDrop": ["ALL"],
            "CapAdd": None,
            "SecurityOpt": ["no-new-privileges:true"],
            "Devices": None,
            "PidsLimit": 64,
            "PidMode": "", "IpcMode": "", "UTSMode": "", "UsernsMode": "",
            "Tmpfs": None, "Binds": None, "VolumesFrom": None,
            "Memory": cc.EXPECTED_MEMORY_BYTES, "NanoCpus": cc.EXPECTED_NANO_CPUS,
            "LogConfig": {"Type": "none"},
        },
        "Config": {
            "User": cc.CONTAINER_USER, "Env": env, "Hostname": cc.CONTAINER_HOSTNAME,
            "Cmd": cc.build_scan_command(run_name=run_name), "Entrypoint": None,
        },
        "Mounts": mounts,
    }


def all_probes_passed() -> list[dict[str, object]]:
    return [
        {"id": probe_id, "passed": True, "reason": "synthetic"}
        for probe_id in ("P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9", "P10", "P11")
    ]


class ScoringTests(unittest.TestCase):
    def setUp(self) -> None:
        # G3: the scorer compares the run's tool hashes against its own,
        # currently-running copy. Tests use synthetic fake hashes, so patch
        # the scorer's own-hash lookup to match -- this is what makes the
        # single-source-of-truth check (§15 G3) test-patchable without
        # touching real files on disk.
        patcher = patch.object(scoring, "_current_tool_hashes")
        self._current_tool_hashes_mock = patcher.start()
        self.addCleanup(patcher.stop)
        self.temp = Path(tempfile.mkdtemp(prefix="offline-score-"))
        self.root = self.temp / "runs"
        self.launch_root = self.temp / "launches"
        self.run = self.root / "run-01"
        self.run.mkdir(parents=True)
        self.launch_root.mkdir(parents=True)
        self.profile_hash = "a" * 64
        self.scanner_hash = "b" * 64
        self.label_hash = "c" * 64
        self.tool_hashes = {name: "9" * 64 for name in TOOL_NAMES}
        self._current_tool_hashes_mock.return_value = dict(self.tool_hashes)
        self.nonce = "1" * 64
        finding = "offline_eval: [closed_assertion] 'closed at 100' vs SESSION-IN-PROGRESS ground truth"
        note = "SESSION IN PROGRESS: SYNTHETIC's regular session is still open as of 2026-09-18."

        def scanned(case_id: str, decision_id: int, label: str, node_hash: str) -> dict[str, object]:
            raw_findings = [finding]
            return {
                "case_id": case_id, "decision_id": decision_id, "label_draft": label,
                "context_origin": "synthetic", "replay_mode": "controlled_context_rescore",
                "execution_status": "scanned", "n_a_reason": None,
                "input_node_sha256": node_hash,
                "session_note": note, "session_note_sha256": sha256_bytes(note.encode()),
                "case_findings_sha256": canonical_json_sha256(raw_findings),
                "raw_findings": raw_findings,
                "mapping_candidates": [{
                    "finding_index": 0,
                    "finding_sha256": sha256_bytes(finding.encode()),
                    "family": "closed_assertion",
                    "quote_sha256": sha256_bytes(b"closed at 100"),
                    "quote_codepoints": 13,
                    "mapping_status": "unique",
                    "candidate_ranges": [{"start": 0, "end_exclusive": 13}],
                    "candidate_overlaps_claim_span": True,
                }],
            }

        cases = [
            scanned("E1", 1, "confirmed_error", "d" * 64),
            scanned("V1", 2, "confirmed_valid", "f" * 64),
            {
                "case_id": "N1", "decision_id": 3, "label_draft": "confirmed_valid",
                "context_origin": "synthetic", "replay_mode": "controlled_context_rescore",
                "execution_status": "not_applicable",
                "n_a_reason": "session_in_progress_not_exact_true",
                "input_node_sha256": "2" * 64,
                "session_note": None, "session_note_sha256": None,
                "case_findings_sha256": canonical_json_sha256([]),
                "raw_findings": [], "mapping_candidates": [],
            },
        ]
        scope_counts = {
            "expected": {
                "unique_evidence": 3, "source_claims": 3, "candidates": 3,
                "candidate_error_drafts": 1, "candidate_valid_drafts": 2,
                "out_of_scope_clear": 0, "ambiguous": 0, "unassessable": 0,
            },
            "actual": {
                "unique_evidence": 3, "source_claims": 3, "candidates": 3,
                "candidate_error_drafts": 1, "candidate_valid_drafts": 2,
                "out_of_scope_clear": 0, "ambiguous": 0, "unassessable": 0,
            },
        }
        scope_statuses = [
            {"case_id": case_id, "decision_id": decision_id, "status": "selected", "reason": "phase3a_session_close_v1"}
            for case_id, decision_id in (("E1", 1), ("V1", 2), ("N1", 3))
        ]
        findings = {
            "schema_version": "offline-eval.findings.v2",
            "profile_sha256": self.profile_hash,
            "scanner_source_sha256": self.scanner_hash,
            "label_source_sha256": self.label_hash,
            "label_source_version": LABEL_SOURCE_VERSION,
            "tool_file_sha256": self.tool_hashes,
            "candidate_case_ids": ["E1", "V1", "N1"],
            "scope_counts": scope_counts,
            "scope_case_statuses": scope_statuses,
            "context_origin": "synthetic",
            "replay_mode": "controlled_context_rescore",
            "cases": cases,
        }
        findings_raw = write_json(self.run / "findings.json", findings)

        isolation = {
            "audit_hook_installed": True,
            "sentinel_blocked": True,
            "os_level_sandbox_confirmed": True,
            "security_boundary": "linux_container_probes_confirmed_plus_python_audit_hook",
            "os_isolation": {
                "status": "confirmed",
                "method": "linux_container_self_probe",
                "reason": "all_probes_passed",
                "probes": all_probes_passed(),
            },
        }
        manifest = {
            "schema_version": "offline-eval.scan-run.v2", "status": "complete",
            "profile_sha256": self.profile_hash,
            "scanner_source_sha256": self.scanner_hash,
            "label_source_sha256": self.label_hash,
            "label_source_version": LABEL_SOURCE_VERSION,
            "input_file_sha256": {"claim-annotations.json": self.label_hash},
            "tool_file_sha256": self.tool_hashes,
            "candidate_case_ids": ["E1", "V1", "N1"],
            "scope_counts": scope_counts,
            "scope_case_statuses": scope_statuses,
            "context_origin": "synthetic",
            "replay_mode": "controlled_context_rescore",
            "counts": {
                "applicable_candidates": 3, "scanned": 2, "n_a": 1,
                "raw_findings": 2, "unique_mappings": 2, "ambiguous_mappings": 0,
            },
            "findings_file_sha256": sha256_bytes(findings_raw),
            "isolation": isolation,
            "launch_nonce": self.nonce,
            "python_version": "3.12.13",
            "container_contract_sha256": self.tool_hashes["container_contract.py"],
        }
        manifest_raw = write_json(self.run / "run-manifest.json", manifest)

        attestation_dir = self.launch_root / "run-01"
        attestation_dir.mkdir(parents=True)
        attestation = {
            "schema_version": cc.ATTESTATION_SCHEMA,
            "run_name": "run-01",
            "launch_nonce": self.nonce,
            "image_ref": cc.IMAGE_REF,
            "image_id": cc.IMAGE_ID,
            "python_version": "3.12.13",
            "container_contract_sha256": self.tool_hashes["container_contract.py"],
            "tool_file_sha256": self.tool_hashes,
            "guard_source_sha256": self.scanner_hash,
            "run_manifest_sha256": sha256_bytes(manifest_raw),
            "inspect_subset": valid_inspect_subset(self.nonce),
            "container_exit_code": 0,
        }
        attestation_raw = write_json(attestation_dir / "launch-attestation.json", attestation)

        source = {
            "run_manifest_sha256": sha256_bytes(manifest_raw),
            "findings_file_sha256": sha256_bytes(findings_raw),
            "profile_sha256": self.profile_hash,
            "scanner_source_sha256": self.scanner_hash,
            "label_source_sha256": self.label_hash,
            "label_source_version": LABEL_SOURCE_VERSION,
            "launch_attestation_sha256": sha256_bytes(attestation_raw),
        }
        common = {
            "profile_sha256": self.profile_hash,
            "scanner_source_sha256": self.scanner_hash,
            "label_source_sha256": self.label_hash,
            "verdict": "hit", "finding_indices": [0],
            "rationale": "Reviewed the exact claim span and finding family.",
            "reviewer": "reviewer-1", "label_version": LABEL_SOURCE_VERSION,
            "label_review_status": "approved",
        }
        review = {
            "schema_version": "offline-eval.hit-review.v2", "source": source,
            "items": [
                {**common, "case_id": "E1", "input_node_sha256": "d" * 64,
                 "case_findings_sha256": cases[0]["case_findings_sha256"], "final_label": "confirmed_error"},
                {**common, "case_id": "V1", "input_node_sha256": "f" * 64,
                 "case_findings_sha256": cases[1]["case_findings_sha256"], "final_label": "confirmed_valid"},
            ],
        }
        self.review = self.run / "hit-review-v1.json"
        write_json(self.review, review)

    def tearDown(self) -> None:
        shutil.rmtree(self.temp)

    def _score(self):
        return score_review(self.run, self.review, self.root, self.launch_root)

    def _rewrite_manifest_and_findings(self, *, manifest=None, findings=None):
        if findings is not None:
            findings_raw = write_json(self.run / "findings.json", findings)
            if manifest is None:
                manifest = json.loads((self.run / "run-manifest.json").read_text("utf-8"))
            manifest["findings_file_sha256"] = sha256_bytes(findings_raw)
        if manifest is not None:
            manifest_raw = write_json(self.run / "run-manifest.json", manifest)
            # Re-point the attestation and the review's binding at the freshly
            # written manifest -- these helper-driven tests exercise semantic
            # (RR1-RR3-style) checks, not the S1-S10 attestation gate, which has
            # its own dedicated tests below.
            attestation_path = self.launch_root / "run-01" / "launch-attestation.json"
            attestation = json.loads(attestation_path.read_text("utf-8"))
            attestation["run_manifest_sha256"] = sha256_bytes(manifest_raw)
            attestation_raw = write_json(attestation_path, attestation)
            review = json.loads(self.review.read_text("utf-8"))
            review["source"]["run_manifest_sha256"] = sha256_bytes(manifest_raw)
            review["source"]["launch_attestation_sha256"] = sha256_bytes(attestation_raw)
            write_json(self.review, review)

    def test_review_bound_scoring_and_na(self) -> None:
        result = self._score()
        self.assertEqual(result["counts"]["tp"], 1)
        self.assertEqual(result["counts"]["fp"], 1)
        self.assertEqual(result["counts"]["n_a"], 1)
        self.assertEqual(result["metrics"]["precision"], 0.5)
        self.assertEqual(result["metrics"]["recall"], 1.0)

    def test_stale_review_hash_is_rejected(self) -> None:
        review = json.loads(self.review.read_text("utf-8"))
        review["source"]["profile_sha256"] = "8" * 64
        write_json(self.review, review)
        with self.assertRaises(ContractViolation) as raised:
            self._score()
        self.assertEqual(raised.exception.issue.code, "REVIEW_HASH_MISMATCH")

    def test_pending_label_is_na_even_with_a_hit_verdict(self) -> None:
        review = json.loads(self.review.read_text("utf-8"))
        for item in review["items"]:
            item["label_review_status"] = "pending"
            item["final_label"] = None
        write_json(self.review, review)
        result = self._score()
        self.assertEqual(result["counts"]["scored"], 0)
        self.assertEqual(result["counts"]["n_a"], 3)
        self.assertIsNone(result["metrics"]["precision"])

    def test_case_findings_hash_is_recomputed(self) -> None:
        findings = json.loads((self.run / "findings.json").read_text("utf-8"))
        findings["cases"][0]["case_findings_sha256"] = "e" * 64
        self._rewrite_manifest_and_findings(findings=findings)
        with self.assertRaises(ContractViolation) as raised:
            self._score()
        self.assertEqual(raised.exception.issue.code, "FILE_HASH_MISMATCH")

    def test_unknown_execution_status_is_rejected(self) -> None:
        findings = json.loads((self.run / "findings.json").read_text("utf-8"))
        findings["cases"][0]["execution_status"] = "mystery"
        self._rewrite_manifest_and_findings(findings=findings)
        with self.assertRaises(ContractViolation) as raised:
            self._score()
        self.assertEqual(raised.exception.issue.code, "INVALID_ENUM")

    def test_mapping_family_and_quote_metadata_are_recomputed(self) -> None:
        baseline_findings = json.loads((self.run / "findings.json").read_text("utf-8"))
        baseline_manifest = json.loads((self.run / "run-manifest.json").read_text("utf-8"))
        mutations = (
            ("family", "completed_candle", "REFERENCE_MISMATCH"),
            ("quote_sha256", "0" * 64, "FILE_HASH_MISMATCH"),
            ("quote_codepoints", 999, "COUNT_MISMATCH"),
        )
        for field, value, expected_code in mutations:
            with self.subTest(field=field):
                findings = json.loads(json.dumps(baseline_findings))
                manifest = json.loads(json.dumps(baseline_manifest))
                findings["cases"][0]["mapping_candidates"][0][field] = value
                self._rewrite_manifest_and_findings(manifest=manifest, findings=findings)
                with self.assertRaises(ContractViolation) as raised:
                    self._score()
                self.assertEqual(raised.exception.issue.code, expected_code)

    def test_missing_label_approval_is_na(self) -> None:
        review = json.loads(self.review.read_text("utf-8"))
        for item in review["items"]:
            item.pop("label_review_status")
            item.pop("final_label")
            item.pop("label_version")
        write_json(self.review, review)
        result = self._score()
        self.assertEqual(result["counts"]["scored"], 0)
        self.assertEqual(result["counts"]["n_a"], 3)
        self.assertEqual(
            {row.get("reason") for row in result["case_statuses"]},
            {"label_review_missing", "session_in_progress_not_exact_true"},
        )

    def test_changed_label_requires_new_version(self) -> None:
        review = json.loads(self.review.read_text("utf-8"))
        review["items"][0]["final_label"] = "confirmed_valid"
        review["items"][0]["label_change_rationale"] = "Synthetic correction."
        write_json(self.review, review)
        with self.assertRaises(ContractViolation) as raised:
            self._score()
        self.assertEqual(raised.exception.issue.code, "REFERENCE_MISMATCH")

        review["items"][0]["label_version"] = "claim-annotations-v2"
        write_json(self.review, review)
        result = self._score()
        self.assertEqual(result["counts"]["tp"], 0)
        self.assertEqual(result["counts"]["fp"], 2)

    def test_unknown_scope_count_key_is_rejected_without_echo(self) -> None:
        marker = "SYNTHETIC_PRIVATE_MARKER"
        findings = json.loads((self.run / "findings.json").read_text("utf-8"))
        findings["scope_counts"]["actual"][marker] = 1
        manifest = json.loads((self.run / "run-manifest.json").read_text("utf-8"))
        manifest["scope_counts"] = findings["scope_counts"]
        self._rewrite_manifest_and_findings(manifest=manifest, findings=findings)
        with self.assertRaises(ContractViolation) as raised:
            self._score()
        self.assertEqual(raised.exception.issue.reference, "findings.scope_counts.actual")
        self.assertNotIn(marker, str(raised.exception.issue.as_dict()))

    # --- D2 gate: manifest v1 / unconfirmed isolation / attestation mismatches (S1-S9) ---

    def test_v1_manifest_is_rejected(self) -> None:
        manifest = json.loads((self.run / "run-manifest.json").read_text("utf-8"))
        manifest["schema_version"] = "offline-eval.scan-run.v1"
        write_json(self.run / "run-manifest.json", manifest)
        with self.assertRaises(ContractViolation) as raised:
            self._score()
        self.assertEqual(raised.exception.issue.code, "INVALID_ENUM")

    def test_unconfirmed_isolation_is_rejected(self) -> None:
        manifest = json.loads((self.run / "run-manifest.json").read_text("utf-8"))
        manifest["isolation"]["os_isolation"]["status"] = "verification_failed"
        manifest["isolation"]["os_level_sandbox_confirmed"] = False
        write_json(self.run / "run-manifest.json", manifest)
        with self.assertRaises(ContractViolation) as raised:
            self._score()
        self.assertEqual(raised.exception.issue.code, "ISOLATION_NOT_CONFIRMED")

    def test_one_failed_probe_among_confirmed_status_is_rejected(self) -> None:
        # An internally-inconsistent manifest: status says confirmed but a probe
        # in its own list says otherwise. Must not be trusted on the status alone.
        manifest = json.loads((self.run / "run-manifest.json").read_text("utf-8"))
        manifest["isolation"]["os_isolation"]["probes"][2]["passed"] = False
        write_json(self.run / "run-manifest.json", manifest)
        with self.assertRaises(ContractViolation) as raised:
            self._score()
        self.assertEqual(raised.exception.issue.code, "ISOLATION_NOT_CONFIRMED")

    def test_missing_attestation_is_rejected(self) -> None:
        shutil.rmtree(self.launch_root / "run-01")
        with self.assertRaises(ContractViolation) as raised:
            self._score()
        self.assertEqual(raised.exception.issue.code, "ATTESTATION_MISSING")

    def test_stale_attestation_manifest_hash_is_rejected(self) -> None:
        attestation_path = self.launch_root / "run-01" / "launch-attestation.json"
        attestation = json.loads(attestation_path.read_text("utf-8"))
        attestation["run_manifest_sha256"] = "7" * 64
        write_json(attestation_path, attestation)
        with self.assertRaises(ContractViolation) as raised:
            self._score()
        self.assertEqual(raised.exception.issue.code, "ATTESTATION_HASH_MISMATCH")

    def test_attestation_nonce_mismatch_is_rejected(self) -> None:
        attestation_path = self.launch_root / "run-01" / "launch-attestation.json"
        attestation = json.loads(attestation_path.read_text("utf-8"))
        attestation["launch_nonce"] = "2" * 64
        write_json(attestation_path, attestation)
        with self.assertRaises(ContractViolation) as raised:
            self._score()
        self.assertEqual(raised.exception.issue.code, "ATTESTATION_NONCE_MISMATCH")

    def test_attestation_inspect_violation_is_rejected(self) -> None:
        attestation_path = self.launch_root / "run-01" / "launch-attestation.json"
        attestation = json.loads(attestation_path.read_text("utf-8"))
        attestation["inspect_subset"]["HostConfig"]["NetworkMode"] = "bridge"
        write_json(attestation_path, attestation)
        with self.assertRaises(ContractViolation) as raised:
            self._score()
        self.assertEqual(raised.exception.issue.code, "ATTESTATION_INSPECT_INVALID")

    def test_attestation_nonzero_exit_is_rejected(self) -> None:
        attestation_path = self.launch_root / "run-01" / "launch-attestation.json"
        attestation = json.loads(attestation_path.read_text("utf-8"))
        attestation["container_exit_code"] = 1
        write_json(attestation_path, attestation)
        with self.assertRaises(ContractViolation) as raised:
            self._score()
        self.assertEqual(raised.exception.issue.code, "CONTAINER_EXIT_NONZERO")

    def test_review_missing_attestation_binding_is_rejected(self) -> None:
        review = json.loads(self.review.read_text("utf-8"))
        review["source"].pop("launch_attestation_sha256")
        write_json(self.review, review)
        with self.assertRaises(ContractViolation) as raised:
            self._score()
        self.assertEqual(raised.exception.issue.code, "INVALID_TYPE")

    def test_review_stale_attestation_binding_is_rejected(self) -> None:
        review = json.loads(self.review.read_text("utf-8"))
        review["source"]["launch_attestation_sha256"] = "6" * 64
        write_json(self.review, review)
        with self.assertRaises(ContractViolation) as raised:
            self._score()
        self.assertEqual(raised.exception.issue.code, "REVIEW_HASH_MISMATCH")

    # --- §14.2: one-condition-at-a-time S1/S2/S5/S6/S7/S10 tests (review #7) ---

    def test_s1_missing_probe_is_rejected(self) -> None:
        manifest = json.loads((self.run / "run-manifest.json").read_text("utf-8"))
        manifest["isolation"]["os_isolation"]["probes"] = manifest["isolation"]["os_isolation"]["probes"][:-1]
        write_json(self.run / "run-manifest.json", manifest)
        with self.assertRaises(ContractViolation) as raised:
            self._score()
        self.assertEqual(raised.exception.issue.code, "ISOLATION_NOT_CONFIRMED")

    def test_s1_duplicate_probe_id_is_rejected(self) -> None:
        manifest = json.loads((self.run / "run-manifest.json").read_text("utf-8"))
        manifest["isolation"]["os_isolation"]["probes"].append({"id": "P1", "passed": True, "reason": "dup"})
        write_json(self.run / "run-manifest.json", manifest)
        with self.assertRaises(ContractViolation) as raised:
            self._score()
        self.assertEqual(raised.exception.issue.code, "ISOLATION_NOT_CONFIRMED")

    def test_s1_non_bool_passed_is_rejected(self) -> None:
        manifest = json.loads((self.run / "run-manifest.json").read_text("utf-8"))
        manifest["isolation"]["os_isolation"]["probes"][0]["passed"] = 1
        write_json(self.run / "run-manifest.json", manifest)
        with self.assertRaises(ContractViolation) as raised:
            self._score()
        self.assertEqual(raised.exception.issue.code, "ISOLATION_NOT_CONFIRMED")

    def test_s1_security_boundary_alone_is_rejected(self) -> None:
        manifest = json.loads((self.run / "run-manifest.json").read_text("utf-8"))
        manifest["isolation"]["security_boundary"] = "some_other_boundary"
        write_json(self.run / "run-manifest.json", manifest)
        with self.assertRaises(ContractViolation) as raised:
            self._score()
        self.assertEqual(raised.exception.issue.code, "ISOLATION_NOT_CONFIRMED")

    def test_s1_os_level_sandbox_confirmed_alone_is_rejected(self) -> None:
        manifest = json.loads((self.run / "run-manifest.json").read_text("utf-8"))
        manifest["isolation"]["os_level_sandbox_confirmed"] = False
        write_json(self.run / "run-manifest.json", manifest)
        with self.assertRaises(ContractViolation) as raised:
            self._score()
        self.assertEqual(raised.exception.issue.code, "ISOLATION_NOT_CONFIRMED")

    def test_s5_findings_manifest_tool_hash_mismatch_is_rejected(self) -> None:
        findings = json.loads((self.run / "findings.json").read_text("utf-8"))
        findings["tool_file_sha256"] = dict(findings["tool_file_sha256"])
        findings["tool_file_sha256"]["cli.py"] = "0" * 64
        self._rewrite_manifest_and_findings(findings=findings)
        with self.assertRaises(ContractViolation) as raised:
            self._score()
        self.assertEqual(raised.exception.issue.code, "REFERENCE_MISMATCH")
        self.assertEqual(raised.exception.issue.reference, "tool_file_sha256")

    def test_s2_reparse_path_is_rejected(self) -> None:
        # §15 G11: a junction in place of the attestation's run-name directory
        # must be treated the same as "attestation missing" (fail-closed).
        attestation_dir = self.launch_root / "run-01"
        real_target = self.temp / "reparse-target"
        real_target.mkdir()
        shutil.move(str(attestation_dir), str(real_target / "run-01"))
        result = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(attestation_dir), str(real_target / "run-01")],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            self.skipTest(f"could not create a junction without elevated privileges: {result.stderr.strip()}")
        with self.assertRaises(ContractViolation) as raised:
            self._score()
        self.assertEqual(raised.exception.issue.code, "ATTESTATION_MISSING")

    def test_s2_attestation_directory_instead_of_file_is_rejected(self) -> None:
        attestation_path = self.launch_root / "run-01" / cc.ATTESTATION_FILENAME
        attestation_path.unlink()
        attestation_path.mkdir()
        with self.assertRaises(ContractViolation) as raised:
            self._score()
        self.assertEqual(raised.exception.issue.code, "ATTESTATION_MISSING")

    def test_s5_attestation_tool_hash_mismatch_is_rejected(self) -> None:
        attestation_path = self.launch_root / "run-01" / cc.ATTESTATION_FILENAME
        attestation = json.loads(attestation_path.read_text("utf-8"))
        attestation["tool_file_sha256"]["cli.py"] = "0" * 64
        write_json(attestation_path, attestation)
        with self.assertRaises(ContractViolation) as raised:
            self._score()
        self.assertEqual(raised.exception.issue.code, "ATTESTATION_TOOL_HASH_MISMATCH")

    def test_s6_image_ref_not_pinned_value_is_rejected(self) -> None:
        attestation_path = self.launch_root / "run-01" / cc.ATTESTATION_FILENAME
        attestation = json.loads(attestation_path.read_text("utf-8"))
        attestation["image_ref"] = "docker.io/library/python@sha256:" + "0" * 64
        write_json(attestation_path, attestation)
        with self.assertRaises(ContractViolation) as raised:
            self._score()
        self.assertEqual(raised.exception.issue.code, "ATTESTATION_CONTRACT_MISMATCH")
        self.assertEqual(raised.exception.issue.reference, "attestation.image_ref")

    def test_s6_python_version_matching_but_not_pinned_is_rejected(self) -> None:
        # Both sides agree with each other but not with the pinned constant --
        # must still be rejected (review #5).
        attestation_path = self.launch_root / "run-01" / cc.ATTESTATION_FILENAME
        attestation = json.loads(attestation_path.read_text("utf-8"))
        attestation["python_version"] = "3.11.0"
        write_json(attestation_path, attestation)
        manifest = json.loads((self.run / "run-manifest.json").read_text("utf-8"))
        manifest["python_version"] = "3.11.0"
        self._rewrite_manifest_and_findings(manifest=manifest)
        with self.assertRaises(ContractViolation) as raised:
            self._score()
        self.assertEqual(raised.exception.issue.code, "ATTESTATION_CONTRACT_MISMATCH")
        self.assertEqual(raised.exception.issue.reference, "attestation.python_version")

    def test_s6_contract_hash_not_matching_tool_hash_set_is_rejected(self) -> None:
        attestation_path = self.launch_root / "run-01" / cc.ATTESTATION_FILENAME
        attestation = json.loads(attestation_path.read_text("utf-8"))
        attestation["container_contract_sha256"] = "0" * 64
        write_json(attestation_path, attestation)
        with self.assertRaises(ContractViolation) as raised:
            self._score()
        self.assertEqual(raised.exception.issue.code, "ATTESTATION_CONTRACT_MISMATCH")
        self.assertEqual(raised.exception.issue.reference, "attestation.container_contract_sha256")

    def test_s6_image_id_not_pinned_value_is_rejected(self) -> None:
        attestation_path = self.launch_root / "run-01" / cc.ATTESTATION_FILENAME
        attestation = json.loads(attestation_path.read_text("utf-8"))
        attestation["image_id"] = "sha256:" + "0" * 64
        write_json(attestation_path, attestation)
        with self.assertRaises(ContractViolation) as raised:
            self._score()
        self.assertEqual(raised.exception.issue.code, "ATTESTATION_CONTRACT_MISMATCH")
        self.assertEqual(raised.exception.issue.reference, "attestation.image_id")

    def test_g1_guard_hash_mismatch_is_rejected(self) -> None:
        attestation_path = self.launch_root / "run-01" / cc.ATTESTATION_FILENAME
        attestation = json.loads(attestation_path.read_text("utf-8"))
        attestation["guard_source_sha256"] = "0" * 64
        write_json(attestation_path, attestation)
        with self.assertRaises(ContractViolation) as raised:
            self._score()
        self.assertEqual(raised.exception.issue.code, "ATTESTATION_GUARD_HASH_MISMATCH")

    def test_g3_tool_hash_not_matching_running_scorer_is_rejected(self) -> None:
        # §15 G3/N3: a run whose tool_file_sha256 set is internally consistent
        # (matches attestation and findings) but does not match *this* scorer's
        # own currently-running source must still be rejected.
        self._current_tool_hashes_mock.return_value = {
            name: ("8" * 64 if name == "cli.py" else h) for name, h in self.tool_hashes.items()
        }
        with self.assertRaises(ContractViolation) as raised:
            self._score()
        self.assertEqual(raised.exception.issue.code, "TOOL_VERSION_MISMATCH")

    def test_s7_replays_f4_mutations_against_the_stored_subset(self) -> None:
        attestation_path = self.launch_root / "run-01" / cc.ATTESTATION_FILENAME
        baseline = json.loads(attestation_path.read_text("utf-8"))
        mutations = {
            "hostname": lambda subset: subset["Config"].__setitem__("Hostname", "other"),
            "cmd": lambda subset: subset["Config"].__setitem__("Cmd", ["python3"]),
            "entrypoint": lambda subset: subset["Config"].__setitem__("Entrypoint", ["/bin/sh"]),
            "memory": lambda subset: subset["HostConfig"].__setitem__("Memory", 1),
            "extra_mount": lambda subset: subset["Mounts"].append(
                {"Type": "bind", "Destination": "/etc/hosts", "RW": True}
            ),
            # §15 G11: replay the rest of F4's validate_inspect checks too, not
            # just a handful -- this is the scorer replaying the *same*
            # validator the launcher used (S7), so every check it enforces
            # must also be exercised against the stored attestation subset.
            "network_mode": lambda subset: subset["HostConfig"].__setitem__("NetworkMode", "bridge"),
            "readonly_rootfs": lambda subset: subset["HostConfig"].__setitem__("ReadonlyRootfs", False),
            "privileged": lambda subset: subset["HostConfig"].__setitem__("Privileged", True),
            "cap_drop": lambda subset: subset["HostConfig"].__setitem__("CapDrop", []),
            "cap_add": lambda subset: subset["HostConfig"].__setitem__("CapAdd", ["SYS_ADMIN"]),
            "security_opt": lambda subset: subset["HostConfig"].__setitem__(
                "SecurityOpt", ["no-new-privileges:true", "seccomp=unconfined"]
            ),
            "group_add": lambda subset: subset["HostConfig"].__setitem__("GroupAdd", ["1000"]),
            "devices": lambda subset: subset["HostConfig"].__setitem__("Devices", [{"PathOnHost": "/dev/kvm"}]),
            "pids_limit": lambda subset: subset["HostConfig"].__setitem__("PidsLimit", 0),
            "pid_mode": lambda subset: subset["HostConfig"].__setitem__("PidMode", "host"),
            "ipc_mode": lambda subset: subset["HostConfig"].__setitem__("IpcMode", "container:other"),
            "uts_mode": lambda subset: subset["HostConfig"].__setitem__("UTSMode", "host"),
            "userns_mode": lambda subset: subset["HostConfig"].__setitem__("UsernsMode", "host"),
            "tmpfs": lambda subset: subset["HostConfig"].__setitem__("Tmpfs", {"/tmp": "size=1"}),
            "binds": lambda subset: subset["HostConfig"].__setitem__("Binds", ["/:/host:ro"]),
            "volumes_from": lambda subset: subset["HostConfig"].__setitem__("VolumesFrom", ["other"]),
            "nano_cpus": lambda subset: subset["HostConfig"].__setitem__("NanoCpus", 1),
            "log_config": lambda subset: subset["HostConfig"].__setitem__("LogConfig", {"Type": "json-file"}),
            "user": lambda subset: subset["Config"].__setitem__("User", "root"),
            "env_extra": lambda subset: subset["Config"].__setitem__(
                "Env", subset["Config"]["Env"] + ["HOST_CANARY=leaked"]
            ),
            "missing_mount": lambda subset: subset["Mounts"].pop(),
            "mount_rw_flipped": lambda subset: subset["Mounts"][-1].__setitem__(
                "RW", not subset["Mounts"][-1]["RW"]
            ),
            "image": lambda subset: subset.__setitem__("Image", "sha256:" + "0" * 64),
        }
        for name, mutate in mutations.items():
            with self.subTest(mutation=name):
                attestation = json.loads(json.dumps(baseline))
                mutate(attestation["inspect_subset"])
                write_json(attestation_path, attestation)
                with self.assertRaises(ContractViolation) as raised:
                    self._score()
                self.assertEqual(raised.exception.issue.code, "ATTESTATION_INSPECT_INVALID")
        write_json(attestation_path, baseline)

    def test_s10_error_output_does_not_reflect_attestation_content(self) -> None:
        marker = "SYNTHETIC_ATTESTATION_MARKER"
        attestation_path = self.launch_root / "run-01" / cc.ATTESTATION_FILENAME
        attestation = json.loads(attestation_path.read_text("utf-8"))
        attestation["image_ref"] = marker
        write_json(attestation_path, attestation)
        with self.assertRaises(ContractViolation) as raised:
            self._score()
        self.assertEqual(raised.exception.issue.code, "ATTESTATION_CONTRACT_MISMATCH")
        self.assertNotIn(marker, str(raised.exception.issue.as_dict()))


if __name__ == "__main__":
    unittest.main()
