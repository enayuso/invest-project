"""Synthetic reviewer regressions across guard, artifacts, and scoring."""

from contextlib import ExitStack, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
import copy
import io
import json
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


TOOL_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = TOOL_DIR.parents[1]
sys.path.insert(0, str(TOOL_DIR))

import cli  # noqa: E402
import container_contract as cc  # noqa: E402
import report  # noqa: E402
from bundle import LoadedBundle, _require_count_map  # noqa: E402
from contracts import ContractViolation, LABEL_SOURCE_VERSION, ValidationResult, sha256_bytes  # noqa: E402
from isolation import IsolationViolation, OsIsolationAssessment, ProbeResult, RuntimePolicy, install_runtime_policy  # noqa: E402
from scoring import score_review  # noqa: E402
from session_adapter import SESSION_NOTE_TEMPLATE, instantiate_guard, prepare_guard, scan_selected_cases  # noqa: E402


SYNTHETIC_NONCE = "3" * 64


def write_json(path, value):
    raw = (json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    path.write_bytes(raw)
    return sha256_bytes(raw)


def _confirmed_isolation_assessment() -> OsIsolationAssessment:
    probes = tuple(
        ProbeResult(probe_id, True, "synthetic_test_only")
        for probe_id in ("P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9", "P10", "P11")
    )
    return OsIsolationAssessment("confirmed", "linux_container_self_probe", "all_probes_passed", probes)


def _valid_inspect_subset(nonce_hex: str, *, run_name: str) -> dict:
    env = sorted(cc.BASE_IMAGE_ENV_ALLOWLIST) + [f"{cc.NONCE_ENV_NAME}={nonce_hex}"]
    mounts = [
        {"Type": "bind", "Destination": destination, "RW": rw}
        for destination, rw in cc.REQUIRED_MOUNT_DESTINATIONS.items()
    ]
    return {
        "Image": cc.IMAGE_ID,
        "HostConfig": {
            "NetworkMode": "none", "ReadonlyRootfs": True, "Privileged": False,
            "CapDrop": ["ALL"], "CapAdd": None, "SecurityOpt": ["no-new-privileges:true"],
            "Devices": None, "PidsLimit": 64,
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


def synthetic_writer(output):
    # No evidence bundle is read. 합성 문장만 고정 가드에 입력한다.
    prepared = prepare_guard(REPO_ROOT, "d3287e59b47cf52a9eaebd3e5d3a4bd59b38377da359ccf17d2ef80d18ba5072")
    note = SESSION_NOTE_TEMPLATE.format(ticker="SYNTHETIC", trade_date="2026-09-18")
    texts = [
        "The stock closed at 100 KRW today.",
        "The previous close was 100 KRW.",
        "If it closes below 100 KRW today, we would reassess.",
        note,
        "The stock closed at 100 KRW today.",
        "The stock closed at 100 KRW on September 17.",
        "The stock closed at 100 KRW on September 18.",
        "The stock closed at 100 KRW today. A rebound is likely.",
        "Foreign investors were net sellers for five sessions. This distribution pattern describes investor flows, not a completed daily candle.",
    ]
    ids = [f"S{i:02}" for i in range(1, 10)]
    statuses = [dict(case_id=key, decision_id=i, status="selected", reason="phase3a_session_close_v1") for i, key in enumerate(ids, 1)]
    for prefix, count, reason in (("X", 17, "clear_label_outside_session_close_v1"), ("A", 13, "ambiguous_label"), ("U", 3, "unassessable_label")):
        statuses.extend(dict(case_id=f"{prefix}{i:02}", decision_id=100 + i, status="excluded", reason=reason) for i in range(count))
    counts = dict(unique_evidence=42, source_claims=42, candidates=9, candidate_error_drafts=3, candidate_valid_drafts=6, out_of_scope_clear=17, ambiguous=13, unassessable=3)
    loaded = LoadedBundle(
        ValidationResult("a" * 64, {"claim-annotations.json": "b" * 64}, counts.copy(), counts.copy(), statuses, prepared.source_sha256),
        dict(candidate_case_ids=ids, session_note_template=SESSION_NOTE_TEMPLATE, context_origin="synthetic", replay_mode="controlled_context_rescore"),
        {i: dict(ticker="SYNTHETIC", trade_date="2026-09-18", session_in_progress=None if i == 5 else True) for i in range(1, 10)},
        {i: dict(text=text) for i, text in enumerate(texts, 1)},
        {key: dict(case_id=key, decision_id=i, label="confirmed_error" if i in (1, 7, 8) else "confirmed_valid", path=["text"], span_start=0, span_end_exclusive=len(text)) for i, (key, text) in enumerate(zip(ids, texts), 1)},
    )
    hashes = report.collect_tool_hashes(TOOL_DIR)
    isolation = install_runtime_policy(
        RuntimePolicy(scanner_filename=prepared.code.co_filename, output_directory=output),
        os_isolation=_confirmed_isolation_assessment(),
    )
    cases = scan_selected_cases(loaded, instantiate_guard(prepared))
    blocked = []
    for event, args in (("socket.connect", (None, None)), ("subprocess.Popen", ()), ("compile", (b"1", "<probe>")), ("open", (str(output / "forbidden.txt"), "w", 0))):
        try:
            sys.audit(event, *args)
        except IsolationViolation:
            blocked.append(event)
    manifest = report.write_scan_artifacts(
        output, loaded, cases, isolation, hashes, launch_nonce=SYNTHETIC_NONCE, python_version="3.12.13"
    )
    # Cannot read the file back, or write the attestation file, from inside this
    # process: the audit hook installed above only allows an exclusive-create open
    # inside output_directory, nothing else. This mirrors the real split -- the host
    # launcher (never inside the audit-hook policy) is the one that reads
    # run-manifest.json back and writes the attestation, after the container has
    # exited. Recompute byte-identical manifest bytes locally (same as
    # report._pretty_bytes) and hand the attestation to the parent process via stdout.
    manifest_raw = (json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
    attestation = {
        "schema_version": cc.ATTESTATION_SCHEMA,
        "run_name": output.name,
        "launch_nonce": SYNTHETIC_NONCE,
        "image_ref": cc.IMAGE_REF,
        "image_id": cc.IMAGE_ID,
        "python_version": manifest["python_version"],
        "container_contract_sha256": hashes["container_contract.py"],
        "tool_file_sha256": dict(sorted(hashes.items())),
        "guard_source_sha256": prepared.source_sha256,
        "run_manifest_sha256": sha256_bytes(manifest_raw),
        "inspect_subset": _valid_inspect_subset(SYNTHETIC_NONCE, run_name=output.name),
        "container_exit_code": 0,
    }
    print(json.dumps(dict(
        blocked=blocked,
        hits=[len(case["raw_findings"]) for case in cases],
        attestation=attestation,
    )))


class ReviewIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="offline-review-")
        cls.root = Path(cls.temp.name)
        cls.run_dir = cls.root / "synthetic-run"
        cls.run_dir.mkdir()
        cls.launch_root = cls.root / "launches"
        cls.launch_dir = cls.launch_root / "synthetic-run"
        cls.hashes = report.collect_tool_hashes(TOOL_DIR)
        child = subprocess.run(
            [str(REPO_ROOT / ".venv" / "Scripts" / "python.exe"), "-I", "-S", "-B", str(Path(__file__).resolve()), "--synthetic-writer", str(cls.run_dir)],
            capture_output=True, text=True, encoding="utf-8", timeout=45,
        )
        if child.returncode or child.stderr:
            cls.temp.cleanup()
            raise AssertionError(f"Synthetic writer failed: {child.returncode}: {child.stderr}")
        cls.child_result = json.loads(child.stdout)
        cls.findings = json.loads((cls.run_dir / "findings.json").read_text("utf-8"))
        cls.manifest = json.loads((cls.run_dir / "run-manifest.json").read_text("utf-8"))
        cls.launch_dir.mkdir(parents=True)
        attestation_path = cls.launch_dir / "launch-attestation.json"
        write_json(attestation_path, cls.child_result["attestation"])
        cls.attestation_raw = attestation_path.read_bytes()
        common = {key: cls.findings[key] for key in ("profile_sha256", "scanner_source_sha256", "label_source_sha256")}
        cls.review = dict(
            schema_version="offline-eval.hit-review.v2",
            source={
                **common,
                "label_source_version": LABEL_SOURCE_VERSION,
                "launch_attestation_sha256": sha256_bytes(cls.attestation_raw),
            },
            items=[],
        )
        for case in cls.findings["cases"]:
            if case["execution_status"] == "scanned":
                cls.review["items"].append({**common, "case_id": case["case_id"], "input_node_sha256": case["input_node_sha256"], "case_findings_sha256": case["case_findings_sha256"], "verdict": "hit" if case["case_id"] in {"S01", "S07", "S09"} else "no_hit", "finding_indices": [0] if case["case_id"] in {"S01", "S07", "S09"} else [], "rationale": "Predefined synthetic full-text claim and independent expected verdict.", "reviewer": "synthetic-review", "label_version": LABEL_SOURCE_VERSION, "label_review_status": "approved", "final_label": case["label_draft"]})

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def score_variant(self, mutate=lambda findings, manifest, review: None):
        findings, manifest, review = copy.deepcopy((self.findings, self.manifest, self.review))
        mutate(findings, manifest, review)
        # Rebind outer hashes so semantic checks, not stale-review checks, reject mutations.
        # 외부 해시를 모두 맞춰 내부 의미 검증을 시험한다. The attestation is re-pointed at the
        # freshly written manifest for the same reason -- these tests exercise RR1-RR3 semantic
        # checks, not the S1-S10 attestation gate (covered separately in test_scoring.py).
        manifest["findings_file_sha256"] = write_json(self.run_dir / "findings.json", findings)
        review["source"]["findings_file_sha256"] = manifest["findings_file_sha256"]
        manifest_hash = write_json(self.run_dir / "run-manifest.json", manifest)
        review["source"]["run_manifest_sha256"] = manifest_hash
        attestation = json.loads(self.attestation_raw.decode("utf-8"))
        attestation["run_manifest_sha256"] = manifest_hash
        review["source"]["launch_attestation_sha256"] = write_json(
            self.launch_dir / "launch-attestation.json", attestation
        )
        path = self.run_dir / "hit-review-v1.json"
        write_json(path, review)
        return score_review(self.run_dir, path, self.root, self.launch_root)

    def test_guard_writer_review_score_and_isolation(self):
        self.assertEqual(self.child_result["hits"], [1, 0, 0, 0, 0, 0, 1, 0, 1])
        self.assertEqual(self.child_result["blocked"], ["socket.connect", "subprocess.Popen", "compile", "open"])
        self.assertEqual(len(self.findings["scope_case_statuses"]), 42)
        self.assertEqual(self.manifest["summary_file_sha256"], sha256_bytes((self.run_dir / "summary.md").read_bytes()))
        self.assertEqual(self.manifest["tool_file_sha256"], self.hashes)
        self.assertEqual(report.collect_tool_hashes(TOOL_DIR), self.hashes)
        self.assertTrue(self.manifest["isolation"]["os_level_sandbox_confirmed"])
        result = self.score_variant()
        self.assertEqual({key: result["counts"][key] for key in ("tp", "fp", "fn", "tn", "n_a")}, dict(tp=2, fp=1, fn=1, tn=4, n_a=1))
        self.assertEqual(result["metrics"], dict(precision=0.666667, recall=0.666667))

    def test_rehashed_mapping_mutations_are_rejected(self):
        mutations = (
            (dict(family="completed_candle"), "REFERENCE_MISMATCH"),
            (dict(quote_sha256="0" * 64), "FILE_HASH_MISMATCH"),
            (dict(quote_codepoints=999), "COUNT_MISMATCH"),
            (dict(candidate_ranges=[dict(start=0, end_exclusive=999)]), "REFERENCE_MISMATCH"),
        )
        for fields, code in mutations:
            with self.subTest(fields=fields), self.assertRaises(ContractViolation) as caught:
                self.score_variant(lambda f, m, r: f["cases"][0]["mapping_candidates"][0].update(fields))
            self.assertEqual(caught.exception.issue.code, code)

    def test_unknown_count_keys_do_not_echo(self):
        marker = "SYNTHETIC_PRIVATE_MARKER"
        for side in ("expected", "actual"):
            def mutate(f, m, r):
                f["scope_counts"][side][marker] = None
                m["scope_counts"] = copy.deepcopy(f["scope_counts"])
            with self.subTest(side=side), self.assertRaises(ContractViolation) as caught:
                self.score_variant(mutate)
            self.assertNotIn(marker, report.render_json(report.invalid_report(caught.exception.issue)))
        for value in (None, "bad", -1, True):
            with self.subTest(value=value), self.assertRaises(ContractViolation) as caught:
                _require_count_map({marker: value}, "profile.expected_counts")
            self.assertNotIn(marker, report.render_json(report.invalid_report(caught.exception.issue)))

    def test_label_absence_and_revision_contract(self):
        def missing(f, m, r):
            for item in r["items"]:
                for key in ("label_review_status", "label_version", "final_label"):
                    item.pop(key)
        result = self.score_variant(missing)
        self.assertEqual(result["counts"]["n_a"], 9)
        self.assertIsNone(result["metrics"]["precision"])
        def changed(f, m, r):
            r["items"][0].update(final_label="confirmed_valid", label_change_rationale="Synthetic correction.")
        with self.assertRaises(ContractViolation) as caught:
            self.score_variant(changed)
        self.assertEqual(caught.exception.issue.code, "REFERENCE_MISMATCH")
        def revised(f, m, r):
            changed(f, m, r)
            r["items"][0]["label_version"] = "claim-annotations-v2"
        result = self.score_variant(revised)
        self.assertEqual((result["counts"]["tp"], result["counts"]["fp"]), (1, 2))
        for mutate in (
            lambda f, m, r: r["items"][0].pop("final_label"),
            lambda f, m, r: r["items"][0].pop("label_review_status"),
            lambda f, m, r: r["source"].update(label_source_version="stale"),
        ):
            with self.assertRaises(ContractViolation):
                self.score_variant(mutate)

    def test_failure_matrix_before_and_after_scanner_start(self):
        for stage in ("before_output", "before_scanner", "after_scanner"):
            for failure, code in ((ValueError("synthetic"), 3), (OSError("synthetic"), 3), (ContractViolation("SYNTHETIC", "runtime", "Synthetic contract failure"), 2), (IsolationViolation("SYNTHETIC", "runtime", "Synthetic isolation failure"), 4)):
                with self.subTest(stage=stage, failure=type(failure).__name__), tempfile.TemporaryDirectory(prefix="offline-failure-") as task_temp:
                    output = Path(task_temp)
                    captured = io.StringIO()
                    assessment = OsIsolationAssessment("verification_unimplemented", None, "synthetic_bookkeeping_only")
                    def fail(*args, **kwargs):
                        if stage == "after_scanner":
                            kwargs["on_scanner_start"]()
                        raise failure
                    with ExitStack() as stack:
                        replacements = dict(assess_os_isolation=assessment, require_material_scan_isolation=assessment, load_validated_bundle=SimpleNamespace(validation=SimpleNamespace(scanner_source_sha256="a" * 64)), prepare_guard=SimpleNamespace(code=(lambda: None).__code__), collect_tool_hashes=self.hashes, prepare_new_output_directory=output, install_runtime_policy=None, instantiate_guard=None)
                        for name, value in replacements.items():
                            stack.enter_context(patch.object(cli, name, return_value=value))
                        target = {"before_output": "load_validated_bundle", "before_scanner": "install_runtime_policy", "after_scanner": "scan_selected_cases"}[stage]
                        stack.enter_context(patch.object(cli, target, side_effect=fail))
                        stack.enter_context(redirect_stdout(captured))
                        actual = cli.main(["scan", "--bundle", "synthetic", "--profile", "synthetic", "--out", str(output)])
                    response = json.loads(captured.getvalue())
                    self.assertEqual(actual, code)
                    self.assertEqual(response["scanner_ran"], stage == "after_scanner")
                    expected_marker = "not_attempted" if stage == "before_output" else "written"
                    self.assertEqual(response["incomplete_marker_status"], expected_marker)
                    self.assertFalse((output / "run-manifest.json").exists())
                    if stage != "before_output":
                        marker = json.loads((output / "run-incomplete.json").read_text("utf-8"))
                        self.assertEqual(marker["scanner_ran"], stage == "after_scanner")
                        self.assertEqual(marker["phase"], "scanner_execution" if stage == "after_scanner" else "runtime_isolation_installation")

    def test_real_cli_gate_precedes_input_and_output(self):
        output = self.root / "must-not-exist"
        captured = io.StringIO()
        with patch.object(cli, "load_validated_bundle") as load, redirect_stdout(captured):
            code = cli.main(["scan", "--bundle", "missing", "--profile", "missing", "--out", str(output)])
        load.assert_not_called()
        response = json.loads(captured.getvalue())
        self.assertEqual(code, 4)
        self.assertEqual(response["errors"][0]["code"], "OS_SANDBOX_VERIFICATION_UNIMPLEMENTED")
        self.assertFalse(response["scanner_ran"])
        self.assertFalse(output.exists())


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--synthetic-writer":
        synthetic_writer(Path(sys.argv[2]))
    else:
        unittest.main()
