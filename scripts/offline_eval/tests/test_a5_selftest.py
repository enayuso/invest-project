"""Tests for a5_selftest.py (design doc §17 J1-J8).

No test in this file ever invokes a real `docker` binary: `run_docker` is
always replaced by a fake dispatcher, exactly like tests/test_launcher.py
does for launcher.run()."""

from pathlib import Path
import json
import subprocess
import sys
import tempfile
import unittest


TOOL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOL_DIR))

import a5_selftest as a5  # noqa: E402
import bundle  # noqa: E402
import container_contract as cc  # noqa: E402
import isolation  # noqa: E402
import session_adapter  # noqa: E402
from contracts import sha256_bytes  # noqa: E402


REPO_ROOT = TOOL_DIR.parents[1]

TOOL_FILE_NAMES = (
    "bundle.py", "cli.py", "container_contract.py", "contracts.py", "isolation.py",
    "launcher.py", "report.py", "scoring.py", "session_adapter.py",
)


def _make_fake_repo(root: Path) -> Path:
    repo_root = root / "repo"
    tool_dir = repo_root / "scripts" / "offline_eval"
    tool_dir.mkdir(parents=True)
    for name in TOOL_FILE_NAMES:
        (tool_dir / name).write_text(f"# fake {name}\n", encoding="utf-8")
    guard_dir = repo_root / "app" / "agents"
    guard_dir.mkdir(parents=True)
    (guard_dir / "session_close_guard.py").write_text("# fake guard\n", encoding="utf-8")
    return repo_root


def _confirmed_isolation_block() -> dict:
    return {
        "os_level_sandbox_confirmed": True,
        "security_boundary": cc.CONFIRMED_SECURITY_BOUNDARY,
        "os_isolation": {
            "status": "confirmed",
            "probes": [{"id": f"P{i}", "passed": True, "reason": "synthetic"} for i in range(1, 12)],
        },
    }


def _valid_full_inspect(nonce_hex: str, run_name: str) -> dict:
    env = sorted(cc.BASE_IMAGE_ENV_ALLOWLIST) + [f"{cc.NONCE_ENV_NAME}={nonce_hex}"]
    mounts = [
        {"Type": "bind", "Destination": destination, "RW": rw, "Source": f"/fake/{index}"}
        for index, (destination, rw) in enumerate(cc.REQUIRED_MOUNT_DESTINATIONS.items())
    ]
    return {
        "Image": cc.IMAGE_ID,
        "HostConfig": {
            "NetworkMode": "none", "ReadonlyRootfs": True, "Privileged": False,
            "CapDrop": ["ALL"], "CapAdd": None, "SecurityOpt": ["no-new-privileges:true"],
            "Devices": None, "PidsLimit": cc.EXPECTED_PIDS_LIMIT,
            "PidMode": "", "IpcMode": "", "UTSMode": "", "UsernsMode": "",
            "Tmpfs": None, "Binds": None, "VolumesFrom": None,
            "Memory": cc.EXPECTED_MEMORY_BYTES, "NanoCpus": cc.EXPECTED_NANO_CPUS,
            "LogConfig": {"Type": "none"}, "GroupAdd": None,
        },
        "Config": {
            "User": cc.CONTAINER_USER, "Env": env, "Hostname": cc.CONTAINER_HOSTNAME,
            "Cmd": cc.build_scan_command(run_name=run_name), "Entrypoint": None,
        },
        "Mounts": mounts,
    }


# ---------------------------------------------------------------------------
# J3: bundle generator
# ---------------------------------------------------------------------------


class SyntheticBundleGeneratorTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temp = tempfile.TemporaryDirectory(prefix="offline-a5-bundle-")
        self.addCleanup(self._temp.cleanup)
        self.root = Path(self._temp.name)
        guard_path = REPO_ROOT / "app" / "agents" / "session_close_guard.py"
        self.guard_sha256 = sha256_bytes(guard_path.read_bytes())

    def test_generated_bundle_passes_bundle_validation(self) -> None:
        bundle_dir, profile_path = a5.write_synthetic_environment(self.root, self.guard_sha256)
        loaded = bundle.load_validated_bundle(bundle_dir, profile_path, REPO_ROOT)
        self.assertEqual(loaded.validation.actual_counts["unique_evidence"], 9)
        self.assertEqual(loaded.validation.actual_counts["candidates"], 9)
        self.assertEqual(loaded.validation.actual_counts["candidate_error_drafts"], 3)
        self.assertEqual(loaded.validation.actual_counts["candidate_valid_drafts"], 6)

    def test_generated_profile_selects_nine_candidates_in_order(self) -> None:
        _bundle_dir, profile_path = a5.write_synthetic_environment(self.root, self.guard_sha256)
        profile = json.loads(profile_path.read_text("utf-8"))
        self.assertEqual(profile["candidate_case_ids"], list(a5.SYNTHETIC_CASE_IDS))
        self.assertEqual(len(profile["candidate_case_ids"]), 9)
        self.assertEqual(profile["out_of_scope_clear_case_ids"], [])

    def test_generated_bundle_reproduces_expected_hit_vector(self) -> None:
        bundle_dir, profile_path = a5.write_synthetic_environment(self.root, self.guard_sha256)
        loaded = bundle.load_validated_bundle(bundle_dir, profile_path, REPO_ROOT)
        prepared = session_adapter.prepare_guard(REPO_ROOT, loaded.validation.scanner_source_sha256)
        scanner = session_adapter.instantiate_guard(prepared)
        cases = session_adapter.scan_selected_cases(loaded, scanner)
        hits = [len(case["raw_findings"]) for case in cases]
        self.assertEqual(hits, list(a5.EXPECTED_HIT_VECTOR))


# ---------------------------------------------------------------------------
# J6: negative-mode argv
# ---------------------------------------------------------------------------


class NegativeArgsTests(unittest.TestCase):
    def _mounts(self) -> cc.HostMounts:
        return cc.HostMounts(
            tool_dir=TOOL_DIR,
            guard_file=REPO_ROOT / "app" / "agents" / "session_close_guard.py",
            phase2_dir=TOOL_DIR,
            phase3a_dir=TOOL_DIR,
            staging_dir=TOOL_DIR,
        )

    def test_negative_args_differ_only_in_the_network_value(self) -> None:
        mounts = self._mounts()
        nonce = "a" * 64
        positive = cc.build_docker_args(mounts=mounts, nonce_hex=nonce, run_name="probe-run")
        negative = a5.build_negative_docker_args(mounts=mounts, nonce_hex=nonce, run_name="probe-run")
        self.assertEqual(len(positive), len(negative))
        diffs = [index for index, (a, b) in enumerate(zip(positive, negative)) if a != b]
        expected_index = positive.index("--network") + 1
        self.assertEqual(diffs, [expected_index])
        self.assertEqual(positive[expected_index], "none")
        self.assertEqual(negative[expected_index], "bridge")

    def test_build_docker_args_itself_is_untouched(self) -> None:
        # a5's negative builder must not mutate the shared pure function.
        mounts = self._mounts()
        nonce = "a" * 64
        before = cc.build_docker_args(mounts=mounts, nonce_hex=nonce, run_name="probe-run")
        a5.build_negative_docker_args(mounts=mounts, nonce_hex=nonce, run_name="probe-run")
        after = cc.build_docker_args(mounts=mounts, nonce_hex=nonce, run_name="probe-run")
        self.assertEqual(before, after)

    def test_negative_network_value_is_rejected_by_validate_inspect(self) -> None:
        nonce = "b" * 64
        run_name = "probe-run"
        payload = _valid_full_inspect(nonce, run_name)
        payload["HostConfig"]["NetworkMode"] = "bridge"
        violations = cc.validate_inspect(payload, nonce_hex=nonce, run_name=run_name, expected_sources=None)
        self.assertIn("NETWORK_MODE_NOT_NONE", violations)


# ---------------------------------------------------------------------------
# J7: evidence-shape collectors
# ---------------------------------------------------------------------------


class EvidenceShapeTests(unittest.TestCase):
    def test_mountinfo_shape_has_mount_point_fs_type_and_options_but_no_source(self) -> None:
        text = (
            "36 35 98:0 / / ro,relatime shared:1 - ext4 /dev/root rw\n"
            "37 36 0:20 / /proc rw,nosuid,nodev,noexec,relatime - proc proc rw\n"
        )
        shape = a5._mountinfo_shape(text)
        self.assertIsNotNone(shape)
        for entry in shape:
            self.assertEqual(set(entry), {"mount_point", "fs_type", "options"})
        mount_points = {entry["mount_point"] for entry in shape}
        self.assertEqual(mount_points, {"/", "/proc"})
        fs_types = {entry["mount_point"]: entry["fs_type"] for entry in shape}
        self.assertEqual(fs_types, {"/": "ext4", "/proc": "proc"})
        # J7: fs_type is a filesystem kind, never a host path -- confirm the
        # real mount sources ("/dev/root", "proc") from the fixture text
        # never leak into the shape as a "source" key.
        for entry in shape:
            self.assertNotIn("source", entry)

    def test_mountinfo_shape_returns_none_on_unparseable_text(self) -> None:
        self.assertIsNone(a5._mountinfo_shape("not a mountinfo line at all"))

    def test_status_shape_keeps_only_selected_fields(self) -> None:
        text = (
            "Name: python\n"
            "Uid: 65534 65534 65534 65534\n"
            "CapEff: 0000000000000000\n"
            "Seccomp: 2\n"
            "NoNewPrivs: 1\n"
            "Threads: 1\n"
        )
        shape = a5._status_shape(text)
        self.assertEqual(
            shape,
            {
                "Uid": "65534 65534 65534 65534",
                "CapEff": "0000000000000000",
                "Seccomp": "2",
                "NoNewPrivs": "1",
            },
        )
        self.assertNotIn("Name", shape)
        self.assertNotIn("Threads", shape)

    def test_collect_probe_evidence_emits_keys_and_targets_only(self) -> None:
        mounts = cc.HostMounts(
            tool_dir=TOOL_DIR,
            guard_file=REPO_ROOT / "app" / "agents" / "session_close_guard.py",
            phase2_dir=TOOL_DIR,
            phase3a_dir=TOOL_DIR,
            staging_dir=TOOL_DIR,
        )
        nonce = "c" * 64
        mountinfo = "36 35 98:0 / / ro,relatime - ext4 /dev/root rw\n"
        status = "Uid: 65534 65534 65534 65534\nCapEff: 0\nNoNewPrivs: 1\nSeccomp: 2\n"

        def fake_run_docker(docker_binary, args, *, env, input_text=None):
            if args[0] == "create":
                return subprocess.CompletedProcess(args, 0, stdout="fake-probe\n", stderr="")
            if args[0] == "inspect":
                payload = _valid_full_inspect(nonce, "unused")
                payload["Mounts"][0]["Source"] = "D:\\Investment_Project\\should-not-appear"
                return subprocess.CompletedProcess(args, 0, stdout=json.dumps([payload]), stderr="")
            if args[:2] == ["start", "-a"]:
                out = {
                    "mountinfo": mountinfo,
                    "status": status,
                    "env_keys": ["PATH", cc.NONCE_ENV_NAME, "SECRET_VALUE_SHOULD_NOT_BE_HERE"],
                    "net_class": ["lo"],
                }
                return subprocess.CompletedProcess(args, 0, stdout=json.dumps(out), stderr="")
            if args[0] == "rm":
                return subprocess.CompletedProcess(args, 0, stdout="", stderr="")
            raise AssertionError(f"unexpected call: {args}")

        evidence = a5._collect_probe_evidence(
            "docker", {}, mounts, nonce, fake_run_docker, network_mode="none"
        )
        self.assertIsNone(evidence["collection_error"])
        self.assertEqual(
            evidence["mountinfo_entries"],
            [{"mount_point": "/", "fs_type": "ext4", "options": sorted(["ro", "relatime"])}],
        )
        self.assertEqual(evidence["status_fields"], {"Uid": "65534 65534 65534 65534", "CapEff": "0", "NoNewPrivs": "1", "Seccomp": "2"})
        # env_keys are keys only -- the fake container is only ever asked for
        # keys, so no value ever reaches this dict in the first place.
        self.assertEqual(set(evidence["env_keys"]), {"PATH", cc.NONCE_ENV_NAME, "SECRET_VALUE_SHOULD_NOT_BE_HERE"})
        self.assertEqual(evidence["net_class_names"], ["lo"])
        # J7: fs_type ("ext4") is present -- it is a filesystem kind, not host
        # information -- but no mount Source and no host path (the fake
        # inspect above deliberately injects a "D:\...should-not-appear"
        # Source to prove it gets dropped) may appear anywhere in the
        # collected evidence, which is exactly what ends up in
        # evidence-shapes.json.
        evidence_text = json.dumps(evidence)
        self.assertNotIn("Source", evidence_text)
        self.assertNotIn("should-not-appear", evidence_text)
        self.assertFalse(a5._contains_host_path(evidence))


# ---------------------------------------------------------------------------
# _contains_host_path
# ---------------------------------------------------------------------------


class HostPathDetectionTests(unittest.TestCase):
    def test_detects_windows_drive_path(self) -> None:
        self.assertTrue(a5._contains_host_path({"Mounts": [{"Marker": "D:\\Investment_Project\\x"}]}))

    def test_clean_inspect_subset_has_no_host_path(self) -> None:
        mounts = [{"Type": "bind", "Destination": d, "RW": rw} for d, rw in cc.REQUIRED_MOUNT_DESTINATIONS.items()]
        subset = {"Image": cc.IMAGE_ID, "HostConfig": {}, "Config": {}, "Mounts": mounts}
        self.assertFalse(a5._contains_host_path(subset))


# ---------------------------------------------------------------------------
# J5: positive (A5) expectation checker
# ---------------------------------------------------------------------------


class PositiveEvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.nonce = "d" * 64
        self.run_name = "eval-run"
        self.manifest = {"isolation": _confirmed_isolation_block()}
        full = _valid_full_inspect(self.nonce, self.run_name)
        self.inspect_subset = cc.inspect_subset(full)
        self.hit_vector = list(a5.EXPECTED_HIT_VECTOR)

    def _checks(self, **overrides) -> dict:
        kwargs = dict(
            exit_code=0,
            manifest=self.manifest,
            hit_vector=self.hit_vector,
            inspect_subset=self.inspect_subset,
            nonce_hex=self.nonce,
            run_name=self.run_name,
        )
        kwargs.update(overrides)
        return a5.evaluate_positive_result(**kwargs)

    def test_all_pass_when_everything_is_correct(self) -> None:
        self.assertTrue(self._checks()["overall"])

    def test_rejects_nonzero_exit(self) -> None:
        checks = self._checks(exit_code=4)
        self.assertFalse(checks["exit_zero"])
        self.assertFalse(checks["overall"])

    def test_rejects_status_not_confirmed(self) -> None:
        block = _confirmed_isolation_block()
        block["os_isolation"] = {"status": "verification_failed", "probes": []}
        checks = self._checks(manifest={"isolation": block})
        self.assertFalse(checks["isolation_confirmed"])
        self.assertFalse(checks["overall"])

    def test_rejects_one_failed_probe(self) -> None:
        block = _confirmed_isolation_block()
        block["os_isolation"]["probes"][0]["passed"] = False
        checks = self._checks(manifest={"isolation": block})
        self.assertFalse(checks["isolation_confirmed"])
        self.assertFalse(checks["overall"])

    def test_rejects_wrong_hit_vector(self) -> None:
        checks = self._checks(hit_vector=[0] * 9)
        self.assertFalse(checks["hit_vector_matches"])
        self.assertFalse(checks["overall"])

    def test_rejects_attestation_with_an_inspect_violation(self) -> None:
        bad = json.loads(json.dumps(self.inspect_subset))
        bad["HostConfig"]["NetworkMode"] = "bridge"
        checks = self._checks(inspect_subset=bad)
        self.assertFalse(checks["attestation_inspect_valid"])
        self.assertFalse(checks["overall"])

    def test_rejects_host_path_in_inspect_subset(self) -> None:
        bad = json.loads(json.dumps(self.inspect_subset))
        bad["Mounts"][0]["HostMarker"] = "D:\\Investment_Project\\evil"
        checks = self._checks(inspect_subset=bad)
        self.assertFalse(checks["no_host_paths_in_inspect_subset"])
        self.assertFalse(checks["overall"])


# ---------------------------------------------------------------------------
# J6: negative (A6) expectation checker
# ---------------------------------------------------------------------------


class NegativeEvaluationTests(unittest.TestCase):
    def _good_report(self) -> dict:
        return {"status": "withheld", "isolation_assessment": {"status": "verification_failed", "reason": "probe_failed:P1"}}

    def test_all_pass_for_the_expected_probe_failure(self) -> None:
        checks = a5.evaluate_negative_result(exit_code=4, container_report=self._good_report(), staged_entries=[])
        self.assertTrue(checks["overall"])

    def test_rejects_exit_code_other_than_four(self) -> None:
        checks = a5.evaluate_negative_result(exit_code=0, container_report=self._good_report(), staged_entries=[])
        self.assertFalse(checks["overall"])

    def test_rejects_missing_withheld_status(self) -> None:
        checks = a5.evaluate_negative_result(exit_code=4, container_report={"status": "complete"}, staged_entries=[])
        self.assertFalse(checks["overall"])

    def test_rejects_reason_not_a_probe_failure(self) -> None:
        report = {"status": "withheld", "isolation_assessment": {"reason": "launch_nonce_missing_or_malformed"}}
        checks = a5.evaluate_negative_result(exit_code=4, container_report=report, staged_entries=[])
        self.assertFalse(checks["overall"])

    def test_rejects_when_output_was_created(self) -> None:
        checks = a5.evaluate_negative_result(exit_code=4, container_report=self._good_report(), staged_entries=["findings.json"])
        self.assertFalse(checks["overall"])


# ---------------------------------------------------------------------------
# End-to-end orchestration with a fake docker dispatcher (no real docker call)
# ---------------------------------------------------------------------------


class FakeDockerA5:
    """Stands in for `run_docker`; dispatches on the docker subcommand and on
    call order (1st create = the run under test, 2nd = the J7 probe)."""

    def __init__(self, *, repo_root: Path, run_name: str, mode: str) -> None:
        self.repo_root = repo_root
        self.run_name = run_name
        self.mode = mode  # "positive_success" | "negative_success"
        self.calls: list[list[str]] = []
        self._create_count = 0
        self._current_kind: str | None = None
        self.nonce: str | None = None

    def __call__(self, docker_binary, args, *, env, input_text=None):
        self.calls.append(list(args))
        if args and args[0] == "create":
            self._create_count += 1
            self._current_kind = "main" if self._create_count == 1 else "probe"
            if self._current_kind == "main":
                index = args.index("--env")
                self.nonce = args[index + 1].split("=", 1)[1]
            return subprocess.CompletedProcess(args, 0, stdout=f"fake-{self._current_kind}\n", stderr="")
        if args and args[0] == "inspect":
            payload = _valid_full_inspect(self.nonce or "0" * 64, self.run_name)
            if self._current_kind == "main" and self.mode == "negative_success":
                payload["HostConfig"]["NetworkMode"] = "bridge"
            return subprocess.CompletedProcess(args, 0, stdout=json.dumps([payload]), stderr="")
        if args[:2] == ["start", "-a"]:
            if self._current_kind == "probe":
                out = {
                    "mountinfo": "36 35 98:0 / / ro,relatime - ext4 /dev/root rw\n",
                    "status": "Uid: 65534 65534 65534 65534\nCapEff: 0\nNoNewPrivs: 1\nSeccomp: 2\n",
                    "env_keys": ["PATH", cc.NONCE_ENV_NAME],
                    "net_class": ["lo"],
                }
                return subprocess.CompletedProcess(args, 0, stdout=json.dumps(out), stderr="")
            if self.mode == "positive_success":
                self._write_success_output()
                return subprocess.CompletedProcess(args, 0, stdout="{}", stderr="")
            if self.mode == "negative_success":
                report = {
                    "schema_version": "offline-eval.scan.v1", "command": "scan", "status": "withheld",
                    "counts": None, "engine_generation_ran": False, "scanner_ran": False,
                    "isolation_assessment": {
                        "status": "verification_failed", "method": "linux_container_self_probe",
                        "reason": "probe_failed:P1", "probes": [],
                    },
                    "incomplete_marker_status": "not_attempted", "historical_replay_certified": False,
                    "operational_accuracy_estimate": None,
                    "errors": [{"code": "OS_SANDBOX_VERIFICATION_FAILED", "reference": "runtime", "description": "withheld"}],
                }
                return subprocess.CompletedProcess(args, 4, stdout=json.dumps(report), stderr="")
            raise AssertionError(f"unhandled mode {self.mode}")
        if args and args[0] == "rm":
            return subprocess.CompletedProcess(args, 0, stdout="", stderr="")
        raise AssertionError(f"unexpected docker invocation: {args}")

    def _write_success_output(self) -> None:
        staging_run_dir = isolation.fixed_selftest_staging_root(self.repo_root) / self.run_name
        run_dir = staging_run_dir / self.run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "schema_version": "offline-eval.scan-run.v2",
            "status": "complete",
            "launch_nonce": self.nonce,
            "python_version": cc.PYTHON_VERSION_STRING,
            "isolation": _confirmed_isolation_block(),
        }
        (run_dir / "run-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        cases = [
            {"case_id": case_id, "raw_findings": ["x"] * hits}
            for case_id, hits in zip(a5.SYNTHETIC_CASE_IDS, a5.EXPECTED_HIT_VECTOR)
        ]
        (run_dir / "findings.json").write_text(json.dumps({"cases": cases}), encoding="utf-8")
        (run_dir / "summary.md").write_text("# summary\n", encoding="utf-8")


class RunSelfTestTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temp = tempfile.TemporaryDirectory(prefix="offline-a5-run-")
        self.addCleanup(self._temp.cleanup)
        self.root = Path(self._temp.name)
        self.repo_root = _make_fake_repo(self.root)

    def _run(self, run_name: str, mode: str):
        docker = FakeDockerA5(repo_root=self.repo_root, run_name=run_name, mode=mode)
        result = a5.run_self_test(
            run_name,
            negative=(mode == "negative_success"),
            repo_root=self.repo_root,
            run_docker=docker,
            resolve_docker_binary=lambda: "docker",
        )
        return result, docker

    def test_positive_success_writes_evidence_and_moves_run(self) -> None:
        result, docker = self._run("a5-run", "positive_success")
        self.assertTrue(result["checks"]["overall"])
        self.assertEqual(result["exit_code"], 0)
        self.assertIsNotNone(result["evidence_shapes_path"])
        self.assertTrue(Path(result["evidence_shapes_path"]).is_file())
        self.assertIsNotNone(result["attestation_path"])
        self.assertTrue(Path(result["attestation_path"]).is_file())
        run_dir = isolation.fixed_selftest_output_root(self.repo_root) / "a5-run"
        self.assertTrue(run_dir.is_dir())
        self.assertEqual(result["run_dir"], str(run_dir))
        # Two container lifecycles (main + probe), each create/inspect/start/rm.
        self.assertEqual(sum(1 for call in docker.calls if call[0] == "create"), 2)

    def test_negative_success_creates_no_output(self) -> None:
        result, _docker = self._run("a6-run", "negative_success")
        self.assertTrue(result["checks"]["overall"])
        self.assertEqual(result["exit_code"], 4)
        self.assertIsNone(result["attestation_path"])
        run_dir = isolation.fixed_selftest_output_root(self.repo_root) / "a6-run"
        self.assertFalse(run_dir.exists())
        # J7 evidence is still collected regardless of outcome.
        self.assertIsNotNone(result["evidence_shapes_path"])
        self.assertTrue(Path(result["evidence_shapes_path"]).is_file())

    def test_duplicate_run_name_is_rejected_before_any_docker_call(self) -> None:
        launch_root = isolation.fixed_selftest_launch_root(self.repo_root)
        launch_root.mkdir(parents=True)
        (launch_root / "dup-run").mkdir()
        docker = FakeDockerA5(repo_root=self.repo_root, run_name="dup-run", mode="positive_success")
        result = a5.run_self_test(
            "dup-run", negative=False, repo_root=self.repo_root, run_docker=docker, resolve_docker_binary=lambda: "docker"
        )
        self.assertFalse(result["checks"]["overall"])
        self.assertEqual(result["errors"][0]["code"], "RUN_ALREADY_EXISTS")
        self.assertEqual(docker.calls, [])


if __name__ == "__main__":
    unittest.main()
