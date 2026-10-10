"""Launcher tests (design §14.2): `launcher._run_docker` is replaced by a fake
dispatcher so these run without Docker, without touching the real evidence
folders under `handoff/evaluations/`, and without ever invoking a real
`docker` binary. A fully synthetic fake repo root is built under a temp
directory for every test."""

from contextlib import ExitStack, redirect_stderr, redirect_stdout
from pathlib import Path
import io
import json
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


TOOL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOL_DIR))

import container_contract as cc  # noqa: E402
import launcher  # noqa: E402
from contracts import sha256_bytes  # noqa: E402
from isolation import fixed_launch_root, fixed_output_root, fixed_staging_root  # noqa: E402


TOOL_FILE_NAMES = (
    "bundle.py", "cli.py", "container_contract.py", "contracts.py", "isolation.py",
    "launcher.py", "report.py", "scoring.py", "session_adapter.py",
)


def _make_fake_repo(root: Path) -> tuple[Path, Path]:
    repo_root = root / "repo"
    tool_dir = repo_root / "scripts" / "offline_eval"
    tool_dir.mkdir(parents=True)
    for name in TOOL_FILE_NAMES:
        (tool_dir / name).write_text(f"# fake {name}\n", encoding="utf-8")
    guard_dir = repo_root / "app" / "agents"
    guard_dir.mkdir(parents=True)
    (guard_dir / "session_close_guard.py").write_text("# fake guard\n", encoding="utf-8")
    (root / "handoff" / "evaluations" / cc.PHASE2_BUNDLE_DIRNAME).mkdir(parents=True)
    (root / "handoff" / "evaluations" / cc.PHASE3A_PROFILE_DIRNAME).mkdir(parents=True)
    return tool_dir, repo_root


def _extract_nonce(args: list[str]) -> str:
    index = args.index("--env")
    return args[index + 1].split("=", 1)[1]


def _inspect_payload(mounts: cc.HostMounts, nonce_hex: str, run_name: str) -> dict:
    env = sorted(cc.BASE_IMAGE_ENV_ALLOWLIST) + [f"{cc.NONCE_ENV_NAME}={nonce_hex}"]
    mount_entries = [
        {"Type": "bind", "Destination": destination, "RW": rw, "Source": str(source)}
        for source, destination, rw in mounts.as_tuple()
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
            "LogConfig": {"Type": "none"},
            # H7: present (as None, the valid value) so validate_inspect's G6
            # GroupAdd check actually runs against these fixtures instead of
            # silently relying on the key being absent.
            "GroupAdd": None,
        },
        "Config": {
            "User": cc.CONTAINER_USER, "Env": env, "Hostname": cc.CONTAINER_HOSTNAME,
            "Cmd": cc.build_scan_command(run_name=run_name), "Entrypoint": None,
        },
        "Mounts": mount_entries,
    }


class FakeDocker:
    """Stands in for `launcher._run_docker`; dispatches on the docker subcommand."""

    def __init__(
        self,
        *,
        repo_root: Path,
        run_name: str,
        image_id: str = cc.IMAGE_ID,
        image_inspect_ok: bool = True,
        create_ok: bool = True,
        inspect_ok: bool = True,
        inspect_violate: bool = False,
        mangle_source: bool = False,
        start_returncode: int = 0,
        start_stdout: str = "",
        write_staged_output: bool = True,
        manifest_mutator=None,
        raw_manifest_override: bytes | None = None,
        extra_file_in_run_dir: str | None = None,
        extra_staged_sibling: str | None = None,
        raise_on_inspect: bool = False,
        rm_returncode: int = 0,
        pre_create_attestation: bool = False,
    ) -> None:
        self.repo_root = repo_root
        self.run_name = run_name
        self.image_id = image_id
        self.image_inspect_ok = image_inspect_ok
        self.create_ok = create_ok
        self.inspect_ok = inspect_ok
        self.inspect_violate = inspect_violate
        self.mangle_source = mangle_source
        self.start_returncode = start_returncode
        self.start_stdout = start_stdout
        self.write_staged_output = write_staged_output
        self.manifest_mutator = manifest_mutator
        self.raw_manifest_override = raw_manifest_override
        self.extra_file_in_run_dir = extra_file_in_run_dir
        self.extra_staged_sibling = extra_staged_sibling
        self.raise_on_inspect = raise_on_inspect
        self.rm_returncode = rm_returncode
        self.pre_create_attestation = pre_create_attestation
        self.calls: list[list[str]] = []
        self.rm_calls: list[list[str]] = []
        self.envs: list[dict[str, str]] = []
        self.nonce: str | None = None
        self._container_counter = 0

    def _mounts(self) -> cc.HostMounts:
        guard_path = self.repo_root / Path(*cc.GUARD_SOURCE_RELATIVE.split("/"))
        return cc.HostMounts(
            tool_dir=self.repo_root / "scripts" / "offline_eval",
            guard_file=guard_path,
            phase2_dir=self.repo_root.parent / "handoff" / "evaluations" / cc.PHASE2_BUNDLE_DIRNAME,
            phase3a_dir=self.repo_root.parent / "handoff" / "evaluations" / cc.PHASE3A_PROFILE_DIRNAME,
            staging_dir=fixed_staging_root(self.repo_root) / self.run_name,
        )

    def __call__(
        self, docker_binary: str, args: list[str], *, env: dict[str, str], input_text: str | None = None
    ) -> subprocess.CompletedProcess:
        self.calls.append(list(args))
        self.envs.append(dict(env))
        if args and args[0] == "inspect" and self.raise_on_inspect:
            raise RuntimeError("synthetic unexpected failure during inspect")
        if args[:2] == ["image", "inspect"]:
            if not self.image_inspect_ok:
                return subprocess.CompletedProcess(args, 1, stdout="", stderr="no such image")
            return subprocess.CompletedProcess(args, 0, stdout=f"{self.image_id}\n", stderr="")
        if args and args[0] == "create":
            self.nonce = _extract_nonce(args)
            if not self.create_ok:
                return subprocess.CompletedProcess(args, 1, stdout="", stderr="create failed")
            self._container_counter += 1
            return subprocess.CompletedProcess(args, 0, stdout=f"fake-container-{self._container_counter}\n", stderr="")
        if args and args[0] == "inspect":
            if not self.inspect_ok:
                return subprocess.CompletedProcess(args, 1, stdout="", stderr="inspect failed")
            payload = _inspect_payload(self._mounts(), self.nonce, self.run_name)
            if self.inspect_violate:
                payload["HostConfig"]["NetworkMode"] = "bridge"
            if self.mangle_source:
                for mount in payload["Mounts"]:
                    mount["Source"] = "/completely/different/path"
            return subprocess.CompletedProcess(args, 0, stdout=json.dumps([payload]), stderr="")
        if args[:2] == ["start", "-a"]:
            if self.write_staged_output:
                self._write_output()
            if self.pre_create_attestation:
                # H1: simulate something already occupying the attestation path
                # (e.g. a leftover from a prior partial run) so the launcher's
                # exclusive-create write fails with ATTESTATION_WRITE_FAILED.
                launch_dir = fixed_launch_root(self.repo_root) / self.run_name
                launch_dir.mkdir(parents=True, exist_ok=True)
                (launch_dir / cc.ATTESTATION_FILENAME).write_text("pre-existing", encoding="utf-8")
            return subprocess.CompletedProcess(args, self.start_returncode, stdout=self.start_stdout, stderr="")
        if args and args[0] == "rm":
            self.rm_calls.append(list(args))
            return subprocess.CompletedProcess(args, self.rm_returncode, stdout="", stderr="rm failed" if self.rm_returncode else "")
        raise AssertionError(f"unexpected docker invocation: {args}")

    def _write_output(self) -> None:
        staging_run_dir = fixed_staging_root(self.repo_root) / self.run_name
        run_dir = staging_run_dir / self.run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        if self.raw_manifest_override is not None:
            (run_dir / "run-manifest.json").write_bytes(self.raw_manifest_override)
        else:
            manifest = {
                "schema_version": "offline-eval.scan-run.v2",
                "status": "complete",
                "launch_nonce": self.nonce,
                "python_version": cc.PYTHON_VERSION_STRING,
                "isolation": {
                    "os_level_sandbox_confirmed": True,
                    "security_boundary": cc.CONFIRMED_SECURITY_BOUNDARY,
                    "os_isolation": {
                        "status": "confirmed",
                        "probes": [{"id": f"P{i}", "passed": True, "reason": "synthetic"} for i in range(1, 12)],
                    },
                },
            }
            if self.manifest_mutator:
                self.manifest_mutator(manifest)
            (run_dir / "run-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        (run_dir / "findings.json").write_text("{}", encoding="utf-8")
        (run_dir / "summary.md").write_text("# summary\n", encoding="utf-8")
        if self.extra_file_in_run_dir:
            (run_dir / self.extra_file_in_run_dir).write_text("extra", encoding="utf-8")
        if self.extra_staged_sibling:
            (staging_run_dir / self.extra_staged_sibling).write_text("sibling", encoding="utf-8")


class LauncherTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._temp = tempfile.TemporaryDirectory(prefix="offline-launcher-")
        self.root = Path(self._temp.name)
        self.addCleanup(self._temp.cleanup)
        self.tool_dir, self.repo_root = _make_fake_repo(self.root)

    def _run(self, run_name: str, **docker_kwargs) -> tuple[int, dict, FakeDocker]:
        docker = FakeDocker(repo_root=self.repo_root, run_name=run_name, **docker_kwargs)
        captured = io.StringIO()
        with ExitStack() as stack:
            stack.enter_context(patch.object(launcher, "TOOL_DIR", self.tool_dir))
            stack.enter_context(patch.object(launcher, "_run_docker", docker))
            stack.enter_context(patch.object(launcher, "_resolve_docker_binary", return_value="docker"))
            stack.enter_context(redirect_stdout(captured))
            code = launcher.run(["--run-name", run_name])
        return code, json.loads(captured.getvalue()), docker

    # --- L1 ---

    def test_l1_duplicate_run_name_is_rejected_before_any_docker_call(self) -> None:
        run_name = "dup-run"
        (fixed_output_root(self.repo_root)).mkdir(parents=True)
        (fixed_output_root(self.repo_root) / run_name).mkdir()
        code, response, docker = self._run(run_name)
        self.assertEqual(code, 2)
        self.assertEqual(response["errors"][0]["code"], "RUN_ALREADY_EXISTS")
        self.assertEqual(docker.calls, [])

    # --- F14 ---

    def test_bytecode_present_is_rejected_before_any_docker_call(self) -> None:
        (self.tool_dir / "__pycache__").mkdir()
        code, response, docker = self._run("bytecode-run")
        self.assertEqual(code, 2)
        self.assertEqual(response["errors"][0]["code"], "BYTECODE_PRESENT")
        self.assertEqual(docker.calls, [])

    # --- L3 ---

    def test_l3_image_id_mismatch_is_exit_4_before_create(self) -> None:
        code, response, docker = self._run("l3-run", image_id="sha256:" + "0" * 64)
        self.assertEqual(code, 4)
        self.assertEqual(response["errors"][0]["code"], "IMAGE_DIGEST_MISMATCH")
        self.assertFalse(any(call and call[0] == "create" for call in docker.calls))

    # --- L5 ---

    def test_l5_validation_failure_removes_container_and_never_starts(self) -> None:
        code, response, docker = self._run("l5-reject-run", inspect_violate=True)
        self.assertEqual(code, 4)
        self.assertEqual(response["errors"][0]["code"], "CONTAINER_CONFIG_REJECTED")
        self.assertTrue(any(call[:2] == ["rm", "-f"] for call in docker.rm_calls))
        self.assertFalse(any(call[:2] == ["start", "-a"] for call in docker.calls))

    def test_l5_inspect_failure_removes_container(self) -> None:
        code, response, docker = self._run("l5-inspect-fail-run", inspect_ok=False)
        self.assertEqual(code, 4)
        self.assertEqual(response["errors"][0]["code"], "CONTAINER_INSPECT_FAILED")
        self.assertTrue(any(call[:2] == ["rm", "-f"] for call in docker.rm_calls))
        self.assertFalse(any(call[:2] == ["start", "-a"] for call in docker.calls))

    def test_l5_create_failure_never_removes_a_container(self) -> None:
        code, response, docker = self._run("l5-create-fail-run", create_ok=False)
        self.assertEqual(code, 4)
        self.assertEqual(response["errors"][0]["code"], "CONTAINER_CREATE_FAILED")
        self.assertEqual(docker.rm_calls, [])

    # --- L8 tampering: none of these may move the run or write an attestation ---

    def _assert_l8_rejected(self, run_name: str, expected_code: str, **docker_kwargs) -> None:
        code, response, _docker = self._run(run_name, **docker_kwargs)
        self.assertEqual(code, 4)
        self.assertEqual(response["errors"][0]["code"], expected_code)
        self.assertFalse((fixed_output_root(self.repo_root) / run_name).exists())
        self.assertFalse((fixed_launch_root(self.repo_root) / run_name).exists())

    def test_l8_extra_file_in_run_folder_is_rejected(self) -> None:
        self._assert_l8_rejected(
            "l8-extra-file", "STAGED_OUTPUT_INCOMPLETE", extra_file_in_run_dir="unexpected.txt"
        )

    def test_l8_sibling_entry_next_to_run_folder_is_rejected(self) -> None:
        self._assert_l8_rejected(
            "l8-sibling", "STAGED_OUTPUT_UNEXPECTED_ENTRY", extra_staged_sibling="unexpected.txt"
        )

    def test_l8_nonce_mismatch_is_rejected(self) -> None:
        self._assert_l8_rejected(
            "l8-nonce", "STAGED_NONCE_MISMATCH", manifest_mutator=lambda m: m.__setitem__("launch_nonce", "0" * 64)
        )

    def test_l8_unconfirmed_isolation_is_rejected(self) -> None:
        def mutate(manifest: dict) -> None:
            manifest["isolation"]["os_isolation"]["status"] = "verification_failed"

        self._assert_l8_rejected("l8-unconfirmed", "STAGED_ISOLATION_NOT_CONFIRMED", manifest_mutator=mutate)

    def test_l8_incomplete_status_is_rejected(self) -> None:
        self._assert_l8_rejected(
            "l8-incomplete", "STAGED_MANIFEST_INCOMPLETE", manifest_mutator=lambda m: m.__setitem__("status", "incomplete")
        )

    def test_l8_malformed_manifest_list_is_rejected(self) -> None:
        self._assert_l8_rejected(
            "l8-list-manifest", "STAGED_MANIFEST_UNPARSEABLE", raw_manifest_override=b"[1, 2, 3]"
        )

    def test_l8_os_isolation_as_string_is_rejected(self) -> None:
        def mutate(manifest: dict) -> None:
            manifest["isolation"]["os_isolation"] = "confirmed"

        self._assert_l8_rejected("l8-string-os-isolation", "STAGED_ISOLATION_NOT_CONFIRMED", manifest_mutator=mutate)

    def test_l8_unparseable_json_is_rejected(self) -> None:
        self._assert_l8_rejected(
            "l8-bad-json", "STAGED_MANIFEST_UNPARSEABLE", raw_manifest_override=b"not json"
        )

    # --- G2: launches folder reservation ---

    def test_g2_existing_launch_folder_is_rejected_before_any_docker_call(self) -> None:
        run_name = "launch-dup-run"
        launch_root = fixed_launch_root(self.repo_root)
        (launch_root / run_name).mkdir(parents=True)
        code, response, docker = self._run(run_name)
        self.assertEqual(code, 2)
        self.assertEqual(response["errors"][0]["code"], "RUN_ALREADY_EXISTS")
        self.assertEqual(docker.calls, [])

    def test_g2_existing_staging_folder_is_rejected_before_any_docker_call(self) -> None:
        run_name = "staging-dup-run"
        staging_root = fixed_staging_root(self.repo_root)
        (staging_root / run_name).mkdir(parents=True)
        code, response, docker = self._run(run_name)
        self.assertEqual(code, 2)
        self.assertEqual(response["errors"][0]["code"], "RUN_ALREADY_EXISTS")
        self.assertEqual(docker.calls, [])

    # --- L3 (extended) ---

    def test_l3_image_inspect_command_failure_is_exit_4_before_create(self) -> None:
        code, response, docker = self._run("l3-inspect-fail-run", image_inspect_ok=False)
        self.assertEqual(code, 4)
        self.assertEqual(response["errors"][0]["code"], "IMAGE_NOT_PRESENT")
        self.assertFalse(any(call and call[0] == "create" for call in docker.calls))

    # --- L5 (extended) ---

    def test_l5_mount_source_mismatch_is_rejected(self) -> None:
        code, response, docker = self._run("l5-source-mismatch-run", mangle_source=True)
        self.assertEqual(code, 4)
        self.assertEqual(response["errors"][0]["code"], "CONTAINER_CONFIG_REJECTED")
        self.assertTrue(any(call[:2] == ["rm", "-f"] for call in docker.rm_calls))
        self.assertFalse(any(call[:2] == ["start", "-a"] for call in docker.calls))

    def test_l5_unexpected_exception_still_removes_container(self) -> None:
        # F12: any exception from create onward, including one that is not a
        # LauncherFailure/ContractViolation, must still remove the container
        # and report safely rather than crash or leak a running container.
        code, response, docker = self._run("l5-raise-run", raise_on_inspect=True)
        self.assertEqual(code, 3)
        self.assertEqual(response["errors"][0]["code"], "INTERNAL_ERROR")
        self.assertTrue(any(call[:2] == ["rm", "-f"] for call in docker.rm_calls))
        self.assertFalse(any(call[:2] == ["start", "-a"] for call in docker.calls))

    # --- container exit code ---

    def test_nonzero_exit_code_does_not_move_run_or_write_attestation(self) -> None:
        run_name = "nonzero-exit-run"
        code, response, docker = self._run(run_name, start_returncode=4)
        self.assertEqual(code, 4)
        self.assertEqual(response["errors"][0]["code"], "CONTAINER_EXIT_NONZERO")
        self.assertFalse((fixed_output_root(self.repo_root) / run_name).exists())
        self.assertFalse((fixed_launch_root(self.repo_root) / run_name / cc.ATTESTATION_FILENAME).exists())

    # --- docker invocation environment ---

    def test_docker_calls_receive_minimal_env_plus_canary_only(self) -> None:
        _code, _response, docker = self._run("env-check-run")
        self.assertTrue(docker.envs)
        for env in docker.envs:
            canary_keys = [key for key in env if key.startswith(launcher.CANARY_ENV_NAME_PREFIX)]
            self.assertEqual(len(canary_keys), 1)
            self.assertLessEqual(set(env) - set(canary_keys), {"SystemRoot", "USERPROFILE", "PATH"})

    # --- success path ---

    def test_success_path_moves_run_and_writes_subset_only_attestation(self) -> None:
        run_name = "ok-run"
        code, response, docker = self._run(run_name)
        self.assertEqual(code, 0)
        self.assertEqual(response["status"], "complete")

        final_run_dir = fixed_output_root(self.repo_root) / run_name
        self.assertEqual(
            {p.name for p in final_run_dir.iterdir()}, {"findings.json", "summary.md", "run-manifest.json"}
        )
        # The staging mount-point folder itself (now empty) is left behind on
        # success; F12's empty-staging cleanup applies to failure paths only.
        self.assertEqual(list((fixed_staging_root(self.repo_root) / run_name).iterdir()), [])

        attestation_path = fixed_launch_root(self.repo_root) / run_name / cc.ATTESTATION_FILENAME
        attestation = json.loads(attestation_path.read_text("utf-8"))
        self.assertEqual(attestation["image_ref"], cc.IMAGE_REF)
        self.assertEqual(attestation["image_id"], cc.IMAGE_ID)
        self.assertEqual(attestation["launch_nonce"], docker.nonce)
        for mount in attestation["inspect_subset"]["Mounts"]:
            self.assertEqual(set(mount), {"Type", "Destination", "RW"})
        self.assertNotIn("Id", attestation["inspect_subset"])
        # G1: the launcher independently hashes the guard source before mounting.
        guard_path = self.repo_root / Path(*cc.GUARD_SOURCE_RELATIVE.split("/"))
        self.assertEqual(attestation["guard_source_sha256"], sha256_bytes(guard_path.read_bytes()))

    # --- G7: L8 pins python_version too, not just the scorer's S6 ---

    def test_g7_staged_python_version_mismatch_is_rejected_before_move(self) -> None:
        self._assert_l8_rejected(
            "g7-python-version",
            "STAGED_PYTHON_VERSION_MISMATCH",
            manifest_mutator=lambda m: m.__setitem__("python_version", "3.11.0"),
        )

    # --- container_report surfaced on a failed run (design §13 point 3 fix) ---

    def test_container_report_is_surfaced_on_nonzero_exit(self) -> None:
        run_name = "container-report-run"
        report_payload = {
            "schema_version": "offline-eval.report.v2",
            "status": "invalid",
            "phase": "bundle_validation",
        }
        code, response, _docker = self._run(
            run_name, start_returncode=4, start_stdout=json.dumps(report_payload)
        )
        self.assertEqual(code, 4)
        self.assertEqual(response["errors"][0]["code"], "CONTAINER_EXIT_NONZERO")
        self.assertEqual(response["container_report"], report_payload)

    def test_container_report_is_none_when_stdout_is_not_json(self) -> None:
        code, response, _docker = self._run(
            "container-report-garbage-run", start_returncode=4, start_stdout="not json"
        )
        self.assertEqual(code, 4)
        self.assertIsNone(response["container_report"])

    # --- docker rm failure after a run is non-fatal but reported (review #9) ---

    def test_docker_rm_failure_after_run_is_non_fatal_but_reported(self) -> None:
        run_name = "rm-fail-run"
        docker = FakeDocker(repo_root=self.repo_root, run_name=run_name, rm_returncode=1)
        captured_stdout = io.StringIO()
        captured_stderr = io.StringIO()
        with ExitStack() as stack:
            stack.enter_context(patch.object(launcher, "TOOL_DIR", self.tool_dir))
            stack.enter_context(patch.object(launcher, "_run_docker", docker))
            stack.enter_context(patch.object(launcher, "_resolve_docker_binary", return_value="docker"))
            stack.enter_context(redirect_stdout(captured_stdout))
            stack.enter_context(redirect_stderr(captured_stderr))
            code = launcher.run(["--run-name", run_name])
        self.assertEqual(code, 0)
        response = json.loads(captured_stdout.getvalue())
        self.assertEqual(response["status"], "complete")
        self.assertIn("docker rm failed", captured_stderr.getvalue())

    # --- H1: attestation-before-rename ordering and its failure paths ---

    def test_h1_attestation_write_failure_leaves_staged_run_unmoved(self) -> None:
        # Something already occupies the attestation path (e.g. a stray file
        # from an earlier partial attempt): the exclusive-create write fails,
        # and the run must stay in staging rather than move without proof.
        run_name = "h1-attest-fail"
        code, response, _docker = self._run(run_name, pre_create_attestation=True)
        self.assertEqual(code, 3)
        self.assertEqual(response["errors"][0]["code"], "ATTESTATION_WRITE_FAILED")
        self.assertFalse((fixed_output_root(self.repo_root) / run_name).exists())
        self.assertTrue(
            (fixed_staging_root(self.repo_root) / run_name / run_name / "run-manifest.json").exists()
        )

    def test_h1_rename_failure_deletes_attestation_and_cleans_launch_dir(self) -> None:
        # The invariant: a run present in phase3-runs always has an
        # attestation. If rename fails *after* the attestation was written,
        # the just-written attestation must be removed (never left orphaned,
        # bound to a run that never made it into phase3-runs) and the empty
        # launch folder cleaned up; the staged run itself is left for
        # inspection since it still holds real files.
        run_name = "h1-rename-fail"
        original_rename = Path.rename

        def failing_rename(self_path: Path, target: Path):
            if self_path.name == run_name and self_path.parent.name == run_name:
                raise OSError("synthetic rename failure")
            return original_rename(self_path, target)

        with patch.object(Path, "rename", failing_rename):
            code, response, _docker = self._run(run_name)
        self.assertEqual(code, 3)
        self.assertEqual(response["errors"][0]["code"], "RUN_MOVE_FAILED")
        self.assertFalse((fixed_output_root(self.repo_root) / run_name).exists())
        self.assertFalse((fixed_launch_root(self.repo_root) / run_name).exists())
        self.assertTrue(
            (fixed_staging_root(self.repo_root) / run_name / run_name / "run-manifest.json").exists()
        )

    def test_h1_success_path_writes_attestation_strictly_before_rename(self) -> None:
        # Positive-path corroboration of the H1 ordering: patch Path.rename to
        # record, at the moment it is called, whether the attestation file
        # already exists on disk.
        run_name = "h1-order-run"
        original_rename = Path.rename
        seen_before_rename: list[bool] = []

        def observing_rename(self_path: Path, target: Path):
            if self_path.name == run_name and self_path.parent.name == run_name:
                attestation_path = fixed_launch_root(self.repo_root) / run_name / cc.ATTESTATION_FILENAME
                seen_before_rename.append(attestation_path.exists())
            return original_rename(self_path, target)

        with patch.object(Path, "rename", observing_rename):
            code, _response, _docker = self._run(run_name)
        self.assertEqual(code, 0)
        self.assertEqual(seen_before_rename, [True])


class RunDockerHelperTests(unittest.TestCase):
    """`launcher._run_docker` itself (not the FakeDocker stand-in used above):
    the subprocess-failure paths L5/L6 depend on."""

    def test_timeout_is_reported_as_docker_timeout(self) -> None:
        with patch.object(
            subprocess, "run", side_effect=subprocess.TimeoutExpired(cmd=["docker"], timeout=1)
        ):
            with self.assertRaises(launcher.LauncherFailure) as raised:
                launcher._run_docker("docker", ["ps"], env={})
        self.assertEqual(raised.exception.code, "DOCKER_TIMEOUT")
        self.assertEqual(raised.exception.exit_code, 3)

    def test_oserror_is_reported_as_docker_invocation_failed(self) -> None:
        with patch.object(subprocess, "run", side_effect=OSError("no such file")):
            with self.assertRaises(launcher.LauncherFailure) as raised:
                launcher._run_docker("docker", ["ps"], env={})
        self.assertEqual(raised.exception.code, "DOCKER_INVOCATION_FAILED")
        self.assertEqual(raised.exception.exit_code, 3)


if __name__ == "__main__":
    unittest.main()
