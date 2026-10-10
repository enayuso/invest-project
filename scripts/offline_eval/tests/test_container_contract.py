from pathlib import Path
import sys
import unittest


TOOL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOL_DIR))

import container_contract as cc  # noqa: E402


VALID_NONCE = "a" * 64
RUN_NAME = "run-01"


def _valid_mounts() -> cc.HostMounts:
    return cc.HostMounts(
        tool_dir=Path("/tmp/tool") if sys.platform != "win32" else Path("C:/tool"),
        guard_file=Path("/tmp/guard.py") if sys.platform != "win32" else Path("C:/guard.py"),
        phase2_dir=Path("/tmp/phase2") if sys.platform != "win32" else Path("C:/phase2"),
        phase3a_dir=Path("/tmp/phase3a") if sys.platform != "win32" else Path("C:/phase3a"),
        staging_dir=Path("/tmp/staging") if sys.platform != "win32" else Path("C:/staging"),
    )


def _valid_inspect(nonce_hex: str = VALID_NONCE, *, run_name: str = RUN_NAME) -> dict:
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


class BuildDockerArgsTests(unittest.TestCase):
    def test_argument_vector_matches_the_fixed_contract(self) -> None:
        args = cc.build_docker_args(mounts=_valid_mounts(), nonce_hex=VALID_NONCE, run_name="run-01")
        self.assertEqual(args[0], "create")
        self.assertIn("--network", args)
        self.assertEqual(args[args.index("--network") + 1], "none")
        self.assertIn("--read-only", args)
        self.assertEqual(args[args.index("--cap-drop") + 1], "ALL")
        self.assertEqual(args[args.index("--user") + 1], cc.CONTAINER_USER)
        self.assertEqual(args[args.index("--hostname") + 1], cc.CONTAINER_HOSTNAME)
        self.assertEqual(args[args.index("--env") + 1], f"{cc.NONCE_ENV_NAME}={VALID_NONCE}")
        self.assertEqual(args.count("--mount"), 5)
        self.assertIn(cc.IMAGE_REF, args)
        self.assertIn(f"{cc.OUTPUT_MOUNT}/run-01", args)

    def test_argument_vector_matches_exactly(self) -> None:
        # A2 (design §16 H2): every non-path token below is a LITERAL string,
        # not assembled from `cc.*` constants -- only the mount *source* paths
        # (host-supplied, platform-dependent) come from the `mounts` object.
        # This is what makes the test able to fail: comparing cc.X against
        # cc.X (the old version of this test) can never catch cc.X itself
        # being wrong. See ContractConstantsTests below for the same principle
        # applied directly to the constants.
        mounts = _valid_mounts()
        args = cc.build_docker_args(mounts=mounts, nonce_hex=VALID_NONCE, run_name="run-01")
        expected = [
            "create",
            "--network", "none",
            "--read-only",
            "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges:true",
            "--user", "65534:65534",
            "--hostname", "offline-eval",
            "--pids-limit", "64",
            "--memory", "1g",
            "--cpus", "2",
            "--env", f"OFFLINE_EVAL_LAUNCH_NONCE={VALID_NONCE}",
            "--log-driver", "none",
            "--workdir", "/work",
            "--mount", f"type=bind,source={mounts.tool_dir},target=/work/repo/scripts/offline_eval,readonly",
            "--mount",
            f"type=bind,source={mounts.guard_file},target=/work/repo/app/agents/session_close_guard.py,readonly",
            "--mount",
            f"type=bind,source={mounts.phase2_dir},target=/work/handoff/evaluations/2026-09-22-phase2,readonly",
            "--mount",
            f"type=bind,source={mounts.phase3a_dir},target=/work/handoff/evaluations/2026-09-22-phase3a,readonly",
            "--mount", f"type=bind,source={mounts.staging_dir},target=/work/handoff/evaluations/phase3-runs",
            "docker.io/library/python@sha256:4766d8b510c428e595d74b9cc5bbb2fae8e26316fffb4adc89908d79aacd58a2",
            "python",
            "-I",
            "-S",
            "-B",
            "/work/repo/scripts/offline_eval/cli.py",
            "scan",
            "--bundle",
            "/work/handoff/evaluations/2026-09-22-phase2",
            "--profile",
            "/work/handoff/evaluations/2026-09-22-phase3a/scope-v1.json",
            "--out",
            "/work/handoff/evaluations/phase3-runs/run-01",
        ]
        self.assertEqual(args, expected)

    def test_forbidden_flags_are_never_present(self) -> None:
        args = cc.build_docker_args(mounts=_valid_mounts(), nonce_hex=VALID_NONCE, run_name="run-01")
        for forbidden in ("--privileged", "--cap-add", "--device", "--pid", "--ipc", "--uts", "--env-file", "-v"):
            self.assertNotIn(forbidden, args)
        # exactly one --env flag (the nonce), never a second env source
        self.assertEqual(args.count("--env"), 1)

    def test_rejects_malformed_nonce_or_run_name(self) -> None:
        with self.assertRaises(ValueError):
            cc.build_docker_args(mounts=_valid_mounts(), nonce_hex="not-hex", run_name="run-01")
        with self.assertRaises(ValueError):
            cc.build_docker_args(mounts=_valid_mounts(), nonce_hex=VALID_NONCE, run_name="../escape")

    def test_relative_mount_source_is_rejected(self) -> None:
        mounts = _valid_mounts()
        bad = cc.HostMounts(
            tool_dir=Path("relative"), guard_file=mounts.guard_file, phase2_dir=mounts.phase2_dir,
            phase3a_dir=mounts.phase3a_dir, staging_dir=mounts.staging_dir,
        )
        with self.assertRaises(ValueError):
            cc.build_docker_args(mounts=bad, nonce_hex=VALID_NONCE, run_name="run-01")


class ValidateInspectTests(unittest.TestCase):
    def test_valid_inspect_has_no_violations(self) -> None:
        self.assertEqual(cc.validate_inspect(_valid_inspect(), nonce_hex=VALID_NONCE, run_name=RUN_NAME), [])

    def test_each_deviation_is_rejected(self) -> None:
        cases = {
            "NETWORK_MODE_NOT_NONE": lambda d: d["HostConfig"].__setitem__("NetworkMode", "bridge"),
            "ROOTFS_NOT_READONLY": lambda d: d["HostConfig"].__setitem__("ReadonlyRootfs", False),
            "PRIVILEGED": lambda d: d["HostConfig"].__setitem__("Privileged", True),
            "CAP_DROP_NOT_ALL": lambda d: d["HostConfig"].__setitem__("CapDrop", []),
            "CAP_ADD_NOT_EMPTY": lambda d: d["HostConfig"].__setitem__("CapAdd", ["SYS_ADMIN"]),
            "DEVICES_NOT_EMPTY": lambda d: d["HostConfig"].__setitem__("Devices", [{"PathOnHost": "/dev/kvm"}]),
            "GROUP_ADD_NOT_EMPTY": lambda d: d["HostConfig"].__setitem__("GroupAdd", ["1000"]),
            "PIDS_LIMIT_MISMATCH": lambda d: d["HostConfig"].__setitem__("PidsLimit", 0),
            "NAMESPACE_MODE_REJECTED:PidMode": lambda d: d["HostConfig"].__setitem__("PidMode", "host"),
            "NAMESPACE_MODE_REJECTED:IpcMode": lambda d: d["HostConfig"].__setitem__("IpcMode", "container:other"),
            "NAMESPACE_MODE_REJECTED:UTSMode": lambda d: d["HostConfig"].__setitem__("UTSMode", "host"),
            "NAMESPACE_MODE_REJECTED:UsernsMode": lambda d: d["HostConfig"].__setitem__("UsernsMode", "host"),
            "TMPFS_NOT_EMPTY": lambda d: d["HostConfig"].__setitem__("Tmpfs", {"/tmp": "size=1"}),
            "BINDS_NOT_EMPTY": lambda d: d["HostConfig"].__setitem__("Binds", ["/:/host:ro"]),
            "VOLUMES_FROM_NOT_EMPTY": lambda d: d["HostConfig"].__setitem__("VolumesFrom", ["other"]),
            "MEMORY_MISMATCH": lambda d: d["HostConfig"].__setitem__("Memory", 1),
            "NANO_CPUS_MISMATCH": lambda d: d["HostConfig"].__setitem__("NanoCpus", 1),
            "LOG_CONFIG_NOT_NONE": lambda d: d["HostConfig"].__setitem__("LogConfig", {"Type": "json-file"}),
            "USER_NOT_PINNED": lambda d: d["Config"].__setitem__("User", "root"),
            "HOSTNAME_MISMATCH": lambda d: d["Config"].__setitem__("Hostname", "other"),
            "ENTRYPOINT_NOT_EMPTY": lambda d: d["Config"].__setitem__("Entrypoint", ["/bin/sh"]),
            "CMD_MISMATCH": lambda d: d["Config"].__setitem__("Cmd", ["python3"]),
            "IMAGE_REF_MISMATCH": lambda d: d.__setitem__("Image", "sha256:" + "0" * 64),
        }
        for expected_code, mutate in cases.items():
            with self.subTest(code=expected_code):
                inspect_subset = _valid_inspect()
                mutate(inspect_subset)
                violations = cc.validate_inspect(inspect_subset, nonce_hex=VALID_NONCE, run_name=RUN_NAME)
                self.assertIn(expected_code, violations)

    def test_security_opt_must_be_exactly_one_entry(self) -> None:
        # §15 G6/N6: exact-match subsumes every unconfined-seccomp spelling, a
        # user seccomp profile, systempaths=unconfined, and any addition --
        # without needing a separate check per spelling.
        variants = {
            "empty": [],
            "missing_entry": ["seccomp=unconfined"],
            "seccomp_equals_unconfined": ["no-new-privileges:true", "seccomp=unconfined"],
            "seccomp_colon_unconfined": ["no-new-privileges:true", "seccomp:unconfined"],
            "user_seccomp_profile": ["no-new-privileges:true", "seccomp=/some/profile.json"],
            "systempaths_unconfined": ["no-new-privileges:true", "apparmor=unconfined,systempaths=unconfined"],
            "duplicate_entry": ["no-new-privileges:true", "no-new-privileges:true"],
            "wrong_order_single": ["true:no-new-privileges"],
        }
        for name, security_opt in variants.items():
            with self.subTest(variant=name):
                inspect_subset = _valid_inspect()
                inspect_subset["HostConfig"]["SecurityOpt"] = security_opt
                violations = cc.validate_inspect(inspect_subset, nonce_hex=VALID_NONCE, run_name=RUN_NAME)
                self.assertIn("SECURITY_OPT_MISMATCH", violations)

    def test_group_add_not_empty_is_rejected(self) -> None:
        inspect_subset = _valid_inspect()
        inspect_subset["HostConfig"]["GroupAdd"] = ["1000"]
        violations = cc.validate_inspect(inspect_subset, nonce_hex=VALID_NONCE, run_name=RUN_NAME)
        self.assertIn("GROUP_ADD_NOT_EMPTY", violations)

    def test_runtime_env_keys_are_not_tolerated_in_inspect_config_env(self) -> None:
        # §15 G5/N5: unlike the process-env probe P9 (isolation.py), inspect's
        # Config.Env is the *requested* configuration and must never contain
        # HOSTNAME/HOME -- Docker/runc only inject those into the process env.
        inspect_subset = _valid_inspect()
        inspect_subset["Config"]["Env"].append(f"HOSTNAME={cc.CONTAINER_HOSTNAME}")
        violations = cc.validate_inspect(inspect_subset, nonce_hex=VALID_NONCE, run_name=RUN_NAME)
        self.assertIn("ENV_ALLOWLIST_MISMATCH", violations)

    def test_extra_env_variable_is_rejected(self) -> None:
        inspect_subset = _valid_inspect()
        inspect_subset["Config"]["Env"].append("HOST_CANARY=leaked")
        violations = cc.validate_inspect(inspect_subset, nonce_hex=VALID_NONCE, run_name=RUN_NAME)
        self.assertIn("ENV_ALLOWLIST_MISMATCH", violations)

    def test_wrong_nonce_value_is_rejected(self) -> None:
        inspect_subset = _valid_inspect(nonce_hex="b" * 64)
        violations = cc.validate_inspect(inspect_subset, nonce_hex=VALID_NONCE, run_name=RUN_NAME)
        self.assertIn("ENV_ALLOWLIST_MISMATCH", violations)


    def test_optional_runtime_env_key_with_wrong_value_is_rejected(self) -> None:
        inspect_subset = _valid_inspect()
        inspect_subset["Config"]["Env"].append("HOSTNAME=someone-elses-host")
        violations = cc.validate_inspect(inspect_subset, nonce_hex=VALID_NONCE, run_name=RUN_NAME)
        self.assertIn("ENV_ALLOWLIST_MISMATCH", violations)

    def test_missing_mount_is_rejected(self) -> None:
        inspect_subset = _valid_inspect()
        inspect_subset["Mounts"] = inspect_subset["Mounts"][:-1]
        violations = cc.validate_inspect(inspect_subset, nonce_hex=VALID_NONCE, run_name=RUN_NAME)
        self.assertIn("MOUNT_SET_MISMATCH", violations)

    def test_output_mount_rw_flag_wrong_way_is_rejected(self) -> None:
        inspect_subset = _valid_inspect()
        for mount in inspect_subset["Mounts"]:
            if mount["Destination"] == cc.OUTPUT_MOUNT:
                mount["RW"] = False
        violations = cc.validate_inspect(inspect_subset, nonce_hex=VALID_NONCE, run_name=RUN_NAME)
        self.assertTrue(any(v.startswith("MOUNT_RW_MISMATCH") for v in violations))

    def test_extra_bind_mount_is_rejected(self) -> None:
        # F4: inverted from the old "tolerated" expectation (review #3) -- Docker
        # reports its own injected /etc/* files outside Mounts, so a Mounts entry
        # at /etc/hosts is a real, unrequested bind and must be rejected.
        inspect_subset = _valid_inspect()
        inspect_subset["Mounts"].append({"Type": "bind", "Destination": "/etc/hosts", "RW": True})
        violations = cc.validate_inspect(inspect_subset, nonce_hex=VALID_NONCE, run_name=RUN_NAME)
        self.assertIn("MOUNT_UNEXPECTED_DESTINATION", violations)

    def test_whole_repo_bound_to_etc_hosts_is_rejected(self) -> None:
        inspect_subset = _valid_inspect()
        inspect_subset["Mounts"].append({"Type": "bind", "Destination": "/etc/hosts", "RW": True})
        violations = cc.validate_inspect(inspect_subset, nonce_hex=VALID_NONCE, run_name=RUN_NAME)
        self.assertTrue(violations)

    def test_mount_source_mismatch_is_rejected_when_sources_expected(self) -> None:
        inspect_subset = _valid_inspect()
        for mount in inspect_subset["Mounts"]:
            mount["Source"] = "/somewhere/else"
        expected_sources = {destination: f"/expected{destination}" for destination in cc.REQUIRED_MOUNT_DESTINATIONS}
        violations = cc.validate_inspect(
            inspect_subset, nonce_hex=VALID_NONCE, run_name=RUN_NAME, expected_sources=expected_sources
        )
        self.assertTrue(any(v.startswith("MOUNT_SOURCE_MISMATCH") for v in violations))

    def test_mount_source_matches_after_normalization(self) -> None:
        inspect_subset = _valid_inspect()
        for mount in inspect_subset["Mounts"]:
            mount["Source"] = f"/run/desktop/mnt/host/d{mount['Destination']}".replace("//", "/")
        expected_sources = {destination: f"D:{destination}" for destination in cc.REQUIRED_MOUNT_DESTINATIONS}
        violations = cc.validate_inspect(
            inspect_subset, nonce_hex=VALID_NONCE, run_name=RUN_NAME, expected_sources=expected_sources
        )
        self.assertEqual([v for v in violations if v.startswith("MOUNT_SOURCE_MISMATCH")], [])

    def test_mount_source_normalization_preserves_drive_letter(self) -> None:
        # §15 G9: dropping the drive letter would make a source on D: compare
        # equal to the same relative path on C: -- the drive letter must
        # survive normalization so they are distinguished.
        inspect_subset = _valid_inspect()
        for mount in inspect_subset["Mounts"]:
            mount["Source"] = f"D:{mount['Destination']}"
        expected_sources = {destination: f"C:{destination}" for destination in cc.REQUIRED_MOUNT_DESTINATIONS}
        violations = cc.validate_inspect(
            inspect_subset, nonce_hex=VALID_NONCE, run_name=RUN_NAME, expected_sources=expected_sources
        )
        self.assertTrue(any(v.startswith("MOUNT_SOURCE_MISMATCH") for v in violations))

    def test_source_not_checked_when_expected_sources_omitted(self) -> None:
        inspect_subset = _valid_inspect()
        for mount in inspect_subset["Mounts"]:
            mount["Source"] = "/anything"
        self.assertEqual(cc.validate_inspect(inspect_subset, nonce_hex=VALID_NONCE, run_name=RUN_NAME), [])

    def test_malformed_input_never_raises(self) -> None:
        for garbage in ({}, {"HostConfig": None}, {"HostConfig": {}, "Config": {}}):
            violations = cc.validate_inspect(garbage, nonce_hex=VALID_NONCE, run_name=RUN_NAME)
            self.assertTrue(violations)

    def test_non_string_mount_destination_never_raises(self) -> None:
        # G4/N4: a non-string Destination (e.g. a list) must be rejected as a
        # violation, never reach the `in <dict>` membership test and raise
        # TypeError for an unhashable key.
        inspect_subset = _valid_inspect()
        inspect_subset["Mounts"][0]["Destination"] = ["not", "a", "string"]
        violations = cc.validate_inspect(inspect_subset, nonce_hex=VALID_NONCE, run_name=RUN_NAME)
        self.assertIn("MOUNT_ENTRY_INVALID", violations)

    def test_invalid_nonce_argument_is_its_own_violation(self) -> None:
        self.assertEqual(
            cc.validate_inspect(_valid_inspect(), nonce_hex="not-hex", run_name=RUN_NAME), ["NONCE_ARGUMENT_INVALID"]
        )

    def test_invalid_run_name_is_its_own_violation(self) -> None:
        violations = cc.validate_inspect(_valid_inspect(), nonce_hex=VALID_NONCE, run_name="../escape")
        self.assertIn("RUN_NAME_INVALID", violations)


class InspectSubsetTests(unittest.TestCase):
    def test_reduces_to_checked_fields_and_drops_source(self) -> None:
        full = _valid_inspect()
        for mount in full["Mounts"]:
            mount["Source"] = "D:/some/host/path"
        full["Id"] = "sha256:" + "1" * 64
        reduced = cc.inspect_subset(full)
        self.assertNotIn("Source", reduced["Mounts"][0])
        self.assertEqual(set(reduced["Mounts"][0]), {"Type", "Destination", "RW"})
        self.assertNotIn("Id", reduced)
        self.assertEqual(
            cc.validate_inspect(reduced, nonce_hex=VALID_NONCE, run_name=RUN_NAME, expected_sources=None), []
        )


class ValidateIsolationBlockTests(unittest.TestCase):
    def _valid_block(self) -> dict:
        return {
            "os_level_sandbox_confirmed": True,
            "security_boundary": cc.CONFIRMED_SECURITY_BOUNDARY,
            "os_isolation": {
                "status": "confirmed",
                "probes": [{"id": f"P{i}", "passed": True} for i in range(1, 12)],
            },
        }

    def test_valid_block_has_no_violations(self) -> None:
        self.assertEqual(cc.validate_isolation_block(self._valid_block()), [])

    def test_not_a_dict_is_rejected(self) -> None:
        self.assertEqual(cc.validate_isolation_block(None), ["ISOLATION_BLOCK_INVALID"])

    def test_missing_probe_is_rejected(self) -> None:
        block = self._valid_block()
        block["os_isolation"]["probes"] = block["os_isolation"]["probes"][:-1]
        self.assertIn("PROBE_SET_MISMATCH", cc.validate_isolation_block(block))

    def test_duplicate_probe_id_is_rejected(self) -> None:
        block = self._valid_block()
        block["os_isolation"]["probes"].append({"id": "P1", "passed": True})
        violations = cc.validate_isolation_block(block)
        self.assertIn("PROBE_DUPLICATE_ID", violations)

    def test_extra_probe_id_is_rejected(self) -> None:
        block = self._valid_block()
        block["os_isolation"]["probes"].append({"id": "P12", "passed": True})
        self.assertIn("PROBE_SET_MISMATCH", cc.validate_isolation_block(block))

    def test_one_failed_probe_is_rejected(self) -> None:
        block = self._valid_block()
        block["os_isolation"]["probes"][0]["passed"] = False
        self.assertIn("PROBE_NOT_PASSED:P1", cc.validate_isolation_block(block))

    def test_probe_passed_not_bool_is_rejected(self) -> None:
        block = self._valid_block()
        block["os_isolation"]["probes"][0]["passed"] = 1
        self.assertIn("PROBE_NOT_PASSED:P1", cc.validate_isolation_block(block))

    def test_non_string_probe_id_never_raises(self) -> None:
        # G4/N4: a non-string probe id (e.g. a list) must be rejected as a
        # violation, never reach the `in <dict>` membership test and raise
        # TypeError for an unhashable key.
        block = self._valid_block()
        block["os_isolation"]["probes"][0]["id"] = ["not", "a", "string"]
        violations = cc.validate_isolation_block(block)
        self.assertIn("PROBE_ENTRY_INVALID", violations)


class ContractConstantsTests(unittest.TestCase):
    """design §16 H2: pin the literal VALUES of the fixed contract constants
    themselves. Every earlier test in this module that compares `cc.X` against
    `cc.X` (or builds its expected value out of `cc.X`) is self-consistency,
    not value-pinning -- it cannot catch `cc.X` itself silently changing. These
    assertions are written against independently-typed literals so that, e.g.,
    changing CONTAINER_USER to "0:0" fails here even though every self-
    referential test in this file would keep passing."""

    def test_image_reference_constants(self) -> None:
        self.assertEqual(
            cc.IMAGE_DIGEST,
            "sha256:4766d8b510c428e595d74b9cc5bbb2fae8e26316fffb4adc89908d79aacd58a2",
        )
        self.assertEqual(
            cc.IMAGE_REF,
            "docker.io/library/python@sha256:4766d8b510c428e595d74b9cc5bbb2fae8e26316fffb4adc89908d79aacd58a2",
        )
        self.assertEqual(
            cc.IMAGE_ID,
            "sha256:4766d8b510c428e595d74b9cc5bbb2fae8e26316fffb4adc89908d79aacd58a2",
        )

    def test_container_identity_constants(self) -> None:
        self.assertEqual(cc.CONTAINER_USER, "65534:65534")
        self.assertEqual(cc.CONTAINER_UID, 65534)
        self.assertEqual(cc.CONTAINER_HOSTNAME, "offline-eval")
        self.assertEqual(cc.WORKDIR, "/work")
        self.assertEqual(cc.NONCE_ENV_NAME, "OFFLINE_EVAL_LAUNCH_NONCE")

    def test_python_version_constants(self) -> None:
        self.assertEqual(cc.PYTHON_VERSION, (3, 12, 13))
        self.assertEqual(cc.PYTHON_VERSION_STRING, "3.12.13")

    def test_resource_limit_constants(self) -> None:
        self.assertEqual(cc.RESOURCE_LIMITS, {"pids_limit": "64", "memory": "1g", "cpus": "2"})
        self.assertEqual(cc.EXPECTED_PIDS_LIMIT, 64)
        self.assertEqual(cc.EXPECTED_MEMORY_BYTES, 1073741824)
        self.assertEqual(cc.EXPECTED_NANO_CPUS, 2000000000)

    def test_required_mount_destinations_and_direction(self) -> None:
        self.assertEqual(
            cc.REQUIRED_MOUNT_DESTINATIONS,
            {
                "/work/repo/scripts/offline_eval": False,
                "/work/repo/app/agents/session_close_guard.py": False,
                "/work/handoff/evaluations/2026-09-22-phase2": False,
                "/work/handoff/evaluations/2026-09-22-phase3a": False,
                "/work/handoff/evaluations/phase3-runs": True,
            },
        )

    def test_base_image_env_allowlist(self) -> None:
        self.assertEqual(
            cc.BASE_IMAGE_ENV_ALLOWLIST,
            frozenset(
                {
                    "PATH=/usr/local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
                    "LANG=C.UTF-8",
                    "GPG_KEY=7169605F62C751356D054A26A821E680E5FA6305",
                    "PYTHON_VERSION=3.12.13",
                    "PYTHON_SHA256=c08bc65a81971c1dd5783182826503369466c7e67374d1646519adf05207b684",
                }
            ),
        )

    def test_tool_source_files(self) -> None:
        self.assertEqual(
            cc.TOOL_SOURCE_FILES,
            (
                "bundle.py",
                "cli.py",
                "container_contract.py",
                "contracts.py",
                "isolation.py",
                "launcher.py",
                "report.py",
                "scoring.py",
                "session_adapter.py",
            ),
        )

    def test_build_scan_command_full_literal_vector(self) -> None:
        # design §16 H2: -I -S -B must be literally present -- these are what
        # makes the container's own interpreter invocation isolated, site-free,
        # and bytecode-free. Dropping any one of them must fail this test.
        self.assertEqual(
            cc.build_scan_command(run_name="run-01"),
            [
                "python",
                "-I",
                "-S",
                "-B",
                "/work/repo/scripts/offline_eval/cli.py",
                "scan",
                "--bundle",
                "/work/handoff/evaluations/2026-09-22-phase2",
                "--profile",
                "/work/handoff/evaluations/2026-09-22-phase3a/scope-v1.json",
                "--out",
                "/work/handoff/evaluations/phase3-runs/run-01",
            ],
        )


if __name__ == "__main__":
    unittest.main()
