"""Pure-function tests for the P1-P11 self-probes (design doc §4, A1, §14.2)."""

from pathlib import Path
import sys
import unittest


TOOL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOL_DIR))

import container_contract as cc  # noqa: E402
import isolation as iso  # noqa: E402


NONCE = "a" * 64


def _mountinfo_line(mount_id, parent_id, mount_point, options, fs_type="ext4", source="/dev/root"):
    return f"{mount_id} {parent_id} 0:{mount_id} / {mount_point} {options} - {fs_type} {source} rw"


def _valid_mountinfo() -> str:
    lines = [_mountinfo_line(1, 0, "/", "ro,relatime")]
    mount_id = 2
    for destination, rw in cc.REQUIRED_MOUNT_DESTINATIONS.items():
        lines.append(_mountinfo_line(mount_id, 1, destination, "rw" if rw else "ro"))
        mount_id += 1
    lines.append(_mountinfo_line(mount_id, 1, "/proc", "rw,nosuid,nodev,noexec,relatime", fs_type="proc"))
    mount_id += 1
    lines.append(_mountinfo_line(mount_id, 1, "/sys", "ro,nosuid,nodev,noexec,relatime", fs_type="sysfs"))
    return "\n".join(lines) + "\n"


def _valid_status_text() -> str:
    return "\n".join(
        [
            "Name:\tpython3",
            "Uid:\t65534\t65534\t65534\t65534",
            "Gid:\t65534\t65534\t65534\t65534",
            "NoNewPrivs:\t1",
            "Seccomp:\t2",
            "CapEff:\t0000000000000000",
        ]
    )


class ProbeParserTests(unittest.TestCase):
    def test_p1_only_loopback(self) -> None:
        self.assertTrue(iso._probe_p1_loopback_only(["lo"]).passed)
        self.assertFalse(iso._probe_p1_loopback_only(["lo", "eth0"]).passed)
        self.assertFalse(iso._probe_p1_loopback_only([]).passed)

    def test_p1_allows_fallback_tunnel_interfaces(self) -> None:
        self.assertTrue(iso._probe_p1_loopback_only(["lo", "tunl0"]).passed)
        self.assertTrue(iso._probe_p1_loopback_only(["lo", "tunl0", "ip6tnl0", "sit0"]).passed)

    def test_p1_rejects_a_real_interface_alongside_tunnels(self) -> None:
        self.assertFalse(iso._probe_p1_loopback_only(["lo", "tunl0", "eth0"]).passed)

    def test_p2_no_routes(self) -> None:
        self.assertTrue(iso._probe_p2_no_routes("Iface\tDestination\tGateway\n").passed)
        self.assertFalse(iso._probe_p2_no_routes("Iface\tDestination\tGateway\neth0\t00000000\t...\n").passed)

    def test_p2_empty_input_is_fail_closed(self) -> None:
        self.assertFalse(iso._probe_p2_no_routes("").passed)
        self.assertFalse(iso._probe_p2_no_routes("\n\n").passed)

    def test_p2_missing_header_is_rejected(self) -> None:
        self.assertFalse(iso._probe_p2_no_routes("eth0\t00000000\t...\n").passed)

    def test_p3_classifier(self) -> None:
        self.assertTrue(iso._probe_p3_from_connect_errno(101, timed_out=False, connected=False).passed)
        self.assertFalse(iso._probe_p3_from_connect_errno(None, timed_out=False, connected=True).passed)
        self.assertFalse(iso._probe_p3_from_connect_errno(None, timed_out=True, connected=False).passed)
        self.assertFalse(iso._probe_p3_from_connect_errno(111, timed_out=False, connected=False).passed)

    def test_p4_root_readonly(self) -> None:
        self.assertTrue(iso._probe_p4_root_readonly(_valid_mountinfo()).passed)
        rw_root = _valid_mountinfo().replace("ro,relatime", "rw,relatime")
        self.assertFalse(iso._probe_p4_root_readonly(rw_root).passed)
        self.assertFalse(iso._probe_p4_root_readonly("not mountinfo at all").passed)

    def test_p5_mount_set_matches_contract(self) -> None:
        self.assertTrue(iso._probe_p5_mount_set(_valid_mountinfo()).passed)

    def test_p5_missing_required_mount_fails(self) -> None:
        lines = _valid_mountinfo().splitlines()
        truncated = "\n".join(line for line in lines if cc.OUTPUT_MOUNT not in line)
        self.assertFalse(iso._probe_p5_mount_set(truncated).passed)

    def test_p5_wrong_rw_direction_fails(self) -> None:
        text = _valid_mountinfo().replace(f"{cc.OUTPUT_MOUNT} rw", f"{cc.OUTPUT_MOUNT} ro")
        self.assertFalse(iso._probe_p5_mount_set(text).passed)

    def test_p5_extra_rw_mount_is_rejected(self) -> None:
        lines = _valid_mountinfo().splitlines()
        lines.append(_mountinfo_line(99, 1, "/host-repo", "rw", fs_type="ext4"))
        self.assertFalse(iso._probe_p5_mount_set("\n".join(lines) + "\n").passed)

    def test_p5_tolerates_docker_injected_etc_files(self) -> None:
        lines = _valid_mountinfo().splitlines()
        lines.append(_mountinfo_line(100, 1, "/etc/hosts", "rw", fs_type="ext4"))
        lines.append(_mountinfo_line(101, 1, "/etc/hostname", "rw", fs_type="ext4"))
        lines.append(_mountinfo_line(102, 1, "/etc/resolv.conf", "rw", fs_type="ext4"))
        self.assertTrue(iso._probe_p5_mount_set("\n".join(lines) + "\n").passed)

    def test_p5_tolerates_pseudo_fs_under_dev(self) -> None:
        lines = _valid_mountinfo().splitlines()
        lines.append(_mountinfo_line(103, 1, "/dev", "rw", fs_type="tmpfs"))
        lines.append(_mountinfo_line(104, 1, "/dev/pts", "rw", fs_type="devpts"))
        lines.append(_mountinfo_line(105, 1, "/dev/mqueue", "rw", fs_type="mqueue"))
        self.assertTrue(iso._probe_p5_mount_set("\n".join(lines) + "\n").passed)

    def test_p5_rejects_non_pseudo_fs_under_proc(self) -> None:
        lines = _valid_mountinfo().splitlines()
        lines.append(_mountinfo_line(106, 1, "/proc/oddity", "rw", fs_type="ext4"))
        self.assertFalse(iso._probe_p5_mount_set("\n".join(lines) + "\n").passed)

    def test_p6_ro_targets_not_writable(self) -> None:
        result = iso._probe_p6_ro_not_writable(["/a", "/b"], lambda path, mode: False)
        self.assertTrue(result.passed)
        result = iso._probe_p6_ro_not_writable(["/a", "/b"], lambda path, mode: path == "/b")
        self.assertFalse(result.passed)

    def test_p7_classifier(self) -> None:
        self.assertTrue(iso._probe_p7_from_write_errno(30, wrote=False).passed)  # EROFS
        self.assertFalse(iso._probe_p7_from_write_errno(None, wrote=True).passed)
        self.assertFalse(iso._probe_p7_from_write_errno(13, wrote=False).passed)  # EACCES, not EROFS

    def test_p8_proc_status(self) -> None:
        self.assertTrue(iso._probe_p8_proc_status(_valid_status_text()).passed)

    def test_p8_root_uid_fails(self) -> None:
        text = _valid_status_text().replace("65534\t65534\t65534\t65534", "0\t0\t0\t0")
        self.assertFalse(iso._probe_p8_proc_status(text).passed)

    def test_p8_no_new_privs_fails(self) -> None:
        text = _valid_status_text().replace("NoNewPrivs:\t1", "NoNewPrivs:\t0")
        self.assertFalse(iso._probe_p8_proc_status(text).passed)

    def test_p8_seccomp_disabled_fails(self) -> None:
        text = _valid_status_text().replace("Seccomp:\t2", "Seccomp:\t0")
        self.assertFalse(iso._probe_p8_proc_status(text).passed)

    def test_p9_environ_exact_match(self) -> None:
        env = {kv.split("=", 1)[0]: kv.split("=", 1)[1] for kv in cc.BASE_IMAGE_ENV_ALLOWLIST}
        env[cc.NONCE_ENV_NAME] = NONCE
        self.assertTrue(iso._probe_p9_environ(env, NONCE).passed)

    def test_p9_allows_hostname_and_home_with_fixed_values(self) -> None:
        env = {kv.split("=", 1)[0]: kv.split("=", 1)[1] for kv in cc.BASE_IMAGE_ENV_ALLOWLIST}
        env[cc.NONCE_ENV_NAME] = NONCE
        env["HOSTNAME"] = cc.CONTAINER_HOSTNAME
        env["HOME"] = "/nonexistent"
        self.assertTrue(iso._probe_p9_environ(env, NONCE).passed)

    def test_p9_rejects_hostname_with_wrong_value(self) -> None:
        env = {kv.split("=", 1)[0]: kv.split("=", 1)[1] for kv in cc.BASE_IMAGE_ENV_ALLOWLIST}
        env[cc.NONCE_ENV_NAME] = NONCE
        env["HOSTNAME"] = "someone-elses-host"
        self.assertFalse(iso._probe_p9_environ(env, NONCE).passed)

    def test_p9_extra_variable_fails(self) -> None:
        env = {kv.split("=", 1)[0]: kv.split("=", 1)[1] for kv in cc.BASE_IMAGE_ENV_ALLOWLIST}
        env[cc.NONCE_ENV_NAME] = NONCE
        env["HOST_CANARY"] = "leaked"
        self.assertFalse(iso._probe_p9_environ(env, NONCE).passed)

    def test_p9_missing_nonce_fails(self) -> None:
        env = {kv.split("=", 1)[0]: kv.split("=", 1)[1] for kv in cc.BASE_IMAGE_ENV_ALLOWLIST}
        self.assertFalse(iso._probe_p9_environ(env, NONCE).passed)

    def test_p10_mirrors_p9(self) -> None:
        passing = iso.ProbeResult("P9", True, "ok")
        failing = iso.ProbeResult("P9", False, "bad")
        self.assertTrue(iso._probe_p10_from_p9(passing).passed)
        self.assertFalse(iso._probe_p10_from_p9(failing).passed)

    def test_p11_python_version(self) -> None:
        self.assertTrue(iso._probe_p11_python_version(cc.PYTHON_VERSION).passed)
        self.assertFalse(iso._probe_p11_python_version((3, 11, 0)).passed)

    def test_mountinfo_parser_rejects_malformed_lines(self) -> None:
        with self.assertRaises(ValueError):
            iso.parse_mountinfo("this is not a mountinfo line\n")


class AssessOsIsolationTests(unittest.TestCase):
    def test_windows_reason_is_container_launcher_required(self) -> None:
        if iso.os.name != "nt":
            self.skipTest("Windows-only branch")
        assessment = iso.assess_os_isolation()
        self.assertEqual(assessment.status, "verification_unimplemented")
        self.assertEqual(assessment.reason, "windows_host_requires_container_launcher")


if __name__ == "__main__":
    unittest.main()
