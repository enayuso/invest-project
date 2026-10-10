"""Defense-in-depth controls for executing one hash-pinned detector source."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
import socket
import sys
from typing import Any

from contracts import ContractViolation, ValidationIssue, path_has_reparse_component
import container_contract as cc


# F13: single source is container_contract; do not redefine the pattern here.
RUN_NAME_RE = cc.RUN_NAME_RE


@dataclass(frozen=True)
class ProbeResult:
    """One self-check from design doc §4 (P1..P11). Pure data, no side effects."""

    probe_id: str
    passed: bool
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {"id": self.probe_id, "passed": self.passed, "reason": self.reason}


@dataclass(frozen=True)
class _MountInfoEntry:
    mount_point: str
    options: frozenset[str]
    fs_type: str
    source: str


def parse_mountinfo(text: str) -> list[_MountInfoEntry]:
    """Pure parser for `/proc/self/mountinfo` (see proc(5)). Raises ValueError on a
    line that does not fit the expected shape -- callers must treat that as a probe
    failure, never as "no information"."""

    entries: list[_MountInfoEntry] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        halves = line.split(" - ")
        if len(halves) != 2:
            raise ValueError("mountinfo line missing separator")
        left = halves[0].split()
        right = halves[1].split()
        if len(left) < 6 or len(right) < 2:
            raise ValueError("mountinfo line has too few fields")
        entries.append(
            _MountInfoEntry(
                mount_point=left[4],
                options=frozenset(left[5].split(",")),
                fs_type=right[0],
                source=right[1],
            )
        )
    return entries


def _probe_p1_loopback_only(net_class_entries: list[str]) -> ProbeResult:
    # F9: a --network none namespace can still carry kernel fallback tunnel
    # devices (tunl0 etc.) alongside lo; anything beyond that set is a real
    # network interface and fails.
    names = {entry for entry in net_class_entries if entry not in (".", "..")}
    if "lo" not in names:
        return ProbeResult("P1", False, "loopback_missing")
    unexpected = names - {"lo"} - cc.FALLBACK_TUNNEL_INTERFACES
    if unexpected:
        return ProbeResult("P1", False, f"unexpected_interfaces:{len(unexpected)}")
    return ProbeResult("P1", True, "only_loopback_interface")


def _probe_p2_no_routes(route_table_text: str) -> ProbeResult:
    # F10: fail-closed. An empty/headerless read is not "no information", it is a
    # probe failure -- a real /proc/net/route always has the `Iface` header line.
    lines = [line for line in route_table_text.splitlines() if line.strip()]
    if not lines or not lines[0].startswith("Iface"):
        return ProbeResult("P2", False, "route_header_missing")
    data_lines = lines[1:]
    if not data_lines:
        return ProbeResult("P2", True, "no_routes")
    return ProbeResult("P2", False, f"routes_present:{len(data_lines)}")


def _probe_p3_from_connect_errno(errno_value: int | None, *, timed_out: bool, connected: bool) -> ProbeResult:
    """Pure classifier for the P3 sentinel. The impure half (the actual connect
    attempt to a TEST-NET-1 address) lives in `_attempt_p3_connect`."""

    if connected:
        return ProbeResult("P3", False, "connect_succeeded")
    if timed_out:
        return ProbeResult("P3", False, "connect_timed_out")
    if errno_value == 101:  # ENETUNREACH
        return ProbeResult("P3", True, "enetunreach_as_expected")
    return ProbeResult("P3", False, f"unexpected_errno:{errno_value}")


def _probe_p4_root_readonly(mountinfo_text: str) -> ProbeResult:
    try:
        entries = parse_mountinfo(mountinfo_text)
    except ValueError:
        return ProbeResult("P4", False, "mountinfo_unparseable")
    root_entries = [entry for entry in entries if entry.mount_point == "/"]
    if not root_entries:
        return ProbeResult("P4", False, "root_mount_not_found")
    if "ro" in root_entries[-1].options:
        return ProbeResult("P4", True, "root_readonly")
    return ProbeResult("P4", False, "root_not_readonly")


def _under_pseudo_fs_root(mount_point: str) -> bool:
    return any(
        mount_point == root or mount_point.startswith(f"{root}/") for root in ("/proc", "/sys", "/dev")
    )


def _probe_p5_mount_set(mountinfo_text: str) -> ProbeResult:
    """F3: exact-match, not "these 5 exist somewhere". Every mountinfo entry must
    be one of: (a) `/` ro, (b) one of the 5 required mounts with the right
    direction, (c) a Docker-injected /etc/* file, (d) a pseudo-fs entry under
    /proc, /sys, or /dev. Anything else fails the probe."""

    try:
        entries = parse_mountinfo(mountinfo_text)
    except ValueError:
        return ProbeResult("P5", False, "mountinfo_unparseable")
    seen_required: dict[str, frozenset[str]] = {}
    for entry in entries:
        point = entry.mount_point
        if point == "/":
            if "ro" not in entry.options:
                return ProbeResult("P5", False, "root_not_readonly")
            continue
        if point in cc.REQUIRED_MOUNT_DESTINATIONS:
            if point in seen_required:
                return ProbeResult("P5", False, f"duplicate_required_mount:{point}")
            seen_required[point] = entry.options
            continue
        if point in cc.DOCKER_INJECTED_MOUNT_PREFIXES:
            continue
        if _under_pseudo_fs_root(point) and entry.fs_type in cc.PSEUDO_FS_TYPES:
            continue
        return ProbeResult("P5", False, f"unexpected_mount:{point}")
    missing = set(cc.REQUIRED_MOUNT_DESTINATIONS) - set(seen_required)
    if missing:
        return ProbeResult("P5", False, f"missing_mounts:{len(missing)}")
    for destination, expect_rw in cc.REQUIRED_MOUNT_DESTINATIONS.items():
        options = seen_required[destination]
        is_rw = "rw" in options
        is_ro = "ro" in options
        if expect_rw and not is_rw:
            return ProbeResult("P5", False, f"expected_rw_got_ro:{destination}")
        if not expect_rw and not is_ro:
            return ProbeResult("P5", False, f"expected_ro_got_rw:{destination}")
    return ProbeResult("P5", True, "mount_set_matches_contract")


def _probe_p6_ro_not_writable(ro_targets: list[str], writable_check: Any) -> ProbeResult:
    """`writable_check` is `os.access` (or a stand-in in tests): a callable taking
    (path, mode) -> bool."""

    for target in ro_targets:
        if writable_check(target, os.W_OK):
            return ProbeResult("P6", False, f"writable:{target}")
    return ProbeResult("P6", True, "ro_mounts_not_writable")


def _probe_p7_from_write_errno(errno_value: int | None, *, wrote: bool) -> ProbeResult:
    if wrote:
        return ProbeResult("P7", False, "root_write_succeeded")
    if errno_value == 30:  # EROFS
        return ProbeResult("P7", True, "erofs_as_expected")
    return ProbeResult("P7", False, f"unexpected_errno:{errno_value}")


def _probe_p8_proc_status(status_text: str) -> ProbeResult:
    fields: dict[str, str] = {}
    for line in status_text.splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        fields[key.strip()] = value.strip()
    uid_line = fields.get("Uid", "")
    uids = uid_line.split()
    if len(uids) != 4 or any(not uid.isdigit() for uid in uids):
        return ProbeResult("P8", False, "uid_line_unparseable")
    if any(uid == "0" for uid in uids):
        return ProbeResult("P8", False, "uid_is_root")
    if fields.get("CapEff") not in {"0000000000000000", "0"}:
        return ProbeResult("P8", False, "cap_eff_nonzero")
    if fields.get("NoNewPrivs") != "1":
        return ProbeResult("P8", False, "no_new_privs_not_set")
    if fields.get("Seccomp") != "2":
        return ProbeResult("P8", False, "seccomp_not_filter_mode")
    return ProbeResult("P8", True, "proc_status_matches_contract")


def _probe_p9_environ(env: dict[str, str], nonce_hex: str) -> ProbeResult:
    if cc.NONCE_RE.fullmatch(nonce_hex) is None:
        return ProbeResult("P9", False, "nonce_argument_invalid")
    entries = [f"{key}={value}" for key, value in env.items()]
    if len(entries) != len(set(entries)):
        return ProbeResult("P9", False, "duplicate_env_entry")
    # G5: P9 reads the container's process environment, where Docker/runc
    # legitimately inject HOSTNAME (and HOME when unset) -- unlike inspect's
    # Config.Env (G5 in container_contract.validate_inspect), runtime keys are
    # tolerated here, with their values still pinned exactly.
    if not cc.env_list_matches_contract(entries, nonce_hex, allow_runtime_keys=True):
        return ProbeResult("P9", False, "env_allowlist_mismatch")
    return ProbeResult("P9", True, "env_matches_allowlist_plus_nonce")


def _probe_p10_from_p9(p9: ProbeResult) -> ProbeResult:
    # No host canary can reach the container: P9's exact-match check already
    # forbids any environment variable outside the fixed allowlist + nonce, so a
    # passing P9 implies no host canary leaked in. See design doc §4 P10.
    return ProbeResult("P10", p9.passed, "implied_by_p9_exact_env_match")


def _probe_p11_python_version(version_info: tuple[int, int, int]) -> ProbeResult:
    if tuple(version_info) == cc.PYTHON_VERSION:
        return ProbeResult("P11", True, "python_version_pinned")
    return ProbeResult("P11", False, f"unexpected_version:{'.'.join(map(str, version_info))}")


class IsolationViolation(Exception):
    """A safe failure raised before a prohibited runtime action occurs."""

    def __init__(self, code: str, reference: str, description: str) -> None:
        super().__init__(description)
        self.issue = ValidationIssue(code, reference, description)


@dataclass(frozen=True)
class OsIsolationAssessment:
    status: str
    method: str | None
    reason: str
    probes: tuple[ProbeResult, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "method": self.method,
            "reason": self.reason,
            "probes": [probe.as_dict() for probe in self.probes],
        }


@dataclass(frozen=True)
class IsolationStatus:
    audit_hook_installed: bool
    sentinel_blocked: bool
    os_level_sandbox_confirmed: bool
    security_boundary: str
    os_isolation: OsIsolationAssessment | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "audit_hook_installed": self.audit_hook_installed,
            "sentinel_blocked": self.sentinel_blocked,
            "os_level_sandbox_confirmed": self.os_level_sandbox_confirmed,
            "security_boundary": self.security_boundary,
            "os_isolation": self.os_isolation.as_dict() if self.os_isolation else None,
        }


class RuntimePolicy:
    """Audit-hook policy; useful defense, explicitly not an OS sandbox."""

    _BLOCKED_PREFIXES = (
        "socket.",
        "subprocess.",
        "ctypes.",
        "winreg.",
    )
    _BLOCKED_EVENTS = {
        "builtins.input",
        "builtins.breakpoint",
        "compile",
        "os.system",
        "os.putenv",
        "os.unsetenv",
        "os.chdir",
        "os.fchdir",
        "os.chdir",
        "os.remove",
        "os.rename",
        "os.replace",
        "os.rmdir",
        "os.mkdir",
        "os.listdir",
        "os.scandir",
        "os.spawn",
        "os.startfile",
        "glob.glob",
        "glob.glob/2",
        "webbrowser.open",
    }

    def __init__(self, *, scanner_filename: str, output_directory: Path | None) -> None:
        self.scanner_filename = scanner_filename
        self.output_directory = output_directory.resolve(strict=True) if output_directory else None

    def _deny(self, event: str) -> None:
        raise IsolationViolation(
            "ISOLATION_BLOCKED",
            "runtime",
            f"Runtime isolation blocked event category: {event.split('.', 1)[0]}",
        )

    def _allow_exclusive_output_open(self, args: tuple[Any, ...]) -> bool:
        if self.output_directory is None or not args:
            return False
        target = args[0]
        if not isinstance(target, (str, bytes, os.PathLike)):
            return False
        try:
            path = Path(target).resolve(strict=False)
        except (OSError, ValueError):
            return False
        if path.parent != self.output_directory:
            return False
        mode = args[1] if len(args) > 1 else None
        return isinstance(mode, str) and "x" in mode and "a" not in mode

    def handle_event(self, event: str, args: tuple[Any, ...]) -> None:
        if event == "offline_eval.sentinel.allowed":
            return
        if event == "offline_eval.sentinel.forbidden":
            self._deny(event)
        if event == "exec":
            code = args[0] if args else None
            if getattr(code, "co_filename", None) == self.scanner_filename:
                return
            self._deny(event)
        if event == "open":
            if self._allow_exclusive_output_open(args):
                return
            self._deny(event)
        if event == "import" or event.startswith("import."):
            self._deny(event)
        if event in self._BLOCKED_EVENTS or event.startswith(self._BLOCKED_PREFIXES):
            self._deny(event)
        if event.startswith(("os.exec", "os.spawn", "subprocess", "socket", "ctypes", "winreg")):
            self._deny(event)


def install_runtime_policy(
    policy: RuntimePolicy,
    *,
    os_isolation: OsIsolationAssessment,
) -> IsolationStatus:
    """Install and prove the audit hook is active using a non-operational sentinel."""

    sys.addaudithook(policy.handle_event)
    sentinel_blocked = False
    try:
        sys.audit("offline_eval.sentinel.forbidden")
    except IsolationViolation:
        sentinel_blocked = True
    if not sentinel_blocked:
        raise IsolationViolation(
            "ISOLATION_PREFLIGHT_FAILED",
            "runtime",
            "Runtime sentinel was not blocked",
        )
    sys.audit("offline_eval.sentinel.allowed")
    confirmed = os_isolation.status == "confirmed"
    return IsolationStatus(
        audit_hook_installed=True,
        sentinel_blocked=True,
        os_level_sandbox_confirmed=confirmed,
        # G8: single source is container_contract; do not repeat the literal here.
        security_boundary=(
            cc.CONFIRMED_SECURITY_BOUNDARY
            if confirmed
            else "python_audit_hook_defense_in_depth_not_os_sandbox"
        ),
        os_isolation=os_isolation,
    )


def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


def _attempt_p3_connect() -> ProbeResult:
    # RFC 5737 TEST-NET-1: guaranteed non-routable, never a real destination.
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(2.0)
    try:
        sock.connect(("192.0.2.1", 9))
        return _probe_p3_from_connect_errno(None, timed_out=False, connected=True)
    except socket.timeout:
        return _probe_p3_from_connect_errno(None, timed_out=True, connected=False)
    except OSError as exc:
        return _probe_p3_from_connect_errno(exc.errno, timed_out=False, connected=False)
    finally:
        sock.close()


def _attempt_p7_root_write() -> ProbeResult:
    target = "/offline_eval_ro_probe"
    try:
        with open(target, "x"):
            pass
    except OSError as exc:
        return _probe_p7_from_write_errno(exc.errno, wrote=False)
    else:
        try:
            os.remove(target)
        except OSError:
            pass
        return _probe_p7_from_write_errno(None, wrote=True)


def _run_all_probes(nonce_hex: str) -> tuple[ProbeResult, ...]:
    """Impure: reads real `/proc`, `/sys`, `os.environ`, and attempts a real
    connect + a real write. Only meaningful on Linux; callers gate on `os.name`."""

    def safe(probe_id: str, fn: Any) -> ProbeResult:
        # H6: a probe that reads /proc or /sys text can also hit non-ASCII bytes
        # (UnicodeDecodeError) or a value that parses but has the wrong shape
        # (ValueError, e.g. an int() conversion) -- both must fail-closed exactly
        # like a plain read failure, never propagate as an unhandled exception.
        try:
            return fn()
        except (OSError, UnicodeDecodeError, ValueError):
            return ProbeResult(probe_id, False, "read_error")

    p1 = safe("P1", lambda: _probe_p1_loopback_only(os.listdir("/sys/class/net")))
    p2 = safe("P2", lambda: _probe_p2_no_routes(_read_text("/proc/net/route")))
    p3 = safe("P3", _attempt_p3_connect)
    p4 = safe("P4", lambda: _probe_p4_root_readonly(_read_text("/proc/self/mountinfo")))
    p5 = safe("P5", lambda: _probe_p5_mount_set(_read_text("/proc/self/mountinfo")))
    p6 = safe(
        "P6",
        lambda: _probe_p6_ro_not_writable(
            [dest for dest, rw in cc.REQUIRED_MOUNT_DESTINATIONS.items() if not rw], os.access
        ),
    )
    p7 = safe("P7", _attempt_p7_root_write)
    p8 = safe("P8", lambda: _probe_p8_proc_status(_read_text("/proc/self/status")))
    p9 = safe("P9", lambda: _probe_p9_environ(dict(os.environ), nonce_hex))
    p10 = _probe_p10_from_p9(p9)
    p11 = _probe_p11_python_version(sys.version_info[:3])
    return (p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11)


def assess_os_isolation() -> OsIsolationAssessment:
    """Describe direct-host support honestly; no caller-controlled bypass is accepted."""

    if os.name == "nt":
        return OsIsolationAssessment(
            status="verification_unimplemented",
            method=None,
            reason="windows_host_requires_container_launcher",
        )

    nonce_hex = os.environ.get(cc.NONCE_ENV_NAME, "")
    if cc.NONCE_RE.fullmatch(nonce_hex) is None:
        return OsIsolationAssessment(
            status="verification_failed",
            method="linux_container_self_probe",
            reason="launch_nonce_missing_or_malformed",
        )
    probes = _run_all_probes(nonce_hex)
    failed = [probe for probe in probes if not probe.passed]
    if failed:
        return OsIsolationAssessment(
            status="verification_failed",
            method="linux_container_self_probe",
            reason=f"probe_failed:{failed[0].probe_id}",
            probes=probes,
        )
    return OsIsolationAssessment(
        status="confirmed",
        method="linux_container_self_probe",
        reason="all_probes_passed",
        probes=probes,
    )


def _raise_for_unconfirmed_assessment(assessment: OsIsolationAssessment) -> None:
    if assessment.status == "verification_unimplemented":
        raise IsolationViolation(
            "OS_SANDBOX_VERIFICATION_UNIMPLEMENTED",
            "runtime",
            "OS-level sandbox verification is not implemented; material evidence scan is withheld",
        )
    raise IsolationViolation(
        "OS_SANDBOX_VERIFICATION_FAILED",
        "runtime",
        "OS-level sandbox verification failed; material evidence scan is withheld",
    )


def require_material_scan_isolation(
    assessment: OsIsolationAssessment | None = None,
) -> OsIsolationAssessment:
    # H5: accept an already-computed assessment instead of always calling
    # assess_os_isolation() a second time (design doc §16 H5 -- two calls means
    # two live P3/P7 probe attempts, which can disagree with each other and
    # with what got recorded). `assessment` is NOT a caller-controlled bypass:
    # the only caller (cli.py) must pass the return value of its own single
    # `assess_os_isolation()` call, never a synthesized or injected value.
    if assessment is None:
        assessment = assess_os_isolation()
    if assessment.status != "confirmed":
        _raise_for_unconfirmed_assessment(assessment)
    return assessment


def fixed_output_root(repo_root: Path) -> Path:
    return repo_root.parent / "handoff" / "evaluations" / cc.PHASE3_RUNS_DIRNAME


def fixed_staging_root(repo_root: Path) -> Path:
    return repo_root.parent / "handoff" / "evaluations" / cc.PHASE3_STAGING_DIRNAME


def fixed_launch_root(repo_root: Path) -> Path:
    return repo_root.parent / "handoff" / "evaluations" / cc.PHASE3_LAUNCHES_DIRNAME


# design §17 J2: separate roots for a5_selftest.py's synthetic-bundle runs.
# Deliberately parallel to, and never overlapping with, the three fixed_*_root
# functions above -- the scorer never reads these paths, so a selftest run
# cannot be scored or mistaken for a real one.
def fixed_selftest_output_root(repo_root: Path) -> Path:
    return repo_root.parent / "handoff" / "evaluations" / cc.PHASE3_SELFTEST_RUNS_DIRNAME


def fixed_selftest_staging_root(repo_root: Path) -> Path:
    return repo_root.parent / "handoff" / "evaluations" / cc.PHASE3_SELFTEST_STAGING_DIRNAME


def fixed_selftest_launch_root(repo_root: Path) -> Path:
    return repo_root.parent / "handoff" / "evaluations" / cc.PHASE3_SELFTEST_LAUNCHES_DIRNAME


def prepare_new_output_directory(output_path: Path, output_root: Path) -> Path:
    """Create one direct child only; never overwrite, append, delete, or follow links."""

    if not output_path.is_absolute() or not output_root.is_absolute():
        raise ContractViolation("PATH_REJECTED", "output", "Output paths must be absolute")
    if not RUN_NAME_RE.fullmatch(output_path.name):
        raise ContractViolation("PATH_REJECTED", "output", "Output folder name is not allowed")
    base = output_root.parent
    if not base.exists() or not base.is_dir() or path_has_reparse_component(base):
        raise ContractViolation("PATH_REJECTED", "output_root", "Fixed output parent is unavailable or unsafe")
    root = output_root.resolve(strict=False)
    if path_has_reparse_component(root):
        raise ContractViolation("PATH_REJECTED", "output_root", "Output root contains a reparse point")
    requested_parent = output_path.parent.resolve(strict=False)
    if requested_parent != root:
        raise ContractViolation("PATH_REJECTED", "output", "Output must be a direct child of the fixed root")
    if not output_root.exists():
        output_root.mkdir()
    if not output_root.is_dir() or path_has_reparse_component(output_root):
        raise ContractViolation("PATH_REJECTED", "output_root", "Output root is unsafe")
    if output_path.exists():
        raise ContractViolation("PATH_REJECTED", "output", "Output folder already exists")
    output_path.mkdir()
    if path_has_reparse_component(output_path):
        raise ContractViolation("PATH_REJECTED", "output", "Created output folder is unsafe")
    return output_path.resolve(strict=True)
