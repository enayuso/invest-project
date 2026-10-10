"""Single source of truth for the OS-isolation container contract (design §3, §10.2, §14).

Every fixed value the host launcher, the container's own self-probe (isolation.py),
and the scorer's attestation cross-check (scoring.py) must agree on lives here, once,
as a pure constant or a pure function. Nothing in this module touches the filesystem,
the network, or a subprocess.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any


# Pinned 2026-09-22 by the user via `docker pull python:3.12.13-slim-bookworm` +
# `docker image inspect` (design doc §3.1). Re-pin explicitly if this ever changes;
# do not follow tags.
#
# Docker's reference grammar allows either NAME:TAG or NAME@DIGEST, never both
# combined, for `image inspect` / `create` / `run` lookups -- confirmed empirically:
# `python:3.12.13-slim-bookworm@sha256:...` returns "No such image", while
# `python@sha256:...` resolves. IMAGE_TAG is kept only for human-readable logging.
#
# design §14.0: the short `python@sha256:...` reference is an unreliable lookup (it
# resolved once, then failed "No such image" on a later attempt). The fully
# normalized `docker.io/library/python@sha256:...` reference is what resolves
# reliably (F1). IMAGE_ID is the image ID docker reports for this pinned image
# (`{{.Id}}` / inspect `.Image`); with the containerd snapshotter it is the same
# value as IMAGE_DIGEST, but it is pinned as its own constant because it is what
# `docker inspect` actually reports, not the repo digest.
IMAGE_TAG = "python:3.12.13-slim-bookworm"
IMAGE_DIGEST = "sha256:4766d8b510c428e595d74b9cc5bbb2fae8e26316fffb4adc89908d79aacd58a2"
IMAGE_REF = f"docker.io/library/python@{IMAGE_DIGEST}"
IMAGE_ID = IMAGE_DIGEST
PYTHON_VERSION = (3, 12, 13)
PYTHON_VERSION_STRING = ".".join(str(part) for part in PYTHON_VERSION)

NONCE_ENV_NAME = "OFFLINE_EVAL_LAUNCH_NONCE"
NONCE_RE = re.compile(r"[0-9a-f]{64}\Z")  # 32 random bytes, hex-encoded

ATTESTATION_SCHEMA = "offline-eval.launch-attestation.v1"
ATTESTATION_FILENAME = "launch-attestation.json"

CONTAINER_HOSTNAME = "offline-eval"

# Config.Env of the pinned base image, confirmed read-only via
# `docker image inspect --format "{{json .Config.Env}}" <IMAGE_REF>` on 2026-09-22.
# If IMAGE_REF is ever re-pinned, this set must be re-confirmed the same way.
BASE_IMAGE_ENV_ALLOWLIST = frozenset(
    {
        "PATH=/usr/local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "LANG=C.UTF-8",
        "GPG_KEY=7169605F62C751356D054A26A821E680E5FA6305",
        "PYTHON_VERSION=3.12.13",
        "PYTHON_SHA256=c08bc65a81971c1dd5783182826503369466c7e67374d1646519adf05207b684",
    }
)

# design §14.1 F2: Docker/runc inject HOSTNAME (and HOME, when unset) into the
# process environment; these never appear in `docker inspect`'s Config.Env. They
# are optional (may or may not be present) but, if present, must have exactly one
# of these fixed values -- nothing else is tolerated under these keys.
RUNTIME_OPTIONAL_ENV_KEYS = frozenset({"HOSTNAME", "HOME"})
RUNTIME_OPTIONAL_ENV_VALUES = frozenset(
    {
        f"HOSTNAME={CONTAINER_HOSTNAME}",
        "HOME=/nonexistent",
        "HOME=/",
    }
)

CONTAINER_USER = "65534:65534"
CONTAINER_UID = 65534

WORKDIR = "/work"
REPO_MOUNT = "/work/repo/scripts/offline_eval"
GUARD_MOUNT = "/work/repo/app/agents/session_close_guard.py"
PHASE2_MOUNT = "/work/handoff/evaluations/2026-09-22-phase2"
PHASE3A_MOUNT = "/work/handoff/evaluations/2026-09-22-phase3a"
OUTPUT_MOUNT = "/work/handoff/evaluations/phase3-runs"
PROFILE_FILENAME = "scope-v1.json"

# Host-side names, relative to repo_root / repo_root.parent (see bundle.py's own
# FIXED_SCANNER_REFERENCE for the same convention). The launcher resolves these,
# once, into absolute host paths; nothing else on the host is allowed to invent a
# different mount source for this tool.
GUARD_SOURCE_RELATIVE = "app/agents/session_close_guard.py"
TOOL_SOURCE_RELATIVE = "scripts/offline_eval"
PHASE2_BUNDLE_DIRNAME = "2026-09-22-phase2"
PHASE3A_PROFILE_DIRNAME = "2026-09-22-phase3a"

# Destination (container path) -> expected read-write flag. Fixed set, fixed order
# for argument construction; membership is what matters for inspect validation.
REQUIRED_MOUNT_DESTINATIONS: dict[str, bool] = {
    REPO_MOUNT: False,
    GUARD_MOUNT: False,
    PHASE2_MOUNT: False,
    PHASE3A_MOUNT: False,
    OUTPUT_MOUNT: True,
}

# design §14.1 F3: destinations Docker Desktop (WSL2 backend) injects into every
# container's mount namespace (visible in /proc/self/mountinfo) even with
# --network none, so the container has a usable hostname and can resolve
# `localhost`. Tolerated *only* in the kernel's own mountinfo view (P5). They must
# never appear as an explicit `docker inspect` Mounts bind entry (F4) -- if they do,
# that is a real user-added bind, not Docker's own injected file.
DOCKER_INJECTED_MOUNT_PREFIXES = ("/etc/hosts", "/etc/hostname", "/etc/resolv.conf")

# design §14.1 F3: kernel-virtual filesystem types tolerated anywhere under
# /proc, /sys, or /dev in mountinfo (including those roots themselves).
PSEUDO_FS_TYPES = frozenset(
    {"proc", "sysfs", "tmpfs", "devtmpfs", "devpts", "mqueue", "cgroup", "cgroup2"}
)

# design §14.1 F9: kernel fallback tunnel devices that can exist in a --network
# none namespace even though no interface was explicitly configured. Tolerated by
# P1 alongside `lo`; anything else is a real network interface and fails.
FALLBACK_TUNNEL_INTERFACES = frozenset(
    {"tunl0", "ip6tnl0", "sit0", "gre0", "gretap0", "erspan0", "ip6gre0", "ip_vti0", "ip6_vti0"}
)

RESOURCE_LIMITS = {"pids_limit": "64", "memory": "1g", "cpus": "2"}
EXPECTED_PIDS_LIMIT = int(RESOURCE_LIMITS["pids_limit"])
# design: memory 1g => bytes, cpus 2 => NanoCpus, as `docker inspect` reports them.
EXPECTED_MEMORY_BYTES = 1073741824
EXPECTED_NANO_CPUS = 2000000000

RUN_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,80}\Z")

REQUIRED_PROBE_IDS = frozenset({f"P{i}" for i in range(1, 12)})
CONFIRMED_SECURITY_BOUNDARY = "linux_container_probes_confirmed_plus_python_audit_hook"

# design §14.1 F13 / §15 G8: single source for the fixed evidence-root directory
# names, so isolation.py's fixed_*_root helpers do not repeat these literals.
PHASE3_RUNS_DIRNAME = "phase3-runs"
PHASE3_STAGING_DIRNAME = "phase3-staging"
PHASE3_LAUNCHES_DIRNAME = "phase3-launches"

# design §17 J2: a5_selftest.py's synthetic-bundle runs land under a completely
# separate set of root directories, never these. The scorer's fixed roots
# (isolation.fixed_output_root / fixed_launch_root) only ever point at the
# PHASE3_*_DIRNAME roots above, so a selftest run is structurally impossible to
# score or confuse with a real run -- no extra scoring-side check is needed.
PHASE3_SELFTEST_RUNS_DIRNAME = "phase3-selftest-runs"
PHASE3_SELFTEST_STAGING_DIRNAME = "phase3-selftest-staging"
PHASE3_SELFTEST_LAUNCHES_DIRNAME = "phase3-selftest-launches"

# design §15 G3: single source for the tool source files whose hashes are
# cross-checked across manifest, findings, attestation, and the scorer's own
# running copy (report.collect_tool_hashes / scoring._current_tool_hashes).
TOOL_SOURCE_FILES = (
    "bundle.py",
    "cli.py",
    "container_contract.py",
    "contracts.py",
    "isolation.py",
    "launcher.py",
    "report.py",
    "scoring.py",
    "session_adapter.py",
)

# design §11.4 / §14.1 F15: fixed limitation statements, carried verbatim into the
# run manifest (`limitations`) and the README, so no downstream reader can miss them.
LIMITATIONS = (
    "This does not defend against a deliberate kernel exploit escape.",
    "The launch attestation is an integrity binding, not a cryptographic signature.",
    "The Python audit hook remains a defense-in-depth layer, not an OS-level sandbox.",
    "This tool makes no claim about real-evidence accuracy or profitability.",
    # design §16 H3: the probe count is 10 independent self-checks plus one
    # (P10) derived from P9's own result, not 11 independent checks.
    "The self-check count is 10 independent probes plus P10, which is derived "
    "from P9's result rather than an independent check.",
)


@dataclass(frozen=True)
class HostMounts:
    """Host-side absolute source paths for the 5 fixed bind mounts."""

    tool_dir: Path
    guard_file: Path
    phase2_dir: Path
    phase3a_dir: Path
    staging_dir: Path

    def as_tuple(self) -> tuple[tuple[Path, str, bool], ...]:
        return (
            (self.tool_dir, REPO_MOUNT, False),
            (self.guard_file, GUARD_MOUNT, False),
            (self.phase2_dir, PHASE2_MOUNT, False),
            (self.phase3a_dir, PHASE3A_MOUNT, False),
            (self.staging_dir, OUTPUT_MOUNT, True),
        )


def _mount_arg(source: Path, target: str, *, readonly: bool) -> str:
    # `type=bind,source=...,target=...` avoids the classic `-v host:container:mode`
    # colon-delimited syntax, which breaks on Windows drive letters (`D:\...`).
    spec = f"type=bind,source={source},target={target}"
    if readonly:
        spec += ",readonly"
    return spec


def build_scan_command(*, run_name: str) -> list[str]:
    if RUN_NAME_RE.fullmatch(run_name) is None:
        raise ValueError("run_name format is not allowed")
    return [
        "python",
        "-I",
        "-S",
        "-B",
        f"{REPO_MOUNT}/cli.py",
        "scan",
        "--bundle",
        PHASE2_MOUNT,
        "--profile",
        f"{PHASE3A_MOUNT}/{PROFILE_FILENAME}",
        "--out",
        f"{OUTPUT_MOUNT}/{run_name}",
    ]


def build_docker_args(*, mounts: HostMounts, nonce_hex: str, run_name: str) -> list[str]:
    """Pure: the exact `docker create ...` argv (excluding the `docker` binary itself).

    Every value here is either a fixed constant from this module or a caller-supplied
    absolute host path / nonce / run name, each validated before use. This is the only
    place that is allowed to construct a docker argument vector for this tool.
    """

    if NONCE_RE.fullmatch(nonce_hex) is None:
        raise ValueError("nonce format is not allowed")
    for path in (mounts.tool_dir, mounts.guard_file, mounts.phase2_dir, mounts.phase3a_dir, mounts.staging_dir):
        if not path.is_absolute():
            raise ValueError("mount source paths must be absolute")

    command = build_scan_command(run_name=run_name)

    args: list[str] = ["create"]
    args += ["--network", "none"]
    args += ["--read-only"]
    args += ["--cap-drop", "ALL"]
    args += ["--security-opt", "no-new-privileges:true"]
    args += ["--user", CONTAINER_USER]
    args += ["--hostname", CONTAINER_HOSTNAME]
    args += ["--pids-limit", RESOURCE_LIMITS["pids_limit"]]
    args += ["--memory", RESOURCE_LIMITS["memory"]]
    args += ["--cpus", RESOURCE_LIMITS["cpus"]]
    args += ["--env", f"{NONCE_ENV_NAME}={nonce_hex}"]
    args += ["--log-driver", "none"]
    args += ["--workdir", WORKDIR]
    for source, target, rw in mounts.as_tuple():
        args += ["--mount", _mount_arg(source, target, readonly=not rw)]
    args += [IMAGE_REF]
    args += command
    return args


def env_list_matches_contract(env_entries: list[str], nonce_hex: str, *, allow_runtime_keys: bool) -> bool:
    """Pure: True iff `env_entries` ("KEY=VALUE" strings, from either `docker
    inspect`'s Config.Env or a flattened os.environ) is exactly the base-image
    allowlist, plus the one nonce entry, plus (only when `allow_runtime_keys`)
    zero or more of the fixed runtime keys/values in RUNTIME_OPTIONAL_ENV_VALUES.

    design §15 G5: `docker inspect`'s Config.Env (the requested configuration)
    never legitimately contains HOSTNAME/HOME -- those are injected by
    Docker/runc into the *process* environment only. So `allow_runtime_keys`
    is False for inspect validation and True only for P9's process-env probe.
    """

    nonce_entries = [entry for entry in env_entries if entry.startswith(f"{NONCE_ENV_NAME}=")]
    if len(nonce_entries) != 1 or nonce_entries[0] != f"{NONCE_ENV_NAME}={nonce_hex}":
        return False
    remaining = [entry for entry in env_entries if entry not in nonce_entries]
    if len(remaining) != len(set(remaining)):
        return False
    for entry in remaining:
        key = entry.split("=", 1)[0]
        if allow_runtime_keys and key in RUNTIME_OPTIONAL_ENV_KEYS:
            if entry not in RUNTIME_OPTIONAL_ENV_VALUES:
                return False
        elif entry not in BASE_IMAGE_ENV_ALLOWLIST:
            return False
    base_present = {entry for entry in remaining if entry in BASE_IMAGE_ENV_ALLOWLIST}
    return base_present == set(BASE_IMAGE_ENV_ALLOWLIST)


def _normalize_mount_source(source: str) -> str:
    """Pure: normalize a host bind-mount source for equality comparison across
    Windows path spellings and Docker Desktop's WSL2 host-mount prefix
    (design §14.1 F5). Confirmed against real `docker inspect` output only at A5.

    design §15 G9: the drive letter is preserved (not stripped), so `C:\\X` and
    `D:\\X` normalize to distinct values `/c/X` and `/d/X` -- dropping the
    drive letter would make sources on different drives compare equal."""

    normalized = source.strip().lower().replace("\\", "/")
    desktop_prefix = "/run/desktop/mnt/host/"
    if normalized.startswith(desktop_prefix):
        rest = normalized[len(desktop_prefix):]
        normalized = rest if rest.startswith("/") else f"/{rest}"
    else:
        drive_match = re.match(r"^([a-z]):(/.*)?$", normalized)
        if drive_match:
            normalized = f"/{drive_match.group(1)}{drive_match.group(2) or '/'}"
    return normalized


def inspect_subset(full: dict[str, Any]) -> dict[str, Any]:
    """Pure: reduce a full `docker inspect` object to just the fields
    `validate_inspect` checks, dropping Mounts' Source (host paths) entirely
    (design §14.1 F6). This is the only shape the launcher is allowed to persist
    into `launch-attestation.json`."""

    host_config = full.get("HostConfig") if isinstance(full.get("HostConfig"), dict) else {}
    config = full.get("Config") if isinstance(full.get("Config"), dict) else {}
    raw_mounts = full.get("Mounts") if isinstance(full.get("Mounts"), list) else []
    mounts = [
        {"Type": mount.get("Type"), "Destination": mount.get("Destination"), "RW": mount.get("RW")}
        for mount in raw_mounts
        if isinstance(mount, dict)
    ]
    host_config_keys = (
        "NetworkMode", "ReadonlyRootfs", "Privileged", "CapDrop", "CapAdd", "SecurityOpt",
        "Devices", "PidsLimit", "PidMode", "IpcMode", "UTSMode", "UsernsMode",
        "Tmpfs", "Binds", "VolumesFrom", "Memory", "NanoCpus", "LogConfig", "GroupAdd",
    )
    config_keys = ("User", "Env", "Hostname", "Cmd", "Entrypoint")
    return {
        "Image": full.get("Image"),
        "HostConfig": {key: host_config.get(key) for key in host_config_keys},
        "Config": {key: config.get(key) for key in config_keys},
        "Mounts": mounts,
    }


def validate_inspect(
    inspect_subset: dict[str, Any],
    *,
    nonce_hex: str,
    run_name: str,
    expected_sources: dict[str, Any] | None = None,
) -> list[str]:
    """Pure: check a `docker inspect` result (or a stored copy of one) against the
    contract. Returns a sorted list of violation codes; an empty list means the
    contract is satisfied. Never raises on malformed input -- malformed input is
    itself a violation.

    `expected_sources` (Destination -> host path) is checked only when given
    (the launcher's own `docker inspect`, L5); a stored attestation subset has no
    Source (F6), so the scorer's replay (S7) passes None and skips that check.

    This only checks the fields listed below. It does not, and cannot, prove the
    absence of a kernel escape; it proves the *requested* configuration was applied.
    """

    violations: set[str] = set()
    if NONCE_RE.fullmatch(nonce_hex) is None:
        violations.add("NONCE_ARGUMENT_INVALID")
        return sorted(violations)

    try:
        expected_cmd: list[str] | None = build_scan_command(run_name=run_name)
    except ValueError:
        violations.add("RUN_NAME_INVALID")
        expected_cmd = None

    host_config = inspect_subset.get("HostConfig")
    config = inspect_subset.get("Config")
    if not isinstance(host_config, dict):
        violations.add("INSPECT_MISSING_HOST_CONFIG")
    if not isinstance(config, dict):
        violations.add("INSPECT_MISSING_CONFIG")
    if violations:
        return sorted(violations)

    if host_config.get("NetworkMode") != "none":
        violations.add("NETWORK_MODE_NOT_NONE")
    if host_config.get("ReadonlyRootfs") is not True:
        violations.add("ROOTFS_NOT_READONLY")
    if host_config.get("Privileged") is not False:
        violations.add("PRIVILEGED")

    cap_drop = host_config.get("CapDrop")
    if not (isinstance(cap_drop, list) and {str(item).upper() for item in cap_drop} == {"ALL"}):
        violations.add("CAP_DROP_NOT_ALL")
    cap_add = host_config.get("CapAdd")
    if cap_add not in (None, [], [""]):
        violations.add("CAP_ADD_NOT_EMPTY")

    # design §15 G6: exact-match, not "contains the right entry somewhere" --
    # this alone rejects every unconfined-seccomp spelling (`seccomp=unconfined`,
    # `seccomp:unconfined`), a user seccomp profile, `systempaths=unconfined`,
    # and any other addition, without needing a separate check per spelling.
    if host_config.get("SecurityOpt") != ["no-new-privileges:true"]:
        violations.add("SECURITY_OPT_MISMATCH")

    if host_config.get("GroupAdd") not in (None, []):
        violations.add("GROUP_ADD_NOT_EMPTY")

    devices = host_config.get("Devices")
    if devices not in (None, []):
        violations.add("DEVICES_NOT_EMPTY")

    pids_limit = host_config.get("PidsLimit")
    if pids_limit not in (EXPECTED_PIDS_LIMIT, str(EXPECTED_PIDS_LIMIT)):
        violations.add("PIDS_LIMIT_MISMATCH")

    for mode_key in ("PidMode", "IpcMode", "UTSMode", "UsernsMode"):
        mode_value = host_config.get(mode_key)
        if mode_value not in (None, ""):
            text = str(mode_value)
            if text == "host" or text.startswith("container:"):
                violations.add(f"NAMESPACE_MODE_REJECTED:{mode_key}")

    if host_config.get("Tmpfs") not in (None, {}):
        violations.add("TMPFS_NOT_EMPTY")
    if host_config.get("Binds") not in (None, []):
        violations.add("BINDS_NOT_EMPTY")
    if host_config.get("VolumesFrom") not in (None, []):
        violations.add("VOLUMES_FROM_NOT_EMPTY")

    if host_config.get("Memory") != EXPECTED_MEMORY_BYTES:
        violations.add("MEMORY_MISMATCH")
    if host_config.get("NanoCpus") != EXPECTED_NANO_CPUS:
        violations.add("NANO_CPUS_MISMATCH")

    log_config = host_config.get("LogConfig")
    if not isinstance(log_config, dict) or log_config.get("Type") != "none":
        violations.add("LOG_CONFIG_NOT_NONE")

    if config.get("User") != CONTAINER_USER:
        violations.add("USER_NOT_PINNED")
    if config.get("Hostname") != CONTAINER_HOSTNAME:
        violations.add("HOSTNAME_MISMATCH")
    if config.get("Entrypoint") not in (None, []):
        violations.add("ENTRYPOINT_NOT_EMPTY")
    if expected_cmd is not None and config.get("Cmd") != expected_cmd:
        violations.add("CMD_MISMATCH")

    env_list = config.get("Env")
    if not isinstance(env_list, list) or not all(isinstance(item, str) for item in env_list):
        violations.add("ENV_MISSING")
    # G5: inspect's Config.Env is the *requested* configuration; HOSTNAME/HOME
    # are injected only into the container's process environment (P9), never
    # legitimately reported here, so runtime keys are not tolerated in this check.
    elif not env_list_matches_contract(env_list, nonce_hex, allow_runtime_keys=False):
        violations.add("ENV_ALLOWLIST_MISMATCH")

    mounts = inspect_subset.get("Mounts")
    if not isinstance(mounts, list):
        violations.add("MOUNTS_MISSING")
    else:
        seen: dict[str, dict[str, Any]] = {}
        for mount in mounts:
            if not isinstance(mount, dict):
                violations.add("MOUNT_ENTRY_INVALID")
                continue
            if mount.get("Type") != "bind":
                violations.add("MOUNT_TYPE_NOT_BIND")
                continue
            destination = mount.get("Destination")
            # G4: a non-string Destination (e.g. a list) must never reach the
            # `in <dict>` membership test below -- that would raise TypeError
            # (unhashable key) instead of returning a violation code.
            if not isinstance(destination, str):
                violations.add("MOUNT_ENTRY_INVALID")
                continue
            if destination not in REQUIRED_MOUNT_DESTINATIONS:
                # No exceptions here (design §14.1 F4): Docker reports its own
                # injected /etc/* files via HostsPath etc., not via Mounts. A
                # Mounts entry outside the fixed 5 is a real, unrequested bind.
                violations.add("MOUNT_UNEXPECTED_DESTINATION")
                continue
            if destination in seen:
                violations.add("MOUNT_DUPLICATE_DESTINATION")
                continue
            seen[destination] = mount
        missing = set(REQUIRED_MOUNT_DESTINATIONS) - set(seen)
        if missing:
            violations.add("MOUNT_SET_MISMATCH")
        else:
            for destination, expected_rw in REQUIRED_MOUNT_DESTINATIONS.items():
                mount = seen[destination]
                if mount.get("RW") is not expected_rw:
                    violations.add(f"MOUNT_RW_MISMATCH:{destination}")
                if expected_sources is not None:
                    expected_source = expected_sources.get(destination)
                    actual_source = mount.get("Source")
                    if (
                        expected_source is None
                        or not isinstance(actual_source, str)
                        or _normalize_mount_source(actual_source) != _normalize_mount_source(str(expected_source))
                    ):
                        violations.add(f"MOUNT_SOURCE_MISMATCH:{destination}")

    image = inspect_subset.get("Image")
    if image != IMAGE_ID:
        violations.add("IMAGE_REF_MISMATCH")

    return sorted(violations)


def validate_isolation_block(isolation_block: Any) -> list[str]:
    """Pure: check a manifest's `isolation` block reports confirmed OS isolation
    with the exact P1-P11 probe set, each individually passed (design §14.1 F8,
    used by both the scorer's S1 and the launcher's L8 -- single source, F13)."""

    if not isinstance(isolation_block, dict):
        return ["ISOLATION_BLOCK_INVALID"]
    violations: list[str] = []
    if isolation_block.get("os_level_sandbox_confirmed") is not True:
        violations.append("SANDBOX_NOT_CONFIRMED")
    if isolation_block.get("security_boundary") != CONFIRMED_SECURITY_BOUNDARY:
        violations.append("SECURITY_BOUNDARY_MISMATCH")
    os_isolation = isolation_block.get("os_isolation")
    if not isinstance(os_isolation, dict):
        return violations + ["OS_ISOLATION_BLOCK_INVALID"]
    if os_isolation.get("status") != "confirmed":
        violations.append("STATUS_NOT_CONFIRMED")
    probes = os_isolation.get("probes")
    if not isinstance(probes, list):
        return violations + ["PROBES_INVALID"]
    seen: dict[str, Any] = {}
    for probe in probes:
        if not isinstance(probe, dict):
            violations.append("PROBE_ENTRY_INVALID")
            continue
        probe_id = probe.get("id")
        # G4: a non-string id (e.g. a list) must never reach the `in <dict>`
        # membership test below -- that would raise TypeError instead of
        # returning a violation code.
        if not isinstance(probe_id, str):
            violations.append("PROBE_ENTRY_INVALID")
            continue
        if probe_id in seen:
            violations.append("PROBE_DUPLICATE_ID")
            continue
        seen[probe_id] = probe.get("passed")
    if set(seen) != REQUIRED_PROBE_IDS:
        violations.append("PROBE_SET_MISMATCH")
    for probe_id, passed in seen.items():
        if passed is not True:
            violations.append(f"PROBE_NOT_PASSED:{probe_id}")
    return violations
