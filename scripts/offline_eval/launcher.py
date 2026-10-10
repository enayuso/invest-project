"""Host-side launcher for the OS-isolation container (design doc §5, L1-L10).

Runs on Windows. Never reads bundle/evidence content -- the only file this process
opens for content (not just hashing) is a `docker` CLI invocation's own JSON output
and, on success, the produced run-manifest.json's raw bytes (for hashing only).

No `shell=True`, ever. Every docker invocation is a fixed argument vector built by
`container_contract.build_docker_args` or a small fixed list here. The only value a
caller supplies is `--run-name`.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import secrets
import shutil
import subprocess
import sys
import time
from typing import Any


TOOL_DIR = Path(__file__).resolve().parent
if str(TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(TOOL_DIR))

import container_contract as cc  # noqa: E402
from contracts import ContractViolation, read_limited_file, sha256_bytes  # noqa: E402
from isolation import (  # noqa: E402
    fixed_launch_root,
    fixed_output_root,
    fixed_staging_root,
    prepare_new_output_directory,
)
from report import collect_tool_hashes  # noqa: E402


DOCKER_TIMEOUT_SECONDS = 600
CANARY_ENV_NAME_PREFIX = "OFFLINE_EVAL_HOST_CANARY_"
REQUIRED_RUN_FILES = ("findings.json", "summary.md", "run-manifest.json")


class LauncherFailure(Exception):
    def __init__(
        self, exit_code: int, code: str, description: str, *, container_report: dict[str, Any] | None = None
    ) -> None:
        super().__init__(description)
        self.exit_code = exit_code
        self.code = code
        self.description = description
        # cli.py's own stdout is already a vetted, evidence-free safe JSON object
        # (design doc: "raw evidence text, excerpts... never returned to stdout").
        # Surfacing it here is what makes a failed run diagnosable without a second
        # Docker invocation -- see design doc §13, point 3.
        self.container_report = container_report


def _write(payload: dict[str, Any]) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    sys.stdout.write(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")


def _fail(
    exit_code: int, code: str, description: str, *, container_report: dict[str, Any] | None = None
) -> LauncherFailure:
    return LauncherFailure(exit_code, code, description, container_report=container_report)


def _parse_container_report(stdout_text: str) -> dict[str, Any] | None:
    if not stdout_text:
        return None
    try:
        parsed = json.loads(stdout_text)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _resolve_docker_binary() -> str:
    docker = shutil.which("docker")
    if docker is None:
        raise _fail(4, "DOCKER_NOT_FOUND", "docker CLI was not found on PATH")
    return docker


def _subprocess_env(canary_name: str, canary_value: str) -> dict[str, str]:
    # Deliberately minimal: only what the Windows process loader and the docker CLI
    # itself need to run, plus one canary variable. This is *not* the host's real
    # environment -- it is used to prove that nothing beyond this minimal set can
    # possibly reach the container (design doc P10).
    env = {
        "SystemRoot": os.environ.get("SystemRoot", r"C:\Windows"),
        "USERPROFILE": os.environ.get("USERPROFILE", ""),
        "PATH": os.environ.get("PATH", ""),
        canary_name: canary_value,
    }
    return {key: value for key, value in env.items() if value}


def _run_docker(
    docker_binary: str, args: list[str], *, env: dict[str, str], input_text: str | None = None
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            [docker_binary, *args],
            capture_output=True,
            text=True,
            shell=False,
            env=env,
            input=input_text,
            timeout=DOCKER_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise _fail(3, "DOCKER_TIMEOUT", "A docker invocation exceeded its timeout") from exc
    except OSError as exc:
        raise _fail(3, "DOCKER_INVOCATION_FAILED", "Failed to invoke the docker CLI") from exc


def _l1_check_run_name(run_name: str, repo_root: Path) -> tuple[Path, Path, Path]:
    if cc.RUN_NAME_RE.fullmatch(run_name) is None:
        raise _fail(2, "USAGE_ERROR", "Run name format is not allowed")
    runs_root = fixed_output_root(repo_root)
    staging_root = fixed_staging_root(repo_root)
    launch_root = fixed_launch_root(repo_root)
    if (runs_root / run_name).exists():
        raise _fail(2, "RUN_ALREADY_EXISTS", "A completed run with this name already exists")
    if (staging_root / run_name).exists():
        raise _fail(2, "RUN_ALREADY_EXISTS", "A staged run with this name already exists")
    # G2 (N2): reject before any docker call if a launch folder from an earlier,
    # never-completed attempt is still there -- otherwise the container starts,
    # a real scan happens, and only then does attestation-writing fail because
    # the launch folder already exists (wasting a real-evidence scan and
    # leaving a run nothing can score).
    if (launch_root / run_name).exists():
        raise _fail(2, "RUN_ALREADY_EXISTS", "A launch record with this name already exists")
    return runs_root, staging_root, launch_root


def _check_no_bytecode(tool_dir: Path) -> None:
    # F14: a stray .pyc/__pycache__ under the tool folder could let the container
    # run something other than the hash-pinned .py source. Refuse to launch;
    # do not delete it automatically -- a human confirms what happened.
    for path in tool_dir.rglob("*"):
        if path.name == "__pycache__" and path.is_dir():
            raise _fail(2, "BYTECODE_PRESENT", "Compiled bytecode artifacts were found under the tool directory")
        if path.is_file() and path.suffix == ".pyc":
            raise _fail(2, "BYTECODE_PRESENT", "Compiled bytecode artifacts were found under the tool directory")


def _l2_tool_and_guard_hashes(repo_root: Path) -> tuple[dict[str, str], Path, str]:
    _check_no_bytecode(TOOL_DIR)
    tool_hashes = collect_tool_hashes(TOOL_DIR)
    guard_path = repo_root / Path(*cc.GUARD_SOURCE_RELATIVE.split("/"))
    if not guard_path.is_file():
        raise _fail(2, "GUARD_SOURCE_MISSING", "Guard source file is missing")
    # G1: hash the guard source from the host side, before it is ever mounted
    # into the container, so the scorer can cross-check it against what the
    # container itself hashed after reading the (possibly re-mounted) file.
    guard_hash = sha256_bytes(guard_path.read_bytes())
    return tool_hashes, guard_path, guard_hash


def _l3_check_image_digest(docker_binary: str, env: dict[str, str]) -> None:
    # F1: compare the image ID docker actually reports, not a repo-digest
    # substring lookup (that lookup proved unreliable -- design §14.0).
    result = _run_docker(
        docker_binary,
        ["image", "inspect", "--format", "{{.Id}}", cc.IMAGE_REF],
        env=env,
    )
    if result.returncode != 0:
        raise _fail(4, "IMAGE_NOT_PRESENT", "The pinned image is not present locally; it was not pulled automatically")
    image_id = result.stdout.strip()
    if image_id != cc.IMAGE_ID:
        raise _fail(4, "IMAGE_DIGEST_MISMATCH", "The local image digest does not match the pinned contract")


def _l5_create_and_validate(
    docker_binary: str, env: dict[str, str], mounts: cc.HostMounts, nonce_hex: str, run_name: str
) -> tuple[str, dict[str, Any]]:
    args = cc.build_docker_args(mounts=mounts, nonce_hex=nonce_hex, run_name=run_name)
    created = _run_docker(docker_binary, args, env=env)
    if created.returncode != 0:
        raise _fail(4, "CONTAINER_CREATE_FAILED", "docker create failed")
    container_id = created.stdout.strip()
    # F12: any exception from here on (including a timeout while inspecting)
    # must still remove the container before propagating.
    try:
        inspected = _run_docker(docker_binary, ["inspect", container_id], env=env)
        if inspected.returncode != 0:
            raise _fail(4, "CONTAINER_INSPECT_FAILED", "docker inspect failed before start")
        try:
            inspect_list = json.loads(inspected.stdout)
            inspect_subset = inspect_list[0]
        except (json.JSONDecodeError, IndexError, TypeError) as exc:
            raise _fail(4, "CONTAINER_INSPECT_UNPARSEABLE", "docker inspect output was not usable") from exc
        expected_sources = {destination: str(source) for source, destination, _rw in mounts.as_tuple()}
        violations = cc.validate_inspect(
            inspect_subset, nonce_hex=nonce_hex, run_name=run_name, expected_sources=expected_sources
        )
        if violations:
            raise _fail(4, "CONTAINER_CONFIG_REJECTED", "Container configuration does not satisfy the isolation contract")
    except Exception:
        _run_docker(docker_binary, ["rm", "-f", container_id], env=env)
        raise
    return container_id, inspect_subset


def _l6_l7_start_and_remove(docker_binary: str, env: dict[str, str], container_id: str) -> tuple[int, str]:
    started = _run_docker(docker_binary, ["start", "-a", container_id], env=env)
    exit_code = started.returncode
    removed = _run_docker(docker_binary, ["rm", container_id], env=env)
    if removed.returncode != 0:
        # Non-fatal for the run itself, but must not be silent.
        sys.stderr.write("docker rm failed after run; manual cleanup may be required\n")
    return exit_code, started.stdout


def _l8_validate_staged_output(staging_run_dir: Path, run_name: str, nonce_hex: str) -> tuple[dict[str, Any], Path]:
    # F11: defensive against a malformed or tampered manifest, and against
    # anything besides exactly the <RUN> folder appearing next to it.
    try:
        staged_entries = sorted(entry.name for entry in staging_run_dir.iterdir())
    except OSError as exc:
        raise _fail(4, "STAGED_OUTPUT_MISSING", "The container did not produce an output folder") from exc
    if staged_entries != [run_name]:
        raise _fail(4, "STAGED_OUTPUT_UNEXPECTED_ENTRY", "The staging root does not contain exactly the run folder")

    staged_run_dir = staging_run_dir / run_name
    if not staged_run_dir.is_dir():
        raise _fail(4, "STAGED_OUTPUT_MISSING", "The container did not produce an output folder")
    actual_files = sorted(entry.name for entry in staged_run_dir.iterdir())
    if actual_files != sorted(REQUIRED_RUN_FILES):
        raise _fail(4, "STAGED_OUTPUT_INCOMPLETE", "The staged run folder does not contain exactly the expected files")

    # H4: bounded read, same limit the scorer applies -- an unbounded read_bytes()
    # would turn an oversized staged manifest into a memory/latency problem instead
    # of a clean, fail-closed rejection.
    try:
        manifest_raw = read_limited_file(staged_run_dir / "run-manifest.json", "run-manifest.json")
    except ContractViolation as exc:
        raise _fail(4, "STAGED_MANIFEST_TOO_LARGE", "The staged run manifest exceeds the size limit") from exc
    try:
        manifest = json.loads(manifest_raw)
    except json.JSONDecodeError as exc:
        raise _fail(4, "STAGED_MANIFEST_UNPARSEABLE", "The staged run manifest was not valid JSON") from exc
    if not isinstance(manifest, dict):
        raise _fail(4, "STAGED_MANIFEST_UNPARSEABLE", "The staged run manifest was not a JSON object")
    if manifest.get("status") != "complete":
        raise _fail(4, "STAGED_MANIFEST_INCOMPLETE", "The staged run manifest does not report a complete run")
    if manifest.get("launch_nonce") != nonce_hex:
        raise _fail(4, "STAGED_NONCE_MISMATCH", "The staged run manifest nonce does not match this launch")
    # G7: the scorer's S6 also pins this, but catching it here means a launch
    # with the wrong interpreter never gets an attestation at all.
    if manifest.get("python_version") != cc.PYTHON_VERSION_STRING:
        raise _fail(4, "STAGED_PYTHON_VERSION_MISMATCH", "The staged run manifest python_version is not the pinned contract value")
    if cc.validate_isolation_block(manifest.get("isolation")):
        raise _fail(4, "STAGED_ISOLATION_NOT_CONFIRMED", "The staged run manifest does not report confirmed isolation")
    return manifest, staged_run_dir


def _cleanup_empty_staging(staging_run_dir: Path | None) -> None:
    # F12: only remove a failed staging or launch folder when it is empty --
    # if the container (or a partial attestation write) left anything behind,
    # keep it for a human to inspect. G2 reuses this for the launch folder too.
    if staging_run_dir is None:
        return
    try:
        if staging_run_dir.exists() and not any(staging_run_dir.iterdir()):
            staging_run_dir.rmdir()
    except OSError:
        pass


def run(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--run-name", required=True)
    try:
        args = parser.parse_args(argv)
    except SystemExit:
        _write({"schema_version": "offline-eval.launch.v1", "status": "invalid", "errors": [
            {"code": "USAGE_ERROR", "reference": "arguments", "description": "Invalid launcher arguments"}
        ]})
        return 2

    run_name = args.run_name
    repo_root = TOOL_DIR.parents[1]
    docker_binary: str | None = None
    container_id: str | None = None
    staging_run_dir: Path | None = None
    launch_dir: Path | None = None
    env: dict[str, str] = {}

    try:
        runs_root, staging_root, launch_root = _l1_check_run_name(run_name, repo_root)
        tool_hashes, guard_path, guard_hash = _l2_tool_and_guard_hashes(repo_root)
        docker_binary = _resolve_docker_binary()
        canary_name = f"{CANARY_ENV_NAME_PREFIX}{secrets.token_hex(8)}"
        canary_value = secrets.token_hex(16)
        env = _subprocess_env(canary_name, canary_value)
        _l3_check_image_digest(docker_binary, env)

        staging_run_dir = prepare_new_output_directory(staging_root / run_name, staging_root)
        # G2: reserve the launch folder *before* `docker create` -- if this
        # collides or fails, nothing has been started yet.
        launch_root.mkdir(parents=True, exist_ok=True)
        launch_dir = prepare_new_output_directory(launch_root / run_name, launch_root)
        mounts = cc.HostMounts(
            tool_dir=TOOL_DIR,
            guard_file=guard_path,
            phase2_dir=repo_root.parent / "handoff" / "evaluations" / cc.PHASE2_BUNDLE_DIRNAME,
            phase3a_dir=repo_root.parent / "handoff" / "evaluations" / cc.PHASE3A_PROFILE_DIRNAME,
            staging_dir=staging_run_dir,
        )
        for source in (mounts.tool_dir, mounts.guard_file, mounts.phase2_dir, mounts.phase3a_dir):
            if not source.exists():
                raise _fail(2, "MOUNT_SOURCE_MISSING", "A required mount source does not exist on the host")

        nonce_hex = secrets.token_hex(32)
        container_id, inspect_subset_full = _l5_create_and_validate(docker_binary, env, mounts, nonce_hex, run_name)
        exit_code, container_stdout = _l6_l7_start_and_remove(docker_binary, env, container_id)
        container_id = None  # removed by _l6_l7_start_and_remove regardless of exit_code

        if exit_code != 0:
            raise _fail(
                4,
                "CONTAINER_EXIT_NONZERO",
                "The container exited with a non-zero status",
                container_report=_parse_container_report(container_stdout),
            )

        manifest, staged_run_dir = _l8_validate_staged_output(staging_run_dir, run_name, nonce_hex)

        # H1: read the manifest bytes from the *staged* location and write the
        # attestation *before* the rename, not after. The invariant this
        # preserves: a run present in phase3-runs always has an attestation.
        # If we wrote the attestation after the rename and failed in between,
        # a run would land in phase3-runs with no attestation and no way for a
        # later run to reuse its name (L1 sees it as already existing) -- the
        # exact damage N2 was written to prevent, just moved to a new seam.
        # Rename does not change file content, so this hash is also the final
        # manifest's hash once the rename succeeds.
        manifest_raw = read_limited_file(staged_run_dir / "run-manifest.json", "run-manifest.json")

        attestation = {
            "schema_version": cc.ATTESTATION_SCHEMA,
            "run_name": run_name,
            "launch_nonce": nonce_hex,
            "image_ref": cc.IMAGE_REF,
            "image_id": cc.IMAGE_ID,
            "python_version": manifest.get("python_version"),
            "container_contract_sha256": tool_hashes.get("container_contract.py"),
            "tool_file_sha256": dict(sorted(tool_hashes.items())),
            # G1: hashed on the host, before mounting, for the scorer to cross-
            # check against the container's own scanner_source_sha256.
            "guard_source_sha256": guard_hash,
            "run_manifest_sha256": sha256_bytes(manifest_raw),
            # F6: only the validated subset, never the full inspect (no host paths).
            "inspect_subset": cc.inspect_subset(inspect_subset_full),
            "container_exit_code": exit_code,
            "created_at_unix": time.time(),
        }
        attestation_raw = (json.dumps(attestation, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
        attestation_path = launch_dir / cc.ATTESTATION_FILENAME
        try:
            with attestation_path.open("xb") as handle:
                handle.write(attestation_raw)
        except OSError as exc:
            raise _fail(3, "ATTESTATION_WRITE_FAILED", "Failed to write the launch attestation") from exc

        final_run_dir = runs_root / run_name
        runs_root.mkdir(parents=True, exist_ok=True)
        try:
            staged_run_dir.rename(final_run_dir)
        except OSError as exc:
            # H1: the run must never land in phase3-runs without an attestation.
            # Undo the attestation we just wrote (best-effort) and leave the run
            # in staging where L1 will still refuse to reuse its name.
            try:
                attestation_path.unlink()
            except OSError:
                pass
            raise _fail(3, "RUN_MOVE_FAILED", "Failed to move the staged run into the fixed output root") from exc
        staging_run_dir = None  # moved; nothing left to clean up on a later failure

        _write(
            {
                "schema_version": "offline-eval.launch.v1",
                "status": "complete",
                "run_name": run_name,
                "run_manifest_sha256": sha256_bytes(manifest_raw),
                "launch_attestation_sha256": sha256_bytes(attestation_raw),
                "errors": [],
            }
        )
        return 0
    except LauncherFailure as failure:
        if container_id is not None and docker_binary is not None:
            _run_docker(docker_binary, ["rm", "-f", container_id], env=env)
        _cleanup_empty_staging(staging_run_dir)
        _cleanup_empty_staging(launch_dir)
        _write(
            {
                "schema_version": "offline-eval.launch.v1",
                "status": "failed",
                "run_name": run_name,
                "errors": [{"code": failure.code, "reference": "launcher", "description": failure.description}],
                "container_report": failure.container_report,
            }
        )
        return failure.exit_code
    except ContractViolation as exc:
        if container_id is not None and docker_binary is not None:
            _run_docker(docker_binary, ["rm", "-f", container_id], env=env)
        _cleanup_empty_staging(staging_run_dir)
        _cleanup_empty_staging(launch_dir)
        _write(
            {
                "schema_version": "offline-eval.launch.v1",
                "status": "failed",
                "run_name": run_name,
                "errors": [exc.issue.as_dict()],
            }
        )
        return 2
    except Exception:
        # F12: anything unanticipated still gets cleaned up and reported safely,
        # never a silent crash or a leftover running container.
        if container_id is not None and docker_binary is not None:
            _run_docker(docker_binary, ["rm", "-f", container_id], env=env)
        _cleanup_empty_staging(staging_run_dir)
        _cleanup_empty_staging(launch_dir)
        _write(
            {
                "schema_version": "offline-eval.launch.v1",
                "status": "failed",
                "run_name": run_name,
                "errors": [
                    {"code": "INTERNAL_ERROR", "reference": "launcher", "description": "Unexpected launcher failure"}
                ],
            }
        )
        return 3


if __name__ == "__main__":
    raise SystemExit(run())
