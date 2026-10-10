"""A5/A6 synthetic-bundle self-test harness for the OS-isolation container
contract (design doc §17, J1-J8).

This is a diagnostic tool, not part of the scored pipeline. It generates a
fully self-consistent SYNTHETIC evidence bundle + phase3a-style profile (9
cases, zero real evidence), mounts it at the exact same container paths real
evidence uses, and runs it through the unmodified `container_contract`
pure functions (`build_docker_args`, `validate_inspect`, `inspect_subset`) so
the contract under test is the one that will actually be deployed (J1).

Results land under a completely separate root (`phase3-selftest-runs` /
`-staging` / `-launches`, J2) that the scorer never reads, so a selftest run
can never be scored or confused with a real one.

`--negative` (A6) builds a deliberately contract-violating argv (`--network
bridge`) inside this file only; `container_contract.build_docker_args` is
never modified (J6).

Running this file (in either mode) requires a running Docker daemon: it
creates, inspects, starts, and removes real containers. Do not invoke it in
an environment where Docker must not be touched.
"""

from __future__ import annotations

import argparse
import json
import re
import secrets
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable


TOOL_DIR = Path(__file__).resolve().parent
if str(TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(TOOL_DIR))

import bundle  # noqa: E402
import container_contract as cc  # noqa: E402
import isolation  # noqa: E402
import launcher  # noqa: E402
from contracts import canonical_json_sha256, sha256_bytes  # noqa: E402
from session_adapter import SESSION_NOTE_TEMPLATE  # noqa: E402


# ---------------------------------------------------------------------------
# J3: the synthetic bundle -- 9 cases, 0 real evidence.
# ---------------------------------------------------------------------------

SYNTHETIC_TICKER = "SYNTHETIC"
SYNTHETIC_TRADE_DATE = "2026-09-18"
SYNTHETIC_SPLIT = "development_representative_unopened"

# The same nine sentences tests/test_review_integration.py's synthetic_writer
# feeds the guard, in the same order, so the reproduced hit vector is the same
# fixed value: [1, 0, 0, 0, 0, 0, 1, 0, 1].
SYNTHETIC_TEXTS = (
    "The stock closed at 100 KRW today.",
    "The previous close was 100 KRW.",
    "If it closes below 100 KRW today, we would reassess.",
    SESSION_NOTE_TEMPLATE.format(ticker=SYNTHETIC_TICKER, trade_date=SYNTHETIC_TRADE_DATE),
    "The stock closed at 100 KRW today.",
    "The stock closed at 100 KRW on September 17.",
    "The stock closed at 100 KRW on September 18.",
    "The stock closed at 100 KRW today. A rebound is likely.",
    "Foreign investors were net sellers for five sessions. This distribution "
    "pattern describes investor flows, not a completed daily candle.",
)
SYNTHETIC_CASE_IDS = tuple(f"S{i:02}" for i in range(1, 10))
# 1-based positions (matching the synthetic_writer fixture) that get the
# confirmed_error draft label; everything else is confirmed_valid.
SYNTHETIC_ERROR_LABEL_POSITIONS = frozenset({1, 7, 8})
EXPECTED_HIT_VECTOR = (1, 0, 0, 0, 0, 0, 1, 0, 1)


def _evidence_payload(decision_id: int, text: str) -> dict[str, Any]:
    # `full_transcript` differs per case (it embeds decision_id) even where two
    # cases share the same `text`, so every case's transcript_sha256 is unique
    # and split-manifest's group_transcript duplicate-group logic stays trivial.
    return {"text": text, "full_transcript": {"decision_id": decision_id, "body": text}}


def _duplicate_groups(rows: list[dict[str, Any]], key_fn: Callable[[dict[str, Any]], str]) -> list[dict[str, Any]]:
    # Mirrors bundle._duplicate_groups exactly (same grouping, same shape) so
    # the generated split-manifest.json is self-consistent with what
    # bundle.py recomputes and cross-checks at load time.
    groups: dict[str, list[int]] = {}
    for row in rows:
        groups.setdefault(key_fn(row), []).append(row["id"])
    return [{"key": key, "ids": sorted(ids)} for key, ids in sorted(groups.items()) if len(ids) > 1]


def build_synthetic_documents(guard_source_sha256: str) -> dict[str, Any]:
    """Pure: build every JSON document the synthetic 9-case bundle needs,
    hash/count self-consistent with bundle.py's cross-checks (design §17 J3).
    Returns logical-name -> JSON value; callers encode, hash, and write it."""

    inventory: list[dict[str, Any]] = []
    evidence_by_id: dict[int, dict[str, Any]] = {}
    claims: list[dict[str, Any]] = []
    for position, (case_id, text) in enumerate(zip(SYNTHETIC_CASE_IDS, SYNTHETIC_TEXTS), start=1):
        payload = _evidence_payload(position, text)
        evidence_by_id[position] = payload
        row = {
            "id": position,
            "evidence_path": f"evidence/{position}.json",
            "evidence_sha256": canonical_json_sha256(payload),
            "transcript_sha256": canonical_json_sha256(payload["full_transcript"]),
            "ticker": SYNTHETIC_TICKER,
            "trade_date": SYNTHETIC_TRADE_DATE,
            "representative": True,
            "diagnostic": True,
            "split_proposal": SYNTHETIC_SPLIT,
            # Not read by bundle.py; read by session_adapter.scan_selected_cases.
            "session_in_progress": None if position == 5 else True,
        }
        inventory.append(row)
        label = "confirmed_error" if position in SYNTHETIC_ERROR_LABEL_POSITIONS else "confirmed_valid"
        node = payload["text"]
        claims.append(
            {
                "case_id": case_id,
                "decision_id": position,
                "evidence_path": row["evidence_path"],
                "evidence_sha256": row["evidence_sha256"],
                "label": label,
                "review_status": "draft_single_reviewer",
                "path": ["text"],
                "node_sha256_utf8": sha256_bytes(node.encode("utf-8")),
                "span_start": 0,
                "span_end_exclusive": len(node),
                "needle": node,
            }
        )

    split_rows = [
        {
            "id": row["id"],
            "split": SYNTHETIC_SPLIT,
            "group_original": str(row["id"]),
            "group_ticker_day": f"{row['ticker']}|{row['trade_date']}",
            "group_transcript": row["transcript_sha256"],
            "representative": row["representative"],
            "diagnostic": row["diagnostic"],
        }
        for row in inventory
    ]
    duplicate_groups = {
        "group_ticker_day": _duplicate_groups(inventory, lambda r: f"{r['ticker']}|{r['trade_date']}"),
        "group_transcript": _duplicate_groups(inventory, lambda r: r["transcript_sha256"]),
    }

    representative_ids = sorted(row["id"] for row in inventory if row["representative"])
    diagnostic_ids = {row["id"] for row in inventory if row["diagnostic"]}
    collection_summary = {
        "representative_n": len(representative_ids),
        "diagnostic_found_n": len(diagnostic_ids),
        "unique_evidence_n": len(inventory),
        "overlap_n": len(set(representative_ids) & diagnostic_ids),
        "sample_days": [
            {"trade_date": SYNTHETIC_TRADE_DATE, "ids": representative_ids, "n": len(representative_ids)}
        ],
        "split_counts": {SYNTHETIC_SPLIT: len(inventory)},
    }

    error_count = len(SYNTHETIC_ERROR_LABEL_POSITIONS)
    expected_counts = {
        "unique_evidence": len(inventory),
        "source_claims": len(claims),
        "candidates": len(claims),
        "candidate_error_drafts": error_count,
        "candidate_valid_drafts": len(claims) - error_count,
        "out_of_scope_clear": 0,
        "ambiguous": 0,
        "unassessable": 0,
    }

    profile = {
        "schema_version": bundle.PROFILE_SCHEMA,
        # This must equal the *mounted* bundle directory's basename, which is
        # always the fixed container path's tail (cc.PHASE2_MOUNT), regardless
        # of what the host-side temp folder happens to be named.
        "source_bundle": cc.PHASE2_BUNDLE_DIRNAME,
        "replay_mode": "controlled_context_rescore",
        "engine_generation_allowed": False,
        "holdout_scoring_allowed": False,
        "context_origin": "synthetic",
        "session_note_template": SESSION_NOTE_TEMPLATE,
        "candidate_case_ids": list(SYNTHETIC_CASE_IDS),
        "out_of_scope_clear_case_ids": [],
        "expected_counts": expected_counts,
        "scanner": {
            "name": "session_close",
            "source_reference": bundle.FIXED_SCANNER_REFERENCE,
            # J4: the real guard file's real hash -- the guard is code, not
            # synthetic data, and is mounted unchanged.
            "source_bytes_sha256": guard_source_sha256,
        },
    }

    return {
        "inventory.json": inventory,
        "claim-annotations.json": {"items": claims},
        "split-manifest.json": {
            "items": split_rows,
            "counts": {SYNTHETIC_SPLIT: len(split_rows)},
            "duplicate_groups": duplicate_groups,
        },
        "collection-summary.json": collection_summary,
        "evidence": evidence_by_id,
        "profile": profile,
    }


def _pretty(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def write_synthetic_environment(root: Path, guard_source_sha256: str) -> tuple[Path, Path]:
    """Impure: write a synthetic bundle directory + phase3a-style profile
    under `root`, named exactly like the real evidence mount points so the
    generated profile's `source_bundle` field is valid in both contexts:
    a host-only `cli.py validate` sanity check, and a real container mount
    (design §17 J1, J3). Returns (bundle_dir, profile_path)."""

    documents = build_synthetic_documents(guard_source_sha256)
    bundle_dir = root / cc.PHASE2_BUNDLE_DIRNAME
    evidence_dir = bundle_dir / "evidence"
    evidence_dir.mkdir(parents=True)
    profile_dir = root / cc.PHASE3A_PROFILE_DIRNAME
    profile_dir.mkdir(parents=True)

    hashes: dict[str, str] = {}
    for filename in bundle.BUNDLE_FILES:
        raw = _pretty(documents[filename])
        (bundle_dir / filename).write_bytes(raw)
        hashes[filename] = sha256_bytes(raw)

    for decision_id, payload in documents["evidence"].items():
        (evidence_dir / f"{decision_id}.json").write_bytes(_pretty(payload))

    profile = dict(documents["profile"])
    profile["source_metadata_file_sha256"] = hashes
    profile_path = profile_dir / cc.PROFILE_FILENAME
    profile_path.write_bytes(_pretty(profile))
    return bundle_dir, profile_path


# ---------------------------------------------------------------------------
# J6: the deliberately contract-violating argv, built only in this file.
# ---------------------------------------------------------------------------


def build_negative_docker_args(*, mounts: cc.HostMounts, nonce_hex: str, run_name: str) -> list[str]:
    """Pure: the exact `build_docker_args` vector with `--network none`
    replaced by `--network bridge`. `container_contract.build_docker_args`
    itself is never touched (design §17 J6) -- this is the only place this
    tool is allowed to construct a non-compliant argument vector, and it
    exists only to simulate someone bypassing the launcher entirely."""

    args = cc.build_docker_args(mounts=mounts, nonce_hex=nonce_hex, run_name=run_name)
    index = args.index("--network")
    negative = list(args)
    negative[index + 1] = "bridge"
    return negative


# ---------------------------------------------------------------------------
# J7: shape-only evidence collection (a second, diagnostic-only container).
# ---------------------------------------------------------------------------

_PROBE_SCRIPT = (
    "import json, os\n"
    "def rd(p):\n"
    "    try:\n"
    "        with open(p, 'r', encoding='utf-8') as fh:\n"
    "            return fh.read()\n"
    "    except OSError:\n"
    "        return None\n"
    "try:\n"
    "    net = sorted(os.listdir('/sys/class/net'))\n"
    "except OSError:\n"
    "    net = None\n"
    "print(json.dumps({\n"
    "    'mountinfo': rd('/proc/self/mountinfo'),\n"
    "    'status': rd('/proc/self/status'),\n"
    "    'env_keys': sorted(os.environ.keys()),\n"
    "    'net_class': net,\n"
    "}))\n"
)

_STATUS_SHAPE_FIELDS = ("Uid", "Gid", "CapEff", "CapPrm", "CapInh", "CapBnd", "NoNewPrivs", "Seccomp")


def build_probe_args(*, mounts: cc.HostMounts, nonce_hex: str, network_mode: str) -> list[str]:
    """Pure: a second, diagnostic-only `docker create ...` argv. Same mounts
    and security posture as the run under test, but the command is a fixed
    inline script that reports shapes only (never values) for J7 evidence.
    This never replaces or mutates `build_docker_args`'s own argv."""

    if cc.NONCE_RE.fullmatch(nonce_hex) is None:
        raise ValueError("nonce format is not allowed")
    for path in (mounts.tool_dir, mounts.guard_file, mounts.phase2_dir, mounts.phase3a_dir, mounts.staging_dir):
        if not path.is_absolute():
            raise ValueError("mount source paths must be absolute")

    args: list[str] = ["create"]
    args += ["--network", network_mode]
    args += ["--read-only"]
    args += ["--cap-drop", "ALL"]
    args += ["--security-opt", "no-new-privileges:true"]
    args += ["--user", cc.CONTAINER_USER]
    args += ["--hostname", cc.CONTAINER_HOSTNAME]
    args += ["--pids-limit", cc.RESOURCE_LIMITS["pids_limit"]]
    args += ["--memory", cc.RESOURCE_LIMITS["memory"]]
    args += ["--cpus", cc.RESOURCE_LIMITS["cpus"]]
    args += ["--env", f"{cc.NONCE_ENV_NAME}={nonce_hex}"]
    args += ["--log-driver", "none"]
    args += ["--workdir", cc.WORKDIR]
    for source, target, rw in mounts.as_tuple():
        args += ["--mount", cc._mount_arg(source, target, readonly=not rw)]
    args += [cc.IMAGE_REF]
    args += ["python", "-I", "-S", "-B", "-c", _PROBE_SCRIPT]
    return args


def _mountinfo_shape(text: str) -> list[dict[str, Any]] | None:
    # J7: destination (mount_point), options, and fs_type -- fs_type is a
    # filesystem kind (proc, sysfs, tmpfs, overlay, ...), not host
    # information, and is needed to check cc.PSEUDO_FS_TYPES against real
    # Docker output. `entry.source` is never included -- that is the one
    # field on _MountInfoEntry that can carry host-identifying information.
    try:
        entries = isolation.parse_mountinfo(text)
    except ValueError:
        return None
    return [
        {"mount_point": entry.mount_point, "fs_type": entry.fs_type, "options": sorted(entry.options)}
        for entry in entries
    ]


def _status_shape(text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip()
        if key in _STATUS_SHAPE_FIELDS:
            fields[key] = value.strip()
    return fields


def _empty_evidence() -> dict[str, Any]:
    return {
        "schema_version": "offline-eval.a5-selftest.evidence.v1",
        "mountinfo_entries": None,
        "status_fields": None,
        "env_keys": None,
        "net_class_names": None,
        "host_inspect_subset": None,
        "collection_error": None,
    }


def _collect_probe_evidence(
    docker_binary: str,
    env: dict[str, str],
    mounts: cc.HostMounts,
    nonce_hex: str,
    run_docker: Callable[..., Any],
    *,
    network_mode: str,
) -> dict[str, Any]:
    """Best-effort J7 evidence collection via a second container. Never
    raises -- a collection failure must never mask the primary A5/A6 result."""

    evidence = _empty_evidence()
    try:
        args = build_probe_args(mounts=mounts, nonce_hex=nonce_hex, network_mode=network_mode)
        created = run_docker(docker_binary, args, env=env)
        if created.returncode != 0:
            evidence["collection_error"] = "PROBE_CREATE_FAILED"
            return evidence
        container_id = created.stdout.strip()
        try:
            inspected = run_docker(docker_binary, ["inspect", container_id], env=env)
            if inspected.returncode == 0:
                try:
                    full = json.loads(inspected.stdout)[0]
                    if isinstance(full, dict):
                        evidence["host_inspect_subset"] = cc.inspect_subset(full)
                except (json.JSONDecodeError, IndexError, TypeError):
                    pass
            started = run_docker(docker_binary, ["start", "-a", container_id], env=env)
            if started.returncode == 0 and started.stdout:
                try:
                    payload = json.loads(started.stdout)
                except json.JSONDecodeError:
                    payload = None
                if isinstance(payload, dict):
                    mountinfo_text = payload.get("mountinfo")
                    if isinstance(mountinfo_text, str):
                        evidence["mountinfo_entries"] = _mountinfo_shape(mountinfo_text)
                    status_text = payload.get("status")
                    if isinstance(status_text, str):
                        evidence["status_fields"] = _status_shape(status_text)
                    env_keys = payload.get("env_keys")
                    if isinstance(env_keys, list):
                        evidence["env_keys"] = sorted(str(key) for key in env_keys)
                    net_class = payload.get("net_class")
                    if isinstance(net_class, list):
                        evidence["net_class_names"] = sorted(str(name) for name in net_class)
            elif evidence["collection_error"] is None:
                evidence["collection_error"] = "PROBE_START_FAILED"
        finally:
            run_docker(docker_binary, ["rm", "-f", container_id], env=env)
    except Exception:
        evidence["collection_error"] = "PROBE_COLLECTION_FAILED"
    return evidence


# ---------------------------------------------------------------------------
# J5: acceptance checks (pure, given already-collected results).
# ---------------------------------------------------------------------------

# Negative lookbehind: a bare Windows drive letter is never itself preceded by
# another word character, unlike (for example) ".../bin:/usr/..." in a PATH
# value, which would otherwise false-positive on "n:/".
_HOST_PATH_RE = re.compile(r"(?<![A-Za-z0-9])[A-Za-z]:[\\/]")


def _contains_host_path(value: Any, extra_markers: tuple[str, ...] = ()) -> bool:
    """Pure: True if any string in this JSON-shaped value looks like an
    absolute Windows host path, or contains one of `extra_markers`."""

    if isinstance(value, str):
        if _HOST_PATH_RE.search(value):
            return True
        return any(marker and marker in value for marker in extra_markers)
    if isinstance(value, dict):
        return any(_contains_host_path(item, extra_markers) for item in value.values())
    if isinstance(value, list):
        return any(_contains_host_path(item, extra_markers) for item in value)
    return False


def evaluate_positive_result(
    *,
    exit_code: int | None,
    manifest: dict[str, Any] | None,
    hit_vector: list[int] | None,
    inspect_subset: dict[str, Any] | None,
    nonce_hex: str,
    run_name: str,
) -> dict[str, Any]:
    """Pure: J5's five A5 acceptance checks, given already-collected results.
    Never raises -- malformed input just fails the relevant check."""

    checks: dict[str, Any] = {}
    checks["exit_zero"] = exit_code == 0
    isolation_block = manifest.get("isolation") if isinstance(manifest, dict) else None
    checks["isolation_confirmed"] = (
        isinstance(isolation_block, dict) and cc.validate_isolation_block(isolation_block) == []
    )
    checks["hit_vector_matches"] = hit_vector == list(EXPECTED_HIT_VECTOR)
    if isinstance(inspect_subset, dict):
        replay_violations = cc.validate_inspect(
            inspect_subset, nonce_hex=nonce_hex, run_name=run_name, expected_sources=None
        )
        checks["attestation_inspect_valid"] = replay_violations == []
        checks["no_host_paths_in_inspect_subset"] = not _contains_host_path(inspect_subset)
    else:
        checks["attestation_inspect_valid"] = False
        checks["no_host_paths_in_inspect_subset"] = False
    checks["overall"] = all(bool(value) for value in checks.values())
    return checks


def evaluate_negative_result(
    *, exit_code: int | None, container_report: dict[str, Any] | None, staged_entries: list[str]
) -> dict[str, Any]:
    """Pure: J6's A6 acceptance checks, given already-collected results."""

    checks: dict[str, Any] = {}
    checks["exit_is_4"] = exit_code == 4
    checks["status_withheld"] = isinstance(container_report, dict) and container_report.get("status") == "withheld"
    reason = ""
    if checks["status_withheld"]:
        assessment = container_report.get("isolation_assessment") if container_report else None
        if isinstance(assessment, dict):
            reason = assessment.get("reason", "")
    checks["probe_failure_visible"] = isinstance(reason, str) and reason.startswith("probe_failed:")
    checks["no_output_created"] = staged_entries == []
    checks["overall"] = all(bool(value) for value in checks.values())
    return checks


# ---------------------------------------------------------------------------
# Orchestration (the only part that touches Docker).
# ---------------------------------------------------------------------------


def _fail_result(run_name: str, negative: bool, code: str, description: str) -> dict[str, Any]:
    return {
        "schema_version": "offline-eval.a5-selftest.v1",
        "run_name": run_name,
        "mode": "negative" if negative else "positive",
        "exit_code": None,
        "container_report": None,
        "hit_vector": None,
        "checks": {"overall": False},
        "evidence_shapes_path": None,
        "attestation_path": None,
        "run_dir": None,
        "errors": [{"code": code, "description": description}],
    }


def _run_docker_default(
    docker_binary: str, args: list[str], *, env: dict[str, str], input_text: str | None = None
) -> Any:
    return launcher._run_docker(docker_binary, args, env=env, input_text=input_text)


def run_self_test(
    run_name: str,
    *,
    negative: bool,
    repo_root: Path,
    run_docker: Callable[..., Any] | None = None,
    resolve_docker_binary: Callable[[], str] | None = None,
) -> dict[str, Any]:
    """Run one A5 (positive) or A6 (negative) self-test. This is the one
    function that touches Docker; every function it calls into is pure or
    file-local. `run_docker`/`resolve_docker_binary` default to launcher's
    real (subprocess-invoking) helpers, and tests replace them with fakes."""

    if run_docker is None:
        run_docker = _run_docker_default
    if resolve_docker_binary is None:
        resolve_docker_binary = launcher._resolve_docker_binary

    if cc.RUN_NAME_RE.fullmatch(run_name) is None:
        return _fail_result(run_name, negative, "USAGE_ERROR", "run_name format is not allowed")

    runs_root = isolation.fixed_selftest_output_root(repo_root)
    staging_root = isolation.fixed_selftest_staging_root(repo_root)
    launch_root = isolation.fixed_selftest_launch_root(repo_root)
    for root in (runs_root, staging_root, launch_root):
        if (root / run_name).exists():
            return _fail_result(run_name, negative, "RUN_ALREADY_EXISTS", "A selftest run with this name already exists")

    guard_path = repo_root / Path(*cc.GUARD_SOURCE_RELATIVE.split("/"))
    if not guard_path.is_file():
        return _fail_result(run_name, negative, "GUARD_SOURCE_MISSING", "Guard source file is missing")
    guard_sha256 = sha256_bytes(guard_path.read_bytes())

    try:
        docker_binary = resolve_docker_binary()
    except launcher.LauncherFailure as failure:
        return _fail_result(run_name, negative, failure.code, failure.description)

    canary_name = f"{launcher.CANARY_ENV_NAME_PREFIX}{secrets.token_hex(8)}"
    canary_value = secrets.token_hex(16)
    env = launcher._subprocess_env(canary_name, canary_value)

    with tempfile.TemporaryDirectory(prefix="offline-eval-a5-") as synthetic_root_name:
        synthetic_root = Path(synthetic_root_name)
        bundle_dir, profile_path = write_synthetic_environment(synthetic_root, guard_sha256)

        launch_root.mkdir(parents=True, exist_ok=True)
        launch_dir = isolation.prepare_new_output_directory(launch_root / run_name, launch_root)
        staging_root.mkdir(parents=True, exist_ok=True)
        staging_run_dir = isolation.prepare_new_output_directory(staging_root / run_name, staging_root)

        mounts = cc.HostMounts(
            tool_dir=TOOL_DIR,
            guard_file=guard_path,
            phase2_dir=bundle_dir,
            phase3a_dir=profile_path.parent,
            staging_dir=staging_run_dir,
        )
        nonce_hex = secrets.token_hex(32)
        if negative:
            args = build_negative_docker_args(mounts=mounts, nonce_hex=nonce_hex, run_name=run_name)
        else:
            args = cc.build_docker_args(mounts=mounts, nonce_hex=nonce_hex, run_name=run_name)

        created = run_docker(docker_binary, args, env=env)
        if created.returncode != 0:
            return _fail_result(run_name, negative, "CONTAINER_CREATE_FAILED", "docker create failed")
        container_id = created.stdout.strip()

        inspect_subset_full: dict[str, Any] | None = None
        try:
            inspected = run_docker(docker_binary, ["inspect", container_id], env=env)
            if inspected.returncode == 0:
                try:
                    parsed = json.loads(inspected.stdout)[0]
                    inspect_subset_full = parsed if isinstance(parsed, dict) else None
                except (json.JSONDecodeError, IndexError, TypeError):
                    inspect_subset_full = None

            started = run_docker(docker_binary, ["start", "-a", container_id], env=env)
            exit_code = started.returncode
            container_report = launcher._parse_container_report(started.stdout)
        finally:
            run_docker(docker_binary, ["rm", "-f", container_id], env=env)

        evidence = _collect_probe_evidence(
            docker_binary,
            env,
            mounts,
            nonce_hex,
            run_docker,
            network_mode="bridge" if negative else "none",
        )

        manifest: dict[str, Any] | None = None
        hit_vector: list[int] | None = None
        staged_run_dir = staging_run_dir / run_name
        staged_entries: list[str] = []
        if staged_run_dir.is_dir():
            staged_entries = sorted(entry.name for entry in staged_run_dir.iterdir())
            manifest_path = staged_run_dir / "run-manifest.json"
            if manifest_path.is_file():
                try:
                    loaded_manifest = json.loads(manifest_path.read_text("utf-8"))
                    manifest = loaded_manifest if isinstance(loaded_manifest, dict) else None
                except json.JSONDecodeError:
                    manifest = None
            findings_path = staged_run_dir / "findings.json"
            if findings_path.is_file():
                try:
                    findings = json.loads(findings_path.read_text("utf-8"))
                    cases = findings.get("cases") if isinstance(findings, dict) else None
                    if isinstance(cases, list):
                        hit_vector = [
                            len(case.get("raw_findings", []))
                            for case in sorted(cases, key=lambda item: item.get("case_id", ""))
                        ]
                except (json.JSONDecodeError, TypeError):
                    hit_vector = None

        inspect_subset = cc.inspect_subset(inspect_subset_full) if inspect_subset_full is not None else None

        attestation_path_str: str | None = None
        final_run_dir: Path | None = None
        if negative:
            checks = evaluate_negative_result(
                exit_code=exit_code, container_report=container_report, staged_entries=staged_entries
            )
        else:
            checks = evaluate_positive_result(
                exit_code=exit_code,
                manifest=manifest,
                hit_vector=hit_vector,
                inspect_subset=inspect_subset,
                nonce_hex=nonce_hex,
                run_name=run_name,
            )
            if checks["overall"] and manifest is not None and inspect_subset is not None:
                attestation = {
                    "schema_version": cc.ATTESTATION_SCHEMA,
                    "run_name": run_name,
                    "launch_nonce": nonce_hex,
                    "image_ref": cc.IMAGE_REF,
                    "image_id": cc.IMAGE_ID,
                    "python_version": manifest.get("python_version"),
                    "inspect_subset": inspect_subset,
                    "container_exit_code": exit_code,
                    "selftest": True,
                }
                attestation_path = launch_dir / cc.ATTESTATION_FILENAME
                attestation_path.write_bytes(_pretty(attestation))
                attestation_path_str = str(attestation_path)
                if staged_run_dir.is_dir():
                    runs_root.mkdir(parents=True, exist_ok=True)
                    final_target = runs_root / run_name
                    staged_run_dir.rename(final_target)
                    final_run_dir = final_target

        evidence_path = launch_dir / "evidence-shapes.json"
        evidence_path.write_bytes(_pretty(evidence))

        return {
            "schema_version": "offline-eval.a5-selftest.v1",
            "run_name": run_name,
            "mode": "negative" if negative else "positive",
            "exit_code": exit_code,
            "container_report": container_report,
            "hit_vector": hit_vector,
            "checks": checks,
            "evidence_shapes_path": str(evidence_path),
            "attestation_path": attestation_path_str,
            "run_dir": str(final_run_dir) if final_run_dir else (str(staged_run_dir) if staged_run_dir.is_dir() else None),
            "errors": [],
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--run-name")
    parser.add_argument("--negative", action="store_true")
    args = parser.parse_args(argv)
    run_name = args.run_name or f"{'a6' if args.negative else 'a5'}-selftest-{secrets.token_hex(4)}"
    repo_root = TOOL_DIR.parents[1]
    result = run_self_test(run_name, negative=args.negative, repo_root=repo_root)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    sys.stdout.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
    return 0 if result.get("checks", {}).get("overall") else 1


if __name__ == "__main__":
    raise SystemExit(main())
