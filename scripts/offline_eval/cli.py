"""Isolated command-line entry point for Phase 3 offline evaluation."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


TOOL_DIR = Path(__file__).resolve().parent
if str(TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(TOOL_DIR))

import container_contract as cc  # noqa: E402
from bundle import load_validated_bundle, validate_bundle  # noqa: E402
from contracts import ContractViolation, ValidationIssue  # noqa: E402
from isolation import (  # noqa: E402
    assess_os_isolation,
    IsolationViolation,
    RuntimePolicy,
    fixed_launch_root,
    fixed_output_root,
    install_runtime_policy,
    prepare_new_output_directory,
    require_material_scan_isolation,
)
from report import (  # noqa: E402
    collect_tool_hashes,
    error_report,
    internal_error_report,
    isolation_error_report,
    render_json,
    scan_stdout_report,
    valid_report,
    write_incomplete_marker,
    write_scan_artifacts,
)
from scoring import score_review  # noqa: E402
from session_adapter import instantiate_guard, prepare_guard, scan_selected_cases  # noqa: E402


class SafeArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise ContractViolation("USAGE_ERROR", "arguments", "Invalid offline-eval command arguments")


def _parser() -> argparse.ArgumentParser:
    parser = SafeArgumentParser(add_help=False, exit_on_error=False)
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate", add_help=False, exit_on_error=False)
    validate.add_argument("--bundle", required=True)
    validate.add_argument("--profile", required=True)
    scan = commands.add_parser("scan", add_help=False, exit_on_error=False)
    scan.add_argument("--bundle", required=True)
    scan.add_argument("--profile", required=True)
    scan.add_argument("--out", required=True)
    score = commands.add_parser("score", add_help=False, exit_on_error=False)
    score.add_argument("--run", required=True)
    score.add_argument("--review", required=True)
    return parser


def _write(report: dict[str, object]) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    sys.stdout.write(render_json(report) + "\n")


def _record_incomplete(
    output: Path | None,
    issue: ValidationIssue,
    *,
    phase: str,
    scanner_ran: bool,
    tool_hashes: dict[str, str] | None,
) -> str:
    if output is None:
        return "not_attempted"
    try:
        write_incomplete_marker(
            output,
            issue,
            phase=phase,
            scanner_ran=scanner_ran,
            tool_file_sha256=tool_hashes,
        )
    except Exception:
        return "write_failed"
    return "written"


def main(argv: list[str] | None = None) -> int:
    command = "unknown"
    output: Path | None = None
    tool_hashes: dict[str, str] | None = None
    scanner_ran = False
    phase = "argument_validation"
    isolation_assessment = None
    try:
        phase = "argument_validation"
        try:
            args = _parser().parse_args(argv)
        except argparse.ArgumentError as exc:
            raise ContractViolation(
                "USAGE_ERROR",
                "arguments",
                "Invalid offline-eval command arguments",
            ) from exc
        command = args.command
        repo_root = TOOL_DIR.parents[1]
        if command == "validate":
            phase = "bundle_validation"
            result = validate_bundle(Path(args.bundle), Path(args.profile), repo_root)
            _write(valid_report(result))
            return 0
        if command == "score":
            phase = "review_scoring"
            score = score_review(
                Path(args.run),
                Path(args.review),
                fixed_output_root(repo_root),
                fixed_launch_root(repo_root),
            )
            _write(score)
            return 0

        phase = "os_isolation_verification"
        # H5: assess exactly once. require_material_scan_isolation() is always
        # given *this* call's own return value -- never a second, independently
        # computed assessment and never a caller-supplied stand-in -- so there is
        # no path for "confirmed" to be injected from outside this one probe run.
        isolation_assessment = assess_os_isolation()
        os_isolation = require_material_scan_isolation(assessment=isolation_assessment)
        phase = "bundle_validation"
        loaded = load_validated_bundle(Path(args.bundle), Path(args.profile), repo_root)
        phase = "scanner_preparation"
        prepared = prepare_guard(repo_root, loaded.validation.scanner_source_sha256)
        tool_hashes = collect_tool_hashes(TOOL_DIR)
        phase = "output_preparation"
        output = prepare_new_output_directory(Path(args.out), fixed_output_root(repo_root))
        policy = RuntimePolicy(scanner_filename=prepared.code.co_filename, output_directory=output)
        phase = "runtime_isolation_installation"
        isolation = install_runtime_policy(policy, os_isolation=os_isolation)
        scanner = instantiate_guard(prepared)
        phase = "scanner_execution"
        def mark_scanner_started() -> None:
            nonlocal scanner_ran
            scanner_ran = True

        cases = scan_selected_cases(loaded, scanner, on_scanner_start=mark_scanner_started)
        phase = "artifact_write"
        raw_nonce = os.environ.get(cc.NONCE_ENV_NAME)
        launch_nonce = raw_nonce if raw_nonce and cc.NONCE_RE.fullmatch(raw_nonce) else None
        python_version = ".".join(str(part) for part in sys.version_info[:3])
        manifest = write_scan_artifacts(
            output,
            loaded,
            cases,
            isolation,
            tool_hashes,
            launch_nonce=launch_nonce,
            python_version=python_version,
        )
        _write(scan_stdout_report(output.name, manifest))
        return 0
    except IsolationViolation as exc:
        marker_status = _record_incomplete(
            output,
            exc.issue,
            phase=phase,
            scanner_ran=scanner_ran,
            tool_hashes=tool_hashes,
        )
        _write(isolation_error_report(
            exc.issue,
            scanner_ran=scanner_ran,
            isolation_assessment=isolation_assessment.as_dict() if isolation_assessment else None,
            incomplete_marker_status=marker_status,
        ))
        return 4
    except ContractViolation as exc:
        marker_status = _record_incomplete(
            output if command == "scan" else None,
            exc.issue,
            phase=phase,
            scanner_ran=scanner_ran,
            tool_hashes=tool_hashes,
        )
        _write(error_report(
            command,
            exc.issue,
            scanner_ran=scanner_ran,
            incomplete_marker_status=marker_status,
        ))
        return 2
    except Exception:
        issue = ValidationIssue("INTERNAL_ERROR", "offline_eval", "Unexpected internal failure")
        marker_status = _record_incomplete(
            output if command == "scan" else None,
            issue,
            phase=phase,
            scanner_ran=scanner_ran,
            tool_hashes=tool_hashes,
        )
        if command == "validate":
            _write(internal_error_report())
        else:
            _write(error_report(
                command,
                issue,
                scanner_ran=scanner_ran,
                incomplete_marker_status=marker_status,
            ))
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
