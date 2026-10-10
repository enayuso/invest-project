"""Hash-pinned adapter for the production session-close detector."""

from __future__ import annotations

import ast
import builtins
from dataclasses import dataclass
from datetime import date
import logging
from pathlib import Path
import re
from types import CodeType
from typing import Any, Callable

from bundle import FIXED_SCANNER_REFERENCE, LoadedBundle
from contracts import (
    ContractViolation,
    MAX_TEXT_CODEPOINTS,
    canonical_json_sha256,
    read_limited_file,
    require_dict,
    require_safe_absolute_path,
    require_sha256,
    resolve_json_path,
    sha256_bytes,
)
from isolation import IsolationViolation


SESSION_NOTE_TEMPLATE = "SESSION IN PROGRESS: {ticker}'s regular session is still open as of {trade_date}."
ALLOWED_FAMILIES = {"closed_assertion", "completed_candle"}
_FINDING_RE = re.compile(
    r"offline_eval: \[(closed_assertion|completed_candle)\] (.+) "
    r"vs SESSION-IN-PROGRESS ground truth\Z",
    re.DOTALL,
)
_SIMPLE_REPR_ESCAPES = {
    "\\": "\\",
    "'": "'",
    '"': '"',
    "a": "\a",
    "b": "\b",
    "f": "\f",
    "n": "\n",
    "r": "\r",
    "t": "\t",
    "v": "\v",
}
_HEX_DIGITS = frozenset("0123456789abcdefABCDEF")


@dataclass(frozen=True)
class PreparedGuard:
    code: CodeType
    source_sha256: str


def _validate_source_imports(source: str) -> None:
    try:
        tree = ast.parse(source, filename=FIXED_SCANNER_REFERENCE)
    except SyntaxError as exc:
        raise ContractViolation(
            "SCANNER_SOURCE_INVALID",
            FIXED_SCANNER_REFERENCE,
            "Scanner source cannot be parsed",
        ) from exc
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = {alias.name for alias in node.names}
            if not names.issubset({"logging", "re"}):
                raise ContractViolation(
                    "SCANNER_IMPORT_REJECTED",
                    FIXED_SCANNER_REFERENCE,
                    "Scanner source imports a module outside the fixed allowlist",
                )
        elif isinstance(node, ast.ImportFrom):
            if node.level != 0 or node.module != "__future__":
                raise ContractViolation(
                    "SCANNER_IMPORT_REJECTED",
                    FIXED_SCANNER_REFERENCE,
                    "Scanner source imports a module outside the fixed allowlist",
                )


def prepare_guard(repo_root: Path, expected_sha256: str) -> PreparedGuard:
    source_path = repo_root / Path(*FIXED_SCANNER_REFERENCE.split("/"))
    source_path = require_safe_absolute_path(source_path, FIXED_SCANNER_REFERENCE, kind="file")
    raw = read_limited_file(source_path, FIXED_SCANNER_REFERENCE)
    actual_sha256 = sha256_bytes(raw)
    if actual_sha256 != require_sha256(expected_sha256, "scanner_source_sha256"):
        raise ContractViolation(
            "FILE_HASH_MISMATCH",
            FIXED_SCANNER_REFERENCE,
            "Scanner source hash differs from validated profile",
        )
    try:
        source = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ContractViolation(
            "SCANNER_SOURCE_INVALID",
            FIXED_SCANNER_REFERENCE,
            "Scanner source is not UTF-8",
        ) from exc
    _validate_source_imports(source)
    try:
        code = compile(source, FIXED_SCANNER_REFERENCE, "exec", dont_inherit=True)
    except (SyntaxError, ValueError) as exc:
        raise ContractViolation(
            "SCANNER_SOURCE_INVALID",
            FIXED_SCANNER_REFERENCE,
            "Scanner source cannot be compiled",
        ) from exc
    return PreparedGuard(code=code, source_sha256=actual_sha256)


def instantiate_guard(prepared: PreparedGuard) -> Callable[..., list[str]]:
    real_import = builtins.__import__

    def fixed_import(
        name: str,
        globals_: dict[str, Any] | None = None,
        locals_: dict[str, Any] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        if level != 0 or name not in {"__future__", "logging", "re"}:
            raise ContractViolation(
                "SCANNER_IMPORT_REJECTED",
                FIXED_SCANNER_REFERENCE,
                "Scanner attempted an import outside the fixed allowlist",
            )
        return real_import(name, globals_, locals_, fromlist, level)

    safe_builtins = vars(builtins).copy()
    safe_builtins["__import__"] = fixed_import
    namespace: dict[str, Any] = {
        "__name__": "offline_eval_session_close_guard",
        "__file__": FIXED_SCANNER_REFERENCE,
        "__package__": None,
        "__builtins__": safe_builtins,
    }
    exec(prepared.code, namespace, namespace)
    logger = namespace.get("logger")
    if isinstance(logger, logging.Logger):
        logger.handlers = [logging.NullHandler()]
        logger.propagate = False
        logger.disabled = True
    scanner = namespace.get("scan_session_close_claims")
    if not callable(scanner):
        raise ContractViolation(
            "SCANNER_ENTRYPOINT_MISSING",
            FIXED_SCANNER_REFERENCE,
            "Fixed scanner entry point is missing",
        )
    return scanner


def _find_all(text: str, needle: str) -> list[int]:
    if not needle:
        return []
    positions: list[int] = []
    start = 0
    while True:
        found = text.find(needle, start)
        if found < 0:
            return positions
        positions.append(found)
        start = found + 1


def _decode_bounded_repr_string(literal: str, reference: str) -> str:
    """Decode the exact repr() subset emitted by the pinned scanner without compile()."""

    if len(literal) > (MAX_TEXT_CODEPOINTS * 10) + 2:
        raise ContractViolation(
            "SCANNER_OUTPUT_INVALID",
            reference,
            "Scanner quote literal exceeds its encoded size limit",
        )
    if len(literal) < 2 or literal[0] not in {"'", '"'} or literal[-1] != literal[0]:
        raise ContractViolation(
            "SCANNER_OUTPUT_INVALID",
            reference,
            "Scanner quote is not a plain string repr",
        )
    quote_mark = literal[0]
    decoded: list[str] = []
    index = 1
    end = len(literal) - 1
    while index < end:
        character = literal[index]
        if character == quote_mark or character in {"\n", "\r"}:
            raise ContractViolation(
                "SCANNER_OUTPUT_INVALID",
                reference,
                "Scanner quote contains an invalid unescaped character",
            )
        if character != "\\":
            decoded.append(character)
            index += 1
            continue
        index += 1
        if index >= end:
            raise ContractViolation(
                "SCANNER_OUTPUT_INVALID",
                reference,
                "Scanner quote ends with an incomplete escape",
            )
        escape = literal[index]
        if escape in _SIMPLE_REPR_ESCAPES:
            decoded.append(_SIMPLE_REPR_ESCAPES[escape])
            index += 1
            continue
        widths = {"x": 2, "u": 4, "U": 8}
        width = widths.get(escape)
        if width is None:
            raise ContractViolation(
                "SCANNER_OUTPUT_INVALID",
                reference,
                "Scanner quote uses an unsupported escape",
            )
        digits = literal[index + 1:index + 1 + width]
        if len(digits) != width or any(digit not in _HEX_DIGITS for digit in digits):
            raise ContractViolation(
                "SCANNER_OUTPUT_INVALID",
                reference,
                "Scanner quote contains an invalid hexadecimal escape",
            )
        codepoint = int(digits, 16)
        if codepoint > 0x10FFFF or 0xD800 <= codepoint <= 0xDFFF:
            raise ContractViolation(
                "SCANNER_OUTPUT_INVALID",
                reference,
                "Scanner quote contains an invalid Unicode scalar",
            )
        decoded.append(chr(codepoint))
        index += width + 1
    value = "".join(decoded)
    if len(value) > MAX_TEXT_CODEPOINTS or repr(value) != literal:
        raise ContractViolation(
            "SCANNER_OUTPUT_INVALID",
            reference,
            "Scanner quote is not a canonical bounded string repr",
        )
    return value


def parse_finding_contract(raw: str, index: int) -> tuple[str, str]:
    """Return the family and decoded quote from one fixed scanner output."""

    reference = f"finding[{index}]"
    if len(raw) > (MAX_TEXT_CODEPOINTS * 10) + 128:
        raise ContractViolation(
            "SCANNER_OUTPUT_INVALID",
            reference,
            "Scanner finding exceeds its encoded size limit",
        )
    match = _FINDING_RE.fullmatch(raw)
    if match is None:
        raise ContractViolation(
            "SCANNER_OUTPUT_INVALID",
            reference,
            "Scanner finding does not match the fixed output contract",
        )
    family = match.group(1)
    if family not in ALLOWED_FAMILIES:
        raise ContractViolation(
            "SCANNER_OUTPUT_INVALID",
            reference,
            "Scanner finding has an unknown family",
        )
    literal = match.group(2)
    quote = _decode_bounded_repr_string(literal, reference)
    return family, quote


def _parse_finding(raw: str, node: str, claim_start: int, claim_end: int, index: int) -> dict[str, Any]:
    family, quote = parse_finding_contract(raw, index)
    positions = _find_all(node, quote)
    if len(positions) == 1:
        start = positions[0]
        end = start + len(quote)
        mapping_status = "unique"
        overlaps_claim = max(start, claim_start) < min(end, claim_end)
        ranges = [{"start": start, "end_exclusive": end}]
    elif not positions:
        mapping_status = "not_found"
        overlaps_claim = None
        ranges = []
    else:
        mapping_status = "ambiguous_multiple_locations"
        overlaps_claim = None
        ranges = [
            {"start": start, "end_exclusive": start + len(quote)}
            for start in positions
        ]
    return {
        "finding_index": index,
        "finding_sha256": sha256_bytes(raw.encode("utf-8")),
        "family": family,
        "quote_sha256": sha256_bytes(quote.encode("utf-8")),
        "quote_codepoints": len(quote),
        "mapping_status": mapping_status,
        "candidate_ranges": ranges,
        "candidate_overlaps_claim_span": overlaps_claim,
    }


def _valid_trade_date(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = date.fromisoformat(value)
    except ValueError:
        return None
    return value if parsed.isoformat() == value else None


def scan_selected_cases(
    loaded: LoadedBundle,
    scanner: Callable[..., list[str]],
    on_scanner_start: Callable[[], None] | None = None,
) -> list[dict[str, Any]]:
    profile = require_dict(loaded.profile, "profile")
    if profile.get("session_note_template") != SESSION_NOTE_TEMPLATE:
        raise ContractViolation(
            "REFERENCE_MISMATCH",
            "profile.session_note_template",
            "Session note template differs from the approved contract",
        )
    candidate_ids = profile.get("candidate_case_ids")
    if not isinstance(candidate_ids, list):
        raise ContractViolation("INVALID_TYPE", "profile.candidate_case_ids", "Candidate list is missing")
    results: list[dict[str, Any]] = []
    for case_id in sorted(candidate_ids):
        claim = loaded.claims_by_case[case_id]
        decision_id = claim["decision_id"]
        inventory = loaded.inventory_by_id[decision_id]
        node = resolve_json_path(loaded.evidence_by_id[decision_id], claim["path"], f"{case_id}.path")
        base = {
            "case_id": case_id,
            "decision_id": decision_id,
            "label_draft": claim["label"],
            "input_node_sha256": sha256_bytes(node.encode("utf-8")),
            "context_origin": profile.get("context_origin"),
            "replay_mode": profile.get("replay_mode"),
        }
        trade_date = _valid_trade_date(inventory.get("trade_date"))
        ticker = inventory.get("ticker")
        if inventory.get("session_in_progress") is not True:
            results.append({
                **base,
                "execution_status": "not_applicable",
                "n_a_reason": "session_in_progress_not_exact_true",
                "session_note": None,
                "session_note_sha256": None,
                "raw_findings": [],
                "case_findings_sha256": canonical_json_sha256([]),
                "mapping_candidates": [],
            })
            continue
        if trade_date is None or not isinstance(ticker, str) or not ticker:
            results.append({
                **base,
                "execution_status": "not_applicable",
                "n_a_reason": "session_metadata_missing_or_invalid",
                "session_note": None,
                "session_note_sha256": None,
                "raw_findings": [],
                "case_findings_sha256": canonical_json_sha256([]),
                "mapping_candidates": [],
            })
            continue
        note = SESSION_NOTE_TEMPLATE.format(ticker=ticker, trade_date=trade_date)
        try:
            if on_scanner_start is not None:
                on_scanner_start()
            raw_findings = scanner(text=node, session_note=note, agent_name="offline_eval")
        except IsolationViolation:
            raise
        except ContractViolation:
            raise
        except Exception:
            results.append({
                **base,
                "execution_status": "execution_error",
                "n_a_reason": "scanner_execution_error",
                "session_note": note,
                "session_note_sha256": sha256_bytes(note.encode("utf-8")),
                "raw_findings": [],
                "case_findings_sha256": canonical_json_sha256([]),
                "mapping_candidates": [],
            })
            continue
        if not isinstance(raw_findings, list) or any(not isinstance(item, str) for item in raw_findings):
            raise ContractViolation(
                "SCANNER_OUTPUT_INVALID",
                case_id,
                "Scanner must return a list of strings",
            )
        mappings = [
            _parse_finding(
                raw,
                node,
                claim["span_start"],
                claim["span_end_exclusive"],
                index,
            )
            for index, raw in enumerate(raw_findings)
        ]
        results.append({
            **base,
            "execution_status": "scanned",
            "n_a_reason": None,
            "session_note": note,
            "session_note_sha256": sha256_bytes(note.encode("utf-8")),
            "raw_findings": raw_findings,
            "case_findings_sha256": canonical_json_sha256(raw_findings),
            "mapping_candidates": mappings,
        })
    return results
