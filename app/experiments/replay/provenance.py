"""Provenance of a replay run (spec E): what code, model and settings produced it.

The pinned source hashes turn the suite red when a production file the harness copies
from changes (intended coupling; refresh the pin in the stage-7 deploy bundle after
re-checking `runner.py` against the original). Hashes normalise CRLF -> LF so Windows
and the Linux server agree.
운영 소스 해시 고정 + 모델/설정/캐시 키 — 운영 코드가 바뀌면 테스트가 빨개지는 것이 의도.

`d260d27` (the freeze baseline) hashes of the same items, computed with `git show d260d27:<path>`
at implementation time (HEAD `a2b0276`). Only TWO items differ from HEAD, both RM request-capture
hooks (default OFF, `rm_request_capture_enabled`): `app/llm/client.py` (observe_first_native_*
calls around the native request) and `pipeline._run_pipeline_impl` (`capture_research_manager`
context around `research_manager.synthesize`). Every other pinned item is byte-identical, except `app/llm/request_capture.py`, which
DOES NOT EXIST at d260d27 (added later by 56f29ba: capture-only, default OFF; no d260d27 hash).

BASELINE d260d27 hashes (key: sha256):
    file:app/agents/research_manager.py: fbb8b6539705f214fd28e604ea6285d40cc591096a111105f2f684563b19cacf
    file:app/agents/trader.py: b5a855f3911131ad7ca45eb8406b391bee8128aae2af56b637130633345dd33c
    file:app/agents/risk_debators.py: 24273919f51faf1c69cfc772371131fc8fa64b92a7006412d4027c35fa856389
    file:app/agents/debater.py: 688ac81d06465e3f2e1b6b172fa983f7f2da322f8555ea8456423761e7e207f8
    file:app/agents/portfolio_manager.py: dc4eef5c617ec6c4e67eb8ee4ed826d7e66897eed34f17e65c952748d780d6a7
    file:app/agents/untrusted_content.py: a52ca7771d7a543aeffdf0a40d6199e440ea15f56e709577aae1db6087b7b377
    file:app/agents/numeric_guard.py: 06041eeaaad8fe615971a0ce1ac448c30ab1eb348204b099e6bd4bca091577d3
    file:app/agents/unit_scale_guard.py: a980b9b344f8384345daae48aa2cdee93d465399bb93028e1eefc83ebd388030
    file:app/agents/ratio_label_guard.py: 5d54f0cbf2f955ab7c8b7f8ef6fd057c8966c5bb4545b8940bd1756cb93789a0
    file:app/agents/period_mismatch_guard.py: 16f625fc92d269b78a988c963562d39cb3c15752c9122ba9b4514ef1097457eb
    file:app/orchestrator/debate.py: 7816bd46780fa8a32e91d4f5d2085d704b56561a3f3f90a1661a0c99bb906232
    file:app/state.py: 905fcf3783b262ae96edfa615eda3fdc09fa0d3a6f3fef70f92f2347bdeaf0fb
    file:app/llm/__init__.py: 435455faff33c7d083f04acf69cac6957840ed9b3e0dbecd36563cee19568ef2
    file:app/llm/client.py: 7b6e2cd9061d706b3afed0991763f93a4210053abb907fc8033b2c945213c210
    file:app/llm/shared.py: 3ab41476204b56af0ee5215c6b5cb84cd68ea0bc6f3b43e4d02f768f506eb2b9
    file:app/llm/language.py: ce8dec7b8028e13633803f74e74f4165fc89b1a53ebb468592898d0483f0095b
    file:app/llm/leak_markers.py: eabc8eafc554a2e56285c9fc46907d55d66c3dd51ec18c7db57f611ed766de5c
    file:app/llm/metrics.py: 25bf7ee4b5ea5ca922d2579c9c5f908b36debe6749e239dc5757309326b47eb9
    file:app/dataflows/session_state.py: b9a80a050dcbe7f7200d53a760ce0ee84f24af21d60abed066a3b3b7c4bd2bda
    file:app/live/market_hours.py: 26390cfb5f4ad848cf9ca0a055b440c3efc0177e127afb57c115d9651a148545
    file:app/live/kr_holidays.py: 3cdc396499f7a9e73cc367953c99159adf6fbd79155e3b8c55a6e07133a4eca8
    func:app/orchestrator/pipeline.py::_run_pipeline_impl: f26e386b7bcf1294552aa07441fbf99e6b5251ff0195330149659b92b62bb49f
    func:app/orchestrator/pipeline.py::_apply_hysteresis_gate: dabc12f600cba946875cef04c7a451105a532d34a77c5bc35e989f79b3f5470c
    func:app/orchestrator/pipeline.py::_consensus_override_tier: ec3fa3aafa061e2184bd0390cdde8829d8218de19bd763605883fba5d0b7f8f5
    func:app/orchestrator/pipeline.py::_apply_consensus_override_gate: 66a526bd0e59775af713abccc700d848dc3331acd274404d545435f9aabc67e6
    func:app/orchestrator/pipeline.py::_desk_consensus_note: 5112876f4549b1d961e23f500ad5a5313cf7397e1e3aff7bf9566d666755c61a
    func:app/orchestrator/pipeline.py::_degradation_note: 4169e39bca13431bb1ea0d30b99e6c04789d3f4d613add402d4b803cf1de8062
    func:app/orchestrator/pipeline.py::_apply_degradation_notice: d2ea060be50f566082dc2b6aaae90cd28d07ed649345af3c1f9a56a1ce882bd9
    func:app/agents/tool_loop.py::ticker_label: cc89551e886dffaed2a58f7ff1d261fc8ab4f23aed9e1659fdbd5ab358009917

"""

from __future__ import annotations

import hashlib
import inspect
import json
import re
import subprocess
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Mapping

from app.experiments.replay.contract import INPUT_CONTRACT_VERSION, canonical_json

BASELINE_COMMIT = "d260d27"

PINNED_SOURCE_HASHES: dict[str, str] = {
    "file:app/agents/research_manager.py": "fbb8b6539705f214fd28e604ea6285d40cc591096a111105f2f684563b19cacf",
    "file:app/agents/trader.py": "b5a855f3911131ad7ca45eb8406b391bee8128aae2af56b637130633345dd33c",
    "file:app/agents/risk_debators.py": "24273919f51faf1c69cfc772371131fc8fa64b92a7006412d4027c35fa856389",
    "file:app/agents/debater.py": "688ac81d06465e3f2e1b6b172fa983f7f2da322f8555ea8456423761e7e207f8",
    "file:app/agents/portfolio_manager.py": "dc4eef5c617ec6c4e67eb8ee4ed826d7e66897eed34f17e65c952748d780d6a7",
    "file:app/agents/untrusted_content.py": "a52ca7771d7a543aeffdf0a40d6199e440ea15f56e709577aae1db6087b7b377",
    "file:app/agents/numeric_guard.py": "06041eeaaad8fe615971a0ce1ac448c30ab1eb348204b099e6bd4bca091577d3",
    "file:app/agents/unit_scale_guard.py": "a980b9b344f8384345daae48aa2cdee93d465399bb93028e1eefc83ebd388030",
    "file:app/agents/ratio_label_guard.py": "5d54f0cbf2f955ab7c8b7f8ef6fd057c8966c5bb4545b8940bd1756cb93789a0",
    "file:app/agents/period_mismatch_guard.py": "16f625fc92d269b78a988c963562d39cb3c15752c9122ba9b4514ef1097457eb",
    "file:app/orchestrator/debate.py": "7816bd46780fa8a32e91d4f5d2085d704b56561a3f3f90a1661a0c99bb906232",
    "file:app/state.py": "905fcf3783b262ae96edfa615eda3fdc09fa0d3a6f3fef70f92f2347bdeaf0fb",
    "file:app/llm/__init__.py": "435455faff33c7d083f04acf69cac6957840ed9b3e0dbecd36563cee19568ef2",
    "file:app/llm/client.py": "27df7ffe8b73551b07d61786f636880c0a52c5dbe9b014d4631426653a353585",
    "file:app/llm/shared.py": "3ab41476204b56af0ee5215c6b5cb84cd68ea0bc6f3b43e4d02f768f506eb2b9",
    "file:app/llm/language.py": "ce8dec7b8028e13633803f74e74f4165fc89b1a53ebb468592898d0483f0095b",
    "file:app/llm/leak_markers.py": "eabc8eafc554a2e56285c9fc46907d55d66c3dd51ec18c7db57f611ed766de5c",
    "file:app/llm/metrics.py": "25bf7ee4b5ea5ca922d2579c9c5f908b36debe6749e239dc5757309326b47eb9",
    "file:app/llm/request_capture.py": "779630a82394bea65bec902d7960e5bd82184d3a063b70479fc9401824bcb9c1",
    "file:app/dataflows/session_state.py": "b9a80a050dcbe7f7200d53a760ce0ee84f24af21d60abed066a3b3b7c4bd2bda",
    "file:app/live/market_hours.py": "26390cfb5f4ad848cf9ca0a055b440c3efc0177e127afb57c115d9651a148545",
    "file:app/live/kr_holidays.py": "3cdc396499f7a9e73cc367953c99159adf6fbd79155e3b8c55a6e07133a4eca8",
    "func:app/orchestrator/pipeline.py::_run_pipeline_impl": "b1305c2439b6fef281a0604d6bc283c3856a175925407ffeda9f6256020cb6ac",
    "func:app/orchestrator/pipeline.py::_apply_hysteresis_gate": "dabc12f600cba946875cef04c7a451105a532d34a77c5bc35e989f79b3f5470c",
    "func:app/orchestrator/pipeline.py::_consensus_override_tier": "ec3fa3aafa061e2184bd0390cdde8829d8218de19bd763605883fba5d0b7f8f5",
    "func:app/orchestrator/pipeline.py::_apply_consensus_override_gate": "66a526bd0e59775af713abccc700d848dc3331acd274404d545435f9aabc67e6",
    "func:app/orchestrator/pipeline.py::_desk_consensus_note": "5112876f4549b1d961e23f500ad5a5313cf7397e1e3aff7bf9566d666755c61a",
    "func:app/orchestrator/pipeline.py::_degradation_note": "4169e39bca13431bb1ea0d30b99e6c04789d3f4d613add402d4b803cf1de8062",
    "func:app/orchestrator/pipeline.py::_apply_degradation_notice": "d2ea060be50f566082dc2b6aaae90cd28d07ed649345af3c1f9a56a1ce882bd9",
    "func:app/agents/tool_loop.py::ticker_label": "cc89551e886dffaed2a58f7ff1d261fc8ab4f23aed9e1659fdbd5ab358009917",
}

# Pinned files (spec E) — also the file set used for the `git status` / `git diff` checks.
PINNED_FILES: tuple[str, ...] = tuple(k[len("file:"):] for k in PINNED_SOURCE_HASHES if k.startswith("file:"))
PINNED_FUNCS: tuple[tuple[str, str], ...] = tuple(
    tuple(k[len("func:"):].split("::")) for k in PINNED_SOURCE_HASHES if k.startswith("func:")  # type: ignore[misc]
)


def normalize_text(text: str) -> str:
    return text.replace("\r\n", "\n")


def text_hash(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def file_hash(path: Path) -> str:
    return text_hash(path.read_bytes().decode("utf-8"))


def _module_name(rel_path: str) -> str:
    name = rel_path[:-3].replace("/", ".")
    return name[: -len(".__init__")] if name.endswith(".__init__") else name


def function_hash(rel_path: str, func_name: str) -> str:
    import importlib

    module = importlib.import_module(_module_name(rel_path))
    return text_hash(inspect.getsource(getattr(module, func_name)))  # getsource unwraps block stubs


def current_source_hashes(repo_root: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for rel in PINNED_FILES:
        out[f"file:{rel}"] = file_hash(repo_root / rel)
    for rel, func in PINNED_FUNCS:
        out[f"func:{rel}::{func}"] = function_hash(rel, func)
    return out


def source_drift(current: Mapping[str, str], pinned: Mapping[str, str] = PINNED_SOURCE_HASHES) -> list[str]:
    return sorted(k for k in set(pinned) | set(current) if pinned.get(k) != current.get(k))


DRIFT_MESSAGE = (
    "운영 파일이 바뀜 → runner 조합을 원본과 다시 대조한 뒤 pin 갱신 "
    "(a production file the harness copies from changed: re-check runner.py against the original, then refresh the pin)"
)


# --------------------------------------------------------------------------- git


def _git(repo_root: Path, *args: str, runner: Callable[..., Any] = subprocess.run) -> subprocess.CompletedProcess:
    return runner(["git", *args], cwd=repo_root, capture_output=True, text=True, timeout=60)


def git_state(repo_root: Path, *, runner: Callable[..., Any] = subprocess.run) -> dict[str, Any]:
    """HEAD, whether the pinned files are dirty, and the porcelain lines for them."""
    head = _git(repo_root, "rev-parse", "HEAD", runner=runner)
    status = _git(repo_root, "status", "--porcelain", "--", *_diff_files(), runner=runner)
    lines = [ln for ln in status.stdout.splitlines() if ln.strip()]
    return {
        "head": head.stdout.strip() if head.returncode == 0 else None,
        "pinned_files_dirty": bool(lines),
        "dirty_lines": lines,
    }


def _diff_files() -> tuple[str, ...]:
    return (*PINNED_FILES, "app/orchestrator/pipeline.py", "app/agents/tool_loop.py")


def commits_differ(repo_root: Path, a: str, b: str, *, runner: Callable[..., Any] = subprocess.run) -> bool:
    """`git diff --quiet <a> <b> -- <pinned files>`: True when the pinned files differ."""
    proc = _git(repo_root, "diff", "--quiet", a, b, "--", *_diff_files(), runner=runner)
    if proc.returncode not in (0, 1):
        raise RuntimeError(f"git diff failed for {a}..{b}: {proc.stderr.strip()}")
    return proc.returncode == 1


def commit_differs_from_baseline(repo_root: Path, commit: str, *, runner: Callable[..., Any] = subprocess.run) -> bool:
    return commits_differ(repo_root, BASELINE_COMMIT, commit, runner=runner)


def classify_commit(repo_root: Path, commit: str, *, runner: Callable[..., Any] = subprocess.run) -> dict[str, Any]:
    """Era-drift rule (review M3): a freeze-era commit is DRIFT only if its downstream files differ
    from the baseline `d260d27` AND ALSO differ from HEAD (what the harness actually runs, pinned by
    hashes). A commit identical to either one is not drift. An unknown commit is drift."""
    try:
        vs_base = commits_differ(repo_root, BASELINE_COMMIT, commit, runner=runner)
        vs_head = commits_differ(repo_root, "HEAD", commit, runner=runner)
    except (RuntimeError, OSError) as exc:
        return {"vs_baseline": None, "vs_head": None, "drift": True, "note": f"unknown commit ({exc})"}
    return {"vs_baseline": vs_base, "vs_head": vs_head, "drift": vs_base and vs_head}


# --------------------------------------------------------------------------- model info

Fetch = Callable[[str, float], tuple[int, bytes]]


def _http_get(url: str, timeout: float) -> tuple[int, bytes]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310 - fixed http(s) endpoint from settings
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, b""


def model_info(base_url: str, *, fetch: Fetch = _http_get, timeout: float = 15.0) -> dict[str, Any]:
    """`GET {base}/models` (id, max_model_len, response sha256) and `GET .../version` when it
    answers 200. These are plain GETs, not completions."""
    base = base_url.rstrip("/")
    status, body = fetch(f"{base}/models", timeout)
    if status != 200:
        raise RuntimeError(f"GET {base}/models returned HTTP {status}")
    payload = json.loads(body.decode("utf-8"))
    data = payload.get("data") or []
    if not data:
        raise RuntimeError("GET /models returned no models")
    first = data[0]
    info: dict[str, Any] = {
        "id": first.get("id"),
        "max_model_len": first.get("max_model_len"),
        "all_ids": [d.get("id") for d in data],
        "models_response_sha256": hashlib.sha256(body).hexdigest(),
        "version": None,
    }
    root = re.sub(r"/v1$", "", base)
    for url in (f"{base}/version", f"{root}/version"):
        try:
            code, vbody = fetch(url, timeout)
        except Exception:  # noqa: BLE001 - optional field
            continue
        if code == 200:
            info["version"] = vbody.decode("utf-8", "replace")[:200]
            break
    return info


# --------------------------------------------------------------------------- keys


def compute_base_key(
    *, source_hashes: Mapping[str, str], pinned_values: Mapping[str, Any], model: Mapping[str, Any],
    caseset_sha256: str, contract_version: str = INPUT_CONTRACT_VERSION,
) -> str:
    """Computed BEFORE a variant is applied. git commit, dirty flag, K and harness version are
    recorded elsewhere but deliberately NOT part of the key (unrelated commits must not
    invalidate the baseline; the source hashes are the real criterion)."""
    payload = {
        "contract_version": contract_version,
        "source_hashes": dict(sorted(source_hashes.items())),
        "pinned_values": dict(sorted(pinned_values.items())),
        "model": {"id": model.get("id"), "max_model_len": model.get("max_model_len")},
        "caseset_sha256": caseset_sha256,
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def make_run_id(arm: str, repeat: int, base_key: str, tag: str | None = None) -> str:
    rid = f"{arm}__r{repeat}__{base_key[:12]}"
    return f"{tag}__{rid}" if tag else rid


# --------------------------------------------------------------------------- node fingerprints

NODES = ("rm", "trader", "risk_debate", "pm", "gates")

COMMON_MODULES = (
    "app.llm.client", "app.llm.shared", "app.llm.language", "app.agents.untrusted_content",
    "app.agents.debater", "app.agents.numeric_guard", "app.agents.unit_scale_guard",
    "app.agents.ratio_label_guard", "app.agents.period_mismatch_guard", "app.state",
)
COMMON_SETTINGS = (
    "llm_provider", "llm_model", "llm_max_output_tokens", "llm_strip_reasoning_leak",
    "injection_flag_rest_guard",
)
NODE_MODULES: dict[str, tuple[str, ...]] = {
    "rm": ("app.agents.research_manager",),
    "trader": ("app.agents.trader",),
    "risk_debate": ("app.agents.risk_debators", "app.orchestrator.debate"),
    "pm": ("app.agents.portfolio_manager",),
    "gates": (),
}
NODE_FUNCTIONS: dict[str, tuple[tuple[str, str], ...]] = {
    "rm": (), "trader": (), "risk_debate": (),
    "pm": (
        ("app.orchestrator.pipeline", "_desk_consensus_note"),
        ("app.orchestrator.pipeline", "_degradation_note"),
    ),
    "gates": (
        ("app.orchestrator.pipeline", "_apply_hysteresis_gate"),
        ("app.orchestrator.pipeline", "_apply_consensus_override_gate"),
        ("app.orchestrator.pipeline", "_apply_unjustified_exit_gate"),
        ("app.orchestrator.pipeline", "_apply_degradation_notice"),
        ("app.orchestrator.pipeline", "_consensus_override_tier"),
    ),
}
NODE_SETTINGS: dict[str, tuple[str, ...]] = {
    "rm": ("research_manager_reasoning",),
    "trader": ("ai_profile",),
    "risk_debate": (
        "max_risk_rounds", "debate_parallel_first_round", "debate_opening_turn_prompt",
        "debate_own_previous_turn", "debate_round1_max_output_tokens",
    ),
    "pm": ("ai_profile",),
    "gates": (
        "hysteresis_min_confidence_to_flip", "consensus_override_min_confidence",
        "consensus_override_min_confidence_bearish",
    ),
}


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _canon_value(value: Any, depth: int = 0) -> str:
    """Canonical text for a module-level value (spec E fingerprint definition)."""
    import re as _re

    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        return repr(value)
    if isinstance(value, _re.Pattern):
        return f"re:{value.pattern!r}:{int(value.flags)}"
    if depth < 4:
        if isinstance(value, (list, tuple)):
            return "[" + ",".join(_canon_value(v, depth + 1) for v in value) + "]"
        if isinstance(value, (set, frozenset)):
            return "{" + ",".join(sorted(_canon_value(v, depth + 1) for v in value)) + "}"
        if isinstance(value, dict):
            items = sorted((_canon_value(k, depth + 1), _canon_value(v, depth + 1)) for k, v in value.items())
            return "{" + ",".join(f"{k}:{v}" for k, v in items) + "}"
    return f"<{type(value).__name__}>"


def _callable_digest(obj: Any, owner: str) -> str:
    if getattr(obj, "__module__", None) == owner:
        try:
            return _digest(normalize_text(inspect.getsource(obj)))
        except (OSError, TypeError):
            return _digest(f"nosource:{getattr(obj, '__qualname__', '?')}")
    return f"{getattr(obj, '__module__', '?')}.{getattr(obj, '__qualname__', type(obj).__name__)}"


def module_digest(module: Any) -> str:
    parts: list[str] = []
    for name in sorted(vars(module)):
        if name.startswith("__"):
            continue
        value = vars(module)[name]
        if inspect.ismodule(value):
            parts.append(f"{name}=module:{value.__name__}")
        elif inspect.isfunction(value) or inspect.isclass(value):
            parts.append(f"{name}={_callable_digest(value, module.__name__)}")
        else:
            parts.append(f"{name}={_digest(_canon_value(value))}")
    return _digest("\n".join(parts))


def node_fingerprints(settings_obj: Any, harness_digests: Mapping[str, str] | None = None) -> dict[str, dict[str, Any]]:
    """Per-node fingerprint = digests of the node's modules (+ the common ones), its named pipeline
    functions, the settings it reads, and the harness node function. `parts` lets a diff say WHY."""
    import importlib

    harness_digests = harness_digests or {}
    out: dict[str, dict[str, Any]] = {}
    for node in NODES:
        parts: dict[str, str] = {}
        for mod_name in (*COMMON_MODULES, *NODE_MODULES[node]):
            parts[f"module:{mod_name}"] = module_digest(importlib.import_module(mod_name))
        for mod_name, func in NODE_FUNCTIONS[node]:
            obj = getattr(importlib.import_module(mod_name), func)
            parts[f"func:{mod_name}.{func}"] = _digest(normalize_text(inspect.getsource(obj)))
        for key in (*COMMON_SETTINGS, *NODE_SETTINGS[node]):
            value = getattr(settings_obj, key, None)
            parts[f"setting:{key}"] = _digest(_canon_value(getattr(value, "value", value)))
        parts["harness"] = harness_digests.get(node, "")
        out[node] = {"digest": _digest(canonical_json(parts)), "parts": parts}
    return out


def fingerprint_diff(a: Mapping[str, Any], b: Mapping[str, Any]) -> list[str]:
    """Names of the parts that differ between two node fingerprints."""
    pa, pb = a.get("parts", {}), b.get("parts", {})
    return sorted(k for k in set(pa) | set(pb) if pa.get(k) != pb.get(k))
