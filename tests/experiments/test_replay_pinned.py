"""app.experiments.replay.pinned — stdlib-only pinned-settings loader (spec A.2/E).
Synthetic only: no DB, no LLM, no network."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from app.experiments.replay import pinned


@pytest.fixture(autouse=True)
def _block_db(monkeypatch):
    """conftest does not block the DB; make any accidental connection raise (spec G)."""
    import asyncpg

    def _boom(*args, **kwargs):
        raise AssertionError("DB access attempted in a replay test")

    monkeypatch.setattr(asyncpg, "connect", _boom)
    monkeypatch.setattr(asyncpg, "create_pool", _boom)
    import app.memory.db as memory_db

    monkeypatch.setattr(memory_db, "get_pool", _boom)


def _doc() -> dict:
    return copy.deepcopy(pinned.default_document())


def test_default_document_loads_and_is_consistent():
    p = pinned.parse(_doc())
    assert p.values["LLM_MODEL"] == "nvidia/Qwen3.6-35B-A3B-NVFP4"
    assert p.sources["LLM_MODEL"] == "server_env"  # user confirmed
    assert p.values["RM_REQUEST_CAPTURE_ENABLED"] is False
    assert set(p.values) == set(pinned.KEYED_KEYS)
    assert p.keyed_values() == dict(sorted(p.values.items()))


def test_missing_key_refused():
    doc = _doc()
    del doc["settings"]["MAX_RISK_ROUNDS"]
    with pytest.raises(pinned.PinnedError, match="missing required keys"):
        pinned.parse(doc)
    doc = _doc()
    doc["settings"]["MAX_RISK_ROUNDS"]["value"] = None
    with pytest.raises(pinned.PinnedError, match="no value"):
        pinned.parse(doc)


@pytest.mark.parametrize("key", ["DATABASE_URL", "LLM_API_KEY", "KIS_APP_SECRET", "SOME_TOKEN", "LLM_BASE_URL"])
def test_unknown_or_secret_key_refused(key):
    doc = _doc()
    doc["settings"][key] = {"value": "x", "source": "server_env", "config_line": None}
    with pytest.raises(pinned.PinnedError, match="secret-looking"):
        pinned.parse(doc)
    doc = _doc()
    doc["settings"]["SOME_OTHER"] = {"value": 1, "source": "server_env", "config_line": None}
    with pytest.raises(pinned.PinnedError, match="allow-list"):
        pinned.parse(doc)
    doc = _doc()
    doc["record_only"][key] = {"value": "x"}
    with pytest.raises(pinned.PinnedError, match="secret-looking"):
        pinned.parse(doc)


def test_token_named_required_keys_are_not_mistaken_for_secrets():
    # LLM_MAX_OUTPUT_TOKENS contains "TOKEN" but is an allow-listed required key.
    assert "LLM_MAX_OUTPUT_TOKENS" in pinned.parse(_doc()).values


def test_code_default_drift_refused():
    class FakeField:
        def __init__(self, default):
            self.default = default

    fields = {k.lower(): FakeField(v[1]) for k, v in pinned._ROWS.items()}
    p = pinned.parse(_doc())
    assert pinned.check_code_defaults(p, fields) == []

    doc = _doc()
    doc["settings"]["HYSTERESIS_MIN_CONFIDENCE_TO_FLIP"]["value"] = 0.65  # pinned says 0.65 ...
    drifted = pinned.parse(doc)
    problems = pinned.check_code_defaults(drifted, fields)  # ... but the code default is 0.6
    assert problems and "HYSTERESIS_MIN_CONFIDENCE_TO_FLIP" in problems[0]

    # server_env entries are exempt from the code-default comparison
    fields["llm_model"] = FakeField("Qwen/Qwen3.6-35B-A3B-FP8")
    assert pinned.check_code_defaults(p, fields) == []


def test_code_defaults_match_real_settings_fields():
    """Every `code_default` row equals the real Settings default (imports app.config
    here, in a test process, never in the CLI module)."""
    from app.config import Settings

    p = pinned.parse(_doc())
    assert pinned.check_code_defaults(p, Settings.model_fields) == []


def test_apply_to_environ():
    p = pinned.parse(_doc())
    env: dict[str, str] = {}
    applied = pinned.apply_to_environ(p, k=4, environ=env)
    assert env["LLM_MAX_CONCURRENT_REQUESTS"] == "4"
    assert env["DEBATE_PARALLEL_FIRST_ROUND"] == "false"
    assert env["LLM_STRIP_REASONING_LEAK"] == "true"
    assert env["HYSTERESIS_MIN_CONFIDENCE_TO_FLIP"] == "0.6"
    assert env["LLM_MODEL"] == "nvidia/Qwen3.6-35B-A3B-NVFP4"
    assert applied == env
    assert "AUTO_TRADE_MAX_CONCURRENT_ANALYSES" not in env  # record-only
    assert "KIS_ACCOUNT_MODE" not in env
    with pytest.raises(pinned.PinnedError):
        pinned.apply_to_environ(p, k=0, environ={})


def test_verify_mismatch():
    p = pinned.parse(_doc())

    class FakeSettings:
        ai_profile = type("E", (), {"value": "production"})()
        max_debate_rounds = 2
        max_risk_rounds = 2
        llm_model = "nvidia/Qwen3.6-35B-A3B-NVFP4"
        llm_provider = "vllm"
        llm_max_output_tokens = 16000
        debate_round1_max_output_tokens = 1800
        llm_strip_reasoning_leak = True
        research_manager_reasoning = True
        injection_flag_rest_guard = True
        debate_parallel_first_round = False
        debate_opening_turn_prompt = True
        debate_own_previous_turn = True
        hysteresis_min_confidence_to_flip = 0.6
        consensus_override_min_confidence = 0.75
        consensus_override_min_confidence_bearish = 0.85
        unjustified_exit_gate_enabled = True
        rm_request_capture_enabled = False

    assert pinned.verify_against_settings(p, FakeSettings) == []
    FakeSettings.max_risk_rounds = 3
    FakeSettings.hysteresis_min_confidence_to_flip = 0.65
    problems = pinned.verify_against_settings(p, FakeSettings)
    assert len(problems) == 2 and any("MAX_RISK_ROUNDS" in x for x in problems)


def test_capture_enabled_refused():
    doc = _doc()
    doc["settings"]["RM_REQUEST_CAPTURE_ENABLED"]["value"] = True
    with pytest.raises(pinned.PinnedError, match="must be false"):
        pinned.parse(doc)


def test_type_and_schema_checks(tmp_path):
    doc = _doc()
    doc["settings"]["MAX_DEBATE_ROUNDS"]["value"] = "2"
    with pytest.raises(pinned.PinnedError, match="expected integer"):
        pinned.parse(doc)
    doc = _doc()
    doc["settings"]["DEBATE_PARALLEL_FIRST_ROUND"]["value"] = 0
    with pytest.raises(pinned.PinnedError, match="boolean"):
        pinned.parse(doc)
    doc = _doc()
    doc["schema"] = "other"
    with pytest.raises(pinned.PinnedError, match="schema"):
        pinned.parse(doc)
    doc = _doc()
    doc["settings"]["LLM_MODEL"]["source"] = "guess"
    with pytest.raises(pinned.PinnedError, match="source"):
        pinned.parse(doc)

    path = tmp_path / "p.json"
    path.write_text(json.dumps(_doc()), encoding="utf-8")
    loaded = pinned.load(path)
    assert len(loaded.sha256) == 64
    with pytest.raises(pinned.PinnedError, match="cannot read"):
        pinned.load(tmp_path / "missing.json")
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(pinned.PinnedError, match="invalid JSON"):
        pinned.load(path)


def test_pinned_module_does_not_load_config():
    code = (
        "import sys, app.experiments.replay.pinned as p; "
        "assert 'app.config' not in sys.modules; "
        "p.parse(p.default_document())"
    )
    repo = Path(__file__).resolve().parents[2]
    proc = subprocess.run([sys.executable, "-c", code], cwd=repo, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


def test_cli_import_does_not_load_config():
    code = "import sys, app.experiments.replay_downstream; assert 'app.config' not in sys.modules"
    repo = Path(__file__).resolve().parents[2]
    proc = subprocess.run([sys.executable, "-c", code], cwd=repo, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
