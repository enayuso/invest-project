"""app.experiments.replay.provenance — source-hash pin, keys, fingerprints (spec E, G).
Synthetic only: model info uses a fake fetcher, git uses a fake runner."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from app.experiments.replay import provenance as prov
from replay_fixtures import install_db_block

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _block_db(monkeypatch):
    install_db_block(monkeypatch)


def test_source_hash_drift():
    current = prov.current_source_hashes(REPO)
    drift = prov.source_drift(current)
    assert drift == [], f"{prov.DRIFT_MESSAGE}: {drift}"
    assert set(current) == set(prov.PINNED_SOURCE_HASHES) and len(current) == 30


def test_drift_is_detected():
    current = dict(prov.current_source_hashes(REPO))
    key = "file:app/agents/research_manager.py"
    current[key] = "0" * 64
    assert prov.source_drift(current) == [key]
    del current[key]
    assert prov.source_drift(current) == [key]


def test_crlf_normalized():
    assert prov.text_hash("a\r\nb") == prov.text_hash("a\nb")
    assert prov.text_hash("a\nb") != prov.text_hash("a\nc")


def test_baseline_docstring_records_the_two_capture_differences():
    doc = prov.__doc__
    assert "d260d27" in doc and "request-capture" in doc
    assert "func:app/orchestrator/pipeline.py::_run_pipeline_impl" in doc
    assert "file:app/llm/client.py" in doc


def _key(**over):
    args = dict(
        source_hashes={"file:a": "1", "file:b": "2"},
        pinned_values={"MAX_RISK_ROUNDS": 2, "HYSTERESIS_MIN_CONFIDENCE_TO_FLIP": 0.6},
        model={"id": "m", "max_model_len": 1000},
        caseset_sha256="abc",
    )
    args.update(over)
    return prov.compute_base_key(**args)


def test_base_key_changes():
    base = _key()
    assert _key(model={"id": "other", "max_model_len": 1000}) != base
    assert _key(model={"id": "m", "max_model_len": 2000}) != base
    assert _key(pinned_values={"MAX_RISK_ROUNDS": 3, "HYSTERESIS_MIN_CONFIDENCE_TO_FLIP": 0.6}) != base
    assert _key(source_hashes={"file:a": "1", "file:b": "changed"}) != base
    assert _key(caseset_sha256="other") != base
    # key order does not matter, the model response hash / version do not enter the key
    assert _key(source_hashes={"file:b": "2", "file:a": "1"}) == base
    assert _key(model={"id": "m", "max_model_len": 1000, "version": "x", "models_response_sha256": "y"}) == base
    # git commit / dirty flag / K are not inputs at all (signature has no such parameter)
    with pytest.raises(TypeError):
        prov.compute_base_key(  # type: ignore[call-arg]
            source_hashes={}, pinned_values={}, model={}, caseset_sha256="", git_commit="abc",
        )


def test_run_id():
    key = "f" * 64
    assert prov.make_run_id("baseline", 1, key) == "baseline__r1__ffffffffffff"
    assert prov.make_run_id("baseline", 2, key, tag="smoke") == "smoke__baseline__r2__ffffffffffff"
    assert ":" not in prov.make_run_id("baseline", 1, key, tag="pilot_k4")


def test_model_info_with_fake_fetch():
    calls = []

    def fetch(url, timeout):
        calls.append(url)
        if url.endswith("/models"):
            body = json.dumps({"data": [{"id": "nvidia/Qwen3.6-35B-A3B-NVFP4", "max_model_len": 262144}]})
            return 200, body.encode()
        if url == "http://h:1/v1/version":
            return 404, b""
        if url == "http://h:1/version":
            return 200, b'{"version":"0.20"}'
        return 500, b""

    info = prov.model_info("http://h:1/v1/", fetch=fetch)
    assert info["id"] == "nvidia/Qwen3.6-35B-A3B-NVFP4" and info["max_model_len"] == 262144
    assert info["version"] == '{"version":"0.20"}' and len(info["models_response_sha256"]) == 64
    assert calls[0] == "http://h:1/v1/models"

    with pytest.raises(RuntimeError, match="HTTP 503"):
        prov.model_info("http://h:1/v1", fetch=lambda u, t: (503, b""))


class _Runner:
    def __init__(self, returncode=0, stdout=""):
        self.returncode, self.stdout, self.calls = returncode, stdout, []

    def __call__(self, cmd, **kwargs):
        self.calls.append(cmd)
        return subprocess.CompletedProcess(cmd, self.returncode, stdout=self.stdout, stderr="")


def test_git_helpers_use_read_only_commands():
    runner = _Runner(stdout=" M app/agents/trader.py\n")
    state = prov.git_state(REPO, runner=runner)
    assert state["pinned_files_dirty"] is True and state["dirty_lines"] == [" M app/agents/trader.py"]
    assert all(c[1] in {"rev-parse", "status"} for c in runner.calls)

    same = _Runner(returncode=0)
    assert prov.commit_differs_from_baseline(REPO, "abc1234", runner=same) is False
    assert same.calls[0][:4] == ["git", "diff", "--quiet", "d260d27"]
    assert prov.commit_differs_from_baseline(REPO, "abc1234", runner=_Runner(returncode=1)) is True
    with pytest.raises(RuntimeError):
        prov.commit_differs_from_baseline(REPO, "abc1234", runner=_Runner(returncode=128))


def test_node_fingerprints_stable_and_sensitive(monkeypatch):
    from app.agents import portfolio_manager, research_manager
    from app.config import settings

    base = prov.node_fingerprints(settings, {"rm": "h"})
    again = prov.node_fingerprints(settings, {"rm": "h"})
    assert {n: v["digest"] for n, v in base.items()} == {n: v["digest"] for n, v in again.items()}
    assert set(base) == set(prov.NODES)

    # a PM constant change touches ONLY the pm node
    monkeypatch.setattr(portfolio_manager, "_FLAT_SUFFIX", portfolio_manager._FLAT_SUFFIX + "\n[x]")
    changed = prov.node_fingerprints(settings, {"rm": "h"})
    differing = [n for n in prov.NODES if changed[n]["digest"] != base[n]["digest"]]
    assert differing == ["pm"]
    assert prov.fingerprint_diff(base["pm"], changed["pm"]) == ["module:app.agents.portfolio_manager"]

    # an RM constant change touches only rm
    monkeypatch.undo()
    monkeypatch.setattr(research_manager, "_FLAT_SUFFIX", research_manager._FLAT_SUFFIX + " y")
    changed = prov.node_fingerprints(settings, {"rm": "h"})
    assert [n for n in prov.NODES if changed[n]["digest"] != base[n]["digest"]] == ["rm"]

    # a shared (common) module change touches every node
    monkeypatch.undo()
    from app.agents import debater

    monkeypatch.setattr(debater, "render_reports", lambda reports, identity="": "x")
    changed = prov.node_fingerprints(settings, {"rm": "h"})
    assert all(changed[n]["digest"] != base[n]["digest"] for n in prov.NODES)


def test_settings_change_touches_the_reading_node(monkeypatch):
    from app.config import settings

    base = prov.node_fingerprints(settings)
    monkeypatch.setattr(settings, "hysteresis_min_confidence_to_flip", 0.65)
    changed = prov.node_fingerprints(settings)
    assert [n for n in prov.NODES if changed[n]["digest"] != base[n]["digest"]] == ["gates"]


def test_request_capture_is_pinned_and_documented_as_absent_at_baseline():
    assert "file:app/llm/request_capture.py" in prov.PINNED_SOURCE_HASHES
    assert "DOES NOT EXIST at d260d27" in prov.__doc__


class _DiffRunner:
    """Fake `git diff --quiet`: exit 1 (differs) when the {a}..{b} pair is in `differs`, else 0."""

    def __init__(self, differs, fail=()):
        self.differs, self.fail, self.calls = set(differs), set(fail), []

    def __call__(self, cmd, **kwargs):
        self.calls.append(cmd)
        assert cmd[:3] == ["git", "diff", "--quiet"]
        pair = (cmd[3], cmd[4])
        if pair in self.fail:
            return subprocess.CompletedProcess(cmd, 128, stdout="", stderr="fatal: bad revision")
        return subprocess.CompletedProcess(cmd, 1 if pair in self.differs else 0, stdout="", stderr="")


def test_classify_commit_drift_needs_a_difference_from_both_baseline_and_head():
    base = prov.BASELINE_COMMIT
    # same as the baseline (only later, unrelated work differs from HEAD) -> not drift
    same_as_base = _DiffRunner(differs={("HEAD", "82dac90")})
    assert prov.classify_commit(REPO, "82dac90", runner=same_as_base) == {"vs_baseline": False, "vs_head": True, "drift": False}
    # same as HEAD (what the harness runs) but different from the baseline -> not drift
    same_as_head = _DiffRunner(differs={(base, "9a787bc")})
    assert prov.classify_commit(REPO, "9a787bc", runner=same_as_head) == {"vs_baseline": True, "vs_head": False, "drift": False}
    # different from both -> drift
    both = _DiffRunner(differs={(base, "aaaaaaa"), ("HEAD", "aaaaaaa")})
    assert prov.classify_commit(REPO, "aaaaaaa", runner=both)["drift"] is True
    # identical to both baseline and HEAD -> not drift
    assert prov.classify_commit(REPO, "bbbbbbb", runner=_DiffRunner(differs=())) ["drift"] is False
    # an unknown commit cannot be classified: drift, with a note
    unknown = prov.classify_commit(REPO, "deadbee", runner=_DiffRunner(differs=(), fail={(base, "deadbee")}))
    assert unknown["drift"] is True and unknown["vs_baseline"] is None and "unknown commit" in unknown["note"]
    # every diff is limited to the pinned file set (incl. request_capture.py)
    assert all("app/llm/request_capture.py" in c for c in same_as_base.calls)
