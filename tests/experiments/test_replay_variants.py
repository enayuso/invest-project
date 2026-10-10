"""app.experiments.replay.variants — fork reuse and the fingerprint guards (spec B, G).
Real agent code, fake LLM, blocked DB; the baseline run and the variant run share a tmp dir."""

from __future__ import annotations

import copy
import json
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from app.agents import portfolio_manager, research_manager
from app.config import settings
from app.experiments.replay import provenance, runner, variants
from app.llm import llm_client
from replay_fixtures import FakeLLM, install_db_block, make_inputs, make_row, make_transcript

KST = ZoneInfo("Asia/Seoul")
SATURDAY = lambda: datetime(2026, 9, 26, 10, 0, tzinfo=KST)  # noqa: E731


@pytest.fixture(autouse=True)
def _block_db(monkeypatch):
    install_db_block(monkeypatch)


@pytest.fixture
def fake(monkeypatch):
    fake = FakeLLM()
    monkeypatch.setattr(llm_client, "complete", fake.complete)
    monkeypatch.setattr(llm_client, "structured_complete", fake.structured_complete)
    instrumentation = runner.Instrumentation()
    instrumentation.install()
    yield fake
    instrumentation.uninstall()


def _cases(n=2):
    return [make_inputs(make_row(id=i, ticker=f"00000{i}.KS", transcript=make_transcript())) for i in range(1, n + 1)]


def _cfg(run_dir, **kw):
    return runner.RunConfig(run_dir=run_dir, run_id=kw.pop("run_id", "r"), echo=None, now_fn=SATURDAY, heartbeat_s=3600, **kw)


async def _baseline(tmp_path, cases):
    cfg = _cfg(tmp_path / "baseline")
    fingerprints = provenance.node_fingerprints(settings, runner.harness_node_digests())
    session = runner.RunSession(cfg, {
        "run_id": "r", "base_key": "K", "arm": "baseline", "variant": "baseline", "node_fingerprints": fingerprints,
    })
    out = await session.run(cases)
    assert out.status == "done"
    return cfg.run_dir, fingerprints


def _reader(run_dir):
    return lambda case_id: runner.read_ok_case(run_dir, case_id)


def _strip_volatile(block):
    block = copy.deepcopy(block)
    for key in ("source", "wall_s", "calls", "tokens", "baseline_wall_s"):
        block.pop(key, None)
    return block


async def test_fork_reuses_baseline(fake, monkeypatch, tmp_path):
    cases = _cases(2)
    base_dir, base_fp = await _baseline(tmp_path, cases)
    baseline_calls = len(fake.calls)
    assert baseline_calls == 2 * 9

    variant = variants.get_variant("_test_pm_marker", allow_test_variant=True)
    ctx = variants.apply_variant(variant, setattr_fn=monkeypatch.setattr)
    assert ctx.patches and ctx.patches[0]["target"].endswith("_FLAT_SUFFIX")
    assert ctx.patches[0]["before"] != ctx.patches[0]["after"]
    current_fp = provenance.node_fingerprints(settings, runner.harness_node_digests())
    variants.validate_fork(
        variant=variant, base_key="K",
        baseline_manifest={"base_key": "K", "status": "done", "node_fingerprints": base_fp},
        current_fingerprints=current_fp,
    )

    cfg = _cfg(tmp_path / "variant", variant=variant.name, fork_at=variant.fork_at, baseline_reader=_reader(base_dir))
    session = runner.RunSession(cfg, {"run_id": "r", "base_key": "K", "arm": "variant", "variant": variant.name})
    out = await session.run(cases)
    assert out.status == "done" and out.done == 2

    new_calls = fake.calls[baseline_calls:]
    assert {c["label"] for c in new_calls} == {"portfolio_manager"} and len(new_calls) == 2
    assert all("[replay-test-variant]" in c["messages"][0]["content"] for c in new_calls)

    for case in cases:
        base_case = runner.read_ok_case(base_dir, case.case_id)
        var_case = runner.read_ok_case(cfg.run_dir, case.case_id)
        for node in ("rm", "trader", "risk_debate"):
            assert var_case["nodes"][node]["source"] == "baseline_cache"
            assert base_case["nodes"][node]["source"] == "llm"
            assert _strip_volatile(var_case["nodes"][node]) == _strip_volatile(base_case["nodes"][node])
        assert var_case["nodes"]["pm"]["source"] == "llm" and var_case["nodes"]["gates"]["source"] == "llm"
        assert var_case["input_hash"] == base_case["input_hash"]
        # LLM-cost accounting counts only the calls made in THIS run
        assert var_case["calls_summary"]["n"] == 1 and var_case["tokens_total"] == 10
        assert var_case["summary"]["rm"] == base_case["summary"]["rm"]
        assert var_case["fork_at"] == "pm" and var_case["variant"] == "_test_pm_marker"

    # the PM saw the same user prompt in both arms (only the system prompt carries the marker)
    base_pm = [c for c in fake.calls[:baseline_calls] if c["label"] == "portfolio_manager"]
    assert [c["messages"][1] for c in base_pm] == [c["messages"][1] for c in new_calls]
    assert all(c["messages"][0]["content"] + "\n[replay-test-variant]" == n["messages"][0]["content"]
               for c, n in zip(base_pm, new_calls))


async def test_fingerprint_guard(fake, monkeypatch, tmp_path):
    cases = _cases(1)
    base_dir, base_fp = await _baseline(tmp_path, cases)
    manifest = {"base_key": "K", "status": "done", "node_fingerprints": base_fp}

    # a variant that also touches a node reused from the cache (RM constant + fork_at pm)
    def touches_rm(ctx):
        ctx.patch(research_manager, "_FLAT_SUFFIX", research_manager._FLAT_SUFFIX + " changed")
        ctx.patch(portfolio_manager, "_FLAT_SUFFIX", portfolio_manager._FLAT_SUFFIX + " changed")

    bad = variants.Variant("bad", "pm", "touches rm", touches_rm)
    variants.apply_variant(bad, setattr_fn=monkeypatch.setattr)
    fp = provenance.node_fingerprints(settings, runner.harness_node_digests())
    with pytest.raises(variants.VariantRefused, match="reused from the baseline cache"):
        variants.validate_fork(variant=bad, base_key="K", baseline_manifest=manifest, current_fingerprints=fp)
    monkeypatch.undo()

    # a no-op variant (same value written back) is refused unless explicitly allowed
    noop = variants.Variant("noop", "pm", "no-op", lambda ctx: ctx.patch(portfolio_manager, "_FLAT_SUFFIX", portfolio_manager._FLAT_SUFFIX))
    variants.apply_variant(noop, setattr_fn=monkeypatch.setattr)
    fp = provenance.node_fingerprints(settings, runner.harness_node_digests())
    with pytest.raises(variants.VariantRefused, match="no-op"):
        variants.validate_fork(variant=noop, base_key="K", baseline_manifest=manifest, current_fingerprints=fp)
    variants.validate_fork(variant=noop, base_key="K", baseline_manifest=manifest, current_fingerprints=fp, allow_noop=True)
    monkeypatch.undo()

    # base_key / status of the baseline are checked first
    good = variants.get_variant("_test_pm_marker", allow_test_variant=True)
    with pytest.raises(variants.VariantRefused, match="base_key"):
        variants.validate_fork(variant=good, base_key="OTHER", baseline_manifest=manifest, current_fingerprints=fp)
    with pytest.raises(variants.VariantRefused, match="status"):
        variants.validate_fork(
            variant=good, base_key="K", baseline_manifest={**manifest, "status": "paused_window"}, current_fingerprints=fp,
        )
    assert base_dir.exists()


def test_test_variants_need_the_flag_and_unknown_is_refused():
    with pytest.raises(variants.VariantRefused, match="test-only"):
        variants.get_variant("_test_pm_marker")
    with pytest.raises(variants.VariantRefused, match="unknown variant"):
        variants.get_variant("nope")
    assert variants.get_variant("baseline").fork_at == "rm"
    assert variants.get_variant("_test_pm_marker", allow_test_variant=True).fork_at == "pm"


async def test_input_hash_mismatch_refused(fake, monkeypatch, tmp_path):
    cases = _cases(2)
    base_dir, _ = await _baseline(tmp_path, cases)
    tampered = base_dir / "cases" / f"{cases[0].case_id}.json"
    data = json.loads(tampered.read_text(encoding="utf-8"))
    data["input_hash"] = "0" * 64
    tampered.write_text(json.dumps(data), encoding="utf-8")

    variant = variants.get_variant("_test_pm_marker", allow_test_variant=True)
    variants.apply_variant(variant, setattr_fn=monkeypatch.setattr)
    cfg = _cfg(tmp_path / "variant", variant=variant.name, fork_at="pm", baseline_reader=_reader(base_dir), max_attempts=1)
    out = await runner.RunSession(cfg, {"run_id": "r", "base_key": "K", "arm": "variant", "variant": variant.name}).run(cases)
    assert (out.done, out.errors) == (1, 1)  # only the tampered case fails
    err = json.loads(next((cfg.run_dir / "errors").glob("*.json")).read_text(encoding="utf-8"))
    assert err["error"]["type"] == "ForkError" and "input_hash" in err["error"]["message"]
    assert err["case_id"] == cases[0].case_id


async def test_missing_or_failed_baseline_case_is_a_case_error(fake, monkeypatch, tmp_path):
    cases = _cases(1)
    variant = variants.get_variant("_test_pm_marker", allow_test_variant=True)
    variants.apply_variant(variant, setattr_fn=monkeypatch.setattr)
    cfg = _cfg(tmp_path / "variant", variant=variant.name, fork_at="pm", baseline_reader=lambda cid: None, max_attempts=1)
    out = await runner.RunSession(cfg, {"run_id": "r", "base_key": "K", "arm": "variant", "variant": variant.name}).run(cases)
    assert out.errors == 1 and out.done == 0
    assert fake.calls == []  # nothing was sent to the LLM


async def test_neutralization_identical_across_arms(fake, monkeypatch, tmp_path):
    """Same stored row -> same inputs in the baseline and the variant arm."""
    row = make_row(transcript=make_transcript(kr_risk_signals="K"))
    del row["full_transcript"]["kr_financials_recency"]
    a, b = make_inputs(row), make_inputs(row)
    assert a.input_hash() == b.input_hash()
    assert a.period_note == b.period_note == "" and a.kr_gt == b.kr_gt == "" and a.kr_risk_extra == b.kr_risk_extra == "K"

    base_dir, _ = await _baseline(tmp_path, [a])
    variant = variants.get_variant("_test_pm_marker", allow_test_variant=True)
    variants.apply_variant(variant, setattr_fn=monkeypatch.setattr)
    cfg = _cfg(tmp_path / "variant", variant=variant.name, fork_at="pm", baseline_reader=_reader(base_dir))
    await runner.RunSession(cfg, {"run_id": "r", "base_key": "K", "arm": "variant", "variant": variant.name}).run([b])
    base_case = runner.read_ok_case(base_dir, a.case_id)
    var_case = runner.read_ok_case(cfg.run_dir, b.case_id)
    assert base_case["input_hash"] == var_case["input_hash"]
    assert base_case["covariates"] == var_case["covariates"]
    assert base_case["covariates"]["period_note_neutralized"] and base_case["covariates"]["kr_gt_neutralized"]


def test_patch_context_records_before_and_after(monkeypatch):
    ctx = variants.VariantContext(setattr_fn=monkeypatch.setattr)
    original = settings.hysteresis_min_confidence_to_flip
    ctx.patch_setting("hysteresis_min_confidence_to_flip", 0.65)
    assert settings.hysteresis_min_confidence_to_flip == 0.65
    assert ctx.patches[0]["target"].endswith("hysteresis_min_confidence_to_flip")
    assert ctx.patches[0]["before"] != ctx.patches[0]["after"]
    monkeypatch.undo()
    assert settings.hysteresis_min_confidence_to_flip == original
