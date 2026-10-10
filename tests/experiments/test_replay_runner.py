"""app.experiments.replay.runner — node composition, guards, gates, per-case call capture,
checkpoint/resume, breakers (spec A.6/A.7/D, G). The agent code runs for real; only the
LLM (and the DB, which is blocked) is fake. Deterministic: no randomness, no network."""

from __future__ import annotations

import asyncio
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from app.agents import portfolio_manager, research_manager, risk_debators, trader
from app.experiments.replay import runner, safety
from app.llm import llm_client
from app.llm import metrics as llm_metrics
from app.orchestrator import pipeline
from app.state import render_research_plan
from replay_fixtures import FakeLLM, install_db_block, make_inputs, make_row, make_transcript

KST = ZoneInfo("Asia/Seoul")


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


def _inputs(ticker="005930.KS", id=1, **transcript):
    row = make_row(id=id, ticker=ticker, transcript=make_transcript(**transcript))
    return make_inputs(row)


async def _run(inp, **kw):
    return await runner.run_case(inp, run_id="r", **kw)


async def test_node_composition_kwargs(fake, monkeypatch):
    fake.rm, fake.trader_action = "Buy", "Buy"  # differs from the stored Hold/Hold
    spies: dict[str, dict] = {}

    def spy(module, name, key):
        orig = getattr(module, name)

        async def wrapper(*args, **kwargs):
            spies[key] = {"args": args, "kwargs": kwargs}
            return await orig(*args, **kwargs)

        monkeypatch.setattr(module, name, wrapper)

    def spy_sync(module, name, key):
        orig = getattr(module, name)

        def wrapper(*args, **kwargs):
            spies[key] = {"args": args, "kwargs": kwargs}
            return orig(*args, **kwargs)

        monkeypatch.setattr(module, name, wrapper)

    spy(research_manager, "synthesize", "rm")
    spy(trader, "propose", "trader")
    spy_sync(risk_debators, "build_all", "build_all")
    spy(portfolio_manager, "decide", "decide")

    inp = _inputs(position_context="POS", volatility_context="V", risk_cap_context="C", kr_risk_signals="K")
    inp = inp.with_session("SESSION", {"source": "stored", "agree": True})
    result = await _run(inp)

    rm = spies["rm"]["kwargs"]
    assert rm["held"] is False and rm["period_note"] == "" and rm["position_context"] == "POS"
    assert rm["identity"] == inp.identity_note and rm["session_note"] == "SESSION" and rm["language"] == "en"
    assert list(spies["rm"]["args"][0]) == list(inp.reports_dict())  # ordered reports

    from app.config import AIProfile

    tr = spies["trader"]["kwargs"]
    assert tr["profile"] == AIProfile.PRODUCTION and tr["held"] is False and tr["period_note"] == ""

    from app.state import ResearchPlan, TraderProposal

    new_plan = ResearchPlan.model_validate(result["nodes"]["rm"]["output"])
    new_proposal = TraderProposal.model_validate(result["nodes"]["trader"]["output"])
    assert new_plan.recommendation.value == "Buy"  # NOT the stored Hold
    ba = spies["build_all"]
    assert ba["args"][2] == inp.risk_extra == "POS\n\nK"
    assert ba["kwargs"]["sizing_context"] == "V\n\nC" == inp.debate_sizing_context
    assert ba["kwargs"]["research_plan_note"] == render_research_plan(new_plan)
    assert ba["kwargs"]["held"] is False

    decide = spies["decide"]["kwargs"]
    expected_note = pipeline._desk_consensus_note(new_plan, new_proposal, "K", "V\n\nC")
    assert decide["desk_consensus_note"] == expected_note and "Buy" in expected_note
    assert result["nodes"]["pm"]["desk_consensus_note"] == expected_note
    assert decide["degradation_note"] == "" and decide["held"] is False
    assert decide["period_note"] == "" and decide["sibling_context"] == inp.sibling_context
    assert spies["decide"]["args"][2:8] == (
        inp.past_context, "POS", "V", "C", inp.portfolio_risk_context, inp.execution_status_context,
    )
    assert result["status"] == "ok" and result["sweeps"] == "not_run"
    assert result["stored"]["rm"] == "Hold" and result["summary"]["rm"] == "Buy"
    assert [c["label"] for c in fake.calls][:2] == ["research_manager", "trader"]


async def test_turn_guard_kr_block(fake, monkeypatch):
    seen: list[dict] = []
    orig = runner.enforce_numeric_guards

    def spy(text, **kwargs):
        seen.append(kwargs)
        return orig(text, **kwargs)

    monkeypatch.setattr(runner, "enforce_numeric_guards", spy)
    await _run(_inputs(kr_risk_signals="K"))
    turn_calls = [k for k in seen if k["node"].startswith("risk_debate[")]
    assert len(turn_calls) == 6
    assert turn_calls[0]["node"] == "risk_debate[r1 Aggressive]"
    assert all(k["kr_block"] == "\n\nK" and k["soft_eok_block"] == "" and k["usd_check"] for k in turn_calls)
    assert all(k["language"] == "en" for k in turn_calls)
    nodes = {k["node"] for k in seen}
    assert {"research_plan.rationale", "research_plan.strategic_actions", "trader_proposal.reasoning",
            "portfolio_manager.executive_summary", "portfolio_manager.investment_thesis"} <= nodes
    card = [k for k in seen if k["node"].startswith("portfolio_manager.card_reason")]
    assert len(card) == 2 and all(k["inline_note"] is False and k["banner"] is False for k in card)
    field_calls = [k for k in seen if "." in k["node"]]
    assert all(k["kr_block"] == "" and k["banner"] is False for k in field_calls)

    seen.clear()
    row = make_row(transcript=make_transcript())
    del row["full_transcript"]["kr_risk_signals"]
    await _run(make_inputs(row))
    assert all(k["kr_block"] == "" for k in seen)


@pytest.mark.parametrize(
    "rm, tr, pm, conf, final, gated, hysteresis",
    [
        ("Hold", "Hold", "Buy", 0.5, "Hold", True, False),   # consensus override below the 0.75 bar
        ("Buy", "Buy", "Buy", 0.7, "Buy", None, False),       # no gate applies
        ("Buy", "Buy", "Buy", 0.55, "Hold", None, True),      # hysteresis: below 0.6 to flip
    ],
)
async def test_gates(fake, rm, tr, pm, conf, final, gated, hysteresis):
    fake.rm, fake.trader_action, fake.pm, fake.confidence = rm, tr, pm, conf
    result = await _run(_inputs())
    gates = result["nodes"]["gates"]
    assert gates["final_output"]["rating"] == final == result["summary"]["final"]
    assert result["nodes"]["pm"]["pre_gate_output"]["rating"] == pm == result["summary"]["pm_pre_gate"]
    if gated is None:
        assert gates["consensus_override"] == {} or gates["consensus_override"]["gated"] is False
    else:
        assert gates["consensus_override"]["gated"] is gated
        assert gates["final_output"]["executive_summary"].startswith("[Consensus override gate:")
    assert gates["hysteresis_applied"] is hysteresis
    assert gates["hysteresis_from"] == ("Buy" if hysteresis else None)
    assert gates["unjustified_exit_gate"] == {} and gates["degradation_applied"] is False
    assert result["summary"]["final_bullish"] is (final == "Buy")


async def test_tokens_and_calls_per_case(fake):
    fake.tokens = {"000001.KS": 10, "000002.KS": 1000}
    a = _inputs("000001.KS", id=1)
    b = _inputs("000002.KS", id=2)
    results = await asyncio.gather(_run(a), _run(b))  # two cases in flight at once
    calls_a, calls_b = results[0]["calls_summary"], results[1]["calls_summary"]
    assert calls_a["n"] == calls_b["n"] == 9  # RM + Trader + 6 risk turns + PM
    assert results[0]["tokens_total"] == 10 * calls_a["n"]
    assert results[1]["tokens_total"] == 1000 * calls_b["n"]
    assert sum(n["tokens"] for n in results[0]["nodes"].values()) == results[0]["tokens_total"]
    by_label = calls_a["prompt_sha256_by_label"]
    assert set(by_label) >= {"research_manager", "trader", "portfolio_manager", "risk_aggressive", "risk_neutral"}
    assert len(by_label["risk_aggressive"]) == 2 and all(len(s) == 64 for s in by_label["research_manager"])
    assert calls_a["n_structured"] == 3 and calls_a["n_fallback"] == 0 and calls_a["n_length"] == 0
    # identical inputs -> identical RM prompt bytes across runs (AA-1f premise)
    again = await _run(a)
    assert again["calls_summary"]["prompt_sha256_by_label"]["research_manager"] == by_label["research_manager"]
    # calls made outside a case are not captured anywhere
    assert runner._CASE_CALLS.get() is None


async def test_instrumentation_uninstall_restores(monkeypatch):
    fake = FakeLLM()
    monkeypatch.setattr(llm_client, "structured_complete", fake.structured_complete)
    original_record = llm_metrics.record
    instrumentation = runner.Instrumentation()
    instrumentation.install()
    assert llm_metrics.record is not original_record
    instrumentation.uninstall()
    assert llm_metrics.record is original_record
    assert llm_client.structured_complete == fake.structured_complete


def _cases(n, start=1):
    return [_inputs(f"00000{i}.KS", id=i) for i in range(start, start + n)]


def _session(tmp_path, **cfg_kw):
    cfg = runner.RunConfig(
        run_dir=tmp_path / "run", run_id="r", k=cfg_kw.pop("k", 1), echo=None,
        now_fn=cfg_kw.pop("now_fn", lambda: datetime(2026, 9, 26, 10, 0, tzinfo=KST)),  # Saturday
        heartbeat_s=3600, **cfg_kw,
    )
    return runner.RunSession(cfg, {"run_id": "r", "base_key": "k1", "arm": "baseline", "variant": "baseline"})


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


async def test_checkpoint_resume_idempotent(fake, tmp_path):
    cases = _cases(3)
    state = {"failed": False}

    def fail(label, ticker):
        if label == "research_manager" and ticker == "000002.KS" and not state["failed"]:
            state["failed"] = True
            return RuntimeError("boom")
        return None

    fake.tokens = {c.ticker: 10 for c in cases}
    fake.fail = fail
    session = _session(tmp_path, max_attempts=2)
    out = await session.run(cases)
    run_dir = tmp_path / "run"
    assert (out.done, out.errors, out.status, out.exit_code) == (2, 1, "done", runner.EXIT_INCOMPLETE)  # 76: one case left in error
    assert (out.remaining, out.retryable) == (1, 1)
    ok_files = sorted(p.name for p in (run_dir / "cases").glob("*.json"))
    assert len(ok_files) == 2 and (run_dir / "errors" / f"{cases[1].case_id}.a1.json").exists()
    err = json.loads((run_dir / "errors" / f"{cases[1].case_id}.a1.json").read_text(encoding="utf-8"))
    assert err["status"] == "error" and err["error"]["node"] == "rm" and err["error"]["type"] == "RuntimeError"
    first_sha = _sha(run_dir / "cases" / f"{cases[0].case_id}.json")
    calls_before = len(fake.calls)

    session2 = _session(tmp_path, max_attempts=2)
    out2 = await session2.run(cases)
    assert (out2.done, out2.skipped_ok, out2.errors) == (1, 2, 0)
    assert out2.exit_code == 0 and out2.remaining == 0  # 0 only when every case has an ok result
    only_second = {c["ticker"] for c in fake.calls[calls_before:]}
    assert only_second == {"000002.KS"}  # finished cases are not called again
    assert _sha(run_dir / "cases" / f"{cases[0].case_id}.json") == first_sha
    assert len(list((run_dir / "cases").glob("*.json"))) == 3

    calls_after = len(fake.calls)
    out3 = await _session(tmp_path, max_attempts=2).run(cases)
    assert out3.done == 0 and out3.skipped_ok == 3 and len(fake.calls) == calls_after

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "done" and manifest["resumed"] == 2 and manifest["base_key"] == "k1"
    log = (run_dir / "progress.log").read_text(encoding="utf-8").splitlines()
    assert sum(" ok " in ln for ln in log) == 3 and any("error attempt=1" in ln for ln in log)


async def test_max_attempts_gives_up(fake, tmp_path):
    cases = _cases(1)
    fake.tokens = {cases[0].ticker: 10}
    fake.fail = lambda label, ticker: RuntimeError("always")
    for _ in range(3):
        await _session(tmp_path, max_attempts=2).run(cases)
    errors = list((tmp_path / "run" / "errors").glob("*.json"))
    assert len(errors) == 2  # attempts 1 and 2, then it stops retrying
    out = await _session(tmp_path, max_attempts=2).run(cases)
    assert out.gave_up == 1 and out.done == 0


async def test_manifest_mismatch_refused(fake, tmp_path):
    await _session(tmp_path).run(_cases(1))
    cfg = runner.RunConfig(run_dir=tmp_path / "run", run_id="r", echo=None, now_fn=lambda: datetime(2026, 9, 26, 10, 0, tzinfo=KST))
    other = runner.RunSession(cfg, {"run_id": "r", "base_key": "DIFFERENT", "arm": "baseline", "variant": "baseline"})
    with pytest.raises(runner.ManifestMismatch):
        await other.run(_cases(1))


async def test_network_breaker_pauses(fake, tmp_path):
    import httpx
    import openai

    cases = _cases(6)
    fake.tokens = {c.ticker: 10 for c in cases}
    fake.fail = lambda label, ticker: openai.APIConnectionError(request=httpx.Request("POST", "http://llm/v1"))
    out = await _session(tmp_path, max_consecutive_failures=3).run(cases)
    assert out.status == "paused_network" and out.exit_code == runner.EXIT_PAUSED
    assert out.errors == 3 and out.done == 0
    assert len(list((tmp_path / "run" / "errors").glob("*.json"))) == 3  # the other 3 were never assigned
    manifest = json.loads((tmp_path / "run" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "paused_network"
    assert runner.is_network_error(OSError("x")) and runner.is_network_error(asyncio.TimeoutError())
    assert not runner.is_network_error(ValueError("x"))
    assert out.gave_up == 0


async def test_window_closes_midrun(fake, tmp_path):
    cases = _cases(3)
    fake.tokens = {c.ticker: 10 for c in cases}
    times = iter([
        datetime(2026, 9, 22, 15, 41, tzinfo=KST),  # Tuesday after the window: allowed
        datetime(2026, 9, 23, 8, 30, tzinfo=KST),   # next morning window opens
    ])
    out = await _session(tmp_path, now_fn=lambda: next(times)).run(cases)
    assert out.status == "paused_window" and out.exit_code == runner.EXIT_PAUSED and out.done == 1
    assert len(list((tmp_path / "run" / "cases").glob("*.json"))) == 1  # in-flight case finished, nothing new assigned
    # resume once the window is closed again
    out2 = await _session(tmp_path).run(cases)
    assert out2.status == "done" and out2.done == 2 and out2.skipped_ok == 1


async def test_allow_market_hours_override(fake, tmp_path):
    cases = _cases(2)
    fake.tokens = {c.ticker: 10 for c in cases}
    tuesday_10am = lambda: datetime(2026, 9, 22, 10, 0, tzinfo=KST)  # noqa: E731
    blocked = await _session(tmp_path, now_fn=tuesday_10am).run(cases)
    assert blocked.status == "paused_window" and blocked.done == 0
    allowed = await _session(tmp_path, now_fn=tuesday_10am, allow_market_hours=True).run(cases)
    assert allowed.status == "done" and allowed.done == 2


async def test_safety_violation_aborts_run(tmp_path):
    cases = _cases(3)

    async def bad_case(inp, **kwargs):
        raise safety.ReplaySafetyViolation("blocked: store_pending")

    cfg = runner.RunConfig(
        run_dir=tmp_path / "run", run_id="r", echo=None, k=2, heartbeat_s=3600, case_fn=bad_case,
        now_fn=lambda: datetime(2026, 9, 26, 10, 0, tzinfo=KST),
    )
    out = await runner.RunSession(cfg, {"run_id": "r", "base_key": "k", "arm": "baseline", "variant": "baseline"}).run(cases)
    assert out.status == "aborted_safety" and out.exit_code == runner.EXIT_SAFETY
    assert out.done == 0


async def test_per_case_timeout_recorded_as_error(tmp_path):
    cases = _cases(1)

    async def slow_case(inp, **kwargs):
        await asyncio.sleep(5)

    cfg = runner.RunConfig(
        run_dir=tmp_path / "run", run_id="r", echo=None, case_timeout=0.05, heartbeat_s=3600, case_fn=slow_case,
        now_fn=lambda: datetime(2026, 9, 26, 10, 0, tzinfo=KST),
    )
    out = await runner.RunSession(cfg, {"run_id": "r", "base_key": "k", "arm": "baseline", "variant": "baseline"}).run(cases)
    assert out.errors == 1
    err = json.loads(next((tmp_path / "run" / "errors").glob("*.json")).read_text(encoding="utf-8"))
    assert err["error"]["type"] in {"TimeoutError"}


async def test_case_json_shape_and_roundtrip_of_node_outputs(fake):
    result = await _run(_inputs(kr_risk_signals="K"))
    assert result["schema"] == "replay-case-v1" and result["case_id"].startswith("005930.KS_")
    assert set(result["nodes"]) == set(runner.NODES)
    for name in runner.NODES:
        assert result["nodes"][name]["source"] == "llm"
    assert set(result["summary"]) == {
        "rm", "trader", "neutral_final", "pm_pre_gate", "final", "confidence",
        "rm_bullish", "final_bullish", "neutral_bullish",
    }
    assert result["summary"]["neutral_final"] == "Hold" and result["summary"]["neutral_bullish"] is False
    assert result["covariates"]["interim"] is True and result["covariates"]["kr_risk_neutralized"] is False
    json.dumps(result)  # fully serialisable
    # every node output re-validates into the production models (fork reuse depends on it)
    from app.state import PortfolioDecision, ResearchPlan, TraderProposal

    ResearchPlan.model_validate(result["nodes"]["rm"]["output"])
    TraderProposal.model_validate(result["nodes"]["trader"]["output"])
    pm_again = PortfolioDecision.model_validate(result["nodes"]["pm"]["pre_gate_output"])
    assert runner.dump_model(pm_again) == result["nodes"]["pm"]["pre_gate_output"]
    assert result["nodes"]["risk_debate"]["stances"] == {"r1": "Hold", "final": "Hold"}
    assert len(result["nodes"]["risk_debate"]["turns"]) == 6


async def test_clock_untouched_by_case_run(fake):
    from app.clock import LiveClock, get_clock

    before = get_clock()
    await _run(_inputs())
    assert get_clock() is before and isinstance(before, LiveClock)


async def test_case_runs_with_all_safety_blocks_installed(fake, monkeypatch):
    """The downstream path must never touch a blocked function: run a full case with every
    block installed — any trip would raise ReplaySafetyViolation (a BaseException)."""
    safety.reset_trips()
    report = safety.install_blocks(setattr_fn=monkeypatch.setattr)
    assert report.missing == [] and report.installed
    result = await _run(_inputs(kr_risk_signals="K"))
    assert result["status"] == "ok" and safety.trips() == []


async def test_instrumentation_with_the_real_llm_client(monkeypatch):
    """Real LLMClient.complete/structured_complete with only the SDK call stubbed: the
    per-case capture sees label, tokens, fallback flag, finish reason and the prompt hash."""
    from types import SimpleNamespace

    from app.state import FlatTraderProposal

    async def fake_create(**kwargs):
        content = (
            json.dumps({"action": "Hold", "reasoning": "r"})
            if "response_format" in kwargs else "risk turn text"
        )
        message = SimpleNamespace(content=content, tool_calls=None, reasoning_content=None)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message, finish_reason="stop")],
            usage=SimpleNamespace(prompt_tokens=7, completion_tokens=3, total_tokens=10),
        )

    monkeypatch.setattr(llm_client._client.chat.completions, "create", fake_create)
    instrumentation = runner.Instrumentation()
    instrumentation.install()
    try:
        calls = runner.CaseCalls()
        token = runner._CASE_CALLS.set(calls)
        try:
            messages = [{"role": "user", "content": "hello"}]
            await llm_client.complete(messages, label="risk_neutral")
            proposal = await llm_client.structured_complete(messages, FlatTraderProposal, label="trader")
        finally:
            runner._CASE_CALLS.reset(token)
    finally:
        instrumentation.uninstall()
    assert proposal.action.value == "Hold"
    assert [r["label"] for r in calls.records] == ["risk_neutral", "trader"]
    assert [r["structured"] for r in calls.records] == [False, True]
    assert all(r["total_tokens"] == 10 and r["finish_reason"] == "stop" and r["fallback_used"] is False for r in calls.records)
    assert {r["prompt_sha256"] for r in calls.records} == {runner.prompt_sha256(messages)}
    assert [o["label"] for o in calls.outer] == ["risk_neutral", "trader"]
    summary = runner._calls_summary(calls)
    assert summary["n"] == 2 and summary["n_structured"] == 1 and summary["n_fallback"] == 0


def _chained_connection_error(label="research_manager", ticker=""):
    """What `LLMClient.structured_complete` raises for a dead vLLM: ValueError from APIConnectionError."""
    import httpx
    import openai

    inner = openai.APIConnectionError(request=httpx.Request("POST", "http://llm/v1"))
    outer = ValueError("structured_complete: both native response_format and the fallback failed")
    outer.__cause__ = inner
    return outer


def test_is_network_error_walks_exception_chains():
    import httpx
    import openai

    assert runner.is_network_error(_chained_connection_error()) is True
    assert runner.is_network_error(ValueError("x")) is False
    # via __context__ (implicit chaining) and through CaseNodeError
    try:
        try:
            raise openai.APITimeoutError(request=httpx.Request("POST", "http://llm/v1"))
        except Exception:
            raise ValueError("wrapped")
    except ValueError as exc:
        assert runner.is_network_error(exc) is True
        assert runner.is_network_error(runner.CaseNodeError("rm", exc)) is True
    assert runner.is_network_error(runner.CaseNodeError("rm", _chained_connection_error())) is True
    assert runner.is_network_error(runner.CaseNodeError("rm", ValueError("plain"))) is False
    # a cyclic chain terminates
    a, b = ValueError("a"), ValueError("b")
    a.__cause__, b.__cause__ = b, a
    assert runner.is_network_error(a) is False
    b.__cause__ = OSError("net")
    assert runner.is_network_error(a) is True


async def test_dead_vllm_pauses_the_run_and_does_not_burn_attempts(fake, tmp_path):
    """Review M1: every case fails at RM with the chained ValueError. The breaker must trip,
    the run must pause (75, not 'done'), and no case may use up an attempt or be given up on."""
    cases = _cases(6)
    fake.tokens = {c.ticker: 10 for c in cases}
    fake.fail = _chained_connection_error
    out = await _session(tmp_path, max_consecutive_failures=3, max_attempts=2).run(cases)
    assert out.status == "paused_network" and out.exit_code == runner.EXIT_PAUSED
    assert out.errors == 3 and out.done == 0 and out.gave_up == 0
    errors = list((tmp_path / "run" / "errors").glob("*.json"))
    assert len(errors) == 3 and all(".net" in p.name and ".a" not in p.name.split(".net")[0][-3:] for p in errors)
    assert json.loads(errors[0].read_text(encoding="utf-8"))["network"] is True
    # resuming keeps attempt number 1 for those cases (nothing was consumed), even after many pauses
    for _ in range(3):
        again = await _session(tmp_path, max_consecutive_failures=3, max_attempts=2).run(cases)
        assert again.status == "paused_network" and again.gave_up == 0
    fake.fail = None
    done = await _session(tmp_path, max_attempts=2).run(cases)
    assert done.status == "done" and done.done == 6 and done.gave_up == 0 and done.exit_code == 0
    ok = json.loads(next((tmp_path / "run" / "cases").glob("*.json")).read_text(encoding="utf-8"))
    assert ok["attempt"] == 1  # never advanced by connectivity failures


async def test_disk_error_is_not_a_network_error(fake, tmp_path, monkeypatch):
    cases = _cases(1)
    fake.tokens = {cases[0].ticker: 10}
    real_write = runner.write_json

    def flaky(path, obj):
        if "cases" in Path(path).parts:
            raise PermissionError("locked")
        return real_write(path, obj)

    monkeypatch.setattr(runner, "write_json", flaky)
    out = await _session(tmp_path, max_attempts=1).run(cases)
    assert out.errors == 1 and out.status == "done"  # not paused_network
    err = json.loads(next((tmp_path / "run" / "errors").glob("*.json")).read_text(encoding="utf-8"))
    assert err["error"]["type"] == "PersistError" and err["network"] is False


async def test_heartbeat_write_failure_does_not_kill_the_run(fake, tmp_path, monkeypatch):
    cases = _cases(2)
    fake.tokens = {c.ticker: 10 for c in cases}
    real_write = runner.write_json
    state = {"fail": True}

    def flaky(path, obj):
        if Path(path).name == "manifest.json" and state["fail"] and obj.get("heartbeat", {}).get("done", 0) == 1:
            raise PermissionError("locked by AV")
        return real_write(path, obj)

    monkeypatch.setattr(runner, "write_json", flaky)
    logs = []
    session = _session(tmp_path)
    session.cfg.echo = logs.append
    out = await session.run(cases)
    assert out.status == "done" and out.done == 2
    assert any("manifest write failed" in ln for ln in logs)


async def test_manifest_windows_are_per_process(fake, tmp_path):
    cases = _cases(2)
    fake.tokens = {c.ticker: 10 for c in cases}
    await _session(tmp_path).run(cases[:1])
    await _session(tmp_path).run(cases)
    manifest = json.loads((tmp_path / "run" / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["windows"]) == 2  # one window per harness process, the pause between is not covered
    assert all(w["start"] <= w["end"] for w in manifest["windows"])
    assert manifest["windows"][0]["end"] <= manifest["windows"][1]["start"]


async def test_swallowed_block_trip_aborts_the_run(tmp_path):
    """A trip that production code swallowed (no exception reaches the harness) still aborts."""
    cases = _cases(2)

    async def sneaky_case(inp, **kwargs):
        safety._record_trip("app.memory.store.store_pending")  # what a stub does before raising
        return {"summary": {"rm": "Hold", "final": "Hold"}, "wall_s": 0, "tokens_total": 0, "status": "ok"}

    safety.reset_trips()
    try:
        cfg = runner.RunConfig(
            run_dir=tmp_path / "run", run_id="r", echo=None, heartbeat_s=3600, case_fn=sneaky_case,
            now_fn=lambda: datetime(2026, 9, 26, 10, 0, tzinfo=KST),
        )
        out = await runner.RunSession(cfg, {"run_id": "r", "base_key": "k", "arm": "baseline", "variant": "baseline"}).run(cases)
        assert out.status == "aborted_safety" and out.exit_code == runner.EXIT_SAFETY and out.done == 0
        assert not list((tmp_path / "run" / "cases").glob("*.json"))  # the suspicious result was not persisted
    finally:
        safety.reset_trips()
