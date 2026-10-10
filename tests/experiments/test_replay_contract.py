"""app.experiments.replay.contract — stored row -> ReplayInputs (spec A.3-A.5, G).
Synthetic rows only: no DB, no LLM, no network."""

from __future__ import annotations

import dataclasses
import inspect
import json
from datetime import date, datetime

import pytest

from app.experiments.replay import contract
from replay_fixtures import (
    UTC, install_db_block, make_inputs, make_row, make_transcript, research_turns, risk_turns,
)


@pytest.fixture(autouse=True)
def _block_db(monkeypatch):
    install_db_block(monkeypatch)


def test_report_order_restored():
    inp = make_inputs()
    assert tuple(name for name, _ in inp.reports) == contract.ANALYST_ORDER
    assert list(inp.reports_dict()) == list(contract.ANALYST_ORDER)

    missing = make_transcript()
    del missing["reports"]["Sentiment Analyst"]
    with pytest.raises(contract.ContractError, match="report keys"):
        make_inputs(make_row(transcript=missing))
    extra = make_transcript()
    extra["reports"]["Extra Analyst"] = "x"
    with pytest.raises(contract.ContractError, match="report keys"):
        make_inputs(make_row(transcript=extra))


def test_analyst_order_matches_pipeline_literal():
    from app.orchestrator import pipeline

    source = inspect.getsource(pipeline._run_pipeline_impl)
    assert contract.ANALYST_ORDER_SOURCE_LITERAL in source
    assert list(contract.ANALYST_ORDER) == [
        "Market Analyst", "Fundamentals Analyst", "News + Macro Analyst", "Sentiment Analyst",
    ]


def test_identity_note():
    note = contract.identity_note_text("005930.KS", "Samsung Electronics")
    assert note == (
        "The company under analysis is Samsung Electronics (005930.KS). This "
        "ticker-to-name mapping is verified against the exchange's own registry and "
        "overrides any other name. If a report below names a different company, that "
        "name is wrong — the underlying data is still for this ticker. Do not dismiss "
        "or discount a report's content because of a name mismatch, and do not repeat "
        "the wrong name."
    )
    assert contract.identity_note_text("005930.KS", "").startswith("The company under analysis is 005930.KS. This ")
    from app.orchestrator import pipeline

    source = inspect.getsource(pipeline._run_pipeline_impl)
    for chunk in contract.IDENTITY_NOTE_CHUNKS:
        assert chunk in source, f"identity note chunk drifted from pipeline: {chunk!r}"


def test_session_prepass():
    from app.clock import LiveClock, get_clock

    def case(created_at, stored, cid):
        row = make_row(
            id=cid, created_at=created_at,
            transcript=make_transcript(**({} if stored is None else {"session_in_progress": stored})),
        )
        if stored is None:
            del row["full_transcript"]["session_in_progress"]
        return make_inputs(row)

    tue_1000 = datetime(2026, 9, 22, 1, 0, tzinfo=UTC)  # 10:00 KST, Tuesday, session open
    cases = [
        case(tue_1000, True, 1),                                     # 1 stored True, agrees
        case(tue_1000, False, 2),                                    # 2 stored False wins over FakeClock
        case(datetime(2026, 9, 2, 1, 0, tzinfo=UTC), None, 3),       # 3 no stored value -> FakeClock
        case(datetime(2026, 9, 22, 6, 35, tzinfo=UTC), True, 4),     # 4 stored True, 15:35 KST is closed
    ]
    original = get_clock()
    out = contract.session_prepass(cases)
    assert get_clock() is original and isinstance(get_clock(), LiveClock)  # 5
    assert out[0].session_note and out[0].session_meta == {"source": "stored", "agree": True}
    assert out[1].session_note == "" and out[1].session_meta == {"source": "stored", "agree": False}
    assert out[2].session_note and out[2].session_meta == {"source": "fakeclock", "agree": None}
    assert out[3].session_note and out[3].session_meta == {"source": "stored", "agree": False}

    naive = dataclasses.replace(cases[0], created_at=datetime(2026, 9, 22, 10, 0))
    with pytest.raises(ValueError):  # 6
        contract.session_prepass([naive])
    assert get_clock() is original


def test_kr_risk_and_risk_extra():
    inp = make_inputs(make_row(transcript=make_transcript(kr_risk_signals="K", position_context="POS")))
    assert inp.kr_risk_extra == "K" and inp.risk_extra == "POS\n\nK" and inp.kr_risk_neutralized is False
    row = make_row(transcript=make_transcript(position_context="POS"))
    del row["full_transcript"]["kr_risk_signals"]
    inp = make_inputs(row)
    assert inp.kr_risk_extra == "" and inp.kr_risk_neutralized is True and inp.risk_extra == "POS"
    # an explicitly stored empty string is a stored value, not neutralised
    inp = make_inputs(make_row(transcript=make_transcript(kr_risk_signals="", position_context="POS")))
    assert inp.kr_risk_neutralized is False and inp.risk_extra == "POS"


def test_debate_sizing_context():
    f = contract.derive_debate_sizing_context
    assert f("V", "C") == "V\n\nC"
    assert f("", "C") == "C"
    assert f("V", None) == "V"
    assert f("", "") == ""
    inp = make_inputs(make_row(transcript=make_transcript(volatility_context="V", risk_cap_context="C")))
    assert inp.debate_sizing_context == "V\n\nC"


def test_neutralized_fields():
    from app.orchestrator import pipeline

    inp = make_inputs()
    assert inp.period_note == inp.kr_gt == inp.kr_news_gt == ""
    assert pipeline._degradation_note(list(inp.degraded_analysts)) == ""
    assert inp.held is False and inp.language == "en"
    cov = inp.covariates()
    assert cov["period_note_neutralized"] and cov["kr_gt_neutralized"]


def test_debate_state_rebuild():
    inp = make_inputs()
    state = inp.research_debate()
    assert state.parties == ["Bull", "Bear"] and state.rounds_completed == 2 and len(state.turns) == 4
    assert state.transcript().startswith("[Round 1 — Bull]\nbull one")
    assert "[Round 2 — Bear]\nbear two" in state.transcript()
    state.turns.append(state.turns[0])  # a node may mutate its copy ...
    assert len(inp.research_debate().turns) == 4  # ... but every call gets a fresh instance

    bad = make_transcript(research_debate=research_turns()[:3])
    with pytest.raises(contract.ContractError, match="research_debate"):
        make_inputs(make_row(transcript=bad))
    bad = make_transcript(risk_debate=risk_turns()[:5])
    with pytest.raises(contract.ContractError, match="risk_debate"):
        make_inputs(make_row(transcript=bad))


def test_stored_parsing():
    assert contract.parse_stored_rm("**Recommendation**: Buy\n\n**Rationale**: x") == "Buy"
    assert contract.parse_stored_rm("no header") is None
    assert contract.parse_stored_trader("**Action**: Hold\n\n**Reasoning**: x") == "Hold"
    assert contract.parse_pre_gate("Hold", {"gated": True, "pm_rating": "Buy"}) == "Buy"
    assert contract.parse_pre_gate("Hold", {"gated": False, "pm_rating": "Buy"}) == "Hold"
    assert contract.parse_pre_gate("Hold", {}, "x [Hysteresis gate: downgraded from Buy to Hold — c]") == "Buy"
    assert contract.parse_pre_gate("Hold", None, "plain") == "Hold"

    row = make_row(
        rating="Hold",
        transcript=make_transcript(
            research_plan="**Recommendation**: Buy\n\n**Rationale**: r",
            trader_proposal="**Action**: Buy\n\n**Reasoning**: t",
            risk_debate=risk_turns("Buy"),
            investment_thesis="a\n\n[Hysteresis gate: downgraded from Buy to Hold — confidence (55%) ...",
        ),
        confidence=0.55,
    )
    stored = make_inputs(row).stored
    assert (stored.rm, stored.trader, stored.neutral_final, stored.pm_pre_gate, stored.final) == (
        "Buy", "Buy", "Buy", "Buy", "Hold",
    )
    assert stored.confidence == 0.55


def test_input_hash():
    base = make_inputs()
    reordered_dict = make_transcript()
    reordered_dict["reports"] = dict(reversed(list(reordered_dict["reports"].items())))
    assert make_inputs(make_row(transcript=reordered_dict)).input_hash() == base.input_hash()

    changed = make_transcript(past_context="PAST!")
    assert make_inputs(make_row(transcript=changed)).input_hash() != base.input_hash()

    other_stored = make_row(rating="Buy", confidence=0.9)
    other_stored["full_transcript"]["research_plan"] = "**Recommendation**: Buy\n\n**Rationale**: r"
    assert make_inputs(other_stored).input_hash() == base.input_hash()  # stored results are not inputs

    swapped = dataclasses.replace(base, reports=tuple(reversed(base.reports)))
    assert swapped.input_hash() != base.input_hash()  # report order is part of the input
    assert dataclasses.replace(base, session_note="x").input_hash() != base.input_hash()


def test_extract_filters():
    ok = make_row(id=10, ticker="000660.KS")
    held = make_row(id=11, ticker="000661.KS", held_at_decision=True)
    held_null = make_row(id=12, ticker="000662.KS", held_at_decision=None)
    af1 = make_row(id=13, ticker="000663.KS", analysts_failed=1)
    bad_reports = make_transcript()
    del bad_reports["reports"]["Market Analyst"]
    broken = make_row(id=14, ticker="000664.KS", transcript=bad_reports)
    dup_old = make_row(id=20, ticker="000670.KS", created_at=datetime(2026, 9, 22, 1, 0, tzinfo=UTC))
    dup_new = make_row(id=21, ticker="000670.KS", created_at=datetime(2026, 9, 22, 5, 0, tzinfo=UTC))
    # dedup happens BEFORE the af0 filter: the newer duplicate has af1, so the ticker drops out entirely
    dup2_old = make_row(id=30, ticker="000680.KS", created_at=datetime(2026, 9, 22, 1, 0, tzinfo=UTC))
    dup2_new = make_row(
        id=31, ticker="000680.KS", created_at=datetime(2026, 9, 22, 5, 0, tzinfo=UTC), analysts_failed=2
    )
    rows = [ok, held, held_null, af1, broken, dup_old, dup_new, dup2_old, dup2_new]
    sel = contract.select_cases(rows)
    assert {c["memory_id"] for c in sel.cases} == {10, 21}
    by_id = {e["memory_id"]: e["reason"] for e in sel.excluded}
    assert by_id[20] == "dedup_older_duplicate" and by_id[30] == "dedup_older_duplicate"
    assert by_id[31] == "analysts_failed_nonzero" and by_id[13] == "analysts_failed_nonzero"
    assert by_id[11] == "held_at_decision_true" and by_id[12] == "held_at_decision_null"
    assert by_id[14].startswith("contract: report keys")
    assert sel.counts["input_rows"] == 9 and sel.counts["after_dedup"] == 7
    assert sel.counts["after_af0"] == 5 and sel.counts["after_unheld"] == 3 and sel.counts["after_contract"] == 2
    assert sel.counts["excluded_by_reason"]["dedup_older_duplicate"] == 2

    only_dev = contract.select_cases(rows, max_trade_date=date(2026, 9, 16))
    assert only_dev.cases == []


def test_era_and_split():
    assert contract.era_of(datetime(2026, 9, 17, 3, 10, 52, tzinfo=UTC), date(2026, 9, 17)) == "freeze"
    assert contract.era_of(datetime(2026, 9, 17, 3, 10, 51, tzinfo=UTC), date(2026, 9, 17)) == "mixed_0917"
    assert contract.era_of(datetime(2026, 9, 10, 3, 0, tzinfo=UTC), date(2026, 9, 10)) == "pre_freeze"
    assert contract.split_of(date(2026, 9, 16)) == "dev" and contract.split_of(date(2026, 9, 17)) == "holdout"
    rec = contract.record_from_row(make_row(id=7, ticker="005930.KS", trade_date=date(2026, 9, 22)))
    assert rec["case_id"] == "005930.KS_20260922_m7" and ":" not in rec["case_id"]
    fr = contract.select_cases([make_row(id=1)], era="freeze")
    assert len(fr.cases) == 1 and contract.select_cases([make_row(id=1)], era="pre_freeze").cases == []


def test_desk_note_stored_match():
    from app.orchestrator import pipeline
    from app.state import PortfolioRating, ResearchPlan, TraderAction, TraderProposal

    plan = ResearchPlan(recommendation=PortfolioRating.HOLD, rationale="x", strategic_actions="x")
    proposal = TraderProposal(action=TraderAction.HOLD, reasoning="x")
    note = pipeline._desk_consensus_note(plan, proposal, "K", "VOL\n\nCAP")
    freeze_row = make_row(
        created_at=datetime(2026, 9, 22, 1, 0, tzinfo=UTC),
        transcript=make_transcript(desk_consensus_note=note),
    )
    assert make_inputs(freeze_row).contract_checks["desk_note_stored_match"] is True
    bad = make_row(transcript=make_transcript(desk_consensus_note=note + " tampered"))
    assert make_inputs(bad).contract_checks["desk_note_stored_match"] is False
    # non-freeze rows are not comparable (the note changed over time)
    old = make_row(id=9, trade_date=date(2026, 9, 10), created_at=datetime(2026, 9, 10, 1, 0, tzinfo=UTC))
    assert make_inputs(old).contract_checks["desk_note_stored_match"] is None


def test_caseset_roundtrip_and_tamper(tmp_path):
    recs = [contract.record_from_row(make_row(id=i, ticker=f"00000{i}.KS")) for i in (1, 2)]
    jsonl, meta_path = contract.write_caseset(tmp_path, "cs", recs, {"sql": contract.EXTRACT_SQL})
    loaded, meta = contract.load_caseset(tmp_path, "cs")
    assert loaded == json.loads(json.dumps(recs)) and meta["n_cases"] == 2
    jsonl.write_text(jsonl.read_text(encoding="utf-8") + "\n{}\n", encoding="utf-8")
    with pytest.raises(contract.ContractError, match="sha256"):
        contract.load_caseset(tmp_path, "cs")
    assert meta_path.exists()


def test_atomic_write_retries_permission_error(tmp_path, monkeypatch):
    import os

    real_replace, calls = os.replace, {"n": 0}

    def flaky(src, dst):
        calls["n"] += 1
        if calls["n"] < 3:
            raise PermissionError("locked")
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", flaky)
    target = tmp_path / "a" / "b.json"
    contract.atomic_write_text(target, "hi", retries=5, delay=0)
    assert target.read_text(encoding="utf-8") == "hi" and calls["n"] == 3


def test_missing_context_key_is_contract_error():
    row = make_row()
    del row["full_transcript"]["past_context"]
    with pytest.raises(contract.ContractError, match="past_context"):
        make_inputs(row)
    row = make_row(held_at_decision=True)
    with pytest.raises(contract.ContractError, match="held_at_decision"):
        make_inputs(row)
