"""app.audit.rm_leak -- pure classifier tests plus DB-faked aggregation/CLI
parsing (same fake-pool convention as tests/audit/test_rating_shift.py)."""

from __future__ import annotations

import json
import sys
from datetime import date, datetime, timezone

import pytest

from app.audit import __main__ as audit_main
from app.audit import rm_leak
from app.audit.rm_leak import classify_research_plan as classify


def _plan(rationale: str, rec: str = "Hold") -> str:
    return f"**Recommendation**: {rec}\n\n**Rationale**: {rationale}"


# ---------------------------------------------------------------------------
# classifier
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("start", [
    "The user wants a synthesis of the debate", "The model should decide", "Synthesizing the arguments",
    "Okay, let's see", "Let me think", "I need to weigh", "Here's the plan", "<|channel|>analysis",
    "사용자가 요청한",
])
def test_v1_positive(start):
    assert "v1" in classify(_plan(start + " ..."))


@pytest.mark.parametrize("start", [
    "The analysis requires", "The request is to", "The provided debate", "Analyzes all four views",
    "Analyses the provided", "Evaluating the provided inputs", "Based on the four reports", "To synthesize this",
])
def test_v2_positive(start):
    kinds = classify(_plan(start))
    assert "v2" in kinds


@pytest.mark.parametrize("body", [
    "After weighing the debate, I will structure the answer",
    "The call is Hold. Thinking Process: first",
    "The response must integrate all views",
    "Fine so far <think>hmm</think>",
])
def test_broad_positive_anywhere(body):
    kinds = classify(_plan("After weighing the arguments. " + body))
    assert "broad" in kinds


@pytest.mark.parametrize("start", [
    "<i><b>INTERIM-WARNING INTEGRATION:</b></i> text", "<B>Bold</B>", "<p>para", "<DIV>x", "<br>x", "  <span>x",
])
def test_html_positive(start):
    assert "html" in classify(_plan(start))


@pytest.mark.parametrize("start", [
    "After weighing the rebuttals, Hold.", "After reviewing the debate, Hold.", "After evaluating the data, Hold.",
    "RECOMMENDATION: HOLD (stay out)", "**Recommendation: HOLD (Stay Out)** because", "The bear case is stronger.",
    "The debate centers on valuation.", "**Rating: Sell** given",
])
def test_normal_starts_negative(start):
    assert classify(_plan(start)) == ()


def test_only_rationale_start_matters_for_start_patterns():
    # 'The user' in the Recommendation header area (before the marker) must not fire v1.
    assert classify("**Recommendation**: Hold\n\nThe user note.\n\n**Rationale**: After weighing the case.") == ()
    # but a start pattern right after the marker does.
    assert classify(_plan("The user asks")) == ("v1",)


def test_no_rationale_marker_uses_whole_text():
    assert classify("Okay, here we go") == ("v1",)
    assert classify("After weighing the debate we Hold") == ()


def test_dict_plan():
    assert classify({"recommendation": "Hold", "rationale": "The task is to"}) == ("v2",)
    assert classify({"recommendation": "Hold", "rationale": "After weighing the case."}) == ()
    assert classify({"recommendation": "Hold"}) == ()
    assert classify({"rationale": "x", "extra": "Self-Correction applied"}) == ("broad",)


@pytest.mark.parametrize("plan", [None, "", "   ", 5, [], {}.get("x")])
def test_missing_or_unusable_plan_is_clean_and_absent(plan):
    assert classify(plan) == ()
    assert rm_leak.has_research_plan(plan) is False


def test_multiple_kinds_overlap_in_order():
    assert classify(_plan("The response must integrate all inputs")) == ("v1", "broad")


def test_is_holding():
    assert rm_leak.is_holding({"position_context": "Currently holding 3 shares."})
    assert not rm_leak.is_holding({"position_context": "Currently flat."})
    assert not rm_leak.is_holding({})


# ---------------------------------------------------------------------------
# aggregation (DB faked)
# ---------------------------------------------------------------------------

class FakeConn:
    def __init__(self, rows):
        self.rows = rows
        self.queries: list[tuple] = []

    async def fetch(self, query, *args):
        self.queries.append((query, args))
        return self.rows


class _CM:
    def __init__(self, conn):
        self.conn = conn

    async def __aenter__(self):
        return self.conn

    async def __aexit__(self, *exc):
        return False


class FakePool:
    def __init__(self, conn):
        self.conn = conn

    def acquire(self):
        return _CM(self.conn)


def _row(i, day, ts, plan, pos="Currently flat."):
    transcript = {"research_plan": plan, "position_context": pos}
    return {
        "id": i, "ticker": "005930.KS", "trade_date": day, "created_at": ts,
        "full_transcript": json.dumps(transcript) if i % 2 else transcript,
    }


D1, D2 = date(2026, 10, 1), date(2026, 10, 2)
T = lambda h: datetime(2026, 10, 2, h, 0, tzinfo=timezone.utc)  # noqa: E731


def _rows():
    return [
        _row(1, D1, T(1), _plan("The user asks"), pos="Currently holding 2 shares."),
        _row(2, D1, T(2), _plan("After weighing the case")),
        _row(3, D2, T(3), _plan("<i><b>X</b></i> y")),
        _row(4, D2, T(5), _plan("After reviewing the debate")),
        _row(5, D2, T(6), None),
    ]


def test_build_report_counts():
    rep = rm_leak.build_report(
        _rows(), since=D1, until=None, market="domestic", scope="live", account_mode=None,
        after_ts=T(4), show=1,
    )
    assert rep.total.n == 4 and rep.total.contaminated == 2
    assert rep.days[D1].contaminated == 1 and rep.days[D2].kinds["html"] == 1
    assert rep.after_total.n == 1 and rep.after_total.contaminated == 0
    assert rep.holding["holding"].contaminated == 1 and rep.holding["holding"].n == 1
    assert rep.holding["not_holding"].n == 3
    assert rep.n_no_plan == 1
    assert [s["id"] for s in rep.shown] == [1]
    assert len(rep.shown[0]["head"]) <= 80
    text = rm_leak.render_rm_leak(rep)
    assert "TOTAL" in text and "after-ts" in text and "id=1" in text and "id=3" not in text
    parsed = json.loads(rm_leak.report_to_json(rep))
    assert parsed["total"]["contaminated"] == 2 and len(parsed["contaminated"]) == 2


def test_show_zero_prints_no_rows():
    rep = rm_leak.build_report(
        _rows(), since=D1, until=None, market="domestic", scope="live", account_mode=None, after_ts=None,
    )
    assert rep.shown == [] and "Contaminated rows" not in rm_leak.render_rm_leak(rep)
    assert rep.after_total is None


@pytest.mark.asyncio
async def test_compute_query_filters(monkeypatch):
    conn = FakeConn(_rows())

    async def fake_pool():
        return FakePool(conn)

    monkeypatch.setattr(rm_leak, "get_pool", fake_pool)
    monkeypatch.setattr(rm_leak, "get_current_user_id", lambda: 7)
    rep = await rm_leak.compute_rm_leak(D1, until=D2, account_mode="real", after_ts=T(4))
    query, args = conn.queries[0]
    assert "is_backtest = false" in query and ".KQ" in query and "NOT (" not in query
    assert args == ("real", D1, D2, 7)
    assert rep.total.n == 4
    await rm_leak.compute_rm_leak(D1, market="overseas", scope="backtest")
    q2, a2 = conn.queries[1]
    assert "NOT (" in q2 and "is_backtest = true" in q2 and a2 == (D1, 7)


@pytest.mark.asyncio
async def test_compute_rejects_bad_args():
    with pytest.raises(ValueError):
        await rm_leak.compute_rm_leak(D1, market="x")
    with pytest.raises(ValueError):
        await rm_leak.compute_rm_leak(D1, after_ts=datetime(2026, 10, 2))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["audit", *argv])
    return audit_main._parse_args()


def test_cli_requires_since(monkeypatch):
    with pytest.raises(SystemExit):
        _parse(monkeypatch, "rm-leak")


def test_cli_parses(monkeypatch):
    a = _parse(monkeypatch, "rm-leak", "--since", "2026-10-02", "--after-ts", "2026-10-02T10:18:07+09:00",
               "--show", "3", "--json")
    assert a.command == "rm-leak" and a.since == "2026-10-02" and a.show == 3 and a.json


def test_cli_flags_rejected_on_other_commands(monkeypatch):
    with pytest.raises(SystemExit):
        _parse(monkeypatch, "report", "--since", "2026-10-02")


@pytest.mark.asyncio
async def test_cli_main_dispatch(monkeypatch, capsys):
    seen = {}

    async def fake_compute(since, **kw):
        seen["since"], seen["kw"] = since, kw
        return rm_leak.build_report(
            _rows(), since=since, until=None, market="domestic", scope="live", account_mode=None,
            after_ts=kw["after_ts"],
        )

    monkeypatch.setattr(audit_main, "compute_rm_leak", fake_compute)
    monkeypatch.setattr(sys, "argv", ["audit", "rm-leak", "--since", "2026-10-01",
                                      "--after-ts", "2026-10-02T04:00:00Z"])
    await audit_main._main()
    out = capsys.readouterr().out
    assert seen["since"] == D1 and seen["kw"]["market"] == "domestic" and seen["kw"]["scope"] == "live"
    assert seen["kw"]["after_ts"].tzinfo is not None
    assert "TOTAL" in out
