"""app.audit.seat_stance — asyncpg pool/connection faked, no real Postgres.
Same fake-pool convention as tests/audit/test_rating_shift.py."""

from __future__ import annotations

import json
from datetime import date

import pytest

from app.audit import seat_stance


class FakeConn:
    def __init__(self, fetch_result=None):
        self.fetch_result = fetch_result if fetch_result is not None else []
        self.queries: list[tuple] = []

    async def fetch(self, query, *args):
        self.queries.append((query, args))
        return self.fetch_result


class _AcquireCM:
    def __init__(self, conn):
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, *exc):
        return False


class FakePool:
    def __init__(self, conn):
        self._conn = conn

    def acquire(self):
        return _AcquireCM(self._conn)


async def _async_return(value):
    return value


def _turn(speaker, content, round_):
    return {"speaker": speaker, "content": content, "round": round_}


def _row(id_, ticker, rating, trader_action, transcript, trade_date=None):
    return {
        "id": id_,
        "ticker": ticker,
        "trade_date": trade_date or date(2026, 9, 21),
        "rating": rating,
        "trader_action": trader_action,
        "full_transcript": transcript,
    }


BUY_TURN = "**Final Recommendation:** Buy — the setup is compelling."
HOLD_TURN = "**Final Recommendation:** Hold — risk/reward is unfavorable now."


# ---------------------------------------------------------------------------
# _build_seat_record
# ---------------------------------------------------------------------------

def test_single_round_has_no_similarity_or_softening():
    rec = seat_stance._build_seat_record("Aggressive", [_turn("Aggressive", BUY_TURN, 1)])
    assert rec.r1_final_similarity is None
    assert rec.near_duplicate is False
    assert rec.softened_from_bullish is None
    assert rec.r1.stance == "Buy"
    assert rec.final is rec.r1


def test_softened_from_bullish_true_when_r1_buy_and_final_hold():
    rec = seat_stance._build_seat_record(
        "Aggressive", [_turn("Aggressive", BUY_TURN, 1), _turn("Aggressive", HOLD_TURN, 2)],
    )
    assert rec.r1.stance == "Buy"
    assert rec.final.stance == "Hold"
    assert rec.softened_from_bullish is True


def test_softened_from_bullish_false_when_stance_stays_bullish():
    rec = seat_stance._build_seat_record(
        "Aggressive", [_turn("Aggressive", BUY_TURN, 1), _turn("Aggressive", BUY_TURN, 2)],
    )
    assert rec.softened_from_bullish is False


def test_softened_from_bullish_none_when_a_round_does_not_parse():
    rec = seat_stance._build_seat_record(
        "Aggressive", [_turn("Aggressive", "no verdict-shaped text here", 1), _turn("Aggressive", HOLD_TURN, 2)],
    )
    assert rec.r1.stance is None
    assert rec.softened_from_bullish is None


def test_near_duplicate_flagged_above_threshold():
    text = "A" * 500 + " concluding with Hold."
    almost_same = text + " tiny addition"
    rec = seat_stance._build_seat_record(
        "Conservative", [_turn("Conservative", text, 1), _turn("Conservative", almost_same, 2)],
    )
    assert rec.r1_final_similarity >= seat_stance._NEAR_DUPLICATE_RATIO
    assert rec.near_duplicate is True


def test_not_near_duplicate_when_rounds_differ_substantially():
    rec = seat_stance._build_seat_record(
        "Conservative",
        [_turn("Conservative", "Completely different opening argument about valuation.", 1),
         _turn("Conservative", "An entirely unrelated closing argument about liquidity risk.", 2)],
    )
    assert rec.near_duplicate is False


# ---------------------------------------------------------------------------
# _build_decision — desk_buy / final_round_buy_defenders derivation
# ---------------------------------------------------------------------------

def test_desk_buy_true_requires_both_rm_and_trader_bullish():
    transcript = {
        "research_plan": "**Recommendation**: Buy",
        "risk_debate": [
            _turn("Aggressive", BUY_TURN, 1), _turn("Aggressive", HOLD_TURN, 2),
            _turn("Conservative", HOLD_TURN, 1), _turn("Conservative", HOLD_TURN, 2),
            _turn("Neutral", BUY_TURN, 1), _turn("Neutral", HOLD_TURN, 2),
        ],
    }
    row = _row(1624, "036540.KS", "Hold", "Buy", transcript)
    decision = seat_stance._build_decision(row)
    assert decision.desk_buy is True
    assert decision.final_round_buy_defenders == 0
    assert decision.aggressive_softened is True


def test_desk_buy_false_when_trader_not_bullish():
    transcript = {"research_plan": "**Recommendation**: Buy", "risk_debate": [_turn("Aggressive", BUY_TURN, 1)]}
    row = _row(1, "A.KS", "Buy", "Hold", transcript)
    decision = seat_stance._build_decision(row)
    assert decision.desk_buy is False


def test_final_round_buy_defenders_counts_bullish_final_stances():
    transcript = {
        "research_plan": "**Recommendation**: Hold",
        "risk_debate": [
            _turn("Aggressive", BUY_TURN, 1),
            _turn("Conservative", HOLD_TURN, 1),
            _turn("Neutral", BUY_TURN, 1),
        ],
    }
    row = _row(2, "B.KS", "Hold", "Hold", transcript)
    decision = seat_stance._build_decision(row)
    assert decision.final_round_buy_defenders == 2  # Aggressive and Neutral, single-round each


# ---------------------------------------------------------------------------
# compute_seat_stance — validation, filters, skipping empty risk_debate
# ---------------------------------------------------------------------------

async def test_compute_seat_stance_ticker_and_market_mutually_exclusive():
    with pytest.raises(ValueError):
        await seat_stance.compute_seat_stance(ticker="A.KS", market="domestic")


async def test_compute_seat_stance_validates_market_scope_account_mode():
    with pytest.raises(ValueError):
        await seat_stance.compute_seat_stance(market="bogus")


async def test_compute_seat_stance_skips_rows_without_risk_debate(monkeypatch):
    rows = [
        _row(1, "A.KS", "Hold", "Hold", {"research_plan": "**Recommendation**: Hold"}),
        _row(2, "B.KS", "Hold", "Hold", {"research_plan": "**Recommendation**: Hold", "risk_debate": []}),
        _row(3, "C.KS", "Hold", "Hold",
             {"research_plan": "**Recommendation**: Hold", "risk_debate": [_turn("Neutral", HOLD_TURN, 1)]}),
    ]
    conn = FakeConn(fetch_result=rows)
    monkeypatch.setattr(seat_stance, "get_pool", lambda: _async_return(FakePool(conn)))
    monkeypatch.setattr(seat_stance, "get_current_user_id", lambda: "default")

    report = await seat_stance.compute_seat_stance()

    assert report.n_no_debate == 2
    assert report.n_scanned == 1
    assert report.decisions[0].id == 3


async def test_compute_seat_stance_applies_filters(monkeypatch):
    conn = FakeConn(fetch_result=[])
    monkeypatch.setattr(seat_stance, "get_pool", lambda: _async_return(FakePool(conn)))
    monkeypatch.setattr(seat_stance, "get_current_user_id", lambda: "default")

    await seat_stance.compute_seat_stance(
        market="domestic", scope="live", account_mode="real",
        since=date(2026, 9, 21), until=date(2026, 9, 21),
    )

    query, params = conn.queries[0]
    assert "ticker LIKE '%.KS'" in query
    assert "is_backtest = false" in query
    assert "account_mode = " in query
    assert "trade_date >= " in query
    assert "trade_date <= " in query
    assert "real" in params
    assert date(2026, 9, 21) in params


async def test_compute_seat_stance_handles_transcript_as_json_string(monkeypatch):
    transcript_str = json.dumps({
        "research_plan": "**Recommendation**: Hold",
        "risk_debate": [_turn("Neutral", HOLD_TURN, 1)],
    })
    conn = FakeConn(fetch_result=[_row(9, "A.KS", "Hold", "Hold", transcript_str)])
    monkeypatch.setattr(seat_stance, "get_pool", lambda: _async_return(FakePool(conn)))
    monkeypatch.setattr(seat_stance, "get_current_user_id", lambda: "default")

    report = await seat_stance.compute_seat_stance()

    assert report.n_scanned == 1
    assert report.decisions[0].seats["Neutral"].r1.stance == "Hold"


# ---------------------------------------------------------------------------
# render_seat_stance
# ---------------------------------------------------------------------------

def test_render_flags_desk_buy_vanished_and_near_duplicate():
    text = "B" * 500 + " concluding with Hold."
    almost_same = text + " tiny addition"
    decision = seat_stance.SeatStanceDecision(
        id=1624, ticker="036540.KS", trade_date=date(2026, 9, 21),
        rm_recommendation="Buy", trader_action="Buy", final_rating="Hold", desk_buy=True,
        seats={
            "Aggressive": seat_stance._build_seat_record(
                "Aggressive", [_turn("Aggressive", BUY_TURN, 1), _turn("Aggressive", HOLD_TURN, 2)]),
            "Conservative": seat_stance._build_seat_record(
                "Conservative", [_turn("Conservative", text, 1), _turn("Conservative", almost_same, 2)]),
            "Neutral": seat_stance._build_seat_record(
                "Neutral", [_turn("Neutral", BUY_TURN, 1), _turn("Neutral", HOLD_TURN, 2)]),
        },
        final_round_buy_defenders=0,
    )
    report = seat_stance.SeatStanceReport(
        market="domestic", scope="live", account_mode="real", n_scanned=1, n_no_debate=0,
        decisions=[decision],
    )
    rendered = seat_stance.render_seat_stance(report)
    assert "id=1624" in rendered
    assert "DESK-BUY-VANISHED" in rendered
    assert "NEAR-DUPLICATE" in rendered
    assert "SOFTENED-FROM-BULLISH" in rendered
    assert "provisional" in rendered.lower()


def test_render_no_flags_message_when_nothing_flagged():
    decision = seat_stance.SeatStanceDecision(
        id=1, ticker="A.KS", trade_date=date(2026, 9, 21),
        rm_recommendation="Hold", trader_action="Hold", final_rating="Hold", desk_buy=False,
        seats={"Neutral": seat_stance._build_seat_record("Neutral", [_turn("Neutral", HOLD_TURN, 1)])},
        final_round_buy_defenders=0,
    )
    report = seat_stance.SeatStanceReport(
        market=None, scope=None, account_mode=None, n_scanned=1, n_no_debate=0, decisions=[decision],
    )
    rendered = seat_stance.render_seat_stance(report)
    assert "No flagged decisions" in rendered
