"""app.experiments.candle_ab — every I/O dependency faked (ledger clear,
memory clear, backtest_scope, run_pipeline, the real intraday_narration_note
itself). No network/DB/LLM (F-stage). Verifies: the `before` wrapper strips
the candle-pattern clause and the `after` wrapper does not, the loud-failure
contract when the clause is unexpectedly missing, the empty-note passthrough,
the arm-switching patch actually lands on `app.orchestrator.pipeline`'s own
binding, the tier parser, and the interleaved/rotated run plan."""

from __future__ import annotations

from datetime import date

import pytest

from app.experiments import candle_ab


_NOTE_WITH_CLAUSE = (
    "SESSION IN PROGRESS: 067290.KS's regular session is still open as of "
    "2026-09-15. Today has no open, no close, and no daily range yet."
    + candle_ab._CANDLE_CLAUSE
)


# ---------------------------------------------------------------------------
# _before_note / _after_note — the arm wrappers themselves
# ---------------------------------------------------------------------------


def test_before_note_strips_the_candle_clause(monkeypatch):
    monkeypatch.setattr(candle_ab, "_real_intraday_narration_note", lambda ticker, day: _NOTE_WITH_CLAUSE)
    result = candle_ab._before_note("067290.KS", date(2026, 9, 15))
    assert candle_ab._CANDLE_CLAUSE not in result
    assert "SESSION IN PROGRESS" in result


def test_after_note_keeps_the_candle_clause(monkeypatch):
    monkeypatch.setattr(candle_ab, "_real_intraday_narration_note", lambda ticker, day: _NOTE_WITH_CLAUSE)
    result = candle_ab._after_note("067290.KS", date(2026, 9, 15))
    assert candle_ab._CANDLE_CLAUSE in result
    assert result == _NOTE_WITH_CLAUSE


def test_before_note_raises_when_clause_is_missing_from_a_nonempty_note(monkeypatch):
    monkeypatch.setattr(
        candle_ab, "_real_intraday_narration_note",
        lambda ticker, day: "SESSION IN PROGRESS: some note with no candle clause at all.",
    )
    with pytest.raises(AssertionError, match="_CANDLE_CLAUSE not found"):
        candle_ab._before_note("067290.KS", date(2026, 9, 15))


def test_before_note_returns_empty_string_unchanged(monkeypatch):
    monkeypatch.setattr(candle_ab, "_real_intraday_narration_note", lambda ticker, day: "")
    assert candle_ab._before_note("067290.KS", date(2026, 9, 15)) == ""


def test_after_note_returns_empty_string_unchanged(monkeypatch):
    monkeypatch.setattr(candle_ab, "_real_intraday_narration_note", lambda ticker, day: "")
    assert candle_ab._after_note("067290.KS", date(2026, 9, 15)) == ""


# ---------------------------------------------------------------------------
# _candle_arm — patches app.orchestrator.pipeline's OWN binding, restores it
# ---------------------------------------------------------------------------


def test_candle_arm_patches_the_pipeline_modules_own_binding(monkeypatch):
    monkeypatch.setattr(candle_ab, "_real_intraday_narration_note", lambda ticker, day: _NOTE_WITH_CLAUSE)
    original = candle_ab.pipeline_module.intraday_narration_note
    captured: dict[str, str] = {}
    with candle_ab._candle_arm(candle_ab.ARM_BEFORE, captured):
        patched = candle_ab.pipeline_module.intraday_narration_note
        assert patched is not original
        note = patched("067290.KS", date(2026, 9, 15))
        assert candle_ab._CANDLE_CLAUSE not in note
        assert captured["note"] == note
    assert candle_ab.pipeline_module.intraday_narration_note is original


def test_candle_arm_after_arm_keeps_the_clause(monkeypatch):
    monkeypatch.setattr(candle_ab, "_real_intraday_narration_note", lambda ticker, day: _NOTE_WITH_CLAUSE)
    captured: dict[str, str] = {}
    with candle_ab._candle_arm(candle_ab.ARM_AFTER, captured):
        note = candle_ab.pipeline_module.intraday_narration_note("067290.KS", date(2026, 9, 15))
        assert candle_ab._CANDLE_CLAUSE in note
        assert captured["note"] == note


def test_candle_arm_restores_even_when_the_block_raises(monkeypatch):
    monkeypatch.setattr(candle_ab, "_real_intraday_narration_note", lambda ticker, day: _NOTE_WITH_CLAUSE)
    original = candle_ab.pipeline_module.intraday_narration_note
    with pytest.raises(RuntimeError):
        with candle_ab._candle_arm(candle_ab.ARM_AFTER, {}):
            raise RuntimeError("boom")
    assert candle_ab.pipeline_module.intraday_narration_note is original


# ---------------------------------------------------------------------------
# tier_counts — parses [<tier>] markers out of violation strings
# ---------------------------------------------------------------------------


def test_tier_counts_parses_completed_candle_and_closed_assertion():
    violations = [
        "MarketAnalyst: [completed_candle] 'x' vs SESSION-IN-PROGRESS ground truth",
        "PortfolioManager: [closed_assertion] 'y' vs SESSION-IN-PROGRESS ground truth",
    ]
    assert candle_ab.tier_counts(violations) == {"completed_candle": 1, "closed_assertion": 1}


def test_tier_counts_handles_multiple_of_the_same_tier():
    violations = [
        "A: [completed_candle] 'x' vs SESSION-IN-PROGRESS ground truth",
        "B: [completed_candle] 'y' vs SESSION-IN-PROGRESS ground truth",
    ]
    assert candle_ab.tier_counts(violations) == {"completed_candle": 2}


def test_tier_counts_empty_list_returns_empty_dict():
    assert candle_ab.tier_counts([]) == {}


# ---------------------------------------------------------------------------
# _plan_runs — arm interleaving
# ---------------------------------------------------------------------------


def test_plan_runs_interleaves_both_arms_within_each_repeat():
    runs = candle_ab._plan_runs(["067290.KS"], [date(2026, 9, 15)], repeats=1)
    assert [r[2] for r in runs] == [candle_ab.ARM_BEFORE, candle_ab.ARM_AFTER]


def test_plan_runs_flips_arm_order_on_alternate_repeats():
    runs = candle_ab._plan_runs(["067290.KS"], [date(2026, 9, 15)], repeats=2)
    rep1 = [r[2] for r in runs if r[3] == 1]
    rep2 = [r[2] for r in runs if r[3] == 2]
    assert rep1 == [candle_ab.ARM_BEFORE, candle_ab.ARM_AFTER]
    assert rep2 == [candle_ab.ARM_AFTER, candle_ab.ARM_BEFORE]


def test_plan_runs_covers_every_ticker_date_arm_repeat_combination():
    tickers, dates, repeats = ["067290.KS", "024060.KS"], [date(2026, 9, 15), date(2026, 9, 16)], 3
    runs = candle_ab._plan_runs(tickers, dates, repeats)
    expected = {
        (t, d, arm, r)
        for t in tickers for d in dates for r in range(1, repeats + 1) for arm in candle_ab._ARMS
    }
    assert set(runs) == expected
    assert len(runs) == len(tickers) * len(dates) * repeats * len(candle_ab._ARMS)
