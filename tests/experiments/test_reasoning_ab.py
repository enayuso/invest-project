"""app.experiments.reasoning_ab — every I/O dependency faked (ledger clear,
memory clear, backtest_scope, run_pipeline itself). No network/DB/LLM
(F-stage). Verifies: the reasoning-arm override is applied during the
run_pipeline call and always restored (even on exception), the arm label is
written into the SAME "language" field app.experiments.language_ab_report
groups by, the fixed pipeline language survives separately, and the
isolation/interleaving properties mirror app.experiments.language_ab's own
(reused, not reimplemented) behavior."""

from __future__ import annotations

import json
from datetime import date

import pytest

from app.experiments import reasoning_ab
from app.state import PipelineState, PortfolioDecision, PortfolioRating


def _decision(rating=PortfolioRating.BUY, confidence=0.6) -> PortfolioDecision:
    return PortfolioDecision(
        rating=rating,
        executive_summary="summary",
        investment_thesis="thesis",
        confidence=confidence,
    )


@pytest.fixture
def call_log(monkeypatch):
    """Same fake shape as tests/experiments/test_language_ab.py's own
    `call_log` fixture — records isolation-dependency calls plus
    run_pipeline, and captures the reasoning setting AS SEEN by run_pipeline
    so a test can assert the arm was actually applied at call time, not just
    before/after."""
    log: list[tuple] = []

    async def fake_clear_positions(tickers):
        log.append(("clear_positions", tuple(tickers)))

    async def fake_clear_memory(tickers):
        log.append(("clear_memory", tuple(tickers)))

    class _FakeBacktestScope:
        def __enter__(self):
            log.append(("backtest_scope_enter",))
            return None

        def __exit__(self, *exc):
            log.append(("backtest_scope_exit",))
            return False

    def fake_backtest_scope():
        return _FakeBacktestScope()

    async def fake_run_pipeline(ticker, trade_date, as_of=None, language="en", profile=None, **kwargs):
        log.append((
            "run_pipeline", ticker, trade_date, as_of, language, profile,
            reasoning_ab.settings.research_manager_reasoning,
        ))
        state = PipelineState(ticker=ticker, trade_date=trade_date)
        state.final_decision = _decision()
        return state

    monkeypatch.setattr(reasoning_ab, "clear_backtest_positions", fake_clear_positions)
    monkeypatch.setattr(reasoning_ab, "clear_backtest_memory", fake_clear_memory)
    monkeypatch.setattr(reasoning_ab, "backtest_scope", fake_backtest_scope)
    monkeypatch.setattr(reasoning_ab, "run_pipeline", fake_run_pipeline)
    return log


# ---------------------------------------------------------------------------
# _reasoning_arm — the override itself
# ---------------------------------------------------------------------------


def test_reasoning_arm_true_sets_the_setting_and_restores_it(monkeypatch):
    monkeypatch.setattr(reasoning_ab.settings, "research_manager_reasoning", True)
    with reasoning_ab._reasoning_arm(False) as original:
        assert original is True
        assert reasoning_ab.settings.research_manager_reasoning is False
    assert reasoning_ab.settings.research_manager_reasoning is True


def test_reasoning_arm_restores_even_when_the_block_raises(monkeypatch):
    monkeypatch.setattr(reasoning_ab.settings, "research_manager_reasoning", True)
    with pytest.raises(RuntimeError):
        with reasoning_ab._reasoning_arm(False):
            raise RuntimeError("boom")
    # A leaked override would silently contaminate every later run in an
    # unattended multi-hour sweep — this is the whole point of the fixture.
    assert reasoning_ab.settings.research_manager_reasoning is True


# ---------------------------------------------------------------------------
# _run_pipeline_with_retry — the arm reaches run_pipeline's actual call
# ---------------------------------------------------------------------------


async def test_run_pipeline_with_retry_applies_reason_on_during_the_call(monkeypatch, call_log):
    monkeypatch.setattr(reasoning_ab.settings, "research_manager_reasoning", False)  # prod default, pre-call
    await reasoning_ab._run_pipeline_with_retry(
        "AAPL", date(2026, 7, 1), reasoning_ab.ARM_ON, "en", attempts=3, retry_delay=0
    )
    run_call = next(c for c in call_log if c[0] == "run_pipeline")
    assert run_call[6] is True  # research_manager_reasoning seen by run_pipeline
    assert reasoning_ab.settings.research_manager_reasoning is False  # restored after


async def test_run_pipeline_with_retry_applies_reason_off_during_the_call(monkeypatch, call_log):
    monkeypatch.setattr(reasoning_ab.settings, "research_manager_reasoning", True)  # prod default, pre-call
    await reasoning_ab._run_pipeline_with_retry(
        "AAPL", date(2026, 7, 1), reasoning_ab.ARM_OFF, "en", attempts=3, retry_delay=0
    )
    run_call = next(c for c in call_log if c[0] == "run_pipeline")
    assert run_call[6] is False
    assert reasoning_ab.settings.research_manager_reasoning is True  # restored after


async def test_run_pipeline_with_retry_passes_the_fixed_language_through_unchanged(call_log):
    await reasoning_ab._run_pipeline_with_retry(
        "AAPL", date(2026, 7, 1), reasoning_ab.ARM_ON, "en", attempts=3, retry_delay=0
    )
    run_call = next(c for c in call_log if c[0] == "run_pipeline")
    assert run_call[4] == "en"  # language — untouched by the arm


async def test_run_pipeline_with_retry_follows_the_documented_isolation_order(call_log):
    await reasoning_ab._run_pipeline_with_retry(
        "AAPL", date(2026, 7, 1), reasoning_ab.ARM_ON, "en", attempts=3, retry_delay=0
    )
    assert [c[0] for c in call_log] == [
        "clear_positions", "clear_memory", "backtest_scope_enter", "run_pipeline", "backtest_scope_exit",
    ]


# ---------------------------------------------------------------------------
# _plan_runs — arm interleaving (same time-drift confound rationale as
# language_ab._plan_runs)
# ---------------------------------------------------------------------------


def test_plan_runs_interleaves_both_arms_within_each_repeat():
    runs = reasoning_ab._plan_runs(["AAPL"], [date(2026, 7, 1)], repeats=1)
    assert [r[2] for r in runs] == [reasoning_ab.ARM_ON, reasoning_ab.ARM_OFF]


def test_plan_runs_flips_arm_order_on_alternate_repeats():
    runs = reasoning_ab._plan_runs(["AAPL"], [date(2026, 7, 1)], repeats=2)
    rep1 = [r[2] for r in runs if r[3] == 1]
    rep2 = [r[2] for r in runs if r[3] == 2]
    assert rep1 == [reasoning_ab.ARM_ON, reasoning_ab.ARM_OFF]
    assert rep2 == [reasoning_ab.ARM_OFF, reasoning_ab.ARM_ON]


def test_plan_runs_covers_every_ticker_date_arm_repeat_combination():
    tickers, dates, repeats = ["AAPL", "005930.KS"], [date(2026, 7, 1), date(2026, 7, 2)], 3
    runs = reasoning_ab._plan_runs(tickers, dates, repeats)
    expected = {
        (t, d, arm, r)
        for t in tickers for d in dates for r in range(1, repeats + 1) for arm in reasoning_ab._ARMS
    }
    assert set(runs) == expected
    assert len(runs) == len(tickers) * len(dates) * repeats * len(reasoning_ab._ARMS)


# ---------------------------------------------------------------------------
# _run_one — result file contents (arm-as-language alias, fixed pipeline
# language, consensus_override passthrough)
# ---------------------------------------------------------------------------


async def test_run_one_writes_a_result_file_with_the_arm_in_the_filename(tmp_path, call_log):
    await reasoning_ab._run_one(
        "AAPL", date(2026, 7, 1), reasoning_ab.ARM_ON, "en", repeat=1, out_dir=tmp_path,
        run_idx=1, total=1, resume=False, attempts=3, retry_delay=0,
    )
    out_path = tmp_path / "AAPL_2026-07-01_reason_on_r1.json"
    assert out_path.exists()


async def test_run_one_writes_the_arm_into_the_language_field_and_keeps_pipeline_language_separate(tmp_path, call_log):
    await reasoning_ab._run_one(
        "AAPL", date(2026, 7, 1), reasoning_ab.ARM_OFF, "en", repeat=1, out_dir=tmp_path,
        run_idx=1, total=1, resume=False, attempts=3, retry_delay=0,
    )
    record = json.loads((tmp_path / "AAPL_2026-07-01_reason_off_r1.json").read_text(encoding="utf-8"))
    assert record["language"] == "reason_off"  # alias for language_ab_report's existing rollups
    assert record["arm"] == "reason_off"
    assert record["pipeline_language"] == "en"  # the real, fixed pipeline language
    assert record["research_manager_reasoning"] is False


async def test_run_one_includes_consensus_override_in_the_record(tmp_path, call_log, monkeypatch):
    async def fake_run_pipeline_with_override(ticker, trade_date, as_of=None, language="en", profile=None, **kwargs):
        state = PipelineState(ticker=ticker, trade_date=trade_date)
        state.final_decision = _decision()
        state.consensus_override = {"tier": "neutral", "gated": True}
        return state

    monkeypatch.setattr(reasoning_ab, "run_pipeline", fake_run_pipeline_with_override)
    await reasoning_ab._run_one(
        "AAPL", date(2026, 7, 1), reasoning_ab.ARM_ON, "en", repeat=1, out_dir=tmp_path,
        run_idx=1, total=1, resume=False, attempts=3, retry_delay=0,
    )
    record = json.loads((tmp_path / "AAPL_2026-07-01_reason_on_r1.json").read_text(encoding="utf-8"))
    assert record["consensus_override"] == {"tier": "neutral", "gated": True}


async def test_run_one_skips_when_resume_and_file_already_exists(tmp_path, call_log):
    out_path = tmp_path / "AAPL_2026-07-01_reason_on_r1.json"
    out_path.write_text("{}", encoding="utf-8")
    await reasoning_ab._run_one(
        "AAPL", date(2026, 7, 1), reasoning_ab.ARM_ON, "en", repeat=1, out_dir=tmp_path,
        run_idx=1, total=1, resume=True, attempts=3, retry_delay=0,
    )
    assert call_log == []  # never even started


async def test_run_one_writes_the_record_then_raises_when_past_context_is_not_empty(tmp_path, monkeypatch, call_log):
    async def fake_run_pipeline_dirty(ticker, trade_date, as_of=None, language="en", profile=None, **kwargs):
        state = PipelineState(ticker=ticker, trade_date=trade_date)
        state.final_decision = _decision()
        state.past_context = "leaked prior case"
        return state

    monkeypatch.setattr(reasoning_ab, "run_pipeline", fake_run_pipeline_dirty)
    with pytest.raises(RuntimeError, match="memory isolation FAILED"):
        await reasoning_ab._run_one(
            "AAPL", date(2026, 7, 1), reasoning_ab.ARM_ON, "en", repeat=1, out_dir=tmp_path,
            run_idx=1, total=1, resume=False, attempts=3, retry_delay=0,
        )
    # Diagnosable: the record survives on disk despite the raise.
    assert (tmp_path / "AAPL_2026-07-01_reason_on_r1.json").exists()
