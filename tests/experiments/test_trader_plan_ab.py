"""app.experiments.trader_plan_ab — every I/O dependency faked (ledger
clear, memory clear, backtest_scope, run_pipeline itself). No network/DB/LLM
(F-stage). Covers the `--screen-first` flag added 2026-09-17 (module
docstring's addendum): the `after` arm only runs for a (ticker, trade_date,
repeat) unit whose `before` arm actually landed on RM=Underweight, and
`--resume` still works by reading a stored `before` result back off disk
rather than re-invoking `run_pipeline`."""

from __future__ import annotations

import json
from datetime import date

import pytest

from app.experiments import trader_plan_ab
from app.state import (
    PipelineState,
    PortfolioDecision,
    PortfolioRating,
    ResearchPlan,
    TraderAction,
    TraderProposal,
)


def _decision(rating=PortfolioRating.SELL) -> PortfolioDecision:
    return PortfolioDecision(
        rating=rating, executive_summary="summary", investment_thesis="thesis", confidence=0.6,
    )


def _plan(recommendation=PortfolioRating.UNDERWEIGHT) -> ResearchPlan:
    return ResearchPlan(recommendation=recommendation, rationale="r", strategic_actions="s")


def _proposal(text="trim the position by 25%") -> TraderProposal:
    return TraderProposal(action=TraderAction.SELL, reasoning=text)


@pytest.fixture
def call_log(monkeypatch):
    """Same fake shape as tests/experiments/test_reasoning_ab.py's own
    `call_log` fixture. `state_factory` lets each test control what
    `run_pipeline` returns per call (keyed by call index) so the
    before-arm's rm_rating can be set independently of the after-arm's."""
    log: list[tuple] = []
    states: list[PipelineState | Exception] = []

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

    async def fake_run_pipeline(ticker, trade_date, as_of=None, language="ko", profile=None, **kwargs):
        log.append(("run_pipeline", ticker, trade_date, kwargs.get("force_refresh")))
        if not states:
            raise AssertionError("test did not queue enough fake states/errors")
        next_item = states.pop(0)
        if isinstance(next_item, Exception):
            raise next_item
        return next_item

    monkeypatch.setattr(trader_plan_ab, "clear_backtest_positions", fake_clear_positions)
    monkeypatch.setattr(trader_plan_ab, "clear_backtest_memory", fake_clear_memory)
    monkeypatch.setattr(trader_plan_ab, "backtest_scope", fake_backtest_scope)
    monkeypatch.setattr(trader_plan_ab, "run_pipeline", fake_run_pipeline)
    return log, states


def _make_state(ticker: str, trade_date: date, rm_rating, trader_text="trim by 25%") -> PipelineState:
    state = PipelineState(ticker=ticker, trade_date=trade_date)
    state.research_plan = _plan(rm_rating)
    state.trader_proposal = _proposal(trader_text)
    state.final_decision = _decision()
    return state


def _run_one_calls(log):
    return [c for c in log if c[0] == "run_pipeline"]


# ---------------------------------------------------------------------------
# --screen-first OFF (default) — unchanged behavior
# ---------------------------------------------------------------------------


async def test_screen_first_off_runs_both_arms_for_every_unit(tmp_path, call_log):
    log, states = call_log
    states.append(_make_state("AAPL", date(2026, 1, 1), PortfolioRating.HOLD))
    states.append(_make_state("AAPL", date(2026, 1, 1), PortfolioRating.HOLD))

    before_record = await trader_plan_ab._run_one(
        "AAPL", date(2026, 1, 1), trader_plan_ab.ARM_BEFORE, "ko", repeat=1, out_dir=tmp_path,
        run_idx=1, total=2, resume=False, attempts=1, retry_delay=0,
    )
    after_record = await trader_plan_ab._run_one(
        "AAPL", date(2026, 1, 1), trader_plan_ab.ARM_AFTER, "ko", repeat=1, out_dir=tmp_path,
        run_idx=2, total=2, resume=False, attempts=1, retry_delay=0,
    )

    assert len(_run_one_calls(log)) == 2
    assert before_record["rm_underweight"] is False
    assert after_record["rm_underweight"] is False
    assert (tmp_path / "AAPL_2026-01-01_before_r1.json").exists()
    assert (tmp_path / "AAPL_2026-01-01_after_r1.json").exists()


async def test_screen_first_off_main_runs_both_arms_even_when_before_is_not_underweight(tmp_path, monkeypatch, call_log):
    log, states = call_log
    # _plan_runs interleaves before/after per repeat=1: before then after.
    states.append(_make_state("AAPL", date(2026, 1, 1), PortfolioRating.HOLD))
    states.append(_make_state("AAPL", date(2026, 1, 1), PortfolioRating.HOLD))

    monkeypatch.setattr(
        "sys.argv",
        [
            "trader_plan_ab", "--tickers", "AAPL", "--dates", "2026-01-01", "--repeats", "1",
            "--out-dir", str(tmp_path / "out"), "--attempts", "1",
        ],
    )
    await trader_plan_ab._main()

    assert len(_run_one_calls(log)) == 2  # after ran despite before landing on Hold
    assert (tmp_path / "out" / "AAPL_2026-01-01_before_r1.json").exists()
    assert (tmp_path / "out" / "AAPL_2026-01-01_after_r1.json").exists()


# ---------------------------------------------------------------------------
# --screen-first ON
# ---------------------------------------------------------------------------


async def test_screen_first_on_runs_after_when_before_is_underweight(tmp_path, monkeypatch, call_log):
    log, states = call_log
    states.append(_make_state("AAPL", date(2026, 1, 1), PortfolioRating.UNDERWEIGHT))
    states.append(_make_state("AAPL", date(2026, 1, 1), PortfolioRating.UNDERWEIGHT))

    monkeypatch.setattr(
        "sys.argv",
        [
            "trader_plan_ab", "--tickers", "AAPL", "--dates", "2026-01-01", "--repeats", "1",
            "--out-dir", str(tmp_path / "out"), "--attempts", "1", "--screen-first",
        ],
    )
    await trader_plan_ab._main()

    calls = _run_one_calls(log)
    assert len(calls) == 2
    assert (tmp_path / "out" / "AAPL_2026-01-01_before_r1.json").exists()
    assert (tmp_path / "out" / "AAPL_2026-01-01_after_r1.json").exists()


async def test_screen_first_on_skips_after_when_before_is_not_underweight(tmp_path, monkeypatch, call_log, capsys):
    log, states = call_log
    states.append(_make_state("AAPL", date(2026, 1, 1), PortfolioRating.HOLD))

    monkeypatch.setattr(
        "sys.argv",
        [
            "trader_plan_ab", "--tickers", "AAPL", "--dates", "2026-01-01", "--repeats", "1",
            "--out-dir", str(tmp_path / "out"), "--attempts", "1", "--screen-first",
        ],
    )
    await trader_plan_ab._main()

    calls = _run_one_calls(log)
    assert len(calls) == 1  # only the before arm called run_pipeline
    assert (tmp_path / "out" / "AAPL_2026-01-01_before_r1.json").exists()
    assert not (tmp_path / "out" / "AAPL_2026-01-01_after_r1.json").exists()
    assert not (tmp_path / "out" / "AAPL_2026-01-01_after_r1_error.json").exists()

    out = capsys.readouterr().out
    assert "SCREENED OUT" in out
    assert "rm_rating='Hold'" in out


async def test_screen_first_on_skips_after_when_before_errors(tmp_path, monkeypatch, call_log, capsys):
    log, states = call_log
    states.append(RuntimeError("pipeline exploded"))

    monkeypatch.setattr(
        "sys.argv",
        [
            "trader_plan_ab", "--tickers", "AAPL", "--dates", "2026-01-01", "--repeats", "1",
            "--out-dir", str(tmp_path / "out"), "--attempts", "1", "--screen-first",
        ],
    )
    await trader_plan_ab._main()  # must not crash

    calls = _run_one_calls(log)
    assert len(calls) == 1  # only the before arm attempted
    assert (tmp_path / "out" / "AAPL_2026-01-01_before_r1_error.json").exists()
    assert not (tmp_path / "out" / "AAPL_2026-01-01_after_r1.json").exists()
    assert not (tmp_path / "out" / "AAPL_2026-01-01_after_r1_error.json").exists()

    out = capsys.readouterr().out
    assert "SCREENED OUT" in out
    assert "rm_rating=None" in out


async def test_screen_first_with_resume_reads_existing_before_file_without_rerunning(tmp_path, monkeypatch, call_log):
    """An existing `before` result file (already on disk from a prior run)
    must be read back to decide whether to run `after` — without calling
    run_pipeline for the already-done before arm."""
    log, states = call_log
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    existing_before = out_dir / "AAPL_2026-01-01_before_r1.json"
    existing_before.write_text(
        json.dumps({"rm_rating": "Underweight", "rm_underweight": True}), encoding="utf-8",
    )
    # Only the after arm should actually call run_pipeline.
    states.append(_make_state("AAPL", date(2026, 1, 1), PortfolioRating.UNDERWEIGHT))

    monkeypatch.setattr(
        "sys.argv",
        [
            "trader_plan_ab", "--tickers", "AAPL", "--dates", "2026-01-01", "--repeats", "1",
            "--out-dir", str(out_dir), "--attempts", "1", "--screen-first", "--resume",
        ],
    )
    await trader_plan_ab._main()

    calls = _run_one_calls(log)
    assert len(calls) == 1  # before arm was NOT re-run
    assert calls[0][1:3] == ("AAPL", date(2026, 1, 1))
    assert (out_dir / "AAPL_2026-01-01_after_r1.json").exists()


async def test_screen_first_with_resume_skips_after_when_existing_before_was_not_underweight(tmp_path, monkeypatch, call_log):
    log, states = call_log
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "AAPL_2026-01-01_before_r1.json").write_text(
        json.dumps({"rm_rating": "Hold", "rm_underweight": False}), encoding="utf-8",
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "trader_plan_ab", "--tickers", "AAPL", "--dates", "2026-01-01", "--repeats", "1",
            "--out-dir", str(out_dir), "--attempts", "1", "--screen-first", "--resume",
        ],
    )
    await trader_plan_ab._main()

    assert _run_one_calls(log) == []  # nothing invoked at all — before resumed, after screened out
    assert not (out_dir / "AAPL_2026-01-01_after_r1.json").exists()


async def test_screen_first_with_resume_skips_after_when_existing_after_file_present(tmp_path, monkeypatch, call_log):
    """Existing `--resume` semantics for `after` are untouched: if the after
    file already exists, it's skipped without needing to re-check
    rm_underweight at all."""
    log, states = call_log
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "AAPL_2026-01-01_before_r1.json").write_text(
        json.dumps({"rm_rating": "Underweight", "rm_underweight": True}), encoding="utf-8",
    )
    (out_dir / "AAPL_2026-01-01_after_r1.json").write_text(
        json.dumps({"rm_rating": "Underweight", "rm_underweight": True}), encoding="utf-8",
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "trader_plan_ab", "--tickers", "AAPL", "--dates", "2026-01-01", "--repeats", "1",
            "--out-dir", str(out_dir), "--attempts", "1", "--screen-first", "--resume",
        ],
    )
    await trader_plan_ab._main()

    assert _run_one_calls(log) == []  # both arms resumed from disk, nothing invoked


# ---------------------------------------------------------------------------
# _plan_screen_first_units
# ---------------------------------------------------------------------------


def test_plan_screen_first_units_covers_every_ticker_date_repeat_combination():
    tickers, dates, repeats = ["AAPL", "005930.KS"], [date(2026, 1, 1), date(2026, 1, 2)], 2
    units = trader_plan_ab._plan_screen_first_units(tickers, dates, repeats)
    expected = {(t, d, r) for t in tickers for d in dates for r in range(1, repeats + 1)}
    assert set(units) == expected
    assert len(units) == len(tickers) * len(dates) * repeats
