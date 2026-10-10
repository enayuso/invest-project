"""app.experiments.language_ab — every I/O dependency faked (ledger clear,
memory clear, backtest_scope, run_pipeline itself). No network/DB/LLM
(F-stage). Verifies the isolation ordering and safety properties the module
docstring promises: positions cleared -> memory cleared -> run_pipeline
called inside backtest_scope() with as_of always set, one run's failure
doesn't stop the sweep, and a checkpoint file blocks a re-run under
--resume."""

from __future__ import annotations

import json
from datetime import date

import pytest

from app.experiments import language_ab
from app.llm import metrics as llm_metrics
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
    """Records, in order, every call into the three isolation dependencies
    plus run_pipeline itself, so a test can assert the exact sequence the
    module docstring promises."""
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

    async def fake_run_pipeline(ticker, trade_date, as_of=None, language="ko", profile=None, **kwargs):
        log.append(("run_pipeline", ticker, trade_date, as_of, language, profile))
        state = PipelineState(ticker=ticker, trade_date=trade_date)
        state.final_decision = _decision()
        return state

    monkeypatch.setattr(language_ab, "clear_backtest_positions", fake_clear_positions)
    monkeypatch.setattr(language_ab, "clear_backtest_memory", fake_clear_memory)
    monkeypatch.setattr(language_ab, "backtest_scope", fake_backtest_scope)
    monkeypatch.setattr(language_ab, "run_pipeline", fake_run_pipeline)
    return log


async def test_run_pipeline_with_retry_follows_the_documented_isolation_order(call_log):
    await language_ab._run_pipeline_with_retry("AAPL", date(2026, 7, 1), "ko", attempts=3, retry_delay=0)

    assert call_log == [
        ("clear_positions", ("AAPL",)),
        ("clear_memory", ("AAPL",)),
        ("backtest_scope_enter",),
        ("run_pipeline", "AAPL", date(2026, 7, 1), date(2026, 7, 1), "ko", language_ab.AIProfile.PRODUCTION),
        ("backtest_scope_exit",),
    ]


async def test_run_pipeline_with_retry_always_passes_as_of(call_log):
    # as_of MUST equal trade_date — a None as_of would run live and risk a
    # real KIS paper order (module docstring, step 3).
    await language_ab._run_pipeline_with_retry("AAPL", date(2026, 7, 1), "en", attempts=3, retry_delay=0)
    run_call = next(c for c in call_log if c[0] == "run_pipeline")
    assert run_call[3] == date(2026, 7, 1)  # as_of


async def test_run_pipeline_with_retry_retries_on_failure_and_eventually_succeeds(monkeypatch, call_log):
    monkeypatch.setattr(language_ab.settings, "backtest_pipeline_retry_attempts", 3)
    monkeypatch.setattr(language_ab.settings, "backtest_pipeline_retry_delay_seconds", 0.0)

    attempts = {"n": 0}
    real_run_pipeline = language_ab.run_pipeline

    async def flaky_run_pipeline(*args, **kwargs):
        attempts["n"] += 1
        if attempts["n"] < 2:
            raise RuntimeError("transient vLLM outage")
        return await real_run_pipeline(*args, **kwargs)

    monkeypatch.setattr(language_ab, "run_pipeline", flaky_run_pipeline)

    state, original_n_same, original_n_cross, degraded, llm_summary, attempts_used = await language_ab._run_pipeline_with_retry(
        "AAPL", date(2026, 7, 1), "ko", attempts=3, retry_delay=0
    )
    assert degraded == []
    assert attempts_used == 2

    assert attempts["n"] == 2
    assert state.final_decision is not None
    assert original_n_same == language_ab.settings.memory_n_same  # restored by then, so equals the real setting
    assert original_n_cross == language_ab.settings.memory_n_cross  # restored by then, so equals the real setting
    # clears re-run on every attempt (including the failed one)
    assert call_log.count(("clear_positions", ("AAPL",))) == 2
    assert call_log.count(("clear_memory", ("AAPL",))) == 2


async def test_run_pipeline_with_retry_raises_after_exhausting_attempts(monkeypatch, call_log):
    monkeypatch.setattr(language_ab.settings, "backtest_pipeline_retry_attempts", 2)
    monkeypatch.setattr(language_ab.settings, "backtest_pipeline_retry_delay_seconds", 0.0)

    async def always_fails(*args, **kwargs):
        raise RuntimeError("permanent failure")

    monkeypatch.setattr(language_ab, "run_pipeline", always_fails)

    with pytest.raises(RuntimeError, match="permanent failure"):
        await language_ab._run_pipeline_with_retry("AAPL", date(2026, 7, 1), "ko", attempts=3, retry_delay=0)


# ---------------------------------------------------------------------------
# _no_retrieved_memory — same-/cross-ticker memory isolation (3rd design fix,
# 2026-08-14: an earlier version of this only zeroed memory_n_cross, leaving
# memory_n_same untouched on the WRONG assumption that clear_backtest_memory
# alone handled same-ticker retrieval — it doesn't, for any ticker with real
# live trading history. See the contextmanager's own docstring.)
# ---------------------------------------------------------------------------


def test_no_retrieved_memory_zeroes_and_restores_both_settings(monkeypatch):
    monkeypatch.setattr(language_ab.settings, "memory_n_same", 2)
    monkeypatch.setattr(language_ab.settings, "memory_n_cross", 3)

    with language_ab._no_retrieved_memory() as (orig_same, orig_cross):
        assert (orig_same, orig_cross) == (2, 3)
        assert language_ab.settings.memory_n_same == 0
        assert language_ab.settings.memory_n_cross == 0

    assert language_ab.settings.memory_n_same == 2
    assert language_ab.settings.memory_n_cross == 3


def test_no_retrieved_memory_restores_even_when_the_block_raises(monkeypatch):
    monkeypatch.setattr(language_ab.settings, "memory_n_same", 1)
    monkeypatch.setattr(language_ab.settings, "memory_n_cross", 5)

    with pytest.raises(RuntimeError):
        with language_ab._no_retrieved_memory():
            assert language_ab.settings.memory_n_same == 0
            assert language_ab.settings.memory_n_cross == 0
            raise RuntimeError("boom")

    assert language_ab.settings.memory_n_same == 1
    assert language_ab.settings.memory_n_cross == 5


async def test_run_pipeline_with_retry_zeroes_both_memory_settings_during_run_pipeline_call(monkeypatch, call_log):
    monkeypatch.setattr(language_ab.settings, "memory_n_same", 2)
    monkeypatch.setattr(language_ab.settings, "memory_n_cross", 3)
    observed = {}

    async def fake_run_pipeline(ticker, trade_date, as_of=None, language="ko", profile=None, **kwargs):
        observed["memory_n_same"] = language_ab.settings.memory_n_same
        observed["memory_n_cross"] = language_ab.settings.memory_n_cross
        state = PipelineState(ticker=ticker, trade_date=trade_date)
        state.final_decision = _decision()
        return state

    monkeypatch.setattr(language_ab, "run_pipeline", fake_run_pipeline)

    await language_ab._run_pipeline_with_retry("AAPL", date(2026, 7, 1), "ko", attempts=3, retry_delay=0)

    assert observed["memory_n_same"] == 0  # zeroed for the actual pipeline call
    assert observed["memory_n_cross"] == 0  # zeroed for the actual pipeline call
    assert language_ab.settings.memory_n_same == 2  # restored afterwards
    assert language_ab.settings.memory_n_cross == 3  # restored afterwards


# ---------------------------------------------------------------------------
# _plan_runs — language interleaving (3rd design fix: time-drift confound)
# ---------------------------------------------------------------------------


def test_plan_runs_interleaves_languages_within_each_repeat():
    runs = language_ab._plan_runs(["AAPL"], [date(2026, 7, 1)], ["ko", "en"], repeats=1)
    languages = [r[2] for r in runs]
    assert languages == ["ko", "en"]  # not all-ko-then-all-en


def test_plan_runs_flips_language_order_on_alternate_repeats():
    runs = language_ab._plan_runs(["AAPL"], [date(2026, 7, 1)], ["ko", "en"], repeats=3)
    # (ticker, date, language, repeat)
    by_repeat = {r: [run[2] for run in runs if run[3] == r] for r in (1, 2, 3)}
    assert by_repeat[1] == ["ko", "en"]
    assert by_repeat[2] == ["en", "ko"]
    assert by_repeat[3] == ["ko", "en"]


def test_parse_args_rejects_attempts_below_one(monkeypatch, capsys):
    """P3 regression: --attempts 0 used to make _run_pipeline_with_retry's
    loop a no-op, falling through to a bare `raise AssertionError
    ("unreachable")` deep inside a retry helper instead of a clear CLI
    usage error at the argument-parsing boundary."""
    monkeypatch.setattr(
        "sys.argv", ["language_ab.py", "--tickers", "AAPL", "--dates", "2026-07-01", "--attempts", "0"]
    )
    with pytest.raises(SystemExit):
        language_ab._parse_args()
    assert "--attempts must be >= 1" in capsys.readouterr().err


def test_plan_runs_cyclically_rotates_three_or_more_languages():
    """P3 regression: a plain reversed() only swaps the first/last of 3+
    languages — the middle one never moves and keeps a fixed position bias
    across every repeat. Cyclic rotation moves every position."""
    runs = language_ab._plan_runs(["AAPL"], [date(2026, 7, 1)], ["a", "b", "c"], repeats=3)
    by_repeat = {r: [run[2] for run in runs if run[3] == r] for r in (1, 2, 3)}
    assert by_repeat[1] == ["a", "b", "c"]
    assert by_repeat[2] == ["b", "c", "a"]
    assert by_repeat[3] == ["c", "a", "b"]


def test_plan_runs_covers_every_ticker_date_language_repeat_combination():
    tickers, dates, languages, repeats = ["AAPL", "005930.KS"], [date(2026, 7, 1), date(2026, 7, 2)], ["ko", "en"], 3
    runs = language_ab._plan_runs(tickers, dates, languages, repeats)
    expected = {
        (t, d, lang, r) for t in tickers for d in dates for lang in languages for r in range(1, repeats + 1)
    }
    assert set(runs) == expected
    assert len(runs) == len(expected)  # no duplicates


def test_plan_runs_filenames_are_unaffected_by_the_new_ordering():
    # --resume keys off the filename alone, which encodes (ticker, date,
    # language, repeat) — reordering the runs list must not change any name.
    runs = language_ab._plan_runs(["AAPL"], [date(2026, 7, 1)], ["ko", "en"], repeats=2)
    names = {language_ab._run_filename(t, d, lang, r) for (t, d, lang, r) in runs}
    assert names == {
        "AAPL_2026-07-01_ko_r1.json",
        "AAPL_2026-07-01_en_r1.json",
        "AAPL_2026-07-01_en_r2.json",
        "AAPL_2026-07-01_ko_r2.json",
    }


# ---------------------------------------------------------------------------
# _run_one — checkpointing, resume, error recording, isolation warning
# ---------------------------------------------------------------------------


async def test_run_one_writes_a_result_file_with_the_expected_name(tmp_path, call_log):
    await language_ab._run_one("AAPL", date(2026, 7, 1), "ko", 1, tmp_path, 1, 1, resume=False, attempts=3, retry_delay=0)

    out_path = tmp_path / "AAPL_2026-07-01_ko_r1.json"
    assert out_path.exists()
    record = json.loads(out_path.read_text(encoding="utf-8"))
    assert record["ticker"] == "AAPL"
    assert record["date"] == "2026-07-01"
    assert record["language"] == "ko"
    assert record["repeat"] == 1
    assert record["final_decision"]["rating"] == "Buy"
    assert record["isolation_violation"] is False
    assert record["memory_n_same_override"] == 0
    assert record["memory_n_same_production"] == language_ab.settings.memory_n_same
    assert record["memory_n_cross_override"] == 0
    assert record["memory_n_cross_production"] == language_ab.settings.memory_n_cross


async def test_run_one_leaves_no_leftover_tmp_file_after_a_successful_write(tmp_path, call_log):
    """Regression: out_path is now written via a same-directory .tmpNNN file
    + os.replace (atomic on both POSIX and Windows) so a --resume check
    (out_path.exists()) can never observe a truncated write. The tmp file
    must not linger once the run completes normally."""
    await language_ab._run_one("AAPL", date(2026, 7, 1), "ko", 1, tmp_path, 1, 1, resume=False, attempts=3, retry_delay=0)

    names = {p.name for p in tmp_path.iterdir()}
    assert names == {"AAPL_2026-07-01_ko_r1.json"}


async def test_run_one_skips_when_resume_and_file_already_exists(tmp_path, call_log):
    out_path = tmp_path / "AAPL_2026-07-01_ko_r1.json"
    out_path.write_text("{}", encoding="utf-8")

    await language_ab._run_one("AAPL", date(2026, 7, 1), "ko", 1, tmp_path, 1, 1, resume=True, attempts=3, retry_delay=0)

    # No pipeline call should have happened — the file already existed.
    assert call_log == []
    assert out_path.read_text(encoding="utf-8") == "{}"  # untouched


async def test_run_one_does_not_skip_without_resume_even_if_file_exists(tmp_path, call_log):
    out_path = tmp_path / "AAPL_2026-07-01_ko_r1.json"
    out_path.write_text("{}", encoding="utf-8")

    await language_ab._run_one("AAPL", date(2026, 7, 1), "ko", 1, tmp_path, 1, 1, resume=False, attempts=3, retry_delay=0)

    record = json.loads(out_path.read_text(encoding="utf-8"))
    assert record["ticker"] == "AAPL"  # overwritten with a real result


async def test_run_one_records_a_failure_as_error_json_and_does_not_raise(tmp_path, monkeypatch, call_log):
    monkeypatch.setattr(language_ab.settings, "backtest_pipeline_retry_attempts", 1)

    async def always_fails(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(language_ab, "run_pipeline", always_fails)

    await language_ab._run_one("AAPL", date(2026, 7, 1), "ko", 1, tmp_path, 1, 1, resume=False, attempts=3, retry_delay=0)

    result_path = tmp_path / "AAPL_2026-07-01_ko_r1.json"
    error_path = tmp_path / "AAPL_2026-07-01_ko_r1_error.json"
    assert not result_path.exists()
    assert error_path.exists()
    error_record = json.loads(error_path.read_text(encoding="utf-8"))
    assert error_record["ticker"] == "AAPL"
    assert "boom" in error_record["error"]


async def test_run_one_writes_the_record_then_raises_when_past_context_is_not_empty(tmp_path, monkeypatch, call_log):
    """2026-08-14: isolation is now structurally guaranteed (memory_n_same
    AND memory_n_cross both zeroed — see _no_retrieved_memory), so a
    non-empty past_context means the override itself regressed, not a
    benign data-dependent case. Fail fast — write the record for diagnosis,
    then abort the whole sweep — rather than silently continue producing
    more incomparable runs (exactly what happened to 16/40 stage-1 runs
    before this fix existed)."""

    async def fake_run_pipeline(ticker, trade_date, as_of=None, language="ko", profile=None, **kwargs):
        state = PipelineState(ticker=ticker, trade_date=trade_date)
        state.final_decision = _decision()
        state.past_context = "leaked past case from another run"
        return state

    monkeypatch.setattr(language_ab, "run_pipeline", fake_run_pipeline)

    with pytest.raises(RuntimeError, match="memory isolation FAILED"):
        await language_ab._run_one("AAPL", date(2026, 7, 1), "ko", 1, tmp_path, 1, 1, resume=False, attempts=3, retry_delay=0)

    record = json.loads((tmp_path / "AAPL_2026-07-01_ko_r1.json").read_text(encoding="utf-8"))
    assert record["isolation_violation"] is True


# ---------------------------------------------------------------------------
# Degraded analyst reports + force_refresh (4th design fix)
#
# app.orchestrator.pipeline swallows an analyst failure and substitutes a
# placeholder report instead of raising, so a data-starved run still returns
# a final decision and looks like a clean success. These verify the runner
# notices, retries, and records it — and that the analyst-report cache is
# bypassed so repeats are genuinely independent.
# ---------------------------------------------------------------------------

_PLACEHOLDER = "(리포트를 생성하지 못했습니다 — httpx.ReadTimeout: )"


def test_degraded_report_names_spots_the_pipeline_placeholder():
    reports = {
        "Market Analyst": "real analysis",
        "Fundamentals Analyst": _PLACEHOLDER,
        "News + Macro Analyst": _PLACEHOLDER,
    }
    assert language_ab.degraded_report_names(reports) == ["Fundamentals Analyst", "News + Macro Analyst"]


def test_degraded_report_names_is_empty_for_a_clean_run():
    assert language_ab.degraded_report_names({"Market Analyst": "real analysis"}) == []


@pytest.mark.asyncio
async def test_run_pipeline_with_retry_passes_force_refresh_to_bypass_the_report_cache(monkeypatch, call_log):
    observed = {}

    async def fake_run_pipeline(ticker, trade_date, **kwargs):
        observed.update(kwargs)
        state = PipelineState(ticker=ticker, trade_date=trade_date)
        state.final_decision = _decision()
        return state

    monkeypatch.setattr(language_ab, "run_pipeline", fake_run_pipeline)
    await language_ab._run_pipeline_with_retry("AAPL", date(2026, 7, 1), "ko", attempts=1, retry_delay=0)

    assert observed["force_refresh"] is True


@pytest.mark.asyncio
async def test_run_pipeline_with_retry_retries_a_degraded_run_then_succeeds(monkeypatch, call_log):
    calls = {"n": 0}

    async def flaky(ticker, trade_date, **kwargs):
        calls["n"] += 1
        state = PipelineState(ticker=ticker, trade_date=trade_date)
        state.final_decision = _decision()
        state.reports = (
            {"Fundamentals Analyst": _PLACEHOLDER} if calls["n"] == 1 else {"Fundamentals Analyst": "real"}
        )
        return state

    monkeypatch.setattr(language_ab, "run_pipeline", flaky)
    _, _, _, degraded, _, attempts_used = await language_ab._run_pipeline_with_retry(
        "AAPL", date(2026, 7, 1), "ko", attempts=3, retry_delay=0
    )

    assert calls["n"] == 2  # retried the degraded run rather than accepting it
    assert degraded == []
    assert attempts_used == 2


@pytest.mark.asyncio
async def test_run_pipeline_with_retry_returns_degradation_after_exhausting_attempts(monkeypatch, call_log):
    async def always_degraded(ticker, trade_date, **kwargs):
        state = PipelineState(ticker=ticker, trade_date=trade_date)
        state.final_decision = _decision()
        state.reports = {"Fundamentals Analyst": _PLACEHOLDER}
        return state

    monkeypatch.setattr(language_ab, "run_pipeline", always_degraded)
    state, _, _, degraded, _, attempts_used = await language_ab._run_pipeline_with_retry(
        "AAPL", date(2026, 7, 1), "ko", attempts=2, retry_delay=0
    )

    # Returned rather than raised — the transcript explains WHY it degraded.
    assert degraded == ["Fundamentals Analyst"]
    assert state.final_decision is not None
    assert attempts_used == 2  # exhausted the budget, still degraded on the last try


# ---------------------------------------------------------------------------
# LLM call/token metrics collection (Part A) — drain-per-attempt discipline
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_llm_metrics():
    llm_metrics.clear()
    yield
    llm_metrics.clear()


@pytest.mark.asyncio
async def test_run_pipeline_with_retry_drains_stale_records_from_a_failed_attempt(monkeypatch, call_log):
    """A record left over from a PRIOR failed attempt must not leak into the
    successful attempt's summary — drain() at the top of every attempt loop
    iteration is what prevents that (module spec Part A2)."""
    calls = {"n": 0}

    async def flaky(ticker, trade_date, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            llm_metrics.record(
                label="attempt1_leftover", model="m", reasoning=False, structured=False,
                latency_seconds=1.0, prompt_tokens=999, completion_tokens=999, total_tokens=1998,
            )
            raise RuntimeError("transient failure")
        llm_metrics.record(
            label="attempt2_real", model="m", reasoning=False, structured=False,
            latency_seconds=1.0, prompt_tokens=10, completion_tokens=10, total_tokens=20,
        )
        state = PipelineState(ticker=ticker, trade_date=trade_date)
        state.final_decision = _decision()
        return state

    monkeypatch.setattr(language_ab, "run_pipeline", flaky)
    _, _, _, _, llm_summary, attempts_used = await language_ab._run_pipeline_with_retry(
        "AAPL", date(2026, 7, 1), "ko", attempts=3, retry_delay=0
    )

    assert attempts_used == 2
    assert llm_summary["llm_calls"] == 1
    assert llm_summary["llm_total_tokens"] == 20
    assert "attempt1_leftover" not in llm_summary["llm_by_label"]
    assert "attempt2_real" in llm_summary["llm_by_label"]


@pytest.mark.asyncio
async def test_run_pipeline_with_retry_summarizes_a_clean_success(monkeypatch, call_log):
    async def fake_run_pipeline(ticker, trade_date, **kwargs):
        llm_metrics.record(
            label="market_analyst", model="m", reasoning=False, structured=True,
            latency_seconds=1.0, prompt_tokens=100, completion_tokens=50, total_tokens=150,
            fallback_used=True,
        )
        llm_metrics.record(
            label="trader", model="m", reasoning=False, structured=False,
            latency_seconds=1.0, prompt_tokens=None, completion_tokens=None, total_tokens=None,
        )
        state = PipelineState(ticker=ticker, trade_date=trade_date)
        state.final_decision = _decision()
        return state

    monkeypatch.setattr(language_ab, "run_pipeline", fake_run_pipeline)
    _, _, _, _, llm_summary, attempts_used = await language_ab._run_pipeline_with_retry(
        "AAPL", date(2026, 7, 1), "ko", attempts=3, retry_delay=0
    )

    assert attempts_used == 1
    assert llm_summary["llm_calls"] == 2
    assert llm_summary["llm_prompt_tokens"] == 100  # None-safe sum
    assert llm_summary["llm_completion_tokens"] == 50
    assert llm_summary["llm_total_tokens"] == 150
    assert llm_summary["llm_fallback_calls"] == 1
    assert llm_summary["llm_by_label"]["market_analyst"] == {"calls": 1, "total_tokens": 150}
    assert llm_summary["llm_by_label"]["trader"] == {"calls": 1, "total_tokens": 0}
    assert llm_summary["llm_metrics_truncated"] is False


@pytest.mark.asyncio
async def test_run_one_writes_llm_metrics_fields_into_the_result_json(tmp_path, monkeypatch, call_log):
    async def fake_run_pipeline(ticker, trade_date, **kwargs):
        llm_metrics.record(
            label="market_analyst", model="m", reasoning=False, structured=True,
            latency_seconds=1.0, prompt_tokens=100, completion_tokens=50, total_tokens=150,
        )
        state = PipelineState(ticker=ticker, trade_date=trade_date)
        state.final_decision = _decision()
        return state

    monkeypatch.setattr(language_ab, "run_pipeline", fake_run_pipeline)
    await language_ab._run_one("AAPL", date(2026, 7, 1), "ko", 1, tmp_path, 1, 1, resume=False, attempts=3, retry_delay=0)

    record = json.loads((tmp_path / "AAPL_2026-07-01_ko_r1.json").read_text(encoding="utf-8"))
    assert record["llm_calls"] == 1
    assert record["llm_total_tokens"] == 150
    assert record["llm_attempts_used"] == 1
    assert record["llm_by_label"]["market_analyst"]["calls"] == 1


async def test_run_one_flags_a_degraded_run_in_its_result_file(tmp_path, monkeypatch, call_log, capsys):
    async def degraded_run(ticker, trade_date, **kwargs):
        state = PipelineState(ticker=ticker, trade_date=trade_date)
        state.final_decision = _decision()
        state.reports = {"Fundamentals Analyst": _PLACEHOLDER, "Market Analyst": "real"}
        return state

    monkeypatch.setattr(language_ab, "run_pipeline", degraded_run)
    await language_ab._run_one(
        "AAPL", date(2026, 7, 1), "ko", 1, tmp_path, 1, 1, resume=False, attempts=1, retry_delay=0
    )

    record = json.loads((tmp_path / "AAPL_2026-07-01_ko_r1.json").read_text(encoding="utf-8"))
    assert record["degraded"] is True
    assert record["degraded_reports"] == ["Fundamentals Analyst"]
    assert record["force_refresh"] is True
    assert "DEGRADED" in capsys.readouterr().out
