"""run_pipeline — full wiring test with every I/O dependency (analysts, debate,
judgment nodes, ledger, memory, portfolio risk) faked. No network/DB/LLM
(D-stage). Exercises the sequencing/branching logic pipeline.py itself owns,
not the internals of the modules it calls (those are covered in their own
test files)."""

from __future__ import annotations

import asyncio
from datetime import date

import pytest

from app.execution.ledger import Position, RebalanceResult
from app.execution.portfolio_risk import PortfolioSnapshot
from app.orchestrator import debate as debate_mod
from app.orchestrator import pipeline
from app.web import state as web_state
from app.state import (
    DebateState,
    FlatPortfolioDecision,
    FlatPortfolioRating,
    PortfolioDecision,
    PortfolioRating,
    ResearchPlan,
    TraderAction,
    TraderProposal,
)


class FakeInfo:
    def __init__(self, name: str):
        self.name = name


def _install_common_fakes(monkeypatch, *, analyst_side_effects: dict[str, Exception] | None = None):
    """Wires every external dependency run_pipeline touches to a deterministic
    fake, for a ticker with as_of=None (live) unless overridden per-test."""
    analyst_side_effects = analyst_side_effects or {}
    calls = {"analysts": [], "rebalance": None, "store_pending": None, "append_note": None}

    async def fake_verified_snapshot(ticker, as_of=None):
        return "close=100.00"

    monkeypatch.setattr(pipeline, "get_verified_snapshot", fake_verified_snapshot)
    monkeypatch.setattr(pipeline.market_analyst, "analyze", _sync_wrap(analyst_side_effects, calls, "Market Analyst", "technical: bullish"))
    monkeypatch.setattr(pipeline.fundamentals_analyst, "analyze", _sync_wrap(analyst_side_effects, calls, "Fundamentals Analyst", "fundamentals: solid"))
    monkeypatch.setattr(pipeline.news_macro_analyst, "analyze", _sync_wrap(analyst_side_effects, calls, "News + Macro Analyst", "news: quiet"))
    monkeypatch.setattr(pipeline.sentiment_analyst, "analyze", _sync_wrap(analyst_side_effects, calls, "Sentiment Analyst", "sentiment: neutral"))

    monkeypatch.setattr(
        pipeline.bull_researcher, "build",
        lambda reports, language="ko", identity="", session_note="": (lambda state, party: _const("bull argument")),
    )
    monkeypatch.setattr(
        pipeline.bear_researcher, "build",
        lambda reports, language="ko", identity="", session_note="": (lambda state, party: _const("bear argument")),
    )

    async def fake_run_debate(
        *, state, speakers, max_rounds, on_round=None, language="ko", check_convergence=True,
        parallel_first_round=False, turn_guard=None,
    ):
        state.rounds_completed = 1
        return state

    monkeypatch.setattr(pipeline, "run_debate", fake_run_debate)

    plan = ResearchPlan(recommendation=PortfolioRating.BUY, rationale="r", strategic_actions="a")

    async def fake_synthesize(
        reports, debate, language="ko", identity="", held=True, session_note="", period_note="",
        position_context="",
    ):
        return plan

    monkeypatch.setattr(pipeline.research_manager, "synthesize", fake_synthesize)

    proposal = TraderProposal(action=TraderAction.BUY, reasoning="go long")

    async def fake_propose(
        reports, debate, research_plan, position_context="", language="ko", profile=None, identity="", held=True,
        session_note="", period_note="",
    ):
        return proposal

    monkeypatch.setattr(pipeline.trader, "propose", fake_propose)
    monkeypatch.setattr(
        pipeline.risk_debators, "build_all",
        lambda reports, trader_proposal, extra_context="", language="ko", identity="", held=True, session_note="", period_note="", sizing_context="", research_plan_note="": {
            "Aggressive": lambda state, party: _const("aggressive take"),
            "Conservative": lambda state, party: _const("conservative take"),
            "Neutral": lambda state, party: _const("neutral take"),
        },
    )

    async def fake_get_kr_risk_signals(ticker, as_of=None):
        return "risk signals"

    monkeypatch.setattr(pipeline, "get_kr_risk_signals", fake_get_kr_risk_signals)

    async def fake_get_similar_context(ticker, situation_text, as_of=None):
        return "past lesson: stopped out early last time"

    monkeypatch.setattr(pipeline, "get_similar_context", fake_get_similar_context)

    async def fake_get_same_day_sibling_context(ticker, trade_date, as_of=None, language="en"):
        return ""  # no same-day sibling by default — dedicated tests override this

    monkeypatch.setattr(pipeline, "get_same_day_sibling_context", fake_get_same_day_sibling_context)

    async def fake_get_position(ticker):
        return Position(cash=1000.0, shares=0)

    async def fake_get_last_trade(ticker):
        return None

    monkeypatch.setattr(pipeline, "get_position", fake_get_position)
    monkeypatch.setattr(pipeline, "get_last_trade", fake_get_last_trade)

    async def fake_get_atr_pct(ticker, as_of=None):
        return 0.015

    monkeypatch.setattr(pipeline, "get_atr_pct", fake_get_atr_pct)

    async def fake_get_last_close(ticker, as_of=None):
        return 150.0

    monkeypatch.setattr(pipeline, "get_last_close", fake_get_last_close)

    async def fake_get_ticker_display_info(ticker):
        return FakeInfo("Apple Inc.")

    monkeypatch.setattr(pipeline, "get_ticker_display_info", fake_get_ticker_display_info)

    # _company_identity (app.agents.identity_gate's alias source) touches
    # krx_master/overseas_master/yfinance directly, in addition to the
    # kis_provider wrapper mocked above — all three would otherwise attempt
    # a real network call for a fresh (uncached) ticker like "AAPL".
    async def _noop_ensure_loaded():
        return None

    monkeypatch.setattr(pipeline.krx_master, "ensure_loaded", _noop_ensure_loaded)
    monkeypatch.setattr(pipeline.krx_master, "get", lambda ticker: None)
    monkeypatch.setattr(pipeline.overseas_master, "ensure_loaded", _noop_ensure_loaded)
    monkeypatch.setattr(pipeline.overseas_master, "get", lambda ticker: None)

    async def fake_yf_get_ticker_display_info(ticker):
        return FakeInfo("Apple Inc.")

    monkeypatch.setattr(pipeline, "yf_get_ticker_display_info", fake_yf_get_ticker_display_info)

    # app.agents.identity_gate's internals (does this report actually name
    # the ticker) are covered by their own test file (test_identity_gate.py)
    # — this file's fixture report texts ("technical: bullish" etc.) are
    # deliberately minimal and were never meant to satisfy that check, same
    # as this file's own docstring says about not re-testing called modules'
    # internals here.
    def fake_enforce_identity(report, ticker, display_name, aliases, *, weak_aliases=frozenset(), language="en", kis_english_name="", agent_name=""):
        return report, None

    monkeypatch.setattr(pipeline, "enforce_identity", fake_enforce_identity)

    # Only reached when as_of is None (live mode) — mocked regardless so a
    # live-mode test never attempts a real Postgres connection or KIS call.
    async def fake_real_account_snapshot(market):
        return PortfolioSnapshot(per_ticker_value={"AAPL": 1500.0}, total_value=10000.0)

    async def fake_compute_correlation_clusters(tickers, as_of=None, lookback_days=None):
        return {}

    async def fake_current_drawdown_fraction(market, total_value):
        return 0.0

    # 2026-09-10: state.kis_broker() is otherwise unmocked, and on a cache
    # miss triggers a real KIS auth attempt with a 15s httpx timeout (see
    # app/brokers/kis/auth.py) at BOTH of pipeline.py's live-only call sites
    # (_execution_status_context and the portfolio-risk-context computation)
    # — measured adding 20-32s to 7 of this file's tests for a fallback path
    # ("no broker connected, falling back to internal ledger") these tests
    # already rely on regardless. Mocking it to return None immediately
    # produces byte-identical behavior in ~5s instead.
    async def fake_kis_broker():
        return None

    monkeypatch.setattr(pipeline, "real_account_snapshot", fake_real_account_snapshot)
    monkeypatch.setattr(pipeline, "compute_correlation_clusters", fake_compute_correlation_clusters)
    monkeypatch.setattr(pipeline, "current_drawdown_fraction", fake_current_drawdown_fraction)
    monkeypatch.setattr(web_state, "kis_broker", fake_kis_broker)

    # confidence=0.9: these tests exercise entry/allowlist/rebalance behavior
    # on a BUY, not the hysteresis gate (app.orchestrator.pipeline._apply_
    # hysteresis_gate) — an unset confidence would now get this flat-ticker
    # BUY downgraded to Hold by that gate, which is correct production
    # behavior but not what this fixture means to test.
    decision = PortfolioDecision(
        rating=PortfolioRating.BUY, executive_summary="s", investment_thesis="thesis text", confidence=0.9
    )

    async def fake_decide(*args, **kwargs):
        return decision

    monkeypatch.setattr(pipeline.portfolio_manager, "decide", fake_decide)

    async def fake_store_pending(
        ticker, trade_date, situation_text, rating, decision_text, transcript=None, is_backtest=False,
        trader_action=None, confidence=None, held_at_decision=None,
        card_reason_ko=None, card_reason_en=None, total_tokens=None, decision_trigger="auto",
        analysts_failed=None,
    ):
        calls["store_pending"] = {
            "ticker": ticker, "rating": rating, "transcript": transcript, "is_backtest": is_backtest,
            "trader_action": trader_action, "confidence": confidence, "held_at_decision": held_at_decision,
            "card_reason_ko": card_reason_ko, "card_reason_en": card_reason_en, "total_tokens": total_tokens,
            "decision_trigger": decision_trigger, "analysts_failed": analysts_failed,
        }
        return 42

    monkeypatch.setattr(pipeline, "store_pending", fake_store_pending)

    async def fake_append_execution_note(decision_id, note):
        calls["append_note"] = {"decision_id": decision_id, "note": note}

    monkeypatch.setattr(pipeline, "append_execution_note", fake_append_execution_note)

    rebalance_result_holder = {
        "result": RebalanceResult(side="buy", shares=5, shares_after=5, stop_loss=140.0, action_label="Buy")
    }

    async def fake_rebalance_to_target(ticker, trade_date, price, target_fraction, reason, stop_loss=None,
                                        current_atr_pct=None, portfolio_cap_override=None, decision_id=None,
                                        as_of=None, rating=None):
        calls["rebalance"] = {
            "ticker": ticker, "price": price, "target_fraction": target_fraction,
            "portfolio_cap_override": portfolio_cap_override, "decision_id": decision_id, "as_of": as_of,
            "rating": rating,
        }
        return rebalance_result_holder["result"]

    monkeypatch.setattr(pipeline, "rebalance_to_target", fake_rebalance_to_target)

    return calls, rebalance_result_holder


async def _const(value):
    return value


def _sync_wrap(analyst_side_effects, calls, name, report_text):
    async def _analyze(*args, **kwargs):
        calls["analysts"].append(name)
        if name in analyst_side_effects:
            raise analyst_side_effects[name]
        return report_text

    return _analyze


@pytest.fixture(autouse=True)
def _clear_pipeline_caches():
    pipeline._reports_cache.clear()
    pipeline._company_identity_cache.clear()
    yield
    pipeline._reports_cache.clear()
    pipeline._company_identity_cache.clear()


async def test_backtest_scope_tags_memory_row_even_without_as_of(monkeypatch):
    """2026-09-15: an experiment harness needs the LIVE call shape (no
    `as_of`) for a guard that only fires intraday, but runs it inside
    `backtest_scope()`. Its ledger writes were already isolated; the memory
    row was not, and 36 replay rows landed tagged as live decisions that
    `get_similar_context` then serves to real ones."""
    from app.config import backtest_scope

    calls, _ = _install_common_fakes(monkeypatch)

    with backtest_scope():
        await pipeline.run_pipeline("AAPL", date(2026, 7, 16))

    assert calls["store_pending"]["is_backtest"] is True


async def test_live_run_outside_backtest_scope_is_not_tagged(monkeypatch):
    calls, _ = _install_common_fakes(monkeypatch)

    await pipeline.run_pipeline("AAPL", date(2026, 7, 16))

    assert calls["store_pending"]["is_backtest"] is False


async def test_run_pipeline_backtest_mode_full_flow(monkeypatch):
    calls, _ = _install_common_fakes(monkeypatch)

    result = await pipeline.run_pipeline("AAPL", date(2026, 7, 16), as_of=date(2026, 7, 16))

    assert set(calls["analysts"]) == {"Market Analyst", "Fundamentals Analyst", "News + Macro Analyst", "Sentiment Analyst"}
    assert result.final_decision.rating == PortfolioRating.BUY
    assert result.decision_id == 42
    # Backtest mode: portfolio_cap_override defaults to DISABLED_PORTFOLIO_CAPS (inf total value).
    assert calls["rebalance"]["portfolio_cap_override"][1] == {}
    assert calls["rebalance"]["price"] == 150.0
    assert calls["append_note"] is None  # no adjustment_note on this rebalance result
    assert calls["store_pending"]["is_backtest"] is True
    # Trader proposed BUY; Portfolio Manager's final rating is also BUY here
    # (see _install_common_fakes) — recorded separately so a future stats
    # pass can measure how often the debate actually changed this.
    assert calls["store_pending"]["trader_action"] == "Buy"
    # decision_trigger defaults to "auto" when the caller (here, a plain
    # run_pipeline call with no override) doesn't pass one.
    assert calls["store_pending"]["decision_trigger"] == "auto"


async def test_run_pipeline_passes_manual_decision_trigger_through_to_store_pending(monkeypatch):
    calls, _ = _install_common_fakes(monkeypatch)

    await pipeline.run_pipeline(
        "AAPL", date(2026, 7, 16), as_of=date(2026, 7, 16), decision_trigger="manual",
    )

    assert calls["store_pending"]["decision_trigger"] == "manual"


async def test_run_pipeline_populates_sibling_context_separately_from_past_context(monkeypatch):
    # id=949 (950160.KS) gap: a same-day sibling's reasoning must reach the
    # Portfolio Manager's prompt via its OWN field/kwarg — not merged into
    # past_context (get_similar_context's resolved-only "graded lessons").
    calls, _ = _install_common_fakes(monkeypatch)

    async def fake_get_same_day_sibling_context(ticker, trade_date, as_of=None, language="en"):
        return (
            "Earlier decisions on THIS SAME ticker today (outcome not yet known):\n\n"
            "[00:21 | Sell | confidence 0.90 | execution outcome not recorded]\nWHY: quick ratio 0%"
        )

    monkeypatch.setattr(pipeline, "get_same_day_sibling_context", fake_get_same_day_sibling_context)

    captured = {}

    async def capturing_decide(*args, **kwargs):
        captured.update(kwargs)
        return PortfolioDecision(
            rating=PortfolioRating.BUY, executive_summary="s", investment_thesis="t", confidence=0.9
        )

    monkeypatch.setattr(pipeline.portfolio_manager, "decide", capturing_decide)

    result = await pipeline.run_pipeline("AAPL", date(2026, 7, 16))  # as_of=None -> live

    assert "quick ratio 0%" in result.sibling_context
    assert captured["sibling_context"] == result.sibling_context
    # past_context stays exactly what get_similar_context returned — the
    # sibling text must never leak into it.
    assert result.past_context == "past lesson: stopped out early last time"


async def test_run_pipeline_transcript_records_every_context_the_portfolio_manager_read(monkeypatch):
    """2026-08-25 fix: the transcript blob (the only durable record of what
    the Portfolio Manager actually saw) used to omit 7 of the context
    strings passed into decide() — a stored decision couldn't later answer
    "did the PM know about X" for a drawdown breaker, a VI halt, a same-day
    sibling, or a forced stop-loss exit. Asserted against decide()'s ACTUAL
    captured args, not hardcoded strings, so this keeps holding as the
    context text itself evolves."""
    calls, _ = _install_common_fakes(monkeypatch)

    captured = {}

    async def capturing_decide(
        trader_proposal, risk_debate, past_context, position_context, volatility_context,
        risk_cap_context, portfolio_risk_context, execution_status_context,
        *, sibling_context="", language="ko", profile=None, held=False, identity="", session_note="", period_note="",
        degradation_note="", desk_consensus_note="",
    ):
        captured.update(
            past_context=past_context,
            position_context=position_context,
            volatility_context=volatility_context,
            risk_cap_context=risk_cap_context,
            portfolio_risk_context=portfolio_risk_context,
            execution_status_context=execution_status_context,
            sibling_context=sibling_context,
        )
        return PortfolioDecision(
            rating=PortfolioRating.BUY, executive_summary="s", investment_thesis="t", confidence=0.9
        )

    monkeypatch.setattr(pipeline.portfolio_manager, "decide", capturing_decide)

    await pipeline.run_pipeline("AAPL", date(2026, 7, 16))  # as_of=None -> live, so the
    # live-only strings (portfolio_risk_context/execution_status_context) are non-empty

    transcript = calls["store_pending"]["transcript"]
    for key in (
        "past_context", "position_context", "volatility_context", "risk_cap_context",
        "portfolio_risk_context", "execution_status_context", "sibling_context",
    ):
        assert transcript[key] == captured[key], key
    assert transcript["portfolio_risk_context"] != ""
    assert transcript["execution_status_context"] != ""
    assert transcript["company_name"] == "Apple Inc."


async def test_run_pipeline_transcript_context_is_empty_in_backtest_mode(monkeypatch):
    """The two live-only context strings are stored as "" during a backtest
    — that IS what the Portfolio Manager received (see their own
    docstrings), not an unknown to be distinguished from a real empty."""
    calls, _ = _install_common_fakes(monkeypatch)

    await pipeline.run_pipeline("AAPL", date(2026, 7, 16), as_of=date(2026, 7, 16))

    transcript = calls["store_pending"]["transcript"]
    assert transcript["portfolio_risk_context"] == ""
    assert transcript["execution_status_context"] == ""


async def test_run_pipeline_transcript_is_json_serializable(monkeypatch):
    """Guards against a future context object that isn't a plain str —
    full_transcript is a jsonb column (app.memory.store.store_pending does
    json.dumps on it), so a non-serializable value would fail silently at
    write time in production (store_pending degrades quietly on any
    exception) while passing every mocked-store_pending test here."""
    import json

    calls, _ = _install_common_fakes(monkeypatch)

    await pipeline.run_pipeline("AAPL", date(2026, 7, 16))

    json.dumps(calls["store_pending"]["transcript"])  # must not raise


async def test_run_pipeline_backtest_mode_passes_as_of_through_to_sibling_context(monkeypatch):
    # Live-only feature — app.memory.store.get_same_day_sibling_context
    # itself enforces the as_of guard (see tests/memory/test_memory_store.py),
    # this only proves the pipeline actually threads as_of through rather
    # than hardcoding None, so a backtest can't accidentally get the live
    # behavior by omission.
    calls, _ = _install_common_fakes(monkeypatch)
    captured_as_of = {}

    async def fake_get_same_day_sibling_context(ticker, trade_date, as_of=None, language="en"):
        captured_as_of["value"] = as_of
        return ""

    monkeypatch.setattr(pipeline, "get_same_day_sibling_context", fake_get_same_day_sibling_context)

    result = await pipeline.run_pipeline("AAPL", date(2026, 7, 16), as_of=date(2026, 7, 16))

    assert captured_as_of["value"] == date(2026, 7, 16)
    assert result.sibling_context == ""


async def test_run_pipeline_retries_unrequested_cancelled_error_recording_action_label(monkeypatch):
    """id=377/id=949 shape: set_action_label raises CancelledError but this
    task itself was never asked to cancel — must retry once and succeed
    instead of leaving action_label permanently unrecorded."""
    calls, _ = _install_common_fakes(monkeypatch)
    attempts = {"n": 0}

    async def flaky_set_action_label(decision_id, action_label):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise asyncio.CancelledError()
        calls["set_action_label"] = (decision_id, action_label)

    monkeypatch.setattr(pipeline, "set_action_label", flaky_set_action_label)
    monkeypatch.setattr(pipeline, "task_was_actually_cancelled", lambda: False)

    result = await pipeline.run_pipeline("AAPL", date(2026, 7, 16), as_of=date(2026, 7, 16))

    assert attempts["n"] == 2
    assert calls["set_action_label"] == (42, "Buy")
    assert result.decision_id == 42


async def test_run_pipeline_propagates_real_task_cancellation_recording_action_label(monkeypatch):
    """A genuine outer cancellation (this task itself was told to stop) must
    propagate immediately, with no retry — retrying here would be masking a
    real shutdown/cancellation signal, not recovering from a spurious one."""
    calls, _ = _install_common_fakes(monkeypatch)
    attempts = {"n": 0}

    async def always_cancelled_set_action_label(decision_id, action_label):
        attempts["n"] += 1
        raise asyncio.CancelledError()

    monkeypatch.setattr(pipeline, "set_action_label", always_cancelled_set_action_label)
    monkeypatch.setattr(pipeline, "task_was_actually_cancelled", lambda: True)

    with pytest.raises(asyncio.CancelledError):
        await pipeline.run_pipeline("AAPL", date(2026, 7, 16), as_of=date(2026, 7, 16))

    assert attempts["n"] == 1


async def test_run_pipeline_gives_up_after_one_retry_on_persistent_unrequested_cancellation(monkeypatch):
    """Even an unrequested CancelledError shouldn't retry forever — after
    the one retry also fails, it must propagate (loudly logged) rather than
    loop or silently swallow it."""
    calls, _ = _install_common_fakes(monkeypatch)
    attempts = {"n": 0}

    async def always_cancelled_set_action_label(decision_id, action_label):
        attempts["n"] += 1
        raise asyncio.CancelledError()

    monkeypatch.setattr(pipeline, "set_action_label", always_cancelled_set_action_label)
    monkeypatch.setattr(pipeline, "task_was_actually_cancelled", lambda: False)

    with pytest.raises(asyncio.CancelledError):
        await pipeline.run_pipeline("AAPL", date(2026, 7, 16), as_of=date(2026, 7, 16))

    assert attempts["n"] == 2


async def test_run_pipeline_reuses_cached_reports_and_skips_analysts(monkeypatch):
    calls, _ = _install_common_fakes(monkeypatch)
    pipeline._reports_cache[("AAPL", "2026-07-16", None, pipeline.DEFAULT_LANGUAGE)] = (
        __import__("time").monotonic(),
        {"Market Analyst": "cached bullish report", "Fundamentals Analyst": "x",
         "News + Macro Analyst": "x", "Sentiment Analyst": "x"},
    )

    result = await pipeline.run_pipeline("AAPL", date(2026, 7, 16))

    assert calls["analysts"] == []  # analysts never invoked — served from cache
    assert result.reports["Market Analyst"] == "cached bullish report"


async def test_run_pipeline_force_refresh_bypasses_cache_and_reruns_analysts(monkeypatch):
    # AI-engine efficiency audit (2026-07-27): the SEMI_AUTO "다시 분석" button
    # threads force_refresh=True through run_cycle_now -> run_pipeline so an
    # explicit user request never silently serves reports from within the
    # 15-minute TTL window (see run_pipeline's own docstring on force_refresh).
    calls, _ = _install_common_fakes(monkeypatch)
    pipeline._reports_cache[("AAPL", "2026-07-16", None, pipeline.DEFAULT_LANGUAGE)] = (
        __import__("time").monotonic(),
        {"Market Analyst": "cached bullish report", "Fundamentals Analyst": "x",
         "News + Macro Analyst": "x", "Sentiment Analyst": "x"},
    )

    result = await pipeline.run_pipeline("AAPL", date(2026, 7, 16), force_refresh=True)

    assert set(calls["analysts"]) == {"Market Analyst", "Fundamentals Analyst", "News + Macro Analyst", "Sentiment Analyst"}
    assert result.reports["Market Analyst"] == "technical: bullish"  # fresh, not the pre-seeded cache value
    # A forced run still writes fresh results back, so a later non-forced
    # caller (e.g. a chat follow-up) benefits from the cache as usual.
    cached_after = pipeline._cached_reports("AAPL", "2026-07-16", None, pipeline.DEFAULT_LANGUAGE)
    assert cached_after is not None
    assert cached_after["Market Analyst"] == "technical: bullish"


async def test_run_pipeline_degrades_failed_analyst_to_placeholder(monkeypatch):
    calls, _ = _install_common_fakes(
        monkeypatch, analyst_side_effects={"News + Macro Analyst": RuntimeError("LLM connection dropped")}
    )

    result = await pipeline.run_pipeline("AAPL", date(2026, 7, 16), as_of=date(2026, 7, 16))

    assert "리포트를 생성하지 못했습니다" in result.reports["News + Macro Analyst"]
    assert result.reports["Market Analyst"] == "technical: bullish"  # siblings still succeed
    # A degraded run must not be cached (would keep serving the placeholder for the full TTL).
    assert pipeline._cached_reports("AAPL", "2026-07-16", date(2026, 7, 16), "ko") is None
    # 2026-08-25: a degraded run used to be indistinguishable from a full
    # one anywhere in the DB — win-rate stats pooled a 1-analyst decision
    # with a 4-analyst one. Recorded on PipelineState, the transcript, and
    # a dedicated memory_entries column (see app.memory.db's own comment).
    assert result.degraded_analysts == ["News + Macro Analyst"]
    assert calls["store_pending"]["analysts_failed"] == 1
    assert calls["store_pending"]["transcript"]["degraded_analysts"] == ["News + Macro Analyst"]


async def test_analyst_cancellation_is_recorded_as_a_degraded_analyst(monkeypatch):
    """2026-08-25: asyncio.CancelledError subclasses BaseException, not
    Exception, so it used to fall through the `isinstance(result, Exception)`
    classifier's else-branch as if it were report text, crashing at
    enforce_identity with TypeError instead of degrading normally. An
    UNREQUESTED cancellation (task_was_actually_cancelled() False, the
    common spurious-cancellation shape — see id=377/id=949) must be recorded
    like any other analyst failure, not crash the run."""
    calls, _ = _install_common_fakes(
        monkeypatch, analyst_side_effects={"News + Macro Analyst": asyncio.CancelledError()}
    )
    monkeypatch.setattr(pipeline, "task_was_actually_cancelled", lambda: False)

    result = await pipeline.run_pipeline("AAPL", date(2026, 7, 16), as_of=date(2026, 7, 16))

    assert result.degraded_analysts == ["News + Macro Analyst"]
    assert "CancelledError" in result.reports["News + Macro Analyst"]
    assert calls["store_pending"]["analysts_failed"] == 1
    # A degraded run must not be cached, same as an ordinary exception.
    assert pipeline._cached_reports("AAPL", "2026-07-16", date(2026, 7, 16), "ko") is None


async def test_genuine_analyst_cancellation_propagates_instead_of_degrading(monkeypatch):
    """A REAL outer cancellation (task_was_actually_cancelled() True — this
    task actually being told to stop, e.g. shutdown) must tear the run down,
    not get recorded as a data-quality degradation."""
    _install_common_fakes(
        monkeypatch, analyst_side_effects={"News + Macro Analyst": asyncio.CancelledError()}
    )
    monkeypatch.setattr(pipeline, "task_was_actually_cancelled", lambda: True)

    with pytest.raises(asyncio.CancelledError):
        await pipeline.run_pipeline("AAPL", date(2026, 7, 16), as_of=date(2026, 7, 16))


async def test_run_pipeline_records_zero_failed_analysts_on_a_clean_run(monkeypatch):
    calls, _ = _install_common_fakes(monkeypatch)

    await pipeline.run_pipeline("AAPL", date(2026, 7, 16), as_of=date(2026, 7, 16))

    # Explicitly 0, not None — a clean run is a KNOWN-good run, not unknown.
    assert calls["store_pending"]["analysts_failed"] == 0
    assert calls["store_pending"]["transcript"]["degraded_analysts"] == []


async def test_run_pipeline_records_multiple_failed_analysts(monkeypatch):
    calls, _ = _install_common_fakes(
        monkeypatch,
        analyst_side_effects={
            "Market Analyst": RuntimeError("LLM connection dropped"),
            "Sentiment Analyst": RuntimeError("LLM timeout"),
        },
    )

    result = await pipeline.run_pipeline("AAPL", date(2026, 7, 16), as_of=date(2026, 7, 16))

    assert calls["store_pending"]["analysts_failed"] == 2
    assert set(result.degraded_analysts) == {"Market Analyst", "Sentiment Analyst"}
    assert set(calls["store_pending"]["transcript"]["degraded_analysts"]) == {"Market Analyst", "Sentiment Analyst"}


async def test_run_pipeline_cache_hit_records_zero_failed_analysts(monkeypatch):
    """A cached run was necessarily a CLEAN run (a degraded cycle is never
    cached, see _store_cached_reports's call site) — 0 here is a KNOWN
    clean, not merely an unchecked default."""
    calls, _ = _install_common_fakes(monkeypatch)
    pipeline._store_cached_reports(
        "AAPL", "2026-07-16", date(2026, 7, 16), "ko",
        {"Market Analyst": "cached", "Fundamentals Analyst": "cached",
         "News + Macro Analyst": "cached", "Sentiment Analyst": "cached"},
    )

    await pipeline.run_pipeline("AAPL", date(2026, 7, 16), as_of=date(2026, 7, 16))

    assert calls["store_pending"]["analysts_failed"] == 0


async def test_run_pipeline_applies_identity_banner_and_skips_caching_on_violation(monkeypatch):
    # Uses the REAL app.agents.identity_gate.enforce_identity (undoing this
    # file's usual bypass — see _install_common_fakes) to verify the wiring
    # itself: a report naming the wrong company gets a correction banner and
    # the whole cycle's reports are withheld from the 15-minute cache, same
    # as a degraded/failed analyst run. See app.agents.identity_gate's
    # module docstring for the 2026-08-20 316140.KS incident this exists to
    # catch.
    from app.agents.identity_gate import enforce_identity

    calls, _ = _install_common_fakes(monkeypatch)
    monkeypatch.setattr(pipeline, "enforce_identity", enforce_identity)

    # The other three analysts' default fixture text ("fundamentals: solid"
    # etc., from _install_common_fakes) names no company at all, so with the
    # real gate active they'd ALSO trip Detector B — give them a clean,
    # correctly-named report here so only the deliberately-bad Market
    # Analyst report is the thing under test.
    async def fake_analyze_clean(*args, **kwargs):
        return "Apple Inc. (AAPL): no notable change."

    monkeypatch.setattr(pipeline.fundamentals_analyst, "analyze", fake_analyze_clean)
    monkeypatch.setattr(pipeline.news_macro_analyst, "analyze", fake_analyze_clean)
    monkeypatch.setattr(pipeline.sentiment_analyst, "analyze", fake_analyze_clean)

    async def fake_analyze_market(*args, **kwargs):
        calls["analysts"].append("Market Analyst")
        return "KB Financial Group (AAPL) reported strong iPhone sales this quarter."

    monkeypatch.setattr(pipeline.market_analyst, "analyze", fake_analyze_market)

    result = await pipeline.run_pipeline("AAPL", date(2026, 7, 16), as_of=date(2026, 7, 16))

    assert "IDENTITY CORRECTION" in result.reports["Market Analyst"]
    assert "iPhone sales" in result.reports["Market Analyst"]  # original content preserved
    assert result.reports["Fundamentals Analyst"] == "Apple Inc. (AAPL): no notable change."  # unaffected, clean
    assert len(result.identity_violations) == 1
    assert "Market Analyst" in result.identity_violations[0]
    # A violating run must not be cached (would keep serving the banner-
    # corrected text for the full TTL instead of retrying clean next time).
    assert pipeline._cached_reports("AAPL", "2026-07-16", date(2026, 7, 16), "ko") is None


async def test_run_pipeline_name_omission_gets_a_neutral_note_and_is_still_cached(monkeypatch):
    # 2026-09-01: a report that never states a company name (the News + Macro
    # Analyst's pure-macro reports, the Sentiment skip report before its own
    # fix) is a name OMISSION, not a wrong-company substitution — it goes to
    # identity_unnamed with a neutral note, and unlike identity_violations it
    # does NOT withhold the cycle from the 15-minute report cache.
    from app.agents.identity_gate import enforce_identity

    _install_common_fakes(monkeypatch)
    monkeypatch.setattr(pipeline, "enforce_identity", enforce_identity)

    async def fake_named(*args, **kwargs):
        return "Apple Inc. (AAPL): steady quarter, no surprises."

    async def fake_unnamed(*args, **kwargs):
        return "The macro backdrop is risk-off; rates stay elevated and the dollar is firm."

    monkeypatch.setattr(pipeline.market_analyst, "analyze", fake_named)
    monkeypatch.setattr(pipeline.fundamentals_analyst, "analyze", fake_named)
    monkeypatch.setattr(pipeline.sentiment_analyst, "analyze", fake_named)
    monkeypatch.setattr(pipeline.news_macro_analyst, "analyze", fake_unnamed)

    result = await pipeline.run_pipeline("AAPL", date(2026, 7, 16), as_of=date(2026, 7, 16))

    assert result.identity_violations == []
    assert result.identity_unnamed == ["News + Macro Analyst: unnamed"]
    assert "IDENTITY NOTE" in result.reports["News + Macro Analyst"]
    assert "IDENTITY CORRECTION" not in result.reports["News + Macro Analyst"]
    # ...and the run WAS cached (an omission is not a cache-blocker). Default
    # language is "en" (app.llm.language.DEFAULT_LANGUAGE).
    assert pipeline._cached_reports("AAPL", "2026-07-16", date(2026, 7, 16), "en") is not None


async def test_run_pipeline_threads_held_to_research_manager_and_trader(monkeypatch):
    """2026-08-28 domestic AI audit wiring test: `held` (computed once from
    `_position_context`, already used by the Portfolio Manager) must reach
    the Research Manager and Trader too — see
    app.agents.research_manager.synthesize/app.agents.trader.propose's own
    `held` docstrings for the flat-schema mechanism this feeds."""
    calls, _ = _install_common_fakes(monkeypatch)

    async def fake_get_position(ticker):
        return Position(cash=1000.0, shares=7)  # non-zero — held=True

    monkeypatch.setattr(pipeline, "get_position", fake_get_position)

    held_seen = {}

    plan = ResearchPlan(recommendation=PortfolioRating.SELL, rationale="r", strategic_actions="a")

    async def fake_synthesize(
        reports, debate, language="ko", identity="", held=True, session_note="", period_note="",
        position_context="",
    ):
        held_seen["research_manager"] = held
        return plan

    monkeypatch.setattr(pipeline.research_manager, "synthesize", fake_synthesize)

    proposal = TraderProposal(action=TraderAction.SELL, reasoning="trim")

    async def fake_propose(
        reports, debate, research_plan, position_context="", language="ko", profile=None, identity="", held=True,
        session_note="", period_note="",
    ):
        held_seen["trader"] = held
        return proposal

    monkeypatch.setattr(pipeline.trader, "propose", fake_propose)

    await pipeline.run_pipeline("005930.KS", date(2026, 7, 16), as_of=date(2026, 7, 16))

    assert held_seen == {"research_manager": True, "trader": True}


async def test_run_pipeline_threads_position_context_to_research_manager(monkeypatch):
    """2026-09-16 fix (engine audit item 2): `_position_context` was already
    computed before this call and threaded to the Trader/Portfolio Manager,
    but not to the Research Manager — even though the pipeline deliberately
    moved the `_position_context` fetch above this call in 2026-08-28 so the
    Research Manager would not reason about position blind. See
    app.agents.research_manager.synthesize's own `position_context`
    docstring."""
    calls, _ = _install_common_fakes(monkeypatch)

    async def fake_get_position(ticker):
        return Position(cash=1000.0, shares=7)

    monkeypatch.setattr(pipeline, "get_position", fake_get_position)

    captured = {}

    plan = ResearchPlan(recommendation=PortfolioRating.SELL, rationale="r", strategic_actions="a")

    async def fake_synthesize(
        reports, debate, language="ko", identity="", held=True, session_note="", period_note="",
        position_context="",
    ):
        captured["research_manager"] = position_context
        return plan

    monkeypatch.setattr(pipeline.research_manager, "synthesize", fake_synthesize)

    await pipeline.run_pipeline("005930.KS", date(2026, 7, 16), as_of=date(2026, 7, 16))

    assert captured["research_manager"]
    assert "7" in captured["research_manager"]


async def test_run_pipeline_threads_held_and_records_illegal_flat_ratings(monkeypatch):
    """2026-09-01, id=1156/1157/1158: `held` must reach risk_debators.build_all
    (flat -> Buy/Hold-only menu), and a flat-ticker debate turn that declares
    Sell/Underweight/Overweight is recorded in the decision record so an
    operator can see the prompt fix lapse."""
    calls, _ = _install_common_fakes(monkeypatch)

    async def fake_flat_position(ticker):
        return Position(cash=1000.0, shares=0)  # flat -> held=False

    monkeypatch.setattr(pipeline, "get_position", fake_flat_position)

    held_seen = {}

    def fake_build_all(
        reports, trader_proposal, extra_context="", language="ko", identity="", held=True,
        session_note="", period_note="", sizing_context="", research_plan_note="",
    ):
        held_seen["risk"] = held
        return {
            "Aggressive": lambda s, p: _const("open a starter position"),
            "Conservative": lambda s, p: _const("CONCLUSION: Underweight (25%) — trim exposure"),
            "Neutral": lambda s, p: _const("CONCLUSION: Buy (100%)"),
        }

    monkeypatch.setattr(pipeline.risk_debators, "build_all", fake_build_all)
    monkeypatch.setattr(pipeline, "run_debate", debate_mod.run_debate)

    async def no_convergence(*a, **k):
        raise AssertionError("convergence check must not run")

    monkeypatch.setattr(debate_mod.llm_client, "structured_complete", no_convergence)

    result = await pipeline.run_pipeline("005930.KS", date(2026, 7, 16), as_of=date(2026, 7, 16))

    assert held_seen == {"risk": False}
    assert result.risk_rating_violations
    assert any("Underweight" in v for v in result.risk_rating_violations)
    assert calls["store_pending"]["transcript"]["risk_rating_violations"] == result.risk_rating_violations


async def test_transcript_carries_neutral_reversal(monkeypatch):
    """2026-09-04, id=1212/1217: check_neutral_reversal's record must reach the
    stored transcript under "neutral_reversal" — present as `{}` when this
    fixture's single-round debate has no round-1/final pair to compare, not
    simply absent from the dict, so a later SQL
    `full_transcript ? 'neutral_reversal'` existence-check is meaningful."""
    calls, _ = _install_common_fakes(monkeypatch)

    result = await pipeline.run_pipeline("005930.KS", date(2026, 7, 16), as_of=date(2026, 7, 16))

    assert result.neutral_reversal == {}
    assert "neutral_reversal" in calls["store_pending"]["transcript"]
    assert calls["store_pending"]["transcript"]["neutral_reversal"] == result.neutral_reversal


async def test_transcript_carries_halt_status_violations(monkeypatch):
    """2026-09-07, id=1238 (351320.KS): halt_status_guard.scan_halt_claims's
    findings must reach the stored transcript under "halt_status_violations"
    — present as `[]` here (a backtest, as_of is not None, so
    execution_status_context is always "" and the guard never even attempts a
    scan — see its own docstring), not simply absent, so a later SQL
    `full_transcript ? 'halt_status_violations'` existence-check is
    meaningful."""
    calls, _ = _install_common_fakes(monkeypatch)

    result = await pipeline.run_pipeline("005930.KS", date(2026, 7, 16), as_of=date(2026, 7, 16))

    assert result.halt_status_violations == []
    assert "halt_status_violations" in calls["store_pending"]["transcript"]
    assert calls["store_pending"]["transcript"]["halt_status_violations"] == result.halt_status_violations


async def test_transcript_carries_technical_claim_violations(monkeypatch):
    """2026-09-07 fix: technical_claim_guard.scan_technical_claims's findings
    and the indicator_block_available flag must both reach the stored
    transcript, present (not simply absent) even in the common clean case, so
    a later SQL `full_transcript ? 'technical_claim_violations'` /
    `full_transcript ? 'indicator_block_available'` existence-check is
    meaningful."""
    calls, _ = _install_common_fakes(monkeypatch)

    result = await pipeline.run_pipeline("005930.KS", date(2026, 7, 16), as_of=date(2026, 7, 16))

    assert result.technical_claim_violations == []
    assert "technical_claim_violations" in calls["store_pending"]["transcript"]
    assert (
        calls["store_pending"]["transcript"]["technical_claim_violations"] == result.technical_claim_violations
    )
    assert "indicator_block_available" in calls["store_pending"]["transcript"]


async def test_transcript_carries_long_horizon_block_available(monkeypatch):
    """2026-09-16 fix: market_analyst.analyze's `long_horizon_block_out`
    existed but the pipeline never passed it, so the 52-week block had no
    audit trail while indicator_block_available/flow_block_available both
    do for their own blocks."""
    calls, _ = _install_common_fakes(monkeypatch)

    result = await pipeline.run_pipeline("005930.KS", date(2026, 7, 16), as_of=date(2026, 7, 16))

    assert "long_horizon_block_available" in calls["store_pending"]["transcript"]
    assert (
        calls["store_pending"]["transcript"]["long_horizon_block_available"]
        == result.long_horizon_block_available
    )


async def test_transcript_carries_refusal_violations(monkeypatch):
    """2026-09-14 fix: refusal_guard.scan_refusal's findings must reach the
    stored transcript under "refusal_violations", present (not simply absent)
    even in the common clean case, so a later SQL
    `full_transcript ? 'refusal_violations'` existence-check is meaningful."""
    calls, _ = _install_common_fakes(monkeypatch)

    result = await pipeline.run_pipeline("005930.KS", date(2026, 7, 16), as_of=date(2026, 7, 16))

    assert result.refusal_violations == []
    assert "refusal_violations" in calls["store_pending"]["transcript"]
    assert calls["store_pending"]["transcript"]["refusal_violations"] == result.refusal_violations


async def test_transcript_carries_business_description_violations(monkeypatch):
    """2026-09-07, id=1237 (294630.KS/서남):
    business_description_guard.scan_business_description's findings must
    reach the stored transcript under "business_description_violations" —
    present as `[]` here (this fixture's fake analyst reports never trigger
    an identity_gate violation, so the guard is never even called), not
    simply absent, so a later SQL
    `full_transcript ? 'business_description_violations'` existence-check is
    meaningful."""
    calls, _ = _install_common_fakes(monkeypatch)

    result = await pipeline.run_pipeline("005930.KS", date(2026, 7, 16), as_of=date(2026, 7, 16))

    assert result.business_description_violations == []
    assert "business_description_violations" in calls["store_pending"]["transcript"]
    assert (
        calls["store_pending"]["transcript"]["business_description_violations"]
        == result.business_description_violations
    )
    assert calls["store_pending"]["transcript"]["indicator_block_available"] == result.indicator_block_available


async def test_transcript_carries_injection_flags(monkeypatch):
    """Fix C (2026-09-04, #212): the model's own one-sentence injection reason,
    formerly log-only, must reach the stored transcript under "injection_flags"
    — present as `[]` in the common no-flag case, not simply absent, so a later
    SQL `full_transcript ? 'injection_flags'` existence-check is meaningful."""
    calls, _ = _install_common_fakes(monkeypatch)

    result = await pipeline.run_pipeline("005930.KS", date(2026, 7, 16), as_of=date(2026, 7, 16))

    assert result.injection_flags == []
    assert "injection_flags" in calls["store_pending"]["transcript"]
    assert calls["store_pending"]["transcript"]["injection_flags"] == result.injection_flags


# ---------------- risk-debate sizing context (2026-09-02, id=1170, 498400.KS) ----------------
#
# The risk debate previously argued position sizing blind to the real cap:
# Conservative asserted "we cannot reduce position size (it's all-or-nothing)"
# and Neutral's deciding turn adopted it as verified fact. The fix computes
# risk_cap_context/volatility_context once, before the debate, and threads
# them into risk_debators.build_all's new sizing_context kwarg — reusing the
# SAME value at the existing PM-facing call site rather than recomputing it
# (the anti-drift property these tests lock).


async def test_run_pipeline_routes_sizing_context_to_the_risk_debate(monkeypatch):
    calls, _ = _install_common_fakes(monkeypatch)
    seen = {}

    def fake_build_all(reports, tp, extra_context="", language="ko", identity="", held=True,
                        session_note="", period_note="", sizing_context="", research_plan_note=""):
        seen["sizing_context"] = sizing_context
        return {p: (lambda s, party: _const("take")) for p in ("Aggressive", "Conservative", "Neutral")}

    monkeypatch.setattr(pipeline.risk_debators, "build_all", fake_build_all)

    await pipeline.run_pipeline("AAPL", date(2026, 7, 16), as_of=date(2026, 7, 16))

    # get_atr_pct is faked to 0.015 (1.5%, at/below the sizing target) and
    # get_last_close to 150.0 with a $1000 fake position -> both the
    # volatility narrative and the risk-cap narrative are real, computed
    # text, not stubs — this is the regression lock for id=1170.
    assert "volatility" in seen["sizing_context"].lower()
    assert "risk-cap" in seen["sizing_context"].lower() or "risk cap" in seen["sizing_context"].lower() or "new shares" in seen["sizing_context"].lower()


async def test_run_pipeline_risk_cap_context_computed_once_and_shared(monkeypatch):
    """Anti-drift lock: _risk_cap_context must be invoked exactly once per
    ticker/cycle, and the SAME value must reach risk_debators.build_all,
    portfolio_manager.decide, and the stored transcript — never two
    separately-computed figures that could disagree (the exact failure mode
    live-rebalance-equity-basis-mismatch-2026-08-14 describes for equity
    reads more broadly)."""
    calls, _ = _install_common_fakes(monkeypatch)

    call_count = {"n": 0}

    async def fake_risk_cap_context(ticker, price, *, as_of=None):
        call_count["n"] += 1
        return f"FAKE RISK CAP for price={price}"

    monkeypatch.setattr(pipeline, "_risk_cap_context", fake_risk_cap_context)

    seen_build_all = {}

    def fake_build_all(reports, tp, extra_context="", language="ko", identity="", held=True,
                        session_note="", period_note="", sizing_context="", research_plan_note=""):
        seen_build_all["sizing_context"] = sizing_context
        return {p: (lambda s, party: _const("take")) for p in ("Aggressive", "Conservative", "Neutral")}

    monkeypatch.setattr(pipeline.risk_debators, "build_all", fake_build_all)

    captured_decide = {}

    async def capturing_decide(trader_proposal, risk_debate, past_context, position_context, volatility_context,
                                risk_cap_context, portfolio_risk_context, execution_status_context,
                                *, sibling_context="", language="ko", profile=None, held=False, identity="",
                                session_note="", period_note="", degradation_note="", desk_consensus_note=""):
        captured_decide["risk_cap_context"] = risk_cap_context
        return PortfolioDecision(rating=PortfolioRating.BUY, executive_summary="s", investment_thesis="t", confidence=0.9)

    monkeypatch.setattr(pipeline.portfolio_manager, "decide", capturing_decide)

    await pipeline.run_pipeline("AAPL", date(2026, 7, 16), as_of=date(2026, 7, 16))

    assert call_count["n"] == 1
    assert captured_decide["risk_cap_context"] == "FAKE RISK CAP for price=150.0"
    assert captured_decide["risk_cap_context"] in seen_build_all["sizing_context"]
    assert calls["store_pending"]["transcript"]["risk_cap_context"] == captured_decide["risk_cap_context"]


async def test_run_pipeline_sizing_context_fallback_when_pre_debate_price_fetch_fails(monkeypatch):
    """If the pre-debate guarded price fetch fails, the pipeline must still
    complete, the debate proceeds without the risk-cap block (today's
    pre-fix behavior for that piece), and the PM-facing call site's own
    fallback (`if risk_cap_context is None: ...`) still produces a valid
    risk_cap_context."""
    calls, _ = _install_common_fakes(monkeypatch)

    call_count = {"n": 0}

    async def flaky_get_last_close(ticker, as_of=None):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("simulated pre-debate price-fetch failure")
        return 150.0

    monkeypatch.setattr(pipeline, "get_last_close", flaky_get_last_close)

    seen_build_all = {}

    def fake_build_all(reports, tp, extra_context="", language="ko", identity="", held=True,
                        session_note="", period_note="", sizing_context="", research_plan_note=""):
        seen_build_all["sizing_context"] = sizing_context
        return {p: (lambda s, party: _const("take")) for p in ("Aggressive", "Conservative", "Neutral")}

    monkeypatch.setattr(pipeline.risk_debators, "build_all", fake_build_all)

    captured_decide = {}

    async def capturing_decide(trader_proposal, risk_debate, past_context, position_context, volatility_context,
                                risk_cap_context, portfolio_risk_context, execution_status_context,
                                *, sibling_context="", language="ko", profile=None, held=False, identity="",
                                session_note="", period_note="", degradation_note="", desk_consensus_note=""):
        captured_decide["risk_cap_context"] = risk_cap_context
        return PortfolioDecision(rating=PortfolioRating.BUY, executive_summary="s", investment_thesis="t", confidence=0.9)

    monkeypatch.setattr(pipeline.portfolio_manager, "decide", capturing_decide)

    result = await pipeline.run_pipeline("AAPL", date(2026, 7, 16), as_of=date(2026, 7, 16))

    assert result.final_decision is not None  # pipeline completed despite the failure
    assert "new shares" not in seen_build_all["sizing_context"].lower()
    assert "risk-cap" not in seen_build_all["sizing_context"].lower()
    # The PM-facing fallback still computed a real, non-empty risk cap figure
    # (not the "data unavailable" placeholder _risk_cap_context returns for a
    # non-positive price or a resolve_equity_basis failure) — the boilerplate
    # phrase "volatility data is unavailable" can legitimately appear inside a
    # real computed risk-cap narrative, so check for the specific placeholder
    # sentence, not the substring "unavailable".
    assert captured_decide["risk_cap_context"]
    assert "Risk-cap data unavailable" not in captured_decide["risk_cap_context"]
    assert "new shares" in captured_decide["risk_cap_context"].lower() or "ZERO added shares" in captured_decide["risk_cap_context"]
    assert calls["store_pending"]["transcript"]["risk_cap_context"] == captured_decide["risk_cap_context"]


async def test_run_pipeline_pre_debate_sizing_call_happens_after_research_debate_before_risk_debate(monkeypatch):
    """Ordering lock: the pre-debate risk_cap_context computation must run
    AFTER the Bull/Bear research debate and BEFORE the risk debate's
    run_debate call — not moved earlier (it would add latency ahead of the
    research debate, which has no need for it) nor later (that would defeat
    the whole point of routing it to the debate)."""
    calls, _ = _install_common_fakes(monkeypatch)
    order = []

    async def fake_run_debate(*, state, speakers, max_rounds, on_round=None, language="ko",
                               check_convergence=True, parallel_first_round=False, turn_guard=None):
        order.append("research_debate" if "Bull" in speakers else "risk_debate")
        state.rounds_completed = 1
        return state

    monkeypatch.setattr(pipeline, "run_debate", fake_run_debate)

    async def fake_risk_cap_context(ticker, price, *, as_of=None):
        order.append("pre_debate_risk_cap_context")
        return "cap text"

    monkeypatch.setattr(pipeline, "_risk_cap_context", fake_risk_cap_context)

    await pipeline.run_pipeline("AAPL", date(2026, 7, 16), as_of=date(2026, 7, 16))

    assert order == ["research_debate", "pre_debate_risk_cap_context", "risk_debate"]


async def test_run_pipeline_execution_inputs_unaffected_by_pre_debate_calls(monkeypatch):
    """The most important constraint this design protects: current_atr_pct/
    fill_price that feed rebalance_to_target must still come exclusively
    from the untouched context_tasks gather, never from the new pre-debate
    _atr_context/get_last_close calls — even when those two return different
    values. Locks against the pre-debate calls becoming a second, drifting
    source of execution-input truth."""
    calls, _ = _install_common_fakes(monkeypatch)

    atr_calls = {"n": 0}

    async def fake_atr_context(ticker, as_of, *, held):
        atr_calls["n"] += 1
        if atr_calls["n"] == 1:
            return 0.99, "PRE-DEBATE-ONLY ATR TEXT (must never reach execution)"
        return 0.015, "GATHER ATR TEXT"

    monkeypatch.setattr(pipeline, "_atr_context", fake_atr_context)

    close_calls = {"n": 0}

    async def fake_get_last_close(ticker, as_of=None):
        close_calls["n"] += 1
        return 999.0 if close_calls["n"] == 1 else 150.0

    monkeypatch.setattr(pipeline, "get_last_close", fake_get_last_close)

    captured_rebalance = {}

    async def fake_rebalance_to_target(ticker, trade_date, price, target_fraction, reason, stop_loss=None,
                                        current_atr_pct=None, portfolio_cap_override=None, decision_id=None,
                                        as_of=None, rating=None):
        captured_rebalance["price"] = price
        captured_rebalance["current_atr_pct"] = current_atr_pct
        return RebalanceResult(side="buy", shares=5, shares_after=5, stop_loss=140.0, action_label="Buy")

    monkeypatch.setattr(pipeline, "rebalance_to_target", fake_rebalance_to_target)

    await pipeline.run_pipeline("AAPL", date(2026, 7, 16), as_of=date(2026, 7, 16))

    assert atr_calls["n"] >= 2
    assert close_calls["n"] >= 2
    # The pre-debate (first) calls returned 0.99 / 999.0 — the executed
    # trade must show the gather's (second-call) values instead.
    assert captured_rebalance["current_atr_pct"] == 0.015
    assert captured_rebalance["price"] == 150.0


async def test_run_pipeline_closed_session_threads_empty_session_note(monkeypatch):
    """Backtest / closed session: intraday_narration_note returns "" and every
    downstream node's prompt stays byte-identical (2026-09-01, id=1157)."""
    calls, _ = _install_common_fakes(monkeypatch)
    seen = {}

    async def fake_synthesize(
        reports, debate, language="ko", identity="", held=True, session_note="", period_note="",
        position_context="",
    ):
        seen["research_manager"] = session_note
        return ResearchPlan(recommendation=PortfolioRating.HOLD, rationale="r", strategic_actions="a")

    async def fake_propose(reports, debate, plan, position_context="", language="ko", profile=None,
                           identity="", held=True, session_note="", period_note=""):
        seen["trader"] = session_note
        return TraderProposal(action=TraderAction.HOLD, reasoning="r")

    def fake_build_all(
        reports, tp, extra_context="", language="ko", identity="", held=True,
        session_note="", period_note="", sizing_context="", research_plan_note="",
    ):
        seen["risk"] = session_note
        return {p: (lambda s, party: _const("take")) for p in ("Aggressive", "Conservative", "Neutral")}

    async def capturing_decide(*args, session_note="", period_note="", **kwargs):
        seen["portfolio_manager"] = session_note
        return PortfolioDecision(rating=PortfolioRating.HOLD, executive_summary="s", investment_thesis="t", confidence=0.7)

    def fake_bull_build(reports, language="ko", identity="", session_note=""):
        seen["bull"] = session_note
        return lambda state, party: _const("bull argument")

    def fake_bear_build(reports, language="ko", identity="", session_note=""):
        seen["bear"] = session_note
        return lambda state, party: _const("bear argument")

    monkeypatch.setattr(pipeline.research_manager, "synthesize", fake_synthesize)
    monkeypatch.setattr(pipeline.trader, "propose", fake_propose)
    monkeypatch.setattr(pipeline.risk_debators, "build_all", fake_build_all)
    monkeypatch.setattr(pipeline.portfolio_manager, "decide", capturing_decide)
    monkeypatch.setattr(pipeline.bull_researcher, "build", fake_bull_build)
    monkeypatch.setattr(pipeline.bear_researcher, "build", fake_bear_build)

    result = await pipeline.run_pipeline("005930.KS", date(2026, 7, 16), as_of=date(2026, 7, 16))

    assert seen == {
        "research_manager": "", "trader": "", "risk": "", "portfolio_manager": "", "bull": "", "bear": "",
    }
    assert result.session_in_progress is False
    assert calls["store_pending"]["transcript"]["session_in_progress"] is False


async def test_run_pipeline_intraday_threads_the_warning_to_every_downstream_node(monkeypatch):
    calls, _ = _install_common_fakes(monkeypatch)
    monkeypatch.setattr(pipeline, "intraday_narration_note", lambda t, d: "SESSION IN PROGRESS: test note")
    seen = {}

    async def fake_synthesize(
        reports, debate, language="ko", identity="", held=True, session_note="", period_note="",
        position_context="",
    ):
        seen["research_manager"] = session_note
        return ResearchPlan(recommendation=PortfolioRating.HOLD, rationale="r", strategic_actions="a")

    async def fake_propose(reports, debate, plan, position_context="", language="ko", profile=None,
                           identity="", held=True, session_note="", period_note=""):
        seen["trader"] = session_note
        return TraderProposal(action=TraderAction.HOLD, reasoning="r")

    def fake_build_all(
        reports, tp, extra_context="", language="ko", identity="", held=True,
        session_note="", period_note="", sizing_context="", research_plan_note="",
    ):
        seen["risk"] = session_note
        return {p: (lambda s, party: _const("take")) for p in ("Aggressive", "Conservative", "Neutral")}

    async def capturing_decide(*args, session_note="", period_note="", **kwargs):
        seen["portfolio_manager"] = session_note
        return PortfolioDecision(rating=PortfolioRating.HOLD, executive_summary="s", investment_thesis="t", confidence=0.7)

    def fake_bull_build(reports, language="ko", identity="", session_note=""):
        seen["bull"] = session_note
        return lambda state, party: _const("bull argument")

    def fake_bear_build(reports, language="ko", identity="", session_note=""):
        seen["bear"] = session_note
        return lambda state, party: _const("bear argument")

    monkeypatch.setattr(pipeline.research_manager, "synthesize", fake_synthesize)
    monkeypatch.setattr(pipeline.trader, "propose", fake_propose)
    monkeypatch.setattr(pipeline.risk_debators, "build_all", fake_build_all)
    monkeypatch.setattr(pipeline.portfolio_manager, "decide", capturing_decide)
    monkeypatch.setattr(pipeline.bull_researcher, "build", fake_bull_build)
    monkeypatch.setattr(pipeline.bear_researcher, "build", fake_bear_build)

    result = await pipeline.run_pipeline("005930.KS", date(2026, 7, 16), as_of=date(2026, 7, 16))

    assert set(seen) == {"research_manager", "trader", "risk", "portfolio_manager", "bull", "bear"}
    assert all(v == "SESSION IN PROGRESS: test note" for v in seen.values())
    assert result.session_in_progress is True
    assert calls["store_pending"]["transcript"]["session_in_progress"] is True


async def test_run_pipeline_threads_the_interim_period_note_to_every_downstream_node(monkeypatch):
    """2026-09-01 audit #5: get_kr_financials' interim-period warning must
    reach RM / Trader / risk / PM (not the Bull/Bear pair)."""
    calls, _ = _install_common_fakes(monkeypatch)

    async def fake_fund(*args, period_note_out=None, **kwargs):
        calls["analysts"].append("Fundamentals Analyst")
        if period_note_out is not None:
            period_note_out.append("KOREAN INTERIM-PERIOD WARNING: test")
        return "fundamentals: solid"

    monkeypatch.setattr(pipeline.fundamentals_analyst, "analyze", fake_fund)
    seen = {}

    async def fake_synthesize(
        reports, debate, language="ko", identity="", held=True, session_note="", period_note="",
        position_context="",
    ):
        seen["research_manager"] = period_note
        return ResearchPlan(recommendation=PortfolioRating.HOLD, rationale="r", strategic_actions="a")

    async def fake_propose(reports, debate, plan, position_context="", language="ko", profile=None,
                           identity="", held=True, session_note="", period_note=""):
        seen["trader"] = period_note
        return TraderProposal(action=TraderAction.HOLD, reasoning="r")

    def fake_build_all(
        reports, tp, extra_context="", language="ko", identity="", held=True,
        session_note="", period_note="", sizing_context="", research_plan_note="",
    ):
        seen["risk"] = period_note
        return {p: (lambda s, party: _const("take")) for p in ("Aggressive", "Conservative", "Neutral")}

    async def capturing_decide(*args, session_note="", period_note="", **kwargs):
        seen["portfolio_manager"] = period_note
        return PortfolioDecision(rating=PortfolioRating.HOLD, executive_summary="s", investment_thesis="t", confidence=0.7)

    monkeypatch.setattr(pipeline.research_manager, "synthesize", fake_synthesize)
    monkeypatch.setattr(pipeline.trader, "propose", fake_propose)
    monkeypatch.setattr(pipeline.risk_debators, "build_all", fake_build_all)
    monkeypatch.setattr(pipeline.portfolio_manager, "decide", capturing_decide)

    await pipeline.run_pipeline("005930.KS", date(2026, 7, 16), as_of=date(2026, 7, 16))

    assert set(seen) == {"research_manager", "trader", "risk", "portfolio_manager"}
    assert all(v == "KOREAN INTERIM-PERIOD WARNING: test" for v in seen.values())


_KIA_KR_BLOCK_FOR_PIPELINE = """
[Income statement] (억원)
  202606 (interim, 6 months — NOT a full year): revenue=625,389억원 gross_profit=1억원 operating_profit=48,337억원 net_income=41,579억원
  202412 (FY, 12 months): revenue=1,074,500억원 gross_profit=1억원 operating_profit=126,671억원 net_income=97,800억원
"""


def _spy_store_cached(monkeypatch):
    """Record whether run_pipeline chose to write the report cache — a direct
    read of the cache-gate decision (the backtest read path is keyed on
    `as_of` and does not always round-trip, so `_cached_reports` after the run
    is not a reliable probe of the gate)."""
    seen = {"stored": False}
    real = pipeline._store_cached_reports

    def spy(*args, **kwargs):
        seen["stored"] = True
        return real(*args, **kwargs)

    monkeypatch.setattr(pipeline, "_store_cached_reports", spy)
    return seen


async def test_run_pipeline_records_period_mismatch_violations(monkeypatch):
    """audit #5 §4 + 2026-09-03 Tier split: a Tier-2 finding (here, one the
    Fundamentals Analyst's own guard reported, with no period frame re-derived
    by the pipeline) is recorded in the decision record (detect-only) but does
    NOT withhold the report cache."""
    calls, _ = _install_common_fakes(monkeypatch)
    stored = _spy_store_cached(monkeypatch)

    async def fake_fund(*args, period_violations_out=None, **kwargs):
        calls["analysts"].append("Fundamentals Analyst")
        if period_violations_out is not None:
            period_violations_out.append("Fundamentals Analyst: period_mismatch: '-62%' matches interim 202606 / FY 202412 operating_profit (-61.8%)")
        return "fundamentals: solid"

    monkeypatch.setattr(pipeline.fundamentals_analyst, "analyze", fake_fund)

    result = await pipeline.run_pipeline("005930.KS", date(2026, 7, 16), as_of=date(2026, 7, 16))

    assert result.period_mismatch_violations
    assert calls["store_pending"]["transcript"]["period_mismatch_violations"] == result.period_mismatch_violations
    assert stored["stored"] is True  # Tier-2 only — the cache is NOT withheld


async def test_run_pipeline_records_kr_financials_recency(monkeypatch):
    """2026-09-07 audit #6: get_kr_financials' recency facts, filled by the
    Fundamentals Analyst on the non-cache path, must land on state and in the
    stored transcript so a batch audit can count stale-financials decisions."""
    calls, _ = _install_common_fakes(monkeypatch)

    async def fake_fund(*args, recency_out=None, **kwargs):
        calls["analysts"].append("Fundamentals Analyst")
        if recency_out is not None:
            recency_out.update({
                "latest_period": "202412", "latest_label": "202412 (FY, 12 months)",
                "months_stale": 21, "interim": False,
            })
        return "fundamentals: solid"

    monkeypatch.setattr(pipeline.fundamentals_analyst, "analyze", fake_fund)

    result = await pipeline.run_pipeline("294630.KS", date(2026, 9, 7), as_of=date(2026, 9, 7))

    assert result.kr_financials_recency["months_stale"] == 21
    assert calls["store_pending"]["transcript"]["kr_financials_recency"]["latest_period"] == "202412"


async def test_transcript_carries_staleness_narrative_violations_clean_case(monkeypatch):
    """2026-09-09, id=1351: staleness_narrative_guard.scan_staleness_narrative's
    findings must reach the stored transcript — present as `[]` here (this
    fixture's fake financials are not stale, so the guard never even attempts
    a scan), not simply absent, so a later SQL
    `full_transcript ? 'staleness_narrative_violations'` existence-check is
    meaningful."""
    calls, _ = _install_common_fakes(monkeypatch)

    result = await pipeline.run_pipeline("005930.KS", date(2026, 7, 16), as_of=date(2026, 7, 16))

    assert result.staleness_narrative_violations == []
    assert "staleness_narrative_violations" in calls["store_pending"]["transcript"]
    assert (
        calls["store_pending"]["transcript"]["staleness_narrative_violations"]
        == result.staleness_narrative_violations
    )


async def test_run_pipeline_flags_the_id_1351_shape_end_to_end(monkeypatch):
    """2026-09-09, id=1351 (198440.KS): a Fundamentals Analyst report that
    presents stale (21-month) financials as the company's current condition,
    with no staleness acknowledgment anywhere, must produce exactly one
    staleness_narrative_violations finding that reaches both `result` and
    the stored transcript."""
    calls, _ = _install_common_fakes(monkeypatch)

    stale_report = (
        "As of September 9, 2026, Kangdong C&L presents a complex fundamental "
        "picture characterized by significant operational deterioration "
        "compared to its peak years."
    )

    async def fake_fund(*args, recency_out=None, **kwargs):
        calls["analysts"].append("Fundamentals Analyst")
        if recency_out is not None:
            recency_out.update({
                "latest_period": "202412", "latest_label": "202412 (FY, 12 months)",
                "months_stale": 21, "interim": False,
            })
        return stale_report

    monkeypatch.setattr(pipeline.fundamentals_analyst, "analyze", fake_fund)

    result = await pipeline.run_pipeline("198440.KS", date(2026, 9, 9), as_of=date(2026, 9, 9))

    assert len(result.staleness_narrative_violations) == 1
    assert (
        calls["store_pending"]["transcript"]["staleness_narrative_violations"]
        == result.staleness_narrative_violations
    )


async def test_transcript_carries_absence_reframing_violations_clean_case(monkeypatch):
    """2026-09-09 P1 audit item ⑤ (ids 1349/1350/1351/1352/1358/1359):
    absence_reframing_guard.scan_absence_reframing's findings must reach the
    stored transcript — present as `[]` here (this fixture's fake reports
    carry no absence-reframing language), not simply absent, so a later SQL
    `full_transcript ? 'absence_reframing_violations'` existence-check is
    meaningful."""
    calls, _ = _install_common_fakes(monkeypatch)

    result = await pipeline.run_pipeline("005930.KS", date(2026, 7, 16), as_of=date(2026, 7, 16))

    assert result.absence_reframing_violations == []
    assert "absence_reframing_violations" in calls["store_pending"]["transcript"]
    assert (
        calls["store_pending"]["transcript"]["absence_reframing_violations"]
        == result.absence_reframing_violations
    )


async def test_run_pipeline_flags_the_id_1349_shape_end_to_end(monkeypatch):
    """2026-09-09, id=1349 (950250.KS): a report reframing the absence of
    insider selling as management confidence/reduced fraud risk must produce
    exactly one absence_reframing_violations finding that reaches both
    `result` and the stored transcript."""
    calls, _ = _install_common_fakes(monkeypatch)

    spin_report = (
        "the absence of insider selling suggests management does not believe the current price "
        "represents fair value, reducing the likelihood of a fraudulent pump-and-dump exit."
    )

    async def fake_fund(*args, **kwargs):
        calls["analysts"].append("Fundamentals Analyst")
        return spin_report

    monkeypatch.setattr(pipeline.fundamentals_analyst, "analyze", fake_fund)

    result = await pipeline.run_pipeline("950250.KS", date(2026, 9, 9), as_of=date(2026, 9, 9))

    assert len(result.absence_reframing_violations) == 1
    assert (
        calls["store_pending"]["transcript"]["absence_reframing_violations"]
        == result.absence_reframing_violations
    )


async def test_transcript_carries_derived_value_mismatch_violations_clean_case(monkeypatch):
    """2026-09-09 P1 audit item ⑤ (ids 1357/1353):
    derived_value_mismatch_guard.scan_derived_value_mismatches's findings
    must reach the stored transcript — present as `[]` here (this fixture's
    fake reports carry no derived-value arithmetic), not simply absent, so a
    later SQL `full_transcript ? 'derived_value_mismatch_violations'`
    existence-check is meaningful."""
    calls, _ = _install_common_fakes(monkeypatch)

    result = await pipeline.run_pipeline("005930.KS", date(2026, 7, 16), as_of=date(2026, 7, 16))

    assert result.derived_value_mismatch_violations == []
    assert "derived_value_mismatch_violations" in calls["store_pending"]["transcript"]
    assert (
        calls["store_pending"]["transcript"]["derived_value_mismatch_violations"]
        == result.derived_value_mismatch_violations
    )


async def test_run_pipeline_flags_the_id_1357_shape_end_to_end(monkeypatch):
    """2026-09-09, id=1357 (032820.KS): a Fundamentals Analyst report that
    states a PER inconsistent with its own cited EPS and price must produce
    exactly one derived_value_mismatch_violations finding that reaches both
    `result` and the stored transcript."""
    calls, _ = _install_common_fakes(monkeypatch)

    mismatch_report = (
        "**Price/Earnings Ratio (PER):** **~2,728x**. This extremely high multiple reflects "
        "the tiny EPS (₩62) relative to the stock price (~₩13,640)."
    )

    async def fake_fund(*args, **kwargs):
        calls["analysts"].append("Fundamentals Analyst")
        return mismatch_report

    monkeypatch.setattr(pipeline.fundamentals_analyst, "analyze", fake_fund)

    result = await pipeline.run_pipeline("032820.KS", date(2026, 9, 9), as_of=date(2026, 9, 9))

    assert len(result.derived_value_mismatch_violations) == 1
    assert (
        calls["store_pending"]["transcript"]["derived_value_mismatch_violations"]
        == result.derived_value_mismatch_violations
    )


async def test_run_pipeline_threads_news_headline_soft_ground_truth_to_the_guards(monkeypatch):
    """2026-09-07 audit #3: the raw KIS headline block captured from the News
    Analyst becomes unit_scale_guard's soft ground truth for every downstream
    node, and a 억원->billion carry-through is flagged (detect-only) and rolled
    up on state.news_amount_scans."""
    calls, _ = _install_common_fakes(monkeypatch)

    async def fake_news(*args, news_block_out=None, **kwargs):
        calls["analysts"].append("News + Macro Analyst")
        if news_block_out is not None:
            news_block_out.append("[20260904] 서남, 공급 계약 6.6억원 (매출액대비 19.56 %)")
        # the News report itself carries the 10x carry-through
        return "The desk notes a ₩6.6 billion supply contract."

    monkeypatch.setattr(pipeline.news_macro_analyst, "analyze", fake_news)

    result = await pipeline.run_pipeline("294630.KS", date(2026, 9, 7), as_of=date(2026, 9, 7))

    scans = result.news_amount_scans
    assert scans["headline_amounts"] == 1
    assert scans["total_flags"] >= 1
    assert "News + Macro Analyst" in scans["flagged_nodes"]
    assert calls["store_pending"]["transcript"]["news_amount_scans"]["total_flags"] >= 1
    # detect-only: the wrong span is still in the stored report
    assert "₩6.6 billion" in result.reports["News + Macro Analyst"]
    # Fix 3 regression pin (2026-09-09): the soft GT must reach DOWNSTREAM node
    # scans (debates / research_plan / trader / PM), not just the News node —
    # a regression dropping soft_eok_block= from _guard_fields /
    # _numeric_turn_guard would leave every non-analyst scan at
    # soft_ground_truth_values 0 and still pass every assertion above.
    node_scans = calls["store_pending"]["transcript"]["unit_scale_scan"]
    downstream = {
        n: s for n, s in node_scans.items()
        if n not in ("News + Macro Analyst", "Market Analyst", "Sentiment Analyst", "Fundamentals Analyst")
    }
    assert downstream, "no downstream node scans recorded"
    assert any(s["soft_ground_truth_values"] > 0 for s in downstream.values())


async def test_run_pipeline_tier1_period_mismatch_blocks_cache(monkeypatch):
    """2026-09-03: a Tier-1 (strong) period-mismatch finding — the pipeline
    re-scans the real Fundamentals report against the supplied 억원 block, and
    a period-framed Fundamentals-Analyst collapse withholds the cycle from the
    report cache (same treatment identity/unit-scale violations already get)."""
    calls, _ = _install_common_fakes(monkeypatch)
    stored = _spy_store_cached(monkeypatch)

    async def fake_fund(*args, kr_ground_truth_out=None, **kwargs):
        calls["analysts"].append("Fundamentals Analyst")
        if kr_ground_truth_out is not None:
            kr_ground_truth_out.append(_KIA_KR_BLOCK_FOR_PIPELINE)
        return "Operating profit shows a sharp decline of roughly -62% versus its 2024 peak."

    monkeypatch.setattr(pipeline.fundamentals_analyst, "analyze", fake_fund)

    result = await pipeline.run_pipeline("005930.KS", date(2026, 7, 16), as_of=date(2026, 7, 16))

    assert result.period_mismatch_violations
    assert result.period_mismatch_scans.get("Fundamentals Analyst", {}).get("strong") == 1
    assert stored["stored"] is False  # Tier-1 — cache withheld


async def test_run_pipeline_no_interim_threads_empty_period_note(monkeypatch):
    calls, _ = _install_common_fakes(monkeypatch)  # fake fundamentals never fills period_note_out
    seen = {}

    async def fake_synthesize(
        reports, debate, language="ko", identity="", held=True, session_note="", period_note="",
        position_context="",
    ):
        seen["rm"] = period_note
        return ResearchPlan(recommendation=PortfolioRating.HOLD, rationale="r", strategic_actions="a")

    async def capturing_decide(*args, session_note="", period_note="", **kwargs):
        seen["pm"] = period_note
        return PortfolioDecision(rating=PortfolioRating.HOLD, executive_summary="s", investment_thesis="t", confidence=0.7)

    monkeypatch.setattr(pipeline.research_manager, "synthesize", fake_synthesize)
    monkeypatch.setattr(pipeline.portfolio_manager, "decide", capturing_decide)

    await pipeline.run_pipeline("005930.KS", date(2026, 7, 16), as_of=date(2026, 7, 16))
    assert seen == {"rm": "", "pm": ""}


async def test_run_pipeline_fetches_position_context_after_debate_before_research_manager(monkeypatch):
    """Call-order lock (2026-08-28): _position_context must run AFTER the
    Bull/Bear debate (which doesn't need it and shouldn't pay its DB/KIS
    latency) but BEFORE the Research Manager (which does need `held` now —
    see pipeline.py's own comment at this call site for why here and not
    earlier/later). Guards against a future refactor silently moving it back."""
    calls, _ = _install_common_fakes(monkeypatch)
    order: list[str] = []

    real_position_context = pipeline._position_context

    async def tracking_position_context(ticker, as_of):
        order.append("position_context")
        return await real_position_context(ticker, as_of)

    monkeypatch.setattr(pipeline, "_position_context", tracking_position_context)

    real_run_debate = pipeline.run_debate

    async def tracking_run_debate(*args, **kwargs):
        order.append("run_debate")
        return await real_run_debate(*args, **kwargs)

    monkeypatch.setattr(pipeline, "run_debate", tracking_run_debate)

    plan = ResearchPlan(recommendation=PortfolioRating.BUY, rationale="r", strategic_actions="a")

    async def fake_synthesize(
        reports, debate, language="ko", identity="", held=True, session_note="", period_note="",
        position_context="",
    ):
        order.append("research_manager")
        return plan

    monkeypatch.setattr(pipeline.research_manager, "synthesize", fake_synthesize)

    await pipeline.run_pipeline("005930.KS", date(2026, 7, 16), as_of=date(2026, 7, 16))

    # Both debates (Bull/Bear, then Risk) call run_debate — only the FIRST
    # matters here, so index rather than assert exact full order.
    first_debate = order.index("run_debate")
    position_idx = order.index("position_context")
    rm_idx = order.index("research_manager")
    assert first_debate < position_idx < rm_idx


async def test_run_pipeline_records_unit_scale_violations_and_skips_caching(monkeypatch):
    """Wiring test for app.agents.unit_scale_guard's out-param (2026-08-28):
    fundamentals_analyst.analyze extends `unit_violations_out` in place when
    it corrects a 억원-to-billion mislabel — verify pipeline.py threads that
    into state.unit_scale_violations, the stored transcript, AND withholds
    the cycle from the report cache, same as identity_violations/
    degraded_analysts already do. Fakes fundamentals_analyst.analyze itself
    rather than the real unit_scale_guard end-to-end (that's covered in
    tests/agents/test_fundamentals_analyst.py) — this test is only about the
    pipeline-level plumbing."""
    calls, _ = _install_common_fakes(monkeypatch)

    async def fake_analyze_with_violation(*args, unit_violations_out=None, **kwargs):
        calls["analysts"].append("Fundamentals Analyst")
        if unit_violations_out is not None:
            unit_violations_out.append("billion_10x: '951,542 billion KRW' -> '95,154.2 billion KRW'")
        return "Revenue reached 95,154.2 billion KRW [unit corrected from \"951,542 billion KRW\"]."

    monkeypatch.setattr(pipeline.fundamentals_analyst, "analyze", fake_analyze_with_violation)

    result = await pipeline.run_pipeline("005930.KS", date(2026, 7, 16), as_of=date(2026, 7, 16))

    assert len(result.unit_scale_violations) == 1
    assert "billion_10x" in result.unit_scale_violations[0]
    assert calls["store_pending"]["transcript"]["unit_scale_violations"] == result.unit_scale_violations
    # A violating run must not be cached, same reasoning as an identity
    # violation or a degraded analyst.
    assert pipeline._cached_reports("005930.KS", "2026-07-16", date(2026, 7, 16), "ko") is None


async def test_run_pipeline_cache_hit_records_no_unit_scale_violations(monkeypatch):
    """A cached run was necessarily a CLEAN run — same reasoning as
    identity_violations/degraded_analysts on a cache hit."""
    calls, _ = _install_common_fakes(monkeypatch)
    pipeline._store_cached_reports(
        "005930.KS", "2026-07-16", date(2026, 7, 16), "ko",
        {"Market Analyst": "cached", "Fundamentals Analyst": "cached",
         "News + Macro Analyst": "cached", "Sentiment Analyst": "cached"},
    )

    result = await pipeline.run_pipeline("005930.KS", date(2026, 7, 16), as_of=date(2026, 7, 16))

    assert result.unit_scale_violations == []


async def test_risk_debate_turn_guard_sees_program_trade_ground_truth_research_debate_does_not(monkeypatch):
    """2026-09-11 (decision id=1386, 006360.KS): a risk-debate turn wrote a
    billion_10x-shaped miscorrection of the program-trading 억원 figure
    (`kr_risk_extra`, fetched from get_kr_risk_signals right before the risk
    debate) — "correcting" a correct +15.65 billion KRW figure to the wrong
    +156.5 billion KRW, calling the correct one a fabrication. Before this
    fix, `_numeric_turn_guard`'s `kr_block` only ever carried the Fundamentals
    Analyst's 억원 block (`kr_gt`), which never contains program-trading
    figures, so this couldn't be auto-corrected turn-by-turn.

    `_numeric_turn_guard` is one closure shared by both the research
    (Bull/Bear) and risk (Aggressive/Conservative/Neutral) debates. The fix
    relies on Python's closure-by-reference semantics: `kr_risk_extra` is
    declared empty before the closure is defined, and only assigned its real
    value (from get_kr_risk_signals) right before the risk debate runs — AFTER
    the research debate has already completed. So the SAME closure sees ""
    for the research debate and the real value for the risk debate, purely
    from call-order, with no per-stage branching in the guard itself.

    This test invokes each stage's `turn_guard` from inside a fake
    `run_debate`, at the exact point in pipeline execution order the real
    `run_debate` would call it — capturing/storing results there (not after
    `run_pipeline` returns) is essential, since by the time `run_pipeline`
    returns, `kr_risk_extra` has long since been assigned its real value and
    a guard captured-then-called-later would no longer exhibit the
    closure-timing behavior under test."""
    calls, _ = _install_common_fakes(monkeypatch)

    async def fake_get_kr_risk_signals(ticker, as_of=None):
        return (
            "Korean risk signals for 006360.KS (source: KIS)\n"
            "[Program trading] latest net-buy: +406,376 shares (+156.5억원)"
        )

    monkeypatch.setattr(pipeline, "get_kr_risk_signals", fake_get_kr_risk_signals)

    guarded_results: dict[str, str] = {}
    # billion_10x-shaped: mantissa 156.5 copied straight from the 156.5억원
    # ground truth instead of correctly divided by 10 -> 15.65 billion KRW.
    miscorrected_turn = (
        "Correcting the earlier claim: the program trade net buy was "
        "actually 156.5 billion KRW, not 15.65 billion KRW as previously "
        "stated (that figure was fabricated)."
    )

    async def fake_run_debate(
        *, state, speakers, max_rounds, on_round=None, language="ko", check_convergence=True,
        parallel_first_round=False, turn_guard=None,
    ):
        stage = "research_debate" if "Bull" in speakers else "risk_debate"
        assert turn_guard is not None
        guarded_results[stage] = turn_guard(miscorrected_turn, "Neutral", 1)
        state.rounds_completed = 1
        return state

    monkeypatch.setattr(pipeline, "run_debate", fake_run_debate)

    await pipeline.run_pipeline("006360.KS", date(2026, 9, 11), as_of=date(2026, 9, 11))

    # Risk debate: kr_risk_extra is populated by the time its turn_guard
    # runs, so the 156.5 billion KRW mantissa-copy is caught and corrected
    # to the right magnitude (15.7 billion KRW at enforce_unit_scale's own
    # 1-decimal rounding of 156.5/10 = 15.65) — the original wrong span is
    # kept visible only inside the "[unit corrected from ...]" annotation,
    # not as a bare, uncorrected claim.
    assert "15.7 billion KRW" in guarded_results["risk_debate"]
    assert '[unit corrected from "156.5 billion KRW"]' in guarded_results["risk_debate"]
    assert "actually 156.5 billion KRW, not" not in guarded_results["risk_debate"]

    # Research debate: its turn_guard call happens BEFORE kr_risk_extra is
    # fetched (deliberately — see pipeline.py's comment above that fetch,
    # id=1170/498400.KS), so the same closure has no program-trading ground
    # truth yet and the miscorrection passes through untouched (no
    # "[unit corrected ...]" annotation, no rewritten span).
    assert guarded_results["research_debate"].strip() == miscorrected_turn
    assert "[unit corrected" not in guarded_results["research_debate"]


async def test_run_pipeline_records_adjustment_note_when_rebalance_shrinks_trade(monkeypatch):
    calls, holder = _install_common_fakes(monkeypatch)
    holder["result"] = RebalanceResult(
        side="buy", shares=2, shares_after=2, stop_loss=140.0, action_label="Buy",
        adjustment_note="risk cap reduced the add from 5 to 2 shares",
    )

    result = await pipeline.run_pipeline("AAPL", date(2026, 7, 16), as_of=date(2026, 7, 16))

    assert calls["append_note"] == {"decision_id": 42, "note": "risk cap reduced the add from 5 to 2 shares"}
    assert result.rebalance_adjustment_note == "risk cap reduced the add from 5 to 2 shares"


async def test_run_pipeline_persists_a_rating_action_gap_note(monkeypatch):
    """2026-09-01 audit #10: a directional rating that sized to no position
    change now returns an adjustment_note, and the pipeline must persist it
    via append_execution_note + set_action_label('Hold')."""
    calls, holder = _install_common_fakes(monkeypatch)
    gap_note = (
        "No position change despite a directional rating: the Buy rating's target allocation "
        "(58.80% of 200,000 equity = 117,600) does not cover one share at 1,431,500.00, so "
        "floor(117,600 / 1,431,500.00) = 0 shares — nothing reduced it. Nothing was bought; "
        "the position stays flat at 0 shares."
    )
    holder["result"] = RebalanceResult(
        side=None, shares=0, shares_after=0, stop_loss=None, action_label="Hold",
        adjustment_note=gap_note,
    )
    seen = {}
    async def fake_set_action_label(decision_id, action_label):
        seen["set_action_label"] = (decision_id, action_label)
    monkeypatch.setattr(pipeline, "set_action_label", fake_set_action_label)

    result = await pipeline.run_pipeline("005930.KS", date(2026, 7, 16), as_of=date(2026, 7, 16))

    assert calls["append_note"] == {"decision_id": 42, "note": gap_note}
    assert seen["set_action_label"] == (42, "Hold")
    assert result.rebalance_adjustment_note == gap_note


# ---------------------------------------------------------------------------
# trade_allowlist auto-add on a BUY decision (live only)
# ---------------------------------------------------------------------------


async def test_run_pipeline_auto_adds_to_trade_allowlist_on_buy_when_not_listed(monkeypatch):
    from app.config import settings
    from app.web import state as web_state

    _install_common_fakes(monkeypatch)  # fake_decide returns a BUY rating
    monkeypatch.setattr(settings, "trade_allowlist_domestic", [])
    monkeypatch.setattr(settings, "trade_allowlist_overseas", [])  # ticker not on the trade allowlist

    added = []

    async def fake_add(ticker):
        added.append(ticker)
        return "added", None

    monkeypatch.setattr(web_state, "add_ticker_to_trade_allowlist", fake_add)

    await pipeline.run_pipeline("AAPL", date(2026, 7, 16))  # live (as_of=None)

    assert added == ["AAPL"]


async def test_run_pipeline_marks_a_fresh_auto_add_for_later_reconciliation(monkeypatch):
    # 2026-07-29: a BUY decision's auto-add is speculative at this point
    # (no approval/fill yet in SEMI_AUTO) — must be marked so a later
    # rejected/expired/failed order can be pruned back off the allowlist
    # instead of sitting on it forever.
    from app.config import settings
    from app.web import state as web_state

    _install_common_fakes(monkeypatch)
    monkeypatch.setattr(settings, "trade_allowlist_domestic", [])
    monkeypatch.setattr(settings, "trade_allowlist_overseas", [])

    async def fake_add(ticker):
        return "added", None

    marked = []

    async def fake_mark(ticker):
        marked.append(ticker)

    monkeypatch.setattr(web_state, "add_ticker_to_trade_allowlist", fake_add)
    monkeypatch.setattr(web_state, "mark_trade_allowlist_auto_added", fake_mark)

    await pipeline.run_pipeline("AAPL", date(2026, 7, 16))

    assert marked == ["AAPL"]


async def test_run_pipeline_does_not_mark_when_add_reports_already_existing(monkeypatch):
    # A race where the ticker got added by another caller between the
    # `not in trade_allowlist` check and this call — add_ticker_to_trade_
    # allowlist reports "exists", so this decision isn't the one that put
    # it there and shouldn't claim/mark it.
    from app.config import settings
    from app.web import state as web_state

    _install_common_fakes(monkeypatch)
    monkeypatch.setattr(settings, "trade_allowlist_domestic", [])
    monkeypatch.setattr(settings, "trade_allowlist_overseas", [])

    async def fake_add(ticker):
        return "exists", None

    async def fail_if_called(ticker):
        raise AssertionError("must not mark when add_ticker_to_trade_allowlist reported 'exists'")

    monkeypatch.setattr(web_state, "add_ticker_to_trade_allowlist", fake_add)
    monkeypatch.setattr(web_state, "mark_trade_allowlist_auto_added", fail_if_called)

    await pipeline.run_pipeline("AAPL", date(2026, 7, 16))


async def test_run_pipeline_does_not_auto_add_when_already_on_trade_allowlist(monkeypatch):
    from app.config import settings
    from app.web import state as web_state

    _install_common_fakes(monkeypatch)
    monkeypatch.setattr(settings, "trade_allowlist_domestic", [])
    monkeypatch.setattr(settings, "trade_allowlist_overseas", ["AAPL"])

    async def fail_if_called(ticker):
        raise AssertionError("should not auto-add a ticker already on the trade allowlist")

    monkeypatch.setattr(web_state, "add_ticker_to_trade_allowlist", fail_if_called)

    await pipeline.run_pipeline("AAPL", date(2026, 7, 16))


async def test_run_pipeline_does_not_auto_add_on_hold_decision(monkeypatch):
    from app.config import settings
    from app.web import state as web_state

    calls, _ = _install_common_fakes(monkeypatch)
    monkeypatch.setattr(settings, "trade_allowlist_domestic", [])
    monkeypatch.setattr(settings, "trade_allowlist_overseas", [])
    hold_decision = PortfolioDecision(rating=PortfolioRating.HOLD, executive_summary="s", investment_thesis="t")

    async def fake_decide_hold(*args, **kwargs):
        return hold_decision

    monkeypatch.setattr(pipeline.portfolio_manager, "decide", fake_decide_hold)

    async def fail_if_called(ticker):
        raise AssertionError("should not auto-add on a Hold decision")

    monkeypatch.setattr(web_state, "add_ticker_to_trade_allowlist", fail_if_called)

    await pipeline.run_pipeline("AAPL", date(2026, 7, 16))


async def test_run_pipeline_records_an_alert_when_buy_is_blocked_from_the_trade_allowlist(monkeypatch):
    # 2026-09-03: a BUY blocked here (leveraged/unsupported-product/cash-
    # bond-fund/derivative-income-fund/unresolved-name) used to be visible
    # only via a single logger.warning line — now it also lands in the
    # alerts table so it's countable instead of re-derived from logs.
    from app.config import settings
    from app.live.alerts import Alert
    from app.web import state as web_state

    _install_common_fakes(monkeypatch)
    monkeypatch.setattr(settings, "trade_allowlist_domestic", [])
    monkeypatch.setattr(settings, "trade_allowlist_overseas", [])

    async def fake_add(ticker):
        raise web_state.UnsupportedProductTypeError(f"{ticker} is a MUTUALFUND (not a plain stock or ETF)")

    recorded = []

    async def fake_record_alert(alert):
        recorded.append(alert)

    monkeypatch.setattr(web_state, "add_ticker_to_trade_allowlist", fake_add)
    monkeypatch.setattr("app.live.alerts.record_alert", fake_record_alert)

    await pipeline.run_pipeline("AAPL", date(2026, 7, 16))

    assert len(recorded) == 1
    alert = recorded[0]
    assert isinstance(alert, Alert)
    assert alert.ticker == "AAPL"
    assert alert.kind == "trade_allowlist_blocked"
    assert "UnsupportedProductTypeError" in alert.message


async def test_run_pipeline_does_not_auto_add_during_backtest(monkeypatch):
    from app.config import settings
    from app.web import state as web_state

    _install_common_fakes(monkeypatch)
    monkeypatch.setattr(settings, "trade_allowlist_domestic", [])
    monkeypatch.setattr(settings, "trade_allowlist_overseas", [])  # would trigger auto-add if this were live

    async def fail_if_called(ticker):
        raise AssertionError("backtest must never touch settings.trade_allowlist")

    monkeypatch.setattr(web_state, "add_ticker_to_trade_allowlist", fail_if_called)

    await pipeline.run_pipeline("AAPL", date(2026, 7, 16), as_of=date(2026, 7, 16))


async def test_risk_debate_runs_the_full_schedule(monkeypatch):
    """2026-08-25: the risk debate's convergence check used to end the
    3-party debate after round 1 57.5% of the time — an artifact of
    comparing round 1 against an empty prior-rounds section, not real early
    agreement (see app.orchestrator.debate.run_debate's `2 <= round_num`
    guard). Restores the REAL run_debate (the common fixture fakes it) to
    prove the risk debate actually runs both rounds end to end and never
    calls the convergence LLM."""
    _install_common_fakes(monkeypatch)
    monkeypatch.setattr(pipeline, "run_debate", debate_mod.run_debate)

    def fail_if_called(*args, **kwargs):
        raise AssertionError("risk debate must never call the convergence LLM (check_convergence=False)")

    monkeypatch.setattr(
        debate_mod, "llm_client", type("X", (), {"structured_complete": staticmethod(fail_if_called)})()
    )

    speaker_calls: list[tuple[str, int]] = []

    def _tracking_speaker(name: str):
        async def speak(state, party):
            speaker_calls.append((party, state.rounds_completed + 1))
            return f"{name} take"
        return speak

    monkeypatch.setattr(
        pipeline.risk_debators, "build_all",
        lambda reports, trader_proposal, extra_context="", language="ko", identity="", held=True, session_note="", period_note="", sizing_context="", research_plan_note="": {
            "Aggressive": _tracking_speaker("Aggressive"),
            "Conservative": _tracking_speaker("Conservative"),
            "Neutral": _tracking_speaker("Neutral"),
        },
    )

    result = await pipeline.run_pipeline("AAPL", date(2026, 7, 16), as_of=date(2026, 7, 16))

    assert result.risk_debate.rounds_completed == 2
    assert result.risk_debate.converged is False
    assert sorted(speaker_calls) == sorted(
        [("Aggressive", 1), ("Conservative", 1), ("Neutral", 1),
         ("Aggressive", 2), ("Conservative", 2), ("Neutral", 2)]
    )


# ---------------------------------------------------------------------------
# Shared PM-narrative-contradiction notice (2026-09-11) —
# PipelineState.pm_narrative_contradiction_notice_applied /
# app.orchestrator.pipeline._PM_CONTRADICTION_NOTICE.
# ---------------------------------------------------------------------------


async def test_pm_contradiction_notice_appended_when_pm_own_text_is_flagged(monkeypatch):
    """A PM-attributed finding in session_close_violations (the PM's own
    investment_thesis narrating today's still-open session as a completed
    candle) must be surfaced back into investment_thesis, not just logged
    into the transcript and left invisible to a human reading the decision."""
    calls, _ = _install_common_fakes(monkeypatch)
    monkeypatch.setattr(pipeline, "intraday_narration_note", lambda t, d: "SESSION IN PROGRESS: test note")

    flagged_thesis = (
        "thesis text. The stock closed near the session low, a textbook "
        "distribution pattern."
    )

    async def capturing_decide(*args, **kwargs):
        return PortfolioDecision(
            rating=PortfolioRating.HOLD, executive_summary="s",
            investment_thesis=flagged_thesis, confidence=0.9,
        )

    monkeypatch.setattr(pipeline.portfolio_manager, "decide", capturing_decide)

    result = await pipeline.run_pipeline("005930.KS", date(2026, 7, 16), as_of=date(2026, 7, 16))

    assert result.session_close_violations, "expected the PM's own text to trip session_close_violations"
    assert any(v.startswith("portfolio_manager:") for v in result.session_close_violations)
    assert result.pm_narrative_contradiction_notice_applied is True
    notice = pipeline._PM_CONTRADICTION_NOTICE["en"]
    assert result.final_decision.investment_thesis.endswith(notice)
    assert result.final_decision.investment_thesis.startswith(flagged_thesis)
    assert (
        calls["store_pending"]["transcript"]["pm_narrative_contradiction_notice_applied"] is True
    )


async def test_pm_contradiction_notice_not_appended_on_a_clean_cycle(monkeypatch):
    """The common case: nothing in the PM's own text was flagged by any sweep
    guard this cycle, so investment_thesis is left byte-identical."""
    calls, _ = _install_common_fakes(monkeypatch)

    result = await pipeline.run_pipeline("005930.KS", date(2026, 7, 16), as_of=date(2026, 7, 16))

    assert result.pm_narrative_contradiction_notice_applied is False
    assert result.final_decision.investment_thesis == "thesis text"
    assert (
        calls["store_pending"]["transcript"]["pm_narrative_contradiction_notice_applied"] is False
    )


async def test_pm_contradiction_notice_does_not_collide_with_hysteresis_gate_prefix(monkeypatch):
    """When the hysteresis gate ALSO fires this cycle (a flat Buy with no
    stated confidence, downgraded to Hold), its own bracketed prefix on
    executive_summary must be untouched by this notice — the notice only
    ever appends to investment_thesis, never executive_summary — while
    investment_thesis still gets both the gate's own note AND this
    contradiction notice, proving the two never collide."""
    calls, _ = _install_common_fakes(monkeypatch)
    monkeypatch.setattr(pipeline, "intraday_narration_note", lambda t, d: "SESSION IN PROGRESS: test note")

    flagged_thesis = (
        "thesis text. The stock closed near the session low, a textbook "
        "distribution pattern."
    )

    async def capturing_decide(*args, **kwargs):
        return FlatPortfolioDecision(
            rating=FlatPortfolioRating.BUY, executive_summary="s",
            investment_thesis=flagged_thesis, confidence=None,
        )

    monkeypatch.setattr(pipeline.portfolio_manager, "decide", capturing_decide)

    result = await pipeline.run_pipeline("AAPL", date(2026, 7, 16), as_of=date(2026, 7, 16))

    assert result.final_decision.rating == FlatPortfolioRating.HOLD
    assert result.final_decision.executive_summary.startswith("[Hysteresis gate:")
    assert result.final_decision.executive_summary.endswith("s")
    assert "Hysteresis gate" in result.final_decision.investment_thesis
    assert result.pm_narrative_contradiction_notice_applied is True
    assert result.final_decision.investment_thesis.endswith(pipeline._PM_CONTRADICTION_NOTICE["en"])
