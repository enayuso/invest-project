"""Shared synthetic helpers for the tests/experiments/test_replay_*.py files.

Plain module (not a conftest): each test file installs its own autouse DB block
through `install_db_block` so the safety net is visible in every file (spec G).
No DB, no LLM, no network anywhere in here.
"""

from __future__ import annotations

import asyncio
import json
from datetime import date, datetime, timezone
from typing import Any

UTC = timezone.utc


def install_db_block(monkeypatch) -> None:
    """Make every accidental DB access raise: asyncpg.connect/create_pool and the
    app pool accessor. tests/conftest.py does not block the DB by itself."""
    import asyncpg

    def _boom(*args, **kwargs):
        raise AssertionError("DB access attempted in a replay test")

    monkeypatch.setattr(asyncpg, "connect", _boom)
    monkeypatch.setattr(asyncpg, "create_pool", _boom)
    import app.memory.db as memory_db

    monkeypatch.setattr(memory_db, "get_pool", _boom)


REPORTS_ORDER = ["Market Analyst", "Fundamentals Analyst", "News + Macro Analyst", "Sentiment Analyst"]


def research_turns() -> list[dict[str, Any]]:
    return [
        {"speaker": "Bull", "content": "bull one", "round": 1},
        {"speaker": "Bear", "content": "bear one", "round": 1},
        {"speaker": "Bull", "content": "bull two", "round": 2},
        {"speaker": "Bear", "content": "bear two", "round": 2},
    ]


def risk_turns(neutral_final: str = "Hold") -> list[dict[str, Any]]:
    turns = []
    for rnd in (1, 2):
        for speaker in ("Aggressive", "Conservative", "Neutral"):
            content = f"{speaker} argument r{rnd}"
            if speaker == "Neutral":
                content += f"\n\nFinal Decision: {neutral_final}" if rnd == 2 else "\n\nFinal Decision: Hold"
            turns.append({"speaker": speaker, "content": content, "round": rnd})
    return turns


def make_transcript(**overrides: Any) -> dict[str, Any]:
    transcript: dict[str, Any] = {
        # deliberately NOT in ANALYST_ORDER: the contract must reorder them
        "reports": {
            "Sentiment Analyst": "sentiment report",
            "Market Analyst": "market report",
            "News + Macro Analyst": "news report",
            "Fundamentals Analyst": "fundamentals report",
        },
        "research_debate": research_turns(),
        "risk_debate": risk_turns(),
        "research_plan": "**Recommendation**: Hold\n\n**Rationale**: r\n\n**Strategic Actions**: a",
        "trader_proposal": "**Action**: Hold\n\n**Reasoning**: t",
        "investment_thesis": "thesis text",
        "kr_risk_signals": "K",
        "kr_financials_recency": {"interim": True},
        "session_in_progress": True,
        "consensus_override": {},
        "neutral_reversal": {},
        "degraded_analysts": [],
        "past_context": "PAST",
        "sibling_context": "",
        "desk_consensus_note": "",
        "position_context": "Currently flat (no position)",
        "volatility_context": "VOL",
        "risk_cap_context": "CAP",
        "portfolio_risk_context": "PRC",
        "execution_status_context": "EXEC",
        "company_name": "Samsung Electronics",
    }
    transcript.update(overrides)
    return transcript


def make_row(
    id: int = 1,
    ticker: str = "005930.KS",
    trade_date: date = date(2026, 9, 22),
    created_at: datetime | None = None,
    rating: str = "Hold",
    trader_action: str = "Hold",
    confidence: float | None = 0.5,
    held_at_decision: bool | None = False,
    analysts_failed: int | None = 0,
    transcript: dict[str, Any] | None = None,
    llm_model: str = "nvidia/Qwen3.6-35B-A3B-NVFP4",
    git_commit: str = "abc1234",
) -> dict[str, Any]:
    return {
        "id": id,
        "ticker": ticker,
        "trade_date": trade_date,
        "created_at": created_at or datetime(2026, 9, 22, 1, 0, tzinfo=UTC),
        "rating": rating,
        "trader_action": trader_action,
        "confidence": confidence,
        "held_at_decision": held_at_decision,
        "analysts_failed": analysts_failed,
        "decision_trigger": "auto",
        "account_mode": "real",
        "is_backtest": False,
        "git_commit": git_commit,
        "llm_model": llm_model,
        "user_id": "default",
        "full_transcript": make_transcript() if transcript is None else transcript,
    }


def make_inputs(row: dict[str, Any] | None = None, **build_kwargs: Any):
    """Row -> ReplayInputs via the real contract code (session_note stays "")."""
    from app.experiments.replay import contract

    record = contract.record_from_row(row or make_row())
    return contract.build_inputs(record, **build_kwargs)


def _llm_bits():
    from app.domain.models.llm import LLMMessage
    from app.llm import metrics as llm_metrics

    return LLMMessage, llm_metrics


class FakeLLM:
    """Label-dispatching fake for llm_client.complete / structured_complete."""

    def __init__(self, rm="Hold", trader_action="Hold", pm="Hold", confidence=0.7, tokens=None, fail=None):
        self.rm, self.trader_action, self.pm, self.confidence = rm, trader_action, pm, confidence
        self.tokens = tokens or {}  # ticker -> tokens per call
        self.fail = fail  # callable(label, ticker) -> Exception | None
        self.calls: list[dict] = []
        self.pm_extras: dict = {}

    def _ticker(self, messages) -> str:
        blob = json.dumps(messages)
        for ticker in self.tokens:
            if ticker in blob:
                return ticker
        return ""

    def _maybe_fail(self, label, ticker):
        if self.fail:
            exc = self.fail(label, ticker)
            if exc is not None:
                raise exc

    def _record(self, label, structured, ticker):
        total = self.tokens.get(ticker, 10)
        _, llm_metrics = _llm_bits()
        llm_metrics.record(
            label=label, model="fake", reasoning=False, structured=structured, latency_seconds=0.01,
            prompt_tokens=total // 2, completion_tokens=total - total // 2, total_tokens=total,
            fallback_used=False, finish_reason="stop", max_tokens=100,
        )

    async def complete(self, messages, *, tools=None, reasoning=False, temperature=0.0, label="", language="en", max_tokens=None):
        ticker = self._ticker(messages)
        self.calls.append({"label": label, "kind": "complete", "messages": messages, "ticker": ticker})
        await asyncio.sleep(0.005)
        self._maybe_fail(label, ticker)
        self._record(label, False, ticker)
        text = f"{label} argues about the stock."
        if label == "risk_neutral":
            text += "\n\nFinal Decision: Hold"
        return _llm_bits()[0](content=text)

    async def structured_complete(self, messages, schema, *, reasoning=False, temperature=0.0, label="", language="en"):
        ticker = self._ticker(messages)
        self.calls.append({"label": label, "kind": "structured", "messages": messages, "ticker": ticker, "schema": schema.__name__})
        await asyncio.sleep(0.005)
        self._maybe_fail(label, ticker)
        self._record(label, True, ticker)
        name = schema.__name__
        if name.endswith("ResearchPlan"):
            return schema(recommendation=self.rm, rationale="rationale text", strategic_actions="actions text")
        if name.endswith("TraderProposal"):
            return schema(action=self.trader_action, reasoning="trader reasoning")
        if name.endswith("PortfolioDecision"):
            fields = dict(
                rating=self.pm, executive_summary="executive summary", investment_thesis="investment thesis",
                confidence=self.confidence, card_reason_ko="카드 사유", card_reason_en="card reason",
            )
            if self.pm in ("Buy", "Overweight"):
                fields["stop_loss"] = 100.0
            fields.update(self.pm_extras)
            return schema(**fields)
        raise AssertionError(f"unexpected schema {name}")


class _FakeTx:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class FakeReadonlyConn:
    """Stands in for the read-only asyncpg connection: canned rows for the invariant snapshot
    queries, records the SQL so a test can assert nothing but SELECTs was sent."""

    def __init__(self):
        self.sql: list[str] = []
        self.tx_kwargs: list[dict] = []

    def transaction(self, **kwargs):
        self.tx_kwargs.append(kwargs)
        return _FakeTx()

    async def fetchrow(self, sql, *args):
        self.sql.append(sql)
        if "count(*)" in sql:
            return {"max_id": 5, "n": 5, "max_at": datetime(2026, 9, 29, tzinfo=UTC)}
        return {"d": "digest"}

    async def fetch(self, sql, *args):
        self.sql.append(sql)
        return []

    async def close(self):
        return None


def desk_note_for(rm: str = "Hold", trader_action: str = "Hold", kr_risk: str = "K", sizing: str = "VOL\n\nCAP") -> str:
    """The stored desk note a freeze-era row would carry (computed with the real function)."""
    from app.orchestrator import pipeline
    from app.state import PortfolioRating, ResearchPlan, TraderAction, TraderProposal

    plan = ResearchPlan(recommendation=PortfolioRating(rm), rationale="x", strategic_actions="x")
    proposal = TraderProposal(action=TraderAction(trader_action), reasoning="x")
    return pipeline._desk_consensus_note(plan, proposal, kr_risk, sizing)
