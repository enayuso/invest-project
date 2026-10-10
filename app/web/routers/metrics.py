"""LLM call cost/latency observability (agent-architecture-roadmap stage 0)
— read-only view over app.llm.metrics's in-process ring buffer, so the
agent-slimming decision (stage 1) has real per-agent token/latency data
instead of a hand count of call sites."""

from __future__ import annotations

from fastapi import APIRouter

from app.llm import metrics as llm_metrics

router = APIRouter()


@router.get("/api/metrics/llm")
async def llm_metrics_endpoint(recent_limit: int = 20) -> dict:
    return {
        "summary": [
            {
                "label": s.label,
                "call_count": s.call_count,
                "total_tokens": s.total_tokens,
                "mean_latency_seconds": s.mean_latency_seconds,
            }
            for s in llm_metrics.summary()
        ],
        "recent": [
            {
                "timestamp": m.timestamp.isoformat(),
                "label": m.label,
                "model": m.model,
                "reasoning": m.reasoning,
                "structured": m.structured,
                "latency_seconds": m.latency_seconds,
                "prompt_tokens": m.prompt_tokens,
                "completion_tokens": m.completion_tokens,
                "total_tokens": m.total_tokens,
                "finish_reason": m.finish_reason,
            }
            for m in llm_metrics.recent(recent_limit)
        ],
    }
