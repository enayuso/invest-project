"""metrics router — TestClient, no network (agent-architecture-roadmap stage 0)."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.llm import metrics as llm_metrics
from app.web.routers import metrics as metrics_router


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(metrics_router.router)
    return TestClient(app, base_url="http://127.0.0.1:8094")


@pytest.fixture(autouse=True)
def _clean():
    llm_metrics.clear()
    yield
    llm_metrics.clear()


def test_empty_when_no_calls_recorded(client):
    resp = client.get("/api/metrics/llm")
    assert resp.status_code == 200
    assert resp.json() == {"summary": [], "recent": []}


def test_reports_summary_and_recent_after_calls(client):
    llm_metrics.record(
        label="market_analyst", model="qwen", reasoning=False, structured=False,
        latency_seconds=1.2, prompt_tokens=90, completion_tokens=10, total_tokens=100,
    )
    llm_metrics.record(
        label="trader", model="qwen", reasoning=True, structured=True,
        latency_seconds=0.5, prompt_tokens=45, completion_tokens=5, total_tokens=50,
    )

    resp = client.get("/api/metrics/llm")
    body = resp.json()
    assert len(body["summary"]) == 2
    assert body["summary"][0]["label"] == "market_analyst"  # higher total_tokens sorts first
    assert body["recent"][0]["label"] == "trader"  # most recent first


def test_recent_exposes_finish_reason(client):
    llm_metrics.record(
        label="market_analyst", model="qwen", reasoning=False, structured=False,
        latency_seconds=1.2, prompt_tokens=90, completion_tokens=10, total_tokens=100,
        finish_reason="length",
    )
    resp = client.get("/api/metrics/llm")
    assert resp.json()["recent"][0]["finish_reason"] == "length"


def test_recent_limit_param_is_respected(client):
    for i in range(5):
        llm_metrics.record(
            label=f"call{i}", model="qwen", reasoning=False, structured=False,
            latency_seconds=0.1, prompt_tokens=1, completion_tokens=1, total_tokens=2,
        )
    resp = client.get("/api/metrics/llm?recent_limit=2")
    assert len(resp.json()["recent"]) == 2
