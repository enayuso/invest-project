"""Tests for the 7 static MCP resources (guides.py + server registration)."""

from __future__ import annotations

import ast
import asyncio
import inspect
import json
import socket
from datetime import time
from pathlib import Path

import pytest

from stock_advisor_mcp import coverage, guides, integrity, market_hours, server, yf_gateway
from stock_advisor_mcp.prompts.loader import load_analysis_script

_SNAPSHOT = Path(__file__).parent / "data" / "server_surface_snapshot_2026-10-07.json"
_NEW_SENTENCE = (
    "도구 사용 순서·표기 규칙·지표 해석·가격 연속성 점검의 자세한 설명은 "
    "stock-advisor://guide/... 리소스(서버 소개는 stock-advisor://about)에 있습니다."
)

_SCRIPT_SENTENCE = (
    "종목 종합 분석이나 매수/관망/회피 판단 요청이면 먼저 get_analysis_script를 호출해 "
    "받은 절차를 따르세요(단순 시세·지표 조회에는 호출하지 않음)."
)

_SNAPSHOT_ANCHOR = "장마감 후면 session_status='closed'와 today_close를 반환합니다. "
_SESSION_SENTENCE = "오늘 봉은 한국 시각 17:00 전까지 미확정(in_progress)으로 취급하며(장 마감 직후 포함), 판정 근거는 session_basis에 있습니다. "

_EXPECTED = {
    "stock-advisor://about": "서버 소개",
    "stock-advisor://guide/workflow": "도구 사용 순서와 오류",
    "stock-advisor://guide/data-coverage": "제공하지 않는 데이터",
    "stock-advisor://guide/units": "표기 규칙",
    "stock-advisor://guide/indicators": "기술 지표 해석",
    "stock-advisor://guide/price-integrity": "가격 연속성 점검",
    "stock-advisor://prompts/analysis-script": "분석 대본(열람용)",
}


def _read(uri: str) -> str:
    contents = list(asyncio.run(server.mcp.read_resource(uri)))
    assert len(contents) == 1
    return contents[0].content


def test_list_resources_exact():
    resources = asyncio.run(server.mcp.list_resources())
    assert {str(r.uri): r.title for r in resources} == _EXPECTED
    assert len(resources) == 7
    assert all(r.mime_type == "text/markdown" for r in resources)
    assert asyncio.run(server.mcp.list_resource_templates()) == []


@pytest.mark.parametrize("uri", list(_EXPECTED))
def test_each_resource_reads_non_empty_text(uri):
    text = _read(uri)
    assert isinstance(text, str) and text.strip()


def test_data_coverage_contains_every_item():
    text = _read("stock-advisor://guide/data-coverage")
    for item in coverage.DATA_WE_DO_NOT_HAVE:
        assert item in text


def test_price_integrity_reads_constants(monkeypatch):
    text = _read("stock-advisor://guide/price-integrity")
    assert "30.5%" in text
    assert f"{integrity.LONG_GAP_CALENDAR_DAYS}일 이상" in text
    assert integrity.PRICE_INTEGRITY_NOTICE in text
    for name, window in integrity.INDICATOR_WINDOWS.items():
        assert f"| `{name}` | {window} |" in text

    monkeypatch.setattr(integrity, "LIMIT_BREACH_THRESHOLD", 0.4)
    monkeypatch.setattr(integrity, "LONG_GAP_CALENDAR_DAYS", 21)
    patched = _read("stock-advisor://guide/price-integrity")
    assert "40%" in patched and "30.5%" not in patched
    assert "21일 이상" in patched


def test_analysis_script_resource_equals_loader():
    assert _read("stock-advisor://prompts/analysis-script") == load_analysis_script()


def test_about_and_workflow_mention_key_facts():
    about = _read("stock-advisor://about")
    assert "0.1.0" in about
    for uri in _EXPECTED:
        assert uri in about
    assert "5분" in about
    assert "get_analysis_script" in about
    assert "도구 (6개)" in about
    workflow = _read("stock-advisor://guide/workflow")
    for name in (
        "StaleDataError",
        "NoDataError",
        "TickerNotFoundError",
        "AmbiguousTickerError",
        "KrxMasterError",
        "내부 오류:",
        "get_analysis_script",
    ):
        assert name in workflow


def test_public_surface_unchanged_vs_snapshot():
    snap = json.loads(_SNAPSHOT.read_text(encoding="utf-8"))

    tools = [
        {
            "name": t.name,
            "description": t.description,
            "inputSchema": t.model_dump(mode="json", by_alias=True, exclude_none=True).get("inputSchema"),
        }
        for t in asyncio.run(server.mcp.list_tools())
    ]
    assert len(tools) == 6
    assert sum(t["name"] == "get_analysis_script" for t in tools) == 1
    tools = [t for t in tools if t["name"] != "get_analysis_script"]
    expected_tools = json.loads(json.dumps(snap["tools"]))
    changed = [t for t in expected_tools if t["name"] == "get_price_snapshot"]
    assert len(changed) == 1
    assert changed[0]["description"].count(_SNAPSHOT_ANCHOR) == 1
    changed[0]["description"] = changed[0]["description"].replace(
        _SNAPSHOT_ANCHOR, _SNAPSHOT_ANCHOR + _SESSION_SENTENCE
    )
    assert tools == expected_tools

    prompts = [
        {
            "name": p.name,
            "title": getattr(p, "title", None),
            "description": p.description,
            "arguments": p.model_dump(mode="json", by_alias=True, exclude_none=True).get("arguments"),
        }
        for p in asyncio.run(server.mcp.list_prompts())
    ]
    assert prompts == snap["prompts"]

    assert server.mcp.instructions == (
        snap["instructions_before"] + " " + _NEW_SENTENCE + " " + _SCRIPT_SENTENCE
    )


def test_guides_module_has_no_network_imports():
    tree = ast.parse(inspect.getsource(guides))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            imported.add(base)
            imported.update(f"{base}.{a.name}".strip(".") for a in node.names)
    forbidden = ("yfinance", "yf_gateway", "cache", "httpx", "requests", "market_data", "fundamentals")
    for name in imported:
        assert not any(part in forbidden for part in name.split(".")), name


def test_reading_resources_makes_no_network_calls(monkeypatch):
    def boom(*_a, **_kw):
        raise AssertionError("network access attempted while reading a resource")

    for attr in dir(yf_gateway):
        if attr.startswith("get_"):
            monkeypatch.setattr(yf_gateway, attr, boom)
    monkeypatch.setattr(socket, "getaddrinfo", boom)
    try:
        import httpx

        monkeypatch.setattr(httpx.Client, "send", boom)
        monkeypatch.setattr(httpx.AsyncClient, "send", boom)
    except ImportError:
        pass

    for uri in _EXPECTED:
        assert _read(uri).strip()


def test_session_rule_text_is_derived_from_constant(monkeypatch):
    units = _read("stock-advisor://guide/units")
    about = _read("stock-advisor://about")
    for name in (
        "no_bar_today",
        "market_state_regular",
        "before_close_confirm_time",
        "after_close_confirm_time",
        "bar_date_after_today",
    ):
        assert name in units
    assert "17:00" in units and "17:00" in about

    monkeypatch.setattr(market_hours, "CLOSE_CONFIRM_TIME_KST", time(18, 0))
    patched_units = _read("stock-advisor://guide/units")
    assert "18:00" in patched_units and "17:00" not in patched_units
    assert "18:00" in _read("stock-advisor://about")


def test_workflow_step1_wording():
    workflow = _read("stock-advisor://guide/workflow")
    assert "1. 종목 데이터 도구 중에서는 항상 `resolve_ticker`를 먼저 호출합니다(0번의 절차 안내 도구는 예외)." in workflow


def test_analysis_script_tool_matches_prompt_and_surface():
    from stock_advisor_mcp.coverage import DATA_WE_DO_NOT_HAVE_NOTICE

    assert len(asyncio.run(server.mcp.list_tools())) == 6
    assert len(asyncio.run(server.mcp.list_prompts())) == 1
    assert len(asyncio.run(server.mcp.list_resources())) == 7
    assert asyncio.run(server.mcp.list_resource_templates()) == []

    plain = server.get_analysis_script()
    assert plain["script"] == server.analysis_script()
    assert plain["applies_to"]
    assert server.get_analysis_script("005930")["script"] == server.analysis_script("005930")
    assert server.get_analysis_script("005930")["script"].startswith("분석 대상: 005930\n\n")

    desc = next(t.description for t in asyncio.run(server.mcp.list_tools()) if t.name == "get_analysis_script")
    for phrase in ("가장 먼저", "resolve_ticker보다 먼저", "단순 조회에는 호출하지 마세요"):
        assert phrase in desc
    assert desc.endswith(" " + DATA_WE_DO_NOT_HAVE_NOTICE)


def test_analysis_script_tool_via_mcp_call_path_and_no_network(monkeypatch):
    def boom(*_a, **_kw):
        raise AssertionError("network access attempted by get_analysis_script")

    for attr in dir(yf_gateway):
        if attr.startswith("get_"):
            monkeypatch.setattr(yf_gateway, attr, boom)
    monkeypatch.setattr(socket, "getaddrinfo", boom)
    import httpx

    monkeypatch.setattr(httpx.Client, "send", boom)
    monkeypatch.setattr(httpx.AsyncClient, "send", boom)

    result = asyncio.run(server.mcp.call_tool("get_analysis_script", {"ticker": "005930"}))
    assert not result.is_error
    payload = json.loads("".join(getattr(c, "text", "") for c in result.content))
    assert payload["script"] == server.analysis_script("005930")
    assert payload["applies_to"]
