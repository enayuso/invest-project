"""Fundamentals Analyst — run_tool_loop's llm_client mocked, KIS calls stubbed,
no network (C-stage)."""

from __future__ import annotations

import app.agents.fundamentals_analyst as fundamentals_analyst
from app.agents import tool_loop
from app.llm.client import LLMMessage


class FakeLLM:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls: list[dict] = []

    async def complete(self, messages, *, tools=None, reasoning=False, temperature=0.0, label="", language="ko", max_tokens=None):
        self.calls.append({"messages": messages, "tools": tools})
        return self._responses.pop(0)


async def test_krx_ticker_injects_kis_financials_and_opinions(monkeypatch):
    async def fake_financials(ticker, as_of=None, period_note_out=None, recency_out=None):
        return "PER 12.3 | PBR 1.1"

    async def fake_opinion(ticker, as_of=None):
        return "매수 목표가 90000"

    monkeypatch.setattr(fundamentals_analyst, "get_kr_financials", fake_financials)
    monkeypatch.setattr(fundamentals_analyst, "get_kr_analyst_opinion", fake_opinion)
    fake = FakeLLM([LLMMessage(content="fundamentals report")])
    monkeypatch.setattr(tool_loop, "llm_client", fake)

    result = await fundamentals_analyst.analyze("005930.KS", "2026-07-16")

    assert result == "fundamentals report"
    prompt = fake.calls[0]["messages"][1]["content"]
    assert "PER 12.3 | PBR 1.1" in prompt
    assert "매수 목표가 90000" in prompt


async def test_krx_ticker_injects_the_real_kis_sector_classification(monkeypatch):
    # 2026-09-09, P1 audit item #3 (id=1354, 273060.KS/와이즈버즈): the
    # Fundamentals Analyst had no access to the exchange-registered industry
    # classification at all, and inferred "software/SaaS provider" purely
    # from a 100% gross-margin artifact. get_kr_sector_name's result must
    # reach the prompt.
    async def fake_financials(ticker, as_of=None, period_note_out=None, recency_out=None):
        return "PER 12.3 | PBR 1.1"

    async def fake_opinion(ticker, as_of=None):
        return "매수 목표가 90000"

    async def fake_sector_name(ticker, as_of=None):
        return "Exchange-registered KRX industry classification (authoritative, from KIS — not inferred from financial ratios): 광고 (KOSDAQ)."

    monkeypatch.setattr(fundamentals_analyst, "get_kr_financials", fake_financials)
    monkeypatch.setattr(fundamentals_analyst, "get_kr_analyst_opinion", fake_opinion)
    monkeypatch.setattr(fundamentals_analyst, "get_kr_sector_name", fake_sector_name)
    fake = FakeLLM([LLMMessage(content="fundamentals report")])
    monkeypatch.setattr(tool_loop, "llm_client", fake)

    await fundamentals_analyst.analyze("273060.KS", "2026-09-09")

    prompt = fake.calls[0]["messages"][1]["content"]
    assert "광고 (KOSDAQ)" in prompt
    assert "not inferred from financial ratios" in prompt


async def test_krx_ticker_tolerates_empty_sector_name(monkeypatch):
    # "" (backtest date, or a fetch failure) must not inject a stray line or
    # break the prompt — see get_kr_sector_name's own fail-open contract.
    async def fake_financials(ticker, as_of=None, period_note_out=None, recency_out=None):
        return "PER 12.3 | PBR 1.1"

    async def fake_opinion(ticker, as_of=None):
        return "매수 목표가 90000"

    async def fake_sector_name(ticker, as_of=None):
        return ""

    monkeypatch.setattr(fundamentals_analyst, "get_kr_financials", fake_financials)
    monkeypatch.setattr(fundamentals_analyst, "get_kr_analyst_opinion", fake_opinion)
    monkeypatch.setattr(fundamentals_analyst, "get_kr_sector_name", fake_sector_name)
    fake = FakeLLM([LLMMessage(content="fundamentals report")])
    monkeypatch.setattr(tool_loop, "llm_client", fake)

    result = await fundamentals_analyst.analyze("005930.KS", "2026-07-16")

    assert result == "fundamentals report"


def test_system_prompt_warns_against_inferring_business_model_from_margin():
    # id=1354's own failure shape: a 100%-gross-margin/zero-COGS artifact
    # (a KRX reporting-classification quirk, not literal zero delivery cost)
    # read as proof of a "software/SaaS" business model for an advertising
    # agency. Pin the guidance that stops the inference at the source.
    assert "NOT by itself evidence of a" in fundamentals_analyst._BASE_SYSTEM_PROMPT
    assert "SG&A rather" in fundamentals_analyst._BASE_SYSTEM_PROMPT
    assert "never relabel the company into a different" in fundamentals_analyst._BASE_SYSTEM_PROMPT


async def test_krx_ticker_corrects_a_unit_scale_mislabel_in_the_report(monkeypatch):
    """2026-08-28 domestic AI audit wiring test: a report that restates a
    supplied 억원 figure as "billion KRW" 10x too large must come back
    corrected, with the violation recorded via the out-param — see
    app.agents.unit_scale_guard for the detection/correction logic itself."""
    async def fake_financials(ticker, as_of=None, period_note_out=None, recency_out=None):
        return "revenue=951,542억원(95.15조원) (= 95,154.2 billion KRW = 95.15 trillion KRW)"

    async def fake_opinion(ticker, as_of=None):
        return "매수 목표가 90000"

    monkeypatch.setattr(fundamentals_analyst, "get_kr_financials", fake_financials)
    monkeypatch.setattr(fundamentals_analyst, "get_kr_analyst_opinion", fake_opinion)
    fake = FakeLLM([LLMMessage(content="Revenue reached 951,542 billion KRW this year.")])
    monkeypatch.setattr(tool_loop, "llm_client", fake)

    violations: list[str] = []
    result = await fundamentals_analyst.analyze("005930.KS", "2026-07-16", unit_violations_out=violations)

    assert "95,154.2 billion KRW" in result
    assert '[unit corrected from "951,542 billion KRW"]' in result
    assert len(violations) == 1


async def test_krx_analyze_path_arms_the_blind_spot_warning(monkeypatch, caplog):
    """Wiring: analyze() must pass supplied_kr_block=True, so a report that
    quotes large KRW figures matching NONE of the supplied 억원 values trips
    the id=1158 "SUSPECTED BLIND SPOT" warning (which is gated off for every
    other node — see tests/agents/test_unit_scale_guard.py)."""
    async def fake_financials(ticker, as_of=None, period_note_out=None, recency_out=None):
        return "a=111,111억원 b=222,222억원"

    async def fake_opinion(ticker, as_of=None):
        return "매수 목표가 90000"

    monkeypatch.setattr(fundamentals_analyst, "get_kr_financials", fake_financials)
    monkeypatch.setattr(fundamentals_analyst, "get_kr_analyst_opinion", fake_opinion)
    fake = FakeLLM([LLMMessage(content="revenue ₩625,389 billion and profit ₩48,337 billion")])
    monkeypatch.setattr(tool_loop, "llm_client", fake)

    with caplog.at_level("WARNING", logger="app.agents.unit_scale_guard"):
        await fundamentals_analyst.analyze("005930.KS", "2026-07-16")

    assert "SUSPECTED BLIND SPOT" in caplog.text


async def test_krx_ticker_corrects_a_ratio_label_mislabel_in_the_report(monkeypatch):
    """2026-09-01 domestic AI audit wiring test (item #4): a report that
    glosses KIS's 부채비율 as an assets-denominated ratio ("debt-to-assets")
    must come back corrected, with the violation recorded via the same
    out-param unit_scale_guard already uses — see app.agents.ratio_label_guard
    for the detection/correction logic itself."""
    async def fake_financials(ticker, as_of=None, period_note_out=None, recency_out=None):
        return "debt_to_equity_ratio=372.75% (부채비율 = liabilities÷EQUITY)"

    async def fake_opinion(ticker, as_of=None):
        return "매수 목표가 90000"

    monkeypatch.setattr(fundamentals_analyst, "get_kr_financials", fake_financials)
    monkeypatch.setattr(fundamentals_analyst, "get_kr_analyst_opinion", fake_opinion)
    fake = FakeLLM([LLMMessage(
        content="The company carries a 372.75% debt-to-assets ratio."
    )])
    monkeypatch.setattr(tool_loop, "llm_client", fake)

    violations: list[str] = []
    result = await fundamentals_analyst.analyze("005930.KS", "2026-07-16", unit_violations_out=violations)

    assert "total EQUITY" in result
    assert '[label corrected from "' in result
    assert len(violations) == 1
    assert violations[0].startswith("Fundamentals Analyst: debt_ratio_denominator")


async def test_non_krx_ticker_skips_the_unit_scale_guard(monkeypatch):
    # No kr_block at all for a non-.KS ticker — nothing to check against,
    # and the guard must not touch the report or raise.
    fake = FakeLLM([LLMMessage(content="Revenue reached 5,000 billion KRW.")])
    monkeypatch.setattr(tool_loop, "llm_client", fake)

    violations: list[str] = []
    result = await fundamentals_analyst.analyze("AAPL", "2026-07-16", unit_violations_out=violations)

    assert result == "Revenue reached 5,000 billion KRW."
    assert violations == []


async def test_non_krx_ticker_skips_kis_block_entirely(monkeypatch):
    async def should_not_be_called(ticker, as_of=None):
        raise AssertionError("KIS financials should only be fetched for .KS tickers")

    monkeypatch.setattr(fundamentals_analyst, "get_kr_financials", should_not_be_called)
    monkeypatch.setattr(fundamentals_analyst, "get_kr_analyst_opinion", should_not_be_called)
    fake = FakeLLM([LLMMessage(content="fundamentals report")])
    monkeypatch.setattr(tool_loop, "llm_client", fake)

    result = await fundamentals_analyst.analyze("AAPL", "2026-07-16")

    assert result == "fundamentals report"
    prompt = fake.calls[0]["messages"][1]["content"]
    assert prompt.startswith("Analyze AAPL's fundamentals")


async def test_company_name_grounds_the_prompt(monkeypatch):
    # Real incident (2026-07-24): without this, the model wrote up correct
    # KIS financial data for 000660.KS under a completely different
    # company's name across two separate backtest dates — the underlying
    # figures were right, only the identity it chose to write down was
    # wrong. Grounding the prompt with the real name (same as the other
    # three analysts) is the fix.
    async def fake_financials(ticker, as_of=None, period_note_out=None, recency_out=None):
        return "PER 12.3 | PBR 1.1"

    async def fake_opinion(ticker, as_of=None):
        return "매수 목표가 90000"

    monkeypatch.setattr(fundamentals_analyst, "get_kr_financials", fake_financials)
    monkeypatch.setattr(fundamentals_analyst, "get_kr_analyst_opinion", fake_opinion)
    fake = FakeLLM([LLMMessage(content="final report")])
    monkeypatch.setattr(tool_loop, "llm_client", fake)

    result = await fundamentals_analyst.analyze("000660.KS", "2026-07-16", company_name="SK hynix Inc.")

    assert result == "final report"
    user_prompt = fake.calls[0]["messages"][1]["content"]
    assert "Analyze SK hynix Inc. (000660.KS)'s fundamentals" in user_prompt


async def test_krx_ticker_excludes_tools_with_no_korean_source(monkeypatch):
    """The system prompt already tells the model to use the supplied domestic
    opinions instead of the near-empty generic consensus tool for Korean
    names — dropping it from the offered tools entirely (rather than leaving
    it there to be correctly ignored) avoids a wasted round-trip. Same
    treatment for get_institutional_ownership (2026-09-16 fix: KRX has no
    13F-style filing, so this tool always returns empty for a .KS ticker —
    it was left in the list by oversight when the KR addendum was written).
    The remaining tools (get_short_interest included — it returns real KRX
    data, just not what the US-market name implies) have no KR substitute
    and stay offered."""

    async def fake_financials(ticker, as_of=None, period_note_out=None, recency_out=None):
        return "PER 12.3"

    async def fake_opinion(ticker, as_of=None):
        return "매수"

    monkeypatch.setattr(fundamentals_analyst, "get_kr_financials", fake_financials)
    monkeypatch.setattr(fundamentals_analyst, "get_kr_analyst_opinion", fake_opinion)
    fake = FakeLLM([LLMMessage(content="report")])
    monkeypatch.setattr(tool_loop, "llm_client", fake)

    await fundamentals_analyst.analyze("005930.KS", "2026-07-16")

    tool_names = {t["function"]["name"] for t in fake.calls[0]["tools"]}
    assert "get_analyst_consensus" not in tool_names
    assert "get_institutional_ownership" not in tool_names
    assert "get_short_interest" in tool_names


async def test_non_krx_ticker_keeps_generic_analyst_consensus_tool(monkeypatch):
    fake = FakeLLM([LLMMessage(content="report")])
    monkeypatch.setattr(tool_loop, "llm_client", fake)

    await fundamentals_analyst.analyze("AAPL", "2026-07-16")

    tool_names = {t["function"]["name"] for t in fake.calls[0]["tools"]}
    assert "get_analyst_consensus" in tool_names
    assert "get_institutional_ownership" in tool_names


async def test_krx_system_prompt_explains_institutional_ownership_has_no_kr_source(monkeypatch):
    """2026-09-16 fix (engine audit item 1): the system prompt used to tell
    the model, unconditionally, that institutional ownership/analyst
    coverage are "real, independent signals worth weighing" — directly
    contradicted for a .KS ticker by the very next paragraph saying the
    generic tools are Korea-thin. The KR addendum must explain per-tool what
    actually exists instead of leaving a blanket contradiction."""
    async def fake_financials(ticker, as_of=None, period_note_out=None, recency_out=None):
        return "PER 12.3"

    async def fake_opinion(ticker, as_of=None):
        return "매수"

    monkeypatch.setattr(fundamentals_analyst, "get_kr_financials", fake_financials)
    monkeypatch.setattr(fundamentals_analyst, "get_kr_analyst_opinion", fake_opinion)
    fake = FakeLLM([LLMMessage(content="report")])
    monkeypatch.setattr(tool_loop, "llm_client", fake)

    await fundamentals_analyst.analyze("005930.KS", "2026-07-16")

    system_prompt = fake.calls[0]["messages"][0]["content"]
    assert "get_institutional_ownership has no Korean equivalent" in system_prompt
    assert "Wall Street's analyst coverage" not in system_prompt


async def test_non_krx_system_prompt_forbids_krx_short_selling_reference(monkeypatch):
    fake = FakeLLM([LLMMessage(content="report")])
    monkeypatch.setattr(tool_loop, "llm_client", fake)

    await fundamentals_analyst.analyze("AAPL", "2026-07-16")

    system_prompt = fake.calls[0]["messages"][0]["content"]
    assert "Do not reference KRX short-selling" in system_prompt
    assert "billion-KRW and trillion-KRW equivalents" not in system_prompt


async def test_krx_user_prompt_mentions_krx_program_flow_not_wall_street(monkeypatch):
    async def fake_financials(ticker, as_of=None, period_note_out=None, recency_out=None):
        return "PER 12.3"

    async def fake_opinion(ticker, as_of=None):
        return "매수"

    monkeypatch.setattr(fundamentals_analyst, "get_kr_financials", fake_financials)
    monkeypatch.setattr(fundamentals_analyst, "get_kr_analyst_opinion", fake_opinion)
    fake = FakeLLM([LLMMessage(content="report")])
    monkeypatch.setattr(tool_loop, "llm_client", fake)

    await fundamentals_analyst.analyze("005930.KS", "2026-07-16")

    user_prompt = fake.calls[0]["messages"][1]["content"]
    assert "KRX short-selling / program-flow" in user_prompt
    assert "Wall Street" not in user_prompt


async def test_non_krx_user_prompt_keeps_wall_street_instruction(monkeypatch):
    fake = FakeLLM([LLMMessage(content="report")])
    monkeypatch.setattr(tool_loop, "llm_client", fake)

    await fundamentals_analyst.analyze("AAPL", "2026-07-16")

    user_prompt = fake.calls[0]["messages"][1]["content"]
    assert "Wall Street's analyst coverage, institutional ownership, short interest" in user_prompt


async def test_trusted_tool_result_passed_through_unwrapped(monkeypatch):
    async def fake_balance_sheet(ticker, as_of=None):
        return "total assets: 1,000,000"

    monkeypatch.setattr(fundamentals_analyst, "get_balance_sheet", fake_balance_sheet)
    fake = FakeLLM([
        LLMMessage(content="", tool_calls=[{"id": "c1", "name": "get_balance_sheet", "arguments": {"ticker": "AAPL"}}]),
        LLMMessage(content="final report"),
    ])
    monkeypatch.setattr(tool_loop, "llm_client", fake)

    await fundamentals_analyst.analyze("AAPL", "2026-07-16")

    tool_result_message = fake.calls[1]["messages"][-1]
    assert tool_result_message["content"] == "total assets: 1,000,000"
