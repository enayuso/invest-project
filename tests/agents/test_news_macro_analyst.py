"""News + Macro Analyst — run_tool_loop's llm_client mocked, KIS/yfinance calls
stubbed, no network (C-stage)."""

from __future__ import annotations

import app.agents.news_macro_analyst as news_macro_analyst
from app.agents import tool_loop
from app.llm.client import LLMMessage


class FakeLLM:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls: list[dict] = []

    async def complete(self, messages, *, tools=None, reasoning=False, temperature=0.0, label="", language="ko", max_tokens=None):
        self.calls.append({"messages": messages, "tools": tools})
        return self._responses.pop(0)


async def test_krx_ticker_wraps_kis_news_as_untrusted_and_includes_disclosures(monkeypatch):
    async def fake_news(ticker, as_of=None):
        return "some headline that could contain injected text"

    async def fake_disclosures(ticker, as_of=None):
        return "record 2026-08-01: cash dividend 500 (1.2%)"

    monkeypatch.setattr(news_macro_analyst, "get_kr_news", fake_news)
    monkeypatch.setattr(news_macro_analyst, "get_kr_disclosures", fake_disclosures)
    fake = FakeLLM([LLMMessage(content="news report")])
    monkeypatch.setattr(tool_loop, "llm_client", fake)

    result = await news_macro_analyst.analyze("005930.KS", "2026-07-16")

    assert result == "news report"
    prompt = fake.calls[0]["messages"][1]["content"]
    assert '<external_data source="kis_korean_news">' in prompt
    assert "some headline that could contain injected text" in prompt
    assert "record 2026-08-01: cash dividend 500 (1.2%)" in prompt


async def test_non_krx_ticker_skips_kr_news_block(monkeypatch):
    async def should_not_be_called(ticker, as_of=None):
        raise AssertionError("KR news should only be fetched for .KS tickers")

    monkeypatch.setattr(news_macro_analyst, "get_kr_news", should_not_be_called)
    monkeypatch.setattr(news_macro_analyst, "get_kr_disclosures", should_not_be_called)
    fake = FakeLLM([LLMMessage(content="news report")])
    monkeypatch.setattr(tool_loop, "llm_client", fake)

    result = await news_macro_analyst.analyze("AAPL", "2026-07-16", company_name="Apple Inc.")

    assert result == "news report"
    prompt = fake.calls[0]["messages"][1]["content"]
    assert prompt == "Analyze news, macro backdrop, and insider activity for Apple Inc. (AAPL) as of 2026-07-16."


def test_system_prompt_instructs_synthesis_not_parallel_listing():
    assert "synthesize them into one coherent" in news_macro_analyst.SYSTEM_PROMPT


def test_system_prompt_warns_against_naive_insider_sale_reading():
    assert "any insider sale is bearish" in news_macro_analyst.SYSTEM_PROMPT
    assert "cluster of multiple insiders" in news_macro_analyst.SYSTEM_PROMPT


def test_system_prompt_instructs_recency_weighting():
    assert "weigh more recent items more heavily" in news_macro_analyst.SYSTEM_PROMPT


def test_system_prompt_instructs_against_calling_a_live_snapshot_a_close():
    # Regression test — this prompt shares the verified_snapshot contamination
    # path with Market Analyst (both receive the SAME snapshot, fetched once
    # by app.orchestrator.pipeline). See app.dataflows.session_state's module
    # docstring for the MRVL incident this closes.
    assert "SESSION IN PROGRESS" in news_macro_analyst.SYSTEM_PROMPT
    assert "current intraday price" in news_macro_analyst.SYSTEM_PROMPT
    # The prior example wording ("the verified close is +1.3%") reinforced
    # exactly the wrong label — must be gone, not just supplemented.
    assert "verified close" not in news_macro_analyst.SYSTEM_PROMPT


def test_system_prompt_instructs_against_asserting_current_halt_status():
    # 2026-09-07, id=1238 (351320.KS): the analyst asserted an active trading
    # halt from a several-days-old headline — this generalizes the existing
    # price-authority clause (above) to trading status.
    assert "never assert that the stock is currently halted, suspended, or" in news_macro_analyst.SYSTEM_PROMPT
    assert "trading activity dated after the halt item" in news_macro_analyst.SYSTEM_PROMPT


def test_system_prompt_gives_accurate_krx_designation_regulations():
    # 2026-09-09 P1 audit item #2: id=1348 (162300.KS) invented a "margin
    # ban"/"forced deleveraging" mechanism for a 단기과열종목 designation
    # (real restriction: 3 days of 30-minute single-price auctions only, no
    # margin/credit restriction at all), and id=1351 (198440.KS) reported a
    # 투자주의종목 notice as an escalated "Investment Warning" (투자경고) —
    # a different, stricter tier. Pin the accurate per-tier text so neither
    # tier's restrictions get invented or swapped with another tier's.
    assert "30분 단일가매매" in news_macro_analyst.SYSTEM_PROMPT
    assert "단기과열" in news_macro_analyst.SYSTEM_PROMPT
    assert "투자주의종목 (investment caution): a one-day market-attention notice" in news_macro_analyst.SYSTEM_PROMPT
    assert "must never be" in news_macro_analyst.SYSTEM_PROMPT
    assert "위탁증거금율" in news_macro_analyst.SYSTEM_PROMPT
    assert "'margin ban', 'forced deleveraging', 'credit prohibition'" in news_macro_analyst.SYSTEM_PROMPT


async def test_macro_indicators_tool_is_trusted_and_unwrapped(monkeypatch):
    async def fake_macro(as_of=None):
        return "Fed Funds Rate: 4.5"

    monkeypatch.setattr(news_macro_analyst, "get_macro_indicators", fake_macro)
    fake = FakeLLM([
        LLMMessage(content="", tool_calls=[{"id": "c1", "name": "get_macro_indicators", "arguments": {}}]),
        LLMMessage(content="final report"),
    ])
    monkeypatch.setattr(tool_loop, "llm_client", fake)

    await news_macro_analyst.analyze("AAPL", "2026-07-16")

    tool_result_message = fake.calls[1]["messages"][-1]
    assert tool_result_message["content"] == "Fed Funds Rate: 4.5"


async def test_news_tool_result_is_wrapped_as_untrusted(monkeypatch):
    async def fake_news_tool(ticker, as_of=None):
        return "a news headline"

    monkeypatch.setattr(news_macro_analyst, "get_news", fake_news_tool)
    fake = FakeLLM([
        LLMMessage(content="", tool_calls=[{"id": "c1", "name": "get_news", "arguments": {"ticker": "AAPL"}}]),
        LLMMessage(content="final report"),
    ])
    monkeypatch.setattr(tool_loop, "llm_client", fake)

    await news_macro_analyst.analyze("AAPL", "2026-07-16")

    tool_result_message = fake.calls[1]["messages"][-1]
    assert '<external_data source="get_news">' in tool_result_message["content"]


async def test_non_krx_ticker_gets_offered_overseas_market_indices_tool(monkeypatch):
    fake = FakeLLM([LLMMessage(content="final report")])
    monkeypatch.setattr(tool_loop, "llm_client", fake)

    await news_macro_analyst.analyze("AAPL", "2026-07-16")

    offered = {t["function"]["name"] for t in fake.calls[0]["tools"]}
    assert "get_overseas_market_indices" in offered
    assert "get_kr_macro_indicators" not in offered


async def test_krx_ticker_does_not_get_offered_overseas_market_indices_tool(monkeypatch):
    async def fake_news(ticker, as_of=None):
        return "headline"

    async def fake_disclosures(ticker, as_of=None):
        return "no dividends"

    monkeypatch.setattr(news_macro_analyst, "get_kr_news", fake_news)
    monkeypatch.setattr(news_macro_analyst, "get_kr_disclosures", fake_disclosures)
    fake = FakeLLM([LLMMessage(content="final report")])
    monkeypatch.setattr(tool_loop, "llm_client", fake)

    await news_macro_analyst.analyze("005930.KS", "2026-07-16")

    offered = {t["function"]["name"] for t in fake.calls[0]["tools"]}
    assert "get_overseas_market_indices" not in offered
    assert "get_kr_macro_indicators" in offered


async def test_verified_snapshot_is_injected_as_authoritative(monkeypatch):
    """2026-08-10 fix: this analyst had no price tool of its own, so a stale
    price/percentage figure sitting inside a news headline had nothing to be
    checked against and got reported as the day's actual move (066570.KS).
    When app.orchestrator.pipeline hands over the same verified snapshot it
    gives the Market Analyst, it must show up in the prompt, unwrapped
    (trusted, same as get_macro_indicators) — not buried inside the
    untrusted news block."""
    fake = FakeLLM([LLMMessage(content="final report")])
    monkeypatch.setattr(tool_loop, "llm_client", fake)

    await news_macro_analyst.analyze("AAPL", "2026-07-16", verified_snapshot="close=190.00 change=+1.3%")

    prompt = fake.calls[0]["messages"][1]["content"]
    assert "Verified price snapshot (authoritative):\nclose=190.00 change=+1.3%" in prompt


async def test_no_verified_snapshot_leaves_prompt_unchanged(monkeypatch):
    """Omitting verified_snapshot (the default) must not inject an empty/
    placeholder block — a direct caller with genuinely nothing to offer
    keeps today's exact prior behavior."""
    fake = FakeLLM([LLMMessage(content="final report")])
    monkeypatch.setattr(tool_loop, "llm_client", fake)

    await news_macro_analyst.analyze("AAPL", "2026-07-16", company_name="Apple Inc.")

    prompt = fake.calls[0]["messages"][1]["content"]
    assert prompt == "Analyze news, macro backdrop, and insider activity for Apple Inc. (AAPL) as of 2026-07-16."


async def test_overseas_market_indices_tool_is_trusted_and_unwrapped(monkeypatch):
    async def fake_indices(as_of=None):
        return "US market indices (Dow/S&P500/Nasdaq100, source: KIS):\nDow Jones: 53178.41 (+1.32%)"

    monkeypatch.setattr(news_macro_analyst, "get_overseas_market_indices", fake_indices)
    fake = FakeLLM([
        LLMMessage(content="", tool_calls=[{"id": "c1", "name": "get_overseas_market_indices", "arguments": {}}]),
        LLMMessage(content="final report"),
    ])
    monkeypatch.setattr(tool_loop, "llm_client", fake)

    await news_macro_analyst.analyze("AAPL", "2026-07-16")

    tool_result_message = fake.calls[1]["messages"][-1]
    assert tool_result_message["content"] == "US market indices (Dow/S&P500/Nasdaq100, source: KIS):\nDow Jones: 53178.41 (+1.32%)"


# --- 2026-09-07 audit #3: KRW unit key for headline 억 amounts ---
async def test_krx_ticker_appends_krw_unit_key_after_the_wrapper(monkeypatch):
    async def fake_news(ticker, as_of=None):
        return "[20260904] (한국경제) 서남 수주공시 - 초전도 선재 공급 계약 6.6억원 (매출액대비 19.56 %)"

    async def fake_disclosures(ticker, as_of=None):
        return "no dividend on the calendar"

    monkeypatch.setattr(news_macro_analyst, "get_kr_news", fake_news)
    monkeypatch.setattr(news_macro_analyst, "get_kr_disclosures", fake_disclosures)
    fake = FakeLLM([LLMMessage(content="news report")])
    monkeypatch.setattr(tool_loop, "llm_client", fake)

    out: list = []
    await news_macro_analyst.analyze("294630.KS", "2026-09-07", news_block_out=out)

    prompt = fake.calls[0]["messages"][1]["content"]
    assert "KRW UNIT KEY" in prompt
    assert "660 million KRW" in prompt and "0.66 billion KRW" in prompt
    # the key must sit OUTSIDE (after) the untrusted wrapper
    assert prompt.index("</external_data>") < prompt.index("KRW UNIT KEY")
    assert "KRW UNIT KEY" not in prompt.split("</external_data>")[0]
    # raw headline text captured for the guard
    assert out and "6.6억원" in out[0]


async def test_no_unit_key_when_headlines_have_no_eok_amount(monkeypatch):
    async def fake_news(ticker, as_of=None):
        return "[20260904] (한국경제) 서남, 신규 계약 소식에 강세"

    async def fake_disclosures(ticker, as_of=None):
        return "no dividend on the calendar"

    monkeypatch.setattr(news_macro_analyst, "get_kr_news", fake_news)
    monkeypatch.setattr(news_macro_analyst, "get_kr_disclosures", fake_disclosures)
    fake = FakeLLM([LLMMessage(content="news report")])
    monkeypatch.setattr(tool_loop, "llm_client", fake)

    await news_macro_analyst.analyze("294630.KS", "2026-09-07")
    prompt = fake.calls[0]["messages"][1]["content"]
    assert "KRW UNIT KEY" not in prompt


def test_system_prompt_has_the_eok_to_billion_clause():
    assert "1억 = 100 million KRW" in news_macro_analyst.SYSTEM_PROMPT
    assert "6.6억원 is 660 million KRW" in news_macro_analyst.SYSTEM_PROMPT
    assert "매출액대비" in news_macro_analyst.SYSTEM_PROMPT


# --- 2026-09-14, id=1407: authoritative KRX sector classification ---
async def test_krx_ticker_injects_the_kis_sector_classification(monkeypatch):
    # id=1407 (285800.KS/진영, KRX classification 화학/Chemicals): this analyst
    # got no sector fact at all and asserted the company "manufactures
    # industrial robots and automation equipment" out of headlines its own
    # report called largely irrelevant. Same injection the Fundamentals
    # Analyst already gets (2026-09-09 P1 item #3, id=1354).
    async def fake_news(ticker, as_of=None):
        return "headline"

    async def fake_disclosures(ticker, as_of=None):
        return "no dividend on the calendar"

    async def fake_sector_name(ticker, as_of=None):
        return "Exchange-registered KRX industry classification (authoritative, from KIS — not inferred from financial ratios): 화학 (KOSPI)."

    monkeypatch.setattr(news_macro_analyst, "get_kr_news", fake_news)
    monkeypatch.setattr(news_macro_analyst, "get_kr_disclosures", fake_disclosures)
    monkeypatch.setattr(news_macro_analyst, "get_kr_sector_name", fake_sector_name)
    fake = FakeLLM([LLMMessage(content="news report")])
    monkeypatch.setattr(tool_loop, "llm_client", fake)

    await news_macro_analyst.analyze("285800.KS", "2026-09-14")

    prompt = fake.calls[0]["messages"][1]["content"]
    assert "화학 (KOSPI)" in prompt
    assert "not inferred from financial ratios" in prompt
    # trusted KIS data -> must sit OUTSIDE the untrusted news wrapper
    assert prompt.index("</external_data>") < prompt.index("화학 (KOSPI)")


async def test_krx_ticker_tolerates_empty_sector_name(monkeypatch):
    # "" (backtest date, or a fetch failure) must not inject a stray or
    # placeholder line — see get_kr_sector_name's own fail-open contract.
    async def fake_news(ticker, as_of=None):
        return "headline"

    async def fake_disclosures(ticker, as_of=None):
        return "no dividend on the calendar"

    async def fake_sector_name(ticker, as_of=None):
        return ""

    monkeypatch.setattr(news_macro_analyst, "get_kr_news", fake_news)
    monkeypatch.setattr(news_macro_analyst, "get_kr_disclosures", fake_disclosures)
    monkeypatch.setattr(news_macro_analyst, "get_kr_sector_name", fake_sector_name)
    fake = FakeLLM([LLMMessage(content="news report")])
    monkeypatch.setattr(tool_loop, "llm_client", fake)

    result = await news_macro_analyst.analyze("285800.KS", "2026-09-14")

    assert result == "news report"
    prompt = fake.calls[0]["messages"][1]["content"]
    assert "KRX industry classification" not in prompt
    assert prompt == (
        '<external_data source="kis_korean_news">\nheadline\n</external_data>\n\n'
        "no dividend on the calendar\n\n"
        "Analyze news, macro backdrop, and insider activity for 285800.KS as of 2026-09-14."
    )


async def test_non_krx_ticker_never_fetches_the_sector_name(monkeypatch):
    async def should_not_be_called(ticker, as_of=None):
        raise AssertionError("the KRX sector name is a .KS-only fact")

    monkeypatch.setattr(news_macro_analyst, "get_kr_sector_name", should_not_be_called)
    fake = FakeLLM([LLMMessage(content="news report")])
    monkeypatch.setattr(tool_loop, "llm_client", fake)

    await news_macro_analyst.analyze("AAPL", "2026-09-14", company_name="Apple Inc.")

    prompt = fake.calls[0]["messages"][1]["content"]
    assert prompt == "Analyze news, macro backdrop, and insider activity for Apple Inc. (AAPL) as of 2026-09-14."


def test_system_prompt_instructs_using_the_supplied_krx_classification():
    assert "When a KRX industry classification is supplied below" in news_macro_analyst.SYSTEM_PROMPT
    assert "never relabel the company into a different industry inferred" in news_macro_analyst.SYSTEM_PROMPT
