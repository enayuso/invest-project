"""Market Analyst — run_tool_loop's llm_client mocked, dataflow calls stubbed,
no network (C-stage)."""

from __future__ import annotations

import asyncio

import pytest

import app.agents.market_analyst as market_analyst
from app.agents import tool_loop
from app.llm.client import LLMMessage


@pytest.fixture(autouse=True)
def _default_indicators(monkeypatch):
    """2026-09-07 fix: get_indicators is now always fetched inside analyze()
    (not just offered as an optional tool) — every pre-existing test in this
    file predates that change and doesn't stub it, so without this autouse
    default they'd all make a real (network) get_indicators call. Individual
    tests that care about the indicator block's own content re-monkeypatch
    this after the fixture runs, which simply overrides it for that test."""

    async def fake_indicators(ticker, as_of=None):
        return (
            "Technical indicators for X (source: KIS):\n"
            "SMA(20)=100.00  SMA(50)=90.00  RSI(14)=50.0  MACD=1.00 (signal=1.00 hist=0.00)"
        )

    monkeypatch.setattr(market_analyst, "get_indicators", fake_indicators)


@pytest.fixture(autouse=True)
def _default_long_horizon(monkeypatch):
    """2026-09-16 fix: get_long_horizon_context is now always fetched inside
    analyze() (same unconditional-injection pattern as get_indicators/
    get_investor_flow) — stub it by default so pre-existing tests that don't
    care about this block don't make a real (network) call. Individual tests
    that care re-monkeypatch this after the fixture runs."""

    async def fake_long_horizon(ticker, as_of=None):
        return (
            "Longer-horizon price context for X (53 weekly bars, source: KIS):\n"
            "  window: 2025-09-15 .. 2026-09-14\n"
            "  52-week high: 100.00 (week ending 2026-01-01)\n"
            "  52-week low: 80.00 (week ending 2025-10-01)\n"
            "  latest weekly close: 95.00 (-5.0% from the 52-week high, +18.8% above the 52-week low)\n"
            "  trailing returns from weekly closes: 1M +1.0% | 3M +2.0% | 6M +5.0% | 12M +10.0%\n"
            "  weekly closes, oldest to newest (last 13): 90.00 91.00 92.00 95.00"
        )

    monkeypatch.setattr(market_analyst, "get_long_horizon_context", fake_long_horizon)


class FakeLLM:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls: list[dict] = []

    async def complete(self, messages, *, tools=None, reasoning=False, temperature=0.0, label="", language="ko", max_tokens=None):
        self.calls.append({"messages": messages, "tools": tools})
        return self._responses.pop(0)


async def test_krx_ticker_prepends_market_regime_and_injects_investor_flow(monkeypatch):
    async def fake_regime(as_of=None):
        return "KOSPI: 2500.00 (+1.20%)"

    sector_calls = {}

    async def fake_sector_breadth(as_of=None, ticker=None):
        sector_calls["ticker"] = ticker
        return "Top gaining: 반도체 +3.00%"

    async def fake_snapshot(ticker, as_of=None):
        return "close=70000"

    monkeypatch.setattr(market_analyst, "get_kr_market_regime", fake_regime)
    monkeypatch.setattr(market_analyst, "get_kr_sector_breadth", fake_sector_breadth)
    monkeypatch.setattr(market_analyst, "get_verified_snapshot", fake_snapshot)
    # 2026-09-16: get_investor_flow is no longer offered as a re-callable
    # tool (see TOOLS' module-level comment) — its data reaches the model
    # only via the unconditional flow_block injection below, so the FakeLLM
    # script here has just the one final response, no tool-call round trip.
    fake = FakeLLM([LLMMessage(content="final technical report")])

    async def fake_investor_flow(ticker, as_of=None):
        return "foreign net buy: +1,000,000"

    async def fake_vi_status(ticker, as_of=None):
        return "005930.KS: 오늘 VI 발동 이력 없음"

    monkeypatch.setattr(market_analyst, "get_investor_flow", fake_investor_flow)
    monkeypatch.setattr(market_analyst, "get_kr_vi_status", fake_vi_status)
    monkeypatch.setattr(tool_loop, "llm_client", fake)

    out: list[str] = []
    result = await market_analyst.analyze("005930.KS", "2026-07-16", flow_block_out=out)

    assert result == "final technical report"
    first_user_prompt = fake.calls[0]["messages"][1]["content"]
    assert "KOSPI: 2500.00 (+1.20%)" in first_user_prompt
    assert "Top gaining: 반도체 +3.00%" in first_user_prompt
    # get_kr_sector_breadth is now called with the ticker so it can inject
    # this stock's own KRX industry sector line.
    assert sector_calls["ticker"] == "005930.KS"
    system_prompt = fake.calls[0]["messages"][0]["content"]
    assert "never present a ranked sector's gain as a tailwind" in system_prompt
    # 2026-09-11: get_investor_flow's result is unconditionally fetched and
    # injected into the FIRST user prompt (same "don't rely on the model
    # choosing to call the right tool" fix as get_indicators, 2026-09-07).
    assert "Verified investor flow (authoritative, KRX only):" in first_user_prompt
    assert "foreign net buy: +1,000,000" in first_user_prompt
    assert out == ["foreign net buy: +1,000,000"]
    # 2026-09-16: get_investor_flow's own re-callable tool entry was dropped
    # (same rationale as get_indicators — a re-call returns the exact figure
    # already injected above, zero new information for the round trip); only
    # get_kr_vi_status remains a KR-only re-callable tool.
    assert not any(t["function"]["name"] == "get_investor_flow" for t in fake.calls[0]["tools"])
    assert any(t["function"]["name"] == "get_kr_vi_status" for t in fake.calls[0]["tools"])


async def test_non_krx_ticker_has_no_market_regime_prefix(monkeypatch):
    async def should_not_be_called(as_of=None):
        raise AssertionError("get_kr_market_regime should only be called for .KS tickers")

    async def sector_breadth_should_not_be_called(as_of=None, ticker=None):
        raise AssertionError("get_kr_sector_breadth should only be called for .KS tickers")

    async def fake_snapshot(ticker, as_of=None):
        return "close=190.00"

    monkeypatch.setattr(market_analyst, "get_kr_market_regime", should_not_be_called)
    monkeypatch.setattr(market_analyst, "get_kr_sector_breadth", sector_breadth_should_not_be_called)
    monkeypatch.setattr(market_analyst, "get_verified_snapshot", fake_snapshot)
    fake = FakeLLM([LLMMessage(content="final report")])
    monkeypatch.setattr(tool_loop, "llm_client", fake)

    result = await market_analyst.analyze("AAPL", "2026-07-16", company_name="Apple Inc.")

    assert result == "final report"
    user_prompt = fake.calls[0]["messages"][1]["content"]
    assert user_prompt.endswith("Analyze Apple Inc. (AAPL) as of 2026-07-16. Produce a technical analysis report.")
    # get_kr_vi_status is Korea-only — must not be offered for a non-KRX ticker.
    # (get_investor_flow is never offered as a tool at all, any market — see
    # the 2026-09-16 TOOLS comment — so it isn't asserted here any more.)
    assert not any(t["function"]["name"] == "get_kr_vi_status" for t in fake.calls[0]["tools"])
    # No investor-flow block at all for a non-KR ticker (2026-09-11 fix) —
    # the block is Korea-only, same as the tool it replaces.
    assert "Verified investor flow" not in user_prompt
    # WDC-regression guard: the KR sector-attribution instruction must not leak
    # into a non-KR prompt; the explicit non-KR negative addendum must be there.
    system_prompt = fake.calls[0]["messages"][0]["content"]
    assert "never present a ranked sector's gain as a tailwind" not in system_prompt
    assert "does not trade on a Korean exchange" in system_prompt


async def test_passed_in_verified_snapshot_skips_its_own_fetch(monkeypatch):
    """2026-08-10: app.orchestrator.pipeline now fetches this once and hands
    it to both Market Analyst and News + Macro Analyst, instead of Market
    Analyst fetching its own — passing it in must use it as-is and skip the
    internal get_verified_snapshot call entirely (no doubled API cost)."""

    async def should_not_be_called(ticker, as_of=None):
        raise AssertionError("get_verified_snapshot should not be called when verified_snapshot is passed in")

    monkeypatch.setattr(market_analyst, "get_verified_snapshot", should_not_be_called)
    fake = FakeLLM([LLMMessage(content="final report")])
    monkeypatch.setattr(tool_loop, "llm_client", fake)

    await market_analyst.analyze("AAPL", "2026-07-16", verified_snapshot="close=336.91 high=339.57 low=334.02")

    user_prompt = fake.calls[0]["messages"][1]["content"]
    assert "Verified price snapshot (authoritative):\nclose=336.91 high=339.57 low=334.02" in user_prompt


async def test_verified_snapshot_is_prefetched_and_injected(monkeypatch):
    """2026-07-28 fix: relying on the system prompt alone to make the model
    call get_verified_snapshot before stating a price was unenforced — a
    prior version could just quote get_stock_data's close instead. Now the
    verified figure is always in the prompt regardless of tool choice."""

    async def fake_snapshot(ticker, as_of=None):
        return "close=336.91 high=339.57 low=334.02"

    monkeypatch.setattr(market_analyst, "get_verified_snapshot", fake_snapshot)
    fake = FakeLLM([LLMMessage(content="final report")])
    monkeypatch.setattr(tool_loop, "llm_client", fake)

    await market_analyst.analyze("AAPL", "2026-07-16")

    user_prompt = fake.calls[0]["messages"][1]["content"]
    assert "Verified price snapshot (authoritative):" in user_prompt
    assert "close=336.91 high=339.57 low=334.02" in user_prompt


_BREADTH_NOT_IN_RANKING = (
    "Korean sector breadth (source: KIS, live snapshot) — a WHOLE-MARKET ranking of KRX "
    "industry indices, NOT the analyzed stock's own sector:\n"
    "Top gaining: 보험 +2.00%, 화학 +1.00%\n"
    "Top losing: 건설 -3.00%\n"
    "This stock's own KRX industry sector: 일반서비스 (KOSDAQ). Its index is NOT in the "
    "ranking above, so no move is known for this stock's own sector.\n"
    "(footer)"
)

_BREADTH_MACHINERY = (
    "Korean sector breadth (source: KIS, live snapshot) — a WHOLE-MARKET ranking:\n"
    "Top gaining: 기계·장비 +1.00%, 화학 +0.50%\n"
    "This stock's own KRX industry sector: 기계·장비 (KOSPI), +1.00% today — ranked 1 of 4 "
    "in the ranking above. This line, and only this line, is the authoritative sector.\n"
    "(footer)"
)


async def _run_analyze_with_breadth(monkeypatch, *, ticker, company_name, breadth, report_text):
    async def fake_regime(as_of=None):
        return "KOSPI: 2500.00 (+1.20%)"

    async def fake_sector_breadth(as_of=None, ticker=None):
        return breadth

    async def fake_snapshot(t, as_of=None):
        return "close=70000"

    monkeypatch.setattr(market_analyst, "get_kr_market_regime", fake_regime)
    monkeypatch.setattr(market_analyst, "get_kr_sector_breadth", fake_sector_breadth)
    monkeypatch.setattr(market_analyst, "get_verified_snapshot", fake_snapshot)

    async def fake_flow(t, as_of=None):
        return "flow"

    monkeypatch.setattr(market_analyst, "get_investor_flow", fake_flow)
    monkeypatch.setattr(market_analyst, "get_kr_vi_status", fake_flow)
    monkeypatch.setattr(tool_loop, "llm_client", FakeLLM([LLMMessage(content=report_text)]))
    return await market_analyst.analyze(ticker, "2026-09-03", company_name=company_name)


async def test_sector_reaudit_warns_on_strict_misattribution(monkeypatch, caplog):
    import logging

    with caplog.at_level(logging.WARNING, logger="app.agents.market_analyst"):
        await _run_analyze_with_breadth(
            monkeypatch,
            ticker="142760.KS",
            company_name="Appun",
            breadth=_BREADTH_NOT_IN_RANKING,
            report_text="Appun operates in the insurance sector, a competitive space.",
        )
    warnings = [r for r in caplog.records if "sector re-audit" in r.message]
    assert len(warnings) == 1
    assert "보험" in warnings[0].message and "일반서비스" in warnings[0].message


async def test_sector_reaudit_silent_when_claim_is_not_a_strict_template(monkeypatch, caplog):
    import logging

    with caplog.at_level(logging.WARNING, logger="app.agents.market_analyst"):
        await _run_analyze_with_breadth(
            monkeypatch,
            ticker="336260.KS",
            company_name="Doosan Fuel Cell",
            breadth=_BREADTH_MACHINERY,
            report_text="Doosan Fuel Cell operates in the machinery/fuel cell equipment space.",
        )
    assert not [r for r in caplog.records if "sector re-audit" in r.message]


async def test_system_prompt_instructs_against_describing_a_live_session_as_closed(monkeypatch):
    """Prompt-regression test for the MRVL incident (2026-08-26, 09:42 ET —
    see app.dataflows.session_state's module docstring): the data-label fix
    alone isn't enough on its own merits without this instruction, since the
    system prompt already told the model the snapshot was "authoritative" —
    that framing is exactly what let a wrong label get amplified instead of
    questioned. Asserts the instruction text survives, not just the data fix."""

    async def fake_snapshot(ticker, as_of=None):
        return "last=237.04 session_high=240.77 session_low=236.79 volume_so_far=769562"

    monkeypatch.setattr(market_analyst, "get_verified_snapshot", fake_snapshot)
    fake = FakeLLM([LLMMessage(content="final report")])
    monkeypatch.setattr(tool_loop, "llm_client", fake)

    await market_analyst.analyze("MRVL", "2026-08-26")

    system_prompt = fake.calls[0]["messages"][0]["content"]
    assert "SESSION IN PROGRESS" in system_prompt
    assert "concluded trading" in system_prompt


async def test_indicator_block_is_fetched_and_injected(monkeypatch):
    """2026-09-07 fix: get_indicators was previously offered only as an
    optional tool — live audit found the model skipped it 35.7% of the time
    on domestic tickers. The real value must now always be injected into the
    prompt, the same way verified_snapshot already is."""

    async def fake_snapshot(ticker, as_of=None):
        return "close=336.91 high=339.57 low=334.02"

    async def fake_indicators(ticker, as_of=None):
        return "Technical indicators for AAPL:\nSMA(20)=330.00  SMA(50)=320.00  RSI(14)=55.0  MACD=1.20 (signal=1.00 hist=0.20)"

    monkeypatch.setattr(market_analyst, "get_verified_snapshot", fake_snapshot)
    monkeypatch.setattr(market_analyst, "get_indicators", fake_indicators)
    fake = FakeLLM([LLMMessage(content="final report")])
    monkeypatch.setattr(tool_loop, "llm_client", fake)

    out: list[str] = []
    await market_analyst.analyze("AAPL", "2026-07-16", indicator_block_out=out)

    user_prompt = fake.calls[0]["messages"][1]["content"]
    assert "Verified technical indicators (authoritative):" in user_prompt
    assert "SMA(20)=330.00" in user_prompt
    assert len(out) == 1
    assert "SMA(20)=330.00" in out[0]


async def test_indicator_fetch_failure_degrades_to_not_available(monkeypatch):
    async def fake_snapshot(ticker, as_of=None):
        return "close=336.91 high=339.57 low=334.02"

    async def fake_indicators_raises(ticker, as_of=None):
        raise RuntimeError("KIS returned too few candles for AAPL indicators")

    monkeypatch.setattr(market_analyst, "get_verified_snapshot", fake_snapshot)
    monkeypatch.setattr(market_analyst, "get_indicators", fake_indicators_raises)
    fake = FakeLLM([LLMMessage(content="final report")])
    monkeypatch.setattr(tool_loop, "llm_client", fake)

    out: list[str] = []
    result = await market_analyst.analyze("AAPL", "2026-07-16", indicator_block_out=out)

    assert result == "final report"
    user_prompt = fake.calls[0]["messages"][1]["content"]
    assert "NOT AVAILABLE" in user_prompt
    assert len(out) == 1
    assert "NOT AVAILABLE" in out[0]


async def test_flow_block_is_fetched_and_injected(monkeypatch):
    """2026-09-11 fix: get_investor_flow was previously offered only as an
    optional tool — a live audit (id=1389/320000.KS) found the model instead
    writing a fabricated, hedged flow figure ("Net Seller (-2,000 est.)")
    into its report with no real tool call backing it. The real value must
    now always be injected into the prompt, the same way get_indicators
    already is (2026-09-07)."""

    async def fake_regime(as_of=None):
        return "KOSPI: 2500.00 (+1.20%)"

    async def fake_sector_breadth(as_of=None, ticker=None):
        return "Top gaining: 반도체 +3.00%"

    async def fake_snapshot(ticker, as_of=None):
        return "close=70000"

    async def fake_flow(ticker, as_of=None):
        return "Investor net-buy flow for 005930.KS: foreign +500,000, institution -200,000"

    monkeypatch.setattr(market_analyst, "get_kr_market_regime", fake_regime)
    monkeypatch.setattr(market_analyst, "get_kr_sector_breadth", fake_sector_breadth)
    monkeypatch.setattr(market_analyst, "get_verified_snapshot", fake_snapshot)
    monkeypatch.setattr(market_analyst, "get_investor_flow", fake_flow)
    fake = FakeLLM([LLMMessage(content="final report")])
    monkeypatch.setattr(tool_loop, "llm_client", fake)

    out: list[str] = []
    await market_analyst.analyze("005930.KS", "2026-07-16", flow_block_out=out)

    user_prompt = fake.calls[0]["messages"][1]["content"]
    assert "Verified investor flow (authoritative, KRX only):" in user_prompt
    assert "foreign +500,000, institution -200,000" in user_prompt
    assert len(out) == 1
    assert "foreign +500,000, institution -200,000" in out[0]


async def test_flow_fetch_failure_degrades_to_not_available(monkeypatch):
    async def fake_regime(as_of=None):
        return "KOSPI: 2500.00 (+1.20%)"

    async def fake_sector_breadth(as_of=None, ticker=None):
        return "Top gaining: 반도체 +3.00%"

    async def fake_snapshot(ticker, as_of=None):
        return "close=70000"

    async def fake_flow_raises(ticker, as_of=None):
        raise RuntimeError("KIS returned an error for 005930.KS investor flow")

    monkeypatch.setattr(market_analyst, "get_kr_market_regime", fake_regime)
    monkeypatch.setattr(market_analyst, "get_kr_sector_breadth", fake_sector_breadth)
    monkeypatch.setattr(market_analyst, "get_verified_snapshot", fake_snapshot)
    monkeypatch.setattr(market_analyst, "get_investor_flow", fake_flow_raises)
    fake = FakeLLM([LLMMessage(content="final report")])
    monkeypatch.setattr(tool_loop, "llm_client", fake)

    out: list[str] = []
    result = await market_analyst.analyze("005930.KS", "2026-07-16", flow_block_out=out)

    assert result == "final report"
    user_prompt = fake.calls[0]["messages"][1]["content"]
    assert "NOT AVAILABLE" in user_prompt
    assert len(out) == 1
    assert "NOT AVAILABLE" in out[0]


def test_system_prompt_forbids_estimated_indicator_values():
    assert "Never estimate, assume, approximate, or infer an indicator value" in market_analyst._BASE_SYSTEM_PROMPT
    assert "overbought is RSI above 70, oversold is RSI below 30" in market_analyst._BASE_SYSTEM_PROMPT


def test_kr_addendum_forbids_estimated_flow_values():
    assert (
        "never estimate, guess, or infer a flow direction or number"
        in market_analyst._KR_SYSTEM_PROMPT_ADDENDUM
    )
    assert "no flow data was available this cycle" in market_analyst._KR_SYSTEM_PROMPT_ADDENDUM


def test_kr_addendum_forbids_index_contribution_claims():
    assert (
        "not this stock's contribution to, weight in, or impact on the index"
        in market_analyst._KR_SYSTEM_PROMPT_ADDENDUM
    )


def test_indicators_not_duplicated_as_a_tool():
    """2026-09-16 (engine audit follow-up): get_indicators used to also be
    offered here as a re-callable tool, alongside being unconditionally
    injected into the prompt (see `analyze`'s indicator_block, 2026-09-07).
    A re-call returns the exact deterministic figures already in the
    prompt — zero new information for the tool round trip's worth of
    tokens — so it was dropped from TOOLS. This inverts the prior test
    (`test_indicators_still_offered_as_a_tool`); get_indicators itself is
    unchanged and still fetched/injected every call (see
    `test_indicator_block_is_fetched_and_injected` above)."""
    assert not any(t["function"]["name"] == "get_indicators" for t in market_analyst.TOOLS)


def test_investor_flow_not_duplicated_as_a_tool():
    """Same rationale as `test_indicators_not_duplicated_as_a_tool` above —
    get_investor_flow's data is unconditionally injected (see
    `test_flow_block_is_fetched_and_injected`), so re-offering it as a
    callable tool was pure redundancy."""
    assert not any(t["function"]["name"] == "get_investor_flow" for t in market_analyst.TOOLS)


def test_get_stock_data_tool_has_no_lookback_days_param():
    """2026-09-16 fix: `lookback_days` never actually changed what
    get_stock_data rendered (always the last 10 sessions — see
    `_yf_stock_data`'s own note) — a dead parameter the model could set with
    no effect, dropped from the schema rather than left to be re-discovered."""
    stock_data_tool = next(t for t in market_analyst.TOOLS if t["function"]["name"] == "get_stock_data")
    assert "lookback_days" not in stock_data_tool["function"]["parameters"]["properties"]


async def test_long_horizon_block_is_fetched_and_injected(monkeypatch):
    """2026-09-16 fix (engine audit): get_stock_data's tool schema advertised
    `lookback_days: default 180` but the daily table always rendered just
    the last 10 sessions — the model was expected to describe a 52-week
    range or multi-month trend with zero supporting data. A real value must
    now always be injected into the prompt, the same unconditional-injection
    pattern as get_indicators (2026-09-07) / get_investor_flow (2026-09-11)."""

    async def fake_snapshot(ticker, as_of=None):
        return "close=336.91 high=339.57 low=334.02"

    async def fake_long_horizon(ticker, as_of=None):
        return "Longer-horizon price context for AAPL (53 weekly bars, source: yfinance):\n  52-week high: 400.00"

    monkeypatch.setattr(market_analyst, "get_verified_snapshot", fake_snapshot)
    monkeypatch.setattr(market_analyst, "get_long_horizon_context", fake_long_horizon)
    fake = FakeLLM([LLMMessage(content="final report")])
    monkeypatch.setattr(tool_loop, "llm_client", fake)

    out: list[str] = []
    await market_analyst.analyze("AAPL", "2026-07-16", long_horizon_block_out=out)

    user_prompt = fake.calls[0]["messages"][1]["content"]
    assert "Longer-horizon price context for AAPL" in user_prompt
    assert "52-week high: 400.00" in user_prompt
    assert len(out) == 1
    assert "52-week high: 400.00" in out[0]


async def test_long_horizon_fetch_failure_degrades_to_not_available(monkeypatch):
    async def fake_snapshot(ticker, as_of=None):
        return "close=336.91 high=339.57 low=334.02"

    async def fake_long_horizon_raises(ticker, as_of=None):
        raise ValueError("AAPL: only 6 weekly bars, need at least 12 for long-horizon context")

    monkeypatch.setattr(market_analyst, "get_verified_snapshot", fake_snapshot)
    monkeypatch.setattr(market_analyst, "get_long_horizon_context", fake_long_horizon_raises)
    fake = FakeLLM([LLMMessage(content="final report")])
    monkeypatch.setattr(tool_loop, "llm_client", fake)

    out: list[str] = []
    result = await market_analyst.analyze("AAPL", "2026-07-16", long_horizon_block_out=out)

    assert result == "final report"
    user_prompt = fake.calls[0]["messages"][1]["content"]
    assert "NOT AVAILABLE" in user_prompt
    assert len(out) == 1
    assert "NOT AVAILABLE" in out[0]


def test_system_prompt_instructs_long_horizon_block_usage():
    assert "longer-horizon price-context block" in market_analyst._BASE_SYSTEM_PROMPT
    assert "cannot support any statement about a multi-week or multi-month move" in market_analyst._BASE_SYSTEM_PROMPT


def test_system_prompt_forbids_upgrading_a_short_window_to_52_week():
    """2026-09-16 fix: the long-horizon block only says '52-week' when a full
    year of weekly bars exists (see app.dataflows.long_horizon), otherwise a
    shorter label like '36-week' — the model must not launder that back into
    a 52-week claim."""
    assert "never restate a shorter window as a 52-week high" in market_analyst._BASE_SYSTEM_PROMPT


async def test_independent_vendor_fetches_run_concurrently(monkeypatch):
    """2026-09-16 fix: indicators / long-horizon / investor-flow are
    mutually independent vendor calls that used to be awaited one after
    another, paying three round-trips of latency per ticker."""
    in_flight = {"now": 0, "peak": 0}

    def _tracked(text):
        async def _fn(ticker, as_of=None):
            in_flight["now"] += 1
            in_flight["peak"] = max(in_flight["peak"], in_flight["now"])
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            in_flight["now"] -= 1
            return text
        return _fn

    async def fake_regime(as_of=None):
        return "KOSPI: 2500.00 (+1.20%)"

    async def fake_sector_breadth(as_of=None, ticker=None):
        return "Top gaining: 반도체 +3.00%"

    async def fake_snapshot(ticker, as_of=None):
        return "close=70000"

    monkeypatch.setattr(market_analyst, "get_kr_market_regime", fake_regime)
    monkeypatch.setattr(market_analyst, "get_kr_sector_breadth", fake_sector_breadth)
    monkeypatch.setattr(market_analyst, "get_verified_snapshot", fake_snapshot)
    monkeypatch.setattr(market_analyst, "get_indicators", _tracked("RSI(14)=50.0"))
    monkeypatch.setattr(market_analyst, "get_long_horizon_context", _tracked("52-week high: 1"))
    monkeypatch.setattr(market_analyst, "get_investor_flow", _tracked("flow"))
    monkeypatch.setattr(market_analyst, "get_kr_vi_status", _tracked("no VI"))
    fake = FakeLLM([LLMMessage(content="final report")])
    monkeypatch.setattr(tool_loop, "llm_client", fake)

    await market_analyst.analyze("005930.KS", "2026-07-16")

    assert in_flight["peak"] == 3
