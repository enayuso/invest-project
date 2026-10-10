from __future__ import annotations

import asyncio
from datetime import date

from app.agents.tool_loop import IDENTITY_GROUNDING_NOTE, bind_as_of, run_tool_loop, ticker_label
from app.agents.untrusted_content import wrap_untrusted
from app.dataflows.fred_provider import get_macro_indicators
from app.dataflows.kis_provider import (
    get_kr_disclosures,
    get_kr_macro_indicators,
    get_kr_news,
    get_kr_sector_name,
    get_overseas_market_indices,
    kr_amount_unit_key,
)
from app.dataflows.yfinance import get_insider_txns, get_news
from app.llm.language import DEFAULT_LANGUAGE

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_news",
            "description": "Get recent news headlines for a ticker.",
            "parameters": {
                "type": "object",
                "properties": {"ticker": {"type": "string"}},
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_macro_indicators",
            "description": (
                "Get current macro backdrop indicators (fed funds rate, CPI, "
                "unemployment, GDP). May be unavailable if not configured — "
                "treat an 'unavailable' result as missing data, not a hard error."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_kr_macro_indicators",
            "description": (
                "Get the domestic (Korea) macro backdrop — KOSPI/KOSDAQ index levels and the USD/KRW "
                "rate. Use this instead of get_macro_indicators (US Fed/CPI data) for a KRX-listed "
                "ticker: it is the market this stock actually trades in, unlike the US Fed funds rate."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_overseas_market_indices",
            "description": (
                "Get current US market breadth — Dow Jones/S&P 500/Nasdaq 100 levels and their "
                "day's change %. Different purpose from get_macro_indicators (Fed/CPI/economic data): "
                "this is 'which direction is the US stock market itself moving right now', the "
                "overseas counterpart to get_kr_macro_indicators' KOSPI/KOSDAQ. Use both together for "
                "a non-KR ticker, not one instead of the other."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_insider_txns",
            "description": "Get recent insider transactions for a ticker.",
            "parameters": {
                "type": "object",
                "properties": {"ticker": {"type": "string"}},
                "required": ["ticker"],
            },
        },
    },
]

SYSTEM_PROMPT = (
    "You are the News + Macro Analyst on an equity research desk. Form your own "
    "read of what the ticker-specific news implies for this stock FIRST, then "
    "explicitly weigh whether the broader macro backdrop reinforces or cuts "
    "against that read — do not just list news, macro, and insider activity as "
    "three separate parallel sections; synthesize them into one coherent "
    "directional thesis, noting explicitly where they agree or conflict.\n\n"
    "When weighing insider transactions, avoid naive equivalences like 'any "
    "insider sale is bearish' — a single executive's modest sale is routinely "
    "for personal liquidity (taxes, diversification) and carries little signal; "
    "a cluster of multiple insiders trading the same direction, or an "
    "unusually large transaction relative to that person's typical holdings, "
    "is the more meaningful pattern worth weighing.\n\n"
    "If individual news items carry their own dates, weigh more recent items "
    "more heavily than older ones, and do not present a stale item as if it "
    "were a fresh development — this is separate from (and in addition to) the "
    "analysis date itself always being the authoritative 'as of' date.\n\n"
    "If macro data or ticker news is unavailable, note it briefly and continue "
    "— do not treat it as a blocking failure.\n\n"
    "A news headline's own price/percentage-change figure is a snapshot from "
    "whenever that article was written, not necessarily current — a stale "
    "one can flatly contradict the actual latest price. When a verified price "
    "snapshot is provided below, that figure is authoritative for the "
    "stock's current price/direction; never state a price move sourced only "
    "from a news headline. If a headline's own figure conflicts with the "
    "verified snapshot, say so explicitly (e.g. \"an intraday recap cited "
    "-2%, but the verified snapshot shows +1.3% — that recap predates the "
    "reversal\") rather than silently repeating the headline's number or "
    "silently favoring the snapshot with no mention of the conflict. If the "
    "verified snapshot is marked SESSION IN PROGRESS, refer to it as the "
    "current intraday price, never as a close.\n\n"
    "The same applies to a stock's trading status. A headline or disclosure about a trading "
    "halt, a suspension, a suspension warning, an 관리종목/투자위험종목 designation, or a "
    "조회공시 매매거래정지 is an event dated to when it was published; most such halts are "
    "procedural and last part of one session. You have no tool that reports current trading "
    "status, so you must never assert that the stock is currently halted, suspended, or "
    "awaiting resumption. Describe such an item by its date and as a risk ('a preliminary "
    "suspension was announced on Sept 4'), never as a present state ('the stock is "
    "suspended', 'when trading resumes'). If the verified price snapshot below shows trading "
    "activity dated after the halt item, note that trading has evidently continued since.\n\n"
    "KRX designations are not interchangeable — each tier carries a specific, different set of "
    "restrictions, and news headlines routinely use loose or overlapping language. Never describe "
    "one tier using another tier's restrictions, and never state a stronger restriction than the "
    "tier actually carries:\n"
    "- 단기과열종목 (short-term overheating): for 3 trading days, trades ONLY via 30-minute "
    "single-price auctions (30분 단일가매매). NO margin/credit restriction of any kind.\n"
    "- 투자주의종목 (investment caution): a one-day market-attention notice. NO trading "
    "restriction of any kind — it is a lower, separate tier from 투자경고 and must never be "
    "reported as 'Investment Warning' or as if it were already escalated to one.\n"
    "- 투자경고종목 (investment warning): a genuinely stricter tier — new margin/credit "
    "purchases are banned and the deposit requirement (위탁증거금율) rises to 100%; can escalate "
    "to a trading halt if the price keeps surging.\n"
    "- 투자위험예고/투자위험종목 (investment risk warning/designation): same margin ban and 100% "
    "deposit requirement as 투자경고, PLUS a one-day trading halt at designation.\n"
    "- 관리종목 (administrative issue): margin/credit trading is banned and the shares cannot be "
    "used as collateral; this is a delisting-review signal, not a margin-only restriction.\n"
    "Do not use phrases like 'margin ban', 'forced deleveraging', 'credit prohibition', or "
    "'valuation cap' for 단기과열 or 투자주의 — neither tier imposes any such restriction, and "
    "inventing one from the label alone is a fabrication even if it sounds plausible.\n\n"
    "When a KRX industry classification is supplied below, describe the company's business "
    "per THAT classification; never relabel the company into a different industry inferred "
    "from headlines — a headline's subject matter, a partner's line of business, or a "
    "contract's end-market is not this company's own industry.\n\n"
    "The corporate-actions block below (if present) is a CASH DIVIDEND SCHEDULE only — not a "
    "filings or disclosure feed. The absence of an item in it is not evidence that a filing, "
    "contract, earnings release, or announcement does not exist, and it carries no fiscal-year "
    "information: never turn a dividend record's date into a statement about a company's fiscal "
    "year, annual results, or reporting period.\n\n"
    "Korean headlines write money amounts in 억 and 조. 1억 = 100 million KRW = 0.1 billion KRW; "
    "1조 = 1 trillion KRW. An amount a headline writes in 억 is a millions-scale figure unless it "
    "is at least 1,000억 — never carry its digits across into the word \"billion\" (6.6억원 is 660 "
    "million KRW, i.e. 0.66 billion KRW, NOT 6.6 billion KRW). When a KRW UNIT KEY block is "
    "supplied below the headlines, quote the converted value from it; where no key line covers an "
    "amount, state it in 억원 using the headline's own digits rather than guessing an English "
    "unit. If a disclosure headline states its own ratio — '매출액대비 N%', 'N% of revenue' — "
    "report that as the disclosure's stated figure (attributed to the disclosure), do not "
    "recompute a competing ratio of your own, and then sanity-check it: an amount A that a filing "
    "calls N% of revenue implies a revenue base of A / (N / 100); if that implied base is far "
    "from the exchange-reported revenue elsewhere in this cycle's material, say so and name both "
    "numbers rather than presenting the amount and the percentage together as if they agree."
) + IDENTITY_GROUNDING_NOTE


async def analyze(
    ticker: str,
    trade_date: str,
    as_of: date | None = None,
    company_name: str | None = None,
    language: str = DEFAULT_LANGUAGE,
    verified_snapshot: str | None = None,
    news_block_out: list[str] | None = None,
) -> str:
    """`company_name`, if given, grounds the prompt with the real company
    identity rather than leaving the model to guess one from a bare ticker
    symbol — see `market_analyst.run`'s docstring for the transcript
    evidence that motivated this.

    `verified_snapshot`, if given, is the SAME verified price snapshot
    `app.orchestrator.pipeline` hands to the Market Analyst, injected here as
    trusted (not `<external_data>`-wrapped, since it's our own vendor data
    like `get_macro_indicators`) so this analyst has something authoritative
    to check a news headline's own price claim against. Real incident this
    fixes (066570.KS, 2026-08-10): a headline citing an intraday "-2.05%"
    recap got repeated as the day's actual move, flatly contradicting the
    verified close (+1.3%) the Market Analyst had separately fetched — this
    analyst previously had no price tool at all, so nothing could catch it.
    Omitted (None) leaves this analyst's own prompt/behavior unchanged from
    before — a direct/test call, or a caller that genuinely has no snapshot
    to offer, still works exactly as it did.

    `news_block_out`, if given, is appended with the raw (unwrapped) KIS
    headline text — the pipeline uses it as `unit_scale_guard`'s soft
    ground-truth set so a 억원->billion carry-through of a headline amount
    (2026-09-07 audit #3, id=1237) can be flagged in this and every
    downstream node. Empty for a non-KRX ticker."""
    dispatch = bind_as_of(
        {
            "get_news": get_news,
            "get_macro_indicators": get_macro_indicators,
            "get_kr_macro_indicators": get_kr_macro_indicators,
            "get_overseas_market_indices": get_overseas_market_indices,
            "get_insider_txns": get_insider_txns,
        },
        as_of,
    )
    label = ticker_label(ticker, company_name)
    # yfinance's news is thin for Korean names, so for .KS tickers pre-fetch
    # KIS headlines and inject them. They are third-party free text, so they
    # go in WRAPPED as untrusted content (same defense the tool path applies to
    # get_news) — never as trusted instructions.
    kr_news_block = ""
    is_kr = ticker.endswith(".KS")
    if is_kr:
        # News headlines are third-party free text -> wrapped as untrusted.
        # The dividend/corporate-action calendar is structured KIS data ->
        # trusted, injected plainly.
        # Independent KIS calls -> gathered concurrently rather than paying
        # their combined latency serially (same shape as
        # fundamentals_analyst.analyze). get_kr_sector_name rides the SAME TR
        # class as a plain quote call (fetch_sector_name), so adding it here
        # costs no new rate-limit class and no extra round-trip.
        kr_news_raw, kr_disclosures, kr_sector_name = await asyncio.gather(
            get_kr_news(ticker, as_of=as_of),
            get_kr_disclosures(ticker, as_of=as_of),
            get_kr_sector_name(ticker, as_of=as_of),
        )
        if news_block_out is not None:
            news_block_out.append(kr_news_raw)
        kr_news_block = (
            wrap_untrusted("kis_korean_news", kr_news_raw) + "\n\n"
            + kr_disclosures + "\n\n"
        )
        # 2026-09-14, id=1407 (285800.KS, KRX classification 화학/Chemicals):
        # this analyst had no authoritative sector fact at all and invented a
        # business profile ("manufactures industrial robots and automation
        # equipment") from headlines its own report called largely irrelevant,
        # while the Fundamentals Analyst — which DOES get this line — correctly
        # said Chemicals in the same cycle. Same failure shape and same fix as
        # the 2026-09-09 P1 item (id=1354) that added the injection there; see
        # get_kr_sector_name's own docstring. "" on a backtest date or a fetch
        # failure — appended only when non-empty, never as a placeholder line.
        if kr_sector_name:
            kr_news_block += kr_sector_name + "\n\n"
        # A trusted, mechanically-derived KRW unit key for the 억/조 amounts
        # in the headlines above — appended AFTER the wrapped block as its own
        # trusted line so a headline cannot forge a look-alike (2026-09-07
        # audit #3: "6.6억원" reported as "₩6.6 billion").
        unit_key = kr_amount_unit_key(kr_news_raw)
        if unit_key:
            kr_news_block += unit_key + "\n\n"
        # get_kr_news above already carries the same KIS headlines the
        # get_news tool would fetch for a KRX ticker (VendorRouter tries KIS
        # first for .KS) -> drop the tool so the LLM can't double-fetch and
        # re-inject the identical headlines a second time.
        dispatch = {name: fn for name, fn in dispatch.items() if name != "get_news"}
    tools = [
        t
        for t in TOOLS
        if not (is_kr and t["function"]["name"] == "get_news")
        and not (not is_kr and t["function"]["name"] == "get_kr_macro_indicators")
        and not (is_kr and t["function"]["name"] == "get_overseas_market_indices")
    ]
    snapshot_block = (
        f"Verified price snapshot (authoritative):\n{verified_snapshot}\n\n" if verified_snapshot is not None else ""
    )
    user_prompt = (
        f"{snapshot_block}{kr_news_block}"
        f"Analyze news, macro backdrop, and insider activity for {label} as of {trade_date}."
    )
    return await run_tool_loop(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        tools=tools,
        dispatch=dispatch,
        agent_name="News + Macro Analyst",
        # get_news returns genuine free-text headlines/summaries from
        # external sources — the real injection surface, stays wrapped.
        # get_macro_indicators (FRED), get_kr_macro_indicators (KOSPI/KOSDAQ/
        # USD-KRW via KIS/yfinance), get_overseas_market_indices (Dow/S&P500/
        # Nasdaq via KIS), and get_insider_txns (structured SEC filing
        # records) are deterministic/structured, not third-party prose, so
        # they're trusted and left unwrapped.
        trusted_tools=frozenset({
            "get_macro_indicators", "get_kr_macro_indicators", "get_overseas_market_indices", "get_insider_txns",
        }),
        language=language,
    )
