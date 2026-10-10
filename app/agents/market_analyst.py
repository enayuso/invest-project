from __future__ import annotations

import asyncio
import logging
import re
from datetime import date

from app.agents.tool_loop import IDENTITY_GROUNDING_NOTE, bind_as_of, run_tool_loop, ticker_label
from app.dataflows.kis_provider import (
    get_investor_flow,
    get_kr_market_regime,
    get_kr_sector_breadth,
    get_kr_vi_status,
)
from app.llm.language import DEFAULT_LANGUAGE
from app.dataflows.yfinance import get_indicators, get_long_horizon_context, get_stock_data, get_verified_snapshot

logger = logging.getLogger(__name__)

# 2026-09-16: `get_indicators`/`get_investor_flow` used to also be offered
# here as re-callable tools, alongside being unconditionally injected into
# the prompt below (see `analyze`'s indicator_block/flow_block). Re-calling
# either returns the exact same deterministic figures already in the
# prompt — zero new information for the extra tool-call round-trip's worth
# of tokens — so both were dropped from the tool list (they're still
# imported and called unconditionally in `analyze`, just not re-offered as
# tools a model can invoke again mid-analysis). `get_stock_data`'s
# `lookback_days` param is gone too: it never actually changed what was
# rendered (always the last 10 sessions — see `_yf_stock_data`'s own note),
# so it was a parameter the model could set with no effect. Real multi-week
# context now comes from the unconditionally-injected long-horizon block
# instead (see `analyze`'s `long_horizon` block, `long_horizon.py`).
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_stock_data",
            "description": "Get the last 10 daily OHLCV sessions for a ticker.",
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
            "name": "get_verified_snapshot",
            "description": (
                "Get a deterministic, verified price snapshot for a ticker. Always "
                "call this before stating any exact price figure. During market hours "
                "it returns the still-forming session instead of a completed daily "
                "bar; the returned text says explicitly which one it is."
            ),
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
            "name": "get_kr_vi_status",
            "description": (
                "Check whether a KOREAN (.KS) ticker is currently under a VI "
                "(변동성완화장치, volatility interruption) halt — if so, an order "
                "placed against it may not fill even if the underlying signal is "
                "sound. Korea-only, live snapshot only (not available for a "
                "backtest date)."
            ),
            "parameters": {
                "type": "object",
                "properties": {"ticker": {"type": "string"}},
                "required": ["ticker"],
            },
        },
    },
]

_BASE_SYSTEM_PROMPT = (
    "You are the Market Analyst on an equity research desk. Reason over price "
    "action and technical indicators to produce a clear, evidence-based technical "
    "read on the stock. A verified price snapshot is provided below — that figure "
    "is authoritative; use it (not get_stock_data's raw series or your own memory) "
    "for any exact price you state in your report. "
    "The snapshot states whether the session it covers is COMPLETE or STILL IN "
    "PROGRESS: a completed session reports close=/high=/low=/volume=, while an "
    "open session reports last=/session_high=/session_low=/volume_so_far= and is "
    "marked SESSION IN PROGRESS. When it is in progress, treat those figures as a "
    "mid-session snapshot: say the stock is trading at that level intraday, never "
    "that the day closed, finished, settled, or 'concluded trading' there; never "
    "call the partial range a daily or closing range; and never read the partial "
    "volume as a full day's volume or as evidence of unusually weak participation. "
    "Use get_stock_data and the verified technical-indicator block below to "
    "support your analysis. "
    "Write a prose report; do not just list raw numbers."
    " A verified technical-indicator block is provided below and is likewise "
    "authoritative: it carries the only SMA(20), SMA(50), RSI(14) and MACD "
    "values that exist for this ticker. State a numeric value for a technical "
    "indicator ONLY if that exact value appears in that block. Never estimate, "
    "assume, approximate, or infer an indicator value, an indicator range, or a "
    "moving-average crossover from price action, and never present one hedged "
    "with 'estimated', 'likely', 'approximately', 'assuming' or '추정' — a hedge "
    "word does not make an invented number safe, and downstream stages read it "
    "as a real signal. If the block says NOT AVAILABLE, say plainly that no "
    "indicator values were available this cycle and reason from price action "
    "alone. Conventional thresholds are fixed: overbought is RSI above 70, "
    "oversold is RSI below 30; a value between them is neutral and must not be "
    "described as approaching either extreme."
    " A longer-horizon price-context block is provided below, built from "
    "weekly bars covering up to the last 52 weeks. It is the ONLY multi-week "
    "history you have — the daily OHLCV table from get_stock_data covers "
    "just the last 10 sessions and cannot support any statement about a "
    "multi-week or multi-month move. Use the long-horizon block for the "
    "stated price range, the distance from that range's high or low, and "
    "the 1/3/6/12-month trend, and quote its figures as given. The block "
    "names its own window: it says '52-week' only when a full year of "
    "weekly bars exists, and otherwise a shorter label such as '36-week' "
    "plus an explicit note. Use exactly the label the block uses and never "
    "restate a shorter window as a 52-week high, low or range. Do not infer "
    "a 52-week extreme, a multi-month trend, a support or resistance level, or a "
    "moving-average crossover history from the 10-session daily table, and "
    "do not state such a figure hedged with 'approximately', 'roughly', "
    "'likely' or '추정'. If the block says NOT AVAILABLE, say plainly that no "
    "longer-horizon price context was available this cycle and confine your "
    "read to the last 10 sessions and the verified indicator values."
)

# Only ever appended for a .KS ticker (see `analyze` below) — a static prompt
# that always mentioned "for Korean tickers, call get_investor_flow" regardless
# of market let the model pattern-match into writing a fabricated "foreign/
# institutional net-buy flow" section for a non-KR ticker even though
# get_investor_flow/get_kr_vi_status were never in its tool list for that
# ticker (confirmed live: WDC, a NASDAQ name, got an invented "한국 시장(.KS)
# 특유의 수급" paragraph — 2026-08-03/04). Splitting the instruction out so it
# only exists in the prompt when the tool actually exists in `tools` closes
# that gap; `_NON_KR_SYSTEM_PROMPT_ADDENDUM` below closes the other half (an
# explicit "don't" for the market where the instruction is absent instead of
# just silently missing).
_KR_SYSTEM_PROMPT_ADDENDUM = (
    " A verified investor-flow (수급) block is provided below and is likewise "
    "authoritative — it carries the only individual/foreign/institution "
    "net-buy figures that exist for this ticker. State a specific flow figure "
    "ONLY if it appears in that block; never estimate, guess, or infer a flow "
    "direction or number, and never present one hedged with 'estimated', "
    "'likely', 'est.', 'approximately' or '추정' — a hedge word does not make "
    "an invented figure safe, and downstream stages read it as real. If the "
    "block says NOT AVAILABLE, say plainly that no flow data was available "
    "this cycle and do not otherwise speculate about foreign/institutional "
    "positioning."
    " Also call get_kr_vi_status to check for an active VI halt, "
    "and note in your report if one is in effect, since it affects whether an "
    "order can fill right now."
    " The Korean sector-breadth block in the user message is a WHOLE-MARKET "
    "ranking of KRX industry indices, not a statement about this stock. Never "
    "write that this stock belongs to, operates in, is a player in, is "
    "represented by, or is part of a sector merely because that sector appears "
    "in that ranking, and never present a ranked sector's gain as a tailwind "
    "for this stock on that basis alone — with no ticker-to-sector mapping "
    "behind it, such a claim is fabricated exactly like an invented net-buy "
    "figure. If the block contains a line naming this stock's own KRX industry "
    "sector, that line is the ONLY authoritative sector for this stock: use it, "
    "and if it says the sector is unknown or absent from the ranking, say the "
    "sector linkage is unknown instead of inferring one. You may still describe "
    "what the company does in your own words from the verified company name and "
    "any fundamentals given to you — a business description is not a "
    "sector-index claim; what is forbidden is attaching the company to a sector "
    "because of where that sector sits in today's ranking."
    " The KOSPI/KOSDAQ figures in the market-regime line are those INDICES' OWN "
    "day changes. They are not this stock's contribution to, weight in, or "
    "impact on the index. Never restate an index's move as a contribution by "
    "this stock — a single stock's index contribution is not supplied to you "
    "and cannot be derived from anything you have been given."
)

# Non-KR tickers never get get_investor_flow/get_kr_vi_status in their tools
# list (see `analyze`'s `is_kr` branch) — this explicit negative instruction
# exists because omitting KR-specific guidance silently wasn't enough on its
# own to stop the model from writing Korean-market boilerplate anyway (see
# `_KR_SYSTEM_PROMPT_ADDENDUM`'s docstring note above for the live case that
# motivated this).
_NON_KR_SYSTEM_PROMPT_ADDENDUM = (
    " This ticker does not trade on a Korean exchange. Do not reference "
    "foreign/institutional KRX net-buy flow, a KRX VI (volatility interruption) "
    "halt, or any other Korea-market-specific supply-demand signal for it — "
    "no such tool or data was provided for this ticker, so any such figure "
    "would be fabricated."
)


async def analyze(
    ticker: str,
    trade_date: str,
    as_of: date | None = None,
    company_name: str | None = None,
    language: str = DEFAULT_LANGUAGE,
    verified_snapshot: str | None = None,
    indicator_block_out: list[str] | None = None,
    flow_block_out: list[str] | None = None,
    long_horizon_block_out: list[str] | None = None,
) -> str:
    """`as_of` bounds every tool call to point-in-time data for backtesting
    (Phase 2). It's bound into the dispatch closures below, never exposed to
    the LLM as a tool parameter, so the model can't request future data.

    `company_name`, if given, is included alongside the bare ticker so the
    model has no room to guess a company identity from the symbol alone —
    this bit non-US tickers badly (e.g. a live transcript for 373220.KS,
    LG Energy Solution, opened "Technical Analysis Report: 373220.KS
    (SK Hynix Inc.)", a hallucinated identity that then propagated through
    the rest of that decision's reasoning).

    `verified_snapshot`, if given, is used as-is instead of this function
    fetching its own — `app.orchestrator.pipeline` hoists the fetch above
    its analyst gather so News + Macro Analyst can be handed the SAME
    verified figure (see that module's own docstring for the incident this
    fixes: a stale/wrong price claim sitting inside a news headline had
    nothing to be checked against). Fetched internally when omitted (e.g. a
    direct/test call) so this stays independently usable.

    Either way, the snapshot is injected into the prompt (same "don't rely
    on the model choosing to call the right tool" fix `fundamentals_analyst.
    py`'s `kr_block` already applies for Korean financials) rather than left
    as a tool the model might skip — a prior version only told the model to
    call it in the system prompt, with nothing enforcing that it actually
    did before stating a price. Still offered as a tool too, in case the
    model wants to re-check mid-analysis.

    `indicator_block_out`, if given, is appended with the raw technical-
    indicator text this call injected into the prompt (2026-09-07 fix, live
    audit: 35.7% of domestic Market Analyst reports had NO real indicator
    value at all — the model simply skipped calling get_indicators, not a
    data-availability gap — vs 13.7% for the same agent on overseas tickers).
    That text is either the real `get_indicators` output or the "NOT
    AVAILABLE" fallback below; either way it is what `app.agents.
    technical_claim_guard.scan_technical_claims` checks this report's claims
    against, mirroring `period_mismatch_guard.scan_period_mismatch`'s
    `strong_out=`/`scan_out=` out-param idiom.

    `flow_block_out`, if given, is appended with the raw investor-flow (수급)
    text this call injected into the prompt (2026-09-11 fix — the investor-
    flow analogue of the 2026-09-07 `indicator_block_out` fix above: a live
    audit found `get_investor_flow` skipped or effectively unused the same
    way, e.g. id=1389/320000.KS where the Market Analyst's own report table
    listed "Foreign Flow | Net Seller (-2,000 est.)" with an explicit "est."
    tag and no real tool call backing it, which downstream debate then
    treated as confirmed fact). That text is either the real
    `get_investor_flow` output or the "NOT AVAILABLE" fallback below; either
    way it is what a future guard could check this report's flow claims
    against, though no such guard is added by this change — see the KR
    system-prompt addendum for the anti-hedging instruction that goes with
    it.

    `long_horizon_block_out`, if given, is appended with the raw 52-week
    price-context text this call injected into the prompt (2026-09-16 fix —
    see `long_horizon.py`'s module docstring: the daily OHLCV table only
    ever covered the last 10 sessions despite the tool schema advertising
    180 days, so the model was expected to describe a 52-week range or a
    multi-month trend with no data behind it). Same NOT AVAILABLE fallback
    idiom as `indicator_block_out`/`flow_block_out` above."""
    is_kr = ticker.endswith(".KS")
    tools_map = {
        "get_stock_data": get_stock_data,
        "get_verified_snapshot": get_verified_snapshot,
    }
    if is_kr:
        # Korea-only signal — the tool's own description says it returns a
        # not-supported note for US tickers, so offering it there just risks
        # a wasted round-trip. Same "drop what this market can't use" pattern
        # news_macro_analyst.py already applies to get_news for KR tickers.
        tools_map["get_kr_vi_status"] = get_kr_vi_status
    dispatch = bind_as_of(tools_map, as_of)
    tools = [t for t in TOOLS if t["function"]["name"] in tools_map]
    label = ticker_label(ticker, company_name)
    # For Korean tickers, prepend the market-wide regime (KOSPI/KOSDAQ) and
    # sector breadth so the technical read is framed against whether the
    # whole market/sector is up or crashing — context the per-stock tools
    # don't carry. The regime line is whole-market; the breadth ranking is
    # whole-market too, but `get_kr_sector_breadth(ticker=...)` now also
    # resolves and injects THIS stock's own KRX industry sector via the price
    # TR's `bstp_kor_isnm` (fetch_sector_name) — that injected line is the
    # only authoritative per-stock sector fact, the ranking itself still just
    # answers "broad rotation or idiosyncratic".
    regime = ""
    sector_breadth = ""
    if is_kr:
        # Independent calls (neither's result feeds the other) — run
        # concurrently instead of paying their combined latency twice.
        market_regime, sector_breadth = await asyncio.gather(
            get_kr_market_regime(as_of=as_of), get_kr_sector_breadth(as_of=as_of, ticker=ticker)
        )
        regime = f"{market_regime}\n\n{sector_breadth}\n\n"
    snapshot = verified_snapshot if verified_snapshot is not None else await get_verified_snapshot(ticker, as_of=as_of)
    snapshot_block = f"Verified price snapshot (authoritative):\n{snapshot}\n\n"
    # 2026-09-07 fix: previously get_indicators was offered only as a tool the
    # model could optionally invoke — live audit found it skipped 35.7% of the
    # time on domestic tickers (vs 13.7% overseas, same agent, pure
    # non-determinism), and ~2.4% of those skips got papered over with a
    # hedged, fabricated value ("Estimated RSI: ~35-40") instead of an honest
    # "unavailable" note. Injecting the real value unconditionally (same "don't
    # rely on the model choosing to call the right tool" fix as the verified
    # snapshot above) closes the gap outright. `_kis_indicators` legitimately
    # raises on thin history (new listing, thin ETF, <51 candles) or a vendor
    # error — caught locally here, not hoisted into pipeline.py's analyst
    # gather, so a data problem for this one ticker degrades only this one
    # indicator block instead of risking the whole gather (which already runs
    # with return_exceptions=True, but this keeps the safe degradation point
    # local regardless).
    # 2026-09-16 fix: these three fetches are mutually independent vendor
    # calls (technical indicators, long-horizon weekly bars, KR investor
    # flow) that used to be awaited one after another, paying three
    # round-trips of latency per ticker. Gathered with return_exceptions=True
    # so each one still degrades LOCALLY to its own NOT AVAILABLE text
    # exactly as before — a thin-history or vendor error for one block must
    # not fail the analyst or suppress the other two. The KR regime/sector-
    # breadth gather above is deliberately left separate: folding it in here
    # would put five simultaneous KIS requests behind every ticker, and the
    # pipeline already runs four analysts concurrently. The prompt is
    # assembled from these values below in a fixed order, so nothing about
    # what the model sees changes.
    _fetches = [get_indicators(ticker, as_of=as_of), get_long_horizon_context(ticker, as_of=as_of)]
    if is_kr:
        _fetches.append(get_investor_flow(ticker, as_of=as_of))
    _results = await asyncio.gather(*_fetches, return_exceptions=True)
    indicators, long_horizon = _results[0], _results[1]
    flow = _results[2] if is_kr else None

    # 2026-09-07 fix: previously get_indicators was offered only as a tool the
    # model could optionally invoke — live audit found it skipped 35.7% of the
    # time on domestic tickers (vs 13.7% overseas, same agent, pure
    # non-determinism), and ~2.4% of those skips got papered over with a
    # hedged, fabricated value ("Estimated RSI: ~35-40") instead of an honest
    # "unavailable" note. Injecting the real value unconditionally (same "don't
    # rely on the model choosing to call the right tool" fix as the verified
    # snapshot above) closes the gap outright. `_kis_indicators` legitimately
    # raises on thin history (new listing, thin ETF, <51 candles) or a vendor
    # error — caught locally here, not hoisted into pipeline.py's analyst
    # gather, so a data problem for this one ticker degrades only this one
    # indicator block instead of risking the whole gather (which already runs
    # with return_exceptions=True, but this keeps the safe degradation point
    # local regardless).
    if isinstance(indicators, BaseException):
        logger.warning("Market Analyst: indicators unavailable for %s", ticker, exc_info=indicators)
        indicators = (
            "NOT AVAILABLE this cycle (insufficient price history or a vendor error). "
            "No RSI, moving-average or MACD value exists for this ticker right now."
        )
    if indicator_block_out is not None:
        indicator_block_out.append(indicators)
    indicator_block = f"Verified technical indicators (authoritative):\n{indicators}\n\n"
    # get_stock_data's tool schema advertised `lookback_days: default 180`
    # but only ever rendered the last 10 daily sessions regardless (see
    # `_yf_stock_data`'s own note) — the model was expected to write
    # "52-week high", "3-month trend" style prose with zero supporting data.
    # Fetched unconditionally (same "don't rely on the model choosing to
    # call the right tool" fix as the indicator/flow blocks above) from
    # weekly bars rather than widening the daily table, which stays fixed at
    # 10 sessions (see `long_horizon.py`'s module docstring for why: the
    # in-flight candle-pattern A/B and `session_close`-style guards are
    # keyed to that table's exact shape). Raises on too little history (new
    # listing, <12 weekly bars) or a vendor error — caught locally, same
    # local-degradation pattern as the indicator/flow blocks.
    if isinstance(long_horizon, BaseException):
        logger.warning("Market Analyst: long-horizon context unavailable for %s", ticker, exc_info=long_horizon)
        long_horizon = (
            "NOT AVAILABLE this cycle (insufficient price history or a vendor error). "
            "No 52-week high/low or multi-month return exists for this ticker right now."
        )
    if long_horizon_block_out is not None:
        long_horizon_block_out.append(long_horizon)
    long_horizon_block = f"{long_horizon}\n\n"
    # 2026-09-11 fix: the investor-flow analogue of the 2026-09-07 indicator
    # fix above — get_investor_flow was previously offered only as an
    # optional tool the model could choose to call. A live audit (id=1389,
    # 320000.KS) found the model instead writing a fabricated, hedged flow
    # figure ("Net Seller (-2,000 est.)") into its report with no real tool
    # call backing it, which downstream debate then treated as confirmed
    # fact. Fetched unconditionally here (KR tickers only — same
    # Korea-only tool as before) and injected into the prompt, same
    # "don't rely on the model choosing to call the right tool" fix as the
    # indicator block. `get_investor_flow` itself already degrades to a
    # Korean-language "KIS not configured" string rather than raising, but
    # does not guard the underlying `broker.fetch_investor_trend()` call, so
    # a vendor/network error there is still caught here, exactly like
    # `get_indicators` above.
    flow_block = ""
    if is_kr:
        if isinstance(flow, BaseException):
            logger.warning("Market Analyst: investor flow unavailable for %s", ticker, exc_info=flow)
            flow = (
                "NOT AVAILABLE this cycle (KIS not configured, or a vendor error). "
                "No investor-flow (수급) data exists for this ticker right now."
            )
        if flow_block_out is not None:
            flow_block_out.append(flow)
        flow_block = f"Verified investor flow (authoritative, KRX only):\n{flow}\n\n"
    user_prompt = (
        f"{regime}{snapshot_block}{indicator_block}{long_horizon_block}{flow_block}"
        f"Analyze {label} as of {trade_date}. Produce a technical analysis report."
    )
    market_addendum = _KR_SYSTEM_PROMPT_ADDENDUM if is_kr else _NON_KR_SYSTEM_PROMPT_ADDENDUM
    system_prompt = _BASE_SYSTEM_PROMPT + market_addendum + IDENTITY_GROUNDING_NOTE
    report = await run_tool_loop(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        tools=tools,
        dispatch=dispatch,
        agent_name="Market Analyst",
        # All of this analyst's tools return deterministic vendor price/
        # technical data — not free-form third-party text, so none are
        # wrapped as "possibly-manipulated" content, which would contradict
        # the prompt's instruction to treat them as ground truth.
        trusted_tools=frozenset({"get_stock_data", "get_verified_snapshot", "get_kr_vi_status"}),
        language=language,
    )
    if is_kr:
        _reaudit_sector_attribution(ticker, company_name, report, sector_breadth)
    return report


# The 4 strict attribution templates that scored ~80% precision in the
# 2026-09 audit — subject must be the analyzed company/ticker, object must
# end in the literal word "sector". Deliberately narrow: a miss is fine, a
# stray warning is just log noise but should stay rare.
_ATTR_CONNECTIVES = (
    "operates in", "operates within", "is a player in", "a leading player in the",
    "is part of the", "represented by companies like",
)
# KRX Korean industry name -> English keywords an LLM report would use.
# Only needs the sectors an attribution claim actually lands on; unmapped
# sectors simply never trigger a warning (conservative by construction).
_SECTOR_EN_KEYWORDS = {
    "보험": ("insurance",),
    "증권": ("securities", "brokerage"),
    "금융": ("financial", "finance", "banking"),
    "은행": ("bank",),
    "건설": ("construction", "builder"),
    "화학": ("chemical",),
    "제약": ("pharmaceutical", "pharma"),
    "전기·전자": ("electronics", "electrical", "semiconductor", "chip"),
    "기계·장비": ("machinery", "equipment"),
    "운송장비·부품": ("auto", "automotive", "vehicle", "transport equipment"),
    "유통": ("retail", "distribution"),
    "통신": ("telecom", "telecommunication"),
    "전기·가스": ("utility", "utilities", "power", "gas"),
    "운송·창고": ("logistics", "shipping", "warehousing"),
    "섬유·의류": ("textile", "apparel", "clothing"),
    "음식료·담배": ("food", "beverage", "tobacco"),
    "종이·목재": ("paper", "timber", "wood"),
    "금속": ("metal", "steel"),
    "비금속": ("non-metal", "cement"),
    "의료·정밀기기": ("medical device", "precision instrument"),
    "일반서비스": ("services",),
    "IT 서비스": ("it service", "software"),
    "오락·문화": ("entertainment", "media", "leisure"),
    "부동산": ("real estate", "property"),
    "기타제조": ("manufacturing",),
}


def _reaudit_sector_attribution(
    ticker: str, company_name: str | None, report: str, sector_breadth: str
) -> None:
    """Log-only (no text/state/banner change): warn when the report makes a
    strict sector-attribution claim about the analyzed stock, the claimed
    sector is in today's whole-market breadth ranking, and it differs from
    the stock's own sector line (`fetch_sector_name`). Gated on the own-sector
    line having resolved — an undetermined line means we can't judge."""
    m = re.search(
        r"This stock's own KRX industry sector: ([^(.\n]+?) \(", sector_breadth
    )
    if not m:
        return
    real_sector = m.group(1).strip()
    ranking = "\n".join(
        ln for ln in sector_breadth.splitlines() if ln.startswith(("Top gaining:", "Top losing:"))
    )
    report_l = report.lower()
    subjects = [s for s in (company_name, ticker, ticker.split(".")[0]) if s]
    if not any(s.lower() in report_l for s in subjects):
        return
    if not any(c in report_l for c in _ATTR_CONNECTIVES):
        return
    # Needs an "<something> sector" attribution phrase to exist at all.
    if not re.search(r"[a-z/ ]+ sector\b", report_l):
        return
    for kr_name, keywords in _SECTOR_EN_KEYWORDS.items():
        if kr_name == real_sector or kr_name not in ranking:
            continue
        if any(f"{kw} sector" in report_l for kw in keywords):
            logger.warning(
                "Market Analyst sector re-audit: %s report attributes it to the '%s' sector "
                "(in today's breadth ranking) but its own KRX industry sector is '%s'",
                ticker, kr_name, real_sector,
            )
            return
