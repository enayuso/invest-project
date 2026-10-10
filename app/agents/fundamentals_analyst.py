from __future__ import annotations

import asyncio
from datetime import date

from app.agents.tool_loop import IDENTITY_GROUNDING_NOTE, bind_as_of, run_tool_loop, ticker_label
from app.agents.numeric_guard import enforce_numeric_guards
from app.dataflows.kis_provider import get_kr_analyst_opinion, get_kr_financials, get_kr_sector_name
from app.llm.language import DEFAULT_LANGUAGE
from app.dataflows.yfinance import (
    get_analyst_consensus,
    get_balance_sheet,
    get_cashflow,
    get_earnings_estimates,
    get_fundamentals,
    get_institutional_ownership,
    get_short_interest,
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_fundamentals",
            "description": "Get key fundamental metrics (valuation, margins, growth, cash) for a ticker.",
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
            "name": "get_balance_sheet",
            "description": "Get the most recent balance sheet statement for a ticker.",
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
            "name": "get_cashflow",
            "description": "Get the most recent cashflow statement for a ticker.",
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
            "name": "get_analyst_consensus",
            "description": (
                "Get Wall Street analyst coverage for a ticker: recent upgrades/downgrades by firm, the "
                "recommendation trend (buy/hold/sell counts), and consensus price targets."
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
            "name": "get_institutional_ownership",
            "description": "Get top institutional holders for a ticker and the overall % held by institutions/insiders.",
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
            "name": "get_short_interest",
            "description": "Get short-interest stats for a ticker (shares short, short ratio, % of float).",
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
            "name": "get_earnings_estimates",
            "description": (
                "Get a ticker's earnings track record (estimate vs. actual EPS and surprise % by quarter) "
                "and forward consensus earnings estimates."
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
    "You are the Fundamentals Analyst on an equity research desk. Reason in plain "
    "prose over filings, financial statements, and Wall Street's own coverage to "
    "assess the company's financial health, valuation, growth trajectory, and how "
    "the market's positioned around it. Use the tools available to ground every "
    "claim in the actual reported figures — not just the company's own filings "
    "but whichever positioning and estimate signals were actually provided for "
    "THIS ticker, which differ by market and are spelled out below. A signal not "
    "named below for this ticker does not exist for it.\n\n"
    "Form your own view of the company's financial health from the underlying "
    "filings first — then compare it against what the analyst/positioning "
    "signals you actually received already imply. Treat consensus figures "
    "(price targets, recommendation counts) as one input to weigh, not a "
    "conclusion to defer to; explicitly flag it if your own read of the filings "
    "disagrees with consensus, rather than letting a well-known number "
    "substitute for your own analysis.\n\n"
    "Analyst coverage, institutional ownership, short interest, and earnings "
    "estimates are thin or missing for many tickers (small-caps, recent listings, "
    "low-coverage names) — if a tool returns no data or an 'unavailable' result, "
    "note it briefly and continue with the signals you do have; do not treat it "
    "as a blocking failure, and never invent a figure to fill the gap.\n\n"
    "A 100% (or near-100%) gross margin, or a reported cost-of-goods-sold of zero, is "
    "NOT by itself evidence of a 'software', 'SaaS', or 'purely intangible/digital "
    "product' business model — Korean exchange-reported income statements often bucket "
    "an entire non-manufacturing company's labor and service costs under SG&A rather "
    "than cost of goods sold, which mechanically produces a 100% gross margin for many "
    "ordinary service businesses (e.g. an advertising agency) that have real, non-trivial "
    "delivery costs. When a KRX industry classification is supplied below, describe the "
    "business per THAT classification; never relabel the company into a different "
    "industry (e.g. 'software firm', 'SaaS provider') just because its margin structure "
    "looks like one. If the margin figure still seems surprising given the stated "
    "industry, say so as a data-interpretation caveat, not as grounds to override the "
    "classification with your own guess."
)

# 2026-09-16 (engine audit follow-up, item 1): this used to be one paragraph
# inside the shared SYSTEM_PROMPT, unconditionally telling the model "Wall
# Street analyst ratings/price targets, institutional ownership, short
# interest, and the earnings beat/miss track record are real, independent
# signals worth weighing" — directly contradicted, for a .KS ticker, by the
# KR paragraph a few lines below it saying the generic tools have "thin,
# unreliable coverage" and to prefer the supplied domestic data instead. The
# tools' ACTUAL .KS behavior is more specific than either paragraph claimed:
# get_analyst_consensus isn't even offered (see `analyze`'s `is_kr` branch);
# get_institutional_ownership has no KIS equivalent at all (KRX carries no
# 13F-style filing) and comes back empty; get_short_interest and
# get_earnings_estimates DO return real KIS-sourced data for a .KS ticker —
# just not the same thing the US-market wording implies (KRX daily
# short-selling/program-flow, not a US short-interest ratio; domestic
# brokerage consensus estimates, not a US earnings-surprise track record).
# A blanket "these are all Korea-thin" instruction would have suppressed
# real, usable KRX signals — see this addendum's per-tool breakdown instead.
_KR_SYSTEM_PROMPT_ADDENDUM = (
    "For Korean (.KS) tickers, the generic fundamentals/earnings/consensus tools "
    "have thin, unreliable coverage. When exchange-reported Korean financial "
    "statements and domestic brokerage opinions are supplied to you at the top of "
    "the request, base the financial-health, growth, and valuation sections on "
    "THOSE figures (revenue, operating/net income, equity, ROE, EPS, BPS, debt "
    "ratio, growth, PER, PBR, current price, market cap) and read the analyst view "
    "from the supplied domestic opinions/target prices — not from the generic "
    "tools or your own memory. Every Korean figure is supplied with its "
    "billion-KRW and trillion-KRW equivalents already computed IN PARENTHESES "
    "right after it, e.g. \"3,053,729억원 (= 305,372.9 billion KRW = 30.54 "
    "trillion KRW)\" — when you write a KRW amount as \"billion\", copy ONLY "
    "that parenthetical number, never the raw 억원 digit run in front of it. "
    "1억원 is 0.1 billion KRW, NOT 1 billion KRW — treating the 억원 figure and "
    "its billion-KRW label as interchangeable (as if 억원 and billion were just "
    "two names for the same number) is a 10x error and the single most common "
    "mistake on this desk. Never convert a KRW figure yourself; always quote "
    "the supplied equivalent verbatim. "
    "If a supplied figure looks implausibly large or small for the company, "
    "say so as a data-quality caveat; do NOT conclude the data belongs to a "
    "different company."
    " Which of your tools carry Korean data for this ticker and which do not: "
    "get_fundamentals, get_balance_sheet and get_earnings_estimates are served from "
    "exchange-reported KIS data / domestic brokerage consensus and are usable. "
    "get_short_interest for a Korean ticker returns KRX daily short-selling intensity "
    "and program-trade net buying — a domestic supply-demand signal. It is NOT a "
    "US-style short-interest ratio, days-to-cover, or percent-of-float figure: describe "
    "it in those KRX terms and never restate it as a squeeze setup measured in days to "
    "cover. get_analyst_consensus is not offered to you for this ticker at all — the "
    "analyst view comes only from the domestic brokerage opinions supplied above. "
    "get_institutional_ownership has no Korean equivalent and is not offered to you "
    "for this ticker either: KRX has no 13F-style "
    "institutional-holdings filing, so nothing exists for this ticker. Do not report, "
    "estimate, or reason about Wall Street analyst coverage, a Wall Street consensus "
    "price target, or an institutional/insider ownership percentage for this ticker — "
    "no such data was provided, so any such figure would be fabricated. Simply omit "
    "those topics rather than writing a paragraph about their absence. get_cashflow is "
    "the one statement tool without a Korean source; if it comes back empty, say so in "
    "one clause and move on."
)

# The Wall-Street framing this used to state unconditionally for every
# ticker, restored here for non-Korean tickers only — see
# `_KR_SYSTEM_PROMPT_ADDENDUM`'s comment above for why a blanket version of
# this contradicted the KR paragraph for .KS names.
_NON_KR_SYSTEM_PROMPT_ADDENDUM = (
    "This ticker trades outside Korea, so the analyst-coverage, institutional-"
    "ownership, short-interest and earnings-estimate tools are the real US-market "
    "ones: analyst ratings/price targets, institutional ownership, short interest "
    "(shares short, short ratio, % of float) and the earnings beat/miss track record "
    "are all real, independent signals worth weighing alongside the filings "
    "themselves. Treat consensus figures (price targets, recommendation counts) as "
    "one input to weigh, not a conclusion to defer to; explicitly flag it if your own "
    "read of the filings disagrees with consensus, rather than letting a well-known "
    "number substitute for your own analysis. Do not reference KRX short-selling, "
    "program-trade flow, or any Korea-market-specific supply-demand signal for this "
    "ticker — no such tool or data was provided, so any such figure would be "
    "fabricated."
)


def _system_prompt(is_kr: bool) -> str:
    return (
        _BASE_SYSTEM_PROMPT
        + "\n\n"
        + (_KR_SYSTEM_PROMPT_ADDENDUM if is_kr else _NON_KR_SYSTEM_PROMPT_ADDENDUM)
        + IDENTITY_GROUNDING_NOTE
    )


_KR_REPORT_INSTRUCTION = (
    "Produce a fundamentals report — cover the company's own exchange-reported "
    "financials, and also weigh the supplied domestic brokerage opinions, the "
    "consensus earnings estimates, and the KRX short-selling / program-flow signal "
    "before concluding."
)
_NON_KR_REPORT_INSTRUCTION = (
    "Produce a fundamentals report — cover the company's own financials, but also "
    "check Wall Street's analyst coverage, institutional ownership, short interest, "
    "and earnings track record before concluding."
)


async def analyze(
    ticker: str,
    trade_date: str,
    as_of: date | None = None,
    company_name: str | None = None,
    language: str = DEFAULT_LANGUAGE,
    unit_violations_out: list[str] | None = None,
    unit_scans_out: dict[str, dict[str, int]] | None = None,
    kr_ground_truth_out: list[str] | None = None,
    period_note_out: list[str] | None = None,
    period_violations_out: list[str] | None = None,
    recency_out: dict | None = None,
) -> str:
    """`company_name`, if given, grounds the prompt with the real company
    identity (see `ticker_label`'s docstring) — this analyst was the one
    holdout that never received it, on the assumption that its own tools
    (which return the real name via yfinance's longName/KIS's own data)
    made it immune. Confirmed wrong live (2026-07-24, backtest for
    000660.KS/SK하이닉스): correct KIS financial data reached the model, but
    it twice wrote up the report under a completely different real
    company's name ("에코프로비엠") across two separate historical dates —
    the underlying figures were right, only the label the model itself
    chose to write down was wrong. Grounding it here matches the other
    three analysts (Market/News+Macro/Sentiment), all of which already
    take this same parameter for the same reason.

    `unit_violations_out`, if given, is extended with a description of every
    억원-to-billion/trillion 10x mislabel `app.agents.unit_scale_guard` found
    and corrected in the report (2026-08-28 domestic AI audit), AND every
    debt-ratio denominator mislabel `app.agents.ratio_label_guard` found and
    corrected (2026-09-01 domestic AI audit, item #4 — `debt_ratio_denominator`
    entries) — an out-param, not a return-value change, so callers that don't
    care about the report text's degradation history (existing callers,
    tests) don't have to unpack a new tuple shape. Mirrors how
    `app.orchestrator.pipeline` already threads `identity_violations`
    alongside each analyst's report.

    `unit_scans_out`, if given, gets `{"Fundamentals Analyst": {...}}` with
    `app.agents.unit_scale_guard`'s positive scan counts (examined / ground
    truth / corrected / flagged / unverified) — populated even on a clean
    report, so `unit_scale_violations=[]` can be distinguished from a regex
    blind spot (2026-09-01, decision id=1158/기아: nine 10x figures, zero
    caught, empty list read as clean).

    `kr_ground_truth_out`, if given, is appended with the assembled `kr_block`
    (the exact Korean financial text handed to the model) as soon as it is
    built — BEFORE the LLM call, so it is populated even if this analyst
    fails. `app.orchestrator.pipeline` threads it to every other node's
    `app.agents.numeric_guard` call so a magnitude error anywhere in the
    pipeline is checked against the same 억원 figures, without a second
    `get_kr_financials` fetch (that would double KIS ledger load — see
    commit a06c279).

    `period_note_out`, if given, gets `get_kr_financials`'s interim-period
    warning when the most recent Korean financial row is a leading interim
    period, and/or its FINANCIAL-DATA RECENCY warning when the newest row is
    a full fiscal year or more stale (2026-09-07 audit #6, id=1237) — the
    pipeline threads it downstream (2026-09-01 audit #5, decision id=1158: a
    period mismatch flipped the sign of the trend and every agent carried it
    forward). Empty when the newest row is a fresh full year.

    `recency_out`, if given, is passed straight to `get_kr_financials` and
    `.update()`d with `{latest_period, latest_label, months_stale, interim}`
    (recorded on `PipelineState.kr_financials_recency`)."""
    is_kr = ticker.endswith(".KS")
    tools_map = {
        "get_fundamentals": get_fundamentals,
        "get_balance_sheet": get_balance_sheet,
        "get_cashflow": get_cashflow,
        "get_short_interest": get_short_interest,
        "get_earnings_estimates": get_earnings_estimates,
    }
    if not is_kr:
        # get_analyst_consensus is the one tool this analyst's own system
        # prompt already tells the model NOT to trust for Korean names (use
        # the supplied domestic opinions instead, see kr_block below) — so
        # for .KS tickers it's dropped from the list entirely rather than
        # left there to be offered and (correctly) ignored, same "don't
        # offer what this market can't use" pattern news_macro_analyst.py
        # applies to get_news and market_analyst.py now applies to
        # get_investor_flow.
        tools_map["get_analyst_consensus"] = get_analyst_consensus
        # 2026-09-16: get_institutional_ownership was left in the .KS list by
        # oversight when the KR addendum was written — that addendum states
        # outright that KRX has no 13F-style institutional-holdings filing
        # and forbids reporting an ownership percentage, while the router
        # registers this category as yfinance-only/optional (KIS has no
        # equivalent TR). Offering it for a Korean name was a guaranteed-
        # empty round-trip against the model's own instructions — same
        # is_kr gate as get_analyst_consensus above.
        tools_map["get_institutional_ownership"] = get_institutional_ownership
    dispatch = bind_as_of(tools_map, as_of)
    tools = [t for t in TOOLS if t["function"]["name"] in tools_map]
    # For Korean tickers, KIS's exchange-reported financials are far richer than
    # yfinance's — but the model won't reliably *choose* a KIS tool from a
    # crowded tool list, so instead of offering it as one more tool we fetch it
    # up front and inject it into the request. That guarantees the data reaches
    # the analysis regardless of the model's tool-selection behavior.
    kr_block = ""
    if is_kr:
        # Independent KIS calls — run concurrently instead of paying their
        # combined latency twice.
        kr_financials, kr_analyst_opinion, kr_sector_name = await asyncio.gather(
            get_kr_financials(
                ticker, as_of=as_of, period_note_out=period_note_out, recency_out=recency_out,
            ),
            get_kr_analyst_opinion(ticker, as_of=as_of),
            get_kr_sector_name(ticker, as_of=as_of),
        )
        kr_block = (
            "Exchange-reported Korean financial statements (authoritative, from KIS — "
            "use these for this company's financials):\n"
            f"{kr_financials}\n\n"
            "Domestic brokerage investment opinions (from KIS — use these instead of "
            "the near-empty generic analyst consensus for this Korean name):\n"
            f"{kr_analyst_opinion}\n\n"
        )
        # 2026-09-09, P1 audit item #3 (id=1354): the one authoritative fact
        # this analyst previously had zero access to — see get_kr_sector_name's
        # own docstring for the incident (a 100% gross-margin artifact read as
        # "software/SaaS" business model for an actual advertising agency).
        # "" on a backtest date or a fetch failure — appended only when present.
        if kr_sector_name:
            kr_block += f"{kr_sector_name}\n\n"
    if kr_block and kr_ground_truth_out is not None:
        # Recorded before the LLM call so it survives an analyst failure —
        # the pipeline threads it to every other node's numeric_guard.
        kr_ground_truth_out.append(kr_block)
    label = ticker_label(ticker, company_name)
    user_prompt = (
        f"{kr_block}"
        f"Analyze {label}'s fundamentals as of {trade_date}. "
        + (_KR_REPORT_INSTRUCTION if is_kr else _NON_KR_REPORT_INSTRUCTION)
    )
    report = await run_tool_loop(
        system_prompt=_system_prompt(is_kr),
        user_prompt=user_prompt,
        tools=tools,
        dispatch=dispatch,
        agent_name="Fundamentals Analyst",
        # All of this analyst's tools return structured, deterministic
        # vendor data (financial statements, consensus ratings/targets,
        # ownership/short-interest figures) — not free-form third-party
        # prose — so none need the "possibly-manipulated" framing.
        trusted_tools=frozenset({
            "get_fundamentals", "get_balance_sheet", "get_cashflow", "get_analyst_consensus",
            "get_institutional_ownership", "get_short_interest", "get_earnings_estimates",
        }),
        language=language,
    )
    # Only the KR path has any supplied 억원 figure to check the report
    # against (kr_block empty otherwise) — see unit_scale_guard's own
    # docstring for why this is Fundamentals-Analyst-only (the only analyst
    # that receives this data at all).
    if is_kr and kr_block:
        # No `soft_eok_block=` here on purpose: this analyst is never shown the
        # news-headline block (news_macro_analyst owns it), so a digit match
        # between this report and a headline amount would have no provenance —
        # the same "judged against ground truth it never received" mistake the
        # #211 blind-spot false positive was. Its figures are also the ones the
        # hard-GT pass explains best. `soft_ground_truth_values: 0` here is
        # correct, not a gap (2026-09-09 Fix 3 verification).
        report, violations = enforce_numeric_guards(
            report, node="Fundamentals Analyst", kr_block=kr_block, language=language,
            usd_check=True, currency_check=True, supplied_kr_block=True, scans_out=unit_scans_out,
            period_out=period_violations_out,
        )
        if violations and unit_violations_out is not None:
            unit_violations_out.extend(violations)
    return report
