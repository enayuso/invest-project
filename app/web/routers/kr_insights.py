"""Korean-market analysis data (supply-demand, financials, analyst opinion,
risk signals, news) as structured tables for the dashboard panel."""

from __future__ import annotations

import logging
from datetime import timedelta

from fastapi import APIRouter, HTTPException

from app.clock import get_clock
from app.execution.kis_bridge import route_ticker
from app.web import state

logger = logging.getLogger("app.web")

router = APIRouter()

_KR_INSIGHT_KINDS = frozenset({"investor-flow", "financials", "opinion", "risk", "news"})


def _table(columns, signed, rows):
    return {"type": "table", "columns": columns, "signed": signed, "rows": rows}


async def _kr_insight_data(kind: str, code: str) -> dict:
    b = await state.kis_broker()
    if kind == "investor-flow":
        flows = (await b.fetch_investor_trend(code))[:12]
        est = None
        try:
            e = await b.fetch_investor_estimate(code)
            est = e[-1] if e else None
        except Exception:
            pass
        note = (f"당일 추정 — 외국인 {est.foreign_net:+,} · 기관 {est.institution_net:+,}"
                if est else "")
        return {**_table(
            ["일자", "종가", "개인", "외국인", "기관"], [2, 3, 4],
            [[f.day, f"{f.close:,.0f}", f"{f.individual_net:+,}", f"{f.foreign_net:+,}",
              f"{f.institution_net:+,}"] for f in flows],
        ), "caption": "투자자별 순매수 (주, +매수/−매도)", "note": note}

    if kind == "financials":
        inc = (await b.fetch_kr_income_statement(code))[:4]
        bal = (await b.fetch_kr_balance_sheet(code))[:4]
        rat = (await b.fetch_kr_financial_ratio(code))[:4]
        sections = []
        if inc:
            sections.append({"title": "손익계산서 (억원)", **_table(
                ["결산", "매출액", "영업이익", "순이익"], [],
                [[s.period, f"{s.revenue:,.0f}", f"{s.operating_profit:,.0f}", f"{s.net_income:,.0f}"] for s in inc])})
        if rat:
            sections.append({"title": "주요 재무비율", **_table(
                ["결산", "ROE%", "EPS", "BPS", "부채비율%"], [],
                [[s.period, f"{s.roe:.2f}", f"{s.eps:,.0f}", f"{s.bps:,.0f}", f"{s.debt_ratio:.1f}"] for s in rat])})
        if bal:
            sections.append({"title": "재무상태표 (억원)", **_table(
                ["결산", "자산총계", "부채총계", "자본총계"], [],
                [[s.period, f"{s.total_assets:,.0f}", f"{s.total_liabilities:,.0f}", f"{s.total_equity:,.0f}"] for s in bal])})
        return {"type": "sections", "sections": sections}

    if kind == "opinion":
        end = get_clock().today()
        ops = await b.fetch_invest_opinion(code, (end - timedelta(days=180)).strftime("%Y%m%d"), end.strftime("%Y%m%d"))
        ops = sorted(ops, key=lambda o: o.day, reverse=True)[:12]
        return {**_table(
            ["일자", "의견", "직전", "목표가", "전일종가"], [],
            [[o.day, o.opinion, o.prev_opinion, f"{o.target_price:,.0f}", f"{o.prev_close:,.0f}"] for o in ops],
        ), "caption": "국내 증권사 투자의견·목표가"}

    if kind == "risk":
        end = get_clock().today()
        shorts = (await b.fetch_daily_short_sale(code, (end - timedelta(days=30)).strftime("%Y%m%d"), end.strftime("%Y%m%d")))
        shorts = sorted(shorts, key=lambda s: s.day, reverse=True)[:10]
        note = ""
        try:
            prog = await b.fetch_program_trade(code)
            if prog:
                note = f"프로그램 순매수(최근): {prog[0].net_buy_qty:+,}주"
        except Exception:
            pass
        return {**_table(
            ["일자", "종가", "공매도량", "공매도비중%", "누적비중%"], [],
            [[s.day, f"{s.close:,.0f}", f"{s.short_volume:,}", f"{s.short_ratio:.2f}", f"{s.accum_short_ratio:.2f}"] for s in shorts],
        ), "caption": "공매도 추이", "note": note}

    if kind == "news":
        items = (await b.fetch_news_titles(code))[:15]
        return {"type": "news", "items": [
            {"time": f"{n.day[4:6]}/{n.day[6:]} {n.time[:2]}:{n.time[2:4]}" if n.day else "",
             "source": n.source, "title": n.title} for n in items]}


@router.get("/api/kr/{kind}/{ticker}")
async def kr_insight(kind: str, ticker: str) -> dict:
    """Korean-market analysis data (supply-demand, financials, analyst
    opinion, risk signals, news) as structured tables for the dashboard
    panel. Content is Korean-only for now — a deliberate, known scope
    (see the merge plan's decision point) — only tab labels/section titles
    are wired through the client i18n system."""
    if kind not in _KR_INSIGHT_KINDS:
        raise HTTPException(404, f"unknown insight kind {kind!r}")
    broker = await state.kis_broker()
    if broker is None:
        raise HTTPException(503, "KIS broker not configured")
    krx_code, _ = route_ticker(ticker.upper())
    if krx_code is None:
        return {"unsupported": True}

    async def _load():
        try:
            return await _kr_insight_data(kind, krx_code)
        except Exception as exc:
            logger.warning("KR insight %s failed for %s", kind, ticker, exc_info=True)
            return {"error": f"데이터를 불러오지 못했습니다: {exc}"}

    return await state.cached(f"kr:{kind}:{krx_code}", 30, _load)
