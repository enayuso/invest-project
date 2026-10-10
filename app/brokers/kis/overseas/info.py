# app/brokers/kis/overseas/info.py
# 해외주식 기타 정보 (뉴스/상품기본정보) — 국내와 대칭이지만 응답 envelope가 다름(outblock1)
from app.brokers.kis.overseas.quote import _OVERSEAS_NEWS_PATH, _OVERSEAS_NEWS_TR
from app.brokers.kis.parser import _as_rows, _num
from app.domain.models import IndexQuote, NewsTitle

_OVERSEAS_INFO_TR = "CTPF1702R"  # 해외주식 상품기본정보
_OVERSEAS_INFO_PATH = "/uapi/overseas-price/v1/quotations/search-info"
_OVERSEAS_INDEX_TR = "FHKST03030100"  # 해외주식 종목/지수/환율기간별시세(일/주/월/년)
_OVERSEAS_INDEX_PATH = "/uapi/overseas-price/v1/quotations/inquire-daily-chartprice"


class OverseasInfoMixin:
    async def fetch_overseas_company_name(self, symbol: str, prdt_type_cd: str = "512") -> tuple[str, str]:
        """해외주식 상품기본정보 — (한글 상품약어명, 영문 상품약어명).
        `prdt_type_cd` defaults to 512 (나스닥) matching this project's
        current NASDAQ-only overseas trading scope elsewhere (order routing,
        fetch_overseas_price/candles' "NAS" default) — pass a different code
        (513 NYSE / 529 AMEX / etc.) for other exchanges.

        Field names confirmed against KIS's own official column mapping for
        this TR (`examples_llm/overseas_stock/search_info/chk_search_info.py`):
        `prdt_name` (상품명, Korean) / `prdt_eng_name` (상품영문명) — the
        previous `prdt_abrv_name`/`prdt_eng_abrv_name` guess (carried over
        from the domestic TR's naming convention) doesn't exist on this one
        and always returned ("", ""). A missing field still degrades safely
        to ("", "") — callers fall back to yfinance's own name — but this TR
        actually does have the data now."""
        resp = await self._request(
            "GET", _OVERSEAS_INFO_PATH, _OVERSEAS_INFO_TR,
            params={"PRDT_TYPE_CD": prdt_type_cd, "PDNO": symbol},
        )
        body = self._checked_body(resp, f"overseas company info for {symbol}")
        out = body.get("output", {})
        return out.get("prdt_name", ""), out.get("prdt_eng_name", "")

    async def fetch_overseas_news_titles(
        self, symbol: str, exchange: str = "", nation: str = ""
    ) -> list[NewsTitle]:
        """해외뉴스종합(제목) — `title`/`source` are third-party free text,
        same untrusted-input treatment as the domestic news TR (the caller
        must wrap before it enters an LLM prompt). Response envelope key is
        `outblock1`, not `output`/`output1`/`output2` like every other TR in
        this package — a real KIS inconsistency, not a typo here."""
        resp = await self._request(
            "GET", _OVERSEAS_NEWS_PATH, _OVERSEAS_NEWS_TR,
            params={
                "INFO_GB": "", "CLASS_CD": "", "NATION_CD": nation,
                "EXCHANGE_CD": exchange, "SYMB": symbol,
                "DATA_DT": "", "DATA_TM": "", "CTS": "",
            },
        )
        body = self._checked_body(resp, f"overseas news for {symbol}")
        # _as_rows: outblock1 comes back as a bare dict (not a list) when
        # exactly one article matches — same normalization fetch_overseas_news
        # (quote.py) already applies to this same TR; without it, iterating a
        # dict yields its keys (strings) and item.get(...) crashes.
        items = _as_rows(body.get("outblock1"))
        return [
            NewsTitle(
                day=item.get("data_dt", ""),
                time=item.get("data_tm", ""),
                title=(item.get("title") or "").strip(),
                source=(item.get("source") or "").strip(),
            )
            for item in items
        ]

    async def fetch_overseas_index_quote(self, symbol: str, name: str, start: str, end: str) -> IndexQuote:
        """해외지수 현재가 — the overseas counterpart to DomesticInfoMixin.
        fetch_index_price, via the same TR this project already uses for
        overseas market-cap-adjacent lookups (FHKST03030100, FID_COND_
        MRKT_DIV_CODE="N" for 해외지수). Only three US index codes are
        queryable through this TR (per KIS's own docs): Dow Jones (symbol
        ".DJI" — note the leading dot), S&P 500 (symbol "SPX" — no dot),
        and Nasdaq 100 ("NDX", no dot) / Nasdaq Composite ("COMP", no dot).
        Confirmed live (2026-08-04): every other guessed code (.SPX, .NDX,
        .IXIC, DJI without the dot, ...) silently returns an all-zero
        output1 rather than an error — KIS's own docs point to a master-
        code file (포럼 > FAQ > 종목정보 다운로드(해외) > 해외지수) for
        anything beyond these four, not guessable from the API alone.

        `start`/`end` (YYYYMMDD) are required by the TR but don't appear to
        affect output1 (always "as of now", like fetch_index_price) —
        still passed through rather than hardcoded so a caller building
        output2 history off the same call has a real window to work with.
        `change`/`change_pct` are read straight off ovrs_nmix_prdy_vrss/
        prdy_ctrt — confirmed live these come back already signed (a
        negative session shows a leading "-"), no separate sign-flag
        field to apply, same convention as the domestic index quote."""
        resp = await self._request(
            "GET", _OVERSEAS_INDEX_PATH, _OVERSEAS_INDEX_TR,
            params={
                "FID_COND_MRKT_DIV_CODE": "N",
                "FID_INPUT_ISCD": symbol,
                "FID_INPUT_DATE_1": start,
                "FID_INPUT_DATE_2": end,
                "FID_PERIOD_DIV_CODE": "D",
            },
        )
        out = self._checked_body(resp, f"overseas index quote {symbol}").get("output1", {})
        return IndexQuote(
            code=symbol,
            name=name,
            value=_num(out.get("ovrs_nmix_prpr")),
            change=_num(out.get("ovrs_nmix_prdy_vrss")),
            change_pct=_num(out.get("prdy_ctrt")),
        )
