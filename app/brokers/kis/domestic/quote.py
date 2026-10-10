from app.brokers.kis.const import KISBrokerError
from app.brokers.kis.parser import _as_rows, _kis_datetime_to_iso, _num
from app.clock import get_clock
from app.domain.models import BidAsk, Candle, MinuteCandle, StockDesignations, StockStatus, StockValuation

_PRICE_TR_ID = "FHKST01010100"  # 주식현재가 시세 (실전/모의 동일)
_PRICE_PATH = "/uapi/domestic-stock/v1/quotations/inquire-price"
_ASKING_PRICE_TR_ID = "FHKST01010200"  # 주식현재가 호가/예상체결 (실전/모의 동일)
_ASKING_PRICE_PATH = "/uapi/domestic-stock/v1/quotations/inquire-asking-price-exp-ccn"
_CANDLES_TR_ID = "FHKST03010100"  # 기간별시세 (일/주/월/년)
_CANDLES_PATH = "/uapi/domestic-stock/v1/quotations/inquire-daily-itemchartprice"
_STOCK_STATUS_TR_ID = "CTPF1002R"  # 주식기본조회 (실전/모의 동일)
_STOCK_STATUS_PATH = "/uapi/domestic-stock/v1/quotations/search-stock-info"
_MINUTE_CHART_TR_ID = "FHKST03010200"  # 주식당일분봉조회 (intraday minute OHLCV)
_MINUTE_CHART_PATH = "/uapi/domestic-stock/v1/quotations/inquire-time-itemchartprice"
_NEWS_TR_ID = "FHKST01011800"  # 종합 시황/공시(제목) [국내주식-141]
_NEWS_PATH = "/uapi/domestic-stock/v1/quotations/news-title"


class DomesticQuoteMixin:
    # ---------------- quotations (실전/모의 공통 TR) ----------------

    async def fetch_price(self, code: str) -> float:
        """주식현재가 — the live traded price for a KRX code."""
        resp = await self._request(
            "GET", _PRICE_PATH, _PRICE_TR_ID,
            params={"FID_COND_MRKT_DIV_CODE": "J", "FID_INPUT_ISCD": code},
        )
        body = self._checked_body(resp, f"price inquiry for {code}")
        price = _num(body.get("output", {}).get("stck_prpr"))
        if price <= 0:
            raise KISBrokerError(f"price inquiry for {code} returned no price")
        return price

    async def fetch_asking_price(self, code: str) -> BidAsk:
        """주식현재가 호가/예상체결 — best bid/ask (1호가) for a liquidity/
        spread pre-filter. Same TR on paper and real (unlike CTPF1002R/
        CTPF1702R's company-name TRs — this one isn't account-mode gated,
        confirmed by this TR ID being identical for both env_dv branches in
        KIS's own reference implementation)."""
        resp = await self._request(
            "GET", _ASKING_PRICE_PATH, _ASKING_PRICE_TR_ID,
            params={"FID_COND_MRKT_DIV_CODE": "J", "FID_INPUT_ISCD": code},
        )
        body = self._checked_body(resp, f"asking price inquiry for {code}")
        out = body.get("output1", {})
        return BidAsk(
            code=code,
            bid=_num(out.get("bidp1")),
            ask=_num(out.get("askp1")),
            last=_num(out.get("stck_prpr")),
        )

    async def fetch_valuation(self, code: str) -> StockValuation:
        """Same TR/endpoint as fetch_price (주식현재가 시세) — per/pbr/eps/bps/
        market-cap ride along in that response already, just unparsed
        before this. Kept as a separate method rather than changing
        fetch_price's return shape (fetch_price is a plain float getter
        used elsewhere; no need to touch its contract for this)."""
        resp = await self._request(
            "GET", _PRICE_PATH, _PRICE_TR_ID,
            params={"FID_COND_MRKT_DIV_CODE": "J", "FID_INPUT_ISCD": code},
        )
        body = self._checked_body(resp, f"valuation inquiry for {code}")
        out = body.get("output", {})
        return StockValuation(
            code=code,
            price=_num(out.get("stck_prpr")),
            per=_num(out.get("per")),
            pbr=_num(out.get("pbr")),
            eps=_num(out.get("eps")),
            bps=_num(out.get("bps")),
            market_cap=_num(out.get("hts_avls")),
        )

    async def fetch_market_warning(self, code: str) -> str:
        """`mrkt_warn_cls_code` (시장경고코드) — same TR/endpoint as
        fetch_price/fetch_valuation/fetch_sector_name (주식현재가 시세,
        FHKST01010100), zero new network round-trip beyond what a caller
        already pays for one of those. Empty string for "no designation" —
        confirmed convention for every other Y/N-shaped field this same TR
        returns (`admn_item_yn`/`tr_stop_yn` elsewhere in this codebase
        follow the identical "blank/'N' means clean" shape) and corroborated
        by three separate KIS ranking-API request parameters that classify
        this exact KRX designation family with `0` meaning "전체"/no
        specific tier (분류 구분 코드: 0:전체, 1:관리종목, 2:투자주의,
        3:투자경고, 4:투자위험예고, 5:투자위험 — see 국내주식 이격도/시가총액/
        신고신저 순위 API request params). This module does NOT independently
        distinguish 투자주의/투자경고/투자위험 tiers by number — the caller
        (`app.execution.kis_bridge._market_warning_block`) only checks
        presence/absence, which is exactly what `mrkt_warn_cls_code not in
        ("", "0", "00")` gives without betting on which literal code the
        live server uses for the negative case. Not independently verified
        against a live example carrying a non-empty value (no ticker under
        an active KRX warning designation was available to test against) —
        the caller logs the raw code on every match specifically so the
        first live hit either confirms this or surfaces a wrong sentinel
        immediately."""
        resp = await self._request(
            "GET", _PRICE_PATH, _PRICE_TR_ID,
            params={"FID_COND_MRKT_DIV_CODE": "J", "FID_INPUT_ISCD": code},
        )
        body = self._checked_body(resp, f"market warning inquiry for {code}")
        return (body.get("output", {}).get("mrkt_warn_cls_code") or "").strip()

    async def fetch_designations(self, code: str) -> StockDesignations:
        """KRX pre-screening designation flags bundled into one call — same
        TR/endpoint as fetch_market_warning/fetch_price/fetch_valuation/
        fetch_sector_name (주식현재가 시세, FHKST01010100), zero new network
        round-trip. Supersedes fetch_market_warning for
        app.execution.kis_bridge's designation check (that method itself is
        left in place — its own tests exercise it directly).

        2026-09-08 (§4-5, screening audit follow-up): `short_over_yn`
        (단기과열여부)/`sltr_yn`(정리매매여부) were sitting unparsed in the
        same response `mrkt_warn_cls_code` already came from. Both are
        `_yn`-suffixed fields, so "Y" (case-insensitive, trimmed) is the
        only value read as a designation — same convention this codebase
        already uses for `admn_item_yn`/`tr_stop_yn` (fetch_stock_status
        above). `mang_issu_cls_code` (관리종목여부) is captured RAW, not
        parsed to a bool — see StockDesignations' own docstring for why its
        enum direction cannot be assumed from the field name/label alone."""
        resp = await self._request(
            "GET", _PRICE_PATH, _PRICE_TR_ID,
            params={"FID_COND_MRKT_DIV_CODE": "J", "FID_INPUT_ISCD": code},
        )
        body = self._checked_body(resp, f"designation inquiry for {code}")
        out = body.get("output", {})
        return StockDesignations(
            code=code,
            market_warning=(out.get("mrkt_warn_cls_code") or "").strip(),
            short_overheat=(out.get("short_over_yn") or "").strip().upper() == "Y",
            liquidation=(out.get("sltr_yn") or "").strip().upper() == "Y",
            administrative_raw=(out.get("mang_issu_cls_code") or "").strip(),
        )

    async def fetch_sector_name(self, code: str) -> tuple[str, str]:
        """(업종 한글명, 대표시장명) — from 주식현재가 시세 (FHKST01010100)'s
        `bstp_kor_isnm` / `rprs_mrkt_kor_name`. Same TR/endpoint as
        fetch_valuation, real==paper, so it is NOT subject to the CTPF1002R
        (real-account-only) limitation identity_gate's docstring describes.
        Returned name is a standard KRX industry name and joins char-for-char
        with kis_provider._KR_REAL_SECTOR_NAMES (verified live: 270520->'일반서비스',
        131760->'기계·장비', 088350->'보험'). Names outside the allowlist do occur
        (026040->'기타제조') — the caller must handle that."""
        resp = await self._request(
            "GET", _PRICE_PATH, _PRICE_TR_ID,
            params={"FID_COND_MRKT_DIV_CODE": "J", "FID_INPUT_ISCD": code},
        )
        body = self._checked_body(resp, f"sector name inquiry for {code}")
        out = body.get("output", {})
        return (
            (out.get("bstp_kor_isnm") or "").strip(),
            (out.get("rprs_mrkt_kor_name") or "").strip(),
        )

    async def fetch_stock_status(self, code: str) -> StockStatus:
        """주식기본조회 — per-stock trading-halt / administrative-issue status,
        checked in app.execution.kis_bridge right before a KRX order is
        submitted. PRDT_TYPE_CD=300 covers 주식/ETF/ETN/ELW, the only kinds
        this project trades domestically.

        Real-account only — confirmed live that a paper account gets
        EGW02006 ("모의투자 TR 이 아닙니다") for this whole TR, not just the
        name fields. Callers on a paper broker will see this raise before
        korean_name/english_name are ever reached."""
        resp = await self._request(
            "GET", _STOCK_STATUS_PATH, _STOCK_STATUS_TR_ID,
            params={"PRDT_TYPE_CD": "300", "PDNO": code}, priority="high",
        )
        body = self._checked_body(resp, f"stock status inquiry for {code}")
        out = body.get("output", {})
        return StockStatus(
            code=code,
            trading_halted=out.get("tr_stop_yn") == "Y",
            administrative_issue=out.get("admn_item_yn") == "Y",
            nxt_trading_halted=out.get("nxt_tr_stop_yn") == "Y",
            korean_name=out.get("prdt_abrv_name", ""),
            english_name=out.get("prdt_eng_abrv_name", ""),
            security_group_code=out.get("scty_grp_id_cd", ""),
        )

    # Real-time WebSocket approval-key issuance lives in
    # app.brokers.kis_ws.fetch_approval_key (a standalone function, not a
    # KISBroker method — that module's RealtimeFeed is self-contained) —
    # not duplicated here.

    async def fetch_daily_candles(
        self, code: str, start: str, end: str, period: str = "D"
    ) -> list[Candle]:
        """기간별시세 — adjusted OHLCV between `start` and `end` (YYYYMMDD),
        oldest first. `period` D=일 / W=주 / M=월 / Y=년봉. The API caps one
        call at 100 rows, so the caller widens the range for coarser periods."""
        if period not in ("D", "W", "M", "Y"):
            raise ValueError(f"period must be D/W/M/Y, got {period!r}")
        resp = await self._request(
            "GET", _CANDLES_PATH, _CANDLES_TR_ID,
            params={
                "FID_COND_MRKT_DIV_CODE": "J",
                "FID_INPUT_ISCD": code,
                "FID_INPUT_DATE_1": start,
                "FID_INPUT_DATE_2": end,
                "FID_PERIOD_DIV_CODE": period,
                "FID_ORG_ADJ_PRC": "0",  # 0 = 수정주가
            },
        )
        body = self._checked_body(resp, f"daily candles for {code}")
        candles = [
            Candle(
                day=row.get("stck_bsop_date", ""),
                open=_num(row.get("stck_oprc")),
                high=_num(row.get("stck_hgpr")),
                low=_num(row.get("stck_lwpr")),
                close=_num(row.get("stck_clpr")),
                volume=int(_num(row.get("acml_vol"))),
            )
            for row in body.get("output2", [])
            if row.get("stck_bsop_date")
        ]
        candles.sort(key=lambda c: c.day)
        return candles

    async def fetch_minute_candles(self, code: str, end_hour: str = "") -> list[MinuteCandle]:
        """주식당일분봉조회 — today's 1-minute OHLCV up to `end_hour` (HHMMSS;
        empty = latest), oldest first. Returns the most recent ~30 minutes."""
        resp = await self._request(
            "GET", _MINUTE_CHART_PATH, _MINUTE_CHART_TR_ID,
            params={
                "FID_COND_MRKT_DIV_CODE": "J", "FID_INPUT_ISCD": code,
                "FID_INPUT_HOUR_1": end_hour or get_clock().now().strftime("%H%M%S"),
                "FID_PW_DATA_INCU_YN": "Y", "FID_ETC_CLS_CODE": "",
            },
        )
        body = self._checked_body(resp, f"minute candles for {code}")
        candles = [
            MinuteCandle(
                time=row.get("stck_cntg_hour", ""),
                open=_num(row.get("stck_oprc")),
                high=_num(row.get("stck_hgpr")),
                low=_num(row.get("stck_lwpr")),
                close=_num(row.get("stck_prpr")),
                volume=int(_num(row.get("cntg_vol"))),
            )
            for row in body.get("output2", [])
            if row.get("stck_cntg_hour")
        ]
        candles.sort(key=lambda c: c.time)
        return candles

    async def fetch_news(self, code: str = "", limit: int = 10) -> list[dict]:
        """종합 시황/공시(제목) — KIS-only news for the frontend (see
        app.live.poller's module docstring for the "frontend is KIS-only,
        no yfinance fallback" policy). Title/source/time only: unlike
        yfinance's get_news_items, this KIS TR has no article link or
        thumbnail — dropped from the response entirely rather than faked."""
        resp = await self._request(
            "GET", _NEWS_PATH, _NEWS_TR_ID,
            params={
                "FID_NEWS_OFER_ENTP_CODE": "", "FID_COND_MRKT_CLS_CODE": "",
                "FID_INPUT_ISCD": code, "FID_TITL_CNTT": "",
                "FID_INPUT_DATE_1": "", "FID_INPUT_HOUR_1": "",
                "FID_RANK_SORT_CLS_CODE": "", "FID_INPUT_SRNO": "",
            },
        )
        body = self._checked_body(resp, f"news inquiry for {code or '(all)'}")
        rows = _as_rows(body.get("output"))
        items = []
        for row in rows[:limit]:
            title = row.get("hts_pbnt_titl_cntt", "")
            if not title:
                continue
            items.append({
                "title": title,
                "publisher": row.get("dorg", ""),
                "published_at": _kis_datetime_to_iso(row.get("data_dt", ""), row.get("data_tm", "")),
            })
        return items
