import logging
from collections import Counter

from app.brokers.kis.client import MAX_PAGINATION_PAGES
from app.brokers.kis.const import KISBrokerError
from app.brokers.kis.parser import _num
from app.clock import get_clock
from app.domain.models import (
    DividendInfo,
    IndexDailyBar,
    IndexQuote,
    InvestmentOpinion,
    InvestorEstimate,
    InvestorFlow,
    NewsTitle,
    ProgramTrade,
    SectorIndex,
    ShortSaleDay,
    VIStatus,
)

logger = logging.getLogger(__name__)

_VI_STATUS_TR_ID = "FHPST01390000"  # 변동성완화장치(VI) 현황 (시세 TR, 실전/모의 공통 추정)
_VI_STATUS_PATH = "/uapi/domestic-stock/v1/quotations/inquire-vi-status"
_INDEX_PRICE_TR_ID = "FHPUP02100000"  # 국내 업종/지수 현재가 (KOSPI 0001 / KOSDAQ 1001)
_INDEX_PRICE_PATH = "/uapi/domestic-stock/v1/quotations/inquire-index-price"
_HOLIDAY_TR_ID = "CTCA0903R"  # 국내휴장일조회
_HOLIDAY_PATH = "/uapi/domestic-stock/v1/quotations/chk-holiday"
_INDEX_DAILY_CHART_TR_ID = "FHKUP03500100"  # 국내주식업종기간별시세(일/주/월/년) — KOSPI/KOSDAQ 과거 일봉
_INDEX_DAILY_CHART_PATH = "/uapi/domestic-stock/v1/quotations/inquire-daily-indexchartprice"
_INDEX_CATEGORY_TR_ID = "FHPUP02140000"  # 업종별 지수 시세 (섹터 등락)
_INDEX_CATEGORY_PATH = "/uapi/domestic-stock/v1/quotations/inquire-index-category-price"
# FID_BLNG_CLS_CODE means a DIFFERENT enum per market for this TR (KIS doc):
#   K (거래소/KOSPI): 0 전업종 / 1 기타구분 / 2 자본금구분 / 3 상업별구분
#   Q (코스닥):       0 전업종 / 1 기타구분 / 2 벤처구분   / 3 일반구분
# See fetch_sector_indices's own docstring for why KOSDAQ uses "3", not "0".
_INDEX_CATEGORY_BLNG_CLS = {"0001": "0", "1001": "3"}
# 수급 (한국시장 고유 신호 — yfinance엔 없음). 둘 다 모의서버 지원 확인됨.
_INVESTOR_TREND_TR_ID = "FHKST01010900"  # 종목별 투자자 매매동향 (일별)
_INVESTOR_TREND_PATH = "/uapi/domestic-stock/v1/quotations/inquire-investor"
_INVESTOR_ESTIMATE_TR_ID = "HHPTJ04160200"  # 종목별 외국인/기관 추정 가집계 (당일)
_INVESTOR_ESTIMATE_PATH = "/uapi/domestic-stock/v1/quotations/investor-trend-estimate"
# 국내 증권사 투자의견 (yfinance 컨센서스는 한국 종목에서 거의 빔). 모의 지원 확인됨.
_INVEST_OPINION_TR_ID = "FHKST663300C0"  # 종목 투자의견 (증권사 컨센서스 이력)
_INVEST_OPINION_PATH = "/uapi/domestic-stock/v1/quotations/invest-opinion"
# 리스크 신호 (종목단위). 둘 다 모의 지원 확인됨.
_SHORT_SALE_TR_ID = "FHPST04830000"  # 종목별 일별 공매도
_SHORT_SALE_PATH = "/uapi/domestic-stock/v1/quotations/daily-short-sale"
_PROGRAM_TRADE_TR_ID = "FHPPG04650101"  # 종목별 프로그램매매 동향
_PROGRAM_TRADE_PATH = "/uapi/domestic-stock/v1/quotations/program-trade-by-stock"
_NEWS_TITLE_TR_ID = "FHKST01011800"  # 종목 뉴스 제목 (자유텍스트 — 인젝션 방어 필요)
_NEWS_TITLE_PATH = "/uapi/domestic-stock/v1/quotations/news-title"
_DIVIDEND_TR_ID = "HHKDB669102C0"  # 예탁원 배당 일정 (기업행위)
_DIVIDEND_PATH = "/uapi/domestic-stock/v1/ksdinfo/dividend"


class DomesticInfoMixin:
    async def fetch_sector_indices(self, market: str = "0001") -> list[SectorIndex]:
        """업종별 지수 시세 — sector regime (섹터별 등락). `market` 0001 KOSPI /
        1001 KOSDAQ. A market-breadth view the agents/dashboard otherwise lack.

        Paginated (tr_cont "M"/"F" means more rows) up to MAX_PAGINATION_PAGES
        — KIS's own sample for this TR recurses on tr_cont same as
        fetch_vi_status, and like that TR it carries no CTX_AREA_FK/NK
        continuation keys, so it can't reuse the shared `_paginate` helper.
        A single unpaginated call previously silently dropped every sector
        past whatever the first page held.

        KIS doc investigation (TR FHPUP02140000, 국내업종 구분별전체시세, examples_llm
        inquire_index_category_price): FID_INPUT_ISCD is the index-universe code
        (0001 KOSPI / 1001 KOSDAQ), FID_MRKT_CLS_CODE is K:거래소 / Q:코스닥,
        FID_BLNG_CLS_CODE means a DIFFERENT enum per market (see
        `_INDEX_CATEGORY_BLNG_CLS`'s own comment). The sector rows come back in
        output2 (output1 is a single index-summary row); each row carries
        `bstp_cls_code` and `hts_kor_isnm` but NO per-row market indicator
        field. The continuation contract is: recurse while the response
        `tr_cont` header is "M"/"F", re-sending `tr_cont="N"` with the same
        params.

        Observed defect (2026-09): `fetch_sector_indices("1001")` returned KOSPI
        industry rows (`0025 보험`, `0006 섬유·의류`, `0005 음식료·담배`, `0027 제조`)
        mixed into the KOSDAQ response, plus duplicate rows, non-deterministically
        across calls — the server does not reliably scope the response to
        FID_MRKT_CLS_CODE, and pagination re-serves overlapping chunks. Since no
        row carries a market field, we filter by the code prefix each market's
        industry codes carry (KOSPI = `00NN`, KOSDAQ = `10NN`), dedupe by
        `bstp_cls_code` (keep first), and sort by code so the result is
        deterministic regardless of page interleaving.

        2026-09-09 root cause + fix: with FID_BLNG_CLS_CODE="0" (전업종),
        KOSDAQ's response is capped at 100 raw rows drawn from a larger,
        time-varying pool with no reliable tr_cont continuation — so the
        number of genuine KOSDAQ (`10NN`) rows in any one call swung 0/1/2/6/
        7/12/17/19 across repeated live probes minutes apart, i.e. coverage is
        NOT just capped but randomly sampled every call. `app.experiments.
        sector_index_probe` (deleted once this was decided — see git history,
        commit before this one, if it needs resurrecting) swept
        FID_BLNG_CLS_CODE against KOSDAQ (Q) and found "3" (일반구분) returns
        exactly 17 rows, byte-identical across 3 repeats 5 minutes apart
        (KST 09:52-10:19) and again in a follow-up single run ~20 minutes
        later (KST 10:28, investigating a separate combo) — under the
        100-row cap, no cap_reached, every row genuinely prefix
        `10`, 16 of 17 names already in `_KR_REAL_SECTOR_NAMES` (the 17th,
        출판·매체복제, was added there as part of this fix). "2" (벤처구분)
        returns 20 rows with `bstp_cls_code` AND `hts_kor_isnm` both blank on
        every row — useless. "1" (기타구분) is stable too but mostly returns
        market-cap/company-tier rows (코스닥 소형주, 우량기업, 벤처기업, ...),
        not industry sectors.

        Known, accepted gap: blng3's 17 rows do not include 제조/건설
        (manufacturing/construction) or 오락·문화 (entertainment/culture),
        which occasionally surface in blng0's unstable samples. These read as
        rollup/aggregate-level industry rows (제조 "manufacturing" vs. the
        leaf categories 화학/제약/금속/... blng3 already returns) rather than
        sectors blng3 is missing outright — including them would double-count
        against their own leaf constituents in the breadth ranking. The cost
        of the gap is a missing `_own_sector_line` line for a stock in one of
        those three sectors (`app.dataflows.kis_provider`'s "not in ranking"
        branch — an honest "unknown this cycle", never a misattribution),
        not a wrong answer. Revisit only if live `sector indices scan` logs
        show `kept` for KOSDAQ drifting from 17 (see that log line below)."""
        if market not in ("0001", "1001"):
            raise ValueError(f"market must be '0001' (KOSPI) or '1001' (KOSDAQ), got {market!r}")
        params = {
            "FID_COND_MRKT_DIV_CODE": "U", "FID_INPUT_ISCD": market,
            "FID_COND_SCR_DIV_CODE": "20214",
            "FID_MRKT_CLS_CODE": "K" if market == "0001" else "Q",
            "FID_BLNG_CLS_CODE": _INDEX_CATEGORY_BLNG_CLS[market],
        }
        rows: list[dict] = []
        req_tr_cont = ""
        pages_consumed = 0
        cap_reached = False
        for _ in range(MAX_PAGINATION_PAGES):
            pages_consumed += 1
            resp = await self._request(
                "GET", _INDEX_CATEGORY_PATH, _INDEX_CATEGORY_TR_ID, params=params, tr_cont=req_tr_cont,
            )
            body = self._checked_body(resp, "sector indices")
            rows.extend(body.get("output2") or body.get("output1") or [])
            if resp.headers.get("tr_cont", "").strip() not in ("M", "F"):
                break
            req_tr_cont = "N"
        else:
            cap_reached = True
            logger.warning(
                "sector indices: hit MAX_PAGINATION_PAGES=%d with more pages still available — "
                "results truncated", MAX_PAGINATION_PAGES,
            )
        market_prefix = "00" if market == "0001" else "10"
        other_prefix = "10" if market == "0001" else "00"
        seen: set[str] = set()
        deduped: list[dict] = []
        # Diagnostics only (2026-09-04, extended 2026-09-09) — no effect on
        # filtering/dedupe/sort/return. The other-market-prefix count is what a
        # follow-up design hinges on: it says whether a call for one market also
        # hands back the other market's rows that we currently discard.
        #
        # 2026-09-09 addition: a raw-code-prefix histogram + a sample of rows
        # whose prefix is neither the requested market's nor the other market's.
        # Without this we cannot tell whether the ~70-99 rows dropped per KOSDAQ
        # call are genuinely non-sector products (derivatives/themes) or real
        # KOSDAQ sector rows under a prefix we don't recognize — see
        # [[screening-designation-prefilter-2026-09-08]]-adjacent §섹터breadth
        # investigation (2026-09-09).
        prefix_histogram: Counter[str] = Counter()
        unknown_prefix_samples: list[str] = []
        requested_prefix_rows = other_prefix_rows = dup_dropped = no_prefix_or_name_dropped = 0
        for row in rows:
            code = (row.get("bstp_cls_code") or "").strip() if isinstance(row, dict) else ""
            name = (row.get("hts_kor_isnm") or "").strip() if isinstance(row, dict) else ""
            code_prefix = code[:2] if code else "??"
            prefix_histogram[code_prefix] += 1
            if code.startswith(market_prefix):
                requested_prefix_rows += 1
            elif code.startswith(other_prefix):
                other_prefix_rows += 1
            elif len(unknown_prefix_samples) < 10:
                unknown_prefix_samples.append(f"{code}:{name}")
            if not (isinstance(row, dict) and row.get("hts_kor_isnm")):
                no_prefix_or_name_dropped += 1
                continue
            if not code.startswith(market_prefix):
                no_prefix_or_name_dropped += 1
                continue
            if code in seen:
                dup_dropped += 1
                continue
            seen.add(code)
            deduped.append(row)
        deduped.sort(key=lambda r: (r.get("bstp_cls_code") or "").strip())
        histogram_str = " ".join(
            f"{prefix}:{count}" for prefix, count in sorted(prefix_histogram.items(), key=lambda kv: -kv[1])
        )
        logger.info(
            "sector indices scan: market=%s pages=%d cap_reached=%s raw_rows=%d "
            "requested_prefix_rows=%d other_market_prefix_rows=%d dup_dropped=%d "
            "no_prefix_or_name_dropped=%d kept=%d prefix_histogram=[%s] unknown_prefix_samples=%s",
            market, pages_consumed, cap_reached, len(rows),
            requested_prefix_rows, other_prefix_rows, dup_dropped,
            no_prefix_or_name_dropped, len(deduped), histogram_str, unknown_prefix_samples,
        )
        return [
            SectorIndex(
                code=(row.get("bstp_cls_code") or "").strip(),
                name=(row.get("hts_kor_isnm") or "").strip(),
                value=_num(row.get("bstp_nmix_prpr")),
                change_pct=_num(row.get("bstp_nmix_prdy_ctrt")),
            )
            for row in deduped
        ]

    async def fetch_vi_status(
        self,
        *,
        market: str = "0",
        rank_sort: str = "0",
        div_cls: str = "0",
        code: str = "",
        business_day: str = "",
    ) -> list[VIStatus]:
        """변동성완화장치(VI) 현황 — the stocks currently under a VI halt.

        VI (volatility interruption) pauses matching for a couple of minutes,
        so an order placed against a stock in VI won't fill — this is a
        diagnosis tool for that. `market` 0=전체/K=거래소/Q=코스닥,
        `rank_sort` 0=전체/1=정적/2=동적/3=정적&동적, `div_cls` 0=전체/1=상승/
        2=하락. `business_day` defaults to today (KST); VI only exists during a
        live session so the current day is the meaningful query.

        This IS wired up as a real safety gate — app.execution.kis_bridge's
        `_vi_block` calls `is_vi_triggered` (built on this method) to block
        orders against a currently-halted stock. An earlier version of this
        docstring claimed the opposite ("no caller applies this"), which was
        already stale by the time it was written.

        Paginated (tr_cont "M" means more rows) up to MAX_PAGINATION_PAGES —
        unlike every other paginated TR in this package, this one doesn't
        carry CTX_AREA_FK/NK continuation keys in the request, just
        `tr_cont` itself (confirmed against KIS's own sample), so it can't
        reuse the shared `_paginate` helper. On a day with many VI events,
        the target `code` could otherwise sit past page 1 and never be seen
        — a false "not halted" for exactly the gate this feeds.
        """
        if market not in ("0", "K", "Q"):
            raise ValueError(f"market must be '0' (all), 'K' (KRX), or 'Q' (KOSDAQ), got {market!r}")
        if rank_sort not in ("0", "1", "2", "3"):
            raise ValueError(f"rank_sort must be one of 0/1/2/3, got {rank_sort!r}")
        if div_cls not in ("0", "1", "2"):
            raise ValueError(f"div_cls must be one of 0/1/2, got {div_cls!r}")
        params = {
            "FID_DIV_CLS_CODE": div_cls,
            "FID_COND_SCR_DIV_CODE": "20139",
            "FID_MRKT_CLS_CODE": market,
            "FID_INPUT_ISCD": code,
            "FID_RANK_SORT_CLS_CODE": rank_sort,
            "FID_INPUT_DATE_1": business_day or get_clock().today().strftime("%Y%m%d"),
            "FID_TRGT_CLS_CODE": "",
            "FID_TRGT_EXLS_CLS_CODE": "",
        }
        rows: list[dict] = []
        req_tr_cont = ""
        for _ in range(MAX_PAGINATION_PAGES):
            resp = await self._request(
                "GET", _VI_STATUS_PATH, _VI_STATUS_TR_ID, params=params, tr_cont=req_tr_cont, priority="high",
            )
            body = self._checked_body(resp, "VI status inquiry")
            rows.extend(body.get("output") or body.get("output1") or [])
            if resp.headers.get("tr_cont", "").strip() != "M":
                break
            req_tr_cont = "N"
        else:
            logger.warning(
                "VI status inquiry hit MAX_PAGINATION_PAGES=%d with more pages still available — results truncated",
                MAX_PAGINATION_PAGES,
            )
        return [
            VIStatus(
                code=row.get("mksc_shrn_iscd", ""),
                name=row.get("hts_kor_isnm", ""),
                vi_cls_code=row.get("vi_cls_code", ""),
                vi_kind_code=row.get("vi_kind_code", ""),
                triggered_at=row.get("cntg_vi_hour", ""),
                released_at=row.get("vi_cncl_hour", ""),
                vi_price=_num(row.get("vi_prc")),
                vi_count=int(_num(row.get("vi_count"))),
            )
            for row in rows
        ]

    async def is_vi_triggered(self, code: str) -> bool:
        """True if `code` has an unresolved VI halt right now — a quick
        pre-order / order-failure-diagnosis check built on fetch_vi_status.

        fetch_vi_status returns the day's full VI history for the code, not
        just currently-active halts — a VI that triggered and was released
        hours ago still shows up in the same query. `released_at` (해제시간,
        empty until the halt is actually lifted) is what distinguishes an
        unresolved halt from a resolved one; checking membership alone (as
        this used to) reports every stock that was EVER VI-halted today as
        still halted, for the rest of the day."""
        return any(s.code == code and not s.released_at for s in await self.fetch_vi_status(code=code))

    async def fetch_index_price(self, code: str = "0001") -> IndexQuote:
        """국내 지수 현재가 — 0001 KOSPI / 1001 KOSDAQ. Market-regime context
        the agents otherwise lack (they see only the single stock)."""
        names = {"0001": "KOSPI", "1001": "KOSDAQ"}
        resp = await self._request(
            "GET", _INDEX_PRICE_PATH, _INDEX_PRICE_TR_ID,
            params={"FID_COND_MRKT_DIV_CODE": "U", "FID_INPUT_ISCD": code},
        )
        out = self._checked_body(resp, f"index price {code}").get("output", {})
        return IndexQuote(
            code=code,
            name=names.get(code, code),
            value=_num(out.get("bstp_nmix_prpr")),
            change=_num(out.get("bstp_nmix_prdy_vrss")),
            change_pct=_num(out.get("bstp_nmix_prdy_ctrt")),
        )

    async def fetch_index_daily_chart(
        self, code: str, start: str, end: str, period: str = "D"
    ) -> list[IndexDailyBar]:
        """국내 지수(KOSPI 0001 / KOSDAQ 1001) 기간별 일봉 — `fetch_index_price`
        above is a same-day snapshot only; a trend signal (e.g. a 200-day
        moving average) needs this historical series instead. `start`/`end`
        are YYYYMMDD, `period` D/W/M/Y (일/주/월/년, default daily). Most-
        recent-first or -last isn't guaranteed by the TR — callers should
        sort by `day` themselves rather than assume an order.

        Paginated via `_paginate` (2026-08-06) — this TR caps well short of
        a 200-day moving average's needed history in one page (confirmed
        live: 3 days, 49/49 calls for a ~350-calendar-day window all
        returned too few bars for app.execution.market_regime's Faber
        signal to ever compute). The example KIS client code for this exact
        TR (inquire_daily_indexchartprice) recurses on the response
        tr_cont header ("M"/"F") with no CTX_AREA_FK/NK fields at all — this
        TR's continuation apparently doesn't need them, unlike balance/
        executions — but sending them anyway (as empty strings, same as
        every other `_paginate` caller) is harmless: KIS ignores unknown
        query params, and `_paginate`'s own loop already stops correctly
        based on the tr_cont header alone regardless of whether the FK/NK
        fields are present in the response body (`body.get(...)` on a
        missing key harmlessly returns the same empty string already sent).

        Deduped by `day` (keep-first) as a defensive measure, not a
        confirmed-necessary one: if this TR's continuation turns out to
        NOT actually advance without those FK/NK fields (unconfirmed either
        way against a real account as of this writing), each "next page"
        would silently re-serve the same first chunk instead of erroring,
        corrupting the SMA with duplicated closes rather than just being
        short on history. Deduping makes that failure mode degrade to
        "fewer distinct days than expected" (caught by `_faber_signal`'s
        own `len(closes) < window` check) instead of a silently wrong
        average."""
        params = {
            "FID_COND_MRKT_DIV_CODE": "U",
            "FID_INPUT_ISCD": code,
            "FID_INPUT_DATE_1": start,
            "FID_INPUT_DATE_2": end,
            "FID_PERIOD_DIV_CODE": period,
            "CTX_AREA_FK100": "",
            "CTX_AREA_NK100": "",
        }
        by_day: dict[str, float] = {}
        async for body in self._paginate(
            "GET", _INDEX_DAILY_CHART_PATH, _INDEX_DAILY_CHART_TR_ID, params,
            what=f"index daily chart {code}",
        ):
            for row in body.get("output2", []):
                day = row.get("stck_bsop_date", "")
                if day and day not in by_day:
                    by_day[day] = _num(row.get("bstp_nmix_prpr"))
        return [IndexDailyBar(day=day, close=close) for day, close in by_day.items()]

    async def fetch_is_market_open_on(self, day: str) -> bool:
        """국내휴장일조회 — whether KRX actually traded on `day` (YYYYMMDD):
        weekends, KRX holidays (설날/추석/신정 etc.), and one-off market
        closures all read as opnd_yn != 'Y'. KIS's own docs ask that this
        TR be called at most once a day (it's tied to their own ledger
        service) — callers should cache the result per date rather than
        re-querying, see app.live.kr_holidays for that caching layer; this
        method is the bare, uncached KIS call.

        Confirmed live (2026-08-04): this TR rejects with EGW02006 ("모의투자
        TR이 아닙니다") on a PAPER account, same rejection this project has
        already seen from fetch_overseas_company_name — only usable on a
        real account. Untested against a real account so far; callers must
        treat a failure here as "unknown", not "definitely open" or
        "definitely closed" (see kr_holidays.is_market_open_on's own
        degrade-on-failure contract). Paginated via the shared `_paginate`
        helper (this TR's continuation keys are bare CTX_AREA_FK/NK, no
        numeric suffix — `ctx_suffix=""` matches) since KIS's own sample
        recurses on tr_cont "M"/"F" here too — a single unpaginated call
        risked returning a multi-day calendar page where `day` wasn't
        actually the first row. `bass_dt` is checked on every row instead of
        blindly trusting rows[0], for the same reason."""
        params = {"BASS_DT": day, "CTX_AREA_NK": "", "CTX_AREA_FK": ""}
        rows: list[dict] = []
        async for body in self._paginate(
            "GET", _HOLIDAY_PATH, _HOLIDAY_TR_ID, params, what=f"holiday check {day}", ctx_suffix="",
        ):
            page_rows = body.get("output") or []
            rows.extend(page_rows if isinstance(page_rows, list) else [page_rows])
        matched = [row for row in rows if row.get("bass_dt") == day]
        if not matched:
            raise KISBrokerError(f"holiday check {day} returned no matching bass_dt row")
        return matched[0].get("opnd_yn") == "Y"

    async def fetch_investor_trend(self, code: str) -> list[InvestorFlow]:
        """종목별 투자자 매매동향 — daily individual/foreign/institution net-buy
        for `code`, most-recent first as KIS returns it. This is the Korean
        market's core supply-demand signal and has no yfinance equivalent."""
        resp = await self._request(
            "GET", _INVESTOR_TREND_PATH, _INVESTOR_TREND_TR_ID,
            params={"FID_COND_MRKT_DIV_CODE": "J", "FID_INPUT_ISCD": code},
        )
        body = self._checked_body(resp, f"investor trend for {code}")
        rows = body.get("output") or body.get("output1") or []
        return [
            InvestorFlow(
                day=row.get("stck_bsop_date", ""),
                close=_num(row.get("stck_clpr")),
                individual_net=int(_num(row.get("prsn_ntby_qty"))),
                foreign_net=int(_num(row.get("frgn_ntby_qty"))),
                institution_net=int(_num(row.get("orgn_ntby_qty"))),
            )
            for row in rows
            if row.get("stck_bsop_date")
        ]

    async def fetch_investor_estimate(self, code: str) -> list[InvestorEstimate]:
        """종목별 외국인/기관 추정 가집계 — today's intraday estimated net-buy.
        Only meaningful during / just after a live session (it's an estimate
        for the current day)."""
        resp = await self._request(
            "GET", _INVESTOR_ESTIMATE_PATH, _INVESTOR_ESTIMATE_TR_ID,
            params={"MKSC_SHRN_ISCD": code},
        )
        body = self._checked_body(resp, f"investor estimate for {code}")
        rows = body.get("output2") or body.get("output") or body.get("output1") or []
        if isinstance(rows, dict):
            rows = [rows]
        return [
            InvestorEstimate(
                slot=row.get("bsop_hour_gb", ""),
                foreign_net=int(_num(row.get("frgn_fake_ntby_qty"))),
                institution_net=int(_num(row.get("orgn_fake_ntby_qty"))),
                total_net=int(_num(row.get("sum_fake_ntby_qty"))),
            )
            for row in rows
        ]

    async def fetch_dividend_schedule(self, code: str, start: str, end: str) -> list[DividendInfo]:
        """예탁원 배당 일정 — dividend records for `code` between `start` and
        `end` (YYYYMMDD). A corporate-action calendar yfinance doesn't give
        for Korean names."""
        resp = await self._request(
            "GET", _DIVIDEND_PATH, _DIVIDEND_TR_ID,
            params={"CTS": "", "GB1": "0", "F_DT": start, "T_DT": end, "SHT_CD": code, "HIGH_GB": ""},
        )
        body = self._checked_body(resp, f"dividend schedule for {code}")
        rows = body.get("output1") or body.get("output") or body.get("output2") or []
        return [
            DividendInfo(
                record_date=row.get("record_date", ""),
                kind=(row.get("divi_kind") or "").strip(),
                cash_amount=_num(row.get("per_sto_divi_amt")),
                cash_rate=_num(row.get("divi_rate")),
                pay_date=row.get("divi_pay_dt", ""),
            )
            for row in rows
            if row.get("record_date")
        ]

    async def fetch_news_titles(self, code: str) -> list[NewsTitle]:
        """종목 뉴스 제목 — recent news headlines for `code`. Titles are
        third-party free text (the classic prompt-injection surface), so the
        caller must wrap them before feeding an LLM."""
        resp = await self._request(
            "GET", _NEWS_TITLE_PATH, _NEWS_TITLE_TR_ID,
            params={
                "FID_NEWS_OFER_ENTP_CODE": "",
                "FID_COND_MRKT_CLS_CODE": "",
                "FID_INPUT_ISCD": code,
                "FID_TITL_CNTT": "",
                "FID_INPUT_DATE_1": "",
                "FID_INPUT_HOUR_1": "",
                "FID_RANK_SORT_CLS_CODE": "",
                "FID_INPUT_SRNO": "",
            },
        )
        body = self._checked_body(resp, f"news titles for {code}")
        return [
            NewsTitle(
                day=row.get("data_dt", ""),
                time=row.get("data_tm", ""),
                title=(row.get("hts_pbnt_titl_cntt") or "").strip(),
                source=(row.get("dorg") or "").strip(),
            )
            for row in (body.get("output1") or body.get("output") or [])
            if (row.get("hts_pbnt_titl_cntt") or "").strip()
        ]

    async def fetch_daily_short_sale(self, code: str, start: str, end: str) -> list[ShortSaleDay]:
        """종목별 일별 공매도 — daily short-sell volume and its share of turnover
        for `code` between `start` and `end` (YYYYMMDD). The per-day series is
        in output2 (output1 is a current-price summary)."""
        resp = await self._request(
            "GET", _SHORT_SALE_PATH, _SHORT_SALE_TR_ID,
            params={
                "FID_COND_MRKT_DIV_CODE": "J",
                "FID_INPUT_ISCD": code,
                "FID_INPUT_DATE_1": start,
                "FID_INPUT_DATE_2": end,
            },
        )
        body = self._checked_body(resp, f"daily short sale for {code}")
        return [
            ShortSaleDay(
                day=row.get("stck_bsop_date", ""),
                close=_num(row.get("stck_clpr")),
                short_volume=int(_num(row.get("ssts_cntg_qty"))),
                short_ratio=_num(row.get("ssts_vol_rlim")),
                accum_short_ratio=_num(row.get("acml_ssts_cntg_qty_rlim")),
            )
            for row in body.get("output2", [])
            if row.get("stck_bsop_date")
        ]

    async def fetch_program_trade(self, code: str) -> list[ProgramTrade]:
        """종목별 프로그램매매 동향 — program (index-arbitrage/basket) net-buy
        for `code`. Program flow is a distinct pressure from ordinary
        investor flow, so it's a separate risk-side signal."""
        resp = await self._request(
            "GET", _PROGRAM_TRADE_PATH, _PROGRAM_TRADE_TR_ID,
            params={"FID_COND_MRKT_DIV_CODE": "J", "FID_INPUT_ISCD": code},
        )
        body = self._checked_body(resp, f"program trade for {code}")
        rows = body.get("output") or body.get("output1") or []
        return [
            ProgramTrade(
                slot=row.get("bsop_hour", ""),
                net_buy_qty=int(_num(row.get("whol_smtn_ntby_qty"))),
                net_buy_amount=_num(row.get("whol_smtn_ntby_tr_pbmn")),
            )
            for row in rows
        ]

    async def fetch_invest_opinion(self, code: str, start: str, end: str) -> list[InvestmentOpinion]:
        """종목 투자의견 — domestic brokerage opinions/target prices for `code`
        between `start` and `end` (YYYYMMDD). yfinance's analyst consensus is
        near-empty for Korean names, which this fills."""
        resp = await self._request(
            "GET", _INVEST_OPINION_PATH, _INVEST_OPINION_TR_ID,
            params={
                "FID_COND_MRKT_DIV_CODE": "J",
                "FID_COND_SCR_DIV_CODE": "16633",
                "FID_INPUT_ISCD": code,
                "FID_INPUT_DATE_1": start,
                "FID_INPUT_DATE_2": end,
            },
        )
        body = self._checked_body(resp, f"invest opinion for {code}")
        return [
            InvestmentOpinion(
                day=row.get("stck_bsop_date", ""),
                opinion=row.get("invt_opnn", "").strip(),
                prev_opinion=row.get("rgbf_invt_opnn", "").strip(),
                target_price=_num(row.get("hts_goal_prc")),
                prev_close=_num(row.get("stck_prdy_clpr")),
            )
            for row in (body.get("output") or body.get("output1") or [])
            if row.get("stck_bsop_date")
        ]
