import logging
import re

from app.brokers.kis.parser import _num
from app.domain.models import (
    BalanceSheet,
    EarningsEstimate,
    FinancialRatio,
    GrowthRatioDetail,
    IncomeStatement,
    OtherMajorRatio,
    ProfitabilityRatio,
    StabilityRatio,
)

# 국내 재무 (한국 종목은 yfinance 재무가 빈약 — KIS가 더 정확·풍부). 모의 지원 확인됨.
_FIN_INCOME_TR_ID = "FHKST66430200"  # 국내주식 손익계산서
_FIN_INCOME_PATH = "/uapi/domestic-stock/v1/finance/income-statement"
_FIN_BALANCE_TR_ID = "FHKST66430100"  # 국내주식 재무상태표
_FIN_BALANCE_PATH = "/uapi/domestic-stock/v1/finance/balance-sheet"
_FIN_RATIO_TR_ID = "FHKST66430300"  # 국내주식 재무비율
_FIN_RATIO_PATH = "/uapi/domestic-stock/v1/finance/financial-ratio"
_ESTIMATE_TR_ID = "HHKST668300C0"  # 국내주식 종목추정실적
_ESTIMATE_PATH = "/uapi/domestic-stock/v1/quotations/estimate-perform"
# Four more granular ratio TRs, confirmed live 2026-08-04 — each only
# carries fields FinancialRatio above doesn't already have (see each
# dataclass's own comment in app.domain.models.market).
_STABILITY_RATIO_TR_ID = "FHKST66430600"  # 국내주식 안정성비율
_STABILITY_RATIO_PATH = "/uapi/domestic-stock/v1/finance/stability-ratio"
_PROFIT_RATIO_TR_ID = "FHKST66430400"  # 국내주식 수익성비율
_PROFIT_RATIO_PATH = "/uapi/domestic-stock/v1/finance/profit-ratio"
_GROWTH_RATIO_TR_ID = "FHKST66430800"  # 국내주식 성장성비율
_GROWTH_RATIO_PATH = "/uapi/domestic-stock/v1/finance/growth-ratio"
_OTHER_MAJOR_RATIOS_TR_ID = "FHKST66430500"  # 국내주식 기타주요비율
_OTHER_MAJOR_RATIOS_PATH = "/uapi/domestic-stock/v1/finance/other-major-ratios"

logger = logging.getLogger(__name__)


class DomesticFinanceMixin:
    async def fetch_kr_income_statement(self, code: str, div_cls: str = "0") -> list[IncomeStatement]:
        """국내주식 손익계산서 — revenue/profit lines by settlement period,
        most recent first. `div_cls` "0" is the annual series."""
        resp = await self._request(
            "GET", _FIN_INCOME_PATH, _FIN_INCOME_TR_ID,
            params={"FID_DIV_CLS_CODE": div_cls, "fid_cond_mrkt_div_code": "J", "fid_input_iscd": code},
        )
        body = self._checked_body(resp, f"income statement for {code}")
        return [
            IncomeStatement(
                period=row.get("stac_yymm", ""),
                revenue=_num(row.get("sale_account")),
                gross_profit=_num(row.get("sale_totl_prfi")),
                operating_profit=_num(row.get("bsop_prti")),
                net_income=_num(row.get("thtr_ntin")),
            )
            for row in (body.get("output") or body.get("output1") or [])
            if row.get("stac_yymm")
        ]

    async def fetch_kr_balance_sheet(self, code: str, div_cls: str = "0") -> list[BalanceSheet]:
        """국내주식 재무상태표 — asset/liability/equity totals by period."""
        resp = await self._request(
            "GET", _FIN_BALANCE_PATH, _FIN_BALANCE_TR_ID,
            params={"FID_DIV_CLS_CODE": div_cls, "fid_cond_mrkt_div_code": "J", "fid_input_iscd": code},
        )
        body = self._checked_body(resp, f"balance sheet for {code}")
        return [
            BalanceSheet(
                period=row.get("stac_yymm", ""),
                total_assets=_num(row.get("total_aset")),
                total_liabilities=_num(row.get("total_lblt")),
                total_equity=_num(row.get("total_cptl")),
                current_assets=_num(row.get("cras")),
                current_liabilities=_num(row.get("flow_lblt")),
            )
            for row in (body.get("output") or body.get("output1") or [])
            if row.get("stac_yymm")
        ]

    async def fetch_kr_earnings_estimate(self, code: str) -> list[EarningsEstimate]:
        """국내주식 종목추정실적 — 증권사 컨센서스 추정 매출액/영업이익/순이익
        (연도 뒤 "E"가 붙은 기간은 추정치). KIS의 공식 예제 코드는 이 TR의
        응답 컬럼 매핑을 다른 API(ELW) 것과 착각해 잘못 문서화해뒀으므로,
        output2의 6개 행 순서는 문서가 아니라 삼성전자 실제 재무수치와
        대조해 검증했다. output3(추정 EPS/PER 등으로 추정됨)은 같은 방식
        으로 검증하지 못해 의도적으로 제외."""
        resp = await self._request("GET", _ESTIMATE_PATH, _ESTIMATE_TR_ID, params={"SHT_CD": code})
        body = self._checked_body(resp, f"earnings estimate for {code}")
        periods = [p.get("dt", "") for p in (body.get("output4") or [])]
        rows = body.get("output2") or []
        if len(rows) < 6 or not periods:
            return []
        # `periods` (output4) and each row's data1..dataN (output2) are two
        # separately-indexed KIS response blocks — nothing guarantees they
        # stay in lockstep. If a row has fewer dataN columns than there are
        # periods, `.get(f"data{i+1}")` on the missing ones silently fell
        # through _num to 0.0, indistinguishable from a real consensus
        # estimate of zero. Cap periods to however many data columns every
        # row of the 6 actually has, and warn instead of guessing.
        data_col_count = min(
            (sum(1 for k in row if re.fullmatch(r"data\d+", k)) for row in rows[:6]),
            default=0,
        )
        if data_col_count < len(periods):
            logger.warning(
                "earnings estimate for %s: output4 has %d periods but output2 rows only carry %d "
                "data columns — truncating to avoid fabricating zero estimates",
                code, len(periods), data_col_count,
            )
            periods = periods[:data_col_count]
        if not periods:
            return []
        return [
            EarningsEstimate(
                period=periods[i],
                revenue=_num(rows[0].get(f"data{i + 1}")),
                revenue_growth=_num(rows[1].get(f"data{i + 1}")),
                operating_profit=_num(rows[2].get(f"data{i + 1}")),
                operating_profit_growth=_num(rows[3].get(f"data{i + 1}")),
                net_income=_num(rows[4].get(f"data{i + 1}")),
                net_income_growth=_num(rows[5].get(f"data{i + 1}")),
            )
            for i in range(len(periods))
        ]

    async def fetch_kr_financial_ratio(self, code: str, div_cls: str = "0") -> list[FinancialRatio]:
        """국내주식 재무비율 — growth rates, ROE, EPS/BPS, debt ratio by period."""
        resp = await self._request(
            "GET", _FIN_RATIO_PATH, _FIN_RATIO_TR_ID,
            params={"FID_DIV_CLS_CODE": div_cls, "fid_cond_mrkt_div_code": "J", "fid_input_iscd": code},
        )
        body = self._checked_body(resp, f"financial ratio for {code}")
        return [
            FinancialRatio(
                period=row.get("stac_yymm", ""),
                revenue_growth=_num(row.get("grs")),
                op_profit_growth=_num(row.get("bsop_prfi_inrt")),
                net_income_growth=_num(row.get("ntin_inrt")),
                roe=_num(row.get("roe_val")),
                eps=_num(row.get("eps")),
                bps=_num(row.get("bps")),
                debt_ratio=_num(row.get("lblt_rate")),
            )
            for row in (body.get("output") or body.get("output1") or [])
            if row.get("stac_yymm")
        ]

    async def fetch_kr_stability_ratio(self, code: str, div_cls: str = "0") -> list[StabilityRatio]:
        """국내주식 안정성비율 — borrowing dependency/current/quick ratios,
        the balance-sheet-liquidity metrics FinancialRatio's debt_ratio
        alone doesn't cover."""
        resp = await self._request(
            "GET", _STABILITY_RATIO_PATH, _STABILITY_RATIO_TR_ID,
            params={"fid_input_iscd": code, "fid_div_cls_code": div_cls, "fid_cond_mrkt_div_code": "J"},
        )
        body = self._checked_body(resp, f"stability ratio for {code}")
        return [
            StabilityRatio(
                period=row.get("stac_yymm", ""),
                borrowing_dependency=_num(row.get("bram_depn")),
                current_ratio=_num(row.get("crnt_rate")),
                quick_ratio=_num(row.get("quck_rate")),
            )
            for row in (body.get("output") or [])
            if row.get("stac_yymm")
        ]

    async def fetch_kr_profitability_ratio(self, code: str, div_cls: str = "0") -> list[ProfitabilityRatio]:
        """국내주식 수익성비율 — margin ratios (gross/net margin, return on
        capital/equity) that FinancialRatio's plain ROE doesn't cover."""
        resp = await self._request(
            "GET", _PROFIT_RATIO_PATH, _PROFIT_RATIO_TR_ID,
            params={"fid_input_iscd": code, "FID_DIV_CLS_CODE": div_cls, "fid_cond_mrkt_div_code": "J"},
        )
        body = self._checked_body(resp, f"profitability ratio for {code}")
        return [
            ProfitabilityRatio(
                period=row.get("stac_yymm", ""),
                return_on_capital=_num(row.get("cptl_ntin_rate")),
                return_on_equity=_num(row.get("self_cptl_ntin_inrt")),
                net_margin=_num(row.get("sale_ntin_rate")),
                gross_margin=_num(row.get("sale_totl_rate")),
            )
            for row in (body.get("output") or [])
            if row.get("stac_yymm")
        ]

    async def fetch_kr_growth_ratio_detail(self, code: str, div_cls: str = "0") -> list[GrowthRatioDetail]:
        """국내주식 성장성비율 — equity/total-asset growth only (revenue/
        operating-profit growth are already covered by fetch_kr_financial_
        ratio's grs/bsop_prfi_inrt, same TR field names, so not repeated
        here to avoid two sources of truth for the same number)."""
        resp = await self._request(
            "GET", _GROWTH_RATIO_PATH, _GROWTH_RATIO_TR_ID,
            params={"fid_input_iscd": code, "fid_div_cls_code": div_cls, "fid_cond_mrkt_div_code": "J"},
        )
        body = self._checked_body(resp, f"growth ratio detail for {code}")
        return [
            GrowthRatioDetail(
                period=row.get("stac_yymm", ""),
                equity_growth=_num(row.get("equt_inrt")),
                asset_growth=_num(row.get("totl_aset_inrt")),
            )
            for row in (body.get("output") or [])
            if row.get("stac_yymm")
        ]

    async def fetch_kr_other_major_ratios(self, code: str, div_cls: str = "0") -> list[OtherMajorRatio]:
        """국내주식 기타주요비율 — payout ratio, EVA, EBITDA, EV/EBITDA.
        Confirmed live (2026-08-04, 005930): payout_rate/eva came back
        0.00 for a period where EBITDA/EV_EBITDA had real values — left
        as-is (KIS's own data, not this codebase's to second-guess), but
        a caller surfacing this to an LLM should treat an exact 0.00 as
        possibly "not provided this period" rather than a confident zero."""
        resp = await self._request(
            "GET", _OTHER_MAJOR_RATIOS_PATH, _OTHER_MAJOR_RATIOS_TR_ID,
            params={"fid_input_iscd": code, "fid_div_cls_code": div_cls, "fid_cond_mrkt_div_code": "J"},
        )
        body = self._checked_body(resp, f"other major ratios for {code}")
        return [
            OtherMajorRatio(
                period=row.get("stac_yymm", ""),
                payout_ratio=_num(row.get("payout_rate")),
                eva=_num(row.get("eva")),
                ebitda=_num(row.get("ebitda")),
                ev_ebitda=_num(row.get("ev_ebitda")),
            )
            for row in (body.get("output") or [])
            if row.get("stac_yymm")
        ]
