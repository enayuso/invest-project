# app/domain/models/market.py
# Quote/candle/ranking/fundamentals domain models (broker-agnostic).
# 시세/캔들/순위/재무 도메인 모델

from dataclasses import dataclass


@dataclass
class StockStatus:
    """Per-stock trading-halt / administrative-issue status (주식기본조회).

    Distinct from a market-wide circuit breaker (KOSPI/KOSDAQ-level halt),
    which KIS only exposes over a real-time WebSocket subscription
    (market-operation info, H0STMKO0) — there is no REST "is a CB active"
    call. This is the cheaper, per-stock counterpart: administrative/
    disclosure-driven halts, queryable on demand."""

    code: str  # 종목번호 (pdno)
    trading_halted: bool  # 거래정지여부 (tr_stop_yn)
    administrative_issue: bool  # 관리종목여부 (admn_item_yn) — KRX 관리종목 지정
    nxt_trading_halted: bool  # NXT 거래정지여부 (nxt_tr_stop_yn)
    korean_name: str = ""  # 상품약어명, 실전계좌 한정(prdt_abrv_name) — 모의투자는 EGW02006으로 이 TR 전체를 거부
    english_name: str = ""  # 상품영문약어명, 실전계좌 한정(prdt_eng_abrv_name)
    security_group_code: str = ""  # 증권그룹ID코드 (scty_grp_id_cd), 실전계좌 한정 — ST=주권,
    # EF=국내ETF, FE=해외ETF, MF=증권투자회사(뮤추얼펀드), RT=리츠 등. yfinance의 quoteType이
    # 국내 소형주를 종종 MUTUALFUND로 오분류하는 문제(2026-09-03, 228340.KS/142760.KS 확인)의
    # 근본 수정: 이 코드가 실제 상품유형의 권위 있는 출처 — app.dataflows.kis_provider가
    # 이 값으로 yfinance quote_type을 덮어쓴다.


@dataclass
class StockDesignations:
    """KRX pre-screening designation flags (주식현재가 시세, FHKST01010100) —
    NOT an execution-time gate (see app.execution.kis_bridge.designation_block's
    own docstring: this is a cost-saving pre-filter, never added to
    execution_blockers). All four ride along in the same response
    fetch_market_warning/fetch_price/fetch_valuation/fetch_sector_name
    already pay for — zero extra network round-trips.

    2026-09-08 screening audit: of 265 domestic Hold decisions in the prior
    30 days, 3.0% carried an active market-warning designation and another
    7.6% carried a short-overheat/liquidation/administrative designation —
    all four predictable from this one TR response without spending an LLM
    analysis. Low-liquidity designation (`low_current_yn`, the largest slice
    at 8.7%) is deliberately NOT here — that field lives on a DIFFERENT TR
    (주식현재가 시세2, FHPST01010000) this codebase does not yet call; adding
    it means a second network round-trip and is scoped out for now.

    `administrative_raw` is kept as the RAW code, not a bool, because KIS's
    reference implementation documents this field's Korean label as "관리
    종목여부" (a yes/no question) while its own field name carries a
    `_cls_code` suffix (a classification code, like `mrkt_warn_cls_code`) —
    a genuine contradiction with no enum documented anywhere. Guessing the
    "clean" sentinel wrong here doesn't just miss a designation, it drops
    every non-designated candidate (the same class of risk that made
    market_warning's own rollout ship detect-only until a live hit
    confirmed its convention — see fetch_market_warning's docstring)."""

    code: str
    market_warning: str = ""  # 시장경고코드 (mrkt_warn_cls_code) — "" = none, KIS does not document a stable non-"" sentinel
    short_overheat: bool = False  # 단기과열여부 (short_over_yn == "Y")
    liquidation: bool = False  # 정리매매여부 (sltr_yn == "Y")
    administrative_raw: str = ""  # 관리종목여부 (mang_issu_cls_code), RAW — enum direction unconfirmed, see class docstring


@dataclass
class BidAsk:
    """Best bid/ask (1호가) — for a liquidity/spread pre-filter, not order
    execution (KIS market orders don't need a quoted price at all; this is
    purely a "how wide is this ticker's spread right now" signal)."""

    code: str
    bid: float  # 매수호가1 (best bid, domestic bidp1 / overseas pbid1)
    ask: float  # 매도호가1 (best ask, domestic askp1 / overseas pask1)
    last: float  # 현재가/최근 체결가 (domestic stck_prpr / overseas last)


@dataclass
class VIStatus:
    code: str  # 유가증권 단축 종목코드 (mksc_shrn_iscd)
    name: str  # HTS 한글 종목명 (hts_kor_isnm)
    vi_cls_code: str  # VI 발동상태 (vi_cls_code)
    vi_kind_code: str  # VI 종류코드 — 정적/동적 (vi_kind_code)
    triggered_at: str  # VI 발동시간 HHMMSS (cntg_vi_hour)
    released_at: str  # VI 해제시간 HHMMSS (vi_cncl_hour)
    vi_price: float  # VI 발동가격 (vi_prc)
    vi_count: int  # VI 발동횟수 (vi_count)


@dataclass
class InvestorFlow:
    """One day of 종목별 투자자 순매수 (net-buy share counts, + = net buyer)."""
    day: str  # 주식 영업일자 (stck_bsop_date, YYYYMMDD)
    close: float  # 종가 (stck_clpr)
    individual_net: int  # 개인 순매수수량 (prsn_ntby_qty)
    foreign_net: int  # 외국인 순매수수량 (frgn_ntby_qty)
    institution_net: int  # 기관계 순매수수량 (orgn_ntby_qty)


@dataclass
class InvestorEstimate:
    """당일 외국인/기관 추정 가집계 순매수 (intraday, estimated)."""
    slot: str  # 입력구분/추정시점 (bsop_hour_gb)
    foreign_net: int  # 외국인 가집계 순매수수량 (frgn_fake_ntby_qty)
    institution_net: int  # 기관 가집계 순매수수량 (orgn_fake_ntby_qty)
    total_net: int  # 합산 가집계 순매수수량 (sum_fake_ntby_qty)


@dataclass
class IncomeStatement:
    period: str  # 결산년월 (stac_yymm, YYYYMM)
    revenue: float  # 매출액 (sale_account)
    gross_profit: float  # 매출총이익 (sale_totl_prfi)
    operating_profit: float  # 영업이익 (bsop_prti)
    net_income: float  # 당기순이익 (thtr_ntin)


@dataclass
class BalanceSheet:
    period: str  # 결산년월 (stac_yymm)
    total_assets: float  # 자산총계 (total_aset)
    total_liabilities: float  # 부채총계 (total_lblt)
    total_equity: float  # 자본총계 (total_cptl)
    current_assets: float  # 유동자산 (cras)
    current_liabilities: float  # 유동부채 (flow_lblt)


@dataclass
class FinancialRatio:
    period: str  # 결산년월 (stac_yymm)
    revenue_growth: float  # 매출액증가율 (grs)
    op_profit_growth: float  # 영업이익증가율 (bsop_prfi_inrt)
    net_income_growth: float  # 순이익증가율 (ntin_inrt)
    roe: float  # ROE (roe_val)
    eps: float  # EPS (eps)
    bps: float  # BPS (bps)
    debt_ratio: float  # 부채비율 (lblt_rate)


# The four dataclasses below are separate, more granular KIS ratio TRs
# (finance/stability-ratio, profit-ratio, growth-ratio, other-major-ratios)
# — confirmed live 2026-08-04 to exist alongside (not duplicating)
# FinancialRatio above. Each only carries fields FinancialRatio doesn't
# already have (e.g. GrowthRatio below omits revenue/op-profit growth,
# already covered above under the same KIS field names grs/bsop_prfi_inrt).


@dataclass
class StabilityRatio:
    period: str  # 결산년월 (stac_yymm)
    borrowing_dependency: float  # 차입금의존도 % (bram_depn)
    current_ratio: float  # 유동비율 % (crnt_rate)
    quick_ratio: float  # 당좌비율 % (quck_rate)


@dataclass
class ProfitabilityRatio:
    period: str  # 결산년월 (stac_yymm)
    return_on_capital: float  # 총자본순이익율 % (cptl_ntin_rate)
    return_on_equity: float  # 자기자본순이익율 % (self_cptl_ntin_inrt)
    net_margin: float  # 매출액순이익율 % (sale_ntin_rate)
    gross_margin: float  # 매출액총이익율 % (sale_totl_rate)


@dataclass
class GrowthRatioDetail:
    period: str  # 결산년월 (stac_yymm)
    equity_growth: float  # 자기자본증가율 % (equt_inrt)
    asset_growth: float  # 총자산증가율 % (totl_aset_inrt)


@dataclass
class OtherMajorRatio:
    period: str  # 결산년월 (stac_yymm)
    payout_ratio: float  # 배당성향 % (payout_rate)
    eva: float  # EVA, 억원 (eva)
    ebitda: float  # EBITDA, 억원 (ebitda)
    ev_ebitda: float  # EV/EBITDA 배수 (ev_ebitda)


@dataclass
class EarningsEstimate:
    """One period's 증권사 컨센서스 추정실적 (estimate_perform output2) — field
    order confirmed against Samsung Electronics' actual reported figures
    (KIS's own example code ships a copy-pasted/wrong column mapping for
    this TR, so this is verified against real numbers, not the docs)."""
    period: str  # 결산년월, 추정 연도는 "E" 접미사 (output4.dt, e.g. "2026.12E")
    revenue: float  # 매출액, 억원 (output2 row0.dataN)
    revenue_growth: float  # 매출액증가율% (output2 row1.dataN)
    operating_profit: float  # 영업이익, 억원 (output2 row2.dataN)
    operating_profit_growth: float  # 영업이익증가율% (output2 row3.dataN)
    net_income: float  # 순이익, 억원 (output2 row4.dataN)
    net_income_growth: float  # 순이익증가율% (output2 row5.dataN)


@dataclass
class InvestmentOpinion:
    """One dated 국내 증권사 투자의견 record for a stock."""
    day: str  # 주식영업일자 (stck_bsop_date, YYYYMMDD)
    opinion: str  # 투자의견 (invt_opnn, e.g. "매수"/"중립")
    prev_opinion: str  # 직전투자의견 (rgbf_invt_opnn)
    target_price: float  # HTS목표가격 (hts_goal_prc)
    prev_close: float  # 주식전일종가 (stck_prdy_clpr)


@dataclass
class ShortSaleDay:
    """One day of 종목별 공매도 (short-selling)."""
    day: str  # 주식영업일자 (stck_bsop_date)
    close: float  # 종가 (stck_clpr)
    short_volume: int  # 공매도체결수량 (ssts_cntg_qty)
    short_ratio: float  # 당일 공매도 거래량 비중 % (ssts_vol_rlim)
    accum_short_ratio: float  # 누적 공매도 비중 % (acml_ssts_cntg_qty_rlim)


@dataclass
class ProgramTrade:
    """종목별 프로그램매매 (전체 합계 순매수)."""
    slot: str  # 영업시간/구분 (bsop_hour)
    net_buy_qty: int  # 전체합계 순매수수량 (whol_smtn_ntby_qty)
    net_buy_amount: float  # 전체합계 순매수 거래대금 (whol_smtn_ntby_tr_pbmn)


@dataclass
class NewsTitle:
    """One 종목 뉴스 headline. `title` is third-party free text — the caller
    must treat it as untrusted (wrap before it enters an LLM prompt)."""
    day: str  # 작성일자 (data_dt, YYYYMMDD)
    time: str  # 작성시간 (data_tm, HHMMSS)
    title: str  # HTS 공시 제목 내용 (hts_pbnt_titl_cntt)
    source: str  # 자료원 (dorg)


@dataclass
class DividendInfo:
    """예탁원 배당 일정 (corporate action)."""
    record_date: str  # 기준일 (record_date)
    kind: str  # 배당종류 (divi_kind)
    cash_amount: float  # 현금배당금 (per_sto_divi_amt)
    cash_rate: float  # 현금배당률 % (divi_rate)
    pay_date: str  # 배당금지급일 (divi_pay_dt)


@dataclass
class IndexQuote:
    """지수 현재가 — market-regime context. Shared shape for both the
    domestic (KOSPI/KOSDAQ, DomesticInfoMixin.fetch_index_price) and
    overseas (Dow/S&P500/Nasdaq, OverseasInfoMixin.fetch_overseas_index_
    quote) index snapshots — same fields, different TR/field-name mapping
    underneath."""
    code: str  # domestic: 0001 KOSPI / 1001 KOSDAQ. overseas: .DJI/SPX/NDX/COMP
    name: str
    value: float  # 지수 현재가
    change: float  # 전일대비 (signed)
    change_pct: float  # 전일대비율 % (signed)


@dataclass
class IndexDailyBar:
    """One day of a market index's historical close (국내주식업종기간별시세,
    일봉) — the KOSPI/KOSDAQ time series a trend/moving-average signal
    (e.g. Faber's 10-month rule) needs; IndexQuote above is only a
    same-day snapshot with no history."""
    day: str  # 영업 일자 (stck_bsop_date), YYYYMMDD
    close: float  # 업종 지수 현재가 (bstp_nmix_prpr)


@dataclass
class RankedStock:
    """One row of a ranking screen (거래량순위 등)."""
    rank: int  # data_rank
    code: str  # mksc_shrn_iscd
    name: str  # hts_kor_isnm
    price: float  # stck_prpr
    change_pct: float  # prdy_ctrt (signed)
    volume: int  # acml_vol


@dataclass
class SectorIndex:
    """업종별 지수 한 줄 (sector regime)."""
    code: str  # 업종코드 (bstp_cls_code)
    name: str  # 업종명 (hts_kor_isnm)
    value: float  # 지수 (bstp_nmix_prpr)
    change_pct: float  # 전일대비율 % (bstp_nmix_prdy_ctrt)


@dataclass
class MinuteCandle:
    time: str  # 체결시각 HHMMSS (stck_cntg_hour)
    open: float
    high: float
    low: float
    close: float
    volume: int


@dataclass
class Candle:
    day: str  # YYYYMMDD (stck_bsop_date)
    open: float
    high: float
    low: float
    close: float
    volume: int


@dataclass
class StockValuation:
    """Live valuation multiples from 주식현재가 시세 (same TR fetch_price
    already calls) — per/pbr/eps/bps/market_cap were present in the
    response but unparsed until this was added."""
    code: str
    price: float  # 주식 현재가 (stck_prpr)
    per: float  # PER (per)
    pbr: float  # PBR (pbr)
    eps: float  # EPS (eps)
    bps: float  # BPS (bps)
    market_cap: float  # 시가총액 (hts_avls), 단위: 억원 — live-verified against 005930
    # (market_cap / price ≈ actual shares outstanding only when treated as 억원)


@dataclass
class OverseasValuation:
    """해외주식 현재가상세 (KIS TR HHDFS76200200) — per/pbr/market-cap/52wk
    hi-lo in one call; separate dataclass from StockValuation since the
    market-cap unit and available fields (52wk hi/lo, no eps/bps parsed
    here) differ from the domestic TR."""
    symbol: str
    price: float  # 현재가 (last)
    per: float  # PER (perx)
    pbr: float  # PBR (pbrx)
    market_cap: float  # 시가총액 (tomv), 단위: 통화(curr) 그대로 — 억 단위 아님
    week52_high: float  # 52주최고가 (h52p)
    week52_low: float  # 52주최저가 (l52p)
