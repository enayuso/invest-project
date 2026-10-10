# app/domain/models/account.py
# Account/order/execution domain models (broker-agnostic).
# 계좌/주문/체결 도메인 모델

from dataclasses import dataclass, field


@dataclass(frozen=True)
class KISCredentials:
    """A decrypted-shape credential set for one (user, account_mode).

    Moved here (2026-08-07 single-account rollback, see
    docs/single-user-rollback-guide.md) from app.credentials.store,
    which built this from a per-user encrypted Postgres row; the shape is
    unchanged so app.brokers.kis.broker.broker_from_credentials and every
    other consumer needed no changes — only Settings.kis_credentials() now
    constructs one, straight from env-sourced fields.

    Short-lived by intention — hold it for the duration of building a broker,
    don't stash it somewhere long-lived.

    No __repr__ override: dataclass reprs would print the secret, so this
    class defines one that doesn't. See below."""

    user_id: str
    account_mode: str
    app_key: str
    app_secret: str
    account_8: str
    account_prod: str
    hts_id: str | None

    @property
    def is_real(self) -> bool:
        return self.account_mode == "real"

    def __repr__(self) -> str:
        # The default dataclass repr puts app_secret in every log line, stack
        # trace, and pytest assertion diff that touches one of these. Real
        # incident class, not hypothetical: an exception carrying this object
        # is exactly how secrets end up in a log aggregator.
        # 기본 dataclass repr은 시크릿을 그대로 출력하므로 재정의 — 예외/로그 경유로
        # 유출되는 가장 흔한 경로임.
        return (
            f"KISCredentials(user_id={self.user_id!r}, account_mode={self.account_mode!r}, "
            f"account_8=***, app_key=***, app_secret=***, "
            f"hts_id={'set' if self.hts_id else None})"
        )


@dataclass
class BrokerPosition:
    code: str  # 6-digit KRX code, e.g. "005930"
    name: str  # Korean instrument name as returned by KIS
    quantity: int
    avg_price: float
    current_price: float
    eval_amount: float
    pnl_amount: float
    pnl_pct: float


@dataclass
class BrokerBalance:
    cash_deposit: float  # 예수금총금액 (dnca_tot_amt)
    total_eval_amount: float  # 총평가금액 (tot_evlu_amt)
    total_pnl_amount: float  # 평가손익합계 (evlu_pfls_smtl_amt)
    purchase_amount: float  # 매입금액합계 (pchs_amt_smtl_amt)
    # D+2 정산 예수금 (가수도정산금액, prvs_rcdl_excc_amt) — domestic only;
    # overseas fetch_overseas_balance leaves this 0.0 (no D+2 concept there).
    # 국내 D+2예수금
    d2_cash_deposit: float = 0.0
    positions: list[BrokerPosition] = field(default_factory=list)

    def held_quantity(self, code: str) -> int:
        """Quantity currently held for `code` (a KRX code or a US symbol,
        whichever this balance's `positions` are keyed by), or 0 if not
        held. Was independently reimplemented as a linear scan in
        app.execution.kis_bridge (`_held_domestic_quantity`/
        `_held_overseas_quantity`) and app.execution.portfolio_risk
        (`real_shares_held`) — same lookup, same "0 means not held" default,
        just a `for`-loop in one place and `next(...)` in the other."""
        return next((p.quantity for p in self.positions if p.code == code), 0)


@dataclass
class BuyableInfo:
    cash: float  # 주문가능현금 (ord_psbl_cash)
    max_qty: int  # 최대매수수량 (max_buy_qty / nrcvb_buy_qty)
    max_amount: float  # 최대매수금액 (max_buy_amt / nrcvb_buy_amt)


@dataclass
class SellableInfo:
    sellable_qty: int  # 주문가능수량 (ord_psbl_qty) — safe max to sell right now
    holding_qty: int  # 잔고수량 (cblc_qty)
    already_sold_qty: int  # 매도수량 (sll_qty) — shares tied up in pending/settling sells


@dataclass
class OrderResult:
    order_no: str  # 주문번호 (ODNO)
    order_time: str  # 주문시각 (ORD_TMD, HHMMSS)
    message: str  # KIS 응답 메시지 (msg1)
    org_no: str = ""  # 한국거래소전송주문조직번호 (KRX_FWDG_ORD_ORGNO) — required to
    # cancel/revise this order later. The 정정취소가능주문조회 TR that would
    # otherwise supply it does not exist on the paper server, so we capture it
    # here at submission time (default "" for responses that omit it).


@dataclass
class BrokerExecution:
    order_date: str  # 주문일자 (ord_dt, YYYYMMDD)
    order_no: str  # 주문번호 (odno)
    code: str  # 종목코드 (pdno)
    name: str  # 종목명 (prdt_name)
    side: str  # "buy" | "sell"
    # Typed float (unlike BrokerPosition.quantity above, which is kept int) —
    # these three come from KIS's own execution-report TR values, which can
    # be genuinely fractional pre-settlement. BrokerPosition.quantity is
    # deliberately floored+warned at the boundary instead, specifically to
    # avoid leaking a float into order sizing (see
    # app.brokers.kis.overseas.account's own comment for the 2026-08-19
    # AXTI/GOOG/GOOGL IGW00013/IGW00007 incident this fixed). Domestic
    # parsing here still produces whole numbers.
    order_qty: float  # 주문수량
    filled_qty: float  # 총체결수량 (tot_ccld_qty)
    remaining_qty: float  # 미체결수량 (rmn_qty)
    avg_price: float  # 평균체결가 (avg_prvs)
    filled_amount: float  # 총체결금액 (tot_ccld_amt)
    # 추정제비용합계 (prsm_tlex_smtl) — commission + tax (매도 시 증권거래세 포함,
    # 매수는 세금 없이 수수료만) for this fill. KIS labels it "추정" (estimated),
    # not a certified final amount. Overseas executions always leave this 0.0 —
    # the TR currently used there (해외주식 주문체결내역) doesn't expose an
    # equivalent per-fill breakdown.
    fee_tax: float = 0.0
    # 취소여부 (cncl_yn == "Y") — KIS's own rmn_qty (미체결수량) doesn't always
    # net out to 0 once an order is cancelled/expired, so callers deciding
    # "still open" must check this too, not just remaining_qty > 0. Overseas
    # executions leave this False because the overseas 주문체결내역 TR exposes
    # no cncl_yn equivalent — but it DOES expose rejection fields, which the
    # `rejected` flag below covers, so an overseas "still open" check must
    # test `rejected`, not `cancelled`.
    # 해외는 cncl_yn이 없어 항상 False — 대신 거부(rejected)를 봐야 함.
    cancelled: bool = False
    # 거부사유(rjct_rson)/거부사유명(rjct_rson_name) — KIS ACCEPTED the request
    # but the order was REJECTED (e.g. insufficient orderable funds), so it
    # never reached the market. Such a row still reports the full quantity as
    # 미체결(nccs_qty), so "still open" checks must exclude it or a rejected
    # order lingers in the open-orders list for the whole lookback window.
    # 거부된 주문은 미체결수량이 그대로 남아 있어, 이걸 안 보면 영원히 열린 주문처럼 보인다.
    rejected: bool = False


@dataclass
class OverseasCash:
    """One currency's 외화예수금 line from the overseas present-balance TR."""
    currency: str  # 통화코드 (crcy_cd)
    # 외화예수금 — NOT frcr_dncl_amt_2 despite the field's own name: that one
    # was live-verified to always read 0.000000 on this account, so this is
    # actually computed as output3.frcr_evlu_tota / frst_bltn_exrt instead.
    # See app.brokers.kis.overseas.account.fetch_overseas_present_balance's
    # own docstring for the full reasoning.
    deposit: float
    orderable: float  # 외화 출금/주문가능금액 (frcr_drwg_psbl_amt_1)
    eval_amount: float  # 외화평가금액 (frcr_evlu_amt2)
    # False when no output2 row matched the requested currency — every field
    # above is then a zeroed placeholder, NOT a confirmed zero balance. A
    # caller that treats `matched=False` the same as a genuine zero balance
    # would silently show/act on a wrong number instead of "unknown."
    # currency 매칭 실패 시 False — 이때 나머지 필드는 실제 0원이 아니라 조회 실패의 자리값.
    matched: bool = True
