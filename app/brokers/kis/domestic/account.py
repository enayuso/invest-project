from app.brokers.kis.const import KISBrokerError
from app.brokers.kis.parser import _num
from app.domain.models import (
    BrokerPosition,
    BrokerBalance,
    BuyableInfo,
    SellableInfo,
)

# 주식잔고조회. Reference: open-trading-api examples
# examples_llm/domestic_stock/inquire_balance/ (TR TTTC8434R real / VTTC8434R
# paper, same path, INQR_DVSN=02 종목별, tr_cont M/F 연속조회). The position-
# weight risk gate divides by output2.tot_evlu_amt (총평가금액). An alternative
# asset-weight source is inquire_account_balance (TR CTRP6548R, HTS [0891]
# 자산비중) if a category breakdown is ever needed.
_BALANCE_TR_ID_REAL = "TTTC8434R"  # 주식잔고조회 (실전투자)
_BALANCE_TR_ID_PAPER = "VTTC8434R"  # 주식잔고조회 (모의투자)
_BALANCE_PATH = "/uapi/domestic-stock/v1/trading/inquire-balance"

_BUYABLE_TR_ID_REAL = "TTTC8908R"  # 매수가능조회 (실전투자)
_BUYABLE_TR_ID_PAPER = "VTTC8908R"  # 매수가능조회 (모의투자)
_BUYABLE_PATH = "/uapi/domestic-stock/v1/trading/inquire-psbl-order"
# 매도가능수량조회. Real TR confirmed (TTTC8408R, examples_llm/domestic_stock/
# inquire_psbl_sell — that file hardcodes the real ID with no env_dv branch,
# no paper ID documented there). The paper ID here (VTTC8408R) follows KIS's
# usual V-prefix convention but isn't independently confirmed in the
# examples — kept since it's what this project's own paper-server testing
# already used (see kis-integration-status memory: this TR itself is
# 모의 미지원/EGW02006 on paper regardless of the exact ID, confirmed live).
_SELLABLE_TR_ID_REAL = "TTTC8408R"
_SELLABLE_TR_ID_PAPER = "VTTC8408R"
_SELLABLE_PATH = "/uapi/domestic-stock/v1/trading/inquire-psbl-sell"


class DomesticAccountMixin:
    async def fetch_balance(self) -> BrokerBalance:
        params = {
            "CANO": self._account_8,
            "ACNT_PRDT_CD": self._account_prod,
            # "Y" prices domestic holdings off the time-outside single-price
            # (시간외단일가) instead of the regular session price — KIS's
            # official spec value for this field is "N" outside that narrow
            # use case, and no comment here ever justified deviating from it.
            "AFHR_FLPR_YN": "N",
            "OFL_YN": "",
            "INQR_DVSN": "02",  # 종목별
            "UNPR_DVSN": "02",
            "FUND_STTL_ICLD_YN": "N",
            "FNCG_AMT_AUTO_RDPT_YN": "N",
            "PRCS_DVSN": "01",  # 전일매매포함 / 01:당일매매포함 / 02:당일매매미포함
            "CTX_AREA_FK100": "",
            "CTX_AREA_NK100": "",
        }

        positions: list[BrokerPosition] = []
        summary: dict | None = None

        async for body in self._paginate(
            "GET", _BALANCE_PATH, self._tr(_BALANCE_TR_ID_REAL, _BALANCE_TR_ID_PAPER), params,
            what="balance inquiry", priority="high", lane="account",
        ):
            positions.extend(self._parse_position(row) for row in body.get("output1", []))
            if body.get("output2"):
                summary = body["output2"][0]

        if summary is None:
            raise KISBrokerError("balance inquiry returned no output2 summary")

        # Zero-quantity rows linger for D-2 days after a full sell; drop them.
        positions = [p for p in positions if p.quantity > 0]
        return BrokerBalance(
            cash_deposit=_num(summary.get("dnca_tot_amt")),
            # Summed from positions, not output2's tot_evlu_amt — that field
            # is cash_deposit + securities value combined (총자산), not the
            # pure holdings evaluation the "총평가금액" tile is meant to show.
            # 총평가금액=보유종목 평가금액 합계(예수금 미포함), 원본 tot_evlu_amt는 총자산 개념이라 배제
            total_eval_amount=sum(p.eval_amount for p in positions),
            total_pnl_amount=_num(summary.get("evlu_pfls_smtl_amt")),
            purchase_amount=_num(summary.get("pchs_amt_smtl_amt")),
            # D+2예수금 (가수도정산금액) — cash actually withdrawable/orderable
            # once T+2 settlement clears, distinct from same-day dnca_tot_amt.
            d2_cash_deposit=_num(summary.get("prvs_rcdl_excc_amt")),
            positions=positions,
        )

    @staticmethod
    def _parse_position(row: dict) -> BrokerPosition:
        return BrokerPosition(
            code=row.get("pdno", ""),
            name=row.get("prdt_name", ""),
            quantity=int(_num(row.get("hldg_qty"))),
            avg_price=_num(row.get("pchs_avg_pric")),
            current_price=_num(row.get("prpr")),
            eval_amount=_num(row.get("evlu_amt")),
            pnl_amount=_num(row.get("evlu_pfls_amt")),
            pnl_pct=_num(row.get("evlu_pfls_rt")),
        )

    async def fetch_buyable(self, code: str, price: float | None = None) -> BuyableInfo:
        """매수가능조회 — how much cash / how many shares of `code` the account
        can buy right now. `price=None` asks against a market order."""
        params = {
            "CANO": self._account_8,
            "ACNT_PRDT_CD": self._account_prod,
            "PDNO": code,
            "ORD_UNPR": str(int(price)) if price else "0",
            "ORD_DVSN": "00" if price else "01",  # 00 지정가 / 01 시장가
            "CMA_EVLU_AMT_ICLD_YN": "N",
            "OVRS_ICLD_YN": "N",
        }
        resp = await self._request(
            "GET", _BUYABLE_PATH, self._tr(_BUYABLE_TR_ID_REAL, _BUYABLE_TR_ID_PAPER), params=params,
            priority="high", lane="account",
        )
        body = self._checked_body(resp, "buyable inquiry")
        out = body.get("output", {})
        # max_buy_* is the leveraged ceiling; nrcvb_buy_* is the cash-only
        # (미수 없는) figure — the conservative one is what a safety check wants.
        return BuyableInfo(
            cash=_num(out.get("ord_psbl_cash")),
            max_qty=int(_num(out.get("nrcvb_buy_qty") or out.get("max_buy_qty"))),
            max_amount=_num(out.get("nrcvb_buy_amt") or out.get("max_buy_amt")),
        )

    async def fetch_sellable(self, code: str) -> SellableInfo:
        """매도가능수량조회 — how many shares of `code` can be sold right now.

        `ord_psbl_qty` already nets out shares locked in pending/unsettled
        sells, so it is the safe ceiling for a sell order; the holding
        quantity from a balance inquiry is NOT (it can double-count shares an
        earlier sell order already committed).

        Unwired: app.execution.kis_bridge's actual sell-ceiling gate uses
        balance-derived logic instead (see domestic_sell_over_held), since
        this TR is unverified against the paper server. Kept as a real,
        available capability for future use, not deleted.
        """
        params = {
            "CANO": self._account_8,
            "ACNT_PRDT_CD": self._account_prod,
            "PDNO": code,
        }
        resp = await self._request(
            "GET", _SELLABLE_PATH, self._tr(_SELLABLE_TR_ID_REAL, _SELLABLE_TR_ID_PAPER), params=params,
            priority="high", lane="account",
        )
        body = self._checked_body(resp, f"sellable inquiry for {code}")
        out = body.get("output", {})
        if isinstance(out, list):  # some TRs wrap the single row in a list
            out = out[0] if out else {}
        return SellableInfo(
            sellable_qty=int(_num(out.get("ord_psbl_qty"))),
            holding_qty=int(_num(out.get("cblc_qty"))),
            already_sold_qty=int(_num(out.get("sll_qty"))),
        )
