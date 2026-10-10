import logging

from app.brokers.kis.parser import _num, _order_result
from app.domain.models import BrokerExecution, OrderResult

logger = logging.getLogger(__name__)

_OVERSEAS_ORDER_TR_REAL = {"buy": "TTTT1002U", "sell": "TTTT1006U"}  # 미국 (실전)
# Paper sell is NOT "V" + the real TR id — unlike buy (TTTT1002U -> VTTT1002U,
# a simple prefix swap), KIS's own official example
# (open-trading-api/examples_llm/overseas_stock/order/order.py) shows the
# paper sell TR is a genuinely different number, VTTT1001U, not VTTT1006U.
# Confirmed live 2026-07-20: VTTT1006U got "90000000 모의투자에서는 해당업무가
# 제공되지 않습니다" even though the user could sell the same paper position
# via KIS's own mobile app — i.e. paper overseas selling itself works fine,
# this module was just calling the wrong TR id.
# 모의 매도 TR은 실전 TR에 단순히 V만 붙인 게 아님 — KIS 공식 예시 기준 VTTT1001U가 맞음.
_OVERSEAS_ORDER_TR_PAPER = {"buy": "VTTT1002U", "sell": "VTTT1001U"}  # 미국 (모의)
_OVERSEAS_ORDER_PATH = "/uapi/overseas-stock/v1/trading/order"
_OVERSEAS_EXECUTIONS_TR_REAL = "TTTS3035R"  # 해외 주문체결내역 (실전)
_OVERSEAS_EXECUTIONS_TR_PAPER = "VTTS3035R"  # 해외 주문체결내역 (모의)
_OVERSEAS_EXECUTIONS_PATH = "/uapi/overseas-stock/v1/trading/inquire-ccnl"
_OVERSEAS_REVISE_CANCEL_TR_REAL = "TTTT1004U"  # 해외주식 정정취소주문 (미국, 실전)
_OVERSEAS_REVISE_CANCEL_TR_PAPER = "VTTT1004U"  # 해외주식 정정취소주문 (미국, 모의)
_OVERSEAS_REVISE_CANCEL_PATH = "/uapi/overseas-stock/v1/trading/order-rvsecncl"


def _is_rejected(row: dict) -> bool:
    """KIS marks a refused order via 거부사유(rjct_rson)/거부사유명(rjct_rson_name)
    on the 주문체결내역 response. Deliberately conservative — only an explicitly
    non-empty, non-"0" reason counts as rejected, so a server that omits these
    fields entirely behaves exactly as before rather than hiding a genuinely
    open order. 필드가 없으면 기존 동작 그대로(주문을 숨기지 않음)."""
    reason = str(row.get("rjct_rson") or "").strip()
    reason_name = str(row.get("rjct_rson_name") or "").strip()
    return bool(reason_name) or (bool(reason) and reason != "0")


class OverseasOrderMixin:
    async def submit_overseas_order(
        self, code: str, side: str, quantity: int, price: float, exchange: str = "NASD"
    ) -> OrderResult:
        """해외주식 주문(미국) — LIMIT only: the paper server rejects market
        orders overseas, so `price` is mandatory. Same guard-free contract as
        submit_order: every safety check lives in kis_bridge."""
        if side not in _OVERSEAS_ORDER_TR_PAPER:
            raise ValueError(f"side must be 'buy' or 'sell', got {side!r}")
        if quantity <= 0:
            raise ValueError(f"quantity must be positive, got {quantity}")
        if quantity != int(quantity):
            # Final defense line: a float reaching here means an upstream layer
            # (overseas balance parsing, real_shares_held) regressed. Fail loudly
            # instead of silently flooring — a quiet floor would let that
            # regression hide again. This is exactly how KIS rejected
            # ORD_QTY="111.0" with IGW00013/IGW00007 on 2026-08-19.
            raise ValueError(f"ORD_QTY must be a whole number of shares, got {quantity!r}")
        quantity = int(quantity)
        if price is None or price <= 0:
            raise ValueError("overseas paper orders are limit-only: a positive price is required")
        body = {
            "CANO": self._account_8,
            "ACNT_PRDT_CD": self._account_prod,
            "OVRS_EXCG_CD": exchange,
            "PDNO": code,
            "ORD_QTY": str(quantity),
            "OVRS_ORD_UNPR": f"{price:.2f}",
            "CTAC_TLNO": "",
            "MGCO_APTM_ODNO": "",
            "SLL_TYPE": "00" if side == "sell" else "",
            "ORD_SVR_DVSN_CD": "0",
            "ORD_DVSN": "00",  # 지정가
        }
        tr_id = self._tr(_OVERSEAS_ORDER_TR_REAL, _OVERSEAS_ORDER_TR_PAPER)[side]
        resp = await self._request("POST", _OVERSEAS_ORDER_PATH, tr_id, json_body=body, priority="high")
        parsed = self._checked_body(resp, f"overseas {side} order for {code}")
        result = _order_result(parsed)
        logger.info(
            "KIS overseas %s order accepted: %s x%d @ %.2f (order_no=%s, %s)",
            side, code, quantity, price, result.order_no, result.message,
        )
        return result

    async def fetch_overseas_executions(self, start: str, end: str) -> list[BrokerExecution]:
        """해외주식 주문체결내역 — 전종목/전체 조회.

        `PDNO`/`OVRS_EXCG_CD` "all" sentinel differs by account mode: paper
        only accepts "" (빈 값), real requires "%" — confirmed against the
        official KIS example (examples_llm/overseas_stock/inquire_ccnl),
        whose docstring states the "" form is 모의투자 전용. Getting this
        wrong on the real branch doesn't fail loudly: `rt_cd` still comes
        back "0" with an empty `output`, which reconciliation.py then reads
        as "every overseas trade never reached KIS" (CRITICAL, including
        stop-loss exits reported as UNPROTECTED) and /api/pnl reports zero
        overseas realized P&L — both silently wrong, not an exception.

        Fee/tax fields: confirmed NOT present on this TR (TTTS3035R/
        VTTS3035R) — its full official field list has no 수수료/세금/제비용
        column of any kind (unlike the domestic counterpart's prsm_tlex_smtl).
        That data lives on 해외주식 일별거래내역/기간손익 TRs instead, not here —
        see project memory: realized-pnl-report-done. A prior TEMP DEBUG
        block logging raw rows to hunt for a fee field lived here; removed
        now that the answer is confirmed "doesn't exist on this TR"."""
        all_ = "%" if self._is_real else ""
        params = {
            "CANO": self._account_8,
            "ACNT_PRDT_CD": self._account_prod,
            "PDNO": all_,
            "ORD_STRT_DT": start,
            "ORD_END_DT": end,
            "SLL_BUY_DVSN": "00",
            "CCLD_NCCS_DVSN": "00",
            "OVRS_EXCG_CD": all_,
            "SORT_SQN": "DS",
            "ORD_DT": "",
            "ORD_GNO_BRNO": "",
            "ODNO": "",
            "CTX_AREA_NK200": "",
            "CTX_AREA_FK200": "",
        }
        executions: list[BrokerExecution] = []
        async for body in self._paginate(
            "GET", _OVERSEAS_EXECUTIONS_PATH,
            self._tr(_OVERSEAS_EXECUTIONS_TR_REAL, _OVERSEAS_EXECUTIONS_TR_PAPER), params,
            what="overseas executions inquiry", ctx_suffix="200", priority="high", lane="account",
        ):
            for row in body.get("output", []):
                executions.append(
                    BrokerExecution(
                        order_date=row.get("ord_dt", ""),
                        order_no=row.get("odno", ""),
                        code=row.get("pdno", ""),
                        name=row.get("prdt_name", ""),
                        side="sell" if row.get("sll_buy_dvsn_cd") == "01" else "buy",
                        # float, not int(...) — overseas fills can be fractional
                        # (see BrokerExecution's own field comment); truncating
                        # here made a 0.5-share fill read as filled_qty=0.
                        order_qty=_num(row.get("ft_ord_qty")),
                        filled_qty=_num(row.get("ft_ccld_qty")),
                        remaining_qty=_num(row.get("nccs_qty")),
                        avg_price=_num(row.get("ft_ccld_unpr3")),
                        filled_amount=_num(row.get("ft_ccld_amt3")),
                        # A rejected order (e.g. 주문가능금액 부족) still reports its
                        # full quantity as 미체결, so without this it reads as a
                        # permanently-open order. See BrokerExecution.rejected.
                        rejected=_is_rejected(row),
                    )
                )
        return executions

    async def cancel_overseas_order(
        self, order_no: str, code: str, quantity: int, exchange: str = "NASD"
    ) -> OrderResult:
        """해외주식 정정취소 — full cancel (전량) of an open overseas order.
        Unlike domestic, the overseas cancel needs only the original ODNO (no
        separate org number), so an order discovered via fetch_overseas_executions
        (rows with remaining_qty > 0) can be cancelled directly. Guard-free like
        the submit methods.

        No `price` parameter (unlike the domestic cancel path's-adjacent
        methods): KIS's own spec for this TR requires OVRS_ORD_UNPR="0" on a
        cancel (RVSE_CNCL_DVSN_CD="02"), not the order's real limit price —
        confirmed against examples_llm/overseas_stock/order_rvsecncl. Sending
        the real price here previously risked a rejection or unintended
        repricing on a genuine cancel."""
        if not order_no:
            raise ValueError("cancel_overseas_order needs the original order_no")
        if quantity <= 0:
            raise ValueError(f"quantity must be positive, got {quantity}")
        if quantity != int(quantity):
            # Same final defense as submit_overseas_order — a caller could pass
            # fetch_overseas_executions' remaining_qty (float, from _num) straight
            # through here.
            raise ValueError(f"ORD_QTY must be a whole number of shares, got {quantity!r}")
        quantity = int(quantity)
        body = {
            "CANO": self._account_8,
            "ACNT_PRDT_CD": self._account_prod,
            "OVRS_EXCG_CD": exchange,
            "PDNO": code,
            "ORGN_ODNO": order_no,
            "RVSE_CNCL_DVSN_CD": "02",  # 02 취소
            "ORD_QTY": str(quantity),
            "OVRS_ORD_UNPR": "0",  # required "0" on cancel, per KIS spec
            "MGCO_APTM_ODNO": "",
            "ORD_SVR_DVSN_CD": "0",
        }
        resp = await self._request(
            "POST", _OVERSEAS_REVISE_CANCEL_PATH,
            self._tr(_OVERSEAS_REVISE_CANCEL_TR_REAL, _OVERSEAS_REVISE_CANCEL_TR_PAPER), json_body=body,
            priority="high",
        )
        parsed = self._checked_body(resp, f"overseas cancel {order_no}")
        result = _order_result(parsed)
        logger.info("KIS overseas cancel accepted: orgn_odno=%s (%s)", order_no, result.message)
        return result
