import logging

from app.brokers.kis.parser import _num
from app.domain.models import BrokerBalance, BrokerPosition, OverseasCash

logger = logging.getLogger("app.brokers.kis")

_OVERSEAS_BALANCE_TR_REAL = "TTTS3012R"  # 해외주식 잔고 (실전)
_OVERSEAS_BALANCE_TR_PAPER = "VTTS3012R"  # 해외주식 잔고 (모의)
_OVERSEAS_BALANCE_PATH = "/uapi/overseas-stock/v1/trading/inquire-balance"
_OVERSEAS_PRESENT_BAL_TR_REAL = "CTRP6504R"  # 해외주식 체결기준 현재잔고 (외화예수금 포함, 실전)
_OVERSEAS_PRESENT_BAL_TR_PAPER = "VTRP6504R"  # 해외주식 체결기준 현재잔고 (외화예수금 포함, 모의)
_OVERSEAS_PRESENT_BAL_PATH = "/uapi/overseas-stock/v1/trading/inquire-present-balance"


class OverseasAccountMixin:
    async def fetch_overseas_balance(
        self, exchange: str = "NASD", currency: str = "USD"
    ) -> BrokerBalance:
        """해외주식 잔고 — USD figures. This TR has no cash-deposit field
        (that's a separate present-balance TR), so cash_deposit is 0 here;
        callers wanting orderable USD should use fetch_overseas_price plus
        the domestic deposit, or add the psamount TR later."""
        params = {
            "CANO": self._account_8,
            "ACNT_PRDT_CD": self._account_prod,
            "OVRS_EXCG_CD": exchange,
            "TR_CRCY_CD": currency,
            "CTX_AREA_FK200": "",
            "CTX_AREA_NK200": "",
        }
        positions: list[BrokerPosition] = []
        saw_output2 = False

        async for body in self._paginate(
            "GET", _OVERSEAS_BALANCE_PATH, self._tr(_OVERSEAS_BALANCE_TR_REAL, _OVERSEAS_BALANCE_TR_PAPER),
            params, what="overseas balance inquiry", ctx_suffix="200", priority="high", lane="account",
        ):
            for row in body.get("output1", []):
                raw_qty = _num(row.get("ovrs_cblc_qty"))
                if raw_qty <= 0:
                    continue
                # BrokerPosition.quantity is declared int (see domain/models/account.py),
                # matching the domestic sibling (domestic/account.py). A leaked float here
                # ends up verbatim in ORD_QTY (overseas/order.py) — KIS rejects "111.0"
                # with IGW00013/IGW00007 (2026-08-19 AXTI/GOOG/GOOGL incident). Only floor
                # (never round up — rounding up would sell shares we don't hold) and warn
                # so a genuine fractional remainder (e.g. from a stock split) is visible
                # instead of silently truncated.
                qty = int(raw_qty)
                if qty != raw_qty:
                    logger.warning(
                        "overseas balance: %s reported a fractional quantity %s — floored to "
                        "%d for order sizing (possible split/dividend remainder); the "
                        "remaining %s share(s) cannot be sold by this system",
                        row.get("ovrs_pdno", ""), raw_qty, qty, raw_qty - qty,
                    )
                positions.append(
                    BrokerPosition(
                        code=row.get("ovrs_pdno", ""),
                        name=row.get("ovrs_item_name", ""),
                        quantity=qty,
                        avg_price=_num(row.get("pchs_avg_pric")),
                        current_price=_num(row.get("now_pric2")),
                        eval_amount=_num(row.get("ovrs_stck_evlu_amt")),
                        pnl_amount=_num(row.get("frcr_evlu_pfls_amt")),
                        pnl_pct=_num(row.get("evlu_pfls_rt")),
                    )
                )
            if body.get("output2"):
                saw_output2 = True

        if not saw_output2:
            # Unlike the domestic sibling (domestic/account.py), this does NOT
            # raise on a missing output2 — every field this method actually
            # uses comes from output1/positions (see the comments below on
            # why output2's own aggregate fields aren't trusted), and it's
            # unverified whether a genuinely flat overseas account still gets
            # an output2 summary row the way domestic always does. Logging
            # only, so a real "TR didn't respond properly" case is at least
            # visible without risking a false failure on an empty account.
            logger.warning("overseas balance inquiry: no output2 summary seen on any page")

        return BrokerBalance(
            cash_deposit=0.0,
            total_eval_amount=sum(p.eval_amount for p in positions),
            # Summed from positions (frcr_evlu_pfls_amt, genuinely USD), not
            # output2's tot_evlu_pfls_amt/ovrs_tot_pfls — confirmed against
            # the KIS app itself to be a KRW-converted total despite the
            # "total pnl" name, which was producing a nonsense -4168% (KRW
            # loss divided by a USD cost basis) before this fix.
            # 평가손익도 포지션 합산(원본 필드는 원화 환산값이라 신뢰 안 함)
            total_pnl_amount=sum(p.pnl_amount for p in positions),
            # Summed from positions, not output2's frcr_pchs_amt1 — that field
            # has returned implausibly large (KRW-scale) values on the paper
            # server, the same reason total_eval_amount above is derived from
            # positions rather than trusted from output2.
            # 매입금액은 포지션 합산(원본 필드 신뢰 안 함, 위 total_eval_amount와 동일 패턴)
            purchase_amount=sum(p.avg_price * p.quantity for p in positions),
            positions=positions,
        )

    async def fetch_overseas_present_balance(self, currency: str = "USD") -> OverseasCash:
        """해외주식 체결기준 현재잔고 — fills the foreign-deposit hole that
        fetch_overseas_balance leaves at 0. output2 has one row per currency
        for orderable/eval; deposit comes from output3 instead (see below).

        Note on WCRC_FRCR_DVSN_CD ("01" 원화 / "02" 외화, in theory a
        request-level switch for which currency basis the response is
        denominated in): live-verified 2026-07-20 that both values return
        byte-identical output2/output3 on this (paper) account — there is no
        separate 원화 figure to get out of this TR here, so this always
        calls "02". A caller wanting a floor for "is there enough cash"
        should apply its own fallback against `deposit`, not expect this
        method to fabricate one — see kis_bridge.insufficient_cash_block."""
        params = {
            "CANO": self._account_8,
            "ACNT_PRDT_CD": self._account_prod,
            "WCRC_FRCR_DVSN_CD": "02",
            "NATN_CD": "000",  # 전체 국가
            "TR_MKET_CD": "00",
            "INQR_DVSN_CD": "00",
        }
        resp = await self._request(
            "GET", _OVERSEAS_PRESENT_BAL_PATH,
            self._tr(_OVERSEAS_PRESENT_BAL_TR_REAL, _OVERSEAS_PRESENT_BAL_TR_PAPER), params=params,
            priority="high", lane="account",
        )
        body = self._checked_body(resp, "overseas present balance")
        orderable = eval_amount = rate = 0.0
        matched = False
        for row in body.get("output2", []):
            if row.get("crcy_cd") == currency:
                orderable = _num(row.get("frcr_drwg_psbl_amt_1"))
                eval_amount = _num(row.get("frcr_evlu_amt2"))
                rate = _num(row.get("frst_bltn_exrt"))
                matched = True
                break
        else:
            logger.warning(
                "overseas present-balance: no output2 row matched currency=%s — raw body: %s",
                currency, body,
            )
        # output2's own per-currency frcr_dncl_amt_2 (외화예수금액2) has been
        # live-verified to read 0.000000 for every currency on this account
        # even when real USD cash is held — output3.frcr_evlu_tota(외화평가총액)
        # is the reliable source: output3.tot_asst_amt(총자산) equals
        # frcr_evlu_tota + evlu_amt_smtl(보유종목평가금액) exactly, confirming
        # frcr_evlu_tota is the pure cash figure, just KRW-converted, so it's
        # divided back to `currency` here using this same response's own
        # exchange rate (frst_bltn_exrt) for a same-timestamp conversion.
        output3 = body.get("output3") or {}
        if matched and not rate:
            # currency WAS found in output2 (matched=True is correct — see
            # OverseasCash.matched's docstring, this isn't the "no row"
            # case), but frst_bltn_exrt came back 0/missing so the division
            # below can't run — deposit ends up 0.0 looking like a genuine
            # figure instead of an unconvertible one. Surfaced here since
            # matched=True gives no other signal a caller could check.
            logger.warning(
                "overseas present-balance: currency=%s matched but exchange rate (frst_bltn_exrt) "
                "is 0/missing — deposit forced to 0.0, not a real balance",
                currency,
            )
        deposit = _num(output3.get("frcr_evlu_tota")) / rate if rate else 0.0
        # orderable (frcr_drwg_psbl_amt_1) is the SAME output2 category as
        # frcr_dncl_amt_2 above — confirmed 2026-07-20 to also read 0.000000
        # for every currency on this account (raw response inspected
        # directly), even with real USD cash present per `deposit` above.
        # Deliberately returned as-is, NOT floored to `deposit` here — that
        # correction belongs to whichever caller needs a "can this order be
        # afforded" answer (kis_bridge.insufficient_cash_block), not to this
        # plain data fetch. A caller that blindly trusted this raw zero for
        # display (as the dashboard's 원화예수금 label did) would show a
        # phantom "no cash" figure right next to the correct `deposit` one.
        return OverseasCash(
            currency=currency, deposit=deposit, orderable=orderable,
            eval_amount=eval_amount, matched=matched,
        )
