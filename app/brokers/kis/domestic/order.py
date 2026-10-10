import logging

from app.brokers.kis.parser import _num, _order_result
from app.domain.models import BrokerExecution, OrderResult

logger = logging.getLogger(__name__)

_ORDER_TR_ID_REAL = {"buy": "TTTC0012U", "sell": "TTTC0011U"}  # 주식주문(현금), 실전투자
_ORDER_TR_ID_PAPER = {"buy": "VTTC0012U", "sell": "VTTC0011U"}  # 주식주문(현금), 모의투자
_ORDER_PATH = "/uapi/domestic-stock/v1/trading/order-cash"
_REVISE_CANCEL_TR_ID_REAL = "TTTC0013U"  # 주식주문(정정취소), 실전투자
_REVISE_CANCEL_TR_ID_PAPER = "VTTC0013U"  # 주식주문(정정취소), 모의투자
_REVISE_CANCEL_PATH = "/uapi/domestic-stock/v1/trading/order-rvsecncl"
_EXECUTIONS_TR_ID_REAL = "TTTC0081R"  # 주식일별주문체결조회 (실전, 3개월 이내)
_EXECUTIONS_TR_ID_PAPER = "VTTC0081R"  # 주식일별주문체결조회 (모의, 3개월 이내)
_EXECUTIONS_PATH = "/uapi/domestic-stock/v1/trading/inquire-daily-ccld"

# 2026-09-07 fee/tax fix (see fetch_domestic_fee_tax below): `prsm_tlex_smtl`
# (추정제비용합계) is never taken on faith — a real KIS example's own field-
# label dictionary was found to have its labels shifted by one entry (labels
# unreliable, JSON key NAMES still authoritative), so a mislabeled response
# could just as easily hand this code a different field's value (e.g. the
# group's total filled amount) under the `prsm_tlex_smtl` key. A real
# domestic commission+tax is a small fraction of notional (~0.015% buy-side
# commission, ~0.2% sell-side commission+거래세) — 2% is a generous upper
# bound that still rejects a value anywhere near the size of the notional
# itself.
_FEE_PLAUSIBLE_MAX_RATIO = 0.02

# (account_8, code, order_date, side) -> resolved fee/tax for that group.
# Module-level, not per-broker-instance: cheap, immutable-once-set (a CLOSED
# past trading day's fee never changes), and naturally scoped by account_8 in
# the key so multiple accounts in one process never collide. Only entries for
# `order_date < today` are ever stored — see fetch_domestic_fee_tax's own
# comment for why today's own fills are deliberately never cached.
_FEE_GROUP_CACHE: dict[tuple[str, str, str, str], float] = {}


class DomesticOrderMixin:
    async def submit_order(
        self, code: str, side: str, quantity: int, price: float | None = None
    ) -> OrderResult:
        """주식주문(현금) — market order when `price` is None, limit otherwise.

        Deliberately NO safety logic here: this method does exactly what it is
        told. Every guard (enable flag, order-value cap, trade allowlist check,
        dry-run) lives in app/execution/kis_bridge.py so there is a single
        choke point to audit.
        """
        if side not in _ORDER_TR_ID_PAPER:
            raise ValueError(f"side must be 'buy' or 'sell', got {side!r}")
        if quantity <= 0:
            raise ValueError(f"quantity must be positive, got {quantity}")
        if quantity != int(quantity):
            # Defensive guard mirroring the overseas order path (2026-08-19 KIS
            # rejected a float ORD_QTY there). Domestic parsing is already
            # integer-only, so this should never trigger — it exists so a
            # future regression fails loudly instead of silently mis-sizing.
            raise ValueError(f"ORD_QTY must be a whole number of shares, got {quantity!r}")
        quantity = int(quantity)
        body = {
            "CANO": self._account_8,
            "ACNT_PRDT_CD": self._account_prod,
            "PDNO": code,
            "ORD_DVSN": "00" if price else "01",  # 00 지정가 / 01 시장가
            "ORD_QTY": str(quantity),  # KIS wants strings in POST bodies
            # round(), not int() — KRX prices are whole won, but a computed
            # float (e.g. 71235.0 arriving as 71234.99999998) truncates one
            # won low under int(); round() lands on the intended price.
            "ORD_UNPR": str(round(price)) if price else "0",
            "EXCG_ID_DVSN_CD": "KRX",
            "SLL_TYPE": "",
            "CNDT_PRIC": "",
        }
        tr_id = self._tr(_ORDER_TR_ID_REAL, _ORDER_TR_ID_PAPER)[side]
        resp = await self._request("POST", _ORDER_PATH, tr_id, json_body=body, priority="high")
        parsed = self._checked_body(resp, f"{side} order for {code}")
        result = _order_result(parsed)
        logger.info(
            "KIS %s order accepted: %s x%d (order_no=%s, %s)",
            side, code, quantity, result.order_no, result.message,
        )
        return result

    async def cancel_order(
        self,
        order_no: str,
        org_no: str,
        quantity: int,
        price: float | None = None,
        ord_dvsn: str = "00",
    ) -> OrderResult:
        """주식주문(정정취소) — cancel a still-open domestic order (전량취소).

        `order_no` (ODNO) and `org_no` (KRX_FWDG_ORD_ORGNO) both come from the
        OrderResult returned when the order was placed: the 정정취소가능주문조회
        TR that normally supplies them is absent on the paper server (VTTC0084R
        → 없는 서비스 코드, real TTTC0084R → 모의투자 TR 아님), so an order can
        only be cancelled within a session that still holds its OrderResult.

        `ord_dvsn`/`price` must match the original order. Guard-free like
        submit_order: every safety check lives in kis_bridge.
        """
        if not order_no or not org_no:
            raise ValueError("cancel_order needs both order_no and org_no from the original OrderResult")
        if quantity <= 0:
            raise ValueError(f"quantity must be positive, got {quantity}")
        if quantity != int(quantity):
            raise ValueError(f"ORD_QTY must be a whole number of shares, got {quantity!r}")
        quantity = int(quantity)
        body = {
            "CANO": self._account_8,
            "ACNT_PRDT_CD": self._account_prod,
            "KRX_FWDG_ORD_ORGNO": org_no,
            "ORGN_ODNO": order_no,
            "ORD_DVSN": ord_dvsn,
            "RVSE_CNCL_DVSN_CD": "02",  # 02 취소 (01 정정)
            "ORD_QTY": str(quantity),
            "ORD_UNPR": str(round(price)) if price else "0",  # round(), not int() — see submit_order's comment
            "QTY_ALL_ORD_YN": "Y",  # 잔량 전부 취소
            "EXCG_ID_DVSN_CD": "KRX",
        }
        resp = await self._request(
            "POST", _REVISE_CANCEL_PATH, self._tr(_REVISE_CANCEL_TR_ID_REAL, _REVISE_CANCEL_TR_ID_PAPER),
            json_body=body, priority="high",
        )
        parsed = self._checked_body(resp, f"cancel order {order_no}")
        result = _order_result(parsed)
        logger.info("KIS cancel accepted: orgn_odno=%s (%s)", order_no, result.message)
        return result

    async def revise_order(
        self, order_no: str, org_no: str, quantity: int, new_price: float, ord_dvsn: str = "00"
    ) -> OrderResult:
        """주식주문(정정) — change an open order's price/quantity (전량 기준).
        Same org_no requirement and guard-free contract as cancel_order; only
        the division code (01 정정) and the new limit price differ."""
        if not order_no or not org_no:
            raise ValueError("revise_order needs both order_no and org_no from the original OrderResult")
        if quantity <= 0 or new_price <= 0:
            raise ValueError("revise_order needs a positive quantity and new price")
        if ord_dvsn != "00":
            # A revise carries the ORIGINAL order's ord_dvsn; for a market order
            # (01) KIS silently ignores ORD_UNPR and returns "revise accepted",
            # so new_price would have no effect while the UI reports success.
            # A market order can't be repriced — cancel and re-submit instead.
            raise ValueError(
                f"cannot revise the price of a market order (ord_dvsn={ord_dvsn!r}); "
                "cancel it and submit a new order"
            )
        if quantity != int(quantity):
            # Same final defense line as submit_order/cancel_order above and
            # both overseas order methods — this was the ONE of the five that
            # never got it (2026-08-19 float-ORD_QTY incident), so a whole-number
            # float here serialized as ORD_QTY="111.0", the exact malformed body
            # KIS rejected with IGW00013/IGW00007. Domestic parsing is
            # integer-only and the sole caller passes a pydantic int, so this
            # should never trigger — it exists so a future regression fails
            # loudly instead of silently mis-sizing.
            raise ValueError(f"ORD_QTY must be a whole number of shares, got {quantity!r}")
        quantity = int(quantity)
        body = {
            "CANO": self._account_8,
            "ACNT_PRDT_CD": self._account_prod,
            "KRX_FWDG_ORD_ORGNO": org_no,
            "ORGN_ODNO": order_no,
            "ORD_DVSN": ord_dvsn,
            "RVSE_CNCL_DVSN_CD": "01",  # 01 정정 (02 취소)
            "ORD_QTY": str(quantity),
            "ORD_UNPR": str(round(new_price)),  # round(), not int() — see submit_order's comment
            "QTY_ALL_ORD_YN": "Y",
            "EXCG_ID_DVSN_CD": "KRX",
        }
        resp = await self._request(
            "POST", _REVISE_CANCEL_PATH, self._tr(_REVISE_CANCEL_TR_ID_REAL, _REVISE_CANCEL_TR_ID_PAPER),
            json_body=body, priority="high",
        )
        parsed = self._checked_body(resp, f"revise order {order_no}")
        result = _order_result(parsed)
        logger.info("KIS revise accepted: orgn_odno=%s -> %s @ %s (%s)",
                    order_no, result.order_no, new_price, result.message)
        return result

    async def fetch_daily_executions(self, start: str, end: str) -> list[BrokerExecution]:
        """주식일별주문체결조회 — all orders (filled or not) between `start` and
        `end` (YYYYMMDD, within the last 3 months for this TR)."""
        params = {
            "CANO": self._account_8,
            "ACNT_PRDT_CD": self._account_prod,
            "INQR_STRT_DT": start,
            "INQR_END_DT": end,
            "SLL_BUY_DVSN_CD": "00",  # 전체
            "PDNO": "",
            "CCLD_DVSN": "00",  # 체결/미체결 전체
            # 01 정순(날짜 오름차순): MAX_PAGINATION_PAGES 절단이 실제로
            # 일어나면(단타처럼 하루 체결이 많은 경우) 조회기간 앞쪽(평단가 계산에
            # 필요한 오래된 매수)이 아니라 뒤쪽(최근 체결)이 잘리도록 함 — 90일
            # 평단가 왜곡 완화.
            "INQR_DVSN": "01",
            "INQR_DVSN_3": "00",
            "ORD_GNO_BRNO": "",
            "ODNO": "",
            "INQR_DVSN_1": "",
            "CTX_AREA_FK100": "",
            "CTX_AREA_NK100": "",
        }
        executions: list[BrokerExecution] = []
        async for body in self._paginate(
            "GET", _EXECUTIONS_PATH, self._tr(_EXECUTIONS_TR_ID_REAL, _EXECUTIONS_TR_ID_PAPER), params,
            what="daily executions inquiry", priority="high", lane="account",
        ):
            for row in body.get("output1", []):
                executions.append(
                    BrokerExecution(
                        order_date=row.get("ord_dt", ""),
                        order_no=row.get("odno", ""),
                        code=row.get("pdno", ""),
                        name=row.get("prdt_name", ""),
                        side="sell" if row.get("sll_buy_dvsn_cd") == "01" else "buy",
                        order_qty=int(_num(row.get("ord_qty"))),
                        filled_qty=int(_num(row.get("tot_ccld_qty"))),
                        remaining_qty=int(_num(row.get("rmn_qty"))),
                        avg_price=_num(row.get("avg_prvs")),
                        filled_amount=_num(row.get("tot_ccld_amt")),
                        # `output1` (this row) has NO fee/tax field at all —
                        # `prsm_tlex_smtl` lives only in `output2`, this call's
                        # separate one-per-call summary object. Left 0.0 here;
                        # fetch_domestic_fee_tax fills it in afterward via a
                        # narrow per-(ticker, date, side) probe. See that
                        # method's own docstring and project memory:
                        # realized-pnl-report-done (2026-09-04/09-07).
                        fee_tax=0.0,
                        cancelled=row.get("cncl_yn") == "Y",
                    )
                )
        return executions

    async def _probe_fee(
        self, code: str, order_date: str, side: str, group_filled_qty: float, group_filled_amount: float,
    ) -> float | None:
        """One narrow `inquire_daily_ccld` call scoped to a single (ticker,
        day, side) — `PDNO=code`, `INQR_STRT_DT=INQR_END_DT=order_date`,
        `SLL_BUY_DVSN_CD` fixed to just this side, `CCLD_DVSN="01"` (filled
        only). At this narrow scope, `output2.prsm_tlex_smtl` is exactly this
        group's own commission+tax total (unlike an unscoped call across the
        whole report window, where it would be one account-wide total with
        no per-ticker/per-day attribution possible).

        Returns `None` — never a guess — on ANY of: the request itself
        failing, a multi-page response (a single narrow day/ticker/side
        query should never paginate; if it does, output2's scope for that
        page is not trustworthy), a quantity mismatch against what the
        caller already observed in `output1` (output2 disagreeing with
        output1 about how many shares filled means something about this
        query is not what it looks like), or the returned fee falling
        outside `_FEE_PLAUSIBLE_MAX_RATIO` of the group's own notional (the
        guard against a mislabeled/shifted field handing back some other
        quantity entirely — see `_FEE_PLAUSIBLE_MAX_RATIO`'s own comment).
        Every rejection here means the caller keeps `fee_tax=0.0` for that
        group and logs a warning — never a fabricated number in a real-money
        report."""
        params = {
            "CANO": self._account_8,
            "ACNT_PRDT_CD": self._account_prod,
            "INQR_STRT_DT": order_date,
            "INQR_END_DT": order_date,
            "SLL_BUY_DVSN_CD": "01" if side == "sell" else "02",
            "PDNO": code,
            "CCLD_DVSN": "01",  # 체결만
            "INQR_DVSN": "01",
            "INQR_DVSN_3": "00",
            "ORD_GNO_BRNO": "",
            "ODNO": "",
            "INQR_DVSN_1": "",
            "CTX_AREA_FK100": "",
            "CTX_AREA_NK100": "",
        }
        try:
            resp = await self._request(
                "GET", _EXECUTIONS_PATH, self._tr(_EXECUTIONS_TR_ID_REAL, _EXECUTIONS_TR_ID_PAPER),
                params=params, priority="high", lane="account",
            )
            body = self._checked_body(resp, f"pnl fee probe {code} {order_date} {side}")
        except Exception:
            logger.warning("pnl fee probe failed for %s %s %s", code, order_date, side, exc_info=True)
            return None
        if resp.headers.get("tr_cont", "").strip() in ("M", "F"):
            logger.warning(
                "pnl fee probe for %s %s %s got a multi-page response — output2 scope untrustworthy, skipping",
                code, order_date, side,
            )
            return None
        output2 = body.get("output2") or {}
        if isinstance(output2, list):
            output2 = output2[0] if output2 else {}
        probe_qty = _num(output2.get("tot_ccld_qty"))
        if abs(probe_qty - group_filled_qty) > 1:
            logger.warning(
                "pnl fee probe for %s %s %s: output2 qty %.0f does not match observed %.0f — skipping",
                code, order_date, side, probe_qty, group_filled_qty,
            )
            return None
        fee = _num(output2.get("prsm_tlex_smtl"))
        notional = group_filled_amount if group_filled_amount > 0 else _num(output2.get("tot_ccld_amt"))
        if not (0 < fee <= _FEE_PLAUSIBLE_MAX_RATIO * notional):
            logger.warning(
                "pnl fee probe for %s %s %s: fee %.1f implausible against notional %.1f — skipping",
                code, order_date, side, fee, notional,
            )
            return None
        logger.info("pnl fee probe %s %s %s -> %.1f", code, order_date, side, fee)
        return fee

    async def fetch_domestic_fee_tax(
        self, executions: list[BrokerExecution], *, codes: set[str], max_groups: int, today: str,
    ) -> int:
        """Fills in `fee_tax` on the given `executions` (mutated in place)
        for every fill whose ticker is in `codes`, via one narrow probe call
        per (ticker, order date, side) group that actually has a filled
        execution — see `_probe_fee` for why a group, not a per-call total,
        is the right unit. Only ever RAISES a fill's `fee_tax` above 0.0;
        never invents a value `_probe_fee` didn't itself validate.

        Multiple same-group fills (e.g. two partial fills of the same
        ticker/day/side) split the group's resolved fee proportionally by
        `filled_amount` (falls back to `filled_qty` if the group's total
        filled_amount is 0) — same proportional-attribution convention
        `app.execution.pnl._match_realized_trades` already uses for a sell
        that only partially matches an earlier buy.

        Groups are attempted most-recent-order-date-first, capped at
        `max_groups` (settings.kis_pnl_fee_probe_max_groups) — see that
        setting's own comment for the latency budget this caps. `today`
        (YYYYMMDD) marks which groups are safe to cache in the module-level
        `_FEE_GROUP_CACHE`: a CLOSED day's fee is final and reused across
        calls, but today's own fills could still see more activity later
        this same trading day, so they are always re-probed, never cached.

        Returns the number of groups that did NOT end up with a resolved
        fee — either skipped outright past `max_groups`, or attempted but
        rejected by `_probe_fee`'s validity checks. The caller (the /api/pnl
        route) surfaces this as `fee_partial` when > 0, exactly the same
        "best-effort, say so when it wasn't complete" pattern as that same
        response's `start_clamped` flag."""
        groups: dict[tuple[str, str, str], list[BrokerExecution]] = {}
        for e in executions:
            if e.filled_qty <= 0 or e.code not in codes:
                continue
            groups.setdefault((e.code, e.order_date, e.side), []).append(e)

        ordered = sorted(groups.items(), key=lambda kv: kv[0][1], reverse=True)
        attempted, skipped_over_cap = ordered[:max_groups], ordered[max_groups:]
        unresolved = len(skipped_over_cap)

        for (code, order_date, side), fills in attempted:
            cache_key = (self._account_8, code, order_date, side)
            fee = _FEE_GROUP_CACHE.get(cache_key)
            if fee is None:
                group_qty = sum(f.filled_qty for f in fills)
                group_amount = sum(f.filled_amount for f in fills)
                fee = await self._probe_fee(code, order_date, side, group_qty, group_amount)
                if fee is not None and order_date < today:
                    _FEE_GROUP_CACHE[cache_key] = fee
            if fee is None:
                unresolved += 1
                continue
            total_amount = sum(f.filled_amount for f in fills)
            total_qty = sum(f.filled_qty for f in fills)
            for f in fills:
                if total_amount > 0:
                    weight = f.filled_amount / total_amount
                elif total_qty > 0:
                    weight = f.filled_qty / total_qty
                else:
                    weight = 0.0
                f.fee_tax = fee * weight
        return unresolved
