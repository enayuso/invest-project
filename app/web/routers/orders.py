"""Order-ticket endpoints — buyable/sellable inquiries, order placement,
open orders, cancel/revise, exec-mode badge, risk-rule settings, and the
kill-switch/daily-loss safety state (agent-architecture-roadmap stage 2)."""

from __future__ import annotations

import logging
import re

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.clock import get_clock
from app.config import settings
from app.account_mode import ACCOUNT_MODES, is_valid_mode
from app.execution import pending_orders, portfolio_risk, risk_rules, runtime_settings, safety_switch
from app.execution.kis_bridge import place_manual_order, route_ticker
from app.execution.ledger import _record_kis_order_no, record_trade
from app.live.alerts import Alert, record_alert
from app.web.account_mode_middleware import require_kis
from app.web import state

_ORDER_NO_RE = re.compile(r"order_no=(\d+)")

logger = logging.getLogger("app.web")

router = APIRouter()

# Routes that genuinely need a live KIS account behind them return 412 until
# this deployment has credentials configured for the active account mode
# (app.web.account_mode_middleware.require_kis — replaces the old
# app.auth.dependencies.require_credentials onboarding gate, 2026-08-07
# single-account rollback). Deliberately per-route rather than router-wide:
# /api/exec-mode, /api/settings/* and the pending-order LIST endpoints are
# configuration and ledger reads that the dashboard itself needs to render
# even before KIS is configured, so gating the whole router would make the
# setup state unable to display itself.
# 라우터 전체가 아니라 라우트별로 적용 — KIS 미설정 상태에서도 대시보드 자체는
# 표시할 수 있어야 함.
_needs_kis = [Depends(require_kis)]



class OrderRequest(BaseModel):
    ticker: str
    side: str  # "buy" | "sell"
    shares: int
    order_type: str = "market"  # "market" | "limit"
    limit_price: float | None = None
    reference_price: float | None = None  # current price shown in the UI
    approved: bool = False  # user confirmed a large order past the approval threshold


class RiskRulesRequest(BaseModel):
    # Partial update: only the keys present are changed. Values are validated
    # against risk_rules.RULE_SPEC server-side.
    values: dict[str, object]
    # Which account these limits belong to. Omitted = the request's own mode,
    # which is what the settings tab wants. Onboarding sends it explicitly:
    # a user who registers both credential sets resolves to 모의투자, so a
    # plain save there would file their 실전 limits under paper and leave the
    # real caps at 0. See risk_rules.save.
    # 미지정이면 현재 모드 — 온보딩은 명시해야 실전 한도가 제 스코프에 저장됨.
    account_mode: str | None = None


class RuntimeSettingsRequest(BaseModel):
    # Partial update: only the keys present are changed. Validated against
    # runtime_settings.RUNTIME_SPEC (and the real-account safety guard)
    # server-side.
    ai_profile: str | None = None
    execution_mode: str | None = None


class CancelRequest(BaseModel):
    order_no: str


class ReviseRequest(BaseModel):
    order_no: str
    new_price: float


class KillSwitchRequest(BaseModel):
    engaged: bool
    reason: str = ""


class RejectPendingOrderRequest(BaseModel):
    reason: str = ""


@router.get("/api/buyable/{ticker}", dependencies=_needs_kis)
async def buyable_endpoint(ticker: str, price: float | None = None) -> dict:
    """Order-ticket 매수가능: cash and max shares for a domestic ticker. US
    orderable cash needs the overseas present-balance TR (a later phase), so
    for US symbols this returns available=False rather than a wrong number."""
    broker = await state.kis_broker()
    if broker is None:
        raise HTTPException(503, "KIS broker not configured")
    krx_code, us_symbol = route_ticker(ticker.upper())
    try:
        if krx_code is not None:
            info = await broker.fetch_buyable(krx_code, price)
            return {"available": True, "cash": info.cash, "max_qty": info.max_qty,
                    "max_amount": info.max_amount, "currency": "KRW"}
        if us_symbol is not None:
            # US orderable cash comes from the overseas present-balance TR
            # (USD deposit/drawable) — no per-symbol max-qty TR on the paper server.
            cash = await broker.fetch_overseas_present_balance("USD")
            if not cash.matched:
                # cash.orderable/deposit are zeroed placeholders here, not a
                # confirmed $0 balance (see OverseasCash.matched) — reporting
                # them as real would silently show a wrong number instead of
                # this endpoint's own documented "available=False" contract.
                # 통화 매칭 실패 시 확정된 0원이 아니므로 available=False로 응답.
                return {"available": False, "reason": "USD balance lookup returned no matching currency row"}
            return {"available": True, "cash": cash.orderable, "max_qty": None,
                    "max_amount": cash.deposit, "currency": "USD"}
    except Exception as exc:
        raise HTTPException(502, f"buyable inquiry failed: {exc}") from exc
    return {"available": False, "reason": "buyable inquiry not supported for this ticker"}


@router.get("/api/sellable/{ticker}", dependencies=_needs_kis)
async def sellable_endpoint(ticker: str) -> dict:
    """Order-ticket 매도가능: shares currently held (balance-derived ceiling,
    the same figure app.execution.kis_bridge.domestic_sell_over_held uses).

    Reads state.kis_account_snapshot()'s 8s cache instead of calling the
    broker directly — this endpoint is hit every time the order ticket
    loads, and was previously an uncached throttled KIS call on top of
    whatever /api/portfolio and the gate chain already fetched."""
    broker = await state.kis_broker()
    if broker is None:
        raise HTTPException(503, "KIS broker not configured")
    upper = ticker.upper()
    krx_code, us_symbol = route_ticker(upper)
    try:
        _account, positions = await state.kis_account_snapshot()
    except Exception as exc:
        raise HTTPException(502, f"sellable inquiry failed: {exc}") from exc
    key = f"{krx_code}.KS" if krx_code is not None else us_symbol
    held = positions.get(key, {}).get("quantity", 0)
    return {"sellable_qty": held}


@router.post("/api/order", dependencies=_needs_kis)
async def place_order_endpoint(req: OrderRequest) -> dict:
    """Submit a user-initiated order through the shared kis_bridge safety
    gates (same gate chain as the automated mirror path — routing, master
    switch, value cap, dry-run, circuit-breaker, per-stock halt, sell
    ceiling), then record a filled order to the ledger so positions/PnL stay
    in sync. Returns the disposition; never places an order that skips a gate."""
    result = await place_manual_order(
        req.ticker.upper(),
        req.side,
        req.shares,
        order_type=req.order_type,
        limit_price=req.limit_price,
        reference_price=req.reference_price,
        approved=req.approved,
        broker=await state.kis_broker(),
    )
    # Register the cancel/revise handle FIRST, before any ledger write.
    # This order is load-bearing: place_manual_order only reports ok once a
    # REAL order has reached KIS, and the org_no it carries here is the only
    # handle this app will ever have for cancelling that order — the paper
    # server has no 정정취소가능주문조회 TR to recover it after the fact (see
    # app.web.state._cancelable_orders_by_scope's own comment). Registering it
    # AFTER the ledger write, as this did until 2026-08-26, meant a ledger
    # failure left the order live at KIS, absent from the ledger, AND
    # permanently uncancelable through this app, all at once.
    # 원장 기록보다 먼저 등록 — 주문은 이미 KIS에 실제로 나갔으므로 취소 핸들이
    # 원장 기록 성공 여부에 종속되면 안 됨.
    if result.ok and result.org_no:
        state.cancelable_orders()[result.order_no] = {
            "org_no": result.org_no,
            "shares": result.shares,
            "ord_dvsn": "00" if req.order_type == "limit" else "01",
            "price": req.limit_price if req.order_type == "limit" else None,
        }
    if result.ok:  # reflect the new order/fill without waiting out the cache TTL
        state.invalidate_cache(state.user_key("account"), state.user_key("executions"))
        # Also drop portfolio_risk's short read-only snapshot cache so a
        # narrative/pre-filter caller isn't served a pre-fill balance.
        from app.execution.portfolio_risk import invalidate_account_read_cache
        invalidate_account_read_cache()

    # Record every order that actually reached KIS to the ledger — market and
    # limit alike (2026-09-04, real-incident fix: a LIMIT sell (057680.KS)
    # that never got a ledger row at all went undiscovered for days, because
    # /api/trades is now 100% KIS-sourced (the local-ledger merge it used to
    # do was removed in an earlier, unrelated change) and neither of this
    # system's drift detectors (fill_monitor.py, reconciliation.py) can catch
    # a row that was never written in the first place — both work "start from
    # a `trades` row, look for a matching KIS execution", so a missing row is
    # structurally invisible to either.
    #
    # This used to record ONLY market orders (they fill immediately, so the
    # reference price is a good-enough fill price, corrected later from the
    # KIS execution by reconciliation) on the theory that a limit order's
    # eventual fill would surface via /api/trades once it actually filled —
    # true when this was written, no longer true now that that endpoint
    # dropped its local-ledger merge. Recording a limit order optimistically
    # at submission (like the automated pipeline's mirror_trade_to_kis/
    # rebalance_to_target path already does) brings it inside fill_monitor's
    # EXISTING detect-only envelope instead: `kis_order_no` below is what lets
    # `_pending_fill_checks` find it, and if the order never fills or gets
    # cancelled, fill_monitor's `unconfirmed_fill`/`unconfirmed_fill_final`
    # alerts fire within its normal ~30-minute window — same as any other
    # order. This does NOT add any auto-correction: `fill_monitor_auto_
    # correct_enabled` stays False (its own enable-threshold isn't met yet,
    # unrelated to this fix); this only makes the order visible to alerting
    # that already exists.
    #
    # Known, accepted trade-off (not fixed here, and not a new risk this
    # introduces — every other sell path in this system, automated and
    # forced-exit alike, already records before the fill is confirmed the
    # same way): recording a limit SELL immediately drops `shares_after`
    # before the sell is actually filled, so `get_open_stop_loss` can stop
    # watching a position that, if the limit order never fills, is still
    # real and unprotected until fill_monitor's alert catches the gap.
    #
    # None = no ledger write was applicable (the order never went out).
    # True/False = attempted and succeeded/failed. A failure here must NOT
    # reach the client as a 500: the order is already live at KIS, and
    # trading.js's submitOrder turns any non-2xx into "주문 요청 실패", which
    # would invite a human to resubmit an order that actually went through.
    # 이미 체결/접수된 주문을 "실패"로 보여주면 중복 주문 위험.
    ledger_recorded: bool | None = None
    if result.ok:
        ledger_recorded = False
        try:
            # record_trade already degrades quietly on a DB failure (returns
            # None — app.execution.ledger's module docstring), so the common
            # failure is a None return, not an exception; both are reported the
            # same way here. asyncio.CancelledError is a BaseException and is
            # deliberately NOT swallowed by this `except Exception`.
            trade_id = await record_trade(
                req.ticker.upper(),
                get_clock().today(),
                result.side,
                result.shares,
                result.price or 0.0,
                reason=f"manual dashboard order; KIS order_no={result.order_no}",
                # Known upfront, unlike the auto paths — place_manual_order already
                # confirmed this order reached KIS by the time we get here.
                mirror_disposition=f"submitted: manual dashboard {req.order_type} order; order_no={result.order_no}",
                kis_order_no=result.order_no,
            )
            ledger_recorded = trade_id is not None
            # Carry the trade_id forward on the cancel-handle registry entry
            # (registered above, before this write) so a later /api/order/
            # revise can best-effort update this row's kis_order_no once KIS
            # assigns a new one — see that endpoint below. Guarded with .get
            # in case the entry was already popped by a fast concurrent cancel.
            if ledger_recorded and result.order_no in state.cancelable_orders():
                state.cancelable_orders()[result.order_no]["trade_id"] = trade_id
        except Exception as exc:  # noqa: BLE001 - the order is already live at KIS; never 500 over the record
            logger.error(
                "manual order %s %s x%d reached KIS (order_no=%s) but the ledger write RAISED: %s",
                result.side, result.ticker, result.shares, result.order_no, exc, exc_info=True,
            )
        if not ledger_recorded:
            logger.error(
                "manual order %s %s x%d is LIVE at KIS (order_no=%s) with no ledger row — "
                "positions/PnL will understate it until reconciliation catches up",
                result.side, result.ticker, result.shares, result.order_no,
            )
            # Same "a human needs to look at this, there is nothing to auto-undo"
            # convention as app.execution.pending_orders' pending_order_unlinked
            # alert and ledger.py's suspect_stop_provenance alert. record_alert
            # is itself best-effort and never raises (app.live.alerts).
            await record_alert(
                Alert(
                    ticker=result.ticker,
                    kind="manual_order_unrecorded",
                    message=(
                        f"Manual dashboard order {result.side} {result.ticker} x{result.shares} "
                        f"was accepted by KIS (order_no={result.order_no}) but could not be written "
                        "to the internal ledger. The order is REAL and still live — do not resubmit; "
                        "verify against KIS and correct the ledger manually."
                    ),
                    triggered_at=get_clock().now(),
                )
            )
    return {
        "status": result.status,
        "message": result.message,
        "order_no": result.order_no,
        "org_no": result.org_no,
        "ticker": result.ticker,
        "side": result.side,
        "shares": result.shares,
        "price": result.price,
        "ledger_recorded": ledger_recorded,
    }


@router.get("/api/exec-mode")
async def exec_mode_endpoint() -> dict:
    """Whether real orders can go out — drives the order ticket's mode
    badge. `enabled`/`dry_run` stay config-only by design: arming a live
    account stays a deliberate server-side action (there is no endpoint to
    change these). `execution_mode` is different — it only changes approval
    routing (see ExecutionMode's own docstring) and can never bypass a risk
    gate, so it's dashboard-toggleable via /api/settings/runtime; this
    endpoint reports whatever is currently in effect (override or config
    default) via runtime_settings.rule()."""
    return {
        "enabled": settings.kis_execution_enabled,
        "dry_run": settings.kis_execution_dry_run,
        "broker_configured": await state.kis_broker() is not None,
        "execution_mode": runtime_settings.rule("execution_mode").value,
    }


@router.get("/api/settings/risk")
async def get_risk_rules() -> dict:
    """Current risk-rule values (runtime override or settings default) + which
    keys are overridden + the per-currency approval-threshold info the order
    ticket needs to decide when to prompt for approval."""
    return current_risk_rules_payload()


@router.post("/api/settings/risk")
async def save_risk_rules(req: RiskRulesRequest) -> dict:
    """Persist risk-rule overrides. The master switch / dry-run are NOT here
    by design (config-only). Returns the new resolved set, or 400 on a bad
    value / 503 if the store is unreachable."""
    values = {k: v for k, v in req.values.items() if v is not None}
    if not values:
        raise HTTPException(status_code=400, detail="no rule values supplied")
    if req.account_mode is not None and not is_valid_mode(req.account_mode):
        raise HTTPException(
            status_code=400,
            detail=f"unknown account mode {req.account_mode!r}, expected one of {list(ACCOUNT_MODES)}",
        )
    try:
        await risk_rules.save(values, account_mode=req.account_mode)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    return current_risk_rules_payload()


def current_risk_rules_payload() -> dict:
    """Read the convenience fields out of the already-resolved set rather than
    calling `rule()` again.

    `rule()` raises for a mode-scoped money limit the deployment hasn't
    configured (correct — an order gate must not proceed without a cap), while
    `current()` reports those as None for display. Re-resolving here defeated
    that softening and 500'd the settings screen for every brand-new user,
    whose mode resolves to 'real' before they have credentials. Confirmed live,
    2026-08-06.
    이미 완화 처리된 값을 재조회하면 완화가 무의미해짐 — 신규 사용자의 설정 화면이
    500 나던 원인."""
    resolved = risk_rules.current()
    values = resolved["values"]
    return {
        **resolved,
        "approval_enabled": bool(values.get("kis_approval_required_enabled")),
        "approval_threshold_krw": values.get("kis_approval_threshold_krw"),
        "approval_threshold_usd": values.get("kis_approval_threshold_usd"),
    }


@router.get("/api/settings/runtime")
async def get_runtime_settings() -> dict:
    """Current AI Profile / Trading Mode (runtime override or settings
    default), plus kis_account_mode read-only for the dashboard's Trading
    Environment badge — that value is never part of RUNTIME_SPEC and can't
    be changed through this endpoint."""
    return runtime_settings.current()


@router.post("/api/settings/runtime")
async def save_runtime_settings(req: RuntimeSettingsRequest) -> dict:
    """Persist an ai_profile/execution_mode override, effective immediately
    (no restart). kis_account_mode is NOT here by design — Trading
    Environment stays config-file-only, see docs/safety-and-risk.md 5.1.
    400 on a bad enum value or the real-account safety guard, 503 if the
    store is unreachable."""
    values = {k: v for k, v in req.model_dump().items() if v is not None}
    if not values:
        raise HTTPException(status_code=400, detail="no runtime setting values supplied")
    try:
        return await runtime_settings.save(values)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@router.get("/api/orders/open", dependencies=_needs_kis)
async def open_orders_endpoint() -> list[dict]:
    """Still-open (unfilled or partially filled) orders from the KIS paper
    account over the last week. `cancelable` is true only for orders this
    process placed in the current (user, account_mode) scope (see
    state.cancelable_orders() — an order placed in another session, another
    account mode, or before a restart shows cancelable=False; the paper
    server has no lookup TR that would let us recover its org_no after the
    fact)."""
    broker = await state.kis_broker()
    if broker is None:
        return []

    def _row(e, ticker: str, currency: str, cancelable: bool) -> dict:
        return {
            "order_no": e.order_no, "ticker": ticker, "name": e.name, "currency": currency,
            "side": e.side, "order_qty": e.order_qty, "filled_qty": e.filled_qty,
            "remaining_qty": e.remaining_qty, "price": e.avg_price, "cancelable": cancelable,
        }

    domestic, overseas = await state.recent_executions()
    rows: list[dict] = [
        _row(e, f"{e.code}.KS", "KRW", e.order_no in state.cancelable_orders())
        # KIS's rmn_qty doesn't always net to 0 once an order is actually
        # cancelled/expired/rejected — cncl_yn (취소) and rjct_rson (거부) are the
        # real signals, or such an order lingers here forever looking like it's
        # still open. Domestic sets `cancelled`, overseas sets `rejected`; both
        # branches state the same rule so a future parser change on either side
        # can't silently reintroduce this bug.
        #
        # Safety: this can only ever drop rows KIS itself explicitly marked
        # cancelled/rejected. If the server omits those fields (or returns them
        # empty), behavior is identical to before — a genuinely-open order can
        # never be hidden by this filter.
        for e in domestic if e.remaining_qty > 0 and not e.cancelled and not e.rejected
    ]
    rows += [
        _row(e, e.code, "USD", False)
        for e in overseas if e.remaining_qty > 0 and not e.cancelled and not e.rejected
    ]
    return rows


@router.post("/api/order/cancel", dependencies=_needs_kis)
async def cancel_order_endpoint(req: CancelRequest) -> dict:
    """Cancel a still-open domestic order placed in this process."""
    broker = await state.kis_broker()
    if broker is None:
        raise HTTPException(503, "KIS broker not configured")
    info = state.cancelable_orders().get(req.order_no)
    if info is None:
        raise HTTPException(
            409,
            "order not cancelable: only orders placed in this session can be "
            "cancelled (no order-org number on record for it)",
        )
    try:
        result = await broker.cancel_order(
            req.order_no, info["org_no"], info["shares"],
            price=info["price"], ord_dvsn=info["ord_dvsn"],
        )
    except Exception as exc:
        raise HTTPException(502, f"cancel failed: {exc}") from exc
    state.cancelable_orders().pop(req.order_no, None)
    state.invalidate_cache(state.user_key("account"), state.user_key("executions"))
    return {"status": "cancelled", "order_no": req.order_no, "message": result.message}


@router.post("/api/order/revise", dependencies=_needs_kis)
async def revise_order_endpoint(req: ReviseRequest) -> dict:
    """Revise a still-open domestic order's price (session-placed orders only)."""
    broker = await state.kis_broker()
    if broker is None:
        raise HTTPException(503, "KIS broker not configured")
    info = state.cancelable_orders().get(req.order_no)
    if info is None:
        raise HTTPException(409, "order not revisable: only orders placed in this session can be revised")
    if req.new_price <= 0:
        raise HTTPException(400, "new_price must be positive")
    if info["ord_dvsn"] != "00":
        raise HTTPException(
            400,
            "market orders cannot be repriced — cancel the order and submit a new one",
        )
    try:
        result = await broker.revise_order(
            req.order_no, info["org_no"], info["shares"], req.new_price, ord_dvsn=info["ord_dvsn"],
        )
    except Exception as exc:
        raise HTTPException(502, f"revise failed: {exc}") from exc
    # A revise normally yields a NEW order_no; carry the registry entry to it.
    # Two accepted-revise shapes must NOT lose the entry:
    #   * empty result.order_no — _checked_body (app/brokers/kis/client.py)
    #     raises on rt_cd != "0", so a revise_order that RETURNS at all means
    #     KIS accepted it; an empty ODNO here is the "output": null response
    #     app/brokers/kis/parser.py documents from a real incident, not a
    #     failure. (KIS's own order_rvsecncl reference documents ODNO on
    #     success, so this is a defensive case, not a contract.)
    #   * result.order_no == the original — the insert-then-pop below deleted
    #     the entry it had just written.
    # In both, the order is still live at KIS and the ORIGINAL order_no stays
    # the only handle this app and the operator have for it, so the entry is
    # updated in place under its existing key rather than dropped. Dropping it
    # made a later /api/order/cancel 409 ("not cancelable") on an order that
    # was still open — the app forgetting the org_no it can never re-fetch
    # (no 정정취소가능주문조회 TR on the paper server).
    # 정정이 접수됐는데 새 ODNO가 없거나 기존과 같으면, 기존 키 그대로 제자리 갱신.
    registry = state.cancelable_orders()
    effective_order_no = result.order_no or req.order_no
    registry[effective_order_no] = {**info, "price": req.new_price}
    if effective_order_no != req.order_no:
        registry.pop(req.order_no, None)
    state.invalidate_cache(state.user_key("account"), state.user_key("executions"))

    # Best-effort: carry the `trades` row's kis_order_no forward to whatever
    # new order number KIS just assigned. Without this, a revised limit order
    # (now recorded to the ledger at submission — see /api/order above) would
    # leave its ledger row pointing at the STALE order number, and
    # fill_monitor would search KIS for an order number KIS no longer
    # reports — a guaranteed false-positive unconfirmed_fill/
    # unconfirmed_fill_final on every single revise. `_record_kis_order_no`
    # itself never raises (same best-effort convention as
    # `_record_mirror_disposition`), but this is wrapped anyway so a monkey-
    # patched or future-changed version can never turn a successful KIS
    # revise into a 500 for the caller — the broker-side revise already
    # succeeded regardless of whether this bookkeeping does.
    trade_id = info.get("trade_id")
    if trade_id is not None and effective_order_no != req.order_no:
        try:
            await _record_kis_order_no(trade_id, effective_order_no)
        except Exception:  # noqa: BLE001 - never let ledger bookkeeping break a successful revise response
            logger.warning(
                "revise_order: could not update trades.kis_order_no for trade_id=%s (%s -> %s)",
                trade_id, req.order_no, effective_order_no, exc_info=True,
            )
    return {"status": "revised", "order_no": effective_order_no, "message": result.message}


@router.get("/api/execution/safety")
async def execution_safety_status() -> dict:
    """Kill-switch state (account-wide) + today's loss vs. the daily
    max-loss breaker threshold, per market (domestic/overseas never
    combined) — the dashboard's safety panel (agent-architecture-roadmap
    stage 2)."""
    engaged, reason, state_unknown = await safety_switch.kill_switch_status()
    return {
        "kill_switch_engaged": engaged,
        "kill_switch_reason": reason,
        "kill_switch_state_unknown": state_unknown,
        "daily_loss": {
            "domestic": await safety_switch.daily_loss_status("domestic"),
            "overseas": (
                await safety_switch.daily_loss_status("overseas")
                if settings.overseas_trading_enabled or portfolio_risk.overseas_flag_contradicted()
                else {"available": False, "reason": "overseas_disabled"}
            ),
        },
    }


@router.post("/api/execution/safety/kill-switch")
async def set_kill_switch(req: KillSwitchRequest) -> dict:
    """Engage/disengage the kill switch — a deliberate manual override that
    blocks (or un-blocks) every order regardless of any other gate. No
    approval-threshold-style auto path here by design: only a human toggles
    this."""
    if req.engaged:
        if not req.reason.strip():
            raise HTTPException(400, "a reason is required to engage the kill switch")
        try:
            await safety_switch.engage_kill_switch(req.reason.strip())
        except Exception as exc:
            # engage/disengage_kill_switch have no internal try/except (unlike
            # kill_switch_status's own fail-closed read) — a transient DB
            # hiccup here previously reached the client as a raw 500 with a
            # plain-text body ("Internal Server Error"), which the dashboard's
            # `r.json()` call then failed to parse, surfacing a confusing
            # SyntaxError instead of the actual problem (confirmed live
            # 2026-08-06). Same "translate to a real HTTP error" convention
            # already used by buyable/sellable/cancel/revise above.
            raise HTTPException(502, f"kill-switch engage failed: {exc}") from exc
    else:
        try:
            await safety_switch.disengage_kill_switch()
        except Exception as exc:
            raise HTTPException(502, f"kill-switch disengage failed: {exc}") from exc
    engaged, reason, _state_unknown = await safety_switch.kill_switch_status()
    return {"kill_switch_engaged": engaged, "kill_switch_reason": reason}


async def _pending_order_payload(order: pending_orders.PendingOrder) -> dict:
    # name/currency: same state.get_display_name source as every other panel
    # (holdings, decisions, trade history) — the pending-order tab/modal
    # previously showed only the bare ticker code, unreadable for anyone not
    # fluent in KRX codes. Covers tickers never added to the trade allowlist too
    # (e.g. a FULL_AUTO screener candidate), unlike reading state[ticker]
    # client-side, which is only populated for the current trade allowlist.
    name = await state.get_display_name(order.ticker)
    info = state._display_info.get(order.ticker)
    currency = info.currency if info else ("KRW" if order.ticker.endswith(".KS") else "USD")

    # `price` is the request-time reference price (a market order's fill
    # isn't known until KIS actually matches it) — for an approved order
    # whose resolution logged a KIS order_no, look up that order's real
    # fill price from KIS's own execution history (the SAME source
    # /api/trades reads — the internal `trades` ledger table is a
    # different, KIS-independent paper-ledger record with no order_no on
    # it at all, so that was the wrong place to look) so the history list
    # can show both instead of only the stale pre-fill estimate forever.
    # None for anything not yet approved+filled (still pending, rejected,
    # approval failed, or the execution isn't in the last 7 days KIS keeps).
    fill_price = None
    if order.resolution:
        m = _ORDER_NO_RE.search(order.resolution)
        if m:
            order_no = m.group(1)
            domestic, overseas = await state.recent_executions()
            match = next((e for e in [*domestic, *overseas] if e.order_no == order_no), None)
            if match is not None:
                fill_price = match.avg_price

    return {
        "id": order.id,
        "ticker": order.ticker,
        "name": name,
        "currency": currency,
        "trade_date": order.trade_date.isoformat(),
        "side": order.side,
        "shares": order.shares,
        "price": order.price,
        "fill_price": fill_price,
        "reason": order.reason,
        "status": order.status,
        "resolution": order.resolution,
        "created_at": order.created_at.isoformat(),
        "resolved_at": order.resolved_at.isoformat() if order.resolved_at else None,
    }


@router.get("/api/pending-orders")
async def list_pending_orders_endpoint() -> list[dict]:
    """Every AI-proposed order still awaiting explicit human approval —
    SEMI_AUTO mode's every order, plus any FULL_AUTO order the approval
    threshold held (see app.config.ExecutionMode)."""
    try:
        orders = await pending_orders.list_pending_orders()
    except Exception as exc:
        # list_pending_orders/list_resolved_orders have no internal
        # try/except — a transient DB hiccup previously reached the client
        # as a raw 500 with a plain-text body. Same "translate to a real
        # HTTP error" convention as the kill-switch fix (confirmed live
        # 2026-08-06: 10 unhandled DB-timeout crashes across these two
        # pending-order endpoints in one day).
        raise HTTPException(502, f"failed to load pending orders: {exc}") from exc
    return [await _pending_order_payload(o) for o in orders]


@router.get("/api/pending-orders/history")
async def pending_order_history_endpoint(limit: int = 50) -> list[dict]:
    """Most-recently-resolved (approved/approve_failed/rejected) orders —
    the record of what a human decided on each approval-modal prompt.
    `resolution` carries the KIS disposition string for an approval
    (submitted/dry-run/FAILED/...) or the rejection reason."""
    try:
        orders = await pending_orders.list_resolved_orders(limit)
    except Exception as exc:
        raise HTTPException(502, f"failed to load pending order history: {exc}") from exc
    return [await _pending_order_payload(o) for o in orders]


@router.post("/api/pending-orders/{order_id}/approve", dependencies=_needs_kis)
async def approve_pending_order_endpoint(order_id: int) -> dict:
    """Re-runs the full current risk-gate chain (not the state from when the
    proposal was created) and submits if it still clears — see
    pending_orders.approve_pending_order's own docstring."""
    try:
        disposition = await pending_orders.approve_pending_order(order_id, broker=await state.kis_broker())
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return {"id": order_id, "disposition": disposition}


@router.post("/api/pending-orders/{order_id}/reject")
async def reject_pending_order_endpoint(order_id: int, req: RejectPendingOrderRequest) -> dict:
    try:
        await pending_orders.reject_pending_order(order_id, req.reason)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return {"id": order_id, "status": "rejected"}
