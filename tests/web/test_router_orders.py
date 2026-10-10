"""orders router — exec-mode badge, risk-rule settings, buyable/sellable, and
order placement. KIS broker/kis_bridge stubbed, no network (E-stage)."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.config import settings
from app.execution import risk_rules, runtime_settings
from app.web import state
from app.config import current_account_mode
from app.user_context import get_current_user_id
from conftest import broker_stub, bypass_kis_gate
from app.web.routers import orders


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(orders.router)
    bypass_kis_gate(app)
    return TestClient(app, base_url="http://127.0.0.1:8094")


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.setattr(risk_rules, "_overrides", {})
    monkeypatch.setattr(runtime_settings, "_overrides", {})
    orig_broker = state.kis_broker
    state._cancelable_orders_by_scope.clear()
    yield
    state.kis_broker = orig_broker
    state._cancelable_orders_by_scope.clear()


def test_exec_mode_reflects_settings_and_broker_configured(client, monkeypatch):
    from app.config import ExecutionMode

    monkeypatch.setattr(settings, "kis_execution_enabled", True)
    monkeypatch.setattr(settings, "kis_execution_dry_run", True)
    monkeypatch.setattr(settings, "execution_mode", ExecutionMode.MANUAL)
    state.kis_broker = broker_stub(object())
    resp = client.get("/api/exec-mode")
    assert resp.json() == {
        "enabled": True, "dry_run": True, "broker_configured": True, "execution_mode": "manual",
    }


def test_exec_mode_broker_not_configured(client):
    state.kis_broker = broker_stub(None)
    resp = client.get("/api/exec-mode")
    assert resp.json()["broker_configured"] is False


def test_get_risk_rules_includes_approval_fields(client, monkeypatch):
    monkeypatch.setattr(settings, "kis_max_order_krw", 5_000_000.0)
    resp = client.get("/api/settings/risk")
    assert resp.status_code == 200
    body = resp.json()
    assert "approval_enabled" in body
    assert "approval_threshold_krw" in body
    assert body["values"]["kis_max_order_krw"] == 5_000_000.0


def test_save_risk_rules_rejects_empty_payload(client):
    resp = client.post("/api/settings/risk", json={"values": {}})
    assert resp.status_code == 400


async def test_save_risk_rules_rejects_invalid_value(client, monkeypatch):
    async def fake_save(values, account_mode=None):
        raise ValueError("kis_max_order_krw must be positive")

    monkeypatch.setattr(risk_rules, "save", fake_save)
    resp = client.post("/api/settings/risk", json={"values": {"kis_max_order_krw": -1}})
    assert resp.status_code == 400


async def test_save_risk_rules_returns_503_when_store_unreachable(client, monkeypatch):
    async def fake_save(values, account_mode=None):
        raise RuntimeError("db unreachable")

    monkeypatch.setattr(risk_rules, "save", fake_save)
    resp = client.post("/api/settings/risk", json={"values": {"kis_max_order_krw": 1000}})
    assert resp.status_code == 503


def test_get_runtime_settings_includes_read_only_account_mode(client, monkeypatch):
    monkeypatch.setattr(settings, "kis_account_mode", "paper")
    resp = client.get("/api/settings/runtime")
    assert resp.status_code == 200
    body = resp.json()
    assert body["kis_account_mode"] == "paper"
    assert body["values"]["ai_profile"] == "production"


def test_save_runtime_settings_rejects_empty_payload(client):
    resp = client.post("/api/settings/runtime", json={})
    assert resp.status_code == 400


def test_save_runtime_settings_updates_resolved_value(client, monkeypatch):
    async def fake_save(values, user_id=None, account_mode=None):
        # _overrides is keyed by (user, account_mode) — user alone since Phase
        # 4, mode added 2026-08-06 so a paper override cannot answer a real
        # read. A flat update here would write a key rule() never looks at.
        who = user_id or get_current_user_id()
        mode = account_mode or current_account_mode()
        runtime_settings._overrides.setdefault((who, mode), {}).update(
            {k: runtime_settings._coerce(k, v) for k, v in values.items()}
        )
        return runtime_settings.current(who, mode)

    monkeypatch.setattr(runtime_settings, "save", fake_save)
    resp = client.post("/api/settings/runtime", json={"execution_mode": "full_auto"})
    assert resp.status_code == 200
    assert resp.json()["values"]["execution_mode"] == "full_auto"


async def test_save_runtime_settings_rejects_real_account_safety_violation(client, monkeypatch):
    async def fake_save(values, account_mode=None):
        raise ValueError("ai_profile_live_ack=True required")

    monkeypatch.setattr(runtime_settings, "save", fake_save)
    resp = client.post("/api/settings/runtime", json={"ai_profile": "test_aggressive"})
    assert resp.status_code == 400


async def test_save_runtime_settings_returns_503_when_store_unreachable(client, monkeypatch):
    async def fake_save(values, account_mode=None):
        raise RuntimeError("db unreachable")

    monkeypatch.setattr(runtime_settings, "save", fake_save)
    resp = client.post("/api/settings/runtime", json={"execution_mode": "manual"})
    assert resp.status_code == 503


def test_buyable_returns_503_when_broker_not_configured(client):
    state.kis_broker = broker_stub(None)
    resp = client.get("/api/buyable/005930.KS")
    assert resp.status_code == 503


def test_sellable_returns_503_when_broker_not_configured(client):
    state.kis_broker = broker_stub(None)
    resp = client.get("/api/sellable/005930.KS")
    assert resp.status_code == 503


async def test_buyable_us_symbol_reports_cash_when_currency_matched(client, monkeypatch):
    from app.domain.models import OverseasCash

    class _FakeBroker:
        async def fetch_overseas_present_balance(self, currency="USD"):
            return OverseasCash(currency=currency, deposit=1000.0, orderable=800.0, eval_amount=0.0, matched=True)

    state.kis_broker = broker_stub(_FakeBroker())
    resp = client.get("/api/buyable/AAPL")
    assert resp.status_code == 200
    body = resp.json()
    assert body == {"available": True, "cash": 800.0, "max_qty": None, "max_amount": 1000.0, "currency": "USD"}


async def test_buyable_us_symbol_unavailable_when_currency_unmatched(client, monkeypatch):
    # Regression: a currency-mismatched present-balance response used to be
    # reported as a confirmed $0 cash figure instead of "not available."
    from app.domain.models import OverseasCash

    class _FakeBroker:
        async def fetch_overseas_present_balance(self, currency="USD"):
            return OverseasCash(currency=currency, deposit=0.0, orderable=0.0, eval_amount=0.0, matched=False)

    state.kis_broker = broker_stub(_FakeBroker())
    resp = client.get("/api/buyable/AAPL")
    assert resp.status_code == 200
    body = resp.json()
    assert body["available"] is False


async def test_sellable_reads_held_quantity_from_cached_snapshot(client, monkeypatch):
    state.kis_broker = broker_stub(object())

    async def fake_snapshot():
        return {}, {"005930.KS": {"quantity": 10}}

    monkeypatch.setattr(state, "kis_account_snapshot", fake_snapshot)
    resp = client.get("/api/sellable/005930.KS")
    assert resp.status_code == 200
    assert resp.json() == {"sellable_qty": 10}


class _FakeOrderResult:
    def __init__(self, ok=True, status="filled", message="ok", order_no="123", org_no="ORG1",
                 ticker="AAPL", side="buy", shares=1, price=150.0):
        self.ok = ok
        self.status = status
        self.message = message
        self.order_no = order_no
        self.org_no = org_no
        self.ticker = ticker
        self.side = side
        self.shares = shares
        self.price = price


async def test_place_order_records_market_order_to_ledger(client, monkeypatch):
    recorded = []

    async def fake_place_manual_order(*args, **kwargs):
        return _FakeOrderResult()

    async def fake_record_trade(*args, **kwargs):
        recorded.append((args, kwargs))

    monkeypatch.setattr(orders, "place_manual_order", fake_place_manual_order)
    monkeypatch.setattr(orders, "record_trade", fake_record_trade)

    resp = client.post("/api/order", json={"ticker": "AAPL", "side": "buy", "shares": 1})
    assert resp.status_code == 200
    assert resp.json()["status"] == "filled"
    assert len(recorded) == 1
    entry = state.cancelable_orders()["123"]
    assert "AAPL" in entry["org_no"] or entry["org_no"] == "ORG1"


async def test_place_order_does_not_record_unfilled_order(client, monkeypatch):
    recorded = []

    async def fake_place_manual_order(*args, **kwargs):
        return _FakeOrderResult(ok=False, status="blocked", order_no=None, org_no=None)

    async def fake_record_trade(*args, **kwargs):
        recorded.append(1)

    monkeypatch.setattr(orders, "place_manual_order", fake_place_manual_order)
    monkeypatch.setattr(orders, "record_trade", fake_record_trade)

    resp = client.post("/api/order", json={"ticker": "AAPL", "side": "buy", "shares": 1})
    assert resp.json()["status"] == "blocked"
    assert recorded == []


async def test_nan_limit_price_reaches_gate_as_nan_and_is_blocked(client, monkeypatch):
    """A literal JSON `NaN` limit_price (not valid JSON, but both Python's
    `json` module and pydantic parse the literal token without complaint)
    used to sail through the gate chain unblocked because any `>`/`>=`
    comparison against float('nan') is False in Python — see
    tests/execution/test_kis_bridge.py for the direct gate-function
    regression tests. This confirms the HTTP layer hands the NaN through to
    place_manual_order verbatim (not stripped/rejected by pydantic) and that
    a blocked disposition still surfaces as a normal HTTP 200 with a
    "blocked" status in the body, matching place_manual_order's own
    "never raises" contract."""
    import math

    seen = {}

    async def fake_place(ticker, side, shares, **kw):
        seen.update(kw)
        return _FakeOrderResult(ok=False, status="blocked", order_no=None, org_no=None)

    monkeypatch.setattr(orders, "place_manual_order", fake_place)
    state.kis_broker = broker_stub(object())

    resp = client.post(
        "/api/order",
        content=b'{"ticker":"005930.KS","side":"buy","shares":10,"order_type":"limit","limit_price":NaN}',
        headers={"content-type": "application/json"},
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "blocked"
    assert math.isnan(seen["limit_price"])


def test_cancel_order_rejects_unknown_order_no(client):
    state.kis_broker = broker_stub(object())
    resp = client.post("/api/order/cancel", json={"order_no": "does-not-exist"})
    assert resp.status_code == 409


def test_open_orders_empty_when_broker_not_configured(client):
    state.kis_broker = broker_stub(None)
    resp = client.get("/api/orders/open")
    assert resp.json() == []


def _overseas_exec(order_no: str, code: str, *, rejected: bool):
    from app.domain.models import BrokerExecution

    return BrokerExecution(
        order_date="20260818", order_no=order_no, code=code, name=code,
        side="buy", order_qty=6, filled_qty=0, remaining_qty=6,
        avg_price=0.0, filled_amount=0.0, rejected=rejected,
    )


def test_open_orders_excludes_rejected_overseas_orders(client, monkeypatch):
    # A KIS-rejected overseas order keeps its full 미체결수량, so without the
    # `rejected` filter it showed as an open order for the whole 7-day lookback
    # window and could never be dismissed (overseas rows are cancelable=False).
    state.kis_broker = broker_stub(object())

    async def fake_recent_executions():
        return [], [
            _overseas_exec("0031000", "SNDK", rejected=True),
            _overseas_exec("0031001", "AAPL", rejected=False),
        ]

    monkeypatch.setattr(orders.state, "recent_executions", fake_recent_executions)

    rows = client.get("/api/orders/open").json()
    assert [r["ticker"] for r in rows] == ["AAPL"]
    assert rows[0]["remaining_qty"] == 6


# ----------------- safety switch (agent-architecture-roadmap stage 2) -----------------


async def test_execution_safety_status_reports_kill_switch_and_daily_loss(client, monkeypatch):
    async def fake_kill_switch_status():
        return True, "manual halt", False

    async def fake_daily_loss_status(market):
        return {"available": True, "breached": False, "market": market}

    monkeypatch.setattr(orders.safety_switch, "kill_switch_status", fake_kill_switch_status)
    monkeypatch.setattr(orders.safety_switch, "daily_loss_status", fake_daily_loss_status)

    resp = client.get("/api/execution/safety")
    assert resp.status_code == 200
    body = resp.json()
    assert body["kill_switch_engaged"] is True
    assert body["kill_switch_reason"] == "manual halt"
    assert body["kill_switch_state_unknown"] is False
    assert body["daily_loss"]["domestic"]["breached"] is False
    assert body["daily_loss"]["overseas"]["breached"] is False


async def test_safety_status_skips_overseas_daily_loss_when_disabled(client, monkeypatch):
    calls = []

    async def fake_kill_switch_status():
        return False, None, False

    async def fake_daily_loss_status(market):
        calls.append(market)
        return {"available": True, "breached": False, "market": market}

    monkeypatch.setattr(orders.settings, "overseas_trading_enabled", False)
    monkeypatch.setattr(orders.portfolio_risk, "overseas_flag_contradicted", lambda: False)
    monkeypatch.setattr(orders.safety_switch, "kill_switch_status", fake_kill_switch_status)
    monkeypatch.setattr(orders.safety_switch, "daily_loss_status", fake_daily_loss_status)

    resp = client.get("/api/execution/safety")
    assert resp.status_code == 200
    body = resp.json()
    assert calls == ["domestic"]  # overseas never called
    assert body["daily_loss"]["overseas"] == {"available": False, "reason": "overseas_disabled"}


async def test_safety_status_still_reports_domestic_when_overseas_disabled(client, monkeypatch):
    async def fake_kill_switch_status():
        return False, None, False

    async def fake_daily_loss_status(market):
        return {"available": True, "breached": False, "market": market}

    monkeypatch.setattr(orders.settings, "overseas_trading_enabled", False)
    monkeypatch.setattr(orders.portfolio_risk, "overseas_flag_contradicted", lambda: False)
    monkeypatch.setattr(orders.safety_switch, "kill_switch_status", fake_kill_switch_status)
    monkeypatch.setattr(orders.safety_switch, "daily_loss_status", fake_daily_loss_status)

    resp = client.get("/api/execution/safety")
    body = resp.json()
    assert body["daily_loss"]["domestic"] == {"available": True, "breached": False, "market": "domestic"}


async def test_set_kill_switch_engage_requires_reason(client):
    resp = client.post("/api/execution/safety/kill-switch", json={"engaged": True, "reason": "   "})
    assert resp.status_code == 400


async def test_set_kill_switch_engage_calls_engage_with_reason(client, monkeypatch):
    calls = []

    async def fake_engage(reason):
        calls.append(reason)

    async def fake_status():
        return True, "emergency stop", False

    monkeypatch.setattr(orders.safety_switch, "engage_kill_switch", fake_engage)
    monkeypatch.setattr(orders.safety_switch, "kill_switch_status", fake_status)

    resp = client.post("/api/execution/safety/kill-switch", json={"engaged": True, "reason": "emergency stop"})
    assert resp.status_code == 200
    assert calls == ["emergency stop"]
    assert resp.json() == {"kill_switch_engaged": True, "kill_switch_reason": "emergency stop"}


async def test_set_kill_switch_disengage_calls_disengage(client, monkeypatch):
    calls = []

    async def fake_disengage():
        calls.append(1)

    async def fake_status():
        return False, None, False

    monkeypatch.setattr(orders.safety_switch, "disengage_kill_switch", fake_disengage)
    monkeypatch.setattr(orders.safety_switch, "kill_switch_status", fake_status)

    resp = client.post("/api/execution/safety/kill-switch", json={"engaged": False})
    assert resp.status_code == 200
    assert calls == [1]
    assert resp.json() == {"kill_switch_engaged": False, "kill_switch_reason": None}


# ---------------------------------------------------------------------------
# engage_kill_switch/disengage_kill_switch have no internal try/except
# (unlike kill_switch_status's own fail-closed read) — confirmed live
# 2026-08-06: a transient DB timeout there used to reach the client as a raw
# 500 with a plain-text body, which the dashboard's `r.json()` call then
# failed to parse, surfacing a confusing SyntaxError instead of the real
# problem. These confirm the router now translates that into a real
# HTTPException (JSON body, same 502 convention as buyable/sellable/cancel/
# revise elsewhere in this file) instead of letting it crash raw.
# ---------------------------------------------------------------------------


async def test_set_kill_switch_disengage_db_failure_returns_502_json(client, monkeypatch):
    async def fake_disengage():
        raise TimeoutError("db timeout")

    monkeypatch.setattr(orders.safety_switch, "disengage_kill_switch", fake_disengage)

    resp = client.post("/api/execution/safety/kill-switch", json={"engaged": False})
    assert resp.status_code == 502
    assert "kill-switch disengage failed" in resp.json()["detail"]


async def test_set_kill_switch_engage_db_failure_returns_502_json(client, monkeypatch):
    async def fake_engage(reason):
        raise TimeoutError("db timeout")

    monkeypatch.setattr(orders.safety_switch, "engage_kill_switch", fake_engage)

    resp = client.post("/api/execution/safety/kill-switch", json={"engaged": True, "reason": "test"})
    assert resp.status_code == 502
    assert "kill-switch engage failed" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# pending-order endpoints
# ---------------------------------------------------------------------------


def _fake_pending_order(order_id=1, status="pending", resolution=None):
    from datetime import date, datetime

    from app.execution.pending_orders import PendingOrder

    return PendingOrder(
        id=order_id, ticker="005930.KS", trade_date=date(2026, 7, 16), side="buy", shares=5,
        price=150_000.0, reason="NEEDS_APPROVAL: over threshold", status=status, resolution=resolution,
        created_at=datetime(2026, 7, 16, 9, 0, 0), resolved_at=None,
    )


async def test_list_pending_orders_returns_payload(client, monkeypatch):
    async def fake_list():
        return [_fake_pending_order()]

    monkeypatch.setattr(orders.pending_orders, "list_pending_orders", fake_list)
    resp = client.get("/api/pending-orders")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == 1
    assert body[0]["ticker"] == "005930.KS"
    assert body[0]["status"] == "pending"


# list_pending_orders/list_resolved_orders have no internal try/except —
# confirmed live 2026-08-06: 10 unhandled DB-timeout crashes across these two
# endpoints in one day, same raw-500-plain-text-body bug already fixed for
# the kill switch above.
async def test_list_pending_orders_db_failure_returns_502_json(client, monkeypatch):
    async def fake_list():
        raise TimeoutError("db timeout")

    monkeypatch.setattr(orders.pending_orders, "list_pending_orders", fake_list)
    resp = client.get("/api/pending-orders")
    assert resp.status_code == 502
    assert "failed to load pending orders" in resp.json()["detail"]


async def test_pending_order_history_db_failure_returns_502_json(client, monkeypatch):
    async def fake_history(limit=50):
        raise TimeoutError("db timeout")

    monkeypatch.setattr(orders.pending_orders, "list_resolved_orders", fake_history)
    resp = client.get("/api/pending-orders/history")
    assert resp.status_code == 502
    assert "failed to load pending order history" in resp.json()["detail"]


async def test_approve_pending_order_returns_disposition(client, monkeypatch):
    async def fake_approve(order_id, *, broker=None):
        assert order_id == 1
        return "submitted: mock order"

    monkeypatch.setattr(orders.pending_orders, "approve_pending_order", fake_approve)
    resp = client.post("/api/pending-orders/1/approve")
    assert resp.status_code == 200
    assert resp.json() == {"id": 1, "disposition": "submitted: mock order"}


async def test_approve_pending_order_404_when_not_found(client, monkeypatch):
    async def fake_approve(order_id, *, broker=None):
        raise ValueError("pending order 1 not found or already resolved")

    monkeypatch.setattr(orders.pending_orders, "approve_pending_order", fake_approve)
    resp = client.post("/api/pending-orders/1/approve")
    assert resp.status_code == 404


async def test_reject_pending_order_returns_status(client, monkeypatch):
    captured = {}

    async def fake_reject(order_id, reason=""):
        captured.update(order_id=order_id, reason=reason)

    monkeypatch.setattr(orders.pending_orders, "reject_pending_order", fake_reject)
    resp = client.post("/api/pending-orders/1/reject", json={"reason": "changed my mind"})
    assert resp.status_code == 200
    assert resp.json() == {"id": 1, "status": "rejected"}
    assert captured == {"order_id": 1, "reason": "changed my mind"}


async def test_reject_pending_order_404_when_not_found(client, monkeypatch):
    async def fake_reject(order_id, reason=""):
        raise ValueError("pending order 1 not found or already resolved")

    monkeypatch.setattr(orders.pending_orders, "reject_pending_order", fake_reject)
    resp = client.post("/api/pending-orders/1/reject", json={})
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Cancel registry does not cross an account-mode switch (moved from
# tests/web/test_zz_audit_scratch.py::test_A_cancel_registry_crosses_account_mode_switch
# once app.web.state._cancelable_orders_by_scope fixed the underlying leak).
# ---------------------------------------------------------------------------
async def test_cancel_registry_does_not_cross_account_mode_switch(client, monkeypatch):
    """Place an order while in PAPER, switch to REAL (the documented switch
    teardown: invalidate_cached_mode + invalidate_broker_cache + stop feed),
    then try to cancel by that order_no. The paper-mode registry entry must
    not be visible from real-mode scope, so the cancel must 409 instead of
    reaching the REAL broker with the PAPER order's org_no/qty."""
    monkeypatch.setattr(settings, "kis_account_mode", "paper")

    async def fake_place(*a, **k):
        return _FakeOrderResult(order_no="0000012345", org_no="ORG1")

    async def fake_record(*a, **k):
        return None

    monkeypatch.setattr(orders, "place_manual_order", fake_place)
    monkeypatch.setattr(orders, "record_trade", fake_record)
    monkeypatch.setattr(state, "invalidate_cache", lambda *a, **k: None)

    class PaperBroker:
        is_real = False

    class RealBroker:
        is_real = True

        def __init__(self):
            self.calls = []

        async def cancel_order(self, order_no, org_no, shares, price=None, ord_dvsn="00"):
            self.calls.append((order_no, org_no, shares, price, ord_dvsn))

            class R:
                message = "cancelled on REAL account"

            return R()

    state.kis_broker = broker_stub(PaperBroker())
    r = client.post("/api/order", json={"ticker": "005930.KS", "side": "buy",
                                         "shares": 10, "order_type": "limit",
                                         "limit_price": 71000})
    assert r.status_code == 200, r.text
    assert "0000012345" in state.cancelable_orders()

    # --- simulate the account-mode switch performed by PUT /api/account-mode
    from app.web.account_mode_middleware import invalidate_cached_mode
    invalidate_cached_mode()
    real = RealBroker()
    state.kis_broker = broker_stub(real)
    monkeypatch.setattr(settings, "kis_account_mode", "real")

    r2 = client.post("/api/order/cancel", json={"order_no": "0000012345"})
    assert r2.status_code == 409, r2.text
    assert not real.calls, "registry crossed the mode switch and drove the REAL broker"


# ---------------------------------------------------------------------------
# Finding C (moved from tests/web/test_zz_audit_scratch.py::test_C_order_
# placed_but_ledger_write_raises): a ledger-write failure after a REAL KIS
# submission must never look like the order itself failed — the order is
# already live at KIS by the time record_trade runs.
# ---------------------------------------------------------------------------
async def test_place_order_registers_cancel_handle_even_when_ledger_write_fails(client, monkeypatch):
    async def fake_place(*a, **k):
        return _FakeOrderResult(order_no="0000012345", org_no="ORG1")

    async def boom(*a, **k):
        raise RuntimeError("DB timeout")

    async def fake_alert(alert):
        pass

    monkeypatch.setattr(orders, "place_manual_order", fake_place)
    monkeypatch.setattr(orders, "record_trade", boom)
    monkeypatch.setattr(orders, "record_alert", fake_alert)
    monkeypatch.setattr(state, "invalidate_cache", lambda *a, **k: None)
    state.kis_broker = broker_stub(object())

    resp = client.post("/api/order", json={"ticker": "005930.KS", "side": "buy", "shares": 10})

    assert resp.status_code == 200
    assert resp.json()["status"] == "filled"
    assert resp.json()["ledger_recorded"] is False
    assert "0000012345" in state.cancelable_orders()


async def test_place_order_reports_ledger_not_recorded_when_record_trade_returns_none(client, monkeypatch):
    """record_trade's OWN documented contract (app.execution.ledger) is to
    degrade quietly and return None on a DB failure — no exception at all.
    This is the more common real-world shape of Finding C than an outright
    raise, and must be reported (and alerted) the same way."""
    alerts = []

    async def fake_place(*a, **k):
        return _FakeOrderResult(order_no="0000012345", org_no="ORG1")

    async def fake_record(*a, **k):
        return None

    async def fake_alert(alert):
        alerts.append(alert)

    monkeypatch.setattr(orders, "place_manual_order", fake_place)
    monkeypatch.setattr(orders, "record_trade", fake_record)
    monkeypatch.setattr(orders, "record_alert", fake_alert)
    monkeypatch.setattr(state, "invalidate_cache", lambda *a, **k: None)
    state.kis_broker = broker_stub(object())

    resp = client.post("/api/order", json={"ticker": "005930.KS", "side": "buy", "shares": 10})

    assert resp.status_code == 200
    assert resp.json()["ledger_recorded"] is False
    assert "0000012345" in state.cancelable_orders()
    assert len(alerts) == 1
    assert alerts[0].kind == "manual_order_unrecorded"


async def test_place_order_reports_ledger_recorded_on_success(client, monkeypatch):
    alerts = []

    async def fake_place(*a, **k):
        return _FakeOrderResult(order_no="0000012345", org_no="ORG1")

    async def fake_record(*a, **k):
        return 7

    async def fake_alert(alert):
        alerts.append(alert)

    monkeypatch.setattr(orders, "place_manual_order", fake_place)
    monkeypatch.setattr(orders, "record_trade", fake_record)
    monkeypatch.setattr(orders, "record_alert", fake_alert)
    monkeypatch.setattr(state, "invalidate_cache", lambda *a, **k: None)
    state.kis_broker = broker_stub(object())

    resp = client.post("/api/order", json={"ticker": "005930.KS", "side": "buy", "shares": 10})

    assert resp.json()["ledger_recorded"] is True
    assert alerts == []


async def test_place_order_records_limit_order_to_ledger(client, monkeypatch):
    """2026-09-04 real-incident fix: a manual LIMIT sell (057680.KS) never
    got a ledger row at all — this endpoint used to record ONLY market
    orders. A limit order must now be recorded optimistically at submission
    exactly like the market path already was, so it lands inside
    fill_monitor's existing detect-only envelope instead of being silently
    unrecorded forever (see the docstring above this endpoint's ledger
    block)."""
    recorded = []

    async def fake_place(*a, **k):
        return _FakeOrderResult(order_no="0000012345", org_no="ORG1")

    async def fake_record(*a, **k):
        recorded.append((a, k))
        return 1

    monkeypatch.setattr(orders, "place_manual_order", fake_place)
    monkeypatch.setattr(orders, "record_trade", fake_record)
    monkeypatch.setattr(state, "invalidate_cache", lambda *a, **k: None)
    state.kis_broker = broker_stub(object())

    resp = client.post("/api/order", json={"ticker": "005930.KS", "side": "buy",
                                            "shares": 10, "order_type": "limit", "limit_price": 71000})

    assert resp.json()["ledger_recorded"] is True
    assert len(recorded) == 1
    # Cancel-handle registration (ord_dvsn/price) still happens correctly,
    # same as before this fix.
    entry = state.cancelable_orders()["0000012345"]
    assert entry["ord_dvsn"] == "00"
    assert entry["price"] == 71000


async def test_place_order_limit_records_kis_order_no_and_submitted_prefix(client, monkeypatch):
    recorded = []

    async def fake_place(*a, **k):
        return _FakeOrderResult(order_no="0000012345", org_no="ORG1")

    async def fake_record(*a, **k):
        recorded.append((a, k))
        return 1

    monkeypatch.setattr(orders, "place_manual_order", fake_place)
    monkeypatch.setattr(orders, "record_trade", fake_record)
    monkeypatch.setattr(state, "invalidate_cache", lambda *a, **k: None)
    state.kis_broker = broker_stub(object())

    resp = client.post("/api/order", json={"ticker": "005930.KS", "side": "sell",
                                            "shares": 10, "order_type": "limit", "limit_price": 71000})

    assert resp.status_code == 200
    assert len(recorded) == 1
    _args, kwargs = recorded[0]
    assert kwargs["kis_order_no"] == "0000012345"
    assert kwargs["mirror_disposition"].startswith("submitted: ")


async def test_place_order_limit_records_limit_price_not_reference_price(client, monkeypatch):
    """result.price is exec_price (kis_bridge.place_manual_order) — the
    LIMIT price for a limit order, never the request's own reference_price.
    This confirms the recorded trade price matches what actually went out,
    not whatever reference figure the UI happened to show."""
    recorded = []

    async def fake_place(*a, **k):
        # exec_price mirrors kis_bridge's own resolution: limit_price for a
        # limit order, distinct from the request's reference_price (99999).
        return _FakeOrderResult(order_no="0000012345", org_no="ORG1", price=71000.0)

    async def fake_record(ticker, trade_date, side, shares, price, **k):
        recorded.append(price)
        return 1

    monkeypatch.setattr(orders, "place_manual_order", fake_place)
    monkeypatch.setattr(orders, "record_trade", fake_record)
    monkeypatch.setattr(state, "invalidate_cache", lambda *a, **k: None)
    state.kis_broker = broker_stub(object())

    resp = client.post("/api/order", json={
        "ticker": "005930.KS", "side": "sell", "shares": 10, "order_type": "limit",
        "limit_price": 71000, "reference_price": 99999,
    })

    assert resp.status_code == 200
    assert recorded == [71000.0]


async def test_place_order_limit_ledger_failure_alerts_and_does_not_500(client, monkeypatch):
    alerts = []

    async def fake_place(*a, **k):
        return _FakeOrderResult(order_no="0000012345", org_no="ORG1")

    async def fake_record(*a, **k):
        return None

    async def fake_alert(alert):
        alerts.append(alert)

    monkeypatch.setattr(orders, "place_manual_order", fake_place)
    monkeypatch.setattr(orders, "record_trade", fake_record)
    monkeypatch.setattr(orders, "record_alert", fake_alert)
    monkeypatch.setattr(state, "invalidate_cache", lambda *a, **k: None)
    state.kis_broker = broker_stub(object())

    resp = client.post("/api/order", json={"ticker": "005930.KS", "side": "sell",
                                            "shares": 10, "order_type": "limit", "limit_price": 71000})

    assert resp.status_code == 200
    assert resp.json()["ledger_recorded"] is False
    assert len(alerts) == 1
    assert alerts[0].kind == "manual_order_unrecorded"


async def test_place_order_limit_registers_cancel_handle_before_ledger_write(client, monkeypatch):
    async def fake_place(*a, **k):
        return _FakeOrderResult(order_no="0000012345", org_no="ORG1")

    async def boom(*a, **k):
        raise RuntimeError("DB timeout")

    async def fake_alert(alert):
        pass

    monkeypatch.setattr(orders, "place_manual_order", fake_place)
    monkeypatch.setattr(orders, "record_trade", boom)
    monkeypatch.setattr(orders, "record_alert", fake_alert)
    monkeypatch.setattr(state, "invalidate_cache", lambda *a, **k: None)
    state.kis_broker = broker_stub(object())

    resp = client.post("/api/order", json={"ticker": "005930.KS", "side": "sell",
                                            "shares": 10, "order_type": "limit", "limit_price": 71000})

    assert resp.status_code == 200
    assert resp.json()["ledger_recorded"] is False
    # The cancel handle was registered before the ledger write raised.
    assert "0000012345" in state.cancelable_orders()


# ---------------------------------------------------------------------------
# Finding D (moved from tests/web/test_zz_audit_scratch.py::test_D_revise_
# empty_order_no_orphans_registry): an accepted revise that returns no (or
# the same) new order_no must not lose the cancel-registry entry.
# ---------------------------------------------------------------------------
async def test_revise_keeps_cancel_handle_when_broker_returns_no_new_order_no(client, monkeypatch):
    class Broker:
        def __init__(self):
            self.cancel_calls = []

        async def revise_order(self, order_no, org_no, shares, new_price, ord_dvsn="00"):
            class R:
                order_no = ""  # KIS accepted but returned no new ODNO
                message = "정정 접수"

            return R()

        async def cancel_order(self, order_no, org_no, shares, price=None, ord_dvsn="00"):
            self.cancel_calls.append((order_no, org_no, shares, price, ord_dvsn))

            class R:
                message = "cancelled"

            return R()

    broker = Broker()
    state.kis_broker = broker_stub(broker)
    monkeypatch.setattr(state, "invalidate_cache", lambda *a, **k: None)
    state.cancelable_orders()["0000012345"] = {
        "org_no": "ORG1", "shares": 10, "ord_dvsn": "00", "price": 71000.0,
    }

    r = client.post("/api/order/revise", json={"order_no": "0000012345", "new_price": 72000})
    assert r.status_code == 200, r.text
    assert r.json()["order_no"] == "0000012345"
    entry = state.cancelable_orders()["0000012345"]
    assert entry["price"] == 72000

    r2 = client.post("/api/order/cancel", json={"order_no": "0000012345"})
    assert r2.status_code == 200, r2.text
    assert broker.cancel_calls == [("0000012345", "ORG1", 10, 72000, "00")]


async def test_revise_keeps_cancel_handle_when_broker_returns_same_order_no(client, monkeypatch):
    class Broker:
        def __init__(self):
            self.cancel_calls = []

        async def revise_order(self, order_no, org_no, shares, new_price, ord_dvsn="00"):
            class R:
                pass

            R.order_no = order_no  # same number as the original
            R.message = "정정 접수"
            return R()

        async def cancel_order(self, order_no, org_no, shares, price=None, ord_dvsn="00"):
            self.cancel_calls.append((order_no, org_no, shares, price, ord_dvsn))

            class R:
                message = "cancelled"

            return R()

    broker = Broker()
    state.kis_broker = broker_stub(broker)
    monkeypatch.setattr(state, "invalidate_cache", lambda *a, **k: None)
    state.cancelable_orders()["0000012345"] = {
        "org_no": "ORG1", "shares": 10, "ord_dvsn": "00", "price": 71000.0,
    }

    r = client.post("/api/order/revise", json={"order_no": "0000012345", "new_price": 72000})
    assert r.status_code == 200, r.text
    entry = state.cancelable_orders()["0000012345"]
    assert entry["price"] == 72000

    r2 = client.post("/api/order/cancel", json={"order_no": "0000012345"})
    assert r2.status_code == 200, r2.text
    assert broker.cancel_calls == [("0000012345", "ORG1", 10, 72000, "00")]


async def test_revise_moves_cancel_handle_to_new_order_no(client, monkeypatch):
    """The normal path — KIS returns a genuinely new order number — must
    still move the registry entry, not always keep the old key."""
    class Broker:
        async def revise_order(self, order_no, org_no, shares, new_price, ord_dvsn="00"):
            class R:
                order_no = "0000099999"
                message = "정정 접수"

            return R()

    state.kis_broker = broker_stub(Broker())
    monkeypatch.setattr(state, "invalidate_cache", lambda *a, **k: None)
    state.cancelable_orders()["0000012345"] = {
        "org_no": "ORG1", "shares": 10, "ord_dvsn": "00", "price": 71000.0,
    }

    r = client.post("/api/order/revise", json={"order_no": "0000012345", "new_price": 72000})
    assert r.status_code == 200, r.text
    assert r.json()["order_no"] == "0000099999"
    assert "0000012345" not in state.cancelable_orders()
    entry = state.cancelable_orders()["0000099999"]
    assert entry["price"] == 72000


async def test_revise_rejects_market_order_with_400(client, monkeypatch):
    """A market order (ord_dvsn != "00") can't be repriced — KIS silently
    ignores ORD_UNPR and returns success, so the router must refuse it up
    front rather than let the UI report a no-op revise as done."""
    class Broker:
        async def revise_order(self, *a, **k):
            raise AssertionError("broker should not be called for a market-order revise")

    state.kis_broker = broker_stub(Broker())
    state.cancelable_orders()["0000012345"] = {
        "org_no": "ORG1", "shares": 10, "ord_dvsn": "01", "price": 0.0,
    }

    r = client.post("/api/order/revise", json={"order_no": "0000012345", "new_price": 72000})
    assert r.status_code == 400
    assert "market order" in r.text.lower()


async def test_revise_leaves_registry_intact_when_broker_raises(client, monkeypatch):
    class Broker:
        async def revise_order(self, order_no, org_no, shares, new_price, ord_dvsn="00"):
            raise RuntimeError("KIS timeout")

    state.kis_broker = broker_stub(Broker())
    state.cancelable_orders()["0000012345"] = {
        "org_no": "ORG1", "shares": 10, "ord_dvsn": "00", "price": 71000.0,
    }

    r = client.post("/api/order/revise", json={"order_no": "0000012345", "new_price": 72000})

    assert r.status_code == 502
    entry = state.cancelable_orders()["0000012345"]
    assert entry["price"] == 71000.0


# ---------------------------------------------------------------------------
# 2026-09-04 real-incident fix, secondary part: a revise gets a NEW KIS order
# number, but the `trades` row written at original submission (which, after
# the primary fix above, now happens for limit orders too) kept the STALE
# kis_order_no — fill_monitor would then search KIS for an order number KIS
# no longer reports, a guaranteed false-positive unconfirmed_fill on every
# revise. `_record_kis_order_no` (app.execution.ledger) is wired in via the
# `trade_id` the /api/order ledger write now stashes on the cancel-handle
# registry entry.
# ---------------------------------------------------------------------------
async def test_revise_updates_trade_kis_order_no(client, monkeypatch):
    class Broker:
        async def revise_order(self, order_no, org_no, shares, new_price, ord_dvsn="00"):
            class R:
                order_no = "0000099999"
                message = "정정 접수"

            return R()

    state.kis_broker = broker_stub(Broker())
    monkeypatch.setattr(state, "invalidate_cache", lambda *a, **k: None)
    state.cancelable_orders()["0000012345"] = {
        "org_no": "ORG1", "shares": 10, "ord_dvsn": "00", "price": 71000.0, "trade_id": 42,
    }
    calls = []

    async def fake_record_kis_order_no(trade_id, order_no):
        calls.append((trade_id, order_no))

    monkeypatch.setattr(orders, "_record_kis_order_no", fake_record_kis_order_no)

    r = client.post("/api/order/revise", json={"order_no": "0000012345", "new_price": 72000})

    assert r.status_code == 200, r.text
    assert r.json()["order_no"] == "0000099999"
    assert calls == [(42, "0000099999")]


async def test_revise_ledger_update_failure_does_not_break_response(client, monkeypatch):
    """The broker-side revise already succeeded regardless of whether the
    ledger's kis_order_no bookkeeping does — a failure there must never turn
    a successful revise into a 500 for the caller."""
    class Broker:
        async def revise_order(self, order_no, org_no, shares, new_price, ord_dvsn="00"):
            class R:
                order_no = "0000099999"
                message = "정정 접수"

            return R()

    state.kis_broker = broker_stub(Broker())
    monkeypatch.setattr(state, "invalidate_cache", lambda *a, **k: None)
    state.cancelable_orders()["0000012345"] = {
        "org_no": "ORG1", "shares": 10, "ord_dvsn": "00", "price": 71000.0, "trade_id": 42,
    }

    async def boom(trade_id, order_no):
        raise RuntimeError("DB timeout")

    monkeypatch.setattr(orders, "_record_kis_order_no", boom)

    r = client.post("/api/order/revise", json={"order_no": "0000012345", "new_price": 72000})

    assert r.status_code == 200, r.text
    assert r.json()["status"] == "revised"
    assert r.json()["order_no"] == "0000099999"
