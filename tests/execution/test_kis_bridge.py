"""kis_bridge gate tests — every safety gate, in order, plus the live path."""

from __future__ import annotations

import asyncio
from datetime import date, timedelta

import pytest

from app.brokers.kis import BrokerBalance, BrokerPosition, BuyableInfo, OrderResult, OverseasCash, StockStatus
from app.clock import local_today
from app.domain.models import BidAsk, StockDesignations
from app.config import settings
from app.execution import kis_bridge


def _async_get_broker(broker):
    """kis_bridge.get_broker became a coroutine function in the multi-user
    migration (Phase 2) — it resolves the CURRENT USER's credentials from
    Postgres now, rather than reading one process-wide env credential set."""
    async def get_broker(name="kis", **kwargs):
        return broker

    return get_broker
from app.execution.kis_bridge import mirror_trade_to_kis, over_value_cap, place_manual_order, requires_approval

TODAY = date.today()
# mirror_trade_to_kis's trade-date gate now checks each market's OWN local
# date (local_today) rather than a shared KST date.today() — KST and US
# Eastern can disagree on the calendar date for hours at a time, so an AAPL
# test must use the overseas-local date, not the KST one, to look like
# "today" rather than a historical replay.
TODAY_OVERSEAS = local_today("overseas")

# Captured before the portfolio_caps_clear autouse fixture (below) neutralizes
# kis_bridge.portfolio_caps_block for every test in this module — the direct
# unit tests of portfolio_caps_block's own logic call this real reference
# instead of the (module-attribute) name the autouse fixture stubs out.
_real_portfolio_caps_block = kis_bridge.portfolio_caps_block


@pytest.fixture(autouse=True)
def market_open(monkeypatch):
    """Neutralize the domestic circuit-breaker gate for every test in this
    module — these gates predate that merged safety gate and assume a normal,
    open market (CB not active). Tests that exercise the CB gate itself
    re-patch this to True/None in their own body."""
    monkeypatch.setattr(
        kis_bridge.kis_market_status, "is_circuit_breaker_active", lambda code: False
    )


@pytest.fixture(autouse=True)
def safety_switch_clear(monkeypatch):
    """Neutralize the kill-switch/daily-loss gate for every test in this
    module — these predate that gate and assume it's never engaged. Tests
    that exercise it directly re-patch app.execution.kis_bridge.safety_switch
    in their own body."""
    async def _no_block(side, ticker):
        return None

    monkeypatch.setattr(kis_bridge.safety_switch, "safety_gate_block", _no_block)


@pytest.fixture(autouse=True)
def portfolio_caps_clear(monkeypatch):
    """Neutralize the manual-order portfolio-caps gate for every test in this
    module — it does real I/O (ledger position + live portfolio snapshot)
    that these pre-existing tests never set up. Tests that exercise it
    directly re-patch app.execution.kis_bridge.portfolio_caps_block."""
    async def _no_block(ticker, side, shares, price):
        return None

    monkeypatch.setattr(kis_bridge, "portfolio_caps_block", _no_block)


@pytest.fixture(autouse=True)
def _market_lock_noop(monkeypatch):
    """place_manual_order now wraps its gate-chain-through-submission in
    app.execution.portfolio_risk.market_lock (2026-08-06, cross-process
    safety) — a real Postgres advisory lock that these pre-existing gate
    tests never set up. Neutralized to a plain no-op for every test in this
    module, same convention as portfolio_caps_clear above; tests exercising
    real locking behavior live in test_portfolio_risk.py."""
    from contextlib import nullcontext

    monkeypatch.setattr(kis_bridge, "market_lock", lambda market: nullcontext())


@pytest.fixture(autouse=True)
def _fresh_order_lock(monkeypatch):
    """A fresh asyncio.Lock per test — module-level asyncio primitives bind
    to whatever event loop first acquires them, and pytest-asyncio gives
    each test function its own loop, so reusing the same kis_bridge._order_lock
    object across tests raises 'bound to a different event loop' on the
    second test that touches it."""
    monkeypatch.setattr(kis_bridge, "_order_lock", asyncio.Lock())


@pytest.fixture(autouse=True)
def _fake_pending_orders(monkeypatch):
    """A held (NEEDS_APPROVAL) auto order now persists to pending_orders —
    stub the DB-backed create call so these pre-existing gate tests don't
    require a real Postgres. Tests exercising the persistence itself use
    pending_orders' own dedicated test module instead. `count_pending`
    (the queue-length cap gate, 2026-08-06) is stubbed to 0 by default —
    the tests exercising that cap directly re-patch it in their own body."""
    from app.execution import pending_orders

    async def fake_create(ticker, trade_date, side, shares, price, reason, trade_id=None):
        return 1

    async def fake_count_pending():
        return 0

    monkeypatch.setattr(pending_orders, "create_pending_order", fake_create)
    monkeypatch.setattr(pending_orders, "count_pending", fake_count_pending)


class FakeBroker:
    def __init__(
        self, fail: bool = False, held: int = 1000,
        equity: float = 0.0, held_eval: float = 0.0,
        halted: bool = False, admin_issue: bool = False,
        vi_triggered: bool = False,
        mrkt_warn_cls_code: str = "",
        designations: StockDesignations | None = None,
        buyable_cash: float = 100_000_000.0,
        overseas_orderable: float = 100_000_000.0,
        overseas_held: float = 1000.0,
        bid: float = 100.0,
        ask: float = 101.0,
    ):
        self.orders: list[tuple] = []
        self.overseas_orders: list[tuple] = []
        self.fail = fail
        self.held = held  # domestic shares of 005930 the sell guard sees
        self.equity = equity  # 총평가금액 for the position-weight gate
        self.held_eval = held_eval  # 005930 eval already in the account
        self.halted = halted  # per-stock trading-halt gate
        self.admin_issue = admin_issue  # KRX administrative-issue flag
        self.vi_triggered = vi_triggered  # VI halt gate
        self.mrkt_warn_cls_code = mrkt_warn_cls_code  # KRX market-warning pre-screening gate
        self.designations = designations  # StockDesignations override for fetch_designations, or None
        self.buyable_cash = buyable_cash  # domestic cash-sufficiency gate
        self.overseas_orderable = overseas_orderable  # overseas cash-sufficiency gate
        self.overseas_held = overseas_held  # AAPL shares the overseas sell guard sees
        self.bid = bid  # spread_blocker's domestic/overseas quote
        self.ask = ask
        self.closed = False

    async def fetch_asking_price(self, code):
        return BidAsk(code=code, bid=self.bid, ask=self.ask, last=0.0)

    async def fetch_overseas_asking_price(self, symbol, exchange="NAS"):
        return BidAsk(code=symbol, bid=self.bid, ask=self.ask, last=0.0)

    async def fetch_stock_status(self, code):
        return StockStatus(
            code=code, trading_halted=self.halted,
            administrative_issue=self.admin_issue, nxt_trading_halted=False,
        )

    async def is_vi_triggered(self, code):
        return self.vi_triggered

    async def fetch_market_warning(self, code):
        return self.mrkt_warn_cls_code

    async def fetch_designations(self, code):
        if self.designations is not None:
            return self.designations
        return StockDesignations(code=code, market_warning=self.mrkt_warn_cls_code)

    async def fetch_buyable(self, code, price=None):
        return BuyableInfo(cash=self.buyable_cash, max_qty=1_000_000, max_amount=self.buyable_cash)

    async def fetch_overseas_present_balance(self, currency="USD"):
        return OverseasCash(
            currency=currency, deposit=self.overseas_orderable,
            orderable=self.overseas_orderable, eval_amount=0.0,
        )

    async def fetch_balance(self):
        return BrokerBalance(
            cash_deposit=0.0,
            total_eval_amount=self.equity,
            total_pnl_amount=0.0,
            purchase_amount=0.0,
            positions=[
                BrokerPosition(
                    code="005930", name="삼성전자", quantity=self.held,
                    avg_price=0.0, current_price=0.0, eval_amount=self.held_eval,
                    pnl_amount=0.0, pnl_pct=0.0,
                )
            ],
        )

    async def fetch_overseas_balance(self):
        return BrokerBalance(
            cash_deposit=0.0,
            total_eval_amount=0.0,
            total_pnl_amount=0.0,
            purchase_amount=0.0,
            positions=[
                BrokerPosition(
                    code="AAPL", name="Apple", quantity=self.overseas_held,
                    avg_price=0.0, current_price=0.0, eval_amount=0.0,
                    pnl_amount=0.0, pnl_pct=0.0,
                )
            ],
        )

    async def submit_order(self, code, side, quantity, price=None):
        if self.fail:
            raise RuntimeError("boom")
        self.orders.append((code, side, quantity, price))
        return OrderResult(
            order_no="0000117057", order_time="090512",
            message="모의투자 매수주문 완료", org_no="00950",
        )

    async def submit_overseas_order(self, code, side, quantity, price, exchange="NASD"):
        if self.fail:
            raise RuntimeError("boom")
        self.overseas_orders.append((code, side, quantity, price, exchange))
        return OrderResult(order_no="0030089601", order_time="223045", message="모의투자 매수주문 완료")

    async def aclose(self):
        self.closed = True


@pytest.fixture
def live_settings(monkeypatch):
    monkeypatch.setattr(settings, "kis_execution_enabled", True)
    monkeypatch.setattr(settings, "kis_execution_dry_run", False)
    monkeypatch.setattr(settings, "kis_max_order_krw", 1_000_000.0)
    monkeypatch.setattr(settings, "trade_allowlist_domestic", ["005930.KS"])
    monkeypatch.setattr(settings, "trade_allowlist_overseas", ["AAPL"])


async def test_unroutable_ticker_is_permanently_blocked(live_settings):
    # BLOCKED, not skipped — this call will never reach KIS regardless of
    # KIS_EXECUTION_ENABLED, so app.execution.ledger._mirror_never_reached_kis
    # must reverse whatever ledger entry the caller already recorded.
    # "skipped:" is reserved for the two cases where the ledger is
    # deliberately the sole source of truth (execution disabled, dry-run).
    result = await mirror_trade_to_kis("BTC-USD", TODAY, "buy", 1, 200.0)
    assert result.startswith("BLOCKED") and "neither" in result


async def test_non_today_trade_date_is_permanently_blocked(live_settings):
    # Same "skipped: -> BLOCKED:" fix as the unroutable-ticker case above.
    # Confirmed live 2026-08-06: an overseas stop-loss forced-exit mirrored
    # with the wrong (KST, not local-to-market) date used to fail this
    # check as "skipped:" and was never reversed — the ledger recorded the
    # position closed while KIS still held it. See app.execution.ledger.
    # check_and_apply_stop_loss's own local_today fix for the other half.
    stale_date = TODAY_OVERSEAS - timedelta(days=1)
    result = await mirror_trade_to_kis("AAPL", stale_date, "buy", 1, 200.0)
    assert result.startswith("BLOCKED") and "not today" in result


async def test_non_positive_shares_is_permanently_blocked(live_settings):
    result = await mirror_trade_to_kis("AAPL", TODAY_OVERSEAS, "buy", 0, 200.0)
    assert result.startswith("BLOCKED") and "non-positive" in result


# place_manual_order already validates its own exec_price > 0
# ("a positive reference price is required..."); mirror_trade_to_kis had
# no equivalent — with price==0 every value-based gate (over_value_cap,
# requires_approval, insufficient_cash_block, over_position_pct) computes
# a notional of 0 and passes trivially, letting the order through to
# submit at the real market price with every cap silently bypassed.
async def test_non_positive_price_is_permanently_blocked(live_settings):
    result = await mirror_trade_to_kis("AAPL", TODAY_OVERSEAS, "buy", 1, 0.0)
    assert result.startswith("BLOCKED") and "non-positive price" in result


async def test_us_ticker_routes_to_overseas_limit_order(live_settings):
    broker = FakeBroker()
    result = await mirror_trade_to_kis("AAPL", TODAY_OVERSEAS, "buy", 1, 315.5, broker=broker)
    assert result.startswith("submitted") and "NASDAQ limit" in result
    assert broker.overseas_orders == [("AAPL", "buy", 1, 315.5, "NASD")]
    assert broker.orders == []  # never touches the domestic order path


async def test_us_order_value_capped_in_usd(live_settings, monkeypatch):
    monkeypatch.setattr(settings, "kis_max_order_usd", 700.0)
    result = await mirror_trade_to_kis("AAPL", TODAY_OVERSEAS, "buy", 3, 315.5, broker=FakeBroker())  # $946.50
    assert result.startswith("BLOCKED") and "USD" in result


async def test_master_switch_off_by_default(monkeypatch):
    monkeypatch.setattr(settings, "kis_execution_enabled", False)
    result = await mirror_trade_to_kis("005930.KS", TODAY, "buy", 1, 70000)
    assert "disabled" in result



# ---------------- leveraged/inverse + unsupported quote-type filter (unsupported_product_block) ----------------
# Safety-gate audit (2026-07-27): is_leveraged_or_inverse itself is unit-tested
# in tests/domain/test_leverage_filter.py, but neither entry point had an
# integration test proving a leveraged/inverse-named ticker is actually
# blocked end-to-end — the gap this section closes.


async def test_leveraged_ticker_blocks_auto_order(live_settings, monkeypatch):
    from app.dataflows.yfinance.market_data import TickerDisplayInfo

    async def fake_display_info(ticker):
        return TickerDisplayInfo(symbol=ticker, name="KODEX 레버리지", previous_close=0.0, currency="KRW")

    monkeypatch.setattr("app.dataflows.kis_provider.get_ticker_display_info", fake_display_info)
    broker = FakeBroker()
    r = await mirror_trade_to_kis("005930.KS", TODAY, "buy", 1, 70000, broker=broker)
    assert r.startswith("BLOCKED") and "leveraged/inverse" in r
    assert broker.orders == []


async def test_leveraged_ticker_blocks_manual_order(live_settings, monkeypatch):
    # The manual dashboard ticket is the path the gate's own docstring calls
    # out as most at risk — a user can type any ticker with no upstream
    # (trade allowlist/screener) name filter having run first.
    from app.dataflows.yfinance.market_data import TickerDisplayInfo

    async def fake_display_info(ticker):
        return TickerDisplayInfo(symbol=ticker, name="ProShares UltraPro Short QQQ", previous_close=0.0, currency="USD")

    monkeypatch.setattr("app.dataflows.kis_provider.get_ticker_display_info", fake_display_info)
    broker = FakeBroker()
    r = await place_manual_order("SQQQ", "buy", 1, order_type="limit", limit_price=10.0, broker=broker)
    assert r.status == "blocked" and "leveraged/inverse" in r.message
    assert broker.overseas_orders == []


async def test_unsupported_quote_type_blocks_manual_order(live_settings, monkeypatch):
    # place_manual_order deliberately skips the trade_allowlist gate (state.py's
    # UnsupportedProductTypeError check), so a mutual fund entered by hand had
    # no check anywhere on this path until this gate started looking at
    # info.quote_type too (2026-08-27, P1-2 domain-folder audit).
    from app.dataflows.yfinance.market_data import TickerDisplayInfo

    async def fake_display_info(ticker):
        return TickerDisplayInfo(
            symbol=ticker, name="Some Mutual Fund", previous_close=0.0, currency="KRW", quote_type="MUTUALFUND"
        )

    monkeypatch.setattr("app.dataflows.kis_provider.get_ticker_display_info", fake_display_info)
    broker = FakeBroker()
    r = await place_manual_order("218150.KS", "buy", 1, order_type="limit", limit_price=10000.0, broker=broker)
    assert r.status == "blocked" and "MUTUALFUND" in r.message
    assert broker.orders == []


async def test_non_leveraged_ticker_not_blocked_by_name_filter(live_settings, monkeypatch):
    # Confirms the gate is discriminating, not accidentally blocking everything.
    from app.dataflows.yfinance.market_data import TickerDisplayInfo

    async def fake_display_info(ticker):
        return TickerDisplayInfo(symbol=ticker, name="삼성전자", previous_close=0.0, currency="KRW")

    monkeypatch.setattr("app.dataflows.kis_provider.get_ticker_display_info", fake_display_info)
    r = await mirror_trade_to_kis("005930.KS", TODAY, "buy", 1, 70000, broker=FakeBroker())
    assert r.startswith("submitted")


# ---------------- cash-like / bond fund filter (2026-09-01) ----------------
# 481050.KS "KODEX CD1년금리플러스액티브(합성)" reached the analysts; the
# screener universe now drops it, and this gate is the choke-point backstop.
# is_cash_or_bond_fund itself is unit-tested in tests/domain/test_product_filter.py.


def _bond_fund_display_info():
    from app.dataflows.yfinance.market_data import TickerDisplayInfo

    async def fake_display_info(ticker):
        return TickerDisplayInfo(
            symbol=ticker, name="KODEX CD1년금리플러스액티브(합성)", previous_close=0.0, currency="KRW"
        )

    return fake_display_info


async def test_cash_bond_fund_blocks_auto_buy(live_settings, monkeypatch):
    # Same trick as the leveraged-name test above — reuse an allowlisted
    # ticker (005930.KS) with a faked bond-fund name so the trade_allowlist
    # gate isn't what stops it.
    monkeypatch.setattr("app.dataflows.kis_provider.get_ticker_display_info", _bond_fund_display_info())
    broker = FakeBroker()
    r = await mirror_trade_to_kis("005930.KS", TODAY, "buy", 1, 70000, broker=broker)
    assert r.startswith("BLOCKED") and "cash-like/bond fund" in r
    assert broker.orders == []


async def test_cash_bond_fund_blocks_manual_buy(live_settings, monkeypatch):
    monkeypatch.setattr("app.dataflows.kis_provider.get_ticker_display_info", _bond_fund_display_info())
    broker = FakeBroker()
    r = await place_manual_order("481050.KS", "buy", 1, order_type="limit", limit_price=70000.0, broker=broker)
    assert r.status == "blocked" and "cash-like/bond fund" in r.message
    assert broker.orders == []


async def test_cash_bond_fund_does_not_block_sell(live_settings, monkeypatch):
    # The gate is buy/sell-common — a filter that also fired on SELL would
    # trap a position you already hold (stop-loss included). SELL must pass.
    monkeypatch.setattr("app.dataflows.kis_provider.get_ticker_display_info", _bond_fund_display_info())
    broker = FakeBroker()
    r = await mirror_trade_to_kis("005930.KS", TODAY, "sell", 1, 70000, broker=broker)
    assert r.startswith("submitted")
    assert broker.orders and broker.orders[0][1] == "sell"


async def test_cash_bond_fund_buy_allowed_when_flag_off(live_settings, monkeypatch):
    monkeypatch.setattr(settings, "kis_block_cash_bond_funds", False)
    monkeypatch.setattr("app.dataflows.kis_provider.get_ticker_display_info", _bond_fund_display_info())
    r = await mirror_trade_to_kis("005930.KS", TODAY, "buy", 1, 70000, broker=FakeBroker())
    assert r.startswith("submitted")


# ---- covered-call / option-income fund filter (2026-09-02, audit id=1170) --
# 498400.KS "KODEX 200타겟위클리커버드콜" — 100% equity, no bond sleeve, so it
# is NOT caught by is_cash_or_bond_fund; the derivative-income predicate is
# what closes the gap. Same buy-only, choke-point-backstop shape as above.


def _covered_call_display_info():
    from app.dataflows.yfinance.market_data import TickerDisplayInfo

    async def fake_display_info(ticker):
        return TickerDisplayInfo(
            symbol=ticker, name="KODEX 200타겟위클리커버드콜", previous_close=0.0, currency="KRW"
        )

    return fake_display_info


async def test_covered_call_fund_blocks_auto_buy_with_its_own_message(live_settings, monkeypatch):
    monkeypatch.setattr("app.dataflows.kis_provider.get_ticker_display_info", _covered_call_display_info())
    broker = FakeBroker()
    r = await mirror_trade_to_kis("005930.KS", TODAY, "buy", 1, 70000, broker=broker)
    assert r.startswith("BLOCKED") and "covered-call / option-income fund" in r
    assert "cash-like/bond fund" not in r
    assert broker.orders == []


async def test_covered_call_fund_does_not_block_sell(live_settings, monkeypatch):
    monkeypatch.setattr("app.dataflows.kis_provider.get_ticker_display_info", _covered_call_display_info())
    broker = FakeBroker()
    r = await mirror_trade_to_kis("005930.KS", TODAY, "sell", 1, 70000, broker=broker)
    assert r.startswith("submitted")
    assert broker.orders and broker.orders[0][1] == "sell"


async def test_covered_call_fund_buy_allowed_when_flag_off(live_settings, monkeypatch):
    monkeypatch.setattr(settings, "kis_block_cash_bond_funds", False)
    monkeypatch.setattr("app.dataflows.kis_provider.get_ticker_display_info", _covered_call_display_info())
    r = await mirror_trade_to_kis("005930.KS", TODAY, "buy", 1, 70000, broker=FakeBroker())
    assert r.startswith("submitted")


async def test_unresolved_name_blocks_auto_order(live_settings, monkeypatch):
    # NYSE-listed 3x ETFs (SOXL, UPRO, TNA, ...) have no entry in the
    # NASDAQ-only overseas master file, so get_ticker_display_info falls
    # back to the bare ticker string as the "name" — that used to read as
    # "definitely not leveraged" instead of "we have no idea".
    from app.dataflows.yfinance.market_data import TickerDisplayInfo

    async def fake_display_info(ticker):
        return TickerDisplayInfo(symbol=ticker, name=ticker, previous_close=0.0, currency="USD")

    monkeypatch.setattr("app.dataflows.kis_provider.get_ticker_display_info", fake_display_info)
    monkeypatch.setattr(settings, "trade_allowlist_overseas", ["SOXL"])
    r = await mirror_trade_to_kis("SOXL", TODAY_OVERSEAS, "buy", 1, 30.0, broker=FakeBroker())
    assert r.startswith("BLOCKED") and "no resolvable" in r


async def test_historical_replay_blocked(live_settings):
    result = await mirror_trade_to_kis("005930.KS", TODAY - timedelta(days=3), "buy", 1, 70000)
    assert "not today" in result


async def test_off_trade_allowlist_blocked(live_settings, monkeypatch):
    monkeypatch.setattr(settings, "trade_allowlist_domestic", [])
    monkeypatch.setattr(settings, "trade_allowlist_overseas", ["AAPL"])
    result = await mirror_trade_to_kis("005930.KS", TODAY, "buy", 1, 70000)
    assert result.startswith("BLOCKED") and "trade allowlist" in result


async def test_order_value_cap(live_settings):
    result = await mirror_trade_to_kis("005930.KS", TODAY, "buy", 100, 70000)  # 7,000,000 KRW
    assert result.startswith("BLOCKED") and "cap" in result


# over_value_cap is buy-only inside mirror_trade_to_kis (2026-08-06 fix):
# app.execution.ledger's own pre-shrink sizing chain only ever shrinks a
# BUY to fit this cap (reducing exposure was never what it exists to
# limit) — applying the hard block to sells too meant a large rebalance
# sell or stop-loss forced exit whose notional happened to exceed the cap
# was permanently BLOCKED with no partial-exit path. Same "sells always
# pass" exemption every other portfolio-wide risk gate in this codebase
# already grants. place_manual_order (a human's explicit order) keeps the
# unconditional check on both sides — see test_manual_value_cap_blocks and
# test_manual_sell_over_cap_blocked below.
async def test_order_value_cap_does_not_block_a_sell(live_settings):
    result = await mirror_trade_to_kis("005930.KS", TODAY, "sell", 100, 70000, broker=FakeBroker())  # 7,000,000 KRW
    assert result.startswith("submitted")


async def test_manual_sell_over_cap_still_blocked(live_settings):
    # The manual path is unaffected by the mirror_trade_to_kis-only
    # exemption above — a human's explicit oversized sell is still blocked
    # and reported, never silently let through.
    broker = FakeBroker(held=100)
    r = await place_manual_order("005930.KS", "sell", 100, reference_price=70000, broker=broker)  # 7,000,000 KRW
    assert r.status == "blocked" and "cap" in r.message


def test_over_value_cap_does_not_block_exactly_at_the_cap(monkeypatch):
    # Deliberate boundary: over_value_cap is strict `>`, so a value exactly
    # at the cap passes through to requires_approval instead of being
    # auto-blocked here (see over_value_cap's own docstring).
    monkeypatch.setattr(settings, "kis_max_order_krw", 5_000_000.0)
    assert over_value_cap("005930", 50, 100_000) is None  # exactly 5,000,000


def test_over_value_cap_blocks_strictly_above_the_cap(monkeypatch):
    monkeypatch.setattr(settings, "kis_max_order_krw", 5_000_000.0)
    result = over_value_cap("005930", 51, 100_000)  # 5,100,000 > cap
    assert result is not None and "cap" in result


def test_requires_approval_triggers_exactly_at_the_threshold(monkeypatch):
    # requires_approval is `>=`, so a value exactly at the cap (which
    # over_value_cap just let through) still needs approval — no gap
    # between the two gates.
    monkeypatch.setattr(settings, "kis_approval_required_enabled", True)
    monkeypatch.setattr(settings, "kis_approval_threshold_krw", 5_000_000.0)
    result = requires_approval("005930", 50, 100_000, approved=False)  # exactly 5,000,000
    assert result is not None and "NEEDS_APPROVAL" in result


# Real incident: a literal JSON `NaN` limit_price (both Python's json module
# and pydantic parse the literal token into float('nan') without complaint,
# even though standard JSON doesn't technically allow it) reached
# place_manual_order and sailed through over_value_cap/requires_approval
# unblocked, because any `>`/`>=` comparison against NaN is False in Python.
def test_over_value_cap_blocks_nan_price(monkeypatch):
    monkeypatch.setattr(settings, "kis_max_order_krw", 5_000_000.0)
    result = over_value_cap("005930", 100_000, float("nan"))
    assert result is not None and "BLOCKED" in result and "not a finite number" in result


def test_over_value_cap_still_blocks_a_legitimately_huge_value(monkeypatch):
    # Control case: a normal (non-NaN) absurdly large price was already
    # correctly blocked before the NaN fix — proves the fix didn't change
    # this pre-existing behavior.
    monkeypatch.setattr(settings, "kis_max_order_krw", 5_000_000.0)
    result = over_value_cap("005930", 100_000, 1e12)
    assert result is not None and "cap" in result


def test_requires_approval_blocks_nan_price(monkeypatch):
    monkeypatch.setattr(settings, "kis_approval_required_enabled", True)
    monkeypatch.setattr(settings, "kis_approval_threshold_krw", 5_000_000.0)
    result = requires_approval("005930", 100_000, float("nan"), approved=False)
    assert result is not None and "NEEDS_APPROVAL" in result and "not a finite number" in result


def test_requires_approval_still_triggers_for_a_legitimately_huge_value(monkeypatch):
    monkeypatch.setattr(settings, "kis_approval_required_enabled", True)
    monkeypatch.setattr(settings, "kis_approval_threshold_krw", 5_000_000.0)
    result = requires_approval("005930", 100_000, 1e12, approved=False)
    assert result is not None and "NEEDS_APPROVAL" in result


async def test_dry_run_stops_before_submission(live_settings, monkeypatch):
    monkeypatch.setattr(settings, "kis_execution_dry_run", True)
    broker = FakeBroker()
    result = await mirror_trade_to_kis("005930.KS", TODAY, "buy", 1, 70000, broker=broker)
    assert result.startswith("dry-run") and broker.orders == []


async def test_live_submission_strips_suffix_and_reports(live_settings):
    broker = FakeBroker()
    result = await mirror_trade_to_kis("005930.KS", TODAY, "buy", 1, 70000, broker=broker)
    assert result.startswith("submitted") and "0000117057" in result
    assert broker.orders == [("005930", "buy", 1, None)]  # market order, bare code
    assert not broker.closed  # injected broker is caller-owned


async def test_domestic_sell_blocked_when_exceeding_held(live_settings):
    broker = FakeBroker(held=1)  # only 1 share held
    result = await mirror_trade_to_kis("005930.KS", TODAY, "sell", 5, 70000, broker=broker)
    assert result.startswith("BLOCKED") and "held 1" in result
    assert broker.orders == []  # never reaches submission


async def test_domestic_sell_allowed_within_held(live_settings):
    broker = FakeBroker(held=10)
    result = await mirror_trade_to_kis("005930.KS", TODAY, "sell", 3, 70000, broker=broker)
    assert result.startswith("submitted")
    assert broker.orders == [("005930", "sell", 3, None)]


async def test_sell_ceiling_gate_reads_a_fresh_balance_even_inside_a_cached_scope(live_settings):
    # 2026-09-01: the lock-scoped balance cache must NOT reach the sell
    # ceiling — force_exit_position doesn't take market_lock, so a cached
    # held-quantity could let an over-sell through.
    from app.execution.portfolio_risk import cached_balance_fetches

    calls = {"n": 0}
    broker = FakeBroker(held=10)
    orig = broker.fetch_balance

    async def counting():
        calls["n"] += 1
        return await orig()

    broker.fetch_balance = counting
    async with cached_balance_fetches():
        await kis_bridge.domestic_sell_over_held(broker, "005930", "sell", 3)
        await kis_bridge.domestic_sell_over_held(broker, "005930", "sell", 3)
    assert calls["n"] == 2  # each gate call fetched fresh, no sharing


# ---------------- overseas sell-ceiling gate (overseas_sell_over_held) ----------------


async def test_overseas_sell_over_held_blocks_when_exceeding_held():
    broker = FakeBroker(overseas_held=1)
    result = await kis_bridge.overseas_sell_over_held(broker, "AAPL", "sell", 5)
    assert result is not None and "BLOCKED" in result and "held 1" in result


async def test_overseas_sell_over_held_allows_within_held():
    broker = FakeBroker(overseas_held=10)
    assert await kis_bridge.overseas_sell_over_held(broker, "AAPL", "sell", 3) is None


# ---------------- stale position assumption gate (assumed_position_mismatch_block) ----------------


async def test_assumed_position_mismatch_noop_when_not_supplied():
    # A caller that has no assumption to verify (place_manual_order) — no I/O,
    # no block.
    broker = FakeBroker(held=1)

    async def should_not_be_called():
        raise AssertionError("must not fetch a balance when assumed_current_shares is None")

    broker.fetch_balance = should_not_be_called
    assert await kis_bridge.assumed_position_mismatch_block(broker, "005930", None, None) is None


async def test_assumed_position_mismatch_blocks_domestic_when_stale():
    broker = FakeBroker(held=10)  # real account actually holds 10
    result = await kis_bridge.assumed_position_mismatch_block(broker, "005930", None, 0)  # sizing assumed 0
    assert result is not None and "BLOCKED" in result and "assumed 0" in result and "shows 10" in result


async def test_assumed_position_mismatch_allows_domestic_when_fresh():
    broker = FakeBroker(held=10)
    assert await kis_bridge.assumed_position_mismatch_block(broker, "005930", None, 10) is None


async def test_assumed_position_mismatch_blocks_overseas_when_stale():
    broker = FakeBroker(overseas_held=1)  # real account actually holds 1
    result = await kis_bridge.assumed_position_mismatch_block(broker, None, "AAPL", 255)  # sizing assumed 255
    assert result is not None and "BLOCKED" in result and "assumed 255" in result and "shows 1" in result


async def test_assumed_position_mismatch_allows_overseas_fractional_within_tolerance():
    broker = FakeBroker(overseas_held=10.0000001)  # floating-point noise, not a real mismatch
    assert await kis_bridge.assumed_position_mismatch_block(broker, None, "AAPL", 10.0) is None


async def test_mirror_blocks_before_other_gates_on_stale_assumption(live_settings):
    # A wrong BUY sized off a stale "0 held" assumption when 10 are actually
    # held (2026-07-24 incident class) — no sell-ceiling would ever catch
    # this (it's a buy), so this is the gate that has to.
    broker = FakeBroker(held=10)
    result = await mirror_trade_to_kis(
        "005930.KS", TODAY, "buy", 5, 70000, broker=broker, assumed_current_shares=0
    )
    assert result.startswith("BLOCKED") and "stale position assumption" in result
    assert broker.orders == []  # never reaches submission


async def test_mirror_proceeds_when_assumption_matches_fresh_balance(live_settings):
    broker = FakeBroker(held=10)
    result = await mirror_trade_to_kis(
        "005930.KS", TODAY, "buy", 5, 70000, broker=broker, assumed_current_shares=10
    )
    assert result.startswith("submitted")


async def test_overseas_sell_over_held_noop_for_buys():
    # Returns before ever calling fetch_overseas_balance — no I/O for a buy.
    async def should_not_be_called():
        raise AssertionError("must not fetch overseas balance for a buy")

    broker = FakeBroker()
    monkeypatch_broker = broker
    monkeypatch_broker.fetch_overseas_balance = should_not_be_called
    assert await kis_bridge.overseas_sell_over_held(broker, "AAPL", "buy", 5) is None


async def test_overseas_sell_over_held_zero_when_ticker_not_in_positions():
    broker = FakeBroker(overseas_held=10)  # held under code "AAPL"
    result = await kis_bridge.overseas_sell_over_held(broker, "TSLA", "sell", 1)
    assert result is not None and "held 0" in result


async def test_overseas_sell_blocked_when_exceeding_held(live_settings):
    # Regression test for the bug class this gate closes: a stop-loss
    # force-sell (app.execution.ledger.check_and_apply_stop_loss) sells
    # whatever the LEDGER's shares_after says, which can be inflated by an
    # unreversed rejected/expired pending order (see
    # app.execution.ledger.reverse_pending_trade) — before this gate, an
    # overseas sell had no real-balance ceiling at all and would submit for
    # more shares than actually held. Only 1 real AAPL share held; ledger
    # thinks 5 (the NAVER-incident pattern, overseas analogue).
    broker = FakeBroker(overseas_held=1)
    result = await mirror_trade_to_kis("AAPL", TODAY_OVERSEAS, "sell", 5, 315.5, broker=broker)
    assert result.startswith("BLOCKED") and "held 1" in result
    assert broker.overseas_orders == []  # never reaches submission


async def test_overseas_sell_allowed_within_held(live_settings):
    broker = FakeBroker(overseas_held=10)
    result = await mirror_trade_to_kis("AAPL", TODAY_OVERSEAS, "sell", 3, 315.5, broker=broker)
    assert result.startswith("submitted")
    assert broker.overseas_orders == [("AAPL", "sell", 3, 315.5, "NASD")]


async def test_manual_overseas_sell_blocked_when_exceeding_held(live_settings):
    broker = FakeBroker(overseas_held=1)
    r = await place_manual_order(
        "AAPL", "sell", 5, order_type="limit", limit_price=315.5, broker=broker
    )
    assert r.status == "blocked" and "held 1" in r.message
    assert broker.overseas_orders == []


async def test_submission_failure_never_raises(live_settings):
    result = await mirror_trade_to_kis("005930.KS", TODAY, "sell", 1, 70000, broker=FakeBroker(fail=True))
    assert result.startswith("FAILED") and "boom" in result


# ----------------- manual order (dashboard order ticket) -----------------


async def test_manual_order_master_switch_off(monkeypatch):
    monkeypatch.setattr(settings, "kis_execution_enabled", False)
    r = await place_manual_order("005930.KS", "buy", 1, reference_price=70000, broker=FakeBroker())
    assert r.status == "blocked" and "disabled" in r.message


async def test_manual_order_dry_run(live_settings, monkeypatch):
    monkeypatch.setattr(settings, "kis_execution_dry_run", True)
    broker = FakeBroker()
    r = await place_manual_order("005930.KS", "buy", 1, reference_price=70000, broker=broker)
    assert r.status == "dry-run" and broker.orders == []


async def test_dry_run_without_explicit_broker_still_validates_broker_backed_gates(live_settings, monkeypatch):
    # Previously: dry-run + no broker passed -> broker stayed None -> every
    # broker-backed gate (halt/VI/cash/sell-ceiling/position-pct) silently
    # skipped, even though the dry-run result claimed those gates were
    # validated. Now a broker is resolved via get_broker regardless of
    # dry-run, so a halted stock is still caught.
    monkeypatch.setattr(settings, "kis_execution_dry_run", True)
    broker = FakeBroker(halted=True)
    monkeypatch.setattr(kis_bridge, "get_broker", _async_get_broker(broker))
    r = await place_manual_order("005930.KS", "buy", 1, reference_price=70000)
    assert r.status == "blocked" and "halted" in r.message
    # get_broker() is the process-wide shared singleton (see kis_bridge's own
    # comment on this) — a caller resolving it this way must NEVER close it,
    # since other concurrent callers may still be using the same instance.
    # (This used to assert `broker.closed is True` — that was the live bug:
    # a blocked/no-op order call closed the shared broker for everyone.)
    assert broker.closed is False


async def test_dry_run_without_explicit_broker_reports_would_submit_when_clean(live_settings, monkeypatch):
    broker = FakeBroker()
    monkeypatch.setattr(settings, "kis_execution_dry_run", True)
    monkeypatch.setattr(kis_bridge, "get_broker", _async_get_broker(broker))
    r = await place_manual_order("005930.KS", "buy", 1, reference_price=70000)
    assert r.status == "dry-run" and "would submit" in r.message
    assert broker.orders == []
    assert broker.closed is False  # shared singleton — never closed by a mid-request caller


async def test_manual_domestic_market_buy_submits_and_carries_org_no(live_settings):
    broker = FakeBroker()
    r = await place_manual_order("005930.KS", "buy", 1, reference_price=70000, broker=broker)
    assert r.ok and r.order_no == "0000117057" and r.org_no == "00950"
    assert broker.orders == [("005930", "buy", 1, None)]  # market: price None


async def test_manual_domestic_limit_buy_passes_price(live_settings):
    broker = FakeBroker()
    r = await place_manual_order(
        "005930.KS", "buy", 1, order_type="limit", limit_price=68000, broker=broker
    )
    assert r.ok and broker.orders == [("005930", "buy", 1, 68000)]


async def test_manual_limit_requires_price(live_settings):
    r = await place_manual_order("005930.KS", "buy", 1, order_type="limit", broker=FakeBroker())
    assert r.status == "blocked" and "limit price" in r.message


async def test_manual_overseas_market_rejected(live_settings):
    r = await place_manual_order("AAPL", "buy", 1, order_type="market", reference_price=200, broker=FakeBroker())
    assert r.status == "blocked" and "limit-only" in r.message


async def test_manual_overseas_limit_submits(live_settings):
    broker = FakeBroker()
    r = await place_manual_order("AAPL", "buy", 1, order_type="limit", limit_price=315.5, broker=broker)
    assert r.ok and broker.overseas_orders == [("AAPL", "buy", 1, 315.5, "NASD")]


async def test_manual_value_cap_blocks(live_settings):
    r = await place_manual_order("005930.KS", "buy", 100, reference_price=70000, broker=FakeBroker())
    assert r.status == "blocked" and "cap" in r.message  # 7,000,000 > 1,000,000


async def test_normal_huge_market_order_is_blocked(live_settings):
    # Control: a normal (non-NaN) huge order was already correctly blocked
    # by the cap before the NaN fix below — this test's only role is
    # proving the fix didn't accidentally change normal-path blocking.
    r = await place_manual_order(
        "005930.KS", "buy", 1_000_000, order_type="market", reference_price=71000.0, broker=FakeBroker()
    )
    assert r.status == "blocked" and "cap" in r.message


async def test_manual_limit_order_with_nan_price_is_blocked_not_submitted(live_settings):
    # Real incident: limit_price=NaN passes place_manual_order's own
    # pre-check (`limit_price is None or limit_price <= 0` is False for
    # NaN), so it used to fall through and reach over_value_cap, which
    # (before the fix) silently let it through too — the order sailed
    # through the ENTIRE gate chain unblocked. broker.submit_order raising
    # is this test's own guarantee that a NaN order never reaches the broker.
    class NeverCalledBroker(FakeBroker):
        async def submit_order(self, *a, **k):
            raise AssertionError("NaN-priced order must never reach the broker")

    broker = NeverCalledBroker()
    r = await place_manual_order(
        "005930.KS", "buy", 1_000_000, order_type="limit", limit_price=float("nan"), broker=broker
    )
    assert r.status == "blocked"
    assert "not a finite number" in r.message
    assert broker.orders == []


async def test_manual_market_order_with_nan_reference_price_is_blocked_not_submitted(live_settings):
    # Same incident as above, via the market-order path: reference_price
    # (not limit_price) is exec_price for a market order, and price_arg is
    # None for submit_order — so, before the fix, a NaN reference_price
    # reached the broker's submit_order with no error surfaced anywhere.
    class NeverCalledBroker(FakeBroker):
        async def submit_order(self, *a, **k):
            raise AssertionError("NaN-priced order must never reach the broker")

    broker = NeverCalledBroker()
    r = await place_manual_order(
        "005930.KS", "buy", 1_000_000, order_type="market", reference_price=float("nan"), broker=broker
    )
    assert r.status == "blocked"
    assert "not a finite number" in r.message
    assert broker.orders == []


# market_lock now wraps place_manual_order's gate-chain-through-submission
# too (2026-08-06) — _order_lock alone is process-local (asyncio.Lock) and
# does nothing against a second process (the auto path, or another
# dashboard instance) racing the same account. Without this, a manual
# order and a concurrent auto decision for a different ticker in the same
# market could each see a pre-trade balance and jointly approve exposure
# neither alone would.
async def test_manual_order_acquires_market_lock_before_submitting(live_settings, monkeypatch):
    events = []

    class _TrackingLock:
        def __init__(self, market):
            self.market = market

        async def __aenter__(self):
            events.append(("lock_acquired", self.market))
            return self

        async def __aexit__(self, *exc):
            events.append("lock_released")
            return False

    monkeypatch.setattr(kis_bridge, "market_lock", lambda market: _TrackingLock(market))
    broker = FakeBroker()
    r = await place_manual_order("005930.KS", "buy", 1, reference_price=70000, broker=broker)

    assert r.ok
    assert events[0] == ("lock_acquired", "domestic")
    assert events[-1] == "lock_released"
    assert len(broker.orders) == 1  # submission happened while the lock was held


async def test_manual_sell_over_held_blocked(live_settings):
    broker = FakeBroker(held=1)
    # keep 5 x 150,000 = 750,000 under the 1,000,000 cap so the sell-ceiling
    # gate (not the value cap) is what blocks this.
    r = await place_manual_order("005930.KS", "sell", 5, reference_price=150000, broker=broker)
    assert r.status == "blocked" and "held 1" in r.message
    assert broker.orders == []


async def test_manual_order_failure_never_raises(live_settings):
    r = await place_manual_order("005930.KS", "buy", 1, reference_price=70000, broker=FakeBroker(fail=True))
    assert r.status == "failed" and "boom" in r.message


# ----------------- A-1 risk rules: approval threshold -----------------


@pytest.fixture
def approval_on(monkeypatch):
    monkeypatch.setattr(settings, "kis_approval_required_enabled", True)
    monkeypatch.setattr(settings, "kis_approval_threshold_krw", 500_000.0)
    monkeypatch.setattr(settings, "kis_approval_threshold_usd", 400.0)


async def test_approval_blocks_large_manual_order_without_approval(live_settings, approval_on):
    # 5 x 150,000 = 750,000 >= 500,000 threshold (and under the 1,000,000 cap)
    r = await place_manual_order("005930.KS", "buy", 5, reference_price=150000, broker=FakeBroker())
    assert r.status == "blocked" and "NEEDS_APPROVAL" in r.message


async def test_approval_lets_large_manual_order_through_when_approved(live_settings, approval_on):
    broker = FakeBroker()
    r = await place_manual_order(
        "005930.KS", "buy", 5, reference_price=150000, approved=True, broker=broker
    )
    assert r.ok and broker.orders == [("005930", "buy", 5, None)]


async def test_approval_ignores_small_order(live_settings, approval_on):
    broker = FakeBroker()  # 1 x 150,000 = 150,000 < 500,000 threshold
    r = await place_manual_order("005930.KS", "buy", 1, reference_price=150000, broker=broker)
    assert r.ok


async def test_approval_disabled_lets_large_order_through(live_settings):
    broker = FakeBroker()  # rule off by default
    r = await place_manual_order("005930.KS", "buy", 5, reference_price=150000, broker=broker)
    assert r.ok


async def test_approval_holds_large_auto_order(live_settings, approval_on):
    # auto path never has approval, so a large mirror order is held — and
    # now persisted to pending_orders instead of just being logged/dropped.
    r = await mirror_trade_to_kis("005930.KS", TODAY, "buy", 5, 150000, broker=FakeBroker())
    assert r.startswith("PENDING_USER_APPROVAL: NEEDS_APPROVAL")
    assert "pending_order_id=1" in r


async def test_approval_blocks_new_pending_order_when_queue_is_full(live_settings, approval_on, monkeypatch):
    """Queue-length cap (2026-08-06) — distinct from the TTL, which bounds
    how long a row lingers, not how many can pile up within one TTL window.
    A new proposal past settings.pending_order_max_count must be BLOCKED
    (same disposition family every other gate uses, so the ledger's
    speculative entry gets reversed like any other never-reached-KIS
    order) rather than silently queued on top of an already-full one."""
    from app.execution import pending_orders

    async def fake_count_pending():
        return settings.pending_order_max_count  # already at the cap

    create_calls = []

    async def fake_create(*args, **kwargs):
        create_calls.append(args)
        return 999  # must never be reached

    monkeypatch.setattr(pending_orders, "count_pending", fake_count_pending)
    monkeypatch.setattr(pending_orders, "create_pending_order", fake_create)

    r = await mirror_trade_to_kis("005930.KS", TODAY, "buy", 5, 150000, broker=FakeBroker())
    assert r.startswith("BLOCKED: pending approval queue full")
    assert create_calls == []


# ----------------- A-1 risk rules: position-weight cap -----------------


@pytest.fixture
def position_pct_on(monkeypatch):
    monkeypatch.setattr(settings, "kis_max_position_pct_enabled", True)
    monkeypatch.setattr(settings, "kis_max_position_pct", 20.0)


async def test_position_pct_blocks_buy_over_cap(live_settings, position_pct_on):
    # equity 1,000,000; already holding 100,000 of 005930; buying 150,000 more
    # → (100,000 + 150,000) / 1,000,000 = 25% > 20% cap
    broker = FakeBroker(equity=1_000_000.0, held_eval=100_000.0)
    r = await place_manual_order("005930.KS", "buy", 1, reference_price=150000, broker=broker)
    assert r.status == "blocked" and "% of account" in r.message
    assert broker.orders == []


async def test_position_pct_allows_buy_within_cap(live_settings, position_pct_on):
    # (0 + 150,000) / 1,000,000 = 15% < 20%
    broker = FakeBroker(equity=1_000_000.0, held_eval=0.0)
    r = await place_manual_order("005930.KS", "buy", 1, reference_price=150000, broker=broker)
    assert r.ok


async def test_position_pct_ignores_sells(live_settings, position_pct_on):
    # a sell never trips the weight cap even when weight is already high
    broker = FakeBroker(equity=1_000_000.0, held_eval=900_000.0, held=100)
    r = await place_manual_order("005930.KS", "sell", 1, reference_price=150000, broker=broker)
    assert r.ok


async def test_position_pct_disabled_allows_over_cap(live_settings):
    broker = FakeBroker(equity=1_000_000.0, held_eval=900_000.0)  # rule off
    r = await place_manual_order("005930.KS", "buy", 1, reference_price=150000, broker=broker)
    assert r.ok


async def test_position_pct_skipped_when_equity_unknown(live_settings, position_pct_on):
    broker = FakeBroker(equity=0.0, held_eval=0.0)  # can't compute → don't block
    r = await place_manual_order("005930.KS", "buy", 1, reference_price=150000, broker=broker)
    assert r.ok


async def test_position_pct_blocks_auto_buy_over_cap(live_settings, position_pct_on):
    broker = FakeBroker(equity=1_000_000.0, held_eval=100_000.0)
    r = await mirror_trade_to_kis("005930.KS", TODAY, "buy", 1, 150000, broker=broker)
    assert r.startswith("BLOCKED") and "% of account" in r
    assert broker.orders == []


async def test_position_pct_blocks_nan_price(position_pct_on):
    # Real incident: NaN propagates through `(held_eval + shares * price) /
    # equity * 100` and makes the `>` comparison silently False — the NaN
    # guard runs before any of that arithmetic, so it never needs a real
    # balance (this FakeBroker's fetch_balance is never even called).
    broker = FakeBroker(equity=1_000_000.0, held_eval=100_000.0)
    result = await kis_bridge.over_position_pct(broker, "005930", "buy", 1, float("nan"))
    assert result is not None and "BLOCKED" in result and "not a finite number" in result


# ---------------- circuit-breaker + per-stock halt gates (merged) ----------------


async def test_circuit_breaker_active_blocks_domestic(live_settings, monkeypatch):
    monkeypatch.setattr(kis_bridge.kis_market_status, "is_circuit_breaker_active", lambda code: True)
    broker = FakeBroker()
    r = await mirror_trade_to_kis("005930.KS", TODAY, "buy", 1, 70000, broker=broker)
    assert r.startswith("BLOCKED") and "circuit breaker" in r
    assert broker.orders == []


async def test_circuit_breaker_unknown_fails_closed(live_settings, monkeypatch):
    # Unknown state (feed not connected / mapping empty) blocks by default.
    monkeypatch.setattr(kis_bridge.kis_market_status, "is_circuit_breaker_active", lambda code: None)
    monkeypatch.setattr(settings, "kis_cb_fail_open_when_unknown", False)
    broker = FakeBroker()
    r = await mirror_trade_to_kis("005930.KS", TODAY, "buy", 1, 70000, broker=broker)
    assert r.startswith("BLOCKED") and "unknown" in r
    assert broker.orders == []


async def test_circuit_breaker_unknown_fail_open_allows(live_settings, monkeypatch):
    monkeypatch.setattr(kis_bridge.kis_market_status, "is_circuit_breaker_active", lambda code: None)
    monkeypatch.setattr(settings, "kis_cb_fail_open_when_unknown", True)
    broker = FakeBroker()
    r = await mirror_trade_to_kis("005930.KS", TODAY, "buy", 1, 70000, broker=broker)
    assert r.startswith("submitted")


# 2026-09-04: cb_unknown_reason distinguishes WHY the state is unknown so the
# block message doesn't read as strong evidence against a freshly-selected
# screening candidate whose realtime subscription simply hasn't been ACKed
# yet (see kis_market_status.cb_unknown_reason's own docstring — this mirrors
# the "execution-mechanics-isn't-evidence" pattern from f79213e/2916498,
# applied to the block message text itself).


async def test_circuit_breaker_unknown_code_not_subscribed_uses_softer_pending_wording(live_settings, monkeypatch):
    monkeypatch.setattr(kis_bridge.kis_market_status, "is_circuit_breaker_active", lambda code: None)
    monkeypatch.setattr(kis_bridge.kis_market_status, "cb_unknown_reason", lambda code: "code_not_subscribed")
    monkeypatch.setattr(settings, "kis_cb_fail_open_when_unknown", False)
    broker = FakeBroker()
    r = await mirror_trade_to_kis("005930.KS", TODAY, "buy", 1, 70000, broker=broker)
    assert r.startswith("BLOCKED")
    assert "pending" in r.lower()
    assert "mapping not yet populated" not in r
    assert broker.orders == []


async def test_circuit_breaker_unknown_feed_disconnected_keeps_strong_wording(live_settings, monkeypatch):
    monkeypatch.setattr(kis_bridge.kis_market_status, "is_circuit_breaker_active", lambda code: None)
    monkeypatch.setattr(kis_bridge.kis_market_status, "cb_unknown_reason", lambda code: "feed_disconnected")
    monkeypatch.setattr(settings, "kis_cb_fail_open_when_unknown", False)
    broker = FakeBroker()
    r = await mirror_trade_to_kis("005930.KS", TODAY, "buy", 1, 70000, broker=broker)
    assert r.startswith("BLOCKED")
    assert "unknown" in r
    assert "not connected" in r
    assert "mapping not yet populated" not in r
    assert broker.orders == []


async def test_circuit_breaker_unknown_registration_capped_wording(live_settings, monkeypatch):
    monkeypatch.setattr(kis_bridge.kis_market_status, "is_circuit_breaker_active", lambda code: None)
    monkeypatch.setattr(kis_bridge.kis_market_status, "cb_unknown_reason", lambda code: "registration_capped")
    monkeypatch.setattr(kis_bridge.kis_market_status, "registration_usage", lambda: (40, 40))
    monkeypatch.setattr(settings, "kis_cb_fail_open_when_unknown", False)
    broker = FakeBroker()
    r = await mirror_trade_to_kis("005930.KS", TODAY, "buy", 1, 70000, broker=broker)
    assert r.startswith("BLOCKED")
    assert "budget exhausted (40/40)" in r
    assert "never subscribed" in r
    assert "not a transient ACK delay" in r
    assert "pending" not in r.lower()
    assert broker.orders == []


async def test_circuit_breaker_unknown_registration_capped_without_feed_omits_usage(live_settings, monkeypatch):
    monkeypatch.setattr(kis_bridge.kis_market_status, "is_circuit_breaker_active", lambda code: None)
    monkeypatch.setattr(kis_bridge.kis_market_status, "cb_unknown_reason", lambda code: "registration_capped")
    monkeypatch.setattr(kis_bridge.kis_market_status, "registration_usage", lambda: None)
    monkeypatch.setattr(settings, "kis_cb_fail_open_when_unknown", False)
    r = await mirror_trade_to_kis("005930.KS", TODAY, "buy", 1, 70000, broker=FakeBroker())
    assert "budget exhausted, this code" in r


async def test_circuit_breaker_unknown_subscribe_rejected_wording(live_settings, monkeypatch):
    monkeypatch.setattr(kis_bridge.kis_market_status, "is_circuit_breaker_active", lambda code: None)
    monkeypatch.setattr(kis_bridge.kis_market_status, "cb_unknown_reason", lambda code: "subscribe_rejected")
    monkeypatch.setattr(settings, "kis_cb_fail_open_when_unknown", False)
    broker = FakeBroker()
    r = await mirror_trade_to_kis("005930.KS", TODAY, "buy", 1, 70000, broker=broker)
    assert r.startswith("BLOCKED")
    assert "rejected the real-time subscription" in r
    assert "pending" not in r.lower()
    assert broker.orders == []


async def test_circuit_breaker_unknown_feed_not_started_keeps_strong_wording(live_settings, monkeypatch):
    monkeypatch.setattr(kis_bridge.kis_market_status, "is_circuit_breaker_active", lambda code: None)
    monkeypatch.setattr(kis_bridge.kis_market_status, "cb_unknown_reason", lambda code: "feed_not_started")
    monkeypatch.setattr(settings, "kis_cb_fail_open_when_unknown", False)
    broker = FakeBroker()
    r = await mirror_trade_to_kis("005930.KS", TODAY, "buy", 1, 70000, broker=broker)
    assert r.startswith("BLOCKED")
    assert "unknown" in r
    assert "not started" in r
    assert "mapping not yet populated" not in r
    assert broker.orders == []


async def test_stock_halt_blocks_domestic(live_settings):
    broker = FakeBroker(halted=True)
    r = await mirror_trade_to_kis("005930.KS", TODAY, "buy", 1, 70000, broker=broker)
    assert r.startswith("BLOCKED") and "halted" in r
    assert broker.orders == []


async def test_stock_halt_check_failure_proceeds_without_blocking(live_settings, monkeypatch):
    # Best-effort fail-open (safety-gate audit, 2026-07-27): the halt/
    # admin-issue check itself erroring shouldn't block the order — KIS's
    # own order call still rejects a genuinely halted stock. Mirrors
    # test_vi_check_failure_proceeds_without_blocking's pattern for the
    # sibling gate.
    broker = FakeBroker()

    async def broken_status(code):
        raise RuntimeError("boom")

    monkeypatch.setattr(broker, "fetch_stock_status", broken_status)
    r = await mirror_trade_to_kis("005930.KS", TODAY, "buy", 1, 70000, broker=broker)
    assert r.startswith("submitted")


async def test_stock_halt_check_paper_unsupported_adds_note(live_settings, monkeypatch):
    # The EGW02006 branch (safety-gate audit, 2026-07-27): a paper account
    # can't run this check at all (KIS rejects the whole TR), so the gate
    # proceeds AND the final order message carries a visible note explaining
    # why the check was skipped — never silently proceeding with no trace.
    broker = FakeBroker()

    async def paper_unsupported(code):
        raise RuntimeError("EGW02006 모의투자 TR 이 아닙니다")

    monkeypatch.setattr(broker, "fetch_stock_status", paper_unsupported)
    r = await mirror_trade_to_kis("005930.KS", TODAY, "buy", 1, 70000, broker=broker)
    assert r.startswith("submitted")
    assert "모의투자 미지원" in r


async def test_circuit_breaker_ignores_overseas(live_settings, monkeypatch):
    # US path never touches the domestic CB gate even when it would block.
    monkeypatch.setattr(kis_bridge.kis_market_status, "is_circuit_breaker_active", lambda code: True)
    broker = FakeBroker()
    r = await mirror_trade_to_kis("AAPL", TODAY_OVERSEAS, "buy", 1, 315.5, broker=broker)
    assert r.startswith("submitted") and "NASDAQ" in r


# ---------------- kill switch / daily-loss breaker (safety_switch) ----------------


async def test_kill_switch_blocks_auto_buy(live_settings, monkeypatch):
    async def engaged(side, ticker):
        return "BLOCKED: kill switch engaged (manual emergency stop)"

    monkeypatch.setattr(kis_bridge.safety_switch, "safety_gate_block", engaged)
    broker = FakeBroker()
    r = await mirror_trade_to_kis("005930.KS", TODAY, "buy", 1, 70000, broker=broker)
    assert r.startswith("BLOCKED") and "kill switch" in r
    assert broker.orders == []


async def test_kill_switch_blocks_manual_sell_too(live_settings, monkeypatch):
    # A kill switch is a true emergency stop — unlike the portfolio-level
    # caps, it blocks sells as well as buys.
    async def engaged(side, ticker):
        return "BLOCKED: kill switch engaged (manual emergency stop)"

    monkeypatch.setattr(kis_bridge.safety_switch, "safety_gate_block", engaged)
    broker = FakeBroker(held=10)
    r = await place_manual_order("005930.KS", "sell", 1, reference_price=70000, broker=broker)
    assert r.status == "blocked" and "kill switch" in r.message
    assert broker.orders == []


async def test_safety_gate_receives_the_actual_order_side(live_settings, monkeypatch):
    seen = []

    async def record_side(side, ticker):
        seen.append(side)
        return None

    monkeypatch.setattr(kis_bridge.safety_switch, "safety_gate_block", record_side)
    await mirror_trade_to_kis("005930.KS", TODAY, "sell", 1, 70000, broker=FakeBroker(held=5))
    assert seen == ["sell"]


async def test_safety_gate_checked_before_value_cap(live_settings, monkeypatch):
    # If the safety gate blocks, callers should never even reach later gates.
    async def engaged(side, ticker):
        return "BLOCKED: kill switch engaged (test)"

    monkeypatch.setattr(kis_bridge.safety_switch, "safety_gate_block", engaged)
    # An order that would ALSO fail the value cap — the safety-gate message
    # must be what's returned, proving it runs first.
    r = await mirror_trade_to_kis("005930.KS", TODAY, "buy", 1000, 70000)
    assert "kill switch" in r
    assert "cap" not in r


# ---------------- VI (변동성완화장치) halt gate ----------------


async def test_vi_triggered_blocks_domestic_order(live_settings):
    broker = FakeBroker(vi_triggered=True)
    r = await mirror_trade_to_kis("005930.KS", TODAY, "buy", 1, 70000, broker=broker)
    assert r.startswith("BLOCKED") and "VI" in r
    assert broker.orders == []


async def test_vi_triggered_blocks_domestic_sell_too(live_settings):
    # VI pauses matching entirely — unlike portfolio-level caps, it blocks sells as well.
    broker = FakeBroker(vi_triggered=True, held=10)
    r = await place_manual_order("005930.KS", "sell", 1, reference_price=70000, broker=broker)
    assert r.status == "blocked" and "VI" in r.message
    assert broker.orders == []


async def test_vi_not_triggered_allows_order(live_settings):
    broker = FakeBroker(vi_triggered=False)
    r = await mirror_trade_to_kis("005930.KS", TODAY, "buy", 1, 70000, broker=broker)
    assert r.startswith("submitted")


async def test_vi_check_failure_proceeds_without_blocking(live_settings, monkeypatch):
    broker = FakeBroker()

    async def broken_vi(code):
        raise RuntimeError("boom")

    monkeypatch.setattr(broker, "is_vi_triggered", broken_vi)
    r = await mirror_trade_to_kis("005930.KS", TODAY, "buy", 1, 70000, broker=broker)
    assert r.startswith("submitted")


async def test_vi_ignores_overseas(live_settings):
    # US path never touches the domestic VI gate.
    broker = FakeBroker(vi_triggered=True)
    r = await mirror_trade_to_kis("AAPL", TODAY_OVERSEAS, "buy", 1, 315.5, broker=broker)
    assert r.startswith("submitted")


# ---------------- cash-sufficiency gate (insufficient_cash_block) ----------------


async def test_domestic_buy_blocked_when_cash_insufficient(live_settings):
    broker = FakeBroker(buyable_cash=50_000.0)  # 1 share @ 70,000 > 50,000 available
    r = await mirror_trade_to_kis("005930.KS", TODAY, "buy", 1, 70000, broker=broker)
    assert r.startswith("BLOCKED") and "cash" in r
    assert broker.orders == []


async def test_domestic_buy_allowed_within_available_cash(live_settings):
    broker = FakeBroker(buyable_cash=100_000.0)
    r = await mirror_trade_to_kis("005930.KS", TODAY, "buy", 1, 70000, broker=broker)
    assert r.startswith("submitted")


async def test_domestic_sell_never_checked_against_buyable_cash(live_settings):
    # Sells need held shares, not cash — insufficient_cash_block is a no-op for them.
    broker = FakeBroker(buyable_cash=0.0, held=10)
    r = await mirror_trade_to_kis("005930.KS", TODAY, "sell", 1, 70000, broker=broker)
    assert r.startswith("submitted")


async def test_overseas_buy_blocked_when_cash_insufficient(live_settings):
    broker = FakeBroker(overseas_orderable=100.0)  # 1 share @ 315.5 > 100 available
    r = await mirror_trade_to_kis("AAPL", TODAY_OVERSEAS, "buy", 1, 315.5, broker=broker)
    assert r.startswith("BLOCKED") and "cash" in r
    assert broker.overseas_orders == []


async def test_overseas_buy_allowed_within_available_cash(live_settings):
    broker = FakeBroker(overseas_orderable=1000.0)
    r = await mirror_trade_to_kis("AAPL", TODAY_OVERSEAS, "buy", 1, 315.5, broker=broker)
    assert r.startswith("submitted")


async def test_overseas_buy_blocked_when_currency_unmatched(live_settings, monkeypatch):
    # cash.matched=False means the present-balance response never actually
    # confirmed a $0 balance — must still block (fail-safe), just not word
    # it as if the balance is a known 0.
    broker = FakeBroker(overseas_orderable=1000.0)

    async def unmatched_balance(currency="USD"):
        return OverseasCash(currency=currency, deposit=0.0, orderable=0.0, eval_amount=0.0, matched=False)

    monkeypatch.setattr(broker, "fetch_overseas_present_balance", unmatched_balance)
    r = await mirror_trade_to_kis("AAPL", TODAY_OVERSEAS, "buy", 1, 315.5, broker=broker)
    assert r.startswith("BLOCKED") and "inconclusive" in r
    assert broker.overseas_orders == []


async def test_manual_domestic_buy_blocked_when_cash_insufficient(live_settings):
    broker = FakeBroker(buyable_cash=50_000.0)
    r = await place_manual_order("005930.KS", "buy", 1, reference_price=70000, broker=broker)
    assert r.status == "blocked" and "cash" in r.message


async def test_cash_check_failure_proceeds_without_blocking(live_settings, monkeypatch):
    broker = FakeBroker()

    async def broken_buyable(code, price=None):
        raise RuntimeError("boom")

    monkeypatch.setattr(broker, "fetch_buyable", broken_buyable)
    r = await mirror_trade_to_kis("005930.KS", TODAY, "buy", 1, 70000, broker=broker)
    assert r.startswith("submitted")


async def test_insufficient_cash_block_blocks_nan_price():
    # Real incident: NaN made `order_value > buyable.cash` silently False,
    # so this gate never blocked a NaN-priced buy regardless of cash on hand.
    broker = FakeBroker(buyable_cash=1_000_000.0)
    result = await kis_bridge.insufficient_cash_block(broker, "005930", None, "buy", 1_000_000, float("nan"))
    assert result is not None and "BLOCKED" in result and "not a finite number" in result


async def test_insufficient_cash_block_still_blocks_a_normal_insufficient_price():
    # Control: a normal price that genuinely exceeds available cash was
    # already correctly blocked before the NaN fix.
    broker = FakeBroker(buyable_cash=50_000.0)
    result = await kis_bridge.insufficient_cash_block(broker, "005930", None, "buy", 1, 70_000.0)
    assert result is not None and "cash" in result


# ---------------- execution_blockers / available_cash_native (read-only context helpers) ----------------
# 2026-07-28 AI decision-context gap: these reuse the same gates above (no new
# KIS calls, no enforcement) so the Portfolio Manager's narrative can't drift
# from what actually blocks a real order.


async def test_execution_blockers_empty_when_nothing_blocks(live_settings):
    broker = FakeBroker()
    result = await kis_bridge.execution_blockers(broker, "005930", "005930.KS")
    assert result == []


async def test_execution_blockers_reports_circuit_breaker(live_settings, monkeypatch):
    monkeypatch.setattr(kis_bridge.kis_market_status, "is_circuit_breaker_active", lambda code: True)
    broker = FakeBroker()
    result = await kis_bridge.execution_blockers(broker, "005930", "005930.KS")
    assert any("circuit breaker" in b for b in result)


async def test_execution_blockers_reports_halt(live_settings):
    broker = FakeBroker(halted=True)
    result = await kis_bridge.execution_blockers(broker, "005930", "005930.KS")
    assert any("trading-halted" in b for b in result)


async def test_execution_blockers_reports_vi(live_settings):
    broker = FakeBroker(vi_triggered=True)
    result = await kis_bridge.execution_blockers(broker, "005930", "005930.KS")
    assert any("VI" in b for b in result)


async def test_execution_blockers_can_report_multiple_at_once(live_settings, monkeypatch):
    monkeypatch.setattr(kis_bridge.kis_market_status, "is_circuit_breaker_active", lambda code: True)
    broker = FakeBroker(halted=True, vi_triggered=True)
    result = await kis_bridge.execution_blockers(broker, "005930", "005930.KS")
    assert len(result) == 3


# 2026-09-07 (id=1238 halt-status narrative fix): `checked_out` tells a caller
# WHICH of the three gates actually completed a real check this call — not
# merely "returned no block" — so pipeline._execution_status_context can tell
# a genuinely confirmed-clear read apart from one where a gate silently
# didn't run (see execution_blockers' own docstring).


async def test_execution_blockers_reports_all_three_checks_when_clean(live_settings):
    broker = FakeBroker()
    checked: list[str] = []
    result = await kis_bridge.execution_blockers(broker, "005930", "005930.KS", checked_out=checked)
    assert result == []
    assert set(checked) == {"circuit_breaker", "stock_halt", "vi"}


async def test_execution_blockers_omits_check_that_raised(live_settings, monkeypatch):
    broker = FakeBroker()

    async def broken_status(code):
        raise RuntimeError("boom")

    monkeypatch.setattr(broker, "fetch_stock_status", broken_status)
    checked: list[str] = []
    result = await kis_bridge.execution_blockers(broker, "005930", "005930.KS", checked_out=checked)
    assert result == []  # best-effort: the raising check still proceeds without blocking
    assert "stock_halt" not in checked
    assert set(checked) == {"circuit_breaker", "vi"}


# ---------------- pre_screening_blockers (circuit-breaker deliberately excluded) ----------------
# 2026-07-29: unlike execution_blockers, must NOT fail-closed on an unsubscribed
# ticker's "unknown" circuit-breaker state — a fresh momentum candidate isn't on
# the real-time feed yet (add_code only fires from add_ticker_to_trade_allowlist).


async def test_pre_screening_blockers_empty_when_nothing_blocks(live_settings):
    broker = FakeBroker()
    result = await kis_bridge.pre_screening_blockers(broker, "005930", "005930.KS")
    assert result == []


async def test_pre_screening_blockers_ignores_circuit_breaker_unknown_state(live_settings, monkeypatch):
    """The exact scenario this function exists for: circuit-breaker state
    unknown (unsubscribed candidate) must not block, unlike execution_blockers."""
    monkeypatch.setattr(kis_bridge.kis_market_status, "is_circuit_breaker_active", lambda code: None)
    broker = FakeBroker()
    result = await kis_bridge.pre_screening_blockers(broker, "005930", "005930.KS")
    assert result == []


async def test_pre_screening_blockers_ignores_circuit_breaker_even_when_active(live_settings, monkeypatch):
    monkeypatch.setattr(kis_bridge.kis_market_status, "is_circuit_breaker_active", lambda code: True)
    broker = FakeBroker()
    result = await kis_bridge.pre_screening_blockers(broker, "005930", "005930.KS")
    assert result == []


async def test_pre_screening_blockers_reports_halt(live_settings):
    broker = FakeBroker(halted=True)
    result = await kis_bridge.pre_screening_blockers(broker, "005930", "005930.KS")
    assert any("trading-halted" in b for b in result)


async def test_pre_screening_blockers_reports_vi(live_settings):
    broker = FakeBroker(vi_triggered=True)
    result = await kis_bridge.pre_screening_blockers(broker, "005930", "005930.KS")
    assert any("VI" in b for b in result)


# ---------------- pre_screening_blockers is halt/VI only again (§4-5, 2026-09-08) ----


async def test_pre_screening_blockers_no_longer_checks_designations():
    """market-warning (and the new short-overheat/liquidation flags) moved
    OUT of pre_screening_blockers on 2026-09-08 (§4-5) into a standalone
    `designation_block`, so the forced held-position review path
    (app.live.auto_trade_loop._held_needing_review/_cash_freed_up_for_review)
    can be exempted from it — see designation_block's own docstring. A
    warned/overheated/liquidating code must NOT be reported here any more."""
    broker = FakeBroker(designations=StockDesignations(
        code="082850", market_warning="2", short_overheat=True, liquidation=True,
    ))
    result = await kis_bridge.pre_screening_blockers(broker, "082850", "082850.KS")
    assert result == []


# ---------------- designation_block (2026-09-08 audit: ids 1312-1315; §4-5 extension) --


async def test_designation_block_reports_market_warning(live_settings):
    broker = FakeBroker(designations=StockDesignations(code="082850", market_warning="2"))
    result = await kis_bridge.designation_block(broker, "082850", "082850.KS")
    assert result is not None and "시장경고" in result and "'2'" in result


async def test_designation_block_ignores_zero_market_warning_code():
    """"0"/"00" are the KRX "no designation" sentinel, not a warning — see
    fetch_market_warning's own docstring for the corroborating evidence."""
    broker = FakeBroker(designations=StockDesignations(code="005930", market_warning="0"))
    assert await kis_bridge.designation_block(broker, "005930", "005930.KS") is None
    broker = FakeBroker(designations=StockDesignations(code="005930", market_warning="00"))
    assert await kis_bridge.designation_block(broker, "005930", "005930.KS") is None


async def test_designation_block_reports_short_overheat(live_settings):
    broker = FakeBroker(designations=StockDesignations(code="005930", short_overheat=True))
    result = await kis_bridge.designation_block(broker, "005930", "005930.KS")
    assert result is not None and "단기과열" in result


async def test_designation_block_reports_liquidation(live_settings):
    broker = FakeBroker(designations=StockDesignations(code="005930", liquidation=True))
    result = await kis_bridge.designation_block(broker, "005930", "005930.KS")
    assert result is not None and "정리매매" in result


async def test_designation_block_lists_every_reason(live_settings):
    broker = FakeBroker(designations=StockDesignations(
        code="005930", market_warning="3", short_overheat=True, liquidation=True,
    ))
    result = await kis_bridge.designation_block(broker, "005930", "005930.KS")
    assert result is not None
    assert "시장경고" in result and "단기과열" in result and "정리매매" in result


async def test_designation_block_never_uses_administrative_raw_to_drop():
    """mang_issu_cls_code's enum direction is unconfirmed (see
    StockDesignations' own docstring) — a non-empty value must be logged,
    never used to drop a candidate, until a live hit confirms the sentinel."""
    broker = FakeBroker(designations=StockDesignations(code="005930", administrative_raw="1"))
    result = await kis_bridge.designation_block(broker, "005930", "005930.KS")
    assert result is None


async def test_designation_block_ignores_clean_designations():
    broker = FakeBroker(designations=StockDesignations(code="005930"))
    assert await kis_bridge.designation_block(broker, "005930", "005930.KS") is None


async def test_designation_block_proceeds_on_error(monkeypatch):
    """Best-effort: a raising check proceeds without blocking, same
    convention as _stock_halt_block/_vi_block."""
    broker = FakeBroker()

    async def broken(code):
        raise RuntimeError("boom")

    monkeypatch.setattr(broker, "fetch_designations", broken)
    result = await kis_bridge.designation_block(broker, "005930", "005930.KS")
    assert result is None


async def test_designation_block_records_checked_out():
    broker = FakeBroker(designations=StockDesignations(code="005930", market_warning="1"))
    checked: list[str] = []
    await kis_bridge.designation_block(broker, "005930", "005930.KS", checked_out=checked)
    assert checked == ["designation"]


async def test_designation_block_no_checked_out_on_error(monkeypatch):
    broker = FakeBroker()

    async def broken(code):
        raise RuntimeError("boom")

    monkeypatch.setattr(broker, "fetch_designations", broken)
    checked: list[str] = []
    await kis_bridge.designation_block(broker, "005930", "005930.KS", checked_out=checked)
    assert checked == []


async def test_designation_not_added_to_execution_blockers(live_settings):
    """Cost-saving pre-filter only — a designation must NOT block an actual
    order (unlike halt/VI/circuit-breaker)."""
    broker = FakeBroker(designations=StockDesignations(
        code="005930", market_warning="3", short_overheat=True, liquidation=True,
    ))
    checked: list[str] = []
    result = await kis_bridge.execution_blockers(broker, "005930", "005930.KS", checked_out=checked)
    assert result == []
    assert "designation" not in checked
    assert "market_warning" not in checked


# ---------------- spread_pct / spread_blocker (liquidity pre-filter) ----------------
# screening-method-improvement-ideas item #3.


def test_spread_pct_computes_fraction_of_mid():
    assert kis_bridge.spread_pct(bid=99.0, ask=101.0) == pytest.approx(0.02)


def test_spread_pct_none_when_ask_not_above_bid():
    assert kis_bridge.spread_pct(bid=100.0, ask=100.0) is None
    assert kis_bridge.spread_pct(bid=101.0, ask=100.0) is None  # crossed/locked


def test_spread_pct_none_when_bid_or_ask_non_positive():
    assert kis_bridge.spread_pct(bid=0.0, ask=101.0) is None
    assert kis_bridge.spread_pct(bid=99.0, ask=0.0) is None


async def test_spread_blocker_none_when_within_limit(live_settings, monkeypatch):
    monkeypatch.setattr(settings, "max_spread_pct", 0.05)
    broker = FakeBroker(bid=99.0, ask=101.0)  # ~2% spread
    result = await kis_bridge.spread_blocker(broker, "005930.KS")
    assert result is None


async def test_spread_blocker_reports_wide_spread_domestic(live_settings, monkeypatch):
    monkeypatch.setattr(settings, "max_spread_pct", 0.01)
    broker = FakeBroker(bid=95.0, ask=105.0)  # ~10% spread
    result = await kis_bridge.spread_blocker(broker, "005930.KS")
    assert result is not None and "spread too wide" in result


async def test_spread_blocker_reports_wide_spread_overseas(live_settings, monkeypatch):
    monkeypatch.setattr(settings, "max_spread_pct", 0.01)
    broker = FakeBroker(bid=95.0, ask=105.0)
    result = await kis_bridge.spread_blocker(broker, "AAPL")
    assert result is not None and "AAPL" in result


async def test_spread_blocker_none_when_quote_unusable(live_settings, monkeypatch):
    monkeypatch.setattr(settings, "max_spread_pct", 0.01)
    broker = FakeBroker(bid=0.0, ask=0.0)  # market closed, zeroed-out quote
    result = await kis_bridge.spread_blocker(broker, "005930.KS")
    assert result is None


async def test_spread_blocker_proceeds_when_quote_fetch_fails(live_settings, monkeypatch):
    async def raising_asking_price(code):
        raise RuntimeError("KIS call failed")

    broker = FakeBroker()
    monkeypatch.setattr(broker, "fetch_asking_price", raising_asking_price)
    result = await kis_bridge.spread_blocker(broker, "005930.KS")
    assert result is None


async def test_available_cash_native_domestic(live_settings):
    broker = FakeBroker(buyable_cash=123_456.0)
    result = await kis_bridge.available_cash_native(broker, "005930", None, 70000)
    assert result == (123_456.0, "KRW")


async def test_available_cash_native_overseas_uses_orderable(live_settings):
    broker = FakeBroker(overseas_orderable=789.0)
    result = await kis_bridge.available_cash_native(broker, None, "AAPL", 315.5)
    assert result == (789.0, "USD")


async def test_available_cash_native_overseas_falls_back_to_deposit_when_orderable_zero(live_settings, monkeypatch):
    # Same confirmed-2026-07-20 quirk insufficient_cash_block already floors:
    # orderable reads 0 for every currency on this account regardless of the
    # real balance, so deposit is the trustworthy floor.
    broker = FakeBroker()

    async def zero_orderable(currency="USD"):
        return OverseasCash(currency=currency, deposit=555.0, orderable=0.0, eval_amount=0.0)

    monkeypatch.setattr(broker, "fetch_overseas_present_balance", zero_orderable)
    result = await kis_bridge.available_cash_native(broker, None, "AAPL", 315.5)
    assert result == (555.0, "USD")


async def test_available_cash_native_overseas_none_when_currency_unmatched(live_settings, monkeypatch):
    broker = FakeBroker()

    async def unmatched_balance(currency="USD"):
        return OverseasCash(currency=currency, deposit=0.0, orderable=0.0, eval_amount=0.0, matched=False)

    monkeypatch.setattr(broker, "fetch_overseas_present_balance", unmatched_balance)
    result = await kis_bridge.available_cash_native(broker, None, "AAPL", 315.5)
    assert result is None


async def test_available_cash_native_none_on_lookup_failure(live_settings, monkeypatch):
    broker = FakeBroker()

    async def broken_buyable(code, price=None):
        raise RuntimeError("boom")

    monkeypatch.setattr(broker, "fetch_buyable", broken_buyable)
    result = await kis_bridge.available_cash_native(broker, "005930", None, 70000)
    assert result is None


# ---------------- portfolio-level caps gate (manual orders only) ----------------


async def test_portfolio_caps_block_noop_for_sells():
    # Returns before ever importing ledger/portfolio_risk — no I/O for a sell.
    assert await _real_portfolio_caps_block("AAPL", "sell", 5, 100.0) is None


async def test_portfolio_caps_block_none_when_caps_not_exceeded(monkeypatch):
    import app.execution.ledger as ledger_mod
    import app.execution.portfolio_risk as portfolio_risk_mod

    async def fake_apply_live_caps(ticker, target_shares, price, position_shares_before):
        return target_shares, None  # not capped

    monkeypatch.setattr(ledger_mod, "get_position", _fake_get_position_factory(0))
    monkeypatch.setattr(portfolio_risk_mod, "apply_live_portfolio_caps", fake_apply_live_caps)
    result = await _real_portfolio_caps_block("AAPL", "buy", 5, 100.0)
    assert result is None


async def test_portfolio_caps_block_blocks_when_capped(monkeypatch):
    import app.execution.ledger as ledger_mod
    import app.execution.portfolio_risk as portfolio_risk_mod

    async def fake_apply_live_caps(ticker, target_shares, price, position_shares_before):
        return position_shares_before + 2, "Portfolio concentration cap: AAPL would reach 40%."

    monkeypatch.setattr(ledger_mod, "get_position", _fake_get_position_factory(0))
    monkeypatch.setattr(portfolio_risk_mod, "apply_live_portfolio_caps", fake_apply_live_caps)
    result = await _real_portfolio_caps_block("AAPL", "buy", 5, 100.0)
    assert result is not None
    assert "concentration cap" in result
    assert "Only 2 share(s) would fit." in result


async def test_portfolio_caps_block_prefers_real_shares_held_over_local_ledger(monkeypatch):
    # Confirmed gap in the 2026-08-04 paper-to-real switch review: a real
    # account's first-ever order for a ticker bought outside this app has no
    # `trades` row yet, so get_position alone reports shares=0 even though
    # the real account already holds some — making this cap blind to a
    # position that already exists. real_shares_held must win when it has an
    # answer, same correction ledger.rebalance_to_target/
    # check_and_apply_stop_loss already apply.
    import app.execution.ledger as ledger_mod
    import app.execution.portfolio_risk as portfolio_risk_mod

    captured_before = {}

    async def fake_apply_live_caps(ticker, target_shares, price, position_shares_before):
        captured_before["shares"] = position_shares_before
        return target_shares, None  # not capped

    async def fake_real_shares_held(ticker):
        return 100  # real KIS holding, unknown to the local ledger

    monkeypatch.setattr(ledger_mod, "get_position", _fake_get_position_factory(0))  # local ledger says flat
    monkeypatch.setattr(portfolio_risk_mod, "apply_live_portfolio_caps", fake_apply_live_caps)
    monkeypatch.setattr(portfolio_risk_mod, "real_shares_held", fake_real_shares_held)

    result = await _real_portfolio_caps_block("AAPL", "buy", 5, 100.0)

    assert result is None
    assert captured_before["shares"] == 100  # not the local ledger's 0


async def test_portfolio_caps_block_falls_back_to_local_ledger_when_no_real_broker(monkeypatch):
    import app.execution.ledger as ledger_mod
    import app.execution.portfolio_risk as portfolio_risk_mod

    captured_before = {}

    async def fake_apply_live_caps(ticker, target_shares, price, position_shares_before):
        captured_before["shares"] = position_shares_before
        return target_shares, None

    async def fake_real_shares_held(ticker):
        return None  # no live broker connected (e.g. a backtest process)

    monkeypatch.setattr(ledger_mod, "get_position", _fake_get_position_factory(7))
    monkeypatch.setattr(portfolio_risk_mod, "apply_live_portfolio_caps", fake_apply_live_caps)
    monkeypatch.setattr(portfolio_risk_mod, "real_shares_held", fake_real_shares_held)

    result = await _real_portfolio_caps_block("AAPL", "buy", 5, 100.0)

    assert result is None
    assert captured_before["shares"] == 7  # the local ledger's own count


async def test_portfolio_caps_block_failure_proceeds_without_blocking(monkeypatch):
    import app.execution.ledger as ledger_mod

    async def broken_get_position(ticker):
        raise RuntimeError("db down")

    monkeypatch.setattr(ledger_mod, "get_position", broken_get_position)
    result = await _real_portfolio_caps_block("AAPL", "buy", 5, 100.0)
    assert result is None


async def test_manual_order_blocked_by_portfolio_caps(live_settings, monkeypatch):
    async def engaged(ticker, side, shares, price):
        return "BLOCKED: Portfolio concentration cap: AAPL would reach 40%. Only 2 share(s) would fit."

    monkeypatch.setattr(kis_bridge, "portfolio_caps_block", engaged)
    r = await place_manual_order("AAPL", "buy", 5, order_type="limit", limit_price=100.0, broker=FakeBroker())
    assert r.status == "blocked" and "concentration cap" in r.message


class _Position:
    def __init__(self, shares: int, cash: float = 0.0):
        self.shares = shares
        self.cash = cash


def _fake_get_position_factory(shares: int):
    async def fake_get_position(ticker):
        return _Position(shares)

    return fake_get_position


# ---------------- _order_lock (TOCTOU serialization) ----------------


class RacyBroker(FakeBroker):
    """Cash is mutable, real state — decremented on each submitted order —
    unlike FakeBroker's fixed buyable_cash. Used to prove _order_lock
    actually serializes concurrent gate-check-then-submit sequences rather
    than letting two callers both read the same pre-order balance. The
    asyncio.sleep(0) calls are deliberate yield points: without the lock,
    they'd give the event loop a chance to interleave the two calls."""

    def __init__(self, cash: float, price_per_share: float = 10.0):
        super().__init__(buyable_cash=cash)
        self.submit_count = 0
        self._price_per_share = price_per_share

    async def fetch_buyable(self, code, price=None):
        await asyncio.sleep(0)
        return BuyableInfo(cash=self.buyable_cash, max_qty=1_000_000, max_amount=self.buyable_cash)

    async def submit_order(self, code, side, quantity, price=None):
        # mirror_trade_to_kis calls submit_order without a price for domestic
        # market orders, so the order's value is tracked via the fixed
        # per-share price the test itself used for the cash-sufficiency
        # check, not this call's own (unpassed) price argument.
        await asyncio.sleep(0)
        self.submit_count += 1
        self.buyable_cash -= quantity * self._price_per_share
        return OrderResult(order_no=f"ORD{self.submit_count}", order_time="090000", message="ok", org_no="ORG")


async def test_order_lock_serializes_concurrent_orders_against_shared_cash(live_settings):
    # Two orders of 60 each against a shared 100 balance: only one can fit.
    # Without _order_lock serializing the check-then-submit sequence, both
    # could read the same pre-order cash=100 and both pass.
    broker = RacyBroker(cash=100.0)
    results = await asyncio.gather(
        mirror_trade_to_kis("005930.KS", TODAY, "buy", 6, 10.0, broker=broker),
        mirror_trade_to_kis("005930.KS", TODAY, "buy", 6, 10.0, broker=broker),
    )
    submitted = [r for r in results if r.startswith("submitted")]
    blocked = [r for r in results if r.startswith("BLOCKED")]
    assert len(submitted) == 1
    assert len(blocked) == 1
    assert broker.submit_count == 1


async def test_order_lock_serializes_manual_orders_too(live_settings):
    broker = RacyBroker(cash=100.0)
    results = await asyncio.gather(
        place_manual_order("005930.KS", "buy", 6, reference_price=10.0, broker=broker),
        place_manual_order("005930.KS", "buy", 6, reference_price=10.0, broker=broker),
    )
    submitted = [r for r in results if r.ok]
    blocked = [r for r in results if r.status == "blocked"]
    assert len(submitted) == 1
    assert len(blocked) == 1
    assert broker.submit_count == 1
