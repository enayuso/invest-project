"""Single choke point between the paper ledger and the KIS paper account.

Every safety gate for real order submission lives HERE and only here —
`KISBroker.submit_order`/`submit_overseas_order` are deliberately
guard-free, and no other module may call them for trading. Two entry points
share these gates:

- `mirror_trade_to_kis` — the automated path: mirrors a trade the decision
  pipeline just recorded in the ledger.
- `place_manual_order` (app/execution manual orders) — the user-initiated
  path from the dashboard order ticket.

The gates, grouped by who applies them:

Shared (both paths), via the helpers below:
- Routable tickers only: KRX ("005930.KS" → "005930", market order) or US
  symbols ("AAPL", NASDAQ limit order — the paper server has no overseas
  market orders).
- Kill switch + daily max-loss circuit breaker (`app/execution/safety_switch.py`,
  agent-architecture-roadmap stage 2) — checked first among all gates, before
  even the master switch, since these are the "stop everything" mechanisms.
  Portfolio-wide, applies to both domestic and overseas, buys and sells (kill
  switch) or buys only (daily-loss breaker — sells always pass).
- Master switch (`KIS_EXECUTION_ENABLED`, default off).
- Positive share count.
- Futures/leveraged/inverse product name check (`KIS_BLOCK_LEVERAGED_INVERSE`,
  default on) — semi-auto/full-auto modes remove the human's per-trade
  judgment, and this system's sizing/stop-loss logic was never designed for
  margin-call/liquidation risk, so these are excluded outright rather than
  sized down. See `app.domain.leverage_filter`.
- Per-order value cap (`KIS_MAX_ORDER_KRW` / `KIS_MAX_ORDER_USD`).
- Approval threshold (`KIS_APPROVAL_*`, opt-in): an order at/above the value
  threshold needs explicit approval — the manual path supplies it from the
  dashboard, the auto path never does (so a large auto-order is held).
- Market-wide circuit-breaker check (domestic only) — a zero-cost in-memory
  read of `app.live.kis_market_status`'s state, a thin gate layer on top of
  `app.brokers.kis.domestic.websocket.RealtimeFeed` (the real-time WebSocket connection
  itself — market-operation status plus trade ticks/orderbook/fill
  notifications, started in `app.web.app`'s lifespan; KIS exposes no REST
  "is a circuit breaker active" query, so this can't be a REST call like the
  per-stock check below). Checked first among the two KIS-specific checks
  since it's free (no network round trip) and broader in scope (market-wide,
  not per-stock) — no reason to pay for a REST call if this already blocks.
  `True` (circuit breaker active) blocks. `None` (feed not connected, this
  code was never subscribed, or the `mkop_cls_code` value mapping isn't
  populated yet — see that module's docstring) is treated as
  **fail-closed**: also blocks, by deliberate, confirmed policy
  (`kis_cb_fail_open_when_unknown`, default `False`) — unlike the per-stock
  check's best-effort fail-open, since a market-wide halt is more severe
  than a single stock's, and this table starts empty until a value mapping
  is supplied from outside this codebase. Does NOT cover KOSDAQ (the
  trade allowlist's KRX tickers are all KOSPI-listed).
- KRX trading-halt / administrative-issue check (domestic only) — one REST
  call (`KISBroker.fetch_stock_status`, 주식기본조회) right before the order
  goes out; blocks if the stock is halted or KRX-administrative-issue-
  flagged. Best-effort: if the check itself errors (network/API hiccup),
  this proceeds to the order anyway — a genuinely halted stock is still
  rejected by KIS's own order call, so this is an early, clearer-error
  safety net, not the only line of defense. Nor US tickers, which have no
  equivalent check here.
- VI (변동성완화장치) halt check (domestic only, `_vi_block`) — REST call to
  `KISBroker.is_vi_triggered` (변동성완화장치 현황). Unlike the circuit-breaker
  gate above, this needs no external value-mapping confirmation: the TR
  returns the list of currently-halted codes directly. Best-effort, same
  fail-open-on-error convention as the per-stock halt check above.
- Cash-sufficiency check (`insufficient_cash_block`, buys only, both
  domestic and overseas) — an order whose estimated value exceeds what KIS
  itself reports as actually available to spend (매수가능조회 domestically,
  the present-balance TR's orderable USD overseas) is blocked before
  submission. Best-effort, same fail-open-on-error convention.
- Position-weight cap (`KIS_MAX_POSITION_PCT`, opt-in, domestic buys): blocks
  a buy that would push the ticker above the configured % of account equity.
- Dry-run (`KIS_EXECUTION_DRY_RUN`, default on): logs what would have been
  sent, then stops.

Manual-only:
- Portfolio-level caps (`portfolio_caps_block`, buys only) — the automated
  path's share count is already pre-shrunk by these same caps upstream (see
  `app.execution.ledger.rebalance_to_target`), so this is only wired into
  `place_manual_order`: without it, the order ticket could freely exceed
  concentration / correlation-bucket / drawdown-breaker limits the
  automated path always respects. Blocks outright (doesn't silently
  resize) — a manual order is a deliberate user action.
- Sell-quantity ceiling (domestic only): a KRX sell may not exceed the
  shares actually held. The dedicated 매도가능수량조회 TR (`KISBroker.
  fetch_sellable`) is unverified against the paper server, so this derives
  the ceiling from the balance inquiry instead — coarser (it doesn't net
  out shares locked in same-day unsettled sells) but it stops the one
  mistake that matters on paper: selling more than we own.

Mirror-only:
- Today's trades only — a backtest replaying history through
  `rebalance_to_target` must never leak orders into a live account.
- Trade allowlist membership — the agent can only trade what it watches.
- Stale position assumption (`assumed_position_mismatch_block`) — re-checks
  the caller's assumed pre-trade share count (what `rebalance_to_target`'s
  sizing math, or a stop-loss exit, believed was held) against a FRESH
  balance read taken right before submission. Broader than the sell-ceiling
  gate above: catches a wrong BUY sized off a stale assumption too, not just
  a SELL that would exceed a ceiling. Manual orders have no such upstream
  assumption to re-check (the user picks the quantity directly against
  what they're looking at right now), so this is mirror-only.

Mirroring is fire-and-report: the ledger entry is already recorded when
this runs, and a mirror failure must never crash the decision pipeline —
so this returns a disposition string instead of raising. The asymmetry is
deliberate: better a paper trade that silently didn't reach KIS (visible
in the next reconciliation) than a KIS order with no local record.
`place_manual_order` never raises either, returning a `ManualOrderResult`
with a status instead.
"""

from __future__ import annotations

import asyncio
import logging
import math
import re
from dataclasses import dataclass
from datetime import date

from app.brokers import KISBroker, get_broker, krx_code_from_ticker
from app.brokers.kis.const import is_permanent_refusal
from app.clock import local_today
from app.config import settings
from app.domain.leverage_filter import is_leveraged_or_inverse, is_unsupported_product_type
from app.domain.product_filter import out_of_scope_reason
from app.execution import risk_rules, safety_switch
from app.execution.portfolio_risk import market_lock, ticker_market
from app.live import kis_market_status

logger = logging.getLogger(__name__)

# Serializes every balance-backed gate (cash sufficiency, sell-ceiling,
# position-weight, portfolio caps) against actual order submission, across
# BOTH entry points. Without this, two nearly-simultaneous orders (a manual
# order racing an auto-mirrored one, or two quick manual clicks) could each
# read the same pre-order balance/position snapshot, both pass every gate,
# and both submit — jointly oversubscribing cash or overselling shares that
# neither order alone would have breached. A single process-wide lock is a
# deliberate blunt fix: this is a personal, single-account system with low
# order volume, so serializing submission (never true concurrent orders,
# rather than fine-grained per-ticker locking) trades a negligible amount of
# latency for eliminating the race entirely.
_order_lock = asyncio.Lock()

_US_SYMBOL = re.compile(r"^[A-Z]{1,5}$")

# KIS's own error code for "TR not available on the paper server" — the
# per-stock halt check (fetch_stock_status) hits this every time in paper
# mode, and was previously swallowed into a bare "proceeding without it"
# log line with no user-visible trace. Distinguishing this from a genuine
# transient failure (network blip, rate limit) lets the order result say
# specifically *why* the check was skipped instead of staying silent.
_PAPER_UNSUPPORTED_CODE = "EGW02006"


def _is_paper_unsupported(exc: Exception) -> bool:
    return _PAPER_UNSUPPORTED_CODE in str(exc)

# Fires once per process PER REASON the first time the circuit-breaker gate
# sees that specific unknown state, so a transient "code not subscribed yet"
# warning early in a cycle doesn't suppress a later, more serious "feed
# disconnected" warning (or vice versa) for the rest of the process's life.
# Shared by both entry points — one warning per reason, not one per entry
# point. 2026-09-04: was a single global bool before kis_market_status grew
# distinct reasons (see cb_unknown_reason).
_warned_cb_state_unknown_reasons: set[str] = set()


def _krx_code(ticker: str) -> str | None:
    return krx_code_from_ticker(ticker)


def _us_symbol(ticker: str) -> str | None:
    return ticker if _US_SYMBOL.match(ticker) else None


def route_ticker(ticker: str) -> tuple[str | None, str | None]:
    """(krx_code, us_symbol) for a ticker — at most one is non-None, both
    None when the ticker is neither a KRX nor a US symbol."""
    krx_code = _krx_code(ticker)
    us_symbol = _us_symbol(ticker) if krx_code is None else None
    return krx_code, us_symbol


async def unsupported_product_block(ticker: str, side: str) -> str | None:
    """A disposition string if `ticker` is a futures/leveraged/inverse-named
    product or an unsupported quote type (mutual fund, etc.), else None.
    Shared by both entry points — the mirror path is already filtered
    upstream (trade_allowlist_add and the screener universe both reject
    these by name/quote-type, see app.domain.leverage_filter), but a manual
    dashboard order can name any ticker with no upstream check at all
    (place_manual_order skips the trade_allowlist gate on purpose), so this
    is the one gate both paths actually need it for. Leverage/quote-type
    checks are off entirely when `kis_block_leveraged_inverse` is False.

    `side` gates the out-of-scope-product check ONLY (below) — cash-like/bond
    funds AND covered-call/option-income funds: that check runs on a BUY,
    never a SELL. This function is the buy/sell-common choke point, and a
    name filter that also fired on SELL would trap a position — you could no
    longer exit (stop-loss included) a bond ETF or covered-call ETF you
    already hold, or one added before this filter existed. The leverage/quote-type checks
    keep their existing side-agnostic behavior on purpose (a separate,
    pre-existing risk call, out of scope here).
    side는 현금성/채권형 펀드 체크에만 적용 — 매도까지 막으면 보유분을 손절
    포함 청산 불가가 됨. 레버리지·quote_type 체크의 기존 동작은 그대로 둠.

    Best-effort on a hard lookup FAILURE (the try/except below): proceeds,
    same convention as this file's other informational/advisory checks
    (halt, VI, cash) — a transient yfinance/master-file outage shouldn't
    block every order for an already-vetted, already-allowlisted ticker.

    NOT best-effort on a lookup that SUCCEEDS but resolves to no real name
    (`info.name == ticker`, the bare symbol handed back unchanged) — that
    used to proceed too, on the theory that "an unresolvable name is exactly
    as uninformative as a blank one" (which correctly proceeds, since an
    empty name matches no keyword and no multiplier pattern). It isn't
    uninformative: it's "we don't know", and treating unknown as safe is
    exactly backwards for a gate whose only job is refusing what it can't
    positively clear. Confirmed reachable: an NYSE-listed 3x ETF (SOXL,
    UPRO, TNA, ...) has no entry in the NASDAQ-only overseas master file
    (app.live.overseas_master, this project's whole overseas scope), so
    `get_ticker_display_info` falls back to the bare ticker string, which
    contains no leverage keyword and no digit+x pattern — this gate used to
    wave it through with no evidence it was ever safe."""
    if not settings.kis_block_leveraged_inverse:
        return None
    try:
        from app.dataflows.kis_provider import get_ticker_display_info  # local: avoid import at module load time

        info = await get_ticker_display_info(ticker)
    except Exception:
        logger.warning("leveraged/inverse name check failed for %s — proceeding without it", ticker, exc_info=True)
        return None
    if info.name == ticker:
        return f"BLOCKED: {ticker} has no resolvable company/product name — cannot confirm it isn't leveraged/inverse"
    if is_leveraged_or_inverse(info.name):
        return f"BLOCKED: {ticker} ({info.name}) is a futures/leveraged/inverse product — not supported"
    if is_unsupported_product_type(info.quote_type):
        return f"BLOCKED: {ticker} is a {info.quote_type} (not a plain stock or ETF) — not supported"
    if side == "buy" and settings.kis_block_cash_bond_funds:
        reason = out_of_scope_reason(info.name)
        if reason == "cash_or_bond_fund":
            return f"BLOCKED: {ticker} ({info.name}) is a cash-like/bond fund — outside this system's stock-analysis scope"
        if reason == "derivative_income_fund":
            return (
                f"BLOCKED: {ticker} ({info.name}) is a covered-call / option-income fund — "
                "its payoff is option-driven, outside this system's company-analysis scope"
            )
    return None


def max_order_value_cap(krx_code: str | None) -> tuple[float, str]:
    """(cap, currency) for a KRX code (KRW) or a US ticker (USD, `krx_code`
    None) — the same per-currency lookup `over_value_cap` uses, factored out
    so a caller that wants to size DOWN to the cap (app.execution.ledger's
    sizing chain, alongside its 1%-risk/portfolio caps) doesn't have to
    duplicate it, instead of only being usable to detect a violation."""
    return (
        (risk_rules.rule("kis_max_order_krw"), "KRW")
        if krx_code
        else (risk_rules.rule("kis_max_order_usd"), "USD")
    )


def over_value_cap(krx_code: str | None, shares: int, price: float) -> str | None:
    """A disposition string if the estimated order value exceeds the
    currency's cap, else None. Shared by both entry points.

    Deliberately strict `>`, not `>=`: a value exactly at the cap still needs
    approval (see `requires_approval`'s `>=`), it isn't auto-blocked here —
    see risk_rules._validate_cross_field's docstring for the full boundary.
    캡과 정확히 같은 값은 여기서 차단되지 않고 requires_approval의 승인 대상이 됨(의도된 동작).

    Kept as a hard block here even though app.execution.ledger's sizing
    chain now pre-shrinks an AUTO order to fit this same cap (see
    max_order_value_cap above) — this stays the defense-in-depth backstop
    for `place_manual_order` (a human's explicit order size is blocked and
    reported, never silently resized) and for any future auto caller that
    bypasses the ledger's sizing chain.

    Also guards against a non-finite `price` (NaN/Inf): confirmed live via a
    literal JSON `NaN` in `limit_price` reaching this function, since any
    `>`/`>=` comparison against NaN is False in Python and would otherwise
    silently pass every value-based gate."""
    if not math.isfinite(price):
        return f"BLOCKED: order price {price} is not a finite number"
    est_value = shares * price
    cap, currency = max_order_value_cap(krx_code)
    if est_value > cap:
        return (
            f"BLOCKED: estimated order value {est_value:,.2f} {currency} exceeds "
            f"cap {cap:,.2f} (KIS_MAX_ORDER_{currency})"
        )
    return None


async def _held_domestic_quantity(broker: KISBroker, krx_code: str) -> int:
    """Shares of `krx_code` currently held, from the balance inquiry — the
    paper-server-safe stand-in for 매도가능수량조회 (sell-ceiling gate).

    Deliberately a FRESH fetch, never the lock-scoped `_cached_fetch_balance`:
    `force_exit_position` (stop-loss exits) takes only a per-ticker lock, not
    `market_lock`, so a forced exit of THIS ticker can land mid-hold — and a
    cached held-quantity from before it would let an over-sell order through.
    On a buy this whole gate is already a no-op (`if side != "sell"`), so the
    extra fetch costs nothing there.
    매도 상한만은 항상 신선하게 — 강제청산이 락을 안 잡아 캐시된 보유수량은 초과매도를 통과시킬 수 있음."""
    balance = await broker.fetch_balance()
    return balance.held_quantity(krx_code)


async def domestic_sell_over_held(broker: KISBroker, krx_code: str, side: str, shares: int) -> str | None:
    """A disposition string if a domestic sell would exceed held shares, else
    None. No-op for buys and overseas. Shared by both entry points."""
    if side != "sell":
        return None
    held = await _held_domestic_quantity(broker, krx_code)
    if shares > held:
        return f"BLOCKED: sell {shares} exceeds held {held} (balance-derived sellable ceiling)"
    return None


async def _held_overseas_quantity(broker: KISBroker, us_symbol: str) -> float:
    """Shares of `us_symbol` currently held, from the overseas balance
    inquiry — the overseas counterpart to `_held_domestic_quantity`. `float`,
    not `int`: overseas share trading supports fractional quantities (see
    `app.brokers.kis.overseas.account.fetch_overseas_balance`'s own
    `quantity` field)."""
    balance = await broker.fetch_overseas_balance()
    return balance.held_quantity(us_symbol)


async def overseas_sell_over_held(broker: KISBroker, us_symbol: str, side: str, shares: int) -> str | None:
    """A disposition string if an overseas sell would exceed held shares,
    else None. No-op for buys and domestic. Mirrors `domestic_sell_over_held`
    — until this existed, an overseas sell had no real-balance ceiling at
    all, so a stop-loss force-sell against a ledger position inflated by an
    unreversed rejected/expired pending order (see
    `app.execution.ledger.reverse_pending_trade`) could submit a KIS sell
    order for more shares than were actually held."""
    if side != "sell":
        return None
    held = await _held_overseas_quantity(broker, us_symbol)
    if shares > held:
        return f"BLOCKED: sell {shares} exceeds held {held:g} (overseas balance-derived sellable ceiling)"
    return None


async def assumed_position_mismatch_block(
    broker: KISBroker,
    krx_code: str | None,
    us_symbol: str | None,
    assumed_current_shares: float | None,
) -> str | None:
    """A disposition string if `assumed_current_shares` — what the caller's
    sizing math (rebalance_to_target's `current_shares`, or a stop-loss
    force-exit's `position.shares`) believed was held BEFORE this order —
    no longer matches a FRESH balance read taken right now, else None.
    No-op when `assumed_current_shares` is None (a caller that doesn't pass
    it, e.g. place_manual_order — a manual order's quantity is picked by
    the user directly against whatever they're looking at right now, not
    computed from an assumed prior position, so there's nothing to compare
    against here).

    This is deliberately broader than `domestic_sell_over_held`/
    `overseas_sell_over_held`: those only catch a SELL whose quantity
    exceeds real holdings. A stale assumption can just as easily produce a
    wrong BUY (e.g. the caller believed 0 shares were held and sized a "buy
    5" from that baseline when 10 were actually already held) — no ceiling
    is breached, but the delta itself was computed from a false premise.
    Checking the ASSUMPTION itself, not just whether the resulting order
    happens to exceed a ceiling, catches that case too.

    Real incidents this generalizes the fix for (see app.execution.ledger.
    rebalance_to_target's own docstring and memory: kis-integration-status):
    an AI decision reasoning from a stale/wrong current-position figure
    (255 vs 1 share; 0 vs 10 shares) already happened more than once. The
    upstream fix (real_shares_held/real_equity_native grounding) addresses
    the common case, but this is the last checkpoint before an order
    actually reaches KIS — it catches a stale assumption regardless of
    WHERE upstream it crept back in (a code path that doesn't call
    real_shares_held, a race between this decision and a manual/other
    trade, or a future regression), not just the specific cause already
    fixed once.

    Fractional (overseas) comparisons use a small tolerance instead of
    exact equality — real-world quantities here are still whole shares in
    practice, but the balance API's own type is `float` (see
    `_held_overseas_quantity`), so exact `==` would be fragile against
    incidental floating-point noise."""
    if assumed_current_shares is None:
        return None
    if krx_code is not None:
        actual = await _held_domestic_quantity(broker, krx_code)
        if actual != assumed_current_shares:
            return (
                f"BLOCKED: stale position assumption for {krx_code} — sizing assumed "
                f"{assumed_current_shares} share(s) held, but a fresh balance check right "
                f"now shows {actual}. Refusing to submit an order computed from an outdated "
                "position; the caller should re-fetch and re-size."
            )
        return None
    if us_symbol is not None:
        actual = await _held_overseas_quantity(broker, us_symbol)
        if abs(actual - assumed_current_shares) > 1e-6:
            return (
                f"BLOCKED: stale position assumption for {us_symbol} — sizing assumed "
                f"{assumed_current_shares:g} share(s) held, but a fresh balance check right "
                f"now shows {actual:g}. Refusing to submit an order computed from an outdated "
                "position; the caller should re-fetch and re-size."
            )
        return None
    return None


def requires_approval(
    krx_code: str | None, shares: int, price: float, approved: bool
) -> str | None:
    """A disposition string if this order needs human approval it doesn't
    have, else None. Rule (승인 임계금액): an order valued at or above the
    per-currency threshold may not proceed unless `approved` is True — the
    auto path never sets it (large auto-orders are held), the manual path
    sets it from the dashboard's confirmation. Off when the rule is disabled.

    Sync (no broker needed), so it is enforced on every path including
    dry-run. Reads settings today; the A-3 runtime store will resolve on top.

    Also guards against a non-finite `price` (NaN/Inf) — same real incident
    as `over_value_cap`'s own guard: a NaN price makes `>=` silently False,
    so this required approval instead of skipping it."""
    if not risk_rules.rule("kis_approval_required_enabled") or approved:
        return None
    if not math.isfinite(price):
        return f"NEEDS_APPROVAL: order price {price} is not a finite number; explicit approval required"
    value = shares * price
    currency = "KRW" if krx_code else "USD"
    threshold = (
        risk_rules.rule("kis_approval_threshold_krw")
        if krx_code
        else risk_rules.rule("kis_approval_threshold_usd")
    )
    if value >= threshold:
        return (
            f"NEEDS_APPROVAL: order value {value:,.2f} {currency} >= approval threshold "
            f"{threshold:,.2f} (KIS_APPROVAL_THRESHOLD_{currency}); explicit approval required"
        )
    return None


async def over_position_pct(
    broker: KISBroker, krx_code: str | None, side: str, shares: int, price: float
) -> str | None:
    """A disposition string if a domestic BUY would push the ticker's weight
    in the account above the configured cap (종목당 최대 포지션 비중), else
    None. No-op for sells (they reduce weight), when the rule is disabled, or
    when account equity is unavailable (can't compute → don't block).

    Weight basis (decided): the denominator is 총평가금액 (`tot_evlu_amt`,
    total account value INCLUDING cash) — a position-concentration limit is
    conventionally measured against total capital (NAV), not against the
    holdings sub-total. On a paper/cash account this ≈ 순자산금액 (nass_amt).
    So Weight = (already-held eval + this order's value) / 총평가금액.
    (KIS balance output2 also exposes `evlu_amt_smtl_amt` = holdings-only
    sub-total; we deliberately do NOT use it as the denominator.)

    Overseas is NOT enforced (assumption): a foreign position's weight would
    need FX normalization to the account base currency, which is not wired up
    yet. Overseas buys are treated as no-op here (documented TODO) — see
    README. Needs the balance inquiry, so it runs on the broker-backed path
    only.

    Also guards against a non-finite `price` (NaN/Inf) — same real incident
    as `over_value_cap`'s own guard: NaN would propagate through the weight
    division and make the `>` comparison silently False.
    """
    if not risk_rules.rule("kis_max_position_pct_enabled") or side != "buy" or krx_code is None:
        return None
    if not math.isfinite(price):
        return f"BLOCKED: order price {price} is not a finite number"
    # Buy-side concentration cap — same freshness class as
    # apply_live_portfolio_caps, and only ever reached inside a market_lock
    # hold. Uses the lock-scoped balance cache so re-enabling this gate
    # doesn't re-add a ledger inquiry the sizing step just made.
    # 락 스코프 잔고 캐시 사용 — 재활성 시 조회가 다시 늘지 않도록.
    from app.execution.portfolio_risk import _cached_fetch_balance  # local: circular-import convention in this file

    balance = await _cached_fetch_balance(broker)
    equity = balance.total_eval_amount  # 총평가금액(tot_evlu_amt): 현금 포함 계좌 전체
    if equity <= 0:
        logger.warning("position-pct gate skipped: account equity unavailable (0)")
        return None
    held_eval = next((p.eval_amount for p in balance.positions if p.code == krx_code), 0.0)
    new_weight = (held_eval + shares * price) / equity * 100
    cap = risk_rules.rule("kis_max_position_pct")
    if new_weight > cap:
        return (
            f"BLOCKED: {krx_code} would be {new_weight:.1f}% of account "
            f"(cap {cap:.1f}%, KIS_MAX_POSITION_PCT)"
        )
    return None


async def _circuit_breaker_block(krx_code: str, *, checked_out: list[str] | None = None) -> str | None:
    """A disposition string if the market-wide circuit-breaker gate blocks
    this order, else None. Shared by both entry points — see the module
    docstring for the fail-closed-on-unknown policy.

    2026-09-04: when the state is unknown, branches on
    kis_market_status.cb_unknown_reason(krx_code) rather than using one
    message for every cause. A freshly-selected screening candidate whose
    realtime subscription simply hasn't been ACKed yet ("code_not_subscribed"
    with the feed itself connected) is a routine, likely-transient timing gap
    — not evidence the ticker itself is risky — and its wording is softened
    so the downstream PM/risk-debate prompt doesn't read a pending
    subscription as a reason to downgrade a Buy (same "execution-mechanics-
    isn't-evidence" pattern as f79213e/2916498, applied to the block MESSAGE
    text itself rather than a prompt). A dead/never-started feed
    ("feed_disconnected"/"feed_not_started") is a real, non-transient safety
    gap and keeps the existing strong wording.

    2026-09-07 (halt-status narrative fix): `checked_out`, if given, gets
    `"circuit_breaker"` appended when this gate reached a genuinely KNOWN
    state (`cb_active` True or False) — NOT when it's `None` (unknown,
    including the fail-open case), since "unknown, allowed through" is
    exactly the kind of not-really-verified outcome execution_blockers'
    caller (pipeline._execution_status_context) needs to tell apart from a
    real, confirmed-clear read. See execution_blockers's own docstring."""
    cb_active = kis_market_status.is_circuit_breaker_active(krx_code)
    if cb_active is not None and checked_out is not None:
        checked_out.append("circuit_breaker")
    if cb_active is True:
        cb_status = kis_market_status.current_status(krx_code)
        mkop_cls_code = cb_status.market_op_code if cb_status is not None else "?"
        return f"BLOCKED: circuit breaker active for {krx_code} (mkop_cls_code={mkop_cls_code!r})"
    if cb_active is None:
        reason = kis_market_status.cb_unknown_reason(krx_code)
        if reason not in _warned_cb_state_unknown_reasons:
            logger.warning(
                "circuit-breaker gate saw an unknown state for %s (%s) — "
                "kis_cb_fail_open_when_unknown=%s",
                krx_code, reason, settings.kis_cb_fail_open_when_unknown,
            )
            _warned_cb_state_unknown_reasons.add(reason)
        if not settings.kis_cb_fail_open_when_unknown:
            # 2026-10-02: the two causes below used to share the "ACK pending,
            # likely transient" wording though neither is transient. Facts
            # only — no ticker-risk wording (execution-mechanics-isn't-evidence).
            if reason == "registration_capped":
                usage = kis_market_status.registration_usage()
                budget = f" ({usage[0]}/{usage[1]})" if usage is not None else ""
                return (
                    f"BLOCKED: circuit-breaker state unknown for {krx_code} — KIS real-time "
                    f"registration budget exhausted{budget}, this code was never subscribed "
                    "(not a transient ACK delay) (kis_cb_fail_open_when_unknown=false)"
                )
            if reason == "subscribe_rejected":
                return (
                    f"BLOCKED: circuit-breaker state unknown for {krx_code} — KIS rejected the "
                    "real-time subscription for this code (kis_cb_fail_open_when_unknown=false)"
                )
            if reason == "code_not_subscribed":
                return (
                    f"BLOCKED: circuit-breaker state unknown for {krx_code} — realtime "
                    "subscription pending (feed connected, ACK not yet received for this "
                    "ticker) — likely transient, expected to clear once the subscription is "
                    f"acknowledged (kis_cb_fail_open_when_unknown=false)"
                )
            return (
                f"BLOCKED: circuit-breaker state unknown for {krx_code} — "
                f"{'realtime feed not connected' if reason == 'feed_disconnected' else 'realtime feed not started'} "
                "(kis_cb_fail_open_when_unknown=false)"
            )
    return None


async def _stock_halt_block(
    broker: KISBroker, krx_code: str, ticker: str, *, checked_out: list[str] | None = None
) -> tuple[str | None, str | None]:
    """(blocked, note). `blocked` is a disposition string if the per-stock
    REST halt/administrative-issue check blocks this order, else None.
    Best-effort: proceeds (blocked=None) if the check itself errors — a
    genuinely halted stock is still rejected by KIS's own order call.
    `note` is a non-blocking advisory surfaced in the final order message
    when the check was skipped for a KNOWN reason (paper server doesn't
    support this TR at all — 모의 미지원, confirmed via EGW02006) rather
    than a genuine transient error, so the order result says specifically
    why the safety check didn't run instead of silently proceeding with no
    trace. Shared by both entry points.

    2026-09-07 (halt-status narrative fix): `checked_out`, if given, gets
    `"stock_halt"` appended only when `fetch_stock_status` actually
    completed (the try block below did not raise) — the paper-unsupported
    and generic-error branches both leave the real status unknown, so
    neither counts as a completed check. See `_circuit_breaker_block`'s own
    note on the same convention."""
    try:
        status = await broker.fetch_stock_status(krx_code)
    except Exception as exc:
        if _is_paper_unsupported(exc):
            logger.info("KIS stock-status check unsupported in paper mode for %s — proceeding without it", ticker)
            return None, "매매정지 확인 생략(모의투자 미지원 — 실전에서는 정상 동작)"
        logger.warning("KIS stock-status check failed for %s — proceeding without it", ticker, exc_info=True)
        return None, None
    if checked_out is not None:
        checked_out.append("stock_halt")
    if status.trading_halted or status.administrative_issue:
        return (
            f"BLOCKED: {ticker} is currently trading-halted or an administrative issue "
            f"(tr_stop_yn={status.trading_halted}, admn_item_yn={status.administrative_issue})"
        ), None
    return None, None


async def _vi_block(
    broker: KISBroker, krx_code: str, ticker: str, *, checked_out: list[str] | None = None
) -> str | None:
    """A disposition string if `krx_code` is currently under a VI
    (변동성완화장치, volatility interruption) halt, else None. Unlike the
    market-wide circuit-breaker gate, this doesn't need any VI_CLS_CODE
    value-mapping guesswork — `is_vi_triggered` checks REST 변동성완화장치
    현황 (inquire_vi_status), which returns the *list* of currently-halted
    codes directly, confirmed working on the paper server (see
    app.brokers.kis.domestic.info.fetch_vi_status's docstring). Best-effort:
    proceeds if the check itself errors — a genuinely VI-halted stock's
    order simply won't fill, same convention as the per-stock halt check.
    Applies to both buys and sells (VI pauses matching entirely, not just
    new exposure). Shared by both entry points.

    2026-09-07 (halt-status narrative fix): `checked_out`, if given, gets
    `"vi"` appended only when `is_vi_triggered` actually completed — same
    convention as `_stock_halt_block`'s own `checked_out`."""
    try:
        triggered = await broker.is_vi_triggered(krx_code)
    except Exception:
        logger.warning("VI status check failed for %s — proceeding without it", ticker, exc_info=True)
        return None
    if checked_out is not None:
        checked_out.append("vi")
    if triggered:
        return f"BLOCKED: {ticker} is currently under a VI (변동성완화장치) halt"
    return None


async def insufficient_cash_block(
    broker: KISBroker, krx_code: str | None, us_symbol: str | None, side: str, shares: int, price: float
) -> str | None:
    """A disposition string if a BUY's estimated value exceeds what KIS
    itself reports as actually available to spend right now, else None.
    No-op for sells (a sell needs held shares, not cash — see
    `domestic_sell_over_held`). Domestic uses 매수가능조회 (`fetch_buyable`);
    overseas uses the present-balance TR's orderable USD (the same field
    `/api/buyable` already surfaces for display — this is the first place
    it's enforced as a gate rather than just shown). Best-effort: proceeds
    if the check itself errors — KIS's own order call still rejects an
    over-cash order, so this is an early, clearer-error safety net, not the
    only line of defense, same convention as the per-stock halt check.
    Shared by both entry points.

    Also guards against a non-finite `price` (NaN/Inf) — same real incident
    as `over_value_cap`'s own guard: NaN would make both the domestic and
    overseas `>` comparisons below silently False."""
    if side != "buy":
        return None
    if not math.isfinite(price):
        return f"BLOCKED: order price {price} is not a finite number"
    order_value = shares * price
    if krx_code is not None:
        try:
            buyable = await broker.fetch_buyable(krx_code, price)
        except Exception:
            logger.warning("cash-sufficiency check failed for %s — proceeding without it", krx_code, exc_info=True)
            return None
        if order_value > buyable.cash:
            return (
                f"BLOCKED: order value {order_value:,.2f} KRW exceeds available cash "
                f"{buyable.cash:,.2f} KRW (매수가능조회)"
            )
        return None
    if us_symbol is not None:
        try:
            cash = await broker.fetch_overseas_present_balance("USD")
        except Exception:
            logger.warning("cash-sufficiency check failed for %s — proceeding without it", us_symbol, exc_info=True)
            return None
        if not cash.matched:
            # cash.orderable is a zeroed placeholder here, not a confirmed $0
            # (see OverseasCash.matched) — still block (same fail-safe
            # direction as every other "state unknown" gate in this file),
            # just don't word it as if we know the real balance is 0.
            # 확정된 0원이 아니므로 문구만 다르게, 차단 방향은 동일하게 유지.
            return "BLOCKED: USD cash-sufficiency check inconclusive — no matching currency row in present-balance response"
        # orderable (frcr_drwg_psbl_amt_1) reads 0.000000 for every currency
        # on this account regardless of whether real cash is held — confirmed
        # 2026-07-20 by inspecting the raw present-balance response directly
        # (not just this account's quirk: WCRC_FRCR_DVSN_CD 01 vs 02 returned
        # byte-identical bodies too, so there's no alternate call that fixes
        # it either). `deposit` (derived from output3.frcr_evlu_tota, verified
        # reliable) is the floor: this account carries no margin/short
        # exposure, so orderable cash can never legitimately exceed total
        # deposit — using it here only ever turns a wrongly-blocked, fundable
        # order into a correctly-allowed one, never the reverse (a genuinely
        # empty account has deposit == 0 too). Deliberately NOT baked into
        # fetch_overseas_present_balance itself — the dashboard's 원화예수금
        # display reads that same raw orderable and must show the genuine
        # (zero) figure, not this gate's own floor.
        # 모의계좌에서 orderable이 통화 무관하게 항상 0으로 읽힘(원문 응답 직접 확인) — 신용/공매도
        # 없는 계좌라 orderable이 deposit을 못 넘으므로, 주문 게이트에서만 deposit으로 대체.
        available = cash.orderable if cash.orderable > 0 else cash.deposit
        if order_value > available:
            return (
                f"BLOCKED: order value {order_value:,.2f} USD exceeds available cash "
                f"{available:,.2f} USD (해외 주문가능금액)"
            )
        return None
    return None


async def execution_blockers(
    broker: KISBroker, krx_code: str, ticker: str, *, checked_out: list[str] | None = None
) -> list[str]:
    """Read-only aggregate of the three domestic execution-time gates above
    (`_circuit_breaker_block`/`_stock_halt_block`/`_vi_block`) — for callers
    that want to know WHY an order might get blocked BEFORE actually
    attempting one (currently: the Portfolio Manager's decision context, so
    its narrative doesn't keep recommending a trade that's silently blocked
    every cycle at execution time). Calls the exact same gates with no order
    attempt or side effect, so this can never drift from what actually
    blocks a real order. Domestic only — VI/circuit-breaker have no KIS
    overseas equivalent. Empty list if nothing currently blocks.
    실행 게이트 3개를 그대로 재사용한 읽기전용 조회 — AI 판단 전에 차단 사유를 미리 보여줌.

    2026-09-07 (id=1238 halt-status narrative incident): `checked_out`, if
    given, is extended with the name of each of the three gates
    (`"circuit_breaker"`/`"stock_halt"`/`"vi"`) that actually COMPLETED a
    real check this call — not merely "returned no block". A gate whose
    check was swallowed by an exception, skipped as paper-mode-unsupported,
    or left in a genuinely unknown state is absent from this list even
    though it also didn't add anything to the returned blockers. This lets
    `pipeline._execution_status_context` tell "confirmed clear" (all three
    present) apart from "could not fully verify this cycle" (one or more
    missing) instead of treating every empty `blockers` list as an
    equally-confirmed all-clear — the gap that let a stale news headline's
    false halt claim go uncorrected in the PM's decision text (id=1238,
    351320.KS) even though the broker-side gates themselves were never
    actually wrong."""
    blockers: list[str] = []
    cb = await _circuit_breaker_block(krx_code, checked_out=checked_out)
    if cb:
        blockers.append(cb)
    halt, _halt_note = await _stock_halt_block(broker, krx_code, ticker, checked_out=checked_out)
    if halt:
        blockers.append(halt)
    vi = await _vi_block(broker, krx_code, ticker, checked_out=checked_out)
    if vi:
        blockers.append(vi)
    return blockers


async def designation_block(
    broker: KISBroker, krx_code: str, ticker: str, *, checked_out: list[str] | None = None
) -> str | None:
    """A disposition string if `krx_code` currently carries a KRX
    pre-screening designation this function trusts enough to act on, else
    None. Public (no leading underscore) — called directly from
    app.live.auto_trade_loop's own `_drop_designated`, not from
    `pre_screening_blockers` (see the 2026-09-08 §4-5 note below for why).

    Started as `_market_warning_block` (2026-09-08, domestic AI audit: all
    four decisions that cycle — 082850/033790/387690/014950, ids 1312-1315 —
    were investment-warning/caution-designated names that reached full
    4-analyst + debate + PM analysis, ~150-200K tokens each, only to end in
    Hold with the designation cited as the PM's main reason not to buy).
    Extended the same day (§4-5 follow-up) to also read `short_over_yn`
    (단기과열여부) and `sltr_yn`(정리매매여부) — both sat unparsed in the
    exact same TR response. Consolidated into ONE call via
    `fetch_designations` rather than three separate blocker functions: the
    three flags share a single response, so splitting them would triple
    the network cost this filter exists specifically to avoid (mirrors how
    `_stock_halt_block` below already judges two conditions,
    tr_stop_yn/admn_item_yn, off one response rather than being two
    functions).

    `mang_issu_cls_code` (관리종목여부) is read but NOT included in the
    returned reasons — its enum direction isn't documented anywhere (see
    `StockDesignations`' own docstring) and guessing wrong would drop every
    non-designated candidate. It's logged at INFO whenever non-empty so the
    first live hit can confirm or refute the assumed sentinel, the same
    detect-then-promote path `mrkt_warn_cls_code` itself went through.

    Same TR (FHKST01010100) `_stock_halt_block`/`fetch_price`/
    `fetch_valuation`/`fetch_sector_name` already use — real AND paper both
    supported (unlike `_stock_halt_block`'s CTPF1002R, this is a plain
    market-data TR, not account-gated). Best-effort: proceeds if the check
    itself errors, same convention as `_stock_halt_block`/`_vi_block` — a
    genuinely-designated name that slips through here still gets Hold from
    the LLM most of the time (this is a cost-saving pre-filter, not a new
    safety gate: a designated name is not thereby unexecutable, so this is
    NOT added to `execution_blockers`).

    2026-09-08: `checked_out`, if given, gets `"designation"` appended only
    when `fetch_designations` actually completed — same convention as
    `_stock_halt_block`/`_vi_block`'s own `checked_out`.

    §4-5 exemption note: this is deliberately NOT wired into
    `pre_screening_blockers` any more (it was, briefly, as
    `_market_warning_block`) because `pre_screening_blockers` runs
    unconditionally over every candidate INCLUDING a forced held-position
    review (`_held_needing_review`/`_cash_freed_up_for_review`) — and a
    position freshly designated administrative (delisting risk) or under
    a market-warning tier is exactly the moment that forced review exists
    to catch, not a candidate to silently drop before the AI ever sees it.
    `app.live.auto_trade_loop._drop_designated` calls this function
    directly and exempts the forced-review set the same way `_drop_illiquid`
    already exempts `held_review` — see that filter's own docstring."""
    try:
        d = await broker.fetch_designations(krx_code)
    except Exception:
        logger.warning("KIS designation check failed for %s — proceeding without it", ticker, exc_info=True)
        return None
    if checked_out is not None:
        checked_out.append("designation")
    if d.administrative_raw:
        logger.info(
            "designation_block: %s carries mang_issu_cls_code=%r — NOT used to drop yet "
            "(enum direction unconfirmed, see StockDesignations' docstring)",
            ticker, d.administrative_raw,
        )
    reasons: list[str] = []
    if d.market_warning and d.market_warning not in ("0", "00"):
        reasons.append(f"시장경고 mrkt_warn_cls_code={d.market_warning!r}")
    if d.short_overheat:
        reasons.append("단기과열 short_over_yn=Y")
    if d.liquidation:
        reasons.append("정리매매 sltr_yn=Y")
    if not reasons:
        return None
    return f"BLOCKED: {ticker} carries a KRX designation ({'; '.join(reasons)})"


async def pre_screening_blockers(broker: KISBroker, krx_code: str, ticker: str) -> list[str]:
    """Same idea as `execution_blockers` — read-only, no order attempt —
    but deliberately WITHOUT `_circuit_breaker_block`, for a caller that
    wants to drop a candidate before it's necessarily on the trade allowlist at
    all (app.live.auto_trade_loop's momentum-discovered candidates, to
    skip a full LLM analysis on one that can't execute anyway). The
    circuit-breaker gate's "unknown state" branch fails closed by default
    (`kis_cb_fail_open_when_unknown=False`) whenever a code isn't
    currently subscribed to the real-time market-status feed — and a
    fresh momentum candidate typically ISN'T yet, since `add_code()` only
    fires from `add_ticker_to_trade_allowlist` (dashboard add or the AI
    auto-add-on-BUY hook), which happens later than candidate discovery.
    Using the full `execution_blockers` here would therefore fail-closed
    almost every non-trade allowlist candidate on "unknown," not genuine
    circuit-breaker activity — confirmed by this codebase's own test
    suite the day this function was added. `_stock_halt_block`/`_vi_block`
    have no such dependency (direct REST polls, not a subscription-gated
    feed), so they're safe to reuse here unmodified. The circuit-breaker
    gate itself is NOT skipped for the ticker's whole lifecycle — it's
    still checked later by `execution_blockers` (Portfolio Manager
    context) and for real at order-execution time, by which point a
    ticker that's actually going to trade has normally been through
    `add_ticker_to_trade_allowlist` and is subscribed. Domestic only,
    same scope as `execution_blockers`.

    2026-09-08: briefly also ran a KRX market-warning check here
    (`_market_warning_block`) — moved out the same day (§4-5) to
    `designation_block`, called separately by app.live.auto_trade_loop's
    `_drop_designated` so the forced held-position review path can be
    exempted from it. See `designation_block`'s own docstring for the
    full reasoning; this function's scope is back to halt/VI only."""
    blockers: list[str] = []
    halt, _halt_note = await _stock_halt_block(broker, krx_code, ticker)
    if halt:
        blockers.append(halt)
    vi = await _vi_block(broker, krx_code, ticker)
    if vi:
        blockers.append(vi)
    return blockers


def spread_pct(bid: float, ask: float) -> float | None:
    """Pure — bid-ask spread as a fraction of the mid price. None if the
    quote isn't usable (non-positive bid/ask, or ask <= bid — a crossed/
    locked or simply missing quote, e.g. outside continuous trading
    hours when KIS's asking-price TRs return zeros for everything but the
    last trade — confirmed live, 2026-07-29, for the overseas TR with the
    US market closed) rather than treated as a real zero-width spread."""
    if bid <= 0 or ask <= 0 or ask <= bid:
        return None
    mid = (bid + ask) / 2
    return (ask - bid) / mid


async def spread_blocker(broker: KISBroker, ticker: str) -> str | None:
    """A disposition string if `ticker`'s current bid-ask spread exceeds
    `settings.max_spread_pct`, else None — screening-method-improvement-
    ideas item #3 (유동성/스프레드 필터 부재): a thin, wide-spread name (a
    one-day event-driven volume spike, not real liquidity) could otherwise
    reach full LLM analysis and a BUY decision on momentum alone. Domestic
    AND overseas, unlike VI/circuit-breaker (KRX-only) — picks the right
    quote call via `route_ticker`. Best-effort: a quote fetch failure or
    an unusable quote (see `spread_pct`) proceeds rather than blocking —
    same "cost-saving pre-filter, never stricter than necessary"
    convention as `pre_screening_blockers`."""
    krx_code, us_symbol = route_ticker(ticker)
    try:
        if krx_code is not None:
            quote = await broker.fetch_asking_price(krx_code)
        elif us_symbol is not None:
            quote = await broker.fetch_overseas_asking_price(us_symbol)
        else:
            return None
    except Exception:
        logger.warning("spread check failed for %s — proceeding without it", ticker, exc_info=True)
        return None
    pct = spread_pct(quote.bid, quote.ask)
    if pct is None:
        return None
    if pct > settings.max_spread_pct:
        return f"BLOCKED: {ticker} spread too wide ({pct:.2%} of mid, > {settings.max_spread_pct:.2%} limit)"
    return None


async def available_cash_native(
    broker: KISBroker, krx_code: str | None, us_symbol: str | None, price: float
) -> tuple[float, str] | None:
    """Real orderable cash in native currency, read-only — same source
    `insufficient_cash_block` enforces against, exposed here for context-
    building instead of gating an order. Deliberately NOT sharing code with
    `insufficient_cash_block` itself: that function's overseas branch has a
    deliberate BLOCKING behavior on `not cash.matched` (inconclusive data
    fails closed) that doesn't translate to a narrative-only reading — reusing
    it here would mean either duplicating that nuance anyway or risking a
    change to the live-trading gate for a context-only feature. Returns
    `(amount, currency)`, or `None` if the check itself errors or the
    overseas response didn't match a currency row (inconclusive, not a
    confirmed figure). 실거래 게이트(insufficient_cash_block)는 그대로 두고,
    같은 소스를 읽기전용으로만 재조회."""
    if krx_code is not None:
        try:
            buyable = await broker.fetch_buyable(krx_code, price)
        except Exception:
            logger.warning("available_cash_native: buyable lookup failed for %s", krx_code, exc_info=True)
            return None
        return buyable.cash, "KRW"
    if us_symbol is not None:
        try:
            cash = await broker.fetch_overseas_present_balance("USD")
        except Exception:
            logger.warning("available_cash_native: present-balance lookup failed for %s", us_symbol, exc_info=True)
            return None
        if not cash.matched:
            return None
        # Same orderable-reads-zero fallback as insufficient_cash_block — see
        # that function's comment for the confirmed 2026-07-20 root cause.
        available = cash.orderable if cash.orderable > 0 else cash.deposit
        return available, "USD"
    return None


async def portfolio_caps_block(ticker: str, side: str, shares: int, price: float) -> str | None:
    """A disposition string if a MANUAL buy would breach a portfolio-level
    cap (concentration / correlation-bucket / drawdown breaker —
    app.execution.portfolio_risk), else None.

    The automated rebalance path already has its share count pre-shrunk by
    these same caps before it ever reaches kis_bridge (see
    app.execution.ledger.rebalance_to_target's own apply_live_portfolio_caps
    call) — a manual dashboard order has no such upstream step, so without
    this the order ticket could freely blow through portfolio-wide caps
    that the automated path always respects. Manual-path only (not called
    from mirror_trade_to_kis) for exactly that reason.

    Sells never blocked (same "reduce, never loosen" philosophy every other
    portfolio-level cap uses). Blocks outright rather than silently
    resizing — unlike the automated path, a manual order is a deliberate
    user action, so the user picks the smaller size themselves rather than
    having it picked for them. Best-effort: proceeds if the check itself
    errors, same convention as every other broker-backed gate here.

    Prefers the real KIS-reported quantity over the internal ledger's own
    count, same "real balance over internal ledger" correction already
    applied in app.execution.ledger.rebalance_to_target/
    check_and_apply_stop_loss — without it, a real account's first-ever
    order for a ticker bought outside this app (no `trades` row yet) would
    have get_position return shares=0, making this cap blind to a position
    that already exists (confirmed as a real gap in the 2026-08-04
    paper-to-real switch review, unlike the two call sites above which
    already had this correction)."""
    if side != "buy":
        return None
    # Local imports: app.execution.ledger imports this module (mirror_trade_to_kis),
    # so a top-level import back here would be circular — same pattern
    # app.execution.portfolio_risk itself already uses for the same reason.
    from app.execution.ledger import get_position
    from app.execution.portfolio_risk import apply_live_portfolio_caps, real_shares_held

    try:
        position = await get_position(ticker)
        current_shares = position.shares
        real_shares = await real_shares_held(ticker)
        if real_shares is not None:
            current_shares = real_shares
        target_shares = current_shares + shares
        capped_shares, note = await apply_live_portfolio_caps(ticker, target_shares, price, current_shares)
    except Exception:
        logger.warning("portfolio-cap check failed for %s — proceeding without it", ticker, exc_info=True)
        return None
    if capped_shares < target_shares:
        allowed = max(capped_shares - current_shares, 0)
        return f"BLOCKED: {note or 'a portfolio-level cap would be exceeded'} Only {allowed} share(s) would fit."
    return None


async def _domestic_gates_block(
    broker: KISBroker, krx_code: str, ticker: str, side: str, shares: int, price: float
) -> tuple[str | None, str | None]:
    """(blocked, note) — the ordered domestic-only gate sequence
    (circuit-breaker -> per-stock halt -> VI halt -> sell-ceiling ->
    position-weight), identical for both entry points and previously
    duplicated inline in each. `blocked` is the first blocking disposition
    string, or None once every gate passes. `note` carries a non-blocking
    advisory (currently only the per-stock halt check's "skipped, known
    paper-mode gap" message) through to the final order result even when
    nothing blocked.
    국내 전용 게이트 순서 (양쪽 진입점에서 동일하게 반복되던 부분을 통합)"""
    if (blocked := await _circuit_breaker_block(krx_code)) is not None:
        return blocked, None
    blocked, note = await _stock_halt_block(broker, krx_code, ticker)
    if blocked is not None:
        return blocked, note
    if (blocked := await _vi_block(broker, krx_code, ticker)) is not None:
        return blocked, note
    if (blocked := await domestic_sell_over_held(broker, krx_code, side, shares)) is not None:
        return blocked, note
    if (over := await over_position_pct(broker, krx_code, side, shares, price)) is not None:
        return over, note
    return None, note


async def mirror_trade_to_kis(
    ticker: str,
    trade_date: date,
    side: str,
    shares: int,
    price: float,
    *,
    broker: KISBroker | None = None,
    approved: bool = False,
    assumed_current_shares: float | None = None,
    trade_id: int | None = None,
) -> str:
    """Mirror a just-recorded ledger trade to KIS — KRX market order or US
    NASDAQ limit order at the ledger's own price.

    `approved`: False for every normal automated call (the default) — an
    order that then needs approval (KIS_APPROVAL_*) is parked in
    app.execution.pending_orders instead of just being logged and dropped.
    True only when app.execution.pending_orders.approve_pending_order calls
    back in after a human approved it, bypassing the approval-threshold hold
    for this specific order (everything else — cash, VI, circuit-breaker,
    caps — still applies fresh, re-checked at this later point in time).

    `assumed_current_shares`, if given, is what the caller's sizing math
    believed was held before this trade (rebalance_to_target's
    `current_shares`, a stop-loss exit's `position.shares`) — re-verified
    against a fresh balance read right before submission (see
    `assumed_position_mismatch_block`). None (the default) skips this check
    for a caller that has no such assumption to verify.

    `trade_id`, if given, is the id of the `trades` row the caller already
    wrote for this same trade (ledger-first, mirror-second — see this
    function's own callers). Forwarded to `create_pending_order` when this
    call crosses the approval threshold below, so `_reject_or_expire` can
    tell a real, ledger-backed proposal apart from one with nothing to
    reverse — see app.execution.pending_orders's `trade_id` column comment.
    None (the default) is only safe for a caller whose own approval path is
    bypassed (`approved=True`, e.g. a stop-loss forced exit), since that
    skips the branch that would otherwise create an unlinked pending order.

    Returns a human-readable disposition ("skipped: …", "dry-run: …",
    "submitted: …", "FAILED: …", "PENDING_USER_APPROVAL: …") for the caller
    to log. Never raises.
    """
    # "skipped:" is reserved for the two cases where the ledger is
    # deliberately the sole source of truth and there is nothing to
    # reconcile against (KIS_EXECUTION_ENABLED=false below, and dry-run
    # further down) — see _mirror_never_reached_kis's own docstring in
    # app.execution.ledger. Every OTHER refusal below is permanent (this
    # exact call will never reach KIS, execution enabled or not) and must
    # return "BLOCKED:" so the caller's ledger entry actually gets reversed.
    # Confirmed live 2026-08-06: the trade-date check just below used to
    # return "skipped:", so a stop-loss forced-exit that failed it (see
    # app.execution.ledger.check_and_apply_stop_loss's own KST/ET fix) was
    # never reversed — the ledger recorded the position closed while KIS
    # still held it, and consumed the stop-loss row in the process, leaving
    # an unprotected real position with no further monitoring.
    krx_code, us_symbol = route_ticker(ticker)
    if krx_code is None and us_symbol is None:
        return f"BLOCKED: {ticker} is neither a KRX nor a US ticker"
    if not settings.kis_execution_enabled:
        return "skipped: KIS execution disabled (KIS_EXECUTION_ENABLED=false)"
    # local_today, not a single shared get_clock().today() (KST) — KST
    # midnight falls mid-session for NASDAQ, so an overseas decision made
    # right before KST midnight (trade_date stamped with local_today at
    # analysis time — see app.live.auto_trade_loop) would otherwise look
    # like a historical replay the instant this check runs after midnight,
    # and get its execution silently skipped even though the pipeline that
    # produced it finished moments ago in the same still-open session.
    if trade_date != local_today("domestic" if krx_code is not None else "overseas"):
        return f"BLOCKED: trade date {trade_date} is not today (historical replay?)"
    if ticker not in settings.trade_allowlist:
        return f"BLOCKED: {ticker} is not on the trade allowlist"
    if shares <= 0:
        return f"BLOCKED: non-positive share count {shares}"
    if price <= 0:
        # place_manual_order already validates this on its own exec_price
        # (see "a positive reference price is required..." above) — this
        # path had no equivalent. With price==0, over_value_cap/
        # requires_approval/insufficient_cash_block/over_position_pct all
        # compute a notional of 0 and pass trivially, so every value-based
        # gate silently no-ops instead of failing closed, and the order
        # still submits at the real market price (price isn't even passed
        # to submit_order — see its own call below).
        return f"BLOCKED: non-positive price {price}"
    if (blocked := await unsupported_product_block(ticker, side)) is not None:
        return blocked

    if (halted := await safety_switch.safety_gate_block(side, ticker)) is not None:
        return halted
    # side == "buy" only: this cap's own docstring already says the ledger's
    # sizing chain pre-shrinks an AUTO order to fit it — but that pre-shrink
    # (app.execution.ledger.rebalance_to_target's 1%-risk-rule/value-cap
    # blocks) is itself buy-only, since reducing exposure was never the
    # thing this cap exists to limit. Applying it here unconditionally meant
    # a SELL (a normal large rebalance, or a stop-loss forced exit) whose
    # notional happened to exceed the cap was permanently BLOCKED with no
    # partial-exit path — the same "sells always pass, only buys increase
    # exposure" exemption every other portfolio-wide risk gate in this
    # codebase already grants (see safety_switch.daily_loss_block's
    # identical side check). place_manual_order keeps the unconditional
    # check on both sides — a human's explicit oversized order is still
    # blocked and reported there, never silently let through.
    if side == "buy" and (capped := over_value_cap(krx_code, shares, price)) is not None:
        return capped
    if (needs := requires_approval(krx_code, shares, price, approved=approved)) is not None:
        # Previously just returned `needs` here, discarding it — the
        # ledger's own paper position was already recorded regardless (see
        # rebalance_to_target's "ledger record first, KIS mirror second"),
        # so a held order had no way for a human to ever see or act on it
        # again. Persisted instead so app.execution.pending_orders can
        # surface it for approve/reject. `approved` is always False when
        # this branch is reachable (True bypasses requires_approval above).
        from app.execution.pending_orders import count_pending, create_pending_order  # local: avoid a circular import at module load

        # Queue-length ceiling (2026-08-06), distinct from the TTL above —
        # the TTL bounds how long a row lingers, not how many can pile up
        # WITHIN one TTL window if several tickers all propose large orders
        # in the same cycle. Past this many, a human clearly isn't keeping
        # up with approvals, so a new proposal is BLOCKED here (reversing
        # the ledger's speculative entry like any other never-reached-KIS
        # order, via the same "BLOCKED:" disposition family every other
        # gate in this function already uses) rather than silently growing
        # an already-unmanaged queue further.
        if (pending_count := await count_pending()) >= settings.pending_order_max_count:
            return (
                f"BLOCKED: pending approval queue full ({pending_count}/{settings.pending_order_max_count}) — "
                "resolve existing pending orders before proposing new ones"
            )
        pending_id = await create_pending_order(
            ticker, trade_date, side, shares, price, reason=needs, trade_id=trade_id
        )
        return f"PENDING_USER_APPROVAL: {needs} (pending_order_id={pending_id})"
    order_desc = (
        f"KIS market {side} {krx_code} x{shares} (~{shares * price:,.0f} KRW)"
        if krx_code
        else f"KIS NASDAQ limit {side} {us_symbol} x{shares} @ {price:.2f} USD"
    )
    if settings.kis_execution_dry_run:
        # Plain ASCII punctuation only: dispositions get logged, and a cp949
        # Windows console chokes on fancier dashes.
        return f"dry-run: would submit {order_desc}; set KIS_EXECUTION_DRY_RUN=false to go live"

    # `get_broker("kis")` returns the one process-wide cached instance (see
    # its own docstring — every call site shares it so the request throttle
    # is actually shared too), never a private one this function could
    # legitimately close afterward. A caller-owned/close-on-exit variant of
    # this used to exist here (`own_broker`, closing whatever `get_broker`
    # returned in `finally`) — harmless back when `get_broker` constructed a
    # fresh instance per call, but became a live bug once broker instances
    # started being cached process-wide: a caller with no broker to pass
    # (auto_trade_loop's own stop-loss/rebalance path, which never supplies
    # one) would fetch the SHARED singleton, then close it when this
    # function returned — including on a path that just BLOCKED the order
    # and never touched the network — permanently breaking every other
    # concurrent user of that same shared broker (account/order/rank/etc.)
    # until the process restarted. Real incident: a blocked stop-loss sell
    # closed the shared client mid-session; every subsequent KIS call in the
    # process failed with "Cannot send a request, as the client has been
    # closed" from then on.
    if broker is None:
        broker = await get_broker("kis")
    if broker is None:
        return "FAILED: KIS credentials not configured"
    try:
        async with _order_lock:
            note = None
            if (
                stale := await assumed_position_mismatch_block(broker, krx_code, us_symbol, assumed_current_shares)
            ) is not None:
                return stale
            if krx_code:
                blocked, note = await _domestic_gates_block(broker, krx_code, ticker, side, shares, price)
                if blocked is not None:
                    return blocked
            if us_symbol:
                if (blocked := await overseas_sell_over_held(broker, us_symbol, side, shares)) is not None:
                    return blocked
            if (
                cash_blocked := await insufficient_cash_block(broker, krx_code, us_symbol, side, shares, price)
            ) is not None:
                return cash_blocked
            if krx_code:
                result = await broker.submit_order(krx_code, side, shares)
            else:
                result = await broker.submit_overseas_order(us_symbol, side, shares, price)
        suffix = f" [{note}]" if note else ""
        # Local import: app.execution.ledger imports THIS module (route_ticker,
        # mirror_trade_to_kis) at module load, so importing it back here at
        # module level would be circular — same pattern already used
        # elsewhere in this codebase for the identical shape of dependency
        # (e.g. app.execution.pending_orders above). Stores the order number
        # BEFORE returning, not folded into the disposition string alone —
        # app.live.fill_monitor needs this as an exact join key against
        # KIS's own execution-history TR, not something to regex back out of
        # human-readable text.
        from app.execution.ledger import _record_kis_order_no

        await _record_kis_order_no(trade_id, result.order_no)
        return f"submitted: {order_desc}; order_no={result.order_no} ({result.message}){suffix}"
    except Exception as exc:  # noqa: BLE001 - mirror must never crash the pipeline
        # ERROR, not warning (2026-08-19): this branch means the order never
        # reached KIS at all — on the stop-loss path that leaves a real
        # position sitting past its stop with nothing protecting it. msg_cd
        # (see const.is_permanent_refusal) tells a caller whether retrying
        # the identical payload is pointless — before this, AXTI's stop-loss
        # got resubmitted 17 times over 43 minutes because "FAILED: ..." gave
        # no way to distinguish a doomed malformed order (IGW00013) from a
        # transient hiccup worth retrying.
        msg_cd = getattr(exc, "msg_cd", "") or "none"
        kind = "permanent" if is_permanent_refusal(exc) else "transient"
        logger.error(
            "KIS mirror order FAILED for %s %s x%d — msg_cd=%s (%s): %s",
            side, ticker, shares, msg_cd, kind, exc, exc_info=True,
        )
        # The "[<msg_cd> <kind>]" marker is parsed back out by
        # ledger.failure_is_permanent — keep it right after "FAILED: " if
        # this format ever changes, and keep the "FAILED:" prefix itself
        # unchanged (ledger._mirror_never_reached_kis matches on it).
        return f"FAILED: [{msg_cd} {kind}] {exc}"


@dataclass
class ManualOrderResult:
    status: str  # "blocked" | "dry-run" | "submitted" | "failed"
    message: str
    ticker: str
    side: str
    shares: int
    order_no: str = ""
    org_no: str = ""  # domestic only — needed to cancel this order later
    price: float | None = None  # limit price, or the reference price for a market order

    @property
    def ok(self) -> bool:
        return self.status == "submitted"


def _describe_manual(
    krx_code: str | None, us_symbol: str | None, side: str, shares: int,
    order_type: str, exec_price: float,
) -> str:
    if krx_code:
        kind = "market" if order_type == "market" else f"limit @ {exec_price:,.0f}"
        return f"KIS {kind} {side} {krx_code} x{shares} (~{shares * exec_price:,.0f} KRW)"
    return f"KIS NASDAQ limit {side} {us_symbol} x{shares} @ {exec_price:.2f} USD"


async def place_manual_order(
    ticker: str,
    side: str,
    shares: int,
    *,
    order_type: str = "market",
    limit_price: float | None = None,
    reference_price: float | None = None,
    approved: bool = False,
    broker: KISBroker | None = None,
) -> ManualOrderResult:
    """User-initiated order from the dashboard order ticket.

    Goes through the SAME safety gates as `mirror_trade_to_kis` (route, kill
    switch/daily-loss breaker, master switch, value cap, approval threshold,
    circuit-breaker, per-stock halt, VI halt, cash sufficiency, domestic
    sell-ceiling, position-weight cap, dry-run), minus the
    mirror-only today/trade-allowlist gates — a manual order is by definition a
    deliberate here-and-now action on a ticker the user picked. Also runs
    `portfolio_caps_block`, which the automated path doesn't need here since
    its share count is already pre-shrunk by the same caps upstream.

    `order_type` is "market" or "limit"; overseas is limit-only on the paper
    server. `reference_price` (the current price shown in the UI) is used for
    the value cap on a market order and should be recorded to the ledger by
    the caller. Never raises — returns a ManualOrderResult with a status.

    Ledger recording is intentionally the caller's job (app/execution/ledger
    imports this module, so importing it back here would be circular) —
    mirrors the same "caller records, choke point only guards" split as
    `mirror_trade_to_kis`.
    """
    krx_code, us_symbol = route_ticker(ticker)

    def block(msg: str, price: float | None = None) -> ManualOrderResult:
        return ManualOrderResult("blocked", msg, ticker, side, shares, price=price)

    if krx_code is None and us_symbol is None:
        return block(f"{ticker} is neither a KRX nor a US ticker")
    if side not in ("buy", "sell"):
        return block(f"side must be 'buy' or 'sell', got {side!r}")
    if shares <= 0:
        return block(f"non-positive share count {shares}")
    if order_type not in ("market", "limit"):
        return block(f"order_type must be 'market' or 'limit', got {order_type!r}")
    if us_symbol and order_type == "market":
        return block("overseas paper orders are limit-only; choose a limit price")
    if order_type == "limit" and (limit_price is None or limit_price <= 0):
        return block("a limit order requires a positive limit price")
    if not settings.kis_execution_enabled:
        return block("KIS execution disabled (KIS_EXECUTION_ENABLED=false)")
    if (blocked := await unsupported_product_block(ticker, side)) is not None:
        return block(blocked)
    if (halted := await safety_switch.safety_gate_block(side, ticker)) is not None:
        return block(halted)

    # Price used for the value cap and (by the caller) the ledger record:
    # the limit price for a limit order, otherwise the UI's reference price.
    exec_price = limit_price if order_type == "limit" else reference_price
    if exec_price is None or exec_price <= 0:
        return block("a positive reference price is required for a market order")
    if (capped := over_value_cap(krx_code, shares, exec_price)) is not None:
        return block(capped, price=exec_price)
    if (needs := requires_approval(krx_code, shares, exec_price, approved)) is not None:
        return block(needs, price=exec_price)

    # Resolve a broker for the balance-backed risk gates (per-stock halt,
    # sell-ceiling, position-pct, cash sufficiency). These must be validated
    # even in dry-run, per this function's own docstring — so a broker is
    # created here whenever the caller didn't already supply one, regardless
    # of dry-run. Was previously gated on `not settings.kis_execution_dry_run`,
    # which meant any caller that reaches this function without a shared
    # broker (the dashboard always passes one, so this went unnoticed there)
    # got a "validated" dry-run result that had actually skipped every
    # broker-backed gate, with no indication of that in the returned message.
    # See mirror_trade_to_kis's own comment on this same `get_broker("kis")`
    # call — it's the process-wide shared singleton, never a private
    # instance this function could legitimately close afterward.
    if broker is None:
        broker = await get_broker("kis")
    try:
        # market_lock (2026-08-06): _order_lock just below is process-local
        # (asyncio.Lock) — it serializes this function's own gate-chain-
        # through-submission against ANOTHER call in the SAME process, but
        # does nothing against a second process (the auto path via
        # app.execution.ledger.rebalance_to_target, or another dashboard
        # instance) racing the same account. rebalance_to_target's own
        # market_lock (a Postgres advisory lock, cross-process) already
        # wraps its own balance-read-through-mirror sequence end to end —
        # this manual path had no equivalent, so a manual order submitted
        # while an auto decision for a different ticker in the same market
        # was mid-flight could see a pre-trade balance and jointly approve
        # exposure neither alone would (portfolio_caps_block's own docstring
        # already assumes this manual-vs-auto race is covered; it wasn't).
        async with market_lock(ticker_market(ticker)):
            async with _order_lock:
                note = None
                if broker is not None:
                    if krx_code:
                        blocked, note = await _domestic_gates_block(broker, krx_code, ticker, side, shares, exec_price)
                        if blocked is not None:
                            return block(blocked, price=exec_price)
                    if us_symbol:
                        overseas_blocked = await overseas_sell_over_held(broker, us_symbol, side, shares)
                        if overseas_blocked is not None:
                            return block(overseas_blocked, price=exec_price)
                    cash_blocked = await insufficient_cash_block(broker, krx_code, us_symbol, side, shares, exec_price)
                    if cash_blocked is not None:
                        return block(cash_blocked, price=exec_price)
                if (caps_blocked := await portfolio_caps_block(ticker, side, shares, exec_price)) is not None:
                    return block(caps_blocked, price=exec_price)
                suffix = f" [{note}]" if note else ""

                order_desc = _describe_manual(krx_code, us_symbol, side, shares, order_type, exec_price)
                if settings.kis_execution_dry_run:
                    return ManualOrderResult(
                        "dry-run", f"would submit {order_desc}{suffix}", ticker, side, shares, price=exec_price
                    )
                if broker is None:
                    return ManualOrderResult("failed", "KIS credentials not configured", ticker, side, shares)

                if krx_code:
                    price_arg = limit_price if order_type == "limit" else None
                    result = await broker.submit_order(krx_code, side, shares, price=price_arg)
                else:
                    result = await broker.submit_overseas_order(us_symbol, side, shares, limit_price)
                return ManualOrderResult(
                    "submitted",
                    f"{order_desc}; order_no={result.order_no} ({result.message}){suffix}",
                    ticker, side, shares,
                    order_no=result.order_no, org_no=result.org_no, price=exec_price,
                )
    except Exception as exc:  # noqa: BLE001 - never crash the request
        msg_cd = getattr(exc, "msg_cd", "") or "none"
        kind = "permanent" if is_permanent_refusal(exc) else "transient"
        logger.error(
            "manual order failed for %s %s x%d — msg_cd=%s (%s)", side, ticker, shares, msg_cd, kind, exc_info=True,
        )
        return ManualOrderResult("failed", f"[{msg_cd} {kind}] {exc}", ticker, side, shares)
