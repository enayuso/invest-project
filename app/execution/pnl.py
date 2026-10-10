# app/execution/pnl.py
# Realized P&L over an arbitrary date range — which ticker was bought at
# what price, sold at what price, and the resulting profit/loss. Nothing in
# this codebase computed this before (confirmed by investigation): /api/trades
# only lists individual fills with no P&L column, and /api/portfolio only
# shows UNREALIZED P&L for currently-held positions (a sold-off ticker
# disappears from that view entirely).
#
# Deliberately sourced from KIS's own execution history (fetch_daily_executions/
# fetch_overseas_executions — already used by app.execution.reconciliation), NOT
# the local `trades` ledger — that table has no column distinguishing a
# backtest-written row from a real live one (see project memory:
# live-backtest-past-context-leak-fixed), so mixing it into a P&L report risks
# quietly blending in trades that never really happened. KIS execution history
# can never contain backtest activity at all, since a backtest never calls the
# real broker.
#
# Known limitation (accepted): the domestic execution TR only exposes the last
# ~3 months (see KISBroker.fetch_daily_executions's own docstring) — a ticker
# bought earlier than that and sold inside the query window will show an
# average cost computed only from whatever buys happen to also fall inside the
# fetch window, not its true full history. No merge with the local ledger to
# work around this (see the backtest-contamination reason above) — the API
# layer surfaces this as `start_clamped` so the frontend can warn about it
# instead of silently presenting a number that looks exact but isn't.
# 실현손익(기간 내 매수/매도 짝지어 손익 계산) — KIS 체결내역 기준(로컬 원장 아님).
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

from app.config import Market
from app.domain.models import BrokerExecution

_LOOKBACK_LIMIT_DAYS = 90  # KIS's own ~3-month cap on the execution-history TRs


@dataclass
class RealizedTrade:
    ticker: str
    name: str
    market: Market
    sell_date: date
    shares: int
    buy_avg_price: float
    sell_price: float
    realized_pnl: float
    realized_pnl_pct: float
    currency: str
    # Buy-side (weighted-average, attributed to the matched shares) + sell-side
    # commission/tax for this closed lot. See BrokerExecution.fee_tax for what
    # this is sourced from and its "estimated" caveat.
    fee_tax: float = 0.0
    net_pnl: float = 0.0
    net_pnl_pct: float = 0.0
    # False for every overseas row (2026-09-08 fee/tax audit) — the overseas
    # execution TR (BrokerExecution.fee_tax's own docstring) never carries a
    # fee/tax figure at all, so `fee_tax`/`net_pnl` above are 0.0/==realized_pnl
    # not because this trade was free, but because it was never measured.
    # Domestic is always True: `fee_partial` (app.web.routers.portfolio) is
    # the separate, narrower "some domestic probe groups were skipped" signal
    # — this flag is the wider "no data source exists for this row at all."
    fee_measured: bool = True


@dataclass
class PnlSummary:
    # No unscoped total_pnl/total_fee_tax/total_net_pnl (removed 2026-08-06)
    # — domestic trades are KRW, overseas are USD, and summing raw floats
    # across both currencies produced a number that isn't KRW, isn't USD,
    # and isn't meaningful in either — a mixed-currency figure with no
    # unit. The `market` query param on /api/pnl also only ever filtered
    # the `rows` list, never these unscoped totals, so requesting
    # market=domestic still returned a "total" contaminated with overseas
    # figures. Every field below is scoped to one market/currency, which
    # was already correct regardless of that filter (computed by checking
    # trade.market per-trade, not from the query param).
    total_pnl_domestic: float = 0.0
    total_pnl_overseas: float = 0.0
    total_fee_tax_domestic: float = 0.0
    total_fee_tax_overseas: float = 0.0
    total_net_pnl_domestic: float = 0.0
    total_net_pnl_overseas: float = 0.0
    win_count: int = 0
    loss_count: int = 0
    # True when this result includes at least one overseas row — meaning
    # total_fee_tax_overseas/total_net_pnl_overseas are 0.0/==gross by data
    # absence, not by measurement, and win_count/loss_count below counted
    # every overseas trade by its GROSS realized_pnl (a trade that was
    # actually a net loser after real, unrecorded commission can still be
    # counted here as a win). See RealizedTrade.fee_measured's own docstring.
    fee_unavailable_overseas: bool = False

    @property
    def win_rate(self) -> float | None:
        total = self.win_count + self.loss_count
        return self.win_count / total if total else None


def earliest_fetchable_date(today: date) -> date:
    """The oldest date KIS's execution-history TRs can return data for,
    from `today` — used to widen the actual fetch window past the caller's
    own `start` so a position opened before `start` still gets as accurate
    an average cost as the TR's own cap allows."""
    return today - timedelta(days=_LOOKBACK_LIMIT_DAYS)


def _order_no_sort_key(order_no: str | None) -> int:
    """Same-day tie-breaker for `_match_realized_trades`'s fill ordering —
    `order_date` alone doesn't distinguish same-day fills, and `order_no`
    increments over the trading day, so sorting by it approximates fill
    order. Falls back to 0 (order_date alone still governs) on any
    non-numeric value instead of raising — `int(e.order_no or 0)` used to
    crash this whole function (and every caller of it, i.e. the entire
    /api/pnl response for every user) on a single execution with a
    non-numeric order_no."""
    try:
        return int(order_no or 0)
    except (TypeError, ValueError):
        return 0


def _match_realized_trades(
    executions: list[BrokerExecution], market: Market, currency: str, start: date, end: date,
) -> list[tuple[str, RealizedTrade | None]]:
    """Pure matching logic (no I/O) — weighted-average cost per ticker, same
    convention KIS's own `avg_price` (평단가) already uses for a currently-held
    position (see app.brokers.kis.domestic/overseas.account), so a realized
    number here means the same thing as the unrealized one shown elsewhere in
    this dashboard. Returns (ticker, RealizedTrade | None) pairs in
    chronological order — None entries are buy fills (nothing realized yet),
    kept in the return shape only so callers needing a raw trace can see them;
    `compute_realized_pnl` filters them out.

    Processes ALL given executions (expected to span from the earliest
    fetchable date through `end`, not just [start, end]) so a buy before
    `start` still updates the running average cost — only sells landing
    inside [start, end] are actually reported."""
    by_ticker: dict[str, list[BrokerExecution]] = {}
    for execution in executions:
        if execution.filled_qty <= 0:
            continue  # never filled (cancelled/rejected/still open) — nothing realized
        by_ticker.setdefault(execution.code, []).append(execution)

    results: list[tuple[str, RealizedTrade | None]] = []
    for code, fills in by_ticker.items():
        fills.sort(key=lambda e: (e.order_date, _order_no_sort_key(e.order_no)))
        avg_cost = 0.0
        avg_buy_fee = 0.0  # weighted-average buy-side fee/tax per share, same convention as avg_cost
        shares_held = 0
        ticker = f"{code}.KS" if market == "domestic" else code
        for fill in fills:
            if fill.side == "buy":
                new_shares = shares_held + fill.filled_qty
                avg_cost = (avg_cost * shares_held + fill.avg_price * fill.filled_qty) / new_shares
                buy_fee_per_share = fill.fee_tax / fill.filled_qty if fill.filled_qty else 0.0
                avg_buy_fee = (avg_buy_fee * shares_held + buy_fee_per_share * fill.filled_qty) / new_shares
                shares_held = new_shares
                results.append((ticker, None))
            else:
                matched_qty = min(fill.filled_qty, shares_held)
                shares_held -= matched_qty
                if matched_qty <= 0:
                    continue  # a sell with nothing on record to have sold (pre-window position)
                sell_date = date(int(fill.order_date[:4]), int(fill.order_date[4:6]), int(fill.order_date[6:]))
                if not (start <= sell_date <= end):
                    continue
                realized_pnl = (fill.avg_price - avg_cost) * matched_qty
                # Sell-side fee/tax (commission + 증권거래세) is levied on the whole
                # sell fill; attribute the matched-qty's share of it proportionally
                # when a fill only partially matches against shares_held.
                sell_fee_matched = (
                    fill.fee_tax * (matched_qty / fill.filled_qty) if fill.filled_qty else 0.0
                )
                buy_fee_matched = avg_buy_fee * matched_qty
                fee_tax = buy_fee_matched + sell_fee_matched
                net_pnl = realized_pnl - fee_tax
                results.append((
                    ticker,
                    RealizedTrade(
                        ticker=ticker,
                        name=fill.name,
                        market=market,
                        sell_date=sell_date,
                        shares=matched_qty,
                        buy_avg_price=avg_cost,
                        sell_price=fill.avg_price,
                        realized_pnl=realized_pnl,
                        realized_pnl_pct=(realized_pnl / (avg_cost * matched_qty)) if avg_cost else 0.0,
                        currency=currency,
                        fee_tax=fee_tax,
                        net_pnl=net_pnl,
                        net_pnl_pct=(net_pnl / (avg_cost * matched_qty)) if avg_cost else 0.0,
                        fee_measured=market == "domestic",
                    ),
                ))
    return results


async def compute_realized_pnl(
    start: date,
    end: date,
    *,
    domestic_executions: list[BrokerExecution],
    overseas_executions: list[BrokerExecution],
) -> tuple[list[RealizedTrade], PnlSummary]:
    """Entry point: caller fetches the KIS execution history (typically via
    state._kis_broker.fetch_daily_executions/fetch_overseas_executions, from
    earliest_fetchable_date(today) through `end`) and passes it in here —
    keeps this module broker-agnostic and independently testable, same
    reasoning as app.execution.reconciliation.find_unmirrored_trades."""
    domestic_matches = _match_realized_trades(domestic_executions, "domestic", "KRW", start, end)
    overseas_matches = _match_realized_trades(overseas_executions, "overseas", "USD", start, end)

    trades = [trade for _, trade in (*domestic_matches, *overseas_matches) if trade is not None]
    trades.sort(key=lambda t: t.sell_date)

    summary = PnlSummary()
    for trade in trades:
        if trade.market == "domestic":
            summary.total_pnl_domestic += trade.realized_pnl
            summary.total_fee_tax_domestic += trade.fee_tax
            summary.total_net_pnl_domestic += trade.net_pnl
        else:
            summary.total_pnl_overseas += trade.realized_pnl
            summary.total_fee_tax_overseas += trade.fee_tax
            summary.total_net_pnl_overseas += trade.net_pnl
            summary.fee_unavailable_overseas = True
        # win/loss reflects the actual economic outcome (after fee/tax), not
        # the pre-cost gross difference — for overseas rows this is really
        # gross (fee_measured=False, fee_tax always 0.0), so a marginal
        # overseas trade can be counted as a win despite unrecorded costs.
        if trade.net_pnl > 0:
            summary.win_count += 1
        elif trade.net_pnl < 0:
            summary.loss_count += 1

    return trades, summary
