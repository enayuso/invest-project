"""Portfolio endpoint — real KIS account state per trade allowlist ticker."""

from __future__ import annotations

import asyncio
import logging
from datetime import date, timedelta

from fastapi import APIRouter, Depends, HTTPException, Query

from app.clock import get_clock
from app.config import settings
from app.execution.ledger import get_open_stop_loss
from app.execution.pnl import compute_realized_pnl, earliest_fetchable_date
from app.live.store import latest
from app.web.account_mode_middleware import require_kis
from app.web import state

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



@router.get("/api/portfolio", dependencies=_needs_kis)
async def portfolio() -> dict:
    """Real KIS-account state per trade allowlist ticker, plus a live price for
    tickers KIS doesn't currently hold. `account` carries the KIS balance
    summary (null when unconfigured/unreachable).

    Only reflects the actual KIS (paper or real) account — this used to
    fall back to the local paper ledger's own shares/cash for a ticker KIS
    didn't hold, indistinguishable in the UI from a real position. The
    ledger records every AI rebalance regardless of whether it ever
    actually reached KIS (see `ledger.rebalance_to_target`'s "ledger
    record first, KIS mirror second" ordering), so that fallback showed
    phantom holdings (e.g. NVDA reading 58 shares nobody actually held at
    KIS) and a "cash" figure with no real per-ticker KIS equivalent
    (KIS's own cash balance is account-level, already on the account
    summary card, not per ticker). Dropped per explicit request — a
    ticker not in `kis_positions` now shows 0 shares / no venue rather
    than the ledger's own bookkeeping.

    Rows cover `settings.trade_allowlist` PLUS any ticker `kis_positions` reports
    real shares for but that isn't (or no longer is) on the trade allowlist —
    without this, a real holding the trade allowlist doesn't know about (e.g. it
    fell off on a server restart — trade allowlist isn't DB-persisted — or was
    never auto-added because it's a SELL-only ticker, never a BUY) was
    invisible here even though KIS itself still reports the shares."""
    account, kis_positions = await state.kis_account_snapshot()
    tickers = list(settings.trade_allowlist)
    tickers.extend(t for t in kis_positions if t not in tickers)
    # Only for tickers KIS actually reports shares held — a flat ticker's
    # most recent ledger row (if any) is stale/irrelevant here (see
    # get_open_stop_loss's own "most recent trade" semantics), and querying
    # it for the whole trade allowlist on every 30s poll would be pure waste.
    held_tickers = [t for t in tickers if kis_positions.get(t)]
    stop_loss_results = await asyncio.gather(*(get_open_stop_loss(t) for t in held_tickers))
    stop_losses = {t: r[1] if r else None for t, r in zip(held_tickers, stop_loss_results)}
    rows = []
    for ticker in tickers:
        tick = latest(ticker)
        current_price = tick.price if tick else None
        # get_display_name lazily fetches-and-caches into _display_info on
        # a miss (see its own docstring) — must run BEFORE reading
        # _display_info here, not after, or a ticker whose cache entry
        # was never populated at startup (e.g. a transient fetch failure —
        # get_display_name degrades to returning the bare ticker rather
        # than raising) reads its OWN just-fetched entry as still missing,
        # showing a blank currency forever even though the cache got
        # filled one line too late to matter for this same response.
        name = await state.get_display_name(ticker)
        info = state._display_info.get(ticker)
        kis_pos = kis_positions.get(ticker)
        rows.append(
            {
                "ticker": ticker,
                # Always the same name source as every other panel (never
                # a KIS position's own Korean-only name field) — see
                # state.get_display_name.
                "name": name,
                "currency": info.currency if info else None,
                "cash": None,
                "shares": kis_pos["quantity"] if kis_pos else 0,
                "current_price": kis_pos["current_price"] if kis_pos else current_price,
                "market_value": kis_pos["eval_amount"] if kis_pos else None,
                "venue": "kis" if kis_pos else None,
                "avg_price": kis_pos["avg_price"] if kis_pos else None,
                "pnl_amount": kis_pos["pnl_amount"] if kis_pos else None,
                "pnl_pct": kis_pos["pnl_pct"] if kis_pos else None,
                "eval_amount": kis_pos["eval_amount"] if kis_pos else None,
                "stop_loss": stop_losses.get(ticker),
            }
        )
    return {"account": account, "rows": rows}


@router.get("/api/trades", dependencies=_needs_kis)
async def trades(limit: int = Query(default=20, ge=0)) -> list[dict]:
    """Recent executions, newest first: the KIS paper account's actual
    orders from the last week (venue "kis", status filled/partial/pending
    from the fill counts). KIS unavailability leaves this empty.

    The local paper-ledger's own rows (venue "paper") used to be merged in
    here too, but that ledger records every AI rebalance regardless of
    whether it ever actually reached KIS (see rebalance_to_target's
    "ledger record first, KIS mirror second" ordering) — next to real KIS
    fills, that read as confusing noise (a "trade" the user never actually
    placed or saw execute). Dropped per explicit request rather than
    reconciling the two views."""
    rows = []
    if await state.kis_broker() is not None:
        domestic, overseas_execs = await state.recent_executions()  # cached, shared with open-orders
        overseas_codes = {e.order_no for e in overseas_execs}
        executions = [*domestic, *overseas_execs]
        # order_date (KIS's ord_dt) is day-only, no time-of-day field exists
        # on this TR — order_no is KIS's own sequential order number, the
        # only thing that breaks ties between same-day trades (domestic and
        # overseas were also just concatenated as two separate blocks, so
        # without this same-day trades wouldn't interleave by recency at
        # all). Numeric, not lexicographic, since order_no's digit count can
        # differ.
        executions.sort(key=lambda e: (e.order_date, int(e.order_no or 0)), reverse=True)
        for e in executions:
            if e.filled_qty >= e.order_qty and e.order_qty > 0:
                status = "filled"
            elif e.filled_qty > 0:
                status = "partial"
            else:
                status = "pending"
            is_overseas = e.order_no in overseas_codes
            ticker = e.code if is_overseas else f"{e.code}.KS"
            rows.append(
                {
                    "ticker": ticker,
                    # Same unified name source as the paper-ledger rows
                    # above, not the KIS execution's own (Korean-only)
                    # name field.
                    "name": await state.get_display_name(ticker),
                    "currency": "USD" if is_overseas else "KRW",
                    "trade_date": f"{e.order_date[:4]}-{e.order_date[4:6]}-{e.order_date[6:]}",
                    "side": e.side,
                    "shares": e.order_qty,
                    "price": e.avg_price,
                    "stop_loss": None,
                    "reason": f"KIS 주문번호 {e.order_no} (체결 {e.filled_qty}/{e.order_qty})",
                    "venue": "kis",
                    "status": status,
                    # KIS's own execution record, not tied to any local
                    # decision row.
                    "decision_id": None,
                }
            )

    return rows[:limit]


@router.get("/api/pnl", dependencies=_needs_kis)
async def pnl(start: str | None = None, end: str | None = None, market: str = "all") -> dict:
    """Realized profit/loss over a date range — which ticker was bought at
    what (weighted-average) price, sold at what price, and the resulting
    gain/loss. See app.execution.pnl's own module docstring for why this is
    sourced from KIS's real execution history rather than the local `trades`
    ledger, and for the ~3-month lookback cap on that history.

    `start`/`end` are ISO dates (`YYYY-MM-DD`); default to a 30-day window
    ending today. A `start` older than the TR's own cap is silently clamped
    (never a 400 — a slightly-off date range is still useful) and reported
    back via `start_clamped` so the frontend can warn instead of presenting
    a number that looks exact but isn't."""
    today = get_clock().today()
    try:
        end_date = date.fromisoformat(end) if end else today
        requested_start = date.fromisoformat(start) if start else end_date - timedelta(days=30)
    except ValueError as exc:
        # date.fromisoformat raises ValueError on a malformed string (e.g.
        # ?start=not-a-date) — unguarded, that propagated as an unhandled 500
        # instead of the 422 a bad query param should get.
        raise HTTPException(422, f"start/end must be ISO dates (YYYY-MM-DD): {exc}") from exc
    floor = earliest_fetchable_date(today)
    query_start = max(requested_start, floor)
    start_clamped = requested_start < floor

    domestic_executions, overseas_executions = [], []
    broker = await state.kis_broker()
    if broker is not None:
        # Always fetch from `floor` (not `query_start`) through `end_date` —
        # a ticker's cost basis needs every buy the TR can still see, not
        # just the ones after the caller's own start date.
        fetch_start, fetch_end = floor.strftime("%Y%m%d"), end_date.strftime("%Y%m%d")
        try:
            domestic_executions = await broker.fetch_daily_executions(fetch_start, fetch_end)
        except Exception:
            logger.warning("KIS domestic executions failed for /api/pnl", exc_info=True)
        try:
            overseas_executions = await broker.fetch_overseas_executions(fetch_start, fetch_end)
        except Exception:
            logger.warning("KIS overseas executions failed for /api/pnl", exc_info=True)

    # Fee/tax enrichment (2026-09-07 — root cause: app.brokers.kis.domestic.
    # order.fetch_daily_executions's own execution rows have no fee field at
    # all, see that method's own comment). Only the tickers that actually
    # produced a REPORTED sell (inside [query_start, end_date], not the wider
    # fetch window used for cost-basis) are worth the extra probe calls — a
    # ticker only bought, never sold in this window, never shows a row here.
    # `getattr` guard: broker stubs in tests (and any future non-KIS broker)
    # that don't implement this optional enrichment simply skip it, same
    # fail-open posture as the two fetches above.
    fee_partial = False
    probe = getattr(broker, "fetch_domestic_fee_tax", None) if broker is not None else None
    if probe is not None and domestic_executions:
        sold_codes = {
            e.code
            for e in domestic_executions
            if e.side == "sell" and e.filled_qty > 0 and _yyyymmdd_in_range(e.order_date, query_start, end_date)
        }
        if sold_codes:
            try:
                unresolved = await probe(
                    domestic_executions, codes=sold_codes,
                    max_groups=settings.kis_pnl_fee_probe_max_groups,
                    today=get_clock().today().strftime("%Y%m%d"),
                )
                fee_partial = unresolved > 0
            except Exception:
                logger.warning("domestic fee/tax enrichment failed for /api/pnl", exc_info=True)
                fee_partial = True

    trades, summary = await compute_realized_pnl(
        query_start, end_date, domestic_executions=domestic_executions, overseas_executions=overseas_executions,
    )
    if market in ("domestic", "overseas"):
        trades = [t for t in trades if t.market == market]

    return {
        "rows": [
            {
                "ticker": t.ticker,
                # Same unified name source as /api/trades above, not KIS's
                # own (Korean-only) execution name field.
                "name": await state.get_display_name(t.ticker),
                "market": t.market,
                "sell_date": t.sell_date.isoformat(),
                "shares": t.shares,
                "buy_avg_price": t.buy_avg_price,
                "sell_price": t.sell_price,
                "realized_pnl": t.realized_pnl,
                "realized_pnl_pct": t.realized_pnl_pct,
                "fee_tax": t.fee_tax,
                "net_pnl": t.net_pnl,
                "net_pnl_pct": t.net_pnl_pct,
                "fee_measured": t.fee_measured,
                "currency": t.currency,
            }
            for t in trades
        ],
        "summary": {
            # No unscoped total_pnl/total_fee_tax/total_net_pnl (removed
            # 2026-08-06) — domestic is KRW, overseas is USD, and summing
            # them produced a mixed-currency figure with no real unit. See
            # PnlSummary's own docstring.
            "total_pnl_domestic": summary.total_pnl_domestic,
            "total_pnl_overseas": summary.total_pnl_overseas,
            "total_fee_tax_domestic": summary.total_fee_tax_domestic,
            "total_fee_tax_overseas": summary.total_fee_tax_overseas,
            "total_net_pnl_domestic": summary.total_net_pnl_domestic,
            "total_net_pnl_overseas": summary.total_net_pnl_overseas,
            "win_count": summary.win_count,
            "loss_count": summary.loss_count,
            "win_rate": summary.win_rate,
        },
        "start": query_start.isoformat(),
        "end": end_date.isoformat(),
        "start_clamped": start_clamped,
        # True only when at least one domestic fee/tax probe group (see
        # app.brokers.kis.domestic.order.fetch_domestic_fee_tax) was skipped
        # past the max-groups cap, rejected as implausible, or the whole
        # enrichment step raised — the corresponding rows' fee_tax/net_pnl
        # may understate the real cost. Same "best-effort, say so when it
        # wasn't complete" pattern as start_clamped above, not a new banner
        # style: absent/false in the normal case.
        "fee_partial": fee_partial,
        # True when this result includes any overseas row (2026-09-08 fee/tax
        # audit) — overseas fee_tax/net_pnl are always 0.0/==gross because no
        # KIS TR used here exposes an overseas commission figure at all, not
        # because those trades were free. Distinct from fee_partial above
        # (which means "some domestic data was skipped") — this means "no
        # overseas data source exists yet." See PnlSummary.fee_unavailable_
        # overseas's own docstring.
        "fee_unavailable_overseas": summary.fee_unavailable_overseas,
    }


def _yyyymmdd_in_range(yyyymmdd: str, start: date, end: date) -> bool:
    """`BrokerExecution.order_date` ("YYYYMMDD", KIS's own format) parsed and
    range-checked against the report's requested (not fetch) window. Returns
    False (never raises) on a malformed/empty date — the same fail-safe
    posture as `_order_no_sort_key` in app.execution.pnl: a ticker whose date
    can't be parsed is simply not treated as a reported sell, never crashes
    the whole /api/pnl response over one bad field."""
    try:
        d = date(int(yyyymmdd[:4]), int(yyyymmdd[4:6]), int(yyyymmdd[6:]))
    except (TypeError, ValueError, IndexError):
        return False
    return start <= d <= end
