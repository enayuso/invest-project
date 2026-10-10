"""Faber-style trend signal, per market — proactively widens the portfolio
cash floor (app.execution.portfolio_risk) when the broad market itself looks
like it's rolling over, rather than only reacting once THIS account's own
drawdown breaker has already tripped.

Faber, "A Quantitative Approach to Tactical Asset Allocation" (2007): price
below its own trailing long moving average -> Risk-Off, at/above -> Risk-On.
Domestic uses KOSPI (지수코드 0001, via a dedicated KIS index-chart TR);
overseas uses SPY as the S&P 500 proxy (via app.dataflows.kis_provider's
existing KIS-first/yfinance-fallback `get_price_series` — no separate broker
integration needed, unlike KOSPI, since a US index-chart TR doesn't exist in
this codebase and yfinance already covers SPY's full history for free).
KOSDAQ is not covered separately (KOSPI is Faber's own broad-index choice,
and this project's domestic cash floor is applied to the whole domestic book
at once, not split by exchange).

LIVE-only, same reasoning as portfolio_risk.py's own portfolio_drawdown_state
table: a backtest replaying a past date must never read today's actual
market regime (that would leak present-day information into a historical
decision) — only app.execution.portfolio_risk.apply_live_portfolio_caps
(never the pure apply_portfolio_caps, and never the backtest harness) calls
into this module.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from datetime import date, timedelta
from typing import Literal

import asyncpg

from app.clock import local_today
from app.config import settings
from app.execution import _schema_bootstrap
from app.memory.db import get_pool
from app.execution import risk_rules

logger = logging.getLogger(__name__)

Market = Literal["domestic", "overseas"]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS market_regime_state (
    market       TEXT PRIMARY KEY CHECK (market IN ('domestic', 'overseas')),
    is_risk_off  BOOLEAN NOT NULL,
    index_close  DOUBLE PRECISION NOT NULL,
    sma_value    DOUBLE PRECISION NOT NULL,
    computed_on  DATE NOT NULL,
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""

_schema_ready = False
_schema_lock = asyncio.Lock()


async def _ready_pool() -> asyncpg.Pool:
    """Double-checked locking around the first-time DDL — shared with the
    other app.execution modules via _schema_bootstrap."""
    return await _schema_bootstrap.ready_pool(sys.modules[__name__], _SCHEMA)


def _faber_signal(closes_by_day: list[tuple[str, float]], window: int) -> tuple[bool, float, float] | None:
    """Pure — no I/O. `closes_by_day`: (day string, close) pairs, any order
    (neither KIS nor get_price_series guarantees one). Returns (is_risk_off,
    latest_close, sma_value), or None if fewer than `window` bars are
    available (can't compute a meaningful SMA yet — e.g. a data-source
    outage truncated the history)."""
    if len(closes_by_day) < window:
        return None
    ordered = sorted(closes_by_day, key=lambda pair: pair[0])
    closes = [c for _, c in ordered if c > 0]
    if len(closes) < window:
        return None
    sma = sum(closes[-window:]) / window
    latest = closes[-1]
    return latest < sma, latest, sma


async def _already_computed_today(market: Market, today: date) -> bool:
    try:
        pool = await _ready_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT computed_on FROM market_regime_state WHERE market = $1", market
            )
        return bool(row and row["computed_on"] == today)
    except Exception:
        logger.warning("market_regime: could not check existing %s state, proceeding to recompute", market, exc_info=True)
        return False


async def _persist_signal(market: Market, is_risk_off: bool, latest_close: float, sma_value: float, today: date) -> None:
    try:
        pool = await _ready_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO market_regime_state (market, is_risk_off, index_close, sma_value, computed_on, updated_at)
                VALUES ($1, $2, $3, $4, $5, now())
                ON CONFLICT (market) DO UPDATE
                SET is_risk_off = $2, index_close = $3, sma_value = $4, computed_on = $5, updated_at = now()
                """,
                market, is_risk_off, latest_close, sma_value, today,
            )
        logger.info(
            "market_regime: %s regime refreshed — %s (%.2f vs %d-day SMA %.2f)",
            market, "Risk-Off" if is_risk_off else "Risk-On", latest_close, risk_rules.rule("faber_sma_window_days"), sma_value,
        )
    except Exception:
        logger.warning("market_regime: could not persist refreshed %s state", market, exc_info=True)


async def _cash_floor_pct(market: Market) -> float:
    """Shared reader behind `domestic_cash_floor_pct`/`overseas_cash_floor_pct`
    below — degrades to the Risk-On floor on any read failure or if no
    signal has been computed yet for `market`: this is a PROACTIVE tightening
    layered on top of the existing static floor, so "can't tell" must never
    be treated as more permissive than the static default already was — but
    it also must never block trading outright on its own (that reactive job
    belongs to portfolio_drawdown_breaker_pct instead)."""
    try:
        pool = await _ready_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT is_risk_off FROM market_regime_state WHERE market = $1", market
            )
    except Exception:
        logger.warning("market_regime: could not read %s regime state", market, exc_info=True)
        return risk_rules.rule("portfolio_cash_floor_pct")
    if row and row["is_risk_off"]:
        return risk_rules.rule("portfolio_cash_floor_pct_risk_off")
    return risk_rules.rule("portfolio_cash_floor_pct")


async def refresh_domestic_regime(force: bool = False) -> None:
    """Computes and persists today's Faber signal for KOSPI — idempotent per
    calendar day (a no-op if already computed today, unless `force=True`),
    so it's safe to call on every auto_trade_loop cycle the same way
    app.live.krx_master.ensure_loaded() is: cheap after the first successful
    call each day. Entirely best-effort — logs and returns quietly on any
    failure (broker not connected, KIS call failed, not enough history yet).
    This is a risk-reducing enhancement layered on TOP OF the existing
    static cash floor (app.execution.portfolio_risk), never a replacement
    that could itself become a single point of failure blocking trading."""
    from app.web import state  # local import — avoid circular import, see app.execution.portfolio_risk's own note

    # local_today("domestic"), not a shared KST-based get_clock().today() —
    # this module previously used the latter for both markets, which for
    # domestic happens to be the same KST day anyway (KRX's own timezone),
    # but keeping both refresh functions consistent avoids re-introducing
    # the overseas mismatch (see refresh_overseas_regime below) by copy-paste.
    today = local_today("domestic")
    if not force and await _already_computed_today("domestic", today):
        return

    broker = await state.kis_broker()

    if broker is None:
        logger.info("market_regime: KIS broker not connected — skipping domestic Faber signal refresh")
        return

    window = risk_rules.rule("faber_sma_window_days")
    # ~1.6x calendar-to-trading-day ratio covers weekends/holidays with room
    # to spare; +30 extra days of slack for a holiday-heavy stretch.
    start = today - timedelta(days=int(window * 1.6) + 30)
    # end = today - 1 day: today's own bar may still be forming (this can run
    # mid-session, every auto_trade_loop cycle) — always exclude it rather
    # than risk treating a partial-day close as final, which would then
    # stick for the rest of the day via _already_computed_today's idempotency.
    end = today - timedelta(days=1)
    try:
        bars = await broker.fetch_index_daily_chart(
            "0001", start.strftime("%Y%m%d"), end.strftime("%Y%m%d")
        )
    except Exception:
        logger.warning("market_regime: fetch_index_daily_chart failed, keeping prior domestic state", exc_info=True)
        return

    signal = _faber_signal([(bar.day, bar.close) for bar in bars], window)
    if signal is None:
        logger.warning(
            "market_regime: only %d KOSPI bars available, need %d — skipping domestic refresh", len(bars), window
        )
        return
    is_risk_off, latest_close, sma_value = signal
    await _persist_signal("domestic", is_risk_off, latest_close, sma_value, today)


async def refresh_overseas_regime(force: bool = False) -> None:
    """Overseas counterpart to `refresh_domestic_regime` above — same
    idempotent-per-day/best-effort contract, but sourced from SPY's daily
    close series via app.dataflows.kis_provider.get_price_series (KIS-first,
    yfinance-fallback) rather than a dedicated broker method: no US
    index-chart TR exists in this codebase, and yfinance already gives SPY's
    full history for free, so a second KIS integration isn't worth building
    for this. No broker/state dependency at all — safe to call even before
    a KIS session exists."""
    # local_today("overseas") (US Eastern), not get_clock().today() (KST) —
    # SPY's session runs through KST midnight, so the shared KST clock can
    # roll to a new calendar day mid-session and make `_already_computed_today`
    # falsely treat a stale/partial signal as "already done for today" for
    # what is, in New York, still the same trading day. Same fix already
    # applied to ledger.py/kis_bridge.py's stop-loss and mirror-order paths.
    today = local_today("overseas")
    if not force and await _already_computed_today("overseas", today):
        return

    window = risk_rules.rule("faber_sma_window_days")
    start = today - timedelta(days=int(window * 1.6) + 30)
    # end = today - 1 day: exclude today's still-forming SPY bar, same reason
    # as refresh_domestic_regime's identical guard above.
    end = today - timedelta(days=1)
    try:
        from app.dataflows.kis_provider import get_price_series

        series = await get_price_series("SPY", start, end)
    except Exception:
        logger.warning("market_regime: SPY get_price_series failed, keeping prior overseas state", exc_info=True)
        return

    signal = _faber_signal([(day.isoformat(), close) for day, close in series.items()], window)
    if signal is None:
        logger.warning(
            "market_regime: only %d SPY bars available, need %d — skipping overseas refresh", len(series), window
        )
        return
    is_risk_off, latest_close, sma_value = signal
    await _persist_signal("overseas", is_risk_off, latest_close, sma_value, today)


async def domestic_cash_floor_pct() -> float:
    """Effective cash floor for the domestic market right now — see
    `_cash_floor_pct`'s own docstring for the degrade-on-failure contract."""
    return await _cash_floor_pct("domestic")


async def overseas_cash_floor_pct() -> float:
    """Effective cash floor for the overseas market right now — see
    `_cash_floor_pct`'s own docstring for the degrade-on-failure contract."""
    return await _cash_floor_pct("overseas")
