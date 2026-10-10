"""Tick-based real-time stop-loss monitor.

The missing piece between the minute poller (`app.live.poller`, records
ticks into `app.live.store`) and the once-an-hour daily-bar stop-loss loop
(`app.web.state.stop_loss_loop`). That loop reads `get_daily_low_series` —
for a domestic (KRX) ticker, that day's own candle only finalizes once the
day closes, so a price crashing through a stop mid-session goes completely
unprotected until the NEXT hourly pass reads that day's already-closed bar.
For an overseas ticker there's no KRX websocket at all, so its "real-time"
price is already only as fresh as the 60s REST poller — still a large
improvement over once an hour.

This loop closes that gap on a much shorter interval
(`settings.exit_monitor_interval_seconds`, default 5s), reading whichever
source is actually freshest for each market via
`app.live.last_price.freshest_tick` (domestic websocket feed / overseas
poller cache — see that module's own docstring for why the two differ and
why neither is a network fetch). Zero extra API cost either way, since
both are already-maintained caches.

Fail-closed toward NOT selling: a missing or stale tick means the current
price is unknown, and forcing an exit off out-of-date information is worse
than a late-but-correct exit from the hourly backstop, which keeps running
unchanged as a safety net under this loop. Same posture as
`RealtimeFeed.current_status`'s reconnect-staleness gate (app.brokers.kis.
domestic.websocket) — old data must read as "unknown," never as "safe" or
"definitely breached."

Overseas execution caveat: KIS has no immediate-fill market order for
overseas equities (paper: limit-only both sides; real: sell-side MOO/MOC
only, neither of which fills immediately) — see `force_exit_position` /
`mirror_trade_to_kis`. A forced exit here still submits the same order
`check_and_apply_stop_loss` already does; an overseas fill can still lag or
fail to land, which `app.execution.reconciliation` already watches for
(CRITICAL "UNPROTECTED") independently of this loop.

Trailing stop (2026-08-10): every pass also feeds the live price into
`app.execution.trailing_stop.update_high_water_mark`, then checks the
current tick against `effective_stop_loss` (max of the PM's own stop and
the trailing floor) rather than the plain stop-loss — see that module's
own docstring for the ratchet-only safety argument. ATR (needed to size
the trailing distance) is cached at `_ATR_CACHE_TTL_SECONDS` — it's a
14-day rolling figure that does not meaningfully change within an hour,
and `update_high_water_mark` only actually USES a freshly-computed value
when a holding streak resets anyway (see that function's own docstring),
so fetching it fresh every 5s would be pure waste, not added accuracy.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import timezone

from app.clock import get_clock
from app.config import account_mode_scope, settings
from app.dataflows.kis_provider import get_atr_pct
from app.execution import trailing_stop
from app.execution.ledger import ForcedExitOutcome, force_exit_position, get_open_stop_loss_entry_time
from app.live import kis_market_status
from app.live.last_price import freshest_tick
from app.live.market_hours import is_market_open
from app.user_context import serviced_scopes, user_scope
from app.web import trade_allowlist_store

logger = logging.getLogger("app.live.exit_monitor")

_ATR_CACHE_TTL_SECONDS = 3600.0


async def _cached_atr_pct(ticker: str) -> float | None:
    """Best-effort — a fetch failure degrades to None (compute_trailing_pct's
    own flat fallback), never blocks the exit check itself. Not user-scoped
    (app.web.state.cached's `user_key`): ATR is public market data, same
    for everyone, same convention as that module's own ranking/index caches.
    Local import: app.web.state imports app.live.exit_monitor (to start this
    module's own loop at boot) — importing it back at module load would be
    circular, same reasoning as app.live.auto_trade_loop's identical local
    `from app.web import state` imports elsewhere in this package."""
    from app.web.state import cached

    try:
        return await cached(f"exit_monitor_atr:{ticker}", _ATR_CACHE_TTL_SECONDS, lambda: get_atr_pct(ticker))
    except Exception:
        logger.warning("exit_monitor: ATR fetch failed for %s — trailing distance falls back to flat %%", ticker, exc_info=True)
        return None


async def check_exit_for_ticker(ticker: str) -> str:
    """One ticker's real-time stop-loss check — pure enough to unit test
    directly, mirroring `app.execution.ledger.check_stops_for_ticker`'s own
    role for the daily-bar path."""
    if not is_market_open(ticker):
        return f"{ticker}: market closed — skipping this pass."

    open_stop = await get_open_stop_loss_entry_time(ticker)
    if open_stop is None:
        return f"{ticker}: no open stop-loss on record (flat, or none was set)."
    entry_time, entry_price, pm_stop_loss = open_stop

    price_and_time = freshest_tick(ticker)
    if price_and_time is None:
        return f"{ticker}: no live tick yet — skipping this pass."
    price, received_at = price_and_time
    if received_at <= entry_time:
        # Same-day (or same-instant, in a fast backtest/replay) entry: a
        # tick from before the position opened cannot be a breach of ITS
        # stop — see get_open_stop_loss_entry_time's own docstring for why
        # this needs the real entry TIMESTAMP, not just entry_date.
        return f"{ticker}: latest tick predates this position's entry — skipping this pass."

    # get_clock().now() defaults to a naive local datetime (LiveClock.now's
    # own docstring / datetime.now(None)) — both tick sources above are
    # always timezone-aware UTC (app.live.store.Tick.timestamp;
    # RealtimeFeed.tick_received_at). Subtracting a naive from an aware
    # datetime raises TypeError, so this must ask for UTC explicitly.
    tick_age = (get_clock().now(timezone.utc) - received_at).total_seconds()
    if tick_age > settings.exit_monitor_max_tick_age_seconds:
        logger.warning(
            "exit_monitor: %s tick is %.0fs old (> %.0fs max) — treating price as unknown, not exiting",
            ticker, tick_age, settings.exit_monitor_max_tick_age_seconds,
        )
        return f"{ticker}: latest tick is stale ({tick_age:.0f}s) — skipping this pass."

    atr_pct = await _cached_atr_pct(ticker)
    high_water, trailing_pct = await trailing_stop.update_high_water_mark(
        ticker, entry_time, entry_price, price, atr_pct
    )
    stop_loss = trailing_stop.effective_stop_loss(pm_stop_loss, entry_price, high_water, trailing_pct)

    if price > stop_loss:
        return f"{ticker}: stop-loss ({stop_loss:.2f}, PM {pm_stop_loss}) not breached (last tick {price})."

    # Three distinct numbers, three distinct arguments (2026-08-27) — before
    # this, `price` was the only one that fit, so `trades.stop_loss` on every
    # tick-triggered exit recorded the breaching TICK instead of the floor
    # that was actually breached. The daily-bar backstop had the mirror-image
    # version of the same defect (it passed the FLOOR, which then landed in
    # `trades.price`) — see `force_exit_position`'s own 2026-08-27 note and
    # the 047040.KS incident it records. `exit_price` is unchanged: it is the
    # ORDER price, and lowering an overseas sell limit to the tick is exactly
    # what makes it fill.
    outcome = await force_exit_position(
        ticker, get_clock().today(), price, "real-time tick monitor",
        fill_price=price, breach_floor=stop_loss,
    )
    if outcome.status == "already_flat":
        return f"{ticker}: stop-loss ({stop_loss:.2f}) breached at tick price {price}, but already flat."
    if outcome.status == "cooldown":
        return (
            f"{ticker}: stop-loss ({stop_loss:.2f}) breached at tick {price} — exit suppressed "
            f"({outcome.detail}); POSITION STILL OPEN AND UNPROTECTED."
        )
    if not outcome.ok:
        # ERROR, not the "sold" line this branch used to fall through to at
        # DEBUG (see exit_monitor_loop below) — this is precisely the
        # failure mode that went unnoticed for 43 minutes on AXTI
        # (2026-08-19): a stop-loss breach whose exit order never reached
        # KIS, logged as if it had succeeded.
        logger.error(
            "exit_monitor: %s stop-loss exit FAILED at tick %.4f (stop %.2f) — position still open "
            "and UNPROTECTED. KIS: %s", ticker, price, stop_loss, outcome.disposition,
        )
        return (
            f"{ticker}: stop-loss ({stop_loss:.2f}) breached at tick {price} — exit FAILED, "
            f"POSITION STILL OPEN AND UNPROTECTED. KIS: {outcome.disposition}"
        )
    return f"{ticker}: stop-loss ({stop_loss:.2f}) breached at tick price {price} — sold {outcome.shares} shares."


async def exit_monitor_loop() -> None:
    """One pass per serviced scope per interval, each inside `user_scope` +
    `account_mode_scope` — same reasoning as every other background loop in
    this package (`poller.run_poller_loop`, `app.web.state.stop_loss_loop`):
    force_exit_position reads/writes this scope's own account and ledger,
    so nothing here may run un-bound."""
    interval_seconds = settings.exit_monitor_interval_seconds
    while True:
        for scope in await serviced_scopes():
            try:
                with user_scope(scope.user_id), account_mode_scope(scope.account_mode):
                    await trade_allowlist_store.ensure_loaded(scope.user_id, scope.account_mode)
                    for ticker in settings.trade_allowlist:
                        try:
                            result = await check_exit_for_ticker(ticker)
                            logger.debug(
                                "exit_monitor check for %s (%s/%s): %s",
                                ticker, scope.user_id, scope.account_mode, result,
                            )
                        except Exception:
                            logger.exception(
                                "exit_monitor check failed for %s (%s/%s) (will retry next interval)",
                                ticker, scope.user_id, scope.account_mode,
                            )
            except Exception:
                logger.exception(
                    "exit_monitor pass failed for %s/%s — continuing with the other scopes",
                    scope.user_id, scope.account_mode,
                )
        await asyncio.sleep(interval_seconds)
