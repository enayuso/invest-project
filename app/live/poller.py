"""Cheap, LLM-free minute-level polling of the trade allowlist.

Feeds `app.live.store` (the in-memory tick cache the dashboard reads) and
runs the code-only alert thresholds (`app.live.alerts`) on every tick.
Deliberately never touches the multi-agent decision pipeline — see
`app/daemon.py`'s docstring for why running the full debate on a per-minute
clock would be both wasteful and pointless.

KIS-only by policy: everything the frontend displays must come from the
same broker as the candles/orderbook/account panel, not yfinance — yfinance
stays backend-only (AI-analysis inputs it's actually good at: cashflow,
insider/ownership, earnings estimates), never the live tick feed. No
cross-provider fallback either: if KIS has nothing for this ticker this
interval (broker not configured, feed not ready, REST call failed), this
poll is simply skipped rather than filling in from yfinance.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

from app.brokers import get_broker, krx_code_from_ticker
from app.config import account_mode_scope, settings
from app.user_context import ServicedScope, serviced_scopes, user_scope
from app.live import kis_market_status
from app.live.alerts import check_thresholds, filter_new_alerts, record_alert
from app.live.market_hours import is_market_open
from app.live.store import Tick, persist_snapshot, record_tick

logger = logging.getLogger("app.live.poller")

# krx_code -> last raw 누적거래량 (acml_vol) seen from the KIS websocket feed.
# H0STCNT0 reports CUMULATIVE daily volume, but app.live.store's tick buffer
# is also seeded at startup from yfinance's get_intraday_series, which is
# PER-BAR volume (Volume traded in that one minute) — see market_data.py.
# Recording the raw cumulative figure into the same buffer that seed uses
# guarantees a bogus "volume spike" the moment a live tick follows a seeded
# one (a huge cumulative number vs a small per-bar one), and even without
# the seed mixed in, comparing raw cumulative values across a short window
# trends toward "spike" all day just because the total keeps growing. Storing
# the delta since the previous poll keeps every tick in the buffer on the
# same per-interval footing as the yfinance seed.
_last_cumulative_volume: dict[str, int] = {}


def forget_ticker(ticker: str) -> None:
    """Drops `ticker`'s cumulative-volume baseline — called from
    `app.web.state.remove_ticker_from_trade_allowlist` (2026-08-13) when a
    ticker leaves the allowlist. Without this, re-adding the same ticker
    later would compare its very first new tick's cumulative volume against
    a baseline recorded during a PREVIOUS, possibly much earlier, stint on
    the allowlist — producing a nonsense delta (a spurious 0 if the new
    cumulative happens to be lower, or a bogus huge "spike" if it's higher
    from a live session's worth of untracked trading in between). Keyed by
    krx_code, not the yfinance-style ticker string, so this needs the same
    lookup `poll_once` itself uses; a ticker with no KRX code (overseas)
    never appears in `_last_cumulative_volume` to begin with, so there's
    nothing to drop for it."""
    krx_code = krx_code_from_ticker(ticker)
    if krx_code is not None:
        _last_cumulative_volume.pop(krx_code, None)


async def poll_once(ticker: str) -> None:
    if not is_market_open(ticker):
        # Don't record a tick at all while the market's closed — recording
        # the same last-known price on every poll would otherwise draw the
        # chart forward in time as a flat line long after trading actually
        # stopped, and over many hours closed could flood the fixed-size
        # tick buffer (`_MAX_TICKS_PER_TICKER=500`, ~a trading day at
        # 1-minute resolution) with duplicate-price ticks, evicting the real
        # session's history well before the market reopens. The chart's
        # x-axis is index-based, not wall-clock-based (see `drawChart`/
        # `detailLayout` in index.html), so simply not adding a tick here
        # means the chart stays frozen at the last real point and picks up
        # exactly where it left off on the next real tick after the market
        # reopens — no gap-handling needed on the frontend.
        return

    krx_code = krx_code_from_ticker(ticker)
    price: float | None = None
    volume = 0

    if krx_code is not None:
        # KRX: prefer the already-connected KIS real-time feed (H0STCNT0) —
        # price+누적거래량 both included, no extra API call. Only reaches for
        # a REST call (still KIS, fetch_price) when the feed isn't up yet,
        # hasn't received this code's first tick (e.g. just subscribed), OR
        # (2026-08-13) the tick it has is too old to trust.
        #
        # Why staleness needs an explicit check: `feed.ticks[krx_code]` is
        # never cleared on a rejected re-subscribe (e.g. hitting KIS's
        # 41-subscription websocket cap after a reconnect) — it just stops
        # being refreshed and freezes at whatever price arrived before the
        # rejection. Before this fix, this branch trusted that frozen value
        # unconditionally and `record_tick` below re-stamped it with
        # `datetime.now(timezone.utc)`, laundering an arbitrarily stale
        # price into looking like a tick received THIS instant, forever.
        # Two consumers trust that timestamp at face value —
        # `auto_trade_loop._held_needing_review`'s stop-loss-distance check
        # and `_warn_overnight_gap_risk` — so a held position genuinely
        # closing in on its stop would silently stop qualifying for AI
        # re-review. (Real order execution is NOT affected: `exit_monitor.py`
        # reads `feed.ticks`/`feed.latest_tick_received_at` straight off the
        # feed for its own staleness gate, never through this re-stamped
        # copy — do not change that module to "fix" this.)
        #
        # `feed.latest_tick_received_at` already folds in the same
        # `is_subscribed` re-ACK check `exit_monitor`/`current_status` rely
        # on (see its own docstring), so a rejected re-subscribe reports
        # `None` here too — not merely an old-but-present timestamp — and
        # is handled by the same fallback as an outright missing tick.
        feed = kis_market_status.get_active_feed()
        ws_tick = feed.ticks.get(krx_code) if feed is not None else None
        ws_tick_received_at = feed.latest_tick_received_at(krx_code) if feed is not None else None
        ws_tick_age_seconds = (
            (datetime.now(timezone.utc) - ws_tick_received_at).total_seconds()
            if ws_tick_received_at is not None
            else None
        )
        # Threshold derived from poll_interval_seconds (config.py) rather
        # than a new dedicated setting: poll_interval_seconds already IS
        # this loop's definition of "how fresh does data need to be", and
        # introducing a second knob that has to be kept in sync with it
        # would just be one more place to forget to update. 2x, not 1x: the
        # feed pushes ticks on its own schedule, not in lockstep with this
        # loop's clock, so a perfectly healthy tick that landed just before
        # the PREVIOUS poll can legitimately be up to ~1 interval old by the
        # time this poll reads it; gating at exactly 1x would flag good
        # ticks as stale on a routine, not exceptional, basis.
        ws_tick_is_fresh = (
            ws_tick is not None
            and ws_tick_age_seconds is not None
            and ws_tick_age_seconds <= 2 * settings.poll_interval_seconds
        )
        if ws_tick_is_fresh:
            price = ws_tick.price
            cumulative = ws_tick.volume
            last_cumulative = _last_cumulative_volume.get(krx_code)
            if last_cumulative is None or cumulative < last_cumulative:
                # No previous reading (first tick this process, or right
                # after a subscribe/reconnect), or the exchange's cumulative
                # counter is lower than last seen (new trading session) — no
                # valid delta, so record 0 rather than the full cumulative
                # total or a negative number.
                volume = 0
            else:
                volume = cumulative - last_cumulative
            _last_cumulative_volume[krx_code] = cumulative
        else:
            if ws_tick is not None:
                # A tick actually exists here — it's being deliberately
                # discarded for being stale/unsubscribed, not the routine
                # "nothing received yet" case. Must be visible: a rejected
                # re-subscribe would otherwise degrade silently and forever
                # (see block comment above).
                logger.warning(
                    "poller: discarding stale KIS websocket tick for %s (age=%s) — falling back to REST",
                    ticker,
                    f"{ws_tick_age_seconds:.0f}s" if ws_tick_age_seconds is not None else "unknown/not subscribed",
                )
                # Also drop the stale baseline itself — without this, the
                # NEXT fresh tick after a long gap (subscription rejected for
                # 30 minutes, say) would diff against a cumulative reading
                # from before the gap, turning 30 minutes of real volume into
                # one tick's delta and firing a false volume-spike alert.
                # Popping it forces the next fresh tick through the
                # "no previous reading" branch below (volume=0, safe).
                _last_cumulative_volume.pop(krx_code, None)
            broker = await get_broker("kis")
            if broker is not None:
                try:
                    price = await broker.fetch_price(krx_code)
                except Exception:
                    logger.warning("KIS price fetch failed for %s — will retry next interval", ticker, exc_info=True)
                    return
    else:
        # Overseas: no KIS real-time feed exists for this (KRX-only
        # websocket), so it's a REST call every interval. KIS's overseas
        # quote TR doesn't carry volume the way H0STCNT0 does, so volume
        # stays 0 here — an accepted gap, not a bug (see module docstring).
        broker = await get_broker("kis")
        if broker is not None:
            try:
                price = await broker.fetch_overseas_price(ticker)
            except Exception:
                logger.warning("KIS overseas price fetch failed for %s — will retry next interval", ticker, exc_info=True)
                return

    if price is None:
        return  # no KIS broker configured, or feed not ready yet — skip, no fallback

    record_tick(ticker, Tick(price=price, volume=volume, timestamp=datetime.now(timezone.utc)))
    await persist_snapshot(ticker)
    await _run_alert_checks(ticker)


async def _run_alert_checks(ticker: str) -> None:
    # Guarded like the quote fetch above: `check_thresholds` is pure/
    # synchronous today, but any future edge case in it (or in `record_alert`)
    # raising here would otherwise propagate out of `poll_once`, through
    # `run_poller_loop`'s `asyncio.gather`, out of the `while True` loop
    # entirely, and permanently kill the live tick feed with no crash and no
    # visible error — unlike `_stop_loss_loop` (app/web/app.py), which already
    # guards each ticker individually for exactly this reason.
    try:
        for alert in filter_new_alerts(check_thresholds(ticker)):
            logger.info("alert: %s", alert.message)
            await record_alert(alert)
    except Exception:
        logger.warning("alert threshold check failed for %s — will retry next interval", ticker, exc_info=True)


async def poll_plan() -> tuple[ServicedScope | None, list[str]]:
    """The deduplicated union of every serviced scope's trade allowlist.

    Stays ONE shared loop over the union rather than a loop per user, and that
    is a deliberate exception to how the other background loops work
    (docs/multi-user-credentials-plan.md Phase 6). What this loop fetches is
    public market data — a price is a price. It never calls an LLM and never
    places an order, so nothing it does is attributable to a user, and polling
    the same ticker once per user would just multiply identical requests
    against KIS's shared per-key rate limit.
    시세는 공개 데이터라 사용자별로 나눠 받을 이유가 없고, 사용자 수만큼
    같은 요청을 반복하면 KIS 초당 제한만 잡아먹음.

    Reading each scope's list still requires binding it — the lists are
    per (user, mode) and only exist after a load, exactly as in the other loops.
    Deduplicated preserving first-seen order so the log stays readable.

    Returns the scope to poll AS, alongside the tickers. `poll_once` resolves a
    broker from the current identity, and only a user with stored credentials
    has one — so the pass has to run bound to somebody. The first serviced
    scope is used, and any of them would do: a quote is public, so which app key
    fetches it does not change the answer.
    시세 조회에도 앱키가 필요하므로 누군가의 스코프로 실행해야 하며, 공개
    데이터라 누구의 키인지는 결과에 영향이 없음.

    **The quota cost is real and is the reason this returns a scope instead of
    hiding it.** KIS rate-limits per app key, so one user's key carries the
    whole union's polling. That is still strictly fewer requests than polling
    per user (which would refetch the same ticker N times), but it is not free
    and it is not fair. If it becomes a problem the fix is to rotate the
    polling scope between passes, not to poll per user.
    한 사용자의 앱키가 전체 폴링을 감당함 — 사용자별 폴링보다는 훨씬 적지만
    공짜는 아니며, 문제가 되면 주기마다 스코프를 번갈아 쓰는 것이 해법."""
    from app.web import trade_allowlist_store

    scopes = await serviced_scopes()
    seen: dict[str, None] = {}
    for scope in scopes:
        try:
            with user_scope(scope.user_id), account_mode_scope(scope.account_mode):
                await trade_allowlist_store.ensure_loaded(scope.user_id, scope.account_mode)
                for ticker in settings.trade_allowlist:
                    seen.setdefault(ticker, None)
        except Exception:
            logger.exception(
                "could not read the allowlist for %s/%s — polling the rest",
                scope.user_id, scope.account_mode,
            )
    return (scopes[0] if scopes else None), list(seen)


async def run_poller_loop() -> None:
    logger.info("poller loop starting — interval=%ss", settings.poll_interval_seconds)
    while True:
        try:
            # Re-resolved every pass, not captured once: a user who registers
            # credentials or adds a ticker mid-session must start being polled
            # without a restart. Cheap — one query plus in-memory reads.
            # 매 주기마다 다시 계산 — 세션 중 추가된 사용자·종목이 재시작 없이 반영됨.
            scope, tickers = await poll_plan()
            if scope is not None and tickers:
                # Bound for the whole pass: poll_once -> get_broker() needs an
                # identity with credentials, and without one it silently fetches
                # nothing for KRX tickers.
                with user_scope(scope.user_id), account_mode_scope(scope.account_mode):
                    await asyncio.gather(*(poll_once(t) for t in tickers))
        except Exception:
            # Same reasoning as _run_alert_checks' own guard just above, at
            # the level of the whole pass — `get_broker()` inside poll_once
            # is NOT wrapped (unlike the fetch calls right after it), and it
            # genuinely raises on an ordinary DB-pool hiccup (get_pool's
            # documented backoff-window RuntimeError) or a credential-decrypt
            # failure. Before this, either one propagated straight out of
            # `while True` and killed the live tick feed for the rest of the
            # process — silently: this task is only ever awaited at shutdown
            # with return_exceptions=True (app.web.state), so the exception
            # was never logged or even retrieved. Matches the sibling loops'
            # existing convention (stop_loss_loop, auto_trade_loop) of
            # wrapping the whole pass, not just the per-ticker work inside it.
            logger.exception("poller pass failed — will retry next interval")
        try:
            # Piggybacks on this loop's own short cadence rather than a new
            # timer — 2026-08-24, extended 2026-08-28 to the cb_only zone —
            # see app.web.state.reconcile_feed_tier_promotions's own
            # docstring for what this catches (a screening candidate parked
            # in a degraded zone that just got bought, needing its full
            # trade-tick/circuit-breaker coverage restored). Its own broad
            # except so a failure here degrades to "not reconciled this
            # tick, try again next tick" instead of taking the whole poller
            # down with it — same posture as the pass above.
            from app.web.state import reconcile_feed_tier_promotions

            await reconcile_feed_tier_promotions()
        except Exception:
            logger.exception("feed tier promotion check failed — will retry next interval")
        await asyncio.sleep(settings.poll_interval_seconds)
