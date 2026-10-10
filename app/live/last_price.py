"""The freshest price this process actually knows for a ticker, read from
already-maintained in-memory caches only — never a network fetch.

Extracted from `app.live.exit_monitor._live_price` (2026-08-27) because a
SECOND caller now needs the identical read: `app.execution.ledger.
force_exit_position`, recording what a forced exit really sold at. Neither
caller may block: the tick monitor runs every 5s, and a forced safety exit
must never wait on the network (see `force_exit_position`'s own docstring).
Both requirements are satisfied by the same property — every source below
is a dict lookup against a cache some other loop already maintains, so this
function performs no I/O, awaits nothing, and is safe to call from inside
`_force_exit_locks`.

- Domestic (KRX): the live `RealtimeFeed.ticks`/`latest_tick_received_at`
  directly (`app.brokers.kis.domestic.websocket`), NOT `app.live.store`.
  `app.live.poller` only samples that same feed into `app.live.store` once
  per `settings.poll_interval_seconds` (60s default) — confirmed live
  (2026-08-10): reading `app.live.store` here meant the exit monitor's own
  ticks were routinely 30-60s old despite running every 5s.
- Overseas: `app.live.store` (`app.live.poller`'s own tick cache) — there
  is no KIS websocket for overseas at all, so 60s IS this ticker's real
  resolution; nothing fresher exists to read.

Returns `None` when nothing is known (no feed, ticker not subscribed —
the domestic websocket caps at 41 registrations — or the poller has not
sampled this ticker yet). `None` means UNKNOWN, never "safe" and never
"zero": every caller must decide for itself what to do with an unknown
price, and both current callers fail toward the conservative answer (the
tick monitor does not sell; the ledger records no fill price and says so).
This function deliberately applies NO staleness policy of its own — it
returns the timestamp and lets each caller apply its own bound, because
"too old to sell on" (30s) and "too old to record as the fill price"
(15 min) are genuinely different questions.
"""

from __future__ import annotations

from datetime import datetime

from app.brokers import krx_code_from_ticker
from app.live import kis_market_status
from app.live.store import latest


def freshest_tick(ticker: str) -> tuple[float, datetime] | None:
    """(price, received_at UTC) from whichever source is actually freshest
    for this ticker's market — see this module's own docstring. `None`
    means no current price is available at all."""
    krx_code = krx_code_from_ticker(ticker)
    if krx_code is not None:
        feed = kis_market_status.get_active_feed()
        if feed is None:
            return None
        tick = feed.ticks.get(krx_code)
        received_at = feed.latest_tick_received_at(krx_code)
        if tick is None or received_at is None:
            return None
        return tick.price, received_at

    tick = latest(ticker)
    if tick is None:
        return None
    return tick.price, tick.timestamp
