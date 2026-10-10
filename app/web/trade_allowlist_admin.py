"""Bulk, validated trade_allowlist reseed — the official path for a
deliberate large change to who's on the domestic/overseas allowlist.

Just editing `.env`'s TRADE_ALLOWLIST_DOMESTIC/_OVERSEAS doesn't work once
anything has ever been persisted: `trade_allowlist_store.load()` always
restores whatever's in Postgres over the `.env` default at boot (by design
— an AI auto-added holding surviving a restart matters more than a stale
env value winning). The 2026-07-29 market-cap top-10 reseed hit exactly
this: the env edit was silently overridden by the DB every time the server
started, and nobody noticed until the next day (see circuit-breaker-block-
new-ticker-investigation's own notes). Hand-writing the `trade_allowlist_state`
row directly would "work" but skips every check add_ticker_to_trade_allowlist
normally runs (ticker-format shape, leverage/inverse, unsupported product type).

`reseed_trade_allowlist` reuses that same validated add/remove path for
every ticker that changes, so a bulk reseed gets identical guarantees to
clicking the dashboard's add/remove buttons one at a time — just without
actually doing that by hand for 10-20 tickers.
"""

from __future__ import annotations

from app.config import Market, settings
from app.web import state, trade_allowlist_store


async def reseed_trade_allowlist(market: Market, desired: list[str]) -> dict:
    """Replace `market`'s trade_allowlist with exactly `desired`. Loads the
    current persisted state first (not whatever `.env` happens to say),
    diffs against `desired`, then adds/removes only what actually changed
    through the normal validated path — best-effort per ticker, so one
    rejected symbol (bad format / leveraged / unsupported product type)
    doesn't abort the rest of the run. Returns a summary dict rather than
    raising, since a partial reseed with a reported rejection list is more
    useful here than an all-or-nothing CLI command."""
    await trade_allowlist_store.load()
    current = set(settings.trade_allowlist_for_market(market))
    desired_set = set(desired)
    to_remove = sorted(current - desired_set)
    to_add = sorted(desired_set - current)

    removed = []
    for ticker in to_remove:
        await state.remove_ticker_from_trade_allowlist(ticker)
        removed.append(ticker)

    added = []
    rejected = {}
    for ticker in to_add:
        try:
            await state.add_ticker_to_trade_allowlist(ticker)
            added.append(ticker)
        except Exception as exc:  # noqa: BLE001 - reject this ticker, keep going
            rejected[ticker] = str(exc)

    return {
        "market": market,
        "kept": sorted(current & desired_set),
        "added": added,
        "removed": removed,
        "rejected": rejected,
    }
