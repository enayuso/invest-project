"""Tiny in-process TTL cache (spec S:3-3).

The MCP server runs as one long-lived stdio process per Claude Desktop
chat session, and a single "analyze this ticker" turn calls several of
our tools against the same symbol in quick succession. Spec S:3-3 asks
for a 5-minute TTL on price/snapshot data and a 24-hour TTL on
fundamentals, "필수" because a single analysis calls the tools
repeatedly. This is not meant to survive process restarts (unlike the
KRX master-file disk cache) -- it is purely an in-memory guard against
hammering yfinance within one session.
"""

from __future__ import annotations

import time
from typing import Callable, TypeVar

T = TypeVar("T")

_store: dict[tuple[str, str], tuple[float, object]] = {}


def cached(namespace: str, key: str, ttl_seconds: float, fetch: Callable[[], T]) -> T:
    now = time.monotonic()
    cache_key = (namespace, key)
    hit = _store.get(cache_key)
    if hit is not None:
        expires_at, value = hit
        if now < expires_at:
            return value  # type: ignore[return-value]
    value = fetch()
    _store[cache_key] = (now + ttl_seconds, value)
    return value


def clear_for_tests() -> None:
    _store.clear()
