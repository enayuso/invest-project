"""Circuit-breaker gate layer on top of app.brokers.kis_ws.RealtimeFeed.

The real-time WebSocket connection/parsing itself lives in
app.brokers.kis_ws.RealtimeFeed (H0STMKO0 market-operation status, plus
trade ticks/orderbook/fill notifications) — that module owns the wire
protocol, reconnect logic, and AES handling. This module owns only what's
specific to using its market-operation data as a safety gate:

- A module-level reference to whichever RealtimeFeed instance is currently
  running (set by app.web.app's lifespan, read by
  app.execution.kis_bridge's gate 7) — so the gate doesn't need its own
  duplicate connection or the two modules to import each other.
- The mkop_cls_code/vi_cls_code VALUE interpretation (which raw codes mean
  "circuit breaker active" vs "normal," "VI triggered" vs "cleared").
  MKOP_CLS_CODE's circuit-breaker values (174/182/184, confirmed 2026-07-16
  from KIS's own field-spec appendix) are filled in below;
  VI_CLS_CODE's still isn't and continues to return None (genuinely
  unknown, not False) until sourced the same way.
"""

from __future__ import annotations

from app.brokers.kis.domestic.websocket import _MARKET_STATUS_TR, MarketStatus, RealtimeFeed

_active_feed: RealtimeFeed | None = None


def set_active_feed(feed: RealtimeFeed | None) -> None:
    global _active_feed
    _active_feed = feed


def get_active_feed() -> RealtimeFeed | None:
    """Read-only access to whichever feed set_active_feed() last set — lets
    app.live.poller reuse the same live tick stream (price+volume, already
    connected) instead of a separate yfinance REST poll for KRX tickers,
    without app.live having to import app.web (see set_active_feed's own
    docstring for why the reference is pushed in rather than pulled)."""
    return _active_feed


def current_status(code: str) -> MarketStatus | None:
    """Delegates to RealtimeFeed.current_status, which applies the
    disconnected-staleness check — see that method's own docstring."""
    return _active_feed.current_status(code) if _active_feed is not None else None


# MKOP_CLS_CODE (장운영구분코드) values meaning an active circuit breaker,
# per KIS's own field-spec appendix (사용자 확인, 2026-07-16):
#   174 서킷브레이크 발동 (triggered)
#   182 서킷브레이크 장중동시마감 (mid-session single-price close, CB-driven)
#   184 서킷브레이크 개시 (initiated)
# 175/185 ("해제", cleared) are deliberately excluded — those mean the
# circuit breaker just lifted, i.e. normal again. Ordinary session-phase
# codes (110 장전동시호가, 112 장개시, 129 장마감, etc.) are correctly NOT
# in this set, so they resolve to "not active" (False), not "unknown."
_MKOP_CLS_CODE_HALT_VALUES: frozenset[str] = frozenset({"174", "182", "184"})

# VI_CLS_CODE (VI적용구분코드) still has no confirmed value mapping —
# stays empty (is_vi_active keeps returning None) until sourced the same way.
_VI_CLS_CODE_ACTIVE_VALUES: frozenset[str] = frozenset()


def _interpret_mkop_cls_code(mkop_cls_code: str) -> bool | None:
    if not _MKOP_CLS_CODE_HALT_VALUES:
        return None
    return mkop_cls_code in _MKOP_CLS_CODE_HALT_VALUES


def _interpret_vi_cls_code(vi_cls_code: str) -> bool | None:
    if not _VI_CLS_CODE_ACTIVE_VALUES:
        return None
    return vi_cls_code in _VI_CLS_CODE_ACTIVE_VALUES


def is_circuit_breaker_active(code: str) -> bool | None:
    """What app.execution.kis_bridge's gate 7 calls — `code` is a 6-digit KRX
    code. None if the feed isn't connected or this code was never
    subscribed — genuinely unknown, which the gate's confirmed fail-closed
    policy (kis_cb_fail_open_when_unknown) treats as block, not proceed.

    If a status frame HAS arrived for this code, interpret it as usual. If
    none ever has but the feed is connected and this code's H0STMKO0
    registration was actually ACKed by KIS (not just attempted — see
    RealtimeFeed.is_subscribed's own docstring for why "attempted" isn't
    enough), that's not "unknown" — H0STMKO0 only pushes a frame on an
    actual event (trading halt/VI/CB), so on an ordinary day with no such
    event, "no frame yet" is the permanent steady state, not a brief
    startup gap (confirmed 2026-07-21: a quiet KRX code can go all session
    with zero frames while genuinely fine). Treat that case as "not active"
    (False)."""
    status = current_status(code)
    if status is not None:
        return _interpret_mkop_cls_code(status.market_op_code)
    if _active_feed is not None and _active_feed.connected and _active_feed.is_subscribed(_MARKET_STATUS_TR, code):
        return False
    return None


def cb_unknown_reason(code: str) -> str:
    """Pure diagnostic for app.execution.kis_bridge's gate 7: WHY
    is_circuit_breaker_active(code) came back None, so the block message can
    tell a transient "not subscribed yet" apart from a genuine "no feed at
    all." Mirrors is_circuit_breaker_active's own None branch exactly — kept
    here (not duplicated in kis_bridge) so kis_bridge never has to reach into
    RealtimeFeed internals (`connected`, `is_subscribed`) directly.

    Callers should only call this when is_circuit_breaker_active(code) is
    already known to be None — with a status frame present this always
    returns a value from the same branches is_circuit_breaker_active would
    have used, which would be misleading to present as "why it's unknown."

    2026-09-04: replaces a previous "mapping not yet populated" message
    that was simply wrong — `_MKOP_CLS_CODE_HALT_VALUES` has been populated
    since 2026-07-16 and `_interpret_mkop_cls_code` never returns None; every
    real None has always come from one of the three reasons below."""
    if _active_feed is None:
        return "feed_not_started"
    if not _active_feed.connected:
        return "feed_disconnected"
    # 2026-10-02: "code_not_subscribed" used to also cover codes the feed never
    # even tried to register (40-registration budget spent) or that KIS
    # rejected — neither is a transient ACK delay.
    if code in _active_feed.capped_codes:
        return "registration_capped"
    if (_MARKET_STATUS_TR, code) in _active_feed.rejected_keys:
        return "subscribe_rejected"
    return "code_not_subscribed"


def registration_usage() -> tuple[int, int] | None:
    """(used, max) of the active feed's registration budget, or None with no
    feed — lets kis_bridge word the registration_capped block message without
    reaching into RealtimeFeed internals."""
    if _active_feed is None:
        return None
    return _active_feed.registration_count(), _active_feed.MAX_REGISTRATIONS


def is_vi_active(code: str) -> bool | None:
    """Ticker-code-based counterpart to is_circuit_breaker_active, for the
    same parsed-but-not-gated VI data (dashboard/audit use later — not wired
    as a blocking gate; confirmed scope is circuit-breaker only for now)."""
    status = current_status(code)
    if status is None:
        return None
    return _interpret_vi_cls_code(status.vi_code)
