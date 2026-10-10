"""MKOP_CLS_CODE circuit-breaker value mapping — pure, no network."""

from __future__ import annotations

from dataclasses import replace

from app.brokers.kis.domestic.websocket import _MARKET_STATUS_TR, MarketStatus, RealtimeFeed
from app.live import kis_market_status as kms

_BASE = MarketStatus(
    code="005930", trading_halted=False, halt_reason="",
    market_op_code="", vi_code="", stock_state_code="",
)


def _feed_with_status(market_op_code: str) -> RealtimeFeed:
    feed = RealtimeFeed(app_key="k", app_secret="s", codes=["005930"])
    feed.statuses["005930"] = replace(_BASE, market_op_code=market_op_code)
    # A status frame having actually arrived for this code means KIS ACKed
    # its H0STMKO0 registration — current_status() now requires is_subscribed
    # too (2026-08-10 fix), so a real feed would already have this. These
    # tests are about market_op_code value interpretation, not the
    # subscribed/reconnect-staleness gates themselves.
    feed._subscribed.add((_MARKET_STATUS_TR, "005930"))
    # connected=True: a status frame having arrived means we were connected
    # when it did. current_status()'s staleness check only applies while
    # disconnected (see that method's own docstring) — these tests are about
    # value interpretation, not the reconnect-staleness path.
    feed.connected = True
    return feed


def test_circuit_breaker_triggered_codes_block():
    for code in ("174", "182", "184"):
        kms.set_active_feed(_feed_with_status(code))
        assert kms.is_circuit_breaker_active("005930") is True
    kms.set_active_feed(None)


def test_circuit_breaker_cleared_codes_do_not_block():
    for code in ("175", "185"):
        kms.set_active_feed(_feed_with_status(code))
        assert kms.is_circuit_breaker_active("005930") is False
    kms.set_active_feed(None)


def test_ordinary_session_codes_do_not_block():
    # 112 장개시 (regular open) — must resolve to False, not None/"unknown".
    kms.set_active_feed(_feed_with_status("112"))
    assert kms.is_circuit_breaker_active("005930") is False
    kms.set_active_feed(None)


def test_no_data_is_unknown():
    kms.set_active_feed(_feed_with_status("112"))
    assert kms.is_circuit_breaker_active("000660") is None  # never subscribed/no data for this code
    kms.set_active_feed(None)
    assert kms.is_circuit_breaker_active("005930") is None  # no feed active at all


def test_connected_and_subscribed_with_no_event_yet_is_not_active():
    # 2026-07-21: a quiet KRX code can go a whole session with zero
    # H0STMKO0 frames — that's the steady state, not "unknown." "Subscribed"
    # now means KIS actually ACKed rt_cd="0" for this registration (see
    # RealtimeFeed.is_subscribed), not merely that we attempted it.
    feed = RealtimeFeed(app_key="k", app_secret="s", codes=["005930"])
    feed.connected = True
    feed._subscribed.add((_MARKET_STATUS_TR, "005930"))
    kms.set_active_feed(feed)
    assert kms.is_circuit_breaker_active("005930") is False
    kms.set_active_feed(None)


def test_subscribed_but_not_connected_with_no_event_yet_is_unknown():
    feed = RealtimeFeed(app_key="k", app_secret="s", codes=["005930"])
    feed.connected = False  # dropped/reconnecting — default, but explicit here
    feed._subscribed.add((_MARKET_STATUS_TR, "005930"))
    kms.set_active_feed(feed)
    assert kms.is_circuit_breaker_active("005930") is None
    kms.set_active_feed(None)


def test_connected_but_code_not_subscribed_is_unknown():
    feed = RealtimeFeed(app_key="k", app_secret="s", codes=["005930"])
    feed.connected = True
    feed._subscribed.add((_MARKET_STATUS_TR, "005930"))
    kms.set_active_feed(feed)
    assert kms.is_circuit_breaker_active("000660") is None  # 000660 was never ACKed
    kms.set_active_feed(None)


def test_attempted_but_not_acked_subscription_is_unknown_not_false():
    # The exact P0 gap this fix closes: a code that was SENT a subscribe
    # message (so it's in feed.codes) but whose ACK was rejected (never
    # added to _subscribed) must NOT resolve to "not halted" just because
    # the socket is connected and the code was in the initial batch.
    feed = RealtimeFeed(app_key="k", app_secret="s", codes=["005930"])
    feed.connected = True
    kms.set_active_feed(feed)
    assert kms.is_circuit_breaker_active("005930") is None
    kms.set_active_feed(None)


def test_stale_status_after_disconnect_is_unknown():
    feed = _feed_with_status("112")  # sets connected=True
    feed.connected = False
    feed._status_received_at["005930"] = 0.0  # far in the past
    kms.set_active_feed(feed)
    assert kms.is_circuit_breaker_active("005930") is None
    kms.set_active_feed(None)


def test_recent_status_survives_a_brief_disconnect():
    import time

    feed = _feed_with_status("174")  # circuit breaker triggered
    feed.connected = False
    feed._status_received_at["005930"] = time.monotonic()  # just received
    kms.set_active_feed(feed)
    assert kms.is_circuit_breaker_active("005930") is True
    kms.set_active_feed(None)


def test_reconnected_but_resubscribe_rejected_is_unknown_not_stale():
    # The exact gap this fix closes: reconnect succeeds (connected=True
    # again), but re-registering THIS code's H0STMKO0 subscription was
    # rejected (e.g. MAX SUBSCRIBE OVER recurring) — run()'s reconnect
    # `finally` clears _subscribed but never clears `statuses`, so without
    # the is_subscribed check, the pre-reconnect status (from a triggered
    # circuit breaker that may have long since lifted, or vice versa) would
    # be trusted forever once connected=True, since the disconnected-only
    # staleness clock never runs again.
    feed = _feed_with_status("174")  # pre-reconnect: circuit breaker triggered
    feed._subscribed.clear()  # re-subscribe never got re-ACKed after reconnect
    feed.connected = True  # but the socket itself did reconnect
    kms.set_active_feed(feed)
    assert kms.is_circuit_breaker_active("005930") is None
    kms.set_active_feed(None)


# 2026-10-02: cb_unknown_reason five branches
def test_cb_unknown_reason_five_branches():
    try:
        kms.set_active_feed(None)
        assert kms.cb_unknown_reason("005930") == "feed_not_started"

        feed = RealtimeFeed(app_key="k", app_secret="s", codes=["005930"])
        kms.set_active_feed(feed)
        feed.capped_codes.add("000660")  # disconnected wins over capped
        assert kms.cb_unknown_reason("000660") == "feed_disconnected"

        feed.connected = True
        assert kms.cb_unknown_reason("000660") == "registration_capped"

        feed.rejected_keys.add((_MARKET_STATUS_TR, "035420"))
        assert kms.cb_unknown_reason("035420") == "subscribe_rejected"

        # capped outranks rejected
        feed.rejected_keys.add((_MARKET_STATUS_TR, "000660"))
        assert kms.cb_unknown_reason("000660") == "registration_capped"

        assert kms.cb_unknown_reason("005930") == "code_not_subscribed"
    finally:
        kms.set_active_feed(None)


def test_registration_usage():
    try:
        kms.set_active_feed(None)
        assert kms.registration_usage() is None
        feed = RealtimeFeed(app_key="k", app_secret="s", codes=["005930"])
        kms.set_active_feed(feed)
        assert kms.registration_usage() == (feed.registration_count(), RealtimeFeed.MAX_REGISTRATIONS)
    finally:
        kms.set_active_feed(None)
