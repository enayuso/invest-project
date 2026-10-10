"""app.live.exit_monitor.check_exit_for_ticker — the tick-based real-time
stop-loss check. Uses an overseas ticker (AAPL) for the shared staleness/
entry-time/breach logic, which is source-agnostic and reads
`app.live.store` (`latest`) for that market — see
`app.live.last_price.freshest_tick`'s own docstring for why domestic
instead reads the KIS websocket feed directly, covered separately below.

`app.execution.trailing_stop` itself has its own dedicated test file
(tests/execution/test_trailing_stop.py) — here it's stubbed as a pass-
through (update_high_water_mark returns the entry price with no trailing
component, effective_stop_loss returns the PM's own stop unchanged) so
these tests stay focused on exit_monitor's own branching, not trailing
math. A separate section below tests the trailing wiring itself."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.live import exit_monitor, last_price
from app.live.store import Tick


def _tick(price: float, age_seconds: float = 0.0) -> Tick:
    now = datetime.now(timezone.utc)
    return Tick(price=price, volume=0, timestamp=now - timedelta(seconds=age_seconds))


def _stub_trailing_passthrough(monkeypatch):
    """No trailing effect at all — effective stop == the PM's own stop."""

    async def fake_update(ticker, entry_time, entry_price, current_price, atr_pct):
        return entry_price, 0.0

    monkeypatch.setattr(exit_monitor.trailing_stop, "update_high_water_mark", fake_update)
    monkeypatch.setattr(
        exit_monitor.trailing_stop, "effective_stop_loss",
        lambda pm_stop_loss, entry_price, high_water_price, trailing_pct: pm_stop_loss,
    )


async def _no_atr(monkeypatch):
    async def fake_atr(ticker):
        return None

    monkeypatch.setattr(exit_monitor, "_cached_atr_pct", fake_atr)


async def test_skips_when_market_closed(monkeypatch):
    monkeypatch.setattr(exit_monitor, "is_market_open", lambda ticker: False)
    result = await exit_monitor.check_exit_for_ticker("AAPL")
    assert "market closed" in result


async def test_skips_when_no_open_stop_loss(monkeypatch):
    monkeypatch.setattr(exit_monitor, "is_market_open", lambda ticker: True)

    async def fake_open_stop(ticker):
        return None

    monkeypatch.setattr(exit_monitor, "get_open_stop_loss_entry_time", fake_open_stop)
    result = await exit_monitor.check_exit_for_ticker("AAPL")
    assert "no open stop-loss" in result


async def test_skips_when_no_live_tick_yet(monkeypatch):
    monkeypatch.setattr(exit_monitor, "is_market_open", lambda ticker: True)

    async def fake_open_stop(ticker):
        return (datetime.now(timezone.utc) - timedelta(hours=1), 100.0, 10.0)

    monkeypatch.setattr(exit_monitor, "get_open_stop_loss_entry_time", fake_open_stop)
    monkeypatch.setattr(last_price, "latest", lambda ticker: None)
    result = await exit_monitor.check_exit_for_ticker("AAPL")
    assert "no live tick yet" in result


async def test_skips_a_tick_from_before_the_position_entered(monkeypatch):
    # Same-day entry: a tick recorded BEFORE this position opened must never
    # be read as a breach of ITS stop — see get_open_stop_loss_entry_time's
    # own docstring for why this needs the real entry timestamp.
    entry_time = datetime.now(timezone.utc)
    monkeypatch.setattr(exit_monitor, "is_market_open", lambda ticker: True)

    async def fake_open_stop(ticker):
        return (entry_time, 100.0, 10.0)

    monkeypatch.setattr(exit_monitor, "get_open_stop_loss_entry_time", fake_open_stop)
    stale_tick = Tick(price=5.0, volume=0, timestamp=entry_time - timedelta(seconds=1))
    monkeypatch.setattr(last_price, "latest", lambda ticker: stale_tick)
    result = await exit_monitor.check_exit_for_ticker("AAPL")
    assert "predates this position's entry" in result


async def test_skips_a_stale_tick_rather_than_exit_on_unknown_price(monkeypatch):
    monkeypatch.setattr(exit_monitor.settings, "exit_monitor_max_tick_age_seconds", 30.0)
    monkeypatch.setattr(exit_monitor, "is_market_open", lambda ticker: True)

    async def fake_open_stop(ticker):
        return (datetime.now(timezone.utc) - timedelta(hours=1), 100.0, 10.0)

    monkeypatch.setattr(exit_monitor, "get_open_stop_loss_entry_time", fake_open_stop)
    # Price is well below the stop, but the tick is too old to trust.
    monkeypatch.setattr(last_price, "latest", lambda ticker: _tick(5.0, age_seconds=60.0))

    async def should_not_be_called(*args, **kwargs):
        raise AssertionError("must not force-exit on a stale tick")

    monkeypatch.setattr(exit_monitor, "force_exit_position", should_not_be_called)
    result = await exit_monitor.check_exit_for_ticker("AAPL")
    assert "stale" in result


async def test_not_breached_when_tick_price_above_stop(monkeypatch):
    monkeypatch.setattr(exit_monitor, "is_market_open", lambda ticker: True)
    _stub_trailing_passthrough(monkeypatch)
    await _no_atr(monkeypatch)

    async def fake_open_stop(ticker):
        return (datetime.now(timezone.utc) - timedelta(hours=1), 100.0, 10.0)

    monkeypatch.setattr(exit_monitor, "get_open_stop_loss_entry_time", fake_open_stop)
    monkeypatch.setattr(last_price, "latest", lambda ticker: _tick(11.0, age_seconds=1.0))

    async def should_not_be_called(*args, **kwargs):
        raise AssertionError("must not force-exit when price is above the stop")

    monkeypatch.setattr(exit_monitor, "force_exit_position", should_not_be_called)
    result = await exit_monitor.check_exit_for_ticker("AAPL")
    assert "not breached" in result


async def test_breached_tick_triggers_force_exit(monkeypatch):
    monkeypatch.setattr(exit_monitor, "is_market_open", lambda ticker: True)
    _stub_trailing_passthrough(monkeypatch)
    await _no_atr(monkeypatch)

    async def fake_open_stop(ticker):
        return (datetime.now(timezone.utc) - timedelta(hours=1), 100.0, 10.0)

    monkeypatch.setattr(exit_monitor, "get_open_stop_loss_entry_time", fake_open_stop)
    monkeypatch.setattr(last_price, "latest", lambda ticker: _tick(9.0, age_seconds=1.0))

    captured = {}

    async def fake_force_exit(ticker, exit_date, exit_price, reason, **kwargs):
        captured.update(ticker=ticker, exit_price=exit_price, reason=reason)
        return exit_monitor.ForcedExitOutcome(status="exited", shares=5, disposition="mirrored")

    monkeypatch.setattr(exit_monitor, "force_exit_position", fake_force_exit)
    result = await exit_monitor.check_exit_for_ticker("AAPL")
    assert captured["ticker"] == "AAPL"
    assert captured["exit_price"] == 9.0
    assert "sold 5 shares" in result


async def test_breach_but_already_flat_is_reported_not_treated_as_error(monkeypatch):
    monkeypatch.setattr(exit_monitor, "is_market_open", lambda ticker: True)
    _stub_trailing_passthrough(monkeypatch)
    await _no_atr(monkeypatch)

    async def fake_open_stop(ticker):
        return (datetime.now(timezone.utc) - timedelta(hours=1), 100.0, 10.0)

    monkeypatch.setattr(exit_monitor, "get_open_stop_loss_entry_time", fake_open_stop)
    monkeypatch.setattr(last_price, "latest", lambda ticker: _tick(9.0, age_seconds=1.0))

    async def fake_force_exit(ticker, exit_date, exit_price, reason, **kwargs):
        return exit_monitor.ForcedExitOutcome(status="already_flat")

    monkeypatch.setattr(exit_monitor, "force_exit_position", fake_force_exit)
    result = await exit_monitor.check_exit_for_ticker("AAPL")
    assert "already flat" in result


# ---------------------------------------------------------------------------
# Trailing stop wiring — update_high_water_mark/effective_stop_loss are
# real (not stubbed) here, to confirm exit_monitor actually feeds them and
# acts on the RESULT, not just the PM's plain stop-loss.
# ---------------------------------------------------------------------------


class _FakeConn:
    """Minimal fake for trailing_stop's UPSERT — enough to exercise the
    real update_high_water_mark/effective_stop_loss math end to end from
    exit_monitor, without a real Postgres connection."""

    def __init__(self):
        self.rows: dict[tuple, dict] = {}

    async def execute(self, ddl):
        return None

    async def fetchrow(self, query, ticker, account_mode, user_id, entry_time, candidate_high, trailing_pct):
        key = (ticker, account_mode, user_id)
        existing = self.rows.get(key)
        if existing is not None and existing["entry_time"] == entry_time:
            high_water = max(existing["high_water_price"], candidate_high)
            pct = existing["trailing_pct"]
        else:
            high_water = candidate_high
            pct = trailing_pct
        self.rows[key] = {"entry_time": entry_time, "high_water_price": high_water, "trailing_pct": pct}
        return {"high_water_price": high_water, "trailing_pct": pct}


class _AcquireCtx:
    def __init__(self, conn):
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, *exc_info):
        return False


class _FakePool:
    def __init__(self, conn):
        self._conn = conn

    def acquire(self):
        return _AcquireCtx(self._conn)


async def test_trailing_floor_can_trigger_an_exit_the_plain_pm_stop_would_not(monkeypatch):
    monkeypatch.setattr(exit_monitor, "is_market_open", lambda ticker: True)
    monkeypatch.setattr(
        exit_monitor.trailing_stop.risk_rules, "rule",
        lambda key: {
            "trailing_stop_activation_pct": 0.04,
            "trailing_stop_fallback_pct": 0.1,
            "trailing_stop_atr_multiplier": 1.75,
        }[key],
    )
    await _no_atr(monkeypatch)

    conn = _FakeConn()

    async def fake_ready_pool():
        return _FakePool(conn)

    monkeypatch.setattr(exit_monitor.trailing_stop, "_ready_pool", fake_ready_pool)

    entry_time = datetime.now(timezone.utc) - timedelta(hours=1)
    entry_price = 100.0
    pm_stop_loss = 80.0  # far below — plain PM stop would never trigger here

    async def fake_open_stop(ticker):
        return (entry_time, entry_price, pm_stop_loss)

    monkeypatch.setattr(exit_monitor, "get_open_stop_loss_entry_time", fake_open_stop)

    # Ran up to 150 (well past the 4% activation threshold), now pulling
    # back to 130 — trailing floor = 150 * (1 - 0.1) = 135, tighter than
    # the plain PM stop of 80. Must trigger on the REAL effective_stop_loss
    # math, not the PM's own stop alone.
    monkeypatch.setattr(last_price, "latest", lambda ticker: _tick(150.0, age_seconds=1.0))

    async def _noop_force_exit(*args, **kwargs):
        return exit_monitor.ForcedExitOutcome(status="already_flat")

    monkeypatch.setattr(exit_monitor, "force_exit_position", _noop_force_exit)
    await exit_monitor.check_exit_for_ticker("AAPL")  # establishes the high-water mark at 150

    monkeypatch.setattr(last_price, "latest", lambda ticker: _tick(130.0, age_seconds=1.0))
    captured = {}

    async def fake_force_exit(ticker, exit_date, exit_price, reason, **kwargs):
        captured.update(exit_price=exit_price)
        return exit_monitor.ForcedExitOutcome(status="exited", shares=1, disposition="mirrored")

    monkeypatch.setattr(exit_monitor, "force_exit_position", fake_force_exit)
    result = await exit_monitor.check_exit_for_ticker("AAPL")
    assert captured["exit_price"] == 130.0
    assert "sold 1 shares" in result


# ---------------------------------------------------------------------------
# freshest_tick — domestic reads the KIS websocket feed directly, not
# app.live.store (2026-08-10 fix: app.live.poller only samples the feed
# into app.live.store once per poll_interval_seconds/60s, too coarse for
# this loop's 5s interval to ever see a fresh tick).
# ---------------------------------------------------------------------------


class _FakeTick:
    def __init__(self, price: float):
        self.price = price


class _FakeFeed:
    def __init__(self, ticks: dict, received_at: dict):
        self.ticks = ticks
        self._received_at = received_at

    def latest_tick_received_at(self, code):
        return self._received_at.get(code)


async def test_domestic_ticker_reads_the_websocket_feed_not_the_store(monkeypatch):
    monkeypatch.setattr(exit_monitor, "is_market_open", lambda ticker: True)
    received_at = datetime.now(timezone.utc)
    feed = _FakeFeed({"005930": _FakeTick(9.0)}, {"005930": received_at})
    monkeypatch.setattr(exit_monitor.kis_market_status, "get_active_feed", lambda: feed)
    _stub_trailing_passthrough(monkeypatch)
    await _no_atr(monkeypatch)

    async def should_not_be_called(ticker):
        raise AssertionError("domestic ticker must not read app.live.store")

    monkeypatch.setattr(last_price, "latest", should_not_be_called)

    async def fake_open_stop(ticker):
        return (received_at - timedelta(hours=1), 100.0, 10.0)

    monkeypatch.setattr(exit_monitor, "get_open_stop_loss_entry_time", fake_open_stop)

    captured = {}

    async def fake_force_exit(ticker, exit_date, exit_price, reason, **kwargs):
        captured.update(exit_price=exit_price)
        return exit_monitor.ForcedExitOutcome(status="exited", shares=3, disposition="mirrored")

    monkeypatch.setattr(exit_monitor, "force_exit_position", fake_force_exit)
    result = await exit_monitor.check_exit_for_ticker("005930.KS")
    assert captured["exit_price"] == 9.0
    assert "sold 3 shares" in result


async def test_domestic_ticker_no_feed_is_no_live_tick(monkeypatch):
    monkeypatch.setattr(exit_monitor.kis_market_status, "get_active_feed", lambda: None)
    monkeypatch.setattr(exit_monitor, "is_market_open", lambda ticker: True)

    async def fake_open_stop(ticker):
        return (datetime.now(timezone.utc) - timedelta(hours=1), 100.0, 10.0)

    monkeypatch.setattr(exit_monitor, "get_open_stop_loss_entry_time", fake_open_stop)
    result = await exit_monitor.check_exit_for_ticker("005930.KS")
    assert "no live tick yet" in result


async def test_domestic_ticker_unacked_tick_is_no_live_tick(monkeypatch):
    # latest_tick_received_at returning None models a tick present in
    # feed.ticks but whose H0STCNT0 re-subscribe was never ACKed (e.g.
    # after a reconnect) — must not be trusted as current.
    feed = _FakeFeed({"005930": _FakeTick(9.0)}, {})
    monkeypatch.setattr(exit_monitor.kis_market_status, "get_active_feed", lambda: feed)
    monkeypatch.setattr(exit_monitor, "is_market_open", lambda ticker: True)

    async def fake_open_stop(ticker):
        return (datetime.now(timezone.utc) - timedelta(hours=1), 100.0, 10.0)

    monkeypatch.setattr(exit_monitor, "get_open_stop_loss_entry_time", fake_open_stop)
    result = await exit_monitor.check_exit_for_ticker("005930.KS")
    assert "no live tick yet" in result


# ---------------------------------------------------------------------------
# Three prices, three arguments (2026-08-27) — the tick monitor holds both
# the breaching TICK and the floor it breached, and must hand them to
# force_exit_position separately. Before this, `trades.stop_loss` on every
# tick-triggered exit recorded the tick instead of the floor. The ORDER
# price (the positional `exit_price`) must not move.
# ---------------------------------------------------------------------------


async def test_tick_exit_records_the_tick_as_the_fill_and_the_floor_as_the_stop(monkeypatch):
    monkeypatch.setattr(exit_monitor, "is_market_open", lambda ticker: True)
    await _no_atr(monkeypatch)

    async def fake_update(ticker, entry_time, entry_price, current_price, atr_pct):
        return entry_price, 0.0

    monkeypatch.setattr(exit_monitor.trailing_stop, "update_high_water_mark", fake_update)
    # Effective floor 95.0, deliberately DIFFERENT from the PM's own stop
    # (100.0) and from the breaching tick (94.0), so the assertions below
    # can only pass if all three numbers travel independently.
    monkeypatch.setattr(
        exit_monitor.trailing_stop, "effective_stop_loss",
        lambda pm_stop_loss, entry_price, high_water_price, trailing_pct: 95.0,
    )

    async def fake_open_stop(ticker):
        return (datetime.now(timezone.utc) - timedelta(hours=1), 100.0, 100.0)

    monkeypatch.setattr(exit_monitor, "get_open_stop_loss_entry_time", fake_open_stop)
    monkeypatch.setattr(last_price, "latest", lambda ticker: _tick(94.0, age_seconds=1.0))

    captured = {}

    async def fake_force_exit(ticker, exit_date, exit_price, reason, **kwargs):
        captured.update(exit_price=exit_price, kwargs=kwargs)
        return exit_monitor.ForcedExitOutcome(status="exited", shares=7, disposition="mirrored")

    monkeypatch.setattr(exit_monitor, "force_exit_position", fake_force_exit)
    result = await exit_monitor.check_exit_for_ticker("AAPL")

    assert captured["kwargs"]["fill_price"] == 94.0
    assert captured["kwargs"]["breach_floor"] == 95.0
    # The ORDER price is unmoved: still the breaching tick, exactly as
    # before 2026-08-27 (lowering an overseas sell limit to the tick is
    # what makes it fill).
    assert captured["exit_price"] == 94.0
    assert "sold 7 shares" in result
