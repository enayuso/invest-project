"""app.live.poller.run_poller_loop — the loop must survive an exception from
any single pass, not die silently for the rest of the process.

Real incident this guards against: `get_broker()` inside `poll_once` is not
individually try/excepted (unlike the fetch calls right after it), and it
genuinely raises on an ordinary DB-pool-contention hiccup
(`app.memory.db.get_pool`'s documented backoff-window RuntimeError) or a
credential-decrypt failure. Before this fix, that exception propagated
straight out of `while True` and ended the task — and since this task is
only ever awaited at shutdown with `return_exceptions=True`
(`app.web.state`), the exception was never logged or even retrieved. The
live tick feed (and therefore stop-loss-proximity review, which reads
`app.live.store.latest()`) would silently freeze until a process restart.
"""

from __future__ import annotations

import pytest

from app.live import poller
from app.live import store


class _StopLoop(Exception):
    pass


async def _run_two_passes(monkeypatch):
    """Runs exactly two iterations of run_poller_loop's while-True body by
    making asyncio.sleep raise on the second call, breaking the loop right
    after it — same technique as test_auto_trade_loop.py's `_run_one_cycle`."""
    calls = 0

    async def fake_sleep(seconds):
        nonlocal calls
        calls += 1
        if calls >= 2:
            raise _StopLoop()

    monkeypatch.setattr(poller.asyncio, "sleep", fake_sleep)
    try:
        await poller.run_poller_loop()
    except _StopLoop:
        pass


async def test_loop_survives_poll_plan_raising(monkeypatch):
    calls = 0

    async def failing_poll_plan():
        nonlocal calls
        calls += 1
        raise RuntimeError("database unavailable (in backoff window)")

    monkeypatch.setattr(poller, "poll_plan", failing_poll_plan)
    await _run_two_passes(monkeypatch)
    assert calls == 2  # the second pass ran — the first failure did not kill the loop


async def test_loop_survives_poll_once_raising(monkeypatch):
    from app.user_context import ServicedScope

    calls = 0

    async def fake_poll_plan():
        nonlocal calls
        calls += 1
        return ServicedScope("default", "paper"), ["AAPL"]

    async def failing_poll_once(ticker):
        raise RuntimeError("get_broker: credential decrypt failed")

    monkeypatch.setattr(poller, "poll_plan", fake_poll_plan)
    monkeypatch.setattr(poller, "poll_once", failing_poll_once)
    await _run_two_passes(monkeypatch)
    assert calls == 2  # the second pass's poll_plan ran — the first pass's failure did not kill the loop


async def test_loop_logs_the_failure(monkeypatch, caplog):
    async def failing_poll_plan():
        raise RuntimeError("boom")

    monkeypatch.setattr(poller, "poll_plan", failing_poll_plan)
    with caplog.at_level("ERROR", logger="app.live.poller"):
        await _run_two_passes(monkeypatch)
    assert any("poller pass failed" in r.message for r in caplog.records)


class TestKrxVolumeIsDeltaNotCumulative:
    """poll_once must record the CHANGE in KIS's cumulative daily volume
    (acml_vol) since the last poll, not the raw cumulative figure — see
    poller.py's `_last_cumulative_volume` docstring. Comparing raw cumulative
    values in app.live.alerts.check_thresholds' lookback window otherwise
    guarantees false "volume spike" alerts (the total only ever grows), and
    is outright incompatible with app.live.store's yfinance-seeded ticks,
    which are per-bar, not cumulative.
    """

    TICKER = "005930.KS"
    KRX_CODE = "005930"

    def setup_method(self, _method):
        poller._last_cumulative_volume.clear()
        store._ticks.pop(self.TICKER, None)

    def _stub_common(self, monkeypatch, cumulative_volume: int, price: float = 70000.0, tick_age_seconds: float = 0.0):
        from datetime import datetime, timedelta, timezone

        from app.brokers.kis.domestic.websocket import Tick as WsTick

        received_at = datetime.now(timezone.utc) - timedelta(seconds=tick_age_seconds)
        krx_code = self.KRX_CODE

        class _FakeFeed:
            ticks = {
                krx_code: WsTick(
                    code=krx_code, time="", price=price, change_pct=0.0,
                    volume=cumulative_volume, ask1=0.0, bid1=0.0,
                )
            }

            def latest_tick_received_at(self, code):
                return received_at if code == krx_code else None

        monkeypatch.setattr(poller, "is_market_open", lambda _t: True)
        monkeypatch.setattr(poller.kis_market_status, "get_active_feed", lambda: _FakeFeed())

        async def _noop(*_a, **_kw):
            return None

        monkeypatch.setattr(poller, "persist_snapshot", _noop)
        monkeypatch.setattr(poller, "_run_alert_checks", _noop)

    async def test_first_tick_since_restart_records_zero_not_full_cumulative(self, monkeypatch):
        self._stub_common(monkeypatch, cumulative_volume=1_000_000)
        await poller.poll_once(self.TICKER)
        assert store.get_ticks(self.TICKER)[-1].volume == 0

    async def test_second_tick_records_only_the_delta(self, monkeypatch):
        self._stub_common(monkeypatch, cumulative_volume=1_000_000)
        await poller.poll_once(self.TICKER)

        self._stub_common(monkeypatch, cumulative_volume=1_004_500)
        await poller.poll_once(self.TICKER)
        assert store.get_ticks(self.TICKER)[-1].volume == 4_500

    async def test_cumulative_counter_rollover_records_zero_not_negative(self, monkeypatch):
        self._stub_common(monkeypatch, cumulative_volume=1_000_000)
        await poller.poll_once(self.TICKER)

        # A new trading session's counter comes in lower than what was last
        # seen — must not go negative.
        self._stub_common(monkeypatch, cumulative_volume=500)
        await poller.poll_once(self.TICKER)
        assert store.get_ticks(self.TICKER)[-1].volume == 0


class TestStaleWebsocketTickFallsBackToRest:
    """poll_once must not launder an old/unsubscribed websocket tick into a
    fresh-looking one — see poller.py's block comment on `ws_tick_is_fresh`
    for the incident (a rejected re-subscribe after hitting KIS's
    41-subscription cap freezes `feed.ticks[code]`, which used to get
    re-stamped with `datetime.now()` on every poll forever)."""

    TICKER = "005930.KS"
    KRX_CODE = "005930"

    def setup_method(self, _method):
        poller._last_cumulative_volume.clear()
        store._ticks.pop(self.TICKER, None)

    def _stub(self, monkeypatch, *, tick_age_seconds: float | None, rest_price: float | None = 71000.0):
        from datetime import datetime, timedelta, timezone

        from app.brokers.kis.domestic.websocket import Tick as WsTick

        received_at = (
            None if tick_age_seconds is None
            else datetime.now(timezone.utc) - timedelta(seconds=tick_age_seconds)
        )
        krx_code = self.KRX_CODE

        class _FakeFeed:
            ticks = {
                krx_code: WsTick(
                    code=krx_code, time="", price=70000.0, change_pct=0.0,
                    volume=1_000_000, ask1=0.0, bid1=0.0,
                )
            }

            def latest_tick_received_at(self, code):
                return received_at if code == krx_code else None

        class _FakeBroker:
            async def fetch_price(self, _krx_code):
                return rest_price

        monkeypatch.setattr(poller, "is_market_open", lambda _t: True)
        monkeypatch.setattr(poller.kis_market_status, "get_active_feed", lambda: _FakeFeed())

        async def _fake_get_broker(_name):
            return _FakeBroker()

        monkeypatch.setattr(poller, "get_broker", _fake_get_broker)

        async def _noop(*_a, **_kw):
            return None

        monkeypatch.setattr(poller, "persist_snapshot", _noop)
        monkeypatch.setattr(poller, "_run_alert_checks", _noop)

    async def test_fresh_tick_is_used_directly(self, monkeypatch):
        self._stub(monkeypatch, tick_age_seconds=1.0)  # well under 2x60s
        await poller.poll_once(self.TICKER)
        assert store.get_ticks(self.TICKER)[-1].price == 70000.0

    async def test_stale_tick_beyond_2x_poll_interval_falls_back_to_rest(self, monkeypatch, caplog):
        stale_age = 2 * poller.settings.poll_interval_seconds + 1
        self._stub(monkeypatch, tick_age_seconds=stale_age)
        with caplog.at_level("WARNING", logger="app.live.poller"):
            await poller.poll_once(self.TICKER)
        assert store.get_ticks(self.TICKER)[-1].price == 71000.0  # REST fallback price, not the stale ws tick
        assert any("discarding stale KIS websocket tick" in r.message for r in caplog.records)

    async def test_unsubscribed_tick_none_received_at_falls_back_to_rest(self, monkeypatch, caplog):
        # Simulates a rejected re-subscribe: the tick is still sitting in
        # feed.ticks, but latest_tick_received_at reports None because
        # is_subscribed(_TRADE_TR, code) is False.
        self._stub(monkeypatch, tick_age_seconds=None)
        with caplog.at_level("WARNING", logger="app.live.poller"):
            await poller.poll_once(self.TICKER)
        assert store.get_ticks(self.TICKER)[-1].price == 71000.0
        assert any("discarding stale KIS websocket tick" in r.message for r in caplog.records)


class TestForgetTicker:
    """poller.forget_ticker must clear the cumulative-volume baseline so a
    re-added ticker doesn't compute its first delta against a leftover
    baseline from a previous stint on the allowlist."""

    TICKER = "005930.KS"
    KRX_CODE = "005930"

    def test_forget_ticker_clears_cumulative_baseline(self):
        poller._last_cumulative_volume[self.KRX_CODE] = 1_000_000
        poller.forget_ticker(self.TICKER)
        assert self.KRX_CODE not in poller._last_cumulative_volume

    def test_forget_ticker_is_a_noop_for_a_ticker_with_no_krx_code(self):
        poller._last_cumulative_volume["005930"] = 1_000_000
        poller.forget_ticker("AAPL")  # overseas — no krx_code, nothing to drop
        assert poller._last_cumulative_volume["005930"] == 1_000_000
