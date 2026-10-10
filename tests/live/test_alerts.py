"""app.live.alerts.filter_new_alerts — the poller-side cooldown that stops a
sustained threshold breach from re-emitting an identical alert on every poll.

check_thresholds() itself stays pure/synchronous and untouched by cooldown
state (see its own docstring) — filter_new_alerts is the mutable-state layer
poller.py applies to its output before logging/persisting.
"""

from __future__ import annotations

from datetime import datetime, timezone

import app.memory.db as memory_db
from app.config import account_mode_scope, current_account_mode
from app.live import alerts as alerts_module
from app.live.alerts import Alert, filter_new_alerts, prune_old_alerts


def _alert(ticker: str = "005930.KS", kind: str = "price_move") -> Alert:
    return Alert(ticker=ticker, kind=kind, message="test", triggered_at=datetime.now(timezone.utc))


def setup_function(_fn) -> None:
    alerts_module._last_fired_at.clear()


def test_first_alert_always_passes():
    assert len(filter_new_alerts([_alert()])) == 1


def test_repeat_within_cooldown_is_suppressed(monkeypatch):
    monkeypatch.setattr(alerts_module.settings, "alert_cooldown_seconds", 300)
    now = [1000.0]
    monkeypatch.setattr(alerts_module.time, "monotonic", lambda: now[0])

    first = filter_new_alerts([_alert()])
    assert len(first) == 1

    now[0] += 10  # well within the 300s cooldown
    second = filter_new_alerts([_alert()])
    assert second == []


def test_repeat_after_cooldown_fires_again(monkeypatch):
    monkeypatch.setattr(alerts_module.settings, "alert_cooldown_seconds", 300)
    now = [1000.0]
    monkeypatch.setattr(alerts_module.time, "monotonic", lambda: now[0])

    filter_new_alerts([_alert()])
    now[0] += 301
    again = filter_new_alerts([_alert()])
    assert len(again) == 1


def test_different_kind_same_ticker_is_independent(monkeypatch):
    monkeypatch.setattr(alerts_module.settings, "alert_cooldown_seconds", 300)
    now = [1000.0]
    monkeypatch.setattr(alerts_module.time, "monotonic", lambda: now[0])

    filter_new_alerts([_alert(kind="price_move")])
    volume_alert = filter_new_alerts([_alert(kind="volume_spike")])
    assert len(volume_alert) == 1  # different kind, not suppressed by price_move's cooldown


def test_different_ticker_same_kind_is_independent(monkeypatch):
    monkeypatch.setattr(alerts_module.settings, "alert_cooldown_seconds", 300)
    now = [1000.0]
    monkeypatch.setattr(alerts_module.time, "monotonic", lambda: now[0])

    filter_new_alerts([_alert(ticker="005930.KS")])
    other = filter_new_alerts([_alert(ticker="000660.KS")])
    assert len(other) == 1


def test_forget_ticker_clears_cooldown_for_both_alert_kinds(monkeypatch):
    monkeypatch.setattr(alerts_module.settings, "alert_cooldown_seconds", 300)
    now = [1000.0]
    monkeypatch.setattr(alerts_module.time, "monotonic", lambda: now[0])

    filter_new_alerts([_alert(ticker="005930.KS", kind="price_move")])
    filter_new_alerts([_alert(ticker="005930.KS", kind="volume_spike")])
    mode = current_account_mode()
    assert (mode, "005930.KS", "price_move") in alerts_module._last_fired_at
    assert (mode, "005930.KS", "volume_spike") in alerts_module._last_fired_at

    alerts_module.forget_ticker("005930.KS")

    assert (mode, "005930.KS", "price_move") not in alerts_module._last_fired_at
    assert (mode, "005930.KS", "volume_spike") not in alerts_module._last_fired_at


def test_forget_ticker_leaves_other_tickers_cooldowns_alone(monkeypatch):
    monkeypatch.setattr(alerts_module.settings, "alert_cooldown_seconds", 300)
    now = [1000.0]
    monkeypatch.setattr(alerts_module.time, "monotonic", lambda: now[0])

    filter_new_alerts([_alert(ticker="005930.KS")])
    filter_new_alerts([_alert(ticker="000660.KS")])

    alerts_module.forget_ticker("005930.KS")

    assert (current_account_mode(), "000660.KS", "price_move") in alerts_module._last_fired_at


def test_cooldown_is_per_account_mode(monkeypatch):
    """A paper-mode alert must never suppress the real-money alert for the
    same (ticker, kind) — the mode is a runtime-switchable contextvar."""
    monkeypatch.setattr(alerts_module.settings, "alert_cooldown_seconds", 300)
    now = [1000.0]
    monkeypatch.setattr(alerts_module.time, "monotonic", lambda: now[0])

    with account_mode_scope("paper"):
        paper_fired = filter_new_alerts([_alert()])
    assert len(paper_fired) == 1

    now[0] += 10  # well within the 300s cooldown, but a different mode
    with account_mode_scope("real"):
        real_fired = filter_new_alerts([_alert()])
    assert len(real_fired) == 1  # NOT suppressed by paper's cooldown


class _FakeConn:
    def __init__(self, execute_result="DELETE 0"):
        self.queries: list[tuple] = []
        self._execute_result = execute_result

    async def execute(self, query, *args):
        self.queries.append((query, args))
        return self._execute_result


class _FakePool:
    def __init__(self, conn):
        self._conn = conn

    def acquire(self):
        conn = self._conn

        class _CM:
            async def __aenter__(self_):
                return conn

            async def __aexit__(self_, *exc):
                return False

        return _CM()


async def _async_return(value):
    return value


async def test_prune_old_alerts_deletes_past_horizon(monkeypatch):
    conn = _FakeConn("DELETE 7")
    monkeypatch.setattr(memory_db, "get_pool", lambda: _async_return(_FakePool(conn)))
    monkeypatch.setattr(alerts_module, "_schema_ready", True)

    deleted = await prune_old_alerts()

    assert deleted == 7
    query, args = conn.queries[0]
    assert "DELETE FROM alerts" in query and "triggered_at <" in query
    assert args == (str(alerts_module._ALERT_RETENTION_DAYS),)


async def test_prune_old_alerts_returns_zero_on_db_failure(monkeypatch):
    def _boom():
        raise RuntimeError("pool down")

    monkeypatch.setattr(memory_db, "get_pool", _boom)
    assert await prune_old_alerts() == 0


def test_alert_retention_horizon_is_at_least_30_days():
    assert alerts_module._ALERT_RETENTION_DAYS >= 30


def test_forget_ticker_clears_cooldown_across_account_modes(monkeypatch):
    monkeypatch.setattr(alerts_module.settings, "alert_cooldown_seconds", 300)
    now = [1000.0]
    monkeypatch.setattr(alerts_module.time, "monotonic", lambda: now[0])

    with account_mode_scope("paper"):
        filter_new_alerts([_alert()])
    with account_mode_scope("real"):
        filter_new_alerts([_alert()])
    assert ("paper", "005930.KS", "price_move") in alerts_module._last_fired_at
    assert ("real", "005930.KS", "price_move") in alerts_module._last_fired_at

    alerts_module.forget_ticker("005930.KS")

    assert ("paper", "005930.KS", "price_move") not in alerts_module._last_fired_at
    assert ("real", "005930.KS", "price_move") not in alerts_module._last_fired_at
