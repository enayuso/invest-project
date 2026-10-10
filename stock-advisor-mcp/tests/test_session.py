from datetime import date, datetime

import pandas as pd
import pytest

from stock_advisor_mcp import cache, freshness, session
from stock_advisor_mcp.market_hours import KST


def _bars(dates: list[str], closes: list[float]) -> pd.DataFrame:
    idx = pd.to_datetime(dates)
    return pd.DataFrame(
        {
            "Open": closes,
            "High": [c + 1 for c in closes],
            "Low": [c - 1 for c in closes],
            "Close": closes,
            "Volume": [1000] * len(dates),
        },
        index=idx,
    )


@pytest.fixture(autouse=True)
def clean_state():
    cache.clear_for_tests()
    freshness.reset_reference_cache_for_tests()
    yield
    cache.clear_for_tests()
    freshness.reset_reference_cache_for_tests()


def _patch_reference(monkeypatch, latest_day: str):
    monkeypatch.setattr(session, "latest_trading_day", lambda: date.fromisoformat(latest_day))


def test_session_view_in_progress_excludes_last_bar_from_completed(monkeypatch):
    _patch_reference(monkeypatch, "2026-09-21")
    monkeypatch.setattr(
        session,
        "fetch_daily_bars",
        lambda symbol, period: _bars(["2026-09-18", "2026-09-21"], [100.0, 105.0]),
    )
    monkeypatch.setattr(session, "fetch_info", lambda symbol: {"marketState": "REGULAR"})
    monkeypatch.setattr(session, "_now_kst", lambda: datetime(2026, 9, 21, 11, 0, tzinfo=KST))

    view = session.get_session_view("005930.KS", period="5d")
    assert view.session_status == "in_progress"
    assert view.session_basis == "market_state_regular"
    assert len(view.completed_bars) == 1
    assert view.completed_bars.index[-1].date() == date(2026, 9, 18)


def test_session_view_closed_keeps_last_bar_in_completed(monkeypatch):
    _patch_reference(monkeypatch, "2026-09-21")
    monkeypatch.setattr(
        session,
        "fetch_daily_bars",
        lambda symbol, period: _bars(["2026-09-18", "2026-09-21"], [100.0, 105.0]),
    )
    monkeypatch.setattr(session, "fetch_info", lambda symbol: {"marketState": "CLOSED"})
    monkeypatch.setattr(session, "_now_kst", lambda: datetime(2026, 9, 21, 18, 0, tzinfo=KST))

    view = session.get_session_view("005930.KS", period="5d")
    assert view.session_status == "closed"
    assert view.session_basis == "after_close_confirm_time"
    assert len(view.completed_bars) == 2
    assert view.completed_bars.index[-1].date() == date(2026, 9, 21)


def test_session_view_raises_on_empty_bars(monkeypatch):
    _patch_reference(monkeypatch, "2026-09-21")
    monkeypatch.setattr(session, "fetch_daily_bars", lambda symbol, period: _bars([], []))
    monkeypatch.setattr(session, "fetch_info", lambda symbol: {})

    with pytest.raises(freshness.NoDataError):
        session.get_session_view("036540.KS", period="5d")


def test_session_view_raises_when_stale(monkeypatch):
    _patch_reference(monkeypatch, "2026-09-21")
    monkeypatch.setattr(
        session,
        "fetch_daily_bars",
        lambda symbol, period: _bars(["2026-09-16", "2026-09-17", "2026-09-18"], [100.0, 101.0, 102.0]),
    )
    monkeypatch.setattr(session, "fetch_info", lambda symbol: {"marketState": "REGULAR"})

    with pytest.raises(freshness.StaleDataError):
        session.get_session_view("005930.KQ", period="5d")
