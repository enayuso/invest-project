"""Tests for the pure session-status rule and its integration through session/market_data/indicators."""

from __future__ import annotations

import ast
import inspect
from datetime import date, datetime, time, timezone

import pandas as pd
import pytest

from stock_advisor_mcp import cache, freshness, indicators, market_data, market_hours, session
from stock_advisor_mcp.market_hours import KST, decide_session_status


def _kst(y, mo, d, h, mi, s=0) -> datetime:
    return datetime(y, mo, d, h, mi, s, tzinfo=KST)


TODAY = date(2026, 10, 7)  # Wednesday


def test_observed_defect_1501_post_is_in_progress():
    assert decide_session_status(TODAY, _kst(2026, 10, 7, 15, 1), "POST") == (
        "in_progress",
        "before_close_confirm_time",
    )


def test_after_confirm_time_is_closed_and_boundary():
    assert decide_session_status(TODAY, _kst(2026, 10, 7, 17, 5), "POST") == ("closed", "after_close_confirm_time")
    assert decide_session_status(TODAY, _kst(2026, 10, 7, 16, 59, 59), "POST")[0] == "in_progress"
    assert decide_session_status(TODAY, _kst(2026, 10, 7, 17, 0, 0), "POST") == ("closed", "after_close_confirm_time")


def test_no_bar_today_is_closed():
    assert decide_session_status(date(2026, 10, 6), _kst(2026, 10, 7, 10, 0), "PRE") == ("closed", "no_bar_today")
    # Saturday with Friday's bar
    assert decide_session_status(date(2026, 10, 9), _kst(2026, 10, 10, 11, 0), "CLOSED") == ("closed", "no_bar_today")


def test_early_regular_and_missing_market_state():
    assert decide_session_status(TODAY, _kst(2026, 10, 7, 8, 50), "PRE")[0] == "in_progress"
    assert decide_session_status(TODAY, _kst(2026, 10, 7, 17, 30), "REGULAR") == (
        "in_progress",
        "market_state_regular",
    )
    assert decide_session_status(TODAY, _kst(2026, 10, 7, 16, 0), None)[0] == "in_progress"


def test_bar_after_today_is_in_progress():
    assert decide_session_status(date(2026, 10, 8), _kst(2026, 10, 7, 12, 0), "CLOSED") == (
        "in_progress",
        "bar_date_after_today",
    )


def test_timezone_handling():
    utc_now = datetime(2026, 10, 7, 6, 1, tzinfo=timezone.utc)  # 15:01 KST
    assert decide_session_status(TODAY, utc_now, "POST") == ("in_progress", "before_close_confirm_time")
    # 16:00 UTC on Oct 6 is already Oct 7 01:00 KST
    utc_prev = datetime(2026, 10, 6, 16, 0, tzinfo=timezone.utc)
    assert decide_session_status(TODAY, utc_prev, None)[0] == "in_progress"
    # naive is treated as KST
    assert decide_session_status(TODAY, datetime(2026, 10, 7, 17, 0), None)[0] == "closed"


def test_confirm_time_constant_is_used(monkeypatch):
    monkeypatch.setattr(market_hours, "CLOSE_CONFIRM_TIME_KST", time(18, 0))
    assert decide_session_status(TODAY, _kst(2026, 10, 7, 17, 30), None)[0] == "in_progress"


def test_market_hours_module_has_no_forbidden_imports():
    tree = ast.parse(inspect.getsource(market_hours))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            imported.add(base)
            imported.update(f"{base}.{a.name}".strip(".") for a in node.names)
    forbidden = {"yfinance", "session", "cache", "yf_gateway", "mcp", "server"}
    for name in imported:
        assert not (set(name.split(".")) & forbidden), name


# ------------------------------------------------------------ integration


@pytest.fixture(autouse=True)
def clean_state():
    cache.clear_for_tests()
    freshness.reset_reference_cache_for_tests()
    yield
    cache.clear_for_tests()
    freshness.reset_reference_cache_for_tests()


def _seoul_bars(last_date: str, n: int = 40) -> pd.DataFrame:
    idx = pd.bdate_range(end=last_date, periods=n, tz="Asia/Seoul")
    closes = [100.0 + i for i in range(n)]
    return pd.DataFrame(
        {
            "Open": closes,
            "High": [c + 1 for c in closes],
            "Low": [c - 1 for c in closes],
            "Close": closes,
            "Volume": [1000] * n,
        },
        index=idx,
    )


def _setup(monkeypatch, now: datetime, market_state="POST"):
    bars = _seoul_bars("2026-10-07")
    monkeypatch.setattr(session, "latest_trading_day", lambda: date(2026, 10, 7))
    monkeypatch.setattr(session, "fetch_daily_bars", lambda symbol, period: bars)
    monkeypatch.setattr(session, "fetch_info", lambda symbol: {"marketState": market_state, "previousClose": 138.0})
    monkeypatch.setattr(session, "_now_kst", lambda: now)


def test_integration_1501_post_snapshot_history_indicators(monkeypatch):
    _setup(monkeypatch, _kst(2026, 10, 7, 15, 1))

    snap = market_data.get_price_snapshot("005930.KS")
    assert snap["session_status"] == "in_progress"
    assert snap["session_basis"] == "before_close_confirm_time"
    assert "today_partial" in snap and "close" not in snap["today_partial"]
    assert "today_close" not in snap
    assert list(snap).index("session_basis") == list(snap).index("session_status") + 1

    hist = market_data.get_price_history("005930.KS", period="3mo")
    assert hist["bars_through"] == "2026-10-06"
    assert all(b["date"] != "2026-10-07" for b in hist["bars"])

    ind = indicators.get_technical_indicators("005930.KS")
    assert ind["computed_through"] == "2026-10-06"


def test_integration_after_confirm_time_is_closed(monkeypatch):
    _setup(monkeypatch, _kst(2026, 10, 7, 17, 5))

    snap = market_data.get_price_snapshot("005930.KS")
    assert snap["session_status"] == "closed"
    assert snap["session_basis"] == "after_close_confirm_time"
    assert snap["today_close"]["date"] == "2026-10-07"
    assert market_data.get_price_history("005930.KS")["bars_through"] == "2026-10-07"
    assert indicators.get_technical_indicators("005930.KS")["computed_through"] == "2026-10-07"
