"""End-to-end regression tests for `get_price_snapshot` (spec S:2-2).

These pin the tool's *actual returned number*, not just `format_pct` in
isolation. The 100x bug (already_percent=True on a raw ratio) passed all
28 pre-existing tests because none of them exercised this call site --
`format_pct` was only ever tested directly with inputs where the flag
was correct. If `already_percent=True` is ever reintroduced here, these
tests fail loudly: 0.2 instead of ~19.51.
"""

from __future__ import annotations

import pandas as pd
import pytest

from stock_advisor_mcp import market_data
from stock_advisor_mcp.session import SessionView


def _bars(dates: list[str], closes: list[float]) -> pd.DataFrame:
    idx = pd.to_datetime(dates)
    return pd.DataFrame(
        {
            "Open": closes,
            "High": [c + 50 for c in closes],
            "Low": [c - 50 for c in closes],
            "Close": closes,
            "Volume": [1_000_000] * len(dates),
        },
        index=idx,
    )


def test_get_price_snapshot_change_pct_closed_session_is_true_percentage(monkeypatch):
    # 036540.KQ-shaped scenario from the bug report: prev_close 9,020 -> last 10,780.
    bars = _bars(["2026-09-18", "2026-09-19"], [9020.0, 10780.0])
    view = SessionView(bars=bars, completed_bars=bars, session_status="closed", market_state_raw="CLOSED")
    monkeypatch.setattr(market_data.session, "get_session_view", lambda yf_symbol, period: view)
    monkeypatch.setattr(market_data.session, "fetch_info", lambda yf_symbol: {"previousClose": 9020.0})

    result = market_data.get_price_snapshot("036540.KQ")

    # Correct value is ~19.51 (a real, large move). The 100x bug produced 0.2.
    assert result["change_pct_vs_prev_close"] == pytest.approx(19.51, abs=0.01)
    assert result["change_pct_vs_prev_close"] > 1.0


def test_get_price_snapshot_change_pct_in_progress_session_is_true_percentage(monkeypatch):
    bars = _bars(["2026-09-18", "2026-09-21"], [9020.0, 10780.0])
    view = SessionView(bars=bars, completed_bars=bars.iloc[:-1], session_status="in_progress", market_state_raw="REGULAR")
    monkeypatch.setattr(market_data.session, "get_session_view", lambda yf_symbol, period: view)
    monkeypatch.setattr(market_data.session, "fetch_info", lambda yf_symbol: {"previousClose": 9020.0})

    result = market_data.get_price_snapshot("036540.KQ")

    assert result["change_pct_vs_prev_close"] == pytest.approx(19.51, abs=0.01)
    assert "today_partial" in result
    # Design invariant 1 (README): no "close" field while a session is in
    # progress -- neither nested under today_partial nor as a top-level
    # today_close key (that only appears once the session has closed).
    assert "close" not in result.get("today_partial", {})
    assert "today_close" not in result


def test_get_price_snapshot_change_pct_small_move(monkeypatch):
    # A small, plausible daily move should stay small -- guards against
    # over-correcting in the other direction (e.g. always *100 no matter what).
    bars = _bars(["2026-09-18", "2026-09-19"], [10000.0, 10050.0])
    view = SessionView(bars=bars, completed_bars=bars, session_status="closed", market_state_raw="CLOSED")
    monkeypatch.setattr(market_data.session, "get_session_view", lambda yf_symbol, period: view)
    monkeypatch.setattr(market_data.session, "fetch_info", lambda yf_symbol: {"previousClose": 10000.0})

    result = market_data.get_price_snapshot("005930.KS")

    assert result["change_pct_vs_prev_close"] == pytest.approx(0.5, abs=0.01)
