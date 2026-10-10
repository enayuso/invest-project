"""Tests for price-continuity checks (integrity.py) and their tool wiring.

Synthetic data only; no network.
"""

from __future__ import annotations

import ast
import asyncio
import inspect

import numpy as np
import pandas as pd
import pytest

from stock_advisor_mcp import indicators, integrity, market_data, server
from stock_advisor_mcp.session import SessionView


def _frame(closes, dates=None, volumes=None) -> pd.DataFrame:
    n = len(closes)
    if dates is None:
        dates = pd.bdate_range("2026-01-05", periods=n)
    else:
        dates = pd.to_datetime(dates)
    return pd.DataFrame(
        {
            "Open": closes,
            "High": closes,
            "Low": closes,
            "Close": closes,
            "Volume": volumes if volumes is not None else [1000] * n,
        },
        index=dates,
    )


def test_normal_series_ok():
    r = integrity.assess_bars(_frame([100.0, 101.0, 99.5, 102.0]))
    assert r["status"] == "ok"
    assert r["bars_checked"] == 4
    assert r["limit_breaches"] == [] and r["zero_volume_dates"] == [] and r["long_gaps"] == []


@pytest.mark.parametrize("mult", [1.299, 0.701, 1.305])
def test_within_threshold_no_breach(mult):
    # 1.305 is exactly the boundary: strictly greater is required.
    r = integrity.assess_bars(_frame([100.0, 100.0 * mult]))
    assert r["limit_breaches"] == []
    assert r["status"] == "ok"


@pytest.mark.parametrize("mult,pct", [(1.31, 31.0), (0.0505, -94.95), (2.4, 140.0)])
def test_breaches(mult, pct):
    f = _frame([100.0, 100.0 * mult])
    r = integrity.assess_bars(f)
    assert r["status"] == "anomalies_found"
    (b,) = r["limit_breaches"]
    assert b["date"] == f.index[1].date().isoformat()
    assert b["prev_close"] == 100.0
    assert b["change_pct"] == pytest.approx(pct, abs=0.01)


def test_first_row_never_breaches():
    assert integrity.assess_bars(_frame([5.0]))["limit_breaches"] == []


def test_nan_close_skipped():
    assert integrity.assess_bars(_frame([100.0, np.nan, 500.0]))["limit_breaches"] == []


def test_zero_prev_close_skipped():
    assert integrity.assess_bars(_frame([0.0, 500.0]))["limit_breaches"] == []


def test_zero_volume_dates():
    f = _frame([100.0, 100.0, 100.0], volumes=[10, 0, np.nan])
    r = integrity.assess_bars(f)
    assert r["zero_volume_dates"] == [f.index[1].date().isoformat()]
    assert r["status"] == "anomalies_found"


def test_long_gap_boundary():
    r15 = integrity.assess_bars(_frame([1.0, 1.0], dates=["2026-01-01", "2026-01-16"]))
    assert r15["long_gaps"] == [{"from": "2026-01-01", "to": "2026-01-16", "calendar_days": 15}]
    assert r15["status"] == "anomalies_found"
    r14 = integrity.assess_bars(_frame([1.0, 1.0], dates=["2026-01-01", "2026-01-15"]))
    assert r14["long_gaps"] == []


def test_empty_frame():
    r = integrity.assess_bars(_frame([]))
    assert r == {
        "status": "ok",
        "bars_checked": 0,
        "checked_through": None,
        "limit_breaches": [],
        "zero_volume_dates": [],
        "long_gaps": [],
    }


def test_affected_indicators_breach_on_last_bar():
    f = _frame([100.0] * 10 + [10.0])
    r = integrity.assess_bars(f)
    assert integrity.affected_indicators(f, r["limit_breaches"]) == sorted(integrity.INDICATOR_WINDOWS)


def test_affected_indicators_breach_30_bars_back():
    f = _frame([100.0] * 5 + [10.0] * 31)
    r = integrity.assess_bars(f)
    assert len(r["limit_breaches"]) == 1
    # breach bar is index 5; 36 bars total -> distance 30
    assert integrity.affected_indicators(f, r["limit_breaches"]) == sorted(["sma_60", "sma_120", "macd"])


def test_affected_indicators_no_breach():
    assert integrity.affected_indicators(_frame([1.0, 1.0]), []) == []


def test_assess_single_move():
    b = integrity.assess_single_move(100.0, 50.0, "2026-10-07")
    assert b["status"] == "anomalies_found"
    assert b["bars_checked"] == 1 and b["checked_through"] == "2026-10-07"
    assert b["limit_breaches"] == [
        {"date": "2026-10-07", "prev_close": 100.0, "close": 50.0, "change_pct": -50.0}
    ]
    assert integrity.assess_single_move(100.0, 101.0, "2026-10-07")["status"] == "ok"
    for prev, price in [(None, 5.0), (0.0, 5.0), (100.0, None)]:
        r = integrity.assess_single_move(prev, price, "2026-10-07")
        assert r["status"] == "ok" and r["limit_breaches"] == []


def test_notice_only_via_helper_and_only_on_anomalies():
    ok = integrity.assess_bars(_frame([100.0, 101.0]))
    bad = integrity.assess_bars(_frame([100.0, 10.0]))
    assert "notice" not in ok and "notice" not in bad
    assert "notice" not in integrity.assess_single_move(100.0, 10.0, "2026-10-07")
    assert "notice" not in integrity.with_notice(ok)
    assert integrity.with_notice(bad)["notice"] == integrity.PRICE_INTEGRITY_NOTICE
    assert "notice" not in bad  # helper copies


def test_integrity_module_does_not_import_io_layers():
    tree = ast.parse(inspect.getsource(integrity))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module.split(".")[0] if node.module else "")
            imported.update(a.name for a in node.names if node.level)
    banned = {"yfinance", "session", "cache", "yf_gateway", "mcp", "server", ""}
    assert not (imported & banned), imported & banned


def _patch_view(monkeypatch, module, view):
    monkeypatch.setattr(module.session, "get_session_view", lambda yf_symbol, period: view)


def test_price_history_has_integrity_block(monkeypatch):
    f = _frame([100.0, 10.0, 11.0])
    _patch_view(monkeypatch, market_data, SessionView(f, f, "closed", "CLOSED"))
    r = market_data.get_price_history("005930.KS")
    assert r["price_integrity"]["status"] == "anomalies_found"
    assert r["price_integrity"]["notice"] == integrity.PRICE_INTEGRITY_NOTICE
    assert r["price_integrity"]["bars_checked"] == 3


def test_price_history_in_progress_checks_only_completed(monkeypatch):
    f = _frame([100.0, 101.0, 10.0])  # last (in-progress) bar is the breach
    _patch_view(monkeypatch, market_data, SessionView(f, f.iloc[:-1], "in_progress", "REGULAR"))
    r = market_data.get_price_history("005930.KS")
    assert r["price_integrity"]["status"] == "ok"
    assert "notice" not in r["price_integrity"]
    assert r["price_integrity"]["bars_checked"] == 2


def test_snapshot_has_integrity_block_closed(monkeypatch):
    f = _frame([100.0, 10.0], dates=["2026-10-05", "2026-10-06"])
    _patch_view(monkeypatch, market_data, SessionView(f, f, "closed", "CLOSED"))
    monkeypatch.setattr(market_data.session, "fetch_info", lambda s: {"previousClose": 100.0})
    pi = market_data.get_price_snapshot("005930.KS")["price_integrity"]
    assert pi["status"] == "anomalies_found"
    assert pi["checked_through"] == "2026-10-06"
    assert "notice" in pi


def test_snapshot_has_integrity_block_in_progress(monkeypatch):
    f = _frame([100.0, 101.0], dates=["2026-10-05", "2026-10-06"])
    _patch_view(monkeypatch, market_data, SessionView(f, f.iloc[:-1], "in_progress", "REGULAR"))
    monkeypatch.setattr(market_data.session, "fetch_info", lambda s: {"previousClose": 100.0})
    pi = market_data.get_price_snapshot("005930.KS")["price_integrity"]
    assert pi["status"] == "ok"
    assert "notice" not in pi
    assert len(pi["checked_through"]) == 10


def test_indicators_has_integrity_block_and_affected(monkeypatch):
    closes = [100.0 + (i % 3) for i in range(130)] + [10.0] + [10.0 + (i % 3) for i in range(10)]
    f = _frame(closes)
    _patch_view(monkeypatch, indicators, SessionView(f, f, "closed", "CLOSED"))
    pi = indicators.get_technical_indicators("005930.KS")["price_integrity"]
    assert pi["status"] == "anomalies_found"
    assert "notice" in pi
    assert "sma_20" in pi["affected_indicators"] and "sma_120" in pi["affected_indicators"]
    assert "sma_5" not in pi["affected_indicators"]  # breach is 10 bars back


def test_indicators_clean_has_empty_affected(monkeypatch):
    f = _frame([100.0 + (i % 3) for i in range(30)])
    _patch_view(monkeypatch, indicators, SessionView(f, f, "closed", "CLOSED"))
    pi = indicators.get_technical_indicators("005930.KS")["price_integrity"]
    assert pi["status"] == "ok"
    assert pi["affected_indicators"] == []


def test_tool_descriptions_mention_price_integrity():
    tools = {t.name: t.description or "" for t in asyncio.run(server.mcp.list_tools())}
    for name in ("get_price_snapshot", "get_price_history", "get_technical_indicators"):
        assert "price_integrity" in tools[name]
