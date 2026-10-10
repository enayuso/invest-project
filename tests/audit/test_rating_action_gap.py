"""app.audit.stats.compute_rating_action_gap — asyncpg faked, no real
Postgres. 2026-09-01 audit #10."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from app.audit import stats
from tests.audit.test_audit_stats import FakeConn, FakePool, _async_return


def _row(id, ticker, rating, action_label, decision_text="thesis", trade_date=None):
    return {
        "id": id, "ticker": ticker, "rating": rating, "action_label": action_label,
        "decision_text": decision_text, "trade_date": trade_date or date.today(),
    }


async def _run(monkeypatch, rows, **kw):
    conn = FakeConn(fetch_result=rows)
    monkeypatch.setattr(stats, "get_pool", lambda: _async_return(FakePool(conn)))
    return await stats.compute_rating_action_gap(**kw)


async def test_classifies_aligned_gap_and_unrecorded(monkeypatch):
    rows = [
        _row(1, "A.KS", "Buy", "Buy"),                # aligned
        _row(2, "B.KS", "Overweight", "Increase"),    # aligned
        _row(3, "C.KS", "Buy", "Hold",
             "thesis\n\n[EXECUTION NOTE: No position change despite a directional rating: floor(100 / 1,431,500.00) = 0 shares.]"),
        _row(4, "D.KS", "Sell", "Hold", "thesis"),    # gap, no note
        _row(5, "E.KS", "Buy", None, "thesis"),       # unrecorded
    ]
    r = await _run(monkeypatch, rows)
    assert r.total_decisions == 5
    assert r.gap_count == 2
    assert r.unrecorded_count == 1
    assert ("Buy", "Hold") in r.gap_pairs and ("Sell", "Hold") in r.gap_pairs
    # number-normalized note grouping
    assert any("floor(N / N) = N shares" in k for k in r.gap_reason_counts)
    assert r.gap_reason_counts["(no execution note)"] == 1


async def test_multiple_execution_notes_are_joined(monkeypatch):
    rows = [_row(1, "A.KS", "Buy", "Hold",
                 "t\n\n[EXECUTION NOTE: 1% risk cap reduced the intended add from 3 to 0 shares.]"
                 "\n\n[EXECUTION NOTE: No position change despite a directional rating: floor(1 / 2) = 0 shares.]")]
    r = await _run(monkeypatch, rows)
    key = next(iter(r.gap_reason_counts))
    assert " | " in key
    assert "risk cap" in key and "No position change" in key


async def test_window_and_dedup_are_reflected_in_the_sql(monkeypatch):
    conn = FakeConn(fetch_result=[])
    monkeypatch.setattr(stats, "get_pool", lambda: _async_return(FakePool(conn)))
    await stats.compute_rating_action_gap(days=1, market="domestic", scope="live", account_mode="real")
    query, args = conn.queries[0]
    assert "DISTINCT ON (ticker, trade_date)" in query
    assert "status = 'resolved'" not in query  # deliberately not gated
    assert "ticker LIKE '%.KS'" in query and "is_backtest = false" in query
    assert date.today() - timedelta(days=1) in args


async def test_render_empty_window(monkeypatch):
    r = await _run(monkeypatch, [])
    out = stats.render_rating_action_gap(r)
    assert "No decisions in this window." in out


async def test_render_lists_the_regression_signal_line(monkeypatch):
    rows = [_row(1, "A.KS", "Buy", "Hold", "t")]
    r = await _run(monkeypatch, rows)
    out = stats.render_rating_action_gap(r)
    assert "(no execution note)" in out
    assert "regression signal for the 2026-09-01 audit" in out
    assert "id=1  " in out
