"""app.backtest.atr_sweep_report — list_experiments faked, no real DB.

2026-09-11 fix: --limit used to bound the most recent rows across the WHOLE
backtest_runs table, not just this sweep's own label prefix, so unrelated
experiments recorded since could push this sweep's own older rows out of the
window with no indication anything was dropped. The report now passes
label_prefix through to list_experiments so the SQL-level LIMIT is scoped to
this sweep."""

from __future__ import annotations

import sys

import pytest

from app.backtest import atr_sweep_report


async def test_main_calls_list_experiments_with_scoped_label_prefix(monkeypatch, capsys):
    captured = {}

    async def fake_list_experiments(limit=20, label_prefix=None):
        captured["limit"] = limit
        captured["label_prefix"] = label_prefix
        return []

    monkeypatch.setattr(atr_sweep_report, "list_experiments", fake_list_experiments)
    monkeypatch.setattr(
        sys, "argv", ["atr_sweep_report", "--label-prefix", "trailing-sweep", "--limit", "50"]
    )
    await atr_sweep_report._main()

    assert captured["limit"] == 50
    # Must match the exact marker _atr_value_from_label builds internally
    # (marker = f"{prefix}-atr") -- the SQL filter and the Python-side
    # filter must agree on what counts as "this sweep's rows".
    assert captured["label_prefix"] == "trailing-sweep-atr"
