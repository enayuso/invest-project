"""app.backtest.parallel_portfolio.pick_groups — pure selection/validation
logic. No subprocess spawning tested here (same "thin plumbing over
asyncio.create_subprocess_exec, not worth mocking at the OS-process level"
convention as test_parallel.py)."""

from __future__ import annotations

import pytest

from app.backtest import parallel_portfolio


def test_pick_groups_round_robin_partitions_explicit_tickers():
    result = parallel_portfolio.pick_groups("A,B,C,D,E", 2, trade_allowlist=[])
    assert result == [["A", "C", "E"], ["B", "D"]]


def test_pick_groups_defaults_to_trade_allowlist_when_no_tickers_given():
    result = parallel_portfolio.pick_groups(None, 2, trade_allowlist=["A", "B", "C"])
    assert sum(len(g) for g in result) == 3
    assert {t for g in result for t in g} == {"A", "B", "C"}


def test_pick_groups_every_ticker_appears_in_exactly_one_group():
    result = parallel_portfolio.pick_groups("A,B,C,D,E,F,G", 3, trade_allowlist=[])
    all_tickers = [t for g in result for t in g]
    assert sorted(all_tickers) == ["A", "B", "C", "D", "E", "F", "G"]
    assert len(all_tickers) == len(set(all_tickers))  # no ticker duplicated across groups


def test_pick_groups_rejects_duplicate_tickers():
    with pytest.raises(ValueError, match="duplicate ticker"):
        parallel_portfolio.pick_groups("A,B,A", 2, trade_allowlist=[])


def test_pick_groups_rejects_zero_or_negative_groups():
    with pytest.raises(ValueError, match="num-groups must be >= 1"):
        parallel_portfolio.pick_groups("A,B", 0, trade_allowlist=[])


def test_pick_groups_rejects_more_groups_than_tickers():
    with pytest.raises(ValueError, match="exceeds distinct ticker count"):
        parallel_portfolio.pick_groups("A,B", 5, trade_allowlist=[])


def test_pick_groups_strips_whitespace():
    result = parallel_portfolio.pick_groups(" A , B ", 1, trade_allowlist=[])
    assert result == [["A", "B"]]


# ---------------------------------------------------------------------------
# _stdout_log_path — 2026-08-12 collision fix: two separate invocations
# sharing the same --num-groups count (e.g. a domestic run and an overseas
# run launched back to back) must never resolve to the same physical file.
# ---------------------------------------------------------------------------


def test_stdout_log_path_differs_by_label_for_the_same_group_index():
    domestic = parallel_portfolio._stdout_log_path("baseline-domestic-20260812", 0)
    overseas = parallel_portfolio._stdout_log_path("baseline-overseas-20260812", 0)
    assert domestic != overseas


def test_stdout_log_path_is_stable_for_the_same_label_and_group():
    a = parallel_portfolio._stdout_log_path("baseline-domestic-20260812", 1)
    b = parallel_portfolio._stdout_log_path("baseline-domestic-20260812", 1)
    assert a == b


def test_stdout_log_path_falls_back_to_pid_without_a_label(monkeypatch):
    monkeypatch.setattr(parallel_portfolio.os, "getpid", lambda: 4242)
    path = parallel_portfolio._stdout_log_path(None, 0)
    assert "pid4242" in path
    assert "group0" in path


# ---------------------------------------------------------------------------
# overall_exit_code re-export — parallel_portfolio's own _main uses this to
# decide its process exit code (see test_parallel.py for the pure-logic
# tests of the function itself).
# ---------------------------------------------------------------------------


def test_overall_exit_code_is_imported_from_parallel():
    results = [(0, ["A"], 0, 1.0), (1, ["B"], 1, 1.0)]
    assert parallel_portfolio.overall_exit_code(code for _, _, code, _ in results) == 1
