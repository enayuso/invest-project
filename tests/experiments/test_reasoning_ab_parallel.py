"""app.experiments.reasoning_ab_parallel.pick_groups — pure selection/
validation logic. No subprocess spawning tested here (same "thin plumbing
over asyncio.create_subprocess_exec, not worth mocking at the OS-process
level" convention as test_parallel.py/test_parallel_portfolio.py)."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.experiments import reasoning_ab_parallel as rap


def test_pick_groups_round_robin_partitions_tickers():
    result = rap.pick_groups("A,B,C,D,E", 2)
    assert result == [["A", "C", "E"], ["B", "D"]]


def test_pick_groups_every_ticker_appears_in_exactly_one_group():
    result = rap.pick_groups("A,B,C,D,E,F,G", 3)
    all_tickers = [t for g in result for t in g]
    assert sorted(all_tickers) == ["A", "B", "C", "D", "E", "F", "G"]
    assert len(all_tickers) == len(set(all_tickers))  # no ticker duplicated across groups


def test_pick_groups_rejects_duplicate_tickers():
    """The core enforcement of the "a ticker's runs must never split across
    groups" rule — a duplicate would let two children's clear_backtest_
    positions/clear_backtest_memory calls for the same ticker race."""
    with pytest.raises(ValueError, match="duplicate ticker"):
        rap.pick_groups("005930.KS,AAPL,005930.KS", 2)


def test_pick_groups_rejects_zero_or_negative_groups():
    with pytest.raises(ValueError, match="num-groups must be >= 1"):
        rap.pick_groups("A,B", 0)


def test_pick_groups_rejects_more_groups_than_tickers():
    with pytest.raises(ValueError, match="exceeds distinct ticker count"):
        rap.pick_groups("A,B", 5)


def test_pick_groups_strips_whitespace():
    result = rap.pick_groups(" A , B ", 1)
    assert result == [["A", "B"]]


def test_pick_groups_uneven_split_has_no_trailing_empty_group():
    # 5 tickers into 3 groups: round-robin gives 2/2/1, never a 4th empty list.
    result = rap.pick_groups("A,B,C,D,E", 3)
    assert len(result) == 3
    assert all(result)


# ---------------------------------------------------------------------------
# _stdout_log_path — lives inside the shared --out-dir (not a separate logs/
# path keyed by label/pid), so uniqueness comes from out_dir itself.
# ---------------------------------------------------------------------------


def test_stdout_log_path_is_nested_under_the_given_out_dir():
    out_dir = Path("results") / "reasoning_ab" / "20260908T000000"
    path = rap._stdout_log_path(out_dir, 1)
    assert path == out_dir / "group1.log"


def test_stdout_log_path_differs_by_group_index_for_the_same_out_dir():
    out_dir = Path("results") / "reasoning_ab" / "20260908T000000"
    a = rap._stdout_log_path(out_dir, 0)
    b = rap._stdout_log_path(out_dir, 1)
    assert a != b
