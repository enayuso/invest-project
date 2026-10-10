"""app.backtest.parallel.pick_batch_tickers — pure selection/validation logic.
No subprocess spawning tested here (that's thin plumbing over
asyncio.create_subprocess_exec, not worth mocking at the OS-process level)."""

from __future__ import annotations

import pytest

from app.backtest import parallel
from app.config import settings


def test_pick_batch_tickers_explicit_list():
    result = parallel.pick_batch_tickers("AAPL, MSFT , NVDA", None)
    assert result == ["AAPL", "MSFT", "NVDA"]


def test_pick_batch_tickers_rejects_explicit_duplicates():
    with pytest.raises(ValueError, match="duplicate ticker"):
        parallel.pick_batch_tickers("AAPL,MSFT,AAPL", None)


def test_pick_batch_tickers_random_n_picks_distinct_from_trade_allowlist(monkeypatch):
    monkeypatch.setattr(settings, "trade_allowlist_domestic", [])
    monkeypatch.setattr(settings, "trade_allowlist_overseas", ["AAPL", "MSFT", "NVDA", "GOOGL"])
    result = parallel.pick_batch_tickers(None, 3)
    assert len(result) == 3
    assert len(set(result)) == 3
    assert all(t in ["AAPL", "MSFT", "NVDA", "GOOGL"] for t in result)


def test_pick_batch_tickers_random_n_exceeding_trade_allowlist_raises(monkeypatch):
    monkeypatch.setattr(settings, "trade_allowlist_domestic", [])
    monkeypatch.setattr(settings, "trade_allowlist_overseas", ["AAPL", "MSFT"])
    with pytest.raises(ValueError, match="exceeds distinct trade_allowlist size"):
        parallel.pick_batch_tickers(None, 5)


def test_pick_batch_tickers_random_n_dedupes_trade_allowlist_first(monkeypatch):
    # A trade_allowlist with accidental duplicate entries shouldn't make --random-n
    # look larger than it safely can be.
    monkeypatch.setattr(settings, "trade_allowlist_domestic", [])
    monkeypatch.setattr(settings, "trade_allowlist_overseas", ["AAPL", "AAPL", "MSFT"])
    result = parallel.pick_batch_tickers(None, 2)
    assert sorted(result) == ["AAPL", "MSFT"]


# ---------------------------------------------------------------------------
# overall_exit_code — 2026-09-11 fix: _main() used to always exit 0 even
# when every launched group failed, so app.backtest.atr_sweep reported a
# candidate whose every group crashed as "OK".
# ---------------------------------------------------------------------------


def test_overall_exit_code_all_zero_is_zero():
    assert parallel.overall_exit_code([0, 0, 0]) == 0


def test_overall_exit_code_any_nonzero_is_one():
    assert parallel.overall_exit_code([0, 1, 0]) == 1
    assert parallel.overall_exit_code([2, 3]) == 1


def test_overall_exit_code_negative_signal_code_counts_as_failure():
    assert parallel.overall_exit_code([-9]) == 1  # SIGKILL'd child


def test_overall_exit_code_empty_is_zero():
    assert parallel.overall_exit_code([]) == 0


def test_overall_exit_code_accepts_a_generator():
    assert parallel.overall_exit_code(c for c in [0, 0]) == 0


def test_stdout_log_path_splits_by_market_and_replaces_dots(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # default_log_path creates real directories — keep it contained
    domestic = parallel._stdout_log_path("005930.KS")
    overseas = parallel._stdout_log_path("AAPL")
    assert domestic.replace("\\", "/") == "logs/domestic/parallel_005930_KS.log"
    assert overseas.replace("\\", "/") == "logs/overseas/parallel_AAPL.log"
