"""app.backtest.harness — select_trading_dates (pure) and _process_ticker_date
(every I/O dependency faked: pipeline, ledger, reflection, memory store). No
network/DB/LLM (F-stage). run_backtest's own orchestration is exercised via
one heavily-mocked end-to-end test."""

from __future__ import annotations

import logging
from datetime import date, timedelta

import pytest

from app.backtest import harness
from app.config import backtest_scope, settings
from app.execution import ledger
from app.execution.ledger import Position, RebalanceResult
from app.state import PortfolioDecision, PortfolioRating


def test_configure_cli_logging_routes_warnings_to_file(tmp_path):
    log_path = tmp_path / "backtest.log"
    root = logging.getLogger()
    prev_handlers, prev_level = list(root.handlers), root.level
    try:
        harness.configure_cli_logging(str(log_path))
        logging.getLogger("app.agents.untrusted_content").warning("some flagged text")
        for h in root.handlers:
            h.flush()
        assert log_path.exists()
        assert "some flagged text" in log_path.read_text(encoding="utf-8")
    finally:
        for h in list(root.handlers):
            root.removeHandler(h)
            h.close()
        for h in prev_handlers:
            root.addHandler(h)
        root.setLevel(prev_level)


def test_market_subdir_classifies_by_ks_suffix():
    assert harness.market_subdir("005930.KS") == "domestic"
    assert harness.market_subdir("AAPL") == "overseas"


def test_default_log_path_splits_by_market(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # creates real directories — keep it contained
    domestic = harness.default_log_path("005930.KS", "backtest.log")
    overseas = harness.default_log_path("AAPL", "backtest.log")
    assert domestic.replace("\\", "/") == "logs/domestic/backtest.log"
    assert overseas.replace("\\", "/") == "logs/overseas/backtest.log"
    assert (tmp_path / "logs" / "domestic").is_dir()
    assert (tmp_path / "logs" / "overseas").is_dir()


def test_pick_random_date_range_respects_span_bounds():
    for _ in range(20):
        start, end = harness.pick_random_date_range(min_span_days=60, max_span_days=120, max_lookback_days=730)
        assert start < end
        assert 60 <= (end - start).days <= 120
        assert end <= date.today()
        assert start >= date.today() - timedelta(days=730)


def test_pick_random_ticker_and_range_picks_from_trade_allowlist(monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "trade_allowlist_domestic", [])
    monkeypatch.setattr(settings, "trade_allowlist_overseas", ["AAA", "BBB"])
    for _ in range(20):
        ticker, start, end = harness.pick_random_ticker_and_range()
        assert ticker in ("AAA", "BBB")
        assert start < end


def test_select_trading_dates_subsamples_at_cadence():
    prices = {date(2026, 1, d): 100.0 + d for d in range(1, 11)}
    result = harness.select_trading_dates(prices, cadence_days=3)
    assert result == [date(2026, 1, 1), date(2026, 1, 4), date(2026, 1, 7), date(2026, 1, 10)]


def test_select_trading_dates_floors_cadence_at_one():
    prices = {date(2026, 1, 1): 100.0, date(2026, 1, 2): 101.0}
    result = harness.select_trading_dates(prices, cadence_days=0)
    assert result == [date(2026, 1, 1), date(2026, 1, 2)]


def _decision(rating=PortfolioRating.BUY) -> PortfolioDecision:
    return PortfolioDecision(rating=rating, executive_summary="s", investment_thesis="t")


@pytest.fixture(autouse=True)
def _stub_run_pipeline_dependencies(monkeypatch):
    async def fake_get_open_stop_loss(ticker):
        return None

    async def fake_get_open_stop_loss_entry_time(ticker):
        # None by default -> the gap check falls back to the plain
        # check_and_apply_stop_loss path unchanged, matching every existing
        # test in this file written before the trailing-stop replay
        # existed. Tests that want the trailing path override this
        # explicitly (see the dedicated section below).
        return None

    monkeypatch.setattr(harness, "get_open_stop_loss", fake_get_open_stop_loss)
    monkeypatch.setattr(harness, "get_open_stop_loss_entry_time", fake_get_open_stop_loss_entry_time)


async def test_process_ticker_date_skips_gap_check_when_no_prev(monkeypatch):
    calls = {"gap_check": False}

    async def should_not_be_called(ticker):
        calls["gap_check"] = True
        raise AssertionError("get_open_stop_loss should not matter without prev")

    async def fake_run_pipeline(ticker, trade_date, as_of=None, portfolio_cap_override=None, profile=None):
        class FakeState:
            final_decision = _decision()
            rebalance_adjustment_note = None
            decision_id = 1

        return FakeState()

    monkeypatch.setattr(harness, "run_pipeline", fake_run_pipeline)

    record = await harness._process_ticker_date(
        "AAPL", date(2026, 7, 16), None, {date(2026, 7, 16): 150.0}, {date(2026, 7, 16): 400.0}, "SPY"
    )
    assert record.trade_date == date(2026, 7, 16)
    assert record.close == 150.0
    assert record.decision.rating == PortfolioRating.BUY


async def test_process_ticker_date_applies_gap_stop_loss_check_when_prev_exists(monkeypatch):
    checked = {}

    async def fake_get_open_stop_loss(ticker):
        return ("buy", 140.0)

    async def fake_get_daily_low_series(ticker, start, end):
        return {date(2026, 7, 17): 145.0, date(2026, 7, 18): 138.0}

    async def fake_check_and_apply_stop_loss(ticker, gap_lows, stop_loss, reason=""):
        checked["gap_lows"] = gap_lows
        checked["stop_loss"] = stop_loss

    async def fake_reflect(*args, **kwargs):
        return "reflection text"

    async def fake_resolve_outcome(*args, **kwargs):
        checked["resolved"] = True

    async def fake_run_pipeline(ticker, trade_date, as_of=None, portfolio_cap_override=None, profile=None):
        class FakeState:
            final_decision = _decision()
            rebalance_adjustment_note = None
            decision_id = 1

        return FakeState()

    monkeypatch.setattr(harness, "get_open_stop_loss", fake_get_open_stop_loss)
    monkeypatch.setattr(harness, "get_daily_low_series", fake_get_daily_low_series)
    monkeypatch.setattr(harness, "check_and_apply_stop_loss", fake_check_and_apply_stop_loss)
    monkeypatch.setattr(harness, "reflect", fake_reflect)
    monkeypatch.setattr(harness, "resolve_outcome", fake_resolve_outcome)
    monkeypatch.setattr(harness, "run_pipeline", fake_run_pipeline)

    prev = harness.BacktestRecord(trade_date=date(2026, 7, 16), decision=_decision(), close=150.0, decision_id=7)
    prices = {date(2026, 7, 16): 150.0, date(2026, 7, 19): 155.0}
    benchmark_prices = {date(2026, 7, 16): 400.0, date(2026, 7, 19): 405.0}

    record = await harness._process_ticker_date("AAPL", date(2026, 7, 19), prev, prices, benchmark_prices, "SPY")

    # Gap lows filtered to strictly-after prev.trade_date.
    assert checked["gap_lows"] == {date(2026, 7, 17): 145.0, date(2026, 7, 18): 138.0}
    assert checked["stop_loss"] == 140.0
    assert checked["resolved"] is True
    assert record.close == 155.0


# ---------------------------------------------------------------------------
# Fix 1 (2026-08-27): resolve_outcome is now scoped by decision_id (the
# memory_entries.id from `prev`'s own pending row) rather than by
# (ticker, trade_date), which more than one row can share — see
# app.memory.store.resolve_outcome's own docstring. BacktestRecord carries
# this forward as `decision_id` (from PipelineState.decision_id, set by
# store_pending); a day store_pending failed for leaves it None, which must
# skip the resolve call rather than fall back to the unsafe natural key.
# ---------------------------------------------------------------------------


async def test_process_ticker_date_passes_prev_decision_id_to_resolve_outcome(monkeypatch):
    captured = {}

    async def fake_get_open_stop_loss(ticker):
        return None

    async def fake_reflect(*args, **kwargs):
        return "reflection text"

    async def fake_resolve_outcome(decision_id, ticker, trade_date_, *rest, **kwargs):
        captured["decision_id"] = decision_id

    async def fake_run_pipeline(ticker, trade_date, as_of=None, portfolio_cap_override=None, profile=None):
        class FakeState:
            final_decision = _decision()
            rebalance_adjustment_note = None
            decision_id = 99

        return FakeState()

    monkeypatch.setattr(harness, "get_open_stop_loss", fake_get_open_stop_loss)
    monkeypatch.setattr(harness, "reflect", fake_reflect)
    monkeypatch.setattr(harness, "resolve_outcome", fake_resolve_outcome)
    monkeypatch.setattr(harness, "run_pipeline", fake_run_pipeline)

    prev = harness.BacktestRecord(trade_date=date(2026, 7, 16), decision=_decision(), close=150.0, decision_id=42)
    prices = {date(2026, 7, 16): 150.0, date(2026, 7, 19): 155.0}
    benchmark_prices = {date(2026, 7, 16): 400.0, date(2026, 7, 19): 405.0}

    record = await harness._process_ticker_date("AAPL", date(2026, 7, 19), prev, prices, benchmark_prices, "SPY")

    # prev's OWN decision_id (42) is what gets resolved — never the new
    # record's decision_id (99), which belongs to a still-pending row.
    assert captured["decision_id"] == 42
    assert record.decision_id == 99


# ---------------------------------------------------------------------------
# 2026-09-01 PIT fix: resolve_outcome's `resolved_on` must be the REPLAYED
# date (record.trade_date), never date.today()/local_today() — a wall-clock
# stamp would make get_similar_context's `resolved_on < as_of` bound exclude
# every backtest-resolved row from every later replay in the same run. See
# app.memory.db's `resolved_on` column comment for the full incident.
# ---------------------------------------------------------------------------


async def test_process_ticker_date_passes_replayed_date_as_resolved_on(monkeypatch):
    captured = {}

    async def fake_get_open_stop_loss(ticker):
        return None

    async def fake_reflect(*args, **kwargs):
        return "reflection text"

    async def fake_resolve_outcome(*args, resolved_on=None, **kwargs):
        captured["resolved_on"] = resolved_on

    async def fake_run_pipeline(ticker, trade_date, as_of=None, portfolio_cap_override=None, profile=None):
        class FakeState:
            final_decision = _decision()
            rebalance_adjustment_note = None
            decision_id = 99

        return FakeState()

    monkeypatch.setattr(harness, "get_open_stop_loss", fake_get_open_stop_loss)
    monkeypatch.setattr(harness, "reflect", fake_reflect)
    monkeypatch.setattr(harness, "resolve_outcome", fake_resolve_outcome)
    monkeypatch.setattr(harness, "run_pipeline", fake_run_pipeline)

    prev = harness.BacktestRecord(trade_date=date(2026, 7, 16), decision=_decision(), close=150.0, decision_id=42)
    prices = {date(2026, 7, 16): 150.0, date(2026, 7, 19): 155.0}
    benchmark_prices = {date(2026, 7, 16): 400.0, date(2026, 7, 19): 405.0}

    await harness._process_ticker_date("AAPL", date(2026, 7, 19), prev, prices, benchmark_prices, "SPY")

    # The replayed date this call is FOR (2026-07-19), not prev's own
    # trade_date (2026-07-16) and not a wall-clock date.today() — this is
    # the date prev's outcome actually became knowable in simulated time.
    assert captured["resolved_on"] == date(2026, 7, 19)


async def test_process_ticker_date_skips_resolve_when_prev_decision_id_is_none(monkeypatch, caplog):
    """store_pending failing for `prev`'s own day (DB hiccup) leaves
    `decision_id` None — must not crash, and must not fall back to
    resolving by (ticker, trade_date), which is exactly the unsafe natural
    key this fix removed."""

    async def fake_get_open_stop_loss(ticker):
        return None

    async def should_not_be_called(*args, **kwargs):
        raise AssertionError("resolve_outcome must not run with no decision_id to scope it by")

    async def fail_if_called(*args, **kwargs):
        raise AssertionError("reflect must not run either — there is nothing to resolve")

    async def fake_run_pipeline(ticker, trade_date, as_of=None, portfolio_cap_override=None, profile=None):
        class FakeState:
            final_decision = _decision()
            rebalance_adjustment_note = None
            decision_id = 99

        return FakeState()

    monkeypatch.setattr(harness, "get_open_stop_loss", fake_get_open_stop_loss)
    monkeypatch.setattr(harness, "reflect", fail_if_called)
    monkeypatch.setattr(harness, "resolve_outcome", should_not_be_called)
    monkeypatch.setattr(harness, "run_pipeline", fake_run_pipeline)

    prev = harness.BacktestRecord(trade_date=date(2026, 7, 16), decision=_decision(), close=150.0, decision_id=None)
    prices = {date(2026, 7, 16): 150.0, date(2026, 7, 19): 155.0}
    benchmark_prices = {date(2026, 7, 16): 400.0, date(2026, 7, 19): 405.0}

    with caplog.at_level("WARNING", logger="app.backtest.harness"):
        record = await harness._process_ticker_date("AAPL", date(2026, 7, 19), prev, prices, benchmark_prices, "SPY")

    assert record.close == 155.0  # the new record itself still gets built and returned
    assert any("no decision_id" in r.getMessage() for r in caplog.records)


async def test_process_ticker_date_skips_resolve_on_zero_prev_close(monkeypatch):
    # Regression: a vendor data glitch (literal 0.0 close) used to raise
    # ZeroDivisionError computing raw_return, aborting the whole backtest.
    async def should_not_be_called(*args, **kwargs):
        raise AssertionError("resolve_outcome must not run when the return calc is unsafe")

    async def fake_reflect(*args, **kwargs):
        raise AssertionError("reflect must not run when the return calc is unsafe")

    async def fake_run_pipeline(ticker, trade_date, as_of=None, portfolio_cap_override=None, profile=None):
        class FakeState:
            final_decision = _decision()
            rebalance_adjustment_note = None
            decision_id = 1

        return FakeState()

    async def fake_get_open_stop_loss(ticker):
        return None

    monkeypatch.setattr(harness, "get_open_stop_loss", fake_get_open_stop_loss)
    monkeypatch.setattr(harness, "resolve_outcome", should_not_be_called)
    monkeypatch.setattr(harness, "reflect", fake_reflect)
    monkeypatch.setattr(harness, "run_pipeline", fake_run_pipeline)

    prev = harness.BacktestRecord(trade_date=date(2026, 7, 16), decision=_decision(), close=0.0, decision_id=7)
    prices = {date(2026, 7, 16): 0.0, date(2026, 7, 19): 155.0}
    benchmark_prices = {date(2026, 7, 16): 400.0, date(2026, 7, 19): 405.0}

    record = await harness._process_ticker_date("AAPL", date(2026, 7, 19), prev, prices, benchmark_prices, "SPY")
    assert record.close == 155.0  # the new record itself still gets built and returned


async def test_process_ticker_date_skips_resolve_on_missing_earlier_benchmark_price(monkeypatch):
    # price_on_or_before raises KeyError when the benchmark has no trading
    # day at or before prev.trade_date at all.
    async def should_not_be_called(*args, **kwargs):
        raise AssertionError("resolve_outcome must not run when the benchmark price is unavailable")

    async def fake_run_pipeline(ticker, trade_date, as_of=None, portfolio_cap_override=None, profile=None):
        class FakeState:
            final_decision = _decision()
            rebalance_adjustment_note = None
            decision_id = 1

        return FakeState()

    async def fake_get_open_stop_loss(ticker):
        return None

    monkeypatch.setattr(harness, "get_open_stop_loss", fake_get_open_stop_loss)
    monkeypatch.setattr(harness, "resolve_outcome", should_not_be_called)
    monkeypatch.setattr(harness, "run_pipeline", fake_run_pipeline)

    prev = harness.BacktestRecord(trade_date=date(2026, 7, 16), decision=_decision(), close=150.0, decision_id=7)
    prices = {date(2026, 7, 16): 150.0, date(2026, 7, 19): 155.0}
    benchmark_prices = {date(2026, 7, 20): 405.0}  # nothing on or before prev.trade_date

    record = await harness._process_ticker_date("AAPL", date(2026, 7, 19), prev, prices, benchmark_prices, "SPY")
    assert record.close == 155.0


async def test_process_ticker_date_gap_check_value_error_is_swallowed(monkeypatch):
    async def fake_get_open_stop_loss(ticker):
        return ("buy", 140.0)

    async def fake_get_daily_low_series(ticker, start, end):
        raise ValueError("no price history returned")

    async def fake_reflect(*args, **kwargs):
        return "reflection text"

    async def fake_resolve_outcome(*args, **kwargs):
        pass

    async def fake_run_pipeline(ticker, trade_date, as_of=None, portfolio_cap_override=None, profile=None):
        class FakeState:
            final_decision = _decision()
            rebalance_adjustment_note = None
            decision_id = 1

        return FakeState()

    monkeypatch.setattr(harness, "get_open_stop_loss", fake_get_open_stop_loss)
    monkeypatch.setattr(harness, "get_daily_low_series", fake_get_daily_low_series)
    monkeypatch.setattr(harness, "reflect", fake_reflect)
    monkeypatch.setattr(harness, "resolve_outcome", fake_resolve_outcome)
    monkeypatch.setattr(harness, "run_pipeline", fake_run_pipeline)

    prev = harness.BacktestRecord(trade_date=date(2026, 7, 16), decision=_decision(), close=150.0, decision_id=7)
    prices = {date(2026, 7, 16): 150.0, date(2026, 7, 19): 155.0}
    benchmark_prices = {date(2026, 7, 16): 400.0, date(2026, 7, 19): 405.0}

    # Must not raise despite get_daily_low_series failing.
    record = await harness._process_ticker_date("AAPL", date(2026, 7, 19), prev, prices, benchmark_prices, "SPY")
    assert record.close == 155.0


# ---------------------------------------------------------------------------
# Trailing-stop replay wiring — check_and_apply_trailing_stop's own math is
# covered by tests/execution/test_ledger.py; these confirm _process_ticker_
# date calls it with the right inputs and threads high_water/entry_price
# forward via BacktestRecord, entirely in memory (never touching the live
# trailing_stop_state table).
# ---------------------------------------------------------------------------


def _stub_common_pipeline(monkeypatch):
    async def fake_reflect(*args, **kwargs):
        return "reflection text"

    async def fake_resolve_outcome(*args, **kwargs):
        return None

    async def fake_run_pipeline(ticker, trade_date, as_of=None, portfolio_cap_override=None, profile=None):
        class FakeState:
            final_decision = _decision()
            rebalance_adjustment_note = None
            decision_id = 1

        return FakeState()

    monkeypatch.setattr(harness, "reflect", fake_reflect)
    monkeypatch.setattr(harness, "resolve_outcome", fake_resolve_outcome)
    monkeypatch.setattr(harness, "run_pipeline", fake_run_pipeline)


async def test_process_ticker_date_uses_trailing_check_when_entry_info_available(monkeypatch):
    _stub_common_pipeline(monkeypatch)

    async def fake_get_open_stop_loss(ticker):
        return date(2026, 7, 16), 80.0

    entry_time = object()  # opaque identity marker — see harness's own comment on this

    async def fake_entry_time(ticker):
        return entry_time, 100.0, 80.0

    async def fake_daily_lows(ticker, start, end):
        return {date(2026, 7, 17): 148.0, date(2026, 7, 18): 130.0}

    async def fake_daily_highs(ticker, start, end):
        return {date(2026, 7, 17): 150.0, date(2026, 7, 18): 132.0}

    async def fake_atr_pct(ticker, as_of=None):
        return 0.04

    captured = {}

    async def fake_trailing_check(ticker, gap_highs, gap_lows, pm_stop_loss, entry_price, high_water, trailing_pct, reason=""):
        captured.update(
            gap_highs=gap_highs, gap_lows=gap_lows, pm_stop_loss=pm_stop_loss,
            entry_price=entry_price, high_water=high_water, trailing_pct=trailing_pct,
        )
        return date(2026, 7, 18), 150.0

    async def should_not_be_called(*args, **kwargs):
        raise AssertionError("plain check_and_apply_stop_loss must not run when entry info is available")

    monkeypatch.setattr(harness, "get_open_stop_loss", fake_get_open_stop_loss)
    monkeypatch.setattr(harness, "get_open_stop_loss_entry_time", fake_entry_time)
    monkeypatch.setattr(harness, "get_daily_low_series", fake_daily_lows)
    monkeypatch.setattr(harness, "get_daily_high_series", fake_daily_highs)
    monkeypatch.setattr(harness, "get_atr_pct", fake_atr_pct)
    monkeypatch.setattr(harness, "check_and_apply_trailing_stop", fake_trailing_check)
    monkeypatch.setattr(harness, "check_and_apply_stop_loss", should_not_be_called)

    prev = harness.BacktestRecord(trade_date=date(2026, 7, 16), decision=_decision(), close=150.0, decision_id=7)
    prices = {date(2026, 7, 16): 150.0, date(2026, 7, 19): 155.0}
    benchmark_prices = {date(2026, 7, 16): 400.0, date(2026, 7, 19): 405.0}

    record = await harness._process_ticker_date("AAPL", date(2026, 7, 19), prev, prices, benchmark_prices, "SPY")

    # First time seeing this ticker's open stop -> high_water re-baselined to entry_price.
    assert captured["entry_price"] == 100.0
    assert captured["high_water"] == 100.0
    assert captured["gap_lows"] == {date(2026, 7, 17): 148.0, date(2026, 7, 18): 130.0}
    assert captured["gap_highs"] == {date(2026, 7, 17): 150.0, date(2026, 7, 18): 132.0}
    # trailing_high_water threaded forward onto the returned record for the NEXT gap.
    assert record.trailing_high_water == 150.0
    assert record.trailing_entry_price == 100.0


async def test_process_ticker_date_carries_high_water_forward_within_same_streak(monkeypatch):
    _stub_common_pipeline(monkeypatch)

    async def fake_get_open_stop_loss(ticker):
        return date(2026, 7, 16), 80.0

    same_entry_price = 100.0

    async def fake_entry_time(ticker):
        return object(), same_entry_price, 80.0  # entry_time identity irrelevant here — price is the reset key

    async def fake_daily_lows(ticker, start, end):
        return {date(2026, 7, 21): 160.0}

    async def fake_daily_highs(ticker, start, end):
        return {date(2026, 7, 21): 165.0}

    async def fake_atr_pct(ticker, as_of=None):
        return None

    captured = {}

    async def fake_trailing_check(ticker, gap_highs, gap_lows, pm_stop_loss, entry_price, high_water, trailing_pct, reason=""):
        captured["high_water_in"] = high_water
        return None, max(high_water, max(gap_highs.values()))

    monkeypatch.setattr(harness, "get_open_stop_loss", fake_get_open_stop_loss)
    monkeypatch.setattr(harness, "get_open_stop_loss_entry_time", fake_entry_time)
    monkeypatch.setattr(harness, "get_daily_low_series", fake_daily_lows)
    monkeypatch.setattr(harness, "get_daily_high_series", fake_daily_highs)
    monkeypatch.setattr(harness, "get_atr_pct", fake_atr_pct)
    monkeypatch.setattr(harness, "check_and_apply_trailing_stop", fake_trailing_check)

    # prev already carries a high_water of 150 from an earlier gap, same entry price.
    prev = harness.BacktestRecord(
        trade_date=date(2026, 7, 19), decision=_decision(), close=150.0,
        trailing_high_water=150.0, trailing_entry_price=same_entry_price, decision_id=7,
    )
    prices = {date(2026, 7, 19): 150.0, date(2026, 7, 22): 162.0}
    benchmark_prices = {date(2026, 7, 19): 400.0, date(2026, 7, 22): 405.0}

    record = await harness._process_ticker_date("AAPL", date(2026, 7, 22), prev, prices, benchmark_prices, "SPY")

    # Same entry price -> NOT reset; the prior 150 high-water carries in as input.
    assert captured["high_water_in"] == 150.0
    assert record.trailing_high_water == 165.0  # ratcheted up by fake_trailing_check's own return


async def test_process_ticker_date_resets_high_water_on_new_entry_price(monkeypatch):
    _stub_common_pipeline(monkeypatch)

    async def fake_get_open_stop_loss(ticker):
        return date(2026, 7, 16), 70.0

    async def fake_entry_time(ticker):
        return object(), 90.0, 70.0  # a DIFFERENT entry price than prev's 100.0 -> add-on/reopen

    async def fake_daily_lows(ticker, start, end):
        return {date(2026, 7, 21): 88.0}

    async def fake_daily_highs(ticker, start, end):
        return {date(2026, 7, 21): 92.0}

    async def fake_atr_pct(ticker, as_of=None):
        return None

    captured = {}

    async def fake_trailing_check(ticker, gap_highs, gap_lows, pm_stop_loss, entry_price, high_water, trailing_pct, reason=""):
        captured["high_water_in"] = high_water
        captured["entry_price_in"] = entry_price
        return None, high_water

    monkeypatch.setattr(harness, "get_open_stop_loss", fake_get_open_stop_loss)
    monkeypatch.setattr(harness, "get_open_stop_loss_entry_time", fake_entry_time)
    monkeypatch.setattr(harness, "get_daily_low_series", fake_daily_lows)
    monkeypatch.setattr(harness, "get_daily_high_series", fake_daily_highs)
    monkeypatch.setattr(harness, "get_atr_pct", fake_atr_pct)
    monkeypatch.setattr(harness, "check_and_apply_trailing_stop", fake_trailing_check)

    prev = harness.BacktestRecord(
        trade_date=date(2026, 7, 19), decision=_decision(), close=150.0,
        trailing_high_water=150.0, trailing_entry_price=100.0, decision_id=7,
    )
    prices = {date(2026, 7, 19): 150.0, date(2026, 7, 22): 91.0}
    benchmark_prices = {date(2026, 7, 19): 400.0, date(2026, 7, 22): 405.0}

    await harness._process_ticker_date("AAPL", date(2026, 7, 22), prev, prices, benchmark_prices, "SPY")

    # Reset to the NEW entry price, not the stale 150 high-water from before.
    assert captured["entry_price_in"] == 90.0
    assert captured["high_water_in"] == 90.0


# ---------------------------------------------------------------------------
# _run_pipeline_with_retry — outer retry around a stuck/outaged LLM endpoint
# (project memory: llm-model-swap-gpt-oss-issue's 91-minute outage), on top
# of the LLM client's own much shorter retry budget.
# ---------------------------------------------------------------------------


async def test_run_pipeline_with_retry_returns_immediately_on_success(monkeypatch):
    calls = []

    async def fake_run_pipeline(ticker, trade_date, as_of=None, portfolio_cap_override=None, profile=None):
        calls.append(1)

        class FakeState:
            final_decision = _decision()

        return FakeState()

    async def fake_sleep(seconds):
        raise AssertionError("must not sleep when the first attempt succeeds")

    monkeypatch.setattr(harness, "run_pipeline", fake_run_pipeline)
    monkeypatch.setattr(harness.asyncio, "sleep", fake_sleep)

    result = await harness._run_pipeline_with_retry("AAPL", date(2026, 7, 16), None)
    assert len(calls) == 1
    assert result.final_decision is not None


async def test_run_pipeline_with_retry_retries_on_exception_then_succeeds(monkeypatch):
    attempts = []
    sleeps = []

    async def fake_run_pipeline(ticker, trade_date, as_of=None, portfolio_cap_override=None, profile=None):
        attempts.append(1)
        if len(attempts) < 3:
            raise RuntimeError("simulated transient LLM outage")

        class FakeState:
            final_decision = _decision()

        return FakeState()

    async def fake_sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr(harness, "run_pipeline", fake_run_pipeline)
    monkeypatch.setattr(harness.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(settings, "backtest_pipeline_retry_attempts", 5)
    monkeypatch.setattr(settings, "backtest_pipeline_retry_delay_seconds", 600.0)

    result = await harness._run_pipeline_with_retry("AAPL", date(2026, 7, 16), None)
    assert len(attempts) == 3
    assert sleeps == [600.0, 600.0]  # slept once per FAILED attempt, not after the final success
    assert result.final_decision is not None


async def test_run_pipeline_with_retry_raises_after_exhausting_all_attempts(monkeypatch):
    attempts = []

    async def always_fails(ticker, trade_date, as_of=None, portfolio_cap_override=None, profile=None):
        attempts.append(1)
        raise RuntimeError("persistent outage")

    async def fake_sleep(seconds):
        pass

    monkeypatch.setattr(harness, "run_pipeline", always_fails)
    monkeypatch.setattr(harness.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(settings, "backtest_pipeline_retry_attempts", 3)

    with pytest.raises(RuntimeError, match="persistent outage"):
        await harness._run_pipeline_with_retry("AAPL", date(2026, 7, 16), None)
    assert len(attempts) == 3  # exactly the configured attempt count, no more/less


async def test_process_ticker_date_raises_when_no_final_decision(monkeypatch):
    async def fake_run_pipeline(ticker, trade_date, as_of=None, portfolio_cap_override=None, profile=None):
        class FakeState:
            final_decision = None
            rebalance_adjustment_note = None

        return FakeState()

    monkeypatch.setattr(harness, "run_pipeline", fake_run_pipeline)

    with pytest.raises(RuntimeError, match="produced no final decision"):
        await harness._process_ticker_date(
            "AAPL", date(2026, 7, 16), None, {date(2026, 7, 16): 150.0}, {date(2026, 7, 16): 400.0}, "SPY"
        )


async def test_run_backtest_end_to_end_with_all_dependencies_mocked(monkeypatch):
    ticker_prices = {date(2026, 7, d): 100.0 + d for d in (14, 15, 16)}
    benchmark_prices = {date(2026, 7, d): 400.0 + d for d in (14, 15, 16)}

    async def fake_get_price_series(ticker, start, end):
        return benchmark_prices if ticker == "SPY" else ticker_prices

    async def fake_get_open_stop_loss(ticker):
        return None

    async def fake_run_pipeline(ticker, trade_date, as_of=None, portfolio_cap_override=None, profile=None):
        class FakeState:
            final_decision = _decision()
            rebalance_adjustment_note = None
            decision_id = 1

        return FakeState()

    async def fake_reflect(*args, **kwargs):
        return "reflection"

    async def fake_resolve_outcome(*args, **kwargs):
        pass

    async def fake_get_position(ticker):
        return Position(cash=1000.0, shares=1)

    async def fake_clear_backtest_positions(tickers):
        pass

    monkeypatch.setattr(harness, "get_price_series", fake_get_price_series)
    monkeypatch.setattr(harness, "get_open_stop_loss", fake_get_open_stop_loss)
    monkeypatch.setattr(harness, "run_pipeline", fake_run_pipeline)
    monkeypatch.setattr(harness, "reflect", fake_reflect)
    monkeypatch.setattr(harness, "resolve_outcome", fake_resolve_outcome)
    monkeypatch.setattr(harness, "get_position", fake_get_position)
    monkeypatch.setattr(harness, "clear_backtest_positions", fake_clear_backtest_positions)

    result = await harness.run_backtest("AAPL", date(2026, 7, 14), date(2026, 7, 16), cadence_days=1)

    assert result.ticker == "AAPL"
    assert len(result.records) == 3
    assert result.buy_and_hold_return is not None


async def test_run_backtest_raises_when_fewer_than_two_trading_dates(monkeypatch):
    async def fake_get_price_series(ticker, start, end):
        return {date(2026, 7, 14): 100.0}

    monkeypatch.setattr(harness, "get_price_series", fake_get_price_series)

    with pytest.raises(ValueError, match="need at least 2"):
        await harness.run_backtest("AAPL", date(2026, 7, 14), date(2026, 7, 14))


# ---------------------------------------------------------------------------
# Replay byte-identity guard (2026-08-27). force_exit_position grew a
# `fill_price`/`breach_floor` pair and a live-tick price resolver so a live
# forced exit stops recording the stop THRESHOLD in `trades.price` (the
# 047040.KS id=280 incident). A replay must be completely unaffected: the
# harness calls check_and_apply_stop_loss / check_and_apply_trailing_stop,
# whose signatures did not change, so neither new kwarg is ever passed —
# and _resolve_forced_exit_price short-circuits on is_backtest_context()
# before it reads any cache. This asserts the resulting row is identical to
# what it always was: price == stop_loss, adjustment_note NULL.
# ---------------------------------------------------------------------------


async def test_gap_stop_loss_replay_writes_no_price_basis_note(monkeypatch):
    captured = {}

    async def fake_get_position(ticker):
        return Position(cash=0.0, shares=4)

    async def fake_real_shares_held(ticker):
        return None

    async def fake_get_open_stop_loss(ticker):
        return (date(2026, 7, 16), 140.0)

    async def fake_record_trade(ticker, trade_date, side, shares, price, reason, **kwargs):
        captured.update(price=price, **kwargs)
        return 1

    async def fake_mirror_trade_to_kis(ticker, trade_date, side, shares, price, assumed_current_shares, approved=False, trade_id=None):
        return "skipped: backtest"

    async def fake_record_mirror_disposition(trade_id, disposition):
        return None

    async def fake_record_alert(alert):
        return None

    def should_not_be_called(ticker):
        raise AssertionError("a replay must never read the live tick cache")

    monkeypatch.setattr(ledger, "get_position", fake_get_position)
    monkeypatch.setattr(ledger, "real_shares_held", fake_real_shares_held)
    monkeypatch.setattr(ledger, "get_open_stop_loss", fake_get_open_stop_loss)
    monkeypatch.setattr(ledger, "record_trade", fake_record_trade)
    monkeypatch.setattr(ledger, "mirror_trade_to_kis", fake_mirror_trade_to_kis)
    monkeypatch.setattr(ledger, "_record_mirror_disposition", fake_record_mirror_disposition)
    monkeypatch.setattr(ledger, "record_alert", fake_record_alert)
    monkeypatch.setattr(ledger, "freshest_tick", should_not_be_called)

    with backtest_scope():
        outcome = await ledger.check_and_apply_stop_loss(
            "AAPL", {date(2026, 7, 18): 138.0}, 140.0, "replay"
        )

    assert outcome is not None and outcome.breach_date == date(2026, 7, 18)
    assert captured["price"] == 140.0
    assert captured["stop_loss"] == 140.0
    assert captured["price"] == captured["stop_loss"]
    assert captured["adjustment_note"] is None
