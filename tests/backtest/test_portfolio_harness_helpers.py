"""app.backtest.portfolio_harness — pure helper functions only
(select_shared_trading_dates, _rotate, _tally_cap_triggers, _alpha_series).
run_portfolio_backtest's full orchestration reuses harness._process_ticker_date,
already covered directly in test_backtest_harness.py, so it isn't re-mocked
here (F-stage)."""

from __future__ import annotations

from datetime import date

import pytest

from app.backtest import portfolio_harness as ph
from app.backtest.metrics import BacktestRecord, alpha_win
from app.execution.ledger import Position
from app.state import PortfolioDecision, PortfolioRating


def _decision(rating=PortfolioRating.BUY) -> PortfolioDecision:
    return PortfolioDecision(rating=rating, executive_summary="s", investment_thesis="t")


def _record(d: date, close: float, rating=PortfolioRating.BUY) -> BacktestRecord:
    return BacktestRecord(trade_date=d, decision=_decision(rating), close=close)


# ---------------------------------------------------------------------------
# _select_cli_tickers — domestic/overseas separation audit (2026-07-27)
# ---------------------------------------------------------------------------


def test_select_cli_tickers_explicit_tickers_wins_over_market():
    result = ph._select_cli_tickers("AAPL, 005930.KS", "domestic", ["MSFT", "000660.KS"])
    assert result == ["AAPL", "005930.KS"]


def test_select_cli_tickers_all_returns_full_trade_allowlist():
    trade_allowlist = ["AAPL", "005930.KS", "MSFT"]
    assert ph._select_cli_tickers(None, "all", trade_allowlist) == trade_allowlist


def test_select_cli_tickers_domestic_filters_to_ks_suffix():
    trade_allowlist = ["AAPL", "005930.KS", "MSFT", "000660.KS"]
    assert ph._select_cli_tickers(None, "domestic", trade_allowlist) == ["005930.KS", "000660.KS"]


def test_select_cli_tickers_overseas_filters_out_ks_suffix():
    trade_allowlist = ["AAPL", "005930.KS", "MSFT", "000660.KS"]
    assert ph._select_cli_tickers(None, "overseas", trade_allowlist) == ["AAPL", "MSFT"]


# ---------------------------------------------------------------------------
# select_shared_trading_dates
# ---------------------------------------------------------------------------


def test_select_shared_trading_dates_intersects_across_tickers():
    per_ticker = {
        "AAPL": {date(2026, 1, d): 100.0 for d in (1, 2, 3, 4)},
        "MSFT": {date(2026, 1, d): 200.0 for d in (2, 3, 4, 5)},
    }
    result = ph.select_shared_trading_dates(per_ticker, cadence_days=1)
    assert result == [date(2026, 1, 2), date(2026, 1, 3), date(2026, 1, 4)]


def test_select_shared_trading_dates_raises_on_empty_input():
    with pytest.raises(ValueError, match="no tickers given"):
        ph.select_shared_trading_dates({}, cadence_days=1)


def test_select_shared_trading_dates_raises_when_fewer_than_two_shared_dates():
    per_ticker = {
        "AAPL": {date(2026, 1, 1): 100.0},
        "MSFT": {date(2026, 1, 1): 200.0, date(2026, 1, 2): 201.0},
    }
    with pytest.raises(ValueError, match="need at least 2"):
        ph.select_shared_trading_dates(per_ticker, cadence_days=1)


# ---------------------------------------------------------------------------
# _rotate
# ---------------------------------------------------------------------------


def test_rotate_shifts_cyclically():
    assert ph._rotate(["A", "B", "C"], 1) == ["B", "C", "A"]
    assert ph._rotate(["A", "B", "C"], 2) == ["C", "A", "B"]


def test_rotate_wraps_shift_larger_than_length():
    assert ph._rotate(["A", "B", "C"], 4) == ["B", "C", "A"]


def test_rotate_empty_list_returns_empty():
    assert ph._rotate([], 3) == []


# ---------------------------------------------------------------------------
# _tally_cap_triggers
# ---------------------------------------------------------------------------


def test_tally_cap_triggers_matches_case_insensitively():
    counts: dict[str, int] = {}
    ph._tally_cap_triggers("Portfolio Concentration Cap reduced this trade", counts)
    assert counts == {"portfolio concentration cap": 1}


def test_tally_cap_triggers_ignores_none_note():
    counts: dict[str, int] = {}
    ph._tally_cap_triggers(None, counts)
    assert counts == {}


def test_tally_cap_triggers_accumulates_across_calls():
    counts: dict[str, int] = {}
    ph._tally_cap_triggers("1% risk cap applied", counts)
    ph._tally_cap_triggers("1% risk cap applied again", counts)
    assert counts["1% per-ticker risk cap"] == 2


def test_tally_cap_triggers_matches_multiple_labels_in_one_note():
    counts: dict[str, int] = {}
    ph._tally_cap_triggers("1% risk cap and portfolio drawdown breaker both fired", counts)
    assert counts["1% per-ticker risk cap"] == 1
    assert counts["portfolio drawdown breaker"] == 1


def test_tally_cap_triggers_matches_the_atr_gate_note():
    # 2026-09-02 fail-closed ATR gate (app.execution.ledger.rebalance_to_target)
    # — its adjustment_note opens with this exact phrase.
    counts: dict[str, int] = {}
    ph._tally_cap_triggers(
        "No new position opened this cycle: 14-day ATR is unavailable for this ticker (value: None), "
        "so neither the volatility scale-down ... could be applied.",
        counts,
    )
    assert counts == {"ATR unavailable (new entry blocked)": 1}


# ---------------------------------------------------------------------------
# _alpha_series
# ---------------------------------------------------------------------------


def test_alpha_series_computes_raw_minus_benchmark_return():
    records = [_record(date(2026, 1, 1), 100.0), _record(date(2026, 1, 2), 110.0)]
    benchmark = {date(2026, 1, 1): 400.0, date(2026, 1, 2): 404.0}
    result = ph._alpha_series(records, benchmark)
    # raw = 0.10, benchmark = 0.01 -> alpha = 0.09
    assert result == pytest.approx([0.09])


def test_alpha_series_empty_for_single_record():
    records = [_record(date(2026, 1, 1), 100.0)]
    assert ph._alpha_series(records, {date(2026, 1, 1): 400.0}) == []


# ---------------------------------------------------------------------------
# _alpha_records / alpha_win — direction-aware win scoring (2026-09-11 fix:
# win_rate previously never read prev.decision.rating and used the undirected
# alpha>0 test app.audit.stats abandoned on 2026-08-13).
# ---------------------------------------------------------------------------


def test_alpha_records_pairs_alpha_with_prevs_rating():
    records = [
        _record(date(2026, 1, 1), 100.0, PortfolioRating.SELL),
        _record(date(2026, 1, 2), 110.0, PortfolioRating.BUY),
    ]
    benchmark = {date(2026, 1, 1): 400.0, date(2026, 1, 2): 404.0}
    result = ph._alpha_records(records, benchmark)
    assert result == [("Sell", pytest.approx(0.09))]  # prev's rating (Sell), not nxt's (Buy)


def test_alpha_win_buy_wins_on_positive_alpha():
    assert alpha_win("Buy", 0.09) is True
    assert alpha_win("Buy", -0.09) is False


def test_alpha_win_sell_wins_by_underperforming():
    # Sell/Underweight succeed by UNDERPERFORMING the benchmark (avoided/
    # reduced exposure to an underperformer) — the exact inverse of Buy.
    assert alpha_win("Sell", -0.109) is True
    assert alpha_win("Underweight", 0.09) is False


def test_alpha_win_hold_is_unscored():
    assert alpha_win("Hold", 0.09) is None
    assert alpha_win("Hold", -0.09) is None


def test_alpha_win_nan_alpha_is_unscored():
    assert alpha_win("Buy", float("nan")) is None


# ---------------------------------------------------------------------------
# screening_candidate_selector — replicates live's momentum re-ranking
# ---------------------------------------------------------------------------


async def test_screening_candidate_selector_returns_only_screened_picks(monkeypatch):
    async def fake_top_candidates(universe, top_n, lookback_days, as_of, use_volatility_adjustment, use_correlation_dedup):
        return ["005930.KS"]

    async def fake_get_position(ticker):
        return Position(cash=0.0, shares=0)  # nothing held

    monkeypatch.setattr(ph, "top_candidates", fake_top_candidates)
    monkeypatch.setattr(ph, "get_position", fake_get_position)

    selector = ph.screening_candidate_selector(top_n=1)
    result = await selector(date(2026, 7, 15), ["005930.KS", "000660.KS", "035420.KS"])
    assert result == ["005930.KS"]


async def test_screening_candidate_selector_always_includes_held_tickers(monkeypatch):
    # 000660.KS didn't make the momentum cut, but it's currently held —
    # must still appear, mirroring _held_needing_review's forced-review
    # principle (simplified: any held position, not just near-stop-loss).
    async def fake_top_candidates(universe, top_n, lookback_days, as_of, use_volatility_adjustment, use_correlation_dedup):
        return ["005930.KS"]

    async def fake_get_position(ticker):
        return Position(cash=0.0, shares=5) if ticker == "000660.KS" else Position(cash=0.0, shares=0)

    monkeypatch.setattr(ph, "top_candidates", fake_top_candidates)
    monkeypatch.setattr(ph, "get_position", fake_get_position)

    selector = ph.screening_candidate_selector(top_n=1)
    result = await selector(date(2026, 7, 15), ["005930.KS", "000660.KS", "035420.KS"])
    assert result == ["005930.KS", "000660.KS"]


async def test_screening_candidate_selector_dedupes_screened_and_held_overlap(monkeypatch):
    async def fake_top_candidates(universe, top_n, lookback_days, as_of, use_volatility_adjustment, use_correlation_dedup):
        return ["005930.KS"]

    async def fake_get_position(ticker):
        return Position(cash=0.0, shares=5) if ticker == "005930.KS" else Position(cash=0.0, shares=0)

    monkeypatch.setattr(ph, "top_candidates", fake_top_candidates)
    monkeypatch.setattr(ph, "get_position", fake_get_position)

    selector = ph.screening_candidate_selector(top_n=1)
    result = await selector(date(2026, 7, 15), ["005930.KS", "000660.KS"])
    assert result == ["005930.KS"]  # not duplicated


async def test_screening_candidate_selector_passes_as_of_and_params_through(monkeypatch):
    captured = {}

    async def fake_top_candidates(universe, top_n, lookback_days, as_of, use_volatility_adjustment, use_correlation_dedup):
        captured["as_of"] = as_of
        captured["top_n"] = top_n
        captured["lookback_days"] = lookback_days
        captured["use_volatility_adjustment"] = use_volatility_adjustment
        captured["use_correlation_dedup"] = use_correlation_dedup
        return []

    async def fake_get_position(ticker):
        return Position(cash=0.0, shares=0)

    monkeypatch.setattr(ph, "top_candidates", fake_top_candidates)
    monkeypatch.setattr(ph, "get_position", fake_get_position)

    selector = ph.screening_candidate_selector(
        top_n=3, lookback_days=15, use_volatility_adjustment=False, use_correlation_dedup=False
    )
    await selector(date(2026, 7, 15), ["005930.KS"])
    assert captured == {
        "as_of": date(2026, 7, 15), "top_n": 3, "lookback_days": 15,
        "use_volatility_adjustment": False, "use_correlation_dedup": False,
    }
