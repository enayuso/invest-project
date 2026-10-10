"""app.backtest.scalping_harness — round-trip return arithmetic, in
particular the 2026-08-21 fee-netting change (see project memory:
scalping-backtest-infra-gap and the commit that added
app.execution.costs). No cached-candle/DB fixtures needed here — these
exercise `_compute_return_pct` directly against hand-built trades, which is
where the compounding formula actually lives."""

from __future__ import annotations

import pytest

from app.backtest.fill_simulation import SimulatedFill
from app.backtest.minute_candles import HistoricalMinuteCandle
from app.backtest.scalping_harness import ScalpingTrade, _compute_return_pct, run_scalping_backtest_day
from app.config import settings
from datetime import date


def _fill(side: str, executed_price: float) -> SimulatedFill:
    return SimulatedFill(
        signal_time="093000",
        requested_latency_seconds=30,
        filled_time="093030",
        fill_price=executed_price,
        slippage_bps=0.0,
        side=side,
        executed_price=executed_price,
        realized_latency_seconds=30,
        gap_inflated=False,
    )


def _round_trip(buy_price: float, sell_price: float) -> list[ScalpingTrade]:
    return [
        ScalpingTrade(side="buy", signal_time="093000", fill=_fill("buy", buy_price)),
        ScalpingTrade(side="sell", signal_time="093100", fill=_fill("sell", sell_price)),
    ]


def test_domestic_round_trip_nets_out_commission_plus_tax(monkeypatch):
    monkeypatch.setattr(settings, "transaction_cost_bps", 5.0)
    monkeypatch.setattr(settings, "sell_tax_bps_domestic", 20.0)

    # Gross return is exactly 0% (sell price == buy price) so the entire
    # result below is attributable to fees alone.
    trades = _round_trip(100.0, 100.0)
    return_pct = _compute_return_pct(trades, "005930.KS")

    buy_fee_rate = 5.0 / 10_000
    sell_fee_rate = 25.0 / 10_000  # commission + domestic sell tax
    expected = ((100.0 * (1 - sell_fee_rate)) / (100.0 * (1 + buy_fee_rate)) - 1) * 100
    assert return_pct == pytest.approx(expected)
    assert return_pct == pytest.approx(-0.2998500749625155)


def test_overseas_round_trip_is_charged_commission_only(monkeypatch):
    monkeypatch.setattr(settings, "transaction_cost_bps", 5.0)
    monkeypatch.setattr(settings, "transaction_cost_bps_overseas", 5.0)
    monkeypatch.setattr(settings, "sell_tax_bps_domestic", 20.0)

    trades = _round_trip(100.0, 100.0)
    return_pct = _compute_return_pct(trades, "AAPL")

    buy_fee_rate = 5.0 / 10_000
    sell_fee_rate = 5.0 / 10_000  # no ".KS" suffix, no Korean tax
    expected = ((100.0 * (1 - sell_fee_rate)) / (100.0 * (1 + buy_fee_rate)) - 1) * 100
    assert return_pct == pytest.approx(expected)
    assert return_pct == pytest.approx(-0.09995002498750516)


def test_zero_fees_reproduces_the_old_pre_fee_compounding_exactly(monkeypatch):
    # Backward-compat pin: with both cost knobs at 0, the new fee-aware
    # formula collapses to the old bare sell_price/buy_price compounding —
    # proves this change is fee-only and didn't perturb the compounding
    # logic itself.
    monkeypatch.setattr(settings, "transaction_cost_bps", 0.0)
    monkeypatch.setattr(settings, "sell_tax_bps_domestic", 0.0)

    trades = _round_trip(100.0, 105.0)
    return_pct = _compute_return_pct(trades, "005930.KS")

    assert return_pct == pytest.approx((105.0 / 100.0 - 1) * 100)
    assert return_pct == pytest.approx(5.0)


def test_dangling_buy_with_no_matching_sell_is_ignored(monkeypatch):
    monkeypatch.setattr(settings, "transaction_cost_bps", 5.0)
    monkeypatch.setattr(settings, "sell_tax_bps_domestic", 20.0)

    trades = [ScalpingTrade(side="buy", signal_time="093000", fill=_fill("buy", 100.0))]
    assert _compute_return_pct(trades, "005930.KS") is None


def _candle(hhmmss: str, price: float) -> HistoricalMinuteCandle:
    return HistoricalMinuteCandle(time=hhmmss, open=price, high=price, low=price, close=price, volume=1000)


def test_zero_latency_signal_fills_at_the_next_bars_open_not_its_own_signal_bar():
    # 2026-09-11 fix: a signal firing on candle idx must never fill at that
    # same candle's own open — the close it decided on (and every entry
    # signal in this folder decides off close) is only knowable once that
    # bar ends, one bar later.
    candles = [_candle("093000", 100.0), _candle("093100", 101.0), _candle("093200", 102.0)]

    def fire_buy_once(candles, idx, has_position):
        return "buy" if idx == 0 and not has_position else None

    result = run_scalping_backtest_day(
        candles, fire_buy_once, ticker="AAPL", trade_date=date(2026, 1, 1), latency_seconds=0.0, slippage_bps=0.0,
    )
    buy = next(t for t in result.trades if t.side == "buy")
    assert buy.fill.fill_price == 101.0  # candles[1].open, never candles[0].open (100.0)
