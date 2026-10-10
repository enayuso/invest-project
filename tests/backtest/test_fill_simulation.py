"""app.backtest.fill_simulation — the 2026-09-11 fix for the latency clock
being anchored to a bar's raw START while every signal in this folder
decides off that bar's CLOSE (only knowable at start + BAR_DURATION_SECONDS).
Before this fix, latency_seconds=0 filled at the signal bar's own open (a
price that printed 60s before the triggering close existed), and every
latency in (0, 60] produced an identical result."""

from __future__ import annotations

from app.backtest.fill_simulation import BAR_DURATION_SECONDS, decision_time_for_bar, simulate_fill
from app.backtest.minute_candles import HistoricalMinuteCandle


def _candle(hhmmss: str, open_: float) -> HistoricalMinuteCandle:
    return HistoricalMinuteCandle(time=hhmmss, open=open_, high=open_, low=open_, close=open_, volume=1000)


_CANDLES = [
    _candle("090000", 100.0),
    _candle("090100", 101.0),
    _candle("090200", 102.0),
    _candle("090300", 103.0),
    _candle("090400", 104.0),
]


def test_decision_time_for_bar_is_bar_start_plus_bar_duration():
    assert BAR_DURATION_SECONDS == 60
    assert decision_time_for_bar(_CANDLES[0]) == "090100"


def test_zero_latency_fills_at_the_next_bars_open_not_the_signal_bars_own_open():
    decision_time = decision_time_for_bar(_CANDLES[0])  # "090100"
    fill = simulate_fill(_CANDLES, decision_time, 0, "buy", slippage_bps=0.0)
    assert fill is not None
    assert fill.filled_time == "090100"
    assert fill.fill_price == 101.0  # next bar's open, never candle[0]'s own open (100.0)
    assert fill.realized_latency_seconds == 0


def test_distinct_latencies_no_longer_collapse_to_the_same_fill():
    # Before the fix, latency_seconds=0 (anchored to the bar START) fired
    # at the SAME bar as any latency up to 60s (the next bar's open) --
    # collapsing the "instant reaction" baseline into the same result as a
    # 30-90s reaction. Anchored to the bar CLOSE instead, 0/90/150 now land
    # on three distinct bars, restoring the latency axis.
    decision_time = decision_time_for_bar(_CANDLES[0])  # "090100"
    fill_0 = simulate_fill(_CANDLES, decision_time, 0, "buy", slippage_bps=0.0)
    fill_60 = simulate_fill(_CANDLES, decision_time, 60, "buy", slippage_bps=0.0)
    fill_120 = simulate_fill(_CANDLES, decision_time, 120, "buy", slippage_bps=0.0)
    assert (fill_0.filled_time, fill_0.realized_latency_seconds) == ("090100", 0)
    assert (fill_60.filled_time, fill_60.realized_latency_seconds) == ("090200", 60)
    assert (fill_120.filled_time, fill_120.realized_latency_seconds) == ("090300", 120)
    # All three distinct -- the collapsed-axis regression this fix closes.
    prices = {fill_0.fill_price, fill_60.fill_price, fill_120.fill_price}
    assert len(prices) == 3
