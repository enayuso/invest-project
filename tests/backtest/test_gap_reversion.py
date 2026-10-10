"""app.backtest.gap_reversion — the pre-registered 2026-08-24 overnight-gap
idiosyncratic-reversion signal, exercised against small hand-built inputs.
No cache/DB fixtures needed — see that module's own docstring on why these
thresholds must never be tuned against real cached candles."""

from __future__ import annotations

from app.backtest.gap_reversion import classify_gap_day, make_gap_reversion_signal
from app.backtest.minute_candles import HistoricalMinuteCandle


def _time(minute_offset: int) -> str:
    total_minutes = 9 * 60 + minute_offset
    return f"{total_minutes // 60:02d}{total_minutes % 60:02d}00"


def _candle(minute_offset: int, o: float, h: float, l: float, c: float, v: int = 1000) -> HistoricalMinuteCandle:
    return HistoricalMinuteCandle(time=_time(minute_offset), open=o, high=h, low=l, close=c, volume=v)


# --- classify_gap_day ------------------------------------------------------


def _flat_trailing_closes(n: int = 20, price: float = 100.0) -> list[float]:
    """A trailing series with zero volatility — used as a baseline that
    tests then perturb, since GAP_RESIDUAL_MIN_SIGMA scales with trailing
    vol and a flat series alone can't qualify (trailing_vol == 0)."""
    return [price] * n


def test_does_not_qualify_with_zero_trailing_volatility():
    # Even a huge idiosyncratic-looking gap can't be judged "large relative
    # to trailing vol" when trailing vol is exactly zero.
    result = classify_gap_day(
        ticker_prior_closes=_flat_trailing_closes(),
        ticker_prev_close=100.0, ticker_today_open=90.0,
        market_prev_close=100.0, market_today_open=100.0,
    )
    assert result.trailing_vol == 0.0
    assert result.qualifies is False


def _noisy_trailing_closes() -> list[float]:
    # Alternating +1%/-1% daily returns -> a real, nonzero trailing stdev.
    closes = [100.0]
    for i in range(19):
        closes.append(closes[-1] * (1.01 if i % 2 == 0 else 0.99))
    return closes


def test_qualifies_on_a_large_idiosyncratic_gap_down_with_no_market_counterpart():
    trailing = _noisy_trailing_closes()
    result = classify_gap_day(
        ticker_prior_closes=trailing,
        ticker_prev_close=100.0, ticker_today_open=95.0,  # -5% gap
        market_prev_close=100.0, market_today_open=100.0,  # market flat
    )
    assert result.gap_pct < 0
    assert result.qualifies is True


def test_does_not_qualify_when_gap_is_explained_by_the_market():
    trailing = _noisy_trailing_closes()
    # Ticker gaps down 5%, but so does the market by almost the same
    # amount — the residual (idiosyncratic) component is tiny.
    result = classify_gap_day(
        ticker_prior_closes=trailing,
        ticker_prev_close=100.0, ticker_today_open=95.0,
        market_prev_close=100.0, market_today_open=95.2,
    )
    assert result.qualifies is False


def test_does_not_qualify_on_a_gap_up_even_if_idiosyncratic_long_only():
    trailing = _noisy_trailing_closes()
    # A gap UP that the market doesn't share is just as "idiosyncratic" by
    # the residual math, but this system is long-only — there's no way to
    # fade a gap up without shorting, so it must never qualify.
    result = classify_gap_day(
        ticker_prior_closes=trailing,
        ticker_prev_close=100.0, ticker_today_open=105.0,
        market_prev_close=100.0, market_today_open=100.0,
    )
    assert result.qualifies is False


def test_does_not_qualify_below_the_minimum_absolute_gap_floor():
    trailing = _noisy_trailing_closes()
    # A gap that's technically many trailing-sigma large (because trailing
    # vol here is tiny) but below the absolute floor (0.5%) must not qualify.
    result = classify_gap_day(
        ticker_prior_closes=[100.0, 100.001, 100.0, 100.001, 100.0],
        ticker_prev_close=100.0, ticker_today_open=99.7,  # -0.3%, below GAP_MIN_ABS_PCT
        market_prev_close=100.0, market_today_open=100.0,
    )
    assert result.qualifies is False


# --- make_gap_reversion_signal ---------------------------------------------


def test_signal_does_not_enter_when_qualifies_is_false():
    candles = [_candle(i, 95.0, 95.5, 94.5, 95.0) for i in range(40)]
    signal = make_gap_reversion_signal(qualifies=False)
    for idx in range(len(candles)):
        assert signal(candles, idx, False) is None


def test_signal_enters_at_bar_zero_when_qualifies_is_true():
    candles = [_candle(0, 95.0, 95.5, 94.5, 95.0)]
    signal = make_gap_reversion_signal(qualifies=True)
    assert signal(candles, 0, False) == "buy"


def test_signal_never_enters_again_after_bar_zero_even_if_still_flat():
    # Guards against a bug where the entry condition re-fires on a later
    # bar if the caller somehow calls it flat again — idx == 0 is the only
    # entry opportunity this signal ever takes.
    candles = [_candle(i, 95.0, 95.5, 94.5, 95.0) for i in range(5)]
    signal = make_gap_reversion_signal(qualifies=True)
    assert signal(candles, 0, False) == "buy"
    # Simulate the harness marking it flat again (shouldn't happen in
    # practice since the harness tracks position_open itself, but the
    # signal's own idx==0 guard should hold regardless).
    assert signal(candles, 1, False) is None


def test_signal_exits_at_the_fixed_time_stop():
    candles = [_candle(i, 95.0, 95.2, 94.8, 95.0) for i in range(31)]
    signal = make_gap_reversion_signal(qualifies=True)
    assert signal(candles, 0, False) == "buy"
    for idx in range(1, 30):
        assert signal(candles, idx, True) is None  # minutes_held 1..29, no exit yet
    assert signal(candles, 30, True) == "sell"  # minutes_held == 30 -> time stop


def test_signal_exits_early_on_stop_loss_before_the_time_stop():
    candles = [_candle(0, 95.0, 95.2, 94.8, 95.0)]
    signal = make_gap_reversion_signal(qualifies=True)
    assert signal(candles, 0, False) == "buy"  # entry_price = 95.0

    # -0.8% or worse triggers the stop: 95.0 * (1 - 0.008) = 94.24
    candles.append(_candle(1, 95.0, 95.0, 94.0, 94.2))
    assert signal(candles, 1, True) == "sell"


def test_signal_stop_fires_on_an_intrabar_breach_that_closes_back_above_it():
    # 2026-09-11 fix: the stop compares candles[idx].low (worst intrabar
    # price), not close-only -- this module's own design rationale is
    # protecting against a real intraday breach, which a close-only check
    # misses on exactly the bar where the breach happens and price recovers.
    candles = [_candle(0, 95.0, 95.2, 94.8, 95.0)]
    signal = make_gap_reversion_signal(qualifies=True)
    assert signal(candles, 0, False) == "buy"  # entry_price = 95.0

    # stop = 95.0 * (1 - 0.008) = 94.24; this bar's low breaches it but
    # closes back above -- a close-only check would have missed the breach.
    candles.append(_candle(1, 94.9, 95.0, 94.1, 94.9))
    assert signal(candles, 1, True) == "sell"


def test_fresh_instance_has_no_memory_of_a_previous_instance():
    candles = [_candle(0, 95.0, 95.5, 94.5, 95.0)]
    first = make_gap_reversion_signal(qualifies=True)
    assert first(candles, 0, False) == "buy"

    second = make_gap_reversion_signal(qualifies=True)
    assert second(candles, 0, False) == "buy"
