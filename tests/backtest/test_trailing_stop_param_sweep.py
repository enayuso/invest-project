"""app.backtest.trailing_stop_param_sweep._simulate_one_trade — 2026-09-11
fix for the same-day high-before-low false breach this module used to
reintroduce (the bug production's app.execution.trailing_stop has a
persisted watermark to prevent, the 047040.KS incident), and the exit-fill
gap-through fix (a stop cannot fill above the day's open when the open
already gapped through it).

All tests pass atr_pct=0.10, atr_multiplier=1.0 to pin trailing_pct=0.10
exactly, independent of settings.trailing_stop_fallback_pct. Entry price is
always 100; trailing_stop_activation_pct defaults to 0.04, so the floor is
unactivated (returns the 0.0 PM floor) until high_water climbs above 104."""

from __future__ import annotations

from datetime import date, timedelta

from app.backtest.trailing_stop_param_sweep import TickerSeries, _simulate_one_trade

_D0 = date(2026, 1, 1)
_ATR_PCT = 0.10
_ATR_MULT = 1.0


def _series(closes: list[float], highs: list[float], lows: list[float], opens: list[float]) -> TickerSeries:
    dates = [_D0 + timedelta(days=i) for i in range(len(closes))]
    return TickerSeries(
        dates=dates,
        closes=dict(zip(dates, closes)),
        highs=dict(zip(dates, highs)),
        lows=dict(zip(dates, lows)),
        opens=dict(zip(dates, opens)),
    )


def _run(series: TickerSeries):
    return _simulate_one_trade("TEST", series, entry_idx=0, atr_pct=_ATR_PCT, atr_multiplier=_ATR_MULT)


def test_no_same_day_false_breach_when_only_that_days_own_high_activates_the_floor():
    # d1: high 130 activates a floor of 117 for d2 (unactivated during d1
    #     itself: high_water is still just entry=100 when d1 is tested).
    # d2: high 140, low 118 -- floor tested is 117 (from d1's ratchet),
    #     118 > 117: no breach. high_water ratchets to 140 AFTER the test.
    # d3: floor now 126 (140*0.9), low 119 <= 126 -> breach on d3, not d2.
    #     The pre-fix code folded d2's own high (140) into high_water BEFORE
    #     testing d2's low, producing a fabricated breach on d2 at 126.
    series = _series(
        closes=[100.0, 128.0, 135.0, 120.0],
        highs=[100.0, 130.0, 140.0, 128.0],
        lows=[100.0, 125.0, 118.0, 119.0],
        opens=[100.0, 126.0, 128.0, 127.0],
    )
    result = _run(series)
    assert result.exit_reason == "trailing_stop"
    assert result.exit_date == series.dates[3]
    assert result.exit_price == 126.0  # min(floor=126, open=127)


def test_gap_through_fills_at_the_open_not_the_stop_level():
    # Same setup as above but d3 opens BELOW the 126 floor (earnings gap).
    series = _series(
        closes=[100.0, 128.0, 135.0, 120.0],
        highs=[100.0, 130.0, 140.0, 128.0],
        lows=[100.0, 125.0, 118.0, 110.0],
        opens=[100.0, 126.0, 128.0, 110.0],
    )
    result = _run(series)
    assert result.exit_reason == "trailing_stop"
    assert result.exit_price == 110.0  # gapped through -- fills at the open, not the 126 floor


def test_unambiguous_breach_from_a_prior_days_floor_still_fires_on_the_correct_day():
    # d1 activates a floor of 117 for d2 (130*0.9); d2's own low (116)
    # breaches that PRE-existing floor directly, regardless of ratchet
    # order (d2's own high of 119 wouldn't raise the floor above 117
    # anyway: 119*0.9=107.1) -- must fire on d2, not d1.
    series = _series(
        closes=[100.0, 128.0, 118.0],
        highs=[100.0, 130.0, 119.0],
        lows=[100.0, 125.0, 116.0],
        opens=[100.0, 128.0, 118.0],
    )
    result = _run(series)
    assert result.exit_reason == "trailing_stop"
    assert result.exit_date == series.dates[2]
    assert result.exit_price == 117.0


def test_no_breach_falls_through_to_period_end():
    series = _series(
        closes=[100.0, 105.0, 110.0],
        highs=[100.0, 106.0, 111.0],
        lows=[100.0, 104.0, 109.0],
        opens=[100.0, 105.0, 110.0],
    )
    result = _run(series)
    assert result.exit_reason == "period_end"
    assert result.exit_date == series.dates[-1]
    assert result.exit_price == 110.0
