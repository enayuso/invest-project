"""app.backtest.auction_imbalance — the pre-registered 2026-08-24
closing-auction imbalance signal, exercised against small hand-built
inputs. No cache/DB fixtures needed — see that module's own docstring on
why these thresholds must never be tuned against real cached candles."""

from __future__ import annotations

from app.backtest.auction_imbalance import (
    AUCTION_MIN_WINDOW_BARS,
    compute_auction_imbalance,
    make_auction_imbalance_signal,
)
from app.backtest.minute_candles import HistoricalMinuteCandle


def _time(hour: int, minute: int) -> str:
    return f"{hour:02d}{minute:02d}00"


def _candle(hour: int, minute: int, o: float, h: float, l: float, c: float, v: int = 1000) -> HistoricalMinuteCandle:
    return HistoricalMinuteCandle(time=_time(hour, minute), open=o, high=h, low=l, close=c, volume=v)


def _flat_bars(start_hour: int, start_min: int, end_hour: int, end_min: int, price: float = 100.0, v: int = 1000):
    """One flat-price bar per minute from (start_hour,start_min) to
    (end_hour,end_min) inclusive — used for the pre-window baseline that
    tests don't care about, and as a template perturbed by individual
    tests for the window/auction bars they DO care about."""
    out = []
    total_start = start_hour * 60 + start_min
    total_end = end_hour * 60 + end_min
    for total in range(total_start, total_end + 1):
        out.append(_candle(total // 60, total % 60, price, price, price, price, v=v))
    return out


def _base_day(window_clv_sign: float = 1.0, auction_volume: int = 50_000) -> list[HistoricalMinuteCandle]:
    """A full synthetic trading day: flat pre-window bars (09:00-14:29),
    45 window bars (14:30-15:14) each closing near its high (if
    window_clv_sign > 0) or its low (if < 0) to drive a clean CLV sign,
    4 decision-window bars (15:15-15:18), and a final auction print at
    15:30."""
    candles = _flat_bars(9, 0, 14, 29)
    for total in range(14 * 60 + 30, 15 * 60 + 15):  # 14:30..15:14 inclusive, 45 bars
        hour, minute = total // 60, total % 60
        # v=100_000 per bar so 45 bars clears AUCTION_MIN_WINDOW_NOTIONAL_KRW
        # (300M) at a ~100 price — real large-cap names clear this easily.
        if window_clv_sign > 0:
            candles.append(_candle(hour, minute, 100.0, 101.0, 99.0, 100.9, v=100_000))  # closes near high
        else:
            candles.append(_candle(hour, minute, 100.0, 101.0, 99.0, 99.1, v=100_000))  # closes near low
    for total in range(15 * 60 + 15, 15 * 60 + 19):  # 15:15..15:18
        candles.append(_candle(total // 60, total % 60, 100.0, 100.5, 99.5, 100.0, v=1000))
    candles.append(_candle(15, 30, 100.0, 100.0, 100.0, 100.0, v=auction_volume))  # the auction print
    return candles


def _flat_market_day() -> list[HistoricalMinuteCandle]:
    """A market-proxy day with zero imbalance and zero drift, sized the
    same as _base_day, so ticker-vs-market residuals equal the ticker's
    own raw values in tests that don't care about the market side."""
    candles = _flat_bars(9, 0, 15, 18, price=100.0)
    candles.append(_candle(15, 30, 100.0, 100.0, 100.0, 100.0, v=50_000))
    return candles


# --- compute_auction_imbalance ---------------------------------------------


def test_ineligible_when_window_has_too_few_bars():
    # A day with a window of only 10 bars (< AUCTION_MIN_WINDOW_BARS).
    short_window = _flat_bars(14, 30, 14, 39)  # 10 bars, not 45
    candles = _flat_bars(9, 0, 14, 29) + short_window + [
        _candle(15, 30, 100.0, 100.0, 100.0, 100.0, v=50_000),
    ]
    result = compute_auction_imbalance(candles, _flat_market_day())
    assert result.eligible is False
    assert "window bars" in result.ineligible_reason


def test_eligibility_ignores_a_missing_or_thin_auction_print():
    # 2026-09-11 fix: eligibility used to depend on the "153000" auction
    # print existing and clearing a volume-share threshold against the
    # FULL DAY's total volume — both facts unknowable at the 15:15 decision
    # time this eligibility gates entry into (look-ahead). Neither a
    # stripped auction bar nor a near-zero auction_volume may affect
    # eligibility now; the day is judged only on pre-15:14 window data.
    stripped = [c for c in _base_day() if c.time != "153000"]
    assert compute_auction_imbalance(stripped, _flat_market_day()).eligible is True
    assert compute_auction_imbalance(_base_day(auction_volume=1), _flat_market_day()).eligible is True


def test_eligibility_uses_only_pre_decision_bars():
    # The full day and a day truncated to bars at or before the measurement
    # window's own end (151400, strictly before the 151500 decision bar)
    # must be judged IDENTICALLY eligible with IDENTICAL imb/drift values —
    # proving no bar at or after the decision time is read.
    full_day = _base_day()
    pre_decision_only = [c for c in full_day if c.time <= "151400"]
    market_day = _flat_market_day()
    full_result = compute_auction_imbalance(full_day, market_day)
    truncated_result = compute_auction_imbalance(pre_decision_only, market_day)
    assert full_result.eligible is True
    assert truncated_result.eligible is True
    assert truncated_result.imb_raw == full_result.imb_raw
    assert truncated_result.drift_resid == full_result.drift_resid


def test_eligible_day_has_positive_imb_raw_when_window_closes_near_highs():
    result = compute_auction_imbalance(_base_day(window_clv_sign=1.0), _flat_market_day())
    assert result.eligible is True
    assert result.imb_raw > 0.5  # closes consistently near the bar high -> CLV near +1


def test_eligible_day_has_negative_imb_raw_when_window_closes_near_lows():
    result = compute_auction_imbalance(_base_day(window_clv_sign=-1.0), _flat_market_day())
    assert result.eligible is True
    assert result.imb_raw < -0.5


def test_imb_resid_nets_out_a_shared_market_wide_component():
    # Both the ticker and the market proxy show the SAME one-sided drift
    # (window closes near highs for both) — the idiosyncratic residual
    # should collapse toward zero even though imb_raw itself is large.
    ticker_day = _base_day(window_clv_sign=1.0)
    market_day = _base_day(window_clv_sign=1.0)
    result = compute_auction_imbalance(ticker_day, market_day)
    assert result.eligible is True
    assert abs(result.imb_resid) < 0.05


def test_drift_reflects_window_close_relative_to_pre_window_vwap():
    # Pre-window bars are flat at 100; window bars close at 100.9 -> a
    # positive drift versus that pre-window VWAP.
    result = compute_auction_imbalance(_base_day(window_clv_sign=1.0), _flat_market_day())
    assert result.drift > 0


# --- make_auction_imbalance_signal ------------------------------------------


def test_signal_does_not_fire_when_imbalance_is_none():
    candles = _base_day()
    signal = make_auction_imbalance_signal(None)
    for idx in range(len(candles)):
        assert signal(candles, idx, False) is None


def test_signal_does_not_fire_when_imbalance_is_ineligible():
    # Too few window bars is still a valid ineligibility path (unlike the
    # removed auction-print-based one — see test_eligibility_ignores_a_
    # missing_or_thin_auction_print).
    short_window = _flat_bars(14, 30, 14, 39)  # 10 bars, not 45
    candles = _flat_bars(9, 0, 14, 29) + short_window + [_candle(15, 30, 100.0, 100.0, 100.0, 100.0, v=50_000)]
    imb = compute_auction_imbalance(candles, _flat_market_day())
    assert imb.eligible is False
    signal = make_auction_imbalance_signal(imb)
    for idx in range(len(candles)):
        assert signal(candles, idx, False) is None


def test_signal_fires_exactly_at_the_decision_window_start():
    imb = compute_auction_imbalance(_base_day(), _flat_market_day())
    assert imb.eligible is True
    candles = _base_day()
    signal = make_auction_imbalance_signal(imb)

    decision_idx = next(i for i, c in enumerate(candles) if c.time == "151500")
    for idx in range(decision_idx):
        assert signal(candles, idx, False) is None
    assert signal(candles, decision_idx, False) == "buy"


def test_signal_never_fires_after_the_decision_latest_time():
    imb = compute_auction_imbalance(_base_day(), _flat_market_day())
    candles = _base_day()
    signal = make_auction_imbalance_signal(imb)

    auction_idx = next(i for i, c in enumerate(candles) if c.time == "153000")
    assert signal(candles, auction_idx, False) is None


def test_signal_never_proposes_a_second_entry_or_a_sell():
    imb = compute_auction_imbalance(_base_day(), _flat_market_day())
    candles = _base_day()
    signal = make_auction_imbalance_signal(imb)

    decision_idx = next(i for i, c in enumerate(candles) if c.time == "151500")
    assert signal(candles, decision_idx, False) == "buy"
    # Same bar again, still reported flat by the caller — must not fire twice.
    assert signal(candles, decision_idx, False) is None
    # Once holding, this signal never proposes "sell" — see its own
    # docstring on relying on the harness's end-of-day force-close.
    for idx in range(decision_idx, len(candles)):
        assert signal(candles, idx, True) is None


def test_threshold_gated_run_skips_a_day_below_the_imbalance_threshold():
    # A day with only a mild imbalance (window closes near highs but not
    # overwhelmingly) and a market proxy that shares almost none of it —
    # imb_resid should land below AUCTION_IMB_TOP_THRESHOLD (0.15).
    mild_candles = _flat_bars(9, 0, 14, 29)
    for total in range(14 * 60 + 30, 15 * 60 + 15):
        hour, minute = total // 60, total % 60
        mild_candles.append(_candle(hour, minute, 100.0, 100.2, 99.8, 100.02, v=100_000))  # barely off midpoint
    mild_candles += [_candle(t // 60, t % 60, 100.0, 100.1, 99.9, 100.0, v=1000) for t in range(15 * 60 + 15, 15 * 60 + 19)]
    mild_candles.append(_candle(15, 30, 100.0, 100.0, 100.0, 100.0, v=50_000))

    imb = compute_auction_imbalance(mild_candles, _flat_market_day())
    assert imb.eligible is True
    assert imb.imb_resid < 0.15

    signal = make_auction_imbalance_signal(imb, require_threshold=True)
    decision_idx = next(i for i, c in enumerate(mild_candles) if c.time == "151500")
    assert signal(mild_candles, decision_idx, False) is None


def test_threshold_gated_run_fires_above_the_imbalance_threshold_with_positive_drift():
    imb = compute_auction_imbalance(_base_day(window_clv_sign=1.0), _flat_market_day())
    assert imb.eligible is True
    assert imb.imb_resid >= 0.15
    assert imb.drift_resid > 0

    candles = _base_day(window_clv_sign=1.0)
    signal = make_auction_imbalance_signal(imb, require_threshold=True)
    decision_idx = next(i for i, c in enumerate(candles) if c.time == "151500")
    assert signal(candles, decision_idx, False) == "buy"


def test_min_window_bars_constant_matches_the_45_bar_window_convention():
    # Sanity check that the pre-registered floor is still below the
    # window's own natural bar count (14:30-15:14 inclusive = 45 bars) —
    # guards against the constant ever being edited to exceed the window
    # itself, which would make every day structurally ineligible.
    assert AUCTION_MIN_WINDOW_BARS <= 45
