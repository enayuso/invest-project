"""app.backtest.scalping_signals — the three pre-registered viability-
measurement signals (ORB, volume-spike, VWAP-reversion), exercised against
small hand-built `HistoricalMinuteCandle` sequences. No cache/DB fixtures
needed — see this module's own docstring on why these thresholds must never
be tuned against real cached candles."""

from __future__ import annotations

from app.backtest.minute_candles import HistoricalMinuteCandle
from app.backtest.scalping_signals import (
    make_orb_holdclose_signal,
    make_orb_signal,
    make_volume_spike_signal,
    make_vwap_reversion_signal,
)


def _time(minute_offset: int) -> str:
    """09:00 + minute_offset, as HHMMSS — matches the market-open convention
    ORB's opening range is built on."""
    total_minutes = 9 * 60 + minute_offset
    return f"{total_minutes // 60:02d}{total_minutes % 60:02d}00"


def _candle(minute_offset: int, o: float, h: float, l: float, c: float, v: int = 1000) -> HistoricalMinuteCandle:
    return HistoricalMinuteCandle(time=_time(minute_offset), open=o, high=h, low=l, close=c, volume=v)


# --- ORB ---------------------------------------------------------------


def _flat_opening_range(bars: int = 30, price: float = 100.0) -> list[HistoricalMinuteCandle]:
    return [_candle(i, price, price, price, price) for i in range(bars)]


def test_orb_does_not_fire_before_bar_30_even_on_a_huge_jump():
    candles = _flat_opening_range()
    # A huge jump inserted mid-opening-range shouldn't trigger anything —
    # the range itself isn't finalized until all 30 bars are seen.
    candles[10] = _candle(10, 100.0, 500.0, 100.0, 500.0)
    signal = make_orb_signal()
    for idx in range(len(candles)):
        assert signal(candles, idx, False) is None


def test_orb_fires_on_first_close_above_or_high_after_bar_30():
    candles = _flat_opening_range()  # OR high = 100.0
    candles.append(_candle(30, 100.0, 100.0, 100.0, 99.0))   # bar 30: no breakout
    candles.append(_candle(31, 100.0, 101.5, 100.0, 101.0))  # bar 31: breakout

    signal = make_orb_signal()
    assert signal(candles, 30, False) is None
    assert signal(candles, 31, False) == "buy"


def test_orb_does_not_fire_a_second_entry_while_holding():
    # 2026-09-11 re-anchor fix: entry at idx=30 only PROVISIONALLY sets
    # target/stop off cur.close — the real fill (simulate_fill, default
    # latency_seconds=30) doesn't resolve until idx=32 (one bar for
    # decision_time_for_bar's own +60s, plus the 30s latency rounds up to
    # the next whole-minute bar). idx=31 must therefore return None: not
    # "no target/stop hit" but "no fill to check against yet".
    candles = [_candle(i, 100.0, 102.0, 98.0, 100.0) for i in range(30)]  # OR high=102, low=98, width=4
    candles.append(_candle(30, 100.0, 103.0, 100.0, 103.0))  # breakout
    candles.append(_candle(31, 103.0, 104.0, 102.0, 103.0))  # fill not resolved yet at this bar

    signal = make_orb_signal()
    assert signal(candles, 30, False) == "buy"
    assert signal(candles, 31, True) is None  # already holding, but no real fill to check yet


def test_orb_target_hit_fires_sell_anchored_to_the_real_fill_price():
    # entry breakout at idx=30 (close=103, provisional); real fill resolves
    # at idx=32 -- executed_price = candles[32].open * (1 + 5bps) = 103.0515.
    # target = 103.0515 + 1.0*4(width) = 107.0515 -- candles[32]'s own high
    # (108) clears it, so the sell fires on the SAME bar the fill resolves.
    candles = [_candle(i, 100.0, 102.0, 98.0, 100.0) for i in range(30)]  # OR high=102, low=98, width=4
    candles.append(_candle(30, 100.0, 103.0, 100.0, 103.0))  # breakout
    candles.append(_candle(31, 103.0, 104.0, 102.0, 103.0))  # fill not resolved yet
    candles.append(_candle(32, 103.0, 108.0, 103.0, 104.0))  # fill bar: open=103 -> exec=103.0515; high 108 >= target

    signal = make_orb_signal()
    assert signal(candles, 30, False) == "buy"
    assert signal(candles, 31, True) is None
    assert signal(candles, 32, True) == "sell"


def test_orb_stop_hit_fires_sell_anchored_to_the_real_fill_price():
    # Same fill mechanics as the target test: executed_price = 103.0515,
    # stop = 103.0515 - 0.5*4 = 101.0515 -- candles[32]'s own low (100)
    # breaches it.
    candles = [_candle(i, 100.0, 102.0, 98.0, 100.0) for i in range(30)]  # width=4
    candles.append(_candle(30, 100.0, 103.0, 100.0, 103.0))  # breakout
    candles.append(_candle(31, 103.0, 104.0, 102.0, 103.0))  # fill not resolved yet
    candles.append(_candle(32, 103.0, 103.0, 100.0, 102.0))  # fill bar: open=103 -> exec=103.0515; low 100 <= stop

    signal = make_orb_signal()
    assert signal(candles, 30, False) == "buy"
    assert signal(candles, 31, True) is None
    assert signal(candles, 32, True) == "sell"


def test_orb_never_fills_returns_none_forever_if_harness_would_never_fill():
    # If the whole day ends before simulate_fill can find a candle at/after
    # its target time, the signal must never fabricate a target/stop off
    # the provisional close — it just returns None (see simulate_fill's own
    # None contract for "no fillable data left this session").
    candles = [_candle(i, 100.0, 102.0, 98.0, 100.0) for i in range(30)]
    candles.append(_candle(30, 100.0, 103.0, 100.0, 103.0))  # breakout, last candle of the day

    signal = make_orb_signal()
    assert signal(candles, 30, False) == "buy"
    assert signal(candles, 30, True) is None  # (has_position on the same last bar) no fill possible


def test_orb_second_entry_blocked_only_after_a_confirmed_fill_fresh_instance_unaffected():
    # 2026-09-11 fix: entries_today is now incremented only once a proposed
    # entry's fill is CONFIRMED (has_position transitioning to True), not at
    # mere proposal time — see the "if not state['anchored']" branch's own
    # comment. This test used to call the SAME instance twice with
    # has_position=False both times (never confirming a fill) and expect
    # the second call to be blocked; under the fix that sequence no longer
    # blocks anything (entries_today correctly stays 0, since no fill was
    # ever confirmed), so the old two-call shape no longer demonstrates the
    # daily cap at all. Replaced with a realistic sequence: propose, confirm
    # the fill via has_position=True (mirroring test_orb_target_hit_fires_
    # sell_anchored_to_the_real_fill_price's own idx=30/31/32 fill timing),
    # go flat again, THEN check a later breakout is correctly blocked.
    candles = [_candle(i, 100.0, 102.0, 98.0, 100.0) for i in range(30)]  # OR high=102, low=98, width=4
    candles.append(_candle(30, 100.0, 103.0, 100.0, 103.0))  # breakout
    candles.append(_candle(31, 103.0, 104.0, 102.0, 103.0))  # fill not resolved yet
    candles.append(_candle(32, 103.0, 105.0, 103.0, 104.0))  # fill bar: exec=103.0515; no target/stop hit here

    first = make_orb_signal()
    assert first(candles, 30, False) == "buy"
    assert first(candles, 31, True) is None  # no confirmed fill yet
    assert first(candles, 32, True) is None  # fill confirmed here (entries_today now 1); no target/stop hit

    # Position closes (however it closed doesn't matter to the signal, only
    # that it's flat again) — a fresh breakout later the same day must now
    # be blocked, since a REAL fill was confirmed earlier.
    candles.append(_candle(33, 104.0, 110.0, 104.0, 110.0))  # would-be second breakout
    assert first(candles, 33, False) is None  # entries_today >= 1 (confirmed), correctly blocked

    # A brand new instance — as scripts/run_scalping_viability.py constructs
    # per (ticker, day) — sees this as a fresh day and must still be able to
    # fire the entry. Reusing `first` here instead (as the multi-day
    # `run_scalping_backtest` wrapper would, per this module's docstring)
    # would incorrectly return None, because `entries_today` from the
    # "previous day" already hit its cap of 1.
    second = make_orb_signal()
    assert second(candles, 30, False) == "buy"


def test_orb_unfilled_entry_does_not_consume_the_daily_cap():
    # 2026-09-11 fix regression test: a proposed entry whose fill never
    # resolves (harness would never fill it — same setup as test_orb_never_
    # fills_returns_none_forever_if_harness_would_never_fill above, a
    # breakout on the LAST candle of the available data) must NOT consume
    # entries_today. A later, separate entry opportunity (once more data
    # extends past where the first attempt failed) must still fire "buy".
    candles = [_candle(i, 100.0, 102.0, 98.0, 100.0) for i in range(30)]
    candles.append(_candle(30, 100.0, 103.0, 100.0, 103.0))  # breakout, last candle of the day (for now)

    signal = make_orb_signal()
    assert signal(candles, 30, False) == "buy"
    assert signal(candles, 30, True) is None  # no fillable data left yet — proposal never confirmed

    # More data now "arrives" (simulating a legitimately truncated feed that
    # resumes, or simply a later bar within the same session) — a fresh
    # breakout must still fire, since entries_today was never actually
    # incremented for the unfilled first attempt.
    candles.append(_candle(31, 103.0, 110.0, 103.0, 110.0))
    assert signal(candles, 31, False) == "buy"


# --- ORB hold-to-close (pre-registered 2026-08-24, second exit hypothesis) --


def test_orb_holdclose_entry_matches_orb_exactly():
    candles = _flat_opening_range()  # OR high = 100.0
    candles.append(_candle(30, 100.0, 100.0, 100.0, 99.0))   # bar 30: no breakout
    candles.append(_candle(31, 100.0, 101.5, 100.0, 101.0))  # bar 31: breakout

    signal = make_orb_holdclose_signal()
    assert signal(candles, 30, False) is None
    assert signal(candles, 31, False) == "buy"


def test_orb_holdclose_never_sells_while_holding_regardless_of_price():
    candles = [_candle(i, 100.0, 102.0, 98.0, 100.0) for i in range(30)]  # OR high=102, low=98
    candles.append(_candle(30, 100.0, 103.0, 100.0, 103.0))  # breakout

    signal = make_orb_holdclose_signal()
    assert signal(candles, 30, False) == "buy"
    # A plain make_orb_signal() would stop out here (stop = 103 - 0.5*4 = 101,
    # this bar's low of 90 blows through it) — the hold-to-close variant has
    # no target/stop at all, so it must stay flat-signal (None) regardless.
    candles.append(_candle(31, 103.0, 103.0, 90.0, 95.0))
    assert signal(candles, 31, True) is None
    # Even a huge favorable spike doesn't trigger an exit either — only the
    # harness's own end-of-day force-close ever closes this position.
    candles.append(_candle(32, 95.0, 200.0, 95.0, 200.0))
    assert signal(candles, 32, True) is None


def test_orb_holdclose_does_not_fire_a_second_entry_while_holding():
    candles = [_candle(i, 100.0, 102.0, 98.0, 100.0) for i in range(30)]
    candles.append(_candle(30, 100.0, 103.0, 100.0, 103.0))  # breakout

    signal = make_orb_holdclose_signal()
    assert signal(candles, 30, False) == "buy"
    assert signal(candles, 31, True) is None


def test_orb_holdclose_second_entry_blocked_only_after_a_confirmed_fill_fresh_instance_unaffected():
    # 2026-09-11 fix: same restructuring as test_orb_second_entry_blocked_
    # only_after_a_confirmed_fill_fresh_instance_unaffected above, adapted
    # to holdclose's simpler "ride to EOD" exit (no target/stop to avoid
    # hitting — confirm the fill via a single has_position=True call, go
    # flat, then check a later breakout is blocked). The old version called
    # the SAME instance twice with has_position=False both times, which
    # never confirms a fill — under the fix, entries_today is only
    # incremented once has_position observes a CONFIRMED fill, so that old
    # sequence no longer demonstrates the daily cap.
    candles = [_candle(i, 100.0, 102.0, 98.0, 100.0) for i in range(30)]
    candles.append(_candle(30, 100.0, 103.0, 100.0, 103.0))  # breakout

    first = make_orb_holdclose_signal()
    assert first(candles, 30, False) == "buy"
    candles.append(_candle(31, 103.0, 103.0, 100.0, 103.0))  # fill confirmation bar
    assert first(candles, 31, True) is None  # fill confirmed here (entries_today now 1)

    # Position closes (the signal doesn't care how — only that it's flat
    # again) — a fresh breakout later the same day must now be blocked.
    candles.append(_candle(32, 103.0, 110.0, 103.0, 110.0))  # would-be second breakout
    assert first(candles, 32, False) is None  # entries_today >= 1 (confirmed), correctly blocked

    second = make_orb_holdclose_signal()
    assert second(candles, 30, False) == "buy"


def test_orb_holdclose_unfilled_entry_does_not_consume_the_daily_cap():
    # 2026-09-11 fix regression test, holdclose variant: a proposed entry
    # the harness never actually fills (has_position stays False on every
    # subsequent call, since no fill was confirmed) must not consume
    # entries_today. A later, separate entry opportunity must still fire.
    candles = [_candle(i, 100.0, 102.0, 98.0, 100.0) for i in range(30)]
    candles.append(_candle(30, 100.0, 103.0, 100.0, 103.0))  # breakout, never gets filled by the harness

    signal = make_orb_holdclose_signal()
    assert signal(candles, 30, False) == "buy"
    # Harness's own simulate_fill call (outside this signal function) found
    # no fillable data and never opened a position -- the NEXT call still
    # sees has_position=False, unlike a confirmed fill which would flip it
    # to True. entries_today must still be 0 at this point.
    candles.append(_candle(31, 103.0, 110.0, 103.0, 110.0))
    assert signal(candles, 31, False) == "buy"  # still fires -- the day's cap was never actually consumed


# --- volume spike --------------------------------------------------------


def _baseline_candles(n: int = 20, volume: int = 1000) -> list[HistoricalMinuteCandle]:
    return [_candle(i, 100.0, 100.5, 99.5, 100.0, v=volume) for i in range(n)]


def test_volume_spike_fires_when_all_three_conditions_clear():
    candles = _baseline_candles()
    # close in top 25% of (low=99, high=101) range: >= 99 + 0.75*2 = 100.5
    # bar return (close-open)/open > 0.30%: open=100.0, close=100.5 -> 0.50%
    candles.append(_candle(20, 100.0, 101.0, 99.0, 100.5, v=5000))  # 5x median (1000)

    signal = make_volume_spike_signal()
    assert signal(candles, 20, False) == "buy"


def test_volume_spike_does_not_fire_when_close_is_at_the_bars_low():
    candles = _baseline_candles()
    # Volume and bar return clear, but close == low fails the range-position leg.
    candles.append(_candle(20, 100.0, 101.0, 99.0, 99.0, v=5000))

    signal = make_volume_spike_signal()
    assert signal(candles, 20, False) is None


def test_volume_spike_does_not_fire_before_20_bars_of_history():
    candles = _baseline_candles(n=19)
    candles.append(_candle(19, 100.0, 101.0, 99.0, 100.5, v=5000))

    signal = make_volume_spike_signal()
    assert signal(candles, 19, False) is None


def test_volume_spike_respects_5_minute_lockout_after_an_exit():
    candles = _baseline_candles()
    candles.append(_candle(20, 100.0, 101.0, 99.0, 100.5, v=5000))  # entry bar

    signal = make_volume_spike_signal()
    assert signal(candles, 20, False) == "buy"

    # Time-stop exit at bar 30 (10 bars later).
    for i in range(21, 30):
        candles.append(_candle(i, 100.5, 100.6, 100.4, 100.5, v=1000))
    candles.append(_candle(30, 100.5, 100.6, 100.4, 100.5, v=1000))
    assert signal(candles, 30, True) == "sell"  # 30 - 20 = 10 bars -> time stop

    # Within the 5-minute lockout (bars 31-34), a fresh qualifying spike must not fire.
    candles.append(_candle(31, 100.0, 101.0, 99.0, 100.5, v=5000))
    assert signal(candles, 31, False) is None


def test_volume_spike_fourth_same_day_entry_does_not_fire():
    candles = _baseline_candles()
    signal = make_volume_spike_signal()
    idx = 20

    for entry_num in range(3):
        candles.append(_candle(idx, 100.0, 101.0, 99.0, 100.5, v=5000))
        assert signal(candles, idx, False) == "buy"
        idx += 1
        # Immediate stop-loss exit next bar to free up flat state quickly.
        candles.append(_candle(idx, 100.5, 100.5, 99.0, 99.5, v=1000))
        assert signal(candles, idx, True) == "sell"
        idx += 1 + 5  # clear the 5-minute lockout before the next attempt
        for gap in range(idx - 5, idx):
            if gap >= len(candles):
                candles.append(_candle(gap, 100.5, 100.6, 100.4, 100.5, v=1000))

    # 4th attempt: entries_today already at VOL_SPIKE_MAX_ENTRIES_PER_DAY (3).
    candles.append(_candle(idx, 100.0, 101.0, 99.0, 100.5, v=5000))
    assert signal(candles, idx, False) is None


def test_volume_spike_unfilled_entry_does_not_consume_the_daily_cap():
    # 2026-09-11 fix regression test, volume-spike variant: a proposed entry
    # the harness never actually confirms (has_position stays False on the
    # NEXT call too, since no fill happened) must not consume entries_today
    # — otherwise 3 consecutive unfilled proposals could exhaust
    # VOL_SPIKE_MAX_ENTRIES_PER_DAY on pure phantom entries and block every
    # real, fillable spike for the rest of the day.
    candles = _baseline_candles()
    candles.append(_candle(20, 100.0, 101.0, 99.0, 100.5, v=5000))  # qualifying spike, proposed entry

    signal = make_volume_spike_signal()
    assert signal(candles, 20, False) == "buy"
    # Harness's simulate_fill never resolved this proposal (e.g. no
    # fillable data left) — has_position correctly stays False on the next
    # call, unlike a confirmed fill which would flip it True.
    candles.append(_candle(21, 100.5, 100.6, 100.4, 100.5, v=1000))
    assert signal(candles, 21, False) is None  # no NEW spike on this bar -- not a cap block, just no signal

    # A later, genuinely qualifying spike must still fire -- entries_today
    # was never actually incremented for the unfilled first attempt.
    candles.append(_candle(22, 100.0, 101.0, 99.0, 100.5, v=5000))
    assert signal(candles, 22, False) == "buy"


def test_volume_spike_stop_fires_on_an_intrabar_breach_that_closes_back_above_it():
    # 2026-09-11 fix: the stop must compare against cur.low (the worst
    # price seen intrabar), not cur.close -- a bar that trades through the
    # stop level and closes back up is a real breach a market-on-touch stop
    # would have filled near, previously missed entirely by a close-only check.
    candles = _baseline_candles()
    candles.append(_candle(20, 100.0, 101.0, 99.0, 100.5, v=5000))  # entry, entry_price=100.5
    signal = make_volume_spike_signal()
    assert signal(candles, 20, False) == "buy"

    # stop = 100.5 * (1 - 0.004) = 100.098; this bar's low breaches it but
    # closes back above -- a close-only check would have missed the breach.
    candles.append(_candle(21, 100.4, 100.6, 100.0, 100.4, v=1000))
    assert signal(candles, 21, True) == "sell"


# --- VWAP reversion --------------------------------------------------------


# All VWAP fixtures start their bars at minute-offset 30 (09:30 KST) so
# every bar under test falls inside the 09:30-14:30 entry window — the
# entry-window check would otherwise mask the down-bar-count logic these
# tests exist to exercise (a bar before 09:30 returns None from the window
# check alone, regardless of what the down-bar condition would have said).


def test_vwap_reversion_does_not_fire_with_only_2_down_bars():
    # Bars far below a VWAP built from a much higher opening, with only 2
    # consecutive down-bars (need 3) — index 0 is flat, not down, so the
    # last-3-bars window at idx=2 contains only 2 down bars.
    candles = [_candle(30, 110.0, 110.0, 110.0, 110.0, v=1000)]
    candles.append(_candle(31, 105.0, 105.0, 100.0, 101.0, v=1000))  # down bar 1
    candles.append(_candle(32, 101.0, 101.0, 97.0, 98.0, v=1000))    # down bar 2 (only 2 so far)

    signal = make_vwap_reversion_signal()
    for idx in range(len(candles)):
        assert signal(candles, idx, False) is None


def test_vwap_reversion_fires_when_band_and_3_down_bars_both_hold():
    candles = [_candle(30, 110.0, 110.0, 110.0, 110.0, v=1000)]
    candles.append(_candle(31, 105.0, 105.0, 100.0, 101.0, v=1000))  # down bar
    candles.append(_candle(32, 101.0, 101.0, 97.0, 98.0, v=1000))    # down bar
    candles.append(_candle(33, 98.0, 98.0, 90.0, 91.0, v=1000))      # 3rd consecutive down-bar, well below VWAP

    signal = make_vwap_reversion_signal()
    for idx in range(3):
        assert signal(candles, idx, False) is None
    assert signal(candles, 3, False) == "buy"


def test_vwap_reversion_time_stop_fires_at_the_20_minute_boundary():
    candles = [_candle(30, 110.0, 110.0, 110.0, 110.0, v=1000)]
    candles.append(_candle(31, 105.0, 105.0, 100.0, 101.0, v=1000))
    candles.append(_candle(32, 101.0, 101.0, 97.0, 98.0, v=1000))
    candles.append(_candle(33, 98.0, 98.0, 90.0, 91.0, v=1000))  # entry bar, idx=3

    signal = make_vwap_reversion_signal()
    for idx in range(3):
        signal(candles, idx, False)
    assert signal(candles, 3, False) == "buy"

    # Hold with price staying below VWAP and above the stop, so neither the
    # VWAP-reached exit nor the stop fires before the 20-minute time stop.
    # entry_idx=3, so minutes_held = idx - 3; the time stop needs
    # minutes_held >= 20, i.e. idx == 23.
    for i in range(4, 24):
        candles.append(_candle(30 + i, 91.0, 91.2, 90.8, 91.0, v=1000))

    for idx in range(4, 23):  # minutes_held 1..19 -> no exit yet
        assert signal(candles, idx, True) is None
    assert signal(candles, 23, True) == "sell"  # minutes_held == 20 -> time stop


def test_vwap_reversion_stop_fires_on_an_intrabar_breach_that_closes_back_above_it():
    # 2026-09-11 fix: the stop compares cur.low (worst intrabar price), not
    # cur.close only -- the reversion target (cur.close >= vwap) stays
    # close-only by design, it is a genuinely different question.
    candles = [_candle(30, 110.0, 110.0, 110.0, 110.0, v=1000)]
    candles.append(_candle(31, 105.0, 105.0, 100.0, 101.0, v=1000))
    candles.append(_candle(32, 101.0, 101.0, 97.0, 98.0, v=1000))
    candles.append(_candle(33, 98.0, 98.0, 90.0, 91.0, v=1000))  # entry bar, idx=3, entry_price=91.0

    signal = make_vwap_reversion_signal()
    for idx in range(3):
        signal(candles, idx, False)
    assert signal(candles, 3, False) == "buy"

    # stop = 91.0 * (1 - 0.005) = 90.545; this bar's low breaches it but
    # closes back above (and stays below vwap, so the target clause can't
    # be what fires it) -- a close-only check would have missed the breach.
    candles.append(_candle(34, 90.9, 91.0, 90.3, 90.9, v=1000))
    assert signal(candles, 4, True) == "sell"
