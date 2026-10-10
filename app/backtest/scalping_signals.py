"""Three candidate intraday Entry Engine signals for the scalping viability
measurement (see project memory: scalping-backtest-infra-gap's 2026-08-21
"direction re-check" — the question this module exists to help answer is
whether ANY OHLCV-derived rule-based signal can clear the ~30-40bp all-in
domestic round-trip cost hurdle established in `app.execution.costs`
(20bp sell-only Securities Transaction Tax as of 2026-01-01 + commission
both sides + spread-crossing slippage), not to produce a strategy that
"looks good" on this system's own cached candles.

All thresholds below were chosen from cost geometry (the round-trip hurdle
above) and published microstructure/momentum literature BEFORE any of them
were run against this system's cached minute candles. They are frozen —
tuning a threshold after seeing its backtest result would make the result
answer "did we find parameters that worked on this specific sample" instead
of "does this class of signal work," which is p-hacking and would defeat
the entire point of running this measurement. If a signal here needs
retuning, that is a new pre-registration with a new date, not an edit to
the constants below.

Each factory returns a fresh, STATEFUL closure implementing
`app.backtest.scalping_harness.EntrySignalFn`. Per that module's own
docstring, `run_scalping_backtest` (the multi-day convenience runner) reuses
one `entry_signal_fn` object across every day in its date range — but
`EntrySignalFn`'s signature carries no `entry_price`, so any signal that
needs to remember an entry price, a lockout timer, or a per-day entry count
(all three below do) MUST hold that as closure state. Reusing one instance
across days therefore leaks state across the day boundary: e.g.
`make_orb_signal()` remembers `entries_today`, and a stale instance would
refuse to take day 2's first breakout because it thinks day 1's entry still
counts. For that reason, callers of these factories (in particular
`scripts/run_scalping_viability.py`) must call the factory again — a brand
new instance — for every (ticker, day), and must call
`app.backtest.scalping_harness.run_scalping_backtest_day` directly rather
than the multi-day `run_scalping_backtest` wrapper, which has no hook to
recreate the signal per day.
"""

from __future__ import annotations

from app.backtest.fill_simulation import decision_time_for_bar, simulate_fill
from app.backtest.minute_candles import HistoricalMinuteCandle
from app.backtest.scalping_harness import EntrySignalFn

# PRE-REGISTERED 2026-08-24 — DO NOT TUNE
# Opening range window: first 30 one-minute bars (09:00-09:29 KST).
ORB_RANGE_BARS = 30
# PRE-REGISTERED 2026-08-24 — DO NOT TUNE
# Entries close at 14:00 KST to leave enough of the session for a target,
# stop, or the harness's own end-of-day force-close to actually happen.
ORB_ENTRY_CUTOFF = "140000"
# PRE-REGISTERED 2026-08-24 — DO NOT TUNE
# Target/stop as multiples of the opening-range width — asymmetric 2:1
# reward:risk is the standard ORB convention (Crabel 1990; realistic-cost
# validation in Zarattini & Aziz 2023).
ORB_TARGET_RANGE_MULT = 1.0
ORB_STOP_RANGE_MULT = 0.5

# PRE-REGISTERED 2026-08-24 — DO NOT TUNE
# Trailing baseline window for the volume-spike signal.
VOL_SPIKE_BASELINE_BARS = 20
# PRE-REGISTERED 2026-08-24 — DO NOT TUNE
# 4x median volume, close in the top quartile of its own bar range, and a
# >0.30% bar return are the three legs of "this bar was informed buying,
# not noise" (volume-return co-movement: Karpoff 1987).
VOL_SPIKE_VOLUME_MULT = 4.0
VOL_SPIKE_RANGE_POSITION = 0.75
VOL_SPIKE_MIN_BAR_RETURN = 0.0030
# PRE-REGISTERED 2026-08-24 — DO NOT TUNE
VOL_SPIKE_MAX_ENTRIES_PER_DAY = 3
VOL_SPIKE_LOCKOUT_MINUTES = 5
VOL_SPIKE_TIME_STOP_BARS = 10
VOL_SPIKE_STOP_PCT = -0.0040

# PRE-REGISTERED 2026-08-24 — DO NOT TUNE
# VWAP-reversion entry/exit window and thresholds — transitory inventory
# pressure reverting toward VWAP (Roll 1984; Nagel 2012).
VWAP_ENTRY_START = "093000"
VWAP_ENTRY_END = "143000"
VWAP_BAND_PCT = 0.008
VWAP_DOWNBAR_COUNT = 3
VWAP_STOP_PCT = -0.0050
VWAP_TIME_STOP_MINUTES = 20


def _time_to_minutes(hhmmss: str) -> int:
    """`HistoricalMinuteCandle.time` is "HHMMSS" (minute_candles.py's own
    schema comment: `time TEXT ... -- HHMMSS, stck_cntg_hour`). Converted to
    minutes-since-midnight once here so every signal below compares window
    boundaries the same way instead of re-deriving it three times."""
    return int(hhmmss[:2]) * 60 + int(hhmmss[2:4])


def make_orb_signal(latency_seconds: float = 30.0, slippage_bps: float = 5.0) -> EntrySignalFn:
    """Opening-range breakout. See this module's docstring — construct a
    fresh instance per (ticker, trading day); reusing one across days makes
    `entries_today` and `or_high`/`or_low` from day 1 leak into day 2.

    `latency_seconds`/`slippage_bps` MUST match whatever
    `app.backtest.scalping_harness.run_scalping_backtest_day` is actually
    called with (both default to that function's own defaults) — target/
    stop are re-anchored to the REAL entry fill price (2026-09-11 fix,
    below), which this closure can only compute by replaying the same
    `simulate_fill` call the harness will make; a mismatched latency/
    slippage here would silently anchor to a fill price the harness never
    actually used."""
    state = {
        "or_high": None,
        "or_low": None,
        "entries_today": 0,
        "entry_price": None,
        "target": None,
        "stop": None,
        # Set once the breakout bar fires; cleared (anchored=True) once the
        # real fill price is known and target/stop are re-anchored to it.
        "signal_idx": None,
        "anchored": False,
    }

    def signal(candles: list[HistoricalMinuteCandle], idx: int, has_position: bool) -> str | None:
        if idx < ORB_RANGE_BARS:
            return None  # still inside the opening-range window itself (bars 0..29)

        if state["or_high"] is None:
            range_bars = candles[:ORB_RANGE_BARS]
            state["or_high"] = max(c.high for c in range_bars)
            state["or_low"] = min(c.low for c in range_bars)

        cur = candles[idx]

        if has_position:
            # The breakout bar's cur.close is only a PROVISIONAL entry
            # price — the real fill (per app.backtest.scalping_harness)
            # lands at a LATER bar's open, via simulate_fill. Re-anchor
            # target/stop to that real fill price as soon as it's known,
            # instead of measuring reward:risk from a price the position
            # was never actually filled at (2026-09-11 fix).
            if not state["anchored"]:
                fill = simulate_fill(
                    candles, decision_time_for_bar(candles[state["signal_idx"]]),
                    latency_seconds, "buy", slippage_bps,
                )
                if fill is None:
                    return None  # the harness itself never filled this entry
                fill_idx = next(i for i, c in enumerate(candles) if c.time == fill.filled_time)
                if idx < fill_idx:
                    # Not filled yet as of this bar — reading fill_idx's own
                    # open here (before it happens) would be look-ahead.
                    return None
                or_width = state["or_high"] - state["or_low"]
                state["entry_price"] = fill.executed_price
                state["target"] = fill.executed_price + ORB_TARGET_RANGE_MULT * or_width
                state["stop"] = fill.executed_price - ORB_STOP_RANGE_MULT * or_width
                state["anchored"] = True
                # Count the entry against today's cap HERE — the moment the
                # fill is actually CONFIRMED — not at proposal time below
                # (2026-09-11 fix). Before this, a proposed breakout whose
                # simulate_fill call here (or the harness's own, upstream)
                # never resolved to a real fill still permanently consumed
                # this day's one-entry cap, silently blocking a later,
                # genuinely fillable breakout on the same day. A day with a
                # legitimately truncated candle feed (more likely since the
                # minute_candles.py _num-split fix started dropping corrupted
                # bars instead of keeping them at a wrong 0.0) could then
                # report zero real trades when a complete session might have
                # produced one.
                # 확정 체결 시점에만 당일 진입횟수를 소모 — 미체결 제안 시점에는 소모하지 않음.
                state["entries_today"] += 1
            if cur.high >= state["target"] or cur.low <= state["stop"]:
                return "sell"
            return None

        if state["entries_today"] >= 1:
            return None  # one entry per day maximum
        if _time_to_minutes(cur.time) > _time_to_minutes(ORB_ENTRY_CUTOFF):
            return None  # too late in the session for a new entry
        if cur.close > state["or_high"]:
            # Provisional only — re-anchored to the real fill price above,
            # once has_position is True and the harness's own fill resolves.
            state["signal_idx"] = idx
            state["entry_price"] = cur.close
            or_width = state["or_high"] - state["or_low"]
            state["target"] = cur.close + ORB_TARGET_RANGE_MULT * or_width
            state["stop"] = cur.close - ORB_STOP_RANGE_MULT * or_width
            # entries_today is now incremented only once the fill is
            # CONFIRMED (see the "if not state['anchored']" branch above) —
            # not here at mere proposal time. See the comment there for why.
            return "buy"
        return None

    return signal


def make_orb_holdclose_signal() -> EntrySignalFn:
    """Opening-range breakout with a hold-to-close exit instead of an
    ATR-style target/stop.

    PRE-REGISTERED 2026-08-24 as a second, separate hypothesis about why
    `make_orb_signal()` failed viability (mean_net_bp=-63.3, 95% CI
    [-86.0, -39.8], 33.9% win rate, mean hold 125.8 minutes on the
    2026-08-24 15-ticker/60-day measurement — see project memory
    scalping-backtest-infra-gap.md). An Opus design review of that result
    (that figure was measured under scalping_viability's pre-2026-09-11
    signal-to-fill holding_minutes definition, which overstated every hold
    by roughly one entry-fill latency bar — the design conclusion below is
    unaffected, since it turns on the fingerprint shape, not the exact
    minute count)
    flagged the 125.8-minute mean hold alongside a 33.9% win rate as the
    fingerprint of a trend-following payoff being truncated by a stop that
    fires inside the noise band — the published ORB literature (Crabel
    1990; Zarattini & Aziz 2023) evaluates a full-day hold with ATR-based
    position sizing, not an intraday target/stop-out. This variant tests
    that specific hypothesis in isolation: the entry trigger is UNCHANGED
    from make_orb_signal() (ORB_RANGE_BARS/ORB_ENTRY_CUTOFF above are not
    being retuned — same numbers, same "DO NOT TUNE" freeze), only the exit
    policy differs. This is a new pre-registration, not an edit to
    make_orb_signal(), so both can be compared head-to-head against the
    same entries.

    Never proposes "sell" itself — the position rides until
    `run_scalping_backtest_day`'s own end-of-day force-close realizes it at
    the session's last candle. Construct a fresh instance per (ticker,
    trading day) — same reasoning as this module's other factories."""
    state = {
        "or_high": None,
        "or_low": None,
        "entries_today": 0,
        # Tracks whether THIS position's confirmed fill has already been
        # counted against entries_today (2026-09-11 fix — see make_orb_
        # signal's own comment for the general bug this closes: incrementing
        # entries_today at mere proposal time, before the harness's
        # simulate_fill call confirms a real fill happened, lets an
        # unfilled proposal near a truncated session phantom-consume the
        # day's entry budget). Unlike make_orb_signal, this variant has no
        # target/stop re-anchoring logic to piggyback the count onto, so a
        # dedicated flag is used instead.
        "entry_counted": False,
    }

    def signal(candles: list[HistoricalMinuteCandle], idx: int, has_position: bool) -> str | None:
        if idx < ORB_RANGE_BARS:
            return None  # still inside the opening-range window itself (bars 0..29)

        if state["or_high"] is None:
            range_bars = candles[:ORB_RANGE_BARS]
            state["or_high"] = max(c.high for c in range_bars)
            state["or_low"] = min(c.low for c in range_bars)

        if has_position:
            # has_position is only ever True here once the harness's own
            # simulate_fill call has actually confirmed a fill (see
            # scalping_harness.run_scalping_backtest_day) — count the entry
            # HERE, on first observing it, not back at proposal time below.
            # 확정 체결(has_position=True) 시점에만 1회 카운트.
            if not state["entry_counted"]:
                state["entries_today"] += 1
                state["entry_counted"] = True
            return None  # no target/stop — ride to the harness's EOD force-close

        # Flat again (either never entered yet, or this position already
        # closed) — reset the per-position flag. Not strictly load-bearing
        # today (the cap is exactly 1/day and can never re-open for a second
        # real entry the same day), but kept for defensiveness/consistency
        # with the other two signals' identical pattern, so a future change
        # to the cap doesn't have to remember to add this.
        state["entry_counted"] = False

        cur = candles[idx]
        if state["entries_today"] >= 1:
            return None  # one entry per day maximum
        if _time_to_minutes(cur.time) > _time_to_minutes(ORB_ENTRY_CUTOFF):
            return None  # too late in the session for a new entry
        if cur.close > state["or_high"]:
            # entries_today is incremented only once has_position confirms
            # the fill (above), not here at mere proposal time.
            return "buy"
        return None

    return signal


def make_volume_spike_signal() -> EntrySignalFn:
    """Volume-spike momentum. See this module's docstring — construct a
    fresh instance per (ticker, trading day); reusing one across days makes
    `entries_today`/`last_exit_idx` from day 1 leak into day 2."""
    state = {
        "entries_today": 0,
        "entry_idx": None,
        "entry_price": None,
        "last_exit_idx": None,
        # Same purpose/rationale as make_orb_holdclose_signal's own
        # "entry_counted" flag (2026-09-11 fix) — see that comment and make_
        # orb_signal's for the general bug: entries_today must be counted
        # against a CONFIRMED fill (has_position transitioning True), never
        # at proposal time, or an unfilled proposal near a truncated session
        # phantom-consumes the day's entry budget.
        "entry_counted": False,
    }

    def signal(candles: list[HistoricalMinuteCandle], idx: int, has_position: bool) -> str | None:
        cur = candles[idx]

        if has_position:
            if not state["entry_counted"]:
                state["entries_today"] += 1
                state["entry_counted"] = True
            time_stop = idx - state["entry_idx"] >= VOL_SPIKE_TIME_STOP_BARS
            # Intrabar (cur.low), not close-only (2026-09-11 fix): a bar
            # that traded through the stop level and closed back above it
            # is a real stop breach a market-on-touch stop would have
            # filled near — a close-only check missed it entirely, same
            # convention make_orb_signal's target/stop already used.
            worst_pnl_pct = (cur.low - state["entry_price"]) / state["entry_price"]
            stop_hit = worst_pnl_pct <= VOL_SPIKE_STOP_PCT
            if time_stop or stop_hit:
                state["last_exit_idx"] = idx
                return "sell"
            return None

        # Flat again — reset the per-position counted flag (see the
        # has_position branch above) before the entries_today cap check
        # below, so a subsequent real entry the same day is counted too.
        state["entry_counted"] = False

        if idx < VOL_SPIKE_BASELINE_BARS:
            return None  # insufficient baseline to judge a "spike" against
        if state["entries_today"] >= VOL_SPIKE_MAX_ENTRIES_PER_DAY:
            return None
        if state["last_exit_idx"] is not None and idx - state["last_exit_idx"] < VOL_SPIKE_LOCKOUT_MINUTES:
            return None  # post-exit lockout

        baseline = sorted(c.volume for c in candles[idx - VOL_SPIKE_BASELINE_BARS:idx])
        median_volume = baseline[len(baseline) // 2]
        if median_volume <= 0:
            return None
        bar_range = cur.high - cur.low
        if bar_range <= 0:
            return None
        range_position = (cur.close - cur.low) / bar_range
        bar_return = (cur.close - cur.open) / cur.open if cur.open else 0.0

        if (
            cur.volume >= VOL_SPIKE_VOLUME_MULT * median_volume
            and range_position >= VOL_SPIKE_RANGE_POSITION
            and bar_return > VOL_SPIKE_MIN_BAR_RETURN
        ):
            state["entry_idx"] = idx
            state["entry_price"] = cur.close
            # entries_today is incremented only once has_position confirms
            # the fill (see the has_position branch above), not here at
            # proposal time. Setting entry_idx/entry_price eagerly here is
            # still safe even if this proposal never fills: both are read
            # ONLY inside the has_position branch (reachable only after a
            # confirmed fill), so an unfilled proposal's values are silently
            # overwritten by the next real attempt before ever being read.
            return "buy"
        return None

    return signal


def make_vwap_reversion_signal() -> EntrySignalFn:
    """Session-VWAP mean reversion. See this module's docstring — construct
    a fresh instance per (ticker, trading day); reusing one across days
    would carry the running VWAP accumulator and `entry_idx` state into the
    next day's session."""
    state = {
        "cum_pv": 0.0,
        "cum_v": 0.0,
        "entry_idx": None,
        "entry_price": None,
    }

    def signal(candles: list[HistoricalMinuteCandle], idx: int, has_position: bool) -> str | None:
        cur = candles[idx]
        # Typical price (H+L+C)/3, not close-only — the standard VWAP
        # formula and what most trading venues quote as "VWAP".
        typical_price = (cur.high + cur.low + cur.close) / 3
        state["cum_pv"] += typical_price * cur.volume
        state["cum_v"] += cur.volume
        vwap = state["cum_pv"] / state["cum_v"] if state["cum_v"] else cur.close

        if has_position:
            minutes_held = idx - state["entry_idx"]
            # Intrabar (cur.low) for the STOP only, not the reversion
            # target (cur.close >= vwap) -- a stop check must compare
            # against the worst price seen intrabar (2026-09-11 fix, same
            # reasoning as the volume-spike signal above); the target is a
            # genuine "has price reverted" question, close-only by design.
            worst_pnl_pct = (cur.low - state["entry_price"]) / state["entry_price"]
            if cur.close >= vwap or worst_pnl_pct <= VWAP_STOP_PCT or minutes_held >= VWAP_TIME_STOP_MINUTES:
                return "sell"
            return None

        t = _time_to_minutes(cur.time)
        if t < _time_to_minutes(VWAP_ENTRY_START) or t > _time_to_minutes(VWAP_ENTRY_END):
            return None
        if idx < VWAP_DOWNBAR_COUNT - 1:
            return None  # not enough bars yet to check the 3-down-bar condition

        recent = candles[idx - (VWAP_DOWNBAR_COUNT - 1):idx + 1]
        all_down = all(c.close < c.open for c in recent)
        below_band = cur.close <= vwap * (1 - VWAP_BAND_PCT)

        if below_band and all_down:
            state["entry_idx"] = idx
            state["entry_price"] = cur.close
            return "buy"
        return None

    return signal
