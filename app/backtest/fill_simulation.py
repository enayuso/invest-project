"""Slippage/execution-latency simulation for scalping backtests (Phase 2 of
docs/scalping-backtest-infra-plan.md). Answers "if the Entry Engine reacted
to a signal N seconds slower, what price would it actually have gotten?" —
the missing piece that made reaction-speed comparison impossible before this.

Deliberately separate from app.execution.ledger (the live/paper execution
path) — this module never touches a real or paper order, it only replays
app.backtest.minute_candles.HistoricalMinuteCandle rows that already
happened. Nothing here is wired into live trading.

Missing-minute policy (see minute_candles.py's Phase 1 finding that KIS's
own historical minute-chart response can have real gaps, e.g. a confirmed
30-minute gap in 000660.KS on 2026-07-13): this module does NOT invent a
price for a missing minute. `simulate_fill` finds the first candle at or
after the requested execution time, so a gap simply shows up as
`realized_latency_seconds` being larger than the requested
`latency_seconds` — the model reports "it would have taken longer to fill,
because there was no print for a while," rather than guessing a fill price
that never actually traded.

One-minute granularity limitation: a candle's `time` is treated as its bar
start, and `open` is used as the fill price — the earliest tradeable print
once the target second is reached. This can't resolve sub-minute latency
differences (e.g. 2s vs 40s within the same bar look identical) — a real
limitation of minute-bar data, not something this module can fix. Comparing
latency values whose difference is smaller than the bar width GENUINELY
isn't measurable with this data source, and callers should not treat such a
comparison as informative (see compare_reaction_speeds's docstring).
"""

from __future__ import annotations

from dataclasses import dataclass

from app.backtest.minute_candles import HistoricalMinuteCandle


def _to_seconds(hhmmss: str) -> int:
    h, m, s = int(hhmmss[:2]), int(hhmmss[2:4]), int(hhmmss[4:6])
    return h * 3600 + m * 60 + s


def _from_seconds(total_seconds: int) -> str:
    h, rem = divmod(total_seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}{m:02d}{s:02d}"


# 1-minute bars: HistoricalMinuteCandle.time is the bar's START, but every
# entry signal in this folder (scalping_signals.py, gap_reversion.py)
# decides off `cur.close` — information only knowable once the bar ENDS,
# i.e. at bar start + BAR_DURATION_SECONDS. See decision_time_for_bar.
BAR_DURATION_SECONDS = 60


def decision_time_for_bar(candle: HistoricalMinuteCandle) -> str:
    """The moment a decision based on `candle`'s close was actually
    knowable — bar start + BAR_DURATION_SECONDS, never the bar's own raw
    `time` (its start). Passing a bar's raw start as `simulate_fill`'s
    `signal_time` lets latency_seconds=0 fill at that SAME bar's own open —
    a price that printed 60s before the triggering close existed — and
    collapses every latency in (0, 60] into an identical bar-end fill."""
    return _from_seconds(_to_seconds(candle.time) + BAR_DURATION_SECONDS)


@dataclass
class SimulatedFill:
    signal_time: str  # HHMMSS the signal fired at
    requested_latency_seconds: float
    filled_time: str  # HHMMSS of the bar actually used
    fill_price: float  # before slippage
    slippage_bps: float
    side: str  # "buy" or "sell"
    executed_price: float  # after slippage
    realized_latency_seconds: float  # filled_time - signal_time, >= requested
    gap_inflated: bool  # True if realized > requested due to a missing-minute gap


def simulate_fill(
    candles: list[HistoricalMinuteCandle],
    signal_time: str,
    latency_seconds: float,
    side: str,
    slippage_bps: float = 5.0,
) -> SimulatedFill | None:
    """Finds the first candle at or after (signal_time + latency_seconds) and
    fills at its `open`. Returns None if no such candle exists in `candles`
    (e.g. the target time is past the last bar of the day — the signal came
    too late in the session for this latency to be fillable at all, which is
    itself a meaningful backtest outcome, not an error).

    `side` must be "buy" or "sell" — slippage always moves the executed
    price against the trader (buy pays more, sell receives less), matching
    this project's existing "slippage never helps you" convention
    (app/execution/ledger.py's transaction_cost_bps is the same direction).

    `signal_time` must be the moment the decision was actually KNOWABLE —
    for a signal that decides off a candle's close, that's
    `decision_time_for_bar(candle)` (bar start + 60s), never a bar's raw
    `time` (its start). Passing a bar start here means latency_seconds=0
    fills at a price that printed before the triggering information existed."""
    if side not in ("buy", "sell"):
        raise ValueError(f"side must be 'buy' or 'sell', got {side!r}")

    signal_s = _to_seconds(signal_time)
    target_s = signal_s + latency_seconds

    candidates = sorted(candles, key=lambda c: c.time)
    chosen: HistoricalMinuteCandle | None = None
    for c in candidates:
        if _to_seconds(c.time) >= target_s:
            chosen = c
            break
    if chosen is None:
        return None

    filled_s = _to_seconds(chosen.time)
    realized_latency = filled_s - signal_s
    slippage_mult = 1 + slippage_bps / 10_000 if side == "buy" else 1 - slippage_bps / 10_000

    return SimulatedFill(
        signal_time=signal_time,
        requested_latency_seconds=latency_seconds,
        filled_time=chosen.time,
        fill_price=chosen.open,
        slippage_bps=slippage_bps,
        side=side,
        executed_price=chosen.open * slippage_mult,
        realized_latency_seconds=realized_latency,
        gap_inflated=realized_latency > latency_seconds + BAR_DURATION_SECONDS,  # tolerance for the bar width itself
    )


def compare_reaction_speeds(
    candles: list[HistoricalMinuteCandle],
    signal_time: str,
    side: str,
    latency_options_seconds: list[float],
    slippage_bps: float = 5.0,
) -> list[SimulatedFill]:
    """Runs simulate_fill for each candidate latency so their executed prices
    can be compared side by side — this is the tool the reaction-speed
    decision (docs/scalping-design.md 7.1) depends on. Entries whose target
    time falls past the last candle are silently omitted (see
    simulate_fill's docstring on None) rather than raising, since "no data
    that late" is an expected outcome near the end of a session, not a bug.

    Caveat callers must respect: `signal_time` should be `decision_time_for_bar
    (candle)` (bar CLOSE), not the bar's raw start — resolution is one bar
    FROM THAT CLOSE, so latency_seconds=0 is the correct "reacted the
    instant the close printed" baseline (fills at the next bar's open), and
    two latency values close enough together to land in the same next bar
    will produce IDENTICAL results here — that's not this function failing
    to distinguish them, it's the 1-minute bar width being coarser than the
    difference being tested (see this module's docstring). Don't read
    "identical price" as "latency doesn't matter" when the tested values
    differ by less than ~60s."""
    results = []
    for latency in latency_options_seconds:
        fill = simulate_fill(candles, signal_time, latency, side, slippage_bps)
        if fill is not None:
            results.append(fill)
    return results
