"""Overnight-gap idiosyncratic reversion — pre-registered 2026-08-24.

Second-round candidate signal (Opus design review after all three round-1
signals in app.backtest.scalping_signals — ORB, volume-spike, VWAP-reversion
— failed viability; see project memory scalping-backtest-infra-gap.md).
Distinct from that module's factories in one structural way: eligibility
for a given (ticker, trading day) is a CROSS-SECTIONAL decision (this
ticker's own overnight gap vs the broader market's same-day gap, sized
against this ticker's own trailing volatility) that cannot be computed from
this ticker's candles alone. classify_gap_day() below does that
classification once per (ticker, day), BEFORE the day's backtest loop runs;
the resulting bool is baked into a fresh make_gap_reversion_signal(qualifies)
instance. Unlike scalping_signals.py's zero-argument factories, this one
necessarily takes an argument — there's no way to decide "is today's gap
idiosyncratic" from inside a signal that only ever sees one ticker's own
candles one bar at a time.

Causal story (long-only, matching this project's convention — see
app.backtest.scalping_harness's own docstring): KRX has no meaningful
continuous pre-market session, so overnight information AND overnight
retail order-queue accumulation both clear in one 09:00 opening-auction
print. A gap fully explained by the broader market's own overnight gap is
information (beta) — fading it is directional risk-taking dressed up as
mean reversion. A large gap specific to one name, with no market-wide
counterpart, is more likely retail order-queue overshoot: a non-informed,
price-insensitive counterparty rather than a forecast of continued
direction. Only gap-DOWN days are tradeable here (buy, expecting a partial
reversion up) — this system is long-only.

PRE-REGISTERED 2026-08-24 — DO NOT TUNE. All thresholds below are set from
the mechanism above, not fit to this system's cached candles. If this needs
retuning, that is a new pre-registration with a new date — see
app.backtest.scalping_signals's own docstring for why."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import pstdev

from app.backtest.minute_candles import HistoricalMinuteCandle
from app.backtest.scalping_harness import EntrySignalFn

# PRE-REGISTERED 2026-08-24 — DO NOT TUNE
# Trailing window (in trading days) used to size "this ticker's own normal
# overnight volatility" — needs to be long enough to not be dominated by one
# recent outlier day, short enough to track a name's current regime.
GAP_TRAILING_DAYS = 20
# A gap must be at least this many trailing-volatility standard deviations
# more negative than the market proxy's own same-day gap to count as
# idiosyncratic rather than market-wide beta.
GAP_RESIDUAL_MIN_SIGMA = 1.5
# Floor so a "statistically large" gap on an extremely low-vol name still
# clears a minimum absolute size before it's worth a ~30-40bp round trip.
GAP_MIN_ABS_PCT = 0.005
# Hold horizon — Opus design review's own "20-60 minutes" window, fixed at
# the midpoint rather than tuned against this system's cached candles.
GAP_HOLD_MINUTES = 30
# Protects against the reversal-strategy tail risk the design review flagged
# explicitly: a gap that was real information, not overshoot, keeps falling
# instead of reverting.
GAP_STOP_PCT = -0.008


@dataclass
class GapClassification:
    gap_pct: float
    market_gap_pct: float
    residual_gap_pct: float
    trailing_vol: float
    qualifies: bool


def _daily_return_stdev(prior_closes: list[float]) -> float:
    """Population stdev of close-to-close returns over whatever trailing
    window is available. Needs at least 3 closes (2 returns) to be
    meaningful — an early ticker-day in a freshly backfilled range simply
    has a shorter trailing sample and returns 0.0, which classify_gap_day
    treats as "cannot classify, does not qualify" rather than dividing by
    zero or raising."""
    if len(prior_closes) < 3:
        return 0.0
    returns = [
        (prior_closes[i] - prior_closes[i - 1]) / prior_closes[i - 1]
        for i in range(1, len(prior_closes))
        if prior_closes[i - 1]
    ]
    return pstdev(returns) if len(returns) >= 2 else 0.0


def classify_gap_day(
    ticker_prior_closes: list[float],
    ticker_prev_close: float,
    ticker_today_open: float,
    market_prev_close: float,
    market_today_open: float,
) -> GapClassification:
    """`ticker_prior_closes` is this ticker's own trailing daily closes,
    oldest-first, NOT including today — used only to size the trailing
    volatility. The gap itself is prev_close -> today_open for both the
    ticker and the market proxy on the same calendar trading day."""
    trailing_vol = _daily_return_stdev(ticker_prior_closes)
    gap_pct = (ticker_today_open - ticker_prev_close) / ticker_prev_close if ticker_prev_close else 0.0
    market_gap_pct = (
        (market_today_open - market_prev_close) / market_prev_close if market_prev_close else 0.0
    )
    residual_gap_pct = gap_pct - market_gap_pct

    qualifies = (
        trailing_vol > 0
        and gap_pct < 0  # long-only: only a gap DOWN is tradeable (fade = buy)
        and abs(gap_pct) >= GAP_MIN_ABS_PCT
        and residual_gap_pct <= -GAP_RESIDUAL_MIN_SIGMA * trailing_vol
    )
    return GapClassification(
        gap_pct=gap_pct, market_gap_pct=market_gap_pct,
        residual_gap_pct=residual_gap_pct, trailing_vol=trailing_vol,
        qualifies=qualifies,
    )


def make_gap_reversion_signal(qualifies: bool) -> EntrySignalFn:
    """`qualifies` is the result of classify_gap_day() for this specific
    (ticker, day), computed by the caller before this day's backtest loop
    runs — see this module's own docstring on why this factory takes an
    argument unlike scalping_signals.py's zero-arg factories.

    Buys at the session's first bar when qualifies is True; exits at a
    fixed time stop (GAP_HOLD_MINUTES) or a stop-loss, whichever comes
    first — no take-profit target, matching the design review's "hold
    20-60 minutes" rather than a price target this module has no
    pre-registered basis to pick. Construct a fresh instance per (ticker,
    trading day) — same reasoning as scalping_signals.py's factories."""
    state = {"entry_idx": None, "entry_price": None}

    def signal(candles: list[HistoricalMinuteCandle], idx: int, has_position: bool) -> str | None:
        if has_position:
            minutes_held = idx - state["entry_idx"]
            # Intrabar (candles[idx].low), not close-only (2026-09-11 fix):
            # the design review's own rationale for this stop is protecting
            # against "a gap that was real information, not overshoot,
            # keeps falling instead of reverting" -- a close-only check
            # misses exactly the bar where that breach happens intrabar and
            # then closes back up.
            worst_pnl_pct = (candles[idx].low - state["entry_price"]) / state["entry_price"]
            if minutes_held >= GAP_HOLD_MINUTES or worst_pnl_pct <= GAP_STOP_PCT:
                return "sell"
            return None

        if qualifies and idx == 0:
            state["entry_idx"] = idx
            state["entry_price"] = candles[idx].close
            return "buy"
        return None

    return signal
