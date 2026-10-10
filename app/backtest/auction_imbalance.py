"""Closing-auction imbalance anticipation — pre-registered 2026-08-24.

Third round-2 candidate signal (Opus design review after all three round-1
signals in app.backtest.scalping_signals failed viability, and after
round-2's ORB hold-to-close variant was refuted and its gap-reversion
variant came back inconclusive on sample size — see project memory
scalping-backtest-infra-gap.md). Ranked by that review as the most
theoretically solid mechanism of the five it proposed, but explicitly
gated on real auction-imbalance data for a production version — this
module implements the OHLCV-only proxy version to test the mechanism at
all before investing in that feed.

Causal story (long-only, matching this project's convention): KRX closes
with a single-price call auction (no continuous trading in the last ~10
minutes; one clearing price at 15:30). Index-fund/ETF creation-redemption
and MOC-equivalent institutional flow must print at that auction price
regardless of price impact — that demand is price-INSENSITIVE. Providing
liquidity ~15 minutes early to a name showing a persistent, market-relative
one-sided drift into the close earns a risk premium for absorbing a forced
flow, not a forecast of an informed move. The counterparty is a mandate,
not an informed trader.

Confirmed against this system's actual cached data: a trading day's cached
candles do NOT end with a normal continuous-session bar around 15:19 — they
jump straight to a final bar timestamped "153000", which IS the real
closing single-price auction print (KRX's actual 15:20-15:30 call-auction
clearing price), distinct from the last continuous-trading price. This
lets the exit be the literal auction execution rather than an
approximation.

## Sign convention — why CLV, not sign(close-open) or the tick rule

There is no trade-direction data in 1-min OHLCV, so a per-bar "which side
of the book got hit" proxy is needed. The close-location-value (CLV,
`(2*close - high - low) / (high - low)`, in [-1, +1]) is continuous — a bar
that closes at its own midpoint contributes ~0 rather than a full ±1 on a
one-tick move, which is far less noisy than a hard sign. It is also
self-contained per bar, unlike the tick rule (compare to the PRIOR bar's
close), which inherits the previous bar's move and would double-count a
single drift across consecutive bars — fatal for a statistic whose whole
purpose is measuring *persistence* of one-sided flow.

## Injection pattern — same reason as app.backtest.gap_reversion

Like that module's `classify_gap_day`, judging whether today's drift is
idiosyncratic (vs market-wide beta shared by every name that day) needs a
market-proxy series that a single-ticker EntrySignalFn has no way to see.
compute_auction_imbalance() is called once per (ticker, day) by the
CALLER, and the resulting AuctionImbalance is injected into
make_auction_imbalance_signal() — unlike gap_reversion, the injected value
is the whole dataclass (not just a bool), because the decile-based
viability analysis (app.backtest.auction_imbalance_report) needs the
continuous imb_resid/drift_resid values alongside every trade, not just a
pass/fail.

PRE-REGISTERED 2026-08-24 — DO NOT TUNE. All thresholds are set from the
mechanism and from this system's own fill-simulation/harness mechanics
(see the constants' own comments), not fit to cached candles. Retuning is
a new pre-registration with a new date, matching this codebase's existing
convention (scalping_signals.py's own docstring)."""

from __future__ import annotations

from dataclasses import dataclass

from app.backtest.minute_candles import HistoricalMinuteCandle
from app.backtest.scalping_harness import EntrySignalFn

# PRE-REGISTERED 2026-08-24 — DO NOT TUNE
# Measurement window for the imbalance/drift proxy — the design review's
# own "~14:30-15:10" window, ending strictly before the decision bar so
# there is no lookahead into the bar being traded on.
AUCTION_WINDOW_START = "143000"
AUCTION_WINDOW_END = "151400"
# The decision bar — 5 minutes of margin before the 15:20 auction freeze,
# so a 30s-latency fill lands at 15:16, still inside continuous trading.
AUCTION_DECISION_TIME = "151500"
# Hard ceiling: app.backtest.fill_simulation.simulate_fill flags a fill as
# gap_inflated once realized latency exceeds latency_seconds + 60s. A
# "buy" fired any later than this would find no continuous bar before the
# 15:30 auction print, realize a ~10+ minute latency, and get the whole
# trade excluded as a data-quality gap rather than counted — this ceiling
# keeps every entry inside the continuous session.
# 2026-09-11: moved 151800 -> 151700 (a mechanics correction following the
# fill_simulation clock fix, not a tuning change — see this project's
# pre-registration rule). scalping_harness now passes
# decision_time_for_bar(candle) (bar start + 60s) as simulate_fill's
# signal_time instead of the bar's raw start, so a signal at this ceiling
# now decides 60s later than before; the ceiling moves back the same 60s
# to keep the same real-world "still fills inside continuous trading"
# guarantee this constant's own derivation depends on.
AUCTION_DECISION_LATEST_TIME = "151700"
# Of the ~45 possible one-minute bars in the window — a window missing
# more than ~20% of its bars can't measure persistence of one-sided flow.
AUCTION_MIN_WINDOW_BARS = 35
# So imb_raw isn't dominated by a single block trade — trivially cleared
# by every large-cap name in this system's trade allowlist.
AUCTION_MIN_WINDOW_NOTIONAL_KRW = 300_000_000
# Data-independent stand-in for "top decile" (~15% net one-sided notional)
# — used only by the secondary, threshold-gated run (require_threshold=True),
# not by the primary unconditional decile study.
AUCTION_IMB_TOP_THRESHOLD = 0.15


@dataclass
class AuctionImbalance:
    imb_raw: float
    market_imb_raw: float
    imb_resid: float
    drift: float
    market_drift: float
    drift_resid: float
    window_bars: int
    window_notional: float
    eligible: bool
    ineligible_reason: str | None


def _window(candles: list[HistoricalMinuteCandle], start: str, end: str) -> list[HistoricalMinuteCandle]:
    return [c for c in candles if start <= c.time <= end]


def _signed_and_notional(candles: list[HistoricalMinuteCandle]) -> tuple[float, float]:
    """Sum of (CLV * notional) and sum of notional over `candles` — the two
    accumulators imb_raw is built from. See this module's own docstring on
    why CLV is the sign proxy."""
    total_signed = 0.0
    total_notional = 0.0
    for c in candles:
        typical_price = (c.high + c.low + c.close) / 3
        notional = typical_price * c.volume
        bar_range = c.high - c.low
        clv = (2 * c.close - c.high - c.low) / bar_range if bar_range > 0 else 0.0
        total_signed += clv * notional
        total_notional += notional
    return total_signed, total_notional


def _vwap(candles: list[HistoricalMinuteCandle]) -> float | None:
    total_pv = 0.0
    total_v = 0.0
    for c in candles:
        typical_price = (c.high + c.low + c.close) / 3
        total_pv += typical_price * c.volume
        total_v += c.volume
    return (total_pv / total_v) if total_v > 0 else None


def _ineligible(window_bars: int, reason: str) -> AuctionImbalance:
    return AuctionImbalance(
        imb_raw=0.0, market_imb_raw=0.0, imb_resid=0.0,
        drift=0.0, market_drift=0.0, drift_resid=0.0,
        window_bars=window_bars, window_notional=0.0,
        eligible=False, ineligible_reason=reason,
    )


def compute_auction_imbalance(
    ticker_candles: list[HistoricalMinuteCandle],
    market_candles: list[HistoricalMinuteCandle],
) -> AuctionImbalance:
    """`ticker_candles`/`market_candles` are one FULL trading day's candles
    each (090000..153000, per this system's actual cache shape), for the
    ticker and the market proxy (069500.KS, KODEX 200 ETF) respectively.
    Fields on an ineligible result are meaningless placeholders — callers
    must check `.eligible` before reading them (same convention as
    app.backtest.gap_reversion.classify_gap_day's `.qualifies`)."""
    ticker_candles = sorted(ticker_candles, key=lambda c: c.time)
    market_candles = sorted(market_candles, key=lambda c: c.time)

    t_window = _window(ticker_candles, AUCTION_WINDOW_START, AUCTION_WINDOW_END)
    m_window = _window(market_candles, AUCTION_WINDOW_START, AUCTION_WINDOW_END)
    if len(t_window) < AUCTION_MIN_WINDOW_BARS:
        return _ineligible(len(t_window), "insufficient ticker window bars")
    if len(m_window) < AUCTION_MIN_WINDOW_BARS:
        return _ineligible(len(t_window), "insufficient market-proxy window bars")

    t_signed, t_notional = _signed_and_notional(t_window)
    m_signed, m_notional = _signed_and_notional(m_window)
    if t_notional < AUCTION_MIN_WINDOW_NOTIONAL_KRW:
        return _ineligible(len(t_window), "insufficient window notional")

    imb_raw = (t_signed / t_notional) if t_notional else 0.0
    market_imb_raw = (m_signed / m_notional) if m_notional else 0.0
    imb_resid = imb_raw - market_imb_raw

    t_pre = _window(ticker_candles, "090000", AUCTION_WINDOW_END)
    m_pre = _window(market_candles, "090000", AUCTION_WINDOW_END)
    t_vwap_pre = _vwap(t_pre)
    m_vwap_pre = _vwap(m_pre)
    if not t_vwap_pre or not m_vwap_pre:
        return _ineligible(len(t_window), "vwap unavailable")

    drift = (t_window[-1].close / t_vwap_pre) - 1
    market_drift = (m_window[-1].close / m_vwap_pre) - 1
    drift_resid = drift - market_drift

    # 2026-09-11: this used to also require a "153000" auction-print bar to
    # exist and carry >= AUCTION_MIN_AUCTION_VOLUME_SHARE of ticker_candles'
    # FULL-DAY volume before marking the day eligible. Both facts are only
    # knowable after the 15:15 decision bar (AUCTION_DECISION_TIME) this
    # eligibility gates entry into — i.e. it selected for days where the
    # closing-auction premise turned out to hold, which is unknowable look-
    # ahead and inflated every decile's measured return. Eligibility below
    # this point depends only on ticker_candles/market_candles data at or
    # before AUCTION_WINDOW_END (151400), same look-ahead-free discipline as
    # the measurement window above.
    return AuctionImbalance(
        imb_raw=imb_raw, market_imb_raw=market_imb_raw, imb_resid=imb_resid,
        drift=drift, market_drift=market_drift, drift_resid=drift_resid,
        window_bars=len(t_window), window_notional=t_notional,
        eligible=True, ineligible_reason=None,
    )


def make_auction_imbalance_signal(
    imb: AuctionImbalance | None, *, require_threshold: bool = False,
) -> EntrySignalFn:
    """`imb` is the result of compute_auction_imbalance() for this specific
    (ticker, day), computed by the caller before this day's backtest loop
    runs — see this module's own docstring on why the value injected is
    the whole dataclass, not just a bool.

    `require_threshold=False` (default) is the PRIMARY run: enters
    unconditionally on every eligible ticker-day, so the decile-sort
    analysis downstream has a full, unconditional sample to sort by
    imb_resid rather than a small pre-filtered one (see
    app.backtest.auction_imbalance_report — this is what fixed
    gap_reversion's 9-trade sample-size problem). `require_threshold=True`
    is the SECONDARY, production-shaped run that only trades the
    proxy-for-"top decile" (AUCTION_IMB_TOP_THRESHOLD) with a
    market-relative upward drift.

    Never proposes "sell" — the position rides to
    run_scalping_backtest_day's own end-of-day force-close, which (per
    this module's docstring) realizes at the literal 153000 auction-print
    bar, since that bar is the last one in this system's cached daily
    data. Construct a fresh instance per (ticker, trading day)."""
    state = {"fired": False}

    def signal(candles: list[HistoricalMinuteCandle], idx: int, has_position: bool) -> str | None:
        if has_position:
            return None  # never self-exits — see docstring above

        if imb is None or not imb.eligible:
            return None
        if require_threshold and not (imb.imb_resid >= AUCTION_IMB_TOP_THRESHOLD and imb.drift_resid > 0):
            return None
        if state["fired"]:
            return None

        cur_time = candles[idx].time
        if cur_time < AUCTION_DECISION_TIME:
            return None
        if cur_time > AUCTION_DECISION_LATEST_TIME:
            return None

        state["fired"] = True
        return "buy"

    return signal
