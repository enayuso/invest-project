"""Statistics/reporting for the scalping viability measurement (see project
memory: scalping-backtest-infra-gap's 2026-08-21 direction re-check). Pure
functions over `app.backtest.scalping_harness.DayResult` objects already
produced elsewhere — no DB, no KIS, no filesystem. `scripts/run_scalping_viability.py`
is the only caller that touches I/O (the Postgres candle cache) and JSON output.

The core statistical judgment call here is the CLUSTERED bootstrap (see
`bootstrap_ci_clustered` below): trades within one ticker-day are not
independent draws — they share a gap-day, a regime, or simply the fact that
one bad opening print poisons every signal built on that day's opening
range or session VWAP. Treating each trade as an independent observation
(the naive per-trade t-test also computed here for comparison) understates
the true uncertainty and would make a lucky sample look more conclusive
than it is — exactly the kind of false confidence a real viability
threshold has to guard against.
"""

from __future__ import annotations

import random
import statistics
from dataclasses import dataclass
from datetime import date
from enum import Enum

from app.backtest.scalping_harness import DayResult, ScalpingTrade
from app.execution.costs import transaction_cost_bps_for

# PRE-REGISTERED 2026-08-24 — DO NOT TUNE
# A partial session (e.g. the confirmed 2026-07-13 ~30-minute gap) silently
# redefines the opening range / session VWAP baseline for any signal built
# on "the first N bars" or "cumulative volume so far" — a short day isn't a
# smaller sample of a normal day, it's a different, unmodeled regime.
DEFAULT_MIN_CANDLES = 300

# --- verdict thresholds (rationale inline on each) -------------------------

# PRE-REGISTERED 2026-08-24 — DO NOT TUNE
# Below this, per-ticker and pooled statistics are dominated by sampling
# noise regardless of what the point estimate says.
VIABLE_MIN_TRADES = 400
VIABLE_MIN_TICKER_DAYS = 500
# PRE-REGISTERED 2026-08-24 — DO NOT TUNE
# The clustered-CI lower bound must clear zero with real margin (not just
# graze it) — 2.0bp is small relative to the ~30-40bp round-trip cost this
# whole measurement exists to test against, but it's enough to say "this
# isn't zero by luck."
VIABLE_CI_LOWER_BOUND_BP = 2.0
# PRE-REGISTERED 2026-08-24 — DO NOT TUNE
# The point estimate itself must clear a meaningfully positive bar, not
# just a statistically-nonzero one — 10bp/round-trip is the threshold this
# measurement treats as "worth building an execution layer for."
VIABLE_MIN_MEAN_NET_BP = 10.0
# PRE-REGISTERED 2026-08-24 — DO NOT TUNE
# A signal that only works on a handful of names isn't a strategy, it's
# overfitting to those names' idiosyncrasies.
VIABLE_MIN_TICKERS_POSITIVE = 10
VIABLE_MIN_TICKERS_TESTED = 15
# PRE-REGISTERED 2026-08-24 — DO NOT TUNE
# Drawdown-to-gain ratio cap — a strategy whose worst drawdown is more than
# 3x its total realized gain isn't survivable in practice even if its mean
# is positive.
VIABLE_MAX_DRAWDOWN_TO_GAIN_RATIO = 3.0

# PRE-REGISTERED 2026-08-24 — DO NOT TUNE
# If gap-inflated fills make up more than 10% of all extracted round trips,
# or the gap-excluded set's mean differs from the included set's mean by
# more than this many bp, the sample itself is telling us the KIS data
# gaps are non-random with respect to the signal (e.g. gaps cluster on
# high-volatility days that also happen to be when the signal fires) — the
# result is unreliable in either direction, not just "noisy."
INCONCLUSIVE_GAP_FRACTION = 0.10
INCONCLUSIVE_GAP_MEAN_DIFF_BP = 20.0


class Verdict(str, Enum):
    VIABLE = "VIABLE"
    NOT_VIABLE = "NOT VIABLE"
    INCONCLUSIVE = "INCONCLUSIVE"


@dataclass
class RoundTrip:
    ticker: str
    trade_date: date
    entry_price: float
    exit_price: float
    net_bp: float
    gap_inflated: bool
    holding_minutes: float
    # The ENTRY leg's actual fill time (HHMMSS) — the only reliable
    # within-date ordering key. holding_minutes is a DURATION, not a clock:
    # sorting by it (as _max_drawdown_bp used to) scrambles same-date trades
    # whose entry times don't correlate with how long they were held.
    entry_filled_time: str


def _hhmmss_to_seconds(hhmmss: str) -> int:
    """Local copy of `app.backtest.fill_simulation._to_seconds`'s HHMMSS
    parsing — that one is module-private, and this module already treats
    `app.backtest.scalping_harness`/`fill_simulation` as read-only per this
    task's scope, so the tiny parse is duplicated here rather than exported."""
    return int(hhmmss[:2]) * 3600 + int(hhmmss[2:4]) * 60 + int(hhmmss[4:6])


def extract_round_trips(day_result: DayResult) -> list[RoundTrip]:
    """Pairs buy->sell trades from one day's `DayResult.trades` into closed
    round trips. `net_bp` replicates
    `app.backtest.scalping_harness._compute_return_pct`'s per-leg fee
    formula exactly (same `transaction_cost_bps_for` call, same
    buy-pays-more/sell-receives-less direction) so a single round trip's
    number here is consistent with what that function compounds across a
    whole day — this function does not reinvent the formula, it applies it
    per-trade instead of compounding it."""
    round_trips: list[RoundTrip] = []
    open_trade: ScalpingTrade | None = None
    for trade in day_result.trades:
        if trade.side == "buy":
            open_trade = trade
        elif trade.side == "sell" and open_trade is not None:
            buy_fee_rate = transaction_cost_bps_for(day_result.ticker, "buy") / 10_000
            sell_fee_rate = transaction_cost_bps_for(day_result.ticker, "sell") / 10_000
            entry_price = open_trade.fill.executed_price
            exit_price = trade.fill.executed_price
            equity_mult = (exit_price * (1 - sell_fee_rate)) / (entry_price * (1 + buy_fee_rate))
            # Fill-to-fill, not signal-to-fill: open_trade.signal_time is
            # when the ENTRY decision fired, before its own fill latency —
            # using it here overstated every hold by however long the entry
            # itself took to fill (2026-09-11 fix).
            holding_minutes = (
                _hhmmss_to_seconds(trade.fill.filled_time) - _hhmmss_to_seconds(open_trade.fill.filled_time)
            ) / 60.0
            round_trips.append(RoundTrip(
                ticker=day_result.ticker,
                trade_date=day_result.trade_date,
                entry_price=entry_price,
                exit_price=exit_price,
                net_bp=(equity_mult - 1) * 10_000,
                gap_inflated=open_trade.fill.gap_inflated or trade.fill.gap_inflated,
                holding_minutes=holding_minutes,
                entry_filled_time=open_trade.fill.filled_time,
            ))
            open_trade = None
    return round_trips


@dataclass
class ExclusionAccounting:
    excluded_gap_count: int
    excluded_gap_mean_net_bp: float | None
    excluded_short_day_count: int  # ticker-days dropped for candle count < min_candles
    included_count: int


def split_excluded(round_trips: list[RoundTrip]) -> tuple[list[RoundTrip], list[RoundTrip]]:
    """Returns (included, excluded_for_gap) — gap-inflated legs are dropped
    from the viability statistics entirely (see `SimulatedFill.gap_inflated`'s
    docstring: a real KIS data-quality issue, not a signal-quality one)."""
    included = [rt for rt in round_trips if not rt.gap_inflated]
    excluded = [rt for rt in round_trips if rt.gap_inflated]
    return included, excluded


@dataclass
class SummaryStats:
    n_trades: int
    n_ticker_days: int
    mean_net_bp: float
    median_net_bp: float
    stdev_net_bp: float
    win_rate: float
    p5_net_bp: float
    p95_net_bp: float
    mean_holding_minutes: float
    worst_trade_net_bp: float
    max_drawdown_bp: float


def _percentile(sorted_values: list[float], pct: float) -> float:
    if not sorted_values:
        return 0.0
    idx = min(len(sorted_values) - 1, max(0, round(pct / 100 * (len(sorted_values) - 1))))
    return sorted_values[idx]


def _max_drawdown_bp(round_trips: list[RoundTrip]) -> float:
    """Equal-weight compounded equity curve, pooled across the whole sample
    in chronological order (not per-ticker — this measures the drawdown an
    account running this signal across all 15 tickers would actually see).
    Returned in bp terms (positive number = size of the drawdown).

    Ordered by `entry_filled_time` (the entry leg's real fill clock), not
    `holding_minutes` — a duration, not a clock, so sorting by it scrambled
    same-date trades whose entry times don't correlate with how long they
    were held (2026-09-11 fix). `ticker` is a deterministic tie-break only:
    two tickers entering in the same 1-minute bar are genuinely unordered
    at this data resolution."""
    ordered = sorted(round_trips, key=lambda rt: (rt.trade_date, rt.entry_filled_time, rt.ticker))
    equity = 1.0
    peak = 1.0
    max_dd = 0.0
    for rt in ordered:
        equity *= 1 + rt.net_bp / 10_000
        peak = max(peak, equity)
        max_dd = max(max_dd, (peak - equity) / peak)
    return max_dd * 10_000


def summarize(round_trips: list[RoundTrip]) -> SummaryStats:
    n = len(round_trips)
    if n == 0:
        return SummaryStats(0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    net_bps = [rt.net_bp for rt in round_trips]
    sorted_bps = sorted(net_bps)
    ticker_days = {(rt.ticker, rt.trade_date) for rt in round_trips}
    return SummaryStats(
        n_trades=n,
        n_ticker_days=len(ticker_days),
        mean_net_bp=statistics.mean(net_bps),
        median_net_bp=statistics.median(net_bps),
        stdev_net_bp=statistics.stdev(net_bps) if n > 1 else 0.0,
        win_rate=sum(1 for b in net_bps if b > 0) / n,
        p5_net_bp=_percentile(sorted_bps, 5),
        p95_net_bp=_percentile(sorted_bps, 95),
        mean_holding_minutes=statistics.mean(rt.holding_minutes for rt in round_trips),
        worst_trade_net_bp=min(net_bps),
        max_drawdown_bp=_max_drawdown_bp(round_trips),
    )


def per_ticker_summaries(round_trips: list[RoundTrip]) -> dict[str, SummaryStats]:
    by_ticker: dict[str, list[RoundTrip]] = {}
    for rt in round_trips:
        by_ticker.setdefault(rt.ticker, []).append(rt)
    return {ticker: summarize(trips) for ticker, trips in by_ticker.items()}


def bootstrap_ci_naive(round_trips: list[RoundTrip], confidence: float = 0.95) -> tuple[float, float]:
    """Individual-trade t-test CI (normal approximation via
    `statistics.NormalDist`, adequate at the trade counts this measurement
    requires — see VIABLE_MIN_TRADES). Ignores that trades sharing a
    ticker-day are correlated, so this CI is expected to be NARROWER than
    `bootstrap_ci_clustered` below — that narrowness is overconfidence, not
    a better estimate, and is reported only so the two can be compared
    side by side."""
    net_bps = [rt.net_bp for rt in round_trips]
    n = len(net_bps)
    if n < 2:
        return (0.0, 0.0)
    mean = statistics.mean(net_bps)
    sem = statistics.stdev(net_bps) / (n ** 0.5)
    z = statistics.NormalDist().inv_cdf(0.5 + confidence / 2)
    return (mean - z * sem, mean + z * sem)


def bootstrap_ci_clustered(
    round_trips: list[RoundTrip],
    n_bootstrap: int = 10_000,
    confidence: float = 0.95,
    seed: int | None = None,
) -> tuple[float, float]:
    """Resamples whole (ticker, trade_date) BLOCKS with replacement — not
    individual trades — because trades within one ticker-day share a
    common regime (a shared opening print, a shared gap, a shared news
    event), so treating them as independent draws (the naive CI above)
    understates the true sampling uncertainty. Each bootstrap iteration
    recomputes the mean net_bp over every trade contained in the resampled
    blocks (a ticker-day with more trades in it contributes more trades to
    that iteration's mean, same as it would in the real sample)."""
    blocks: dict[tuple[str, date], list[float]] = {}
    for rt in round_trips:
        blocks.setdefault((rt.ticker, rt.trade_date), []).append(rt.net_bp)
    block_values = list(blocks.values())
    if not block_values:
        return (0.0, 0.0)

    rng = random.Random(seed)
    means = []
    for _ in range(n_bootstrap):
        resampled_blocks = [rng.choice(block_values) for _ in block_values]
        pooled = [v for block in resampled_blocks for v in block]
        means.append(statistics.mean(pooled) if pooled else 0.0)

    means.sort()
    lo_pct = (1 - confidence) / 2 * 100
    hi_pct = (1 - (1 - confidence) / 2) * 100
    return (_percentile(means, lo_pct), _percentile(means, hi_pct))


@dataclass
class ViabilityReport:
    verdict: Verdict
    verdict_reason: str
    pooled: SummaryStats
    per_ticker: dict[str, SummaryStats]
    ci_clustered: tuple[float, float]
    ci_naive: tuple[float, float]
    exclusions: ExclusionAccounting
    n_tickers_tested: int
    n_tickers_positive_mean: int


def build_report(
    all_round_trips: list[RoundTrip],
    *,
    excluded_short_day_count: int,
    n_tickers_tested: int,
    n_bootstrap: int = 10_000,
    seed: int | None = None,
) -> ViabilityReport:
    """Runs the full exclusion -> bootstrap -> verdict pipeline. `all_round_trips`
    is every round trip extracted before gap exclusion — this function does
    the gap split itself so the exclusion accounting in the returned report
    is always consistent with what fed the verdict."""
    included, excluded_gap = split_excluded(all_round_trips)

    exclusions = ExclusionAccounting(
        excluded_gap_count=len(excluded_gap),
        excluded_gap_mean_net_bp=(
            statistics.mean(rt.net_bp for rt in excluded_gap) if excluded_gap else None
        ),
        excluded_short_day_count=excluded_short_day_count,
        included_count=len(included),
    )

    pooled = summarize(included)
    per_ticker = per_ticker_summaries(included)
    ci_clustered = bootstrap_ci_clustered(included, n_bootstrap=n_bootstrap, seed=seed)
    ci_naive = bootstrap_ci_naive(included)
    n_tickers_positive = sum(1 for stats in per_ticker.values() if stats.mean_net_bp > 0)

    verdict, reason = _compute_verdict(
        pooled=pooled,
        ci_clustered=ci_clustered,
        exclusions=exclusions,
        n_tickers_tested=n_tickers_tested,
        n_tickers_positive=n_tickers_positive,
        total_round_trips=len(all_round_trips),
    )

    return ViabilityReport(
        verdict=verdict,
        verdict_reason=reason,
        pooled=pooled,
        per_ticker=per_ticker,
        ci_clustered=ci_clustered,
        ci_naive=ci_naive,
        exclusions=exclusions,
        n_tickers_tested=n_tickers_tested,
        n_tickers_positive_mean=n_tickers_positive,
    )


def _compute_verdict(
    *,
    pooled: SummaryStats,
    ci_clustered: tuple[float, float],
    exclusions: ExclusionAccounting,
    n_tickers_tested: int,
    n_tickers_positive: int,
    total_round_trips: int,
) -> tuple[Verdict, str]:
    """The INCONCLUSIVE gap-bias check runs FIRST and, if triggered,
    overrides whatever the CI/threshold checks below would otherwise say —
    a sample where gap exclusion is large or gap-biased is telling us the
    data quality problem is entangled with the result, not that the result
    itself is trustworthy-but-negative."""
    gap_fraction = exclusions.excluded_gap_count / total_round_trips if total_round_trips else 0.0
    if gap_fraction > INCONCLUSIVE_GAP_FRACTION:
        return Verdict.INCONCLUSIVE, (
            f"gap-excluded fraction {gap_fraction:.1%} exceeds "
            f"{INCONCLUSIVE_GAP_FRACTION:.0%} of all extracted round trips"
        )
    if (
        exclusions.excluded_gap_mean_net_bp is not None
        and abs(exclusions.excluded_gap_mean_net_bp - pooled.mean_net_bp) > INCONCLUSIVE_GAP_MEAN_DIFF_BP
    ):
        return Verdict.INCONCLUSIVE, (
            f"gap-excluded mean ({exclusions.excluded_gap_mean_net_bp:.1f}bp) differs from "
            f"included mean ({pooled.mean_net_bp:.1f}bp) by more than {INCONCLUSIVE_GAP_MEAN_DIFF_BP}bp "
            "— gap exclusion is not random with respect to the signal"
        )

    if pooled.n_trades < VIABLE_MIN_TRADES:
        return Verdict.NOT_VIABLE, f"only {pooled.n_trades} trades, below the {VIABLE_MIN_TRADES} minimum"
    if pooled.n_ticker_days < VIABLE_MIN_TICKER_DAYS:
        return Verdict.NOT_VIABLE, (
            f"only {pooled.n_ticker_days} (ticker, date) pairs, below the {VIABLE_MIN_TICKER_DAYS} minimum"
        )
    if ci_clustered[0] <= VIABLE_CI_LOWER_BOUND_BP:
        return Verdict.NOT_VIABLE, (
            f"clustered-bootstrap CI lower bound {ci_clustered[0]:.2f}bp does not exceed "
            f"{VIABLE_CI_LOWER_BOUND_BP}bp"
        )
    if pooled.mean_net_bp < VIABLE_MIN_MEAN_NET_BP:
        return Verdict.NOT_VIABLE, (
            f"pooled mean net_bp {pooled.mean_net_bp:.2f} below the {VIABLE_MIN_MEAN_NET_BP}bp minimum"
        )
    if n_tickers_positive < VIABLE_MIN_TICKERS_POSITIVE:
        return Verdict.NOT_VIABLE, (
            f"only {n_tickers_positive} of {n_tickers_tested} tickers have positive mean net_bp, "
            f"below the {VIABLE_MIN_TICKERS_POSITIVE} minimum"
        )
    total_gain_bp = max(pooled.mean_net_bp * pooled.n_trades, 1e-9)
    if pooled.max_drawdown_bp > VIABLE_MAX_DRAWDOWN_TO_GAIN_RATIO * total_gain_bp:
        return Verdict.NOT_VIABLE, (
            f"max drawdown {pooled.max_drawdown_bp:.1f}bp exceeds "
            f"{VIABLE_MAX_DRAWDOWN_TO_GAIN_RATIO}x total gain ({total_gain_bp:.1f}bp)"
        )

    return Verdict.VIABLE, "all viability thresholds met"
