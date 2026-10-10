"""Decile-based viability reporting for the closing-auction imbalance
signal (app.backtest.auction_imbalance, pre-registered 2026-08-24).

Deliberately separate from app.backtest.scalping_viability's
VIABLE/NOT_VIABLE/INCONCLUSIVE machinery. That report's thresholds
(VIABLE_MIN_TRADES=400, VIABLE_MIN_TICKER_DAYS=500) were pre-registered
for round-1's high-frequency signals and are structurally unreachable for
a signal that fires AT MOST ONCE per (ticker, day) — across this system's
15-ticker/~60-day universe that's a hard ceiling of roughly 900 possible
trades even if every single day qualified. This signal's own falsifiable
kill criterion (from the Opus design review that proposed it) is a
decile-sort test — does net return scale monotonically with measured
imbalance magnitude? — which scalping_viability has no machinery for at
all, so this module builds a separate, parallel report rather than
forcing the signal through the wrong tool.

This module also corrects a clustering mistake that would otherwise creep
in silently: this signal holds every one of the 15 tickers over the SAME
14-minute window on the SAME calendar date, so the dominant correlation
in the sample is cross-sectional WITHIN a date, not within a (ticker,
date) pair. scalping_viability.bootstrap_ci_clustered blocks on (ticker,
date) — but each such block here contains exactly one trade (this signal
never trades twice in one ticker-day), so that CI would silently degenerate
to the naive per-trade CI and understate the true uncertainty exactly the
way the clustered bootstrap exists to prevent. bootstrap_ci_date_clustered
below blocks on trade_date alone instead — the correct unit here."""

from __future__ import annotations

import random
import statistics
from dataclasses import dataclass
from datetime import date

from app.backtest.scalping_viability import RoundTrip

# PRE-REGISTERED 2026-08-24 — DO NOT TUNE
# Replaces scalping_viability.VIABLE_MIN_TRADES for a once-per-ticker-day
# structure — see this module's own docstring.
AUCTION_MIN_DECILE_TRADES = 40
# The real sample size for this signal is DATES, not trades (every ticker
# on one date shares that date's common market factor) — see docstring.
AUCTION_MIN_DISTINCT_DATES = 40
# From the design review's original falsifiable kill criterion.
AUCTION_KILL_MIN_TOP_DECILE_BP = 15.0
# Operationalizes "the effect strengthens monotonically with imbalance
# magnitude" — below this, call the decile pattern noise regardless of
# the top decile's own point estimate.
AUCTION_MONOTONE_MIN_SPEARMAN = 0.60
# Within-date permutation p-value ceiling for the top-minus-bottom decile
# spread — the day-neutral test, isolating the cross-sectional imbalance
# signal from market beta shared by every name on that date.
AUCTION_PERM_MAX_P = 0.05
AUCTION_PERM_ITERATIONS = 10_000
# Same rationale as scalping_viability.INCONCLUSIVE_GAP_FRACTION —
# duplicated rather than imported since this module intentionally does
# not depend on that one's verdict machinery.
INCONCLUSIVE_GAP_FRACTION = 0.10


@dataclass
class AuctionTrade:
    round_trip: RoundTrip
    imb_resid: float
    drift_resid: float


@dataclass
class DecileStats:
    decile: int  # 1 (lowest imb_resid, the control arm) .. 10 (highest)
    n_trades: int
    n_dates: int
    imb_resid_min: float
    imb_resid_max: float
    mean_net_bp: float
    median_net_bp: float
    win_rate: float
    ci_date_clustered: tuple[float, float]


@dataclass
class DecileReport:
    deciles: list[DecileStats]
    spearman: float
    permutation_p: float
    excluded_gap_count: int
    excluded_gap_mean_net_bp: float | None
    gap_fraction: float
    data_quality_ok: bool
    top_decile_pass: bool
    verdict: str  # "PASS" | "KILL" | "DIRECTIONAL_INSUFFICIENT" | "DATA_QUALITY_INCONCLUSIVE"
    verdict_reason: str


def _percentile(sorted_values: list[float], pct: float) -> float:
    if not sorted_values:
        return 0.0
    idx = min(len(sorted_values) - 1, max(0, round(pct / 100 * (len(sorted_values) - 1))))
    return sorted_values[idx]


def bootstrap_ci_date_clustered(
    trades: list[AuctionTrade], n_bootstrap: int = 10_000, confidence: float = 0.95, seed: int | None = None,
) -> tuple[float, float]:
    """Resamples whole trade_date blocks with replacement — see this
    module's own docstring on why date, not (ticker, date), is the correct
    cluster unit for this signal."""
    blocks: dict[date, list[float]] = {}
    for t in trades:
        blocks.setdefault(t.round_trip.trade_date, []).append(t.round_trip.net_bp)
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


def _split_deciles(trades: list[AuctionTrade]) -> list[list[AuctionTrade]]:
    """Pooled equal-COUNT deciles sorted by imb_resid ascending — decile 1
    is the lowest (control-arm) imbalance, decile 10 the highest (the one
    the mechanism predicts should be profitable)."""
    ordered = sorted(trades, key=lambda t: t.imb_resid)
    n = len(ordered)
    deciles: list[list[AuctionTrade]] = []
    for d in range(10):
        start = n * d // 10
        end = n * (d + 1) // 10
        deciles.append(ordered[start:end])
    return deciles


def decile_table(
    trades: list[AuctionTrade], n_bootstrap: int = 10_000, seed: int | None = None,
) -> list[DecileStats]:
    deciles = _split_deciles(trades)
    stats_list: list[DecileStats] = []
    for i, bucket in enumerate(deciles, start=1):
        if not bucket:
            stats_list.append(DecileStats(i, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, (0.0, 0.0)))
            continue
        net_bps = [t.round_trip.net_bp for t in bucket]
        dates = {t.round_trip.trade_date for t in bucket}
        stats_list.append(DecileStats(
            decile=i,
            n_trades=len(bucket),
            n_dates=len(dates),
            imb_resid_min=min(t.imb_resid for t in bucket),
            imb_resid_max=max(t.imb_resid for t in bucket),
            mean_net_bp=statistics.mean(net_bps),
            median_net_bp=statistics.median(net_bps),
            win_rate=sum(1 for b in net_bps if b > 0) / len(net_bps),
            ci_date_clustered=bootstrap_ci_date_clustered(bucket, n_bootstrap=n_bootstrap, seed=seed),
        ))
    return stats_list


def _rank(values: list[float]) -> list[float]:
    """Average-rank (ties share the mean of their tied positions) — the
    standard input transform for Spearman's rank correlation."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg_rank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def _spearman(xs: list[float], ys: list[float]) -> float:
    rx, ry = _rank(xs), _rank(ys)
    mean_rx, mean_ry = statistics.mean(rx), statistics.mean(ry)
    cov = sum((a - mean_rx) * (b - mean_ry) for a, b in zip(rx, ry))
    var_x = sum((a - mean_rx) ** 2 for a in rx)
    var_y = sum((b - mean_ry) ** 2 for b in ry)
    if var_x <= 0 or var_y <= 0:
        return 0.0
    return cov / (var_x * var_y) ** 0.5


def spearman_decile(deciles: list[DecileStats]) -> float:
    """Spearman rank correlation between decile index (1..10) and decile
    mean_net_bp — operationalizes "the effect strengthens monotonically
    with imbalance magnitude" from the original kill criterion. Deciles
    with zero trades are dropped rather than treated as a 0.0 data point
    (a coverage gap, not a real observation)."""
    populated = [d for d in deciles if d.n_trades > 0]
    if len(populated) < 3:
        return 0.0
    return _spearman([d.decile for d in populated], [d.mean_net_bp for d in populated])


def within_date_permutation_test(
    trades: list[AuctionTrade], n_permutations: int = AUCTION_PERM_ITERATIONS, seed: int | None = None,
) -> float:
    """Permutes imb_resid values among a date's OWN tickers only (holding
    that date's common market factor fixed) and recomputes the pooled
    top-decile-minus-bottom-decile mean net_bp spread — isolates the
    cross-sectional imbalance signal from market-wide beta shared by
    every name on that date, unlike a naive full-sample permutation.
    Returns the p-value: the fraction of permutations whose spread is >=
    the observed spread (one-sided — the mechanism predicts top > bottom,
    not merely "different")."""
    by_date: dict[date, list[AuctionTrade]] = {}
    for t in trades:
        by_date.setdefault(t.round_trip.trade_date, []).append(t)

    def _top_minus_bottom_spread(all_trades: list[AuctionTrade]) -> float:
        deciles = _split_deciles(all_trades)
        top, bottom = deciles[-1], deciles[0]
        if not top or not bottom:
            return 0.0
        return (
            statistics.mean(t.round_trip.net_bp for t in top)
            - statistics.mean(t.round_trip.net_bp for t in bottom)
        )

    observed = _top_minus_bottom_spread(trades)

    rng = random.Random(seed)
    at_least_as_extreme = 0
    for _ in range(n_permutations):
        permuted: list[AuctionTrade] = []
        for day_trades in by_date.values():
            resids = [t.imb_resid for t in day_trades]
            rng.shuffle(resids)
            permuted.extend(
                AuctionTrade(round_trip=t.round_trip, imb_resid=r, drift_resid=t.drift_resid)
                for t, r in zip(day_trades, resids)
            )
        if _top_minus_bottom_spread(permuted) >= observed:
            at_least_as_extreme += 1

    return at_least_as_extreme / n_permutations


def build_decile_report(
    trades: list[AuctionTrade], *,
    n_bootstrap: int = 10_000, n_permutations: int = AUCTION_PERM_ITERATIONS, seed: int | None = None,
) -> DecileReport:
    """Runs the full gap-exclusion -> decile-sort -> monotonicity ->
    permutation -> verdict pipeline. See this module's own docstring for
    why the verdict categories differ from scalping_viability's."""
    included = [t for t in trades if not t.round_trip.gap_inflated]
    excluded = [t for t in trades if t.round_trip.gap_inflated]
    gap_fraction = len(excluded) / len(trades) if trades else 0.0
    excluded_gap_mean = statistics.mean(t.round_trip.net_bp for t in excluded) if excluded else None

    if gap_fraction > INCONCLUSIVE_GAP_FRACTION:
        return DecileReport(
            deciles=[], spearman=0.0, permutation_p=1.0,
            excluded_gap_count=len(excluded), excluded_gap_mean_net_bp=excluded_gap_mean,
            gap_fraction=gap_fraction, data_quality_ok=False, top_decile_pass=False,
            verdict="DATA_QUALITY_INCONCLUSIVE",
            verdict_reason=f"gap-excluded fraction {gap_fraction:.1%} exceeds {INCONCLUSIVE_GAP_FRACTION:.0%}",
        )

    if not included:
        return DecileReport(
            deciles=[], spearman=0.0, permutation_p=1.0,
            excluded_gap_count=len(excluded), excluded_gap_mean_net_bp=excluded_gap_mean,
            gap_fraction=gap_fraction, data_quality_ok=True, top_decile_pass=False,
            verdict="DATA_QUALITY_INCONCLUSIVE", verdict_reason="no included trades to report on",
        )

    deciles = decile_table(included, n_bootstrap=n_bootstrap, seed=seed)
    spearman = spearman_decile(deciles)
    perm_p = within_date_permutation_test(included, n_permutations=n_permutations, seed=seed)

    top = deciles[-1]
    top_pass = (
        top.n_trades >= AUCTION_MIN_DECILE_TRADES
        and top.n_dates >= AUCTION_MIN_DISTINCT_DATES
        and top.mean_net_bp >= AUCTION_KILL_MIN_TOP_DECILE_BP
        and top.ci_date_clustered[0] > 0
        and spearman >= AUCTION_MONOTONE_MIN_SPEARMAN
        and perm_p <= AUCTION_PERM_MAX_P
    )

    # Whether the top decile even has enough raw sample (trades AND distinct
    # dates) to say anything definitive about it at all. PASS above already
    # requires this (top_pass's first two clauses) — but until this fix, the
    # KILL branch below checked NEITHER: it could fire purely off top.
    # mean_net_bp/ci_date_clustered[0]/spearman computed from whatever the
    # top decile's sample happened to be, even 2-3 trades on 1-2 distinct
    # dates. Since this signal's top decile is, by construction, the
    # smallest slice of an already-small (15-ticker/~60-day, ~900-trade
    # ceiling — see module docstring) universe, an early or partial run can
    # trivially produce a severely underpowered top decile whose metrics
    # look bad from pure noise — and a confidently-worded, permanent "KILL"
    # ("indistinguishable from noise") should only ever fire when there was
    # actually enough data to distinguish signal from noise in the first
    # place. This reuses the SAME pre-registered AUCTION_MIN_DECILE_TRADES/
    # AUCTION_MIN_DISTINCT_DATES thresholds PASS already uses — symmetric
    # application of an existing gate, not a new or retuned threshold.
    # PASS는 이미 표본 크기를 요구하지만 KILL은 그렇지 않아서, 표본이 적은
    # top decile이 순수 노이즈로 확정적 KILL을 유발할 수 있었음 — 대칭 수정.
    top_decile_has_enough_sample = top.n_trades >= AUCTION_MIN_DECILE_TRADES and top.n_dates >= AUCTION_MIN_DISTINCT_DATES

    if top_pass:
        verdict, reason = "PASS", (
            f"top decile mean {top.mean_net_bp:.1f}bp, CI lower {top.ci_date_clustered[0]:.1f}bp, "
            f"spearman {spearman:.2f}, permutation p={perm_p:.4f} — clears all pre-registered PASS thresholds"
        )
    elif (
        top_decile_has_enough_sample
        and (top.mean_net_bp < AUCTION_KILL_MIN_TOP_DECILE_BP or top.ci_date_clustered[0] <= 0)
        and spearman < AUCTION_MONOTONE_MIN_SPEARMAN
    ):
        verdict, reason = "KILL", (
            f"top decile mean {top.mean_net_bp:.1f}bp / CI lower {top.ci_date_clustered[0]:.1f}bp below "
            f"{AUCTION_KILL_MIN_TOP_DECILE_BP}bp AND spearman {spearman:.2f} below {AUCTION_MONOTONE_MIN_SPEARMAN} "
            "— non-monotone decile pattern, indistinguishable from noise"
        )
    elif not top_decile_has_enough_sample:
        verdict, reason = "DIRECTIONAL_INSUFFICIENT", (
            f"top decile has only {top.n_trades} trades / {top.n_dates} distinct dates — below the "
            f"pre-registered minimums ({AUCTION_MIN_DECILE_TRADES} trades / {AUCTION_MIN_DISTINCT_DATES} dates) "
            "PASS itself requires, so neither a PASS nor a KILL verdict would be trustworthy off this sample "
            "(a definitive KILL off a top decile this thin would itself just be noise, not a real result — see "
            "this module's own honest-power caveat: 15 tickers is underpowered for a cross-sectional decile study)"
        )
    else:
        verdict, reason = "DIRECTIONAL_INSUFFICIENT", (
            f"top decile mean {top.mean_net_bp:.1f}bp, spearman {spearman:.2f}, permutation p={perm_p:.4f} — "
            "does not clear the pre-registered PASS bar but is not a clean KILL either (see this module's "
            "own honest-power caveat: 15 tickers is underpowered for a cross-sectional decile study)"
        )

    return DecileReport(
        deciles=deciles, spearman=spearman, permutation_p=perm_p,
        excluded_gap_count=len(excluded), excluded_gap_mean_net_bp=excluded_gap_mean,
        gap_fraction=gap_fraction, data_quality_ok=True, top_decile_pass=top_pass,
        verdict=verdict, verdict_reason=reason,
    )
