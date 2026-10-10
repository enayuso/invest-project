"""Pure factor-evaluation statistics for the offline screening measurement
tooling (app.backtest.screening_sweep, app.audit.signal_ic).

Same "code, not model judgment" boundary as app.backtest.metrics: every
function here is a pure transform over numbers already produced elsewhere
— no network, no DB, no clock, no pandas, no scipy, no app.execution /
app.backtest.harness import. `None` (never NaN, never a misleading 0.0)
is the "cannot measure" signal throughout, kept deliberately distinct
from a real 0.0 result — an ICIR denominator built on 0.0-for-undefined
is corrupted, so `spearman` / `rank_ic` return `None` for a degenerate
cross-section rather than folding it into an average as if it were a
measured zero.

The rank / spearman / percentile kernels here are the canonical copies:
app.audit.signal_ic and app.backtest.auction_imbalance_report each grew
their own identical (or near-identical) local copy before this module
existed — signal_ic now delegates to these, auction_imbalance_report
keeps its 0.0-convention `_spearman` since that report's monotonicity
test wants a number there, not a None.
"""

from __future__ import annotations

import math
import random
import statistics

INTERPRETATION_GUIDE = """\
INTERPRETATION GUIDE (read before drawing any conclusion)

RankIC (cross-sectional Spearman of factor score vs forward return, per date):
  |IC| < 0.02   noise — indistinguishable from no relationship on this universe size
  0.02 - 0.05   the band a genuine, tradeable equity factor lives in
  0.05 - 0.10   strong — rare and worth a hard second look for leakage
  |IC| > 0.10   suspect — almost always look-ahead / point-in-time contamination
  The SIGN is itself a result: a persistently negative RankIC means the
  factor is a reversal signal, not a broken momentum signal.

ICIR (mean RankIC / stdev of RankIC across dates — "is the edge consistent
or just lucky on average"):
  < 0.3         not consistent — the mean IC, even if positive, is swamped by date-to-date swing
  0.3 - 0.5     usable consistency
  > 0.5         high consistency (again: check for leakage before celebrating)
  NOT annualized — cadence varies per run; only compare same-cadence variants.

HARD VERDICT RULES (any one true => do not promote the variant):
  1. excess-return 95% CI crosses 0            => unproven (the pick set did not
                                                  reliably beat its own universe)
  2. n_eff < 20                                => underpowered — the point estimate
                                                  is not backed by enough independent dates
  3. mean turnover * universe round-trip cost
     >= mean excess return                     => the edge is eaten by trading cost
  4. RankIC sign disagrees with excess-return
     sign                                      => internally inconsistent; the ranking
                                                  and the pick performance tell opposite
                                                  stories — treat the whole result as noise
"""


def rank_avg(values: list[float]) -> list[float]:
    """Average-rank (ties get the mean of their positions). The canonical
    copy — app.audit.signal_ic._rank and
    app.backtest.auction_imbalance_report._rank are the same algorithm.
    Pure Python rather than scipy.stats.rankdata (scipy is not in this
    project's venv — see project memory scipy-dependency-gap-2026-08-25)."""
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


def spearman(xs: list[float], ys: list[float], min_pairs: int = 3) -> float | None:
    """Pearson correlation on ranks. `None` if fewer than `min_pairs`
    observations, or if either side has zero variance ("cannot measure"
    must be distinct from "no correlation"). Uses the `None` convention
    from signal_ic._spearman, NOT the 0.0 convention from
    auction_imbalance_report._spearman (which is a different purpose)."""
    if len(xs) != len(ys) or len(xs) < min_pairs:
        return None
    rx, ry = rank_avg(list(xs)), rank_avg(list(ys))
    mean_rx = sum(rx) / len(rx)
    mean_ry = sum(ry) / len(ry)
    cov = sum((a - mean_rx) * (b - mean_ry) for a, b in zip(rx, ry))
    var_x = sum((a - mean_rx) ** 2 for a in rx)
    var_y = sum((b - mean_ry) ** 2 for b in ry)
    if var_x <= 0 or var_y <= 0:
        return None
    return cov / (var_x * var_y) ** 0.5


def percentile(sorted_values: list[float], pct: float) -> float:
    """Nearest-rank percentile of an ALREADY-sorted-ascending list. Same
    as scalping_viability._percentile / auction_imbalance_report._percentile.
    `pct` in [0, 100]. Empty input -> 0.0 (matches those two)."""
    if not sorted_values:
        return 0.0
    idx = min(len(sorted_values) - 1, max(0, round(pct / 100 * (len(sorted_values) - 1))))
    return sorted_values[idx]


def _pairs(
    scores: dict[str, float | None], forward_returns: dict[str, float | None]
) -> tuple[list[float], list[float]]:
    """Pairwise-drop any ticker whose score OR forward return is None (or
    missing on the other side). Returns (scores_list, forward_list) aligned."""
    xs: list[float] = []
    ys: list[float] = []
    for ticker, score in scores.items():
        forward = forward_returns.get(ticker)
        if score is None or forward is None:
            continue
        xs.append(score)
        ys.append(forward)
    return xs, ys


def rank_ic(
    scores: dict[str, float | None],
    forward_returns: dict[str, float | None],
    min_cross_section: int = 5,
) -> float | None:
    """Spearman between factor score at date t and the forward return, over
    the WHOLE intersecting universe (NOT truncated to top_n — truncating
    collapses score variance and shrinks IC toward 0). Pairwise-drops any
    ticker whose score or forward return is None. Fewer than
    `min_cross_section` usable pairs -> None. The sign of the result IS a
    result: a negative RankIC means reversal, not a broken signal."""
    xs, ys = _pairs(scores, forward_returns)
    if len(xs) < min_cross_section:
        return None
    return spearman(xs, ys, min_pairs=min_cross_section)


def icir(ic_series: list[float | None]) -> float | None:
    """mean(IC) / stdev(IC) over the non-None entries. Fewer than 3 valid
    entries -> None. stdev == 0 -> None (a "perfectly consistent" edge on
    2-3 dates is a data-starved artefact, not a real one). NOT annualized:
    cadence varies per run, so render() prints cadence / holding in the
    header and only compares same-cadence variants."""
    valid = [x for x in ic_series if x is not None]
    if len(valid) < 3:
        return None
    sd = statistics.stdev(valid)
    if sd == 0:
        return None
    return statistics.mean(valid) / sd


def excess_return(pick_returns: list[float], universe_returns: list[float]) -> float | None:
    """mean(top_n forward return) - mean(that date's whole eligible universe
    forward return). The universe itself is the benchmark, so market
    direction and universe-composition effects cancel on both sides.
    Either side empty -> None."""
    if not pick_returns or not universe_returns:
        return None
    return statistics.mean(pick_returns) - statistics.mean(universe_returns)


def quantile_spread(
    scores: dict[str, float | None],
    forward_returns: dict[str, float | None],
    fraction: float = 0.2,
) -> float | None:
    """mean(top fraction-quantile returns) - mean(bottom fraction-quantile
    returns), quantiles cut by score. Each quantile is
    max(1, round(n * fraction)) names. Fewer than 5 usable pairs -> None."""
    xs, ys = _pairs(scores, forward_returns)
    n = len(xs)
    if n < 5:
        return None
    order = sorted(range(n), key=lambda i: xs[i])
    k = max(1, round(n * fraction))
    bottom = [ys[i] for i in order[:k]]
    top = [ys[i] for i in order[-k:]]
    return statistics.mean(top) - statistics.mean(bottom)


def effective_sample_size(n_dates: int, cadence_days: int, holding_days: int) -> float | None:
    """  overlap = max(1, ceil(holding_days / cadence_days))
         n_eff   = n_dates / overlap

    Counts DATES, not picks (same-day picks share that day's market and
    sector shocks — they are not independent observations), then deflates
    by the holding-vs-cadence overlap (the alphalens overlapping-data
    problem: consecutive sampled dates whose holding windows overlap carry
    correlated returns). A conservative heuristic, NOT a Newey-West HAC
    estimator — it does not look at the returns themselves, only the
    schedule. `n_dates <= 0` -> None; `cadence_days <= 0` -> None (no
    ZeroDivision)."""
    if n_dates <= 0 or cadence_days <= 0:
        return None
    overlap = max(1, math.ceil(holding_days / cadence_days))
    return n_dates / overlap


def block_bootstrap_ci(
    blocks: list[list[float]],
    n_bootstrap: int = 10_000,
    confidence: float = 0.95,
    seed: int | None = None,
) -> tuple[float, float] | None:
    """Resample whole blocks with replacement, pool every value in the
    resampled blocks, take the mean, repeat. Same algorithm as
    scalping_viability.bootstrap_ci_clustered but generic (list[list[float]]
    rather than domain types). Empty -> None. Fewer than 3 blocks -> None
    (a bootstrap over 1-2 blocks is meaningless). The sweep always passes
    an explicit `seed` (default 0) for reproducibility. Output is
    (percentile 2.5, percentile 97.5) for confidence=0.95."""
    usable = [b for b in blocks if b]
    if len(usable) < 3:
        return None
    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(n_bootstrap):
        resampled = [rng.choice(usable) for _ in usable]
        pooled = [v for block in resampled for v in block]
        means.append(statistics.mean(pooled) if pooled else 0.0)
    means.sort()
    lo_pct = (1 - confidence) / 2 * 100
    hi_pct = (1 - (1 - confidence) / 2) * 100
    return (percentile(means, lo_pct), percentile(means, hi_pct))


def turnover(prev_picks: list[str], picks: list[str]) -> float | None:
    """|picks - prev_picks| / len(picks) — the fraction of today's pick set
    that is new since the previous sampled date. Order-independent (set
    difference). Empty `picks` -> None. The caller skips the first date
    (no prev)."""
    if not picks:
        return None
    return len(set(picks) - set(prev_picks)) / len(picks)


def worth_analyzing_rate(returns: list[float], bar: float = 0.03) -> float | None:
    """Fraction of picks whose |forward return| >= `bar`. Screening's real
    job is "surface tickers worth ~17 LLM calls", not "maximize return" —
    dodging a big drop is also a win, and a pick that goes +/-1% wasted the
    pipeline. `bar` = 0.03: roughly 8x a KRX round-trip cost (~0.35%, see
    CHANGELOG #122) and below the typical 3-10% ATR stop band. Empty -> None."""
    if not returns:
        return None
    return sum(1 for r in returns if abs(r) >= bar) / len(returns)


def upside_hit_rate(returns: list[float], bar: float = 0.03) -> float | None:
    """Fraction of picks whose forward return >= `bar` — a long-only reality
    check, secondary to `worth_analyzing_rate`. Empty -> None."""
    if not returns:
        return None
    return sum(1 for r in returns if r >= bar) / len(returns)
