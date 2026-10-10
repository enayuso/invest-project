"""Measures whether app.chat.screening's momentum score actually predicts
forward returns — a question nothing in this codebase answered before
(Qlib PoC, 2026-08-21: see qlib-poc-decision-2026-08-21 project memory for
why this is ~60 lines of plain pandas instead of a `qlib` dependency —
Qlib's own `calc_ic` is itself just a cross-sectional correlation per date,
mlflow forces a `pandas<3` downgrade, and its required `gym` dependency has
been unmaintained since 2022).

`IC` = same-date Pearson correlation between a ticker's score and its
forward return; `RankIC` = the Spearman (rank) version, the primary metric
here since the scan universe is only ~30-60 names — a handful of outlier
scores/returns can swing a Pearson correlation on that small a cross-
section, while rank correlation is robust to it. `ICIR` = mean RankIC /
its std-dev across dates, Qlib's "is the edge consistent or just lucky on
average" summary stat.

Not to be confused with Qlib's own "information_ratio", which (despite the
name) is annualized Sharpe of a return series, not a rank-correlation
statistic — a different thing measured by app.backtest.metrics already,
deliberately left unannualized there (see that module's own docstring on
why cadence-dependent annualization is misleading here).

LLM-free and execution-free by construction (same "code, not model
judgment" boundary as app.chat.screening and app.backtest.screening_sweep
themselves) — this only reads historical scores/prices and computes
correlations, never trades or feeds a live decision.

Not wired into the `python -m app.audit` CLI yet — that's a deliberate
scope decision, not an oversight — so for now this is exercised only via
its test suite (tests/audit/test_signal_ic.py), as a library module."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import date

import pandas as pd

from app.backtest.factor_metrics import rank_avg, spearman
from app.backtest.screening_sweep import _forward_return
from app.chat.screening import _scored_candidates

# Fewer names than this in a single date's cross-section and a correlation
# is noise, not signal — this scan universe (~30-60 tickers, often fewer
# after data gaps) is already thin; below this floor we skip the date
# rather than report a number nobody should act on.
_MIN_CROSS_SECTION = 5


def _rank(values: list[float]) -> list[float]:
    """Average-rank (ties share the mean of their tied ranks). Kernel
    consolidated into app.backtest.factor_metrics.rank_avg; the
    app.audit -> app.backtest coupling a local copy used to avoid already
    exists at the `_forward_return` import above, so there is nothing left
    to protect by duplicating the ~10 lines."""
    return rank_avg(values)


def _spearman(xs: list[float], ys: list[float]) -> float | None:
    """Spearman rank correlation — None for a degenerate (zero-variance)
    input rather than 0.0 or NaN. Delegates to
    app.backtest.factor_metrics.spearman (the `None`-convention copy, not
    auction_imbalance_report's 0.0-convention one). `min_pairs=1` keeps the
    prior behaviour: `_daily_ic` only calls this once the cross-section
    already cleared `_MIN_CROSS_SECTION`, so the thin-input guard lives
    there, not here."""
    return spearman(xs, ys, min_pairs=1)


@dataclass(frozen=True)
class DailyIC:
    trade_date: date
    n: int
    ic: float | None
    rank_ic: float | None


@dataclass(frozen=True)
class ICSummary:
    daily: list[DailyIC]
    n_dates_scored: int
    mean_ic: float | None
    mean_rank_ic: float | None
    rank_icir: float | None


async def _daily_ic(
    universe: list[str],
    trade_date: date,
    lookback_days: int,
    holding_days: int,
    *,
    use_volatility_adjustment: bool = True,
) -> DailyIC:
    """One date's cross-sectional IC/RankIC between `universe`'s screening
    score (as of `trade_date`) and each ticker's forward return over the
    next `holding_days`. `use_volatility_adjustment` picks the same score
    app.chat.screening._ranked_and_deduped would rank by (`score` = ATR-
    adjusted, or `raw_pct` = plain momentum) — pass the market's own
    settings.screening_volatility_adjustment_{domestic,overseas} value to
    reproduce what that market's live screening actually ranks by."""
    scored = await _scored_candidates(universe, lookback_days, as_of=trade_date)
    if not scored:
        return DailyIC(trade_date, 0, None, None)

    tickers = list(scored)
    forward_returns = await asyncio.gather(
        *(_forward_return(ticker, trade_date, holding_days) for ticker in tickers)
    )
    score_attr = "score" if use_volatility_adjustment else "raw_pct"
    rows = [
        (getattr(scored[ticker], score_attr), forward)
        for ticker, forward in zip(tickers, forward_returns)
        if forward is not None
    ]
    if len(rows) < _MIN_CROSS_SECTION:
        return DailyIC(trade_date, len(rows), None, None)

    frame = pd.DataFrame(rows, columns=["score", "forward_return"])
    return DailyIC(
        trade_date=trade_date,
        n=len(rows),
        ic=float(frame["score"].corr(frame["forward_return"], method="pearson")),
        rank_ic=_spearman(frame["score"].tolist(), frame["forward_return"].tolist()),
    )


def summarize(daily: list[DailyIC]) -> ICSummary:
    """Pure — aggregates per-date IC/RankIC into the headline mean/ICIR
    stats. Dates below `_MIN_CROSS_SECTION` (rank_ic is None) are excluded
    from the average rather than counted as zero — a thin-cross-section
    date says nothing about the score's predictive power, in either
    direction."""
    scored = [d for d in daily if d.rank_ic is not None]
    if not scored:
        return ICSummary(daily=daily, n_dates_scored=0, mean_ic=None, mean_rank_ic=None, rank_icir=None)

    ic_series = pd.Series([d.ic for d in scored])
    rank_ic_series = pd.Series([d.rank_ic for d in scored])
    std = rank_ic_series.std()
    return ICSummary(
        daily=daily,
        n_dates_scored=len(scored),
        mean_ic=float(ic_series.mean()),
        mean_rank_ic=float(rank_ic_series.mean()),
        rank_icir=float(rank_ic_series.mean() / std) if std and std > 0 else None,
    )


async def measure_signal_ic(
    universe: list[str],
    trade_dates: list[date],
    lookback_days: int,
    holding_days: int,
    *,
    use_volatility_adjustment: bool = True,
) -> ICSummary:
    """The entry point: IC/RankIC/ICIR of the screening score across every
    date in `trade_dates`, run concurrently (same "gather across
    independent dates" pattern app.backtest.screening_sweep.sweep already
    uses — each date is a fully independent scan+forward-return
    computation, nothing shared to serialize on)."""
    daily = await asyncio.gather(
        *(
            _daily_ic(universe, trade_date, lookback_days, holding_days, use_volatility_adjustment=use_volatility_adjustment)
            for trade_date in trade_dates
        )
    )
    return summarize(list(daily))
