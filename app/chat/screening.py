"""Cheap, code-only momentum screening — no LLM involved in the ranking
itself, same "code, not model judgment" principle as the execution ledger
and the live alert thresholds.

`universe` defaults to the trade allowlist for now (see app.chat.router). Open-
market scanning later means passing a bigger universe in here plus a
volume/liquidity pre-filter ahead of it — this function's shape doesn't
need to change, only what's passed to it.

Ranking is volatility-adjusted (raw move / 14-day ATR%%), not raw percent
change — see `risk_adjusted_score`'s docstring for why a single-criterion
raw-momentum rank is a structural trend-chasing risk (screening-method-
improvement-ideas, priority item #1). The ranked pool is also pruned of
over-correlated near-duplicates before the final top_n cut — see
`dedupe_correlated`'s docstring for priority item #4.

`get_price_series`/`get_atr_pct` both come from `app.dataflows.kis_provider`
(KIS-first, yfinance fallback), not straight from the yfinance package
(2026-08-21 fix, found during the Qlib IC-measurement PoC — see
qlib-poc-decision-2026-08-21 project memory). Before this, the momentum
numerator (`_momentum`'s `raw_pct`, via `get_price_series`) went straight to
yfinance while the denominator (`atr_pct`) already went through kis_provider
— so a KRX ticker's score mixed a yfinance-sourced move with a KIS-sourced
volatility, half-defeating the whole reason kis_provider exists (yfinance's
KRX closes were observed off by ~2%%, see that module's own docstring)."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import date, timedelta

from app.clock import get_clock
from app.config import settings
from app.dataflows.kis_provider import get_atr_pct, get_price_series
from app.execution.portfolio_risk import compute_correlation_clusters, ticker_market

logger = logging.getLogger(__name__)


def pct_change(series: dict[date, float]) -> float | None:
    """Pure — first-to-last percent change of a date-keyed price series.
    Split out from `_momentum` below so the actual ranking math is
    unit-testable without a network call, same pattern as
    app.backtest.metrics's pure compute_* functions."""
    if len(series) < 2:
        return None
    dates = sorted(series)
    first, last = series[dates[0]], series[dates[-1]]
    return (last - first) / first if first else None


def risk_adjusted_score(raw_pct: float, atr_pct: float | None) -> float:
    """Pure — Sharpe-like momentum score: the raw move divided by the
    ticker's own 14-day ATR%% (volatility), so a big move on a wildly
    volatile ticker doesn't automatically outrank a smaller move on a
    stable one purely because it moved more. Reuses the same ATR data
    already fetched for stop-loss sizing (app.execution.sizing,
    app.orchestrator.pipeline) rather than a new data source. Falls back
    to the raw move itself when ATR is unavailable or non-positive —
    degrades to the old ranking for that one ticker rather than dropping
    it from consideration entirely.

    This guard only works because `get_atr_pct` returns `None` (never
    NaN) for "not enough history yet" (2026-08-21 fix, ATR TA-Lib PoC
    review) — `atr_pct <= 0` is silently always False for a NaN, which
    used to let a NaN ATR through to `raw_pct / atr_pct` below and hand
    `rank()`'s `sorted()` an unorderable NaN score."""
    if atr_pct is None or atr_pct <= 0:
        return raw_pct
    return raw_pct / atr_pct


def rank(scores: dict[str, float | None], top_n: int) -> list[tuple[str, float]]:
    """Pure — highest-scoring tickers first, dropping any with no score.
    Doesn't care what the score means (raw momentum, risk-adjusted, or
    anything else) — callers decide that by what they put in `scores`."""
    ranked = sorted(
        ((ticker, score) for ticker, score in scores.items() if score is not None),
        key=lambda pair: pair[1],
        reverse=True,
    )
    return ranked[:top_n]


def dedupe_correlated(ranked: list[tuple[str, float]], clusters: dict[str, int]) -> list[tuple[str, float]]:
    """Pure — drops every ranked entry after the first (highest-scored,
    since `ranked` is assumed pre-sorted descending) one in each
    correlation cluster (app.execution.portfolio_risk.compute_correlation_
    clusters). Without this, the momentum leaderboard can end up dominated
    by one theme moving together (e.g. several semiconductor names all
    surfacing at once) — screening-method-improvement-ideas item #4. A
    ticker missing from `clusters` (its own price history fetch failed) is
    never dropped — treated as its own singleton, same "don't fail closed
    on a data gap" convention as `risk_adjusted_score`'s ATR fallback."""
    seen_clusters: set[int] = set()
    out = []
    for ticker, score in ranked:
        cluster_id = clusters.get(ticker)
        if cluster_id is not None:
            if cluster_id in seen_clusters:
                continue
            seen_clusters.add(cluster_id)
        out.append((ticker, score))
    return out


async def _ranked_and_deduped(
    scored: dict[str, MomentumScore],
    top_n: int,
    as_of: date | None = None,
    use_volatility_adjustment: bool = True,
    use_correlation_dedup: bool = True,
) -> list[tuple[str, float]]:
    """Ranks `scored` and drops over-correlated near-duplicates before the
    top_n cut. Clusters only a widened pool (top_n x
    settings.screening_correlation_pool_multiplier), not the whole scored
    set — correlation clustering fetches a full price history per ticker
    (compute_correlation_clusters), so clustering all ~30 scan candidates
    every cycle would add real, mostly-wasted cost for names that were
    never going to make top_n anyway. Skips clustering entirely (and its
    network cost) when the pool is already <= top_n — nothing to dedupe.
    Best-effort: a clustering failure degrades to the plain ranked pool
    rather than blocking screening on it.

    `use_volatility_adjustment`/`use_correlation_dedup` gate the ATR-adjusted
    ranking and the correlation dedup independently — added after
    screening-improvement-quality-review's out-of-sample ablation found both
    stages consistently hurt the overseas pool (3/3 independent periods)
    while domestic stayed direction-unstable (each of 3 periods went a
    different way, no consistent signal either way). Both default True
    (today's long-standing production behavior, domestic keeps them on);
    the overseas caller passes both False."""
    score_key = "score" if use_volatility_adjustment else "raw_pct"
    pool_size = min(len(scored), top_n * settings.screening_correlation_pool_multiplier)
    pool = rank({ticker: getattr(s, score_key) for ticker, s in scored.items()}, pool_size)
    if not use_correlation_dedup or len(pool) <= top_n:
        return pool[:top_n]
    try:
        clusters = await compute_correlation_clusters([ticker for ticker, _ in pool], as_of=as_of)
    except Exception:
        logger.warning("screening: correlation clustering failed — skipping dedup this pass", exc_info=True)
        return pool[:top_n]
    return dedupe_correlated(pool, clusters)[:top_n]


@dataclass(frozen=True)
class MomentumScore:
    raw_pct: float
    atr_pct: float | None
    score: float


async def _momentum(ticker: str, lookback_days: int, as_of: date | None = None) -> MomentumScore | None:
    end = as_of or get_clock().today()
    start = end - timedelta(days=lookback_days)
    try:
        series = await get_price_series(ticker, start, end)
    except Exception:
        # Broad on purpose (not just ValueError): get_price_series' yfinance
        # fallback leg is unguarded (app.dataflows.kis_provider) and its own
        # retry-to-thread re-raises whatever the last attempt saw verbatim —
        # network errors, yfinance rate limits, internal KeyError/IndexError,
        # not just the empty-data ValueError. Matches get_atr_pct's own
        # except Exception below and dedupe's clustering failure handling —
        # this module's own convention is "never fail-closed on one
        # ticker's data gap". asyncio.CancelledError is BaseException, not
        # Exception, so a real cancellation still propagates through this.
        logger.warning("screening: price series unavailable for %s — excluded from this pass", ticker, exc_info=True)
        return None
    raw = pct_change(series)
    if raw is None:
        return None
    try:
        atr_pct = await get_atr_pct(ticker, as_of=as_of)
    except Exception:
        logger.warning("screening: ATR unavailable for %s — falling back to raw momentum", ticker, exc_info=True)
        atr_pct = None
    return MomentumScore(raw_pct=raw, atr_pct=atr_pct, score=risk_adjusted_score(raw, atr_pct))


async def _scored_candidates(
    universe: list[str], lookback_days: int, as_of: date | None = None
) -> dict[str, MomentumScore]:
    # return_exceptions=True: one ticker's _momentum call raising something
    # _momentum itself didn't already catch (a logic bug, not a data-source
    # failure) must not take down the whole scan — same "one ticker's
    # problem stays one ticker's problem" convention as the rest of this
    # module. gather with return_exceptions=True also captures a real
    # asyncio.CancelledError as a plain "result" instead of propagating it,
    # so it must be re-raised explicitly below rather than silently treated
    # as just another failed ticker — this scan runs inside
    # app.live.auto_trade_loop's cycle, and swallowing a shutdown-driven
    # cancellation here would delay/skip the loop's own teardown.
    results = await asyncio.gather(
        *(_momentum(t, lookback_days, as_of=as_of) for t in universe), return_exceptions=True
    )
    for result in results:
        if isinstance(result, asyncio.CancelledError):
            raise result
    scored: dict[str, MomentumScore] = {}
    for ticker, result in zip(universe, results):
        if isinstance(result, BaseException):
            logger.warning("screening: scoring failed for %s — excluded from this pass", ticker, exc_info=result)
            continue
        if result is not None:
            scored[ticker] = result
    return scored


@dataclass(frozen=True)
class ScreeningPass:
    """One screening pass's full output — the ranked/deduped/cut picks AND
    the whole scored cross-section behind them, plus which of
    `MomentumScore`'s fields the ranking actually sorted by.

    Exists because `top_candidates` (below) deliberately returns only the
    tickers a caller can act on, which is all any pre-2026-09-08 caller
    needed — but measuring whether this screening produces alpha at all
    (app.live.screening_log) requires the SCORES of the names that did NOT
    get picked too (truncating a cross-section to top_n collapses its score
    variance and shrinks RankIC toward zero — see
    app.backtest.factor_metrics.rank_ic's own docstring). This is a
    strictly wider return type over the SAME computation — no ranking
    behavior changes.
    선택되지 않은 종목의 점수까지 있어야 라이브 RankIC 계산이 가능해서 추가된 반환 타입."""

    scored: dict[str, MomentumScore]
    ranked: list[tuple[str, float]]
    score_key: str  # "score" (ATR-adjusted) or "raw_pct" — whichever field was ranked
    # 2026-09-09 §0단계 계측(screening-cycle-rebalance-lock-analysis): scores
    # for tickers passed via screening_pass's observe_only — NEVER fed into
    # _ranked_and_deduped, so this field cannot change `ranked`/`score_key`
    # no matter what's in it. Exists purely so app.live.screening_log can
    # record what a rebalance_interval_days-locked ticker's score WOULD have
    # been, to answer "would this have made top_n if it hadn't been locked".
    observed: dict[str, MomentumScore] = field(default_factory=dict)


async def screening_pass(
    universe: list[str],
    top_n: int,
    lookback_days: int,
    as_of: date | None = None,
    use_volatility_adjustment: bool = True,
    use_correlation_dedup: bool = True,
    observe_only: list[str] | None = None,
) -> ScreeningPass:
    scored = await _scored_candidates(universe, lookback_days, as_of=as_of)
    ranked = await _ranked_and_deduped(
        scored,
        top_n,
        as_of=as_of,
        use_volatility_adjustment=use_volatility_adjustment,
        use_correlation_dedup=use_correlation_dedup,
    )
    # MUST match _ranked_and_deduped's own score_key — update together.
    score_key = "score" if use_volatility_adjustment else "raw_pct"
    # observe_only tickers are scored but deliberately kept out of `scored`
    # itself (and thus out of _ranked_and_deduped above) — dedupe against
    # universe first so a caller's mistake (passing an already-scored
    # ticker) can't trigger a redundant _momentum call.
    extra = [t for t in (observe_only or []) if t not in scored]
    observed = await _scored_candidates(extra, lookback_days, as_of=as_of) if extra else {}
    return ScreeningPass(scored=scored, ranked=ranked, score_key=score_key, observed=observed)


async def top_candidates(
    universe: list[str],
    top_n: int = 5,
    lookback_days: int = 10,
    as_of: date | None = None,
    use_volatility_adjustment: bool = True,
    use_correlation_dedup: bool = True,
) -> list[str]:
    """The same momentum ranking `screen()` renders as prose, but returning
    tickers a caller can actually act on (e.g. app.live.auto_trade_loop
    feeding candidates into run_pipeline) instead of a display string.
    `as_of` defaults to today (live use); a historical date makes this
    point-in-time, for app.backtest.screening_sweep's parameter-validation
    harness — nothing in live callers changes since they never pass it.
    `use_volatility_adjustment`/`use_correlation_dedup` — see
    `_ranked_and_deduped`'s docstring; app.live.auto_trade_loop passes both
    False for the overseas pool only."""
    result = await screening_pass(
        universe,
        top_n,
        lookback_days,
        as_of=as_of,
        use_volatility_adjustment=use_volatility_adjustment,
        use_correlation_dedup=use_correlation_dedup,
    )
    return [ticker for ticker, _ in result.ranked]


async def _screen_one_market(
    universe: list[str], top_n: int, lookback_days: int,
    use_volatility_adjustment: bool, use_correlation_dedup: bool,
) -> tuple[list[tuple[str, float]], dict[str, MomentumScore]]:
    scored = await _scored_candidates(universe, lookback_days)
    ranked = await _ranked_and_deduped(
        scored, top_n,
        use_volatility_adjustment=use_volatility_adjustment,
        use_correlation_dedup=use_correlation_dedup,
    )
    return ranked, scored


def _format_market_section(
    market_label: str, ranked: list[tuple[str, float]], scored: dict[str, MomentumScore],
    lookback_days: int, use_volatility_adjustment: bool,
) -> str:
    adj_note = "volatility-adjusted" if use_volatility_adjustment else "raw momentum, not volatility-adjusted"
    lines = [f"{market_label} ({lookback_days}-day price change, {adj_note}):"]
    for ticker, score in ranked:
        s = scored[ticker]
        atr_note = f"{s.atr_pct:.1%} ATR" if s.atr_pct is not None else "ATR unavailable"
        score_label = "risk-adjusted score" if use_volatility_adjustment else "momentum score"
        lines.append(f"  {ticker}: {s.raw_pct:+.2%} over the last {lookback_days} trading days ({atr_note}, {score_label} {score:+.2f})")
    return "\n".join(lines)


async def screen(universe: list[str], top_n: int = 5, lookback_days: int | None = None) -> str:
    """Momentum-ranked candidates from `universe`, rendered as prose for the
    chat "what's moving?" answer — display only, never touches an order.

    Domestic and overseas are scored, deduped, and ranked as two entirely
    SEPARATE pools (never merged into one combined ranking) using the same
    per-market lookback/volatility-adjustment/correlation-dedup settings
    `app.live.auto_trade_loop` uses live (app.config.settings), so this
    answer reflects the same ranking the live engine actually acts on —
    before this, screen() hardcoded lookback_days=10 and always ran
    volatility-adjustment+dedup for both markets, which silently diverged
    from live once the two markets' settings diverged (overseas runs both
    OFF — screening-improvement-quality-review's out-of-sample ablation
    found both stages hurt the overseas pool in 3/3 periods).

    `lookback_days=None` (default) reads each market's own
    settings.momentum_lookback_days_domestic/_overseas; passing an explicit
    int uses that value for both markets (kept for existing callers/tests
    that don't care about the per-market distinction).

    Note: top_n is a display cutoff (always 5 per market here), not
    app.config.settings.auto_trade_max_tickers_domestic/_overseas — that
    setting caps how many candidates the live cycle spends LLM budget
    analyzing, an execution cost concept with no meaning for a free chat
    answer. Since _ranked_and_deduped's correlation-clustering pool size is
    top_n-dependent, a top_n that differs from the live cycle's can make
    this answer's dedup result differ from live's by a name or two even
    when every other setting matches — not a bug, just a consequence of
    top_n being purely a "where to cut the display" choice here."""
    domestic = [t for t in universe if ticker_market(t) == "domestic"]
    overseas = [t for t in universe if ticker_market(t) == "overseas"]

    domestic_lookback = lookback_days if lookback_days is not None else settings.momentum_lookback_days_domestic
    overseas_lookback = lookback_days if lookback_days is not None else settings.momentum_lookback_days_overseas
    domestic_vol_adj = settings.screening_volatility_adjustment_domestic
    overseas_vol_adj = settings.screening_volatility_adjustment_overseas
    domestic_dedup = settings.screening_correlation_dedup_domestic
    overseas_dedup = settings.screening_correlation_dedup_overseas

    (domestic_ranked, domestic_scored), (overseas_ranked, overseas_scored) = await asyncio.gather(
        _screen_one_market(domestic, top_n, domestic_lookback, domestic_vol_adj, domestic_dedup)
        if domestic else _empty_market(),
        _screen_one_market(overseas, top_n, overseas_lookback, overseas_vol_adj, overseas_dedup)
        if overseas else _empty_market(),
    )

    sections = []
    if domestic_ranked:
        sections.append(_format_market_section(
            "Domestic (KRX)", domestic_ranked, domestic_scored, domestic_lookback, domestic_vol_adj
        ))
    if overseas_ranked:
        sections.append(_format_market_section(
            "Overseas", overseas_ranked, overseas_scored, overseas_lookback, overseas_vol_adj
        ))
    if not sections:
        return "No momentum data available for the current trade allowlist."
    return "Momentum-ranked candidates (trade allowlist only):\n" + "\n".join(sections)


async def _empty_market() -> tuple[list[tuple[str, float]], dict[str, MomentumScore]]:
    return [], {}
