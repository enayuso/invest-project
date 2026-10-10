"""Offline, LLM-free measurement of app.chat.screening's candidate-selection
quality — screening-method-improvement-ideas priority item #5 ("모멘텀
파라미터가 백테스트로 검증된 값이 아님"), now generalised from a bare
lookback grid to arbitrary screening VARIANTS.

Run it as::

    python -u -m app.backtest.screening_sweep --start 2026-01-02 --end 2026-06-30

`python -u` (unbuffered) matters: a past run appeared to hang for ~an hour
because the per-date progress lines were sitting in a stdout buffer while
the network phase churned. Every progress line below also passes
`flush=True` as a belt-and-braces second guard.

Deliberately separate from app.backtest.harness's daily replay: that loop
runs the full 4-analyst + 2-debate + 3-decision-node pipeline against a
live LLM per date, unusable for sweeping a grid across many dates. This
tool never touches the LLM or app.execution.ledger — it reproduces
app.chat.screening's ranking (`_scored_candidates` -> `rank` ->
`dedupe_correlated`, assembled here directly rather than via
`top_candidates`, so the intermediate scored set is reusable) at each
historical date and scores the resulting picks by pure forward return
plus the factor statistics in app.backtest.factor_metrics (RankIC, ICIR,
excess-return vs the pick set's own universe, quantile spread, turnover,
"worth analyzing" rate, block-bootstrap CIs, effective sample size).

It tells you whether a variant WOULD HAVE surfaced better tickers — it
does NOT replace the full pipeline backtest, since a real trade also
depends on the AI's rating, sizing, and stop-loss, none of which this
touches.

TOOLING ONLY: nothing here is imported by app.chat.screening or
app.live.auto_trade_loop, and it reads no `settings.screening_*` value
except `screening_correlation_pool_multiplier` (replicated below, kept in
lockstep with the live formula by a test).

2026-09-02 (screening-improvement-research #3, universe-axis
diversification) added ONE additional live-read path to this otherwise
network-free-except-Phase-A tool: `--universe-axes` resolves the sweep's
universe by calling into app.live.universe's KRX discovery functions
against LIVE KIS ranking state (`_universe_from_live_axes`), instead of the
default `--tickers`/trade_allowlist path. This does not weaken the
tooling-only invariant above — it is still nothing app.live.auto_trade_loop
imports — but it does mean the universe itself, not just the subsequent
Phase A candle fetch, is now a live snapshot with no historical/point-in-
time equivalent. Two runs of `--universe-axes` on different days are NOT
comparable to each other (KIS's ranking TRs answer "right now", not "as of
the date I asked about"); only the arms WITHIN one comparison run are
comparable, since a fair A/B needs both arms resolved from the same day's
snapshot. `--dump-universe` exists specifically to make that resolved
ticker list inspectable/diffable across the two arms of one run.
`--universe-max-price` (added the same day, screening-improvement-research
#3 follow-up, the A/B lever for `app.config.
auto_trade_universe_price_ceiling_domestic`) is an even more "right now"-
shaped live-read than the axes themselves: it asks KIS to rank only WITHIN
a price-bounded set, so the same two-runs-not-comparable caveat applies
with even more force to it.

Two-phase execution (Phase A = all network, Phase B = pure + clustering)
structurally prevents the past bug where the same (ticker, date) candle
fetch happened once per variant instead of once total.

Within a date, Phase A opens one `candle_run_cache()` scope so the distinct
lookbacks' `get_price_series` + `get_atr_pct` calls collapse onto one wide
KIS fetch per (ticker, end) — N*(2D+1) requests per date become N*2.
Lookbacks > `_MAX_KIS_CANDLE_RANGE_DAYS` (100) still route to yfinance
untouched (the range gate in `get_price_series` precedes the KIS path), so
a 120-day variant is not silently truncated.

2026-09-02 (screening-improvement-research #4, residual momentum) adds a
second `Variant` axis, `momentum_mode` ("total", today's live formula, or
"residual", app.domain.residual_momentum's market-model residual — see
that module's docstring for what it measures and why). This is a
MEASUREMENT-ONLY addition, same invariant as `--universe-axes` above:
nothing in app.chat.screening or app.live.auto_trade_loop reads
`momentum_mode` or imports app.domain.residual_momentum. A residual
variant needs one extra network leg Phase A didn't previously fetch — the
market benchmark's own return over the same window as each ticker's — so
`sweep()` resolves which market(s) the universe touches and fetches ONE
benchmark series per (market, date) inside the same `candle_run_cache()`
scope Phase A already opens (`_benchmark_series` / `_benchmark_pct_by_
lookback`, collapsing per-lookback slicing the same way `candle_run_cache`
already collapses the ticker legs — see those functions' own docstrings).
A residual variant whose `lookback_days` exceeds `_RESIDUAL_MAX_LOOKBACK_
DAYS` is rejected by `sweep()` before any network call: past that cap the
ticker numerator (via `get_price_series`'s own range gate) would silently
fall back to yfinance while the KOSPI benchmark leg (`fetch_index_daily_
chart`, a dedicated KIS TR with no such cap) would not — reintroducing
the exact KIS-vs-yfinance source-mixing the 2026-08-21 fix (this module's
own earlier comment, and app.chat.screening's module docstring) removed
from the live ranking numerator/denominator pair.

`--dump-json PATH` (2026-09-30, T1) writes ONE machine-readable dump of
everything the run already computed (all scores, forward returns with their
actual exit price date, per-variant observations + signed scores) for
offline re-analysis; it adds no fetches and changes no output when omitted.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal

from app.backtest import factor_metrics
from app.backtest.harness import select_trading_dates
from app.backtest.metrics import compute_buy_and_hold_return
from app.chat.screening import (
    MomentumScore,
    _scored_candidates,
    dedupe_correlated,
    pct_change,
    rank,
    risk_adjusted_score,
)
from app.config import settings
from app.domain.residual_momentum import residual_pct

# The ranking numerator was moved off yfinance and onto KIS in the
# 2026-08-21 fix (see app.chat.screening's module docstring: a KRX ticker's
# score used to mix a yfinance-sourced move with a KIS-sourced ATR). The
# measurement layer had the same source-mixing — `_forward_return` scored
# picks on yfinance KRX closes, ~2% off, which is roughly the effect size
# this whole tool is trying to measure — so it moves here too.
from app.dataflows.kis_provider import candle_run_cache, get_price_series
from app.execution.costs import transaction_cost_bps_for
from app.execution.portfolio_risk import compute_correlation_clusters, ticker_market

# Mirrors app.dataflows.kis_provider._MAX_KIS_CANDLE_RANGE_DAYS (100) — kept
# as its own named constant (not imported) the same way `_pool_size` below
# duplicates app.chat.screening's pool-size formula rather than importing
# it: a test (test_residual_max_lookback_matches_kis_provider_constant)
# pins the two values equal so they can never silently drift apart. A
# residual variant's ticker-move numerator (get_price_series, KIS-first,
# yfinance past this cap) and its KOSPI benchmark denominator
# (fetch_index_daily_chart, always KIS, no cap of its own) must never split
# across sources for the same date — see the module docstring's 2026-09-02
# addition and the 2026-08-21 source-mixing incident it references.
_RESIDUAL_MAX_LOOKBACK_DAYS = 100
_KOSPI_INDEX_CODE = "0001"  # same KIS index code app.execution.market_regime.refresh_domestic_regime uses

# A normal weekend + one holiday is at most ~4 calendar days; anything
# wider than that between the requested end date and the latest available
# price means the feed genuinely doesn't cover the target window (recent
# listing, delisted, or the target date hasn't happened yet relative to
# when this sweep is run) — not an ordinary non-trading-day gap.
_FORWARD_RETURN_DATA_GAP_TOLERANCE_DAYS = 4

# The trading-day calendar fetch in `_main` intentionally STAYS on yfinance:
# it needs a ~6-month range (wider than KIS's ~100-row candle cap, so KIS
# would silently truncate it) and only needs the list of dates, not prices.
from app.dataflows.yfinance import get_price_series as _calendar_price_series


@dataclass
class Variant:
    """One screening configuration to measure. `sign=-1` flips the ranking
    (a reversal hypothesis: pick the bottom of the momentum ranking).

    `momentum_mode` (2026-09-02, screening-improvement-research #4):
    "total" (default) reproduces today's live `app.chat.screening` ranking
    numerator (`MomentumScore.raw_pct`) unchanged; "residual" replaces it
    with `app.domain.residual_momentum.residual_pct(raw_pct, benchmark_pct)`
    — the ticker's move minus the market's own move over the same window —
    before the existing volatility-adjustment/dedup/sign machinery runs
    unchanged on top of it. Appended LAST (after `sign`) so every existing
    positional or keyword `Variant(...)` construction, in this file's own
    CLI parsing and in tests, is unaffected by its addition."""

    label: str
    lookback_days: int
    use_volatility_adjustment: bool = True
    use_correlation_dedup: bool = True
    sign: int = 1
    momentum_mode: Literal["total", "residual"] = "total"


@dataclass
class DateObservation:
    trade_date: date
    picks: list[str]
    pick_forward_returns: list[float]  # non-None forward returns of the picks
    universe_forward_returns: list[float]  # non-None forward returns of the whole scored universe
    excess_return: float | None
    rank_ic: float | None
    quantile_spread: float | None
    turnover: float | None
    # 2026-09-02 additions (screening-improvement-research #4) — both are
    # PER-TICKER-DATE observability, not per-variant: `residual_fallbacks`
    # counts, for a "residual" variant on this date, how many of that
    # date's scored tickers had no resolvable benchmark_pct (see
    # app.domain.residual_momentum.residual_pct's "never fail-closed on one
    # data gap" fallback) — always 0 for a "total" variant, since it never
    # looks a benchmark up at all. `dedup_drop_rate` is the fraction of the
    # pre-dedup top_n leaderboard that `dedupe_correlated` had to replace
    # with a lower-ranked pick — `None` when `use_correlation_dedup` is
    # False for this variant (not applicable, not "zero drops").
    residual_fallbacks: int = 0
    dedup_drop_rate: float | None = None
    # 2026-09-11: True when use_correlation_dedup was on for this date but
    # compute_correlation_clusters raised (e.g. a yfinance rate-limit burst)
    # — dedupe_correlated({}) drops nothing, so a degraded date silently
    # behaved exactly like use_correlation_dedup=False. dedup_drop_rate
    # stays None on a degraded date (not 0.0 — 0.0 would read as "clustered
    # fine, nothing correlated" and drag mean_dedup_drop_rate toward 0).
    dedup_degraded: bool = False


@dataclass
class VariantResult:
    variant: Variant
    n_dates: int  # dates that produced at least one pick
    observations: list[DateObservation] = field(default_factory=list)
    mean_forward_return: float | None = None  # pooled over every pick, every date (old-output continuity)
    mean_excess_return: float | None = None
    excess_return_ci95: tuple[float, float] | None = None
    mean_rank_ic: float | None = None
    rank_ic_ci95: tuple[float, float] | None = None
    icir: float | None = None
    mean_quantile_spread: float | None = None
    mean_turnover: float | None = None
    worth_rate: float | None = None
    upside_rate: float | None = None
    n_eff: float | None = None
    # 2026-09-02 additions — see DateObservation's own docstring for what
    # each per-date field means; these are the roll-ups `_summarise_variant`
    # computes from them. `residual_fallback_dates` counts DATES (not
    # tickers) where at least one fallback happened, so `render()`'s
    # X/n_dates fraction reads as "how much of this variant's own run was
    # actually measuring residual momentum" rather than a raw ticker count
    # with no denominator. `mean_dedup_drop_rate` is `None` whenever no
    # observation had a non-None `dedup_drop_rate` (e.g. dedup was off for
    # every date, or every pool was already <= top_n).
    residual_fallback_dates: int = 0
    mean_dedup_drop_rate: float | None = None
    # Count of dates where correlation clustering raised while dedup was
    # requested — see DateObservation.dedup_degraded's own docstring.
    dedup_degraded_dates: int = 0


async def _forward_detail(
    ticker: str, trade_date: date, holding_days: int
) -> tuple[float | None, date | None, date | None]:
    """Buy-and-hold return for `ticker` from `trade_date` to `trade_date +
    holding_days` — how well a screening pick that day would have done,
    independent of anything the AI/execution layer would have decided.
    None on any data gap (delisted, too-recent listing, no trading day at
    or after the target end) rather than raising — one bad ticker/date
    combination shouldn't kill the whole sweep.

    2026-09-11 fix: `compute_buy_and_hold_return` -> `price_on_or_before`
    falls back to the closest date STRICTLY BEFORE the target if the target
    itself isn't in `prices` — correct for an ordinary weekend/holiday gap,
    but if `prices`' actual history ends well before `target_end` (a recent
    IPO whose history doesn't reach that far forward yet, a delisting, or
    this sweep simply being run with trade dates close to "today" so the
    target date hasn't happened yet), that fallback silently substitutes a
    much earlier price and returns a shrunken-window "return" with no
    signal that anything was truncated — which then gets pooled into
    mean_forward_return/mean_excess_return as if it were a genuine
    holding_days-ahead observation. The check below rejects any gap wider
    than `_FORWARD_RETURN_DATA_GAP_TOLERANCE_DAYS` between the requested
    end and the latest price actually available, while still allowing a
    plain weekend/holiday gap through unchanged (e.g. target_end lands on a
    Saturday, the latest price is the preceding Friday — a 1-2 day gap,
    well under the tolerance — so `price_on_or_before`'s normal, intended
    fallback keeps working exactly as before for that case).
    목표 종료일까지 실제 가격 데이터가 도달하지 못하면(상장 초기/상장폐지/
    아직 도래하지 않은 날짜) 잘린 구간의 수익률을 그대로 반환하지 않고 None."""
    # Returns (forward_return, entry_price_date, exit_price_date): the last two
    # are the price dates actually used (price_on_or_before), so a caller can
    # count realised trading days held; all None whenever the return is None.
    # 실제 사용된 진입/청산 가격일 — 장기휴장 0일 보유 식별용.
    end = trade_date + timedelta(days=holding_days + 5)  # padding for weekends/holidays
    try:
        prices = await get_price_series(ticker, trade_date, end)
    except ValueError:
        return None, None, None
    if not prices:
        return None, None, None
    target_end = trade_date + timedelta(days=holding_days)
    if max(prices) < target_end - timedelta(days=_FORWARD_RETURN_DATA_GAP_TOLERANCE_DAYS):
        return None, None, None  # data doesn't reach the target window — would silently return a truncated-window "return"
    try:
        ret = compute_buy_and_hold_return(prices, trade_date, target_end)
    except KeyError:
        return None, None, None
    entry_date = max(d for d in prices if d <= trade_date)
    exit_date = max(d for d in prices if d <= target_end)
    return ret, entry_date, exit_date


async def _forward_return(ticker: str, trade_date: date, holding_days: int) -> float | None:
    """Return-only wrapper over `_forward_detail` — same value and same
    2026-09-11 truncation-guard semantics for every existing caller.
    수익률만 필요한 기존 호출자용 래퍼."""
    return (await _forward_detail(ticker, trade_date, holding_days))[0]


def _variant_score(s: MomentumScore, variant: Variant, benchmark_pct: float | None) -> float:
    """Recomputes a variant's UNSIGNED score from `MomentumScore`'s raw
    components — `s.raw_pct`/`s.atr_pct` — instead of reading the
    precomputed `s.score` field, because a "residual" variant needs a
    different numerator than whatever `app.chat.screening._momentum`
    already divided (it only ever computes the total-momentum numerator).
    `variant.sign` is NOT applied here — callers multiply it in afterward,
    same as the pre-2026-09-02 `variant.sign * getattr(s, score_key)` this
    replaces.

    For `momentum_mode="total"` this MUST reproduce today's live formula
    exactly: `risk_adjusted_score(s.raw_pct, s.atr_pct)` when volatility-
    adjustment is on, `s.raw_pct` otherwise — which is definitionally equal
    to `s.score`/`s.raw_pct` respectively, since `s.score` was built by the
    same `risk_adjusted_score` call over the same two inputs
    (`app.chat.screening._momentum`). A test
    (test_total_mode_score_matches_precomputed_momentum_score) pins this so
    a future refactor of this function can't silently move the CONTROL arm
    while only intending to touch the "residual" treatment arm."""
    numerator = s.raw_pct
    if variant.momentum_mode == "residual":
        numerator = residual_pct(s.raw_pct, benchmark_pct)
    if not variant.use_volatility_adjustment:
        return numerator
    return risk_adjusted_score(numerator, s.atr_pct)


async def _benchmark_series(market: str, end: date, max_lookback_days: int) -> dict[date, float] | None:
    """The market benchmark's own date-keyed close series for `market`,
    covering `[end - max_lookback_days, end]` — the raw material
    `_benchmark_pct_by_lookback` slices per distinct lookback below. Never
    raises: any failure (broker unavailable, KIS call failed, empty
    result) is caught and logged as a progress-line warning (matching this
    module's existing unbuffered-print convention — see the module
    docstring's `python -u` note — rather than a new logging channel just
    for this one function), returning `None` so the caller degrades that
    (market, date) to total momentum for every affected ticker.

    domestic deliberately goes through `broker.fetch_index_daily_chart`
    (KOSPI, `_KOSPI_INDEX_CODE`) — the SAME dedicated KIS index-chart TR
    `app.execution.market_regime.refresh_domestic_regime` already uses —
    rather than `get_price_series(benchmark_for_market("domestic"))`
    (= "^KS11", app.execution.portfolio_risk), which would route through
    yfinance and reintroduce for the benchmark leg the exact KIS-vs-
    yfinance source-mixing app.chat.screening's module docstring documents
    as a real prior bug for the ticker leg (2026-08-21 fix). The broker is
    resolved the same way `_universe_from_live_axes` above does
    (`app.brokers.get_broker("kis")`) — a CLI-tool accessor, NOT
    `app.web.state.kis_broker()`, which is scoped to the web app's
    connection lifecycle and has no business being reached from this
    offline sweep.

    overseas uses SPY via `app.dataflows.kis_provider.get_price_series` —
    the identical reason `app.execution.market_regime.refresh_overseas_
    regime` does: no US index-chart TR exists in this codebase, and SPY
    through kis_provider is source-consistent with how this tool's own
    overseas ticker numerators are already fetched (this module's earlier
    2026-08-21 comment on `get_price_series` above).

    Point-in-time correct — the query is bounded by `end`, same as every
    other historical read in Phase A — UNLIKE `--universe-axes`, which
    reads live KIS ranking state with no historical equivalent (see the
    module docstring's 2026-09-02 #3 addition). A residual-momentum sweep
    over a past date range is fully reproducible; a `--universe-axes` run
    is not, and the two "reads live state" caveats are unrelated to each
    other."""
    start = end - timedelta(days=max_lookback_days)
    try:
        if market == "domestic":
            from app.brokers import get_broker

            broker = await get_broker("kis")
            bars = await broker.fetch_index_daily_chart(
                _KOSPI_INDEX_CODE, start.strftime("%Y%m%d"), end.strftime("%Y%m%d")
            )
            return {
                date(int(b.day[:4]), int(b.day[4:6]), int(b.day[6:])): b.close
                for b in bars
                if b.close > 0
            }
        return await get_price_series("SPY", start, end)
    except Exception as exc:  # noqa: BLE001 - best-effort, see docstring
        print(f"  ! benchmark fetch failed for market={market} end={end}: {exc}", flush=True)
        return None


async def _benchmark_pct_by_lookback(
    markets: set[str], d: date, distinct_lookbacks: list[int]
) -> dict[tuple[str, int, date], float | None]:
    """Fetches `_benchmark_series` exactly ONCE per market for date `d`
    (at the WIDEST of `distinct_lookbacks`), then slices that one series
    down to each distinct lookback's own `[d - lb, d]` window and reduces
    it with `pct_change` — the same "collapse one wide fetch, slice per
    lookback client-side" trick `candle_run_cache()`/`_fetch_kis_candles`
    already perform for the ticker legs in Phase A (see that mechanism's
    own comment in app.dataflows.kis_provider), applied here to the
    benchmark leg instead: one wide fetch per (market, date), never once
    per ticker and never once per lookback.

    The slice reproduces `app.chat.screening._momentum`'s own window
    exactly — first-to-last `pct_change` over `[d - lb, d]` — so the
    benchmark leg and the ticker leg it gets subtracted from
    (`residual_pct`) stay unit-consistent (same window, same convention).
    A market whose `_benchmark_series` call returned `None` contributes
    `None` for every one of its (market, lb, d) keys, which
    `_variant_score`/`residual_pct` then treats as "degrade this ticker to
    total momentum for this date" — see `residual_pct`'s own docstring."""
    out: dict[tuple[str, int, date], float | None] = {}
    max_lb = max(distinct_lookbacks) if distinct_lookbacks else 0
    for market in markets:
        series = await _benchmark_series(market, d, max_lb)
        for lb in distinct_lookbacks:
            if series is None:
                out[(market, lb, d)] = None
                continue
            window_start = d - timedelta(days=lb)
            sliced = {day: px for day, px in series.items() if window_start <= day <= d}
            out[(market, lb, d)] = pct_change(sliced)
    return out


# --- --dump-json collector ------------------------------------------------

@dataclass
class _DumpCollector:
    """In-memory sink for `--dump-json`. Filled by `_phase_a`/`sweep` with
    values they ALREADY compute (no extra fetches); `None` collector = flag
    off = zero behavior change. Only fully-completed dates are recorded, so
    an "incomplete" dump never contains a half-scored date.
    `--dump-json`용 수집기 — 이미 계산된 값만 담는다(추가 fetch 없음)."""

    scores: dict[str, dict[str, dict[str, float | None]]] = field(default_factory=dict)  # "lb|date" -> ticker -> raw/atr
    benchmark_pct: dict[str, dict[str, float | None]] = field(default_factory=dict)  # "lb|date" -> market -> pct
    forward: dict[str, dict[str, dict[str, Any]]] = field(default_factory=dict)  # date -> ticker -> detail
    completed_dates: list[date] = field(default_factory=list)
    signed_scores: dict[int, dict[str, dict[str, float | None]]] = field(default_factory=dict)  # variant idx -> date -> ticker -> score
    variant_results: list[VariantResult] = field(default_factory=list)  # variants fully finished so far
    meta: dict[str, Any] = field(default_factory=dict)  # filled by _main (argv, args, universe, ...)

    def record_date(
        self,
        d: date,
        scored_by_lb: dict[int, dict[str, MomentumScore]],
        forward_detail: dict[str, tuple[float | None, date | None, date | None]],
        benchmark: dict[tuple[str, int, date], float | None],
    ) -> None:
        for lb, scored in scored_by_lb.items():
            self.scores[f"{lb}|{d.isoformat()}"] = {
                t: {"raw_pct": s.raw_pct, "atr_pct": s.atr_pct} for t, s in scored.items()
            }
        for (market, lb, bd), pct in benchmark.items():
            self.benchmark_pct.setdefault(f"{lb}|{bd.isoformat()}", {})[market] = pct
        self.forward[d.isoformat()] = {
            t: {
                "fwd_return": ret,
                "entry_date": entry.isoformat() if entry else None,
                "exit_price_date": exit_.isoformat() if exit_ else None,
            }
            for t, (ret, entry, exit_) in forward_detail.items()
        }
        self.completed_dates.append(d)


# --- Phase A: all network -------------------------------------------------

async def _phase_a(
    universe: list[str],
    trade_dates: list[date],
    distinct_lookbacks: list[int],
    holding_days: int,
    *,
    concurrency: int = 8,
    benchmark_markets: set[str] | None = None,
    collector: _DumpCollector | None = None,
) -> tuple[
    dict[tuple[int, date], dict[str, MomentumScore]],
    dict[date, dict[str, float | None]],
    dict[tuple[str, int, date], float | None],
]:
    """For each date: score `universe` once per DISTINCT lookback (cached by
    (lookback, date)), compute the forward return of EVERY ticker in
    the universe — not just the picks, since RankIC needs the whole
    cross-section — and, when `benchmark_markets` is given (any variant in
    this run uses `momentum_mode="residual"` — see `sweep()`), fetch each
    named market's benchmark return once per date too (third return value,
    keyed `(market, lookback_days, date)` — see `_benchmark_pct_by_
    lookback`). Sequential across dates (bounds rate-limit risk and
    keeps progress legible); concurrent within a date via gather +
    Semaphore. This is the only part that touches the network.

    `benchmark_markets=None` (the default, every pre-2026-09-02 caller and
    every "total"-only sweep) skips the benchmark fetch entirely — zero
    added network cost for a run that never needed it."""
    sem = asyncio.Semaphore(concurrency)
    scored_cache: dict[tuple[int, date], dict[str, MomentumScore]] = {}
    forward_cache: dict[date, dict[str, float | None]] = {}
    benchmark_cache: dict[tuple[str, int, date], float | None] = {}

    async def _score(lb: int, d: date) -> tuple[int, dict[str, MomentumScore]]:
        async with sem:
            return lb, await _scored_candidates(universe, lb, as_of=d)

    async def _price(t: str, d: date) -> tuple[str, float | None]:
        async with sem:
            return t, await _forward_return(t, d, holding_days)

    async def _price_detail(t: str, d: date) -> tuple[str, tuple[float | None, date | None, date | None]]:
        # --dump-json only: same value as _forward_return plus the price dates used.
        async with sem:
            return t, await _forward_detail(t, d, holding_days)

    total = len(trade_dates)
    for i, d in enumerate(trade_dates, start=1):
        # One candle cache scoped to this single date. The D lookbacks'
        # get_price_series + get_atr_pct calls all use end=d, so they
        # collapse onto one (ticker, end) key each -> KIS requests per
        # ticker go from 2D+1 to 2 (one wide 100-day fetch for end=d, one
        # for _forward_return's end=d+holding+5). Values are unchanged:
        # _fetch_kis_candles re-slices the wide window client-side to each
        # call's own [start, end], so the same subset comes back.
        # Per-date, not whole-run: nothing is reused across dates (end
        # differs every date), so widening the scope adds zero hits and
        # grows the dict to N*dates*2 with no eviction. And _forward_return
        # (end=d+holding+5) stays a distinct key from scoring (end=d), so
        # future candles never leak into the scoring path (point-in-time).
        #
        # The benchmark fetch below also runs inside this same scope —
        # the overseas leg (SPY, via get_price_series) participates in the
        # exact same KIS candle-run cache as any other US-ticker-shaped
        # symbol this date, so a universe that happens to already fetch SPY
        # (unlikely, but not impossible) would coalesce onto the same
        # in-flight request rather than firing a second one.
        with candle_run_cache():
            scored_pairs = await asyncio.gather(*(_score(lb, d) for lb in distinct_lookbacks))
            for lb, scored in scored_pairs:
                scored_cache[(lb, d)] = scored
            detail_here: dict[str, tuple[float | None, date | None, date | None]] = {}
            if collector is None:
                priced_pairs = await asyncio.gather(*(_price(t, d) for t in universe))
                forward_cache[d] = {t: r for t, r in priced_pairs}
            else:
                detail_pairs = await asyncio.gather(*(_price_detail(t, d) for t in universe))
                detail_here = dict(detail_pairs)
                forward_cache[d] = {t: v[0] for t, v in detail_here.items()}
            bench_here: dict[tuple[str, int, date], float | None] = {}
            bench_note = "-"
            if benchmark_markets:
                bench_here = await _benchmark_pct_by_lookback(benchmark_markets, d, distinct_lookbacks)
                benchmark_cache.update(bench_here)
                needed = [
                    benchmark_cache.get((m, lb, d)) for m in benchmark_markets for lb in distinct_lookbacks
                ]
                bench_note = "ok" if needed and all(v is not None for v in needed) else "degraded"
        if collector is not None:
            collector.record_date(
                d, {lb: scored_cache[(lb, d)] for lb in distinct_lookbacks}, detail_here, bench_here
            )
        scored_here = len(scored_cache[(distinct_lookbacks[0], d)])
        priced_here = sum(1 for r in forward_cache[d].values() if r is not None)
        print(f"[{i}/{total}] {d} scored={scored_here} priced={priced_here} bench={bench_note}", flush=True)

    return scored_cache, forward_cache, benchmark_cache


# --- Phase B: pure + correlation clustering only -------------------------

async def _clusters_cached(
    cache: dict[tuple[frozenset[str], date], dict[str, int] | None],
    pool_tickers: list[str],
    trade_date: date,
) -> dict[str, int] | None:
    """`None` means clustering FAILED for this (pool, date) — distinct from
    `{}`, a genuine "nothing correlated" result. Without this distinction,
    a caller can't tell "dedup ran and found nothing" from "dedup never
    ran" — dedupe_correlated({}) drops nothing either way, so a failure
    silently made the dedup=on arm behave exactly like dedup=off, with no
    warning (2026-09-11 fix).

    The failure IS still negative-cached under this (pool, date) key —
    dates are already independent cache keys, so this never suppresses a
    later date's clustering; the only re-hit is a different variant
    producing the identical pool on the SAME date, where an immediate retry
    would re-issue the exact price-history fetches that just failed (the
    rate-limit burst, if that's the cause, is still in progress)."""
    key = (frozenset(pool_tickers), trade_date)
    if key not in cache:
        try:
            cache[key] = await compute_correlation_clusters(pool_tickers, as_of=trade_date)
        except Exception as exc:
            print(f"  ! correlation clustering failed — skipping dedup for date={trade_date} "
                  f"pool={len(pool_tickers)}: {exc}", flush=True)
            cache[key] = None
    return cache[key]


def _pool_size(scored_count: int, top_n: int) -> int:
    # MUST match app.chat.screening._ranked_and_deduped — update together.
    return min(scored_count, top_n * settings.screening_correlation_pool_multiplier)


async def _variant_observations(
    variant: Variant,
    trade_dates: list[date],
    scored_cache: dict[tuple[int, date], dict[str, MomentumScore]],
    forward_cache: dict[date, dict[str, float | None]],
    top_n: int,
    cluster_cache: dict[tuple[frozenset[str], date], dict[str, int] | None],
    benchmark_cache: dict[tuple[str, int, date], float | None] | None = None,
    signed_sink: dict[str, dict[str, float | None]] | None = None,
) -> list[DateObservation]:
    """`benchmark_cache` (2026-09-02, screening-improvement-research #4) is
    only consulted when `variant.momentum_mode == "residual"` — a "total"
    variant never looks it up at all, so passing `None` (every pre-existing
    caller) is safe and behaves exactly as before this addition.

    `dedup_drop_rate` is computed from the SAME `pool` variable both before
    and after `dedupe_correlated` reassigns it — `pre_dedup_top` is
    captured immediately after `rank()` builds the pool, before dedup ever
    runs, so it reflects the leaderboard `app.chat.screening.dedupe_
    correlated` would otherwise never let this caller see.

    `signed_sink` (--dump-json only, default None = no-op): receives
    `{iso_date: signed_scores}` for every date that yields an observation —
    the exact values the ranking used (sign already applied)."""
    benchmark_cache = benchmark_cache or {}
    observations: list[DateObservation] = []
    prev_picks: list[str] | None = None

    for d in trade_dates:
        scored = scored_cache.get((variant.lookback_days, d), {})
        forward = forward_cache.get(d, {})
        if not scored:
            continue

        residual_fallbacks = 0
        signed_scores: dict[str, float | None] = {}
        for t, s in scored.items():
            benchmark_pct = None
            if variant.momentum_mode == "residual":
                benchmark_pct = benchmark_cache.get((ticker_market(t), variant.lookback_days, d))
                if benchmark_pct is None:
                    residual_fallbacks += 1
            signed_scores[t] = variant.sign * _variant_score(s, variant, benchmark_pct)

        pool = rank(signed_scores, _pool_size(len(scored), top_n))
        pre_dedup_top = [t for t, _ in pool[:top_n]]
        dedup_drop_rate: float | None = None
        dedup_degraded = False
        if variant.use_correlation_dedup:
            if len(pool) > top_n:
                clusters = await _clusters_cached(cluster_cache, [t for t, _ in pool], d)
                if clusters is None:
                    dedup_degraded = True
                else:
                    pool = dedupe_correlated(pool, clusters)
            # 2026-09-11 fix: the denominator here MUST be len(pre_dedup_top),
            # not top_n. Whenever len(pool) > top_n above (dedup actually had
            # more than top_n candidates to choose from), pre_dedup_top always
            # has exactly top_n elements, so this is behavior-identical to the
            # old top_n-denominator formula. The bug was the OTHER branch: on
            # a thin-universe day (fewer scored tickers than top_n, so
            # _pool_size caps pool below top_n), the `if len(pool) > top_n`
            # guard above is False and dedupe_correlated never runs — pool is
            # untouched, so pre_dedup_top and post_dedup_top are the identical
            # list. The old code still divided by the fixed top_n, so overlap
            # (== len(pre_dedup_top), which is < top_n on a thin day)
            # produced a fabricated positive "drop rate" on a day where dedup
            # never ran and nothing was ever dropped. Dividing by
            # len(pre_dedup_top) instead makes that case correctly yield 0.0.
            # 얇은 유니버스에서 dedup이 아예 실행되지 않은 날에도 분모가 top_n
            # 고정이라 가짜 drop rate가 나오던 버그 — 분모를 실제 pick 개수로.
            if pre_dedup_top and not dedup_degraded:
                post_dedup_top = [t for t, _ in pool[:top_n]]
                overlap = len(set(pre_dedup_top) & set(post_dedup_top))
                dedup_drop_rate = 1 - overlap / len(pre_dedup_top)
        picks = [t for t, _ in pool[:top_n]]
        if not picks:
            continue

        pick_fwd = [forward[t] for t in picks if forward.get(t) is not None]
        universe_fwd = [r for t, r in forward.items() if t in scored and r is not None]

        observations.append(
            DateObservation(
                trade_date=d,
                picks=picks,
                pick_forward_returns=pick_fwd,
                universe_forward_returns=universe_fwd,
                excess_return=factor_metrics.excess_return(pick_fwd, universe_fwd),
                rank_ic=factor_metrics.rank_ic(signed_scores, forward),
                quantile_spread=factor_metrics.quantile_spread(signed_scores, forward),
                turnover=factor_metrics.turnover(prev_picks, picks) if prev_picks is not None else None,
                residual_fallbacks=residual_fallbacks,
                dedup_drop_rate=dedup_drop_rate,
                dedup_degraded=dedup_degraded,
            )
        )
        prev_picks = picks
        if signed_sink is not None:
            signed_sink[d.isoformat()] = dict(signed_scores)

    return observations


def _summarise_variant(
    variant: Variant,
    observations: list[DateObservation],
    *,
    cadence_days: int,
    holding_days: int,
    worth_bar: float,
    bootstrap_iters: int,
    seed: int,
) -> VariantResult:
    pooled_picks = [r for o in observations for r in o.pick_forward_returns]
    excesses = [o.excess_return for o in observations if o.excess_return is not None]
    rank_ics = [o.rank_ic for o in observations]
    spreads = [o.quantile_spread for o in observations if o.quantile_spread is not None]
    turnovers = [o.turnover for o in observations if o.turnover is not None]

    # Date-clustered blocks for the excess-return CI: ONE value per date
    # (that date's own excess_return, already mean(picks) - mean(universe))
    # in a single-element block -- matching ic_blocks' convention just below,
    # and matching mean_excess_return's own weighting (a plain mean over one
    # number per date). This used to be each date's raw per-pick demeaned
    # returns pooled across resampled blocks, which implicitly weighted each
    # date by its own pick count (dates with more picks contributed more
    # individual values to the pooled mean) -- a DIFFERENT statistic than
    # the date-weighted mean_excess_return the CI is supposed to describe,
    # and the two diverge whenever pick counts vary across dates (a thin-
    # universe day picks fewer than top_n). Resampling dates (not individual
    # picks) still captures the within-date correlation structure this
    # docstring originally cared about -- it's a cluster-bootstrap over
    # cluster MEANS now, not over raw pooled observations.
    # 포인트추정치(날짜별 평균의 평균)와 같은 가중치로 신뢰구간을 재구성 —
    # 기존 방식은 날짜별 pick 개수에 따라 암묵적으로 가중치가 달라졌음.
    excess_blocks: list[list[float]] = [[o.excess_return] for o in observations if o.excess_return is not None]
    ic_blocks = [[ic] for ic in rank_ics if ic is not None]

    # 2026-09-02 roll-ups — see DateObservation/VariantResult's own
    # docstrings for what each field means.
    residual_fallback_dates = sum(1 for o in observations if o.residual_fallbacks > 0)
    drop_rates = [o.dedup_drop_rate for o in observations if o.dedup_drop_rate is not None]
    mean_dedup_drop_rate = (sum(drop_rates) / len(drop_rates)) if drop_rates else None
    dedup_degraded_dates = sum(1 for o in observations if o.dedup_degraded)

    return VariantResult(
        variant=variant,
        n_dates=len(observations),
        observations=observations,
        mean_forward_return=(sum(pooled_picks) / len(pooled_picks)) if pooled_picks else None,
        mean_excess_return=(sum(excesses) / len(excesses)) if excesses else None,
        excess_return_ci95=factor_metrics.block_bootstrap_ci(
            excess_blocks, n_bootstrap=bootstrap_iters, seed=seed
        ),
        mean_rank_ic=(
            sum(x for x in rank_ics if x is not None) / len([x for x in rank_ics if x is not None])
            if any(x is not None for x in rank_ics)
            else None
        ),
        rank_ic_ci95=factor_metrics.block_bootstrap_ci(ic_blocks, n_bootstrap=bootstrap_iters, seed=seed),
        icir=factor_metrics.icir(rank_ics),
        mean_quantile_spread=(sum(spreads) / len(spreads)) if spreads else None,
        mean_turnover=(sum(turnovers) / len(turnovers)) if turnovers else None,
        worth_rate=factor_metrics.worth_analyzing_rate(pooled_picks, bar=worth_bar),
        upside_rate=factor_metrics.upside_hit_rate(pooled_picks, bar=worth_bar),
        n_eff=factor_metrics.effective_sample_size(len(observations), cadence_days, holding_days),
        residual_fallback_dates=residual_fallback_dates,
        mean_dedup_drop_rate=mean_dedup_drop_rate,
        dedup_degraded_dates=dedup_degraded_dates,
    )


async def sweep(
    universe: list[str],
    trade_dates: list[date],
    variants: list[Variant],
    top_n: int,
    holding_days: int,
    cadence_days: int,
    *,
    worth_bar: float = 0.03,
    bootstrap_iters: int = 10_000,
    seed: int = 0,
    collector: _DumpCollector | None = None,
) -> list[VariantResult]:
    """Phase A once for every distinct lookback across `variants`, then
    Phase B per variant. See the module docstring for why the two phases
    are split.

    2026-09-02 (residual momentum): if any `variant.momentum_mode ==
    "residual"`, this resolves which market(s) `universe` actually touches
    (`ticker_market`, app.execution.portfolio_risk) and passes that set
    into `_phase_a` so it fetches each market's benchmark once per date —
    see `_phase_a`/`_benchmark_pct_by_lookback`'s own docstrings. Rejects
    (SystemExit, before any network call — same fail-loud-fail-before-I/O
    shape as `_check_universe_source_args`) any residual variant whose
    `lookback_days` exceeds `_RESIDUAL_MAX_LOOKBACK_DAYS` — see that
    constant's own comment for the KIS-vs-yfinance source-mixing this
    prevents. A run with no residual variant pays zero extra cost:
    `benchmark_markets` stays `None` and `_phase_a` skips the benchmark
    fetch entirely."""
    distinct_lookbacks = sorted({v.lookback_days for v in variants})

    needs_benchmark = any(v.momentum_mode != "total" for v in variants)
    if needs_benchmark:
        bad = [v.label for v in variants if v.momentum_mode != "total" and v.lookback_days > _RESIDUAL_MAX_LOOKBACK_DAYS]
        if bad:
            raise SystemExit(
                f"residual variants {bad} exceed lookback {_RESIDUAL_MAX_LOOKBACK_DAYS}d — the ticker "
                "numerator would route to yfinance while the KOSPI benchmark stays on KIS (source mixing)"
            )
    benchmark_markets = {ticker_market(t) for t in universe} if needs_benchmark else None

    scored_cache, forward_cache, benchmark_cache = await _phase_a(
        universe, trade_dates, distinct_lookbacks, holding_days,
        benchmark_markets=benchmark_markets, collector=collector,
    )

    cluster_cache: dict[tuple[frozenset[str], date], dict[str, int] | None] = {}
    results: list[VariantResult] = []
    for v_idx, variant in enumerate(variants):
        signed_sink: dict[str, dict[str, float | None]] | None = {} if collector is not None else None
        observations = await _variant_observations(
            variant, trade_dates, scored_cache, forward_cache, top_n, cluster_cache,
            benchmark_cache=benchmark_cache, signed_sink=signed_sink,
        )
        results.append(
            _summarise_variant(
                variant,
                observations,
                cadence_days=cadence_days,
                holding_days=holding_days,
                worth_bar=worth_bar,
                bootstrap_iters=bootstrap_iters,
                seed=seed,
            )
        )
        if collector is not None:
            collector.signed_scores[v_idx] = signed_sink or {}
            collector.variant_results.append(results[-1])
    return results


# --- --dump-json writer -------------------------------------------------

_DUMP_SCHEMA = "screening-sweep-dump-v1"


def _jsonable(obj: Any) -> Any:
    """Recursively make `obj` strict-JSON safe: NaN/inf -> None, date/datetime
    -> ISO string, tuple -> list, non-str dict keys -> str.
    NaN/inf는 null로 — 엄격 JSON(allow_nan=False)을 보장."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj


def _git_info() -> tuple[str | None, bool | None]:
    """(HEAD sha, dirty?) of the repo this module lives in; (None, None)-style
    per-field fallback on any failure — provenance only, never fatal."""
    cwd = str(Path(__file__).resolve().parents[2])

    def _run(*cmd: str) -> str | None:
        try:
            out = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=15, check=True)
            return out.stdout
        except Exception:  # noqa: BLE001 - provenance is best-effort
            return None

    head = _run("git", "rev-parse", "HEAD")
    porcelain = _run("git", "status", "--porcelain")
    return (head.strip() if head else None), (bool(porcelain.strip()) if porcelain is not None else None)


def _observation_dump(o: DateObservation, forward: dict[str, dict[str, Any]], signed: dict[str, float | None]) -> dict[str, Any]:
    fwd_by_pick = {
        t: forward.get(t, {}).get("fwd_return") for t in o.picks if forward.get(t, {}).get("fwd_return") is not None
    }
    return {
        "picks": o.picks,
        "pick_forward_returns": fwd_by_pick,
        "universe_forward_returns": o.universe_forward_returns,
        "excess_return": o.excess_return,
        "rank_ic": o.rank_ic,
        "quantile_spread": o.quantile_spread,
        "turnover": o.turnover,
        "residual_fallbacks": o.residual_fallbacks,
        "dedup_drop_rate": o.dedup_drop_rate,
        "dedup_degraded": o.dedup_degraded,
        "signed_scores": signed,
    }


def _build_dump(collector: _DumpCollector, *, status: str, finished_at: datetime) -> dict[str, Any]:
    meta = collector.meta
    variants_out = []
    for v_idx, r in enumerate(collector.variant_results):
        summary = {k: v for k, v in asdict(r).items() if k not in ("variant", "observations")}
        signed_by_date = collector.signed_scores.get(v_idx, {})
        variants_out.append(
            {
                "variant": asdict(r.variant),
                "observations": {
                    o.trade_date.isoformat(): _observation_dump(
                        o, collector.forward.get(o.trade_date.isoformat(), {}), signed_by_date.get(o.trade_date.isoformat(), {})
                    )
                    for o in r.observations
                },
                "summary": summary,
            }
        )
    fwd_total = sum(len(v) for v in collector.forward.values())
    fwd_null = sum(1 for v in collector.forward.values() for d in v.values() if d["fwd_return"] is None)
    dump = {
        "schema": _DUMP_SCHEMA,
        "status": status,
        "started_at": meta.get("started_at"),
        "finished_at": finished_at,
        "git_head": meta.get("git_head"),
        "git_dirty": meta.get("git_dirty"),
        "argv": meta.get("argv"),
        "args": meta.get("args"),
        "universe": meta.get("universe"),
        "trade_dates": collector.completed_dates,
        "trade_dates_requested": meta.get("trade_dates_requested"),
        "holding_days": meta.get("holding_days"),
        "cadence_days": meta.get("cadence_days"),
        "top_n": meta.get("top_n"),
        "benchmark": meta.get("benchmark"),
        "scores": collector.scores,
        "benchmark_pct": collector.benchmark_pct,
        "forward": collector.forward,
        "variants": variants_out,
        "counters": {
            "forward_total": fwd_total,
            "forward_null": fwd_null,
            "residual_fallback_dates": {r.variant.label: r.residual_fallback_dates for r in collector.variant_results},
            "dedup_degraded_dates": {r.variant.label: r.dedup_degraded_dates for r in collector.variant_results},
        },
    }
    return _jsonable(dump)


def _write_dump(path: str, collector: _DumpCollector, *, status: str) -> None:
    """Atomic strict-JSON write (tmp in the same dir + os.replace), UTF-8 no BOM."""
    payload = json.dumps(
        _build_dump(collector, status=status, finished_at=datetime.now(timezone.utc)),
        ensure_ascii=False, allow_nan=False,
    )
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f"{target.name}.{os.getpid()}.tmp")
    try:
        tmp.write_bytes(payload.encode("utf-8"))
        os.replace(tmp, target)
    finally:
        if tmp.exists():
            tmp.unlink()


# --- rendering ----------------------------------------------------------

def _fmt_pct(v: float | None) -> str:
    return f"{v:+.2%}" if v is not None else "-"


def _fmt_num(v: float | None, places: int = 3) -> str:
    return f"{v:+.{places}f}" if v is not None else "-"


def _fmt_ci(ci: tuple[float, float] | None, pct: bool = True) -> str:
    if ci is None:
        return "-"
    if pct:
        return f"[{ci[0]:+.2%},{ci[1]:+.2%}]"
    return f"[{ci[0]:+.3f},{ci[1]:+.3f}]"


def _universe_round_trip_cost(universe: list[str]) -> float:
    """Average buy+sell cost across the universe, as a fraction (not bps)."""
    if not universe:
        return 0.0
    per_ticker = [
        (transaction_cost_bps_for(t, "buy") + transaction_cost_bps_for(t, "sell")) / 10_000
        for t in universe
    ]
    return sum(per_ticker) / len(per_ticker)


def render(
    results: list[VariantResult],
    *,
    universe: list[str],
    cadence_days: int,
    holding_days: int,
    seed: int,
    bootstrap_iters: int,
) -> str:
    n_dates = max((r.n_dates for r in results), default=0)
    lines = [
        f"universe={len(universe)} cadence={cadence_days}d holding={holding_days}d "
        f"dates={n_dates} seed={seed} boot={bootstrap_iters}",
        "",
    ]
    header = (
        f"{'variant':<18} {'n':>3} {'n_eff':>6} {'fwd_ret':>9} {'excess':>9} "
        f"{'excess_CI95':>19} {'IC':>7} {'IC_CI95':>17} {'ICIR':>7} {'spread':>9} "
        f"{'turn':>6} {'worth%':>7} {'up%':>6} {'resid_fb':>10} {'dedup_drop':>10} {'dedup_deg':>10}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    ordered = sorted(
        results,
        key=lambda r: (r.mean_excess_return is None, -(r.mean_excess_return or 0.0)),
    )
    for r in ordered:
        lines.append(
            f"{r.variant.label:<18} {r.n_dates:>3} "
            f"{(f'{r.n_eff:.1f}' if r.n_eff is not None else '-'):>6} "
            f"{_fmt_pct(r.mean_forward_return):>9} {_fmt_pct(r.mean_excess_return):>9} "
            f"{_fmt_ci(r.excess_return_ci95):>19} {_fmt_num(r.mean_rank_ic):>7} "
            f"{_fmt_ci(r.rank_ic_ci95, pct=False):>17} {_fmt_num(r.icir, 2):>7} "
            f"{_fmt_pct(r.mean_quantile_spread):>9} "
            f"{(f'{r.mean_turnover:.2f}' if r.mean_turnover is not None else '-'):>6} "
            f"{(f'{r.worth_rate:.0%}' if r.worth_rate is not None else '-'):>7} "
            f"{(f'{r.upside_rate:.0%}' if r.upside_rate is not None else '-'):>6} "
            f"{(f'{r.residual_fallback_dates}/{r.n_dates}' if r.n_dates else '-'):>10} "
            f"{(f'{r.mean_dedup_drop_rate:.0%}' if r.mean_dedup_drop_rate is not None else '-'):>10} "
            f"{(f'{r.dedup_degraded_dates}/{r.n_dates}' if r.n_dates else '-'):>10}"
        )

    lines.append("")
    round_trip = _universe_round_trip_cost(universe)
    for r in ordered:
        for warning in _hard_rule_warnings(r, round_trip):
            lines.append(f"!  {r.variant.label}: {warning}")

    lines.append("")
    lines.append(factor_metrics.INTERPRETATION_GUIDE)
    return "\n".join(lines)


def _hard_rule_warnings(r: VariantResult, round_trip_cost: float) -> list[str]:
    out: list[str] = []
    if r.excess_return_ci95 is not None and r.excess_return_ci95[0] <= 0 <= r.excess_return_ci95[1]:
        out.append("excess-return 95% CI crosses 0 — unproven")
    if r.n_eff is not None and r.n_eff < 20:
        out.append(f"n_eff={r.n_eff:.1f} < 20 — underpowered")
    if (
        r.mean_turnover is not None
        and r.mean_excess_return is not None
        and r.mean_turnover * round_trip_cost >= r.mean_excess_return
    ):
        out.append(
            f"turnover cost ({r.mean_turnover * round_trip_cost:+.2%}) >= excess return "
            f"({r.mean_excess_return:+.2%}) — edge eaten by trading cost"
        )
    if (
        r.mean_rank_ic is not None
        and r.mean_excess_return is not None
        and r.mean_rank_ic != 0
        and r.mean_excess_return != 0
        and (r.mean_rank_ic > 0) != (r.mean_excess_return > 0)
    ):
        out.append("RankIC sign disagrees with excess-return sign — internally inconsistent, treat as noise")
    # 2026-09-02: a "residual" variant that degraded to total momentum on
    # more than 20% of its dates (see DateObservation.residual_fallbacks'
    # docstring) isn't a clean A/B against the "total" arm any more — most
    # of what it measured was total momentum wearing a residual label.
    if r.residual_fallback_dates and r.n_dates and r.residual_fallback_dates / r.n_dates > 0.2:
        out.append(
            f"residual arm degraded to total momentum on "
            f"{r.residual_fallback_dates / r.n_dates:.0%} of dates — not a clean A/B, "
            "fix the benchmark fetch and re-run"
        )
    # 2026-09-11: any dedup failure corrupts this arm's identity as a clean
    # dedup=on vs dedup=off A/B (see DateObservation.dedup_degraded) — no
    # 20% threshold like the residual-fallback warning above, since even a
    # single degraded date means part of this run silently behaved as
    # dedup=off while still being reported/labeled as dedup=on.
    if r.dedup_degraded_dates and r.n_dates:
        out.append(
            f"correlation clustering failed on {r.dedup_degraded_dates}/{r.n_dates} dates — "
            "this arm partly behaved as dedup=off, not a clean A/B"
        )
    return out


# --- universe-from-live-axes (screening-improvement-research #3) -------

_UNIVERSE_AXES = ("trading_value", "market_cap", "fluctuation_losers", "disparity_low")


async def _universe_from_live_axes(
    axes: list[str], count: int, *, min_volume: int, disparity_window_days: int, max_price: float | None = None
) -> list[str]:
    """Resolve a sweep universe from LIVE KRX discovery axes instead of a
    fixed ticker list — the ONLY network call anywhere in this tool outside
    Phase A (see the module docstring's 2026-09-02 addition for the
    consequence: it reads live KIS ranking state, so two runs on different
    days are not comparable to each other, only the two arms WITHIN one
    comparison run are, since these ranking TRs have no historical/point-
    in-time equivalent — a fair A/B needs both arms resolved from the same
    day's snapshot).

    `axes` names one or more of `trading_value` (krx_top_n), `market_cap`
    (krx_market_cap_top_n), `fluctuation_losers` (krx_fluctuation_losers_n),
    or `disparity_low` (krx_disparity_low_n) — see app.live.universe for
    what each axis actually surfaces and why. Unknown axis names raise
    SystemExit immediately, before any broker/network call, naming the bad
    axis — the same "fail loud, fail before I/O" shape as this module's
    other CLI validation (`_variants_from_args`'s mutual-exclusion check).
    Results are unioned via `dict.fromkeys`, preserving `axes`' given
    order, so the same ticker named by two axes appears once, at the
    position of whichever axis listed it first."""
    for axis in axes:
        if axis not in _UNIVERSE_AXES:
            raise SystemExit(f"unknown --universe-axes entry {axis!r} — must be one of {', '.join(_UNIVERSE_AXES)}")

    from app.brokers import get_broker
    from app.live.universe import krx_disparity_low_n, krx_fluctuation_losers_n, krx_market_cap_top_n, krx_top_n

    broker = await get_broker("kis")
    tickers: list[str] = []
    for axis in axes:
        if axis == "trading_value":
            tickers += await krx_top_n(broker, count, max_price=max_price)
        elif axis == "market_cap":
            tickers += await krx_market_cap_top_n(broker, count, max_price=max_price)
        elif axis == "fluctuation_losers":
            tickers += await krx_fluctuation_losers_n(broker, count, min_volume=min_volume, max_price=max_price)
        else:  # "disparity_low"
            tickers += await krx_disparity_low_n(
                broker, count, window_days=disparity_window_days, min_volume=min_volume, max_price=max_price
            )
    return list(dict.fromkeys(tickers))


# --- CLI ---------------------------------------------------------------

def _check_universe_source_args(args: argparse.Namespace) -> None:
    """`--tickers` and `--universe-axes` pick two different universe
    sources (a fixed list vs. a live KRX ranking read) — pass at most one,
    same "reject both, don't silently pick a precedence" shape as
    `_variants_from_args`'s own mutual-exclusion check."""
    if args.universe_axes and args.tickers:
        raise SystemExit("pass either --tickers or --universe-axes, not both")


def _variants_from_args(args: argparse.Namespace) -> list[Variant]:
    """`momentum_mode` (2026-09-02) is validated here — any variant (from
    either the `--variants-json` or the `--lookback-options` grid path)
    whose `momentum_mode` isn't `"total"`/`"residual"` raises SystemExit
    naming the bad value, before `sweep()` ever runs (network cost is a
    non-issue for the lookback-grid path — it never sets `momentum_mode`
    at all — but the JSON path can, so both are validated the same way for
    one consistent contract)."""
    if args.variants_json and args.lookback_options is not None:
        raise SystemExit("pass either --lookback-options or --variants-json, not both")
    if args.variants_json:
        with open(args.variants_json, encoding="utf-8") as fh:
            raw = json.load(fh)
        if not isinstance(raw, list):
            raise SystemExit("--variants-json must contain a JSON list of variant objects")
        variants = []
        for i, d in enumerate(raw):
            variants.append(
                Variant(
                    label=str(d.get("label", f"variant{i}")),
                    lookback_days=int(d["lookback_days"]),
                    use_volatility_adjustment=bool(d.get("use_volatility_adjustment", True)),
                    use_correlation_dedup=bool(d.get("use_correlation_dedup", True)),
                    sign=int(d.get("sign", 1)),
                    momentum_mode=str(d.get("momentum_mode", "total")),
                )
            )
    else:
        spec = args.lookback_options if args.lookback_options is not None else "5,10,15,20"
        variants = [Variant(label=f"lb{int(n)}", lookback_days=int(n)) for n in spec.split(",")]

    bad = [v.momentum_mode for v in variants if v.momentum_mode not in ("total", "residual")]
    if bad:
        raise SystemExit(f"unknown momentum_mode {bad[0]!r} — must be 'total' or 'residual'")
    return variants


async def _main() -> None:
    if sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="Measure app.chat.screening variants offline (LLM-free, code-only). "
        "Run with `python -u -m app.backtest.screening_sweep` to avoid buffered-stdout hangs."
    )
    parser.add_argument("--start", required=True, type=date.fromisoformat)
    parser.add_argument("--end", required=True, type=date.fromisoformat)
    parser.add_argument("--tickers", type=str, default=None, help="comma-separated; defaults to the domestic trade_allowlist")
    parser.add_argument(
        "--universe-axes", type=str, default=None,
        help="comma-separated KRX discovery axes to build the universe from LIVE: "
             "trading_value,market_cap,fluctuation_losers,disparity_low. Mutually exclusive with --tickers.",
    )
    parser.add_argument("--universe-axis-count", type=int, default=30)
    parser.add_argument("--universe-min-volume", type=int, default=0)
    parser.add_argument(
        "--universe-max-price", type=float, default=None,
        help="KIS-side upper price bound (KRW) for every discovery axis — the A/B lever for "
             "app.config.auto_trade_universe_price_ceiling_domestic. Omit for no ceiling (the control arm).",
    )
    parser.add_argument("--disparity-window-days", type=int, default=20)
    parser.add_argument(
        "--dump-universe", type=str, default=None,
        help="write the resolved ticker list to this JSON path, so two arms of a comparison can be diffed by name.",
    )
    parser.add_argument("--lookback-options", type=str, default=None, help="e.g. 5,10,15,20 — mutually exclusive with --variants-json")
    parser.add_argument("--variants-json", type=str, default=None, help="path to a JSON list of variant dicts")
    parser.add_argument("--top-n", type=int, default=settings.auto_trade_max_tickers_domestic)
    parser.add_argument("--holding-days", type=int, default=10)
    parser.add_argument("--cadence-days", type=int, default=5, help="sample every Nth trading day, not every day")
    parser.add_argument("--worth-bar", type=float, default=0.03)
    parser.add_argument("--bootstrap-iters", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--dump-json", type=str, default=None,
        help="write ONE machine-readable JSON dump (schema screening-sweep-dump-v1: all scores, forward returns "
             "with exit price dates, per-variant observations + signed scores) to this path for offline "
             "re-analysis. Off by default; stdout and results are identical when omitted.",
    )
    args = parser.parse_args()
    started_at = datetime.now(timezone.utc)

    variants = _variants_from_args(args)
    _check_universe_source_args(args)

    if args.universe_axes:
        universe = await _universe_from_live_axes(
            args.universe_axes.split(","),
            args.universe_axis_count,
            min_volume=args.universe_min_volume,
            disparity_window_days=args.disparity_window_days,
            max_price=args.universe_max_price,
        )
    else:
        if not args.tickers:
            from app.web import trade_allowlist_store

            await trade_allowlist_store.load()
        universe = args.tickers.split(",") if args.tickers else list(settings.trade_allowlist_domestic)
    if not universe:
        raise SystemExit(
            "no tickers to sweep — pass --tickers, --universe-axes, or configure a domestic trade_allowlist"
        )
    if args.dump_universe:
        with open(args.dump_universe, "w", encoding="utf-8") as fh:
            json.dump(universe, fh, ensure_ascii=False, indent=2)

    calendar_prices = await _calendar_price_series(universe[0], args.start, args.end)
    trade_dates = select_trading_dates(calendar_prices, args.cadence_days)
    if not trade_dates:
        raise SystemExit(f"no trading dates in [{args.start}, {args.end}] for {universe[0]}")

    print(
        f"Sweeping {len(variants)} variant(s) over {len(trade_dates)} trade dates "
        f"({args.start}..{args.end}, cadence={args.cadence_days}), universe={universe}, "
        f"top_n={args.top_n}, holding_days={args.holding_days}...",
        flush=True,
    )
    collector: _DumpCollector | None = None
    if args.dump_json:
        git_head, git_dirty = _git_info()
        needs_benchmark = any(v.momentum_mode != "total" for v in variants)
        collector = _DumpCollector(
            meta={
                "started_at": started_at,
                "git_head": git_head,
                "git_dirty": git_dirty,
                "argv": list(sys.argv),
                "args": vars(args),
                "universe": list(universe),
                "trade_dates_requested": list(trade_dates),
                "holding_days": args.holding_days,
                "cadence_days": args.cadence_days,
                "top_n": args.top_n,
                "benchmark": (
                    {"markets": sorted({ticker_market(t) for t in universe}), "used": True} if needs_benchmark else None
                ),
            }
        )
    try:
        results = await sweep(
            universe,
            trade_dates,
            variants,
            args.top_n,
            args.holding_days,
            args.cadence_days,
            worth_bar=args.worth_bar,
            bootstrap_iters=args.bootstrap_iters,
            seed=args.seed,
            collector=collector,
        )
    except BaseException:
        # Best-effort partial dump; a write failure must never mask the real error.
        if collector is not None and collector.completed_dates:
            try:
                _write_dump(args.dump_json, collector, status="incomplete")
                print(f"! sweep aborted — partial dump (status=incomplete) written to {args.dump_json}", file=sys.stderr, flush=True)
            except Exception as dump_exc:  # noqa: BLE001
                print(f"! could not write partial dump to {args.dump_json}: {dump_exc}", file=sys.stderr, flush=True)
        raise
    print(
        render(
            results,
            universe=universe,
            cadence_days=args.cadence_days,
            holding_days=args.holding_days,
            seed=args.seed,
            bootstrap_iters=args.bootstrap_iters,
        )
    )
    if collector is not None:
        try:
            _write_dump(args.dump_json, collector, status="complete")
        except Exception as dump_exc:  # noqa: BLE001
            # The sweep output above was already printed; fail loudly (non-zero) so the missing dump is noticed.
            raise SystemExit(f"--dump-json: failed to write {args.dump_json}: {dump_exc}") from dump_exc
        print(f"dump written: {args.dump_json}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    asyncio.run(_main())
