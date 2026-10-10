"""Read side of the §4-1 live screening instrumentation (2026-09-08): joins
`app.live.screening_log`'s per-cycle snapshots against realized forward
returns and reports whether the LIVE screening ranking actually produces
alpha — the question project memory screening-analysis-2026-09-08 found
nothing in this codebase could answer, since nothing recorded the whole
scored cross-section each cycle before now.

Two responsibilities live here:
  1. `resolve_due_forward_returns` — fills in the `forward_returns` table
     this module owns, called once per live cycle (app.live.auto_trade_loop,
     same idempotent-call convention as `app.memory.auto_resolve.
     resolve_all_due`; this project has no OS-level cron, see
     app.daemon's own docstring).
  2. `compute_screening_ic`/`render_screening_ic` — the report, driven from
     `python -m app.audit screening-ic` (a human runs this; it is
     deliberately NOT wired into any automatic cycle, same posture as
     app.audit.weekly).

Reuses `app.backtest.screening_sweep._forward_return` for the actual price
lookup (KIS-first, yfinance fallback, same calendar-day holding convention
as the offline sweep) rather than reimplementing it — `app.audit.signal_ic`
already established this app.audit -> app.backtest.screening_sweep coupling
for the same reason (one canonical forward-return definition, not two that
can quietly drift apart).

Domestic and overseas are NEVER pooled — same project-wide rule
app.live.auto_trade_loop's own module docstring states for every other
domestic/overseas metric here, since the two markets run different
lookback/volatility-adjustment/dedup settings (screening_volatility_
adjustment_overseas=False, screening_correlation_dedup_overseas=False)."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import date, timedelta

from app.backtest import factor_metrics
from app.backtest.screening_sweep import _forward_return
from app.clock import Market, get_clock
from app.config import settings
from app.memory.db import get_pool

logger = logging.getLogger("app.audit.screening_ic")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS forward_returns (
    ticker         TEXT NOT NULL,
    trade_date     DATE NOT NULL,
    holding_days   INTEGER NOT NULL,
    -- NULL with a non-null resolved_at means "we measured and there was a
    -- data gap" (delisted, too-recent listing, no trading day at or after
    -- the target end) — deliberately distinct from "no row yet", so a
    -- permanently-unmeasurable pair is not refetched from KIS forever.
    -- `attempts` bounds the retry for the TRANSIENT case (a KIS hiccup),
    -- otherwise indistinguishable from the permanent one.
    forward_return DOUBLE PRECISION,
    attempts       SMALLINT NOT NULL DEFAULT 1,
    resolved_at    TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (ticker, trade_date, holding_days)
);
"""

_schema_ready = False
_schema_lock = asyncio.Lock()


async def _ready_pool():
    global _schema_ready
    pool = await get_pool()
    if _schema_ready:
        return pool
    async with _schema_lock:
        if not _schema_ready:
            async with pool.acquire() as conn:
                await conn.execute(_SCHEMA)
            _schema_ready = True
    return pool


async def resolve_due_forward_returns(holding_days: int | None = None) -> int:
    """Fetches and stores the forward return for every (ticker, trade_date)
    a screening snapshot recorded that's old enough to measure and not yet
    resolved (or resolved with a gap and under the retry cap). Best-effort:
    never raises, logs a warning and returns 0 on any failure — this must
    never block a live cycle.

    Returns the number of (ticker, trade_date) pairs resolved this pass."""
    if not settings.screening_snapshot_log_enabled:
        return 0
    holding = holding_days if holding_days is not None else settings.screening_forward_holding_days
    max_attempts = settings.screening_forward_resolve_max_attempts
    max_per_pass = settings.screening_forward_resolve_max_per_pass
    concurrency = settings.screening_forward_resolve_concurrency
    # Only dates whose holding window (plus weekend/holiday padding) has
    # fully elapsed have a real close price to measure against — see
    # _forward_return's own `holding_days + 5` padding.
    cutoff = get_clock().today() - timedelta(days=holding + 5)
    try:
        pool = await _ready_pool()
        async with pool.acquire() as conn:
            due = await conn.fetch(
                """
                SELECT DISTINCT s.ticker, s.trade_date
                FROM screening_snapshots s
                LEFT JOIN forward_returns f
                       ON f.ticker = s.ticker AND f.trade_date = s.trade_date AND f.holding_days = $1
                WHERE s.scored
                  AND s.trade_date <= $2
                  AND (f.ticker IS NULL OR (f.forward_return IS NULL AND f.attempts < $3))
                ORDER BY s.trade_date, s.ticker
                LIMIT $4
                """,
                holding, cutoff, max_attempts, max_per_pass,
            )
        if not due:
            return 0

        from app.dataflows.kis_provider import candle_run_cache

        semaphore = asyncio.Semaphore(concurrency)

        async def _resolve_one(ticker: str, trade_date: date) -> tuple[str, date, float | None]:
            async with semaphore:
                value = await _forward_return(ticker, trade_date, holding)
                return ticker, trade_date, value

        with candle_run_cache():
            resolved = await asyncio.gather(
                *(_resolve_one(row["ticker"], row["trade_date"]) for row in due),
                return_exceptions=True,
            )

        now = get_clock().now()
        rows = []
        for item in resolved:
            if isinstance(item, BaseException):
                if isinstance(item, asyncio.CancelledError):
                    raise item
                logger.warning("forward-return resolution failed for one ticker/date", exc_info=item)
                continue
            ticker, trade_date, value = item
            rows.append((ticker, trade_date, holding, value, now))

        if not rows:
            return 0

        async with pool.acquire() as conn:
            async with conn.transaction():
                for ticker, trade_date, holding_d, value, resolved_at in rows:
                    await conn.execute(
                        "INSERT INTO forward_returns (ticker, trade_date, holding_days, forward_return, attempts, resolved_at) "
                        "VALUES ($1, $2, $3, $4, 1, $5) "
                        "ON CONFLICT (ticker, trade_date, holding_days) DO UPDATE "
                        "SET forward_return = EXCLUDED.forward_return, "
                        "attempts = forward_returns.attempts + 1, "
                        "resolved_at = EXCLUDED.resolved_at",
                        ticker, trade_date, holding_d, value, resolved_at,
                    )
        return len(rows)
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.warning("forward-return resolution pass failed — will retry next cycle", exc_info=True)
        return 0


@dataclass(frozen=True)
class DateCrossSection:
    trade_date: date
    n_scored: int
    rank_ic: float | None
    excess_return: float | None
    quantile_spread: float | None
    turnover: float | None
    picks: list[str]
    # 2026-09-09 §0단계 계측: how many of this date's n_scored rows came from
    # the observe-only pass (score_observational) rather than the real
    # ranking. n_scored - n_observational is what the pre-§0단계 cross-
    # section size would have been — see ScreeningICReport.
    # mixed_observational_coverage's own docstring for why this matters.
    n_observational: int = 0


@dataclass(frozen=True)
class LockCohortStats:
    """Answers screening-cycle-rebalance-lock-analysis-2026-09-09's central
    question directly: does the rebalance_interval_days throttle actually
    cost the ranking anything? Splits every scored-or-observed ticker across
    all dates in the report into three cohorts:
      picked  — selected AND selection_reason='momentum' (the ranking's own
                picks, an execution-cost-bounded subset of `passed`)
      passed  — executable AND NOT selected (the ranking saw it, ranked it,
                just didn't make the top_n budget cut)
      locked  — rebalance_locked (the ranking never saw it at all this
                cycle — this is the OBSERVE-ONLY population)
    If the throttle is harmless, `locked`'s mean forward return should look
    statistically indistinguishable from `passed`'s, and few locked tickers
    should land inside that date's counterfactual top-`max_tickers` by
    score. If `mean_locked_in_counterfactual_top_n` is close to
    `max_tickers`, the ranking is starved — it would have picked mostly
    locked names had they been eligible, meaning the throttle (not a quiet
    market) is what's determining `final` most days. This is descriptive,
    not a significance test — treat n_dates the same way `n_eff` gates the
    RankIC verdict above (see _MIN_N_EFF_FOR_VERDICT)."""

    n_dates: int
    mean_fwd_picked: float | None
    mean_fwd_passed: float | None
    mean_fwd_locked: float | None
    n_fwd_picked: int
    n_fwd_passed: int
    n_fwd_locked: int
    mean_locked_in_counterfactual_top_n: float | None
    mean_locked_share_of_universe: float | None


@dataclass(frozen=True)
class ScreeningICReport:
    market: Market
    start: date
    end: date
    holding_days: int
    per_date: list[DateCrossSection] = field(default_factory=list)
    n_dates: int = 0
    mean_rank_ic: float | None = None
    rank_ic_ci95: tuple[float, float] | None = None
    icir: float | None = None
    mean_excess_return: float | None = None
    excess_return_ci95: tuple[float, float] | None = None
    mean_quantile_spread: float | None = None
    mean_turnover: float | None = None
    worth_rate: float | None = None
    upside_rate: float | None = None
    n_eff: float | None = None
    min_n_scored: int | None = None
    max_n_scored: int | None = None
    settings_variants: list[str] = field(default_factory=list)
    # 2026-09-09 §0단계 계측.
    lock_cohorts: LockCohortStats | None = None
    # 2026-09-18 체류도 분해 — see AxisCohortStats' own docstring. Empty list
    # when the range has no screening_snapshots rows with usable axes data.
    axis_cohorts: list[AxisCohortStats] = field(default_factory=list)
    # TRUE when this date range straddles the deployment of §0단계 (some
    # dates have n_observational=0 because the feature was off/not yet
    # deployed, others have n_observational>0). Pooling RankIC/excess-return
    # across that boundary averages two DIFFERENT cross-section definitions
    # (executable-only vs whole-universe) as if they were one factor —
    # narrow the date range instead of trusting the averages above when
    # this is True.
    mixed_observational_coverage: bool = False


async def compute_screening_ic(
    market: Market,
    start: date,
    end: date,
    *,
    holding_days: int | None = None,
    user_id: str | None = None,
    account_mode: str | None = None,
) -> ScreeningICReport:
    """Per-date RankIC/excess-return/quantile-spread/turnover over the
    FIRST screening cycle of each trading day in [start, end] (the later
    same-day cycles have an artificially narrowed universe —
    `run_cycle_now`'s `exclude_already_analyzed_today=True` — which would
    bias the cross-section if pooled in), then the same roll-ups
    `app.backtest.screening_sweep._summarise_variant` computes offline, so a
    live number is directly comparable to the sweep tables that produced
    today's ranking parameters."""
    from app.config import current_account_mode
    from app.user_context import get_current_user_id

    holding = holding_days if holding_days is not None else settings.screening_forward_holding_days
    uid = user_id or get_current_user_id()
    mode = account_mode or current_account_mode()

    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            WITH first_cycle AS (
                SELECT DISTINCT ON (trade_date) trade_date, captured_at
                FROM screening_snapshots
                WHERE user_id = $1 AND account_mode = $2 AND market = $3
                  AND trade_date BETWEEN $4 AND $5
                ORDER BY trade_date, captured_at
            )
            SELECT s.trade_date, s.ticker, s.score, s.selected, s.selection_reason,
                   s.lookback_days, s.volatility_adjusted, s.correlation_dedup,
                   s.executable, s.score_observational, s.rebalance_locked, s.max_tickers,
                   f.forward_return
            FROM screening_snapshots s
            JOIN first_cycle c ON c.trade_date = s.trade_date AND c.captured_at = s.captured_at
            LEFT JOIN forward_returns f
                   ON f.ticker = s.ticker AND f.trade_date = s.trade_date AND f.holding_days = $6
            WHERE s.user_id = $1 AND s.account_mode = $2 AND s.market = $3 AND s.scored
            ORDER BY s.trade_date, s.ticker
            """,
            uid, mode, market, start, end, holding,
        )
        # 2026-09-18 체류도 분해 needs EVERY capture of every day (not just
        # the first cycle the query above scopes to — first-cycle-only would
        # undercount a ticker's daily appearance rate and miss axes it only
        # entered on in a later same-day cycle), so this is a second,
        # separately-scoped fetch rather than a reuse of `rows` above.
        axis_rows = await conn.fetch(
            """
            SELECT s.trade_date, s.ticker, s.captured_at, s.axes, s.score,
                   f.forward_return
            FROM screening_snapshots s
            LEFT JOIN forward_returns f
                   ON f.ticker = s.ticker AND f.trade_date = s.trade_date AND f.holding_days = $6
            WHERE s.user_id = $1 AND s.account_mode = $2 AND s.market = $3 AND s.scored
              AND s.trade_date BETWEEN $4 AND $5
            ORDER BY s.trade_date, s.ticker, s.captured_at
            """,
            uid, mode, market, start, end, holding,
        )

    by_date: dict[date, list] = {}
    for row in rows:
        by_date.setdefault(row["trade_date"], []).append(row)

    variants_seen: set[str] = set()
    per_date: list[DateCrossSection] = []
    prev_picks: list[str] | None = None
    for trade_date in sorted(by_date):
        day_rows = by_date[trade_date]
        scores: dict[str, float | None] = {}
        forward: dict[str, float | None] = {}
        picks: list[str] = []
        for row in day_rows:
            scores[row["ticker"]] = row["score"]
            forward[row["ticker"]] = row["forward_return"]
            variants_seen.add(
                f"lb{row['lookback_days']}_vol{row['volatility_adjusted']}_dedup{row['correlation_dedup']}"
            )
            if row["selected"] and row["selection_reason"] == "momentum":
                picks.append(row["ticker"])

        pick_fwd = [forward[t] for t in picks if forward.get(t) is not None]
        universe_fwd = [r for r in forward.values() if r is not None]
        n_observational = sum(1 for row in day_rows if row["score_observational"])

        per_date.append(DateCrossSection(
            trade_date=trade_date,
            n_scored=len(day_rows),
            rank_ic=factor_metrics.rank_ic(scores, forward),
            excess_return=factor_metrics.excess_return(pick_fwd, universe_fwd),
            quantile_spread=factor_metrics.quantile_spread(scores, forward),
            turnover=factor_metrics.turnover(prev_picks, picks) if prev_picks is not None else None,
            picks=picks,
            n_observational=n_observational,
        ))
        prev_picks = picks

    lock_cohorts = _compute_lock_cohorts(by_date)
    axis_cohorts = _compute_axis_cohorts(axis_rows, holding)
    mixed_observational_coverage = (
        any(d.n_observational > 0 for d in per_date) and any(d.n_observational == 0 for d in per_date)
    )

    rank_ics = [d.rank_ic for d in per_date]
    excesses = [d.excess_return for d in per_date if d.excess_return is not None]
    spreads = [d.quantile_spread for d in per_date if d.quantile_spread is not None]
    turnovers = [d.turnover for d in per_date if d.turnover is not None]

    pooled_picks: list[float] = []
    excess_blocks: list[list[float]] = []
    for d in per_date:
        rows_d = by_date[d.trade_date]
        universe_fwd = [row["forward_return"] for row in rows_d if row["forward_return"] is not None]
        pick_fwd = [row["forward_return"] for row in rows_d if row["ticker"] in d.picks and row["forward_return"] is not None]
        pooled_picks.extend(pick_fwd)
        if not pick_fwd or not universe_fwd:
            continue
        uni_mean = sum(universe_fwd) / len(universe_fwd)
        excess_blocks.append([r - uni_mean for r in pick_fwd])
    ic_blocks = [[ic] for ic in rank_ics if ic is not None]

    n_scored_list = [d.n_scored for d in per_date]

    return ScreeningICReport(
        market=market,
        start=start,
        end=end,
        holding_days=holding,
        per_date=per_date,
        n_dates=len(per_date),
        mean_rank_ic=(sum(x for x in rank_ics if x is not None) / len([x for x in rank_ics if x is not None])
                      if any(x is not None for x in rank_ics) else None),
        rank_ic_ci95=factor_metrics.block_bootstrap_ci(ic_blocks, seed=0),
        icir=factor_metrics.icir(rank_ics),
        mean_excess_return=(sum(excesses) / len(excesses)) if excesses else None,
        excess_return_ci95=factor_metrics.block_bootstrap_ci(excess_blocks, seed=0),
        mean_quantile_spread=(sum(spreads) / len(spreads)) if spreads else None,
        mean_turnover=(sum(turnovers) / len(turnovers)) if turnovers else None,
        worth_rate=factor_metrics.worth_analyzing_rate(pooled_picks),
        upside_rate=factor_metrics.upside_hit_rate(pooled_picks),
        # cadence_days=1: a live snapshot exists once per trading day (first
        # cycle only, per the query above), unlike the offline sweep's
        # sampled cadence — see this module's own docstring.
        n_eff=factor_metrics.effective_sample_size(len(per_date), cadence_days=1, holding_days=holding),
        min_n_scored=min(n_scored_list) if n_scored_list else None,
        max_n_scored=max(n_scored_list) if n_scored_list else None,
        settings_variants=sorted(variants_seen),
        lock_cohorts=lock_cohorts,
        axis_cohorts=axis_cohorts,
        mixed_observational_coverage=mixed_observational_coverage,
    )


def _compute_lock_cohorts(by_date: dict[date, list]) -> LockCohortStats | None:
    """Pure — see `LockCohortStats`'s own docstring for what this answers.
    Returns None when nothing in the range has a rebalance_locked=true row
    at all (e.g. a range entirely before §0단계 계측 was deployed, where
    every row's rebalance_locked default is false) — there's nothing to
    compare against in that case, so the report should render "-" rather
    than a misleadingly-empty-but-present stat block."""
    picked_fwd: list[float] = []
    passed_fwd: list[float] = []
    locked_fwd: list[float] = []
    locked_in_top_n_fractions: list[float] = []
    locked_share_fractions: list[float] = []
    any_locked = False

    for trade_date, rows in by_date.items():
        n_locked_today = sum(1 for row in rows if row["rebalance_locked"])
        if n_locked_today:
            any_locked = True
        # Appended on EVERY date (not just ones with a lock) — otherwise this
        # average is conditioned on "days with at least one lock" while
        # mean_locked_in_counterfactual_top_n below is not, and the two
        # numbers silently answer different questions over the same range
        # (2026-09-10 fix — see this function's own review notes).
        locked_share_fractions.append(n_locked_today / len(rows))

        max_tickers_today = next((row["max_tickers"] for row in rows if row["max_tickers"] is not None), None)
        scored_rows = [row for row in rows if row["score"] is not None]
        if max_tickers_today and scored_rows:
            top_n_today = sorted(scored_rows, key=lambda row: row["score"], reverse=True)[:max_tickers_today]
            locked_in_top_n = sum(1 for row in top_n_today if row["rebalance_locked"])
            locked_in_top_n_fractions.append(locked_in_top_n)

        for row in rows:
            fwd = row["forward_return"]
            if fwd is None:
                continue
            if row["rebalance_locked"]:
                locked_fwd.append(fwd)
            elif row["selected"] and row["selection_reason"] == "momentum":
                picked_fwd.append(fwd)
            elif row["executable"] and not row["selected"]:
                passed_fwd.append(fwd)

    if not any_locked:
        return None

    def _mean(xs: list[float]) -> float | None:
        return sum(xs) / len(xs) if xs else None

    return LockCohortStats(
        n_dates=len(by_date),
        mean_fwd_picked=_mean(picked_fwd),
        mean_fwd_passed=_mean(passed_fwd),
        mean_fwd_locked=_mean(locked_fwd),
        n_fwd_picked=len(picked_fwd),
        n_fwd_passed=len(passed_fwd),
        n_fwd_locked=len(locked_fwd),
        mean_locked_in_counterfactual_top_n=_mean(locked_in_top_n_fractions),
        mean_locked_share_of_universe=_mean(locked_share_fractions),
    )


@dataclass(frozen=True)
class AxisCohortStats:
    """2026-09-18 체류도(persistence) 분해 — edge-measurement-2026-09-17이 확정한
    "작업 계층(스크리닝 순위)에 예측력이 없다"는 결과를 어느 발굴 축(axis)이
    끌어내리는지, 그리고 그 축의 예측력이 스쳐가는 노출(transient)에서 오는지
    반복적으로 남는 노출(persistent)에서 오는지 갈라서 본다.

    관측 단위는 (ticker, trade_date) — 같은 종목의 하루 여러 캡처를 하나로
    접는다(캡처 단위로 세면 같은 종목이 하루 20~30번 계상된다). 각 ticker-day의
    axes는 그날 그 종목의 모든 캡처 axes 합집합이다.

    두 가지 코호트 모드를 모두 낸다:
      marginal  — axes 집합이 이 axis를 포함하는 모든 ticker-day (겹침 허용:
                  한 ticker-day가 여러 축의 marginal에 동시에 들어갈 수 있다)
      exclusive — axes 집합이 정확히 {이 axis} 하나뿐인 ticker-day

    그리고 등장률(그 종목이 등장한 캡처 수 / 그날 전체 캡처 수)로 두 버킷:
      transient   — 등장률 <= 0.2
      persistent  — 등장률 >  0.2

    기준선은 지수 벤치마크가 아니라 그 날짜에 스코어링된(scored=true) 전체
    ticker-day의 횡단면 평균 forward_return이다 — mean_excess는 각 ticker-day의
    forward_return에서 그 평균을 뺀 값의 평균. 모집단은 scored=true 전체이며
    selected로 거르지 않는다(축 품질과 하류 필터 효과가 섞이는 것을 피하기
    위해 — compute_screening_ic 본문 상단 쿼리와 다른 이유다).

    n_dates/n_ticker_days는 forward_return 해소 여부와 무관하게 이 코호트에
    속한 관측치 수(모집단 크기, 지금 당장도 유효) — n_resolved는 그중 실제로
    forward_return이 채워진 날짜 수(통계 검정력의 분모). forward_returns
    테이블은 2026-09-18 기준 0행이고 2026-09-23경부터 채워지기 시작하므로,
    한동안 n_resolved=0·mean_excess=None인 것이 정상이다 — 그 경우
    render_screening_ic는 "차이 없음"이 아니라 "데이터 없음"으로 찍어야 한다
    (n_resolved>0인데 n_eff가 _MIN_N_EFF_FOR_VERDICT 미만인 경우는 별도로
    "판정불가(underpowered)"로 찍는다 — 이 두 상태를 혼동하지 않는 것이
    이 프로젝트에서 오늘만 두 번 결론을 뒤집은 실수라 각별히 구분한다)."""

    axis: str
    mode: str  # "marginal" | "exclusive"
    bucket: str  # "transient" | "persistent"
    n_dates: int
    n_ticker_days: int
    n_resolved: int
    mean_excess: float | None
    excess_ci95: tuple[float, float] | None
    rank_ic: float | None
    n_eff: float | None


_APPEARANCE_RATE_TRANSIENT_MAX = 0.2


def _compute_axis_cohorts(rows: list, holding_days: int) -> list[AxisCohortStats]:
    """Pure — mirrors `_compute_lock_cohorts`'s pattern. `rows` is the raw,
    UN-first-cycle-filtered per-capture query result (every screening cycle
    of every day in range, not just the first), each row needing
    trade_date/ticker/captured_at/axes/score/forward_return. Uses `.get()`
    (not `row[...]`) so a caller/test fixture missing captured_at/axes
    degrades to "no axis data" instead of raising — deliberately, since
    `compute_screening_ic`'s pre-existing tests build fixtures without those
    two columns and must keep passing unmodified (regression constraint)."""
    by_date: dict[date, list] = {}
    for row in rows:
        trade_date = row.get("trade_date")
        if trade_date is None:
            continue
        by_date.setdefault(trade_date, []).append(row)

    # Fold every capture down to one record per (trade_date, ticker).
    folded_by_date: dict[date, dict[str, dict]] = {}
    for trade_date, day_rows in by_date.items():
        n_captures = len({row.get("captured_at") for row in day_rows})
        tickers: dict[str, dict] = {}
        for row in day_rows:
            ticker = row.get("ticker")
            if ticker is None:
                continue
            entry = tickers.setdefault(ticker, {
                "axes": set(), "scores": [], "forward_return": None, "n_appear": 0,
            })
            entry["axes"].update(row.get("axes") or [])
            score = row.get("score")
            if score is not None:
                entry["scores"].append(score)
            fwd = row.get("forward_return")
            if fwd is not None:
                entry["forward_return"] = fwd
            entry["n_appear"] += 1
        for entry in tickers.values():
            entry["appearance_rate"] = entry["n_appear"] / n_captures if n_captures else 0.0
            entry["mean_score"] = sum(entry["scores"]) / len(entry["scores"]) if entry["scores"] else None
        folded_by_date[trade_date] = tickers

    # Cross-sectional baseline per date: mean forward_return over every
    # scored ticker-day that date with a resolved return (NOT an index
    # benchmark — see AxisCohortStats' own docstring).
    baseline_by_date: dict[date, float | None] = {}
    for trade_date, tickers in folded_by_date.items():
        fwds = [t["forward_return"] for t in tickers.values() if t["forward_return"] is not None]
        baseline_by_date[trade_date] = (sum(fwds) / len(fwds)) if fwds else None

    all_axes: set[str] = set()
    for tickers in folded_by_date.values():
        for entry in tickers.values():
            all_axes.update(entry["axes"])

    out: list[AxisCohortStats] = []
    for axis in sorted(all_axes):
        for mode in ("marginal", "exclusive"):
            for bucket in ("persistent", "transient"):
                dates_seen: set[date] = set()
                n_ticker_days = 0
                resolved_dates: set[date] = set()
                excess_blocks: list[list[float]] = []
                ic_per_date: list[float] = []

                for trade_date, tickers in folded_by_date.items():
                    baseline = baseline_by_date[trade_date]
                    block_excess: list[float] = []
                    day_scores: dict[str, float] = {}
                    day_fwds: dict[str, float] = {}
                    matched_any = False
                    for ticker, entry in tickers.items():
                        axes = entry["axes"]
                        in_cohort = (axis in axes) if mode == "marginal" else (axes == {axis})
                        if not in_cohort:
                            continue
                        entry_bucket = (
                            "transient" if entry["appearance_rate"] <= _APPEARANCE_RATE_TRANSIENT_MAX
                            else "persistent"
                        )
                        if entry_bucket != bucket:
                            continue
                        matched_any = True
                        n_ticker_days += 1
                        fwd = entry["forward_return"]
                        if fwd is not None and baseline is not None:
                            block_excess.append(fwd - baseline)
                        if entry["mean_score"] is not None and fwd is not None:
                            day_scores[ticker] = entry["mean_score"]
                            day_fwds[ticker] = fwd
                    if matched_any:
                        dates_seen.add(trade_date)
                    if block_excess:
                        resolved_dates.add(trade_date)
                        excess_blocks.append(block_excess)
                    date_ic = factor_metrics.rank_ic(day_scores, day_fwds) if day_scores else None
                    if date_ic is not None:
                        ic_per_date.append(date_ic)

                if n_ticker_days == 0:
                    continue  # this (axis, mode, bucket) combination never occurred in range

                pooled_excess = [v for block in excess_blocks for v in block]
                mean_excess = (sum(pooled_excess) / len(pooled_excess)) if pooled_excess else None
                excess_ci95 = factor_metrics.block_bootstrap_ci(excess_blocks, seed=0) if excess_blocks else None
                mean_ic = (sum(ic_per_date) / len(ic_per_date)) if ic_per_date else None
                n_eff = factor_metrics.effective_sample_size(
                    len(resolved_dates), cadence_days=1, holding_days=holding_days
                )

                out.append(AxisCohortStats(
                    axis=axis, mode=mode, bucket=bucket,
                    n_dates=len(dates_seen), n_ticker_days=n_ticker_days,
                    n_resolved=len(resolved_dates),
                    mean_excess=mean_excess, excess_ci95=excess_ci95,
                    rank_ic=mean_ic, n_eff=n_eff,
                ))
    return out


_MIN_N_EFF_FOR_VERDICT = 20


def render_screening_ic(report: ScreeningICReport) -> str:
    lines = [
        f"market={report.market} {report.start}..{report.end} holding={report.holding_days}d "
        f"dates={report.n_dates}",
    ]
    if report.min_n_scored is not None and report.max_n_scored is not None:
        lines.append(f"cross-section size: min={report.min_n_scored} max={report.max_n_scored}")
    lines.append("")

    if len(report.settings_variants) > 1:
        lines.append(
            "! WARNING: more than one (lookback, volatility_adjusted, correlation_dedup) combination "
            f"appears in this range: {', '.join(report.settings_variants)} — the settings changed "
            "mid-period, so the averages below pool two different factors as if they were one."
        )
        lines.append("")

    lines.append(
        f"n_eff={_fmt_num(report.n_eff, 1)}  mean_RankIC={_fmt_num(report.mean_rank_ic)}  "
        f"RankIC_CI95={_fmt_ci(report.rank_ic_ci95, pct=False)}  ICIR={_fmt_num(report.icir, 2)}"
    )
    lines.append(
        f"mean_excess_return={_fmt_pct(report.mean_excess_return)}  "
        f"excess_CI95={_fmt_ci(report.excess_return_ci95)}  "
        f"mean_quantile_spread={_fmt_pct(report.mean_quantile_spread)}  "
        f"mean_turnover={_fmt_num(report.mean_turnover, 2)}"
    )
    lines.append(
        f"worth_analyzing_rate={_fmt_pct(report.worth_rate, digits=0)}  "
        f"upside_hit_rate={_fmt_pct(report.upside_rate, digits=0)}"
    )
    lines.append("")

    if report.lock_cohorts is not None:
        lc = report.lock_cohorts
        lines.append(
            "rebalance_interval_days lock cohorts (screening-cycle-rebalance-lock-analysis-2026-09-09):"
        )
        lines.append(
            f"  mean_fwd_return: picked={_fmt_pct(lc.mean_fwd_picked)} (n={lc.n_fwd_picked})  "
            f"passed={_fmt_pct(lc.mean_fwd_passed)} (n={lc.n_fwd_passed})  "
            f"locked={_fmt_pct(lc.mean_fwd_locked)} (n={lc.n_fwd_locked})"
        )
        lines.append(
            f"  mean_locked_share_of_universe={_fmt_pct(lc.mean_locked_share_of_universe, digits=0)}  "
            f"mean_locked_in_counterfactual_top_n={_fmt_num(lc.mean_locked_in_counterfactual_top_n, 2)}"
        )
        if lc.n_dates < _MIN_N_EFF_FOR_VERDICT:
            lines.append(
                f"  ! n_dates={lc.n_dates} < {_MIN_N_EFF_FOR_VERDICT} — underpowered FOR THE THREE "
                "mean_fwd_return VALUES ABOVE ONLY (they need forward_return + statistical power). "
                "mean_locked_share_of_universe / mean_locked_in_counterfactual_top_n are a census over "
                "actual rows, not a sample — they're exact for however many dates are here, though a "
                "single date (or few) may not be REPRESENTATIVE of the full rebalance_interval_days "
                "cycle (a sawtooth pattern — see screening-cycle-rebalance-lock-analysis-2026-09-09). "
                "Do not act on the mean_fwd_return comparison yet; the other two are still usable now."
            )
        lines.append("")
    elif report.n_dates == 0:
        lines.append(
            "rebalance_interval_days lock cohorts: - (no screening_snapshots rows at all in this range/"
            "scope — check --account-mode (real vs paper) and the date range before concluding "
            "\"nothing was locked\")"
        )
        lines.append("")
    else:
        lines.append("rebalance_interval_days lock cohorts: - (rows exist but none are rebalance_locked "
                     "in this range — either §0단계 계측 was not yet deployed for these dates, or "
                     "nothing was actually locked)")
        lines.append("")

    lines.append(
        "axis / persistence cohorts (2026-09-18 체류도 분해, 관측단위=ticker-day, "
        "marginal=축포함 exclusive=그축만단독, transient=등장률<=20% persistent=등장률>20%):"
    )
    if not report.axis_cohorts:
        lines.append(
            "  - (이 범위에 axes 데이터가 있는 screening_snapshots 행이 없음 — 범위/스코프 확인)"
        )
    else:
        if not any(c.n_resolved > 0 for c in report.axis_cohorts):
            lines.append(
                "  ! forward_returns 테이블에 이 범위 데이터가 아직 없음(만기 도달 전 — "
                "2026-09-23경부터 채워짐). 아래 n_dates/n_ticker_days(모집단 크기)는 지금도 유효하나 "
                "mean_excess/RankIC가 전부 '데이터없음'인 것이 정상 — '차이 없음'으로 읽지 말 것."
            )
        # Axis column is at least 18 wide but grows to fit the longest axis name plus a 2-space gap:
        # 'fluctuation_losers' is exactly 18 chars, which used to butt up against the mode column.
        axis_w = max(18, max(len(c.axis) for c in report.axis_cohorts) + 2)
        lines.append(
            f"  {'axis':<{axis_w}}{'mode':<11}{'bucket':<12}{'n_dates':>8}{'n_tdays':>9}  "
            "mean_excess / CI95 / RankIC  [verdict]"
        )
        for c in report.axis_cohorts:
            stats = (
                f"mean_excess={_fmt_pct(c.mean_excess)}  CI95={_fmt_ci(c.excess_ci95)}  "
                f"RankIC={_fmt_num(c.rank_ic)}"
            )
            if c.n_resolved == 0:
                verdict = "데이터없음(forward_return 미해결)"
            elif c.n_eff is None or c.n_eff < _MIN_N_EFF_FOR_VERDICT:
                verdict = f"판정불가(underpowered, n_eff={_fmt_num(c.n_eff, 1)}<{_MIN_N_EFF_FOR_VERDICT})"
            else:
                verdict = "판정가능"
            lines.append(
                f"  {c.axis:<{axis_w}}{c.mode:<11}{c.bucket:<12}{c.n_dates:>8}{c.n_ticker_days:>9}  "
                f"{stats}  [{verdict}]"
            )
    lines.append("")

    for warning in _hard_rule_warnings(report):
        lines.append(f"!  {warning}")
    lines.append("")

    lines.append(factor_metrics.INTERPRETATION_GUIDE)
    return "\n".join(lines)


def _hard_rule_warnings(report: ScreeningICReport) -> list[str]:
    out: list[str] = []
    if report.mixed_observational_coverage:
        out.append(
            "this range mixes dates with and without §0단계 관측용 스코어링 (screening_snapshot_"
            "observational_scoring_enabled) — the cross-section definition changed mid-range "
            "(executable-only vs whole-universe), so the RankIC/excess-return averages above pool "
            "two different factors as if they were one. Narrow the date range instead of trusting them."
        )
    if report.excess_return_ci95 is not None and report.excess_return_ci95[0] <= 0 <= report.excess_return_ci95[1]:
        out.append("excess-return 95% CI crosses 0 — unproven (screening picks did not reliably beat the universe)")
    if report.n_eff is not None and report.n_eff < _MIN_N_EFF_FOR_VERDICT:
        out.append(
            f"n_eff={report.n_eff:.1f} < {_MIN_N_EFF_FOR_VERDICT} — underpowered. Do NOT change any "
            "screening parameter off this number alone; keep collecting."
        )
    if (
        report.mean_rank_ic is not None
        and report.mean_excess_return is not None
        and report.mean_rank_ic != 0
        and report.mean_excess_return != 0
        and (report.mean_rank_ic > 0) != (report.mean_excess_return > 0)
    ):
        out.append("RankIC sign disagrees with excess-return sign — internally inconsistent, treat as noise")
    return out


def _fmt_num(value: float | None, digits: int = 3) -> str:
    return f"{value:.{digits}f}" if value is not None else "-"


def _fmt_pct(value: float | None, digits: int = 2) -> str:
    return f"{value:+.{digits}%}" if value is not None else "-"


def _fmt_ci(ci: tuple[float, float] | None, pct: bool = True) -> str:
    if ci is None:
        return "-"
    if pct:
        return f"[{ci[0]:+.2%}, {ci[1]:+.2%}]"
    return f"[{ci[0]:+.3f}, {ci[1]:+.3f}]"
