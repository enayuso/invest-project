"""Persists every live screening cycle's WHOLE scored cross-section (not
just the final picks) so a later batch can join it against realized forward
returns and finally measure whether `app.chat.screening`'s momentum ranking
produces any alpha live.

The ranking's direction/parameters (momentum sign=+1, 15-day lookback, ATR
normalization, correlation dedup) have already been validated repeatedly by
offline sweeps (`app.backtest.screening_sweep`, `app.backtest.factor_metrics`)
— see project memory screening-analysis-2026-09-08. What was never measured
is whether the LIVE cycle's picks actually beat the names it didn't pick:
nothing recorded which tickers were scored/ranked each cycle, so there was
nothing to join against N-days-later returns. That is the sole reason this
module exists.

Recording every scored ticker (not just the top_n cut) matters because
RankIC is defined over the WHOLE cross-section — truncating to top_n
collapses score variance and shrinks IC toward zero (see
`app.backtest.factor_metrics.rank_ic`'s own docstring).

This is research/observability logging, not a trading path — same posture
as `app.live.orderbook_log`/`app.memory.store`: a DB hiccup here must never
affect candidate selection, so every entry point degrades quietly (a
warning log and an early return) rather than raising.

랭킹 알고리즘 자체(방향/파라미터)는 이미 여러 차례 오프라인 스윕으로 검증됐지만, 라이브에서
실제로 알파가 나오는지는 전혀 계측되지 않았음 — 이 모듈은 그 계측의 쓰기 쪽.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import date, datetime

from app.chat.screening import MomentumScore
from app.clock import Market
from app.config import current_account_mode, settings
from app.memory.db import get_pool
from app.memory.store import _current_git_commit
from app.user_context import get_current_user_id

logger = logging.getLogger("app.live.screening_log")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS screening_snapshots (
    id                  BIGSERIAL PRIMARY KEY,
    -- One value shared by every row of one (market, cycle) batch. Computed
    -- APP-SIDE (get_clock().now(), passed as a parameter) — NOT SQL now() —
    -- because asyncpg does not auto-wrap a loop of execute() calls in one
    -- transaction, so SQL now() could differ per row and break the (market,
    -- captured_at) cycle grouping this table's whole aggregation depends on.
    captured_at         TIMESTAMPTZ NOT NULL,
    -- The market's OWN calendar date (local_today(market)), not a shared KST
    -- date for both — same reasoning as app.live.auto_trade_loop's
    -- today_domestic/today_overseas split.
    trade_date          DATE NOT NULL,
    market              TEXT NOT NULL CHECK (market IN ('domestic', 'overseas')),
    ticker              TEXT NOT NULL,
    -- The score the ranking ACTUALLY sorted by this cycle: MomentumScore.
    -- score when volatility-adjusted, MomentumScore.raw_pct otherwise. NULL
    -- when the ticker was never scored at all this cycle (its price/ATR
    -- fetch failed). Since 2026-09-09 (§0단계 계측) this is populated even
    -- for a ticker the ranking never saw (rebalance_interval_days-locked,
    -- max_open-capped, or unexecutable) when
    -- screening_snapshot_observational_scoring_enabled is on — see
    -- score_observational below for how to tell the two cases apart.
    score               DOUBLE PRECISION,
    -- Stored separately from `score` so a later analysis can recompute a
    -- DIFFERENT variant's score off the same snapshot (what
    -- app.backtest.screening_sweep._variant_score does offline) without a
    -- point-in-time KIS refetch for a past date.
    raw_pct             DOUBLE PRECISION,
    atr_pct             DOUBLE PRECISION,
    -- 1-based rank by `score` DESC within this cycle's WHOLE scored-or-
    -- observed set — NOT the top_n cut, NOT post-dedup. Named score_rank,
    -- not rank: `rank` is a Postgres window-function name and reads
    -- ambiguously otherwise. NULL when score IS NULL. Since 2026-09-09 this
    -- ranks across scored+observed together (see score_observational) —
    -- the RANKING ITSELF never used the observed half, this column is a
    -- read-side convenience for the counterfactual question "where would
    -- a locked ticker have landed".
    score_rank          INTEGER,
    -- Funnel flags mirroring _select_market_candidates' own stages.
    eligible            BOOLEAN NOT NULL,
    executable          BOOLEAN NOT NULL,
    -- TRUE if this ticker had a MomentumScore this cycle, from EITHER the
    -- real ranking pass (`scored`) OR the observe-only pass (`observed`) —
    -- see score_observational to tell which. app.audit.screening_ic's
    -- cross-section queries filter on this column, and the whole point of
    -- §0단계 계측 is to widen what they see to the full universe.
    scored              BOOLEAN NOT NULL,
    in_ranked_pool      BOOLEAN NOT NULL,
    selected            BOOLEAN NOT NULL,
    -- Why a selected row was selected. 'momentum' = the ranking picked it.
    -- 'held_review'/'cash_freed' = the forced risk-review paths, which are
    -- exempt from the ranking, the eligibility narrowing, AND the
    -- max_tickers budget — these MUST be excluded from any RankIC/excess-
    -- return computation, since they are not screening picks at all. NULL
    -- when selected = false.
    selection_reason    TEXT CHECK (selection_reason IS NULL OR selection_reason IN ('momentum', 'held_review', 'cash_freed')),
    -- Every discovery axis that admitted this ticker into the universe this
    -- cycle — a ticker routinely enters on several at once (turnover/
    -- market-cap are near-collinear on KRX). Empty array is possible for a
    -- forced-review ticker the `exclude` set had removed from the universe.
    axes                TEXT[] NOT NULL DEFAULT '{}',
    -- The ranking configuration this snapshot was produced under, stored
    -- per row (not assumed from current settings at read time) because
    -- these settings have already changed more than once — pooling RankIC
    -- across a settings change would measure two different factors as one.
    lookback_days       INTEGER NOT NULL,
    volatility_adjusted BOOLEAN NOT NULL,
    correlation_dedup   BOOLEAN NOT NULL,
    top_n_requested     INTEGER NOT NULL,
    max_tickers         INTEGER NOT NULL,
    -- Same tagging axes every other live/memory table carries. account_mode
    -- matters here specifically because _drop_unexecutable and
    -- _domestic_price_ceiling both read the REAL account, so a paper-mode
    -- and a real-mode cycle can produce different universes on the same day.
    user_id             TEXT NOT NULL DEFAULT 'default',
    account_mode        TEXT NOT NULL DEFAULT 'paper' CHECK (account_mode IN ('paper', 'real')),
    git_commit          TEXT
);

-- 2026-09-09 §0단계 계측(screening-cycle-rebalance-lock-analysis-2026-09-09).
-- TRUE = this row's score came from the observe-only pass
-- (app.chat.screening.ScreeningPass.observed), never from the real ranking
-- pass (`scored`) — the ticker was locked/capped/unexecutable this cycle
-- and the ranking never saw its score. A row this is TRUE for had ZERO
-- effect on any trading decision. DEFAULT FALSE so every pre-2026-09-09
-- row (where locked tickers simply had no score at all) keeps its old,
-- correct meaning.
ALTER TABLE screening_snapshots ADD COLUMN IF NOT EXISTS score_observational BOOLEAN NOT NULL DEFAULT FALSE;
-- Splits `eligible = false` into its two distinct causes, which the
-- `eligible` flag alone cannot distinguish: this ticker was inside
-- rebalance_interval_days of its last decision (app.live.auto_trade_loop.
-- _partition_analyzed_within), as opposed to dropped by the max_open cap
-- (_drop_new_positions_over_cap) or unexecutable. Needed to compare "the
-- 5-day-locked cohort" against "the cohort the ranking actually picked
-- from" without an already-held/capped ticker (different risk shape)
-- contaminating the locked side of that comparison.
ALTER TABLE screening_snapshots ADD COLUMN IF NOT EXISTS rebalance_locked BOOLEAN NOT NULL DEFAULT FALSE;
ALTER TABLE screening_snapshots ADD COLUMN IF NOT EXISTS rebalance_unlock_date DATE;

-- Idempotency guard: a retried write of the same batch collides here instead
-- of double-counting a date's cross-section.
CREATE UNIQUE INDEX IF NOT EXISTS screening_snapshots_cycle_ticker_idx
    ON screening_snapshots (user_id, account_mode, market, captured_at, ticker);

-- Backs app.audit.screening_ic's per-date aggregation (first-cycle-per-date
-- pick, ordered by captured_at).
CREATE INDEX IF NOT EXISTS screening_snapshots_date_market_idx
    ON screening_snapshots (user_id, account_mode, market, trade_date, captured_at);
"""

_schema_ready = False
_schema_lock = asyncio.Lock()


async def _ready_pool():
    """Double-checked locking around the first-time DDL — same pattern as
    app.live.orderbook_log._ready_pool / app.backtest.minute_candles."""
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


@dataclass(frozen=True)
class ScreeningSnapshotContext:
    market: Market
    captured_at: datetime
    trade_date: date
    lookback_days: int
    volatility_adjusted: bool
    correlation_dedup: bool
    top_n_requested: int
    max_tickers: int


async def record_screening_snapshot(
    ctx: ScreeningSnapshotContext,
    *,
    universe: list[str],
    eligible: list[str],
    executable: list[str],
    scored: dict[str, MomentumScore],
    score_key: str,
    ranked: list[tuple[str, float]],
    final: list[str],
    held_review: list[str],
    cash_freed: list[str],
    axes_by_ticker: dict[str, list[str]],
    observed: dict[str, MomentumScore] | None = None,
    rebalance_locked: dict[str, date] | None = None,
) -> int:
    """Writes one row per ticker seen this cycle (universe plus any forced-
    review ticker not in `universe` — `exclude_already_analyzed_today` can
    remove a held ticker from `universe` while `_held_needing_review` still
    surfaces it). `score_key` must be the SAME field
    `app.chat.screening._ranked_and_deduped` ranked by ("score" when
    volatility-adjusted, "raw_pct" otherwise) — passing a different one here
    would record a score the live ranking never actually used.

    `observed` (app.chat.screening.ScreeningPass.observed) and
    `rebalance_locked` are both optional/default-empty so any pre-2026-09-09
    caller keeps writing byte-identical rows. `observed` scores are recorded
    (score_observational=true) but were NEVER part of the ranking — see
    score_observational's own DDL comment above.

    Degrades quietly on any failure (never raises) — this is observability,
    not a trading path. Returns the number of rows written (0 if disabled or
    on failure)."""
    if not settings.screening_snapshot_log_enabled:
        return 0
    try:
        observed = observed or {}
        rebalance_locked = rebalance_locked or {}
        tickers = list(dict.fromkeys(list(universe) + list(final)))
        if not tickers:
            return 0

        eligible_set = frozenset(eligible)
        executable_set = frozenset(executable)
        ranked_set = frozenset(t for t, _ in ranked)
        final_set = frozenset(final)
        held_set = frozenset(held_review)
        cash_set = frozenset(cash_freed)

        # `scored` wins on overlap (should never actually overlap —
        # screening_pass dedupes observe_only against its own universe — but
        # if it ever does, the real ranking's value is the one that matters).
        all_scored = {**observed, **scored}

        # 1-based rank by score_key DESC over scored+observed together —
        # computed once here, not per-ticker, so this is O(n log n) not
        # O(n^2). The ranking itself only ever sorted `scored`; this column
        # additionally ranks the observe-only tickers for the counterfactual
        # "where would a locked ticker have landed" question.
        by_score = sorted(
            all_scored.items(), key=lambda item: getattr(item[1], score_key), reverse=True
        )
        score_rank_of = {ticker: i + 1 for i, (ticker, _) in enumerate(by_score)}

        user_id = get_current_user_id()
        account_mode = current_account_mode()
        git_commit = _current_git_commit()

        rows = []
        for ticker in tickers:
            s = all_scored.get(ticker)
            if ticker in held_set:
                selection_reason = "held_review"
            elif ticker in cash_set:
                selection_reason = "cash_freed"
            elif ticker in final_set:
                selection_reason = "momentum"
            else:
                selection_reason = None
            rows.append((
                ctx.captured_at,
                ctx.trade_date,
                ctx.market,
                ticker,
                getattr(s, score_key) if s is not None else None,
                s.raw_pct if s is not None else None,
                s.atr_pct if s is not None else None,
                score_rank_of.get(ticker),
                ticker in eligible_set,
                ticker in executable_set,
                s is not None,
                ticker in ranked_set,
                ticker in final_set,
                selection_reason,
                axes_by_ticker.get(ticker, []),
                ctx.lookback_days,
                ctx.volatility_adjusted,
                ctx.correlation_dedup,
                ctx.top_n_requested,
                ctx.max_tickers,
                user_id,
                account_mode,
                git_commit,
                ticker in observed and ticker not in scored,
                ticker in rebalance_locked,
                rebalance_locked.get(ticker),
            ))

        pool = await _ready_pool()
        inserted = 0
        async with pool.acquire() as conn:
            async with conn.transaction():
                for row in rows:
                    status = await conn.execute(
                        "INSERT INTO screening_snapshots ("
                        "captured_at, trade_date, market, ticker, score, raw_pct, atr_pct, score_rank, "
                        "eligible, executable, scored, in_ranked_pool, selected, selection_reason, axes, "
                        "lookback_days, volatility_adjusted, correlation_dedup, top_n_requested, max_tickers, "
                        "user_id, account_mode, git_commit, score_observational, rebalance_locked, "
                        "rebalance_unlock_date"
                        ") VALUES ("
                        "$1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, "
                        "$20, $21, $22, $23, $24, $25, $26"
                        ") ON CONFLICT (user_id, account_mode, market, captured_at, ticker) DO NOTHING",
                        *row,
                    )
                    try:
                        inserted += int(status.rsplit(" ", 1)[-1])
                    except (ValueError, AttributeError):
                        pass
        return inserted
    except Exception:  # noqa: BLE001 - research logging must never break the live cycle
        logger.warning("screening snapshot logging failed — continuing without it this cycle", exc_info=True)
        return 0
