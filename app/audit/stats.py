"""Self-audit: is a given rating actually worth anything, based on realized
outcomes already sitting in `memory_entries`? Answers "how often does a
'Buy' rating produce positive alpha, and by how much on average" per rating
tier — using only columns that already exist (`rating`, `alpha_return`,
`raw_return`, `status`), no schema change.

Unlike the in-pipeline memory functions (`app/memory/store.py`), this is an
on-demand reporting tool, not a live decision dependency — a DB failure
here should be visible to whoever ran the report, not silently swallowed.

`compute_hit_rate_by_rating` counts only the MOST RECENT resolved entry per
(ticker, trade_date) — repeatedly re-running the same backtest case (e.g.
to verify a prompt/code fix) has no unique constraint stopping it from
inserting another row for the same case each time, which would otherwise
let one re-tested case outweigh many genuinely distinct ones in the
aggregate stats. Nothing is deleted, though: `get_decision_history` below
still surfaces every attempt for a given (ticker, trade_date), in order —
the before/after comparison the aggregate view intentionally no longer
shows.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

from app.memory.db import get_pool
from app.user_context import get_current_user_id


# Which SIGN of alpha means the call was RIGHT, per rating. Buy/Overweight
# increase exposure, so the ticker beating its benchmark is the win. Sell and
# Underweight REDUCE exposure, so the win is the opposite: the benchmark
# beating the ticker is what makes stepping out of it the correct call.
#
# Until 2026-08-13 this module scored `alpha_return > 0` as a win for EVERY
# rating, which measured the two exposure-reducing tiers exactly backwards.
# Live evidence at that point: Sell (n=11) reported a 72.7% win rate when the
# direction-aware figure was 27.3%, and Underweight (n=5) reported 0.0% when
# it was actually 100%. The stage gate's first-tier metric is alpha win rate,
# so this fed a wrong sign straight into the decision about whether to trust
# the system with real capital.
#
# Hold used to be entirely absent from this map: a Hold on an existing
# position wins when alpha > 0 (the thing we kept outperformed), but a Hold
# while FLAT means "stay in cash," which wins when alpha < 0 (the thing we
# passed on underperformed) — opposite tests, and nothing on the row
# distinguished them before `held_at_decision` existed (added 2026-08-13,
# Stage 1/F5 — see app.memory.db's column comment). The SQL below now
# buckets `rating = 'Hold'` into 'Hold (held)' / 'Hold (flat)' /
# 'Hold (held/flat unknown)' using that column, so only genuinely unknown
# rows (predating the column, or where the pre-decision position lookup
# itself failed) still fall through to the unscored bucket.
# 매수계열은 알파>0이 정답, 매도계열은 알파<0이 정답. Hold는 held_at_decision
# 컬럼(2026-08-13 추가)으로 보유중/현금중을 구분해 이제 반대 방향으로 채점하고,
# 그 컬럼이 없던(또는 조회 실패한) 과거 행만 미채점으로 남는다.
#
# 2026-08-20: 'Hold (held)' only — NOT 'Hold (flat)' — was found to be scored
# against the wrong counterfactual. This system is long-only and cannot
# rotate into the benchmark index, so the real alternative to "keep holding
# what I own" is "sell into cash," not "buy the index instead" — the correct
# win test is `raw_return > 0` (did the position actually make money), not
# `alpha_return > 0` (did it beat the index). 'Hold (flat)' keeps alpha as
# its test: the alternative there genuinely IS "should I have bought," so
# comparing against the benchmark (not just whether the stock happened to
# rise) is the right question. See project memory
# hold-scoring-gap-and-hold-held-defect-2026-08-20.md — a naive alpha-based
# reading of this bucket (26.4% win rate, n=53) looked like a serious AI
# defect; on raw_return the same clean sample scores 60.5%. User decision
# (2026-08-20): raw_return is the correct test for this bucket, not alpha.
# Twin of app.backtest.metrics._ALPHA_WIN_DIRECTION for in-run backtest
# scoring, where held/flat is unknowable so Hold is unscored rather than
# split into (held)/(flat) the way this module's SQL does.
_WIN_DIRECTION: dict[str, int] = {
    "Buy": 1,
    "Overweight": 1,
    "Underweight": -1,
    "Sell": -1,
    "Hold (held)": 1,
    "Hold (flat)": -1,
    # 2026-08-20 — see _GATED_LABEL_RE's own comment for why these exist.
    # Scored on the model's ORIGINAL (pre-gate) call, using alpha like every
    # other directional rating — these are "should I have entered/exited"
    # judgments, not a held-position cash-counterfactual question, so
    # Hold (held)'s raw_return override above does not apply here.
    "Buy (gated)": 1,
    "Sell (gated)": -1,
}
_UNSCORED_CONDITION = "unscored (needs held/flat)"
# Ratings whose win test is `raw_return`, not `alpha_return` — see
# _WIN_DIRECTION's own 2026-08-20 comment.
_RAW_RETURN_RATINGS: frozenset[str] = frozenset({"Hold (held)"})


def _win_condition_label(rating: str) -> str:
    direction = _WIN_DIRECTION.get(rating)
    if direction is None:
        return _UNSCORED_CONDITION
    metric = "raw" if rating in _RAW_RETURN_RATINGS else "alpha"
    return f"{metric} > 0" if direction > 0 else f"{metric} < 0"


@dataclass
class RatingStats:
    # NOT a raw 5-way PortfolioRating any more (Stage 1, 2026-08-13) — this is
    # the SCORING BUCKET label: Buy/Overweight/Underweight/Sell unchanged, but
    # 'Hold' is split into 'Hold (held)', 'Hold (flat)', and
    # 'Hold (held/flat unknown)' (see `_WIN_DIRECTION`'s own comment for why).
    rating: str
    n: int  # resolved AND scorable entries (see `unscorable_n`)
    # Fraction of `n` where alpha moved in the direction that makes this
    # rating correct — see `_WIN_DIRECTION`. `None` for a rating whose
    # correct direction is not determinable (Hold) or with no scorable rows.
    win_rate: float | None
    # Raw integer win count SQL already computes (win_rate = wins / n) —
    # carried through separately so callers (e.g. directional_totals) don't
    # have to reconstruct an int from a rounded float. None under the same
    # condition win_rate is None (no determinable direction, or n == 0).
    wins: int | None = None
    mean_alpha: float | None = None
    mean_raw_return: float | None = None
    win_condition: str = "alpha > 0"
    # Rows dropped from every figure above because their stored return is
    # NULL or NaN. Surfaced rather than silently excluded because they were
    # previously counted as WINS: Postgres sorts NaN above every real number,
    # so the old `alpha_return > 0` test was true for a NaN row.
    # NaN 행은 예전에 '승'으로 잡혔음(Postgres가 NaN을 최대값으로 취급) — 이제
    # 제외하되 개수를 따로 보여줌.
    unscorable_n: int = 0
    # Sample standard deviation of alpha (SQL stddev_samp), for a future
    # confidence-interval calculation (out of scope for this stage — see
    # Stage 1 design doc §5-4) — computed and stored now while the query is
    # already being touched, but deliberately NOT surfaced in `render()` yet.
    stdev_alpha: float | None = None
    # How many of `n` came from held_at_decision_inferred (2026-08-20 ledger
    # backfill, ~97.2% measured agreement) rather than the directly-observed
    # held_at_decision column — only ever nonzero for the two Hold buckets,
    # since only Hold's scoring direction depends on this at all. Surfaced
    # so a report never presents a reconstructed row as directly observed
    # without saying so — see app.memory.db's held_at_decision_inferred
    # column comment.
    inferred_n: int = 0


# Buckets the Portfolio Manager's 5-way final rating to the same 3-way
# direction the Trader proposes (app.state.TraderAction) — Buy/Overweight
# both mean "increase exposure" and Underweight/Sell both mean "decrease
# exposure" in this project's own existing sizing convention
# (app.execution.sizing.target_allocation_for: Buy 100%, Overweight 75%,
# Underweight 25%, Sell 0% — a monotonic spectrum, not five unrelated
# labels). A naive string equality against `rating` would count every
# Overweight/Underweight final rating as "the debate changed the outcome"
# even when the Trader and the debate agreed on direction the whole time.
_RATING_TO_DIRECTION = {
    "Buy": "Buy",
    "Overweight": "Buy",
    "Hold": "Hold",
    "Underweight": "Sell",
    "Sell": "Sell",
}


@dataclass
class DebateImpactStats:
    n: int
    changed: int
    changed_pct: float
    breakdown: dict[str, int]  # "{trader_action}->{final_direction}" -> count


@dataclass
class DecisionAttempt:
    """One row of `get_decision_history` — a single past attempt at the
    same (ticker, trade_date), in case it was replayed more than once
    (e.g. re-run after a prompt/code fix to check whether the outcome or
    reasoning changed)."""
    created_at: datetime
    rating: str
    status: str
    alpha_return: float | None
    raw_return: float | None
    reflection: str | None
    is_backtest: bool
    account_mode: str


async def compute_hit_rate_by_rating(
    ticker: str | None = None, market: str | None = None, scope: str | None = None,
    account_mode: str | None = None, trigger: str | None = None, basis: str = "variable",
    since: date | None = None, until: date | None = None,
) -> list[RatingStats]:
    """One row per rating tier that has at least one resolved entry
    (`ticker=None` means all tickers), counting only the latest attempt per
    (ticker, trade_date) — see module docstring. Raises on a DB failure —
    see module docstring for why this doesn't degrade quietly like the
    pipeline's own memory calls.

    `market` ("domestic", "overseas", or None — default, no filter) restricts
    to KRX (`.KS` suffix) or non-KRX tickers, same convention as
    app.execution.portfolio_risk.ticker_market. Domestic/overseas separation
    audit (2026-07-27, docs/domestic-overseas-separation-plan.md): without
    this, a report with no `ticker` given pooled domestic and overseas
    tickers into one win-rate/alpha number by default, with no way to see
    either market's hit rate on its own.

    `scope` ("live", "backtest", or None — default, no filter, same pooling
    convention as `market` above) restricts to `is_backtest = false`/`true`
    rows. Added 2026-08-03 alongside app.memory.auto_resolve's live
    auto-resolve scheduler — before that, live decisions almost never
    resolved (the manual CLI to do so had no evidence of ever actually
    being run), so this query silently pooling live+backtest resolved
    entries never actually mixed meaningfully different populations. Now
    that live resolves every cycle, the default (None, pooled) will start
    blending live and backtest hit rates into one number unless a caller
    passes `scope` explicitly — kept as an opt-in rather than changing the
    default, matching `market`'s own established pattern in this function.

    `account_mode` ("paper", "real", or None — default, no filter, same
    pooling convention as `market`/`scope` above) restricts to
    `account_mode = 'paper'`/`'real'` rows (2026-08-03 — see project memory:
    user-account-mode-db-separation-design). Without this, once real-mode
    trading starts, a report with no explicit filter would blend paper's
    and real's hit rates into one number with no way to see either
    account's own performance track record — same reasoning as `scope`.

    `basis` ("variable", the default, or "horizon", Stage 1/2026-08-13):
    which return columns to score against. "variable" (unchanged behavior)
    uses the existing `raw_return`/`alpha_return` — a window that varies
    5 to 112+ calendar days across live rows by design (see
    `_common_eval_date`'s own docstring — it exists specifically to never
    stall). "horizon" instead uses `horizon_raw_return`/`horizon_alpha_
    return` — a fixed `settings.memory_fixed_horizon_trading_days`-trading-
    day window (see app.memory.db's `horizon_*` column comments and
    app.memory.auto_resolve's `_horizon_eval_date`) that IS comparable
    across rows, at the cost of many rows still being NULL (not yet
    matured, or beyond the backfill window) — those rows fall out via the
    same NULL/NaN `ok` filter as "variable" does. Column names are chosen
    from this fixed, hardcoded pair rather than ever being interpolated
    from caller input — no injection surface.

    `since`/`until` (date, both optional, inclusive — Stage 2/weekly report,
    2026-08-13): restrict to rows whose `horizon_eval_date` falls in
    [since, until]. This is the ANCHOR COLUMN regardless of `basis` — even
    when `basis="variable"` scores against `alpha_return`/`raw_return`, the
    time window is still defined by `horizon_eval_date` (the fixed
    10-trading-day mark), because it is the only eval-date column that means
    the same thing across every row (see `basis`'s own docstring on why
    `alpha_return`'s window varies 5 to 112+ calendar days). A side effect:
    rows with `horizon_eval_date IS NULL` (not yet matured, or beyond the
    backfill window — see that column's own comment) can never match a
    since/until filter, even under `basis="variable"` where their
    alpha_return may already be populated. Passing neither (both None,
    the default) applies no date filter at all, same as before this
    parameter existed.

    `trigger` ("auto", "manual", or None — default, no filter, same pooling
    convention as `market`/`scope`/`account_mode` above — 2026-08-20):
    restricts to `decision_trigger`-tagged rows. "auto" MATCHES NULL too
    (`decision_trigger IS DISTINCT FROM 'manual'`) — NULL means the row
    predates this column (app.memory.db), and there is no way to
    retroactively tell those apart from a genuinely auto-triggered row, so
    excluding NULL would drop nearly this table's entire pre-2026-08-20
    history from an "auto-only" view. "manual" matches only rows a human
    explicitly triggered (the Trading tab's analyze button, a chat
    deep-dive, or a CLI/dev-script run) — these are sample-selection
    biased (a person only requests one when already curious or worried
    about that ticker) and should not silently feed the win-rate/alpha
    numbers used to gate capital increases. See app.memory.db's
    decision_trigger column comment."""
    if market not in (None, "domestic", "overseas"):
        raise ValueError(f"market must be 'domestic', 'overseas', or None, got {market!r}")
    if scope not in (None, "live", "backtest"):
        raise ValueError(f"scope must be 'live', 'backtest', or None, got {scope!r}")
    if account_mode not in (None, "paper", "real"):
        raise ValueError(f"account_mode must be 'paper', 'real', or None, got {account_mode!r}")
    if trigger not in (None, "auto", "manual"):
        raise ValueError(f"trigger must be 'auto', 'manual', or None, got {trigger!r}")
    if basis not in ("variable", "horizon"):
        raise ValueError(f"basis must be 'variable' or 'horizon', got {basis!r}")
    alpha_col, raw_col = ("alpha_return", "raw_return") if basis == "variable" else (
        "horizon_alpha_return", "horizon_raw_return",
    )

    conditions = ["status = 'resolved'"]
    params: list[str] = []
    if ticker is not None:
        params.append(ticker)
        conditions.append(f"ticker = ${len(params)}")
    if market == "domestic":
        conditions.append("ticker LIKE '%.KS'")
    elif market == "overseas":
        conditions.append("ticker NOT LIKE '%.KS'")
    if scope == "live":
        conditions.append("is_backtest = false")
    elif scope == "backtest":
        conditions.append("is_backtest = true")
    if account_mode is not None:
        params.append(account_mode)
        conditions.append(f"account_mode = ${len(params)}")
    if trigger == "auto":
        # NULL = predates this column (app.memory.db's decision_trigger
        # comment) — counted as auto, since excluding it would drop the
        # entire pre-tagging history from the very screen used to gate
        # capital. Literal-only, no bind param needed (no injection surface,
        # same as the market/scope fragments above).
        conditions.append("decision_trigger IS DISTINCT FROM 'manual'")
    elif trigger == "manual":
        conditions.append("decision_trigger = 'manual'")
    if since is not None:
        params.append(since)
        conditions.append(f"horizon_eval_date >= ${len(params)}")
    if until is not None:
        params.append(until)
        conditions.append(f"horizon_eval_date <= ${len(params)}")
    # analysts_failed (app.memory.db column comment: "애널리스트 4명 중 실패한
    # 수 — 축소 분석과 정상 분석이 통계에서 섞이던 문제") — exclude only rows
    # explicitly marked degraded (>0); NULL predates this column and stays
    # included, since "unknown" isn't evidence of degradation.
    conditions.append("(analysts_failed IS NULL OR analysts_failed = 0)")
    # Scoped to one person (multi-user migration Phase 4, plan §2.2). Without
    # this, win-rate and alpha blended every user's decisions into one number
    # — and unlike the other Tier B gaps this one is silently WRONG rather
    # than unsafe: the figure looks plausible, so nobody checks it. It is also
    # the number a person would use to decide whether to trust the system with
    # more money.
    # 사용자별로 분리하지 않으면 승률/알파가 전원 합산값이 되는데, 값이 그럴듯해
    # 보여서 아무도 의심하지 않음 — 게다가 이 수치를 보고 투자 규모를 정하게 됨.
    params.append(get_current_user_id())
    conditions.append(f"user_id = ${len(params)}")
    where_clause = " AND ".join(conditions)

    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            rf"""
            WITH latest AS (
                SELECT DISTINCT ON (ticker, trade_date)
                    rating, {alpha_col} AS alpha_return, {raw_col} AS raw_return,
                    held_at_decision, held_at_decision_inferred,
                    -- Observed vs. reconstructed — see app.memory.db's
                    -- held_at_decision_inferred column comment. Used only to
                    -- count `inferred_n` below; the bucketing itself already
                    -- treats the two the same way via COALESCE.
                    held_at_decision IS NULL AND held_at_decision_inferred IS NOT NULL AS is_inferred,
                    -- 2026-08-20: recovers the model's ORIGINAL Buy/Sell call
                    -- when `_apply_hysteresis_gate` (pipeline.py) downgraded it
                    -- to Hold for missing/low confidence. Checks TWO sources,
                    -- in order, because the short-form prefix on
                    -- executive_summary (`gate_prefix` in
                    -- _apply_hysteresis_gate, landed into decision_text) is
                    -- newer than the long-form note on investment_thesis
                    -- (only in full_transcript, never in decision_text) — a
                    -- handful of real rows (e.g. id=639, 316140.KS) predate
                    -- the short-form fix and carry ONLY the long-form note.
                    -- Do not reformat either prefix without updating this
                    -- pattern too — app.experiments.language_ab_report.
                    -- _HYSTERESIS_NOTE_RE parses the same long-form note from
                    -- a different (offline experiment JSON) source. Without
                    -- this, a gate-downgraded row silently scored as a Hold
                    -- judgment using the Hold win test — but the model never
                    -- actually made a Hold call here; the gate overrode it.
                    -- Real incident: id=639 said "전량 매도" (full liquidation)
                    -- in its own prose, was gated to Hold for unset
                    -- confidence, then scored as a losing Hold when the Sell
                    -- call was in fact correct (alpha -6.03%). NULL when the
                    -- row was never gated (the common case).
                    COALESCE(
                        substring(decision_text from '\[Hysteresis gate: downgraded from (\w+) to Hold'),
                        substring(
                            full_transcript->>'investment_thesis'
                            from '\[Hysteresis gate: downgraded from (\w+) to Hold'
                        )
                    ) AS gate_orig_rating
                FROM memory_entries
                WHERE {where_clause}
                ORDER BY ticker, trade_date, created_at DESC
            ), scorable AS (
                -- Bucketing order: a gate-downgraded row is scored on the
                -- model's ORIGINAL call first — it is never actually a Hold
                -- judgment, gated or not, so it must not fall into the Hold
                -- split below. Only then does Hold get split into
                -- held/flat/unknown buckets using COALESCE(held_at_decision,
                -- held_at_decision_inferred) — the observed column (Stage 1,
                -- 2026-08-13) when present, else the 2026-08-20 ledger-
                -- reconstructed backfill for rows that predate it (97.2%
                -- measured agreement — see that column's own comment). Only
                -- rows where NEITHER column is set (predates the column AND
                -- wasn't backfilled — e.g. the backtest-replay contamination
                -- rows, or a row outside the backfill's 3-day trade_date/
                -- created_at window) stay unscored. `IS TRUE`/`IS FALSE` are
                -- NULL-safe (unlike `=`/`<>`).
                SELECT
                    CASE
                        WHEN gate_orig_rating IS NOT NULL THEN gate_orig_rating || ' (gated)'
                        WHEN rating = 'Hold' AND COALESCE(held_at_decision, held_at_decision_inferred) IS TRUE
                            THEN 'Hold (held)'
                        WHEN rating = 'Hold' AND COALESCE(held_at_decision, held_at_decision_inferred) IS FALSE
                            THEN 'Hold (flat)'
                        WHEN rating = 'Hold' THEN 'Hold (held/flat unknown)'
                        ELSE rating
                    END AS rating,
                    alpha_return, raw_return, is_inferred,
                    -- `<> 'NaN'` is a real NaN test in Postgres, which (unlike
                    -- IEEE 754) defines NaN = NaN as true and sorts NaN above
                    -- every other float. That same ordering rule is why the old
                    -- `alpha_return > 0` win test silently counted NaN rows as
                    -- wins, so they have to be excluded explicitly here.
                    alpha_return IS NOT NULL AND alpha_return <> 'NaN'::float8
                        AND raw_return IS NOT NULL AND raw_return <> 'NaN'::float8 AS ok
                FROM latest
            )
            SELECT
                rating,
                COUNT(*) FILTER (WHERE ok) AS n,
                COUNT(*) FILTER (WHERE ok AND is_inferred) AS inferred_n,
                COUNT(*) FILTER (WHERE NOT ok) AS unscorable,
                -- Direction-aware: see _WIN_DIRECTION. Anything with no
                -- determinable direction (bare 'Hold (held/flat unknown)')
                -- falls through to NULL, which FILTER treats as not-matching,
                -- so its `wins` is 0 and the caller reports win_rate as None
                -- rather than 0%. 'Hold (held)' alone tests `raw_return`, not
                -- `alpha_return` — see _WIN_DIRECTION's 2026-08-20 comment for
                -- why (long-only system, real counterfactual is cash, not the
                -- benchmark). The two '(gated)' buckets test the model's
                -- ORIGINAL directional call the same way Buy/Sell always are.
                COUNT(*) FILTER (WHERE ok AND CASE
                    WHEN rating IN ('Buy', 'Overweight', 'Buy (gated)') THEN alpha_return > 0
                    WHEN rating IN ('Sell', 'Underweight', 'Sell (gated)') THEN alpha_return < 0
                    WHEN rating = 'Hold (held)' THEN raw_return > 0
                    WHEN rating = 'Hold (flat)' THEN alpha_return < 0
                    ELSE NULL
                END) AS wins,
                AVG(alpha_return) FILTER (WHERE ok) AS mean_alpha,
                AVG(raw_return) FILTER (WHERE ok) AS mean_raw,
                STDDEV_SAMP(alpha_return) FILTER (WHERE ok) AS stdev_alpha
            FROM scorable
            GROUP BY rating
            ORDER BY rating
            """,
            *params,
        )

    return [
        RatingStats(
            rating=row["rating"],
            n=row["n"],
            # None (not 0.0) when the rating has no determinable correct
            # direction, or when every row for it was unscorable — both mean
            # "no figure", which must not read as "0% of calls were right".
            win_rate=(
                row["wins"] / row["n"]
                if row["n"] and row["rating"] in _WIN_DIRECTION
                else None
            ),
            wins=row["wins"] if row["wins"] is not None else None,
            mean_alpha=float(row["mean_alpha"]) if row["mean_alpha"] is not None else None,
            mean_raw_return=float(row["mean_raw"]) if row["mean_raw"] is not None else None,
            win_condition=_win_condition_label(row["rating"]),
            unscorable_n=row["unscorable"],
            stdev_alpha=(
                float(row["stdev_alpha"])
                if "stdev_alpha" in row.keys() and row["stdev_alpha"] is not None
                else None
            ),
            inferred_n=row["inferred_n"],
        )
        for row in rows
    ]


async def compute_debate_impact(
    ticker: str | None = None, market: str | None = None, scope: str | None = None,
    account_mode: str | None = None, trigger: str | None = None,
    since: date | None = None, until: date | None = None,
) -> DebateImpactStats:
    """How often the risk debate + Portfolio Manager actually changed the
    Trader's own initial Buy/Hold/Sell call, over every decision that has a
    recorded `trader_action` — rows written before that column existed
    (NULL) are excluded, not misread as "never changed" (see the column's
    own comment in app.memory.db and [[agent-architecture-roadmap]]'s
    "unverified risk-debate contribution" item this is meant to eventually
    answer). Unlike `compute_hit_rate_by_rating`, this doesn't filter to
    `status='resolved'` — whether the debate changed direction is knowable
    the moment a decision is made, independent of whether its outcome has
    resolved yet.

    Same latest-per-(ticker, trade_date) dedup as `compute_hit_rate_by_rating`
    — a re-run of the same backtest case (e.g. to check a prompt change)
    shouldn't outweigh a distinct one in the aggregate.

    `scope`/`account_mode`/`trigger` — see `compute_hit_rate_by_rating`'s own
    docstring for the same "live"/"backtest"/None(pooled, default),
    "paper"/"real"/None(pooled, default), and "auto"/"manual"/None(pooled,
    default) conventions.

    `since`/`until` (date, both optional, inclusive — Stage 2/weekly report,
    2026-08-13): UNLIKE `compute_hit_rate_by_rating`/
    `compute_confidence_calibration`, this does NOT anchor on
    `horizon_eval_date` — this function deliberately includes rows that
    haven't resolved yet (see above), so most of them have no eval date at
    all. It falls back to `created_at` instead (converted to KST calendar
    date, matching the Monday-Sunday KST week boundary the weekly report
    itself uses — see app.audit.weekly._week_bounds), since "did the debate
    change this decision" is knowable the moment the decision is made,
    independent of when/whether it later resolves."""
    if market not in (None, "domestic", "overseas"):
        raise ValueError(f"market must be 'domestic', 'overseas', or None, got {market!r}")
    if scope not in (None, "live", "backtest"):
        raise ValueError(f"scope must be 'live', 'backtest', or None, got {scope!r}")
    if account_mode not in (None, "paper", "real"):
        raise ValueError(f"account_mode must be 'paper', 'real', or None, got {account_mode!r}")
    if trigger not in (None, "auto", "manual"):
        raise ValueError(f"trigger must be 'auto', 'manual', or None, got {trigger!r}")

    conditions = ["trader_action IS NOT NULL"]
    params: list[str] = []
    if ticker is not None:
        params.append(ticker)
        conditions.append(f"ticker = ${len(params)}")
    if market == "domestic":
        conditions.append("ticker LIKE '%.KS'")
    elif market == "overseas":
        conditions.append("ticker NOT LIKE '%.KS'")
    if scope == "live":
        conditions.append("is_backtest = false")
    elif scope == "backtest":
        conditions.append("is_backtest = true")
    if account_mode is not None:
        params.append(account_mode)
        conditions.append(f"account_mode = ${len(params)}")
    if trigger == "auto":
        conditions.append("decision_trigger IS DISTINCT FROM 'manual'")
    elif trigger == "manual":
        conditions.append("decision_trigger = 'manual'")
    if since is not None:
        params.append(since)
        conditions.append(f"(created_at AT TIME ZONE 'Asia/Seoul')::date >= ${len(params)}")
    if until is not None:
        params.append(until)
        conditions.append(f"(created_at AT TIME ZONE 'Asia/Seoul')::date <= ${len(params)}")
    # Scoped to one person (multi-user migration Phase 4, plan §2.2). Without
    # this, win-rate and alpha blended every user's decisions into one number
    # — and unlike the other Tier B gaps this one is silently WRONG rather
    # than unsafe: the figure looks plausible, so nobody checks it. It is also
    # the number a person would use to decide whether to trust the system with
    # more money.
    # 사용자별로 분리하지 않으면 승률/알파가 전원 합산값이 되는데, 값이 그럴듯해
    # 보여서 아무도 의심하지 않음 — 게다가 이 수치를 보고 투자 규모를 정하게 됨.
    params.append(get_current_user_id())
    conditions.append(f"user_id = ${len(params)}")
    where_clause = " AND ".join(conditions)

    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            f"""
            SELECT DISTINCT ON (ticker, trade_date) trader_action, rating
            FROM memory_entries
            WHERE {where_clause}
            ORDER BY ticker, trade_date, created_at DESC
            """,
            *params,
        )

    breakdown: dict[str, int] = {}
    changed = 0
    for row in rows:
        direction = _RATING_TO_DIRECTION.get(row["rating"], row["rating"])
        key = f"{row['trader_action']}->{direction}"
        breakdown[key] = breakdown.get(key, 0) + 1
        if direction != row["trader_action"]:
            changed += 1

    n = len(rows)
    return DebateImpactStats(
        n=n,
        changed=changed,
        changed_pct=(changed / n) if n else 0.0,
        breakdown=breakdown,
    )


async def get_decision_history(ticker: str, trade_date: date) -> list[DecisionAttempt]:
    """Every attempt on record for this exact (ticker, trade_date), oldest
    first — the before/after comparison `compute_hit_rate_by_rating`'s
    latest-only aggregate intentionally no longer shows. Includes pending
    (not yet resolved) attempts too, unlike the aggregate stats, since a
    fix's effect on the analysis/reasoning itself may be visible before
    the outcome resolves."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT created_at, rating, status, alpha_return, raw_return, reflection,
                is_backtest, account_mode
            FROM memory_entries
            WHERE ticker = $1 AND trade_date = $2 AND user_id = $3
            ORDER BY created_at ASC
            """,
            ticker,
            trade_date,
            get_current_user_id(),
        )
    return [
        DecisionAttempt(
            created_at=row["created_at"],
            rating=row["rating"],
            status=row["status"],
            alpha_return=float(row["alpha_return"]) if row["alpha_return"] is not None else None,
            raw_return=float(row["raw_return"]) if row["raw_return"] is not None else None,
            reflection=row["reflection"],
            is_backtest=row["is_backtest"],
            account_mode=row["account_mode"],
        )
        for row in rows
    ]


# Rating buckets that represent an actual exposure change — used by render()
# to keep the "Directional decisions" headline meaning exactly what it says,
# now that 'Hold (held)'/'Hold (flat)' also have a determinable win_rate
# (Stage 1, 2026-08-13) but are not themselves an exposure-direction bet.
#
# This tally is PARTIAL BY DESIGN, not a bug (re-confirmed 2026-08-26 during
# the app/web audit, which flagged a '(gated)' row sitting outside it as a
# possible "accounting gap"): a hysteresis-gate-downgraded rating (e.g. "Buy
# (gated)") is deliberately excluded here for the same reason 'Hold (held)'/
# 'Hold (flat)' are — no exposure change actually occurred, gated or not.
# Every excluded bucket still gets its own N/win-rate row in the full
# per-rating table (both the CLI report and the web audit tab); nothing is
# hidden, only kept out of a headline that specifically means "directional
# bets taken." See test_gated_buckets_excluded_from_directional_headline for
# the pinned intent.
_DIRECTIONAL_RATINGS = {"Buy", "Overweight", "Sell", "Underweight"}


@dataclass
class DirectionalTotals:
    scored_n: int
    scored_wins: int
    win_rate: float | None
    unscored_hold_n: int
    unscorable_n: int


def directional_totals(stats: list[RatingStats]) -> DirectionalTotals:
    """The "Directional decisions"/"Directional win rate" headline figures,
    lifted out of `render()` (accuracy web API, 2026-08-20) so the CLI and
    any other caller compute this identically and can never drift apart.
    Covers Buy/Overweight/Sell/Underweight only — see `_DIRECTIONAL_RATINGS`
    and `render()`'s own comment for why 'Hold (held)'/'Hold (flat)' are
    excluded from this specific tally despite having a determinable
    win_rate."""
    scored = [s for s in stats if s.rating in _DIRECTIONAL_RATINGS and s.win_rate is not None]
    scored_n = sum(s.n for s in scored)
    # A scored row always contributes its N to the denominator, so it must
    # contribute its wins to the numerator too. SQL fills `wins` directly;
    # a caller that only has the rounded `win_rate` (win_rate is not None but
    # wins is None) would otherwise add N to the denominator and 0 to the
    # numerator, dragging the headline toward 0. Reconstruct the count there.
    scored_wins = sum(
        s.wins if s.wins is not None else round(s.win_rate * s.n)
        for s in scored
    )
    unscored_hold_n = sum(s.n for s in stats if s.rating not in _DIRECTIONAL_RATINGS and s.win_rate is None)
    unscorable_n = sum(s.unscorable_n for s in stats)
    return DirectionalTotals(
        scored_n=scored_n,
        scored_wins=scored_wins,
        win_rate=(scored_wins / scored_n) if scored_n else None,
        unscored_hold_n=unscored_hold_n,
        unscorable_n=unscorable_n,
    )


def render(
    stats: list[RatingStats], ticker: str | None = None, market: str | None = None, run_scope: str | None = None,
    account_mode: str | None = None, run_trigger: str | None = None, basis: str = "variable",
) -> str:
    scope = ticker if ticker else ("all tickers" if market is None else f"all {market} tickers")
    scope = f"{scope}, {run_scope}" if run_scope else scope
    scope = f"{scope}, {account_mode}" if account_mode else scope
    scope = f"{scope}, {run_trigger}" if run_trigger else scope
    title = "# Self-Audit — hit rate by rating"
    if basis == "horizon":
        title += " (fixed 10-trading-day horizon)"
    lines = [f"{title} ({scope})", ""]

    if not stats:
        lines.append("No resolved decisions yet — nothing to audit.")
        return "\n".join(lines)

    # Column widened from 12 to 26 (Stage 1, 2026-08-13) to fit
    # 'Hold (held/flat unknown)' (24 chars) without wrapping.
    lines.append(
        f"{'Rating':<26} {'N':>4} {'Win rate':>10} {'Mean alpha':>12} {'Mean raw':>10}  Win condition"
    )
    lines.append("-" * 84)
    for s in stats:
        win = f"{s.win_rate:>9.1%}" if s.win_rate is not None else f"{'—':>9}"
        alpha = f"{s.mean_alpha:>+11.2%}" if s.mean_alpha is not None else f"{'—':>11}"
        raw = f"{s.mean_raw_return:>+9.2%}" if s.mean_raw_return is not None else f"{'—':>9}"
        # 2026-08-20: how many of N came from held_at_decision_inferred (the
        # ledger-reconstruction backfill) rather than being directly
        # observed — only ever nonzero for the two Hold buckets. Printed
        # inline rather than as a separate column so a reader can't miss it
        # next to the exact N it qualifies.
        inferred_note = f" ({s.inferred_n} inferred)" if s.inferred_n else ""
        lines.append(f"{s.rating:<26} {s.n:>4} {win} {alpha} {raw}  {s.win_condition}{inferred_note}")

    # Headline covers only the ratings that ARE an exposure-direction bet —
    # 'Hold (held)'/'Hold (flat)' now also have a determinable win_rate
    # (Stage 1, 2026-08-13) but are deliberately kept out of this specific
    # tally so "Directional decisions (Buy/Overweight/Sell/Underweight)"
    # keeps meaning exactly what it says; their own N/win-rate is still
    # visible in the table above, not hidden.
    totals = directional_totals(stats)

    lines += [
        "",
        f"Directional decisions (Buy/Overweight/Sell/Underweight): {totals.scored_n}",
        (
            f"Directional win rate: {totals.win_rate:.1%}"
            if totals.win_rate is not None
            else "Directional win rate: N/A"
        ),
        f"Unscored (Hold — held vs flat not recorded): {totals.unscored_hold_n}",
        "",
        "'Win rate' is DIRECTION-AWARE: Buy/Overweight win on alpha > 0, Sell/Underweight win on",
        "alpha < 0 (stepping out of a name the benchmark then beat is a correct call). Scoring every",
        "rating on alpha > 0 measured the two exposure-reducing tiers backwards until 2026-08-13.",
        "Hold is now split into 'Hold (held)' and 'Hold (flat)' using the held_at_decision column",
        "(added 2026-08-13), falling back to held_at_decision_inferred (a 2026-08-20 ledger-",
        "reconstruction backfill for rows that predate that column, ~97.2% measured agreement — see",
        "'(N inferred)' next to a row's N above) when the observed column is unset. Only rows where",
        "neither column is set — genuinely unrecoverable, e.g. the pre-decision position lookup",
        "itself failed — remain 'Hold (held/flat unknown)' and stay unscored, shown in the",
        "'Unscored' line below. 'Hold (flat)' wins on alpha < 0, like Sell — the alternative to",
        "staying in cash is buying, so beating the benchmark is the right bar. 'Hold (held)' wins",
        "on raw_return > 0 instead (2026-08-20) — this system is long-only and cannot rotate into",
        "the index, so the real alternative to keeping a position is selling into cash, not buying",
        "the benchmark; scoring it on alpha made a genuinely profitable hold look like a loss simply",
        "for lagging a rallying index. See project memory hold-scoring-gap-and-hold-held-defect-",
        "2026-08-20.md for the investigation that found this.",
        "A row whose rating the hysteresis gate downgraded to Hold for missing/low confidence",
        "(app.orchestrator.pipeline._apply_hysteresis_gate) is scored on the model's ORIGINAL",
        "Buy/Sell call, not as a Hold — shown as 'Buy (gated)'/'Sell (gated)' in the table above,",
        "scored on alpha like their ungated counterparts. The gate only ever overrides an ENTRY or a",
        "FULL EXIT for low conviction; it never touches Overweight/Underweight or a genuine Hold.",
        "'Mean alpha'/'mean raw' are simple averages across resolved decisions, not annualized",
        "or weighted by holding period — and holding_days currently varies widely between rows.",
    ]
    if totals.unscorable_n:
        lines.append(
            f"WARNING: {totals.unscorable_n} resolved row(s) excluded entirely — stored return is NULL or NaN. "
            "These previously counted as WINS (Postgres sorts NaN above every real number)."
        )
    if run_scope is None:
        lines.append(
            "NOTE: pooling live and backtest decisions together (pass --scope live/backtest to split them) — "
            "since app.memory.auto_resolve started auto-resolving live decisions (2026-08-03), this number "
            "may blend two differently-sourced populations."
        )
    if account_mode is None:
        lines.append(
            "NOTE: pooling paper and real decisions together (pass --account-mode paper/real to split them) — "
            "once real-mode trading starts, this number may blend two differently-sourced populations."
        )
    if run_trigger is None:
        lines.append(
            "NOTE: pooling auto-triggered and manually-triggered decisions together (pass --trigger auto to "
            "exclude manual one-off analyses, which are sample-selection biased) — win rate below is not the "
            "number to gate capital on."
        )
    return "\n".join(lines)


def render_debate_impact(
    stats: DebateImpactStats, ticker: str | None = None, market: str | None = None, run_scope: str | None = None,
    account_mode: str | None = None, run_trigger: str | None = None,
) -> str:
    scope = ticker if ticker else ("all tickers" if market is None else f"all {market} tickers")
    scope = f"{scope}, {run_scope}" if run_scope else scope
    scope = f"{scope}, {account_mode}" if account_mode else scope
    scope = f"{scope}, {run_trigger}" if run_trigger else scope
    lines = [f"# Self-Audit — risk-debate impact ({scope})", ""]

    if stats.n == 0:
        lines.append(
            "No decisions with a recorded trader_action yet — this column was added "
            "2026-07-28, so only decisions made since then have one."
        )
        return "\n".join(lines)

    lines.append(f"Decisions counted: {stats.n}")
    lines.append(f"Debate changed the Trader's proposed direction: {stats.changed} ({stats.changed_pct:.1%})")
    lines.append("")
    lines.append("Breakdown (Trader proposal -> final direction):")
    for key in sorted(stats.breakdown):
        lines.append(f"  {key:<20} {stats.breakdown[key]}")
    lines += [
        "",
        "'Final direction' buckets the 5-way rating to Buy (Buy/Overweight), Hold, or "
        "Sell (Underweight/Sell) — the same increase/hold/decrease spectrum "
        "app.execution.sizing.target_allocation_for already scales by.",
        "This says nothing about whether the change was CORRECT (see `report` for hit "
        "rate/alpha by rating) — only how often the debate had any effect at all.",
    ]
    return "\n".join(lines)


def render_history(attempts: list[DecisionAttempt], ticker: str, trade_date: date) -> str:
    lines = [f"# Decision history — {ticker} on {trade_date}", ""]
    if not attempts:
        lines.append("No attempts on record for this (ticker, date).")
        return "\n".join(lines)

    for i, a in enumerate(attempts, start=1):
        alpha = f"{a.alpha_return:+.2%}" if a.alpha_return is not None else "n/a"
        raw = f"{a.raw_return:+.2%}" if a.raw_return is not None else "n/a"
        env = f"{'backtest' if a.is_backtest else 'live'}/{a.account_mode}"
        lines.append(
            f"[{i}] {a.created_at:%Y-%m-%d %H:%M:%S}  rating={a.rating}  status={a.status}  "
            f"alpha={alpha}  raw={raw}  env={env}"
        )
        if a.reflection:
            lines.append(f"    reflection: {a.reflection}")

    lines += [
        "",
        f"{len(attempts)} attempt(s) total — only the LATEST counts toward "
        "`python -m app.audit report`'s aggregate stats.",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Confidence calibration (Stage 1, F5, 2026-08-13)
#
# `PortfolioDecision.confidence` directly scales the actual capital committed
# via app.execution.sizing.confidence_size_modifier, but until the
# `confidence` column existed (see app.memory.db's own comment) nothing
# recorded it — so "did a high-confidence call actually do better than a
# low-confidence one" could never be checked after the fact. This answers
# exactly that: realized win rate / mean signed alpha, bucketed by the same
# confidence bands sizing.py already uses to decide position size.
# ---------------------------------------------------------------------------

# MUST mirror app.execution.sizing._CONFIDENCE_MODIFIER_BANDS exactly — kept
# as a literal copy here (not imported) per this stage's design, rather than
# importing from app.execution up into app.audit (a layering direction this
# codebase avoids elsewhere too — see app.memory.store's own comment on why
# it duplicates a small pure helper rather than cross-importing). A mismatch
# between the two would mean this report describes a DIFFERENT sizing scheme
# than the one actually executing trades — if sizing.py's bands ever change,
# this list must be updated in the same commit.
# app.execution.sizing의 confidence 밴드와 반드시 동일해야 함 — 한쪽만 바꾸면
# 리포트가 실제 사이징과 어긋난다.
_CONFIDENCE_BUCKETS: list[tuple[str, float, float]] = [
    ("[0.90, 1.00]", 0.90, 1.0),
    ("[0.75, 0.90)", 0.75, 0.9),
    ("[0.60, 0.75)", 0.60, 0.7),
    ("[0.50, 0.60)", 0.50, 0.5),
    ("<0.50 (skipped)", 0.0, 0.0),
]


@dataclass
class ConfidenceBucketStats:
    label: str
    lower: float
    size_multiplier: float
    n: int
    win_rate: float | None
    mean_signed_alpha: float | None
    unscorable_n: int


@dataclass
class ConfidenceCalibration:
    buckets: list[ConfidenceBucketStats]
    n_with_confidence: int
    n_without_confidence: int
    signed_alpha_corr: float | None


async def compute_confidence_calibration(
    ticker: str | None = None, market: str | None = None, scope: str | None = None,
    account_mode: str | None = None, trigger: str | None = None, basis: str = "variable",
    since: date | None = None, until: date | None = None,
) -> ConfidenceCalibration:
    """Realized win rate / mean SIGNED alpha per confidence bucket — the same
    filter/dedup/scoping conventions as `compute_hit_rate_by_rating` (see its
    own docstring for `market`/`scope`/`account_mode`/`trigger`/`basis`;
    identical validation and rationale, not repeated here).

    "Signed alpha" flips sign the same way `_WIN_DIRECTION` does: alpha as-is
    for Buy/Overweight/Hold-while-held (exposure increased or kept), negated
    for Sell/Underweight/Hold-while-flat (exposure decreased or avoided) — so
    a positive signed alpha always means "the call was right," regardless of
    direction, letting every rating share one number line for this report.

    Rows with `confidence IS NULL` (every row written before this column
    existed, plus any where the model didn't return a value) are excluded
    from every bucket entirely — counted once in `n_without_confidence`,
    never guessed into a bucket.

    Correlation (`signed_alpha_corr`, Postgres `corr(confidence,
    signed_alpha)`) is reported ACROSS buckets, over every scorable row with
    both a confidence and a determinable signed alpha — `None` if too few
    rows exist for Postgres to compute one (fewer than 2, or zero variance).
    This is a correlation, not causation, and is not adjusted for anything
    (market regime, position size, etc.) — see `render_confidence_
    calibration`'s own warning text.

    `since`/`until` (date, both optional, inclusive — Stage 2/weekly report,
    2026-08-13): same `horizon_eval_date` anchor, same None-means-no-filter
    semantics as `compute_hit_rate_by_rating`'s own docstring (not repeated
    here) — applies regardless of `basis`."""
    if market not in (None, "domestic", "overseas"):
        raise ValueError(f"market must be 'domestic', 'overseas', or None, got {market!r}")
    if scope not in (None, "live", "backtest"):
        raise ValueError(f"scope must be 'live', 'backtest', or None, got {scope!r}")
    if account_mode not in (None, "paper", "real"):
        raise ValueError(f"account_mode must be 'paper', 'real', or None, got {account_mode!r}")
    if trigger not in (None, "auto", "manual"):
        raise ValueError(f"trigger must be 'auto', 'manual', or None, got {trigger!r}")
    if basis not in ("variable", "horizon"):
        raise ValueError(f"basis must be 'variable' or 'horizon', got {basis!r}")
    alpha_col, raw_col = ("alpha_return", "raw_return") if basis == "variable" else (
        "horizon_alpha_return", "horizon_raw_return",
    )

    conditions = ["status = 'resolved'"]
    params: list[str] = []
    if ticker is not None:
        params.append(ticker)
        conditions.append(f"ticker = ${len(params)}")
    if market == "domestic":
        conditions.append("ticker LIKE '%.KS'")
    elif market == "overseas":
        conditions.append("ticker NOT LIKE '%.KS'")
    if scope == "live":
        conditions.append("is_backtest = false")
    elif scope == "backtest":
        conditions.append("is_backtest = true")
    if account_mode is not None:
        params.append(account_mode)
        conditions.append(f"account_mode = ${len(params)}")
    if trigger == "auto":
        conditions.append("decision_trigger IS DISTINCT FROM 'manual'")
    elif trigger == "manual":
        conditions.append("decision_trigger = 'manual'")
    if since is not None:
        params.append(since)
        conditions.append(f"horizon_eval_date >= ${len(params)}")
    if until is not None:
        params.append(until)
        conditions.append(f"horizon_eval_date <= ${len(params)}")
    # analysts_failed (app.memory.db column comment: "애널리스트 4명 중 실패한
    # 수 — 축소 분석과 정상 분석이 통계에서 섞이던 문제") — exclude only rows
    # explicitly marked degraded (>0); NULL predates this column and stays
    # included, since "unknown" isn't evidence of degradation.
    conditions.append("(analysts_failed IS NULL OR analysts_failed = 0)")
    # Same per-user scoping rationale as compute_hit_rate_by_rating above.
    params.append(get_current_user_id())
    conditions.append(f"user_id = ${len(params)}")
    where_clause = " AND ".join(conditions)

    common_ctes = rf"""
        latest AS (
            SELECT DISTINCT ON (ticker, trade_date)
                rating, {alpha_col} AS alpha_return, {raw_col} AS raw_return, confidence,
                held_at_decision, held_at_decision_inferred,
                -- 2026-08-20: recovers the model's ORIGINAL Buy/Sell call when
                -- `_apply_hysteresis_gate` downgraded it to Hold — see
                -- compute_hit_rate_by_rating's `latest` CTE for the full
                -- rationale/comment, copied verbatim here.
                COALESCE(
                    substring(decision_text from '\[Hysteresis gate: downgraded from (\w+) to Hold'),
                    substring(
                        full_transcript->>'investment_thesis'
                        from '\[Hysteresis gate: downgraded from (\w+) to Hold'
                    )
                ) AS gate_orig_rating
            FROM memory_entries
            WHERE {where_clause}
            ORDER BY ticker, trade_date, created_at DESC
        ), scored AS (
            SELECT
                confidence,
                CASE
                    WHEN gate_orig_rating = 'Buy' THEN alpha_return
                    WHEN gate_orig_rating = 'Sell' THEN -alpha_return
                    WHEN rating IN ('Buy', 'Overweight') THEN alpha_return
                    WHEN rating IN ('Sell', 'Underweight') THEN -alpha_return
                    WHEN rating = 'Hold' AND COALESCE(held_at_decision, held_at_decision_inferred) IS TRUE
                        THEN raw_return
                    WHEN rating = 'Hold' AND COALESCE(held_at_decision, held_at_decision_inferred) IS FALSE
                        THEN -alpha_return
                    ELSE NULL
                END AS signed_alpha
            FROM latest
            WHERE confidence IS NOT NULL
        )
    """

    pool = await get_pool()
    async with pool.acquire() as conn:
        bucket_rows = await conn.fetch(
            f"""
            WITH {common_ctes}, bucketed AS (
                SELECT
                    CASE
                        WHEN confidence >= 0.90 THEN '{_CONFIDENCE_BUCKETS[0][0]}'
                        WHEN confidence >= 0.75 THEN '{_CONFIDENCE_BUCKETS[1][0]}'
                        WHEN confidence >= 0.60 THEN '{_CONFIDENCE_BUCKETS[2][0]}'
                        WHEN confidence >= 0.50 THEN '{_CONFIDENCE_BUCKETS[3][0]}'
                        ELSE '{_CONFIDENCE_BUCKETS[4][0]}'
                    END AS bucket,
                    signed_alpha,
                    -- Same NaN-as-a-win guard as compute_hit_rate_by_rating's
                    -- `ok` filter above — signed_alpha can be NaN for the same
                    -- vendor-bad-close reason alpha_return can.
                    signed_alpha IS NOT NULL AND signed_alpha <> 'NaN'::float8 AS ok
                FROM scored
            )
            SELECT
                bucket,
                COUNT(*) FILTER (WHERE ok) AS n,
                COUNT(*) FILTER (WHERE NOT ok) AS unscorable,
                COUNT(*) FILTER (WHERE ok AND signed_alpha > 0) AS wins,
                AVG(signed_alpha) FILTER (WHERE ok) AS mean_signed_alpha
            FROM bucketed
            GROUP BY bucket
            """,
            *params,
        )
        totals_rows = await conn.fetch(
            f"""
            WITH {common_ctes}
            SELECT
                (SELECT COUNT(*) FROM latest) AS total,
                (SELECT COUNT(*) FROM scored) AS with_confidence,
                (
                    SELECT corr(confidence, signed_alpha) FROM scored
                    WHERE signed_alpha IS NOT NULL AND signed_alpha <> 'NaN'::float8
                        AND confidence IS NOT NULL
                ) AS alpha_corr
            """,
            *params,
        )

    by_bucket = {row["bucket"]: row for row in bucket_rows}
    buckets = [
        ConfidenceBucketStats(
            label=label,
            lower=lower,
            size_multiplier=size_multiplier,
            n=(by_bucket[label]["n"] if label in by_bucket else 0),
            win_rate=(
                by_bucket[label]["wins"] / by_bucket[label]["n"]
                if label in by_bucket and by_bucket[label]["n"]
                else None
            ),
            mean_signed_alpha=(
                float(by_bucket[label]["mean_signed_alpha"])
                if label in by_bucket and by_bucket[label]["mean_signed_alpha"] is not None
                else None
            ),
            unscorable_n=(by_bucket[label]["unscorable"] if label in by_bucket else 0),
        )
        for label, lower, size_multiplier in _CONFIDENCE_BUCKETS
    ]

    totals = totals_rows[0] if totals_rows else {"total": 0, "with_confidence": 0, "alpha_corr": None}
    n_with_confidence = totals["with_confidence"] or 0
    n_without_confidence = (totals["total"] or 0) - n_with_confidence
    signed_alpha_corr = float(totals["alpha_corr"]) if totals["alpha_corr"] is not None else None

    return ConfidenceCalibration(
        buckets=buckets,
        n_with_confidence=n_with_confidence,
        n_without_confidence=n_without_confidence,
        signed_alpha_corr=signed_alpha_corr,
    )


def render_confidence_calibration(
    stats: ConfidenceCalibration, ticker: str | None = None, market: str | None = None,
    run_scope: str | None = None, account_mode: str | None = None, run_trigger: str | None = None,
    basis: str = "variable",
) -> str:
    scope = ticker if ticker else ("all tickers" if market is None else f"all {market} tickers")
    scope = f"{scope}, {run_scope}" if run_scope else scope
    scope = f"{scope}, {account_mode}" if account_mode else scope
    scope = f"{scope}, {run_trigger}" if run_trigger else scope
    title = "# Self-Audit — confidence calibration"
    if basis == "horizon":
        title += " (fixed 10-trading-day horizon)"
    lines = [f"{title} ({scope})", ""]

    if stats.n_with_confidence == 0:
        lines.append(
            f"No decisions with a recorded confidence yet ({stats.n_without_confidence} resolved "
            "row(s) predate the confidence column, added 2026-08-13) — nothing to calibrate."
        )
        return "\n".join(lines)

    lines.append(
        f"{'Confidence':<16} {'Size x':>7} {'N':>4} {'Win rate':>10} {'Mean signed alpha':>18}"
    )
    lines.append("-" * 60)
    for b in stats.buckets:
        win = f"{b.win_rate:>9.1%}" if b.win_rate is not None else f"{'—':>9}"
        alpha = f"{b.mean_signed_alpha:>+17.2%}" if b.mean_signed_alpha is not None else f"{'—':>17}"
        lines.append(f"{b.label:<16} {b.size_multiplier:>6.1f}x {b.n:>4} {win} {alpha}")

    corr_text = f"{stats.signed_alpha_corr:+.3f}" if stats.signed_alpha_corr is not None else "N/A (too few scorable rows)"
    lines += [
        "",
        f"Confidence <-> signed-alpha correlation: {corr_text}",
        f"Excluded — no confidence recorded (predates the 2026-08-13 column): {stats.n_without_confidence}",
        "",
        "'Signed alpha' flips sign the same way win-rate scoring does: positive means the call was",
        "right, regardless of direction (Buy/Overweight/Hold-held score alpha as-is; Sell/Underweight/",
        "Hold-flat score -alpha) — see app.audit.stats._WIN_DIRECTION.",
        "Size multiplier is what app.execution.sizing.confidence_size_modifier actually applies at that",
        "confidence — this must be read alongside that function; if the two ever disagree, one of them",
        "is stale.",
        "Correlation is NOT causation, and is meaningless on a small sample — treat it as a hint to",
        "investigate, not a conclusion.",
        "The '<0.50 (skipped)' bucket was NEVER EXECUTED (sizing.py skips the trade entirely below 0.50)"
        " — its win rate is a counterfactual: what would have happened if it had been executed anyway.",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# rating vs action_label gap — 2026-09-01 audit #10. When the AI's `rating`
# is directional (Buy/Sell/...) but `action_label` came out Hold, nothing was
# traded. The only pre-existing check was a throwaway validator
# (tiered_screen_validation.py) that looked the other direction. This answers
# "how often, and — from the persisted [EXECUTION NOTE] — why".
# ---------------------------------------------------------------------------

# (rating, action_label) pairs that mean the same thing — kept in lockstep
# with signals.js `_RATING_ACTION_SAME`; a change here needs the same change
# there. Anything not in this set (with a non-null action_label) is a gap.
_ALIGNED_RATING_ACTION = frozenset({
    ("Buy", "Buy"), ("Sell", "Sell"), ("Hold", "Hold"),
    ("Overweight", "Increase"), ("Underweight", "Decrease"),
})
_EXECUTION_NOTE_RE = re.compile(r"\[EXECUTION NOTE:\s*([^\]]*)\]")
_NUM_RE = re.compile(r"\d[\d,]*(?:\.\d+)?%?")


@dataclass
class RatingActionGap:
    window_start: date
    window_end: date
    total_decisions: int = 0
    pair_counts: dict[tuple[str, str], int] = field(default_factory=dict)
    gap_pairs: set[tuple[str, str]] = field(default_factory=set)
    unrecorded_count: int = 0
    gap_count: int = 0
    gap_reason_counts: dict[str, int] = field(default_factory=dict)
    gap_rows: list[tuple[int, date, str, str, str | None, str | None]] = field(default_factory=list)


async def compute_rating_action_gap(
    market: str | None = None, scope: str | None = None, account_mode: str | None = None,
    trigger: str | None = None, days: int = 30, limit: int = 40,
) -> RatingActionGap:
    """Every decision whose `rating` disagrees with its `action_label` (a
    directional call that produced no trade), over the last `days` of
    `trade_date`. Same market/scope/account_mode/trigger filter and
    DISTINCT ON (ticker, trade_date) dedup as `compute_hit_rate_by_rating`.

    Deliberately NOT gated on `status = 'resolved'` (unlike the other reports
    here): a gap is visible the moment the decision is made, and the primary
    use of this report is checking the cycle that just ran. Also not gated on
    `analysts_failed`.

    The `why` breakdown reads the `[EXECUTION NOTE: ...]` text
    `app.memory.store.append_execution_note` appends to `decision_text`;
    numbers are normalized to `N` before grouping. A gap row with no note
    lands in `(no execution note)` — that count is the regression signal for
    the audit #10 fix (should trend to zero for post-deploy decisions)."""
    if market not in (None, "domestic", "overseas"):
        raise ValueError(f"market must be 'domestic', 'overseas', or None, got {market!r}")
    if scope not in (None, "live", "backtest"):
        raise ValueError(f"scope must be 'live', 'backtest', or None, got {scope!r}")
    if account_mode not in (None, "paper", "real"):
        raise ValueError(f"account_mode must be 'paper', 'real', or None, got {account_mode!r}")
    if trigger not in (None, "auto", "manual"):
        raise ValueError(f"trigger must be 'auto', 'manual', or None, got {trigger!r}")

    window_end = date.today()
    window_start = window_end - timedelta(days=days)
    conditions: list[str] = []
    params: list = []
    if market == "domestic":
        conditions.append("ticker LIKE '%.KS'")
    elif market == "overseas":
        conditions.append("ticker NOT LIKE '%.KS'")
    if scope == "live":
        conditions.append("is_backtest = false")
    elif scope == "backtest":
        conditions.append("is_backtest = true")
    if account_mode is not None:
        params.append(account_mode)
        conditions.append(f"account_mode = ${len(params)}")
    if trigger == "auto":
        conditions.append("decision_trigger IS DISTINCT FROM 'manual'")
    elif trigger == "manual":
        conditions.append("decision_trigger = 'manual'")
    params.append(window_start)
    conditions.append(f"trade_date >= ${len(params)}")
    params.append(get_current_user_id())
    conditions.append(f"user_id = ${len(params)}")
    where_clause = " AND ".join(conditions)

    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            f"""
            SELECT DISTINCT ON (ticker, trade_date)
                id, ticker, trade_date, rating, action_label, decision_text
            FROM memory_entries
            WHERE {where_clause}
            ORDER BY ticker, trade_date, created_at DESC
            """,
            *params,
        )

    result = RatingActionGap(window_start=window_start, window_end=window_end)
    reason_counter: Counter[str] = Counter()
    for row in rows:
        result.total_decisions += 1
        rating = row["rating"]
        action = row["action_label"]
        pair = (rating, action if action is not None else "(null)")
        result.pair_counts[pair] = result.pair_counts.get(pair, 0) + 1
        if action is None:
            result.unrecorded_count += 1
            result.gap_rows.append((row["id"], row["trade_date"], row["ticker"], rating, None, None))
            continue
        if (rating, action) in _ALIGNED_RATING_ACTION:
            continue
        result.gap_count += 1
        result.gap_pairs.add(pair)
        notes = _EXECUTION_NOTE_RE.findall(row["decision_text"] or "")
        joined = " | ".join(n.strip() for n in notes) if notes else None
        reason_counter[_NUM_RE.sub("N", joined) if joined else "(no execution note)"] += 1
        result.gap_rows.append((row["id"], row["trade_date"], row["ticker"], rating, action, joined))

    result.gap_reason_counts = dict(reason_counter)
    result.gap_rows.sort(key=lambda t: (t[1], t[0]), reverse=True)
    result.gap_rows = result.gap_rows[:limit]
    return result


def render_rating_action_gap(
    r: RatingActionGap, *, market: str | None = None, run_scope: str | None = None,
    account_mode: str | None = None, run_trigger: str | None = None,
) -> str:
    scope_bits = [
        market or "all markets", run_scope or "live+backtest",
        account_mode or "all accounts", run_trigger or "all triggers",
    ]
    lines = [
        f"# Self-Audit — rating vs action gap ({', '.join(scope_bits)})",
        "",
        f"Window: {r.window_start} .. {r.window_end} (trade_date)   Decisions (deduped): {r.total_decisions}",
        "",
    ]
    if r.total_decisions == 0:
        lines.append("No decisions in this window.")
        return "\n".join(lines)

    lines.append("## 1. rating -> action_label matrix")
    lines.append(f"  {'rating':<12} {'action_label':<14} {'n':>5}  gap")
    for (rating, action), n in sorted(r.pair_counts.items(), key=lambda kv: (-kv[1], kv[0])):
        mark = "!" if action == "(null)" else ("*" if (rating, action) in r.gap_pairs else "")
        lines.append(f"  {rating:<12} {action:<14} {n:>5}  {mark}")
    gap_total = r.gap_count + r.unrecorded_count
    pct = gap_total / r.total_decisions * 100 if r.total_decisions else 0.0
    lines += [
        f"GAP TOTAL: {gap_total} / {r.total_decisions} ({pct:.1f}%)  "
        f"[{r.gap_count} rating/action mismatch, {r.unrecorded_count} no action_label recorded]",
        "",
        "## 2. why (execution notes on gap rows, numbers normalized to N)",
    ]
    if not r.gap_reason_counts and not r.unrecorded_count:
        lines.append("  (no gaps)")
    for reason, count in sorted(r.gap_reason_counts.items(), key=lambda kv: -kv[1]):
        lines.append(f"  {count:>4}  {reason}")
    if r.unrecorded_count:
        lines.append(f"  {r.unrecorded_count:>4}  (no action_label recorded — rebalance metadata never persisted)")
    lines += ["", "## 3. gap decisions (most recent first)"]
    for _id, td, ticker, rating, action, note in r.gap_rows:
        act = action if action is not None else "(null)"
        lines.append(f"  id={_id}  {td}  {ticker:<12} {rating} -> {act}   {note or ''}".rstrip())
    lines += [
        "",
        "The '(no execution note)' count in block 2 is the regression signal for the 2026-09-01 audit",
        "#10 fix — for decisions made after deploy it should be 0. The gap COUNT itself stays non-zero",
        "until the affordability filter (audit #6) stops unbuyable names reaching the pipeline.",
    ]
    return "\n".join(lines)