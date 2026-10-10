"""Two-phase memory store: a pending entry at decision time, resolved later
once the realized return is known — same shape as TradingAgents' append-only
log, but with real cosine-similarity retrieval via pgvector instead of
recency-only lookup ("find situations like today," not "find the last N
times").

Every public function degrades quietly (log + no-op / empty result) if
Postgres isn't reachable — same "optional category" resilience convention as
`app/dataflows/router.py`, so a DB outage never breaks decision-making.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any
from zoneinfo import ZoneInfo

import subprocess

import asyncpg

from app.config import current_account_mode, settings
from app.memory.db import get_pool
from app.memory.embeddings import embed_async
from app.user_context import get_current_user_id

logger = logging.getLogger(__name__)

# Cached after the first call — a decision is written every LLM pipeline
# run (multiple times per auto_trade_loop cycle), and the commit this
# process has checked out cannot change without a restart, so re-running
# `git rev-parse` on every single decision would be pure overhead. `None`
# is cached too (not retried) — same reasoning as app.backtest.experiments.
# current_git_commit's own "best-effort, never worth blocking on" stance,
# just applied to a hot path instead of a once-per-run one.
#
# Deliberately duplicated rather than imported from app.backtest.
# experiments (the existing backtest_runs.git_commit tagging this mirrors)
# — app.memory is a lower layer that app.backtest itself depends on
# (app.backtest.harness imports app.memory.store/reflection), so importing
# back up to app.backtest from here would be a layering inversion even
# though it happens not to form an literal cycle today. Same "duplicate a
# small pure helper rather than cross-import" convention already used for
# the Market Literal in app.clock/app.execution.portfolio_risk.
_cached_git_commit: str | None = None
_git_commit_resolved = False


def _current_git_commit() -> str | None:
    """`settings.git_commit` (the GIT_COMMIT env var, baked into the image at
    build time) first, falling back to `git rev-parse` for a local checkout.

    The env var is not an optimization — it is the only path that works in
    the deployed container, where .dockerignore excludes .git and the slim
    base image has no git binary, so the subprocess below failed silently on
    every single deployed decision. See `settings.git_commit`'s own comment
    for the live-DB evidence. Kept in this order (env wins) so a local run
    from a checkout still self-identifies without any env setup.
    배포 환경에서는 환경변수만 동작하고, 로컬 체크아웃에서는 git 명령이 받쳐줌."""
    global _cached_git_commit, _git_commit_resolved
    if not _git_commit_resolved:
        if settings.git_commit:
            _cached_git_commit = settings.git_commit
        else:
            try:
                result = subprocess.run(
                    ["git", "rev-parse", "--short", "HEAD"],
                    capture_output=True, text=True, timeout=5, check=True,
                )
                _cached_git_commit = result.stdout.strip() or None
            except Exception:
                _cached_git_commit = None
        _git_commit_resolved = True
    return _cached_git_commit


async def store_pending(
    ticker: str,
    trade_date: date,
    situation_text: str,
    rating: str,
    decision_text: str,
    transcript: dict[str, Any] | None = None,
    is_backtest: bool = False,
    trader_action: str | None = None,
    confidence: float | None = None,
    held_at_decision: bool | None = None,
    card_reason_ko: str | None = None,
    card_reason_en: str | None = None,
    total_tokens: int | None = None,
    decision_trigger: str = "auto",
    analysts_failed: int | None = None,
) -> int | None:
    """Returns the new row's id (used to link a live chat answer or the
    dashboard's decision history back to its full transcript) — `None` if
    the write itself failed, same degrade-quietly convention as everything
    else here.

    `confidence` (Stage 1, 2026-08-13 — app.state.PortfolioDecision.
    confidence): the Portfolio Manager's own 0.0-1.0 conviction, read from
    the FINAL decision object (i.e. after `_apply_hysteresis_gate`, which
    preserves confidence via `model_copy` — so this is the same value that
    actually reached app.execution.sizing.confidence_size_modifier and
    scaled real capital). Both `None` (the model didn't return one) and
    `0.0` (max size cut / trade skipped) are legitimate and must stay
    distinguishable — see the column's own comment in app.memory.db.

    `held_at_decision` (Stage 1, 2026-08-13 — app.orchestrator.pipeline's
    `_position_context`): whether a position was already held immediately
    BEFORE this decision. Lets a future stats pass score a Hold rating in
    the correct direction (held -> wins on alpha > 0, flat -> wins on
    alpha < 0) instead of excluding every Hold from scoring — see the
    column's own comment in app.memory.db and app.audit.stats's
    `_WIN_DIRECTION`.

    `is_backtest` (app.orchestrator.pipeline passes `as_of is not None`):
    tags a historical-replay decision so `get_recent_entries` can exclude
    it from what the dashboard shows as recent activity, without excluding
    it from this table entirely — app.audit's tools and get_similar_context
    still need every row. See the column's own comment in app.memory.db.

    `trader_action` (app.state.TraderAction.value, e.g. "Buy"/"Hold"/"Sell"):
    the Trader's own proposal BEFORE the risk debate and Portfolio Manager
    run — recorded alongside the final `rating` so a future stats pass
    (app.audit.stats) can measure how often the debate actually changed the
    outcome, instead of only ever seeing what the Portfolio Manager decided.
    See the column's own comment in app.memory.db for why this can't be a
    naive equality check against `rating` (3-way vs 5-way).

    `git_commit`/`llm_model` (2026-08-12): the code+prompt version and the
    model that produced this decision, both derived automatically — no
    caller passes either. Without these, a model swap
    (project memory: llm-model-swap-gpt-oss-issue) or a prompt change
    (project memory: agents-prompt-quality-review) left no trace on any
    individual decision, only on whole backtest runs
    (app.backtest.experiments' identical git_commit tagging), so a
    before/after comparison of either kind of change was impossible once
    memory of exactly when it landed faded. See the columns' own comments
    in app.memory.db.

    `card_reason_ko`/`card_reason_en` (app.state.PortfolioDecision.
    card_reason_ko/card_reason_en): short, display-only "why" text for the
    dashboard signal card, in each UI language, kept separate from
    `decision_text`/`executive_summary` for the reasons in that field's own
    description. Both optional — a model that omits them (or an older row
    predating the columns) just leaves the card falling back to
    `decision_text`-derived text, never a hard failure.

    `total_tokens` (app.orchestrator.pipeline, via app.llm.metrics.
    current_run_tokens()): this one ticker/decision's total LLM token usage
    across every analyst/debate/judgment-node call in the run — display-only
    (the dashboard signal card's cost footer), never used for sizing/
    execution. `None` for any row predating this column, or if the
    contextvar-scoped accumulator somehow wasn't active.

    `decision_trigger` (app.orchestrator.run_pipeline, default "auto"):
    'auto' if the scheduled auto-trade loop's own screening picked this
    ticker, 'manual' if a human picked THIS SPECIFIC ticker out of band
    (the Trading tab's analyze button, a chat deep-dive question, or a CLI/
    dev-script run). Defaults to "auto" so the auto-loop call site needs no
    change; every manual entry point passes "manual" explicitly. Exists so
    app.audit's win-rate/confidence-calibration queries can exclude manual
    one-off analyses, which are sample-selection biased (a person only
    requests one when already curious/worried about that ticker) and would
    otherwise silently skew the numbers [[stage-gate-criteria-draft]] uses
    to gate capital increases. See the column's own comment in
    app.memory.db.

    `analysts_failed` (app.orchestrator.pipeline, 2026-08-25): how many of
    the 4 analysts (Market/Fundamentals/News+Macro/Sentiment) failed and
    were replaced by a placeholder report this cycle — 0 for a full run,
    1-3 for a degraded one that still produced a rating, `None` for any row
    predating this column (no signal survives to backfill it, same
    reasoning as `decision_trigger` above). Exists so app.audit's win-rate/
    confidence-calibration queries can filter a degraded decision out of
    (or report it separately from) a full-analysis one instead of pooling
    them — see the column's own comment in app.memory.db."""
    try:
        pool = await get_pool()
        embedding = await embed_async(situation_text)
        payload = json.dumps(transcript) if transcript is not None else None
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO memory_entries
                    (ticker, trade_date, situation_text, situation_embedding, rating, decision_text, full_transcript, user_id, is_backtest, trader_action, account_mode, git_commit, llm_model, confidence, held_at_decision, card_reason_ko, card_reason_en, total_tokens, decision_trigger, analysts_failed)
                VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20)
                RETURNING id
                """,
                ticker,
                trade_date,
                situation_text,
                embedding,
                rating,
                decision_text,
                payload,
                get_current_user_id(),
                is_backtest,
                trader_action,
                current_account_mode(),
                _current_git_commit(),
                settings.llm_model,
                confidence,
                held_at_decision,
                card_reason_ko,
                card_reason_en,
                total_tokens,
                decision_trigger,
                analysts_failed,
            )
        return row["id"]
    except Exception as exc:  # noqa: BLE001 - (OSError, PostgresError) missed
        # asyncpg.InterfaceError (pool closing/contended — not a PostgresError
        # subclass) and any other read/write failure; all mean the same thing
        # for this "degrade quietly" module: log and continue, don't crash.
        # PostgresError만으로는 못 잡는 InterfaceError 등도 조용히 무시하도록 확장.
        logger.warning("memory store_pending failed, continuing without it: %s", exc)
        return None


async def find_recent_pending_id(ticker: str, trade_date: date, situation_text: str) -> int | None:
    """Looks for a `memory_entries` row that `store_pending` may have already
    committed for this exact (ticker, trade_date, situation_text) before an
    unrequested `asyncio.CancelledError` (see id=377/id=949 in
    app.orchestrator.pipeline's docstrings) surfaced after the INSERT but
    before its result reached the caller — i.e. the write actually
    succeeded, only the confirmation was lost. A caller that hits that
    exact shape calls this BEFORE retrying `store_pending`, so a spurious
    cancellation can't turn one decision into two rows.

    Scoped to the last 5 minutes (`created_at`) — situation_text is
    deterministic for a given (ticker, trade_date) within one pipeline run,
    but a much older row with the same text (a prior day's re-run of the
    same backtest date, for instance) is not the write this call just
    almost lost, so it must not be picked up here. `None` on no match, same
    convention as `store_pending`'s own `None` return; also degrades
    quietly on a DB failure since a caller falling back to a plain retry is
    always safe."""
    try:
        pool = await get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT id FROM memory_entries
                WHERE ticker = $1 AND trade_date = $2 AND situation_text = $3
                    AND user_id = $4 AND account_mode = $5
                    AND created_at > now() - interval '5 minutes'
                ORDER BY id DESC
                LIMIT 1
                """,
                ticker,
                trade_date,
                situation_text,
                get_current_user_id(),
                current_account_mode(),
            )
        return row["id"] if row is not None else None
    except Exception as exc:  # noqa: BLE001 - same fail-open convention as store_pending
        logger.warning("memory find_recent_pending_id failed, continuing without it: %s", exc)
        return None


async def get_transcript(decision_id: int) -> dict[str, Any] | None:
    """Full agent-by-agent transcript for one past decision, for the
    dashboard's "see how the agents got here" view — NOT used by the
    pipeline, so (same reasoning as `get_recent_entries`) this deliberately
    does not degrade quietly on a DB failure."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT ticker, trade_date, rating, decision_text, full_transcript FROM memory_entries "
            "WHERE id = $1 AND user_id = $2",
            decision_id,
            get_current_user_id(),
        )
    if row is None:
        return None
    return {
        "ticker": row["ticker"],
        "trade_date": row["trade_date"].isoformat(),
        "rating": row["rating"],
        "decision_text": row["decision_text"],
        "transcript": json.loads(row["full_transcript"]) if row["full_transcript"] else None,
    }


async def resolve_outcome(
    decision_id: int,
    ticker: str,
    trade_date: date,
    raw_return: float,
    alpha_return: float,
    holding_days: int,
    reflection: str,
    benchmark_ticker: str,
    *,
    resolved_on: date,
) -> None:
    """Identified by `id` rather than `(ticker, trade_date)` — same reason as
    `set_horizon_outcome` right below (read that function's own docstring
    first): more than one row can share the same (ticker, trade_date), e.g.
    a same-day Sell followed by a fresh Buy on the same ticker ~21 minutes
    apart (id=949/950160.KS, 2026-08-21 — see
    `get_same_day_sibling_context`'s own docstring in this module for that
    incident). Updating by the natural key used to let ONE row's reflection
    (computed from whichever row the caller happened to be resolving) get
    written onto EVERY pending row matching that (ticker, trade_date) —
    including, since neither this query nor the caller filtered on
    `is_backtest`, a live pending row getting clobbered by a backtest
    replay resolving the same natural key, or vice versa.

    `ticker`/`trade_date` are still accepted (used only in the caller's own
    warning log on failure, kept here as an extra AND-condition for
    defense-in-depth) but `id` is what actually scopes the UPDATE to exactly
    one row.

    `benchmark_ticker` (2026-08-12): records which index `alpha_return`
    was actually computed against. Before this, the column existed
    (`memory_entries.benchmark_ticker`, defaulting to 'SPY') but nothing
    ever wrote it, so it silently claimed 'SPY' for every row regardless of
    what was actually used — including the 144 domestic decisions
    `app.memory.auto_resolve` had, until today, genuinely scored against
    SPY (see project memory: benchmark-mismatch-fixed) and the ones
    correctly re-scored against KOSPI afterward. Both would have shown
    'SPY' in this column either way. Required (no default) — both real
    callers (`app.memory.auto_resolve.resolve_ticker` and
    `app.backtest.harness`) always pass it now, so the old "no
    benchmark_ticker" fallback branch (which left the column untouched) is
    dead code and has been removed.

    `resolved_on` (2026-09-01, keyword-only, required — no default): the
    market date this outcome actually became knowable, i.e. `eval_date` for
    a live resolution or the replayed `trade_date` for a backtest one — see
    `memory_entries.resolved_on`'s own column comment (app.memory.db) for
    why this must be a caller-supplied simulated date, never `local_today()`
    or any other wall-clock value. No default because there is no sound
    one: guessing `trade_date + holding_days` here would just silently
    re-derive what the caller already computed to produce `eval_date` in
    the first place, hiding a caller that forgot to pass the real value
    behind an answer that happens to be numerically identical only because
    `holding_days` IS `resolved_on - trade_date` by definition. This is the
    write side of `get_similar_context`'s point-in-time bound — without it,
    a backtest replaying a historical date could retrieve THIS row's
    outcome/reflection before, in simulated time, it existed."""
    try:
        pool = await get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE memory_entries
                SET raw_return = $4, alpha_return = $5, holding_days = $6,
                    reflection = $7, benchmark_ticker = $8, resolved_on = $9, status = 'resolved'
                WHERE id = $1 AND ticker = $2 AND trade_date = $3 AND status = 'pending'
                """,
                decision_id, ticker, trade_date, raw_return, alpha_return, holding_days, reflection,
                benchmark_ticker, resolved_on,
            )
    except Exception as exc:  # noqa: BLE001 - same broadened fail-open policy as store_pending above
        logger.warning(
            "memory resolve_outcome failed for id=%s (%s %s), continuing without it: %s",
            decision_id, ticker, trade_date, exc,
        )


async def set_horizon_outcome(
    decision_id: int,
    horizon_raw_return: float,
    horizon_alpha_return: float,
    horizon_trading_days: int,
    horizon_eval_date: date,
) -> None:
    """Fills in the fixed-trading-day-horizon columns (Stage 1, 2026-08-13 —
    see their own comments in app.memory.db) for an ALREADY-resolved row,
    identified by `id` rather than `(ticker, trade_date)` — unlike
    `resolve_outcome`, more than one row can share the same (ticker,
    trade_date) (repeated backtest runs of the same case; see this module's
    own dedup convention in app.audit.stats), so updating by the natural key
    could silently touch the wrong row or several at once.

    Deliberately does NOT touch `status` (already 'resolved') and does NOT
    re-run `resolve_outcome` or `reflect()` — this is purely an additional
    measurement on top of a decision that's already fully resolved.

    Re-validates both figures are finite immediately before writing (on top
    of whatever check the caller already did) — the exact NaN-as-a-win bug
    this project already hit once for `alpha_return`/`raw_return` (see
    `app.memory.auto_resolve._usable`'s own docstring) would reappear here
    if a NaN close ever made it this far, so this is a second guard rather
    than trusting the caller alone."""
    if not (math.isfinite(horizon_raw_return) and math.isfinite(horizon_alpha_return)):
        logger.warning(
            "set_horizon_outcome: refusing to store non-finite return(s) for id=%s (raw=%s, alpha=%s)",
            decision_id, horizon_raw_return, horizon_alpha_return,
        )
        return
    try:
        pool = await get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE memory_entries
                SET horizon_raw_return = $2, horizon_alpha_return = $3,
                    horizon_trading_days = $4, horizon_eval_date = $5
                WHERE id = $1
                """,
                decision_id,
                horizon_raw_return,
                horizon_alpha_return,
                horizon_trading_days,
                horizon_eval_date,
            )
    except Exception as exc:  # noqa: BLE001 - same broadened fail-open policy as resolve_outcome above
        logger.warning("memory set_horizon_outcome failed for id=%s: %s", decision_id, exc)


async def flag_bad_entry(decision_id: int, reason: str) -> None:
    """Excludes a row from future `get_similar_context` retrieval without
    deleting it (keeps the row itself as an audit trail) — for a decision
    later found to rest on a false premise, e.g. the pre-fix ticker-identity
    hallucination bug (see `market_analyst.run`'s docstring), so a
    hallucinated "lesson" already embedded and stored stops being cited as
    if it were validated fact in future retrieval."""
    try:
        pool = await get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE memory_entries
                SET flagged_bad = TRUE, reflection = COALESCE(reflection, '') || $2
                WHERE id = $1
                """,
                decision_id,
                f"\n\n[FLAGGED BAD: {reason}]",
            )
    except Exception as exc:  # noqa: BLE001 - same broadened fail-open policy as store_pending above
        logger.warning("memory flag_bad_entry failed for id=%s: %s", decision_id, exc)


async def append_execution_note(decision_id: int, note: str) -> None:
    """Appends a factual note to a stored decision's text without altering
    its reasoning — for when execution-time code (e.g. the 1% risk cap or
    the no-resolvable-stop guard in `app.execution.ledger.rebalance_to_target`)
    changes the trade below what the decision's own narrative described.
    Without this, a stale pre-execution narrative persists as the permanent
    record and can resurface via `get_similar_context` in a *future*
    decision's past context, compounding the mismatch instead of just
    misdescribing the current one."""
    try:
        pool = await get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE memory_entries
                SET decision_text = decision_text || $2
                WHERE id = $1
                """,
                decision_id,
                f"\n\n[EXECUTION NOTE: {note}]",
            )
    except Exception as exc:  # noqa: BLE001 - same broadened fail-open policy as store_pending above
        logger.warning("memory append_execution_note failed for id=%s: %s", decision_id, exc)


async def set_action_label(decision_id: int, action_label: str) -> None:
    """Records what the rebalance ACTUALLY did (Buy/Sell/Increase/Decrease/
    Hold — app.execution.sizing.rebalance_action_label) alongside the
    decision's own `rating` column, which is a different, unrelated
    concept (see that function's docstring). Same degrade-quietly
    convention as append_execution_note — a DB hiccup here must not break
    the pipeline that already recorded the decision and placed the trade."""
    try:
        pool = await get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                "UPDATE memory_entries SET action_label = $2 WHERE id = $1",
                decision_id,
                action_label,
            )
    except Exception as exc:  # noqa: BLE001 - same broadened fail-open policy as append_execution_note above
        logger.warning("memory set_action_label failed for id=%s: %s", decision_id, exc)


async def get_similar_context(
    ticker: str,
    situation_text: str,
    as_of: date | None = None,
    n_same: int | None = None,
    n_cross: int | None = None,
) -> str:
    """`as_of` excludes any entry dated after it — without this bound, a
    backtest replaying an early historical date could retrieve memory from
    entries dated later (stored by a different, later run), which would leak
    future information into a historical decision exactly like the raw data
    layer's point-in-time bound prevents (see `app/dataflows/*_provider.py`).
    `None` (live mode) applies no bound.

    LIVE (`as_of is None`) also excludes `is_backtest` rows entirely
    (2026-08-04) — this query had no such filter at all before, unlike
    every other consumer of this table (get_recent_entries excludes them
    for the dashboard, app.audit's stats keep them separate too — see
    project memory: pm-risk-debate-synthesis-bug-fixed's sibling finding).
    A backtest run happening to overlap with live meant its fictional,
    historical-replay decisions could surface as "past lessons" in the live
    Portfolio Manager's own reasoning. Backtesting (`as_of` set) keeps
    seeing `is_backtest` rows unfiltered — a replay is expected to learn
    from its own earlier-round decisions in the same run.

    TWO bounds, not one, are needed to keep that temporally sound, and
    until 2026-09-01 only the first existed: `trade_date <= as_of` bounds
    when the DECISION was made, but `status='resolved'` +
    `alpha_return IS NOT NULL` + `reflection` only become true once
    `app.memory.auto_resolve.resolve_ticker` /
    `app.backtest.harness._process_ticker_date` grade the outcome, which
    happens `holding_days` (>= `settings.memory_min_holding_days`, unbounded
    above — 5 to 112+ calendar days observed live) LATER. A replay of date T
    could therefore read a graded outcome and a reflection/lesson that, at
    T, had not happened yet — a look-ahead leak into the Portfolio
    Manager's own context, confounding exactly the backtest alpha numbers
    [[stage-gate-criteria-draft]] gates capital on.
    `resolved_on < as_of` (STRICT, not `<=`: the harness resolves a row
    AFTER running that date's own pipeline, and a multi-ticker replay's
    rotation resolves one ticker's row mid-date while another ticker still
    has to run at the same `as_of`) is the second bound — see
    `memory_entries.resolved_on`'s column comment (app.memory.db) for its
    exact semantics. `resolved_on IS NOT NULL` is required alongside it:
    fail-closed for the (should-be-empty, see the resolved_on backfill)
    set of pre-existing rows with no recoverable resolution date, rather
    than treating a missing bound as "resolved long ago."
    LIVE (`as_of is None`) applies neither the `trade_date` nor the
    `resolved_on` bound and is byte-identically unchanged by this — "resolved
    by now" is the correct and only available answer when now IS now.
    백테스트가 과거 날짜 T를 재생할 때 아직 채점되지 않았어야 할 미래 결과/리플렉션을
    읽던 문제 — 결정 시점(trade_date)뿐 아니라 채점 시점(resolved_on)도 함께 잘라야
    시점 정합성이 성립한다. 라이브는 두 바운드 모두 적용 안 됨(영향 없음)."""
    n_same = settings.memory_n_same if n_same is None else n_same
    n_cross = settings.memory_n_cross if n_cross is None else n_cross
    live_only_clause = "AND NOT is_backtest" if as_of is None else ""
    # 2026-09-01: the second point-in-time bound (see this function's own
    # docstring) — reuses $4 (as_of), no new bind parameter. Absent in live
    # mode, so the live query string is unchanged by this fix.
    pit_resolution_clause = "" if as_of is None else "AND resolved_on IS NOT NULL AND resolved_on < $4"

    # Domestic/overseas separation (2026-07-27): cross-ticker retrieval used
    # to rank purely by embedding similarity with no market awareness at
    # all, so a KRX ticker's "similar situations" could surface US tickers'
    # past cases and vice versa. Same-market matches are now fetched first
    # and take priority; a cross-market match is only pulled in to fill
    # remaining slots when there aren't enough same-market ones (user's
    # explicit choice: "같은 시장 우선 + 보조 허용", not a hard block) — see
    # docs/domestic-overseas-separation-plan.md.
    is_domestic = ticker.endswith(".KS")
    same_market_clause = "ticker LIKE '%.KS'" if is_domestic else "ticker NOT LIKE '%.KS'"
    other_market_clause = "ticker NOT LIKE '%.KS'" if is_domestic else "ticker LIKE '%.KS'"

    user_id = get_current_user_id()
    try:
        pool = await get_pool()
        embedding = await embed_async(situation_text)
        async with pool.acquire() as conn:
            account_mode = current_account_mode()
            max_distance = settings.memory_similarity_max_distance
            same_rows = await conn.fetch(
                f"""
                SELECT trade_date, rating, decision_text, reflection, raw_return, alpha_return
                FROM memory_entries
                WHERE ticker = $1 AND status = 'resolved' AND NOT flagged_bad
                    AND ($4::date IS NULL OR trade_date <= $4)
                    {pit_resolution_clause}
                    AND user_id = $5
                    AND account_mode = $6
                    AND raw_return IS NOT NULL AND raw_return <> 'NaN'::float8
                    AND alpha_return IS NOT NULL AND alpha_return <> 'NaN'::float8
                    AND situation_embedding <=> $2 <= $7
                    {live_only_clause}
                ORDER BY situation_embedding <=> $2
                LIMIT $3
                """,
                ticker,
                embedding,
                n_same,
                as_of,
                user_id,
                account_mode,
                max_distance,
            )
            cross_rows_same_market = await conn.fetch(
                f"""
                SELECT ticker, trade_date, rating, reflection, alpha_return
                FROM memory_entries
                WHERE ticker != $1 AND {same_market_clause} AND status = 'resolved' AND NOT flagged_bad
                    AND ($4::date IS NULL OR trade_date <= $4)
                    {pit_resolution_clause}
                    AND user_id = $5
                    AND account_mode = $6
                    AND alpha_return IS NOT NULL AND alpha_return <> 'NaN'::float8
                    AND situation_embedding <=> $2 <= $7
                    {live_only_clause}
                ORDER BY situation_embedding <=> $2
                LIMIT $3
                """,
                ticker,
                embedding,
                n_cross,
                as_of,
                user_id,
                account_mode,
                max_distance,
            )
            cross_rows_other_market = []
            remaining = n_cross - len(cross_rows_same_market)
            if remaining > 0:
                cross_rows_other_market = await conn.fetch(
                    f"""
                    SELECT ticker, trade_date, rating, reflection, alpha_return
                    FROM memory_entries
                    WHERE ticker != $1 AND {other_market_clause} AND status = 'resolved' AND NOT flagged_bad
                        AND ($4::date IS NULL OR trade_date <= $4)
                        {pit_resolution_clause}
                        AND user_id = $5
                        AND account_mode = $6
                        AND alpha_return IS NOT NULL AND alpha_return <> 'NaN'::float8
                        AND situation_embedding <=> $2 <= $7
                        {live_only_clause}
                    ORDER BY situation_embedding <=> $2
                    LIMIT $3
                    """,
                    ticker,
                    embedding,
                    remaining,
                    as_of,
                    user_id,
                    account_mode,
                    max_distance,
                )
    except Exception as exc:  # noqa: BLE001 - same broadened fail-open policy as store_pending above
        logger.warning("memory get_similar_context failed, continuing without past context: %s", exc)
        return ""

    if not same_rows and not cross_rows_same_market and not cross_rows_other_market:
        return ""

    parts = []
    if same_rows:
        parts.append(f"Past analyses of {ticker} with a similar setup (most similar first):")
        parts.extend(_format_same_ticker(r) for r in same_rows)
    if cross_rows_same_market:
        parts.append("Similar situations from other tickers (same market):")
        parts.extend(_format_cross_ticker(r) for r in cross_rows_same_market)
    if cross_rows_other_market:
        parts.append(
            "Similar situations from other tickers (different market — shown only because "
            "too few same-market matches were found):"
        )
        parts.extend(_format_cross_ticker(r) for r in cross_rows_other_market)
    return "\n\n".join(parts)


def _format_same_ticker(row: asyncpg.Record) -> str:
    tag = f"[{row['trade_date']} | {row['rating']} | raw {row['raw_return']:+.1%} | alpha {row['alpha_return']:+.1%}]"
    parts = [tag, f"DECISION:\n{row['decision_text']}"]
    if row["reflection"]:
        parts.append(f"REFLECTION:\n{row['reflection']}")
    return "\n\n".join(parts)


def _format_cross_ticker(row: asyncpg.Record) -> str:
    tag = f"[{row['trade_date']} | {row['ticker']} | {row['rating']} | alpha {row['alpha_return']:+.1%}]"
    return f"{tag}\n{row['reflection'] or '(no reflection recorded)'}"


async def get_same_day_sibling_context(
    ticker: str, trade_date: date, as_of: date | None = None, language: str = "en"
) -> str:
    """A same-day, same-ticker decision made earlier THIS trade_date —
    deliberately status-agnostic (unlike `get_similar_context`'s `status =
    'resolved'` filter), so a still-`pending` sibling's reasoning is visible
    too. `status='resolved'`/`'pending'` there is a data-availability
    distinction (a pending row has no graded outcome/reflection yet, not a
    trust signal), and this function reads neither of those columns, so it
    doesn't defeat that filter's purpose — it's an entirely separate,
    additive lookup for a different question ("what did THIS SAME ticker's
    analysis conclude a few minutes/hours ago today"), not "what similar
    past setups resolved well."

    Closes the id=949 (950160.KS, 2026-08-21) gap: a strong Sell judgment
    (confidence 0.90, going-concern risk) was `status='pending'` when a
    fresh BUY was decided for the same ticker ~21 minutes later — not
    because that BUY disagreed and overrode it, but because
    `get_similar_context`'s resolved-only filter meant it structurally
    could not see it at all.

    LIVE ONLY (`as_of is None`) — a backtest replaying one historical date
    has no "earlier today" in any intraday sense (daily-granularity
    replay), and language_ab's isolation invariant requires every backtest
    field besides `past_context` to stay untouched by live-only retrieval.
    `settings.memory_sibling_context_n <= 0` also short-circuits before
    touching the pool (0 is the kill switch)."""
    limit = settings.memory_sibling_context_n
    if as_of is not None or limit <= 0:
        return ""

    user_id = get_current_user_id()
    try:
        pool = await get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT created_at, rating, confidence, action_label, decision_trigger,
                       card_reason_ko, card_reason_en, decision_text
                FROM memory_entries
                WHERE ticker = $1 AND trade_date = $2 AND NOT is_backtest AND NOT flagged_bad
                    AND user_id = $3 AND account_mode = $4
                ORDER BY created_at DESC
                LIMIT $5
                """,
                ticker,
                trade_date,
                user_id,
                current_account_mode(),
                limit,
            )
    except Exception as exc:  # noqa: BLE001 - same broadened fail-open policy as get_similar_context above
        logger.warning("memory get_same_day_sibling_context failed, continuing without it: %s", exc)
        return ""

    if not rows:
        return ""

    header = (
        "Earlier decisions on THIS SAME ticker today (outcome not yet known — these are "
        "this system's own prior judgments from a few minutes/hours ago, not graded lessons):"
    )
    return "\n\n".join([header] + [_format_sibling(r, ticker, language) for r in rows])


def _format_sibling(row: asyncpg.Record, ticker: str, language: str) -> str:
    tz = ZoneInfo("Asia/Seoul") if ticker.endswith(".KS") else ZoneInfo("America/New_York")
    local_time = row["created_at"].astimezone(tz).strftime("%H:%M")
    confidence = f"confidence {row['confidence']:.2f}" if row["confidence"] is not None else "confidence n/a"
    execution = row["action_label"] or "execution outcome not recorded"
    trigger_note = " | manual (human-requested, not the scheduled system)" if row["decision_trigger"] == "manual" else ""
    tag = f"[{local_time} | {row['rating']} | {confidence} | {execution}{trigger_note}]"
    body = (
        (row["card_reason_ko"] if language == "ko" else row["card_reason_en"])
        or row["card_reason_en"]
        or row["card_reason_ko"]
        or (row["decision_text"] or "")[:400]
    )
    return f"{tag}\nWHY: {body}"


async def tickers_analyzed_on(trade_date: date) -> set[str]:
    """Distinct tickers with at least one decision already recorded for
    `trade_date` — app.live.auto_trade_loop's manual "다시 분석" trigger
    uses this to exclude them from the candidate pool, so re-clicking
    surfaces different tickers instead of the same momentum leaders every
    time. Same "no silent degrade" convention as get_recent_entries (an
    empty result here would silently defeat the whole point of excluding
    anything), so a DB failure raises rather than returning an empty set."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT DISTINCT ticker FROM memory_entries WHERE trade_date = $1 AND user_id = $2",
            trade_date,
            get_current_user_id(),
        )
    return {row["ticker"] for row in rows}


async def tickers_with_pending_resolution(cutoff: date) -> set[str]:
    """Distinct LIVE tickers with at least one still-pending decision whose
    `trade_date` is old enough (<= cutoff) to resolve — feeds
    app.memory.auto_resolve's scheduler, which walks every such ticker once
    per cycle. `is_backtest = false` excludes historical-replay rows: the
    backtest harness already resolves its own pending entries as it walks
    forward through known future dates, and must never have this live
    scheduler resolve them a second time (or a live-computed reflection
    leak into what should be a point-in-time historical replay). Same
    "raise rather than silently return empty" convention as
    `tickers_analyzed_on` — an empty result here would silently mean
    nothing ever gets auto-resolved, which is exactly the bug this
    function exists to prevent."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT DISTINCT ticker FROM memory_entries "
            "WHERE status = 'pending' AND trade_date <= $1 AND is_backtest = false AND user_id = $2",
            cutoff,
            get_current_user_id(),
        )
    return {row["ticker"] for row in rows}


@dataclass
class DecisionEntry:
    """One row from `memory_entries`, for display purposes (e.g. the
    dashboard) — not used by the pipeline itself."""

    id: int
    ticker: str
    trade_date: date
    rating: str
    status: str
    raw_return: float | None
    alpha_return: float | None
    reflection: str | None
    decision_text: str
    created_at: datetime
    # Buy/Sell/Increase/Decrease/Hold, or None if the rebalance hasn't run
    # (or predates this column) — see set_action_label's own docstring for
    # why this differs from `rating`.
    action_label: str | None = None
    # Short display-only "why" for the dashboard signal card, in each UI
    # language — see app.state's _PortfolioDecisionFields comment and the
    # card_reason_ko/card_reason_en column comments in app.memory.db. None
    # for rows predating these columns or where the model omitted them.
    card_reason_ko: str | None = None
    card_reason_en: str | None = None
    # Display-only LLM token-usage total for this decision — see
    # app.memory.db's total_tokens column comment. None for rows predating
    # this column.
    total_tokens: int | None = None
    # 'auto' or 'manual' — see app.memory.db's decision_trigger column
    # comment. None for rows predating this column; the frontend treats
    # anything that isn't literally 'manual' as auto.
    decision_trigger: str | None = None


async def get_recent_entries(
    ticker: str | None = None, limit: int = 50, include_backtest: bool = False
) -> list[DecisionEntry]:
    """Most recent decisions (pending or resolved), for on-demand reporting
    (the live dashboard, via `app.web.routers.chat_decisions`) — NOT used by
    the pipeline, so unlike every other function in this file, this
    deliberately does not degrade quietly on a DB failure. Same reasoning as
    `app/audit/stats.py`: a report silently showing "no data" instead of
    "couldn't connect" would be misleading to whoever's reading it.

    `include_backtest=False` (the default, and what the dashboard caller
    uses): excludes historical-replay rows (see the is_backtest column's own
    comment in app.memory.db) so a backtest run never shows up looking like
    a real recent decision. app.audit's own queries read memory_entries
    directly, not through this function, and are unaffected either way —
    backtest rows are exactly what those tools report on.

    Always scoped to the CURRENT account_mode (2026-08-04) — unlike
    `include_backtest`, there's no opt-in toggle for this one: the
    dashboard has no legitimate reason to show the OTHER account's recent
    decisions mixed in with this one's."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        if ticker is None:
            rows = await conn.fetch(
                f"""
                SELECT id, ticker, trade_date, rating, status, raw_return, alpha_return, reflection, decision_text, created_at, action_label, card_reason_ko, card_reason_en, total_tokens, decision_trigger
                FROM memory_entries
                WHERE account_mode = $1 AND user_id = $3 {"" if include_backtest else "AND is_backtest = false"}
                ORDER BY trade_date DESC, id DESC
                LIMIT $2
                """,
                current_account_mode(),
                limit,
                get_current_user_id(),
            )
        else:
            rows = await conn.fetch(
                f"""
                SELECT id, ticker, trade_date, rating, status, raw_return, alpha_return, reflection, decision_text, created_at, action_label, card_reason_ko, card_reason_en, total_tokens, decision_trigger
                FROM memory_entries
                WHERE ticker = $1 AND account_mode = $2 AND user_id = $4 {"" if include_backtest else "AND is_backtest = false"}
                ORDER BY trade_date DESC, id DESC
                LIMIT $3
                """,
                ticker,
                current_account_mode(),
                limit,
                get_current_user_id(),
            )

    return [
        DecisionEntry(
            id=row["id"],
            ticker=row["ticker"],
            trade_date=row["trade_date"],
            rating=row["rating"],
            status=row["status"],
            raw_return=row["raw_return"],
            alpha_return=row["alpha_return"],
            reflection=row["reflection"],
            decision_text=row["decision_text"],
            created_at=row["created_at"],
            action_label=row["action_label"],
            card_reason_ko=row["card_reason_ko"],
            card_reason_en=row["card_reason_en"],
            total_tokens=row["total_tokens"],
            decision_trigger=row["decision_trigger"],
        )
        for row in rows
    ]


async def get_last_decision_time(ticker: str) -> datetime | None:
    """Timestamp of the most recent LIVE (is_backtest = false) full-pipeline
    decision for `ticker`, or None if it has never been analyzed —
    app.live.auto_trade_loop's rebalance-interval gate uses this to decide
    whether a candidate is due for a fresh 4-analyst+debate re-analysis or
    still within `settings.rebalance_interval_days` of its last one. A real
    same-day flip-flop this gate is meant to prevent: 379800.KS was bought at
    05:16 and sold again at 05:45 the same morning (2026-08-03) after being
    re-analyzed from scratch ~30 minutes later — the fundamentals/dividend
    date cited in the reversal hadn't changed in that window, only which way
    the LLM's own re-run happened to land. Backtest rows are excluded so a
    historical replay's own decisions never suppress a live re-analysis (and
    vice versa) — same is_backtest boundary `get_recent_entries` already
    draws. Degrades quietly (returns None, i.e. "treat as never analyzed")
    on a DB failure, since this is a cost-saving throttle, not a safety
    gate — a missed skip just costs one redundant LLM pass, never a missed
    risk check."""
    try:
        pool = await get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT MAX(created_at) AS last_decision
                FROM memory_entries
                WHERE ticker = $1 AND is_backtest = false AND user_id = $2 AND account_mode = $3
                """,
                ticker,
                get_current_user_id(),
                current_account_mode(),
            )
    except Exception:
        logger.warning("get_last_decision_time: DB read failed for %s", ticker, exc_info=True)
        return None
    return row["last_decision"] if row else None


async def tickers_cash_floor_capped(tickers: list[str]) -> list[str]:
    """Which of `tickers` had their most recent LIVE decision's exposure
    increase blocked purely by the portfolio cash floor
    (app.execution.portfolio_risk.apply_portfolio_caps writes "Portfolio
    cash floor: ..." into decision_text via append_execution_note) —
    app.live.auto_trade_loop's cash-freed-up re-review bypass uses this to
    find candidates worth a live (no-LLM) cash-headroom recheck, instead of
    re-running the full 4-analyst pipeline on every trade allowlist ticker just to
    find out nothing changed. A ticker whose last decision was a genuine
    Hold/Sell, or was capped by a DIFFERENT reason (1% risk cap, max open
    positions, concentration/correlation), doesn't match this and stays
    subject to the normal rebalance_interval_days throttle — this is
    deliberately narrow to the one cap a later cash inflow can actually
    relax. Degrades quietly (empty list) on a DB failure, same convention
    as get_last_decision_time — this is a cost-saving bypass, not a safety
    gate, so a missed match just costs one skipped early recheck, never a
    missed risk check."""
    if not tickers:
        return []
    try:
        pool = await get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT DISTINCT ON (ticker) ticker, decision_text
                FROM memory_entries
                WHERE ticker = ANY($1) AND is_backtest = false AND user_id = $2 AND account_mode = $3
                ORDER BY ticker, created_at DESC
                """,
                tickers,
                get_current_user_id(),
                current_account_mode(),
            )
    except Exception:
        logger.warning("tickers_cash_floor_capped: DB read failed", exc_info=True)
        return []
    return [row["ticker"] for row in rows if "Portfolio cash floor:" in row["decision_text"]]


async def clear_backtest_memory(tickers: list[str]) -> None:
    """Deletes every `is_backtest=true` row for `tickers` (current
    account_mode/user_id) from `memory_entries`, so a fresh replay never
    inherits `past_context` written by an EARLIER, unrelated backtest run.

    `app.execution.ledger.clear_backtest_positions` clears `trades` for
    exactly this reason (see that function's own docstring), but `trades`
    and `memory_entries` are two separate tables — clearing one says nothing
    about the other. `get_similar_context`'s own `live_only_clause` (this
    module, ~line 406) is the concrete mechanism that makes the gap matter:
    `"AND NOT is_backtest" if as_of is None else ""` means LIVE reads filter
    `is_backtest` rows out, but BACKTEST reads (`as_of` set) see every
    `is_backtest=true` row unfiltered, on the assumption that a replay is
    only ever learning from its OWN earlier rounds in the SAME run. An A/B
    experiment comparing languages breaks that assumption: without this
    function, a Korean-language run's `memory_entries` rows would still be
    sitting there (with `is_backtest=true`) the next time an English-language
    run for the same ticker/date executes `backtest_scope()`d, and
    `get_similar_context` would hand them back as "past lessons" — silently
    smuggling one arm's reasoning into the other and making any Korean-
    vs-English comparison meaningless. `clear_backtest_positions` alone does
    not catch this because it only ever touches `trades`.

    Deliberately does NOT degrade quietly on a DB failure, unlike every
    other function in this module (see this module's own docstring for the
    "optional category" convention the rest follow) — same reasoning as
    `clear_backtest_positions`: a silent failure here would let exactly the
    cross-run memory contamination this function exists to prevent slip
    back in unnoticed, so a broken clear should fail loudly instead of
    producing a quietly-contaminated comparison."""
    if not tickers:
        return
    pool = await get_pool()
    async with pool.acquire() as conn:
        deleted = await conn.fetch(
            "DELETE FROM memory_entries WHERE ticker = ANY($1) AND is_backtest = true "
            "AND account_mode = $2 AND user_id = $3 RETURNING id",
            tickers,
            current_account_mode(),
            get_current_user_id(),
        )
    if deleted:
        logger.info("clear_backtest_memory: removed %d stale backtest memory row(s) for %s", len(deleted), tickers)
