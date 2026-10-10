"""Postgres + pgvector connection pool and schema bootstrap for the memory /
reflection store (Phase 3). A dedicated, standalone instance per this
project's isolation requirement — never shared with any other
repo/service; stand it up via `docker-compose.yml` at the repo root.
"""

from __future__ import annotations

import asyncio
import time

import asyncpg
from pgvector.asyncpg import register_vector

from app.config import settings

_SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;

-- Registry for the user_id tag already stamped on every trades/memory_entries/
-- alerts row (app.user_context.get_current_user_id) — created here, first,
-- so those tables' own FK constraints (added in their own modules) always
-- have something to reference by the time they run. Not itself a real
-- multi-tenant boundary yet: nothing here enforces who can read what — see
-- project memory: db-user-separation.
CREATE TABLE IF NOT EXISTS users (
    user_id      TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,
    active       BOOLEAN NOT NULL DEFAULT true,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Every row written before this table existed already carries this exact
-- id (see trades/memory_entries/alerts' own `user_id TEXT ... DEFAULT
-- 'default'` columns) — seeded here so an FK added later against those
-- tables never fails on pre-existing data.
INSERT INTO users (user_id, display_name) VALUES ('default', 'Default')
ON CONFLICT (user_id) DO NOTHING;

CREATE TABLE IF NOT EXISTS memory_entries (
    id                  SERIAL PRIMARY KEY,
    ticker              TEXT NOT NULL,
    trade_date          DATE NOT NULL,
    situation_text      TEXT NOT NULL,
    situation_embedding VECTOR(384) NOT NULL,
    rating              TEXT NOT NULL,
    decision_text       TEXT NOT NULL,
    benchmark_ticker    TEXT NOT NULL DEFAULT 'SPY',
    raw_return          DOUBLE PRECISION,
    alpha_return        DOUBLE PRECISION,
    holding_days        INTEGER,
    reflection          TEXT,
    status              TEXT NOT NULL DEFAULT 'pending',
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- The table above predates this column, so it's an ALTER rather than part
-- of the CREATE — full agent-by-agent transcript (analyst reports, debate
-- turns, research plan, trader proposal), so a decision's reasoning can be
-- inspected later instead of only ever seeing the final rating.
ALTER TABLE memory_entries ADD COLUMN IF NOT EXISTS full_transcript JSONB;

-- Excludes a row from `get_similar_context`'s retrieval without deleting it
-- (keeps the row for its own audit trail). Exists because the pre-fix
-- ticker-identity hallucination bug (see market_analyst.py's docstring) could
-- have written a reflection/decision citing a hallucinated company identity
-- as an established "lesson" — a false premise that, once embedded, would
-- otherwise keep resurfacing as if it were validated fact for every future
-- decision on that ticker (or, via cross-ticker similarity, on a topically
-- related one).
ALTER TABLE memory_entries ADD COLUMN IF NOT EXISTS flagged_bad BOOLEAN NOT NULL DEFAULT FALSE;

-- Multi-instance seam (agent-architecture-roadmap stage 4) — every row is
-- tagged with the user_context module's single fixed id today. Not yet a
-- real multi-tenant boundary (every query still reads/writes globally,
-- unfiltered by this column); the point is that adding it now, while the
-- table is small, is cheap — backfilling it after real usage accumulates
-- would not be.
ALTER TABLE memory_entries ADD COLUMN IF NOT EXISTS user_id TEXT NOT NULL DEFAULT 'default';

-- Buy/Sell/Increase/Decrease/Hold — what the rebalance ACTUALLY did to the
-- position (current vs post-risk-cap target share count), set after the
-- fact by app.execution.ledger.rebalance_action_label via
-- app.memory.store.set_action_label. Deliberately separate from `rating`
-- (the AI's own Buy/Overweight/Hold/Underweight/Sell conviction judgment,
-- unrelated to actual current holdings) — the two can disagree (e.g. an
-- Underweight rating on a position far below even that reduced target
-- still nets out as a Buy), which is exactly the confusion this column
-- exists to resolve for the dashboard. NULL until the rebalance runs
-- (pipeline.py calls set_action_label right after rebalance_to_target
-- returns), and forever NULL for anything with no linked rebalance at all.
ALTER TABLE memory_entries ADD COLUMN IF NOT EXISTS action_label TEXT;

-- Distinguishes a historical-replay decision (app.backtest) from a genuine
-- live one, at read time rather than by excluding backtest rows from this
-- table at all — a backtest decision's row is still exactly what
-- app.audit's report/history/experiments tools and get_similar_context's
-- past-lessons retrieval are FOR (see their own docstrings), so it must
-- keep being written. What must NOT happen is a backtest run showing up
-- to the user as if it were a real recent decision — get_recent_entries
-- (the live dashboard's /api/decisions) excludes these by default;
-- app.audit's own queries are untouched and
-- keep reading everything. Real incident this fixes (2026-07-24): a batch
-- of backtest runs' historical-dated decisions appeared in the live
-- dashboard's decision history/signal cards, indistinguishable from real
-- ones, because nothing here recorded which was which.
ALTER TABLE memory_entries ADD COLUMN IF NOT EXISTS is_backtest BOOLEAN NOT NULL DEFAULT FALSE;

-- The Trader's own Buy/Hold/Sell proposal, BEFORE the risk debate and
-- Portfolio Manager ever see it (app.state.TraderAction) — set once,
-- alongside `rating`, by app.orchestrator.pipeline's store_pending call.
-- Deliberately separate from both `rating` (the Portfolio Manager's final
-- 5-way Buy/Overweight/Hold/Underweight/Sell conviction) and `action_label`
-- (what the rebalance ACTUALLY did to the position, set after the fact) —
-- this column exists so a future stats pass can measure how often the risk
-- debate + Portfolio Manager actually changed the Trader's own initial
-- call, instead of only ever seeing the final outcome. NULL for any row
-- written before this column existed. See app.audit.stats for the mapping
-- this needs (TraderAction is 3-way, rating is 5-way — not a 1:1 diff).
-- 리스크 토론이 시작되기 전 Trader 자체 제안 — 최종 rating과 별개로 저장해서
-- 나중에 "토론이 실제로 결론을 바꾼 비율"을 계산할 수 있게 함(지금은 저장만).
ALTER TABLE memory_entries ADD COLUMN IF NOT EXISTS trader_action TEXT;

-- Paper/real account separation (2026-08-04 — see project memory:
-- user-account-mode-db-separation-design, priority-3 "여유 있을 때" item).
-- Same rationale as is_backtest above, one axis over: without this, the
-- dashboard's win-rate/alpha stats (app.audit.stats) had no way to tell a
-- paper-account decision from a real-account one, so once real trading
-- starts, they'd silently blend into one number with no way to see either
-- account's own track record. Backfilled to 'paper' since every row so
-- far predates real-mode use (same reasoning as trades/portfolio_
-- drawdown_state's identical backfill).
ALTER TABLE memory_entries ADD COLUMN IF NOT EXISTS account_mode TEXT NOT NULL DEFAULT 'paper'
    CHECK (account_mode IN ('paper', 'real'));

-- Decision version tagging (2026-08-12) — before this, a model swap
-- ([[llm-model-swap-gpt-oss-issue]]) or a prompt change
-- ([[agents-prompt-quality-review]]) left no trace on the DECISION itself:
-- `app.backtest.experiments`'s `backtest_runs.git_commit` already tags
-- whole backtest RUNS, but individual live decisions (and the individual
-- backtest decisions sitting in this same table) had nothing, so a
-- before/after comparison across either kind of change was impossible
-- once enough time had passed to lose track of it by memory alone. Both
-- populated automatically inside app.memory.store.store_pending (see its
-- own comment) — no caller needs to pass either. NULL for every row
-- written before this column existed; that's expected, not a bug to chase.
-- 모델 교체·프롬프트 개선 전후 비교가 불가능했던 문제 — 판단 시점에 자동 기록.
ALTER TABLE memory_entries ADD COLUMN IF NOT EXISTS git_commit TEXT;
ALTER TABLE memory_entries ADD COLUMN IF NOT EXISTS llm_model TEXT;

-- Stage 1 (2026-08-13) — F5: the Portfolio Manager's own 0.0-1.0 conviction
-- (app.state.PortfolioDecision.confidence) directly scales the actual
-- capital committed via app.execution.sizing.confidence_size_modifier
-- (>=0.90 -> 1.0x, >=0.75 -> 0.9x, >=0.60 -> 0.7x, >=0.50 -> 0.5x, below
-- that -> 0.0x, trade skipped) — yet until now nothing recorded it, so
-- "did a high-confidence call actually do better than a low-confidence one"
-- could never be checked after the fact. Nullable, no DEFAULT: the LLM can
-- omit confidence entirely (the field itself is `float | None`), and 0.0 is
-- a real, meaningful value (max size cut / trade skipped) that must not be
-- confused with "never recorded" — a DEFAULT of 0.0 would make those two
-- cases indistinguishable. NULL for every row written before this column
-- existed, and for any row where the model didn't return a confidence.
-- 실제 투입 자본을 직접 스케일하는 값인데 지금까지 저장이 안 됨 — 확신도 높은
-- 판단이 실제로 더 좋았는지 사후검증이 불가능했음. None은 PM이 확신도를 안 낸 경우.
ALTER TABLE memory_entries ADD COLUMN IF NOT EXISTS confidence DOUBLE PRECISION;

-- Stage 1 (2026-08-13) — F5: position state immediately BEFORE this decision
-- was made (app.orchestrator.pipeline's `_position_context`, `shares > 0`),
-- captured at decision time. Named `held_at_decision`, not `held`, to avoid
-- any confusion with `action_label` (what the rebalance ACTUALLY did,
-- recorded AFTER the fact) — the two answer different questions and can
-- disagree. Exists because a Hold rating's correct direction is opposite
-- depending on this: Hold while already holding wins on alpha > 0 (the
-- position we kept outperformed), Hold while flat (in cash) wins on
-- alpha < 0 (the thing we passed on underperformed) — with no column to
-- tell the two apart, every Hold had to be excluded from scoring entirely
-- (see app.audit.stats's `_WIN_DIRECTION` and its own comment). NULL for
-- every row written before this column existed.
--
-- 2026-08-20 correction: the comment used to say this was "permanently
-- unrecoverable for those, not a bug to chase" — that turned out to be
-- wrong. `held_at_decision_inferred` (below) reconstructs it from the
-- `trades` ledger at 97.2% measured agreement against the 144 rows where
-- this column WAS observed directly. This column itself is still never
-- backfilled — it keeps its precise meaning (observed, preferring the real
-- broker balance; see `_position_context`) — the reconstruction lives in
-- the separate column on purpose so a 2.8%-error inference can never be
-- silently read back as the real thing.
-- 판단 직전 포지션 보유 여부 — Hold의 정답 방향이 보유중(알파>0)/현금중(알파<0)
-- 으로 정반대인데 이를 구분할 컬럼이 없어 Hold가 채점에서 통째로 제외돼 있었음.
ALTER TABLE memory_entries ADD COLUMN IF NOT EXISTS held_at_decision BOOLEAN;

-- Backfilled reconstruction of `held_at_decision` for the 177 live rows
-- that predate that column (created 2026-07-22~08-13) but are otherwise
-- genuine same-day decisions (not the 47-row backtest-replay contamination
-- found the same day this column was added — those got retagged
-- `is_backtest = true` instead, see project memory
-- backtest-live-db-shared-contamination.md for the class of bug and
-- ticker-identity/HANDOFF-2026-08-20-* memory for this specific instance).
-- Reconstructed by `scripts/backfill_held_at_decision.py` from the internal
-- `trades` ledger: the most recent trade (by `created_at`, not `trade_date`
-- — same-day round trips need timestamp ordering to resolve correctly,
-- confirmed against 29 ticker-days with same-day multi-trade churn)
-- strictly before this decision's own `created_at`, for the same ticker/
-- account_mode/user_id, `shares_after > 0` (no prior trade = flat = false).
-- Measured 97.2% agreement (140/144) against the rows where the real
-- `held_at_decision` was independently observed from the live broker
-- balance — the 4 disagreements are all the same direction (ledger says
-- held, the real account was actually flat: paper-ledger drift, same
-- failure mode `_position_context`'s own docstring already documents for
-- why the live pipeline prefers the real broker balance over this ledger
-- in the first place). NEVER written by the live pipeline, and NEVER
-- read as if it were `held_at_decision` itself — app.audit.stats/weekly
-- only fall back to it via COALESCE(held_at_decision, held_at_decision_
-- inferred) when the observed column is NULL, and track how many rows in
-- each bucket came from inference so a report never presents an inferred
-- row as directly observed.
ALTER TABLE memory_entries ADD COLUMN IF NOT EXISTS held_at_decision_inferred BOOLEAN;

-- Stage 1 (2026-08-13) — F6: a SECOND, fixed-trading-day-count return
-- figure alongside the existing `raw_return`/`alpha_return`/`holding_days`.
-- Those three are intentionally variable-window (`_common_eval_date` picks
-- whatever date both the ticker and benchmark next have a usable close on,
-- specifically so resolution never stalls — see that function's own
-- comment) — anywhere from ~5 to 112+ calendar days across live rows today,
-- which means they cannot feed any statistically sound confidence-interval
-- calculation: a 5-day alpha and a 112-day alpha are not comparable
-- quantities. These four columns instead measure at exactly
-- `settings.memory_fixed_horizon_trading_days` (=10) TRADING days out
-- (app.memory.auto_resolve's `_horizon_eval_date`), so every row's alpha is
-- over the same window and future confidence-interval/statistical-
-- significance work can actually use it. `horizon_trading_days` records the
-- trading days actually counted (should always equal the fixed N once
-- filled); `horizon_eval_date` is the date that measurement landed on.
-- NULL is the expected, common state — either the row hasn't matured to N
-- trading days yet, or price history doesn't reach far enough to compute
-- it (see `backfill_horizon_returns`'s window bound) — NOT a bug.
-- 기존 alpha_return은 5~112일이 섞인 가변창이라 신뢰구간 계산에 못 씀 — 이
-- 컬럼들은 거래일 고정 N(=10)이라 향후 통계적 신뢰구간 계산에 사용 가능. NULL은
-- 아직 성숙 안 됐거나 가격 데이터로 계산 불가한 경우이며 버그가 아님.
ALTER TABLE memory_entries ADD COLUMN IF NOT EXISTS horizon_raw_return DOUBLE PRECISION;
ALTER TABLE memory_entries ADD COLUMN IF NOT EXISTS horizon_alpha_return DOUBLE PRECISION;
ALTER TABLE memory_entries ADD COLUMN IF NOT EXISTS horizon_trading_days INTEGER;
ALTER TABLE memory_entries ADD COLUMN IF NOT EXISTS horizon_eval_date DATE;

-- Short, display-only "why" for the dashboard signal card, distinct from
-- executive_summary (the action plan) — see app.state's
-- _PortfolioDecisionFields comment for the full rationale. Both languages
-- are stored so either UI language renders correctly from one decision.
-- Nullable with no default: legitimately absent for rows predating this
-- column and for any decision where the model omitted the optional field;
-- the frontend falls back to executive_summary-derived text for those.
ALTER TABLE memory_entries ADD COLUMN IF NOT EXISTS card_reason_ko TEXT;
ALTER TABLE memory_entries ADD COLUMN IF NOT EXISTS card_reason_en TEXT;

-- Total LLM token usage (prompt+completion, every analyst/debate/judgment-
-- node call) for the one ticker/decision this row represents — see
-- app.llm.metrics.track_run_tokens/current_run_tokens for how it's
-- accumulated per run. Display-only (the dashboard signal card's cost
-- footer); never read by sizing/execution. NULL for any row predating this
-- column.
ALTER TABLE memory_entries ADD COLUMN IF NOT EXISTS total_tokens INTEGER;

-- How this decision was TRIGGERED, as opposed to what data window it saw
-- (`is_backtest`, above) — the two are separate axes and both must be
-- filterable independently. 'auto' = the ticker was chosen by the scheduled
-- app.live.auto_trade_loop cycle's own screening (this includes the
-- dashboard's "다시 분석" button, which re-runs that same screening — the
-- human picked the TIMING, not the tickers). 'manual' = a human picked THIS
-- SPECIFIC ticker out of band: the Trading tab's "AI 종합분석 실행" button
-- (/api/analyze/{ticker}), a chat deep_dive question about one ticker, and
-- the `python -m app.main --ticker` / scripts.run_auto_trade_cycle_now dev
-- entry points. Exists because app.audit's win-rate/confidence-calibration
-- queries had no way to exclude manual runs, and a manual run is triggered
-- by a person who is already curious or worried about that one ticker —
-- textbook sample-selection bias, silently feeding the very numbers
-- [[stage-gate-criteria-draft]] uses to gate capital increases. Also drives
-- the AI 시그널 card badge so a one-off human-requested analysis cannot
-- masquerade as the automated desk's own standing call.
-- Nullable with NO DEFAULT, deliberately — unlike is_backtest/account_mode
-- (whose backfill value was known true for every pre-existing row), the
-- manual-analyze button predates this column, so an unknown subset of the
-- ~350 existing rows genuinely ARE manual and no signal survives to
-- identify them (the auto loop's per-cycle ticker list was never stored).
-- Backfilling 'auto' would assert something false; NULL honestly means
-- "predates trigger tagging". Same reasoning as `confidence`/
-- `held_at_decision` above. Every row written from now on gets an explicit
-- value (app.memory.store.store_pending defaults it to 'auto'), so NULL is
-- a closed, never-growing legacy set. Audit queries treat NULL as auto —
-- see app.audit.stats's `trigger` param for why.
-- 판단이 "어떻게 촉발됐는지" — 데이터 창(is_backtest)과는 다른 축. 사람이 특정
-- 종목을 직접 골라 돌린 분석은 표본선택 편향이라 적중률 통계에서 빼야 함.
-- NULL은 이 컬럼 도입 전 행(소급 판별 불가) — 통계에서는 auto로 취급.
ALTER TABLE memory_entries ADD COLUMN IF NOT EXISTS decision_trigger TEXT;
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'memory_entries_decision_trigger_check'
    ) THEN
        ALTER TABLE memory_entries ADD CONSTRAINT memory_entries_decision_trigger_check
            CHECK (decision_trigger IS NULL OR decision_trigger IN ('auto', 'manual'));
    END IF;
END $$;

-- How many of the four analysts (Market/Fundamentals/News+Macro/Sentiment)
-- FAILED for this decision and were replaced by a placeholder note. 0 = a
-- full four-analyst run; 1-3 = a degraded run that still produced a rating
-- (app.orchestrator.pipeline's analyst gather deliberately degrades rather
-- than aborting, so the other analysts' work isn't discarded). Exists
-- because nothing distinguished the two: app.audit's win-rate and
-- confidence-calibration numbers — the ones [[stage-gate-criteria-draft]]
-- uses to gate capital increases — pooled a one-analyst decision with a
-- four-analyst one, and the only trace of the difference was a log line
-- that rotated away (2026-08-25).
-- The names of the failed analysts are in full_transcript->'degraded_
-- analysts'; this column exists so a SQL filter never has to open the blob.
-- Nullable with NO DEFAULT, same convention as confidence/held_at_decision/
-- decision_trigger above: an existing row genuinely predates the tracking
-- and no signal survives to reconstruct it, so backfilling 0 would assert
-- something false. Every row written from now on gets an explicit value.
-- 애널리스트 4명 중 실패한 수 — 축소 분석과 정상 분석이 통계에서 섞이던 문제.
ALTER TABLE memory_entries ADD COLUMN IF NOT EXISTS analysts_failed SMALLINT;

-- The market date whose close made this row's outcome knowable — the exit
-- side of the return measurement. NOT the wall-clock instant the resolving
-- UPDATE ran (that's what a TIMESTAMPTZ DEFAULT now() would give you, and
-- it would be actively wrong: a backtest replay resolves a row at REAL
-- wall-clock time, months/years after the SIMULATED trade_date it belongs
-- to, so bounding retrieval against now() would exclude every backtest-
-- resolved row from every later replay in the same run). This is a DATE,
-- caller-supplied, same domain as trade_date/horizon_eval_date.
--
-- 2026-09-01 fix — app.memory.store.get_similar_context bounded retrieval
-- only on trade_date (when the DECISION was made), never on when the row
-- was actually GRADED (status='resolved' + alpha_return + reflection all
-- become true only after app.memory.auto_resolve.resolve_ticker /
-- app.backtest.harness._process_ticker_date runs, `holding_days` later —
-- 5 to 112+ calendar days observed live, exactly one day later in a
-- backtest replay). A backtest replaying date T could therefore retrieve a
-- graded outcome/reflection that, AT T, had not happened yet — a look-ahead
-- leak into the Portfolio Manager's own context, confounding exactly the
-- backtest alpha numbers [[stage-gate-criteria-draft]] gates capital on.
-- Live is unaffected (as_of=None skips the new bound entirely; no live
-- caller ever passes as_of).
--
-- By construction resolved_on = trade_date + holding_days for every row
-- either writer produces (auto_resolve.py: eval_date - trade_date;
-- harness.py: record.trade_date - prev.trade_date), which is what makes
-- the backfill below EXACT rather than an approximation. Nullable with NO
-- DEFAULT, same convention as analysts_failed/confidence/held_at_decision
-- above — a pre-existing resolved row with holding_days IS NULL has no
-- recoverable resolution date and stays NULL rather than being guessed.
-- 이 행의 결과가 실제로 알려진 시점(채점 시점) — 결정 시점(trade_date)이 아님.
ALTER TABLE memory_entries ADD COLUMN IF NOT EXISTS resolved_on DATE;

-- One-time, idempotent, EXACT reconstruction for every row resolved before
-- this column existed — see the column comment above for why
-- trade_date + holding_days is exact, not a guess. A no-op after the first
-- boot (the resolved_on IS NULL guard), and self-heals any row written by
-- an older running image during a mixed-deploy window.
UPDATE memory_entries
SET resolved_on = trade_date + holding_days
WHERE status = 'resolved' AND resolved_on IS NULL AND holding_days IS NOT NULL;

-- HNSW (not ivfflat): no lists-tuning step needed and works correctly even
-- on an empty/small table, unlike ivfflat which wants to be built after
-- the table is populated — this schema bootstrap script also runs on a
-- fresh/empty database, so HNSW is the only one of the two that's safe to
-- create unconditionally here. vector_cosine_ops matches the `<=>` cosine-
-- distance operator every similarity query in app.memory.store already
-- uses (get_similar_context's ORDER BY situation_embedding <=> $2). Added
-- while memory_entries is still small (756 rows) — cheap now, expensive to
-- retrofit under load later.
CREATE INDEX IF NOT EXISTS memory_entries_situation_embedding_hnsw_idx
    ON memory_entries USING hnsw (situation_embedding vector_cosine_ops);

-- Backs get_similar_context's same_rows query (WHERE ticker = $1 AND
-- status = 'resolved' AND ...) — a plain covering index, unrelated to the
-- vector search above.
CREATE INDEX IF NOT EXISTS memory_entries_ticker_status_idx
    ON memory_entries (ticker, status);

-- Ties every row to a registered person — same reasoning and same
-- pg_constraint-guarded re-run as app.execution.ledger's identical trades
-- FK. Safe within this same _SCHEMA string since `users` (and its 'default'
-- seed row) is created earlier above, in the same script.
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'memory_entries_user_id_fkey') THEN
        ALTER TABLE memory_entries ADD CONSTRAINT memory_entries_user_id_fkey
            FOREIGN KEY (user_id) REFERENCES users(user_id);
    END IF;
END $$;

-- Experiment tracking (app.backtest.experiments) — one row per backtest CLI
-- invocation, tagging its git commit/label alongside the resulting summary
-- metrics. Complements memory_entries (individual decisions): this is
-- about backtest RUNS as a whole, so "did this prompt change actually help"
-- can be answered by comparing rows here instead of re-deriving "what code
-- was this?" from memory or diffing raw decisions by hand.
CREATE TABLE IF NOT EXISTS backtest_runs (
    id                  SERIAL PRIMARY KEY,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    git_commit          TEXT,
    label               TEXT,
    kind                TEXT NOT NULL,
    tickers             TEXT NOT NULL,
    benchmark_ticker    TEXT NOT NULL,
    start_date          DATE NOT NULL,
    end_date            DATE NOT NULL,
    cadence_days        INTEGER NOT NULL,
    n_dates             INTEGER NOT NULL,
    return_pct          DOUBLE PRECISION,
    benchmark_return_pct DOUBLE PRECISION,
    sharpe              DOUBLE PRECISION,
    sortino             DOUBLE PRECISION,
    max_drawdown        DOUBLE PRECISION,
    cvar_95             DOUBLE PRECISION,
    win_rate            DOUBLE PRECISION,
    mean_alpha          DOUBLE PRECISION
);

-- 2026-09-11: return_pct must be the REAL ledger return (actual position
-- sizing + stop-loss), same convention app.backtest.reporting already
-- applies to risk_metrics — a single-ticker run used to record its
-- exposure-proxy return (compute_strategy_return: ignores real sizing/
-- stop-loss) into return_pct instead, indistinguishable from a portfolio
-- run's genuinely-actual portfolio_return in the same column. The proxy is
-- kept here rather than dropped, for signal-level A/B comparisons that
-- never move the ledger (e.g. a trailing-stop sweep) — NULL for every
-- pre-existing row and for every kind='portfolio' row (no exposure proxy
-- exists there).
ALTER TABLE backtest_runs ADD COLUMN IF NOT EXISTS exposure_return_pct DOUBLE PRECISION;

-- Mid-run checkpoint for a labeled multi-ticker portfolio backtest
-- (app.backtest.checkpoint / app.backtest.portfolio_harness) — lets a
-- process killed partway through a multi-hour replay resume from its last
-- completed trade date instead of re-walking the whole thing. This is
-- specifically for a hard kill (server power cut — see project memory:
-- server-power-schedule-blocker), not an LLM hiccup (already handled
-- in-process by harness._run_pipeline_with_retry, no checkpoint needed for
-- that). One row per label, overwritten in place as the run progresses,
-- deleted once the run finishes normally — a leftover row only means a
-- run under that label never finished.
CREATE TABLE IF NOT EXISTS backtest_checkpoints (
    label                 TEXT PRIMARY KEY,
    params                JSONB NOT NULL,
    state                 JSONB NOT NULL,
    completed_date_index  INTEGER NOT NULL,
    updated_at            TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""

_pool: asyncpg.Pool | None = None
_pool_lock = asyncio.Lock()

# Applied to every connection this app opens (bootstrap connection and every
# pooled one) — see database_idle_in_transaction_timeout_seconds/
# database_statement_timeout_seconds's own comment in config.py for why:
# a hard-killed client can't run cleanup code, so the server itself has to
# be the one to reap a stuck session, or a single orphaned connection can
# block every other connection's schema bootstrap indefinitely (as it did
# on 2026-07-20 — see that incident's postmortem in this module's git log).
_SERVER_SETTINGS = {
    "idle_in_transaction_session_timeout": f"{int(settings.database_idle_in_transaction_timeout_seconds * 1000)}",
    "statement_timeout": f"{int(settings.database_statement_timeout_seconds * 1000)}",
    "idle_session_timeout": f"{int(settings.database_idle_session_timeout_seconds * 1000)}",
}


async def _register_vector_codec(conn: asyncpg.Connection) -> None:
    await register_vector(conn)


async def _bootstrap_pool() -> asyncpg.Pool:
    # Bootstrap on a bare connection first: register_vector() requires
    # the `vector` type to already exist, so the extension/table must be
    # created before any pooled connection tries to register the codec.
    # `timeout=` bounds the connect step; `command_timeout=` bounds each
    # query (including this bootstrap's own CREATE TABLE/ALTER TABLE
    # script, and — for the pool below — the RESET asyncpg runs on every
    # connection release) — deliberately two different settings now, see
    # database_command_timeout_seconds's own comment in app.config for why
    # reusing the connect timeout for both used to cause the query itself
    # to be cut off early. A genuinely unresponsive server (TCP accepts,
    # then never replies to anything — exactly what this project's shared
    # dev Postgres did on 2026-07-20) can still stall past both in
    # practice; the outer asyncio.wait_for in get_pool() is the actual
    # backstop that guarantees this whole function returns or raises
    # within the deadline, regardless of which asyncpg-internal timeout
    # does or doesn't cover it.
    # DB가 TCP는 받아주는데 그 뒤로 전혀 응답이 없는 경우까지 확실히 타임아웃 처리.
    bootstrap_conn = await asyncpg.connect(
        settings.database_url,
        timeout=settings.database_connect_timeout_seconds,
        command_timeout=settings.database_command_timeout_seconds,
        server_settings=_SERVER_SETTINGS,
    )
    try:
        await bootstrap_conn.execute(_SCHEMA)
    finally:
        await bootstrap_conn.close()

    return await asyncpg.create_pool(
        settings.database_url, init=_register_vector_codec,
        timeout=settings.database_connect_timeout_seconds,
        command_timeout=settings.database_command_timeout_seconds,
        min_size=settings.database_pool_min_size,
        max_size=settings.database_pool_max_size,
        server_settings=_SERVER_SETTINGS,
    )


# Set on every failed get_pool() attempt, cleared on success — see
# get_pool()'s fail-fast check below.
_last_failure_at: float | None = None


def _recent_failure() -> bool:
    return (
        _last_failure_at is not None
        and time.monotonic() - _last_failure_at < settings.database_failure_backoff_seconds
    )


async def get_pool() -> asyncpg.Pool:
    """Double-checked locking: without the lock, several coroutines calling
    this concurrently before the pool exists (e.g. loading N tickers' tick
    snapshots at startup) would each see `_pool is None` and each call
    `asyncpg.create_pool()` themselves — N full pools against one small
    Postgres instance is exactly how you hit "too many clients already".

    Fails fast (no connection attempt at all) for
    `database_failure_backoff_seconds` after the most recent failure —
    without this, every DB-touching call during an outage independently paid
    the full connect+schema-bootstrap timeout in turn (serialized through
    the same lock), and each one was a fresh connection attempt against an
    already-struggling server rather than a backoff. Every caller already
    catches this function's exceptions and degrades (see ledger.get_position,
    risk_rules.load, etc.), so raising sooner only makes that same fallback
    path faster, not less safe."""
    global _pool, _last_failure_at
    if _pool is not None:
        return _pool
    if _recent_failure():
        raise RuntimeError(
            f"database unavailable (connection failed within the last "
            f"{settings.database_failure_backoff_seconds:.0f}s, not retrying yet)"
        )

    async with _pool_lock:
        if _pool is not None:  # someone else finished while we waited for the lock
            return _pool
        if _recent_failure():  # someone else's attempt failed while we waited for the lock
            raise RuntimeError(
                f"database unavailable (connection failed within the last "
                f"{settings.database_failure_backoff_seconds:.0f}s, not retrying yet)"
            )

        try:
            # A hard outer deadline (see _bootstrap_pool's own docstring for
            # why the inner asyncpg timeouts alone weren't enough in
            # practice). Sized for THREE sequential bounded phases inside
            # _bootstrap_pool, not just one: (1) the bootstrap
            # asyncpg.connect(), bounded by database_connect_timeout_seconds;
            # (2) that connection's bootstrap_conn.execute(_SCHEMA) — the
            # full multi-statement CREATE EXTENSION + ~20 ALTERs script —
            # bounded by database_command_timeout_seconds; (3)
            # asyncpg.create_pool(), which itself opens database_pool_min_
            # size new connections, each again bounded by database_connect_
            # timeout_seconds. The old budget (connect_timeout + 1) only
            # covered roughly phase (1) alone — on a cold or loaded Postgres,
            # hitting this timeout sets _last_failure_at, which then makes
            # EVERY DB call (including store_pending) fail-fast for
            # database_failure_backoff_seconds, i.e. live decisions get made
            # and traded with nothing recorded in memory_entries. +1s margin
            # on top so a bootstrap that legitimately takes right up to its
            # own combined limit isn't cut off by the outer one first.
            _pool = await asyncio.wait_for(
                _bootstrap_pool(),
                timeout=settings.database_connect_timeout_seconds * 2
                + settings.database_command_timeout_seconds
                + 1,
            )
        except Exception:
            _last_failure_at = time.monotonic()
            raise
        _last_failure_at = None
        return _pool
