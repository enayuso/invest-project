"""Mid-run checkpoint/resume for `app.backtest.portfolio_harness.
run_portfolio_backtest` — the fix for a specific, real failure mode: the
dev server this project's backtests run on loses power most evenings (see
project memory: server-power-schedule-blocker), which kills the Python
process outright with no chance to run cleanup code. Before this module,
`run_portfolio_backtest` only ever persisted a result once, at the very
end (`record_experiment`, called from `_main()`), so a run killed 90% of
the way through a multi-hour multi-month replay lost 100% of its LLM calls
and had to start over from date zero.

This is deliberately NOT a general crash-recovery framework. It only
covers a hard kill between trade dates — an LLM hiccup mid-date is already
absorbed in-process by `harness._run_pipeline_with_retry` without ever
reaching here. A crash mid-date (between two tickers in the same date's
rotation) IS resumed at ticker granularity (`in_flight_date_index`/
`completed_tickers_this_date` below): a resume replays the SAME in-flight
date's SAME rotation order, skipping the tickers already committed within
it, rather than re-walking the whole date from ticker zero. This is
required, not merely an optimization: `app.execution.ledger.
rebalance_to_target`'s 1% risk cap bounds only the shares a trade ADDS to
the current position (`capped_target = current_shares + max_incremental_
shares`), not the resulting position's total risk — so re-processing an
already-filled (ticker, date) would grant a SECOND cap-sized increment on
top of the first, overshooting what an uninterrupted run would have held.
If the mid-ticker save itself fails (best-effort, like every save here),
the date falls back to being re-walked from its start and that overshoot
can occur — an accepted rare degradation, not the normal path.

Only active when the caller passes a `label` — the same label
`app.backtest.experiments.record_experiment` already tags the finished run
with, so a checkpoint and its eventual result share one identifier a human
already has to type anyway (`--label baseline-domestic-20260812`). No
label means no checkpoint, matching every existing caller that never set
one — fully backward compatible.

Resuming an interrupted run does NOT call `app.execution.ledger.
clear_backtest_positions` — see that function's own docstring for why it
normally must run (so run N+1 of a repeated backtest doesn't inherit run
N's ending position). A checkpoint resume is the ONE case that must skip
it: the whole point is to keep the SAME simulated positions the killed
process's earlier trade dates already committed to Postgres. The caller
(`run_portfolio_backtest`) is responsible for only skipping the clear when
`load_checkpoint` actually returns a state, not just because a label was
given.

Resume is validated against `_params()` — tickers/date-range/cadence/
benchmark AND the code's current git commit. The commit check specifically
guards against a subtler failure than a crash: resuming a checkpoint made
before a prompt/pipeline change since landed would otherwise silently
splice dates decided under the OLD code with dates decided under the NEW
one into what looks like one comparable backtest result. Any mismatch —
including the commit — degrades to a fresh run, never a partial/best-guess
resume.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import date, datetime

from app.backtest.experiments import current_git_commit
from app.backtest.metrics import BacktestRecord
from app.memory.db import get_pool
from app.state import PortfolioDecision

logger = logging.getLogger(__name__)


@dataclass
class CheckpointState:
    completed_date_index: int  # trade_dates[0 : completed_date_index + 1] are fully processed
    prev_records: dict[str, BacktestRecord]  # ticker -> its own most recent record (absent = never had one)
    per_ticker_records: dict[str, list[BacktestRecord]]
    per_ticker_equity_curves: dict[str, list[float]]
    portfolio_equity_curve: list[float]
    running_high_water_mark: float
    ever_selected: set[str] = field(default_factory=set)
    cap_trigger_counts: dict[str, int] = field(default_factory=dict)
    # Ticker-granular mid-date resume (see module docstring). -1 means no
    # partially-processed date exists (a fresh run, or a checkpoint saved
    # right after a date's full rotation completed). When it equals
    # completed_date_index + 1 on load, the date at that index was
    # in-flight when the process died: in_flight_rotated_tickers is that
    # date's EXACT rotation order (persisted rather than re-derived, since
    # re-deriving via candidate_selector on resume could return a different
    # set/order — e.g. screening_candidate_selector sees now-held tickers
    # the partial run already bought), and completed_tickers_this_date is
    # the subset already committed, to be skipped on resume.
    in_flight_date_index: int = -1
    in_flight_rotated_tickers: list[str] = field(default_factory=list)
    completed_tickers_this_date: set[str] = field(default_factory=set)
    # How many trade dates this run will process in total, once it's a
    # complete run — 0 (unknown) is a valid value for a checkpoint saved by
    # code older than this field, never a reason to fail; a listing just
    # can't show a fraction for it, only the raw completed count. Purely
    # informational (app.backtest.checkpoint.list_checkpoints/`python -m
    # app.audit checkpoints`) — never read by load_checkpoint's own resume
    # decision, which only needs completed_date_index.
    total_dates: int = 0
    # The USD-equivalent aggregate starting point, captured ONCE via a real
    # portfolio_snapshot while every ticker is provably flat (see
    # portfolio_harness's own computation) — not re-derived on resume, so a
    # ticker whose starting_cash_for() hit its raw-USD FX-lookup fallback at
    # run start stays counted at that same (possibly degraded) value for
    # the whole run's portfolio_return math. 0.0 (falsy) means "unknown" —
    # either a checkpoint saved by code older than this field, or a run
    # with zero trade_dates — and portfolio_harness falls back to the old
    # blind settings.backtest_starting_cash * len(tickers) estimate in that
    # case, same as before this field existed.
    starting_total_equity_usd: float = 0.0
    # The git commit this run's checkpoint chain was started under, captured
    # ONCE (either freshly at a genuine fresh start, or inherited unchanged
    # from a successfully-resumed checkpoint's own stored value — see
    # portfolio_harness's own resolution) and threaded into every subsequent
    # save_checkpoint() call for the life of this run. NOT re-derived from
    # current_git_commit() on every save (that was the 2026-09-11 bug this
    # field fixes): if it were, a mid-run deploy into the same checkout would
    # silently re-stamp every later save with the NEW commit, so a resume
    # after a crash would compare the (already-updated) stored value against
    # a freshly re-evaluated current one and find them matching even though
    # the run actually spans two different commits' worth of decisions —
    # exactly the splicing this mechanism exists to prevent. None (git
    # unavailable, or a checkpoint saved by code older than this field)
    # degrades to a fresh current_git_commit() capture, same convention as
    # every other .get(..., default) field here.
    # 이 실행의 체크포인트 체인이 시작된 시점의 git commit — 한 번만 캡처해서
    # 매 저장마다 재평가하지 않음(재평가하면 재개 안전장치 자체가 무력화됨).
    git_commit: str | None = None


@dataclass
class CheckpointProgress:
    """One row's worth of `python -m app.audit checkpoints` — a lightweight
    read that does NOT require knowing the original run's tickers/date
    range/cadence (unlike load_checkpoint, which must match those exactly
    before it'll resume anything). This is for a human glancing at "how far
    along is it", not for deciding whether to resume."""

    label: str
    tickers: list[str]
    completed_date_index: int
    total_dates: int
    latest_portfolio_equity: float | None
    updated_at: datetime


def _record_to_json(record: BacktestRecord) -> dict:
    return {
        "trade_date": record.trade_date.isoformat(),
        # PortfolioDecision is a strict superset of FlatPortfolioDecision's
        # fields and PortfolioRating a strict superset of FlatPortfolioRating's
        # values ("Buy"/"Hold" both present) — so ANY decision this replay
        # ever produced round-trips through PortfolioDecision.model_validate
        # on the way back in, regardless of which of the two the pipeline
        # actually returned at the time. Only `render_portfolio_decision`
        # (the reflection prompt) and plain attribute reads touch the
        # restored object.
        "decision": record.decision.model_dump(mode="json"),
        "close": record.close,
        "adjustment_note": record.adjustment_note,
        "trailing_high_water": record.trailing_high_water,
        "trailing_entry_price": record.trailing_entry_price,
        # resolve_outcome now finds its DB row by `id`, not by
        # (ticker, trade_date) — so decision_id MUST round-trip through the
        # checkpoint, or harness.py's post-resume resolve on the first date
        # after a resume has nothing to resolve against. A checkpoint saved
        # before this field existed degrades to decision_id=None on load
        # (see _record_from_json's .get below), which harness.py already
        # handles by skipping that one resolve with a warning, not crashing.
        "decision_id": record.decision_id,
    }


def _record_from_json(payload: dict) -> BacktestRecord:
    return BacktestRecord(
        trade_date=date.fromisoformat(payload["trade_date"]),
        decision=PortfolioDecision.model_validate(payload["decision"]),
        close=payload["close"],
        adjustment_note=payload["adjustment_note"],
        trailing_high_water=payload["trailing_high_water"],
        trailing_entry_price=payload["trailing_entry_price"],
        # .get: absent on a checkpoint saved by code older than this field,
        # never a reason to fail — see the comment in _record_to_json.
        decision_id=payload.get("decision_id"),
    )


def _params(
    tickers: list[str], start: date, end: date, cadence_days: int, benchmark_ticker: str, git_commit: str | None,
) -> dict:
    """A checkpoint is only safe to resume if it was made by a run with the
    EXACT same shape — a different ticker set, date range, cadence, or
    benchmark produces a different set of trade dates and a different
    portfolio book, so blindly resuming into it would silently splice two
    unrelated replays together. `tickers` is sorted so group membership
    (not declaration order) is what's compared — `_rotate`'s per-date shift
    already makes order-within-a-date immaterial to the result.

    `git_commit` (same best-effort helper `app.backtest.experiments.
    record_experiment` already tags every finished run with) closes a gap
    checkpointing itself introduces: without it, resuming a run days after
    a crash — after a prompt/pipeline code change has since landed — would
    silently splice dates decided under the OLD code with dates decided
    under the NEW one into what looks like a single, comparable backtest
    result. None (git unavailable) still compares equal to None, matching
    every OTHER best-effort git tag in this codebase — this is a resume
    safety net, not a hard requirement to have git at all.

    This function no longer COMPUTES git_commit itself — it's passed in
    explicitly by the caller. `load_checkpoint`'s resume-gate call always
    passes a freshly re-evaluated `current_git_commit()` (that fresh
    re-evaluation, at the moment of resume, IS the safety check). Every
    `save_checkpoint` call during a run instead passes a value FROZEN once
    for the life of that run (see `CheckpointState.git_commit`'s own
    docstring for why re-deriving it on every save defeated this whole
    mechanism)."""
    return {
        "tickers": sorted(tickers),
        "start": start.isoformat(),
        "end": end.isoformat(),
        "cadence_days": cadence_days,
        "benchmark_ticker": benchmark_ticker,
        "git_commit": git_commit,
    }


async def load_checkpoint(
    label: str | None, tickers: list[str], start: date, end: date, cadence_days: int, benchmark_ticker: str
) -> CheckpointState | None:
    """Returns None (safe to start a fresh run) unless `label` is given AND
    a checkpoint exists under it AND that checkpoint's params match this
    invocation exactly — every other case degrades to a fresh run rather
    than risk resuming into the wrong state, matching this project's
    general "fail-closed on ambiguity" convention for anything that could
    otherwise silently corrupt the ledger's positions."""
    if not label:
        return None
    try:
        pool = await get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT params, state, completed_date_index FROM backtest_checkpoints WHERE label = $1", label
            )
    except Exception:
        logger.warning("could not read backtest checkpoint for label=%s — starting fresh", label, exc_info=True)
        return None
    if row is None:
        return None

    expected = _params(tickers, start, end, cadence_days, benchmark_ticker, git_commit=current_git_commit())
    stored_params = json.loads(row["params"])
    if stored_params != expected:
        logger.warning(
            "backtest checkpoint for label=%s doesn't match this run's parameters "
            "(checkpoint=%s, this run=%s) — ignoring it and starting fresh",
            label, stored_params, expected,
        )
        return None

    try:
        state = json.loads(row["state"])
        return CheckpointState(
            completed_date_index=row["completed_date_index"],
            prev_records={t: _record_from_json(r) for t, r in state["prev_records"].items()},
            per_ticker_records={t: [_record_from_json(r) for r in rs] for t, rs in state["per_ticker_records"].items()},
            per_ticker_equity_curves=state["per_ticker_equity_curves"],
            portfolio_equity_curve=state["portfolio_equity_curve"],
            running_high_water_mark=state["running_high_water_mark"],
            ever_selected=set(state["ever_selected"]),
            cap_trigger_counts=state["cap_trigger_counts"],
            total_dates=state.get("total_dates", 0),  # .get: absent on a checkpoint saved before this field existed
            # .get: all three absent on a checkpoint saved before mid-date
            # resume existed — -1/[]/set() means "no in-flight date", the
            # same as a checkpoint saved right after a date's full rotation.
            in_flight_date_index=state.get("in_flight_date_index", -1),
            in_flight_rotated_tickers=list(state.get("in_flight_rotated_tickers", [])),
            completed_tickers_this_date=set(state.get("completed_tickers_this_date", [])),
            starting_total_equity_usd=state.get("starting_total_equity_usd", 0.0),
            # .get with no explicit default: None is already this field's
            # own dataclass default (a checkpoint saved before this field
            # existed correctly degrades to None, same outcome as every
            # other backward-compat .get here, just without a redundant
            # literal default to keep in sync with the dataclass one).
            git_commit=state.get("git_commit"),
        )
    except Exception:
        logger.warning(
            "backtest checkpoint for label=%s matched params but failed to deserialize — starting fresh",
            label, exc_info=True,
        )
        return None


async def save_checkpoint(
    label: str | None,
    tickers: list[str],
    start: date,
    end: date,
    cadence_days: int,
    benchmark_ticker: str,
    state: CheckpointState,
) -> None:
    """Best-effort, same convention as app.live.alerts.record_alert — a
    checkpoint write failure must never abort the replay it exists to
    protect. Called once per fully-completed trade date AND once per ticker
    within an in-flight date (see portfolio_harness's own call sites and
    this module's docstring on ticker-granular resume), so a failure here
    just means this one save isn't durable yet, not that the whole run is
    lost; the next call tries again."""
    if not label:
        return
    try:
        params = _params(tickers, start, end, cadence_days, benchmark_ticker, git_commit=state.git_commit)
        payload = {
            "prev_records": {t: _record_to_json(r) for t, r in state.prev_records.items()},
            "per_ticker_records": {t: [_record_to_json(r) for r in rs] for t, rs in state.per_ticker_records.items()},
            "per_ticker_equity_curves": state.per_ticker_equity_curves,
            "portfolio_equity_curve": state.portfolio_equity_curve,
            "running_high_water_mark": state.running_high_water_mark,
            "ever_selected": sorted(state.ever_selected),
            "cap_trigger_counts": state.cap_trigger_counts,
            "total_dates": state.total_dates,
            "in_flight_date_index": state.in_flight_date_index,
            "in_flight_rotated_tickers": state.in_flight_rotated_tickers,
            "completed_tickers_this_date": sorted(state.completed_tickers_this_date),
            "starting_total_equity_usd": state.starting_total_equity_usd,
            "git_commit": state.git_commit,
        }
        pool = await get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO backtest_checkpoints (label, params, state, completed_date_index, updated_at)
                VALUES ($1, $2::jsonb, $3::jsonb, $4, now())
                ON CONFLICT (label) DO UPDATE
                SET params = $2::jsonb, state = $3::jsonb, completed_date_index = $4, updated_at = now()
                """,
                label, json.dumps(params), json.dumps(payload), state.completed_date_index,
            )
    except Exception:
        logger.warning("could not persist backtest checkpoint for label=%s — continuing without it", label, exc_info=True)


async def clear_checkpoint(label: str | None) -> None:
    """Called once the replay finishes normally (see portfolio_harness's
    own call site, right alongside record_experiment) — a leftover row
    would otherwise look like an unfinished run to a LATER invocation that
    happens to reuse the same label, silently resuming into an already-
    complete run's tail state instead of starting the new one fresh.
    Best-effort: a failure here is cosmetic (one stale row), never worth
    risking the result that already finished."""
    if not label:
        return
    try:
        pool = await get_pool()
        async with pool.acquire() as conn:
            await conn.execute("DELETE FROM backtest_checkpoints WHERE label = $1", label)
    except Exception:
        logger.warning("could not clear backtest checkpoint for label=%s (cosmetic — leaves one stale row)", label, exc_info=True)


async def list_checkpoints() -> list[CheckpointProgress]:
    """Every currently-unfinished checkpointed run — a row here means a run
    under that label started and hasn't (yet) reached `clear_checkpoint`,
    which fires only on normal completion. Most-recently-updated first, the
    same "what's happening right now" ordering `app.backtest.experiments.
    list_experiments` uses for finished runs. On-demand/reporting only
    (same convention as list_experiments): a DB failure here should be
    visible to whoever ran the CLI, not silently swallowed."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT label, params, state, completed_date_index, updated_at "
            "FROM backtest_checkpoints ORDER BY updated_at DESC"
        )
    progress: list[CheckpointProgress] = []
    for row in rows:
        params = json.loads(row["params"])
        state = json.loads(row["state"])
        equity_curve = state.get("portfolio_equity_curve") or []
        progress.append(
            CheckpointProgress(
                label=row["label"],
                tickers=params.get("tickers", []),
                completed_date_index=row["completed_date_index"],
                total_dates=state.get("total_dates", 0),
                latest_portfolio_equity=equity_curve[-1] if equity_curve else None,
                updated_at=row["updated_at"],
            )
        )
    return progress


def render_checkpoints(progress: list[CheckpointProgress]) -> str:
    if not progress:
        return "No unfinished checkpointed runs (a label only appears here between its first checkpoint and completion)."

    lines = [
        f"{'Label':<32} {'Progress':>12} {'Tickers':<24} {'Equity':>12} {'Updated'}",
        "-" * 100,
    ]
    for p in progress:
        done = p.completed_date_index + 1
        progress_str = f"{done}/{p.total_dates}" if p.total_dates else f"{done}/?"
        tickers = ",".join(p.tickers)
        tickers = tickers if len(tickers) <= 24 else tickers[:21] + "..."
        equity = f"{p.latest_portfolio_equity:,.0f}" if p.latest_portfolio_equity is not None else "n/a"
        label = p.label if len(p.label) <= 32 else p.label[:29] + "..."
        lines.append(f"{label:<32} {progress_str:>12} {tickers:<24} {equity:>12} {p.updated_at:%Y-%m-%d %H:%M}")
    return "\n".join(lines)
