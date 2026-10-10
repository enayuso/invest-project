"""Multi-ticker counterpart to `app.backtest.harness` — replays the full
pipeline across the TRADE ALLOWLIST, not one ticker in isolation, so that
`app.execution.portfolio_risk`'s concentration/correlation-bucket/drawdown
caps (which only ever bind when more than one ticker's book is visible at
once) actually get exercised during a backtest instead of sitting untested
until the first real live decision.

Reuses `app.backtest.harness._process_ticker_date` for the actual
per-(ticker, date) walk (gap-stop-loss check, pipeline call, reflection) —
see that function's docstring — so this module only adds what's genuinely
different about walking many tickers at once: shared trade-date selection,
ticker rotation, per-ticker state isolation, and the aggregate portfolio
equity curve.

The portfolio-level caps are LIVE-ONLY in the sense that `portfolio_risk_state`
(the persisted high-water mark) must never be touched by a backtest — see
`app.execution.portfolio_risk`'s module docstring. This harness tracks its
own local, in-memory high-water mark instead and passes a freshly computed
(snapshot, clusters, drawdown_fraction) into `run_pipeline`'s
`portfolio_cap_override` for every single ticker decision, so the caps are
genuinely enforced within the walk without ever reading or writing that
table.

`--screened` (`screening_candidate_selector` below), added 2026-08-03,
replicates live's momentum re-ranking (app.chat.screening.top_candidates,
point-in-time) so a backtest only processes each date's candidates the same
way a live cycle would, instead of every `--tickers`/trade_allowlist entry every
date — closing the gap screening-method-improvement-ideas and this project's
own audit stats flagged: without it, a backtest could "succeed" on tickers/
dates live would never have looked at. Does not replicate the trading-value/
turnover DISCOVERY step (no historical version of that KIS ranking TR
exists) — see the selector's own docstring for that boundary. Off by
default (identical to this module's pre-2026-08-03 behavior) so existing
callers/tests are unaffected.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import date

from app.backtest.checkpoint import CheckpointState, clear_checkpoint, load_checkpoint, save_checkpoint
from app.backtest.experiments import current_git_commit, record_experiment
from app.backtest.harness import (
    _process_ticker_date,
    configure_cli_logging,
    log_progress,
    pick_random_date_range,
    select_trading_dates,
)
from app.backtest.metrics import (
    BacktestRecord,
    BacktestResult,
    RiskMetrics,
    alpha_win,
    compute_buy_and_hold_return,
    compute_ledger_period_returns,
    compute_period_returns,
    compute_strategy_return,
    cvar,
    max_drawdown,
    price_on_or_before,
    sharpe_ratio,
    sortino_ratio,
)
from app.chat.screening import top_candidates
from app.config import backtest_scope, settings
from app.dataflows.kis_provider import get_long_range_price_series as get_price_series
from app.execution.ledger import clear_backtest_positions, get_position
from app.execution.portfolio_risk import compute_correlation_clusters, portfolio_snapshot, ticker_market

logger = logging.getLogger(__name__)

# Windows ties stdout to the OS console codepage (cp949 on a Korean locale)
# unless told otherwise — this module's own report output (em dashes, etc.)
# would otherwise raise UnicodeEncodeError. Same fix already applied to
# app/backtest/__main__.py, parallel_portfolio.py, atr_sweep.py,
# screening_sweep.py, and app/audit/__main__.py — this file was the one
# sibling CLI missing it (found 2026-08-11: it crashed on '—' right
# after a full multi-hour sweep group finished analyzing, before
# record_experiment ever ran — see that call's own comment below).
if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

CandidateSelector = Callable[[date, list[str]], Awaitable[list[str]]]

# Substrings `ledger.rebalance_to_target`/`portfolio_risk.apply_portfolio_caps`
# actually write into `RebalanceResult.adjustment_note` — matched case-
# insensitively against each decision's note to tally which mechanism fired,
# for checkpoint 5's "did the new caps actually trigger" reporting. Order
# matters only in that these five phrases are mutually exclusive substrings
# of the notes those two modules produce (a multi-cap note is a space-joined
# concatenation, so more than one label can match the same note).
_CAP_LABELS: dict[str, str] = {
    "no resolvable stop-loss": "no resolvable stop-loss (blocked)",
    "1% risk cap": "1% per-ticker risk cap",
    "portfolio concentration cap": "portfolio concentration cap",
    "portfolio correlation-bucket cap": "portfolio correlation-bucket cap",
    "portfolio drawdown breaker": "portfolio drawdown breaker",
    # 2026-09-02 fail-closed ATR gate (app.execution.ledger.rebalance_to_target,
    # app.domain.executability.blocks_new_entry_without_atr) — the opening
    # phrase of the adjustment_note it emits when a new entry is refused for
    # lacking a usable 14-day ATR.
    "no new position opened this cycle": "ATR unavailable (new entry blocked)",
}


def select_shared_trading_dates(per_ticker_prices: dict[str, dict[date, float]], cadence_days: int) -> list[date]:
    """Intersection (NOT union) of every ticker's own trading-date set, then
    the exact same `select_trading_dates` subsampling the single-ticker
    harness uses, applied to the intersected list. A union would push the
    shared calendar close to "every weekday" (any one market's holiday
    would still leave every OTHER ticker with a decision date this ticker
    doesn't have) and defeat cadence_days' whole cost-bounding purpose."""
    if not per_ticker_prices:
        raise ValueError("no tickers given")
    date_sets = [set(prices) for prices in per_ticker_prices.values()]
    shared_dates = {d: 1.0 for d in sorted(set.intersection(*date_sets))}
    trade_dates = select_trading_dates(shared_dates, cadence_days)
    if len(trade_dates) < 2:
        raise ValueError(
            f"only {len(trade_dates)} shared trading date(s) across {len(per_ticker_prices)} tickers at "
            f"cadence_days={cadence_days} — need at least 2 to compute a return"
        )
    return trade_dates


def _rotate(tickers: list[str], shift: int) -> list[str]:
    """Deterministic cyclic shift by the date's index in the trade-date
    list. A fixed processing order gives whichever ticker is declared first
    in the trade_allowlist permanent first claim on the concentration/correlation
    budget on every single date — a pure artifact of declaration order, not
    a real signal. Rotation removes that bias while staying fully
    reproducible (the same date index always produces the same order)."""
    if not tickers:
        return []
    shift = shift % len(tickers)
    return tickers[shift:] + tickers[:shift]


def screening_candidate_selector(
    top_n: int,
    lookback_days: int = 10,
    use_volatility_adjustment: bool = True,
    use_correlation_dedup: bool = True,
) -> CandidateSelector:
    """Builds a `CandidateSelector` that replicates live's actual screening
    step (app.chat.screening.top_candidates, point-in-time via `as_of`)
    instead of `run_portfolio_backtest`'s default (process the whole
    `universe` every date) — screening-method-improvement-ideas's still-
    open gap: neither this harness nor the single-ticker one previously
    reproduced the momentum re-ranking that decides which candidates a live
    cycle would even bother analyzing, so a backtest could "succeed" on
    tickers/dates live would never have looked at in the first place.

    Does NOT replicate live's trading-value/turnover DISCOVERY step
    (app.live.universe.krx_top_n etc.) — that's a live-only KIS ranking TR
    with no historical/point-in-time equivalent (confirmed: no date
    parameter, no cached historical ranking table exists in this codebase).
    `universe` here is therefore a fixed candidate pool (the trade_allowlist or
    whatever `--tickers` gives), and this selector only replicates the
    MOMENTUM RE-RANKING stage applied on top of it — the same partial-
    fidelity boundary app.backtest.screening_sweep already documents for
    its own (LLM-free, pipeline-free) sweep.

    A ticker currently held (nonzero position, from the internal paper
    ledger `get_position` — same source `_process_ticker_date`'s own
    gap-stop-loss check already uses) is always included regardless of
    today's momentum rank, mirroring app.live.auto_trade_loop's
    `_held_needing_review` forced-review principle (simplified: any held
    position, not narrowed to stop-loss-proximity/top-loss-amount — the
    live version's specific thresholds aren't reproduced here)."""

    async def _select(trade_date: date, universe: list[str]) -> list[str]:
        screened = await top_candidates(
            universe,
            top_n=top_n,
            lookback_days=lookback_days,
            as_of=trade_date,
            use_volatility_adjustment=use_volatility_adjustment,
            use_correlation_dedup=use_correlation_dedup,
        )
        held = []
        for ticker in universe:
            position = await get_position(ticker)
            if position.shares > 0:
                held.append(ticker)
        return list(dict.fromkeys(screened + held))

    return _select


def _tally_cap_triggers(note: str | None, counts: dict[str, int]) -> None:
    if not note:
        return
    lowered = note.lower()
    for needle, label in _CAP_LABELS.items():
        if needle in lowered:
            counts[label] = counts.get(label, 0) + 1


def _alpha_records(records: list[BacktestRecord], benchmark_prices: dict[date, float]) -> list[tuple[str, float]]:
    """Per-decision (rating, raw_return - benchmark_return) for consecutive
    record pairs, recomputed here directly from this run's own
    records/benchmark rather than read back from the shared `memory_entries`
    table, since that table has no run-id column to scope a query to just
    this backtest window (see the pre-existing gap flagged in
    `app.execution.portfolio_risk`'s checkpoint notes).

    The rating is `prev`'s — the alpha realized between prev and nxt scores
    PREV's call, the same pairing `_process_ticker_date` uses to resolve
    `prev.decision_id`'s outcome. `.value` because `prev.decision` may be a
    `PortfolioDecision` or `FlatPortfolioDecision` (different enum classes,
    same string values)."""
    pairs = []
    for prev, nxt in zip(records, records[1:]):
        raw_return = (nxt.close - prev.close) / prev.close
        benchmark_return = (
            price_on_or_before(benchmark_prices, nxt.trade_date)
            - price_on_or_before(benchmark_prices, prev.trade_date)
        ) / price_on_or_before(benchmark_prices, prev.trade_date)
        pairs.append((prev.decision.rating.value, raw_return - benchmark_return))
    return pairs


def _alpha_series(records: list[BacktestRecord], benchmark_prices: dict[date, float]) -> list[float]:
    """Alpha values only, no rating — kept for callers that just want the
    level (e.g. `mean_alpha_return`), not the direction-aware win test."""
    return [alpha for _, alpha in _alpha_records(records, benchmark_prices)]


@dataclass
class PortfolioBacktestResult:
    tickers: list[str]
    benchmark_ticker: str
    trade_dates: list[date]
    per_ticker_results: dict[str, BacktestResult]
    portfolio_equity_curve: list[float]  # one aggregate USD mark per trade_date, after that date's full rotation
    portfolio_risk_metrics: RiskMetrics  # from portfolio_equity_curve, via the unmodified metrics.py functions
    starting_total_equity_usd: float
    final_total_equity_usd: float
    portfolio_return: float
    portfolio_benchmark_return: float
    # Direction-aware (see app.backtest.metrics.alpha_win): Buy/Overweight win
    # on alpha_return > 0, Sell/Underweight win on alpha_return < 0 (they
    # succeeded by underperforming — reduced/avoided exposure). Hold is
    # excluded from the tally entirely (held/flat is unknowable in a replay),
    # so the denominator is n_scored_decisions, not n_resolved_decisions.
    win_rate: float | None
    mean_alpha_return: float | None
    n_resolved_decisions: int
    n_scored_decisions: int = 0
    n_unscored_holds: int = 0
    cap_trigger_counts: dict[str, int] = field(default_factory=dict)


async def run_portfolio_backtest(
    tickers: list[str],
    start: date,
    end: date,
    cadence_days: int = 5,
    benchmark_ticker: str = "SPY",
    candidate_selector: CandidateSelector | None = None,
    label: str | None = None,
) -> PortfolioBacktestResult:
    """`candidate_selector`, if given, narrows which of `tickers` actually
    get processed on each `trade_date` (e.g. `screening_candidate_selector`
    above, to replicate live's momentum re-ranking instead of processing
    the whole universe every date). Defaults to None — every ticker every
    date, this function's original behavior — so existing callers are
    unaffected.

    `label`, if given, turns on checkpoint/resume (app.backtest.checkpoint)
    — after every fully-completed trade date, this run's progress is saved
    under that label, and a matching prior checkpoint (same tickers/date
    range/cadence/benchmark) is resumed from instead of replaying every
    date from scratch. Defaults to None — no checkpointing at all, this
    function's original behavior — so a caller that never sets a label
    (there's only ever been the one, `_main` below) is unaffected."""
    # The raw `label` (`--label`), NOT this key, is what record_experiment
    # tags every finished run's row with — kept as the user typed it so a
    # query like "every backtest_runs row for this baseline sweep" still
    # works across app.backtest.parallel_portfolio's several concurrent
    # groups, which all share ONE `--label` on purpose. That sharing is
    # exactly what would corrupt the checkpoint table if used as-is:
    # `backtest_checkpoints.label` is a PRIMARY KEY, so two groups
    # (disjoint ticker subsets, same raw label, separate OS processes)
    # checkpointing under the identical key would clobber each other's
    # row — whichever save() lands last wins the slot, and the loser's
    # resumability for its own trade dates is gone (safely refused on
    # resume by _params()'s tickers check, just wastefully). Appending the
    # sorted ticker set gives each group's own subset a distinct row
    # automatically, with zero coordination needed between the parent and
    # its children — pick_groups already guarantees disjoint tickers.
    checkpoint_key = f"{label}:{','.join(sorted(tickers))}" if label else None
    per_ticker_prices = {ticker: await get_price_series(ticker, start, end) for ticker in tickers}
    benchmark_prices = await get_price_series(benchmark_ticker, start, end)

    trade_dates = select_shared_trading_dates(per_ticker_prices, cadence_days)
    log_progress(
        "replay_start", tickers=len(tickers), dates=len(trade_dates), range=f"{start}..{end}", cadence=cadence_days
    )

    checkpoint = await load_checkpoint(checkpoint_key, tickers, start, end, cadence_days, benchmark_ticker)

    # Freeze this run's git_commit ONCE, mirroring starting_total_equity_
    # usd's own resolution below — see CheckpointState.git_commit's own
    # docstring for the bug this fixes (re-deriving current_git_commit() on
    # every save_checkpoint call let a mid-run deploy silently re-stamp
    # later saves, defeating load_checkpoint's own resume-safety gate above).
    if checkpoint is not None and checkpoint.git_commit is not None:
        # Resumed run: reuse the checkpoint's own frozen commit verbatim —
        # already validated to match current HEAD by load_checkpoint's own
        # gate above; recomputing it again here would reopen exactly the
        # mid-run-drift hole this whole mechanism exists to close. None
        # means either git is genuinely unavailable (a fresh capture below
        # would ALSO be None, so no observable difference) or this
        # checkpoint predates the field (backward-compat degrade, same
        # convention as every other field here).
        # 재개 시엔 체크포인트에 저장된 값을 그대로 재사용 — 다시 계산하지 않음.
        run_git_commit = checkpoint.git_commit
    else:
        run_git_commit = current_git_commit()

    if checkpoint is None:
        # Wipes any stale is_backtest=true rows a PRIOR run (e.g. a different
        # candidate in the same app.backtest.atr_sweep, replaying the exact same
        # tickers/date range) left behind — without this, this run would
        # silently inherit that run's ending positions instead of starting
        # flat, breaking a parameter sweep's "identical starting point per
        # candidate" premise. See clear_backtest_positions's own docstring.
        #
        # Skipped entirely when RESUMING a checkpoint — the whole point of a
        # resume is to keep the same simulated positions the killed
        # process's earlier trade dates already committed; clearing them
        # here would defeat the checkpoint before the loop below even
        # starts. See app.backtest.checkpoint's own module docstring.
        await clear_backtest_positions(tickers)

    # Keyed by ticker, not a flat list — each ticker's own `prev` record
    # (for its gap-stop-loss check and period-return computation) must stay
    # correctly isolated across the flattened (date x rotated-ticker) walk.
    # "Whatever ran immediately before in the loop" is a DIFFERENT ticker
    # under any per-date rotation, so that shortcut would gap-check and
    # reflect a ticker's decision against another ticker's prior record.
    if checkpoint is not None:
        log_progress(
            "checkpoint_resumed", label=label,
            from_date_index=checkpoint.completed_date_index + 1, of=len(trade_dates),
        )
        prev_records: dict[str, BacktestRecord] = dict(checkpoint.prev_records)
        per_ticker_records: dict[str, list[BacktestRecord]] = {
            ticker: list(checkpoint.per_ticker_records.get(ticker, [])) for ticker in tickers
        }
        per_ticker_equity_curves: dict[str, list[float]] = {
            ticker: list(checkpoint.per_ticker_equity_curves.get(ticker, [])) for ticker in tickers
        }
        portfolio_equity_curve: list[float] = list(checkpoint.portfolio_equity_curve)
        cap_trigger_counts: dict[str, int] = dict(checkpoint.cap_trigger_counts)
        running_high_water_mark = checkpoint.running_high_water_mark
        ever_selected: set[str] = set(checkpoint.ever_selected)
        resume_from_date_index = checkpoint.completed_date_index + 1
        # Ticker-granular mid-date resume (see checkpoint.py's module
        # docstring) — only trust in_flight_* if it actually points at the
        # date this run is about to resume into; anything else (stale, or a
        # checkpoint saved right after a date's full rotation) falls back
        # to reprocessing the whole date, matching pre-existing behavior.
        if checkpoint.in_flight_date_index == resume_from_date_index:
            resume_in_flight_rotation = list(checkpoint.in_flight_rotated_tickers)
            resume_in_flight_done = set(checkpoint.completed_tickers_this_date)
        else:
            resume_in_flight_rotation = []
            resume_in_flight_done = set()
    else:
        prev_records = {}
        per_ticker_records = {ticker: [] for ticker in tickers}
        per_ticker_equity_curves = {ticker: [] for ticker in tickers}
        portfolio_equity_curve = []
        cap_trigger_counts = {}
        # Local, in-memory (or, once checkpointed, Postgres) only — never
        # reads/writes the live-only `portfolio_risk_state` table (see
        # module docstring).
        running_high_water_mark = 0.0
        # Tracks every ticker that was ever actually selected on some date, so
        # a screened run can honestly report which of `tickers` never got a
        # single look (no silent caps — see this project's own workflow
        # conventions) rather than that only being visible as an empty
        # per-ticker record list.
        ever_selected = set()
        resume_from_date_index = 0
        resume_in_flight_rotation = []
        resume_in_flight_done = set()

    # Every ledger read/write in this loop (candidate_selector's "already
    # held" get_position check, portfolio_snapshot's per-ticker positions,
    # _process_ticker_date's gap stop-loss and rebalance) must see this
    # replay's OWN trades table rows only — never a real live position, and
    # never leak its own speculative fills back into live's view. See
    # project memory: backtest-live-db-shared-contamination.
    with backtest_scope():
        if checkpoint is not None and checkpoint.starting_total_equity_usd:
            # Reuse verbatim — positions are no longer flat on a resume, so
            # this can't be re-snapshotted; it was already captured once,
            # while genuinely flat, by the run this resumes.
            starting_total_equity_usd = checkpoint.starting_total_equity_usd
        elif trade_dates:
            # Positions are provably flat here — clear_backtest_positions
            # ran above (fresh run) or a resume's in-flight/completed state
            # hasn't touched trade_dates[0] yet if this branch is reached
            # (checkpoint had no stored value, i.e. it predates this field)
            # — so this snapshot is a genuine starting mark. It is the SAME
            # computation every later equity mark uses (portfolio_snapshot),
            # so a ticker whose starting_cash_for() hit its raw-USD FX-
            # lookup fallback is counted at its real (possibly degraded)
            # value here too, instead of assumed at the full configured
            # starting cash — see ledger.starting_cash_for's own fallback.
            starting_total_equity_usd = (
                await portfolio_snapshot(
                    tickers, as_of=trade_dates[0],
                    price_overrides={t: per_ticker_prices[t][trade_dates[0]] for t in tickers},
                )
            ).total_value
        else:
            starting_total_equity_usd = settings.backtest_starting_cash * len(tickers)

        for date_index, trade_date in enumerate(trade_dates):
            if date_index < resume_from_date_index:
                # Already fully processed and checkpointed by a prior,
                # killed run — `date_index` is still enumerated (not
                # sliced out of trade_dates) so `_rotate`'s per-date shift
                # stays identical to what an unresumed run would have used
                # for every later date.
                continue
            # If this date was in-flight when a prior run died mid-rotation,
            # replay its EXACT persisted rotation order rather than
            # re-deriving it (candidate_selector could return a different
            # set/order now — e.g. screening_candidate_selector sees the
            # ticker the partial run already bought) — and pre-populate
            # done_this_date so those tickers' already-committed cap-bounded
            # increment isn't applied a second time (rebalance_to_target's
            # 1% cap bounds only the ADD, not the resulting total).
            if date_index == resume_from_date_index and resume_in_flight_rotation:
                rotated_tickers = resume_in_flight_rotation
                done_this_date = set(resume_in_flight_done)
                ever_selected.update(rotated_tickers)
            else:
                todays_tickers = await candidate_selector(trade_date, tickers) if candidate_selector else tickers
                ever_selected.update(todays_tickers)
                rotated_tickers = _rotate(todays_tickers, date_index)
                done_this_date = set()
            # Cached by (tickers, as_of) inside compute_correlation_clusters
            # itself, so hoisting this above the per-ticker loop just avoids
            # redundant cache lookups rather than being strictly required —
            # still the expensive part (one yfinance history fetch per ticker
            # on a cache miss), so it's computed once per date, not per ticker.
            clusters = await compute_correlation_clusters(tickers, as_of=trade_date)

            # This date's close is already known for every ticker (fetched
            # upfront into per_ticker_prices) and can't change as the day's
            # rotation proceeds — only positions can. Passed into every
            # portfolio_snapshot call below so it skips re-fetching the same
            # historical close from yfinance on each of the (len(tickers) + 1)
            # snapshot calls this one date makes.
            price_overrides = {t: per_ticker_prices[t][trade_date] for t in tickers}

            for ticker in rotated_tickers:
                if ticker in done_this_date:
                    # Already committed pre-crash — see the mid-date resume
                    # comment above the rotated_tickers assignment.
                    continue
                # Position is always fetched fresh (portfolio_snapshot never
                # caches positions): a ticker decided later in today's rotation
                # must see the trades every earlier ticker in today's rotation
                # already recorded.
                snapshot = await portfolio_snapshot(tickers, as_of=trade_date, price_overrides=price_overrides)
                if snapshot.total_value > running_high_water_mark:
                    running_high_water_mark = snapshot.total_value
                    drawdown_frac = 0.0
                else:
                    drawdown_frac = (
                        (running_high_water_mark - snapshot.total_value) / running_high_water_mark
                        if running_high_water_mark > 0
                        else 0.0
                    )

                prev = prev_records.get(ticker)
                record = await _process_ticker_date(
                    ticker,
                    trade_date,
                    prev,
                    per_ticker_prices[ticker],
                    benchmark_prices,
                    benchmark_ticker,
                    portfolio_cap_override=(snapshot, clusters, drawdown_frac),
                )
                per_ticker_records[ticker].append(record)
                prev_records[ticker] = record
                _tally_cap_triggers(record.adjustment_note, cap_trigger_counts)

                position = await get_position(ticker)
                per_ticker_equity_curves[ticker].append(position.market_value(per_ticker_prices[ticker][trade_date]))
                done_this_date.add(ticker)

                # Ticker-granular checkpoint (see checkpoint.py's module
                # docstring) — best-effort, same as the end-of-date save
                # below: a failure here just means this one ticker's
                # progress isn't durable yet. completed_date_index stays at
                # date_index - 1 (the last FULLY completed date; -1 at the
                # very first date, matching the fresh-run sentinel) since
                # this date itself is still in flight.
                await save_checkpoint(
                    checkpoint_key, tickers, start, end, cadence_days, benchmark_ticker,
                    CheckpointState(
                        completed_date_index=date_index - 1,
                        prev_records=prev_records,
                        per_ticker_records=per_ticker_records,
                        per_ticker_equity_curves=per_ticker_equity_curves,
                        portfolio_equity_curve=portfolio_equity_curve,
                        running_high_water_mark=running_high_water_mark,
                        ever_selected=ever_selected,
                        cap_trigger_counts=cap_trigger_counts,
                        total_dates=len(trade_dates),
                        in_flight_date_index=date_index,
                        in_flight_rotated_tickers=list(rotated_tickers),
                        completed_tickers_this_date=set(done_this_date),
                        starting_total_equity_usd=starting_total_equity_usd,
                        git_commit=run_git_commit,
                    ),
                )

            # One portfolio-level equity mark per date — computed once ALL
            # tickers in today's rotation have been decided, not per ticker, so
            # it reflects the day's fully-settled aggregate book rather than a
            # partially-updated intermediate state.
            end_of_date_snapshot = await portfolio_snapshot(tickers, as_of=trade_date, price_overrides=price_overrides)
            portfolio_equity_curve.append(end_of_date_snapshot.total_value)

            # Checkpoint AFTER this date's rotation is fully committed
            # (every ticker decided, the ledger's trades table already has
            # every fill, the aggregate equity mark above already appended)
            # — never mid-rotation, so a resume never has to reconstruct a
            # partially-decided date. Best-effort (see save_checkpoint's own
            # docstring): a save failure here costs at most this one date's
            # worth of resumability, never the replay itself.
            await save_checkpoint(
                checkpoint_key, tickers, start, end, cadence_days, benchmark_ticker,
                CheckpointState(
                    completed_date_index=date_index,
                    prev_records=prev_records,
                    per_ticker_records=per_ticker_records,
                    per_ticker_equity_curves=per_ticker_equity_curves,
                    portfolio_equity_curve=portfolio_equity_curve,
                    running_high_water_mark=running_high_water_mark,
                    ever_selected=ever_selected,
                    cap_trigger_counts=cap_trigger_counts,
                    total_dates=len(trade_dates),
                    # date_index is now FULLY completed -- clear the
                    # in-flight marker atomically with that completion.
                    in_flight_date_index=-1,
                    in_flight_rotated_tickers=[],
                    completed_tickers_this_date=set(),
                    starting_total_equity_usd=starting_total_equity_usd,
                    git_commit=run_git_commit,
                ),
            )

    # Replay finished normally — a checkpoint left under this label would
    # otherwise look like an unfinished run to a LATER invocation reusing
    # the same label. See clear_checkpoint's own docstring.
    await clear_checkpoint(checkpoint_key)

    never_selected = [t for t in tickers if t not in ever_selected]
    if never_selected:
        log_progress("never_screened_in", tickers=never_selected, count=len(never_selected))

    per_ticker_results: dict[str, BacktestResult] = {}
    all_scored: list[tuple[str, float]] = []
    for ticker in tickers:
        records = per_ticker_records[ticker]
        if not records:
            # Only reachable with a candidate_selector that never once
            # picked this ticker across the whole window — nothing to
            # compute a BacktestResult from (compute_period_returns/
            # compute_buy_and_hold_return both assume at least one record).
            # Already surfaced above via `never_screened_in`, not silently
            # dropped.
            continue
        ticker_prices = per_ticker_prices[ticker]

        period_returns = compute_period_returns(records)
        exposure_risk_metrics = RiskMetrics(
            sharpe=sharpe_ratio(period_returns),
            sortino=sortino_ratio(period_returns),
            max_drawdown=max_drawdown(period_returns),
            cvar_95=cvar(period_returns),
        )

        ledger_period_returns = compute_ledger_period_returns(per_ticker_equity_curves[ticker])
        ledger_risk_metrics = RiskMetrics(
            sharpe=sharpe_ratio(ledger_period_returns),
            sortino=sortino_ratio(ledger_period_returns),
            max_drawdown=max_drawdown(ledger_period_returns),
            cvar_95=cvar(ledger_period_returns),
        )

        per_ticker_results[ticker] = BacktestResult(
            ticker=ticker,
            benchmark_ticker=benchmark_ticker,
            records=records,
            strategy_return=compute_strategy_return(records),
            buy_and_hold_return=compute_buy_and_hold_return(ticker_prices, trade_dates[0], trade_dates[-1]),
            benchmark_return=compute_buy_and_hold_return(benchmark_prices, trade_dates[0], trade_dates[-1]),
            exposure_risk_metrics=exposure_risk_metrics,
            ledger_risk_metrics=ledger_risk_metrics,
        )
        all_scored.extend(_alpha_records(records, benchmark_prices))

    portfolio_period_returns = compute_ledger_period_returns(portfolio_equity_curve)
    portfolio_risk_metrics = RiskMetrics(
        sharpe=sharpe_ratio(portfolio_period_returns),
        sortino=sortino_ratio(portfolio_period_returns),
        max_drawdown=max_drawdown(portfolio_period_returns),
        cvar_95=cvar(portfolio_period_returns),
    )

    # starting_total_equity_usd was already computed above (inside the
    # `with backtest_scope():` block, before any date's rotation) — a real
    # portfolio_snapshot at trade_dates[0] while every ticker was provably
    # flat, not a blind settings.backtest_starting_cash * len(tickers)
    # multiply (see that computation's own comment for why: a ticker whose
    # starting_cash_for() hit its raw-USD FX-lookup fallback must be
    # counted at its real, possibly-degraded starting value, not assumed
    # at the full configured amount).
    final_total_equity_usd = portfolio_equity_curve[-1] if portfolio_equity_curve else starting_total_equity_usd
    portfolio_return = (
        (final_total_equity_usd - starting_total_equity_usd) / starting_total_equity_usd
        if starting_total_equity_usd
        else 0.0
    )

    # Hold decisions are unscorable (held/flat unknown in a replay — see
    # alpha_win) and excluded from `wins` entirely, not counted as losses.
    wins = [w for rating, alpha in all_scored if (w := alpha_win(rating, alpha)) is not None]

    return PortfolioBacktestResult(
        tickers=tickers,
        benchmark_ticker=benchmark_ticker,
        trade_dates=trade_dates,
        per_ticker_results=per_ticker_results,
        portfolio_equity_curve=portfolio_equity_curve,
        portfolio_risk_metrics=portfolio_risk_metrics,
        starting_total_equity_usd=starting_total_equity_usd,
        final_total_equity_usd=final_total_equity_usd,
        portfolio_return=portfolio_return,
        portfolio_benchmark_return=compute_buy_and_hold_return(benchmark_prices, trade_dates[0], trade_dates[-1]),
        win_rate=(sum(wins) / len(wins)) if wins else None,
        mean_alpha_return=(sum(a for _, a in all_scored) / len(all_scored)) if all_scored else None,
        n_resolved_decisions=len(all_scored),
        n_scored_decisions=len(wins),
        n_unscored_holds=len(all_scored) - len(wins),
        cap_trigger_counts=cap_trigger_counts,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay the trading-agent pipeline across the full trade_allowlist at once, coordinating "
        "portfolio-level risk caps (concentration, correlation-bucket exposure, aggregate drawdown breaker) "
        "the same way live trading would.\n\n"
        "    python -m app.backtest.portfolio_harness --start 2026-01-01 --end 2026-04-08 --cadence-days 7\n"
        "    python -m app.backtest.portfolio_harness --start 2026-01-01 --end 2026-04-08 --screened --top-n 5"
    )
    parser.add_argument("--tickers", help="Comma-separated tickers (default: the configured trade_allowlist)")
    parser.add_argument(
        "--market", default="all", choices=["domestic", "overseas", "all"],
        help="When --tickers isn't given, restrict the trade_allowlist to KRX (.KS) or non-KRX tickers instead "
        "of pooling both into one portfolio run (domestic/overseas separation audit, 2026-07-27 — see "
        "docs/domestic-overseas-separation-plan.md). Ignored if --tickers is given explicitly.",
    )
    parser.add_argument("--start", help="YYYY-MM-DD")
    parser.add_argument("--end", help="YYYY-MM-DD")
    parser.add_argument(
        "--random", action="store_true",
        help="Pick a random past date range instead of --start/--end (tickers unaffected — "
        "still --tickers or the full trade_allowlist).",
    )
    parser.add_argument("--cadence-days", type=int, default=5)
    parser.add_argument("--benchmark", default="SPY")
    parser.add_argument("--label", default=None, help="Optional tag for this run, e.g. 'prompt-v2' (see app.audit experiments).")
    parser.add_argument(
        "--screened", action="store_true",
        help="Replicate live's momentum re-ranking (app.chat.screening.top_candidates, point-in-time) instead "
        "of processing every --tickers/trade_allowlist entry on every date — see screening_candidate_selector's "
        "own docstring for what this does and does NOT reproduce (no historical trading-value discovery). "
        "Without this flag, behavior is unchanged from before this option existed.",
    )
    parser.add_argument(
        "--top-n", type=int, default=5, help="--screened only: candidates kept per date after momentum re-ranking.",
    )
    parser.add_argument(
        "--lookback-days", type=int, default=10, help="--screened only: momentum lookback window in trading days.",
    )
    parser.add_argument(
        "--no-volatility-adjustment", action="store_true",
        help="--screened only: rank by raw momentum instead of ATR-adjusted score.",
    )
    parser.add_argument(
        "--no-correlation-dedup", action="store_true",
        help="--screened only: skip pruning over-correlated near-duplicates before the top-n cut.",
    )
    args = parser.parse_args()
    if args.random:
        if args.start or args.end:
            parser.error("--random cannot be combined with --start/--end")
    elif not (args.start and args.end):
        parser.error("--start and --end are required unless --random is given")
    return args


def _select_cli_tickers(tickers_arg: str | None, market: str, trade_allowlist: list[str]) -> list[str]:
    """Explicit `--tickers` always wins; otherwise `market` narrows the
    configured trade_allowlist to one side instead of pooling both (domestic/
    overseas separation audit, 2026-07-27 — see
    docs/domestic-overseas-separation-plan.md). `market="all"` (the CLI
    default) preserves the pre-existing pooled behavior."""
    if tickers_arg:
        return [t.strip() for t in tickers_arg.split(",")]
    if market == "all":
        return list(trade_allowlist)
    return [t for t in trade_allowlist if ticker_market(t) == market]


async def _main() -> None:
    from app.backtest import reporting  # local import — reporting.py imports this module, so a top-level

    # import here would be circular (same convention as ledger.py <-> portfolio_risk.py).
    args = _parse_args()
    configure_cli_logging()
    if not args.tickers:
        # Refresh from Postgres before falling back to the trade_allowlist — see
        # app.backtest.__main__'s identical call for why (this standalone
        # CLI never goes through app.web.state's startup, the only place
        # that normally loads it, so settings.trade_allowlist would
        # otherwise be .env.local's static seed list). Skipped when
        # --tickers is given explicitly since it wins outright either way.
        from app.web import trade_allowlist_store

        await trade_allowlist_store.load()
    tickers = _select_cli_tickers(args.tickers, args.market, settings.trade_allowlist)
    if args.random:
        start, end = pick_random_date_range()
        log_progress("random_pick", start=start, end=end)
    else:
        start, end = date.fromisoformat(args.start), date.fromisoformat(args.end)
    selector = (
        screening_candidate_selector(
            top_n=args.top_n,
            lookback_days=args.lookback_days,
            use_volatility_adjustment=not args.no_volatility_adjustment,
            use_correlation_dedup=not args.no_correlation_dedup,
        )
        if args.screened
        else None
    )
    result = await run_portfolio_backtest(
        tickers=tickers,
        start=start,
        end=end,
        cadence_days=args.cadence_days,
        benchmark_ticker=args.benchmark,
        candidate_selector=selector,
        label=args.label,
    )
    # Recorded BEFORE the report is printed — a rendering/console-encoding
    # failure below must never cost a result that already finished the full
    # (potentially hours-long) replay. Found the hard way 2026-08-11: this
    # used to run after print(), so a Windows console UnicodeEncodeError on
    # the report's own em dashes killed the process before a single row ever
    # reached backtest_runs, on every group, every retry.
    try:
        exp_id = await record_experiment(
            kind="portfolio",
            tickers=tickers,
            benchmark_ticker=args.benchmark,
            start=start,
            end=end,
            cadence_days=args.cadence_days,
            n_dates=len(result.trade_dates),
            return_pct=result.portfolio_return,
            benchmark_return_pct=result.portfolio_benchmark_return,
            risk_metrics=result.portfolio_risk_metrics,
            win_rate=result.win_rate,
            mean_alpha=result.mean_alpha_return,
            label=args.label or ("screened" if args.screened else None),
        )
        log_progress("experiment_recorded", id=exp_id)
    except Exception as exc:
        log_progress("experiment_record_failed", error=str(exc))

    print(await reporting.render_portfolio(result))


if __name__ == "__main__":
    asyncio.run(_main())
