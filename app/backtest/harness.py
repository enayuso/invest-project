"""Replays the Phase-1 pipeline across historical trading dates with
point-in-time data enforcement (see `app/dataflows/*_provider.py`), then
scores the resulting decisions against buy-and-hold and a benchmark.

Each trading date is a full 4-analyst + 2-debate + 3-decision-node pipeline
run against the live LLM — not free or instant — so `cadence_days` exists to
bound cost by subsampling actual trading dates rather than replaying every
single day.
"""

from __future__ import annotations

import asyncio
import logging
import random
from datetime import date, timedelta
from pathlib import Path

from app.backtest.metrics import (
    BacktestRecord,
    BacktestResult,
    RiskMetrics,
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
from app.config import AIProfile, backtest_scope, settings
from app.dataflows.kis_provider import (
    get_atr_pct,
    get_daily_high_series,
    get_daily_low_series,
    get_long_range_price_series as get_price_series,
)
from app.execution import trailing_stop
from app.execution.ledger import (
    check_and_apply_stop_loss,
    check_and_apply_trailing_stop,
    clear_backtest_positions,
    get_open_stop_loss,
    get_open_stop_loss_entry_time,
    get_position,
)
from app.execution.portfolio_risk import PortfolioSnapshot
from app.memory.reflection import reflect
from app.memory.store import resolve_outcome
from app.orchestrator.pipeline import run_pipeline
from app.state import PipelineState, render_portfolio_decision

logger = logging.getLogger(__name__)


def configure_cli_logging(log_path: str = "backtest.log") -> None:
    """Routes this run's log noise (vendor-fallback notices, INJECTION_SUSPECTED
    false-positive flags, gap-check failures) to a file instead of the
    console, so the per-date `print()` progress lines in run_backtest/
    run_portfolio_backtest are the only thing visible while a long backtest
    runs. Call once from a CLI's `_main()`, before the backtest itself —
    nothing here is needed live; a real failure already surfaces via a
    raised exception or the final report/`python -m app.audit report`."""
    root = logging.getLogger()
    root.setLevel(logging.WARNING)
    for handler in list(root.handlers):
        root.removeHandler(handler)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s"))
    root.addHandler(file_handler)


def market_subdir(ticker: str) -> str:
    """"domestic" or "overseas" by this project's one ticker-shape convention
    (`.KS` suffix) — the same domestic/overseas split already applied
    everywhere else (risk caps, cash pools, candidate discovery via
    app.live.universe) applied to log file layout too, so a batch of mixed
    KR/US backtests (app.backtest.parallel --random-n) doesn't dump every
    log flat into one directory."""
    return "domestic" if ticker.endswith(".KS") else "overseas"


def default_log_path(ticker: str, filename: str = "backtest.log") -> str:
    """`logs/domestic/<filename>` or `logs/overseas/<filename>` for `ticker`
    — creates the directory (not the file) if it doesn't exist yet, so the
    caller's own file open (e.g. configure_cli_logging's FileHandler) never
    fails with "no such directory"."""
    directory = Path("logs") / market_subdir(ticker)
    directory.mkdir(parents=True, exist_ok=True)
    return str(directory / filename)


def log_progress(event: str, **fields: object) -> None:
    """One consistent `[backtest] event key=value ...` line for every
    progress message this module and its two CLIs print — ticker/date pick,
    replay start, per-date analyze/done. A single shared format (not each
    call site inventing its own sentence) so scanning or grepping a run's
    output doesn't require parsing several different styles."""
    rendered = " ".join(f"{key}={value}" for key, value in fields.items())
    print(f"[backtest] {event}" + (f" {rendered}" if rendered else ""))


def select_trading_dates(prices: dict[date, float], cadence_days: int) -> list[date]:
    all_dates = sorted(prices)
    return all_dates[:: max(cadence_days, 1)]


def pick_random_date_range(
    min_span_days: int = 60, max_span_days: int = 120, max_lookback_days: int = 730
) -> tuple[date, date]:
    """A random past (start, end) date range — shared by both CLIs'
    `--random` flag. Plain `random`, not an LLM call: picking a range to
    test is a trivial selection task with nothing for a model's judgment
    to add, unlike the actual BUY/HOLD/SELL decision the backtest exists
    to score."""
    span_days = random.randint(min_span_days, max_span_days)
    earliest_start = date.today() - timedelta(days=max_lookback_days)
    latest_start = date.today() - timedelta(days=span_days + 1)
    start = earliest_start + timedelta(days=random.randint(0, (latest_start - earliest_start).days))
    end = start + timedelta(days=span_days)
    return start, end


def pick_random_ticker_and_range(
    min_span_days: int = 60, max_span_days: int = 120, max_lookback_days: int = 730
) -> tuple[str, date, date]:
    """Picks one trade_allowlist ticker and a random past date range, for the
    single-ticker CLI's `--random` flag — a manually re-run backtest tends
    to drift toward re-testing the same one ticker/period (AAPL, the
    README's example range) over and over, which skews
    `python -m app.audit report`'s win-rate/alpha stats toward whatever
    that one ticker/period happened to do rather than a representative
    sample."""
    ticker = random.choice(list(settings.trade_allowlist))
    start, end = pick_random_date_range(min_span_days, max_span_days, max_lookback_days)
    return ticker, start, end


async def _run_pipeline_with_retry(
    ticker: str,
    trade_date: date,
    portfolio_cap_override: tuple[PortfolioSnapshot, dict[str, int], float] | None,
) -> PipelineState:
    """Retries `run_pipeline` on ANY exception, up to
    `settings.backtest_pipeline_retry_attempts` times, sleeping
    `settings.backtest_pipeline_retry_delay_seconds` between attempts.

    A OUTAGE on the vLLM endpoint (this project has observed one lasting 91
    minutes — see project memory: llm-model-swap-gpt-oss-issue), not just a
    transient blip, outlasts the LLM client's own retry budget
    (max_retries=5 x llm_timeout_seconds, at most a few minutes total). One
    unhandled exception here previously crashed the WHOLE group process —
    unattended multi-hour runs (app.backtest.atr_sweep, left running
    overnight) lost every date replayed so far in that group, since results
    only get written to backtest_runs once at the very end of a run with no
    per-date checkpoint. This buys real (if bounded, not unlimited) extra
    time to ride an outage out before that happens — see the settings'
    own docstring for how the defaults were chosen."""
    attempts = settings.backtest_pipeline_retry_attempts
    for attempt in range(1, attempts + 1):
        try:
            return await run_pipeline(
                ticker, trade_date, as_of=trade_date, portfolio_cap_override=portfolio_cap_override,
                profile=AIProfile.PRODUCTION,
            )
        except Exception:
            if attempt == attempts:
                raise
            logger.warning(
                "%s %s: pipeline call failed (attempt %d/%d) — retrying in %.0fs",
                ticker, trade_date, attempt, attempts, settings.backtest_pipeline_retry_delay_seconds,
                exc_info=True,
            )
            await asyncio.sleep(settings.backtest_pipeline_retry_delay_seconds)
    raise AssertionError("unreachable")  # loop always returns or raises on the last attempt


async def _process_ticker_date(
    ticker: str,
    trade_date: date,
    prev: BacktestRecord | None,
    ticker_prices: dict[date, float],
    benchmark_prices: dict[date, float],
    benchmark_ticker: str,
    portfolio_cap_override: tuple[PortfolioSnapshot, dict[str, int], float] | None = None,
) -> BacktestRecord:
    """One (ticker, trade_date) step of a backtest replay: the gap
    stop-loss check against `prev`, the pipeline call itself, and — once
    `prev` exists — resolving `prev`'s outcome now that this date's close is
    known. Shared by the single-ticker `run_backtest` below and the
    multi-ticker `app.backtest.portfolio_harness`, so the two never risk
    drifting into two subtly different replay semantics.

    `prev` must be THIS ticker's own previous record. In the multi-ticker
    harness that means a `dict[str, BacktestRecord]` keyed by ticker, not
    "whatever ran immediately before in a flattened date x ticker walk" —
    under any per-date ticker rotation that's a different ticker's record,
    which would gap-check and reflect against the wrong prior decision
    entirely.

    `portfolio_cap_override`, if given, is threaded straight through to
    `run_pipeline` — `None` (the single-ticker default) means the pipeline
    falls back to `DISABLED_PORTFOLIO_CAPS` (no real cross-ticker portfolio
    to coordinate for one ticker alone); the multi-ticker harness supplies
    its own per-date local state instead.
    """
    # Check the position's actual *open* stop-loss (the ledger's, frozen
    # at whatever the last real Buy/Sell set it to) against every
    # intervening trading day's low — not just the sampled dates —
    # *before* running the next decision, so that decision's own
    # rebalance sees a correctly up-to-date position if a stop-out
    # already happened in the gap. cadence_days samples decisions, not
    # price monitoring.
    #
    # Deliberately NOT `prev.decision.stop_loss`: a Hold never writes a
    # trade row, so the Portfolio Manager's freshly re-issued stop_loss on a
    # Hold decision is never actually enforced live (the live path only
    # ever reads the ledger's open stop via `get_open_stop_loss`, which is
    # untouched by a Hold). Using the raw per-decision field here silently
    # re-tightened the operative stop on every Hold in backtests only — a
    # real backtest/live semantic mismatch that produced misleadingly
    # frequent stop-outs (confirmed on NVDA: exposure-proxy return +13.98%
    # vs. real ledger +0.17%, entirely explained by this divergence, not
    # genuine whipsaw trading).
    trailing_high_water = prev.trailing_high_water if prev is not None else None
    trailing_entry_price = prev.trailing_entry_price if prev is not None else None
    if prev is not None:
        open_stop = await get_open_stop_loss(ticker)
        if open_stop is not None:
            _, stop_loss = open_stop
            try:
                gap_lows = await get_daily_low_series(ticker, prev.trade_date, trade_date)
                gap_lows = {d: low for d, low in gap_lows.items() if d > prev.trade_date}

                # Trailing-stop replay (2026-08-10) — same ratchet the live
                # exit_monitor applies, replayed day by day here entirely
                # in memory (trailing_high_water/trailing_entry_price are
                # this loop's own state, threaded record-to-record via
                # BacktestRecord — see that dataclass's own docstring for
                # why this never touches the live trailing_stop_state
                # table). get_open_stop_loss_entry_time's `entry_time`
                # return value is used only as an OPAQUE identity marker
                # here (has this position's underlying row changed since
                # last gap?), never as a wall-clock comparison — unlike
                # app.live.exit_monitor, a backtest's simulated trade_date
                # has no relationship to when this replay actually executes.
                entry_info = await get_open_stop_loss_entry_time(ticker)
                if entry_info is None:
                    # No entry price resolvable (e.g. stop-loss row exists
                    # but the read failed) — fall back to the plain check,
                    # same as before this feature existed.
                    await check_and_apply_stop_loss(
                        ticker, gap_lows, stop_loss, reason=f"gap check before {trade_date}"
                    )
                else:
                    _, entry_price, _ = entry_info
                    if trailing_high_water is None or trailing_entry_price != entry_price:
                        # First time seeing an open stop for this ticker, or
                        # the underlying position changed (add-on/reopen)
                        # since the last gap — re-baseline, mirroring
                        # trailing_stop.update_high_water_mark's own
                        # entry_time-mismatch reset for the live path.
                        trailing_high_water = entry_price
                        trailing_entry_price = entry_price
                    gap_highs = await get_daily_high_series(ticker, prev.trade_date, trade_date)
                    gap_highs = {d: high for d, high in gap_highs.items() if d > prev.trade_date}
                    try:
                        atr_pct = await get_atr_pct(ticker, as_of=trade_date)
                    except Exception:
                        atr_pct = None
                    trailing_pct = trailing_stop.compute_trailing_pct(atr_pct)
                    _, trailing_high_water = await check_and_apply_trailing_stop(
                        ticker, gap_highs, gap_lows, stop_loss, entry_price, trailing_high_water, trailing_pct,
                        reason=f"gap check before {trade_date}",
                    )
            except ValueError:
                # A best-effort safety check layered on top of the sampled
                # decisions, not a core requirement of the replay — a single
                # gap the price vendor (KIS, yfinance fallback) returns no
                # data for (rate limit, transient hiccup) shouldn't blow up a
                # multi-week backtest and discard everything replayed so far.
                logger.warning(
                    "gap stop-loss check failed for %s between %s and %s — skipping this gap",
                    ticker, prev.trade_date, trade_date, exc_info=True,
                )

    # Printed before the (~20-30s) pipeline call below, not just after —
    # without this, the terminal shows nothing at all while that call is
    # in flight, which reads as "did it hang?" on every single date.
    log_progress("analyzing", date=trade_date, ticker=ticker, stages="4 analysts+2 debates+3 decisions")

    # Pinned to PRODUCTION regardless of settings.ai_profile — a backtest must
    # stay deterministic and comparable across runs whatever profile a
    # developer's local env happens to be testing the live pipeline with.
    # 백테스트는 로컬 ai_profile 설정과 무관하게 항상 PRODUCTION 고정.
    state = await _run_pipeline_with_retry(ticker, trade_date, portfolio_cap_override)
    if state.final_decision is None:
        raise RuntimeError(f"pipeline produced no final decision for {trade_date}")
    record = BacktestRecord(
        trade_date=trade_date,
        decision=state.final_decision,
        close=ticker_prices[trade_date],
        adjustment_note=state.rebalance_adjustment_note,
        trailing_high_water=trailing_high_water,
        trailing_entry_price=trailing_entry_price,
        decision_id=state.decision_id,
    )
    # Completion line pairs with the "analyzing" line printed before the
    # pipeline call above — printed (not logged) so it shows up even with
    # no logging config set up, same convention as the CLI's own final
    # `print(reporting.render(...))`.
    conf = state.final_decision.confidence
    log_progress(
        "done", date=trade_date, ticker=ticker, rating=state.final_decision.rating.value,
        conf=f"{conf:.2f}" if conf is not None else "n/a",
    )

    # The harness already walks trade dates in order with known future
    # closes, so as soon as this record lands, `prev`'s pending memory entry
    # (written by run_pipeline -> store_pending) can be resolved — no
    # separate deferred step needed here, unlike the live single-run CLI in
    # `app/memory/__main__.py`. The final date stays pending, same as
    # TradingAgents' own semantics (no future close to resolve into yet).
    if prev is not None:
        if prev.decision_id is None:
            # store_pending failed for prev's own day (DB hiccup — degrades
            # quietly, see app.memory.store's module docstring) — there is
            # no row to resolve by id, and resolving by (ticker, trade_date)
            # is exactly the unsafe fallback this fix removed (see
            # app.memory.store.resolve_outcome's own docstring). Same
            # "fail-open, don't crash a multi-year backtest over one missing
            # id" convention this function already uses for a missing price
            # (the `except (KeyError, ZeroDivisionError)` branch below).
            logger.warning(
                "no decision_id recorded for %s %s — skipping resolve_outcome (store_pending likely failed "
                "that day)",
                ticker, prev.trade_date,
            )
            return record
        try:
            # `price_on_or_before`, not direct indexing: the ticker's and
            # benchmark's trading calendars can diverge (e.g. a KRX ticker's
            # trading day landing on a US holiday SPY has no price for) — this
            # can raise KeyError if no earlier benchmark price exists at all.
            benchmark_prev = price_on_or_before(benchmark_prices, prev.trade_date)
            benchmark_now = price_on_or_before(benchmark_prices, record.trade_date)
            if prev.close == 0 or benchmark_prev == 0:
                raise ZeroDivisionError("zero close/benchmark price")
            raw_return = (record.close - prev.close) / prev.close
            benchmark_return = (benchmark_now - benchmark_prev) / benchmark_prev
        except (KeyError, ZeroDivisionError):
            # A vendor data glitch (literal 0.0 close) or a benchmark with no
            # earlier trading day at all shouldn't abort a multi-week/-year
            # backtest and discard everything replayed so far — same
            # "skip this one, keep going" convention as the gap stop-loss
            # check above. This entry just stays pending, unresolved.
            logger.warning(
                "return/alpha calc failed for %s between %s and %s — leaving pending",
                ticker, prev.trade_date, record.trade_date, exc_info=True,
            )
            return record

        alpha_return = raw_return - benchmark_return
        holding_days = (record.trade_date - prev.trade_date).days

        # held_at_decision=None (not derived from anything BacktestRecord
        # tracks): trailing_entry_price/trailing_high_water reflect
        # post-trade stop-loss replay state as of THIS iteration's gap
        # check, i.e. AFTER prev's own decision already executed — not
        # whether a position was open BEFORE prev's decision was made,
        # which is the question app.orchestrator.pipeline's own
        # `_position_context` answers live via a point-in-time `get_
        # position` call that this harness never captures into
        # BacktestRecord. Reusing either field here would misdescribe
        # exactly the cases where the Hold-direction distinction matters
        # (a Buy-from-flat or Sell-to-flat on `prev` itself). `reflect`
        # already treats `held_at_decision=None` as "omit the Hold-specific
        # note" (see its own docstring) — strictly better than the old
        # fully-backwards-for-Sell prompt, so passing None here rather than
        # inventing a fragile proxy is the correct choice, not a shortcut.
        reflection_text = await reflect(
            render_portfolio_decision(prev.decision), raw_return, alpha_return, benchmark_ticker,
            rating=prev.decision.rating.value, held_at_decision=None,
        )
        await resolve_outcome(
            prev.decision_id, ticker, prev.trade_date, raw_return, alpha_return, holding_days, reflection_text,
            benchmark_ticker, resolved_on=record.trade_date,
        )

    return record


async def run_backtest(
    ticker: str,
    start: date,
    end: date,
    cadence_days: int = 5,
    benchmark_ticker: str = "SPY",
) -> BacktestResult:
    ticker_prices = await get_price_series(ticker, start, end)
    benchmark_prices = await get_price_series(benchmark_ticker, start, end)

    trade_dates = select_trading_dates(ticker_prices, cadence_days)
    if len(trade_dates) < 2:
        raise ValueError(
            f"only {len(trade_dates)} trading date(s) in [{start}, {end}] at "
            f"cadence_days={cadence_days} — need at least 2 to compute a return"
        )
    log_progress("replay_start", ticker=ticker, dates=len(trade_dates), range=f"{start}..{end}", cadence=cadence_days)

    # Wipes any stale is_backtest=true row a PRIOR run (e.g. a different
    # candidate in the same app.backtest.atr_sweep) left for this ticker —
    # without this, this run would silently inherit that run's ending
    # position instead of starting flat. See clear_backtest_positions's own
    # docstring.
    await clear_backtest_positions([ticker])

    # Every ledger read/write below (_process_ticker_date's gap stop-loss
    # checks, get_position's mark-to-market) must see this replay's OWN
    # trades table rows only — never a real live position, and never leak
    # its own speculative fills back into live's view. See project memory:
    # backtest-live-db-shared-contamination.
    with backtest_scope():
        records: list[BacktestRecord] = []
        equity_curve: list[float] = []
        prev: BacktestRecord | None = None
        for trade_date in trade_dates:
            record = await _process_ticker_date(
                ticker, trade_date, prev, ticker_prices, benchmark_prices, benchmark_ticker
            )
            records.append(record)
            # Mark-to-market against the ledger's actual post-trade position (this
            # date's rebalance, plus any gap stop-loss already applied above) —
            # the real equity curve, not the signed-exposure proxy.
            position = await get_position(ticker)
            equity_curve.append(position.market_value(ticker_prices[trade_date]))
            prev = record

    period_returns = compute_period_returns(records)
    exposure_risk_metrics = RiskMetrics(
        sharpe=sharpe_ratio(period_returns),
        sortino=sortino_ratio(period_returns),
        max_drawdown=max_drawdown(period_returns),
        cvar_95=cvar(period_returns),
    )

    ledger_period_returns = compute_ledger_period_returns(equity_curve)
    ledger_risk_metrics = RiskMetrics(
        sharpe=sharpe_ratio(ledger_period_returns),
        sortino=sortino_ratio(ledger_period_returns),
        max_drawdown=max_drawdown(ledger_period_returns),
        cvar_95=cvar(ledger_period_returns),
    )

    return BacktestResult(
        ticker=ticker,
        benchmark_ticker=benchmark_ticker,
        records=records,
        strategy_return=compute_strategy_return(records),
        buy_and_hold_return=compute_buy_and_hold_return(ticker_prices, trade_dates[0], trade_dates[-1]),
        benchmark_return=compute_buy_and_hold_return(benchmark_prices, trade_dates[0], trade_dates[-1]),
        exposure_risk_metrics=exposure_risk_metrics,
        ledger_risk_metrics=ledger_risk_metrics,
    )
