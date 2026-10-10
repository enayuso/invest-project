"""Minute-level backtest loop for scalping Entry Engine experiments (Phase 3
of docs/scalping-backtest-infra-plan.md).

Bridges app.backtest.minute_candles (Phase 1 — historical data + Postgres
cache) and app.backtest.fill_simulation (Phase 2 — latency/slippage) into a
runnable day-by-day simulation loop. Deliberately separate from
app.backtest.harness's daily/LLM-driven loop, per docs/scalping-design.md
2절's "Deep Analysis is slow, Trading Decision is fast" split — this loop
never calls an LLM and never touches app.execution.ledger (paper/live
execution, DB-backed, owns actual position state). It only replays
already-cached minute candles.

It DOES import `app.execution.costs` (2026-08-21) for the commission/tax
formula used to net fees out of each round trip below — that module is
deliberately NOT `app.execution.ledger`: pure functions only, no DB I/O, no
paper/live position state, nothing that would compromise the isolation this
docstring is protecting. The isolation being guarded here is from stateful/
DB-backed code, not from a pure domain formula that both the live ledger
and this backtest must agree on — see that module's own docstring for why
it exists as a third module rather than either duplicating the formula here
or reaching into the ledger for it.

`entry_signal_fn` is intentionally pluggable and NOT specified by this
module — the actual Entry Engine logic (docs/scalping-design.md 7절 item 2)
is still an open design decision. `demo_momentum_signal` below exists only
to prove the loop mechanics end-to-end (data flow, fill simulation
integration, PnL aggregation); it is a placeholder, not a strategy
recommendation, and has not been evaluated for profitability.

Reads only from the Phase 1 cache — does not call KIS itself. Callers must
backfill via scripts.backfill_minute_candles first; a day with no cached
data is skipped rather than raising (a holiday or a not-yet-backfilled day
shouldn't kill the whole range).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Callable

from app.backtest.fill_simulation import SimulatedFill, decision_time_for_bar, simulate_fill
from app.backtest.minute_candles import HistoricalMinuteCandle, get_cached_minute_day
from app.execution.costs import transaction_cost_bps_for

# (candles so far, current index, has_open_position) -> "buy" / "sell" / None
EntrySignalFn = Callable[[list[HistoricalMinuteCandle], int, bool], "str | None"]


@dataclass
class ScalpingTrade:
    side: str  # "buy" or "sell"
    signal_time: str  # HHMMSS the signal fired at (before latency)
    fill: SimulatedFill


@dataclass
class DayResult:
    ticker: str
    trade_date: date
    trades: list[ScalpingTrade]
    return_pct: float | None  # None if no completed round trip that day


def _compute_return_pct(trades: list[ScalpingTrade], ticker: str) -> float | None:
    """Compounds each completed buy->sell round trip. A dangling buy with no
    matching sell (shouldn't happen — run_scalping_backtest_day force-closes
    at day end) is ignored rather than counted as an open, unrealized PnL —
    this is a closed-trade return series only.

    Fees (2026-08-21) are netted out of each leg exactly like
    `app.execution.ledger.record_trade` treats them — cash deducted from the
    trade, not baked into the stored price — via the shared
    `app.execution.costs.transaction_cost_bps_for` (broker commission both
    sides, plus Korea's sell-only Securities Transaction Tax on a domestic
    `.KS` sell). Before this, this loop charged `slippage_bps` only (5bp/side
    = ~10bp round trip via fill_simulation) against a real KRX round trip of
    ~30-40bp (20bp sell tax + commission + spread crossing) — every prior
    scalping result, including the Phase 3 demo's 146-184 trades/day at
    -11~13%, was that optimistic. `slippage_bps` and this commission
    component are NOT individually calibrated against real spread-crossing
    cost (commission at 5bp/side here is standing in for part of that too);
    what matters is the total landing in the realistic 30-40bp range, and
    erring conservative (never understating cost) is deliberate."""
    equity_mult = 1.0
    entry_price: float | None = None
    completed_any = False
    for t in trades:
        if t.side == "buy":
            entry_price = t.fill.executed_price
        elif t.side == "sell" and entry_price is not None:
            buy_fee_rate = transaction_cost_bps_for(ticker, "buy") / 10_000
            sell_fee_rate = transaction_cost_bps_for(ticker, "sell") / 10_000
            equity_mult *= (
                (t.fill.executed_price * (1 - sell_fee_rate))
                / (entry_price * (1 + buy_fee_rate))
            )
            entry_price = None
            completed_any = True
    if not completed_any:
        return None
    return (equity_mult - 1) * 100


def run_scalping_backtest_day(
    candles: list[HistoricalMinuteCandle],
    entry_signal_fn: EntrySignalFn,
    *,
    ticker: str,
    trade_date: date,
    latency_seconds: float = 30.0,
    slippage_bps: float = 5.0,
) -> DayResult:
    """Long-only (matches this project's project-wide long-only convention,
    app/execution/sizing.py) — entry_signal_fn is only ever asked for a
    "buy" while flat or a "sell" while holding; anything else from it is a
    no-op for that minute."""
    candles = sorted(candles, key=lambda c: c.time)
    trades: list[ScalpingTrade] = []
    position_open = False

    for idx in range(len(candles)):
        signal = entry_signal_fn(candles, idx, position_open)
        # decision_time_for_bar, not candles[idx].time (the bar's raw
        # START): every signal here decides off cur.close, which is only
        # knowable once the bar ENDS. Passing the bar start let
        # latency_seconds=0 fill at that same bar's own open — a price that
        # printed 60s before the triggering close existed (see
        # fill_simulation.decision_time_for_bar's own docstring).
        decision_time = decision_time_for_bar(candles[idx])
        if signal == "buy" and not position_open:
            fill = simulate_fill(candles, decision_time, latency_seconds, "buy", slippage_bps)
            if fill is None:
                continue  # no fillable data left this session (near close) — skip, don't force it
            trades.append(ScalpingTrade(side="buy", signal_time=decision_time, fill=fill))
            position_open = True
        elif signal == "sell" and position_open:
            fill = simulate_fill(candles, decision_time, latency_seconds, "sell", slippage_bps)
            if fill is None:
                continue
            trades.append(ScalpingTrade(side="sell", signal_time=decision_time, fill=fill))
            position_open = False

    if position_open and candles:
        # Force-close at the last candle's own close — there's no more data
        # left in the session to simulate a slow reaction against, so this
        # is an immediate fill (0 latency), not run back through
        # simulate_fill's forward-looking search (which would find nothing).
        last = candles[-1]
        last_decision_time = decision_time_for_bar(last)
        trades.append(ScalpingTrade(
            side="sell", signal_time=last_decision_time,
            fill=SimulatedFill(
                signal_time=last_decision_time, requested_latency_seconds=0, filled_time=last.time,
                fill_price=last.close, slippage_bps=slippage_bps, side="sell",
                executed_price=last.close * (1 - slippage_bps / 10_000),
                realized_latency_seconds=0, gap_inflated=False,
            ),
        ))

    return DayResult(ticker=ticker, trade_date=trade_date, trades=trades,
                      return_pct=_compute_return_pct(trades, ticker))


def _weekdays(start: date, end: date):
    d = start
    while d <= end:
        if d.weekday() < 5:  # KRX holidays aren't modeled — see backfill_minute_candles.py
            yield d
        d += timedelta(days=1)


async def run_scalping_backtest(
    ticker: str,
    start_date: date,
    end_date: date,
    entry_signal_fn: EntrySignalFn,
    *,
    latency_seconds: float = 30.0,
    slippage_bps: float = 5.0,
) -> list[DayResult]:
    """Cache-only multi-day runner — see this module's docstring on why it
    never calls KIS itself."""
    results = []
    for trade_date in _weekdays(start_date, end_date):
        candles = await get_cached_minute_day(ticker, trade_date)
        if not candles:
            continue
        results.append(run_scalping_backtest_day(
            candles, entry_signal_fn, ticker=ticker, trade_date=trade_date,
            latency_seconds=latency_seconds, slippage_bps=slippage_bps,
        ))
    return results


def demo_momentum_signal(candles: list[HistoricalMinuteCandle], idx: int, has_position: bool) -> str | None:
    """PLACEHOLDER ONLY. Buys on any uptick while flat, sells on any downtick
    while holding — this is the simplest possible rule that produces
    frequent trades, chosen purely to exercise every code path in the loop
    above (multi-trade days, force-close, gap handling). It is NOT a
    strategy recommendation and has not been evaluated for profitability;
    the actual Entry Engine logic remains an open decision
    (docs/scalping-design.md 7절 item 2)."""
    if idx == 0:
        return None
    prev, cur = candles[idx - 1], candles[idx]
    if not has_position and cur.close > prev.close:
        return "buy"
    if has_position and cur.close < prev.close:
        return "sell"
    return None
