"""LLM-free trailing-stop ATR-multiplier comparison — replays
`app.execution.trailing_stop`'s actual ratchet math (`effective_stop_loss`,
the same `max(pm_stop_loss, high_water*(1-trailing_pct))` formula the live
exit_monitor and backtest harness use) directly against real historical
daily highs/lows, with NO AI pipeline, NO KIS live/paper order calls, and NO
DB writes — only read-only historical price fetches (KIS candles or
yfinance, whichever `get_price_series`/`get_daily_high_series`/
`get_daily_low_series`/`get_atr_pct` already resolve to).

Why this exists (2026-08-11): `app.backtest.atr_sweep` (AI-in-the-loop,
one real pipeline decision per ticker/date) turned out unusable for this
specific question — every decision in a short window came back Hold, so
zero trailing-stop exits ever fired to compare, and even with trades, an
AI decision's own run-to-run variance would have been indistinguishable
from the ATR multiplier's actual effect. Isolating the exit math from the
entry decision removes both problems: entries are a fixed, deterministic
rule (see `_entry_indices`), so every candidate replays the EXACT same set
of trades and only the trailing distance differs — matching the
methodology of Dai, Marshall, Nguyen & Visaltanachoti's own trailing-stop
study (this project's cited source): systematic entries across years of
real price data, not a handful of hand-picked cases.

Deliberately NOT integrated with the live `trailing_stop_state` table or
`ledger.check_and_apply_trailing_stop` (the AI-pipeline backtest's own
replay function) — this is a distinct, narrower tool for exactly one
question ("which ATR multiplier wins on the exit math alone"), not a
general-purpose backtest.

    python -m app.backtest.trailing_stop_param_sweep \
        --tickers 005930.KS,000660.KS,035420.KS,005490.KS,051910.KS,006400.KS,AAPL,MSFT,NVDA,GOOGL,AMZN,META \
        --start 2023-08-01 --end 2026-08-08 --entry-interval-days 10 \
        --atr-multipliers 1.0,1.25,1.5,1.75,2.0,2.25,2.5
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import sys
from dataclasses import dataclass
from datetime import date

from app.config import settings
from app.dataflows.kis_provider import (
    get_atr_pct,
    get_daily_high_series,
    get_daily_low_series,
    get_daily_open_series,
    get_long_range_price_series as get_price_series,
)
from app.execution.trailing_stop import effective_stop_loss

if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

logger_prefix = "[trailing_stop_param_sweep]"

# A real position never has a "no stop at all" floor — 0.0 here only means
# "PM contributes nothing to the max()", so effective_stop_loss reduces to
# pure trailing behavior (activation-gated, then high_water*(1-trailing_pct)).
# This intentionally isolates the trailing mechanism itself; a real PM
# stop-loss would only ever push the effective floor UP, never down, so
# leaving it out cannot make any candidate look artificially safer than a
# live run would — only removes a source that's identical across every
# candidate anyway (see module docstring).
_NO_PM_FLOOR = 0.0

# Skip entries too close to the end of the range — no room to observe
# whether/when a stop would have fired, so they'd just be same-day
# period_end trades that add noise without information.
_MIN_HOLDING_ROOM_DAYS = 5


@dataclass
class TickerSeries:
    dates: list[date]
    closes: dict[date, float]
    highs: dict[date, float]
    lows: dict[date, float]
    opens: dict[date, float]


@dataclass
class TradeResult:
    ticker: str
    entry_date: date
    entry_price: float
    exit_date: date
    exit_price: float
    exit_reason: str  # "trailing_stop" | "period_end"

    @property
    def return_pct(self) -> float:
        return self.exit_price / self.entry_price - 1.0


async def _load_ticker_series(ticker: str, start: date, end: date) -> TickerSeries:
    closes, highs, lows, opens = await asyncio.gather(
        get_price_series(ticker, start, end),
        get_daily_high_series(ticker, start, end),
        get_daily_low_series(ticker, start, end),
        get_daily_open_series(ticker, start, end),
    )
    dates = sorted(set(closes) & set(highs) & set(lows) & set(opens))
    return TickerSeries(dates=dates, closes=closes, highs=highs, lows=lows, opens=opens)


def _entry_indices(series: TickerSeries, interval_days: int) -> list[int]:
    """Every `interval_days`-th trading day as an independent hypothetical
    entry — deterministic and identical across every ATR candidate, so a
    return difference between candidates can only come from the trailing
    math, never from which trades got picked (see module docstring)."""
    last_usable = len(series.dates) - 1 - _MIN_HOLDING_ROOM_DAYS
    return list(range(0, max(last_usable, 0), interval_days))


async def _atr_pct_at_entry(ticker: str, entry_date: date) -> float | None:
    try:
        return await get_atr_pct(ticker, as_of=entry_date)
    except Exception:
        return None


def _simulate_one_trade(
    ticker: str, series: TickerSeries, entry_idx: int, atr_pct: float | None, atr_multiplier: float
) -> TradeResult:
    entry_date = series.dates[entry_idx]
    entry_price = series.closes[entry_date]
    trailing_pct = (
        atr_multiplier * atr_pct if atr_pct is not None and atr_pct > 0 else settings.trailing_stop_fallback_pct
    )
    # Test the floor as it stood BEFORE today's high ratchets it, then
    # ratchet after — never fold day d's own high into the floor that day
    # d's own low is tested against. From daily bars alone, which of a day's
    # high/low printed first is unknowable; assuming high-before-low (the
    # pre-fix behavior) is exactly the same-day false-breach production's
    # `app.execution.trailing_stop` has a persisted watermark to prevent
    # (the 047040.KS incident) — it fabricates an early exit at a floor that
    # never existed yet when the low printed. This ordering is conservative
    # in the other direction (a real intraday breach that only became
    # possible after today's high ratchet fires one day late, on tomorrow's
    # test), which is the safer bias for a return-measurement tool: it can
    # only make a tight multiplier look BETTER than reality, never fabricate
    # a loss that never happened.
    high_water = entry_price
    for d in series.dates[entry_idx + 1 :]:
        floor_in_force = effective_stop_loss(_NO_PM_FLOOR, entry_price, high_water, trailing_pct)
        if floor_in_force > _NO_PM_FLOOR and series.lows[d] <= floor_in_force:
            # A stop is a market-on-touch order — if the day's open already
            # gapped through the floor, the realizable fill is that open,
            # not the floor level itself.
            fill_price = min(floor_in_force, series.opens[d]) if d in series.opens else floor_in_force
            return TradeResult(ticker, entry_date, entry_price, d, fill_price, "trailing_stop")
        high_water = max(high_water, series.highs[d])
    last_date = series.dates[-1]
    return TradeResult(ticker, entry_date, entry_price, last_date, series.closes[last_date], "period_end")


def _trades_for_candidate(
    tickers: list[str],
    series_by_ticker: dict[str, TickerSeries],
    entry_atr_by_ticker: dict[str, dict[date, float | None]],
    atr_multiplier: float,
) -> list[TradeResult]:
    trades = []
    for ticker in tickers:
        series = series_by_ticker[ticker]
        for entry_idx in _entry_indices(series, _CURRENT_INTERVAL_DAYS):
            entry_date = series.dates[entry_idx]
            atr_pct = entry_atr_by_ticker[ticker][entry_date]
            trades.append(_simulate_one_trade(ticker, series, entry_idx, atr_pct, atr_multiplier))
    return trades


# Module-level, set once in _main before _trades_for_candidate is called —
# avoids threading one more argument through every candidate's call just to
# carry a value that's constant across the whole sweep run.
_CURRENT_INTERVAL_DAYS = 10


def _summarize(trades: list[TradeResult]) -> dict[str, float | int]:
    returns = [t.return_pct for t in trades]
    n = len(returns)
    mean = statistics.fmean(returns) if n else 0.0
    median = statistics.median(returns) if n else 0.0
    stdev = statistics.pstdev(returns) if n > 1 else 0.0
    sharpe = mean / stdev if stdev > 0 else float("nan")
    win_rate = sum(1 for r in returns if r > 0) / n if n else 0.0
    tail_n = max(1, round(n * 0.05))
    cvar95 = statistics.fmean(sorted(returns)[:tail_n]) if n else 0.0
    stopped_out = sum(1 for t in trades if t.exit_reason == "trailing_stop")
    return {
        "n_trades": n,
        "mean_return_pct": mean,
        "median_return_pct": median,
        "sharpe": sharpe,
        "win_rate": win_rate,
        "cvar95_pct": cvar95,
        "stopped_out_pct": stopped_out / n if n else 0.0,
    }


def _render(atr_multiplier: float, stats: dict[str, float | int]) -> str:
    return (
        f"{atr_multiplier:>6.2f}  {stats['n_trades']:>7d}  "
        f"{stats['mean_return_pct']:+8.2%}  {stats['median_return_pct']:+8.2%}  "
        f"{stats['sharpe']:+7.2f}  {stats['win_rate']:7.1%}  {stats['cvar95_pct']:+8.2%}  "
        f"{stats['stopped_out_pct']:7.1%}"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tickers", required=True, help="Comma-separated ticker universe.")
    parser.add_argument("--start", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD")
    parser.add_argument("--entry-interval-days", type=int, default=10, help="Trading days between hypothetical entries, per ticker.")
    parser.add_argument("--atr-multipliers", required=True, help="Comma-separated candidates, e.g. '1.0,1.5,2.0'.")
    return parser.parse_args()


async def _main() -> None:
    global _CURRENT_INTERVAL_DAYS
    args = _parse_args()
    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    start, end = date.fromisoformat(args.start), date.fromisoformat(args.end)
    multipliers = [float(m.strip()) for m in args.atr_multipliers.split(",") if m.strip()]
    _CURRENT_INTERVAL_DAYS = args.entry_interval_days

    print(f"{logger_prefix} loading price history for {len(tickers)} ticker(s)...")
    series_list = await asyncio.gather(*(_load_ticker_series(t, start, end) for t in tickers))
    series_by_ticker = dict(zip(tickers, series_list))

    print(f"{logger_prefix} fetching entry-date ATR%% (shared across every candidate)...")
    entry_atr_by_ticker: dict[str, dict[date, float | None]] = {}
    for ticker in tickers:
        series = series_by_ticker[ticker]
        entry_dates = [series.dates[i] for i in _entry_indices(series, args.entry_interval_days)]
        atrs = await asyncio.gather(*(_atr_pct_at_entry(ticker, d) for d in entry_dates))
        entry_atr_by_ticker[ticker] = dict(zip(entry_dates, atrs))
    total_entries = sum(len(v) for v in entry_atr_by_ticker.values())
    print(f"{logger_prefix} {total_entries} entries per candidate ({len(multipliers)} candidates)")

    print(f"\n{'ATR mult.':>8}  {'n':>7}  {'mean ret':>8}  {'median':>8}  {'sharpe':>7}  {'win%':>7}  {'CVaR95':>8}  {'stopped%':>8}")
    print("-" * 76)
    for multiplier in multipliers:
        trades = _trades_for_candidate(tickers, series_by_ticker, entry_atr_by_ticker, multiplier)
        stats = _summarize(trades)
        print(_render(multiplier, stats))

    print(
        f"\n{logger_prefix} NOTE: trades are independent per-entry simulations (not a compounding "
        "portfolio) — this isolates the trailing-stop exit math from position sizing/portfolio caps, "
        "which are unrelated to the parameter being compared here."
    )


if __name__ == "__main__":
    asyncio.run(_main())
