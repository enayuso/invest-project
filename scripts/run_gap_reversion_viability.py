"""Dev-only utility: measure the overnight-gap idiosyncratic-reversion
signal (app.backtest.gap_reversion, pre-registered 2026-08-24) against real
KIS minute-bar data.

Separate script from scripts/run_scalping_viability.py because this
signal's eligibility is CROSS-SECTIONAL (needs a market-proxy series and
each ticker's own trailing daily closes) — unlike the self-contained
per-(ticker, day) signals in app.backtest.scalping_signals, this one can't
be classified from inside a single day's own candles.

Cache-only, like run_scalping_viability.py — reads
app.backtest.minute_candles.get_cached_minute_day for both the tickers and
the market proxy, never calls KIS. A calendar day missing from either the
ticker's or the market proxy's cache is skipped for that ticker (not
backfilled) — run scripts.backfill_minute_candles for both first, e.g.:
    python -m scripts.backfill_minute_candles 069500.KS 20260601 20260821

Usage:
    python -m scripts.run_gap_reversion_viability \\
        --tickers 005930.KS,000660.KS --market 069500.KS \\
        --start 20260601 --end 20260821 --out report.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import date, datetime, timedelta

from app.backtest.gap_reversion import GAP_TRAILING_DAYS, classify_gap_day, make_gap_reversion_signal
from app.backtest.minute_candles import get_cached_minute_day
from app.backtest.scalping_harness import run_scalping_backtest_day
from app.backtest.scalping_viability import RoundTrip, build_report, extract_round_trips


def _weekdays(start: date, end: date):
    d = start
    while d <= end:
        if d.weekday() < 5:  # KRX holidays aren't modeled — see backfill_minute_candles.py
            yield d
        d += timedelta(days=1)


async def _daily_open_close(ticker: str, days: list[date]) -> dict[date, tuple[float, float]]:
    """(open, close) of each cached trading day for `ticker` — a day absent
    from the cache (holiday or not yet backfilled) is simply absent from
    the returned dict, not an error."""
    out: dict[date, tuple[float, float]] = {}
    for d in days:
        candles = await get_cached_minute_day(ticker, d)
        if candles:
            sorted_candles = sorted(candles, key=lambda c: c.time)
            out[d] = (sorted_candles[0].open, sorted_candles[-1].close)
    return out


async def _run_one_ticker(ticker: str, market: str, days: list[date]) -> tuple[list[RoundTrip], int, int]:
    """Returns (round_trips, n_days_qualified, n_days_evaluated). A day is
    "evaluated" once it has both a prior close (for the gap) and enough
    trailing history to classify at all; the first several trading days in
    any range can't be evaluated (no prior day) or won't qualify (trailing
    vol still building toward GAP_TRAILING_DAYS) — that's expected, not a
    bug in this runner."""
    ticker_bars = await _daily_open_close(ticker, days)
    market_bars = await _daily_open_close(market, days)
    trading_days = sorted(d for d in days if d in ticker_bars and d in market_bars)

    round_trips: list[RoundTrip] = []
    n_qualified = 0
    n_evaluated = 0

    for i, d in enumerate(trading_days):
        if i == 0:
            continue  # no prior close yet — nothing to gap from
        prior_days = trading_days[max(0, i - GAP_TRAILING_DAYS):i]
        trailing_closes = [ticker_bars[p][1] for p in prior_days]
        prev_close = ticker_bars[trading_days[i - 1]][1]
        today_open, _ = ticker_bars[d]
        market_prev_close = market_bars[trading_days[i - 1]][1]
        market_today_open, _ = market_bars[d]

        classification = classify_gap_day(
            trailing_closes, prev_close, today_open, market_prev_close, market_today_open,
        )
        n_evaluated += 1
        if classification.qualifies:
            n_qualified += 1

        candles = await get_cached_minute_day(ticker, d)
        signal_fn = make_gap_reversion_signal(classification.qualifies)
        day_result = run_scalping_backtest_day(candles, signal_fn, ticker=ticker, trade_date=d)
        round_trips.extend(extract_round_trips(day_result))

    return round_trips, n_qualified, n_evaluated


def _print_report(report, n_qualified: int, n_evaluated: int) -> None:
    print(f"\nqualified ticker-days: {n_qualified}/{n_evaluated}")
    print(f"\n{'ticker':<12}{'n_trades':>10}{'n_days':>8}{'mean_bp':>10}{'win_rate':>10}")
    for ticker, stats in sorted(report.per_ticker.items()):
        print(
            f"{ticker:<12}{stats.n_trades:>10}{stats.n_ticker_days:>8}"
            f"{stats.mean_net_bp:>10.2f}{stats.win_rate:>10.1%}"
        )

    p = report.pooled
    print("\n-- pooled --")
    print(f"n_trades={p.n_trades}  n_ticker_days={p.n_ticker_days}")
    print(f"mean_net_bp={p.mean_net_bp:.2f}  median_net_bp={p.median_net_bp:.2f}  stdev={p.stdev_net_bp:.2f}")
    print(f"win_rate={p.win_rate:.1%}  p5={p.p5_net_bp:.2f}  p95={p.p95_net_bp:.2f}")
    print(f"mean_holding_minutes={p.mean_holding_minutes:.1f}  worst_trade_bp={p.worst_trade_net_bp:.2f}")
    print(f"max_drawdown_bp={p.max_drawdown_bp:.2f}")

    print("\n-- confidence intervals (95%) --")
    print(f"clustered bootstrap (block = ticker-day, correct for correlation): {report.ci_clustered}")
    print(f"naive per-trade (ignores clustering, expected to be narrower):     {report.ci_naive}")

    print("\n-- exclusions --")
    e = report.exclusions
    print(f"excluded_gap_count={e.excluded_gap_count}  excluded_gap_mean_net_bp={e.excluded_gap_mean_net_bp}")
    print(f"excluded_short_day_count={e.excluded_short_day_count}  included_count={e.included_count}")

    print(f"\nVERDICT: {report.verdict.value} ({report.verdict_reason})")


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", required=True, help="comma-separated, e.g. 005930.KS,000660.KS")
    parser.add_argument("--market", default="069500.KS", help="market proxy ticker (default: KODEX 200 ETF)")
    parser.add_argument("--start", required=True, help="YYYYMMDD")
    parser.add_argument("--end", required=True, help="YYYYMMDD")
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--out", default=None, help="optional path to write a JSON report")
    args = parser.parse_args()

    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    start = datetime.strptime(args.start, "%Y%m%d").date()
    end = datetime.strptime(args.end, "%Y%m%d").date()
    days = list(_weekdays(start, end))

    print(f"running gap_reversion over {tickers} vs market={args.market} {start}..{end} ...")

    all_round_trips: list[RoundTrip] = []
    total_qualified = 0
    total_evaluated = 0
    for ticker in tickers:
        rts, n_qual, n_eval = await _run_one_ticker(ticker, args.market, days)
        print(f"  {ticker}: {n_qual}/{n_eval} ticker-days qualified, {len(rts)} round trips")
        all_round_trips.extend(rts)
        total_qualified += n_qual
        total_evaluated += n_eval

    if not all_round_trips:
        print("\nno qualifying gap-down days produced a round trip in this sample — nothing to report")
        return

    report = build_report(
        all_round_trips, excluded_short_day_count=0,
        n_tickers_tested=len(tickers), n_bootstrap=args.bootstrap,
    )
    _print_report(report, total_qualified, total_evaluated)

    if args.out:
        out = {
            "signal": "gap_reversion",
            "verdict": report.verdict.value,
            "verdict_reason": report.verdict_reason,
            "pooled": vars(report.pooled),
            "per_ticker": {ticker: vars(stats) for ticker, stats in report.per_ticker.items()},
            "ci_clustered": report.ci_clustered,
            "ci_naive": report.ci_naive,
            "exclusions": vars(report.exclusions),
            "n_tickers_tested": report.n_tickers_tested,
            "n_tickers_positive_mean": report.n_tickers_positive_mean,
            "n_qualified_ticker_days": total_qualified,
            "n_evaluated_ticker_days": total_evaluated,
        }
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
