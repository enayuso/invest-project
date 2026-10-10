"""Dev-only utility: measure whether a candidate intraday scalping signal
(`app.backtest.scalping_signals`) clears the real domestic round-trip cost
hurdle (`app.execution.costs`, ~30-40bp all-in as of the 2026-01-01 tax
change) — see project memory: scalping-backtest-infra-gap's 2026-08-21
direction re-check for why this measurement exists before any capital-split
UI work.

Cache-only, like `scripts/backfill_minute_candles.py` — reads
`app.backtest.minute_candles.get_cached_minute_day` from Postgres and never
calls KIS, so it needs no broker/credentials and is safe to run at any time
(including during market hours) without touching the live REST/websocket
throttle. A ticker-day not already in the cache is skipped, not
backfilled — run `scripts.backfill_minute_candles` first if a range isn't
covered yet.

Each (signal, ticker, day) gets a FRESH signal instance — see
`app.backtest.scalping_signals`'s module docstring on why reusing one
instance across days would leak state (entry counts, opening-range levels,
running VWAP) across the day boundary. `run_scalping_backtest_day` is
called directly per day for the same reason; the multi-day
`run_scalping_backtest` convenience wrapper in `scalping_harness.py` reuses
one signal object across its whole date range and is deliberately not used
here.

Usage:
    python -m scripts.run_scalping_viability --signal orb \\
        --tickers 005930.KS,000660.KS --start 20260701 --end 20260727
    python -m scripts.run_scalping_viability --signal all \\
        --tickers 005930.KS --start 20260701 --end 20260727 --out report.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import date, datetime, timedelta

from app.backtest.minute_candles import get_cached_minute_day
from app.backtest.scalping_harness import run_scalping_backtest_day
from app.backtest.scalping_signals import (
    make_orb_holdclose_signal,
    make_orb_signal,
    make_volume_spike_signal,
    make_vwap_reversion_signal,
)
from app.backtest.scalping_viability import RoundTrip, ViabilityReport, build_report, extract_round_trips

_SIGNAL_FACTORIES = {
    "orb": make_orb_signal,
    "orb_holdclose": make_orb_holdclose_signal,
    "vol_spike": make_volume_spike_signal,
    "vwap_revert": make_vwap_reversion_signal,
}


def _weekdays(start: date, end: date):
    d = start
    while d <= end:
        if d.weekday() < 5:  # KRX holidays aren't modeled — see backfill_minute_candles.py
            yield d
        d += timedelta(days=1)


async def _run_one_signal(
    signal_name: str,
    tickers: list[str],
    start: date,
    end: date,
    min_candles: int,
    n_bootstrap: int,
) -> tuple[list[RoundTrip], int]:
    """Returns (all_round_trips, excluded_short_day_count) for one signal
    across every (ticker, weekday-in-range)."""
    factory = _SIGNAL_FACTORIES[signal_name]
    all_round_trips: list[RoundTrip] = []
    excluded_short_day_count = 0

    for ticker in tickers:
        for trade_date in _weekdays(start, end):
            candles = await get_cached_minute_day(ticker, trade_date)
            if candles is None:
                print(f"  [{signal_name}] {ticker} {trade_date}: not cached, skipped")
                continue
            if len(candles) < min_candles:
                print(
                    f"  [{signal_name}] {ticker} {trade_date}: only {len(candles)} candles "
                    f"(< {min_candles} min-candles floor), excluded as a short day"
                )
                excluded_short_day_count += 1
                continue

            signal_fn = factory()  # fresh instance for this (ticker, day) — see module docstring
            day_result = run_scalping_backtest_day(candles, signal_fn, ticker=ticker, trade_date=trade_date)
            all_round_trips.extend(extract_round_trips(day_result))

    return all_round_trips, excluded_short_day_count


def _print_report(signal_name: str, report: ViabilityReport) -> None:
    print(f"\n=== {signal_name} ===")
    print(f"{'ticker':<12}{'n_trades':>10}{'n_days':>8}{'mean_bp':>10}{'win_rate':>10}")
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


def _report_to_json(signal_name: str, report: ViabilityReport, round_trips: list[RoundTrip]) -> dict:
    return {
        "signal": signal_name,
        "verdict": report.verdict.value,
        "verdict_reason": report.verdict_reason,
        "pooled": vars(report.pooled),
        "per_ticker": {ticker: vars(stats) for ticker, stats in report.per_ticker.items()},
        "ci_clustered": report.ci_clustered,
        "ci_naive": report.ci_naive,
        "exclusions": vars(report.exclusions),
        "n_tickers_tested": report.n_tickers_tested,
        "n_tickers_positive_mean": report.n_tickers_positive_mean,
        "raw_round_trips": [
            {
                "ticker": rt.ticker,
                "trade_date": rt.trade_date.isoformat(),
                "entry_price": rt.entry_price,
                "exit_price": rt.exit_price,
                "net_bp": rt.net_bp,
                "gap_inflated": rt.gap_inflated,
                "holding_minutes": rt.holding_minutes,
            }
            for rt in round_trips
        ],
    }


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--signal", required=True, choices=["orb", "orb_holdclose", "vol_spike", "vwap_revert", "all"])
    parser.add_argument("--tickers", required=True, help="comma-separated, e.g. 005930.KS,000660.KS")
    parser.add_argument("--start", required=True, help="YYYYMMDD")
    parser.add_argument("--end", required=True, help="YYYYMMDD")
    parser.add_argument("--min-candles", type=int, default=300)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--out", default=None, help="optional path to write a JSON report")
    args = parser.parse_args()

    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    start = datetime.strptime(args.start, "%Y%m%d").date()
    end = datetime.strptime(args.end, "%Y%m%d").date()
    signal_names = list(_SIGNAL_FACTORIES) if args.signal == "all" else [args.signal]

    # Each signal is a separate hypothesis — never pooled together, per this
    # measurement's own design (three independent signal ideas, not one
    # combined strategy).
    json_reports = []
    for signal_name in signal_names:
        print(f"\nrunning {signal_name} over {tickers} {start}..{end} ...")
        round_trips, excluded_short_day_count = await _run_one_signal(
            signal_name, tickers, start, end, args.min_candles, args.bootstrap,
        )
        report = build_report(
            round_trips,
            excluded_short_day_count=excluded_short_day_count,
            n_tickers_tested=len(tickers),
            n_bootstrap=args.bootstrap,
        )
        _print_report(signal_name, report)
        if args.out:
            json_reports.append(_report_to_json(signal_name, report, round_trips))

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(json_reports, f, indent=2, ensure_ascii=False)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
