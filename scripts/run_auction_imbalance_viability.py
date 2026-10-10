"""Dev-only utility: measure the closing-auction imbalance signal
(app.backtest.auction_imbalance, pre-registered 2026-08-24) against real
KIS minute-bar data.

Structurally modeled on scripts/run_gap_reversion_viability.py (another
cross-sectional signal that needs a market-proxy series), with one
difference: this signal needs each day's FULL minute candles for both the
ticker and the market proxy (not just daily open/close), so market
candles are loaded once per date and reused across every ticker on that
date rather than reloaded per (ticker, date) pair.

Cache-only — reads app.backtest.minute_candles.get_cached_minute_day for
both the tickers and the market proxy, never calls KIS. A calendar day
missing from either the ticker's or the market proxy's cache is skipped
for that ticker.

Emits TWO reports, per this signal's own design memo:
1. The primary output — a decile-sort report (app.backtest.
   auction_imbalance_report.build_decile_report) from an UNCONDITIONAL
   entry on every eligible ticker-day, sorted by measured imbalance
   afterward. This is where the actual PASS/KILL/DIRECTIONAL_INSUFFICIENT
   verdict comes from.
2. A secondary baseline — app.backtest.scalping_viability.build_report on
   the same unconditional round trips, answering the (expected-negative)
   question "is buying every large cap at 15:16 and selling into the
   auction profitable with no imbalance filter at all?" This is a free
   sanity check, not the signal's verdict — see auction_imbalance_report's
   own docstring on why VIABLE_MIN_TRADES-style thresholds are the wrong
   tool for a once-per-ticker-day signal.

Usage:
    python -m scripts.run_auction_imbalance_viability \\
        --tickers 005930.KS,000660.KS --market 069500.KS \\
        --start 20260601 --end 20260821 --out report.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import date, datetime, timedelta

from app.backtest.auction_imbalance import compute_auction_imbalance, make_auction_imbalance_signal
from app.backtest.auction_imbalance_report import AuctionTrade, build_decile_report
from app.backtest.minute_candles import HistoricalMinuteCandle, get_cached_minute_day
from app.backtest.scalping_harness import run_scalping_backtest_day
from app.backtest.scalping_viability import RoundTrip, build_report, extract_round_trips

# Truncated-session guard (see this module's design memo point 2.3): a
# cached day whose last bar is earlier than this has no real auction
# print to trade into — the harness's own end-of-day force-close would
# silently exit at a continuous-session price instead.
_MIN_LAST_BAR_TIME = "152000"


def _weekdays(start: date, end: date):
    d = start
    while d <= end:
        if d.weekday() < 5:  # KRX holidays aren't modeled — see backfill_minute_candles.py
            yield d
        d += timedelta(days=1)


def _has_real_auction_print(candles: list[HistoricalMinuteCandle]) -> bool:
    return bool(candles) and max(c.time for c in candles) >= _MIN_LAST_BAR_TIME


async def _run_one_date(
    trade_date: date, tickers: list[str], market: str, require_threshold: bool,
) -> list[AuctionTrade]:
    market_candles = await get_cached_minute_day(market, trade_date)
    if not market_candles or not _has_real_auction_print(market_candles):
        return []

    trades: list[AuctionTrade] = []
    for ticker in tickers:
        ticker_candles = await get_cached_minute_day(ticker, trade_date)
        if not ticker_candles or not _has_real_auction_print(ticker_candles):
            continue

        imb = compute_auction_imbalance(ticker_candles, market_candles)
        if not imb.eligible:
            continue

        signal_fn = make_auction_imbalance_signal(imb, require_threshold=require_threshold)
        day_result = run_scalping_backtest_day(ticker_candles, signal_fn, ticker=ticker, trade_date=trade_date)
        for rt in extract_round_trips(day_result):
            trades.append(AuctionTrade(round_trip=rt, imb_resid=imb.imb_resid, drift_resid=imb.drift_resid))

    return trades


def _print_decile_report(report) -> None:
    print(f"\n{'decile':>7}{'n_trades':>10}{'n_dates':>9}{'imb_resid range':>22}{'mean_bp':>10}{'ci_lo':>9}{'ci_hi':>9}{'win_rate':>10}")
    for d in report.deciles:
        rng = f"[{d.imb_resid_min:.3f},{d.imb_resid_max:.3f}]" if d.n_trades else "-"
        print(
            f"{d.decile:>7}{d.n_trades:>10}{d.n_dates:>9}{rng:>22}{d.mean_net_bp:>10.2f}"
            f"{d.ci_date_clustered[0]:>9.2f}{d.ci_date_clustered[1]:>9.2f}{d.win_rate:>10.1%}"
        )
    print(f"\nspearman (decile index vs mean_net_bp): {report.spearman:.3f}")
    print(f"within-date permutation p-value (top-bottom spread): {report.permutation_p:.4f}")
    print(f"\n-- exclusions --")
    print(f"excluded_gap_count={report.excluded_gap_count}  excluded_gap_mean_net_bp={report.excluded_gap_mean_net_bp}")
    print(f"gap_fraction={report.gap_fraction:.1%}")
    print(f"\nVERDICT: {report.verdict} ({report.verdict_reason})")


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", required=True, help="comma-separated, e.g. 005930.KS,000660.KS")
    parser.add_argument("--market", default="069500.KS", help="market proxy ticker (default: KODEX 200 ETF)")
    parser.add_argument("--start", required=True, help="YYYYMMDD")
    parser.add_argument("--end", required=True, help="YYYYMMDD")
    parser.add_argument(
        "--require-threshold", action="store_true",
        help="secondary production-shaped run: only trade the AUCTION_IMB_TOP_THRESHOLD-gated proxy "
             "instead of the primary unconditional decile-sort run",
    )
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--permutations", type=int, default=10_000)
    parser.add_argument("--out", default=None, help="optional path to write a JSON report")
    args = parser.parse_args()

    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    start = datetime.strptime(args.start, "%Y%m%d").date()
    end = datetime.strptime(args.end, "%Y%m%d").date()

    print(
        f"running auction_imbalance ({'threshold-gated' if args.require_threshold else 'unconditional decile-sort'}) "
        f"over {tickers} vs market={args.market} {start}..{end} ..."
    )

    all_trades: list[AuctionTrade] = []
    for trade_date in _weekdays(start, end):
        all_trades.extend(await _run_one_date(trade_date, tickers, args.market, args.require_threshold))

    n_dates = len({t.round_trip.trade_date for t in all_trades})
    print(f"\ntotal round trips: {len(all_trades)} across {n_dates} distinct dates")

    if not all_trades:
        print("\nno eligible ticker-day produced a round trip in this sample — nothing to report")
        return

    decile_report = build_decile_report(all_trades, n_bootstrap=args.bootstrap, n_permutations=args.permutations)
    _print_decile_report(decile_report)

    all_round_trips: list[RoundTrip] = [t.round_trip for t in all_trades]
    baseline = build_report(
        all_round_trips, excluded_short_day_count=0,
        n_tickers_tested=len(tickers), n_bootstrap=args.bootstrap,
    )
    print("\n-- secondary baseline (scalping_viability.build_report, unconditional, no imbalance filter) --")
    p = baseline.pooled
    print(f"n_trades={p.n_trades}  mean_net_bp={p.mean_net_bp:.2f}  win_rate={p.win_rate:.1%}")
    print(f"clustered CI (ticker,date blocks — degenerate here, see this signal's own report docstring): {baseline.ci_clustered}")
    print(f"BASELINE VERDICT: {baseline.verdict.value} ({baseline.verdict_reason})")

    if args.out:
        out = {
            "signal": "auction_imbalance",
            "require_threshold": args.require_threshold,
            "decile_report": {
                "verdict": decile_report.verdict,
                "verdict_reason": decile_report.verdict_reason,
                "spearman": decile_report.spearman,
                "permutation_p": decile_report.permutation_p,
                "gap_fraction": decile_report.gap_fraction,
                "deciles": [vars(d) for d in decile_report.deciles],
            },
            "baseline_report": {
                "verdict": baseline.verdict.value,
                "verdict_reason": baseline.verdict_reason,
                "pooled": vars(baseline.pooled),
            },
            "n_trades": len(all_trades),
            "n_distinct_dates": n_dates,
        }
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
