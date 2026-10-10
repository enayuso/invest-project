"""Dev-only utility: scans cached (or newly fetched) minute candles for gaps
beyond the expected closing-auction window, per docs/data-reliability-check-plan.md
(Phase 1 of the "데이터 신뢰성 점검" loop).

Reuses app.backtest.minute_candles's cache-first fetch — no new data-access
logic. A "gap" here is any span of >1 consecutive minute inside 09:00-15:19
with no cached row; the 15:19-15:30 closing single-price-auction window is
excluded (verified normal on 2026-07-24, see docs/scalping-backtest-infra-plan.md
Phase 1). Records every gap regardless of length (short gaps aren't assumed
to be anomalies, just recorded — the interpretation is left to whoever reads
the report, not filtered out here) and classifies each as "short" (<=5min)
or "long" (>5min, the 000660.KS 2026-07-13 case that started this loop was
30min).

Usage:
    python -m scripts.scan_minute_data_gaps 005930.KS,000660.KS 20260701 20260727
Must run in the same environment (.env / DB) as the actual dashboard — uses
this deployment's own KIS credentials (app.config.Settings.kis_credentials,
2026-08-07 single-account rollback), so there is no longer a `--user-email`
to pick whose account to run as.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
from datetime import date, datetime, timedelta

from app.backtest.minute_candles import fetch_and_cache_minute_day
from app.brokers import get_broker
from app.brokers.kis.broker import krx_code_from_ticker

_MARKET_OPEN = 9 * 60  # 09:00 in minutes-since-midnight
_AUCTION_START = 15 * 60 + 19  # 15:19 — closing single-price auction begins
_SHORT_GAP_MAX_MINUTES = 5


@dataclass
class Gap:
    ticker: str
    trade_date: date
    start_minute: int
    end_minute: int

    @property
    def length_minutes(self) -> int:
        return self.end_minute - self.start_minute

    @property
    def category(self) -> str:
        return "short" if self.length_minutes <= _SHORT_GAP_MAX_MINUTES else "long"

    def __str__(self) -> str:
        def fmt(m: int) -> str:
            return f"{m // 60:02d}:{m % 60:02d}"
        return f"{self.ticker} {self.trade_date} {fmt(self.start_minute)}~{fmt(self.end_minute)} ({self.length_minutes}min, {self.category})"


def _to_minute(hhmmss: str) -> int:
    return int(hhmmss[:2]) * 60 + int(hhmmss[2:4])


def find_gaps(ticker: str, trade_date: date, times: list[str]) -> list[Gap]:
    minutes = sorted(set(_to_minute(t) for t in times))
    gaps = []
    for i in range(len(minutes) - 1):
        cur, nxt = minutes[i], minutes[i + 1]
        if nxt - cur <= 1:
            continue
        # Skip a gap that's entirely (or mostly) the closing auction window —
        # a gap starting at/after 15:19 is the expected one, not an anomaly.
        if cur >= _AUCTION_START:
            continue
        gaps.append(Gap(ticker=ticker, trade_date=trade_date, start_minute=cur, end_minute=nxt))
    return gaps


def _daterange(start: date, end: date):
    d = start
    while d <= end:
        if d.weekday() < 5:
            yield d
        d += timedelta(days=1)


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("tickers", help="comma-separated, e.g. 005930.KS,000660.KS")
    parser.add_argument("start", help="YYYYMMDD")
    parser.add_argument("end", help="YYYYMMDD")
    args = parser.parse_args()

    broker = await get_broker()
    if broker is None:
        print("no KIS credentials configured on this deployment — set them in the env "
              "(app.config.Settings.kis_credentials) before running this script")
        return

    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    start = datetime.strptime(args.start, "%Y%m%d").date()
    end = datetime.strptime(args.end, "%Y%m%d").date()

    all_gaps: list[Gap] = []
    scanned_days = 0
    for ticker in tickers:
        code = krx_code_from_ticker(ticker)
        if code is None:
            print(f"{ticker}: skipped (not a recognized .KS ticker)")
            continue
        for trade_date in _daterange(start, end):
            candles = await fetch_and_cache_minute_day(broker, ticker, code, trade_date)
            if not candles:
                continue  # holiday or genuinely no data — not this scanner's concern
            scanned_days += 1
            gaps = find_gaps(ticker, trade_date, [c.time for c in candles])
            all_gaps.extend(gaps)
            status = f"{len(gaps)} gap(s)" if gaps else "clean"
            print(f"{ticker} {trade_date}: {len(candles)} candles, {status}")

    print(f"\n=== summary: {scanned_days} trading day(s) scanned, {len(all_gaps)} anomalous gap(s) found ===")
    if not all_gaps:
        return

    short = [g for g in all_gaps if g.category == "short"]
    long_ = [g for g in all_gaps if g.category == "long"]
    print(f"short (<= {_SHORT_GAP_MAX_MINUTES}min): {len(short)}, long: {len(long_)}")

    by_ticker: dict[str, int] = {}
    for g in all_gaps:
        by_ticker[g.ticker] = by_ticker.get(g.ticker, 0) + 1
    print("by ticker:", ", ".join(f"{t}={n}" for t, n in sorted(by_ticker.items(), key=lambda kv: -kv[1])))

    print("\nall gaps (longest first):")
    for g in sorted(all_gaps, key=lambda g: -g.length_minutes):
        print(f"  {g}")


if __name__ == "__main__":
    asyncio.run(main())
