"""Dev-only utility: backfill historical 1-minute candles into the
minute_candles Postgres cache (app.backtest.minute_candles, Phase 1 of
docs/scalping-backtest-infra-plan.md) for one or more KRX tickers over a
date range.

Skips weekends only — KRX holidays aren't modeled here, a holiday date just
comes back with zero rows from KIS and is skipped without being cached (see
minute_candles._store_minute_day's docstring on why empty days aren't
persisted as "fetched").

Usage:
    python -m scripts.backfill_minute_candles 005930.KS 20260701 20260727
    python -m scripts.backfill_minute_candles 005930.KS,000660.KS 20260701 20260727
Must run in the same environment (.env / DB) as the actual dashboard — uses
this deployment's own KIS credentials (app.config.Settings.kis_credentials,
2026-08-07 single-account rollback), so there is no longer a `--user-email`
to pick whose account to run as.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from datetime import date, datetime, timedelta

from app.backtest.minute_candles import fetch_and_cache_minute_day
from app.brokers import get_broker
from app.brokers.kis.broker import krx_code_from_ticker

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _daterange(start: date, end: date):
    d = start
    while d <= end:
        if d.weekday() < 5:  # Mon-Fri only; KRX holidays are not modeled here
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

    for ticker in tickers:
        code = krx_code_from_ticker(ticker)
        if code is None:
            print(f"{ticker}: skipped (not a recognized .KS ticker)")
            continue
        for trade_date in _daterange(start, end):
            try:
                candles = await fetch_and_cache_minute_day(broker, ticker, code, trade_date)
            except Exception as exc:
                print(f"{ticker} {trade_date}: FAILED - {exc}")
                continue
            print(f"{ticker} {trade_date}: {len(candles)} minute candles")


if __name__ == "__main__":
    asyncio.run(main())
