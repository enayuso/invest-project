"""Dev-only utility: verify whether TR FHKST03010230 (주식일별분봉조회 — past-date
minute candles) is usable on the currently configured KIS account (paper by
default). Read-only quote call, no orders. Written for the scalping-design
Phase 0 check (docs/scalping-backtest-infra-plan.md) — this project's other
past-date TR (CTPF1002R, 주식기본조회) is confirmed paper-rejected, but this
one hasn't been tried live yet.

Usage: python -m scripts.check_daily_minute_chart_tr [CODE] [YYYYMMDD] [HHMMSS]
Defaults: CODE=005930 (삼성전자), a recent past weekday, HHMMSS=130000.
Must run in the same environment (.env / DB) as the actual dashboard — uses
this deployment's own KIS credentials (app.config.Settings.kis_credentials,
2026-08-07 single-account rollback), so there is no longer a `--user-email`
to pick whose account to run as.
"""

from __future__ import annotations

import argparse
import asyncio
import logging

from app.brokers import get_broker
from app.brokers.kis.const import KISBrokerError
from app.config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

_TR_ID = "FHKST03010230"  # 주식일별분봉조회 [국내주식-213]
_PATH = "/uapi/domestic-stock/v1/quotations/inquire-time-dailychartprice"


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("code", nargs="?", default="005930")
    parser.add_argument("date", nargs="?", default="20260724")
    parser.add_argument("hour", nargs="?", default="130000")
    args = parser.parse_args()

    broker = await get_broker()
    if broker is None:
        print("no KIS credentials configured on this deployment — set them in the env "
              "(app.config.Settings.kis_credentials) before running this script")
        return

    print(f"account_mode={settings.kis_account_mode} code={args.code} date={args.date} hour={args.hour}")
    try:
        resp = await broker._request(
            "GET", _PATH, _TR_ID,
            params={
                "FID_COND_MRKT_DIV_CODE": "J",
                "FID_INPUT_ISCD": args.code,
                "FID_INPUT_HOUR_1": args.hour,
                "FID_INPUT_DATE_1": args.date,
                "FID_PW_DATA_INCU_YN": "N",
                "FID_FAKE_TICK_INCU_YN": "",
            },
        )
        body = broker._checked_body(resp, f"daily-minute-chart for {args.code}")
    except KISBrokerError as exc:
        print(f"REJECTED: {exc}")
        return

    output2 = body.get("output2", [])
    print(f"SUCCESS: rt_cd=0, output2 rows={len(output2)}")
    for row in output2[:5]:
        print(row)
    if len(output2) > 5:
        print(f"... ({len(output2) - 5} more rows)")


if __name__ == "__main__":
    asyncio.run(main())
