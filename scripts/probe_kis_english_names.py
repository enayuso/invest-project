"""Dev-only utility: dump KIS's own 종목명 (Korean abbrev + English abbrev) for
a list of KRX codes, via the 주식기본조회 TR (`fetch_stock_status`, TR
CTPF1002R). Read-only quote call, no orders.

Written for identity-gate audit #2 (2026-09-01): the last residual false
positives are romanization / transliteration variance ("Geumho Electric" vs
"Kumho", "KBI메탈" vs "KBI Metal") that normalization can't bridge. KIS's own
`prdt_eng_abrv_name` could become an authoritative English alias — but it's an
ABBREVIATION (e.g. "SamsungElec", per tests/brokers/test_kis_broker.py) and
this TR is real-account only (paper -> EGW02006). This script is step 1: see
the actual field values before wiring anything into app.orchestrator.pipeline.
_company_identity.

Usage: python -m scripts.probe_kis_english_names [CODE ...]
Default codes: the 2026-09-01 cycle's tickers + a few reference large caps.

⚠️ RUN ONLY WITH THE LIVE DASHBOARD DAEMON STOPPED. KIS issues one token per
app key; running this alongside a live daemon can invalidate the daemon's
token mid-session (project memory: no double local daemons). Real-account
only — a paper account will raise EGW02006 for every code.
"""

from __future__ import annotations

import argparse
import asyncio
import logging

from app.brokers import get_broker
from app.brokers.kis.const import KISBrokerError
from app.config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

_DEFAULT_CODES = ["024840", "001210", "066970", "481050", "005930", "000660", "316140", "012330"]


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("codes", nargs="*", default=_DEFAULT_CODES, help="bare 6-digit KRX codes")
    args = parser.parse_args()

    broker = await get_broker()
    if broker is None:
        print("no KIS credentials configured on this deployment — set them in the env before running")
        return

    print(f"account_mode={settings.kis_account_mode}\n")
    print(f"{'code':<8} {'korean (prdt_abrv_name)':<28} english (prdt_eng_abrv_name)")
    print("-" * 72)
    for code in args.codes:
        try:
            status = await broker.fetch_stock_status(code)
            print(f"{code:<8} {status.korean_name:<28} {status.english_name!r}")
        except KISBrokerError as exc:
            print(f"{code:<8} ERROR: {exc}")
        except Exception as exc:  # noqa: BLE001 - dev script, surface anything
            print(f"{code:<8} ERROR: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    asyncio.run(main())
