"""`get_price_snapshot` and `get_price_history` (spec S:2-2 / S:2-3).

The two structural rules this module exists to enforce:

1. Today's in-progress bar is never called a "close". While the KRX
   session is open, `get_price_snapshot` returns a `today_partial` block
   whose last-traded-price field is named `last_price` -- there is no
   field named `close` anywhere in the in-progress path. This is spec
   S:1 principle 1 ("프롬프트가 아니라 스키마로 막는다") applied literally:
   a field that doesn't exist can't be misread as a close.
2. `get_price_history` never contains the in-progress bar. Once the
   session is open, that day's row is dropped from the completed-bars
   array (it's available separately via `get_price_snapshot`).

Both tools run the freshness guard (via `session.get_session_view`)
before returning anything, so a wrong-suffix stale/cross-market response
is a loud error instead of a quiet bad analysis.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from . import freshness, integrity, session
from .units import format_pct

_KST = timezone(timedelta(hours=9))


def get_price_snapshot(yf_symbol: str) -> dict:
    view = session.get_session_view(yf_symbol, period="5d")
    bars = view.bars
    info = session.fetch_info(yf_symbol)

    prev_close = info.get("previousClose")
    if prev_close is None:
        # fall back to the second-to-last bar's close
        if len(bars) < 2:
            raise freshness.NoDataError(
                f"{yf_symbol}: cannot determine previous close (yfinance gave no previousClose "
                "and fewer than 2 daily bars are available)."
            )
        prev_close = float(bars["Close"].iloc[-2])

    result: dict = {
        "yf_symbol": yf_symbol,
        "session_status": view.session_status,
        "session_basis": view.session_basis,
        "prev_close": round(float(prev_close), 2),
    }

    last_row = bars.iloc[-1]
    if view.session_status == "in_progress":
        last_price = float(last_row["Close"])  # last traded price so far today, NOT a close
        result["today_partial"] = {
            "open": round(float(last_row["Open"]), 2),
            "high": round(float(last_row["High"]), 2),
            "low": round(float(last_row["Low"]), 2),
            "last_price": round(last_price, 2),
            "volume": int(last_row["Volume"]),
            "as_of": datetime.now(_KST).isoformat(),
        }
        reference_price = last_price
        on_date = datetime.now(_KST).date().isoformat()
    else:
        close_price = float(last_row["Close"])
        result["today_close"] = {
            "date": bars.index[-1].date().isoformat(),
            "open": round(float(last_row["Open"]), 2),
            "high": round(float(last_row["High"]), 2),
            "low": round(float(last_row["Low"]), 2),
            "close": round(close_price, 2),
            "volume": int(last_row["Volume"]),
        }
        reference_price = close_price
        on_date = result["today_close"]["date"]

    result["change_pct_vs_prev_close"] = format_pct((reference_price - prev_close) / prev_close)
    result["price_integrity"] = integrity.with_notice(
        integrity.assess_single_move(float(prev_close), reference_price, on_date)
    )
    return result


def get_price_history(yf_symbol: str, period: str = "3mo") -> dict:
    view = session.get_session_view(yf_symbol, period=period)
    completed = view.completed_bars
    if len(completed) == 0:
        raise freshness.NoDataError(
            f"{yf_symbol}: zero completed daily bars available for period={period!r} after excluding "
            "the in-progress session. Try a longer period, or wait until after this session closes."
        )

    bars_list = []
    for idx, row in completed.iterrows():
        bars_list.append(
            {
                "date": idx.date().isoformat(),
                "open": round(float(row["Open"]), 2),
                "high": round(float(row["High"]), 2),
                "low": round(float(row["Low"]), 2),
                "close": round(float(row["Close"]), 2),
                "volume": int(row["Volume"]),
            }
        )

    return {
        "yf_symbol": yf_symbol,
        "bars": bars_list,
        "bars_through": completed.index[-1].date().isoformat(),
        "price_integrity": integrity.with_notice(integrity.assess_bars(completed)),
    }
