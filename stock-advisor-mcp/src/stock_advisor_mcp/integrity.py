"""Price-continuity checks (pure functions, no I/O).

Why this exists: yfinance can hand back series with splits, reverse
splits, capital reductions or trading halts that show up as daily moves
KRX's +/-30% price limit makes impossible in normal trading. A model
that sees such a bar and narrates "-95% crash" is describing a data
artifact. These functions only *detect and report*; they never guess the
cause. The tool layer decides when to attach `PRICE_INTEGRITY_NOTICE`
(see `with_notice`).

Deliberately free of yfinance/session/cache/server imports so they can be
tested on synthetic frames and never trigger network access.
"""

from __future__ import annotations

import pandas as pd

# KRX daily limit +/-30%, plus 0.5%p tolerance because yfinance's
# dividend-adjusted closes can make consecutive-bar ratios drift slightly.
LIMIT_BREACH_THRESHOLD = 0.305
LONG_GAP_CALENDAR_DAYS = 15

# Completed-bar windows each indicator needs. RSI/ATR/MACD are exponential
# (a breach never fully leaves them), so these windows are an approximation
# of how long a breach visibly distorts the value.
INDICATOR_WINDOWS = {
    "sma_5": 5,
    "sma_20": 20,
    "bollinger_bands": 20,
    "volume_sma_20": 20,
    "rsi_14": 15,
    "atr_14": 15,
    "macd": 35,
    "sma_60": 60,
    "sma_120": 120,
}

PRICE_INTEGRITY_NOTICE = (
    "가격제한폭(±30%)을 넘는 일간 변동은 정상 거래로 생길 수 없습니다. "
    "액면분할·병합·감자·권리락 등 기준가 재산정, 거래 재개, 또는 데이터 오류일 수 있으며 "
    "이 도구는 원인을 확인할 수 없습니다. 해당 구간을 포함한 수익률·지표를 근거로 쓰지 말고 "
    "'가격 연속성 확인 불가'로 서술하세요."
)


def _valid(x) -> bool:
    return x is not None and x == x


def _breach(prev_close, close, on_date: str) -> dict | None:
    if not _valid(prev_close) or not _valid(close) or prev_close <= 0:
        return None
    ratio = float(close) / float(prev_close) - 1.0
    if abs(ratio) > LIMIT_BREACH_THRESHOLD:
        return {
            "date": on_date,
            "prev_close": round(float(prev_close), 2),
            "close": round(float(close), 2),
            "change_pct": round(ratio * 100.0, 2),
        }
    return None


def assess_bars(bars: pd.DataFrame) -> dict:
    """Check completed daily bars for limit breaches, zero volume, long gaps."""
    result: dict = {
        "status": "ok",
        "bars_checked": int(len(bars)),
        "checked_through": None,
        "limit_breaches": [],
        "zero_volume_dates": [],
        "long_gaps": [],
    }
    if len(bars) == 0:
        return result

    dates = [pd.Timestamp(i) for i in bars.index]
    closes = list(bars["Close"])
    volumes = list(bars["Volume"]) if "Volume" in bars.columns else [None] * len(bars)
    result["checked_through"] = dates[-1].date().isoformat()

    for i in range(len(bars)):
        v = volumes[i]
        if _valid(v) and v == 0:
            result["zero_volume_dates"].append(dates[i].date().isoformat())
        if i == 0:
            continue
        b = _breach(closes[i - 1], closes[i], dates[i].date().isoformat())
        if b is not None:
            result["limit_breaches"].append(b)
        gap = (dates[i] - dates[i - 1]).days
        if gap >= LONG_GAP_CALENDAR_DAYS:
            result["long_gaps"].append(
                {
                    "from": dates[i - 1].date().isoformat(),
                    "to": dates[i].date().isoformat(),
                    "calendar_days": int(gap),
                }
            )

    if result["limit_breaches"] or result["zero_volume_dates"] or result["long_gaps"]:
        result["status"] = "anomalies_found"
    return result


def affected_indicators(bars: pd.DataFrame, limit_breaches: list[dict]) -> list[str]:
    """Indicators whose window still contains a breach bar (sorted, unique)."""
    dates = [pd.Timestamp(i).date().isoformat() for i in bars.index]
    affected: set[str] = set()
    for br in limit_breaches:
        try:
            pos = dates.index(br["date"])
        except ValueError:
            continue
        distance = len(dates) - 1 - pos
        for name, window in INDICATOR_WINDOWS.items():
            if distance < window:
                affected.add(name)
    return sorted(affected)


def assess_single_move(prev_close: float | None, price: float | None, on_date: str) -> dict:
    """C1 only, for a single prev_close -> price move (snapshot)."""
    result: dict = {
        "status": "ok",
        "bars_checked": 1,
        "checked_through": on_date,
        "limit_breaches": [],
        "zero_volume_dates": [],
        "long_gaps": [],
    }
    b = _breach(prev_close, price, on_date)
    if b is not None:
        result["limit_breaches"].append(b)
        result["status"] = "anomalies_found"
    return result


def with_notice(assessment: dict) -> dict:
    """Copy of `assessment`, plus the notice only when anomalies were found."""
    out = dict(assessment)
    if out.get("status") == "anomalies_found":
        out["notice"] = PRICE_INTEGRITY_NOTICE
    return out
