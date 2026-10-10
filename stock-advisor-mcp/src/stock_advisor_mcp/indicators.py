"""`get_technical_indicators` (spec S:2-4).

All values here are computed by this tool, never by the model (spec S:1
principle 2). The in-progress bar is excluded from every calculation --
including it would make every indicator change value continuously
through the trading session, the same "value is a function of scan
time" trap the sibling trading system hit with its screening ranks.
`computed_through` tells the caller exactly which bar the numbers stop
at, so the model can say "as of 2026-09-18" instead of implying these
are today's live values.

Computation is delegated to `stockstats` (BSD-3-Clause, license
confirmed before this dependency was added -- see spec S:5), which is
also what Tauric Research's TradingAgents uses for the same indicators.
"""

from __future__ import annotations

from stockstats import StockDataFrame

from . import freshness, integrity, session

# 120-day MA needs meaningfully more than 120 calendar days of history to
# have 120 *trading* days once weekends/holidays are excluded.
_HISTORY_PERIOD = "1y"

_REQUIRED_TRADING_DAYS = {
    "sma_5": 5,
    "sma_20": 20,
    "sma_60": 60,
    "sma_120": 120,
    "volume_sma_20": 20,
}


def _round_or_none(value) -> float | None:
    if value is None:
        return None
    try:
        if value != value:  # NaN
            return None
    except TypeError:
        return None
    return round(float(value), 4)


def get_technical_indicators(yf_symbol: str) -> dict:
    view = session.get_session_view(yf_symbol, period=_HISTORY_PERIOD)
    completed = view.completed_bars
    if len(completed) == 0:
        raise freshness.NoDataError(
            f"{yf_symbol}: zero completed daily bars available to compute indicators from "
            "(after excluding the in-progress session)."
        )

    df = completed[["Open", "High", "Low", "Close", "Volume"]].rename(columns=str.lower).copy()
    n = len(df)
    sdf = StockDataFrame.retype(df.copy())

    missing: list[str] = []

    def field(key: str, requirement_name: str | None = None) -> float | None:
        if requirement_name is not None and n < _REQUIRED_TRADING_DAYS[requirement_name]:
            missing.append(requirement_name)
            return None
        try:
            value = sdf[key].iloc[-1]
        except Exception:
            missing.append(key)
            return None
        result = _round_or_none(value)
        if result is None:
            missing.append(key)
        return result

    result = {
        "yf_symbol": yf_symbol,
        "computed_through": completed.index[-1].date().isoformat(),
        "rsi_14": field("rsi_14"),
        "macd": {
            "macd": field("macd"),
            "signal": field("macds"),
            "histogram": field("macdh"),
        },
        "moving_averages": {
            "sma_5": field("close_5_sma", "sma_5"),
            "sma_20": field("close_20_sma", "sma_20"),
            "sma_60": field("close_60_sma", "sma_60"),
            "sma_120": field("close_120_sma", "sma_120"),
        },
        "bollinger_bands": {
            "mid": field("boll"),
            "upper": field("boll_ub"),
            "lower": field("boll_lb"),
        },
        "atr_14": field("atr_14"),
        "volume_sma_20": field("volume_20_sma", "volume_sma_20"),
        "missing": sorted(set(missing)),
    }
    block = integrity.with_notice(integrity.assess_bars(completed))
    block["affected_indicators"] = integrity.affected_indicators(completed, block["limit_breaches"])
    result["price_integrity"] = block
    return result
