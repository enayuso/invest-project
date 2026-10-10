"""End-to-end regression tests for `get_fundamentals` ROE / debt_ratio
(spec S:2-5).

Same rationale as `test_market_data.py`: pin what the tool actually
returns for `roe`/`debt_ratio`, not just `format_pct` in isolation. The
100x bug (already_percent=True on a raw net_income/equity or
liabilities/equity ratio) made a healthy 10.43% ROE come back as 0.1.
"""

from __future__ import annotations

import pandas as pd
import pytest

from stock_advisor_mcp import fundamentals


def _period_df(rows: dict[str, float], period: str = "2025-12-31") -> pd.DataFrame:
    col = pd.Timestamp(period)
    return pd.DataFrame({col: rows}).astype(float)


@pytest.fixture(autouse=True)
def fake_gateway(monkeypatch):
    monkeypatch.setattr(
        fundamentals.session,
        "fetch_info",
        lambda yf_symbol: {
            "quoteType": "EQUITY",
            "marketCap": 400_000_000_000_000,
            "currentPrice": None,
            "regularMarketPrice": None,
            "previousClose": None,
            "sharesOutstanding": None,
        },
    )
    financials = _period_df(
        {
            "Total Revenue": 1_000_000_000,
            "Operating Income": 200_000_000,
            "Net Income": 104_300_000,
            "Diluted EPS": 100.0,
        }
    )
    balance_sheet = _period_df(
        {
            "Stockholders Equity": 1_000_000_000,
            "Total Liabilities Net Minority Interest": 310_000_000,
        }
    )
    monkeypatch.setattr(fundamentals.yf_gateway, "get_financials", lambda yf_symbol: financials)
    monkeypatch.setattr(fundamentals.yf_gateway, "get_quarterly_financials", lambda yf_symbol: pd.DataFrame())
    monkeypatch.setattr(fundamentals.yf_gateway, "get_balance_sheet", lambda yf_symbol: balance_sheet)


def test_get_fundamentals_roe_is_true_percentage_not_raw_ratio():
    # net_income / equity = 104_300_000 / 1_000_000_000 = 0.1043 -> 10.43%.
    # The 100x bug returned 0.1 here.
    result = fundamentals.get_fundamentals("005930.KS")
    assert result["roe"]["value"] == pytest.approx(10.43, abs=0.005)
    assert result["roe"]["value"] > 1.0


def test_get_fundamentals_debt_ratio_is_true_percentage_not_raw_ratio():
    # liabilities / equity = 310_000_000 / 1_000_000_000 = 0.31 -> 31.0%.
    # The 100x bug returned 0.31.
    result = fundamentals.get_fundamentals("005930.KS")
    assert result["debt_ratio"]["value"] == pytest.approx(31.0, abs=0.01)
    assert result["debt_ratio"]["value"] > 1.0


def test_get_fundamentals_missing_price_marks_per_and_pbr_as_missing(monkeypatch):
    # price is None (no currentPrice/regularMarketPrice/previousClose) ->
    # per["reason"] / pbr["reason"] get set, but the old code forgot to also
    # append "per"/"pbr" to `missing`, unlike every other branch that sets a
    # reason. Pin the fix: both must show up in `missing`.
    monkeypatch.setattr(
        fundamentals.session,
        "fetch_info",
        lambda yf_symbol: {
            "quoteType": "EQUITY",
            "marketCap": 400_000_000_000_000,
            "currentPrice": None,
            "regularMarketPrice": None,
            "previousClose": None,
            "sharesOutstanding": None,
        },
    )
    result = fundamentals.get_fundamentals("005930.KS")
    assert result["per"]["value"] is None
    assert result["per"]["reason"]
    assert result["pbr"]["value"] is None
    assert result["pbr"]["reason"]
    assert "per" in result["missing"]
    assert "pbr" in result["missing"]


def test_get_fundamentals_etf_short_circuits_before_ratio_math(monkeypatch):
    monkeypatch.setattr(
        fundamentals.session,
        "fetch_info",
        lambda yf_symbol: {"quoteType": "ETF", "marketCap": 1_000_000_000},
    )
    result = fundamentals.get_fundamentals("379800.KS")
    assert result["is_etf"] is True
    assert result["financials_available"] is False
