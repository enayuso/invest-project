"""`get_fundamentals` (spec S:2-5).

Ratios (PER/PBR/ROE/부채비율) are computed here, never left for the model
to derive from raw numbers (spec S:1 principle 2) -- step-1 measurement
found yfinance's own `trailingPE`/`priceToBook` are `None` for every KRX
ticker tested, so there is no shortcut available anyway. Every money
field goes through `units.format_krw` (spec S:3-1); every ratio that
cannot be computed (loss-making PER, zero equity, ETF with no financial
statements) comes back as `null` with a `reason` string instead of an
estimate (spec S:3-4 / S:1 principle 3).
"""

from __future__ import annotations

from . import session, yf_gateway
from .resolver import TickerNotFoundError
from .units import format_krw, format_pct

_ANNUAL_YEARS = 5
_QUARTERS = 4

_REVENUE_ROWS = ("Total Revenue", "Operating Revenue")
_OPERATING_INCOME_ROWS = ("Operating Income",)
_NET_INCOME_ROWS = ("Net Income", "Net Income Common Stockholders")
_EQUITY_ROWS = ("Stockholders Equity", "Common Stock Equity")
_LIABILITIES_ROWS = ("Total Liabilities Net Minority Interest",)
_DILUTED_EPS_ROWS = ("Diluted EPS", "Basic EPS")


def _first_present_row(df, candidates: tuple[str, ...]):
    for name in candidates:
        if name in df.index:
            return df.loc[name]
    return None


def _merge_period_rows(financials, limit: int) -> list[dict]:
    """One entry per period with revenue/operating_income/net_income
    together, instead of three separate parallel arrays a model would
    have to zip itself."""
    revenue_row = _first_present_row(financials, _REVENUE_ROWS)
    op_income_row = _first_present_row(financials, _OPERATING_INCOME_ROWS)
    net_income_row = _first_present_row(financials, _NET_INCOME_ROWS)

    periods: list = []
    for row in (revenue_row, op_income_row, net_income_row):
        if row is not None:
            periods = list(row.index[:limit])
            break
    if not periods:
        return []

    def value_at(row, period):
        if row is None or period not in row.index:
            return None
        v = row[period]
        if v is None or v != v:
            return None
        return float(v)

    out = []
    for period in periods:
        out.append(
            {
                "period_end": period.date().isoformat(),
                "revenue": format_krw(value_at(revenue_row, period)),
                "operating_income": format_krw(value_at(op_income_row, period)),
                "net_income": format_krw(value_at(net_income_row, period)),
            }
        )
    return out


def get_fundamentals(yf_symbol: str) -> dict:
    info = session.fetch_info(yf_symbol)
    quote_type = info.get("quoteType")
    if not quote_type:
        raise TickerNotFoundError(
            f"{yf_symbol}: yfinance returned no quoteType -- cannot confirm this symbol exists."
        )
    is_etf = quote_type == "ETF"

    market_cap = format_krw(info.get("marketCap"))

    if is_etf:
        return {
            "yf_symbol": yf_symbol,
            "is_etf": True,
            "market_cap": market_cap,
            "financials_available": False,
            "reason": "ETF는 발행사(자산운용사)만 있고 개별 기업 재무제표를 발행하지 않습니다. "
            "PER/PBR/ROE/부채비율을 계산할 근거 데이터가 없습니다.",
            "missing": ["revenue", "operating_income", "net_income", "per", "pbr", "roe", "debt_ratio"],
        }

    financials = yf_gateway.get_financials(yf_symbol)
    quarterly = yf_gateway.get_quarterly_financials(yf_symbol)
    balance_sheet = yf_gateway.get_balance_sheet(yf_symbol)

    annual = _merge_period_rows(financials, _ANNUAL_YEARS)
    quarterly_rows = _merge_period_rows(quarterly, _QUARTERS)

    equity_row = _first_present_row(balance_sheet, _EQUITY_ROWS)
    liabilities_row = _first_present_row(balance_sheet, _LIABILITIES_ROWS)
    net_income_row = _first_present_row(financials, _NET_INCOME_ROWS)
    eps_row = _first_present_row(financials, _DILUTED_EPS_ROWS)

    missing: list[str] = []

    latest_equity = None
    if equity_row is not None and len(equity_row) > 0:
        v = equity_row.iloc[0]
        latest_equity = float(v) if v == v else None
    if latest_equity is None:
        missing.append("equity")

    latest_liabilities = None
    if liabilities_row is not None and len(liabilities_row) > 0:
        v = liabilities_row.iloc[0]
        latest_liabilities = float(v) if v == v else None
    if latest_liabilities is None:
        missing.append("total_liabilities")

    latest_net_income = None
    if net_income_row is not None and len(net_income_row) > 0:
        v = net_income_row.iloc[0]
        latest_net_income = float(v) if v == v else None
    if latest_net_income is None:
        missing.append("net_income")

    latest_eps = None
    if eps_row is not None and len(eps_row) > 0:
        v = eps_row.iloc[0]
        latest_eps = float(v) if v == v else None

    price = info.get("currentPrice") or info.get("regularMarketPrice") or info.get("previousClose")
    shares_outstanding = info.get("sharesOutstanding")

    # PER
    per: dict = {"value": None, "reason": None}
    if price is None:
        per["reason"] = "현재가를 확인할 수 없어 PER을 계산할 수 없습니다."
        missing.append("per")
    elif latest_eps is None:
        per["reason"] = "EPS 데이터가 없어 PER을 계산할 수 없습니다."
        missing.append("per")
    elif latest_eps <= 0:
        per["reason"] = "최근 회계연도 EPS가 0 이하(적자)라 PER이 의미가 없습니다."
        missing.append("per")
    else:
        per["value"] = round(price / latest_eps, 2)

    # PBR
    pbr: dict = {"value": None, "reason": None}
    if price is None:
        pbr["reason"] = "현재가를 확인할 수 없어 PBR을 계산할 수 없습니다."
        missing.append("pbr")
    elif latest_equity is None or shares_outstanding is None or shares_outstanding == 0:
        pbr["reason"] = "자본총계 또는 발행주식수 데이터가 없어 PBR을 계산할 수 없습니다."
        missing.append("pbr")
    elif latest_equity <= 0:
        pbr["reason"] = "자본총계가 0 이하라 PBR이 의미가 없습니다."
        missing.append("pbr")
    else:
        book_value_per_share = latest_equity / shares_outstanding
        pbr["value"] = round(price / book_value_per_share, 2)

    # ROE
    roe: dict = {"value": None, "reason": None}
    if latest_net_income is None or latest_equity is None:
        roe["reason"] = "순이익 또는 자본총계 데이터가 없어 ROE를 계산할 수 없습니다."
        missing.append("roe")
    elif latest_equity <= 0:
        roe["reason"] = "자본총계가 0 이하라 ROE가 의미가 없습니다."
        missing.append("roe")
    else:
        roe["value"] = format_pct(latest_net_income / latest_equity)

    # Debt ratio (부채비율 = 부채총계 / 자본총계 * 100)
    debt_ratio: dict = {"value": None, "reason": None}
    if latest_liabilities is None or latest_equity is None:
        debt_ratio["reason"] = "부채총계 또는 자본총계 데이터가 없어 부채비율을 계산할 수 없습니다."
        missing.append("debt_ratio")
    elif latest_equity <= 0:
        debt_ratio["reason"] = "자본총계가 0 이하라 부채비율이 의미가 없습니다."
        missing.append("debt_ratio")
    else:
        debt_ratio["value"] = format_pct(latest_liabilities / latest_equity)

    return {
        "yf_symbol": yf_symbol,
        "is_etf": False,
        "market_cap": market_cap,
        "financials_available": True,
        "annual": annual,
        "quarterly": quarterly_rows,
        "equity": format_krw(latest_equity),
        "total_liabilities": format_krw(latest_liabilities),
        "per": per,
        "pbr": pbr,
        "roe": roe,
        "debt_ratio": debt_ratio,
        "missing": sorted(set(missing)),
    }
