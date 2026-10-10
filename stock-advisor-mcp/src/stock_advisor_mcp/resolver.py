"""`resolve_ticker` -- the single entry point every other tool depends on.

Design principle: **guessing the exchange suffix is forbidden everywhere
in this codebase.** Every other tool in this project accepts only a
`yf_symbol` that already carries the correct `.KS`/`.KQ` suffix, produced
here from the KRX master files (never from a heuristic, never from
yfinance's own name fields). That makes a suffix mixup structurally
impossible for the other four tools -- they simply never see a bare
code.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from . import yf_gateway
from .coverage import DATA_WE_DO_NOT_HAVE
from .errors import AdvisorError
from .krx_master import KrxMasterError, MasterLookupResult, get_master

_SUFFIXED = re.compile(r"^(?P<code>\d{6})\.(?P<suffix>KS|KQ)$", re.IGNORECASE)
_BARE_CODE = re.compile(r"^\d{6}$")

_MARKET_TO_SUFFIX = {"KOSPI": "KS", "KOSDAQ": "KQ"}


class TickerNotFoundError(AdvisorError):
    """The code was not found in either KRX master file, or a name query
    matched zero (or more than one, ambiguously) companies. Per spec
    S:2-1, this must be an error -- never a guessed symbol."""


class AmbiguousTickerError(AdvisorError):
    def __init__(self, query: str, candidates: list[MasterLookupResult]):
        self.candidates = candidates
        listing = ", ".join(f"{c.name_ko}({c.code}.{_MARKET_TO_SUFFIX[c.market]})" for c in candidates)
        super().__init__(f"'{query}'가 여러 종목과 일치합니다: {listing}. 정확한 코드를 사용하세요.")


@dataclass(frozen=True)
class ResolvedTicker:
    code: str
    market: str  # "KOSPI" | "KOSDAQ"
    yf_symbol: str
    name_ko: str
    is_etf: bool
    data_coverage: list[str]

    def as_dict(self) -> dict:
        return {
            "code": self.code,
            "market": self.market,
            "yf_symbol": self.yf_symbol,
            "name_ko": self.name_ko,
            "is_etf": self.is_etf,
            "data_coverage": self.data_coverage,
        }


def _extract_bare_code(query: str) -> str | None:
    q = query.strip()
    m = _SUFFIXED.match(q)
    if m:
        return m.group("code")
    if _BARE_CODE.match(q):
        return q
    return None


def _lookup_is_etf(yf_symbol: str) -> bool:
    try:
        info = yf_gateway.get_info(yf_symbol)
    except Exception as exc:  # network/HTTP error from yfinance
        raise TickerNotFoundError(
            f"{yf_symbol}: yfinance quote lookup failed while checking ETF status ({exc}). "
            "Refusing to assume is_etf=false."
        ) from exc
    quote_type = info.get("quoteType")
    if not quote_type:
        raise TickerNotFoundError(
            f"{yf_symbol}: yfinance returned no quoteType -- this usually means yfinance has no "
            "listing for this exact symbol even though the KRX master file does. Refusing to guess."
        )
    return quote_type == "ETF"


def resolve_ticker(query: str) -> ResolvedTicker:
    """query: a 6-digit code ("005930"), a code with suffix ("005930.KS"),
    or a Korean company name ("삼성전자")."""
    master = get_master()
    try:
        master.ensure_loaded()
    except KrxMasterError:
        raise  # propagate as-is: caller must not fall back to guessing

    bare_code = _extract_bare_code(query)
    if bare_code is not None:
        hit = master.lookup_code(bare_code)
        if hit is None:
            raise TickerNotFoundError(
                f"'{bare_code}' is not in either KRX master file (KOSPI or KOSDAQ). "
                "It may be delisted, a non-stock instrument, or mistyped."
            )
    else:
        candidates = master.search_name(query, limit=8)
        exact = [c for c in candidates if c.name_ko == query.strip()]
        if len(exact) == 1:
            hit = exact[0]
        elif len(candidates) == 1:
            hit = candidates[0]
        elif len(candidates) == 0:
            raise TickerNotFoundError(f"No KRX-listed company name matches '{query}'.")
        else:
            raise AmbiguousTickerError(query, candidates)

    suffix = _MARKET_TO_SUFFIX[hit.market]
    yf_symbol = f"{hit.code}.{suffix}"
    is_etf = _lookup_is_etf(yf_symbol)

    return ResolvedTicker(
        code=hit.code,
        market=hit.market,
        yf_symbol=yf_symbol,
        name_ko=hit.name_ko,
        is_etf=is_etf,
        data_coverage=list(DATA_WE_DO_NOT_HAVE),
    )
