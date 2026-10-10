"""In-memory alpha recomputation for still-pending live decisions -- audit-side
only. Nothing here writes to the DB, generates a reflection, or calls
`app.memory.auto_resolve.resolve_outcome`.

WHY THIS EXISTS (2026-09-28 investigation, handoff/CURRENT.md): real-account
domestic decisions can stay `status='pending'` long after their holding
window because `auto_resolve.resolve_ticker` asks yfinance for the STORED
ticker string. This repo stores KOSDAQ names as '.KS' by convention, but
Yahoo only serves them under '.KQ', so the price lookup fails and the row
never resolves. `entry-paired` scores `status='resolved'` rows only, so those
names silently drop out of the sample (selection bias). Fixing the resolver
would generate new reflections that later flow into `past_context` -- a
behavior change under the freeze (RULES §2) -- so this module only
RE-MEASURES them for the audit, in memory, with a '.KS'<->'.KQ' fallback.

Design contract (Opus spec, 2026-09-28):
* alpha formula and benchmark are identical to `auto_resolve`
  (raw - benchmark, benchmark = `benchmark_for_market("domestic")`).
* variable basis evaluates at the FIRST settled common close on/after
  `trade_date + settings.memory_min_holding_days`, NOT `auto_resolve.
  _common_eval_date` (that returns the LATEST common close, so a backlog row
  would get a 10-26 day window and stop being comparable to the 5-7 day rows
  that resolved promptly). The horizon basis reuses `_horizon_eval_date`.
* `session_in_progress` is always asked about the STORED ticker (.KS): it
  classifies a non-'.KS' symbol as overseas, which would accept today's
  still-forming bar as a settled close.
* `_UNKNOWN_TICKER_CACHE` is never read, written or cleared here.
"""

from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass, field
from datetime import date, timedelta

from app.config import settings
from app.dataflows.session_state import session_in_progress
from app.dataflows.yfinance import get_price_series
from app.execution.portfolio_risk import benchmark_for_market
from app.memory.auto_resolve import _horizon_eval_date, _usable

logger = logging.getLogger(__name__)

REASON_PRICE_UNAVAILABLE = "price_unavailable"
REASON_BENCHMARK_UNAVAILABLE = "benchmark_unavailable"
REASON_NO_ENTRY_CLOSE = "no_entry_close"
REASON_NOT_MATURED = "not_matured"
REASON_FETCH_ERROR_PREFIX = "fetch_error:"  # + exception type; a hard failure, never "missing data"


@dataclass(frozen=True)
class RecomputedRow:
    ticker: str  # stored ticker (never the symbol actually used to fetch prices)
    trade_date: date
    rating: str
    held_at_decision: bool | None
    held_at_decision_inferred: bool | None
    alpha_return: float
    raw_return: float
    benchmark_return: float
    eval_date: date
    holding_days: int
    price_symbol: str  # '.KS' or '.KQ' -- the symbol that actually returned bars

    def as_report_row(self) -> dict:
        """Shape `entry_paired._build_paired_report` / `_classify_hold` read."""
        return {
            "trade_date": self.trade_date, "ticker": self.ticker, "rating": self.rating,
            "alpha_return": self.alpha_return,
            "held_at_decision": self.held_at_decision,
            "held_at_decision_inferred": self.held_at_decision_inferred,
            "source": "recomputed",
        }


@dataclass(frozen=True)
class RecomputeFailure:
    ticker: str
    trade_date: date
    rating: str
    reason: str  # a REASON_* constant, or REASON_FETCH_ERROR_PREFIX + exception type name


@dataclass(frozen=True)
class PendingRecomputeResult:
    rows: list[RecomputedRow] = field(default_factory=list)
    failures: list[RecomputeFailure] = field(default_factory=list)
    n_alt_symbol_tickers: int = 0  # DISTINCT TICKERS whose bars only came back under the alternate suffix
    n_tickers_fetched: int = 0  # distinct tickers whose price fetch succeeded (either suffix)


def alternate_krx_symbol(ticker: str) -> str | None:
    """'.KS' <-> '.KQ'; anything else has no alternate. 6-digit KRX codes are
    unique across markets, so the same code under the other suffix is the same
    security."""
    if ticker.endswith(".KS"):
        return ticker[: -len(".KS")] + ".KQ"
    if ticker.endswith(".KQ"):
        return ticker[: -len(".KQ")] + ".KS"
    return None


async def fetch_price_series_with_krx_fallback(
    ticker: str, start: date, end: date,
) -> tuple[str, dict[date, float]]:
    """(symbol actually used, close series). Tries `ticker` first; only a
    failed/empty lookup falls back to the alternate suffix -- a missing ENTRY
    close is deliberately not a fallback reason (never mix two symbols'
    series). Raises ValueError when every symbol fails."""
    try:
        series = await get_price_series(ticker, start, end)
        if series:
            return ticker, series
    except ValueError:
        pass
    alternate = alternate_krx_symbol(ticker)
    if alternate is None:
        raise ValueError(f"no price history for {ticker!r} and no alternate symbol")
    series = await get_price_series(alternate, start, end)
    if not series:
        raise ValueError(f"no price history for {ticker!r} or {alternate!r}")
    return alternate, series


def first_common_eval_date(
    session_ticker: str, benchmark_ticker: str,
    prices: dict[date, float], benchmark_prices: dict[date, float],
    not_before: date,
) -> date | None:
    """EARLIEST date on/after `not_before` where BOTH series carry a usable
    close and neither session is still forming. `session_ticker` must be the
    stored '.KS' ticker (see module docstring)."""
    shared = sorted(d for d in prices.keys() & benchmark_prices.keys() if d >= not_before)
    for candidate in shared:
        if (
            _usable(prices.get(candidate))
            and _usable(benchmark_prices.get(candidate))
            and not session_in_progress(session_ticker, candidate)
            and not session_in_progress(benchmark_ticker, candidate)
        ):
            return candidate
    return None


async def recompute_pending_alphas(
    pending_rows, *, basis: str, today: date,
) -> PendingRecomputeResult:
    """`pending_rows`: Records/dicts with ticker, trade_date, rating,
    held_at_decision, held_at_decision_inferred. Sequential on purpose (one
    benchmark fetch, one fetch per ticker, deterministic order)."""
    if basis not in ("variable", "horizon"):
        raise ValueError(f"basis must be 'variable' or 'horizon', got {basis!r}")
    rows = list(pending_rows)
    if not rows:
        return PendingRecomputeResult()

    benchmark = benchmark_for_market("domestic")
    rows_out: list[RecomputedRow] = []
    failures: list[RecomputeFailure] = []

    def _fail(row, reason: str) -> None:
        failures.append(RecomputeFailure(row["ticker"], row["trade_date"], row["rating"], reason))

    # Only ValueError means "Yahoo has no bars for this symbol" (get_price_series'
    # contract). Anything else (rate limit, network, a code bug) must not be
    # relabeled as missing data: the benchmark case re-raises, the per-ticker
    # case records `fetch_error:<ExcType>` so it stays visible in the report.
    try:
        benchmark_prices = await get_price_series(benchmark, min(r["trade_date"] for r in rows), today)
    except ValueError:
        logger.warning("pending recompute: benchmark %s unavailable", benchmark, exc_info=True)
        for row in rows:
            _fail(row, REASON_BENCHMARK_UNAVAILABLE)
        return PendingRecomputeResult(rows=[], failures=failures)

    by_ticker: dict[str, list] = {}
    for row in rows:
        by_ticker.setdefault(row["ticker"], []).append(row)

    n_alt = 0
    n_fetched = 0
    for ticker in sorted(by_ticker):
        ticker_rows = by_ticker[ticker]
        try:
            symbol, prices = await fetch_price_series_with_krx_fallback(
                ticker, min(r["trade_date"] for r in ticker_rows), today,
            )
        except ValueError:
            logger.warning("pending recompute: no prices for %s (both suffixes)", ticker, exc_info=True)
            for row in ticker_rows:
                _fail(row, REASON_PRICE_UNAVAILABLE)
            continue
        except Exception as exc:
            logger.warning("pending recompute: fetch error for %s", ticker, exc_info=True)
            for row in ticker_rows:
                _fail(row, f"{REASON_FETCH_ERROR_PREFIX}{type(exc).__name__}")
            continue
        n_fetched += 1
        if symbol != ticker:
            n_alt += 1

        for row in ticker_rows:
            trade_date = row["trade_date"]
            if not _usable(prices.get(trade_date)) or not _usable(benchmark_prices.get(trade_date)):
                _fail(row, REASON_NO_ENTRY_CLOSE)
                continue

            if basis == "variable":
                not_before = trade_date + timedelta(days=settings.memory_min_holding_days)
                eval_date = first_common_eval_date(ticker, benchmark, prices, benchmark_prices, not_before)
                holding_days = (eval_date - trade_date).days if eval_date is not None else 0
            else:
                horizon = _horizon_eval_date(
                    ticker, benchmark, prices, benchmark_prices, trade_date,
                    settings.memory_fixed_horizon_trading_days,
                )
                eval_date, holding_days = horizon if horizon is not None else (None, 0)
            if eval_date is None:
                _fail(row, REASON_NOT_MATURED)
                continue

            raw_return = (prices[eval_date] - prices[trade_date]) / prices[trade_date]
            benchmark_return = (
                benchmark_prices[eval_date] - benchmark_prices[trade_date]
            ) / benchmark_prices[trade_date]
            rows_out.append(RecomputedRow(
                ticker=ticker, trade_date=trade_date, rating=row["rating"],
                held_at_decision=row["held_at_decision"],
                held_at_decision_inferred=row["held_at_decision_inferred"],
                alpha_return=raw_return - benchmark_return,
                raw_return=raw_return, benchmark_return=benchmark_return,
                eval_date=eval_date, holding_days=holding_days, price_symbol=symbol,
            ))

    rows_out.sort(key=lambda r: (r.trade_date, r.ticker))
    return PendingRecomputeResult(
        rows=rows_out, failures=failures, n_alt_symbol_tickers=n_alt, n_tickers_fetched=n_fetched,
    )


def holding_days_stats(rows: list[RecomputedRow]) -> tuple[int | None, float | None, int | None]:
    """(min, median, max) of holding_days, or (None, None, None) when empty."""
    if not rows:
        return None, None, None
    days = [r.holding_days for r in rows]
    return min(days), statistics.median(days), max(days)
