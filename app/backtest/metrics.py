"""Rating-to-exposure scoring for backtest replay.

A lightweight proxy metric appropriate for Phase 2 — real position sizing,
stop-loss, and execution belong to Phase 4. This just answers "would
following these decisions have beaten buy-and-hold," treating each decided
rating as a signed exposure held until the next decision date.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from app.state import PortfolioDecision, PortfolioRating

EXPOSURE = {
    PortfolioRating.BUY: 1.0,
    PortfolioRating.OVERWEIGHT: 0.5,
    PortfolioRating.HOLD: 0.0,
    PortfolioRating.UNDERWEIGHT: -0.5,
    PortfolioRating.SELL: -1.0,
}


@dataclass
class BacktestRecord:
    trade_date: date
    decision: PortfolioDecision
    close: float
    adjustment_note: str | None = None  # RebalanceResult.adjustment_note, if a risk cap changed the outcome
    # Trailing-stop replay state (2026-08-10) — carried forward from one
    # record to the next by app.backtest.harness's own loop (never
    # persisted to app.execution.trailing_stop's live DB table, which is
    # keyed by real account/user and must never be touched by a backtest
    # replay). `None` until a position with an open stop-loss and a
    # resolvable entry has been seen at least once.
    trailing_high_water: float | None = None
    trailing_entry_price: float | None = None  # the entry this high_water is measured from; a changed
    # value (add-on / reopened position) is this replay's own signal to
    # reset trailing_high_water, mirroring app.execution.trailing_stop's
    # entry_time-mismatch reset for the live path.
    # memory_entries.id for the pending row `store_pending` wrote for THIS
    # record's decision (app.state.PipelineState.decision_id, set inside
    # app.orchestrator.pipeline's `_run_pipeline_impl`) — `None` only if
    # store_pending itself failed for that day (DB hiccup; degrades quietly,
    # see app.memory.store's own module docstring). `app.backtest.harness`
    # needs this to resolve THIS EXACT row by id rather than by
    # (ticker, trade_date), which more than one row can share — see
    # app.memory.store.resolve_outcome's own docstring for why that used to
    # be unsafe.
    decision_id: int | None = None


@dataclass
class RiskMetrics:
    """Per-decision-period risk metrics (NOT annualized — decision cadence
    varies with `cadence_days`, so a fixed sqrt(252)-style annualization
    factor would silently misrepresent risk for anything but daily sampling.
    `None` means undefined for too little/degenerate data, rather than a
    misleading 0.0 or an unbounded value."""

    sharpe: float | None
    sortino: float | None
    max_drawdown: float | None
    cvar_95: float | None

    def is_empty(self) -> bool:
        return all(v is None for v in (self.sharpe, self.sortino, self.max_drawdown, self.cvar_95))


@dataclass
class BacktestResult:
    ticker: str
    benchmark_ticker: str
    records: list[BacktestRecord]
    strategy_return: float
    buy_and_hold_return: float
    benchmark_return: float
    exposure_risk_metrics: RiskMetrics
    ledger_risk_metrics: RiskMetrics


def compute_period_returns(records: list[BacktestRecord]) -> list[float]:
    """The per-period (exposure * realized return) series underlying
    `compute_strategy_return`'s compounded total — exposed separately here
    since Sharpe/Sortino/drawdown/CVaR all need the individual periods, not
    just the final compounded number. This is a signal-quality proxy only:
    it ignores real position sizing and stop-loss exits — see
    `compute_ledger_period_returns` for the metric that reflects what
    actually got executed."""
    return [
        EXPOSURE[prev.decision.rating] * (nxt.close - prev.close) / prev.close
        for prev, nxt in zip(records, records[1:])
    ]


def compute_ledger_period_returns(equity_curve: list[float]) -> list[float]:
    """Period returns from the real paper ledger's mark-to-market equity at
    each decision date — reflects actual ATR-scaled position sizing and any
    stop-loss exits triggered between decisions, unlike
    `compute_period_returns`'s signed-exposure proxy. This is what the
    primary risk metrics (Sharpe/Sortino/drawdown/CVaR) should be computed
    from; the exposure-proxy series is a secondary, simplified signal only."""
    returns = []
    for prev, nxt in zip(equity_curve, equity_curve[1:]):
        if prev == 0:
            continue  # degenerate: equity fully depleted, no meaningful return to report
        returns.append((nxt - prev) / prev)
    return returns


def compute_strategy_return(records: list[BacktestRecord]) -> float:
    """Compounds the exposure decided at date i over the realized return from
    date i to date i+1. The rating decided on the final date is never
    realized — there's no date i+1 left to hold it into."""
    equity = 1.0
    for period_return in compute_period_returns(records):
        equity *= 1.0 + period_return
    return equity - 1.0


def price_on_or_before(prices: dict[date, float], target: date) -> float:
    """The most recent available price on or before `target`. A plain
    `prices[target]` assumes `target` came from the same market's trading
    calendar as `prices` — true when both are the ticker's own dates, but
    not when `prices` is a *benchmark's* series and `target` is a date the
    ticker traded on but the benchmark's market didn't (e.g. a KRX-listed
    ticker's trading day landing on a US holiday SPY doesn't have a price
    for, or vice versa). Falls back to the closest earlier trading day
    instead of raising."""
    if target in prices:
        return prices[target]
    earlier = [d for d in prices if d < target]
    if not earlier:
        raise KeyError(f"no price on or before {target}")
    return prices[max(earlier)]


def compute_buy_and_hold_return(prices: dict[date, float], first: date, last: date) -> float:
    return (price_on_or_before(prices, last) - price_on_or_before(prices, first)) / price_on_or_before(prices, first)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _stdev(values: list[float]) -> float:
    """Sample standard deviation (n-1 denominator) — the conventional choice
    when treating backtest returns as a sample estimating population risk."""
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    variance = sum((v - m) ** 2 for v in values) / (len(values) - 1)
    return variance**0.5


def sharpe_ratio(returns: list[float], risk_free_rate: float = 0.0) -> float | None:
    """Mean excess return / stdev of returns, per decision period (not
    annualized — see `RiskMetrics` docstring). `None` if fewer than 2
    periods or the return series has zero variance (e.g. every period
    identical, most commonly an all-Hold backtest)."""
    if len(returns) < 2:
        return None
    excess = [r - risk_free_rate for r in returns]
    stdev = _stdev(excess)
    if stdev == 0.0:
        return None
    return _mean(excess) / stdev


def sortino_ratio(returns: list[float], risk_free_rate: float = 0.0) -> float | None:
    """Like `sharpe_ratio` but only penalizes downside deviation. Downside
    deviation divides by the TOTAL number of periods (not just the count of
    negative ones) — the standard Sortino/van der Meer definition, treating
    non-negative periods as zero deviation rather than excluding them,
    which would understate the denominator and inflate the ratio. `None` if
    fewer than 2 periods, or no downside periods exist at all (undefined,
    not infinitely good)."""
    if len(returns) < 2:
        return None
    excess = [r - risk_free_rate for r in returns]
    downside = [min(e, 0.0) ** 2 for e in excess]
    if not any(d > 0 for d in downside):
        return None
    downside_dev = (sum(downside) / len(downside)) ** 0.5
    return _mean(excess) / downside_dev


def annualized_sharpe(returns: list[float], cadence_days: int, risk_free_rate: float = 0.0) -> float | None:
    """Qlib-style annualized Sharpe — what Qlib's own `risk_analysis` calls
    "information_ratio" (a naming quirk worth flagging: that's NOT the more
    common "excess return over a benchmark" definition of IR, just mean/
    std of the return series itself, annualized; see
    qlib-poc-decision-2026-08-21 project memory) — mean/stdev of per-period
    excess return, scaled by sqrt(trading days per year / `cadence_days`).

    Deliberately a SEPARATE function from `sharpe_ratio`, not a replacement
    for `RiskMetrics.sharpe` — that field stays unannualized for the reason
    its own docstring already gives (`cadence_days` varies run to run, so a
    single fixed sqrt(252) factor would misrepresent risk for anything but
    daily sampling). This instead takes `cadence_days` explicitly from the
    caller, for one that specifically wants a cross-cadence-comparable
    figure and accepts the "252 trading days/year" approximation that
    implies. `None` under the same conditions as `sharpe_ratio` (which this
    wraps), or if `cadence_days` isn't positive."""
    period_sharpe = sharpe_ratio(returns, risk_free_rate)
    if period_sharpe is None or cadence_days <= 0:
        return None
    periods_per_year = 252 / cadence_days
    return period_sharpe * periods_per_year**0.5


def max_drawdown(returns: list[float]) -> float | None:
    """Largest peak-to-trough decline in the cumulative equity curve implied
    by `returns`, as a negative fraction (e.g. -0.15 = a 15% drawdown at the
    worst point). `None` if there are no periods at all."""
    if not returns:
        return None
    equity = 1.0
    peak = 1.0
    worst = 0.0
    for r in returns:
        equity *= 1.0 + r
        peak = max(peak, equity)
        drawdown = (equity - peak) / peak
        worst = min(worst, drawdown)
    return worst


def cvar(returns: list[float], alpha: float = 0.05) -> float | None:
    """Conditional Value at Risk (Expected Shortfall) at the given alpha
    level: the average return across the worst `alpha` fraction of periods.
    `None` if there isn't at least one full tail period to average.

    Requires at least 2 tail periods to average -- `n_tail` is pinned at 1
    by the max(1, ...) below for every len(returns) up to 29 at alpha=0.05
    (it first exceeds 1 at len(returns)==30), meaning without this guard
    "CVaR" would silently BE the single worst observed period for the vast
    majority of realistic backtest lengths, indistinguishable in its output
    from a genuine multi-period tail average -- while every caller already
    renders a None result as "N/A", so there's an existing, correct
    degrade path for this to fall back to.
    타점이 2개 미만이면 '평균'이 아니라 최악값 1개와 동일해지므로 None 처리."""
    n_tail = max(1, round(len(returns) * alpha))
    if len(returns) < n_tail or n_tail < 2:
        return None
    worst_sorted = sorted(returns)[:n_tail]
    return _mean(worst_sorted)


# Mirror of app.audit.stats._WIN_DIRECTION for in-run backtest scoring. 'Hold'
# is deliberately ABSENT: stats.py splits a live Hold into 'Hold (held)'
# (win test raw_return > 0) vs 'Hold (flat)' (win test alpha_return < 0) using
# memory_entries.held_at_decision — information a backtest replay never
# captures (see harness._process_ticker_date's held_at_decision=None comment).
# A rating with no direction here is unscored, never counted as a loss.
_ALPHA_WIN_DIRECTION: dict[str, int] = {
    "Buy": 1,
    "Overweight": 1,
    "Underweight": -1,
    "Sell": -1,
}


def alpha_win(rating: str, alpha_return: float) -> bool | None:
    """Direction-aware win test for a single (rating, alpha_return) pair —
    Sell/Underweight win by UNDERPERFORMING the benchmark (successfully
    reduced/avoided exposure), not by beating it. `None` (unscored) for a
    rating with no determinable direction (Hold) or a NaN alpha_return."""
    direction = _ALPHA_WIN_DIRECTION.get(rating)
    if direction is None:
        return None
    if alpha_return != alpha_return:  # NaN guard, same convention as app.audit.stats
        return None
    return alpha_return > 0 if direction > 0 else alpha_return < 0