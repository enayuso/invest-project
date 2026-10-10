"""Single source of truth for the per-side trade cost formula — broker
commission (domestic vs overseas, since the two are unrelated KIS fee
schedules) plus, on a domestic (`.KS`) SELL only, the Korean Securities
Transaction Tax (증권거래세) — shared by the live/paper ledger
(`app.execution.ledger.record_trade`/`reverse_pending_trade`) and the
backtest harnesses (`app.backtest.scalping_harness`) so a tax-rate change
or future ETF-exemption handling is never edited in one place and forgotten
in the other. Direct precedent: `app.dataflows.indicators` was split out of
the yfinance path for exactly this reason — RSI was shared but SMA/MACD
were duplicated across the yfinance and KIS paths until that module existed
(see oss-candidates-license-review-2026-08-21 project memory).

Pure functions only — no I/O, no DB, no broker, no clock — so a live ledger
write and a historical backtest replay both compute the identical number
from the identical formula instead of two hand-copies of the same rule
drifting apart the next time either one is edited.

Market split (2026-09-08 fee/tax/slippage audit): before this, every caller
here charged the DOMESTIC commission (`transaction_cost_bps`, 5bp) on
overseas trades too — silently understating real US round-trip cost by
roughly 5x in every overseas backtest/sweep/live figure that reads this
module. `.KS` suffix is the same domestic/overseas convention
`app.execution.portfolio_risk.ticker_market` uses (duplicated inline here
rather than imported, to keep this module free of that heavier module's
own dependencies).

Known gap, carried over from `settings.sell_tax_bps_domestic`'s own
docstring: Korean ETFs are exempt from 증권거래세 and there is no
instrument-type detection here, so an ETF sell is over-charged — the
conservative direction (never flatters a strategy), left as-is until ETF
trading is actually in scope.
"""

from __future__ import annotations

from app.config import settings


def transaction_cost_bps_for(ticker: str, side: str) -> float:
    """Total cost in bps for one side of a trade: broker commission
    (`settings.transaction_cost_bps` domestic / `settings.
    transaction_cost_bps_overseas` overseas, by the `.KS` suffix) plus, on
    a domestic SELL only, the Korean Securities Transaction Tax
    (`settings.sell_tax_bps_domestic`) — see that setting's own docstring
    for the 2026-01-01 rate change and why it's kept separate from the
    commission constants."""
    is_domestic = ticker.endswith(".KS")
    bps = settings.transaction_cost_bps if is_domestic else settings.transaction_cost_bps_overseas
    if side == "sell" and is_domestic:
        bps += settings.sell_tax_bps_domestic
    return bps


def transaction_cost(ticker: str, side: str, notional: float) -> float:
    """`notional` scaled by `transaction_cost_bps_for` — the actual cash
    amount `app.execution.ledger.record_trade` deducts from `cash_after`
    (or, via `reverse_pending_trade`, refunds using the original trade's
    `side`, never the reversal's own — see that function's own comment on
    why using the wrong side there would silently corrupt cash)."""
    return notional * (transaction_cost_bps_for(ticker, side) / 10_000)
