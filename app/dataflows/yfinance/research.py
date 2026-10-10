"""yfinance-backed LLM-agent research providers — keyless, covers core price/
technical/fundamental/news/insider data. yfinance's client is synchronous, so
every call here runs in a worker thread via `_retry_to_thread` (a
retrying wrapper around `asyncio.to_thread`).

These are the router-registered providers: each `_yf_*` implementation is
wrapped by a public `get_*` function that calls `router.call(...)`, and
returns prose `str` for an LLM analyst prompt. See `app.dataflows.yfinance.shared`
for the point-in-time/`as_of` conventions shared across this package.
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import yfinance as yf

from app.dataflows.indicators import format_indicator_line
from app.dataflows.indicators import rsi as _rsi
from app.dataflows.long_horizon import WeeklyBar, format_long_horizon_block
from app.dataflows.router import router
from app.dataflows.session_state import format_verified_snapshot, live_session_footnote

from .shared import _HOLDINGS_LAG_DAYS, _filter_filing_lag, _filter_rows_by_date, _history, _retry_to_thread

# `_rsi` re-exported under its old name (2026-08-21: moved to
# app.dataflows.indicators, a vendor-neutral module both this file and
# app.dataflows.kis_provider now import from directly) — kept as an alias
# here rather than dropped so existing callers/tests importing `_rsi` from
# this module keep working unchanged.


async def _yf_stock_data(ticker: str, as_of: date | None = None) -> str:
    """2026-09-16: used to accept a `lookback_days` param (default 180) that
    only ever widened the underlying fetch — the rendered table was always
    `.tail(10)`, and the label quoted the unused 180 back at the model
    ("last 10 sessions of 180d lookback") as if it meant something. Dropped:
    nothing ever varied it, and multi-week context now has its own real
    source, `get_long_horizon_context` (see `long_horizon.py`), rather than
    a wider version of this 10-row table."""

    def _run() -> str:
        data = _history(ticker, period="180d", as_of=as_of)
        tail = data.tail(10)[["Open", "High", "Low", "Close", "Volume"]]
        # live_session_footnote is a no-op (empty string) once as_of is set or
        # the session has closed — see its own docstring — so this stays
        # byte-identical to before for every already-passing caller/test.
        return (
            f"Daily OHLCV for {ticker} (last 10 sessions):\n{tail.to_string()}"
            + live_session_footnote(ticker, data.index[-1].date())
        )

    return await _retry_to_thread(_run)


async def _yf_long_horizon(ticker: str, as_of: date | None = None) -> str:
    """52-week price context from yfinance weekly bars — see
    `long_horizon.py`'s module docstring. Resamples a daily pull to weekly
    (rather than requesting `interval="1wk"` directly) so the close-to-close
    math here matches `_kis_long_horizon`'s KIS-native weekly candles bar
    for bar; yfinance's own weekly resampling anchors on a different weekday
    boundary than KIS's."""

    def _run() -> str:
        data = _history(ticker, period="380d", as_of=as_of)
        # resample(...).agg(...) labels each bin by its RIGHT edge, so the
        # still-in-progress final week came out stamped with the UPCOMING
        # Friday — an as_of-bounded backtest for e.g. 2026-09-16 printed a
        # "week ending 2026-09-18" line, a future date that reads as
        # look-ahead even though the underlying prices are correctly bounded
        # by _history above. Aggregating the index itself with "last"
        # relabels every bar with the last real trading day it actually
        # contains, which is always <= as_of and matches what the KIS path
        # means by week_end (_kis_long_horizon uses KIS's own trading-date
        # field) — so both vendors now carry identical date semantics.
        data = data.assign(week_end=data.index)
        weekly = (
            data.resample("W-FRI")
            .agg({"High": "max", "Low": "min", "Close": "last", "week_end": "last"})
            .dropna()
        )
        bars = [
            WeeklyBar(row["week_end"].date(), float(row["High"]), float(row["Low"]), float(row["Close"]))
            for _, row in weekly.iterrows()
        ][-53:]
        return format_long_horizon_block(ticker, bars, source="yfinance")

    return await _retry_to_thread(_run)


async def _yf_indicators(ticker: str, as_of: date | None = None) -> str:
    # Deliberately NOT session-aware, unlike _yf_verified_snapshot/
    # _yf_stock_data above: (1) this output carries no per-row date labels
    # for a model to misread as "the day closed" — it's a single indicator
    # line, not a dated table or a snapshot with a `close=` field, so there's
    # no "concluded trading" surface for the MRVL-style mistake to land on;
    # (2) SMA/RSI/MACD reacting to the current, still-moving price during
    # market hours is the same behavior a real-time chart shows and is
    # correct, not a bug; (3) marking this "in progress" while
    # _yf_verified_snapshot's `last=` reflects the identical current price
    # would just create a NEW inconsistency (indicators frozen at yesterday's
    # close while the snapshot moves) instead of removing one.
    def _run() -> str:
        data = _history(ticker, period="6mo", as_of=as_of)
        return format_indicator_line(ticker, data["Close"])

    return await _retry_to_thread(_run)


async def _yf_verified_snapshot(ticker: str, as_of: date | None = None) -> str:
    """Deterministic ground-truth snapshot — analysts must call this before
    making any exact price claim, so they can't invent numbers.

    During live market hours the last row of a `period="5d"` pull is TODAY's
    still-forming bar, not a closed one — yfinance keeps updating it until
    the session ends. Mislabeling that as `close=` is exactly what let the
    Market Analyst describe a 09:42 ET (12 minutes after open) snapshot as
    "concluded trading" (see app.dataflows.session_state's module docstring
    for the full incident). `format_verified_snapshot` swaps in `last=`/
    `session_high=`/`session_low=`/`volume_so_far=` and an explicit
    still-open warning whenever that's the case; a genuinely closed session
    (including every backtest `as_of` call, which is always a past date)
    gets the same `close=`/`high=`/`low=`/`volume=` first line as before.

    The output is no longer byte-identical to the KIS side end-to-end,
    though: `format_verified_snapshot` appends a trailing `source: yfinance`
    line (2026-09-10 dataflows audit, stage 2/3), with an extra caveat when
    this is a KRX (`.KS`) ticker — this provider is the fallback that runs
    when KIS fails, and a bare, unmarked string gave a KIS hiccup no visible
    trace once yfinance's (occasionally ~2%-off, for KRX) number took its
    place under the same "authoritative" label."""

    def _run() -> str:
        data = _history(ticker, period="5d", as_of=as_of)
        last = data.iloc[-1]
        # The index is a tz-aware Timestamp in the exchange's own local
        # timezone, so `.date()` is already that market's local calendar
        # date — exactly what session_in_progress compares against.
        return format_verified_snapshot(
            ticker, data.index[-1].date(),
            close=float(last["Close"]), high=float(last["High"]),
            low=float(last["Low"]), volume=int(last["Volume"]),
            source="yfinance",
        )

    return await _retry_to_thread(_run)


async def _yf_fundamentals(ticker: str, as_of: date | None = None) -> str:
    """Deliberately omits yfinance's own `longName` — it comes from the same
    `get_info()` metadata blob as `sector`/`industry` and has shown real
    cross-symbol mismatches for KRX codes (086790.KS returned as both "KB금융"
    and "현대차제강" in the same week, neither of which is that code's actual
    company, 하나금융지주). The caller already receives a verified company
    name from `get_ticker_display_info`/KRX master file via `ticker_label` —
    re-surfacing a second, occasionally-wrong name here just gives the model
    a chance to trust the wrong one instead.

    Also deliberately omits `totalDebt`/`totalCash`: `get_info()` reports them
    most-recent-quarter with cash INCLUDING short-term investments, while
    `_yf_balance_sheet` reports Total Debt/Net Debt/cash at fiscal year-end
    with cash excluding them. Confirmed live (2026-08-25, ids 963-997) that
    the analyst quotes both sets in one paragraph and lands on a
    self-contradictory leverage read — e.g. INTC stated total cash $29.73B
    (info) and $14.27B (statement) alongside net debt $32.32B (statement);
    HOOD's info pair implies net CASH where the statement shows +$7.37B net
    debt. Every figure was a correct copy; only the mixing was wrong. One
    source, one period."""

    def _run() -> str:
        info = yf.Ticker(ticker).get_info()
        if as_of is not None:
            static_keys = ["sector", "industry"]
            lines = [f"{k}: {info.get(k)}" for k in static_keys if info.get(k) is not None]
            lines.append(
                "(Valuation ratios such as P/E, market cap, and dividend yield are "
                "derived from today's price and share count, not available "
                "point-in-time for a historical backtest date — omitted here.)"
            )
            return f"Fundamentals for {ticker} (static fields only, backtest mode):\n" + "\n".join(lines)

        keys = [
            "sector", "industry", "marketCap", "trailingPE", "forwardPE",
            "priceToBook", "profitMargins", "revenueGrowth", "returnOnEquity",
            "freeCashflow", "dividendYield",
        ]
        lines = [f"{k}: {info.get(k)}" for k in keys if info.get(k) is not None]
        lines.append(
            "(Debt and cash are not listed here — call get_balance_sheet for "
            "Total Debt, Net Debt and cash from one single reporting period.)"
        )
        return f"Fundamentals for {ticker}:\n" + "\n".join(lines)

    return await _retry_to_thread(_run)


def _scaled_unit_str(x: float) -> str:
    """K/M/B/T-suffixed value — the model reads this and copies it, rather
    than doing the K/M/B/T-equivalent conversion itself. T=trillion is
    exactly 1조 and B=billion is exactly 10억 in Korean, so this also
    happens to remove the Korean-unit conversion step, not just the digit
    count. NaN passes through unchanged (pandas' own to_string() renders it
    as "NaN"; nothing to scale)."""
    if pd.isna(x):
        return "NaN"
    magnitude = abs(x)
    if magnitude >= 1e12:
        return f"{x / 1e12:,.3f}T"
    if magnitude >= 1e9:
        return f"{x / 1e9:,.3f}B"
    if magnitude >= 1e6:
        return f"{x / 1e6:,.3f}M"
    if magnitude >= 1e3:
        return f"{x / 1e3:,.3f}K"
    return f"{x:,.0f}"


def _format_large_number_table(df: pd.DataFrame) -> str:
    """`DataFrame.to_string()`'s default float formatting switches to
    scientific notation for large values (e.g. `4.422051e+12`) — confirmed
    live (2026-07-30, 009540.KS reproduction) that the LLM reading this
    reliably mis-scales such values by 10x-100x when converting them to
    prose (a real cash-flow figure of ~4.4 trillion won got reported as
    ~44 trillion, then ~341 trillion, across repeated runs on the same
    ticker). A first fix (plain comma-separated integers) removed the
    scientific-notation step but NOT the underlying problem: the model
    still has to mentally divide a 12-13 digit integer by 1e9/1e12 to say
    "trillion"/"billion", and a live re-check (042660.KS, same day) showed
    it still gets that arithmetic wrong even on a cleanly comma-formatted
    number (reported 4,500억 for an actual ~456억 program-trade figure).
    Pre-dividing here removes that arithmetic from the model entirely — it
    reads "4.422T" and states "4.422조", a direct substitution instead of a
    calculation. Not asserting a specific currency (this table is shared
    across US and Korean tickers) — K/M/B/T is scale-only, same convention
    either market's own reporting already uses in English-language sources."""
    return (
        "(values scaled: K=thousand, M=million, B=billion, T=trillion — "
        "already converted, state them as-is rather than re-deriving)\n"
        + df.to_string(float_format=_scaled_unit_str)
    )


async def _yf_balance_sheet(ticker: str, as_of: date | None = None) -> str:
    def _run() -> str:
        bs = yf.Ticker(ticker).balance_sheet
        if bs.empty:
            raise ValueError(f"no balance sheet returned for {ticker!r}")
        if as_of is not None:
            bs = _filter_filing_lag(bs, as_of)
        return f"Balance sheet for {ticker} (most recent 2 periods):\n{_format_large_number_table(bs.iloc[:, :2])}"

    return await _retry_to_thread(_run)


async def _yf_cashflow(ticker: str, as_of: date | None = None) -> str:
    def _run() -> str:
        cf = yf.Ticker(ticker).cashflow
        if cf.empty:
            raise ValueError(f"no cashflow statement returned for {ticker!r}")
        if as_of is not None:
            cf = _filter_filing_lag(cf, as_of)
        return f"Cashflow statement for {ticker} (most recent 2 periods):\n{_format_large_number_table(cf.iloc[:, :2])}"

    return await _retry_to_thread(_run)


async def _yf_insider_txns(ticker: str, as_of: date | None = None) -> str:
    def _run() -> str:
        txns = yf.Ticker(ticker).insider_transactions
        if txns is None or txns.empty:
            return f"No recent insider transactions reported for {ticker}."
        if as_of is not None:
            start_dates = pd.to_datetime(txns["Start Date"], errors="coerce")
            txns = txns[start_dates <= pd.Timestamp(as_of)]
            if txns.empty:
                return f"No insider transactions on or before {as_of} for {ticker}."
        return f"Recent insider transactions for {ticker}:\n{txns.head(10).to_string()}"

    return await _retry_to_thread(_run)


async def _yf_analyst_consensus(ticker: str, as_of: date | None = None) -> str:
    """Wall Street's own coverage — a genuinely independent signal from our
    own analysts' reasoning, previously not gathered anywhere in this
    pipeline. `upgrades_downgrades` has real per-firm dates and is bounded by
    `as_of` like insider transactions; the recommendation trend and price
    targets are current-consensus snapshots with no historical archive
    (same limitation as `_yf_fundamentals`'s valuation ratios) and are
    dropped in backtest mode rather than silently applied to a past date."""

    def _run() -> str:
        t = yf.Ticker(ticker)
        parts = []

        ud = t.upgrades_downgrades
        if ud is not None and not ud.empty:
            if as_of is not None:
                ud = _filter_rows_by_date(ud, ud.index, as_of)
            if not ud.empty:
                cols = ["Firm", "ToGrade", "FromGrade", "Action", "currentPriceTarget"]
                cols = [c for c in cols if c in ud.columns]
                parts.append(f"Recent analyst actions for {ticker}:\n{ud[cols].head(10).to_string()}")

        if as_of is None:
            summary = t.recommendations_summary
            if summary is not None and not summary.empty:
                parts.append(f"Recommendation trend (Strong Buy/Buy/Hold/Sell/Strong Sell by month):\n{summary.to_string()}")
            targets = t.analyst_price_targets
            if targets:
                parts.append(
                    "Analyst price targets: "
                    f"current={targets.get('current')} mean={targets.get('mean')} "
                    f"median={targets.get('median')} low={targets.get('low')} high={targets.get('high')}"
                )
            info = t.get_info()
            n = info.get("numberOfAnalystOpinions")
            if n:
                parts.append(f"Based on {n} analyst opinions; average rating: {info.get('averageAnalystRating')}")
        else:
            parts.append(
                "(Current consensus rating and price-target snapshot aren't available point-in-time for a "
                "historical backtest date — omitted here; only dated upgrade/downgrade actions are included.)"
            )

        if not parts:
            return f"No analyst coverage data available for {ticker}."
        return "\n\n".join(parts)

    return await _retry_to_thread(_run)


async def _yf_institutional_ownership(ticker: str, as_of: date | None = None) -> str:
    """Who's actually holding the stock and how that's shifted — a "smart
    money positioning" signal distinct from anything technical/fundamental/
    news-based already gathered. Institutional 13F filings lag their
    reporting quarter by up to 45 days, so a backtest date only sees filings
    that would genuinely have been public by then."""

    def _run() -> str:
        t = yf.Ticker(ticker)
        holders = t.institutional_holders
        if holders is None or holders.empty:
            return f"No institutional ownership data available for {ticker}."

        if as_of is not None:
            cutoff = as_of - timedelta(days=_HOLDINGS_LAG_DAYS)
            holders = _filter_rows_by_date(holders, holders["Date Reported"], cutoff)
            if holders.empty:
                return f"No institutional ownership filings available as of {as_of} for {ticker}."

        lines = [f"Top institutional holders for {ticker}:\n{holders.head(10).to_string()}"]
        if as_of is None:
            info = t.get_info()
            pct_inst = info.get("heldPercentInstitutions")
            pct_insiders = info.get("heldPercentInsiders")
            if pct_inst is not None:
                lines.append(f"Held by institutions: {pct_inst:.1%}; held by insiders: {(pct_insiders or 0):.1%}")
        return "\n\n".join(lines)

    return await _retry_to_thread(_run)


async def _yf_short_interest(ticker: str, as_of: date | None = None) -> str:
    """Classic contrarian/squeeze-risk signal. yfinance only exposes the
    current snapshot (no historical archive), so — same convention as the
    valuation ratios in `_yf_fundamentals` — this is dropped entirely in
    backtest mode rather than misapplied to a past date."""
    if as_of is not None:
        return f"Short-interest data for {ticker} is a current-only snapshot, unavailable in backtest mode."

    def _run() -> str:
        info = yf.Ticker(ticker).get_info()
        shares_short = info.get("sharesShort")
        if shares_short is None:
            return f"No short-interest data available for {ticker}."
        prior = info.get("sharesShortPriorMonth")
        change = f" (prior month: {prior:,})" if prior else ""
        return (
            f"Short interest for {ticker}: {shares_short:,} shares short{change}, "
            f"short ratio (days to cover) {info.get('shortRatio')}, "
            f"{(info.get('shortPercentOfFloat') or 0):.2%} of float"
        )

    return await _retry_to_thread(_run)


async def _yf_earnings_estimates(ticker: str, as_of: date | None = None) -> str:
    """Track record (beat/miss history) plus forward consensus estimates —
    upcoming catalysts and how reliably this company has cleared the bar
    historically, neither of which anything else here captures. The history
    has real per-quarter dates and is bounded by `as_of`; forward estimates
    are a current-consensus snapshot like analyst price targets, so they're
    dropped in backtest mode."""

    def _run() -> str:
        t = yf.Ticker(ticker)
        parts = []

        history = t.earnings_dates
        if history is not None and not history.empty:
            if as_of is not None:
                history = _filter_rows_by_date(history, history.index, as_of)
            if not history.empty:
                parts.append(f"Earnings history (estimate vs. actual) for {ticker}:\n{history.head(8).to_string()}")

        if as_of is None:
            estimate = t.earnings_estimate
            if estimate is not None and not estimate.empty:
                parts.append(f"Forward earnings estimates by period for {ticker}:\n{estimate.to_string()}")

        if not parts:
            return f"No earnings estimate data available for {ticker}."
        return "\n\n".join(parts)

    return await _retry_to_thread(_run)


async def _yf_news(ticker: str, as_of: date | None = None) -> str:
    if as_of is not None:
        return (
            f"Ticker news for {ticker} is unavailable in point-in-time backtest mode "
            "(yfinance only exposes current headlines, not a historical archive)."
        )

    def _run() -> str:
        items = yf.Ticker(ticker).news or []
        if not items:
            return f"No recent news found for {ticker}."
        lines = []
        for item in items[:10]:
            content = item.get("content", item)
            title = content.get("title", "(untitled)")
            pub = content.get("pubDate", "")
            lines.append(f"- [{pub}] {title}")
        return f"Recent news for {ticker}:\n" + "\n".join(lines)

    return await _retry_to_thread(_run)


router.register("core_stock", [_yf_stock_data])
router.register("long_horizon", [_yf_long_horizon])
router.register("technical_indicators", [_yf_indicators])
router.register("verified_snapshot", [_yf_verified_snapshot])
router.register("fundamental_data", [_yf_fundamentals])
router.register("balance_sheet", [_yf_balance_sheet])
router.register("cashflow", [_yf_cashflow])
router.register("insider_data", [_yf_insider_txns], optional=True)
router.register("analyst_consensus", [_yf_analyst_consensus], optional=True)
router.register("institutional_ownership", [_yf_institutional_ownership], optional=True)
router.register("short_interest", [_yf_short_interest], optional=True)
router.register("earnings_estimates", [_yf_earnings_estimates], optional=True)
router.register("news_data", [_yf_news])


async def get_stock_data(ticker: str, as_of: date | None = None) -> str:
    return await router.call("core_stock", ticker=ticker, as_of=as_of)


async def get_long_horizon_context(ticker: str, as_of: date | None = None) -> str:
    return await router.call("long_horizon", ticker=ticker, as_of=as_of)


async def get_indicators(ticker: str, as_of: date | None = None) -> str:
    return await router.call("technical_indicators", ticker=ticker, as_of=as_of)


async def get_verified_snapshot(ticker: str, as_of: date | None = None) -> str:
    return await router.call("verified_snapshot", ticker=ticker, as_of=as_of)


async def get_fundamentals(ticker: str, as_of: date | None = None) -> str:
    return await router.call("fundamental_data", ticker=ticker, as_of=as_of)


async def get_balance_sheet(ticker: str, as_of: date | None = None) -> str:
    return await router.call("balance_sheet", ticker=ticker, as_of=as_of)


async def get_cashflow(ticker: str, as_of: date | None = None) -> str:
    return await router.call("cashflow", ticker=ticker, as_of=as_of)


async def get_insider_txns(ticker: str, as_of: date | None = None) -> str:
    return await router.call("insider_data", ticker=ticker, as_of=as_of)


async def get_analyst_consensus(ticker: str, as_of: date | None = None) -> str:
    return await router.call("analyst_consensus", ticker=ticker, as_of=as_of)


async def get_institutional_ownership(ticker: str, as_of: date | None = None) -> str:
    return await router.call("institutional_ownership", ticker=ticker, as_of=as_of)


async def get_short_interest(ticker: str, as_of: date | None = None) -> str:
    return await router.call("short_interest", ticker=ticker, as_of=as_of)


async def get_earnings_estimates(ticker: str, as_of: date | None = None) -> str:
    return await router.call("earnings_estimates", ticker=ticker, as_of=as_of)


async def get_news(ticker: str, as_of: date | None = None) -> str:
    return await router.call("news_data", ticker=ticker, as_of=as_of)
