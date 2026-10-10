"""yfinance-backed internal deterministic data feeds — used directly by
`execution/`, `live/`, `backtest/`, `chat/` for real numeric/structured data
(price series, live quotes, FX rates, display info). NOT routed through
`app.dataflows.router.router` — called directly by name, not `as_of`-bounded
unless individually noted.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone

import pandas as pd
import yfinance as yf

from app.dataflows.indicators import atr_pct as _shared_atr_pct
from app.dataflows.session_state import session_in_progress

from .shared import _CANDLE_CFG, _history, _retry_to_thread

logger = logging.getLogger(__name__)


async def get_ohlcv_candles(ticker: str, period_code: str = "D", as_of: date | None = None) -> list[dict]:
    """OHLCV candles for a ticker via yfinance — used for US names the KIS
    candle TR can't serve (dashboard candlestick chart). period_code D/W/M/Y
    maps to a yfinance interval and range (Y approximated as quarterly).
    Oldest first, point-in-time to `as_of`."""

    def _run() -> list[dict]:
        interval, rng = _CANDLE_CFG.get(period_code.upper(), _CANDLE_CFG["D"])
        df = yf.Ticker(ticker).history(period=rng, interval=interval)
        if df.empty:
            return []
        out = []
        for idx, r in df.iterrows():
            d = idx.date()
            if as_of is not None and d > as_of:
                continue
            out.append({
                "day": d.strftime("%Y%m%d"),
                "open": float(r["Open"]), "high": float(r["High"]),
                "low": float(r["Low"]), "close": float(r["Close"]),
                "volume": int(r["Volume"]) if not pd.isna(r["Volume"]) else 0,
            })
        return out

    return await _retry_to_thread(_run)


async def get_news_items(ticker: str, limit: int = 10) -> list[dict]:
    """Structured news for the live dashboard's "Recent News" panel —
    separate from `get_news`'s prose (that one's for an LLM prompt; this one
    needs real fields: thumbnail, source, publish time, link). Not routed
    through the vendor router or `as_of`-bounded: dashboard-only, live-only,
    same convention as `get_price_series`/`get_last_close`."""

    def _run() -> list[dict]:
        items = yf.Ticker(ticker).news or []
        out = []
        for item in items[:limit]:
            content = item.get("content", item)
            thumbnail = content.get("thumbnail") or {}
            resolutions = thumbnail.get("resolutions") or []
            link = (content.get("canonicalUrl") or {}).get("url") or (content.get("clickThroughUrl") or {}).get("url")
            out.append(
                {
                    "title": content.get("title", "(untitled)"),
                    "summary": content.get("summary", ""),
                    "publisher": (content.get("provider") or {}).get("displayName", ""),
                    "published_at": content.get("pubDate"),
                    "thumbnail_url": resolutions[-1]["url"] if resolutions else None,
                    "link": link,
                }
            )
        return out

    return await _retry_to_thread(_run)


def _series_to_points(row: pd.Series, label_fmt: str) -> list[dict]:
    """A financial-statement row (period columns, most-recent-first) into
    chart-ready {label, value} points, oldest first — yfinance returns these
    newest-column-first, but a trend chart reads left-to-right as time
    moving forward."""
    points = [
        {"label": col.strftime(label_fmt), "value": None if pd.isna(val) else float(val)}
        for col, val in row.items()
    ]
    return list(reversed(points))


async def get_earnings_trend(ticker: str) -> dict:
    """Everything behind the live dashboard's "Earnings Trends" section:
    the EPS beat/miss history, quarterly and annual revenue-vs-earnings, and
    both forward estimate tables. Live-only (current consensus/estimates
    have no historical archive — see `_yf_earnings_estimates`), so this
    isn't `as_of`-bounded or vendor-routed, same as `get_news_items`."""

    def _run() -> dict:
        t = yf.Ticker(ticker)

        eps_history = []
        history = t.earnings_history
        if history is not None and not history.empty:
            for quarter, row in history.iterrows():
                estimate, actual = row.get("epsEstimate"), row.get("epsActual")
                eps_history.append(
                    {
                        "quarter": pd.Timestamp(quarter).strftime("%b %Y"),
                        "estimate": None if pd.isna(estimate) else float(estimate),
                        "actual": None if pd.isna(actual) else float(actual),
                        "surprise_pct": None if pd.isna(row.get("surprisePercent")) else float(row["surprisePercent"]),
                    }
                )

        def _revenue_earnings(stmt: pd.DataFrame) -> list[dict]:
            if stmt is None or stmt.empty or "Total Revenue" not in stmt.index:
                return []
            revenue = _series_to_points(stmt.loc["Total Revenue"], "%b %Y")
            earnings = _series_to_points(stmt.loc["Net Income"], "%b %Y") if "Net Income" in stmt.index else []
            earnings_by_label = {p["label"]: p["value"] for p in earnings}
            return [
                {"label": p["label"], "revenue": p["value"], "earnings": earnings_by_label.get(p["label"])}
                for p in revenue
                if p["value"] is not None or earnings_by_label.get(p["label"]) is not None
            ]

        def _estimate_table(est: pd.DataFrame, actual_key: str) -> dict:
            if est is None or est.empty:
                return {}
            return {
                str(period): {
                    "avg": None if pd.isna(row.get("avg")) else float(row["avg"]),
                    "low": None if pd.isna(row.get("low")) else float(row["low"]),
                    "high": None if pd.isna(row.get("high")) else float(row["high"]),
                    "year_ago": None if pd.isna(row.get(actual_key)) else float(row[actual_key]),
                    "growth": None if pd.isna(row.get("growth")) else float(row["growth"]),
                    "num_analysts": None if pd.isna(row.get("numberOfAnalysts")) else int(row["numberOfAnalysts"]),
                }
                for period, row in est.iterrows()
            }

        return {
            "eps_history": eps_history,
            "revenue_earnings_quarterly": _revenue_earnings(t.quarterly_income_stmt),
            "revenue_earnings_annual": _revenue_earnings(t.income_stmt),
            "revenue_estimate": _estimate_table(t.revenue_estimate, "yearAgoRevenue"),
            "earnings_estimate": _estimate_table(t.earnings_estimate, "yearAgoEps"),
        }

    return await _retry_to_thread(_run)


_UNKNOWN_TICKER_CACHE: dict[str, float] = {}
_UNKNOWN_TICKER_TTL_SECONDS = 600.0  # 10 minutes


async def get_price_series(ticker: str, start: date, end: date) -> dict[date, float]:
    """Real close prices for every trading day in [start, end] — used by the
    backtest harness to pick actual trading dates and compute returns. Not
    routed through the vendor router: this is deterministic historical data
    for the harness itself, not an analyst-facing tool call.

    2026-09-02: a recently-listed/renamed ticker yfinance has never heard of
    (its `history()` call logs "possibly delisted; no timezone found" and
    returns empty, regardless of the date range asked for) was retried the
    full 3 attempts of `_retry_to_thread`'s exponential backoff on EVERY
    call — during a 12-month screening_sweep A/B run (49 trade dates), that
    meant the same handful of bad tickers burned ~1.5s of backoff sleep
    ×3-attempts ×49 dates each, and printed a full traceback every time,
    drowning out the sweep's own progress log. `_UNKNOWN_TICKER_CACHE`
    remembers a ticker that has already exhausted every retry with this
    exact "no price history at all" failure and skips straight to raising
    for `_UNKNOWN_TICKER_TTL_SECONDS` — short enough that a genuinely
    transient/newly-listed-then-backfilled ticker recovers within one
    sweep's typical runtime, not a permanent blacklist. Only THIS failure
    mode is cached: a network blip or rate-limit inside `_run()` still gets
    yfinance's own retry, since those are exactly the transient failures
    `_retry_to_thread` exists to absorb — caching those too would risk
    treating a live rate-limit as "this ticker doesn't exist."""
    now = time.monotonic()
    cached_at = _UNKNOWN_TICKER_CACHE.get(ticker)
    if cached_at is not None and now - cached_at < _UNKNOWN_TICKER_TTL_SECONDS:
        raise ValueError(f"no price history returned for {ticker!r} (cached failure, not retried)")

    def _run() -> dict[date, float]:
        data = yf.Ticker(ticker).history(start=start.isoformat(), end=(end + timedelta(days=1)).isoformat())
        if data.empty:
            raise ValueError(f"no price history returned for {ticker!r} in [{start}, {end}]")
        return {idx.date(): float(row["Close"]) for idx, row in data.iterrows()}

    try:
        return await _retry_to_thread(_run)
    except ValueError:
        _UNKNOWN_TICKER_CACHE[ticker] = now
        raise


async def get_last_close(ticker: str, as_of: date | None = None) -> float:
    """Numeric last close, bounded by `as_of` — used by the execution layer
    for trade fills. Not routed through the vendor router (deterministic
    data for internal use, not an analyst-facing tool call), same convention
    as `get_price_series`/`get_daily_low_series`."""

    def _run() -> float:
        data = _history(ticker, period="5d", as_of=as_of)
        return float(data.iloc[-1]["Close"])

    return await _retry_to_thread(_run)

def _atr(data: pd.DataFrame, window: int = 14, *, denominator_close: float | None = None) -> float | None:
    """Thin wrapper around `app.dataflows.indicators.atr_pct` (2026-08-21,
    ATR TA-Lib PoC, commit 3/3 — see qlib-poc-decision-2026-08-21 project
    memory for why: a rolling-mean ATR can drop off a cliff when a
    volatility spike ages out of its window with no new market
    information, a real measured defect this Wilder-smoothed replacement
    doesn't have). Kept here, taking a DataFrame rather than three Series,
    only for backward compatibility with this module's own `get_atr_pct`
    and the `app.dataflows.yfinance` package's public `_atr` re-export —
    new callers should use `indicators.atr_pct` directly.

    `denominator_close` (2026-09-11, still-forming-bar ATR fix — see
    `get_atr_pct` below and `app.dataflows.kis_provider.get_atr_pct`'s own
    comment for the full rationale): keyword-only, defaults to `None`
    (unchanged behaviour — `indicators.atr_pct` divides by `data["Close"]`'s
    own last row). Forwarded straight through to `indicators.atr_pct`."""
    return _shared_atr_pct(data["High"], data["Low"], data["Close"], window=window, denominator_close=denominator_close)


async def get_atr_pct(ticker: str, as_of: date | None = None) -> float | None:
    """Return the current ATR (14-day) as a percentage of the latest closing price,
    bounded by `as_of` for backtesting. Used for volatility-adjusted position sizing.
    `None` if there isn't enough history yet — see `_atr`'s own docstring.

    2026-09-11: during a live session yfinance's last row (like KIS's) is
    TODAY'S STILL-FORMING bar — see `app.dataflows.kis_provider.get_atr_pct`'s
    comment for the full measured defect and why the fix is "drop the
    in-progress bar from the ATR itself but keep the live price as the
    denominator" rather than falling back to the previous close. `as_of`
    unaffected: `session_in_progress` is always False for a past date, so a
    backtest's historical `as_of` takes the unchanged path below."""
    def _run() -> float | None:
        # "6mo" (~125 rows), not the narrower "30d" (pre-2026-08-21): matches
        # kis_provider.get_atr_pct's own widened window (see that function's
        # comment) so both paths converge to close to the same value on the
        # same ticker/day once talib.ATR's Wilder smoothing replaces the SMA
        # here — a too-short window disagrees with a longer one by several
        # percent for Wilder, unlike the current SMA (window-length-invariant
        # by construction, so this widening alone changes nothing yet).
        data = _history(ticker, period="6mo", as_of=as_of)
        if not data.empty and session_in_progress(ticker, data.index[-1].date()):
            live_price = float(data.iloc[-1]["Close"])
            trimmed = data.iloc[:-1]
            return _atr(trimmed, window=14, denominator_close=live_price)
        return _atr(data, window=14)
    return await _retry_to_thread(_run)


@dataclass
class LiveQuote:
    price: float
    volume: int
    timestamp: datetime


async def get_live_quote(ticker: str) -> LiveQuote:
    """Latest available *completed* minute bar — used by the live poller
    (app.live.poller) for the dashboard feed and alert thresholds.

    Deliberately uses the second-to-last row, not the last: yfinance's most
    recent intraday bar is still accumulating volume until that minute
    closes, so reading it directly reports volume=0 far more often than
    chance alone would suggest — confirmed live against 005930.KS, where a
    poll caught a bar mid-formation at volume=0 and the same bar had a real
    value (93008) just 20 seconds later. Since the poll interval matches the
    1-minute bar width, this isn't rare bad luck, it's a systematic timing
    collision. One bar of extra staleness (at most ~60s) buys a stable, real
    volume figure every time instead of racing the close. Not routed through
    the vendor router (deterministic internal use, no LLM ever sees this
    call), same convention as `get_price_series`/`get_daily_low_series`/
    `get_last_close`."""

    def _run() -> LiveQuote:
        data = yf.Ticker(ticker).history(period="1d", interval="1m")
        if data.empty:
            raise ValueError(f"no intraday data returned for {ticker!r}")
        row_index = -2 if len(data) >= 2 else -1
        row = data.iloc[row_index]
        return LiveQuote(
            price=float(row["Close"]),
            volume=int(row["Volume"]),
            timestamp=data.index[row_index].to_pydatetime(),
        )

    return await _retry_to_thread(_run)


async def get_intraday_series(ticker: str) -> list[LiveQuote]:
    """The full trading day's 1-minute bars so far, not just the latest
    one — the same underlying `history(period="1d", interval="1m")` call
    `get_live_quote` makes already returns the whole session (e.g. all ~390
    bars for a US ticker from market open), it's just discarded down to one
    row there. Used to backfill the live dashboard's chart at startup so it
    shows the complete session immediately instead of only whatever's been
    polled since the process started — Yahoo has been tracking the full day
    regardless of whether our own poller was running. Excludes the final row
    for the same reason `get_live_quote` reads the second-to-last one: the
    most recent bar is often still accumulating volume until that minute
    closes."""

    def _run() -> list[LiveQuote]:
        data = yf.Ticker(ticker).history(period="1d", interval="1m")
        if data.empty:
            return []
        rows = data.iloc[:-1] if len(data) >= 2 else data
        return [
            LiveQuote(price=float(row["Close"]), volume=int(row["Volume"]), timestamp=idx.to_pydatetime())
            for idx, row in rows.iterrows()
        ]

    return await _retry_to_thread(_run)


# --- 5-minute intraday bars, for the trailing-stop backstop's ordering fix ---
#
# 2026-08-27. app.execution.ledger.check_and_apply_trailing_stop's live path
# needs to know, for a day it is walking, whether that day's LOW happened
# before or after the HIGH that tightened the trailing floor. Daily bars
# cannot answer that, and answering it wrong force-liquidated 047040.KS
# (대우건설) at +11% on a low that occurred 30 minutes BEFORE the high whose
# floor it supposedly breached — see that function's own docstring for the
# full incident.
#
# yfinance, not KIS, deliberately: KIS's past-date intraday TR
# (FHKST03010230, app.backtest.minute_candles) is DOMESTIC-ONLY — there is no
# overseas equivalent in this codebase, so a KIS-based fix would leave every
# US position on the broken rule — and it costs 4-6 paged calls per
# (ticker, day) under a global throttle that is 1 request/second on the paper
# account, with gather-batching explicitly not an option here (project
# memory: market-lock-hold-time-investigation). yfinance costs no KIS budget
# at all, serves KRX and US through one call, and is the exact feed that
# verified the 047040.KS incident — its 5-minute low for that session matched
# the daily-bar low the false exit cited to the won (18,760.00), which is the
# empirical basis for the 0.5% reconciliation tolerance the ledger applies to
# these bars.
_INTRADAY_BAR_INTERVAL = "5m"
_INTRADAY_BAR_MINUTES = 5
# A session whose last bar is already older than this is over and immutable;
# anything fresher (today's still-running session, or an empty/failed fetch)
# is only briefly cached so the next hourly backstop pass sees new bars.
_INTRADAY_CACHE_TTL_SECONDS = 3600.0
_INTRADAY_CACHE_SHORT_TTL_SECONDS = 300.0
_INTRADAY_CACHE_MAX_ENTRIES = 512
_intraday_cache: dict[tuple[str, date], tuple[float, list["IntradayBar"]]] = {}


@dataclass
class IntradayBar:
    """One intraday bar, reduced to what a trailing-stop ratchet needs: the
    bar's own high/low and where it sits in the session. `start`/`end` are
    always timezone-aware UTC (yfinance returns an exchange-local tz-aware
    index; a naive index means the day cannot be aligned to a position's
    entry timestamp at all and the whole day is discarded rather than
    guessed at — see get_intraday_bars)."""

    start: datetime
    end: datetime
    high: float
    low: float


async def get_intraday_bars(ticker: str, day: date) -> list[IntradayBar]:
    """`day`'s regular-session 5-minute bars, oldest first — [] when
    unavailable for ANY reason (empty download, rate limit, a naive index, a
    date outside yfinance's ~60-day intraday retention). Never raises: every
    caller's correct response to "no intraday data" is to fall back to a
    coarser rule, not to abort a safety check, so a vendor failure is logged
    and returned as absence.

    `prepost=False`: only the regular session, matching what both the daily
    bar and app.live.exit_monitor's own market-open gate describe.

    Cached in-process per (ticker, day) because the hourly backstop re-walks
    the same closed days every pass for every ticker — a closed session's
    bars are immutable, so re-downloading them is pure yfinance rate-limit
    pressure (HTTP 429 is the documented reason _retry_to_thread exists).
    Today's still-running session, and any empty result, get the short TTL
    instead so a later pass genuinely re-reads them."""
    now = time.monotonic()
    key = (ticker, day)
    cached = _intraday_cache.get(key)
    if cached is not None and now < cached[0]:
        return cached[1]

    def _run() -> list[IntradayBar]:
        data = yf.Ticker(ticker).history(
            start=day.isoformat(),
            end=(day + timedelta(days=1)).isoformat(),
            interval=_INTRADAY_BAR_INTERVAL,
            prepost=False,
        )
        if data.empty:
            return []
        bars: list[IntradayBar] = []
        for idx, row in data.iterrows():
            start = idx.to_pydatetime()
            if start.tzinfo is None:
                return []  # cannot be aligned to an entry timestamp — discard the day
            start = start.astimezone(timezone.utc)
            high, low = row["High"], row["Low"]
            if pd.isna(high) or pd.isna(low):
                continue
            bars.append(
                IntradayBar(
                    start=start,
                    end=start + timedelta(minutes=_INTRADAY_BAR_MINUTES),
                    high=float(high),
                    low=float(low),
                )
            )
        bars.sort(key=lambda b: b.start)
        return bars

    try:
        bars = await _retry_to_thread(_run)
    except Exception:
        logger.warning(
            "intraday 5m bars unavailable for %s on %s — any caller adjudicating that "
            "session's high/low ordering must fall back to its coarser daily-bar rule",
            ticker, day, exc_info=True,
        )
        bars = []

    session_over = bool(bars) and (
        datetime.now(timezone.utc) - bars[-1].end
    ).total_seconds() > _INTRADAY_CACHE_TTL_SECONDS
    ttl = _INTRADAY_CACHE_TTL_SECONDS if session_over else _INTRADAY_CACHE_SHORT_TTL_SECONDS
    while len(_intraday_cache) >= _INTRADAY_CACHE_MAX_ENTRIES:
        _intraday_cache.pop(next(iter(_intraday_cache)))
    _intraday_cache[key] = (now + ttl, bars)
    return bars


_currency_cache: dict[str, str] = {}


def _is_krx_numeric(ticker: str) -> bool:
    """KRX short-circuit for `get_ticker_currency`. Same .KS-suffix +
    all-digit-code shape as app.dataflows.kis_provider._is_krx — duplicated
    here (not imported) because kis_provider imports from this package
    (app.dataflows.yfinance), so importing back would be circular. Same
    convention as app.config.Market and app.dataflows.session_state's own
    duplicate of this rule.
    KRX 판별 규칙(.KS/.KQ 접미사 + 숫자코드)을 순환 import 회피를 위해 로컬 중복 정의.

    Accepts both .KS and .KQ: this repo uses .KS for KOSDAQ tickers too and
    only sees .KQ defensively (identity_gate.py, kis_provider.py), but a
    numeric-code .KQ ticker is unambiguously KRW-quoted, so there's no
    downside to covering it here as well."""
    for suffix in (".KS", ".KQ"):
        if ticker.endswith(suffix) and ticker[: -len(suffix)].isdigit():
            return True
    return False


async def get_ticker_currency(ticker: str) -> str:
    """Quote currency for a ticker (e.g. "USD", "KRW") — used by the
    execution ledger to size starting cash in the ticker's own currency
    instead of a flat USD figure, which silently floors to 0 shares for any
    ticker priced in a currency with much larger nominal share prices (e.g.
    Samsung at ~KRW 200,000+/share against a $100,000 cash figure). Not
    routed through the vendor router (deterministic internal use, no LLM
    ever sees this call), same convention as `get_price_series`/
    `get_daily_low_series`/`get_last_close`.

    KRX tickers short-circuit to "KRW" before any network call, without
    touching the cache below. yfinance's `fast_info.currency` is
    documented-unreliable for `.KS`/`.KQ` codes in this repo — it's the same
    distrust `kis_provider.get_ticker_display_info` already encodes by
    overriding `currency="KRW"` for KRX on the display path; this makes the
    sizing path (this function) consistent with that instead of trusting
    yfinance here alone. KRX 종목은 yfinance 신뢰 불가로 네트워크 호출 없이 KRW 확정.

    For every other ticker, the yfinance result is validated before caching:
    a currency code must be exactly three ASCII uppercase letters, or this
    raises `ValueError` and leaves the cache untouched for that ticker.
    `ledger.py`'s `starting_cash_for` already wraps this call in a bare
    `try/except Exception` and degrades to the raw USD cash figure, so a
    raised ValueError there is unchanged behavior. `portfolio_risk.py`'s two
    call sites are unguarded, but an unvalidated bad value (e.g. None) would
    already have propagated one step further and failed inside
    `get_usd_fx_rate` (`yf.Ticker("None=X").history()` -> empty -> its own
    ValueError) — so raising here is not a new failure mode, only an
    earlier, clearer version of the same one. Not caching a bad read matters
    because before this fix a single bad read was pinned in `_currency_cache`
    for the whole process lifetime (days, in the live daemon).
    검증되지 않은 값은 캐시하지 않음 — 한 번의 오독이 프로세스 수명 내내 고정되는 것을 방지.

    Cached per ticker with NO TTL (deliberately — see below): a ticker's
    quote currency is time-invariant, so `app.execution.portfolio_risk.
    portfolio_snapshot` (which re-values every trade allowlist ticker on
    every call, both live and in the multi-ticker backtest harness) doesn't
    need to re-fetch it from yfinance each time — same rationale as
    `pipeline.py`'s `_company_name_cache`. The original hazard here was never
    staleness (currency doesn't drift), it was caching a *wrong* value — the
    KRX short-circuit and the validation above now prevent that at the
    source, so a TTL would only add repeated network calls to
    `portfolio_snapshot`'s per-ticker loop for zero correctness gain. Do not
    add one back.
    TTL 미적용은 의도적: 통화는 시간불변이므로 문제는 캐시 만료가 아니라 잘못된 값의 캐싱이었고,
    위 KRX 단락회로/검증으로 이미 해결됨."""
    if _is_krx_numeric(ticker):
        return "KRW"
    if ticker not in _currency_cache:
        currency = await _retry_to_thread(lambda: yf.Ticker(ticker).fast_info.currency)
        if not (isinstance(currency, str) and len(currency) == 3 and currency.isascii() and currency.isupper() and currency.isalpha()):
            raise ValueError(f"get_ticker_currency: implausible currency {currency!r} for {ticker}")
        _currency_cache[ticker] = currency
    return _currency_cache[ticker]


# (rate, cached_at) per currency. A backtest run is short-lived enough that
# "cached for the lifetime of the process" was originally fine — but this
# same cache backs the long-running live daemon too (dashboard's KRW/USD
# toggle, ledger currency conversion), where "process lifetime" can be days,
# and FX rates move daily. TTL keeps it fresh there without losing the
# per-run reuse that made this a behavior-preserving change for backtests.
# 장기 실행되는 daemon에서 환율이 프로세스 수명 내내 고정되지 않도록 TTL 적용.
_FX_RATE_TTL_SECONDS = 60 * 60  # 1 hour — FX doesn't need finer freshness here
_fx_rate_cache: dict[str, tuple[float, float]] = {}


async def get_usd_fx_rate(currency: str) -> float:
    """Units of `currency` per 1 USD (e.g. "KRW" -> ~1500.0). Returns 1.0
    for USD itself with no network call. Not routed through the vendor
    router (deterministic internal use), same convention as
    `get_price_series`/`get_daily_low_series`/`get_last_close`.

    Cached per currency for up to `_FX_RATE_TTL_SECONDS`: this call already
    ignores `as_of` and always fetches today's rate regardless of which
    historical date a backtest is asking about (a pre-existing point-in-time
    gap, not something this cache changes) — so within that window it was
    already returning the same value on every call. Caching it is therefore
    behavior-preserving for a single short-lived run, and removes real
    repeated network calls from `portfolio_snapshot`'s per-ticker loop,
    while still refreshing periodically for the long-running live daemon."""
    if currency == "USD":
        return 1.0
    cached = _fx_rate_cache.get(currency)
    if cached is None or time.monotonic() - cached[1] > _FX_RATE_TTL_SECONDS:

        def _run() -> float:
            data = yf.Ticker(f"{currency}=X").history(period="5d")
            if data.empty:
                raise ValueError(f"no FX data returned for USD/{currency}")
            return float(data.iloc[-1]["Close"])

        rate = await _retry_to_thread(_run)
        _fx_rate_cache[currency] = (rate, time.monotonic())
    return _fx_rate_cache[currency][0]


@dataclass
class TickerDisplayInfo:
    symbol: str
    name: str
    previous_close: float
    currency: str
    # yfinance's own instrument classification ("EQUITY", "ETF", "MUTUALFUND",
    # ...) — "" when unknown (e.g. the KIS-override construction path in
    # app.dataflows.kis_provider, which only overrides previous_close/currency
    # and has no reason to re-derive this). See app.domain.leverage_filter's
    # sibling use of this same TickerDisplayInfo for why a non-stock/ETF
    # product matters to a caller deciding what's tradable.
    quote_type: str = ""
    # yfinance's `shortName` — deliberately separate from `name` (which is
    # `longName or shortName`), not a fallback folded into it. 2026-08-28
    # domestic AI audit: a report correctly abbreviated 298040.KS's real name
    # to "Hyosung Heavy" — matched by `identity_gate`'s prefix-alias handling
    # of `longName` ("Hyosung Heavy Industries Corporation") — but a report
    # abbreviating some OTHER ticker's longName in a way `longName` itself
    # doesn't prefix-match can still land on `shortName` verbatim (yfinance's
    # own short form, e.g. "HYOSUNG HEAVY"). Feeding both into
    # `app.orchestrator.pipeline._company_identity`'s weak-alias set gives
    # `identity_gate` two independent chances instead of one. "" when
    # `_looks_like_a_real_name` rejected it (same guard `name` already gets).
    short_name: str = ""


# yfinance's `longName`/`shortName` occasionally comes back as raw, clearly
# non-name data instead of an actual company/fund name — real incidents:
# 218150.KS (yfinance quoteType "MUTUALFUND") returned shortName
# "218150.KS,0P000168Z2,87763", and 076610.KS returned
# "076610.KS,0P0000ZWGU,188260" — both a ticker+internal-id+price string
# glued together on YAHOO'S OWN side (confirmed via direct yfinance queries,
# not anything this codebase constructs) — which then became the card title
# AND fed into the AI's prompt, self-flagged by the model as a data-integrity
# concern ("종목명 혼재"). This is a narrow, low-false-positive check for
# THAT specific glued-together shape, not a general name validator.
#
# A plain "no comma at all" version of this check (used until 2026-08-04)
# was too blunt: plenty of real legal names have exactly one comma before a
# corporate suffix (confirmed live — AXTI's real name "AXT, Inc." got
# discarded by it, leaving the ticker ungrounded and the model guessing a
# wrong company ("Axiomtek") from the bare symbol alone — the exact failure
# mode `IDENTITY_GROUNDING_NOTE` exists to prevent). Both real garbled
# incidents above have TWO commas (ticker, internal-id, price — three
# segments); a single comma is indistinguishable from an ordinary "X, Inc."
# name, so only reject 2+.
_MAX_PLAUSIBLE_NAME_LENGTH = 80


def _looks_like_a_real_name(name: str, ticker: str) -> bool:
    return name.count(",") <= 1 and len(name) <= _MAX_PLAUSIBLE_NAME_LENGTH and name != ticker


# Tickers already warned about via _looks_like_a_real_name — yfinance's
# implausible-name data for a given ticker doesn't change between calls, so
# repeating the same warning every time this ticker is looked up again (every
# auto-trade candidate-scan cycle that reconsiders it, every dashboard
# restart) is pure log noise once the first occurrence has been seen. Capped
# implicitly by the ticker universe (thousands at most, not unbounded).
_warned_implausible_name_tickers: set[str] = set()


async def get_ticker_display_info(ticker: str) -> TickerDisplayInfo:
    """Company name + previous close — fetched once at dashboard startup
    (not per poll) so the live view can show "Apple Inc. +0.8%" instead of
    a bare ticker symbol with no reference point. Not routed through the
    vendor router (deterministic internal use, no LLM ever sees this call),
    same convention as `get_price_series`/`get_daily_low_series`/
    `get_last_close`."""

    def _run() -> TickerDisplayInfo:
        t = yf.Ticker(ticker)
        info = t.get_info()
        fi = t.fast_info
        name = info.get("longName") or info.get("shortName") or ticker
        quote_type = info.get("quoteType") or ""
        raw_short_name = info.get("shortName") or ""
        # Checked independently of `name`'s own plausibility below — even a
        # plausible `longName` doesn't guarantee `shortName` is equally
        # sane, and vice versa (see TickerDisplayInfo.short_name's own
        # docstring for why this is worth carrying at all).
        short_name_implausible = bool(raw_short_name) and not _looks_like_a_real_name(raw_short_name, ticker)
        if not _looks_like_a_real_name(name, ticker):
            if ticker not in _warned_implausible_name_tickers:
                _warned_implausible_name_tickers.add(ticker)
                logger.warning(
                    "get_ticker_display_info: %s returned an implausible name %r — using the bare ticker instead",
                    ticker, name,
                )
            name = ticker
            # Same `info` dict the garbled name came from — quote_type is
            # just as untrustworthy here, not just name. Real incident
            # (2026-08-03, 058610.KS — a real KOSDAQ stock, not a fund):
            # this same garbled-response pattern carried quoteType
            # "MUTUALFUND", which fed straight into
            # app.domain.leverage_filter.is_unsupported_product_type and
            # blocked a legitimate BUY from ever reaching the trade
            # allowlist. Blanking it out here routes it through that
            # function's own "empty/unknown quote_type never matches"
            # convention instead of trusting a corrupted value.
            quote_type = ""
        elif short_name_implausible:
            # longName itself is fine here (the branch above didn't fire),
            # but shortName carries the same garbled `ticker,internal-id,
            # price` tell yfinance pairs with a corrupted quoteType even
            # when longName is untouched — confirmed live 2026-09-03
            # (228340.KS "Tongyang Pile Inc" / 142760.KS "BL Corp", both
            # real KOSDAQ stocks, both quoteType "MUTUALFUND", both with a
            # garbled shortName like "228340.KS,0P00017R1H,41744"). Only
            # quote_type is untrusted here — name is already good.
            if ticker not in _warned_implausible_name_tickers:
                _warned_implausible_name_tickers.add(ticker)
                logger.warning(
                    "get_ticker_display_info: %s has an implausible shortName %r alongside quoteType %r"
                    " — blanking quote_type",
                    ticker, raw_short_name, quote_type,
                )
            quote_type = ""
        short_name = "" if short_name_implausible else raw_short_name
        return TickerDisplayInfo(
            symbol=ticker,
            name=name,
            previous_close=float(fi["previousClose"]),
            currency=fi["currency"],
            quote_type=quote_type,
            short_name=short_name,
        )

    return await _retry_to_thread(_run)


async def get_daily_low_series(ticker: str, start: date, end: date) -> dict[date, float]:
    """Real daily lows for every trading day in [start, end] — used by the
    execution layer to check a stop-loss against actual intraday breaches
    between sampled backtest decision dates, not just close-to-close."""

    def _run() -> dict[date, float]:
        data = yf.Ticker(ticker).history(start=start.isoformat(), end=(end + timedelta(days=1)).isoformat())
        if data.empty:
            raise ValueError(f"no price history returned for {ticker!r} in [{start}, {end}]")
        return {idx.date(): float(row["Low"]) for idx, row in data.iterrows()}

    return await _retry_to_thread(_run)


async def get_daily_high_series(ticker: str, start: date, end: date) -> dict[date, float]:
    """Real daily highs for every trading day in [start, end] — the mirror
    of `get_daily_low_series`, for replaying a trailing stop's high-water
    mark through backtest history (a trailing stop needs the ACTUAL
    intraday peak between sampled decision dates, not just each day's
    close, same reasoning as the low-side stop-loss check)."""

    def _run() -> dict[date, float]:
        data = yf.Ticker(ticker).history(start=start.isoformat(), end=(end + timedelta(days=1)).isoformat())
        if data.empty:
            raise ValueError(f"no price history returned for {ticker!r} in [{start}, {end}]")
        return {idx.date(): float(row["High"]) for idx, row in data.iterrows()}

    return await _retry_to_thread(_run)


async def get_daily_open_series(ticker: str, start: date, end: date) -> dict[date, float]:
    """Real daily opens for every trading day in [start, end] — used to
    determine the realizable fill for a stop order when the day's open
    already gaps through the stop level (a market-on-touch stop cannot fill
    at its own level if the market opened below it)."""

    def _run() -> dict[date, float]:
        data = yf.Ticker(ticker).history(start=start.isoformat(), end=(end + timedelta(days=1)).isoformat())
        if data.empty:
            raise ValueError(f"no price history returned for {ticker!r} in [{start}, {end}]")
        return {idx.date(): float(row["Open"]) for idx, row in data.iterrows()}

    return await _retry_to_thread(_run)
