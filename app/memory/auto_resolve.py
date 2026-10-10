"""Automatic resolution of pending LIVE memory-entry decisions.

The backtest harness resolves its own pending entries automatically as it
walks forward through known future trade dates (see app/backtest/harness.py).
A live decision has no "next date" to trigger that the same way, so this
existed only as a manual, per-ticker step (`python -m app.memory resolve
--ticker AAPL`, see app/memory/__main__.py) — and per project history there's
no evidence that CLI was ever actually run in practice, meaning the
reflection feedback loop (learning from past decisions) was likely never
running live at all. `resolve_all_due` below closes that gap: it's called
once per app.live.auto_trade_loop cycle and walks every ticker with a
pending decision old enough to resolve, no manual trigger required.

`resolve_ticker` (the per-ticker resolution logic) lives here rather than in
__main__.py so both the CLI and the live scheduler share one implementation
instead of the CLI's logic silently drifting from what actually runs live.
"""

from __future__ import annotations

import logging
import math
from datetime import date, timedelta

from app.clock import local_today
from app.config import current_account_mode, settings
from app.dataflows.session_state import session_in_progress
from app.dataflows.yfinance import get_price_series
from app.execution.portfolio_risk import benchmark_for_market, ticker_market
from app.memory.db import get_pool
from app.memory.reflection import reflect
from app.memory.store import resolve_outcome, set_horizon_outcome, tickers_with_pending_resolution
from app.user_context import get_current_user_id

logger = logging.getLogger(__name__)


def _usable(value: float | None) -> bool:
    """A close price this module is willing to compute a return from.

    `math.isfinite` is the load-bearing part: yfinance can return a literal
    NaN close for a thinly-traded KRX name on a day it does carry a bar
    (confirmed 065420.KS, 2026-08-13), and the previous `== 0` guard let that
    straight through — `(nan - entry) / entry` is NaN, which then got STORED
    as the decision's raw_return/alpha_return. That is worse than it sounds,
    because Postgres sorts NaN as greater than every real number, so
    `alpha_return > 0` (app.audit.stats's win test) counts a NaN row as a
    WIN. Two live rows were silently inflating the measured win rate this
    way. 종가가 NaN이면 수익률도 NaN이 되고, Postgres는 NaN을 최대값으로 봐서
    승률에 '승'으로 잡힘 — 저장 전에 걸러야 함.
    """
    return value is not None and math.isfinite(value) and value > 0


def _common_eval_date(
    ticker: str, benchmark_ticker: str, prices: dict[date, float], benchmark_prices: dict[date, float],
    not_before: date,
) -> date | None:
    """Latest date, at or after `not_before`, where BOTH the ticker and the
    benchmark carry a usable close — the point the outcome gets measured at.

    Replaces an unconditional `today in prices` requirement, which silently
    stalled resolution forever rather than a day: a KRX daily bar is not
    published while the session is still open, so a cycle running intraday
    found today's bar missing, hit a bare `continue` with no log line, and
    did exactly the same thing on every subsequent cycle. 25 live decisions
    aged 8+ days were stuck pending this way (2026-08-13) — a silent ~15%
    leak out of the very sample the stage gate's alpha confidence interval
    is waiting on.

    Both series must have a usable close on the SAME date: evaluating the
    ticker on one date and the benchmark on another would subtract returns
    measured over two different windows, which is not alpha.
    종목과 벤치마크가 동시에 정상 종가를 가진 가장 최근 날짜 — 서로 다른 날로
    계산하면 알파가 아니게 됨.

    A candidate date is also rejected if the ticker's or the benchmark's bar
    for it is STILL BEING FORMED (`session_in_progress`) — `resolve_all_due`
    runs every live cycle, including intraday, so the latest "usable" close
    in `prices`/`benchmark_prices` can be an intraday snapshot mislabeled as
    a settled close (same bug class as the live-snapshot-mislabeled-as-close
    MRVL incident this project already fixed once). Falls through to the
    next-earlier date exactly like the `_usable` check already does, rather
    than ever accepting an in-progress bar as final.
    아직 형성 중인 당일 바(장중 스냅샷)를 확정 종가로 오인하지 않도록, 두 시장
    중 하나라도 세션이 열려 있으면 그 날짜는 건너뛰고 더 이전 날짜로 폴백."""
    shared = [d for d in prices.keys() & benchmark_prices.keys() if d >= not_before]
    for candidate in sorted(shared, reverse=True):
        if (
            _usable(prices.get(candidate))
            and _usable(benchmark_prices.get(candidate))
            and not session_in_progress(ticker, candidate)
            and not session_in_progress(benchmark_ticker, candidate)
        ):
            return candidate
    return None


def _horizon_eval_date(
    ticker: str, benchmark_ticker: str, prices: dict[date, float], benchmark_prices: dict[date, float],
    trade_date: date, n_trading_days: int,
) -> tuple[date, int] | None:
    """The date exactly `n_trading_days` TRADING days after `trade_date`,
    counting only dates where BOTH the ticker and the benchmark have a
    usable close — for `settings.memory_fixed_horizon_trading_days`'s
    fixed-window `horizon_*` columns (Stage 1, F6, 2026-08-13). `None` if
    fewer than `n_trading_days` such dates exist yet (row hasn't matured, or
    price history doesn't reach far enough).

    Trading-day counting rather than calendar-day arithmetic: `get_price_
    series` (app.dataflows.yfinance.market_data) only ever returns actual
    trading days as keys, so counting sorted keys directly IS trading-day
    counting — no calendar-vs-trading-day tolerance/adjustment needed.

    Deliberately a completely separate code path from `_common_eval_date`
    above rather than a generalization of it: `_common_eval_date` exists
    specifically to NEVER stall (falls back to whatever the latest usable
    bar is, however short of the nominal window that leaves it — see its
    own docstring for the multi-day live stall this fixed). Making THIS
    function's target date a hard requirement (exactly the Nth trading day,
    no fallback) is the whole point — a fixed window that quietly settled
    for "close enough" would stop being comparable across rows, which is
    the entire reason this column exists. So this returns `None` and lets
    the caller retry later, rather than ever taking a nearby substitute.

    Only dates where the TICKER AND BENCHMARK both have a usable close
    count — evaluating one on a date the other lacks would measure two
    different windows, which is not alpha (same reasoning as
    `_common_eval_date`).
    거래일 고정 N개를 셀 때 캘린더 계산이 아니라 실제 거래일만 카운팅 —
    get_price_series가 실제 거래일만 키로 주므로 정렬한 키 개수 = 거래일 수.
    가변창(_common_eval_date)과 달리 스톨 방지를 위해 대체값을 쓰지 않고, N일이
    안 되면 그냥 None을 돌려주고 다음 사이클에 재시도한다.

    A trading day still in progress (`session_in_progress`) for either
    series never counts toward the N-day tally either — same intraday-
    snapshot-as-settled-close risk as `_common_eval_date` above, and this
    function runs from the same live per-cycle call path. `trade_date`
    itself (the entry side, checked separately above) does NOT need this
    guard: it is always a historical decision date, never today's
    in-progress bar."""
    if not _usable(prices.get(trade_date)) or not _usable(benchmark_prices.get(trade_date)):
        return None
    shared = sorted(
        d
        for d in prices.keys() & benchmark_prices.keys()
        if d > trade_date
        and _usable(prices.get(d))
        and _usable(benchmark_prices.get(d))
        and not session_in_progress(ticker, d)
        and not session_in_progress(benchmark_ticker, d)
    )
    if len(shared) < n_trading_days:
        return None
    return shared[n_trading_days - 1], n_trading_days


def _alternate_krx_symbol(ticker: str) -> str | None:
    """'.KS' <-> '.KQ' for a numeric KRX code; None for anything else (overseas
    tickers, indices like ^KS11). 6-digit KRX codes are unique across markets, so
    the same code under the other suffix is the same security.
    숫자 KRX 코드만 반대 접미사로 바꿈 — 해외 티커와 지수는 대상 아님."""
    for suffix, other in ((".KS", ".KQ"), (".KQ", ".KS")):
        if ticker.endswith(suffix) and ticker[: -len(suffix)].isdigit():
            return ticker[: -len(suffix)] + other
    return None


async def _fetch_prices_with_krx_fallback(ticker: str, rows: list, end: date) -> tuple[dict[date, float], list]:
    """(prices, rows to process). Default path is exactly the old one: the STORED
    symbol's series from the earliest row's trade_date, all rows returned.

    Only when that lookup raises ValueError ("no history" -- Yahoo serves KOSDAQ names
    stored here as '.KS' only as '.KQ') AND `settings.memory_krx_fallback_since` is set,
    the OTHER suffix is tried for the rows with trade_date >= that date; older rows are
    left untouched (they stay pending) and are not returned. With the setting unset, or
    nothing eligible, or both symbols failing, the ORIGINAL ValueError is re-raised so
    behavior and logging match the pre-fallback code. Any non-ValueError (rate limit,
    network) propagates untouched -- it is not evidence the symbol is wrong.

    Callers must keep passing the STORED ticker to `session_in_progress` /
    `_common_eval_date` / `resolve_outcome`: `session_in_progress` classifies a non-'.KS'
    symbol as overseas, which would accept a still-forming bar as a settled close.
    저장 심볼 조회가 ValueError일 때만, 설정된 날짜 이후 행에 한해 반대 접미사를 시도."""
    try:
        return await get_price_series(ticker, min(r["trade_date"] for r in rows), end), rows
    except ValueError as original:
        alternate = _alternate_krx_symbol(ticker)
        since = settings.memory_krx_fallback_since
        if alternate is None or since is None:
            raise
        eligible = [r for r in rows if r["trade_date"] >= since]
        if not eligible:
            raise
        try:
            prices = await get_price_series(alternate, min(r["trade_date"] for r in eligible), end)
        except ValueError as alt_error:
            raise original from alt_error
        logger.info(
            "auto_resolve: %s has no Yahoo history under its stored suffix; priced via %s for %d row(s) "
            "with trade_date >= %s (%d older row(s) left untouched)",
            ticker, alternate, len(eligible), since, len(rows) - len(eligible),
        )
        return prices, eligible


async def resolve_ticker(ticker: str, benchmark_ticker: str | None = None) -> int:
    """Resolves every pending LIVE decision for `ticker` at least
    `settings.memory_min_holding_days` old, computing raw/alpha return
    against `benchmark_ticker` and an LLM reflection on the outcome. Returns
    the number of entries resolved.

    `benchmark_ticker`: `None` (the default, and what every real caller
    uses) resolves via `benchmark_for_market` — KOSPI for a domestic ticker,
    SPY for overseas. An explicit value overrides this (kept for the CLI's
    own `--benchmark` flag and tests), but should not be threaded through
    from `resolve_all_due`'s bulk path any more — see that function's own
    docstring for why a single shared value is wrong there by construction.

    `today`/`cutoff` use `ticker`'s OWN market-local date (`local_today`),
    not a single shared KST `get_clock().today()` — otherwise an overseas
    ticker's holding-days/return calculation would silently shift by
    however many hours KST midnight falls into the still-open NASDAQ
    session, the same class of boundary bug `app.live.auto_trade_loop`'s
    same-day dedup had.

    The outcome is measured at `_common_eval_date` — the latest date both the
    ticker and the benchmark actually have a usable close on, no earlier than
    `trade_date + memory_min_holding_days` — NOT unconditionally at `today`.
    When today's bar exists and is usable this resolves to `today` and the
    result is identical to the previous behavior; when it does not (an
    intraday cycle, a vendor gap, a NaN close) an earlier real bar is used
    instead of the entry never resolving at all. `holding_days` records the
    window actually measured, so a fallback is visible in the data rather
    than implied. 종료 시점은 '오늘'이 아니라 '양쪽 모두 정상 종가가 있는 최근
    날짜' — 오늘 바가 없어도 영구 미해결로 남지 않음."""
    market = ticker_market(ticker)
    resolved_benchmark = benchmark_ticker if benchmark_ticker is not None else benchmark_for_market(market)
    today = local_today(market)
    cutoff = today - timedelta(days=settings.memory_min_holding_days)
    pool = await get_pool()
    async with pool.acquire() as conn:
        pending = await conn.fetch(
            "SELECT id, trade_date, decision_text, rating, held_at_decision, held_at_decision_inferred "
            "FROM memory_entries "
            "WHERE ticker = $1 AND status = 'pending' AND trade_date <= $2 AND is_backtest = false "
            "AND account_mode = $3 AND user_id = $4",
            ticker,
            cutoff,
            current_account_mode(),
            get_current_user_id(),
        )

    if not pending:
        return 0

    # Every pending row for this ticker needs the exact same price series
    # (from the earliest pending trade_date through today) — fetched ONCE
    # here and reused for every row below, rather than once per row (which
    # would be 2N vendor calls for N pending rows on the same ticker every
    # cycle). Mirrors `backfill_horizon_returns`'s own identical pattern
    # (see its docstring) for the same reason.
    prices, pending = await _fetch_prices_with_krx_fallback(ticker, pending, today)
    earliest_trade_date = min(row["trade_date"] for row in pending)
    benchmark_prices = await get_price_series(resolved_benchmark, earliest_trade_date, today)

    resolved_count = 0
    for row in pending:
        trade_date = row["trade_date"]

        # Entry side stays strict — the decision's own trade_date is what the
        # return has to be measured FROM, so there is no sound substitute for
        # a missing/unusable bar there. It is now LOGGED rather than skipped
        # in silence, which is what made the stall below invisible for weeks.
        # 진입일 종가는 대체할 수 없으므로 엄격하게 유지 — 단 조용히 넘기지 않고 기록.
        if not _usable(prices.get(trade_date)) or not _usable(benchmark_prices.get(trade_date)):
            logger.warning(
                "skipping %s %s — no usable entry close for the ticker and/or benchmark (%s)",
                ticker, trade_date, resolved_benchmark,
            )
            continue

        # Exit side takes the latest date both series can actually be read
        # at, no earlier than the minimum holding period — see
        # `_common_eval_date`. The outer `cutoff` query above only guarantees
        # the decision is old ENOUGH; it says nothing about whether a bar
        # exists to measure it against yet.
        earliest_eval = trade_date + timedelta(days=settings.memory_min_holding_days)
        eval_date = _common_eval_date(ticker, resolved_benchmark, prices, benchmark_prices, earliest_eval)
        if eval_date is None:
            logger.warning(
                "skipping %s %s — no usable close on or after %s for both the ticker and %s yet; "
                "will retry next cycle",
                ticker, trade_date, earliest_eval, resolved_benchmark,
            )
            continue

        raw_return = (prices[eval_date] - prices[trade_date]) / prices[trade_date]
        benchmark_return = (
            benchmark_prices[eval_date] - benchmark_prices[trade_date]
        ) / benchmark_prices[trade_date]
        alpha_return = raw_return - benchmark_return
        holding_days = (eval_date - trade_date).days

        held_at_decision = (
            row["held_at_decision"] if row["held_at_decision"] is not None else row["held_at_decision_inferred"]
        )
        reflection_text = await reflect(
            row["decision_text"], raw_return, alpha_return, resolved_benchmark,
            rating=row["rating"], held_at_decision=held_at_decision,
        )
        await resolve_outcome(
            row["id"], ticker, trade_date, raw_return, alpha_return, holding_days, reflection_text,
            resolved_benchmark, resolved_on=eval_date,
        )
        resolved_count += 1

        # Stage 1 (2026-08-13) — F6: opportunistic attempt at the fixed
        # trading-day horizon (see `_horizon_eval_date`'s own docstring),
        # using the SAME price series already fetched above — no extra
        # vendor call. This will usually return None (the fixed horizon,
        # settings.memory_fixed_horizon_trading_days trading days, is
        # normally further out than this variable-window resolution just
        # reached) — that's expected, not an error; `backfill_horizon_
        # returns` (called from `resolve_all_due` below) is the real,
        # comprehensive path that revisits these later once they mature.
        try:
            horizon = _horizon_eval_date(
                ticker, resolved_benchmark, prices, benchmark_prices, trade_date,
                settings.memory_fixed_horizon_trading_days,
            )
            if horizon is not None:
                horizon_eval_date, horizon_trading_days = horizon
                horizon_raw_return = (
                    prices[horizon_eval_date] - prices[trade_date]
                ) / prices[trade_date]
                horizon_benchmark_return = (
                    benchmark_prices[horizon_eval_date] - benchmark_prices[trade_date]
                ) / benchmark_prices[trade_date]
                horizon_alpha_return = horizon_raw_return - horizon_benchmark_return
                await set_horizon_outcome(
                    row["id"], horizon_raw_return, horizon_alpha_return, horizon_trading_days, horizon_eval_date
                )
            else:
                logger.debug(
                    "%s %s — fixed %d-trading-day horizon not yet reached; backfill_horizon_returns will "
                    "retry later",
                    ticker, trade_date, settings.memory_fixed_horizon_trading_days,
                )
        except Exception:
            # Best-effort measurement layer, not a safety gate — must never
            # abort the resolution loop that already succeeded above.
            logger.warning(
                "opportunistic horizon fill failed for %s %s, continuing", ticker, trade_date, exc_info=True
            )

    return resolved_count


async def backfill_horizon_returns(window_days: int | None = None, ticker: str | None = None) -> int:
    """The main path (not the opportunistic one inside `resolve_ticker`
    above) for filling `horizon_*` on rows that are already `status =
    'resolved'` but still missing `horizon_alpha_return` — most rows land
    here rather than through the opportunistic path, since the fixed
    `settings.memory_fixed_horizon_trading_days`-trading-day horizon is
    normally further out than the variable window `resolve_ticker` just
    resolved at. Returns the number of rows filled.

    Deliberately does NOT filter on `is_backtest` — unlike almost every
    other query in this module, this one fills BOTH live and backtest rows,
    because a fixed, comparable horizon is exactly as useful for a
    backtest's own decisions as for a live one, and app.backtest.harness's
    own resolution path is untouched by this (it keeps writing `raw_return`/
    `alpha_return`/`holding_days` exactly as before) — this function is
    purely additive on top of whatever already got resolved, live or
    backtest.

    `window_days` bounds `trade_date` to the last N days (None = no bound,
    i.e. `--all`) — keeps a routine per-cycle call cheap; a row older than
    the window that's still NULL is treated as permanently uncomputable,
    not chased forever. `ticker` narrows to one ticker (mirrors `resolve_
    ticker`'s own single-ticker scope).

    Fetches each ticker's own price series exactly ONCE and reuses it for
    every one of that ticker's pending rows, rather than once per row —
    avoids a vendor-call blowup on a ticker with many still-unfilled rows.
    Each ticker is wrapped in its own try/except (mirrors `resolve_all_due`'s
    own per-ticker isolation) so one ticker's price-fetch failure can't
    abort every other ticker's backfill in the same pass.

    Never touches `status` — every row this reads is already 'resolved';
    only the horizon_* columns are written."""
    conditions = ["status = 'resolved'", "horizon_alpha_return IS NULL"]
    params: list = []
    if window_days is not None:
        cutoff = min(local_today("domestic"), local_today("overseas")) - timedelta(days=window_days)
        params.append(cutoff)
        conditions.append(f"trade_date >= ${len(params)}")
    if ticker is not None:
        params.append(ticker)
        conditions.append(f"ticker = ${len(params)}")
    params.append(get_current_user_id())
    conditions.append(f"user_id = ${len(params)}")
    where_clause = " AND ".join(conditions)

    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            f"""
            SELECT id, ticker, trade_date
            FROM memory_entries
            WHERE {where_clause}
            ORDER BY ticker, trade_date
            """,
            *params,
        )

    by_ticker: dict[str, list] = {}
    for row in rows:
        by_ticker.setdefault(row["ticker"], []).append(row)

    total = 0
    for tkr, entries in by_ticker.items():
        try:
            market = ticker_market(tkr)
            today = local_today(market)
            benchmark = benchmark_for_market(market)
            prices, entries = await _fetch_prices_with_krx_fallback(tkr, entries, today)
            earliest_trade_date = min(e["trade_date"] for e in entries)
            benchmark_prices = await get_price_series(benchmark, earliest_trade_date, today)

            for entry in entries:
                trade_date = entry["trade_date"]
                horizon = _horizon_eval_date(
                    tkr, benchmark, prices, benchmark_prices, trade_date,
                    settings.memory_fixed_horizon_trading_days,
                )
                if horizon is None:
                    continue
                horizon_eval_date, horizon_trading_days = horizon
                horizon_raw_return = (
                    prices[horizon_eval_date] - prices[trade_date]
                ) / prices[trade_date]
                horizon_benchmark_return = (
                    benchmark_prices[horizon_eval_date] - benchmark_prices[trade_date]
                ) / benchmark_prices[trade_date]
                horizon_alpha_return = horizon_raw_return - horizon_benchmark_return
                await set_horizon_outcome(
                    entry["id"], horizon_raw_return, horizon_alpha_return, horizon_trading_days, horizon_eval_date
                )
                total += 1
        except Exception:
            logger.warning("backfill_horizon_returns: failed for ticker %s, continuing", tkr, exc_info=True)

    return total


async def resolve_all_due() -> int:
    """Resolves every LIVE ticker with at least one pending decision old
    enough to resolve — the entry point app.live.auto_trade_loop calls every
    cycle. No separate once-per-day debounce needed (unlike e.g.
    app.execution.market_regime's daily refresh): once an entry resolves its
    `status` flips to 'resolved' and it drops out of
    tickers_with_pending_resolution's query on its own, so calling this every
    cycle just means near-zero-cost empty SELECTs on the (overwhelmingly
    common) cycles where nothing is newly due. Best-effort per ticker: one
    ticker's failure (price fetch, LLM reflection call) must not abort the
    others — mirrors _held_needing_review's per-ticker try/except style.

    No `benchmark_ticker` parameter any more (2026-08-12) — this used to
    default to a single shared 'SPY' and pass it to every ticker regardless
    of market, silently scoring every domestic decision against the wrong
    index. `resolve_ticker` now resolves each ticker's own correct
    benchmark internally (`benchmark_for_market`), which a per-market-mixed
    bulk caller like this one has no business overriding with one shared
    value — see `resolve_ticker`'s own docstring. Confirmed live: this bug
    had already mis-scored all 207 resolved domestic memory_entries against
    SPY while KOSPI sat ~31% below its high — see project memory:
    ai-cycle-audit-2026-08-10's benchmark-mismatch finding.

    The listing query below uses the EARLIER (more lenient) of the two
    markets' local cutoffs, so it can never miss a ticker that's actually
    due under its own market's date — `resolve_ticker` re-derives each
    ticker's own market-correct cutoff independently and simply finds
    nothing to resolve yet for one that only looked due under the other
    market's more lenient date. A few near-zero-cost empty per-ticker
    passes a day early is a fine trade for never silently skipping one."""
    cutoff = min(local_today("domestic"), local_today("overseas")) - timedelta(days=settings.memory_min_holding_days)
    try:
        tickers = await tickers_with_pending_resolution(cutoff)
    except Exception:
        logger.warning("auto_resolve: could not list tickers with pending resolutions", exc_info=True)
        return 0

    total = 0
    for ticker in tickers:
        try:
            total += await resolve_ticker(ticker)
        except Exception:
            logger.warning("auto_resolve: failed to resolve pending entries for %s", ticker, exc_info=True)
    if total:
        logger.info("auto_resolve: resolved %d pending live decision(s) across %d ticker(s)", total, len(tickers))

    # Stage 1 (2026-08-13) — F6 backfill pass. Purely a measurement layer on
    # top of already-resolved rows (live AND backtest — see `backfill_
    # horizon_returns`'s own docstring), so a failure here must NEVER change
    # this function's own return value (the count of pending decisions
    # actually resolved this cycle, which callers may treat as meaningful).
    try:
        await backfill_horizon_returns(settings.memory_horizon_backfill_window_days)
    except Exception:
        logger.warning("auto_resolve: horizon backfill pass failed, continuing", exc_info=True)

    return total
