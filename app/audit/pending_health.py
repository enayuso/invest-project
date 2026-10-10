"""app.audit.pending_health -- weekly health check on `status='pending'` live
decisions, so a silent resolution stall gets caught in days, not weeks.

WHY THIS EXISTS (2026-09-28/29 incident, handoff/knowledge/kosdaq-ks-suffix-
resolver-2026-09-28-29.md): KOSDAQ names are stored here as '.KS', but Yahoo
only serves them as '.KQ', so `app.memory.auto_resolve.resolve_ticker` never
resolved them -- pending share for real/domestic quietly rose from ~6% to 36%
over two weeks before anyone noticed. This module is the recurring check:
default mode (DB-only, safe anytime including market hours) buckets pending
vs "overdue pending" (old enough that production's own resolver would already
have tried it) by ISO week/market/account_mode and warns past a threshold;
`--probe` additionally asks yfinance WHY an overdue row is still pending, by
calling `app.audit.pending_recompute.recompute_pending_alphas` (never
reimplemented here) and keeping only its classification, not its alpha.

Read-only throughout: no INSERT/UPDATE/DELETE and no call to `resolve_outcome`/
`set_horizon_outcome`/`reflect` anywhere (this module itself never imports
`app.memory.auto_resolve` directly -- its cutoff formula is duplicated rather than
imported, see `overdue_cutoffs`'s own docstring; `pending_recompute`, which this
module DOES import for `--probe`, imports two pure helpers from it,
`_horizon_eval_date`/`_usable`, but nothing that writes). This tool renders
numbers; a human decides what (if anything) belongs in `open-issues.md`.

`--probe` deliberately ignores `settings.memory_krx_fallback_since` (the
production activation gate, 2026-09-29): that gate controls whether the
resolver WRITES a resolution and generates a reflection, not whether a row
COULD resolve. Diagnosing "would fixing this help" must not depend on
whether the fix has already been turned on locally -- and this CLI's own
`memory_krx_fallback_since` is a LOCAL setting that may not match the
server's, so it is only ever displayed, never enforced here (see
`current_account_mode`'s own docstring for the same local-vs-server caveat --
this module reads pending rows straight from the DB instead of trusting it)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

from app.audit.pending_recompute import recompute_pending_alphas
from app.audit.weekly import parse_week_label, week_label
from app.clock import local_today
from app.config import settings
from app.memory.db import get_pool
from app.user_context import get_current_user_id

MARKETS = ("domestic", "overseas")
CATEGORY_ALT_SUFFIX = "resolvable_via_alt_suffix"
CATEGORY_STORED_SYMBOL = "resolvable_under_stored_symbol"
DEFAULT_WARN_PCT = 15.0
DEFAULT_MIN_ELIGIBLE = 30
DEFAULT_WEEKS = 8
_MAX_LISTED_TICKERS = 20


def _validate(
    market: str | None, account_mode: str | None, weeks: int, warn_pct: float,
    min_eligible: int, probe: bool,
) -> None:
    if market not in (None, "domestic", "overseas"):
        raise ValueError(f"market must be 'domestic', 'overseas', or None, got {market!r}")
    if account_mode not in (None, "paper", "real"):
        raise ValueError(f"account_mode must be 'paper', 'real', or None, got {account_mode!r}")
    if weeks < 1:
        raise ValueError(f"weeks must be >= 1, got {weeks!r}")
    if not (0 < warn_pct <= 100):
        raise ValueError(f"warn_pct must be in (0, 100], got {warn_pct!r}")
    if min_eligible < 1:
        raise ValueError(f"min_eligible must be >= 1, got {min_eligible!r}")
    if probe and market != "domestic":
        raise ValueError("probe=True requires market='domestic' (the KOSDAQ suffix issue is domestic-only)")


def overdue_cutoffs() -> dict[str, date]:
    """{market: latest trade_date the PRODUCTION resolver would already
    consider old enough to resolve} -- the exact formula `auto_resolve.
    resolve_ticker` uses (`local_today(market) - memory_min_holding_days`),
    duplicated here rather than imported so this module never has a direct
    import of `app.memory.auto_resolve` (a read-only audit tool has no
    business importing the write path)."""
    return {m: local_today(m) - timedelta(days=settings.memory_min_holding_days) for m in MARKETS}


@dataclass(frozen=True)
class PendingHealthBucket:
    week: str  # 'YYYY-Www' (ISO), via app.audit.weekly.week_label
    week_start: date  # Monday of that ISO week
    market: str
    account_mode: str
    n_total: int  # every live row that week, any status
    n_pending: int
    n_eligible: int  # trade_date <= cutoffs[market] -- production's OWN resolve-eligibility test
    n_overdue_pending: int  # eligible AND still pending
    partial: bool  # week_start + 6 > cutoffs[market]: this week's eligibility window isn't even closed yet

    @property
    def overdue_pct(self) -> float | None:
        return None if self.n_eligible == 0 else 100.0 * self.n_overdue_pending / self.n_eligible


@dataclass(frozen=True)
class ResolverFreshness:
    """Per (market, account_mode): is the resolver actually running for this
    group at all? A group can have a low weekly overdue% and still be
    permanently stuck (e.g. paper mode nobody resolves anymore) -- this is
    the all-time view the weekly buckets can't show."""
    market: str
    account_mode: str
    last_resolved_on: date | None
    n_overdue_pending: int  # sum within the report's own date window
    stale: bool  # overdue rows exist but nothing has resolved since before the cutoff


@dataclass(frozen=True)
class ProbeRow:
    ticker: str
    trade_date: date
    account_mode: str
    rating: str
    category: str  # CATEGORY_* or one of pending_recompute's REASON_* strings
    price_symbol: str | None  # set only for the two CATEGORY_* (resolvable) rows
    gate_covers: bool  # would settings.memory_krx_fallback_since (LOCAL) already cover this row?


@dataclass(frozen=True)
class PendingProbeResult:
    today: date
    rows: list[ProbeRow] = field(default_factory=list)
    n_tickers_fetched: int = 0
    n_alt_symbol_tickers: int = 0


@dataclass(frozen=True)
class PendingHealthReport:
    market: str | None
    account_mode: str | None
    weeks: int
    since: date  # Monday of the earliest ISO week shown
    cutoffs: dict[str, date]
    min_holding_days: int
    warn_pct: float
    min_eligible: int
    krx_fallback_since_local: date | None
    buckets: list[PendingHealthBucket] = field(default_factory=list)
    freshness: list[ResolverFreshness] = field(default_factory=list)
    probe: PendingProbeResult | None = None


def _market_clause(market: str | None) -> str:
    if market == "domestic":
        return "AND ticker LIKE '%.KS'"
    if market == "overseas":
        return "AND ticker NOT LIKE '%.KS'"
    return ""


async def _fetch_daily_counts(conn, *, market: str | None, account_mode: str | None, since: date, cutoffs: dict) -> list:
    params: list = [cutoffs["domestic"], cutoffs["overseas"]]
    conditions = [
        "is_backtest = false",
        "trade_date >= $3",
    ]
    params.append(since)
    if account_mode is not None:
        params.append(account_mode)
        conditions.append(f"account_mode = ${len(params)}")
    params.append(get_current_user_id())
    conditions.append(f"user_id = ${len(params)}")
    market_clause = _market_clause(market)
    if market_clause:
        conditions.append(market_clause.removeprefix("AND ").strip())
    where_clause = " AND ".join(conditions)
    query = f"""
        SELECT trade_date,
               CASE WHEN ticker LIKE '%.KS' THEN 'domestic' ELSE 'overseas' END AS market,
               account_mode,
               count(*) AS n_total,
               count(*) FILTER (WHERE status = 'pending') AS n_pending,
               count(*) FILTER (
                   WHERE trade_date <= CASE WHEN ticker LIKE '%.KS' THEN $1::date ELSE $2::date END
               ) AS n_eligible,
               count(*) FILTER (
                   WHERE status = 'pending'
                     AND trade_date <= CASE WHEN ticker LIKE '%.KS' THEN $1::date ELSE $2::date END
               ) AS n_overdue_pending
        FROM memory_entries
        WHERE {where_clause}
        GROUP BY 1, 2, 3
        ORDER BY 1, 2, 3
    """
    return await conn.fetch(query, *params)


async def _fetch_last_resolved(conn, *, market: str | None, account_mode: str | None) -> list:
    params: list = [get_current_user_id()]
    conditions = ["is_backtest = false", "status = 'resolved'", f"user_id = $1"]
    market_clause = _market_clause(market)
    if market_clause:
        conditions.append(market_clause.removeprefix("AND ").strip())
    if account_mode is not None:
        params.append(account_mode)
        conditions.append(f"account_mode = ${len(params)}")
    where_clause = " AND ".join(conditions)
    query = f"""
        SELECT CASE WHEN ticker LIKE '%.KS' THEN 'domestic' ELSE 'overseas' END AS market,
               account_mode, max(resolved_on) AS last_resolved_on
        FROM memory_entries
        WHERE {where_clause}
        GROUP BY 1, 2
    """
    return await conn.fetch(query, *params)


async def _fetch_overdue_pending_rows(conn, *, account_mode: str | None, since: date, cutoff: date) -> list:
    """Domestic only -- the caller (`compute_pending_health` with `probe=True`)
    already enforces market='domestic'. Deliberately the SAME filter the
    production resolver uses (status/is_backtest/trade_date/account_mode/
    user_id) with no rating filter and no DISTINCT ON dedup -- this must see
    every row the resolver would try, not a curated subset."""
    params: list = [cutoff, since, get_current_user_id()]
    conditions = [
        "status = 'pending'", "is_backtest = false", "ticker LIKE '%.KS'",
        "trade_date <= $1", "trade_date >= $2", "user_id = $3",
    ]
    if account_mode is not None:
        params.append(account_mode)
        conditions.append(f"account_mode = ${len(params)}")
    where_clause = " AND ".join(conditions)
    query = f"""
        SELECT ticker, trade_date, rating, held_at_decision, held_at_decision_inferred, account_mode
        FROM memory_entries
        WHERE {where_clause}
        ORDER BY account_mode, trade_date, ticker
    """
    return await conn.fetch(query, *params)


def _build_buckets(daily_rows, *, cutoffs: dict) -> list[PendingHealthBucket]:
    """Aggregates daily counts into (week, market, account_mode) buckets. Whether a
    bucket WARNS is a rendering-time decision (`_bucket_warns`), not baked in here --
    the same buckets are reused unchanged if a caller re-renders with a different
    `warn_pct`/`min_eligible`/`stale_groups`."""
    grouped: dict[tuple[str, str, str], dict] = {}
    for row in daily_rows:
        wk = week_label(row["trade_date"])
        key = (wk, row["market"], row["account_mode"])
        acc = grouped.setdefault(key, {"n_total": 0, "n_pending": 0, "n_eligible": 0, "n_overdue_pending": 0})
        acc["n_total"] += row["n_total"]
        acc["n_pending"] += row["n_pending"]
        acc["n_eligible"] += row["n_eligible"]
        acc["n_overdue_pending"] += row["n_overdue_pending"]

    buckets = []
    for (wk, market, account_mode), acc in grouped.items():
        week_start = parse_week_label(wk)
        partial = (week_start + timedelta(days=6)) > cutoffs[market]
        buckets.append(PendingHealthBucket(
            week=wk, week_start=week_start, market=market, account_mode=account_mode,
            n_total=acc["n_total"], n_pending=acc["n_pending"], n_eligible=acc["n_eligible"],
            n_overdue_pending=acc["n_overdue_pending"], partial=partial,
        ))
    buckets.sort(key=lambda b: (b.market, b.account_mode, b.week_start))
    return buckets


def _bucket_warns(b: PendingHealthBucket, *, warn_pct: float, min_eligible: int, stale_groups: set) -> bool:
    if (b.market, b.account_mode) in stale_groups:
        return False  # STALE groups get one summary line instead of per-week noise
    if b.n_eligible < min_eligible:
        return False
    pct = b.overdue_pct
    return pct is not None and pct >= warn_pct


def _build_freshness(daily_rows, last_resolved_rows, *, cutoffs: dict) -> list[ResolverFreshness]:
    overdue_by_group: dict[tuple[str, str], int] = {}
    for row in daily_rows:
        key = (row["market"], row["account_mode"])
        overdue_by_group[key] = overdue_by_group.get(key, 0) + row["n_overdue_pending"]
    last_by_group = {(r["market"], r["account_mode"]): r["last_resolved_on"] for r in last_resolved_rows}

    out = []
    for key in sorted(set(overdue_by_group) | set(last_by_group)):
        market, account_mode = key
        last_resolved_on = last_by_group.get(key)
        n_overdue = overdue_by_group.get(key, 0)
        stale = n_overdue > 0 and (last_resolved_on is None or last_resolved_on < cutoffs[market])
        out.append(ResolverFreshness(
            market=market, account_mode=account_mode, last_resolved_on=last_resolved_on,
            n_overdue_pending=n_overdue, stale=stale,
        ))
    return out


async def compute_pending_health(
    market: str | None = None, account_mode: str | None = None, weeks: int = DEFAULT_WEEKS,
    warn_pct: float = DEFAULT_WARN_PCT, min_eligible: int = DEFAULT_MIN_ELIGIBLE, probe: bool = False,
) -> PendingHealthReport:
    _validate(market, account_mode, weeks, warn_pct, min_eligible, probe)
    # Capture "today" for every market exactly once at the top -- calling local_today()
    # again later (e.g. right after a KST midnight rollover) could return a different date
    # than the cutoff/since below were computed from.
    today_domestic = local_today("domestic")
    cutoffs = {"domestic": today_domestic - timedelta(days=settings.memory_min_holding_days),
               "overseas": local_today("overseas") - timedelta(days=settings.memory_min_holding_days)}
    since = (today_domestic - timedelta(days=today_domestic.weekday())) - timedelta(weeks=weeks - 1)

    pool = await get_pool()
    async with pool.acquire() as conn:
        async with conn.transaction(isolation="repeatable_read", readonly=True):
            daily_rows = await _fetch_daily_counts(conn, market=market, account_mode=account_mode, since=since, cutoffs=cutoffs)
            last_resolved_rows = await _fetch_last_resolved(conn, market=market, account_mode=account_mode)
            overdue_rows = (
                await _fetch_overdue_pending_rows(conn, account_mode=account_mode, since=since, cutoff=cutoffs["domestic"])
                if probe else []
            )

    freshness = _build_freshness(daily_rows, last_resolved_rows, cutoffs=cutoffs)
    stale_groups = {(f.market, f.account_mode) for f in freshness if f.stale}
    buckets = _build_buckets(daily_rows, cutoffs=cutoffs)

    probe_result = None
    if probe:
        probe_result = await probe_overdue_pending(overdue_rows, today=today_domestic)

    return PendingHealthReport(
        market=market, account_mode=account_mode, weeks=weeks, since=since, cutoffs=cutoffs,
        min_holding_days=settings.memory_min_holding_days, warn_pct=warn_pct, min_eligible=min_eligible,
        krx_fallback_since_local=settings.memory_krx_fallback_since,
        buckets=buckets, freshness=freshness, probe=probe_result,
    )


async def probe_overdue_pending(rows, *, today: date) -> PendingProbeResult:
    """Classifies each overdue pending row via `pending_recompute.
    recompute_pending_alphas` (imported, never reimplemented) -- the alpha
    numbers it computes are thrown away, only `price_symbol` (success) or
    `reason` (failure) is kept. Grouped by account_mode so a same (ticker,
    trade_date) key never collides across modes when mapping results back."""
    since_local = settings.memory_krx_fallback_since
    by_mode: dict[str, list] = {}
    for row in rows:
        by_mode.setdefault(row["account_mode"], []).append(row)

    out_rows: list[ProbeRow] = []
    n_fetched = 0
    n_alt = 0
    for account_mode in sorted(by_mode):
        group = by_mode[account_mode]
        try:
            result = await recompute_pending_alphas(group, basis="variable", today=today)
        except Exception as exc:
            # A total benchmark-fetch failure (rate limit, network) inside recompute_pending_alphas
            # raises rather than returning per-row failures -- this is a DIAGNOSTIC tool, so one
            # mode's benchmark outage must not take down the whole report (including the DB-only
            # weekly table computed above); isolate it to this mode's rows instead.
            for row in group:
                out_rows.append(ProbeRow(
                    ticker=row["ticker"], trade_date=row["trade_date"], account_mode=account_mode,
                    rating=row["rating"], category=f"fetch_error:benchmark:{type(exc).__name__}",
                    price_symbol=None, gate_covers=False,
                ))
            continue
        n_fetched += result.n_tickers_fetched
        n_alt += result.n_alt_symbol_tickers

        for r in result.rows:
            is_alt = r.price_symbol != r.ticker
            out_rows.append(ProbeRow(
                ticker=r.ticker, trade_date=r.trade_date, account_mode=account_mode, rating=r.rating,
                category=CATEGORY_ALT_SUFFIX if is_alt else CATEGORY_STORED_SYMBOL,
                price_symbol=r.price_symbol,
                gate_covers=is_alt and since_local is not None and r.trade_date >= since_local,
            ))
        for f in result.failures:
            out_rows.append(ProbeRow(
                ticker=f.ticker, trade_date=f.trade_date, account_mode=account_mode, rating=f.rating,
                category=f.reason, price_symbol=None, gate_covers=False,
            ))

    return PendingProbeResult(today=today, rows=out_rows, n_tickers_fetched=n_fetched, n_alt_symbol_tickers=n_alt)


def render_pending_health(report: PendingHealthReport) -> str:
    r = report
    gate_str = f"{r.krx_fallback_since_local} (ON)" if r.krx_fallback_since_local else "None (OFF)"
    lines = [
        "# Pending-resolution health (memory_entries, live only -- what auto_resolve.resolve_ticker selects)",
        f"market={r.market or '(both)'}  account_mode={r.account_mode or '(both)'}  weeks={r.weeks} "
        f"(since {r.since})  warn_pct={r.warn_pct:.1f}%  min_eligible={r.min_eligible}",
        f"overdue cutoff = local_today(market) - memory_min_holding_days({r.min_holding_days}d): "
        f"domestic<={r.cutoffs['domestic']}  overseas<={r.cutoffs['overseas']}",
        f"memory_krx_fallback_since (LOCAL setting -- server value may differ): {gate_str}",
        "",
    ]
    if not r.buckets:
        lines.append("(no rows in this range/scope)")
    else:
        header = f"{'week':<9}{'market':<10}{'mode':<7}{'total':>7}{'pending':>9}{'eligible':>10}{'overdue':>9}{'overdue%':>10}"
        lines.append(header)
        lines.append("-" * len(header))
        stale_groups = {(f.market, f.account_mode) for f in r.freshness if f.stale}
        warnings = []
        for b in r.buckets:
            pct_str = f"{b.overdue_pct:.1f}%" if b.overdue_pct is not None else "-"
            warn = _bucket_warns(b, warn_pct=r.warn_pct, min_eligible=r.min_eligible, stale_groups=stale_groups)
            flag = "  !!" if warn else ("  (partial)" if b.partial else "")
            lines.append(
                f"{b.week:<9}{b.market:<10}{b.account_mode:<7}{b.n_total:>7}{b.n_pending:>9}"
                f"{b.n_eligible:>10}{b.n_overdue_pending:>9}{pct_str:>10}{flag}"
            )
            if warn:
                partial_note = " (partial: may be settle lag -- confirm with --probe)" if b.partial else ""
                warnings.append(
                    f"!! WARNING {b.week} {b.market}/{b.account_mode}: overdue-pending {b.overdue_pct:.1f}% "
                    f"({b.n_overdue_pending}/{b.n_eligible}) >= {r.warn_pct:.1f}%{partial_note} -- run "
                    f"`python -m app.audit pending-health --market {b.market} --account-mode {b.account_mode} --probe` "
                    "after 15:30 KST"
                )
        lines.append("")
        lines.extend(warnings)

    lines += ["", "## resolver freshness (max resolved_on = eval date last graded, NOT the resolver-run date; all time)"]
    if not r.freshness:
        lines.append("(no rows)")
    else:
        fheader = f"{'market':<10}{'mode':<7}{'last_resolved_on':<18}{'overdue_pending(window)':>25}  status"
        lines.append(fheader)
        for f in r.freshness:
            status = "!! STALE (nothing resolved since before cutoff)" if f.stale else "ok"
            last = str(f.last_resolved_on) if f.last_resolved_on else "(never)"
            lines.append(f"{f.market:<10}{f.account_mode:<7}{last:<18}{f.n_overdue_pending:>25}  {status}")

    lines += [
        "",
        "NOTES: overdue uses production's own resolve-eligibility cutoff, so weekends/holidays/today's still-",
        "unsettled close can be mixed in -- a 'partial' cell means the eligibility window for that week hasn't",
        "fully closed yet; use --probe's not_matured count to tell settle-lag from a real stall. STALE means the",
        "resolver hasn't cleared anything for that (market, account_mode) group since before the cutoff -- daemon",
        "not running, an account-mode switch left abandoned, or a long holiday are all possible causes. A ticker",
        "stored as '.KQ' would be counted as overseas here (out of this tool's scope). This tool renders numbers",
        "and does not render a verdict -- a human decides what belongs in open-issues.md.",
    ]

    if r.probe is not None:
        p = r.probe
        lines += ["", f"## probe (yfinance, domestic, read-only; today={p.today})"]
        by_mode_cat: dict[tuple[str, str], list] = {}
        for row in p.rows:
            by_mode_cat.setdefault((row.account_mode, row.category), []).append(row)
        if not p.rows:
            lines.append("(no overdue pending rows in this range/scope)")
        else:
            # Category column grows to fit the longest label (min 32) -- 'fetch_error:benchmark:'
            # plus an exception class name can run well past a fixed 32 chars.
            cat_w = max(32, max(len(cat) for _, cat in by_mode_cat) + 2)
            pheader = f"{'mode':<7}{'category':<{cat_w}}{'rows':>6}{'tickers':>9}{'gate_covers':>13}"
            lines.append(pheader)
            for (mode, category), group in sorted(by_mode_cat.items()):
                n_tickers = len({row.ticker for row in group})
                n_covers = sum(1 for row in group if row.gate_covers)
                lines.append(f"{mode:<7}{category:<{cat_w}}{len(group):>6}{n_tickers:>9}{n_covers:>13}")
            lines.append(f"alt-symbol tickers {p.n_alt_symbol_tickers}/{p.n_tickers_fetched} fetched")
            stored_symbol_rows = [row for row in p.rows if row.category == CATEGORY_STORED_SYMBOL]
            if stored_symbol_rows:
                listed = ", ".join(
                    f"{row.ticker} {row.trade_date}({row.account_mode})"
                    for row in stored_symbol_rows[:_MAX_LISTED_TICKERS]
                )
                more = len(stored_symbol_rows) - _MAX_LISTED_TICKERS
                if more > 0:
                    listed += f", +{more} more"
                lines.append(f"{CATEGORY_STORED_SYMBOL}: {listed}")
        lines += [
            "",
            f"probe NOTE: {CATEGORY_STORED_SYMBOL} rows fetch fine under their STORED symbol -- if still pending in "
            "production, suspect the resolver not running / an account-mode switch (see freshness above), an "
            "intermittent Yahoo response, or an LLM reflection failure, not the '.KS'/'.KQ' suffix. gate_covers is "
            "evaluated against the LOCAL memory_krx_fallback_since only.",
        ]
    return "\n".join(lines)
