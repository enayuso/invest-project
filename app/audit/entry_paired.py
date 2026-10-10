"""Same-day paired Buy-vs-Hold(flat) alpha comparison (Analysis (1)), plus a
code-version stratification of the same cohorts (Analysis (3)) -- both from
the engine-audit-2026-09-16 follow-up (see project memory:
engine-audit-2026-09-15.md / trader-collapse-fix-2026-09-16.md context).

WHY THIS EXISTS: the aggregate self-audit (`python -m app.audit report`)
showed, on the fixed 10-trading-day horizon, Buy losing to its own win
condition (mean alpha -1.46%, win condition alpha>0) while Hold(flat) beat
ITS OWN win condition (mean alpha +2.06%, win condition alpha<0) -- i.e.
names the system entered underperformed the benchmark, and names it
declined to enter outperformed it, in BOTH directions at once. That
comparison is CONFOUNDED: Buy and Hold(flat) are different tickers on
different trade_dates, so a market/sector move on whichever days happened
to carry more Buys could produce the same shape with no real discrimination
defect at all.

Analysis (1) controls for that the same way `app.audit.guard_alpha` controls
for a market shock across a guard's violated/clean split: pair Buy rows
against Hold(flat) rows from the SAME trade_date only (holding the day's
market/sector shock and screening universe constant), and test whether the
daily (mean Buy alpha - mean Hold-flat alpha) is reliably negative. This
reuses `guard_alpha`'s own paired-design statistics almost verbatim --
`factor_metrics.block_bootstrap_ci` for the CI and
`app.audit.guard_alpha._permutation_p_value` for the p-value -- rather than
reimplementing that statistics code here. `guard_alpha.py` itself is not
modified.

CRITICAL, deliberate, do not "fix": both cohorts are scored on RAW
`{horizon_,}alpha_return`, never the sign-flipped "signed alpha"
`compute_confidence_calibration`/`guard_alpha` use internally for their own
purposes. This module is comparing two cohorts' forward returns to EACH
OTHER as stock sets, not scoring either cohort against its own win
condition -- flipping the sign on Hold(flat) (as the "signed alpha"
convention would, since Hold(flat) wins on alpha<0) would silently invert
half of the comparison and produce a number that means nothing. If you are
tempted to import/reuse the signed-alpha CASE expression from
`app.audit.stats`/`app.audit.guard_alpha` here, don't -- that is a different
question (was each row's own call right) from this one (did cohort A's
stocks outperform cohort B's stocks).

Held/flat determination for the Hold cohort reuses the EXACT convention
`app.audit.stats.compute_hit_rate_by_rating` already uses:
`COALESCE(held_at_decision, held_at_decision_inferred)` -- the directly
observed column when set, the 2026-08-20 ledger-reconstruction backfill
otherwise (~97.2% measured agreement -- see app.memory.db's
held_at_decision_inferred column comment). A Hold row where NEITHER column
is set is excluded from both cohorts and counted separately
(`n_excluded_unknown_held`), never guessed into either bucket. A Hold row
that resolves to HELD (not flat) is simply not part of this comparison at
all (Analysis (1) is about entered-vs-avoided, and a held position is
neither) -- it is not counted as "excluded" since its state is known, just
not relevant here.

Row selection otherwise reuses the same `market`/`scope`/`account_mode`/
`basis`/`since`/`until` filter semantics as
`app.audit.stats.compute_hit_rate_by_rating` (status='resolved', DISTINCT ON
(ticker, trade_date) ... ORDER BY created_at DESC dedup, analysts_failed
exclusion, user_id scoping, since/until anchored on horizon_eval_date
regardless of basis) -- see that function's docstring for the shared
rationale, not repeated here.

Analysis (3) answers a different question: is Analysis (1)'s result (or the
raw aggregate table's) a property of the system as it stands today, or an
artifact of one era of a codebase that changed constantly across the
sample? `memory_entries.git_commit` records the code version, but was only
populated well after some of this sample was written (0 of 580 rows as of
2026-08-13 -- see app/config.py's `git_commit` setting comment), and even
after that most individual commits carry only a handful of rows -- a raw
per-commit breakdown would be unreadable and each group hopelessly
underpowered. This module buckets by ISO CALENDAR WEEK of `trade_date`
instead: weeks are already contiguous and time-ordered, need no extra
lookup, and in practice group several commits' worth of rows together in
this codebase's actual commit cadence (multiple commits/day some days, per
project memory's CHANGELOG), giving readable, non-degenerate per-bucket
samples without inventing a new grouping key or joining to a commit-log
table this project doesn't otherwise query. Rows with `git_commit IS NULL`
are pulled into their own single bucket (label `(no git_commit recorded)`)
rather than dropped or folded into whichever week they happen to fall in --
a NULL commit is itself informative (pre-tagging era) and blending it into
a week bucket with tagged rows would blur exactly the "which era" question
this analysis exists to answer.

Both analyses are read-only: no INSERT/UPDATE/DELETE anywhere in this
module. Neither renders a verdict -- the decision rule (e.g. "95% CI
excludes 0") belongs to a human reading project memory
(verify-every-fix-same-day.md), not to this tool. A null or reversed result
relative to the framing above is a valid, useful outcome and must be
reported as such, not adjusted to match an expected story."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

from app.audit.guard_alpha import _permutation_p_value
from app.audit.pending_recompute import (
    REASON_NOT_MATURED,
    holding_days_stats,
    recompute_pending_alphas,
)
from app.backtest import factor_metrics
from app.clock import local_today
from app.config import settings
from app.memory.db import get_pool
from app.user_context import get_current_user_id

_NO_COMMIT_LABEL = "(no git_commit recorded)"


def _validate_filters(
    market: str | None, scope: str | None, account_mode: str | None, basis: str,
) -> tuple[str, str]:
    if market not in (None, "domestic", "overseas"):
        raise ValueError(f"market must be 'domestic', 'overseas', or None, got {market!r}")
    if scope not in (None, "live", "backtest"):
        raise ValueError(f"scope must be 'live', 'backtest', or None, got {scope!r}")
    if account_mode not in (None, "paper", "real"):
        raise ValueError(f"account_mode must be 'paper', 'real', or None, got {account_mode!r}")
    if basis not in ("variable", "horizon"):
        raise ValueError(f"basis must be 'variable' or 'horizon', got {basis!r}")
    return ("alpha_return", "raw_return") if basis == "variable" else (
        "horizon_alpha_return", "horizon_raw_return",
    )


def _build_where(
    market: str | None, scope: str | None, account_mode: str | None,
    since: date | None, until: date | None, *, status: str = "resolved",
) -> tuple[str, list]:
    # `status` is interpolated into SQL, so it is closed to two literals; the
    # default reproduces the pre-2026-09-28 string exactly.
    if status not in ("resolved", "pending"):
        raise ValueError(f"status must be 'resolved' or 'pending', got {status!r}")
    conditions = [f"status = '{status}'", "rating IN ('Buy', 'Hold')"]
    params: list = []
    if market == "domestic":
        conditions.append("ticker LIKE '%.KS'")
    elif market == "overseas":
        conditions.append("ticker NOT LIKE '%.KS'")
    if scope == "live":
        conditions.append("is_backtest = false")
    elif scope == "backtest":
        conditions.append("is_backtest = true")
    if account_mode is not None:
        params.append(account_mode)
        conditions.append(f"account_mode = ${len(params)}")
    if since is not None:
        params.append(since)
        conditions.append(f"horizon_eval_date >= ${len(params)}")
    if until is not None:
        params.append(until)
        conditions.append(f"horizon_eval_date <= ${len(params)}")
    # Same analysts_failed exclusion as compute_hit_rate_by_rating/guard_alpha.
    conditions.append("(analysts_failed IS NULL OR analysts_failed = 0)")
    params.append(get_current_user_id())
    conditions.append(f"user_id = ${len(params)}")
    return " AND ".join(conditions), params


async def _fetch_rows(
    *, market: str | None, scope: str | None, account_mode: str | None, basis: str,
    since: date | None, until: date | None, with_git_commit: bool, conn=None,
) -> list:
    alpha_col, _raw_col = _validate_filters(market, scope, account_mode, basis)
    where_clause, params = _build_where(market, scope, account_mode, since, until)
    extra_cols = ", git_commit, created_at" if with_git_commit else ""
    query = rf"""
        WITH latest AS (
            SELECT DISTINCT ON (ticker, trade_date)
                trade_date, ticker, rating, {alpha_col} AS alpha_return,
                held_at_decision, held_at_decision_inferred{extra_cols}
            FROM memory_entries
            WHERE {where_clause}
            ORDER BY ticker, trade_date, created_at DESC
        )
        SELECT * FROM latest
        WHERE alpha_return IS NOT NULL AND alpha_return <> 'NaN'::float8
        ORDER BY trade_date, ticker
    """
    if conn is not None:
        return await conn.fetch(query, *params)
    pool = await get_pool()
    async with pool.acquire() as acquired:
        return await acquired.fetch(query, *params)


def _classify_hold(row) -> bool | None:
    """Returns True (flat), False (held), or None (unknown -- neither
    `held_at_decision` nor `held_at_decision_inferred` is set), using the
    exact same COALESCE-observed-then-inferred convention as
    `app.audit.stats.compute_hit_rate_by_rating`. Only meaningful for
    rating='Hold' rows; callers must not call this on a Buy row."""
    observed = row["held_at_decision"]
    value = observed if observed is not None else row["held_at_decision_inferred"]
    if value is None:
        return None
    return not value  # held_at_decision True means HELD, i.e. not flat


# ---------------------------------------------------------------------------
# Analysis (1): same-day paired Buy vs Hold(flat)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EntryPairedAlphaReport:
    basis: str
    market: str | None
    scope: str | None
    account_mode: str | None
    n_buy_rows: int  # every qualifying Buy row, paired day or not
    n_hold_flat_rows: int  # every qualifying Hold(flat) row, paired day or not
    n_excluded_unknown_held: int  # Hold rows where held/flat could not be determined
    n_paired_days: int  # days with >=1 Buy AND >=1 Hold(flat) row
    n_buy_only_days: int  # days with Buy rows but no Hold(flat) row (informational)
    n_hold_only_days: int  # days with Hold(flat) rows but no Buy row (informational)
    mean_daily_delta: float | None  # mean over paired days of (mean Buy alpha - mean Hold-flat alpha)
    ci95: tuple[float, float] | None  # block-bootstrap 95% CI on mean_daily_delta
    p_value: float | None  # within-day permutation p-value
    start: date | None
    end: date | None


async def compute_entry_paired_alpha(
    market: str | None = None, scope: str | None = None, account_mode: str | None = None,
    basis: str = "variable", since: date | None = None, until: date | None = None,
) -> EntryPairedAlphaReport:
    """See module docstring for the full design. `market`/`scope`/
    `account_mode`/`basis`/`since`/`until` follow
    `app.audit.stats.compute_hit_rate_by_rating`'s exact semantics."""
    rows = await _fetch_rows(
        market=market, scope=scope, account_mode=account_mode, basis=basis,
        since=since, until=until, with_git_commit=False,
    )
    return _build_paired_report(rows, basis=basis, market=market, scope=scope, account_mode=account_mode)


def _build_paired_report(
    rows, *, basis: str, market: str | None, scope: str | None, account_mode: str | None,
) -> EntryPairedAlphaReport:
    buy_by_day: dict[date, list[float]] = {}
    hold_flat_by_day: dict[date, list[float]] = {}
    n_excluded_unknown_held = 0
    dates: set[date] = set()

    for row in rows:
        trade_date = row["trade_date"]
        dates.add(trade_date)
        alpha = row["alpha_return"]
        if row["rating"] == "Buy":
            buy_by_day.setdefault(trade_date, []).append(alpha)
            continue
        # rating == 'Hold'
        is_flat = _classify_hold(row)
        if is_flat is None:
            n_excluded_unknown_held += 1
            continue
        if is_flat:
            hold_flat_by_day.setdefault(trade_date, []).append(alpha)
        # is_flat is False (held): not part of this comparison, not "excluded"

    n_buy_rows = sum(len(v) for v in buy_by_day.values())
    n_hold_flat_rows = sum(len(v) for v in hold_flat_by_day.values())

    paired_days = sorted(set(buy_by_day) & set(hold_flat_by_day))
    n_buy_only_days = len(set(buy_by_day) - set(hold_flat_by_day))
    n_hold_only_days = len(set(hold_flat_by_day) - set(buy_by_day))

    mean_daily_delta: float | None = None
    ci95: tuple[float, float] | None = None
    p_value: float | None = None

    if paired_days:
        day_buy_lists = [buy_by_day[d] for d in paired_days]
        day_hold_lists = [hold_flat_by_day[d] for d in paired_days]
        deltas = [
            (sum(v) / len(v)) - (sum(c) / len(c))
            for v, c in zip(day_buy_lists, day_hold_lists)
        ]
        mean_daily_delta = sum(deltas) / len(deltas)
        blocks = [[d] for d in deltas]
        ci95 = factor_metrics.block_bootstrap_ci(blocks, seed=0)
        p_value = _permutation_p_value(day_buy_lists, day_hold_lists, observed=mean_daily_delta)

    sorted_dates = sorted(dates)
    return EntryPairedAlphaReport(
        basis=basis, market=market, scope=scope, account_mode=account_mode,
        n_buy_rows=n_buy_rows, n_hold_flat_rows=n_hold_flat_rows,
        n_excluded_unknown_held=n_excluded_unknown_held,
        n_paired_days=len(paired_days), n_buy_only_days=n_buy_only_days, n_hold_only_days=n_hold_only_days,
        mean_daily_delta=mean_daily_delta, ci95=ci95, p_value=p_value,
        start=sorted_dates[0] if sorted_dates else None, end=sorted_dates[-1] if sorted_dates else None,
    )


def render_entry_paired_alpha(report: EntryPairedAlphaReport) -> str:
    scope_desc = f"market={report.market or '(both)'}  scope={report.scope or '(both)'}  " \
        f"account_mode={report.account_mode or '(both)'}  basis={report.basis}"
    lines = [
        "# Same-day paired Buy vs Hold(flat) alpha (Analysis 1)",
        scope_desc,
        f"range: {report.start}..{report.end}",
        "",
        f"n_buy_rows={report.n_buy_rows}  n_hold_flat_rows={report.n_hold_flat_rows}  "
        f"n_excluded_unknown_held={report.n_excluded_unknown_held}",
        f"n_paired_days={report.n_paired_days}  "
        f"(days with Buy only: {report.n_buy_only_days}, days with Hold(flat) only: {report.n_hold_only_days})",
        "",
    ]
    if report.n_paired_days == 0:
        lines.append(
            "No days had BOTH a Buy row and a Hold(flat) row -- the paired comparison cannot be "
            "computed from this sample. This is a valid, reportable outcome (not an error)."
        )
        return "\n".join(lines)

    ci_str = (
        f"[{report.ci95[0]:+.4%}, {report.ci95[1]:+.4%}]" if report.ci95 is not None else "n/a (<3 paired days)"
    )
    p_str = f"{report.p_value:.4f}" if report.p_value is not None else "n/a"
    lines += [
        f"mean daily delta (mean Buy alpha - mean Hold(flat) alpha): {report.mean_daily_delta:+.4%}",
        f"95% CI (block bootstrap, seed=0): {ci_str}",
        f"permutation p-value (within-day label permutation, seed=0): {p_str}",
        "",
        "A NEGATIVE delta means the bought names underperformed the avoided names on the same day(s) --",
        "the same direction as the confounded aggregate comparison, but now with the day's market/sector",
        "shock and screening universe held constant. Both cohorts are scored on RAW (not sign-flipped)",
        "alpha_return -- see module docstring for why. No verdict is printed here: read the 95% CI and",
        "p-value against the decision rule in project memory, not against any threshold in this output.",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Analysis (1) with --recompute-pending: sensitivity check, audit-side only
# ---------------------------------------------------------------------------
#
# Real-account KOSDAQ names stored as '.KS' can stay `status='pending'` forever
# because the production resolver's yfinance lookup fails on '.KS' (see
# `app.audit.pending_recompute`). This mode re-measures those rows IN MEMORY
# (no DB write, no reflection) and reports the baseline (resolved only) next
# to resolved + recomputed, so the size of the selection bias is visible.
# The baseline is the number to cite; the recomputed one is a sensitivity
# check (its eval-date rule can differ from production's by a day or so).


@dataclass(frozen=True)
class EntryPairedRecomputeReport:
    baseline: EntryPairedAlphaReport
    combined: EntryPairedAlphaReport
    eval_rule: str
    n_pending_past_cutoff: int  # pending rows past the holding cutoff, BEFORE resolved-sibling shadowing
    n_shadowed_by_resolved: int  # dropped: a resolved sibling already exists for the key
    n_recomputed_rows: int
    n_recomputed_buy: int
    n_recomputed_hold_flat: int
    n_recomputed_hold_held: int
    n_recomputed_hold_unknown: int
    n_recompute_failed: int  # price_unavailable + benchmark_unavailable + no_entry_close
    n_not_matured: int  # reported separately: not a failure, just too early
    failures_by_reason: dict[str, int]
    failed_tickers: list[str]
    n_alt_symbol_tickers: int  # DISTINCT TICKERS (incl. not_matured/failed rows) fetched via the alternate suffix
    n_tickers_fetched: int
    n_recomputed_alt_rows: int  # ROW count: recomputed rows whose prices came from the alternate suffix
    holding_days_min: int | None
    holding_days_median: float | None
    holding_days_max: int | None


async def _fetch_pending_rows(conn, *, market: str, account_mode: str | None, cutoff: date) -> list:
    """Live pending Buy/Hold rows old enough to have resolved (`trade_date <=
    cutoff`), latest-created per (ticker, trade_date)."""
    where_clause, params = _build_where(market, "live", account_mode, None, None, status="pending")
    params.append(cutoff)
    query = rf"""
        WITH latest AS (
            SELECT DISTINCT ON (ticker, trade_date)
                trade_date, ticker, rating, held_at_decision, held_at_decision_inferred, created_at
            FROM memory_entries
            WHERE {where_clause} AND trade_date <= ${len(params)}
            ORDER BY ticker, trade_date, created_at DESC
        )
        SELECT * FROM latest
        ORDER BY trade_date, ticker
    """
    return await conn.fetch(query, *params)


async def _fetch_resolved_keys(conn, *, market: str, account_mode: str | None) -> set[tuple[str, date]]:
    """Every (ticker, trade_date) with a resolved sibling under the same
    filters -- taken BEFORE the NaN-alpha filter, so a resolved row that fell
    out of the baseline still shadows its pending twin."""
    where_clause, params = _build_where(market, "live", account_mode, None, None)
    query = f"SELECT DISTINCT ticker, trade_date FROM memory_entries WHERE {where_clause}"
    return {(r["ticker"], r["trade_date"]) for r in await conn.fetch(query, *params)}


async def compute_entry_paired_alpha_with_recompute(
    market: str | None = None, scope: str | None = None, account_mode: str | None = None,
    basis: str = "variable",
) -> EntryPairedRecomputeReport:
    """Domestic/live only. Baseline, pending and resolved-key reads share one
    read-only REPEATABLE READ snapshot so a concurrently running production
    resolver cannot put a row on both sides."""
    _validate_filters(market, scope, account_mode, basis)
    if market != "domestic":
        raise ValueError("--recompute-pending is domestic-only (overseas has different cutoff/session/benchmark)")
    if scope != "live":
        # The pending and resolved-key reads are live-only; a baseline that also pooled backtest rows
        # could not shadow a live pending twin and could double count the same key.
        raise ValueError("--recompute-pending requires scope='live'")
    if account_mode is None:
        # The production resolver only touches rows of the CURRENT account mode, so pooling modes would
        # pull in pending rows stalled by a mode switch (a different cause than the KOSDAQ suffix issue).
        raise ValueError("--recompute-pending requires an explicit account_mode ('paper' or 'real')")

    cutoff = local_today("domestic") - timedelta(days=settings.memory_min_holding_days)
    pool = await get_pool()
    async with pool.acquire() as conn:
        async with conn.transaction(isolation="repeatable_read", readonly=True):
            rows = await _fetch_rows(
                market=market, scope=scope, account_mode=account_mode, basis=basis,
                since=None, until=None, with_git_commit=False, conn=conn,
            )
            pending = await _fetch_pending_rows(conn, market=market, account_mode=account_mode, cutoff=cutoff)
            resolved_keys = await _fetch_resolved_keys(conn, market=market, account_mode=account_mode)

    eligible = [p for p in pending if (p["ticker"], p["trade_date"]) not in resolved_keys]
    n_shadowed = len(pending) - len(eligible)
    result = await recompute_pending_alphas(eligible, basis=basis, today=local_today("domestic"))

    report_rows = [r.as_report_row() for r in result.rows]
    combined_rows = sorted([*rows, *report_rows], key=lambda r: (r["trade_date"], r["ticker"]))

    n_buy = sum(1 for r in report_rows if r["rating"] == "Buy")
    n_flat = n_held = n_unknown = 0
    for r in report_rows:
        if r["rating"] == "Buy":
            continue
        is_flat = _classify_hold(r)
        if is_flat is None:
            n_unknown += 1
        elif is_flat:
            n_flat += 1
        else:
            n_held += 1

    by_reason: dict[str, int] = {}
    for failure in result.failures:
        by_reason[failure.reason] = by_reason.get(failure.reason, 0) + 1
    hard_failures = [f for f in result.failures if f.reason != REASON_NOT_MATURED]
    h_min, h_median, h_max = holding_days_stats(result.rows)
    eval_rule = (
        f"first settled common close on/after trade_date+{settings.memory_min_holding_days}d "
        "(production resolves at the LATEST common close)"
        if basis == "variable"
        else f"{settings.memory_fixed_horizon_trading_days}th trading day (same as production horizon)"
    )
    return EntryPairedRecomputeReport(
        baseline=_build_paired_report(rows, basis=basis, market=market, scope=scope, account_mode=account_mode),
        combined=_build_paired_report(
            combined_rows, basis=basis, market=market, scope=scope, account_mode=account_mode,
        ),
        eval_rule=eval_rule,
        n_pending_past_cutoff=len(pending), n_shadowed_by_resolved=n_shadowed,
        n_recomputed_rows=len(result.rows), n_recomputed_buy=n_buy,
        n_recomputed_hold_flat=n_flat, n_recomputed_hold_held=n_held, n_recomputed_hold_unknown=n_unknown,
        n_recompute_failed=len(hard_failures), n_not_matured=by_reason.get(REASON_NOT_MATURED, 0),
        failures_by_reason=by_reason,
        failed_tickers=sorted({f.ticker for f in hard_failures}),
        n_alt_symbol_tickers=result.n_alt_symbol_tickers, n_tickers_fetched=result.n_tickers_fetched,
        n_recomputed_alt_rows=sum(1 for r in result.rows if r.price_symbol != r.ticker),
        holding_days_min=h_min, holding_days_median=h_median, holding_days_max=h_max,
    )


def _fmt_delta(report: EntryPairedAlphaReport) -> str:
    return f"{report.mean_daily_delta:+.4%}" if report.mean_daily_delta is not None else "n/a"


def _fmt_ci(report: EntryPairedAlphaReport) -> str:
    return f"[{report.ci95[0]:+.4%}, {report.ci95[1]:+.4%}]" if report.ci95 is not None else "n/a"


def _fmt_p(report: EntryPairedAlphaReport) -> str:
    return f"{report.p_value:.4f}" if report.p_value is not None else "n/a"


def render_entry_paired_recompute(r: EntryPairedRecomputeReport) -> str:
    b, c = r.baseline, r.combined
    lines = [
        "# Same-day paired Buy vs Hold(flat) alpha -- baseline vs recomputed pending (Analysis 1)",
        f"market={b.market}  scope={b.scope or '(both)'}  account_mode={b.account_mode or '(both)'}  basis={b.basis}",
        "",
        f"{'':28}{'baseline(resolved only)':>26}{'recomputed(resolved+recomputed)':>34}",
        f"{'n_buy_rows':28}{b.n_buy_rows:>26}{c.n_buy_rows:>34}",
        f"{'n_hold_flat_rows':28}{b.n_hold_flat_rows:>26}{c.n_hold_flat_rows:>34}",
        f"{'n_excluded_unknown_held':28}{b.n_excluded_unknown_held:>26}{c.n_excluded_unknown_held:>34}",
        f"{'n_paired_days':28}{b.n_paired_days:>26}{c.n_paired_days:>34}",
        f"{'mean_daily_delta':28}{_fmt_delta(b):>26}{_fmt_delta(c):>34}",
        f"{'95% CI (block bootstrap)':28}{_fmt_ci(b):>26}{_fmt_ci(c):>34}",
        f"{'permutation p':28}{_fmt_p(b):>26}{_fmt_p(c):>34}",
        "",
        "## recomputed rows (label: recomputed)",
        f"eval_rule={r.eval_rule}",
        f"n_pending_past_cutoff={r.n_pending_past_cutoff}  n_shadowed_by_resolved={r.n_shadowed_by_resolved}",
        f"n_recomputed_rows={r.n_recomputed_rows}  n_recomputed_buy={r.n_recomputed_buy}  "
        f"n_recomputed_hold_flat={r.n_recomputed_hold_flat}  n_recomputed_hold_held={r.n_recomputed_hold_held}  "
        f"n_recomputed_hold_unknown={r.n_recomputed_hold_unknown}",
        f"n_recompute_failed={r.n_recompute_failed}  n_not_matured={r.n_not_matured}",
        f"alt-symbol: tickers {r.n_alt_symbol_tickers}/{r.n_tickers_fetched}, "
        f"recomputed rows {r.n_recomputed_alt_rows}/{r.n_recomputed_rows}",
        "failures_by_reason="
        + (", ".join(f"{k}={v}" for k, v in sorted(r.failures_by_reason.items())) or "(none)"),
        f"holding_days min/median/max={r.holding_days_min}/{r.holding_days_median}/{r.holding_days_max}",
    ]
    if r.failed_tickers:
        lines.append("failed_tickers=" + ", ".join(r.failed_tickers))
    lines += [
        "",
        "## baseline",
        render_entry_paired_alpha(b),
        "",
        "## recomputed",
        render_entry_paired_alpha(c),
        "",
        "NOTE: recomputed rows exist only in memory: not written to DB, no reflection generated. Eval rule",
        "differs from production's latest-close fallback (see eval_rule). Cite the baseline for decisions;",
        "read the recomputed column as a sensitivity check only. Reasons other than KOSDAQ suffix problems",
        "(e.g. rows too recent to have resolved yet) also land here -- see failures_by_reason. To judge whether the",
        "suffix is the cause, compare recomputed alt ROWS with recomputed rows (both row counts): recomputed rows",
        "whose prices came from the stored '.KS' symbol were stalled by something else. The ticker counts in the",
        "alt-symbol line include not_matured/failed rows and are not comparable to row counts. A symbol that",
        "returns SOME bars under '.KS' but lacks the entry-day close is reported as no_entry_close (two",
        "symbols' series are never mixed).",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Analysis (3): code-version stratification of the same cohorts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VersionBucketStats:
    label: str  # ISO week label "YYYY-Www", or _NO_COMMIT_LABEL
    start_date: date | None
    end_date: date | None
    commits: list[str] = field(default_factory=list)  # distinct commits, first-seen order
    n_buy: int = 0
    mean_buy_alpha: float | None = None
    n_hold_flat: int = 0
    mean_hold_flat_alpha: float | None = None
    naive_delta: float | None = None  # mean_buy - mean_hold_flat, NOT day-paired (see render note)


@dataclass(frozen=True)
class CodeVersionStratificationReport:
    basis: str
    market: str | None
    scope: str | None
    account_mode: str | None
    n_excluded_unknown_held: int
    buckets: list[VersionBucketStats] = field(default_factory=list)


def _iso_week_label(d: date) -> str:
    iso_year, iso_week, _ = d.isocalendar()
    return f"{iso_year}-W{iso_week:02d}"


async def compute_code_version_stratification(
    market: str | None = None, scope: str | None = None, account_mode: str | None = None,
    basis: str = "variable", since: date | None = None, until: date | None = None,
) -> CodeVersionStratificationReport:
    """Buckets the same Buy / Hold(flat) cohorts as `compute_entry_paired_
    alpha` by ISO calendar week of `trade_date` (NULL `git_commit` rows get
    their own bucket instead) -- see module docstring for why this
    bucketing was chosen over a raw per-commit breakdown. `market`/`scope`/
    `account_mode`/`basis`/`since`/`until` follow `compute_entry_paired_
    alpha`'s exact semantics."""
    rows = await _fetch_rows(
        market=market, scope=scope, account_mode=account_mode, basis=basis,
        since=since, until=until, with_git_commit=True,
    )
    return _build_version_report(rows, basis=basis, market=market, scope=scope, account_mode=account_mode)


def _build_version_report(
    rows, *, basis: str, market: str | None, scope: str | None, account_mode: str | None,
) -> CodeVersionStratificationReport:
    # bucket_key -> list of (trade_date, commit, rating_bucket, alpha)
    # rating_bucket is "buy" or "hold_flat"; held rows are skipped, unknown
    # held rows are counted separately, exactly as in Analysis (1).
    entries: dict[str, list[tuple]] = {}
    n_excluded_unknown_held = 0

    for row in rows:
        trade_date = row["trade_date"]
        alpha = row["alpha_return"]
        commit = row["git_commit"]
        if row["rating"] == "Buy":
            bucket_kind = "buy"
        else:
            is_flat = _classify_hold(row)
            if is_flat is None:
                n_excluded_unknown_held += 1
                continue
            if not is_flat:
                continue  # held, not part of this comparison
            bucket_kind = "hold_flat"

        label = _NO_COMMIT_LABEL if commit is None else _iso_week_label(trade_date)
        entries.setdefault(label, []).append((trade_date, commit, bucket_kind, alpha))

    buckets: list[VersionBucketStats] = []
    # NULL-commit bucket first (module docstring: reported as its own bucket,
    # not dropped or blended into a week), then week buckets chronologically.
    ordered_labels = sorted(
        entries.keys(),
        key=lambda label: (label != _NO_COMMIT_LABEL, label),
    )
    for label in ordered_labels:
        items = entries[label]
        dates_in_bucket = [d for d, _c, _k, _a in items]
        # Distinct commits in first-seen (trade_date, then encounter) order.
        seen_commits: list[str] = []
        for d, c, _k, _a in sorted(items, key=lambda item: item[0]):
            if c is not None and c not in seen_commits:
                seen_commits.append(c)
        buy_alphas = [a for _d, _c, k, a in items if k == "buy"]
        hold_flat_alphas = [a for _d, _c, k, a in items if k == "hold_flat"]
        mean_buy = sum(buy_alphas) / len(buy_alphas) if buy_alphas else None
        mean_hold_flat = sum(hold_flat_alphas) / len(hold_flat_alphas) if hold_flat_alphas else None
        naive_delta = mean_buy - mean_hold_flat if mean_buy is not None and mean_hold_flat is not None else None
        buckets.append(VersionBucketStats(
            label=label,
            start_date=min(dates_in_bucket) if dates_in_bucket else None,
            end_date=max(dates_in_bucket) if dates_in_bucket else None,
            commits=seen_commits,
            n_buy=len(buy_alphas), mean_buy_alpha=mean_buy,
            n_hold_flat=len(hold_flat_alphas), mean_hold_flat_alpha=mean_hold_flat,
            naive_delta=naive_delta,
        ))

    return CodeVersionStratificationReport(
        basis=basis, market=market, scope=scope, account_mode=account_mode,
        n_excluded_unknown_held=n_excluded_unknown_held, buckets=buckets,
    )


def render_code_version_stratification(report: CodeVersionStratificationReport) -> str:
    lines = [
        "# Code-version stratification of Buy vs Hold(flat) alpha (Analysis 3)",
        f"market={report.market or '(both)'}  scope={report.scope or '(both)'}  "
        f"account_mode={report.account_mode or '(both)'}  basis={report.basis}",
        f"n_excluded_unknown_held={report.n_excluded_unknown_held}",
        "",
        "Buckets are ISO calendar weeks of trade_date (NULL git_commit rows get their own bucket) --",
        "see module docstring for why a raw per-commit breakdown was rejected as unreadable/underpowered.",
        "",
    ]
    if not report.buckets:
        lines.append("No qualifying rows in this sample.")
        return "\n".join(lines)

    label_width = max(12, max(len(b.label) for b in report.buckets))
    header = (
        f"{'bucket':<{label_width}} {'range':<23} {'n_buy':>6} {'mean_buy':>10} "
        f"{'n_hflat':>8} {'mean_hflat':>11} {'naive_delta':>12}  commits"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for b in report.buckets:
        date_range = f"{b.start_date}..{b.end_date}" if b.start_date else "n/a"
        mean_buy = f"{b.mean_buy_alpha:+.2%}" if b.mean_buy_alpha is not None else "-"
        mean_hflat = f"{b.mean_hold_flat_alpha:+.2%}" if b.mean_hold_flat_alpha is not None else "-"
        naive_delta = f"{b.naive_delta:+.2%}" if b.naive_delta is not None else "-"
        commits_str = ", ".join(c[:8] for c in b.commits) if b.commits else "(none)"
        lines.append(
            f"{b.label:<{label_width}} {date_range:<23} {b.n_buy:>6} {mean_buy:>10} "
            f"{b.n_hold_flat:>8} {mean_hflat:>11} {naive_delta:>12}  {commits_str}"
        )
    lines += [
        "",
        "naive_delta is mean_buy_alpha - mean_hold_flat_alpha WITHIN the bucket, pooled across every day",
        "in it -- NOT the day-paired statistic Analysis 1 computes (`entry-paired`). It is a coarse",
        "stability check only: does the sign/magnitude look similar bucket to bucket, or is it",
        "concentrated in one era? Do not treat naive_delta as a significance-tested figure.",
    ]
    return "\n".join(lines)
