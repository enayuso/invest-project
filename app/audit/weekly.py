"""Weekly self-audit report (Stage 2, 2026-08-13) — a bounded, recurring
snapshot on top of `app.audit.stats`'s all-time/on-demand queries. Answers
"how did this week go, is that in line with the last 8 weeks, and how does
that compare with everything we've ever recorded" in one place, so a person
doesn't have to re-derive that by hand from `python -m app.audit report`
week after week.

This module is READ-ONLY reporting: it never writes to `memory_entries`,
never touches `app.config.settings` or any `.env.local` value, and there is
no code path here that could feed back into sizing/prompts/thresholds. A
person reads the output and decides whether any of those config values
should change by hand (see this stage's own design note on why: sizing_*/
trailing_stop_*/screening_* parameters are `app.config.Settings` fields
requiring a process restart, and prompts are hardcoded Python string
constants across several modules — nothing here is wired to either, on
purpose).

2026-09-08 (fee/tax/slippage audit): the report now also runs
`app.audit.slippage_validation.compute_slippage` for the current week —
the only piece of this module that does real network I/O (a KIS execution-
history fetch), since that's the only way to get a real-fill-vs-ledger-price
comparison. Best-effort like everything else here: a broker-unavailable/
KIS-error condition degrades to a "not available this run" note in the
slippage section rather than failing the whole report — `python -m app.audit
slippage` (the standalone tool this reuses, unchanged) remains the way to
get the full per-trade detail and a wider date range on demand.

Week boundary is Monday 00:00 -> Sunday 23:59:59, in KST (`app.clock.
local_today`'s own convention for "a market's own calendar day" extended to
"a person's own calendar week" — this project's operators are KST-based
regardless of which market a given decision concerns) — NOT the KST/ET split
`app.clock.local_today` uses per-market, since a week boundary is about when
a HUMAN reads the report, not when a market session opens.
"""

from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from app.audit.slippage_validation import SlippageResult, compute_slippage
from app.audit.stats import (
    ConfidenceCalibration,
    RatingStats,
    compute_confidence_calibration,
    compute_hit_rate_by_rating,
)
from app.clock import get_clock
from app.memory.db import get_pool
from app.user_context import get_current_user_id

logger = logging.getLogger(__name__)

_KR_TZ = ZoneInfo("Asia/Seoul")

# Below this many scorable rows, a ratio (win rate, changed%) is rendered as
# "—(n=X)" instead of a percentage — a percentage computed from a handful of
# rows reads as far more confident than it is. Applies to every ratio-style
# figure in this report (not to N counts themselves, which are always shown).
_MIN_N_FOR_RATIO = 10

# F7 panel specifically: below this many rows in a CELL, no win/loss verdict
# wording is printed at all (not even a percentage) — only the count. Higher
# than `_MIN_N_FOR_RATIO` on purpose: F7 slices an already-small population
# (Hold outcomes where the Trader wanted to trade) down further by held/flat,
# so its cells are more sample-starved than a top-level rating row of the
# same N would be, and this is the exact table meant to catch "the debate
# quietly guts every directional call" — a false read here is costlier than
# in the general report.
_MIN_N_FOR_F7_VERDICT = 30


def _week_bounds(anchor: date) -> tuple[date, date]:
    """Monday..Sunday (inclusive) of the KST calendar week containing
    `anchor` — `date.weekday()` is ISO (Monday=0), so this needs no
    timezone math beyond `anchor` already being a KST calendar date."""
    monday = anchor - timedelta(days=anchor.weekday())
    sunday = monday + timedelta(days=6)
    return monday, sunday


def week_label(anchor: date) -> str:
    """'YYYY-Www' ISO week label for a date, used as the report filename
    stem and its own displayed title."""
    iso_year, iso_week, _ = anchor.isocalendar()
    return f"{iso_year}-W{iso_week:02d}"


def parse_week_label(label: str) -> date:
    """Inverse of `week_label` — 'YYYY-Www' -> the Monday of that ISO week.
    Raises ValueError on a malformed label (surfaced as an argparse error by
    the CLI, not swallowed)."""
    year_str, _, week_str = label.partition("-W")
    if not year_str or not week_str:
        raise ValueError(f"expected 'YYYY-Www' (e.g. '2026-W33'), got {label!r}")
    return date.fromisocalendar(int(year_str), int(week_str), 1)


@dataclass
class RatingRow:
    """One rating's stats across the three comparison windows — any window
    missing that rating entirely (no resolved rows) is None, not a zeroed
    RatingStats, so the renderer can tell "0 rows" from "not asked"."""
    rating: str
    this_week: RatingStats | None
    last_8_weeks: RatingStats | None
    cumulative: RatingStats | None


@dataclass
class F7Cell:
    row_label: str  # 'Hold (held)' / 'Hold (flat)' / 'Hold (held/flat unknown)'
    n: int
    wins: int
    win_rate: float | None  # None whenever n < _MIN_N_FOR_F7_VERDICT (or n == 0)
    # How many of `n` came from held_at_decision_inferred rather than the
    # directly-observed column — see app.audit.stats.RatingStats.inferred_n
    # for the same convention.
    inferred_n: int = 0


@dataclass
class F7Panel:
    cells: list[F7Cell]
    total_overridden: int  # sum of n across cells: resolved+scorable overrides


@dataclass
class WeeklyReport:
    week_start: date
    week_end: date
    scope: str
    account_mode: str
    trigger: str | None
    basis: str
    generated_at: datetime
    this_week_n: int
    this_week_unscorable_n: int
    last_8_weeks_n: int
    cumulative_n: int
    ratings: list[RatingRow]
    confidence: ConfidenceCalibration  # cumulative only — see module docstring item (d)
    f7: F7Panel
    # This week's ledger-price-vs-KIS-fill-price comparison (2026-09-08) —
    # None when the underlying compute_slippage call failed (no broker
    # configured for this account_mode, a KIS error, etc.), NOT when it ran
    # and simply found nothing to compare (that's an empty-but-real
    # SlippageResult, matched == []). See slippage_error for what went wrong.
    slippage: SlippageResult | None = None
    slippage_error: str | None = None


async def _hit_rate_map(
    *, scope: str, account_mode: str, trigger: str | None, basis: str, since: date | None, until: date | None,
) -> dict[str, RatingStats]:
    rows = await compute_hit_rate_by_rating(
        market=None, scope=scope, account_mode=account_mode, trigger=trigger, basis=basis, since=since, until=until,
    )
    return {r.rating: r for r in rows}


async def _compute_f7_panel(
    *, scope: str, account_mode: str, trigger: str | None = "auto", basis: str,
) -> F7Panel:
    """Cross-tab of Hold(held)/Hold(flat)/Hold(unknown) outcomes, restricted
    to decisions where the TRADER wanted to actually trade
    (`trader_action <> 'Hold'`) but the final rating came out Hold anyway —
    i.e. the risk debate / Portfolio Manager overrode a directional call
    down to Hold. Deliberately computed over the FULL history (no since/
    until — same reasoning as `compute_confidence_calibration`'s cumulative-
    only use in `compute_weekly_report`: a single week's worth of overridden
    Holds is almost never enough rows for `_MIN_N_FOR_F7_VERDICT` to ever
    fire, so a weekly-windowed version would just print '— (n=...)' every
    week and never actually answer the question F7 exists to answer."""
    if scope not in ("live", "backtest"):
        raise ValueError(f"scope must be 'live' or 'backtest', got {scope!r}")
    if account_mode not in ("paper", "real"):
        raise ValueError(f"account_mode must be 'paper' or 'real', got {account_mode!r}")
    if trigger not in (None, "auto", "manual"):
        raise ValueError(f"trigger must be 'auto', 'manual', or None, got {trigger!r}")
    if basis not in ("variable", "horizon"):
        raise ValueError(f"basis must be 'variable' or 'horizon', got {basis!r}")
    alpha_col = "alpha_return" if basis == "variable" else "horizon_alpha_return"
    raw_col = "raw_return" if basis == "variable" else "horizon_raw_return"
    # Fixed mapping, never interpolates caller input directly — no injection
    # surface, same argument as the alpha_col/raw_col pair above.
    trigger_sql = {
        None: "",
        "auto": " AND decision_trigger IS DISTINCT FROM 'manual'",
        "manual": " AND decision_trigger = 'manual'",
    }[trigger]

    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            f"""
            WITH latest AS (
                SELECT DISTINCT ON (ticker, trade_date)
                    rating, trader_action, {alpha_col} AS alpha_return,
                    {raw_col} AS raw_return,
                    COALESCE(held_at_decision, held_at_decision_inferred) AS held_effective,
                    held_at_decision IS NULL AND held_at_decision_inferred IS NOT NULL AS is_inferred
                FROM memory_entries
                WHERE status = 'resolved'
                    AND is_backtest = $1
                    AND account_mode = $2
                    AND trader_action IS NOT NULL
                    AND trader_action <> 'Hold'
                    AND rating = 'Hold'
                    AND user_id = $3
                    {trigger_sql}
                ORDER BY ticker, trade_date, created_at DESC
            ), bucketed AS (
                SELECT
                    CASE
                        WHEN held_effective IS TRUE THEN 'Hold (held)'
                        WHEN held_effective IS FALSE THEN 'Hold (flat)'
                        ELSE 'Hold (held/flat unknown)'
                    END AS bucket,
                    alpha_return,
                    raw_return,
                    held_effective,
                    is_inferred,
                    -- Mirrors compute_hit_rate_by_rating's `ok` filter: must
                    -- also require raw_return to be present/non-NaN, since
                    -- Hold(held) is now scored on raw_return, not alpha_return
                    -- (see the `wins` CASE below).
                    alpha_return IS NOT NULL AND alpha_return <> 'NaN'::float8
                        AND raw_return IS NOT NULL AND raw_return <> 'NaN'::float8 AS ok
                FROM latest
            )
            SELECT
                bucket,
                COUNT(*) FILTER (WHERE ok) AS n,
                COUNT(*) FILTER (WHERE ok AND is_inferred) AS inferred_n,
                COUNT(*) FILTER (WHERE ok AND CASE
                    WHEN held_effective IS TRUE THEN raw_return > 0
                    WHEN held_effective IS FALSE THEN alpha_return < 0
                    ELSE NULL
                END) AS wins
            FROM bucketed
            GROUP BY bucket
            """,
            scope == "backtest",
            account_mode,
            get_current_user_id(),
        )

    by_bucket = {row["bucket"]: row for row in rows}
    cells = []
    for label in ("Hold (held)", "Hold (flat)", "Hold (held/flat unknown)"):
        row = by_bucket.get(label)
        n = row["n"] if row else 0
        wins = row["wins"] if row and row["wins"] is not None else 0
        inferred_n = row["inferred_n"] if row and row["inferred_n"] is not None else 0
        win_rate = (
            wins / n
            if row and n >= _MIN_N_FOR_F7_VERDICT and label != "Hold (held/flat unknown)"
            else None
        )
        cells.append(F7Cell(row_label=label, n=n, wins=wins, win_rate=win_rate, inferred_n=inferred_n))

    return F7Panel(cells=cells, total_overridden=sum(c.n for c in cells))


async def compute_weekly_report(
    week: date | None = None, scope: str = "live", account_mode: str = "paper",
    trigger: str | None = "auto", basis: str = "variable",
) -> WeeklyReport:
    """`week`: any date inside the target ISO week (default: today in KST).
    `scope`/`account_mode`: unlike `app.audit.stats`'s own functions, these
    default to "live"/"paper" (not None/pooled) — a recurring weekly report
    that silently blended live+backtest or paper+real every run would be a
    much easier mistake to miss than a one-off `python -m app.audit report`
    invocation, so this picks the safer default instead of matching that
    module's own opt-in convention. Pass explicitly to see another slice.

    `trigger` ("auto", the default, "manual", or None for pooled — 2026-08-20):
    same reasoning as `scope`/`account_mode` above — a recurring report that
    silently blended auto-triggered and sample-selection-biased manual
    one-off analyses would be an easy mistake to miss, so this defaults to
    "auto" rather than `app.audit.stats`'s own None-pooled convention. See
    `compute_hit_rate_by_rating`'s own docstring for what "auto" matches
    (includes NULL — rows predating the decision_trigger column).

    `basis`: which return columns to SCORE against (`compute_hit_rate_by_
    rating`'s own `basis` parameter) — the time WINDOW (this week / last 8
    weeks / cumulative) is always anchored on `horizon_eval_date` regardless
    of this, see that function's own docstring."""
    if scope not in ("live", "backtest"):
        raise ValueError(f"scope must be 'live' or 'backtest', got {scope!r}")
    if account_mode not in ("paper", "real"):
        raise ValueError(f"account_mode must be 'paper' or 'real', got {account_mode!r}")
    if trigger not in (None, "auto", "manual"):
        raise ValueError(f"trigger must be 'auto', 'manual', or None, got {trigger!r}")
    if basis not in ("variable", "horizon"):
        raise ValueError(f"basis must be 'variable' or 'horizon', got {basis!r}")

    anchor = week if week is not None else get_clock().now(_KR_TZ).date()
    week_start, week_end = _week_bounds(anchor)
    eight_weeks_start = week_start - timedelta(weeks=7)  # 8 KST weeks total, inclusive of this week

    this_week_map, last_8_weeks_map, cumulative_map = [
        await _hit_rate_map(scope=scope, account_mode=account_mode, trigger=trigger, basis=basis, since=s, until=u)
        for s, u in ((week_start, week_end), (eight_weeks_start, week_end), (None, None))
    ]

    all_ratings = sorted(set(this_week_map) | set(last_8_weeks_map) | set(cumulative_map))
    ratings = [
        RatingRow(
            rating=r,
            this_week=this_week_map.get(r),
            last_8_weeks=last_8_weeks_map.get(r),
            cumulative=cumulative_map.get(r),
        )
        for r in all_ratings
    ]

    confidence = await compute_confidence_calibration(
        market=None, scope=scope, account_mode=account_mode, trigger=trigger, basis=basis,
    )
    f7 = await _compute_f7_panel(scope=scope, account_mode=account_mode, trigger=trigger, basis=basis)

    # Best-effort, real network I/O (KIS execution history) — see module
    # docstring. Never lets a broker/KIS problem take down the rest of this
    # otherwise DB-only report.
    slippage: SlippageResult | None = None
    slippage_error: str | None = None
    try:
        slippage = await compute_slippage(start=week_start, end=week_end, account_mode=account_mode)
    except Exception as exc:
        logger.warning("weekly report: slippage section unavailable this run", exc_info=True)
        slippage_error = str(exc) or exc.__class__.__name__

    return WeeklyReport(
        week_start=week_start,
        week_end=week_end,
        scope=scope,
        account_mode=account_mode,
        trigger=trigger,
        basis=basis,
        generated_at=get_clock().now(_KR_TZ),
        this_week_n=sum(s.n for s in this_week_map.values()),
        this_week_unscorable_n=sum(s.unscorable_n for s in this_week_map.values()),
        last_8_weeks_n=sum(s.n for s in last_8_weeks_map.values()),
        cumulative_n=sum(s.n for s in cumulative_map.values()),
        ratings=ratings,
        confidence=confidence,
        f7=f7,
        slippage=slippage,
        slippage_error=slippage_error,
    )


def _fmt_ratio(win_rate: float | None, n: int) -> str:
    if win_rate is None:
        return f"—(n={n})" if n else "—"
    if n < _MIN_N_FOR_RATIO:
        return f"—(n={n})"
    return f"{win_rate:.1%}"


def _fmt_stats_cell(s: RatingStats | None) -> str:
    if s is None:
        return f"{'N/A':>6} {'—':>10}"
    return f"{s.n:>6} {_fmt_ratio(s.win_rate, s.n):>10}"


def render_weekly_report(report: WeeklyReport) -> str:
    lines = [
        f"# Weekly Self-Audit — {week_label(report.week_start)} "
        f"({report.week_start.isoformat()} ~ {report.week_end.isoformat()}, KST)",
        f"scope={report.scope}  account_mode={report.account_mode}  trigger={report.trigger or 'pooled'}  "
        f"basis={report.basis}",
        f"generated: {report.generated_at:%Y-%m-%d %H:%M:%S} KST",
        "",
    ]

    # (b) N and unscorable/unknown BEFORE any ratio, per the design note.
    lines.append(
        f"This week: N={report.this_week_n} (unscorable={report.this_week_unscorable_n})   "
        f"Last 8 weeks: N={report.last_8_weeks_n}   Cumulative: N={report.cumulative_n}"
    )
    lines.append("")

    if not report.ratings:
        lines.append("No resolved decisions in any window yet — nothing to compare.")
    else:
        lines.append(
            f"{'Rating':<26} {'This week':>17} {'Last 8 weeks':>17} {'Cumulative':>17}"
        )
        lines.append(f"{'':<26} {'N':>6} {'Win rate':>10} {'N':>6} {'Win rate':>10} {'N':>6} {'Win rate':>10}")
        lines.append("-" * 96)
        for row in report.ratings:
            lines.append(
                f"{row.rating:<26} "
                f"{_fmt_stats_cell(row.this_week)}  "
                f"{_fmt_stats_cell(row.last_8_weeks)}  "
                f"{_fmt_stats_cell(row.cumulative)}"
            )
        lines.append("")
        lines.append(
            "Win rate is direction-aware (Buy/Overweight win on alpha>0, Sell/Underweight/Hold(flat) win "
            "on alpha<0 — see app.audit.stats._WIN_DIRECTION). Hold(held) wins on raw_return>0, not alpha "
            "(2026-08-20) — this system is long-only and cannot rotate into the index, so the real "
            "alternative to keeping a position is cash, not the benchmark (see _RAW_RETURN_RATINGS). A "
            "'Buy (gated)'/'Sell (gated)' row is a Hold the confidence-based hysteresis gate downgraded — "
            "scored on the model's ORIGINAL call like ungated Buy/Sell, not as a Hold (see "
            "app.orchestrator.pipeline._apply_hysteresis_gate). "
            f"Cells below n={_MIN_N_FOR_RATIO} show '—(n=X)' instead of a percentage — too few rows to read "
            "as a rate."
        )

    # F7 panel
    lines += ["", "## F7 — directional calls overridden down to Hold", ""]
    if report.f7.total_overridden == 0:
        lines.append(
            "No resolved decisions found where the Trader proposed a trade (trader_action != Hold) but "
            "the final rating came out Hold."
        )
    else:
        lines.append(
            f"Cumulative (all-time), Trader wanted to trade but final rating was Hold — {report.f7.total_overridden} row(s):"
        )
        lines.append(f"{'Held/flat at decision':<28} {'N':>6} {'Result':>20}")
        lines.append("-" * 56)
        for cell in report.f7.cells:
            if cell.n == 0:
                result = "n=0"
            elif cell.win_rate is not None:
                result = f"{cell.win_rate:.1%} (n={cell.n})"
            else:
                result = f"n={cell.n} (< {_MIN_N_FOR_F7_VERDICT}, no verdict)"
            inferred_note = f" ({cell.inferred_n} inferred)" if cell.inferred_n else ""
            lines.append(f"{cell.row_label:<28} {cell.n:>6} {result:>20}{inferred_note}")
        lines.append("")
        lines.append(
            f"Cells with n < {_MIN_N_FOR_F7_VERDICT} intentionally show NO win/loss verdict, only the "
            "count — this table exists to catch a systematic override problem, and a false read here "
            "(from a handful of rows) is worse than none."
        )
        lines.append(
            "'Hold (held/flat unknown)' rows never get a verdict (held_at_decision was NULL at decision "
            "time — direction isn't determinable) regardless of N."
        )

    # (d) Confidence calibration — cumulative only, per the design note: a
    # week's worth of rows is too small to bucket by 5 confidence bands and
    # still say anything.
    lines += ["", "## Confidence calibration (cumulative only — weekly sample too small to bucket)", ""]
    if report.confidence.n_with_confidence == 0:
        lines.append(
            f"No decisions with a recorded confidence yet ({report.confidence.n_without_confidence} "
            "resolved row(s) predate the confidence column)."
        )
    else:
        lines.append(f"{'Confidence':<16} {'N':>6} {'Win rate':>10}")
        for b in report.confidence.buckets:
            lines.append(f"{b.label:<16} {b.n:>6} {_fmt_ratio(b.win_rate, b.n):>10}")
        lines.append(f"Excluded — no confidence recorded: {report.confidence.n_without_confidence}")

    # Slippage (2026-09-08) — condensed: the full per-trade detail and a
    # wider/custom date range stay on `python -m app.audit slippage`
    # (app.audit.slippage_validation.render_slippage), this is just enough
    # to notice a week going badly without running that separately.
    lines += _render_weekly_slippage_section(report)

    # (e) variable vs horizon footnote.
    lines += [
        "",
        "NOTE: this week / last-8-weeks / cumulative windows are always anchored on horizon_eval_date "
        "(the fixed 10-trading-day mark) regardless of --basis — a row with horizon_eval_date IS NULL "
        "(not yet matured) can never appear in any of the three windows above, even under basis=variable "
        "where its alpha_return may already be populated. basis only controls which return COLUMNS are "
        "scored (variable-window alpha_return/raw_return vs the fixed-window horizon_alpha_return/"
        "horizon_raw_return) — see app.audit.stats.compute_hit_rate_by_rating's own docstring.",
    ]
    return "\n".join(lines)


def _render_weekly_slippage_section(report: WeeklyReport) -> list[str]:
    """Condensed slippage summary for `render_weekly_report` — same-day-
    matched rows only (delayed/threshold/unusable rows are still counted in
    the header line so they don't just vanish, but their own detail stays on
    `python -m app.audit slippage`). Mirrors
    `app.audit.slippage_validation.render_slippage`'s own same_day filter
    and stats so the two never quietly compute a different number for the
    same underlying rows."""
    lines = ["", f"## Slippage — {week_label(report.week_start)} (ledger price vs actual KIS fill)", ""]
    if report.slippage is None:
        lines.append(
            f"Not available this run ({report.slippage_error or 'unknown error'}) — "
            "try `python -m app.audit slippage` directly."
        )
        return lines

    r = report.slippage
    same_day = [row for row in r.matched if not row.price_is_threshold and row.fill_delay_days == 0]
    if not same_day:
        lines.append(
            "No same-day matched trades this week (ledger row + KIS fill both present) — "
            f"ledger-only={len(r.ledger_only)}, KIS-only={len(r.kis_only)}."
        )
        return lines

    values = [row.slippage_pct for row in same_day]
    mean = sum(values) / len(values)
    unfavorable = sum(1 for v in values if v > 0)
    lines.append(
        f"{len(same_day)} matched trade(s) — mean {mean:+.3%}, median {statistics.median(values):+.3%}, "
        f"worst {max(values):+.3%}, unfavorable {unfavorable}/{len(same_day)} ({unfavorable / len(same_day):.1%})"
    )
    lines.append(
        "(positive = filled worse than the recorded ledger price). "
        "Full per-trade detail: `python -m app.audit slippage`."
    )
    return lines
