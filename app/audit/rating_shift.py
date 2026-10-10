"""Research Manager recommendation-distribution audit (2026-09-17): does the
RM's OWN recommendation (not the PM's final `rating` column) look different
before vs. after a given timestamp?

Immediate trigger: commit `a6ac168` (2026-09-16) gave the RM a rating
%-scale note and real position_context for the first time, and a
before/after eyeball comparison suggested the rating mix shifted hard toward
Hold. This module makes that comparison a durable, re-runnable command
instead of a throwaway `scratchpad/` script (one was already lost once —
see project memory) — it is also a standing health metric ("what is the AI
actually deciding, and has that shifted"), not a one-off check for this
single incident. It deliberately encodes NO pass/fail threshold: the
decision rule belongs to a human reading the numbers, not to this tool.

CRITICAL TRAP this module exists to avoid: `full_transcript->'research_plan'`
is a rendered MARKDOWN STRING on live rows (e.g. `'**Recommendation**:
Hold\n\n**Rationale**: ...'`), not a structured dict. Treating it as a dict
unconditionally (`research_plan["recommendation"]`) silently returns nothing
for every live row and produces a misleading all-`None` result — this
happened once already (2026-09-17). Both shapes are handled explicitly; the
"otherwise" case (missing key, unparseable string, unexpected type) is
counted as `n_unknown` and reported, never silently dropped.

Public API:
  `parse_boundary`      -> datetime (tz-aware)
  `compute_rating_shift` -> RatingShiftReport
  `render_rating_shift`  -> str

Row selection reuses the same `market`/`scope`/`account_mode` filter
semantics as `app.audit.stats.compute_confidence_calibration` /
`app.audit.guard_alpha.compute_guard_alpha` (ticker LIKE/NOT LIKE '%.KS' for
market, `is_backtest` for scope, `account_mode` column, `user_id` scoping
via `get_current_user_id`) — see those functions' docstrings for the shared
rationale, not repeated here. Deliberate difference: no `status = 'resolved'`
filter and no `trigger` filter — this audit is about what the RM decided,
not about resolved trading outcomes, so unresolved/recent rows (which
dominate any "after" cohort right after a deploy) must not be excluded.

Read-only: no INSERT/UPDATE/DELETE anywhere in this module."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone

from app.audit.guard_alpha import _parse_transcript
from app.memory.db import get_pool
from app.user_context import get_current_user_id

_RECOMMENDATION_RE = re.compile(r"\*\*Recommendation\*\*:\s*([A-Za-z]+)")


@dataclass(frozen=True)
class CohortRatingCounts:
    label: str
    n: int
    counts: dict[str, int] = field(default_factory=dict)
    n_unknown: int = 0


@dataclass(frozen=True)
class RatingShiftReport:
    boundary: datetime
    market: str | None
    scope: str | None
    account_mode: str | None
    before: CohortRatingCounts
    after: CohortRatingCounts


def parse_boundary(value: str) -> datetime:
    """Parses an ISO8601 datetime CLI argument, accepting a trailing 'Z' (not
    handled by `datetime.fromisoformat` on all supported Python versions) and
    defaulting to UTC when no offset is given, since `memory_entries.
    created_at` is stored as `timestamptz` and a naive/aware comparison would
    otherwise raise."""
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _extract_rm_recommendation(transcript: dict) -> tuple[str | None, bool]:
    """Returns `(recommendation, parsed)`. `parsed=False` means the row could
    not be classified (missing `research_plan`, an unparseable markdown
    string, or an unexpected type) and must be counted as unknown, not
    dropped or silently folded into any rating bucket."""
    research_plan = transcript.get("research_plan")
    if isinstance(research_plan, str):
        match = _RECOMMENDATION_RE.search(research_plan)
        if match:
            return match.group(1), True
        return None, False
    if isinstance(research_plan, dict):
        recommendation = research_plan.get("recommendation")
        if isinstance(recommendation, str) and recommendation:
            return recommendation, True
        return None, False
    return None, False


def _build_cohort(rows: list, label: str) -> CohortRatingCounts:
    counts: dict[str, int] = {}
    n_unknown = 0
    for row in rows:
        transcript = _parse_transcript(row["full_transcript"])
        recommendation, parsed = _extract_rm_recommendation(transcript)
        if not parsed:
            n_unknown += 1
            continue
        counts[recommendation] = counts.get(recommendation, 0) + 1
    return CohortRatingCounts(label=label, n=len(rows), counts=counts, n_unknown=n_unknown)


async def compute_rating_shift(
    boundary: datetime, *, market: str | None = None, scope: str | None = None,
    account_mode: str | None = None,
) -> RatingShiftReport:
    """Splits `memory_entries` into two cohorts at `boundary` (a tz-aware
    datetime, compared against `created_at`) and reports the Research
    Manager's own recommendation distribution in each. `market`/`scope`/
    `account_mode` follow `compute_guard_alpha`'s exact semantics (module
    docstring). Encodes no verdict/threshold — reports the distribution
    only."""
    if boundary.tzinfo is None:
        raise ValueError("boundary must be tz-aware — use parse_boundary()")
    if market not in (None, "domestic", "overseas"):
        raise ValueError(f"market must be 'domestic', 'overseas', or None, got {market!r}")
    if scope not in (None, "live", "backtest"):
        raise ValueError(f"scope must be 'live', 'backtest', or None, got {scope!r}")
    if account_mode not in (None, "paper", "real"):
        raise ValueError(f"account_mode must be 'paper', 'real', or None, got {account_mode!r}")

    conditions = []
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
    params.append(get_current_user_id())
    conditions.append(f"user_id = ${len(params)}")
    where_clause = " AND ".join(conditions)

    query = f"""
        SELECT ticker, trade_date, created_at, full_transcript
        FROM memory_entries
        WHERE {where_clause}
        ORDER BY created_at
    """

    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(query, *params)

    before_rows = [row for row in rows if row["created_at"] < boundary]
    after_rows = [row for row in rows if row["created_at"] >= boundary]

    return RatingShiftReport(
        boundary=boundary, market=market, scope=scope, account_mode=account_mode,
        before=_build_cohort(before_rows, "before"), after=_build_cohort(after_rows, "after"),
    )


def render_rating_shift(report: RatingShiftReport) -> str:
    lines = [
        f"Research Manager recommendation distribution  boundary={report.boundary.isoformat()}",
        f"filters: market={report.market or '(both)'}  scope={report.scope or '(both)'}  "
        f"account_mode={report.account_mode or '(both)'}",
        "",
    ]
    for cohort in (report.before, report.after):
        lines.append(f"=== {cohort.label} boundary (n={cohort.n}) ===")
        if cohort.n == 0:
            lines.append("  (no rows)")
            lines.append("")
            continue
        for rating, count in sorted(cohort.counts.items(), key=lambda kv: (-kv[1], kv[0])):
            pct = count / cohort.n * 100
            lines.append(f"  {rating:<14} {count:>5}  {pct:5.1f}%")
        pct_unknown = cohort.n_unknown / cohort.n * 100
        lines.append(f"  {'(unparsed)':<14} {cohort.n_unknown:>5}  {pct_unknown:5.1f}%")
        lines.append("")
    lines.append(
        "Note: this reports the RM's own recommendation, not the PM's final `rating` column, "
        "and encodes no pass/fail threshold — read the shift against the decision criterion in "
        "project memory, not against anything printed here."
    )
    return "\n".join(lines)
