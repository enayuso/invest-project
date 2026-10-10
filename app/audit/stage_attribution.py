"""Stage attribution: which decision stage (Research Manager -> Trader -> Neutral
risk debator -> Portfolio Manager -> gates) ADDS or REMOVES discrimination,
measured as same-trade_date paired raw forward alpha (design:
handoff/knowledge/stage-attribution-design-2026-09-29.md).

WHY THIS EXISTS: the aggregate audits say the system's Buy names underperform
and its Hold names outperform, but they cannot say WHERE in the stack that
happens. This tool extracts each stage's own stance from `full_transcript`
and asks, per stage and per hand-off between two stages: does the stance
separate forward alpha within the same trade_date (Q1), and was each
override/promotion of an earlier stage's stance right on average (Q2)? Q3
projects how many more decision days are needed before any verdict is
possible.

SCOPE: one (market, account_mode) per run, live rows only (is_backtest=false),
auto-triggered by default, deduplicated to the latest row per (ticker,
trade_date). Only decisions that were FLAT at decision time are tested (held
rows have a different 5-level-vs-2-value schema; they are counted, not
tested). Both cohorts are scored on RAW alpha (never sign-flipped): this
compares two stock sets' forward returns to EACH OTHER, same as
`app.audit.entry_paired`. Stances are collapsed to bullish (Buy/Overweight)
vs not-bullish (Hold/Underweight/Sell): in a flat position RM/Trader/PM can
only emit Buy/Hold, so a 3-way split would be unmeasurable. Date filters apply
to trade_date, NOT horizon_eval_date (unlike entry-paired/guard-alpha).

PRE-REGISTERED VERDICT RULE (fixed before any directional result was seen;
do not tune it to a result). `_verdict` evaluates in this order:
  1. LOW_COVERAGE   -- the stage(s) involved were parsed on <90% of the scored
                       flat rows (`_MIN_STAGE_COVERAGE`).
  2. UNDERPOWERED   -- n_paired_days < 20 (`guard_alpha._MIN_DAYS_FOR_VERDICT`),
                       or either cohort has < 20 paired rows
                       (`guard_alpha._MIN_ROWS_PER_GROUP`), or q is None.
  3. FINDING(+/-)   -- Benjamini-Hochberg q < 0.05 AND the block-bootstrap CI
                       excludes 0 AND delta and the CI share a sign AND both
                       robustness deltas (`delta_ex_extreme`: rows with
                       |alpha| >= 0.30 removed; `delta_first_per_ticker`: only
                       each ticker's first row per cohort) keep delta's sign.
                       If the robustness deltas disagree: INCONCLUSIVE
                       (outlier/ticker-driven).
  4. INCONCLUSIVE   -- exactly one of {q < 0.05, CI excludes 0}.
  5. NO_EFFECT      -- q >= 0.05 and the CI contains 0; the CI half-width is
                       printed as the resolution ("no discrimination detected
                       at +/-x").
Supplementary: `material` = |delta| >= 0.005 (0.5 percentage points over the
10-trading-day horizon). THIS THRESHOLD IS AN UNVALIDATED ASSUMPTION (not
checked against measured round-trip cost); a non-material FINDING is printed
"FINDING (immaterial)". If the CI over consecutive 5-paired-day blocks
(`ci95_5day_blocks`, a guard against the overlapping 10-day windows) contains
0, the row is flagged "fragile: overlapping horizons" (a flag only).
A verdict is only issued at n_paired_days >= 20; below that every row is
UNDERPOWERED, its numbers are MASKED by default and revealed only with
--show-underpowered (prefixed PROVISIONAL). Underpowered rows never carry a
directional label.

INTERPRETATION (pure functions):
  Q1: FINDING(+) correct-direction discrimination; FINDING(-) REVERSED
      discrimination; NO_EFFECT "no discrimination detected at +/-x".
  Block transitions X->Y (X bullish, Y not): 1st contrast blocked - agreed_not
      (both not bullish): positive = Y discarded X's information. 2nd contrast
      blocked - passed (both bullish): negative = Y selected correctly inside
      X's bullish set. `interpret_block`: DESTROYS = 1st material FINDING(+)
      and 2nd not FINDING(-); STRICT_BUT_INFORMATIVE = 1st FINDING(+) and 2nd
      FINDING(-); ADDS_VALUE = 2nd material FINDING(-) and 1st not FINDING(+)
      (and the 1st is determinable); otherwise NOT_DETERMINED.
  Promotions (X not bullish, Y bullish) vs agreed_not: `interpret_promotion`
      material FINDING(+) = ADDS_VALUE, material FINDING(-) = DESTROYS.
  "Stage Y destroys value" is only claimed when the incoming block transition
  is DESTROYS AND coverage >= 90% AND (when --era-boundary is used) the
  transition is not ERA_HETEROGENEOUS.

CITING: to quote the CURRENT system, rerun with --start-date 2026-09-18 (a
09-17 deploy mixed several behavior changes) and require the same verdict. A
full-range FINDING is only citable as "past (since 09-01) pooled". Cite the
default run; --recompute-pending is a sensitivity check.

DELIBERATE NON-GOALS: `stats.py`/`guard_alpha.py` score a consensus-downgraded
row as Hold(flat); this tool does not touch or reconcile that. It never reads
`language_ab_report.pre_gate_rating` (experiment-record shaped); only its
regexes are reused.

Strictly read-only: SELECT inside a REPEATABLE READ READ ONLY transaction, no
INSERT/UPDATE/DELETE, no LLM, no `run_pipeline`. The only network use is
yfinance under `--recompute-pending` (reuses `app.audit.pending_recompute`).
"""

from __future__ import annotations

import math
import re
import statistics
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import date, datetime, timedelta

from app.agents.risk_debators import _neutral_verdict
from app.audit.entry_paired import _classify_hold
from app.audit.guard_alpha import (
    _MIN_DAYS_FOR_VERDICT,
    _MIN_ROWS_PER_GROUP,
    _apply_bh_correction,
    _parse_transcript,
    _permutation_p_value,
)
from app.audit.pending_recompute import (
    REASON_NOT_MATURED,
    holding_days_stats,
    recompute_pending_alphas,
)
from app.audit.rating_shift import _extract_rm_recommendation
from app.backtest import factor_metrics
from app.clock import local_today
from app.config import settings
from app.experiments.language_ab_report import (
    _CONSENSUS_MARKER,
    _CONSENSUS_NOTE_RE,
    _HYSTERESIS_MARKER,
    _HYSTERESIS_NOTE_RE,
)
from app.memory.db import get_pool
from app.user_context import get_current_user_id

_MIN_STAGE_COVERAGE = 0.90
_MATERIAL_DELTA = 0.005
_EXTREME_ALPHA = 0.30
_BLOCK_DAYS = 5
_ERA_MIN_PAIRED = 5
_Q_ALPHA = 0.05
_PROJECTION_WINDOW = 20

STAGES = ("RM", "Trader", "Neutral", "PM_pre", "Post_consensus", "Final")
Q1_STAGES = ("RM", "Trader", "Neutral", "PM_pre", "Final")
TRANSITIONS = (
    ("RM", "Trader"), ("Trader", "Neutral"), ("Neutral", "PM_pre"), ("PM_pre", "Final"), ("RM", "PM_pre"),
)

_LEVELS = ("Buy", "Overweight", "Hold", "Underweight", "Sell")
_BULLISH_LEVELS = frozenset({"Buy", "Overweight"})
_TRADER_ACTION_RE = re.compile(r"\*\*Action\*\*:\s*([A-Za-z]+)")
# decision_text carries only the short, number-free variant of the hysteresis note.
_HYSTERESIS_PREFIX_RE = re.compile(r"\[Hysteresis gate: downgraded from (\w+) to Hold")
_UNSET = object()


# ---------------------------------------------------------------------------
# Stage extraction (pure)
# ---------------------------------------------------------------------------


def normalize_level(value) -> str | None:
    """`.capitalize()` then accept only the five rating words; anything else None."""
    if not isinstance(value, str):
        return None
    level = value.strip().capitalize()
    return level if level in _LEVELS else None


def stance(value) -> str | None:
    """'bullish' (Buy/Overweight) / 'neutral' (Hold) / 'bearish' (Underweight/Sell) / None."""
    level = normalize_level(value)
    if level is None:
        return None
    if level in _BULLISH_LEVELS:
        return "bullish"
    return "neutral" if level == "Hold" else "bearish"


def _is_bullish(level: str | None) -> bool | None:
    """Bullish vs not-bullish; None when the stage label is unknown."""
    return None if level is None else level in _BULLISH_LEVELS


@dataclass(frozen=True)
class StageLabels:
    rm: str | None = None
    rm_parsed: bool = False
    trader: str | None = None
    trader_source: str | None = None  # "column" / "markdown" / None
    trader_mismatch: bool = False  # column and rendered markdown both present but differ
    neutral: str | None = None
    pm_pre: str | None = None
    post_consensus: str | None = None
    final: str | None = None
    gate_parse_failed: bool = False
    gate_record_note_mismatch: bool = False
    consensus_gated: bool = False

    def level(self, stage: str) -> str | None:
        return {
            "RM": self.rm, "Trader": self.trader, "Neutral": self.neutral, "PM_pre": self.pm_pre,
            "Post_consensus": self.post_consensus, "Final": self.final,
        }[stage]


def _col(row, name: str):
    try:
        return row[name]
    except KeyError:
        return None


def _extract_trader(row, transcript: dict) -> tuple[str | None, str | None, bool]:
    column = normalize_level(_col(row, "trader_action"))
    proposal = transcript.get("trader_proposal")
    markdown = None
    if isinstance(proposal, str):
        match = _TRADER_ACTION_RE.search(proposal)
        markdown = normalize_level(match.group(1)) if match else None
    if column is not None:
        return column, "column", markdown is not None and markdown != column
    if markdown is not None:
        return markdown, "markdown", False
    return None, None, False


def _extract_neutral(transcript: dict) -> str | None:
    debate = transcript.get("risk_debate")
    if not isinstance(debate, list):
        return None
    turns = [t for t in debate if isinstance(t, dict) and t.get("speaker") == "Neutral"]
    if not turns:
        return None
    last = max(enumerate(turns), key=lambda it: (it[1].get("round") or 0, it[0]))[1]
    verdict, _span = _neutral_verdict(str(last.get("content", "")))
    return normalize_level(verdict)


def _recover_pm_pre(row, transcript: dict) -> tuple[str | None, bool, bool, bool]:
    """(pm_pre, consensus_gated, parse_failed, record_note_mismatch), in the
    design's fixed order: structured consensus record, structured
    unjustified-exit record, hysteresis note, legacy consensus note, else the
    final rating. A marker that cannot be parsed (or both hysteresis and
    consensus markers at once) yields pm_pre=None, parse_failed=True."""
    thesis = transcript.get("investment_thesis")
    thesis = thesis if isinstance(thesis, str) else ""
    decision_text = _col(row, "decision_text")
    decision_text = decision_text if isinstance(decision_text, str) else ""

    consensus = transcript.get("consensus_override")
    if isinstance(consensus, dict) and consensus and consensus.get("gated") is True:
        pm = normalize_level(consensus.get("pm_rating"))
        note = _CONSENSUS_NOTE_RE.search(thesis)
        mismatch = bool(note) and normalize_level(note.group("orig")) != pm
        return pm, True, pm is None, mismatch
    unjustified = transcript.get("unjustified_exit_gate")
    if isinstance(unjustified, dict) and unjustified.get("gated") is True:
        return "Sell", False, False, False

    hyst_thesis = _HYSTERESIS_MARKER in thesis
    cons_thesis = _CONSENSUS_MARKER in thesis
    hyst_text = _HYSTERESIS_MARKER in decision_text
    if (hyst_thesis or hyst_text) and cons_thesis:
        return None, False, True, False
    if hyst_thesis:
        match = _HYSTERESIS_NOTE_RE.search(thesis)
        orig = normalize_level(match.group("orig")) if match else None
        return orig, False, orig is None, False
    if hyst_text:
        match = _HYSTERESIS_PREFIX_RE.search(decision_text)
        orig = normalize_level(match.group(1)) if match else None
        return orig, False, orig is None, False
    if cons_thesis:
        match = _CONSENSUS_NOTE_RE.search(thesis)
        orig = normalize_level(match.group("orig")) if match else None
        return orig, True, orig is None, False
    return normalize_level(_col(row, "rating")), False, False, False


def extract_stage_labels(row, transcript: dict) -> StageLabels:
    rm_raw, rm_parsed = _extract_rm_recommendation(transcript)
    rm = normalize_level(rm_raw) if rm_parsed else None
    trader, trader_source, trader_mismatch = _extract_trader(row, transcript)
    pm_pre, consensus_gated, parse_failed, mismatch = _recover_pm_pre(row, transcript)
    if parse_failed or pm_pre is None:
        post = None
    else:
        post = "Hold" if consensus_gated else pm_pre
    return StageLabels(
        rm=rm, rm_parsed=rm is not None, trader=trader, trader_source=trader_source,
        trader_mismatch=trader_mismatch, neutral=_extract_neutral(transcript), pm_pre=pm_pre,
        post_consensus=post, final=normalize_level(_col(row, "rating")),
        gate_parse_failed=parse_failed, gate_record_note_mismatch=mismatch, consensus_gated=consensus_gated,
    )


def _holding_state(row) -> str:
    """'flat' / 'held' / 'unknown', via entry_paired's COALESCE(observed, inferred)."""
    flat = _classify_hold(row)
    return "unknown" if flat is None else ("flat" if flat else "held")


# ---------------------------------------------------------------------------
# Rows
# ---------------------------------------------------------------------------


def _valid_alpha(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and not math.isnan(value)


@dataclass(frozen=True)
class AttributionRow:
    ticker: str
    trade_date: date
    created_at: datetime | None
    git_commit: str | None
    status: str | None
    alpha: float | None
    source: str  # "resolved" / "recomputed" / "pending"
    holding: str  # "flat" / "held" / "unknown"
    labels: StageLabels

    @property
    def scored(self) -> bool:
        return self.status == "resolved" and _valid_alpha(self.alpha)


def _to_attribution_row(row) -> AttributionRow:
    transcript = _parse_transcript(_col(row, "full_transcript"))
    status = _col(row, "status")
    source = _col(row, "source") or ("resolved" if status == "resolved" else "pending")
    return AttributionRow(
        ticker=row["ticker"], trade_date=row["trade_date"], created_at=_col(row, "created_at"),
        git_commit=_col(row, "git_commit"), status=status, alpha=_col(row, "alpha_return"), source=source,
        holding=_holding_state(row), labels=extract_stage_labels(row, transcript),
    )


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProjectionStats:
    elig_rate: float
    realization: float | None
    backlog: float
    extra_decision_days: int | None  # None = "never at current rate"
    n_eligible_days: int = 0


@dataclass(frozen=True)
class ContrastStats:
    name: str
    group: str = ""  # "Q1" / "Q2"
    stages: tuple[str, ...] = ()
    n_a: int = 0
    n_b: int = 0
    n_a_paired: int = 0
    n_b_paired: int = 0
    n_paired_days: int = 0
    n_one_sided_days: int = 0
    n_tickers: int = 0
    delta: float | None = None
    median_delta: float | None = None
    ci95: tuple[float, float] | None = None
    p_value: float | None = None
    q_value: float | None = None
    n_extreme: int = 0
    delta_ex_extreme: float | None = None
    delta_first_per_ticker: float | None = None
    ci95_5day_blocks: tuple[float, float] | None = None
    identical_to: str | None = None
    coverage: float = 1.0
    verdict: str = "NOT_TESTED"
    direction: str | None = None
    material: bool | None = None
    fragile: bool = False
    projection: ProjectionStats | None = None

    @property
    def underpowered_condition(self) -> bool:
        return (
            self.n_paired_days < _MIN_DAYS_FOR_VERDICT or self.n_a_paired < _MIN_ROWS_PER_GROUP
            or self.n_b_paired < _MIN_ROWS_PER_GROUP or self.q_value is None
        )


def _by_day(rows) -> dict[date, list[float]]:
    out: dict[date, list[float]] = {}
    for trade_date, _ticker, alpha in rows:
        out.setdefault(trade_date, []).append(alpha)
    return out


def _mean_delta(a_rows, b_rows) -> float | None:
    a_by, b_by = _by_day(a_rows), _by_day(b_rows)
    days = sorted(set(a_by) & set(b_by))
    if not days:
        return None
    deltas = [statistics.fmean(a_by[d]) - statistics.fmean(b_by[d]) for d in days]
    return sum(deltas) / len(deltas)


def _first_per_ticker(rows):
    first: dict[str, tuple] = {}
    for item in sorted(rows, key=lambda r: r[0]):
        first.setdefault(item[1], item)
    return list(first.values())


def _paired_contrast(name: str, a_rows, b_rows, *, group: str = "", stages: tuple[str, ...] = ()) -> ContrastStats:
    """`a_rows`/`b_rows`: iterables of (trade_date, ticker, alpha). Day-paired
    delta_d = mean(A alpha on d) - mean(B alpha on d), RAW alpha, over days
    with both cohorts; Delta = mean delta_d. Permutation p is within-day; q is
    filled in later by the caller (BH across the whole family)."""
    a_rows, b_rows = list(a_rows), list(b_rows)
    a_by, b_by = _by_day(a_rows), _by_day(b_rows)
    days = sorted(set(a_by) & set(b_by))
    base = dict(
        name=name, group=group, stages=stages, n_a=len(a_rows), n_b=len(b_rows),
        n_one_sided_days=len(set(a_by) ^ set(b_by)),
    )
    if not days:
        return ContrastStats(**base)
    a_lists = [a_by[d] for d in days]
    b_lists = [b_by[d] for d in days]
    deltas = [statistics.fmean(a) - statistics.fmean(b) for a, b in zip(a_lists, b_lists)]
    delta = sum(deltas) / len(deltas)
    day_set = set(days)
    paired_a = [r for r in a_rows if r[0] in day_set]
    paired_b = [r for r in b_rows if r[0] in day_set]

    def _trim(rows):
        return [r for r in rows if abs(r[2]) < _EXTREME_ALPHA]

    blocks = [deltas[i:i + _BLOCK_DAYS] for i in range(0, len(deltas) - _BLOCK_DAYS + 1, _BLOCK_DAYS)]
    return ContrastStats(
        **base,
        n_a_paired=len(paired_a), n_b_paired=len(paired_b), n_paired_days=len(days),
        n_tickers=len({r[1] for r in paired_a} | {r[1] for r in paired_b}),
        delta=delta, median_delta=statistics.median(deltas),
        ci95=factor_metrics.block_bootstrap_ci([[d] for d in deltas], seed=0),
        p_value=_permutation_p_value(a_lists, b_lists, observed=delta),
        n_extreme=sum(1 for r in paired_a + paired_b if abs(r[2]) >= _EXTREME_ALPHA),
        delta_ex_extreme=_mean_delta(_trim(paired_a), _trim(paired_b)),
        delta_first_per_ticker=_mean_delta(_first_per_ticker(paired_a), _first_per_ticker(paired_b)),
        ci95_5day_blocks=factor_metrics.block_bootstrap_ci(blocks, seed=0),
    )


@dataclass(frozen=True)
class VerdictResult:
    verdict: str
    direction: str | None = None  # "+" / "-" for FINDING
    material: bool | None = None
    fragile: bool = False
    reason: str = ""


def _sign(value: float | None) -> int:
    return 0 if value is None or value == 0 else (1 if value > 0 else -1)


def _verdict(
    *, n_paired_days: int, q: float | None, ci: tuple[float, float] | None, delta: float | None,
    n_a_paired: int | None = None, n_b_paired: int | None = None,
    delta_ex_extreme=_UNSET, delta_first_per_ticker=_UNSET, coverage: float = 1.0,
    ci95_5day_blocks: tuple[float, float] | None = None,
) -> VerdictResult:
    """The pre-registered rule (module docstring). `_UNSET` robustness values
    mean "not supplied, skip that check"; an explicit None means "could not be
    computed", which blocks a FINDING."""
    if coverage < _MIN_STAGE_COVERAGE:
        return VerdictResult("LOW_COVERAGE", reason=f"stage coverage {coverage:.0%} < {_MIN_STAGE_COVERAGE:.0%}")
    if (
        n_paired_days < _MIN_DAYS_FOR_VERDICT
        or (n_a_paired is not None and n_a_paired < _MIN_ROWS_PER_GROUP)
        or (n_b_paired is not None and n_b_paired < _MIN_ROWS_PER_GROUP)
        or q is None
    ):
        return VerdictResult("UNDERPOWERED")
    ci_excludes_zero = ci is not None and (ci[0] > 0 or ci[1] < 0)
    q_sig = q < _Q_ALPHA
    if q_sig and ci_excludes_zero:
        sign = _sign(delta)
        ci_sign = 1 if ci[0] > 0 else -1
        robust = all(
            v is _UNSET or _sign(v) == sign for v in (delta_ex_extreme, delta_first_per_ticker)
        )
        if sign != ci_sign:
            return VerdictResult("INCONCLUSIVE", reason="delta and CI disagree in sign")
        if not robust:
            return VerdictResult("INCONCLUSIVE", reason="outlier/ticker-driven (robustness deltas disagree)")
        fragile = ci95_5day_blocks is not None and ci95_5day_blocks[0] <= 0 <= ci95_5day_blocks[1]
        return VerdictResult(
            "FINDING", direction="+" if sign > 0 else "-", material=abs(delta) >= _MATERIAL_DELTA, fragile=fragile,
        )
    if q_sig or ci_excludes_zero:
        return VerdictResult("INCONCLUSIVE", reason="only one of q<0.05 / CI excludes 0")
    return VerdictResult("NO_EFFECT")


def _is_finding(c, direction: str, *, material: bool = False) -> bool:
    if c is None or c.verdict != "FINDING" or c.direction != direction:
        return False
    return bool(c.material) if material else True


def interpret_block(primary, secondary) -> str:
    """Block-transition interpretation (blocked-agreed_not, blocked-passed)."""
    if primary is None or primary.verdict in ("UNDERPOWERED", "LOW_COVERAGE", "NOT_TESTED"):
        return "NOT_DETERMINED"
    second_neg = _is_finding(secondary, "-")
    if _is_finding(primary, "+"):
        if second_neg:
            return "STRICT_BUT_INFORMATIVE"
        if primary.material:
            return "DESTROYS"
        return "NOT_DETERMINED"
    if _is_finding(secondary, "-", material=True):
        return "ADDS_VALUE"
    return "NOT_DETERMINED"


def interpret_promotion(primary) -> str:
    if _is_finding(primary, "+", material=True):
        return "ADDS_VALUE"
    if _is_finding(primary, "-", material=True):
        return "DESTROYS"
    return "NOT_DETERMINED"


def interpret_q1(c: ContrastStats) -> str | None:
    if c.verdict == "FINDING":
        suffix = "" if c.material else " (immaterial)"
        if c.direction == "+":
            return "discriminates in the correct direction" + suffix
        return "discriminates in the REVERSED direction (bullish stance underperforms)" + suffix
    if c.verdict == "NO_EFFECT" and c.ci95 is not None:
        return f"no discrimination detected at +/-{(c.ci95[1] - c.ci95[0]) / 2:.2%}"
    return None


def _power_projection(
    *, decision_dates, eligible_dates, n_paired_days: int, max_scored_date: date | None,
    window: int = _PROJECTION_WINDOW,
) -> ProjectionStats:
    """Q3. elig_rate: share of the latest `window` decision trade_dates on which
    BOTH cohorts exist (decision time, ignoring maturation). realization: of the
    eligible dates up to the latest scored date, the share that actually became
    a paired scored day. backlog: eligible dates after the latest scored date,
    discounted by realization. No calendar date is projected: add the 10
    trading-day maturation lag yourself."""
    eligible = set(eligible_dates)
    recent = sorted(set(decision_dates))[-window:]
    elig_rate = sum(1 for d in recent if d in eligible) / len(recent) if recent else 0.0
    matured = [d for d in eligible if max_scored_date is not None and d <= max_scored_date]
    realization = min(1.0, n_paired_days / len(matured)) if matured else None
    after = [d for d in eligible if max_scored_date is None or d > max_scored_date]
    backlog = len(after) * realization if realization is not None else 0.0
    if elig_rate <= 0 or not realization:
        extra = None
    else:
        need = max(0.0, _MIN_DAYS_FOR_VERDICT - n_paired_days - backlog)
        extra = math.ceil(round(need / (elig_rate * realization), 9))
    return ProjectionStats(
        elig_rate=elig_rate, realization=realization, backlog=backlog, extra_decision_days=extra,
        n_eligible_days=len(eligible),
    )


# ---------------------------------------------------------------------------
# Cohort definitions
# ---------------------------------------------------------------------------


def _transition_cell(x_level: str | None, y_level: str | None) -> str | None:
    xb, yb = _is_bullish(x_level), _is_bullish(y_level)
    if xb is None or yb is None:
        return None
    if xb and not yb:
        return "blocked"
    if xb and yb:
        return "passed"
    return "promoted" if yb else "agreed_not"


@dataclass(frozen=True)
class _TestSpec:
    name: str
    group: str
    stages: tuple[str, ...]
    assign: Callable[[AttributionRow], str | None]  # -> "A" / "B" / None


def _q1_spec(stage: str) -> _TestSpec:
    def assign(row: AttributionRow) -> str | None:
        bullish = _is_bullish(row.labels.level(stage))
        return None if bullish is None else ("A" if bullish else "B")

    return _TestSpec(f"Q1 {stage}: bullish - not_bullish", "Q1", (stage,), assign)


def _q2_spec(x: str, y: str, a_cell: str, b_cell: str) -> _TestSpec:
    def assign(row: AttributionRow) -> str | None:
        cell = _transition_cell(row.labels.level(x), row.labels.level(y))
        return "A" if cell == a_cell else ("B" if cell == b_cell else None)

    return _TestSpec(f"Q2 {x}->{y}: {a_cell} - {b_cell}", "Q2", (x, y), assign)


def _all_specs() -> list[_TestSpec]:
    specs = [_q1_spec(s) for s in Q1_STAGES]
    for x, y in TRANSITIONS:
        specs.append(_q2_spec(x, y, "blocked", "agreed_not"))
        specs.append(_q2_spec(x, y, "blocked", "passed"))
        if (x, y) != ("PM_pre", "Final"):  # gates only downgrade: promotion is structurally impossible
            specs.append(_q2_spec(x, y, "promoted", "agreed_not"))
    return specs


def _cohort_rows(spec: _TestSpec, rows) -> tuple[list, list]:
    a, b = [], []
    for row in rows:
        side = spec.assign(row)
        if side == "A":
            a.append((row.trade_date, row.ticker, row.alpha))
        elif side == "B":
            b.append((row.trade_date, row.ticker, row.alpha))
    return a, b


def _stage_coverage(rows, stage: str) -> float:
    return sum(1 for r in rows if r.labels.level(stage) is not None) / len(rows) if rows else 0.0


def _identical_stage(rows, stage: str) -> str | None:
    """First EARLIER Q1 stage whose stance matches this one on every scored row."""
    if not rows:
        return None
    index = Q1_STAGES.index(stage)
    mine = [_is_bullish(r.labels.level(stage)) for r in rows]
    if all(v is None for v in mine):
        return None  # an entirely unobserved stage is "no data", not "identical"
    for earlier in Q1_STAGES[:index]:
        if mine == [_is_bullish(r.labels.level(earlier)) for r in rows]:
            return earlier
    return None


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StageSummary:
    stage: str
    counts: dict[str, int]
    n_unknown: int
    coverage_all_flat: float
    coverage_scored_flat: float


@dataclass(frozen=True)
class FlowRow:
    x: str
    y: str
    n_both_known: int
    n_x_bullish: int
    blocked: int
    passed: int
    agreed_not: int
    promoted: int
    exact_agreement: float | None


@dataclass(frozen=True)
class TransitionBlock:
    x: str
    y: str
    primary: ContrastStats | None  # blocked - agreed_not
    secondary: ContrastStats | None  # blocked - passed
    promoted: ContrastStats | None  # promoted - agreed_not; None = structurally n/a
    block_interpretation: str
    promotion_interpretation: str
    destroys_claim: bool


@dataclass(frozen=True)
class EraRow:
    name: str
    delta_before: float | None
    n_before: int
    delta_after: float | None
    n_after: int
    status: str  # "ok" / "n/a" / "ERA_HETEROGENEOUS"


@dataclass(frozen=True)
class StageAttributionReport:
    market: str
    account_mode: str
    trigger: str
    basis: str
    since: date | None
    until: date | None
    era_boundary: datetime | None
    n_rows: int
    n_flat: int
    n_held: int
    n_unknown_held: int
    n_pending_flat: int
    n_scored_flat: int
    n_scored_held: int
    n_scored_days: int
    n_decision_days: int
    scored_range: tuple[date, date] | None
    n_mixed_status_keys: int
    n_rm_unparsed: int
    n_trader_markdown: int
    n_trader_column_markdown_mismatch: int
    n_gate_parse_failed: int
    n_gate_record_note_mismatch: int
    stage_summaries: list[StageSummary] = field(default_factory=list)
    flow: list[FlowRow] = field(default_factory=list)
    q1: list[ContrastStats] = field(default_factory=list)
    transitions: list[TransitionBlock] = field(default_factory=list)
    n_bh_family: int = 0
    era: list[EraRow] = field(default_factory=list)
    era_commits: dict[str, list[str]] = field(default_factory=dict)

    @property
    def contrasts(self) -> list[ContrastStats]:
        out = list(self.q1)
        for block in self.transitions:
            out += [c for c in (block.primary, block.secondary, block.promoted) if c is not None]
        return out


def _era_rows(specs, scored_flat, boundary: datetime) -> list[EraRow]:
    before = [r for r in scored_flat if r.created_at is not None and r.created_at < boundary]
    after = [r for r in scored_flat if r.created_at is not None and r.created_at >= boundary]
    out = []
    for spec in specs:
        deltas, counts = [], []
        for part in (before, after):
            a, b = _cohort_rows(spec, part)
            deltas.append(_mean_delta(a, b))
            counts.append(len(set(d for d, _t, _a in a) & set(d for d, _t, _a in b)))
        (d0, d1), (n0, n1) = deltas, counts
        if n0 < _ERA_MIN_PAIRED or n1 < _ERA_MIN_PAIRED or d0 is None or d1 is None:
            status = "n/a"
        else:
            status = "ERA_HETEROGENEOUS" if _sign(d0) != _sign(d1) else "ok"
        out.append(EraRow(spec.name, d0, n0, d1, n1, status))
    return out


def _build_report(
    rows, *, market: str, account_mode: str, trigger: str | None = None, basis: str = "horizon",
    since: date | None = None, until: date | None = None, era_boundary: datetime | None = None,
    n_mixed_status_keys: int = 0,
) -> StageAttributionReport:
    attr = [_to_attribution_row(r) for r in rows]
    flat = [r for r in attr if r.holding == "flat"]
    scored_flat = [r for r in flat if r.scored]
    decision_dates = sorted({r.trade_date for r in flat})
    max_scored = max((r.trade_date for r in scored_flat), default=None)

    stage_summaries = []
    for stage in STAGES:
        counts: dict[str, int] = {}
        for r in flat:
            level = r.labels.level(stage)
            if level is not None:
                counts[level] = counts.get(level, 0) + 1
        stage_summaries.append(StageSummary(
            stage=stage, counts=counts, n_unknown=sum(1 for r in flat if r.labels.level(stage) is None),
            coverage_all_flat=_stage_coverage(flat, stage), coverage_scored_flat=_stage_coverage(scored_flat, stage),
        ))

    flow = []
    for x, y in TRANSITIONS:
        known = [r for r in flat if r.labels.level(x) is not None and r.labels.level(y) is not None]
        cells = [_transition_cell(r.labels.level(x), r.labels.level(y)) for r in known]
        flow.append(FlowRow(
            x=x, y=y, n_both_known=len(known), n_x_bullish=sum(1 for r in known if _is_bullish(r.labels.level(x))),
            blocked=cells.count("blocked"), passed=cells.count("passed"), agreed_not=cells.count("agreed_not"),
            promoted=cells.count("promoted"),
            exact_agreement=(
                sum(1 for r in known if r.labels.level(x) == r.labels.level(y)) / len(known) if known else None
            ),
        ))

    specs = _all_specs()
    raw: list[ContrastStats] = []
    for spec in specs:
        identical = _identical_stage(scored_flat, spec.stages[0]) if spec.group == "Q1" else None
        a_all, b_all = _cohort_rows(spec, flat)
        if identical is not None:
            a_s, b_s = _cohort_rows(spec, scored_flat)
            stats = ContrastStats(
                name=spec.name, group=spec.group, stages=spec.stages, n_a=len(a_s), n_b=len(b_s),
                identical_to=identical,
            )
        else:
            a_s, b_s = _cohort_rows(spec, scored_flat)
            stats = _paired_contrast(spec.name, a_s, b_s, group=spec.group, stages=spec.stages)
        eligible = {d for d in {r[0] for r in a_all} & {r[0] for r in b_all}}
        stats = replace(
            stats,
            coverage=min(_stage_coverage(scored_flat, s) for s in spec.stages),
            projection=_power_projection(
                decision_dates=decision_dates, eligible_dates=eligible, n_paired_days=stats.n_paired_days,
                max_scored_date=max_scored,
            ),
        )
        raw.append(stats)

    family_idx = [i for i, c in enumerate(raw) if c.p_value is not None and c.identical_to is None]
    family = [raw[i] for i in family_idx]
    _apply_bh_correction(family)
    for i, c in zip(family_idx, family):
        raw[i] = c
    judged = []
    for c in raw:
        if c.identical_to is not None:
            judged.append(c)
            continue
        v = _verdict(
            n_paired_days=c.n_paired_days, q=c.q_value, ci=c.ci95, delta=c.delta,
            n_a_paired=c.n_a_paired, n_b_paired=c.n_b_paired,
            delta_ex_extreme=c.delta_ex_extreme if c.n_paired_days else _UNSET,
            delta_first_per_ticker=c.delta_first_per_ticker if c.n_paired_days else _UNSET,
            coverage=c.coverage, ci95_5day_blocks=c.ci95_5day_blocks,
        )
        judged.append(replace(c, verdict=v.verdict, direction=v.direction, material=v.material, fragile=v.fragile))
    by_name = {c.name: c for c in judged}

    era: list[EraRow] = []
    era_commits: dict[str, list[str]] = {}
    era_status: dict[str, str] = {}
    if era_boundary is not None:
        era = _era_rows(specs, scored_flat, era_boundary)
        era_status = {e.name: e.status for e in era}
        for label, pick in (("before", lambda r: r.created_at < era_boundary), ("after", lambda r: r.created_at >= era_boundary)):
            commits: list[str] = []
            for r in sorted((r for r in flat if r.created_at is not None and pick(r)), key=lambda r: r.created_at):
                if r.git_commit and r.git_commit[:8] not in commits:
                    commits.append(r.git_commit[:8])
            era_commits[label] = commits

    q1 = [by_name[_q1_spec(s).name] for s in Q1_STAGES]
    blocks = []
    for x, y in TRANSITIONS:
        primary = by_name[_q2_spec(x, y, "blocked", "agreed_not").name]
        secondary = by_name[_q2_spec(x, y, "blocked", "passed").name]
        promoted = None if (x, y) == ("PM_pre", "Final") else by_name[_q2_spec(x, y, "promoted", "agreed_not").name]
        interp = interpret_block(primary, secondary)
        destroys = (
            interp == "DESTROYS" and primary.coverage >= _MIN_STAGE_COVERAGE
            and era_status.get(primary.name) != "ERA_HETEROGENEOUS"
        )
        blocks.append(TransitionBlock(
            x=x, y=y, primary=primary, secondary=secondary, promoted=promoted, block_interpretation=interp,
            promotion_interpretation=interpret_promotion(promoted) if promoted is not None else "n/a",
            destroys_claim=destroys,
        ))

    held = [r for r in attr if r.holding == "held"]
    return StageAttributionReport(
        market=market, account_mode=account_mode, trigger=trigger or "auto", basis=basis, since=since, until=until,
        era_boundary=era_boundary, n_rows=len(attr), n_flat=len(flat), n_held=len(held),
        n_unknown_held=sum(1 for r in attr if r.holding == "unknown"),
        n_pending_flat=sum(1 for r in flat if r.status == "pending"), n_scored_flat=len(scored_flat),
        n_scored_held=sum(1 for r in held if r.scored), n_scored_days=len({r.trade_date for r in scored_flat}),
        n_decision_days=len(decision_dates),
        scored_range=(min(r.trade_date for r in scored_flat), max_scored) if scored_flat else None,
        n_mixed_status_keys=n_mixed_status_keys,
        n_rm_unparsed=sum(1 for r in attr if not r.labels.rm_parsed),
        n_trader_markdown=sum(1 for r in attr if r.labels.trader_source == "markdown"),
        n_trader_column_markdown_mismatch=sum(1 for r in attr if r.labels.trader_mismatch),
        n_gate_parse_failed=sum(1 for r in attr if r.labels.gate_parse_failed),
        n_gate_record_note_mismatch=sum(1 for r in attr if r.labels.gate_record_note_mismatch),
        stage_summaries=stage_summaries, flow=flow, q1=q1, transitions=blocks, n_bh_family=len(family),
        era=era, era_commits=era_commits,
    )


# ---------------------------------------------------------------------------
# DB access (read-only)
# ---------------------------------------------------------------------------


def _validate(market, account_mode, trigger, basis) -> str:
    if market not in ("domestic", "overseas"):
        raise ValueError(f"market must be 'domestic' or 'overseas', got {market!r}")
    if account_mode not in ("paper", "real"):
        raise ValueError(f"account_mode must be 'paper' or 'real', got {account_mode!r}")
    if trigger not in (None, "auto", "manual"):
        raise ValueError(f"trigger must be 'auto', 'manual', or None, got {trigger!r}")
    if basis not in ("variable", "horizon"):
        raise ValueError(f"basis must be 'variable' or 'horizon', got {basis!r}")
    return "alpha_return" if basis == "variable" else "horizon_alpha_return"


def _build_where(
    market: str, account_mode: str, trigger: str | None, since: date | None, until: date | None,
) -> tuple[str, list]:
    conditions = ["is_backtest = false"]
    conditions.append("ticker LIKE '%.KS'" if market == "domestic" else "ticker NOT LIKE '%.KS'")
    params: list = [account_mode]
    conditions.append("account_mode = $1")
    conditions.append(
        "decision_trigger = 'manual'" if trigger == "manual" else "decision_trigger IS DISTINCT FROM 'manual'"
    )
    # trade_date, NOT horizon_eval_date: this audit does not depend on maturation.
    if since is not None:
        params.append(since)
        conditions.append(f"trade_date >= ${len(params)}")
    if until is not None:
        params.append(until)
        conditions.append(f"trade_date <= ${len(params)}")
    conditions.append("(analysts_failed IS NULL OR analysts_failed = 0)")
    params.append(get_current_user_id())
    conditions.append(f"user_id = ${len(params)}")
    return " AND ".join(conditions), params


async def _fetch_rows(conn, alpha_col: str, where: str, params: list) -> list:
    query = rf"""
        WITH latest AS (
            SELECT DISTINCT ON (ticker, trade_date)
                id, ticker, trade_date, created_at, git_commit, status, rating, trader_action,
                {alpha_col} AS alpha_return, held_at_decision, held_at_decision_inferred,
                full_transcript, decision_text
            FROM memory_entries
            WHERE {where}
            ORDER BY ticker, trade_date, created_at DESC
        )
        SELECT * FROM latest ORDER BY trade_date, ticker
    """
    return await conn.fetch(query, *params)


async def _fetch_mixed_status_count(conn, where: str, params: list) -> int:
    query = (
        "SELECT COUNT(*) AS n FROM ("
        f"SELECT 1 FROM memory_entries WHERE {where} "
        "GROUP BY ticker, trade_date HAVING COUNT(DISTINCT status) > 1) mixed"
    )
    rows = await conn.fetch(query, *params)
    return int(rows[0]["n"]) if rows else 0


async def _fetch_resolved_keys(conn, where: str, params: list) -> set[tuple[str, date]]:
    query = f"SELECT DISTINCT ticker, trade_date FROM memory_entries WHERE status = 'resolved' AND {where}"
    return {(r["ticker"], r["trade_date"]) for r in await conn.fetch(query, *params)}


async def compute_stage_attribution(
    *, market: str | None, account_mode: str | None, trigger: str | None = None, basis: str = "horizon",
    since: date | None = None, until: date | None = None, era_boundary: datetime | None = None,
) -> StageAttributionReport:
    """Live scope only. `market` and `account_mode` are REQUIRED (a silent
    default pooled paper+real once and caused the 09-28 mixing)."""
    alpha_col = _validate(market, account_mode, trigger, basis)
    where, params = _build_where(market, account_mode, trigger, since, until)
    pool = await get_pool()
    async with pool.acquire() as conn:
        async with conn.transaction(isolation="repeatable_read", readonly=True):
            rows = await _fetch_rows(conn, alpha_col, where, params)
            n_mixed = await _fetch_mixed_status_count(conn, where, params)
    return _build_report(
        rows, market=market, account_mode=account_mode, trigger=trigger, basis=basis, since=since, until=until,
        era_boundary=era_boundary, n_mixed_status_keys=n_mixed,
    )


# ---------------------------------------------------------------------------
# --recompute-pending: sensitivity check, audit-side only
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StageAttributionRecomputeReport:
    baseline: StageAttributionReport
    combined: StageAttributionReport
    eval_rule: str
    n_pending_past_cutoff: int
    n_shadowed_by_resolved: int
    n_skipped_not_flat: int
    n_recomputed_rows: int
    n_recompute_failed: int
    n_not_matured: int
    failures_by_reason: dict[str, int]
    failed_tickers: list[str]
    n_alt_symbol_tickers: int
    n_tickers_fetched: int
    n_recomputed_alt_rows: int
    holding_days_min: int | None
    holding_days_median: float | None
    holding_days_max: int | None


async def compute_stage_attribution_with_recompute(
    *, market: str | None, account_mode: str | None, trigger: str | None = None, basis: str = "horizon",
    since: date | None = None, until: date | None = None, era_boundary: datetime | None = None,
) -> StageAttributionRecomputeReport:
    """Domestic only. Pending flat rows past the holding cutoff are re-priced IN
    MEMORY (with the '.KS'<->'.KQ' fallback) using their own stored transcript,
    and the WHOLE report is rebuilt (BH q depends on every test's p). Baseline
    and recompute reads share one REPEATABLE READ snapshot."""
    alpha_col = _validate(market, account_mode, trigger, basis)
    if market != "domestic":
        raise ValueError("--recompute-pending is domestic-only (overseas has different cutoff/session/benchmark)")
    where, params = _build_where(market, account_mode, trigger, since, until)
    cutoff = local_today("domestic") - timedelta(days=settings.memory_min_holding_days)
    pool = await get_pool()
    async with pool.acquire() as conn:
        async with conn.transaction(isolation="repeatable_read", readonly=True):
            rows = await _fetch_rows(conn, alpha_col, where, params)
            n_mixed = await _fetch_mixed_status_count(conn, where, params)
            resolved_keys = await _fetch_resolved_keys(conn, where, params)

    pending = [r for r in rows if r["status"] == "pending" and r["trade_date"] <= cutoff]
    flat_pending = [r for r in pending if _holding_state(r) == "flat"]
    eligible = [r for r in flat_pending if (r["ticker"], r["trade_date"]) not in resolved_keys]
    result = await recompute_pending_alphas(eligible, basis=basis, today=local_today("domestic"))
    rec_by_key = {(r.ticker, r.trade_date): r for r in result.rows}
    combined_rows = []
    for r in rows:
        rec = rec_by_key.get((r["ticker"], r["trade_date"]))
        if rec is None:
            combined_rows.append(r)
        else:
            combined_rows.append({**dict(r), "status": "resolved", "alpha_return": rec.alpha_return, "source": "recomputed"})

    common = dict(
        market=market, account_mode=account_mode, trigger=trigger, basis=basis, since=since, until=until,
        era_boundary=era_boundary, n_mixed_status_keys=n_mixed,
    )
    by_reason: dict[str, int] = {}
    for failure in result.failures:
        by_reason[failure.reason] = by_reason.get(failure.reason, 0) + 1
    hard = [f for f in result.failures if f.reason != REASON_NOT_MATURED]
    h_min, h_median, h_max = holding_days_stats(result.rows)
    eval_rule = (
        f"first settled common close on/after trade_date+{settings.memory_min_holding_days}d "
        "(production resolves at the LATEST common close)"
        if basis == "variable"
        else f"{settings.memory_fixed_horizon_trading_days}th trading day (same as production horizon)"
    )
    return StageAttributionRecomputeReport(
        baseline=_build_report(rows, **common), combined=_build_report(combined_rows, **common),
        eval_rule=eval_rule, n_pending_past_cutoff=len(pending),
        n_shadowed_by_resolved=len(flat_pending) - len(eligible), n_skipped_not_flat=len(pending) - len(flat_pending),
        n_recomputed_rows=len(result.rows), n_recompute_failed=len(hard),
        n_not_matured=by_reason.get(REASON_NOT_MATURED, 0), failures_by_reason=by_reason,
        failed_tickers=sorted({f.ticker for f in hard}), n_alt_symbol_tickers=result.n_alt_symbol_tickers,
        n_tickers_fetched=result.n_tickers_fetched,
        n_recomputed_alt_rows=sum(1 for r in result.rows if r.price_symbol != r.ticker),
        holding_days_min=h_min, holding_days_median=h_median, holding_days_max=h_max,
    )


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _pct(value: float | None) -> str:
    return f"{value:+.4%}" if value is not None else "-"


def _ci_str(ci) -> str:
    return f"[{ci[0]:+.4%}, {ci[1]:+.4%}]" if ci is not None else "-"


def _num(value, digits: int = 4) -> str:
    return f"{value:.{digits}f}" if value is not None else "-"


def _is_masked(c: ContrastStats) -> bool:
    return c.verdict == "UNDERPOWERED" or (c.verdict == "LOW_COVERAGE" and c.underpowered_condition)


def _verdict_label(c: ContrastStats) -> str:
    if c.identical_to is not None:
        return f"identical_to={c.identical_to} (not tested, excluded from BH m)"
    if c.verdict == "FINDING":
        label = f"FINDING({c.direction})" + ("" if c.material else " (immaterial)")
        return label + (" fragile: overlapping horizons" if c.fragile else "")
    if c.verdict == "NO_EFFECT" and c.ci95 is not None:
        return f"NO_EFFECT (resolution +/-{(c.ci95[1] - c.ci95[0]) / 2:.2%})"
    return c.verdict


def _contrast_line(c: ContrastStats, show_underpowered: bool) -> str:
    head = f"{c.name:<40} {c.n_a:>5} {c.n_b:>5} {c.n_a_paired:>5} {c.n_b_paired:>5} {c.n_paired_days:>5} "
    if c.identical_to is not None:
        return head + _verdict_label(c)
    if _is_masked(c) and not show_underpowered:
        return head + f"{'(masked)':>10} {'(masked)':>22} {'(masked)':>8} {'(masked)':>8}  {_verdict_label(c)}"
    body = (
        f"{_pct(c.delta):>10} {_ci_str(c.ci95):>22} {_num(c.p_value):>8} {_num(c.q_value):>8}  {_verdict_label(c)}"
    )
    if _is_masked(c):
        return head + body + f"  PROVISIONAL (n_paired_days={c.n_paired_days}<{_MIN_DAYS_FOR_VERDICT}) - not a finding"
    return head + body


_CONTRAST_HEADER = (
    f"{'test':<40} {'nA':>5} {'nB':>5} {'pA':>5} {'pB':>5} {'days':>5} "
    f"{'delta':>10} {'CI95':>22} {'p':>8} {'q':>8}  verdict"
)


def _fmt_proj(p: ProjectionStats | None) -> str:
    if p is None:
        return "-"
    real = "-" if p.realization is None else f"{p.realization:.2f}"
    extra = "never at current rate" if p.extra_decision_days is None else f"+{p.extra_decision_days} decision TD"
    return f"elig_rate={p.elig_rate:.2f} realization={real} backlog={p.backlog:.1f}  {extra}"


def render_stage_attribution(report: StageAttributionReport, *, show_underpowered: bool = False) -> str:
    r = report
    lines = [
        "# Stage attribution -- which stage adds or removes discrimination (read-only)",
        f"market={r.market}  scope=live  account_mode={r.account_mode}  trigger={r.trigger}  basis={r.basis}",
        f"trade_date filter: {r.since or '(none)'}..{r.until or '(none)'}   "
        f"scored trade_date range: "
        + (f"{r.scored_range[0]}..{r.scored_range[1]}" if r.scored_range else "(none)"),
        "Decision rule pre-registered (see module docstring); verdicts only at n_paired_days>=20.",
        "",
        "## Coverage funnel",
        f"decision rows (latest per ticker/trade_date): {r.n_rows}   decision days (flat): {r.n_decision_days}",
        f"flat={r.n_flat}  held={r.n_held}  held-state unknown={r.n_unknown_held}   (only flat rows are tested)",
        f"flat scored={r.n_scored_flat} over {r.n_scored_days} trade_dates   flat still pending={r.n_pending_flat}",
        f"held rows scored (not tested): {r.n_scored_held}",
    ]
    if r.n_mixed_status_keys > 0:
        lines.append(
            f"! WARNING: {r.n_mixed_status_keys} (ticker, trade_date) keys carry more than one status; the latest "
            "row was used, so a resolved sibling may be hidden behind a pending row."
        )
    lines += ["", "## Stage extraction (flat rows)"]
    lines.append(f"{'stage':<15}" + "".join(f"{lv:>12}" for lv in _LEVELS) + f"{'unknown':>9}{'cov(all)':>10}{'cov(scored)':>13}")
    for s in r.stage_summaries:
        lines.append(
            f"{s.stage:<15}" + "".join(f"{s.counts.get(lv, 0):>12}" for lv in _LEVELS)
            + f"{s.n_unknown:>9}{s.coverage_all_flat:>10.1%}{s.coverage_scored_flat:>13.1%}"
        )
    lines.append(
        f"RM unparsed={r.n_rm_unparsed}  trader via markdown fallback={r.n_trader_markdown}  "
        f"trader column/markdown mismatch={r.n_trader_column_markdown_mismatch}  "
        f"gate parse failed={r.n_gate_parse_failed}  gate record/note mismatch={r.n_gate_record_note_mismatch}"
    )
    lines += ["", "## Flow (decision-time counts, no alpha)"]
    lines.append(f"{'transition':<20}{'known':>7}{'X bull':>8}{'blocked':>9}{'passed':>8}{'agr_not':>9}{'promoted':>10}{'exact%':>8}")
    for f in r.flow:
        agree = "-" if f.exact_agreement is None else f"{f.exact_agreement:.1%}"
        lines.append(
            f"{f.x + '->' + f.y:<20}{f.n_both_known:>7}{f.n_x_bullish:>8}{f.blocked:>9}{f.passed:>8}"
            f"{f.agreed_not:>9}{f.promoted:>10}{agree:>8}"
        )
    lines += ["", "## Q1 -- does the stage's stance separate forward alpha? (A=bullish, B=not bullish)", _CONTRAST_HEADER]
    for c in r.q1:
        lines.append(_contrast_line(c, show_underpowered))
    for c in r.q1:
        text = interpret_q1(c)
        if text:
            lines.append(f"  {c.stages[0]}: {text}")
    lines += ["", "## Q2 -- were overrides/promotions right? (positive blocked-agreed_not = downstream discarded information)", _CONTRAST_HEADER]
    for block in r.transitions:
        for c in (block.primary, block.secondary, block.promoted):
            if c is not None:
                lines.append(_contrast_line(c, show_underpowered))
        if block.promoted is None:
            lines.append(f"{'Q2 ' + block.x + '->' + block.y + ': promoted':<40} n/a: gates only downgrade")
    for block in r.transitions:
        if block.block_interpretation not in ("NOT_DETERMINED",) or block.promotion_interpretation not in ("NOT_DETERMINED", "n/a"):
            lines.append(
                f"  {block.x}->{block.y}: block={block.block_interpretation}  promotion={block.promotion_interpretation}"
            )
        if block.destroys_claim:
            lines.append(f"  => stage {block.y} destroys value relative to {block.x} (coverage and era checks passed)")
    lines += ["", f"BH family size m = {r.n_bh_family} (tests with a p-value, identical stages excluded)"]

    lines += ["", "## Q3 -- power projection (no calendar dates: add the +10 trading-day maturation lag)"]
    for c in r.contrasts:
        if c.identical_to is None:
            lines.append(f"{c.name:<40} {_fmt_proj(c.projection)}")

    if r.era_boundary is not None:
        lines += ["", f"## Era split at {r.era_boundary.isoformat()} (created_at)"]
        lines.append(f"{'test':<40} {'before':>10} {'pairs':>6} {'after':>10} {'pairs':>6}  status")
        for e in r.era:
            lines.append(f"{e.name:<40} {_pct(e.delta_before):>10} {e.n_before:>6} {_pct(e.delta_after):>10} {e.n_after:>6}  {e.status}")
        for label in ("before", "after"):
            lines.append(f"git_commit {label}: " + (", ".join(r.era_commits.get(label, [])) or "(none)"))

    lines += [
        "",
        "## Notes",
        "- Both cohorts are scored on RAW alpha (not sign-flipped); stances are bullish (Buy/Overweight) vs not.",
        "- Only flat decisions are tested; held decisions are counted above.",
        "- material = |delta| >= 0.50%p is an unvalidated assumption (not checked against round-trip cost).",
        "- Masked rows are hidden by default (--show-underpowered reveals them as PROVISIONAL).",
        "- Cite the current system by rerunning with --start-date 2026-09-18; a full-range result is past-pooled only.",
    ]
    return "\n".join(lines)


def render_stage_attribution_recompute(
    r: StageAttributionRecomputeReport, *, show_underpowered: bool = False,
) -> str:
    b, c = r.baseline, r.combined
    lines = [
        "# Stage attribution -- baseline vs recomputed pending (sensitivity check)",
        f"market={b.market}  scope=live  account_mode={b.account_mode}  trigger={b.trigger}  basis={b.basis}",
        f"eval_rule={r.eval_rule}",
        f"n_pending_past_cutoff={r.n_pending_past_cutoff}  n_shadowed_by_resolved={r.n_shadowed_by_resolved}  "
        f"n_skipped_not_flat={r.n_skipped_not_flat}",
        f"n_recomputed_rows={r.n_recomputed_rows}  n_recompute_failed={r.n_recompute_failed}  "
        f"n_not_matured={r.n_not_matured}",
        f"alt-symbol: tickers {r.n_alt_symbol_tickers}/{r.n_tickers_fetched}, "
        f"recomputed rows {r.n_recomputed_alt_rows}/{r.n_recomputed_rows}",
        "failures_by_reason=" + (", ".join(f"{k}={v}" for k, v in sorted(r.failures_by_reason.items())) or "(none)"),
        f"holding_days min/median/max={r.holding_days_min}/{r.holding_days_median}/{r.holding_days_max}",
    ]
    if r.failed_tickers:
        lines.append("failed_tickers=" + ", ".join(r.failed_tickers))
    lines += ["", "## baseline", render_stage_attribution(b, show_underpowered=show_underpowered),
              "", "## recomputed (resolved + recomputed)", render_stage_attribution(c, show_underpowered=show_underpowered),
              "",
              "NOTE: recomputed rows exist only in memory: not written to DB, no reflection generated. Cite the",
              "baseline; read the recomputed report as a sensitivity check only."]
    return "\n".join(lines)
