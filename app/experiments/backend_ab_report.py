"""Aggregates `app.experiments.backend_ab` output into a Claude-vs-Qwen
hallucination-guard-violation comparison table.

The core computation (`compute_backend_ab_report`) is a pure function over
already-loaded record dicts — no I/O of its own — specifically so it can be
unit-tested without a GPU, a DB, or a real sweep run (see
`tests/experiments/test_backend_ab_report.py`). `_main`/`_parse_args` are a
thin CLI wrapper around it.

Statistics reuse `app.audit.guard_alpha`'s existing permutation-test/BH-
correction machinery directly rather than reimplementing it — see
`_permutation_p_value`/`_apply_bh_correction` imports below. Pairing is by
`(ticker, date, repeat)`; a pair is only used when BOTH arms have a matching,
non-degraded, non-isolation-violating record. `guard_violations_total` is the
primary metric (`Δ = claude - qwen` per pair); `llm_fallback_calls` /
`llm_truncated_calls` (summed over the PAIRED sample) and `degraded` (counted
over ALL completed runs in the directory, not just the paired ones — a
degraded run is invisible to pairing but must not be invisible to the
comparison) are secondary metrics that can turn an otherwise-winning result
into MIXED.

    python -m app.experiments.backend_ab_report --dir results/backend_ab/20260914T120000
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from app.audit.guard_alpha import _N_PERMUTATIONS, _PERMUTATION_SEED, _apply_bh_correction, _permutation_p_value

if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

logger_prefix = "[backend_ab_report]"

ARM_CLAUDE = "claude"
ARM_QWEN = "qwen"


@dataclass(frozen=True)
class _FieldStats:
    """Per-guard-field breakdown row — shaped exactly so
    `app.audit.guard_alpha._apply_bh_correction` (which only ever touches
    `.p_value` and rewrites `.q_value` via `dataclasses.replace`) works on a
    `list[_FieldStats]` unmodified."""

    field_name: str
    n_fires: int
    mean_delta: float | None
    p_value: float | None
    q_value: float | None
    underpowered: bool


@dataclass(frozen=True)
class Census:
    completed: int
    degraded: int
    isolation_violation: int
    errors: int


@dataclass(frozen=True)
class BackendAbReport:
    claude_census: Census
    qwen_census: Census
    candidate_groups: int
    paired: int
    dropped_missing_one_arm: int
    dropped_degraded: int
    dropped_isolation_violation: int
    mean_delta: float | None
    p_value: float | None
    claude_fallback_paired: int
    qwen_fallback_paired: int
    claude_truncated_paired: int
    qwen_truncated_paired: int
    verdict: str
    verdict_detail: str
    field_stats: list[_FieldStats] = field(default_factory=list)
    min_pairs: int = 20
    min_field_fires: int = 10


def _load_records(directory: Path) -> tuple[list[dict], dict[str, int]]:
    """Loads every `*.json` record, skipping `*_error.json` (those files are
    counted per arm and returned separately, not treated as a completed
    run)."""
    records: list[dict] = []
    error_counts: dict[str, int] = defaultdict(int)
    for path in sorted(directory.glob("*.json")):
        if path.name.endswith(".tmp") or ".tmp" in path.suffixes:
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if path.stem.endswith("_error"):
            arm = data.get("arm")
            if arm:
                error_counts[arm] += 1
            continue
        records.append(data)
    return records, dict(error_counts)


def _census_for_arm(records: list[dict], error_counts: dict[str, int], arm: str) -> Census:
    arm_records = [r for r in records if r.get("arm") == arm]
    return Census(
        completed=len(arm_records),
        degraded=sum(1 for r in arm_records if r.get("degraded")),
        isolation_violation=sum(1 for r in arm_records if r.get("isolation_violation")),
        errors=error_counts.get(arm, 0),
    )


def _pair_key(record: dict) -> tuple[Any, Any, Any]:
    return (record.get("ticker"), record.get("date"), record.get("repeat"))


def compute_backend_ab_report(
    records: list[dict], error_counts: dict[str, int] | None = None,
    *, min_pairs: int = 20, min_field_fires: int = 10,
) -> BackendAbReport:
    error_counts = error_counts or {}
    claude_census = _census_for_arm(records, error_counts, ARM_CLAUDE)
    qwen_census = _census_for_arm(records, error_counts, ARM_QWEN)

    by_key: dict[tuple, dict[str, dict]] = defaultdict(dict)
    for r in records:
        arm = r.get("arm")
        if arm not in (ARM_CLAUDE, ARM_QWEN):
            continue
        by_key[_pair_key(r)][arm] = r

    candidate_groups = len(by_key)
    dropped_missing_one_arm = 0
    dropped_degraded = 0
    dropped_isolation_violation = 0

    paired_claude: list[dict] = []
    paired_qwen: list[dict] = []
    for key, arms in by_key.items():
        claude_r = arms.get(ARM_CLAUDE)
        qwen_r = arms.get(ARM_QWEN)
        if claude_r is None or qwen_r is None:
            dropped_missing_one_arm += 1
            continue
        if claude_r.get("degraded") or qwen_r.get("degraded"):
            dropped_degraded += 1
            continue
        if claude_r.get("isolation_violation") or qwen_r.get("isolation_violation"):
            dropped_isolation_violation += 1
            continue
        paired_claude.append(claude_r)
        paired_qwen.append(qwen_r)

    paired = len(paired_claude)

    claude_totals = [r.get("guard_violations_total", 0) or 0 for r in paired_claude]
    qwen_totals = [r.get("guard_violations_total", 0) or 0 for r in paired_qwen]
    deltas = [c - q for c, q in zip(claude_totals, qwen_totals)]
    mean_delta = sum(deltas) / len(deltas) if deltas else None

    p_value = None
    if paired:
        p_value = _permutation_p_value(
            [[c] for c in claude_totals], [[q] for q in qwen_totals], observed=mean_delta,
        )

    claude_fallback_paired = sum(r.get("llm_fallback_calls", 0) or 0 for r in paired_claude)
    qwen_fallback_paired = sum(r.get("llm_fallback_calls", 0) or 0 for r in paired_qwen)
    claude_truncated_paired = sum(r.get("llm_truncated_calls", 0) or 0 for r in paired_claude)
    qwen_truncated_paired = sum(r.get("llm_truncated_calls", 0) or 0 for r in paired_qwen)

    # Per-guard-field breakdown — union of every key seen in either arm's
    # `guard_violations` dict across the paired sample, never hardcoded.
    field_names: set[str] = set()
    for r in paired_claude + paired_qwen:
        field_names.update((r.get("guard_violations") or {}).keys())

    field_stats: list[_FieldStats] = []
    for name in sorted(field_names):
        claude_vals = [(r.get("guard_violations") or {}).get(name, 0) or 0 for r in paired_claude]
        qwen_vals = [(r.get("guard_violations") or {}).get(name, 0) or 0 for r in paired_qwen]
        n_fires = sum(claude_vals) + sum(qwen_vals)
        underpowered = paired < min_pairs or n_fires < min_field_fires
        if underpowered:
            field_stats.append(
                _FieldStats(field_name=name, n_fires=n_fires, mean_delta=None, p_value=None, q_value=None, underpowered=True)
            )
            continue
        f_deltas = [c - q for c, q in zip(claude_vals, qwen_vals)]
        f_mean_delta = sum(f_deltas) / len(f_deltas)
        f_p = _permutation_p_value([[c] for c in claude_vals], [[q] for q in qwen_vals], observed=f_mean_delta)
        field_stats.append(
            _FieldStats(field_name=name, n_fires=n_fires, mean_delta=f_mean_delta, p_value=f_p, q_value=None, underpowered=False)
        )

    _apply_bh_correction(field_stats)

    verdict, verdict_detail = _compute_verdict(
        paired=paired, min_pairs=min_pairs, claude_totals=claude_totals, qwen_totals=qwen_totals,
        mean_delta=mean_delta, p_value=p_value,
        claude_degraded=claude_census.degraded, qwen_degraded=qwen_census.degraded,
        claude_fallback_paired=claude_fallback_paired, qwen_fallback_paired=qwen_fallback_paired,
        claude_truncated_paired=claude_truncated_paired, qwen_truncated_paired=qwen_truncated_paired,
    )

    return BackendAbReport(
        claude_census=claude_census,
        qwen_census=qwen_census,
        candidate_groups=candidate_groups,
        paired=paired,
        dropped_missing_one_arm=dropped_missing_one_arm,
        dropped_degraded=dropped_degraded,
        dropped_isolation_violation=dropped_isolation_violation,
        mean_delta=mean_delta,
        p_value=p_value,
        claude_fallback_paired=claude_fallback_paired,
        qwen_fallback_paired=qwen_fallback_paired,
        claude_truncated_paired=claude_truncated_paired,
        qwen_truncated_paired=qwen_truncated_paired,
        verdict=verdict,
        verdict_detail=verdict_detail,
        field_stats=field_stats,
        min_pairs=min_pairs,
        min_field_fires=min_field_fires,
    )


def _compute_verdict(
    *, paired: int, min_pairs: int, claude_totals: list[int], qwen_totals: list[int],
    mean_delta: float | None, p_value: float | None,
    claude_degraded: int, qwen_degraded: int,
    claude_fallback_paired: int, qwen_fallback_paired: int,
    claude_truncated_paired: int, qwen_truncated_paired: int,
) -> tuple[str, str]:
    total_violations = sum(claude_totals) + sum(qwen_totals)

    # 1. UNDERPOWERED
    if paired < min_pairs or total_violations == 0:
        reason = (
            f"only {paired} paired run(s) (< min_pairs={min_pairs})"
            if paired < min_pairs
            else "both arms recorded zero guard violations across the entire paired sample"
        )
        return "UNDERPOWERED", f"no conclusion, needs more paired runs — {reason}"

    assert mean_delta is not None and p_value is not None

    secondary_worse = []
    if claude_degraded > qwen_degraded:
        secondary_worse.append(f"degraded count worse on claude ({claude_degraded} > {qwen_degraded})")
    if claude_fallback_paired > qwen_fallback_paired:
        secondary_worse.append(f"llm_fallback_calls worse on claude ({claude_fallback_paired} > {qwen_fallback_paired})")
    if claude_truncated_paired > qwen_truncated_paired:
        secondary_worse.append(f"llm_truncated_calls worse on claude ({claude_truncated_paired} > {qwen_truncated_paired})")

    # 2. CLAUDE_WINS
    if mean_delta < 0 and p_value < 0.05 and not secondary_worse:
        return "CLAUDE_WINS", f"mean_delta={mean_delta:.4f}, p={p_value:.4f}, no secondary metric got worse"

    # 3. QWEN_WINS
    if mean_delta > 0 and p_value < 0.05:
        return "QWEN_WINS", f"mean_delta={mean_delta:.4f}, p={p_value:.4f}"

    # 4. NO_EFFECT
    if p_value >= 0.05:
        return "NO_EFFECT", (
            f"mean_delta={mean_delta:.4f}, p={p_value:.4f} — not distinguishable at this sample size"
        )

    # 5. MIXED (mean_delta < 0, p < 0.05, but a secondary metric got worse)
    return "MIXED", (
        f"mean_delta={mean_delta:.4f} favors claude on the primary metric, p={p_value:.4f}, but: "
        + "; ".join(secondary_worse)
    )


def _fmt_num(value, digits: int = 4) -> str:
    return f"{value:.{digits}f}" if value is not None else "-"


def render_backend_ab_report(report: BackendAbReport) -> str:
    lines = []
    lines.append("=== Census (per arm, ALL completed runs in --dir, before pairing) ===")
    for label, census in (("claude", report.claude_census), ("qwen", report.qwen_census)):
        lines.append(
            f"  {label:<8} completed={census.completed:>4}  degraded={census.degraded:>4}  "
            f"isolation_violation={census.isolation_violation:>4}  errors={census.errors:>4}"
        )
    lines.append("")
    lines.append("=== Pairing ((ticker, date, repeat)) ===")
    lines.append(
        f"  candidate groups={report.candidate_groups}  paired={report.paired}  "
        f"dropped(missing one arm)={report.dropped_missing_one_arm}  "
        f"dropped(degraded)={report.dropped_degraded}  "
        f"dropped(isolation_violation)={report.dropped_isolation_violation}"
    )
    lines.append("")
    lines.append("=== Overall: guard_violations_total, claude - qwen, paired ===")
    lines.append(f"  mean_delta={_fmt_num(report.mean_delta)}  p={_fmt_num(report.p_value)}")
    lines.append(
        f"  llm_fallback_calls (paired sum): claude={report.claude_fallback_paired} "
        f"qwen={report.qwen_fallback_paired}"
    )
    lines.append(
        f"  llm_truncated_calls (paired sum): claude={report.claude_truncated_paired} "
        f"qwen={report.qwen_truncated_paired}"
    )
    lines.append(f"  VERDICT: {report.verdict} — {report.verdict_detail}")
    lines.append("")
    lines.append("=== Per-guard-field breakdown (union of keys seen in either arm's guard_violations) ===")
    header = f"{'field':<40} {'n_fires':>8} {'delta':>9} {'p':>8} {'q':>8}"
    lines.append(header)
    lines.append("-" * len(header))
    for s in report.field_stats:
        if s.underpowered:
            lines.append(f"{s.field_name:<40} {s.n_fires:>8} {'-':>9} {'-':>8} {'-':>8}")
            lines.append(
                f"    ! UNDERPOWERED (needs >= {report.min_field_fires} combined fires, saw {s.n_fires}) "
                "— do NOT treat this as a finding"
            )
        else:
            lines.append(
                f"{s.field_name:<40} {s.n_fires:>8} {_fmt_num(s.mean_delta):>9} "
                f"{_fmt_num(s.p_value):>8} {_fmt_num(s.q_value):>8}"
            )
    lines.append("")
    lines.append(
        f"Legend: permutation test seed={_PERMUTATION_SEED}, {_N_PERMUTATIONS} resamples "
        "(reused from app.audit.guard_alpha, not reimplemented). BH correction applied across every "
        "per-field row that produced a p-value (adequately powered rows only). Degraded-run counts in "
        "the census are over ALL completed runs in --dir, not just the paired sample — degraded runs "
        "are excluded from pairing and would otherwise be invisible to this comparison; "
        "llm_fallback_calls/llm_truncated_calls are summed over the PAIRED sample only. "
        "NO_EFFECT means not distinguishable at this sample size, not proof of no difference."
    )
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", required=True, help="Directory of app.experiments.backend_ab output.")
    parser.add_argument("--min-pairs", type=int, default=20, help="Minimum paired runs before any verdict beyond UNDERPOWERED.")
    parser.add_argument(
        "--min-field-fires", type=int, default=10,
        help="Minimum combined (both arms) fires for a per-field row before it prints real numbers.",
    )
    return parser.parse_args()


def _main() -> None:
    args = _parse_args()
    directory = Path(args.dir)
    records, error_counts = _load_records(directory)
    report = compute_backend_ab_report(
        records, error_counts, min_pairs=args.min_pairs, min_field_fires=args.min_field_fires,
    )
    print(render_backend_ab_report(report))


if __name__ == "__main__":
    _main()
