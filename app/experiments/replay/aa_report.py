"""A/A report: two identical baseline runs (R1, R2) against the stored results (spec F).

Pass lines are PRE-REGISTERED in `app.experiments.replay.__init__`'s docstring; the constants
below repeat them and a test compares the two so they cannot drift. This module never
scores alpha: an A/A run validates the harness, it is not an evaluation.
A/A 리포트 — 합격선은 replay/__init__ docstring에 사전 등록된 값과 같아야 한다.

`bull(x)`: x in {Buy, Overweight}. Population C: R1 and R2 both ok and a parsable stored
result. `d(A, B) = mean[bull(A_i) != bull(B_i)]`. McNemar is the exact binomial (math.comb only).
"""

from __future__ import annotations

import csv
import io
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from app.experiments.replay import safety

BULL = frozenset({"Buy", "Overweight"})

# Pre-registered thresholds (spec F table).
T_COMPLETION = 0.98          # AA-1c
T_FALLBACK = 0.03            # AA-1d (supplement 1: absolute cap)
T_LENGTH = 0.028             # AA-1e  = 2 x 1.4%
T_RM_SHA = 1.0               # AA-1f (supplement 2)
T_DESK_NOTE = 1.0            # AA-1g
T_AA2A = 0.08                # d(R1,R2) RM
T_AA2B = 0.03                # d(R1,R2) final
T_AA3A = 0.05                # d(R1,stored) - d(R1,R2), RM, percentage points
T_AA3B = 0.05                # |bullrate_R1 - bullrate_stored| RM
T_AA3C = 0.05                # McNemar p >= 0.05
T_AA3D = 0.10                # |d_A - d_B| < 10 pp (supplement 3: B = interim False U None)
EPS = 1e-9

VERDICT_NOGO = "No-go (AA-1 failed)"
VERDICT_CONTRACT = "input-contract defect: investigate (AA-3 failed)"
VERDICT_THREE = "AA-2 only failed: switch to 3 repeats per arm (repeat 1..3)"
VERDICT_PASS = "PASS"


def bull(x: str | None) -> bool | None:
    return None if x is None else x in BULL


# --------------------------------------------------------------------------- statistics


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact binomial McNemar p: min(1, 2 * sum_{i<=min(b,c)} C(n,i) 0.5^n), n=b+c."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1))
    return min(1.0, 2.0 * tail / (2 ** n))


def fisher_exact(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact p for the 2x2 table [[a, b], [c, d]]."""
    row1, row2, col1 = a + b, c + d, a + c
    total = row1 + row2

    def prob(x: int) -> float:
        return math.comb(row1, x) * math.comb(row2, col1 - x) / math.comb(total, col1)

    observed = prob(a)
    lo, hi = max(0, col1 - row2), min(row1, col1)
    return min(1.0, sum(p for p in (prob(x) for x in range(lo, hi + 1)) if p <= observed * (1 + 1e-9)))


def percentile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def disagreement(pairs: Iterable[tuple[bool | None, bool | None]]) -> tuple[float | None, int]:
    """(mean[a != b], n) over the pairs where both sides are known."""
    known = [(a, b) for a, b in pairs if a is not None and b is not None]
    if not known:
        return None, 0
    return sum(1 for a, b in known if a != b) / len(known), len(known)


# --------------------------------------------------------------------------- data


@dataclass
class RunData:
    manifest: dict[str, Any]
    cases: dict[str, dict[str, Any]]  # ok cases only, by case_id
    path: str = ""


def load_run(run_dir: Path) -> RunData:
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    cases: dict[str, dict[str, Any]] = {}
    for path in sorted((run_dir / "cases").glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("status") == "ok":
            cases[data["case_id"]] = data
    return RunData(manifest=manifest, cases=cases, path=str(run_dir))


def _get(case: Mapping[str, Any], group: str, key: str) -> Any:
    return (case.get(group) or {}).get(key)


def _interim(case: Mapping[str, Any]) -> bool | None:
    return (case.get("covariates") or {}).get("interim")


def _check(value: float | None, ok: Callable[[float], bool]) -> str:
    if value is None:
        return "N/A"
    return "PASS" if ok(value) else "FAIL"


def _row(rid: str, metric: str, value: Any, threshold: str, verdict: str, n: Any = None) -> dict[str, Any]:
    return {"id": rid, "metric": metric, "value": value, "threshold": threshold, "verdict": verdict, "n": n}


# --------------------------------------------------------------------------- report


def population(r1: RunData, r2: RunData) -> list[str]:
    """Case ids in C: ok in both runs and a parsable stored RM/final result."""
    ids = []
    for case_id in sorted(set(r1.cases) & set(r2.cases)):
        stored = r1.cases[case_id].get("stored") or {}
        if stored.get("rm") is not None and stored.get("final") is not None:
            ids.append(case_id)
    return ids


def _decisions(cases: Mapping[str, Mapping[str, Any]], ids: Sequence[str], group: str, key: str) -> list[str | None]:
    return [_get(cases[i], group, key) for i in ids]


def _pair_d(a: list[str | None], b: list[str | None]) -> tuple[float | None, int]:
    return disagreement(zip((bull(x) for x in a), (bull(x) for x in b)))


def _vs_stored(r: RunData, ids: Sequence[str], key: str, rr_d: float | None, stored_key: str | None = None) -> dict[str, Any]:
    stored_key = stored_key or key
    run_vals = _decisions(r.cases, ids, "summary", key)
    stored_vals = _decisions(r.cases, ids, "stored", stored_key)
    d, n = _pair_d(run_vals, stored_vals)
    rate_run = sum(1 for x in run_vals if bull(x)) / len(ids) if ids else None
    rate_stored = sum(1 for x in stored_vals if bull(x)) / len(ids) if ids else None
    b = sum(1 for x, s in zip(run_vals, stored_vals) if bull(x) and not bull(s))
    c = sum(1 for x, s in zip(run_vals, stored_vals) if bull(s) and not bull(x))
    return {
        "d_vs_stored": d, "n": n, "bullrate_run": rate_run, "bullrate_stored": rate_stored,
        "bullrate_gap": None if rate_run is None else abs(rate_run - rate_stored),
        "b": b, "c": c, "mcnemar_p": mcnemar_exact(b, c),
        "aa3a_excess": None if d is None or rr_d is None else d - rr_d,
    }


def interim_split(r1: RunData, ids: Sequence[str]) -> dict[str, Any]:
    """AA-3d: RM disagreement R1 vs stored, group A (interim True) vs group B (False U None)."""
    def group(pred: Callable[[bool | None], bool]) -> tuple[float | None, int, int]:
        sel = [i for i in ids if pred(_interim(r1.cases[i]))]
        d, n = _pair_d(_decisions(r1.cases, sel, "summary", "rm"), _decisions(r1.cases, sel, "stored", "rm"))
        return d, n, round((d or 0.0) * n)

    d_a, n_a, k_a = group(lambda v: v is True)
    d_b, n_b, k_b = group(lambda v: v is not True)
    three = {name: group(lambda v, want=want: v is want) for name, want in (("true", True), ("false", False), ("none", None))}
    gap = None if d_a is None or d_b is None else abs(d_a - d_b)
    fisher = fisher_exact(k_a, n_a - k_a, k_b, n_b - k_b) if n_a and n_b else None
    return {
        "d_A": d_a, "n_A": n_a, "mismatch_A": k_a, "d_B": d_b, "n_B": n_b, "mismatch_B": k_b,
        "gap": gap, "restore_period_note_needed": gap is not None and not gap < T_AA3D - EPS,
        "fisher_p": fisher,
        "three_way": {k: {"d": v[0], "n": v[1]} for k, v in three.items()},
    }


def _dist(values: Iterable[str | None]) -> dict[str, int]:
    out: dict[str, int] = {}
    for v in values:
        key = "None" if v is None else v
        out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items()))


def _pctl(values: Sequence[float]) -> dict[str, float | None]:
    return {"p50": percentile(values, 0.5), "p90": percentile(values, 0.9)}


class PairError(Exception):
    """R1/R2 are not a valid A/A pair (CLI exit code 2)."""


def validate_pair(r1: RunData, r2: RunData) -> None:
    """Review M4: refuse a pair that cannot be an A/A comparison (the same directory twice would
    give a false PASS). Both manifests must describe two DIFFERENT finished baseline repeats of
    the same code/model/settings/case set."""
    m1, m2 = r1.manifest, r2.manifest
    problems: list[str] = []
    if not m1.get("base_key") or m1.get("base_key") != m2.get("base_key"):
        problems.append(f"base_key differs or is missing ({m1.get('base_key')!r} vs {m2.get('base_key')!r})")
    if m1.get("run_id") is None or m1.get("run_id") == m2.get("run_id"):
        problems.append(f"run_id must differ (both {m1.get('run_id')!r})")
    if m1.get("repeat") is None or m1.get("repeat") == m2.get("repeat"):
        problems.append(f"repeat must differ (both {m1.get('repeat')!r})")
    for label, m in (("R1", m1), ("R2", m2)):
        if m.get("variant") != "baseline" or m.get("fork_at") != "rm":
            problems.append(f"{label} must be a plain baseline run (variant={m.get('variant')!r}, fork_at={m.get('fork_at')!r})")
        if m.get("status") != "done":
            problems.append(f"{label} status is {m.get('status')!r}, not 'done'")
    sha1, sha2 = (m1.get("caseset") or {}).get("sha256"), (m2.get("caseset") or {}).get("sha256")
    if not sha1 or sha1 != sha2:
        problems.append(f"case set sha256 differs or is missing ({sha1!r} vs {sha2!r})")
    if problems:
        raise PairError("R1/R2 are not a valid A/A pair: " + "; ".join(problems))


def build_report(
    r1: RunData, r2: RunData, *, case_tickers: set[str] | None = None,
    confirmed: Mapping[str, Iterable[int]] | None = None, confirm_info: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    ids = population(r1, r2)
    rows: list[dict[str, Any]] = []

    # --- AA-1 ---
    tickers = case_tickers if case_tickers is not None else {c.get("ticker") for c in r1.cases.values()}
    inv1, inv2 = (r1.manifest.get("invariants") or {}), (r2.manifest.get("invariants") or {})
    if inv1.get("start") and inv2.get("end"):
        after = dict(inv2["end"])
        merged: dict[str, list] = {}
        seen_ids: set[tuple[str, Any]] = set()
        for snap in (inv1.get("end") or {}, inv2["end"]):
            for table, rws in (snap.get("new_rows") or {}).items():
                for rw in rws:
                    if (table, rw.get("id")) not in seen_ids:  # R2's list (since its own start) overlaps R1's
                        seen_ids.add((table, rw.get("id")))
                        merged.setdefault(table, []).append(rw)
        after["new_rows"] = merged
        windows = safety.parse_windows([*(r1.manifest.get("windows") or []), *(r2.manifest.get("windows") or [])])
        result = safety.compare_invariants(
            inv1["start"], after, tickers, run_windows=windows, confirmed=confirmed,
        )
        rows.append(_row(
            "AA-1a", "DB invariants R1 start -> R2 end", result.status,
            "FAIL 0 (list WARNs; possibly-live rows need --confirm-live-rows)",
            "PASS" if result.status in ("PASS", "WARN") else "FAIL",
        ))
        inv_detail = result.as_dict()
        inv_detail["run_windows"] = [[a_.isoformat(), b_.isoformat()] for a_, b_ in windows]
        inv_detail["confirm"] = dict(confirm_info) if confirm_info else None
    else:
        rows.append(_row("AA-1a", "DB invariants R1 start -> R2 end", "missing", "FAIL 0", "FAIL"))
        inv_detail = {"status": "MISSING"}
    trips = int(r1.manifest.get("safety_trips") or 0) + int(r2.manifest.get("safety_trips") or 0)
    aborted = "aborted_safety" in (r1.manifest.get("status"), r2.manifest.get("status"))
    rows.append(_row("AA-1b", "safety trips", trips, "0", "PASS" if trips == 0 and not aborted else "FAIL"))
    completion: dict[str, Any] = {}
    for label, run in (("R1", r1), ("R2", r2)):
        total = int(run.manifest.get("n_cases") or 0) or len(run.cases)
        rate = len(run.cases) / total if total else None
        first = sum(1 for c in run.cases.values() if c.get("attempt", 1) == 1) / total if total else None
        completion[label] = {"ok": len(run.cases), "n_cases": total, "rate": rate, "first_attempt_rate": first}
        rows.append(_row("AA-1c", f"completion {label}", rate, ">= 98%", _check(rate, lambda x: x >= T_COMPLETION - EPS), total))
    calls = [c.get("calls_summary") or {} for run in (r1, r2) for c in run.cases.values()]
    n_struct = sum(c.get("n_structured", 0) for c in calls)
    n_fallback = sum(c.get("n_fallback", 0) for c in calls)
    n_all = sum(c.get("n", 0) for c in calls)
    n_length = sum(c.get("n_length", 0) for c in calls)
    fb = n_fallback / n_struct if n_struct else None
    ln = n_length / n_all if n_all else None
    rows.append(_row("AA-1d", "fallback_used share of structured calls", fb, "<= 3% (absolute cap)", _check(fb, lambda x: x <= T_FALLBACK + EPS), n_struct))
    rows.append(_row("AA-1e", "finish_reason=length share of all calls", ln, "<= 2.8%", _check(ln, lambda x: x <= T_LENGTH + EPS), n_all))
    sha_mismatch = [
        i for i in ids
        if (r1.cases[i].get("calls_summary") or {}).get("prompt_sha256_by_label", {}).get("research_manager")
        != (r2.cases[i].get("calls_summary") or {}).get("prompt_sha256_by_label", {}).get("research_manager")
        or not (r1.cases[i].get("calls_summary") or {}).get("prompt_sha256_by_label", {}).get("research_manager")
    ]
    sha_share = (len(ids) - len(sha_mismatch)) / len(ids) if ids else None
    rows.append(_row("AA-1f", "RM prompt sha256 R1 == R2", sha_share, "100%", _check(sha_share, lambda x: x >= T_RM_SHA - EPS), len(ids)))
    desk = [c for c in r1.cases.values() if (c.get("contract_checks") or {}).get("desk_note_stored_match") is not None]
    desk_match = sum(1 for c in desk if c["contract_checks"]["desk_note_stored_match"]) / len(desk) if desk else None
    rows.append(_row("AA-1g", "desk_note_stored_match (freeze era)", desk_match, "100%", _check(desk_match, lambda x: x >= T_DESK_NOTE - EPS), len(desk)))

    # --- AA-2 ---
    rm_1, rm_2 = _decisions(r1.cases, ids, "summary", "rm"), _decisions(r2.cases, ids, "summary", "rm")
    fin_1, fin_2 = _decisions(r1.cases, ids, "summary", "final"), _decisions(r2.cases, ids, "summary", "final")
    d_rm, n_rm = _pair_d(rm_1, rm_2)
    d_fin, n_fin = _pair_d(fin_1, fin_2)
    rows.append(_row("AA-2a", "d(R1,R2) RM", d_rm, "<= 8%", _check(d_rm, lambda x: x <= T_AA2A + EPS), n_rm))
    rows.append(_row("AA-2b", "d(R1,R2) final (after gates)", d_fin, "<= 3%", _check(d_fin, lambda x: x <= T_AA2B + EPS), n_fin))
    info: dict[str, Any] = {}
    for key, stored_key in (("neutral_final", "neutral_final"), ("trader", "trader"), ("pm_pre_gate", "pm_pre_gate")):
        d, n = _pair_d(_decisions(r1.cases, ids, "summary", key), _decisions(r2.cases, ids, "summary", key))
        info[key] = {"d_R1_R2": d, "n": n}
        rows.append(_row("AA-2c", f"d(R1,R2) {key} (info)", d, "info", "INFO", n))

    # --- AA-3 ---
    rm_vs = _vs_stored(r1, ids, "rm", d_rm)
    rows.append(_row("AA-3a", "d(R1,stored) RM - d(R1,R2) RM", rm_vs["aa3a_excess"], "<= +5pp", _check(rm_vs["aa3a_excess"], lambda x: x <= T_AA3A + EPS), rm_vs["n"]))
    rows.append(_row("AA-3b", "|bullrate_R1 - bullrate_stored| RM", rm_vs["bullrate_gap"], "<= 5pp", _check(rm_vs["bullrate_gap"], lambda x: x <= T_AA3B + EPS), rm_vs["n"]))
    rows.append(_row("AA-3c", "McNemar exact p (R1 vs stored, RM)", rm_vs["mcnemar_p"], ">= 0.05", _check(rm_vs["mcnemar_p"], lambda x: x >= T_AA3C - EPS), rm_vs["b"] + rm_vs["c"]))
    split = interim_split(r1, ids)
    rows.append(_row("AA-3d", "|d_A - d_B| RM vs stored, A=interim True, B=False U None", split["gap"], "< 10pp", _check(split["gap"], lambda x: x < T_AA3D - EPS), (split["n_A"], split["n_B"])))
    info_final = _vs_stored(r1, ids, "final", d_fin)
    info_r2_rm = _vs_stored(r2, ids, "rm", d_rm)
    info_r2_final = _vs_stored(r2, ids, "final", d_fin)
    for name, block in (("final R1 vs stored", info_final), ("RM R2 vs stored", info_r2_rm), ("final R2 vs stored", info_r2_final)):
        rows.append(_row("info", f"AA-3a~c on {name}", {k: block[k] for k in ("d_vs_stored", "bullrate_gap", "mcnemar_p")}, "not used in the verdict", "INFO", block["n"]))

    verdict = decide(rows)

    # --- distributions / covariates ---
    dist = {}
    for key in ("rm", "trader", "neutral_final", "pm_pre_gate", "final"):
        dist[key] = {
            "stored": _dist(_decisions(r1.cases, ids, "stored", key)),
            "R1": _dist(_decisions(r1.cases, ids, "summary", key)),
            "R2": _dist(_decisions(r2.cases, ids, "summary", key)),
        }
    conf = {
        label: _mean([_get(run.cases[i], "summary", "confidence") for i in ids]) for label, run in (("R1", r1), ("R2", r2))
    }
    conf["stored"] = _mean([_get(r1.cases[i], "stored", "confidence") for i in ids])
    gates = {
        label: {
            "consensus_gated": sum(1 for i in ids if (run.cases[i]["nodes"]["gates"].get("consensus_override") or {}).get("gated")),
            "hysteresis_applied": sum(1 for i in ids if run.cases[i]["nodes"]["gates"].get("hysteresis_applied")),
        }
        for label, run in (("R1", r1), ("R2", r2)) if all("nodes" in run.cases[i] for i in ids)
    }
    resources = {}
    for label, run in (("R1", r1), ("R2", r2)):
        cs = [run.cases[i] for i in ids]
        resources[label] = {
            "tokens_total": _pctl([c.get("tokens_total") or 0 for c in cs]),
            "wall_s": _pctl([c.get("wall_s") or 0 for c in cs]),
            "tokens_by_node": {
                node: _pctl([(c.get("nodes", {}).get(node) or {}).get("tokens") or 0 for c in cs])
                for node in ("rm", "trader", "risk_debate", "pm", "gates")
            },
        }
    covariate_mismatch = {}
    for name, extract in (
        ("interim", lambda c: str(_interim(c))),
        ("kr_risk_neutralized", lambda c: str((c.get("covariates") or {}).get("kr_risk_neutralized"))),
        ("session_agree", lambda c: str(((c.get("covariates") or {}).get("session_meta") or {}).get("agree"))),
    ):
        groups: dict[str, list[str]] = {}
        for i in ids:
            groups.setdefault(extract(r1.cases[i]), []).append(i)
        covariate_mismatch[name] = {
            g: dict(zip(("d_R1_stored", "n"), _pair_d(_decisions(r1.cases, sel, "summary", "rm"), _decisions(r1.cases, sel, "stored", "rm"))))
            for g, sel in sorted(groups.items())
        }
    changed = changed_cases(r1, r2, ids)
    return {
        "verdict": verdict, "rows": rows, "n_population": len(ids), "invariants": inv_detail,
        "completion": completion, "rm_vs_stored": rm_vs, "interim": split, "info_2c": info,
        "distributions": dist, "confidence_mean": conf, "gates": gates, "resources": resources,
        "covariate_rm_mismatch": covariate_mismatch, "rm_prompt_sha_mismatch_cases": sha_mismatch,
        "n_changed_cases": len(changed), "r1": r1.path, "r2": r2.path,
        "manifest_keys": {"r1": r1.manifest.get("base_key"), "r2": r2.manifest.get("base_key")},
    }


def _mean(values: Iterable[float | None]) -> float | None:
    vs = [v for v in values if v is not None]
    return sum(vs) / len(vs) if vs else None


def decide(rows: Sequence[Mapping[str, Any]]) -> str:
    """Pre-registered order: AA-1 fail -> No-go; AA-3 fail -> contract defect; AA-2 only -> 3 repeats."""
    failed = {r["id"] for r in rows if r["verdict"] == "FAIL"}
    if any(i.startswith("AA-1") for i in failed):
        return VERDICT_NOGO
    if any(i.startswith("AA-3") for i in failed):
        return VERDICT_CONTRACT
    if any(i.startswith("AA-2") for i in failed):
        return VERDICT_THREE
    return VERDICT_PASS


# --------------------------------------------------------------------------- changed cases


def changed_cases(r1: RunData, r2: RunData, ids: Sequence[str]) -> list[dict[str, Any]]:
    """Cases whose RM or final bull flag differs between R1, R2 and stored (any pair)."""
    out = []
    for i in ids:
        a, b = r1.cases[i], r2.cases[i]
        kinds: list[str] = []
        for key in ("rm", "final"):
            v1, v2, vs = bull(_get(a, "summary", key)), bull(_get(b, "summary", key)), bull(_get(a, "stored", key))
            for tag, x, y in (("R1≠R2", v1, v2), ("R1≠S", v1, vs), ("R2≠S", v2, vs)):
                if x != y and tag not in kinds:
                    kinds.append(tag)
        if not kinds:
            continue
        out.append({
            "case_id": i, "ticker": a.get("ticker"), "trade_date": a.get("trade_date"), "interim": _interim(a),
            "stored_rm": _get(a, "stored", "rm"), "r1_rm": _get(a, "summary", "rm"), "r2_rm": _get(b, "summary", "rm"),
            "stored_final": _get(a, "stored", "final"), "r1_final": _get(a, "summary", "final"), "r2_final": _get(b, "summary", "final"),
            "stored_neutral": _get(a, "stored", "neutral_final"), "r1_neutral": _get(a, "summary", "neutral_final"),
            "r2_neutral": _get(b, "summary", "neutral_final"),
            "r1_conf": _get(a, "summary", "confidence"), "r2_conf": _get(b, "summary", "confidence"),
            "change_kind": " + ".join(kinds),
        })
    return out


CSV_COLUMNS = [
    "case_id", "ticker", "trade_date", "interim", "stored_rm", "r1_rm", "r2_rm", "stored_final", "r1_final",
    "r2_final", "stored_neutral", "r1_neutral", "r2_neutral", "r1_conf", "r2_conf", "change_kind",
]


def changed_cases_csv(rows: Sequence[Mapping[str, Any]]) -> str:
    buf = io.StringIO(newline="")
    writer = csv.DictWriter(buf, fieldnames=CSV_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({k: row.get(k) for k in CSV_COLUMNS})
    return buf.getvalue()


# --------------------------------------------------------------------------- rendering


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return "" if value is None else str(value)


def render_markdown(report: Mapping[str, Any], changed: Sequence[Mapping[str, Any]]) -> str:
    lines = [f"# A/A report", "", f"**Verdict: {report['verdict']}**  (n = {report['n_population']} cases in C)", "",
             "| ID | metric | value | threshold | verdict | n |", "|---|---|---|---|---|---|"]
    for r in report["rows"]:
        lines.append(f"| {r['id']} | {r['metric']} | {_fmt(r['value'])} | {r['threshold']} | {r['verdict']} | {_fmt(r['n'])} |")
    split = report["interim"]
    lines += ["", "## AA-3d detail (interim split)", "",
              f"- A (interim True): d = {_fmt(split['d_A'])}, n = {split['n_A']}; B (False U None): d = {_fmt(split['d_B'])}, n = {split['n_B']}",
              f"- Fisher exact p (informational, does not decide): {_fmt(split['fisher_p'])}",
              f"- three-way: {json.dumps(split['three_way'], ensure_ascii=False)}",
              f"- restore period_note needed: {split['restore_period_note_needed']}"]
    lines += ["", "## Completion (ok / case set, first-attempt rate)", "",
              "```", json.dumps(report["completion"], indent=1), "```"]
    inv = report["invariants"]
    lines += ["", "## DB invariants", "", f"- status: {inv.get('status')}"]
    lines += [f"- FAIL: {m}" for m in inv.get("fails", [])] + [f"- WARN: {m}" for m in inv.get("warns", [])]
    for r in inv.get("candidates", []):
        lines.append(f"- FAIL CANDIDATE (needs human confirmation, --confirm-live-rows): {r['table']} id={r['id']} {r.get('ticker')} "
                     f"created {r.get('at')} is_backtest={r.get('is_backtest')} account_mode={r.get('account_mode')}")
    for r in inv.get("released", []):
        lines.append(f"- released by the human as live: {r['table']} id={r['id']} {r.get('ticker')}")
    if inv.get("confirm"):
        lines.append(f"- confirmation file: {json.dumps(inv['confirm'], ensure_ascii=False)}")
    lines.append(f"- harness run windows: {json.dumps(inv.get('run_windows', []))}")
    lines += ["", "AA-1d note: the denominator counts HTTP-call-level records (`metrics.record`), so a native "
              "structured failure followed by its fallback is 2 records; the ratio is per record, not per logical call."]
    lines += ["", "## Rating distributions (stored / R1 / R2)", ""]
    for key, groups in report["distributions"].items():
        lines.append(f"- **{key}**: " + "; ".join(f"{g}: {json.dumps(d, ensure_ascii=False)}" for g, d in groups.items()))
    lines += ["", f"- confidence mean: {json.dumps(report['confidence_mean'])}", f"- gates fired: {json.dumps(report['gates'])}",
              "", "## Resources (per case)", "", f"```\n{json.dumps(report['resources'], indent=1)}\n```",
              "", "## RM disagreement vs stored by covariate", "", f"```\n{json.dumps(report['covariate_rm_mismatch'], indent=1, ensure_ascii=False)}\n```",
              "", f"## Changed cases: {len(changed)}", ""]
    if changed:
        lines += ["| case | interim | stored rm/final | R1 rm/final | R2 rm/final | kind |", "|---|---|---|---|---|---|"]
        for c in changed:
            lines.append(
                f"| {c['case_id']} | {c['interim']} | {c['stored_rm']}/{c['stored_final']} | {c['r1_rm']}/{c['r1_final']} "
                f"| {c['r2_rm']}/{c['r2_final']} | {c['change_kind']} |"
            )
    lines += ["", "No alpha is scored here; a single A/A run never validates a variant."]
    return "\n".join(lines) + "\n"


def write_report(out_dir: Path, report: Mapping[str, Any], changed: Sequence[Mapping[str, Any]]) -> dict[str, Path]:
    from app.experiments.replay.contract import atomic_write_text

    paths = {
        "json": out_dir / "aa_report.json", "md": out_dir / "aa_report.md",
        "csv": out_dir / "aa_changed_cases.csv", "changed_md": out_dir / "aa_changed_cases.md",
    }
    atomic_write_text(paths["json"], json.dumps(report, indent=1, ensure_ascii=False, default=str) + "\n")
    atomic_write_text(paths["md"], render_markdown(report, changed))
    atomic_write_text(paths["csv"], changed_cases_csv(changed))
    md = ["# Changed cases (ratings only; safe to copy to handoff)", "", "| case | interim | S rm/final/neutral | R1 | R2 | kind |", "|---|---|---|---|---|---|"]
    for c in changed:
        md.append(
            f"| {c['case_id']} | {c['interim']} | {c['stored_rm']}/{c['stored_final']}/{c['stored_neutral']} "
            f"| {c['r1_rm']}/{c['r1_final']}/{c['r1_neutral']} | {c['r2_rm']}/{c['r2_final']}/{c['r2_neutral']} | {c['change_kind']} |"
        )
    atomic_write_text(paths["changed_md"], "\n".join(md) + "\n")
    return paths


def load_confirm_file(path: Path) -> tuple[dict[str, list[int]], dict[str, Any]]:
    """`{"memory_entries": [ids], "trades": [ids], "pending_orders": [ids]}` verified by the human."""
    data = path.read_bytes()
    raw = json.loads(data.decode("utf-8"))
    if not isinstance(raw, dict):
        raise PairError("confirm-live-rows file must be a JSON object {table: [ids]}")
    allowed = {"memory_entries", "trades", "pending_orders"}
    bad = sorted(set(raw) - allowed)
    if bad:
        raise PairError(f"confirm-live-rows file has unknown tables {bad}; allowed {sorted(allowed)}")
    ids = {t: [int(i) for i in v] for t, v in raw.items()}
    import hashlib

    return ids, {"path": str(path), "sha256": hashlib.sha256(data).hexdigest(), "ids": ids}


def make_report(
    r1_dir: Path, r2_dir: Path, out_root: Path, stamp: str, confirm_file: Path | None = None
) -> tuple[dict[str, Any], dict[str, Path]]:
    r1, r2 = load_run(r1_dir), load_run(r2_dir)
    validate_pair(r1, r2)
    confirmed, info = load_confirm_file(confirm_file) if confirm_file else (None, None)
    report = build_report(r1, r2, confirmed=confirmed, confirm_info=info)
    ids = population(r1, r2)
    paths = write_report(out_root / f"aa_{stamp}", report, changed_cases(r1, r2, ids))
    return report, paths
