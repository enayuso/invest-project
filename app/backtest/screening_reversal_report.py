"""T2: merge per-chunk `screening_sweep --dump-json` dumps and compute the
pre-registered statistics of the screening-reversal validation (stage 2).

Run offline (no network, no DB, no LLM)::

    python -m app.backtest.screening_reversal_report --grid dev_h4_c4 --mode check

Modes (a three-step gate so the analysed set cannot change silently):
  check  integrity + acceptance + reconstruction + instrument checks only; prints NO metric numbers
  freeze check passes -> writes accepted_<grid>.json (chunk sha256 list)
  report requires that manifest, re-verifies every sha256, then computes all statistics

Design: handoff/knowledge/screening-reversal-t2-design-2026-10-01.md. Pre-registration:
screening-reversal-stage1-prereg-2026-09-30.md (section numbers are cited in comments).
Tooling only (RULES section 2 item 4): nothing in the live path imports this module, and it
writes only under the gitignored `results/` folder. It deliberately does NOT import
`app.backtest.screening_sweep` (heavy); its few JSON helpers are re-written here.
오프라인 구간 덤프 병합 + 사전 등록 통계 (판정 문구는 코드·규칙 ID에서만 생성).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import statistics
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

from app.backtest import factor_metrics as fm
from app.chat.screening import risk_adjusted_score
from app.domain.residual_momentum import residual_pct

# --- pre-registered constants (comment = prereg / design section) ----------

DUMP_SCHEMA = "screening-sweep-dump-v1"
REPORT_SCHEMA = "screening-reversal-report-v1"
ACCEPTED_SCHEMA = "screening-reversal-accepted-v1"

PRIMARY = "rev_lb5"  # H1, prereg A
BASELINE = "live_mom_lb15"  # H2, prereg A
EXPLORATORY = ("rev_lb10", "rev_lb15", "rev_lb20", "rev_lb5_raw", "rev_lb5_resid")  # Holm family, prereg E
DIAGNOSTIC = "mom_lb5"  # instrument check, prereg A
COST_RT = 0.0030  # round-trip cost, prereg D
COST_STRESS = (0.0050, 0.0080)  # prereg D
MME = {4: 0.0030, 10: 0.0050}  # minimum meaningful effect per holding, prereg E
ALPHA_1S = 0.025  # one-sided alpha, prereg E
N_EFF_MIN = 20  # prereg E-1
BOOT_N = 10_000  # fixed by prereg
SEED = 0
HALF_SPLIT = {"dev": date(2026, 3, 1)}  # prereg E robustness (period split)
MIN_DATES = {4: 55, 10: 24}  # prereg G
FALLBACK_CHUNK_MAX = 8  # runbook 3.1
FALLBACK_GRID_MAX = 50  # prereg G
SCORED_MIN_NUM, SCORED_MIN_DEN = 80, 86  # prereg G: scored/priced >= ceil(|U| * 80 / 86)
RESID_FB_MAX = 0.20  # prereg G
TRIM_K = 2  # prereg E robustness (trim 2 per side)
WINSOR = (2.5, 97.5)  # prereg E robustness
CONC_WARN = 0.5  # prereg E robustness (top-3 contribution share)
MBB_LEN = 3  # prereg C (moving block of 3 dates)
H3_RHO_MIN = 0.9  # prereg E H3
N_Q = 5
TOL = 1e-12

PREREG_VARIANTS: list[dict[str, Any]] = [  # prereg A, order matters
    {"label": "live_mom_lb15", "lookback_days": 15, "use_volatility_adjustment": True, "use_correlation_dedup": True, "sign": 1, "momentum_mode": "total"},
    {"label": "rev_lb5", "lookback_days": 5, "use_volatility_adjustment": True, "use_correlation_dedup": True, "sign": -1, "momentum_mode": "total"},
    {"label": "rev_lb10", "lookback_days": 10, "use_volatility_adjustment": True, "use_correlation_dedup": True, "sign": -1, "momentum_mode": "total"},
    {"label": "rev_lb15", "lookback_days": 15, "use_volatility_adjustment": True, "use_correlation_dedup": True, "sign": -1, "momentum_mode": "total"},
    {"label": "rev_lb20", "lookback_days": 20, "use_volatility_adjustment": True, "use_correlation_dedup": True, "sign": -1, "momentum_mode": "total"},
    {"label": "rev_lb5_raw", "lookback_days": 5, "use_volatility_adjustment": False, "use_correlation_dedup": True, "sign": -1, "momentum_mode": "total"},
    {"label": "rev_lb5_resid", "lookback_days": 5, "use_volatility_adjustment": True, "use_correlation_dedup": True, "sign": -1, "momentum_mode": "residual"},
    {"label": "mom_lb5", "lookback_days": 5, "use_volatility_adjustment": True, "use_correlation_dedup": True, "sign": 1, "momentum_mode": "total"},
]
HOLDOUT_LABELS = ["rev_lb5", "live_mom_lb15", "mom_lb5"]  # prereg G (holdout variants)

LOG_NEEDLES = (  # same meaning as the runner's CountOf: case-insensitive, per line
    ("fallback_price", "KIS get_price_series failed"),
    ("atr_fallback", "KIS get_atr_pct failed"),
    ("price_unavailable", "price series unavailable"),
    ("cluster_failed", "correlation clustering failed"),
    ("benchmark_failed", "benchmark fetch failed"),
    ("read_timeout", "ReadTimeout"),
    ("egw00201", "EGW00201"),
)
OBS_KEYS = (
    "picks", "pick_forward_returns", "universe_forward_returns", "excess_return", "rank_ic",
    "quantile_spread", "turnover", "residual_fallbacks", "dedup_drop_rate", "dedup_degraded", "signed_scores",
)
IGNORED_ARGS = ("start", "end", "dump_json")  # args that legitimately differ per chunk
T2_PATHS = ("app/backtest/screening_reversal_report.py", "tests/backtest/test_screening_reversal_report.py")

# Verdict wording lives in ONE place: code -> (English, Korean, prereg section). 판정 문구 단일 출처.
VERDICT_TEXT: dict[str, tuple[str, str, str]] = {
    "INVALID": ("rejected by the acceptance rules, re-run", "수용 기준 실패, 무효(재실행)", "prereg G, E-0"),
    "UNDERPOWERED": ("inconclusive (not enough evidence)", "결론 불가", "prereg E-1/E-5"),
    "INCONSISTENT": ("RankIC and excess return disagree in sign: treat as noise", "RankIC와 초과수익 부호 불일치: 노이즈", "prereg E-2"),
    "PASS": ("net CI lower bound above zero", "순 CI 하한 > 0", "prereg E-3"),
    "FAIL": ("net CI upper bound below the minimum meaningful effect", "순 CI 상한 < 최소 의미 효과", "prereg E-4"),
    "SUPPORTED": ("H2 supported: live momentum underperforms its universe", "H2 지지", "prereg E (H2)"),
    "REFUTED": ("H2 refuted: live momentum beats its universe", "H2 반박", "prereg E (H2)"),
    "INCONCLUSIVE": ("H2 inconclusive", "H2 결론 불가", "prereg E (H2)"),
    "H3_SUPPORTED": ("quantile monotonicity supported", "H3 지지", "prereg E (H3)"),
    "TAIL_ONLY": ("only the Q5-Q1 tail separates", "꼬리 효과만", "prereg E (H3)"),
    "NOT_SUPPORTED": ("quantile monotonicity not supported", "H3 불지지", "prereg E (H3)"),
}


class InputError(Exception):
    """Bad usage or missing input file -> exit code 2."""


@dataclass
class Check:
    id: str
    prereg_ref: str
    ok: bool
    severity: str  # "integrity" | "acceptance" | "warning"
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "prereg_ref": self.prereg_ref, "ok": self.ok, "severity": self.severity, "detail": self.detail}


# --- small JSON / file helpers (re-written, not imported from screening_sweep) ------

def _jsonable(obj: Any) -> Any:
    """NaN/inf -> None, date -> ISO, tuple -> list, keys -> str. 엄격 JSON 보장."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_bytes().decode("utf-8-sig"))
    except ValueError as exc:  # JSONDecodeError and UnicodeDecodeError are ValueErrors
        raise InputError(f"malformed JSON in {path}: {exc}") from exc


def _write_atomic(path: Path, payload: str) -> None:
    """tmp in the same folder + os.replace, UTF-8 without BOM."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_bytes(payload.encode("utf-8"))
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _mean(xs: list[float]) -> float | None:
    """Plain sum/len, the same formula the sweep tool uses for its date means."""
    return (sum(xs) / len(xs)) if xs else None


def _git(*cmd: str) -> str | None:
    cwd = str(Path(__file__).resolve().parents[2])
    try:
        out = subprocess.run(["git", *cmd], cwd=cwd, capture_output=True, text=True, timeout=15, check=True)
        return out.stdout
    except Exception:  # noqa: BLE001 - provenance is best-effort
        return None


def git_changed_paths(head_a: str, head_b: str) -> list[str] | None:
    """Paths changed between two commits; None when git cannot answer. (monkeypatched in tests)"""
    out = _git("diff", "--name-only", head_a, head_b)
    return None if out is None else [p.strip() for p in out.splitlines() if p.strip()]


def role_of(grid: str) -> str:
    if grid.startswith("dev_"):
        return "dev"
    if grid.startswith("val_") or grid.startswith("holdout"):
        return "holdout"
    raise InputError(f"unknown grid role for {grid!r}: expected dev_*, val_* or holdout*")


# --- input loading ------------------------------------------------------------

def load_plan(base: Path, grid: str) -> dict[str, Any]:
    path = base / "chunks.json"
    if not path.is_file():
        raise InputError(f"missing input {path}")
    plan = _read_json(path)
    g = ((plan.get("grids") if isinstance(plan, dict) else None) or {}).get(grid)
    if g is None:
        raise InputError(f"grid {grid!r} not in {path}")
    if (not isinstance(g, dict) or any(k not in g for k in ("chunks", "all_dates", "holding", "cadence")) or not isinstance(g["chunks"], list)
            or any(not isinstance(c, dict) or not {"chunk", "dates"} <= set(c) for c in g["chunks"])):
        raise InputError(f"malformed grid entry {grid!r} in {path}")
    if g.get("holding") not in MME:
        raise InputError(f"holding {g.get('holding')!r} has no pre-registered minimum effect (MME)")
    return g


def load_runlog(base: Path) -> list[dict[str, Any]]:
    path = base / "runlog.jsonl"
    if not path.is_file():
        raise InputError(f"missing input {path}")
    recs: list[dict[str, Any]] = []
    for raw in path.read_bytes().decode("utf-8-sig").splitlines():
        line = raw.lstrip("﻿").strip()  # first line carries a BOM (PowerShell Add-Content)
        if line:
            try:
                rec = json.loads(line)
            except ValueError as exc:
                raise InputError(f"broken runlog line in {path}: {exc}") from exc
            if not isinstance(rec, dict):
                raise InputError(f"runlog line is not an object in {path}")
            recs.append(rec)
    return recs


_PROGRESS_ANY = re.compile(r"^\[\d+/\d+\]")
_PROGRESS_FULL = re.compile(r"^\[(\d+)/(\d+)\]\s+(\d{4}-\d{2}-\d{2})\s+scored=(\d+)\s+priced=(\d+)")
_FALLBACK_TICKER = re.compile(r"failed for (\S+?),", re.IGNORECASE)


def scan_log(path: Path) -> dict[str, Any]:
    """Count the seven runner needles (case-insensitive, per line) and the `[i/N]` progress lines.
    Price fallbacks between `[i-1/N]` and `[i/N]` belong to date i, after `[N/N]` to phase B.
    폴백 줄을 날짜(또는 phaseB)에 귀속 — 서술용."""
    text = path.read_bytes().decode("utf-8-sig", errors="replace")
    lines = text.splitlines()
    counts = {key: sum(1 for ln in lines if needle.lower() in ln.lower()) for key, needle in LOG_NEEDLES}
    progress: list[dict[str, Any]] = []
    fallbacks: list[dict[str, str]] = []
    pending: list[str] = []
    n_progress = 0
    for ln in lines:
        if _PROGRESS_ANY.match(ln):
            n_progress += 1
            m = _PROGRESS_FULL.match(ln)
            if m:
                progress.append({"i": int(m[1]), "n": int(m[2]), "date": m[3], "scored": int(m[4]), "priced": int(m[5])})
                fallbacks += [{"ticker": t, "at": m[3]} for t in pending]
                pending = []
        elif "kis get_price_series failed" in ln.lower():
            m2 = _FALLBACK_TICKER.search(ln)
            pending.append(m2[1] if m2 else "?")
    fallbacks += [{"ticker": t, "at": "phaseB"} for t in pending]
    return {"counts": counts, "n_progress": n_progress, "progress": progress, "fallbacks": fallbacks}


def _kst_naive_from_utc_iso(s: str) -> datetime:
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).replace(tzinfo=None) + timedelta(hours=9)


def link_runlog(chunk: str, dump: dict[str, Any], records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Runlog records of this chunk whose time window contains the dump: rec.start <= dump.started (KST)
    <= start + 5 min and dump.finished <= rec.end <= finished + 10 min. A 1 s tolerance covers the runner's
    second-truncated timestamps. Returns (matches, other attempts of the same chunk). 시간창으로 기록 연결."""
    started = _kst_naive_from_utc_iso(dump["started_at"])
    finished = _kst_naive_from_utc_iso(dump["finished_at"])
    one = timedelta(seconds=1)
    matches: list[dict[str, Any]] = []
    others = 0
    for rec in records:
        if rec.get("chunk") != chunk:
            continue
        try:
            rs, re_ = datetime.fromisoformat(rec["start_kst"]), datetime.fromisoformat(rec["end_kst"])
        except (KeyError, TypeError, ValueError):  # unusable record: counts as "another attempt", never a match
            others += 1
            continue
        if rs - one <= started <= rs + timedelta(minutes=5) and finished <= re_ + one and re_ <= finished + timedelta(minutes=10):
            matches.append(rec)
        else:
            others += 1
    return matches, others


@dataclass
class ChunkInfo:
    name: str
    path: Path
    dump: dict[str, Any]
    sha: str
    dates: list[str]
    scan: dict[str, Any] | None = None
    rec: dict[str, Any] | None = None
    discarded_attempts: int = 0
    txt_sha: str | None = None


@dataclass
class MergedGrid:
    grid: str
    role: str
    dates: list[str]
    universe: list[str]
    holding: int
    cadence: int
    top_n: int
    variants: list[dict[str, Any]]
    lookbacks: list[int]
    scores: dict[tuple[int, str], dict[str, dict[str, float | None]]]
    benchmark: dict[tuple[int, str], dict[str, float | None]]
    forward: dict[str, dict[str, tuple[float | None, str | None, str | None]]]
    obs: dict[str, dict[str, dict[str, Any]]]
    chunk_of: dict[str, str]
    worth_bar: float = 0.03
    extra: dict[str, Any] = field(default_factory=dict)  # check-time details (reconstruction, recheck, instrument)


def merge(plan: dict[str, Any], chunks: dict[str, ChunkInfo], grid: str, role: str) -> MergedGrid:
    first = next(iter(chunks.values())).dump
    variants = [v["variant"] for v in first["variants"]]
    merged = MergedGrid(
        grid=grid, role=role, dates=[], universe=list(first["universe"]), holding=int(first["holding_days"]),
        cadence=int(first["cadence_days"]), top_n=int(first["top_n"]), variants=variants,
        lookbacks=sorted({v["lookback_days"] for v in variants}), scores={}, benchmark={}, forward={},
        obs={v["label"]: {} for v in variants}, chunk_of={}, worth_bar=float((first.get("args") or {}).get("worth_bar", 0.03)),
    )
    for pc in plan["chunks"]:
        info = chunks.get(pc["chunk"])
        if info is None:
            continue
        d = info.dump
        for day in d["trade_dates"]:
            if day in merged.chunk_of:
                raise InputError(f"date {day} present in two chunks")
            merged.dates.append(day)
            merged.chunk_of[day] = info.name
            merged.forward[day] = {
                t: (v["fwd_return"], v["entry_date"], v["exit_price_date"]) for t, v in d["forward"][day].items()
            }
            for lb in merged.lookbacks:
                merged.scores[(lb, day)] = d["scores"][f"{lb}|{day}"]
                merged.benchmark[(lb, day)] = (d.get("benchmark_pct") or {}).get(f"{lb}|{day}", {})
        for v in d["variants"]:
            merged.obs[v["variant"]["label"]].update(v["observations"])
    return merged


# --- validation (integrity I*, acceptance A*) ---------------------------------

def _load_cost_fn() -> Callable[[str, str], float]:
    from app.execution.costs import transaction_cost_bps_for  # lazy: pulls settings

    return transaction_cost_bps_for


def cost_crosscheck(universe: list[str]) -> Check:
    """Warn (never fail) if the U86 mean round-trip cost, computed like the sweep's `_universe_round_trip_cost`
    (buy+sell bps per ticker / 10000, averaged), differs from COST_RT. 비용 교차 확인(경고만)."""
    try:
        fn = _load_cost_fn()
        per = [(fn(t, "buy") + fn(t, "sell")) / 10_000 for t in universe]
        rt = sum(per) / len(per)
    except Exception as exc:  # noqa: BLE001 - cross-check is best-effort
        return Check("C1", "prereg D", False, "warning", f"cost cross-check unavailable: {type(exc).__name__}: {exc}")
    ok = abs(rt - COST_RT) <= TOL
    return Check("C1", "prereg D", ok, "warning", f"universe mean round-trip {rt * 100:.4f}% == COST_RT {COST_RT * 100:.4f}%" if ok else f"universe mean round-trip {rt * 100:.4f}% differs from COST_RT {COST_RT * 100:.4f}%")


def _chk(checks: list[Check], id_: str, ref: str, ok: bool, detail: str, severity: str = "integrity") -> bool:
    checks.append(Check(id_, ref, bool(ok), severity, detail))
    return bool(ok)


def _structure_problems(d: dict[str, Any]) -> list[str]:
    """Key structure the later stages rely on (I5)."""
    need = ("universe", "trade_dates", "trade_dates_requested", "holding_days", "cadence_days", "top_n", "benchmark",
            "scores", "benchmark_pct", "forward", "variants", "args", "git_head", "git_dirty", "started_at", "finished_at")
    miss = [k for k in need if k not in d]
    if miss:
        return [f"missing keys {miss}"]
    probs: list[str] = []
    try:
        dates = list(d["trade_dates"])
        universe = list(d["universe"])
        if set(d["forward"]) != set(dates):
            probs.append("forward dates != trade_dates")
        for day, per in d["forward"].items():
            if list(per) != universe:
                probs.append(f"forward tickers != universe on {day}")
                break
            if any(set(v) != {"fwd_return", "entry_date", "exit_price_date"} for v in per.values()):
                probs.append(f"forward entry keys on {day}")
                break
        variants = d["variants"]
        if not variants or any(not {"variant", "observations", "summary"} <= set(v) for v in variants):
            probs.append("variants entries need variant/observations/summary")
            return probs
        lbs = sorted({v["variant"]["lookback_days"] for v in variants})
        for lb in lbs:
            for day in dates:
                key = f"{lb}|{day}"
                s = d["scores"].get(key)
                if s is None or any(set(x) != {"raw_pct", "atr_pct"} for x in s.values()):
                    probs.append(f"scores[{key}] missing or malformed")
                    break
        has_resid = any(v["variant"]["momentum_mode"] == "residual" for v in variants)
        for v in variants:
            for day, o in v["observations"].items():
                if day not in dates or any(k not in o for k in OBS_KEYS) or not isinstance(o["signed_scores"], dict):
                    probs.append(f"observation {v['variant']['label']}@{day} malformed")
                    break
        if has_resid:
            bad = [f"{lb}|{day}" for lb in lbs for day in dates if f"{lb}|{day}" not in d["benchmark_pct"]]
            if bad:
                probs.append(f"benchmark_pct missing {bad[:2]}")
            if (d.get("benchmark") or {}).get("markets") != ["domestic"]:
                probs.append("benchmark.markets != ['domestic']")
    except (KeyError, TypeError, AttributeError) as exc:
        probs.append(f"structure error {type(exc).__name__}: {exc}")
    return probs


def expected_variants(role: str) -> list[dict[str, Any]]:
    if role == "dev":
        return PREREG_VARIANTS
    by = {v["label"]: v for v in PREREG_VARIANTS}
    return [by[x] for x in HOLDOUT_LABELS]


def _progress_problems(i: ChunkInfo) -> list[str]:
    """The .txt progress lines must name the chunk's dates, with scored/priced equal to the dump-derived counts
    (so a swapped .txt whose seven counts happen to match cannot pass). 진행 줄 날짜·건수 대조."""
    probs: list[str] = []
    prog = i.scan["progress"] if i.scan else []
    if i.scan and i.scan["n_progress"] == len(i.dates) and [p["date"] for p in prog] != i.dates:
        probs.append(f"{i.name}: progress dates != chunk dates")
        return probs
    lb0 = min(v["variant"]["lookback_days"] for v in i.dump["variants"])
    for p in prog:
        d = p["date"]
        if d not in i.dump["forward"]:
            continue
        sc = len(i.dump["scores"][f"{lb0}|{d}"])
        pr = sum(1 for x in i.dump["forward"][d].values() if x["fwd_return"] is not None)
        if (p["scored"], p["priced"]) != (sc, pr):
            probs.append(f"{i.name}: progress {d} scored/priced {p['scored']}/{p['priced']} != dump {sc}/{pr}")
    return probs


def _min_scored(universe_size: int) -> int:
    return math.ceil(universe_size * SCORED_MIN_NUM / SCORED_MIN_DEN)


def validate(
    base: Path, grid: str, role: str, plan: dict[str, Any], mode: str, opts: dict[str, Any],
) -> tuple[list[Check], dict[str, ChunkInfo], MergedGrid | None, list[dict[str, Any]]]:
    """Run I1-I10 and A1-A8 in dependency order. Returns (checks, chunk infos, merged grid or None, runlog)."""
    checks: list[Check] = []
    records = load_runlog(base)
    u86_path = base / "u86.txt"
    if not u86_path.is_file():
        raise InputError(f"missing input {u86_path}")
    u86 = [t for t in u86_path.read_text(encoding="utf-8-sig").strip().split(",") if t]
    allow_incomplete = bool(opts.get("allow_incomplete_set"))
    chunks: dict[str, ChunkInfo] = {}

    # I1 / I2: plan and files
    planned = [c["chunk"] for c in plan["chunks"]]
    concat = [d for c in plan["chunks"] for d in c["dates"]]
    ok1 = concat == plan["all_dates"] and len(set(concat)) == len(concat)
    _chk(checks, "I1", "runbook 1", ok1, f"{len(concat)} planned dates over {len(planned)} chunks, concat==all_dates, no overlap" if ok1 else "chunk dates != all_dates or overlap")
    found = sorted(p.stem for p in (base / "chunks").glob(f"{grid}_c*.json")) if (base / "chunks").is_dir() else []
    extra, missing = sorted(set(found) - set(planned)), sorted(set(planned) - set(found))
    ok2 = bool(found) and not extra and (not missing or allow_incomplete)
    _chk(checks, "I2", "runbook 4.1", ok2,
         f"{len(found)}/{len(planned)} chunk dumps" + (f", unplanned {extra}" if extra else "") + (f", missing {len(missing)} (allowed)" if missing and allow_incomplete else (f", missing {missing}" if missing else "")))
    if not (ok1 and ok2):
        return checks, chunks, None, records

    # I3 / I5: schema, status, structure, per dump
    probs3: list[str] = []
    probs5: list[str] = []
    plan_by = {c["chunk"]: c for c in plan["chunks"]}
    for name in found:
        path = base / "chunks" / f"{name}.json"
        try:
            dump = _read_json(path)
        except (OSError, InputError) as exc:  # malformed dump JSON is an integrity failure (exit 4), not a usage error
            probs3.append(f"{name}: unreadable ({exc})")
            continue
        if not isinstance(dump, dict):
            probs3.append(f"{name}: top level is not an object")
            continue
        if dump.get("schema") != DUMP_SCHEMA or dump.get("status") != "complete":
            probs3.append(f"{name}: schema={dump.get('schema')!r} status={dump.get('status')!r}")
        sp = _structure_problems(dump)
        if sp:
            probs5.append(f"{name}: {'; '.join(sp[:2])}")
        chunks[name] = ChunkInfo(name, path, dump, sha256_file(path), plan_by[name]["dates"])
    _chk(checks, "I3", "T1 dump", not probs3, f"{len(chunks)} dumps schema {DUMP_SCHEMA}, status complete" if not probs3 else "; ".join(probs3))
    _chk(checks, "I5", "T2 A", not probs5, "forward/scores/observations key structure ok" if not probs5 else "; ".join(probs5))
    if probs3 or probs5:
        return checks, chunks, None, records

    infos = list(chunks.values())
    d0 = infos[0].dump

    # I4: dates
    p4 = [i.name for i in infos if not (i.dump["trade_dates"] == i.dump["trade_dates_requested"] == i.dates)]
    _chk(checks, "I4", "runbook 2", not p4, "trade_dates == trade_dates_requested == planned chunk dates" if not p4 else f"mismatch in {p4}")

    # I6: cross-chunk identity
    p6: list[str] = []
    for i in infos[1:]:
        d = i.dump
        for key in ("universe", "holding_days", "cadence_days", "top_n", "benchmark"):
            if d[key] != d0[key]:
                p6.append(f"{i.name}.{key}")
        a, b = {k: v for k, v in d["args"].items() if k not in IGNORED_ARGS}, {k: v for k, v in d0["args"].items() if k not in IGNORED_ARGS}
        if a != b:
            p6.append(f"{i.name}.args")
    for i in infos:
        if i.dump["universe"] != u86:
            p6.append(f"{i.name}.universe!=u86.txt")
        if i.dump["holding_days"] != plan["holding"] or i.dump["cadence_days"] != plan["cadence"]:
            p6.append(f"{i.name}.holding/cadence!=plan")
    _chk(checks, "I6", "prereg B/C", not p6, "universe(order)/holding/cadence/top_n/benchmark/args identical, universe==u86.txt" if not p6 else "differs: " + ", ".join(p6[:6]))

    # I7: variant definitions
    exp = expected_variants(role)
    p7 = [i.name for i in infos if [v["variant"] for v in i.dump["variants"]] != exp]
    detail7 = f"{len(exp)} variants match prereg A in order" if not p7 else f"definition differs in {p7}"
    if role == "dev":
        vp = base / "variants_dev.json"
        if not vp.is_file():
            raise InputError(f"missing input {vp}")
        if _read_json(vp) != PREREG_VARIANTS:
            p7.append("variants_dev.json")
            detail7 = "variants_dev.json differs from prereg A"
    _chk(checks, "I7", "prereg A", not p7, detail7 if not p7 else f"definition differs in {p7}")
    if p4 or p6 or p7:
        return checks, chunks, None, records

    merged = merge(plan, chunks, grid, role)

    # I8: git heads
    heads = {i.name: i.dump.get("git_head") for i in infos}
    uniq = sorted({h for h in heads.values() if h}, key=lambda h: h)
    p8: list[str] = []
    note8 = "single git_head"
    if len(uniq) != 1 or any(h is None for h in heads.values()):
        allowed = [a.strip() for a in (opts.get("allow_heads") or []) if a.strip()]
        covered = lambda h: any(h.startswith(a) for a in allowed)  # noqa: E731
        ref = uniq[0] if uniq else None
        via_allow = False
        for h in uniq[1:]:
            changed = git_changed_paths(ref, h)
            if changed is None:  # --allow-heads only applies when git cannot answer; otherwise the path rule decides
                if allowed and covered(ref) and covered(h):
                    via_allow = True
                else:
                    p8.append(f"cannot diff {ref[:8]}..{h[:8]} (use --allow-heads)")
            elif any(p not in T2_PATHS and not p.endswith(".md") for p in changed):
                p8.append(f"{ref[:8]}..{h[:8]} changed {[p for p in changed if p not in T2_PATHS and not p.endswith('.md')][:3]}")
        if not uniq or any(h is None for h in heads.values()):
            p8.append("missing git_head")
        note8 = f"{len(uniq)} heads allowed by --allow-heads (git could not diff)" if via_allow else f"{len(uniq)} heads, differences limited to T2/tests/md"
    _chk(checks, "I8", "T2 Q6", not p8, note8 if not p8 else "; ".join(p8))

    # I9 / I10: runlog link and txt recount
    p9: list[str] = []
    p10: list[str] = []
    for i in infos:
        matches, others = link_runlog(i.name, i.dump, records)
        i.discarded_attempts = others
        if len(matches) != 1:
            p9.append(f"{i.name}: {len(matches)} matching runlog records")
        else:
            i.rec = matches[0]
            if not (i.dump["git_head"] or "").startswith(str(i.rec.get("git_head") or "\0")):
                p9.append(f"{i.name}: runlog git_head {i.rec.get('git_head')} !~ dump")
        txt = i.path.with_suffix(".txt")
        if not txt.is_file():
            p10.append(f"{i.name}: missing .txt")
            continue
        i.scan = scan_log(txt)
        i.txt_sha = sha256_file(txt)
        p10 += _progress_problems(i)
        if i.rec is not None:
            diff = [k for k, _ in LOG_NEEDLES if i.scan["counts"][k] != i.rec.get(k)]
            if diff:
                p10.append(f"{i.name}: runlog != txt for {diff}")
            if i.rec.get("progress_lines") not in (None, i.scan["n_progress"]):
                p10.append(f"{i.name}: progress_lines runlog != txt")
        if i.scan["n_progress"] != len(i.dates):
            p10.append(f"{i.name}: {i.scan['n_progress']} progress lines != {len(i.dates)} dates")
    n_att = sum(i.discarded_attempts for i in infos)
    _chk(checks, "I9", "runbook 4.1", not p9, f"{len(infos)} dumps each linked to exactly 1 runlog record, discarded attempts {n_att}" if not p9 else "; ".join(p9))
    _chk(checks, "I10", "prereg G", not p10, "txt recount of the 7 needles == runlog, progress lines == dates" if not p10 else "; ".join(p10))

    # I11 / I12 (need only the merged grid)
    sc = self_consistency(merged)
    _chk(checks, "I11", "T2 B", sc["mismatch"] == 0 and not sc["structure_problems"] and sc["turnover"]["mismatch"] == 0,
         f"reconstruction exact {sc['exact']}/{sc['total']}, near {sc['near']}, mismatch {sc['mismatch']}; turnover exact {sc['turnover']['exact']}/{sc['turnover']['total']}, mismatch {sc['turnover']['mismatch']}"
         + (f"; picks problems {len(sc['structure_problems'])}" if sc["structure_problems"] else ""))
    rc = [chunk_summary_recheck(i.dump) for i in infos]
    tot, hit = sum(r["total"] for r in rc), sum(r["matched"] for r in rc)
    _chk(checks, "I11b", "T2 B", tot == hit, f"chunk summary recheck exact {hit}/{tot}" if tot == hit else f"chunk summary mismatch {hit}/{tot}: {[m for r in rc for m in r['mismatches']][:3]}")
    inst = instrument_check(merged)
    _chk(checks, "I12", "prereg G", inst["failures"] == [], f"instrument ok {inst['ok']}/{inst['dates']} dates (tie-explained {inst['tie_explained']})" if not inst["failures"] else f"instrument failed on {inst['failures'][:3]}")
    checks.append(cost_crosscheck(merged.universe))
    sc_details = {"self_consistency": sc, "chunk_summary_recheck": rc, "instrument": inst}
    merged.extra = sc_details

    # acceptance A1-A8
    a1 = [i.name for i in infos if i.rec is None or i.rec.get("exit") != 0]
    _chk(checks, "A1", "prereg G", not a1, "all runlog exit codes 0" if not a1 else f"exit != 0 in {a1}", "acceptance")
    a2 = [f"{i.name}:{(i.rec or {}).get('fallback_price')}" for i in infos if i.rec is None or (i.rec.get("fallback_price") or 0) > FALLBACK_CHUNK_MAX]
    _chk(checks, "A2", "runbook 3.1", not a2, f"each chunk fallback <= {FALLBACK_CHUNK_MAX}" if not a2 else f"fallback > {FALLBACK_CHUNK_MAX}: {a2}", "acceptance")
    tot_fb = sum((i.rec or {}).get("fallback_price") or 0 for i in infos)
    _chk(checks, "A3", "runbook 3.1", tot_fb <= FALLBACK_GRID_MAX, f"grid fallback total {tot_fb} <= {FALLBACK_GRID_MAX}" if tot_fb <= FALLBACK_GRID_MAX else f"grid fallback total {tot_fb} > {FALLBACK_GRID_MAX}", "acceptance")
    need = _min_scored(len(merged.universe))
    lb0 = merged.lookbacks[0]
    low: list[str] = []
    min_scored = min_priced = None
    for day in merged.dates:
        n_sc = len(merged.scores[(lb0, day)])
        n_pr = sum(1 for v in merged.forward[day].values() if v[0] is not None)
        min_scored = n_sc if min_scored is None else min(min_scored, n_sc)
        min_priced = n_pr if min_priced is None else min(min_priced, n_pr)
        if n_sc < need or n_pr < need:
            low.append(day)
    _chk(checks, "A4", "prereg G, T2 Q5", not low, f"min scored {min_scored}, min priced {min_priced} >= {need}" if not low else f"scored/priced below {need} on {low[:4]}", "acceptance")
    deg = {v["label"]: sum(1 for o in merged.obs[v["label"]].values() if o["dedup_degraded"]) for v in merged.variants if v["use_correlation_dedup"]}
    bad_deg = {k: n for k, n in deg.items() if n}
    _chk(checks, "A5", "prereg G", not bad_deg, "dedup_degraded 0 on every dedup variant" if not bad_deg else f"dedup_degraded {bad_deg}", "acceptance")
    need_dates = MIN_DATES[merged.holding]
    if allow_incomplete:
        _chk(checks, "A6", "prereg G", True, f"skipped (incomplete set): {len(merged.dates)} dates, need {need_dates}", "warning")
    else:
        _chk(checks, "A6", "prereg G", len(merged.dates) >= need_dates, f"{len(merged.dates)} dates >= {need_dates}" if len(merged.dates) >= need_dates else f"{len(merged.dates)} dates < {need_dates}", "acceptance")
    mh = [i.name for i in infos if i.rec is None or i.rec.get("market_hours_override")]
    if mh and opts.get("accept_market_hours_override"):
        _chk(checks, "A7", "runbook 2", True, f"market-hours override accepted by flag for {mh}", "warning")
    else:
        _chk(checks, "A7", "runbook 2", not mh, "no market-hours override" if not mh else f"market_hours_override in {mh}", "acceptance")
    dirty = [i.name for i in infos if i.dump.get("git_dirty") is True]
    if not dirty:
        _chk(checks, "A8", "T2 A", True, "git_dirty false on every dump", "acceptance")
    elif mode == "check":
        _chk(checks, "A8", "T2 A", True, f"git_dirty in {dirty}: freeze/report would refuse", "warning")
    elif opts.get("allow_dirty"):
        _chk(checks, "A8", "T2 A", True, f"git_dirty in {dirty} allowed by --allow-dirty", "warning")
    else:
        _chk(checks, "A8", "T2 A", False, f"git_dirty true in {dirty}", "acceptance")
    return checks, chunks, merged, records


# --- reconstruction / self-consistency / instrument ---------------------------

def _cmp(a: Any, b: Any) -> str:
    if a is None and b is None:
        return "exact"
    if a is None or b is None:
        return "mismatch"
    if a == b:
        return "exact"
    return "near" if abs(a - b) <= TOL else "mismatch"


def _cmp_seq(a: list[Any], b: list[Any]) -> str:
    if len(a) != len(b):
        return "mismatch"
    st = [_cmp(x, y) for x, y in zip(a, b)]
    return "mismatch" if "mismatch" in st else ("near" if "near" in st else "exact")


def reconstruct_signed(v: dict[str, Any], scores: dict[str, dict[str, float | None]], bench: dict[str, float | None]) -> tuple[dict[str, float], int]:
    """Same formula as the tool's `_variant_score` (sweep :308-332, :650): residual numerator, optional ATR
    adjustment, then sign. Returns (signed scores in the stored key order, residual fallback count)."""
    out: dict[str, float] = {}
    fb = 0
    for t, s in scores.items():
        num = s["raw_pct"]
        if v["momentum_mode"] == "residual":
            b = bench.get("domestic")  # domestic-only is guaranteed by I5 (benchmark.markets) and I6 (universe == u86, all .KS)
            if b is None:
                fb += 1
            num = residual_pct(num, b)
        u = risk_adjusted_score(num, s["atr_pct"]) if v["use_volatility_adjustment"] else num
        out[t] = v["sign"] * u
    return out, fb


def recompute_observation(merged: MergedGrid, v: dict[str, Any], day: str, o: dict[str, Any]) -> tuple[dict[str, str], list[str]]:
    """Recompute one stored observation. Returns ({field: exact|near|mismatch}, structural problems)."""
    lb = v["lookback_days"]
    sc = merged.scores[(lb, day)]
    fw = {t: x[0] for t, x in merged.forward[day].items()}
    signed, fb = reconstruct_signed(v, sc, merged.benchmark[(lb, day)])
    res: dict[str, str] = {}
    stored = o["signed_scores"]
    res["signed_scores"] = "mismatch" if list(stored) != list(signed) else _cmp_seq(list(stored.values()), list(signed.values()))
    uni = [fw[t] for t in fw if t in sc and fw[t] is not None]
    res["universe_forward_returns"] = _cmp_seq(o["universe_forward_returns"], uni)
    picks = o["picks"]
    pick_fwd = [fw[t] for t in picks if fw.get(t) is not None]
    stored_pf = o["pick_forward_returns"]
    exp_pf = {t: fw[t] for t in picks if fw.get(t) is not None}
    res["pick_forward_returns"] = "mismatch" if list(stored_pf) != list(exp_pf) else _cmp_seq(list(stored_pf.values()), list(exp_pf.values()))
    res["excess_return"] = _cmp(o["excess_return"], fm.excess_return(pick_fwd, uni))
    res["rank_ic"] = _cmp(o["rank_ic"], fm.rank_ic(signed, fw))
    res["quantile_spread"] = _cmp(o["quantile_spread"], fm.quantile_spread(signed, fw))
    res["residual_fallbacks"] = "exact" if o["residual_fallbacks"] == fb else "mismatch"
    probs: list[str] = []
    if not picks or len(picks) > merged.top_n or len(set(picks)) != len(picks) or any(t not in sc for t in picks):
        probs.append(f"{v['label']}@{day}: picks not a duplicate-free subset of scored, 1..top_n")
    return res, probs


def self_consistency(merged: MergedGrid) -> dict[str, Any]:
    """Recompute every stored per-date value (picks themselves are not reproducible: the cluster map is
    not in the dump) and re-derive the merged-order turnover. 저장값 vs 재계산 비교."""
    tally = {"exact": 0, "near": 0, "mismatch": 0}
    by_field: dict[str, dict[str, int]] = {}
    mism: list[str] = []
    struct: list[str] = []
    tt = {"exact": 0, "near": 0, "mismatch": 0}
    first_ok = 0
    first_total = 0
    for v in merged.variants:
        label = v["label"]
        prev: list[str] | None = None
        seen_chunks: set[str] = set()
        for day in merged.dates:
            o = merged.obs[label].get(day)
            if o is None:
                continue
            res, probs = recompute_observation(merged, v, day, o)
            struct += probs
            for f, st in res.items():
                tally[st] += 1
                by_field.setdefault(f, {"exact": 0, "near": 0, "mismatch": 0})[st] += 1
                if st == "mismatch" and len(mism) < 10:
                    mism.append(f"{label}@{day}.{f}")
            merged_turn = fm.turnover(prev, o["picks"]) if prev is not None else None
            chunk = merged.chunk_of[day]
            is_first = chunk not in seen_chunks  # tool restarts prev_picks per chunk (sweep :634)
            seen_chunks.add(chunk)
            if is_first:
                first_total += 1
                first_ok += 1 if o["turnover"] is None else 0
                if o["turnover"] is not None:
                    tt["mismatch"] += 1
                    mism.append(f"{label}@{day}.turnover(first date of chunk must be null)")
                else:
                    tt["exact"] += 1
            else:
                st = _cmp(o["turnover"], merged_turn)
                tt[st] += 1
                if st == "mismatch" and len(mism) < 10:
                    mism.append(f"{label}@{day}.turnover")
            prev = o["picks"]
    return {
        **tally, "total": sum(tally.values()), "by_field": by_field, "mismatches": mism, "structure_problems": struct,
        "turnover": {**tt, "total": sum(tt.values()), "chunk_first_null": f"{first_ok}/{first_total}"},
    }


def merged_turnover(merged: MergedGrid, label: str) -> dict[str, float | None]:
    """Turnover recomputed across chunk boundaries in merged date order (descriptive only)."""
    out: dict[str, float | None] = {}
    prev: list[str] | None = None
    for day in merged.dates:
        o = merged.obs[label].get(day)
        if o is None:
            continue
        out[day] = fm.turnover(prev, o["picks"]) if prev is not None else None
        prev = o["picks"]
    return out


def _chunk_variant_summary(v: dict[str, Any], observations: dict[str, dict[str, Any]], args: dict[str, Any], holding: int, cadence: int) -> dict[str, Any]:
    """Mirror of sweep `_summarise_variant` (:712-778) over one dump's own dates."""
    obs = list(observations.values())
    excesses = [o["excess_return"] for o in obs if o["excess_return"] is not None]
    ics = [o["rank_ic"] for o in obs]
    spreads = [o["quantile_spread"] for o in obs if o["quantile_spread"] is not None]
    n_iter, seed = int(args["bootstrap_iters"]), int(args["seed"])
    ic_ok = [x for x in ics if x is not None]
    ci_e = fm.block_bootstrap_ci([[x] for x in excesses], n_bootstrap=n_iter, seed=seed)
    ci_i = fm.block_bootstrap_ci([[x] for x in ic_ok], n_bootstrap=n_iter, seed=seed)
    return {
        "mean_excess_return": (sum(excesses) / len(excesses)) if excesses else None,
        "excess_return_ci95": list(ci_e) if ci_e else None,
        "mean_rank_ic": (sum(ic_ok) / len(ic_ok)) if ic_ok else None,
        "rank_ic_ci95": list(ci_i) if ci_i else None,
        "icir": fm.icir(ics),
        "mean_quantile_spread": (sum(spreads) / len(spreads)) if spreads else None,
        "n_eff": fm.effective_sample_size(len(obs), cadence, holding),
    }


def chunk_summary_recheck(dump: dict[str, Any]) -> dict[str, Any]:
    """Recompute each stored variant `summary` from the dump's own observations; exact equality proves
    this module's bootstrap replica is bit-identical to the tool's."""
    total = matched = 0
    mism: list[str] = []
    for v in dump["variants"]:
        want = _chunk_variant_summary(v["variant"], v["observations"], dump["args"], dump["holding_days"], dump["cadence_days"])
        total += 1
        bad = [k for k, x in want.items() if v["summary"].get(k) != x]
        if bad:
            mism.append(f"{v['variant']['label']}:{bad}")
        else:
            matched += 1
    return {"matched": matched, "total": total, "mismatches": mism}


def _tie_crosses_boundary(scores: dict[str, float | None], fw: dict[str, float | None]) -> bool:
    """True when equal scores straddle the 20% bottom/top cut (stable sort then splits them by dict order)."""
    xs = [s for t, s in scores.items() if s is not None and fw.get(t) is not None]
    n = len(xs)
    if n < 5:
        return False
    k = max(1, round(n * 0.2))
    ordered = sorted(xs)
    return ordered[k - 1] == ordered[k] or ordered[n - k - 1] == ordered[n - k]


def instrument_check(merged: MergedGrid) -> dict[str, Any]:
    """Per date: mom_lb5 is the exact mirror of rev_lb5 (signed scores, IC, quantile spread)."""
    fails: list[str] = []
    ok = ties = n = 0
    if DIAGNOSTIC not in merged.obs or PRIMARY not in merged.obs:
        return {"dates": 0, "ok": 0, "tie_explained": 0, "failures": [f"{DIAGNOSTIC}/{PRIMARY} missing"]}
    for day in merged.dates:
        r, m = merged.obs[PRIMARY].get(day), merged.obs[DIAGNOSTIC].get(day)
        if r is None and m is None:
            continue
        n += 1
        if r is None or m is None:
            fails.append(f"{day}: observation on one side only")
            continue
        sr, sm = r["signed_scores"], m["signed_scores"]
        why: list[str] = []
        if set(sr) != set(sm) or any(sm[t] != -sr[t] for t in sr if sr[t] is not None and t in sm):
            why.append("signed")
        ic_r, ic_m = r["rank_ic"], m["rank_ic"]
        if (ic_r is None) != (ic_m is None) or (ic_r is not None and abs(ic_r + ic_m) > TOL):
            why.append("ic")
        sp_r, sp_m = r["quantile_spread"], m["quantile_spread"]
        tie = False
        if (sp_r is None) != (sp_m is None) or (sp_r is not None and abs(sp_r + sp_m) > TOL):
            fw = {t: x[0] for t, x in merged.forward[day].items()}
            if _tie_crosses_boundary(sr, fw):
                tie = True
            else:
                why.append("spread")
        if why:
            fails.append(f"{day}: {','.join(why)}")
        else:
            ok += 1
            ties += 1 if tie else 0
    return {"dates": n, "ok": ok, "tie_explained": ties, "failures": fails}


# --- statistics ---------------------------------------------------------------

def bootstrap_means(values: list[float], n_bootstrap: int, seed: int | None) -> list[float] | None:
    """Sorted bootstrap means; replica of `factor_metrics.block_bootstrap_ci` for one value per block
    (same RNG call order, same `statistics.mean`) so results are bit-identical. <3 blocks -> None."""
    blocks = [[v] for v in values]
    if len(blocks) < 3:
        return None
    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(n_bootstrap):
        resampled = [rng.choice(blocks) for _ in blocks]
        pooled = [v for block in resampled for v in block]
        means.append(statistics.mean(pooled) if pooled else 0.0)
    means.sort()
    return means


def ci_from_means(means: list[float] | None, confidence: float = 0.95) -> tuple[float, float] | None:
    if means is None:
        return None
    lo_pct = (1 - confidence) / 2 * 100
    hi_pct = (1 - (1 - confidence) / 2) * 100
    return (fm.percentile(means, lo_pct), fm.percentile(means, hi_pct))


def p_boot_one_sided(means: list[float] | None) -> float | None:
    """Exact bootstrap p for H0 mean <= 0 (Davison-Hinkley): (1 + #{m <= 0}) / (B + 1)."""
    if means is None:
        return None
    return (1 + sum(1 for m in means if m <= 0)) / (len(means) + 1)


def boundary_disagree(ci: tuple[float, float] | None, p_boot: float | None) -> bool:
    """True when the CI rule (lower bound > 0) and the exact bootstrap p (<= alpha) give different answers,
    e.g. #{m<=0}=250 at B=10,000: CI lower bound > 0 but p=0.025097 > 0.025. 경계에서 두 규칙이 갈림."""
    if ci is None or p_boot is None:
        return False
    return (ci[0] > 0) != (p_boot <= ALPHA_1S)


def boundary_flag(n30: dict[str, Any]) -> str | None:
    """Verdict-line note when the CI rule and p_boot disagree (the verdict itself stays CI-rule-based, prereg E)."""
    if not n30.get("boundary_disagree"):
        return None
    lo, p = n30["ci"][0], n30["p_boot"]
    return f"boundary: CI lower bound {'>' if lo > 0 else '<='} 0 but p_boot={p:.6f} {'<=' if p <= ALPHA_1S else '>'} {ALPHA_1S} (verdict stays CI-rule-based)"


def p_normal_one_sided(mean: float | None, ci: tuple[float, float] | None) -> float | None:
    """Prereg E stage-2 normal approximation: SE = (hi - lo) / 3.92, p = 1 - Phi(mean / SE). Reference only."""
    if mean is None or ci is None:
        return None
    se = (ci[1] - ci[0]) / 3.92
    if se <= 0:
        return 0.0 if mean > 0 else (1.0 if mean < 0 else 0.5)
    return 0.5 * math.erfc((mean / se) / math.sqrt(2))


def moving_block_ci(values: list[float], n_bootstrap: int, seed: int | None, block_len: int = MBB_LEN) -> tuple[float, float] | None:
    """Overlapping moving-block bootstrap (blocks of `block_len` dates, ceil(n/L) draws, truncated to n)."""
    n = len(values)
    if n < 5:
        return None
    k, starts = math.ceil(n / block_len), n - block_len + 1
    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(n_bootstrap):
        pooled: list[float] = []
        for _ in range(k):
            s = rng.randrange(starts)
            pooled.extend(values[s:s + block_len])
        means.append(statistics.mean(pooled[:n]))
    means.sort()
    return ci_from_means(means)


def holm(p_by_label: dict[str, float | None], gate_open: bool, invalid: set[str] | None = None, alpha: float = ALPHA_1S, m: int = len(EXPLORATORY)) -> dict[str, dict[str, Any]]:
    """Holm step-down over the exploratory family (size m). Gate closed -> no rejection. An INVALID or
    p=None variant is never rejected (and, being a failure, ends the step-down). 가족 크기 m 고정."""
    invalid = invalid or set()
    order = sorted(p_by_label, key=lambda k: (p_by_label[k] is None, p_by_label[k] if p_by_label[k] is not None else 1.0))
    out: dict[str, dict[str, Any]] = {}
    going = gate_open
    for rank, label in enumerate(order, start=1):
        p = p_by_label[label]
        thr = alpha / (m - rank + 1)
        rej = bool(going and p is not None and label not in invalid and p * (m - rank + 1) <= alpha + 1e-12)
        if not rej:
            going = False
        out[label] = {"p": p, "rank": rank, "threshold": thr, "rejected": rej}
    return out


def verdict(*, n_eff: float | None, mean_ic: float | None, mean_gross: float | None, ci_net: tuple[float, float] | None,
            mme: float, invalid: bool = False, exploratory: bool = False, gate_open: bool = True, holm_rejected: bool = False) -> dict[str, Any]:
    """Prereg E ladder, steps 0-5, mechanical. Rule ids E0..E5 (exploratory PASS is "E3+Holm")."""
    flags: list[str] = []
    if invalid:
        return {"code": "INVALID", "rule": "E0", "flags": flags}
    if n_eff is None or n_eff < N_EFF_MIN:
        return {"code": "UNDERPOWERED", "rule": "E1", "flags": flags}
    if mean_ic not in (None, 0) and mean_gross not in (None, 0) and (mean_ic > 0) != (mean_gross > 0):  # same 0-handling as sweep :1073-1080
        return {"code": "INCONSISTENT", "rule": "E2", "flags": flags}
    if ci_net is not None and ci_net[0] > 0:
        if not exploratory:
            return {"code": "PASS", "rule": "E3", "flags": flags}
        if gate_open and holm_rejected:
            return {"code": "PASS", "rule": "E3+Holm", "flags": flags}
        flags.append("CI>0, 게이트 닫힘(서술만, 승격 없음)" if not gate_open else "CI>0, Holm 미기각(서술만, 승격 없음)")
        return {"code": "UNDERPOWERED", "rule": "E5", "flags": flags}
    if ci_net is not None and ci_net[1] < mme:
        return {"code": "FAIL", "rule": "E4", "flags": flags}
    return {"code": "UNDERPOWERED", "rule": "E5", "flags": flags}


def h2_verdict(*, n_eff: float | None, ci_gross: tuple[float, float] | None, mean_ic: float | None, mean_gross: float | None) -> dict[str, Any]:
    """H2 on the live-momentum baseline: gross excess, no cost. CI upper < 0 supports, lower > 0 refutes."""
    flags: list[str] = []
    if n_eff is None or n_eff < N_EFF_MIN:
        return {"code": "UNDERPOWERED", "flags": flags}
    code = "INCONCLUSIVE"
    if ci_gross is not None:
        code = "SUPPORTED" if ci_gross[1] < 0 else ("REFUTED" if ci_gross[0] > 0 else "INCONCLUSIVE")
    if mean_ic not in (None, 0) and mean_gross not in (None, 0) and (mean_ic > 0) != (mean_gross > 0):
        flags.append("warning: RankIC and excess sign disagree")
    return {"code": code, "flags": flags}


def assign_quintiles(pairs: list[tuple[str, float, float]]) -> list[list[float]]:
    """pairs = (ticker, score, fwd). Sort by (score, ticker) ascending; position i (0-based) -> q = floor(5i/n)+1.
    Returns 5 lists of forward returns, Q1 (lowest score) first."""
    n = len(pairs)
    ordered = sorted(pairs, key=lambda p: (p[1], p[0]))
    qs: list[list[float]] = [[] for _ in range(N_Q)]
    for i, (_, _, f) in enumerate(ordered):
        qs[(N_Q * i) // n].append(f)
    return qs


def h3_verdict(*, n_eff: float | None, rho: float | None, ci51: tuple[float, float] | None, q4_minus_q2: float | None) -> str:
    if n_eff is None or n_eff < N_EFF_MIN:
        return "UNDERPOWERED"
    tail = ci51 is not None and ci51[0] > 0
    if tail and rho is not None and rho >= H3_RHO_MIN - 1e-12 and q4_minus_q2 is not None and q4_minus_q2 > 0:
        return "H3_SUPPORTED"
    return "TAIL_ONLY" if tail else "NOT_SUPPORTED"


def h3_compute(merged: MergedGrid, label: str, n_boot: int, seed: int, n_eff: float | None) -> dict[str, Any]:
    """Quintile monotonicity from the stored signed scores (Q5 = the side this variant picks). No correlation
    dedup here by design (design D: dedup only shapes the top-4 portfolio)."""
    per_q: list[list[float]] = [[] for _ in range(N_Q)]
    diffs: list[float] = []
    skipped: list[str] = []
    for day in merged.dates:
        o = merged.obs[label].get(day)
        if o is None:
            continue
        fw = merged.forward[day]
        pairs = [(t, s, fw[t][0]) for t, s in o["signed_scores"].items() if s is not None and fw.get(t) and fw[t][0] is not None]
        if len(pairs) < N_Q:
            skipped.append(day)
            continue
        base = statistics.mean(p[2] for p in pairs)
        qs = assign_quintiles(pairs)
        for k in range(N_Q):
            per_q[k].append(statistics.mean(qs[k]) - base)
        diffs.append(statistics.mean(qs[4]) - statistics.mean(qs[0]))
    q_mean = [_mean(x) for x in per_q]
    full = all(x is not None for x in q_mean)
    rho = fm.spearman([1, 2, 3, 4, 5], q_mean, min_pairs=5) if full else None  # type: ignore[arg-type]
    ci51 = ci_from_means(bootstrap_means(diffs, n_boot, seed))
    q42 = (q_mean[3] - q_mean[1]) if full else None  # type: ignore[operator]
    return {"q_mean": q_mean, "rho": rho, "diff51_mean": _mean(diffs), "diff51_ci": ci51, "q4_minus_q2": q42,
            "n_dates": len(diffs), "skipped_dates": skipped, "verdict": h3_verdict(n_eff=n_eff, rho=rho, ci51=ci51, q4_minus_q2=q42),
            "q5_series": per_q[4]}


def trimmed_mean(values: list[float], k: int = TRIM_K) -> float | None:
    """Mean after dropping the k lowest and k highest values; fewer than 2k+1 values -> None."""
    if len(values) < 2 * k + 1 or len(values) < 5:
        return None
    s = sorted(values)
    return statistics.mean(s[k:len(s) - k])


@dataclass
class DateBook:
    """One date's pick and universe forward returns by ticker (non-None only)."""
    day: str
    picks: dict[str, float]
    universe: dict[str, float]


def _book_excess(picks: dict[str, float], universe: dict[str, float]) -> float | None:
    return fm.excess_return(list(picks.values()), list(universe.values()))


def winsorized_excess(books: list[DateBook], pcts: tuple[float, float] = WINSOR) -> list[float]:
    """Per-date cross-sectional winsorisation (nearest-rank cuts of that date's universe) applied to the picks
    and the universe alike, then excess = mean(picks) - mean(universe)."""
    out: list[float] = []
    for b in books:
        s = sorted(b.universe.values())
        if not s or not b.picks:
            continue
        lo, hi = fm.percentile(s, pcts[0]), fm.percentile(s, pcts[1])
        clip = lambda x: min(max(x, lo), hi)  # noqa: E731
        out.append(statistics.mean(clip(x) for x in b.picks.values()) - statistics.mean(clip(x) for x in b.universe.values()))
    return out


def ticker_contributions(books: list[DateBook]) -> dict[str, float]:
    """C_t = sum_d (fwd_t - U_d) / |P_d|, divided by the number of dates, so sum_t C_t = mean excess."""
    c: dict[str, float] = {}
    for b in books:
        if not b.picks or not b.universe:
            continue
        u = statistics.mean(b.universe.values())
        for t, f in b.picks.items():
            c[t] = c.get(t, 0.0) + (f - u) / len(b.picks)
    n = len(books)
    return {t: v / n for t, v in c.items()}


def leave_one_out_excess(books: list[DateBook], drop: str) -> tuple[list[float], list[str]]:
    """Remove `drop` from picks AND universe (no replacement: the cluster map is not in the dump). Dates whose
    picks (or universe) become empty are skipped and reported."""
    out: list[float] = []
    skipped: list[str] = []
    for b in books:
        picks = {t: f for t, f in b.picks.items() if t != drop}
        uni = {t: f for t, f in b.universe.items() if t != drop}
        e = _book_excess(picks, uni)
        if e is None:
            skipped.append(b.day)
        else:
            out.append(e)
    return out, skipped


def period_halves(days: list[str], values: list[float], split: date) -> dict[str, Any]:
    first = [v for d, v in zip(days, values) if date.fromisoformat(d) < split]
    second = [v for d, v in zip(days, values) if date.fromisoformat(d) >= split]
    return {"first": _mean(first), "second": _mean(second), "n_first": len(first), "n_second": len(second)}


def _kr_holiday_lookup(day: date) -> bool | None:
    """True/False from the static KRX table, None when its year is not covered. Lazy import: the module
    drags asyncpg in (no connection is made). (monkeypatched in tests)"""
    try:
        from app.live import kr_holidays as kh

        if day.year not in kh._KR_MARKET_HOLIDAYS:
            return None
        return bool(kh.is_kr_market_holiday(day))
    except Exception:  # noqa: BLE001
        return None


def realised_trading_days(entry: date, exit_: date, lookup: Callable[[date], bool | None] | None = None) -> int | None:
    """Weekdays in (entry, exit] that are not KRX holidays; None ("unknown") if a year is not in the table."""
    lookup = lookup or _kr_holiday_lookup
    n = 0
    d = entry + timedelta(days=1)
    while d <= exit_:
        if d.weekday() < 5:
            h = lookup(d)
            if h is None:
                return None
            n += 0 if h else 1
        d += timedelta(days=1)
    return n


def zero_day_flags(merged: MergedGrid, label: str, lookup: Callable[[date], bool | None] | None = None) -> dict[str, dict[str, Any]]:
    """Per date: zero realised holding days when a majority of tickers with a value have exit_price_date <= entry_date."""
    out: dict[str, dict[str, Any]] = {}
    for day in merged.dates:
        if day not in merged.obs[label]:
            continue
        vals = [x for x in merged.forward[day].values() if x[0] is not None and x[1] and x[2]]
        zero = bool(vals) and sum(1 for x in vals if date.fromisoformat(x[2]) <= date.fromisoformat(x[1])) * 2 > len(vals)
        cal = realised_trading_days(date.fromisoformat(day), date.fromisoformat(day) + timedelta(days=merged.holding), lookup)
        modal = max(set((x[1], x[2]) for x in vals), key=lambda p: sum(1 for x in vals if (x[1], x[2]) == p)) if vals else None
        days_real = realised_trading_days(date.fromisoformat(modal[0]), date.fromisoformat(modal[1]), lookup) if modal else None
        out[day] = {"zero_day": zero, "calendar_days": cal, "realised_days": days_real,
                    "calendar_mismatch": cal is not None and (cal == 0) != zero}
    return out


def build_books(merged: MergedGrid, label: str) -> list[DateBook]:
    v = next(x for x in merged.variants if x["label"] == label)
    books: list[DateBook] = []
    for day in merged.dates:
        o = merged.obs[label].get(day)
        if o is None or o["excess_return"] is None:
            continue
        sc = merged.scores[(v["lookback_days"], day)]
        fw = merged.forward[day]
        books.append(DateBook(day, dict(o["pick_forward_returns"]), {t: x[0] for t, x in fw.items() if t in sc and x[0] is not None}))
    return books


def evidence_ladder(*, primary_code: str, ic_ci: tuple[float, float] | None, h3_code: str, mean_net: float | None,
                    q5_net: float | None, first_half_net: float | None) -> dict[str, Any]:
    """Prereg E evidence ladder. level A = H1 PASS; level B = B1..B6 all true."""
    b = {
        "B1_ic_ci_lower_gt_0": ic_ci is not None and ic_ci[0] > 0,
        "B2_h3_supported": h3_code == "H3_SUPPORTED",
        "B3_net_mean_gt_0": mean_net is not None and mean_net > 0,
        "B4_top_quantile_net_gt_0": q5_net is not None and q5_net > 0,
        "B5_first_half_net_gt_0": first_half_net is not None and first_half_net > 0,
        "B6_not_inconsistent": primary_code != "INCONSISTENT",
    }
    return {"level_A": primary_code == "PASS", "level_B": all(b.values()), "B": b}


def stop_rule(*, role: str, primary_code: str, level_b: bool, mean_ic: float | None) -> dict[str, str]:
    if primary_code == "INVALID":
        return {"code": "NOT_EVALUATED", "text": "primary run invalid: ladder not evaluated"}
    if role == "holdout":
        if primary_code in ("FAIL", "INCONSISTENT") or (mean_ic is not None and mean_ic <= 0):
            return {"code": "STOP-3", "text": "STOP-3: terminate; no changes, no re-run of the holdout"}
        return {"code": "NONE", "text": "STOP-3 not triggered"}
    if primary_code in ("FAIL", "INCONSISTENT"):
        return {"code": "STOP-1", "text": "STOP-1: terminate, do not open the validation period"}
    if primary_code == "UNDERPOWERED" and not level_b:
        return {"code": "STOP-2", "text": "STOP-2: terminate, evidence ladder level B not reached"}
    return {"code": "OPEN_OK", "text": "validation-opening condition met (final decision: user)"}


# --- report assembly ----------------------------------------------------------

def variant_stats(merged: MergedGrid, label: str, n_boot: int, seed: int) -> dict[str, Any]:
    """Everything the tables need for one variant (all costs, CIs, p-values, IC, spread, hit rates)."""
    days = [d for d in merged.dates if d in merged.obs[label]]
    obs = [merged.obs[label][d] for d in days]
    ex_days = [d for d, o in zip(days, obs) if o["excess_return"] is not None]
    ex = [o["excess_return"] for o in obs if o["excess_return"] is not None]
    n_obs = len(obs)
    out: dict[str, Any] = {
        "label": label, "n_obs": n_obs, "n_excess": len(ex), "n_eff": fm.effective_sample_size(n_obs, merged.cadence, merged.holding),
        "days": days, "ex_days": ex_days, "excess": ex,
    }
    g_means = bootstrap_means(ex, n_boot, seed)
    g_ci = ci_from_means(g_means)
    out["gross"] = {"mean": _mean(ex), "ci": g_ci}
    out["net"] = {}
    shift = 0.0
    for c in (COST_RT, *COST_STRESS):
        net = [e - c for e in ex]
        means = bootstrap_means(net, n_boot, seed)
        ci = ci_from_means(means)
        mean = _mean(net)
        entry = {"mean": mean, "ci": ci, "hit": (sum(1 for x in net if x > 0) / len(net)) if net else None}
        if c == COST_RT:
            entry["p_boot"] = p_boot_one_sided(means)
            entry["p_norm"] = p_normal_one_sided(mean, ci)
            entry["boundary_disagree"] = boundary_disagree(ci, entry["p_boot"])
        if ci is not None and g_ci is not None:
            shift = max(shift, abs(ci[0] - (g_ci[0] - c)), abs(ci[1] - (g_ci[1] - c)))
        out["net"][c] = entry
    out["ci_shift_max_abs_diff"] = shift
    ics = [o["rank_ic"] for o in obs]
    ic_ok = [x for x in ics if x is not None]
    out["rank_ic"] = {"mean": _mean(ic_ok), "ci": ci_from_means(bootstrap_means(ic_ok, n_boot, seed)), "icir": fm.icir(ics), "n": len(ic_ok)}
    sp = [o["quantile_spread"] for o in obs if o["quantile_spread"] is not None]
    out["spread"] = {"mean": _mean(sp)}
    out["hit_gross"] = (sum(1 for x in ex if x > 0) / len(ex)) if ex else None
    pooled = [x for o in obs for x in o["pick_forward_returns"].values()]
    out["worth"] = fm.worth_analyzing_rate(pooled, bar=merged.worth_bar)
    out["upside"] = fm.upside_hit_rate(pooled, bar=merged.worth_bar)
    tm = merged_turnover(merged, label)
    out["turnover_merged"] = _mean([v for v in tm.values() if v is not None])
    out["turnover_by_date"] = tm
    v_def = next(v for v in merged.variants if v["label"] == label)
    fb_dates = sum(1 for o in obs if o["residual_fallbacks"] > 0)
    out["resid_fb_dates"] = fb_dates
    out["resid_invalid"] = v_def["momentum_mode"] == "residual" and n_obs > 0 and fb_dates / n_obs > RESID_FB_MAX
    return out


def robustness(merged: MergedGrid, stats: dict[str, Any], h3: dict[str, Any], n_boot: int, seed: int, role: str,
               lookup: Callable[[date], bool | None] | None = None) -> dict[str, Any]:
    """PRIMARY robustness: every item is the point estimate of the net (0.30%) mean being > 0."""
    ex, days, c = stats["excess"], stats["ex_days"], COST_RT
    books = build_books(merged, PRIMARY)
    r: dict[str, Any] = {}
    tm = trimmed_mean(ex)
    r["trimmed"] = {"net_mean": None if tm is None else tm - c}
    wins = winsorized_excess(books)
    r["winsorized"] = {"net_mean": None if not wins else _mean(wins) - c}
    contrib = ticker_contributions(books)
    top = sorted(contrib.items(), key=lambda kv: kv[1], reverse=True)
    mean_gross = stats["gross"]["mean"]
    share3 = (sum(v for _, v in top[:3]) / mean_gross) if top and mean_gross else None
    loo_vals, loo_skipped = leave_one_out_excess(books, top[0][0]) if top else ([], [])
    r["concentration"] = {"top_ticker": top[0][0] if top else None, "share3": share3, "warn": share3 is not None and share3 > CONC_WARN,
                          "top10": top[:10], "loo_net_mean": None if not loo_vals else _mean(loo_vals) - c, "loo_gross_mean": _mean(loo_vals),
                          "loo_skipped_dates": loo_skipped}
    split = HALF_SPLIT.get(role)
    halves = period_halves(days, [e - c for e in ex], split) if split else None
    r["halves"] = halves
    r["cost50"] = {"net_mean": stats["net"][COST_STRESS[0]]["mean"], "net80_mean": stats["net"][COST_STRESS[1]]["mean"]}
    zd = zero_day_flags(merged, PRIMARY, lookup)
    keep = [e for d, e in zip(days, ex) if not zd.get(d, {}).get("zero_day")]
    r["zero_day"] = {"n_zero_days": sum(1 for d in days if zd.get(d, {}).get("zero_day")), "net_mean_excl": None if not keep else _mean(keep) - c,
                     "calendar_mismatch": [d for d in days if zd.get(d, {}).get("calendar_mismatch")], "by_date": zd}
    mbb = moving_block_ci([e - c for e in ex], n_boot, seed)
    r["moving_block_ci"] = {"ci": mbb, "lower_gt_0": None if mbb is None else mbb[0] > 0, "sensitivity_only": True}
    r["h3_q5_net"] = None if not h3["q5_series"] else _mean(h3["q5_series"]) - c
    return r


def build_report(merged: MergedGrid, checks: list[Check], chunks: dict[str, ChunkInfo], plan: dict[str, Any], base: Path,
                 n_boot: int = BOOT_N, seed: int = SEED, lookup: Callable[[date], bool | None] | None = None) -> dict[str, Any]:
    """All statistics (report mode). Pure over the merged dumps."""
    role = merged.role
    mme = MME[merged.holding]
    aux = merged.holding != 4
    labels = [v["label"] for v in merged.variants]
    stats = {lab: variant_stats(merged, lab, n_boot, seed) for lab in labels}
    acceptance_ok = all(ch.ok for ch in checks if ch.severity in ("integrity", "acceptance"))
    shift = max((s["ci_shift_max_abs_diff"] for s in stats.values()), default=0.0)
    s1 = Check("S1", "T2 C", shift <= TOL, "integrity", f"net CI == gross CI - cost (max abs diff {shift:.1e})")
    checks.append(s1)

    def vd(lab: str, exploratory: bool, gate_open: bool = True, holm_rej: bool = False) -> dict[str, Any]:
        s = stats[lab]
        out = verdict(n_eff=s["n_eff"], mean_ic=s["rank_ic"]["mean"], mean_gross=s["gross"]["mean"], ci_net=s["net"][COST_RT]["ci"], mme=mme,
                      invalid=(not acceptance_ok) or s["resid_invalid"], exploratory=exploratory, gate_open=gate_open, holm_rejected=holm_rej)
        note = boundary_flag(s["net"][COST_RT])
        if note and out["code"] != "INVALID" and s["n_eff"] is not None and s["n_eff"] >= N_EFF_MIN:  # n_eff<20 wording stays unchanged
            out["flags"].append(note)
        return out

    verdicts: dict[str, dict[str, Any]] = {}
    holm_out: dict[str, dict[str, Any]] = {}
    holm_norm: dict[str, dict[str, Any]] = {}
    if PRIMARY in stats:
        verdicts[PRIMARY] = vd(PRIMARY, False)
    gate = verdicts.get(PRIMARY, {}).get("code") == "PASS"
    expl = [e for e in EXPLORATORY if e in stats]
    if expl:
        invalid = {e for e in expl if (not acceptance_ok) or stats[e]["resid_invalid"]}
        holm_out = holm({e: stats[e]["net"][COST_RT]["p_boot"] for e in expl}, gate, invalid)
        holm_norm = holm({e: stats[e]["net"][COST_RT]["p_norm"] for e in expl}, gate, invalid)
        for e in expl:
            verdicts[e] = vd(e, True, gate, holm_out[e]["rejected"])
            verdicts[e]["holm_norm_differs"] = holm_norm[e]["rejected"] != holm_out[e]["rejected"]
    h2: dict[str, Any] | None = None
    if BASELINE in stats:
        s = stats[BASELINE]
        h2 = h2_verdict(n_eff=s["n_eff"], ci_gross=s["gross"]["ci"], mean_ic=s["rank_ic"]["mean"], mean_gross=s["gross"]["mean"])
        if not acceptance_ok:
            h2 = {"code": "INVALID", "flags": []}
    h3 = {lab: h3_compute(merged, lab, n_boot, seed, stats[lab]["n_eff"]) for lab in labels}
    rob: dict[str, Any] | None = None
    ladder: dict[str, Any] | None = None
    stop: dict[str, str] | None = None
    if PRIMARY in stats and role == "dev":
        rob = robustness(merged, stats[PRIMARY], h3[PRIMARY], n_boot, seed, role, lookup)
        code = verdicts[PRIMARY]["code"]
        rp: Any = "n/a"
        if code == "PASS":
            h = rob["halves"] or {}
            parts = [rob["trimmed"]["net_mean"], rob["winsorized"]["net_mean"], rob["concentration"]["loo_net_mean"], h.get("first"), rob["cost50"]["net_mean"], rob["zero_day"]["net_mean_excl"]]
            rp = all(p is not None and p > 0 for p in parts)
        rob["robust_pass"] = rp
        if not aux:
            ladder = evidence_ladder(primary_code=code, ic_ci=stats[PRIMARY]["rank_ic"]["ci"], h3_code=h3[PRIMARY]["verdict"],
                                     mean_net=stats[PRIMARY]["net"][COST_RT]["mean"], q5_net=rob["h3_q5_net"], first_half_net=(rob["halves"] or {}).get("first"))
            stop = stop_rule(role=role, primary_code=code, level_b=ladder["level_B"], mean_ic=stats[PRIMARY]["rank_ic"]["mean"])
    elif PRIMARY in stats and role == "holdout":
        stop = stop_rule(role=role, primary_code=verdicts[PRIMARY]["code"], level_b=False, mean_ic=stats[PRIMARY]["rank_ic"]["mean"])
    per_date: list[dict[str, Any]] = []
    if PRIMARY in stats:
        zd = zero_day_flags(merged, PRIMARY, lookup)
        tm = stats[PRIMARY]["turnover_by_date"]
        for day in stats[PRIMARY]["days"]:
            o = merged.obs[PRIMARY][day]
            e = o["excess_return"]
            per_date.append({"date": day, "chunk": merged.chunk_of[day], "excess": e, "net30": None if e is None else e - COST_RT, "rank_ic": o["rank_ic"],
                             "spread": o["quantile_spread"], "turnover_merged": tm.get(day), "picks": o["picks"],
                             "realised_days": zd[day]["realised_days"], "zero_day": zd[day]["zero_day"]})
    return {
        "stats": stats, "verdicts": verdicts, "holm": holm_out, "holm_norm": holm_norm, "h2": h2, "h3": h3, "robustness": rob,
        "ladder": ladder, "stop": stop, "per_date": per_date, "aux_horizon": aux, "mme": mme, "gate_open": gate, "n_boot": n_boot,
    }


# --- rendering ------------------------------------------------------------------

FOOTER = "excess_return (vs the U86 cross-section), NOT alpha_return (vs the index) - RULES section 4"


def _pct(v: float | None, places: int = 2) -> str:
    return "-" if v is None else f"{v * 100:+.{places}f}%"


def _num(v: float | None, places: int = 3) -> str:
    return "-" if v is None else f"{v:+.{places}f}"


def _ci(ci: Any, pct: bool = True) -> str:
    if ci is None:
        return "-"
    return f"[{_pct(ci[0], 4)},{_pct(ci[1], 4)}]" if pct else f"[{_num(ci[0])},{_num(ci[1])}]"


def _holm_note(h: dict[str, Any], code: str) -> str:
    """Clarify a p-only Holm rejection whose verdict is not PASS (INCONSISTENT, UNDERPOWERED, ...)."""
    return f" (p-only rejection; verdict {code})" if h["rejected"] and code != "PASS" else ""


def _uw(n: int) -> str:
    return f"UNDERPOWERED, {n} dates"


def _check_lines(checks: list[Check]) -> list[str]:
    return [f"  {c.id:<4} [{'ok ' if c.ok else 'BAD'}] ({c.severity}) {c.detail}  <{c.prereg_ref}>" for c in checks]


def _t5(chunks: dict[str, ChunkInfo], merged: MergedGrid | None) -> list[str]:
    head = "T5 chunks: chunk | dates | fb atr pu cl bf rt eg (txt=runlog) | min scored/priced | dedup_deg | resid_fb dates | discarded | sha256[:16]"
    rows = [head]
    for name, i in chunks.items():
        cnt = i.scan["counts"] if i.scan else {}
        eq = "=" if i.rec and i.scan and all(cnt.get(k) == i.rec.get(k) for k, _ in LOG_NEEDLES) else "!="
        mins = mind = None
        deg = 0
        rfb: list[str] = []
        d = i.dump
        lbs = sorted({v["variant"]["lookback_days"] for v in d["variants"]})
        for day in i.dates:
            a = len(d["scores"][f"{lbs[0]}|{day}"])
            b = sum(1 for x in d["forward"][day].values() if x["fwd_return"] is not None)
            mins = a if mins is None else min(mins, a)
            mind = b if mind is None else min(mind, b)
        for v in d["variants"]:
            deg += sum(1 for o in v["observations"].values() if o["dedup_degraded"])
            if v["variant"]["momentum_mode"] == "residual":
                rfb.append(f"{v['variant']['label']}:{sum(1 for o in v['observations'].values() if o['residual_fallbacks'] > 0)}")
        nums = " ".join(str(cnt.get(k, "?")) for k, _ in LOG_NEEDLES)
        rows.append(f"  {name} | {len(i.dates)} | {nums} ({eq}) | {mins}/{mind} | {deg} | {','.join(rfb) or '-'} | {i.discarded_attempts} | {i.sha[:16]}")
    return rows


def render_check_text(grid: str, role: str, mode: str, status: str, checks: list[Check], chunks: dict[str, ChunkInfo], merged: MergedGrid | None, exit_code: int) -> str:
    n_dates = len(merged.dates) if merged else 0
    lines = [
        f"screening-reversal T2  grid={grid} role={role} mode={mode} status={status} exit={exit_code}",
        f"inputs: {len(chunks)} chunk dumps, {n_dates} dates" + (f", holding={merged.holding}d cadence={merged.cadence}d" if merged else ""),
        "checks: " + f"{sum(1 for c in checks if c.ok)} ok, {sum(1 for c in checks if not c.ok)} bad (metrics are not printed in check mode)",
        *_check_lines(checks), *_t5(chunks, merged),
    ]
    return "\n".join(lines)


def render_text(grid: str, mode: str, status: str, checks: list[Check], chunks: dict[str, ChunkInfo], merged: MergedGrid, rep: dict[str, Any], exit_code: int) -> str:
    st, vs = rep["stats"], rep["verdicts"]
    lines = [
        f"screening-reversal T2  grid={grid} role={merged.role} mode={mode} status={status} exit={exit_code}",
        f"inputs: {len(chunks)} chunk dumps, {len(merged.dates)} dates, holding={merged.holding}d cadence={merged.cadence}d universe={len(merged.universe)}"
        + ("  [AUXILIARY horizon: no ladder, no STOP]" if rep["aux_horizon"] else ""),
        f"bootstrap={rep['n_boot']} seed={SEED}  minimum meaningful effect={_pct(rep['mme'])}  cost round-trip={_pct(COST_RT)}",
        "", "== Checks ==", *_check_lines(checks), "", "== Verdict block (rule ids refer to prereg E) ==",
    ]

    def vline(lab: str) -> str:
        s, v = st[lab], vs[lab]
        if v["code"] != "INVALID" and (s["n_eff"] is None or s["n_eff"] < N_EFF_MIN):
            return f"  {lab}: {_uw(s['n_obs'])}"
        txt = VERDICT_TEXT[v["code"]]
        extra = f" flags={v['flags']}" if v["flags"] else ""
        return f"  {lab}: {v['code']} (rule {v['rule']}, {txt[2]}) - {txt[0]} / {txt[1]}{extra}"

    if PRIMARY in st:
        lines.append("H1 primary:")
        lines.append(vline(PRIMARY))
    exp = [e for e in EXPLORATORY if e in st]
    if exp:
        lines.append(f"Holm (exploratory, gate {'open' if rep['gate_open'] else 'closed'}, alpha={ALPHA_1S}, m={len(EXPLORATORY)}):")
        for e in exp:
            h = rep["holm"][e]
            hn = rep["holm_norm"][e]
            n_note = " (p_norm Holm differs)" if h["rejected"] != hn["rejected"] else ""
            lines.append(vline(e) + f"  p_boot={_num(h['p'], 6)} rank={h['rank']} thr={h['threshold']:.6f} rejected={h['rejected']}{_holm_note(h, vs[e]['code'])}{n_note}")
    if rep["h2"] is not None:
        s = st[BASELINE]
        if s["n_eff"] is None or s["n_eff"] < N_EFF_MIN:
            lines.append(f"H2 {BASELINE}: {_uw(s['n_obs'])}")
        else:
            code = rep["h2"]["code"]
            tx = VERDICT_TEXT.get(code, (code, code, "prereg E (H2)"))
            lines.append(f"H2 {BASELINE}: {code} ({tx[2]}) - {tx[0]} / {tx[1]}" + (f" {rep['h2']['flags']}" if rep["h2"]["flags"] else ""))
    if PRIMARY in rep["h3"]:
        s = st[PRIMARY]
        if s["n_eff"] is None or s["n_eff"] < N_EFF_MIN:
            lines.append(f"H3 {PRIMARY}: {_uw(s['n_obs'])}")
        else:
            code = rep["h3"][PRIMARY]["verdict"]
            tx = VERDICT_TEXT.get(code, (code, code, "prereg E (H3)"))
            lines.append(f"H3 {PRIMARY}: {code} ({tx[2]}) - {tx[0]} / {tx[1]}")
    rob = rep["robustness"]
    s = st.get(PRIMARY)
    under = s is not None and (s["n_eff"] is None or s["n_eff"] < N_EFF_MIN)
    if rob is not None:
        lines.append(f"Robustness {PRIMARY}: " + (_uw(s["n_obs"]) if under else f"robust_pass={rob['robust_pass']}"))
    if rep["ladder"] is not None:
        lines.append("Evidence ladder: " + (_uw(s["n_obs"]) if under else f"level A={rep['ladder']['level_A']} level B={rep['ladder']['level_B']} {rep['ladder']['B']}"))
    if rep["stop"] is not None:
        lines.append("STOP rule: " + (_uw(s["n_obs"]) if under else rep["stop"]["text"]))
    lines += ["", "T1 metrics (gross / net 0.30% / p-values / RankIC / spread):"]
    lines.append("  variant | n n_eff | gross mean [CI95] | net30 mean [CI95] | p_boot p_norm | net50 net80 | IC mean [CI95] ICIR | spread | hit gross/net | worth up | turnover(merged)")
    for lab, x in st.items():
        n30 = x["net"][COST_RT]
        lines.append(
            f"  {lab} | {x['n_obs']} {('%.1f' % x['n_eff']) if x['n_eff'] is not None else '-'} | {_pct(x['gross']['mean'])} {_ci(x['gross']['ci'])} | {_pct(n30['mean'])} {_ci(n30['ci'])} | "
            f"{_num(n30.get('p_boot'), 6)} {_num(n30.get('p_norm'), 6)} | {_pct(x['net'][COST_STRESS[0]]['mean'])} {_pct(x['net'][COST_STRESS[1]]['mean'])} | "
            f"{_num(x['rank_ic']['mean'])} {_ci(x['rank_ic']['ci'], False)} {_num(x['rank_ic']['icir'], 2)} | {_pct(x['spread']['mean'])} | "
            f"{_pct(x['hit_gross'], 0)}/{_pct(n30['hit'], 0)} | {_pct(x['worth'], 0)} {_pct(x['upside'], 0)} | {_num(x['turnover_merged'], 2)}"
            + (f" | {_uw(x['n_obs'])}" if x["n_eff"] is None or x["n_eff"] < N_EFF_MIN else ""))
    lines += ["", "T3 H3 quintiles (mean fwd minus date mean, Q1..Q5; rho; Q5-Q1 [CI]; Q4-Q2):"]
    for lab, h in rep["h3"].items():
        lines.append(f"  {lab} | q={[_pct(q, 3) for q in h['q_mean']]} rho={_num(h['rho'], 2)} q5-q1={_pct(h['diff51_mean'])} {_ci(h['diff51_ci'])} q4-q2={_pct(h['q4_minus_q2'])} dates={h['n_dates']} skipped={len(h['skipped_dates'])}")
    if rob is not None:
        c = rob["concentration"]
        hv = rob["halves"] or {}
        lines += ["", "T4 robustness (PRIMARY, net 0.30% unless noted):",
                  f"  trimmed(2/side)={_pct(rob['trimmed']['net_mean'])} winsorized={_pct(rob['winsorized']['net_mean'])} leave-top-ticker-out({c['top_ticker']})={_pct(c['loo_net_mean'])} "
                  f"first-half={_pct(hv.get('first'))} (n={hv.get('n_first')}) second-half={_pct(hv.get('second'))} (n={hv.get('n_second')}) cost50={_pct(rob['cost50']['net_mean'])} cost80={_pct(rob['cost50']['net80_mean'])}",
                  f"  zero-day dates={rob['zero_day']['n_zero_days']} net-excl={_pct(rob['zero_day']['net_mean_excl'])} calendar-mismatch={rob['zero_day']['calendar_mismatch']}",
                  f"  moving-block(3) CI={_ci(rob['moving_block_ci']['ci'])} (sensitivity only) top-3 contribution share={_num(c['share3'], 2)}" + (" WARN >0.5" if c["warn"] else ""),
                  "  top-10 ticker contributions: " + ", ".join(f"{t}={_pct(v, 3)}" for t, v in c["top10"])]
    lines += ["", "T6 PRIMARY per date: date | chunk | excess | net30 | IC | spread | turnover(merged) | picks | realised days | zero-day"]
    for r in rep["per_date"]:
        lines.append(f"  {r['date']} | {r['chunk']} | {_pct(r['excess'])} | {_pct(r['net30'])} | {_num(r['rank_ic'])} | {_pct(r['spread'])} | {_num(r['turnover_merged'], 2)} | {','.join(r['picks'])} | {r['realised_days'] if r['realised_days'] is not None else 'unknown'} | {'ZERO' if r['zero_day'] else '-'}")
    lines += ["", *_t5(chunks, merged), "", FOOTER]
    return "\n".join(lines)


def _t2_head() -> str | None:
    """HEAD of the repo holding this module, suffixed '+dirty' when the T2 files themselves have uncommitted changes."""
    head = (_git("rev-parse", "HEAD") or "").strip() or None
    status = _git("status", "--porcelain", "--", *T2_PATHS)
    return f"{head}+dirty" if head and status and status.strip() else head


def _report_json(grid: str, role: str, mode: str, status: str, exit_code: int, checks: list[Check], chunks: dict[str, ChunkInfo], plan: dict[str, Any],
                 merged: MergedGrid | None, rep: dict[str, Any] | None, base: Path) -> dict[str, Any]:
    extra = getattr(merged, "extra", None) if merged else None
    out: dict[str, Any] = {
        "schema": REPORT_SCHEMA, "status": status, "mode": mode, "grid": grid, "role": role, "exit_code": exit_code,
        "generated_at": datetime.now(timezone.utc).isoformat(), "git_head_of_T2": _t2_head(),
        "prereg": {"doc": "screening-reversal-stage1-prereg-2026-09-30.md", "constants": {
            "PRIMARY": PRIMARY, "BASELINE": BASELINE, "EXPLORATORY": list(EXPLORATORY), "DIAGNOSTIC": DIAGNOSTIC, "COST_RT": COST_RT,
            "COST_STRESS": list(COST_STRESS), "MME": MME, "ALPHA_1S": ALPHA_1S, "N_EFF_MIN": N_EFF_MIN, "BOOT_N": BOOT_N, "SEED": SEED,
            "HALF_SPLIT": {k: v.isoformat() for k, v in HALF_SPLIT.items()}, "MIN_DATES": MIN_DATES, "FALLBACK_CHUNK_MAX": FALLBACK_CHUNK_MAX,
            "FALLBACK_GRID_MAX": FALLBACK_GRID_MAX, "RESID_FB_MAX": RESID_FB_MAX, "TRIM_K": TRIM_K, "WINSOR": list(WINSOR), "CONC_WARN": CONC_WARN,
            "MBB_LEN": MBB_LEN, "H3_RHO_MIN": H3_RHO_MIN, "N_Q": N_Q}},
        "inputs": {"chunks_json_sha256": sha256_file(base / "chunks.json") if (base / "chunks.json").is_file() else None,
                   "chunks": [{"chunk": i.name, "path": str(i.path), "sha256": i.sha, "dates": i.dates, "runlog": i.rec,
                               "txt_counts": (i.scan or {}).get("counts"), "fallback_attribution": (i.scan or {}).get("fallbacks"),
                               "discarded_attempts": i.discarded_attempts} for i in chunks.values()]},
        "checks": [c.to_dict() for c in checks],
    }
    if extra:
        out.update({"self_consistency": extra["self_consistency"], "chunk_summary_recheck": extra["chunk_summary_recheck"], "instrument": extra["instrument"]})
    if rep is not None:
        out["variants"] = {}
        for lab, s in rep["stats"].items():
            v = rep["verdicts"].get(lab)
            out["variants"][lab] = {
                "n_obs": s["n_obs"], "n_eff": s["n_eff"], "gross": s["gross"],
                "net": {str(c): x for c, x in s["net"].items()}, "rank_ic": s["rank_ic"], "spread": s["spread"],
                "hit": {"gross": s["hit_gross"]}, "worth": s["worth"], "upside": s["upside"], "turnover_merged": s["turnover_merged"],
                "verdict": v, "holm": rep["holm"].get(lab), "resid_invalid": s["resid_invalid"]}
        out.update({"h2": rep["h2"], "h3": {k: {x: y for x, y in h.items() if x != "q5_series"} for k, h in rep["h3"].items()},
                    "robustness": rep["robustness"], "ladder": rep["ladder"], "stop": rep["stop"], "per_date": rep["per_date"]})
    return _jsonable(out)


# --- orchestration ----------------------------------------------------------------

def write_outputs(out_dir: Path, grid: str, mode: str, text: str, report: dict[str, Any]) -> None:
    _write_atomic(out_dir / f"{grid}_{mode}.txt", text + "\n")
    _write_atomic(out_dir / f"{grid}_{mode}.json", json.dumps(_jsonable(report), ensure_ascii=False, allow_nan=False, indent=1))


def _rec_sha(rec: dict[str, Any] | None) -> str | None:
    if not rec:
        return None
    return hashlib.sha256(json.dumps(rec, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def manifest_entries(chunks: dict[str, ChunkInfo]) -> list[dict[str, Any]]:
    return [{"chunk": i.name, "path": str(i.path), "sha256": i.sha, "txt_sha256": i.txt_sha, "runlog_sha256": _rec_sha(i.rec),
             "runlog_end_kst": (i.rec or {}).get("end_kst"), "git_head": i.dump.get("git_head")} for i in chunks.values()]


def _manifest_key(entries: list[dict[str, Any]]) -> dict[str, tuple[Any, Any, Any]]:
    return {e["chunk"]: (e.get("sha256"), e.get("txt_sha256"), e.get("runlog_sha256")) for e in entries}


def write_manifest(out_dir: Path, grid: str, chunks: dict[str, ChunkInfo]) -> Path:
    path = out_dir / f"accepted_{grid}.json"
    payload = {"schema": ACCEPTED_SCHEMA, "grid": grid, "frozen_at": datetime.now(timezone.utc).isoformat(), "chunks": manifest_entries(chunks)}
    _write_atomic(path, json.dumps(_jsonable(payload), ensure_ascii=False, allow_nan=False, indent=1))
    return path


def _load_manifest(path: Path) -> dict[str, Any]:
    m = _read_json(path)
    if not isinstance(m, dict) or not isinstance(m.get("chunks"), list) or any(not isinstance(c, dict) or "chunk" not in c or "sha256" not in c for c in m["chunks"]):
        raise InputError(f"malformed accepted list {path}")
    return m


def run(args: argparse.Namespace, n_bootstrap: int = BOOT_N, lookup: Callable[[date], bool | None] | None = None) -> tuple[int, str]:
    """Execute one mode; returns (exit code, text). Raises InputError for exit 2."""
    grid, mode = args.grid, args.mode
    role = role_of(grid)
    if args.allow_incomplete_set and mode != "check":
        raise InputError("--allow-incomplete-set is only valid with --mode check")
    base, out_dir = Path(args.base_dir), Path(args.out_dir)
    plan = load_plan(base, grid)
    opts = {"allow_incomplete_set": args.allow_incomplete_set, "allow_heads": [x for x in (args.allow_heads or "").split(",") if x],
            "allow_dirty": args.allow_dirty, "accept_market_hours_override": args.accept_market_hours_override}
    manifest_path = out_dir / f"accepted_{grid}.json"
    manifest: dict[str, Any] | None = None
    if mode == "report":
        if not manifest_path.is_file():
            raise InputError(f"report mode needs {manifest_path} (run --mode freeze first)")
        manifest = _load_manifest(manifest_path)
    old_manifest = _load_manifest(manifest_path) if mode == "freeze" and manifest_path.is_file() else None
    checks, chunks, merged, _ = validate(base, grid, role, plan, mode, opts)
    cur = _manifest_key(manifest_entries(chunks))
    unchanged = False
    if manifest is not None:
        acc = _manifest_key(manifest["chunks"])
        _chk(checks, "F1", "T2 A", acc == cur, "accepted list == current dumps (chunk set, dump/txt sha256, runlog record)" if acc == cur else f"accepted list differs: {sorted(set(acc) ^ set(cur)) or [n for n in acc if acc[n] != cur.get(n)]}")
    if old_manifest is not None:
        unchanged = _manifest_key(old_manifest["chunks"]) == cur
        if not unchanged:
            _chk(checks, "F0", "T2 A", False, f"{manifest_path.name} exists and differs from the current set; remove it deliberately to re-freeze", "acceptance")
    bad_integrity = any(not c.ok and c.severity == "integrity" for c in checks)
    bad_acceptance = any(not c.ok and c.severity == "acceptance" for c in checks)
    exit_code = 4 if bad_integrity else (3 if bad_acceptance else 0)
    status = "refused" if exit_code else ("checked" if mode == "check" else "ok")
    rep: dict[str, Any] | None = None
    if exit_code == 0 and mode == "check":
        text = render_check_text(grid, role, mode, status, checks, chunks, merged, exit_code)
    elif exit_code == 0 and mode == "freeze":
        status = "frozen"
        if not unchanged:
            write_manifest(out_dir, grid, chunks)
        text = render_check_text(grid, role, mode, status, checks, chunks, merged, exit_code) + (f"\naccepted list unchanged: {manifest_path}" if unchanged else f"\naccepted list written: {manifest_path}")
    elif exit_code == 0 and merged is not None:
        rep = build_report(merged, checks, chunks, plan, base, n_boot=n_bootstrap, lookup=lookup)
        if any(not c.ok and c.severity in ("integrity", "acceptance") for c in checks):  # S1 failed after the statistics were computed
            exit_code, status, rep = 4, "refused", None
            text = render_check_text(grid, role, mode, status, checks, chunks, merged, exit_code)
        else:
            text = render_text(grid, mode, status, checks, chunks, merged, rep, exit_code)
    else:
        text = render_check_text(grid, role, mode, status, checks, chunks, merged, exit_code)
    write_outputs(out_dir, grid, mode, text, _report_json(grid, role, mode, status, exit_code, checks, chunks, plan, merged, rep, base))
    return exit_code, text


def main(argv: list[str] | None = None, *, n_bootstrap: int = BOOT_N) -> int:
    if hasattr(sys.stdout, "reconfigure") and (getattr(sys.stdout, "encoding", "") or "").lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    parser = argparse.ArgumentParser(description="Merge screening-reversal chunk dumps and compute the pre-registered statistics (offline).")
    parser.add_argument("--grid", required=True)
    parser.add_argument("--mode", required=True, choices=("check", "freeze", "report"))
    parser.add_argument("--base-dir", default="results/screening_reversal")
    parser.add_argument("--out-dir", default="results/screening_reversal/report")
    parser.add_argument("--allow-incomplete-set", action="store_true", help="check mode only: accept a partial chunk set")
    parser.add_argument("--allow-heads", default="", help="comma-separated git SHA prefixes accepted when chunk dumps differ")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--accept-market-hours-override", action="store_true")
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return int(exc.code or 0)
    try:
        code, text = run(args, n_bootstrap=n_bootstrap)
    except InputError as exc:
        print(f"input error: {exc}", file=sys.stderr)
        return 2
    print(text)
    return code


if __name__ == "__main__":
    sys.exit(main())
