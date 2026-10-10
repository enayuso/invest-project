"""CLI of the offline downstream-replay harness (stage 2 of the fast improvement loop).

    python -m app.experiments.replay_downstream extract   --name aa_freeze_0929 --era freeze --min-trade-date 2026-09-17 --max-trade-date 2026-09-29
    python -m app.experiments.replay_downstream extract   --name pilot12 --from-caseset dev_0916 --sample 12 --seed 20261001 --min-stored-rm-buy 4
    python -m app.experiments.replay_downstream preflight --pinned P --caseset aa_freeze_0929 --k 4
    python -m app.experiments.replay_downstream pilot     --pinned P --caseset pilot12 --ks 2,4,6
    python -m app.experiments.replay_downstream run       --pinned P --caseset aa_freeze_0929 --arm baseline --repeat 1 --k 4
    python -m app.experiments.replay_downstream aa-report --r1 DIR --r2 DIR
    python -m app.experiments.replay_downstream status    --run DIR

Exit codes: 0 done and every case has an ok result, 2 config refused (pinned settings, market hours,
dirty/drifted sources, invalid A/A pair), 3 safety violation (run aborted), 75 paused (window closed /
network; same command resumes), 76 loop finished but some cases are still in error (rerun the same
command to retry the retryable ones).

IMPORT ORDER IS THE POINT (spec A.2): this module's top level imports only the stdlib and
`replay.pinned`. `app.config.Settings` reads env at import time, so the pinned values are
applied to `os.environ` first and the heavy harness modules are imported afterwards.
이 모듈 최상위는 stdlib + replay.pinned만 import — 고정 설정을 환경변수에 넣은 뒤에 app.*를 지연 import한다.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import re
import subprocess
import sys
import threading
import time
import urllib.request
from datetime import date, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from app.experiments.replay import HARNESS_VERSION, pinned

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "results" / "replay_downstream"
CASESETS, RUNS, REPORTS = RESULTS / "casesets", RESULTS / "runs", RESULTS / "reports"
DEFAULT_PINNED = RESULTS / "pinned" / "server-2026-09-30.json"
EXIT_OK, EXIT_CONFIG, EXIT_SAFETY, EXIT_PAUSED, EXIT_INCOMPLETE = 0, 2, 3, 75, 76


class CliRefusal(Exception):
    """A refusal that maps to exit code 2 (message printed to stderr)."""


def _stamp() -> str:
    return datetime.now().strftime("%Y%m%dT%H%M%S")  # CLI-only wall label for file names (no colons)


def _utf8() -> None:
    for stream in (sys.stdout, sys.stderr):
        if getattr(stream, "encoding", "utf-8") and stream.encoding.lower() != "utf-8":
            stream.reconfigure(encoding="utf-8")


# --------------------------------------------------------------------------- bootstrap


class Boot:
    """Result of the ordered bootstrap: pinned values applied, settings verified, heavy modules loaded."""

    def __init__(self, p: pinned.Pinned, settings: Any, drift: list[str]):
        self.pinned, self.settings, self.code_default_drift = p, settings, drift


def bootstrap(args: argparse.Namespace, k: int) -> Boot:
    p = pinned.load(getattr(args, "pinned", None) or DEFAULT_PINNED)  # 1. stdlib only
    pinned.apply_to_environ(p, k=k)                                    # 2. env BEFORE app.config
    from app.config import Settings, settings                          # 3. now the heavy imports are safe

    drift = pinned.check_code_defaults(p, Settings.model_fields)
    if drift and not getattr(args, "accept_default_drift", False):
        raise CliRefusal("code default drift (pass --accept-default-drift to record and continue): " + "; ".join(drift))
    mismatch = pinned.verify_against_settings(p, settings)
    if mismatch:
        raise CliRefusal("pinned settings differ from the imported Settings: " + "; ".join(mismatch))
    from app.experiments.replay import safety  # noqa: F401  (captures the original asyncpg.connect)

    recorded_timeout = p.record_only.get("LLM_TIMEOUT_SECONDS")
    if recorded_timeout is not None and float(settings.llm_timeout_seconds) != float(recorded_timeout):
        print(
            f"[replay] WARNING: local LLM_TIMEOUT_SECONDS={settings.llm_timeout_seconds} differs from the recorded "
            f"server value {recorded_timeout} (record-only setting; affects retries/timeouts, not the pinned keys)",
            file=sys.stderr,
        )

    problems = safety.verify_process_state(k)
    if problems:
        raise CliRefusal("process state check failed: " + "; ".join(problems))
    return Boot(p, settings, drift)


# --------------------------------------------------------------------------- pure helpers (stdlib)


def parse_prometheus(text: str, names: Sequence[str]) -> dict[str, float]:
    """Sum of every sample line of each wanted metric (labels ignored)."""
    out = {n: 0.0 for n in names}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        for name in names:
            if line.startswith(name) and (len(line) == len(name) or line[len(name)] in " {"):
                try:
                    out[name] += float(line.rsplit(" ", 1)[1])
                except ValueError:
                    pass
    return out


def percentile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def select_k_star(stats: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
    """Pre-registered K* rule (spec D): the largest K with zero errors/timeouts whose throughput
    beats the next-lower K by >= 20% and whose RM-call p90 latency is <= 2.5x that of the smallest K.
    A foreign-load share above 20% is reported as "shared" and asks the user whether to step down."""
    ks = sorted(stats)
    if not ks:
        return {"k_star": None, "reason": "no pilot data", "shared": False}
    first = ks[0]
    chosen, reason = None, "smallest K already has errors/timeouts"
    if stats[first].get("errors", 0) == 0:
        chosen, reason = first, "smallest K is the fallback"
        for prev, k in zip(ks, ks[1:]):
            s, ps = stats[k], stats[prev]
            base_p90 = stats[first].get("rm_p90")
            gain = (s["cases_per_hour"] / ps["cases_per_hour"]) if ps.get("cases_per_hour") else 0.0
            latency_ok = base_p90 is None or s.get("rm_p90") is None or s["rm_p90"] <= 2.5 * base_p90
            if s.get("errors", 0) == 0 and gain >= 1.2 and latency_ok:
                chosen, reason = k, f"K={k}: +{(gain - 1) * 100:.0f}% throughput vs K={prev}, RM p90 within 2.5x of K={first}"
            else:
                reason = f"stopped at K={prev}: K={k} failed (errors={s.get('errors')}, gain={gain:.2f}, latency_ok={latency_ok})"
                break
    foreign = max((s.get("foreign_share", 0.0) or 0.0 for s in stats.values()), default=0.0)
    return {"k_star": chosen, "reason": reason, "shared": foreign > 0.20, "foreign_share_max": foreign}


def http_get_text(url: str, timeout: float = 10.0) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310 - endpoint from settings
        return resp.read().decode("utf-8", "replace")


# --------------------------------------------------------------------------- extract


def _sample_cases(records: list[dict], sample: int, seed: int, min_buy: int) -> list[dict]:
    from app.experiments.replay import contract

    rng = random.Random(seed)
    ordered = sorted(records, key=lambda r: (r["trade_date"], r["ticker"]))
    buys = [r for r in ordered if contract.parse_stored_rm(r["transcript"].get("research_plan")) == "Buy"]
    if len(buys) < min_buy:
        raise CliRefusal(f"only {len(buys)} stored-RM-Buy cases in the parent set, need {min_buy}")
    chosen = rng.sample(buys, min_buy)
    ids = {r["case_id"] for r in chosen}
    rest = [r for r in ordered if r["case_id"] not in ids]
    if sample - len(chosen) > len(rest):
        raise CliRefusal(f"parent set has too few cases for --sample {sample}")
    chosen += rng.sample(rest, sample - len(chosen))
    return sorted(chosen, key=lambda r: (r["trade_date"], r["ticker"]))


def cmd_extract(args: argparse.Namespace) -> int:
    boot = bootstrap(args, k=1)
    from app.experiments.replay import contract, provenance, safety

    CASESETS.mkdir(parents=True, exist_ok=True)
    max_debate, max_risk = boot.settings.max_debate_rounds, boot.settings.max_risk_rounds
    git = provenance.git_state(REPO_ROOT)
    if args.from_caseset:
        parent, parent_meta = contract.load_caseset(CASESETS, args.from_caseset)
        records = _sample_cases(parent, args.sample, args.seed, args.min_stored_rm_buy)
        meta = {
            "derived_from": args.from_caseset, "parent_sha256": parent_meta["jsonl_sha256"], "sample": args.sample,
            "seed": args.seed, "min_stored_rm_buy": args.min_stored_rm_buy, "head": git["head"],
            "max_debate_rounds": max_debate, "max_risk_rounds": max_risk,
            "llm_models": _count(r["columns"].get("llm_model") for r in records),
        }
        jsonl, meta_path = contract.write_caseset(CASESETS, args.name, records, meta)
        print(f"[extract] {len(records)} cases -> {jsonl}")
        return EXIT_OK

    min_date = date.fromisoformat(args.min_trade_date) if args.min_trade_date else date(2000, 1, 1)
    max_date = date.fromisoformat(args.max_trade_date) if args.max_trade_date else date(2100, 1, 1)

    async def fetch() -> tuple[list[Any], dict[str, Any]]:
        conn = await safety.open_readonly_connection()
        try:
            async with safety.readonly_transaction(conn):
                rows = await contract.fetch_rows(conn, min_date, max_date)
            overrides = await safety.fetch_relevant_overrides(conn)
            return rows, overrides
        finally:
            await conn.close()

    rows, overrides = asyncio.run(fetch())
    sel = contract.select_cases(
        rows, max_debate_rounds=max_debate, max_risk_rounds=max_risk, min_trade_date=min_date, max_trade_date=max_date,
    )
    kept, excluded, drift_by_commit = apply_era_and_drift(
        sel.cases, sel.excluded, args.era, lambda commit: provenance.classify_commit(REPO_ROOT, commit),
    )
    excluded_by_reason = _count(e["reason"] for e in excluded)
    era_counts_before_filter = _count(r["era"] for r in sel.cases)
    meta = {
        "sql": contract.EXTRACT_SQL, "params": {"min_trade_date": min_date.isoformat(), "max_trade_date": max_date.isoformat(), "era": args.era},
        "counts": sel.counts, "n_after_era": len(kept), "excluded": excluded, "head": git["head"],
        "excluded_by_reason_final": excluded_by_reason, "era_counts_before_filter": era_counts_before_filter,
        "era_freeze_drift_cases": era_counts_before_filter.get("freeze_drift", 0),
        "git_commit_distribution": _count(r["columns"].get("git_commit") for r in kept),
        "llm_models": _count(r["columns"].get("llm_model") for r in kept),
        "era_distribution": _count(r["era"] for r in kept), "drift_by_commit": drift_by_commit,
        "max_debate_rounds": max_debate, "max_risk_rounds": max_risk, "real_overrides": overrides,
        "extracted_at": _stamp(),
    }
    bad = [o for o in overrides.get("rows", []) if o.get("account_mode") == "real"]
    if bad:
        meta["warning"] = "real-mode overrides of gate/profile settings exist: `run` will refuse"
    jsonl, meta_path = contract.write_caseset(CASESETS, args.name, kept, meta)
    print(f"[extract] {len(kept)} cases (of {sel.counts['input_rows']} rows) -> {jsonl}")
    print(f"[extract] excluded by reason (final, incl. era filter): {json.dumps(excluded_by_reason, ensure_ascii=False)}")
    print(f"[extract] era before the era filter: {json.dumps(era_counts_before_filter)}")
    for commit, info in sorted(drift_by_commit.items()):
        print(f"[extract] commit {commit or '(none)'}: vs_baseline={info['vs_baseline']} vs_head={info['vs_head']} "
              f"drift={info['drift']}{' ' + info['note'] if info.get('note') else ''}")
    if bad:
        print(f"[extract] WARNING: {meta['warning']}", file=sys.stderr)
    return EXIT_OK


def apply_era_and_drift(
    cases: list[dict[str, Any]], excluded: list[dict[str, Any]], era: str | None, classify: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Mark freeze-era cases whose commit really drifted (`classify(commit)` -> {vs_baseline, vs_head,
    drift}; rule in `provenance.classify_commit`) as `freeze_drift`, then apply the `--era` filter.
    Returns (kept, all exclusions incl. era ones, per-commit classification)."""
    drift_by_commit: dict[str, Any] = {}
    kept: list[dict[str, Any]] = []
    out_excluded = list(excluded)
    for rec in cases:
        if rec["era"] == "freeze":
            commit = rec["columns"].get("git_commit") or ""
            if commit not in drift_by_commit:
                drift_by_commit[commit] = (
                    classify(commit) if commit
                    else {"vs_baseline": None, "vs_head": None, "drift": True, "note": "no git_commit"}
                )
            if drift_by_commit[commit]["drift"]:
                rec["era"] = "freeze_drift"
        if era and rec["era"] != era:
            out_excluded.append({"memory_id": rec["memory_id"], "reason": f"era_{rec['era']}"})
            continue
        kept.append(rec)
    return kept, out_excluded, drift_by_commit


def _count(values: Any) -> dict[str, int]:
    out: dict[str, int] = {}
    for v in values:
        out[str(v)] = out.get(str(v), 0) + 1
    return dict(sorted(out.items()))


# --------------------------------------------------------------------------- preflight / provenance


def gather_provenance(boot: Boot, records: list[dict], caseset_sha: str, *, allow_dirty: bool) -> dict[str, Any]:
    """Source hashes, git state, model info, case-set/model consistency. Raises CliRefusal."""
    from app.experiments.replay import provenance

    current = provenance.current_source_hashes(REPO_ROOT)
    drift = provenance.source_drift(current)
    if drift:
        raise CliRefusal(f"{provenance.DRIFT_MESSAGE}: {drift}")
    git = provenance.git_state(REPO_ROOT)
    if git["pinned_files_dirty"] and not allow_dirty:
        raise CliRefusal(f"pinned source files have uncommitted changes (--allow-dirty records it): {git['dirty_lines']}")
    try:
        model = provenance.model_info(boot.settings.llm_base_url)
    except Exception as exc:  # noqa: BLE001 - any GET failure is a refusal
        raise CliRefusal(f"cannot read the model endpoint: {exc}") from exc
    pinned_model = boot.pinned.values["LLM_MODEL"]
    if model["id"] != pinned_model:
        raise CliRefusal(f"/models id {model['id']!r} != pinned LLM_MODEL {pinned_model!r}")
    case_models = {r["columns"].get("llm_model") for r in records}
    if case_models != {pinned_model}:
        raise CliRefusal(f"case set llm_model values {sorted(map(str, case_models))} != pinned {pinned_model!r}")
    base_key = provenance.compute_base_key(
        source_hashes=current, pinned_values=boot.pinned.keyed_values(), model=model, caseset_sha256=caseset_sha,
    )
    return {"source_hashes": current, "git": git, "model": model, "base_key": base_key}


def cmd_preflight(args: argparse.Namespace) -> int:
    boot = bootstrap(args, k=args.k)
    from app.experiments.replay import contract, safety

    records, meta = contract.load_caseset(CASESETS, args.caseset)
    prov = gather_provenance(boot, records, meta["jsonl_sha256"], allow_dirty=args.allow_dirty)

    async def overrides() -> dict[str, Any]:
        conn = await safety.open_readonly_connection()
        try:
            return await safety.fetch_relevant_overrides(conn)
        finally:
            await conn.close()

    found = asyncio.run(overrides())
    real = [o for o in found.get("rows", []) if o.get("account_mode") == "real"]
    if real:
        raise CliRefusal(f"real-mode overrides of gate/profile settings exist: {real}")
    inputs = [
        contract.build_inputs(r, max_debate_rounds=boot.settings.max_debate_rounds, max_risk_rounds=boot.settings.max_risk_rounds)
        for r in records
    ]
    freeze = [i for i in inputs if i.contract_checks.get("desk_note_stored_match") is not None]
    desk_ok = sum(1 for i in inputs if i.contract_checks.get("desk_note_stored_match") is True)
    print(f"[preflight] OK  cases={len(records)} model={prov['model']['id']} max_model_len={prov['model']['max_model_len']}")
    print(f"[preflight] base_key={prov['base_key'][:12]} head={prov['git']['head']} dirty={prov['git']['pinned_files_dirty']}")
    print(f"[preflight] desk_note_stored_match {desk_ok}/{len(freeze)} (freeze-era cases)")
    print(f"[preflight] K={args.k} limiter/env verified; pinned={boot.pinned.sha256[:12]} code_default_drift={boot.code_default_drift or 'none'}")
    return EXIT_OK


# --------------------------------------------------------------------------- run


async def _run_async(args: argparse.Namespace, boot: Boot, inputs: list[Any], prov: dict[str, Any], run_dir: Path, run_id: str) -> int:
    from app.config import settings
    from app.experiments.replay import provenance, runner, safety, variants

    # --- variant + fork guards (base_key was computed BEFORE the variant is applied) ---
    variant = variants.get_variant(args.arm, allow_test_variant=args.allow_test_variant)
    baseline_reader = None
    if args.baseline_run:
        baseline_dir = Path(args.baseline_run)
        baseline_manifest = json.loads((baseline_dir / "manifest.json").read_text(encoding="utf-8"))
        baseline_reader = lambda case_id: runner.read_ok_case(baseline_dir, case_id)  # noqa: E731
    elif variant.fork_at != "rm":
        raise CliRefusal(f"variant {variant.name} forks at '{variant.fork_at}': pass --baseline-run DIR")
    ctx = variants.apply_variant(variant)
    fingerprints = provenance.node_fingerprints(settings, runner.harness_node_digests())
    if args.baseline_run:
        variants.validate_fork(
            variant=variant, base_key=prov["base_key"], baseline_manifest=baseline_manifest,
            current_fingerprints=fingerprints, allow_noop=args.allow_noop_variant,
        )

    manifest_base = {
        "schema": "replay-run-v1", "run_id": run_id, "base_key": prov["base_key"], "arm": args.arm,
        "repeat": args.repeat, "variant": variant.name, "fork_at": variant.fork_at if args.baseline_run else "rm",
        "tag": args.run_tag, "k": args.k, "harness_version": HARNESS_VERSION, "contract_version": "rm-start-v1",
        "caseset": {"name": args.caseset, "sha256": prov["caseset_sha256"]}, "n_cases": len(inputs),
        "pinned": {"path": boot.pinned.path, "sha256": boot.pinned.sha256, "values": boot.pinned.keyed_values(),
                   "sources": boot.pinned.sources, "record_only": boot.pinned.record_only},
        "code_default_drift": boot.code_default_drift, "source_hashes": prov["source_hashes"], "model": prov["model"],
        "git": prov["git"], "node_fingerprints": fingerprints, "variant_patches": ctx.patches,
        "baseline_run": args.baseline_run,
        "flags": {"allow_market_hours": args.allow_market_hours, "allow_dirty": args.allow_dirty,
                  "accept_default_drift": args.accept_default_drift, "allow_test_variant": args.allow_test_variant,
                  "allow_noop_variant": args.allow_noop_variant},
        "python": sys.version.split()[0], "platform": sys.platform,
    }
    existing_path = run_dir / "manifest.json"
    resumed = existing_path.exists()
    if resumed and json.loads(existing_path.read_text(encoding="utf-8")).get("status") == "aborted_safety":
        raise CliRefusal("this run aborted on a safety violation; inspect safety_trips.jsonl before starting a new run")
    case_ids = [i.memory_id for i in inputs]
    tickers = {i.ticker for i in inputs}
    safety.set_trips_path(run_dir / "safety_trips.jsonl")
    cfg = runner.RunConfig(
        run_dir=run_dir, run_id=run_id, arm=args.arm, repeat=args.repeat, variant=variant.name,
        fork_at=variant.fork_at if args.baseline_run else "rm", k=args.k, case_timeout=args.case_timeout,
        max_attempts=args.max_attempts, max_consecutive_failures=args.max_consecutive_failures,
        allow_market_hours=args.allow_market_hours, baseline_reader=baseline_reader,
    )
    session = runner.RunSession(cfg, manifest_base)

    conn = await safety.open_readonly_connection()
    try:
        real_overrides = [
            o for o in (await safety.fetch_relevant_overrides(conn)).get("rows", []) if o.get("account_mode") == "real"
        ]
        if real_overrides:
            raise CliRefusal(f"real-mode overrides of gate/profile settings exist: {real_overrides}")
        start = await safety.snapshot_invariants(conn, case_ids)
    finally:
        await conn.close()
    invariants: dict[str, Any] = {"start": start}
    if resumed:
        prior = json.loads(existing_path.read_text(encoding="utf-8")).get("invariants") or {}
        invariants["start"] = prior.get("start") or start
    session.manifest["invariants"] = invariants
    safety.keep_awake(True)
    outcome = None
    try:
        outcome = await session.run(inputs)
    finally:
        safety.keep_awake(False)
        # The end snapshot is taken even when the run raised (e.g. aborted_error), so the manifest
        # always carries the DB evidence for what happened.
        try:
            conn = await safety.open_readonly_connection()
            try:
                end = await safety.snapshot_invariants(conn, case_ids, since=invariants["start"])
            finally:
                await conn.close()
            result = safety.compare_invariants(
                invariants["start"], end, tickers, run_windows=safety.parse_windows(session.manifest.get("windows", [])),
            )
            session.manifest["invariants"] = {"start": invariants["start"], "end": end, "result": result.as_dict()}
            session.write_manifest(best_effort=True)
            print(f"[run] invariants={result.status} trips={safety.trip_count()}")
            for msg in result.fails + result.warns:
                print(f"[run] invariant: {msg}")
            for row in result.candidates:
                print(f"[run] invariant candidate (confirm with aa-report --confirm-live-rows): {row['table']} id={row['id']}")
        except Exception as exc:  # noqa: BLE001 - never mask the run's own outcome/exception
            print(f"[run] WARNING: end-of-run DB invariant snapshot failed: {exc}", file=sys.stderr)
    print(f"[run] status={outcome.status} done={outcome.done} errors={outcome.errors} skipped_ok={outcome.skipped_ok} "
          f"gave_up={outcome.gave_up} remaining={outcome.remaining}")
    print(f"[run] run_dir={run_dir}")
    return outcome.exit_code


def cmd_run(args: argparse.Namespace) -> int:
    boot = bootstrap(args, k=args.k)
    from app.experiments.replay import contract, provenance, runner, safety, variants

    variants.get_variant(args.arm, allow_test_variant=args.allow_test_variant)  # cheap refusal first
    safety.assert_offhours(safety.live_now_kst(), allow_market_hours=args.allow_market_hours)
    records, meta = contract.load_caseset(CASESETS, args.caseset)
    if args.max_cases:
        records = sorted(records, key=lambda r: (r["trade_date"], r["ticker"]))[: args.max_cases]
    prov = gather_provenance(boot, records, meta["jsonl_sha256"], allow_dirty=args.allow_dirty)
    prov["caseset_sha256"] = meta["jsonl_sha256"]
    inputs = [
        contract.build_inputs(r, max_debate_rounds=boot.settings.max_debate_rounds, max_risk_rounds=boot.settings.max_risk_rounds)
        for r in records
    ]
    inputs = contract.session_prepass(inputs)  # serial, before any async work (spec A.5)
    run_id = provenance.make_run_id(args.arm, args.repeat, prov["base_key"], args.run_tag)
    run_dir = RUNS / run_id
    report = safety.install_blocks()
    runner.Instrumentation().install()
    runner.stdout_utf8()
    print(f"[run] {run_id}  cases={len(inputs)}  K={args.k}  blocks={len(report.installed)}+{len(report.aliases)} aliases", flush=True)
    print(f"[run] run_dir={run_dir}", flush=True)
    return asyncio.run(_run_async(args, boot, inputs, prov, run_dir, run_id))


# --------------------------------------------------------------------------- pilot


def _pilot_stats(run_dir: Path, wall: float, foreign: list[float], returncode: int) -> dict[str, Any]:
    cases = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((run_dir / "cases").glob("*.json"))]
    errors = len(list((run_dir / "errors").glob("*.json"))) if (run_dir / "errors").exists() else 0
    if returncode != 0:  # 75 paused / 3 safety / 76 incomplete / 2 refused all disqualify this K
        errors += 1
    walls = [c["wall_s"] for c in cases]
    labels: dict[str, list[float]] = {}
    tokens = 0
    for c in cases:
        tokens += c.get("tokens_total") or 0
        for node in c["nodes"].values():
            for call in node.get("calls", []):
                labels.setdefault(call["label"] or "", []).append(call["latency_seconds"] or 0.0)
    return {
        "batch_wall_s": wall, "cases": len(cases), "errors": errors, "returncode": returncode,
        "cases_per_hour": len(cases) / wall * 3600 if wall and cases else 0.0,
        "case_wall": {"p50": percentile(walls, 0.5), "p90": percentile(walls, 0.9)},
        "label_latency": {k: {"p50": percentile(v, 0.5), "p90": percentile(v, 0.9)} for k, v in sorted(labels.items())},
        "rm_p90": percentile(labels.get("research_manager", []), 0.9),
        "tokens_per_s": tokens / wall if wall else 0.0,
        "foreign_share": (sum(1 for f in foreign if f >= 1) / len(foreign)) if foreign else 0.0,
        "foreign_samples": len(foreign),
    }


def render_pilot_report(stats: Mapping[int, Mapping[str, Any]], decision: Mapping[str, Any]) -> str:
    lines = ["# Pilot report", "", "| K | batch wall s | cases | cases/h | case wall p50/p90 | RM p90 s | tokens/s | errors | foreign>=1 share |",
             "|---|---|---|---|---|---|---|---|---|"]
    for k, s in sorted(stats.items()):
        cw = s["case_wall"]
        lines.append(
            f"| {k} | {s['batch_wall_s']:.0f} | {s['cases']} | {s['cases_per_hour']:.1f} | {cw['p50']}/{cw['p90']} | "
            f"{s['rm_p90']} | {s['tokens_per_s']:.1f} | {s['errors']} | {s['foreign_share']:.0%} ({s['foreign_samples']} samples) |"
        )
    lines += ["", "## Label latency (p50/p90 s)", ""]
    for k, s in sorted(stats.items()):
        lines.append(f"- K={k}: " + "; ".join(f"{lab}: {v['p50']:.1f}/{v['p90']:.1f}" for lab, v in s["label_latency"].items() if v["p50"] is not None))
    lines += ["", f"**K\\* = {decision['k_star']}** ({decision['reason']})"]
    if decision["shared"]:
        lines.append("Foreign vLLM load was above 20% of samples: the vLLM is SHARED. Ask the user whether to step K* down one level.")
    return "\n".join(lines) + "\n"


def cmd_pilot(args: argparse.Namespace) -> int:
    boot = bootstrap(args, k=1)
    from app.experiments.replay import safety

    safety.assert_offhours(safety.live_now_kst(), allow_market_hours=args.allow_market_hours)
    root = re.sub(r"/v1/?$", "", boot.settings.llm_base_url.rstrip("/"))
    stats: dict[int, dict[str, Any]] = {}
    REPORTS.mkdir(parents=True, exist_ok=True)
    for k in [int(x) for x in args.ks.split(",")]:
        cmd = [sys.executable, "-u", "-m", "app.experiments.replay_downstream", "run", "--pinned", str(args.pinned or DEFAULT_PINNED),
               "--caseset", args.caseset, "--arm", "baseline", "--repeat", "1", "--k", str(k), "--run-tag", f"pilot_k{k}_{_stamp()}",
               "--max-attempts", "1"]
        if args.allow_market_hours:
            cmd.append("--allow-market-hours")
        if args.allow_dirty:
            cmd.append("--allow-dirty")
        if args.accept_default_drift:
            cmd.append("--accept-default-drift")
        started, foreign = time.monotonic(), []
        proc = subprocess.Popen(cmd, cwd=REPO_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8")
        run_dir: Path | None = None
        next_poll = 0.0
        lines: list[str] = []
        threading.Thread(target=_pump, args=(proc.stdout, lines), daemon=True).start()
        while proc.poll() is None:
            if time.monotonic() >= next_poll:
                next_poll = time.monotonic() + 15
                try:
                    m = parse_prometheus(http_get_text(f"{root}/metrics"), ["vllm:num_requests_running"])
                    own = _child_in_flight(lines)
                    foreign.append(max(0.0, m["vllm:num_requests_running"] - own))
                except Exception:  # noqa: BLE001 - the metrics endpoint is optional
                    pass
            time.sleep(1.0)
        wall = time.monotonic() - started
        found = [ln.split("run_dir=", 1)[1] for ln in lines if "run_dir=" in ln]
        run_dir = Path(found[-1]) if found else None
        if run_dir is None:
            raise CliRefusal(f"K={k}: the child run printed no run_dir (exit {proc.returncode}); see its output above")
        stats[k] = _pilot_stats(run_dir, wall, foreign, proc.returncode)
    decision = select_k_star(stats)
    path = REPORTS / f"pilot_{_stamp()}.md"
    path.write_text(render_pilot_report(stats, decision), encoding="utf-8", newline="\n")
    print(f"[pilot] report -> {path}\n[pilot] K* = {decision['k_star']} ({decision['reason']})")
    return EXIT_OK


def _pump(stream: Any, lines: list[str]) -> None:
    """Echo the child's output and keep it for run_dir / heartbeat lookups."""
    for raw in stream:
        line = raw.rstrip()
        lines.append(line)
        print(line, flush=True)


def _child_in_flight(lines: Sequence[str]) -> float:
    """Best effort: the child's own in-flight LLM calls, read from its manifest heartbeat."""
    for ln in reversed(lines):
        if "run_dir=" in ln:
            manifest = Path(ln.split("run_dir=", 1)[1]) / "manifest.json"
            try:
                return float(json.loads(manifest.read_text(encoding="utf-8"))["heartbeat"].get("llm_in_flight") or 0)
            except (OSError, ValueError, KeyError):
                return 0.0
    return 0.0


# --------------------------------------------------------------------------- aa-report / status


def cmd_aa_report(args: argparse.Namespace) -> int:
    from app.experiments.replay import aa_report

    confirm = Path(args.confirm_live_rows) if args.confirm_live_rows else None
    report, paths = aa_report.make_report(Path(args.r1), Path(args.r2), REPORTS, _stamp(), confirm_file=confirm)
    print(f"[aa-report] verdict: {report['verdict']}  (n={report['n_population']})")
    for r in report["rows"]:
        if r["verdict"] in ("FAIL", "N/A"):
            print(f"[aa-report] {r['verdict']}: {r['id']} {r['metric']} = {r['value']}")
    print(f"[aa-report] {paths['md']}")
    return EXIT_OK


def cmd_status(args: argparse.Namespace) -> int:
    run_dir = Path(args.run)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    n_ok = len(list((run_dir / "cases").glob("*.json"))) if (run_dir / "cases").exists() else 0
    n_err = len(list((run_dir / "errors").glob("*.json"))) if (run_dir / "errors").exists() else 0
    print(json.dumps({
        "run_id": manifest.get("run_id"), "status": manifest.get("status"), "n_cases": manifest.get("n_cases"),
        "ok_files": n_ok, "error_files": n_err, "heartbeat": manifest.get("heartbeat"),
        "safety_trips": manifest.get("safety_trips"), "k": manifest.get("k"),
    }, indent=1, ensure_ascii=False))
    return EXIT_OK


# --------------------------------------------------------------------------- parser / main


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m app.experiments.replay_downstream", description=__doc__.split("\n")[0],
        epilog="Exit codes: 0 done (all cases ok), 2 refused, 3 safety violation, 75 paused (resume with the same "
               "command), 76 finished with cases still in error (rerun to retry).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser, *, k: bool = True) -> None:
        p.add_argument("--pinned", default=None, help=f"pinned settings JSON (default {DEFAULT_PINNED})")
        p.add_argument("--accept-default-drift", action="store_true")
        p.add_argument("--allow-dirty", action="store_true")
        if k:
            p.add_argument("--k", type=int, default=4, help="concurrent cases (= LLM limiter width)")

    p = sub.add_parser("extract", help="build a case set from the DB (read-only SELECT) or derive one")
    common(p, k=False)
    p.add_argument("--name", required=True)
    p.add_argument("--era", choices=["freeze", "pre_freeze", "mixed_0917", "freeze_drift"])
    p.add_argument("--min-trade-date")
    p.add_argument("--max-trade-date")
    p.add_argument("--from-caseset")
    p.add_argument("--sample", type=int, default=12)
    p.add_argument("--seed", type=int, default=20261001)
    p.add_argument("--min-stored-rm-buy", type=int, default=0)
    p.set_defaults(func=cmd_extract)

    p = sub.add_parser("preflight", help="verify pinned settings, sources, model endpoint and case set (no LLM calls)")
    common(p)
    p.add_argument("--caseset", required=True)
    p.set_defaults(func=cmd_preflight)

    p = sub.add_parser("pilot", help="measure throughput for several K (needs the market window closed)")
    common(p, k=False)
    p.add_argument("--caseset", required=True)
    p.add_argument("--ks", default="2,4,6")
    p.add_argument("--allow-market-hours", action="store_true")
    p.set_defaults(func=cmd_pilot)

    p = sub.add_parser("run", help="one (arm, repeat) run; resumable")
    common(p)
    p.add_argument("--caseset", required=True)
    p.add_argument("--arm", default="baseline", help="variant name (registry in replay/variants.py)")
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--baseline-run")
    p.add_argument("--max-cases", type=int)
    p.add_argument("--run-tag")
    p.add_argument("--case-timeout", type=float, default=1800.0)
    p.add_argument("--max-attempts", type=int, default=2)
    p.add_argument("--max-consecutive-failures", type=int, default=3)
    p.add_argument("--allow-market-hours", action="store_true")
    p.add_argument("--allow-test-variant", action="store_true")
    p.add_argument("--allow-noop-variant", action="store_true")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("aa-report", help="compare two baseline runs against the stored results")
    p.add_argument("--r1", required=True)
    p.add_argument("--r2", required=True)
    p.add_argument("--confirm-live-rows", help='JSON {"memory_entries": [ids], "trades": [ids], "pending_orders": [ids]} '
                   "the human verified as live rows (releases FAIL CANDIDATE rows outside the harness run windows)")
    p.set_defaults(func=cmd_aa_report)

    p = sub.add_parser("status", help="show a run's manifest status")
    p.add_argument("--run", required=True)
    p.set_defaults(func=cmd_status)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    _utf8()
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except pinned.PinnedError as exc:
        print(f"[replay] refused: {exc}", file=sys.stderr)
        return EXIT_CONFIG
    except CliRefusal as exc:
        print(f"[replay] refused: {exc}", file=sys.stderr)
        return EXIT_CONFIG
    except Exception as exc:  # noqa: BLE001 - map the harness' own refusals, re-raise the rest
        name = type(exc).__name__
        if name in {"MarketWindowError", "VariantRefused", "ManifestMismatch", "ContractError", "PairError"}:
            print(f"[replay] refused: {exc}", file=sys.stderr)
            return EXIT_CONFIG
        raise
    except BaseException as exc:  # ReplaySafetyViolation is a BaseException on purpose
        if type(exc).__name__ == "ReplaySafetyViolation":
            print(f"[replay] SAFETY VIOLATION, aborting: {exc}", file=sys.stderr)
            return EXIT_SAFETY
        raise


if __name__ == "__main__":
    sys.exit(main())
