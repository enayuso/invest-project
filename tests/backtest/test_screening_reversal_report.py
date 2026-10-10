"""app.backtest.screening_reversal_report (T2) — synthetic dumps built with the real factor functions,
hand-computed expectations, no network/DB/LLM. Test numbers follow design section G.
합성 덤프 + 손계산 기대값 (설계서 G의 테스트 번호)."""

from __future__ import annotations

import json
import math
import random
import re
import statistics
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from app.backtest import factor_metrics as fm
from app.backtest import screening_reversal_report as mod
from app.chat.screening import rank

DEFAULT_CHUNKS = [
    ["2025-09-01", "2025-09-05", "2025-09-11", "2025-09-17"],
    ["2025-09-24", "2025-09-30", "2025-10-13", "2025-10-17"],
]
LBS = [5, 10, 15, 20]
BANNED = ("PASS", "FAIL", "SUPPORTED", "REFUTED", "TAIL_ONLY", "INCONSISTENT", "지지", "반박")


@pytest.fixture(autouse=True)
def _relax(monkeypatch):
    """Small synthetic grids: relax the pre-registered minimum date counts; fake the holiday table."""
    monkeypatch.setattr(mod, "MIN_DATES", {4: 1, 10: 1})
    monkeypatch.setattr(mod, "_kr_holiday_lookup", lambda d: False)
    monkeypatch.setattr(mod, "_load_cost_fn", lambda: (lambda t, side: 5.0 if side == "buy" else 25.0))


# --- synthetic grid builder ------------------------------------------------------------

def _summary(obs: list[dict], n_boot: int, holding: int, cadence: int) -> dict:
    ex = [o["excess_return"] for o in obs if o["excess_return"] is not None]
    ics = [o["rank_ic"] for o in obs]
    ic_ok = [x for x in ics if x is not None]
    sp = [o["quantile_spread"] for o in obs if o["quantile_spread"] is not None]
    ci_e = fm.block_bootstrap_ci([[x] for x in ex], n_bootstrap=n_boot, seed=0)
    ci_i = fm.block_bootstrap_ci([[x] for x in ic_ok], n_bootstrap=n_boot, seed=0)
    return {
        "mean_excess_return": sum(ex) / len(ex) if ex else None, "excess_return_ci95": list(ci_e) if ci_e else None,
        "mean_rank_ic": sum(ic_ok) / len(ic_ok) if ic_ok else None, "rank_ic_ci95": list(ci_i) if ci_i else None,
        "icir": fm.icir(ics), "mean_quantile_spread": sum(sp) / len(sp) if sp else None,
        "n_eff": fm.effective_sample_size(len(obs), cadence, holding),
    }


def _make_grid(tmp_path, *, grid="dev_h4_c4", chunk_dates=None, n_tickers=10, holding=4, cadence=4, fallbacks=None,
               null_fwd=(), null_bench=(), degraded=(), n_boot=200, heads=None, dirty=(), mutate=None, zero_dates=(), signal=0.0):
    base = tmp_path / "rev"
    (base / "chunks").mkdir(parents=True)
    universe = [f"T{i:02d}.KS" for i in range(n_tickers)]
    chunk_dates = chunk_dates or DEFAULT_CHUNKS
    names = [f"{grid}_c{i + 1:02d}" for i in range(len(chunk_dates))]
    fallbacks = fallbacks or [0] * len(names)
    heads = heads or ["a" * 40] * len(names)
    variants = mod.expected_variants(mod.role_of(grid))
    all_dates = [d for cd in chunk_dates for d in cd]
    plan = {"calendar_ticker": universe[0], "min_per_date": 3.7, "grids": {grid: {
        "start": all_dates[0], "end": all_dates[-1], "holding": holding, "cadence": cadence, "n_dates": len(all_dates), "all_dates": all_dates,
        "chunks": [{"chunk": n, "n_dates": len(cd), "start": cd[0], "end": cd[-1], "dates": cd} for n, cd in zip(names, chunk_dates)]}}}
    dumps, txts, runlog = {}, {}, []
    for ci, (name, dates) in enumerate(zip(names, chunk_dates)):
        scores, bench, forward = {}, {}, {}
        for d in dates:
            for lb in LBS:
                rng = random.Random(f"s|{d}|{lb}")
                scores[f"{lb}|{d}"] = {t: {"raw_pct": round(rng.uniform(-0.1, 0.1), 6), "atr_pct": round(rng.uniform(0.01, 0.05), 6)} for t in universe}
                b = round(rng.uniform(-0.02, 0.02), 6)
                bench[f"{lb}|{d}"] = {"domestic": None if (lb, d) in null_bench else b}
            rng = random.Random(f"f|{d}")
            forward[d] = {}
            for t in universe:
                noise = 0.01 if signal else 0.05
                s5 = scores[f"5|{d}"][t]
                r = round(rng.uniform(-noise, noise) + signal * (-s5["raw_pct"] / s5["atr_pct"]), 6)  # signal>0: reversal pays
                gone = (d, t) in null_fwd
                exit_day = d if d in zero_dates else (date.fromisoformat(d) + timedelta(days=4)).isoformat()
                if d in zero_dates:
                    r = 0.0
                forward[d][t] = {"fwd_return": None if gone else r, "entry_date": None if gone else d,
                                 "exit_price_date": None if gone else exit_day}
        vout = []
        for v in variants:
            lb, obs, prev = v["lookback_days"], {}, None
            for d in dates:
                sc = scores[f"{lb}|{d}"]
                signed, fb = {}, 0
                for t, s in sc.items():
                    num = s["raw_pct"]
                    if v["momentum_mode"] == "residual":
                        b = bench[f"{lb}|{d}"]["domestic"]
                        if b is None:
                            fb += 1
                        else:
                            num = num - b
                    signed[t] = v["sign"] * (num / s["atr_pct"] if v["use_volatility_adjustment"] else num)
                fwd = {t: x["fwd_return"] for t, x in forward[d].items()}
                picks = [t for t, _ in rank(signed, 4)]
                pf = {t: fwd[t] for t in picks if fwd[t] is not None}
                uni = [r for t, r in fwd.items() if t in sc and r is not None]
                obs[d] = {"picks": picks, "pick_forward_returns": pf, "universe_forward_returns": uni,
                          "excess_return": fm.excess_return(list(pf.values()), uni), "rank_ic": fm.rank_ic(signed, fwd),
                          "quantile_spread": fm.quantile_spread(signed, fwd), "turnover": fm.turnover(prev, picks) if prev is not None else None,
                          "residual_fallbacks": fb, "dedup_drop_rate": 0.0, "dedup_degraded": (v["label"], d) in degraded, "signed_scores": signed}
                prev = picks
            vout.append({"variant": dict(v), "observations": obs, "summary": _summary(list(obs.values()), n_boot, holding, cadence)})
        started = datetime(2026, 9, 30, 8, 0, 2, 500000, tzinfo=timezone.utc) + timedelta(hours=ci)
        finished = started + timedelta(minutes=30)
        dumps[name] = {
            "schema": mod.DUMP_SCHEMA, "status": "complete", "started_at": started.isoformat(), "finished_at": finished.isoformat(),
            "git_head": heads[ci], "git_dirty": ci in dirty, "argv": [], "universe": universe, "trade_dates": list(dates),
            "trade_dates_requested": list(dates), "holding_days": holding, "cadence_days": cadence, "top_n": 4,
            "benchmark": {"markets": ["domestic"], "used": True}, "scores": scores, "benchmark_pct": bench, "forward": forward, "variants": vout,
            "args": {"start": dates[0], "end": dates[-1], "tickers": ",".join(universe), "variants_json": "v.json", "top_n": 4, "holding_days": holding,
                     "cadence_days": cadence, "worth_bar": 0.03, "bootstrap_iters": n_boot, "seed": 0, "dump_json": name},
        }
        lines = ["Sweeping 8 variant(s) ..."]
        for i, d in enumerate(dates, start=1):
            lines.append(f"[{i}/{len(dates)}] {d} scored={n_tickers} priced={n_tickers - sum(1 for t in universe if (d, t) in null_fwd)} bench=ok")
            if i == 1:
                lines += [f"KIS get_price_series failed for T0{k}.KS, falling back to yfinance" for k in range(fallbacks[ci])]
        txts[name] = "\n".join(lines) + "\n"
        runlog.append({
            "chunk": name, "start_kst": ((started + timedelta(hours=9) - timedelta(seconds=5)).replace(tzinfo=None, microsecond=0)).isoformat(),
            "end_kst": ((finished + timedelta(hours=9) + timedelta(seconds=1)).replace(tzinfo=None, microsecond=0)).isoformat(),
            "minutes": 30.0, "exit": 0, "range": f"{dates[0]}..{dates[-1]}", "holding": holding, "cadence": cadence, "progress_lines": len(dates),
            "last_progress": lines[-1], "fallback_price": fallbacks[ci], "atr_fallback": 0, "price_unavailable": 0, "cluster_failed": 0,
            "benchmark_failed": 0, "read_timeout": 0, "egw00201": 0, "market_hours_override": False, "git_head": heads[ci][:7]})
    ctx = SimpleNamespace(dumps=dumps, txts=txts, runlog=runlog, plan=plan, u86=",".join(universe), variants_dev=list(mod.PREREG_VARIANTS))
    if mutate:
        mutate(ctx)
    (base / "chunks.json").write_text(json.dumps(ctx.plan), encoding="utf-8")
    (base / "u86.txt").write_text(ctx.u86, encoding="utf-8")
    (base / "variants_dev.json").write_text(json.dumps(ctx.variants_dev), encoding="utf-8")
    (base / "runlog.jsonl").write_bytes(b"\xef\xbb\xbf" + "\n".join(json.dumps(r) for r in ctx.runlog).encode("utf-8") + b"\n")
    for n, d in ctx.dumps.items():
        (base / "chunks" / f"{n}.json").write_text(json.dumps(d), encoding="utf-8")
    for n, t in ctx.txts.items():
        (base / "chunks" / f"{n}.txt").write_text(t, encoding="utf-8")
    return SimpleNamespace(base=base, out=tmp_path / "out", grid=grid, names=names, universe=universe, dates=all_dates, plan=ctx.plan)


def _cli(G, mode, *extra, n_boot=200):
    return mod.main(["--grid", G.grid, "--mode", mode, "--base-dir", str(G.base), "--out-dir", str(G.out), *extra], n_bootstrap=n_boot)


def _json(G, mode):
    return json.loads((G.out / f"{G.grid}_{mode}.json").read_text(encoding="utf-8"))


def _checks(G, mode="check"):
    return {c["id"]: c for c in _json(G, mode)["checks"]}


def _bad(G, mode="check"):
    return sorted(i for i, c in _checks(G, mode).items() if not c["ok"])


def _pipeline(G, *extra, n_boot=200):
    assert _cli(G, "freeze", *extra) == 0
    return _cli(G, "report", *extra, n_boot=n_boot)


# --- 1-17 merge and refusal --------------------------------------------------------------

def test_01_normal_grid_passes_check(tmp_path, capsys):
    G = _make_grid(tmp_path)
    assert _cli(G, "check") == 0
    j = _json(G, "check")
    assert j["status"] == "checked" and j["schema"] == mod.REPORT_SCHEMA and j["role"] == "dev"
    ids = _checks(G)
    assert all(c["ok"] for c in ids.values())
    for need in ["I1", "I2", "I3", "I4", "I5", "I6", "I7", "I8", "I9", "I10", "I11", "I12", "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8"]:
        assert need in ids
    # 2 chunks x 8 variants x 4 dates x 7 recomputed fields
    assert "exact 448/448" in ids["I11"]["detail"]
    assert "variants" not in j  # no metrics in check mode
    assert capsys.readouterr().out.startswith("screening-reversal T2")


def test_02_status_incomplete_is_refused(tmp_path):
    def m(c):
        c.dumps["dev_h4_c4_c02"]["status"] = "incomplete"
    G = _make_grid(tmp_path, mutate=m)
    assert _cli(G, "check") == 4
    assert _bad(G) == ["I3"] and _json(G, "check")["status"] == "refused"


def test_03_schema_v0_is_refused(tmp_path):
    def m(c):
        c.dumps["dev_h4_c4_c01"]["schema"] = "screening-sweep-dump-v0"
    G = _make_grid(tmp_path, mutate=m)
    assert _cli(G, "check") == 4 and "I3" in _bad(G)


def test_04_missing_chunk_refused_unless_incomplete_allowed(tmp_path):
    G = _make_grid(tmp_path)
    (G.base / "chunks" / "dev_h4_c4_c02.json").unlink()
    assert _cli(G, "check") == 4 and _bad(G) == ["I2"]
    assert _cli(G, "check", "--allow-incomplete-set") == 0
    assert _checks(G)["A6"]["severity"] == "warning"
    assert _cli(G, "freeze", "--allow-incomplete-set") == 2  # check-only flag


def test_05_unplanned_chunk_is_refused_even_when_incomplete_allowed(tmp_path):
    G = _make_grid(tmp_path)
    (G.base / "chunks" / "dev_h4_c4_c09.json").write_bytes((G.base / "chunks" / "dev_h4_c4_c01.json").read_bytes())
    (G.base / "chunks" / "discarded").mkdir()
    (G.base / "chunks" / "discarded" / "dev_h4_c4_c01__x.json").write_text("{}", encoding="utf-8")  # sub-folder ignored
    assert _cli(G, "check", "--allow-incomplete-set") == 4 and _bad(G) == ["I2"]


def test_06_plan_missing_a_date_is_refused(tmp_path):
    def m(c):
        c.plan["grids"]["dev_h4_c4"]["all_dates"].pop()
    G = _make_grid(tmp_path, mutate=m)
    assert _cli(G, "check") == 4 and _bad(G) == ["I1"]


def test_07_overlapping_chunk_dates_are_refused(tmp_path):
    def m(c):
        g = c.plan["grids"]["dev_h4_c4"]
        g["chunks"][1]["dates"] = [g["chunks"][0]["dates"][-1], *g["chunks"][1]["dates"][1:]]
    G = _make_grid(tmp_path, mutate=m)
    assert _cli(G, "check") == 4 and "I1" in _bad(G)


def test_08_universe_order_differs_between_chunks(tmp_path):
    def m(c):
        d = c.dumps["dev_h4_c4_c02"]
        u = list(d["universe"])
        u[0], u[1] = u[1], u[0]
        d["universe"] = u
        for day, per in d["forward"].items():
            d["forward"][day] = {t: per[t] for t in u}
    G = _make_grid(tmp_path, mutate=m)
    assert _cli(G, "check") == 4 and "I6" in _bad(G)


def test_09_rev_lb5_sign_tamper_is_refused(tmp_path):
    def m(c):
        for d in c.dumps.values():
            next(v for v in d["variants"] if v["variant"]["label"] == "rev_lb5")["variant"]["sign"] = 1
    G = _make_grid(tmp_path, mutate=m)
    assert _cli(G, "check") == 4 and _bad(G) == ["I7"]


def test_10_git_head_differences(tmp_path, monkeypatch):
    G = _make_grid(tmp_path, heads=["a" * 40, "b" * 40])
    monkeypatch.setattr(mod, "git_changed_paths", lambda a, b: None)
    assert _cli(G, "check") == 4 and _bad(G) == ["I8"]
    assert _cli(G, "check", "--allow-heads", "aaaa,bbbb") == 0
    monkeypatch.setattr(mod, "git_changed_paths", lambda a, b: ["app/live/auto_trade_loop.py"])
    assert _cli(G, "check", "--allow-heads", "aaaa,bbbb") == 4  # --allow-heads applies only when git cannot diff
    monkeypatch.setattr(mod, "git_changed_paths", lambda a, b: ["app/backtest/screening_reversal_report.py", "tests/backtest/test_screening_reversal_report.py", "notes/x.md"])
    assert _cli(G, "check") == 0
    monkeypatch.setattr(mod, "git_changed_paths", lambda a, b: ["app/live/auto_trade_loop.py"])
    assert _cli(G, "check") == 4 and _bad(G) == ["I8"]


def test_11_dirty_dump_refused_by_freeze_and_report_only(tmp_path):
    G = _make_grid(tmp_path, dirty=(0,))
    assert _cli(G, "check") == 0 and _checks(G)["A8"]["severity"] == "warning"
    assert _cli(G, "freeze") == 3 and _bad(G, "freeze") == ["A8"]
    assert _cli(G, "freeze", "--allow-dirty") == 0


def test_12_chunk_fallback_limit_is_8(tmp_path):
    G = _make_grid(tmp_path, fallbacks=[9, 0])
    assert _cli(G, "check") == 3 and _bad(G) == ["A2"]
    G2 = _make_grid(tmp_path / "ok", fallbacks=[8, 0])
    assert _cli(G2, "check") == 0


def test_13_grid_fallback_total_limit_is_50(tmp_path):
    chunks = [[f"2025-09-{i:02d}"] for i in range(1, 9)]
    G = _make_grid(tmp_path / "a", chunk_dates=chunks, fallbacks=[7] * 7 + [0])  # 49
    assert _cli(G, "check") == 0 and "total 49" in _checks(G)["A3"]["detail"]
    G2 = _make_grid(tmp_path / "b", chunk_dates=chunks, fallbacks=[7] * 8)  # 56, every chunk <= 8
    assert _cli(G2, "check") == 3 and _bad(G2) == ["A3"]


def test_14_runlog_vs_txt_counts_case_insensitive(tmp_path):
    def m(c):
        c.txts["dev_h4_c4_c01"] += "kis GET_PRICE_SERIES FAILED for X.KS, fallback\nKIS get_price_series failed for Y.KS\nKis Get_Price_Series Failed for Z.KS\n"
        c.runlog[0]["fallback_price"] = 3
    G = _make_grid(tmp_path, mutate=m)
    assert _cli(G, "check") == 0  # all three mixed-case lines are counted
    def m2(c):
        m(c)
        c.runlog[0]["fallback_price"] = 2
    G2 = _make_grid(tmp_path / "b", mutate=m2)
    assert _cli(G2, "check") == 4 and _bad(G2) == ["I10"]


def test_15_runlog_time_window_picks_exactly_one_record(tmp_path):
    def stale(c):
        old = dict(c.runlog[0])
        old["start_kst"], old["end_kst"] = "2026-09-29T10:00:00", "2026-09-29T10:30:00"
        c.runlog.insert(0, old)
    G = _make_grid(tmp_path, mutate=stale)
    assert _cli(G, "check") == 0
    assert "discarded attempts 1" in _checks(G)["I9"]["detail"]
    assert _json(G, "check")["inputs"]["chunks"][0]["discarded_attempts"] == 1

    def twin(c):
        c.runlog.append(dict(c.runlog[0]))
    G2 = _make_grid(tmp_path / "b", mutate=twin)
    assert _cli(G2, "check") == 4 and "I9" in _bad(G2)


def test_16_priced_9_of_10_fails_acceptance(tmp_path):
    G = _make_grid(tmp_path, null_fwd={("2025-09-05", "T03.KS")})
    assert _cli(G, "check") == 3 and _bad(G) == ["A4"]
    assert "below 10" in _checks(G)["A4"]["detail"] and mod._min_scored(86) == 80 and mod._min_scored(10) == 10


def test_17a_dedup_degraded_is_refused_and_17b_resid_invalid_only_that_variant(tmp_path):
    G = _make_grid(tmp_path, degraded={("rev_lb10", "2025-09-05")})
    assert _cli(G, "check") == 3 and _bad(G) == ["A5"]
    G2 = _make_grid(tmp_path / "r", chunk_dates=[DEFAULT_CHUNKS[0]], null_bench={(5, "2025-09-01"), (5, "2025-09-11")})
    assert _pipeline(G2) == 0
    v = _json(G2, "report")["variants"]
    assert v["rev_lb5_resid"]["verdict"]["code"] == "INVALID" and v["rev_lb5_resid"]["resid_invalid"] is True  # 2/4 > 20%
    assert all(v[k]["verdict"] is None or v[k]["verdict"]["code"] != "INVALID" for k in v if k != "rev_lb5_resid")


# --- 18-21 reconstruction and instrument --------------------------------------------------

def test_18_excess_tamper_is_refused(tmp_path):
    def m(c):
        o = c.dumps["dev_h4_c4_c02"]["variants"][1]["observations"]["2025-09-30"]
        o["excess_return"] += 1e-6
    G = _make_grid(tmp_path, mutate=m)
    assert _cli(G, "check") == 4 and "I11" in _bad(G)


def test_19_signed_scores_tamper_is_refused(tmp_path):
    def m(c):
        o = c.dumps["dev_h4_c4_c01"]["variants"][2]["observations"]["2025-09-11"]
        t = next(iter(o["signed_scores"]))
        o["signed_scores"][t] += 1e-6
    G = _make_grid(tmp_path, mutate=m)
    assert _cli(G, "check") == 4 and "I11" in _bad(G)


def test_20_chunk_boundary_turnover(tmp_path):
    G = _make_grid(tmp_path)
    assert _cli(G, "check") == 0
    assert _json(G, "check")["self_consistency"]["turnover"]["chunk_first_null"] == "16/16"  # 8 variants x 2 chunks
    plan = mod.load_plan(G.base, G.grid)
    _, _, merged, _ = mod.validate(G.base, G.grid, "dev", plan, "check", {})
    t = mod.merged_turnover(merged, "rev_lb5")
    first_c2 = "2025-09-24"
    expect = fm.turnover(merged.obs["rev_lb5"]["2025-09-17"]["picks"], merged.obs["rev_lb5"][first_c2]["picks"])
    assert t["2025-09-01"] is None and t[first_c2] == expect and expect is not None
    assert merged.obs["rev_lb5"][first_c2]["turnover"] is None  # stored value restarts per chunk

    def bad_first(c):
        c.dumps["dev_h4_c4_c02"]["variants"][1]["observations"]["2025-09-24"]["turnover"] = 0.5
    assert _cli(_make_grid(tmp_path / "x", mutate=bad_first), "check") == 4

    def bad_mid(c):
        c.dumps["dev_h4_c4_c02"]["variants"][1]["observations"]["2025-09-30"]["turnover"] = 0.123
    assert _cli(_make_grid(tmp_path / "y", mutate=bad_mid), "check") == 4


def test_21_instrument_check_and_tie_explained(tmp_path):
    G = _make_grid(tmp_path)
    assert _cli(G, "check") == 0 and _json(G, "check")["instrument"]["ok"] == 8

    def m(c):
        mom = next(v for v in c.dumps["dev_h4_c4_c01"]["variants"] if v["variant"]["label"] == "mom_lb5")
        mom["observations"]["2025-09-05"]["rank_ic"] += 0.01
    assert _cli(_make_grid(tmp_path / "x", mutate=m), "check") == 4

    # five tickers, k=1, rev signed A=2,B=1,C=0,D=-1,E=-1 and fwd .01..05: spreads sum to 0.01 but the D/E tie straddles the cut
    sr = {"A": 2.0, "B": 1.0, "C": 0.0, "D": -1.0, "E": -1.0}
    sm = {t: -v for t, v in sr.items()}
    fw = {"A": 0.01, "B": 0.02, "C": 0.03, "D": 0.04, "E": 0.05}
    assert fm.quantile_spread(sr, fw) + fm.quantile_spread(sm, fw) == pytest.approx(0.01)
    assert mod._tie_crosses_boundary(sr, fw)

    def obs(signed):
        return {"signed_scores": signed, "rank_ic": fm.rank_ic(signed, fw), "quantile_spread": fm.quantile_spread(signed, fw)}
    merged = _mini_merged({"rev_lb5": {"d1": obs(sr)}, "mom_lb5": {"d1": obs(sm)}}, {"d1": {t: (f, "d1", "d1") for t, f in fw.items()}})
    res = mod.instrument_check(merged)
    assert res["failures"] == [] and res["tie_explained"] == 1
    # without a tie the same mismatch fails
    sr2 = {"A": 2.0, "B": 1.0, "C": 0.0, "D": -1.0, "E": -2.0}
    sm2 = {t: -v for t, v in sr2.items()}
    o_rev, o_mom = obs(sr2), obs(sm2)
    o_mom["quantile_spread"] += 0.01
    merged2 = _mini_merged({"rev_lb5": {"d1": o_rev}, "mom_lb5": {"d1": o_mom}}, {"d1": {t: (f, "d1", "d1") for t, f in fw.items()}})
    assert mod.instrument_check(merged2)["failures"] == ["d1: spread"]


def _mini_merged(obs, forward, scores=None, holding=4, cadence=4, role="dev"):
    dates = sorted(forward)
    variants = [{"label": k, "lookback_days": 5, "use_volatility_adjustment": True, "use_correlation_dedup": True, "sign": 1, "momentum_mode": "total"} for k in obs]
    return mod.MergedGrid(grid="dev_x", role=role, dates=dates, universe=[], holding=holding, cadence=cadence, top_n=4, variants=variants,
                          lookbacks=[5], scores=scores or {}, benchmark={}, forward=forward, obs=obs, chunk_of={d: "c" for d in dates})


# --- 22-28 statistics primitives ----------------------------------------------------------

G6 = [0.010, -0.004, 0.006, 0.002, 0.008, -0.001]


def test_22_ci_shift_property():
    gross = mod.ci_from_means(mod.bootstrap_means(G6, 2000, 0))
    assert gross == pytest.approx((-0.0006666666666666666, 0.007333333333333333), abs=1e-15)
    for cost, mean, lo, hi in [(0.003, 0.0005, -0.0036666666666666666, 0.004333333333333333), (0.005, -0.0015, None, None), (0.008, -0.0045, None, None)]:
        net = [x - cost for x in G6]
        assert sum(net) / 6 == pytest.approx(mean, abs=1e-15)
        ci = mod.ci_from_means(mod.bootstrap_means(net, 2000, 0))
        assert abs(ci[0] - (gross[0] - cost)) <= 1e-12 and abs(ci[1] - (gross[1] - cost)) <= 1e-12
        if lo is not None:
            assert ci == pytest.approx((lo, hi), abs=1e-15)
    assert sum(G6) / 6 == pytest.approx(0.0035)


def test_23_bootstrap_replica_is_bit_identical_to_tool():
    vals = [0.01, -0.02, 0.03, 0.005, -0.004, 0.012, 0.0, 0.02]
    assert mod.ci_from_means(mod.bootstrap_means(vals, 500, 0)) == fm.block_bootstrap_ci([[v] for v in vals], n_bootstrap=500, seed=0)
    assert mod.ci_from_means(mod.bootstrap_means(vals, 10_000, 0)) == fm.block_bootstrap_ci([[v] for v in vals], n_bootstrap=10_000, seed=0)
    assert mod.bootstrap_means([0.1, 0.2], 50, 0) is None and mod.ci_from_means(None) is None


def test_24_p_values():
    B = 200
    assert mod.p_boot_one_sided(mod.bootstrap_means([0.01] * 5, B, 0)) == pytest.approx(1 / (B + 1))
    assert mod.p_boot_one_sided(mod.bootstrap_means([-0.01] * 5, B, 0)) == 1.0
    assert mod.p_boot_one_sided(None) is None
    assert mod.p_normal_one_sided(0.0005, (-0.002, 0.003)) == pytest.approx(0.34752910690110383, abs=1e-15)
    assert mod.p_normal_one_sided(0.004, (0.0005, 0.0075)) == pytest.approx(0.012545461435946571, abs=1e-15)


def test_25_holm():
    rej = lambda r: sorted(k for k, v in r.items() if v["rejected"])  # noqa: E731
    mk = lambda ps: {f"e{i}": p for i, p in enumerate(ps)}  # noqa: E731
    assert rej(mod.holm(mk([0.001, 0.004, 0.02, 0.03, 0.5]), True)) == ["e0", "e1"]
    assert rej(mod.holm(mk([0.006, 0.001, 0.9, 0.9, 0.9]), True)) == ["e0", "e1"]
    assert rej(mod.holm(mk([0.005, 0.9, 0.9, 0.9, 0.9]), True)) == ["e0"]  # exactly alpha/5 rejects
    assert rej(mod.holm(mk([0.001, 0.004, 0.02, 0.03, 0.5]), False)) == []
    r = mod.holm(mk([0.001, 0.002, 0.003, 0.9, 0.9]), True, invalid={"e0"})
    assert rej(r) == [] and r["e0"]["rank"] == 1  # INVALID is never rejected and ends the step-down
    assert mod.holm(mk([0.004, None, 0.9, 0.9, 0.9]), True)["e1"]["rejected"] is False


@pytest.mark.parametrize("kw, code, rule", [
    (dict(n_eff=19, mean_ic=-0.05, mean_gross=0.01, ci_net=(0.001, 0.02)), "UNDERPOWERED", "E1"),
    (dict(n_eff=61, mean_ic=0.05, mean_gross=-0.001, ci_net=(0.001, 0.02)), "INCONSISTENT", "E2"),
    (dict(n_eff=61, mean_ic=0.05, mean_gross=0.01, ci_net=(0.001, 0.02)), "PASS", "E3"),
    (dict(n_eff=61, mean_ic=-0.05, mean_gross=-0.01, ci_net=(-0.02, 0.0029)), "FAIL", "E4"),
    (dict(n_eff=61, mean_ic=0.05, mean_gross=0.01, ci_net=(-0.004, 0.010)), "UNDERPOWERED", "E5"),
    (dict(n_eff=61, mean_ic=0.05, mean_gross=0.01, ci_net=None), "UNDERPOWERED", "E5"),
    (dict(n_eff=61, mean_ic=0.05, mean_gross=0.01, ci_net=(-0.02, 0.0049), mme=0.005), "FAIL", "E4"),
    (dict(n_eff=61, mean_ic=0.05, mean_gross=0.01, ci_net=(-0.02, 0.0050), mme=0.005), "UNDERPOWERED", "E5"),  # strict inequality
    (dict(n_eff=61, mean_ic=0.0, mean_gross=-0.001, ci_net=(-0.02, 0.0029)), "FAIL", "E4"),  # IC exactly 0 is not INCONSISTENT
    (dict(n_eff=61, mean_ic=0.05, mean_gross=0.01, ci_net=(0.001, 0.02), invalid=True), "INVALID", "E0"),
])
def test_26_verdict_ladder(kw, code, rule):
    out = mod.verdict(**{"mme": 0.003, **kw})
    assert (out["code"], out["rule"]) == (code, rule) and out["flags"] == []


def test_26b_exploratory_gate_closed_keeps_underpowered_with_flag():
    base = dict(n_eff=61, mean_ic=0.05, mean_gross=0.01, ci_net=(0.001, 0.02), mme=0.003, exploratory=True)
    closed = mod.verdict(**base, gate_open=False)
    assert closed["code"] == "UNDERPOWERED" and "게이트 닫힘" in closed["flags"][0]
    assert mod.verdict(**base, gate_open=True, holm_rejected=True)["code"] == "PASS"
    assert mod.verdict(**base, gate_open=True, holm_rejected=False)["code"] == "UNDERPOWERED"


def test_27_h2():
    h = lambda **kw: mod.h2_verdict(**{"n_eff": 61, "mean_ic": -0.02, "mean_gross": -0.01, **kw})  # noqa: E731
    assert h(n_eff=19, ci_gross=(-0.02, -0.01))["code"] == "UNDERPOWERED"
    assert h(ci_gross=(-0.03, -0.001))["code"] == "SUPPORTED"
    assert h(ci_gross=(0.001, 0.03), mean_gross=0.01, mean_ic=0.02)["code"] == "REFUTED"
    assert h(ci_gross=(-0.01, 0.02))["code"] == "INCONCLUSIVE" and h(ci_gross=None)["code"] == "INCONCLUSIVE"
    assert h(ci_gross=(-0.03, -0.001), mean_ic=0.02)["flags"]  # sign disagreement warning


def test_28_h3_quintiles_and_verdict():
    qs = mod.assign_quintiles([(f"T{i:03d}", float(i), 0.0) for i in range(86)])
    assert [len(q) for q in qs] == [18, 17, 17, 17, 17]
    assert [len(q) for q in mod.assign_quintiles([(f"T{i}", float(i), 0.0) for i in range(7)])] == [2, 1, 2, 1, 1]
    # ties on score are broken by ticker code
    qs2 = mod.assign_quintiles([("B", 1.0, 0.2), ("A", 1.0, 0.1), ("C", 1.0, 0.3), ("D", 1.0, 0.4), ("E", 1.0, 0.5)])
    assert [q[0] for q in qs2] == [0.1, 0.2, 0.3, 0.4, 0.5]
    assert fm.spearman([1, 2, 3, 4, 5], [3, 1, 2, 5, 4]) == pytest.approx(0.6)
    ok_ci, ok42 = (0.001, 0.02), 0.01
    v = lambda rho, ci=ok_ci, q=ok42, n=25: mod.h3_verdict(n_eff=n, rho=rho, ci51=ci, q4_minus_q2=q)  # noqa: E731
    assert v(1.0) == "H3_SUPPORTED" and v(0.9) == "H3_SUPPORTED" and v(0.6) == "TAIL_ONLY"
    assert v(1.0, q=-0.01) == "TAIL_ONLY" and v(1.0, ci=(-0.01, 0.02)) == "NOT_SUPPORTED" and v(1.0, n=19) == "UNDERPOWERED"
    # computed end to end on 3 dates x 10 tickers with fwd increasing in the signed score
    fwd, obs = {}, {}
    for k, d in enumerate(["d1", "d2", "d3"]):
        signed = {f"T{i:02d}": float(i) for i in range(10)}
        fwd[d] = {t: (0.001 * (k + 1) * s, d, d) for t, s in signed.items()}
        obs[d] = {"signed_scores": signed}
    h = mod.h3_compute(_mini_merged({"rev_lb5": obs}, fwd), "rev_lb5", 200, 0, 25)
    assert h["verdict"] == "H3_SUPPORTED" and h["rho"] == pytest.approx(1.0) and h["diff51_ci"][0] > 0
    flat = {d: {t: (0.01, d, d) for t in o["signed_scores"]} for d, o in obs.items()}
    h2 = mod.h3_compute(_mini_merged({"rev_lb5": obs}, flat), "rev_lb5", 200, 0, 25)
    assert h2["verdict"] == "NOT_SUPPORTED" and h2["rho"] is None
    assert mod.h3_compute(_mini_merged({"rev_lb5": {"d1": {"signed_scores": {"A": 1.0, "B": 2.0}}}}, {"d1": {"A": (0.1, "d1", "d1"), "B": (0.2, "d1", "d1")}}), "rev_lb5", 50, 0, 25)["skipped_dates"] == ["d1"]


# --- 29-34 robustness ----------------------------------------------------------------------

def test_29_trimmed_mean():
    assert mod.trimmed_mean([0.05, -0.04, 0.01, 0.02, 0.00, -0.01, 0.03, 0.015]) == pytest.approx(0.01125)
    assert mod.trimmed_mean([0.1, 0.2, 0.3, 0.4]) is None


def test_30_winsorised_excess():
    uni = {f"t{i:02d}": 0.0 for i in range(20)} | {f"t{i:02d}": 0.02 for i in range(20, 40)} | {"t40": 1.0}
    book = mod.DateBook("d", {"t40": 1.0, "t39": 0.02}, uni)
    raw = mod._book_excess(book.picks, book.universe)
    assert raw == pytest.approx(0.51 - 1.4 / 41)  # 0.4758536585...
    # nearest-rank cuts of 41 sorted values: index 1 -> 0.0, index 39 -> 0.02, so the 1.0 outlier is clipped to 0.02
    assert mod.winsorized_excess([book]) == [pytest.approx(0.4 / 41)]  # 0.00975609756...


def test_31_ticker_contributions_and_leave_one_out():
    uni = {"A": 0.10, "B": -0.005, "C": -0.005, "D": 0.01} | {f"Z{i}": 0.0 for i in range(6)}
    book = mod.DateBook("d", {"A": 0.10, "B": -0.005, "C": -0.005}, uni)
    c = mod.ticker_contributions([book])
    assert c["A"] == pytest.approx(0.03) and c["B"] == pytest.approx(-0.005) and c["C"] == pytest.approx(-0.005)
    mean_ex = mod._book_excess(book.picks, book.universe)
    assert mean_ex == pytest.approx(0.02) and sum(c.values()) == pytest.approx(mean_ex)
    top3 = sorted(c.values(), reverse=True)[:3]
    assert sum(top3) / mean_ex == pytest.approx(1.0) and sum(top3) / mean_ex > mod.CONC_WARN
    loo, skipped = mod.leave_one_out_excess([book], "A")
    assert loo == [pytest.approx(-0.005)] and skipped == []
    assert loo[0] - mod.COST_RT == pytest.approx(-0.008)
    alone = mod.DateBook("e", {"A": 0.1}, {"A": 0.1, "B": 0.0})
    assert mod.leave_one_out_excess([alone], "A") == ([], ["e"])  # picks become empty: skipped and reported


def test_32_period_halves():
    r = mod.period_halves(["2026-02-27", "2026-03-01", "2026-03-05"], [0.01, 0.02, 0.03], mod.HALF_SPLIT["dev"])
    assert r["first"] == pytest.approx(0.01) and r["second"] == pytest.approx(0.025) and (r["n_first"], r["n_second"]) == (1, 2)


def test_33_zero_day_exclusion_and_realised_days():
    def fw(day, exits):
        return {t: (0.0, day, e) for t, e in zip("ABC", exits)}
    forward = {"2026-01-05": fw("2026-01-05", ["2026-01-05", "2026-01-05", "2026-01-09"]),  # 2 of 3 exit <= entry: zero-day
               "2026-01-12": fw("2026-01-12", ["2026-01-16"] * 3), "2026-01-19": fw("2026-01-19", ["2026-01-23"] * 3)}
    merged = _mini_merged({"rev_lb5": {d: {} for d in forward}}, forward)
    z = mod.zero_day_flags(merged, "rev_lb5", lambda d: False)
    assert [z[d]["zero_day"] for d in sorted(z)] == [True, False, False]
    assert z["2026-01-05"]["calendar_mismatch"] is True  # calendar says 3 trading days, data says zero
    excess = [0.0, 0.01, 0.02]  # first date is the zero-day one (0.0 return)
    kept = [e for d, e in zip(sorted(z), excess) if not z[d]["zero_day"]]
    assert sum(excess) / 3 == pytest.approx(0.01) and sum(kept) / len(kept) == pytest.approx(0.015)
    # realised trading days: weekdays in (entry, exit] that are not holidays; unknown year -> None
    assert mod.realised_trading_days(date(2026, 1, 5), date(2026, 1, 9), lambda d: False) == 4
    assert mod.realised_trading_days(date(2026, 1, 5), date(2026, 1, 9), lambda d: d == date(2026, 1, 7)) == 3
    assert mod.realised_trading_days(date(2026, 1, 5), date(2026, 1, 9), lambda d: None) is None


def test_33b_real_holiday_table_christmas():
    import importlib
    try:
        real = importlib.import_module("app.live.kr_holidays")._KR_MARKET_HOLIDAYS
    except Exception:  # noqa: BLE001
        pytest.skip("kr_holidays not importable here")
    assert 2025 in real
    n = mod.realised_trading_days(date(2025, 12, 24), date(2025, 12, 28), lookup=_real_lookup)
    assert n == 1  # 12-25 holiday, 12-26 Friday, weekend
    assert mod.realised_trading_days(date(2023, 1, 2), date(2023, 1, 4), lookup=_real_lookup) is None  # year not in the table


def _real_lookup(day):
    from app.live import kr_holidays as kh
    return None if day.year not in kh._KR_MARKET_HOLIDAYS else bool(kh.is_kr_market_holiday(day))


def test_34_moving_block_ci():
    assert mod.moving_block_ci([0.01] * 4, 100, 0) is None
    assert mod.moving_block_ci([0.01] * 10, 200, 0) == (0.01, 0.01)
    vals = [0.01, 0.02, 0.015, 0.03, 0.005, 0.02, 0.01, 0.025]
    ci = mod.moving_block_ci(vals, 400, 0)
    # independent replica of the documented procedure: k=ceil(8/3)=3 blocks of 3, 6 start points, truncated to n
    rng, means = random.Random(0), []
    for _ in range(400):
        pooled = []
        for _ in range(3):
            s = rng.randrange(6)
            pooled += vals[s:s + 3]
        means.append(statistics.mean(pooled[:8]))
    means.sort()
    assert ci == (fm.percentile(means, 2.5), fm.percentile(means, 97.5)) and ci[0] > 0
    assert mod.moving_block_ci([-0.01, 0.02, -0.03, 0.0, 0.01], 200, 0)[0] < 0


def test_34b_ladder_and_stop_rules():
    kw = dict(ic_ci=(0.01, 0.05), h3_code="H3_SUPPORTED", mean_net=0.001, q5_net=0.002, first_half_net=0.001)
    lad = mod.evidence_ladder(primary_code="UNDERPOWERED", **kw)
    assert lad["level_B"] and not lad["level_A"]
    assert not mod.evidence_ladder(primary_code="UNDERPOWERED", **{**kw, "first_half_net": -0.001})["level_B"]
    assert not mod.evidence_ladder(primary_code="INCONSISTENT", **kw)["level_B"]
    s = lambda **k: mod.stop_rule(**{"role": "dev", "level_b": False, "mean_ic": -0.05, **k})["code"]  # noqa: E731
    assert s(primary_code="FAIL") == "STOP-1" and s(primary_code="INCONSISTENT") == "STOP-1"
    assert s(primary_code="UNDERPOWERED") == "STOP-2" and s(primary_code="UNDERPOWERED", level_b=True) == "OPEN_OK"
    assert s(primary_code="PASS") == "OPEN_OK" and s(primary_code="INVALID") == "NOT_EVALUATED"
    assert s(role="holdout", primary_code="PASS", mean_ic=-0.01) == "STOP-3" and s(role="holdout", primary_code="FAIL", mean_ic=0.05) == "STOP-3"
    assert s(role="holdout", primary_code="PASS", mean_ic=0.05) == "NONE"
    assert "no changes" in mod.stop_rule(role="holdout", primary_code="FAIL", level_b=False, mean_ic=0.1)["text"]


def test_34c_scan_log_attribution(tmp_path):
    p = tmp_path / "x.txt"
    p.write_text("\n".join([
        "Sweeping ...", "KIS get_price_series failed for A.KS, falling back", "[1/2] 2025-09-01 scored=86 priced=86 bench=ok",
        "kis GET_PRICE_SERIES FAILED for B.KS, x", "Traceback", "[2/2] 2025-09-05 scored=85 priced=86 bench=ok",
        "KIS get_price_series failed for C.KS, y", "x ReadTimeout y", "EGW00201 twice", "price series unavailable for Z"]), encoding="utf-8")
    s = mod.scan_log(p)
    assert s["counts"]["fallback_price"] == 3 and s["counts"]["read_timeout"] == 1 and s["counts"]["egw00201"] == 1 and s["counts"]["price_unavailable"] == 1
    assert s["n_progress"] == 2 and [(f["ticker"], f["at"]) for f in s["fallbacks"]] == [("A.KS", "2025-09-01"), ("B.KS", "2025-09-05"), ("C.KS", "phaseB")]


# --- 35-37 assembly, rendering, CLI -------------------------------------------------------

def test_35_underpowered_wording_and_check_mode_has_no_numbers(tmp_path, capsys):
    G = _make_grid(tmp_path)
    assert _cli(G, "check") == 0
    check_text = (G.out / "dev_h4_c4_check.txt").read_text(encoding="utf-8")
    assert not re.search(r"[+-]\d+\.\d+%", check_text) and "T5 chunks" in check_text
    assert _pipeline(G) == 0
    text = (G.out / "dev_h4_c4_report.txt").read_text(encoding="utf-8")
    block = text.split("== Verdict block", 1)[1].split("T1 metrics", 1)[0]
    assert block.count("UNDERPOWERED, 8 dates") >= 9  # H1, 5 exploratory, H2, H3, robustness, ladder, STOP
    for w in BANNED:
        assert w not in block, w
    assert mod.FOOTER in text and "alpha_return" in text
    j = _json(G, "report")
    assert j["variants"]["rev_lb5"]["verdict"]["code"] == "UNDERPOWERED" and j["status"] == "ok"
    assert j["ladder"] is not None and j["stop"] is not None


def test_36_full_24_date_h10_grid_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(mod, "MIN_DATES", {4: 55, 10: 24})
    ds = [(date(2025, 9, 1) + timedelta(days=14 * i)).isoformat() for i in range(24)]
    G = _make_grid(tmp_path, grid="dev_h10_c10", chunk_dates=[ds[0:8], ds[8:16], ds[16:24]], holding=10, cadence=10, n_tickers=12)
    assert _cli(G, "report") == 2  # report needs the frozen accepted list
    assert _cli(G, "check") == 0 and _cli(G, "freeze") == 0
    acc = json.loads((G.out / "accepted_dev_h10_c10.json").read_text(encoding="utf-8"))
    assert [c["chunk"] for c in acc["chunks"]] == G.names and all(len(c["sha256"]) == 64 for c in acc["chunks"])
    assert _cli(G, "report") == 0
    raw = (G.out / "dev_h10_c10_report.json").read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf")
    j = json.loads(raw, parse_constant=lambda c: pytest.fail(f"non-strict JSON constant {c}"))
    assert (G.out / "dev_h10_c10_report.txt").read_bytes()[:3] != b"\xef\xbb\xbf"
    assert j["status"] == "ok" and j["variants"]["rev_lb5"]["n_eff"] == 24 and j["ladder"] is None and j["stop"] is None  # auxiliary horizon
    assert j["variants"]["rev_lb5"]["verdict"]["code"] in {"PASS", "FAIL", "INCONSISTENT", "UNDERPOWERED"}
    assert all(i in {c["id"] for c in j["checks"]} for i in ("S1", "F1")) and all(c["ok"] for c in j["checks"])
    assert len(j["per_date"]) == 24 and j["robustness"]["moving_block_ci"]["sensitivity_only"] is True
    text = (G.out / "dev_h10_c10_report.txt").read_text(encoding="utf-8")
    assert "AUXILIARY horizon" in text and "minimum meaningful effect=+0.50%" in text
    # report mean in T6 equals the T1 gross mean
    pd = [r["excess"] for r in j["per_date"]]
    assert sum(pd) / len(pd) == pytest.approx(j["variants"]["rev_lb5"]["gross"]["mean"])
    assert [v for v in j["variants"]["rev_lb5"]["net"]] == ["0.003", "0.005", "0.008"]
    # a changed dump after freeze is rejected
    p = G.base / "chunks" / f"{G.names[0]}.json"
    p.write_bytes(p.read_bytes() + b"\n")
    assert _cli(G, "report") == 4 and _bad(G, "report") == ["F1"]


def test_37_main_exit_codes(tmp_path, capsys):
    G = _make_grid(tmp_path / "g")
    assert _cli(G, "check") == 0
    assert mod.main(["--grid", "weird_grid", "--mode", "check", "--base-dir", str(G.base), "--out-dir", str(G.out)]) == 2
    assert mod.main(["--grid", "dev_h4_c4", "--mode", "check", "--base-dir", str(tmp_path / "nowhere"), "--out-dir", str(G.out)]) == 2
    assert mod.main(["--grid", "dev_h4_c4", "--mode", "nonsense"]) == 2
    assert mod.main(["--grid", "dev_h99_c9", "--mode", "check", "--base-dir", str(G.base), "--out-dir", str(G.out)]) == 2  # grid not in the plan
    assert _cli(_make_grid(tmp_path / "h", fallbacks=[9, 0]), "check") == 3
    G4 = _make_grid(tmp_path / "i", mutate=lambda c: c.dumps["dev_h4_c4_c01"].update(schema="x"))
    assert _cli(G4, "check") == 4
    out = capsys.readouterr()
    assert "input error" in out.err


def test_37b_holdout_role_runs_stop3_only(tmp_path):
    ds = [(date(2024, 9, 2) + timedelta(days=7 * i)).isoformat() for i in range(6)]
    G = _make_grid(tmp_path, grid="val_h4_c4", chunk_dates=[ds[:3], ds[3:]])
    assert _pipeline(G) == 0
    j = _json(G, "report")
    assert j["role"] == "holdout" and set(j["variants"]) == set(mod.HOLDOUT_LABELS)
    assert j["robustness"] is None and j["ladder"] is None and j["stop"] is not None
    assert "STOP rule: UNDERPOWERED, 6 dates" in (G.out / "val_h4_c4_report.txt").read_text(encoding="utf-8")


# --- review round 2 additions -------------------------------------------------------------

def _load_merged(G):
    plan = mod.load_plan(G.base, G.grid)
    _, _, merged, _ = mod.validate(G.base, G.grid, mod.role_of(G.grid), plan, "check", {})
    return merged


def test_33c_zero_day_exclusion_through_real_robustness(tmp_path):
    G = _make_grid(tmp_path, chunk_dates=[DEFAULT_CHUNKS[0] + DEFAULT_CHUNKS[1]], zero_dates={"2025-09-11"})
    merged = _load_merged(G)
    stats = mod.variant_stats(merged, "rev_lb5", 200, 0)
    h3 = mod.h3_compute(merged, "rev_lb5", 200, 0, stats["n_eff"])
    rob = mod.robustness(merged, stats, h3, 200, 0, "dev", lambda d: False)
    assert rob["zero_day"]["n_zero_days"] == 1 and rob["zero_day"]["calendar_mismatch"] == ["2025-09-11"]
    ex = dict(zip(stats["ex_days"], stats["excess"]))
    assert ex["2025-09-11"] == 0.0  # every ticker returns 0 on the zero-day date
    others = [v for d, v in ex.items() if d != "2025-09-11"]
    assert rob["zero_day"]["net_mean_excl"] == pytest.approx(sum(others) / len(others) - mod.COST_RT)
    assert rob["zero_day"]["by_date"]["2025-09-11"]["zero_day"] is True


def test_38_boundary_disagree_ci_rule_vs_p_boot(monkeypatch):
    # B=10,000 with exactly 250 means <= 0: percentile index 250 is positive (CI lower bound > 0) but p = 251/10001 > 0.025
    means = [-1.0] * 250 + [0.5] * 9750
    ci = mod.ci_from_means(means)
    p = mod.p_boot_one_sided(means)
    assert ci[0] == 0.5 and p == pytest.approx(251 / 10001) and p > mod.ALPHA_1S
    assert mod.boundary_disagree(ci, p) is True
    ci249, p249 = mod.ci_from_means([-1.0] * 249 + [0.5] * 9751), mod.p_boot_one_sided([-1.0] * 249 + [0.5] * 9751)
    assert ci249[0] == 0.5 and p249 == pytest.approx(250 / 10001) and p249 <= mod.ALPHA_1S and mod.boundary_disagree(ci249, p249) is False
    ci251, p251 = mod.ci_from_means([-1.0] * 251 + [0.5] * 9749), mod.p_boot_one_sided([-1.0] * 251 + [0.5] * 9749)
    assert ci251[0] == -1.0 and mod.boundary_disagree(ci251, p251) is False
    assert mod.boundary_disagree(None, 0.01) is False and mod.boundary_disagree(ci, None) is False
    # through variant_stats: inject the crafted bootstrap means for every series
    monkeypatch.setattr(mod, "bootstrap_means", lambda values, n, seed: list(means))
    obs = {f"d{i}": {"excess_return": 0.01 * i, "rank_ic": 0.1, "quantile_spread": 0.0, "pick_forward_returns": {}, "residual_fallbacks": 0, "picks": ["A"]} for i in range(5)}
    merged = _mini_merged({"rev_lb5": obs}, {d: {} for d in obs})
    n30 = mod.variant_stats(merged, "rev_lb5", 10_000, 0)["net"][mod.COST_RT]
    assert n30["boundary_disagree"] is True and n30["p_boot"] == pytest.approx(251 / 10001)
    note = mod.boundary_flag(n30)
    assert "CI lower bound > 0" in note and "p_boot=0.025097 > 0.025" in note and "CI-rule-based" in note
    assert mod.boundary_flag({"boundary_disagree": False}) is None


def test_39_cost_crosscheck_both_ways(tmp_path, monkeypatch):
    u = ["A.KS", "B.KS"]
    ok = mod.cost_crosscheck(u)
    assert ok.ok and ok.severity == "warning" and ok.id == "C1"
    monkeypatch.setattr(mod, "_load_cost_fn", lambda: (lambda t, side: 5.0 if side == "buy" else 30.0))  # 35 bps != 30
    bad = mod.cost_crosscheck(u)
    assert not bad.ok and bad.severity == "warning" and "differs" in bad.detail
    G = _make_grid(tmp_path)
    assert _cli(G, "check") == 0 and _checks(G)["C1"]["ok"] is False  # a warning never changes the exit code

    def boom():
        raise RuntimeError("settings unreadable")
    monkeypatch.setattr(mod, "_load_cost_fn", boom)
    unavailable = mod.cost_crosscheck(u)
    assert not unavailable.ok and unavailable.severity == "warning" and "unavailable" in unavailable.detail
    assert _cli(G, "check") == 0


def test_41_variant_entry_missing_summary_is_integrity_failure(tmp_path):
    def m(c):
        del c.dumps["dev_h4_c4_c01"]["variants"][0]["summary"]
    G = _make_grid(tmp_path, mutate=m)
    assert _cli(G, "check") == 4 and _bad(G) == ["I5"]


def test_42_file_format_errors_map_to_exit_codes(tmp_path):
    def run_with(sub, mutate=None, mode="check"):
        G = _make_grid(tmp_path / sub, mutate=mutate)
        return G, _cli(G, mode)

    G, _ = run_with("a")
    (G.base / "chunks.json").write_text("{not json", encoding="utf-8")
    assert _cli(G, "check") == 2
    (G.base / "chunks.json").write_text(json.dumps({"grids": {"dev_h4_c4": {"holding": 4}}}), encoding="utf-8")
    assert _cli(G, "check") == 2  # grid entry lacks chunks/all_dates/cadence
    G2 = _make_grid(tmp_path / "b")
    with open(G2.base / "runlog.jsonl", "ab") as fh:
        fh.write(b"{broken line\n")
    assert _cli(G2, "check") == 2
    G3 = _make_grid(tmp_path / "c", mutate=lambda c: c.runlog[0].pop("start_kst"))
    assert _cli(G3, "check") == 4 and "I9" in _bad(G3)  # unusable record is never a match
    G4 = _make_grid(tmp_path / "d")
    (G4.base / "chunks" / "dev_h4_c4_c01.json").write_text("{truncated", encoding="utf-8")
    assert _cli(G4, "check") == 4 and _bad(G4) == ["I3"]
    G5 = _make_grid(tmp_path / "e")
    assert _cli(G5, "freeze") == 0
    mp = G5.out / "accepted_dev_h4_c4.json"
    mp.write_text("[1, 2", encoding="utf-8")
    assert _cli(G5, "report") == 2
    mp.write_text(json.dumps({"schema": mod.ACCEPTED_SCHEMA}), encoding="utf-8")
    assert _cli(G5, "report") == 2
    assert _cli(G5, "freeze") == 2  # a malformed existing list is not silently overwritten


def test_43_swapped_txt_with_matching_counts_is_rejected(tmp_path):
    def other_date(c):
        c.txts["dev_h4_c4_c01"] = c.txts["dev_h4_c4_c01"].replace("[1/4] 2025-09-01", "[1/4] 2025-09-02")
    G = _make_grid(tmp_path / "a", mutate=other_date)
    assert _cli(G, "check") == 4 and _bad(G) == ["I10"] and "progress dates" in _checks(G)["I10"]["detail"]

    def other_priced(c):
        c.txts["dev_h4_c4_c01"] = c.txts["dev_h4_c4_c01"].replace("[2/4] 2025-09-05 scored=10 priced=10", "[2/4] 2025-09-05 scored=10 priced=9")
    G2 = _make_grid(tmp_path / "b", mutate=other_priced)
    assert _cli(G2, "check") == 4 and _bad(G2) == ["I10"] and "scored/priced" in _checks(G2)["I10"]["detail"]


def test_44_freeze_never_overwrites_a_different_list_and_f1_covers_txt_and_runlog(tmp_path):
    G = _make_grid(tmp_path)
    assert _cli(G, "freeze") == 0
    mp = G.out / "accepted_dev_h4_c4.json"
    first = mp.read_bytes()
    assert _cli(G, "freeze") == 0 and mp.read_bytes() == first  # identical set: left untouched
    entry = json.loads(first)["chunks"][0]
    assert len(entry["txt_sha256"]) == 64 and len(entry["runlog_sha256"]) == 64
    # a different (still valid) dump: freeze refuses to replace the list
    p = G.base / "chunks" / f"{G.names[0]}.json"
    p.write_bytes(p.read_bytes() + b"\n")
    assert _cli(G, "freeze") == 3 and _bad(G, "freeze") == ["F0"] and mp.read_bytes() == first
    p.write_bytes(p.read_bytes()[:-1])  # restore
    assert _cli(G, "report") == 0
    # changed .txt (counts unchanged) after freezing
    t = G.base / "chunks" / f"{G.names[1]}.txt"
    t.write_bytes(t.read_bytes() + b"harmless extra line\n")
    assert _cli(G, "report") == 4 and _bad(G, "report") == ["F1"]
    t.write_bytes(t.read_bytes()[: -len(b"harmless extra line\n")])
    assert _cli(G, "report") == 0
    # changed runlog record (still linked) after freezing
    log = (G.base / "runlog.jsonl").read_bytes().decode("utf-8-sig").splitlines()
    rec = json.loads(log[0])
    rec["minutes"] = 31.0
    (G.base / "runlog.jsonl").write_bytes(("\n".join([json.dumps(rec), *log[1:]]) + "\n").encode("utf-8"))
    assert _cli(G, "report") == 4 and _bad(G, "report") == ["F1"]


def test_45_holm_note_for_p_only_rejection():
    assert mod._holm_note({"rejected": True}, "INCONSISTENT") == " (p-only rejection; verdict INCONSISTENT)"
    assert mod._holm_note({"rejected": True}, "PASS") == "" and mod._holm_note({"rejected": False}, "INCONSISTENT") == ""


def test_46_strong_reversal_end_to_end_pass_gate_robust(tmp_path):
    ds = [(date(2025, 9, 1) + timedelta(days=4 * i)).isoformat() for i in range(60)]
    G = _make_grid(tmp_path, chunk_dates=[ds[0:20], ds[20:40], ds[40:60]], signal=0.003, n_boot=500)
    assert _pipeline(G, n_boot=500) == 0
    j = _json(G, "report")
    v = j["variants"]
    assert v["rev_lb5"]["n_eff"] == 60
    assert (v["rev_lb5"]["verdict"]["code"], v["rev_lb5"]["verdict"]["rule"]) == ("PASS", "E3")
    assert v["rev_lb5"]["net"]["0.003"]["ci"][0] > 0 and v["rev_lb5"]["rank_ic"]["mean"] > 0
    assert j["ladder"]["level_A"] is True and j["stop"]["code"] == "OPEN_OK"
    assert j["robustness"]["robust_pass"] is True and j["h3"]["rev_lb5"]["verdict"] in {"H3_SUPPORTED", "TAIL_ONLY"}
    # Holm: gate open, rejections are a prefix of the ascending-p order and obey p * (m - rank + 1) <= alpha
    holm = {k: x["holm"] for k, x in v.items() if x["holm"]}
    assert set(holm) == set(mod.EXPLORATORY)
    ordered = sorted(holm.values(), key=lambda h: h["rank"])
    assert [h["p"] for h in ordered] == sorted(h["p"] for h in ordered)
    flags = [h["rejected"] for h in ordered]
    assert flags == sorted(flags, reverse=True) and any(flags)  # a prefix of True then False
    for h in ordered:
        if h["rejected"]:
            assert h["p"] * (len(mod.EXPLORATORY) - h["rank"] + 1) <= mod.ALPHA_1S + 1e-12
    for k, h in holm.items():
        if h["rejected"]:
            assert v[k]["verdict"]["code"] in {"PASS", "INCONSISTENT"}
            if v[k]["verdict"]["code"] == "PASS":
                assert v[k]["verdict"]["rule"] == "E3+Holm"
        else:
            assert v[k]["verdict"]["code"] != "PASS"
    text = (G.out / "dev_h4_c4_report.txt").read_text(encoding="utf-8")
    assert re.search(r"p_boot=[+-]\d\.\d{6} ", text) and re.search(r"\[[+-]\d+\.\d{4}%,[+-]\d+\.\d{4}%\]", text)  # boundary-visible precision
    assert "git_head_of_T2" in j
