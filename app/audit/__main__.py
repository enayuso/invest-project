"""Self-audit CLI — hit-rate-by-rating report over resolved decisions.

    python -m app.audit report
    python -m app.audit report --ticker AAPL
    python -m app.audit report --market domestic
    python -m app.audit report --market overseas
    python -m app.audit report --scope live
    python -m app.audit report --scope backtest
    python -m app.audit report --account-mode paper
    python -m app.audit report --account-mode real

    python -m app.audit history --ticker AAPL --date 2026-04-01

    python -m app.audit experiments
    python -m app.audit experiments --limit 50

    python -m app.audit checkpoints

    python -m app.audit debate-impact
    python -m app.audit debate-impact --market domestic
    python -m app.audit debate-impact --scope live
    python -m app.audit debate-impact --account-mode paper

    python -m app.audit confidence
    python -m app.audit confidence --ticker AAPL
    python -m app.audit confidence --basis horizon

    python -m app.audit report --basis horizon

    python -m app.audit screen-validation
    python -m app.audit screen-validation --start-date 2026-08-01

    python -m app.audit slippage
    python -m app.audit slippage --start-date 2026-06-01 --end-date 2026-08-13

    python -m app.audit reconcile
    python -m app.audit reconcile --days 90

    python -m app.audit weekly
    python -m app.audit weekly --week 2026-W33
    python -m app.audit weekly --scope live --account-mode paper
    python -m app.audit weekly --out reports/weekly/

    python -m app.audit report --trigger auto
    python -m app.audit report --trigger manual
    python -m app.audit weekly --trigger manual

    python -m app.audit rating-action-gap
    python -m app.audit rating-action-gap --scope live --account-mode real --days 1
    python -m app.audit rating-action-gap --market domestic --limit 100

    python -m app.audit screening-ic --market domestic
    python -m app.audit screening-ic --market domestic --weeks 8
    python -m app.audit screening-ic --market overseas --holding-days 5
    python -m app.audit screening-ic --market domestic --account-mode real --start-date 2026-09-10 --end-date 2026-09-10

    python -m app.audit guard-alpha --scope live --account-mode real
    python -m app.audit guard-alpha --scope live --account-mode real --basis horizon

    python -m app.audit rating-shift --boundary 2026-09-16T04:53:16Z
    python -m app.audit rating-shift --boundary 2026-09-16T04:53:16Z --market domestic --scope live
    python -m app.audit rating-shift --boundary 2026-09-16T04:53:16Z --account-mode real

    python -m app.audit entry-paired --market domestic --scope live --account-mode paper --basis horizon
    python -m app.audit entry-paired --market domestic --scope live --account-mode real --recompute-pending
    python -m app.audit entry-version-strat --market domestic --scope live --account-mode paper --basis horizon

    python -m app.audit seat-stance --market domestic --scope live --account-mode real
    python -m app.audit seat-stance --ticker 036540.KS --start-date 2026-09-21 --end-date 2026-09-21

    python -m app.audit pending-health --market domestic --account-mode real
    python -m app.audit pending-health --market domestic --account-mode real --probe

    python -m app.audit rm-leak --since 2026-09-01 --until 2026-10-02
    python -m app.audit rm-leak --since 2026-10-02 --after-ts 2026-10-02T10:18:07+09:00 --show 5
    python -m app.audit rm-leak --since 2026-10-01 --account-mode real --json

    python -m app.audit stage-attribution --market domestic --account-mode real
    python -m app.audit stage-attribution --market domestic --account-mode real --start-date 2026-09-18 --recompute-pending
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import date, timedelta
from pathlib import Path

from app.audit.stats import (
    compute_confidence_calibration,
    compute_debate_impact,
    compute_hit_rate_by_rating,
    compute_rating_action_gap,
    get_decision_history,
    render,
    render_confidence_calibration,
    render_debate_impact,
    render_history,
    render_rating_action_gap,
)
from app.audit.entry_paired import (
    compute_code_version_stratification,
    compute_entry_paired_alpha,
    compute_entry_paired_alpha_with_recompute,
    render_code_version_stratification,
    render_entry_paired_alpha,
    render_entry_paired_recompute,
)
from app.audit.guard_alpha import (
    compute_guard_alpha,
    compute_guard_alpha_with_recompute,
    render_guard_alpha,
    render_guard_alpha_recompute,
)
from app.audit.pending_health import (
    DEFAULT_WARN_PCT,
    DEFAULT_WEEKS,
    compute_pending_health,
    render_pending_health,
)
from app.audit.rm_leak import compute_rm_leak, render_rm_leak, report_to_json
from app.audit.rating_shift import compute_rating_shift, parse_boundary, render_rating_shift
from app.audit.reconciliation_report import compute_reconciliation, render_reconciliation
from app.audit.screening_ic import compute_screening_ic, render_screening_ic
from app.audit.seat_stance import compute_seat_stance, render_seat_stance
from app.audit.slippage_validation import compute_slippage, render_slippage
from app.audit.stage_attribution import (
    compute_stage_attribution,
    compute_stage_attribution_with_recompute,
    render_stage_attribution,
    render_stage_attribution_recompute,
)
from app.audit.tiered_screen_validation import render_screen_validation, simulate_tiered_screen
from app.audit.weekly import compute_weekly_report, parse_week_label, render_weekly_report, week_label
from app.backtest.checkpoint import list_checkpoints, render_checkpoints
from app.backtest.experiments import list_experiments, render_experiments

if (sys.stdout.encoding or "").lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hit-rate-by-rating audit over resolved decisions.")
    parser.add_argument(
        "command",
        choices=[
            "report", "history", "experiments", "checkpoints", "debate-impact", "confidence",
            "screen-validation", "slippage", "reconcile", "weekly", "rating-action-gap", "screening-ic",
            "guard-alpha", "rating-shift", "entry-paired", "entry-version-strat", "seat-stance",
            "pending-health", "rm-leak", "stage-attribution",
        ],
    )
    parser.add_argument(
        "--ticker", default=None,
        help="report/debate-impact/confidence/guard-alpha/seat-stance: limit to one ticker. history: required.",
    )
    parser.add_argument(
        "--market", default=None, choices=["domestic", "overseas"],
        help="report/debate-impact/guard-alpha/rating-shift/entry-paired/entry-version-strat/seat-stance/"
        "pending-health: limit to KRX "
        "(.KS) or non-KRX tickers, instead of pooling both (domestic/overseas separation audit, 2026-07-27 — "
        "see docs/domestic-overseas-separation-plan.md). Mutually exclusive with --ticker (a single ticker "
        "already implies its own market). pending-health's --probe additionally REQUIRES domestic.",
    )
    parser.add_argument(
        "--scope", default=None, choices=["live", "backtest"],
        help="report/debate-impact/guard-alpha/rating-shift/entry-paired/entry-version-strat/seat-stance: limit to "
        "live-resolved or backtest-resolved decisions, instead of pooling both (2026-08-03, once "
        "app.memory.auto_resolve started auto-resolving live decisions every cycle, pooling silently blends "
        "two differently-sourced populations). pending-health is always live-only (the production resolver "
        "never touches backtest rows) and rejects this flag.",
    )
    parser.add_argument(
        "--account-mode", default=None, choices=["paper", "real"], dest="account_mode",
        help="report/debate-impact/guard-alpha/rating-shift/entry-paired/entry-version-strat/seat-stance/"
        "pending-health: limit to "
        "paper-account or real-account decisions, instead of pooling both (2026-08-03 — see project memory: "
        "user-account-mode-db-separation-design).",
    )
    parser.add_argument(
        "--boundary", default=None,
        help="rating-shift: required, ISO8601 datetime (e.g. 2026-09-16T04:53:16Z) — the split point "
        "between the 'before' and 'after' cohorts, compared against memory_entries.created_at.",
    )
    parser.add_argument(
        "--trigger", default=None, choices=["auto", "manual"],
        help="report/confidence/debate-impact/weekly/guard-alpha: restrict to auto-triggered (scheduled "
        "loop) or manually-triggered (a human picked the ticker) decisions. Omit to pool both for "
        "report/confidence/debate-impact — a NOTE is printed when pooled. weekly and guard-alpha default "
        "to 'auto' regardless (2026-08-20 — see compute_weekly_report's own docstring; guard-alpha follows "
        "the same precedent since a hand-picked ticker would bias the violated-vs-clean cohort comparison).",
    )
    parser.add_argument("--date", default=None, help="history: required, YYYY-MM-DD — which (ticker, date) to show.")
    parser.add_argument("--limit", type=int, default=20, help="experiments: how many recent runs to show.")
    parser.add_argument(
        "--start-date", default=None, dest="start_date",
        help="screen-validation: earliest trade_date to include (YYYY-MM-DD, default 2026-08-01). "
        "slippage: earliest trade_date (default 90 days back, KIS's own TR cap). "
        "screening-ic: earliest trade_date (default: --weeks back from --end-date/today) — use this "
        "instead of --weeks to pin an exact range, e.g. isolating a single day or starting from a "
        "known deploy date so a settings/observational-scoring change mid-range doesn't get pooled in. "
        "guard-alpha/entry-paired/entry-version-strat: earliest horizon_eval_date to include (default: no "
        "lower bound). NOTE: horizon_eval_date is NULL until a row's fixed horizon matures, so a recent "
        "--start-date can return zero rows even with plenty of scorable data — that cohort just hasn't "
        "matured yet. seat-stance: earliest trade_date to include (default: no lower bound) — unlike "
        "guard-alpha this filters on trade_date, not horizon_eval_date, since seat-stance doesn't depend "
        "on forward-return maturation.",
    )
    parser.add_argument(
        "--end-date", default=None, dest="end_date",
        help="slippage: latest trade_date to include (YYYY-MM-DD, default today). "
        "screening-ic: latest trade_date (default today). "
        "guard-alpha/entry-paired/entry-version-strat: latest horizon_eval_date to include (default: no "
        "upper bound). Same NULL-until-matured caveat as --start-date above applies. "
        "seat-stance: latest trade_date to include (default: no upper bound).",
    )
    parser.add_argument(
        "--days", type=int, default=30,
        help="reconcile: how many days back to check ledger trades against KIS execution history "
        "(default 30, capped to 90 — KIS's own TR lookback limit). "
        "rating-action-gap: how many days of trade_date to scan (default 30).",
    )
    parser.add_argument(
        "--basis", default=None, choices=["variable", "horizon"],
        help="report/confidence/weekly/guard-alpha/entry-paired/entry-version-strat/stage-attribution: "
        "'variable' (default; stage-attribution defaults to 'horizon') "
        "uses the existing variable-window alpha_return/raw_return; 'horizon' uses the "
        "fixed-trading-day-window horizon_alpha_return/horizon_raw_return (Stage 1, 2026-08-13 — see "
        "app.memory.db's horizon_* column comments).",
    )
    parser.add_argument(
        "--week", default=None,
        help="weekly: ISO week to report on, 'YYYY-Www' (e.g. '2026-W33') — default this week (KST).",
    )
    parser.add_argument(
        "--out", default=None,
        help="weekly: directory to also save the report to, as '<dir>/YYYY-Www.md' (created if missing). "
        "Always printed to stdout regardless of this flag.",
    )
    parser.add_argument(
        "--weeks", type=int, default=None,
        help="screening-ic: how many weeks back from --end-date/today to report on (default 4) — "
        "ignored if --start-date is given. pending-health: how many ISO weeks of trade_date to show, "
        "counting the current (partial) week (default 8).",
    )
    parser.add_argument(
        "--warn-pct", type=float, default=None, dest="warn_pct",
        help="pending-health only: print '!! WARNING' when a (week, market, account_mode) cell's "
        "overdue-pending share of eligible rows (trade_date <= local_today(market) - "
        "memory_min_holding_days) is >= this percent (default 15.0 — 2026-09 real-account baseline "
        "weeks ran 5.6-5.9%%, the KOSDAQ '.KS' stall week hit 36%%). Cells with fewer than 30 eligible "
        "rows are shown but never warned on.",
    )
    parser.add_argument(
        "--probe", action="store_true",
        help="pending-health only, domestic only: also classify each overdue pending row via yfinance "
        "(stored symbol, then the '.KS'<->'.KQ' alternate) as resolvable_via_alt_suffix / "
        "resolvable_under_stored_symbol / not_matured / no_entry_close / price_unavailable / "
        "fetch_error:*. Read-only: no DB write, no reflection, ignores memory_krx_fallback_since "
        "(that gate gets checked against, not applied by, this diagnostic). Run after 15:30 KST for "
        "settled closes.",
    )
    parser.add_argument(
        "--holding-days", type=int, default=None, dest="holding_days",
        help="screening-ic: forward-return holding window in calendar days "
        "(default settings.screening_forward_holding_days).",
    )
    parser.add_argument(
        "--recompute-pending", action="store_true", dest="recompute_pending",
        help="entry-paired/guard-alpha only, domestic + live only: also re-measure pending decisions past "
        "their holding window IN MEMORY (KOSDAQ names stored as '.KS' never resolve in production because "
        "Yahoo only serves them as '.KQ'), and print baseline (resolved only) next to resolved+recomputed. "
        "guard-alpha rebuilds the WHOLE report (all keys recomputed together, since Benjamini-Hochberg q "
        "depends on every tested key's p-value, not just the ones that gained rows) -- roughly 2x the "
        "plain guard-alpha runtime (permutation testing runs twice), plus the price-fetch time for "
        "pending tickers; ~1-2 minutes is normal. Read-only: no DB write, no reflection. Cite the "
        "baseline; the recomputed column is a sensitivity check. Cannot be combined with "
        "--start-date/--end-date.",
    )
    parser.add_argument("--since", default=None, help="rm-leak: required, earliest trade_date (YYYY-MM-DD).")
    parser.add_argument(
        "--until", default=None,
        help="rm-leak: latest trade_date, inclusive (YYYY-MM-DD, default no upper bound).",
    )
    parser.add_argument(
        "--after-ts", default=None, dest="after_ts",
        help="rm-leak: ISO8601 datetime; also report rows with created_at at/after it separately "
        "(e.g. after a restart). Naive values are taken as UTC.",
    )
    parser.add_argument(
        "--show", type=int, default=0,
        help="rm-leak: print the first N contaminated rows' rationale head (80 chars). Default 0.",
    )
    parser.add_argument("--json", action="store_true", help="rm-leak: emit JSON instead of the text table.")
    parser.add_argument(
        "--era-boundary", default=None, dest="era_boundary",
        help="stage-attribution only: ISO8601 datetime; also report each contrast's delta before/after it "
        "(compared against created_at) and flag ERA_HETEROGENEOUS when the signs differ.",
    )
    parser.add_argument(
        "--show-underpowered", action="store_true", dest="show_underpowered",
        help="stage-attribution only: reveal the numbers of UNDERPOWERED rows (masked by default), each "
        "prefixed PROVISIONAL.",
    )
    args = parser.parse_args()
    if args.basis is None:
        # Only stage-attribution defaults to the fixed-horizon basis; every other command keeps 'variable'.
        args.basis = "horizon" if args.command == "stage-attribution" else "variable"
    if args.command == "stage-attribution":
        if not args.market:
            parser.error("stage-attribution requires --market domestic|overseas (never pooled)")
        if not args.account_mode:
            parser.error("stage-attribution requires --account-mode paper|real (never pooled)")
        if args.scope == "backtest":
            parser.error("stage-attribution is live-only")
        if args.scope is None:
            args.scope = "live"
        if args.ticker:
            parser.error("stage-attribution does not take --ticker")
        for flag in ("start_date", "end_date"):
            if getattr(args, flag):
                try:
                    date.fromisoformat(getattr(args, flag))
                except ValueError:
                    parser.error(f"--{flag.replace('_', '-')} must be YYYY-MM-DD")
        if args.era_boundary:
            try:
                parse_boundary(args.era_boundary)
            except ValueError:
                parser.error("--era-boundary must be an ISO8601 datetime")
    elif args.era_boundary or args.show_underpowered:
        parser.error("--era-boundary/--show-underpowered apply to stage-attribution only")
    if args.command == "rm-leak":
        if not args.since:
            parser.error("rm-leak requires --since YYYY-MM-DD")
        if args.ticker:
            parser.error("rm-leak does not take --ticker (use --market)")
        if args.show < 0:
            parser.error("--show must be >= 0")
    elif args.since or args.until or args.after_ts or args.show or args.json:
        parser.error("--since/--until/--after-ts/--show/--json apply to rm-leak only")
    if args.recompute_pending:
        if args.command not in ("entry-paired", "guard-alpha", "stage-attribution"):
            parser.error("--recompute-pending applies to entry-paired, guard-alpha and stage-attribution only")
        if args.market != "domestic":
            parser.error("--recompute-pending requires --market domestic")
        if args.scope != "live":
            parser.error("--recompute-pending requires --scope live")
        if not args.account_mode:
            parser.error("--recompute-pending requires --account-mode paper|real")
        if (args.start_date or args.end_date) and args.command != "stage-attribution":
            # stage-attribution filters on trade_date, which is known for pending rows too.
            parser.error("--recompute-pending cannot be combined with --start-date/--end-date")
    if args.command == "history" and not (args.ticker and args.date):
        parser.error("history requires both --ticker and --date")
    if args.ticker and args.market:
        parser.error("--ticker and --market are mutually exclusive")
    if args.command == "screening-ic" and not args.market:
        parser.error("screening-ic requires --market domestic|overseas (domestic/overseas are never pooled)")
    if args.command == "rating-shift" and not args.boundary:
        parser.error("rating-shift requires --boundary <ISO8601 datetime>")
    if args.command != "pending-health":
        if args.probe:
            parser.error("--probe applies to pending-health only")
        if args.warn_pct is not None:
            parser.error("--warn-pct applies to pending-health only")
    else:
        if args.scope == "backtest":
            parser.error("pending-health is live-only (the production resolver never touches backtest rows)")
        if args.probe and args.market != "domestic":
            parser.error("--probe requires --market domestic")
        if args.warn_pct is not None and not (0 < args.warn_pct <= 100):
            parser.error("--warn-pct must be in (0, 100]")
        if args.weeks is not None and args.weeks < 1:
            parser.error("--weeks must be >= 1")
        if args.ticker or args.start_date or args.end_date:
            parser.error("pending-health does not take --ticker/--start-date/--end-date")
    return args


async def _main() -> None:
    args = _parse_args()
    if args.command == "history":
        trade_date = date.fromisoformat(args.date)
        attempts = await get_decision_history(args.ticker, trade_date)
        print(render_history(attempts, args.ticker, trade_date))
    elif args.command == "experiments":
        experiments = await list_experiments(args.limit)
        print(render_experiments(experiments))
    elif args.command == "checkpoints":
        progress = await list_checkpoints()
        print(render_checkpoints(progress))
    elif args.command == "screen-validation":
        start_date = date.fromisoformat(args.start_date) if args.start_date else date(2026, 8, 1)
        result = await simulate_tiered_screen(start_date=start_date, account_mode=args.account_mode or "paper")
        print(render_screen_validation(result))
    elif args.command == "slippage":
        start_date = date.fromisoformat(args.start_date) if args.start_date else None
        end_date = date.fromisoformat(args.end_date) if args.end_date else None
        result = await compute_slippage(start=start_date, end=end_date, account_mode=args.account_mode or "paper")
        print(render_slippage(result))
    elif args.command == "reconcile":
        result = await compute_reconciliation(days=args.days)
        print(render_reconciliation(result))
    elif args.command == "screening-ic":
        from app.clock import get_clock

        end = date.fromisoformat(args.end_date) if args.end_date else get_clock().today()
        weeks = args.weeks if args.weeks is not None else 4
        start = date.fromisoformat(args.start_date) if args.start_date else end - timedelta(weeks=weeks)
        report = await compute_screening_ic(
            args.market, start, end, holding_days=args.holding_days, account_mode=args.account_mode,
        )
        print(render_screening_ic(report))
    elif args.command == "rating-action-gap":
        gap = await compute_rating_action_gap(
            market=args.market, scope=args.scope, account_mode=args.account_mode,
            trigger=args.trigger, days=args.days, limit=args.limit,
        )
        print(render_rating_action_gap(
            gap, market=args.market, run_scope=args.scope,
            account_mode=args.account_mode, run_trigger=args.trigger,
        ))
    elif args.command == "weekly":
        week = parse_week_label(args.week) if args.week else None
        scope = args.scope or "live"
        account_mode = args.account_mode or "paper"
        trigger = args.trigger or "auto"
        report = await compute_weekly_report(
            week=week, scope=scope, account_mode=account_mode, trigger=trigger, basis=args.basis,
        )
        text = render_weekly_report(report)
        print(text)
        if args.out:
            out_dir = Path(args.out)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{week_label(report.week_start)}.md"
            out_path.write_text(text, encoding="utf-8")
            print(f"\nSaved to {out_path}", file=sys.stderr)
    elif args.command == "debate-impact":
        impact = await compute_debate_impact(
            args.ticker, market=args.market, scope=args.scope, account_mode=args.account_mode,
            trigger=args.trigger,
        )
        print(render_debate_impact(
            impact, args.ticker, market=args.market, run_scope=args.scope, account_mode=args.account_mode,
            run_trigger=args.trigger,
        ))
    elif args.command == "guard-alpha" and args.recompute_pending:
        recompute_report = await compute_guard_alpha_with_recompute(
            market=args.market, scope=args.scope, account_mode=args.account_mode,
            trigger=args.trigger, basis=args.basis,
        )
        print(render_guard_alpha_recompute(recompute_report))
    elif args.command == "guard-alpha":
        since = date.fromisoformat(args.start_date) if args.start_date else None
        until = date.fromisoformat(args.end_date) if args.end_date else None
        report = await compute_guard_alpha(
            args.ticker, market=args.market, scope=args.scope, account_mode=args.account_mode,
            trigger=args.trigger, basis=args.basis, since=since, until=until,
        )
        print(render_guard_alpha(report))
    elif args.command == "rating-shift":
        boundary = parse_boundary(args.boundary)
        shift = await compute_rating_shift(
            boundary, market=args.market, scope=args.scope, account_mode=args.account_mode,
        )
        print(render_rating_shift(shift))
    elif args.command == "entry-paired" and args.recompute_pending:
        recompute_report = await compute_entry_paired_alpha_with_recompute(
            market=args.market, scope=args.scope, account_mode=args.account_mode, basis=args.basis,
        )
        print(render_entry_paired_recompute(recompute_report))
    elif args.command == "entry-paired":
        since = date.fromisoformat(args.start_date) if args.start_date else None
        until = date.fromisoformat(args.end_date) if args.end_date else None
        report = await compute_entry_paired_alpha(
            market=args.market, scope=args.scope, account_mode=args.account_mode,
            basis=args.basis, since=since, until=until,
        )
        print(render_entry_paired_alpha(report))
    elif args.command == "entry-version-strat":
        since = date.fromisoformat(args.start_date) if args.start_date else None
        until = date.fromisoformat(args.end_date) if args.end_date else None
        report = await compute_code_version_stratification(
            market=args.market, scope=args.scope, account_mode=args.account_mode,
            basis=args.basis, since=since, until=until,
        )
        print(render_code_version_stratification(report))
    elif args.command == "seat-stance":
        since = date.fromisoformat(args.start_date) if args.start_date else None
        until = date.fromisoformat(args.end_date) if args.end_date else None
        report = await compute_seat_stance(
            ticker=args.ticker, market=args.market, scope=args.scope, account_mode=args.account_mode,
            since=since, until=until,
        )
        print(render_seat_stance(report))
    elif args.command == "stage-attribution":
        since = date.fromisoformat(args.start_date) if args.start_date else None
        until = date.fromisoformat(args.end_date) if args.end_date else None
        era_boundary = parse_boundary(args.era_boundary) if args.era_boundary else None
        common = dict(
            market=args.market, account_mode=args.account_mode, trigger=args.trigger, basis=args.basis,
            since=since, until=until, era_boundary=era_boundary,
        )
        if args.recompute_pending:
            recompute_report = await compute_stage_attribution_with_recompute(**common)
            print(render_stage_attribution_recompute(recompute_report, show_underpowered=args.show_underpowered))
        else:
            report = await compute_stage_attribution(**common)
            print(render_stage_attribution(report, show_underpowered=args.show_underpowered))
    elif args.command == "rm-leak":
        leak = await compute_rm_leak(
            date.fromisoformat(args.since),
            until=date.fromisoformat(args.until) if args.until else None,
            market=args.market or "domestic", scope=args.scope or "live",
            account_mode=args.account_mode,
            after_ts=parse_boundary(args.after_ts) if args.after_ts else None, show=args.show,
        )
        print(report_to_json(leak) if args.json else render_rm_leak(leak))
    elif args.command == "pending-health":
        report = await compute_pending_health(
            market=args.market, account_mode=args.account_mode,
            weeks=args.weeks if args.weeks is not None else DEFAULT_WEEKS,
            warn_pct=args.warn_pct if args.warn_pct is not None else DEFAULT_WARN_PCT, probe=args.probe,
        )
        print(render_pending_health(report))
    elif args.command == "confidence":
        calibration = await compute_confidence_calibration(
            args.ticker, market=args.market, scope=args.scope, account_mode=args.account_mode,
            trigger=args.trigger, basis=args.basis,
        )
        print(render_confidence_calibration(
            calibration, args.ticker, market=args.market, run_scope=args.scope,
            account_mode=args.account_mode, run_trigger=args.trigger, basis=args.basis,
        ))
    else:
        stats = await compute_hit_rate_by_rating(
            args.ticker, market=args.market, scope=args.scope, account_mode=args.account_mode,
            trigger=args.trigger, basis=args.basis,
        )
        print(render(
            stats, args.ticker, market=args.market, run_scope=args.scope,
            account_mode=args.account_mode, run_trigger=args.trigger, basis=args.basis,
        ))


if __name__ == "__main__":
    asyncio.run(_main())