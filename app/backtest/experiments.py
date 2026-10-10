"""Experiment tracking — one row per backtest CLI invocation (`backtest_runs`,
schema in app.memory.db), tagging its git commit/label alongside the
resulting summary metrics. Complements `app.audit.stats` (which is about
individual DECISIONS, ticker+date); this is about backtest RUNS as a whole,
so "did this prompt/code change actually help" can be answered by listing
and comparing rows here instead of re-deriving "what code was this?" from
memory or manually diffing raw decisions.

An on-demand reporting/bookkeeping tool, not a live decision dependency — a
DB failure here should be visible to whoever ran the backtest CLI, not
silently swallowed (same convention as app.audit.stats)."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from datetime import date, datetime

from app.backtest.metrics import RiskMetrics
from app.config import settings
from app.memory.db import get_pool


def current_git_commit() -> str | None:
    """Best-effort short commit hash for whatever's checked out right now —
    None (not raised) if git isn't available or this isn't a git checkout,
    since tagging a run with its code version is a nice-to-have, not
    something that should ever block recording the run's actual results.

    `settings.git_commit` (the GIT_COMMIT env var baked into the image at
    build time) is checked first for the same reason as its twin in
    app.memory.store._current_git_commit: inside the deployed container the
    subprocess below can never succeed (.dockerignore excludes .git, the slim
    image has no git binary), so a containerized backtest tagged every run
    with a NULL commit. See `settings.git_commit`'s own comment.
    배포 컨테이너에서는 git 명령이 실패하므로 환경변수를 먼저 확인."""
    if settings.git_commit:
        return settings.git_commit
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5, check=True,
        )
        return result.stdout.strip() or None
    except Exception:
        return None


@dataclass
class ExperimentRecord:
    id: int
    created_at: datetime
    git_commit: str | None
    label: str | None
    kind: str  # "single" | "portfolio"
    tickers: str  # comma-joined
    benchmark_ticker: str
    start_date: date
    end_date: date
    cadence_days: int
    n_dates: int
    return_pct: float | None
    exposure_return_pct: float | None
    benchmark_return_pct: float | None
    sharpe: float | None
    sortino: float | None
    max_drawdown: float | None
    cvar_95: float | None
    win_rate: float | None
    mean_alpha: float | None


async def record_experiment(
    *,
    kind: str,
    tickers: list[str],
    benchmark_ticker: str,
    start: date,
    end: date,
    cadence_days: int,
    n_dates: int,
    return_pct: float | None,
    benchmark_return_pct: float | None,
    risk_metrics: RiskMetrics,
    win_rate: float | None = None,
    mean_alpha: float | None = None,
    label: str | None = None,
    exposure_return_pct: float | None = None,
) -> int:
    """Persists one experiment row (tagged with the current git commit,
    best-effort) and returns its id. `risk_metrics` should be the REAL
    (ledger/portfolio-equity-based) risk metrics, not the exposure-proxy
    ones — same "actual, not hypothetical" preference `reporting.py` already
    applies when it labels one primary and the other supplemental. `return_pct`
    is held to the SAME rule: the real ledger/portfolio-equity return, never
    the exposure-proxy one — `exposure_return_pct` (optional, defaults to
    None) is where a signal-quality proxy belongs instead, kept separately
    so a "single" run's exposure proxy and a "portfolio" run's genuinely
    actual return are never mixed in the same `return_pct` column."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO backtest_runs (
                git_commit, label, kind, tickers, benchmark_ticker, start_date, end_date,
                cadence_days, n_dates, return_pct, benchmark_return_pct,
                sharpe, sortino, max_drawdown, cvar_95, win_rate, mean_alpha, exposure_return_pct
            ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18)
            RETURNING id
            """,
            current_git_commit(), label, kind, ",".join(tickers), benchmark_ticker, start, end,
            cadence_days, n_dates, return_pct, benchmark_return_pct,
            risk_metrics.sharpe, risk_metrics.sortino, risk_metrics.max_drawdown, risk_metrics.cvar_95,
            win_rate, mean_alpha, exposure_return_pct,
        )
    return row["id"]


async def list_experiments(limit: int = 20, label_prefix: str | None = None) -> list[ExperimentRecord]:
    """Most recent experiments first — the whole point is comparing several
    runs at a glance, so newest-first matches how you'd actually scan this
    (did the last change I made help, compared to the few before it).

    `label_prefix`, when given, filters to labels starting with it BEFORE
    applying `limit` -- without this, `limit` bounds the most recent rows
    across EVERY experiment ever recorded in backtest_runs, so a caller
    scoping to one sweep's label prefix (app.backtest.atr_sweep_report) could
    silently lose that sweep's own older rows to unrelated experiments
    recorded since, with no indication anything was dropped.
    label_prefix가 주어지면 SQL 단계에서 필터링 — limit이 전체 테이블이 아니라
    이 접두사에 해당하는 행에만 적용되도록.

    `label_prefix` is operator-supplied via a CLI --label-prefix flag (not
    adversarial/untrusted input), so no LIKE-wildcard (%/_) escaping is
    applied beyond the parameterization already used below — same practice
    as app.audit.stats's plain `ticker LIKE '%.KS'`."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        if label_prefix is not None:
            rows = await conn.fetch(
                """
                SELECT id, created_at, git_commit, label, kind, tickers, benchmark_ticker,
                       start_date, end_date, cadence_days, n_dates, return_pct, exposure_return_pct,
                       benchmark_return_pct, sharpe, sortino, max_drawdown, cvar_95, win_rate, mean_alpha
                FROM backtest_runs
                WHERE label LIKE $2
                ORDER BY created_at DESC
                LIMIT $1
                """,
                limit, f"{label_prefix}%",
            )
        else:
            rows = await conn.fetch(
                """
                SELECT id, created_at, git_commit, label, kind, tickers, benchmark_ticker,
                       start_date, end_date, cadence_days, n_dates, return_pct, exposure_return_pct,
                       benchmark_return_pct, sharpe, sortino, max_drawdown, cvar_95, win_rate, mean_alpha
                FROM backtest_runs
                ORDER BY created_at DESC
                LIMIT $1
                """,
                limit,
            )
    return [ExperimentRecord(**dict(row)) for row in rows]


def render_experiments(experiments: list[ExperimentRecord]) -> str:
    if not experiments:
        return "No experiments recorded yet."

    lines = [
        f"{'ID':>4} {'When':<16} {'Commit':<8} {'Kind':<9} {'Tickers':<20} {'Return':>8} "
        f"{'Bench':>8} {'Sharpe':>7} {'MaxDD':>7} {'WinRate':>8} {'Label'}",
        "-" * 110,
    ]
    for e in experiments:
        ret = f"{e.return_pct:+.2%}" if e.return_pct is not None else "n/a"
        bench = f"{e.benchmark_return_pct:+.2%}" if e.benchmark_return_pct is not None else "n/a"
        sharpe = f"{e.sharpe:+.2f}" if e.sharpe is not None else "n/a"
        max_dd = f"{e.max_drawdown:+.2%}" if e.max_drawdown is not None else "n/a"
        win_rate = f"{e.win_rate:.1%}" if e.win_rate is not None else "n/a"
        tickers = e.tickers if len(e.tickers) <= 20 else e.tickers[:17] + "..."
        lines.append(
            f"{e.id:>4} {e.created_at:%Y-%m-%d %H:%M} {(e.git_commit or 'n/a'):<8} {e.kind:<9} "
            f"{tickers:<20} {ret:>8} {bench:>8} {sharpe:>7} {max_dd:>7} {win_rate:>8} {e.label or ''}"
        )
    return "\n".join(lines)
