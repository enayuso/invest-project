"""Web exposure for `app.audit` — read-only self-audit/accuracy reporting.

No new statistics live here: every number comes from `app.audit.stats`/
`app.audit.weekly`, already used by the `python -m app.audit` CLI and
already test-covered there. This module only adds an HTTP-shaped view over
those same functions, for the "적중률" (accuracy) frontend tab.

No `Depends(require_kis)` on this router, deliberately — these are DB-only
reads with no dependency on a configured broker, and `require_kis` also
triggers `ensure_market_feed()`, an unwanted side effect for a reporting
route. A KIS-unconfigured deployment must still be able to render this
screen.

`account_mode` is intentionally NOT a query parameter: this deployment runs
exactly one active mode at a time (`app.config.current_account_mode()`), and
a control that let the UI page into the *other* mode's track record would
misread as "how is my current mode doing" — the response echoes the active
mode instead so the frontend can label it.

`trigger` defaults to `"auto"` here — UNLIKE `app.audit.stats`'s own
None-pooled convention for every other filter. This endpoint exists for
exactly one purpose: a person reading a win rate to decide whether to put
more real money behind the system ([[stage-gate-criteria-draft]]'s capital-
increase ladder). A default that silently mixed in decisions the human
self-selected by clicking "analyze" on tickers they were already worried
about would make the gate number look fine while measuring a biased
population, and would keep doing so forever since nobody re-checks a
plausible number. `app.audit.weekly.compute_weekly_report` already broke
`stats.py`'s pooled-by-default convention for `scope`/`account_mode` with
the identical argument — this endpoint follows that same precedent.
`?trigger=all` (pooled) and `?trigger=manual` (inverted view) are one query
param away.

Every ratio/return figure is returned as a bare fraction (0.373, -0.0142) —
never pre-multiplied by 100, never pre-formatted as a string. The frontend's
own `fmtDelta`/percentage helpers already do that.
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, HTTPException, Query

from app.audit.stats import (
    compute_confidence_calibration,
    compute_debate_impact,
    compute_hit_rate_by_rating,
    directional_totals,
)
from app.audit.weekly import (
    _MIN_N_FOR_F7_VERDICT,
    _MIN_N_FOR_RATIO,
    compute_weekly_report,
    parse_week_label,
    week_label,
)
from app.config import current_account_mode

router = APIRouter()
logger = logging.getLogger(__name__)


def _validate_market(market: str | None) -> None:
    if market not in (None, "domestic", "overseas"):
        raise HTTPException(400, f"market must be 'domestic', 'overseas', or omitted, got {market!r}")


def _validate_scope(scope: str) -> None:
    if scope not in ("live", "backtest"):
        raise HTTPException(400, f"scope must be 'live' or 'backtest', got {scope!r}")


def _validate_basis(basis: str) -> None:
    if basis not in ("variable", "horizon"):
        raise HTTPException(400, f"basis must be 'variable' or 'horizon', got {basis!r}")


def _validate_trigger(trigger: str) -> None:
    if trigger not in ("auto", "manual", "all"):
        raise HTTPException(400, f"trigger must be 'auto', 'manual', or 'all', got {trigger!r}")


@router.get("/api/audit/accuracy")
async def audit_accuracy(
    market: str | None = Query(default=None),
    scope: str = Query(default="live"),
    trigger: str = Query(default="auto"),
    basis: str = Query(default="variable"),
) -> dict:
    _validate_market(market)
    _validate_scope(scope)
    _validate_trigger(trigger)
    _validate_basis(basis)
    account_mode = current_account_mode()
    trigger_arg = None if trigger == "all" else trigger

    try:
        ratings, confidence, debate = await asyncio.gather(
            compute_hit_rate_by_rating(
                market=market, scope=scope, account_mode=account_mode, trigger=trigger_arg, basis=basis,
            ),
            compute_confidence_calibration(
                market=market, scope=scope, account_mode=account_mode, trigger=trigger_arg, basis=basis,
            ),
            compute_debate_impact(market=market, scope=scope, account_mode=account_mode, trigger=trigger_arg),
        )
    except Exception:
        logger.warning("audit accuracy query failed", exc_info=True)
        raise HTTPException(503, "audit query failed — see server log")

    totals = directional_totals(ratings)

    return {
        "filters": {
            "market": market, "scope": scope, "trigger": trigger, "basis": basis, "account_mode": account_mode,
        },
        "thresholds": {"min_n_for_ratio": _MIN_N_FOR_RATIO, "min_n_for_f7_verdict": _MIN_N_FOR_F7_VERDICT},
        "headline": {
            "directional_n": totals.scored_n,
            "directional_win_rate": totals.win_rate,
            "unscored_hold_n": totals.unscored_hold_n,
            "unscorable_n": totals.unscorable_n,
        },
        "ratings": [
            {
                "rating": r.rating,
                "n": r.n,
                "win_rate": r.win_rate,
                "mean_alpha": r.mean_alpha,
                "mean_raw_return": r.mean_raw_return,
                "win_condition": r.win_condition,
                "unscorable_n": r.unscorable_n,
                "stdev_alpha": r.stdev_alpha,
                # How many of `n` are reconstructed from the trades ledger
                # (held_at_decision_inferred) rather than directly observed —
                # only ever nonzero for the two Hold buckets. See
                # app.memory.db's held_at_decision_inferred column comment.
                "inferred_n": r.inferred_n,
            }
            for r in ratings
        ],
        "confidence": {
            "n_with_confidence": confidence.n_with_confidence,
            "n_without_confidence": confidence.n_without_confidence,
            "signed_alpha_corr": confidence.signed_alpha_corr,
            "buckets": [
                {
                    "label": b.label,
                    "lower": b.lower,
                    "size_multiplier": b.size_multiplier,
                    "n": b.n,
                    "win_rate": b.win_rate,
                    "mean_signed_alpha": b.mean_signed_alpha,
                    "unscorable_n": b.unscorable_n,
                }
                for b in confidence.buckets
            ],
        },
        "debate": {
            "n": debate.n,
            "changed": debate.changed,
            "changed_pct": debate.changed_pct,
            "breakdown": debate.breakdown,
        },
    }


def _rating_stats_json(s) -> dict | None:
    if s is None:
        return None
    return {
        "n": s.n,
        "win_rate": s.win_rate,
        "mean_alpha": s.mean_alpha,
        "mean_raw_return": s.mean_raw_return,
        "win_condition": s.win_condition,
        "unscorable_n": s.unscorable_n,
        "stdev_alpha": s.stdev_alpha,
        "inferred_n": s.inferred_n,
    }


@router.get("/api/audit/weekly")
async def audit_weekly(
    week: str | None = Query(default=None),
    scope: str = Query(default="live"),
    trigger: str = Query(default="auto"),
    basis: str = Query(default="variable"),
) -> dict:
    _validate_scope(scope)
    _validate_trigger(trigger)
    _validate_basis(basis)
    account_mode = current_account_mode()
    trigger_arg = None if trigger == "all" else trigger

    week_anchor = None
    if week is not None:
        try:
            week_anchor = parse_week_label(week)
        except ValueError as exc:
            raise HTTPException(400, str(exc))

    try:
        report = await compute_weekly_report(
            week=week_anchor, scope=scope, account_mode=account_mode, trigger=trigger_arg, basis=basis,
        )
    except Exception:
        logger.warning("audit weekly query failed", exc_info=True)
        raise HTTPException(503, "audit query failed — see server log")

    return {
        "week_label": week_label(report.week_start),
        "week_start": report.week_start.isoformat(),
        "week_end": report.week_end.isoformat(),
        "generated_at": report.generated_at.isoformat(),
        "filters": {
            "scope": report.scope, "trigger": trigger, "basis": report.basis, "account_mode": report.account_mode,
        },
        "thresholds": {"min_n_for_ratio": _MIN_N_FOR_RATIO, "min_n_for_f7_verdict": _MIN_N_FOR_F7_VERDICT},
        "totals": {
            "this_week_n": report.this_week_n,
            "this_week_unscorable_n": report.this_week_unscorable_n,
            "last_8_weeks_n": report.last_8_weeks_n,
            "cumulative_n": report.cumulative_n,
        },
        "ratings": [
            {
                "rating": row.rating,
                "this_week": _rating_stats_json(row.this_week),
                "last_8_weeks": _rating_stats_json(row.last_8_weeks),
                "cumulative": _rating_stats_json(row.cumulative),
            }
            for row in report.ratings
        ],
        "f7": {
            "total_overridden": report.f7.total_overridden,
            "cells": [
                {
                    "row_label": c.row_label, "n": c.n, "wins": c.wins, "win_rate": c.win_rate,
                    "inferred_n": c.inferred_n,
                }
                for c in report.f7.cells
            ],
        },
    }
