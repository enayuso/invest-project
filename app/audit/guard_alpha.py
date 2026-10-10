"""Guard-vs-realized-alpha instrument (2026-09-14): does a guard violation
recorded in `memory_entries.full_transcript` actually correlate with a WORSE
realized trading outcome? Nobody had ever measured this — guards were being
fixed in guessed-priority order. This module replaces the guess with a
measurement, and REFUSES to render a verdict when the sample is too small
(see `_MIN_DAYS_FOR_VERDICT`/`_MIN_ROWS_PER_GROUP` below) — a confident-
looking number from a handful of trading days would be worse than useless.

Public API mirrors `app.audit.screening_ic`:
  `compute_guard_alpha` -> `GuardAlphaReport`
  `render_guard_alpha` -> str

Row selection reuses the EXACT filter/dedup/scoping conventions of
`app.audit.stats.compute_confidence_calibration` (status='resolved', DISTINCT
ON (ticker, trade_date) ... ORDER BY ... created_at DESC, analysts_failed
guard, user_id scoping, since/until on horizon_eval_date) — see that
function's docstring for the shared rationale, not repeated here. One
deliberate difference: `trigger` defaults to auto-only here (manual runs
excluded by default), following `compute_weekly_report`'s precedent instead
— a human hand-picking a ticker is sample-selection bias that would directly
corrupt a violated-vs-clean cohort comparison.

Guard keys (every top-level `full_transcript` key ending in `_violations` or
`_flags`) are discovered DYNAMICALLY per run, never hardcoded, so a new
guard is picked up automatically.

Some decision signals worth measuring are NOT guard keys (`neutral_reversal`,
`consensus_override`) and cannot simply be added to the suffix list, because
raw truthy/falsy binning on them is actively misleading:
  - `consensus_override` is `{}` (falsy but PRESENT) on almost every row
    where the gate ran at all, and non-empty on a small few — raw binning
    would pair "gate ran, did nothing" against "gate ran and fired" on
    whatever few days both occur, producing a p-value from essentially one
    day's data.
  - `neutral_reversal` is a non-empty dict (truthy) on nearly every row where
    it is present at all — raw binning gives a "clean" group of ~zero,
    permanently unpairable, for the wrong reason (there is no "absent"
    signal to exploit, but the shape still defeats a plain truthy check).
`_DERIVED_SIGNALS` is a registry of `name -> transcript -> True|False|None`
classifiers, separate from and additional to `_guard_keys` discovery, that
recover a real two-sided contrast from these fields. `None` means "exclude
this row from this signal's comparison" — the exact same semantics as a
guard key being absent from a transcript. This `None`/excluded state is
deliberate and important: it is what prevents (a) diluting the
Benjamini-Hochberg `m` with a signal that cannot possibly pair (see
`_apply_bh_correction`), and (b) a calendar confound from comparing rows
where a field is structurally absent against rows where it is present. A
signal that never pairs costs nothing — it always yields `p_value=None` and
is never counted into `m`.

Statistics are a within-day paired comparison (the binding constraint is
independent TRADING DAYS, not rows — same-day rows share a market/sector
shock) with Benjamini-Hochberg FDR correction across all keys tested in one
run. See `_compute_key_stats` for the exact algorithm. This is a research/
audit instrument only: no INSERT/UPDATE/DELETE anywhere in this module."""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import date, timedelta

import numpy as np

from app.audit.pending_recompute import REASON_NOT_MATURED, holding_days_stats, recompute_pending_alphas
from app.backtest import factor_metrics
from app.clock import local_today
from app.config import settings
from app.memory.db import get_pool
from app.user_context import get_current_user_id

_MIN_DAYS_FOR_VERDICT = 20
_MIN_ROWS_PER_GROUP = 20
_N_PERMUTATIONS = 20_000
_PERMUTATION_SEED = 0


@dataclass(frozen=True)
class GuardKeyStats:
    key: str
    n_violated: int
    n_clean: int
    n_paired_days: int
    paired_mean_delta: float | None
    paired_ci95: tuple[float, float] | None
    p_value: float | None
    q_value: float | None
    naive_mean_violated: float | None
    naive_mean_clean: float | None
    naive_delta: float | None
    rating_mix_violated: dict[str, int] = field(default_factory=dict)
    rating_mix_clean: dict[str, int] = field(default_factory=dict)
    verdict: str = "UNDERPOWERED"  # one of "FINDING" / "NO_EFFECT" / "UNDERPOWERED"
    no_pairing_reason: str | None = None

    @property
    def underpowered(self) -> bool:
        return self.verdict == "UNDERPOWERED"


@dataclass(frozen=True)
class GuardAlphaReport:
    basis: str
    n_rows: int
    n_days: int
    start: date | None
    end: date | None
    manual_excluded: bool
    n_keys_discovered: int
    n_keys_tested: int
    holding_days_min: int | None
    holding_days_max: int | None
    holding_days_mean: float | None
    key_stats: list[GuardKeyStats] = field(default_factory=list)
    # Count of registered `_DERIVED_SIGNALS` (neutral_reversed,
    # neutral_reversal_uncited, consensus_override_gated) — reported
    # SEPARATELY from n_keys_discovered (raw guard keys only), never folded
    # in, so a reader can always tell how many of each kind ran this cycle.
    n_derived_signals: int = 0


def _parse_transcript(raw) -> dict:
    """`full_transcript` may arrive as a JSON string or an already-decoded
    dict depending on the asyncpg/driver path taken to fetch it — handle
    both rather than assuming one."""
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            decoded = json.loads(raw)
        except (ValueError, TypeError):
            return {}
        return decoded if isinstance(decoded, dict) else {}
    return {}


def _guard_keys(transcript: dict) -> list[str]:
    return [k for k in transcript.keys() if k.endswith("_violations") or k.endswith("_flags")]


def _raw_key_classifier(key: str):
    """Wraps a raw guard-key lookup as a `transcript -> True|False|None`
    classifier, reproducing `_compute_key_stats`'s ORIGINAL behavior exactly:
    key absent -> None (excluded), present-truthy -> True, present-falsy ->
    False. Used so the raw-key path and the derived-signal path share one
    binning implementation without changing raw-key results at all."""
    def classify(transcript: dict):
        if key not in transcript:
            return None
        return bool(transcript[key])
    return classify


def _classify_neutral_reversed(transcript: dict):
    """derived:neutral_reversed — did the Neutral risk debater flip its
    round-1 verdict at all this cycle? None (excluded) when `neutral_reversal`
    is absent, not a dict, or an empty dict — those are "the field never
    fired", not "it fired and found no reversal"."""
    record = transcript.get("neutral_reversal")
    if not isinstance(record, dict) or not record:
        return None
    return bool(record.get("reversed"))


def _classify_neutral_reversal_uncited(transcript: dict):
    """derived:neutral_reversal_uncited — AMONG flips, was the flip uncited
    (no new fact, no later-turn argument named)? Excluded (None) whenever the
    flip itself didn't happen (non-flip is not part of this contrast) or the
    inputs needed to judge citedness are missing. Prefers the precomputed
    `reversal_basis` field (added 2026-09-17 by app.agents.risk_debators);
    falls back to deriving from `new_fact_cited`/`later_turn_argument_cited`
    only when both are present, for older stored records. CRITICAL: a
    missing `reversal_basis` on an old record must NOT be read as "neither"
    — roughly 10 stored rows predate `later_turn_argument_cited` and are
    genuinely unjudgeable; misreading them as uncited is exactly the error
    this signal exists to avoid, so the fallback returns None when its own
    inputs are also missing."""
    record = transcript.get("neutral_reversal")
    if not isinstance(record, dict) or not record:
        return None
    if not record.get("reversed"):
        return None
    if "reversal_basis" in record:
        return record["reversal_basis"] == "neither"
    if "new_fact_cited" in record and "later_turn_argument_cited" in record:
        return not record["new_fact_cited"] and not record["later_turn_argument_cited"]
    return None


def _classify_consensus_override_gated(transcript: dict):
    """derived:consensus_override_gated — when the PM overrode a unanimous
    desk (the gate applied at all), did the gate downgrade it to Hold? None
    (excluded) when `consensus_override` is absent, not a dict, or the empty
    dict `{}` — `{}` means the gate did not apply this cycle, which is a
    non-observation, not a "clean" (gate applied and passed) observation."""
    record = transcript.get("consensus_override")
    if not isinstance(record, dict) or not record:
        return None
    return bool(record.get("gated"))


# name -> transcript -> True|False|None. Additional to, and reported
# separately from, `_guard_keys` discovery — see module docstring for why
# these three can't just be raw guard keys.
_DERIVED_SIGNALS: dict[str, "Callable[[dict], bool | None]"] = {
    "derived:neutral_reversed": _classify_neutral_reversed,
    "derived:neutral_reversal_uncited": _classify_neutral_reversal_uncited,
    "derived:consensus_override_gated": _classify_consensus_override_gated,
}


# `gate_orig_rating` recovery, shared verbatim between the main resolved-row
# query and the pending-row query the recompute mode reads (§B below) — a
# module-level constant rather than two copies, so the two queries can never
# drift apart. If you touch this, it changes BOTH queries and `_signed_alpha`
# (the Python reimplementation of the `scored` CTE's CASE expression below)
# all at once — see `_signed_alpha`'s own docstring.
_GATE_ORIG_RATING_SQL = r"""COALESCE(
                    substring(decision_text from '\[Hysteresis gate: downgraded from (\w+) to Hold'),
                    substring(
                        full_transcript->>'investment_thesis'
                        from '\[Hysteresis gate: downgraded from (\w+) to Hold'
                    )
                )"""


def _validate_filters(
    market: str | None, scope: str | None, account_mode: str | None, trigger: str | None, basis: str,
) -> tuple[str, str]:
    if market not in (None, "domestic", "overseas"):
        raise ValueError(f"market must be 'domestic', 'overseas', or None, got {market!r}")
    if scope not in (None, "live", "backtest"):
        raise ValueError(f"scope must be 'live', 'backtest', or None, got {scope!r}")
    if account_mode not in (None, "paper", "real"):
        raise ValueError(f"account_mode must be 'paper', 'real', or None, got {account_mode!r}")
    if trigger not in (None, "auto", "manual"):
        raise ValueError(f"trigger must be 'auto', 'manual', or None, got {trigger!r}")
    if basis not in ("variable", "horizon"):
        raise ValueError(f"basis must be 'variable' or 'horizon', got {basis!r}")
    return ("alpha_return", "raw_return") if basis == "variable" else (
        "horizon_alpha_return", "horizon_raw_return",
    )


def _build_where(
    ticker: str | None, market: str | None, scope: str | None, account_mode: str | None,
    trigger: str | None, since: date | None, until: date | None, *, status: str = "resolved",
) -> tuple[str, list, bool]:
    # `status` is interpolated into SQL, so it is closed to two literals; the
    # default reproduces the pre-2026-09-29 string exactly.
    if status not in ("resolved", "pending"):
        raise ValueError(f"status must be 'resolved' or 'pending', got {status!r}")
    conditions = [f"status = '{status}'"]
    params: list = []
    if ticker is not None:
        params.append(ticker)
        conditions.append(f"ticker = ${len(params)}")
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
    # trigger: default to auto-only (module docstring — sample-selection bias
    # from manually-picked tickers would corrupt the cohort comparison).
    manual_excluded = True
    if trigger in (None, "auto"):
        conditions.append("decision_trigger IS DISTINCT FROM 'manual'")
    else:
        conditions.append("decision_trigger = 'manual'")
        manual_excluded = False
    if since is not None:
        params.append(since)
        conditions.append(f"horizon_eval_date >= ${len(params)}")
    if until is not None:
        params.append(until)
        conditions.append(f"horizon_eval_date <= ${len(params)}")
    # Same analysts_failed exclusion as compute_confidence_calibration.
    conditions.append("(analysts_failed IS NULL OR analysts_failed = 0)")
    params.append(get_current_user_id())
    conditions.append(f"user_id = ${len(params)}")
    return " AND ".join(conditions), params, manual_excluded


async def _fetch_rows(
    *, ticker: str | None, market: str | None, scope: str | None, account_mode: str | None,
    trigger: str | None, basis: str, since: date | None, until: date | None, conn=None,
) -> tuple[list, bool]:
    alpha_col, raw_col = _validate_filters(market, scope, account_mode, trigger, basis)
    where_clause, params, manual_excluded = _build_where(
        ticker, market, scope, account_mode, trigger, since, until,
    )

    # NOTE: the `latest`/`scored` CTEs below are copied verbatim from
    # app.audit.stats.compute_confidence_calibration's `common_ctes` (the
    # gate_orig_rating recovery + signed-alpha CASE) rather than imported or
    # refactored into a shared helper — deliberate duplication, see this
    # module's own docstring / task spec: refactoring the shared CTE would
    # risk changing the hit-rate/confidence reports that gate real capital
    # decisions, out of scope for a read-only audit tool. If you touch the
    # signed-alpha logic, update ALL THREE copies (stats.py's SQL, this
    # query, and `_signed_alpha`'s Python reimplementation below).
    query = rf"""
        WITH latest AS (
            SELECT DISTINCT ON (ticker, trade_date)
                trade_date, ticker, rating, {alpha_col} AS alpha_return, {raw_col} AS raw_return,
                held_at_decision, held_at_decision_inferred, full_transcript, holding_days,
                {_GATE_ORIG_RATING_SQL} AS gate_orig_rating
            FROM memory_entries
            WHERE {where_clause}
            ORDER BY ticker, trade_date, created_at DESC
        ), scored AS (
            SELECT
                trade_date, ticker, rating, full_transcript, holding_days,
                CASE
                    WHEN gate_orig_rating = 'Buy' THEN alpha_return
                    WHEN gate_orig_rating = 'Sell' THEN -alpha_return
                    WHEN rating IN ('Buy', 'Overweight') THEN alpha_return
                    WHEN rating IN ('Sell', 'Underweight') THEN -alpha_return
                    WHEN rating = 'Hold' AND COALESCE(held_at_decision, held_at_decision_inferred) IS TRUE
                        THEN raw_return
                    WHEN rating = 'Hold' AND COALESCE(held_at_decision, held_at_decision_inferred) IS FALSE
                        THEN -alpha_return
                    ELSE NULL
                END AS signed_alpha
            FROM latest
        )
        SELECT trade_date, ticker, rating, full_transcript, holding_days, signed_alpha
        FROM scored
        WHERE signed_alpha IS NOT NULL AND signed_alpha <> 'NaN'::float8
        ORDER BY trade_date, ticker
    """

    if conn is not None:
        rows = await conn.fetch(query, *params)
    else:
        pool = await get_pool()
        async with pool.acquire() as acquired:
            rows = await acquired.fetch(query, *params)
    return rows, manual_excluded


def _signed_alpha(
    rating: str | None, gate_orig_rating: str | None, alpha: float | None, raw: float | None,
    held_at_decision: bool | None, held_at_decision_inferred: bool | None,
) -> float | None:
    """Python reimplementation of the `scored` CTE's CASE expression in
    `_fetch_rows` above, for rows that have no SQL row to compute it from yet
    (a still-pending row recomputed in memory, §B below). Branch order and
    semantics must match that CASE expression EXACTLY — see `_GATE_ORIG_RATING_SQL`'s
    own comment: update all three copies together."""
    if gate_orig_rating == "Buy":
        value = alpha
    elif gate_orig_rating == "Sell":
        value = None if alpha is None else -alpha
    elif rating in ("Buy", "Overweight"):
        value = alpha
    elif rating in ("Sell", "Underweight"):
        value = None if alpha is None else -alpha
    elif rating == "Hold":
        held = held_at_decision if held_at_decision is not None else held_at_decision_inferred
        if held is True:
            value = raw
        elif held is False:
            value = None if alpha is None else -alpha
        else:
            value = None
    else:
        value = None
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return value


async def compute_guard_alpha(
    ticker: str | None = None, market: str | None = None, scope: str | None = None,
    account_mode: str | None = None, trigger: str | None = None, basis: str = "variable",
    since: date | None = None, until: date | None = None,
) -> GuardAlphaReport:
    """Measures whether each guard's violation flag correlates with a worse
    realized signed alpha, using a within-day paired design (see module
    docstring). `ticker`/`market`/`scope`/`account_mode`/`trigger`/`basis`/
    `since`/`until` follow `app.audit.stats.compute_confidence_calibration`'s
    exact semantics, with one difference: `trigger=None` (the default) here
    means auto-only (manual excluded), not pooled — see module docstring.

    TRAP (matches compute_confidence_calibration's convention, do not "fix"):
    `since`/`until` filter on `horizon_eval_date`, which is NULL for every row
    whose fixed horizon has not matured yet. A recent `since` can silently
    return zero rows even when plenty of scorable rows exist, because that
    whole cohort's horizon hasn't matured — check row counts before assuming
    an empty result means "no violations happened"."""
    rows, manual_excluded = await _fetch_rows(
        ticker=ticker, market=market, scope=scope, account_mode=account_mode, trigger=trigger,
        basis=basis, since=since, until=until,
    )
    return _build_report(rows, basis=basis, manual_excluded=manual_excluded)


def _build_report(rows, *, basis: str, manual_excluded: bool) -> GuardAlphaReport:
    parsed_rows = []
    all_keys: set[str] = set()
    for row in rows:
        transcript = _parse_transcript(row["full_transcript"])
        keys = _guard_keys(transcript)
        all_keys.update(keys)
        parsed_rows.append({
            "trade_date": row["trade_date"],
            "ticker": row["ticker"],
            "rating": row["rating"],
            "transcript": transcript,
            "holding_days": row["holding_days"],
            "signed_alpha": row["signed_alpha"],
        })

    dates = sorted({r["trade_date"] for r in parsed_rows})
    holding_values = [r["holding_days"] for r in parsed_rows if r["holding_days"] is not None]

    key_stats = [_compute_key_stats(key, parsed_rows) for key in sorted(all_keys)]
    key_stats.extend(
        _compute_key_stats(name, parsed_rows, classify=classify)
        for name, classify in _DERIVED_SIGNALS.items()
    )

    # Benjamini-Hochberg FDR correction across every key that actually
    # produced a p-value this run (a key with no paired days is never
    # "tested" in the BH sense — see _apply_bh_correction's own docstring).
    _apply_bh_correction(key_stats)
    n_keys_tested = sum(1 for s in key_stats if s.p_value is not None)

    # Re-evaluate the verdict now that q-values are final. Three states:
    # UNDERPOWERED (any sample gate fails, including "no p-value at all"),
    # NO_EFFECT (adequate sample, q>=0.05 -> measured, no detectable effect,
    # deprioritize), FINDING (adequate sample, q<0.05).
    final_stats = []
    for stats_row in key_stats:
        well_powered = (
            stats_row.n_paired_days >= _MIN_DAYS_FOR_VERDICT
            and stats_row.n_violated >= _MIN_ROWS_PER_GROUP
            and stats_row.n_clean >= _MIN_ROWS_PER_GROUP
            and stats_row.q_value is not None
        )
        if not well_powered:
            verdict = "UNDERPOWERED"
        elif stats_row.q_value < 0.05:
            verdict = "FINDING"
        else:
            verdict = "NO_EFFECT"
        final_stats.append(replace(stats_row, verdict=verdict))

    return GuardAlphaReport(
        basis=basis,
        n_rows=len(parsed_rows),
        n_days=len(dates),
        start=dates[0] if dates else None,
        end=dates[-1] if dates else None,
        manual_excluded=manual_excluded,
        n_keys_discovered=len(all_keys),
        n_keys_tested=n_keys_tested,
        n_derived_signals=len(_DERIVED_SIGNALS),
        holding_days_min=min(holding_values) if holding_values else None,
        holding_days_max=max(holding_values) if holding_values else None,
        holding_days_mean=(sum(holding_values) / len(holding_values)) if holding_values else None,
        key_stats=final_stats,
    )


def _compute_key_stats(key: str, parsed_rows: list[dict], *, classify=None) -> GuardKeyStats:
    # 1c: a row is assigned to a group ONLY IF the classifier says True/False
    # for it. `classify is None` (a raw guard key) -> the classifier is
    # "present in transcript with a truthy/falsy value", reproducing this
    # function's original raw-key-only behavior exactly (do not change this
    # default's semantics — several tests and the live BH `m` depend on it).
    # A classifier returning None for a row -> EXCLUDED entirely from this
    # key/signal's comparison — most guards were added on specific dates, so
    # treating "no signal" as "clean" would compare post-guard rows against
    # pre-guard rows, a pure calendar confound rather than a guard effect.
    # Derived signals reuse this same None-excludes semantics deliberately
    # (see module docstring / `_DERIVED_SIGNALS`).
    if classify is None:
        classify = _raw_key_classifier(key)
    violated_rows = []
    clean_rows = []
    for row in parsed_rows:
        transcript = row["transcript"]
        value = classify(transcript)
        if value is None:
            continue
        if value:
            violated_rows.append(row)
        else:
            clean_rows.append(row)

    n_violated = len(violated_rows)
    n_clean = len(clean_rows)

    rating_mix_violated = _rating_mix(violated_rows)
    rating_mix_clean = _rating_mix(clean_rows)

    naive_mean_violated = (
        sum(r["signed_alpha"] for r in violated_rows) / n_violated if n_violated else None
    )
    naive_mean_clean = sum(r["signed_alpha"] for r in clean_rows) / n_clean if n_clean else None
    naive_delta = (
        naive_mean_violated - naive_mean_clean
        if naive_mean_violated is not None and naive_mean_clean is not None
        else None
    )

    if not violated_rows or not clean_rows:
        return GuardKeyStats(
            key=key, n_violated=n_violated, n_clean=n_clean, n_paired_days=0,
            paired_mean_delta=None, paired_ci95=None, p_value=None, q_value=None,
            naive_mean_violated=naive_mean_violated, naive_mean_clean=naive_mean_clean,
            naive_delta=naive_delta, rating_mix_violated=rating_mix_violated,
            rating_mix_clean=rating_mix_clean, verdict="UNDERPOWERED",
            no_pairing_reason="0 violated" if not violated_rows else "0 clean",
        )

    by_day_violated: dict[date, list[float]] = {}
    by_day_clean: dict[date, list[float]] = {}
    for row in violated_rows:
        by_day_violated.setdefault(row["trade_date"], []).append(row["signed_alpha"])
    for row in clean_rows:
        by_day_clean.setdefault(row["trade_date"], []).append(row["signed_alpha"])

    paired_days = sorted(set(by_day_violated) & set(by_day_clean))
    if not paired_days:
        return GuardKeyStats(
            key=key, n_violated=n_violated, n_clean=n_clean, n_paired_days=0,
            paired_mean_delta=None, paired_ci95=None, p_value=None, q_value=None,
            naive_mean_violated=naive_mean_violated, naive_mean_clean=naive_mean_clean,
            naive_delta=naive_delta, rating_mix_violated=rating_mix_violated,
            rating_mix_clean=rating_mix_clean, verdict="UNDERPOWERED",
            no_pairing_reason="no day has both groups",
        )

    day_violated_lists = [by_day_violated[d] for d in paired_days]
    day_clean_lists = [by_day_clean[d] for d in paired_days]
    deltas = [
        (sum(v) / len(v)) - (sum(c) / len(c))
        for v, c in zip(day_violated_lists, day_clean_lists)
    ]
    paired_mean_delta = sum(deltas) / len(deltas)

    blocks = [[d] for d in deltas]
    paired_ci95 = factor_metrics.block_bootstrap_ci(blocks, seed=0)

    p_value = _permutation_p_value(day_violated_lists, day_clean_lists, observed=paired_mean_delta)

    return GuardKeyStats(
        key=key, n_violated=n_violated, n_clean=n_clean, n_paired_days=len(paired_days),
        paired_mean_delta=paired_mean_delta, paired_ci95=paired_ci95, p_value=p_value,
        q_value=None,  # filled in by _apply_bh_correction
        naive_mean_violated=naive_mean_violated, naive_mean_clean=naive_mean_clean,
        naive_delta=naive_delta, rating_mix_violated=rating_mix_violated,
        rating_mix_clean=rating_mix_clean, verdict="UNDERPOWERED",  # re-evaluated after BH
        no_pairing_reason=None,
    )


def _rating_mix(rows: list[dict]) -> dict[str, int]:
    mix: dict[str, int] = {}
    for row in rows:
        rating = row["rating"] or "(none)"
        mix[rating] = mix.get(rating, 0) + 1
    return mix


def _permutation_p_value(
    day_violated_lists: list[list[float]], day_clean_lists: list[list[float]], *, observed: float,
) -> float:
    """Within-day label permutation test: on each resample, within each
    paired day independently, randomly permute the violated/clean labels
    among that day's rows (group SIZES fixed), recompute the paired-mean-
    delta statistic, and count how often |statistic| >= |observed|.
    scipy is not installed — numpy only. Deterministic under `seed`."""
    rng = np.random.default_rng(_PERMUTATION_SEED)
    n_days = len(day_violated_lists)
    # Pre-combine each day's pooled values and group sizes once.
    pooled_per_day = []
    n_violated_per_day = []
    for v, c in zip(day_violated_lists, day_clean_lists):
        pooled_per_day.append(np.array(v + c, dtype=float))
        n_violated_per_day.append(len(v))

    count = 0
    abs_observed = abs(observed)
    for _ in range(_N_PERMUTATIONS):
        total = 0.0
        for day_idx in range(n_days):
            pooled = pooled_per_day[day_idx]
            nv = n_violated_per_day[day_idx]
            perm = rng.permutation(pooled)
            v_mean = perm[:nv].mean()
            c_mean = perm[nv:].mean()
            total += v_mean - c_mean
        stat = total / n_days
        if abs(stat) >= abs_observed:
            count += 1
    return (count + 1) / (_N_PERMUTATIONS + 1)


def _apply_bh_correction(key_stats: list[GuardKeyStats]) -> None:
    """Benjamini-Hochberg FDR correction across every key tested this run,
    applied in place (mutating each GuardKeyStats' q_value via object
    replacement — dataclasses are frozen, see the loop body). Implemented
    directly per task spec: no scipy/statsmodels dependency."""
    testable = [(i, s) for i, s in enumerate(key_stats) if s.p_value is not None]
    if not testable:
        return
    m = len(testable)
    ordered = sorted(testable, key=lambda item: item[1].p_value)
    q_by_index: dict[int, float] = {}
    running_min = 1.0
    for rank in range(m, 0, -1):
        idx, stats_row = ordered[rank - 1]
        candidate = stats_row.p_value * m / rank
        running_min = min(running_min, candidate)
        q_by_index[idx] = running_min

    for idx, stats_row in testable:
        key_stats[idx] = replace(stats_row, q_value=q_by_index[idx])


# ---------------------------------------------------------------------------
# --recompute-pending: sensitivity check, audit-side only (2026-09-29)
#
# Real-account KOSDAQ names stored as '.KS' can stay `status='pending'`
# forever because the production resolver's yfinance lookup fails on '.KS'
# (see app.audit.pending_recompute). Citing only resolved rows silently
# excludes that whole backlog from every key's cohort (confirmed live,
# 2026-09-29: 73/73 pending tickers past the cutoff needed the '.KQ'
# alternate to price at all — exactly the KOSDAQ suffix population, not a
# random subsample), which can both underpower a key and skew its rating_mix
# (the excluded rows are not missing at random). This mode re-measures
# pending rows IN MEMORY (no DB write,
# no reflection) using each row's OWN full_transcript (recorded at analysis
# time, before resolution — confirmed identical shape to a resolved row's)
# plus a recomputed alpha, and rebuilds the WHOLE report (not just the
# affected keys) because Benjamini-Hochberg's q-values depend on every
# tested key's p-value: adding rows to key A changes key B's q too.
#
# `memory_krx_fallback_since` is deliberately never read here: that gate
# controls whether the PRODUCTION resolver writes a resolution and a
# reflection, not whether a row is measurable. Diagnosing "does the pending
# backlog change this finding" must not depend on whether that switch has
# been flipped locally.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GuardAlphaRecomputeReport:
    baseline: GuardAlphaReport
    combined: GuardAlphaReport
    market: str
    scope: str
    account_mode: str
    trigger: str | None
    eval_rule: str
    n_pending_past_cutoff: int  # pending rows past the holding cutoff, BEFORE resolved-sibling shadowing
    n_shadowed_by_resolved: int
    n_recomputed_rows: int  # rows pending_recompute priced successfully
    n_recomputed_unscorable: int  # priced, but _signed_alpha was None (e.g. Hold with unknown held state)
    recomputed_rating_mix: dict[str, int]  # only the rows actually folded into `combined`
    recomputed_added: dict[str, tuple[int, int]]  # key -> (n_violated_added, n_clean_added), keys with >0 only
    n_recompute_failed: int  # price_unavailable + benchmark_unavailable + no_entry_close + fetch_error:*
    n_not_matured: int  # reported separately: not a failure, just too early
    failures_by_reason: dict[str, int]
    failed_tickers: list[str]
    n_alt_symbol_tickers: int
    n_tickers_fetched: int
    n_recomputed_alt_rows: int
    holding_days_min: int | None
    holding_days_median: float | None
    holding_days_max: int | None


async def _fetch_pending_rows(conn, *, market: str, account_mode: str, trigger: str | None, cutoff: date) -> list:
    """Live pending rows old enough to have resolved (`trade_date <= cutoff`),
    latest-created per (ticker, trade_date) — same filters `_fetch_rows` uses
    (status swapped to 'pending', no rating restriction: guard_alpha pools
    every rating, unlike entry_paired's Buy/Hold split)."""
    where_clause, params, _ = _build_where(
        None, market, "live", account_mode, trigger, None, None, status="pending",
    )
    params.append(cutoff)
    query = f"""
        WITH latest AS (
            SELECT DISTINCT ON (ticker, trade_date)
                trade_date, ticker, rating, held_at_decision, held_at_decision_inferred,
                full_transcript, created_at,
                {_GATE_ORIG_RATING_SQL} AS gate_orig_rating
            FROM memory_entries
            WHERE {where_clause} AND trade_date <= ${len(params)}
            ORDER BY ticker, trade_date, created_at DESC
        )
        SELECT * FROM latest
        ORDER BY trade_date, ticker
    """
    return await conn.fetch(query, *params)


async def _fetch_resolved_keys(conn, *, market: str, account_mode: str, trigger: str | None) -> set[tuple[str, date]]:
    """Every (ticker, trade_date) with a resolved sibling under the same
    filters (including `trigger`) -- taken BEFORE the NaN/NULL signed_alpha
    filter `_fetch_rows` applies, so a resolved row that fell out of the
    baseline for that reason still shadows its pending twin."""
    where_clause, params, _ = _build_where(None, market, "live", account_mode, trigger, None, None)
    query = f"SELECT DISTINCT ticker, trade_date FROM memory_entries WHERE {where_clause}"
    rows = await conn.fetch(query, *params)
    return {(r["ticker"], r["trade_date"]) for r in rows}


async def compute_guard_alpha_with_recompute(
    market: str | None = None, scope: str | None = None, account_mode: str | None = None,
    trigger: str | None = None, basis: str = "variable",
) -> GuardAlphaRecomputeReport:
    """Domestic/live only, explicit account_mode required (same reasoning as
    `entry_paired.compute_entry_paired_alpha_with_recompute`: `pending_recompute`
    only knows the KRX session/benchmark, and the production resolver only
    ever touches the CURRENT account mode's rows -- a paper-mode backlog
    stalled by an account-mode switch is a different phenomenon from a
    '.KS'/'.KQ' suffix stall and must not be silently pooled in)."""
    _validate_filters(market, scope, account_mode, trigger, basis)
    if market != "domestic":
        raise ValueError("--recompute-pending is domestic-only (pending_recompute only knows the KRX session/benchmark)")
    if scope != "live":
        raise ValueError("--recompute-pending is live-only (pending backtest rows are not this tool's concern)")
    if account_mode is None:
        raise ValueError("--recompute-pending requires an explicit account_mode ('paper' or 'real')")

    cutoff = local_today("domestic") - timedelta(days=settings.memory_min_holding_days)
    pool = await get_pool()
    async with pool.acquire() as conn:
        async with conn.transaction(isolation="repeatable_read", readonly=True):
            rows, manual_excluded = await _fetch_rows(
                ticker=None, market=market, scope=scope, account_mode=account_mode, trigger=trigger,
                basis=basis, since=None, until=None, conn=conn,
            )
            pending = await _fetch_pending_rows(conn, market=market, account_mode=account_mode, trigger=trigger, cutoff=cutoff)
            resolved_keys = await _fetch_resolved_keys(conn, market=market, account_mode=account_mode, trigger=trigger)

    eligible = [p for p in pending if (p["ticker"], p["trade_date"]) not in resolved_keys]
    n_shadowed = len(pending) - len(eligible)
    pending_by_key = {(p["ticker"], p["trade_date"]): p for p in eligible}

    result = await recompute_pending_alphas(eligible, basis=basis, today=local_today("domestic"))

    rec_rows: list[dict] = []
    n_unscorable = 0
    rating_mix_recomputed: dict[str, int] = {}
    for rec in result.rows:
        pending_row = pending_by_key.get((rec.ticker, rec.trade_date))
        gate_orig_rating = pending_row["gate_orig_rating"] if pending_row is not None else None
        held = pending_row["held_at_decision"] if pending_row is not None else None
        held_inferred = pending_row["held_at_decision_inferred"] if pending_row is not None else None
        signed = _signed_alpha(rec.rating, gate_orig_rating, rec.alpha_return, rec.raw_return, held, held_inferred)
        if signed is None:
            n_unscorable += 1
            continue
        rec_rows.append({
            "trade_date": rec.trade_date, "ticker": rec.ticker, "rating": rec.rating,
            "full_transcript": pending_row["full_transcript"] if pending_row is not None else None,
            "holding_days": rec.holding_days, "signed_alpha": signed,
        })
        rating_mix_recomputed[rec.rating] = rating_mix_recomputed.get(rec.rating, 0) + 1

    baseline = _build_report(rows, basis=basis, manual_excluded=manual_excluded)
    combined_raw = sorted([*rows, *rec_rows], key=lambda r: (r["trade_date"], r["ticker"]))
    combined = _build_report(combined_raw, basis=basis, manual_excluded=manual_excluded)

    all_keys = sorted({s.key for s in baseline.key_stats} | {s.key for s in combined.key_stats})
    recomputed_added: dict[str, tuple[int, int]] = {}
    for key in all_keys:
        classify = _DERIVED_SIGNALS[key] if key in _DERIVED_SIGNALS else _raw_key_classifier(key)
        n_v = n_c = 0
        for row in rec_rows:
            value = classify(_parse_transcript(row["full_transcript"]))
            if value is None:
                continue
            if value:
                n_v += 1
            else:
                n_c += 1
        if n_v or n_c:
            recomputed_added[key] = (n_v, n_c)

    by_reason: dict[str, int] = {}
    for failure in result.failures:
        by_reason[failure.reason] = by_reason.get(failure.reason, 0) + 1
    hard_failures = [f for f in result.failures if f.reason != REASON_NOT_MATURED]
    h_min, h_median, h_max = holding_days_stats(result.rows)
    eval_rule = (
        f"first settled common close on/after trade_date+{settings.memory_min_holding_days}d "
        "(production resolves at the LATEST common close)"
        if basis == "variable"
        else f"{settings.memory_fixed_horizon_trading_days}th trading day (same as production horizon)"
    )

    return GuardAlphaRecomputeReport(
        baseline=baseline, combined=combined, market=market, scope=scope, account_mode=account_mode,
        trigger=trigger, eval_rule=eval_rule,
        n_pending_past_cutoff=len(pending), n_shadowed_by_resolved=n_shadowed,
        n_recomputed_rows=len(rec_rows), n_recomputed_unscorable=n_unscorable,
        recomputed_rating_mix=rating_mix_recomputed, recomputed_added=recomputed_added,
        n_recompute_failed=len(hard_failures), n_not_matured=by_reason.get(REASON_NOT_MATURED, 0),
        failures_by_reason=by_reason, failed_tickers=sorted({f.ticker for f in hard_failures}),
        n_alt_symbol_tickers=result.n_alt_symbol_tickers, n_tickers_fetched=result.n_tickers_fetched,
        n_recomputed_alt_rows=sum(1 for r in result.rows if r.price_symbol != r.ticker),
        holding_days_min=h_min, holding_days_median=h_median, holding_days_max=h_max,
    )


def render_guard_alpha_recompute(r: GuardAlphaRecomputeReport) -> str:
    b, c = r.baseline, r.combined
    lines = [
        "# Guard-vs-realized-alpha -- baseline vs recomputed pending",
        f"market={r.market}  scope={r.scope}  account_mode={r.account_mode}  "
        f"trigger={r.trigger or '(auto, default)'}  basis={b.basis}",
        "",
        f"{'':30}{'baseline(resolved only)':>26}{'recomputed(resolved+recomputed)':>34}",
        f"{'rows':30}{b.n_rows:>26}{c.n_rows:>34}",
        f"{'days (all distinct trade_dates,':30}{'':>26}{'':>34}",
        f"{' NOT paired days)':30}{b.n_days:>26}{c.n_days:>34}",
        f"{'range':30}{f'{b.start}..{b.end}':>26}{f'{c.start}..{c.end}':>34}",
        f"{'keys discovered':30}{b.n_keys_discovered:>26}{c.n_keys_discovered:>34}",
        f"{'derived signals':30}{b.n_derived_signals:>26}{c.n_derived_signals:>34}",
        f"{'keys tested (BH m)':30}{b.n_keys_tested:>26}{c.n_keys_tested:>34}",
        "",
        "## per-key comparison (union of keys seen in either column -- BH m changing moves EVERY key's q,",
        "## even one with zero recomputed rows added)",
        f"{'key':<40}{'b:viol/clean/days':>20}{'b:q':>9}{'b:flag':>13}"
        f"{'c:viol/clean/days':>20}{'c:q':>9}{'c:flag':>13}{'added(v/c)':>13}",
    ]
    b_by_key = {s.key: s for s in b.key_stats}
    c_by_key = {s.key: s for s in c.key_stats}
    all_keys = sorted(set(b_by_key) | set(c_by_key))
    for key in all_keys:
        bs = b_by_key.get(key)
        cs = c_by_key.get(key)
        b_vcd = f"{bs.n_violated}/{bs.n_clean}/{bs.n_paired_days}" if bs else "-"
        c_vcd = f"{cs.n_violated}/{cs.n_clean}/{cs.n_paired_days}" if cs else "-"
        added = r.recomputed_added.get(key)
        added_str = f"{added[0]}/{added[1]}" if added else "0/0"
        lines.append(
            f"{key:<40}{b_vcd:>20}{_fmt_num(bs.q_value if bs else None, 4):>9}{(bs.verdict if bs else '-'):>13}"
            f"{c_vcd:>20}{_fmt_num(cs.q_value if cs else None, 4):>9}{(cs.verdict if cs else '-'):>13}{added_str:>13}"
        )
        if bs is None:
            lines.append("    (new key: only present in recomputed rows)")
        elif bs.underpowered and cs is not None and not cs.underpowered:
            lines.append("    !! POWER GATE CROSSED ONLY WITH RECOMPUTED ROWS -- not a verdict; baseline stays UNDERPOWERED")
        elif bs is not None and cs is not None and not bs.underpowered and not cs.underpowered and bs.verdict != cs.verdict:
            lines.append("    !! VERDICT SENSITIVE TO PENDING ROWS")
        elif added is None or added == (0, 0):
            if bs is not None and cs is not None and bs.q_value != cs.q_value:
                lines.append("    (no recomputed rows for this key; q moved only via BH m)")

    lines += [
        "",
        "## recomputed rows",
        f"eval_rule={r.eval_rule}",
        f"n_pending_past_cutoff={r.n_pending_past_cutoff}  n_shadowed_by_resolved={r.n_shadowed_by_resolved}",
        f"n_recomputed_rows={r.n_recomputed_rows}  n_recomputed_unscorable={r.n_recomputed_unscorable}",
        f"recomputed_rating_mix={_fmt_mix(r.recomputed_rating_mix)}",
        f"n_recompute_failed={r.n_recompute_failed}  n_not_matured={r.n_not_matured}",
        f"alt-symbol: tickers {r.n_alt_symbol_tickers}/{r.n_tickers_fetched}, recomputed rows "
        f"{r.n_recomputed_alt_rows}/{r.n_recomputed_rows}",
        "failures_by_reason=" + (", ".join(f"{k}={v}" for k, v in sorted(r.failures_by_reason.items())) or "(none)"),
        f"holding_days min/median/max={r.holding_days_min}/{r.holding_days_median}/{r.holding_days_max}",
    ]
    if r.failed_tickers:
        lines.append("failed_tickers=" + ", ".join(r.failed_tickers))

    lines += ["", "## baseline", render_guard_alpha(b), "", "## recomputed", render_guard_alpha(c), ""]
    lines += [
        "NOTE: cite the baseline for decisions; recomputed is a sensitivity check only -- the same "
        "UNDERPOWERED/NO_EFFECT/FINDING discipline applies to BOTH columns. Recomputed rows exist only in "
        "memory (not written to DB, no reflection generated) and their eval rule differs from production's "
        "latest-close fallback (see eval_rule above). Pending rows skew toward recent dates and KOSDAQ names "
        "-- always check rating_mix (in the baseline/recomputed dumps above) before trusting a delta shift: "
        "a changed cohort composition, not a guard effect, can produce it. memory_krx_fallback_since is not "
        "read or applied here -- it governs production writes, not what this diagnostic measures.",
    ]
    return "\n".join(lines)


def render_guard_alpha(report: GuardAlphaReport) -> str:
    lines = [
        f"basis={report.basis}  rows={report.n_rows}  days={report.n_days}  "
        f"range={report.start}..{report.end}",
        f"manual-trigger rows excluded: {report.manual_excluded}  "
        f"keys discovered: {report.n_keys_discovered}  "
        f"derived signals registered: {report.n_derived_signals}  "
        f"keys actually tested (BH m, multiple-comparison count): {report.n_keys_tested}",
    ]
    if report.basis == "variable":
        lines.append(
            f"holding_days: min={_fmt_num(report.holding_days_min, 0)} "
            f"max={_fmt_num(report.holding_days_max, 0)} mean={_fmt_num(report.holding_days_mean, 1)}"
        )
        if (
            report.holding_days_min is not None and report.holding_days_max is not None
            and (report.holding_days_max - report.holding_days_min) > 5
        ):
            lines.append(
                "! WARNING: holding_days range spans more than 5 days under basis='variable' — a "
                "variable-window alpha is only comparable across rows when the window is roughly "
                "uniform (see app/memory/db.py horizon_* column comments). Consider --basis horizon."
            )
    lines.append("")

    if not report.key_stats:
        lines.append(
            "No guard keys (`*_violations`/`*_flags`) and no derived signals "
            "(`derived:*`) found in any row's full_transcript."
        )
        return "\n".join(lines)

    header = (
        f"{'key':<40} {'n_viol':>7} {'n_clean':>7} {'n_days':>7} {'delta':>9} "
        f"{'CI95':>20} {'p':>8} {'q':>8}  flag"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for s in report.key_stats:
        flag = s.verdict
        ci_str = _fmt_ci(s.paired_ci95)
        # `derived:` prefix already makes derived signals unmistakable in the
        # `key` column itself (module docstring / _DERIVED_SIGNALS), so no
        # separate marker column is needed — just an explanatory note below.
        lines.append(
            f"{s.key:<40} {s.n_violated:>7} {s.n_clean:>7} {s.n_paired_days:>7} "
            f"{_fmt_num(s.paired_mean_delta, 4):>9} {ci_str:>20} "
            f"{_fmt_num(s.p_value, 4):>8} {_fmt_num(s.q_value, 4):>8}  {flag}"
        )
        if s.key.startswith("derived:"):
            lines.append(
                "    (derived signal, not a raw guard key — see module docstring / "
                "_DERIVED_SIGNALS for its None/excluded semantics)"
            )
        if s.no_pairing_reason:
            lines.append(f"    (no paired days — {s.no_pairing_reason})")
        lines.append(
            f"    naive pooled (CONFOUNDED, ignores trade_date): "
            f"violated_mean={_fmt_num(s.naive_mean_violated, 4)}  clean_mean={_fmt_num(s.naive_mean_clean, 4)}  "
            f"delta={_fmt_num(s.naive_delta, 4)}"
        )
        lines.append(
            f"    rating mix — violated: {_fmt_mix(s.rating_mix_violated)}  "
            f"clean: {_fmt_mix(s.rating_mix_clean)}"
        )
        if s.verdict == "UNDERPOWERED":
            lines.append(
                f"    ! UNDERPOWERED (needs n_paired_days>={_MIN_DAYS_FOR_VERDICT}, "
                f"n_violated>={_MIN_ROWS_PER_GROUP}, n_clean>={_MIN_ROWS_PER_GROUP}, and a p-value) — "
                "do NOT treat this as a finding or change guard priority off this row. Keep collecting."
            )
        elif s.verdict == "NO_EFFECT":
            lines.append(
                "    NO_EFFECT: sample was adequate (all size gates passed) and no association with "
                "realized alpha was detected at this power (q>=0.05) — this is a genuine measurement, "
                "not a data shortage. Treat as a signal to DEPRIORITIZE this guard, not to keep collecting."
            )
    lines.append("")
    lines.append(
        "Reminder: p-values above are within-day label-permutation p-values (seed=0, "
        f"{_N_PERMUTATIONS} resamples); q-values are Benjamini-Hochberg corrected across the "
        f"{report.n_keys_tested} keys that actually produced a p-value this run (out of "
        f"{report.n_keys_discovered} keys discovered — a key with no paired days is never entered "
        "into the correction). A key with q<0.05 but fewer than "
        f"{_MIN_DAYS_FOR_VERDICT} paired days or fewer than {_MIN_ROWS_PER_GROUP} rows per group is "
        "still marked UNDERPOWERED and must not be acted on; a key with an adequate sample and q>=0.05 "
        "is marked NO_EFFECT (a real finding of no detectable association, not a data shortage)."
    )
    return "\n".join(lines)


def _fmt_num(value, digits: int = 3) -> str:
    return f"{value:.{digits}f}" if value is not None else "-"


def _fmt_ci(ci: tuple[float, float] | None) -> str:
    if ci is None:
        return "-"
    return f"[{ci[0]:+.4f}, {ci[1]:+.4f}]"


def _fmt_mix(mix: dict[str, int]) -> str:
    if not mix:
        return "(none)"
    return " / ".join(f"{rating} {count}" for rating, count in sorted(mix.items(), key=lambda kv: -kv[1]))
