"""Market-model residual momentum — the part of a ticker's move that the
broad market/benchmark did NOT explain.

`app.chat.screening.risk_adjusted_score` (today's live ranking formula)
divides a ticker's raw move by its own ATR%% — a Sharpe-like, volatility-
normalized score, but with no adjustment for how much of that move was just
the whole market moving. On a day the index itself rallies 3%%, every
liquid large-cap in the universe tends to print a positive `raw_pct`
regardless of anything specific to that company, and total-momentum ranking
cannot tell "this ticker led the market" apart from "this ticker was
dragged along by it". `residual_pct` computes the market-model residual —
raw move minus `beta` times the benchmark's move over the SAME window — so
the ranking numerator becomes "how much did this ticker move beyond what
the market itself explains", the textbook residual-momentum factor
(Blitz/Huij/Martens and the broader cross-sectional-momentum literature).

WHY this matters concretely in THIS system, not abstractly:
`app.execution.portfolio_risk.compute_correlation_clusters` +
`app.chat.screening.dedupe_correlated` exist to repair, AFTER ranking,
exactly the pathology residual momentum would avoid creating during
ranking — screening-method-improvement-ideas item #4 describes a momentum
leaderboard dominated by one co-moving theme (e.g. several semiconductor
names all surfacing together because the whole sub-sector rallied, not
because any one of them individually outperformed). Dedup can only drop
names ONCE the leaderboard is already themed, and only by discarding a
lower-ranked duplicate outright — it never changes the ranking itself, so
a theme that shares one market-wide driver (the broad index rallying, not
a sector-specific catalyst) still fills most of the pre-dedup pool before
dedup ever gets a chance to act, and can starve the pool of legitimately
different names entirely if the pool multiplier isn't wide enough. Residual
momentum attacks the cause instead of the symptom: if the whole move was
market-beta-mediated, the residual for every affected ticker is small,
so none of them dominate the ranking numerator in the first place.

Dedup stays regardless — it is not being replaced. It is the backstop for
correlation that ISN'T market-beta-mediated: two supply-chain-linked names
(a chipmaker and its dominant equipment supplier, say) can move together
against a flat broad index for reasons that have nothing to do with
market beta, and no benchmark-based adjustment removes that. Residual
momentum and correlation dedup catch two different mechanisms of the same
observed pathology (a themed leaderboard); this module only replaces
`raw_pct` as dedup's ranking INPUT for a caller that opts in
(`momentum_mode="residual"` in app.backtest.screening_sweep), never dedup
itself.

STATUS 2026-09-02: measurement only. Nothing in `app.chat.screening` or
`app.live.auto_trade_loop` imports this module — it exists purely so
`app.backtest.screening_sweep` can A/B residual-vs-total ranking with the
#1 measurement tooling (RankIC, ICIR, excess-return block-bootstrap CI,
n_eff — see app.backtest.factor_metrics) before any live ranking path
changes. Same order #2 (reversal hypothesis, measured and REJECTED — see
project memory screening-reversal-sweep-result-2026-09-02) and #3
(universe-axis diversification, measured, deployed behind a default-off
toggle) already followed: measure first, change the live path only if the
measurement earns it. Nothing here is imported anywhere near a live order.

Why `beta` is fixed at 1.0 here, and a per-ticker OLS beta is deliberately
NOT implemented in this pass: at the live domestic screening lookback
(`settings.momentum_lookback_days_domestic`, 15 calendar days — roughly
9 trading-day return pairs), a 9-observation OLS slope's standard error on
KRX large-cap returns is comparable in magnitude to the slope itself —
this universe's betas mostly sit in a 0.8-1.3 band, and a beta estimate
that noisy would contribute MORE variance to the residual than the actual
beta dispersion it is meant to capture. Fitting an unstable beta on that
little data would not sharpen the residual; it would launder estimation
noise into the ranking numerator and call it signal. This is a
pre-specified, deliberately deferred follow-up (would need `MomentumScore`
to carry the underlying return vector, which would touch
`app.chat.screening`, violating this pass's zero-diff constraint on that
module) — not an oversight, and not stubbed out here: there is no
`estimate_beta` function anywhere in this module.

`benchmark_pct is None` semantics: returns `raw_pct` UNCHANGED — degrades
to total momentum for that one ticker/date, never dropped from
consideration. Same "never fail-closed on one data gap" convention as
`risk_adjusted_score`'s ATR-unavailable fallback and `dedupe_correlated`'s
missing-cluster fallback (both in app.chat.screening) — a benchmark fetch
failure is exactly the same *shape* of gap as a missing ATR or a missing
correlation cluster, and gets the same treatment. The CALLER is
responsible for logging/counting how often this degradation happens — a
"residual" arm that silently fell back to total momentum on most of its
dates isn't actually measuring residual momentum, it's measuring total
momentum with extra steps. `app.backtest.screening_sweep` counts this per
date (`DateObservation.residual_fallbacks`) and renders it
(`VariantResult.residual_fallback_dates`) with a hard-rule warning above
20%% of dates, for exactly this reason.

Both `raw_pct` and `benchmark_pct` must come from the SAME window and the
SAME first-to-last convention as `app.chat.screening.pct_change` (first
price in the sorted date range vs. last) — mixing windows (e.g. a 10-day
ticker move against a 20-day benchmark move) is a silent unit error, not a
different modeling choice. This function does no window bookkeeping itself
by design (it is a pure two-number transform, unit-testable without a
network call, same as `pct_change`/`risk_adjusted_score`); the caller
owns getting the windows to match.
"""

from __future__ import annotations


def residual_pct(raw_pct: float, benchmark_pct: float | None, beta: float = 1.0) -> float:
    """Pure — market-model residual: `raw_pct - beta * benchmark_pct`, the
    part of a ticker's move the benchmark's own move does not explain.
    `benchmark_pct is None` (benchmark fetch unavailable) returns `raw_pct`
    unchanged rather than raising or returning `None` — see the module
    docstring's "never fail-closed on one data gap" section for why, and
    for why the caller — not this function — is responsible for counting
    how often that fallback happens. `beta` defaults to 1.0 (see the module
    docstring for why a per-ticker estimated beta is deliberately deferred,
    not implemented here)."""
    if benchmark_pct is None:
        return raw_pct
    return raw_pct - beta * benchmark_pct
