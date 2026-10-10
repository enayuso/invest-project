"""Whether a ticker could be bought at all this cycle, given the account's
real size — pure domain logic (no I/O, no `app.config`/`app.execution`
imports), same Domain-First convention as `app.domain.product_filter` /
`app.domain.leverage_filter`.

Built for the 2026-09-01 real-account audit, root cause #6: with a
₩200,000 account, 31 of 41 candidates that cycle sized to 0 shares no matter
what the AI decided, and 17 of 41 had a 1-share price above the account's
entire cash — yet every one of them still burned the full ~20-LLM-call
multi-agent pipeline (Market/News/Fundamentals/Sentiment analysts, bull/bear
researchers, three risk debators, research_manager, trader, portfolio_manager).

The four functions below are the SAME arithmetic kernels
`app.execution.ledger`'s `rebalance_to_target`/`resolve_stop_loss` use to
compute the real, binding position size — extracted here (not
reimplemented) so `app.live.auto_trade_loop`'s pre-ranking filter can never
silently drift from what the ledger would actually allow. `ledger.py` calls
these same functions; a change to the real sizing math changes both call
sites at once. See `best_case_buyable_shares`'s docstring for why the
combined check is deliberately the most GENEROUS reading of every input
(never stricter than the real gate) rather than a faithful simulation of
one specific rating/confidence/stop-loss combination.

실계좌 규모로 볼 때 이번 사이클에 이 종목을 애초에 살 수 있는지 계산 —
`ledger.py`의 실제 사이징 산술과 같은 커널을 공유해 필터가 원장과 갈라지지 않게 함.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def allocation_target_shares(equity: float, target_fraction: float, price: float) -> int:
    """`floor(equity * target_fraction / price)`, never negative. Mirrors
    `app.execution.ledger.rebalance_to_target`'s `target_shares = math.floor(
    equity * target_fraction / price)`.

    `price <= 0` returns 0 rather than raising `ZeroDivisionError` — the
    ledger call site never sees a non-positive price in practice (upstream
    guarantees it), so this is a pure strengthening over the inline
    expression it replaces, not an observed behavior change."""
    if equity <= 0 or target_fraction <= 0 or price <= 0:
        return 0
    return max(math.floor(equity * target_fraction / price), 0)


def min_stop_distance(
    price: float, atr_pct: float | None, min_stop_atr_multiple: float
) -> float | None:
    """The ATR-derived floor on stop-loss distance `app.execution.ledger.
    resolve_stop_loss` enforces — `price * atr_pct * min_stop_atr_multiple`,
    clamped to at most `price * 0.95` so extreme volatility (atr_pct >=
    1/multiple) can never push the floor to or past the fill price.

    `None` when `atr_pct` is `None` or non-positive — no ATR data means no
    floor, matching `resolve_stop_loss`'s "degrade, don't fabricate"
    convention: an unresolvable floor is treated as no constraint, never as
    zero risk allowed."""
    if atr_unavailable(atr_pct) or price <= 0:
        return None
    distance = price * atr_pct * min_stop_atr_multiple
    return min(distance, price * 0.95)


def atr_unavailable(atr_pct: float | None) -> bool:
    """The single definition of "this ticker has no usable 14-day ATR%",
    shared by every caller so the predicate can never drift.

    `None` OR non-positive. Not just `None`: `app.dataflows.indicators.
    atr_pct` returns a literal `0.0` when the latest close is 0, and both
    consumers of this value already collapse `0` into the same state as
    `None` — `min_stop_distance` above returns `None` for `atr_pct <= 0`,
    and `app.execution.sizing.vol_adjusted_allocation_for` scales only
    `if current_atr_pct is not None and current_atr_pct > 0`. Gating on
    `is None` alone would leave that hole open."""
    return atr_pct is None or atr_pct <= 0


def blocks_new_entry_without_atr(atr_pct: float | None, *, current_shares: int) -> bool:
    """True when this cycle must refuse to OPEN a position because there is
    no usable ATR to size or protect it with — the fail-closed gate added
    2026-09-02 (decision id=1169, 950260.KS).

    Without ATR, two independent safeties disable silently and at the same
    time: `app.execution.sizing.vol_adjusted_allocation_for` applies no
    volatility scale-down (a Buy keeps its full 1.0 base allocation), and
    `min_stop_distance` above returns `None`, so `app.execution.ledger.
    resolve_stop_loss` neither floors nor validates the LLM's proposed
    stop. The 1% risk cap is not a backstop for that: it derives
    `risk_per_share` FROM that same unvalidated stop, so a tight invented
    stop makes the cap effectively non-binding. Full size, no scaling, no
    floor, no cap — on a name we by definition know least about.

    `current_shares > 0` returns False unconditionally. This gate is about
    NEW exposure only. A held position already carries a resolved stop and
    the trailing-stop machinery; refusing to act on it would be a
    fail-OPEN on risk (trapping a position we cannot value), and this
    codebase never forces a sale of shares already held — see
    `app.execution.ledger.rebalance_to_target`'s 1%-risk-cap comment and
    `app.live.auto_trade_loop._drop_unexecutable`'s "a currently-held
    ticker is always kept".

    Pure predicate, no settings read — the kill switch
    (`settings.auto_trade_require_atr_for_new_entry`) is checked by each
    caller, since this module imports no `app.config` (Domain-First)."""
    return current_shares <= 0 and atr_unavailable(atr_pct)


def risk_capped_shares(equity: float, risk_per_share: float, max_risk_pct: float) -> int | None:
    """The "1% rule" cap on shares addable this trade —
    `floor(equity * max_risk_pct / risk_per_share)`, never negative. Mirrors
    `app.execution.ledger.rebalance_to_target`'s `max_incremental_shares =
    max(math.floor(max_dollar_risk / risk_per_share), 0)`.

    `None` when `risk_per_share <= 0` — the ledger only ever computes this
    cap inside `if risk_per_share > 0:`; when that's not true there IS no
    risk-per-share to bound against, so `None` means "uncapped by this
    rule," not zero. `equity <= 0` is NOT special-cased to `None` here
    (unlike the other three kernels in this module) — the ledger call site
    would compute `max_dollar_risk = 0` and floor to 0 shares in that case,
    and this function preserves that exact (restrictive, not permissive)
    behavior rather than the "degrade to unconstrained" meaning `None`
    otherwise carries."""
    if risk_per_share <= 0:
        return None
    return max(math.floor(equity * max_risk_pct / risk_per_share), 0)


def cash_affordable_shares(cash: float, price: float) -> int:
    """`floor(cash / price)`, never negative. No ledger equivalent — the
    ledger's sizing math is expressed in shares of equity, not a cash
    balance check; `app.execution.kis_bridge` gates cash separately, in
    order value, at mirror time. This exists only for the pre-ranking
    filter, which needs a same-currency cash ceiling before any order is
    ever built."""
    if cash <= 0 or price <= 0:
        return 0
    return max(math.floor(cash / price), 0)


@dataclass(frozen=True)
class ExecutabilityInputs:
    """`price`/`atr_pct` describe the candidate ticker; `equity`/`cash`
    describe the account, in the SAME currency as `price` (never converted
    here — the caller is responsible for querying the right market's
    account snapshot for this ticker's currency)."""

    price: float
    atr_pct: float | None
    equity: float
    cash: float


@dataclass(frozen=True)
class SizingLimits:
    """The three `risk_rules.rule(...)` values this needs, passed in rather
    than read here — this module stays free of `app.execution`/`app.config`
    imports (Domain-First: settings/DB-backed overrides are an adapter
    concern, not a domain one). `max_allocation_fraction` is the caller's
    own `app.execution.sizing.vol_adjusted_allocation_for(BUY, atr_pct)`
    result — BUY at full (1.0) confidence is the most generous rating this
    system's sizing ever produces, so it's the right anchor for "the best
    case this ticker could ever get," matching this module's "never
    stricter than the real gate" posture."""

    max_allocation_fraction: float
    max_risk_pct_of_equity: float
    min_stop_atr_multiple: float


def affordable_price_ceiling(
    equity: float,
    cash: float,
    assumed_atr_pct: float,
    limits: SizingLimits,
    *,
    slack_multiple: float = 1.0,
) -> float | None:
    """The inverse of `best_case_buyable_shares`: not "how many shares of
    THIS ticker can I buy", but "what is the highest PRICE a ticker could
    have and still be sizeable to >=1 share", for one assumed ATR% applied
    to every ticker a discovery-time ranking might return. Built 2026-09-02
    (screening-improvement-research #3 follow-up) for
    `app.live.auto_trade_loop._domestic_price_ceiling`, which needs this
    number BEFORE any candidate exists to look an ATR up for — see that
    function's own docstring for the chicken-and-egg this breaks.

    `best_case_buyable_shares` computes three caps on SHARE COUNT from a
    known price (`allocation_target_shares`, `risk_capped_shares` via
    `min_stop_distance`, `cash_affordable_shares`) and takes the minimum.
    This function runs each of those three derivations backwards to solve
    for the highest PRICE at which the corresponding cap would still be
    >=1 share, and takes the minimum of the three prices — the same
    "never stricter than the real gate" posture as `best_case_buyable_
    shares`, just solved for the other variable:

        cap            | price <= ...                          | derived from
        ---------------|----------------------------------------|------------------------
        allocation     | equity * max_allocation_fraction        | allocation_target_shares
        risk           | equity * max_risk_pct_of_equity /       | risk_capped_shares +
                        |   stop_fraction * slack_multiple        | min_stop_distance
        cash           | cash                                    | cash_affordable_shares

    `stop_fraction` mirrors `min_stop_distance`'s own `min(distance, price *
    0.95)` clamp, re-expressed as a fraction of price (`assumed_atr_pct *
    min_stop_atr_multiple`, clamped to at most 0.95) so it can be divided
    into rather than multiplied by — see that function's docstring for why
    the 0.95 ceiling exists (extreme volatility must never push the stop
    floor to or past the fill price). Only the RISK term gets
    `slack_multiple`: the cash and allocation terms are exact given the
    assumed ATR (no ATR assumption enters either derivation at all — cash
    is a flat ceiling and allocation only ever depends on equity and the
    allocation fraction), so widening them would be pure error, not
    caution. The risk term is the only one where the ASSUMPTION
    (`assumed_atr_pct`) does the work, so it's the only one slack can
    honestly buy back.

    Call this with `assumed_atr_pct = risk_rules.rule("sizing_target_atr_pct")`
    specifically, not some other guess, and pair it with
    `max_allocation_fraction=1.0` — not because 1.0 is a second assumption
    stacked on the first, but because at exactly the target ATR,
    `app.execution.sizing.vol_adjusted_allocation_for` applies NO
    volatility scale-down by construction (`min(1.0, atr_target /
    current_atr_pct)` reduces to `min(1.0, 1.0) == 1.0` at
    `current_atr_pct == atr_target`, and any ticker with a LOWER real ATR
    than the target would get exactly the same multiplier of 1.0). So
    `max_allocation_fraction=1.0` is not a separate guess riding on the ATR
    guess — it is what the real sizing path actually returns for any name
    at or below the assumed ATR. The two assumptions collapse into one.

    `slack_multiple` and the asymmetry it exists for: this function assumes
    ONE ATR for a whole ranking of tickers it has never individually
    quoted — the real ATR of whichever ticker ends up admitted is only
    knowable AFTER it's already in the pool, which is exactly the
    chicken-and-egg this exists to break. If the real ATR turns out HIGHER
    than assumed, the TRUE ceiling is LOWER than what this function
    computed — the ceiling wrongly ADMITS a name into the KIS-side ranking
    that then gets correctly dropped by
    `app.live.auto_trade_loop._drop_unexecutable` once its real ATR is
    known. That costs one price+ATR quote and no LLM call — cheap. If the
    real ATR turns out LOWER than assumed, the TRUE ceiling is HIGHER than
    computed — the ceiling wrongly EXCLUDES a name that could have been
    traded, and this is never recoverable: a KIS ranking TR bounded by
    `FID_INPUT_PRICE_2` simply never returns a ticker priced above the
    ceiling it was called with, so there is no downstream chance to
    reconsider it. False-admit is cheap and self-correcting; false-exclude
    is silent and permanent. That asymmetry is why the default
    (`slack_multiple=1.0` here, `settings.
    auto_trade_universe_price_ceiling_slack_multiple=1.5` at the call site)
    is deliberately GENEROUS, not tight — a future reader should not
    "tighten" this value to make the filter look sharper; sharper here
    means quietly losing affordable names forever.

    `None` return semantics: means "no ceiling should be applied, this
    account cannot usefully constrain the ranking", not "ceiling is zero" —
    same convention as `risk_capped_shares`/`min_stop_distance` elsewhere
    in this module. Triggered by any non-positive `equity`/`cash`/
    `assumed_atr_pct`/`slack_multiple`, or any non-positive
    `limits.max_risk_pct_of_equity`/`limits.min_stop_atr_multiple`/
    `limits.max_allocation_fraction` — every one of those would make at
    least one of the three price terms above meaningless (a zero or
    negative bound, or a division by zero).

    See `app.live.auto_trade_loop._domestic_price_ceiling` for the actual
    caller: it resolves `equity`/`cash` from `cached_account_snapshot`,
    reads the three `SizingLimits` fields via `risk_rules.rule(...)` (never
    `settings.` directly, for the same anti-drift reason
    `_drop_unexecutable` already reads them that way), and applies its own
    min/max sanity floor/ceiling on the result before ever putting it on
    the wire as `FID_INPUT_PRICE_2`."""
    if equity <= 0 or cash <= 0 or assumed_atr_pct <= 0 or slack_multiple <= 0:
        return None
    if limits.max_risk_pct_of_equity <= 0 or limits.min_stop_atr_multiple <= 0:
        return None
    if limits.max_allocation_fraction <= 0:
        return None
    stop_fraction = min(assumed_atr_pct * limits.min_stop_atr_multiple, 0.95)
    risk_ceiling = equity * limits.max_risk_pct_of_equity / stop_fraction * slack_multiple
    allocation_ceiling = equity * limits.max_allocation_fraction
    return min(risk_ceiling, allocation_ceiling, cash)


@dataclass(frozen=True)
class ExecutabilityVerdict:
    max_buyable_shares: int
    allocation_cap_shares: int
    risk_cap_shares: int | None  # None = no ATR data -> uncapped by this rule
    cash_cap_shares: int
    binding_constraint: str  # "allocation" | "risk" | "cash" -- whichever cap is smallest, win or lose
    reason: str


def best_case_buyable_shares(
    inputs: ExecutabilityInputs, limits: SizingLimits
) -> ExecutabilityVerdict:
    """The most this ticker could ever be sized to buy this cycle, under the
    single most generous reading of every input this system's real sizing
    math would ever produce: maximum confidence (BUY, `max_allocation_fraction`
    already reflects that), a stop-loss sitting exactly on the ATR floor
    (the tightest stop `resolve_stop_loss` will ever actually allow — a
    tighter AI-proposed stop gets widened to this floor, never honored), and
    every OTHER real-execution constraint ignored (portfolio concentration/
    correlation/drawdown caps, the domestic cash floor, the per-order value
    cap, transaction costs) — none of those can ever make the real result
    LARGER than what's computed here, only smaller.

    So `max_buyable_shares == 0` means "not affordable no matter what the AI
    decides this cycle" with no false positives: the real execution chain
    can only end up at this number or below it, never above."""
    allocation_cap = allocation_target_shares(inputs.equity, limits.max_allocation_fraction, inputs.price)
    stop_distance = min_stop_distance(inputs.price, inputs.atr_pct, limits.min_stop_atr_multiple)
    risk_cap = (
        risk_capped_shares(inputs.equity, stop_distance, limits.max_risk_pct_of_equity)
        if stop_distance is not None
        else None
    )
    cash_cap = cash_affordable_shares(inputs.cash, inputs.price)

    caps = {"allocation": allocation_cap, "cash": cash_cap}
    if risk_cap is not None:
        caps["risk"] = risk_cap
    binding_constraint, max_shares = min(caps.items(), key=lambda kv: kv[1])

    risk_cap_text = f"{risk_cap}주" if risk_cap is not None else "무제한(ATR 없음)"
    reason = (
        f"배분캡 floor({inputs.equity:,.0f}*{limits.max_allocation_fraction:.0%}/{inputs.price:,.2f})="
        f"{allocation_cap}주, 리스크캡={risk_cap_text}, "
        f"현금캡 floor({inputs.cash:,.0f}/{inputs.price:,.2f})={cash_cap}주 "
        f"→ 최소 {max_shares}주 (구속: {binding_constraint})"
    )

    return ExecutabilityVerdict(
        max_buyable_shares=max_shares,
        allocation_cap_shares=allocation_cap,
        risk_cap_shares=risk_cap,
        cash_cap_shares=cash_cap,
        binding_constraint=binding_constraint,
        reason=reason,
    )


def is_unexecutable(inputs: ExecutabilityInputs, limits: SizingLimits) -> bool:
    """`True` when this ticker cannot be sized to even 1 share this cycle,
    under the most generous reading of every input — see
    `best_case_buyable_shares`."""
    return best_case_buyable_shares(inputs, limits).max_buyable_shares <= 0
