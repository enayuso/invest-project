"""Rating -> target allocation mapping — position sizing enforced in code,
not the Trader's loose `position_sizing` prose string.

Long-only: Sell means 0% allocation, not a short position. No margin/
shorting in this project's scope (consistent with other v1 scope cuts —
multi-asset config and a custom DL model were both deferred for the same
reason); revisit only if a real gap shows up in practice.

Volatility adjustment: if `current_atr_pct` is provided, the base allocation
is scaled down when volatility exceeds the target; it never scales up above
the base allocation (no leverage).

Confidence adjustment: if `confidence` (PortfolioDecision.confidence) is
provided, the same down-only philosophy applies — a low-conviction call gets
a smaller position, but confidence can never inflate a position beyond the
rating's own base allocation. Applied uniformly regardless of direction
(same as the ATR adjustment above), not just when adding exposure: a
low-confidence Underweight/Sell still shrinks toward the base only, it
doesn't force extra selling beyond what the rating already implies, since
`rebalance_to_target`'s own caps (app/execution/ledger.py) are what's
actually direction-aware.

A missing (None) confidence is treated as the LOWEST confidence band, i.e.
as a skip — not as "no adjustment". See `confidence_size_modifier`.

2026-09-10 (실계좌 사이징 이원화 조사): 이 모듈이 반환하는 목표비중은 대부분의 실거래
상황에서 실제로 관철되지 않는다 — `app.execution.ledger.rebalance_to_target`의 1% 리스크
캡(`sizing_max_risk_pct_of_equity` / `sizing_min_stop_atr_multiple`)이 ATR >=
`sizing_target_atr_pct`(기본 2%)인 모든 종목에서 이 모듈의 결과보다 항상 더 타이트하게
작동한다.

  이 모듈의 결과 (2%~8% ATR 구간, 25% 바닥에 안 걸린 경우) = base × (2% / ATR)
  리스크 캡이 허용하는 비중                                  = 1% / (2.5 × ATR)

두 식의 비율은 ATR 값과 무관하게 정확히 5.0배(base=1.0 기준, Overweight/Underweight처럼
base가 1.0 미만이면 격차는 더 벌어진다 — 리스크 캡 쪽은 base와 무관하게 고정이므로) —
Buy(100%)든 Overweight(75%)든 confidence 배율이 몇이든, ATR이 2% 이상인 한 실제 진입
비중을 정하는 건 거의 항상 리스크 캡이지 이 모듈이 아니다. ATR이 8%를 넘어 이 모듈이
25% 바닥(`sizing_min_vol_multiplier`)에 걸리면 격차는 더 벌어진다(ATR 20%에서 12.5배).

2026-09-10 실계좌 8건(2026-09-03~09-10) 감사에서 실측: 의도한(등급+ATR+confidence 반영)
비중 대비 실제 체결 비중이 2.6~6.0배 축소돼 있었음 — 버그가 아니라 이 산술적 구조가
설계대로 작동한 결과. 즉 이 모듈은 사실상 "리스크 캡이 없다면 얼마나 살 것인가"라는
상한선/방향성 신호이지, ATR >= 2%인 종목의 실제 사이징 결정자가 아니다. 방향(등급)과
크기(리스크캡)를 분리해서 인지할 것 — 등급을 더 세밀하게 조정해도 최종 체결 비중은
거의 안 바뀐다. 실제 사이징을 등급에 연동시키려면 리스크 예산 자체를 등급별로
차등화해야 하며(별도 설계·사용자 승인 필요), 이 문서화는 동작을 바꾸지 않은 상태에서
그 사실을 명시하는 것만을 목적으로 한다. 상세: memory
sizing-dual-track-audit-2026-09-10.
"""

from __future__ import annotations

import logging

from app.config import settings
from app.state import PortfolioRating
from app.execution import risk_rules

logger = logging.getLogger(__name__)

_BASE_ALLOCATION: dict[PortfolioRating, float | None] = {
    PortfolioRating.BUY: 1.0,
    PortfolioRating.OVERWEIGHT: 0.75,
    PortfolioRating.HOLD: None,  # unchanged - no rebalancing trade
    PortfolioRating.UNDERWEIGHT: 0.25,
    PortfolioRating.SELL: 0.0,
}

# Model Confidence -> Position Size Modifier. Below the first threshold means
# "not confident enough to trade at all" (0.0x); above the last means "fully
# confident" (1.0x) — deliberately capped there, since confidence is a
# measure of the model's own conviction, not a real accuracy guarantee, and
# should never be read as license to size up beyond the rating's own base
# allocation. Thresholds are lower-bound-inclusive.
_CONFIDENCE_MODIFIER_BANDS: list[tuple[float, float]] = [
    (0.90, 1.0),
    (0.75, 0.9),
    (0.60, 0.7),
    (0.50, 0.5),
]


def confidence_size_modifier(confidence: float | None) -> float:
    """Map a 0.0-1.0 model-confidence value to a position-size multiplier.
    `confidence` below 0.5 returns 0.0 (no trade) rather than a small
    nonzero size — an unconfident call is a reason to skip, not to shrink.

    `None` (the model omitted the field) returns 0.0, exactly as a stated
    0.0 would: a missing self-assessment is the worst case, not a neutral
    one. Reading it as "no adjustment" (the behavior before 2026-08-20) gave
    the model a direct incentive to omit the field — an honest 0.65 was cut
    to 0.7x while a silent omission kept full size (real incident 2026-08-19:
    GOOG confidence 0.65 -> 0.7x, GOOGL unset -> full 49%). This also makes
    app.state.PortfolioDecision.confidence's own field description ("an unset
    value is treated as the lowest possible confidence") literally true, and
    matches app.orchestrator.pipeline._apply_hysteresis_gate, which already
    treats None as below its bar rather than exempt from it."""
    if confidence is None:
        return 0.0
    for threshold, modifier in _CONFIDENCE_MODIFIER_BANDS:
        if confidence >= threshold:
            return modifier
    return 0.0


def vol_adjusted_allocation_for(
    rating: PortfolioRating,
    current_atr_pct: float | None = None,
) -> float | None:
    """The rating's base allocation with ONLY the volatility (ATR)
    adjustment applied — the answer to "how will ATR scaling change each
    rating?", a different question from "what should this position's target
    actually be" (`target_allocation_for`). Returns None for Hold, same as
    `target_allocation_for`.

    Split out deliberately: `_atr_context` (app/orchestrator/pipeline.py)
    previews these numbers to the Portfolio Manager BEFORE the model has
    stated any confidence, so it genuinely has none to pass. Before this
    split, that caller expressed "no confidence yet" by omitting the
    argument — the same way the execution path expressed "the model gave no
    confidence" — so the execution path could not safely read a missing
    confidence as a low one. Two callers, two questions, two functions; no
    sentinel, no flag, no ambiguous default."""
    base = _BASE_ALLOCATION[rating]
    if base is None:
        return None  # Hold - no rebalancing trade, no adjustment applies
    if current_atr_pct is not None and current_atr_pct > 0:
        # Down-only adjustment: never exceed the base allocation.
        atr_target = risk_rules.rule("sizing_target_atr_pct")
        min_mult = risk_rules.rule("sizing_min_vol_multiplier")
        base *= max(min_mult, min(1.0, atr_target / current_atr_pct))
    return base


def target_allocation_for(
    rating: PortfolioRating,
    current_atr_pct: float | None = None,
    *,
    confidence: float | None,
) -> float | None:
    """Return the target fraction of equity to allocate to this position,
    adjusted for current volatility and model confidence.

    If `current_atr_pct` is None, no volatility adjustment is applied. If
    provided, scales the base allocation down (only) when volatility exceeds
    `sizing_target_atr_pct`, floored at `sizing_min_vol_multiplier` × base.
    (Computed by `vol_adjusted_allocation_for`, which is also what callers
    that want the ATR-only number — with no confidence in hand yet — should
    use directly.)

    `confidence` is keyword-only and REQUIRED — there is no "just don't pass
    it" path, because there is no honest default. Passing None means "the
    model stated no confidence", which is scored as the LOWEST band (modifier
    0.0 -> skip, see `confidence_size_modifier`), NOT as "no adjustment".
    The old None-means-no-adjustment contract rewarded omitting the field
    with a full-size position while an honestly low number got shrunk; see
    `confidence_size_modifier`'s docstring for the incident. The confidence
    and volatility adjustments compose (multiply), each only ever reducing
    the result.

    Below-threshold confidence (modifier 0.0) on a rating that itself wants
    a position (BUY/OVERWEIGHT/UNDERWEIGHT, base > 0) returns `None` -
    "skip the trade", same as Hold - rather than `0.0`. `rebalance_to_target`
    only treats `None` as skip; `0.0` is itself a valid target (Sell's own
    base allocation) meaning "liquidate to flat". Without this distinction,
    a low-confidence Buy/Overweight/Underweight silently became a forced
    full liquidation of any existing position - the PM prompt already tells
    the model low confidence "skips the trade entirely", so the code must
    match that, not read it as an order to sell out. Sell's own base
    allocation is already 0.0, so this leaves Sell unaffected either way.
    """
    target = vol_adjusted_allocation_for(rating, current_atr_pct)
    if target is None:
        return None  # Hold - no rebalancing trade, no adjustment applies

    modifier = confidence_size_modifier(confidence)
    if modifier == 0.0 and _BASE_ALLOCATION[rating] != 0.0:
        if confidence is None:
            logger.warning(
                "sizing skipped a %s: the model stated no confidence, which is scored as the "
                "lowest band (no trade). See app.execution.sizing.confidence_size_modifier.",
                rating.value,
            )
        return None  # not confident enough to trade at all - skip, don't liquidate
    return target * modifier


# Machine-readable, English (matches PortfolioRating's own enum-value
# capitalization) — frontend localizes via its own _ACTION_KO-style map,
# same pattern as _RATING_KO for PortfolioRating.
def rebalance_action_label(current_shares: int, target_shares: int) -> str:
    """What a rebalance ACTUALLY did to the position, derived purely from
    current vs (post-risk-cap) target share count — deliberately separate
    from the AI's own PortfolioRating (Buy/Overweight/Hold/Underweight/Sell),
    which is a conceptual judgment about conviction, not a literal
    current-vs-target comparison. The two can disagree: an Underweight
    (25%-of-full-conviction) rating on a position currently far BELOW even
    that reduced target still nets out as a BUY here, which is exactly the
    real case (에이치엘비, 2026-07-24) that prompted adding this — a rating
    pill reading "비중 축소" next to an order that actually bought 417
    shares read as contradictory even though both were individually
    correct. Rating and this label answer different questions (target
    conviction vs. resulting action) and are shown side by side, not
    merged."""
    if target_shares == current_shares:
        return "Hold"
    if current_shares <= 0:
        return "Buy"
    if target_shares <= 0:
        return "Sell"
    return "Increase" if target_shares > current_shares else "Decrease"