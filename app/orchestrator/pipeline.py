"""Wires the full chain: concurrent analysts -> bull/bear debate -> research
manager -> trader -> risk debate -> portfolio manager -> execution. Single
ticker. Phase 3 wires the memory/reflection store around the Portfolio
Manager only, per the spec's "funnel memory to exactly one downstream node"
decision (validated empirically — the Portfolio Manager reasons well citing
past lessons on its own; broadcasting past_context to every judgment node
was tried and deliberately not adopted, see project memory for the
reasoning). Phase 4 turns the resulting decision into an actual simulated
trade, position-sized (volatility-adjusted via ATR%) and stop-loss enforced
in code.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from datetime import date
from typing import Any, Literal

from app.agents import (
    bear_researcher,
    bull_researcher,
    fundamentals_analyst,
    market_analyst,
    news_macro_analyst,
    portfolio_manager,
    research_manager,
    risk_debators,
    sentiment_analyst,
    trader,
)
from app.agents.absence_reframing_guard import scan_absence_reframing
from app.agents.business_description_guard import scan_business_description
from app.agents.derived_value_mismatch_guard import scan_derived_value_mismatches
from app.agents.desk_consensus_narrative_guard import scan_desk_narrative
from app.agents.halt_status_guard import scan_halt_claims
from app.agents.price_limit_guard import scan_price_limit_claims
from app.agents.refusal_guard import scan_refusal
from app.agents.regulatory_mechanism_guard import scan_regulatory_mechanism
from app.agents.ground_truth_claim_guard import extract_ground_truth, scan_ground_truth_claims
from app.agents.session_close_guard import scan_session_close_claims
from app.agents.sizing_cap_misuse_guard import scan_sizing_cap_misuse
from app.agents.staleness_narrative_guard import scan_staleness_narrative
from app.agents.technical_claim_guard import scan_technical_claims
from app.agents.verified_price_guard import scan_verified_price_claims
from app.agents.identity_gate import UNNAMED as IDENTITY_UNNAMED
from app.agents.identity_gate import build_alias_set, enforce_identity
from app.agents.numeric_guard import enforce_numeric_guards
from app.agents.period_mismatch_guard import scan_period_mismatch
from app.agents.unit_scale_guard import extract_kr_headline_eok_values
from app.agents.tool_loop import ticker_label
from app.cancellation import task_was_actually_cancelled
from app.config import AIProfile, is_backtest_context, settings
from app.dataflows.kis_provider import _is_us, candle_run_cache, get_atr_pct, get_kis_english_name, get_kr_risk_signals, get_last_close, get_ticker_display_info
from app.dataflows.session_state import intraday_narration_note
from app.dataflows.yfinance import get_verified_snapshot
from app.dataflows.yfinance.market_data import get_ticker_display_info as yf_get_ticker_display_info
from app.domain import executability
from app.live import krx_master, overseas_master, ticker_names
from app.execution import runtime_settings
from app.execution.kis_bridge import available_cash_native, execution_blockers
from app.execution.ledger import (
    REVERSAL_REASON_PREFIX,
    STOP_LOSS_REASON_PREFIX,
    EquityBasisUnavailable,
    get_last_trade,
    get_open_stop_loss,
    get_position,
    is_threshold_placeholder_note,
    rebalance_to_target,
    resolve_equity_basis,
)
from app.execution.portfolio_risk import (
    DISABLED_PORTFOLIO_CAPS,
    AccountUnavailable,
    PortfolioSnapshot,
    cached_account_snapshot,
    cached_balance_fetches,
    compute_correlation_clusters,
    current_drawdown_fraction,
    real_account_snapshot,
    real_position_detail,
    real_shares_held,
    ticker_market,
)
from app.execution.sizing import target_allocation_for, vol_adjusted_allocation_for
from app.llm import metrics as llm_metrics
from app.llm.request_capture import capture_research_manager
from app.llm.language import DEFAULT_LANGUAGE
from app.memory.store import (
    append_execution_note,
    find_recent_pending_id,
    get_same_day_sibling_context,
    get_similar_context,
    set_action_label,
    store_pending,
)
from app.orchestrator.debate import run_debate
from app.execution import risk_rules
from app.state import (
    FlatPortfolioDecision,
    FlatPortfolioRating,
    PipelineState,
    PortfolioRating,
    is_blank_card_reason,
    is_state_flipping_rating,
    render_portfolio_decision,
    render_research_plan,
    render_trader_proposal,
)

logger = logging.getLogger("app.orchestrator")

# Shared PM-narrative-contradiction notice (2026-09-11): every one of the ten
# post-PM sweep guards (session_close, regulatory_mechanism, ground_truth_
# claim, desk_narrative, sizing_cap_misuse, price_limit, verified_price,
# absence_reframing, derived_value_mismatch, halt_status) is detect-only — a
# PM-attributed finding is logged and persisted into the transcript, but the
# PM's own investment_thesis never reflects that its own text was flagged.
# APPENDED (never prefixed — the prefix region of executive_summary belongs
# to the four downgrade gates: _apply_consensus_override_gate /
# _apply_unjustified_exit_gate / _apply_hysteresis_gate /
# _apply_degradation_notice) to investment_thesis only, never
# executive_summary. Same HARD CONSTRAINT as unit_scale_guard's
# _CURRENCY_INVERTED_BANNER / _NEWS_EOK_NOTE / _DERIVED_USD_STALE_NOTE: no
# `$`, no digit, no won/KRW token — identity_gate's foreign-listing
# Detector C counts KRW-shaped/USD-shaped figures across the whole report,
# and this text must never perturb that count.
_PM_CONTRADICTION_NOTICE: dict[str, str] = {
    "en": "\n\n[Note: part of this decision's own narrative was flagged by an automated "
          "consistency check this cycle as unverified or inconsistent with the supplied "
          "data — treat the flagged portion with caution.]",
    "ko": "\n\n[안내: 이 판단 서술의 일부는 이번 사이클에서 자동 일관성 점검에 의해 검증되지 않았거나 "
          "제공된 데이터와 불일치하는 것으로 표시되었습니다 — 해당 부분은 주의해서 참고하십시오.]",
}


def _apply_hysteresis_gate(decision, held: bool):
    """Downgrades a Buy (flat -> entering a new position) or Sell (held ->
    fully exiting) to Hold when the model's own stated confidence doesn't
    clear `hysteresis_min_confidence_to_flip` — see app.config's own
    docstring on that setting for the reasoning (ai-engine-24h-operation-
    analysis P0-1). Only these two STATE-CHANGING transitions are gated;
    Overweight/Underweight (adjusting a position already open) and Hold
    itself pass through untouched — the round-trip-loss incident this
    addresses ([[rebalance-cadence-separation-done]]) was specifically about
    entering and fully exiting on borderline conviction, not about sizing
    adjustments to an existing position.

    Missing confidence (None) is treated as below the bar, not exempt from
    it — "the model had no basis to estimate confidence" is a reason to be
    MORE cautious about a state change, not a free pass around this gate.

    The full numbered note goes on `investment_thesis`, byte-identical to
    before (`app.experiments.language_ab_report._HYSTERESIS_NOTE_RE`/
    `pre_gate_rating` parse this exact format — do not reformat it without
    updating that regex too). But `investment_thesis` is deliberately
    excluded from `render_portfolio_decision`'s short default view (see that
    function's own docstring) and from `trades.reason` — so before this fix,
    the ONE place a human or another system actually sees this decision
    showed "Rating: Hold" next to an `executive_summary` still arguing the
    ORIGINAL (pre-gate) case, with no visible indication anything was
    overridden. Real incident this fixes: paired with a ~2/3 confidence-
    missing rate observed for Korean-language runs (memory:
    ai-reentry-context-2026-08-14's sibling finding from the same
    investigation), this silently produced Hold decisions whose only visible
    text argued Buy — see memory: ai-reentry-context-2026-08-14 for the
    full writeup. Fixed by PREFIXING (not appending — `executive_summary`
    is clamped to 6 lines on the dashboard card, see its own Field
    description; a suffix would just get cut off) a short, number-free
    variant onto `executive_summary` too. Number-free deliberately:
    `language_ab_report.py` scans `executive_summary + investment_thesis`
    together for prose-vs-structured numeric mismatches, and the full note
    already carries the confidence/threshold percentages once, in
    `investment_thesis` — repeating them here would double-count."""
    is_flip = is_state_flipping_rating(decision.rating, held)
    if not is_flip:
        return decision
    threshold = risk_rules.rule("hysteresis_min_confidence_to_flip")
    if decision.confidence is not None and decision.confidence >= threshold:
        return decision
    original_rating = decision.rating.value
    confidence_note = f"{decision.confidence:.0%}" if decision.confidence is not None else "unset"
    hysteresis_note = (
        f"\n\n[Hysteresis gate: downgraded from {original_rating} to Hold — confidence "
        f"({confidence_note}) below the {threshold:.0%} bar required to flip position state.]"
    )
    # Short, number-free variant for the short-form view — see this
    # function's own docstring for why this is a PREFIX, not a suffix, and
    # why it deliberately omits the confidence/threshold percentages.
    gate_prefix = (
        f"[Hysteresis gate: downgraded from {original_rating} to Hold — "
        f"below the confidence bar to flip position state.] "
    )
    hold_rating = FlatPortfolioRating.HOLD if not held else PortfolioRating.HOLD
    return decision.model_copy(
        update={
            "rating": hold_rating,
            "executive_summary": gate_prefix + decision.executive_summary,
            "investment_thesis": decision.investment_thesis + hysteresis_note,
        }
    )


def _consensus_override_tier(decision, research_plan, trader_proposal) -> str | None:
    """The desk-consensus override gate's classifier (2026-09-04 incident,
    id=1212/1217). Returns None — the gate does not apply — which is the
    common case and ALWAYS the case for a rating that does not increase
    exposure. It fires only when the Portfolio Manager chose Buy or
    Overweight while BOTH upstream signals had declined more exposure: the
    Research Manager concluded Hold/Underweight/Sell AND the Trader proposed
    Hold/Sell. In that situation the PM is overruling the whole desk on the
    strength of the risk debate alone (which saw nothing the Trader did not) —
    permitted per the system prompt, but only on a specific new fact, and
    held here to a higher confidence bar than an ordinary call.

    Returns "bearish" (the stricter tier) when the desk was actively negative
    — RM Sell or Underweight AND Trader Sell — and "neutral" otherwise. Never
    fires on Hold/Underweight/Sell from the PM: this gate only ever downgrades
    an exposure INCREASE, never a reduce or a hold (same narrow scope as
    _apply_hysteresis_gate)."""
    if decision.rating.value not in ("Buy", "Overweight"):
        return None
    if research_plan is None or trader_proposal is None:
        return None
    if trader_proposal.action.value not in ("Hold", "Sell"):
        return None
    if research_plan.recommendation.value not in ("Hold", "Underweight", "Sell"):
        return None
    if research_plan.recommendation.value in ("Sell", "Underweight") and trader_proposal.action.value == "Sell":
        return "bearish"
    return "neutral"


def _apply_consensus_override_gate(decision, research_plan, trader_proposal, *, record_out: dict):
    """Structural backstop for the desk-consensus override rule the Portfolio
    Manager's system prompt already states in prose (commit 1, f79213e): when
    the PM increases exposure (Buy/Overweight) against a Research Manager AND
    a Trader who both declined it, require its stated `confidence` to clear a
    higher bar (`consensus_override_min_confidence`, or the stricter
    `consensus_override_min_confidence_bearish` when the desk was actively
    negative) before the trade is allowed to stand. Below the bar — or with
    confidence unset, treated as below it, exactly as _apply_hysteresis_gate
    does — the rating is downgraded to Hold.

    2026-09-04, id=1212/1217: the PM produced a Buy/Overweight overriding a
    Hold/Underweight Research Manager and a Hold/Sell Trader on ordinary
    re-weighed evidence, naming no specific new fact from the risk debate.
    The prompt rule alone did not hold; this makes it mechanical.

    Called BEFORE _apply_hysteresis_gate at the call site: a downgrade to
    Hold here makes that gate a no-op (Hold is not state-flipping), so the
    two bracketed prefixes can never both land on executive_summary.

    Mirrors _apply_hysteresis_gate's shape exactly — the full numbered note
    APPENDED to investment_thesis, a short NUMBER-FREE variant PREFIXED onto
    executive_summary (the dashboard card clamps to ~6 lines; and
    language_ab_report scans executive_summary+investment_thesis together for
    prose-vs-structured numeric mismatch, so the percentages appear once,
    only in the thesis). `record_out` is populated in place for
    PipelineState.consensus_override / the jsonb transcript — including when
    the override PASSED (`gated: False`), so an audit can count both."""
    tier = _consensus_override_tier(decision, research_plan, trader_proposal)
    if tier is None:
        return decision
    threshold = risk_rules.rule(
        "consensus_override_min_confidence_bearish"
        if tier == "bearish"
        else "consensus_override_min_confidence"
    )
    rm = research_plan.recommendation.value
    tr = trader_proposal.action.value
    orig = decision.rating.value
    conf = decision.confidence
    record_out.update(
        {
            "rm": rm,
            "trader": tr,
            "pm_rating": orig,
            "confidence": conf,
            "tier": tier,
            "threshold": threshold,
            "gated": False,
        }
    )
    if conf is not None and conf >= threshold:
        return decision
    record_out["gated"] = True
    conf_str = f"{conf:.0%}" if conf is not None else "unset"
    note = (
        f"\n\n[Consensus override gate: downgraded from {orig} to Hold — the Research "
        f"Manager ({rm}) and the Trader ({tr}) both declined more exposure, and confidence "
        f"({conf_str}) is below the {threshold:.0%} bar required to increase exposure against "
        f"them.]"
    )
    prefix = (
        f"[Consensus override gate: downgraded from {orig} to Hold — increasing "
        "exposure against a unanimous Research Manager/Trader call needs higher "
        "confidence than stated.] "
    )
    held_equiv = not isinstance(decision, FlatPortfolioDecision)
    hold_rating = PortfolioRating.HOLD if held_equiv else FlatPortfolioRating.HOLD
    return decision.model_copy(
        update={
            "rating": hold_rating,
            "executive_summary": prefix + decision.executive_summary,
            "investment_thesis": decision.investment_thesis + note,
        }
    )


# Tiers that actually downgrade the Sell. "both" only — the step-6
# retrospective calibration (963 stored decisions, 7 weeks) found the gate
# fires on exactly one held Sell under "both" (id=1358, the incident), with
# zero false activations. "mechanics" alone would also have caught id=1216 —
# a fundamentally-reasoned exit where the PM never cited the cap — which is a
# human call, not this gate's; "narrative" alone had a PM-text sample of one.
# Widen only on a later calibration.
_UNJUSTIFIED_EXIT_FIRING_TIERS: tuple[str, ...] = ("both",)


def _unjustified_exit_tier(
    decision, research_plan, *, held: bool, atr_pct: float | None,
    current_weight: float | None, misuse_spans: list[str],
) -> str | None:
    """Classifier for `_apply_unjustified_exit_gate` (2026-09-09, P0-2/3,
    id=1358). Returns None — the gate does not apply — for anything that is
    not a HELD Sell while `settings.unjustified_exit_gate_enabled` is on,
    which is the overwhelmingly common case.

    Otherwise returns which of the two P0 signatures are present:
      "narrative"  the risk debate / PM text cited the added-share sizing cap
                   or its ATR scaling as the basis for exiting
                   (`misuse_spans`, from `scan_sizing_cap_misuse`, non-empty)
      "mechanics"  the Research Manager concluded Underweight — a trim, not
                   an exit — while the position is ALREADY at or below the
                   ATR-scaled Underweight target, so the mechanically correct
                   action is no sale; a PM (often following a Trader who
                   escalated the trim to a full Sell) that liquidates anyway
                   has expanded a trim into an exit with no sizing basis.
                   Both step-6 incidents (id=1358, id=1216) had the Trader at
                   Sell, so the Trader's action is deliberately NOT part of
                   this test — it is a symptom of the same error, not an
                   exemption from it.
      "both"       both of the above

    Never fires on a rating that is not Sell, or on a flat ticker — a
    position must exist for its liquidation to be unjustified."""
    if not held or not settings.unjustified_exit_gate_enabled:
        return None
    if decision.rating.value != "Sell":
        return None

    narrative = bool(misuse_spans)
    mechanics = False
    if (
        research_plan is not None
        and research_plan.recommendation.value == "Underweight"
        and current_weight is not None
    ):
        uw_target = vol_adjusted_allocation_for(PortfolioRating.UNDERWEIGHT, atr_pct)
        if uw_target is not None and current_weight <= uw_target:
            mechanics = True

    if narrative and mechanics:
        return "both"
    if narrative:
        return "narrative"
    if mechanics:
        return "mechanics"
    return None


def _apply_unjustified_exit_gate(
    decision, research_plan, trader_proposal, *, held: bool, atr_pct: float | None,
    current_weight: float | None, misuse_spans: list[str], drawdown_frac: float | None,
    execution_blocked: bool, forced_exit_on_book: bool, record_out: dict,
):
    """Downgrades a PM Sell of a HELD position to Hold when it is unjustified
    in the id=1358 sense — see `_unjustified_exit_tier` for the two
    signatures. Enabled (`settings.unjustified_exit_gate_enabled`, default
    True as of the step-6 calibration: 963 stored decisions over 7 weeks, one
    "both"-tier activation, id=1358, zero false activations). Still records
    `gated: False` for a tier that did not reach the firing bar, so the
    calibration can be re-run.

    Hard exceptions — returns `decision` untouched, no record, when a
    code-enforced safety net is already permitting or forcing a reduction and
    must not be second-guessed by this gate:
      - a stop-loss or reversal is the reason on the most recent trade
        (`forced_exit_on_book`)
      - execution is currently blocked for this ticker (`execution_blocked`)
      - the aggregate portfolio drawdown breaker has tripped
        (`drawdown_frac >= portfolio_drawdown_breaker_pct`) — only reductions
        are allowed while it is active

    The caller restricts this to live cycles (`as_of is None`); a backtest
    never reaches it. Mirrors `_apply_consensus_override_gate`'s shape: the
    full note APPENDED to investment_thesis, a NUMBER-FREE prefix on
    executive_summary, `record_out` filled in even when the tier did not
    reach the firing bar (`gated: False`)."""
    tier = _unjustified_exit_tier(
        decision, research_plan, held=held, atr_pct=atr_pct,
        current_weight=current_weight, misuse_spans=misuse_spans,
    )
    if tier is None:
        return decision
    if (
        forced_exit_on_book
        or execution_blocked
        or (drawdown_frac is not None and drawdown_frac >= risk_rules.rule("portfolio_drawdown_breaker_pct"))
    ):
        return decision

    rm = research_plan.recommendation.value if research_plan is not None else None
    tr = trader_proposal.action.value if trader_proposal is not None else None
    uw_target = vol_adjusted_allocation_for(PortfolioRating.UNDERWEIGHT, atr_pct)
    record_out.update(
        {
            "tier": tier,
            "rm": rm,
            "trader": tr,
            "current_weight": current_weight,
            "underweight_target": uw_target,
            "misuse_spans": list(misuse_spans),
            "gated": False,
        }
    )
    if tier not in _UNJUSTIFIED_EXIT_FIRING_TIERS:
        return decision

    record_out["gated"] = True
    note = (
        f"\n\n[Unjustified exit gate: downgraded from Sell to Hold — the added-share risk "
        f"cap and the volatility scaling only ever reduce shares being ADDED and never force "
        f"a sale of shares already held, yet they are the stated basis for this exit; the "
        f"Research Manager ({rm}) and the Trader ({tr}) asked for no full exit, and the "
        f"volatility-scaled target for their level ({uw_target:.1%}) is at or above this "
        f"position's current weight ({current_weight:.1%}), so the mechanical action is no "
        f"sale. No stop-loss, drawdown breaker, or execution blocker is forcing an exit this "
        f"cycle.]"
    )
    prefix = (
        "[Unjustified exit gate: downgraded from Sell to Hold — a full exit justified by the "
        "position-sizing cap, which never forces a sale of shares already held.] "
    )
    return decision.model_copy(
        update={
            "rating": PortfolioRating.HOLD,
            "executive_summary": prefix + decision.executive_summary,
            "investment_thesis": decision.investment_thesis + note,
        }
    )


def _desk_consensus_note(
    research_plan, trader_proposal, kr_risk_extra: str = "", debate_sizing_context: str = ""
) -> str:
    """Prompt-side context for the Portfolio Manager (2026-09-04 incident,
    id=1212/1217): states what the Research Manager concluded and what the
    Trader proposed BEFORE the risk debate ran, so the PM weighs the risk
    debate against them instead of treating it as a fresh verdict that
    replaces them. The commit-1 SYSTEM_PROMPT text ("You are shown, above the
    trader's proposal, what the Research Manager concluded ...") refers to
    this block. `""` when either upstream object is missing => prompt
    byte-identical to before. Threaded into portfolio_manager.decide the same
    way degradation_note is, and also recorded verbatim in the jsonb
    transcript as part of "what the PM actually read".

    2026-09-15 correction: this note used to assert unconditionally that the
    risk debate "saw nothing the Trader did not" — false whenever the risk
    debate actually ran with either of the two inputs
    `app.agents.trader.propose()` never receives: `kr_risk_extra` (KRX
    short-selling/program-flow signals, `.KS` tickers only — assigned from
    `get_kr_risk_signals` below and threaded into `risk_debators.build_all`'s
    `extra_context`) and `debate_sizing_context` (ATR volatility + the 1%
    risk-cap arithmetic, threaded in as its `sizing_context`). Worse, that static
    claim sat right above a second paragraph telling the PM to "name the
    specific new fact the debate introduced" — a standing contradiction, not
    just an inaccuracy, whenever either input was non-empty. This now
    reports what was actually supplied to THIS decision's risk debate (both
    args default to `""`, so a caller that omits them gets the prior
    byte-identical text), and draws the same direction-vs-mechanics line
    `app.agents.risk_debators`'s Neutral prompt already draws ("Decide
    direction first ... only once direction is settled may sizing mechanics
    shape the level and the stop"): the KRX flow signals are substantive
    evidence about the stock and can justify treating the debate as having
    introduced a new fact; the sizing arithmetic is execution mechanics, not
    direction evidence, and must never be accepted as one — id=1217 was
    exactly a risk-cap/valuation-multiple argument used by the PM as if it
    were new direction evidence to override desk consensus.
    """
    if research_plan is None or trader_proposal is None:
        return ""
    rm = research_plan.recommendation.value
    tr = trader_proposal.action.value
    # 2026-09-16 (engine audit follow-up item 3): the Trader's own `action` is
    # a 3-value direction (Buy/Hold/Sell) that cannot express the Research
    # Manager's Underweight (a 25% trim) any differently from Sell (a full
    # exit) — measured live (807 decisions, 2026-08-13..09-16): when the
    # Research Manager concluded Underweight and the Trader's action
    # collapsed to Sell, 46% of those Trader proposals described a full exit
    # in prose even though the plan called for a trim, and the Portfolio
    # Manager's own final Sell rate on that path was 5.0% vs. 0.0% when the
    # Trader stayed at Hold (p=0.013). Naming the Research Manager's real
    # target here — the one place both the PM and (via
    # risk_debators.build_all's new research_plan_note) the risk debate are
    # guaranteed to read the RM's own conclusion — closes the same
    # label-confusion gap `app.agents.trader.SYSTEM_PROMPT` and
    # `research_manager._HELD_RATING_SCALE_NOTE` were given today.
    rm_target = {
        "Buy": "100%", "Overweight": "75%", "Hold": "no change from the current position",
        "Underweight": "a real ~25% position, not an exit", "Sell": "a full exit",
    }.get(rm, "")
    rm_display = f"{rm} ({rm_target})" if rm_target else rm
    has_krx = bool(kr_risk_extra)
    has_sizing = bool(debate_sizing_context)
    if has_krx and has_sizing:
        risk_debate_clause = (
            "; the risk debate below started from the Trader's proposal, and additionally saw "
            "KRX short-selling and program-flow signals for this ticker and the position-sizing "
            "mechanics (ATR volatility and the 1% risk-cap arithmetic), neither of which the "
            "Trader received. The flow signals are substantive evidence about this stock and can "
            "qualify as a new fact; the sizing mechanics are execution arithmetic, not direction "
            "evidence, and never qualify."
        )
    elif has_krx:
        risk_debate_clause = (
            "; the risk debate below started from the Trader's proposal, and additionally saw "
            "KRX short-selling and program-flow signals for this ticker, which the Trader did not "
            "receive. Those are substantive evidence about this stock and can qualify as a new fact."
        )
    elif has_sizing:
        risk_debate_clause = (
            "; the risk debate below started from the Trader's proposal, and additionally saw the "
            "position-sizing mechanics (ATR volatility and the 1% risk-cap arithmetic), which the "
            "Trader did not receive. Those are execution arithmetic, not direction evidence, and "
            "never qualify as a new fact."
        )
    else:
        risk_debate_clause = (
            "; the risk debate below started from the Trader's proposal and saw nothing the Trader "
            "did not."
        )
    note = (
        "DESK CONSENSUS BEFORE THE RISK DEBATE: the Research Manager concluded "
        f"{rm_display}, and the Trader proposed {tr}. Both were formed from the full analyst "
        "reports and the bull/bear debate"
        f"{risk_debate_clause} Weigh the risk debate "
        "against these two, not in place of them."
    )
    if rm in ("Hold", "Underweight", "Sell") and tr in ("Hold", "Sell"):
        note += (
            " Neither of them supports increasing exposure. A Buy or Overweight from you "
            "here overrides the whole desk on the risk debate alone — name the specific new "
            "fact the debate introduced in your executive_summary, or choose Hold."
        )
    return note


_SweepSource = Literal["reports", "research_plan", "research_debate", "risk_debate", "trader_proposal", "pm"]


def _sweep_texts(state, sources: set[_SweepSource]) -> list[tuple[str, str]]:
    """`(agent_name, text)` pairs for every requested source, using the exact
    agent-naming convention each of the post-PM narrative-guard sweep blocks
    (halt_status_violations, absence_reframing_violations,
    derived_value_mismatch_violations, session_close_violations,
    desk_narrative_violations, sizing_cap_misuse_violations,
    price_limit_violations, verified_price_violations —
    app.orchestrator.pipeline, ~line 2210 onward) already used at each call
    site before this helper existed. Pure data collection — no scan is run
    here; each sweep block still calls its own `scan_...` function per pair
    and keeps its own accumulation/logging/`state.xxx_violations` assignment
    exactly as before. Requesting a source whose upstream object is `None`
    (or empty) yields nothing for it, same as every sweep block's own
    pre-existing `if state.research_plan is not None:` / `for _turn in
    state.risk_debate.turns:` guards did.

    - "reports": every analyst report, `state.reports.items()` — agent_name
      is the dict key itself (e.g. "Market Analyst").
    - "research_plan": `render_research_plan(state.research_plan)` under
      "Research Manager" — the exact name every sweep block already used.
    - "research_debate": each `state.research_debate.turns` entry under
      `f"research_{turn.speaker.lower()}"` (e.g. "research_bull") — only
      session_close_violations and verified_price_violations scanned this
      source before; the others never did (see each block's own comment on
      why).
    - "risk_debate": each `state.risk_debate.turns` entry under
      `f"risk_{turn.speaker.lower()}"` (e.g. "risk_conservative").
    - "trader_proposal": `render_trader_proposal(state.trader_proposal)`
      under "trader" — the same rendering used for the jsonb transcript's
      own "trader_proposal" field (~line 2580 below); no prior sweep block
      scanned this source (desk_narrative_violations reads
      `state.trader_proposal.action.value` only as a comparison value, never
      as scanned text), so "trader" is a fresh naming choice, not a copy of
      an existing call site.
    - "pm": the PM's `executive_summary` and `investment_thesis`, each under
      "portfolio_manager" — the exact name every sweep block already used,
      as two separate pairs (matching every existing block's two separate
      `scan_...(...executive_summary...)` / `scan_...(...investment_thesis...)`
      calls).
    """
    out: list[tuple[str, str]] = []
    if "reports" in sources:
        out.extend(state.reports.items())
    if "research_plan" in sources and state.research_plan is not None:
        out.append(("Research Manager", render_research_plan(state.research_plan)))
    if "research_debate" in sources:
        for _turn in state.research_debate.turns:
            out.append((f"research_{_turn.speaker.lower()}", _turn.content))
    if "risk_debate" in sources:
        for _turn in state.risk_debate.turns:
            out.append((f"risk_{_turn.speaker.lower()}", _turn.content))
    if "trader_proposal" in sources and state.trader_proposal is not None:
        out.append(("trader", render_trader_proposal(state.trader_proposal)))
    if "pm" in sources:
        out.append(("portfolio_manager", state.final_decision.executive_summary or ""))
        out.append(("portfolio_manager", state.final_decision.investment_thesis or ""))
    return out


def _degradation_note(degraded: list[str], total: int = 4) -> str:
    """Prompt-side warning for the PM when one or more analyst reports failed
    to generate this cycle and were replaced with a placeholder (2026-09-03
    decision id=1179 — a rate-limited Fundamentals report degraded silently
    and the final decision text showed nothing). `""` when nothing degraded,
    so the common case's prompt is unchanged. Threaded into
    portfolio_manager.decide the same way session_note/period_note are."""
    if not degraded:
        return ""
    n = len(degraded)
    names = ", ".join(degraded)
    return (
        f"ANALYSIS DEGRADATION WARNING: {n} of {total} analyst reports failed to generate "
        f"this cycle and were replaced with a placeholder ({names}). The trader proposal and "
        f"risk debate below were produced without them. Treat this as reduced evidence: state "
        f"the gap explicitly in your executive_summary, and do not express conviction the "
        f"surviving reports do not support."
    )


def _apply_degradation_notice(decision, degraded: list[str], total: int = 4):
    """Mechanically forces the analyst-degradation fact into the decision's
    user-visible fields — mirrors _apply_hysteresis_gate (PREFIX on
    executive_summary since the dashboard card clamps to ~6 lines; APPEND a
    bracketed note to investment_thesis). Unconditional when `degraded` is
    non-empty; a clean run gets `decision` back untouched (zero diff).

    NOT consequential — it changes no rating, sizing, or confidence. The PM
    was already told via `_degradation_note`; this only guarantees the fact
    is visible even if the model's own prose omitted it. 2026-09-03 id=1179.

    The prefixes carry only `n/total` counts (no percentages) — kept away
    from language_ab_report.py's prose-vs-structured numeric-mismatch scan
    over executive_summary + investment_thesis."""
    if not degraded:
        return decision
    n = len(degraded)
    names = ", ".join(degraded)
    updates = {
        "executive_summary": (
            f"[Degraded analysis: {n}/{total} analyst reports failed ({names}) — "
            f"decided without them.] " + decision.executive_summary
        ),
        "investment_thesis": (
            decision.investment_thesis
            + f"\n\n[Analyst degradation: {n} of {total} analyst reports failed this cycle "
            f"and were replaced with a placeholder — {names}. This decision was made on the "
            f"remaining {total - n}.]"
        ),
    }
    if not is_blank_card_reason(decision.card_reason_en):
        updates["card_reason_en"] = f"[Degraded {n}/{total}] " + decision.card_reason_en
    if not is_blank_card_reason(decision.card_reason_ko):
        updates["card_reason_ko"] = f"[분석 축소 {n}/{total}] " + decision.card_reason_ko
    return decision.model_copy(update=updates)


def _native_currency(ticker: str) -> str:
    """The currency the numbers for this ticker are actually denominated in —
    KRW for a `.KS` ticker, USD otherwise (the same suffix-based convention
    `app.execution.kis_bridge` already uses; nothing in this pipeline ever
    converts). Exists because an unlabeled figure handed to an LLM gets
    labelled by the LLM instead, and it guesses dollars: 229200.KS,
    2026-08-10, "약 1,415,380달러" for what was a KRW amount (fixed for
    `_risk_cap_context` below, then re-found 2026-08-25 in
    `_position_context`, which feeds 5 downstream nodes instead of 1 —
    Trader, all three risk debators, and the Portfolio Manager)."""
    return "KRW" if ticker.endswith(".KS") else "USD"


@cached_balance_fetches()  # FIX-2, 2026-09-03 id=1187 audit — see docstring below
async def _position_context(ticker: str, as_of: date | None) -> tuple[bool, str]:
    """Gives the Portfolio Manager explicit awareness of the actual current
    position instead of deciding blind every time — without this, there's no
    way for it to ever reason about re-entering after an exit, since it has
    no memory of having sold. Safe for backtesting without its own `as_of`
    filter: the trades table is only ever populated with dates strictly
    before the one currently being decided, since the backtest harness
    processes dates in chronological order.

    LIVE (`as_of is None`) prefers the REAL KIS-held quantity
    (`real_shares_held`) over the internal simulated ledger's `shares_after`
    for the share count in the narrative — the internal ledger accumulates
    every paper/backtest trade ever recorded for a ticker and can drift
    arbitrarily far from what's actually held (real incident: NAVER showed
    255 shares from the ledger when the real account held 1, and the
    Portfolio Manager's whole narrative reasoned off the wrong number).
    Falls back to the internal ledger if the real balance can't be read
    (broker not connected / KIS call failed) — this is narrative context
    only, not an enforcement path, so degrading rather than blocking here is
    fine. Backtesting (`as_of` set) always uses the internal ledger, since
    there is no real broker to query for a historical replay.

    Returns `(held, text)` — `held` is threaded into
    `portfolio_manager.decide`'s own `held` param, which structurally
    restricts a flat ticker's rating to Buy/Hold (see app.state's
    FlatPortfolioDecision docstring).

    Also appends the currently-set stop-loss (if any) and, live only, real
    avg cost/unrealized P&L — two of the context gaps identified 2026-07-28
    (see project memory: ai-decision-context-gaps). The stop-loss comes from
    `get_open_stop_loss`, the same "most recent trades row" query as
    `get_last_trade` above, so it's backtest-safe for the identical reason
    (no `as_of` filter needed). Without it, a re-analysis proposes a fresh
    stop-loss with no idea what's already enforced, drifting tighter/wider
    across cycles for no market-driven reason. Avg cost/P&L
    (`real_position_detail`) is live-only like `real_shares_held` above —
    KIS already computes these server-side; without surfacing them, the
    Portfolio Manager only ever saw the LAST trade's price, not the true
    (possibly multi-buy) average, and had no idea whether the position was
    currently up or down.

    FLAT branch also distinguishes (2026-08-14) a FORCED stop-loss exit from
    the AI's own Sell decision, and states how far price has moved since —
    "inform, don't block": a hard re-entry gate would permanently exclude
    exactly the "gapped up and never looked back" case, so this only makes
    the Portfolio Manager's own judgment better-informed, never overrides
    it. Real incident this fixes: 005930 was trailing-stop-liquidated at
    198,714 (2026-07-29), re-bought 21h later at 214,500 (+7.94%) — the
    Portfolio Manager had no idea the prior exit was a forced safety stop
    rather than its own considered Sell, or that it was paying a
    meaningfully higher price to get back in. See memory:
    ai-reentry-context-2026-08-14. `get_last_close` here is a SECOND call
    within this run (already called again later, around the Portfolio
    Manager's own price context) — `run_pipeline`'s `candle_run_cache()`
    scope (this module's own module-level comment) makes this a cache hit,
    not a new network round-trip, so this deliberately does not restructure
    the pipeline to fetch-once-and-thread a price through more call sites
    for one extra in-cache read.

    Every money figure below carries its own currency label
    (`_native_currency`) — this function's output reaches FIVE LLM nodes
    (Trader, all three risk debators, Portfolio Manager), five times the
    blast radius of the identical unlabeled-KRW defect `_risk_cap_context`
    was fixed for (229200.KS, 2026-08-10: the Portfolio Manager read a KRW
    risk figure out loud as "약 1,415,380달러"). Nothing here converts
    currency; a `.KS` ticker's prices, avg cost and P&L amount all come back
    from KIS in KRW, a US ticker's in USD.

    FIX-2 (2026-09-03 id=1187 audit): wrapped (via the `@cached_balance_
    fetches()` decorator above) in the SAME request-scoped balance cache
    `app.execution.ledger.rebalance_to_target` uses for its `market_lock`
    hold — `real_shares_held` below and `real_position_detail` further down
    each independently call the broker for the SAME ticker's balance with
    nothing in between that could change it, so the second call was a pure
    duplicate KIS round-trip. Deliberately NOT `enforcement_safe=True`: this
    scope runs OUTSIDE any `market_lock` (this is narrative context for the
    LLM nodes, not an enforcement decision), so `real_account_snapshot`
    itself must still fetch fresh here — see
    `app.execution.portfolio_risk._balance_fetch_enforcement_safe`'s own
    comment for why that distinction matters."""
    position = await get_position(ticker)
    last_trade = await get_last_trade(ticker)
    open_stop = await get_open_stop_loss(ticker)
    currency = _native_currency(ticker)
    stop_loss_note = (
        f" Current stop-loss on record: {open_stop[1]:.2f} {currency}." if open_stop is not None else ""
    )

    real_shares = await real_shares_held(ticker) if as_of is None else None
    shares = real_shares if real_shares is not None else position.shares
    # Only live callers ever attempt the real lookup — a backtest not
    # calling it at all is expected, not a fallback worth flagging.
    source_note = (
        " (internal paper ledger — real KIS balance unavailable)"
        if as_of is None and real_shares is None
        else ""
    )

    if shares > 0:
        # Reachable only when the ledger says flat-then-re-entered but this
        # branch's own `last_trade` still points at a forced-exit row whose
        # `price` is a stop THRESHOLD, not a fill (e.g. a mirror-succeeded/
        # stale-balance edge) — "entered via a sell at <threshold>" would
        # already misdescribe the side, and quoting a threshold as an entry
        # price compounds it. See `_position_context`'s stop-loss branch
        # below for the fuller reasoning; omitting the clause here is
        # strictly safer than narrating a number nothing traded at.
        basis = (
            f" (entered via a {last_trade.side} at {last_trade.price:.2f} {currency} "
            f"on {last_trade.trade_date.isoformat()})"
        ) if last_trade and not is_threshold_placeholder_note(last_trade.adjustment_note) else ""
        pnl_note = ""
        if as_of is None:
            detail = await real_position_detail(ticker)
            if detail is not None:
                pnl_note = (
                    f" Avg cost {detail.avg_price:.2f} {currency}, unrealized P&L {detail.pnl_pct:+.2f}% "
                    f"({detail.pnl_amount:+,.0f} {currency})."
                )
        return True, f"Currently holding {shares} shares{source_note}{basis}.{pnl_note}{stop_loss_note}"

    if last_trade is not None:
        reason = last_trade.reason or ""
        if reason.startswith(REVERSAL_REASON_PREFIX):
            # The most recent row is a correction, not an executed market
            # trade — reaching further back for the pre-reversal row risks
            # surfacing the very trade that was reversed as if it stood.
            # Say only what's safely known: this ticker is flat, and the
            # prior exit can't be reliably characterized from here.
            return False, (
                f"Currently flat (no position){source_note} — the most recent ledger entry for this "
                f"ticker on {last_trade.trade_date.isoformat()} is a correction/reversal, not an "
                f"executed market trade, so how the prior position actually closed cannot be stated "
                f"reliably. Treat prior-exit context as unknown."
            )

        if reason.startswith(STOP_LOSS_REASON_PREFIX) and last_trade.side == "sell":
            # A forced exit's recorded price is NOT always a price. The
            # daily-bar backstop (check_and_apply_stop_loss/
            # check_and_apply_trailing_stop) passed the breached THRESHOLD as
            # the trade price until 2026-08-27 (see
            # app.execution.ledger._resolve_forced_exit_price's own
            # docstring for the 047040.KS id=280 incident), and its live
            # fallback branch still records the threshold today when no tick
            # is knowable — both are marked in `adjustment_note`. Narrating
            # that number as "the price you sold at", and computing a %
            # move against it, is exactly the fabrication this whole context
            # feature exists to prevent (see get_last_trade's own
            # docstring), and it reaches FIVE LLM nodes (Trader, all three
            # risk debators, Portfolio Manager) — not just this one
            # ticker's own next analysis. 047040.KS id=280: the old
            # narrative would have said 19,729.24 against a real ~21,000
            # fill, and every "% versus that stop-out price" derived from it
            # would be off by 6.4%. `price_now` is withheld here too, not
            # just the delta — printing it alongside an admittedly-unknown
            # exit price just invites the model to compute the same bogus
            # ratio itself, and it already reaches the Portfolio Manager
            # through that node's own separate price context regardless.
            if is_threshold_placeholder_note(last_trade.adjustment_note):
                return False, (
                    f"Currently flat (no position){source_note} — this position was FORCE-EXITED by the "
                    f"automated stop-loss on {last_trade.trade_date.isoformat()}, not sold by your own "
                    f"decision. The price it actually sold at was NOT recorded: the ledger holds only the "
                    f"stop threshold {last_trade.price:.2f} {currency}, which is a threshold and NOT a "
                    f"traded price, so both the exit price and any move since are UNKNOWN — do not treat "
                    f"that number as an exit price or reason from it. Re-entry is permitted and is your "
                    f"judgment call: weigh whether the thesis that triggered the stop has actually changed."
                )
            move_note = ""
            try:
                price_now = await get_last_close(ticker, as_of=as_of)
                pct = (price_now - last_trade.price) / last_trade.price * 100
                move_note = f" Current price {price_now:.2f} {currency} is {pct:+.2f}% versus that stop-out price."
            except Exception:
                pass  # narrative-only context — omit the delta, keep the rest of the sentence
            return False, (
                f"Currently flat (no position){source_note} — this position was FORCE-EXITED by the "
                f"automated stop-loss at {last_trade.price:.2f} {currency} on {last_trade.trade_date.isoformat()}, "
                f"not sold by your own decision.{move_note} Re-entry is permitted and is your judgment "
                f"call: weigh whether the thesis that triggered the stop has actually changed."
            )

        return False, (
            f"Currently flat (no position){source_note} — last {last_trade.side} at "
            f"{last_trade.price:.2f} {currency} on {last_trade.trade_date.isoformat()}."
        )

    return False, "No prior position history for this ticker — this would be a new entry."


async def _atr_context(ticker: str, as_of: date | None, *, held: bool) -> tuple[float | None, str]:
    """Tells the Portfolio Manager, in advance, exactly how volatility-based
    sizing (`app/execution/sizing.py::target_allocation_for`) will scale
    whichever rating it picks — without this, it has zero visibility into
    that adjustment and can write a narrative that flatly contradicts the
    trade code actually executes. Real incident: a live MSFT decision said
    "adding to our position... 25% add-on" while the executed trade was a
    SELL, because MSFT's ATR was above target and the Buy rating's 100%
    base allocation got scaled down to 61.43% — below the existing position.
    The trade was correct (the safety mechanism worked as designed); only
    the prose was wrong. Returns the raw ATR%% alongside the context string
    so the caller can reuse it for the actual sizing call instead of a
    second fetch, and `held` determines which ratings the returned text
    mentions, since a flat ticker's only tradeable rating is Buy
    (Overweight/Underweight/Sell all presuppose an existing position — see
    `FlatPortfolioRating` in `app/state.py`)."""
    try:
        current_atr_pct = await get_atr_pct(ticker, as_of=as_of)
    except Exception:
        current_atr_pct = None

    # 2026-09-02 fix: get_atr_pct can return None WITHOUT raising (its own
    # docstring: "None (never NaN) if there isn't enough history yet" -- a
    # recent listing, or thin history). Before this check existed, that case
    # fell through to `current_atr_pct <= target` below and raised
    # `TypeError: '<=' not supported between instances of 'NoneType' and
    # 'float'` -- not caught by the narrow try above, and not caught by the
    # caller's asyncio.gather(return_exceptions=True) wrapper either, since
    # the retry path (pipeline.py's context-gather except block) re-runs
    # this WITHOUT return_exceptions and fails identically, aborting the
    # whole pipeline run for that ticker. `executability.atr_unavailable`
    # also catches a literal `0.0` (a degenerate close), not just `None` --
    # same predicate the fail-closed new-entry gate
    # (app.domain.executability.blocks_new_entry_without_atr,
    # app.execution.ledger.rebalance_to_target) uses, so this narrative and
    # that gate can never disagree about what counts as "no ATR". The raw
    # value (None, 0.0, or negative) is still returned to the caller --
    # 0.0 must stay distinguishable from None for state.atr_gate.
    if executability.atr_unavailable(current_atr_pct):
        if held:
            return current_atr_pct, (
                "Volatility data (14-day ATR) is unavailable for this ticker. No volatility-based "
                "scaling will be applied to your rating and the code-level stop-loss floor cannot be "
                "enforced this cycle; the existing position and its recorded stop-loss are unaffected. "
                "Size and stop conclusions here rest on your own judgment, not on a validated floor."
            )
        return current_atr_pct, (
            "Volatility data (14-day ATR) is unavailable for this ticker. Because neither "
            "volatility-based sizing nor the ATR-derived stop-loss floor can be computed, code will "
            "REFUSE to open a new position in this ticker this cycle regardless of your rating -- a "
            "Buy will be recorded and then blocked at execution. Only Hold is actionable here."
        )

    target = risk_rules.rule("sizing_target_atr_pct")
    if current_atr_pct <= target:
        if held:
            return current_atr_pct, (
                f"Current volatility (14-day ATR) is {current_atr_pct:.1%} of price, at or below the "
                f"{target:.1%} target — no volatility-based scaling applies; each rating gets its full base "
                f"allocation (Buy 100%, Overweight 75%, Underweight 25%, Sell 0%)."
            )
        return current_atr_pct, (
            f"Current volatility (14-day ATR) is {current_atr_pct:.1%} of price, at or below the "
            f"{target:.1%} target — no volatility-based scaling applies; a Buy rating gets its full 100% "
            f"base allocation. (This ticker has no current position — Overweight/Underweight/Sell are not "
            f"valid ratings here; the only alternative to Buy is Hold, which never trades.)"
        )

    if held:
        scaled = {
            rating: vol_adjusted_allocation_for(rating, current_atr_pct)
            for rating in (PortfolioRating.BUY, PortfolioRating.OVERWEIGHT, PortfolioRating.UNDERWEIGHT)
        }
        scaled_lines = ", ".join(f"{rating.value} -> {alloc:.0%} of equity" for rating, alloc in scaled.items())
        return current_atr_pct, (
            f"Current volatility (14-day ATR) is {current_atr_pct:.1%} of price, above the {target:.1%} "
            f"target — code will automatically scale down whichever rating you choose: {scaled_lines}. Hold "
            f"and Sell are unaffected (Hold never trades; Sell is already 0%). Write your executive summary "
            f"and thesis to match the actual resulting position size, not the nominal rating label."
        )

    scaled = {
        rating: vol_adjusted_allocation_for(rating, current_atr_pct)
        for rating in (PortfolioRating.BUY,)
    }
    scaled_lines = ", ".join(f"{rating.value} -> {alloc:.0%} of equity" for rating, alloc in scaled.items())
    return current_atr_pct, (
        f"Current volatility (14-day ATR) is {current_atr_pct:.1%} of price, above the {target:.1%} "
        f"target — code will automatically scale down whichever rating you choose: {scaled_lines}. Hold "
        f"is unaffected (Hold never trades). Overweight/Underweight/Sell are not valid ratings for this "
        f"currently-flat ticker. Write your executive summary and thesis to match the actual resulting "
        f"position size, not the nominal rating label."
    )


async def _risk_cap_context(ticker: str, price: float, *, as_of: date | None) -> str:
    """Explains the "1% rule" risk cap (`app.execution.ledger.rebalance_to_target`)
    to the Portfolio Manager before it picks a stop-loss, since the rule ties
    the stop it chooses directly to how many new shares can actually be
    added — a tighter stop allows more, a wider stop forces fewer. Same "give
    the model the mechanism so its narrative can't silently contradict the
    code" treatment `_atr_context` already gives ATR-based allocation
    scaling; without this the Portfolio Manager has zero visibility into a
    second mechanism that can further shrink whatever size its narrative
    describes.

    `equity` comes from `app.execution.ledger.resolve_equity_basis` — the
    SAME function `rebalance_to_target` calls a few LLM calls later to
    actually size the trade (live: real broker equity when available,
    falling back to the internal ledger; backtest: always the internal
    ledger). The two calls used to disagree (this narrative always used the
    internal ledger's own market_value, the real sizing preferred real
    broker equity when live) — same-ticker, same-cycle, sometimes by several
    times over — which let an Underweight/Sell rating's intended decrease
    compute as an increase (memory: live-rebalance-equity-basis-mismatch-
    2026-08-14, GOOG 2026-08-13). When the real broker equity is unavailable
    here, the LLM is told so explicitly, since the actual sizing call could
    still succeed (or fail) independently ~9 LLM calls later.

    `ticker`'s own native currency (KRW for a .KS ticker, USD otherwise —
    same krx_code-based convention kis_bridge.py already uses), not a
    hardcoded `$`: `equity`/`max_dollar_risk`, and `price` is always in the
    ticker's own native currency (KIS returns KRW for domestic, yfinance/KIS
    return USD for overseas — this function never converts). A hardcoded `$`
    here for a domestic ticker previously had the Portfolio Manager write a
    KRW risk figure out loud as "달러" (dollars) — e.g. 229200.KS,
    2026-08-10, "약 1,415,380달러" for what was actually a KRW amount."""
    if price <= 0:
        return "Risk-cap data unavailable — position sizing will not reflect the stop-loss risk cap."
    try:
        equity, equity_source = await resolve_equity_basis(ticker, price, as_of=as_of)
    except EquityBasisUnavailable as exc:
        # Real account + live + the real broker balance could not be read, so
        # resolve_equity_basis refused to fall back to the internal ledger's
        # fictional paper_starting_cash (see its docstring). The generic branch
        # below would already catch this — this one exists purely to tell the
        # five LLM nodes reading this string the specific, actionable thing:
        # the actual sizing call (ledger.rebalance_to_target, ~9 LLM calls
        # later) hits the SAME refusal and skips the trade outright, so a
        # narrative promising an exposure increase this cycle will not match
        # what execution does. Contrast the live_fallback path below, where
        # sizing might still succeed against a real figure.
        # Deliberately NOT logged here: resolve_equity_basis already logged
        # this exact condition at ERROR right before raising, and
        # rebalance_to_target logs it again at WARNING when it skips the
        # trade — a third line would triple-log one broker outage per cycle.
        # 실계좌 잔고 조회 실패 — 이번 사이클은 리스크캡 수치도, 실제 주문도 없음.
        return (
            f"No risk-cap figure this cycle: this is a real account and the real broker balance "
            f"could not be read just now ({exc.failure or 'reason unknown'}), so there is no equity "
            f"basis to size against. The execution step will skip this trade entirely rather than "
            f"size it against a fictional balance — do not describe an exposure increase as "
            f"something that will happen this cycle."
        )
    except Exception:
        return "Risk-cap data unavailable — position sizing will not reflect the stop-loss risk cap."

    currency = _native_currency(ticker)
    max_risk_pct = risk_rules.rule("sizing_max_risk_pct_of_equity")
    max_dollar_risk = equity * max_risk_pct
    tight_stop_shares = math.floor(max_dollar_risk / (price * 0.03))
    wide_stop_shares = math.floor(max_dollar_risk / (price * 0.10))
    live_fallback = as_of is None and equity_source == "internal_ledger"
    if live_fallback:
        # Log only — no alert here (unlike ledger.rebalance_to_target's own
        # equity-fallback alert, ~9 LLM calls after this): both call sites
        # share one (ticker, "equity_basis_fallback") cooldown key, so an
        # alert fired here now would win the race and suppress the
        # sizing-time one, which is the more important of the two to see
        # (it's the one a real order actually got sized against).
        logger.warning(
            "%s: risk-cap narrative fell back to internal-ledger equity (%.0f) — real broker balance unavailable",
            ticker, equity,
        )
    fallback_note = (
        " (the real account balance could not be read just now — actual execution may end up sizing "
        "off a different, real-account figure if that call succeeds later)"
        if live_fallback
        else ""
    )
    # 2026-09-01, id=1157/id=1158 (first real-account KRX cycle, 200,000 KRW
    # equity): at this equity every KRX name above roughly 66,000 KRW makes
    # BOTH figures below zero — a 1% risk budget cannot absorb even one
    # share's 3%-stop risk. The generic text then reads as an invitation to
    # trade stop distance against share count, a lever that does not exist,
    # and the risk debate acted on it (~12 LLM calls arguing sizing for a
    # position that cannot hold one share). Saying the true thing plainly is
    # the whole point of this function.
    if tight_stop_shares <= 0 and wide_stop_shares <= 0:
        one_share_risk = price * 0.03
        return (
            f"This trade's ADDED shares (not the whole resulting position) are separately capped so their "
            f"risk (shares added x (price - your stop_loss)) never exceeds "
            f"{max_risk_pct:.1%} of equity ({max_dollar_risk:,.0f} {currency} on current equity of "
            f"{equity:,.0f} {currency}{fallback_note}). At this price ({price:.2f} {currency}) and this "
            f"equity, that cap permits ZERO added shares at EVERY stop distance: a single share behind even "
            f"a tight ~3% stop would risk ~{one_share_risk:,.0f} {currency}, already more than the entire "
            f"{max_dollar_risk:,.0f} {currency} budget for this trade. Tightening or widening the stop does "
            f"not change this — there is no stop-loss choice that unlocks an exposure increase here, so do "
            f"not argue about position size as though a size were available, and do not compute a portfolio "
            f"impact for a share count that cannot be bought. Two things ARE still live: the direction/entry "
            f"judgment itself, and the stop-loss level that would apply if a position is ever opened. It "
            f"never forces a sale of shares already held, and is not a valid argument for exiting or "
            f"trimming one — do not cite this cap, the volatility scaling, or a resulting 'trivial' or "
            f"'negligible' share count as a reason to reduce an existing position."
        )
    return (
        f"This trade's ADDED shares (not the whole resulting position) are separately capped so their "
        f"risk (shares added x (price - your stop_loss)) never exceeds "
        f"{max_risk_pct:.1%} of equity ({max_dollar_risk:,.0f} {currency} on current equity "
        f"of {equity:,.0f} {currency}{fallback_note}). A TIGHTER stop allows MORE new shares; a WIDER stop forces "
        f"FEWER — e.g. at the current price ({price:.2f} {currency}), a ~3% stop distance allows up to "
        f"~{tight_stop_shares} new shares, a ~10% stop distance up to ~{wide_stop_shares}. This can shrink the add "
        f"on top of the volatility-based scaling above; it never forces a sale of shares already held, and is "
        f"not a valid argument for exiting or trimming one — do not cite this cap, the volatility scaling, or a "
        f"resulting 'trivial' or 'negligible' share count as a reason to reduce an existing position. If no stop "
        f"can be resolved at all (you omit one and volatility data is unavailable), any exposure increase is "
        f"blocked entirely."
    )


async def _execution_status_context(ticker: str, price: float) -> str:
    """Tells the Portfolio Manager, in advance, whether this ticker is
    currently blocked from actually executing an exposure-increasing trade
    for reasons that have nothing to do with the rating itself — trading
    halt, VI, market-wide circuit breaker (domestic only, no KIS overseas
    equivalent), or insufficient real cash. `app.execution.kis_bridge`
    already enforces all of these silently at execution time; without this,
    the Portfolio Manager can recommend the same blocked trade every cycle
    with no idea why nothing happens. Reuses the exact gate functions
    (`execution_blockers`/`available_cash_native`) so this can never drift
    from what actually blocks a real order — no new KIS calls invented here.
    LIVE ONLY (`as_of is None` callers) — needs a connected `KISBroker`,
    same restriction `_portfolio_risk_context` already has.
    매매정지/VI/서킷브레이커/가용현금 — 실행 게이트를 그대로 재조회해 판단 전에 미리 보여줌."""
    from app.web import state  # local import — avoid circular import, see portfolio_risk.py's note

    broker = await state.kis_broker()

    if broker is None:
        return "Execution status unavailable — broker not connected."

    market = ticker_market(ticker)
    krx_code = ticker[:-3] if market == "domestic" else None
    us_symbol = ticker if market == "overseas" else None
    lines: list[str] = []

    if market == "domestic":
        checked: list[str] = []
        blockers = await execution_blockers(broker, krx_code, ticker, checked_out=checked)
        if blockers:
            lines.append(
                "WARNING: this ticker is currently blocked from any exposure-increasing trade at "
                f"execution time — {'; '.join(blockers)}. A Buy/Overweight rating will be silently "
                "held at the existing position until this clears."
            )
        elif set(checked) == {"circuit_breaker", "stock_halt", "vi"}:
            # 2026-09-07, id=1238 (351320.KS): the News + Macro Analyst asserted
            # a currently-active halt from a several-days-old headline, and
            # nobody downstream — Research Manager, risk debate, PM — caught
            # the contradiction with this exact field, which already said the
            # opposite. All three gates below actually completed (see
            # execution_blockers' own `checked_out` docstring) — this is a
            # genuinely confirmed-clear read, not just an empty blockers list,
            # so it's worded strongly enough to override a stale news claim.
            lines.append(
                "CONFIRMED by the broker just now: no trading halt, no VI, and no "
                "circuit-breaker is currently active for this ticker. This is live broker "
                "data and it overrides any news item or analyst report that says this stock "
                "is halted or suspended — such an item is historical or a warning about a "
                "possible future halt, never a statement about right now."
            )
        else:
            # One or more of the three gates could not actually complete this
            # cycle (an exception, a paper-mode-unsupported skip, or a
            # genuinely unknown circuit-breaker state) — `blockers` being
            # empty here is NOT the same as a confirmed-clear read, so this
            # must not be worded as strongly as the CONFIRMED branch above.
            lines.append(
                "Trading-halt status could not be fully verified this cycle "
                f"(checks that ran: {', '.join(checked) if checked else 'none'}). Do not assert "
                "either that this ticker is halted or that it is definitely clear."
            )

    cash_result = await available_cash_native(broker, krx_code, us_symbol, price)
    if cash_result is not None:
        amount, currency = cash_result
        lines.append(f"Actual available cash to spend right now: {amount:,.2f} {currency}.")
    else:
        lines.append("Actual available cash figure unavailable — do not assume unlimited buying power.")

    return " ".join(lines)


def _render_portfolio_risk_context(
    ticker: str,
    snapshot: PortfolioSnapshot,
    clusters: dict[str, int],
    drawdown_frac: float,
) -> str:
    """Pure formatting half of `_portfolio_risk_context`, split out so a
    backtest caller supplying its own precomputed (snapshot, clusters,
    drawdown_fraction) via `run_pipeline`'s `portfolio_cap_override` can
    give the Portfolio Manager the same "mechanism, not just an unexplained
    number" context the live path gets — without duplicating this
    formatting logic or performing the live-only I/O below."""
    this_value = snapshot.per_ticker_value.get(ticker, 0.0)
    concentration_pct = this_value / snapshot.total_value if snapshot.total_value > 0 else 0.0

    bucket = clusters.get(ticker)
    bucket_mates = [t for t, b in clusters.items() if b == bucket and t != ticker] if bucket is not None else []
    bucket_value = sum(snapshot.per_ticker_value.get(t, 0.0) for t in [ticker, *bucket_mates])
    bucket_pct = bucket_value / snapshot.total_value if snapshot.total_value > 0 else 0.0

    lines = [
        f"{ticker} is currently {concentration_pct:.1%} of aggregate portfolio equity "
        f"(limit {risk_rules.rule('portfolio_concentration_max_pct'):.1%}).",
        (
            f"Its correlation bucket ({', '.join(bucket_mates) if bucket_mates else 'no other correlated names'}) "
            f"is {bucket_pct:.1%} of the portfolio (limit {risk_rules.rule('portfolio_correlation_bucket_max_pct'):.1%})."
        ),
    ]
    if drawdown_frac >= risk_rules.rule("portfolio_drawdown_breaker_pct"):
        lines.append(
            f"Aggregate portfolio equity is currently {drawdown_frac:.1%} below its high-water mark, at or "
            f"past the {risk_rules.rule('portfolio_drawdown_breaker_pct'):.1%} breaker — any exposure-increasing trade "
            f"will be blocked in code regardless of your rating; only Hold/Sell/reductions can execute."
        )
    else:
        lines.append(f"Aggregate portfolio drawdown from its high-water mark is {drawdown_frac:.1%} (no breach).")
    lines.append(
        "These portfolio-level caps apply on top of the per-ticker ones above and can only further reduce an "
        "exposure increase, never force a sale of shares already held."
    )
    return " ".join(lines)


async def _portfolio_risk_context(
    ticker: str,
) -> tuple[tuple[PortfolioSnapshot, dict[str, int], float], str]:
    """Tells the Portfolio Manager, in advance, how the portfolio-level caps
    (concentration, correlation-bucket, drawdown breaker —
    app.execution.portfolio_risk) currently sit relative to their limits,
    same "give the model the mechanism so its narrative can't contradict the
    code" treatment `_atr_context`/`_risk_cap_context` already give the
    per-ticker mechanisms. Unlike those, these caps look across the WHOLE
    trade_allowlist, not just this ticker's own book — nothing upstream of this
    would otherwise notice all 10 tickers going to Buy at once, or that
    several are highly correlated. Returns the computed
    (snapshot, clusters, drawdown_fraction) tuple alongside the context
    string so the caller can pass it straight through to
    `rebalance_to_target`'s `portfolio_cap_override` — avoiding a second,
    redundant fetch and guaranteeing the narrative and the later enforcement
    are computed from the exact same snapshot. LIVE ONLY — fetches the REAL
    connected KIS account balance (via `real_account_snapshot`) scoped to
    `ticker`'s own market (domestic/overseas never combined, see
    `ticker_market`) and the persisted per-market high-water mark; see
    `run_pipeline`'s `portfolio_cap_override` param for the backtest path.

    Fail-CLOSED on `AccountUnavailable` (broker not connected / KIS call
    failed): returns a synthetic 100%-drawdown tuple, which forces
    `apply_portfolio_caps` to block any exposure-increasing trade at
    enforcement time even though the real cause isn't an actual drawdown —
    the narrative string below still tells the Portfolio Manager the real
    reason. Same fail-closed principle as `apply_live_portfolio_caps`'s own
    AccountUnavailable handling, expressed through this function's
    (snapshot, clusters, drawdown_frac) return contract instead.
    실제 잔고 조회 실패 시 fail-closed — 인위적으로 100% 드로우다운을 반환해 노출 증가
    거래를 강제 차단(진짜 원인은 서술 텍스트로 별도 전달)."""
    market = ticker_market(ticker)
    market_trade_allowlist = settings.trade_allowlist_for_market(market)
    try:
        # Narrative only — the tuple returned here is NOT used as
        # portfolio_cap_override in live mode (run_pipeline passes None; the
        # real enforcement fetch happens under that market's lock in
        # rebalance_to_target, see ~line 1521). So the short read-only cache
        # is safe and it's what stops N concurrent analyses' EGW00215 burst.
        # 서술용이라 짧은 읽기 캐시로 충분 — 집행은 락 안에서 별도 조회.
        snapshot = await cached_account_snapshot(market)
    except AccountUnavailable as exc:
        empty = PortfolioSnapshot(per_ticker_value={}, total_value=0.0)
        return (empty, {}, 1.0), (
            f"Portfolio-level risk data unavailable ({exc}) — any exposure-increasing trade is blocked "
            "until the real account snapshot can be read again (fail-closed)."
        )

    try:
        clusters = await compute_correlation_clusters(market_trade_allowlist)
    except Exception:
        clusters = {}

    drawdown_frac = await current_drawdown_fraction(market, snapshot.total_value)
    return (snapshot, clusters, drawdown_frac), _render_portfolio_risk_context(ticker, snapshot, clusters, drawdown_frac)


# LRU-capped (not just TTL-capped like `_reports_cache` below) since a name
# is time-invariant and never expires on its own — without a cap, a
# long-running daemon or a many-ticker backtest would grow this for the
# life of the process, one entry per distinct ticker it ever saw.
_COMPANY_NAME_CACHE_MAX = 2000
# ticker -> (display_name, authoritative_aliases, weak_aliases,
# kis_english_name). The tuple's second/third elements feed
# app.agents.identity_gate — see `_company_identity` below for what
# "authoritative" vs "weak" means. The 4th element is KIS's own
# `prdt_eng_abrv_name` ("" when unavailable, never None), passed to
# identity_gate.enforce_identity as an exact-despaced amnesty needle.
_company_identity_cache: OrderedDict[str, tuple[str, frozenset[str], frozenset[str], str]] = OrderedDict()
# Per-ticker lock so concurrent callers racing an empty cache entry (Market/
# News+Macro/Sentiment analysts all call this for the same fresh candidate
# via asyncio.gather) await one shared fetch instead of each firing their own
# yfinance call — confirmed in practice: a new candidate ticker's first
# analysis round logged the same "implausible name" yfinance warning 3x
# within milliseconds, one per analyst, before this lock existed. Popped
# once the fetch it guards completes (see `_company_identity` below) — a
# caller already holding a reference to this lock instance keeps working
# fine even after its dict entry is gone, and any ticker whose identity is
# now cached never touches this dict again, so it never grows unbounded the
# way `_company_identity_cache` would without its own LRU cap above.
_company_identity_locks: dict[str, asyncio.Lock] = {}


async def _company_identity(ticker: str) -> tuple[str, frozenset[str], frozenset[str], str]:
    """(display_name, authoritative_aliases, weak_aliases, kis_english_name) to ground all four
    analysts' prompts (Market/Fundamentals/News+Macro/Sentiment) and, via
    `app.agents.identity_gate`, to verify what they actually wrote — without
    grounding they see only a bare ticker and are free to hallucinate an
    identity (see `market_analyst.run`'s docstring); without verification a
    model can still override a correct label with its own guess (see
    `app.agents.tool_loop.IDENTITY_GROUNDING_NOTE`'s 086790.KS incident, and
    the 2026-08-20 316140.KS "KB Financial Group" incident that motivated the
    gate itself). Cached per ticker: identity is time-invariant, so a
    backtest walking many trade dates for the same ticker shouldn't refetch
    it on every decision. Fundamentals Analyst was assumed unaffected — its
    own `get_fundamentals` tool returns the real name via yfinance's
    `longName`/KIS's own data, so it was thought immune — but that
    assumption was wrong: confirmed live (2026-07-24) writing up correct KIS
    financial data for 000660.KS under a completely different company's name
    across two separate backtest dates, so it now takes this same grounding
    as the other three.

    "Authoritative" aliases are names sourced the same way `display_name`
    itself is (KIS/KRX/overseas master file) — safe to also inject as the
    displayed label. "Weak" aliases are yfinance's `longName`/`shortName`,
    included ONLY as gate-passing aliases, never as the injected label:
    yfinance's own name can itself be cross-symbol-confused for a `.KS` code
    (documented incident: 086790.KS/하나금융지주 -> yfinance "KB금융" — see
    app.dataflows.yfinance.research's docstring), so treating it as ground
    truth would let the gate accept the exact class of hallucination it
    exists to catch. `identity_gate.enforce_identity` logs (INFO) whenever a
    report only passes via a weak alias, so this deliberately-accepted weak
    spot stays auditable rather than silent.

    app.live.ticker_names (KIS's own hts_kor_isnm for an exact KRX code —
    populated by app.live.universe.krx_top_n) is checked first for a `.KS`
    ticker's display_name, ahead of the yfinance-sourced cache below — same
    fix and reason as app.web.state.get_display_name's identical ordering
    bug: real incident (2026-07-23), yfinance's `get_ticker_display_info`
    returned "에코프로비엠" (a real, different company) for 064260.KS (actually
    다날), which then anchored the Market Analyst's whole report to a wrong
    company's typical price range, echoing that instead of the real ~₩4,560
    quote. KIS can't suffer that cross-symbol confusion for an exact code,
    so it wins whenever known. Unlike the pre-P0-3 version of this function,
    this `.KS`-with-ticker_names path is now cached too (it now also does a
    yfinance call, for the weak-alias set, that must not repeat per
    analyst) — it used to bypass the cache and lock entirely because it had
    nothing to cache besides a dict lookup."""
    if ticker not in _company_identity_cache:
        lock = _company_identity_locks.setdefault(ticker, asyncio.Lock())
        try:
            async with lock:
                # Re-check after acquiring: whichever caller got here first
                # has already populated the cache by the time a racing
                # caller's await on the lock returns.
                if ticker not in _company_identity_cache:
                    kor_name = ticker_names.get(ticker) if ticker.endswith(".KS") else None
                    if ticker.endswith(".KS"):
                        await krx_master.ensure_loaded()
                        master_name = krx_master.get(ticker)
                    elif _is_us(ticker):
                        # overseas_master's value is a KOREAN TRANSLITERATION
                        # (nasmst.cod col 6, e.g. QQQ->"인베스코 QQQ 트러스트") —
                        # kept as an alias below, but NEVER as display_name:
                        # handing it to an English-output LLM as the "verified"
                        # label is the confirmed cause of this ticker class's
                        # 2026-08-20 corruption (see kis_provider.get_ticker_
                        # display_info's matching fix). English display_name
                        # for a US ticker comes from that function instead,
                        # same as before this identity work.
                        await overseas_master.ensure_loaded()
                        master_name = overseas_master.get(ticker)
                    else:
                        master_name = None
                    display_name = kor_name or (master_name if ticker.endswith(".KS") else None)
                    if display_name is None:
                        try:
                            info = await get_ticker_display_info(ticker)
                            display_name = info.name
                        except Exception:
                            display_name = ticker
                    yf_name: str | None = None
                    yf_short_name: str | None = None
                    try:
                        yf_info = await yf_get_ticker_display_info(ticker)
                        yf_name = yf_info.name
                        # `shortName` is a SEPARATE weak alias, not folded
                        # into `yf_name` — 2026-08-28, see
                        # TickerDisplayInfo.short_name's own docstring for
                        # why a report can match one but not the other.
                        yf_short_name = yf_info.short_name
                    except Exception:
                        pass  # weak-alias set just lacks this entry
                    kis_english: str | None = None
                    if ticker.endswith(".KS"):
                        try:
                            kis_english = await get_kis_english_name(ticker)
                        except Exception:
                            kis_english = None
                    logger.info(
                        "identity_source: %s kor=%r kis_en=%r yf=%r yf_short=%r",
                        ticker, display_name, kis_english, yf_name, yf_short_name,
                    )
                    # Names only. The ticker and the bare KRX code used to be
                    # passed here too; they are not names, and both identity
                    # detectors accept "any alias appears" as a pass, so
                    # including them made the gate self-satisfying — see
                    # identity_gate.check_identity's own ticker_forms
                    # subtraction, which now enforces this regardless of what
                    # a caller passes.
                    authoritative = build_alias_set(display_name, kor_name, master_name)
                    weak = build_alias_set(yf_name, yf_short_name, kis_english) - authoritative
                    _company_identity_cache[ticker] = (display_name, authoritative, weak, kis_english or "")
                    if len(_company_identity_cache) > _COMPANY_NAME_CACHE_MAX:
                        _company_identity_cache.popitem(last=False)
        finally:
            # Best-effort cleanup — a racing caller already holding this
            # same lock object (via the setdefault above) is unaffected by
            # its dict entry disappearing; a brand new caller for this
            # ticker never gets here again now that the cache is populated.
            _company_identity_locks.pop(ticker, None)
    else:
        _company_identity_cache.move_to_end(ticker)
    return _company_identity_cache[ticker]


# (ticker, trade_date, as_of, language) -> (cached_at, reports). Re-asking
# about the same ticker/day within the TTL reuses the four analyst reports
# instead of re-running every analyst's tool loop from scratch — a repeated
# deep_dive question was previously full-cost every time. `language` is part
# of the key so a Korean question never serves up reports an earlier English
# question generated (or vice versa) — see app.llm.language. 15 minutes
# (user-chosen freshness/speed tradeoff): short enough that intraday price/
# news moves still show up within one chat session, long enough to skip the
# repeat cost of a follow-up question minutes later. No cross-process
# persistence (same convention as `_company_name_cache`) — a restart just
# cold-starts it.
_REPORTS_CACHE_TTL_SECONDS = 15 * 60
_reports_cache: dict[tuple[str, str, date | None, str], tuple[float, dict[str, str]]] = {}


def _cached_reports(ticker: str, trade_date_str: str, as_of: date | None, language: str) -> dict[str, str] | None:
    hit = _reports_cache.get((ticker, trade_date_str, as_of, language))
    if hit is None:
        return None
    cached_at, reports = hit
    if time.monotonic() - cached_at > _REPORTS_CACHE_TTL_SECONDS:
        return None
    return reports


def _store_cached_reports(
    ticker: str, trade_date_str: str, as_of: date | None, language: str, reports: dict[str, str]
) -> None:
    """Sweeps expired entries before inserting — a TTL-only cache with no
    eviction just grows for the life of the process (every distinct
    ticker/date/as_of/language combo a long-running daemon or a multi-year
    backtest ever sees stays in memory forever, expired or not). This keeps
    the cache's actual size bounded to roughly "entries written within the
    last TTL window," matching what the TTL already implies it should be.
    쓰기 시점마다 만료된 항목을 정리해서 캐시가 프로세스 수명 내내 무한정 커지지 않도록 함."""
    now = time.monotonic()
    expired = [key for key, (cached_at, _) in _reports_cache.items() if now - cached_at > _REPORTS_CACHE_TTL_SECONDS]
    for key in expired:
        del _reports_cache[key]
    _reports_cache[(ticker, trade_date_str, as_of, language)] = (now, reports)


StageCallback = Callable[[str], Awaitable[None]]


async def _emit(on_stage: StageCallback | None, label: str) -> None:
    if on_stage is not None:
        await on_stage(label)


async def run_pipeline(
    ticker: str,
    trade_date: date,
    as_of: date | None = None,
    on_stage: StageCallback | None = None,
    portfolio_cap_override: tuple[PortfolioSnapshot, dict[str, int], float] | None = None,
    language: str = DEFAULT_LANGUAGE,
    profile: AIProfile | None = None,
    force_refresh: bool = False,
    decision_trigger: str = "auto",
) -> PipelineState:
    """Public entry point — see `_run_pipeline_impl` for the actual run.
    Just scopes `candle_run_cache()` to this one call, so Market Analyst's
    price/indicator/snapshot tool calls and this run's own get_atr_pct/
    get_last_close share one underlying KIS candle fetch per ticker instead
    of each firing its own (pure efficiency — no analysis behavior change).
    One call = one cache; two different `run_pipeline` calls (even for the
    same ticker, e.g. a live check followed later by a backtest date) never
    share candle data. Also scopes `llm_metrics.track_run_tokens()` the same
    way, so `_run_pipeline_impl`'s `store_pending` call can read this one
    run's own total LLM token usage (see that function)."""
    with candle_run_cache(), llm_metrics.track_run_tokens(), llm_metrics.track_run_sanitizations(), llm_metrics.track_run_injection_flags():
        return await _run_pipeline_impl(
            ticker, trade_date, as_of, on_stage, portfolio_cap_override, language, profile, force_refresh,
            decision_trigger,
        )


async def _run_pipeline_impl(
    ticker: str,
    trade_date: date,
    as_of: date | None = None,
    on_stage: StageCallback | None = None,
    portfolio_cap_override: tuple[PortfolioSnapshot, dict[str, int], float] | None = None,
    language: str = DEFAULT_LANGUAGE,
    profile: AIProfile | None = None,
    force_refresh: bool = False,
    decision_trigger: str = "auto",
) -> PipelineState:
    """`as_of` bounds every analyst's data access to what was knowable on
    that date (Phase 2 backtesting). `None` (default) means live/current —
    unchanged Phase 1 behavior. `on_stage`, if given, is awaited with a short
    human-readable label at each major step — purely a UX hook (e.g. the
    live chat's progress display); it's a no-op by default so the CLI and
    backtest harness are unaffected. `portfolio_cap_override`, if given, is a
    precomputed (snapshot, clusters, drawdown_fraction) for the multi-ticker
    backtest harness to supply when `as_of` is set — without it, every
    backtest call falls back to `DISABLED_PORTFOLIO_CAPS` (see below), which
    would mean the portfolio-level caps could never actually trigger in any
    backtest. Ignored when `as_of is None` (live always fetches its own
    current state instead). `language` ("ko"/"en", see app.llm.language) is
    threaded into every LLM call in this run — analysts, both debates, and
    every judgment node — so the whole transcript answers in whichever
    language the user's own question was in, not a hardcoded default.
    `profile` (see app.config.AIProfile) only reaches the two decision-
    producing nodes (trader, portfolio manager) — analysts and debates are
    unaffected. `None` (default) resolves via `runtime_settings.rule` (the
    dashboard override if set, else `settings.ai_profile`) so every existing
    caller picks up whatever is currently in effect without code changes;
    callers that need a fixed profile regardless of config/override (e.g.
    the backtest harness, for determinism) pass one explicitly. `force_refresh`
    (default False) skips the 15-minute analyst-report cache lookup below —
    for a caller whose whole reason for running is "the user explicitly
    asked to see this analyzed again right now" (the SEMI_AUTO "다시 분석"
    button, see app.live.auto_trade_loop.run_cycle_now), silently serving
    reports from up to 15 minutes ago would defeat the point of the request.
    The cache is still WRITTEN to after a forced run, so a later non-forced
    call (e.g. a chat follow-up) benefits from it as usual — this only
    bypasses the READ for this one call. `decision_trigger` ("auto", the
    default, or "manual") is a pure passthrough to `store_pending` — this
    function never derives or overrides it. The default preserves the
    auto-trade loop's existing behavior with no call-site change; only the
    entry points where a human picked THIS SPECIFIC ticker out of band (the
    Trading tab's analyze button, a chat deep-dive, CLI/dev-script runs)
    pass "manual" explicitly, so app.audit can exclude sample-selection-
    biased one-off analyses from the win-rate/confidence-calibration numbers
    used to gate capital increases — see the column's own comment in
    app.memory.db."""
    resolved_profile = profile if profile is not None else runtime_settings.rule("ai_profile")
    state = PipelineState(ticker=ticker, trade_date=trade_date)
    trade_date_str = trade_date.isoformat()

    is_kr = ticker.endswith(".KS")
    # Pure and I/O-free (session_state.session_in_progress) — so, unlike
    # get_verified_snapshot below, this is computed on BOTH the cache-hit and
    # analyst-run paths: a cache hit skips the analysts, not the market clock,
    # and an intraday re-analysis is exactly when the downstream narration is
    # the only defense. Empty string (closed session / backtest) keeps every
    # downstream prompt byte-identical. 2026-09-01, id=1157 — the Market
    # Analyst's "session still open" warning never reached the PM.
    session_note = intraday_narration_note(ticker, trade_date)
    state.session_in_progress = bool(session_note)
    # The 억원 ground truth for the numeric guards (see PipelineState.
    # unit_scale_scans / decision id=1158). Filled by fundamentals_analyst on
    # the non-cache path; empty on a cache hit (the guards then fall back to
    # their self-contradiction-only check — the documented fail-open).
    kr_ground_truth: list[str] = []
    kr_gt = ""
    # The raw KIS news-headline block, captured from news_macro_analyst on the
    # non-cache path — the SOFT ground truth for the numeric guards so a
    # headline 억 amount reported 10x large (2026-09-07 audit #3, id=1237) is
    # flagged in this and every downstream node. Rebound at the bottom of the
    # `else` below; on a report-CACHE hit it stays "" for the whole cycle, so
    # `news_eok_10x` cannot fire at all that cycle — deliberate, same
    # no-second-fetch fail-open as `kr_gt` / `verified_snapshot` above (a
    # re-fetch per cached decision re-opens the EGW00215 duplicate-fetch
    # class). Residual risk is small: any cycle that DID carry a unit
    # violation is withheld from the report cache (see the cache-write guard
    # below), so a cached report is never one that would have flagged here.
    kr_news_block_out: list[str] = []
    kr_news_gt = ""
    unit_violations: list[str] = []
    unit_scans: dict[str, dict[str, int]] = {}
    # The interim-period warning (audit #5, id=1158) — filled by
    # fundamentals_analyst on the non-cache path when the most recent Korean
    # financial row is a leading interim period; empty on a cache hit (same
    # no-second-KIS-fetch fail-open as kr_ground_truth) and for a non-KRX
    # ticker. Threaded to the same 4 downstream nodes as session_note.
    kr_period_notes: list[str] = []
    period_note = ""
    # Detect-only findings from app.agents.period_mismatch_guard (audit #5
    # §4) — mirrors unit_violations: filled on the non-cache path, moved to
    # state.period_mismatch_violations after the analyst loop, extended
    # directly on state by the post-loop debate/PM guards.
    period_violations: list[str] = []
    # Tier-1 (strong) subset of period_violations — a Fundamentals-Analyst
    # finding that ALSO carries an explicit period frame (period_mismatch_guard
    # §3). As of 2026-09-03 ONLY this subset gates the report cache; a Tier-2
    # finding (any other node, or no period frame) is recorded but not
    # cache-blocking. Re-derived from the Fundamentals report after the analyst
    # gather (fundamentals_analyst.analyze keeps its old signature — it does
    # not thread this out-param).
    period_strong_violations: list[str] = []
    # {node: PeriodMismatchScan.as_dict()} — the positive scan record; see
    # PipelineState.period_mismatch_scans.
    period_scans: dict = {}
    # Out-param for market_analyst.analyze's injected technical-indicator
    # block (2026-09-07 fix) — either a real get_indicators value or the
    # "NOT AVAILABLE" fallback text. Filled on the non-cache path only, then
    # used below to re-scan the Market Analyst's own report for a hedged/
    # ungrounded claim (app.agents.technical_claim_guard).
    indicator_block_out: list[str] = []
    # Out-param for market_analyst.analyze's injected investor-flow block
    # (2026-09-11 fix) — either a real get_investor_flow value or the
    # "NOT AVAILABLE" fallback text. Filled on the non-cache path only, same
    # "unknown, not known-clean" treatment as indicator_block_out above.
    flow_block_out: list[str] = []
    # Out-param for market_analyst.analyze's injected long-horizon (52-week)
    # block (2026-09-16 fix) — either a real get_long_horizon_context value
    # or the "NOT AVAILABLE" fallback text. Filled on the non-cache path
    # only, same "unknown, not known-clean" treatment as the two above.
    long_horizon_block_out: list[str] = []
    # How stale get_kr_financials' newest exchange-reported period is
    # (2026-09-07 audit #6, id=1237/294630: KIS's annual series ended at
    # FY2024, 20 months out, unflagged). Filled by fundamentals_analyst on
    # the non-cache path for a KRX ticker; stays {} on a cache hit and for a
    # US ticker (same no-second-fetch fail-open as kr_ground_truth).
    kr_financials_recency: dict = {}
    # Fetched by the non-cache branch below (shared with Market/News+Macro
    # Analyst prompts); stays "" on a cache hit — same no-second-fetch
    # fail-open as kr_ground_truth/kr_financials_recency above.
    # app.agents.verified_price_guard.scan_verified_price_claims already
    # treats an empty snapshot as "no ground truth to check against" and
    # returns [] immediately, so this default is safe on that path too.
    # Fetched by the non-cache branch below (shared with Market/News+Macro
    # Analyst prompts); stays "" on a cache hit — same no-second-fetch
    # fail-open as kr_ground_truth/kr_financials_recency above.
    # app.agents.verified_price_guard.scan_verified_price_claims already
    # treats an empty snapshot as "no ground truth to check against" and
    # returns [] immediately, so this default is safe on that path too.
    verified_snapshot: str = ""

    cached_reports = None if force_refresh else _cached_reports(ticker, trade_date_str, as_of, language)
    if cached_reports is not None:
        await _emit(on_stage, "Reusing recent analyst reports for this ticker/date...")
        state.reports = dict(cached_reports)
        # Cached reports already passed the identity gate when first produced
        # (a violating run is never cached — see the `else` branch below), but
        # the identity note threaded to the downstream stages still needs the
        # display name.
        company_name, _identity_aliases, _identity_weak_aliases, _identity_kis_english = await _company_identity(ticker)
        state.identity_violations = []
        state.identity_unnamed = []
        # A cached run was necessarily a CLEAN run — see the `if not
        # degraded_analysts and not identity_violations and not
        # unit_scale_violations` guard below, which withholds any degraded
        # cycle from the cache. So this is a known empty, not an unknown.
        state.degraded_analysts = []
        state.unit_scale_violations = []
        state.unit_scale_scans = {}
        state.period_mismatch_violations = []
        state.period_mismatch_scans = {}
        # A cached run was necessarily a CLEAN run for this guard too (see the
        # same withholding-guard below) — but whether the indicator block was
        # a real value or NOT AVAILABLE isn't preserved by the cache, so this
        # is "unknown", not "known clean" (see PipelineState.
        # indicator_block_available's own docstring).
        state.technical_claim_violations = []
        # NOT the "known clean" case the guards above describe: a cache hit
        # re-scanned no report this cycle, so an empty list here means "not
        # scanned", not "clean report". And because refusal_guard is
        # deliberately NOT wired into the report-cache gate below, a report
        # containing a refusal preamble can itself have been cached — so
        # cache-hit cycles UNDERCOUNT the true refusal rate (2026-09-14).
        state.refusal_violations = []
        state.staleness_narrative_violations = []
        state.indicator_block_available = False
        # Same unknown-not-known-clean treatment as indicator_block_available
        # above — the cache doesn't preserve whether the investor-flow block
        # was real (2026-09-11 fix).
        state.flow_block_available = False
        # Same unknown-not-known-clean treatment — the cache doesn't preserve
        # whether the long-horizon block was real (2026-09-16 fix).
        state.long_horizon_block_available = False
        # A cache hit did no get_kr_financials fetch, so recency is unknown
        # here (not "fresh") — same no-second-fetch fail-open as kr_ground_truth.
        state.kr_financials_recency = {}
        state.news_amount_scans = {}
    else:
        await _emit(on_stage, "Running Market, Fundamentals, News/Macro, and Sentiment analysts...")
        # Fetched once, up front, and handed to BOTH Market Analyst and News +
        # Macro Analyst — not two independent fetches. Real incident this
        # fixes (066570.KS, 2026-08-10): News + Macro Analyst had no price
        # tool of its own, so a news headline's own stale intraday "-2.05%"
        # recap got reported as the day's actual move, flatly contradicting
        # the verified close (+1.3%) Market Analyst fetched separately —
        # nothing existed for either analyst to reconcile against. Sharing
        # the same fetch keeps total API calls unchanged (previously only
        # Market Analyst fetched this) while giving News + Macro Analyst
        # something authoritative to check a headline's price claim against.
        (company_name, identity_aliases, identity_weak_aliases, identity_kis_english), verified_snapshot = await asyncio.gather(
            _company_identity(ticker), get_verified_snapshot(ticker, as_of=as_of)
        )
        # return_exceptions=True: one analyst hitting a dropped/timed-out LLM
        # connection used to cancel every sibling call and take down the whole
        # deep_dive, discarding any reports that had already succeeded. A failed
        # analyst now degrades to a placeholder note instead, so the debate/
        # research stages still get the other three real reports.
        analyst_names = ["Market Analyst", "Fundamentals Analyst", "News + Macro Analyst", "Sentiment Analyst"]
        # Extended in-place by fundamentals_analyst.analyze (out-param, not a
        # return-value change — see that function's own docstring) with any
        # app.agents.unit_scale_guard correction it applied. Declared before
        # the gather since analyze() mutates it as a side effect during the
        # call, not after.
        results = await asyncio.gather(
            market_analyst.analyze(
                ticker, trade_date_str, as_of=as_of, company_name=company_name, language=language,
                verified_snapshot=verified_snapshot, indicator_block_out=indicator_block_out,
                flow_block_out=flow_block_out, long_horizon_block_out=long_horizon_block_out,
            ),
            fundamentals_analyst.analyze(
                ticker, trade_date_str, as_of=as_of, company_name=company_name, language=language,
                unit_violations_out=unit_violations, unit_scans_out=unit_scans,
                kr_ground_truth_out=kr_ground_truth, period_note_out=kr_period_notes,
                period_violations_out=period_violations, recency_out=kr_financials_recency,
            ),
            news_macro_analyst.analyze(
                ticker, trade_date_str, as_of=as_of, company_name=company_name, language=language,
                verified_snapshot=verified_snapshot, news_block_out=kr_news_block_out,
            ),
            sentiment_analyst.analyze(ticker, trade_date_str, as_of=as_of, company_name=company_name, language=language),
            return_exceptions=True,
        )
        state.reports = {}
        # Names, not just a bool: WHICH analyst degraded matters — a missing
        # Market Analyst report means the decision had no verified price/
        # indicator view at all, while a missing Sentiment Analyst one is a
        # far milder handicap. This list being non-empty still does
        # `any_failed`'s original job of refusing to cache a degraded run
        # below; the count also goes to memory_entries.analysts_failed
        # (app.memory.db) so a win-rate query can filter in SQL without
        # opening the transcript blob.
        degraded_analysts: list[str] = []
        identity_violations: list[str] = []
        identity_unnamed: list[str] = []
        business_description_violations: list[str] = []
        kr_gt = kr_ground_truth[0] if kr_ground_truth else ""  # noqa: F841 (rebinds the hoisted default)
        kr_news_gt = kr_news_block_out[0] if kr_news_block_out else ""  # noqa: F841 (rebinds hoisted default)
        period_note = kr_period_notes[0] if kr_period_notes else ""  # noqa: F841 (rebinds hoisted default)
        state.kr_financials_recency = dict(kr_financials_recency)
        for name, result in zip(analyst_names, results):
            # BaseException, not Exception: `asyncio.CancelledError`
            # subclasses BaseException, so an `Exception` test returned False
            # for it and let the exception OBJECT through the `else` branch as
            # if it were report text — `enforce_identity` then raised
            # `TypeError: normalize() argument 2 must be str, not
            # CancelledError` and took the whole decision down with no
            # recorded degradation at all. Confirmed 2026-08-25: gather's
            # `return_exceptions=True` above DOES capture a CancelledError as
            # a result value, so it reached this line every time. The
            # upstream half of this fix is in `tool_loop._call_one`; this is
            # the backstop for any other BaseException an analyst can produce.
            if isinstance(result, BaseException):
                # Backstop only — in practice a genuine outer cancellation
                # cancels the gather await itself, so we never get here with
                # one. Kept because swallowing a real cancellation as a
                # "data-quality degradation" would be wrong in kind: it is
                # this task being told to stop, not an analyst producing bad
                # output. Same classifier as every other cancellation
                # decision in this file.
                if isinstance(result, asyncio.CancelledError) and task_was_actually_cancelled():
                    raise result
                degraded_analysts.append(name)
                logger.warning("%s failed for %s — degrading to a placeholder report", name, ticker, exc_info=result)
                state.reports[name] = f"(리포트를 생성하지 못했습니다 — {type(result).__name__}: {result})"
            else:
                # Verifies the analyst actually named the ticker it was asked
                # to analyze rather than a hallucinated/mistranslated company —
                # see app.agents.identity_gate's module docstring for the
                # 316140.KS "KB Financial Group" incident this exists to catch.
                # Runs on every analyst's output (not just the historically-
                # implicated Fundamentals Analyst): the same failure mode has
                # been observed from Market Analyst too (086790.KS).
                checked, offending = enforce_identity(
                    result, ticker, company_name, identity_aliases,
                    weak_aliases=identity_weak_aliases, language=language,
                    kis_english_name=identity_kis_english, agent_name=name,
                )
                # The Fundamentals Analyst already ran its own numeric_guard
                # inside analyze(); the other analysts never did — id=1157's
                # Market report and id=1156's News+Macro report each carried a
                # magnitude / ratio-label error only Fundamentals was checked
                # for (2026-09-01 first real-account audit).
                if is_kr and name != "Fundamentals Analyst":
                    checked, numeric_desc = enforce_numeric_guards(
                        checked, node=name, kr_block=kr_gt, language=language,
                        usd_check=True, soft_eok_block=kr_news_gt, scans_out=unit_scans,
                        period_out=period_violations, period_scans_out=period_scans,
                    )
                    if numeric_desc:
                        unit_violations.extend(numeric_desc)
                state.reports[name] = checked
                if offending == IDENTITY_UNNAMED:
                    # Name omission, not misattribution — a neutral note, and
                    # NOT a cache-blocker (see state.identity_unnamed).
                    identity_unnamed.append(f"{name}: unnamed")
                elif offending is not None:
                    identity_violations.append(f"{name}: {offending!r}")
                    # Detect-only: scan for a fabricated business/industry
                    # self-description seeded by the wrong name — see
                    # app.agents.business_description_guard's own docstring
                    # (id=1237, 294630.KS/서남). Runs on the SAME report text
                    # `enforce_identity` already banner-corrected (`checked`),
                    # not the pre-banner `result` — the guard's own scope
                    # (identity_gate's general-case banner only) already
                    # excludes UNNAMED/ForeignListingName via `offending`'s
                    # value, so no double-gating is needed here.
                    business_description_violations.extend(
                        scan_business_description(
                            checked, offending_name=offending, company_name=company_name, agent_name=name,
                        )
                    )
        if len(identity_violations) >= 3:
            # 3+ of 4 analysts flagged for a WRONG company (omissions excluded)
            # is far more likely the ALIAS SET being wrong than 3 independent
            # hallucinations — the 064260.KS/028050.KS incidents had exactly
            # this shape. Log-only, deliberately not an auto-downgrade: a real
            # systemic misgrounding must still surface as violations.
            logger.warning(
                "identity_gate: %d/%d analysts flagged for %s (display_name=%r) — "
                "suspect the alias set, not the reports; check _company_identity's sources",
                len(identity_violations), len(analyst_names), ticker, company_name,
            )
        # Only cache a clean run — caching a degraded placeholder, or a report
        # an identity violation forced a correction banner onto, would keep
        # serving that stale/corrected text for the full TTL instead of
        # letting the next run re-derive a clean one. A name OMISSION
        # (identity_unnamed) is deliberately NOT in this guard: its neutral
        # note is stable and a legitimately name-free pure-macro report
        # shouldn't cost every .KS/backtest decision its report reuse.
        #
        # Tier-1 gate input: re-scan the Fundamentals Analyst's OWN report for
        # the strong (period-framed, Fundamentals-sourced) subset + the scan
        # record. Done here rather than through analyze()'s out-params so that
        # function's signature is unchanged. Detect-only — no text is altered;
        # findings not already recorded by the analyst's internal guard call
        # are folded into period_violations so the strong subset stays a
        # subset of the recorded full set.
        fa_report = state.reports.get("Fundamentals Analyst", "")
        if is_kr and fa_report and kr_gt:
            _fa_scans: list = []
            _fa_findings = scan_period_mismatch(
                fa_report, kr_gt, agent_name="Fundamentals Analyst",
                strong_out=period_strong_violations, scan_out=_fa_scans,
            )
            for _d in _fa_findings:
                _entry = f"Fundamentals Analyst: {_d}"
                if _entry not in period_violations:
                    period_violations.append(_entry)
            if _fa_scans:
                period_scans["Fundamentals Analyst"] = _fa_scans[0].as_dict()
        # Detect-only technical-claim re-scan (2026-09-07 fix): checks the
        # Market Analyst's own report against the indicator block Fix 1
        # injected into its prompt this cycle — see
        # app.agents.technical_claim_guard.scan_technical_claims. Only ever
        # scans the Market Analyst's own text, never the debate/PM turns that
        # merely quote it (narrower scope than halt_status_guard's multi-
        # source scan, by design — see that guard module's own docstring).
        # NOT wired into the report-cache gate above.
        _ma_report = state.reports.get("Market Analyst", "")
        _indicator_block = indicator_block_out[0] if indicator_block_out else ""
        state.indicator_block_available = bool(_indicator_block) and "NOT AVAILABLE" not in _indicator_block
        # Same derivation as indicator_block_available above, for the
        # investor-flow block (2026-09-11 fix) — empty for a non-KR ticker,
        # since flow_block_out is only ever appended to for a KR ticker.
        _flow_block = flow_block_out[0] if flow_block_out else ""
        state.flow_block_available = bool(_flow_block) and "NOT AVAILABLE" not in _flow_block
        # Same derivation as the two above, for the long-horizon block
        # (2026-09-16 fix). Detect-only: no guard scans this block — it
        # records whether a real 52-week context reached the prompt at all.
        _long_horizon_block = long_horizon_block_out[0] if long_horizon_block_out else ""
        state.long_horizon_block_available = (
            bool(_long_horizon_block) and "NOT AVAILABLE" not in _long_horizon_block
        )
        state.technical_claim_violations = scan_technical_claims(
            _ma_report, indicator_block=_indicator_block, agent_name="Market Analyst"
        )
        if state.technical_claim_violations:
            logger.warning(
                "%s: technical_claim_violations recorded — the Market Analyst made a hedged/"
                "ungrounded technical-indicator or index-contribution claim this cycle: %s",
                ticker, state.technical_claim_violations,
            )
        # Detect-only refusal-preamble sweep (2026-09-14 fix): scans the four
        # analysts' own reports for the "as an AI model I have no real-time
        # data / that date is past my training cut-off" preamble the model
        # still writes despite DATE_GROUNDING — see
        # app.agents.refusal_guard.scan_refusal. Scans the text AS STORED in
        # state.reports (i.e. post identity/numeric-guard), deliberately the
        # same choice business_description_guard makes. Analysts only — no
        # measured evidence for the downstream agents yet. NOT wired into the
        # report-cache gate, degraded_analysts or analysts_failed.
        _refusal_violations: list[str] = []
        for _agent_name, _report_text in state.reports.items():
            if _agent_name not in analyst_names:
                continue
            _refusal_violations.extend(scan_refusal(_report_text, agent_name=_agent_name))
        state.refusal_violations = _refusal_violations
        if state.refusal_violations:
            logger.warning(
                "%s: refusal_violations recorded — an analyst wrote an 'as an AI model / "
                "past my training cut-off' refusal preamble instead of analysing this "
                "cycle: %s",
                ticker, state.refusal_violations,
            )
        _fa_report = state.reports.get("Fundamentals Analyst", "")
        state.staleness_narrative_violations = scan_staleness_narrative(
            _fa_report, recency=kr_financials_recency, agent_name="Fundamentals Analyst"
        )
        if state.staleness_narrative_violations:
            logger.warning(
                "%s: staleness_narrative_violations recorded — the Fundamentals Analyst "
                "presented stale exchange-reported financials as the company's current "
                "condition despite the injected FINANCIAL-DATA RECENCY warning (id=1351 "
                "shape): %s",
                ticker, state.staleness_narrative_violations,
            )
        if (
            not degraded_analysts and not identity_violations and not unit_violations
            and not period_strong_violations
        ):
            _store_cached_reports(ticker, trade_date_str, as_of, language, dict(state.reports))
        state.identity_violations = identity_violations
        state.identity_unnamed = identity_unnamed
        state.business_description_violations = business_description_violations
        if state.business_description_violations:
            logger.warning(
                "%s: business_description_violations recorded — an analyst report described a "
                "business/industry inconsistent with the registered company name after an "
                "identity violation this cycle (id=1237 shape): %s",
                ticker, state.business_description_violations,
            )
        state.degraded_analysts = degraded_analysts
        state.unit_scale_violations = unit_violations
        state.period_mismatch_violations = period_violations + state.period_mismatch_violations
        state.period_mismatch_scans = period_scans
        state.unit_scale_scans = unit_scans

    identity_note = (
        f"The company under analysis is {ticker_label(ticker, company_name)}. This "
        "ticker-to-name mapping is verified against the exchange's own registry and "
        "overrides any other name. If a report below names a different company, that "
        "name is wrong — the underlying data is still for this ticker. Do not dismiss "
        "or discount a report's content because of a name mismatch, and do not repeat "
        "the wrong name."
    )

    # Declared here (empty) so `_numeric_turn_guard`'s closure can safely
    # read it during the EARLIER research-debate call, before it's assigned
    # the real fetched value below (~line 1985, right before the risk
    # debate) — see that assignment's own comment for why the fetch itself
    # isn't moved earlier. Python resolves a closure's free variable by
    # reference at CALL time, not at `_numeric_turn_guard`'s definition
    # time, so `guard()` for `research_debate` (called before the
    # assignment below executes) sees "" here, while `guard()` for
    # `risk_debate` (called after it) sees the real fetched value — no
    # separate wiring needed per stage.
    kr_risk_extra = ""
    # Numeric guard for every downstream node that consumes or republishes
    # financial figures (2026-09-01, decisions id=1156/1157/1158 — the guards
    # had been wired to the Fundamentals Analyst alone). Findings are recorded
    # in the same flat list, prefixed with the node. Debate/PM violations
    # arrive after the report-cache gate above, so they never affect caching.
    def _numeric_turn_guard(stage: str):
        def guard(content: str, speaker: str, round_num: int) -> str:
            if not is_kr or not content:
                return content
            # 2026-09-11 (decision id=1386, 006360.KS): the program-trading
            # 억원 figure (`kr_risk_extra`, fetched below at ~line 1985) is
            # ground truth the risk debate needs live, not just in the
            # post-debate `ground_truth_claim_guard` sweep — a real turn
            # correctly wrote "+15.65 billion KRW" and a later turn
            # "corrected" it to the wrong "+1.565 billion KRW", calling the
            # correct figure a fabrication, and this per-turn guard had no
            # way to catch a `billion_10x`-shaped miscorrection against it in
            # real time. Concatenating it here only matters for whichever
            # call of this SAME closure happens to run after `kr_risk_extra`
            # is populated — i.e. only `risk_debate` in practice, per the
            # closure-timing mechanism at this variable's declaration above.
            # The research debate's call to this closure still sees
            # `kr_risk_extra == ""` at that point in execution (it runs
            # before the fetch, deliberately — see id=1170/498400.KS at the
            # fetch's own comment for why that ordering isn't changed), so
            # it is unaffected.
            guarded, desc = enforce_numeric_guards(
                content, node=f"{stage}[r{round_num} {speaker}]",
                kr_block=f"{kr_gt}\n\n{kr_risk_extra}" if kr_risk_extra else kr_gt,
                language=language, usd_check=True, soft_eok_block=kr_news_gt,
                scans_out=unit_scans,
                period_out=state.period_mismatch_violations,
                period_scans_out=state.period_mismatch_scans,
            )
            if desc:
                state.unit_scale_violations.extend(desc)
            return guarded
        return guard

    def _guard_fields(obj, fields: list[str], node_prefix: str, **kw):
        """Run the numeric guard over each named text field of a pydantic
        object, returning a dict of the changed ones (for model_copy) and
        recording every finding. Structured fields use banner=False — sibling
        fields would each repeat the same banner."""
        updates: dict[str, str] = {}
        if not is_kr:
            return updates
        for f in fields:
            val = getattr(obj, f, None)
            if not isinstance(val, str) or not val:
                continue
            guarded, desc = enforce_numeric_guards(
                val, node=f"{node_prefix}.{f}", kr_block=kr_gt, language=language,
                usd_check=True, soft_eok_block=kr_news_gt, banner=False, scans_out=unit_scans,
                period_out=state.period_mismatch_violations,
                period_scans_out=state.period_mismatch_scans, **kw,
            )
            if desc:
                state.unit_scale_violations.extend(desc)
            if guarded != val:
                updates[f] = guarded
        return updates

    research_speakers = {
        "Bull": bull_researcher.build(
            state.reports, language=language, identity=identity_note, session_note=session_note
        ),
        "Bear": bear_researcher.build(
            state.reports, language=language, identity=identity_note, session_note=session_note
        ),
    }
    state.research_debate = await run_debate(
        state=state.research_debate,
        speakers=research_speakers,
        max_rounds=settings.max_debate_rounds,
        on_round=lambda n: _emit(on_stage, f"Bull vs. Bear debate — round {n}/{settings.max_debate_rounds}..."),
        language=language,
        # Measured live (200 decisions, 2026-08-21): this debate's
        # convergence check flagged "converged" only 4% of the time — an
        # extra LLM call paid on every decision for a near-zero expected
        # saving. Bull/Bear is structurally adversarial, so round 2 reliably
        # has something new to say. The risk debate below was found
        # (2026-08-25) to have the SAME check disabled, for an unrelated
        # reason (its 57.5% "convergence" was an artifact of comparing round 1
        # against an empty prior-rounds section, not real early agreement) —
        # see run_debate's check_convergence docstring and its own call site.
        check_convergence=False,
        parallel_first_round=settings.debate_parallel_first_round,
        turn_guard=_numeric_turn_guard("research_debate"),
    )

    # Fetched here — right before the first stage that reasons about
    # direction/sizing/action (the Research Manager, as of 2026-08-28; was
    # the Trader before that), not later alongside the other context_tasks
    # right before the Portfolio Manager — so the Research Manager, Trader,
    # and risk debators aren't reasoning about position "blind" the way
    # _position_context's own docstring already describes the Portfolio
    # Manager once did (the real NAVER 255-vs-1-share incident). That
    # incident was already fixed at the Portfolio Manager; this closes the
    # same gap two steps earlier, before any LLM call downstream of here can
    # produce a recommendation/proposal/risk-debate argument that reads
    # oddly against the real position (same "code is right, prose is wrong"
    # pattern the ATR-scaling fix elsewhere in this file already guards
    # against). Computed once, reused at the Trader and Portfolio Manager
    # calls below instead of being fetched again.
    #
    # Deliberately NOT moved further up, ahead of the Bull/Bear debate
    # (research_speakers/run_debate above): that debate argues pure
    # direction and has no need to know the position, and moving a DB+KIS
    # round-trip (position/trade-history/stop-loss/real-balance reads, one
    # of which can time out against a slow/unreachable broker) ahead of it
    # would add that latency to every decision's critical path before any
    # of it is actually needed — 2026-08-28, see `held`'s own new consumers
    # below for why this exact spot is early enough.
    held, position_context = await _position_context(ticker, as_of)

    await _emit(on_stage, "Research Manager synthesizing the debate...")
    with capture_research_manager(
        ticker=ticker, trade_date=trade_date_str, market="domestic" if is_kr else "overseas",
        trigger=decision_trigger, held=held, session_in_progress=state.session_in_progress,
        historical=as_of is not None or is_backtest_context(),
    ):
        state.research_plan = await research_manager.synthesize(
            state.reports, state.research_debate, language=language, identity=identity_note, held=held,
            session_note=session_note, period_note=period_note, position_context=position_context,
        )
    if state.research_plan is not None:
        _plan_updates = _guard_fields(
            state.research_plan, ["rationale", "strategic_actions"], "research_plan",
        )
        if _plan_updates:
            state.research_plan = state.research_plan.model_copy(update=_plan_updates)

    await _emit(on_stage, "Trader drafting a proposal...")
    state.trader_proposal = await trader.propose(
        state.reports, state.research_debate, state.research_plan, position_context,
        language=language, profile=resolved_profile, identity=identity_note, held=held,
        session_note=session_note, period_note=period_note,
    )
    if state.trader_proposal is not None:
        _trader_updates = _guard_fields(
            state.trader_proposal, ["reasoning", "position_sizing"], "trader_proposal",
        )
        if _trader_updates:
            state.trader_proposal = state.trader_proposal.model_copy(update=_trader_updates)

    # Korean risk-side signals (daily short-selling intensity, program-trade
    # net-buy) — distinct pressures yfinance doesn't surface for KRX names.
    # Self-gated to .KS tickers here too (not just inside get_kr_risk_signals)
    # so a non-KRX ticker's risk debate isn't fed a "not supported" string.
    # Position context rides along in the same extra_context slot, so the
    # risk debators see the CURRENT POSITION (share count, avg cost, the stop
    # already on record). That is all it ever conveyed: it says nothing about
    # the 1% risk cap, the equity basis, or ATR scaling, and until 2026-09-02
    # the debators had no access to any of those — id=1170 (498400.KS), where
    # Conservative asserted "we cannot reduce position size (it's
    # all-or-nothing)" and Neutral's deciding turn adopted it as verified
    # fact. The real cap permits a share count that varies with the stop
    # distance; the sizing block threaded below is the fix.
    kr_risk_extra = await get_kr_risk_signals(ticker, as_of=as_of) if ticker.endswith(".KS") else ""
    risk_extra = f"{position_context}\n\n{kr_risk_extra}" if kr_risk_extra else position_context
    # 2026-09-09 P1 audit item #1: recorded (not just used) so a stored
    # transcript can later show whether a debate turn's program-trade/
    # short-selling figure came from this real feed — see
    # PipelineState.kr_risk_signals's own docstring.
    state.kr_risk_signals = kr_risk_extra

    # 2026-09-02 (decision id=1170, 498400.KS): the risk debate previously argued
    # sizing blind to the real cap — Conservative asserted "we cannot reduce
    # position size (it's all-or-nothing)" and Neutral's deciding turn repeated it
    # as fact; the real cap varies with the stop distance argued for. This computes
    # the SAME risk_cap_context/volatility_context the Portfolio Manager sees
    # further below, early, from cache-hit data (candle_run_cache covers this whole
    # run) plus one resolve_equity_basis call that is backtest-safe (as_of is not
    # None short-circuits before any broker call) — NOT a second, drifting
    # implementation. If the price fetch fails here, the debate simply proceeds
    # without this block (today's behavior), and the PM-facing call further below
    # falls back to computing risk_cap_context itself.
    debate_volatility_context = ""
    risk_cap_context: str | None = None
    try:
        # The raw ATR% return here is discarded (assigned to `_`) and never
        # used for anything execution-related — only the narrative string is
        # reused. This is deliberate duplication of the ATR-context CALL (not
        # of the execution-input VALUE): `current_atr_pct`/`fill_price` that
        # feed `rebalance_to_target`/`target_allocation_for` below still come
        # exclusively from the untouched `context_tasks` gather further down.
        # A future reader should not try to unify this call with that one.
        _, debate_volatility_context = await _atr_context(ticker, as_of, held=held)
        _debate_price = await get_last_close(ticker, as_of=as_of)
        if _debate_price and _debate_price > 0:
            # Computed ONCE and shared with the Portfolio Manager below (see
            # the `if risk_cap_context is None:` fallback there) — never
            # computed twice as separate calls with potentially different
            # account/equity readings. That would reintroduce the exact
            # drift-between-two-computations risk this design exists to
            # prevent. Both volatility_context and risk_cap_context are
            # threaded together (not risk_cap_context alone) because its own
            # text references "the volatility-based scaling above" and would
            # dangle without volatility_context appearing first in the same
            # order the PM sees them. No new ATR-gate-specific text is added
            # here — volatility_context already carries the ATR-unavailable
            # message from the 11ed396 commit, so the debate gets that
            # coverage for free without a second source of truth.
            risk_cap_context = await _risk_cap_context(ticker, _debate_price, as_of=as_of)
    except Exception:
        logger.warning("risk-debate sizing context unavailable for %s", ticker, exc_info=True)
    debate_sizing_context = "\n\n".join(
        p for p in (debate_volatility_context, risk_cap_context) if p
    )

    # held=False (flat ticker) rewrites the debate's rating menu from five
    # levels to two (Buy / Hold) — 2026-09-01, decisions id=1156/1157/1158:
    # all three tickers were flat and all three debates converged on a rating
    # (Underweight / Sell) the PM cannot legally hold for a flat position.
    risk_speakers = risk_debators.build_all(
        state.reports, state.trader_proposal, risk_extra, language=language,
        identity=identity_note, held=held, session_note=session_note, period_note=period_note,
        sizing_context=debate_sizing_context,
        research_plan_note=render_research_plan(state.research_plan) if state.research_plan else "",
    )
    state.risk_debate = await run_debate(
        state=state.risk_debate,
        speakers=risk_speakers,
        max_rounds=settings.max_risk_rounds,
        on_round=lambda n: _emit(
            on_stage, f"Risk debate (Aggressive/Conservative/Neutral) — round {n}/{settings.max_risk_rounds}..."
        ),
        language=language,
        # Was the default (True) until 2026-08-25. The 57.5% convergence rate
        # that justified keeping it on was an artifact of asking the model to
        # compare round 1 against an empty prior-rounds section — see
        # run_debate's `2 <= round_num` guard. Every "convergence" it reported
        # cut this 3-party debate to one round and handed the Portfolio
        # Manager an un-revised first-round Neutral turn as the "FINAL
        # SYNTHESIS" (portfolio_manager._render_risk_debate). Explicitly False
        # rather than relying on the new guard making it a no-op at
        # max_rounds=2: the only evidence for its value is now void, so
        # re-enabling it should be a deliberate act.
        check_convergence=False,
        parallel_first_round=settings.debate_parallel_first_round,
        turn_guard=_numeric_turn_guard("risk_debate"),
    )
    # Detect-only: a flat-ticker debate turn that DECLARED a rating the PM
    # cannot hold (Sell/Underweight/Overweight) — silently discarded
    # downstream, so nothing recorded it before (id=1157/1158). The prompt
    # (build_all held=) is the fix; this is how an operator sees it lapse.
    state.risk_rating_violations = risk_debators.check_illegal_ratings(state.risk_debate, held=held)
    # Detect-only companion to the line above: did Neutral's round-1 verdict
    # survive to the final round, and if not, did the reversing turn cite a new
    # fact (f79213e's Neutral clause) or just re-weigh the same evidence
    # (id=1212/1217)? Pure string analysis of turns already in hand — no LLM
    # call, no effect on what the PM reads.
    state.neutral_reversal = risk_debators.check_neutral_reversal(state.risk_debate)

    await _emit(on_stage, "Checking memory for similar past decisions...")
    situation_text = "\n\n".join(state.reports.values())
    # These three are mutually independent reads (no shared data dependency
    # between them), so they run concurrently instead of one after another;
    # only risk_cap_context needs fill_price's result, so it stays a
    # sequential call right after this gather. _position_context was here
    # too until it moved up above (before the Trader) — held/position_context
    # are already available from that earlier call, reused below rather than
    # fetched a second time.
    context_tasks = [
        get_similar_context(ticker, situation_text, as_of=as_of),
        _atr_context(ticker, as_of, held=held),
        get_last_close(ticker, as_of=as_of),
        # Status-agnostic sibling lookup (index 3) — a separate concept from
        # get_similar_context's resolved-only "graded lessons" (index 0), see
        # get_same_day_sibling_context's own docstring for why the two don't
        # share a query or a state field.
        get_same_day_sibling_context(ticker, trade_date, as_of=as_of, language=language),
    ]
    # Portfolio-level caps (app.execution.portfolio_risk) are LIVE-ONLY in
    # the sense that their high-water mark is persisted and must never be
    # read or written by a backtest (a historical replay inheriting today's
    # live equity state, or vice versa, would trip the drawdown breaker for
    # reasons that have nothing to do with that replay) — see
    # `portfolio_risk.py`'s module docstring. A single-ticker backtest has
    # no real cross-ticker portfolio to coordinate anyway, so it gets the
    # inert `DISABLED_PORTFOLIO_CAPS`. The multi-ticker harness computes its
    # own local-state (snapshot, clusters, running-HWM-derived drawdown) per
    # date and passes it in as `portfolio_cap_override`, so its caps are
    # genuinely exercised without ever touching the live `portfolio_risk_state`
    # table. Folded into the same gather when live, since it's just as
    # independent of the other four as they are of each other.
    if as_of is None:
        context_tasks.append(_portfolio_risk_context(ticker))

    def _build_context_tasks() -> list[Awaitable[Any]]:
        tasks: list[Awaitable[Any]] = [
            get_similar_context(ticker, situation_text, as_of=as_of),
            _atr_context(ticker, as_of, held=held),
            get_last_close(ticker, as_of=as_of),
            get_same_day_sibling_context(ticker, trade_date, as_of=as_of, language=language),
        ]
        if as_of is None:
            tasks.append(_portfolio_risk_context(ticker))
        return tasks

    # Unguarded until 2026-08-25 (see id=377/id=949 in the docstrings below
    # this function) — a spurious `asyncio.CancelledError` from any one task
    # here (e.g. a connection-pool internal wait cancelled by something
    # unrelated) used to propagate straight out of `gather`, and a plain
    # transient failure (a one-off KIS quote timeout, a DB blip) did the
    # same since `return_exceptions` wasn't set — either way discarding the
    # ~9-12 LLM calls (~144k tokens) already spent on trader_proposal/
    # risk_debate above with zero trace in the DB. One retry of the WHOLE
    # gather (all four/five tasks are read-only, so redoing an already-
    # succeeded one is harmless) turns a one-off blip into, at worst, one
    # extra round of these queries instead of a silently vanished decision.
    # A genuine outer cancellation (`task_was_actually_cancelled()`) still
    # propagates immediately either way — that one is this task actually
    # being told to stop, not ours to swallow.
    try:
        context_results = await asyncio.gather(*context_tasks, return_exceptions=True)
        failures = [r for r in context_results if isinstance(r, BaseException)]
        if failures:
            raise failures[0]
    except asyncio.CancelledError:
        if task_was_actually_cancelled():
            raise
        logger.warning(
            "%s: context gather hit an unrequested CancelledError — retrying once "
            "rather than discarding the analysis already paid for (see id=377/id=949)",
            ticker,
        )
        context_results = await asyncio.gather(*_build_context_tasks())
    except Exception:
        logger.warning(
            "%s: a context task failed (not a cancellation) — retrying the whole "
            "gather once rather than discarding the analysis already paid for",
            ticker, exc_info=True,
        )
        context_results = await asyncio.gather(*_build_context_tasks())
    state.past_context = context_results[0]
    current_atr_pct, volatility_context = context_results[1]
    fill_price = context_results[2]
    state.sibling_context = context_results[3]
    # 2026-09-02, id=1169 (950260.KS) — see PipelineState.atr_gate's own
    # docstring. Recorded here (not just derivable from volatility_context's
    # prose) because a batch audit needs a clean, structured denominator for
    # "how often is ATR missing and what did the gate do about it" without
    # re-parsing narrative text.
    _atr_available = not executability.atr_unavailable(current_atr_pct)
    if _atr_available:
        _atr_outcome = "ok"
    elif held:
        _atr_outcome = "held_unscaled"
    elif settings.auto_trade_require_atr_for_new_entry:
        _atr_outcome = "blocked_new_entry"
    else:
        _atr_outcome = "gate_disabled"
    state.atr_gate = {
        "atr_pct": current_atr_pct,
        "available": _atr_available,
        "held": held,
        "outcome": _atr_outcome,
    }
    if risk_cap_context is None:
        # The pre-debate computation above (right before risk_debators.build_all)
        # is the normal path; this is the fallback for when its guarded price
        # fetch failed or raised. `fill_price` here comes from the untouched
        # context_tasks gather above, same as it always did.
        risk_cap_context = await _risk_cap_context(ticker, fill_price, as_of=as_of)

    if as_of is None:
        portfolio_cap_data, portfolio_risk_context = context_results[4]
    elif portfolio_cap_override is not None:
        portfolio_cap_data = portfolio_cap_override
        snapshot, clusters, drawdown_frac = portfolio_cap_override
        portfolio_risk_context = _render_portfolio_risk_context(ticker, snapshot, clusters, drawdown_frac)
    else:
        portfolio_cap_data = DISABLED_PORTFOLIO_CAPS
        portfolio_risk_context = ""

    # LIVE ONLY, same restriction as portfolio_risk_context above — needs a
    # connected KISBroker, which a backtest never has.
    execution_status_context = await _execution_status_context(ticker, fill_price) if as_of is None else ""

    await _emit(on_stage, "Portfolio Manager finalizing the decision...")
    _consensus_note = _desk_consensus_note(
        state.research_plan, state.trader_proposal, kr_risk_extra, debate_sizing_context
    )
    state.final_decision = await portfolio_manager.decide(
        state.trader_proposal, state.risk_debate, state.past_context, position_context, volatility_context,
        risk_cap_context, portfolio_risk_context, execution_status_context,
        sibling_context=state.sibling_context,
        language=language, profile=resolved_profile, held=held, identity=identity_note,
        session_note=session_note, period_note=period_note,
        degradation_note=_degradation_note(state.degraded_analysts),
        desk_consensus_note=_consensus_note,
    )
    # Consensus-override gate FIRST (2026-09-04, id=1212/1217): if it downgrades
    # to Hold, is_state_flipping_rating(Hold) is False so _apply_hysteresis_gate
    # below is a no-op and the two bracketed prefixes never both land on
    # executive_summary. Order matters — see _apply_consensus_override_gate.
    _consensus_record: dict = {}
    state.final_decision = _apply_consensus_override_gate(
        state.final_decision, state.research_plan, state.trader_proposal, record_out=_consensus_record,
    )
    state.consensus_override = _consensus_record

    # Unjustified-exit gate (2026-09-09, P0-2/3, id=1358) — between the
    # consensus-override gate and the hysteresis gate: the three are mutually
    # exclusive by rating (consensus = Buy/Overweight, this = held Sell,
    # hysteresis = a flip that survived both), so at most one bracketed
    # prefix ever lands on executive_summary. Default OFF
    # (`settings.unjustified_exit_gate_enabled`); live only — a backtest has
    # no real position weight, drawdown, or execution status to check. Scans
    # the PM's own final text inline here (the step-3 instrumentation block
    # further down re-scans every source for the recorded
    # `sizing_cap_misuse_violations`; this needs only the PM spans, and needs
    # them before the downgrade). See _apply_unjustified_exit_gate.
    _unjustified_exit_record: dict = {}
    if as_of is None:
        _pm_misuse_spans = scan_sizing_cap_misuse(
            state.final_decision.executive_summary or "", agent_name="portfolio_manager", held=held
        ) + scan_sizing_cap_misuse(
            state.final_decision.investment_thesis or "", agent_name="portfolio_manager", held=held
        )
        _current_weight: float | None = None
        _drawdown_frac: float | None = None
        if isinstance(portfolio_cap_data, tuple) and len(portfolio_cap_data) == 3:
            _snapshot, _, _drawdown_frac = portfolio_cap_data
            if _snapshot.total_value > 0:
                _current_weight = _snapshot.per_ticker_value.get(ticker, 0.0) / _snapshot.total_value
        _last_trade = await get_last_trade(ticker)
        _forced_exit_on_book = bool(
            _last_trade is not None
            and (_last_trade.reason or "").startswith((STOP_LOSS_REASON_PREFIX, REVERSAL_REASON_PREFIX))
        )
        state.final_decision = _apply_unjustified_exit_gate(
            state.final_decision, state.research_plan, state.trader_proposal,
            held=held, atr_pct=current_atr_pct, current_weight=_current_weight,
            misuse_spans=_pm_misuse_spans, drawdown_frac=_drawdown_frac,
            execution_blocked="currently blocked" in (execution_status_context or ""),
            forced_exit_on_book=_forced_exit_on_book, record_out=_unjustified_exit_record,
        )
    state.unjustified_exit_gate = _unjustified_exit_record

    state.final_decision = _apply_hysteresis_gate(state.final_decision, held)
    # AFTER the hysteresis gate so both prefixes land on executive_summary in a
    # stable order and the _guard_fields numeric pass below sees the final text.
    state.final_decision = _apply_degradation_notice(state.final_decision, state.degraded_analysts)

    # The PM's user-visible text. executive_summary / investment_thesis feed
    # render_portfolio_decision -> decision_text -> app.audit and pgvector
    # memory, so an uncorrected 10x number becomes a retrievable precedent —
    # correct in place (banner=False: two-to-four sibling fields would each
    # repeat one banner). card_reason_ko/en are length-budgeted card text:
    # inline_note=False replaces the span with the bare corrected value, no
    # marker — an accounted-for silent correction (the violation is still
    # recorded with node attribution, and the full banner is on
    # executive_summary). 2026-09-01, id=1156/1157/1158.
    _pm_updates = _guard_fields(
        state.final_decision, ["executive_summary", "investment_thesis"], "portfolio_manager",
    )
    _pm_updates.update(_guard_fields(
        state.final_decision, ["card_reason_ko", "card_reason_en"], "portfolio_manager",
        inline_note=False,
    ))
    if _pm_updates:
        state.final_decision = state.final_decision.model_copy(update=_pm_updates)

    # If either is missing here, the dashboard signal card falls back to
    # executive_summary (2026-08-19 commit 3b96efa's fallback path) — the
    # exact screen that commit was meant to replace. That fallback runs
    # silently, so nothing else would ever surface "the purpose-built card
    # text didn't generate this cycle" without this. WARNING, not an error:
    # the decision itself is still valid, this only degrades the card.
    if is_blank_card_reason(state.final_decision.card_reason_ko) or is_blank_card_reason(
        state.final_decision.card_reason_en
    ):
        logger.warning(
            "%s: card_reason missing after portfolio_manager (ko=%s, en=%s) rating=%s held=%s "
            "backtest=%s — the dashboard signal card will fall back to executive_summary",
            ticker,
            not is_blank_card_reason(state.final_decision.card_reason_ko),
            not is_blank_card_reason(state.final_decision.card_reason_en),
            state.final_decision.rating.value,
            held,
            as_of is not None,
        )

    # Detect-only halt-status narrative check (2026-09-07, id=1238, 351320.KS):
    # scans every text source the PM saw or produced this cycle against the
    # execution_status_context computed above (scan_halt_claims itself is a
    # no-op unless that context is genuinely CONFIRMED-clear — see its own
    # docstring). Deliberately computed HERE, after the PM's decide() call and
    # its post-processing gates, not moved earlier: this guard's whole point
    # is to catch a false claim that made it all the way INTO the PM's own
    # final text, which doesn't exist until now. No control-flow change, no
    # LLM call, and NOT wired into any report-cache gating — it runs too late
    # in the pipeline for that to be meaningful anyway.
    _halt_violations: list[str] = []
    for _agent_name, _text in _sweep_texts(state, {"reports", "research_plan", "risk_debate", "pm"}):
        _halt_violations.extend(
            scan_halt_claims(_text, exec_status=execution_status_context, agent_name=_agent_name)
        )
    state.halt_status_violations = _halt_violations
    if state.halt_status_violations:
        logger.warning(
            "%s: halt_status_violations recorded — a present-tense halt/suspension claim "
            "contradicted a CONFIRMED-clear execution status this cycle: %s",
            ticker, state.halt_status_violations,
        )

    # Detect-only absence-reframed-as-positive-signal check (2026-09-09 P1
    # audit item ⑤, ids 1349/1350/1351/1352/1358/1359): scans every text
    # source the desk produced this cycle — same multi-node sweep as
    # scan_halt_claims above (report, research plan, risk debate, PM
    # decision) — for missing data (insider activity, investor flow,
    # analyst coverage) reframed as a specific positive/reassuring claim
    # rather than left as uncertainty. No control-flow change, no LLM call,
    # NOT cache-gating (same reasoning as halt_status_violations above).
    _absence_reframing_violations: list[str] = []
    for _agent_name, _text in _sweep_texts(state, {"reports", "research_plan", "risk_debate", "pm"}):
        _absence_reframing_violations.extend(
            scan_absence_reframing(_text, agent_name=_agent_name)
        )
    state.absence_reframing_violations = _absence_reframing_violations
    if state.absence_reframing_violations:
        logger.warning(
            "%s: absence_reframing_violations recorded — missing/absent data was reframed as a "
            "positive signal this cycle: %s",
            ticker, state.absence_reframing_violations,
        )

    # Detect-only derived-value arithmetic mismatch check (2026-09-09 P1
    # audit item ⑤, ids 1357/1353): scans every text source the desk
    # produced this cycle — same multi-node sweep as scan_absence_reframing
    # above — for a stated derived value (PER, stop-loss risk distance %)
    # that is inconsistent with the two ground values (EPS+price,
    # stop+price) cited alongside it. Numeric-derivation check, not a
    # phrasing pattern — see app.agents.derived_value_mismatch_guard's own
    # docstring. No control-flow change, no LLM call, NOT cache-gating.
    _derived_value_mismatch_violations: list[str] = []
    for _agent_name, _text in _sweep_texts(state, {"reports", "research_plan", "risk_debate", "pm"}):
        _derived_value_mismatch_violations.extend(
            scan_derived_value_mismatches(_text, agent_name=_agent_name)
        )
    state.derived_value_mismatch_violations = _derived_value_mismatch_violations
    if state.derived_value_mismatch_violations:
        logger.warning(
            "%s: derived_value_mismatch_violations recorded — a stated derived value did not "
            "match its own cited ground values this cycle: %s",
            ticker, state.derived_value_mismatch_violations,
        )

    # Detect-only intraday-close narrative check (2026-09-09, id=1358,
    # 047040.KS): scans the same text sources as the halt-status check above
    # for a closed-session/completed-candle assertion that contradicts
    # `session_note` (computed at the top of this function) when the session
    # was genuinely still open this cycle — see
    # app.agents.session_close_guard's module docstring. Also scans
    # research_debate (Bull/Bear read the same analyst reports as the risk
    # desk, same reasoning as the verified-price check below) in addition to
    # risk_debate, since id=1358's own false premise originated in an
    # analyst report and reached both debates.
    _session_close_violations: list[str] = []
    for _agent_name, _text in _sweep_texts(
        state, {"reports", "research_plan", "research_debate", "risk_debate", "pm"}
    ):
        _session_close_violations.extend(
            scan_session_close_claims(_text, session_note=session_note, agent_name=_agent_name)
        )
    state.session_close_violations = _session_close_violations
    if state.session_close_violations:
        logger.warning(
            "%s: session_close_violations recorded — a closed-session/completed-candle "
            "claim contradicted this cycle's still-open session: %s",
            ticker, state.session_close_violations,
        )

    # Detect-only false-desk-unanimity narrative check (2026-09-09, id=1352,
    # 028050.KS): `_consensus_note` above already told the PM, verbatim, what
    # the Research Manager and the Trader concluded — this scans the PM's own
    # final text for a claim that the desk was UNANIMOUS when those two enum
    # values (or the PM's own rating) actually disagree. See
    # app.agents.desk_consensus_narrative_guard's module docstring. `[]`
    # immediately (inside the guard) when either upstream object was missing,
    # matching `_desk_consensus_note`'s own `""` contract — only the PM's own
    # text is scanned, since the note itself is the ground truth, not a claim.
    _desk_narrative_violations: list[str] = []
    if state.research_plan is not None and state.trader_proposal is not None:
        _rm_rec = state.research_plan.recommendation.value
        _tr_act = state.trader_proposal.action.value
        _pm_rating = state.final_decision.rating.value
        for _source, _text in _sweep_texts(state, {"pm"}):
            _desk_narrative_violations.extend(
                scan_desk_narrative(_text, rm=_rm_rec, trader=_tr_act, pm_rating=_pm_rating, agent_name=_source)
            )
    state.desk_narrative_violations = _desk_narrative_violations
    if state.desk_narrative_violations:
        logger.warning(
            "%s: desk_narrative_violations recorded — the PM claimed desk unanimity that "
            "contradicts the actual RM/Trader values this cycle: %s",
            ticker, state.desk_narrative_violations,
        )

    # Detect-only sizing-cap-misuse check (2026-09-09, id=1358, 047040.KS):
    # the added-share sizing cap / ATR scaling is injected only into the
    # risk debate and the PM (via `risk_debators.build_all(sizing_context=)`
    # and `portfolio_manager.decide(risk_cap_context=)`), so only those text
    # sources are scanned — the analyst reports and the research debate never
    # see the cap wording and cannot misuse it. `held`-gated inside the guard
    # (a flat ticker has no position for the cap to be misused against), so
    # this is a no-op for most cycles. Step 3 of the P0-2/3 plan — the
    # counter a later retrospective calibration reads before the
    # `unjustified_exit_gate` (default OFF) is enabled. See
    # app.agents.sizing_cap_misuse_guard's module docstring.
    _sizing_cap_misuse_violations: list[str] = []
    for _agent_name, _text in _sweep_texts(state, {"research_plan", "risk_debate", "pm"}):
        _sizing_cap_misuse_violations.extend(
            scan_sizing_cap_misuse(_text, agent_name=_agent_name, held=held)
        )
    state.sizing_cap_misuse_violations = _sizing_cap_misuse_violations
    if state.sizing_cap_misuse_violations:
        logger.warning(
            "%s: sizing_cap_misuse_violations recorded — the added-share sizing cap / volatility "
            "scaling was cited as a reason to exit or cut a held position this cycle: %s",
            ticker, state.sizing_cap_misuse_violations,
        )

    # Detect-only same-session price-limit narrative check (2026-09-08,
    # id=1313, 082850.KS): scans the same text sources as the halt-status
    # check above for a claimed same-session/single-day move exceeding
    # KRX's ±30% daily limit — see app.agents.price_limit_guard's module
    # docstring for the calibration behind this. KRX-only (the ±30% band
    # doesn't exist for a US-listed ticker) and, like the halt-status check,
    # computed here (after the PM's own decide() call) so a claim that made
    # it all the way into the PM's final text is caught too.
    _price_limit_violations: list[str] = []
    if ticker.endswith(".KS"):
        for _agent_name, _text in _sweep_texts(state, {"reports", "research_plan", "risk_debate", "pm"}):
            _price_limit_violations.extend(
                scan_price_limit_claims(_text, agent_name=_agent_name)
            )
    state.price_limit_violations = _price_limit_violations
    if state.price_limit_violations:
        logger.warning(
            "%s: price_limit_violations recorded — a claimed same-session price move "
            "exceeded KRX's ±30%% daily limit this cycle: %s",
            ticker, state.price_limit_violations,
        )

    # Detect-only self-labeled "Verified" price check (2026-09-08, id=1314,
    # 033790.KS): scans the same text sources as the two checks above for a
    # "Verified Close/Price/Snapshot" claim that contradicts the real
    # verified_snapshot fetched for this cycle — see
    # app.agents.verified_price_guard's module docstring. Universal (not
    # KRX-only, unlike price_limit_guard above) — verified_snapshot is
    # fetched for both markets.
    _verified_price_violations: list[str] = []
    # research_debate (Bull/Bear) is scanned here too — unlike
    # halt_status_guard/price_limit_guard above (which only scan risk_debate,
    # matching their own narrower originating incidents), a fabricated
    # "Verified" price from the News + Macro Analyst is exactly as likely to
    # reach the research desk (Bull/Bear read the same analyst reports) as
    # the risk desk, so this guard's own scan covers both.
    for _agent_name, _text in _sweep_texts(
        state, {"reports", "research_plan", "risk_debate", "research_debate", "pm"}
    ):
        _verified_price_violations.extend(
            scan_verified_price_claims(_text, verified_snapshot=verified_snapshot, agent_name=_agent_name)
        )
    state.verified_price_violations = _verified_price_violations
    if state.verified_price_violations:
        logger.warning(
            "%s: verified_price_violations recorded — a self-labeled 'Verified' price claim "
            "contradicted the real verified snapshot this cycle: %s",
            ticker, state.verified_price_violations,
        )

    # Detect-only automatic-regulatory-consequence check (id=1380): scans
    # every text source the desk saw or produced this cycle — full coverage
    # (reports, research plan, both debates, the trader proposal, and the
    # PM's own final text), unlike the narrower legacy sweeps above, since
    # this is a fresh guard with no pre-existing coverage gap to preserve —
    # for a fabricated causal mechanism ("X triggers automated compliance
    # protocols: trading halts, index exclusion, forced liquidation ...")
    # not grounded in any data actually supplied this cycle. See
    # app.agents.regulatory_mechanism_guard's module docstring for why this
    # is a sibling of halt_status_guard, not an extension of it. No control-
    # flow change, no LLM call, NOT cache-gating.
    _supplied_fields: set[str] = set()
    if execution_status_context:
        _supplied_fields.add("execution_status_context")
    if kr_risk_extra:
        _supplied_fields.add("kr_risk_extra")
    if verified_snapshot:
        _supplied_fields.add("verified_snapshot")
    if kr_gt:
        _supplied_fields.add("kr_gt")
    if kr_news_gt:
        _supplied_fields.add("kr_news_gt")
    if kr_financials_recency:
        _supplied_fields.add("kr_financials_recency")
    _supplied_fields_frozen = frozenset(_supplied_fields)

    _regulatory_mechanism_violations: list[str] = []
    for _agent_name, _text in _sweep_texts(
        state, {"reports", "research_plan", "research_debate", "risk_debate", "trader_proposal", "pm"}
    ):
        _regulatory_mechanism_violations.extend(
            scan_regulatory_mechanism(_text, supplied_fields=_supplied_fields_frozen, agent_name=_agent_name)
        )
    state.regulatory_mechanism_violations = _regulatory_mechanism_violations
    if state.regulatory_mechanism_violations:
        logger.warning(
            "%s: regulatory_mechanism_violations recorded — an automatic regulatory/exchange "
            "consequence was asserted without any data supplied this cycle to ground it: %s",
            ticker, state.regulatory_mechanism_violations,
        )

    # Detect-only ground-truth-claim check (2026-09-11, id=1386/id=1394): scans
    # every text source the desk saw or produced this cycle — full coverage,
    # same reasoning as regulatory_mechanism_guard above (a fresh guard, no
    # legacy coverage gap to preserve) — for a PER/PBR/EPS/BPS or
    # program-trading figure attributed to the pipeline's own supplied
    # ground truth (kr_gt / kr_risk_extra), or sitting inside a
    # correction/accusation frame, whose value materially disagrees with
    # that ground truth. See app.agents.ground_truth_claim_guard's module
    # docstring for the id=1386 shape (a debater falsely accused a CORRECT
    # figure of being a "fabrication" and replaced it with a WRONG one) and
    # for the deliberately simple v1 scope this module chose. No control-
    # flow change, no LLM call, NOT cache-gating.
    _ground_truth = extract_ground_truth(kr_gt=kr_gt, kr_risk_extra=kr_risk_extra)
    _ground_truth_claim_violations: list[str] = []
    for _agent_name, _text in _sweep_texts(
        state, {"reports", "research_plan", "research_debate", "risk_debate", "trader_proposal", "pm"}
    ):
        _ground_truth_claim_violations.extend(
            scan_ground_truth_claims(_text, ground_truth=_ground_truth, agent_name=_agent_name)
        )
    state.ground_truth_claim_violations = _ground_truth_claim_violations
    if state.ground_truth_claim_violations:
        logger.warning(
            "%s: ground_truth_claim_violations recorded — a numeric claim materially "
            "disagreed with this cycle's supplied ground truth: %s",
            ticker, state.ground_truth_claim_violations,
        )

    # Shared PM-narrative-contradiction notice (2026-09-11): every sweep guard
    # above is detect-only — a PM-attributed finding (session_close, regulatory_
    # mechanism, ground_truth_claim, desk_narrative, sizing_cap_misuse,
    # price_limit, verified_price, absence_reframing, derived_value_mismatch,
    # halt_status — the real, current list of sweeps whose source set includes
    # "pm", verified against _sweep_texts call sites above) is logged and
    # persisted into the transcript, but the PM's own investment_thesis never
    # reflects that its own text was flagged. This appends (never prepends —
    # the prefix region belongs to the four downgrade gates above:
    # _apply_consensus_override_gate / _apply_unjustified_exit_gate /
    # _apply_hysteresis_gate / _apply_degradation_notice, which all APPEND
    # their own note to investment_thesis too, so this notice simply lands
    # after theirs when one of them also fired this cycle — no collision,
    # since neither this step nor those gates ever touch executive_summary
    # more than once) a short, language-appropriate, digit/currency-token-
    # free notice to investment_thesis whenever ANY sweep flagged the PM's
    # own text. No LLM call, always-on (no settings gate needed — this never
    # touches rating/stop_loss/confidence/price_target, only appends to
    # investment_thesis). `state.final_decision` is always set by this point
    # in the function (the PM decide() call and all four gates above already
    # ran); guarded anyway since this runs unconditionally after them.
    _pm_attributed_violations = [
        v for violations_list in (
            state.session_close_violations, state.regulatory_mechanism_violations,
            state.ground_truth_claim_violations, state.desk_narrative_violations,
            state.sizing_cap_misuse_violations, state.price_limit_violations,
            state.verified_price_violations, state.absence_reframing_violations,
            state.derived_value_mismatch_violations, state.halt_status_violations,
        )
        for v in violations_list if v.startswith("portfolio_manager:")
    ]
    state.pm_narrative_contradiction_notice_applied = bool(_pm_attributed_violations)
    if _pm_attributed_violations and state.final_decision is not None:
        logger.warning(
            "%s: PM's own investment_thesis was flagged by %d sweep guard finding(s) this cycle "
            "— appending a contradiction notice: %s",
            ticker, len(_pm_attributed_violations), _pm_attributed_violations,
        )
        _pm_notice = _PM_CONTRADICTION_NOTICE.get(language, _PM_CONTRADICTION_NOTICE["en"])
        state.final_decision = state.final_decision.model_copy(
            update={
                "investment_thesis": (state.final_decision.investment_thesis or "") + _pm_notice,
            }
        )

    state.reasoning_sanitizations = list(llm_metrics.current_run_sanitizations() or [])
    state.injection_flags = list(llm_metrics.current_run_injection_flags() or [])

    # 2026-09-07 audit #3: the per-decision roll-up of the news-headline 억
    # amount check across every node's unit_scale scan. `flagged_nodes` /
    # `total_flags` being 0 with `headline_amounts` > 0 is the measurement
    # this ships to establish — how often the fix is even exercised — before
    # the news_eok_10x band is ever tightened toward auto-correction.
    _news_flags = {
        n: s.get("news_eok_flagged", 0)
        for n, s in state.unit_scale_scans.items()
        if s.get("news_eok_flagged", 0)
    }
    state.news_amount_scans = (
        {
            "headline_amounts": len(extract_kr_headline_eok_values(kr_news_gt)),
            "flagged_nodes": _news_flags,
            "total_flags": sum(_news_flags.values()),
        }
        if kr_news_gt
        else {}
    )

    transcript = {
        "reports": dict(state.reports),
        "research_debate": [
            {"speaker": t.speaker, "content": t.content, "round": t.round} for t in state.research_debate.turns
        ],
        "research_plan": render_research_plan(state.research_plan) if state.research_plan else None,
        "trader_proposal": render_trader_proposal(state.trader_proposal) if state.trader_proposal else None,
        "risk_debate": [
            {"speaker": t.speaker, "content": t.content, "round": t.round} for t in state.risk_debate.turns
        ],
        # Dropped from render_portfolio_decision's short default answer (it's
        # unbounded and often long) — kept here so the "view details" toggle
        # still surfaces it.
        "investment_thesis": state.final_decision.investment_thesis,
        # Empty in the common case — see app.agents.identity_gate and
        # PipelineState.identity_violations's own docstrings. Recorded here
        # (not just logged) so a later batch audit over stored decisions
        # doesn't have to re-derive violations from raw report text by hand.
        "identity_violations": state.identity_violations,
        # Reports that never stated a company name (a neutral note, not a
        # violation — see PipelineState.identity_unnamed). Kept distinct from
        # identity_violations so an audit isn't misled into treating a
        # legitimately name-free pure-macro report as a hallucination.
        "identity_unnamed": state.identity_unnamed,
        # Empty in the common case — see app.agents.unit_scale_guard and
        # PipelineState.unit_scale_violations's own docstrings. Same
        # recorded-not-just-logged reasoning as identity_violations above.
        "unit_scale_violations": state.unit_scale_violations,
        # Detect-only — a report % that looks like a Korean interim/full-year
        # mis-division framed as a decline (see period_mismatch_guard /
        # PipelineState.period_mismatch_violations, id=1158). Empty is NOT
        # proof the report is clean.
        "period_mismatch_violations": state.period_mismatch_violations,
        # Positive per-node scan record for the above — a node with
        # forbidden_in_band==0 was UNSCANNABLE, not necessarily clean
        # (period_mismatch_guard / PipelineState.period_mismatch_scans).
        "period_mismatch_scans": state.period_mismatch_scans,
        # How stale get_kr_financials' newest period was (audit #6, id=1237):
        # {latest_period, latest_label, months_stale, interim}. `{}` for a US
        # ticker or a cache hit. `_staleness_note` warned downstream when
        # months_stale >= settings.kr_financials_staleness_warn_months.
        "kr_financials_recency": state.kr_financials_recency,
        # Per-decision roll-up of the news-headline 억-amount 10x check
        # (audit #3, id=1237): {headline_amounts, flagged_nodes, total_flags}.
        # `{}` for a US ticker or a cache hit.
        "news_amount_scans": state.news_amount_scans,
        # Positive scan record per node (see PipelineState.unit_scale_scans).
        # A node with ground_truth_values>0, candidates_examined>0,
        # corrected==0 and unverified==candidates_examined is the id=1158
        # blind-spot shape — an empty unit_scale_violations there is NOT a
        # clean report.
        "unit_scale_scan": state.unit_scale_scans,
        # Raw get_kr_risk_signals() text fed into the risk debate this cycle
        # (short-selling intensity, program-trade net-buy) — see
        # PipelineState.kr_risk_signals's own docstring. "" for a non-.KS
        # ticker or a cache hit.
        "kr_risk_signals": state.kr_risk_signals,
        # Empty for a held ticker and in the clean case — see
        # PipelineState.risk_rating_violations / risk_debators.check_illegal_ratings.
        "risk_rating_violations": state.risk_rating_violations,
        # `{}` when there was no comparable Neutral round-1/final pair. When
        # populated: {r1, final, final_round, reversed, new_fact_cited,
        # mechanics_terms, r1_span, final_span} — see
        # risk_debators.check_neutral_reversal (2026-09-04, id=1212/1217).
        # `reversed: True, new_fact_cited: False` is the flip this exists to
        # count; a null `r1`/`final` means "did not parse", not "unchanged".
        "neutral_reversal": state.neutral_reversal,
        # "AgentName: '<span>' vs CONFIRMED-clear broker status" per present-
        # tense halt/suspension claim that contradicted a genuinely confirmed-
        # clear execution_status_context this cycle — see
        # app.agents.halt_status_guard / PipelineState.halt_status_violations
        # (2026-09-07, id=1238, 351320.KS). Empty in the overwhelming common
        # case (no ground truth to scan against, or nothing found).
        "halt_status_violations": state.halt_status_violations,
        # PipelineState.absence_reframing_violations (2026-09-09, P1 item ⑤) —
        # missing data (insider activity/investor flow/analyst coverage)
        # reframed as a positive signal rather than left as uncertainty.
        "absence_reframing_violations": state.absence_reframing_violations,
        # PipelineState.derived_value_mismatch_violations (2026-09-09, P1
        # item ⑤) — a stated derived value (PER, stop-loss risk distance %)
        # inconsistent with its own nearby ground-value citations.
        "derived_value_mismatch_violations": state.derived_value_mismatch_violations,
        # "AgentName: [<family>] '<span>' vs SESSION-IN-PROGRESS ground truth"
        # per closed-session/completed-candle claim that contradicted a
        # genuinely still-open session this cycle — see
        # app.agents.session_close_guard / PipelineState.session_close_violations
        # (2026-09-09, id=1358, 047040.KS). Empty in the common case (session
        # already closed, backtest, or nothing found).
        "session_close_violations": state.session_close_violations,
        # "AgentName: claimed desk unanimity ... vs RM=<rm>/Trader=<trader>/
        # PM=<rating> — '<span>'" per false desk-unanimity claim in the PM's
        # own final text — see app.agents.desk_consensus_narrative_guard /
        # PipelineState.desk_narrative_violations (2026-09-09, id=1352,
        # 028050.KS). Empty when either upstream object was missing or
        # nothing found.
        "desk_narrative_violations": state.desk_narrative_violations,
        # "AgentName: sizing cap cited as exit rationale — '<span>'" per
        # risk-debate/PM sentence drawing an exit-or-reduce conclusion from
        # the added-share sizing cap for a held position — see
        # app.agents.sizing_cap_misuse_guard / PipelineState.sizing_cap_misuse_violations
        # (2026-09-09, id=1358, 047040.KS). Empty for a flat ticker (`held`
        # False) or when nothing found.
        "sizing_cap_misuse_violations": state.sizing_cap_misuse_violations,
        # "AgentName: claimed <N>% same-session move exceeds KRX's ±30% daily
        # limit ..." per claimed price move exceeding the KRX daily limit —
        # see app.agents.price_limit_guard / PipelineState.price_limit_violations
        # (2026-09-08, id=1313, 082850.KS). Only ever populated for a
        # KRX-listed (`.KS`) ticker.
        "price_limit_violations": state.price_limit_violations,
        # "AgentName: claimed 'Verified' price <N> vs real verified snapshot
        # <M> ..." per self-labeled Verified price claim contradicting the
        # real verified snapshot this cycle — see app.agents.
        # verified_price_guard / PipelineState.verified_price_violations
        # (2026-09-08, id=1314, 033790.KS).
        "verified_price_violations": state.verified_price_violations,
        # "AgentName: '<span>' — ungrounded automatic <consequence> claim"
        # per fabricated automatic regulatory/exchange consequence (forced
        # liquidation, delisting, index exclusion, trading halt/suspension,
        # margin/credit-trading ban, institutional-mandate divestment) not
        # grounded in any data supplied this cycle — see app.agents.
        # regulatory_mechanism_guard / PipelineState.regulatory_mechanism_violations
        # (id=1380). Detect-only, full-coverage sweep — see the guard's own
        # module docstring for why it is a sibling of halt_status_guard,
        # not an extension of it.
        "regulatory_mechanism_violations": state.regulatory_mechanism_violations,
        # "AgentName: <label> claimed as <N> vs supplied ground truth <M>
        # [— false correction] — '<span>'" per PER/PBR/program-trading figure
        # attributed to (or "correcting") the pipeline's own supplied ground
        # truth (kr_gt / kr_risk_extra) that materially disagrees with it —
        # see app.agents.ground_truth_claim_guard /
        # PipelineState.ground_truth_claim_violations (id=1386/id=1394,
        # 2026-09-11). Detect-only, full-coverage sweep, same shape as
        # regulatory_mechanism_violations above.
        "ground_truth_claim_violations": state.ground_truth_claim_violations,
        # Whether the shared PM-narrative-contradiction notice was appended to
        # final_decision.investment_thesis this cycle (True) vs left untouched
        # (False) — see PipelineState.pm_narrative_contradiction_notice_applied's
        # own docstring (2026-09-11).
        "pm_narrative_contradiction_notice_applied": state.pm_narrative_contradiction_notice_applied,
        # "AgentName: ..." per hedged/ungrounded technical-indicator claim or
        # index-contribution misattribution found in the Market Analyst's own
        # report this cycle — see app.agents.technical_claim_guard /
        # PipelineState.technical_claim_violations (2026-09-07). Empty in the
        # common case; NOT proof of a clean report on its own — cross-check
        # against indicator_block_available below.
        "technical_claim_violations": state.technical_claim_violations,
        # "AgentName: refusal_total|refusal_hedged — '<span>'" per "as an AI
        # model I have no real-time data / that date is past my training
        # cut-off" refusal preamble written by one of the four analysts
        # instead of analysis — see app.agents.refusal_guard /
        # PipelineState.refusal_violations (2026-09-14, measured 21/1008 for
        # the Fundamentals Analyst plus 1 Market and 1 live News + Macro).
        # Analysts only, detect-only; the tier is a label, nothing branches
        # on it.
        "refusal_violations": state.refusal_violations,
        # PipelineState.staleness_narrative_violations (2026-09-09, id=1351) —
        # stale exchange-reported financials presented as the company's
        # current condition with no staleness acknowledgment anywhere.
        "staleness_narrative_violations": state.staleness_narrative_violations,
        # "AgentName: '<span>' — business/industry description inconsistent
        # with registered name '<company_name>'" per fabricated business/
        # industry self-description found after identity_gate already
        # flagged a report for a wrong company name this cycle — see
        # app.agents.business_description_guard /
        # PipelineState.business_description_violations (2026-09-07,
        # id=1237, 294630.KS/서남). Empty in the common case.
        "business_description_violations": state.business_description_violations,
        # Whether this cycle's Market Analyst indicator block carried a real
        # get_indicators value (True) vs the NOT AVAILABLE fallback or a
        # cached-report cycle (False) — see
        # PipelineState.indicator_block_available's own docstring.
        "indicator_block_available": state.indicator_block_available,
        # Whether this cycle's Market Analyst investor-flow block carried a
        # real get_investor_flow value (True) vs the NOT AVAILABLE fallback
        # or a cached-report cycle (False) — see
        # PipelineState.flow_block_available's own docstring (2026-09-11).
        "flow_block_available": state.flow_block_available,
        # Whether this cycle's Market Analyst long-horizon block carried a
        # real get_long_horizon_context value (True) vs the NOT AVAILABLE
        # fallback or a cached-report cycle (False) — see
        # PipelineState.long_horizon_block_available (2026-09-16).
        "long_horizon_block_available": state.long_horizon_block_available,
        # `{}` when the desk-consensus override gate did not apply (the common
        # case); otherwise {rm, trader, pm_rating, confidence, tier, threshold,
        # gated} — `gated: False` records an override that CLEARED the higher
        # confidence bar, `gated: True` one that was downgraded to Hold. See
        # pipeline._apply_consensus_override_gate (2026-09-04, id=1212/1217).
        "consensus_override": state.consensus_override,
        # `{}` when the unjustified-exit gate did not apply (not a held Sell,
        # flag off, backtest, or a hard exception); otherwise {tier, rm,
        # trader, current_weight, underweight_target, misuse_spans, gated} —
        # `gated: True` downgraded the Sell to Hold, `False` recorded a tier
        # that did not reach the firing bar. See
        # pipeline._apply_unjustified_exit_gate (2026-09-09, P0-2/3, id=1358).
        "unjustified_exit_gate": state.unjustified_exit_gate,
        # Structured denominator for the ATR fail-closed gate — see
        # PipelineState.atr_gate's own docstring (2026-09-02, id=1169).
        "atr_gate": state.atr_gate,
        # Two families, namespaced by entry prefix — see
        # PipelineState.reasoning_sanitizations / app.llm.metrics.
        # track_run_sanitizations for the full shape list. Reasoning-leak
        # entries look like `risk_conservative: think_block` (2026-09-01,
        # id=1156; `untagged_suspected`/`declined_too_short` are NOT strips).
        # Injection-banner entries look like
        # `risk_neutral: injection_banner:rest_self_reversal` (2026-09-01,
        # root cause #9, id=1157/id=1158 — see
        # app.agents.untrusted_content.redact_injection_flag).
        "reasoning_sanitizations": state.reasoning_sanitizations,
        # `{"agent","tier","reason"}` per INJECTION_SUSPECTED report — the
        # model's own one-sentence reason, which used to be log-only. Pairs
        # 1:1 with an `injection_banner:<tier>` entry above. See
        # PipelineState.injection_flags.
        "injection_flags": state.injection_flags,
        # True if the ticker's market was open when this decision was made —
        # see PipelineState.session_in_progress (2026-09-01, id=1157).
        "session_in_progress": state.session_in_progress,
        # Empty in the common case and on any cache hit. The COUNT also
        # goes to its own memory_entries column (`analysts_failed`,
        # app.memory.db) because a win-rate query must be able to filter on
        # it in SQL without digging through jsonb; the NAMES live here
        # because only a human reading one decision needs them.
        "degraded_analysts": state.degraded_analysts,
        # Everything below is context the Portfolio Manager ACTUALLY READ
        # (`portfolio_manager.decide`'s own positional/keyword args, right
        # below) but that nothing durable recorded until 2026-08-25 — the
        # dashboard's "how did the agents get here" view and every later
        # audit could see the debates and the reports, then had to guess at
        # the pieces of machine-generated context that most often explain a
        # surprising rating (a drawdown breaker, a VI halt, a same-day
        # sibling, a forced stop-loss exit). Stored VERBATIM, not
        # summarized: a trimmed copy would look complete while being unable
        # to answer "what exactly did it see", which is the whole reason
        # this dict exists. Sizes are small next to `reports` (four
        # multi-KB analyst reports) — the largest of these, `past_context`,
        # is bounded by settings.memory_n_same + memory_n_cross retrieved
        # rows. `held`/`decision_trigger`/`confidence` are deliberately NOT
        # here — they're first-class memory_entries COLUMNS (app.memory.db),
        # and duplicating a column into the blob invites the two disagreeing.
        "past_context": state.past_context,
        "sibling_context": state.sibling_context,
        # The RM-concluded / Trader-proposed summary block shown to the PM
        # above the trader's proposal (2026-09-04, id=1212/1217) — "" when
        # either upstream object was missing. See _desk_consensus_note.
        "desk_consensus_note": _consensus_note,
        "position_context": position_context,
        "volatility_context": volatility_context,
        "risk_cap_context": risk_cap_context,
        # Live-only strings (empty during a backtest, by design — see their
        # own docstrings) stored as-is, "" and all: that IS what the PM
        # received, not an unknown to be distinguished from a real empty.
        "portfolio_risk_context": portfolio_risk_context,
        "execution_status_context": execution_status_context,
        # Not a `decide` kwarg by this name — the verified ticker->name
        # anchor (`identity_note` above) is threaded into every judgment
        # node, and `identity_violations` above only records anchor
        # FAILURES. Recording the resolved name is what lets a later reader
        # tell "the gate passed" from "the gate never had the right name to
        # begin with".
        "company_name": company_name,
    }
    def _call_store_pending() -> Awaitable[int | None]:
        return store_pending(
            ticker,
            trade_date,
            situation_text,
            state.final_decision.rating.value,
            render_portfolio_decision(state.final_decision),
            transcript=transcript,
            # `as_of is not None` alone missed one real case (2026-09-15): an
            # experiment harness that deliberately runs the LIVE call shape
            # (no `as_of`, because the guard under test only fires while
            # `session_in_progress`) inside `backtest_scope()`. Its ledger
            # writes were isolated, but every memory row it produced landed
            # tagged live — 36 of them in one afternoon, which
            # `get_similar_context` then serves to real decisions as prior
            # experience (live retrieval excludes `is_backtest` rows, so a
            # mis-tagged row is included, not skipped). `backtest_scope()` is
            # the caller's explicit "this is a replay, not a decision"
            # statement; honour it here too. No effect on real backtests —
            # both replay entrypoints already pass `as_of`.
            is_backtest=as_of is not None or is_backtest_context(),
            # The Trader's own Buy/Hold/Sell call BEFORE the risk debate and
            # Portfolio Manager ran — see app.memory.db's trader_action column
            # comment for why this is recorded separately from the final rating.
            trader_action=state.trader_proposal.action.value if state.trader_proposal else None,
            # Stage 1 (2026-08-13) — the value that actually scaled real capital
            # via app.execution.sizing.confidence_size_modifier. Read from
            # state.final_decision (post-hysteresis-gate) rather than an earlier
            # object: _apply_hysteresis_gate's model_copy preserves confidence
            # unchanged, so the value is identical either way, but this keeps it
            # read from the same object whose `rating` is also being stored,
            # rather than two different snapshots that happen to agree today.
            confidence=state.final_decision.confidence,
            # Stage 1 (2026-08-13) — position state immediately BEFORE this
            # decision (already computed above, before the LLM ran) — the only
            # way to later tell a Hold-while-holding from a Hold-while-flat
            # apart. See app.memory.db's held_at_decision column comment.
            held_at_decision=held,
            # Dashboard signal-card "why" text in each UI language — see
            # app.state's _PortfolioDecisionFields comment. Read from
            # state.final_decision (post-hysteresis-gate); that gate's
            # model_copy only overrides rating/executive_summary/
            # investment_thesis, so these two fields pass through unchanged.
            card_reason_ko=state.final_decision.card_reason_ko,
            card_reason_en=state.final_decision.card_reason_en,
            # Total LLM token usage for this one ticker/decision — every
            # record() call made since run_pipeline's track_run_tokens() scope
            # opened (4 analysts + both debates + judgment nodes), read here
            # rather than after this function returns since nothing after this
            # point makes another LLM call. None outside a track_run_tokens()
            # scope (shouldn't happen via run_pipeline, but _run_pipeline_impl
            # itself has no such guarantee if ever called directly).
            total_tokens=llm_metrics.current_run_tokens(),
            decision_trigger=decision_trigger,
            # 0 = full four-analyst run; 1-3 = degraded (see
            # state.degraded_analysts and app.memory.db's own column
            # comment). Read from `state` rather than a local so the
            # cached-reports branch (necessarily clean -> 0) and the
            # fresh-run branch go through the same field.
            analysts_failed=len(state.degraded_analysts),
        )

    # Unguarded until 2026-08-25 (see id=377/id=949 above and in
    # rebalance_to_target's retry block below) — a spurious CancelledError
    # here used to unwind past store_pending's own except-Exception handler
    # (CancelledError subclasses BaseException) with no log, no retry, and
    # no decision recorded, then the pipeline went on to place a real order
    # anyway with nothing in memory_entries to show for it. `store_pending`'s
    # own INSERT is a single statement, so on an unrequested cancellation the
    # write may already have committed before the exception surfaced here —
    # `find_recent_pending_id` checks for that BEFORE retrying, so a
    # cancellation can't turn one decision into two rows.
    try:
        state.decision_id = await _call_store_pending()
    except asyncio.CancelledError:
        if task_was_actually_cancelled():
            raise
        logger.warning(
            "%s: store_pending hit an unrequested CancelledError — checking whether "
            "the insert already committed before retrying (see id=377/id=949)",
            ticker,
        )
        state.decision_id = await find_recent_pending_id(ticker, trade_date, situation_text)
        if state.decision_id is None:
            try:
                state.decision_id = await _call_store_pending()
            except asyncio.CancelledError:
                if task_was_actually_cancelled():
                    raise
                logger.error(
                    "%s: store_pending retry also hit an unrequested CancelledError — "
                    "proceeding fail-open without a recorded decision, same policy as "
                    "an ordinary store_pending DB failure (app.memory.store's own "
                    "degrade-quietly convention)",
                    ticker,
                )
                state.decision_id = None

    # Fail-open, not fail-closed: a missing decision record must never block
    # a real trade (same policy `store_pending`'s own `except Exception`
    # already applies to an ordinary DB failure — this just extends it to
    # the CancelledError path above and makes the gap visible on the trade
    # itself instead of only in a log line that may have rotated away by
    # the time anyone goes looking for why this trade has no matching
    # decision in memory_entries).
    decision_reason = render_portfolio_decision(state.final_decision)
    if state.decision_id is None:
        logger.warning(
            "%s: no decision record for this trade (store_pending failed or was "
            "cancelled) — proceeding fail-open; marking trades.reason for reconciliation",
            ticker,
        )
        decision_reason = f"[NO DECISION RECORD] {decision_reason}"

    # Auto-add to the trade allowlist on an actual BUY decision for a
    # not-yet-allowed ticker — LIVE ONLY (as_of is None; a backtest is a
    # historical replay and must never touch settings.trade_allowlist_domestic/
    # _overseas). Fixes the gap where a screened (not-yet-allowed) candidate
    # could run the full pipeline and get a BUY decision only to have
    # kis_bridge.py's trade-allowlist gate unconditionally block the order —
    # deliberately only on BUY (not every screened candidate) so a
    # Hold-decided candidate doesn't bloat the allowlist. Goes straight onto
    # the trade allowlist (add_ticker_to_trade_allowlist also folds it into
    # the watchlist — see that function — never routed through a watchlist-only
    # first step; a BUY decision is itself the trading-permission signal, no
    # separate watchlist step needed here). Best-effort: a failure here must
    # not abort the decision that's already been made.
    #
    # Also marks the new entry `trade_allowlist_auto_added` (2026-07-29) —
    # this add is speculative at the moment it happens (SEMI_AUTO hasn't
    # even created the pending-approval order yet, let alone had it
    # approved and filled), so app.web.state.reconcile_trade_allowlist can
    # later prune it back off if the order never turns into a real
    # position (rejected, expired, or failed) — without this marker a
    # rejected BUY would otherwise leave the ticker on the allowlist
    # forever. Only marks on an actual "added" status: if the ticker
    # somehow already existed (a race with another caller between the
    # `not in` check above and this call), it's a pre-existing entry with
    # its own provenance, not this decision's to claim.
    # 실제 매수 결정이 나왔을 때만 매매허용목록에 자동 추가 — 백테스트는 대상 아님.
    # 관심종목을 거치지 않고 바로 매매허용목록에 들어감(BUY 결정 자체가 신호).
    # 이때 auto_added로 표시해둬야 나중에 주문이 거절/만료/실패해도 화해 로직이 정리 가능.
    if as_of is None and state.final_decision.rating is PortfolioRating.BUY and ticker not in settings.trade_allowlist:
        from app.web import state as web_state  # local import — avoid shadowing the `state` PipelineState above

        try:
            add_status, _ = await web_state.add_ticker_to_trade_allowlist(ticker)
            if add_status == "added":
                await web_state.mark_trade_allowlist_auto_added(ticker)
        except (
            web_state.LeveragedTickerError,
            web_state.UnsupportedProductTypeError,
            web_state.CashOrBondFundError,
            web_state.DerivativeIncomeFundError,
            web_state.UnresolvedTickerNameError,
        ) as exc:
            # A BUY decision blocked from the trade allowlist used to be
            # visible only as a single logger.warning line below — invisible
            # unless someone was tailing logs at the exact moment. 2026-09-03
            # investigation (identity_gate #6 follow-up) found this gate would
            # have blocked 36% of that day's screened candidates had they been
            # BUY-decided (mostly yfinance quoteType misclassification — see
            # app.dataflows.kis_provider._KRX_SECURITY_GROUP_TO_QUOTE_TYPE for
            # the root-cause fix), discoverable only by re-deriving it from
            # yfinance calls after the fact. Recording it as an alert instead
            # means the next occurrence is counted, not re-derived.
            logger.warning("BUY decision for %s could not reach the trade allowlist: %s", ticker, exc)
            from app.clock import get_clock
            from app.live.alerts import Alert, record_alert

            await record_alert(Alert(
                ticker=ticker,
                kind="trade_allowlist_blocked",
                message=f"BUY decision for {ticker} blocked from the trade allowlist ({type(exc).__name__}): {exc}",
                triggered_at=get_clock().now(),
            ))
        except Exception:
            logger.warning("could not auto-add %s to the trade allowlist after a BUY decision", ticker, exc_info=True)

    await _emit(on_stage, "Recording the trade...")
    try:
        rebalance_result = await rebalance_to_target(
            ticker,
            trade_date,
            fill_price,
            target_allocation_for(state.final_decision.rating, current_atr_pct, confidence=state.final_decision.confidence),
            reason=decision_reason,
            stop_loss=state.final_decision.stop_loss,
            current_atr_pct=current_atr_pct,
            # LIVE (as_of is None): deliberately NOT `portfolio_cap_data` here —
            # that snapshot was fetched at the *start* of this pipeline run,
            # before ~9 LLM calls' worth of wall-clock time, and reusing it for
            # enforcement (not just the narrative above) would let a concurrently-
            # analyzed ticker in the same market commit against a stale balance
            # (see app.execution.portfolio_risk.market_lock's docstring). Passing
            # None makes rebalance_to_target take its own live branch instead
            # (a fresh real_account_snapshot fetch under that market's lock,
            # right before this trade actually commits).
            # BACKTEST (as_of is not None): unchanged — reuses the harness's own
            # precomputed local-state override (or DISABLED_PORTFOLIO_CAPS for a
            # single-ticker backtest), since there's no real account to race on
            # and the live-only HWM table/lock must never be touched here.
            portfolio_cap_override=(portfolio_cap_data if as_of is not None else None),
            decision_id=state.decision_id,
            as_of=as_of,
            rating=state.final_decision.rating,
        )
    except asyncio.CancelledError:
        # UNLIKE store_pending above: never retried. rebalance_to_target
        # submits a real order and writes the ledger — a CancelledError
        # surfacing anywhere inside it (this is the exact id=377/id=949
        # shape: it reproduces that incident's symptom precisely — a
        # decision row exists, action_label is permanently NULL, no
        # execution note, no log) gives no guarantee about how far it got,
        # so retrying could double an order already in flight. The only
        # safe move is to log what's known and propagate — the caller
        # (auto_trade_loop) decides whether/how to re-examine this ticker
        # next cycle, not this function guessing at a safe retry.
        logger.error(
            "%s: rebalance_to_target hit a CancelledError for decision id=%s (%s) — "
            "NOT retrying (submits a real order, not idempotent); propagating so no "
            "action_label/execution note gets recorded for a trade whose actual "
            "outcome is unknown here (see id=377/id=949)",
            ticker, state.decision_id,
            "a real task cancellation" if task_was_actually_cancelled() else "an unrequested cancellation",
        )
        raise
    # The risk cap / no-resolvable-stop guard can shrink the trade below what
    # the narrative already stored via store_pending above described —
    # correct the record rather than let a future decision's past_context
    # retrieve a narrative that doesn't match what actually happened.
    #
    # Both calls' own internal `except Exception` already degrade quietly on
    # an ordinary DB failure (see their docstrings) — this bracketing log is
    # for the gap that isn't: a real incident (2026-08-05, decision id=377,
    # 005490.KS; recurred 2026-08-21, decision id=949, 950160.KS) left
    # action_label permanently NULL with no execution-note appended and NO
    # warning logged anywhere, right around a window with unrelated DB
    # connection-pool timeouts logged minutes later. The leading suspect is
    # asyncio.CancelledError — it subclasses BaseException, not Exception,
    # so a cancelled connection-pool acquire here would silently unwind past
    # both functions' own except-Exception handlers without a trace.
    if state.decision_id is not None:
        logger.info(
            "%s: recording execution metadata (action_label=%s) for decision id=%d",
            ticker, rebalance_result.action_label, state.decision_id,
        )
        # A genuine outer cancellation (shutdown, an enclosing gather/
        # TaskGroup tearing this task down) must still propagate — that's
        # not this task's call to swallow. But `task.cancelling()` (3.11+)
        # is 0 when nothing has actually asked *this* task to cancel, which
        # is exactly the suspected id=377/id=949 shape: a CancelledError
        # surfacing here from something else's cancelled await (e.g. the
        # pool's own internal wait) rather than from us being told to stop.
        # Both calls below are single idempotent UPDATEs (by decision_id),
        # so retrying once on that specific case is safe and turns a
        # permanently-lost action_label into, at worst, a one-cycle delay.
        for attempt in (1, 2):
            try:
                if rebalance_result.adjustment_note:
                    await append_execution_note(state.decision_id, rebalance_result.adjustment_note)
                # Buy/Sell/Increase/Decrease/Hold — what the rebalance
                # actually did to the position, separate from (and can
                # disagree with) state.final_decision.rating's own
                # conviction judgment; see rebalance_action_label's docstring.
                await set_action_label(state.decision_id, rebalance_result.action_label)
                logger.info("%s: execution metadata recorded for decision id=%d", ticker, state.decision_id)
                break
            except asyncio.CancelledError:
                really_cancelled = task_was_actually_cancelled()
                if attempt == 2 or really_cancelled:
                    logger.error(
                        "%s: execution metadata NOT recorded for decision id=%d "
                        "(action_label=%s) — %s",
                        ticker, state.decision_id, rebalance_result.action_label,
                        "propagating a real task cancellation" if really_cancelled
                        else "gave up after retrying an unrequested CancelledError",
                    )
                    raise
                logger.warning(
                    "%s: execution metadata recording for decision id=%d hit an "
                    "unrequested CancelledError (suspected connection-pool "
                    "acquire cancellation, see id=377/id=949) — retrying once",
                    ticker, state.decision_id,
                )
    state.rebalance_adjustment_note = rebalance_result.adjustment_note
    return state
