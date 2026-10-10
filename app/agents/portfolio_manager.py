"""Portfolio Manager — terminal decision-maker. The only node that reads
past-context memory (Postgres+pgvector similarity retrieval, wired in
Phase 3 — see `app/memory/store.py`) and the current actual position
(`app/execution/ledger.py`) — funneled here deliberately rather than
broadcast to every judgment node, so the learning signal stays concentrated
at the one node making the final call instead of diluted everywhere.

`_render_risk_debate` below exists because a live-decision audit
(2026-08-04) found the PM silently adopting a stop-loss/sizing number from
an EARLIER or explicitly-rejected risk-debator turn instead of the last
round's Neutral synthesis (app.agents.risk_debators._STANCES documents
Neutral's job as proposing the actual compromise) — e.g. one decision's
final stop-loss matched only the Aggressive debator's number from a round
Neutral and Conservative had already argued past. `DebateState.transcript()`
(app.state) is a flat, unannotated concatenation shared with the bull/bear
stage (which has no such synthesis-role convention), so nothing in the raw
transcript told the PM which numbers were actually load-bearing — it had to
infer that itself from reading order, and sometimes inferred wrong. Fixed by
annotating the final round's Neutral turn as the debate's authoritative
conclusion, not by changing DebateState/transcript() itself (shared code,
would leak risk-debate-specific semantics into the bull/bear stage too)."""

from __future__ import annotations

import logging
import re

from pydantic import BaseModel, Field

from app.agents.untrusted_content import downstream_injection_warning_for_field, redact_injection_flag
from app.config import AIProfile
from app.llm import llm_client
from app.llm.language import DEFAULT_LANGUAGE
from app.state import (
    DebateState,
    FlatPortfolioDecision,
    PortfolioDecision,
    PortfolioRating,
    TraderProposal,
    is_blank_card_reason,
    render_trader_proposal,
)

logger = logging.getLogger("app.agents.portfolio_manager")

# ---------------------------------------------------------------------------
# Narrow retry-transport schemas (2026-09-07, AI-engine step-0 latency audit).
# Each PM retry below used to hand `structured_complete` the FULL decision
# schema (`PortfolioDecision`/`FlatPortfolioDecision` — executive_summary,
# investment_thesis, price_target, time_horizon, ... a dozen fields) just to
# recover one or two blank ones, even though only the missing field(s) are
# ever grafted back (see each retry's own docstring — "a second generation
# from the same evidence is not more authoritative than the first"). Step-0
# instrumentation found this desk's latency is ~94% decode-bound (completion
# tokens), not prompt-bound — regenerating a full prose decision to fill in
# one number is pure decode waste. These schemas are transport shapes for the
# retry call only, never a decision type: they live here, not in app.state,
# so nothing downstream is tempted to treat one as a `PortfolioDecision`.
#
# `rating` is kept in the confidence/stop_loss schemas (not just the value
# being recovered) specifically to preserve the "different rating => discard
# the whole retry" guard below — a number is only usable as evidence about
# the rating it was stated alongside, and the echoed rating is that
# statement made checkable. `card_reason` has no such field (matching the
# original full-schema retry, which never checked rating for these two
# display-only strings either).
class _ConfidenceFill(BaseModel):
    rating: PortfolioRating = Field(
        description=(
            "The `rating` this value applies to — repeat it exactly as it appeared in the "
            "decision above, unchanged. A confidence value is only usable as evidence about "
            "the rating it is stated alongside; if you would answer with a different rating "
            "now, put that rating here rather than the original one."
        )
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Your conviction that THIS rating is correct, as a 0.0-1.0 fraction — separate "
            "from the rating itself (a Hold can be a 0.9-confidence call, a Buy can be a "
            "0.5 one). This applies to EVERY rating, including Hold, Sell, and Underweight "
            "— a bearish or neutral call needs a stated conviction just as much as a "
            "bullish one does. The execution layer uses this only to scale position size "
            "DOWN, never up, so understating it costs nothing but a smaller trade, while "
            "overstating it risks a larger one than the evidence supports."
        ),
    )


class _ConfidenceCardReasonFill(_ConfidenceFill):
    """`_ConfidenceFill` plus the two card_reason fields — used only when
    those were ALSO missing on the original decision, so the opportunistic
    graft in `_retry_if_confidence_missing` (a pure cost-saving so
    `_retry_if_card_reason_missing` doesn't need a third call right after
    this one) still has something to read. The model isn't told this is
    optional bonus content; it's asked for outright (see
    `_MISSING_CONFIDENCE_AND_CARD_REASON_RETRY_MESSAGE`)."""

    card_reason_ko: str | None = Field(
        default=None,
        description=(
            "One to two short sentences (under ~40 words), IN KOREAN regardless of this "
            "response's own language, stating the concrete issue or data driving this "
            "ticker's outlook — for a dashboard card, not a report. Cite the actual "
            "evidence: supply/demand flows, an earnings result, a specific news item or "
            "disclosure, a technical indicator level, or a valuation metric. Example "
            "style: '외국인 5거래일 연속 순매수, HBM 모멘텀 유효. 5일선 반등 신호 포착.' or "
            "'단기 과열 · RSI 74. 차익실현 매물 출회 가능성 높음.' Never describe the agents, "
            "the debate, or the synthesis process (e.g. do NOT write anything like 'the "
            "Aggressive case was validated' or 'the bull case prevailed'); never just "
            "restate the rating; and never include an action plan, position sizing, "
            "stop-loss, entry/exit, or time horizon — that belongs in executive_summary, "
            "not here."
        ),
    )
    card_reason_en: str | None = Field(
        default=None,
        description=(
            "One to two short sentences (under ~40 words), IN ENGLISH, stating the "
            "concrete issue or data driving this ticker's outlook — for a dashboard "
            "card, not a report. Cite the actual evidence: supply/demand flows, an "
            "earnings result, a specific news item or disclosure, a technical indicator "
            "level, or a valuation metric. Example style: 'Foreign investors net-bought "
            "for 5 straight sessions; HBM momentum intact. 5-day MA showing a rebound "
            "signal.' or 'Short-term overbought, RSI 74. High risk of profit-taking "
            "supply.' Never describe the agents, the debate, or the synthesis process "
            "(e.g. do NOT write anything like 'the Aggressive case was validated' or "
            "'the bull case prevailed'); never just restate the rating; and never "
            "include an action plan, position sizing, stop-loss, entry/exit, or time "
            "horizon — that belongs in executive_summary, not here."
        ),
    )


class _StopLossFill(BaseModel):
    rating: PortfolioRating = Field(
        description=(
            "The `rating` this value applies to — repeat it exactly as it appeared in the "
            "decision above, unchanged. A stop-loss is only usable as evidence about the "
            "rating it is stated alongside; if you would answer with a different rating "
            "now, put that rating here rather than the original one."
        )
    )
    stop_loss: float = Field(
        description=(
            "Stop-loss price for the position this rating implies. Set this whenever the "
            "rating implies holding or adding to a position (Buy/Overweight/Hold); the "
            "execution layer enforces it in code, so it must be a concrete, actionable "
            "price in the instrument's quote currency, not a vague level."
        ),
    )


class _CardReasonFill(BaseModel):
    card_reason_ko: str | None = Field(
        default=None,
        description=(
            "One to two short sentences (under ~40 words), IN KOREAN regardless of this "
            "response's own language, stating the concrete issue or data driving this "
            "ticker's outlook — "
            "for a dashboard card, not a report. Cite the actual evidence: supply/demand "
            "flows, an earnings result, a specific news item or disclosure, a technical "
            "indicator level, or a valuation metric. Example style: '외국인 5거래일 연속 "
            "순매수, HBM 모멘텀 유효. 5일선 반등 신호 포착.' or '단기 과열 · RSI 74. 차익실현 "
            "매물 출회 가능성 높음.' Never describe the agents, the debate, or the synthesis "
            "process (e.g. do NOT write anything like 'the Aggressive case was validated' "
            "or 'the bull case prevailed'); never just restate the rating; and never "
            "include an action plan, position sizing, stop-loss, entry/exit, or time "
            "horizon — that belongs in executive_summary, not here."
        ),
    )
    card_reason_en: str | None = Field(
        default=None,
        description=(
            "One to two short sentences (under ~40 words), IN ENGLISH, stating the "
            "concrete issue or data driving this ticker's outlook — for a dashboard "
            "card, not a report. Cite the actual evidence: supply/demand flows, an "
            "earnings result, a specific news item or disclosure, a technical indicator "
            "level, or a valuation metric. Example style: 'Foreign investors net-bought "
            "for 5 straight sessions; HBM momentum intact. 5-day MA showing a rebound "
            "signal.' or 'Short-term overbought, RSI 74. High risk of profit-taking "
            "supply.' Never describe the agents, the debate, or the synthesis process "
            "(e.g. do NOT write anything like 'the Aggressive case was validated' or "
            "'the bull case prevailed'); never just restate the rating; and never "
            "include an action plan, position sizing, stop-loss, entry/exit, or time "
            "horizon — that belongs in executive_summary, not here."
        ),
    )


_MISSING_CONFIDENCE_RETRY_MESSAGE = (
    "The decision above is missing a required field: `confidence`. Every rating other "
    "than Hold requires it — without it, execution scores this decision at the lowest "
    "confidence band and skips the trade entirely. Do not reconsider the decision. Reply "
    "with ONLY two fields: `rating`, repeated exactly as it appeared above, and "
    "`confidence` as a 0.0-1.0 fraction consistent with the thesis you already wrote."
)

_MISSING_CONFIDENCE_AND_CARD_REASON_RETRY_MESSAGE = (
    "The decision above is missing required fields: `confidence`, and also "
    "`card_reason_ko`/`card_reason_en`. Every rating other than Hold requires confidence "
    "— without it, execution scores this decision at the lowest confidence band and skips "
    "the trade entirely. card_reason_ko/en are required on every decision — the dashboard "
    "signal card renders from them. Do not reconsider the decision. Reply with ONLY these "
    "fields: `rating`, repeated exactly as it appeared above; `confidence` as a 0.0-1.0 "
    "fraction consistent with the thesis you already wrote; and, in one to two short "
    "sentences each, `card_reason_ko` in Korean and `card_reason_en` in English naming the "
    "concrete issue or data behind this outlook — no action plan, no "
    "sizing/stop-loss/horizon, and no mention of the debate or the agents."
)

_MISSING_STOP_LOSS_RETRY_MESSAGE = (
    "The decision above is missing a required field: `stop_loss`. This rating opens or "
    "adds to a position (Buy/Overweight), and `stop_loss` is required whenever new exposure "
    "is being taken on — without it, execution silently falls back to a mechanical ATR-based "
    "floor instead of your own risk assessment. Do not reconsider the decision. Reply with "
    "ONLY two fields: `rating`, repeated exactly as it appeared above, and `stop_loss` as a "
    "price in this ticker's own currency."
)

_MISSING_CARD_REASON_RETRY_MESSAGE = (
    "The decision above is missing required fields: `card_reason_ko` and/or "
    "`card_reason_en`. Both are required on every decision — the dashboard signal card "
    "renders from them. Do not reconsider the decision. Reply with ONLY these fields: one "
    "to two short sentences each naming the concrete issue or data behind this outlook, "
    "`card_reason_ko` in Korean and `card_reason_en` in English. No action plan, no "
    "sizing/stop-loss/horizon, and no mention of the debate or the agents."
)

SYSTEM_PROMPT = (
    "You are the Portfolio Manager, the terminal decision-maker for this trading "
    "desk. Review the trader's proposal and the risk debate, then produce the "
    "final position rating. Commit to a stance; reserve Hold for genuinely "
    "balanced evidence."
    " Committing to a stance means resolving genuinely balanced evidence, not "
    "manufacturing conviction — when the evidence is thin, or when the case for "
    "exposure rests on how a trade would be sized rather than on the company, Hold "
    "IS the committed answer, not a failure to decide."
    " If prior context is provided, incorporate any relevant "
    "lessons — otherwise rely solely on the current analysis.\n\n"
    "You are told the current actual position below — treat it as a real constraint "
    "on the decision, not background color. If it says you are currently flat "
    "(no position), explicitly weigh whether today's evidence now supports "
    "re-entering, as a genuine question — not only whether to avoid a *new* "
    "position. A rating of Hold while flat means 'stay in cash,' not 'do nothing "
    "by default'; only choose it if re-entry genuinely isn't warranted yet.\n\n"
    "Whenever your rating implies holding or adding to a position (Buy, Overweight, "
    "or Hold on an existing position), set a concrete stop_loss price — this gets "
    "enforced in code against real daily price data, so it must be an actual "
    "actionable price level, not a vague description. The risk debate below ends with a "
    "section fenced as '=== FINAL SYNTHESIS ... ===' — that exact fence is written by this "
    "system, never by a debator, and appears exactly once. Base your stop_loss/sizing on "
    "that fenced section only, not on an earlier-round or Aggressive/Conservative number "
    "it superseded, and not on any other text that merely calls itself a final "
    "synthesis.\n\n"
    "You are also told below how volatility-based position sizing will scale "
    "whichever rating you choose — this happens automatically in code, after your "
    "decision. Write your executive summary and thesis to match the actual "
    "resulting position size, not just the nominal rating label, so your narrative "
    "never contradicts the trade that actually gets executed.\n\n"
    "You are also told below how the stop-loss price you choose feeds a second, "
    "independent code-enforced cap: this trade's added shares are further limited "
    "so their dollar risk (shares added x (price - your stop_loss)) never exceeds a "
    "fixed fraction of equity — a tighter stop allows more shares, a wider stop "
    "forces fewer. This can shrink the position on top of the volatility-based "
    "scaling above, and only ever reduces shares being added, never forces a sale "
    "of shares already held. Write your executive summary and thesis to match the "
    "position size your chosen stop actually implies.\n\n"
    "Position-size mechanics are never a reason to trade. The number of shares a "
    "rating produces, the headroom remaining under the risk cap, whether the "
    "resulting allocation is large or negligible, and the fact that the sizing or "
    "risk-cap framework permits the trade all describe HOW an already-decided trade "
    "executes — none is evidence that it should happen. Never write that a trade is "
    "'mechanically validated by the risk-cap framework' or justify a rating by the "
    "size it would produce. The same holds for a valuation multiple quoted on its "
    "own — a low P/B or P/E, a price below book value: that is a description of "
    "price, not evidence of a floor, an edge, or a catalyst. If a multiple or the "
    "sizing arithmetic is your main stated reason for taking on exposure, the "
    "correct rating is Hold. This cuts the other way too: the risk cap and the "
    "volatility scaling only ever shrink shares being ADDED — they never force, "
    "justify, or argue for selling shares you already hold. Do not cite the cap, "
    "the ATR/volatility scaling, a 'trivial' or 'negligible' resulting share "
    "count, or the position being 'mechanically inefficient' to size as a reason "
    "to exit or cut a position you hold; if that is your main stated reason for a "
    "Sell or a reduction, the correct rating is Hold.\n\n"
    "You are shown, above the trader's proposal, what the Research Manager "
    "concluded and what the Trader proposed BEFORE the risk debate ran. Both were "
    "formed from the full analyst reports and the bull/bear debate, which you never "
    "see directly. When BOTH point away from more exposure (Hold, Underweight, or "
    "Sell) and you are nevertheless choosing Buy or Overweight, you are overriding "
    "the entire desk on the strength of the risk debate alone. That is permitted, "
    "but only on a specific new fact: name in `executive_summary` the concrete "
    "piece of evidence the risk debate introduced that they did not have or did not "
    "weigh — a disclosed number, a price level, a filing, a datum. A re-reading, a "
    "re-weighting, a better-argued version of the same evidence, or simply the fact "
    "that the debate concluded differently is NOT a new fact. If you cannot name "
    "one, choose Hold: a position you can only justify by disagreeing with everyone "
    "upstream is a position to skip.\n\n"
    "Call the desk unanimous, in agreement, aligned, or of one mind ONLY when the "
    "Research Manager's conclusion and the Trader's proposal — as stated in the "
    "block above — are both the same as your own rating. If either one differs "
    "from you or from each other, do not describe the desk as unified: state the "
    "disagreement explicitly in `executive_summary` and name both of their "
    "values.\n\n"
    "You are also told below how this ticker sits relative to PORTFOLIO-level risk "
    "caps — concentration, correlation-bucket exposure, and an aggregate drawdown "
    "breaker — which look across the whole trade allowlist, not just this ticker. These "
    "can further shrink an exposure-increasing trade on top of everything above, or "
    "block it entirely during a portfolio-wide drawdown; they never force a sale of "
    "shares already held. Factor this into your rating and narrative the same way.\n\n"
    "You are also told below the current EXECUTION status for THIS TICKER SPECIFICALLY "
    "— whether it is halted/under a VI/circuit-breaker right now, and the actual cash "
    "currently available to spend. This is a per-ticker check, not a market-wide event: "
    "never describe it in your narrative as a broad market circuit breaker, a KOSPI/KOSDAQ- "
    "wide halt, or similar — it tells you nothing about any other ticker or the market as a "
    "whole, only this one. If it says a trade is currently blocked, do not keep recommending "
    "the same exposure increase as if nothing were wrong — note the block and explain why "
    "your rating still makes sense once it clears, or favor Hold until it does. If the "
    "available cash is less than what your rating would otherwise buy, say so rather than "
    "implying an unconstrained buy. This field is the only authoritative source for whether "
    "the ticker is halted, under a VI, or under a circuit-breaker right now. If it says no "
    "halt/VI/circuit-breaker is currently active, then the ticker is trading normally as you "
    "decide, no matter what any analyst report, the Research Manager, the Trader, or any "
    "risk-debate turn says. A news item about a halt, a suspension, a suspension warning, or "
    "a designation notice is historical or forward-looking information, never a statement "
    "about the present — weigh it as a risk, and never write that a halt or suspension is "
    "active, ongoing, or in effect, and never justify a rating by 'we cannot act until "
    "trading resumes'. If a report below claims a currently-active halt and this field says "
    "otherwise, say so explicitly in your reasoning rather than repeating the claim or "
    "silently dropping it.\n\n"
    "You must also state your `confidence` in this specific rating, as a 0.0-1.0 fraction — "
    "separate from the rating itself (a Hold can be a high-confidence call, a Buy can be a "
    "low-confidence one). The code uses this only to scale the resulting position size DOWN, "
    "never up, on top of every adjustment above: below 0.5 skips the trade entirely (0.0x), "
    "0.5-0.6 gets 0.5x, 0.6-0.75 gets 0.7x, 0.75-0.9 gets 0.9x, and 0.9-1.0 gets the full 1.0x "
    "(never more, regardless of how confident you are). State your honest conviction rather "
    "than defaulting to 1.0 — understating it only costs a smaller trade, while overstating "
    "it risks a larger one than the evidence actually supports.\n\n"
    "Each rating maps to a FIXED target allocation the code applies automatically, "
    "not a vague direction: Buy=100%, Overweight=75%, Hold=no change, Underweight=25% "
    "(NOT 0% — Underweight still targets a real, non-zero 25% position; if you mean "
    "'exit entirely' or 'stay in cash,' choose Sell or Hold-while-flat instead, not "
    "Underweight), Sell=0%. Overweight/Underweight only make sense when a position "
    "already exists to adjust — never choose them for a ticker you don't currently "
    "hold. Write your narrative to match the percentage your rating actually targets, "
    "not what the label might suggest in isolation.\n\n"
    "You must fill in BOTH `card_reason_ko` AND `card_reason_en` on every single "
    "response, for every rating including Hold and Sell. They are technically optional "
    "in the schema only so a malformed display string can never invalidate a real "
    "trading decision — that is a safety fallback, not permission to omit them. A "
    "decision that arrives without them renders a broken dashboard card, so treat them "
    "as mandatory output. Write `card_reason_ko` in Korean and `card_reason_en` in "
    "English, both times, regardless of which language the rest of this response is in "
    "— they are not translations of each other's presence, they are two required "
    "fields. Each is one to two short sentences (under ~40 words), stating the concrete "
    "issue or data behind this ticker's outlook, and nothing else: no action plan, no "
    "sizing, no stop-loss, no time horizon (those belong in `executive_summary`), and "
    "no reference whatsoever to this desk's internal process — never name or allude to "
    "the debate, the debators, the trader's proposal, the 'FINAL SYNTHESIS' section, or "
    "the agents involved. Do not copy `executive_summary` into them verbatim, do not "
    "merely restate the rating, and never write a placeholder like 'N/A', 'none', or an "
    "empty string in place of a real reason.\n\n"
    "Mandate constraints: this account is LONG-ONLY with NO leverage — no short "
    "selling, no margin, no options, futures, or other derivatives, and no instrument "
    "other than the common stock of this ticker. The rating enum itself already makes "
    "a short unselectable, but do not let `investment_thesis`/`executive_summary` "
    "narrate one anyway (e.g. 'initiate a short position' — a real 2026-08-28 incident "
    "in an upstream node's rationale for 010170.KS, unexecutable and discarded "
    "downstream). If the case is bearish on a ticker you don't hold, that is Hold, not "
    "a short.\n\n"
    f"{downstream_injection_warning_for_field('investment_thesis')}"
)

# AIProfile suffixes — persona/verbosity only, appended to SYSTEM_PROMPT above.
# See app.config.AIProfile: never touches RiskManager/Broker/Approval.
# 프로필별 페르소나 suffix — 판단 성향만 바꿈, 리스크/브로커/승인에는 영향 없음.
_PROFILE_SUFFIXES: dict[AIProfile, str] = {
    AIProfile.PRODUCTION: "",
    AIProfile.TEST_AGGRESSIVE: (
        "\n\nTEST_AGGRESSIVE mode: this run exists to exercise the order pipeline end-to-end, "
        "not to produce investment-quality calls. Reserve Hold for only the rarest, most "
        "genuinely balanced cases — lean into Buy/Sell/Overweight/Underweight ratings (whichever "
        "are actually available to you below — a flat ticker only offers Buy/Hold) so downstream "
        "approval/execution paths actually see varied order events."
    ),
    AIProfile.DEBUG: (
        "\n\nDEBUG mode: write `investment_thesis` with maximum detail — name every input you "
        "weighed (trader proposal, risk debate, position/volatility/risk-cap context, past "
        "context), state your confidence level explicitly, and walk through the reasoning "
        "steps, not just the conclusion."
    ),
}


# Appended only when `held` is False — a flat ticker's rating is already
# structurally limited to Buy/Hold (see FlatPortfolioDecision below), but
# telling the model why up front keeps its narrative from drifting toward
# "Underweight"/"Sell" language it then can't actually output.
_FLAT_SUFFIX = (
    "\n\nThis ticker is currently FLAT (no shares held) — your rating for it can only be "
    "Buy or Hold, not Overweight/Underweight/Sell, since there is no existing position to "
    "adjust or exit. If you'd otherwise have said 'Underweight' or 'stay in cash,' that "
    "means Hold here."
)


def _system_prompt(profile: AIProfile, held: bool) -> str:
    suffix = _PROFILE_SUFFIXES[profile] + ("" if held else _FLAT_SUFFIX)
    return SYSTEM_PROMPT + suffix


# Every debate turn is LLM prose that may be relaying injected third-party
# text (see app.agents.untrusted_content), and the synthesis marker below is
# the only thing telling the PM which numbers are authoritative — so a turn
# whose own content carries a literal "=== FINAL SYNTHESIS" fence renders a
# second, forged conclusion, earlier in the string than the real one, that
# the PM has no way to tell apart. Escaping the fence RUN rather than
# matching the marker phrase follows wrap_untrusted's reasoning: enumerating
# spelling/spacing variants is unwinnable, removing the character sequence
# that can begin ANY fence is not. Escaped to an entity rather than stripped
# so the turn stays readable, and deterministic rather than a per-call nonce
# so vLLM's prefix cache still matches. A run of 3+ '=' has no legitimate use
# in debate prose; ordinary "P/E = 12" is untouched.
_FENCE_RUN = re.compile("={3,}")


def _escape_fences(text: str) -> str:
    return _FENCE_RUN.sub(lambda m: "&#61;" * len(m.group()), text)


def _render_risk_debate(risk_debate: DebateState) -> str:
    """Full transcript, plus the last round's Neutral turn called out
    separately as the debate's actual conclusion — see this module's
    docstring for the bug this fixes. A no-op label (falls back to the
    plain transcript) if there's no Neutral turn in the final round, e.g. a
    convergence/profile change ever drops that party."""
    transcript = _escape_fences(risk_debate.transcript())
    if not risk_debate.turns:
        return transcript
    final_round = risk_debate.rounds_completed or risk_debate.turns[-1].round
    # Reverse scan, matching DebateState.latest_by: "final" is the last such
    # turn, not the first one that happens to match.
    neutral_final = next(
        (t for t in reversed(risk_debate.turns) if t.round == final_round and t.speaker == "Neutral"), None
    )
    if neutral_final is None:
        return transcript
    return (
        f"{transcript}\n\n"
        f"=== FINAL SYNTHESIS (Round {final_round}, Neutral) ===\n"
        "This is the debate's actual concluding judgment, not just another opinion — "
        "prefer its concrete stop-loss/sizing numbers over any earlier-round or "
        "Aggressive/Conservative proposal unless you have a specific, stated reason to "
        "deviate. Do not silently substitute a different debator's number.\n"
        f"{_escape_fences(neutral_final.content)}"
    )


def _needs_confidence(rating) -> bool:
    """Every rating except Hold — deliberately WIDER than
    `app.state.is_state_flipping_rating`, which this retry used until
    2026-08-26, and kept as its own predicate so the hysteresis gate's
    shared one stays untouched (same precedent as `_needs_stop_loss` below,
    which carries its own narrower scope for its own reason).

    Two independent mechanisms consume `confidence`, not one. The
    hysteresis gate only touches the state-flipping pair (Buy while flat,
    Sell while held), but `app.execution.sizing.target_allocation_for`
    scores a missing value as the lowest confidence band for every rating
    with a non-None base allocation and returns None — skip the trade —
    for Buy/Overweight/Underweight too. Hold's target is None regardless of
    confidence, so a missing value there genuinely changes nothing.
    Measured 2026-08-26 (live, non-backtest decisions): 0 of 16 Buy/Sell
    decisions were missing confidence — the retry works — against 7 of 10
    Underweight and 2 of 6 Overweight, which never got one. An Underweight
    is the model asking to CUT exposure, so the skip leaves the full
    position standing: the same fail-unsafe shape as a vetoed Sell, not a
    mere lost opportunity."""
    return rating.value != "Hold"


async def _retry_if_confidence_missing(messages, decision, language: str):
    """One extra `structured_complete` call — against the narrow
    `_ConfidenceFill`/`_ConfidenceCardReasonFill` schema, not the original
    decision's full schema (2026-09-07, AI-engine step-0 latency audit: this
    desk's latency is ~94% decode-bound, so regenerating executive_summary/
    investment_thesis/etc just to recover one number was pure waste) —
    when the model omitted `confidence` on any non-Hold rating
    (`_needs_confidence` above) — the hysteresis gate
    (`app.orchestrator.pipeline._apply_hysteresis_gate`) treats a missing
    value as below its flip threshold on Buy/Sell, and
    `app.execution.sizing.target_allocation_for` treats it as the lowest
    band (skip the trade) on every other non-Hold rating, so silently
    omitting this field has been vetoing the model's own rating/sizing
    without ever stating a low-confidence number (measured: 9 of 9 live
    gate downgrades since 2026-08-13 were a missing field, not a genuine low
    number). Never loops more than once, and never fails the pipeline over
    this: an exception from the retry degrades quietly back to the original
    decision, matching this codebase's other best-effort-retry modules
    (see app.execution.risk_rules / app.live.fill_monitor).

    On success, only the retried `confidence` value is grafted onto the
    *original* decision (`decision.model_copy(update={"confidence": ...})`,
    following the same validator-skipping precedent as
    `app.orchestrator.pipeline._apply_hysteresis_gate`) — every other field
    (rating, stop_loss, price_target, investment_thesis, ...) is kept from
    the original response. This retry's only job is to fill in one missing
    number; nothing in the retry prompt actually enforces that the rest of
    the decision comes back unchanged, and a second response generated from
    the same evidence is not more authoritative than the first, so it must
    never be allowed to silently flip a Sell into a Buy (or change any other
    field)."""
    if decision.confidence is not None or not _needs_confidence(decision.rating):
        return decision
    # Narrow schema (2026-09-07, AI-engine step-0 latency audit): only ask
    # the model to re-emit `rating` (echoed, for the mismatch guard below)
    # and `confidence` — never the full PortfolioDecision/FlatPortfolioDecision
    # shape, which would force a fresh executive_summary/investment_thesis
    # decode for the sake of one number. Widened to also ask for card_reason
    # when that was ALSO left blank, preserving the cost-saving graft below
    # without a third call.
    needs_card_reason_too = is_blank_card_reason(decision.card_reason_ko) or is_blank_card_reason(
        decision.card_reason_en
    )
    schema = _ConfidenceCardReasonFill if needs_card_reason_too else _ConfidenceFill
    retry_instruction = (
        _MISSING_CONFIDENCE_AND_CARD_REASON_RETRY_MESSAGE
        if needs_card_reason_too
        else _MISSING_CONFIDENCE_RETRY_MESSAGE
    )
    # The original decision is replayed back as an ASSISTANT turn (2026-08-25
    # — see the gemma-4 incident this fixes below) rather than only described
    # in the retry instruction's prose. Before this, `retry_messages` was
    # `[*messages, user_retry_instruction]` — no assistant turn at all, so
    # "your previous response" referred to a turn that didn't exist in the
    # conversation being replayed, and "return the same decision again,
    # unchanged" asked the model to reconstruct an answer it couldn't see.
    # Confirmed live against RedHatAI/gemma-4-26B-A4B-it-NVFP4 (2026-08-25):
    # with no assistant turn, the model ignored the explicit retry request
    # outright (HTTP 200, valid schema, `confidence` simply absent again) —
    # 71% of that day's live decisions were wrongly downgraded via the
    # hysteresis gate as a result (see project memory:
    # gemma4-confidence-omission-2026-08-25). Putting the actual prior
    # answer in front of the model turns this into a genuine "fill in the
    # blank" edit instead of a from-scratch re-roll with no memory of
    # attempt 1 — which independently matters even for a model that DOES
    # comply: without it, a differing rating on retry (a real re-roll) had
    # its confidence grafted onto attempt 1's rating with no way to tell the
    # two apart.
    retry_messages = [
        *messages,
        {"role": "assistant", "content": decision.model_dump_json()},
        {"role": "user", "content": retry_instruction},
    ]
    try:
        retried = await llm_client.structured_complete(
            retry_messages,
            schema,
            # reasoning=False (2026-08-25, ai-engine-token-cost-investigation):
            # this retry re-states an already-made decision to fill in one
            # missing number, not a fresh deliberation — the thinking-mode
            # decode cost of the pipeline's single largest prompt buys nothing
            # here and was never deliberate (just inherited True from the
            # original call's context).
            reasoning=False,
            label="portfolio_manager",
            language=language,
        )
    except Exception:  # noqa: BLE001
        logger.warning(
            "portfolio_manager: confidence retry for rating=%s raised, keeping original decision",
            decision.rating,
            exc_info=True,
        )
        return decision
    if retried.confidence is None:
        logger.info(
            "portfolio_manager: confidence retry fired for rating=%s but confidence is still missing, "
            "keeping original decision",
            decision.rating,
        )
        return decision
    # Compared by .value, not by identity/equality of the enum member itself:
    # a flat decision's `decision.rating` is a `FlatPortfolioRating`, while
    # the narrow schema's echoed `retried.rating` is always a
    # `PortfolioRating` (the narrow schema is shared across the flat/held
    # split, unlike the original full-schema retry which used `type(decision)`
    # and so echoed back the same enum class). Comparing the enum members
    # directly would report every flat retry as a "different rating" and
    # silently discard it — comparing the underlying string value is what the
    # guard actually means: "is this the same rating as before".
    if retried.rating.value != decision.rating.value:
        # A confidence stated about a DIFFERENT rating is not evidence about
        # this one — the retry prompt asks for the same rating unchanged,
        # but nothing enforces that, and a second generation from the same
        # evidence is not more authoritative than the first (this function's
        # own docstring). Discard the whole retry rather than grafting a
        # number that was never actually attached to `decision.rating`.
        logger.warning(
            "portfolio_manager: confidence retry for rating=%s returned a different rating=%s "
            "(confidence=%.2f) — discarding the retry, keeping the original decision unchanged",
            decision.rating, retried.rating, retried.confidence,
        )
        return decision
    logger.info(
        "portfolio_manager: confidence retry fired for rating=%s and produced confidence=%.2f",
        decision.rating,
        retried.confidence,
    )
    # Graft card_reason from this same retried response if the original was
    # missing it too — a pure cost-saving so _retry_if_card_reason_missing
    # (below) doesn't need a THIRD structured_complete call right after this
    # one. Still only ever fills in a field the original left blank; every
    # other field (rating, stop_loss, ...) still comes from the original.
    # Only attempted when `schema` was `_ConfidenceCardReasonFill` — the plain
    # `_ConfidenceFill` response has no card_reason_* attributes to read.
    update = {"confidence": retried.confidence}
    if needs_card_reason_too:
        for field_name in ("card_reason_ko", "card_reason_en"):
            if is_blank_card_reason(getattr(decision, field_name)) and not is_blank_card_reason(
                getattr(retried, field_name)
            ):
                update[field_name] = getattr(retried, field_name).strip()
    return decision.model_copy(update=update)


def _needs_stop_loss(rating) -> bool:
    """Buy/Overweight are the ratings that open or add to a position — the
    only cases where `app.execution.ledger.rebalance_to_target` actually
    uses `stop_loss` to protect new exposure. Hold never trades; Sell exits
    the position entirely, leaving nothing for a stop to protect; Underweight
    reduces but doesn't add, so both are excluded — deliberately narrower
    scope than `_needs_confidence` above (which fires on every non-Hold
    rating), matched tightly to the three confirmed incidents (all Buy)."""
    return rating.value in ("Buy", "Overweight")


async def _retry_if_stop_loss_missing(messages, decision, language: str):
    """One extra `structured_complete` call, same shape as
    `_retry_if_confidence_missing` above, when the model wrote its stop-loss
    reasoning in prose (executive_summary/investment_thesis) but left the
    structured `stop_loss` field empty on a Buy/Overweight rating. Real
    incidents (2026-08-20 950160.KS, 2026-08-25 MRVL, 2026-08-26 002990.KS —
    three separate models across two markets) all show the same shape:
    `app.execution.ledger.resolve_stop_loss` silently substitutes a
    mechanical ATR-floor stop that can differ from the AI's own number by
    20-30+ percentage points, with the substitution only ever visible in
    `trades.adjustment_note` — nowhere the AI's own reasoning or the
    dashboard card is displayed. `app.state`'s
    `_recover_fields_from_rendered_header` already recovers the narrow case
    of a LEADING "**Stop Loss**: value" line, but none of the three
    incidents above wrote it that way — the number was embedded mid-sentence
    in ordinary prose, which no reasonable regex can safely extract (a stray
    dollar figure elsewhere in the paragraph — a price target, a resistance
    level, a risk-cap dollar amount — would be indistinguishable from the
    actual stop by pattern matching alone). This retry sidesteps that
    entirely by asking the model itself, same as the confidence/card_reason
    retries above.

    Never loops more than once, never fails the pipeline over this: an
    exception degrades quietly back to the original decision, same as
    `_retry_if_confidence_missing`.

    Narrow schema (2026-09-07, AI-engine step-0 latency audit): asks only for
    `_StopLossFill` (rating echo + stop_loss), not the full decision schema
    — same reasoning as `_retry_if_confidence_missing`'s narrow-schema
    comment above."""
    if decision.stop_loss is not None or not _needs_stop_loss(decision.rating):
        return decision
    retry_messages = [
        *messages,
        {"role": "assistant", "content": decision.model_dump_json()},
        {"role": "user", "content": _MISSING_STOP_LOSS_RETRY_MESSAGE},
    ]
    try:
        retried = await llm_client.structured_complete(
            retry_messages,
            _StopLossFill,
            reasoning=False,
            label="portfolio_manager",
            language=language,
        )
    except Exception:  # noqa: BLE001
        logger.warning(
            "portfolio_manager: stop_loss retry for rating=%s raised, keeping original decision",
            decision.rating,
            exc_info=True,
        )
        return decision
    if retried.stop_loss is None:
        logger.info(
            "portfolio_manager: stop_loss retry fired for rating=%s but stop_loss is still missing, "
            "keeping original decision",
            decision.rating,
        )
        return decision
    # Compared by .value — see _retry_if_confidence_missing's identical
    # comment (flat decisions echo a `PortfolioRating`, not their own
    # `FlatPortfolioRating`, since the narrow schema is shared).
    if retried.rating.value != decision.rating.value:
        logger.warning(
            "portfolio_manager: stop_loss retry for rating=%s returned a different rating=%s "
            "(stop_loss=%.2f) — discarding the retry, keeping the original decision unchanged",
            decision.rating, retried.rating, retried.stop_loss,
        )
        return decision
    logger.info(
        "portfolio_manager: stop_loss retry fired for rating=%s and produced stop_loss=%.2f",
        decision.rating,
        retried.stop_loss,
    )
    return decision.model_copy(update={"stop_loss": retried.stop_loss})


async def _retry_if_card_reason_missing(messages, decision, language: str):
    """One extra `structured_complete` call, same shape as
    `_retry_if_confidence_missing` above, when the model left
    `card_reason_ko` and/or `card_reason_en` blank — measured 2026-08-19
    (git_commit=3b96efa, 7 decisions): only 1 of 7 had either field filled.
    Both fields are schema-optional on purpose (a malformed display string
    must never invalidate a real trading decision — see app.state's own
    comment on these fields), so nothing enforced them before this: no
    system-prompt emphasis, no retry, no missing-field log. Runs for every
    rating including Hold (unlike the confidence retry, which excludes only
    Hold) since a display reason is expected there too.

    On success, only whichever of the two fields was actually missing is
    grafted onto the *original* decision — never the retried response's
    rating/stop_loss/etc, and never the side that was already filled. A
    second generation from the same evidence is not more authoritative than
    the first, so it must never flip anything but these two display fields.

    Narrow schema (2026-09-07, AI-engine step-0 latency audit): asks only
    for `_CardReasonFill` (the two display strings, no `rating` echo — this
    retry never checked rating agreement even under the old full-schema
    form, since these are display-only fields), not the full decision
    schema — same reasoning as `_retry_if_confidence_missing`'s
    narrow-schema comment above."""
    ko_missing = is_blank_card_reason(decision.card_reason_ko)
    en_missing = is_blank_card_reason(decision.card_reason_en)
    if not (ko_missing or en_missing):
        return decision
    # Drop a literal placeholder (e.g. "N/A") to None before the retry, not
    # after: every failure path below (raise, still-degenerate response)
    # returns `decision` as-is, and a placeholder rendered verbatim on the
    # card is worse than the card's own fallback, which only triggers on a
    # falsy value. Skipped when the field is already None — nothing to drop,
    # and callers (e.g. the `decide()` chain) rely on getting back the exact
    # same object when this retry has nothing to do.
    placeholder_clear = {
        field_name: None
        for field_name in ("card_reason_ko", "card_reason_en")
        if getattr(decision, field_name) is not None and is_blank_card_reason(getattr(decision, field_name))
    }
    if placeholder_clear:
        decision = decision.model_copy(update=placeholder_clear)
    # Same 2026-08-25 fix as _retry_if_confidence_missing above, same
    # gemma-4 incident — see that function's comment for the live-verified
    # reasoning.
    retry_messages = [
        *messages,
        {"role": "assistant", "content": decision.model_dump_json()},
        {"role": "user", "content": _MISSING_CARD_REASON_RETRY_MESSAGE},
    ]
    try:
        retried = await llm_client.structured_complete(
            retry_messages,
            _CardReasonFill,
            # reasoning=False — same reasoning as _retry_if_confidence_missing
            # above: this only fills in two display strings, never re-derives
            # the decision itself.
            reasoning=False,
            label="portfolio_manager",
            language=language,
        )
    except Exception:  # noqa: BLE001
        logger.warning(
            "portfolio_manager: card_reason retry for rating=%s raised, keeping original decision",
            decision.rating,
            exc_info=True,
        )
        return decision
    update = {}
    if ko_missing and not is_blank_card_reason(retried.card_reason_ko):
        update["card_reason_ko"] = retried.card_reason_ko.strip()
    if en_missing and not is_blank_card_reason(retried.card_reason_en):
        update["card_reason_en"] = retried.card_reason_en.strip()
    if not update:
        logger.info(
            "portfolio_manager: card_reason retry fired for rating=%s but both fields are still "
            "missing, keeping original decision",
            decision.rating,
        )
        return decision
    logger.info(
        "portfolio_manager: card_reason retry fired for rating=%s and recovered %s",
        decision.rating,
        ",".join(sorted(update)),
    )
    return decision.model_copy(update=update)


async def decide(
    trader_proposal: TraderProposal,
    risk_debate: DebateState,
    past_context: str = "",
    position_context: str = "",
    volatility_context: str = "",
    risk_cap_context: str = "",
    portfolio_risk_context: str = "",
    execution_status_context: str = "",
    sibling_context: str = "",
    language: str = DEFAULT_LANGUAGE,
    profile: AIProfile = AIProfile.PRODUCTION,
    held: bool = True,
    identity: str = "",
    session_note: str = "",
    period_note: str = "",
    degradation_note: str = "",
    desk_consensus_note: str = "",
) -> PortfolioDecision:
    """`held`: whether the ticker currently has a non-zero position — False
    (flat) structurally restricts the LLM's rating to Buy/Hold via
    `FlatPortfolioDecision` (see app.state's docstring for the bug this
    fixes), then normalizes the result back into a regular
    `PortfolioDecision` so every caller downstream (sizing.py, ledger.py)
    keeps seeing the one familiar type regardless of which schema was used.

    `identity`: verified ticker-name anchor (app.orchestrator.pipeline's
    `identity_note`) — this node is the one place in the whole pipeline that
    reasons toward a rating/sizing decision without ever seeing the raw
    analyst reports (only the trader proposal + risk debate, both already
    prose), so it otherwise has no anchor at all to catch a misnamed company
    that survived that far. See app.agents.identity_gate's module docstring.

    `degradation_note`: set by app.orchestrator.pipeline._degradation_note when
    one or more analyst reports failed to generate and were replaced with a
    placeholder (2026-09-03 decision id=1179 — a rate-limited Fundamentals
    report degraded silently and nothing in the final decision showed it).
    Empty => prompt byte-identical to before.

    `desk_consensus_note`: set by app.orchestrator.pipeline._desk_consensus_note
    — a one-block summary of what the Research Manager concluded and what the
    Trader proposed BEFORE the risk debate ran, placed directly above the
    trader's proposal (the SYSTEM_PROMPT text about overriding the whole desk
    refers to this block). 2026-09-04, id=1212/1217. Empty => prompt
    byte-identical to before."""
    identity_block = f"{identity}\n\n" if identity else ""
    # No placeholder when empty (unlike every other block above) — a
    # same-day sibling is the exception, not the norm (most decisions have
    # none), so the common case's prompt stays byte-identical to before
    # this existed. sibling_context already carries its own header line
    # (get_same_day_sibling_context), so it isn't wrapped in another one
    # here — see that function's docstring for why this is a separate,
    # status-agnostic block rather than folded into "Past context" below
    # (which is resolved-only "graded lessons").
    sibling_block = f"{sibling_context}\n\n" if sibling_context else ""
    # session_note (app.dataflows.session_state.intraday_narration_note) —
    # after identity so the identity prefix stays cache-stable, before the
    # trader/risk-debate blocks it must correct. Empty (closed session /
    # backtest) => byte-identical to before. 2026-09-01, id=1157.
    session_block = f"{session_note}\n\n" if session_note else ""
    # period_note (app.dataflows.kis_provider.interim_period_note) — a KRX
    # interim/full-year mismatch that a Fundamentals report may already have
    # turned into a fake "decline". 2026-09-01, id=1158.
    period_block = f"{period_note}\n\n" if period_note else ""
    # degradation_note (app.orchestrator.pipeline._degradation_note) — one or
    # more analyst reports failed this cycle. Immediately before the trader
    # proposal it must be read against. Empty => byte-identical to before.
    # 2026-09-03, id=1179.
    degradation_block = f"{degradation_note}\n\n" if degradation_note else ""
    # desk_consensus_note (app.orchestrator.pipeline._desk_consensus_note) —
    # the RM conclusion + Trader proposal formed BEFORE the risk debate.
    # Placed immediately above the trader's proposal so it is literally
    # "shown, above the trader's proposal" as SYSTEM_PROMPT promises. Empty
    # => byte-identical to before. 2026-09-04, id=1212/1217.
    consensus_block = f"{desk_consensus_note}\n\n" if desk_consensus_note else ""
    prompt = (
        f"{identity_block}"
        f"{session_block}"
        f"{period_block}"
        f"{degradation_block}"
        f"{consensus_block}"
        f"Trader's proposal:\n{render_trader_proposal(trader_proposal)}\n\n"
        f"Risk debate transcript:\n{_render_risk_debate(risk_debate)}\n\n"
        f"Current position:\n{position_context or '(no position history available)'}\n\n"
        f"Position sizing / volatility:\n{volatility_context or '(no volatility data available)'}\n\n"
        f"Stop-loss risk cap:\n{risk_cap_context or '(no risk-cap data available)'}\n\n"
        f"Portfolio-level risk caps:\n{portfolio_risk_context or '(no portfolio-risk data available)'}\n\n"
        f"Execution status:\n{execution_status_context or '(no execution-status data available)'}\n\n"
        f"{sibling_block}"
        f"Past context:\n{past_context or '(none available)'}"
    )
    messages = [{"role": "system", "content": _system_prompt(profile, held)}, {"role": "user", "content": prompt}]
    if held:
        decision = await llm_client.structured_complete(
            messages,
            PortfolioDecision,
            reasoning=True,
            label="portfolio_manager",
            language=language,
        )
        decision = await _retry_if_confidence_missing(messages, decision, language)
        decision = await _retry_if_stop_loss_missing(messages, decision, language)
        decision = await _retry_if_card_reason_missing(messages, decision, language)
        decision.investment_thesis = redact_injection_flag(decision.investment_thesis, agent_name="Portfolio Manager")
        return decision
    flat_result = await llm_client.structured_complete(
        messages,
        FlatPortfolioDecision,
        reasoning=True,
        label="portfolio_manager",
        language=language,
    )
    flat_result = await _retry_if_confidence_missing(messages, flat_result, language)
    flat_result = await _retry_if_stop_loss_missing(messages, flat_result, language)
    flat_result = await _retry_if_card_reason_missing(messages, flat_result, language)
    # PortfolioDecision and FlatPortfolioDecision share every field except
    # `rating` (app.state._PortfolioDecisionFields) — so this is unpacked
    # wholesale rather than listed field-by-field. Listing fields is exactly
    # what caused a real incident: 2026-08-19 commit 3b96efa added
    # card_reason_ko/card_reason_en to the shared fields but missed this one
    # normalization site, so every flat ticker's decision (e.g. MRNA,
    # 488770.KS) had both fields silently dropped no matter what the model
    # answered. Don't revert to field-by-field "for clarity" — the point of
    # this shape is that a future shared field can't go missing here again.
    shared = flat_result.model_dump(exclude={"rating", "investment_thesis"})
    return PortfolioDecision(
        rating=PortfolioRating(flat_result.rating.value),
        # The only field that's transformed rather than passed through, so
        # it's excluded from the dump above and set explicitly here.
        investment_thesis=redact_injection_flag(flat_result.investment_thesis, agent_name="Portfolio Manager"),
        **shared,
    )
