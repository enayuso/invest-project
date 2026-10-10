"""Typed pipeline state.

Dataclasses hold internal orchestration bookkeeping (never shown to an LLM as
a schema). Pydantic models hold anything an LLM produces as structured output
directly — kept typed end to end, never rendered to markdown and re-parsed
downstream.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)

_NULLISH_FLOAT = {"", "none", "n/a", "na", "null", "nil", "-", "tbd", "unknown"}


def _coerce_optional_float(value):
    if isinstance(value, str) and value.strip().lower() in _NULLISH_FLOAT:
        return None
    return value


def _require_non_empty(v: str) -> str:
    # Observed live (2026-08-03, portfolio_manager): the model can return a
    # schema-valid response with a real decision/rating but blank required
    # text fields (raw content `{"executive_summary": "", ...}`) — pydantic
    # has no complaint since `str` allows "", so this sailed through as a
    # "successful" result with no visible reasoning. Raising here makes that
    # case indistinguishable from any other schema-validation failure, so it
    # hits structured_complete's existing empty/invalid-response
    # fallback+retry path (app.llm.client) instead of being stored as-is.
    if not v or not v.strip():
        raise ValueError("must not be empty")
    return v


# ---------------------------------------------------------------------------
# Debate state — shared by both the bull/bear stage (2 parties) and the risk
# debate stage (3 parties): a generic bounded multi-party loop.
# ---------------------------------------------------------------------------


@dataclass
class DebateTurn:
    speaker: str
    content: str
    round: int


@dataclass
class DebateState:
    parties: list[str]
    turns: list[DebateTurn] = field(default_factory=list)
    rounds_completed: int = 0
    converged: bool = False

    def latest_by(self, speaker: str) -> str | None:
        for turn in reversed(self.turns):
            if turn.speaker == speaker:
                return turn.content
        return None

    def latest_except(self, speaker: str) -> list[DebateTurn]:
        """Most recent turn from every other party — each debator sees all
        other live responses, not just the immediately-prior speaker."""
        latest: dict[str, DebateTurn] = {}
        for turn in self.turns:
            if turn.speaker != speaker:
                latest[turn.speaker] = turn
        return list(latest.values())

    def transcript(self) -> str:
        return "\n\n".join(f"[Round {t.round} — {t.speaker}]\n{t.content}" for t in self.turns)


# One speaker's turn (given the debate-so-far and its own party name, produce
# its next message) and the debate loop's per-round UX hook. Live here, not in
# orchestrator/debate.py, so agents/ (Strategy) can type against these without
# depending on orchestrator/ (Application) — see AI_development_principles.md's
# Dependency Rule.
# Strategy가 Application(orchestrator)에 의존하지 않도록 공용 타입을 state.py에 둠
SpeakerFn = Callable[[DebateState, str], Awaitable[str]]
RoundCallback = Callable[[int], Awaitable[None]]


class DebateConvergence(BaseModel):
    """Structured judgment call deciding whether a debate round added new
    substance. Deliberately a model judgment, not a keyword/similarity
    heuristic, so it stays correct across any input language."""

    converged: bool = Field(
        description=(
            "True if the latest round of responses did not introduce any "
            "materially new argument or rebuttal beyond what was already "
            "said in prior rounds. False if genuinely new substance was added."
        ),
    )
    reason: str = Field(description="One sentence justifying the verdict.")


# ---------------------------------------------------------------------------
# Research Manager
# ---------------------------------------------------------------------------


class PortfolioRating(str, Enum):
    BUY = "Buy"
    OVERWEIGHT = "Overweight"
    HOLD = "Hold"
    UNDERWEIGHT = "Underweight"
    SELL = "Sell"


def is_blank_card_reason(value: str | None) -> bool:
    """None, empty/whitespace-only, or a placeholder the model sometimes
    answers with instead of a real reason — all treated as "missing" since
    any of these rendered verbatim on the dashboard card would be worse
    than the fallback (signals.js `_cardReasonFor` only falls back to the
    other language / executive_summary on a falsy value, so a literal "N/A"
    string would display as-is). Shared by the card_reason retry in
    `app.agents.portfolio_manager` and the missing-card_reason warning in
    `app.orchestrator.pipeline` so the two can never drift apart."""
    if value is None:
        return True
    text = value.strip()
    return not text or text.lower() in {"n/a", "na", "none", "null", "-", "tbd"}


def is_state_flipping_rating(rating: "PortfolioRating | FlatPortfolioRating", held: bool) -> bool:
    """True only for the two transitions that change whether a position
    exists at all: flat -> entering (Buy while `held` is False) and held ->
    fully exiting (Sell while `held` is True). Overweight/Underweight adjust
    an already-open position and Hold changes nothing, so none of those flip
    the underlying state — only these two are gated by
    `app.orchestrator.pipeline._apply_hysteresis_gate`, and by the
    confidence-retry in `app.agents.portfolio_manager.decide` that shares
    this exact predicate so the two mechanisms can never drift apart."""
    return (not held and rating == FlatPortfolioRating.BUY) or (held and rating == PortfolioRating.SELL)


class _ResearchPlanFields(BaseModel):
    """Fields shared by `ResearchPlan` and `FlatResearchPlan` — every field
    except `recommendation` itself (app.state's shared-field-normalization
    convention, see `PortfolioDecision`/`FlatPortfolioDecision`'s own
    docstring for the 2026-08-19 commit 3b96efa incident this shape exists
    to prevent: a field listed by name at ONE normalization site but not the
    other silently drops for every decision that took the other path)."""
    rationale: str = Field(
        description=(
            "Summary of the key points from both sides of the debate, ending "
            "with which arguments led to the recommendation."
        ),
    )
    strategic_actions: str = Field(
        description="Concrete steps for the trader to implement the recommendation.",
    )

    @field_validator("rationale", "strategic_actions")
    @classmethod
    def _non_empty_text(cls, v: str) -> str:
        return _require_non_empty(v)


class ResearchPlan(_ResearchPlanFields):
    recommendation: PortfolioRating = Field(
        description=(
            "The investment recommendation. Exactly one of Buy / Overweight / "
            "Hold / Underweight / Sell. Each maps to a fixed target allocation the "
            "code applies: Buy=100%, Overweight=75%, Hold=no change, Underweight=25% "
            "(a real non-zero position, not 'mostly exit'), Sell=0%. A well-run "
            "bull/bear debate produces "
            "strong arguments on BOTH sides by design — that is not evidence "
            "of genuine balance, it's what a competent debate always looks "
            "like. Ask instead: even a slight lean, does one side's case hold "
            "up better under the rebuttals? If so, commit to that side (Buy/"
            "Overweight or Underweight/Sell) even on a narrow margin. Reserve "
            "Hold only for the rare case where you cannot form any lean at "
            "all after weighing the rebuttals — not merely because both "
            "sides argued well."
        ),
    )


class FlatResearchRecommendation(str, Enum):
    """Recommendations valid for a ticker with zero current shares —
    Overweight/Underweight/Sell all presuppose an existing position to
    adjust or exit, and this system is long-only (see the long-only mandate
    text in research_manager.SYSTEM_PROMPT), so a bearish conclusion on a
    ticker not held is Hold — 'stay out' — never Sell or a short. Real
    incident (2026-08-28 domestic AI audit, 45 decisions): the Research
    Manager recommended Sell — and for 010170.KS explicitly "initiate a
    short position" — on tickers held at zero shares in 33/45 runs, which
    the Portfolio Manager then had to silently translate every time (the
    same pattern `FlatPortfolioRating` above already fixed one node
    downstream of here)."""

    BUY = "Buy"
    HOLD = "Hold"


class FlatResearchPlan(_ResearchPlanFields):
    recommendation: FlatResearchRecommendation = Field(
        description=(
            "The investment recommendation for a ticker with NO current position. "
            "Exactly one of Buy / Hold. Commit to Buy on even a slight bullish lean "
            "after the rebuttals — a 55/45 lean is still a lean. Hold here means "
            "'stay out / do not enter', and is the correct answer for BOTH a "
            "genuinely unresolvable debate AND a bearish conclusion: with no shares "
            "held there is nothing to underweight or sell, and this system never "
            "opens short positions."
        ),
    )


# ---------------------------------------------------------------------------
# Trader
# ---------------------------------------------------------------------------


class TraderAction(str, Enum):
    BUY = "Buy"
    HOLD = "Hold"
    SELL = "Sell"


class _TraderProposalFields(BaseModel):
    """Fields shared by `TraderProposal` and `FlatTraderProposal` — every
    field except `action` itself, same shared-field-normalization shape as
    `_ResearchPlanFields`/`_PortfolioDecisionFields` (see either's docstring
    for the 2026-08-19 commit 3b96efa incident this exists to prevent)."""
    reasoning: str = Field(
        description=(
            "The case for this action, anchored in the analysts' reports and "
            "the research plan. Two to four sentences."
        ),
    )
    entry_price: float | None = Field(default=None, description="Optional entry price target.")
    stop_loss: float | None = Field(default=None, description="Optional stop-loss price.")
    position_sizing: str | None = Field(default=None, description="Optional sizing guidance.")

    @field_validator("entry_price", "stop_loss", mode="before")
    @classmethod
    def _nullish_float_to_none(cls, v):
        return _coerce_optional_float(v)

    @field_validator("reasoning")
    @classmethod
    def _non_empty_text(cls, v: str) -> str:
        return _require_non_empty(v)


class TraderProposal(_TraderProposalFields):
    action: TraderAction = Field(description="The transaction direction: Buy / Hold / Sell.")


class FlatTraderAction(str, Enum):
    """Actions valid with zero shares held: enter (Buy) or stay out (Hold).
    "Sell" would require shares to sell — proposing it on a flat ticker is
    either a no-op the Portfolio Manager must silently translate away (20 of
    45 decisions, 2026-08-28 domestic AI audit) or, read literally, a short
    sale this long-only/no-leverage system cannot execute (see the long-only
    mandate text in trader.SYSTEM_PROMPT)."""

    BUY = "Buy"
    HOLD = "Hold"


class FlatTraderProposal(_TraderProposalFields):
    action: FlatTraderAction = Field(
        description=(
            "The transaction direction for a ticker with NO current position: Buy "
            "(enter) or Hold (stay out). Sell is not available — there are no "
            "shares to sell and this system never opens short positions. If the "
            "plan is bearish, Hold is the correct, fully-committed answer."
        ),
    )


# ---------------------------------------------------------------------------
# Portfolio Manager
# ---------------------------------------------------------------------------

# Whitelisted leading-header markers _recover_fields_from_rendered_header
# (below) will strip from executive_summary and, where the model's own
# structured field for it is empty, back-fill. Deliberately excludes
# "**Rating**"/"**Executive Summary**" — those are handled by
# `_strip_rendered_header`'s narrower, no-backfill logic (rating is
# direction-deciding data, never inferred from prose) — and excludes any
# marker not observed in a real payload, so this stays a recovery of a
# known failure mode rather than a general-purpose header parser.
_HEADER_FIELD_MAP = {
    "Stop Loss": "stop_loss",
    "Price Target": "price_target",
    "Time Horizon": "time_horizon",
    "Confidence": "confidence",
}
_LEADING_HEADER_LINE = re.compile(
    r"^\*\*(" + "|".join(re.escape(k) for k in _HEADER_FIELD_MAP) + r")\*\*:[ \t]*([^\n]*)\n*"
)
# A single unambiguous price, optionally with a currency mark/code and
# thousands separators — a range ("18,000-19,000"), multiple numbers, or
# prose mixed in with the digits all fail to match and are left alone
# rather than guessed at.
_BACKFILL_PRICE = re.compile(r"^[₩$]?\s*(?P<int>\d{1,3}(?:,\d{3})*|\d+)(?P<dec>\.\d+)?\s*(?:KRW|USD|원)?$")


def _parse_backfill_price(raw: str) -> float | None:
    match = _BACKFILL_PRICE.match(raw.strip())
    if not match:
        return None
    value = float(match.group("int").replace(",", "") + (match.group("dec") or ""))
    return value if value > 0 else None


def _parse_backfill_confidence(raw: str) -> float | None:
    text = raw.strip()
    is_percent = text.endswith("%")
    text = text[:-1].strip() if is_percent else text
    if not re.match(r"^\d+(\.\d+)?$", text):
        return None
    value = float(text) / 100.0 if is_percent else float(text)
    return value if 0.0 <= value <= 1.0 else None


class _PortfolioDecisionFields(BaseModel):
    """Shared fields between `PortfolioDecision` (held position, all 5
    ratings) and `FlatPortfolioDecision` (flat position, Buy/Hold only) —
    only `rating`'s type differs between the two, so it's declared on each
    subclass instead of here. See `app.agents.portfolio_manager.decide`'s
    `held` parameter for why a ticker with zero shares gets a structurally
    narrower schema rather than just being asked nicely not to pick
    Overweight/Underweight/Sell (those require an existing position to mean
    anything, and a prompt-only instruction was previously ignored — see
    2026-07-22 bug: a flat ticker rated Underweight resulted in a BUY,
    because Underweight targets 25% of equity in sizing.py regardless of
    whether the model meant "reduce an existing position" or "stay near
    zero," and the flat case has no existing position to reduce)."""

    @model_validator(mode="before")
    @classmethod
    def _recover_fields_from_rendered_header(cls, data):
        """Observed live (decision id 950, 2026-08-21): the same failure
        mode `_strip_rendered_header` below already handles for "**Rating**"/
        "**Executive Summary**", but with different leading markers — AND,
        unlike that case, an actual data field left empty because of it.
        The model wrote "**Stop Loss**: 18,870\\n**Confidence**: 0.65\\n\\n"
        as the first two lines of executive_summary instead of (as well as)
        filling the structured `stop_loss` field, so the execution layer
        fell back to an ATR-floor stop instead of the AI's own number —
        this is a data-loss variant of the id=930 (2026-08-19) text-only
        doubling, with the leading marker changed.

        Recovers a KNOWN, whitelisted set of leading "**Field Name**:
        value" lines (`_HEADER_FIELD_MAP` — never "**Rating**", which
        `_strip_rendered_header` handles separately with NO backfill by
        design: rating is direction-deciding data, never inferred from
        prose) into their real fields, then strips those lines from
        `executive_summary` regardless of whether the value parsed —
        leftover header noise in the prose is worth cleaning up even when
        there's nothing to recover.

        Deliberately conservative in three ways: (1) only the leading run
        of recognized header lines is ever touched — `_LEADING_HEADER_LINE`
        anchors with `^`, so a sentence that happens to start with
        "Stop Loss: ..." after some other line is never reached, and the
        loop stops at the first non-matching line; (2) a field that
        already carries a real value from its own structured slot is
        never overwritten — recovery only ever fills a gap; (3) the value
        must parse as a single unambiguous number/percentage
        (`_parse_backfill_price`/`_parse_backfill_confidence`) — a range
        like "18,000-19,000", multiple numbers, or embedded prose all fail
        to parse and are left alone, so a still-`None` field afterward
        means either no header was found or the model's own text was too
        ambiguous to trust — both correctly fall through to the same ATR-
        floor/lowest-confidence defaults as before this validator existed.

        Every recovery is logged (WARNING, fixed message prefix) so a
        reviewer can grep for how often this fires — this is a safety-net
        for a real but rare (~1.3% of live decisions, measured 2026-08-21)
        failure mode, not a routine code path."""
        if not isinstance(data, dict):
            return data
        text = data.get("executive_summary")
        if not isinstance(text, str):
            return data

        remaining = text.lstrip()
        recovered: dict[str, float | str] = {}
        stripped_any = False
        while True:
            match = _LEADING_HEADER_LINE.match(remaining)
            if not match:
                break
            stripped_any = True
            key, raw_value = match.group(1), match.group(2)
            remaining = remaining[match.end() :]
            field_name = _HEADER_FIELD_MAP[key]
            if data.get(field_name) not in (None, ""):
                continue  # the model's own field already has a value -- never overwrite it
            if field_name == "confidence":
                parsed = _parse_backfill_confidence(raw_value)
            elif field_name == "time_horizon":
                parsed = raw_value.strip() or None
            else:
                parsed = _parse_backfill_price(raw_value)
            if parsed is not None:
                recovered[field_name] = parsed

        if not stripped_any:
            return data
        data = dict(data)
        data["executive_summary"] = remaining
        data.update(recovered)
        logger.warning(
            "state: recovered rendered-header field(s) from executive_summary: %s",
            recovered or "(stripped only, nothing parsed)",
        )
        return data

    executive_summary: str = Field(
        description=(
            "Concise action plan, exactly 3 sentences, in this fixed order: "
            "(1) the rating and its single main driver, (2) the key risk or catalyst that could "
            "change it, (3) the concrete action — entry/exit strategy, position sizing, and time "
            "horizon. Each sentence under ~25 words. This is shown verbatim on a dashboard card "
            "clamped to 6 lines — long or wandering prose gets cut off mid-sentence, so stay terse "
            "and do not pad with restated evidence already covered elsewhere in the decision."
        ),
    )
    investment_thesis: str = Field(
        description="Detailed reasoning anchored in specific evidence from the debates.",
    )
    # card_reason_ko / card_reason_en exist separately from executive_summary
    # because they answer a different question for a different surface:
    # executive_summary is the action plan (rating + risk + what to DO about
    # it) shown in the expanded chat view, while these are the short "why"
    # shown on the compact dashboard signal card — the concrete issue/data
    # behind the outlook, with no process talk and no action plan. Both
    # languages are stored because a decision is generated once (the
    # pipeline itself always runs in English — see app.llm.language) but
    # must display correctly in either UI language without a second call.
    card_reason_ko: str | None = Field(
        default=None,
        description=(
            "One to two short sentences, IN KOREAN regardless of this response's own language, "
            "stating the concrete issue or data driving this ticker's outlook — for a dashboard "
            "card, not a report. Cite the actual evidence: supply/demand flows, an earnings result, "
            "a specific news item or disclosure, a technical indicator level, or a valuation metric. "
            "Example style: '외국인 5거래일 연속 순매수, HBM 모멘텀 유효. 5일선 반등 신호 포착.' or "
            "'단기 과열 · RSI 74. 차익실현 매물 출회 가능성 높음.' Never describe the agents, the "
            "debate, or the synthesis process (e.g. do NOT write anything like 'the Aggressive "
            "case was validated' or 'the bull case prevailed'); never just restate the rating; and "
            "never include an action plan, position sizing, stop-loss, entry/exit, or time horizon "
            "— that belongs in executive_summary, not here."
        ),
    )
    card_reason_en: str | None = Field(
        default=None,
        description=(
            "One to two short sentences, IN ENGLISH, stating the concrete issue or data driving "
            "this ticker's outlook — for a dashboard card, not a report. Cite the actual evidence: "
            "supply/demand flows, an earnings result, a specific news item or disclosure, a "
            "technical indicator level, or a valuation metric. Example style: 'Foreign investors "
            "net-bought for 5 straight sessions; HBM momentum intact. 5-day MA showing a rebound "
            "signal.' or 'Short-term overbought, RSI 74. High risk of profit-taking supply.' Never "
            "describe the agents, the debate, or the synthesis process (e.g. do NOT write anything "
            "like 'the Aggressive case was validated' or 'the bull case prevailed'); never just "
            "restate the rating; and never include an action plan, position sizing, stop-loss, "
            "entry/exit, or time horizon — that belongs in executive_summary, not here."
        ),
    )
    price_target: float | None = Field(default=None, description="Optional target price.")
    time_horizon: str | None = Field(default=None, description="e.g. '3-6 months'.")
    stop_loss: float | None = Field(
        default=None,
        description=(
            "Stop-loss price for the position this rating implies. Set this whenever the "
            "rating implies holding or adding to a position (Buy/Overweight/Hold); the "
            "execution layer enforces it in code, so it must be a concrete, actionable price "
            "in the instrument's quote currency, not a vague level."
        ),
    )
    # Deliberately does NOT mention the hysteresis gate, the 0.6 flip
    # threshold, or that a high value lets a rating through — see
    # app.orchestrator.pipeline._apply_hysteresis_gate. A model that knew
    # omitting this field (or understating it) downgrades its own Sell/Buy
    # to Hold would have a direct incentive to state an artificially high
    # number purely to force its rating past that safety gate, which would
    # defeat the mechanism entirely. Only state the two facts a model needs
    # to fill this in honestly: that it applies to every rating including
    # Hold/Sell/Underweight, and that leaving it unset is scored as the
    # worst case rather than as neutral. Do not "helpfully" add the
    # mechanism back in here.
    confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Your conviction that THIS rating is correct, as a 0.0-1.0 fraction — separate from "
            "the rating itself (a Hold can be a 0.9-confidence call, a Buy can be a 0.5 one). This "
            "applies to EVERY rating, including Hold, Sell, and Underweight — a bearish or neutral "
            "call needs a stated conviction just as much as a bullish one does. The execution layer "
            "uses this only to scale position size DOWN, never up, so understating it costs nothing "
            "but a smaller trade, while overstating it risks a larger one than the evidence supports. "
            "Leaving this unset is not a neutral choice — an unset value is treated as the lowest "
            "possible confidence, so always state your honest estimate rather than omitting it."
        ),
    )

    @field_validator("price_target", "stop_loss", mode="before")
    @classmethod
    def _nullish_float_to_none(cls, v):
        return _coerce_optional_float(v)

    @field_validator("executive_summary", mode="before")
    @classmethod
    def _strip_rendered_header(cls, v):
        """Observed live (decision id 930, 2026-08-19): the model sometimes
        writes the whole rendered decision format INSIDE executive_summary
        (e.g. "**Rating**: Hold\n\n**Executive Summary**: **Rating**: Hold\n\n"
        "**Executive Summary**: \nThe Final Synthesis..."), and the renderer
        then wraps that again — a doubled/triple header in front of the
        actual prose. Strips only a LEADING "**Rating**: <...>" line and/or a
        leading "**Executive Summary**:" marker (repeated, since the real
        example above nests both twice), tolerant of a marker followed by
        newline instead of a space. Never touches either marker if it
        appears anywhere but the very start of the string, and never touches
        the "[Hysteresis gate: ...]" prefix — that one is added afterwards
        by pipeline._apply_hysteresis_gate via model_copy, which does not
        re-run validators, so it should never actually reach here; the
        pattern below only matches "**Rating**"/"**Executive Summary**"
        markers so it can't accidentally eat that prefix even if it did.
        mode="before" runs ahead of `_non_empty_text` below, so a field that
        was NOTHING BUT a header correctly ends up failing the non-empty
        check afterward instead of silently passing."""
        if not isinstance(v, str):
            return v
        text = v
        while True:
            candidate = text.lstrip()
            changed = False
            match = re.match(r"\*\*Rating\*\*:[^\n]*\n+", candidate)
            if match:
                candidate = candidate[match.end() :]
                changed = True
            match = re.match(r"\*\*Executive Summary\*\*:[ \t]*\n*", candidate)
            if match:
                candidate = candidate[match.end() :]
                changed = True
            if not changed:
                break
            text = candidate
        return text

    @field_validator("executive_summary", "investment_thesis")
    @classmethod
    def _non_empty_text(cls, v: str) -> str:
        return _require_non_empty(v)


class PortfolioDecision(_PortfolioDecisionFields):
    rating: PortfolioRating = Field(description="Final position rating.")


class FlatPortfolioRating(str, Enum):
    """Ratings valid for a ticker with zero current shares — Overweight/
    Underweight/Sell all presuppose an existing position to adjust or exit,
    which doesn't exist yet. See `_PortfolioDecisionFields`'s docstring."""

    BUY = "Buy"
    HOLD = "Hold"


class FlatPortfolioDecision(_PortfolioDecisionFields):
    rating: FlatPortfolioRating = Field(description="Final position rating — Buy or Hold only (no position to adjust yet).")


# ---------------------------------------------------------------------------
# Sentiment Analyst
# ---------------------------------------------------------------------------


class SentimentBand(str, Enum):
    BULLISH = "Bullish"
    MILDLY_BULLISH = "Mildly Bullish"
    NEUTRAL = "Neutral"
    MIXED = "Mixed"
    MILDLY_BEARISH = "Mildly Bearish"
    BEARISH = "Bearish"


class SentimentReport(BaseModel):
    overall_band: SentimentBand
    overall_score: float = Field(ge=0.0, le=10.0)
    confidence: Literal["low", "medium", "high"]
    narrative: str = Field(
        description="Source-by-source breakdown, divergences, themes, catalysts and risks.",
    )

    @field_validator("narrative")
    @classmethod
    def _non_empty_text(cls, v: str) -> str:
        return _require_non_empty(v)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Render helpers — turn a structured object into the prose downstream agents
# and the final report consume. The typed value stays the source of truth;
# rendering happens at the point of injection into a prompt, not before.
# ---------------------------------------------------------------------------


def render_sentiment_report(report: SentimentReport) -> str:
    return "\n".join([
        f"**Overall Sentiment**: {report.overall_band.value} (score {report.overall_score:.1f}/10, "
        f"confidence {report.confidence})",
        "",
        report.narrative,
    ])


def render_research_plan(plan: ResearchPlan) -> str:
    return "\n".join([
        f"**Recommendation**: {plan.recommendation.value}",
        "",
        f"**Rationale**: {plan.rationale}",
        "",
        f"**Strategic Actions**: {plan.strategic_actions}",
    ])


def render_trader_proposal(proposal: TraderProposal) -> str:
    parts = [f"**Action**: {proposal.action.value}", "", f"**Reasoning**: {proposal.reasoning}"]
    if proposal.entry_price is not None:
        parts.append(f"**Entry Price**: {proposal.entry_price}")
    if proposal.stop_loss is not None:
        parts.append(f"**Stop Loss**: {proposal.stop_loss}")
    if proposal.position_sizing:
        parts.append(f"**Position Sizing**: {proposal.position_sizing}")
    return "\n".join(parts)


def render_portfolio_decision(decision: PortfolioDecision) -> str:
    """Short by design — this is the default chat answer shown before the
    user expands the "view details" transcript. investment_thesis (the
    unbounded, often long field) deliberately isn't included here; it's
    stored in the transcript instead (see pipeline.py's `transcript` dict)
    so it's still reachable, just not part of the first thing shown.

    A hysteresis-gate downgrade (`app.orchestrator.pipeline.
    _apply_hysteresis_gate`) is still visible here despite that exclusion:
    that function prefixes `executive_summary` itself with a short note
    when it overrides the model's own Buy/Sell into Hold, precisely so this
    short view never shows a bare "Rating: Hold" next to prose still arguing
    the original case with no explanation — see that function's own
    docstring for the incident this fixes.

    card_reason_ko/card_reason_en are deliberately excluded here: they'd
    just duplicate executive_summary's content in two extra languages with
    no new information for this short view. They travel to the frontend via
    their own DB columns and `/api/decisions` keys instead, for the
    dashboard signal card specifically — do not "fix" this omission."""
    parts = [
        f"**Rating**: {decision.rating.value}",
        "",
        f"**Executive Summary**: {decision.executive_summary}",
    ]
    if decision.price_target is not None:
        parts.append(f"**Price Target**: {decision.price_target}")
    if decision.time_horizon:
        parts.append(f"**Time Horizon**: {decision.time_horizon}")
    if decision.stop_loss is not None:
        parts.append(f"**Stop Loss**: {decision.stop_loss}")
    if decision.confidence is not None:
        parts.append(f"**Confidence**: {decision.confidence:.0%}")
    return "\n".join(parts)


@dataclass
class PipelineState:
    ticker: str
    trade_date: date
    instrument_context: str = ""
    reports: dict[str, str] = field(default_factory=dict)
    research_debate: DebateState = field(default_factory=lambda: DebateState(parties=["Bull", "Bear"]))
    research_plan: ResearchPlan | None = None
    trader_proposal: TraderProposal | None = None
    risk_debate: DebateState = field(
        default_factory=lambda: DebateState(parties=["Aggressive", "Conservative", "Neutral"])
    )
    past_context: str = ""  # Postgres+pgvector similarity retrieval, PM-only — wired in Phase 3
    # A same-day, same-ticker decision made earlier THIS cycle-day — status-
    # agnostic (unlike past_context, which only surfaces 'resolved' rows),
    # so a still-pending sibling's reasoning is visible too. Deliberately a
    # separate field, not folded into past_context: (a) app.experiments.
    # language_ab's isolation_violation check asserts on past_context being
    # empty, and this is populated far more often (any same-day re-analysis)
    # than a genuine cross-run memory leak would be; (b) it needs its own
    # framing in the prompt ("no outcome yet") so the LLM doesn't mistake it
    # for a graded lesson. See app.memory.store.get_same_day_sibling_context
    # and the id=949 (950160.KS) incident this closes the gap on. Live-only
    # (always "" in a backtest — see that function's own as_of guard).
    sibling_context: str = ""
    final_decision: PortfolioDecision | None = None
    decision_id: int | None = None  # memory_entries.id once stored — lets a caller fetch the full transcript later
    rebalance_adjustment_note: str | None = None  # RebalanceResult.adjustment_note, if the risk caps changed the outcome
    # "AnalystName: 'offending name'" per analyst report app.agents.identity_gate
    # forced a correction banner onto this cycle — empty when clean (the common
    # case) or when reports came from the 15-minute cache (a cached run was
    # necessarily clean, since a violating run is never cached).
    identity_violations: list[str] = field(default_factory=list)
    # "AnalystName: unnamed" per report that never stated ANY company name
    # (app.agents.identity_gate Detector B -> UNNAMED). Split out from
    # identity_violations 2026-09-01: an omission is a different failure from
    # a wrong-company substitution (the 2026-08-20 downstream damage needs a
    # wrong name to exist), it gets a neutral note not a correction banner,
    # and — unlike identity_violations — it does NOT gate the analyst-report
    # cache (a legitimately name-free pure-macro report shouldn't cost every
    # .KS/backtest decision its 15-minute report reuse).
    # 회사명 미기재 ≠ 오표기 — 캐시를 막지 않음.
    identity_unnamed: list[str] = field(default_factory=list)
    # Analyst names whose report failed this cycle and was replaced with a
    # placeholder note (app.orchestrator.pipeline's analyst gather uses
    # return_exceptions=True and degrades rather than aborting — see its own
    # comment). Empty on a full 4-analyst run and on any cache hit (a
    # degraded run is never cached — see identity_violations above for the
    # same reasoning). Exists because nothing distinguished a 1-3-analyst
    # decision from a full 4-analyst one — app.audit's win-rate/confidence-
    # calibration numbers pooled them (2026-08-25).
    degraded_analysts: list[str] = field(default_factory=list)
    # "kind: 'written' -> 'expected'" per app.agents.unit_scale_guard
    # correction applied to the Fundamentals Analyst's report this cycle —
    # empty when clean (the common case) or on a cache hit (same
    # never-cache-a-violating-run reasoning as identity_violations above).
    # 2026-08-28 domestic AI audit: a 억원-to-billion/trillion 10x mislabel,
    # distinct in kind from identity_violations (a wrong number, not a wrong
    # company name) though the same incident showed one causing the other.
    # Also carries app.agents.ratio_label_guard's `debt_ratio_denominator`
    # entries (2026-09-01 domestic AI audit, item #4 — a debt-ratio
    # denominator mislabel, e.g. KIS's 부채비율 described as
    # "debt-to-assets"). Field name kept as-is deliberately — it's not
    # renamed just because a second guard now feeds it; same out-param, same
    # cache-blocking treatment.
    unit_scale_violations: list[str] = field(default_factory=list)
    # The POSITIVE scan record app.agents.unit_scale_guard now emits alongside
    # the violation list — `{node: {candidates_examined, ground_truth_values,
    # corrected, flagged, unverified}}`. 2026-09-01, decision id=1158
    # (000270.KS/기아): `unit_scale_violations=[]` was recorded on a report
    # with NINE 10x-mislabeled 억원 figures the regex was blind to, and the
    # empty list read as "clean". A node here with `ground_truth_values > 0
    # and candidates_examined > 0 and corrected == 0 and
    # unverified == candidates_examined` is the id=1158 shape — a suspected
    # blind spot, not a clean report. NOT folded into unit_scale_violations
    # (that field gates the report cache; a scan record is not a violation).
    unit_scale_scans: dict[str, dict[str, int]] = field(default_factory=dict)
    # "round N Speaker: 'span'" per risk-debate turn that declared a rating
    # the Portfolio Manager cannot hold for this ticker — always empty for a
    # held ticker (all five levels legal) and empty in the clean case. See
    # app.agents.risk_debators.check_illegal_ratings. 2026-09-01, id=1157's
    # authoritative Neutral synthesis was "CONCLUSION: Underweight (25%)" on a
    # FLAT ticker — discarded silently by the PM, unrecorded until now.
    # Recorded (not just logged) for the same reason identity_violations is:
    # only a stored field lets a batch audit ask "how often does the debate
    # hand the PM a rating it must throw away". NOT cache-gating — a debate is
    # regenerated every cycle.
    risk_rating_violations: list[str] = field(default_factory=list)
    # The Neutral debator's round-1 vs final-round verdict, parsed from its own
    # prose — `{}` when there is no comparable pair. See
    # app.agents.risk_debators.check_neutral_reversal for the shape and for why
    # this is detect-only. 2026-09-04, id=1212/1217: nothing recorded the r1
    # verdict, so "did f79213e's no-reframing-reversals clause hold?" could only
    # be answered by re-reading transcripts by hand. `reversed: True` with
    # `new_fact_cited: False` is the id=1212 shape; a `None` in `r1`/`final`
    # means that turn's verdict did not parse (~9% of live Neutral turns) and
    # must NOT be read as agreement. NOT cache-gating — a debate is regenerated
    # every cycle (same reasoning as risk_rating_violations above).
    neutral_reversal: dict = field(default_factory=dict)
    # "AgentName: '<span>' vs CONFIRMED-clear broker status" per present-tense
    # halt/suspension claim found in a report/debate turn/PM output that
    # contradicts a genuinely CONFIRMED-clear execution_status_context this
    # same cycle — see app.agents.halt_status_guard.scan_halt_claims for the
    # exact scope and the four legitimate framings (dated/warning/lifted/
    # future) it deliberately does NOT flag. Empty in the overwhelming common
    # case (2026-09-07, 21-day measurement: only 5/163 clean-status domestic
    # decisions ever mentioned a halt at all, 4/5 framed correctly) and
    # ALWAYS empty for an overseas ticker, a backtest, or any cycle whose
    # execution_status_context wasn't confirmed-clear — this is not a
    # positive/negative scan pair like unit_scale_scans, since there is no
    # "unscannable" case worth recording separately (no ground truth = no
    # attempt, full stop). 2026-09-07, id=1238 (351320.KS): a stale news
    # headline's false "currently halted" claim propagated uncorrected through
    # the Research Manager, the risk debate, and the PM's final decision text
    # with nothing recording it. Detect-only — computed too late in the
    # pipeline (after the PM's own decide() call) to affect anything, and NOT
    # cache-gating for that same reason.
    halt_status_violations: list[str] = field(default_factory=list)
    # "AgentName: [<family>] '<span>' vs SESSION-IN-PROGRESS ground truth" per
    # closed-session/completed-candle assertion found in a report/debate
    # turn/PM output that contradicts a genuinely still-open session this
    # cycle — see app.agents.session_close_guard.scan_session_close_claims.
    # 2026-09-09, id=1358 (047040.KS): the risk debate's round-2 turns wrote
    # "closed near the session low ... textbook distribution pattern" at
    # 10:16 KST, 46 minutes into the session — the stated #1 reason the
    # Neutral verdict flipped Underweight -> Sell, which executed against a
    # real position minutes later. Same "prompt fix leaking, not absent"
    # relationship halt_status_violations/price_limit_violations have to
    # their own prompt fixes (session_state.intraday_narration_note here).
    # Detect-only — no control-flow change, computed too late in the
    # pipeline to affect anything, and NOT cache-gating for that reason.
    session_close_violations: list[str] = field(default_factory=list)
    # "AgentName: claimed desk unanimity ... vs RM=<rm>/Trader=<trader>/
    # PM=<rating> — '<span>'" per false claim, in the PM's own final text,
    # that the desk was unanimous — see
    # app.agents.desk_consensus_narrative_guard.scan_desk_narrative. 2026-09-09,
    # id=1352 (028050.KS/삼성E&A): `_desk_consensus_note` told the PM verbatim
    # that the Research Manager concluded Underweight and the Trader proposed
    # Sell, and the PM's executive_summary wrote "The desk unanimously
    # recommends HOLD, and I concur" — the held 2 shares this bearish desk
    # wanted reduced were never actually contested in the record a human or a
    # later audit would read. `[]` whenever either upstream object was
    # missing (no desk-consensus block existed this cycle to misdescribe) —
    # same contract as `_desk_consensus_note`'s own `""`. Detect-only — no
    # control-flow change, computed too late in the pipeline to affect
    # anything, and NOT cache-gating for that reason.
    desk_narrative_violations: list[str] = field(default_factory=list)
    # "AgentName: sizing cap cited as exit rationale — '<span>'" per
    # risk-debate turn or PM final-text sentence that names the added-share
    # position-sizing cap (or its ATR/volatility scaling) and draws an
    # exit-or-reduce conclusion from it for a position that is already held
    # — see app.agents.sizing_cap_misuse_guard.scan_sizing_cap_misuse.
    # 2026-09-09, id=1358 (047040.KS/대우건설): the risk debate's Neutral
    # turn wrote "Since the position size is effectively negligible ...
    # (due to the risk cap) ... the optimal choice is to remove the exposure
    # entirely" and the PM's thesis "the 8.0% ATR compresses any theoretical
    # position to a trivial share count ... Exiting eliminates exposure" —
    # a real Sell of 5 held shares followed, with no stop-loss or drawdown
    # breaker forcing it. The prompt says in five places this cap "only ever
    # reduces shares being ADDED, never forces a sale of shares already
    # held". `[]` whenever the ticker is flat (`held` False) — with nothing
    # on the book the cap cannot be misused to liquidate anything. Step 3 of
    # the P0-2/3 plan: the sentence-level counter a later retrospective
    # calibration reads before the `unjustified_exit_gate` (default OFF) is
    # switched on. Detect-only — no control-flow change, NOT cache-gating
    # (same convention as desk_narrative_violations/halt_status_violations
    # above).
    sizing_cap_misuse_violations: list[str] = field(default_factory=list)
    # "AgentName: ..." per hedged/ungrounded technical-indicator claim (RSI,
    # moving-average crossover) or KOSPI/KOSDAQ index-contribution
    # misattribution found in the Market Analyst's own report — see
    # app.agents.technical_claim_guard.scan_technical_claims. 2026-09-07,
    # 21-day live measurement backing this: 91/255 domestic Market Analyst
    # reports had no real indicator value at all (the model skipped
    # get_indicators — Fix 1 below addresses this directly), of which 6/255
    # were papered over with a fabricated hedged RSI value and ~5/255 with a
    # hedged moving-average claim; separately 1/521 reports misattributed the
    # KOSPI index's own day-change as this stock's "contribution" to it
    # (id=1237). Detect-only — no control-flow change, NOT cache-gating.
    technical_claim_violations: list[str] = field(default_factory=list)
    # "AgentName: refusal_total|refusal_hedged — '<span>'" per "as an AI
    # model I have no real-time data / that date is past my training
    # cut-off" refusal preamble found in one of the four analysts' own
    # reports — see app.agents.refusal_guard.scan_refusal. 2026-09-14
    # measurement over 1,008 stored decisions: Fundamentals Analyst 21/1008
    # (2.1%; 1.5% live, 13.9% backtest, one backtest week 61.5%), Market
    # Analyst 1, News + Macro Analyst 1 (LIVE), Sentiment Analyst 0. The
    # DATE_GROUNDING system-prompt clause has told the model not to do this
    # since 2026-08-04 and it still happens, so this records the residual
    # rate rather than re-attempting a prompt fix. The tier is a LABEL only —
    # nothing branches on it. Detect-only — no control-flow change, NOT
    # cache-gating, and deliberately NOT folded into degraded_analysts /
    # analysts_failed (same convention as technical_claim_violations above).
    refusal_violations: list[str] = field(default_factory=list)
    # Whether this cycle's Market Analyst indicator block carried a REAL
    # get_indicators value (True) vs the "NOT AVAILABLE" fallback (False) —
    # the positive-signal counterpart to technical_claim_violations, same
    # "empty violations is not proof of a clean report" lesson
    # unit_scale_scans/period_mismatch_scans already encode: an empty
    # technical_claim_violations here could mean either "grounded and clean"
    # or "nothing was checked because history was too thin this cycle" —
    # this field disambiguates the two. Defaults False (unknown/not computed)
    # for a cached-report cycle, same as the other guard fields above.
    indicator_block_available: bool = False
    # Whether this cycle's Market Analyst investor-flow (수급) block carried a
    # REAL get_investor_flow value (True) vs the "NOT AVAILABLE" fallback
    # (False) — the investor-flow analogue of indicator_block_available above
    # (2026-09-11 fix, id=1389/320000.KS: the model wrote a fabricated,
    # hedged flow figure with no real tool call backing it). Same "empty
    # violations is not proof of a clean report" caveat and same defaults-
    # False-for-a-cached-cycle treatment as indicator_block_available.
    flow_block_available: bool = False
    # Whether this cycle's Market Analyst long-horizon (52-week weekly-bar)
    # block carried a REAL get_long_horizon_context value (True) vs the
    # "NOT AVAILABLE" fallback (False) — the long-horizon analogue of
    # indicator_block_available / flow_block_available above (2026-09-16,
    # engine audit item 1: the daily table only ever covered 10 sessions
    # while the prompt asked for a 52-week read). Same "empty violations is
    # not proof of a clean report" caveat and the same defaults-False-for-a-
    # cached-cycle treatment as the two fields above.
    long_horizon_block_available: bool = False
    # "AgentName: '<span>' — business/industry description inconsistent with
    # registered name '<company_name>'" per fabricated business/industry
    # self-description found in an analyst's own report AFTER identity_gate
    # already flagged that report for a wrong company name this cycle — see
    # app.agents.business_description_guard.scan_business_description.
    # 2026-09-07, id=1237 (294630.KS/서남): identity_gate's correction banner
    # for a mangled name ("Seonam Pharmaceutical Co., Ltd.") used to claim
    # the report's CONTENT was still trustworthy ("disregard the wrong name
    # only, not the content") — it wasn't: the same report's Risk Factors
    # section then said "As a pharmaceutical firm..." contradicting its own
    # correct sector classification stated elsewhere in the same document.
    # 21-day live measurement backing this calibration: 1/259 domestic
    # decisions (0.39%), with 6/7 similar self-descriptions elsewhere being
    # correct — a small, detect-only scope, not a gate. Detect-only — no
    # control-flow change, NOT cache-gating (same convention as
    # technical_claim_violations/halt_status_violations above).
    business_description_violations: list[str] = field(default_factory=list)
    # "AgentName: claimed <N>% same-session move exceeds KRX's ±30% daily
    # limit — '<span>'" per claimed same-session/single-day KRW price move
    # whose magnitude exceeds KRX's daily price-limit band, found in a
    # report/research-plan/risk-debate turn/PM output — see
    # app.agents.price_limit_guard.scan_price_limit_claims. 2026-09-08,
    # id=1313 (082850.KS): the Market Analyst invented a full "session high
    # 10,000 KRW -> low 5,000 KRW, a 50% drawdown... settling at 6,170 KRW"
    # narrative (no VERIFIED SNAPSHOT line anywhere in that report) that
    # propagated uncorrected into the risk debate and the PM's own executive
    # summary. Calibrated against 800 real analyst reports (last 200
    # non-backtest KRX decisions): 6 candidates above threshold, all
    # genuine; zero false positives among 187 sub-threshold matches. Only
    # ever populated for a KRX-listed (`.KS`) ticker — the ±30% band is
    # KRX-specific. Detect-only — no control-flow change, NOT cache-gating
    # (same convention as technical_claim_violations/halt_status_violations
    # above).
    price_limit_violations: list[str] = field(default_factory=list)
    # "AgentName: claimed 'Verified' price <N> vs real verified snapshot
    # <M> — '<span>'" per self-labeled "Verified Close/Price/Snapshot" claim
    # that contradicts the real verified snapshot this cycle — see
    # app.agents.verified_price_guard.scan_verified_price_claims. 2026-09-08,
    # id=1314 (033790.KS/피노): the News + Macro Analyst self-labeled a
    # fabricated ~1,300 KRW price universe "Verified" while the real
    # verified snapshot (also handed to it, and correctly used by the
    # Market/Fundamentals Analysts the same cycle) was ~9,120-9,760 KRW —
    # roughly a 7x gap. Calibrated against 250 live domestic decisions: 22
    # comparable "Verified ...: ₩N" claims, 20 matched the real snapshot
    # exactly, only id=1314's two spans mismatched. Detect-only — no
    # control-flow change, NOT cache-gating (same convention as
    # price_limit_violations/halt_status_violations above).
    verified_price_violations: list[str] = field(default_factory=list)
    # The desk-consensus override gate's structured record (2026-09-04,
    # id=1212/1217 — app.orchestrator.pipeline._apply_consensus_override_gate).
    # `{}` when the gate did not apply (the common case — the PM did not
    # increase exposure against a Hold/Sell Trader and a Hold/Underweight/Sell
    # Research Manager). When populated: {"rm": str, "trader": str,
    # "pm_rating": str, "confidence": float|None, "tier": "bearish"|"neutral",
    # "threshold": float, "gated": bool} — `gated` False means the override
    # cleared the higher confidence bar and stands, True means it was
    # downgraded to Hold. Recorded (not just logged) so a batch audit can ask
    # "how often does the PM override the whole desk, and how often does that
    # survive the gate". NOT cache-gating — a fresh decision every cycle.
    consensus_override: dict = field(default_factory=dict)
    # The unjustified-exit gate's structured record (2026-09-09, P0-2/3,
    # id=1358 — app.orchestrator.pipeline._apply_unjustified_exit_gate).
    # `{}` when the gate did not apply (the common case — not a held Sell, or
    # `settings.unjustified_exit_gate_enabled` is off, or a hard exception
    # applied: backtest, drawdown breaker, execution block, or a stop-loss/
    # reversal on the book). When populated: {"tier":
    # "narrative"|"mechanics"|"both", "rm": str, "trader": str,
    # "current_weight": float|None, "underweight_target": float|None,
    # "misuse_spans": list[str], "gated": bool} — `gated` True means the Sell
    # was downgraded to Hold, False means the tier did not reach the firing
    # bar (currently "both") and the Sell stands. Recorded even when not
    # gated so the retrospective calibration can count how often each tier
    # fires alone. NOT cache-gating — a fresh decision every cycle.
    unjustified_exit_gate: dict = field(default_factory=dict)
    # Two event families, namespaced by entry prefix — see
    # app.llm.metrics.track_run_sanitizations for the full shape list:
    #   - "label: kind" per reasoning leak app.llm.client._strip_reasoning_leak
    #     removed from an LLM response body this run (e.g.
    #     "risk_conservative: think_block"). 2026-09-01, id=1156: a
    #     risk-debate turn published ~330 lines of its own deliberation and
    #     it changed the decision. `untagged_suspected` / `declined_too_short`
    #     entries here are NOT strips — see that function.
    #   - "label: injection_banner:<tier>" per
    #     app.agents.untrusted_content.redact_injection_flag classification
    #     of the text after an INJECTION_SUSPECTED banner. 2026-09-01, root
    #     cause #9, id=1157/id=1158: a risk-debate turn's own self-reversed
    #     verdict (never stripped, only annotated) was the PM's only cited
    #     argument for a Buy. Empty in the common case.
    reasoning_sanitizations: list[str] = field(default_factory=list)
    # `{"agent", "tier", "reason"}` per report that opened with an
    # INJECTION_SUSPECTED line — `reason` is the model's own one-sentence
    # explanation (truncated). Pairs 1:1 with the `injection_banner:<tier>`
    # entry in `reasoning_sanitizations` above (same agent, same terminal
    # tier: empty / rest_kept / rest_leak_suspected / rest_self_reversal);
    # this field only adds the reason. Exists because that reason used to be
    # log-only — the 2026-09-04 injection-banner false-positive investigation
    # had to reconstruct every reason by hand-reading transcripts. Kept
    # separate from `reasoning_sanitizations` so free-text prose cannot break
    # that list's `"node: kind"` parseability. NOT cache-gating.
    # 인젝션 자기신고 사유 저장 — 배치 감사가 SQL로 조회 가능하게.
    injection_flags: list[dict] = field(default_factory=list)
    # True when this decision was made while the ticker's own market's
    # regular session was still open (app.dataflows.session_state.
    # session_in_progress). 2026-09-01, id=1157: the "session still open"
    # warning reached only the analysts, and the PM's decision_text / signal
    # card re-narrated a mid-session snapshot as a closed daily bar. The
    # downstream prompts now carry an intraday note; this flag lets a batch
    # audit pull just the intraday decisions to check their prose — a
    # regex-free alternative to a noisy post-hoc "closed"/"종가" scanner.
    session_in_progress: bool = False
    # "node: period_mismatch: '<span>' matches ..." per report percentage
    # that looks like a Korean interim-vs-full-year mis-division framed as a
    # decline — 2026-09-01, id=1158: H1 2026 operating profit divided by
    # FY2024's annual figure and called a "62% collapse". Detect-only (see
    # app.agents.period_mismatch_guard); empty in the common case and always
    # empty for a non-KRX ticker. Recorded here in FULL (both tiers), but as
    # of 2026-09-03 only the Tier-1 (strong) subset gates the report cache —
    # a Fundamentals-Analyst finding that also carries an explicit period
    # frame. A Tier-2 finding (any other node, or no period frame) is
    # structurally an echo of an upstream finding or a coincidence, so it is
    # recorded but no longer cache-blocking (mirrors unit_scale_guard's
    # Tier-1 autocorrect / Tier-2 flag-only split).
    period_mismatch_violations: list[str] = field(default_factory=list)
    # The POSITIVE scan record app.agents.period_mismatch_guard emits per
    # node alongside the finding list — `{node: {kr_block_present,
    # interim_period, fy_rows, forbidden_in_band, pct_examined, tol_matched,
    # strong, weak, suppressed_*}}`. `forbidden_in_band == 0` means the report
    # was unscannable (no interim row, no FY rows, or every candidate ratio
    # out of band / a placeholder), NOT that it is clean — same
    # empty-list-is-not-evidence caveat as unit_scale_scans (id=1158). NOT
    # cache-gating (a scan record is not a violation).
    period_mismatch_scans: dict = field(default_factory=dict)
    # 2026-09-07, id=1237 (294630.KS). How current get_kr_financials' newest
    # exchange-reported period is: `{latest_period, latest_label,
    # months_stale, interim}`. KIS's annual series for that issuer ended at
    # FY2024 — 20 months before the analysis date — with no staleness signal
    # anywhere, and the desk read the 2024 figures as the company's current
    # condition (audit #6). `_staleness_note` warns downstream past
    # settings.kr_financials_staleness_warn_months; this is the raw record so
    # a batch audit can count, in SQL, how many domestic decisions ran on
    # financials a full fiscal year or more out of date. `{}` for a US ticker
    # or a cache hit (no fetch happened — unknown, not "fresh").
    kr_financials_recency: dict = field(default_factory=dict)
    # 2026-09-07, id=1237 (294630.KS). Per-decision roll-up of
    # unit_scale_guard's news-headline 억-amount check:
    # `{headline_amounts, flagged_nodes: {node: n}, total_flags}`. A KIS
    # headline's "6.6억원" contract was reported by the desk as "₩6.6 billion"
    # (10x) and that drove a bear turn (audit #3). The guard is flag-only
    # (headline text is untrusted); this roll-up is the measurement that must
    # come before the news_eok_10x band is ever tightened toward
    # auto-correction. `{}` for a US ticker or a cache hit.
    news_amount_scans: dict = field(default_factory=dict)
    # 2026-09-02, id=1169 (950260.KS). Whether this decision had a usable
    # 14-day ATR, and what the fail-closed new-entry gate did about it.
    # Recorded, not just logged, because the absence of ATR is invisible in
    # every other stored field yet silently disables BOTH the volatility
    # scale-down and the stop-loss floor. `outcome`: "ok" (ATR present) |
    # "blocked_new_entry" (flat + gate on) | "held_unscaled" (position held,
    # sizing/stop unvalidated but allowed) | "gate_disabled" (no ATR, gate
    # switched off). `atr_pct` is the raw value, so 0.0 is distinguishable
    # from None.
    atr_gate: dict[str, object] = field(default_factory=dict)
    # "AgentName: '<span>' — presents N-month-stale <period> data as current,
    # with no staleness acknowledgment anywhere in the report" per Fundamentals
    # Analyst report that opens with an "As of <calendar date>, <Company>
    # presents/shows/..." framing while the underlying financials are stale
    # (see app.dataflows.kis_provider.kr_financials_recency's months_stale) and
    # nowhere in the report acknowledges the data's true, older period.
    # 2026-09-09 live audit: id=1351 (198440.KS, 21-month-stale FY2024 data)
    # opened "As of September 9, 2026, Kangdong C&L presents a complex
    # fundamental picture ..." with zero staleness disclosure anywhere —
    # despite the injected FINANCIAL-DATA RECENCY warning already telling the
    # model never to present the data as current. Contrast id=1353, same
    # 21-month staleness, whose report explicitly said "**Source Data
    # Period:** Fiscal Year Ended December 31, 2024" and "based solely on the
    # provided financial data ... for the period ending December 31, 2024" —
    # that one must never be flagged, and isn't. Detect-only — a
    # narrative/decision-quality check (not a numeric ground-truth guard), no
    # control-flow change, NOT cache-gating (same convention as
    # technical_claim_violations/halt_status_violations above).
    staleness_narrative_violations: list[str] = field(default_factory=list)
    # "AgentName: '<span>' — <label>" per sentence reframing MISSING data
    # (insider selling/buying, investor net-buy flow, analyst/institutional
    # coverage) as a specific positive/reassuring signal — see
    # app.agents.absence_reframing_guard.scan_absence_reframing. 2026-09-09
    # domestic AI-decision audit, same 12-decision sample as
    # halt_status_violations/staleness_narrative_violations above (ids
    # 1349/1350/1351/1352/1358/1359): "the absence of insider selling
    # suggests management does not believe the current price represents fair
    # value" (id=1349), "the price action itself serves as the primary proxy
    # for sentiment" for missing flow data (id=1350), "[lack of coverage] is
    # a bullish point for short-term traders" (id=1351). Multi-node like
    # halt_status_violations (scans every analyst report, the research plan,
    # the risk debate, and the PM's own final decision text) — NOT scoped to
    # a single analyst like staleness_narrative_violations. Detect-only, no
    # control-flow change, NOT cache-gating (same convention as
    # technical_claim_violations/halt_status_violations above).
    absence_reframing_violations: list[str] = field(default_factory=list)
    # "AgentName: <description> — '<span>'" per stated DERIVED numeric value
    # (PER, stop-loss risk distance %) found inconsistent with its own two
    # nearby GROUND-value citations — see
    # app.agents.derived_value_mismatch_guard.scan_derived_value_mismatches.
    # 2026-09-09 P1 audit item ⑤ ("산술 오류 다발" / arithmetic errors), same
    # 12-decision sample as absence_reframing_violations above: id=1357
    # (032820.KS) stated PER ~2,728x, but its own cited EPS (₩62) and price
    # (~₩13,640) compute to 220.0x; id=1353 (088280.KS) stated a ~3.5%
    # stop-loss risk distance, but its own cited stop (2,400 KRW) and
    # current price (2,815 KRW) compute to 14.7%. Unlike every guard above
    # (a phrasing-pattern check), this is a NUMERIC-DERIVATION check: it
    # recomputes the derived value from the two ground values and compares.
    # Multi-node like halt_status_violations (scans every analyst report,
    # the research plan, the risk debate, and the PM's own final decision
    # text). Detect-only, no control-flow change, NOT cache-gating.
    derived_value_mismatch_violations: list[str] = field(default_factory=list)
    # "AgentName: '<span>' — ungrounded automatic <consequence> claim" per
    # fabricated automatic regulatory/exchange consequence (forced
    # liquidation, delisting, index exclusion, trading halt/suspension,
    # margin/credit-trading ban, institutional-mandate divestment) — see
    # app.agents.regulatory_mechanism_guard.scan_regulatory_mechanism.
    # id=1380: a confirmed delinquent-filer determination was narrated as
    # mechanically triggering "automated compliance protocols: trading
    # halts, exclusion from benchmark indices, and forced liquidation by
    # institutional funds" — none of which was grounded in any data
    # actually supplied this cycle. A sibling of halt_status_violations, not
    # an extension of it: that guard catches a PRESENT-TENSE "is halted
    # right now" claim contradicted by a CONFIRMED-clear broker read; this
    # one catches an AUTOMATIC-CONSEQUENCE causal claim ("triggers"/
    # "automatically"/"mandates" ...) regardless of tense, deferring to
    # halt_status_violations for the halt/suspension family specifically
    # when execution_status_context was supplied this cycle. Full-coverage
    # sweep (every analyst report, the research plan, both debates, the
    # trader proposal, and the PM's own final decision text) — this is a
    # fresh guard with no legacy coverage gap to preserve. Detect-only, no
    # control-flow change, NOT cache-gating.
    regulatory_mechanism_violations: list[str] = field(default_factory=list)
    # "AgentName: <label> claimed as <N> vs supplied ground truth <M>
    # [— false correction] — '<span>'" per numeric claim (PER/PBR or
    # program-trading net-buy) either explicitly attributed to the pipeline's
    # own supplied ground truth (kr_gt / kr_risk_extra) or sitting inside a
    # correction/accusation frame ("fabrication", "tenfold", "10x", "조작",
    # "오류") whose value materially disagrees with that ground truth in
    # EITHER direction — see app.agents.ground_truth_claim_guard.
    # scan_ground_truth_claims. Built for id=1394 (PBR 0.55 attributed-as-
    # 2.44x) and id=1386 (006360.KS: a debater falsely accused a CORRECT
    # "+15.65 billion KRW" program-trading figure of being a "fabrication"
    # and "corrected" it to a WRONG "+1.565 billion KRW" — adopted by both
    # the Research Manager and PM as decisive reasoning). Full-coverage
    # sweep, same as regulatory_mechanism_violations above. Detect-only, no
    # control-flow change, NOT cache-gating.
    ground_truth_claim_violations: list[str] = field(default_factory=list)
    # 2026-09-09 P1 audit item #1 (id=1350/1359): the raw
    # app.orchestrator.pipeline.get_kr_risk_signals() text (short-selling
    # intensity, program-trade net-buy) fed to the risk debators as part of
    # `risk_extra` — previously computed and used but never recorded, so an
    # audit reading a stored transcript could not tell whether a debate
    # turn's "program trading shows net selling of N shares" cited a real
    # KIS feed value or was fabricated; it had to guess from prose
    # consistency alone. "" for a non-.KS ticker, a cache hit, or when the
    # feed call itself failed/returned nothing. Purely observational — does
    # not change any prompt or gate.
    kr_risk_signals: str = ""
    # Whether the shared PM-narrative-contradiction notice was appended to
    # `final_decision.investment_thesis` this cycle (True) vs left untouched
    # (False) — the positive-signal counterpart to the ten sweep-guard
    # `*_violations` fields above (session_close, regulatory_mechanism,
    # ground_truth_claim, desk_narrative, sizing_cap_misuse, price_limit,
    # verified_price, absence_reframing, derived_value_mismatch,
    # halt_status): every one of those guards is detect-only, so a PM-
    # attributed finding used to be logged and persisted into the transcript
    # without ever reaching the decision text a human or the audit trail
    # actually reads as the "reasoning." 2026-09-11. Defaults False (the
    # common case — no PM-attributed finding this cycle).
    pm_narrative_contradiction_notice_applied: bool = False
