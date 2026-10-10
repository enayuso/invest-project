"""Aggressive / Conservative / Neutral risk debators. Identical shape, only
the stance differs, so all three build off the same shared debater factory.
"""

from __future__ import annotations

import logging
import re

from app.agents.debater import make_speaker, render_reports
from app.llm.language import DEFAULT_LANGUAGE
from app.state import DebateState, SpeakerFn, TraderProposal, render_trader_proposal

logger = logging.getLogger(__name__)

# 2026-08-05: a live-decision audit found the Neutral synthesis repeatedly
# landing on a freeform compromise ("reduce the position by 50%", "partial
# profit-taking", "완전한 매도는 불필요" i.e. "not a full exit") that has no
# exact match among the Portfolio Manager's five fixed rating levels — the
# PM then had to force that conclusion into one of the five anyway, but its
# Executive Summary kept quoting the debate's own freeform language (a
# specific share count, a specific percentage) rather than the rating it
# actually picked, so the narrative described an action that never matched
# what the system executed (three real cases same day: 329180.KS, 009540.KS,
# 036930.KS — the Rating computed to Hold/no-net-change while the summary
# described an active partial sell). Root cause: nothing in this module told
# the debators the position-size axis is a five-level discrete choice, not a
# continuous one, so "propose a concrete middle path" naturally produced a
# percentage or partial-exit framing no rating exactly implements. Anchoring
# every stance to the same five levels the PM must ultimately choose from
# means whatever "compromise" Neutral reaches is already something the PM
# CAN execute exactly, with nothing left to mistranslate into the narrative.
# 2026-08-12: self-audit (`python -m app.audit debate-impact`/`report`) over
# 297 live+backtest decisions found this debate structurally converging on
# Hold — a Trader Sell proposal survived to a final Sell only 15% of the
# time (40/61 got walked back to Hold), live ratings were 61% Hold, and
# Hold was simultaneously the WORST-performing rating (46.9% win rate,
# -0.18% mean alpha) while Sell/Overweight — the two ratings this debate
# most often overrode away from — had the BEST (70.0%/+2.24%,
# 66.7%/+2.37%). Root cause: Neutral was instructed to "propose a concrete
# middle path" and that conclusion is then handed to the Portfolio Manager
# as the debate's "authoritative synthesis" (see portfolio_manager.py's
# `_render_risk_debate`) — a middle path between Buy and Sell is
# arithmetically close to Hold almost by construction, regardless of which
# side's evidence was actually stronger. Neutral's role changed below from
# "compromise" to "judge which side's case is better supported and adopt
# that side's level" — landing in between is now something Neutral must
# justify as a genuine tie, not the default outcome of averaging.
# Small-sample caveat: Sell/Overweight are only ~10 decisions each, so this
# could partly be survivorship (the few Sells that fired were the obvious
# ones) rather than pure debate suppression — treat this as a hypothesis to
# A/B-verify (backtest checkpoint/labeling infra already supports comparing
# before/after this commit), not a proven fix.
# 2026-09-01 (first real-account KRX cycle, 200,000 KRW equity, decisions
# id=1156/1157/1158 — all three tickers FLAT / zero shares held): every
# debater was handed all five levels unconditionally, and all three decisions
# had the debate converge on a rating the Portfolio Manager cannot legally
# hold for a flat ticker. id=1157's authoritative Neutral synthesis was
# "CONCLUSION: Underweight (25%)" — silently discarded by the PM. id=1158
# produced Sell(0%)/Sell(0%)/Underweight(25%) — ALL THREE illegal — which
# left Buy as the only non-Hold stance the PM could adopt, and it did.
# id=1156 spent 4 of 6 turns arguing "Underweight" for an account holding
# zero shares. The identical flat restriction already existed in the two
# sibling nodes either side of this one (research_manager._FLAT_SUFFIX,
# portfolio_manager._FLAT_SUFFIX, both hardened by 2026-08-28's dcf1576),
# which did not extend here, and pipeline._atr_context already computes this
# branch — so it was a prompt-coverage asymmetry across three siblings, not a
# missing capability. The decisive clause below is duplicated verbatim from
# those two rather than factored into a shared constant: both siblings carry
# role-specific tailoring a shared suffix cannot express, and this module
# needs the five-level MENU rewritten to two, not a suffix appended.
# tests/agents/test_debater.py asserts all three nodes still share that clause.
_HELD_RATING_SCALE_NOTE = (
    "\n\nThe position-size axis you're debating is not continuous — the Portfolio Manager "
    "who reads your conclusion can only act on one of five fixed levels: Sell (0% — full "
    "exit), Underweight (25% of full-conviction allocation — a real, non-zero target, not "
    "'mostly exit'), Hold (no change from the current position), Overweight (75%), or Buy "
    "(100%). Whatever position-size adjustment you argue for must be expressed as one of "
    "these five levels, not a custom percentage, a fraction of the current holding, or a "
    "'partial reduce/add'/'partial profit-taking' framing — none of those map onto what the "
    "Portfolio Manager can actually execute, and if your argument doesn't fit any of the five, "
    "you must state which single level you're proposing anyway rather than describe an "
    "in-between outcome no rating implements."
)

_FLAT_RATING_SCALE_NOTE = (
    "\n\nThis ticker is currently FLAT (no shares are held). The position-size axis you're "
    "debating is not continuous, and for a flat ticker it is not even five-valued — the "
    "Portfolio Manager who reads your conclusion can only act on TWO levels here: Buy (open "
    "a position at 100% of full-conviction allocation) or Hold (do not enter — stay in cash). "
    "It cannot be Overweight/Underweight/Sell, since there is no existing position to add to, "
    "trim, or exit; a conclusion naming one of those three is discarded outright and your "
    "entire turn is wasted. Do not spend your argument reasoning about whether 25% or 75% is "
    "right, and do not treat 'we are already at 0%, so Hold means Sell' as an ambiguity to "
    "resolve — there is no such ambiguity: Hold means stay out, Sell does not exist here. "
    "Crucially, this does NOT mean the argument is only about how bullish to be: a bearish "
    "conclusion on a flat ticker is expressed as Hold, which here means 'stay out / do not "
    "enter' — a fully committed, directional answer, not a hedge or a failure to decide. Only "
    "the entry decision is yours; there is no exit to decide. Whatever you argue for must be "
    "expressed as exactly Buy or Hold, not a custom percentage, not a 'partial entry', and "
    "not a 'starter position' — none of those map onto what the Portfolio Manager can "
    "actually execute. The stop-loss level you argue for is still fully live and still "
    "matters: it is the stop that will apply if a position is opened this cycle."
)


def _rating_scale_note(held: bool) -> str:
    return _HELD_RATING_SCALE_NOTE if held else _FLAT_RATING_SCALE_NOTE


# 2026-09-02 (decision id=1170, 498400.KS): the risk debate argued position
# sizing while blind to the real cap. Conservative r2 asserted "we cannot
# reduce position size (it's all-or-nothing)"; Neutral r2 -- the deciding
# turn -- adopted it as verified fact and made it the reason for the final
# Hold. Both are false: the real cap (app.orchestrator.pipeline._risk_cap_
# context) permits a share count that varies with the stop distance argued
# for. This preamble is threaded in ONLY when the orchestrator successfully
# computed the real sizing context (see pipeline.py's debate-assembly site)
# -- it describes what the code will do, and deliberately does not argue for
# any rating or direction, to avoid becoming a new injection-banner false-
# positive trigger (this same audit found 6 false positives across 3 live
# decisions, all originating from the pipeline's own imperative-shaped
# correction banners embedded in report bodies).
_SIZING_MECHANICS_PREAMBLE = (
    "How the code will actually size this trade. The block below is the same sizing "
    "arithmetic the Portfolio Manager is shown before it writes the order, reproduced "
    "here word for word rather than summarized, so this debate argues against the real "
    "constraint instead of an assumed one. It describes what the execution code does "
    "with whatever conclusion you reach. It is not an argument for any rating, and "
    "nothing in it favours entering over staying out or the reverse.\n\n"
    "The number of shares that actually reaches the market is produced by this "
    "arithmetic -- the volatility scale-down applied to the rating's base allocation, "
    "and then the stop-loss risk cap on the shares added -- and not by the rating label "
    "on its own. The stop-loss level argued in this debate is one of the inputs, which "
    "means 'how wide a stop' and 'how large a position' are the same question here, not "
    "two independent ones: a stop argued for on volatility grounds silently sets a size, "
    "and a size argued for silently constrains the stop.\n\n"
    "Any claim you make about position size -- that it is fixed, that it cannot be "
    "adjusted, that it is all-or-nothing, that a particular number of shares or "
    "percentage of capital will be committed -- has to come from the figures below. A "
    "sizing claim the figures do not support is a factual error, not a difference of "
    "opinion, and it does not stay contained to your own turn: later turns and the "
    "Portfolio Manager read this transcript as established fact and will build on it "
    "without re-deriving it."
    "\n\nScope of this block. It answers 'how would a decision already made get "
    "executed', never 'should this trade happen'. Use it to correct a factually "
    "wrong sizing claim — that sizing is fixed, that it is all-or-nothing, that a "
    "particular share count is forced — and to argue a stop level or a size. Do NOT "
    "use it as a reason for or against entering: that the arithmetic permits N "
    "shares, that the resulting allocation is or is not large enough to matter, and "
    "that the risk cap would not be breached are statements about the constraint, "
    "not about this company. And do not argue for a tighter stop in order to "
    "increase the share count — the stop belongs at the level this ticker's price "
    "behaviour justifies, and whatever size that implies is the size. Arguing a "
    "stop backwards from a desired position size inverts the purpose of the cap."
)

_SIZING_MECHANICS_FLAT_NOTE = (
    "\n\nOne distinction this ticker makes easy to collapse: it is currently flat, so the "
    "RATING menu is two-valued (Buy or Hold), but the SIZE of the resulting trade is not. "
    "'Buy' is the rating that opens a position; it does not mean 'commit all available "
    "capital', and 'the position is all-or-nothing, so size is not a lever' is not a "
    "correct reading of the two-level menu. How many shares a Buy would actually open, "
    "and how the stop level you argue for changes that number, is precisely what the "
    "block below states."
)


def _sizing_mechanics_preamble(held: bool) -> str:
    return _SIZING_MECHANICS_PREAMBLE if held else (
        _SIZING_MECHANICS_PREAMBLE + _SIZING_MECHANICS_FLAT_NOTE
    )


_AGGRESSIVE_FLAT_SUFFIX = (
    "\n\nBecause this ticker is flat, 'larger sizing' is not available to you as a lever: "
    "the only sizes are 0% (Hold) and 100% (Buy). Argue for entry or against it, and argue "
    "the stop level — not for a size."
)
_CONSERVATIVE_FLAT_SUFFIX = (
    "\n\nBecause this ticker is flat, 'smaller sizing' is not available to you as a lever: "
    "the only sizes are 0% (Hold) and 100% (Buy). Your caution must be expressed either as "
    "Hold — argue that no position should be opened at all — or as Buy with a tighter stop. "
    "Do not infer from the words 'smaller sizing' or 'tighter stop-losses' above that a "
    "position exists; it does not."
)


def _stances(held: bool) -> dict[str, str]:
    note = _rating_scale_note(held)
    aggr_flat = "" if held else _AGGRESSIVE_FLAT_SUFFIX
    cons_flat = "" if held else _CONSERVATIVE_FLAT_SUFFIX
    return {
        "Aggressive": (
            "You are the Aggressive risk debator. Advocate for maximizing return, "
            "favoring conviction and larger position sizing when the case is strong. "
            "Ground your conviction in specific evidence from the analyst reports and "
            "the trader's proposal — a generic 'be more aggressive' lean is not a "
            "case — and state a concrete position-size/stop-loss adjustment you'd "
            "push for relative to the trader's proposal, not just 'go bigger.'"
            f"{note}{aggr_flat}"
        ),
        "Conservative": (
            "You are the Conservative risk debator. Advocate for capital preservation, "
            "favoring caution, smaller sizing, and tighter stop-losses. Ground your "
            "caution in specific risks visible in the analyst reports — a generic "
            "risk-averse lean is not a case — and state a concrete position-size/"
            "stop-loss adjustment you'd push for relative to the trader's proposal, "
            "not just 'be more careful.'"
            f"{note}{cons_flat}"
        ),
        "Neutral": (
            "You are the Neutral risk debator. Your job is to JUDGE, not average — decide "
            "which of the Aggressive or Conservative case is better supported by the specific "
            "evidence in the analyst reports and the trader's proposal, and adopt THAT side's "
            "position-size/stop-loss level as your own conclusion. State plainly which side won "
            "and cite the specific evidence that tipped it — a hedge between the two is not a "
            "judgment. Landing on a genuine middle level (e.g. Hold when the trader proposed "
            "Buy) is warranted ONLY when the two cases are truly evenly matched on the actual "
            "evidence, not as a default compromise position — a close-but-not-equal case should "
            "still resolve toward whichever side has the stronger evidence, even if only somewhat "
            "stronger. Genuine ties are rare, not the norm; if you find yourself splitting the "
            "difference by default, go back and name which side's evidence is actually weaker."
            f"{note}\n\n"
            "Execution mechanics are not directional evidence. The share count a rating "
            "would produce, how much headroom is left under the risk cap, whether the "
            "resulting allocation is 'meaningful' or 'negligible', the fact that the sizing "
            "framework permits or validates a position, and a valuation multiple quoted on "
            "its own (P/B, P/E, price below book) are all facts about HOW a trade would be "
            "executed or HOW it is priced — none of them is evidence about WHETHER the "
            "thesis is sound. 'The risk cap allows a meaningful position' and 'describing "
            "this allocation as negligible is incorrect' answer a question nobody asked: a "
            "constraint that PERMITS a trade is never a reason to TAKE one, and a cheap "
            "multiple is a description of price, not a floor, an edge, or a catalyst. "
            "Decide direction first, on the substantive evidence in the analyst reports and "
            "the trader's proposal alone; only once direction is settled may sizing "
            "mechanics shape the level and the stop. If the winning side's case rests on "
            "execution mechanics or on a multiple in isolation, that side has not won — say "
            "so and judge the other way.\n\n"
            "If you already stated a verdict in an earlier round of this debate, you may "
            "reverse it, but only by naming the specific new fact or argument a later turn "
            "introduced that you did not have then — a disclosed number, a price level, a "
            "filing, a datum. 'On further reflection', a re-framing of who is on which "
            "side, a sharper restatement, or a re-weighting of the same evidence are not "
            "new facts. If there is no such fact, say that your earlier verdict stands and "
            "restate it.\n\n"
            "Your final-round conclusion is treated as this debate's authoritative synthesis "
            "downstream (the Portfolio Manager bases its rating and stop-loss/sizing directly on "
            f"it) — state your conclusion as exactly one of the {'five levels' if held else 'two levels (Buy or Hold)'} "
            "above, explicitly named, not just implied by tone."
        ),
    }


# Byte-identical to the pre-2026-09-01 prompts for a held ticker — keeps
# backtest reproducibility for held tickers and the tests / portfolio_manager
# docstring that reference `_STANCES` by name.
_STANCES = _stances(held=True)

_ILLEGAL_FLAT_RATINGS = ("Sell", "Underweight", "Overweight")

# Only a rating DECLARATION counts, never a mention: a debater correctly
# saying "Underweight is not a valid rating here" (id=1158's Conservative did
# exactly that) must not be flagged. Two shapes, both from the 2026-09-01
# transcripts: a labelled conclusion line ("CONCLUSION: Underweight (25%)")
# and a bare level-with-percentage ("Sell (0%)"). Deliberately narrow — an
# observability signal, not a gate, so a miss costs nothing and a false
# positive costs an operator's attention.
_DECLARED_RATING_RE = re.compile(
    r"(?:CONCLUSION|RECOMMENDATION|RATING|My conclusion|Final(?:\s+\w+)?)\s*[:\-—]\s*"
    r"\**\s*(Sell|Underweight|Overweight)\b",
    re.IGNORECASE,
)
_LEVEL_WITH_PCT_RE = re.compile(
    r"(?<![\w'\"])(Sell|Underweight|Overweight)\s*\(\s*\d{1,3}\s*%"
    r"(?!\s*[):\]\s]*(?:is\s+)?(?:not\b|nonsensical|invalid|n/?a\b|does\s+not|doesn't|makes\s+no\s+sense))",
    re.IGNORECASE,
)


def build_all(
    reports: dict[str, str],
    trader_proposal: TraderProposal,
    extra_context: str = "",
    *,
    language: str = DEFAULT_LANGUAGE,
    identity: str = "",
    held: bool = True,
    session_note: str = "",
    period_note: str = "",
    sizing_context: str = "",
    research_plan_note: str = "",
) -> dict[str, SpeakerFn]:
    """`extra_context`, when given, is appended to what every debator sees —
    used to feed Korean risk signals (short-selling, program trading) that
    aren't in the analyst reports for KRX names.

    `held=False` (the ticker is flat, no shares held) rewrites the rating
    menu from five levels to two (Buy / Hold) — see `_FLAT_RATING_SCALE_NOTE`
    and 2026-09-01 decisions id=1156/1157/1158.

    `session_note` (app.dataflows.session_state.intraday_narration_note) rides
    at the FRONT of the shared context — it's a global premise about the day,
    not a KRX risk label like `extra_context` (2026-09-01, id=1157).

    `sizing_context` (the pipeline's pre-debate risk_cap_context/volatility_context,
    2026-09-02, id=1170) is appended LAST, after everything else in the shared
    context — see `_SIZING_MECHANICS_PREAMBLE`'s module comment for why this
    debate needs it at all. Empty by default so a caller that doesn't pass it
    (e.g. an older test/backtest harness) gets a byte-identical prompt to
    before this change.

    `research_plan_note` (2026-09-16, engine audit follow-up item 3), when
    given, is the Research Manager's rendered plan (`render_research_plan`) —
    this debate previously saw only `trader_proposal`, whose `action` field
    is a 3-value direction (Buy/Hold/Sell) that cannot distinguish the
    Research Manager's Underweight (a 25% trim) from Sell (a full exit).
    Measured live (807 decisions, 2026-08-13..09-16): when the Research
    Manager concluded Underweight and the Trader's action collapsed to Sell,
    46% of those Trader proposals described a full exit in prose even though
    the plan called for a trim — and the Portfolio Manager's own final Sell
    rate in that path was 5.0% vs. 0.0% when the Trader stayed at Hold
    (p=0.013), i.e. the label collapse measurably propagates into a stronger
    exit than the plan intended. Placed BEFORE `trader_proposal` in the
    context (the plan is upstream of the Trader's proposal), empty by
    default so a caller that doesn't pass it gets a byte-identical prompt to
    before this change."""
    context = f"Analyst reports:\n{render_reports(reports, identity=identity)}\n\n"
    if research_plan_note:
        context += f"Research Manager's plan:\n{research_plan_note}\n\n"
    context += f"Trader's proposal:\n{render_trader_proposal(trader_proposal)}"
    if extra_context:
        context += f"\n\n{extra_context}"
    # Global data-context premises ride at the FRONT, session then period
    # (matches the identity -> session -> period order in portfolio_manager).
    if period_note:
        context = f"{period_note}\n\n{context}"
    if session_note:
        context = f"{session_note}\n\n{context}"
    if sizing_context:
        context += f"\n\n{_sizing_mechanics_preamble(held)}\n\n{sizing_context}"
    return {
        stance: make_speaker(prompt, context, label=f"risk_{stance.lower()}", language=language)
        for stance, prompt in _stances(held).items()
    }


def check_illegal_ratings(debate: DebateState, *, held: bool) -> list[str]:
    """`"round N Speaker: 'offending span'"` per risk-debate turn that
    DECLARED a rating the Portfolio Manager cannot hold for this ticker.
    Empty for a held ticker (all five levels legal) and empty in the clean
    case.

    2026-09-01, first real-account KRX cycle: id=1157's authoritative Neutral
    synthesis was "CONCLUSION: Underweight (25%)" and id=1158 declared
    Sell(0%)/Sell(0%)/Underweight(25%) — every one illegal for a flat ticker,
    every one silently discarded by the PM, and nothing in the decision
    record said so. Detection only, on purpose: the PM already handles the
    discard safely, so rewriting the turn (changing what the PM reads) or
    re-running it (~4 more LLM calls) would both be larger interventions than
    the failure warrants — same "inform, don't block" posture
    `_position_context` states. `_FLAT_RATING_SCALE_NOTE` is the fix; this is
    how an operator finds out the fix stopped working."""
    if held:
        return []
    found: list[str] = []
    for turn in debate.turns:
        spans: list[tuple[int, int, str]] = []
        for pattern in (_DECLARED_RATING_RE, _LEVEL_WITH_PCT_RE):
            for m in pattern.finditer(turn.content):
                # Dedup: "CONCLUSION: Underweight (25%)" matches both patterns.
                if any(s < m.end() and m.start() < e for s, e, _ in spans):
                    continue
                spans.append((m.start(), m.end(), m.group(0)))
        for _, _, text in sorted(spans):
            found.append(f"round {turn.round} {turn.speaker}: {text!r}")
    if found:
        logger.warning(
            "risk_debators: %d flat-ticker rating violation(s) in the risk debate — the "
            "Portfolio Manager will discard these silently: %s",
            len(found), found,
        )
    return found


# Verdict-label shapes measured over 516 live Neutral turns (id>=900). The
# `[\*_#\s>`]{0,12}` run is load-bearing, not cosmetic: id=1212's flipped
# verdict was literally "**Final Decision:**\n**Buy**" — label and level on
# separate lines with markdown between them — and a tighter separator misses
# exactly the case this field exists to record. Korean labels are included
# because the language A/B harness can put this node back into Korean.
# `LEVEL` / `PROPOSED LEVEL` added 2026-09-09: id=1158's r2 (the founding
# `check_neutral_reversal` incident) wrote "**Final Decision:**\n\n**Level:
# Buy (100%)**" and went unparsed — so the reversal that named this function
# was not actually being recorded.
_VERDICT_LABEL = (
    r"(?:FINAL\s+)?(?:CONCLUSION|RECOMMENDATION|RECOMMENDED\s+ACTION|RATING|DECISION"
    r"|VERDICT|POSITION|ACTION|JUDGMENT|STANCE|PROPOSED\s+LEVEL|LEVEL)"
    r"|(?:최종\s*)?(?:결론|판단|권고|결정)"
)
_VERDICT_LEVEL = r"(Sell|Underweight|Overweight|Hold|Buy)"
_NEUTRAL_VERDICT_RES = (
    re.compile(rf"(?:{_VERDICT_LABEL})\s*[:\-—]\s*[\*_#\s>`]{{0,12}}{_VERDICT_LEVEL}\b", re.I),
    re.compile(rf"(?:my|the)\s+(?:final\s+)?(?:conclusion|verdict|recommendation|judgment|"
               rf"decision)\s+is\s+[\*_\s]{{0,6}}{_VERDICT_LEVEL}\b", re.I),
    re.compile(rf"(?:therefore|thus|so),?\s+(?:the\s+)?(?:correct|only|appropriate|rational|"
               rf"prudent)?\s*(?:action|decision|conclusion)?\s*(?:is|supports?)\s+"
               rf"(?:to\s+)?[\*_\s]{{0,6}}{_VERDICT_LEVEL}\b", re.I),
)

# The "new fact" phrasings f79213e's Neutral clause asks a reversing turn to
# use. Presence is weak evidence the instruction was FOLLOWED, absence is weak
# evidence it was not — neither is a verdict, which is why this is a recorded
# boolean and not a gate input.
_NEW_FACT_RE = re.compile(
    r"new (?:fact|information|evidence|datum|data point)|did not have (?:then|before)"
    r"|introduced (?:in|by) (?:round|the (?:later|second)) |not available in round 1"
    r"|earlier (?:verdict|conclusion) (?:stands|no longer)|reverse my|reversing my",
    re.I,
)

# The clause's OTHER permitted reversal route — naming a later-turn ARGUMENT,
# not just a new fact. `_NEW_FACT_RE` only tests the fact half of "the specific
# new fact or argument a later turn introduced"; it has no alternative for the
# argument half, and measured 0/10 on the 10 live neutral_reversal records as
# of 2026-09-14 even though 9/10 of those turns explicitly named a later-turn
# argument (e.g. id=1282 "As the Conservative debater correctly calculated, a
# stop at ₩27,800 is virtually guaranteed to be triggered"; id=1324 "the
# Aggressive debater correctly identifies that the consensus median target is
# often conservative"). Anchored on NAMING one of the other two debaters
# (aggressive/conservative) — that speaker attribution is the structural
# signature of "an argument a later turn introduced", and it is what keeps
# this from matching a bare re-weighting of round-1 evidence that cites
# nobody.
#
# This is an OBSERVATION, not a compliance verdict, for two reasons that
# mirror `_MECHANICS_TERMS` above: (1) the clause also excludes "a
# re-weighting of the same evidence" as a reversal justification, and no
# regex can tell "the Conservative introduced this in round 2" apart from
# "the Conservative re-weighted what was already on the table in round 1";
# (2) naming an opponent's argument in order to REJECT it also matches — the
# regex only detects attribution, not agreement. Do not turn this into a
# violation count or a gate input.
#
# Base rate, measured on the 2026-09-14 replay: this pattern matched 9 of 10
# reversed Neutral final turns, but it also matched 98 of 146 non-reversed
# final turns (~67%). 상대 논거를 언급하는 것은 그냥 평범한 토론 문장일 뿐임
# — matching this often even when nothing was reversed means a lone `true`
# here is weak evidence at best and must never be read by itself as "the
# reversal was justified".
_LATER_TURN_ARGUMENT_RE = re.compile(
    r"(?:the\s+)?(?:aggressive|conservative)\s+(?:debat\w+\s+)?(?:correctly\s+)?"
    r"(?:identifies|identified|calculates|calculated|demonstrates|demonstrated|shows|showed|"
    r"argues|argued|points\s+out|pointed\s+out|notes|noted|introduces|introduced|"
    r"raises|raised|establishes|established)",
    re.I,
)

# The exact vocabulary f79213e's Neutral clause names as NON-evidence. Recorded
# as observed terms, never as a violation count — "risk cap" appearing while
# CORRECTLY rebutting a mechanics argument (id=1230 r2 did precisely that) is
# indistinguishable by regex from the id=1212 misuse. Every entry must be
# traceable to the clause prose a few lines up in this same file (the "Execution
# mechanics are not directional evidence" paragraph) — do not add a paraphrase
# or a general "sizing-sounding word" just because it seems plausible; that's
# how this list drifts from the clause it's supposed to measure.
#
# 2026-09-14: "execution mechanics" and bare "negligible" were tried against
# 155 live neutral_reversal records and deliberately dropped. "execution
# mechanics" fires hardest exactly when a COMPLIANT debater quotes the clause
# back before rejecting it ("As instructed, ... execution mechanics ..."), so
# it measures compliance, not misuse. Bare "negligible" matched unrelated
# prose (id=1268: "this volume is negligible" — trading volume, not position
# sizing); "negligible position" below keeps the genuine sizing hits without
# it. 절 원문 그대로여도 준수 시 인용되는 단어는 이 목록에서 제외한다.
_MECHANICS_TERMS = (
    "risk cap", "risk-cap", "share count", "position size", "position sizing",
    "sizing framework", "allocation", "too small to matter", "immaterial",
    "economically inefficient", "mechanically",
    "negligible position", "P/B", "PBR", "P/E", "book value",
    "below book",
)


# How much of the final Neutral turn's tail check_neutral_reversal scans for
# _MECHANICS_TERMS. Deliberately NOT the whole turn — scanning everything would
# match a debater merely SUMMARIZING the other side's mechanics arguments
# (false positives), whereas the concluding passage is where a genuine misuse
# lives. Named so `record`'s new coverage keys can report against it instead of
# a bare literal.
_MECHANICS_TAIL_CHARS = 1200


def _neutral_verdict(text: str) -> tuple[str | None, str]:
    """Last verdict declaration in the turn and its literal span. Reverse-scan
    ("last wins") matches `portfolio_manager._render_risk_debate`'s own
    final-turn convention — a turn that names levels while surveying the
    debate must not have its FIRST mention read as its conclusion.
    `(None, "")` when nothing parses: measured 91% coverage over the 166 live
    Neutral turns since id=1150, so an unparsed turn is a normal outcome, not
    an error, and must never be inferred as a flip."""
    best: tuple[int, str, str] | None = None
    for pattern in _NEUTRAL_VERDICT_RES:
        for m in pattern.finditer(text):
            if best is None or m.start() > best[0]:
                best = (m.start(), m.group(1).capitalize(), m.group(0))
    return (best[1], best[2]) if best else (None, "")


def check_neutral_reversal(debate: DebateState) -> dict:
    """Structured record of whether the Neutral debator's round-1 verdict
    survived to its final round — `{}` when there is no Neutral turn in both
    a round 1 and a later final round (nothing to compare).

    2026-09-04, id=1212/1217: Neutral reversed its own r1 verdict with no new
    fact between rounds (1212 Hold->Buy on "the risk cap allows a meaningful
    position"; 1217 Sell->Overweight on "0.43x P/B" plus the sizing
    arithmetic), and `portfolio_manager._render_risk_debate` promotes only the
    FINAL turn to "=== FINAL SYNTHESIS ===" — so the reversal, and the fact
    that it was unjustified, was invisible to the PM and to every later audit.
    Commit f79213e added a prompt clause forbidding exactly this; measuring
    whether that clause holds required reading transcript prose by hand,
    because nothing recorded the r1 verdict as data. This is that record.

    Detect-only, deliberately. The 2026-09-04 flip cohort was bidirectional —
    id=1220 reversed Buy->Hold and was RIGHT to — so "revert an uncited flip
    to r1" would have restored a Buy on a shell micro-cap. The downstream
    exposure-increase risk is already gated by
    `pipeline._apply_consensus_override_gate`; this field's job is to say
    whether the prompt fix is working, not to act.

    2026-09-14: the clause permits reversing on a new fact OR on "the
    specific ... argument a later turn introduced" — `new_fact_cited` only
    ever tested the fact half. Over the 10 live reversals recorded so far,
    `new_fact_cited` was 0/10 while `later_turn_argument_cited` (naming the
    Aggressive or Conservative debater's argument) was 9/10, so
    `new_fact_cited` alone reads as "nobody complied" when nine visibly took
    the clause's other permitted route. Neither field alone establishes
    compliance, though: the clause's "re-weighting of the same evidence"
    exclusion is not mechanically checkable, and naming an opponent's
    argument in order to reject it also sets `later_turn_argument_cited`.
    둘 다 관찰 기록일 뿐, 준수 여부의 최종 판정은 아님.

    2026-09-17: `reversal_basis` is a derived summary of the two booleans
    above — `"new_fact"` (new_fact_cited True), `"later_turn_argument"`
    (new_fact_cited False, later_turn_argument_cited True), `"neither"`
    (both False), or `""` when `reversed_` is False. It exists only so a
    reader does not have to join two booleans by hand and fall into the
    exact trap id=1539 fell into on 2026-09-17: reading `new_fact_cited:
    false` alone and concluding the clause was violated, when
    `later_turn_argument_cited` was true. THIS FIELD IS A ROUTE
    CLASSIFICATION, NOT A COMPLIANCE VERDICT — everything above about
    `later_turn_argument_cited` still applies to `"later_turn_argument"`:
    naming an opponent's argument in order to REJECT it also produces this
    value, and the clause's "re-weighting of the same evidence" exclusion
    is not mechanically detectable. `reversal_basis == "later_turn_argument"`
    must never be read as "this reversal was legitimate".
    이 필드는 경로 분류일 뿐 준수 여부의 최종 판정이 아니다 — 상대 논거를
    반박하기 위해 인용한 경우에도 값이 찍힌다."""
    neutrals = [t for t in debate.turns if t.speaker == "Neutral"]
    if not neutrals:
        return {}
    final_round = debate.rounds_completed or debate.turns[-1].round
    first = next((t for t in neutrals if t.round == 1), None)
    final = next((t for t in reversed(neutrals) if t.round == final_round), None)
    if first is None or final is None or first is final:
        return {}
    r1_verdict, r1_span = _neutral_verdict(first.content)
    final_verdict, final_span = _neutral_verdict(final.content)
    reversed_ = bool(r1_verdict and final_verdict and r1_verdict != final_verdict)
    tail = final.content[-_MECHANICS_TAIL_CHARS:]
    later_turn_argument_match = _LATER_TURN_ARGUMENT_RE.search(final.content)
    new_fact_cited = bool(_NEW_FACT_RE.search(final.content))
    later_turn_argument_cited = bool(later_turn_argument_match)
    if not reversed_:
        reversal_basis = ""
    elif new_fact_cited:
        reversal_basis = "new_fact"
    elif later_turn_argument_cited:
        reversal_basis = "later_turn_argument"
    else:
        reversal_basis = "neither"
    record = {
        "r1": r1_verdict,
        "final": final_verdict,
        "final_round": final_round,
        "reversed": reversed_,
        "new_fact_cited": new_fact_cited,
        "mechanics_terms": sorted({t for t in _MECHANICS_TERMS if t.lower() in tail.lower()}),
        "r1_span": r1_span,
        "final_span": final_span,
        "final_turn_chars": len(final.content),
        "mechanics_scanned_chars": min(_MECHANICS_TAIL_CHARS, len(final.content)),
        "later_turn_argument_cited": later_turn_argument_cited,
        "later_turn_argument_span": later_turn_argument_match.group(0) if later_turn_argument_match else "",
        "reversal_basis": reversal_basis,
    }
    if reversed_ and not new_fact_cited:
        if later_turn_argument_cited:
            argument_note = "a later-turn argument WAS named (%r)" % record["later_turn_argument_span"]
            logger.info(
                "risk_debators: Neutral reversed its round-1 verdict %s -> %s with no "
                "new-fact citation; %s (mechanics/valuation terms in the concluding "
                "passage: %s) — permitted route per f79213e's Neutral stance clause",
                r1_verdict, final_verdict, argument_note, record["mechanics_terms"] or "none",
            )
        else:
            argument_note = "no later-turn argument was named either"
            logger.warning(
                "risk_debators: Neutral reversed its round-1 verdict %s -> %s with no "
                "new-fact citation; %s (mechanics/valuation terms in the concluding "
                "passage: %s) — see f79213e's Neutral stance clause",
                r1_verdict, final_verdict, argument_note, record["mechanics_terms"] or "none",
            )
    return record
