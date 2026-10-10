"""Detect-only recorder for the Portfolio Manager claiming the desk was
UNANIMOUS when the Research Manager and the Trader actually disagreed with
each other or with the PM's own final rating.

Built for the 2026-09-09 live incident, decision id=1352 (028050.KS/삼성E&A):
`_desk_consensus_note` (app.orchestrator.pipeline) told the PM, verbatim,
"the Research Manager concluded Underweight, and the Trader proposed Sell"
— and the PM's final executive_summary wrote "The desk unanimously
recommends HOLD, and I concur." The rating itself (Hold against a bearish
desk) is inside the PM's authority and this guard does not touch it; what
it makes non-silent is the false claim of agreement, which meant the 2
held shares this bearish desk wanted reduced were never actually
questioned in the decision text a human or a later audit would read.

Ground truth here is TWO enums (`research_plan.recommendation.value`,
`trader_proposal.action.value`), not a string marker like
`app.agents.halt_status_guard`'s — so this guard compares structured values
first and only uses regex to find where in the prose the false claim was
made. `[]` whenever either upstream object is missing (no desk-consensus
block was ever shown to the PM — same `""` contract
`pipeline._desk_consensus_note` itself uses), so a decision with no
comparable ground truth is never scanned, by construction.

Detect-only, deliberately — same reasoning as halt_status_guard: this is a
narrative/decision-quality problem (the note travels into the transcript
regardless, so an audit already has the real RM/Trader values to compare
against), not an execution-safety one. Mirrors
`app.agents.halt_status_guard.scan_halt_claims`'s shape: enum ground truth
in place of the marker check, a claim regex, a negation exemption, and a
sentence-bounded quoted span for dedup."""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# A claim of AGREEMENT ACROSS THE DESK. Bare "consensus" is deliberately
# EXCLUDED: the injected block is literally headed "DESK CONSENSUS BEFORE THE
# RISK DEBATE" (pipeline._desk_consensus_note), so a PM echoing that word is
# describing the block, not asserting agreement — only an unanimity lexeme
# counts.
_UNANIMITY_RE = re.compile(
    r"unanimous(?:ly)?|in\s+unison|no\s+dissent|without\s+dissent|full\s+agreement|"
    r"all\s+(?:three|four|of\s+the\s+\w+)\b[^.\n]{0,30}?(?:agree|concur|recommend|point)|"
    r"every\s+(?:analyst|node|desk\s+member|upstream)\s+(?:agree|concur|recommend)|"
    r"(?:the\s+)?(?:entire|whole)\s+desk\s+(?:agrees|concurs|recommends|is\s+aligned)|"
    r"desk[- ]wide\s+(?:agreement|alignment)|"
    r"만장일치|이견\s*없이|전원\s*동의|모두\s*동의",
    re.IGNORECASE,
)

# The rating level the claim names, when it names one — e.g. "unanimously
# recommends HOLD". Checked within a short window AFTER the unanimity term,
# bounded by BOTH `_CLAIM_WINDOW` chars and the END OF THE SENTENCE the
# unanimity term sits in — whichever comes first. A rating named in the
# FOLLOWING sentence is never the level this claim names: id=1407
# (2026-09-14) wrote "... The desk is unanimously aligned. The risk debate
# thoroughly examined a Buy case ...", and the flat 90-char window reached
# across the period and picked up that "Buy", turning a TRUE unanimity claim
# (RM=Hold/Trader=Hold/PM=Hold) into a reported violation. Truncating is the
# safe failure direction: with no level in the window `claimed_level` is None
# and the check falls back to the conservative `all_agree` test.
_CLAIMED_LEVEL_RE = re.compile(r"\b(Sell|Underweight|Hold|Overweight|Buy)\b", re.IGNORECASE)
_CLAIM_WINDOW = 90  # chars after the unanimity term, to find a named level
_SENTENCE_END_RE = re.compile(r"[.!?\n]")

# Negated / instructional framing (LEFT context only, up to the match) —
# designed in from the start (same lesson as session_close_guard: the
# 2026-09-09 audit's own finding is that halt_status_guard has exactly this
# blind spot, 3/3 false positives on negated claims).
_NEGATED_RE = re.compile(
    r"\b(?:not|no|isn'?t|wasn'?t|hasn'?t|has\s+not|without|absent|lack(?:s|ing)?\s+of|"
    r"far\s+from)\b[^.\n]{0,40}$",
    re.IGNORECASE,
)

# Bucketed direction per rating value — Sell/Underweight both "reduce",
# Overweight/Buy both "increase", Hold is its own bucket. A trader action is
# always one of Buy/Hold/Sell (app.state.TraderAction — this long-only system
# has no Overweight/Underweight trader action), so the same table covers
# both the research recommendation and the trader action.
_DIRECTION = {
    "sell": "reduce", "underweight": "reduce",
    "hold": "hold",
    "overweight": "increase", "buy": "increase",
}


def _quoted_span(text: str, start: int, end: int) -> str:
    """The sentence containing `text[start:end]` — same convention as
    `app.agents.halt_status_guard._quoted_span`, for the same dedup reason."""
    left = text.rfind(".", 0, start)
    left = left + 1 if left != -1 else 0
    right = text.find(".", end)
    right = right if right != -1 else len(text)
    return text[left:right].strip()


def scan_desk_narrative(text: str, *, rm: str, trader: str, pm_rating: str, agent_name: str) -> list[str]:
    """`"AgentName: claimed desk unanimity ... vs RM=<rm>/Trader=<trader>/
    PM=<pm_rating> — '<span>'"` per unanimity claim in `text` that is false
    against the real `rm`/`trader`/`pm_rating` values — `[]` when `rm` or
    `trader` is falsy (no desk-consensus block existed this cycle to
    misdescribe). Two ways a claim is false: (1) the three directions
    (reduce/hold/increase, bucketed via `_DIRECTION`) are not all equal, or
    (2) the claim names a specific level that neither `rm` nor `trader`
    actually held. Detect-only — no control-flow change, no LLM call.
    Dedups identical spans within one call."""
    if not rm or not trader:
        return []
    if not text:
        return []

    rm_dir = _DIRECTION.get(rm.lower())
    tr_dir = _DIRECTION.get(trader.lower())
    pm_dir = _DIRECTION.get((pm_rating or "").lower())
    all_agree = rm_dir is not None and rm_dir == tr_dir == pm_dir

    findings: list[str] = []
    seen: set[str] = set()
    for m in _UNANIMITY_RE.finditer(text):
        start, end = m.start(), m.end()
        left_ctx = text[max(0, start - 40) : start]
        if _NEGATED_RE.search(left_ctx):
            continue

        lookahead = text[end : end + _CLAIM_WINDOW]
        sentence_end = _SENTENCE_END_RE.search(lookahead)
        if sentence_end is not None:
            lookahead = lookahead[: sentence_end.start()]
        claimed_level_m = _CLAIMED_LEVEL_RE.search(lookahead)
        claimed_level = claimed_level_m.group(1) if claimed_level_m else None

        is_false = not all_agree
        if not is_false and claimed_level is not None:
            claimed_dir = _DIRECTION.get(claimed_level.lower())
            actual_levels = {rm.lower(), trader.lower(), (pm_rating or "").lower()}
            if claimed_level.lower() not in actual_levels and claimed_dir != rm_dir:
                is_false = True
        if not is_false:
            continue

        span = _quoted_span(text, start, end)
        if span in seen:
            continue
        seen.add(span)
        findings.append(
            f"{agent_name}: claimed desk unanimity{f' ({claimed_level})' if claimed_level else ''} "
            f"vs RM={rm}/Trader={trader}/PM={pm_rating} — {span!r}"
        )

    if findings:
        logger.warning(
            "desk_consensus_narrative_guard: %s claimed desk unanimity that contradicts the "
            "real RM/Trader/PM values (id=1352 shape) — %s",
            agent_name, findings,
        )
    return findings
