"""Detect-only recorder for an agent inventing an AUTOMATIC regulatory/
exchange consequence ("triggers automated compliance protocols: trading
halts, exclusion from benchmark indices, and forced liquidation by
institutional funds") that is not grounded in any data this pipeline's cycle
actually supplied.

Built for id=1380: a confirmed delinquent-filer determination was narrated as
mechanically triggering a cascade of specific downstream consequences — none
of which came from any tool output this cycle (`execution_status_context`,
`kr_risk_extra`, `verified_snapshot`, `kr_gt`, `kr_news_gt`,
`kr_financials_recency`). The desk stated a *causal mechanism* ("X
automatically triggers Y"), not merely a fact ("KRX designated the stock
불성실공시법인") or a hedge ("could eventually lead to Y") — the causal-frame
requirement below is exactly what separates a genuinely fabricated mechanism
from ordinary reporting or appropriately hedged speculation.

This is a sibling of `app.agents.halt_status_guard`, not a modification of
it — deliberately no overlap in trigger shape. `halt_status_guard` fires on a
PRESENT-TENSE "is halted RIGHT NOW" assertion contradicted by a
CONFIRMED-clear broker read; this guard fires on an AUTOMATIC-CONSEQUENCE
causal claim ("triggers"/"automatically"/"mandates"...) regardless of tense,
and for the halt/suspension consequence specifically, only when
`execution_status_context` was not even supplied this cycle — a live,
present-tense contradiction of a CONFIRMED-clear read stays halt_status_
guard's territory alone. The other four consequence families here
(delisting, index exclusion, margin/credit-trading ban,
institutional-mandate divestment) have no corresponding field in this
pipeline at all, so they are flagged unconditionally whenever the
automatic-consequence frame co-occurs with the consequence term — no
`supplied_fields` value can ever ground them.

Detect-only — no control-flow change, no LLM call, not cache-gating. Mirrors
`app.agents.risk_debators.check_neutral_reversal`'s "positive evidence, not a
gate" role, and copies halt_status_guard's regex/window/exemption/dedup
architecture rather than inventing a new one."""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# A specific automatic regulatory/exchange consequence, EN + KR. Closed
# enumeration by design — a broad "anything bad might happen" scan would
# just re-invent a vague sentiment detector; this only matches the exact
# consequence shapes seen fabricated in practice. Named groups let the
# grounding check below tell which family matched without a second pass.
_CONSEQUENCE_RE = re.compile(
    r"(?P<liquidation>forced\s+liquidation|강제\s*청산)"
    r"|(?P<delisting>delist(?:ed|ing)?|상장\s*폐지)"
    r"|(?P<index_exclusion>(?:index|benchmark)\s+exclusion|exclusion\s+from\s+(?:the\s+)?"
    r"(?:benchmark\s+)?indices|지수\s*(?:편출|제외))"
    r"|(?P<halt>trading\s+halts?|trading\s+suspensions?|halted|suspended|"
    r"거래\s*정지|매매\s*정지)"
    r"|(?P<margin_ban>margin[- ]trading\s+ban|credit[- ]trading\s+ban|"
    r"신용\s*거래\s*금지|대용\s*증권\s*제외)"
    r"|(?P<institutional_divest>institutional[- ]mandate\s+divestment|"
    r"forced\s+divestment\s+by\s+institutional\w*|"
    r"divestment\s+by\s+institutional\w*\s+funds?|"
    r"기관\s*(?:투자자)?\s*의무\s*매도)",
    re.IGNORECASE,
)

_CONSEQUENCE_LABELS = {
    "liquidation": "forced liquidation",
    "delisting": "delisting",
    "index_exclusion": "index exclusion",
    "halt": "trading halt/suspension",
    "margin_ban": "margin/credit-trading ban",
    "institutional_divest": "institutional-mandate divestment",
}

# The causal/automaticity frame that turns a bare fact ("KRX designated the
# stock") into a fabricated MECHANISM ("designation triggers ..."). Required
# to co-occur near the consequence term — without it, this guard never
# fires, by construction (see module docstring).
_AUTOMATIC_FRAME_RE = re.compile(
    r"triggers?|automatically|mandates?|requires?|leads?\s+to|results?\s+in|"
    r"compliance\s+protocols?|"
    r"자동(?:적으)?으로|의무적으로|발동된다",
    re.IGNORECASE,
)

# How close the automatic-frame marker must be to the consequence term to
# count as describing THAT consequence (chars, not tokens — same convention
# as halt_status_guard._CURRENCY_WINDOW). 100 (the wide end of the ~80-100
# range) rather than 90: id=1380's own sentence lists three consequence
# terms after a single shared "triggers automated compliance protocols:"
# frame, and the last of the three ("forced liquidation by institutional
# funds") sits ~94 chars from "triggers" — 90 would miss it.
_FRAME_WINDOW = 100

# Hedged/speculative/warning framing suppresses a flag even when the frame
# and consequence terms both match — "could eventually lead to a
# suspension" must never fire. Checked over a WIDER window than the frame
# check above, same convention as halt_status_guard._EXEMPT_WINDOW.
_EXEMPT_WINDOW = 100
_HEDGE_RE = re.compile(
    r"could|might|may|possibl[ey]|potential(?:ly)?|risk\s+of|warning|"
    r"if\s+(?:the\s+)?(?:krx|korea\s+exchange|regulators?)|hypothetical|"
    r"scenario|were\s+to|in\s+the\s+event|eventually|"
    r"가능성|우려|경고",
    re.IGNORECASE,
)


def _quoted_span(text: str, start: int, end: int) -> str:
    """The sentence containing `text[start:end]`, delimited by the nearest
    `.` on each side (or the text boundary) — same sentence-bounded
    convention as `halt_status_guard._quoted_span`, so a repeated sentence
    dedups identically regardless of where in the text it recurs."""
    left = text.rfind(".", 0, start)
    left = left + 1 if left != -1 else 0
    right = text.find(".", end)
    right = right if right != -1 else len(text)
    return text[left:right].strip()


def scan_regulatory_mechanism(text: str, *, supplied_fields: frozenset[str], agent_name: str) -> list[str]:
    """`"AgentName: '<span>' — ungrounded automatic <consequence> claim"` per
    sentence asserting a specific automatic regulatory/exchange consequence
    with no data this pipeline actually supplied to ground it this cycle.

    `halt`-family claims defer entirely to `halt_status_guard` (this guard's
    own territory is the AUTOMATIC-mechanism framing, not the present-tense
    "is it halted right now" framing) whenever `"execution_status_context"`
    is in `supplied_fields` — that field is exactly what halt_status_guard
    itself checks a present-tense claim against, so a halt claim made while
    it was supplied is left for that guard alone. Every other consequence
    family (delisting, index exclusion, forced liquidation,
    institutional-mandate divestment, margin/credit-trading ban) has no
    corresponding field anywhere in this pipeline, so those are flagged
    unconditionally once the automatic-consequence frame and the consequence
    term co-occur and no hedge/warning framing is nearby. Dedups identical
    spans within one call."""
    if not text:
        return []

    findings: list[str] = []
    seen: set[str] = set()
    for m in _CONSEQUENCE_RE.finditer(text):
        start, end = m.start(), m.end()
        category = m.lastgroup
        if category is None:
            continue

        near = text[max(0, start - _FRAME_WINDOW) : end + _FRAME_WINDOW]
        if not _AUTOMATIC_FRAME_RE.search(near):
            continue

        wide = text[max(0, start - _EXEMPT_WINDOW) : end + _EXEMPT_WINDOW]
        if _HEDGE_RE.search(wide):
            continue

        if category == "halt" and "execution_status_context" in supplied_fields:
            # A live/present halt claim, grounded (or contradicted) by a real
            # broker read this cycle — halt_status_guard's territory alone.
            continue

        span = _quoted_span(text, start, end)
        if span in seen:
            continue
        seen.add(span)
        label = _CONSEQUENCE_LABELS.get(category, category)
        findings.append(f"{agent_name}: {span!r} — ungrounded automatic {label} claim")

    if findings:
        logger.warning(
            "regulatory_mechanism_guard: %s asserted an automatic regulatory/exchange "
            "consequence not grounded in any data supplied this cycle — %s",
            agent_name, findings,
        )
    return findings
