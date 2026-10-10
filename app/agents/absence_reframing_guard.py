"""Detect-only recorder for a report/debate turn/PM-decision sentence that
reframes MISSING data (insider selling/buying activity, investor net-buy
flow, analyst/institutional coverage) as a specific POSITIVE/reassuring
signal, rather than the correct (and, in the same sample, far more common)
treatment of missing data as uncertainty/risk.

Built for the 2026-09-09 domestic AI-decision audit, same 12-decision sample
as `app.agents.halt_status_guard` and `app.agents.staleness_narrative_guard`
(ids 1348-1359). Three distinct phrasing families recurred:

  - id=1349 (950250.KS): "the absence of insider selling suggests management
    does not believe the current price represents fair value, reducing the
    likelihood of a fraudulent pump-and-dump exit."
  - id=1350 (079650.KS): "In the absence of granular flow data, the price
    action itself serves as the primary proxy for sentiment: ... buyers are
    actively absorbing selling pressure." and, separately, "'No Analyst
    Coverage/Liquidity Risk': Lack of coverage reduces information asymmetry
    for insiders but increases opportunity for undervalued discovery."
  - id=1351 (198440.KS): "The lack of analyst coverage ... is a bullish point
    for short-term traders, as it prevents institutional suppression and
    allows retail momentum to drive price action unchecked."

Critically: in this SAME sample, "lack of coverage" / "absence of insider
selling" is used CORRECTLY (as an uncertainty/risk signal) far more often
than it is misused — e.g. id=1349's own risk debate rebuts a sibling turn
("The bull argues the lack of insider *selling* is positive. In micro-caps,
silence is not affirmation; it is indifference."), and other reports say
plainly "Lack of analyst coverage means little external scrutiny or guidance
for investors." / "Zero analyst coverage means no external validation of the
financial narrative." Each template below therefore pairs a narrow anchor
phrase (the "absence of X" claim) with a SPECIFIC nearby positive-spin phrase
tuned against these real negative (must-not-fire) examples from the same
sample — not a broad "absence/lack" + generic sentiment match. Broadening the
spin-phrase lists to catch "more" cases is deliberately out of scope: a false
positive here (flagging a correct risk framing) is worse than a missed case,
the same philosophy as every other guard in this codebase.

This is a narrative/decision-quality problem only, the same class as
`app.agents.halt_status_guard.scan_halt_claims` and
`app.agents.staleness_narrative_guard.scan_staleness_narrative` — detect-only,
no control-flow change, no LLM call, NOT cache-gating. Multi-node like
`scan_halt_claims` (scans every analyst report, the research plan, the risk
debate, and the PM's own final decision text), not single-node like
`scan_staleness_narrative` (which is scoped to the Fundamentals Analyst
only)."""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# --- Template 1: insider silence framed as management confidence / reduced risk ---
# id=1349 (950250.KS): "the absence of insider selling suggests management
# does not believe the current price represents fair value, reducing the
# likelihood of a fraudulent pump-and-dump exit."
# id=1350 (079650.KS): "In the absence of reported insider sales... The lack of
# such noise supports the view that the current price appreciation is organic
# ... however, silence here is not a red flag."
# id=1352 (028050.KS): "In the absence of conflicting insider selling signals...
# We assume neutral-to-positive insider stance given the prevailing buy-side
# pressure."
# id=1358 (047040.KS): "There are no public disclosures of insider buying/selling
# reported... Absence of negative insider signals reduces downside risk from
# management dumping shares."
# Contrast (must NOT fire) — same sample, id=1349's OWN risk debate rebuts a
# sibling turn: "The bull argues the lack of insider *selling* is positive. In
# micro-caps, silence is not affirmation; it is indifference." — the spin list
# below deliberately excludes bare "positive"/"affirmation" for this reason;
# it only matches a SPECIFIC claim of confidence/reduced-risk/organic framing.
_INSIDER_SILENCE_RE = re.compile(
    r"\b(?:absence|lack) of\b[^.]{0,40}?insider\b[^.]{0,40}?\b(?:sell(?:ing)?|sales?|signals?)\b",
    re.IGNORECASE,
)
_INSIDER_CONFIDENCE_SPIN_RE = re.compile(
    r"management (?:does not|doesn't) believe|reduc\w+ the likelihood|"
    r"reduc\w+ (?:the )?(?:downside )?risk|supports? (?:the )?(?:bullish )?(?:view|thesis|case)|"
    r"\borganic\b|not a red flag|neutral-to-positive",
    re.IGNORECASE,
)
_SPIN_WINDOW_1 = 300

# --- Template 2: missing flow/investor data framed as a positive price-action proxy ---
# id=1350 (079650.KS): "detailed individual/foreign/institutional net-buy flow
# data was not retrieved for this cycle due to scope constraints. In the
# absence of granular flow data, the price action itself serves as the primary
# proxy for sentiment: ... buyers are actively absorbing selling pressure."
# id=1352 (028050.KS): "Specific daily net-flow figures for foreign vs.
# domestic institutions were not retrieved in this cycle due to prioritization
# of core price and indicator validation, but the structural setup implies
# healthy absorption."
# NOTE: the between-clause gap uses `[^\n]` (any non-newline char), not
# `[^.]` as in the anchor/spin windows elsewhere in this module — the actual
# id=1352 sentence has "vs." (an abbreviation, not a sentence break) between
# "net-flow" and "not retrieved", which `[^.]{0,60}?` cannot cross, making
# the anchor fail to match its own source incident. A fixed-radius window
# already bounds the scope here, so tolerating an internal "." costs nothing.
_MISSING_FLOW_RE = re.compile(
    r"(?:flow|investor flow|net-buy flow|net-flow)[^\n]{0,60}?(?:not retrieved|not returned|not provided|not available)|"
    r"(?:not retrieved|not returned|not provided|not available)[^\n]{0,60}?(?:flow|investor flow|net-buy flow|net-flow)",
    re.IGNORECASE,
)
_PROXY_SPIN_RE = re.compile(
    r"price action (?:itself )?(?:serves as|is|acts as) (?:the )?(?:primary )?proxy|"
    r"implies healthy absorption|actively absorbing selling pressure|"
    r"structural setup (?:implies|suggests|is) healthy",
    re.IGNORECASE,
)
_SPIN_WINDOW_2 = 400

# --- Template 3: lack of analyst/institutional coverage framed as a bullish opportunity ---
# id=1350 (079650.KS): "'No Analyst Coverage/Liquidity Risk': Lack of coverage
# reduces information asymmetry for insiders but increases opportunity for
# undervalued discovery."
# id=1351 (198440.KS): "The lack of analyst coverage (a bearish point for
# long-term holders) is a bullish point for short-term traders, as it prevents
# institutional suppression and allows retail momentum to drive price action
# unchecked."
# id=1359 (271830.KS): "The absence of heavy institutional coverage often
# correlates with shorter floats. ... shorts may be getting squeezed, fueling
# further upside."
# Contrast (must NOT fire) — same sample, many correct uses: "Lack of analyst
# coverage means little external scrutiny or guidance for investors." / "Zero
# analyst coverage means no external validation of the financial narrative." /
# "The bull dismisses the lack of analyst coverage as a 'contrarian entry'
# opportunity. This is dangerous." (bare "opportunity" is deliberately NOT in
# the spin list for this reason — only "opportunity for undervalued discovery"
# or "increase(s) opportunity" count).
_COVERAGE_ABSENCE_RE = re.compile(
    r"\b(?:lack|absence) of\b[^.]{0,30}?(?:analyst |sell-side |institutional )?coverage\b",
    re.IGNORECASE,
)
_COVERAGE_SPIN_RE = re.compile(
    r"is a bullish point|opportunity for undervalued discovery|increases? opportunity|"
    r"fuel\w* (?:further )?(?:upside|squeeze)|short squeeze|shorts? (?:may be|are|getting) squeezed|"
    r"prevents? institutional suppression|allows? retail momentum",
    re.IGNORECASE,
)
_SPIN_WINDOW_3 = 200


def _quoted_span(text: str, start: int, end: int) -> str:
    """The sentence containing `text[start:end]`, delimited by the nearest
    `.` on each side (or the text boundary) — copied from
    `app.agents.halt_status_guard._quoted_span` verbatim; see that
    function's own docstring for why a sentence-bounded span (not a
    fixed-radius window) is required for this call's dedup to be stable."""
    left = text.rfind(".", 0, start)
    left = left + 1 if left != -1 else 0
    right = text.find(".", end)
    right = right if right != -1 else len(text)
    return text[left:right].strip()


def scan_absence_reframing(text: str, *, agent_name: str) -> list[str]:
    """`"AgentName: '<span>' — <label>"` per sentence found in `text` where an
    "absence of X" anchor phrase (insider selling, missing investor-flow
    data, or lack of analyst/institutional coverage) is followed, within a
    template-specific window, by a SPECIFIC positive-spin phrase reframing
    that absence as a reassuring or bullish signal — see the three template
    comments above for the exact incident quotes and contrast (must-not-fire)
    examples each template was tuned against.

    `[]` for empty `text`. Each of the three templates is checked
    independently over the whole text; a text can trip more than one
    template. Detect-only — no control-flow change, no LLM call; mirrors
    `app.agents.halt_status_guard.scan_halt_claims`'s shape. Dedups identical
    spans within one call."""
    if not text:
        return []

    findings: list[str] = []
    seen: set[str] = set()

    def _check(anchor_re: re.Pattern, spin_re: re.Pattern, window: int, label: str) -> None:
        for m in anchor_re.finditer(text):
            near = text[m.end(): m.end() + window]
            spin_match = spin_re.search(near)
            if not spin_match:
                continue
            span = _quoted_span(text, m.start(), m.end() + spin_match.end())
            if span in seen:
                continue
            seen.add(span)
            findings.append(f"{agent_name}: {span!r} — {label}")

    _check(
        _INSIDER_SILENCE_RE, _INSIDER_CONFIDENCE_SPIN_RE, _SPIN_WINDOW_1,
        "absence of insider selling framed as management confidence/reduced risk",
    )
    _check(
        _MISSING_FLOW_RE, _PROXY_SPIN_RE, _SPIN_WINDOW_2,
        "missing flow/investor data framed as a positive price-action proxy",
    )
    _check(
        _COVERAGE_ABSENCE_RE, _COVERAGE_SPIN_RE, _SPIN_WINDOW_3,
        "lack of analyst/institutional coverage framed as a bullish opportunity",
    )

    if findings:
        logger.warning(
            "absence_reframing_guard: %s reframed missing/absent data as a positive signal — %s",
            agent_name, findings,
        )
    return findings
