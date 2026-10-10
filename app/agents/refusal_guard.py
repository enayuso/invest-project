"""Detect-only recorder for an analyst report that opens with an "as an AI
model I don't have real-time data / that date is past my training cut-off"
refusal preamble instead of analyzing the ticker it was handed.

The production model (`nvidia/Qwen3.6-35B-A3B-NVFP4`) intermittently treats
the analysis date it is given — a real, authoritative date supplied by the
pipeline, not a hypothetical — as beyond its own training cut-off, and writes
a disclaimer to that effect. The system prompt has told it explicitly not to
do this since 2026-08-04 (`DATE_GROUNDING` in
`app.agents.untrusted_content`), and it still happens. This module does NOT
try to fix that again: it RECORDS the occurrence so the residual rate stops
being invisible between audits.

Live measurement backing this (2026-09-14, 1,008 stored decisions):
  - Fundamentals Analyst  21/1008 reports (2.1%) — 1.5% live, 13.9% backtest
                          (one backtest week hit 61.5%)
  - Market Analyst         1
  - News + Macro Analyst   1  (that one was LIVE — 015760.KS)
  - Sentiment Analyst      0
So the shape is predominantly, but not exclusively, the Fundamentals
Analyst — hence all four analysts are scanned, not just that one.

The five patterns below are the EXACT definition used for the 2026-08-28 and
2026-09-14 measurements. Do not "improve" them: the entire value of this
guard is that its counts stay comparable with those two baselines.

CRITICAL false-positive constraint: this guard deliberately does NOT key on
"Important Note" / "Clarification" / similar openers, even though every
observed refusal happens to use one. A prior investigation established that
live reports frequently open with "**Important Note:**" carrying a perfectly
LEGITIMATE data-quality caveat ("reported financials show zero values",
"brokerage vs consolidated figures disagree") — that is proper analyst
behaviour, not a refusal. Only the five refusal patterns decide.

A matched report is labelled `refusal_total` (the model said it cannot and
essentially stopped) or `refusal_hedged` (it disclaimed and then analysed
anyway). That label is ONE PER REPORT, derived from the first refusal marker
in the text — see `_HEDGED_TRAILING_CHARS` — never one per matched span.

Detect-only, exactly like `app.agents.technical_claim_guard` and
`app.agents.business_description_guard`: pure regex, no I/O, no LLM call, no
text mutation. A refusal is semantically a degraded analyst, but this round
is deliberately observation-only — nothing here feeds `degraded_analysts`,
`analysts_failed` or the report-cache gate, because doing so would change
behaviour and break comparability with the very baselines it measures."""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# The five refusal markers, verbatim from the 2026-08-28 / 2026-09-14
# measurement definition. Case-insensitive; each is deliberately narrow
# enough that an ordinary analyst caveat cannot trip it.
#   - "as an ai (model|assistant)" — the canonical opener.
#   - "training cut-off"          — the hyphenated spelling the model uses.
#   - "my last update"            — "beyond my last update" variant.
#   - "cannot provide specific financial" — the refusal's operative clause.
#   - "present day.{0,6}2024"     — "present day (2024)"-shaped date anchoring
#                                   to a stale year.
_REFUSAL_PATTERNS = (
    re.compile(r"as an ai (model|assistant)", re.IGNORECASE),
    re.compile(r"training cut-off", re.IGNORECASE),
    re.compile(r"my last update", re.IGNORECASE),
    re.compile(r"cannot provide specific financial", re.IGNORECASE),
    re.compile(r"present day.{0,6}2024", re.IGNORECASE),
)

# LABEL-ONLY threshold, in characters of text following the end of the
# sentence carrying the report's FIRST refusal marker — the tier is one per
# report, not one per matched span (a later marker near the end of a long
# hedged report would otherwise be labelled `refusal_total` in the same
# report, two contradictory answers to the one question the tier exists to
# answer). Picked from the observed examples: a total refusal stops within a
# line or two of the disclaimer, whereas a hedged one continues into a full
# report (one real case literally continues "The following analysis...").
# NOTHING branches on this value — it only chooses between the
# `refusal_total` and `refusal_hedged` labels in the finding string, so
# mislabeling a borderline case costs a label and nothing else.
_HEDGED_TRAILING_CHARS = 400

# Cap on the quoted span in a finding, so one refusal preamble cannot flood
# the transcript — same intent as the sentence-bounded spans in
# `technical_claim_guard`, but here the "sentence" can be very long.
_SPAN_CAP = 120

_TIER_TOTAL = "refusal_total"
_TIER_HEDGED = "refusal_hedged"

_WHITESPACE_RE = re.compile(r"\s+")


def _sentence_end(text: str, end: int) -> int:
    """Index just past the end of the sentence containing `text[:end]` — the
    nearest `.` at or after `end`, or the end of the text. Used both to bound
    the quoted span and to measure how much analysis follows the disclaimer.
    """
    right = text.find(".", end)
    return right + 1 if right != -1 else len(text)


def _quoted_span(text: str, start: int, sentence_end: int) -> str:
    """The sentence containing the match, delimited by the nearest `.` before
    it (or the text start) and `sentence_end`, whitespace-collapsed to a
    single line and capped at `_SPAN_CAP` characters."""
    left = text.rfind(".", 0, start)
    left = left + 1 if left != -1 else 0
    span = _WHITESPACE_RE.sub(" ", text[left:sentence_end]).strip()
    if len(span) > _SPAN_CAP:
        span = span[:_SPAN_CAP].rstrip() + "..."
    return span


def scan_refusal(text: str, *, agent_name: str) -> list[str]:
    """Findings, one per distinct refusal-preamble sentence found in `text`
    (an analyst's own report — the four analysts only; downstream
    researchers/debaters/trader/PM are deliberately out of scope, there is no
    measured evidence for them yet).

    Each finding is `"{agent_name}: {tier} — '{span}'"`, following
    `technical_claim_guard`'s convention. The tier is computed ONCE PER
    REPORT, from the sentence containing the EARLIEST refusal marker in the
    text, and the same tier is stamped on every finding of that call:

    - `refusal_total`  — under `_HEDGED_TRAILING_CHARS` of text follows that
      first matched sentence: the model said it cannot, and essentially
      stopped.
    - `refusal_hedged` — more than that follows: it wrote the disclaimer and
      then produced analysis anyway.

    The tier is a LABEL only; nothing in the pipeline branches on it.

    `[]` for a clean report, and for the legitimate `**Important Note:**`
    data-quality caveat that shares the refusal's opener but none of its five
    markers. Detect-only — no control-flow change, no LLM call, no text
    mutation. Several markers commonly co-occur inside ONE disclaimer
    sentence; those collapse to a single finding (dedup by span), same as
    `technical_claim_guard.scan_technical_claims`."""
    if not text:
        return []

    matches = sorted(
        (m.start(), m.end()) for pattern in _REFUSAL_PATTERNS for m in pattern.finditer(text)
    )
    if not matches:
        return []

    # ONE tier per report, derived from the EARLIEST match in document order
    # (not the first the pattern loop happens to reach — pattern order is not
    # document order): "did the report continue after the model FIRST said it
    # couldn't?". Computing it per span would let one report emit both labels
    # at once — a later marker near the end of a long hedged report has little
    # text after it — which would corrupt the very count the tier exists for.
    first_sentence_end = _sentence_end(text, matches[0][1])
    tier = (
        _TIER_HEDGED
        if len(text) - first_sentence_end >= _HEDGED_TRAILING_CHARS
        else _TIER_TOTAL
    )

    findings: list[str] = []
    seen: set[str] = set()
    for start, end in matches:
        span = _quoted_span(text, start, _sentence_end(text, end))
        if span in seen:
            continue
        seen.add(span)
        findings.append(f"{agent_name}: {tier} — {span!r}")

    if findings:
        logger.warning(
            "refusal_guard: %s wrote an 'as an AI model / past my training cut-off' "
            "refusal preamble instead of analysing (2026-09-14 measured shape) — %s",
            agent_name, findings,
        )
    return findings
