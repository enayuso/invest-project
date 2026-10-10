"""Detect-only recorder for a Fundamentals Analyst report that presents
stale, exchange-reported financials as the company's CURRENT condition, with
no acknowledgment anywhere in the report that the underlying data is old.

Built for the 2026-09-09 live audit, decision id=1351 (198440.KS, Kangdong
C&L, 21-month-stale FY2024 financials): the report opened "As of September 9,
2026, Kangdong C&L presents a complex fundamental picture characterized by
significant operational deterioration compared to its peak years." — framing
20-plus-month-old FY2024 numbers as the company's condition AS OF TODAY, with
zero acknowledgment anywhere in the report that the data was stale. This
happened despite `app.dataflows.kis_provider.get_kr_financials()` already
injecting a `_staleness_note` warning into the financial-statements text
itself (once `months_stale >= settings.kr_financials_staleness_warn_months`)
that explicitly tells the model never to present the data as "current" /
"latest" / "recent". The model ignored the injected warning.

Contrast case, same audit day: id=1353 (088280.KS, Sonnix), same 21-month
staleness. That report's Fundamentals Analyst correctly wrote "**As of:**
September 9, 2026 / **Source Data Period:** Fiscal Year Ended December 31,
2024" and closed with "This report is based solely on the provided financial
data from KIS/KSX for the period ending December 31, 2024" — explicitly
acknowledging the data's true period throughout. This shape must NEVER be
flagged, and the whole-report staleness-acknowledgment check below exists
specifically so it isn't.

This is a narrative/decision-quality problem only, the same class as
`app.agents.halt_status_guard.scan_halt_claims` and
`app.agents.technical_claim_guard.scan_technical_claims` — detect-only, no
control-flow change, no LLM retry. It is NOT a numeric ground-truth guard:
it never checks whether a number in the report is right or wrong, only
whether the report's own framing acknowledges that the numbers it is citing
are old. A report can cite every figure correctly and still trip this if it
frames those figures as describing the company's condition today; a report
can (in principle) misstate a figure and still correctly disclose staleness,
in which case this guard has nothing to say about it — that is out of scope
here, by design."""

from __future__ import annotations

import logging
import re

from app.config import settings

logger = logging.getLogger(__name__)

_HEAD_CHARS = 500  # the Executive Summary conventionally opens the report

_MONTH_ALT = (
    r"january|february|march|april|may|june|july|august|september|october|november|december|"
    r"jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec"
)

# "As of <calendar date>" — deliberately excludes "as of the latest/most
# recent/last <period>" (a report citing ITS OWN data's period — e.g.
# id=1353's "As of the latest audited period (FY2024)" — is doing exactly
# the right thing and must never trip this).
_AS_OF_CALENDAR_DATE_RE = re.compile(
    rf"\bas of\b(?!\s+(?:the\s+)?(?:latest|most recent|last|that))\s+"
    rf"(?:{_MONTH_ALT})\.?\s+\d{{1,2}},?\s+\d{{4}}",
    re.IGNORECASE,
)

# A present-tense characterization verb shortly after the calendar-date
# clause — "As of <date>, <Company> presents/shows/reflects/demonstrates/
# exhibits/is a ...".
_VERB_WINDOW = 100
_CURRENT_TENSE_VERB_RE = re.compile(
    r"\b(presents?|shows?|reflects?|demonstrates?|exhibits?|characterizes?|is\b)",
    re.IGNORECASE,
)

# Any phrase acknowledging the data itself is old — checked over the WHOLE
# report, not just the head. Present in id=1353, absent in id=1351.
_STALENESS_ACK_RE = re.compile(
    r"months?\s+(?:old|ago|stale)|no newer|not\s+(?:been\s+)?updated|outdated|stale\s+data|"
    r"only\s+data\s+available|most recent\s+(?:period|data)\s+available|"
    r"last\s+(?:reported|available)\s+period|based solely on|"
    r"does not reflect (?:more )?recent|no more recent|"
    r"has not (?:filed|reported)|hasn'?t (?:filed|reported)|"
    r"source data period|data period|"
    rf"fiscal year end(?:ed)?\s+(?:{_MONTH_ALT})",
    re.IGNORECASE,
)


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


def scan_staleness_narrative(report: str, *, recency: dict, agent_name: str) -> list[str]:
    """`"AgentName: '<span>' — presents N-month-stale <period> data as
    current, with no staleness acknowledgment anywhere in the report"` per
    "As of <calendar date>, <Company> presents/shows/..." framing found near
    the head of `report`, but ONLY when the underlying financials are
    genuinely stale (`recency["months_stale"] >=
    settings.kr_financials_staleness_warn_months`) AND the whole report never
    acknowledges that staleness anywhere (`_STALENESS_ACK_RE`).

    `[]` immediately when `recency` is falsy or `months_stale` is `None` or
    below the warn threshold (nothing stale to misrepresent), when `report`
    is empty, or when the report acknowledges the data's true period/age
    anywhere (id=1353 shape) — the acknowledgment check runs over the WHOLE
    report before any date/verb scanning, so a single disclosure anywhere
    suppresses the entire scan for that report. Also `[]` for a calendar-date
    "as of" clause that isn't followed by a present-tense characterization
    verb within `_VERB_WINDOW` chars, and for the "as of the latest/most
    recent/last <period>" framing (a report citing its own data's period,
    which is correct and must never be flagged).

    Detect-only — no control-flow change, no LLM call; a narrative/decision-
    quality check, not a numeric ground-truth guard (see module docstring).
    Dedups identical spans within one call, same as
    `halt_status_guard.scan_halt_claims`."""
    months = recency.get("months_stale") if recency else None
    if months is None or months < settings.kr_financials_staleness_warn_months:
        return []
    if not report:
        return []
    if _STALENESS_ACK_RE.search(report):
        return []

    findings: list[str] = []
    seen: set[str] = set()
    head = report[:_HEAD_CHARS]
    for m in _AS_OF_CALENDAR_DATE_RE.finditer(head):
        window = head[m.end(): m.end() + _VERB_WINDOW]
        if not _CURRENT_TENSE_VERB_RE.search(window):
            continue
        span = _quoted_span(report, m.start(), m.end())
        if span in seen:
            continue
        seen.add(span)
        latest_label = recency.get("latest_label") or recency.get("latest_period") or "?"
        findings.append(
            f"{agent_name}: {span!r} — presents {months}-month-stale {latest_label} data as "
            f"current, with no staleness acknowledgment anywhere in the report"
        )

    if findings:
        logger.warning(
            "staleness_narrative_guard: %s presented %d-month-stale financials as current "
            "with no staleness acknowledgment (id=1351 shape) — %s",
            agent_name, months, findings,
        )
    return findings
