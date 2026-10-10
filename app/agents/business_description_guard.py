"""Detect-only recorder for a report that, after `identity_gate` already
corrected a misnamed company, goes on to describe what that company DOES
(its industry/sector/products/business model) in a way inconsistent with the
company's real registered name — i.e. the hallucinated wrong name didn't just
corrupt the label, it seeded an invented business profile.

Built for the 2026-09-07 live incident, decision id=1237 (294630.KS/서남): the
Market Analyst's report got an `identity_gate` correction banner (its Korean
romanized name "Seonam" was mangled into "Seonam Pharmaceutical Co., Ltd." —
a hallucinated industry suffix, not a real company). That banner's old
wording told downstream readers the report's CONTENT was still trustworthy
("disregard the wrong name only, not the content") — it wasn't: the same
report's own Risk Factors section then said "As a pharmaceutical firm,
regulatory changes..." directly contradicting the report's own correct
sector classification ("전기·전자"/Electronics&Electrical) stated elsewhere
in the same document. This propagated into a risk-debate turn that
fabricated an even more elaborate wrong business profile ("pharmaceutical/
construction hybrid").

This is NOT a cross-company identity leak (no real company named "Seonam
Pharmaceutical" exists) — it's generic plausible-sounding invention, enabled
by the banner's own overly-broad "content is fine" claim. `identity_gate`'s
banner wording was fixed separately (2026-09-07) to stop making that claim;
this module is the detect-only measurement of how often the underlying
failure shape (a fabricated industry self-description) still occurs. Live
measurement: 1/259 domestic decisions over a 21-day window (0.39%), with 6/7
similar self-descriptions elsewhere being correct — a small, narrow,
detect-only scope is the right calibration here, same as
`app.agents.technical_claim_guard` and `app.agents.halt_status_guard`.

Deliberately conditioned on an identity violation having already fired —
this only ever scans a report `identity_gate` already flagged (via its
general-case `_BANNER`, never the `ForeignListingName`/`UNNAMED` special
cases — those are different failure shapes with their own banners), never
the ~250+ clean reports each cycle. Zero LLM calls, zero control-flow
effect — mirrors `technical_claim_guard`/`halt_status_guard`'s pure
regex/string shape exactly."""

from __future__ import annotations

import logging
import re

from app.agents.identity_gate import UNNAMED, ForeignListingName
from app.agents.market_analyst import _SECTOR_EN_KEYWORDS

logger = logging.getLogger(__name__)

# Every English sector/industry keyword `_reaudit_sector_attribution` already
# watches for, flattened to a single set — reused (not duplicated) so the two
# stay in sync, per the design's explicit recommendation.
_ALL_EN_SECTOR_KEYWORDS = frozenset(
    kw for keywords in _SECTOR_EN_KEYWORDS.values() for kw in keywords
)

# EN sector keyword -> Korean sector token(s) that would make that keyword a
# LEGITIMATE self-description of the real registered company name, not a
# hallucination — e.g. 금호전기 containing "전기" legitimately produces
# "Electric"/"electronics", 하이스틸 containing "스틸" legitimately produces
# "steel", JW신약 containing "신약" (new-drug/pharma concept) legitimately
# produces "pharmaceutical", 헥토파이낸셜 containing "파이낸셜" legitimately
# produces "financial". Deliberately a SEPARATE table from
# `_SECTOR_EN_KEYWORDS` (that one maps a KRX sector NAME to keywords used to
# describe the whole market's sector ranking; this one maps a keyword to
# Korean substrings that can legitimately appear inside a single company's
# own NAME) — conflating them would either miss real amnesty cases or grant
# amnesty too broadly.
_EN_TO_KR_AMNESTY = {
    "pharmaceutical": ("제약", "신약", "바이오"),
    "pharma": ("제약", "신약", "바이오"),
    "electric": ("전기",),
    "electronics": ("전기", "전자"),
    "electrical": ("전기", "전자"),
    "construction": ("건설",),
    "builder": ("건설",),
    "chemical": ("화학",),
    "steel": ("스틸", "철강"),
    "metal": ("스틸", "철강", "금속"),
    "financial": ("파이낸셜", "금융"),
    "finance": ("파이낸셜", "금융"),
    "banking": ("은행", "금융"),
    "bank": ("은행",),
    "insurance": ("보험",),
    "securities": ("증권",),
    "brokerage": ("증권",),
}

_SELF_DESC_TEMPLATE = (
    r"(?:as|is|being)\s+an?\s+(?:\w+\s+){{0,3}}{keyword}\w*\s+"
    r"(?:firm|company|manufacturer|producer|player|business|maker)"
)


def _quoted_span(text: str, start: int, end: int) -> str:
    """The sentence containing `text[start:end]`, delimited by the nearest
    `.` on each side (or the text boundary) — copied verbatim from
    `app.agents.halt_status_guard._quoted_span` / `technical_claim_guard`'s
    own copy of the same helper; see either's docstring for why a sentence-
    bounded span (not a fixed-radius window) is required for this call's
    dedup to be stable."""
    left = text.rfind(".", 0, start)
    left = left + 1 if left != -1 else 0
    right = text.find(".", end)
    right = right if right != -1 else len(text)
    return text[left:right].strip()


def _is_amnestied(keyword: str, company_name: str) -> bool:
    """True if `keyword` (an English sector/industry word found in the
    offending/hallucinated name) is actually justified by the REAL
    registered `company_name` — i.e. a corresponding Korean sector token for
    that keyword appears as a substring of the real name. See
    `_EN_TO_KR_AMNESTY`'s own comment for the reasoning and examples."""
    kr_tokens = _EN_TO_KR_AMNESTY.get(keyword.lower())
    if not kr_tokens:
        return False
    return any(tok in company_name for tok in kr_tokens)


def scan_business_description(
    text: str, *, offending_name: str, company_name: str, agent_name: str
) -> list[str]:
    """Findings, one per business/industry self-description in `text` that
    is inconsistent with `company_name` and traces to an English sector/
    industry keyword present in `offending_name` (the wrong name
    `identity_gate` already flagged this cycle) but NOT amnestied by the
    real name.

    `[]` immediately when there is nothing to check: `offending_name` empty,
    the `UNNAMED` sentinel (a name omission, not a substitution — nothing was
    invented to seed a business-profile hallucination), or a
    `ForeignListingName` instance (a genuinely different real company on a
    different exchange — its business description IS that other company's,
    a different, already-handled failure shape, not this module's target).

    Detect-only — no control-flow change, no LLM call. Dedups identical
    spans within one call, same contract as `technical_claim_guard`/
    `halt_status_guard`."""
    if not offending_name or offending_name == UNNAMED or isinstance(offending_name, ForeignListingName):
        return []
    if not text:
        return []

    findings: list[str] = []
    seen: set[str] = set()
    offending_cf = offending_name.casefold()

    for keyword in _ALL_EN_SECTOR_KEYWORDS:
        # Substring, not exact-token, containment — the offending name is
        # often a fused/compound coinage ("Haisteel Co., Ltd.") rather than
        # cleanly space-separated words, so a keyword can appear INSIDE a
        # token rather than as one.
        if keyword not in offending_cf:
            continue
        if _is_amnestied(keyword, company_name):
            continue
        pattern = re.compile(_SELF_DESC_TEMPLATE.format(keyword=re.escape(keyword)), re.IGNORECASE)
        for hit in pattern.finditer(text):
            span = _quoted_span(text, hit.start(), hit.end())
            if span in seen:
                continue
            seen.add(span)
            findings.append(
                f"{agent_name}: {span!r} — business/industry description inconsistent "
                f"with registered name {company_name!r}"
            )

    if findings:
        logger.warning(
            "business_description_guard: %s's report described a business/industry "
            "inconsistent with the registered company name %r after an identity "
            "violation (id=1237 shape) — %s",
            agent_name, company_name, findings,
        )
    return findings
