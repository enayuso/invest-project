"""Detect-only recorder for a report/debate turn that labels its own price
claim "Verified" (echoing this pipeline's own authoritative-price framing —
see `app.dataflows.session_state.format_verified_snapshot`'s "VERIFIED
SNAPSHOT" header, which `app.agents.market_analyst`/`app.agents.
news_macro_analyst` both inject into their prompts verbatim as "Verified
price snapshot (authoritative)") while stating a price that does NOT match
the actual verified snapshot this cycle.

Built for the 2026-09-08 live incident, decision id=1314 (033790.KS/피노):
the News + Macro Analyst — despite being handed the SAME real verified
snapshot as the Market Analyst (both report the stock trading around
9,120-9,760 KRW that session) — wrote "Verified Close (Sept 6): 1,330 KRW
(-2.22%)" and "Verified Close (Sept 7): 1,290 KRW (-3.00%)", then built an
entire bearish thesis around a fabricated "dilutive private placement at
~1,300 KRW/share" narrative — a price universe roughly 7x away from the
real one, self-labeled with the exact word this codebase reserves for
actually-verified data. The Fundamentals Analyst's report for the SAME
cycle correctly used the real ~9,000s price range and real financials, so
this was not a whole-report wrong-company mixup (identity_gate's alias
matching found nothing to flag here — the report never names a different
company, it just states facts belonging to one) — it is a narrower,
narrative-only fabrication that happens to self-label as verified.

Calibrated against 250 live domestic decisions (2026-09-08): comparing
every "Verified Close/Price/Snapshot: ₩N" claim in the News + Macro
Analyst's report against the Market Analyst's own stated current price for
the same cycle (a reliable proxy ground truth — Market Analyst is
separately instructed to only ever quote its injected verified snapshot)
found 22 comparable candidates; 20 matched EXACTLY (ratio 0.0, KRW-for-KRW)
and the only 2 mismatches were id=1314's two spans above (ratio ~0.85). No
false positives, no near-misses — the failure mode this guards against is
rare and produces a huge, unambiguous gap when it happens, not a subtle
one.

Same "detect-only, no control-flow change" pattern as
`app.agents.halt_status_guard.scan_halt_claims` and
`app.agents.price_limit_guard.scan_price_limit_claims`: a narrative/
decision-quality problem only, NOT an execution-safety problem — order
submission and stop-loss monitoring both independently re-check the real
broker price at the time of action, so a fabricated "verified" price here
cannot itself route an order at a fabricated level.

Deliberately anchored on the literal word "Verified" (or "verified") next
to the claim, not on every price mention in a report — an EARLIER
calibration pass matching any KRW figure near a date reference produced
heavy false positives from historical/multi-day price context ("up from
₩3,000 in January" is a legitimate reference, not an error). Requiring the
model to have ITSELF invoked this codebase's reserved "verified" framing is
what keeps this narrow and self-inflicted: the model is claiming an
authority for the number that the pipeline never gave it."""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# The real ground-truth price out of `format_verified_snapshot`'s own output
# shape — `close=` for a completed session, `last=` for one still in
# progress (see that function's docstring for why the label changes).
_GROUND_TRUTH_RE = re.compile(r"\b(?:close|last)=([\d.]+)")

# A signed decimal KRW figure, matching this codebase's other numeric-guard
# conventions (app.agents.unit_scale_guard's `_NUM` family) minus the sign
# group — a price level is meaningful in magnitude regardless of a leading
# +/- token, and none of the observed incident shapes carry one anyway.
_NUM = r"(\d[\d,]*(?:\.\d+)?)"

# Requires the number to sit immediately behind a currency marker (₩ prefix
# or KRW/원 suffix) — NOT just "the nearest number after the word Verified",
# which an earlier calibration pass found usually grabs a DATE instead
# ("Verified Close (Sept 6): 1,330 KRW" — the nearest bare number is "6",
# the date, not the price two tokens later). The currency anchor is what
# lets this skip straight to the actual claim.
_VERIFIED_PRICE_RE = re.compile(
    rf"Verified\s+(?:Close|Price|Snapshot)[^.\n]{{0,80}}?(?:₩\s?{_NUM}|{_NUM}\s?(?:KRW|원))",
    re.IGNORECASE,
)

# How far a claimed price may sit from the real one before this fires — a
# same-session/multi-day drift can legitimately be large (KRX's own ±30%
# daily band, compounded over a couple of sessions), but nothing remotely
# close to id=1314's ~7x gap; 50% clears every genuine same-cycle mention
# in the calibration set (all matched exactly, ratio 0.0) with room to
# spare before the real incident's ~0.85.
_TOLERANCE_RATIO = 0.5


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


def scan_verified_price_claims(text: str, *, verified_snapshot: str, agent_name: str) -> list[str]:
    """`"AgentName: claimed 'Verified' price <N> vs real verified snapshot
    <M> — '<span>'"` per self-labeled "Verified Close/Price/Snapshot" claim
    in `text` whose stated KRW figure differs from the real
    `verified_snapshot` for this cycle by more than `_TOLERANCE_RATIO`.

    `[]` immediately if `verified_snapshot` carries no parseable
    `close=`/`last=` ground truth (nothing to check against — fail open,
    same convention as every other ground-truth-gated guard in this
    package), and `[]` for empty `text`. Detect-only — no control-flow
    change, no LLM call. Dedups identical spans within one call, same as
    `app.agents.halt_status_guard.scan_halt_claims`."""
    if not text or not verified_snapshot:
        return []
    gt_match = _GROUND_TRUTH_RE.search(verified_snapshot)
    if not gt_match:
        return []
    ground_truth = float(gt_match.group(1))
    if ground_truth <= 0:
        return []

    findings: list[str] = []
    seen: set[str] = set()
    for m in _VERIFIED_PRICE_RE.finditer(text):
        claimed_str = m.group(1) or m.group(2)
        claimed = float(claimed_str.replace(",", ""))
        if claimed <= 0:
            continue
        if abs(claimed - ground_truth) / ground_truth <= _TOLERANCE_RATIO:
            continue
        span = _quoted_span(text, m.start(), m.end())
        if span in seen:
            continue
        seen.add(span)
        findings.append(
            f"{agent_name}: claimed 'Verified' price {claimed:,.2f} vs real verified "
            f"snapshot {ground_truth:,.2f} — {span!r}"
        )

    if findings:
        logger.warning(
            "verified_price_guard: %s self-labeled a price claim 'Verified' that contradicts "
            "the real verified snapshot this cycle (id=1314 shape) — %s",
            agent_name, findings,
        )
    return findings
