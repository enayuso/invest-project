"""Detect-only recorder for a report/debate turn that states a "derived"
numeric value (a value that should be a simple function of two other, nearby
"ground" values it cites) which is wildly inconsistent with what those two
ground values actually compute to.

Built for the 2026-09-09 domestic AI-decision audit's last P1 item ("산술
오류 다발" / arithmetic errors), same 12-decision sample as
`app.agents.halt_status_guard` and `app.agents.absence_reframing_guard` (ids
1348-1359). Two concrete, hand-verified instances:

  - id=1357 (032820.KS), Fundamentals Analyst report: "**Price/Earnings Ratio
    (PER):** **~2,728x**. This extremely high multiple reflects the tiny EPS
    (₩62) relative to the stock price (~₩13,640)." PER should be price ÷ EPS
    = 13,640 / 62 = 220.0x, but the report states 2,728x (off by ~12.4x).
    This wrong 2,728x figure then propagated into 15+ other mentions across
    every downstream node in that decision — that propagation is NOT in
    scope for this guard, which only catches the value wherever it is
    independently self-contained with its own EPS+price citation, which is
    how it originally appeared.

  - id=1353 (088280.KS), a risk-debate turn rebutting a bull argument: "The
    central pillar of the Buy case is a stop-loss at 2,400 KRW. The
    proponent claims this represents a ~3.5% risk distance from the current
    price of 2,815 KRW. This is a severe calculation error." The risk
    distance should be (price - stop) / price = (2,815 - 2,400) / 2,815 =
    14.7%, but the claimed figure is ~3.5% (off by ~4.2x). This specific
    instance is a self-correcting rebuttal (the same sentence goes on to
    call it "a severe calculation error") — the guard still fires on it
    regardless, the same way `app.agents.unit_scale_guard` records a
    violation even when a correction is applied elsewhere: this guard is a
    pure MEASUREMENT of how often the underlying arithmetic error occurs at
    all, not an accusation that it went unnoticed.

This is a NEW guard shape, distinct from every other narrative-quality guard
in this codebase: `app.agents.halt_status_guard`,
`app.agents.absence_reframing_guard`, `app.agents.staleness_narrative_guard`,
and friends are all PHRASING-PATTERN guards (text A implies/echoes text B).
This is a NUMERIC-DERIVATION check: it extracts two ground values and one
stated derived value that co-occur near each other in the SAME piece of
text, recomputes the derived value from the two ground values via a known
formula, and flags when the stated value is off from the recomputed value by
a large factor. The closest existing precedent for "recompute and compare
numbers" is `app.agents.ratio_label_guard` (self-refuting-by-definition
check) and `app.agents.unit_scale_guard` (10x magnitude comparison against a
supplied ground-truth set) — neither does this exact two-ground-values/
one-derived-value shape, so this module borrows only their house style
(module docstring conventions, the `_to_float` helper, a logging convention),
not their literal logic.

Given there are only two known concrete instances (a much smaller sample
than the other guards in this audit were tuned against), the extraction
patterns are deliberately kept close to the two observed phrasings,
generalized only lightly (tolerating "PER of X" as well as "PER: X"/"PER
~X", and "$" as well as "₩") — this is NOT a broad general-purpose "any
three numbers near each other" scanner, and the mismatch threshold is
deliberately kept high (>= 2x) so a normal, legitimate rounding difference
(e.g. trailing-vs-forward EPS) never trips it. A false positive here (a
correct statement flagged as wrong) is worse than a missed case.

Detect-only / narrative-quality — no control-flow change, no LLM call, NOT
cache-gating. Multi-node like `app.agents.halt_status_guard.scan_halt_claims`
(scans every analyst report, the research plan, the risk debate, and the
PM's own final decision text) — see `app.orchestrator.pipeline`'s wiring."""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

_MISMATCH_THRESHOLD = 2.0  # stated value must be off from the recomputed value by at least this factor (either direction) to fire — both known instances (12.4x, 4.2x) clear this with large margin; do not lower it.
_WINDOW = 200  # chars searched on EACH SIDE of the "derived value" match for its two ground-value citations — the two real incidents split the derived value and its ground values across adjacent sentences (". "), not always the same sentence, so this must NOT be sentence-bounded.

_MONEY_NUM = r"(\d[\d,]*(?:\.\d+)?)"

# --- Check 1: PER should equal price ÷ EPS ---
# id=1357 (032820.KS): "**Price/Earnings Ratio (PER):** **~2,728x**. This
# extremely high multiple reflects the tiny EPS (₩62) relative to the stock
# price (~₩13,640)." -> 13,640 / 62 = 220.0x, stated 2,728x (12.4x off). The
# literal gap between "PER" and "~2,728" is "):** **~" (markdown-bold
# markup), so the gap below is 30 chars, not 20, to tolerate that plus a
# margin for other punctuation-heavy phrasings.
_PER_STATED_RE = re.compile(
    rf"\bPER\b[^.\n]{{0,30}}?(?:of\s*)?[~≈]?\s*{_MONEY_NUM}\s*x\b",
    re.IGNORECASE,
)
_EPS_RE = re.compile(
    rf"\bEPS\b[^.\n]{{0,15}}?\(?\s*[₩\$]?\s*{_MONEY_NUM}\s*\)?",
    re.IGNORECASE,
)
_PRICE_RE = re.compile(
    rf"\bprice\b[^.\n]{{0,20}}?\(?\s*[~≈]?\s*[₩\$]?\s*{_MONEY_NUM}\s*\)?",
    re.IGNORECASE,
)

# --- Check 2: stop-loss risk distance % should equal (price − stop) ÷ price ---
# id=1353 (088280.KS): "a stop-loss at 2,400 KRW. The proponent claims this
# represents a ~3.5% risk distance from the current price of 2,815 KRW." ->
# (2,815 - 2,400) / 2,815 = 14.7%, stated ~3.5% (4.2x off).
_DISTANCE_PCT_RE = re.compile(
    rf"[~≈]?\s*{_MONEY_NUM}\s*%\s*(?:risk\s+)?distance",
    re.IGNORECASE,
)
_STOP_PRICE_RE = re.compile(
    rf"stop(?:-loss)?\s+at\s*\**\s*[₩\$]?\s*{_MONEY_NUM}",
    re.IGNORECASE,
)
_CURRENT_PRICE_RE = re.compile(
    rf"current\s+price\s+of\s*\**\s*[₩\$]?\s*{_MONEY_NUM}",
    re.IGNORECASE,
)


def _to_float(token: str) -> float:
    return float(token.replace(",", ""))


def _mismatch_ratio(stated: float, computed: float) -> float:
    """>= 1.0 always — how many times bigger the larger of (stated, computed)
    is than the smaller. `float('inf')` if computed is 0 (division guarded by
    the caller before this is ever invoked with a zero)."""
    if computed <= 0 or stated <= 0:
        return float("inf")
    ratio = stated / computed
    return ratio if ratio >= 1 else 1 / ratio


def scan_derived_value_mismatches(text: str, *, agent_name: str) -> list[str]:
    """`"AgentName: <description> — '<span>'"` per stated derived value found
    in `text` that is inconsistent (by >= `_MISMATCH_THRESHOLD`x) with what
    its own two nearby ground-value citations compute to, for two known
    templates: PER vs price/EPS (id=1357 shape), and stop-loss risk-distance
    % vs (price - stop)/price (id=1353 shape).

    `[]` for empty `text`, and `[]` whenever a derived-value mention has no
    matching pair of ground-value citations within `_WINDOW` chars on either
    side — this guard only ever fires when it can independently recompute
    the value itself, never on a bare mention with nothing to check it
    against (e.g. a lone "PER of 75.94x" with no EPS/price nearby).

    ALL ground-value candidates in the window are collected (not just the
    first of each kind), and the stated value is treated as CONSISTENT — no
    finding — as soon as ANY candidate combination reproduces it within
    `_MISMATCH_THRESHOLD`. Only when no combination reconciles does it fire,
    and then it reports the combination with the SMALLEST mismatch ratio.
    See the inline comment at each check's reconciliation for the id=1406
    false positive this closes. A
    self-correcting sentence (the text itself later calls the number a
    "severe calculation error") still fires: this is a measurement of how
    often the underlying arithmetic error occurs at all, not of whether it
    went uncaught downstream. Detect-only — no control-flow change, no LLM
    call. Dedups identical spans within one call."""
    if not text:
        return []

    findings: list[str] = []
    seen: set[str] = set()

    # Check 1: PER vs price/EPS
    for per_m in _PER_STATED_RE.finditer(text):
        stated_per = _to_float(per_m.group(1))
        window_start = max(0, per_m.start() - _WINDOW)
        window_end = per_m.end() + _WINDOW
        near = text[window_start:window_end]
        eps_vals = [_to_float(m.group(1)) for m in _EPS_RE.finditer(near)]
        price_vals = [_to_float(m.group(1)) for m in _PRICE_RE.finditer(near)]
        # EVERY candidate pair, not just the first of each kind. A passage may
        # legitimately discuss more than one basis for the same ratio, and the
        # first EPS in the window is often not the one the stated PER was
        # computed from. Real false positive (id=1406, Fundamentals Analyst):
        # the report correctly derived a trailing PER of 62.6x from FY2025 EPS
        # ₩1,147 and price ₩71,800 AND a forward PE of 6.4x from annualized
        # H1-2026 EPS ₩5,601 — but the old first-match-only lookup paired the
        # stated 62.6x with the FORWARD EPS and reported a mismatch against a
        # calculation the analyst had never made. "Any combination reconciles
        # => not a finding" is the right rule here because this guard exists to
        # measure arithmetic that is wrong on ANY reading of the author's own
        # cited numbers: if some cited pair reproduces the stated value, the
        # author demonstrably had a correct basis in hand and there is no
        # arithmetic error to record. Only when nothing reconciles do we fire,
        # and then with the smallest-mismatch pair — the most charitable
        # reading of what the author meant.
        best: tuple[float, float, float, float] | None = None  # (ratio, eps, price, computed)
        reconciled = False
        for eps_val in eps_vals:
            for price_val in price_vals:
                if eps_val <= 0 or price_val <= 0:
                    continue
                computed_per = price_val / eps_val
                ratio = _mismatch_ratio(stated_per, computed_per)
                if ratio < _MISMATCH_THRESHOLD:
                    reconciled = True
                    break
                if best is None or ratio < best[0]:
                    best = (ratio, eps_val, price_val, computed_per)
            if reconciled:
                break
        if reconciled or best is None:
            continue
        _, eps_val, price_val, computed_per = best
        span = near.strip()
        if span in seen:
            continue
        seen.add(span)
        findings.append(
            f"{agent_name}: PER stated as {stated_per:g}x but EPS ({eps_val:g}) and price "
            f"({price_val:g}) in the same passage compute to {computed_per:.1f}x — {span!r}"
        )

    # Check 2: stop-loss risk distance % vs (price - stop) / price
    for dist_m in _DISTANCE_PCT_RE.finditer(text):
        stated_pct = _to_float(dist_m.group(1))
        window_start = max(0, dist_m.start() - _WINDOW)
        window_end = dist_m.end() + _WINDOW
        near = text[window_start:window_end]
        stop_vals = [_to_float(m.group(1)) for m in _STOP_PRICE_RE.finditer(near)]
        price_vals = [_to_float(m.group(1)) for m in _CURRENT_PRICE_RE.finditer(near)]
        # Same all-candidates reconciliation as check 1 above — a passage that
        # names two different stop levels (e.g. an initial stop and a tightened
        # one) would otherwise be judged against whichever happened to appear
        # first in the window. No false positive has been observed on this
        # check yet, but it carried the identical first-match-only defect that
        # produced id=1406 on check 1, so it is fixed the same way rather than
        # left as a known landmine.
        best_stop: tuple[float, float, float, float] | None = None  # (ratio, stop, price, computed)
        reconciled = False
        for stop_val in stop_vals:
            for price_val in price_vals:
                if price_val <= 0 or stop_val <= 0 or stop_val >= price_val:
                    continue
                computed_pct = (price_val - stop_val) / price_val * 100
                if computed_pct <= 0:
                    continue
                ratio = _mismatch_ratio(stated_pct, computed_pct)
                if ratio < _MISMATCH_THRESHOLD:
                    reconciled = True
                    break
                if best_stop is None or ratio < best_stop[0]:
                    best_stop = (ratio, stop_val, price_val, computed_pct)
            if reconciled:
                break
        if reconciled or best_stop is None:
            continue
        _, stop_val, price_val, computed_pct = best_stop
        span = near.strip()
        if span in seen:
            continue
        seen.add(span)
        findings.append(
            f"{agent_name}: stop-loss risk distance stated as ~{stated_pct:g}% but stop "
            f"({stop_val:g}) and current price ({price_val:g}) in the same passage compute to "
            f"{computed_pct:.1f}% — {span!r}"
        )

    if findings:
        logger.warning(
            "derived_value_mismatch_guard: %s stated a derived value inconsistent with its own "
            "nearby ground-value citations — %s",
            agent_name, findings,
        )
    return findings
