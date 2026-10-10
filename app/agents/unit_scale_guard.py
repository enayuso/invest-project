"""Catches a Fundamentals Analyst report that restated a KIS-sourced 억원
figure as "N billion KRW" 10x too large (or too small) — a self-inflicted
unit-conversion error, not a data problem.

Built for the 2026-08-28 domestic AI audit: 70 reports across 20 tickers
mislabeled a correctly-supplied 억원 figure as "billion KRW" off by exactly
10x (현대차: "951,542 billion KRW" for a true 95,154.2 billion; 두산에너빌리티:
"89,859 billion KRW (8.99 trillion KRW)", the two numbers in the same
sentence already 10x apart). Root cause: `app.dataflows.kis_provider._eok`
used to give a billion-KRW gloss only below 1조 — at or above it, the model
had no supplied "billion" answer to quote and derived one itself, and every
observed derivation just carried the 억원 mantissa straight over. `_eok` now
gives both units unconditionally (see its own 2026-08-28 addendum), which
should prevent most of this at the source; this module is the detection net
for whatever a model still gets wrong anyway (a self-computed figure not
lifted from the supplied text, a stray re-derivation, etc.) — same
detection-net relationship `app.agents.identity_gate` has to its own
source-level fixes.

Two derived-harm cases from the same incident (012330.KS mislabeled "Samsung
Electronics", 032830.KS's supplied revenue judged "too large to be real")
are explicitly OUT of scope here: 012330's mislabeling came from the model's
own (wrong) prior knowledge about company scale, not from a numeric
restatement this module could compare against a supplied ground truth, and
belongs to `app.agents.identity_gate` /
[[ticker-identity-misattribution-2026-08-20]] instead. This module only
corrects the number; `enforce_unit_scale`'s banner also tells downstream
stages not to draw an identity conclusion from a magnitude mismatch, as
cheap cross-coverage for exactly the 032830 shape.

2026-09-01 addendum (first real-account KRX cycle, deep audit of decisions
id=1156/1157/1158). Two blind spots surfaced that let `unit_scale_violations`
come back `[]` on genuinely broken reports:
  - 000270.KS/기아 (id=1158): NINE 억원 figures were each mislabeled 10x large
    ("₩625,389 billion" for a true 62,538.9 billion), and the guard caught
    ZERO of them. `_BILLION_RE`/`_TRILLION_RE` required the currency token
    AFTER the unit word (`951,542 billion KRW`); that report wrote `₩` as a
    PREFIX throughout, and a `₩`-prefixed number is invisible to the old
    patterns. 009150.KS/삼성전기 (id=1157) caught 13 only because that run
    happened to pick the trailing word order. So `unit_scale_violations=[]`
    was never evidence of a clean report — it conflated "clean", "nothing to
    check", and "regex blind to this notation". `scan_unit_scale` /
    `UnitScaleScan` now carry the POSITIVE signal an empty list cannot.
  - Notations still uncovered before this change and now handled: currency as
    a prefix (`₩N billion`, `KRW N billion`); a bare `B`/`T` suffix behind an
    explicit `₩`/`KRW` (`~₩48T`, the span that flipped id=1158's risk debate
    to Buy); `조`/`조원` (Korean trillion-won, no currency token needed); and
    — flag-only — `$N trillion` (a KRW budget rendered as USD trillions,
    `usd_check=True`) and a power-of-10 ladder near-miss (`magnitude_suspect`).
  - `_REL_TOL = 0.005` was too tight for a rounded span: id=1158's true
    48,337억원 renders as 48.337조 and the report's `~₩48T` is 0.70% off it, so
    even a matching regex would not have flagged it. `_APPROX_REL_TOL`, scoped
    to low-precision / approximation-marked spans and applied symmetrically,
    fixes that.
  - The correction marker re-contained the offending span verbatim, so the
    guard was not idempotent — a debate turn quoting a corrected Fundamentals
    sentence would re-flag and nest. `_CORRECTION_MARKER_RE` now skips any
    candidate overlapping an existing marker.

2026-09-02 addendum (live decision id=1168, 024060.KS, a micro-cap). Same
report, two failures on the SAME underlying bug, both traced to a sign
asymmetry: the code compared an unsigned `av = abs(v)` (the report's own
mantissa) against a SIGNED ground truth `e` (from `eok_values`, which carries
`_EOK_RE`'s sign). Neither `_isclose(av, e / 10, tol)` nor `_isclose(av, e,
tol)` can ever succeed when `e` is negative, because `av` is never negative.
  - Bear wrote "-3 billion KRW" for an operating loss Fundamentals correctly
    gave as "-3억 원 (-0.3 billion KRW)" — a real 10x inflation. The sign bug
    blocked the match; the guard silently passed it as `unverified` — a
    THIRD failure mode this module can produce, alongside the two already
    documented above (flagged-wrongly / missed-entirely). `unverified` looks
    identical to "nothing here to check" from the caller's side, which is
    exactly why `UnitScaleScan.suspicious` exists — but id=1168's report
    still didn't trip it, because elsewhere in the SAME report...
  - ...a correctly-stated "-9억 원 (-0.9 billion KRW)" was FALSELY flagged as
    `magnitude_suspect`: the sign bug also blocked this figure from being
    *accepted*, so it fell through to the power-of-10 ladder, where it
    coincidentally collided with an unrelated positive ground-truth value
    (900, from revenue) at the `k=+2` rung.
  - Fix: every ground-truth comparison in the billion/trillion branches now
    reads `abs(e)`, and `_magnitude_suspect` does the same.
  - A second, independent defect: the pre-existing `e / 10 >= 1.0` guard on
    the Tier-1 auto-correct branch (undocumented, born in this file's first
    commit, never revisited — traced through git history) blocked correction
    of any ground truth under 10억원 (1 billion KRW), which is exactly where
    micro-cap P&L figures like id=1168's live. Fixing only
    the sign bug would still miss "-3 billion KRW" (true value 0.3 < the 1.0
    floor). The floor itself was not a deliberate design decision and is not
    removed outright — under it, a 1-2 significant-digit mantissa collides
    with an unrelated supplied value too often for a bare match alone to be
    trusted. Instead: `_BILLION_CORRECT_FLOOR` names the threshold explicitly,
    and below it correction proceeds only with corroboration — `_sign_agrees`
    — since the sign is free evidence a genuine mislabel always carries
    through and a coincidental collision usually does not. Above the floor,
    sign agreement is NOT required (`tests/agents/test_numeric_guard.py`'s
    "-785.2 billion KRW" against a positive "loss=785억원" is a legitimate
    sign-convention difference, not an error, and must still correct).
  - A THIRD, hidden interaction: a model that writes the proper Unicode minus
    U+2212 ("−") — which `_NUM`'s old `-?` (ASCII-only) does not match — was
    previously harmless, because nothing downstream tried to rewrite that
    span. The moment the floor fix ships, it becomes actively corrupting: a
    "−3 billion KRW" span would get matched (variously) by the surrounding
    currency/unit gate without capturing the "−" into the numeric group, and
    the replacement text (freshly signed off `sign`) would be spliced in
    right after the untouched, uncaptured "−" — producing a double-minus
    "−-0.3 billion KRW". `_SIGN` (`[-−]?`) fixes this by making `_NUM` and
    `_EOK_RE` capture the Unicode minus as part of the sign group; `_to_float`
    and `_sig_digits` normalize/strip it accordingly. This is why the sign
    fix, the floor fix, and the Unicode-minus fix all had to ship in ONE
    commit: any one or two of them alone either fail to fix id=1168's defects
    or actively make the double-minus corruption worse. En dash (–, U+2013)
    and em dash (—, U+2014) are deliberately excluded from `_SIGN`: "3–5
    billion KRW" is a range, and reading "–" as a sign would silently turn a
    correct positive 5 into -5.
  - A fourth, smaller, independent gap: bare-suffix currency markers ("3B
    KRW", "176B KRW", "48T KRW" — currency AFTER the number) were not
    detected at all (`candidates_examined: 0`), because the existing regex
    only accepted a currency-PREFIX form ("₩3B", "KRW 3B"). Fixed in the same
    commit (same regex family) with a `{_NUM}\\s*[Bb]\\s+{_KRW_SUFFIX}` /
    `...[Tt]...` alternative, gated by the trailing KRW/won/₩ token exactly as
    strongly as the existing prefix form is gated by its leading token.

2026-09-03 addendum (live decision id=1207, 270520.KS/앱튼, KOSDAQ bio). A
Fundamentals Analyst report was, cover to cover, about a DIFFERENT, foreign-
listed company (Appen Ltd. / APPN, NASDAQ) — every figure USD-denominated,
not one 원/억원/₩ figure anywhere, while the exchange handed the model a full
억원 financial block it made zero use of. `identity_gate` Detector C
(`_foreign_listing_offender`) is the primary net for that shape; this module
adds a currency-composition net at the numeric layer, opt-in via
`currency_check` (Fundamentals Analyst only). New flag-only kind
`currency_inverted` (Tier 2, `expected=None`): a report on a KRX name that
states many USD figures and no KRW figure at all is most likely not a unit
error that can be rescaled — the company itself may be wrong — so the banner
tells downstream stages to discard every figure in it. The `len(eok_values)
>= _MIN_GT_FOR_CURRENCY` premise gate stands in for Detector C's ETF carve-
out at this layer (this module never sees the alias set, so it cannot
re-implement the brand-token check). Banner wording is deliberately free of
`₩` / literal `KRW` / `Korean won` / digit-adjacent 원·억원·조원 / digit+won /
`$digit` so it never trips `identity_gate._KRW_AMOUNT_RE` and silently
disarms Detector C.

2026-09-15 addendum (precision fix for `magnitude_suspect`, no scope change
to any other tier). A 60-day production scan found 131 `magnitude_suspect`
occurrences across 41 decisions (79 distinct (ticker, figure) pairs); a
hand-read sample of 4 found only 1 true positive (id=1436, 024060.KS) against
2 clear false positives (id=1433, 004310.KS: a hypothetical range whose "-"
was a misread range hyphen; id=1441, 278650.KS: the model's own annualized
arithmetic) and 1 undetermined (id=1409, 316140.KS) — precision far below the
9/9 + 13/13 record that justified the existing auto-correcting tiers. Fixed
by exempting those three false-positive classes (range-hyphen misparse,
hypothetical/scenario framing, model-derived arithmetic) from
`magnitude_suspect` ONLY — ported from `period_mismatch_guard`, which solved
the same three classes for its own candidates. `billion_10x`, `trillion_10x`,
`usd_unit_suspect`, and Tier 1 exact-match are untouched by this fix. This
tier is STILL flag-only after this change; promotion to auto-correction
remains BLOCKED until a post-fix precision measurement on fresh live data
justifies it (see `_magnitude_suspect`'s own docstring addendum).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# 2026-09-02: a model writes U+2212 (−, the typographic minus) where `_eok`
# wrote ASCII "-". Accepted here in the ONE position where a dash is
# unambiguously a sign — immediately before the digits of a magnitude the
# surrounding pattern has already gated on a currency/unit token. En dash
# (U+2013) and em dash (U+2014) are deliberately NOT accepted: "3–5 billion
# KRW" is a range, and reading the "–" as a sign would turn a correct
# positive 5 into -5.
_SIGN = r"[-−]?"

# Every 억원 figure supplied to the model, anywhere in the KR financial data
# block — this IS the ground-truth set `check_unit_scale` compares the
# report's own "billion"/"trillion" mentions against. Matches `_eok`'s own
# Korean-unit rendering (`kis_provider.py`), never the English gloss it also
# emits (that would just be comparing the model's arithmetic to itself).
_EOK_RE = re.compile(r"(" + _SIGN + r"[\d,]+(?:\.\d+)?)\s*억원")

# A report's own "N billion KRW" / "N trillion KRW" mentions — `bn` and a
# bare `won`/₩ are accepted since a model paraphrases both. Deliberately NOT
# anchored to a KIS financial-statement label (revenue=, EBITDA=, ...): the
# 2026-08-28 incidents restated a supplied figure inside free prose ("Revenue
# reached 89,859 billion KRW"), not just inside the structured inline labels.
#
# Shared building blocks (2026-09-01): `_KRW_PREFIX` is the notation
# 000270.KS/기아's report used for ALL NINE of its mislabeled figures
# (`₩625,389 billion`, `KRW 625,389 bn`) and that the trailing-currency-only
# patterns could not see. Deliberately NOT a bare `W` — "W" before a number
# occurs in ordinary prose (week numbers, "W 12").
_NUM = r"(" + _SIGN + r"[\d,]+(?:\.\d+)?)"
_KRW_PREFIX = r"(?:₩|KRW\s*)"
_KRW_SUFFIX = r"(?:KRW|won|₩)"

_BILLION_RE = re.compile(
    r"(?:"
    # ₩625,389 billion / KRW 625,389 bn — currency as a PREFIX, unit word
    # followed by anything (`,`, ` in `, `(`). The 9/9 blind spot.
    rf"{_KRW_PREFIX}\s*{_NUM}\s*(?:billion|bn)\b"
    r"|"
    # 951,542 billion KRW — the original 2026-08-28 trailing-currency form.
    rf"{_NUM}\s*(?:billion|bn)\s+{_KRW_SUFFIX}"
    r"|"
    # ₩48B / ₩48.3B — a BARE `B` suffix, accepted ONLY behind an explicit
    # ₩/KRW prefix (an un-gated `\d+B` would match version strings, sizes, IDs).
    rf"{_KRW_PREFIX}\s*{_NUM}\s*[Bb](?![\w])"
    r"|"
    # 176B KRW / -3B KRW — the same bare `B`, currency as a SUFFIX. The
    # trailing `\s+{_KRW_SUFFIX}` is exactly as strong a gate as the leading
    # ₩: "Series B", "Grade B", "48B parameter model" are never followed by
    # whitespace + KRW/won/₩. Missed id=1168's "-3B KRW" entirely
    # (candidates_examined: 0).
    rf"{_NUM}\s*[Bb]\s+{_KRW_SUFFIX}"
    r")",
    re.IGNORECASE,
)
_TRILLION_RE = re.compile(
    r"(?:"
    rf"{_KRW_PREFIX}\s*{_NUM}\s*(?:trillion|tn)\b"
    r"|"
    rf"{_NUM}\s*(?:trillion|tn)\s+{_KRW_SUFFIX}"
    r"|"
    # ~₩48T — bare `T`, ₩-prefix-gated. The span that flipped id=1158's risk
    # debate to Buy.
    rf"{_KRW_PREFIX}\s*{_NUM}\s*[Tt](?![\w])"
    r"|"
    # 48조 / 62.54조원 — 조 is unambiguously Korean trillion-won, no currency
    # token required.
    rf"{_NUM}\s*조원?"
    r"|"
    # 48T KRW — bare `T`, currency as a SUFFIX. Same gate, same reasoning as
    # _BILLION_RE's suffix-B alternative.
    rf"{_NUM}\s*[Tt]\s+{_KRW_SUFFIX}"
    r")",
    re.IGNORECASE,
)

# Korean-won budget figures rendered as USD trillions in id=1158's News+Macro
# report ("~$3T for EV incentives" — ~6x Korea's entire national budget), then
# adopted as a Bull pillar. No ground truth can exist for this, so it is
# flag-only by construction; opt-in via `usd_check` since a legitimate US
# macro figure ("US GDP is $27 trillion") will also match.
_USD_TRILLION_RE = re.compile(
    r"(?:US\$|\$|USD\s*)\s*(-?[\d,]+(?:\.\d+)?)\s*(?:trillion|tn|T(?![\w]))",
    re.IGNORECASE,
)
_USD_SUSPECT_ABOVE = 1.0

# A "N billion KRW (M trillion KRW)" self-contradiction — the 두산에너빌리티
# shape, where the two numbers in the SAME sentence already disagree by 10x
# regardless of whether either matches a supplied 억원 figure. Guards a
# derived (not directly-restated) number the ground-truth check can't see.
# Rebuilt 2026-09-01 from the shared blocks so a `₩`-prefixed pair
# (`₩4,833.7 billion (₩48T)`) and a `조원` right-hand side are both visible —
# the old pattern's currency class was `(?:KRW|won)`, dropping `₩` entirely.
_PAREN_RE = re.compile(
    r"(?:" + _KRW_PREFIX + r"\s*)?" + _NUM + r"\s*(?:billion|bn)\s*(?:" + _KRW_SUFFIX + r")?\s*"
    r"[\(\[]\s*(?:≈|~|about\s+|approx\.?\s*)?(?:" + _KRW_PREFIX + r"\s*)?" + _NUM +
    r"\s*(?:trillion|tn|조원?|[Tt](?![\w]))",
    re.IGNORECASE,
)

# --- 억원<->billion pair (2026-09-15, id=1432, 040300.KS/YTN) ---
# The 억원<->billion sibling of `self_inconsistent` above: a report states its
# OWN 억원 source figure and its OWN billion conversion in the same breath
# ("Revenue peaked in FY2022 at 1,522억 원 ($15.22B KRW)" — true value is
# 152.2 billion, off by exactly 10x), and does so for all three of
# id=1432's headline P&L figures (revenue FY2022/FY2024, net income FY2021).
# Unlike `self_inconsistent` (billion vs trillion — NEITHER side is ground
# truth), the 억원 side here CAN be checked against `eok_values`: if it
# matches a supplied figure, the correct billion value is KNOWN, not
# guessed, so the violation is emitted auto-correctable — see
# `_eok_pair_10x` below.
#
# A SEPARATE, looser regex from `_EOK_RE` on purpose — do NOT touch `_EOK_RE`
# itself, its exact `억원` token is load-bearing for `period_mismatch_guard`
# (see `_EOK_RE`'s own comment above). id=1432's report wrote "1,522억 원"
# (a space before 원) and other figures as bare "...억" with no 원 at all;
# `_EOK_RE` requires the tight two-character `억원` token and matches
# neither. Follows `_HEADLINE_EOK_RE`'s already-established `(억|조)\s*원?`
# precedent for a loose report-prose mention (this one is 억-only — 조 is out
# of scope for this pair check) rather than inventing a third convention.
_EOK_REPORT_RE = re.compile(r"(?<![\d.,])(" + _SIGN + r"[\d,]+(?:\.\d+)?)\s*억\s*원?")

# Window after a report-side 억원 mention to look for its own billion
# conversion. The observed shape is a parenthetical immediately following
# ("1,522억 원 ($15.22B KRW)") — well under 20 characters including the
# currency symbol and unit token. 40 gives headroom for a comma-grouped
# mantissa and a short qualifier ("약", "approximately") without reaching
# past the parenthetical into an unrelated NEXT sentence's own billion/
# trillion mention (a real risk in a Fundamentals report that states several
# magnitudes in sequence). This is its own named constant, not a reuse of
# `_MARKET_CAP_WINDOW` (also 40) — that one is scoped to an unrelated
# comparator + "market cap" noun-phrase check; the value coincides but the
# purpose does not.
_EOK_PAIR_WINDOW = 40

# Skip any candidate that overlaps an already-applied correction marker — the
# marker re-contains the offending span verbatim, so a second pass (a debate
# turn quoting a corrected Fundamentals sentence — routine once the guard fans
# out across nodes) would re-flag and nest it.
_CORRECTION_MARKER_RE = re.compile(
    r"\[(?:unit corrected from|label corrected from|단위 정정|지표 정의 정정|단위 오류 정정)[^\]]*\]"
)

_REL_TOL = 0.005  # 0.5% — float/formatting slop only, not a real tolerance band
# A figure the model rounded or marked approximate ("~₩48T", "₩48B") cannot be
# compared at 0.5%: id=1158's true 48,337억원 renders as 48.337조 and its
# "~₩48T" is 0.70% off — outside _REL_TOL. Widened band, applied ONLY to spans
# with <= 3 significant digits or an approximation marker, and SYMMETRICALLY
# (the accept-first check uses the same band, so a legitimately-rounded CORRECT
# "₩4.8T" is still accepted, not flagged). Do NOT widen _REL_TOL itself.
_APPROX_REL_TOL = 0.02
_APPROX_MARK_RE = re.compile(r"(?:[~≈]|about|approx\.?)\s*$", re.IGNORECASE)
_PAREN_REL_TOL = 0.02

# The band above which a measured 10x mismatch is auto-corrected on the
# mantissa match ALONE. This is the original `e / 10 >= 1.0` (undocumented,
# born in the module's first commit, never revisited) kept intact: below it,
# a 1-2 significant digit mantissa collides with an unrelated supplied 억원
# value often enough that a bare match is not sufficient evidence to rewrite
# a number. Below the floor, corroboration is required instead of refusal —
# see `_sign_agrees`.
_BILLION_CORRECT_FLOOR = 1.0  # billion KRW

# --- currency-composition net (2026-09-03, live decision id=1207 / 270520.KS) ---
# Opt-in via `currency_check` (Fundamentals Analyst only). Premise gates: enough
# supplied 억원 ground truth that a real KRX report could not plausibly be
# KRW-free (`_MIN_GT_FOR_CURRENCY` — also the ETF carve-out at this layer, since
# this module can't see the alias set), and a report long enough to have stated
# its currency at all (`_MIN_REPORT_CHARS`). `_MIN_USD_FIGURES` is held equal to
# `identity_gate._MIN_USD_MENTIONS` by value (NOT imported).
_MIN_GT_FOR_CURRENCY = 5
_MIN_REPORT_CHARS = 600
_MIN_USD_FIGURES = 3

# Number-anchored USD notations only — a bare `\bUSD\b` is NOT counted.
_USD_FIGURE_RE = re.compile(
    r"(?:US\$|\$|USD\s*)\s*-?[\d,]+(?:\.\d+)?"
    r"|[\d,.]+\s*(?:million|billion|bn|trillion)\s+(?:USD|dollars?)",
    re.IGNORECASE,
)
# Deliberately lenient — any hint of a won-denominated figure disarms the flag.
_KRW_TOKEN_RE = re.compile(
    r"₩|\bKRW\b|\bwon\b|Korean won"
    r"|\d[\d,.]*\s*(?:원|억|조|억원|조원|만원|천원)"
    r"|\d[\d,.]*\s*(?:billion|trillion|bn|tn)\s+(?:KRW|won|₩)",
    re.IGNORECASE,
)
# Idempotency: skip the check if this banner is already on the text. Separate
# from `_CORRECTION_MARKER_RE` (which marks in-place unit corrections).
_CURRENCY_BANNER_MARK_RE = re.compile(r"통화 구성 이상|CURRENCY-COMPOSITION ALERT")


def _sign_agrees(a: float, b: float) -> bool:
    """Corroboration for a sub-`_BILLION_CORRECT_FLOOR` 10x match. The
    measured mislabel shape carries the 억원 token's digits AND its minus
    straight through, so sign agreement costs a true positive nothing while
    removing the coincidental-collision class the floor was implicitly
    guarding. NOT required above the floor: a report that writes a loss as
    "-785.2 billion KRW" against a kr_block that wrote it as a positive
    "loss=785억원" is a legitimate sign-convention difference, not evidence
    of anything (tests/agents/test_numeric_guard.py pins exactly that shape)."""
    return (a < 0) == (b < 0)


def _to_float(token: str) -> float:
    return float(token.replace(",", "").replace("−", "-"))


def _mantissa(m: re.Match[str]) -> float:
    """The one non-None numeric group of a multi-alternative match."""
    return _to_float(next(g for g in m.groups() if g is not None))


def _sig_digits(token: str) -> int:
    s = token.replace(",", "").replace("−", "").replace("-", "").strip()
    if "." in s:
        s = s.replace(".", "").lstrip("0")
    else:
        s = s.strip("0")
    return len(s) or 1


def _tol_for(report: str, m: re.Match[str]) -> float:
    """`_APPROX_REL_TOL` when the span is approximation-marked or low-precision
    (<= 3 significant digits), else `_REL_TOL`."""
    preceding = report[max(0, m.start() - 12) : m.start()]
    if _APPROX_MARK_RE.search(preceding):
        return _APPROX_REL_TOL
    num = next((g for g in m.groups() if g is not None), "")
    return _APPROX_REL_TOL if _sig_digits(num) <= 3 else _REL_TOL


def _magnitude_suspect(v: float, eok_values: frozenset[float]) -> bool:
    """A power-of-10 ladder near-miss: does `v * 10**k` for k in
    {1, 2, 3, -2, -3} land on some supplied figure's correct billion
    (`e/10`) or trillion (`e/10_000`) value, within the approx band?

    2026-09-11 addendum (id=1386, 006360.KS): `k=1` was added after a real
    10x-UNDER-shifted write slipped past every existing check. Ground truth
    was `e=156.5` (억원, from `[Program trading] ... (+156.5억원)`). One
    debater correctly wrote "+15.65 billion KRW" (av=15.65 = e/10 — accepted
    by the exact-match check above, not a bug). A second debater then
    "corrected" this to "+1.565 billion KRW" (av=1.565 = e/100), calling the
    correct 15.65 figure a "tenfold exaggeration" — and both the Research
    Manager and PM adopted the wrong correction as decisive reasoning.
    Neither Tier 1 (`_isclose(av, abs(e), tol)`, which only catches the
    OVER-by-10x shape av≈e) nor the old k-tuple caught av=e/100: for k=1,
    `scaled = v * 10 = 1.565 * 10 = 15.65`, which is exactly `e/10 = 15.65`
    — a match. This was verified by hand against this function's own
    `scaled = v * (10.0 ** k)` formula before implementing, not assumed.

    `k=-1` is deliberately NOT added: `scaled = v / 10`, so matching `e/10`
    requires `v ≈ e` — that is exactly Tier 1's own `billion_10x`
    exact-match condition a few lines above this function's call sites, so
    `k=-1` would be entirely redundant with Tier 1 and would only widen the
    false-positive surface (more ladder rungs to coincidentally collide
    with an unrelated supplied value) for zero additional coverage.

    Stays flag-only (`magnitude_suspect`), NOT auto-corrected (Tier 1): this
    is n=1 real example, unlike the measured 9/9 + 13/13 record that
    justified the existing OVER-direction auto-correction — pending more
    calibration data before promotion.

    2026-09-15 addendum (60-day production scan, precision measurement).
    131 `magnitude_suspect` occurrences across 41 decisions, 79 distinct
    (ticker, figure) pairs. A hand-read sample of 4 found only 1 true
    positive (id=1436, 024060.KS: "the company generated negligible revenues
    (~₩11.7 billion)" — real FY2024 revenue was ₩116.85B) against 2 clear
    false positives (id=1433, 004310.KS: "perhaps 50-100 billion KRW if any
    assets existed" — a hypothetical valuation, not a data claim, AND the
    flagged "-100 billion KRW" carried a sign that was itself wrong — the
    "-" is a range hyphen from "50-100", not a minus; id=1441, 278650.KS:
    "Annualizing this H1 figure suggests an operating loss of nearly 9
    billion KRW" — the model's own derived arithmetic from a correctly
    stated input, not a misstated figure) and 1 undetermined (id=1409,
    316140.KS). That precision is far below the 9/9 + 13/13 record above
    that justified the OVER-direction auto-correction, so
    `scan_unit_scale`'s call sites now exempt these three false-positive
    classes via `_magnitude_suspect_exempt` (range-hyphen misparse,
    hypothetical/scenario framing, model-derived/annualized arithmetic —
    ported from `period_mismatch_guard`, which solved the same three
    classes for its own candidates). This function itself is UNCHANGED — the
    exemption is applied only at the point a finding would be recorded, so a
    caller of `_magnitude_suspect` directly still sees the raw ladder-match
    signal.

    Promotion to auto-correction remains BLOCKED regardless of this fix: it
    only removes known false-positive classes from the flag-only signal, it
    does not establish new positive precision. Do not promote this tier to
    Tier 1 until a POST-FIX precision measurement on fresh live data (not
    this same 4-sample set) is taken and reviewed — a future session must
    not promote it on vibes."""
    if v == 0:
        return False
    for k in (1, 2, 3, -2, -3):
        scaled = v * (10.0 ** k)
        for e in eok_values:
            if _isclose(scaled, abs(e) / 10, _APPROX_REL_TOL) or _isclose(scaled, abs(e) / 10_000, _APPROX_REL_TOL):
                return True
    return False


def _isclose(a: float, b: float, rel: float) -> bool:
    if b == 0:
        return a == 0
    return abs(a - b) <= rel * abs(b)


# Digit-extraction idiom reused from this file's own `_NUM`/`_to_float`
# convention: pull the first plain digit run out of a `_USD_FIGURE_RE` match
# and feed it to `_to_float`, rather than inventing a new numeric grammar.
_USD_MANTISSA_RE = re.compile(r"[\d,]+(?:\.\d+)?")
_USD_MAGNITUDE_RE = re.compile(r"million|billion|bn|trillion|tn", re.IGNORECASE)
# `_USD_FIGURE_RE`'s currency-PREFIX alternative (`$28`) stops at the digit
# run and never consumes a following magnitude word — "$28 million USD"
# matches only "$28", leaving " million USD" outside the span. This picks up
# that trailing word (and an optional trailing USD/dollars token) so the
# companion-USD-figure check in `scan_unit_scale` can both value the figure
# correctly and redact the whole phrase, not just the digit prefix.
_USD_TAIL_MAGNITUDE_RE = re.compile(
    r"\s*(?:million|billion|bn|trillion)\b\s*(?:USD|dollars?)?", re.IGNORECASE
)
_USD_MAGNITUDE_MULT = {
    "million": 1e6, "billion": 1e9, "bn": 1e9, "trillion": 1e12, "tn": 1e12,
}


def _usd_value(matched_text: str) -> float | None:
    """Dollar value of a `_USD_FIGURE_RE` match, or None if unparseable.
    Handles '$28 million' / '$28,000,000' / 'US$1.2 billion' / '28 million
    USD' shapes: extracts the leading numeric mantissa via `_to_float` on the
    digit run, then multiplies by 1e6/1e9/1e12 if a million/billion/trillion/
    bn/tn word is present in `matched_text`, else treats the mantissa as
    already in plain dollars."""
    m = _USD_MANTISSA_RE.search(matched_text)
    if m is None:
        return None
    try:
        mantissa = _to_float(m.group(0))
    except ValueError:
        return None
    mag = _USD_MAGNITUDE_RE.search(matched_text)
    if mag is not None:
        return mantissa * _USD_MAGNITUDE_MULT[mag.group(0).lower()]
    return mantissa


def extract_kr_eok_values(kr_block: str) -> frozenset[float]:
    """Every 억원 figure `kr_block` (the exact text handed to the LLM —
    `app.agents.fundamentals_analyst.analyze`'s assembled Korean financial
    data) supplied to the model, as plain floats in 억원 units. This is the
    ground truth `check_unit_scale` checks the report's own claimed
    billion/trillion figures against."""
    return frozenset(_to_float(m.group(1)) for m in _EOK_RE.finditer(kr_block))


# Headline-sourced 억/조 amounts (2026-09-07 audit #3). A SEPARATE regex from
# `_EOK_RE` on purpose: `_EOK_RE` must keep requiring the exact `억원` token
# because `period_mismatch_guard` parses the same Fundamentals block
# positionally against a `<item>=<n>억원` shape. This one is looser — a bare
# 억 or 조, `원` optional — because that is what a news headline actually
# writes ("6.6억", "6.6억원", "6.62억원 규모").
_HEADLINE_EOK_RE = re.compile(r"(?<![\d.,])(" + _SIGN + r"[\d,]+(?:\.\d+)?)\s*(억|조)\s*원?")


def extract_kr_headline_eok_values(news_text: str) -> frozenset[float]:
    """Every 억/조 money amount a `get_kr_news` block writes, normalized to
    억 units — `scan_unit_scale`'s SOFT ground truth (2026-09-07 domestic
    audit #3, id=1237 / 294630.KS: a headline's "6.6억원" contract was
    reported as "₩6.6 billion", a 10x carry-through, and it propagated
    through the whole desk). Soft because a headline is untrusted third-party
    text: a match here flags a suspected 10x, it never auto-rewrites a
    figure.

    STILL flag-only after id=1358 (2026-09-09, 047040.KS: a 4,134.3억원
    contract from three outlets, written "4,134.3 billion KRW", 10x). That
    incident is genuine and, on the surface, "digit-for-digit exact + a
    power-of-10 relationship" is the same shape `billion_10x` auto-corrects.
    Auto-correction was still rejected, for reasons the #211 "기각한 선택지"
    entry ([[unit-scale-billion-recurrence-2026-09-04]]) applies to here too,
    plus one this feed has that the KIS block does not:
      - `get_kr_news` is "the classic prompt-injection surface" (its own
        docstring in app.dataflows.kis_provider). Auto-rewriting a number in
        the desk's inputs FROM planted headline text — and stamping it with
        this guard's `[unit corrected from "..."]` marker, whose meaning is
        "verified against ground truth" — is a materially worse posture than
        a note the model can weigh.
      - Zero live firings in 30 days (357 KRX rows, 169 with a unit scan);
        `soft_ground_truth_values > 0` on only ~8% of decisions, and then
        almost always a single value. n = 1 recoverable true positive (this
        one), which the `magnitude_suspect` de-shadow above already converts
        to the actionable `_NEWS_EOK_NOTE`. No calibration set exists.
      - The correction target is ambiguous: id=1358's own soft GT held
        {4134.0, 4134.27, 4134.3} for one contract.
    If it is ever promoted, do it behind an OFF-by-default setting (the
    `sizing_cap_misuse` gate pattern: instrument -> measure -> enable), only
    for a candidate that: has no hard-GT explanation; has >= 4 significant
    digits; matches within `_REL_TOL` (never `_tol_for`); `_sign_agrees`
    unconditionally; `abs(s) >= 100억`; is corroborated by >= 2 DISTINCT
    headline lines within `_REL_TOL` (needs line provenance, not a bare
    frozenset); and comes from a soft set of <= 10 values (flooding
    defense — `extract_kr_headline_eok_values` currently has neither the
    noise floor nor the row cap `kis_provider.kr_amount_unit_key` applies).
    Promotion criterion: >= 10 corroborated firings over >= 5 decisions and
    >= 4 weeks, manually reviewed at >= 95% true-positive, zero case where the
    corroborating headline was itself rounded. The `news_amount_scans`
    roll-up (app.orchestrator.pipeline) is the counter that measures this."""
    out: set[float] = set()
    for m in _HEADLINE_EOK_RE.finditer(news_text):
        out.add(_to_float(m.group(1)) * (10_000.0 if m.group(2) == "조" else 1.0))
    return frozenset(out)


@dataclass(frozen=True)
class UnitScaleViolation:
    # "billion_10x" | "trillion_10x"  — auto-corrected (Tier 1, expected set)
    # "self_inconsistent" | "magnitude_suspect" | "usd_unit_suspect" |
    #   "currency_inverted" | "news_eok_10x" — flag-only (Tier 2, expected
    #   None: a weaker inference than the measured 10x shape, so a note/banner
    #   is appended but no number is rewritten). "currency_inverted" is the
    #   one kind whose `written` is NOT a pure verbatim span (see
    #   scan_unit_scale). "news_eok_10x" (2026-09-07 audit #3) matches an 억/조
    #   headline amount digit-for-digit — flag-only because the ground truth
    #   is untrusted headline text.
    # "eok_pair_10x" (2026-09-15, id=1432/YTN) — CONDITIONALLY auto-corrected:
    #   a report states its own 억원 figure and its own (10x-wrong) billion
    #   conversion in the same breath. When the 억원 side matches a supplied
    #   `eok_values` entry, the correct billion value is known (not guessed)
    #   and `expected` is set, exactly like Tier 1. When it does NOT match
    #   any supplied figure, the pair is still self-contradictory but neither
    #   side is verified, so `expected=None` (flag-only), same posture as
    #   `self_inconsistent`. Do not assume this kind is always one or the
    #   other — check `expected` on the individual violation.
    kind: str
    written: str  # the offending span, verbatim (except "currency_inverted")
    expected: str | None  # corrected rendering, or None for a flag-only kind
    start: int
    end: int
    # Populated ONLY on "billion_10x"/"trillion_10x" — the corrected KRW value
    # in plain won (not the formatted `expected` string), used solely by the
    # 2026-09-11 companion-USD-figure check (id=1396) to judge whether a
    # nearby USD figure implies a plausible KRW/USD rate against the
    # CORRECTED magnitude. None for every other kind.
    expected_won: float | None = None


@dataclass(frozen=True)
class UnitScaleScan:
    """The POSITIVE signal an empty violation list cannot carry. Decision
    id=1158 (000270.KS/기아, 2026-09-01) recorded `unit_scale_violations=[]`
    on a report containing NINE 10x-mislabeled 억원 figures, and that empty
    list was read as "clean" — the operational failure was the
    indistinguishability of "no violations" from "no candidates examined"
    from "regex blind to this notation"."""

    node: str
    candidates_examined: int  # billion/trillion magnitude spans any pattern matched
    ground_truth_values: int  # len(eok_values) — 억원 figures supplied
    corrected: int  # Tier-1 auto-corrections applied
    flagged: int  # Tier-2 / self_inconsistent notes
    unverified: int  # candidates matching NEITHER a correct value nor any rule

    # Pure observation, always populated regardless of `currency_check` or the
    # currency premise gates (2026-09-03, id=1207). `currency_inverted` is the
    # only one of the three that is a JUDGMENT — set only on a node that
    # opted in via `currency_check`.
    usd_figures: int = 0
    krw_figures: int = 0
    currency_inverted: bool = False

    # Soft (news-headline) ground truth, 2026-09-07 audit #3 — always
    # observed, never gates or auto-corrects. `soft_ground_truth_values` is
    # how many distinct 억/조 headline amounts were supplied;
    # `news_eok_flagged` is how many report figures matched one digit-for-digit
    # (the suspected 10x carry-through).
    soft_ground_truth_values: int = 0
    news_eok_flagged: int = 0

    # 2026-09-11 addendum (id=1396): a companion USD figure computed from a
    # pre-correction (10x-wrong) KRW magnitude, redacted alongside the
    # billion_10x/trillion_10x fix that exposed it. See `_usd_value` /
    # `_sentence_span` and the new pass in `scan_unit_scale`.
    derived_usd_stale: int = 0

    @property
    def suspicious(self) -> bool:
        """Either (a) `currency_inverted` — a KRX-name report awash in USD
        figures with no KRW figure at all (id=1207), or (b) a report that
        quoted large KRW figures, against a ground truth we actually had,
        where NOTHING matched anything — the exact shape id=1158 presented.
        Not proof of a bug; a demand to look."""
        return self.currency_inverted or (
            self.ground_truth_values > 0
            and self.candidates_examined > 0
            and self.corrected == 0
            and self.unverified == self.candidates_examined
        )

    def as_dict(self) -> dict[str, int]:
        return {
            "candidates_examined": self.candidates_examined,
            "ground_truth_values": self.ground_truth_values,
            "corrected": self.corrected,
            "flagged": self.flagged,
            "unverified": self.unverified,
            "usd_figures": self.usd_figures,
            "krw_figures": self.krw_figures,
            "currency_inverted": int(self.currency_inverted),
            "soft_ground_truth_values": self.soft_ground_truth_values,
            "news_eok_flagged": self.news_eok_flagged,
            "derived_usd_stale": self.derived_usd_stale,
        }


def _marker_spans(report: str) -> list[tuple[int, int]]:
    return [(m.start(), m.end()) for m in _CORRECTION_MARKER_RE.finditer(report)]


def _in_marker(start: int, end: int, markers: list[tuple[int, int]]) -> bool:
    return any(s <= start < e or s < end <= e for s, e in markers)


def _sentence_span(text: str, start: int, end: int) -> tuple[int, int]:
    """Char offsets of the sentence containing text[start:end] (nearest '.'
    on each side, or the text boundary) — same convention as
    app.agents.regulatory_mechanism_guard._quoted_span."""
    left = text.rfind(".", 0, start)
    left = left + 1 if left != -1 else 0
    right = text.find(".", end)
    right = right if right != -1 else len(text)
    return left, right


# A comparator symbol/word immediately before the number ("<", ">", "under",
# "below", "over", "above") — 2026-09-08 fix, id=1315 (387690.KS/레메디):
# "This is common for micro-cap stocks (<₩100B market cap) due to low
# trading volume" is a GENERIC industry-tier definition, not a restatement
# of this company's own supplied figure, and `100` happened to land inside
# `_magnitude_suspect`'s wider ±2/±3 power-of-10 tolerance band against an
# unrelated ground-truth value — a textbook false positive of exactly the
# "weaker inference, more collision-prone" shape that function's own
# docstring already warns about.
_COMPARATOR_BEFORE_RE = re.compile(
    r"(?:[<>≤≥]|under|below|over|above|less than|more than|at least|no more than|up to)\s*$",
    re.IGNORECASE,
)
# ...co-occurring with a market-cap/valuation-TIER noun close by (within
# `_MARKET_CAP_WINDOW`) — comparator alone is too broad and would also
# swallow a genuine hedged restatement of the company's OWN figures
# ("revenues under 120 billion KRW/year, having over 400 billion KRW in
# assets" — 900270.KS, id=1259, a real anomaly discussion, not a boilerplate
# tier definition). Requiring BOTH signals is what keeps this narrow:
# calibrated against 200 live Fundamentals Analyst reports (2026-09-08),
# 6 comparator-adjacent billion/trillion candidates total, of which only
# the one genuine "market cap" case above should be excluded — the other
# 4 (900270.KS, revenue/assets discussion) correctly stay in scope.
_MARKET_CAP_TIER_RE = re.compile(r"market\s*cap(?:italization)?", re.IGNORECASE)
_MARKET_CAP_WINDOW = 40


def _is_market_cap_tier_mention(report: str, start: int, end: int) -> bool:
    """True if the `_BILLION_RE`/`_TRILLION_RE` candidate at `report[start:end]`
    is a market-cap/valuation-TIER classification (e.g. "micro-cap stocks
    (<₩100B market cap)") rather than a restatement of this company's own
    supplied figure — see the module-level comment above `_COMPARATOR_BEFORE_RE`
    for the incident and calibration this is built from. Checked BEFORE a
    candidate is examined at all (same "skip, don't even count it" shape as
    `_in_marker`), so it protects both the Tier-1 auto-correcting
    `billion_10x`/`trillion_10x` path and the Tier-2 flag-only
    `magnitude_suspect` path with one exclusion point — a boilerplate tier
    definition could coincidentally land on an EXACT 10x match just as
    easily as an approximate one, so Tier 1 needs this exclusion just as
    much as Tier 2 does, not only the kind this was first noticed on."""
    if not _COMPARATOR_BEFORE_RE.search(report[max(0, start - 15) : start]):
        return False
    window = report[max(0, start - _MARKET_CAP_WINDOW) : end + _MARKET_CAP_WINDOW]
    return bool(_MARKET_CAP_TIER_RE.search(window))


# --- magnitude_suspect precision fixes (2026-09-15). A 60-day production
# scan found 131 magnitude_suspect occurrences across 41 decisions (79
# distinct (ticker, figure) pairs). A hand-read sample of 4 found only 1 true
# positive (id=1436, 024060.KS: "the company generated negligible revenues
# (~₩11.7 billion)" — real FY2024 revenue was ₩116.85B) against 2 clear false
# positives (id=1433, 004310.KS: "a return to a rational valuation, perhaps
# 50-100 billion KRW if any assets existed" — a hypothetical, and the
# flagged "-100 billion KRW" carried a sign that was itself wrong, misread
# off the "50-100" range hyphen; id=1441, 278650.KS: "Annualizing this H1
# figure suggests an operating loss of nearly 9 billion KRW" — the model's
# own derived arithmetic, not a restated data figure) and 1 undetermined
# (id=1409, 316140.KS). That precision is far below the 9/9 and 13/13
# records that justified promoting the auto-correcting tiers, so the fix
# here is ONLY to exempt these three false-positive classes from
# `magnitude_suspect` — never to make this tier auto-correct. Promotion to
# auto-correction remains BLOCKED until a post-fix precision measurement on
# fresh live data justifies it; do not promote this tier on vibes alone.
#
# Ported from `app.agents.period_mismatch_guard`, a sibling guard that
# solved the same three false-positive classes for its own forbidden-ratio
# candidates — same patterns, same application, not reinvented.
# ---------------------------------------------------------------------------

# Rule (1) — range-hyphen misparse, a plain bug (not only a framing
# question): `_SIGN`'s `[-−]?` sits directly against the digit run with no
# intervening `\s*`, so "50-100" has its "-" consumed as this candidate's
# OWN sign — the figure `_mantissa` returns is not merely a range endpoint
# mislabeled as a restated figure, its SIGN is factually wrong (a positive
# range endpoint reported as negative). Two shapes, identical to
# `period_mismatch_guard`'s own rule ⑥: `_RANGE_TAIL_RE` catches a
# hyphen/en-dash/tilde with intervening whitespace before the candidate
# (the sign was NOT consumed, e.g. "50 - 100"); `_RANGE_HYPHEN_RE` catches
# the tight "50-100" shape, checked only when this candidate's own mantissa
# is negative (the sign WAS consumed).
_RANGE_TAIL_RE = re.compile(r"\d+(?:\.\d+)?\s*[-–~]\s*$")
_RANGE_HYPHEN_RE = re.compile(r"\d$")

# Rules (2)/(3) — hypothetical/scenario framing and model-derived
# arithmetic. Vocabulary ported verbatim from
# `period_mismatch_guard._SCENARIO_CTX_RE` / `._HEDGE_CTX_RE`.
_SCENARIO_CTX_RE = re.compile(
    r"could|can extend|may |might |if support|scenario|worst[- ]case|risk of", re.IGNORECASE,
)
_HEDGE_CTX_RE = re.compile(
    r"annually|annualiz|annualis|per annum|\bp\.a\.|\bCAGR\b|annual rate|annual .{0,12}erosion",
    re.IGNORECASE,
)
# Window for both: `period_mismatch_guard` scans this same vocabulary in its
# own `_ITEM_WINDOW` (80 chars each side) — the sibling's already-calibrated
# window for exactly this purpose (a full hedge/scenario clause: "perhaps
# ... if any assets existed", "Annualizing this H1 figure suggests ...").
# Named for this file's own use rather than reusing `_MARKET_CAP_WINDOW`
# (40), which is sized for a single "market cap" noun phrase near a
# comparator — an unrelated, narrower purpose.
_MAGNITUDE_SUSPECT_CTX_WINDOW = 80


def _magnitude_suspect_exempt(report: str, m: re.Match[str]) -> bool:
    """True if a `magnitude_suspect` candidate at `m` should be exempted —
    one of the three false-positive classes surfaced by the 2026-09-15
    60-day scan (see the block comment above): a range endpoint (whose sign
    this file's own `_SIGN` mis-parsed), a hypothetical/scenario figure, or
    the model's own derived arithmetic. Checked ONLY for the flag-only
    `magnitude_suspect` kind — never applied to the auto-correcting
    `billion_10x`/`trillion_10x` path, which is out of scope for this fix."""
    token = next(g for g in m.groups() if g is not None)
    pre = report[: m.start()]
    if _RANGE_TAIL_RE.search(pre):
        return True
    if token.startswith(("-", "−")) and _RANGE_HYPHEN_RE.search(pre):
        return True
    window = report[
        max(0, m.start() - _MAGNITUDE_SUSPECT_CTX_WINDOW) : m.end() + _MAGNITUDE_SUSPECT_CTX_WINDOW
    ]
    if _SCENARIO_CTX_RE.search(window):
        return True
    if _HEDGE_CTX_RE.search(window):
        return True
    return False


def scan_unit_scale(
    report: str,
    eok_values: frozenset[float],
    *,
    usd_check: bool = False,
    currency_check: bool = False,
    soft_eok_values: frozenset[float] = frozenset(),
    node: str = "",
) -> tuple[list[UnitScaleViolation], UnitScaleScan]:
    """Finds every "N billion/trillion KRW" mention in `report` that is
    exactly (within float/formatting slop) 10x a supplied 억원 figure divided
    by its correct unit divisor — i.e. the model wrote the 억원 mantissa
    straight through instead of dividing it down — plus the flag-only shapes
    (`magnitude_suspect`, `usd_unit_suspect`, `self_inconsistent`). Also
    returns a `UnitScaleScan` recording what was examined vs. matched, so an
    empty violation list can be told apart from a blind spot (see id=1158).

    `eok_values` empty (no KR data was supplied at all) skips the ground-truth
    checks entirely — same fail-open posture as `identity_gate.check_identity`
    — but the self-contradiction and USD checks still run.

    Order matters: a candidate that already matches ANY supplied figure's
    correct converted value is accepted and never re-checked against a
    10x-off candidate — this is what keeps the false-positive rate at
    effectively zero."""
    violations: list[UnitScaleViolation] = []
    markers = _marker_spans(report)
    examined = 0
    accepted = 0
    corrected = 0
    suspect = 0
    derived_usd_stale = 0

    if eok_values:
        for m in _BILLION_RE.finditer(report):
            if _in_marker(m.start(), m.end(), markers):
                continue
            if _is_market_cap_tier_mention(report, m.start(), m.end()):
                continue
            examined += 1
            v = _mantissa(m)
            tol = _tol_for(report, m)
            av = abs(v)
            sign = -1.0 if v < 0 else 1.0
            if any(_isclose(av, abs(e) / 10, tol) for e in eok_values):
                accepted += 1
                continue  # correct — matches some supplied figure's true billion-KRW value
            matched = next(
                (e for e in eok_values
                 if _isclose(av, abs(e), tol)
                 and (abs(e) / 10 >= _BILLION_CORRECT_FLOOR or _sign_agrees(v, e))),
                None,
            )
            if matched is not None:
                corrected += 1
                violations.append(UnitScaleViolation(
                    kind="billion_10x", written=m.group(0),
                    expected=f"{sign * abs(matched) / 10:,.1f} billion KRW",
                    start=m.start(), end=m.end(),
                    expected_won=abs(matched) / 10 * 1e9,
                ))
                continue
            if _magnitude_suspect(av, eok_values) and not _magnitude_suspect_exempt(report, m):
                suspect += 1
                violations.append(UnitScaleViolation(
                    kind="magnitude_suspect", written=m.group(0), expected=None,
                    start=m.start(), end=m.end(),
                ))
        for m in _TRILLION_RE.finditer(report):
            if _in_marker(m.start(), m.end(), markers):
                continue
            if _is_market_cap_tier_mention(report, m.start(), m.end()):
                continue
            examined += 1
            v = _mantissa(m)
            av = abs(v)
            sign = -1.0 if v < 0 else 1.0
            tol = _tol_for(report, m)
            if any(_isclose(av, abs(e) / 10_000, tol) for e in eok_values):
                accepted += 1
                continue  # correct
            matched = next(
                (e for e in eok_values
                 # e exact: 억원 mantissa carried straight into 조/trillion (10000x);
                 # e/1000: 10x over; e/100000: 10x under. Correct value is e/10000
                 # in every case.
                 if _isclose(av, abs(e), tol) or _isclose(av, abs(e) / 1_000, tol)
                 or _isclose(av, abs(e) / 100_000, tol)),
                None,
            )
            if matched is not None:
                corrected += 1
                violations.append(UnitScaleViolation(
                    kind="trillion_10x", written=m.group(0),
                    expected=f"{sign * abs(matched) / 10_000:,.2f} trillion KRW",
                    start=m.start(), end=m.end(),
                    expected_won=abs(matched) / 10_000 * 1e12,
                ))
                continue
            if _magnitude_suspect(av, eok_values) and not _magnitude_suspect_exempt(report, m):
                suspect += 1
                violations.append(UnitScaleViolation(
                    kind="magnitude_suspect", written=m.group(0), expected=None,
                    start=m.start(), end=m.end(),
                ))

    # Companion-USD-figure staleness check (2026-09-11, id=1396 / 003350.KS).
    # A Tier-1 correction fixes the KRW span alone; a USD figure computed FROM
    # the pre-correction (10x-wrong) magnitude, elsewhere in the SAME
    # sentence, is untouched by that splice and now sits next to an
    # authoritative-looking corrected KRW figure while still being ~10x wrong
    # itself. Runs only over confirmed billion_10x/trillion_10x corrections
    # (never magnitude_suspect or any other flag-only kind — those have no
    # `expected_won` to judge a companion figure against).
    already_usd_spans: set[tuple[int, int]] = set()
    for v in [v for v in violations if v.kind in ("billion_10x", "trillion_10x") and v.expected_won is not None]:
        s_start, s_end = _sentence_span(report, v.start, v.end)
        sentence = report[s_start:s_end]
        for m in _USD_FIGURE_RE.finditer(sentence):
            seg_start, seg_end = m.start(), m.end()
            usd_text = m.group(0)
            # The currency-PREFIX alternative ("$28") never consumes a
            # trailing magnitude word — extend the span to pick it up (and
            # an optional trailing USD/dollars token) so both the value and
            # the redacted span cover the whole phrase ("$28 million USD"),
            # not just the digit prefix.
            if not _USD_MAGNITUDE_RE.search(usd_text):
                tail = _USD_TAIL_MAGNITUDE_RE.match(sentence, seg_end)
                if tail:
                    seg_end = tail.end()
                    usd_text = sentence[seg_start:seg_end]
            abs_start, abs_end = s_start + seg_start, s_start + seg_end
            if (abs_start, abs_end) in already_usd_spans:
                continue
            if _in_marker(abs_start, abs_end, markers):
                continue
            if abs_start < v.end and abs_end > v.start:
                continue  # overlaps the KRW span itself, not a companion figure
            usd_val = _usd_value(usd_text)
            if usd_val is None or usd_val <= 0:
                continue
            implied_rate = v.expected_won / usd_val
            if 500.0 <= implied_rate <= 3000.0:
                continue  # plausible KRW/USD rate — a genuinely unrelated, correct USD figure
            already_usd_spans.add((abs_start, abs_end))
            derived_usd_stale += 1
            violations.append(UnitScaleViolation(
                kind="derived_usd_stale", written=usd_text, expected=None,
                start=abs_start, end=abs_end,
            ))

    # Self-contradiction check — independent of eok_values, catches a
    # derived (not directly ground-truth-matched) figure like the
    # 두산에너빌리티 case. Skips a span already flagged above to avoid a
    # double-count on the same text.
    already = {(v.start, v.end) for v in violations}
    for m in _PAREN_RE.finditer(report):
        if (m.start(), m.end()) in already or _in_marker(m.start(), m.end(), markers):
            continue
        billions, trillions = _to_float(m.group(1)), _to_float(m.group(2))
        if not _isclose(billions / 1000, trillions, _PAREN_REL_TOL):
            suspect += 1
            violations.append(UnitScaleViolation(
                kind="self_inconsistent", written=m.group(0), expected=None,
                start=m.start(), end=m.end(),
            ))

    # 억원<->billion pair check (2026-09-15, id=1432/YTN) — independent of
    # `eok_values` for DETECTION (the 억원 figure being compared is the
    # report's OWN claimed mention, not the ground-truth set), which mirrors
    # `self_inconsistent`'s unconditional posture above. `eok_values` is used
    # only for the GATING decision (auto-correct vs flag-only) below.
    #
    # Runs after the Tier-1 billion_10x/trillion_10x loops and after
    # self_inconsistent, and never touches a span any of those three already
    # own — Tier 1 (and self_inconsistent) keep precedence on spans they
    # already explain. A span whose ONLY existing flag is the vaguer
    # `magnitude_suspect` is SUPERSEDED: a 10x-off billion figure sitting
    # right next to its own 억원 source is, by construction, also a
    # power-of-10 ladder near-miss of the same ground truth, so
    # `_magnitude_suspect` will typically have already flagged this exact
    # span — this is the same upgrade precedent `news_eok_10x` applies over
    # `magnitude_suspect` a few passes below, and for the same reason: a
    # named, arithmetic pair is a sharper finding than a bare ladder
    # collision.
    hard_owned_spans = {
        (v.start, v.end) for v in violations
        if v.kind in ("billion_10x", "trillion_10x", "self_inconsistent")
    }
    eok_pair_mag_suspect = {
        (v.start, v.end): v for v in violations if v.kind == "magnitude_suspect"
    }
    for em in _EOK_REPORT_RE.finditer(report):
        if _in_marker(em.start(), em.end(), markers):
            continue
        try:
            eok_val = _to_float(em.group(1))
        except ValueError:
            continue
        if eok_val == 0:
            continue
        window_start = em.end()
        bm = _BILLION_RE.search(report, window_start, min(len(report), window_start + _EOK_PAIR_WINDOW))
        if bm is None:
            continue
        span = (bm.start(), bm.end())
        if span in hard_owned_spans or _in_marker(bm.start(), bm.end(), markers):
            continue
        expected_billion = abs(eok_val) / 10.0
        if expected_billion == 0:
            continue
        av = abs(_mantissa(bm))
        # Flag ONLY a power-of-ten discrepancy (k in {1,2,3} or {-1,-2,-3}) —
        # this is the entire precision lever: a legitimately-rounded correct
        # conversion ("1,522억 원 (약 152 billion KRW)") is never a power of
        # ten off and so is never touched, and an arbitrary wrong figure
        # ("1,522억 원 (300 billion KRW)") is likewise left alone — only the
        # exact carried-mantissa shape id=1432 actually exhibited is caught.
        if not any(_isclose(av * (10.0 ** k), expected_billion, _REL_TOL) for k in (1, 2, 3, -1, -2, -3)):
            continue
        sign = -1.0 if eok_val < 0 else 1.0
        # Gating: the 억원 side is verified (matches a SUPPLIED figure) ->
        # the billion side is definitively the model's own arithmetic error
        # and the correct value is known -> auto-correct. Otherwise the pair
        # is merely self-contradictory (as with self_inconsistent) and
        # neither side is pinned down -> flag-only.
        gated = any(_isclose(abs(eok_val), abs(e), _REL_TOL) for e in eok_values)
        superseded = eok_pair_mag_suspect.pop(span, None)
        if superseded is not None:
            violations.remove(superseded)
            suspect -= 1
        if gated:
            corrected += 1
            violations.append(UnitScaleViolation(
                kind="eok_pair_10x", written=bm.group(0),
                expected=f"{sign * expected_billion:,.1f} billion KRW",
                start=bm.start(), end=bm.end(),
            ))
        else:
            suspect += 1
            violations.append(UnitScaleViolation(
                kind="eok_pair_10x", written=bm.group(0), expected=None,
                start=bm.start(), end=bm.end(),
            ))

    # USD-trillion check — opt-in, flag-only, at most one note per text. No
    # ground truth can exist (the figure came from the model's own priors),
    # so there is nothing to correct TO.
    if usd_check:
        for m in _USD_TRILLION_RE.finditer(report):
            if _in_marker(m.start(), m.end(), markers):
                continue
            if _to_float(m.group(1)) >= _USD_SUSPECT_ABOVE:
                suspect += 1
                violations.append(UnitScaleViolation(
                    kind="usd_unit_suspect", written=m.group(0), expected=None,
                    start=m.start(), end=m.end(),
                ))
                break

    # Currency-composition check (2026-09-03, id=1207) — opt-in, flag-only.
    # The two regex passes ALWAYS run (pure observation for `UnitScaleScan`);
    # only the `currency_inverted` judgment is gated on `currency_check` and
    # the premise gates.
    usd_spans = [
        m for m in _USD_FIGURE_RE.finditer(report)
        if not _in_marker(m.start(), m.end(), markers)
    ]
    krw_figures = sum(
        1 for m in _KRW_TOKEN_RE.finditer(report)
        if not _in_marker(m.start(), m.end(), markers)
    )
    usd_figures = len(usd_spans)
    currency_inverted = False
    if (
        currency_check
        and len(eok_values) >= _MIN_GT_FOR_CURRENCY
        and len(report) >= _MIN_REPORT_CHARS
        and not _CURRENCY_BANNER_MARK_RE.search(report)
    ):
        tier_a = krw_figures == 0 and usd_figures >= _MIN_USD_FIGURES
        tier_b = (
            krw_figures <= 1
            and usd_figures >= 8
            and usd_figures >= 5 * max(krw_figures, 1)
        )
        if (tier_a or tier_b) and usd_spans:
            first = usd_spans[0]
            currency_inverted = True
            violations.append(UnitScaleViolation(
                kind="currency_inverted",
                written=(
                    f'{first.group(0)} (+{usd_figures - 1} more USD figures, '
                    f'{krw_figures} KRW figures, {len(eok_values)} supplied 억원 values)'
                ),
                expected=None,
                start=first.start(), end=first.end(),
            ))

    # Soft (news-headline) ground truth — 2026-09-07 audit #3, id=1237. A
    # candidate that survived the hard passes above (no exchange-reported 억원
    # value explained it) is checked against the 억/조 amounts a news headline
    # wrote this cycle. `av ≈ s/divisor` is the model correctly quoting the
    # supplied KRW UNIT KEY — accepted, so it does not read as a blind spot.
    # `av ≈ s` is the 10x carry-through — flagged, never auto-corrected (the
    # ground truth is untrusted headline text). Runs even when `eok_values`
    # is empty.
    #
    # A span already carrying a Tier-1 correction or a stronger flag
    # (self_inconsistent, usd_unit_suspect) is left alone. But a span whose
    # ONLY prior flag is `magnitude_suspect` — the vaguest kind, a bare
    # power-of-10 ladder near-miss with no named ground truth — is UPGRADED
    # to `news_eok_10x` when it also matches a headline amount digit for
    # digit (2026-09-09, id=1358 / 047040.KS: a 4,134.3억원 contract written
    # "4,134.3 billion KRW" got the vague magnitude_suspect label and its
    # generic banner, while a headline amount matched it exactly and the
    # specific news_eok_10x note — "the headline figure is one tenth of this"
    # — is what the desk actually needed). Both kinds are `expected=None`, so
    # this only sharpens the label, the banner, and the news_eok_flagged
    # counter; it never changes an auto-correction.
    news_eok_flagged = 0
    if soft_eok_values:
        mag_suspect_spans = {
            (v.start, v.end): v for v in violations if v.kind == "magnitude_suspect"
        }
        handled = {(v.start, v.end) for v in violations if v.kind != "magnitude_suspect"}
        for rx, divisor in ((_BILLION_RE, 10.0), (_TRILLION_RE, 10_000.0)):
            for m in rx.finditer(report):
                span = (m.start(), m.end())
                if span in handled or _in_marker(m.start(), m.end(), markers):
                    continue
                av = abs(_mantissa(m))
                tol = _tol_for(report, m)
                # A figure the hard pass already accepted against exchange
                # data is correct — never re-judge it against a headline.
                if any(_isclose(av, abs(e) / divisor, tol) for e in eok_values):
                    continue
                if any(_isclose(av, abs(s) / divisor, tol) for s in soft_eok_values):
                    # the model correctly quoted the supplied KRW UNIT KEY —
                    # a headline match is a better explanation than a bare
                    # power-of-10 ladder collision, so clear any magnitude_suspect
                    # that landed on this span too.
                    superseded = mag_suspect_spans.get(span)
                    if superseded is not None:
                        violations.remove(superseded)
                        suspect -= 1
                    accepted += 1
                    continue
                if any(_isclose(av, abs(s), tol) for s in soft_eok_values):
                    news_eok_flagged += 1
                    superseded = mag_suspect_spans.get(span)
                    if superseded is not None:
                        violations.remove(superseded)
                        suspect -= 1
                    violations.append(UnitScaleViolation(
                        kind="news_eok_10x", written=m.group(0), expected=None,
                        start=m.start(), end=m.end(),
                    ))

    unverified = max(0, examined - accepted - corrected - suspect - news_eok_flagged)
    scan = UnitScaleScan(
        node=node,
        candidates_examined=examined,
        ground_truth_values=len(eok_values),
        corrected=corrected,
        flagged=len([v for v in violations if v.expected is None]),
        unverified=unverified,
        usd_figures=usd_figures,
        krw_figures=krw_figures,
        currency_inverted=currency_inverted,
        soft_ground_truth_values=len(soft_eok_values),
        news_eok_flagged=news_eok_flagged,
        derived_usd_stale=derived_usd_stale,
    )
    return violations, scan


def check_unit_scale(report: str, eok_values: frozenset[float]) -> list[UnitScaleViolation]:
    """Backwards-compatible thin wrapper over `scan_unit_scale` — every
    existing caller and test unpacks a plain violation list."""
    return scan_unit_scale(report, eok_values)[0]


_BANNER = {
    "ko": (
        "⚠️ 단위 오류 정정: 이 리포트는 억원 단위 수치를 \"billion\"으로 옮기며 10배 오표기했습니다"
        "(1억원 = 0.1 billion KRW). 해당 수치는 아래에서 정정되어 [단위 정정: 원문 \"...\"]로 "
        "표시됩니다. 정정된 값을 사용하고, 원래의 잘못된 크기만 보고 이 데이터가 다른 회사의 것이라고 "
        "판단하지 마세요.\n\n"
    ),
    "en": (
        '⚠️ UNIT-SCALE CORRECTION: this report misstated one or more KRW figures by 10x '
        '(an 억원 figure\'s digits were carried straight into "billion" — 1억원 = 0.1 billion KRW). '
        "The figures below have been corrected inline, marked [unit corrected from \"...\"]. Use "
        "the corrected values; do NOT infer from the original wrong magnitude that this data "
        "belongs to a different company.\n\n"
    ),
}

# Flag-only, banner-form (2026-09-03, id=1207). HARD CONSTRAINT: neither string
# may contain `₩`, literal `KRW`, `Korean won`, a digit-adjacent 원/억원/조원/
# 만원/천원, a digit+won, or `$digit` — any of those trips
# `identity_gate._KRW_AMOUNT_RE` and silently disarms Detector C (which is what
# currently catches id=1207). Do not reword without re-checking against that
# regex (see tests/agents/test_unit_scale_guard.py test 12).
_CURRENCY_INVERTED_BANNER = {
    "ko": (
        "⚠️ 통화 구성 이상 — 이 리포트의 수치 전체를 근거로 쓰지 마십시오: 국내(KRX) 상장 종목에 대한 "
        "분석인데도, 이 리포트에는 원화로 표기된 수치가 하나도 없고 달러 표기 수치만 반복해서 제시됩니다. "
        "거래소가 제공한 원화 재무제표를 이 리포트는 전혀 사용하지 않았습니다. 다른 회사(해외 상장 종목)의 "
        "자료를 통째로 옮겨 썼을 가능성이 큽니다. 아래의 재무수치·목표주가·지분율·성장률은 어느 하나도 이 "
        "종목의 근거로 사용하지 말고, 단위만 잘못된 것으로 취급하지도 마십시오 — 회사 자체가 다를 수 "
        "있습니다. 이 리포트를 뺀 나머지 리포트로 판단하십시오.\n\n"
    ),
    "en": (
        "⚠️ CURRENCY-COMPOSITION ALERT — do NOT use any figure in this report: this is an analysis "
        "of a Korea-listed (KRX) name, yet the report states no Korean-currency figure at all and "
        "repeatedly states dollar figures instead. It made no use of the exchange-supplied domestic "
        "financial statements. This is most likely an entire report about a DIFFERENT, foreign-listed "
        "company. Treat every financial figure, price target, ownership stake and growth rate below "
        "as unusable evidence for this ticker — this is not a unit error that can be rescaled; the "
        "company itself may be wrong. Reason from the other analysts' reports instead.\n\n"
    ),
}

_SELF_INCONSISTENT_NOTE = {
    "ko": "\n\n(참고: 위 리포트에 billion/trillion KRW 표기가 서로 10배 어긋나는 자체모순 구간이 있습니다 — 원문 그대로 두었으니 두 숫자 중 근거자료와 대조해 확인하세요.)",
    "en": '\n\n(Note: this report also contains a self-inconsistent billion/trillion KRW pair — left '
          "as-is since no supplied ground truth pins down which side is correct; cross-check against "
          "the source figures.)",
}

# Flag-only kinds (Tier 2) — a note is appended, no number is rewritten.
_MAGNITUDE_SUSPECT_NOTE = {
    "ko": "\n\n(참고: 위 리포트의 KRW 수치 중 근거 억원 값의 10의 거듭제곱 배와 근사하게 어긋나는 구간이 있습니다 — 자동 정정은 하지 않았으니 크기 단위를 근거자료와 대조하세요.)",
    "en": "\n\n(Note: one or more KRW figures above are a near power-of-10 off a supplied 억원 "
          "value — NOT auto-corrected (a weaker inference than an exact 10x); verify the magnitude "
          "against the source figures.)",
}
_USD_SUSPECT_NOTE = {
    "ko": "\n\n(참고: 위 리포트에 \"$N trillion\" 규모의 USD 표기가 있습니다 — 한국 원화 예산/수치를 달러 조 단위로 잘못 옮겼을 가능성이 있으니 확인하세요. 원문은 그대로 두었습니다.)",
    "en": "\n\n(Note: this report states a USD figure of $1 trillion or more — for a single-name "
          "Korean equity that is unusual and may be a KRW figure rendered as USD trillions. Left "
          "as-is; verify against the source.)",
}
# news_eok_10x (2026-09-07 audit #3). HARD CONSTRAINT, same as
# _CURRENCY_INVERTED_BANNER: no `₩`, literal `KRW`, `Korean won`, a
# digit-adjacent 원/억원/조원, a digit+won, or `$digit` — any of those trips
# identity_gate._KRW_AMOUNT_RE and silently disarms Detector C. Flag-only:
# the ground truth is untrusted headline text, so nothing is rewritten.
_NEWS_EOK_NOTE = {
    "ko": "\n\n(참고: 위 수치 중 하나 이상이 이번 사이클 한국 뉴스 헤드라인이 억 단위로 쓴 금액과 자릿수까지 그대로 일치합니다 — 헤드라인이 억 단위로 쓴 금액은 여기 적힌 값의 10분의 1입니다. 헤드라인은 제3자 텍스트라 자동 정정하지 않았으니, 근거로 삼기 전에 크기를 헤드라인과 대조하세요.)",
    "en": "\n\n(Note: one or more figures above match, digit for digit, an amount a Korean news "
          "headline this cycle wrote in the 억 unit — a headline amount in that unit is one tenth "
          "of the figure as written here. Left as-is since a headline is third-party text; check "
          "the magnitude against the headline before relying on it.)",
}
# eok_pair_10x, flag-only case only (2026-09-15, id=1432/YTN): the report
# stated its own 억원 figure and its own billion conversion in the same
# breath, off by a power of ten, but the 억원 figure did not match any
# SUPPLIED ground truth — so unlike the auto-corrected case (which needs no
# note; the span is already fixed and covered by `_BANNER`), neither side of
# this pair is verified and it is left in place, same posture as
# `_SELF_INCONSISTENT_NOTE`.
_EOK_PAIR_NOTE = {
    "ko": "\n\n(참고: 위 리포트가 억원 수치와 자신이 계산한 billion 환산치를 같은 문장에 함께 적었는데, 두 값이 10의 거듭제곱 배 어긋납니다 — 이 억원 값은 공급된 근거자료와 일치하지 않아 자동 정정하지 않았으니, 어느 쪽이 맞는지 원자료와 대조하세요.)",
    "en": "\n\n(Note: this report states its own 억원 figure and its own billion-KRW conversion in "
          "the same breath, and the two disagree by a power of ten — NOT auto-corrected, because "
          "this 억원 figure does not match any supplied ground truth; cross-check which side is "
          "correct against the source data.)",
}

# derived_usd_stale (2026-09-11, id=1396 / 003350.KS): a companion USD figure
# computed from a pre-correction KRW magnitude, redacted in place (not just
# noted) — see the `derived_usd_stale` pass in `scan_unit_scale`. HARD
# CONSTRAINT, same reason as `_CURRENCY_INVERTED_BANNER`/`_NEWS_EOK_NOTE`: no
# `$`, no digit, no won/KRW token — a marker containing either would perturb
# `identity_gate` Detector C's USD-figure/KRW-figure count over the whole
# report.
_DERIVED_USD_STALE_NOTE = {
    "en": "[USD equivalent removed — derived from an uncorrected figure]",
    "ko": "[USD 환산액 삭제 — 정정 전 수치에서 파생된 값]",
}


def enforce_unit_scale(
    report: str,
    kr_block: str,
    *,
    language: str = "en",
    agent_name: str = "",
    usd_check: bool = False,
    currency_check: bool = False,
    soft_eok_block: str = "",
    supplied_kr_block: bool = False,
    banner: bool = True,
    inline_note: bool = True,
    scan_out: list[UnitScaleScan] | None = None,
) -> tuple[str, list[str]]:
    """(possibly corrected + banner-prefixed report, violation descriptions).
    Unlike `identity_gate.enforce_identity` (banner-only — it has no way to
    know the right company name), this DOES know the right number, so a
    `billion_10x`/`trillion_10x` violation gets its span replaced in place
    (not just banner-flagged) — the whole point being that downstream stages
    (bull/bear/research manager/trader/PM) read the CORRECTED figure, not a
    banner they may not weigh as strongly as the number itself. The flag-only
    kinds `self_inconsistent`, `magnitude_suspect`, `usd_unit_suspect`, and
    `currency_inverted` have no known-good replacement and are left in
    place — `currency_inverted` gets a leading banner (id=1207), the others a
    trailing note. `derived_usd_stale` (2026-09-11, id=1396) is the one
    flag-only kind that DOES get a replacement: its companion USD figure
    (computed from a pre-correction KRW magnitude) is redacted in place with
    a fixed, currency-token-free marker, same splice pass as the
    billion_10x/trillion_10x corrections.

    `soft_eok_block`, if given, is a `get_kr_news` headline block: its 억/조
    money amounts become a SOFT ground-truth set. A report figure matching
    one of them digit-for-digit (10x carry-through) is flagged `news_eok_10x`
    — never auto-corrected, because a headline is untrusted text (2026-09-07
    audit #3, id=1237). A figure matching a headline amount's correct
    converted value is accepted, so it does not read as a blind spot.

    `usd_check` enables the opt-in `$N trillion` scan (`.KS` pipeline path
    only). `currency_check` enables the opt-in currency-composition net
    (Fundamentals Analyst only — see the module docstring's 2026-09-03
    addendum). `supplied_kr_block` (Fundamentals Analyst only) gates the
    id=1158 "SUSPECTED BLIND SPOT" `logger.warning`: that heuristic's premise
    is block-readership ("was handed the 억원 block, matched none of it"), but
    only the Fundamentals Analyst is ever handed that block — `pipeline.py`
    reuses the same ground-truth text for every other node, which merely
    quotes news-sourced or derived KRW figures. Live 2026-08-29→09-04: the
    warning fired 88 times, ALL false alarms, from nodes that never received
    the block, while the Fundamentals Analyst hit 0/56. The warning is still
    the right id=1158 tripwire for the node that DID read the block, so it is
    gated, not removed; the `logger.info` scan line just above still records
    examined/corrected/unverified for every node regardless.
    `banner=False` suppresses BOTH the leading banner AND the trailing
    flag-only advisory notes (self_inconsistent / magnitude_suspect /
    usd_unit_suspect / news_eok_10x) for a structured field whose siblings
    would each repeat them — the finding still reaches the caller via the
    returned descriptions. `inline_note=False` replaces a span
    with just the corrected value and no `[unit corrected from "..."]` marker
    — for length-budgeted card text; the violation is still recorded, so this
    is an accounted-for silent correction, not an invisible one. `scan_out`,
    if given, is appended with this run's `UnitScaleScan`.

    Replacements are applied in REVERSE span order so each edit's offset
    doesn't shift the positions of earlier-found spans still to be applied."""
    eok_values = extract_kr_eok_values(kr_block)
    violations, scan = scan_unit_scale(
        report, eok_values, usd_check=usd_check, currency_check=currency_check,
        soft_eok_values=extract_kr_headline_eok_values(soft_eok_block),
        node=agent_name,
    )
    logger.info(
        "unit_scale_guard: %s scan — examined=%d ground_truth=%d corrected=%d flagged=%d unverified=%d",
        agent_name or "?", scan.candidates_examined, scan.ground_truth_values,
        scan.corrected, scan.flagged, scan.unverified,
    )
    # The blind-spot warning's premise is "this node was GIVEN the Korean 억원
    # block and still matched none of it" — so it is only meaningful for the one
    # node the block is actually supplied to (the Fundamentals Analyst; see
    # this function's docstring for the `supplied_kr_block` rationale).
    # `pipeline.py` passes the same
    # `kr_gt` block as ground truth to every other node's guard call, so
    # `ground_truth_values > 0` is true for nodes that never saw the data and
    # are legitimately quoting news-sourced / derived KRW figures. Live
    # measurement 2026-08-29→09-04: 88/88 false alarms, Fundamentals at 0/56.
    # 억원 블록을 실제로 받은 노드에서만 이 경고가 유효함 (opt-in 게이트).
    if (
        supplied_kr_block
        and scan.ground_truth_values > 0
        and scan.candidates_examined > 0
        and scan.corrected == 0
        and scan.unverified == scan.candidates_examined
    ):
        logger.warning(
            "unit_scale_guard: %s quoted %d large-KRW figure(s) against %d supplied 억원 "
            "value(s) and matched NONE of them — an empty violation list here is a "
            "SUSPECTED BLIND SPOT, not a clean report (see decision id=1158, 2026-09-01)",
            agent_name or "?", scan.candidates_examined, scan.ground_truth_values,
        )
    if scan.currency_inverted:
        logger.warning(
            "unit_scale_guard: %s report on a KRX name states %d USD figure(s) and %d "
            "KRW figure(s) against %d supplied 억원 value(s) — currency composition is "
            "INVERTED; this is most likely an entire report about a different, "
            "foreign-listed company, not a unit error (see decision id=1207 / "
            "270520.KS, 2026-09-03)",
            agent_name or "?", scan.usd_figures, scan.krw_figures, scan.ground_truth_values,
        )
    if scan.news_eok_flagged:
        # 2026-09-07 audit #3 (id=1237): the first observed true positive the
        # #211 blind-spot gate suppresses for a non-Fundamentals node. This is
        # a NAMED finding (a report figure matches a headline 억 amount
        # digit-for-digit), not the "matched nothing" heuristic whose premise
        # stays false for the News node — so it is not gated on
        # `supplied_kr_block`.
        logger.warning(
            "unit_scale_guard: %s report has %d figure(s) matching a news headline's 억 amount "
            "digit-for-digit (suspected 10x carry-through) against %d headline amount(s) — "
            "flagged, not corrected (headline text is untrusted; see decision id=1237, 2026-09-07)",
            agent_name or "?", scan.news_eok_flagged, scan.soft_ground_truth_values,
        )
    if scan_out is not None:
        scan_out.append(scan)
    if not violations:
        return report, []

    correctable = [v for v in violations if v.expected is not None]
    usd_redactions = [v for v in violations if v.kind == "derived_usd_stale"]
    kinds = {v.kind for v in violations}
    corrected = report
    # Both edit kinds (billion_10x/trillion_10x corrections and
    # derived_usd_stale redactions) reference offsets in the SAME original
    # (pre-edit) `report` text and never overlap (the overlap check in the
    # `derived_usd_stale` scan pass guarantees this), so they are combined
    # into ONE reverse-sorted-by-start splice pass — two sequential passes
    # would corrupt offsets the second pass reads against the already-edited
    # text.
    for v in sorted(correctable + usd_redactions, key=lambda v: v.start, reverse=True):
        if v.kind == "derived_usd_stale":
            replacement = _DERIVED_USD_STALE_NOTE.get(language, _DERIVED_USD_STALE_NOTE["en"])
        else:
            replacement = (
                f'{v.expected} [unit corrected from "{v.written}"]' if inline_note else v.expected
            )
        corrected = corrected[: v.start] + replacement + corrected[v.end :]

    descriptions = [f"{v.kind}: {v.written!r} -> {v.expected!r}" for v in violations]
    logger.warning(
        "unit_scale_guard: %s report had %d unit-scale violation(s): %s",
        agent_name, len(violations), descriptions,
    )
    if correctable and banner:
        corrected = _BANNER.get(language, _BANNER["en"]) + corrected
    if "currency_inverted" in kinds and banner:
        corrected = _CURRENCY_INVERTED_BANNER.get(
            language, _CURRENCY_INVERTED_BANNER["en"]
        ) + corrected
    # The trailing flag-only notes follow the SAME `banner` gate as the two
    # leading banners above (2026-09-09): a `banner=False` call is a
    # structured / length-budgeted field (`_guard_fields` — research_plan,
    # trader_proposal, PM executive_summary/investment_thesis, and the
    # length-clamped card_reason_ko/en), and sibling fields would each append
    # the same ~160-char note. The finding is still recorded in the returned
    # descriptions and the `logger.warning` above regardless.
    if banner:
        if "self_inconsistent" in kinds:
            corrected += _SELF_INCONSISTENT_NOTE.get(language, _SELF_INCONSISTENT_NOTE["en"])
        if "magnitude_suspect" in kinds:
            corrected += _MAGNITUDE_SUSPECT_NOTE.get(language, _MAGNITUDE_SUSPECT_NOTE["en"])
        if "usd_unit_suspect" in kinds:
            corrected += _USD_SUSPECT_NOTE.get(language, _USD_SUSPECT_NOTE["en"])
        if "news_eok_10x" in kinds:
            corrected += _NEWS_EOK_NOTE.get(language, _NEWS_EOK_NOTE["en"])
        if any(v.kind == "eok_pair_10x" and v.expected is None for v in violations):
            corrected += _EOK_PAIR_NOTE.get(language, _EOK_PAIR_NOTE["en"])
    return corrected, descriptions
