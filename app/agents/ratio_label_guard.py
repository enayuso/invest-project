"""Catches a Fundamentals Analyst report that described KIS's 부채비율
(`lblt_rate` — total liabilities ÷ total EQUITY × 100) as an
ASSETS-denominated ratio: "372.75% (Liabilities/Assets)", "a 372.75%
debt-to-assets ratio", and so on. The number is correct; only the model's
own English gloss of the denominator is wrong.

Built for item #4 of the 2026-09-01 domestic AI audit (066970.KS/엘앤에프,
first real-account KRX cycle): the Fundamentals report wrote "Debt Ratio
remains elevated at 372.75% (Liabilities/Assets), though this metric is
distorted by the small equity base" — the same sentence that shows the model
KNEW it was equity-based still stamped the assets denominator on it — and the
Portfolio Manager restated it downstream as "a 372.75% debt-to-assets ratio"
(decision id=1121). Root cause: `app.dataflows.kis_provider.get_kr_financials`
labeled the field a bare `debt_ratio=X%` with no denominator stated, and
"debt ratio" in English financial writing usually DOES mean liabilities ÷
assets — so the model filled the gap with the Western reading. That prompt is
now fixed at the source (the key is `debt_to_equity_ratio`, glossed inline —
see its own 2026-09-01 comment); this module is the detection net for
whatever a model still glosses wrong anyway, the same detection-net
relationship `app.agents.unit_scale_guard` has to `_eok`.

Deliberately a sibling of, not a new kind inside, `unit_scale_guard`: that
module is scoped to a UNIT-CONVERSION magnitude error and its entire
apparatus is a 10x comparison against the supplied 억원 ground-truth set.
This is a METRIC-IDENTITY error — the magnitude is right, the metric's name
is wrong — and needs no external ground truth at all: a liabilities/assets
ratio above 100% is impossible by definition, so a >100% value carrying
assets-denominated wording is self-refuting on its face. Same boundary
`unit_scale_guard`'s own docstring draws when it hands identity-shaped harms
to `app.agents.identity_gate`. It does, however, ride `unit_scale_guard`'s
existing out-param plumbing (`fundamentals_analyst.analyze`'s
`unit_violations_out` -> `PipelineState.unit_scale_violations`) rather than
duplicating a parallel transcript field, since that channel already does
exactly the three things this needs: audit record, UI display, and blocking
the report cache for a corrected run.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# An assets-denominated leverage phrase — "liabilities/assets", "debt-to-assets",
# "부채/자산", "자산 대비 부채" and close variants. Deliberately does NOT match
# an equity-denominated phrase ("liabilities/equity", "자기자본" etc.) — those
# are the CORRECT gloss for KIS's 부채비율 and must never be flagged.
_ASSETS_DENOM = (
    r"(?:"
    r"liabilit(?:y|ies)\s*(?:/|÷|\s*-?\s*to\s*-?\s*|\s+over\s+)\s*(?:total\s+)?assets"
    r"|debts?\s*(?:/|÷|\s*-?\s*to\s*-?\s*)\s*(?:total\s+)?assets"
    r"|(?:총)?자산\s*(?:총계\s*)?대비\s*(?:총)?부채"
    r"|(?:총)?부채\s*(?:/|÷)\s*(?:총)?자산"
    r")"
)
_PCT = r"(\d[\d,]*(?:\.\d+)?)\s*%"

# "372.75% (Liabilities/Assets)" / "372.75% debt-to-assets" — the percentage
# comes first, the (wrong) denominator gloss follows, possibly inside
# parentheses and/or with a few words in between ("a 372.75% debt-to-assets
# ratio").
_PCT_THEN_DENOM_RE = re.compile(
    _PCT + r"[\s,]*(?:[\(\[]\s*)?(?:\w+[\s-]+){0,3}?" + _ASSETS_DENOM,
    re.IGNORECASE,
)
# "Its debt-to-assets ratio of 372.75%" / "Liabilities/assets stands at 78.8%"
# — the denominator gloss comes first, the percentage follows.
_DENOM_THEN_PCT_RE = re.compile(
    _ASSETS_DENOM + r"\s*(?:[\)\]])?(?:[\s,]*\w+){0,4}?[\s,]*(?:of|at|:)?\s*" + _PCT,
    re.IGNORECASE,
)

# A liabilities/assets ratio cannot exceed 100% by definition — that is
# exactly what makes an ASSETS-denominated gloss on a >100% figure
# self-refuting, and lets this module skip needing any external ground
# truth. A reading at or below 100% carrying assets-denominated wording is
# left alone (it's a legitimate borrowing_dependency-shaped statement, not
# this bug).
_IMPOSSIBLE_ABOVE_PCT = 100.0

# Skip any candidate overlapping an already-applied correction marker — the
# marker re-contains the offending span verbatim, so a second pass over
# corrected text (a debate turn quoting a corrected Fundamentals sentence,
# routine once the guard fans out across nodes — 2026-09-01) would re-flag
# and nest it.
_CORRECTION_MARKER_RE = re.compile(
    r"\[(?:label corrected from|unit corrected from|지표 정의 정정|단위 정정)[^\]]*\]"
)

_CORRECTED_GLOSS = {
    "ko": "{pct}% (부채비율 = 총부채 ÷ 자기자본, 자산 대비 비율이 아님)",
    "en": "{pct}% (부채비율 = total liabilities ÷ total EQUITY — an equity-denominated "
          "ratio, NOT liabilities/assets)",
}


@dataclass(frozen=True)
class RatioLabelViolation:
    kind: str
    written: str
    expected: str
    start: int
    end: int


def _to_float(token: str) -> float:
    return float(token.replace(",", ""))


def check_ratio_labels(report: str, *, language: str = "en") -> list[RatioLabelViolation]:
    """Finds every span in `report` that glosses an ABOVE-100% percentage as
    an assets-denominated ratio ("debt-to-assets", "liabilities/assets", ...)
    — impossible by definition, so no external ground truth is needed, unlike
    `unit_scale_guard.check_unit_scale`'s 억원 comparison. A reading at or
    below 100% is left alone entirely: that's a legitimate assets-denominated
    statement (e.g. borrowing_dependency), not this bug.

    Both word orders ("372.75% (Liabilities/Assets)" and "debt-to-assets ratio
    of 372.75%") are checked; a span matched by both is deduplicated by
    overlapping position so it is reported only once."""
    gloss = _CORRECTED_GLOSS.get(language, _CORRECTED_GLOSS["en"])
    markers = [(m.start(), m.end()) for m in _CORRECTION_MARKER_RE.finditer(report)]
    found: dict[tuple[int, int], RatioLabelViolation] = {}
    for pattern in (_PCT_THEN_DENOM_RE, _DENOM_THEN_PCT_RE):
        for m in pattern.finditer(report):
            value = _to_float(m.group(1))
            if value <= _IMPOSSIBLE_ABOVE_PCT:
                continue
            if any(s <= m.start() < e or s < m.end() <= e for s, e in found):
                continue
            if any(s <= m.start() < e or s < m.end() <= e for s, e in markers):
                continue
            found[(m.start(), m.end())] = RatioLabelViolation(
                kind="debt_ratio_denominator",
                written=m.group(0),
                expected=gloss.format(pct=m.group(1)),
                start=m.start(),
                end=m.end(),
            )
    return sorted(found.values(), key=lambda v: v.start)


_BANNER = {
    "ko": (
        "⚠️ 지표 정의 정정: 이 리포트는 부채비율(KIS `lblt_rate`)을 \"자산 대비 부채"
        "(liabilities/assets)\"로 잘못 설명했습니다. 부채비율 = 총부채 ÷ 자기자본 × 100"
        "(한국 회계 관행)이며 100%를 넘는 것이 정상입니다 — 자산 대비 비율은 정의상 100%를 "
        "넘을 수 없습니다. 해당 구간은 아래에서 정정되어 [지표 정의 정정: 원문 \"...\"]로 "
        "표시됩니다. 정정된 정의를 사용하고, 100%를 초과한다는 이유만으로 이 수치가 오류이거나 "
        "과장되었다고 판단하지 마세요.\n\n"
    ),
    "en": (
        '⚠️ RATIO-LABEL CORRECTION: this report described 부채비율 (KIS `lblt_rate`) as a '
        'liabilities/assets ("debt-to-assets") ratio. It is total liabilities ÷ total EQUITY '
        "× 100 — the Korean-market convention — and readings above 100% are normal; a "
        "liabilities/assets ratio cannot exceed 100% by definition. The spans below have been "
        'corrected inline, marked [label corrected from "..."]. Use the corrected definition, '
        "and do NOT treat the figure as erroneous, corrupted, or overstated merely because it "
        "exceeds 100%.\n\n"
    ),
}


def enforce_ratio_labels(
    report: str,
    *,
    language: str = "en",
    agent_name: str = "",
    banner: bool = True,
    inline_note: bool = True,
) -> tuple[str, list[str]]:
    """(possibly corrected + banner-prefixed report, violation descriptions).
    Every violation found is corrected in place (unlike a
    `unit_scale_guard`-style `self_inconsistent` case, a ratio-label
    violation always has a known-good replacement — the number is right,
    only the label was wrong), and replacements are applied in REVERSE span
    order so each edit's offset doesn't shift the positions of earlier-found
    spans still to be applied."""
    violations = check_ratio_labels(report, language=language)
    if not violations:
        return report, []

    corrected = report
    for v in sorted(violations, key=lambda v: v.start, reverse=True):
        replacement = (
            f'{v.expected} [label corrected from "{v.written}"]' if inline_note else v.expected
        )
        corrected = corrected[: v.start] + replacement + corrected[v.end :]

    descriptions = [f"{v.kind}: {v.written!r} -> {v.expected!r}" for v in violations]
    logger.warning(
        "ratio_label_guard: %s report had %d ratio-label violation(s): %s",
        agent_name, len(violations), descriptions,
    )
    if banner:
        corrected = _BANNER.get(language, _BANNER["en"]) + corrected
    return corrected, descriptions
