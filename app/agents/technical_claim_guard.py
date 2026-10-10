"""Detect-only recorder for a Market Analyst report that states an estimated/
hedged technical-indicator value instead of using the real verified indicator
block it is now always given (see `app.agents.market_analyst.analyze`'s
2026-09-07 fix, Fix 1), or that misattributes the KOSPI/KOSDAQ index's own
day-change as this specific stock's "contribution" to the index.

Built from a 21-day live audit (255 domestic Market Analyst reports): 35.7%
had NO real indicator value at all (the model skipped `get_indicators`, not a
data-availability gap — the same tickers got it some cycles and not others).
Of that skip rate, only ~2.4% (6/255) were papered over with a fabricated,
hedged value ("Estimated RSI: ~35-40 (oversold territory approaching extreme
lows)"); the rest honestly disclosed the value was unavailable. Roughly
5/255 hedged a moving-average crossover claim ("Likely crossed below
longer-term averages") the same way. Separately, id=1237 restated the
KOSPI index's own day-change as this small-cap stock's "contribution to the
KOSPI index" (1/521 reports) — a misattribution of the same shape as the
sector-attribution bug `_reaudit_sector_attribution` already guards
(#213/`796fa09`), not fabrication from nothing.

Fix 1 (always-injected indicator block) is the load-bearing fix — it removes
the non-determinism that let the model skip the tool in the first place.
Fix 2 (prompt clauses in `_BASE_SYSTEM_PROMPT`/`_KR_SYSTEM_PROMPT_ADDENDUM`)
tells the model plainly not to hedge or misattribute. THIS module is the same
"detect-only, no control-flow change" pattern as
`app.agents.risk_debators.check_neutral_reversal` and
`app.agents.halt_status_guard.scan_halt_claims`: positive evidence of whether
the prompt/injection fixes are actually working, not a gate.

Unlike `halt_status_guard`, this guard does not need a precondition on some
external ground-truth marker being confirmed first — the Market Analyst
report always has SOMETHING to check its indicator claims against now: either
`indicator_block` carries a real `get_indicators` value, or it explicitly
says NOT AVAILABLE (Fix 1's fallback text). So this scans every Market
Analyst report unconditionally. It also only ever scans the Market Analyst's
OWN report text — not the debate/PM turns that merely quote it — since those
downstream turns did not originate the claim (narrower scan scope than
`halt_status_guard`'s multi-source scan, deliberately)."""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# RSI in either language, avoiding a partial match inside a longer identifier.
_RSI_TERM = r"(?:(?<![A-Za-z])RSI(?![A-Za-z])|상대강도지수|Relative Strength Index)"

# A hedge word sitting within ~80 chars after an RSI mention, followed by a
# 1-3 digit (optionally decimal) number — "Estimated RSI: ~35-40",
# "RSI is roughly 42", "RSI(14)는 약 55 수준". Deliberately EN+KR hedge terms
# in one alternation, same reasoning as halt_status_guard's bilingual terms.
_HEDGED_RSI_RE = re.compile(
    rf"{_RSI_TERM}[^.\n]{{0,80}}?"
    r"(approximately|estimated|estimate[ds]?|roughly|around|about|assuming|likely|"
    r"presumably|approx\.?|추정|약|~)\s*"
    r"(\d{1,3}(?:\.\d+)?)",
    re.IGNORECASE,
)

# A moving-average/crossover claim hedged as inferred from price action
# rather than quoted from the verified indicator block.
_HEDGED_MA_RE = re.compile(
    r"(likely (?:crossed|rising|falling|below|above|hovering|sitting|trading)|"
    r"may have crossed|may be converging|presumably crossed|appears to have crossed|"
    r"probably (?:crossed|below|above)|e\.g\.,?\s*\d{1,3}-day|"
    r"이동평균[^.\n]{0,25}추정|추정[^.\n]{0,15}이동평균)",
    re.IGNORECASE,
)

# A claim that this STOCK contributed to/impacted the KOSPI/KOSDAQ index —
# id=1237's shape, restating the index's own day-change as this stock's
# contribution to it.
_INDEX_CONTRIB_RE = re.compile(
    r"(contribut\w*[^.\n]{0,80}?(?:KOSPI|KOSDAQ|index)|"
    r"(?:KOSPI|KOSDAQ|index)[^.\n]{0,60}?contribut|"
    r"지수[^.\n]{0,15}기여|기여도)",
    re.IGNORECASE,
)

# `format_indicator_line`'s exact RSI(14)= shape (app.dataflows.indicators) —
# used to pull the true value out of `indicator_block` for comparison.
_TRUE_RSI_RE = re.compile(r"RSI\(14\)=(\d{1,3}(?:\.\d+)?)")

_NOT_AVAILABLE_MARKER = "NOT AVAILABLE"


def _merge_spans(spans: list[tuple[int, int]], gap: int = 40) -> list[tuple[int, int]]:
    """Merge overlapping/near-adjacent (start, end) match spans into one —
    `_HEDGED_MA_RE`'s alternatives can each independently match within the
    SAME hedged sentence (e.g. both "e.g., 5-day" and "Likely crossed" fire
    inside "Short-Term MA (e.g., 5-day): Likely crossed below longer-term
    averages"), which without merging would double-count one hedged claim
    as two findings. `gap` is generous on purpose (a single sentence easily
    spans 40+ chars between two hedge phrases) — this guard's whole scope is
    narrow enough that over-merging within one report only ever costs a
    duplicate finding being collapsed, never a genuinely separate claim
    elsewhere in the report (those sit far apart and merge only by accident
    if `gap` were absurdly large, which it is not)."""
    spans = sorted(spans)
    merged: list[tuple[int, int]] = []
    for start, end in spans:
        if merged and start <= merged[-1][1] + gap:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


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


def scan_technical_claims(text: str, *, indicator_block: str, agent_name: str) -> list[str]:
    """Findings, one per hedged/ungrounded technical-indicator claim or
    index-contribution misattribution found in `text` (the Market Analyst's
    own report — do not pass debate/PM turns that merely quote it):

    - A hedged RSI claim ("Estimated RSI: ~35-40") is checked against
      `indicator_block`'s real `RSI(14)=` value (`_TRUE_RSI_RE`): if a real
      value is present, the finding is `rsi_estimate_vs_true` and includes
      both the claimed and true numbers; if `indicator_block` says NOT
      AVAILABLE instead, the finding is `rsi_ungrounded` (no true value to
      compare, but still an ungrounded hedge-claim).
    - A hedged moving-average/crossover claim is `ma_ungrounded`.
    - An index-contribution misattribution is `index_contribution`.

    `[]` for an unhedged claim that simply restates the real indicator value
    (the healthy majority case), and for the legitimate regime-line quoting
    pattern ("KOSPI: 6,902.44 (+215.23, +3.22%)") — neither carries a hedge
    word or a contribution claim, so neither regex fires. Detect-only — no
    control-flow change, no LLM call. Dedups identical spans within one call,
    same as `halt_status_guard.scan_halt_claims`."""
    if not text:
        return []

    findings: list[str] = []
    seen: set[str] = set()

    for m in _HEDGED_RSI_RE.finditer(text):
        span = _quoted_span(text, m.start(), m.end())
        if span in seen:
            continue
        claimed = m.group(2)
        true_match = _TRUE_RSI_RE.search(indicator_block or "")
        if true_match:
            seen.add(span)
            findings.append(
                f"{agent_name}: claimed RSI {claimed!r} vs true RSI(14)={true_match.group(1)!r} — {span!r}"
            )
        elif _NOT_AVAILABLE_MARKER in (indicator_block or ""):
            seen.add(span)
            findings.append(f"{agent_name}: ungrounded hedged RSI claim {claimed!r} — {span!r}")

    ma_spans = _merge_spans([(m.start(), m.end()) for m in _HEDGED_MA_RE.finditer(text)])
    for start, end in ma_spans:
        span = _quoted_span(text, start, end)
        if span in seen:
            continue
        seen.add(span)
        findings.append(f"{agent_name}: ungrounded hedged moving-average claim — {span!r}")

    contrib_spans = _merge_spans([(m.start(), m.end()) for m in _INDEX_CONTRIB_RE.finditer(text)])
    for start, end in contrib_spans:
        span = _quoted_span(text, start, end)
        if span in seen:
            continue
        seen.add(span)
        findings.append(f"{agent_name}: index-contribution misattribution — {span!r}")

    if findings:
        logger.warning(
            "technical_claim_guard: %s made an ungrounded technical-indicator or "
            "index-contribution claim (id=1237/id=1238-adjacent shape) — %s",
            agent_name, findings,
        )
    return findings
