"""Detect-only recorder for an agent citing a PER/PBR/EPS/BPS or program-
trading figure as coming from the pipeline's own supplied ground truth (`kr_gt`
/ `kr_risk_extra`) when the supplied value materially disagrees with the
figure the agent actually wrote — either direction — plus the more
decision-damaging shape of that same bug: an agent "correcting" another
agent's numeric claim as fabricated/wrong when the ORIGINAL claim actually
matched ground truth and the "corrected" replacement does not.

Built for two real incidents:

  - id=1394 (a PBR shape): the supplied `[Valuation]` block gave PBR=0.55,
    but a report wrote "priced at 2.44x P/B" attributed to the data/ground
    truth — a ~4.4x mismatch (well beyond a rounding difference), silently
    adopted downstream.

  - id=1386 (006360.KS, program trading): ground truth was
    `[Program trading] latest net-buy: +406,376 shares (+156.5억원)`
    (156.5억원 = 15.65 billion KRW). One debater correctly wrote "+15.65
    billion KRW". A SECOND debater then wrote, in the actual live
    transcript: "the Aggressive debater asserts that Program Trading data
    shows \"+15.65 billion KRW\" in buys, whereas the provided data table
    explicitly lists \"+156.5억원\" (1.565 billion KRW). An error of
    magnitude ten (15.65bn vs 1.56bn) is not a typo; it is a fabrication of
    material facts." The TRUE ground truth (156.5억원 -> 15.65 billion) was
    accused of being a fabrication, and the accuser's own "1.565 billion" is
    the actually-wrong figure (156.5 / 100, not / 10). Both the Research
    Manager and Portfolio Manager adopted this wrong "correction" as
    decisive reasoning. `app.agents.unit_scale_guard._magnitude_suspect`'s
    2026-09-11 `k=1` addition (same incident) catches the wrong figure as a
    bare power-of-10 ladder near-miss; this module catches the SAME incident
    from a different angle — the attribution/accusation framing around it,
    not just the number.

Chosen scope for the "false correction" sub-case (documented honestly, not
overclaimed): this module does NOT attempt to reliably parse "which span is
the accused original value" vs "which span is the accuser's own replacement
value" out of free-form prose via regex — that discriminator would be
fragile and error-prone on real transcript text. Instead it takes the
simpler, more robust v1 the task's own design notes call out as acceptable:
flag ANY numeric claim — attributed to ground truth OR sitting inside a
correction/accusation frame ("fabrication", "tenfold", "10x",
"exaggerat...", "조작", "오류", "잘못") — whose value materially disagrees
with the supplied ground truth in EITHER direction. A claim that itself
matches ground truth is never flagged, regardless of whether it is
surrounded by an accusation frame (this is what keeps the correct "+15.65
billion KRW" clean even though it sits one sentence away from the
"fabrication" accusation) — so the WRONG value ("+1.565 billion KRW") gets
flagged while the CORRECT value it is being contrasted against does not,
without needing to know which one is "the accusation" and which is "the
replacement". The returned string is tagged distinctly (`— false correction`
suffix) whenever a correction/accusation frame co-occurs near the mismatched
claim, since that combination is the more severe, decision-damaging finding
this guard exists for; a plain mismatched claim with no such frame nearby is
tagged as an ordinary mismatch instead.

Detect-only — no control-flow change, no LLM call, not cache-gating. Modeled
on `app.agents.regulatory_mechanism_guard`'s structure (module shape,
`supplied_fields`-style premise via a `ground_truth` dict instead, sentence-
bounded span dedup) and on `app.agents.derived_value_mismatch_guard`'s
ratio-based mismatch threshold (`_MISMATCH_THRESHOLD`, copied by convention,
not by import)."""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# Same convention as app.agents.derived_value_mismatch_guard._MISMATCH_THRESHOLD:
# a claim is flagged only when it is off from the ground-truth value by at
# least this factor in either direction — a normal rounding/estimation gap
# must never trip this. id=1394's PBR shape (2.44 vs 0.55, ~4.4x) and
# id=1386's program-trading shape (1.565 vs 15.65, 10x) both clear this with
# large margin.
_MISMATCH_THRESHOLD = 2.0

_NUM = r"(-?[\d,]+(?:\.\d+)?)"
# kis_provider's program-trading amount is always rendered with an explicit
# sign (`{amount/1e8:+,.1f}` — see _PROGRAM_TRADING_RE below), so its number
# group must accept a leading "+" as well as "-", unlike the PER/PBR/EPS/BPS
# fields in [Valuation] (kis_provider's `.2f`/`,.0f` formats never emit a
# leading "+").
_SIGNED_NUM = r"([+-]?[\d,]+(?:\.\d+)?)"

# --- Ground-truth extraction ---
# kr_gt's [Valuation] line (app.dataflows.kis_provider, ~line 1265-1270):
#   "  Current price {price:,.0f}원 | PER {per:.2f} | PBR {pbr:.2f} | "
#   "EPS {eps:,.0f} | BPS {bps:,.0f} | Market cap ..."
_VALUATION_RE = re.compile(
    rf"Current price\s*{_NUM}원\s*\|\s*PER\s*{_NUM}\s*\|\s*PBR\s*{_NUM}\s*\|\s*"
    rf"EPS\s*{_NUM}\s*\|\s*BPS\s*{_NUM}",
    re.IGNORECASE,
)

# kr_risk_extra's [Program trading] line (app.dataflows.kis_provider,
# ~line 1551-1554):
#   "\n[Program trading] latest net-buy: {qty:+,} shares "
#   "({amount/1e8:+,.1f}억원)"
_PROGRAM_TRADING_RE = re.compile(
    rf"\[Program trading\][^(]*\(\s*{_SIGNED_NUM}\s*억원\s*\)",
    re.IGNORECASE,
)


def extract_ground_truth(*, kr_gt: str = "", kr_risk_extra: str = "") -> dict[str, float]:
    """Parses PER/PBR/EPS/BPS out of `kr_gt`'s `[Valuation]` line and the
    program-trading net-buy 억원 amount (converted to billion KRW, for direct
    comparison with prose that cites "X billion KRW") out of
    `kr_risk_extra`'s `[Program trading]` line. Missing/absent blocks simply
    omit those keys — never raises, never fabricates a zero."""
    out: dict[str, float] = {}

    m = _VALUATION_RE.search(kr_gt or "")
    if m:
        price, per, pbr, eps, bps = (float(g.replace(",", "")) for g in m.groups())
        out["price"] = price
        out["per"] = per
        out["pbr"] = pbr
        out["eps"] = eps
        out["bps"] = bps

    m = _PROGRAM_TRADING_RE.search(kr_risk_extra or "")
    if m:
        eok = float(m.group(1).replace(",", ""))
        out["program_net_buy_billion"] = eok / 10.0

    return out


# --- Claim extraction ---

# An explicit attribution/ground-truth-citation frame — "the data shows X",
# "the provided table lists X", "ground truth states X", "per the data
# block", "데이터에 따르면", "제공된 표" — checked near a claimed numeric value.
_ATTRIBUTION_RE = re.compile(
    r"(?:the\s+)?(?:data|ground\s*truth|provided\s+(?:data\s+)?table|data\s+block|"
    r"supplied\s+data)\s+(?:shows?|lists?|states?|indicates?)|"
    r"per\s+the\s+data(?:\s+block)?|"
    r"데이터에\s*따르면|제공된\s*표",
    re.IGNORECASE,
)

# A correction/accusation frame — "this figure is incorrect, the actual
# value is X", "fabrication", "tenfold", "10x exaggeration", "조작"/"오류".
_CORRECTION_FRAME_RE = re.compile(
    r"fabricat\w*|tenfold|10x|exaggerat\w*|"
    r"this\s+figure\s+is\s+incorrect|the\s+actual\s+value\s+is|"
    r"조작|오류|잘못",
    re.IGNORECASE,
)

# The program-trading ANCHOR. `_PER_CLAIM_RE`/`_PBR_CLAIM_RE` get their
# anchor for free — they require the literal `PER` / `PBR`/`P/B` token next to
# the number — but "<n> billion KRW" names no metric at all, so without this
# the program-trading check compares ANY billion-KRW figure in the text
# against the program-trading ground truth. Two 2026-09-14 false positives
# this closes: (1) id=1405/id=1408 — `app.agents.unit_scale_guard`'s
# correction banner explains itself with the phrase "1억원 = 0.1 billion KRW",
# so every unit-scale correction was read as a claimed program net-buy of 0.1
# (and, since the banner also says "CORRECTION"/"10x", tagged "— false
# correction"); (2) id=1405 — the Research Manager quoting revenue-scale
# financial-statement figures ("4,876.7 billion KRW"). Required WITHIN
# `_FRAME_WINDOW` of the number, same `_near` convention as the
# attribution/correction frames. "net buy"/"net-buy" is deliberately NOT an
# accepted term: foreign and institutional flows are described as net buys
# too, and accepting it would re-introduce exactly the investor-flow vs
# program-trading confusion this anchor removes.
#
# The bare word "program" is NOT accepted either, and must not be added back.
# The discriminator is grammatical: in the trading sense "program" is a
# MODIFIER preceding another word ("program trading", "program net-buy"),
# while in every unwanted sense it is the HEAD noun preceded by its own
# modifier — "a 50 billion KRW buyback program", "share repurchase program",
# "capex program", "incentive program". Those routinely carry a KRW amount
# right beside them, so a bare alternative would anchor them to the
# program-TRADING ground truth: the same false-positive class this anchor
# exists to close, in a new guise. Hence "program" only counts when FOLLOWED
# by a trading-ish word.
#
# Rejected alternative (2026-09-14): stripping pipeline-inserted correction
# banners from the text before scanning. The anchor closes every observed
# case and does not change what the other guards see.
_PROGRAM_CONTEXT_RE = re.compile(r"program[- ](?:trading|net)|프로그램", re.IGNORECASE)

# How close an attribution/correction frame marker must be to a numeric
# claim to count as describing THAT claim — chars, same window convention
# as app.agents.regulatory_mechanism_guard._FRAME_WINDOW.
_FRAME_WINDOW = 120

# PER/PBR claims: "priced at 2.44x P/B", "PER of 12.3x", "PBR: 0.55x".
_PER_CLAIM_RE = re.compile(rf"\bPER\b[^.\n]{{0,20}}?(?:of\s*)?[~≈]?\s*{_NUM}\s*x\b", re.IGNORECASE)
_PBR_CLAIM_RE = re.compile(
    rf"(?:\bPBR\b[^.\n]{{0,20}}?(?:of\s*)?|priced\s+at\s*)[~≈]?\s*{_NUM}\s*x\s*(?:P/?B)?\b",
    re.IGNORECASE,
)

# Program-trading billion-KRW claims: "+15.65 billion KRW", "+1.565 billion KRW".
_PROGRAM_BILLION_CLAIM_RE = re.compile(
    rf"{_NUM}\s*billion\s+KRW",
    re.IGNORECASE,
)


def _mismatch_ratio(claimed: float, truth: float) -> float:
    """>= 1.0 always — how many times bigger the larger of (claimed, truth)
    is than the smaller. `float('inf')` if either is <= 0 (guarded by the
    caller before invocation with a non-positive ground-truth value)."""
    if truth == 0 or claimed == 0:
        return float("inf")
    ratio = abs(claimed) / abs(truth)
    return ratio if ratio >= 1 else 1 / ratio


# A "." that actually terminates a sentence — i.e. NOT a decimal point
# sitting directly between two digits. This guard's entire subject matter is
# decimal figures, so the plain `.` convention cut its own evidence spans
# mid-number (id=1405, 2026-09-14: a span began just after the "." in
# "4,876.7"). Sibling guards still use the plain convention; changing them is
# out of scope for this fix.
_SENTENCE_DOT_RE = re.compile(r"(?<!\d)\.|\.(?!\d)")


def _quoted_span(text: str, start: int, end: int) -> str:
    """The sentence containing `text[start:end]`, delimited by the nearest
    sentence-terminating `.` on each side (or the text boundary) — same
    sentence-bounded convention as
    `app.agents.regulatory_mechanism_guard._quoted_span`, except that a
    decimal point inside a number does not count as a terminator."""
    left = -1
    for m in _SENTENCE_DOT_RE.finditer(text, 0, start):
        left = m.start()
    left = left + 1 if left != -1 else 0
    right_m = _SENTENCE_DOT_RE.search(text, end)
    right = right_m.start() if right_m is not None else len(text)
    return text[left:right].strip()


def _near(text: str, start: int, end: int) -> str:
    return text[max(0, start - _FRAME_WINDOW) : end + _FRAME_WINDOW]


def scan_ground_truth_claims(text: str, *, ground_truth: dict[str, float], agent_name: str) -> list[str]:
    """`"AgentName: <description> — '<span>'"` per numeric claim in `text`
    that materially disagrees (>= `_MISMATCH_THRESHOLD`x, either direction)
    with `ground_truth`, restricted to claims that are either explicitly
    attributed to the supplied data/ground truth, or sit inside a
    correction/accusation frame ("fabrication", "tenfold", "10x", "조작",
    "오류", ...) — never fires on a bare unattributed number with no
    attribution/correction frame nearby, since that is ordinary prose, not a
    ground-truth claim.

    `[]` whenever `ground_truth` is empty (nothing supplied this cycle) —
    this guard has nothing to compare against and never fabricates a
    baseline. Detect-only — no control-flow change, no LLM call. Dedups
    identical spans within one call."""
    if not text or not ground_truth:
        return []

    findings: list[str] = []
    seen: set[str] = set()

    # 4th element: an ANCHOR regex that must appear near the numeric match for
    # it to count as a claim about this metric at all. PER/PBR need none —
    # their own pattern already requires the metric token — but the bare
    # "<n> billion KRW" shape does (see _PROGRAM_CONTEXT_RE).
    checks: list[tuple[re.Pattern[str], str, str, re.Pattern[str] | None]] = []
    if "per" in ground_truth:
        checks.append((_PER_CLAIM_RE, "per", "PER", None))
    if "pbr" in ground_truth:
        checks.append((_PBR_CLAIM_RE, "pbr", "PBR", None))
    if "program_net_buy_billion" in ground_truth:
        checks.append(
            (_PROGRAM_BILLION_CLAIM_RE, "program_net_buy_billion", "program-trading net-buy", _PROGRAM_CONTEXT_RE)
        )

    for rx, gt_key, label, anchor_rx in checks:
        truth = ground_truth[gt_key]
        if truth == 0:
            continue
        for m in rx.finditer(text):
            claimed = float(m.group(1).replace(",", ""))
            if claimed == 0:
                continue
            if _mismatch_ratio(claimed, truth) < _MISMATCH_THRESHOLD:
                continue  # matches ground truth (within tolerance) — never flagged

            near = _near(text, m.start(), m.end())
            if anchor_rx is not None and not anchor_rx.search(near):
                continue  # not a claim about this metric at all
            attributed = bool(_ATTRIBUTION_RE.search(near))
            correction = bool(_CORRECTION_FRAME_RE.search(near))
            if not (attributed or correction):
                continue  # bare unattributed number — ordinary prose, not a GT claim

            span = _quoted_span(text, m.start(), m.end())
            if span in seen:
                continue
            seen.add(span)

            tag = " — false correction" if correction else ""
            findings.append(
                f"{agent_name}: {label} claimed as {claimed:g} vs supplied ground truth "
                f"{truth:g}{tag} — {span!r}"
            )

    if findings:
        logger.warning(
            "ground_truth_claim_guard: %s cited a numeric value materially disagreeing with "
            "this cycle's supplied ground truth — %s",
            agent_name, findings,
        )
    return findings
