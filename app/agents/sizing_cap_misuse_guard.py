"""Detect-only recorder for a risk-debate turn or the Portfolio Manager
citing the ADDED-SHARE position-sizing cap (or its volatility/ATR scaling)
as the REASON to exit or cut a position that is already held.

Built for the 2026-09-09 live incident, decision id=1358 (047040.KS/
대우건설): the risk debate's Neutral turn wrote "Since the position size is
effectively negligible regardless of the rating (due to the risk cap) ...
the optimal choice is to remove the exposure entirely", the Conservative
turn ran a section headed "The Mechanical Argument for Zero Exposure", and
the PM's own investment_thesis concluded "the 8.0% ATR compresses any
theoretical position to a trivial share count ... Exiting eliminates
exposure to this asymmetric risk profile." A real Sell of 5 held shares
executed minutes later at a -0.98% loss, with no stop-loss or drawdown
breaker forcing it.

The prompt is unambiguous in FIVE places (app.orchestrator.pipeline
`_sizing_cap_context`/`_portfolio_risk_context`, app.agents.portfolio_manager
`_SYSTEM`) that this cap "only ever reduces shares being ADDED, never forces
a sale of shares already held" — so this is that prompt clause LEAKING, not
being absent, the same relationship halt_status_guard / price_limit_guard /
session_close_guard have to their own prompt fixes.

Detect-only, deliberately. The action a gate could take here is refusing an
exit — a risk-RETENTION decision made by regex — and the false-positive
surface (a debater legitimately arguing the cap makes a fresh ADD
pointless, which is correct and common) is real and uncalibrated. This
module is step 3 of the P0-2/3 plan: the sentence-level scanner whose
counts a later retrospective calibration reads before any gate
(`unjustified_exit_gate`, default OFF) is switched on. Mirrors
`app.agents.halt_status_guard.scan_halt_claims`'s shape: a ground-truth
gate (`held`) in place of the marker check, a claim regex, a correct-
restatement exemption, and a sentence-bounded quoted span for dedup.

`held=False` returns `[]` immediately, by construction: with no position on
the book there is nothing the cap could be misused to liquidate — a
debater saying the cap makes a NEW entry too small to bother with is stating
the mechanic correctly, not misusing it."""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# The added-share sizing cap ITSELF (or its volatility/ATR scaling) — not
# every mention of position size. Naming the cap/scaling is the necessary
# first half of the misuse signature; the exit conclusion drawn FROM it is
# the second half (`_EXIT_FROM_CAP_RE` below).
_CAP_TERM_RE = re.compile(
    r"risk[- ]cap|added\s+shares|shares\s+(?:being\s+)?added|"
    r"sizing\s+(?:framework|arithmetic|mechanics|cap|constraint|engine|block|calculus|math)|"
    r"position[- ]siz(?:e|ing)\s+(?:cap|limit|constraint|framework|mechanic)|"
    r"share[- ]count\s+(?:cap|limit)|"
    r"ATR\s+(?:scal|compress|expan|adjust)|volatility[- ](?:scal|adjust|based\s+scal)|"
    r"vol(?:atility)?[- ]adjusted|volatility[- ]scaling|"
    r"target\s+allocation|allocation\s+cap|"
    r"리스크\s*캡|편입\s*한도|목표\s*비중|사이징\s*(?:한도|캡)",
    re.IGNORECASE,
)

# An exit / reduce conclusion. Both directions of the cause->effect wording
# are covered by scanning a two-sentence window (see `scan_sizing_cap_misuse`)
# rather than by duplicating the alternation here.
_EXIT_TERM_RE = re.compile(
    r"\b(sell|selling|exit(?:ing|s)?|liquidat\w+|"
    r"clos(?:e|ing)\s+(?:out\s+)?the\s+position|trim(?:ming)?\s+to\s+zero|"
    r"remov\w+\s+(?:the\s+)?(?:exposure|position)|eliminat\w+\s+(?:the\s+)?(?:exposure|position)|"
    r"zero\s+exposure|clean\s+exit|full\s+exit|reduc\w+\s+(?:the\s+)?(?:position|exposure|holding)|"
    r"scal\w+\s+(?:out|down)\s+the\s+position|cut\s+(?:the\s+)?(?:position|exposure)|"
    r"청산|전량\s*매도|비중\s*축소|보유분\s*(?:을\s*)?(?:매도|줄))\b",
    re.IGNORECASE,
)

# Reasoning connectives that tie the cap term to the exit conclusion — a
# CAP term and an EXIT term co-occurring in the window is not enough on its
# own (a turn can mention both while arguing the OPPOSITE); one of these
# must also be present in the window.
_CONNECTIVE_RE = re.compile(
    r"justif\w*|warrant\w*|support\w*|impl(?:y|ies|ied)|means?\b|requires?|mandates?|forces?|"
    r"dictates?|argues?\s+for|the\s+case\s+for|therefore|hence|thus|so\s+(?:we|the)|"
    r"because|since|given\s+(?:that|the)|driven\s+by|on\s+the\s+basis\s+of|per\s+the|under\s+the|"
    r"validate\w*|the\s+optimal\s+choice|makes?\s+any\s+position|"
    r"근거|때문|따라|정당화",
    re.IGNORECASE,
)

# The prompt's own correct restatement of the mechanic — if the window
# already says this, the turn is echoing the rule, not breaking it. Covers
# the five prompt sites' wording plus the PM system-prompt clauses.
_CORRECT_RESTATEMENT_RE = re.compile(
    r"never\s+forces?\s+a\s+sale|does\s+not\s+force\s+a\s+sale|not\s+force\s+a\s+sale|"
    r"only\s+ever\s+reduces?\s+shares\s+being\s+add|only\s+reduces?\s+(?:new|added)\s+shares|"
    r"only\s+ever\s+reduces?\s+(?:the\s+)?add|"
    r"not\s+a\s+reason\s+to\s+(?:sell|trade)|is\s+never\s+a\s+reason|are\s+never\s+a\s+reason|"
    r"never\s+forces?\s+shares|mechanics\s+are\s+never\s+a\s+reason\s+to\s+trade|"
    r"보유분\s*(?:을\s*)?매도(?:를)?\s*강제하지",
    re.IGNORECASE,
)

# The window is explicitly about sizing a NEW entry / the ADD, not the held
# position — the cap genuinely does limit this, so naming it here is correct
# use, not misuse. Checked only when no unambiguous held-position exit verb
# ("liquidate", "clean exit", "remove the exposure", "zero exposure") is
# present in the window.
_ABOUT_THE_ADD_RE = re.compile(
    r"new\s+(?:purchase|shares|buy|entry|position)|the\s+add\b|added\s+position|"
    r"fresh\s+entry|opening\s+a\s+position|initiat\w+\s+(?:a\s+)?position|"
    r"building\s+(?:a\s+)?position|scal\w+\s+in\b|신규\s*(?:매수|진입)",
    re.IGNORECASE,
)
# Exit verbs that can only mean the EXISTING position — these override
# `_ABOUT_THE_ADD_RE` (a window can discuss the add AND still conclude with a
# held-position liquidation).
_UNAMBIGUOUS_HELD_EXIT_RE = re.compile(
    r"liquidat\w+|clean\s+exit|full\s+exit|zero\s+exposure|"
    r"remov\w+\s+(?:the\s+)?(?:exposure|position)|eliminat\w+\s+(?:the\s+)?(?:exposure|position)|"
    r"exit(?:ing|s)?\s+(?:the\s+position|now|eliminat)|clos(?:e|ing)\s+(?:out\s+)?the\s+position|"
    r"청산|전량\s*매도",
    re.IGNORECASE,
)

_SENT_BOUNDARY_RE = re.compile(r"[.\n](?=\s|$)")


def _sentence_bounds(text: str) -> list[tuple[int, int]]:
    """`(start, end)` for every sentence in `text`, split on `.`/newline that
    is followed by whitespace or end-of-string — a `.` inside "8.0%" or
    "U.S." is NOT a boundary, deliberately: the fixed-`.` split that
    `halt_status_guard._quoted_span` uses shreds the decimal-heavy sizing
    prose this guard reads, which is exactly why the older `mechanics_terms`
    chunk scan could not be reused here."""
    bounds: list[tuple[int, int]] = []
    prev = 0
    for m in _SENT_BOUNDARY_RE.finditer(text):
        bounds.append((prev, m.start()))
        prev = m.end()
    if prev < len(text):
        bounds.append((prev, len(text)))
    return bounds


def scan_sizing_cap_misuse(text: str, *, agent_name: str, held: bool) -> list[str]:
    """`"AgentName: sizing cap cited as exit rationale — '<span>'"` per
    sentence in `text` that names the added-share sizing cap / volatility
    scaling AND, within that sentence or the next, draws an exit-or-reduce
    conclusion from it via a reasoning connective — unless that window also
    contains the prompt's own correct restatement of the mechanic.

    `[]` immediately when `held` is False (nothing on the book to misuse the
    cap against) or `text` is empty. Detect-only — no control-flow change,
    no LLM call. Dedups identical spans within one call."""
    if not held or not text:
        return []

    bounds = _sentence_bounds(text)
    findings: list[str] = []
    seen: set[str] = set()

    for i, (s, e) in enumerate(bounds):
        sentence = text[s:e]
        if not _CAP_TERM_RE.search(sentence):
            continue
        # Two-sentence window: cause ("the cap compresses this to nothing")
        # and conclusion ("so we exit") routinely fall in adjacent sentences.
        w_end = bounds[i + 1][1] if i + 1 < len(bounds) else e
        window = text[s:w_end]
        if not _EXIT_TERM_RE.search(window):
            continue
        if not _CONNECTIVE_RE.search(window):
            continue
        if _CORRECT_RESTATEMENT_RE.search(window):
            continue
        if _ABOUT_THE_ADD_RE.search(window) and not _UNAMBIGUOUS_HELD_EXIT_RE.search(window):
            continue
        span = " ".join(sentence.split()).strip()
        if span in seen:
            continue
        seen.add(span)
        findings.append(f"{agent_name}: sizing cap cited as exit rationale — {span!r}")

    if findings:
        logger.warning(
            "sizing_cap_misuse_guard: %s cited the added-share sizing cap / volatility scaling as a "
            "reason to exit or cut a HELD position (id=1358 shape) — %s",
            agent_name, findings,
        )
    return findings
