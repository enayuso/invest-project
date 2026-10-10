"""Detect-only recorder for a report/debate turn asserting that a ticker is
CURRENTLY halted/suspended when the broker's own execution-status check just
confirmed otherwise.

Built for the 2026-09-07 live incident, decision id=1238 (351320.KS): the
News + Macro Analyst, working from a several-days-old headline about a
"preliminary suspension of trading," asserted the stock IS currently halted
— and that false claim propagated uncorrected through the Research Manager,
the risk debate (46 mentions), and into the Portfolio Manager's final
decision text ("an active regulatory trading halt creating unmanageable
binary gap risk"), even though `app.orchestrator.pipeline`'s own
`execution_status_context` (see `app.execution.kis_bridge.execution_blockers`)
already told the PM, in the same cycle, that no halt/VI/circuit-breaker was
active for this ticker. Nobody caught the contradiction.

This is a narrative/decision-quality problem only — NOT an execution-safety
problem. Order submission (`kis_bridge._domestic_gates_block`) and stop-loss
monitoring (`exit_monitor`/`state.stop_loss_loop`) both independently
re-check real broker status at the time of action, completely independent of
what the PM wrote. So the real fix is Fix 1 (kis_bridge/pipeline — a
stronger, unambiguous "CONFIRMED by the broker just now" wording for the
clean case) and Fix 2/3 (prompt clauses on portfolio_manager/
news_macro_analyst). THIS module is the same "detect-only, no control-flow
change" pattern as `app.agents.risk_debators.check_neutral_reversal`:
positive evidence of whether the prompt fixes are actually working, not a
gate. 21-day live measurement backing this calibration: only 5/163 clean-
execution-status domestic decisions ever mentioned a halt/suspension at all
(~3%), and 4 of those 5 framed it correctly (past event / warning /
already-lifted) — only the id=1238 incident asserted a currently-active
halt. So a rare, narrow, string-only scan is the right scope here, not an
LLM call or a gate.

Only ever scans when `exec_status` carries Fix 1's CONFIRMED-clear marker —
no marker (overseas tickers, backtests, the could-not-verify branch, or a
genuine active blocker) means there is no confirmed ground truth to check a
claim against, so this never fires there, by construction.

2026-09-09 addendum (domestic AI audit): the 21-day calibration above
under-counted the false-positive surface — ~8/12 domestic decisions that
day tripped this guard on a line that CORRECTLY reported no halt
("no active VI halt is currently in effect", "the stock is not currently
subject to a VI halt"), asked whether one existed ("traders should verify
if the stock is currently under a VI halt"), or described a past halt's
aftermath ("post-halt repricing"). `_CURRENCY_RE`'s "active"/"currently"
markers match all of those. Added: `_NEGATED_RE` (negation in a tight
window around the term), `_META_RE` (verification framing anywhere in the
sentence), `if the market/stock is halted` conditional framing, `post-halt`
as a lifted-framing marker, and KRX designation-notice / "suspension risk"
warning framing. The one real present-tense assertion (id=1238) carries
none of these, so it still fires."""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# Fix 1's exact clean-case marker (app.orchestrator.pipeline._execution_status_context)
# — matching on this literal substring, not just "no blockers", is deliberate:
# an EMPTY execution_blockers() list is not the same as a confirmed-clear read
# (one or more of the three underlying gates may simply not have completed
# this cycle — see execution_blockers' own `checked_out` docstring), and this
# guard must never fire against a ground truth that isn't actually confirmed.
_CONFIRMED_MARKER = "CONFIRMED by the broker just now"

# A halt/suspension term, EN or KR. Deliberately includes the bare "halt" (not
# just "trading halt"/"halted") — a currency-of-claim marker like "until the
# halt lifts" already carries the term itself, and splitting hairs between
# "halt" and "halted" here would just create a blind spot for no benefit
# (this guard only ever fires within the narrow CONFIRMED-clear branch above).
_HALT_TERM_RE = re.compile(
    r"(trading halt|halted|halt|suspension of trading|trading suspension|suspended|suspension|"
    r"매매거래정지|거래정지|매매정지|변동성완화|\bVI\b)",
    re.IGNORECASE,
)

# A phrase asserting the halt/suspension is happening RIGHT NOW, not merely
# that one was announced/warned-of/lifted at some point. "post-halt" is
# deliberately NOT here — it means AFTER the halt, i.e. not currently halted
# (id=1354's "post-halt repricing was chaotic" is a past-aftermath
# description); it now lives in `_LIFTED_RE` instead.
_CURRENCY_RE = re.compile(
    r"(active|ongoing|currently|in effect|remains|is halted|is suspended|"
    r"until the halt lifts|when trading resumes|awaiting resumption)",
    re.IGNORECASE,
)

# How close a currency-of-claim marker must be to the halt term to count as
# describing THAT term (chars, not tokens — simpler and adequate at this
# guard's narrow scope; a wider window only makes the guard MORE permissive
# of a false positive slipping past, never less, since the exemptions below
# still get their own separate, wider window regardless).
_CURRENCY_WINDOW = 60

# The four legitimate framings real 21-day live data shows are common and
# correct (see module docstring) — checked over a WIDER window than the
# currency-marker check above, since a date/warning/lifted/future qualifier
# often sits a little further from the halt term itself than "active"/
# "currently" typically does right next to it.
_EXEMPT_WINDOW = 100

# Explicit date attribution near the term ("announced on Sept 4",
# "issued Sept 2", "Aug 31").
_DATE_RE = re.compile(
    r"announced on|dated|\b(?:issued|filed|posted|as of)\s+(?:on\s+)?"
    r"(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?\s*\d{1,2}\b|"
    r"\bon\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?\s*\d{1,2}\b",
    re.IGNORECASE,
)
# Warning / designation / future-risk framing — "X risk" (suspension/halt/
# delisting), a KRX designation notice, and the conditional openers.
_WARNING_RE = re.compile(
    r"warning|possible|potential|risk of|if the (krx|korea exchange)|preliminary|지정예고|could|"
    r"(?:suspension|halt|delisting|trading[- ]stop)\s+risk|risk of (?:a\s+)?(?:suspension|halt)|"
    r"investment\s+caution|투자\s*주의|투자\s*경고|designat\w+\s+(?:the\s+)?(?:stock|it)|"
    r"introduc\w+\s+.{0,20}risk",
    re.IGNORECASE,
)
# Already-lifted / past-aftermath framing.
_LIFTED_RE = re.compile(r"lifted|released|resumed|post-halt|aftermath|unwound|once (?:the\s+)?halt", re.IGNORECASE)
# Future/hypothetical/scenario/conditional framing.
_FUTURE_RE = re.compile(
    r"would|were to|hypothetical|scenario|in the event|if trading|"
    r"can (?:easily |still |quickly )?gap|could gap|may (?:be |get )?halt|"
    r"\bif\s+(?:the\s+)?(?:market|stock|price|it|trading|liquidity)\b",
    re.IGNORECASE,
)

# Negated framing — the halt/suspension is asserted NOT to be in effect
# ("no active VI halt", "the stock is not currently subject to a VI halt").
# 2026-09-09 audit: ~8/12 domestic decisions tripped this guard on a
# correctly-negated VI/halt line, and the ONE real present-tense assertion
# (id=1238, "an active regulatory trading halt creating ... gap risk") has no
# negation near the term. Checked in a TIGHT window around the term (left
# `_NEG_LEFT`, right `_NEG_RIGHT`), NOT the whole sentence — "There is
# currently an active trading halt ... and orders will not fill" has an
# unrelated "not" late in the same sentence and must still fire. The far-left
# "did not include verification of whether ..." case (id=1357) is caught by
# `_META_RE` instead.
# 2026-09-10 audit, id=1369 (028050.KS): "is **not** currently subject to a
# Volatility Interruption (VI) mechanism" — the full phrase "Volatility
# Interruption" between "not" and the "(VI)" abbreviation this guard's own
# term regex matches on pushed the real negation-to-term distance to 56
# chars, past the previous 45-char window. Widened to 70 for margin over
# realistic phrasing-length variance, not just to clear this one case by a
# hair.
_NEG_LEFT = 70
_NEG_RIGHT = 16
_NEGATED_RE = re.compile(
    r"\b(?:no|not|n't|without|never|nor|neither|free (?:of|from)|clear of|absent|"
    r"isn'?t|aren'?t|wasn'?t|weren'?t|hasn'?t|haven'?t)\b",
    re.IGNORECASE,
)
# Meta / verification framing — the sentence is ABOUT checking halt status
# ("a check was performed regarding ... VI halts", "traders should verify
# if ...", "our checks"), not asserting one is active.
_META_RE = re.compile(
    r"\b(?:verif\w+|check(?:ed|s|ing)?\b|our checks|"
    r"a check (?:was|is) (?:performed|done|made)|should (?:verify|check|confirm)|"
    r"traders? should|investors? should|recommend (?:verifying|checking)|"
    r"whether (?:the\s+)?(?:stock|ticker|it) (?:is|was))",
    re.IGNORECASE,
)


def _quoted_span(text: str, start: int, end: int) -> str:
    """The sentence containing `text[start:end]`, delimited by the nearest
    `.` on each side (or the text boundary) — used only for the human-
    readable quoted span in a finding string, never for the proximity/
    exemption checks above (those use fixed-radius char windows). Sentence-
    bounded rather than a fixed-radius window so that the SAME sentence
    repeated twice in one text (see `scan_halt_claims`'s dedup contract)
    always produces the exact same span string regardless of where in the
    text each repetition falls — a fixed-radius window clipped differently
    at each occurrence's position would defeat the dedup."""
    left = text.rfind(".", 0, start)
    left = left + 1 if left != -1 else 0
    right = text.find(".", end)
    right = right if right != -1 else len(text)
    return text[left:right].strip()


def scan_halt_claims(text: str, *, exec_status: str, agent_name: str) -> list[str]:
    """`"AgentName: '<span>' vs CONFIRMED-clear broker status"` per present-
    tense halt/suspension assertion found in `text`, checked ONLY against a
    genuinely CONFIRMED-clear `exec_status` (Fix 1's marker) — `[]`
    immediately for an overseas ticker, a backtest, the could-not-verify
    branch, or an actual active blocker, since none of those give this guard
    a ground truth to contradict. `[]` is also the common live-clean-case
    result even when scanning DID run: this fires only on the rare present-
    tense claim, never on a correctly-dated/warned/lifted/future-framed one
    (see the four exemption regexes above). Detect-only — no control-flow
    change, no LLM call; mirrors
    `app.agents.risk_debators.check_neutral_reversal`'s shape and its
    "positive evidence a prompt fix is working, not a gate" role. Dedups
    identical spans within one call."""
    if not exec_status or _CONFIRMED_MARKER not in exec_status:
        return []
    if not text:
        return []

    findings: list[str] = []
    seen: set[str] = set()
    for m in _HALT_TERM_RE.finditer(text):
        start, end = m.start(), m.end()
        near = text[max(0, start - _CURRENCY_WINDOW) : end + _CURRENCY_WINDOW]
        if not _CURRENCY_RE.search(near):
            continue
        wide = text[max(0, start - _EXEMPT_WINDOW) : end + _EXEMPT_WINDOW]
        if _DATE_RE.search(wide) or _WARNING_RE.search(wide) or _LIFTED_RE.search(wide) or _FUTURE_RE.search(wide):
            continue
        span = _quoted_span(text, start, end)
        # 2026-09-09 audit: negation in a tight window around the term
        # ("no active VI halt"), or verification framing anywhere in the
        # sentence ("our checks did not include verification of whether ...").
        neg_ctx = text[max(0, start - _NEG_LEFT) : end + _NEG_RIGHT]
        if _NEGATED_RE.search(neg_ctx) or _META_RE.search(span):
            continue
        if span in seen:
            continue
        seen.add(span)
        findings.append(f"{agent_name}: {span!r} vs CONFIRMED-clear broker status")

    if findings:
        logger.warning(
            "halt_status_guard: %s asserted a currently-active halt/suspension against a "
            "CONFIRMED-clear broker status (id=1238 shape) — %s",
            agent_name, findings,
        )
    return findings
