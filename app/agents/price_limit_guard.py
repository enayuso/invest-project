"""Detect-only recorder for a claimed same-session/single-day KRW price move
whose magnitude exceeds KRX's daily ±30% price-limit band — a claim that is
false BY CONSTRUCTION for a KRX-listed ticker, independent of any per-ticker
ground truth (no ATR/prev-close fetch needed to know a same-day move can't
exceed 30%).

Built for the 2026-09-08 live incident, decision id=1313 (082850.KS): the
Market Analyst invented a wholesale intraday narrative — "session high of
10,000 KRW to a low of 5,000 KRW, a staggering 50% drawdown... settling at
6,170 KRW" (+37.5% vs the correctly-stated previous close of 4,485 KRW), plus
an invented VI-trigger timestamp and an admittedly-estimated volume figure
("Volume: 1,000,000 shares (Estimated..., actual volume data not fully
detailed in snippet)"). No `VERIFIED SNAPSHOT` line appears anywhere in that
report — the tool call that exists specifically to give the model a
deterministic price to anchor to (`app.dataflows.session_state.
format_verified_snapshot`) was either skipped or silently overridden by the
model's own invention. That fabricated 50%/+37.5% narrative then propagated
uncorrected into the risk debate (7 separate mentions across Bull/Bear/
Aggressive/Conservative turns) and into the Portfolio Manager's own executive
summary ("a VI-triggered 50% intraday collapse, has fractured the order
book") — the same shape as `app.agents.halt_status_guard`'s id=1238 incident,
one guard scope over.

Calibrated against 800 real analyst reports (Market/News+Macro/Fundamentals,
the last 200 non-backtest KRX decisions as of 2026-09-08): candidate
same-session percentage claims above the 30.5% threshold below numbered 6,
and all 6 were genuine — the four id=1313 spans above, a 012210.KS News +
Macro Analyst claim ("the stock hit the KOSDAQ daily limit-up (+33% roughly,
though specific limits vary...)"), and a 043360.KS Market Analyst claim
("Yesterday's explosive 32% single-day gain"). Zero false positives among
187 sub-threshold same-session percentage matches inspected (session-range
positioning like "-6.6% below the session high", "4.7% above the session
low", genuinely-within-band moves like "+29.88% yesterday", and hedged
near-limit mentions like "+29.91%" all correctly stayed under the cutoff).

2026-09-09 addendum (domestic AI audit): two false-positive framings slipped
the `_EXCLUDE_RE` net — an abbreviated P&L line item ("-52% Op Profit",
id=1356; "operating profit" was excluded, "Op Profit" was not) and a
range-POSITION phrase using "of the distance between ... low and high"
(id=1360; "of the way" was excluded, "of the distance" was not). Both are
now covered; no threshold or anchor change.

2026-09-17 addendum (live incident, id=1542, 432980.KS): a retracement-
FRACTION framing slipped the `_EXCLUDE_RE` net — the risk_conservative
debater wrote "The stock failed to hold even 50% of its intraday gains,"
which states what fraction of the day's GAIN was given back, not a
same-session percentage CHANGE, and the same sentence's own actual move was
12.8% (well under the ±30% band). The existing "of the way/distance/range/
gap/spread" exclusion (2026-09-09) covers range-POSITION phrasing but not
this parallel MOVE-fraction shape. Now covered by an "of its/the ...
gains/losses/advance/rally" exclusion; no threshold or anchor change.

Scope is deliberately narrow to same-session/single-day framing — a
directional-percentage-near-a-price-verb net (an earlier calibration pass)
matched 778 candidates with ~94% false positives: multi-year comparisons
("+60.70% in 2024"), YoY/monthly figures ("+29.88% yesterday" driven by a
month-old headline), fundamentals ratios ("Borrowing Dependency rose to
30.71%"), and range-position framing ("72.7% of the way up from the low")
all look identical to a same-day price swing to a loose regex. The anchor
set below requires an EXPLICIT single-session marker (intraday/session high
or low/this session/single-day/daily limit-up-down/circuit breaker/VI/
closing-close/previous close) near the percentage, and a wide exclusion
window still rules out year/quarter/month tokens, fundamentals-ratio nouns,
and "of the way"/"relative location" range-position phrasing, since those
can share the session anchor's own vocabulary (e.g. "session high") while
describing something else entirely (a ratio LEVEL, or a position WITHIN
today's range, not a claimed CHANGE).

This is the same "detect-only, no control-flow change" pattern as
`app.agents.halt_status_guard.scan_halt_claims` and
`app.agents.technical_claim_guard.scan_technical_claims`: a narrative/
decision-quality problem only, NOT an execution-safety problem. Order
submission (`app.execution.kis_bridge`) and stop-loss monitoring
(`exit_monitor`/`state.stop_loss_loop`) both independently re-check the real
broker price at the time of action, completely independent of what the
Market Analyst wrote — so a fabricated 50% narrative here cannot itself
route an order at a fabricated price. The real fix (a stronger prompt/tool
binding on the Market Analyst so it can't invent OHLC figures the verified
snapshot doesn't support) is separate follow-up work; this module is the
detection net for whatever gets through anyway, same relationship
`app.agents.unit_scale_guard` and `app.agents.technical_claim_guard` have to
their own source-level fixes.

Deliberately does NOT check claimed absolute price LEVELS (e.g. "Session
High: 10,000 KRW") against a fetched real previous close — that would need
an extra ground-truth fetch (`app.dataflows.kis_provider.get_last_close`,
queried one day back) wired into the pipeline. The percentage-only check
above already caught every span of the id=1313 incident (an LLM narrating an
extreme same-day move essentially always states a percentage change
somewhere alongside the raw levels), at zero new external dependency — a
natural v2 extension if a future audit finds a report that states levels
without ever stating an accompanying percentage.

Only ever meaningful for a KRX-listed ticker (`.KS` suffix) — the ±30% cap
is a KRX-specific circuit mechanism; US-listed tickers have no equivalent
flat daily band (US limit-up/limit-down mechanisms are price-tier- and
rolling-window-based, not a flat percentage of the prior close). Callers
must gate on `ticker.endswith(".KS")` before invoking this — this module
takes no ticker argument and applies no such gate itself, matching
`app.agents.halt_status_guard`'s own "no ticker argument, caller decides
whether the ground truth is meaningful here" shape."""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# KRX's daily price-limit band is exactly ±30% of the reference (previous
# close) price. `_TOLERANCE_PCT` absorbs legitimate rounding at the boundary
# (a genuine limit-up/limit-down day, after tick-size rounding, can render as
# "29.9%" or "30.1%" depending on how a model does the arithmetic) without
# opening the threshold wide enough to miss a real violation — a claim has to
# clear 30.5% to fire, not merely approach 30%.
_KRX_DAILY_LIMIT_PCT = 30.0
_TOLERANCE_PCT = 0.5
_THRESHOLD_PCT = _KRX_DAILY_LIMIT_PCT + _TOLERANCE_PCT

# A signed/unsigned percentage figure, 1-3 digits with optional decimal —
# same numeric shape used across this project's other numeric guards
# (app.agents.unit_scale_guard's `_NUM`), but without that module's sign
# group: a price-move claim is meaningful in magnitude regardless of the
# textual sign token in front of it (both "+37.5%" and "a 37.5% drop" state
# the same 37.5-point move).
_PCT = r"([+-]?\s?\d{1,3}(?:\.\d+)?)\s?%"

# An EXPLICIT same-session/single-day marker — deliberately narrow (see
# module docstring's calibration note on why a loose directional-verb net
# produces ~94% false positives). "the day"/"ending the day" is included
# specifically for the id=1313 shape ("ending the day with a net gain of
# +37.5%"); "closing (price|at)"/"closed at"/"settled at"/"previous close"
# covers a claim framed around today's close rather than the word
# "intraday"/"session" itself.
_SESSION_ANCHOR = (
    r"(intraday|session high|session low|this session|today'?s? session|"
    r"single[- ]day|1[- ]day|one[- ]day|daily limit|limit[- ]up|limit[- ]down|"
    r"circuit breaker|\bVI\b|volatility interruption|"
    r"ending the day|closing (?:price|at)|closed at|settled at|previous close|"
    r"장중|당일|일일|서킷브레이커|변동성완화)"
)
_WINDOW = 70
_CANDIDATE_RE = re.compile(
    rf"(?:{_SESSION_ANCHOR}[^.\n]{{0,{_WINDOW}}}?{_PCT}|{_PCT}[^.\n]{{0,{_WINDOW}}}?{_SESSION_ANCHOR})",
    re.IGNORECASE,
)

# Checked over a WIDER window than the anchor match itself: a multi-period
# comparison, a fundamentals ratio, or a range-position phrase can share the
# session anchor's own vocabulary ("session high") while describing
# something this guard must not flag — a ratio LEVEL ("Borrowing Dependency
# rose to 30.71%"), a position WITHIN today's range rather than a change
# ("72.7% of the way up from the low"), or a move over a longer period than
# one session ("+60.70% in 2024", "a 37% gain in one month").
_EXCLUDE_RE = re.compile(
    r"(YoY|year-over-year|y/y|QoQ|q/q|quarter-over-quarter|annual(?:ized)?|"
    r"\b20\d{2}\b|FY\s?20|\bH[12]\b|\bfiscal\b|"
    r"month(?:ly)?|week(?:ly)?|quarter(?:ly)?|"
    r"the past (?:ten|[0-9]+) session|over the past|"
    r"since (?:early|late|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)|"
    # range-POSITION phrasing — a location within today's range, not a change
    # (2026-09-09, id=1360: "89% of the distance between its session low and
    # high"). "of the way" was already covered; "of the distance/range/gap"
    # and "between <the> low/high/open/close" are the same shape.
    r"of the (?:way|distance|range|gap|spread)|between (?:the |its )?(?:session )?(?:low|high|open|close)|"
    # retracement-FRACTION phrasing — a percentage OF a move (how much of the
    # day's gain/loss was given back or held), not a same-session percentage
    # CHANGE itself (2026-09-17, id=1542: "failed to hold even 50% of its
    # intraday gains" recorded as a claimed 50% same-session move when the
    # sentence's own actual move was 12.8%, well under the ±30% band).
    r"of (?:its|the) (?:intraday |today'?s? |session )?(?:gains?|losses?|advance|rally)|"
    r"percentile|relative location|position(?:ed|ing)?|"
    r"margin|ROE|ROA|debt|ratio|leverage|dependency|solvency|borrowing|equity(?! market)|"
    # P&L line items, including the abbreviated "Op Profit"/"OP margin" a model
    # writes inline (2026-09-09, id=1356: "-52% Op Profit").
    r"revenue|net income|operating (?:profit|income|margin|loss)|"
    r"\bop\.?\s*(?:profit|income|margin|loss)\b|earnings (?:decay|decline|growth|drop|beat|miss)|"
    r"EPS|P/E|valuation|dividend|payout|"
    r"short interest|ownership|stake|CAGR|multiple|estimate|forecast|guidance|"
    r"budget|spending|GDP|inflation|rate hike|expansion)",
    re.IGNORECASE,
)
_EXCLUDE_WINDOW = 90


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


def scan_price_limit_claims(text: str, *, agent_name: str) -> list[str]:
    """`"AgentName: claimed <N>% same-session move exceeds KRX's ±30% daily
    limit — '<span>'"` per claimed same-session/single-day percentage price
    move whose absolute magnitude exceeds `_THRESHOLD_PCT`, found in `text`.

    Only ever call this for a KRX-listed ticker (`.KS`) — see module
    docstring; this function takes no ticker argument and applies no such
    gate itself. `[]` for empty text, for a percentage claim that doesn't
    clear `_THRESHOLD_PCT`, and for every excluded framing (multi-period
    comparison, fundamentals ratio, range-position phrasing — see
    `_EXCLUDE_RE`). Detect-only — no control-flow change, no LLM call, no
    ground-truth fetch. Dedups identical spans within one call, same as
    `app.agents.halt_status_guard.scan_halt_claims`."""
    if not text:
        return []

    findings: list[str] = []
    seen: set[str] = set()
    for m in _CANDIDATE_RE.finditer(text):
        start, end = m.start(), m.end()
        wide = text[max(0, start - _EXCLUDE_WINDOW) : end + _EXCLUDE_WINDOW]
        if _EXCLUDE_RE.search(wide):
            continue
        pct_match = re.search(_PCT, m.group(0))
        if not pct_match:
            continue
        magnitude = abs(float(pct_match.group(1).replace(" ", "")))
        if magnitude <= _THRESHOLD_PCT:
            continue
        span = _quoted_span(text, start, end)
        if span in seen:
            continue
        seen.add(span)
        findings.append(
            f"{agent_name}: claimed {magnitude:g}% same-session move exceeds KRX's "
            f"±{_KRX_DAILY_LIMIT_PCT:g}% daily limit — {span!r}"
        )

    if findings:
        logger.warning(
            "price_limit_guard: %s claimed a same-session price move exceeding KRX's "
            "±%s%% daily limit (id=1313 shape) — %s",
            agent_name, _KRX_DAILY_LIMIT_PCT, findings,
        )
    return findings
