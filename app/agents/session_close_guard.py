"""Detect-only recorder for a report/debate turn narrating TODAY's still-open
session as if the daily bar had already closed.

Built for the 2026-09-09 live incident, decision id=1358 (047040.KS/대우건설):
the risk debate's round-2 Neutral and Conservative turns wrote "the stock
surged 6% to 21,300 KRW but closed near the session low (20,100 KRW),
rejecting the breakout" and "this is a textbook distribution pattern" —
at 10:16 KST, 46 minutes into the session
(`app.dataflows.session_state.session_in_progress` was True). A completed-
candle read is not merely imprecise here: it was the stated #1 reason the
Neutral verdict flipped Underweight -> Sell between rounds
(`app.agents.risk_debators.check_neutral_reversal` recorded
`reversed: True, new_fact_cited: False` for this exact turn), and that Sell
executed against a real position minutes later.

This is the SAME class of incident `app.dataflows.session_state`'s own
module docstring documents (2026-08-26, id=1013, MRVL) and that
`intraday_narration_note` was written to prevent by injecting an explicit
"do not write closed/finished/settled/concluded" instruction into every
downstream prose node. id=1358 is that prompt fix LEAKING, not being
absent — same relationship price_limit_guard/verified_price_guard/
halt_status_guard have to their own prompt fixes.

Detect-only, deliberately — same reasoning as halt_status_guard: the only
action available to a gate here is refusing an exit (a risk-RETENTION
decision made by regex) or aborting the cycle, neither proportionate to a
single mis-narrated tense, and the false-positive surface is real before
any live calibration exists: `intraday_narration_note` itself contains the
literal words "closed", "finished", "settled", and "concluded trading" (its
own prohibition list), so a node quoting its instruction back would trip a
naive scanner — this module strips the note out of the text before
scanning specifically to avoid that. Mirrors
`app.agents.halt_status_guard.scan_halt_claims`'s shape exactly: a marker-
gated ground truth, a claim regex, a wider-window exemption pass, and a
sentence-bounded quoted span for dedup.

2026-09-15 fix, decision id=1435 (067290.KS/JW신약), live false positive:
"The stock closed at 3,620 on Sept 10 and 3,515 on Sept 11" was flagged even
though Sept 10/11 are genuinely PAST trading days — a correct statement, not
a claim that today's still-open session has closed. The prior-session
exemption already recognized an ISO date (`2026-09-11`) as evidence of a
past-session reference but not a month-name date ("Sept 10"), and the
pipeline's default output language is English, where a month-name date is
the natural form. Fix: the date-based exemption is now TODAY-AWARE rather
than "any recognized date exempts" — it parses today's date out of
`session_note` (which already carries it: "...still open as of
{isoformat}...", injected by `intraday_narration_note`) and, for each
recognized date near a claim (ISO, English month-name, or Korean "N월 N일"),
exempts only when that date is NOT today's. A date equal to today's no
longer exempts — this also TIGHTENS the pre-existing (buggy) behavior where
today's own ISO date wrongly exempted a same-day claim.
(요약: "9월 10일/11일에 마감" 같은 월-이름 과거 날짜도 ISO 날짜처럼 예외 처리하되,
오늘 날짜와 일치하면 더 이상 예외 처리하지 않도록 "오늘 인식" 로직을 추가함.)

Fallback (deliberate, never omit): if today's date cannot be parsed out of
`session_note` for any reason, this falls back to the OLD behavior — any
recognized date exempts — rather than raising or flagging more. A parse
failure must never make this guard noisier than it already was; see
`_parse_today` and its call site.

Known, accepted residual (see also this file's other "detect-only, false-
positive-averse" framing, and `app.agents.absence_reframing_guard`'s
docstring: "a false positive here is worse than a missed case"): a sentence
that mixes TODAY's date with a genuinely past date inside the same
exemption window (e.g. "closed at 3,990 on Sept 15 after closing at 3,515 on
Sept 11", today = Sept 15) will still be exempted — the past date's presence
in the window is enough — and the true positive about today's own claim will
be missed. This is accepted as rare relative to the plain past-dated close
this fix targets, and consistent with this codebase's general bias away
from false positives over false negatives in detect-only guards."""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# `app.dataflows.session_state.intraday_narration_note`'s own leading token —
# matching on this literal marker, not a boolean, keeps this guard's contract
# identical to halt_status_guard's: no injected ground truth means no claim to
# check against, by construction. A backtest or an already-closed session
# yields "" and this returns [] immediately.
_IN_PROGRESS_MARKER = "SESSION IN PROGRESS"

# Family A — a completion verb applied to TODAY's price action. Deliberately
# broad (English + Korean) since the incident text used plain "closed", not a
# more specific "the daily close was".
_CLOSED_ASSERTION_RE = re.compile(
    r"clos(?:ed|ing)\s+(?:at|near|above|below|the|price|level|range|bell|up|down|flat|higher|lower)|"
    r"(?:today'?s?|the)\s+clos(?:e|ing)\b|"
    r"settled\s+(?:at|near)|finished\s+(?:the\s+(?:day|session)|at|near)|"
    r"concluded\s+trading|end(?:ed|ing)\s+the\s+(?:day|session)\s+(?:at|near)|"
    r"daily\s+(?:close|closing)\s+(?:price|range)|on\s+the\s+close|"
    r"종가|마감가|마감\s*했|장\s*마감\s*(?:후|기준)|당일\s*종가",
    re.IGNORECASE,
)

# Family B — vocabulary that PRESUPPOSES a finished daily bar (a completed
# candle). This is the family that actually mattered in id=1358: "closed near
# the session low ... textbook distribution pattern" is a completed-candle
# read, and it was the stated #1 reason for the Sell.
_COMPLETED_CANDLE_RE = re.compile(
    r"distribution\s+pattern|accumulation\s+pattern|textbook\s+\w+\s+pattern|"
    r"(?:daily|today'?s?)\s+candle(?:stick)?|candle(?:stick)?\s+(?:pattern|body|formation)|"
    r"bearish\s+engulfing|shooting\s+star|\bdoji\b|hanging\s+man|marubozu|"
    r"long\s+upper\s+(?:wick|shadow)|upper\s+(?:wick|shadow)\s+(?:of|shows|signals)|"
    r"clos(?:ed|ing)\s+(?:near|at)\s+the\s+(?:session|day'?s?|intraday)\s+(?:low|high)|"
    # 2026-09-11 audit, id=1387 (217820.KS): the PM's own paraphrase — "the
    # intraday price action shows classic distribution as shares dumped from
    # 3,300 KRW to 2,805 KRW" — never uses the literal word "pattern" after
    # "distribution"/"accumulation", so the alternatives above miss it. This
    # is a bare vocabulary widening only: a completion-style verb immediately
    # followed by an intensifier and then distribution/accumulation, with no
    # "pattern" required. The existing exemption passes (_PRIOR_SESSION_RE,
    # _FUTURE_CLOSE_RE, _QUALIFIED_RE, _HEDGE_RE, _MA_RELATIVE_CLOSE_RE) still
    # run over any match from this alternative exactly as for the others.
    r"(?:shows?|exhibit\w*|demonstrat\w*|display\w*|reflects?)\s+(?:classic|textbook|clear)\s+"
    r"(?:distribution|accumulation)\b|"
    r"음봉|양봉|윗꼬리|아랫꼬리|일봉",
    re.IGNORECASE,
)

# Checked over a WIDER window than the match (same two-tier shape as
# halt_status_guard's _CURRENCY_WINDOW/_EXEMPT_WINDOW) — a legitimate
# qualifier often sits a little further from the term than right next to it.
_EXEMPT_WINDOW = 110

# A prior, genuinely completed session — the overwhelmingly common legitimate
# use of "close" in a KRX report ("the previous close of 20,300"). Date-FREE
# markers only now — these stay unconditional (no today-comparison possible
# or needed for them). The date-bearing forms (ISO/month-name/Korean) moved
# to _wide_has_non_today_date below, 2026-09-15 (id=1435 fix), since those
# need to be compared against today's date rather than always exempting.
_PRIOR_SESSION_RE = re.compile(
    r"previous|prior|yesterday|last\s+(?:close|session|night|friday|week|month)|"
    r"preceding|the\s+day\s+before|"
    r"전일|전\s*거래일|어제|지난\s*주|직전\s*거래일",
    re.IGNORECASE,
)

# ISO date, e.g. "2026-09-11" — already-supported form, now compared against
# today (see _wide_has_non_today_date) instead of unconditionally exempting.
_ISO_DATE_RE = re.compile(r"\b(20\d{2})-(\d{2})-(\d{2})\b")

# English month-name date, full or abbreviated, optional trailing "." on the
# abbreviation and optional trailing year: "September 11", "Sept 11",
# "Sep 11", "Sep. 11", "September 11, 2026". 2026-09-15 (id=1435,
# 067290.KS): the pipeline's default English output naturally writes dates
# this way, not as ISO, so the ISO-only exemption missed it.
# 영문 월-이름 날짜(전체/약어, 연도 선택) — 09-15 수정으로 추가.
_MONTH_NAME_DATE_RE = re.compile(
    r"\b(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|"
    r"aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\.?\s+"
    r"(\d{1,2})(?:st|nd|rd|th)?(?:,?\s+(\d{4}))?\b",
    re.IGNORECASE,
)
_MONTH_NUM = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9, "oct": 10,
    "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12,
}

# Korean month/day date: "9월 11일" — included for symmetry with the Korean
# markers already in _PRIOR_SESSION_RE (전일/어제/...), which are date-free.
# 한국어 날짜("9월 11일") — 대칭성을 위해 추가.
_KOREAN_DATE_RE = re.compile(r"(\d{1,2})\s*월\s*(\d{1,2})\s*일")

# Today's date out of session_note's own "...still open as of YYYY-MM-DD..."
# (intraday_narration_note's literal wording) — no new parameter needed since
# the note already carries it.
_TODAY_FROM_NOTE_RE = re.compile(r"as\s+of\s+(20\d{2})-(\d{2})-(\d{2})")


def _parse_today(session_note: str) -> tuple[int, int, int] | None:
    """Today's (year, month, day) parsed out of `session_note`, or None if it
    can't be found — callers MUST treat None as "fall back to exempting on
    any recognized date" (the pre-2026-09-15 behavior), never as license to
    flag more than before. See module docstring's "Fallback" paragraph."""
    m = _TODAY_FROM_NOTE_RE.search(session_note)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def _wide_has_non_today_date(wide: str, today: tuple[int, int, int] | None) -> bool:
    """True if `wide` contains a recognized date (ISO, English month-name, or
    Korean) that is NOT `today` — i.e. a genuine prior-session reference that
    should exempt the claim. When `today` is None (unparseable session_note),
    ANY recognized date exempts, matching the old, looser behavior — see
    `_parse_today`'s docstring and the module docstring's "Fallback"
    paragraph. When a month-name or Korean date carries no year, only month
    and day are compared against today's."""
    for m in _ISO_DATE_RE.finditer(wide):
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if today is None or (y, mo, d) != today:
            return True
    for m in _MONTH_NAME_DATE_RE.finditer(wide):
        month_num = _MONTH_NUM.get(m.group(1).lower())
        if month_num is None:
            continue
        day_num = int(m.group(2))
        year_txt = m.group(3)
        if today is None:
            return True
        if year_txt is not None:
            if (int(year_txt), month_num, day_num) != today:
                return True
        elif (month_num, day_num) != (today[1], today[2]):
            return True
    for m in _KOREAN_DATE_RE.finditer(wide):
        month_num, day_num = int(m.group(1)), int(m.group(2))
        if today is None or (month_num, day_num) != (today[1], today[2]):
            return True
    return False
# Future / conditional / auction framing — "into the close", "if it closes above".
# 2026-09-10 audit: id=1365 "continue ... through the close" (forward
# continuation) and id=1371 "Monitor the close: if ..." (the 'if' precedes
# rather than follows 'close', so the existing if-clos pattern doesn't reach
# it) are both forward-looking, not closure assertions.
_FUTURE_CLOSE_RE = re.compile(
    r"\bwill\b|\bwould\b|\bmay\b|\bcould\b|\bexpect|\banticipat|\blikely\b|"
    r"if\s+(?:it|the\s+stock|shares|price)\s+clos|by\s+the\s+close|into\s+the\s+close|"
    r"before\s+the\s+close|(?:at|toward)\s+today'?s?\s+close\b|closing\s+auction|"
    r"through\s+the\s+close|monitor\s+the\s+close|"
    r"장\s*마감\s*(?:전|까지)|예상|전망",
    re.IGNORECASE,
)
# Negated / instructional framing (LEFT context only, up to the match) — the
# 2026-09-09 audit's own finding is that halt_status_guard has exactly this
# blind spot (3/3 false positives on "No active VI halt"); designed in here
# from the start rather than bolted on after the same mistake repeats.
_NEGATED_RE = re.compile(
    r"\b(?:not|no|never|isn'?t|wasn'?t|hasn'?t|has\s+not|cannot|can'?t|do\s+not|"
    r"don'?t|avoid|rather\s+than|instead\s+of|must\s+not|should\s+not)\b"
    r"[^.\n]{0,45}$",
    re.IGNORECASE,
)
# "close to the 200-day MA" — adjective/adverb, not a price event. The
# highest-volume false-positive shape in a real analyst report.
_CLOSE_ADVERB_RE = re.compile(r"clos(?:e|er|est)\s+to\b|close\s+(?:proximity|above|below)\b", re.IGNORECASE)
# 2026-09-10 audit, id=1376 (080220.KS): "Closing above the SMA(20) confirms
# short-term momentum is positive" — describes price LEVEL relative to a
# moving-average benchmark, not an end-of-session event; the verb-form
# counterpart to _CLOSE_ADVERB_RE's already-exempt adjective form ("close to
# the 200-day MA").
_MA_RELATIVE_CLOSE_RE = re.compile(
    r"clos(?:e|ed|ing)\s+(?:above|below)\s+(?:its\s+|the\s+)?"
    r"(?:\d+[\s-]?(?:day|week|month)\s+)?(?:sma|ema|\bma\b|moving average|vwap)",
    re.IGNORECASE,
)
# The correct phrasing the prompt itself asks for, sitting right next to the term.
# 2026-09-10 audit: id=1366 "closed the morning session" — KRX has no formal
# AM/PM session split; this is colloquial for "so far this morning", the same
# shape as the already-exempt "so far today".
_QUALIFIED_RE = re.compile(
    r"so\s+far\s+(?:today|this\s+session)|as\s+of\s+now|intraday\s+so\s+far|"
    r"morning\s+session|현재\s*까지|장중|오전\s*장",
    re.IGNORECASE,
)

# 2026-09-10 audit: id=1371 "maintain its technical uptrend despite
# distribution patterns" — the Bear debater is REFERENCING a risk the Bull is
# (in the Bear's telling) dismissing, not directly asserting the pattern
# occurred; "despite X"/"risk of X"/"concern about X" frames X as a hedged
# caveat, the same shape halt_status_guard's own _WARNING_RE already exempts
# for a different vocabulary set. Deliberately does NOT catch the guard's own
# pinned true positive (test_flags_id1358_distribution_pattern_span's "This is
# a textbook distribution pattern—...") — that text has no hedge word
# anywhere near the match, by design; verify this remains true after edits.
_HEDGE_RE = re.compile(
    r"despite|risk\s+of|concern(?:s|ed)?\s+(?:about|over)|worr(?:y|ied)\s+about|"
    r"warns?\s+of|warning\s+of",
    re.IGNORECASE,
)


def _quoted_span(text: str, start: int, end: int) -> str:
    """The sentence containing `text[start:end]`, delimited by the nearest
    `.` on each side (or the text boundary) — same convention as
    `app.agents.halt_status_guard._quoted_span`, for the same dedup reason."""
    left = text.rfind(".", 0, start)
    left = left + 1 if left != -1 else 0
    right = text.find(".", end)
    right = right if right != -1 else len(text)
    return text[left:right].strip()


def scan_session_close_claims(text: str, *, session_note: str, agent_name: str) -> list[str]:
    """`"AgentName: [<family>] '<span>' vs SESSION-IN-PROGRESS ground truth"`
    per closed-session/completed-candle assertion found in `text`, checked
    ONLY when `session_note` is genuinely the injected in-progress marker
    (`app.dataflows.session_state.intraday_narration_note`'s return value) —
    `[]` immediately for a closed session, a backtest, or an overseas ticker
    outside market hours, since none of those give this guard a ground truth
    to contradict. The note itself is stripped from `text` first: it
    literally contains "closed"/"finished"/"settled"/"concluded trading" as
    its own prohibition list, and a node quoting its instruction back must
    never count as a violation of it. Detect-only — no control-flow change,
    no LLM call; mirrors `app.agents.halt_status_guard.scan_halt_claims`'s
    shape and its "positive evidence a prompt fix is working, not a gate"
    role. Dedups identical spans within one call."""
    if not session_note or _IN_PROGRESS_MARKER not in session_note:
        return []
    if not text:
        return []
    if session_note in text:
        text = text.replace(session_note, " ")
    # 2026-09-15 (id=1435 fix): today's date, for the date-based prior-session
    # exemption below — None (unparseable) falls back to the old, looser
    # "any recognized date exempts" behavior, never to flagging more.
    today = _parse_today(session_note)

    findings: list[str] = []
    seen: set[str] = set()
    for family, pattern in (("closed_assertion", _CLOSED_ASSERTION_RE), ("completed_candle", _COMPLETED_CANDLE_RE)):
        for m in pattern.finditer(text):
            start, end = m.start(), m.end()
            if _CLOSE_ADVERB_RE.fullmatch(m.group(0)):
                continue
            left_ctx = text[max(0, start - 50) : start]
            if _NEGATED_RE.search(left_ctx):
                continue
            wide = text[max(0, start - _EXEMPT_WINDOW) : end + _EXEMPT_WINDOW]
            if (
                _PRIOR_SESSION_RE.search(wide)
                or _wide_has_non_today_date(wide, today)
                or _FUTURE_CLOSE_RE.search(wide)
                or _QUALIFIED_RE.search(wide)
                or _HEDGE_RE.search(wide)
                or _MA_RELATIVE_CLOSE_RE.search(wide)
            ):
                continue
            span = _quoted_span(text, start, end)
            key = f"{family}:{span}"
            if key in seen:
                continue
            seen.add(key)
            findings.append(f"{agent_name}: [{family}] {span!r} vs SESSION-IN-PROGRESS ground truth")

    if findings:
        logger.warning(
            "session_close_guard: %s narrated today's still-open session as a closed one "
            "(id=1358 shape) — %s",
            agent_name, findings,
        )
    return findings
