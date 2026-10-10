"""Vendor-agnostic LLM-response post-processing helpers shared by every
backend (vLLM/gateway today, Anthropic later), plus the priority-admission
concurrency limiter used to bound how many `complete`/`structured_complete`
calls may be in flight at once.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re

from app.config import settings
from app.llm.leak_markers import UNTAGGED_REASONING_MARKERS as _UNTAGGED_REASONING_MARKERS
from app.llm.leak_markers import UNTAGGED_REASONING_MIN_MARKERS as _UNTAGGED_REASONING_MIN_MARKERS

logger = logging.getLogger(__name__)


def _extract_json(raw: str) -> str:
    """vLLM's `response_format: {"type": "json_schema", "strict": True}` is a
    request, not a guarantee — observed live (2026-07-27 backtest re-run)
    that this model sometimes still wraps the object in markdown noise
    (e.g. "\\n\\n#{...}" or "\\n\\n**{...}"), which model_validate_json
    can't parse as-is. A no-op for already-clean output (the common case) —
    only strips text before the first '{' / after the last '}' when the
    string doesn't already start with one. Returns the input unchanged if
    no braces are found at all, so a genuinely malformed response still
    surfaces its own clear pydantic error instead of being silently
    swallowed here.

    2026-08-26 addition: a different guided-decoding artifact, observed
    live against nvidia/Qwen3.6-35B-A3B-NVFP4 — an EXTRA leading '{' beyond
    what the object's own closing braces balance ("{{...}" for what should
    be "{...}"), deterministic at temperature=0.0. The old fast path missed
    this: it only checks the first character, and "{{..." still starts with
    "{". Detected here by comparing raw '{'/'}' counts (a rough heuristic —
    a string VALUE containing brace characters would defeat it, the same
    class of imprecision this function already accepts in the markdown-
    stripping branch below) and ONLY trusted if the trimmed result actually
    parses — a truncated response (missing closing braces from hitting
    max_tokens, say) also has more '{' than '}' but trimming leading braces
    off that won't produce valid JSON either, so the parse attempt below
    correctly leaves it untouched rather than mangling it further."""
    stripped = raw.strip()
    if stripped.startswith("{"):
        opens, closes = stripped.count("{"), stripped.count("}")
        if opens > closes:
            trimmed = stripped[opens - closes:]
            if trimmed.startswith("{"):
                try:
                    json.loads(trimmed)
                    return trimmed
                except json.JSONDecodeError:
                    pass
        return stripped
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end < start:
        return stripped
    return stripped[start : end + 1]


def _clean_json_payload(raw: str, *, label: str, path: str) -> str:
    """`_extract_json` plus the "had to strip something" observability —
    shared by structured_complete's native AND fallback call sites so both
    log the same way. Before this, only the native path logged when cleanup
    did something; the fallback path (guided-decoding not even attempted)
    is exactly the path where seeing how often cleanup is needed matters
    most, and it had no visibility at all. `path` ("native"/"fallback")
    distinguishes which one logged it, so a rising rate on one specifically
    points at the right place to look (the vLLM server's guided-decoding
    config for "native", or the plain-completion prompt for "fallback")."""
    cleaned = _extract_json(raw) if raw else "{}"
    if raw and cleaned != raw.strip():
        logger.warning(
            "structured_complete label=%s (%s): stripped non-JSON wrapper before parsing (raw prefix: %r)",
            label, path, raw[:80],
        )
    return cleaned


# A paragraph must be reasonably long to count — short repeated fragments
# ("- ", "**Buy**") are normal in bullet/markdown output, not a loop.
_DEGENERATE_MIN_PARAGRAPH_LEN = 20
_DEGENERATE_MIN_REPEATS = 4

# 2026-09-16 addition — see _looks_degenerate's docstring for the incident
# (id=1514) this catches that the paragraph check above cannot: a run-on
# word sequence with no newlines at all, so there are no "blocks" to compare
# in the first place. 20 words is long enough that no legitimate phrase
# (a disclaimer, a repeated table header, boilerplate) recurs verbatim that
# many words in a row within one report by chance — validated against 160
# genuine same-day analyst reports (0 false positives) — while still being
# far shorter than the ~76-word cycle the id=1514 loop actually settled
# into, leaving margin either direction.
_DEGENERATE_NGRAM_WORDS = 20
_DEGENERATE_NGRAM_MIN_REPEATS = 4


# 2026-08-05: live-decision audit found Chinese-character (hanja) contamination
# in supposedly-Korean output escalating well past the single-word slips
# app.llm.language's own explicit anti-hanja examples were added for — one
# market_analyst report came back with an entire technical-indicators section
# written in Chinese, and several other same-day reports had scattered
# Korean/Chinese word-level mixing. Modern Korean writing is 순한글 (pure
# Hangul); any CJK ideograph in a "ko" response is code-switching, not
# legitimate content (unlike Japanese, Korean has no everyday use of Han
# characters mixed into ordinary prose). Same belt-and-suspenders shape as
# _looks_degenerate above — the system-prompt instruction is the primary
# defense, this is the runtime backstop for when it doesn't hold, sharing the
# same retry lever (a different temperature is the only way to get a
# different greedy-decoded output at all).
_HANJA_RE = re.compile(
    r"[㐀-䶿一-鿿豈-﫿\U00020000-\U0002FA1F]"
)  # CJK Unified Ideographs (一-鿿) + Ext A (㐀-䶿) + Compatibility
   # Ideographs (豈-﫿, exactly the block Korean-reading-compatibility
   # hanja live in — the gap the original range missed) + Ext B and beyond /
   # Compatibility Supplement (\U00020000-\U0002FA1F)


def _contains_hanja(content: str) -> bool:
    return bool(_HANJA_RE.search(content))


def _looks_degenerate(content: str) -> bool:
    """True if the same (long enough) line/paragraph (block delimited by
    ANY newline, not just a blank-line-separated paragraph) repeats often
    enough that this is almost certainly a stuck greedy-decoding loop
    rather than genuine content — real trading analysis doesn't
    legitimately repeat an entire block verbatim four-plus times.

    2026-08-26: split on every newline, not just "\n\n" — a loop that
    repeats the same single line back-to-back with no blank line between
    repeats (a plausible decoding-loop shape, distinct from the
    blank-line-separated-paragraph loop this was originally built to catch)
    used to go completely undetected. Splitting on "\n\n" is a strict
    subset of splitting on every newline, so this only widens detection,
    never narrows it — every block the old check caught, the new one still
    catches.

    2026-09-16: live decision id=1514 (Fundamentals Analyst, 379800.KS)
    produced a 103,087-char report that this check completely missed —
    total_tokens=359,737, about 2x the recent per-decision average, and the
    report's tail is a nonsense chain of synonym-then-suffix-mutated words
    ("...spreading widening vastifying extending comprehensivizing
    thoroughfying exhaustivizing meticulouslypaining...") that DOES lock
    into a genuine ~76-word / ~934-char repeating cycle partway through —
    but the entire runaway section (>97,000 chars, from shortly after
    section 6 of the report onward) contains a grand total of two newline
    characters. It is one giant paragraph by this function's own
    definition, so `blocks` above has exactly one entry and can never reach
    _DEGENERATE_MIN_REPEATS regardless of how many times the cycle actually
    repeats inside it. `_has_repeating_word_ngram` below is newline-agnostic
    for exactly this reason: it slides a fixed-size word window across the
    raw text irrespective of line breaks, so a repeating cycle is caught
    whether or not the model ever emits a newline. Deliberately narrower
    than a general "vocabulary expansion" or compression-ratio detector
    (avg-word-length drift, suffix-frequency spikes, etc. were considered
    and set aside) — this only fires once the runaway actually starts
    repeating verbatim, which is what id=1514 did and what a synonym chain
    that never repeats structurally cannot avoid doing once it exhausts
    the vocabulary it keeps drawing on. Exact-match only, same false-positive
    floor as the paragraph check above and separately validated (see that
    constant's comment) against 160 genuine same-day reports with zero hits."""
    blocks = [p.strip() for p in re.split(r"\n+", content) if len(p.strip()) >= _DEGENERATE_MIN_PARAGRAPH_LEN]
    counts: dict[str, int] = {}
    for p in blocks:
        counts[p] = counts.get(p, 0) + 1
        if counts[p] >= _DEGENERATE_MIN_REPEATS:
            return True
    return _has_repeating_word_ngram(content)


def _has_repeating_word_ngram(
    content: str,
    *,
    n: int = _DEGENERATE_NGRAM_WORDS,
    min_repeats: int = _DEGENERATE_NGRAM_MIN_REPEATS,
) -> bool:
    """True if some `n`-word run appears verbatim `min_repeats`+ times
    anywhere in `content`, ignoring line breaks entirely — see
    _looks_degenerate's 2026-09-16 note for why that's necessary. Works for
    any repeat period up to `n` words: at that period, every window whose
    start lands on the same phase of the cycle is identical, and repeats
    accumulate the count for that phase's key well before `min_repeats`
    windows of run-on text have been scanned."""
    words = content.split()
    if len(words) < n * min_repeats:
        return False
    counts: dict[str, int] = {}
    for i in range(len(words) - n + 1):
        gram = " ".join(words[i : i + n])
        counts[gram] = counts.get(gram, 0) + 1
        if counts[gram] >= min_repeats:
            return True
    return False


# 2026-09-01, first real-account KRX cycle. id=1156's risk_debate[1]
# Conservative turn was 19 lines of argument, then ~330 lines of unfiltered
# deliberation, then the answer, ending in a literal "</think>" — the leak
# ~10x longer than the answer, and it CHANGED THE DECISION (the trace
# flip-flopped and settled on a meta-heuristic about its own prompt). id=1157
# leaked ~150 lines including this project's own prompt text quoted verbatim.
# Nothing existing catches it: _looks_degenerate targets repetition, the
# empty-body guard targets blanks, _REASONING_EFFORT targets the OPPOSITE
# failure (all budget spent on hidden reasoning, empty body).
#
# TAG-BASED ONLY, deliberately. The false-positive floor is near zero because
# app.agents.untrusted_content.wrap_untrusted HTML-escapes angle brackets in
# every piece of third-party text before it reaches the model, so injected
# content cannot forge a `<think>` for us to match. The untagged shape
# (id=1157) is detect-only — see tier 4.
_THINK_BLOCK_RE = re.compile(r"<\s*think\s*>.*?<\s*/\s*think\s*>", re.DOTALL | re.IGNORECASE)
_THROUGH_LAST_CLOSE_RE = re.compile(r"^.*<\s*/\s*think\s*>", re.DOTALL | re.IGNORECASE)
_OPEN_TO_END_RE = re.compile(r"<\s*think\s*>.*$", re.DOTALL | re.IGNORECASE)
_ANY_OPEN_RE = re.compile(r"<\s*think\s*>", re.IGNORECASE)
_ANY_CLOSE_RE = re.compile(r"<\s*/\s*think\s*>", re.IGNORECASE)

# Below this many characters, a "sanitized" body is not trusted and the
# ORIGINAL is returned instead — the safety invariant in _strip_reasoning_leak.
_MIN_SANITIZED_BODY_CHARS = 200

# Detect-only (tier 4) — never strips. Two or more DISTINCT markers is the
# threshold: any one appears in ordinary argument, several together is the
# id=1157 shape. Deleting real analysis from a real-money decision is
# invisible and unrecoverable; leaking is ugly but fully auditable, so the
# asymmetry says detect, don't strip.
#
# 2026-09-01 (root cause #9): the marker tuple itself now lives in
# app.llm.leak_markers (imported above) so app.agents.untrusted_content.
# redact_injection_flag can score the same leaked-deliberation shape without
# a second, driftable copy. Kept under these original private names so every
# comment/test below still reads as "this module's own markers."


def _strip_reasoning_leak(content: str, *, label: str) -> tuple[str, str | None]:
    """(body, kind) — `kind` is None when nothing was done, otherwise
    "think_block" / "through_close_tag" / "open_to_end" / "declined_too_short"
    / "untagged_suspected".

    SAFETY INVARIANT: if the input is non-empty, the output is non-empty.
    When stripping would leave fewer than _MIN_SANITIZED_BODY_CHARS
    characters, the ORIGINAL is returned unchanged. complete()'s
    empty-response guard therefore sees exactly the same emptiness it would
    have seen without this function — this change cannot alter that guard's
    firing rate and cannot recreate the 2026-08-31 fallback-loop incident.
    빈 본문을 만들지 않는 것이 이 함수의 제1 불변식."""
    if not settings.llm_strip_reasoning_leak or not content:
        return content, None
    original = content
    kind: str | None = None
    if _THINK_BLOCK_RE.search(content):
        content, kind = _THINK_BLOCK_RE.sub("", content), "think_block"
    if _ANY_CLOSE_RE.search(content) and not _ANY_OPEN_RE.search(content):
        content, kind = _THROUGH_LAST_CLOSE_RE.sub("", content, count=1), "through_close_tag"
    elif _ANY_OPEN_RE.search(content) and not _ANY_CLOSE_RE.search(content):
        content, kind = _OPEN_TO_END_RE.sub("", content, count=1), "open_to_end"

    if kind is None:
        hits = sum(1 for p in _UNTAGGED_REASONING_MARKERS if p.search(original))
        if hits >= _UNTAGGED_REASONING_MIN_MARKERS:
            logger.warning(
                "llm complete label=%s: response looks like it contains untagged model "
                "deliberation (%d markers) — NOT stripped (see _strip_reasoning_leak tier 4); "
                "chars=%d", label, hits, len(original),
            )
            return original, "untagged_suspected"
        return original, None

    stripped = content.strip()
    if len(stripped) < _MIN_SANITIZED_BODY_CHARS:
        logger.warning(
            "llm complete label=%s: declined to strip a reasoning leak (%s) — only %d chars would "
            "remain of %d, so the original is kept rather than manufacturing a near-empty body "
            "(see _strip_reasoning_leak's invariant / the 2026-08-31 fallback-loop incident)",
            label, kind, len(stripped), len(original),
        )
        return original, "declined_too_short"

    logger.warning(
        "llm complete label=%s: stripped a reasoning leak from the response body (%s, removed %d "
        "of %d chars, kept prefix=%r)",
        label, kind, len(original) - len(stripped), len(original), stripped[:80],
    )
    return stripped, kind


# Only two tiers, same convention as app.brokers.base.http_client's KIS
# throttle (ai-engine-24h-operation-analysis P0-1 ③): a caller passing an
# unrecognized string is treated as "low" rather than raising, so an
# unrecognized value is never accidentally "high".
_PRIORITY_RANK = {"high": 0, "low": 1}
# Same starvation guard as the KIS ticket queue, same reason: a steady
# stream of live ("high") calls could otherwise starve out a backtest
# ("low") indefinitely once one is already waiting.
_LOW_STARVATION_LIMIT = 20


class _PriorityWaiter:
    __slots__ = ("rank", "skipped", "event")

    def __init__(self, rank: int) -> None:
        self.rank = rank
        self.skipped = 0
        self.event = asyncio.Event()


class PriorityLimiter:
    """Bounded concurrency (`max_concurrent` requests in flight at once),
    priority-ordered admission when the cap is full — the LLM-client
    counterpart to app.brokers.base.http_client.BaseHttpClient's KIS
    throttle, which never existed here (see ai-engine-24h-operation-
    analysis's P0-1 ③: live and backtest/analysis calls used to share this
    one vLLM endpoint with no concept of priority at all, unlike KIS's
    order/safety calls vs routine polling).

    Shaped differently from the KIS queue on purpose: KIS needs a MINIMUM
    INTERVAL between individual requests (a hard per-second rate limit), so
    its queue serves exactly one ticket at a time. vLLM does its own
    continuous batching server-side, so the relevant constraint here is a
    CONCURRENCY CAP (how many `complete`/`structured_complete` calls may be
    in flight at once) — this class serves up to `max_concurrent` waiters
    simultaneously, and when a slot frees up, the highest-priority waiter
    (not simply the earliest arrival) gets it next.

    `priority` is inferred automatically by the caller from
    `app.config.is_backtest_context()` — no agent call site needs to know
    this class exists, exactly the same "ambient context, zero call-site
    changes" pattern that context var already established for is_backtest
    trades-table isolation.

    No `asyncio.Lock` here (2026-08-26 fix — see acquire()/release()): every
    state mutation below is synchronous, with no `await` inside it, so the
    single-threaded event loop already makes each mutation atomic on its
    own. The lock used to exist anyway, but since its critical sections
    never awaited, it protected nothing — it only forced acquire()'s
    cancellation-cleanup path to be async, which is exactly what let a
    cancellation land *during* cleanup and leak a slot a second time. Making
    every mutation synchronous means a cancellation can only ever land
    before or after a mutation, never inside one — cleanup cannot itself be
    interrupted.

    Real incident this fixes: a waiter cancelled while parked in
    `event.wait()` (this repo already defends against "unrequested"
    CancelledError from a suspected connection-pool issue at several other
    await points — app.orchestrator.pipeline, app.live.auto_trade_loop,
    app.agents.tool_loop) used to stay in `_waiters` forever. The next
    release() would pop that ghost waiter, set its already-abandoned event,
    and NOT decrement `_in_flight` (by design — see release() below) —
    permanently leaking one slot. With `llm_max_concurrent_requests=8`
    (app.config), 8 such cancellations silently wedge the entire AI
    pipeline forever (no timeout on acquire(), no exception — the dashboard
    stays up, judgments just stop appearing)."""

    def __init__(self, max_concurrent: int) -> None:
        self._max_concurrent = max_concurrent
        self._in_flight = 0
        self._waiters: list[_PriorityWaiter] = []

    def _release_slot(self) -> None:
        """Synchronous core of release() — see class docstring for why this
        must never contain an `await`. Also called from acquire()'s
        cancellation cleanup (case 2 below) to hand an already-owned slot
        onward, so this must stay a plain function, not `release()` itself
        (which is `async def` only for call-site compatibility — see below)."""
        if not self._waiters:
            self._in_flight -= 1
            return
        # Starvation guard first — any waiter that's already lost this
        # many rounds to a higher-priority one goes next regardless of
        # rank, same rule and same limit as the KIS ticket queue.
        chosen_idx = next(
            (i for i, w in enumerate(self._waiters) if w.skipped >= _LOW_STARVATION_LIMIT),
            None,
        )
        if chosen_idx is None:
            chosen_idx = min(range(len(self._waiters)), key=lambda i: self._waiters[i].rank)
        chosen = self._waiters.pop(chosen_idx)
        for other in self._waiters:
            other.skipped += 1
        # The slot isn't actually freed — it's handed directly to `chosen`,
        # so `_in_flight` stays exactly as it was. Pop and set happen in the
        # same synchronous call (no await between them) so "popped from
        # _waiters" and "event is set" become one atomic fact — acquire()'s
        # cancellation handler relies on that to tell the two possible
        # cancellation timings apart (see there).
        chosen.event.set()

    async def acquire(self, priority: str = "low") -> None:
        if self._in_flight < self._max_concurrent:
            self._in_flight += 1
            return
        waiter = _PriorityWaiter(rank=_PRIORITY_RANK.get(priority, _PRIORITY_RANK["low"]))
        self._waiters.append(waiter)
        try:
            await waiter.event.wait()
        except asyncio.CancelledError:
            # No `await` anywhere in this handler, all the way to `raise` —
            # required so a second cancellation can't land mid-cleanup (see
            # class docstring). Two distinct timings are possible here,
            # told apart by whether our event was already set:
            if waiter.event.is_set():
                # Case 2: release() already popped us from _waiters and
                # handed us the slot (chosen.event.set(), above) before this
                # task got to resume — asyncio can still deliver the
                # cancellation on the next resumption instead of dropping
                # it. We own the slot at this point, so it must be handed
                # onward (or given back), exactly like a normal release().
                self._release_slot()
            else:
                # Case 1: still genuinely waiting — release() can never have
                # popped us (that only happens together with event.set(),
                # per _release_slot()'s comment), so we are certainly still
                # in _waiters. Remove ourselves; nothing else to undo since
                # we never held a slot.
                self._waiters.remove(waiter)
            raise  # never swallow a real cancellation — see app.cancellation
        return

    async def release(self) -> None:
        self._release_slot()
