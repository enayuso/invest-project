"""Aggregates `app.experiments.language_ab` output into a Korean-vs-English
comparison table.

Every metric function below is a pure function over the already-loaded JSON
records (no I/O of its own) specifically so it can be unit-tested without a
GPU, a DB, or a real sweep run — see `tests/experiments/
test_language_ab_report.py`.

Metrics, and why each one is here rather than a single pass/fail number:

  - Hanja contamination (ko only): the same CJK-ideograph check
    `app.llm.client._contains_hanja` uses (duplicated locally below, not
    imported — importing that private symbol pulls in `app.llm.client`'s
    module-level `llm_client = LLMClient()` construction as a side effect of
    merely loading this reporting module, which contradicts "pure function,
    no I/O of its own" above) re-run over every text field in a record.
    `en` has no equivalent failure mode,
    so it is reported as N/A, not 0% — a 0% would read as "en is cleaner,"
    which is not a claim this metric can support.
  - Self-consistency violations: cheap structural checks (confidence out of
    [0,1], a Buy/Overweight stop_loss at or above the entry price, rating vs.
    prose direction via keyword counts, and an explicit rating claim in the
    prose — e.g. "Buy 등급을 부여합니다" — that contradicts the structured
    rating field) that would be a bug regardless of which language produced
    them. The keyword-count check deliberately skips Hold (see
    `_rating_direction_violation`); the explicit-claim check is what catches
    Hold's real failure mode instead.
  - Magnitude outliers: prose numbers that are ~10x (or ~1/10x) a
    structured field's own value — the exact signature of a comma/units
    confusion bug seen before in this project (see project memory:
    llm-numeric-hallucination-root-causes). Heuristic on purpose: it flags
    CANDIDATES for a human to look at, not a verdict.
  - Run-to-run variance (same ticker/date/language, `--repeats` runs): the
    noise floor. A ko-vs-en gap that doesn't clear this floor isn't a
    language effect, it's decoding variance — reported side by side with
    the ko-vs-en gap specifically so that comparison is unavoidable to make.
  - Cost: mean wall-clock seconds per run, by language.
  - Isolation violations: `record["isolation_violation"]` (this project's
    own `past_context`-non-empty safety-net flag — see `language_ab`'s
    module docstring) count. Any non-zero count here means the OTHER
    metrics in this report may be contaminated and should not be trusted
    until investigated.

    python -m app.experiments.language_ab_report --dir results/language_ab/20260814T120000
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

# Duplicated from app.llm.client's _HANJA_RE/_contains_hanja rather than
# imported — see the module docstring's Hanja contamination bullet for why.
_HANJA_RE = re.compile(r"[一-鿿㐀-䶿]")  # CJK Unified Ideographs + Extension A


def _contains_hanja(content: str) -> bool:
    return bool(_HANJA_RE.search(content))


if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

logger_prefix = "[language_ab_report]"

_BULLISH_KEYWORDS = [
    "상승", "매수", "긍정", "호재", "강세", "buy", "bullish", "upside", "positive", "rally", "outperform",
]
_BEARISH_KEYWORDS = [
    "하락", "매도", "부정", "악재", "약세", "sell", "bearish", "downside", "negative", "decline", "underperform",
]

# Hold is deliberately NOT in here: it isn't a long entry (no position to
# protect with a stop_loss check), and it isn't bearish either. See
# `_rating_direction_violation` for why tone-based checks don't extend to it.
_BULLISH_RATINGS = {"Buy", "Overweight"}
_BEARISH_RATINGS = {"Sell", "Underweight"}

_NUMBER_RE = re.compile(r"\d{1,3}(?:,\d{3})*(?:\.\d+)?")
_MAGNITUDE_LOW, _MAGNITUDE_HIGH = 9.0, 11.0  # "~10x" tolerance band, both directions

# Explicit rating-claim tokens. Note: even in ko output, the rating word
# itself comes through in English ("Buy 등급을 부여합니다") — the LLM does not
# translate Buy/Sell/Hold/Overweight/Underweight, so a single token set
# covers both languages' prose.
_RATING_TOKENS = ("Buy", "Sell", "Hold", "Overweight", "Underweight")
_RATING_TOKEN_CANON = {t.lower(): t for t in _RATING_TOKENS}
_RATING_TOKEN_ALT = "|".join(_RATING_TOKENS)

# Precision-first by construction (per the module's own design principle):
# a bare rating word ("Buy 신호가 약하다", "Buy를 고려했으나 기각") must NOT
# match — only text where the token sits directly in an assignment/verdict
# context does. Two shapes, matching the real ko_r2 record this check exists
# for ("...Buy 등급을 부여합니다..."):
#   - token first, assignment noun/verb right after: "Buy 등급(을) 부여",
#     "Sell 의견", "Hold로 판단", "Buy로 확정"
#   - assignment phrase first, token right after (English style):
#     "rating: Buy", "recommendation: Hold", "assign a Buy", "we rate ... Sell"
# NOTE on why this does NOT use `\b`: Python's `re` module is Unicode-aware,
# so a Hangul syllable counts as a `\w` character just like a Latin letter —
# there is no word-boundary between the Latin "y"/"l" and the Hangul
# syllable that follows it in "Buy로" or "Sell을", which is the single most
# natural way Korean attaches a particle/verb directly to an untranslated
# English token (no space). `\b` silently fails to match there, which is why
# the "(로|으로) + 판단/평가/결정/확정" branch below existed in the code but
# had never actually fired on real ko output. Do NOT change this back to
# `\b` — use "not immediately preceded/followed by an ASCII letter" instead,
# which still treats a directly-attached Korean particle as a boundary.
_RATING_CLAIM_KO_RE = re.compile(
    rf"(?<![A-Za-z])({_RATING_TOKEN_ALT})(?![A-Za-z])\s*"
    r"(?:(?:등급|의견|평가)(?:을|를)?\s*(?:부여|판단|제시|권고|확정)?"
    r"|(?:로|으로)\s*(?:판단|평가|결정|확정)"
    # Josa + explicit assignment verb, no "등급/의견/평가" noun needed —
    # catches "Sell을 권고합니다" / "Buy를 확정합니다". Deliberately narrow:
    # only unambiguous grant/decide verbs, NOT "고려/검토/언급" (consider/
    # review/mention) — those would false-positive on exactly the kind of
    # "mentioned but not assigned" prose this whole regex is precision-first
    # about excluding (see `test_rating_claim_contradicts_field_ignores_a_
    # bare_mention_without_assignment`).
    r"|(?:을|를)\s*(?:권고|제시|부여|확정|결정))",
    re.IGNORECASE,
)
_RATING_CLAIM_EN_RE = re.compile(
    r"(?:\brating\s*[:\-]\s*|\brecommendation\s*[:\-]\s*|\bassign(?:s|ed)?\s+an?\s+"
    r"|\bwe\s+rate\b[^.]{0,60}?\b|\brate\s+(?:it|this|the\s+stock)\s+(?:as\s+)?)"
    rf"({_RATING_TOKEN_ALT})(?![A-Za-z])",
    re.IGNORECASE,
)

# Known limitation (documented, not "fixed" — see module task): a negated
# claim like "Buy를 권고하지 않습니다" ("does NOT recommend Buy") still
# matches the josa+verb branch above up through "권고", because the verb
# itself is what the regex looks for and the negation trails it. This guard
# is a deliberately SHORT lookahead for the most common Korean negation
# markers immediately after a claim match, not a general negation parser —
# a full negation-scope parser is out of scope (module task: "don't build an
# overly complex parser"). Claims followed by one of these within a few
# characters are dropped rather than counted.
_NEGATION_GUARD_WINDOW = 12
_NEGATION_GUARD_RE = re.compile(r"^\s*(?:하지\s*않|않습니다|않다|기각|배제|아니)")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_records(out_dir: Path) -> tuple[list[dict], list[dict]]:
    """Returns (successful run records, error records) — `*_error.json`
    files are kept separate since they carry no decision/report content to
    aggregate, only a failure reason.

    One unreadable/corrupt file is skipped (with a warning), not fatal to the
    whole report — a run's write can be interrupted mid-sweep (power loss,
    kill -9; language_ab.py's out_path write is now atomic going forward, but
    old result directories from before that fix can still hold a truncated
    file), and there is no reason a single bad file should crash a report
    over dozens of otherwise-good ones."""
    records: list[dict] = []
    errors: list[dict] = []
    for path in sorted(out_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            print(f"WARNING: skipping unreadable record {path.name}: {exc!r}", file=sys.stderr)
            continue
        if path.name.endswith("_error.json"):
            errors.append(data)
        else:
            # out_dir.glob("*.json") is unfiltered — any unrelated JSON file
            # dropped into the same directory would otherwise reach every
            # downstream function that indexes record["language"]/["ticker"]/
            # ["date"] with a bare [] and crash with a KeyError far from
            # here. Checked once, at the one place that reads the raw file.
            if not isinstance(data, dict) or not {"ticker", "date", "language"} <= data.keys():
                print(f"WARNING: skipping {path.name} — not a language_ab run record (missing required keys)",
                      file=sys.stderr)
                continue
            records.append(data)
    return records, errors


# ---------------------------------------------------------------------------
# Hanja contamination (ko only)
# ---------------------------------------------------------------------------


def record_text_blob(record: dict) -> str:
    """Every prose field a hanja slip (or a magnitude outlier) could hide
    in: the final decision's own text, every analyst report, both debates'
    full transcripts, the trader proposal, and the research plan."""
    parts: list[str] = []
    final_decision = record.get("final_decision") or {}
    # card_reason_ko/card_reason_en (app/state.py) are separate Korean/
    # English prose fields on the same final_decision model, added after
    # this experiment concluded — the dashboard signal card's short "why",
    # distinct from executive_summary's action plan. card_reason_ko is
    # exactly the kind of free-generated Korean prose a hanja slip would
    # show up in, and it was silently excluded from this scan.
    for key in ("executive_summary", "investment_thesis", "time_horizon", "card_reason_ko", "card_reason_en"):
        value = final_decision.get(key)
        if value:
            parts.append(str(value))
    for text in (record.get("reports") or {}).values():
        if text:
            parts.append(str(text))
    for debate_key in ("research_debate", "risk_debate"):
        debate = record.get(debate_key) or {}
        for turn in debate.get("turns") or []:
            content = turn.get("content")
            if content:
                parts.append(str(content))
    trader_proposal = record.get("trader_proposal") or {}
    if trader_proposal.get("reasoning"):
        parts.append(str(trader_proposal["reasoning"]))
    research_plan = record.get("research_plan") or {}
    for key in ("rationale", "strategic_actions"):
        value = research_plan.get(key)
        if value:
            parts.append(str(value))
    return "\n".join(parts)


def record_has_hanja(record: dict) -> bool:
    return _contains_hanja(record_text_blob(record))


def hanja_contamination_rate(records: list[dict]) -> float | None:
    """Fraction of `records` (expected to already be filtered to
    language == "ko" by the caller) with at least one hanja hit. `None` for
    an empty input — "no ko runs" is not the same as "0% contamination."""
    if not records:
        return None
    return sum(1 for r in records if record_has_hanja(r)) / len(records)


# ---------------------------------------------------------------------------
# Self-consistency violations
# ---------------------------------------------------------------------------


def _rating_direction_violation(rating: str, text: str) -> bool:
    # Hold is intentionally exempt (falls through to `return False` below),
    # not an oversight: a Hold write-up is expected to argue both directions
    # (that's what makes it a Hold rather than a Buy or a Sell), so a raw
    # bullish-vs-bearish keyword count on Hold prose is structurally noisy —
    # it would flag well-reasoned Hold write-ups as often as bad ones. Hold's
    # actual failure mode (prose explicitly asserting a different rating,
    # e.g. "Buy 등급을 부여합니다" under a Hold record) is caught instead by
    # `rating_claim_contradicts_field`, which looks for an explicit verdict
    # rather than counting sentiment words.
    lowered = text.lower()
    bullish_hits = sum(lowered.count(k.lower()) for k in _BULLISH_KEYWORDS)
    bearish_hits = sum(lowered.count(k.lower()) for k in _BEARISH_KEYWORDS)
    if rating in _BULLISH_RATINGS:
        return bearish_hits > bullish_hits
    if rating in _BEARISH_RATINGS:
        return bullish_hits > bearish_hits
    return False


def _find_rating_claims(text: str) -> list[str]:
    """Rating tokens explicitly claimed/assigned in `text`, in order of
    appearance. See the `_RATING_CLAIM_*_RE` docstring-comment above for why
    this is precision-first (bare mentions don't count), and
    `_NEGATION_GUARD_RE`'s docstring for the known limitation on negated
    claims (e.g. "Buy를 권고하지 않습니다") — a short guard catches the common
    case but this is not a full negation parser."""
    hits: list[tuple[int, str]] = []
    for regex in (_RATING_CLAIM_KO_RE, _RATING_CLAIM_EN_RE):
        for m in regex.finditer(text):
            trailing = text[m.end():m.end() + _NEGATION_GUARD_WINDOW]
            if _NEGATION_GUARD_RE.match(trailing):
                continue
            hits.append((m.start(), _RATING_TOKEN_CANON[m.group(1).lower()]))
    hits.sort(key=lambda pair: pair[0])
    return [token for _, token in hits]


def rating_claim_contradicts_field(record: dict) -> bool:
    """True if the prose explicitly assigns a rating (per `_find_rating_claims`)
    that differs from `final_decision.rating`. This is what catches the ko_r2
    case `_rating_direction_violation` misses entirely for Hold: prose that
    says "Buy 등급을 부여합니다" while the structured field says Hold. If
    multiple explicit claims appear (rare), the LAST one in reading order is
    treated as the final claim — mirrors how a human reader would resolve an
    earlier draft sentence being superseded by a later one in the same
    write-up, and matches the ko_r2 record itself (a later "Buy로 확정합니다"
    reaffirms the same contradicting claim, not a retraction of it)."""
    final_decision = record.get("final_decision") or {}
    rating = final_decision.get("rating")
    if not rating:
        return False
    text = f"{final_decision.get('executive_summary', '')} {final_decision.get('investment_thesis', '')}"
    claims = _find_rating_claims(text)
    if not claims:
        return False
    return claims[-1] != rating


def self_consistency_violations(record: dict) -> list[str]:
    """Returns the list of violation names found in `record` (empty if
    none). Deliberately cheap/heuristic per the module docstring — these
    are meant to be counted and spot-checked, not treated as ground truth
    on any single run.

    `rating_claim_contradicts_field` is reclassified to
    `stale_prose_after_gate_override` when the record was hysteresis-gated
    (per `pre_gate_rating`): the "contradiction" in that case isn't the
    model disagreeing with itself, it's `_apply_hysteresis_gate` silently
    swapping the STRUCTURED rating to Hold while leaving the PROSE it wrote
    before the override untouched (the exact ko_r2 case this whole report
    revision was written for — see module memory
    live-rebalance-equity-basis-mismatch-2026-08-14 for the sibling
    equity-basis bug and the project handoff doc for this one). Folding it
    into the language's own self-consistency count would misread a
    language-agnostic pipeline behavior as "ko is less coherent than en,"
    which is not what happened."""
    violations: list[str] = []
    final_decision = record.get("final_decision")
    if not final_decision:
        return violations

    confidence = final_decision.get("confidence")
    if confidence is not None and not (0.0 <= confidence <= 1.0):
        violations.append("confidence_out_of_range")

    rating = final_decision.get("rating")
    stop_loss = final_decision.get("stop_loss")
    entry_price = (record.get("trader_proposal") or {}).get("entry_price")
    if rating in _BULLISH_RATINGS and stop_loss is not None and entry_price is not None and stop_loss >= entry_price:
        violations.append("stop_loss_at_or_above_entry")

    if rating is not None:
        text = f"{final_decision.get('executive_summary', '')} {final_decision.get('investment_thesis', '')}"
        if _rating_direction_violation(rating, text):
            violations.append("rating_text_direction_mismatch")

    if rating_claim_contradicts_field(record):
        _, downgraded, parse_failed = _pre_gate_rating_safe(record)
        if downgraded and not parse_failed:
            violations.append("stale_prose_after_gate_override")
        else:
            violations.append("rating_claim_contradicts_field")

    return violations


# ---------------------------------------------------------------------------
# Magnitude outliers (10x-error signature)
# ---------------------------------------------------------------------------


def _parse_number(token: str) -> float | None:
    try:
        return float(token.replace(",", ""))
    except ValueError:
        return None


def magnitude_outlier_candidates(record: dict) -> list[dict[str, Any]]:
    """Prose numbers (in executive_summary/investment_thesis) that sit
    within `_MAGNITUDE_LOW`-`_MAGNITUDE_HIGH`x (or its inverse) of a
    structured field's own value — candidates for a human to check, not a
    confirmed bug. Structured fields checked: stop_loss, price_target,
    entry_price."""
    final_decision = record.get("final_decision") or {}
    structured: dict[str, float] = {}
    for key in ("stop_loss", "price_target"):
        value = final_decision.get(key)
        if isinstance(value, (int, float)) and value:
            structured[key] = float(value)
    entry_price = (record.get("trader_proposal") or {}).get("entry_price")
    if isinstance(entry_price, (int, float)) and entry_price:
        structured["entry_price"] = float(entry_price)
    if not structured:
        return []

    text = f"{final_decision.get('executive_summary', '')} {final_decision.get('investment_thesis', '')}"
    candidates: list[dict[str, Any]] = []
    for token in _NUMBER_RE.findall(text):
        found = _parse_number(token)
        if found is None or found == 0:
            continue
        for field_name, structured_value in structured.items():
            ratio = found / structured_value
            if _MAGNITUDE_LOW <= ratio <= _MAGNITUDE_HIGH or _MAGNITUDE_LOW <= (1.0 / ratio) <= _MAGNITUDE_HIGH:
                candidates.append({
                    "field": field_name,
                    "structured_value": structured_value,
                    "prose_number": found,
                    "ratio": ratio,
                })
    return candidates


# ---------------------------------------------------------------------------
# Pre-gate rating recovery (Part B — the hysteresis gate's confidence-unset
# downgrade masks the model's own judgment behind a system override; see
# app.orchestrator.pipeline._apply_hysteresis_gate)
# ---------------------------------------------------------------------------

# Mirrors the exact fixed-format note `_apply_hysteresis_gate` appends to
# `investment_thesis` (app/orchestrator/pipeline.py:103-106). Recovering the
# pre-gate rating by REGEX PARSING that note — rather than changing
# pipeline.py to record it structurally — is deliberate: production code is
# out of scope for this investigation (see module task's own constraint),
# and every one of the 6 pilot runs already on disk was generated before any
# such structural field could exist, so a parser is the only way to recover
# their pre-gate judgment retroactively. A structural field would only help
# FUTURE runs; this recovers the ones already collected.
_HYSTERESIS_NOTE_RE = re.compile(
    r"\[Hysteresis gate: downgraded from (?P<orig>\w+) to Hold — confidence "
    r"\((?P<conf>[^)]*)\) below the (?P<pct>[\d.]+)% bar required to flip position state\.\]"
)
_HYSTERESIS_MARKER = "[Hysteresis gate:"

# The desk-consensus override gate (app/orchestrator/pipeline.py's
# _apply_consensus_override_gate, 2026-09-04 id=1212/1217) downgrades a
# Buy/Overweight to Hold the same way the hysteresis gate does, and appends
# its own fixed-format bracketed note to `investment_thesis`. Without parsing
# it here, `pre_gate_rating` would fall through to "never gated" and return
# the post-downgrade Hold as if it were the model's own judgment — the exact
# silent-undercount this whole Part B exists to prevent, one gate deeper.
# Only the leading "downgraded from X to Hold — " is matched; the rest of the
# note carries the RM/Trader/confidence detail and is not needed to recover
# the pre-gate rating.
_CONSENSUS_NOTE_RE = re.compile(
    r"\[Consensus override gate: downgraded from (?P<orig>\w+) to Hold — "
)
_CONSENSUS_MARKER = "[Consensus override gate:"


class HysteresisNoteParseError(ValueError):
    """Raised by `pre_gate_rating` when `investment_thesis` contains a gate
    marker (`[Hysteresis gate:` or `[Consensus override gate:`) but the rest
    of that note doesn't match its expected fixed format.

    This must NOT be swallowed into "no downgrade" — every metric downstream
    of `pre_gate_rating` (variance-by-language, gate-downgrade counts,
    confidence-missing rate) exists specifically to un-hide what the gate
    overrode, so silently treating an unparseable note as "not gated" would
    quietly resurrect the exact bug this whole Part B is trying to fix, just
    one layer deeper. Callers that must keep aggregating past one bad record
    use `_pre_gate_rating_safe` instead and surface the failure as its own
    reported category (see `gate_note_parse_failures`)."""


def pre_gate_rating(record: dict) -> tuple[str | None, bool]:
    """Returns (pre_gate_rating, was_downgraded).

    If `investment_thesis` has neither the `[Hysteresis gate:` nor the
    `[Consensus override gate:` marker, the record was never gated: returns
    `(final_decision.rating, False)` — the current rating already IS the
    model's own judgment.

    If a gate marker is present, parses out the ORIGINAL rating the model
    produced before that gate (`_apply_hysteresis_gate` or
    `_apply_consensus_override_gate` in app.orchestrator.pipeline) downgraded
    it to Hold and returns `(original_rating, True)`. Raises
    `HysteresisNoteParseError` if a marker is present but the text doesn't
    match its expected format — see that exception's own docstring for why
    this is a hard failure rather than a silent fallback. The two gates
    cannot co-occur on one decision (the consensus gate runs first and its
    downgrade to Hold makes the hysteresis gate a no-op), but if both markers
    were ever present the hysteresis one is resolved first.
    """
    final_decision = record.get("final_decision") or {}
    thesis = final_decision.get("investment_thesis") or ""
    if _HYSTERESIS_MARKER in thesis:
        match = _HYSTERESIS_NOTE_RE.search(thesis)
        if not match:
            raise HysteresisNoteParseError(
                f"record {_group_key(record)} r{record.get('repeat')}: investment_thesis contains "
                f"'{_HYSTERESIS_MARKER}' but did not match the expected fixed format — pipeline.py's "
                f"note format may have changed; pre_gate_rating needs updating to match."
            )
        return match.group("orig"), True
    if _CONSENSUS_MARKER in thesis:
        match = _CONSENSUS_NOTE_RE.search(thesis)
        if not match:
            raise HysteresisNoteParseError(
                f"record {_group_key(record)} r{record.get('repeat')}: investment_thesis contains "
                f"'{_CONSENSUS_MARKER}' but did not match the expected fixed format — pipeline.py's "
                f"note format may have changed; pre_gate_rating needs updating to match."
            )
        return match.group("orig"), True
    return final_decision.get("rating"), False


def _pre_gate_rating_safe(record: dict) -> tuple[str | None, bool, bool]:
    """(pre_gate_rating, was_downgraded, parse_failed) — never raises.

    Aggregate metrics over many records can't let one malformed note abort
    the whole report; this is the sanctioned place that catches
    `HysteresisNoteParseError`, and it still refuses to call a parse failure
    "not downgraded" — the third element flags it so callers can report it
    as its own category (`gate_note_parse_failures`) instead of folding it
    into either bucket."""
    try:
        rating, downgraded = pre_gate_rating(record)
        return rating, downgraded, False
    except HysteresisNoteParseError:
        return (record.get("final_decision") or {}).get("rating"), False, True


def gate_note_parse_failures(records: list[dict]) -> list[dict]:
    """Records whose `[Hysteresis gate:` note failed to parse — see
    `HysteresisNoteParseError`. Non-empty means every gate-derived metric
    below is undercounting for these specific records and should be
    investigated, the same way `isolation_violation_records` flags its own
    metrics as suspect."""
    return [r for r in records if _pre_gate_rating_safe(r)[2]]


def gate_downgrade_stats_by_language(records: list[dict]) -> dict[str, dict[str, int]]:
    """Per language: how many runs the hysteresis gate downgraded, and of
    those, how many had confidence unset (None) — the specific failure mode
    this whole investigation started from (ko_r2: model said Buy, confidence
    was unset, gate silently substituted Hold while the prose still argued
    Buy)."""
    stats: dict[str, dict[str, int]] = defaultdict(lambda: {"n_runs": 0, "downgraded": 0, "downgraded_confidence_unset": 0})
    for record in records:
        language = record["language"]
        stats[language]["n_runs"] += 1
        _, downgraded, parse_failed = _pre_gate_rating_safe(record)
        if parse_failed:
            continue
        if downgraded:
            stats[language]["downgraded"] += 1
            confidence = (record.get("final_decision") or {}).get("confidence")
            if confidence is None:
                stats[language]["downgraded_confidence_unset"] += 1
    return dict(stats)


def confidence_missing_rate_by_language(records: list[dict]) -> dict[str, float]:
    """Fraction of runs per language with `final_decision.confidence is
    None`. Promoted to a first-class metric (module task Part B3) because a
    missing confidence is exactly what flips a trade's outcome through the
    hysteresis gate — this is not a cosmetic data-quality stat."""
    by_language: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_language[record["language"]].append(record)
    return {
        language: sum(
            1 for r in group if (r.get("final_decision") or {}).get("confidence") is None
        ) / len(group)
        for language, group in by_language.items()
    }


# ---------------------------------------------------------------------------
# Run-to-run variance (noise floor)
# ---------------------------------------------------------------------------


def _group_key(record: dict) -> tuple[str, str, str]:
    return record["ticker"], record["date"], record["language"]


def _modal_agreement_fraction(ratings: list[str]) -> float | None:
    """Fraction of `ratings` equal to the group's most common ("modal")
    value — 1/R (R = repeat count) at the noisiest a group can be, 1.0 when
    every repeat agreed. Replaces the old binary `rating_disagreement`
    (any-differs) metric: that one saturates to ~100% as repeats grow (see
    module task — "Hold,Buy,Buy" and "Hold,Hold,Buy" both read as 100%
    disagreement even though the second is more stable), so scaling up
    repeats would kill the metric's signal entirely. This one distinguishes
    them: "Hold,Buy,Buy" -> 2/3, "Hold,Hold,Buy" -> 2/3 too (same ratio
    here, but "Buy,Buy,Buy" -> 1.0 and "Hold,Buy,Sell" -> 1/3 — it responds
    to HOW split the group is, not just whether it's split at all).

    Tie handling: when multiple values are tied for most-common (e.g. a
    2-repeat group "Buy,Hold", each appearing once), `collections.Counter.
    most_common` picks one arbitrarily among ties (insertion order, stable
    since Python 3.7) — but note this does NOT affect the returned number:
    every tied value has the SAME count by definition of "tied", so
    `top_count / len(ratings)` is identical no matter which tied value
    `most_common` happens to return first. The choice only matters if a
    caller wanted to know WHICH rating was modal, not the agreement
    fraction — this function only returns the fraction, so ties are a
    non-issue here (see `test_modal_agreement_fraction_tie_is_order_
    independent`).
    """
    if not ratings:
        return None
    counts = Counter(ratings)
    _, top_count = counts.most_common(1)[0]
    return top_count / len(ratings)


def variance_by_group(records: list[dict], *, pre_gate: bool = True) -> dict[tuple[str, str, str], dict[str, Any]]:
    """Per (ticker, date, language) group with 2+ repeats: the modal
    agreement fraction across repeats (see `_modal_agreement_fraction`),
    the legacy binary disagreement flag (kept for backward compat, not the
    report's primary metric anymore), and the confidence stdev across
    repeats — the noise-floor numbers a ko-vs-en gap has to clear before it
    means anything.

    `pre_gate=True` (default) uses `pre_gate_rating` — the model's own
    judgment before `_apply_hysteresis_gate` may have overridden it — since
    that is what "does the MODEL disagree with itself across repeats"
    actually asks. `pre_gate=False` uses the field as stored
    (post-gate, i.e. what was actually traded); the report prints both so
    the gate's own effect on the variance number is visible instead of
    hidden inside a single figure (module task Part B2)."""
    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for record in records:
        groups[_group_key(record)].append(record)

    result: dict[tuple[str, str, str], dict[str, Any]] = {}
    for key, group_records in groups.items():
        ratings = []
        for r in group_records:
            if not r.get("final_decision"):
                continue
            if pre_gate:
                rating, _, parse_failed = _pre_gate_rating_safe(r)
                if parse_failed:
                    continue  # excluded, not silently folded into either bucket — see gate_note_parse_failures
            else:
                rating = (r.get("final_decision") or {}).get("rating")
                if rating is None:
                    # Same exclusion the pre_gate branch already applies via
                    # parse_failed, and the same one
                    # cross_language_modal_rating_by_cell applies — a record
                    # with no usable rating must not count as a rating value
                    # (previously counted as-is, so a mix of e.g. "Buy" and
                    # None across a group's repeats read as a disagreement
                    # even when every actually-rated repeat agreed).
                    continue
            ratings.append(rating)
        confidences = [
            (r.get("final_decision") or {}).get("confidence")
            for r in group_records
            if r.get("final_decision") and (r.get("final_decision") or {}).get("confidence") is not None
        ]
        result[key] = {
            "n_repeats": len(group_records),
            # Usable-rating count, distinct from n_repeats (raw record
            # count) — a group can have 3 repeats but only 1 usable rating
            # (parse failures/missing final_decision excluded above). The
            # noise-floor gate below must key off THIS, not n_repeats, or a
            # group with only 0-1 usable ratings gets scored as if it had a
            # real 2+-way comparison (see variance_summary_by_language).
            "n_ratings": len(ratings),
            "modal_agreement": _modal_agreement_fraction(ratings),
            "rating_disagreement": len(set(ratings)) > 1 if ratings else False,  # legacy, see docstring
            "confidence_stdev": statistics.pstdev(confidences) if len(confidences) >= 2 else None,
        }
    return result


def variance_summary_by_language(records: list[dict], *, pre_gate: bool = True) -> dict[str, dict[str, float | int | None]]:
    """`variance_by_group`, rolled up per language — the "noise floor" row
    the report prints next to the ko-vs-en comparison itself. See
    `variance_by_group` for what `pre_gate` selects.

    `modal_agreement_rate` (mean of each group's modal agreement fraction,
    1.0 = fully stable) is the primary metric — see `_modal_agreement_
    fraction` for why the old `rating_disagreement_rate` saturates at scale.
    `rating_disagreement_rate` is kept for backward compat / reference only.

    `avg_confidence_stdev` is `None` (not `float("nan")`) when no group had
    2+ confidence values to compute a stdev from — e.g. every run in the
    language has `confidence is None` — so the report can print
    "n/a (no confidence values)" instead of a bare `nan`, which reads like a
    calculation bug rather than "there was nothing to compute."
    """
    by_group = variance_by_group(records, pre_gate=pre_gate)
    by_language: dict[str, list[dict]] = defaultdict(list)
    for (_, _, language), stats in by_group.items():
        # n_ratings (usable ratings), not n_repeats (raw record count) — a
        # group with e.g. 3 repeats but only 1 usable rating is a single
        # observation, not a 2+-way agreement/disagreement comparison, and
        # must not enter the noise-floor average as if it were one (see
        # variance_by_group's n_ratings comment).
        if stats["n_ratings"] >= 2:
            by_language[language].append(stats)

    summary: dict[str, dict[str, float | int | None]] = {}
    for language, group_stats in by_language.items():
        disagreement_rate = sum(1 for s in group_stats if s["rating_disagreement"]) / len(group_stats)
        modal_fractions = [s["modal_agreement"] for s in group_stats if s["modal_agreement"] is not None]
        stdevs = [s["confidence_stdev"] for s in group_stats if s["confidence_stdev"] is not None]
        summary[language] = {
            "n_groups": len(group_stats),
            "modal_agreement_rate": statistics.fmean(modal_fractions) if modal_fractions else None,
            "rating_disagreement_rate": disagreement_rate,
            "avg_confidence_stdev": statistics.fmean(stdevs) if stdevs else None,
        }
    return summary


# ---------------------------------------------------------------------------
# Cross-language comparison (does switching language change the judgment?)
# ---------------------------------------------------------------------------


def cross_language_modal_rating_by_cell(records: list[dict]) -> dict[tuple[str, str], dict[str, str]]:
    """Per (ticker, date) cell: each language's modal PRE-GATE rating across
    its repeats (languages with zero usable pre-gate ratings in that cell —
    e.g. all parse-failed — are simply absent from that cell's dict, not
    guessed at). Pre-gate, per module task spec: "게이트 이전 rating 기준으로
    계산해라" — this compares the model's own judgment across languages, not
    a pipeline artifact of the hysteresis gate."""
    cells: dict[tuple[str, str], dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for r in records:
        if not r.get("final_decision"):
            continue
        rating, _, parse_failed = _pre_gate_rating_safe(r)
        if parse_failed or rating is None:
            continue
        cells[(r["ticker"], r["date"])][r["language"]].append(rating)
    return {
        cell_key: {lang: Counter(ratings).most_common(1)[0][0] for lang, ratings in by_lang.items()}
        for cell_key, by_lang in cells.items()
    }


def cross_language_disagreement_rate(
    records: list[dict], lang_a: str = "ko", lang_b: str = "en"
) -> float | None:
    """Fraction of (ticker, date) cells where `lang_a`'s and `lang_b`'s
    modal pre-gate rating differ — the direct answer to "does switching
    language change the judgment?" (module task's own framing). Meant to sit
    next to each language's own `modal_agreement_rate` in the report: if
    this number is larger than either language's internal instability, the
    language IS the effect, not noise. Cells missing either language are
    skipped (nothing to compare). `None` if there are no comparable cells at
    all."""
    by_cell = cross_language_modal_rating_by_cell(records)
    comparable = [
        (stats[lang_a], stats[lang_b]) for stats in by_cell.values() if lang_a in stats and lang_b in stats
    ]
    if not comparable:
        return None
    return sum(1 for a, b in comparable if a != b) / len(comparable)


# ---------------------------------------------------------------------------
# Cost
# ---------------------------------------------------------------------------


def avg_elapsed_by_language(records: list[dict]) -> dict[str, float]:
    by_language: dict[str, list[float]] = defaultdict(list)
    for record in records:
        elapsed = record.get("elapsed_seconds")
        if elapsed is not None:
            by_language[record["language"]].append(elapsed)
    return {language: statistics.fmean(values) for language, values in by_language.items() if values}


# ---------------------------------------------------------------------------
# LLM call/token cost (Part A — separating retry overhead from per-call verbosity)
# ---------------------------------------------------------------------------

# Interpretation guide for the whole section below: if `ko`'s call COUNT is
# higher than `en`'s, that's retry overhead (hanja re-tries in
# app.llm.client re-call the model, each one its own metrics record — see
# app.experiments.language_ab._run_pipeline_with_retry). If instead the call
# count is roughly equal but TOKENS PER CALL is higher, that's the language
# itself being more verbose per call. These are the two competing
# explanations for the 68% ko slowdown found in the pilot, and elapsed-time
# alone cannot tell them apart — this section can.
_LLM_COST_INTERPRETATION_NOTE = (
    "Read this section as: higher CALL COUNT => retry overhead (e.g. hanja "
    "re-tries); higher TOKENS PER CALL at similar call counts => the "
    "language itself is more verbose. elapsed_seconds alone conflates the two."
)


def llm_metrics_present(records: list[dict]) -> bool:
    """False for runs recorded before Part A's instrumentation existed (e.g.
    the pilot data this was validated against) — `llm_calls` is simply
    absent from those JSON files rather than zero. The report must say
    "not instrumented," not print a misleading 0."""
    return any("llm_calls" in r for r in records)


def avg_llm_stats_by_language(records: list[dict]) -> dict[str, dict[str, float]]:
    """Per-language averages of the Part A fields, per run. Skips any record
    missing `llm_calls` (pre-instrumentation runs) rather than treating a
    missing field as 0, which would silently drag the average down."""
    by_language: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        if "llm_calls" in record:
            by_language[record["language"]].append(record)

    summary: dict[str, dict[str, float]] = {}
    for language, group in by_language.items():
        n = len(group)
        calls = [r.get("llm_calls") or 0 for r in group]
        prompt_tokens = [r.get("llm_prompt_tokens") or 0 for r in group]
        completion_tokens = [r.get("llm_completion_tokens") or 0 for r in group]
        total_tokens = [r.get("llm_total_tokens") or 0 for r in group]
        fallback_calls = [r.get("llm_fallback_calls") or 0 for r in group]
        total_calls = sum(calls)
        summary[language] = {
            "n_runs": n,
            "avg_calls": statistics.fmean(calls),
            "avg_prompt_tokens": statistics.fmean(prompt_tokens),
            "avg_completion_tokens": statistics.fmean(completion_tokens),
            "avg_total_tokens": statistics.fmean(total_tokens),
            # Guard against a language whose runs all recorded 0 calls
            # (shouldn't happen for a clean run, but a divide-by-zero here
            # must not crash the whole report over one bad record). None,
            # not float("nan") — same rule avg_confidence_stdev already
            # follows (see its docstring): a bare nan in the printed report
            # reads like a calculation bug, not "there was nothing to
            # compute", and None lets render_report print "n/a" instead.
            "avg_tokens_per_call": (sum(total_tokens) / total_calls) if total_calls else None,
            "fallback_rate": (sum(fallback_calls) / total_calls) if total_calls else None,
        }
    return summary


def top_labels_by_tokens(records: list[dict], language: str, n: int = 5) -> list[tuple[str, int]]:
    """Labels (caller identities, e.g. "market_analyst") ranked by total
    tokens summed across every `language`-arm run's `llm_by_label` — the
    costliest callers within this language, echoing `app.llm.metrics.
    summary`'s own ranking but scoped to one experiment's runs."""
    totals: dict[str, int] = defaultdict(int)
    for record in records:
        if record.get("language") != language:
            continue
        for label, stats in (record.get("llm_by_label") or {}).items():
            totals[label] += stats.get("total_tokens") or 0
    return sorted(totals.items(), key=lambda pair: pair[1], reverse=True)[:n]


# ---------------------------------------------------------------------------
# Isolation violations
# ---------------------------------------------------------------------------


def isolation_violation_records(records: list[dict]) -> list[dict]:
    return [r for r in records if r.get("isolation_violation")]


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------


def _languages(records: list[dict]) -> list[str]:
    seen: list[str] = []
    for record in records:
        if record["language"] not in seen:
            seen.append(record["language"])
    return seen


def degraded_records(records: list[dict]) -> list[dict]:
    """Runs where at least one analyst fell back to a placeholder report.

    These completed and produced a decision, so they look like clean
    successes, but they reached that decision from less data than a clean run
    did. Comparing a four-report Korean run against a two-report English one
    measures data availability, not language, so they are excluded from every
    metric below rather than silently averaged in. See
    `app.experiments.language_ab.degraded_report_names`."""
    return [r for r in records if r.get("degraded")]


def render_report(records: list[dict], errors: list[dict]) -> str:
    lines: list[str] = []
    degraded = degraded_records(records)
    analysed = [r for r in records if not r.get("degraded")]
    languages = _languages(analysed)
    by_language: dict[str, list[dict]] = defaultdict(list)
    for record in analysed:
        by_language[record["language"]].append(record)

    lines.append(
        f"{logger_prefix} {len(analysed)} analysed run(s), {len(degraded)} degraded (excluded), "
        f"{len(errors)} failed run(s)"
    )
    if degraded:
        lines.append(
            "  Degraded runs are EXCLUDED from every metric below — an analyst timed out and was "
            "replaced by a placeholder, so they saw less data than a clean run:"
        )
        by_lang_degraded: dict[str, int] = defaultdict(int)
        for record in degraded:
            by_lang_degraded[record["language"]] += 1
        for language, count in sorted(by_lang_degraded.items()):
            lines.append(f"    {language}: {count}")
        for record in degraded[:20]:
            key = _group_key(record)
            names = ", ".join(record.get("degraded_reports") or [])
            lines.append(f"    {key[0]} {key[1]} lang={key[2]} r{record.get('repeat')}: {names}")
        lines.append(
            "  If the two arms lost different numbers of runs here, the remaining comparison is "
            "also unbalanced — re-run the missing cells before drawing conclusions."
        )
    if not analysed:
        lines.append("")
        lines.append("No clean runs to analyse.")
        return "\n".join(lines)
    lines.append("")

    # --- Hanja contamination -------------------------------------------------
    lines.append("Hanja contamination (ko-only; en has no equivalent failure mode, reported as N/A):")
    for language in languages:
        rate = hanja_contamination_rate(by_language[language]) if language == "ko" else None
        display = f"{rate:.1%}" if rate is not None else "N/A"
        lines.append(f"  {language:>4}: {display}")
    lines.append("")

    # --- Self-consistency ------------------------------------------------
    # stale_prose_after_gate_override is split out on purpose (module task
    # Part B4): it is a hysteresis-gate/pipeline artifact, not a language
    # judgment defect, and folding it into the language's own count is what
    # mis-attributed ko_r2 as "ko self-consistency failure" before this fix.
    lines.append("Self-consistency violations (count / total runs; language-attributable only):")
    for language in languages:
        group = by_language[language]
        n_violations = sum(
            1 for r in group
            if any(v != "stale_prose_after_gate_override" for v in self_consistency_violations(r))
        )
        lines.append(f"  {language:>4}: {n_violations}/{len(group)}")
    lines.append(
        "  (stale_prose_after_gate_override excluded above — see next line: it's a pipeline artifact, "
        "not a language issue)"
    )
    n_stale = sum(
        1 for r in analysed if "stale_prose_after_gate_override" in self_consistency_violations(r)
    )
    lines.append(
        f"  stale_prose_after_gate_override (all languages, pipeline-caused — hysteresis gate silently "
        f"downgraded the rating but left the pre-gate prose in place): {n_stale}/{len(analysed)}"
    )
    lines.append("")

    # --- Magnitude outliers ------------------------------------------------
    lines.append("Magnitude outlier candidates (~10x prose-vs-structured mismatch; human review recommended):")
    for language in languages:
        group = by_language[language]
        flagged = [(r, magnitude_outlier_candidates(r)) for r in group]
        flagged = [(r, c) for r, c in flagged if c]
        lines.append(f"  {language:>4}: {len(flagged)}/{len(group)} run(s) flagged")
        for record, candidates in flagged[:10]:  # cap the printout — full detail is in the JSON files
            key = _group_key(record)
            for c in candidates:
                lines.append(
                    f"    {key[0]} {key[1]} r{record.get('repeat')}: prose={c['prose_number']:.2f} "
                    f"vs {c['field']}={c['structured_value']:.2f} (ratio={c['ratio']:.2f})"
                )
    lines.append("")

    # --- Run-to-run variance (noise floor) ----------------------------------
    # Pre-gate first: that's the model's own judgment, unmasked by
    # _apply_hysteresis_gate's confidence-unset downgrade (module task Part
    # B2). Post-gate follows so the gate's own effect on the number is
    # visible — the delta between the two rows IS what the gate hid.
    #
    # `modal agree%` (mean modal-agreement fraction, see
    # `_modal_agreement_fraction`) is the PRIMARY stability metric — 1.0 =
    # fully stable, closer to 1/n_repeats = maximally split. The legacy
    # `rating disagree%` (any-differs, binary) is kept alongside for
    # reference only: it saturates toward 100% as repeats scale up and
    # cannot distinguish "one outlier repeat" from "no two repeats agree" —
    # see `_modal_agreement_fraction`'s docstring for why it was replaced as
    # the headline number.
    def _fmt_stdev(value: float | None) -> str:
        return f"{value:>15.3f}" if value is not None else f"{'n/a (no confidence values)':>27}"

    lines.append("Run-to-run variance (PRE-gate rating — the model's own judgment, gate override undone):")
    variance_pre = variance_summary_by_language(analysed, pre_gate=True)
    lines.append(
        f"  {'lang':>4}  {'groups':>7}  {'modal agree%':>13}  {'rating disagree%':>17}  {'avg conf stdev':>15}"
    )
    for language in languages:
        stats = variance_pre.get(language)
        if stats is None:
            lines.append(f"  {language:>4}  {'(no repeated groups)':>7}")
            continue
        modal_rate = stats["modal_agreement_rate"]
        modal_display = f"{modal_rate:>12.1%}" if modal_rate is not None else f"{'n/a':>12}"
        lines.append(
            f"  {language:>4}  {stats['n_groups']:>7}  {modal_display}  {stats['rating_disagreement_rate']:>16.1%}  "
            f"{_fmt_stdev(stats['avg_confidence_stdev'])}"
        )
    lines.append("")
    lines.append("Run-to-run variance (POST-gate rating — what was actually traded):")
    variance_post = variance_summary_by_language(analysed, pre_gate=False)
    lines.append(
        f"  {'lang':>4}  {'groups':>7}  {'modal agree%':>13}  {'rating disagree%':>17}  {'avg conf stdev':>15}"
    )
    for language in languages:
        stats = variance_post.get(language)
        if stats is None:
            lines.append(f"  {language:>4}  {'(no repeated groups)':>7}")
            continue
        modal_rate = stats["modal_agreement_rate"]
        modal_display = f"{modal_rate:>12.1%}" if modal_rate is not None else f"{'n/a':>12}"
        lines.append(
            f"  {language:>4}  {stats['n_groups']:>7}  {modal_display}  {stats['rating_disagreement_rate']:>16.1%}  "
            f"{_fmt_stdev(stats['avg_confidence_stdev'])}"
        )
    lines.append("")

    # --- Cross-language comparison ------------------------------------------
    # Sits directly below the two per-language stability tables above on
    # purpose (module task): comparing this number against each language's
    # own modal-agreement rate is what tells "language effect" apart from
    # "decoding noise" — if this is bigger than either language's internal
    # instability, switching language actually moves the judgment.
    lines.append(
        "Cross-language disagreement (ko vs en modal PRE-gate rating per ticker/date cell — "
        "compare against modal agree% above: if this exceeds either language's own instability, "
        "the language itself is the effect, not noise):"
    )
    cross_rate = cross_language_disagreement_rate(analysed)
    if cross_rate is None:
        lines.append("  n/a (no (ticker, date) cell has both ko and en runs)")
    else:
        lines.append(f"  ko vs en: {cross_rate:.1%}")
    lines.append("")

    # --- Hysteresis gate downgrades + confidence-missing rate (Part B3) ----
    lines.append("Hysteresis gate downgrades (language attribution — confidence-unset is the failure mode found in ko_r2):")
    gate_stats = gate_downgrade_stats_by_language(analysed)
    for language in languages:
        stats = gate_stats.get(language, {"n_runs": 0, "downgraded": 0, "downgraded_confidence_unset": 0})
        lines.append(
            f"  {language:>4}: {stats['downgraded']}/{stats['n_runs']} run(s) downgraded, "
            f"{stats['downgraded_confidence_unset']} of those had confidence unset"
        )
    lines.append("")
    lines.append("Confidence-missing rate (final_decision.confidence is None — 1st-class metric, drives the gate above):")
    conf_missing = confidence_missing_rate_by_language(analysed)
    for language in languages:
        rate = conf_missing.get(language)
        lines.append(f"  {language:>4}: {rate:.1%}" if rate is not None else f"  {language:>4}: N/A")
    lines.append("")

    parse_failures = gate_note_parse_failures(analysed)
    if parse_failures:
        lines.append(
            f"WARNING: {len(parse_failures)} record(s) had a '[Hysteresis gate:' marker that failed to "
            f"parse (format may have changed in app.orchestrator.pipeline) — excluded from pre-gate "
            f"rating/variance/downgrade counts above, NOT counted as 'not downgraded':"
        )
        for record in parse_failures[:20]:
            key = _group_key(record)
            lines.append(f"    {key[0]} {key[1]} lang={key[2]} r{record.get('repeat')}")
        lines.append("")

    # --- Cost ----------------------------------------------------------------
    lines.append("Average cost (wall-clock seconds per run):")
    costs = avg_elapsed_by_language(analysed)
    for language in languages:
        cost = costs.get(language)
        lines.append(f"  {language:>4}: {cost:.1f}s" if cost is not None else f"  {language:>4}: N/A")
    lines.append("")

    # --- LLM call/token cost (Part A) ---------------------------------------
    lines.append("LLM call/token cost per run (Part A — separates retry overhead from per-call verbosity):")
    if not llm_metrics_present(analysed):
        lines.append("  not instrumented — these runs predate app.llm.metrics.drain() collection")
    else:
        lines.append(f"  {_LLM_COST_INTERPRETATION_NOTE}")
        llm_stats = avg_llm_stats_by_language(analysed)
        for language in languages:
            stats = llm_stats.get(language)
            if stats is None:
                lines.append(f"  {language:>4}: not instrumented")
                continue
            tok_per_call = "n/a" if stats["avg_tokens_per_call"] is None else f"{stats['avg_tokens_per_call']:.0f}"
            fallback_rate = "n/a" if stats["fallback_rate"] is None else f"{stats['fallback_rate']:.1%}"
            lines.append(
                f"  {language:>4}: calls={stats['avg_calls']:.1f}  "
                f"prompt_tok={stats['avg_prompt_tokens']:.0f}  completion_tok={stats['avg_completion_tokens']:.0f}  "
                f"total_tok={stats['avg_total_tokens']:.0f}  tok/call={tok_per_call}  "
                f"fallback_rate={fallback_rate}"
            )
            top_labels = top_labels_by_tokens(analysed, language)
            if top_labels:
                labels_str = ", ".join(f"{label}={tokens}" for label, tokens in top_labels)
                lines.append(f"        top labels by tokens: {labels_str}")
    lines.append("")

    # --- Isolation violations ------------------------------------------------
    violations = isolation_violation_records(records)
    lines.append(f"Isolation violations (past_context non-empty — see language_ab's module docstring): {len(violations)}")
    if violations:
        lines.append("  WARNING: every other metric above may be contaminated for these runs — investigate before trusting them:")
        for record in violations[:20]:
            key = _group_key(record)
            lines.append(f"    {key[0]} {key[1]} lang={key[2]} r{record.get('repeat')}")

    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", required=True, help="Output directory from a language_ab run.")
    return parser.parse_args()


def _main() -> None:
    args = _parse_args()
    out_dir = Path(args.dir)
    records, errors = load_records(out_dir)
    print(render_report(records, errors))


if __name__ == "__main__":
    _main()
