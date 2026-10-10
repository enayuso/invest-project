"""Catches a report that turned a Korean interim-vs-full-year period
mismatch into a fabricated "decline" / "collapse".

2026-09-01, decision id=1158 (000270.KS/기아): every agent from the
Fundamentals Analyst on divided H1 2026's half-year operating profit
(₩48,337억) by FY2024's annual figure (₩126,671억), got -61.8%, and wrote
"three-period sequential decline" / "62% collapse". Annualized, H1 2026 was
+6.5% vs FY2025. KIS's own YoY growth for that interim period (-10.79% net
income) was supplied and even quoted correctly elsewhere in the same
report; the model computed its own decline from the raw rows anyway.

The source-side fixes (2026-09-01 audit #5 §2/§3 — the inline growth rows,
the named-row PERIOD WARNING, and the downstream `interim_period_note`) are
the primary defense. This is the detection net for a report that made the
mistake anyway, the same relationship `app.agents.unit_scale_guard` has to
`_eok`.

Detect-only, on purpose: unlike a unit-scale error there is no single
"correct value" to swap in — the whole sentence's thesis is wrong, not one
number — so rewriting it is not a regex's job. It logs and records.

The check runs in REVERSE: rather than a fragile "decline word near two
numbers" regex (the model usually writes only the RESULT percentage, not
the two operands), it pre-computes every value the FORBIDDEN division could
produce — `interim_item / fy_item - 1` for every (item, FY row) pair — and
looks for a percentage in the report that matches one of them AND is framed
as a decline AND is NOT one of the exchange's own real YoY growth figures.

2026-09-03 addendum (precision restructure + Tier-1/Tier-2 split). The
detector's FP classes were each tied to a precision rule and removed:
  ① `_FORBIDDEN_BAND` — only a NEGATIVE ratio in (-90, -25) is a candidate
     (a positive ratio is a growth story; below -90 is more likely a real
     crash). Plus unsigned matching when decline-framed, so "62% collapse"
     (no sign) still matches -61.8 (id=1158's actual phrasing).
  ② item coherence — the forbidden division's item name must be within
     `_ITEM_WINDOW` chars of the `%`, and not itself a "... margin/ratio".
  ③ rounding-aware real-growth exemption — a report that rounds KIS's real
     `revenue_growth=-70.16%` to "-70%" is exempt (id=1185).
  ④ precision-aware match tolerance — "-62%" (rounded) gets ±0.5, "-63.31%"
     (computed) gets ±0.15.
  ⑤ context-exclusion regexes (margin/ROE/ratio level, intraday price move,
     annualized/CAGR hedge) scanned in the same narrow window.
  ⑥ scenario / range-tail exclusion — a "50-62%" tail or a "could extend"
     projection is not a stated decline.
  ⑦ `_eok` placeholder rows (`gross_profit=1억원`) are skipped.
  ⑧ findings deduped on (span text, forbidden desc) within one call.

Tier-1 (strong) findings — passes ①-⑧ AND from the Fundamentals Analyst AND
period-framed (`_PERIOD_FRAME_RE`) — gate the report cache in
`app.orchestrator.pipeline`. Tier-2 (everything else that passed) is
record-only. This mirrors `unit_scale_guard`'s Tier-1 autocorrect / Tier-2
flag-only split. `PeriodMismatchScan` carries the positive signal an empty
finding list cannot (see `unit_scale_guard.UnitScaleScan` / id=1158).

Notes (2026-09-03):
  - `_ITEM_WINDOW` (80) is fitted to ONE sample (id=1193's "Net margins" 90
    chars past the gross-margin list). `PeriodMismatchScan.suppressed` is
    the data to retune it against.
  - `_FORBIDDEN_BAND`'s -90 lower bound: a real -95% collapse is
    deliberately missed — that band is more likely a genuine crash than a
    period mis-division.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# A labeled income-statement row from `get_kr_financials`'s output, e.g.
#   "  202606 (interim, 6 months — NOT a full year): revenue=625,389억원(...) ..."
_INC_ROW_RE = re.compile(
    r"^\s*(\d{6})\s*\((interim|FY)[^)]*\):\s*(.+)$", re.MULTILINE | re.IGNORECASE
)
_ITEMS = ("revenue", "gross_profit", "operating_profit", "net_income")
_ITEM_VAL_RE = {
    item: re.compile(rf"\b{item}=\s*(-?[\d,]+(?:\.\d+)?)\s*억원") for item in _ITEMS
}
# The exchange's own YoY growth figures, wherever they appear in kr_block —
# a report percentage matching one of these is KIS's real number, never a
# mis-division, and must never be flagged.
_REAL_GROWTH_RE = re.compile(
    r"(?:revenue|operating_profit|net_income|op_profit)_growth=\s*(-?[\d.]+)\s*%", re.IGNORECASE
)
# Any signed percentage in the report under scrutiny.
_PCT_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s*%")
_DECLINE_WORDS_RE = re.compile(
    r"collaps|plunge|declin|drop|\bfell\b|\bfall\b|deterior|contract|shrink|eros|worsen|"
    r"급감|붕괴|감소|악화|nosedive|tank",
    re.IGNORECASE,
)

# Rule ① — only a forbidden ratio inside this NEGATIVE band is a candidate.
# A positive ratio is a growth story, never a fabricated collapse; below -90
# is more likely a genuine crash than a period mis-division. Kia -61.8 /
# Woori -63.3 are mid-band.
_FORBIDDEN_BAND = (-90.0, -25.0)

# Rule ④ — precision-aware match tolerance (retires the old flat ±1.0).
_MATCH_TOL_ROUNDED = 0.5   # integer notation ("-62%") — a rounded value
_MATCH_TOL_PRECISE = 0.15  # >= 1 decimal ("-18.39%") — a computed value

_REAL_GROWTH_TOL = 0.05    # ±%p — a report figure this close to a real KIS growth is exempt
_FRAMING_WINDOW = 140      # ± chars scanned for a decline word / a period frame
# ± chars scanned for the item name (rule ②) and the context regexes (rule ⑤).
# Narrower than the framing window. Fitted to ONE sample (id=1193's "Net
# margins" 90 chars after the gross-margin list) — `PeriodMismatchScan.
# suppressed` is the data to retune it against.
_ITEM_WINDOW = 80

# Rule ② — the forbidden division's ITEM NAME must appear near the `%`.
_ITEM_SYNONYMS = {
    "revenue":          r"revenue|sales|top[- ]line|turnover|매출액|매출",
    "gross_profit":     r"gross profit|매출총이익",
    "operating_profit": r"operating (?:profit|income|earnings)|영업이익",
    "net_income":       r"net (?:income|profit|earnings)|bottom line|순이익|당기순이익",
}
# A matched item name immediately followed by this is a LEVEL/ratio, not the
# item itself ("net margin" must NOT satisfy "net_income").
_LEVEL_SUFFIX = r"\s*(?:margin|ratio|rate|yield|률|율)"
_ITEM_RE = {
    item: re.compile(rf"(?:{syn})(?!{_LEVEL_SUFFIX})", re.IGNORECASE)
    for item, syn in _ITEM_SYNONYMS.items()
}

# Rule ⑤ — context that means the `%` is not an interim/FY division at all.
# Scanned in the same ±_ITEM_WINDOW window as rule ②.
_LEVEL_CTX_RE = re.compile(
    r"(?:gross|operating|net|ebitda|ebit|profit)\s*margin|"
    r"margin\s*(?:of|was|is|:)|\broe\b|\broa\b|\broic\b|current ratio|quick ratio|"
    r"debt[- ]to[- ]equity|\bd/e\b|payout|dependency|이익률|마진", re.I,
)
_PRICE_CTX_RE = re.compile(
    r"intraday|intra-session|session|drawdown|stop[- ]loss|\bATR\b|"
    r"share price|stock (?:price|fell|dropped)|\d[\d,]*\s*(?:KRW|원|₩)|₩\s?\d|"
    r"high of|low of|rally|rallied|bounce|closed? at", re.I,
)
_HEDGE_CTX_RE = re.compile(
    r"annually|annualiz|annualis|per annum|\bp\.a\.|\bCAGR\b|"
    r"annual rate|annual .{0,12}erosion", re.I,
)

# Rule ⑥ — a `%` pulled out of a "30-40%" range tail, or a scenario projection.
# `_RANGE_TAIL_RE` catches the tilde / en-dash forms where the dash is left in
# `text[:m.start()]`; `_RANGE_HYPHEN_RE` catches "50-62%", where `_PCT_RE`'s
# optional sign consumes the hyphen so the tail ends on a bare digit (a
# trailing space rules out a legitimate "Q3 -62%").
_RANGE_TAIL_RE = re.compile(r"\d+(?:\.\d+)?\s*[-–~]\s*$")   # applied to text[:m.start()]
_RANGE_HYPHEN_RE = re.compile(r"\d$")
_SCENARIO_CTX_RE = re.compile(
    r"could|can extend|may |might |if support|scenario|worst[- ]case|risk of", re.I,
)

# Rule ⑦ — a "gross_profit=1억원" placeholder row is not a real value.
_PLACEHOLDER_ABS = 1.0

# §3 — a Tier-1 (cache-gating) finding must ALSO carry an explicit period
# frame. Scanned in the ±_FRAMING_WINDOW window.
_PERIOD_FRAME_RE = re.compile(
    r"\bFY\b|full[- ]year|20\d\d|peak|\bH[12]\b|\b[12]H\b|half[- ]year|"
    r"six months|6 months|interim|annualiz|반기|연간", re.I,
)


def _round_tol(token: str) -> float:
    """Half a ULP of the report's own written precision: "-70" -> 0.5,
    "-70.2" -> 0.05, "-70.16" -> 0.005 (rule ③)."""
    return 0.5 * 10 ** -len(token.split(".")[1]) if "." in token else 0.5


def _to_float(token: str) -> float:
    return float(token.replace(",", ""))


def _parse_rows(kr_block: str) -> tuple[dict | None, list[dict]]:
    """(latest interim row, [FY rows]) — each a dict of {item: 억원 value}
    plus 'period'. `None` interim when there's no interim row."""
    interim: dict | None = None
    fy: list[dict] = []
    for m in _INC_ROW_RE.finditer(kr_block):
        period, kind, rest = m.group(1), m.group(2).lower(), m.group(3)
        row = {"period": period}
        for item, pat in _ITEM_VAL_RE.items():
            im = pat.search(rest)
            if im:
                row[item] = _to_float(im.group(1))
        if kind == "interim":
            if interim is None or period > interim["period"]:
                interim = row
        else:
            fy.append(row)
    return interim, fy


@dataclass(frozen=True)
class PeriodMismatchScan:
    """The POSITIVE signal an empty finding list cannot carry — same role
    `unit_scale_guard.UnitScaleScan` plays for that guard (see id=1158)."""

    node: str
    kr_block_present: bool
    interim_period: str        # "" = no interim row
    fy_rows: int
    forbidden_in_band: int     # candidates surviving _FORBIDDEN_BAND — 0 => report was unscannable
    pct_examined: int
    tol_matched: int           # numerically matched but then failed a context rule
    strong: int
    weak: int
    suppressed: dict           # {"level_ctx": 1, "price_ctx": 6, "item_mismatch": 2, ...}

    def as_dict(self) -> dict:
        """Flatten `suppressed` into `suppressed_<key>` entries so the whole
        record is a flat dict for the transcript — mirrors how
        `unit_scale_guard`'s scan is stored in `state.unit_scale_scans`."""
        d: dict = {
            "kr_block_present": int(self.kr_block_present),
            "interim_period": self.interim_period,
            "fy_rows": self.fy_rows,
            "forbidden_in_band": self.forbidden_in_band,
            "pct_examined": self.pct_examined,
            "tol_matched": self.tol_matched,
            "strong": self.strong,
            "weak": self.weak,
        }
        for k, v in self.suppressed.items():
            d[f"suppressed_{k}"] = v
        return d


def scan_period_mismatch(
    text: str,
    kr_block: str,
    *,
    agent_name: str = "",
    strong_out: list[str] | None = None,
    scan_out: list[PeriodMismatchScan] | None = None,
) -> list[str]:
    """`"period_mismatch: '<span>' matches ..."` per report percentage that
    looks like a forbidden interim/FY division framed as a decline. Empty
    when there's no interim row, no FY rows, no `kr_block`, or nothing
    matches — an empty list here is NOT proof the report is clean (same
    caveat as `unit_scale_guard`'s scan record / decision id=1158).

    Return signature (`list[str]` of ALL findings) and finding-string format
    are unchanged, so existing callers/fixtures are unaffected.

    `strong_out`, if given, is extended with only the Tier-1 (strong)
    findings — passed ①-⑧ AND `agent_name == "Fundamentals Analyst"` AND a
    `_PERIOD_FRAME_RE` hit in the framing window. Only this subset gates the
    report cache (see `app.orchestrator.pipeline`). `scan_out`, if given, is
    appended with this call's `PeriodMismatchScan` (always — even on an early
    return, so "clean" is distinguishable from "unscannable")."""

    def _record(**kw) -> None:
        if scan_out is None:
            return
        base: dict = dict(
            node=agent_name, kr_block_present=bool(kr_block), interim_period="",
            fy_rows=0, forbidden_in_band=0, pct_examined=0, tol_matched=0,
            strong=0, weak=0, suppressed={},
        )
        base.update(kw)
        scan_out.append(PeriodMismatchScan(**base))

    if not text or not kr_block:
        _record()
        return []

    interim, fy_rows = _parse_rows(kr_block)
    if interim is None or not fy_rows:
        _record(interim_period=(interim or {}).get("period", ""), fy_rows=len(fy_rows))
        return []

    real_growths = [float(m.group(1)) for m in _REAL_GROWTH_RE.finditer(kr_block)]

    # every value the forbidden division could produce, filtered to the band
    forbidden: list[tuple[float, str, str]] = []
    lo, hi = _FORBIDDEN_BAND
    for item in _ITEMS:
        iv = interim.get(item)
        if iv is None or abs(iv) <= _PLACEHOLDER_ABS:  # rule ⑦
            continue
        for fyr in fy_rows:
            fv = fyr.get(item)
            if fv is None or abs(fv) <= _PLACEHOLDER_ABS:  # rule ⑦
                continue
            ratio = (iv / fv - 1.0) * 100.0
            if lo <= ratio <= hi:  # rule ①
                forbidden.append((
                    ratio,
                    f"interim {interim['period']} / FY {fyr['period']} {item} ({ratio:+.1f}%)",
                    item,
                ))
    if not forbidden:
        _record(interim_period=interim["period"], fy_rows=len(fy_rows))
        return []

    suppressed: dict[str, int] = {}

    def _bump(key: str) -> None:
        suppressed[key] = suppressed.get(key, 0) + 1

    found: list[str] = []
    strong_found: list[str] = []
    seen: set[tuple[str, str]] = set()
    pct_examined = 0
    tol_matched = 0

    for m in _PCT_RE.finditer(text):
        pct_examined += 1
        token = m.group(1)
        v = float(token)
        window = text[max(0, m.start() - _FRAMING_WINDOW) : m.end() + _FRAMING_WINDOW]
        declined = v < 0 or _DECLINE_WORDS_RE.search(window) is not None

        # rule ③ — rounding-aware real-growth exemption
        if any(abs(v - g) <= max(_REAL_GROWTH_TOL, _round_tol(token)) for g in real_growths):
            _bump("real_growth_rounded")
            continue

        # rule ④ — precision-aware tolerance; rule ① — unsigned when decline-framed
        tol = _MATCH_TOL_PRECISE if "." in token else _MATCH_TOL_ROUNDED
        hit = next(
            (f for f in forbidden
             if abs(v - f[0]) <= tol or (declined and abs(abs(v) - abs(f[0])) <= tol)),
            None,
        )
        if hit is None:
            continue
        _ratio, desc, item = hit

        nwin = text[max(0, m.start() - _ITEM_WINDOW) : m.end() + _ITEM_WINDOW]
        pre = text[: m.start()]

        # rule ⑥ — range tail / scenario
        if _RANGE_TAIL_RE.search(pre) or (token.startswith("-") and _RANGE_HYPHEN_RE.search(pre)):
            tol_matched += 1
            _bump("range_tail")
            continue
        if _SCENARIO_CTX_RE.search(nwin):
            tol_matched += 1
            _bump("scenario_ctx")
            continue
        # rule ⑤ — context exclusion
        if _LEVEL_CTX_RE.search(nwin):
            tol_matched += 1
            _bump("level_ctx")
            continue
        if _PRICE_CTX_RE.search(nwin):
            tol_matched += 1
            _bump("price_ctx")
            continue
        if _HEDGE_CTX_RE.search(nwin):
            tol_matched += 1
            _bump("hedge_ctx")
            continue
        # rule ② — item coherence
        if not _ITEM_RE[item].search(nwin):
            tol_matched += 1
            _bump("item_mismatch")
            continue
        if not declined:
            continue

        # rule ⑧ — dedupe on (span text, forbidden desc)
        key = (m.group(0), desc)
        if key in seen:
            _bump("dedupe")
            continue
        seen.add(key)

        finding = f"period_mismatch: {m.group(0)!r} matches {desc}"
        found.append(finding)
        if agent_name == "Fundamentals Analyst" and _PERIOD_FRAME_RE.search(window):
            strong_found.append(finding)

    if strong_out is not None:
        strong_out.extend(strong_found)

    _record(
        interim_period=interim["period"], fy_rows=len(fy_rows),
        forbidden_in_band=len(forbidden), pct_examined=pct_examined,
        tol_matched=tol_matched, strong=len(strong_found),
        weak=len(found) - len(strong_found), suppressed=suppressed,
    )

    if found:
        logger.warning(
            "period_mismatch_guard: %s report percentage(s) look like an interim-vs-full-year "
            "mis-division framed as a decline (see decision id=1158, 2026-09-01): %s (strong=%d)",
            agent_name or "?", found, len(strong_found),
        )
    return found
