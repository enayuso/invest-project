"""Research Manager `research_plan` meta/CoT leak audit (detector v3).

The live RM output (`memory_entries.full_transcript->'research_plan'`) was
found to leak the model's own meta-commentary / chain-of-thought into the
rationale (2026-10-01 investigation, re-found 2026-10-02). This module turns
the detector that previously lived only as prose regexes and a throwaway
scratchpad script into a durable, re-runnable READ-ONLY audit:

    python -m app.audit rm-leak --since 2026-09-01 --until 2026-10-02
    python -m app.audit rm-leak --since 2026-10-02 --after-ts 2026-10-02T10:18:07+09:00 --show 5

Detector v3 = v1 start patterns + v2 start patterns + broad markers + HTML
start. A row is contaminated when ANY kind fires; per-kind counts overlap.
Source of truth for the definitions:
handoff/knowledge/rm-research-plan-leak-investigation-2026-10-01.md
(the "10-02 재집계 -- 검출기 v2" section and the section-1 footnotes) and
domestic-ai-decision-audit-2026-10-02.md (id 1821, the HTML variant v2 missed).

`research_plan` is a rendered markdown STRING on live rows but may be a dict
(see `app.audit.rating_shift`); both shapes are handled. No pass/fail
verdict is encoded -- numbers only. Read-only: SELECT only, no writes."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import date, datetime

from app.audit.guard_alpha import _parse_transcript
from app.memory.db import get_pool
from app.user_context import get_current_user_id

# v1 start patterns -- investigation doc section 1, footnote 1 ("meta start").
V1_START_RE = re.compile(
    r"^(The user|The model|The response|The synthesis|Synthesiz|Integrated multi|"
    r"This (response|memo|synthesis|analysis)|Here's|Okay|Thinking|Let me|I will|"
    r"I need|We need|<\||사용자)"
)

# v2 new start patterns -- "10-02 재집계 -- 검출기 v2" section. 0 hits on 542 rows of 09-01~09-30.
# "The debate centers on" is deliberately NOT here (normal rationale, 11 rows).
V2_START_RE = re.compile(
    r"^(The (analysis|request|task|prompt|input|provided|assignment|instruction)s?\b|"
    r"Analy[sz]es? (all|the provided)|Evaluat(es|ing) the provided|"
    r"Based on the (provided|four)|To (synthesize|evaluate|produce))"
)

# Broad markers over the whole research_plan -- section 1 footnote 2, plus the
# two v2 additions ("the response must", "must integrate all").
BROAD_RE = re.compile(
    r"did not pose an explicit question|I will structure|Thinking Process|</?think>|"
    r"<\|[a-z_]+\|>|Self-Correction|Drafting the|the response must|must integrate all"
)

# HTML start -- v3 addition: audit doc 2026-10-02, id 1821 (rationale began
# with `<i><b>INTERIM-WARNING INTEGRATION:</b></i>`), which v2 missed.
HTML_START_RE = re.compile(r"^<(i|b|em|strong|p|div|span|br)\b", re.IGNORECASE)

KINDS = ("v1", "v2", "broad", "html")
_RATIONALE_MARK = "**Rationale**:"
_HOLDING_MARK = "Currently holding"


def _plan_text_and_rationale(research_plan) -> tuple[str | None, str | None]:
    """Returns `(full_text, rationale)`; both None when the plan is absent or
    of an unusable type. Rationale = text after `**Rationale**:` (whole text
    when the marker is missing; the `rationale` field for a dict plan)."""
    if isinstance(research_plan, dict):
        text = json.dumps(research_plan, ensure_ascii=False)
        value = research_plan.get("rationale")
        if isinstance(value, str) and value.strip():
            return text, value
        return text, text
    if isinstance(research_plan, str):
        if not research_plan.strip():
            return None, None
        idx = research_plan.find(_RATIONALE_MARK)
        rationale = research_plan[idx + len(_RATIONALE_MARK):] if idx >= 0 else research_plan
        return research_plan, rationale
    return None, None


def classify_research_plan(research_plan) -> tuple[str, ...]:
    """Contamination kinds (subset of KINDS, in KINDS order) that fire on a
    `research_plan`; empty tuple = clean (or no plan)."""
    text, rationale = _plan_text_and_rationale(research_plan)
    if text is None:
        return ()
    start = (rationale or "").lstrip()
    kinds = []
    if V1_START_RE.search(start):
        kinds.append("v1")
    if V2_START_RE.search(start):
        kinds.append("v2")
    if BROAD_RE.search(text):
        kinds.append("broad")
    if HTML_START_RE.search(start):
        kinds.append("html")
    return tuple(kinds)


def has_research_plan(research_plan) -> bool:
    return _plan_text_and_rationale(research_plan)[0] is not None


def is_holding(transcript: dict) -> bool:
    pc = transcript.get("position_context")
    return isinstance(pc, str) and _HOLDING_MARK in pc


@dataclass
class DayLeak:
    n: int = 0
    contaminated: int = 0
    kinds: dict[str, int] = field(default_factory=lambda: {k: 0 for k in KINDS})

    def add(self, kinds: tuple[str, ...]) -> None:
        self.n += 1
        if kinds:
            self.contaminated += 1
        for k in kinds:
            self.kinds[k] += 1


@dataclass
class LeakReport:
    since: date
    until: date | None
    market: str
    scope: str
    account_mode: str | None
    after_ts: datetime | None
    days: dict[date, DayLeak]
    total: DayLeak
    after_total: DayLeak | None
    holding: dict[str, DayLeak]  # "holding" / "not_holding" over all rows
    n_no_plan: int
    shown: list[dict]  # first `show` contaminated rows, with `head` text
    contaminated_ids: list[tuple] = field(default_factory=list)  # (id, trade_date, kinds)


def build_report(rows, *, since, until, market, scope, account_mode, after_ts, show: int = 0) -> LeakReport:
    days: dict[date, DayLeak] = {}
    total = DayLeak()
    after_total = DayLeak() if after_ts is not None else None
    holding = {"holding": DayLeak(), "not_holding": DayLeak()}
    n_no_plan = 0
    shown: list[dict] = []
    ids: list[tuple] = []
    for row in rows:
        transcript = _parse_transcript(row["full_transcript"])
        plan = transcript.get("research_plan")
        if not has_research_plan(plan):
            n_no_plan += 1
            continue
        kinds = classify_research_plan(plan)
        days.setdefault(row["trade_date"], DayLeak()).add(kinds)
        total.add(kinds)
        holding["holding" if is_holding(transcript) else "not_holding"].add(kinds)
        if after_total is not None and row["created_at"] >= after_ts:
            after_total.add(kinds)
        if kinds:
            ids.append((row["id"], row["trade_date"], kinds))
            if len(shown) < show:
                _, rationale = _plan_text_and_rationale(plan)
                shown.append({
                    "id": row["id"], "ticker": row["ticker"], "trade_date": row["trade_date"],
                    "kinds": kinds, "head": " ".join((rationale or "").split())[:80],
                })
    return LeakReport(
        since=since, until=until, market=market, scope=scope, account_mode=account_mode,
        after_ts=after_ts, days=dict(sorted(days.items())), total=total, after_total=after_total,
        holding=holding, n_no_plan=n_no_plan, shown=shown, contaminated_ids=ids,
    )


async def compute_rm_leak(
    since: date, *, until: date | None = None, market: str = "domestic", scope: str = "live",
    account_mode: str | None = None, after_ts: datetime | None = None, show: int = 0,
) -> LeakReport:
    if market not in ("domestic", "overseas"):
        raise ValueError(f"market must be 'domestic' or 'overseas', got {market!r}")
    if scope not in ("live", "backtest"):
        raise ValueError(f"scope must be 'live' or 'backtest', got {scope!r}")
    if account_mode not in (None, "paper", "real"):
        raise ValueError(f"account_mode must be 'paper', 'real', or None, got {account_mode!r}")
    if after_ts is not None and after_ts.tzinfo is None:
        raise ValueError("after_ts must be tz-aware")

    # Domestic = KRX tickers. KOSDAQ names are stored as '.KS' in practice;
    # '.KQ' is included defensively.
    krx = "(ticker LIKE '%.KS' OR ticker LIKE '%.KQ')"
    conditions = [krx if market == "domestic" else f"NOT {krx}"]
    params: list = []
    conditions.append("is_backtest = false" if scope == "live" else "is_backtest = true")
    if account_mode is not None:
        params.append(account_mode)
        conditions.append(f"account_mode = ${len(params)}")
    params.append(since)
    conditions.append(f"trade_date >= ${len(params)}")
    if until is not None:
        params.append(until)
        conditions.append(f"trade_date <= ${len(params)}")
    params.append(get_current_user_id())
    conditions.append(f"user_id = ${len(params)}")
    query = f"""
        SELECT id, ticker, trade_date, created_at, full_transcript
        FROM memory_entries
        WHERE {' AND '.join(conditions)}
        ORDER BY created_at
    """
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(query, *params)
    return build_report(
        rows, since=since, until=until, market=market, scope=scope, account_mode=account_mode,
        after_ts=after_ts, show=show,
    )


def _row_line(label: str, d: DayLeak) -> str:
    pct = (d.contaminated / d.n * 100) if d.n else 0.0
    kinds = "  ".join(f"{d.kinds[k]:>4}" for k in KINDS)
    return f"  {label:<12} {d.n:>5} {d.contaminated:>6} {pct:5.1f}%  {kinds}"


def render_rm_leak(report: LeakReport) -> str:
    lines = [
        f"RM research_plan leak audit (detector v3)  since={report.since}  until={report.until or '(none)'}",
        f"filters: market={report.market}  scope={report.scope}  "
        f"account_mode={report.account_mode or '(both)'}  after_ts="
        f"{report.after_ts.isoformat() if report.after_ts else '(none)'}",
        "",
        f"  {'date':<12} {'n':>5} {'leak':>6} {'rate':>6}  " + "  ".join(f"{k:>4}" for k in KINDS),
    ]
    for d, day in report.days.items():
        lines.append(_row_line(str(d), day))
    lines.append(_row_line("TOTAL", report.total))
    if report.after_total is not None:
        lines.append(_row_line("after-ts", report.after_total))
        before = DayLeak()
        before.n = report.total.n - report.after_total.n
        before.contaminated = report.total.contaminated - report.after_total.contaminated
        before.kinds = {k: report.total.kinds[k] - report.after_total.kinds[k] for k in KINDS}
        lines.append(_row_line("before-ts", before))
    lines.append("")
    lines.append("By position (all rows in range):")
    lines.append(_row_line("holding", report.holding["holding"]))
    lines.append(_row_line("not_holding", report.holding["not_holding"]))
    if report.n_no_plan:
        lines.append(f"\n(rows without a usable research_plan, excluded: {report.n_no_plan})")
    if report.shown:
        lines.append("\nContaminated rows (rationale head, <=80 chars):")
        for s in report.shown:
            lines.append(f"  id={s['id']} {s['ticker']} {s['trade_date']} [{','.join(s['kinds'])}] {s['head']}")
    lines.append(
        "\nNote: kind columns overlap (a row can fire several); `leak` counts rows once. "
        "Encodes no pass/fail threshold."
    )
    return "\n".join(lines)


def report_to_json(report: LeakReport) -> str:
    def day(d: DayLeak) -> dict:
        return {"n": d.n, "contaminated": d.contaminated, "kinds": dict(d.kinds)}

    payload = {
        "since": report.since.isoformat(),
        "until": report.until.isoformat() if report.until else None,
        "market": report.market, "scope": report.scope, "account_mode": report.account_mode,
        "after_ts": report.after_ts.isoformat() if report.after_ts else None,
        "days": {d.isoformat(): day(v) for d, v in report.days.items()},
        "total": day(report.total),
        "after_total": day(report.after_total) if report.after_total is not None else None,
        "holding": {k: day(v) for k, v in report.holding.items()},
        "n_no_plan": report.n_no_plan,
        "contaminated": [
            {"id": i, "trade_date": d.isoformat(), "kinds": list(k)} for i, d, k in report.contaminated_ids
        ],
        "shown": [
            {**s, "trade_date": s["trade_date"].isoformat(), "kinds": list(s["kinds"])} for s in report.shown
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)
