"""Input contract of the replay harness: stored memory row -> `ReplayInputs`.

`INPUT_CONTRACT_VERSION = "rm-start-v1"`: the replay starts at the Research Manager
with the STORED analyst reports and the STORED Bull/Bear debate, plus the stored
context strings the live pipeline fed the downstream nodes (spec A.3-A.5).
Everything that would need a live source (KIS financial recency, ground-truth
blocks) is neutralised identically in every arm ("period_note", "kr_gt", "kr_news_gt").
저장된 리포트·토론에서 RM 시작 입력을 재구성한다(라이브 전용 입력은 모든 arm에서 동일하게 중화).

Nothing here touches the DB or the network; the extract SQL is text and `fetch_rows`
takes an already-open (read-only) connection from `safety.py`.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, field, replace
from datetime import date, datetime, time as dtime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

INPUT_CONTRACT_VERSION = "rm-start-v1"

# `analyst_names` literal of `pipeline._run_pipeline_impl` (~line 1720), copied on purpose;
# `test_analyst_order_matches_pipeline_literal` fails if the original changes.
ANALYST_ORDER: tuple[str, ...] = (
    "Market Analyst", "Fundamentals Analyst", "News + Macro Analyst", "Sentiment Analyst",
)
ANALYST_ORDER_SOURCE_LITERAL = (
    'analyst_names = ["Market Analyst", "Fundamentals Analyst", "News + Macro Analyst", "Sentiment Analyst"]'
)

# `identity_note` of `pipeline._run_pipeline_impl` (~line 1960): each chunk is one source
# string literal; a test asserts every chunk is still present in the original.
IDENTITY_NOTE_CHUNKS: tuple[str, ...] = (
    "The company under analysis is ",
    ". This ",
    "ticker-to-name mapping is verified against the exchange's own registry and ",
    "overrides any other name. If a report below names a different company, that ",
    "name is wrong — the underlying data is still for this ticker. Do not dismiss ",
    "or discount a report's content because of a name mismatch, and do not repeat ",
    "the wrong name.",
)

FREEZE_START_UTC = datetime(2026, 9, 17, 3, 10, 52, tzinfo=timezone.utc)
FREEZE_START_DATE = date(2026, 9, 17)
DEV_LAST_TRADE_DATE = date(2026, 9, 16)
BULL_RATINGS = frozenset({"Buy", "Overweight"})
LANGUAGE = "en"

# Transcript keys copied into the case set (everything else in `full_transcript` is dropped).
TRANSCRIPT_KEYS: tuple[str, ...] = (
    "reports", "research_debate", "risk_debate", "research_plan", "trader_proposal",
    "investment_thesis", "kr_risk_signals", "kr_financials_recency", "session_in_progress",
    "consensus_override", "neutral_reversal", "degraded_analysts", "past_context",
    "sibling_context", "desk_consensus_note", "position_context", "volatility_context",
    "risk_cap_context", "portfolio_risk_context", "execution_status_context", "company_name",
)
# Stored context strings the downstream nodes read (missing key => ContractError).
REQUIRED_CONTEXT_KEYS: tuple[str, ...] = (
    "position_context", "volatility_context", "risk_cap_context", "past_context",
    "sibling_context", "portfolio_risk_context", "execution_status_context", "company_name",
)

EXTRACT_SQL = """
SELECT DISTINCT ON (ticker, trade_date)
  id, ticker, trade_date, created_at, rating, trader_action, confidence, held_at_decision,
  analysts_failed, decision_trigger, account_mode, is_backtest, git_commit, llm_model, user_id,
  full_transcript
FROM memory_entries
WHERE account_mode='real' AND is_backtest=false AND ticker LIKE '%.KS'
  AND decision_trigger='auto' AND full_transcript IS NOT NULL
  AND trade_date BETWEEN $1 AND $2
ORDER BY ticker, trade_date, created_at DESC
""".strip()


class ContractError(Exception):
    """A stored row cannot be turned into replay inputs (reason in the message)."""


# --------------------------------------------------------------------------- io helpers


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)


def sha256_hex(text: str | bytes) -> str:
    data = text.encode("utf-8") if isinstance(text, str) else text
    return hashlib.sha256(data).hexdigest()


def atomic_write_text(path: Path, text: str, *, retries: int = 5, delay: float = 0.2) -> None:
    """tmp -> fsync -> os.replace. Windows AV/indexers can hold a file briefly, so
    PermissionError is retried (5 x 0.2 s, spec D)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    for attempt in range(retries):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == retries - 1:
                raise
            time.sleep(delay)


# --------------------------------------------------------------------------- derived text


def identity_note_text(ticker: str, company_name: str | None) -> str:
    from app.agents.tool_loop import ticker_label

    head, mid, *rest = IDENTITY_NOTE_CHUNKS
    return f"{head}{ticker_label(ticker, company_name)}{mid}" + "".join(rest)


def derive_risk_extra(position_context: str, kr_risk_extra: str) -> str:
    """`pipeline.py` ~2140: position context, then the KRX risk block if there is one."""
    return f"{position_context}\n\n{kr_risk_extra}" if kr_risk_extra else position_context


def derive_debate_sizing_context(volatility_context: str | None, risk_cap_context: str | None) -> str:
    """`pipeline.py` ~2187: non-empty parts joined by a blank line."""
    return "\n\n".join(p for p in (volatility_context, risk_cap_context) if p)


# --------------------------------------------------------------------------- stored results

_RM_RE = re.compile(r"\*\*Recommendation\*\*:\s*(\w+)")
_TRADER_RE = re.compile(r"\*\*Action\*\*:\s*(\w+)")
_HYSTERESIS_RE = re.compile(r"\[Hysteresis gate: downgraded from (\w+) to Hold")


def parse_stored_rm(text: str | None) -> str | None:
    m = _RM_RE.search(text or "")
    return m.group(1) if m else None


def parse_stored_trader(text: str | None) -> str | None:
    m = _TRADER_RE.search(text or "")
    return m.group(1) if m else None


def parse_pre_gate(final_rating: str, consensus_override: Mapping[str, Any] | None, *texts: str | None) -> str:
    """The PM's rating before the gates: consensus gate record first, then the
    hysteresis note, else the stored final rating (spec A.3)."""
    if consensus_override and consensus_override.get("gated") and consensus_override.get("pm_rating"):
        return str(consensus_override["pm_rating"])
    for text in texts:
        m = _HYSTERESIS_RE.search(text or "")
        if m:
            return m.group(1)
    return final_rating


def parse_neutral_final(risk_turns: Iterable[Mapping[str, Any]]) -> str | None:
    """Verdict of the last Neutral turn (`risk_debators._neutral_verdict`)."""
    from app.agents.risk_debators import _neutral_verdict

    last = None
    for turn in risk_turns:
        if turn.get("speaker") == "Neutral":
            last = turn
    return _neutral_verdict(str(last.get("content", "")))[0] if last else None


@dataclass(frozen=True)
class StoredResults:
    rm: str | None
    trader: str | None
    neutral_final: str | None
    pm_pre_gate: str | None
    final: str | None
    confidence: float | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "rm": self.rm, "trader": self.trader, "neutral_final": self.neutral_final,
            "pm_pre_gate": self.pm_pre_gate, "final": self.final, "confidence": self.confidence,
        }


# --------------------------------------------------------------------------- inputs


@dataclass(frozen=True)
class ReplayInputs:
    case_id: str
    memory_id: int
    ticker: str
    trade_date: date
    created_at: datetime
    era: str
    split: str
    company_name: str
    reports: tuple[tuple[str, str], ...]
    research_turns: tuple[tuple[str, str, int], ...]
    identity_note: str
    position_context: str
    volatility_context: str
    risk_cap_context: str
    past_context: str
    sibling_context: str
    portfolio_risk_context: str
    execution_status_context: str
    kr_risk_extra: str
    kr_risk_neutralized: bool
    risk_extra: str
    debate_sizing_context: str
    stored: StoredResults
    stored_session: bool | None
    interim: bool | None
    llm_model: str | None = None
    git_commit: str | None = None
    session_note: str = ""
    session_meta: dict[str, Any] = field(default_factory=dict)
    held: bool = False
    period_note: str = ""
    kr_gt: str = ""
    kr_news_gt: str = ""
    degraded_analysts: tuple[str, ...] = ()
    language: str = LANGUAGE
    contract_checks: dict[str, Any] = field(default_factory=dict)

    # --- fresh, mutable objects for each node call (nodes may mutate `turns`) ---
    def reports_dict(self) -> dict[str, str]:
        return dict(self.reports)

    def research_debate(self):
        from app.state import DebateState, DebateTurn

        turns = [DebateTurn(speaker=s, content=c, round=r) for s, c, r in self.research_turns]
        return DebateState(
            parties=["Bull", "Bear"], turns=turns,
            rounds_completed=max((r for _, _, r in self.research_turns), default=0),
        )

    def input_hash(self) -> str:
        payload = {
            "reports": [[name, text] for name, text in self.reports],
            "research_debate": [[s, c, r] for s, c, r in self.research_turns],
            "held": self.held,
            "identity_note": self.identity_note,
            "session_note": self.session_note,
            "period_note": self.period_note,
            "position_context": self.position_context,
            "volatility_context": self.volatility_context,
            "risk_cap_context": self.risk_cap_context,
            "past_context": self.past_context,
            "sibling_context": self.sibling_context,
            "portfolio_risk_context": self.portfolio_risk_context,
            "execution_status_context": self.execution_status_context,
            "company_name": self.company_name,
            "kr_risk_extra": self.kr_risk_extra,
            "risk_extra": self.risk_extra,
            "debate_sizing_context": self.debate_sizing_context,
            "degraded_analysts": list(self.degraded_analysts),
            "kr_gt": self.kr_gt,
            "kr_news_gt": self.kr_news_gt,
            "language": self.language,
        }
        return sha256_hex(canonical_json(payload))

    def covariates(self) -> dict[str, Any]:
        return {
            "interim": self.interim,
            "kr_risk_neutralized": self.kr_risk_neutralized,
            "session_meta": dict(self.session_meta),
            "period_note_neutralized": True,
            "kr_gt_neutralized": True,
        }

    def with_session(self, note: str, meta: dict[str, Any]) -> "ReplayInputs":
        return replace(self, session_note=note, session_meta=dict(meta))


def era_of(created_at: datetime, trade_date: date) -> str:
    if created_at >= FREEZE_START_UTC:
        return "freeze"
    if trade_date == FREEZE_START_DATE:
        return "mixed_0917"
    return "pre_freeze"


def split_of(trade_date: date) -> str:
    return "dev" if trade_date <= DEV_LAST_TRADE_DATE else "holdout"


def make_case_id(ticker: str, trade_date: date, memory_id: int) -> str:
    # No colons: the id is a file name on Windows.
    return f"{ticker}_{trade_date:%Y%m%d}_m{memory_id}"


def _as_dict(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return value
    if isinstance(value, (str, bytes)):
        try:
            loaded = json.loads(value)
        except ValueError:
            return None
        return loaded if isinstance(loaded, dict) else None
    return None


def _as_datetime(value: Any) -> datetime:
    parsed = value if isinstance(value, datetime) else datetime.fromisoformat(str(value))
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)  # timestamptz is aware


def _as_date(value: Any) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value))


def record_from_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """A memory_entries row (asyncpg Record or dict) -> JSON-able case-set record,
    keeping only the transcript keys the contract needs."""
    transcript = _as_dict(row["full_transcript"])
    if transcript is None:
        raise ContractError("full_transcript is not a JSON object")
    created_at = _as_datetime(row["created_at"])
    trade_date = _as_date(row["trade_date"])
    return {
        "case_id": make_case_id(row["ticker"], trade_date, int(row["id"])),
        "memory_id": int(row["id"]),
        "ticker": row["ticker"],
        "trade_date": trade_date.isoformat(),
        "created_at": created_at.isoformat(),
        "era": era_of(created_at, trade_date),
        "split": split_of(trade_date),
        "columns": {
            "rating": row["rating"],
            "trader_action": row["trader_action"],
            "confidence": None if row["confidence"] is None else float(row["confidence"]),
            "held_at_decision": row["held_at_decision"],
            "analysts_failed": row["analysts_failed"],
            "decision_trigger": row["decision_trigger"],
            "account_mode": row["account_mode"],
            "is_backtest": row["is_backtest"],
            "git_commit": row["git_commit"],
            "llm_model": row["llm_model"],
        },
        "transcript": {k: transcript[k] for k in TRANSCRIPT_KEYS if k in transcript},
    }


def _need_str(transcript: Mapping[str, Any], key: str, *, allow_none: bool = False) -> str:
    if key not in transcript:
        raise ContractError(f"missing transcript key: {key}")
    value = transcript[key]
    if value is None and allow_none:
        return ""
    if not isinstance(value, str):
        raise ContractError(f"transcript key {key} is not a string")
    return value


def _turns(transcript: Mapping[str, Any], key: str, parties: tuple[str, ...], expected: int | None):
    raw = transcript.get(key)
    if not isinstance(raw, list) or not raw:
        raise ContractError(f"missing or empty {key}")
    turns = []
    for item in raw:
        if not isinstance(item, dict) or item.get("speaker") not in parties:
            raise ContractError(f"{key}: bad turn {item!r:.80}")
        content, rnd = item.get("content"), item.get("round")
        if not isinstance(content, str) or not isinstance(rnd, int):
            raise ContractError(f"{key}: bad turn content/round")
        turns.append((item["speaker"], content, rnd))
    if expected is not None and len(turns) != expected:
        raise ContractError(f"{key}: {len(turns)} turns, expected {expected}")
    return tuple(turns)


def compute_desk_note_match(inputs: ReplayInputs, stored_note: str | None) -> bool | None:
    """Recompute `pipeline._desk_consensus_note` from the STORED RM/Trader + stored
    KRX/sizing context and compare with the stored note (AA-1g). None if not testable."""
    if stored_note is None or inputs.stored.rm is None or inputs.stored.trader is None:
        return None
    from app.orchestrator import pipeline
    from app.state import PortfolioRating, ResearchPlan, TraderAction, TraderProposal

    try:
        plan = ResearchPlan(
            recommendation=PortfolioRating(inputs.stored.rm), rationale="x", strategic_actions="x"
        )
        proposal = TraderProposal(action=TraderAction(inputs.stored.trader), reasoning="x")
    except ValueError:
        return None
    note = pipeline._desk_consensus_note(plan, proposal, inputs.kr_risk_extra, inputs.debate_sizing_context)
    return note == stored_note


def build_inputs(
    record: Mapping[str, Any],
    *,
    max_debate_rounds: int | None = 2,
    max_risk_rounds: int | None = 2,
    check_desk_note: bool = True,
) -> ReplayInputs:
    """Case-set record -> `ReplayInputs` (raises `ContractError` with the reason)."""
    columns, transcript = record["columns"], record["transcript"]
    ticker = record["ticker"]
    trade_date = _as_date(record["trade_date"])
    created_at = _as_datetime(record["created_at"])
    if columns.get("held_at_decision") is not False:
        raise ContractError("held_at_decision is not False")

    reports_raw = transcript.get("reports")
    if not isinstance(reports_raw, dict) or not all(isinstance(v, str) for v in reports_raw.values()):
        raise ContractError("reports missing or not a name->text object")
    if set(reports_raw) != set(ANALYST_ORDER):
        raise ContractError(f"report keys {sorted(reports_raw)} != analyst set")
    reports = tuple((name, reports_raw[name]) for name in ANALYST_ORDER)

    research_turns = _turns(
        transcript, "research_debate", ("Bull", "Bear"),
        None if max_debate_rounds is None else max_debate_rounds * 2,
    )
    if max_risk_rounds is not None:
        _turns(transcript, "risk_debate", ("Aggressive", "Conservative", "Neutral"), max_risk_rounds * 3)

    ctx = {key: _need_str(transcript, key, allow_none=(key == "risk_cap_context")) for key in REQUIRED_CONTEXT_KEYS}
    if "kr_risk_signals" in transcript and transcript["kr_risk_signals"] is not None:
        kr_risk_extra = transcript["kr_risk_signals"]
        if not isinstance(kr_risk_extra, str):
            raise ContractError("kr_risk_signals is not a string")
        neutralized = False
    else:
        kr_risk_extra, neutralized = "", True

    recency = transcript.get("kr_financials_recency")
    interim = recency.get("interim") if isinstance(recency, dict) else None
    session_stored = transcript.get("session_in_progress")
    if session_stored is not None and not isinstance(session_stored, bool):
        raise ContractError("session_in_progress is not a bool")

    stored_rm = parse_stored_rm(transcript.get("research_plan"))
    stored_trader = parse_stored_trader(transcript.get("trader_proposal"))
    final = columns.get("rating")
    stored = StoredResults(
        rm=stored_rm,
        trader=stored_trader or columns.get("trader_action"),
        neutral_final=parse_neutral_final(transcript.get("risk_debate") or []),
        pm_pre_gate=parse_pre_gate(
            final, transcript.get("consensus_override"), transcript.get("investment_thesis")
        ) if final else None,
        final=final,
        confidence=columns.get("confidence"),
    )
    inputs = ReplayInputs(
        case_id=record["case_id"], memory_id=int(record["memory_id"]), ticker=ticker,
        trade_date=trade_date, created_at=created_at, era=record["era"], split=record["split"],
        company_name=ctx["company_name"], reports=reports, research_turns=research_turns,
        identity_note=identity_note_text(ticker, ctx["company_name"]),
        position_context=ctx["position_context"], volatility_context=ctx["volatility_context"],
        risk_cap_context=ctx["risk_cap_context"], past_context=ctx["past_context"],
        sibling_context=ctx["sibling_context"], portfolio_risk_context=ctx["portfolio_risk_context"],
        execution_status_context=ctx["execution_status_context"],
        kr_risk_extra=kr_risk_extra, kr_risk_neutralized=neutralized,
        risk_extra=derive_risk_extra(ctx["position_context"], kr_risk_extra),
        debate_sizing_context=derive_debate_sizing_context(ctx["volatility_context"], ctx["risk_cap_context"]),
        stored=stored, stored_session=session_stored, interim=interim if isinstance(interim, bool) else None,
        llm_model=columns.get("llm_model"), git_commit=columns.get("git_commit"),
    )
    checks: dict[str, Any] = {"report_keys_ok": True}
    checks["desk_note_stored_match"] = (
        compute_desk_note_match(inputs, transcript.get("desk_consensus_note"))
        if check_desk_note and record["era"] == "freeze" else None
    )
    return replace(inputs, contract_checks=checks)


# --------------------------------------------------------------------------- selection


@dataclass
class Selection:
    cases: list[dict[str, Any]]
    excluded: list[dict[str, Any]]
    counts: dict[str, Any]


def select_cases(
    rows: Iterable[Mapping[str, Any]],
    *,
    max_debate_rounds: int | None = 2,
    max_risk_rounds: int | None = 2,
    era: str | None = None,
    min_trade_date: date | None = None,
    max_trade_date: date | None = None,
) -> Selection:
    """Spec A.4 filter order: dedup (latest created_at per ticker/date) -> analysts_failed==0
    -> held_at_decision False -> contract check. Exclusions carry a reason and are counted."""
    rows = list(rows)
    excluded: list[dict[str, Any]] = []
    counts: dict[str, Any] = {"input_rows": len(rows)}

    latest: dict[tuple[str, date], Mapping[str, Any]] = {}
    for row in rows:
        key = (row["ticker"], _as_date(row["trade_date"]))
        cur = latest.get(key)
        if cur is None or _as_datetime(row["created_at"]) > _as_datetime(cur["created_at"]):
            latest[key] = row
    for row in rows:
        key = (row["ticker"], _as_date(row["trade_date"]))
        if latest[key] is not row:
            excluded.append({"memory_id": row["id"], "reason": "dedup_older_duplicate"})
    deduped = sorted(latest.values(), key=lambda r: (_as_date(r["trade_date"]), r["ticker"]))
    counts["after_dedup"] = len(deduped)

    def _keep(step: str, rows_in, predicate, reason):
        kept = []
        for row in rows_in:
            if predicate(row):
                kept.append(row)
            else:
                excluded.append({"memory_id": row["id"], "reason": reason(row)})
        counts[step] = len(kept)
        return kept

    kept = _keep("after_af0", deduped, lambda r: r["analysts_failed"] == 0,
                 lambda r: "analysts_failed_unknown" if r["analysts_failed"] is None else "analysts_failed_nonzero")
    kept = _keep("after_unheld", kept, lambda r: r["held_at_decision"] is False,
                 lambda r: "held_at_decision_null" if r["held_at_decision"] is None else "held_at_decision_true")
    kept = _keep("after_date_range", kept,
                 lambda r: (min_trade_date is None or _as_date(r["trade_date"]) >= min_trade_date)
                 and (max_trade_date is None or _as_date(r["trade_date"]) <= max_trade_date),
                 lambda r: "trade_date_out_of_range")

    cases: list[dict[str, Any]] = []
    for row in kept:
        try:
            record = record_from_row(row)
            if era is not None and record["era"] != era:
                excluded.append({"memory_id": row["id"], "reason": f"era_{record['era']}"})
                continue
            build_inputs(
                record, max_debate_rounds=max_debate_rounds, max_risk_rounds=max_risk_rounds,
                check_desk_note=False,
            )
        except ContractError as exc:
            excluded.append({"memory_id": row["id"], "reason": f"contract: {exc}"})
            continue
        cases.append(record)
    counts["after_contract"] = len(cases)
    reasons: dict[str, int] = {}
    for item in excluded:
        reasons[item["reason"]] = reasons.get(item["reason"], 0) + 1
    counts["excluded_by_reason"] = dict(sorted(reasons.items()))
    return Selection(cases=cases, excluded=excluded, counts=counts)


async def fetch_rows(conn: Any, min_trade_date: date, max_trade_date: date) -> list[Mapping[str, Any]]:
    """Run `EXTRACT_SQL` on an already-open READ-ONLY connection."""
    return [dict(r) for r in await conn.fetch(EXTRACT_SQL, min_trade_date, max_trade_date)]


# --------------------------------------------------------------------------- case-set io


def write_caseset(directory: Path, name: str, records: list[dict[str, Any]], meta: dict[str, Any]) -> tuple[Path, Path]:
    """`{name}.jsonl` (one record per line) + `{name}.meta.json` incl. the jsonl sha256."""
    body = "".join(canonical_json(rec) + "\n" for rec in records)
    jsonl = directory / f"{name}.jsonl"
    meta_path = directory / f"{name}.meta.json"
    atomic_write_text(jsonl, body)
    full_meta = dict(meta)
    full_meta.update({"name": name, "n_cases": len(records), "jsonl_sha256": sha256_hex(body)})
    atomic_write_text(meta_path, json.dumps(full_meta, indent=2, ensure_ascii=False, default=str) + "\n")
    return jsonl, meta_path


def load_caseset(directory: Path, name: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    jsonl = directory / f"{name}.jsonl"
    meta_path = directory / f"{name}.meta.json"
    data = jsonl.read_bytes()
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("jsonl_sha256") != sha256_hex(data):
        raise ContractError(f"case set {name}: jsonl sha256 does not match its meta file")
    records = [json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()]
    return records, meta


# --------------------------------------------------------------------------- session prepass


def session_prepass(cases: list[ReplayInputs]) -> list[ReplayInputs]:
    """Serial pre-step (spec A.5), BEFORE any async work: the session note is derived
    with a FakeClock set to each case's `created_at`, the stored bool wins on a mismatch.
    The process-wide clock is always restored (metrics timestamps depend on it)."""
    from app.clock import KR_TZ, FakeClock, LiveClock, get_clock, set_clock
    from app.dataflows.session_state import intraday_narration_note

    prev = get_clock()
    out: list[ReplayInputs] = []
    try:
        for case in cases:
            if case.created_at.tzinfo is None:
                raise ValueError(f"{case.case_id}: created_at must be timezone-aware")
            set_clock(FakeClock(case.created_at))
            fc_note = intraday_narration_note(case.ticker, case.trade_date)
            if case.stored_session is None:
                note, meta = fc_note, {"source": "fakeclock", "agree": None}
            elif bool(fc_note) == case.stored_session:
                note, meta = fc_note, {"source": "stored", "agree": True}
            elif case.stored_session:  # stored True but FakeClock says closed (boundary time)
                set_clock(FakeClock(datetime.combine(case.trade_date, dtime(12, 0), KR_TZ)))
                note = intraday_narration_note(case.ticker, case.trade_date)
                meta = {"source": "stored", "agree": False}
            else:
                note, meta = "", {"source": "stored", "agree": False}
            out.append(case.with_session(note, meta))
    finally:
        set_clock(prev)
    if not isinstance(get_clock(), LiveClock):
        raise RuntimeError("process clock is not a LiveClock after the session pre-pass")
    return out
