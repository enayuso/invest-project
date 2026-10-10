"""Risk-debate seat consistency instrument (2026-09-21): does each seat
(Aggressive / Conservative / Neutral) argue a stable position across its own
rounds, or does the adversarial structure collapse?

Trigger: the 09-21 domestic decision audit (`handoff/knowledge/
domestic-ai-decision-audit-2026-09-21.md`) found the desk's Buy(100%)
consensus vanish in the risk debate on two decisions (id=1624, 1627) via two
different failure shapes that nothing in the pipeline measures:
  - id=1624: the Aggressive seat's own round-2 turn abandoned Buy and argued
    Hold, attacking its own round-1 turn in the third person ("The Aggressive
    ... debaters have constructed a cohesive narrative ... This argument is
    seductive but fatally flawed"). Round 2 ended with zero seats defending
    the desk's Buy.
  - id=1622: risk_conservative's round-2 turn was a 97.4% `difflib` duplicate
    of its own round-1 turn (two sentences changed, verbatim otherwise) —
    round 2 contributed no new argument for the judging Neutral seat to weigh.
`app.agents.risk_debators.check_neutral_reversal` already records whether the
NEUTRAL seat's final verdict differs from its round-1 verdict, but nothing
records whether AGGRESSIVE or CONSERVATIVE stayed in character, and nothing
measures round-over-round textual duplication for any seat. This module is
that measurement — read-only, over already-stored `full_transcript` rows,
computing no new production field and changing no report, prompt, or gate.
It is a `behavior-change-freeze-2026-09-17` category-4 instrument (pure
measurement/recording), not a rating- or action-affecting change.

Verdict extraction reuses `app.agents.risk_debators._neutral_verdict` (the
same regex, tuned over 516 live turns per that module's own comment) rather
than re-deriving a second copy that would silently drift from it — the
function parses "the last verdict-shaped span in a turn's text" and is not
actually specific to the Neutral seat despite its name; it takes free text.

Public API:
  `compute_seat_stance` -> SeatStanceReport
  `render_seat_stance`  -> str

Row selection reuses the same `market`/`scope`/`account_mode`/`user_id`
filter semantics as `app.audit.rating_shift.compute_rating_shift` — see that
module's docstring for the shared rationale, not repeated here.

`_NEAR_DUPLICATE_RATIO` is provisional: as of 2026-09-21 there is exactly ONE
confirmed positive (id=1622's risk_conservative, ratio=0.974) against a
handful of negatives in the 0.03-0.19 range from the same day's other
decisions. 0.85 sits well clear of both, but this has not been validated
against a real distribution — treat any single run's duplicate count as a
lead to inspect, not a calibrated rate. The renderer always prints raw
ratios, never just the flag, so a reader isn't stuck trusting the threshold.

Read-only: no INSERT/UPDATE/DELETE anywhere in this module."""

from __future__ import annotations

import difflib
from dataclasses import dataclass, field
from datetime import date

from app.agents.risk_debators import _neutral_verdict as _extract_verdict
from app.audit.guard_alpha import _parse_transcript
from app.audit.rating_shift import _extract_rm_recommendation
from app.memory.db import get_pool
from app.user_context import get_current_user_id

_SEATS = ("Aggressive", "Conservative", "Neutral")
_BULLISH_LEVELS = frozenset({"Buy", "Overweight"})
_NEAR_DUPLICATE_RATIO = 0.85


@dataclass(frozen=True)
class SeatRoundStance:
    round: int
    stance: str | None
    span: str
    chars: int


@dataclass(frozen=True)
class SeatRecord:
    speaker: str
    rounds: list[SeatRoundStance]
    r1_final_similarity: float | None  # None when the seat has <2 rounds
    near_duplicate: bool
    softened_from_bullish: bool | None  # None when r1 or final stance didn't parse

    @property
    def r1(self) -> SeatRoundStance | None:
        return self.rounds[0] if self.rounds else None

    @property
    def final(self) -> SeatRoundStance | None:
        return self.rounds[-1] if self.rounds else None


@dataclass(frozen=True)
class SeatStanceDecision:
    id: int
    ticker: str
    trade_date: date
    rm_recommendation: str | None
    trader_action: str | None
    final_rating: str | None
    desk_buy: bool  # RM and Trader both proposed a bullish level before the risk debate
    seats: dict[str, SeatRecord]
    final_round_buy_defenders: int  # seats whose LAST round stance is bullish

    @property
    def aggressive_softened(self) -> bool | None:
        rec = self.seats.get("Aggressive")
        return rec.softened_from_bullish if rec else None

    @property
    def has_near_duplicate_round(self) -> bool:
        return any(rec.near_duplicate for rec in self.seats.values())


@dataclass(frozen=True)
class SeatStanceReport:
    market: str | None
    scope: str | None
    account_mode: str | None
    n_scanned: int  # rows with a non-empty risk_debate
    n_no_debate: int  # rows fetched but with an empty/missing risk_debate
    decisions: list[SeatStanceDecision] = field(default_factory=list)

    @property
    def n_desk_buy(self) -> int:
        return sum(1 for d in self.decisions if d.desk_buy)

    @property
    def n_desk_buy_zero_defenders(self) -> int:
        return sum(1 for d in self.decisions if d.desk_buy and d.final_round_buy_defenders == 0)

    @property
    def n_near_duplicate_rounds(self) -> int:
        return sum(1 for d in self.decisions if d.has_near_duplicate_round)

    @property
    def n_aggressive_softened(self) -> int:
        return sum(1 for d in self.decisions if d.aggressive_softened)


def _build_seat_record(speaker: str, turns: list[dict]) -> SeatRecord:
    """`turns` are this speaker's entries from `full_transcript['risk_debate']`
    (each `{"speaker", "content", "round"}`), already filtered to `speaker`
    and sorted by round ascending. Mirrors `check_neutral_reversal`'s
    first-round/final-round comparison, generalized to any seat."""
    rounds = []
    for t in turns:
        stance, span = _extract_verdict(t["content"])
        rounds.append(SeatRoundStance(round=t["round"], stance=stance, span=span, chars=len(t["content"])))
    if len(rounds) < 2:
        return SeatRecord(speaker=speaker, rounds=rounds, r1_final_similarity=None,
                           near_duplicate=False, softened_from_bullish=None)

    r1_content = turns[0]["content"]
    final_content = turns[-1]["content"]
    similarity = difflib.SequenceMatcher(a=r1_content, b=final_content).ratio()
    near_duplicate = similarity >= _NEAR_DUPLICATE_RATIO

    r1_stance, final_stance = rounds[0].stance, rounds[-1].stance
    if r1_stance is None or final_stance is None:
        softened = None
    else:
        softened = r1_stance in _BULLISH_LEVELS and final_stance not in _BULLISH_LEVELS

    return SeatRecord(speaker=speaker, rounds=rounds, r1_final_similarity=similarity,
                       near_duplicate=near_duplicate, softened_from_bullish=softened)


def _build_decision(row) -> SeatStanceDecision:
    transcript = _parse_transcript(row["full_transcript"])
    risk_debate: list[dict] = transcript.get("risk_debate") or []
    rm_recommendation, _ = _extract_rm_recommendation(transcript)
    trader_action = row["trader_action"]

    seats: dict[str, SeatRecord] = {}
    for speaker in _SEATS:
        turns = sorted((t for t in risk_debate if t.get("speaker") == speaker), key=lambda t: t["round"])
        if turns:
            seats[speaker] = _build_seat_record(speaker, turns)

    final_round_buy_defenders = sum(
        1 for rec in seats.values() if rec.final is not None and rec.final.stance in _BULLISH_LEVELS
    )
    desk_buy = (rm_recommendation in _BULLISH_LEVELS) and (trader_action in _BULLISH_LEVELS)

    return SeatStanceDecision(
        id=row["id"], ticker=row["ticker"], trade_date=row["trade_date"],
        rm_recommendation=rm_recommendation, trader_action=trader_action, final_rating=row["rating"],
        desk_buy=desk_buy, seats=seats, final_round_buy_defenders=final_round_buy_defenders,
    )


async def compute_seat_stance(
    *, ticker: str | None = None, market: str | None = None, scope: str | None = None,
    account_mode: str | None = None, since: date | None = None, until: date | None = None,
) -> SeatStanceReport:
    """Scans `memory_entries` for rows with a non-empty `risk_debate` and
    reports per-seat round-consistency for each. `market`/`scope`/
    `account_mode` follow `compute_rating_shift`'s exact semantics. `since`/
    `until` filter on `trade_date` (inclusive), matching `screening_ic`'s
    convention rather than `guard_alpha`'s `horizon_eval_date` one — this
    audit doesn't depend on forward-return maturation, so trade_date is the
    natural filter and avoids excluding today's still-unresolved rows."""
    if ticker and market:
        raise ValueError("ticker and market are mutually exclusive")
    if market not in (None, "domestic", "overseas"):
        raise ValueError(f"market must be 'domestic', 'overseas', or None, got {market!r}")
    if scope not in (None, "live", "backtest"):
        raise ValueError(f"scope must be 'live', 'backtest', or None, got {scope!r}")
    if account_mode not in (None, "paper", "real"):
        raise ValueError(f"account_mode must be 'paper', 'real', or None, got {account_mode!r}")

    conditions = []
    params: list = []
    if ticker:
        params.append(ticker)
        conditions.append(f"ticker = ${len(params)}")
    elif market == "domestic":
        conditions.append("ticker LIKE '%.KS'")
    elif market == "overseas":
        conditions.append("ticker NOT LIKE '%.KS'")
    if scope == "live":
        conditions.append("is_backtest = false")
    elif scope == "backtest":
        conditions.append("is_backtest = true")
    if account_mode is not None:
        params.append(account_mode)
        conditions.append(f"account_mode = ${len(params)}")
    if since is not None:
        params.append(since)
        conditions.append(f"trade_date >= ${len(params)}")
    if until is not None:
        params.append(until)
        conditions.append(f"trade_date <= ${len(params)}")
    params.append(get_current_user_id())
    conditions.append(f"user_id = ${len(params)}")
    where_clause = " AND ".join(conditions)

    query = f"""
        SELECT id, ticker, trade_date, rating, trader_action, full_transcript
        FROM memory_entries
        WHERE {where_clause}
        ORDER BY created_at
    """

    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(query, *params)

    decisions: list[SeatStanceDecision] = []
    n_no_debate = 0
    for row in rows:
        transcript = _parse_transcript(row["full_transcript"])
        if not transcript.get("risk_debate"):
            n_no_debate += 1
            continue
        decisions.append(_build_decision(row))

    return SeatStanceReport(
        market=market, scope=scope, account_mode=account_mode,
        n_scanned=len(decisions), n_no_debate=n_no_debate, decisions=decisions,
    )


def render_seat_stance(report: SeatStanceReport) -> str:
    lines = [
        f"Risk-debate seat consistency  filters: market={report.market or '(both)'}  "
        f"scope={report.scope or '(both)'}  account_mode={report.account_mode or '(both)'}",
        f"n_scanned={report.n_scanned}  n_no_risk_debate={report.n_no_debate}"
        f"  n_desk_buy={report.n_desk_buy}",
        "",
    ]

    flagged = [
        d for d in report.decisions
        if (d.desk_buy and d.final_round_buy_defenders == 0) or d.has_near_duplicate_round or d.aggressive_softened
    ]
    if not flagged:
        lines.append("No flagged decisions (desk-Buy-with-zero-defenders / near-duplicate round / "
                      "Aggressive-softened) in this window.")
    else:
        lines.append(f"=== {len(flagged)} flagged decision(s) ===")
        for d in flagged:
            lines.append(f"\nid={d.id}  {d.ticker}  {d.trade_date}  "
                         f"RM={d.rm_recommendation}  Trader={d.trader_action}  final_rating={d.final_rating}")
            if d.desk_buy and d.final_round_buy_defenders == 0:
                lines.append(f"  DESK-BUY-VANISHED: RM+Trader both bullish, "
                             f"but 0/{len(d.seats)} seats defend a bullish stance in their final round")
            for speaker, rec in d.seats.items():
                r1, final = rec.r1, rec.final
                r1_desc = f"{r1.stance!r}" if r1 else "n/a"
                final_desc = f"{final.stance!r}" if final and final is not r1 else "(same turn)"
                sim_desc = f"  round1<->final similarity={rec.r1_final_similarity:.3f}" \
                    if rec.r1_final_similarity is not None else ""
                flags = []
                if rec.near_duplicate:
                    flags.append(f"NEAR-DUPLICATE (>={_NEAR_DUPLICATE_RATIO})")
                if rec.softened_from_bullish:
                    flags.append("SOFTENED-FROM-BULLISH")
                flag_desc = f"  [{', '.join(flags)}]" if flags else ""
                lines.append(f"    {speaker:<13} r1={r1_desc:<14} final={final_desc:<14}"
                             f"{sim_desc}{flag_desc}")

    lines += [
        "",
        f"Summary: desk-Buy rows={report.n_desk_buy}, of which vanished-to-zero-defenders="
        f"{report.n_desk_buy_zero_defenders}; near-duplicate rounds in "
        f"{report.n_near_duplicate_rounds} decision(s); Aggressive softened from bullish in "
        f"{report.n_aggressive_softened} decision(s).",
        "",
        f"Note: near-duplicate threshold ({_NEAR_DUPLICATE_RATIO}) is provisional (n=1 confirmed "
        "positive as of 2026-09-21, see module docstring) — read the printed ratios, not just the "
        "flag. This module records observations only; it renders no verdict and gates nothing.",
    ]
    return "\n".join(lines)
