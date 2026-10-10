"""Reflection call — adapts TradingAgents' own reflection prompt
(`tradingagents/graph/reflection.py`): 2-4 terse sentences covering whether
the directional call was right, which part of the thesis held or failed,
and one concrete lesson. Kept terse deliberately since this text gets
re-injected into future prompts verbatim via `app/memory/store.py`.
"""

from __future__ import annotations

import logging

from app.llm import llm_client

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a trading analyst reviewing your own past decision now that the "
    "outcome is known. Write exactly 2-4 sentences of plain prose (no bullets, "
    "no headers, no markdown).\n\n"
    "Cover in order:\n"
    "1. Was the directional call correct? (cite the alpha figure)\n"
    "2. Which part of the investment thesis held or failed?\n"
    "3. One concrete lesson to apply to the next similar analysis.\n\n"
    "Be specific and terse. Your output will be stored verbatim in a decision "
    "log and re-read by future analysts, so every word must earn its place."
)


def _win_direction_note(rating: str | None, held_at_decision: bool | None, benchmark_ticker: str) -> str:
    """One plain-English sentence telling the model which outcome means the
    call was actually correct, so it doesn't default to "positive alpha =
    good" for every rating — that default is backwards for a Sell (a Sell
    is correct when the ticker UNDERPERFORMS the benchmark, i.e. NEGATIVE
    alpha: the thing we exited beat the benchmark means we should have kept
    holding) and doesn't apply at all to a Hold, whose correct direction
    depends on `held_at_decision` (a held position's real alternative is
    selling into cash — measured by raw_return, not alpha — while a flat
    Hold's real alternative is the benchmark itself).

    Mirrors `app.audit.stats._WIN_DIRECTION`'s exact mapping (Buy/Overweight
    win on alpha>0, Sell/Underweight win on alpha<0, Hold(held) wins on
    raw_return>0, Hold(flat) wins on alpha<0) but replicated locally rather
    than imported — app.audit and app.memory shouldn't cross-import over a
    2-line lesson-writing prompt (app.memory is a lower layer app.audit
    itself sits above).

    Returns "" (no note inserted into the prompt) for anything this can't
    confidently classify — `rating is None`, an unrecognized rating string,
    or `rating == "Hold"` with `held_at_decision is None` — rather than
    guessing a direction with no evidence behind it."""
    if rating in ("Buy", "Overweight"):
        return f"this call is correct if alpha_return is positive (the ticker beat {benchmark_ticker})"
    if rating in ("Sell", "Underweight"):
        return (
            "correct if alpha_return is negative (the ticker underperforming "
            f"{benchmark_ticker} means avoiding/reducing it was right)"
        )
    if rating == "Hold" and held_at_decision is True:
        return "correct if raw_return is positive (a held position's real alternative is selling into cash, not the benchmark)"
    if rating == "Hold" and held_at_decision is False:
        return (
            "correct if alpha_return is negative (staying in cash was right if the ticker "
            f"underperformed {benchmark_ticker})"
        )
    return ""


async def reflect(
    decision_text: str,
    raw_return: float,
    alpha_return: float,
    benchmark_ticker: str = "SPY",
    *,
    rating: str | None = None,
    held_at_decision: bool | None = None,
) -> str:
    """`rating`/`held_at_decision` (both optional, default `None`): let the
    prompt tell the model which outcome direction actually means the call
    was correct — see `_win_direction_note`'s own docstring for why, without
    this, the model likely defaults to "positive alpha = good" for every
    rating, which is backwards for a Sell and undefined for a Hold. `None`
    (a caller that hasn't been updated, or genuinely doesn't know) omits the
    note entirely rather than guessing."""
    note = _win_direction_note(rating, held_at_decision, benchmark_ticker)
    prompt = (
        f"Raw return: {raw_return:+.1%}\n"
        f"Alpha vs {benchmark_ticker}: {alpha_return:+.1%}\n"
        + (f"Note: {note}.\n" if note else "")
        + f"\nFinal Decision:\n{decision_text}"
    )
    response = await llm_client.complete(
        [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
        reasoning=False,
        label="reflection",
    )
    content = response.content
    if not content or not content.strip():
        logger.warning(
            "reflect: LLM returned empty/None content (rating=%s) — returning empty string rather than storing it",
            rating,
        )
        return ""
    return content
