"""Retroactive dry-run of the proposed tiered-analysis pre-screen (⑦,
[[ai-engine-24h-operation-analysis]]) over decisions the live pipeline has
already made — answers "if this screen had been running, what would it have
skipped, and would that have been safe?" WITHOUT touching the live pipeline,
execution paths, or any DB write. Read-only, LLM-free (same two data sources
the real screen would use: `memory_entries`/`trades` history and
`app.dataflows.kis_provider`'s point-in-time price/ATR — the same
KIS-first/yfinance-fallback lookup the live pipeline and backtests already
use via `as_of`).

Design doc: this session's ⑦ pre-screen spec (2026-08-13, Opus). Tier 0
(forced full analysis) is a whitelist of "must not skip" conditions; Tier 1
(skip) requires ALL of a narrower set to hold. Five Tier-0 conditions from
that spec are NOT modeled here — volume spike, circuit-breaker/VI history,
earnings-date proximity, portfolio-regime change (Risk-Off transition), and
consecutive-skip-count reset semantics beyond a simple counter — because they
need data/infra this throwaway validator doesn't have. That means this
script's skip-rate is an UPPER BOUND (real deployment would skip less, not
more) and its safety check is correspondingly the one number that matters:
whether ANY sell-side action would have been missed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

from app.dataflows.kis_provider import get_atr_pct, get_last_close
from app.memory.db import get_pool
from app.user_context import get_current_user_id

# ---------------------------------------------------------------------------
# Thresholds — all DRAFT values from the design doc, not yet validated.
# This script's whole purpose is to tell you whether these are safe; expect
# to tune them from its output, not treat them as settled.
# ---------------------------------------------------------------------------
MIN_CONSECUTIVE_HOLDS = 2
PRICE_MOVE_ATR_MULTIPLE = 1.5
STOP_LOSS_TIER0_PCT = 0.05  # existing held_review-style "near stop" proximity
# 2026-08-13 Opus review: originally a 3x-safe-distance escape hatch let held
# positions into Tier 1 anyway — a real Sell (064260.KS, 2026-07-29) fell
# through it, breaking the design doc's "held positions are never screened"
# principle. A price/ATR-based screen fundamentally cannot predict a
# news/fundamentals-driven exit, so no distance threshold is actually safe
# for a held position — Tier 1 now excludes any current holding outright.
STALENESS_LIMIT_DAYS = 15  # was 5 = same as the rebalance throttle, so every revisit auto-forced Tier0
MAX_CONSECUTIVE_SKIPS = 3

_RATING_TO_DIRECTION = {"Buy": "buy", "Overweight": "buy", "Hold": None, "Underweight": "sell", "Sell": "sell"}


@dataclass
class MissedAction:
    ticker: str
    trade_date: date
    rating: str
    action_label: str | None
    direction: str  # "buy" or "sell"


@dataclass
class ScreenValidationResult:
    total_decisions: int = 0
    tier0_count: int = 0
    tier1_count: int = 0
    tier0_reason_counts: dict[str, int] = field(default_factory=dict)
    missed_actions: list[MissedAction] = field(default_factory=list)
    price_lookup_failures: list[tuple[str, date]] = field(default_factory=list)


async def _fetch_decisions(start_date: date, account_mode: str, user_id: str) -> list[dict]:
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            WITH latest AS (
                SELECT DISTINCT ON (ticker, trade_date)
                    ticker, trade_date, rating, action_label, trader_action
                FROM memory_entries
                WHERE is_backtest = false AND account_mode = $1 AND user_id = $2
                  AND trade_date >= $3
                ORDER BY ticker, trade_date, created_at DESC
            )
            SELECT * FROM latest ORDER BY ticker, trade_date
            """,
            account_mode, user_id, start_date,
        )
    return [dict(r) for r in rows]


async def _fetch_trades(start_date: date, account_mode: str, user_id: str) -> dict[str, list[dict]]:
    """Per-ticker chronological trade history (for reconstructing held-shares
    and the stop-loss in effect on any given date). `start_date` is widened
    30 days back so a decision early in the validation window can still see
    a position opened just before it."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT ticker, trade_date, shares_after, stop_loss, created_at
            FROM trades
            WHERE is_backtest = false AND account_mode = $1 AND user_id = $2
              AND trade_date >= $3
            ORDER BY ticker, trade_date, created_at
            """,
            account_mode, user_id, start_date - timedelta(days=30),
        )
    by_ticker: dict[str, list[dict]] = {}
    for r in rows:
        by_ticker.setdefault(r["ticker"], []).append(dict(r))
    return by_ticker


async def _fetch_position_seed(start_date: date, account_mode: str, user_id: str) -> dict[str, dict]:
    """Last known (shares_after, stop_loss) per ticker as of just before the
    30-day-widened trade-history window `_fetch_trades` queries — without
    this, a position opened more than 30 days before `start_date` and never
    traded since reads as shares_held == 0 in `_held_state_before`'s empty-
    history default, which is currently the ONLY remaining Tier-1
    (skip-eligible) gate for a stable ticker: a real Sell on a long-held,
    quiet position could silently undercount this script's headline safety
    number ("would any sell-side action have been missed")."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT DISTINCT ON (ticker) ticker, shares_after, stop_loss
            FROM trades
            WHERE is_backtest = false AND account_mode = $1 AND user_id = $2
              AND trade_date < $3
            ORDER BY ticker, trade_date DESC, created_at DESC
            """,
            account_mode, user_id, start_date - timedelta(days=30),
        )
    return {r["ticker"]: {"shares_after": r["shares_after"], "stop_loss": r["stop_loss"]} for r in rows}


def _held_state_before(
    trades_for_ticker: list[dict], as_of: date, seed: tuple[int, float | None] = (0, None),
) -> tuple[int, float | None]:
    """(shares held, stop_loss in effect) strictly BEFORE `as_of` — the
    position as it stood when that day's decision was being made, not
    counting anything that decision itself might go on to trade. `seed` is
    the position state as of just before the 30-day-widened trade-history
    window began (see `_fetch_position_seed`) — used when this ticker has no
    row in `trades_for_ticker` before `as_of` at all, instead of always
    defaulting to "flat"."""
    shares, stop = seed
    for t in trades_for_ticker:
        if t["trade_date"] >= as_of:
            break
        shares = t["shares_after"]
        if t["stop_loss"] is not None:
            stop = t["stop_loss"]
    return shares, stop


async def simulate_tiered_screen(
    start_date: date = date(2026, 8, 1), account_mode: str = "paper", user_id: str | None = None,
) -> ScreenValidationResult:
    user_id = user_id or get_current_user_id()
    decisions = await _fetch_decisions(start_date, account_mode, user_id)
    trades_by_ticker = await _fetch_trades(start_date, account_mode, user_id)
    position_seed = await _fetch_position_seed(start_date, account_mode, user_id)

    def _seed_for(ticker: str) -> tuple[int, float | None]:
        seed = position_seed.get(ticker)
        return (seed["shares_after"], seed["stop_loss"]) if seed else (0, None)

    result = ScreenValidationResult()
    price_cache: dict[tuple[str, date], float | None] = {}
    atr_cache: dict[tuple[str, date], float | None] = {}

    async def price_at(ticker: str, d: date) -> float | None:
        key = (ticker, d)
        if key not in price_cache:
            try:
                price_cache[key] = await get_last_close(ticker, as_of=d)
            except Exception:
                price_cache[key] = None
        return price_cache[key]

    async def atr_at(ticker: str, d: date) -> float | None:
        key = (ticker, d)
        if key not in atr_cache:
            try:
                atr_cache[key] = await get_atr_pct(ticker, as_of=d)
            except Exception:
                atr_cache[key] = None
        return atr_cache[key]

    # Per-ticker running state across the chronological walk.
    prev_rating: dict[str, str] = {}
    prev_decision_date: dict[str, date] = {}
    consecutive_holds: dict[str, int] = {}
    consecutive_skips: dict[str, int] = {}
    last_full_analysis_date: dict[str, date] = {}

    for row in decisions:
        ticker, trade_date, rating = row["ticker"], row["trade_date"], row["rating"]
        result.total_decisions += 1
        tier0_reasons: list[str] = []

        is_first_ever = ticker not in prev_decision_date
        if is_first_ever:
            tier0_reasons.append("no_prior_decision")
        else:
            if prev_rating[ticker] != "Hold":
                tier0_reasons.append("prior_not_hold")

            atr_pct = await atr_at(ticker, trade_date)
            price_now = await price_at(ticker, trade_date)
            price_then = await price_at(ticker, prev_decision_date[ticker])
            if price_now is None or price_then is None or atr_pct is None:
                result.price_lookup_failures.append((ticker, trade_date))
                tier0_reasons.append("price_data_unavailable")  # unknown -> never safe to skip
            else:
                move_pct = abs(price_now - price_then) / price_then if price_then else 1.0
                if move_pct > PRICE_MOVE_ATR_MULTIPLE * atr_pct:
                    tier0_reasons.append("price_move_beyond_atr")

            if ticker not in last_full_analysis_date or (
                trade_date - last_full_analysis_date[ticker]
            ).days > STALENESS_LIMIT_DAYS:
                tier0_reasons.append("staleness")

            if consecutive_skips.get(ticker, 0) >= MAX_CONSECUTIVE_SKIPS:
                tier0_reasons.append("skip_streak_exceeded")

            shares_held, stop_loss = _held_state_before(
                trades_by_ticker.get(ticker, []), trade_date, seed=_seed_for(ticker),
            )
            if shares_held > 0 and stop_loss is not None and price_now is not None:
                distance_pct = (price_now - stop_loss) / price_now if price_now else 0.0
                if distance_pct <= STOP_LOSS_TIER0_PCT:
                    tier0_reasons.append("near_stop_loss")

        if tier0_reasons:
            tier = 0
            for reason in tier0_reasons:
                result.tier0_reason_counts[reason] = result.tier0_reason_counts.get(reason, 0) + 1
            last_full_analysis_date[ticker] = trade_date
            consecutive_skips[ticker] = 0
        else:
            # Tier 1 candidate — needs 2+ consecutive prior Holds, small
            # recent move (reuse the same ATR check, already computed above
            # and known to have passed), AND must not currently be held.
            # 2026-08-13: a distance-from-stop escape hatch used to let held
            # positions in anyway; a real Sell (064260.KS, 2026-07-29) fell
            # through it — a price/ATR screen can't see a news/fundamentals
            # driven exit, so no distance is actually safe. Held positions
            # are never screened now, full stop (matches the design doc's
            # own stated principle, which the old code didn't actually match).
            eligible = consecutive_holds.get(ticker, 0) >= MIN_CONSECUTIVE_HOLDS
            if eligible:
                shares_held, _ = _held_state_before(
                    trades_by_ticker.get(ticker, []), trade_date, seed=_seed_for(ticker),
                )
                eligible = shares_held == 0
            tier = 1 if eligible else 0
            if tier == 0:
                result.tier0_reason_counts["default_not_stable_enough"] = (
                    result.tier0_reason_counts.get("default_not_stable_enough", 0) + 1
                )
                last_full_analysis_date[ticker] = trade_date
                consecutive_skips[ticker] = 0

        if tier == 0:
            result.tier0_count += 1
        else:
            result.tier1_count += 1
            direction = _RATING_TO_DIRECTION.get(rating)
            if direction is not None and row["action_label"] not in (None, "Hold"):
                result.missed_actions.append(
                    MissedAction(ticker, trade_date, rating, row["action_label"], direction)
                )
            consecutive_skips[ticker] = consecutive_skips.get(ticker, 0) + 1

        prev_rating[ticker] = rating
        prev_decision_date[ticker] = trade_date
        consecutive_holds[ticker] = consecutive_holds.get(ticker, 0) + 1 if rating == "Hold" else 0

    return result


def render_screen_validation(r: ScreenValidationResult) -> str:
    lines = [
        f"총 판단 {r.total_decisions}건 — Tier0(전체분석) {r.tier0_count}건 / "
        f"Tier1(스킵) {r.tier1_count}건 ({r.tier1_count / r.total_decisions * 100:.1f}% 절감)"
        if r.total_decisions else "판단 없음",
        "",
        "Tier0으로 강제된 사유별 건수:",
    ]
    for reason, count in sorted(r.tier0_reason_counts.items(), key=lambda kv: -kv[1]):
        lines.append(f"  {reason:28} {count}")

    sell_missed = [m for m in r.missed_actions if m.direction == "sell"]
    buy_missed = [m for m in r.missed_actions if m.direction == "buy"]
    lines += [
        "",
        f"⚠️ 스킵됐다면 놓쳤을 매도 계열: {len(sell_missed)}건 (반드시 0이어야 함)",
        f"   스킵됐다면 놓쳤을 매수 계열: {len(buy_missed)}건",
    ]
    for m in r.missed_actions:
        lines.append(f"   - {m.trade_date} {m.ticker}: {m.rating} -> {m.action_label} ({m.direction})")

    if r.price_lookup_failures:
        lines += [
            "",
            f"가격 조회 실패(강제로 Tier0 처리됨, 결과 왜곡 가능): {len(r.price_lookup_failures)}건",
        ]
        for ticker, d in r.price_lookup_failures[:10]:
            lines.append(f"   - {d} {ticker}")

    lines += [
        "",
        "⚠️ 미반영: 거래량급증/서킷브레이커/실적임박/포트폴리오 국면전환 — 이 스크립트는 이 네 조건을",
        "   평가하지 않으므로 절감률은 상한선(실제 배포 시 이보다 적게 스킵됨)으로 볼 것.",
    ]
    return "\n".join(lines)
