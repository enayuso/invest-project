"""Confirms a mirrored order's ACTUAL fill against KIS's own execution
history — closing a gap `app.execution.kis_bridge.mirror_trade_to_kis`
leaves open: it records `mirror_disposition = "submitted: ..."` the
instant KIS ACCEPTS an order (an order number comes back), never once it
actually FILLS. `app/brokers/kis/domestic/order.py::submit_order`'s own
log line already says "order accepted", not "filled" — nothing downstream
of it ever re-checks.

Real incidents this closes the DETECTION half of (see project memory:
order-fill-confirmation-gap-2026-08-13): a 10-share phantom position
(008290.KS, order accepted, never actually filled) and a real 5-share
position left with NO stop-loss for over a week (005930.KS, the ledger
believed it had been sold when the sell was never actually filled) — both
undiscovered for weeks because nothing checked back.

Matches on `trades.kis_order_no` (app.execution.ledger._record_kis_order_no)
against `BrokerExecution.order_no` from KIS's own execution-history TR — an
order-number join (via `_same_order_no`, normalized for KIS's own
domestic-vs-overseas zero-padding inconsistency — see that function's own
docstring for the 2026-08-27 incident this fixes), not the (ticker, date,
side) heuristic app.execution.reconciliation has to fall back on for
orders that predate this column or never got a trade_id to attach one to.
Reuses app.web.state.recent_executions() (already 35s-cached, last 7 days)
— no new API calls.

Stage 1, DETECT AND ALERT ONLY, never touched the ledger. Stage 2
(2026-08-13, this file's `_auto_correct`) adds ledger correction for the
cases KIS gives an answer for: a DEFINITIVE dead-order signal of its own
(cancelled/rejected — see `check_pending_fills`'s own docstring for what
counts and, importantly, what does NOT), or an order still carrying an
open remainder that has gone stale past `fill_monitor_give_up_minutes` —
reusing the exact reverse_pending_trade/get_open_stop_loss/
resolve_stop_loss combination already used to manually fix
008290.KS/005930.KS. The ambiguous "no execution record found at all"
case is NEVER auto-corrected (matches app.execution.reconciliation's
existing "detect, human decides" boundary — absence of evidence isn't
evidence of absence).

Off by default (`fill_monitor_auto_correct_enabled`) and gated by three
independent safety checks (see `_auto_correct`'s own docstring): the trade
being corrected must still be this ticker's most recent one (nothing since
has assumed its now-wrong outcome), a stop-loss for any resulting open
position must be determinable (never restores a position with none — the
exact 2026-08-06 incident this traces back to), and a same-day correction
count circuit breaker (`fill_monitor_max_corrections_per_day`) stops ALL
auto-correction for the rest of the day if tripped, since repeated
corrections signal a structural problem this loop shouldn't paper over.
Every correction attempt is alerted regardless of outcome — auto-fixing a
discrepancy must never make it invisible to [[stage-gate-criteria-draft]]'s
"원장 대조 불일치 30일 연속 0건" gate, which counts DETECTED drift.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import timedelta, timezone

from app.clock import get_clock
from app.config import account_mode_scope, current_account_mode, settings
from app.dataflows.kis_provider import get_atr_pct
from app.domain.models import BrokerExecution
from app.execution.ledger import get_position, resolve_stop_loss, reverse_pending_trade
from app.execution.portfolio_risk import market_lock, ticker_market
from app.live.alerts import Alert, filter_new_alerts, record_alert
from app.memory.db import get_pool
from app.user_context import get_current_user_id, serviced_scopes, user_scope

# `reverse_pending_trade` itself prepends "reversal: " to whatever reason it's
# given — this is the part AFTER that prefix, so counting/matching against
# the stored value needs the full `_STORED_REASON_PREFIX` below.
_CORRECTION_REASON = "fill_monitor auto-correct"
_STORED_REASON_PREFIX = f"reversal: {_CORRECTION_REASON}"

logger = logging.getLogger("app.live.fill_monitor")

# How far back to look for trades still needing a fill check — independent
# of fill_monitor_give_up_minutes (which controls how soon THIS loop first
# alerts an unresolved one). 2 days comfortably covers even the slowest
# real delayed fill observed so far (the 8-day 000810.KS/006340.KS case
# turned out to be a mis-dated stop-loss exit, not an actually-slow fill —
# see project memory — but this stays generous since the real cost of
# checking a few extra rows each pass is negligible).
_PENDING_LOOKBACK = timedelta(days=2)

# _pending_fill_checks' own SELECT window is deliberately wider than
# _PENDING_LOOKBACK (the "give up for good" threshold `check_pending_fills`
# applies) — a row that already got the interim "not found" alert must stay
# selectable for the REST of that window so it can still be re-examined and,
# once _PENDING_LOOKBACK is actually reached, get its final give-up alert +
# `_mark_checked`. If the SELECT window were exactly _PENDING_LOOKBACK, a row
# alerted near the edge of it could silently drop out of every future
# pass's result set the moment `created_at` aged past the window, without
# `check_pending_fills` ever getting the chance to close it out. One extra
# day of margin is far more than fill_monitor_interval_seconds could ever
# need to catch up.
_SELECT_LOOKBACK = _PENDING_LOOKBACK + timedelta(days=1)


async def _pending_fill_checks() -> list[dict]:
    """Trades with a known KIS order number not yet resolved
    (`fill_checked_at IS NULL`), scoped to the current (user, account_mode)
    same as every other per-scope loop in this package. Degrades to []
    on any DB failure, same fail-open convention as
    app.execution.reconciliation.fetch_recent_local_trades.

    Includes `fill_unresolved_alerted_at` — see that column's own schema
    comment (app.execution.ledger) — so `check_pending_fills` can tell "never
    alerted yet" from "already alerted once, still watching for a definitive
    answer" without a second query."""
    try:
        pool = await get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, ticker, side, shares, price, created_at, kis_order_no,
                       fill_unresolved_alerted_at
                FROM trades
                WHERE account_mode = $1 AND user_id = $2
                  AND kis_order_no IS NOT NULL AND fill_checked_at IS NULL
                  AND created_at > now() - $3::interval
                ORDER BY created_at ASC
                """,
                current_account_mode(), get_current_user_id(), _SELECT_LOOKBACK,
            )
    except Exception as exc:  # noqa: BLE001 - same broadened fail-open policy as reconciliation.py
        logger.warning("fill_monitor: reading pending trades failed, skipping this pass: %s", exc)
        return []
    return [dict(r) for r in rows]


async def _mark_checked(trade_id: int) -> None:
    """Best-effort, same shape as app.execution.ledger's own follow-up
    writes — a failure here just means this trade gets re-examined (and,
    if still unresolved, re-alerted past its cooldown) next pass, not lost.

    Marks the order GENUINELY resolved (a real fill/no-fill answer, or the
    final _PENDING_LOOKBACK give-up) — see `fill_checked_at`'s own schema
    comment for why the interim "not found yet" case must NOT call this
    (that's `_mark_unresolved_alerted` instead)."""
    try:
        pool = await get_pool()
        async with pool.acquire() as conn:
            await conn.execute("UPDATE trades SET fill_checked_at = now() WHERE id = $1", trade_id)
    except Exception:
        logger.warning("fill_monitor: could not mark trade_id=%d checked — will retry next pass", trade_id, exc_info=True)


async def _mark_unresolved_alerted(trade_id: int) -> None:
    """Records that the interim "no KIS execution record found" alert has
    already fired for this order — suppresses repeating that SAME alert
    every remaining pass within _PENDING_LOOKBACK, without taking the order
    out of `_pending_fill_checks`' result set the way `_mark_checked` would
    (see that column's own schema comment in app.execution.ledger). A
    failure here just means the interim alert can fire again next pass —
    same fail-open direction as `_mark_checked`'s own failure mode, and
    strictly less harmful (a duplicate alert vs. a silently un-reexamined
    order)."""
    try:
        pool = await get_pool()
        async with pool.acquire() as conn:
            await conn.execute("UPDATE trades SET fill_unresolved_alerted_at = now() WHERE id = $1", trade_id)
    except Exception:
        logger.warning(
            "fill_monitor: could not mark trade_id=%d as alerted — its interim alert may repeat next pass",
            trade_id, exc_info=True,
        )


async def _corrections_today() -> int:
    """Same-day count of trades this loop itself wrote via `_auto_correct` —
    the circuit-breaker's own input. Fails CLOSED (returns a number at/above
    the threshold) on any DB error, not 0 — an error here must never look
    like "no corrections yet, safe to keep going" and let a failing DB
    silently disable the one thing standing between it and unbounded
    auto-correction."""
    try:
        pool = await get_pool()
        async with pool.acquire() as conn:
            return await conn.fetchval(
                "SELECT count(*) FROM trades WHERE account_mode = $1 AND user_id = $2 "
                "AND reason LIKE $3 AND created_at >= date_trunc('day', now())",
                current_account_mode(), get_current_user_id(), f"{_STORED_REASON_PREFIX}%",
            )
    except Exception:
        logger.warning("fill_monitor: could not count today's corrections — treating circuit breaker as tripped", exc_info=True)
        return settings.fill_monitor_max_corrections_per_day


async def _is_most_recent_trade(ticker: str, trade_id: int) -> bool:
    """False means something has traded on this ticker since `trade_id` —
    reversing it now would be undoing a premise later trades already built
    on, not a clean correction (the exact reason 005930.KS needed a human to
    collapse 4 rows into one rather than a mechanical reversal)."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        latest_id = await conn.fetchval(
            "SELECT id FROM trades WHERE ticker = $1 AND account_mode = $2 AND user_id = $3 "
            "AND is_backtest = false ORDER BY created_at DESC LIMIT 1",
            ticker, current_account_mode(), get_current_user_id(),
        )
    return latest_id == trade_id


async def _prior_stop_loss(ticker: str, trade_id: int) -> tuple[bool, float | None]:
    """(found_a_prior_row, its stop_loss) — the protection to restore onto a
    reversal of `trade_id` if the reversal leaves shares open.
    found_a_prior_row=False means `trade_id` was this ticker's very first
    trade ever (nothing to restore); the caller derives a fresh ATR-based
    stop instead, same as a brand-new position would get."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT id, stop_loss FROM trades WHERE ticker = $1 AND account_mode = $2 AND user_id = $3 "
            "AND is_backtest = false ORDER BY created_at DESC LIMIT 2",
            ticker, current_account_mode(), get_current_user_id(),
        )
    if len(rows) < 2 or rows[0]["id"] != trade_id:
        return False, None
    return True, rows[1]["stop_loss"]


async def _auto_correct(row: dict, reversal_shares: int, resolution: str) -> bool:
    """Reverses `reversal_shares` of `row` — a mirrored trade KIS has now
    given a definitive dead-order answer for (fully unfilled, or a partial
    fill whose remainder went stale). Returns whether the ledger was
    actually changed; never raises (a failure here must not crash the
    monitor loop — the caller's own alert covers the human-visible side
    either way).

    Two safety gates beyond the caller's own feature-flag/circuit-breaker
    check, both fail towards "alert only, don't touch the ledger":
    - `row` must still be this ticker's MOST RECENT trade.
    - If the reversal leaves shares open, a stop-loss to carry onto them
      must be determinable (the prior trade's own, or a fresh ATR-based
      default for a first-ever trade) — never restores an open position
      with no stop.
    """
    if reversal_shares <= 0:
        return False
    if not await _is_most_recent_trade(row["ticker"], row["id"]):
        logger.info(
            "fill_monitor auto-correct: %s trade_id=%d is no longer this ticker's most recent trade "
            "— skipping, alert only", row["ticker"], row["id"],
        )
        return False

    async with market_lock(ticker_market(row["ticker"])):
        # Mirrors reverse_pending_trade's own clamping so the stop-loss
        # decision below (made BEFORE calling it) matches what it will
        # actually do — see that function's docstring for why sells clamp
        # (long-only book) and buys don't.
        position = await get_position(row["ticker"])
        reversed_side = "sell" if row["side"] == "buy" else "buy"
        clamped = min(reversal_shares, position.shares) if reversed_side == "sell" else reversal_shares
        if clamped <= 0:
            return False
        resulting_shares = position.shares - clamped if reversed_side == "sell" else position.shares + clamped

        prior_stop = None
        if resulting_shares > 0:
            found_prior, prior_stop = await _prior_stop_loss(row["ticker"], row["id"])
            if not found_prior:
                try:
                    atr_pct = await get_atr_pct(row["ticker"])
                except Exception:
                    atr_pct = None
                prior_stop = resolve_stop_loss(None, row["price"], resulting_shares, atr_pct)
            if prior_stop is None:
                logger.warning(
                    "fill_monitor auto-correct: %s trade_id=%d reversal would leave %d shares with no "
                    "determinable stop-loss — refusing to auto-correct, alert only",
                    row["ticker"], row["id"], resulting_shares,
                )
                return False

        await reverse_pending_trade(
            row["ticker"], row["side"], reversal_shares, row["price"],
            reason=f"{_CORRECTION_REASON} ({resolution})", prior_stop_loss=prior_stop,
        )
    return True


def normalize_kis_order_no(order_no: str | None) -> str:
    """Strip and remove leading zeros — see `_same_order_no`'s docstring for
    why domestic (zero-padded) and overseas (not zero-padded) execution-
    history order numbers need this before any comparison or dict-keying."""
    return (order_no or "").strip().lstrip("0")


def _same_order_no(a: str | None, b: str | None) -> bool:
    """KIS's own order-number (odno) representation differs by TR: domestic
    execution history (fetch_daily_executions) zero-pads to 10 digits
    ('0000043249'), but overseas execution history (TTTS3035R/VTTS3035R)
    does not ('43249'). `trades.kis_order_no` always stores the zero-padded
    ODNO from the order-acceptance response, so a plain string `==` compare
    can never match an overseas order — this class of bug went undetected
    because both sides silently reported success (a False match falls
    through to the "not found" branch below, alerted and marked checked,
    exactly like a genuine phantom).

    Real incident confirmed 2026-08-27: every "not found" alert fill_monitor
    had ever raised was for an overseas ticker (MRVL x2, AMZN, GOOGL, GOOG,
    AXTI, SNDK) — zero for domestic. Direct KIS reconciliation showed 4 of
    those 7 (AXTI/GOOG/GOOGL/AMZN) were false positives (fully filled,
    normal orders this bug just could never confirm), and 2 (MRVL, SNDK)
    were genuine phantom fills that should have hit the immediate
    `filled_qty <= 0` branch below (confirmed-dead-order, auto-correct
    eligible) instead of silently falling into "not found" (ambiguous,
    never auto-corrected, and — see `check_pending_fills`'s "not found"
    branch — never re-checked again after this alert fires). In short:
    overseas fill confirmation has never actually worked since this module
    was introduced.

    `app.execution.pnl._order_no_sort_key` already works around the same
    KIS inconsistency with an `int(...)` conversion for sorting — this
    normalizes the same way for matching, in the comparison layer rather
    than the broker layer, because `BrokerExecution.order_no` is shared with
    the overseas cancel path (`ORGN_ODNO`), which must keep echoing back
    whatever raw form KIS itself uses there.

    Never matches on an empty value either side — normalizing '' and
    '0000000000' to the same '' would let two UNRELATED orders (one with no
    number recorded, one genuinely zero) compare equal, which is the worst
    possible failure direction here: mistaking a stranger's fill for this
    order's confirmation would let a real phantom trade read as
    "confirmed filled" instead of merely staying unmatched.

    The normalization itself lives in `normalize_kis_order_no` — other
    modules that need to key/compare KIS order numbers (e.g.
    `app.audit.slippage_validation`) should import that instead of
    re-implementing this stripping logic."""
    norm_a, norm_b = normalize_kis_order_no(a), normalize_kis_order_no(b)
    return bool(norm_a) and norm_a == norm_b


def _find_execution(order_no: str, domestic: list[BrokerExecution], overseas: list[BrokerExecution]) -> BrokerExecution | None:
    for e in domestic:
        if _same_order_no(e.order_no, order_no):
            return e
    for e in overseas:
        if _same_order_no(e.order_no, order_no):
            return e
    return None


async def check_pending_fills(
    domestic: list[BrokerExecution],
    overseas: list[BrokerExecution],
    give_up_after: timedelta,
    *,
    pending: list[dict] | None = None,
) -> None:
    """One pass, pure matching against the given executions (caller
    fetches — same testable-boundary convention as
    app.execution.reconciliation.find_unmirrored_trades). Every branch is
    detect-and-alert only; none of them ever call into app.execution.ledger.

    - No matching execution found at all: ambiguous while young (KIS's own
      execution-history TR can lag slightly behind order acceptance) — the
      first time `give_up_after` passes with still no record, alerts ONCE
      and records that via `_mark_unresolved_alerted` (NOT `_mark_checked`
      — see that function's own docstring for why this must stay
      re-examinable). Every later pass keeps silently re-checking (DEBUG
      log only) until either a definitive answer shows up (handled by the
      branches below, on whichever pass first sees it) or `_PENDING_LOOKBACK`
      itself is reached, at which point a final give-up alert fires and the
      row IS marked checked for good.

      2026-08-27 fix: this branch used to call `_mark_checked` on its very
      first alert, permanently excluding the order from every future pass —
      even though KIS's execution-history TR reliably lagging past
      `give_up_after` (30 min by default) is exactly the case this whole
      branch exists to tolerate. Confirmed live: both of MRVL's last two
      overseas orders hit this exact branch, and one of them (order_no
      0000043249) had, in fact, genuinely filled by the time it was checked
      directly — the alert this loop raised for it was a false positive that
      a later re-check would have silently resolved on its own.
    - Found, and KIS itself says the order is DEAD — `cancelled` (domestic
      cncl_yn) or `rejected` (overseas rjct_rson), or a zero-fill row KIS
      also reports as having nothing left open: alerted immediately and
      marked checked regardless of age. This is the 008290.KS/005930.KS
      failure mode, and once KIS has given a definitive negative there is
      nothing to wait for. Auto-correct reverses only the part that did NOT
      fill (`shares - filled_qty`), not the whole recorded quantity — KIS's
      cncl_yn can sit on an order that partially executed before its
      remainder was cancelled, and reversing the filled part too would
      erase a real, stop-loss-bearing position from the ledger: the exact
      mirror image of the phantom this module exists to catch.

      2026-08-27 fix: this branch used to fire on `filled_qty <= 0` ALONE,
      with no age check and no cancelled/rejected test. But `filled_qty == 0`
      is also precisely what a healthy, still-resting limit order looks
      like — and live auto-trading submits LIMIT orders for every overseas
      ticker (KIS rejects market orders there), so zero fill for the first
      minutes or hours after submission is the NORMAL overseas state. The
      result was a false "order was never filled" alert the moment KIS's
      execution history first showed the order, plus a `_mark_checked` that
      permanently excluded it from every later pass — so the eventual real
      fill was never reconciled, and had
      `fill_monitor_auto_correct_enabled` been on, a live working order
      would have had its ledger entry reversed out from under a fill that
      was about to happen. Same class of mistake as this function's other
      2026-08-27 fix below (closing a row out while the answer was still
      pending), one branch over.
      해외는 지정가만 나가므로 filled_qty==0 은 "실패"가 아니라 "아직 안 붙음"이다.
    - Found, not confirmed dead, still has an open remainder
      (`remaining_qty > 0`) — zero fill and partial fill alike: left
      unresolved (order may still be working) until `give_up_after`, same
      reasoning as the "not found" case, then alerted and marked checked
      with the unfilled quantity reversed. Note the trade-off this leaves
      open: `fill_monitor_give_up_minutes` therefore doubles as "how long a
      resting limit order is allowed to live", so it must be set longer
      than a realistic overseas limit order's life before
      `fill_monitor_auto_correct_enabled` is ever turned on.
    - Found, fully filled: resolved quietly (DEBUG log only, no alert) —
      this is the expected, common case; marked checked. A mismatch against
      the ledger's own recorded price/shares is still deliberately not
      checked here, and this is the one remaining home for it: `execution.
      avg_price`/`execution.filled_qty` in hand on this exact branch are the
      only real fill figures this system ever sees, and `trades.price` is
      still only ever the pipeline's own pre-submission snapshot (see
      `app.audit.slippage_validation`'s module docstring). The agreed design
      for closing that (2026-08-27, deferred deliberately, not forgotten) is
      NON-DESTRUCTIVE: write `avg_price`/`filled_qty` into new
      `trades.actual_fill_price`/`actual_fill_shares` columns and leave
      `price`/`cash_after` untouched, because `price` is what `record_trade`
      derived `cash_after` from and rewriting it retroactively would
      desynchronize the cash chain of every later row. Not done in the same
      change as the forced-exit price fix on purpose: this loop's matching
      logic was itself repaired three times in the two days before that
      (2615403 / 7323096 / 310f99a) and needs a clean baseline first.

    `pending`, if given, is used instead of a fresh `_pending_fill_checks()`
    query — FIX-4 (2026-09-03 id=1187 audit): `fill_monitor_loop` now checks
    for pending rows BEFORE fetching `domestic`/`overseas` at all (a real KIS
    execution-history ledger TR), skipping that fetch entirely when there is
    nothing to reconcile, and passes the same rows it already fetched here
    rather than re-querying the (cheap, local-DB-only) pending list a second
    time. Defaults to None (fetch fresh) so any other/test caller is
    unaffected."""
    # timezone.utc, not the naive default get_clock().now() returns (see
    # app.live.exit_monitor's identical need) — `created_at` comes back
    # from asyncpg as an aware UTC datetime (TIMESTAMPTZ), and subtracting
    # a naive datetime from an aware one raises TypeError.
    now = get_clock().now(timezone.utc)
    fresh_alerts: list[Alert] = []

    # Circuit breaker checked ONCE per pass, not per row — a tripped breaker
    # disables every auto-correct attempt below for the rest of this pass
    # (and, since it re-checks the same same-day count, every later pass
    # today too) but never disables alerting, which stays unconditional.
    auto_correct_enabled = settings.fill_monitor_auto_correct_enabled
    circuit_tripped = False
    if auto_correct_enabled:
        today_count = await _corrections_today()
        if today_count >= settings.fill_monitor_max_corrections_per_day:
            circuit_tripped = True
            fresh_alerts.append(
                Alert(
                    ticker="SYSTEM", kind="fill_monitor_circuit_breaker",
                    message=(
                        f"자동보정이 오늘 {today_count}건 발생해 상한({settings.fill_monitor_max_corrections_per_day}건)을 "
                        "넘었습니다 — 자동보정을 멈추고 알림만 남깁니다. 반복되는 원장 불일치는 개별 주문 문제가 아니라 "
                        "구조적 문제(브로커 장애·대사 로직 오류)일 수 있으니 확인이 필요합니다."
                    ),
                    triggered_at=get_clock().now(),
                )
            )

    for row in (pending if pending is not None else await _pending_fill_checks()):
        age = now - row["created_at"]
        execution = _find_execution(row["kis_order_no"], domestic, overseas)

        if execution is None:
            if age < give_up_after:
                continue  # still young — KIS's execution-history TR may just be lagging
            if age >= _PENDING_LOOKBACK:
                # Final give-up — KIS has never once returned a matching
                # record across the entire lookback window this loop bothers
                # checking. Unlike the interim alert below, this closes the
                # row out for good: further waiting has nothing left to gain.
                fresh_alerts.append(
                    Alert(
                        ticker=row["ticker"], kind="unconfirmed_fill_final",
                        message=(
                            f"{row['side']} x{row['shares']} (order_no={row['kis_order_no']}) — still no KIS "
                            f"execution record after {_PENDING_LOOKBACK.days} day(s) — giving up on automatic "
                            "re-checking; a human needs to reconcile this against the real KIS account directly"
                        ),
                        triggered_at=get_clock().now(),
                    )
                )
                await _mark_checked(row["id"])
                continue
            if row["fill_unresolved_alerted_at"] is not None:
                # Already alerted once for this exact ambiguous state — keep
                # re-checking silently rather than repeating the same alert
                # every remaining pass (that repeat noise, not the re-check
                # itself, is what _mark_checked used to exist to prevent; see
                # this function's own docstring for why marking the row
                # fully checked here instead would have been the wrong fix).
                logger.debug(
                    "fill_monitor: %s order_no=%s still not found (already alerted, will keep re-checking "
                    "until %s day give-up)", row["ticker"], row["kis_order_no"], _PENDING_LOOKBACK.days,
                )
                continue
            fresh_alerts.append(
                Alert(
                    ticker=row["ticker"], kind="unconfirmed_fill",
                    message=(
                        f"{row['side']} x{row['shares']} (order_no={row['kis_order_no']}) — no KIS execution "
                        f"record found {give_up_after.total_seconds() / 60:.0f}+ minutes after submission — "
                        "the ledger may hold a phantom trade "
                        f"(this pass saw {len(domestic)} domestic / {len(overseas)} overseas execution rows — "
                        "zero on this order's own side usually means the lookup itself is broken "
                        "(credentials/sentinel/rate limit), a nonzero count that still doesn't match "
                        "means the order genuinely isn't in KIS's history yet, or the match itself failed). "
                        f"Will keep re-checking silently for up to {_PENDING_LOOKBACK.days} day(s) before "
                        "giving up for good."
                    ),
                    triggered_at=get_clock().now(),
                )
            )
            await _mark_unresolved_alerted(row["id"])
            continue

        # KIS's own definitive dead-order signals — the ONLY grounds for
        # closing an order out immediately (see this function's docstring,
        # 2026-08-27 note: `filled_qty <= 0` on its own is not one of them,
        # it is also what a perfectly healthy resting limit order looks
        # like). `cancelled` covers domestic (cncl_yn), `rejected` covers
        # overseas (rjct_rson) — see BrokerExecution's own field comments
        # for why neither is a substitute for the other. The third case is
        # KIS saying an order neither filled NOR has anything left open:
        # nothing more can ever happen to it, and without this disjunct it
        # would fall through to the fully-filled tail below and get closed
        # out with a DEBUG line claiming "confirmed filled (0 @ ...)".
        # 국내는 cncl_yn, 해외는 rjct_rson — 둘 중 하나라도 서면 확정 사망.
        unfilled = row["shares"] - execution.filled_qty
        kis_confirms_dead = (
            execution.cancelled
            or execution.rejected
            or (execution.filled_qty <= 0 and execution.remaining_qty <= 0)
        )

        if unfilled > 0 and kis_confirms_dead:
            # `unfilled > 0` guard: a fully-filled order carrying a stray
            # cancel/reject flag (a cancel request that lost the race to the
            # fill) has nothing to reverse and nothing to warn about — it
            # falls through to the fully-filled tail instead of being
            # announced as dead.
            if execution.cancelled:
                dead_signal = "cancelled"
            elif execution.rejected:
                dead_signal = "rejected"
            else:
                dead_signal = "zero filled with nothing left open"
            # "order never filled" is kept verbatim for the zero-fill case:
            # it is what lands in the ledger's own `reason` column, and the
            # 008290.KS/005930.KS-era manual corrections used that wording.
            resolution = (
                "order never filled"
                if execution.filled_qty <= 0
                else f"dead after partial fill {execution.filled_qty:g}/{row['shares']}"
            )
            corrected = False
            if auto_correct_enabled and not circuit_tripped:
                corrected = await _auto_correct(row, reversal_shares=int(unfilled), resolution=resolution)
            if execution.filled_qty <= 0:
                message = (
                    f"{row['side']} x{row['shares']} (order_no={row['kis_order_no']}) — KIS confirms this "
                    f"order was never filled ({dead_signal}) — the ledger recorded it as a completed trade "
                    f"anyway{' [자동보정: 원장 되돌림 완료]' if corrected else ''}"
                )
            else:
                message = (
                    f"{row['side']} x{row['shares']} (order_no={row['kis_order_no']}) — KIS confirms this "
                    f"order is dead ({dead_signal}) with only {execution.filled_qty:g}/{row['shares']} "
                    f"filled — the ledger recorded the full quantity as a completed trade"
                    f"{' [자동보정: 미체결분 되돌림 완료]' if corrected else ''}"
                )
            fresh_alerts.append(
                Alert(
                    ticker=row["ticker"], kind="unconfirmed_fill",
                    message=message, triggered_at=get_clock().now(),
                )
            )
            await _mark_checked(row["id"])
            continue

        if execution.remaining_qty > 0:
            # Not confirmed dead, still has an open remainder — zero-fill and
            # partial-fill both land here, and both must WAIT. A zero-fill
            # overseas order is the normal state of a live limit order (live
            # auto-trading submits limit orders for every overseas ticker,
            # KIS rejecting market orders there), so there is nothing to
            # distinguish it from a slow one until the give-up window closes.
            if age < give_up_after:
                continue  # order may still be working
            resolution = (
                f"still open, 0/{row['shares']} filled past give-up"
                if execution.filled_qty <= 0
                else f"partial fill {execution.filled_qty:g}/{row['shares']}"
            )
            corrected = False
            if auto_correct_enabled and not circuit_tripped:
                corrected = await _auto_correct(row, reversal_shares=int(unfilled), resolution=resolution)
            fresh_alerts.append(
                Alert(
                    ticker=row["ticker"], kind="unconfirmed_fill",
                    message=(
                        f"{row['side']} x{row['shares']} (order_no={row['kis_order_no']}) — only "
                        f"{execution.filled_qty:g}/{row['shares']} filled after "
                        f"{give_up_after.total_seconds() / 60:.0f}+ minutes, {execution.remaining_qty:g} still open"
                        " — KIS reports neither a cancellation nor a rejection, so this is a STALE order, not a"
                        " confirmed dead one"
                        f"{' [자동보정: 미체결분 되돌림 완료]' if corrected else ''}"
                    ),
                    triggered_at=get_clock().now(),
                )
            )
            await _mark_checked(row["id"])
            continue

        # Fully filled — the expected, common case. Not alerted; a mismatch
        # against the ledger's own recorded price/shares (stage 2's
        # eventual concern) is deliberately not checked here.
        logger.debug(
            "fill_monitor: %s order_no=%s confirmed filled (%.4g @ %.4g)",
            row["ticker"], row["kis_order_no"], execution.filled_qty, execution.avg_price,
        )
        await _mark_checked(row["id"])

    for alert in filter_new_alerts(fresh_alerts):
        await record_alert(alert)


async def fill_monitor_loop() -> None:
    """One pass per serviced scope per interval, each inside `user_scope` +
    `account_mode_scope` — same reasoning as every other background loop in
    this package (app.live.exit_monitor.exit_monitor_loop,
    app.live.poller.run_poller_loop): `_pending_fill_checks`/`_mark_checked`
    read/write this scope's own trades, so nothing here may run un-bound."""
    from app.web.state import kis_broker, recent_executions  # local import: state.py imports this module's loop

    interval_seconds = settings.fill_monitor_interval_seconds
    give_up_after = timedelta(minutes=settings.fill_monitor_give_up_minutes)
    while True:
        for scope in await serviced_scopes():
            try:
                with user_scope(scope.user_id), account_mode_scope(scope.account_mode):
                    if await kis_broker() is None:
                        continue
                    # FIX-4 (2026-09-03 id=1187 audit): check for pending
                    # rows FIRST, using only the local trades table, and
                    # skip `recent_executions()` (a real KIS execution-
                    # history ledger TR) entirely when there is nothing to
                    # reconcile this pass — every quiet pass used to fire
                    # that TR anyway, adding to the same EGW00215 per-second
                    # quota pressure for no possible outcome.
                    pending = await _pending_fill_checks()
                    if not pending:
                        continue
                    domestic, overseas = await recent_executions()
                    await check_pending_fills(domestic, overseas, give_up_after, pending=pending)
            except Exception:
                logger.exception(
                    "fill_monitor pass failed for %s/%s — continuing with the other scopes",
                    scope.user_id, scope.account_mode,
                )
        await asyncio.sleep(interval_seconds)
