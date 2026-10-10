"""POC recurring recommend/auto-trade loop — same shape as
app.web.state.stop_loss_loop (while-True + asyncio.sleep, started as a
background task in app.web.state's lifespan), but drives the full
multi-agent decision pipeline instead of a cheap price check.

Domestic (KRX) and overseas (US) are kept as two fully separate pools —
never pooled into one momentum ranking — matching this project's existing
domestic/overseas separation everywhere else (ledger's per-ticker
currency-converted cash pools, the dashboard's two account blocks, KIS's own
separate domestic/overseas broker mixins) and real brokerage apps' own UI
convention (국내/해외 tabs, never merged). Each side only runs when ITS
market is actually open (app.live.market_hours) — no analysis for a market
that isn't trading right now, and no candidates spent on it either.
국내/해외를 절대 하나로 합치지 않음 — 각각 따로 스코어링, 각 시장이 열려있을 때만 그 시장만 분석.

Each side: its trade_allowlist entries + a bounded set of screened candidates
(app.live.universe.krx_top_n / overseas_top_n) are pooled, narrowed to what's
actually eligible for a ranking slot this cycle (_eligible_universe — the
rebalance-interval throttle and the max-open-positions cap, both cheap/
deterministic/DB-only), THEN momentum-ranked (app.chat.screening.
top_candidates, LLM-free) down to that side's own
auto_trade_max_tickers_domestic/_overseas (with a small overfetch —
auto_trade_candidate_overfetch_multiplier — spent backfilling any name the
broker-I/O execution-quality filters drop afterward). The eligibility
narrowing MUST happen before ranking, never after: applying the rebalance
throttle to the ranking's own output (as this code briefly did) lets a
locked top-ranked name occupy its slot forever, since momentum ranking
returns nearly the same order every cycle — see _eligible_universe's own
docstring for the 2026-08-26 incident this caused (the entire overseas
candidate pool starved itself for most of a trading session). trade_allowlist
membership does NOT guarantee a ticker gets analyzed this cycle, it's just
one more entry in its market's own scored pool. One exception: a currently-
held ticker whose price has drifted close to its recorded stop-loss gets a
full review regardless of momentum rank, the eligibility narrowing, or the
momentum budget (see _held_needing_review) — the flagging check itself is
cheap enough (one price + one DB read, no LLM) to run on every holding
every cycle, but the ~20-LLM-call full pipeline it feeds into is still
throttled to at most once every held_review_min_interval_days per ticker.

The combined survivors run through app.orchestrator.pipeline.run_pipeline
with bounded concurrency (settings.auto_trade_max_concurrent_analyses,
default 2) rather than one at a time — the ~20-LLM-call pipeline is
dominated by network round-trip latency, so a few tickers in flight at
once still make progress against a single LLM backend. This used to be
strictly sequential (see git history) on the assumption that a single-
instance backend had nothing to gain from it; that assumption didn't
account for I/O-bound LLM latency, and more importantly, portfolio-level
caps (concentration/correlation/drawdown) used to be checked against a
snapshot fetched once per ticker at the START of its own ~20-call run,
reused unchanged all the way to commit — fine when strictly sequential
(ticker N+1 never started until ticker N's trade had already landed), but
unsafe once analysis overlaps: two tickers could each see the other's
pre-trade balance and jointly approve exposure that, combined, breaches a
cap neither alone would. Portfolio-cap enforcement now re-fetches the real
account fresh and is serialized per market at COMMIT time instead (see
app.execution.portfolio_risk.market_lock, app.execution.ledger.
rebalance_to_target) — that's what makes overlapping analysis safe here.
What happens to each resulting decision is entirely settings.execution_mode's
concern, already wired into app.execution.ledger.rebalance_to_target — this
loop does not duplicate that branching, it just calls run_pipeline like
every other caller.

MANUAL mode skips the cycle's analysis entirely rather than running the
(costly) pipeline for recommendation value alone — confirmed with the user:
choosing MANUAL means "no AI involvement," not just "no auto-submission."

A slow cycle simply pushes the next cycle's start later (sleep begins only
after the loop body finishes) — a missed interval is skipped, never queued.
반자동/완전자동 모드에서만 동작 — 수동 모드는 이 반복 분석 자체를 실행하지 않음(비용 방지).
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Literal

from app.brokers.kis.broker import krx_code_from_ticker
from app.chat.screening import screening_pass
from app.clock import Market, get_clock, local_today
from app.config import ExecutionMode, account_mode_scope, current_account_mode, settings
from app.execution import runtime_settings
from app.execution.kis_bridge import designation_block, pre_screening_blockers, spread_blocker
from app.execution.ledger import get_open_stop_loss
from app.live.alerts import Alert, filter_new_alerts, record_alert
from app.live.market_hours import is_domestic_market_open, is_overseas_market_open
from app.memory.store import get_last_decision_time, tickers_analyzed_on, tickers_cash_floor_capped
from app.live.store import latest
from app.live.universe import (
    krx_disparity_low_n,
    krx_fluctuation_losers_n,
    krx_market_cap_top_n,
    krx_top_n,
    overseas_market_cap_top_n,
    overseas_top_n,
)
from app.cancellation import task_was_actually_cancelled
from app.orchestrator.pipeline import run_pipeline
from app.user_context import serviced_scopes, user_scope
from app.web.ticker_validation import is_valid_ticker_format
from app.execution import risk_rules

logger = logging.getLogger("app.live.auto_trade_loop")

Phase = Literal["idle", "manual_skip", "scanning", "analyzing", "waiting_for_manual"]

# 2026-07-21: SEMI_AUTO auto-runs once (see auto_trade_loop below), then
# stops re-triggering itself on the periodic timer — further cycles only
# come from the dashboard's "다시 분석" button, which is only meaningful in
# SEMI_AUTO (FULL_AUTO has no such button; it keeps running on the timer).
# This cooldown is the server-side enforcement for that button — the
# client's own cooldown countdown is a courtesy display, not the real gate.
MANUAL_TRIGGER_COOLDOWN_SECONDS = 5 * 60


class CooldownActive(Exception):
    """Raised by check_manual_trigger_allowed() when the '다시 분석' button
    can't run right now — wrong mode, a cycle already running, or still
    within the cooldown window since the last one finished."""


@dataclass
class AutoTradeStatus:
    """Read-only-from-outside snapshot of what this loop is doing right
    now — exists purely for dashboard visibility (a user watching a
    SEMI_AUTO/FULL_AUTO test has no other way to tell "is it actually
    running" from a quiet terminal between log lines). Only this module
    writes to the shared `_status` instance; app.web.routers only reads
    it via get_status()."""

    phase: Phase = "idle"
    # A set, not a single ticker, since candidates now analyze with bounded
    # concurrency (settings.auto_trade_max_concurrent_analyses) instead of
    # strictly one at a time — see this module's own docstring.
    current_tickers: set[str] = field(default_factory=set)
    # Mirrors current_tickers but carries the most recent raw on_stage label
    # for each in-flight ticker (e.g. "Bull vs. Bear debate — round 1/2...")
    # so the dashboard's analyzing card can show live pipeline progress
    # instead of a generic spinner. Cleared at the same points as
    # current_tickers.
    current_stage: dict[str, str] = field(default_factory=dict)
    tickers_this_cycle: list[str] = field(default_factory=list)
    cycle_started_at: datetime | None = None
    cycle_finished_at: datetime | None = None
    next_cycle_at: datetime | None = None


# One AutoTradeStatus per (user, account_mode). It was a single module
# instance, which with more than one serviced user meant every user's dashboard
# showed whichever cycle happened to run last — and each cycle overwrote the
# previous user's "analyzing NVDA" with its own.
# 상태가 단일 인스턴스면 사용자마다 마지막에 돌아간 사이클 상태가 보임.
_status_by_scope: dict[tuple[str, str], AutoTradeStatus] = {}


def _status_scope() -> tuple[str, str]:
    from app.config import current_account_mode
    from app.user_context import get_current_user_id

    return (get_current_user_id(), current_account_mode())


def _status_for_scope() -> AutoTradeStatus:
    scope = _status_scope()
    status = _status_by_scope.get(scope)
    if status is None:
        status = AutoTradeStatus()
        _status_by_scope[scope] = status
    return status


class _ScopedStatus:
    """Forwards every attribute read/write to the current scope's status.

    A proxy rather than replacing `_status` with a `_status_for_scope()` call at
    each of its ~22 touch sites: the sites are all in this module, all mutate
    two or three fields at a time, and rewriting each one is 22 chances to
    forget a scope and write into the wrong user's status. Same reasoning as
    the Settings ticker-list properties — keep the call shape, move the
    resolution underneath it.
    호출부를 그대로 두고 해석만 아래로 내림 — 22곳을 각각 고치면 한 곳만 놓쳐도
    다른 사용자의 상태에 쓰게 됨.

    Deliberately holds no instance state of its own, so `__setattr__` cannot
    recurse: every write goes to the resolved AutoTradeStatus.
    자체 상태를 두지 않아 __setattr__ 재귀가 발생하지 않음."""

    def __getattr__(self, name: str):
        return getattr(_status_for_scope(), name)

    def __setattr__(self, name: str, value) -> None:
        setattr(_status_for_scope(), name, value)

    def __repr__(self) -> str:
        return f"_ScopedStatus({_status_scope()} -> {_status_for_scope()!r})"


_status = _ScopedStatus()


def get_status() -> AutoTradeStatus:
    """The asking user's own cycle status, in their own mode.

    Called from `/api/auto-trade/status`, which runs inside a request, so the
    scope is the requester's — which is what the dashboard should show.
    요청 컨텍스트에서 호출되므로 요청자 본인의 상태가 반환됨."""
    return _status_for_scope()


def reset_status_scopes() -> None:
    """Drop every scope's status. Tests only."""
    _status_by_scope.clear()
    _semi_auto_done.clear()
    _retry_failures_by_scope.clear()


def _split_trade_allowlist() -> tuple[list[str], list[str]]:
    # Filters out any malformed entry that predates the format check
    # app.web.state.add_ticker_to_trade_allowlist now enforces on the way
    # in — without this, a bad entry already on settings.trade_allowlist
    # (from before that check existed) would keep getting pooled into this
    # cycle's own candidates below, re-analyzed and re-surfaced as an AI
    # signal indefinitely, self-perpetuating until someone manually removes
    # it. Domestic/overseas are now real separate fields (app.config), so
    # this no longer needs its own .KS-suffix split.
    domestic = [t for t in settings.trade_allowlist_domestic if is_valid_ticker_format(t)]
    overseas = [t for t in settings.trade_allowlist_overseas if is_valid_ticker_format(t)]
    return domestic, overseas


async def _partition_analyzed_within(
    tickers: list[str], min_days: int, market: Market
) -> tuple[list[str], dict[str, date]]:
    """Shared throttle: partitions `tickers` into those whose last LIVE
    full-pipeline decision was at least `min_days` ago (or never analyzed —
    get_last_decision_time returns None, always passes) and those still
    locked, keyed to the date each locked ticker becomes eligible again.
    This only throttles REPEAT full analysis, never a first look. Used with
    two different `min_days` values for two different cadences (see
    `_eligible_universe` and `_held_needing_review`'s own docstrings) —
    kept as one function so the two throttles can't quietly drift apart in
    behavior, only in the number passed in.

    `market` decides which timezone "today" is measured in (`local_today`) —
    every caller already has a single-market ticker list at this point
    (`_domestic_candidates`/`_overseas_candidates` never mix them), so this
    is always unambiguous. Before this, a shared KST `get_clock().today()`
    meant an overseas position bought right before KST midnight could look
    "already rebalanced today" one moment and "not yet analyzed today" a
    few minutes later, purely because a KST calendar boundary fell in the
    middle of the still-open NASDAQ session — never because 5 real days, or
    the held-review interval, actually passed.

    The second return value (locked ticker -> its unlock date) exists
    specifically for `_eligible_universe`'s starvation warning: when this
    throttle locks out an entire candidate pool, knowing WHEN it reopens is
    exactly the piece of information that was missing during the
    2026-08-26 overseas starvation incident (the funnel log said
    `after_filters=0` every cycle with no way to tell "quiet market" from
    "every candidate locked until 8/30" apart). Costs nothing extra — it's
    the same get_last_decision_time result callers that don't need it
    already discarded."""
    today = local_today(market)
    kept: list[str] = []
    locked: dict[str, date] = {}
    for ticker in tickers:
        last = await get_last_decision_time(ticker)
        if last is None:
            kept.append(ticker)
            continue
        if (today - last.date()).days >= min_days:
            kept.append(ticker)
        else:
            locked[ticker] = last.date() + timedelta(days=min_days)
    return kept, locked


async def _drop_analyzed_within(tickers: list[str], min_days: int, market: Market) -> list[str]:
    """Thin wrapper over `_partition_analyzed_within` for the two callers
    (`_held_needing_review`, `_cash_freed_up_for_review`) that only ever
    need the survivors, not the unlock-date detail — see that function's
    own docstring for the full throttle semantics."""
    kept, _locked = await _partition_analyzed_within(tickers, min_days, market)
    return kept


async def _held_needing_review(tickers: list[str], market: Market) -> list[str]:
    """Cheap (no LLM, one shared account read + one DB read per ticker)
    pre-filter for currently-held trade_allowlist tickers the momentum ranking
    below didn't already select this cycle. Flags one for a full pipeline
    re-review on EITHER of two independent signals (union, not both
    required):

    1. Price has drifted within auto_trade_held_review_buffer_pct of its
       recorded stop-loss — see that setting's own docstring for the cost
       reasoning. A price already through the stop is the separate
       rule-based stop-loss loop's job (app.web.state.stop_loss_loop), not
       this one's.
    2. Among the auto_trade_top_loss_review_count largest unrealized
       losses (by KRW/USD amount, not return %) across every currently
       held ticker in `tickers` — see that setting's own docstring for why
       amount rather than %.

    "Held" is decided from the real KIS account (via kis_account_snapshot,
    already 35s-cached — one shared fetch here, not one per ticker), not
    the local paper ledger's own get_position. A position bought manually
    via the KIS app, or held from before this system tracked it, was
    never recorded in the ledger's `trades` table — get_position reads
    that back as shares=0 for such a ticker (see ledger.py), so it would
    silently never qualify for this review no matter how close its price
    got to any stop-loss, real or not. 2026-07-29 fix, alongside
    reconcile_trade_allowlist which is what gets such a ticker onto
    `tickers` in the first place.

    The flagged union is then throttled by `held_review_min_interval_days`
    (`_drop_analyzed_within`) — deliberately much shorter than
    `rebalance_interval_days` (a risk signal still needs closer attention
    than a routine rebalance), but not literally every single cycle: a
    ticker sitting near its stop-loss for several consecutive days used to
    get a full ~20-LLM-call re-underwrite every cycle it stayed there. The
    actual stop-loss cut is unaffected — app.web.state.stop_loss_loop is a
    separate, cheap, rule-based (no LLM) check that always runs regardless
    of this throttle."""
    from app.web import state  # local import: state.py imports this module's auto_trade_loop function

    summary, kis_positions = await state.kis_account_snapshot()
    if summary is None and not kis_positions:
        # kis_account_snapshot() collapses "no KIS broker configured" and
        # "the balance call failed" into the same (None, {}) — either way
        # the loop below silently sees zero held positions and skips this
        # entire risk re-review pass for every ticker, indistinguishable
        # from a day with nothing near its stop. state.py itself already
        # warns on the underlying KIS failure (2026-08-06), but that log
        # line never says WHAT the failure cost — this one does, so a
        # dropped review is visible rather than only inferable after the
        # fact from a missed stop.
        logger.warning(
            "_held_needing_review: KIS account snapshot unavailable — skipping risk review for %d ticker(s) this cycle",
            len(tickers),
        )
    stop_flagged = []
    losses: list[tuple[float, str]] = []
    for ticker in tickers:
        pos = kis_positions.get(ticker)
        shares = pos.get("quantity", 0) if pos else 0
        if shares <= 0:
            continue
        pnl_amount = pos.get("pnl_amount")
        if pnl_amount is not None and pnl_amount < 0:
            losses.append((pnl_amount, ticker))
        open_stop = await get_open_stop_loss(ticker)
        if open_stop is None:
            continue
        _, stop_loss = open_stop
        tick = latest(ticker)
        if tick is None or tick.price <= 0 or stop_loss <= 0:
            continue
        distance = (tick.price - stop_loss) / stop_loss
        if 0 <= distance <= settings.auto_trade_held_review_buffer_pct:
            stop_flagged.append(ticker)
    losses.sort(key=lambda pair: pair[0])  # most negative (largest loss) first
    loss_flagged = [ticker for _, ticker in losses[: settings.auto_trade_top_loss_review_count]]
    flagged = list(dict.fromkeys(stop_flagged + loss_flagged))  # union, de-duplicated
    return await _drop_analyzed_within(flagged, settings.held_review_min_interval_days, market)


# Ticker -> the local calendar date (per `local_today(market)`) it last
# fired an overnight_gap_risk alert for. Process-local, like alerts.py's own
# _last_fired_at cooldown — a closed market can span many auto_trade_loop
# cycles (hours overnight, days over a weekend), and re-firing the same
# warning every one of them would bury everything else in the alert feed
# the same way filter_new_alerts' cooldown exists to prevent for its own
# two alert kinds. A new calendar day (a fresh overnight/weekend window)
# warns again if the risk is still there.
# Keyed (account_mode, ticker) — see alerts.py's _last_fired_at note: the
# mode is a runtime-switchable contextvar, and a paper-mode warning must not
# suppress the real-money one for the same ticker on the same day.
_gap_warned_on: dict[tuple[str, str], date] = {}


def forget_ticker(ticker: str) -> None:
    """Drops `ticker`'s overnight-gap-warning cooldown — called from
    `app.web.state.remove_ticker_from_trade_allowlist` (2026-08-13) when a
    ticker leaves the trade allowlist. Without this, re-adding the same
    ticker within the same calendar day would inherit a cooldown from its
    PREVIOUS stint on the allowlist and silently suppress a warning that
    should fire fresh for the new stint, since `_gap_warned_on` only
    resets naturally on a calendar-day rollover, not on membership change.
    Drops the entry under every account_mode."""
    # Drops the entry under every account_mode (k[1] is the ticker) — same
    # reasoning as app.live.alerts.forget_ticker.
    for key in [k for k in _gap_warned_on if k[1] == ticker]:
        _gap_warned_on.pop(key, None)


async def _warn_overnight_gap_risk(tickers: list[str], market: Market) -> None:
    """Deterministic, code-only warning (same 'no LLM judgment' principle as
    app.live.alerts' own docstring) for a currently-held position sitting
    close enough to its stop-loss that an overnight/weekend gap — a move the
    market makes while closed, which nothing here can react to until it
    reopens — could jump straight through the stop before the mechanical
    stop-loss loop (app.web.state.stop_loss_loop) or exit_monitor ever gets
    a live tick to check.

    WARNING ONLY: reuses the exact distance-to-stop math as
    `_held_needing_review`'s stop_flagged branch, but against a wider buffer
    (`overnight_gap_warning_buffer_pct` — see that setting's own docstring
    for why overnight risk warrants an earlier trigger than the intraday
    full-review buffer) and never triggers a pipeline re-review or touches
    the position/size/order — sizing and holding decisions stay exactly
    where they already are (the mechanical stop, and the next cycle's AI
    review once the market reopens).

    Deliberately a no-op while `market` is still open — the day's price
    isn't final yet, and the live tick/pipeline path already covers an
    in-session approach to the stop. `tickers` should be the FULL
    trade_allowlist for `market`, not the momentum-filtered candidate pool:
    _domestic_candidates/_overseas_candidates already return [] once their
    market is closed, which is exactly when this needs to see held
    positions, so callers must pass the allowlist directly (see
    _run_auto_trade_cycle)."""
    if market == "overseas" and not settings.overseas_trading_enabled and not tickers:
        # Hard scope switch (app.config.overseas_trading_enabled) — with no
        # overseas allowlist entries either, there is nothing this leg could
        # warn about, so skip even the (cached, cheap) account-snapshot call
        # below. If the allowlist is non-empty despite the flag being off
        # (a config inconsistency), fall through — the snapshot's own
        # overseas_ok=False already makes the per-ticker loop below a no-op
        # for every overseas ticker (position lookups miss), which is the
        # graceful degradation this informational warning should have
        # anyway; this is not a safety gate, so there is nothing to enforce
        # more strictly here.
        return
    market_open = is_domestic_market_open() if market == "domestic" else is_overseas_market_open()
    if market_open:
        return
    from app.web import state  # local import — see module-level note above

    today = local_today(market)
    summary, kis_positions = await state.kis_account_snapshot()
    if summary is None and not kis_positions:
        # Same observability gap as _held_needing_review's identical
        # unpacking, and the same fix — see that function's comment for why
        # (None, {}) needs its own warning here rather than being silently
        # treated as "nothing held".
        logger.warning(
            "_warn_overnight_gap_risk: KIS account snapshot unavailable — skipping gap-risk check for %d ticker(s) this cycle",
            len(tickers),
        )
    for ticker in tickers:
        pos = kis_positions.get(ticker)
        shares = pos.get("quantity", 0) if pos else 0
        if shares <= 0:
            continue
        if _gap_warned_on.get((current_account_mode(), ticker)) == today:
            continue
        open_stop = await get_open_stop_loss(ticker)
        if open_stop is None:
            continue
        _, stop_loss = open_stop
        tick = latest(ticker)
        if tick is None or tick.price <= 0 or stop_loss <= 0:
            continue
        distance = (tick.price - stop_loss) / stop_loss
        if 0 <= distance <= settings.overnight_gap_warning_buffer_pct:
            # Marked BEFORE record_alert (not after) so a DB hiccup on the
            # persist can't cause the same warning to be recomputed and
            # retried every remaining cycle of the same closed-market
            # window — matching app.live.alerts' own "best-effort, degrade
            # quietly" convention rather than the retry semantics
            # pipeline_failure/cycle_failure use for a genuine outage.
            _gap_warned_on[(current_account_mode(), ticker)] = today
            await record_alert(
                Alert(
                    ticker=ticker,
                    kind="overnight_gap_risk",
                    message=(
                        f"{ticker} closed {distance:+.2%} above its stop-loss "
                        f"({tick.price:.2f} vs {stop_loss:.2f}) — a gap while the "
                        "market is shut could breach it before the next session"
                    ),
                    triggered_at=get_clock().now(),
                )
            )


async def _eligible_universe_detail(
    universe: list[str], trade_allowlist: list[str], market: Market, max_open: int
) -> tuple[list[str], dict[str, date]]:
    """Narrows `universe` down to what's actually eligible to occupy a
    momentum-ranking slot this cycle — applied BEFORE ranking, not as a
    filter on the ranking's own output.

    This is the fix for the 2026-08-26 overseas candidate-starvation
    incident. The rebalance-interval throttle (`rebalance_interval_days`)
    used to run AFTER `top_candidates` had already cut the universe down to
    `auto_trade_max_tickers_overseas` (=4) names. Because momentum ranking
    returns nearly the same order every cycle while the throttle is a
    multi-day sticky lock, once the top 4 names got locked, every following
    cycle re-ranked the SAME 4 names into the SAME top-4 slots and dropped
    all of them again — no ticker ranked 5th or lower could ever get a
    turn. Confirmed live: the overseas universe (41 tickers) was fully
    analyzed within one trading day (2026-08-25), which locked all 41 until
    2026-08-30; the following overseas session's momentum path then
    contributed exactly zero decisions, all night, with AXTI/CBRS/CRWV/WDC
    (last analyzed 2026-08-18~19, otherwise eligible) never once reaching
    the top-N slots that stayed permanently occupied by locked names. The
    session's only decision (MRVL) came from `_held_needing_review`, which
    is deliberately exempt from this throttle (see its own docstring) —
    that's the only reason the pipeline produced anything at all that
    night. Moving this throttle here, upstream of ranking, means a locked
    name simply never competes for a slot, so ranking naturally reaches
    whichever names are actually eligible.

    `exclude_already_analyzed_today` (this module's own caller-side
    same-day filter) already worked this way — applied to the universe
    before it's ever handed to ranking — which is exactly why THAT filter
    never exhibited this starvation failure mode. This function brings the
    two other cheap, deterministic (DB-only, no broker I/O) eligibility
    checks in line with that same convention:

    1. The rebalance-interval throttle itself (was `_drop_recently_
       rebalanced`) — separates the rebalance cadence (how often a ticker
       gets fully re-underwritten by the 4 analysts + debate + Portfolio
       Manager) from the monitoring cadence (how often risk is checked:
       _held_needing_review/stop_loss_loop still run every cycle, on their
       own much shorter throttle). See `rebalance_interval_days`'s own
       config comment for the real same-day flip-flop (379800.KS,
       2026-08-03) this throttle exists to prevent in the first place.
    2. `_drop_new_positions_over_cap` — once `max_open` distinct tickers
       are already open in this market, a first-time entry is doomed to be
       rejected at commit time by app.execution.portfolio_risk's own cap
       anyway, so it shouldn't occupy a ranking slot (or burn LLM budget)
       either.

    Deliberately NOT included here: `_drop_execution_blocked` (halt/VI) and
    `_drop_illiquid` (spread) — those are per-ticker BROKER calls, and
    running them over the full ~60-ticker universe every cycle would spend
    real KIS rate-limit budget (18 req/s live) on names that were never
    going to rank in the top N anyway. Those stay downstream of ranking, on
    an overfetched pool sized by `auto_trade_candidate_overfetch_multiplier`
    so a name they drop can be backfilled from the next rank down instead
    of leaving a slot permanently empty the way this exact shape did for
    the rebalance throttle.

    A ticker drifting toward its stop-loss or sitting on a top unrealized
    loss (`_held_needing_review`) or freshly unblocked by cash headroom
    (`_cash_freed_up_for_review`) is never subject to this function at
    all — both are merged into the candidate pool downstream, after this
    filter and after ranking, exactly as before. A risk signal deserves a
    fresh look regardless of the rebalance cadence, since it isn't a
    routine rebalance.

    Warns (not just logs at INFO) when the universe is nonzero but nothing
    survives — `universe > 0, eligible == 0` is the one funnel state that
    can only mean "every candidate in this market is currently locked out,"
    never a quiet market (a quiet market still leaves a nonzero eligible
    pool with everything filtered out further downstream, by ranking or the
    execution-quality filters). This warning, and the unlock dates in it,
    is what was missing throughout the 2026-08-26 incident — the existing
    funnel log only ever showed `after_filters=0`, indistinguishable from a
    genuinely quiet session until someone went and queried the database.

    Returns `(eligible, rebalance_locked)` — the second value is the
    rebalance-interval-locked subset (ticker -> unlock date), from
    `_partition_analyzed_within`, kept around (2026-09-09 §0단계 계측) so
    `app.live.screening_log` can record which of `eligible`'s exclusions
    were the 5-day throttle specifically, as opposed to the max_open cap —
    nothing in this function's own filtering reads it back."""
    eligible, locked = await _partition_analyzed_within(universe, settings.rebalance_interval_days, market)
    eligible = await _drop_new_positions_over_cap(eligible, trade_allowlist, max_open)
    if universe and not eligible:
        earliest_unlock = min(locked.values()) if locked else None
        logger.warning(
            "auto_trade_loop %s: entire candidate universe (%d ticker(s)) is ineligible this cycle — "
            "%d locked by rebalance_interval_days=%d (earliest unlock %s), the rest blocked by "
            "max_open_positions=%d — the momentum path contributes zero candidates this cycle; only "
            "the forced risk-review path (_held_needing_review/_cash_freed_up_for_review) can still "
            "produce one. This is NOT the same as a quiet market (which would still leave a nonzero "
            "eligible pool) — see the 2026-08-26 starvation incident this warning was added for",
            market, len(universe), len(locked), settings.rebalance_interval_days, earliest_unlock, max_open,
        )
    return eligible, locked


async def _eligible_universe(
    universe: list[str], trade_allowlist: list[str], market: Market, max_open: int
) -> list[str]:
    """Thin wrapper around `_eligible_universe_detail` for callers that only
    need the survivors, not the lock-detail map — same idiom as
    `_drop_analyzed_within` wrapping `_partition_analyzed_within`."""
    eligible, _locked = await _eligible_universe_detail(universe, trade_allowlist, market, max_open)
    return eligible


async def _drop_unexecutable(tickers: list[str], market: Market) -> list[str]:
    """2026-09-01 audit root cause #6 (first real-account KRX cycle, 200,000
    KRW equity): 31 of 41 candidates that cycle could not have been sized to
    even 1 share no matter what the AI decided, and 17 of 41 had a 1-share
    price above the account's entire cash — yet every one still burned the
    full ~20-LLM-call multi-agent pipeline. Drops a candidate whose
    `app.domain.executability.best_case_buyable_shares` — the SAME arithmetic
    kernels `app.execution.ledger` sizes a real trade with — comes out to 0
    under the most generous reading of every input (max confidence BUY, a
    stop-loss sitting exactly on the ATR floor, every other real-execution
    constraint ignored). A ticker this drops genuinely cannot be bought this
    cycle; see that module's own docstring for why 0 here means 0 for real.

    Placed BEFORE momentum ranking (`_eligible_universe`'s own sibling, not
    after `_drop_illiquid`) — this filter's condition is even STICKIER than
    the rebalance-interval throttle `_eligible_universe` already fixed for
    the 2026-08-26 starvation incident: price/ATR/account-equity barely move
    day to day, so an unaffordable name sitting at the top of the momentum
    ranking would occupy a budget slot every single cycle, and
    `auto_trade_candidate_overfetch_multiplier`'s 2x headroom is nowhere
    near enough to guarantee backfill if more than half the overfetched pool
    turns out unaffordable (measured: it was ~75% of the 2026-09-01 sample).
    Running this filter here, upstream of ranking, means ranking only ever
    sees names that could actually be bought, so the next-ranked affordable
    name naturally takes a dropped one's place — no reliance on overfetch at
    all. `_held_needing_review`/`_cash_freed_up_for_review` merge into the
    candidate pool AFTER ranking (see `_select_market_candidates`), so they
    never pass through this function — a risk-review ticker is exempt for
    the same reason it's exempt from the rebalance throttle.

    Never stricter than the real gate:
    - The kill switch (`auto_trade_skip_unexecutable_candidates=False`)
      short-circuits before any account lookup at all.
    - A currently-held ticker is always kept, regardless of price — this
      filter is about NEW exposure, not managing an existing position.
    - `cached_account_snapshot` failing (`AccountUnavailable`) or returning
      non-positive equity skips this filter ENTIRELY for the cycle rather
      than computing affordability against a fabricated/fallback number —
      same "fail-closed, no invented equity basis" policy as
      `app.execution.ledger.EquityBasisUnavailable`. Deliberately calls
      `cached_account_snapshot` (one read-only, TTL-cached call for the
      whole market) rather than resolving equity per ticker — a per-ticker
      real-balance lookup outside `cached_balance_fetches()`'s scope would
      fire one KIS ledger TR per candidate, reproducing the EGW00215
      rate-limit incident this same audit cycle already fixed once
      (see CHANGELOG #183/#184).
    - A missing/failed price quote for a ticker keeps that ticker (same
      best-effort convention as `_drop_illiquid`/`_drop_execution_blocked`).
    - A missing/failed ATR quote is the ONE deliberate exception to "never
      stricter than the real gate": with
      `settings.auto_trade_require_atr_for_new_entry` on (default True), it
      now DROPS the candidate instead of treating the risk cap as uncapped
      — 2026-09-02 (decision id=1169, 950260.KS). This is intentional, not
      a regression: `app.execution.ledger.rebalance_to_target` refuses the
      exact same new entry under the exact same predicate
      (`app.domain.executability.blocks_new_entry_without_atr`) and the
      exact same setting, so a candidate this function let through would
      just burn the ~20-LLM-call pipeline and then be refused at execution
      anyway. A future reader should not "fix" this back to keeping the
      candidate without also removing the ledger-side gate.
    - 2026-09-02 (screening-improvement-research #3 follow-up):
      `_domestic_price_ceiling` now (when its toggle is on) also asks KIS
      for a price-bounded ranking UPSTREAM of this function, via
      `FID_INPUT_PRICE_2` on all four KRX discovery axes. That is a
      HEURISTIC pre-filter over rows this function never even sees,
      computed from ONE assumed ATR% applied to the whole ranking before
      any individual candidate's real ATR is known. It does NOT subsume
      this function, and this function remains the only EXACT answer —
      only here, with each surviving candidate's REAL price and REAL ATR
      in hand, can "affordable" be answered precisely rather than
      assumed. A future reader must not "simplify" this function away
      just because an upstream ceiling now exists; the ceiling can be
      wrong in the account's favor (see `_domestic_price_ceiling`'s
      slack_multiple discussion) in a way only this function catches.
    """
    if not settings.auto_trade_skip_unexecutable_candidates or not tickers:
        return tickers

    from app.execution.portfolio_risk import AccountUnavailable, cached_account_snapshot
    from app.execution.sizing import vol_adjusted_allocation_for
    from app.dataflows.kis_provider import get_atr_pct, get_last_close
    from app.domain.executability import (
        ExecutabilityInputs,
        SizingLimits,
        best_case_buyable_shares,
        blocks_new_entry_without_atr,
    )
    from app.state import PortfolioRating

    try:
        snapshot = await cached_account_snapshot(market)
    except AccountUnavailable:
        logger.warning(
            "auto_trade_loop %s: real account snapshot unavailable — skipping the purchasability "
            "filter entirely this cycle (not computing affordability against a fabricated balance), "
            "keeping all %d candidate(s)",
            market, len(tickers),
        )
        return tickers
    equity = snapshot.total_value
    if equity <= 0:
        return tickers
    cash = max(equity - sum(snapshot.per_ticker_value.values()), 0.0)
    held = {t for t, v in snapshot.per_ticker_value.items() if v > 0}

    # Same values app.execution.ledger sizes a real trade with — read via
    # risk_rules.rule() (user/account-mode DB overrides), never settings.
    # directly, so this filter can't silently diverge from the ledger for a
    # user who has overridden either rule.
    max_risk_pct_of_equity = risk_rules.rule("sizing_max_risk_pct_of_equity")
    min_stop_atr_multiple = risk_rules.rule("sizing_min_stop_atr_multiple")

    candidates = [t for t in tickers if t not in held]
    if not candidates:
        return tickers

    async def _quote(ticker: str) -> tuple[str, float | None, float | None]:
        try:
            price = await get_last_close(ticker)
        except Exception:
            logger.warning("auto_trade_loop %s: price lookup failed for %s — keeping as a candidate", market, ticker)
            return ticker, None, None
        try:
            atr = await get_atr_pct(ticker)
        except Exception:
            # NOT "risk cap uncapped" any more when
            # auto_trade_require_atr_for_new_entry is on — see this
            # function's own docstring for why a missing ATR now drops the
            # candidate outright (2026-09-02, id=1169). Still returned as
            # None rather than raised here: the drop decision itself
            # happens below, uniformly for both a fetch failure and a
            # thin-history None, not inside this helper.
            logger.warning("auto_trade_loop %s: ATR lookup failed for %s", market, ticker)
            atr = None
        return ticker, price, atr

    quotes = await asyncio.gather(*(_quote(t) for t in candidates))

    dropped: set[str] = set()
    for ticker, price, atr in quotes:
        if price is None or price <= 0:
            continue  # never stricter than the real gate — a failed quote keeps the candidate
        if settings.auto_trade_require_atr_for_new_entry and blocks_new_entry_without_atr(
            atr, current_shares=0
        ):
            # current_shares=0 is correct unconditionally here: `candidates`
            # (built above) already excludes every ticker in `held`, so every
            # entry reaching this loop is by construction a new-entry
            # candidate.
            logger.info(
                "auto_trade_loop %s: dropping %s — no usable 14-day ATR (%s), so a new entry would be "
                "sized at the unscaled base allocation with no stop-loss floor; refusing before the "
                "~20-LLM-call pipeline (settings.auto_trade_require_atr_for_new_entry)",
                market, ticker, atr,
            )
            dropped.add(ticker)
            continue
        allocation_fraction = vol_adjusted_allocation_for(PortfolioRating.BUY, atr)
        verdict_inputs = ExecutabilityInputs(price=price, atr_pct=atr, equity=equity, cash=cash)
        verdict_limits = SizingLimits(
            # BUY's own base allocation is 1.0 (see sizing._BASE_ALLOCATION),
            # so vol_adjusted_allocation_for(BUY, ...) never returns None —
            # the `or 1.0` is defensive only.
            max_allocation_fraction=allocation_fraction or 1.0,
            max_risk_pct_of_equity=max_risk_pct_of_equity,
            min_stop_atr_multiple=min_stop_atr_multiple,
        )
        verdict = best_case_buyable_shares(verdict_inputs, verdict_limits)
        if verdict.max_buyable_shares <= 0:
            logger.info("auto_trade_loop %s: dropping %s — %s", market, ticker, verdict.reason)
            dropped.add(ticker)

    return [t for t in tickers if t not in dropped]


async def _cash_freed_up_for_review(trade_allowlist: list[str], market: Market) -> list[str]:
    """Cheap (no LLM) pre-filter catching the one case the rebalance-interval
    throttle (`_eligible_universe`) can't distinguish on its own: a ticker whose last decision
    WANTED to buy/add but got capped to a 0 net change by the portfolio cash
    floor (app.execution.portfolio_risk.apply_portfolio_caps) looks
    identical, from get_last_decision_time's point of view, to one that was
    genuinely decided Hold — both just look "already analyzed recently," so
    it stays locked out of the analysis pool for the full
    rebalance_interval_days even after a later sale frees up cash that would
    now let it through. 2026-08-05 design discussion: only cash-floor-capped
    tickers get this bypass — a 1%-risk-cap, max-open-positions, or
    concentration/correlation cap block means something OTHER than cash was
    the constraint, so more cash wouldn't actually change the outcome, and
    those stay on the normal throttle.

    Re-checks the SAME cash-floor math live instead of storing the cash/
    equity level at block-time for a before/after comparison — if there's
    ANY headroom above the floor right now, that alone is enough evidence
    something changed (the situation that produced zero headroom before
    couldn't otherwise have improved), so no stored baseline is needed.
    Still throttled by held_review_min_interval_days (same as
    _held_needing_review's own risk-driven bypass) so a ticker that stays
    unblocked for days doesn't get a fresh ~20-LLM-call pass every single
    cycle it remains so — this only controls how soon a bypass CAN happen,
    not how often it's re-checked whether headroom exists (that check itself
    is one cached account snapshot, not an LLM call)."""
    capped = await tickers_cash_floor_capped(trade_allowlist)
    if not capped:
        return []
    from app.execution import market_regime
    from app.execution.portfolio_risk import AccountUnavailable, cached_account_snapshot

    try:
        # Cost-saving pre-filter, not a gate (docstring above) — the short
        # read-only snapshot cache is what this already assumed ("one cached
        # account snapshot, not an LLM call").
        snapshot = await cached_account_snapshot(market)
    except AccountUnavailable:
        return []
    if snapshot.total_value <= 0:
        return []
    cash_before = snapshot.total_value - sum(snapshot.per_ticker_value.values())
    cash_floor_pct = (
        await market_regime.domestic_cash_floor_pct()
        if market == "domestic"
        else await market_regime.overseas_cash_floor_pct()
    )
    if cash_before <= cash_floor_pct * snapshot.total_value:
        return []  # still no headroom — nothing has actually changed
    return await _drop_analyzed_within(capped, settings.held_review_min_interval_days, market)


async def _drop_new_positions_over_cap(tickers: list[str], trade_allowlist: list[str], max_open: int) -> list[str]:
    """Cost-saving pre-filter (mirrors `_drop_execution_blocked`'s own
    reasoning): once `max_open` distinct tickers are already open in this
    market, drops any candidate NOT already among them — otherwise a
    first-time entry that's doomed to be blocked anyway burns a full
    ~20-LLM-call pipeline only to be rejected by app.execution.
    portfolio_risk's own max_open_positions_domestic/_overseas cap at commit
    time. Never drops an already-held ticker (adding to an existing position
    doesn't increase the open-position count, and a held ticker may still
    need review regardless of this cap).

    `trade_allowlist`, not `tickers`, is what the open-position count is computed
    from — a currently-held ticker might not re-qualify for `tickers`
    (momentum ranking) this cycle but still counts as open."""
    from app.web import state  # local import — see module-level note above

    _, kis_positions = await state.kis_account_snapshot()

    def _held(ticker: str) -> bool:
        return (kis_positions.get(ticker) or {}).get("quantity", 0) > 0

    open_count = sum(1 for t in trade_allowlist if _held(t))
    if open_count < max_open:
        return tickers
    return [t for t in tickers if _held(t)]


async def _drop_execution_blocked(broker, tickers: list[str]) -> list[str]:
    """Drops any domestic candidate currently trading-halted or an
    administrative issue, or under a VI — before it ever reaches the 4
    analysts + debate + decision stages. Uses `pre_screening_blockers`
    (not the full `execution_blockers`) deliberately: these candidates
    are typically NOT yet on the trade_allowlist/real-time feed at this point
    (see that function's docstring), so the circuit-breaker gate's
    fail-closed-on-"unknown" behavior would otherwise drop nearly every
    fresh candidate for the wrong reason. The circuit-breaker gate itself
    isn't skipped forever — it still runs later in
    `pipeline._execution_status_context` and for real at order time.
    Until this filter existed, a blocked ticker still burned a full LLM
    analysis every cycle only to be silently held at the end. Domestic
    only — no KIS overseas VI equivalent. Best-effort per ticker: a
    check that errors doesn't drop the candidate (same "proceed, KIS's
    own order call is the final backstop" convention
    `pre_screening_blockers` already uses internally) — this is a
    cost-saving pre-filter, not a new safety gate, so it must never be
    stricter than the real one. Note the administrative-issue/halt check
    itself (not VI) is a no-op on a paper account (KIS rejects that TR
    with EGW02006) — see `_stock_halt_block`'s docstring — so this only
    fully filters once real trading is enabled; VI still works on paper.

    A ticker with no KRX code (a US symbol reaching a caller that only ever
    passes domestic ones today) is KEPT, unchecked — the same direction this
    function already fails in for an errored check, and the only correct one
    for a filter documented as never stricter than the real gate. Resolved
    via `krx_code_from_ticker`, not `ticker[:-3]`: the bare slice quietly
    turns "AAPL" into "A" and would then ask KIS about a code that isn't
    this ticker at all, rather than declining to answer.
    KRX 코드가 없는 종목은 검사를 건너뛰고 후보로 유지 — 문자열을 잘라 쓰면
    'AAPL'이 'A'가 되어 엉뚱한 종목을 조회하게 됨."""
    kept = []
    for ticker in tickers:
        krx_code = krx_code_from_ticker(ticker)
        if krx_code is None:
            logger.warning(
                "auto_trade_loop: %s has no KRX code — skipping the domestic halt/VI pre-filter "
                "for it and keeping it as a candidate", ticker,
            )
            kept.append(ticker)
            continue
        try:
            blockers = await pre_screening_blockers(broker, krx_code, ticker)
        except Exception:
            logger.warning(
                "pre_screening_blockers check failed for %s — keeping as a candidate", ticker, exc_info=True
            )
            blockers = []
        if blockers:
            logger.info("auto_trade_loop: dropping %s from candidates — %s", ticker, "; ".join(blockers))
            continue
        kept.append(ticker)
    return kept


async def _drop_designated(broker, tickers: list[str]) -> list[str]:
    """Drops any domestic candidate currently carrying a KRX market-warning/
    short-overheat/liquidation designation (§4-5, 2026-09-08 screening
    audit follow-up) — before it ever reaches the 4 analysts + debate +
    decision stages. A separate filter from `_drop_execution_blocked`
    (not folded into `pre_screening_blockers`) specifically so callers can
    exempt the forced held-position review set — see this function's own
    call site in `_select_market_candidates` for the exemption, and
    `app.execution.kis_bridge.designation_block`'s docstring for the full
    reasoning (a position just designated administrative is exactly the
    moment the forced review exists to catch, not a candidate to
    silently drop before the AI ever sees it).

    Domestic only — these are KRX-specific designations. Best-effort per
    ticker, same never-stricter-than-necessary convention as
    `_drop_execution_blocked`/`_drop_illiquid`: a check that errors keeps
    the candidate rather than dropping it."""
    kept = []
    for ticker in tickers:
        krx_code = krx_code_from_ticker(ticker)
        if krx_code is None:
            kept.append(ticker)
            continue
        try:
            blocked = await designation_block(broker, krx_code, ticker)
        except Exception:
            logger.warning(
                "designation_block check failed for %s — keeping as a candidate", ticker, exc_info=True
            )
            blocked = None
        if blocked:
            logger.info("auto_trade_loop: dropping %s from candidates — %s", ticker, blocked)
            continue
        kept.append(ticker)
    return kept


async def _drop_illiquid(broker, tickers: list[str]) -> list[str]:
    """Drops any candidate (domestic or overseas) whose current bid-ask
    spread exceeds `settings.max_spread_pct` — before it ever reaches the
    4 analysts + debate + decision stages. screening-method-improvement-
    ideas item #3: a thin, event-driven volume spike can rank high on pure
    momentum/turnover without being genuinely liquid. Best-effort per
    ticker via `spread_blocker` (a quote-fetch failure or unusable quote
    proceeds rather than blocking) — same cost-saving-pre-filter,
    never-stricter-than-necessary convention as `_drop_execution_blocked`."""
    kept = []
    for ticker in tickers:
        try:
            blocked = await spread_blocker(broker, ticker)
        except Exception:
            logger.warning("spread_blocker check failed for %s — keeping as a candidate", ticker, exc_info=True)
            blocked = None
        if blocked:
            logger.info("auto_trade_loop: dropping %s from candidates — %s", ticker, blocked)
            continue
        kept.append(ticker)
    return kept


async def _ensure_subscribed(tickers: list[str]) -> None:
    """Best-effort: subscribe every candidate to the realtime feed as soon
    as it's selected for this cycle, instead of only reactively once a BUY
    decision auto-adds it to the trade allowlist (app.web.state.
    add_ticker_to_trade_allowlist's own add_code() call).

    Real incident this addresses (2026-07-30): a momentum candidate not yet
    on the trade allowlist has unknown circuit-breaker status
    (app.live.kis_market_status.is_circuit_breaker_active), which
    app.execution.kis_bridge's execution gate fails closed on — so a BUY
    decided after the ~9-minute analyst+debate+decision pipeline could
    still get blocked at execution simply because its subscription request
    (sent only once the BUY was already decided) hadn't been acknowledged
    yet. Subscribing here instead, at candidate-selection time, gives the
    subscription the full length of that pipeline run to confirm before
    execution ever needs it — the pipeline itself isn't slowed down or
    reordered, this just fires the subscribe request earlier and doesn't
    wait on it. Never touches the trade allowlist itself (add_code doesn't
    require allowlist membership) — a candidate that ends up Hold stays
    subscribed for no cost; that's the existing "quiet code, no forever-Buy
    guarantee" convention `RealtimeFeed.add_code` already has.
    후보로 선정된 시점에 미리 구독 요청 — BUY 판단까지 기다리지 않아서, 분석하는
    9분 동안 구독이 확정될 시간을 벌어줌(분석 자체를 늦추거나 순서를 바꾸지 않음).

    Subscribes with `cb_only=True` (2026-08-28) — a screening candidate only
    needs the circuit-breaker/VI answer this whole function exists to
    guarantee before its BUY decision reaches execution; it does NOT need a
    live trade tick (price has a REST fallback, see RealtimeFeed.add_code's
    docstring), so spending 1 registration instead of 2 here is what makes
    the "every candidate every cycle" pre-subscribe pattern this function
    already uses affordable at all under the 40-registration budget. Before
    this, every cycle's candidate batch could exhaust the whole (fixed,
    all-2TR) budget on its own, permanently starving every later candidate
    and even held-position resubscribes — the 2026-08-28 domestic AI audit
    traced 6 of 11 same-day Buy decisions being blocked on exactly this. A
    code already subscribed at the full 2TR tier (held position, trade
    allowlist member) is unaffected: `add_code`'s existing `code in
    self.codes` early-return means `cb_only=True` here can only ever ADD a
    candidate that wasn't already covered, never downgrade one that was."""
    from app.web import state  # local import — avoid circular import, see run_cycle_now's own note

    # 2026-09-04 (3rd recurrence, see state._candidate_cb_codes's own
    # comment): register every candidate in the persistent registry BEFORE
    # the `_market_feed is None` guard below and regardless of whether
    # add_code succeeds. A live RealtimeFeed's cb_only_codes is in-memory
    # only — reconcile_feed_tier_promotions's stop/ensure rebuild (or simply
    # no feed existing yet this cycle) throws it away, and only this
    # registry survives to let the NEXT _prioritized_krx_codes build re-claim
    # a cb-only slot for the candidate.
    for ticker in tickers:
        code = krx_code_from_ticker(ticker)
        if code is not None:
            state.note_candidate_cb_code(code)

    if state._market_feed is None:
        return
    # 2026-10-02: budget-exhausted candidates are collected first, alerted
    # only after one capacity rebuild attempt. add_code has no unsubscribe
    # protocol, so cb_only candidate slots only ever accumulate intraday
    # (40/40 within a few cycles — 10-01 13/20 and 10-02 6/20 Buys blocked on
    # CB unknown). A full stop/ensure rebuild re-applies the candidate TTL and
    # the reserved-8 priority in _prioritized_krx_codes, freeing the slots.
    # 한도 소진 후보를 모아 두었다가, 연결 재구성 1회 후에도 없는 것만 알림.
    capped: list[tuple[str, str]] = []  # (ticker, code)
    for ticker in tickers:
        code = krx_code_from_ticker(ticker)
        if code is None:
            continue
        try:
            subscribed = await state._market_feed.add_code(code, cb_only=True)
        except Exception:
            logger.warning("could not pre-subscribe %s to the realtime feed", ticker, exc_info=True)
            continue
        if not subscribed:
            capped.append((ticker, code))
    if not capped:
        return
    try:
        await state.rebuild_market_feed_for_capacity(
            reason=f"{len(capped)} screening candidate(s) could not be subscribed (registration budget spent)"
        )
    except Exception:
        logger.warning("capacity rebuild of the realtime feed failed", exc_info=True)
    feed = state._market_feed
    for ticker, code in capped:
        if feed is not None:
            try:
                if await feed.add_code(code, cb_only=True):
                    continue
            except Exception:
                logger.warning("could not re-subscribe %s after capacity rebuild", ticker, exc_info=True)
        # Still not on the feed after the rebuild (or the rebuild was skipped
        # by its interval guard) — this candidate's circuit-breaker state will
        # read as unknown (fail-closed). A plain logger.warning already existed
        # here once (2026-08-05) and went unnoticed for three weeks until the
        # 2026-08-28 audit traced a day's worth of blocked Buys back to it —
        # alerting instead surfaces it where a person actually looks.
        max_reg = feed.MAX_REGISTRATIONS if feed is not None else state.RealtimeFeed.MAX_REGISTRATIONS
        for alert in filter_new_alerts([Alert(
            ticker=ticker, kind="realtime_subscription_capped",
            message=(
                f"{ticker}: KIS 실시간 등록 {max_reg}건 소진 — "
                "서킷브레이커 상태가 unknown으로 남아 매수 판단이 실행 단계에서 차단될 수 있습니다"
            ),
            triggered_at=get_clock().now(),
        )]):
            await record_alert(alert)


def _axis_map(*pairs: tuple[str, list[str]]) -> dict[str, list[str]]:
    """Ticker -> every axis label whose candidate list contains it (§4-4).
    Shared by _domestic_candidates/_overseas_candidates rather than each
    duplicating the loop — this module was bitten once already
    (2026-08-26) by the same starvation bug existing in both markets'
    independently-duplicated bodies, see _select_market_candidates' own
    docstring."""
    out: dict[str, list[str]] = {}
    for axis_label, axis_tickers in pairs:
        for ticker in axis_tickers:
            out.setdefault(ticker, []).append(axis_label)
    return out


async def _select_market_candidates(
    broker,
    market: Market,
    universe: list[str],
    trade_allowlist: list[str],
    *,
    max_tickers: int,
    lookback_days: int,
    use_volatility_adjustment: bool,
    use_correlation_dedup: bool,
    max_open: int,
    axes_by_ticker: dict[str, list[str]] | None = None,
) -> list[str]:
    """Common body of `_domestic_candidates`/`_overseas_candidates` — the two
    were fully duplicated (identical shape, only settings/helper-function
    names differed) until 2026-08-26, and that duplication is exactly why
    the candidate-starvation bug fixed here existed in both without anyone
    noticing the overseas copy first: domestic's faster universe turnover
    happened to keep its own copy of the same bug from ever biting in
    practice. Only `market` differs at the call boundary now; the two
    domestic-only steps below (_ensure_subscribed, _drop_execution_blocked —
    KIS has no overseas VI/halt-check TR equivalent) branch on it inline,
    the same way every other market-conditional helper in this module
    already does, rather than adding a new flag/callback just for this.

    Two-stage filtering, in this specific order — see `_eligible_universe`'s
    own docstring for the full incident history behind stage 1 existing at
    all:

    1. Cheap, deterministic, DB-only eligibility (`_eligible_universe`) —
       BEFORE ranking. A ticker that's locked out (rebalance-interval
       throttle) or doomed to be rejected at commit time anyway
       (max-open-positions cap) never occupies a ranking slot in the first
       place, so it can never starve one out from under an eligible
       ticker ranked lower.
    2. Momentum ranking (`top_candidates`), asked for `max_tickers *
       auto_trade_candidate_overfetch_multiplier` names rather than exactly
       `max_tickers` — the overfetch is spent below, backfilling for
       whatever the broker-I/O execution-quality filters (halt/VI, spread)
       drop. `top_candidates` itself (app.chat.screening) is untouched:
       app.backtest.portfolio_harness/screening_sweep call the same
       function with no rebalance-throttle concept at all, so "rank, then
       cut to top_n" is correct behavior there — the bug was in this
       caller's filter ORDER, never in top_candidates' own contract.

    The forced risk-review paths (`_held_needing_review`,
    `_cash_freed_up_for_review`) are exempt from BOTH stage 1 and the
    `max_tickers` budget below — same as before this refactor: a ticker
    drifting toward its stop-loss or sitting on a top unrealized loss (or
    freshly unblocked by cash headroom) deserves a fresh look regardless of
    the rebalance cadence or the momentum budget, since that's a risk
    signal, not a routine rebalance. They're also exempt from the spread
    filter (a widened spread is often exactly the stress signal this
    forced re-review exists to catch, not a reason to skip it) but NOT from
    `_drop_execution_blocked` (domestic only) — a halted/VI'd ticker
    genuinely can't be traded regardless of who wants to review it; the
    mechanical stop-loss/exit_monitor protects the position either way, so
    that only affects whether the AI gets a look before the mechanical exit
    fires.

    2026-09-01 (audit #6): `_drop_unexecutable` runs here too — right after
    `_eligible_universe`, before ranking, for the same starvation reason as
    stage 1 (see that function's own docstring: an unaffordable name is
    even stickier than the rebalance throttle). The whole selection is
    wrapped in ONE `candle_run_cache()` scope so `_drop_unexecutable`'s
    price/ATR lookups and `top_candidates`' own momentum lookups (which
    fetch the same candles) share one KIS fetch per ticker instead of two —
    this filter's extra broker calls are a net WASH, not an added cost.

    `axes_by_ticker` (§4-1/§4-4, 2026-09-08): which discovery axis admitted
    each ticker into `universe` this cycle, for app.live.screening_log's
    snapshot only — no filtering decision anywhere in this function reads
    it. `None` (the default) is equivalent to an empty map, for tests/
    callers that don't care about axis provenance."""
    from app.dataflows.kis_provider import candle_run_cache

    with candle_run_cache():
        eligible, rebalance_locked = await _eligible_universe_detail(universe, trade_allowlist, market, max_open)
        executable = await _drop_unexecutable(eligible, market)
        if eligible and not executable:
            logger.warning(
                "auto_trade_loop %s: the purchasability filter dropped every eligible candidate "
                "(%d ticker(s)) — nothing in this universe can be sized to even 1 share against the "
                "current real account equity. The momentum path contributes zero candidates this "
                "cycle (the forced risk-review path can still produce one) — this is an account-size "
                "condition, not a pipeline malfunction (audit #6)",
                market, len(eligible),
            )
            for alert in filter_new_alerts([
                Alert(
                    ticker="SYSTEM",
                    kind=f"account_too_small_{market}",
                    message=(
                        f"{market}: {len(eligible)}개 후보 전부 1주도 매수 불가 — 실계좌 자산 기준으로 "
                        f"이번 사이클 신규 진입이 구조적으로 불가능합니다. 예수금을 늘리거나 저가 종목 "
                        f"유니버스를 검토하세요."
                    ),
                    triggered_at=get_clock().now(),
                )
            ]):
                await record_alert(alert)
        top_n_requested = max_tickers * settings.auto_trade_candidate_overfetch_multiplier
        # 2026-09-09 §0단계 계측(screening-cycle-rebalance-lock-analysis): score
        # the universe tickers that fell out of `executable` (rebalance-lock,
        # max_open cap, or unaffordable) too, PURELY for observation —
        # screening_pass keeps these in a separate dict (ScreeningPass.
        # observed) that never reaches _ranked_and_deduped, so `ranked`/
        # `final` below are computed exactly as before this change no matter
        # what's in observe_only. Answers "would this ticker have made top_n
        # if it hadn't been locked out" — see app.audit.screening_ic.
        executable_set = frozenset(executable)
        observe_only = (
            [t for t in universe if t not in executable_set]
            if settings.screening_snapshot_observational_scoring_enabled
            else []
        )
        screening = await screening_pass(
            executable,
            top_n=top_n_requested,
            lookback_days=lookback_days,
            use_volatility_adjustment=use_volatility_adjustment,
            use_correlation_dedup=use_correlation_dedup,
            observe_only=observe_only,
        )
        ranked = [t for t, _ in screening.ranked]
    # Captured here (not re-derived from `ranked` later) because `ranked`
    # doesn't get reassigned again before the funnel log below — this is
    # the funnel instrumentation's look at the ranked-pool size. NOTE this
    # is now the OVERFETCHED pool size (up to max_tickers * multiplier),
    # not the final momentum budget — see `after_filters_count` below for
    # the number actually spent on analysis.
    momentum_top_count = len(ranked)
    held_review = await _held_needing_review(trade_allowlist, market)
    cash_freed = await _cash_freed_up_for_review(trade_allowlist, market)
    combined = list(dict.fromkeys(ranked + held_review + cash_freed))
    # Computed here (not at the budget cut below) because `_drop_designated`
    # needs the same exemption immediately below it — a position under
    # forced risk-review must never be dropped by a cost-saving pre-filter,
    # only ever by a real execution-time gate (_drop_execution_blocked).
    forced = frozenset(held_review) | frozenset(cash_freed)
    if market == "domestic":
        # Fire the subscribe request now, before the ~9-minute analysis
        # pipeline even starts — see _ensure_subscribed's own docstring for
        # the incident this avoids. Best-effort, never blocks/filters
        # candidates on it.
        await _ensure_subscribed(combined)
        execution_ok = await _drop_execution_blocked(broker, combined)
        # §4-5 (2026-09-08): KRX market-warning/short-overheat/liquidation
        # designations, exempt for the forced risk-review set — see
        # _drop_designated's own docstring for why (a position just
        # designated administrative is exactly what that review exists to
        # catch, not a candidate to silently drop before the AI sees it).
        not_designated = await _drop_designated(broker, [t for t in execution_ok if t not in forced])
        execution_ok = list(dict.fromkeys(not_designated + [t for t in execution_ok if t in forced]))
    else:
        execution_ok = combined
    held_set = frozenset(held_review)
    liquid = await _drop_illiquid(broker, [t for t in execution_ok if t not in held_set])
    survivors = list(dict.fromkeys(liquid + [t for t in execution_ok if t in held_set]))
    # Budget cut — spend the overfetched ranking pool's survivors down to
    # `max_tickers`, now that the broker-I/O filters have already had their
    # chance to drop a name and let the next rank down backfill its slot.
    # The forced risk-review tickers are exempt from this cut too (same as
    # before this refactor, when they were appended after the budget cut
    # unconditionally) — a risk signal must never be squeezed out by the
    # momentum budget.
    momentum_survivors = [t for t in survivors if t not in forced]
    kept_momentum = frozenset(momentum_survivors[:max_tickers])
    final = [t for t in survivors if t in forced or t in kept_momentum]
    after_filters_count = len(kept_momentum)
    # Funnel instrumentation — lets a data-source failure (universe itself
    # empty) be told apart from a genuinely quiet market (universe nonzero,
    # everything filtered out downstream), and now also from the
    # 2026-08-26 starvation failure mode (universe nonzero, eligible
    # empty — see _eligible_universe's own WARNING for that case). Always
    # INFO; the empty-universe case below and the empty-eligible case in
    # _eligible_universe each get their own WARNING. Purely observational —
    # no filtering decision reads these counts.
    logger.info(
        "auto_trade_loop %s funnel: universe=%d -> eligible=%d -> executable=%d -> momentum_top=%d -> "
        "after_filters=%d (+held=%d +cash_freed=%d) -> final=%d (observed=%d, rebalance_locked=%d)",
        market,
        len(universe), len(eligible), len(executable), momentum_top_count, after_filters_count,
        len(held_review), len(cash_freed), len(final), len(screening.observed), len(rebalance_locked),
    )
    # §4-1 live screening instrumentation (2026-09-08) — persist THIS cycle's
    # whole cross-section (scores, ranks, axis provenance, and which names
    # were actually selected) so a later batch can join it against realized
    # forward returns and finally answer whether this screening produces any
    # alpha live. Purely observational: nothing above or below reads it, and
    # it degrades quietly (see app.live.screening_log's own docstring) so a
    # DB hiccup can never affect candidate selection.
    # 이 사이클의 전체 후보/점수/랭크/축을 저장 — N일 후 실현 수익률과 조인해서
    # 라이브 RankIC을 계산하기 위한 것이며, 매매 판단에는 전혀 영향 없음.
    try:
        from app.live.screening_log import ScreeningSnapshotContext, record_screening_snapshot

        await record_screening_snapshot(
            ScreeningSnapshotContext(
                market=market,
                captured_at=get_clock().now(),
                trade_date=local_today(market),
                lookback_days=lookback_days,
                volatility_adjusted=use_volatility_adjustment,
                correlation_dedup=use_correlation_dedup,
                top_n_requested=top_n_requested,
                max_tickers=max_tickers,
            ),
            universe=universe,
            eligible=eligible,
            executable=executable,
            scored=screening.scored,
            score_key=screening.score_key,
            ranked=screening.ranked,
            final=final,
            held_review=held_review,
            cash_freed=cash_freed,
            axes_by_ticker=axes_by_ticker or {},
            observed=screening.observed,
            rebalance_locked=rebalance_locked,
        )
    except Exception:
        logger.warning("screening snapshot logging failed this cycle — continuing", exc_info=True)
    if not universe:
        logger.warning(
            "auto_trade_loop %s: candidate universe is empty (0 tickers from any enabled discovery axis "
            "this cycle) — likely a data-source failure, not a quiet market (a quiet market still has a "
            "nonzero universe with everything filtered out downstream)",
            market,
        )
    return final


async def _domestic_price_ceiling() -> float | None:
    """The account-derived KRX price ceiling (`FID_INPUT_PRICE_2`, via
    `app.domain.executability.affordable_price_ceiling`) that
    `_domestic_candidates` threads into all four discovery axes below —
    2026-09-02 (screening-improvement-research #3 follow-up), the
    structural fix for a ₩2,000,000 corporate account raising
    `account_too_small_domestic` every cycle (see this module's own
    `_drop_unexecutable` for that gate) because a large-cap-only universe
    is, by construction, an expensive-shares-only universe.

    Domestic-only, deliberately: overseas discovery (`overseas_top_n`/
    `overseas_market_cap_top_n`) has no `max_price` parameter at all (a
    different KIS TR family, a different currency) and `_overseas_
    candidates` is not touched by this feature — see that function's own
    body, unchanged. Solving the overseas equivalent (USD equity, a
    different cash/risk basis) is a separate problem this pass does not
    attempt.

    Fails toward `None` (no ceiling) on every error path, never toward a
    fabricated or degenerate ceiling that could empty the universe — the
    same "fail-closed, no invented equity basis" convention
    `_drop_unexecutable` already follows for the exact same
    `cached_account_snapshot`/`AccountUnavailable` pair. A missing account
    read here must never turn into "compute affordability from nothing" —
    it must turn into "run this cycle's discovery exactly as if this
    feature didn't exist."

    The min/max guard rails exist for two different failure shapes, not
    one:
    - Below `auto_trade_universe_price_ceiling_min_krw`: the account
      itself is too small for a price ceiling to help — forcing one this
      low would fill the universe with penny-stock/관리종목-adjacent junk
      instead of letting `account_too_small_domestic` fire as the
      correctly-diagnosing signal it already is. Skipping the ceiling
      here does NOT fix the underlying problem (that's a capital
      decision, not a code one — see project memory
      corporate-account-equity-too-small-2026-09-02) but it does keep
      this feature from making a genuinely-too-small account WORSE by
      dressing up an empty pool as a penny-stock pool.
    - At or above `auto_trade_universe_price_ceiling_max_krw`: the
      account is well-funded enough that literally every KRX share price
      already clears the computed ceiling — sending it anyway would put
      an absurd number on the wire for no filtering benefit. This is also
      why there is no SEPARATE equity threshold gating this whole
      function: the ceiling self-disables once the account is big enough,
      so one number does both jobs.

    See `app.domain.executability.affordable_price_ceiling`'s own
    docstring for the full three-term derivation and why `slack_multiple`
    only ever widens the risk term."""
    if not settings.auto_trade_universe_price_ceiling_domestic:
        return None

    from app.execution.portfolio_risk import AccountUnavailable, cached_account_snapshot
    from app.domain.executability import SizingLimits, affordable_price_ceiling

    try:
        snapshot = await cached_account_snapshot("domestic")
    except AccountUnavailable:
        logger.warning(
            "auto_trade_loop domestic: real account snapshot unavailable — building the universe with "
            "NO price ceiling this cycle (not fabricating one from a missing balance); "
            "_drop_unexecutable remains the exact affordability gate downstream"
        )
        return None
    equity = snapshot.total_value
    if equity <= 0:
        return None
    cash = max(equity - sum(snapshot.per_ticker_value.values()), 0.0)

    ceiling = affordable_price_ceiling(
        equity,
        cash,
        risk_rules.rule("sizing_target_atr_pct"),
        SizingLimits(
            max_allocation_fraction=1.0,
            max_risk_pct_of_equity=risk_rules.rule("sizing_max_risk_pct_of_equity"),
            min_stop_atr_multiple=risk_rules.rule("sizing_min_stop_atr_multiple"),
        ),
        slack_multiple=settings.auto_trade_universe_price_ceiling_slack_multiple,
    )
    if ceiling is None:
        return None
    if ceiling >= settings.auto_trade_universe_price_ceiling_max_krw:
        return None
    if ceiling < settings.auto_trade_universe_price_ceiling_min_krw:
        logger.warning(
            "auto_trade_loop domestic: computed price ceiling ₩%s is below the configured floor "
            "₩%s — skipping the ceiling entirely rather than replacing the large-cap universe with "
            "penny-stock rows. This is an account-size condition; expect account_too_small_domestic",
            f"{ceiling:,.0f}", f"{settings.auto_trade_universe_price_ceiling_min_krw:,.0f}",
        )
        return None
    logger.info(
        "auto_trade_loop domestic: KRX discovery price ceiling ₩%s "
        "(equity ₩%s, cash ₩%s, assumed ATR %.2f%%, slack %.2fx)",
        f"{ceiling:,.0f}", f"{equity:,.0f}", f"{cash:,.0f}", risk_rules.rule("sizing_target_atr_pct") * 100,
        settings.auto_trade_universe_price_ceiling_slack_multiple,
    )
    return ceiling


async def _domestic_candidates(broker, exclude: frozenset[str] = frozenset()) -> list[str]:
    if not is_domestic_market_open():
        logger.info("auto_trade_loop: KRX market closed — skipping domestic candidates this cycle")
        return []
    domestic_trade_allowlist, _ = _split_trade_allowlist()
    # Account-derived price ceiling (2026-09-02, screening-improvement-
    # research #3 follow-up) threaded into all four axes below — see
    # _domestic_price_ceiling's own docstring for the full account-fail-
    # closed/min-max-guard story, and
    # app.domain.executability.affordable_price_ceiling for the arithmetic.
    max_price = await _domestic_price_ceiling()
    candidates = await krx_top_n(broker, settings.auto_trade_scan_count_domestic, max_price=max_price)
    market_cap_candidates = await krx_market_cap_top_n(
        broker, settings.auto_trade_scan_count_domestic_market_cap, max_price=max_price
    )
    # Two toggle-gated entry axes (default OFF), added 2026-09-02
    # (screening-improvement-research #3) alongside the turnover/market-cap
    # pair above — see krx_fluctuation_losers_n's docstring for why a third/
    # fourth axis is worth adding (turnover/market-cap collinearity ~35
    # overlapping mega-caps every cycle) and the reversal-sweep memory
    # result (screening-reversal-sweep-result-2026-09-02) for why these are
    # entry axes only: ranking direction stays sign=+1 momentum regardless
    # of which axis admitted a name. Axis-choice and ranking-sign are
    # independent decisions.
    loser_candidates = (
        await krx_fluctuation_losers_n(
            broker,
            settings.auto_trade_scan_count_domestic_fluctuation_losers,
            min_volume=settings.auto_trade_universe_min_volume_domestic,
            max_price=max_price,
        )
        if settings.auto_trade_universe_fluctuation_losers_domestic
        else []
    )
    disparity_candidates = (
        await krx_disparity_low_n(
            broker,
            settings.auto_trade_scan_count_domestic_disparity,
            window_days=settings.auto_trade_universe_disparity_window_days,
            min_volume=settings.auto_trade_universe_min_volume_domestic,
            max_price=max_price,
        )
        if settings.auto_trade_universe_disparity_domestic
        else []
    )
    universe = [
        t for t in dict.fromkeys(
            domestic_trade_allowlist + candidates + market_cap_candidates + loser_candidates + disparity_candidates
        ) if t not in exclude
    ]
    # §4-4 axis provenance (2026-09-08): which discovery axis admitted each
    # ticker. Deliberately a LIST per ticker, not one label — a name
    # routinely enters on several axes at once (the turnover/market-cap pair
    # is near-collinear on KRX, see auto_trade_scan_count_domestic_market_cap's
    # own config comment: ~35 overlapping mega-caps every cycle), so "the"
    # axis does not exist and collapsing to one would silently misattribute
    # every overlap. Observational only — ranking is unaffected (axis choice
    # and ranking direction are independent decisions, project memory
    # screening-reversal-sweep-result-2026-09-02).
    axes_by_ticker = _axis_map(
        ("trade_allowlist", domestic_trade_allowlist),
        ("krx_top_n", candidates),
        ("market_cap_top_n", market_cap_candidates),
        ("fluctuation_losers", loser_candidates),
        ("disparity_low", disparity_candidates),
    )
    return await _select_market_candidates(
        broker,
        "domestic",
        universe,
        domestic_trade_allowlist,
        max_tickers=settings.auto_trade_max_tickers_domestic,
        lookback_days=settings.momentum_lookback_days_domestic,
        use_volatility_adjustment=settings.screening_volatility_adjustment_domestic,
        use_correlation_dedup=settings.screening_correlation_dedup_domestic,
        max_open=risk_rules.rule("max_open_positions_domestic"),
        axes_by_ticker=axes_by_ticker,
    )


async def _overseas_candidates(broker, exclude: frozenset[str] = frozenset()) -> list[str]:
    if not settings.overseas_trading_enabled:
        # Hard scope switch — see app.config.overseas_trading_enabled's own
        # comment. debug, not info: this fires every cycle for a
        # domestic-only deployment, which would otherwise be as noisy as the
        # market-closed branch below is meant not to be.
        logger.debug("auto_trade_loop: overseas_trading_enabled=False — skipping overseas candidates this cycle")
        return []
    if not is_overseas_market_open():
        logger.info("auto_trade_loop: US market closed — skipping overseas candidates this cycle")
        return []
    _, overseas_trade_allowlist = _split_trade_allowlist()
    candidates = await overseas_top_n(broker, settings.auto_trade_scan_count_overseas)
    market_cap_candidates = await overseas_market_cap_top_n(broker, settings.auto_trade_scan_count_overseas_market_cap)
    universe = [
        t for t in dict.fromkeys(overseas_trade_allowlist + candidates + market_cap_candidates) if t not in exclude
    ]
    axes_by_ticker = _axis_map(
        ("trade_allowlist", overseas_trade_allowlist),
        ("overseas_top_n", candidates),
        ("overseas_market_cap_top_n", market_cap_candidates),
    )
    return await _select_market_candidates(
        broker,
        "overseas",
        universe,
        overseas_trade_allowlist,
        max_tickers=settings.auto_trade_max_tickers_overseas,
        lookback_days=settings.momentum_lookback_days_overseas,
        use_volatility_adjustment=settings.screening_volatility_adjustment_overseas,
        use_correlation_dedup=settings.screening_correlation_dedup_overseas,
        max_open=risk_rules.rule("max_open_positions_overseas"),
        axes_by_ticker=axes_by_ticker,
    )


# Which scopes have had their one SEMI_AUTO auto-run. Was a single bool, so
# with two users the second never got its first cycle — the first user's run
# had already flipped the flag.
# 단일 bool이면 두 번째 사용자는 첫 사이클을 아예 못 받음.
_semi_auto_done: set[tuple[str, str]] = set()

# Tickers whose pipeline raised last cycle, and how many consecutive times —
# forced into the NEXT cycle's candidate list regardless of momentum
# ranking (see run_cycle_now below), since a failed ticker otherwise only
# gets retried if it happens to re-qualify on its own. Confirmed live
# (2026-08-03): a ~10min DB/LLM outage failed 7 tickers mid-cycle with no
# guaranteed retry. In-memory only (reset on process restart) — fine, since
# a restart already re-scans everything fresh via exclude_already_analyzed_
# today. Capped per-ticker so a persistently-broken ticker (bad data, not a
# transient outage) doesn't burn an LLM cycle on it forever.
#
# Keyed per (user_id, account_mode), same reason as `_status_by_scope` above
# — this was a single process-global dict until this fix, so one user's
# failed ticker was force-appended into every OTHER user's candidate list on
# their very next cycle (bypassing their own allowlist/momentum ranking
# entirely), and that user's own SUCCESSFUL retry then popped the entry,
# silently cancelling the original user's retry guarantee. In SEMI_AUTO this
# additionally created a pending order for a ticker the affected user never
# allowlisted, since the allowlist gate lives downstream in kis_bridge for
# FULL_AUTO but pending-order creation has no such gate.
_MAX_CONSECUTIVE_RETRY_ATTEMPTS = 3
_retry_failures_by_scope: dict[tuple[str, str], dict[str, int]] = {}


def _retry_failures_for_scope() -> dict[str, int]:
    scope = _status_scope()
    failures = _retry_failures_by_scope.get(scope)
    if failures is None:
        failures = {}
        _retry_failures_by_scope[scope] = failures
    return failures


async def run_cycle_now(exclude_already_analyzed_today: bool = False, force_refresh: bool = False) -> None:
    """One scan+analyze pass — shared by the periodic loop below (every
    interval in FULL_AUTO, once in SEMI_AUTO) and the manual '다시 분석'
    trigger (SEMI_AUTO only, via check_manual_trigger_allowed()).

    `exclude_already_analyzed_today`: without it, a re-run within the same
    day tends to surface the same momentum leaders every time (nothing
    about the ranking itself changed in the last few minutes), which felt
    to the user like "it keeps analyzing the same thing." Originally only
    the manual '다시 분석' trigger (SEMI_AUTO) passed this — the periodic
    FULL_AUTO loop left it False on the theory that re-considering a ticker
    as the day moves is the point of running periodically. In practice this
    system is a multi-day/week decision engine, not an intraday one, so a
    same-day re-analysis rarely reflects genuinely new information; it
    mostly burns an LLM cycle re-deciding on a name whose "분석 중" badge
    then never even shows (see signals.js's renderAnalyzingCards, which
    skips the badge once today already has a decision for that ticker) —
    so the periodic loop now passes this too.

    `force_refresh`: threaded straight through to run_pipeline — also the
    manual trigger's own request, and a separate concern from the exclusion
    above. `exclude_already_analyzed_today` keeps a ticker that already got
    a FINAL decision today from being picked as a candidate again at all;
    `force_refresh` is for the case that slips past that (e.g. a ticker
    some OTHER caller — a chat deep_dive, or a just-started FULL_AUTO cycle —
    analyzed in the last 15 minutes without yet recording a decision here)
    where run_pipeline would otherwise silently reuse those stale analyst
    reports even though the user explicitly asked to see this analyzed
    again right now."""
    from app.execution import market_regime
    from app.memory.auto_resolve import resolve_all_due
    from app.web import state  # local import: state.py imports this module's auto_trade_loop function

    # Status is claimed BEFORE the first await, and the try/finally spans the
    # ENTIRE body from here down (widened again 2026-08-10, second pass).
    #
    # The first pass only covered candidate building onward, leaving the
    # prologue below — `state.kis_broker()` in particular, which does real
    # I/O and can raise — outside the guard. That gap is only reachable from
    # the manual '다시 분석' path, and it is exactly the path that cannot
    # tolerate it: `check_manual_trigger_allowed` sets phase="scanning"
    # itself, synchronously, to reserve the slot against a double-click (see
    # its own docstring), so a broker failure here left phase stuck at
    # "scanning" with no cycle running and no code path left to clear it —
    # every future click then failed its "a cycle is already running" guard,
    # permanently, until a process restart. The periodic loop never hit this
    # (it doesn't pre-set phase), which is why it went unnoticed.
    # 예외가 어디서 나든 상태가 반드시 풀리도록 함수 본문 전체를 감쌈 — 예전엔
    # 후보 선정부터만 감싸서, 그 앞의 broker 조회가 실패하면 수동 트리거가 미리
    # 선점해둔 "scanning"이 영원히 안 풀려 '다시 분석' 버튼이 영구 차단됐음.
    _status.phase = "scanning"
    _status.current_tickers = set()
    _status.current_stage = {}
    _status.cycle_started_at = get_clock().now()
    try:
        broker = await state.kis_broker()
        # Each market's OWN local calendar date (local_today), not a single
        # shared KST `get_clock().today()` — KST midnight falls mid-session for
        # NASDAQ, so a shared date meant an overseas position analyzed right
        # before KST midnight could look "not yet analyzed today" again minutes
        # later, on the wrong side of a boundary that had nothing to do with
        # the still-open trading session. See _drop_analyzed_within's docstring
        # for the same reasoning applied to the rebalance/held-review throttles.
        today_domestic = local_today("domestic")
        today_overseas = local_today("overseas")
        # Idempotent per calendar day (see their own docstrings) — cheap to call
        # at the top of every cycle rather than needing a separate daily
        # scheduler. Best-effort: never raises, so a KIS/yfinance/DB hiccup here
        # can't block the cycle's actual analysis below. Overseas (SPY) has no
        # broker dependency, so it's still worth trying even if `broker` above
        # is None (unlikely in practice, but not assumed away).
        try:
            await market_regime.refresh_domestic_regime()
        except Exception:
            logger.warning("domestic market_regime refresh failed — continuing with the prior cash-floor state", exc_info=True)
        try:
            await market_regime.refresh_overseas_regime()
        except Exception:
            logger.warning("overseas market_regime refresh failed — continuing with the prior cash-floor state", exc_info=True)
        # Daily flat-check backstop for settings.overseas_trading_enabled=
        # False (see that flag's own comment, and portfolio_risk.
        # verify_overseas_flat_if_disabled's docstring). The function itself
        # already fail-safes internally (a probe failure trips the latch,
        # per its own docstring) — this outer try/except is an extra
        # belt-and-suspenders against anything in the wiring itself (e.g. an
        # import error) so a probe failure can never block or crash a cycle.
        try:
            from app.execution import portfolio_risk
            await portfolio_risk.verify_overseas_flat_if_disabled()
        except Exception:
            logger.exception("overseas flat-check backstop failed — continuing this cycle")
        # Auto-resolve any LIVE decision old enough to grade (see
        # app.memory.auto_resolve's own docstring — this used to be a manual CLI
        # step nobody had evidence of ever actually running). No debounce needed:
        # naturally near-zero-cost once nothing is newly due (resolved entries
        # drop out of the underlying query on their own).
        try:
            await resolve_all_due()
        except Exception:
            logger.warning("auto_resolve failed this cycle — will retry next cycle", exc_info=True)
        # Fill in realized forward returns for screening snapshots old enough
        # to measure (§4-1, 2026-09-08) — the join side of app.live.
        # screening_log's write path. Same "call it once per cycle, idempotent,
        # no separate scheduler needed" placement/reasoning as resolve_all_due
        # above: naturally near-zero-cost once nothing new is due, since a
        # resolved (ticker, trade_date) drops out of the underlying query on
        # its own. Bounded per pass (settings.screening_forward_resolve_max_
        # per_pass) so a long backlog can never turn one cycle into a KIS
        # rate-limit incident.
        try:
            from app.audit.screening_ic import resolve_due_forward_returns

            await resolve_due_forward_returns()
        except Exception:
            logger.warning(
                "screening forward-return resolution failed this cycle — will retry next cycle",
                exc_info=True,
            )
        # Overnight/weekend gap-risk warning (ai-engine-24h-operation-analysis
        # §5) — deliberately against the FULL trade_allowlist, not the
        # candidate pool below: _domestic_candidates/_overseas_candidates
        # already return [] once their market is closed, which is exactly
        # when a held position needs this check. Best-effort, like the
        # market_regime refreshes above — a failure here must never block
        # this cycle's actual analysis.
        domestic_trade_allowlist, overseas_trade_allowlist = _split_trade_allowlist()
        try:
            await _warn_overnight_gap_risk(domestic_trade_allowlist, "domestic")
        except Exception:
            logger.warning("overnight gap-risk check failed for domestic — continuing", exc_info=True)
        try:
            await _warn_overnight_gap_risk(overseas_trade_allowlist, "overseas")
        except Exception:
            logger.warning("overnight gap-risk check failed for overseas — continuing", exc_info=True)
        exclude_domestic: frozenset[str] = frozenset()
        exclude_overseas: frozenset[str] = frozenset()
        if exclude_already_analyzed_today:
            try:
                exclude_domestic = frozenset(await tickers_analyzed_on(today_domestic))
                exclude_overseas = frozenset(await tickers_analyzed_on(today_overseas))
            except Exception:
                logger.warning("could not load already-analyzed tickers — not excluding any", exc_info=True)
        # Candidate building (state.reconcile_trade_allowlist, _domestic_candidates/
        # _overseas_candidates, or anything they call) is not individually
        # guarded the way _analyze_one below guards each ticker, so it relies
        # on this same try/finally too — that was the first pass's original
        # reason for existing, unchanged.
        # Sync trade_allowlist against real KIS positions BEFORE building this
        # cycle's candidates (2026-07-29) — a real holding bought outside this
        # system (manually via the KIS app, or predating this tracking) would
        # otherwise never appear here at all: stop_loss_loop, _held_needing_review
        # below, and the live price stream are all trade_allowlist-scoped, and
        # kis_bridge.py's execution gate unconditionally BLOCKS a sell for
        # anything not on the list. Newly-reconciled tickers are force-included
        # in this cycle's analysis below regardless of momentum ranking — a
        # freshly discovered holding almost certainly has no stop-loss on record
        # yet, so it needs a full pipeline run now, not whenever it next happens
        # to rank in the momentum top-N (which, being a real position rather
        # than a fresh candidate, it may never do). Best-effort: reconciliation
        # failures are already handled internally (see its own docstring) and
        # never raise, so this can't break the rest of the cycle.
        newly_reconciled = await state.reconcile_trade_allowlist()
        tickers = await _domestic_candidates(broker, exclude_domestic) + await _overseas_candidates(broker, exclude_overseas)
        # Force-include last cycle's failures alongside newly_reconciled — same
        # reasoning as that one (see its own comment above): a failed ticker
        # never got a final decision, so it wouldn't be in `exclude` anyway, but
        # it also isn't guaranteed to re-qualify for `tickers` above on its own
        # (screening/momentum ranking can drop it). See _retry_failures_for_scope.
        retry_failures = _retry_failures_for_scope()
        tickers = list(dict.fromkeys(tickers + newly_reconciled + list(retry_failures)))
        _status.tickers_this_cycle = tickers
        _status.phase = "analyzing"
        # 2026-07-30: a quiet cycle (every candidate already analyzed today,
        # exclude_already_analyzed_today's whole point) produces zero LLM calls
        # with no other log trace — indistinguishable from an unresponsive LLM
        # backend by log inspection alone until this line existed. See project
        # memory llm-model-swap-gpt-oss-issue.md's "91분 무응답" entry, which
        # turned out likely to be exactly this, not an actual outage.
        if not tickers:
            logger.info("auto_trade_loop: no new candidates this cycle (all already analyzed today or filtered out)")

        semaphore = asyncio.Semaphore(settings.auto_trade_max_concurrent_analyses)

        async def _analyze_one(ticker: str) -> None:
            async with semaphore:
                # 2026-07-30: candidates are selected once at the START of the
                # cycle (_domestic_candidates/_overseas_candidates above already
                # skip a closed market at THAT point) — but with bounded
                # concurrency and a ~6-9 min per-ticker LLM pipeline, a cycle
                # that starts a minute before close can still be mid-analysis
                # well after the market has actually closed (confirmed live:
                # a cycle started 15:28 KST was still analyzing at 15:49, and
                # the resulting BUY got rejected by KIS — "모의투자 장종료입니다" —
                # only at the execution step, after the full ~9-minute pipeline
                # had already run for nothing). Re-checking here, right before
                # spending that pipeline cost, skips a ticker whose market
                # closed while it was queued behind the semaphore instead of
                # discovering that only at order time.
                is_domestic = ticker.endswith(".KS")
                market_open = is_domestic_market_open() if is_domestic else is_overseas_market_open()
                if not market_open:
                    logger.info(
                        "auto_trade_loop: skipping %s — its market closed while queued this cycle", ticker
                    )
                    return
                async def _stage_cb(label: str) -> None:
                    _status.current_stage[ticker] = label

                async def _record_pipeline_failure(ticker: str) -> None:
                    attempts = retry_failures.get(ticker, 0) + 1
                    if attempts >= _MAX_CONSECUTIVE_RETRY_ATTEMPTS:
                        retry_failures.pop(ticker, None)
                        logger.exception(
                            "auto_trade_loop: pipeline failed for %s (attempt %d/%d) — giving up, "
                            "won't force-retry again until it re-qualifies on its own",
                            ticker, attempts, _MAX_CONSECUTIVE_RETRY_ATTEMPTS,
                        )
                        # Self-monitoring gap (ai-engine-24h-operation-analysis, P0-2):
                        # exhausting every retry on a ticker used to be logged and
                        # nothing else — during an unattended stretch (esp. overseas
                        # hours, KST early morning) this is exactly the "LLM/broker
                        # degraded, decisions silently stop" case nobody would notice
                        # until morning. Reuses app.live.alerts' existing
                        # table/cooldown/dashboard feed instead of a new mechanism.
                        for alert in filter_new_alerts(
                            [
                                Alert(
                                    ticker=ticker,
                                    kind="pipeline_failure",
                                    message=(
                                        f"{ticker} pipeline failed {attempts} consecutive attempts — "
                                        "giving up until it re-qualifies on its own"
                                    ),
                                    triggered_at=get_clock().now(),
                                )
                            ]
                        ):
                            await record_alert(alert)
                    else:
                        retry_failures[ticker] = attempts
                        logger.exception(
                            "auto_trade_loop: pipeline failed for %s (attempt %d/%d) — will force-retry next cycle",
                            ticker, attempts, _MAX_CONSECUTIVE_RETRY_ATTEMPTS,
                        )

                _status.current_tickers.add(ticker)
                try:
                    await run_pipeline(
                        ticker,
                        today_domestic if is_domestic else today_overseas,
                        force_refresh=force_refresh,
                        on_stage=_stage_cb,
                    )
                except asyncio.CancelledError:
                    # CancelledError is a BaseException, not an Exception, so
                    # the `except Exception` below never saw it — it used to
                    # escape straight out of this coroutine, and because this
                    # runs under asyncio.gather(return_exceptions=False), that
                    # cancelled every OTHER ticker's in-flight analysis too
                    # (gather's default teardown on any exception), with none
                    # of them ever reaching their own handler to get a retry
                    # entry or alert. See id=377/id=949 (app.orchestrator.
                    # pipeline._run_pipeline_impl's matching guard) — same
                    # root cause (a cancelled DB connection-pool acquire),
                    # different blast radius. A real cancellation of THIS
                    # ticker's task (shutdown tearing down the whole cycle)
                    # must still propagate untouched.
                    if task_was_actually_cancelled():
                        logger.warning(
                            "auto_trade_loop: %s cancelled (real task cancellation, likely shutdown) "
                            "— propagating", ticker,
                        )
                        raise
                    logger.warning(
                        "auto_trade_loop: %s hit an unrequested CancelledError (suspected "
                        "connection-pool acquire cancellation, see id=377/id=949) — treating as "
                        "a per-ticker failure instead of letting it cancel every other ticker",
                        ticker,
                    )
                    await _record_pipeline_failure(ticker)
                except Exception:
                    await _record_pipeline_failure(ticker)
                else:
                    retry_failures.pop(ticker, None)
                finally:
                    # discard, not remove: a duplicate ticker slipping through
                    # twice in the same cycle (shouldn't happen — dict.fromkeys
                    # dedup upstream — but this must never raise either way)
                    # would otherwise KeyError on the second removal.
                    _status.current_tickers.discard(ticker)
                    _status.current_stage.pop(ticker, None)

        await asyncio.gather(*(_analyze_one(ticker) for ticker in tickers))
    finally:
        _status.phase = "idle"
        _status.current_tickers = set()
        _status.current_stage = {}
        _status.cycle_finished_at = get_clock().now()


def check_manual_trigger_allowed() -> None:
    """Pure/sync so app.web.routers.live's POST endpoint can reject
    immediately (429) without waiting on a cycle — the actual run is kicked
    off separately (asyncio.create_task) only once this passes.

    Reserves the "a cycle is running" slot itself (sets `_status.phase`)
    rather than leaving that to `run_cycle_now`. `run_cycle_now` doesn't set
    it until after several awaited calls (state.kis_broker(),
    market_regime refreshes, resolve_all_due(), tickers_analyzed_on() x2) —
    a multi-second window in which this same check still saw the OLD phase
    and passed a second time (a double-click, a client retry, two open
    dashboard tabs). Two concurrent cycles then both reach
    rebalance_to_target for overlapping tickers and, in SEMI_AUTO, both
    land a duplicate pending order for a human to approve. Setting it here,
    synchronously, closes that window: the second caller's check now sees
    "scanning" immediately, not eventually."""
    if runtime_settings.rule("execution_mode") is not ExecutionMode.SEMI_AUTO:
        raise CooldownActive("manual re-analysis is only available in SEMI_AUTO mode")
    if _status.phase in ("scanning", "analyzing"):
        raise CooldownActive("a cycle is already running")
    if _status.cycle_finished_at is not None:
        elapsed = (get_clock().now() - _status.cycle_finished_at).total_seconds()
        remaining = MANUAL_TRIGGER_COOLDOWN_SECONDS - elapsed
        if remaining > 0:
            raise CooldownActive(f"cooldown active, {remaining:.0f}s remaining")
    _status.phase = "scanning"


async def auto_trade_loop() -> None:
    """One pass per serviced identity per interval, each inside `user_scope`.

    This is the loop that can place real orders, so the identity it runs under
    is not bookkeeping — it decides whose broker credentials get used (Phase 2)
    and whose ledger the fills land in. Outside a request nothing binds an
    identity, so without the scope every cycle would run as settings.user_id.
    주문을 실제로 내는 루프라 신원 바인딩이 기록용이 아니라 실행 대상 자체를 결정함.

    Every previously-global piece this loop depended on is now per-scope:
    `runtime_settings.rule("execution_mode")` (Phase 4), the risk rules
    (Phase 4), the ticker lists (Phase 5), and this module's own `_status` /
    semi-auto-first-cycle bookkeeping (Phase 6, above).

    `ensure_loaded` is required, not defensive. The allowlist resolves per
    (user, mode) and a scope is empty until loaded, so without it this loop
    would screen a candidate pool with no allowlist entries in it and place
    orders for a user whose own list it never read.
    스코프는 load() 전까지 비어 있으므로, 이 호출 없이는 사용자의 목록을 읽지 않은
    채로 후보를 고르게 됨.

    One scope's failure must not skip the rest — a broker outage for user A is
    not a reason to skip user B's cycle entirely.
    한 사용자의 실패가 다른 사용자의 사이클을 건너뛰게 하지 않음."""
    from app.web import trade_allowlist_store

    interval_seconds = settings.auto_trade_interval_minutes * 60
    while True:
        for scope in await serviced_scopes():
            try:
                with user_scope(scope.user_id), account_mode_scope(scope.account_mode):
                    await trade_allowlist_store.ensure_loaded(scope.user_id, scope.account_mode)
                    await _run_auto_trade_cycle(scope.user_id, interval_seconds)
            except Exception as exc:
                logger.exception(
                    "auto-trade cycle failed for %s/%s — continuing with the other users",
                    scope.user_id, scope.account_mode,
                )
                # Self-monitoring gap (ai-engine-24h-operation-analysis, P0-2): a
                # whole cycle dying (DB unreachable, trade_allowlist_store failure,
                # etc.) used to be a log line and nothing else. No dedicated
                # "SYSTEM" ticker exists in app.live.store, so this alert's
                # ticker is a sentinel string, not a real tradeable symbol —
                # app.live.alerts never validates it against anything, and the
                # dashboard feed shows it as-is.
                for alert in filter_new_alerts(
                    [
                        Alert(
                            ticker="SYSTEM",
                            kind="cycle_failure",
                            message=f"auto-trade cycle failed for {scope.user_id}/{scope.account_mode}: {exc}",
                            triggered_at=get_clock().now(),
                        )
                    ]
                ):
                    await record_alert(alert)
        await asyncio.sleep(interval_seconds)


async def _run_auto_trade_cycle(user_id: str, interval_seconds: float) -> None:
    """One user's single cycle. Split out of the loop above so the sleep and
    the user iteration live in exactly one place — a `while True` in here
    would starve every identity after the first."""
    scope = _status_scope()
    current_execution_mode = runtime_settings.rule("execution_mode")
    if current_execution_mode is ExecutionMode.MANUAL:
        logger.info("auto_trade_loop: MANUAL mode — skipping this cycle (no AI auto-analysis)")
        _status.phase = "manual_skip"
        _status.current_tickers = set()
        _status.current_stage = {}
        _status.tickers_this_cycle = []
        _status.next_cycle_at = None
    elif current_execution_mode is ExecutionMode.SEMI_AUTO:
        # Auto-runs exactly once (first cycle after startup) — after
        # that, only the dashboard's "다시 분석" button (check_manual_
        # trigger_allowed + run_cycle_now) starts another one. No
        # next_cycle_at: there is no next automatic cycle to count down to.
        if scope not in _semi_auto_done:
            # exclude_already_analyzed_today=True even here (not just
            # the manual button): this "first cycle" flag is in-memory,
            # so a process restart mid-day re-arms it — without this,
            # a restart during market hours would silently re-run the
            # exact same top-movers a second time that day.
            await run_cycle_now(exclude_already_analyzed_today=True)
            _semi_auto_done.add(scope)
        # Only overwrite phase when nothing is actually in flight. Before
        # this guard, every periodic iteration (every interval_seconds,
        # forever — not just the first) unconditionally forced
        # phase="waiting_for_manual" here, even while a manual "다시 분석"
        # cycle (a separate asyncio.create_task started from the dashboard,
        # app.web.routers.live) was mid-flight. That cleared the "a cycle is
        # already running" guard check_manual_trigger_allowed relies on,
        # letting a second click start a concurrent cycle over the same
        # tickers — both reaching rebalance_to_target and, in SEMI_AUTO,
        # both landing a duplicate pending order for a human to approve.
        if _status.phase not in ("scanning", "analyzing"):
            _status.phase = "waiting_for_manual"
            _status.current_tickers = set()
            _status.current_stage = {}
            _status.next_cycle_at = None
    else:  # FULL_AUTO — keeps running every interval, no manual button
        # This system is built for multi-day/week decisions, not
        # intraday re-checking — re-running the full pipeline on a
        # ticker already decided today wastes an LLM cycle for no new
        # signal and hides the re-run behind its stale card (no
        # "analyzing" badge, since one is only shown when today has no
        # decision yet for that ticker — see signals.js's
        # renderAnalyzingCards). Excluding today's already-analyzed
        # tickers keeps each automatic cycle spent on genuinely new names.
        await run_cycle_now(exclude_already_analyzed_today=True)
        _status.next_cycle_at = get_clock().now() + timedelta(seconds=interval_seconds)
