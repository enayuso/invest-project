"""Ledger price vs actual KIS fill price — measures the gap this session's
④ investigation raised: `trades.price` is the pipeline's own fill-price
snapshot (~T+6, see `app.orchestrator.pipeline`'s `fill_price`/
`rebalance_to_target` call), not necessarily what KIS actually filled at
(domestic orders are market orders with no price argument at all; overseas
are limit orders at that snapshot price, so a mid-flight price move means
no fill rather than a bad one). Nothing after order submission writes the
real fill price back into `trades` — this script is the first time the two
are compared. That write-back is still the open half of the problem
(`app.live.fill_monitor` already fetches KIS's own `avg_price`/`filled_qty`
for every mirrored order and deliberately does not compare them — see its
"Found, fully filled" branch), so every number this script produces is
still measuring a SNAPSHOT against a fill, not a fill against a fill.

One class of row is not a slippage observation at all: a forced stop-loss
exit whose `trades.price` holds the breached THRESHOLD — a number nothing
traded at. 047040.KS id=280 recorded 19,729.24 (the trailing floor) against
a real ~21,000 market fill: a fabricated 6.4% adverse slippage, in the
single row class that matters most to the 2→3 gate. `app.execution.ledger.
force_exit_position` stopped creating these on 2026-08-27 — but that fix is
write-time only, so the pre-existing live rows produced by the DAILY-BAR
backstop (which always passed the threshold; the tick monitor never did)
were backfilled with the same marker their live counterparts now carry.
This script matches such a row normally — consuming its KIS execution so it
can't resurface as a phantom `kis_only` miss — then reports it in its own
section instead of any statistic. Detection is
`app.execution.ledger.is_threshold_placeholder_note`, the single owner of
that string; a row with no marker is treated as a real observation.

Directly relevant to [[stage-gate-criteria-draft]]'s 2→3 stage gate
("모의 대비 성과 열화 없음 (슬리피지·체결 확인)") — that condition has never
been measured before this.

Matching, 2-tier (2026-08-13 rework — see project memory:
order-fill-confirmation-gap-2026-08-13; order-number normalization and
priority-path-miss fallthrough added 2026-08-27, see
fill-monitor-order-no-padding-and-giveup-fix-2026-08-27):

1. (priority) Rows with `trades.kis_order_no` set — `app.live.fill_monitor`
   already writes this once `mirror_trade_to_kis` returns an order number —
   are joined directly against KIS execution-history rows by `order_no`.
   Exact, no aggregation, no date window: this is the same join
   `fill_monitor` itself uses. Both sides of the join are normalized via
   `app.live.fill_monitor.normalize_kis_order_no` before comparison/keying
   — domestic execution history zero-pads order numbers to 10 digits while
   overseas does not, and `trades.kis_order_no` always stores the
   zero-padded form, so an un-normalized compare silently fails every
   overseas match. A priority-path miss (no execution found for the
   normalized order_no) is not treated as a genuine miss on its own — it
   falls through to the fallback window-match path below for a second
   chance before being recorded as `ledger_only`.
2. (fallback) Older rows with `kis_order_no IS NULL` (pre-migration, or a
   mode that never mirrors to KIS), plus any priority-path miss from above,
   fall back to the previous (ticker, trade_date, side) weighted-average
   bucket comparison, but now with a `_MAX_FILL_DELAY_DAYS`-wide greedy
   window match against KIS executions (same pattern as
   `app.execution.reconciliation`'s `match_trades_against_kis`, not an
   exact-date dict lookup) — this is what catches a stop-loss ledger row
   recorded on the breakout-bar's date (see
   `ledger.check_and_apply_stop_loss`) whose real KIS mirror order (and
   fill) landed days later.

Either path can produce a *delayed* match (`fill_delay_days > 0`): the
ledger's `trade_date` and the KIS order's actual date disagree. A delayed
match is not measuring slippage (the gap between our recorded price and the
fill price at the *same moment*) — it's measuring market drift over however
many days the fill was late. `render_slippage` keeps those out of the main
mean/median and reports them in their own section instead.

Read-only. No LLM calls. Reuses `app.brokers.get_broker("kis")` and the
same execution-history TRs `app.execution.pnl` already fetches, but does its
own (much simpler) aggregation instead of pnl.py's FIFO cost-basis matching
— that logic answers "what was realized P&L", not "how far off was our
own recorded price."
"""

from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass, field
from datetime import date, timedelta

from app.brokers import get_broker
from app.domain.models.account import BrokerExecution
from app.execution.ledger import is_threshold_placeholder_note
from app.live.fill_monitor import normalize_kis_order_no
from app.memory.db import get_pool
from app.user_context import get_current_user_id

_KIS_LOOKBACK_LIMIT_DAYS = 90  # same TR cap app.execution.pnl documents

# Same rationale/value as app.execution.reconciliation._MAX_FILL_DELAY_DAYS —
# a fallback (no kis_order_no) ledger row can legitimately fill this many
# days after its recorded trade_date (stop-loss rows use the breakout bar's
# date, not the actual order date). Anything later is treated as a genuine
# miss (ledger_only) rather than silently absorbed.
_MAX_FILL_DELAY_DAYS = 14

logger = logging.getLogger(__name__)


@dataclass
class SlippageRow:
    ticker: str
    trade_date: date
    side: str
    ledger_price: float
    ledger_shares: int
    kis_price: float
    kis_shares: float
    # Positive = unfavorable (paid more on a buy, received less on a sell)
    # relative to what the ledger recorded — normalized so every row's sign
    # means the same thing regardless of side.
    slippage_pct: float
    # Days between the ledger's trade_date and the KIS execution's actual
    # order_date. 0 = same-day (order_no exact match or window match that
    # happened to land same-day) — only these belong in the main slippage
    # stats. >0 = the fill genuinely landed later; this measures market
    # drift, not slippage, and is reported separately.
    fill_delay_days: int
    # True when trades.adjustment_note says this row's `price` is a stop
    # THRESHOLD rather than an observed fill (app.execution.ledger's
    # FORCED_EXIT_THRESHOLD_PRICE_NOTE_PREFIX). Still MATCHED against its KIS
    # execution — that consumes the execution so it can't resurface as a
    # spurious `kis_only` — but the resulting `slippage_pct` compares a real
    # fill against a number nothing traded at, so it is not a slippage
    # observation and never enters the mean/median. 047040.KS id=280 alone
    # reported a fabricated -6.4% here, in the one row class the 2→3 stage
    # gate ("모의 대비 성과 열화 없음") is measured on.
    price_is_threshold: bool = False


@dataclass
class SlippageResult:
    matched: list[SlippageRow] = field(default_factory=list)
    ledger_only: list[tuple[str, date, str]] = field(default_factory=list)  # our trade, no KIS record found
    kis_only: list[tuple[str, date, str]] = field(default_factory=list)  # KIS fill, no matching ledger row
    # Rows where a real KIS execution was found and consumed, but the ledger
    # side had price <= 0 (so slippage_pct is undefined) — kept visible here
    # instead of silently vanishing (not `matched`, not `ledger_only`, not
    # `kis_only`; the old behavior of a bare `continue` made the KIS fill
    # look like a phantom or, in the fallback path, disappear entirely).
    unusable_price: list[tuple[str, date, str]] = field(default_factory=list)
    # True when a caller-supplied `start` predates the KIS lookback window
    # and was silently pulled forward to `earliest` — surfaced in the
    # rendered report instead of staying invisible.
    start_clamped: bool = False
    # Threaded through from `compute_slippage`'s own `account_mode` param
    # purely so `render_slippage` can caption the report correctly — the
    # "이 수치는 하한선" caveat below only holds for paper (paper fills are
    # typically more forgiving than real KIS fills), and was hardcoded True
    # regardless of which mode was actually queried until 2026-09-10 (a real
    # run of `app.audit slippage --account-mode real` printed "지금은
    # 모의계좌 값" about its own real-account numbers).
    account_mode: str = "paper"


async def _fetch_ledger_trades(start: date, end: date, account_mode: str, user_id: str) -> list[dict]:
    """Raw (unaggregated) ledger rows — aggregation now happens downstream,
    separately for the order_no-matched and window-matched paths, since they
    need different granularity (order_no path is exact per-row; window path
    still buckets by (ticker, trade_date, side) same as before)."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT ticker, trade_date, side, shares, price, kis_order_no, adjustment_note
            FROM trades
            WHERE is_backtest = false AND account_mode = $1 AND user_id = $2
              AND trade_date BETWEEN $3 AND $4
              AND lower(side) IN ('buy', 'sell')
              -- Same exclusions as app.execution.reconciliation's
              -- fetch_recent_local_trades, same table, same "is this row
              -- checkable against a real KIS fill" question: a reversal row
              -- is a synthetic correction with a flipped side and no real
              -- KIS counterpart by construction, and a row whose
              -- mirror_disposition says it was never sent to KIS (MANUAL
              -- mode, dry-run, backtest replay) or is still SEMI_AUTO-pending
              -- has the same property. Without these, a reversal row can
              -- either inflate ledger_only (can never match) or be
              -- fallback-matched against an unrelated real KIS execution of
              -- the flipped side, fabricating a slippage row.
              AND reason NOT LIKE 'reversal:%'
              AND (mirror_disposition IS NULL OR mirror_disposition LIKE 'submitted:%')
            ORDER BY ticker, trade_date
            """,
            account_mode, user_id, start, end,
        )
    # `trades.side` is stored lowercase ('buy'/'sell'); normalized to Title
    # case here to match KIS-side "Buy"/"Sell" convention used below.
    return [
        {
            "ticker": r["ticker"], "trade_date": r["trade_date"], "side": r["side"].capitalize(),
            "shares": r["shares"], "price": r["price"], "kis_order_no": r["kis_order_no"],
            "adjustment_note": r["adjustment_note"],
        }
        for r in rows
    ]


def _order_date_as_date(order_date: str) -> date:
    return date(int(order_date[:4]), int(order_date[4:6]), int(order_date[6:]))


def _to_execution_key_lists(
    executions: list[BrokerExecution], is_domestic: bool,
) -> list[tuple[str, date, str, BrokerExecution]]:
    """(ticker, order_date, side, execution) for every filled KIS execution,
    ticker normalized to the same convention `trades.ticker` uses (".KS"
    suffix for domestic)."""
    out = []
    for e in executions:
        if e.filled_qty <= 0:
            continue
        ticker = f"{e.code}.KS" if is_domestic else e.code
        try:
            order_date = _order_date_as_date(e.order_date)
        except (ValueError, TypeError) as exc:
            # One malformed/empty ord_dt from a bad KIS row must not take
            # down the whole report — skip just this execution.
            logger.warning(
                "slippage_validation: malformed order_date %r for order_no=%s code=%s, skipping: %s",
                e.order_date, e.order_no, e.code, exc,
            )
            continue
        side = "Buy" if e.side == "buy" else "Sell"
        out.append((ticker, order_date, side, e))
    return out


def _bucket_by_key(
    entries: list[tuple[str, date, str, BrokerExecution]],
) -> dict[tuple[str, date, str], tuple[float, float]]:
    """(ticker, order_date, side) -> (weighted_avg_price, total_filled_qty),
    for whatever executions are passed in (already excludes anything
    consumed by the order_no path when called from compute_slippage)."""
    buckets: dict[tuple[str, date, str], list[tuple[float, float]]] = {}
    for ticker, order_date, side, e in entries:
        buckets.setdefault((ticker, order_date, side), []).append((e.filled_qty, e.avg_price))
    out = {}
    for key, fills in buckets.items():
        total_qty = sum(q for q, _ in fills)
        avg_price = sum(q * p for q, p in fills) / total_qty if total_qty else 0.0
        out[key] = (avg_price, total_qty)
    return out


def _slippage_pct(ledger_price: float, kis_price: float, side: str) -> float | None:
    if ledger_price <= 0:
        return None
    raw_diff_pct = (kis_price - ledger_price) / ledger_price
    return raw_diff_pct if side == "Buy" else -raw_diff_pct


async def compute_slippage(
    start: date | None = None, end: date | None = None, account_mode: str = "paper", user_id: str | None = None,
) -> SlippageResult:
    user_id = user_id or get_current_user_id()
    end = end or date.today()
    earliest = date.today() - timedelta(days=_KIS_LOOKBACK_LIMIT_DAYS)
    requested_start = start
    start = max(start or earliest, earliest)
    start_clamped = requested_start is not None and requested_start < earliest

    broker = await get_broker("kis", user_id=user_id, account_mode=account_mode)
    if broker is None:
        raise RuntimeError("KIS broker not configured — cannot fetch execution history")

    start_str = start.strftime("%Y%m%d")
    # Extend the KIS query past `end` so a fallback (window-matched) fill
    # that lands up to _MAX_FILL_DELAY_DAYS after a trade recorded near the
    # edge of the requested range isn't missed.
    query_end_str = (end + timedelta(days=_MAX_FILL_DELAY_DAYS)).strftime("%Y%m%d")
    domestic_executions = await broker.fetch_daily_executions(start_str, query_end_str)
    overseas_executions = await broker.fetch_overseas_executions(start_str, query_end_str)

    domestic_entries = _to_execution_key_lists(domestic_executions, is_domestic=True)
    overseas_entries = _to_execution_key_lists(overseas_executions, is_domestic=False)
    all_entries = domestic_entries + overseas_entries

    # order_no -> execution, for the priority path. KIS order numbers are
    # unique per order, so this is a safe 1:1 index across both markets.
    kis_by_order_no: dict[str, BrokerExecution] = {}
    for _, _, _, e in all_entries:
        if e.order_no:
            normalized = normalize_kis_order_no(e.order_no)
            if normalized:
                kis_by_order_no[normalized] = e

    ledger_trades = await _fetch_ledger_trades(start, end, account_mode, user_id)

    result = SlippageResult(start_clamped=start_clamped, account_mode=account_mode)
    consumed_order_nos: set[str] = set()
    window_fallback_ledger: list[dict] = []

    # --- Priority path: exact order_no join ---
    for t in ledger_trades:
        order_no = t["kis_order_no"]
        if not order_no:
            window_fallback_ledger.append(t)
            continue
        execution = kis_by_order_no.get(normalize_kis_order_no(order_no))
        if execution is None:
            window_fallback_ledger.append(t)
            continue
        if t["price"] <= 0:
            # A real KIS fill exists — consume it so it can't resurface as a
            # misleading `kis_only` "phantom" (a real ledger row DOES exist,
            # its price is just unusable) — but record it separately instead
            # of a bare `continue`, so it stays visible somewhere.
            consumed_order_nos.add(normalize_kis_order_no(order_no))
            result.unusable_price.append((t["ticker"], t["trade_date"], t["side"]))
            continue
        slippage_pct = _slippage_pct(t["price"], execution.avg_price, t["side"])
        if slippage_pct is None:
            continue
        consumed_order_nos.add(normalize_kis_order_no(order_no))
        fill_delay_days = max(0, (_order_date_as_date(execution.order_date) - t["trade_date"]).days)
        result.matched.append(
            SlippageRow(
                ticker=t["ticker"], trade_date=t["trade_date"], side=t["side"],
                ledger_price=t["price"], ledger_shares=t["shares"],
                kis_price=execution.avg_price, kis_shares=execution.filled_qty,
                slippage_pct=slippage_pct, fill_delay_days=fill_delay_days,
                price_is_threshold=is_threshold_placeholder_note(t["adjustment_note"]),
            )
        )

    # --- Fallback path: (ticker, trade_date, side) bucket + 14-day greedy window match ---
    # Only KIS executions not already claimed by the order_no path are
    # eligible here, and only entries whose ledger row had no kis_order_no
    # at all (never both — that would double-count a single real fill).
    remaining_entries = [
        entry for entry in all_entries
        if normalize_kis_order_no(entry[3].order_no) not in consumed_order_nos
    ]

    ledger_buckets: dict[tuple[str, date, str], list[tuple[int, float, bool]]] = {}
    for t in window_fallback_ledger:
        key = (t["ticker"], t["trade_date"], t["side"])
        ledger_buckets.setdefault(key, []).append(
            (t["shares"], t["price"], is_threshold_placeholder_note(t["adjustment_note"]))
        )

    ledger_bucket_rows = []
    for (ticker, trade_date, side), fills in ledger_buckets.items():
        total_shares = sum(s for s, _, _ in fills)
        avg_price = sum(s * p for s, p, _ in fills) / total_shares if total_shares else 0.0
        ledger_bucket_rows.append(
            {
                "ticker": ticker, "trade_date": trade_date, "side": side,
                "shares": total_shares, "price": avg_price,
                # A weighted average that mixes a threshold placeholder with
                # a real price is neither — one tainted member taints the
                # whole bucket, same "don't average away a fabrication"
                # reasoning as the priority path above.
                "price_is_threshold": any(flag for _, _, flag in fills),
            }
        )

    # Candidate KIS buckets keyed by (ticker, order_date, side), consumed
    # (removed) as they're matched, so one KIS bucket can't satisfy two
    # different ledger buckets.
    kis_pool = _bucket_by_key(remaining_entries)

    for t in sorted(ledger_bucket_rows, key=lambda x: x["trade_date"]):
        candidates = [
            key for key in kis_pool
            if key[0] == t["ticker"] and key[2] == t["side"]
            and t["trade_date"] <= key[1] <= t["trade_date"] + timedelta(days=_MAX_FILL_DELAY_DAYS)
        ]
        if not candidates:
            result.ledger_only.append((t["ticker"], t["trade_date"], t["side"]))
            continue

        kis_price_shares = {key: kis_pool[key] for key in candidates}
        best_key = min(
            candidates,
            key=lambda key: (
                abs(kis_price_shares[key][1] - t["shares"]),
                abs((key[1] - t["trade_date"]).days),
            ),
        )
        kis_price, kis_shares = kis_pool.pop(best_key)
        if t["price"] <= 0:
            # kis_pool.pop above already consumed the KIS bucket, matching
            # the priority path's behavior — record it instead of letting it
            # vanish silently.
            result.unusable_price.append((t["ticker"], t["trade_date"], t["side"]))
            continue
        slippage_pct = _slippage_pct(t["price"], kis_price, t["side"])
        if slippage_pct is None:
            continue
        fill_delay_days = (best_key[1] - t["trade_date"]).days
        result.matched.append(
            SlippageRow(
                ticker=t["ticker"], trade_date=t["trade_date"], side=t["side"],
                ledger_price=t["price"], ledger_shares=t["shares"],
                kis_price=kis_price, kis_shares=kis_shares,
                slippage_pct=slippage_pct, fill_delay_days=fill_delay_days,
                price_is_threshold=t["price_is_threshold"],
            )
        )

    # Anything left in kis_pool after both passes has no ledger counterpart.
    for key in kis_pool:
        result.kis_only.append(key)

    return result


def render_slippage(r: SlippageResult) -> str:
    # Threshold-placeholder rows are pulled out FIRST, before the delay
    # split — they're excluded regardless of fill_delay_days, since the
    # comparison itself (real fill vs. a number nothing traded at) is
    # meaningless no matter how promptly it landed.
    threshold = [row for row in r.matched if row.price_is_threshold]
    usable = [row for row in r.matched if not row.price_is_threshold]
    same_day = [row for row in usable if row.fill_delay_days == 0]
    delayed = [row for row in usable if row.fill_delay_days > 0]

    clamp_note = (
        f"⏳ 요청한 시작일이 KIS 조회 한도({_KIS_LOOKBACK_LIMIT_DAYS}일)보다 오래되어 "
        f"{date.today() - timedelta(days=_KIS_LOOKBACK_LIMIT_DAYS)}(으)로 당겨졌습니다."
        if r.start_clamped else None
    )

    if not same_day:
        header = "대사 가능한 당일 체결 거래가 없습니다 (기간 안에 원장·KIS 양쪽 모두 있는 당일 체결 거래가 0건)."
        if delayed:
            header += f" (지연 매칭 {len(delayed)}건은 있음 — 아래 참고)"
        if threshold:
            header += f" (임계값 기록행 {len(threshold)}건은 제외 — 아래 참고)"
        lines = [clamp_note, header] if clamp_note else [header]
        if threshold:
            lines += ["", _render_threshold_section(threshold)]
        if delayed:
            lines += ["", _render_delayed_section(delayed)]
        if r.unusable_price:
            lines += ["", _render_unusable_price_section(r.unusable_price)]
        return "\n".join(lines)

    n = len(same_day)
    values = sorted(row.slippage_pct for row in same_day)
    mean = sum(values) / n
    median = statistics.median(values)
    worst = values[-1]
    best = values[0]
    unfavorable = sum(1 for v in values if v > 0)

    domestic = [row for row in same_day if row.ticker.endswith(".KS")]
    overseas = [row for row in same_day if not row.ticker.endswith(".KS")]

    def _summary(rows: list[SlippageRow], label: str) -> str:
        if not rows:
            return f"{label}: 대사 가능한 거래 없음"
        vs = [row.slippage_pct for row in rows]
        return (
            f"{label}: {len(rows)}건, 평균 {sum(vs) / len(vs):+.3%}, "
            f"중앙값 {statistics.median(vs):+.3%}, 최악 {max(vs):+.3%}"
        )

    lines = []
    if clamp_note:
        lines += [clamp_note, ""]
    lines += [
        f"대사 성공 {n}건 (당일 매칭만, 원장 전용 {len(r.ledger_only)}건 / KIS 전용 {len(r.kis_only)}건 — 매칭 안 됨)"
        + (f" / 지연 매칭 {len(delayed)}건 별도" if delayed else "")
        + (f" / 임계값 기록행 {len(threshold)}건 제외" if threshold else "")
        + (f" / 가격 불가 {len(r.unusable_price)}건 별도" if r.unusable_price else ""),
        "",
        f"전체 평균 슬리피지: {mean:+.3%}  (양수=원장가보다 불리하게 체결)",
        f"전체 중앙값: {median:+.3%}",
        f"최악: {worst:+.3%}  /  최선: {best:+.3%}",
        f"불리한 방향 비율: {unfavorable}/{n} ({unfavorable / n:.1%})",
        "",
        _summary(domestic, "국내"),
        _summary(overseas, "해외"),
        "",
        "개별 거래 (불리한 순):",
    ]
    for row in sorted(same_day, key=lambda x: -x.slippage_pct)[:15]:
        lines.append(
            f"  {row.trade_date} {row.ticker:12} {row.side:4} 원장={row.ledger_price:,.2f} "
            f"KIS={row.kis_price:,.2f} 슬리피지={row.slippage_pct:+.3%}"
        )

    if r.ledger_only:
        lines += ["", f"⚠️ 원장엔 있는데 KIS 체결내역에 없는 거래 {len(r.ledger_only)}건 (승인대기 중 취소, 14일 초과 지연, 또는 대사 실패):"]
        for ticker, d, side in r.ledger_only[:10]:
            lines.append(f"   - {d} {ticker} {side}")

    if threshold:
        lines += ["", _render_threshold_section(threshold)]

    if delayed:
        lines += ["", _render_delayed_section(delayed)]

    if r.unusable_price:
        lines += ["", _render_unusable_price_section(r.unusable_price)]

    lines += [
        "",
        "참고: order_no가 있는 거래는 KIS 체결내역과 정확히 매칭, 없는 거래(구버전 기록)만",
        "      (종목,날짜,방향) 단위로 14일 이내 그리디 매칭 — 지연 매칭은 위 통계에서 제외됨.",
    ]
    if r.account_mode == "paper":
        lines.append("      지금은 모의계좌 값이라 실계좌보다 관대할 수 있음 — 이 수치는 하한선으로 볼 것.")
    else:
        lines.append(f"      계좌 모드: {r.account_mode} — 실측치.")
    return "\n".join(lines)


def _render_threshold_section(rows: list[SlippageRow]) -> str:
    """Rows whose ledger `price` is a stop THRESHOLD, not a fill — matched
    (so their KIS execution isn't miscounted as unexplained) but excluded
    from every statistic. The gap shown is what the ledger UNDER/OVER-states
    the sale by, not slippage."""
    lines = [
        f"🚫 임계값 기록행 (원장 price가 체결가가 아니라 손절선 — 슬리피지 통계에서 제외, "
        f"아래 차이는 원장 오기록분) {len(rows)}건:",
    ]
    for row in sorted(rows, key=lambda x: -abs(x.slippage_pct))[:10]:
        lines.append(
            f"   - {row.trade_date} {row.ticker:12} {row.side:4} 원장(손절선)={row.ledger_price:,.2f} "
            f"KIS 실체결={row.kis_price:,.2f} 원장 오차={row.slippage_pct:+.3%}"
        )
    return "\n".join(lines)


def _render_unusable_price_section(rows: list[tuple[str, date, str]]) -> str:
    """Rows where a real KIS execution was matched and consumed, but the
    ledger side's price was <= 0 so slippage couldn't be computed — see
    SlippageResult.unusable_price."""
    lines = [
        f"⚠️ 원장 가격이 0 이하라 슬리피지 계산 불가 (실제 KIS 체결은 있음) {len(rows)}건:",
    ]
    for ticker, d, side in rows[:10]:
        lines.append(f"   - {d} {ticker} {side}")
    return "\n".join(lines)


def _render_delayed_section(delayed: list[SlippageRow]) -> str:
    lines = [
        f"⏱️ 지연체결 (별도, 메인 평균에서 제외 — 슬리피지가 아니라 체결 지연 동안의 시장 변동을 재는 값) {len(delayed)}건:",
    ]
    for row in sorted(delayed, key=lambda x: -x.fill_delay_days)[:10]:
        lines.append(
            f"   - {row.trade_date} {row.ticker:12} {row.side:4} 원장={row.ledger_price:,.2f} "
            f"KIS={row.kis_price:,.2f} 지연={row.fill_delay_days}일 변동={row.slippage_pct:+.3%}"
        )
    return "\n".join(lines)
