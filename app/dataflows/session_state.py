"""Shared judgment/formatting so an in-progress (still-forming) daily bar is
never labeled as a closed one.

Real incident this fixes (MRVL, memory_entries.id=1013, 2026-08-26 09:42 ET
— 12 minutes after the US market opened): the verified snapshot tool
returned the still-forming session's last=237.04/high=240.77/low=236.79/
volume=769,562 unconditionally labeled `close=`, and the Market Analyst
wrote "concluded trading on August 26 at $237.04 ... near the lower end of
its daily range" as a result. The actual close that day was $245.11 on
15.9M shares. The model itself flagged the volume as suspiciously low but
shrugged it off as "a partial update" — because nothing in the data said
outright "this session is still open." Worse, the analyst's own system
prompt tells it to treat this snapshot as authoritative, so a wrong label
isn't just uncorrected, it's amplified.

`as_of` is deliberately NOT a parameter here. NOT every caller passes the
row's own market's local date: `app/live/auto_trade_loop.py` correctly
passes `trade_date=local_today(market)`, but `app/chat/router.py` and
`app/web/routers/chat_decisions.py` pass `get_clock().today()` — a KST
date — for BOTH domestic and overseas tickers. KST = ET+13, so the US
regular session (09:30-16:00 ET) maps to KST 22:30-05:00 next day, and a
KST-dated `row_date` only equals `local_today("overseas")` during the
narrow ET 09:30-11:00 window; for the rest of the ~6.5-hour US session the
old strict `==` test failed even though the session was genuinely open,
silently emitting no session-in-progress note (id=1157-shaped failure, one
layer up from the MRVL incident above). The predicate below is therefore
`row_date >= local_today(market)`, not `==`:
- Backtest callers are still provably unaffected: a backtest always passes
  a PAST date, so `row_date >= local_today(market)` is False regardless of
  which calendar the caller used.
- The only newly-True case is a caller whose date is today-or-later on
  `market`'s own calendar while `market` is genuinely open right now — i.e.
  exactly the KST/ET skew above. `is_market_open(ticker)` is what actually
  certifies "this is intraday"; the date comparison only needs to rule out
  a stale/past row, not pin down one exact calendar day.
- This cannot introduce a `session_close_guard` false positive: that guard
  only fires on text claiming the session concluded, and while the market
  is open such a claim is simply wrong to make, so widening when the
  in-progress note fires can only prevent a miss, not cause a false one.
"""

from __future__ import annotations

from datetime import date

from app.clock import local_today
from app.live.market_hours import is_market_open


def session_in_progress(ticker: str, row_date: date) -> bool:
    """Is `row_date`'s daily bar for `ticker` still being formed right now —
    true when `row_date` is that ticker's own market's local today-or-later
    AND that market's regular session is open at this instant. (Not a strict
    `==`: see module docstring for why `>=` is needed and still safe.)

    Uses `local_today(market)`, not `get_clock().today()` (server-local,
    effectively KST) — the US regular session (9:30-16:00 ET) spans KST
    22:30 through the next day's 05:00, crossing KST midnight. Comparing
    against a KST calendar date would misjudge an in-progress US session as
    "yesterday's" for the first several hours of the KST calendar day.

    Regular session only (no pre/post-market): with yfinance's default
    prepost=False, an off-hours row is either absent or already finalized
    from the prior close, and this doesn't attempt to model a KRX after-hours
    single-price session (15:40-16:00) nudging that day's volume further —
    the same kind of known gap `app.live.market_hours` itself already
    documents (no early closes/half-days modeled either)."""
    market = "domestic" if ticker.endswith(".KS") else "overseas"
    # 과거 날짜는 항상 False(백테스트 안전), 시장이 열려있고 그 시장 달력으로
    # 오늘-이후 날짜면 True — 호출자가 KST 날짜를 넘겨도(위 독스트링) 놓치지 않음.
    return is_market_open(ticker) and row_date >= local_today(market)


def _price_position_line(last: float, high: float, low: float, *, in_progress: bool) -> str:
    """One prose line telling the model where `last`/`close` actually sits
    within [low, high] — a directional conclusion, not a computation to
    perform. Empty string when `high == low` (no range to speak of).

    The model was observed computing the range correctly but drawing the
    opposite directional read ("near the session highs" for a price sitting
    in the bottom third), so this states the band outright. Prose form only —
    no `close=`/`last=` token — so app.chat.router._SNAPSHOT_RE and
    scripts/seed_test_pending_order._SNAPSHOT_RE keep parsing the line above."""
    if high == low:
        return ""
    pos = (last - low) / (high - low)
    band = "BOTTOM" if pos < 1 / 3 else ("MIDDLE" if pos < 2 / 3 else "TOP")
    above_low = (last - low) / low * 100
    below_high = (high - last) / high * 100
    # Corrected 2026-09-11: below_high is already >= 0 for any last inside
    # [low, high], so rendering it as `{-below_high:+.1f}%` was a double
    # negative ("-4.8% below the high") that reads as ABOVE the high. Both
    # percentages are now plain positive magnitudes, with direction carried
    # only by the words "above"/"below".
    if in_progress:
        return (
            f"Position in today's range so far: last is {above_low:.1f}% above session_low "
            f"and {below_high:.1f}% below session_high — the {band} third of the range traded "
            f"so far today ({pos * 100:.1f}% of the way up from the low). Read the stock's "
            f"near-term direction from where last actually sits in this range, not from the fact "
            f"that a session high exists; do not call this \"near the session highs\" unless the "
            f"position above is in the TOP third."
        )
    return (
        f"Position in the day's range: close is {above_low:.1f}% above the low and "
        f"{below_high:.1f}% below the high — the {band} third of the day's range."
    )


def format_verified_snapshot(
    ticker: str, day: date, *, close: float, high: float, low: float, volume: int, source: str
) -> str:
    """The one place both vendor providers (app.dataflows.kis_provider,
    app.dataflows.yfinance.research) build the `VERIFIED SNAPSHOT` string an
    analyst prompt injects and treats as authoritative — previously
    duplicated as the same f-string in both files (the KIS side's docstring
    only PROMISED "same text shape," in prose, not in code), which is
    exactly the kind of drift that let one provider go on trusting the other
    to have gotten this right instead of a real code-level guarantee.

    For a closed session the FIRST line is byte-identical to the pre-existing
    format — app.chat.router's `_SNAPSHOT_RE` and
    scripts/seed_test_pending_order.py parse it, and existing tests assert it
    verbatim, so nothing about that first line may change. The text is no
    longer identical end-to-end, though: a final `source: {source}` line is
    always appended now (2026-09-10 dataflows audit, stage 2/3) — both
    vendors previously produced byte-identical output with no indication of
    which one actually answered, so a KIS hiccup (EGW00201 burst, token
    refresh failure, empty candle set) that silently fell through to the
    yfinance fallback (see `VendorRouter.call`'s broad except in
    app.dataflows.router) was invisible to both the model and any later
    transcript audit. That matters specifically for KRX (`.KS`) tickers:
    yfinance's KRX closes were measured ~2% off the official KIS figure
    during integration (see `_kis_verified_snapshot`'s docstring), so a
    fallback-served KRX snapshot also carries an explicit caveat.

    `source` is required, not defaulted: a default would silently mean
    "vendor unknown" and reintroduce exactly the ambiguity this change exists
    to remove, and it would let a future third provider skip declaring
    itself without anyone noticing at review time.
    한국거래소(KRX) 종목이 yfinance 폴백으로 서빙될 때만 오차 경고를 덧붙이고,
    그 외에는 출처만 표기한다 — 어떤 벤더가 응답했는지 사후 감사 가능하도록.

    For a session still in progress, every label changes on purpose
    (close/high/low/volume -> last/session_high/session_low/volume_so_far)
    so a model can't pattern-match the closed-session field names by
    accident, plus an explicit prose warning — belt-and-suspenders with the
    system-prompt instruction, since the prompt alone was exactly what
    failed in the MRVL incident (the model had no data-level signal to
    override its own "authoritative, so just report it" instruction)."""
    day_iso = day.isoformat()
    is_krx = ticker.endswith(".KS")
    if source == "yfinance" and is_krx:
        source_line = (
            f"source: {source} — NOT the exchange-official KIS feed. yfinance's "
            "KRX closes were measured ~2% off the official figure during "
            "integration, so treat these numbers as approximate: do not make "
            "exact-price claims from them, and do not derive a stop-loss or "
            "price-target level off them."
        )
    else:
        source_line = f"source: {source}"
    # Corrected 2026-09-11: session_in_progress() reads the wall clock via
    # is_market_open, so calling it twice could disagree if the two calls
    # straddle the market open/close instant. Call it once and reuse.
    in_progress = session_in_progress(ticker, day)
    pos_line = _price_position_line(close, high, low, in_progress=in_progress)
    if not in_progress:
        first = (
            f"VERIFIED SNAPSHOT for {ticker} as of {day_iso}: "
            f"close={close:.2f} high={high:.2f} low={low:.2f} volume={volume}"
        )
        middle = f"\n{pos_line}" if pos_line else ""
        return f"{first}{middle}\n{source_line}"
    return (
        f"VERIFIED SNAPSHOT for {ticker} as of {day_iso} "
        f"(SESSION IN PROGRESS — not a closed daily bar): "
        f"last={close:.2f} session_high={high:.2f} session_low={low:.2f} "
        f"volume_so_far={volume}\n"
        f"{pos_line + chr(10) if pos_line else ''}"
        f"(The regular session for {ticker} is still open. `last` is the most "
        f"recent trade price and will keep moving until the close; "
        f"`session_high`/`session_low`/`volume_so_far` cover only the part of "
        f"the day traded so far, so the volume WILL look small next to a full "
        f"day's — that is expected, not an anomaly. Do not describe this day as "
        f"closed, finished, settled, or 'concluded trading'; do not call this "
        f"range the daily or closing range.)\n"
        f"{source_line}"
    )


def intraday_narration_note(ticker: str, day: date) -> str:
    """Empty string unless `day` is `ticker`'s own market's still-open regular
    session — so a closed-session/backtest prompt stays byte-identical to
    before this existed, exactly like `live_session_footnote`.

    `format_verified_snapshot`'s in-progress warning only ever reached the
    Market and News+Macro Analysts (they take `verified_snapshot=`).
    2026-09-01, decision id=1157 (009150.KS/삼성전기): those two obeyed it
    ("session that remains open"), then the Portfolio Manager — which never
    saw it — re-narrated the same figures in `decision_text` and the signal
    card as "opened at 1,475,000 and closed near the day's low at 1,431,500".
    The session was open; 1,475,000 was the session HIGH (not the open) and
    `last` sat mid-range. This is `a36fab9` (the MRVL intraday-close incident,
    see this module's docstring) recurring one layer down, in the LLM
    narrative layer. This note is threaded to every downstream prose node
    (Research Manager, Trader, risk debate, Portfolio Manager) so the fact
    that the day is not over travels with the numbers.

    2026-09-14: audited all 141 live `session_close_guard` violations recorded
    to date. 88.7% (125/141) came from nodes that already receive this note —
    so the gap was not delivery, it was vocabulary — and 72% (101/141) were
    the guard's `completed_candle` tier, dominated by candlestick-pattern
    language ("distribution pattern", "long upper shadow", "shooting star",
    etc.) that this note's prohibition list never named. Added the
    candle-pattern clause below to close that gap.
    (요약: 위반의 89%가 이미 이 안내문을 받는 노드에서 발생 — 전달 문제가
    아니라 캔들패턴 어휘가 금지어 목록에 없었던 것이 원인.)"""
    if not session_in_progress(ticker, day):
        return ""
    return (
        f"SESSION IN PROGRESS: {ticker}'s regular session is still open as of "
        f"{day.isoformat()}. Today has no open, no close, and no daily range yet. "
        "Any figure for today is a live snapshot: the latest trade price is NOT "
        "the open and NOT a close, and today's high/low bound only the part of "
        "the session traded so far. Do not write that the stock opened, closed, "
        "finished, settled, or concluded trading today, do not call today's range "
        "the daily or closing range, and do not place the latest price 'near the "
        "low/high of the day' as if the day were over. Say 'so far today' or 'as "
        "of now' instead. This applies to your own prose and to any earlier "
        "report or transcript quoted below that says otherwise."
        " A candlestick pattern cannot be read from a bar that has not finished "
        "forming, so do not name one or reason from one today: no shooting star, "
        "doji, hanging man, marubozu, engulfing, long upper or lower wick or "
        "shadow, and no 'distribution pattern' or 'accumulation pattern' read off "
        "today's price action — every one of those asserts a completed bar. The "
        "distance between the latest price and today's high or low so far is not "
        "a wick and is not evidence of rejection, distribution, or accumulation; "
        "it is simply where the stock has traded up to now, and the remainder of "
        "the session can still erase or extend it. Describe what has happened so "
        "far in plain terms instead."
    )


def live_session_footnote(ticker: str, day: date) -> str:
    """One line to append under an OHLCV table when its last row is today's
    still-open session — empty string (append is then a no-op) once the
    session has closed, so the closed-session output stays byte-identical to
    before this module existed."""
    if not session_in_progress(ticker, day):
        return ""
    return (
        f"\n(NOTE: the final row, {day.isoformat()}, is TODAY'S STILL-OPEN "
        f"session — its Close is the latest trade price, and its High/Low/Volume "
        f"cover only the part of the day traded so far. Every earlier row is a "
        f"completed session.)"
    )
