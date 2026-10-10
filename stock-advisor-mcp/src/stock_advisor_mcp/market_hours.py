"""Pure session-status decision: "is today's daily bar final yet?"

No I/O, no yfinance, no cache, no MCP. The current time is always passed
in, so every rule here is unit-testable without a clock or a network.

Why this exists (2026-10-07 observation): at 15:01 KST -- inside the KRX
regular session, which runs until 15:30 -- yfinance reported
``marketState == "POST"``. The server trusted that single field, called an
intraday price a "close", and computed indicators on today's unfinished
bar. yfinance daily bars are indexed by Asia/Seoul dates and today's bar
exists during the session, so the bar's *date* plus the KST clock is a
far more reliable signal than ``marketState``.

``marketState`` is therefore only used in the conservative direction
("REGULAR" can keep a bar in progress, never close it).
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone

KST = timezone(timedelta(hours=9))

# Regular session closes 15:30 KST and yfinance quotes are ~20 minutes
# delayed. Some days the session runs late (e.g. the CSAT day closes at
# 16:30, +20 min delay = 16:50). 17:00 covers both with margin. Cost: on
# normal weekdays 15:30-17:00 today's bar is still reported as
# in-progress. Late is better than wrong: we prefer reporting a close
# late over calling an intraday price a close.
CLOSE_CONFIRM_TIME_KST = time(17, 0)


def decide_session_status(
    last_bar_date: date, now_kst: datetime, market_state: str | None
) -> tuple[str, str]:
    """Return ``(session_status, session_basis)``.

    Rules, first match wins:

    - last bar before today (pre-open, holiday, weekend)  -> closed / no_bar_today
    - last bar after today (clock anomaly)                -> in_progress / bar_date_after_today
    - last bar today and marketState == "REGULAR"         -> in_progress / market_state_regular
    - last bar today and now < CLOSE_CONFIRM_TIME_KST     -> in_progress / before_close_confirm_time
    - last bar today and now >= CLOSE_CONFIRM_TIME_KST    -> closed / after_close_confirm_time

    ``now_kst`` that is aware in another timezone is converted to KST; a
    naive value is taken to already be KST.
    """
    if now_kst.tzinfo is None:
        now = now_kst
    else:
        now = now_kst.astimezone(KST)
    today = now.date()

    if last_bar_date < today:
        return "closed", "no_bar_today"
    if last_bar_date > today:
        return "in_progress", "bar_date_after_today"
    if market_state == "REGULAR":
        return "in_progress", "market_state_regular"
    if now.time() < CLOSE_CONFIRM_TIME_KST:
        return "in_progress", "before_close_confirm_time"
    return "closed", "after_close_confirm_time"
