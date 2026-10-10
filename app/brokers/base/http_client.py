# app/brokers/base/http_client.py
# Generic throttled-HTTP-client plumbing shared by every broker adapter.
# Broker-specific concerns (auth headers, request retry conditions, response
# envelope checking) do NOT belong here — each broker's own client subclass
# (e.g. app/brokers/kis/client.py) supplies those.
# 브로커 공통 HTTP 기반 — throttle/lifecycle만 담당.
# 인증 헤더/응답 파싱 등 브로커별 로직은 각 브로커의 client.py에서 구현.

from __future__ import annotations

import asyncio
import contextlib
from collections import deque

import httpx

# Only two tiers: order/account-safety calls ("high") vs everything else
# ("low", the default — routine quotes/rank/news/financials/analysis data).
# A caller passing an unrecognized string is treated as "low" by `_RANK.get`
# rather than raising — the explicit-opt-in-to-high convention (see
# KISHttpClient._request) means an unrecognized value is never supposed to
# mean "high" by accident.
_RANK = {"high": 0, "low": 1}
# After this many consecutive ticks where a "low" ticket lost out to a
# "high" one, that low ticket is served next regardless — without this, a
# steady stream of high-priority calls (e.g. auto-trade placing orders
# across many tickers) could starve routine polling (rank/news/candles)
# indefinitely.
_LOW_STARVATION_LIMIT = 20


class _Ticket:
    __slots__ = ("rank", "skipped", "served", "event")

    def __init__(self, rank: int) -> None:
        self.rank = rank
        self.skipped = 0
        self.served = False
        # Woken by whichever coroutine is currently driving `_elect_one`
        # when THIS ticket wins a round — never by the ticket's own holder
        # acquiring a lock. Waiting on a shared lock's FIFO order instead of
        # a per-ticket event was the first (buggy) version of this: a
        # ticket could be marked "won" by priority but its coroutine
        # couldn't actually return from `_throttle` until lock-acquisition
        # order (= arrival order) let it back in, silently undoing the
        # priority reordering. See `BaseHttpClient._throttle`'s docstring.
        self.event = asyncio.Event()


class _AccountLaneHold:
    """Handle yielded by `BaseHttpClient.account_lane()`. `await hold.space()`
    enforces `account_min_request_interval` between individual wire requests
    made WITHIN one lane hold — distinct from the wait `account_lane()`
    itself performs on entry, which only spaces successive *holds* of the
    lane, not the requests fired once inside one. `_paginate` (see
    app/brokers/kis/client.py) can hold the lane for up to
    MAX_PAGINATION_PAGES ledger TRs in a row; without per-page spacing here,
    those pages fired back-to-back at the ordinary `_min_request_interval`
    only, occasionally fast enough to trip KIS's own per-second ledger
    quota (EGW00215) — the id=1187 fail-closed BUY block, 2026-09-03.
    No-op when the lane itself is disabled (`_account_min_interval <= 0`).
    원장조회 레인 보유 구간 안에서 페이지 사이 간격을 강제 — 레인 진입 간격과는 별개."""

    __slots__ = ("_client",)

    def __init__(self, client: "BaseHttpClient") -> None:
        self._client = client

    async def space(self) -> None:
        client = self._client
        if client._account_min_interval <= 0:
            return
        loop = asyncio.get_running_loop()
        if client._account_lane_last is not None:
            wait = client._account_lane_last + client._account_min_interval - loop.time()
            if wait > 0:
                await asyncio.sleep(wait)
        client._account_lane_last = loop.time()


class BaseHttpClient:
    """Owns the shared httpx.AsyncClient, a min-interval request throttle,
    and lifecycle (`aclose`) — the part that's identical across brokers.
    한 요청 간격 스로틀링 및 httpx 클라이언트 수명주기 관리 (브로커 공통)

    The throttle is priority-aware: `_throttle(priority="high")` (order
    submission/cancellation, buyable/sellable, balance, executions, stock/VI
    status — see each KIS mixin's call sites) gets the next available slot
    ahead of "low"-priority callers (quotes, rank, news, financials —
    anything that doesn't affect an order or safety-gate decision), subject
    to the starvation guard above. No priority concept existed before this;
    every caller shared one plain FIFO wait.
    우선순위 인지형 스로틀 — 주문/계좌안전 관련 호출이 일반 시세/랭킹 조회보다
    먼저 처리되도록(기아방지 안전장치 포함). 이전엔 순수 도착순(FIFO)이었음.

    `max_requests_per_second` (0 = disabled) is a SECOND gate layered over the
    min-interval one inside `_elect_one`: KIS counts requests per rolling 1s
    window, and the min interval alone only bounds the *mean* rate — under a
    busy event loop, grants that were correctly spaced at grant time still
    reach the wire in clusters and trip the gateway's per-second cap (EGW00201,
    2026-09-03 decision id=1179). This gate holds back a grant until fewer than
    N of them fall inside the trailing second.
    KIS는 1초 슬라이딩 윈도우로 카운트 — 최소간격은 평균만 제한하므로 별도 게이트 추가."""

    def __init__(
        self,
        *,
        base_url: str,
        client: httpx.AsyncClient | None = None,
        min_request_interval: float = 1.0,
        account_min_request_interval: float = 0.0,
        max_requests_per_second: float = 0.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        # Injectable for tests (httpx.MockTransport); owned client otherwise.
        self._client = client or httpx.AsyncClient(timeout=15.0)
        self._min_request_interval = min_request_interval
        # Sliding 1s-window request cap layered on top of the min interval —
        # see the class docstring. 0.0 disables it (paper: the 1.0s interval
        # already bounds the window). `_grant_times` holds the loop.time() of
        # every grant still inside the trailing second, pruned in `_elect_one`.
        # 1초 슬라이딩 윈도우 요청 상한 (0=비활성).
        self._max_rps = max_requests_per_second
        self._grant_times: deque[float] = deque()
        # A SECOND, coarser gate layered ON TOP OF `_throttle` for ledger-side
        # inquiry TRs only (balance, present-balance, executions, buyable/
        # sellable) — KIS's ledger layer enforces its own per-second cap
        # (EGW00215) that is a different counter from the gateway's EGW00201
        # one `_throttle` spaces. `priority="high"` puts all of these at the
        # front of `_throttle`'s queue with only `min_request_interval`
        # between them, so a market_lock hold firing 3-5 balance TRs bursts
        # far past the ledger cap (2026-09-01 real-account incident). 0.0
        # disables the lane (paper: the 1.0s global interval already bounds
        # it). Order submission/cancel/revise deliberately NEVER take this
        # lane — queuing a stop-loss sell behind inquiry traffic costs more
        # than one EGW00215.
        # 잔고·체결내역 등 원장 조회 TR만 별도 최소간격으로 직렬화 — 주문 경로는 제외.
        self._account_min_interval = account_min_request_interval
        self._account_lane_lock = asyncio.Lock()
        self._account_lane_last: float | None = None
        # `None` (not 0.0) — see `_elect_one`'s first-tick handling. 0.0 let
        # the very first tick of this client's lifetime skip the interval
        # sleep entirely (`loop.time()` is an arbitrary-epoch monotonic
        # clock, typically already far larger than 0.0 + min_request_interval),
        # harmless under plain FIFO but breaking priority fairness: a cold-
        # start burst would let whichever ticket happened to register first
        # win outright, before any concurrently-registering higher-priority
        # ticket even had a chance to be seen.
        self._last_request_at: float | None = None
        # Guards `_waiters` (append/remove) and the `_electing` flag — held
        # only across plain synchronous list/bool operations, never across
        # an await, so it's always instantaneous to acquire/release.
        self._register_lock = asyncio.Lock()
        self._waiters: list[_Ticket] = []
        # True while a `_drive` task is actively running `_elect_one` on
        # behalf of everyone currently queued. Exactly one such task ever
        # runs at a time — started atomically at registration time (see
        # `_throttle`) — not a lock itself, since whoever ISN'T driving must
        # never block waiting for it (they wait on their own ticket's
        # `event` instead).
        self._electing = False
        # Keeps the detached `_drive` task alive — asyncio only holds a weak
        # reference to a task created via `ensure_future`/`create_task`, so
        # without this it could be garbage-collected mid-run (a real, if
        # rare, asyncio footgun) since no caller ever awaits it directly.
        self._background_tasks: set[asyncio.Task] = set()

    async def _throttle(self, priority: str = "low") -> None:
        if self._min_request_interval <= 0:
            return
        ticket = _Ticket(rank=_RANK.get(priority, _RANK["low"]))
        async with self._register_lock:
            self._waiters.append(ticket)
            become_elector = not self._electing
            if become_elector:
                self._electing = True
        if become_elector:
            # Detached, NOT awaited inline — driving the queue to completion
            # can outlast this specific ticket (e.g. this ticket is "high"
            # and wins immediately, but 20 "low" tickets are still queued
            # behind it); this call must still return as soon as ITS OWN
            # ticket is served, not after the whole queue drains. An
            # earlier version awaited this loop inline and one caller ended
            # up waiting for every other caller's turn too, silently
            # defeating the priority ordering for whichever ticket started
            # the queue (see `_drive`'s docstring for a walk-through of the
            # bug, in case it looks tempting to inline this again).
            task = asyncio.ensure_future(self._drive())
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)
        # Every ticket (including the elector's own) confirms via its event
        # rather than assuming `served` — the detached `_drive` task may
        # serve other (higher-priority) tickets first and reach this one
        # later, in any order.
        await ticket.event.wait()

    async def _drive(self) -> None:
        """Repeatedly calls `_elect_one` until `_waiters` is empty, then
        clears `_electing` so the next registrant starts a fresh one.
        Deliberately a detached background task (`asyncio.ensure_future` in
        `_throttle`, never awaited inline by the ticket that started it) —
        an earlier version had the starting caller's own `_throttle` await
        this loop directly, which meant that caller's `_throttle` call
        couldn't return until the ENTIRE queue drained, not just until its
        own ticket won. Concretely: ticket A starts driving, wins
        immediately (tick 1), but 20 more tickets are still queued behind
        it — inline, A's own `_throttle` wouldn't return until all 20 of
        those were ALSO dispatched, even though A's own turn was long over.
        Detaching this into its own task means `_throttle` only ever waits
        on its own ticket's `event`, regardless of who happens to be
        driving."""
        while True:
            await self._elect_one()
            async with self._register_lock:
                if not self._waiters:
                    self._electing = False
                    return

    async def _elect_one(self) -> None:
        """Sleeps out the rest of the current interval, then serves exactly
        one queued ticket — the oldest "high" one, unless an even-older
        "low" one has already been skipped past `_LOW_STARVATION_LIMIT`
        ticks, in which case that starved low ticket goes instead. List
        order is arrival order (append-only in `_throttle`, never
        reordered), so `[t for t in self._waiters if t.rank == R][0]` is
        always the oldest still-waiting ticket of rank R — no separate
        sequence counter needed.

        The very first tick of this client's lifetime (`_last_request_at`
        still `None`) always waits out a full interval rather than firing
        immediately — the one deliberate behavior change from the pre-
        priority-queue throttle. Without it, a cold-start burst (several
        requests registering within microseconds of each other, e.g. the
        dashboard's initial page load) would let whichever ticket happened
        to reach here first win outright, before any concurrently-
        registering higher-priority ticket even had a chance to be seen —
        the exact fairness this queue exists to provide. Costs one
        `min_request_interval` of one-time startup latency in exchange."""
        loop = asyncio.get_running_loop()
        wait = (
            self._min_request_interval if self._last_request_at is None
            else self._last_request_at + self._min_request_interval - loop.time()
        )
        if wait > 0:
            await asyncio.sleep(wait)
        # Second gate: hold this grant until the trailing 1s window has room
        # for it. Prunes expired stamps, then sleeps out the oldest one if the
        # window is full — re-checked in a loop because another driver tick (or
        # a fresh expiry) can change the picture across the sleep.
        if self._max_rps > 0:
            while True:
                now = loop.time()
                while self._grant_times and now - self._grant_times[0] >= 1.0:
                    self._grant_times.popleft()
                if len(self._grant_times) < self._max_rps:
                    break
                await asyncio.sleep(self._grant_times[0] + 1.0 - now)
        self._last_request_at = loop.time()
        if self._max_rps > 0:
            self._grant_times.append(loop.time())
        async with self._register_lock:
            low = [t for t in self._waiters if t.rank == _RANK["low"]]
            high = [t for t in self._waiters if t.rank == _RANK["high"]]
            if low and low[0].skipped >= _LOW_STARVATION_LIMIT:
                winner = low[0]
            elif high:
                winner = high[0]
                for t in low:
                    t.skipped += 1
            else:
                winner = low[0]
            self._waiters.remove(winner)
        winner.served = True
        winner.event.set()

    @contextlib.asynccontextmanager
    async def account_lane(self):
        """Serializes and spaces ledger-side inquiry TRs at
        `account_min_request_interval`, on top of (not instead of) the
        ordinary `_throttle`. Re-entrant use is NOT supported — the lock is a
        plain asyncio.Lock, so a paginating caller must take the lane once
        around its whole loop and pass `lane=None` to the inner `_request`s.
        No-op (`nullcontext`) when the interval is <= 0.
        원장 조회 TR 전용 최소간격 게이트 — 재진입 불가.

        Yields an `_AccountLaneHold` handle (even when the lane is
        disabled — the handle's own `space()` is then a no-op) rather than
        `None`, so a caller holding the lane across a multi-page loop
        (`_paginate` with `lane="account"`) can additionally space out
        individual page requests WITHIN the hold: this method's own
        pre-hold wait only throttles entries into the lane, not the pages
        fired once inside it, and `_min_request_interval` alone let
        `_paginate` fire up to `MAX_PAGINATION_PAGES` ledger TRs back-to-back
        (the EGW00215 fail-closed block on id=1187, 2026-09-03)."""
        if self._account_min_interval <= 0:
            yield _AccountLaneHold(self)
            return
        async with self._account_lane_lock:
            loop = asyncio.get_running_loop()
            if self._account_lane_last is not None:
                wait = self._account_lane_last + self._account_min_interval - loop.time()
                if wait > 0:
                    await asyncio.sleep(wait)
            try:
                yield _AccountLaneHold(self)
            finally:
                self._account_lane_last = asyncio.get_running_loop().time()

    async def aclose(self) -> None:
        await self._client.aclose()
