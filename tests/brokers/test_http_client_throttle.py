"""BaseHttpClient's priority-aware request throttle — no network, just the
in-process ticket queue (BaseHttpClient constructed directly, no broker)."""

from __future__ import annotations

import asyncio

import pytest

from app.brokers.base.http_client import BaseHttpClient, _LOW_STARVATION_LIMIT


def _client(interval: float = 0.01) -> BaseHttpClient:
    return BaseHttpClient(base_url="https://example.com", min_request_interval=interval)


async def test_same_priority_callers_serve_in_arrival_order():
    """Unchanged from the pre-priority-queue behavior: with nothing but
    one priority level in play, it's plain FIFO."""
    client = _client()
    order = []

    async def call(label, delay):
        await asyncio.sleep(delay)
        await client._throttle("low")
        order.append(label)

    await asyncio.gather(
        call("a", 0.0), call("b", 0.001), call("c", 0.002), call("d", 0.003),
    )
    assert order == ["a", "b", "c", "d"]


async def test_high_priority_jumps_a_queued_low_priority_ticket():
    """A "high" call that registers while "low" calls are already queued
    gets served before at least one of them — the whole point of this
    queue existing."""
    client = _client(interval=0.05)
    order = []

    async def call(label, priority, delay_before=0.0):
        if delay_before:
            await asyncio.sleep(delay_before)
        await client._throttle(priority)
        order.append(label)

    await asyncio.gather(
        call("low1", "low"),
        call("low2", "low"),
        call("low3", "low"),
        call("high1", "high", delay_before=0.01),
    )
    assert order.index("high1") < order.index("low3"), order


async def test_starvation_guard_serves_a_long_skipped_low_ticket():
    """A steady stream of "high" callers must not starve a "low" one
    forever — it must win within `_LOW_STARVATION_LIMIT` skips."""
    client = _client(interval=0.005)
    served = []

    async def low_caller():
        await client._throttle("low")
        served.append("low")

    async def high_caller(i):
        await client._throttle("high")
        served.append(f"high{i}")

    low_task = asyncio.create_task(low_caller())
    await asyncio.sleep(0.001)
    high_tasks = [asyncio.create_task(high_caller(i)) for i in range(_LOW_STARVATION_LIMIT + 10)]
    await asyncio.gather(low_task, *high_tasks)

    low_index = served.index("low")
    assert low_index == _LOW_STARVATION_LIMIT, (
        f"expected the low ticket to win right at the {_LOW_STARVATION_LIMIT}th skip, "
        f"got index {low_index}: {served}"
    )


async def test_a_tickets_own_call_returns_as_soon_as_it_wins_not_after_the_whole_queue():
    """Regression test for the bug this priority queue's first draft had:
    the ticket that starts driving the queue (because it happened to
    register when nothing else was queued) must be able to return from
    `_throttle` the moment ITS OWN ticket wins, even if many more tickets
    are still queued behind it — not only after the entire queue drains.
    Concretely: a "high" ticket starts the queue, wins the very first slot,
    but 25 more "high" tickets are queued right behind it — the starting
    ticket's own `_throttle()` call must return at tick 1, not tick 26."""
    client = _client(interval=0.005)
    finish_order = []

    async def call(label, delay_before=0.0):
        if delay_before:
            await asyncio.sleep(delay_before)
        await client._throttle("high")
        finish_order.append(label)

    starter = asyncio.create_task(call("starter"))
    await asyncio.sleep(0.001)  # let `starter` register and become the driver first
    followers = [asyncio.create_task(call(f"follower{i}")) for i in range(25)]
    await asyncio.gather(starter, *followers)

    assert finish_order[0] == "starter", (
        f"the ticket that started driving the queue should finish first (it won tick 1), "
        f"got order: {finish_order}"
    )


async def test_disabled_throttle_returns_immediately():
    """min_request_interval <= 0 means "no throttling at all" — every call
    returns without any wait, regardless of priority."""
    client = _client(interval=0)
    loop = asyncio.get_running_loop()
    start = loop.time()
    await asyncio.gather(*(client._throttle("low") for _ in range(50)))
    assert loop.time() - start < 0.05  # no meaningful delay for 50 "requests"


async def test_all_concurrent_callers_eventually_complete():
    """No ticket is silently lost or double-served under a large mixed-
    priority concurrent burst."""
    client = _client(interval=0.002)
    served = []

    async def call(label, priority):
        await client._throttle(priority)
        served.append(label)

    labels = [f"low{i}" for i in range(15)] + [f"high{i}" for i in range(15)]
    await asyncio.gather(*(
        call(label, "high" if label.startswith("high") else "low") for label in labels
    ))
    assert sorted(served) == sorted(labels)
    assert len(served) == len(set(served))  # no duplicates — nothing served twice


# ---------------------------------------------------------------------------
# max_requests_per_second (2026-09-03, decision id=1179) — a sliding 1s-window
# cap layered over the min-interval throttle. KIS counts per rolling second;
# the min interval alone only bounds the mean rate.
# ---------------------------------------------------------------------------


async def test_max_requests_per_second_caps_a_one_second_window():
    client = BaseHttpClient(
        base_url="https://example.com", min_request_interval=0.001, max_requests_per_second=5,
    )
    loop = asyncio.get_running_loop()
    stamps: list[float] = []

    async def call():
        await client._throttle("low")
        stamps.append(loop.time())

    await asyncio.gather(*(call() for _ in range(12)))

    # No trailing 1.0s window contains more than 5 grants.
    stamps.sort()
    for i, t in enumerate(stamps):
        in_window = [s for s in stamps if t - 1.0 < s <= t]
        assert len(in_window) <= 5, (i, in_window)


async def test_max_rps_zero_disables_the_window():
    loop = asyncio.get_running_loop()

    async def run_burst(**kw) -> float:
        client = BaseHttpClient(
            base_url="https://example.com", min_request_interval=0.02, **kw,
        )
        start = loop.time()
        await asyncio.gather(*(client._throttle("low") for _ in range(12)))
        return loop.time() - start

    # max_rps=0.0 must behave *identically* to the min-interval-only throttle:
    # the window gate is skipped entirely, so 12 grants take ~12 gaps of 0.02s.
    # Both bursts run on the same (possibly slow/contended) host, so compare the
    # disabled-window run against a no-window-param baseline rather than against
    # a fixed wall-clock ceiling — a loaded CI box inflates both equally.
    baseline = await run_burst()
    disabled = await run_burst(max_requests_per_second=0.0)

    # Lower bound: the min-interval spacing is still enforced (not a no-op).
    assert disabled >= 0.20, disabled
    # The window gate would hold grants back further; disabled must not be
    # materially slower than the baseline. Generous absolute slack absorbs
    # scheduler jitter under full-suite CPU contention.
    assert disabled <= baseline + 0.30, (disabled, baseline)


async def test_priority_still_wins_under_the_window_cap():
    client = BaseHttpClient(
        base_url="https://example.com", min_request_interval=0.05, max_requests_per_second=3,
    )
    order = []

    async def call(label, priority, delay_before=0.0):
        if delay_before:
            await asyncio.sleep(delay_before)
        await client._throttle(priority)
        order.append(label)

    await asyncio.gather(
        call("low1", "low"),
        call("low2", "low"),
        call("low3", "low"),
        call("high1", "high", delay_before=0.01),
    )
    assert order.index("high1") < order.index("low3"), order


# ---------------------------------------------------------------------------
# account_lane (2026-09-01) — a second, coarser gate over ledger-inquiry TRs
# only, on top of _throttle. Fixes EGW00215 (ledger-layer per-second cap).
# ---------------------------------------------------------------------------

async def test_account_lane_spaces_consecutive_entries():
    client = BaseHttpClient(
        base_url="https://example.com", min_request_interval=0.0, account_min_request_interval=0.2,
    )
    loop = asyncio.get_running_loop()
    stamps = []
    for _ in range(3):
        async with client.account_lane():
            stamps.append(loop.time())
    # 3 entries, 2 gaps of >= ~0.2s each
    assert stamps[-1] - stamps[0] >= 0.35


async def test_account_lane_is_a_noop_when_interval_is_zero():
    # Paper-account regression guard: 0.0 must not add any delay.
    client = BaseHttpClient(
        base_url="https://example.com", min_request_interval=0.0, account_min_request_interval=0.0,
    )
    loop = asyncio.get_running_loop()
    start = loop.time()
    for _ in range(20):
        async with client.account_lane():
            pass
    assert loop.time() - start < 0.05


# _AccountLaneHold.space() — FIX-3, 2026-09-03 id=1187 audit. account_lane()
# only spaced successive HOLDS of the lane, never individual wire requests
# fired WITHIN one hold — _paginate could hold the lane once and still fire
# up to MAX_PAGINATION_PAGES ledger TRs back-to-back, which is what tripped
# KIS's EGW00215 per-second limit.


async def test_account_lane_hold_space_delays_pages_within_one_hold():
    client = BaseHttpClient(
        base_url="https://example.com", min_request_interval=0.0, account_min_request_interval=0.2,
    )
    loop = asyncio.get_running_loop()
    stamps = []
    async with client.account_lane() as hold:
        for i in range(3):
            if i:
                await hold.space()
            stamps.append(loop.time())
    # Page 0 has nothing to space against yet; page 1's space() stamps the
    # lane for the first time (no prior wait either), and page 2's space()
    # is the first call that actually observes a prior stamp and waits —
    # one ~0.2s gap minimum across the 3 pages.
    assert stamps[-1] - stamps[0] >= 0.15


async def test_account_lane_hold_space_is_a_noop_when_lane_disabled():
    client = BaseHttpClient(
        base_url="https://example.com", min_request_interval=0.0, account_min_request_interval=0.0,
    )
    loop = asyncio.get_running_loop()
    start = loop.time()
    async with client.account_lane() as hold:
        for i in range(20):
            if i:
                await hold.space()
    assert loop.time() - start < 0.05


async def test_account_lane_does_not_delay_non_lane_work():
    # A caller NOT taking the lane is only subject to _throttle, even while
    # the lane is held elsewhere.
    client = BaseHttpClient(
        base_url="https://example.com", min_request_interval=0.0, account_min_request_interval=1.0,
    )
    holder_started = asyncio.Event()

    async def hold_lane():
        async with client.account_lane():
            holder_started.set()
            await asyncio.sleep(0.3)

    async def non_lane():
        await holder_started.wait()
        loop = asyncio.get_running_loop()
        t0 = loop.time()
        await client._throttle("low")  # no lane
        return loop.time() - t0

    _, elapsed = await asyncio.gather(hold_lane(), non_lane())
    assert elapsed < 0.1
