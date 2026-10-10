"""app.llm.client.PriorityLimiter — pure asyncio coordination, no network.
See its own docstring for why this exists (ai-engine-24h-operation-analysis
P0-1 ③): the LLM client had no concurrency cap or priority concept at all,
unlike KIS's order/safety-vs-routine-polling throttle."""

from __future__ import annotations

import asyncio

from app.llm.client import PriorityLimiter, _LOW_STARVATION_LIMIT


async def test_acquire_grants_immediately_while_under_the_cap():
    limiter = PriorityLimiter(max_concurrent=2)
    await limiter.acquire("low")
    await limiter.acquire("high")  # second slot, still under cap — must not block
    # No third acquire attempted — this test only asserts the first two
    # returned without needing a release, i.e. they didn't queue.


async def test_release_with_no_waiters_frees_the_slot():
    limiter = PriorityLimiter(max_concurrent=1)
    await limiter.acquire("low")
    await limiter.release()
    # The slot must be free again — a second acquire should grant immediately.
    acquired = False

    async def try_acquire():
        nonlocal acquired
        await limiter.acquire("low")
        acquired = True

    await asyncio.wait_for(try_acquire(), timeout=1.0)
    assert acquired


async def test_high_priority_waiter_is_served_before_an_earlier_low_priority_one():
    limiter = PriorityLimiter(max_concurrent=1)
    order = []

    async def worker(name, priority):
        await limiter.acquire(priority)
        order.append(name)
        await limiter.release()

    await limiter.acquire("low")  # occupy the only slot
    order.append("initial")
    low_task = asyncio.create_task(worker("queued-low", "low"))
    await asyncio.sleep(0)  # let it register as a waiter before the next one
    high_task = asyncio.create_task(worker("queued-high", "high"))
    await asyncio.sleep(0)

    await limiter.release()  # frees the initial slot — waiters now compete
    await asyncio.gather(low_task, high_task)

    assert order == ["initial", "queued-high", "queued-low"]


async def test_fifo_among_waiters_of_the_same_priority():
    limiter = PriorityLimiter(max_concurrent=1)
    order = []

    async def worker(name):
        await limiter.acquire("low")
        order.append(name)
        await limiter.release()

    await limiter.acquire("low")
    first = asyncio.create_task(worker("first"))
    await asyncio.sleep(0)
    second = asyncio.create_task(worker("second"))
    await asyncio.sleep(0)

    await limiter.release()
    await asyncio.gather(first, second)

    assert order == ["first", "second"]


async def test_unrecognized_priority_is_treated_as_low():
    limiter = PriorityLimiter(max_concurrent=1)
    order = []

    async def worker(name, priority):
        await limiter.acquire(priority)
        order.append(name)
        await limiter.release()

    await limiter.acquire("low")
    weird_task = asyncio.create_task(worker("weird", "not-a-real-priority"))
    await asyncio.sleep(0)
    high_task = asyncio.create_task(worker("high", "high"))
    await asyncio.sleep(0)

    await limiter.release()
    await asyncio.gather(weird_task, high_task)

    assert order == ["high", "weird"]  # "weird" ranked as low, so high still wins


async def test_starvation_guard_eventually_serves_a_low_priority_waiter():
    # A "low" waiter that's been skipped _LOW_STARVATION_LIMIT times must be
    # served next regardless of any "high" waiters still behind it —
    # otherwise a steady stream of live calls could starve out a backtest
    # indefinitely (see PriorityLimiter's own docstring).
    limiter = PriorityLimiter(max_concurrent=1)
    order = []

    async def low_worker():
        await limiter.acquire("low")
        order.append("low")
        await limiter.release()

    async def high_worker(name):
        await limiter.acquire("high")
        order.append(name)
        await limiter.release()

    await limiter.acquire("low")  # occupy the slot
    low_task = asyncio.create_task(low_worker())
    await asyncio.sleep(0)

    # Queue enough "high" arrivals, one release at a time, that "low" would
    # be skipped past the starvation limit if rank alone decided every round.
    high_tasks = []
    for i in range(_LOW_STARVATION_LIMIT + 2):
        high_tasks.append(asyncio.create_task(high_worker(f"high-{i}")))
        await asyncio.sleep(0)
        await limiter.release()
        await asyncio.sleep(0)

    await asyncio.gather(low_task, *high_tasks)

    assert "low" in order
    # It must not have been served dead last after every single high waiter —
    # the guard should have pulled it forward once its skip count hit the limit.
    assert order.index("low") <= _LOW_STARVATION_LIMIT + 1


# ---------------------------------------------------------------------------
# 2026-08-26 fix: a waiter cancelled while parked in event.wait() used to
# stay in _waiters forever, permanently leaking a slot (release() pops the
# ghost waiter and sets its abandoned event without decrementing
# _in_flight). With llm_max_concurrent_requests=8, 8 such cancellations
# silently wedge the whole AI pipeline with no timeout and no exception.
# ---------------------------------------------------------------------------


async def test_cancelled_waiter_does_not_leak_its_slot():
    limiter = PriorityLimiter(max_concurrent=1)
    await limiter.acquire("low")  # occupy the only slot

    task = asyncio.create_task(limiter.acquire("low"))
    await asyncio.sleep(0)  # let it register as a waiter
    assert len(limiter._waiters) == 1

    task.cancel()
    await asyncio.gather(task, return_exceptions=True)

    assert limiter._waiters == []
    assert limiter._in_flight == 1  # only the original holder still counts

    await limiter.release()
    # The slot must be free again — a fresh acquire must grant immediately,
    # not hang forever the way the pre-fix leak would.
    await asyncio.wait_for(limiter.acquire("low"), timeout=1.0)


async def test_cancel_after_slot_handoff_returns_the_slot():
    """The second, subtler leak variant: release() has already popped the
    waiter and set its event (handing it the slot) before the waiter task
    gets to resume — asyncio can still deliver a pending cancellation on
    that next resumption instead of silently dropping it. The slot must be
    handed back/onward, not lost."""
    limiter = PriorityLimiter(max_concurrent=1)
    await limiter.acquire("low")  # occupy the only slot

    task = asyncio.create_task(limiter.acquire("low"))
    await asyncio.sleep(0)  # let it register as a waiter

    # Hand the slot to the waiter WITHOUT letting the waiter task run again
    # first — release()'s own body has no await, so this reliably reaches
    # `chosen.event.set()` (making the waiter's event set) before we cancel.
    await limiter.release()
    assert limiter._waiters == []
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)

    assert limiter._in_flight == 0
    assert limiter._waiters == []
    await asyncio.wait_for(limiter.acquire("low"), timeout=1.0)


async def test_cancel_after_handoff_passes_slot_to_the_next_waiter():
    limiter = PriorityLimiter(max_concurrent=1)
    await limiter.acquire("low")  # occupy the only slot

    first = asyncio.create_task(limiter.acquire("low"))
    await asyncio.sleep(0)
    second = asyncio.create_task(limiter.acquire("low"))
    await asyncio.sleep(0)

    await limiter.release()  # hands the slot to `first`
    first.cancel()  # `first` gives it straight back up without ever running
    await asyncio.gather(first, return_exceptions=True)

    # `second` must now have (or be about to get) the slot instead of the
    # pool being wedged with 0 waiters and a lost slot.
    await asyncio.wait_for(second, timeout=1.0)
    assert limiter._in_flight == 1


async def test_cancellation_is_reraised_not_swallowed():
    limiter = PriorityLimiter(max_concurrent=1)
    await limiter.acquire("low")

    task = asyncio.create_task(limiter.acquire("low"))
    await asyncio.sleep(0)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)

    assert task.cancelled()


async def test_repeated_cancellations_do_not_exhaust_the_pool():
    """Regression for the exact reported failure mode: enough cancellations
    to match llm_max_concurrent_requests must not leave the pool
    permanently short a slot."""
    limiter = PriorityLimiter(max_concurrent=2)
    await limiter.acquire("low")
    await limiter.acquire("low")  # both slots occupied

    for _ in range(10):
        task = asyncio.create_task(limiter.acquire("low"))
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    assert limiter._in_flight == 2
    assert limiter._waiters == []

    await limiter.release()
    await limiter.release()
    # Both slots must be genuinely free — not permanently short from the
    # cancellations above.
    await asyncio.wait_for(limiter.acquire("low"), timeout=1.0)
    await asyncio.wait_for(limiter.acquire("low"), timeout=1.0)


async def test_cancelled_waiter_does_not_distort_starvation_counters():
    limiter = PriorityLimiter(max_concurrent=1)
    await limiter.acquire("low")  # occupy the slot

    doomed = asyncio.create_task(limiter.acquire("high"))
    await asyncio.sleep(0)
    doomed.cancel()
    await asyncio.gather(doomed, return_exceptions=True)

    order = []

    async def worker(name, priority):
        await limiter.acquire(priority)
        order.append(name)
        await limiter.release()

    low_task = asyncio.create_task(worker("queued-low", "low"))
    await asyncio.sleep(0)
    high_task = asyncio.create_task(worker("queued-high", "high"))
    await asyncio.sleep(0)

    await limiter.release()
    await asyncio.gather(low_task, high_task)

    # The cancelled waiter's absence must not have skewed anything — normal
    # priority ordering (high before low) still holds.
    assert order == ["queued-high", "queued-low"]
