"""app.llm.metrics — in-process LLM call observability, pure (agent-
architecture-roadmap stage 0)."""

from __future__ import annotations

import asyncio
from datetime import datetime

import pytest

from app.clock import FakeClock, LiveClock, set_clock
from app.llm import metrics


@pytest.fixture(autouse=True)
def _clean():
    metrics.clear()
    set_clock(FakeClock(datetime(2026, 7, 16, 10, 0)))
    yield
    metrics.clear()
    set_clock(LiveClock())


def _record(label: str, total_tokens: int, latency: float) -> None:
    metrics.record(
        label=label, model="test-model", reasoning=False, structured=False,
        latency_seconds=latency, prompt_tokens=total_tokens - 10, completion_tokens=10,
        total_tokens=total_tokens,
    )


def test_record_stamps_timestamp_from_clock():
    _record("market_analyst", 100, 1.0)
    assert metrics.recent(1)[0].timestamp == datetime(2026, 7, 16, 10, 0)


def test_record_captures_finish_reason():
    metrics.record(
        label="x", model="test-model", reasoning=False, structured=False,
        latency_seconds=1.0, prompt_tokens=1, completion_tokens=1, total_tokens=2,
        finish_reason="length",
    )
    assert metrics.recent(1)[0].finish_reason == "length"


def test_record_finish_reason_defaults_to_none():
    _record("market_analyst", 100, 1.0)
    assert metrics.recent(1)[0].finish_reason is None


def test_recent_returns_newest_first():
    _record("a", 100, 1.0)
    _record("b", 200, 2.0)
    result = metrics.recent(2)
    assert [m.label for m in result] == ["b", "a"]


def test_recent_respects_limit():
    for i in range(5):
        _record(f"call{i}", 10, 0.1)
    assert len(metrics.recent(3)) == 3


def test_summary_groups_by_label_and_sums_tokens():
    _record("market_analyst", 100, 1.0)
    _record("market_analyst", 200, 2.0)
    _record("trader", 50, 0.5)
    result = {s.label: s for s in metrics.summary()}
    assert result["market_analyst"].call_count == 2
    assert result["market_analyst"].total_tokens == 300
    assert result["market_analyst"].mean_latency_seconds == pytest.approx(1.5)
    assert result["trader"].call_count == 1


def test_summary_sorted_by_total_tokens_descending():
    _record("cheap", 10, 0.1)
    _record("expensive", 1000, 5.0)
    labels = [s.label for s in metrics.summary()]
    assert labels == ["expensive", "cheap"]


def test_summary_handles_missing_total_tokens():
    metrics.record(
        label="no_usage", model="test-model", reasoning=False, structured=False,
        latency_seconds=0.5, prompt_tokens=None, completion_tokens=None, total_tokens=None,
    )
    result = metrics.summary()
    assert result[0].total_tokens == 0


def test_clear_empties_records():
    _record("a", 100, 1.0)
    metrics.clear()
    assert metrics.recent(10) == []
    assert metrics.summary() == []


def test_ring_buffer_is_bounded():
    for i in range(metrics._MAX_RECORDS + 10):
        _record(f"call{i}", 1, 0.01)
    assert len(metrics._records) == metrics._MAX_RECORDS
    # Oldest entries evicted first — the buffer keeps only the most recent ones.
    assert metrics._records[0].label == "call10"


def test_drain_returns_records_in_insertion_order():
    _record("a", 100, 1.0)
    _record("b", 200, 2.0)
    result = metrics.drain()
    assert [m.label for m in result] == ["a", "b"]


def test_drain_empties_the_buffer():
    _record("a", 100, 1.0)
    metrics.drain()
    assert metrics.recent(10) == []
    assert metrics.summary() == []


def test_drain_called_again_returns_empty_list():
    _record("a", 100, 1.0)
    metrics.drain()
    assert metrics.drain() == []


def test_buffer_capacity_matches_max_records():
    assert metrics.buffer_capacity() == metrics._MAX_RECORDS


# ---------------------------------------------------------------------------
# track_run_tokens / current_run_tokens
# ---------------------------------------------------------------------------


def test_current_run_tokens_is_none_outside_any_scope():
    assert metrics.current_run_tokens() is None


def test_track_run_tokens_accumulates_records_made_inside_the_scope():
    with metrics.track_run_tokens() as box:
        _record("market_analyst", 100, 1.0)
        _record("bull_researcher", 50, 1.0)
        assert box[0] == 150
        assert metrics.current_run_tokens() == 150


def test_track_run_tokens_ignores_records_made_outside_the_scope():
    _record("before", 999, 1.0)  # no scope active yet
    with metrics.track_run_tokens() as box:
        _record("inside", 10, 1.0)
    assert box[0] == 10


def test_track_run_tokens_resets_after_exiting():
    with metrics.track_run_tokens():
        _record("inside", 10, 1.0)
    assert metrics.current_run_tokens() is None
    _record("after", 20, 1.0)  # not counted anywhere now


def test_track_run_tokens_does_not_count_none_total_tokens():
    with metrics.track_run_tokens() as box:
        metrics.record(
            label="x", model="test-model", reasoning=False, structured=False,
            latency_seconds=1.0, prompt_tokens=None, completion_tokens=None, total_tokens=None,
        )
    assert box[0] == 0


def test_track_run_tokens_scopes_are_independent_when_nested_sequentially():
    # Not true concurrency (a real ContextVar/asyncio-task isolation test
    # would need actual concurrent tasks), but confirms one scope's total
    # doesn't leak into a later, separate scope — the property
    # app.orchestrator.pipeline relies on across sequential run_pipeline()
    # calls for different tickers.
    with metrics.track_run_tokens() as first:
        _record("a", 100, 1.0)
    with metrics.track_run_tokens() as second:
        _record("b", 5, 1.0)
    assert first[0] == 100
    assert second[0] == 5


async def test_track_run_tokens_isolates_concurrent_tasks():
    """The real risk _nested_sequentially above can't cover: contextvars
    are copied at asyncio.Task creation, so a scope opened INSIDE one task
    (matching real usage — one run_pipeline() call is one task) must stay
    invisible to sibling tasks running at the same time, not just
    sequential scopes."""

    async def one_run(name: str, first_tokens: int, second_tokens: int) -> int:
        with metrics.track_run_tokens() as box:
            _record(name, first_tokens, 1.0)
            await asyncio.sleep(0)  # force interleaving with the other tasks
            _record(name, second_tokens, 1.0)
            return box[0]

    results = await asyncio.gather(
        one_run("a", 10, 1),
        one_run("b", 100, 2),
        one_run("c", 1000, 3),
    )
    assert results == [11, 102, 1003]


def test_note_injection_flag_outside_any_scope_is_a_noop():
    # Mirrors note_sanitization: appending outside any track_run_injection_flags
    # scope must not raise and must record nothing.
    assert metrics.current_run_injection_flags() is None
    metrics.note_injection_flag({"agent": "x", "tier": "rest_kept", "reason": "r"})
    assert metrics.current_run_injection_flags() is None


def test_track_run_injection_flags_collects_entries_made_inside_the_scope():
    with metrics.track_run_injection_flags() as box:
        metrics.note_injection_flag({"agent": "risk_neutral", "tier": "empty", "reason": "r"})
        assert metrics.current_run_injection_flags() == [
            {"agent": "risk_neutral", "tier": "empty", "reason": "r"}
        ]
    assert metrics.current_run_injection_flags() is None
    assert box == [{"agent": "risk_neutral", "tier": "empty", "reason": "r"}]


async def test_current_run_tokens_is_none_in_a_sibling_task_without_a_scope():
    async def scoped_task() -> None:
        with metrics.track_run_tokens():
            _record("scoped", 10, 1.0)
            await asyncio.sleep(0)

    async def unscoped_task() -> int | None:
        await asyncio.sleep(0)
        return metrics.current_run_tokens()

    _, unscoped_result = await asyncio.gather(scoped_task(), unscoped_task())
    assert unscoped_result is None
