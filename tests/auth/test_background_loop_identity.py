"""Background loops bind an identity per user per pass (Phase 1, item 5).

The loops run outside any request, so nothing binds an identity for them. Left
alone they inherit the ContextVar's fallback and stamp every row they write
with settings.user_id regardless of whose position or order it was — which is
invisible today (one serviced user) and becomes a cross-user data corruption
the moment it isn't. These tests pin the wiring while it is still cheap to
verify, rather than after the loops are per-user.
루프가 사용자별로 스코프를 여는지 지금 고정 — 사용자가 늘어난 뒤에는
잘못 기록돼도 눈에 띄지 않음.
"""

from __future__ import annotations

import asyncio

import pytest

from app.config import account_mode_scope, settings
from app.live import auto_trade_loop as atl
from app.user_context import ServicedScope, get_current_user_id, serviced_scopes, user_scope
from app.web import state


async def _wait_for(collected: list, count: int, *, ticks: int = 20_000) -> None:
    """Yield until `collected` reaches `count`, then give up and fail.

    An unbounded `while len(x) < n: await sleep(0)` hangs the entire suite when
    the loop under test produces nothing — which is exactly what happened when
    the allowlist became per-user and these tests still seeded one global list.
    A bounded wait turns that into a readable failure in milliseconds.
    무한 대기는 목록이 비면 전체 스위트를 멈추게 하므로 상한을 둠."""
    for _ in range(ticks):
        if len(collected) >= count:
            return
        await asyncio.sleep(0)
    raise AssertionError(
        f"loop produced {len(collected)} of {count} expected observations — "
        "it is almost certainly iterating an empty per-scope list"
    )


async def test_serviced_scopes_pairs_each_user_with_a_mode():
    """A bare user id is not enough: every store the loops touch is keyed by
    (user, mode), so a loop binding only the identity writes into whatever mode
    the process defaults to. conftest supplies one scope by default; this pins
    the SHAPE rather than the membership, which is DB-driven."""
    scopes = await serviced_scopes()
    assert all(isinstance(s, ServicedScope) and s.account_mode in ("paper", "real") for s in scopes)


async def test_stop_loss_loop_runs_each_pass_under_a_bound_identity(monkeypatch):
    # Per user, not once globally: the loop runs each pass inside
    # `user_scope`, and the allowlist is now resolved per (user, mode). Seeding
    # only the ambient scope would leave both users with an empty allowlist,
    # the loop would iterate no tickers, and the wait below would spin forever.
    # 루프가 사용자별 스코프로 실행되므로 목록도 사용자별로 넣어야 함.
    for _u in ("u-alice", "u-bob"):
        with user_scope(_u), account_mode_scope("paper"):
            settings.trade_allowlist_domestic = ["005930.KS"]
            settings.trade_allowlist_overseas = []
    monkeypatch.setattr(settings, "stop_loss_check_interval_hours", 0.0)

    async def fake_scopes() -> list[ServicedScope]:
        # Overrides conftest's single-scope default: these tests exist to prove
        # each scope gets its own pass.
        return [ServicedScope("u-alice", "paper"), ServicedScope("u-bob", "paper")]

    seen: list[str] = []

    async def fake_check(ticker: str) -> str:
        # Read inside the callee, which is where every real persistence call
        # reads it too.
        seen.append(get_current_user_id())
        return "ok"

    monkeypatch.setattr(state, "serviced_scopes", fake_scopes)
    monkeypatch.setattr(state, "check_stops_for_ticker", fake_check)

    task = asyncio.create_task(state.stop_loss_loop())
    await _wait_for(seen, 2)
    task.cancel()

    assert seen[:2] == ["u-alice", "u-bob"]


async def test_stop_loss_loop_restores_identity_after_a_failing_ticker(monkeypatch):
    """A raising check must not leave the next user's pass running as the
    previous one — user_scope's finally is what guarantees it, and a future
    refactor that moves the try/except outside the scope would break it
    silently."""
    # Per user, not once globally: the loop runs each pass inside
    # `user_scope`, and the allowlist is now resolved per (user, mode). Seeding
    # only the ambient scope would leave both users with an empty allowlist,
    # the loop would iterate no tickers, and the wait below would spin forever.
    # 루프가 사용자별 스코프로 실행되므로 목록도 사용자별로 넣어야 함.
    for _u in ("u-alice", "u-bob"):
        with user_scope(_u), account_mode_scope("paper"):
            settings.trade_allowlist_domestic = ["005930.KS"]
            settings.trade_allowlist_overseas = []
    monkeypatch.setattr(settings, "stop_loss_check_interval_hours", 0.0)

    async def fake_scopes() -> list[ServicedScope]:
        # Overrides conftest's single-scope default: these tests exist to prove
        # each scope gets its own pass.
        return [ServicedScope("u-alice", "paper"), ServicedScope("u-bob", "paper")]

    seen: list[str] = []

    async def exploding_check(ticker: str) -> str:
        seen.append(get_current_user_id())
        raise RuntimeError("broker down")

    monkeypatch.setattr(state, "serviced_scopes", fake_scopes)
    monkeypatch.setattr(state, "check_stops_for_ticker", exploding_check)

    task = asyncio.create_task(state.stop_loss_loop())
    await _wait_for(seen, 2)
    task.cancel()

    assert seen[:2] == ["u-alice", "u-bob"]
    # And the loop's own context is clean afterwards.
    assert get_current_user_id() == settings.user_id


async def test_auto_trade_loop_runs_each_cycle_under_a_bound_identity(monkeypatch):
    """The loop that can place real orders. Its bound identity decides whose
    credentials are used (Phase 2) and whose ledger the fills land in."""
    monkeypatch.setattr(settings, "auto_trade_interval_minutes", 0.0)

    async def fake_scopes() -> list[ServicedScope]:
        # Overrides conftest's single-scope default: these tests exist to prove
        # each scope gets its own pass.
        return [ServicedScope("u-alice", "paper"), ServicedScope("u-bob", "paper")]

    seen: list[str] = []

    async def fake_cycle(user_id: str, interval_seconds: float) -> None:
        seen.append((user_id, get_current_user_id()))

    monkeypatch.setattr(atl, "serviced_scopes", fake_scopes)
    monkeypatch.setattr(atl, "_run_auto_trade_cycle", fake_cycle)

    task = asyncio.create_task(atl.auto_trade_loop())
    await _wait_for(seen, 2)
    task.cancel()

    # The bound identity must match the user the cycle was invoked for — the
    # pair, not just the presence of a binding, is the thing that can drift.
    assert seen[:2] == [("u-alice", "u-alice"), ("u-bob", "u-bob")]


async def test_auto_trade_cycle_does_not_sleep_on_its_own(monkeypatch):
    """Regression guard: the interval sleep belongs to the loop, not the cycle.
    A sleep left inside the per-user cycle would serialize users behind one
    full interval each instead of one interval per full pass."""
    import inspect

    source = inspect.getsource(atl._run_auto_trade_cycle)
    assert "asyncio.sleep" not in source
