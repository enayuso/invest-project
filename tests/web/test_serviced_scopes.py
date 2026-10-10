"""Phase 6: who the background work runs for, and per-scope loading.

Two properties carry the weight:

- **A scope must be loaded before its allowlist is read.** The lists are per
  (user, mode) and empty until loaded, so a loop that skips `ensure_loaded`
  iterates nothing and checks nobody's stop-loss — silently, because an empty
  allowlist is indistinguishable from "this user permits nothing". That is a
  missed protective action, not a safe degradation.
- **One scope's failure must not skip the others.** A broker outage for user A
  is not a reason to stop checking user B's stop-losses.

These override conftest's single-scope default on purpose; that default exists
so the rest of the suite keeps the pre-Phase-6 behaviour it was written for.
"""

from __future__ import annotations

import asyncio
import contextlib

import pytest

from app.config import account_mode_scope, settings
from app.live import poller
from app.user_context import ServicedScope, user_scope
from app.web import state, trade_allowlist_store

async def _run_until(loop_coro, collected: list, count: int, *, ticks: int = 20_000) -> None:
    """Run a while-True loop until `collected` fills, then cancel AND AWAIT it.

    Awaiting the cancellation is the part that matters. A task cancelled without
    being awaited stays pending past the end of the test, its event loop closes
    underneath it, and the next file to touch anything it held fails with
    "Event loop is closed" — which reads as that file's bug, not this one's.
    취소만 하고 await하지 않으면 태스크가 테스트보다 오래 살아, 다른 파일에서
    'Event loop is closed'로 나타남.
    """
    task = asyncio.create_task(loop_coro)
    try:
        for _ in range(ticks):
            if len(collected) >= count:
                return
            await asyncio.sleep(0)
        raise AssertionError(f"loop produced {len(collected)} of {count} expected observations")
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


_ALICE = ServicedScope("u-alice", "paper")
_BOB = ServicedScope("u-bob", "real")


@pytest.fixture
def two_scopes(monkeypatch):
    """Alice on paper, Bob on real, each with their own allowlist."""
    async def scopes():
        return [_ALICE, _BOB]

    loaded: list[tuple[str, str]] = []

    async def fake_ensure(user_id, account_mode):
        loaded.append((user_id, account_mode))

    for module in (state, poller):
        monkeypatch.setattr(module, "serviced_scopes", scopes)
    monkeypatch.setattr(trade_allowlist_store, "ensure_loaded", fake_ensure)

    with user_scope(_ALICE.user_id), account_mode_scope(_ALICE.account_mode):
        settings.trade_allowlist_domestic = ["005930.KS"]
        settings.trade_allowlist_overseas = []
    with user_scope(_BOB.user_id), account_mode_scope(_BOB.account_mode):
        settings.trade_allowlist_domestic = ["000660.KS"]
        settings.trade_allowlist_overseas = ["TSLA"]
    return loaded


# ------------------------------------------------------------------ the poller


async def test_poller_polls_the_union_of_every_scope(two_scopes):
    scope, tickers = await poller.poll_plan()
    assert set(tickers) == {"005930.KS", "000660.KS", "TSLA"}
    # Bound to somebody with credentials — poll_once resolves a broker from the
    # current identity, and unbound it would fetch nothing for KRX tickers.
    assert scope == _ALICE


async def test_poller_loads_every_scope_before_reading_it(two_scopes):
    await poller.poll_plan()
    assert set(two_scopes) == {("u-alice", "paper"), ("u-bob", "real")}


async def test_poller_deduplicates_a_ticker_two_users_share(two_scopes, monkeypatch):
    """Polling per user would refetch the same ticker N times against one
    shared per-key rate limit. A price is a price."""
    with user_scope(_BOB.user_id), account_mode_scope(_BOB.account_mode):
        settings.trade_allowlist_domestic = ["005930.KS"]
        settings.trade_allowlist_overseas = []
    _, tickers = await poller.poll_plan()
    assert tickers == ["005930.KS"]


async def test_poller_skips_a_broken_scope_but_polls_the_rest(two_scopes, monkeypatch):
    async def explode_for_bob(user_id, account_mode):
        if user_id == "u-bob":
            raise RuntimeError("credential store down")

    monkeypatch.setattr(trade_allowlist_store, "ensure_loaded", explode_for_bob)
    _, tickers = await poller.poll_plan()
    assert tickers == ["005930.KS"], "alice's tickers must still be polled"


async def test_poller_does_nothing_when_nobody_is_serviced(monkeypatch):
    """No credentials anywhere is a real state — a fresh deployment. It must
    idle, not poll under an identity that cannot authenticate."""
    async def none():
        return []

    monkeypatch.setattr(poller, "serviced_scopes", none)
    scope, tickers = await poller.poll_plan()
    assert scope is None and tickers == []


# --------------------------------------------------- the realtime feed's codes


async def test_feed_subscribes_the_union_so_everyone_gets_cb_coverage(two_scopes, monkeypatch):
    """Circuit-breaker/VI status is public per-ticker data, but only for codes
    the feed actually subscribed. Built from one scope, every other user's
    domestic orders fail closed on 'circuit-breaker status unknown'."""
    async def no_snapshot(market):
        raise RuntimeError("no account")

    monkeypatch.setattr("app.execution.portfolio_risk.real_account_snapshot", no_snapshot)
    codes, _orderbook_zone, _cb_only_zone = await state._prioritized_krx_codes()
    assert set(codes) == {"005930", "000660"}, "both users' KRX codes must be subscribed"


# ------------------------------------------------------------- the stop-loss loop


async def test_stop_loss_loop_loads_each_scope_before_checking_it(two_scopes, monkeypatch):
    monkeypatch.setattr(settings, "stop_loss_check_interval_hours", 0.0)
    checked: list[tuple[str, str]] = []

    async def fake_check(ticker):
        from app.config import current_account_mode
        from app.user_context import get_current_user_id

        checked.append((get_current_user_id(), ticker))
        return "ok"

    monkeypatch.setattr(state, "check_stops_for_ticker", fake_check)

    await _run_until(state.stop_loss_loop(), checked, 3)

    # Alice's one ticker and Bob's two, each under their own identity.
    assert ("u-alice", "005930.KS") in checked
    assert ("u-bob", "000660.KS") in checked
    assert ("u-bob", "TSLA") in checked


async def test_stop_loss_loop_survives_serviced_scopes_raising(monkeypatch):
    """2026-08-26: `serviced_scopes()` itself (not a per-scope/per-ticker
    failure) is the ONE await in this loop body no inner try covered — a
    single DB-pool hiccup there used to propagate straight out of `while
    True` and stop stop-loss enforcement for the rest of the process
    (silently: this task is only ever awaited at shutdown with
    return_exceptions=True). Asserts the loop keeps calling it on the next
    interval instead of dying."""
    monkeypatch.setattr(settings, "stop_loss_check_interval_hours", 0.0)
    calls = 0

    async def flaky_scopes():
        nonlocal calls
        calls += 1
        raise RuntimeError("connection pool exhausted")

    monkeypatch.setattr(state, "serviced_scopes", flaky_scopes)

    task = asyncio.create_task(state.stop_loss_loop())
    try:
        for _ in range(20_000):
            if calls >= 2:
                break
            await asyncio.sleep(0)
        else:
            raise AssertionError(f"serviced_scopes was only called {calls} time(s); the loop appears to have died")
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


async def test_stop_loss_loop_logs_a_serviced_scopes_failure(monkeypatch, caplog):
    monkeypatch.setattr(settings, "stop_loss_check_interval_hours", 0.0)

    async def flaky_scopes():
        raise RuntimeError("connection pool exhausted")

    monkeypatch.setattr(state, "serviced_scopes", flaky_scopes)

    task = asyncio.create_task(state.stop_loss_loop())
    try:
        with caplog.at_level("ERROR", logger="app.web.state"):
            for _ in range(20_000):
                if any("could not resolve serviced scopes" in m for m in caplog.messages):
                    break
                await asyncio.sleep(0)
            else:
                raise AssertionError("expected error log never appeared")
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


async def test_stop_loss_loop_recovers_and_checks_stops_after_a_transient_scope_failure(monkeypatch):
    """The property that actually matters: not just that the loop survives,
    but that it resumes real work (checking stop-losses) once
    serviced_scopes() recovers."""
    monkeypatch.setattr(settings, "stop_loss_check_interval_hours", 0.0)
    calls = 0

    async def flaky_then_alice():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("connection pool exhausted")
        return [_ALICE]

    async def fake_ensure(user_id, account_mode):
        pass

    checked: list[str] = []

    async def fake_check(ticker):
        checked.append(ticker)
        return "ok"

    monkeypatch.setattr(state, "serviced_scopes", flaky_then_alice)
    monkeypatch.setattr(trade_allowlist_store, "ensure_loaded", fake_ensure)
    monkeypatch.setattr(state, "check_stops_for_ticker", fake_check)
    with user_scope(_ALICE.user_id), account_mode_scope(_ALICE.account_mode):
        settings.trade_allowlist_domestic = ["005930.KS"]
        settings.trade_allowlist_overseas = []

    await _run_until(state.stop_loss_loop(), checked, 1)

    assert "005930.KS" in checked


async def test_stop_loss_loop_continues_past_a_failing_scope(two_scopes, monkeypatch):
    """A scope that cannot even be loaded must not stop the others — this is the
    per-scope try, not the per-ticker one."""
    monkeypatch.setattr(settings, "stop_loss_check_interval_hours", 0.0)

    async def explode_for_alice(user_id, account_mode):
        if user_id == "u-alice":
            raise RuntimeError("cannot load alice")

    monkeypatch.setattr(trade_allowlist_store, "ensure_loaded", explode_for_alice)
    checked: list[str] = []

    async def fake_check(ticker):
        checked.append(ticker)
        return "ok"

    monkeypatch.setattr(state, "check_stops_for_ticker", fake_check)

    await _run_until(state.stop_loss_loop(), checked, 2)

    assert set(checked) >= {"000660.KS", "TSLA"}, "bob must still be checked"
    assert "005930.KS" not in checked


# ------------------------------------------------ serviced_scopes itself
#
# conftest stubs this for the whole suite, so these tests reach the real
# implementation through `conftest.REAL`.
#
# Rewritten 2026-08-07 for the single-account rollback
# (docs/single-user-rollback-guide.md Phase 4): serviced_scopes() no
# longer queries `user_broker_credentials JOIN users` (that table is gone —
# credentials are Settings.kis_credentials() env fields now) and can never
# return more than one scope. What these pin instead: no credentials
# configured -> empty list without touching the DB at all; a DB failure
# reading the stored mode preference -> empty list, same fail-idle choice as
# before; and resolve_account_mode's preference logic still applies to the
# one scope that exists.


def _set_kis_creds(monkeypatch, *, paper: bool = False, real: bool = False) -> None:
    """Fake `configured_kis_modes()` without needing real KIS-shaped values —
    only presence matters (app_key/app_secret/account_8 all truthy)."""
    from app.config import settings

    for mode, enabled in (("paper", paper), ("real", real)):
        for field in ("app_key", "app_secret", "account_8"):
            monkeypatch.setattr(settings, f"kis_{field}_{mode}", "x" if enabled else None)


async def test_serviced_scopes_returns_empty_with_no_kis_credentials(monkeypatch):
    """The ordinary state for a fresh deployment, or one whose Jenkins env
    hasn't been wired up yet — must idle, not guess an identity to trade
    under. This is a sync check (env fields), so it must short-circuit before
    ever touching the DB.
    자격증명 미설정은 흔한 정상 상태 — DB를 건드리기 전에 먼저 걸러야 함."""
    from conftest import REAL

    _set_kis_creds(monkeypatch, paper=False, real=False)
    assert await REAL["serviced_scopes"]() == []


async def test_serviced_scopes_returns_empty_when_the_preference_read_fails(monkeypatch):
    """The deliberate choice, and the one worth pinning.

    The tempting fallback is "just use the default mode" — keep working,
    sort of. That would trade under a mode nobody actually confirmed is the
    current one. Idling one interval misses a stop-loss check, a real cost;
    guessing which mode to trade in is a worse one.
    조회 실패시 폴백하지 않음 — 어느 모드인지 모르는 상태로 자동매매를 돌리는
    것보다 한 주기 쉬는 편이 낫음."""
    from app import account_mode
    from conftest import REAL

    _set_kis_creds(monkeypatch, paper=True)

    async def broken_preference():
        raise OSError("connection refused")

    monkeypatch.setattr(account_mode, "stored_preference", broken_preference)
    assert await REAL["serviced_scopes"]() == []


async def test_serviced_scopes_resolves_the_one_scope_to_its_preferred_mode(monkeypatch):
    """Both credential sets configured and a stated preference -> that
    preference wins."""
    from app import account_mode
    from app.config import settings
    from conftest import REAL

    _set_kis_creds(monkeypatch, paper=True, real=True)

    async def preferred():
        return "real"

    monkeypatch.setattr(account_mode, "stored_preference", preferred)
    assert await REAL["serviced_scopes"]() == [ServicedScope(settings.user_id, "real")]


async def test_serviced_scopes_ignores_a_preference_with_no_credentials_behind_it(monkeypatch):
    """Only paper configured, but the stored preference says real (e.g. the
    real credentials were since removed from the env) -> falls back to the
    mode that is actually usable, never silently to a mode with nothing
    behind it."""
    from app import account_mode
    from app.config import settings
    from conftest import REAL

    _set_kis_creds(monkeypatch, paper=True, real=False)

    async def preferred():
        return "real"

    monkeypatch.setattr(account_mode, "stored_preference", preferred)
    assert await REAL["serviced_scopes"]() == [ServicedScope(settings.user_id, "paper")]


async def test_serviced_scopes_defaults_to_paper_when_nothing_chosen(monkeypatch):
    """No stored preference yet, both modes configured -> paper first, never
    silently to real money."""
    from app import account_mode
    from app.config import settings
    from conftest import REAL

    _set_kis_creds(monkeypatch, paper=True, real=True)

    async def unchosen():
        return None

    monkeypatch.setattr(account_mode, "stored_preference", unchosen)
    assert await REAL["serviced_scopes"]() == [ServicedScope(settings.user_id, "paper")]


async def test_ensure_loaded_is_idempotent_per_process(monkeypatch):
    """The loops call this every pass. `load()` runs three migrations, so a
    second call for the same scope has to be a no-op."""
    from conftest import REAL

    trade_allowlist_store.forget_loaded_scopes()
    calls: list[tuple[str, str]] = []

    async def fake_load(user_id=None, account_mode=None):
        calls.append((user_id, account_mode))

    monkeypatch.setattr(trade_allowlist_store, "load", fake_load)
    try:
        await REAL["ensure_loaded"]("u-alice", "paper")
        await REAL["ensure_loaded"]("u-alice", "paper")
        await REAL["ensure_loaded"]("u-alice", "real")
    finally:
        trade_allowlist_store.forget_loaded_scopes()

    # Once per distinct scope, and mode is part of the identity.
    assert calls == [("u-alice", "paper"), ("u-alice", "real")]
