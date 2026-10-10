"""Shared test helpers.

Created during the multi-user migration (Phase 2), when `app.web.state`'s
process-wide `_kis_broker` global became the per-user coroutine
`state.kis_broker()`. Tests that used to inject a fake with

    state.kis_broker = broker_stub(FakeBroker())

now write

    state.kis_broker = broker_stub(FakeBroker())

which keeps the same save-and-restore fixture shape while matching what the
production accessor actually is.
"""

from __future__ import annotations

import importlib

import pytest

# Real implementations of the things the autouse fixtures below stub out, so a
# test that means to exercise them can reach past its own safety net.
REAL: dict[str, object] = {}


@pytest.fixture(autouse=True)
def _one_serviced_scope(monkeypatch):
    """Default every test to exactly one serviced scope: the ambient identity.

    Phase 6 made the background loops, the poller and the realtime feed resolve
    WHO they work for from Postgres (`user_context.serviced_scopes` — users with
    stored KIS credentials). Unstubbed, that has two effects on the suite, both
    bad and neither about the code under test:

      1. With no credentials in a unit test it returns `[]`, so every
         union-building function yields nothing and tests assert `[] == [...]`.
      2. Where it does reach the dev Postgres, the first test to build a pool
         leaves it bound to a closed event loop and later tests fail with
         `Event loop is closed` — the phantom-failure class in the handoff.

    One scope built from `settings.user_id` + the deployment mode reproduces
    exactly the pre-Phase-6 behaviour these tests were written against, with the
    identity now explicit rather than implied. Tests that care about multiple
    scopes override this — see tests/web/test_serviced_scopes.py.
    Phase 6 이전 동작(암묵적 단일 사용자)을 명시적으로 재현 — 다중 스코프를
    검증하는 테스트는 각자 덮어씀.

    Patched at each module's own reference because they all `from ... import
    serviced_scopes`, so each holds its own binding.
    각 모듈이 자체 참조를 가지므로 모두 패치해야 함."""
    from app import user_context
    from app.config import settings

    # Stash the real implementations so a test that specifically exercises them
    # can, rather than being stuck with the stub. Without this the stub is
    # unavoidable and `serviced_scopes` / `ensure_loaded` have no direct tests
    # at all — the two functions Phase 6 turns on.
    # 스텁 때문에 원본을 직접 검증할 수 없게 되는 것을 방지.
    REAL["serviced_scopes"] = user_context.serviced_scopes

    async def _one():
        return [user_context.ServicedScope(settings.user_id, settings.kis_account_mode)]

    monkeypatch.setattr(user_context, "serviced_scopes", _one)
    for module_path in ("app.web.state", "app.live.poller", "app.live.auto_trade_loop"):
        module = importlib.import_module(module_path)
        if hasattr(module, "serviced_scopes"):
            monkeypatch.setattr(module, "serviced_scopes", _one)

    # And treat every scope as already loaded. The loops, the poller and the
    # realtime feed all call `ensure_loaded` before reading an allowlist — they
    # have to, because a user who registers credentials mid-session is serviced
    # without lifespan ever having loaded them. In a unit test that call reaches
    # the dev Postgres and REPLACES the tickers the test just set with whatever
    # is stored there, so a test asserting on its own fixture data sees 20 real
    # tickers instead of its 5 fake ones.
    # 테스트가 직접 설정한 종목이 실제 DB 내용으로 덮어써지는 것을 막음.
    from app.web import trade_allowlist_store

    async def _already_loaded(user_id, account_mode):
        return None

    REAL["ensure_loaded"] = trade_allowlist_store.ensure_loaded
    monkeypatch.setattr(trade_allowlist_store, "ensure_loaded", _already_loaded)


@pytest.fixture(autouse=True)
def _isolate_scoped_ticker_lists():
    """Clear the per-(user, mode) ticker lists between every test.

    The four allowlist/watchlist lists stopped being pydantic fields on the
    Settings singleton (2026-08-06) and became properties over a module-level
    per-scope dict. That dict is process state: without this, a test that
    writes `settings.trade_allowlist_domestic` leaves those tickers visible to
    every later test sharing the same scope.

    That was already true of the old global fields, so this is not a new
    hazard — but it IS newly fixable in one place, and the handoff notes ~40
    phantom failures traced to shared state. Autouse so nobody has to remember.
    예전 전역 필드도 같은 문제가 있었지만, 이제 한 곳에서 정리할 수 있음.
    """
    from app.config import reset_scoped_lists

    from app.web.trade_allowlist_store import forget_loaded_scopes

    reset_scoped_lists()
    forget_loaded_scopes()
    yield
    reset_scoped_lists()
    forget_loaded_scopes()


@pytest.fixture(autouse=True)
def _clear_account_read_snapshot_cache():
    """portfolio_risk.cached_account_snapshot's process-wide TTL cache
    (2026-09-01) — clear between tests so one test's stubbed balance can't
    be served to the next."""
    from app.execution.portfolio_risk import invalidate_account_read_cache

    invalidate_account_read_cache()
    yield
    invalidate_account_read_cache()


def broker_stub(broker):
    """An async stand-in for `app.web.state.kis_broker` returning `broker`.

    Pass None to model "this user has no KIS credentials configured" — the
    state every consumer is supposed to degrade quietly on.
    """
    async def kis_broker():
        return broker

    return kis_broker


def bypass_kis_gate(app):
    """Satisfy the KIS-configured gate on a test app.

    Routes that need a live KIS account carry `require_kis`
    (app.web.account_mode_middleware, 2026-08-07 single-account rollback —
    replaces the old app.auth.dependencies.require_credentials), which checks
    `settings.configured_kis_modes()` and starts the real-time feed. That is
    correct in production and wrong in a router unit test: those tests assert
    what a handler returns, so without this they would additionally be
    asserting that this deployment's env happens to have KIS credentials set
    and that Postgres is reachable (ensure_market_feed), neither of which
    says anything about the handler under test.
    실제로 KIS 설정/DB가 있어야 통과하는 게이트를 라우터 단위 테스트에서는 우회 —
    안 그러면 핸들러 로직과 무관한 이유로 실패함."""
    from app.web.account_mode_middleware import require_kis

    app.dependency_overrides[require_kis] = lambda: None
    return app
