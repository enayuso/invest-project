"""Account mode: policy (app.account_mode) and binding (app.web
.account_mode_middleware).

Policy as of the 2026-08-07 single-account rollback
(docs/single-user-rollback-guide.md): one deployment, one company
account, paper and real both usable — the active mode is this deployment's
own stored preference (`account_mode.stored_preference`/`set_stored_preference`,
backed by the `users` table's single row), not a per-user choice any more.
"""

from __future__ import annotations

import asyncio

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from app import account_mode
from app.config import Settings, account_mode_scope, current_account_mode, settings


def _fake_configured_kis_modes(monkeypatch, modes: set[str]) -> None:
    """`configured_kis_modes` is a plain method on `Settings`, not a pydantic
    field — `monkeypatch.setattr(settings, ...)` raises (pydantic rejects
    setting an attribute that isn't a declared field on an instance), so this
    patches the class instead, which every instance (including the module
    singleton) picks up through normal method resolution."""
    monkeypatch.setattr(Settings, "configured_kis_modes", lambda self: modes)


# ---------------------------------------------------------------- policy

def test_everyone_may_use_both_modes():
    assert set(account_mode.allowed_modes_for()) == {"paper", "real"}


def test_stated_preference_wins_when_backed_by_credentials():
    assert account_mode.resolve_preference("real", {"paper", "real"}) == "real"
    assert account_mode.resolve_preference("paper", {"paper", "real"}) == "paper"


def test_preference_ignored_when_it_has_no_credentials():
    """A stored preference for paper survives deleting those credentials from
    the env without keeping this deployment scoped to a mode whose broker can
    no longer be built."""
    assert account_mode.resolve_preference("paper", {"real"}) == "real"
    assert account_mode.resolve_preference("real", {"paper"}) == "paper"


def test_paper_wins_when_both_configured_and_nothing_chosen():
    """The safer of two available modes engages when nobody asked."""
    assert account_mode.resolve_preference(None, {"paper", "real"}) == "paper"


def test_unonboarded_resolves_to_real_the_required_mode():
    """Nothing is reachable in this state either way; what this decides is that
    the 412 says 'no real credentials', which is the actual next step."""
    assert account_mode.resolve_preference(None, set()) == "real"


def test_garbage_preference_is_ignored():
    assert account_mode.resolve_preference("margin", {"paper"}) == "paper"
    assert account_mode.resolve_preference("", {"real"}) == "real"


async def test_resolve_account_mode_reads_configured_kis_modes(monkeypatch):
    """Single-account deployment (2026-08-07 rollback): `configured_kis_modes()`
    is a sync, in-process check over env fields — resolve_account_mode cannot
    fail the way the old DB-backed `configured_modes(user_id)` could, so there
    is no failure path left to pin here, only that it reads the right source."""
    _fake_configured_kis_modes(monkeypatch, {"paper"})
    assert await account_mode.resolve_account_mode(settings.user_id, "real") == "paper"
    assert await account_mode.resolve_account_mode(settings.user_id, "paper") == "paper"


# ---------------------------------------------------------------- contextvar

def test_scope_binds_and_restores():
    before = current_account_mode()
    with account_mode_scope("real"):
        assert current_account_mode() == "real"
    assert current_account_mode() == before


def test_falls_back_to_global_outside_a_request():
    """CLI, backtest and the background loops keep the pre-migration behavior."""
    assert current_account_mode() == settings.kis_account_mode


async def test_concurrent_requests_do_not_share_a_mode():
    """The multi-tenancy guarantee, for mode rather than identity: two
    concurrently-served requests must never see each other's bound mode."""
    seen: dict[str, list[str]] = {}

    async def run_as(mode: str) -> None:
        with account_mode_scope(mode):
            out = []
            for _ in range(5):
                await asyncio.sleep(0)
                out.append(current_account_mode())
            seen[mode] = out

    await asyncio.gather(run_as("paper"), run_as("real"))
    assert seen["paper"] == ["paper"] * 5
    assert seen["real"] == ["real"] * 5


# ---------------------------------------------------------------- binding

def _app(monkeypatch, *, configured: set[str], preferred: str | None) -> FastAPI:
    from app.web import account_mode_middleware as mw

    async def fake_preference():
        return preferred

    _fake_configured_kis_modes(monkeypatch, configured)
    monkeypatch.setattr(account_mode, "stored_preference", fake_preference)
    monkeypatch.setattr(mw, "_cached_mode", None)

    app = FastAPI()
    app.middleware("http")(mw.bind_account_mode)

    @app.get("/api/mode")
    async def mode() -> dict:
        # Read inside the endpoint: this is what every persistence call site
        # actually uses to scope its rows.
        return {"mode": current_account_mode()}

    return app


@pytest.mark.parametrize(
    "preferred,configured,expected",
    [
        ("real", {"paper", "real"}, "real"),
        ("paper", {"paper", "real"}, "paper"),
        (None, {"paper", "real"}, "paper"),
        (None, {"real"}, "real"),
        ("paper", {"real"}, "real"),
    ],
)
def test_middleware_binds_the_resolved_mode(monkeypatch, preferred, configured, expected):
    client = TestClient(_app(monkeypatch, configured=configured, preferred=preferred), base_url="http://127.0.0.1:8094")
    assert client.get("/api/mode").json() == {"mode": expected}


def test_middleware_caches_the_resolved_mode_across_requests(monkeypatch):
    """Resolved once, then reused — not re-read from the stored preference on
    every request. `invalidate_cached_mode()` (called by `PUT
    /api/account-mode`) is the only thing that should force a re-resolve."""
    from app.web import account_mode_middleware as mw

    calls = 0

    async def counting_preference():
        nonlocal calls
        calls += 1
        return "real"

    _fake_configured_kis_modes(monkeypatch, {"paper", "real"})
    monkeypatch.setattr(account_mode, "stored_preference", counting_preference)
    monkeypatch.setattr(mw, "_cached_mode", None)

    app = FastAPI()
    app.middleware("http")(mw.bind_account_mode)

    @app.get("/api/mode")
    async def mode() -> dict:
        return {"mode": current_account_mode()}

    client = TestClient(app, base_url="http://127.0.0.1:8094")
    client.get("/api/mode")
    client.get("/api/mode")
    assert calls == 1

    mw.invalidate_cached_mode()
    client.get("/api/mode")
    assert calls == 2


async def test_a_mode_switch_during_an_in_flight_lookup_re_resolves_to_the_current_mode(monkeypatch):
    """2026-08-26: this cache has no TTL — invalidate_cached_mode() is the
    only thing that ever clears it. A request that started resolving BEFORE
    a switch commits reads stored_preference() at a snapshot that predates
    the commit (READ COMMITTED) and gets the OLD mode back; if that stale
    result then gets written into the cache, it stays pinned process-wide
    until the next switch or a restart — silently making "effective_from:
    next request" a lie for every subsequent request, not just the one that
    raced. Policy: the in-flight request re-resolves and binds the CURRENT
    mode rather than momentarily executing under the one just switched away
    from (see bind_account_mode's own comment on the generation counter)."""
    from app.web import account_mode_middleware as mw

    release = asyncio.Event()
    calls = 0

    async def gated_preference():
        nonlocal calls
        calls += 1
        if calls == 1:
            await release.wait()
            return "paper"  # the pre-switch snapshot this read started with
        return "real"

    _fake_configured_kis_modes(monkeypatch, {"paper", "real"})
    monkeypatch.setattr(account_mode, "stored_preference", gated_preference)
    monkeypatch.setattr(mw, "_cached_mode", None)
    monkeypatch.setattr(mw, "_cache_generation", 0)

    seen = {}

    async def call_next(request):
        seen["mode"] = current_account_mode()
        return "ok"

    task = asyncio.ensure_future(mw.bind_account_mode(None, call_next))
    await asyncio.sleep(0)  # let it start and block inside gated_preference's release.wait()

    # Simulate PUT /api/account-mode: the preference is already committed to
    # Postgres by this point (that's what makes the in-flight read's eventual
    # "paper" a stale snapshot, not a race on the write itself) — only the
    # invalidation is left to happen.
    mw.invalidate_cached_mode()
    release.set()

    await task
    assert seen["mode"] == "real"
    assert mw._cached_mode == "real"
    assert calls == 2  # one re-resolve, not zero (discarded) and not a retry storm


def test_account_mode_contextvar_propagates_into_a_streaming_response_body(monkeypatch):
    """Confirmed correct in a prior audit pass (F2) — pinned here so the
    scratch repro's finding doesn't silently regress. `set_current_account_mode`
    binds a contextvar token before `call_next` runs; a StreamingResponse's
    generator function runs as a continuation of that same request context
    (not a separate task), so it must still see the bound mode even though
    its body executes after the route handler itself has already returned."""
    from fastapi import FastAPI
    from fastapi.responses import StreamingResponse

    from app.web import account_mode_middleware as mw

    app = FastAPI()

    @app.get("/sse")
    async def sse():
        async def gen():
            await asyncio.sleep(0.01)
            yield f"mode={current_account_mode()}\n"

        return StreamingResponse(gen(), media_type="text/event-stream")

    app.middleware("http")(mw.bind_account_mode)

    async def fake_resolve(uid, pref):
        return "real"

    async def fake_pref():
        return "real"

    monkeypatch.setattr(account_mode, "resolve_account_mode", fake_resolve)
    monkeypatch.setattr(account_mode, "stored_preference", fake_pref)
    monkeypatch.setattr(mw, "_cached_mode", None)

    with TestClient(app, base_url="http://127.0.0.1:8094") as c:
        body = c.get("/sse").text
    assert "mode=real" in body


def test_invalidate_cached_mode_bumps_the_generation(monkeypatch):
    from app.web import account_mode_middleware as mw

    monkeypatch.setattr(mw, "_cache_generation", 0)
    mw.invalidate_cached_mode()
    first = mw._cache_generation
    mw.invalidate_cached_mode()
    second = mw._cache_generation

    assert first != 0
    assert second != first
