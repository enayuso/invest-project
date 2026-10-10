"""FastAPI dashboard + chat interface — the live layer's user-facing side.

Serves a single static page (live trade-allowlist chart, alerts panel, chat box)
and a small JSON/SSE API. Starts background tasks in its own lifespan — the
minute poller (app.live.poller), a lower-frequency stop-loss check loop
(app.execution.ledger.check_stops_for_ticker, one pass over the whole
trade allowlist), and (when KIS credentials + KRX trade-allowlist codes are present)
the real-time market-operation/trade/orderbook/fill feed
(app.brokers.kis_ws.RealtimeFeed) — this app *is* the process's one
long-running job now, not something bolted onto a separate loop; there's no
other place left for a background task to live now that app.daemon is just
a thin uvicorn launcher.

Was single-process/single-user/localhost/no-auth by design for v1. The
multi-user migration (docs/multi-user-credentials-plan.md) briefly added an
email/password session layer (Phase 1-7) — removed 2026-08-07 when this
became a single-account in-house deployment
(docs/single-user-rollback-guide.md): the company trades exactly one
KIS account, so there is no one left to log in AS. Access control is entirely
`dashboard_host` staying loopback + an SSH tunnel (see
app.config._refuse_non_loopback_bind); there is no application-level
authentication in front of any route below, including POST /api/order.

Shared mutable state and its lifespan live in app.web.state; the ~35 route
handlers live in feature-grouped routers under app.web.routers (see
AI_development_principles.md #16 — group by feature, not one file per
endpoint). This module is wiring only.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.web import state
from app.web.account_mode_middleware import bind_account_mode
from app.web.origin_guard import enforce_loopback_origin
from app.web.routers import (
    audit,
    chat_decisions,
    credentials,
    dashboard_summary,
    kr_insights,
    live,
    market_data,
    market_status,
    metrics,
    orders,
    portfolio,
    trade_allowlist,
)

_STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="invest-pipeline live dashboard", lifespan=state.capture_finalizing_lifespan)


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(_STATIC_DIR / "index.html")


# these mounts serve app/web/static/css and app/web/static/js at those paths.
app.mount("/css", StaticFiles(directory=_STATIC_DIR / "css"), name="css")
app.mount("/js", StaticFiles(directory=_STATIC_DIR / "js"), name="js")


@app.middleware("http")
async def _no_cache_for_frontend_assets(request, call_next):
    """Neither StaticFiles nor FileResponse (the `/` index route above) set
    any Cache-Control header on their own, so a browser is free to reuse a
    heuristically-cached copy of index.html/js/css indefinitely with no
    revalidation — confirmed live (2026-07-30): a browser kept serving a
    stale signals.js (missing a just-added export) even across a hard
    reload, silently breaking the page with no console error at all,
    because the browser never even asked the server for a fresh copy. This
    is a single-user local dev dashboard, not a CDN-fronted public site, so
    there is no caching benefit worth trading for that failure mode —
    force revalidation on every load instead.
    브라우저가 index.html/js/css를 캐시로만 재사용하고 서버에 재확인을 아예 안 하는
    문제(2026-07-30 실측) — 매번 새로 받아오게 강제."""
    response = await call_next(request)
    if request.url.path == "/" or request.url.path.startswith(("/css/", "/js/")):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return response


# Binds the active account mode (paper/real) for the duration of the request.
# Registered AFTER the cache-header middleware above, which means it runs
# BEFORE it on the way in (Starlette applies http middleware in reverse
# registration order) — mode must be bound before any handler runs, and the
# cache-header middleware only touches the response on the way out, so
# neither ordering constraint conflicts. There is no identity-binding
# middleware any more (no login — see this module's own docstring);
# app.user_context.get_current_user_id() falls back to settings.user_id on
# every request, which for a single-account deployment is simply correct.
# 미들웨어는 등록 역순으로 실행되므로, 이 모드 바인딩이 핸들러보다 먼저 동작함.
app.middleware("http")(bind_account_mode)


# Registered LAST so it runs FIRST on the way in (reverse registration order,
# same rule the bind_account_mode comment above spells out) — a request from
# an untrusted Host/Origin must be rejected before it can resolve an account
# mode, touch Postgres, or reach any handler. See app.web.origin_guard's own
# module docstring for what this is defending against; short version: a
# loopback bind alone does not stop a browser that someone else's page told
# to POST here.
# 등록 역순 실행이므로 마지막 등록 = 가장 먼저 실행 — 신뢰할 수 없는 요청은
# 계좌모드 조회/DB 접근 이전에 차단되어야 함.
app.middleware("http")(enforce_loopback_origin)


# No auth dependency on any router: there is no session to require (see this
# module's own docstring). Routes that genuinely need a live KIS account use
# their own per-route `Depends(require_kis)` (orders.py, portfolio.py) — a
# 412 for "not configured," never a 401, because there is nobody left to log
# in as.
for _router in (
    credentials.router,
    trade_allowlist.router,
    orders.router,
    market_status.router,
    market_data.router,
    kr_insights.router,
    live.router,
    chat_decisions.router,
    portfolio.router,
    metrics.router,
    dashboard_summary.router,
    audit.router,
):
    app.include_router(_router)
