"""Host + Origin guard — the request-level half of this deployment's access
control, alongside `app.config._refuse_non_loopback_bind`'s bind-level half.

There is no application-level authentication in front of any route (2026-08-07
single-account rollback, see docs/single-user-rollback-guide.md), so
"the socket is only reachable from loopback" was carrying the entire access
model. It does not actually carry all of it, because a loopback bind still
accepts requests a *browser* was tricked into sending from someone else's page:

1. **CSRF.** A cross-origin `POST` with no custom headers and no request body
   is a Fetch "simple request" — no preflight, so the absence of any CORS
   configuration never gets a chance to block it. The browser refuses to let
   the attacker READ the response, but the side effect already happened. Every
   body-less mutating route was reachable this way, including
   `POST /api/pending-orders/{id}/approve`, which submits a real KIS order.
   (Routes taking a JSON body were incidentally protected — `application/json`
   forces a preflight that fails without CORS headers — which made the
   exposure inconsistent rather than deliberate.)

2. **DNS rebinding.** Without Host validation, an attacker domain that
   re-resolves to 127.0.0.1 becomes same-origin as far as the browser is
   concerned, which removes the read restriction in (1) as well: full read of
   portfolio/P&L plus full write, including `POST /api/order`. A loopback bind
   is no defense at all here — the packets genuinely arrive on loopback.

Both checks below are hostname-only, deliberately ignoring port and scheme:
remote access is an SSH tunnel, and `ssh -L 9000:127.0.0.1:8094` makes the
browser's own Host/Origin `localhost:9000`, not the server's `:8094`. Pinning
the port would break every tunnel that doesn't happen to mirror the port.
Ignoring it costs nothing — an attacker page cannot obtain a loopback origin
without already executing code on this machine.
인증 계층이 없는 배포라 '루프백 바인딩'만으로는 브라우저를 경유한 CSRF/DNS 리바인딩을
막지 못함 — Host/Origin을 호스트명 기준으로 검사(포트·스킴은 SSH 터널 때문에 무시).
"""

from __future__ import annotations

import logging
from urllib.parse import urlsplit

from fastapi import Request, status
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

# `urlsplit().hostname` lowercases and strips IPv6 brackets, so these are the
# already-normalized forms it returns.
_LOOPBACK_HOSTNAMES = frozenset({"127.0.0.1", "localhost", "::1"})

# GET/HEAD are deliberately absent: they are checked by the Host guard (which
# runs for every method) but not the Origin guard. A cross-origin GET cannot
# be read by the attacker without CORS headers this app never sends, and
# adding GET here would reject ordinary top-level navigation to the dashboard,
# which carries no Origin at all.
_STATE_CHANGING_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})


def _is_loopback(value: str, *, has_scheme: bool) -> bool:
    """True if `value`'s hostname component is this machine. `has_scheme`
    distinguishes an Origin (`http://localhost:8094`) from a Host header
    (`localhost:8094`, no scheme) — urlsplit needs the `//` prefix to treat
    the latter as an authority rather than a bare path."""
    try:
        hostname = urlsplit(value if has_scheme else f"//{value}").hostname
    except ValueError:
        # Malformed authority (e.g. an unclosed IPv6 bracket) — treat as
        # untrusted rather than letting a parse failure fall through.
        return False
    return hostname in _LOOPBACK_HOSTNAMES


def _deny(request: Request, header: str, value: str) -> JSONResponse:
    # Warning, not info: on a correctly-configured deployment this never
    # fires, so every occurrence is either a real attempt or a setup that
    # needs looking at. Logs the offending value so which of the two is
    # answerable from the log alone.
    logger.warning(
        "blocked %s %s — untrusted %s header %r (this deployment has no authentication; "
        "only loopback Host/Origin are accepted, see app.web.origin_guard)",
        request.method, request.url.path, header, value,
    )
    return JSONResponse(
        {"detail": f"untrusted {header} header — this dashboard only accepts loopback requests"},
        status_code=status.HTTP_403_FORBIDDEN,
    )


async def enforce_loopback_origin(request: Request, call_next):
    """Rejects any request whose `Host` is not loopback (every method), and
    any state-changing request whose `Origin` is present but not loopback.

    A MISSING Origin is allowed on purpose, and that is the crux of the CSRF
    half: per the Fetch standard a browser always sets `Origin` on a request
    whose method is not GET/HEAD — including the `no-cors` cross-origin POST
    that is the attack — so absence means the caller is not a browser (curl,
    the ops `curl http://127.0.0.1:8094/...` checks in the deploy runbook, a
    script). Requiring it instead would break those without blocking anything
    a browser can actually do.
    Origin이 아예 없는 경우는 허용 — 브라우저는 GET/HEAD가 아닌 요청에 Origin을
    항상 붙이므로, 없다는 건 브라우저가 아니라는 뜻(curl/스크립트).

    Deliberately not a Referer fallback (the usual Django-style second
    check): Referer is suppressible by referrer-policy, while Origin is not,
    and Origin alone already covers every browser-originated case.
    """
    host = request.headers.get("host", "")
    if not _is_loopback(host, has_scheme=False):
        return _deny(request, "Host", host)

    if request.method in _STATE_CHANGING_METHODS:
        origin = request.headers.get("origin")
        # "null" (sandboxed iframe, file:// page) parses to hostname None and
        # is rejected by _is_loopback — that is intended, it is not a
        # trustworthy origin.
        if origin is not None and not _is_loopback(origin, has_scheme=True):
            return _deny(request, "Origin", origin)

    return await call_next(request)
