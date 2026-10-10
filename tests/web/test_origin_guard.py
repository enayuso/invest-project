"""app.web.origin_guard — the Host/Origin access check standing in for the
authentication this deployment does not have. Exercised against a tiny local
FastAPI app rather than app.web.app: the real app's lifespan starts pollers
and the KIS feed, and none of that says anything about the middleware."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.web.origin_guard import _is_loopback, enforce_loopback_origin

_LOOPBACK = "http://127.0.0.1:8094"


# --- _is_loopback exotic-form pinning table (2026-08-26 app/web audit) -----
#
# Re-verified during the audit: every exotic Host/Origin form either denies
# (safely) or, in the one accepting case, the parsed hostname genuinely IS
# loopback (a browser cannot forge Host/Origin, so these forms are only
# reachable from something that already has a loopback socket and gains
# nothing from a bypass). No behavior change — this pins the verdict so a
# future refactor of _is_loopback can't silently widen what it accepts.
@pytest.mark.parametrize("host,expected", [
    ("127.0.0.1:8094", True),
    ("localhost", True),
    ("[::1]:8094", True),
    ("LOCALHOST:8094", True),
    ("", False),
    ("evil.com", False),
    ("localhost.", False),
    ("127.0.0.1.", False),
    ("127.1", False),  # non-standard short IPv4 form — fails closed
    ("2130706433", False),  # decimal-integer IPv4 form — fails closed
    ("0.0.0.0", False),
    ("localhost\tevil.com", False),
    ("[::ffff:127.0.0.1]", False),  # IPv4-mapped IPv6 — fails closed
    ("127.0.0.1@evil.com", False),  # userinfo makes the HOSTNAME evil.com
    ("evil.com@127.0.0.1", True),  # userinfo is "evil.com", hostname genuinely IS 127.0.0.1
    ("127.0.0.1:evil", True),  # port is deliberately ignored (SSH tunnels remap it)
    ("localhost:99999999", True),  # same — malformed port, hostname still loopback
])
def test_is_loopback_host_form_table(host, expected):
    assert _is_loopback(host, has_scheme=False) is expected


def test_is_loopback_comma_joined_host_is_accepted_but_unexploitable():
    """The one row that looks odd: urlsplit takes the FIRST of a comma-joined
    Host value, so "localhost:8094, evil.com" parses to "localhost" and is
    accepted. Not exploitable: Host is a forbidden header name (page script
    cannot set it — WHATWG Fetch spec), so a browser only ever sends the
    hostname it actually navigated to; anything that CAN forge this header
    already has a loopback socket and gains nothing from a bypass."""
    assert _is_loopback("localhost:8094, evil.com", has_scheme=False) is True


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()

    @app.get("/read")
    async def read() -> dict:
        return {"ok": True}

    @app.post("/write")
    async def write() -> dict:
        return {"ok": True}

    app.middleware("http")(enforce_loopback_origin)
    return TestClient(app, base_url=_LOOPBACK)


# --- Host guard (every method) — DNS rebinding ------------------------------


def test_loopback_host_is_allowed(client):
    assert client.get("/read").status_code == 200


@pytest.mark.parametrize("host", ["localhost:8094", "127.0.0.1:8094", "127.0.0.1", "[::1]:8094"])
def test_every_loopback_spelling_is_allowed(client, host):
    # Port and IPv6 bracket form must not matter — an SSH tunnel picks its own
    # local port (see the module docstring).
    assert client.get("/read", headers={"Host": host}).status_code == 200


@pytest.mark.parametrize("host", ["evil.com", "attacker.test:8094", "192.168.0.5:8094", ""])
def test_non_loopback_host_is_rejected(client, host):
    # The DNS-rebinding case: packets really do arrive on loopback, so only
    # the Host header distinguishes this from a genuine local request.
    resp = client.get("/read", headers={"Host": host})
    assert resp.status_code == 403
    assert "Host" in resp.json()["detail"]


def test_host_guard_applies_to_reads_not_just_writes(client):
    # Rebinding is used to READ (portfolio, P&L), so a GET must be checked too.
    assert client.get("/read", headers={"Host": "evil.com"}).status_code == 403


# --- Origin guard (state-changing methods) — CSRF ---------------------------


def test_cross_origin_post_is_rejected(client):
    # The body-less "simple request" that no preflight ever blocked:
    # POST /api/pending-orders/{id}/approve submits a real order.
    resp = client.post("/write", headers={"Origin": "https://evil.com"})
    assert resp.status_code == 403
    assert "Origin" in resp.json()["detail"]


def test_same_origin_post_is_allowed(client):
    assert client.post("/write", headers={"Origin": _LOOPBACK}).status_code == 200


def test_post_from_a_different_loopback_port_is_allowed(client):
    # `ssh -L 9000:127.0.0.1:8094` makes the browser's own origin :9000.
    assert client.post("/write", headers={"Origin": "http://localhost:9000"}).status_code == 200


def test_post_without_an_origin_header_is_allowed(client):
    # curl/scripts (and the deploy runbook's own health checks) send none; a
    # browser always sends one on a non-GET/HEAD request, so absence is
    # proof this is not the attack. See the middleware's own docstring.
    assert client.post("/write").status_code == 200


def test_null_origin_is_rejected(client):
    # Sandboxed iframe / file:// page — parses to no hostname, not trusted.
    assert client.post("/write", headers={"Origin": "null"}).status_code == 403


def test_cross_origin_get_is_allowed(client):
    # Not a CSRF vector (no side effect) and rejecting it would break plain
    # navigation to the dashboard, which carries no Origin at all. The Host
    # guard above is what covers reads.
    assert client.get("/read", headers={"Origin": "https://evil.com"}).status_code == 200


@pytest.mark.parametrize("method", ["post", "put", "patch", "delete"])
def test_every_state_changing_method_is_guarded(client, method):
    app_client = client
    app = app_client.app

    @app.api_route("/mutate", methods=["POST", "PUT", "PATCH", "DELETE"])
    async def mutate() -> dict:
        return {"ok": True}

    resp = getattr(app_client, method)("/mutate", headers={"Origin": "https://evil.com"})
    assert resp.status_code == 403
