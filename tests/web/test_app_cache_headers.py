"""app.web.app's no-cache middleware — verifies the actual registered
middleware on the real app object, without running its lifespan (which
opens broker/DB connections) by overriding it with a no-op for the test
client's startup/shutdown cycle."""

from __future__ import annotations

from contextlib import asynccontextmanager

import pytest
from fastapi.testclient import TestClient

from app.web.app import app


@asynccontextmanager
async def _noop_lifespan(_app):
    yield


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(app.router, "lifespan_context", _noop_lifespan)
    with TestClient(app, base_url="http://127.0.0.1:8094") as c:
        yield c


def test_index_route_gets_no_cache_header(client):
    resp = client.get("/")
    assert resp.headers["cache-control"] == "no-cache, no-store, must-revalidate"


def test_css_asset_gets_no_cache_header(client):
    resp = client.get("/css/base.css")
    assert resp.headers["cache-control"] == "no-cache, no-store, must-revalidate"


def test_js_asset_gets_no_cache_header(client):
    resp = client.get("/js/main.js")
    assert resp.headers["cache-control"] == "no-cache, no-store, must-revalidate"


def test_api_route_is_not_forced_no_cache(client):
    # A 404 (this path isn't a registered route) is fine here — the point
    # is only that this middleware doesn't stamp
    # every response with the header, just the frontend-asset paths.
    resp = client.get("/api/does-not-exist")
    assert "cache-control" not in resp.headers or resp.headers["cache-control"] != "no-cache, no-store, must-revalidate"
