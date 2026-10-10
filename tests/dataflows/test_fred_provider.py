"""FRED macro provider — httpx mocked via monkeypatch, no real network (B-stage)."""

from __future__ import annotations

import httpx
import pytest

from app.config import settings
from app.dataflows import fred_provider


class _FakeResponse:
    def __init__(self, payload: dict, status: int = 200):
        self._payload = payload
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            request = httpx.Request("GET", "https://api.stlouisfed.org/fred/series/observations")
            raise httpx.HTTPStatusError("error", request=request, response=httpx.Response(self.status_code, request=request))

    def json(self):
        return self._payload


async def test_raises_when_api_key_not_configured(monkeypatch):
    monkeypatch.setattr(settings, "fred_api_key", "")
    with pytest.raises(RuntimeError, match="FRED_API_KEY not configured"):
        await fred_provider._fred_macro_indicators()


async def test_returns_formatted_indicators_on_success(monkeypatch):
    monkeypatch.setattr(settings, "fred_api_key", "test-key")

    async def fake_get(self, url, params=None, **kwargs):
        series_id = params["series_id"]
        return _FakeResponse({"observations": [{"value": "4.5", "date": "2026-07-01"}]})

    monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)
    result = await fred_provider._fred_macro_indicators()
    assert "Macro indicators (FRED):" in result
    assert "Fed Funds Rate: 4.5 percent per annum (as of 2026-07-01)" in result


async def test_requests_gdpc1_not_bare_gdp(monkeypatch):
    # B-stage fix: `GDP` is the nominal series; real GDP is `GDPC1`. Assert
    # on the actual params sent so a future revert of the series ID is
    # caught by the test suite rather than silently reintroducing the bug.
    monkeypatch.setattr(settings, "fred_api_key", "test-key")
    requested_series_ids = []

    async def fake_get(self, url, params=None, **kwargs):
        requested_series_ids.append(params["series_id"])
        return _FakeResponse({"observations": [{"value": "1.0", "date": "2026-07-01"}]})

    monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)
    await fred_provider._fred_macro_indicators()
    assert "GDPC1" in requested_series_ids
    assert "GDP" not in requested_series_ids


async def test_every_rendered_line_carries_a_unit(monkeypatch):
    monkeypatch.setattr(settings, "fred_api_key", "test-key")

    async def fake_get(self, url, params=None, **kwargs):
        return _FakeResponse({"observations": [{"value": "100.0", "date": "2026-07-01"}]})

    monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)
    result = await fred_provider._fred_macro_indicators()

    assert "percent per annum" in result
    assert "percent" in result
    assert "billions of chained dollars" in result
    # The CPI line must make explicit that it's an index level, not a rate.
    assert "index, 1982-84=100" in result
    assert "not an inflation rate" in result


async def test_http_error_is_sanitized_and_does_not_leak_api_key(monkeypatch):
    monkeypatch.setattr(settings, "fred_api_key", "super-secret-key")

    async def fake_get(self, url, params=None, **kwargs):
        return _FakeResponse({}, status=500)

    monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)
    with pytest.raises(RuntimeError) as exc_info:
        await fred_provider._fred_macro_indicators()
    assert "super-secret-key" not in str(exc_info.value)


async def test_raises_when_all_series_return_no_observations(monkeypatch):
    monkeypatch.setattr(settings, "fred_api_key", "test-key")

    async def fake_get(self, url, params=None, **kwargs):
        return _FakeResponse({"observations": []})

    monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)
    with pytest.raises(RuntimeError, match="FRED returned no observations"):
        await fred_provider._fred_macro_indicators()
