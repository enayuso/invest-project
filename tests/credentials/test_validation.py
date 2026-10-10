"""app.credentials.validation — httpx.MockTransport, no network.

Covers the 2026-08-26 fix: token-issuance failures used to all collapse into
CredentialsRejected regardless of cause (KIS outage, rate limit, malformed
body) — see app/brokers/kis/const.py's is_transient_token_failure and this
module's verify_kis_credentials for the classification this locks in.
"""

from __future__ import annotations

import logging

import httpx
import pytest

from app.credentials import validation
from app.credentials.validation import CredentialCheckUnavailable, CredentialsRejected, verify_kis_credentials

APP_KEY = "test-app-key-12345678"
TOKEN_BODY = {
    "access_token": "test-token-xyz",
    "access_token_token_expired": "2099-01-01 00:00:00",
    "token_type": "Bearer",
    "expires_in": 86400,
}


@pytest.fixture(autouse=True)
def _isolated_token_cache(tmp_path, monkeypatch):
    # token_cache_path_for(app_key) writes to a real path under the repo
    # root by default — every test here must not touch that shared file.
    monkeypatch.setattr(validation, "token_cache_path_for", lambda key: tmp_path / f".cache.{key}.json")


async def _check(handler) -> None:
    await verify_kis_credentials(APP_KEY, "test-secret", "paper", transport=httpx.MockTransport(handler))


async def test_bad_credentials_rejected():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, json={"error_code": "EGW00121", "error_description": "유효하지 않은 AppKey 입니다."})

    with pytest.raises(CredentialsRejected, match="EGW00121"):
        await _check(handler)


async def test_server_error_is_unavailable_not_rejected():
    """Regression: KIS 5xx used to be misreported as bad credentials."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(502, text="Bad Gateway")

    with pytest.raises(CredentialCheckUnavailable):
        await _check(handler)


async def test_issuance_rate_limit_is_unavailable_not_rejected():
    """Regression: EGW00133 (one issuance/min/app-key) used to be reported
    as bad credentials, sending operators to re-check Jenkins for nothing."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, json={"error_code": "EGW00133", "error_description": "발급 요청이 제한되었습니다."})

    with pytest.raises(CredentialCheckUnavailable):
        await _check(handler)


async def test_maintenance_html_page_is_unavailable_not_a_crash():
    """Regression: a 200 with a non-JSON body used to raise an unhandled
    JSONDecodeError, which — unguarded in state.py's lifespan — could take
    down server boot entirely."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>점검 중입니다</html>", headers={"content-type": "text/html"})

    with pytest.raises(CredentialCheckUnavailable):
        await _check(handler)


async def test_non_dict_json_body_is_unavailable_not_a_crash():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=["unexpected", "shape"])

    with pytest.raises(CredentialCheckUnavailable):
        await _check(handler)


async def test_200_without_access_token_is_unavailable():
    """Deliberate behavior change: previously classified as CredentialsRejected
    even though HTTP succeeded — there is no credential-related signal in a
    bare 200 with no access_token, so it can't mean "these keys are wrong"."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"msg1": "empty"})

    with pytest.raises(CredentialCheckUnavailable):
        await _check(handler)


async def test_connect_error_is_unavailable():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    with pytest.raises(CredentialCheckUnavailable):
        await _check(handler)


async def test_unexpected_exception_does_not_propagate(caplog):
    """FINDING 2 regression: any exception KISAuth might raise beyond
    KISBrokerError/httpx.HTTPError must still resolve to
    CredentialCheckUnavailable, never propagate — the caller
    (app.web.state.lifespan) has no guard against a third exception type."""

    class _BoomAuth:
        async def _get_token(self):
            raise ValueError("something KISAuth was never expected to raise")

    def make_boom_auth(**kwargs):
        return _BoomAuth()

    import app.credentials.validation as validation_module

    original = validation_module.KISAuth
    validation_module.KISAuth = make_boom_auth
    try:
        with caplog.at_level(logging.ERROR):
            with pytest.raises(CredentialCheckUnavailable):
                await verify_kis_credentials(
                    APP_KEY, "test-secret", "paper", transport=httpx.MockTransport(lambda r: httpx.Response(200))
                )
        assert any(r.levelno >= logging.ERROR for r in caplog.records)
    finally:
        validation_module.KISAuth = original


async def test_success_warms_cache_and_reuses_it_on_second_call(tmp_path, monkeypatch):
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        return httpx.Response(200, json=TOKEN_BODY)

    cache_path = tmp_path / "token.json"
    monkeypatch.setattr(validation, "token_cache_path_for", lambda key: cache_path)

    await verify_kis_credentials(APP_KEY, "test-secret", "paper", transport=httpx.MockTransport(handler))
    assert cache_path.exists()
    assert len(calls) == 1

    # Re-checking the same credentials within the token's lifetime must not
    # re-issue — that's the whole reason the token is written to the shared
    # cache path instead of discarded (module docstring).
    await verify_kis_credentials(APP_KEY, "test-secret", "paper", transport=httpx.MockTransport(handler))
    assert len(calls) == 1


async def test_client_closed_even_when_construction_fails(monkeypatch):
    """FINDING 3 regression: the httpx.AsyncClient must close even if
    building KISAuth (or token_cache_path_for) raises before the old
    try/finally around auth._get_token() would ever have started.

    httpx.AsyncClient.__aexit__ closes the transport directly rather than
    going through .aclose(), so the client instance itself (not a spied
    method) is what must be inspected — captured here via __aenter__."""
    created: list[httpx.AsyncClient] = []
    real_aenter = httpx.AsyncClient.__aenter__

    async def spy_aenter(self):
        created.append(self)
        return await real_aenter(self)

    monkeypatch.setattr(httpx.AsyncClient, "__aenter__", spy_aenter)

    def boom(key: str):
        raise OSError("disk unavailable")

    monkeypatch.setattr(validation, "token_cache_path_for", boom)

    with pytest.raises(OSError):
        await verify_kis_credentials(
            APP_KEY, "test-secret", "paper", transport=httpx.MockTransport(lambda r: httpx.Response(200))
        )
    assert len(created) == 1
    assert created[0].is_closed
