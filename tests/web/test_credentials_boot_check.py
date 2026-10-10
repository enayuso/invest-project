"""app.web.state._verify_configured_kis_credentials — locks in the "logs and
continues, never raises" boot contract (2026-08-26). This function itself
wasn't changed by the credentials-validation fix, but its safety now
depends entirely on app.credentials.validation.verify_kis_credentials never
raising anything outside {CredentialsRejected, CredentialCheckUnavailable} —
this test guards that contract from the caller's side."""

from __future__ import annotations

import logging

import pytest

from app.config import Settings, settings
from app.credentials.validation import CredentialCheckUnavailable, CredentialsRejected
from app.domain.models import KISCredentials
from app.web import state


@pytest.fixture(autouse=True)
def _one_mode_configured(monkeypatch):
    # Settings is a pydantic BaseSettings model — instance-level setattr of a
    # non-field name raises, so these are patched at the class level instead.
    monkeypatch.setattr(Settings, "configured_kis_modes", lambda self: {"paper"})
    monkeypatch.setattr(
        Settings,
        "kis_credentials",
        lambda self, mode: KISCredentials(
            user_id="default", account_mode=mode, app_key="k", app_secret="s",
            account_8="12345678", account_prod="01", hts_id=None,
        ),
    )


async def test_rejected_credentials_logs_error_and_does_not_raise(monkeypatch, caplog):
    async def boom(*a, **k):
        raise CredentialsRejected("bad key")

    monkeypatch.setattr(state, "verify_kis_credentials", boom)
    with caplog.at_level(logging.ERROR):
        await state._verify_configured_kis_credentials()  # must not raise
    assert any(r.levelno == logging.ERROR for r in caplog.records)


async def test_unavailable_credentials_logs_warning_and_does_not_raise(monkeypatch, caplog):
    async def boom(*a, **k):
        raise CredentialCheckUnavailable("KIS unreachable")

    monkeypatch.setattr(state, "verify_kis_credentials", boom)
    with caplog.at_level(logging.WARNING):
        await state._verify_configured_kis_credentials()  # must not raise
    assert any(r.levelno == logging.WARNING for r in caplog.records)


async def test_success_logs_info_and_does_not_raise(monkeypatch, caplog):
    async def ok(*a, **k):
        return None

    monkeypatch.setattr(state, "verify_kis_credentials", ok)
    with caplog.at_level(logging.INFO):
        await state._verify_configured_kis_credentials()
    assert any("verified at boot" in r.message for r in caplog.records)
