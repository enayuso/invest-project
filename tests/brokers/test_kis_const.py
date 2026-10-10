"""app.brokers.kis.const — classification helpers, no network."""

from __future__ import annotations

import pytest

from app.brokers.kis.const import (
    PERMANENT_MSG_CDS,
    KISBrokerError,
    is_permanent_refusal,
    is_transient_token_failure,
)


@pytest.mark.parametrize(
    "exc,expected",
    [
        (KISBrokerError("x", status_code=500), True),
        (KISBrokerError("x", status_code=502), True),
        (KISBrokerError("x", status_code=429), True),
        (KISBrokerError("x", status_code=200), True),  # 200 with no usable body
        (KISBrokerError("x", msg_cd="EGW00133"), True),  # issuance rate limit
        (KISBrokerError("x", msg_cd="EGW00201"), True),  # call rate limit
        (KISBrokerError("x", msg_cd="EGW00121", status_code=403), False),  # actually-bad app key
        (KISBrokerError("x"), False),  # no status_code, no msg_cd — legacy shape
        (KISBrokerError("x", status_code=403), False),  # 4xx, not enumerated
        (ValueError("x"), False),  # not even a KISBrokerError
    ],
)
def test_is_transient_token_failure(exc, expected):
    assert is_transient_token_failure(exc) is expected


def test_permanent_refusal_classification_unaffected():
    """is_permanent_refusal (order-payload semantics) must stay independent
    of the new token-issuance classification — different meaning, different
    code set, must never be conflated."""
    for code in PERMANENT_MSG_CDS:
        exc = KISBrokerError("x", msg_cd=code)
        assert is_permanent_refusal(exc) is True
        # A permanently-malformed order code says nothing about token
        # issuance — must not accidentally be treated as transient.
        assert is_transient_token_failure(exc) is False

    assert is_permanent_refusal(KISBrokerError("x", msg_cd="EGW00133")) is False


# ---------------------------------------------------------------------------
# is_rate_limited (2026-09-01) — body-substring detection of KIS's two
# distinct per-second caps. Independent of the permanent/token code sets.
# ---------------------------------------------------------------------------

from app.brokers.kis.const import RATE_LIMIT_MSG_CDS, TOKEN_ISSUE_TRANSIENT_MSG_CDS, is_rate_limited


def test_is_rate_limited_detects_both_codes_in_a_body():
    assert is_rate_limited('{"rt_cd":"1","msg_cd":"EGW00201"}') == "EGW00201"
    assert is_rate_limited('HTTP 500 원장에서 ... EGW00215 ...') == "EGW00215"
    assert is_rate_limited('{"rt_cd":"0"}') is None
    assert is_rate_limited("") is None


def test_rate_limit_set_is_independent_of_permanent_and_token_sets():
    # A rate-limit code is transient, an order-payload code is permanent —
    # conflating them would corrupt the resubmission-suppression logic.
    assert RATE_LIMIT_MSG_CDS & PERMANENT_MSG_CDS == frozenset()
    # EGW00201 is deliberately in BOTH the rate-limit set and the token
    # transient set (a token issuance can also be rate-limited); EGW00215 is
    # rate-limit only.
    assert "EGW00215" in RATE_LIMIT_MSG_CDS
    assert "EGW00215" not in TOKEN_ISSUE_TRANSIENT_MSG_CDS
