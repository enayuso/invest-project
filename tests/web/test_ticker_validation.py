"""validate_ticker — pure regex guard (E-stage)."""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from app.web.ticker_validation import validate_ticker


def test_accepts_plain_us_ticker():
    assert validate_ticker("aapl") == "AAPL"


def test_accepts_krx_suffixed_ticker():
    assert validate_ticker("005930.ks") == "005930.KS"


def test_accepts_index_caret_and_fx_equals_symbols():
    assert validate_ticker("^gspc") == "^GSPC"
    assert validate_ticker("krw=x") == "KRW=X"


def test_strips_surrounding_whitespace():
    assert validate_ticker("  aapl  ") == "AAPL"


def test_rejects_disallowed_characters():
    with pytest.raises(HTTPException) as exc_info:
        validate_ticker("AAPL; DROP TABLE")
    assert exc_info.value.status_code == 422


def test_rejects_overly_long_input():
    with pytest.raises(HTTPException):
        validate_ticker("A" * 21)
