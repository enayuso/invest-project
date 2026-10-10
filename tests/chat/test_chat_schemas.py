"""ChatClassification/ChatIntent schema validation (A-stage, pure)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.chat.schemas import ChatClassification, ChatIntent


def test_defaults_tickers_to_empty_list_when_omitted():
    parsed = ChatClassification(intent="general")
    assert parsed.tickers == []


def test_accepts_valid_intent_and_tickers():
    parsed = ChatClassification(intent="deep_dive", tickers=["AAPL", "005930.KS"])
    assert parsed.intent is ChatIntent.DEEP_DIVE
    assert parsed.tickers == ["AAPL", "005930.KS"]


def test_rejects_unknown_intent_value():
    with pytest.raises(ValidationError):
        ChatClassification(intent="not_a_real_intent")


def test_missing_intent_raises():
    with pytest.raises(ValidationError):
        ChatClassification()
