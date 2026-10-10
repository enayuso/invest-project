"""Structured output for the chat intent router — kept typed end to end,
same convention as app.state, not free text parsed back with regex.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class ChatIntent(str, Enum):
    QUICK_FACTUAL = "quick_factual"
    NEWS_QUERY = "news_query"
    SCREENING = "screening"
    DEEP_DIVE = "deep_dive"
    GENERAL = "general"


class ChatClassification(BaseModel):
    intent: ChatIntent = Field(
        description=(
            "quick_factual: a simple current-state question about specific ticker(s) "
            "(price, volume, basic stats). news_query: asking for recent news about "
            "specific ticker(s). screening: a comparative/ranking question about which "
            "stocks (from the trade allowlist) look good or bad right now. deep_dive: "
            "explicitly asking for a full investment recommendation/analysis on one "
            "specific ticker. general: anything else — greetings, small talk, questions "
            "about this assistant/dashboard itself, or any question unrelated to stocks "
            "or this trade allowlist. Default to general whenever the question isn't clearly "
            "about a stock or the trade allowlist."
        ),
    )
    tickers: list[str] = Field(
        default_factory=list,
        description=(
            "Ticker symbols mentioned or clearly implied by the question, uppercase. "
            "Leave empty if the question refers to the trade allowlist generally rather than "
            "specific tickers."
        ),
    )
