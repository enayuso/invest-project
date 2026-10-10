"""Importing this package registers every provider into the shared router."""

from app.dataflows import (  # noqa: F401 - import side effect registers providers
    fred_provider,
    reddit_provider,
    yfinance,
)
from app.dataflows import kis_provider  # noqa: F401 - AFTER yfinance: re-registers price categories KIS-first
from app.dataflows.router import router

__all__ = ["router"]
