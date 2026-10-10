"""Vendor router: tool signatures stay decoupled from the data provider.

Each category is a configured chain of providers tried in order. "Core"
categories (price, fundamentals) fail loudly when every provider in the
chain fails — an analyst should know its data is missing, not silently
reason over nothing. "Optional" categories (e.g. macro) degrade quietly:
the category returns a placeholder string instead of raising, so one flaky
or unconfigured source doesn't take down an analyst's whole report.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

Provider = Callable[..., Awaitable[str]]


@dataclass
class Category:
    name: str
    providers: list[Provider]
    optional: bool


class VendorRouter:
    def __init__(self) -> None:
        self._categories: dict[str, Category] = {}

    def register(self, category: str, providers: list[Provider], *, optional: bool = False) -> None:
        self._categories[category] = Category(category, providers, optional)

    async def call(self, category: str, /, **kwargs: Any) -> str:
        cat = self._categories[category]
        last_error: Exception | None = None
        for provider in cat.providers:
            try:
                return await provider(**kwargs)
            except Exception as exc:  # noqa: BLE001 - fallback chain must catch broadly
                last_error = exc
                logger.warning("vendor %s failed for category %r: %s", provider.__name__, category, exc)
        if cat.optional:
            logger.warning("category %r exhausted all vendors; degrading quietly", category)
            return f"[{category} data unavailable: {last_error}]"
        raise RuntimeError(f"category {category!r} exhausted all configured vendors") from last_error


router = VendorRouter()
