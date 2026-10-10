# app/domain/ports/llm_port.py
# LLM call contract — a Protocol, not a base class, so an existing client
# class (e.g. app.llm.client.LLMClient) satisfies it by matching method
# shapes, with no import-time coupling back to domain/. Same structural-
# typing approach as app.domain.ports.broker_port.BrokerPort.
# LLM 호출 계약 (구조적 타이핑 — 상속 없이 시그니처만 맞으면 충족, BrokerPort와 동일 방식)

from __future__ import annotations

from typing import Any, Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel

from app.domain.models.llm import LLMMessage

T = TypeVar("T", bound=BaseModel)

# Literal "en" instead of importing app.llm.language.DEFAULT_LANGUAGE — domain
# must not import the app.llm adapter package. If that constant ever changes
# from "en", this literal needs to be updated too; an acceptable tradeoff for
# keeping the dependency direction correct.


@runtime_checkable
class LLMPort(Protocol):
    """Chat completion (freeform + tool-calling) and schema-constrained
    structured completion — the two operations every agent module actually
    calls. Deliberately does NOT standardize how a given vendor/model
    translates `reasoning: bool` internally (see
    app.llm.client.LLMClient._reasoning_extra_body) — that's an adapter-
    internal concern, not part of this contract."""

    async def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        reasoning: bool = False,
        temperature: float = 0.0,
        label: str = "",
        language: str = "en",
    ) -> LLMMessage: ...

    async def structured_complete(
        self,
        messages: list[dict[str, Any]],
        schema: type[T],
        *,
        reasoning: bool = False,
        temperature: float = 0.0,
        label: str = "",
        language: str = "en",
    ) -> T: ...
