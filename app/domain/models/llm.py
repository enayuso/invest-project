# app/domain/models/llm.py
# LLM response shape — vendor-agnostic, same "Domain doesn't know about the
# concrete adapter" split as account.py/market.py for brokers (see
# app.domain.ports.llm_port.LLMPort, app.brokers.__init__'s BrokerPort
# precedent). Lives here (not app/llm/client.py) so LLMPort can reference it
# without importing a concrete vendor adapter module.
# LLM 응답 형태 — 브로커의 account.py/market.py와 동일하게 구체 어댑터에서 분리.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class LLMMessage:
    content: str
    reasoning_content: str | None = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
