# app/domain/ports/__init__.py
# Vendor-agnostic contracts. Any new broker adapter (KIS, and future ones)
# or LLM adapter (Qwen3 via vLLM, and future ones) is expected to satisfy
# these Protocols.
# 신규 증권사/LLM 추가시 구현해야 하는 계약(포트) 모음

from app.domain.ports.broker_port import BrokerPort
from app.domain.ports.llm_port import LLMPort
from app.domain.ports.market_data_port import MarketDataPort

__all__ = ["BrokerPort", "LLMPort", "MarketDataPort"]
