from app.config import settings
from app.domain.models.llm import LLMMessage
from app.llm.client import LLMClient


def _build_llm_client():
    """The single provider-selection point for the process-wide `llm_client`
    singleton, branching on `settings.llm_provider` (app.config):

      * "vllm" / "gateway" — app.llm.client.LLMClient, the existing
        OpenAI-compatible adapter. Exactly today's behavior, byte-for-byte —
        this branch is what every deployment still hits by default.
      * "anthropic" — app.llm.anthropic_client.AnthropicLLMClient, the
        native Anthropic Messages API adapter (added 2026-09-14, 0단계 of the
        Claude-backend measurement plan — see project memory). Imported
        lazily, here and only here, so a deployment that never sets
        llm_provider="anthropic" never even imports the `anthropic` package.

    Both branches return an app.domain.ports.llm_port.LLMPort-satisfying
    instance — every call site downstream (agents, chat router, ...) is
    already written against that Protocol, not against LLMClient directly,
    so nothing else needs to change when the backend does.
    프로세스 전역 llm_client 단일 생성 지점 — llm_provider에 따라 백엔드 분기."""
    if settings.llm_provider == "anthropic":
        from app.llm.anthropic_client import AnthropicLLMClient

        return AnthropicLLMClient()
    return LLMClient()


llm_client = _build_llm_client()

# Name -> zero-arg factory returning an app.domain.ports.LLMPort-satisfying
# client instance. Same pattern as app.brokers.__init__'s
# _BROKER_FACTORIES/get_broker(name) — a genuinely different backend (not
# just a different base_url/model for the same OpenAI-compatible wire
# protocol, which is just a settings change) would register here instead of
# being wired into LLMClient itself. "default" mirrors the module-level
# `llm_client` singleton's own provider selection (_build_llm_client) rather
# than hardcoding LLMClient, so the two never disagree about which backend
# "default" means.
# LLM 이름 -> 인스턴스 팩토리 (브로커의 _BROKER_FACTORIES/get_broker와 동일 패턴)
_LLM_FACTORIES = {
    "default": _build_llm_client,
}

# name -> cached instance, populated lazily on first get_llm_client() call —
# same "one instance per name for the process lifetime" reasoning as
# app.brokers.get_broker (a fresh instance per call would mean a fresh
# AsyncOpenAI client/connection pool every time, not a throttle concern here
# but still wasteful).
_llm_instances: dict[str, object] = {}


def get_llm_client(name: str = "default"):
    """Look up an LLM client by name instead of importing a concrete class
    directly — mirrors app.brokers.get_broker(name). The module-level
    `llm_client` singleton above remains the one every agent module actually
    imports today; this factory exists so a genuinely different backend can
    be registered in `_LLM_FACTORIES` and swapped in without touching every
    call site, the same way most of the app still reads
    `app.web.state._kis_broker` directly rather than calling `get_broker()`
    at each site.
    이름으로 LLM 클라이언트 조회(브로커의 get_broker와 동일 패턴) — 기존 llm_client
    전역 싱글턴과 호출부는 그대로 유지, 새 백엔드 추가 시 이 팩토리에 등록."""
    if name not in _llm_instances:
        try:
            factory = _LLM_FACTORIES[name]
        except KeyError:
            raise ValueError(f"unknown LLM client: {name!r}, known: {list(_LLM_FACTORIES)}") from None
        _llm_instances[name] = factory()
    return _llm_instances[name]


__all__ = ["LLMClient", "LLMMessage", "llm_client", "get_llm_client"]
