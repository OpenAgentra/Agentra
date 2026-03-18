"""Provider metadata and lookup helpers."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agentra.config import AgentConfig
    from agentra.llm.base import LLMProvider


@dataclass(frozen=True)
class ProviderSpec:
    """Metadata describing an LLM provider."""

    provider_id: str
    label: str
    default_model: str
    factory_path: str
    api_key_env: str | None = None
    base_url_env: str | None = None
    capabilities: tuple[str, ...] = ("text",)


_PROVIDERS: dict[str, ProviderSpec] = {
    "openai": ProviderSpec(
        provider_id="openai",
        label="OpenAI",
        default_model="gpt-4o",
        factory_path="agentra.llm.openai_provider:OpenAIProvider",
        api_key_env="AGENTRA_OPENAI_API_KEY",
        capabilities=("text", "images", "tools", "embeddings"),
    ),
    "anthropic": ProviderSpec(
        provider_id="anthropic",
        label="Anthropic",
        default_model="claude-3-5-sonnet-latest",
        factory_path="agentra.llm.anthropic_provider:AnthropicProvider",
        api_key_env="AGENTRA_ANTHROPIC_API_KEY",
        capabilities=("text", "images", "tools"),
    ),
    "ollama": ProviderSpec(
        provider_id="ollama",
        label="Ollama",
        default_model="llava",
        factory_path="agentra.llm.ollama_provider:OllamaProvider",
        base_url_env="AGENTRA_OLLAMA_BASE_URL",
        capabilities=("text", "images", "tools", "embeddings"),
    ),
    "gemini": ProviderSpec(
        provider_id="gemini",
        label="Gemini",
        default_model="gemini-3-flash-preview",
        factory_path="agentra.llm.gemini_provider:GeminiProvider",
        api_key_env="AGENTRA_GEMINI_API_KEY",
        base_url_env="AGENTRA_GEMINI_BASE_URL",
        capabilities=("text", "images", "tools"),
    ),
}


def provider_specs() -> tuple[ProviderSpec, ...]:
    """Return all known provider specs in display order."""
    return tuple(_PROVIDERS.values())


def provider_ids() -> tuple[str, ...]:
    """Return all known provider ids."""
    return tuple(_PROVIDERS.keys())


def get_provider_spec(provider_id: str) -> ProviderSpec:
    """Return metadata for *provider_id* or raise a friendly error."""
    try:
        return _PROVIDERS[provider_id]
    except KeyError as exc:
        known = ", ".join(provider_ids())
        raise ValueError(f"Unknown llm_provider: {provider_id!r}. Expected one of: {known}") from exc


def create_provider(config: "AgentConfig") -> "LLMProvider":
    """Instantiate the configured provider from registry metadata."""
    spec = get_provider_spec(config.llm_provider)
    module_name, class_name = spec.factory_path.split(":", 1)
    provider_cls = getattr(import_module(module_name), class_name)
    return provider_cls(config)
