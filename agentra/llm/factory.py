"""Factory that instantiates the correct LLM provider."""

from __future__ import annotations

from agentra.config import AgentConfig
from agentra.llm.base import LLMProvider


def get_provider(config: AgentConfig) -> LLMProvider:
    """Return an :class:`LLMProvider` appropriate for *config*."""
    if config.llm_provider == "openai":
        from agentra.llm.openai_provider import OpenAIProvider

        return OpenAIProvider(config)
    if config.llm_provider == "anthropic":
        from agentra.llm.anthropic_provider import AnthropicProvider

        return AnthropicProvider(config)
    if config.llm_provider == "ollama":
        from agentra.llm.ollama_provider import OllamaProvider

        return OllamaProvider(config)
    if config.llm_provider == "gemini":
        from agentra.llm.gemini_provider import GeminiProvider

        return GeminiProvider(config)
    raise ValueError(f"Unknown llm_provider: {config.llm_provider!r}")
