"""Factory that instantiates the correct LLM provider."""

from __future__ import annotations

from agentra.config import AgentConfig
from agentra.llm.base import LLMProvider
from agentra.llm.registry import create_provider
from agentra.memory.providers import LLMEmbeddingProvider, EmbeddingProvider


def get_provider(config: AgentConfig, *, role: str = "executor") -> LLMProvider:
    """Return an :class:`LLMProvider` appropriate for *config* and *role*."""
    model = _model_for_role(config, role)
    provider_config = config if model == config.llm_model else config.model_copy(update={"llm_model": model})
    return create_provider(provider_config)


def get_embedding_provider(config: AgentConfig) -> EmbeddingProvider:
    """Return the embedding provider configured for the current runtime."""
    return LLMEmbeddingProvider(get_provider(config, role="embedding"))


def _model_for_role(config: AgentConfig, role: str) -> str:
    if role == "planner":
        return config.planner_model or config.llm_model
    if role == "summary":
        return config.summary_model or config.llm_model
    if role == "embedding":
        return config.embedding_model or config.llm_model
    return config.executor_model or config.llm_model
