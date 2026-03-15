"""Embedding provider abstractions for Agentra memory."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Awaitable, Callable


class EmbeddingProvider(ABC):
    """Abstract embedding backend used by memory components."""

    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        """Return an embedding vector for *text*."""


class CallableEmbeddingProvider(EmbeddingProvider):
    """Adapter for async embedding callables."""

    def __init__(self, embed_fn: Callable[[str], Awaitable[list[float]]]) -> None:
        self._embed_fn = embed_fn

    async def embed(self, text: str) -> list[float]:
        return await self._embed_fn(text)


class LLMEmbeddingProvider(EmbeddingProvider):
    """Adapter that uses an LLM provider's embed method."""

    def __init__(self, llm_provider: Any) -> None:
        self._llm_provider = llm_provider

    async def embed(self, text: str) -> list[float]:
        return await self._llm_provider.embed(text)
