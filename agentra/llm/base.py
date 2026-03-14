"""Abstract LLM provider interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class LLMMessage:
    """A single message in a conversation."""

    role: str  # "system" | "user" | "assistant" | "tool"
    content: str
    images: list[str] = field(default_factory=list)  # base64-encoded PNGs
    tool_call_id: Optional[str] = None
    tool_calls: Optional[list[dict[str, Any]]] = None


@dataclass
class LLMResponse:
    """Response from an LLM provider."""

    content: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    finish_reason: str = "stop"
    usage: dict[str, int] = field(default_factory=dict)


class LLMProvider(ABC):
    """Abstract base class every LLM back-end must implement."""

    @abstractmethod
    async def complete(
        self,
        messages: list[LLMMessage],
        tools: Optional[list[dict[str, Any]]] = None,
        temperature: float = 0.2,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """Send *messages* to the model and return its reply."""

    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        """Return a dense embedding vector for *text*."""
