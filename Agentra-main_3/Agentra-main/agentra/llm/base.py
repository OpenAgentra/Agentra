"""Abstract LLM provider and session interfaces."""

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
    tool_name: Optional[str] = None
    tool_success: bool = True
    tool_calls: Optional[list[dict[str, Any]]] = None


@dataclass
class LLMToolResult:
    """A single tool execution result returned to the provider."""

    tool_call_id: str
    name: str
    content: str
    success: bool = True


@dataclass
class LLMResponse:
    """Response from an LLM provider."""

    content: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    finish_reason: str = "stop"
    usage: dict[str, int] = field(default_factory=dict)


class LLMSession(ABC):
    """Provider-owned conversation session."""

    @abstractmethod
    def append(self, message: LLMMessage) -> None:
        """Persist a message in the session history."""

    @abstractmethod
    def append_tool_results(self, results: list[LLMToolResult]) -> None:
        """Persist tool results in the session history."""

    @abstractmethod
    async def complete(
        self,
        *,
        extra_messages: Optional[list[LLMMessage]] = None,
        temperature: float = 0.2,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """Generate the next model response."""

    async def aclose(self) -> None:
        """Close any provider resources held by the session."""


class StatelessLLMSession(LLMSession):
    """Adapter session for providers that only implement stateless completion."""

    def __init__(
        self,
        provider: "LLMProvider",
        *,
        system_message: Optional[LLMMessage] = None,
        tools: Optional[list[dict[str, Any]]] = None,
    ) -> None:
        self._provider = provider
        self._messages: list[LLMMessage] = []
        self._tools = tools
        if system_message is not None:
            self._messages.append(system_message)

    def append(self, message: LLMMessage) -> None:
        self._messages.append(message)

    def append_tool_results(self, results: list[LLMToolResult]) -> None:
        for result in results:
            self._messages.append(
                LLMMessage(
                    role="tool",
                    content=result.content,
                    tool_call_id=result.tool_call_id,
                    tool_name=result.name,
                    tool_success=result.success,
                )
            )

    async def complete(
        self,
        *,
        extra_messages: Optional[list[LLMMessage]] = None,
        temperature: float = 0.2,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        request_messages = list(self._messages)
        if extra_messages:
            request_messages.extend(extra_messages)

        response = await self._provider.complete(
            messages=request_messages,
            tools=self._tools,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if response.content or response.tool_calls:
            self._messages.append(
                LLMMessage(
                    role="assistant",
                    content=response.content,
                    tool_calls=response.tool_calls or None,
                )
            )
        return response

    async def aclose(self) -> None:
        aclose = getattr(self._provider, "aclose", None)
        if callable(aclose):
            await aclose()


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

    def start_session(
        self,
        *,
        system_message: Optional[LLMMessage] = None,
        tools: Optional[list[dict[str, Any]]] = None,
    ) -> LLMSession:
        """Create a conversation session for the provider."""
        return StatelessLLMSession(self, system_message=system_message, tools=tools)
