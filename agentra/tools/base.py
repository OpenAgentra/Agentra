"""Abstract base for all Agentra tools."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ToolResult:
    """Outcome of executing a tool action."""

    success: bool
    output: str = ""
    error: str = ""
    screenshot_b64: Optional[str] = None  # base64 PNG when relevant
    extra_screenshots: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        if self.success:
            return self.output or "OK"
        return f"ERROR: {self.error}"


class BaseTool(ABC):
    """Abstract base class every tool must implement."""

    tool_capabilities: tuple[str, ...] = ()

    @property
    @abstractmethod
    def name(self) -> str:
        """Short snake_case name used to identify this tool."""

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description used in the LLM system prompt."""

    @property
    def capabilities(self) -> tuple[str, ...]:
        """Execution capabilities required by this tool call."""
        return self.tool_capabilities or (self.name,)

    @property
    @abstractmethod
    def schema(self) -> dict[str, Any]:
        """JSON Schema for this tool's parameters (OpenAI tool format)."""

    @abstractmethod
    async def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the tool with the given keyword arguments."""

    async def preview(self, **kwargs: Any) -> dict[str, Any] | None:
        """Return optional non-mutating preview metadata for the pending action."""
        return None

    def to_openai_tool(self) -> dict[str, Any]:
        """Return the OpenAI function-calling schema for this tool."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.schema,
            },
        }
