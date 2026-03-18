"""LLM provider package."""

from __future__ import annotations

from agentra.llm.base import LLMMessage, LLMProvider, LLMResponse, LLMSession, LLMToolResult

__all__ = ["LLMMessage", "LLMProvider", "LLMResponse", "LLMSession", "LLMToolResult", "get_provider"]


def __getattr__(name: str):
    if name == "get_provider":
        from agentra.llm.factory import get_provider

        return get_provider
    raise AttributeError(name)
