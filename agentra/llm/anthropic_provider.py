"""Anthropic LLM provider (Claude 3.x with computer-use support)."""

from __future__ import annotations

import json
from typing import Any, Optional

from agentra.config import AgentConfig
from agentra.llm.base import LLMMessage, LLMProvider, LLMResponse


class AnthropicProvider(LLMProvider):
    """Wraps the ``anthropic`` async client."""

    def __init__(self, config: AgentConfig) -> None:
        try:
            import anthropic  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError("Install 'anthropic' to use AnthropicProvider.") from exc

        api_key = config.anthropic_api_key or ""
        self._client = anthropic.AsyncAnthropic(api_key=api_key)
        self._config = config

    async def complete(
        self,
        messages: list[LLMMessage],
        tools: Optional[list[dict[str, Any]]] = None,
        temperature: float = 0.2,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        system_prompt = ""
        anthropic_messages = self._build_anthropic_messages(messages)
        for msg in messages:
            if msg.role == "system":
                system_prompt = msg.content
                break

        kwargs: dict[str, Any] = {
            "model": self._config.llm_model,
            "messages": anthropic_messages,
            "max_tokens": max_tokens,
        }
        if system_prompt:
            kwargs["system"] = system_prompt
        if tools:
            kwargs["tools"] = self._convert_tools(tools)

        response = await self._client.messages.create(**kwargs)

        text_content = ""
        tool_calls: list[dict[str, Any]] = []
        for block in response.content:
            if block.type == "text":
                text_content += block.text
            elif block.type == "tool_use":
                tool_calls.append(
                    {
                        "id": block.id,
                        "name": block.name,
                        "arguments": block.input,
                    }
                )

        return LLMResponse(
            content=text_content,
            tool_calls=tool_calls,
            finish_reason=response.stop_reason or "stop",
            usage={
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
            },
        )

    async def embed(self, text: str) -> list[float]:
        raise NotImplementedError(
            "Anthropic does not provide an embeddings API. "
            "Use OpenAI or a local sentence-transformers model."
        )

    # ── helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _to_anthropic_message(msg: LLMMessage) -> dict[str, Any]:
        if msg.tool_calls:
            content: list[dict[str, Any]] = []
            if msg.content:
                content.append({"type": "text", "text": msg.content})
            for tool_call in msg.tool_calls:
                content.append(
                    {
                        "type": "tool_use",
                        "id": tool_call["id"],
                        "name": tool_call["name"],
                        "input": tool_call["arguments"],
                    }
                )
            return {"role": "assistant", "content": content}
        if msg.images:
            content: list[dict[str, Any]] = [{"type": "text", "text": msg.content}]
            for img_b64 in msg.images:
                content.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": img_b64,
                        },
                    }
                )
            return {"role": msg.role, "content": content}
        if msg.tool_call_id:
            return {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": msg.tool_call_id,
                        "content": msg.content,
                        "is_error": not msg.tool_success,
                    }
                ],
            }
        return {"role": msg.role, "content": msg.content}

    @classmethod
    def _build_anthropic_messages(cls, messages: list[LLMMessage]) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        pending_tool_results: list[dict[str, Any]] = []

        def flush_tool_results() -> None:
            if pending_tool_results:
                result.append({"role": "user", "content": list(pending_tool_results)})
                pending_tool_results.clear()

        for msg in messages:
            if msg.role == "system":
                continue
            if msg.role == "tool" and msg.tool_call_id:
                pending_tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": msg.tool_call_id,
                        "content": msg.content,
                        "is_error": not msg.tool_success,
                    }
                )
                continue

            flush_tool_results()
            result.append(cls._to_anthropic_message(msg))

        flush_tool_results()
        return result

    @staticmethod
    def _convert_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert OpenAI-style tool schema to Anthropic format."""
        result = []
        for t in tools:
            fn = t.get("function", t)
            result.append(
                {
                    "name": fn["name"],
                    "description": fn.get("description", ""),
                    "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
                }
            )
        return result
