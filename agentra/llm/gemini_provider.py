"""Gemini provider via Google's OpenAI-compatible endpoint."""

from __future__ import annotations

import json
from typing import Any, Optional

from agentra.config import AgentConfig
from agentra.llm.base import LLMMessage, LLMProvider, LLMResponse


class GeminiProvider(LLMProvider):
    """Wrap the Google Gemini OpenAI-compatible chat endpoint."""

    def __init__(self, config: AgentConfig) -> None:
        try:
            import openai  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError("Install 'openai' to use GeminiProvider.") from exc

        api_key = config.gemini_api_key or ""
        base_url = config.gemini_base_url.rstrip("/") + "/"
        self._client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._config = config

    async def complete(
        self,
        messages: list[LLMMessage],
        tools: Optional[list[dict[str, Any]]] = None,
        temperature: float = 0.2,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        gemini_messages = [self._to_openai_message(m) for m in messages]
        kwargs: dict[str, Any] = {
            "model": self._config.llm_model,
            "messages": gemini_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        response = await self._client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        message = choice.message

        tool_calls: list[dict[str, Any]] = []
        if message.tool_calls:
            for tool_call in message.tool_calls:
                arguments = tool_call.function.arguments
                if isinstance(arguments, str):
                    arguments = json.loads(arguments)
                tool_calls.append(
                    {
                        "id": tool_call.id,
                        "name": tool_call.function.name,
                        "arguments": arguments,
                    }
                )

        return LLMResponse(
            content=message.content or "",
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason or "stop",
            usage={
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
            },
        )

    async def embed(self, text: str) -> list[float]:
        raise NotImplementedError(
            "Gemini embeddings are not configured in this provider. "
            "Agentra will fall back to local trivial embeddings for memory."
        )

    @staticmethod
    def _to_openai_message(msg: LLMMessage) -> dict[str, Any]:
        if msg.images:
            content: list[dict[str, Any]] = [{"type": "text", "text": msg.content}]
            for img_b64 in msg.images:
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                    }
                )
            return {"role": msg.role, "content": content}
        if msg.tool_calls:
            return {
                "role": msg.role,
                "content": msg.content,
                "tool_calls": [
                    {
                        "id": tool_call["id"],
                        "type": "function",
                        "function": {
                            "name": tool_call["name"],
                            "arguments": json.dumps(tool_call["arguments"]),
                        },
                    }
                    for tool_call in msg.tool_calls
                ],
            }
        if msg.tool_call_id:
            return {
                "role": "tool",
                "tool_call_id": msg.tool_call_id,
                "content": msg.content,
            }
        return {"role": msg.role, "content": msg.content}
