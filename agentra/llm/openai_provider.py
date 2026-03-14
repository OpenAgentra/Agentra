"""OpenAI LLM provider (supports GPT-4o, GPT-4-vision, etc.)."""

from __future__ import annotations

import base64
from typing import Any, Optional

from agentra.config import AgentConfig
from agentra.llm.base import LLMMessage, LLMProvider, LLMResponse


class OpenAIProvider(LLMProvider):
    """Wraps the ``openai`` async client."""

    def __init__(self, config: AgentConfig) -> None:
        try:
            import openai  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError("Install 'openai' to use OpenAIProvider.") from exc

        api_key = config.openai_api_key or ""
        self._client = openai.AsyncOpenAI(api_key=api_key)
        self._config = config

    async def complete(
        self,
        messages: list[LLMMessage],
        tools: Optional[list[dict[str, Any]]] = None,
        temperature: float = 0.2,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        oai_messages = [self._to_oai_message(m) for m in messages]
        kwargs: dict[str, Any] = {
            "model": self._config.llm_model,
            "messages": oai_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        response = await self._client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        msg = choice.message
        tool_calls = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                import json  # noqa: PLC0415

                tool_calls.append(
                    {
                        "id": tc.id,
                        "name": tc.function.name,
                        "arguments": json.loads(tc.function.arguments),
                    }
                )
        return LLMResponse(
            content=msg.content or "",
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason or "stop",
            usage={
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
            },
        )

    async def embed(self, text: str) -> list[float]:
        response = await self._client.embeddings.create(
            model="text-embedding-3-small",
            input=text,
        )
        return response.data[0].embedding

    # ── helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _to_oai_message(msg: LLMMessage) -> dict[str, Any]:
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
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": __import__("json").dumps(tc["arguments"]),
                        },
                    }
                    for tc in msg.tool_calls
                ],
            }
        if msg.tool_call_id:
            return {
                "role": "tool",
                "tool_call_id": msg.tool_call_id,
                "content": msg.content,
            }
        return {"role": msg.role, "content": msg.content}
