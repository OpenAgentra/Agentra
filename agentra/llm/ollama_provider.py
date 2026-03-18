"""Ollama LLM provider — run open-source models locally for free."""

from __future__ import annotations

import json
from typing import Any, Optional

import httpx

from agentra.config import AgentConfig
from agentra.llm.base import LLMMessage, LLMProvider, LLMResponse


class OllamaProvider(LLMProvider):
    """Calls the Ollama REST API (compatible with OpenAI chat format)."""

    def __init__(self, config: AgentConfig) -> None:
        self._base_url = config.ollama_base_url.rstrip("/")
        self._config = config
        self._client = httpx.AsyncClient(base_url=self._base_url, timeout=120.0)

    async def complete(
        self,
        messages: list[LLMMessage],
        tools: Optional[list[dict[str, Any]]] = None,
        temperature: float = 0.2,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self._config.llm_model,
            "messages": [self._to_ollama_message(m) for m in messages],
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        if tools:
            payload["tools"] = tools

        resp = await self._client.post("/api/chat", json=payload)
        resp.raise_for_status()
        data = resp.json()

        msg = data.get("message", {})
        content = msg.get("content", "")
        tool_calls: list[dict[str, Any]] = []
        for tc in msg.get("tool_calls", []):
            fn = tc.get("function", {})
            args = fn.get("arguments", {})
            if isinstance(args, str):
                args = json.loads(args)
            tool_calls.append(
                {
                    "id": tc.get("id", fn.get("name", "")),
                    "name": fn.get("name", ""),
                    "arguments": args,
                }
            )

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason=data.get("done_reason", "stop"),
        )

    async def embed(self, text: str) -> list[float]:
        resp = await self._client.post(
            "/api/embed",
            json={"model": self._config.llm_model, "input": text},
        )
        resp.raise_for_status()
        data = resp.json()
        # Ollama returns {"embeddings": [[...]]}
        embeddings = data.get("embeddings", data.get("embedding", []))
        if embeddings and isinstance(embeddings[0], list):
            return embeddings[0]
        return embeddings

    async def aclose(self) -> None:
        await self._client.aclose()

    # ── helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _to_ollama_message(msg: LLMMessage) -> dict[str, Any]:
        payload: dict[str, Any] = {"role": msg.role, "content": msg.content}
        if msg.images:
            payload["images"] = msg.images  # Ollama accepts base64 images
        if msg.tool_calls:
            payload["tool_calls"] = [
                {
                    "id": tc.get("id"),
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": tc["arguments"],
                    },
                }
                for tc in msg.tool_calls
            ]
        if msg.tool_call_id:
            payload["role"] = "tool"
            payload["tool_call_id"] = msg.tool_call_id
            if msg.tool_name:
                payload["name"] = msg.tool_name
        return payload
