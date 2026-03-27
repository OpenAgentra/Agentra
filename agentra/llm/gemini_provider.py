"""Gemini provider implemented with Google's official GenAI SDK."""

from __future__ import annotations

import asyncio
import base64
import logging
from typing import Any, Optional

from agentra.config import AgentConfig
from agentra.llm.base import LLMMessage, LLMProvider, LLMResponse, LLMSession, LLMToolResult

logger = logging.getLogger(__name__)


def _ensure_aiohttp_dns_error_alias() -> None:
    try:
        import aiohttp  # noqa: PLC0415
    except ImportError:
        return
    if hasattr(aiohttp, "ClientConnectorDNSError"):
        return
    fallback = getattr(aiohttp, "ClientConnectorError", None)
    if fallback is not None:
        setattr(aiohttp, "ClientConnectorDNSError", fallback)


class GeminiSession(LLMSession):
    """Stateful Gemini session that preserves native model turn history."""

    def __init__(
        self,
        provider: "GeminiProvider",
        *,
        system_message: Optional[LLMMessage] = None,
        tools: Optional[list[dict[str, Any]]] = None,
    ) -> None:
        self._provider = provider
        self._history: list[Any] = []
        self._system_instruction = system_message.content if system_message else None
        self._tools = tools

    def append(self, message: LLMMessage) -> None:
        if message.role == "system":
            self._system_instruction = message.content
            return
        self._history.append(self._provider._message_to_content(message))

    def append_tool_results(self, results: list[LLMToolResult]) -> None:
        if not results:
            return
        self._history.append(self._provider._tool_results_to_content(results))

    async def complete(
        self,
        *,
        extra_messages: Optional[list[LLMMessage]] = None,
        temperature: float = 0.2,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        contents = list(self._history)
        if extra_messages:
            contents.extend(self._provider._message_to_content(message) for message in extra_messages)

        response = await self._provider._generate(
            contents=contents,
            system_instruction=self._system_instruction,
            tools=self._tools,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        content = self._provider._extract_candidate_content(response)
        if content is not None:
            self._history.append(content)
        return self._provider._parse_response(response)

    async def aclose(self) -> None:
        await self._provider.aclose()


class GeminiProvider(LLMProvider):
    """Wrap the Google GenAI SDK for generic multimodal tool use."""

    def __init__(self, config: AgentConfig) -> None:
        try:
            from google import genai  # noqa: PLC0415
            from google.genai import types  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError("Install 'google-genai' to use GeminiProvider.") from exc

        _ensure_aiohttp_dns_error_alias()
        self._genai = genai
        self._types = types
        self._config = config
        self._client = self._create_client()

    def _create_client(self) -> Any:
        return self._genai.Client(
            api_key=self._config.gemini_api_key or None,
            http_options=self._build_http_options(),
        )

    async def _reset_client(self) -> None:
        current = self._client
        self._client = self._create_client()
        try:
            await current.aio.aclose()
        except Exception:  # noqa: BLE001
            logger.debug("Failed to close stale Gemini client", exc_info=True)

    def start_session(
        self,
        *,
        system_message: Optional[LLMMessage] = None,
        tools: Optional[list[dict[str, Any]]] = None,
    ) -> LLMSession:
        return GeminiSession(self, system_message=system_message, tools=tools)

    async def complete(
        self,
        messages: list[LLMMessage],
        tools: Optional[list[dict[str, Any]]] = None,
        temperature: float = 0.2,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        system_instruction = None
        contents = self._messages_to_contents(messages)
        for message in messages:
            if message.role == "system":
                system_instruction = message.content
                break

        response = await self._generate(
            contents=contents,
            system_instruction=system_instruction,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return self._parse_response(response)

    async def embed(self, text: str) -> list[float]:
        raise NotImplementedError(
            "Gemini embeddings are not configured in this provider. "
            "Agentra will fall back to local trivial embeddings for memory."
        )

    async def aclose(self) -> None:
        await self._client.aio.aclose()

    async def _generate(
        self,
        *,
        contents: list[Any],
        system_instruction: Optional[str],
        tools: Optional[list[dict[str, Any]]],
        temperature: float,
        max_tokens: int,
    ) -> Any:
        config = self._types.GenerateContentConfig(
            systemInstruction=system_instruction,
            temperature=temperature,
            maxOutputTokens=max_tokens,
            tools=self._convert_tools(tools),
            automaticFunctionCalling=self._types.AutomaticFunctionCallingConfig(disable=True),
        )
        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            try:
                return await self._client.aio.models.generate_content(
                    model=self._config.llm_model,
                    contents=contents,
                    config=config,
                )
            except Exception as exc:  # noqa: BLE001
                if not self._is_transient_generate_error(exc) or attempt >= max_attempts:
                    raise
                delay = min(2.0, 0.35 * (2 ** (attempt - 1)))
                logger.warning(
                    "Transient Gemini request failure model=%s attempt=%d/%d error=%s; retrying in %.2fs",
                    self._config.llm_model,
                    attempt,
                    max_attempts,
                    exc,
                    delay,
                )
                await self._reset_client()
                await asyncio.sleep(delay)
        raise RuntimeError("Gemini generation retry loop exhausted unexpectedly")

    def _is_transient_generate_error(self, exc: Exception) -> bool:
        try:
            import aiohttp  # noqa: PLC0415
        except ImportError:
            aiohttp = None

        transient_types: tuple[type[BaseException], ...] = (
            asyncio.TimeoutError,
            TimeoutError,
            ConnectionError,
        )
        if isinstance(exc, transient_types):
            return True
        if aiohttp is not None and isinstance(
            exc,
            (
                aiohttp.ServerDisconnectedError,
                aiohttp.ClientConnectionError,
                aiohttp.ClientOSError,
                aiohttp.ClientPayloadError,
            ),
        ):
            return True

        status_code = getattr(exc, "status_code", None)
        if status_code is None:
            status_code = getattr(exc, "code", None)
        try:
            if status_code is not None and int(status_code) >= 500:
                return True
        except (TypeError, ValueError):
            pass

        message = " ".join(
            part
            for part in (
                str(exc),
                str(getattr(exc, "status", "") or ""),
                str(getattr(exc, "message", "") or ""),
            )
            if part
        ).lower()
        transient_markers = (
            "server disconnected",
            "connection reset",
            "connection closed",
            "temporarily unavailable",
            "upstream connect error",
            "remote host closed",
        )
        return any(marker in message for marker in transient_markers)

    def _build_http_options(self) -> Any:
        kwargs: dict[str, Any] = {"api_version": "v1beta"}
        if self._config.gemini_base_url:
            kwargs["base_url"] = self._config.gemini_base_url.rstrip("/")
        return self._types.HttpOptions(**kwargs)

    def _convert_tools(self, tools: Optional[list[dict[str, Any]]]) -> Optional[list[Any]]:
        if not tools:
            return None
        declarations = []
        for tool in tools:
            fn = tool.get("function", tool)
            declarations.append(
                self._types.FunctionDeclaration(
                    name=fn["name"],
                    description=fn.get("description", ""),
                    parametersJsonSchema=fn.get(
                        "parameters",
                        {"type": "object", "properties": {}},
                    ),
                )
            )
        return [self._types.Tool(functionDeclarations=declarations)]

    def _messages_to_contents(self, messages: list[LLMMessage]) -> list[Any]:
        contents: list[Any] = []
        pending_tool_results: list[LLMMessage] = []

        def flush_tool_results() -> None:
            if pending_tool_results:
                contents.append(self._tool_messages_to_content(pending_tool_results))
                pending_tool_results.clear()

        for message in messages:
            if message.role == "system":
                continue
            if message.role == "tool":
                pending_tool_results.append(message)
                continue

            flush_tool_results()
            contents.append(self._message_to_content(message))

        flush_tool_results()
        return contents

    def _message_to_content(self, message: LLMMessage) -> Any:
        parts = []

        if message.role == "tool":
            if not message.tool_name:
                raise ValueError(
                    "Gemini tool messages require tool_name so function responses can be reconstructed."
                )
            parts.append(self._function_response_part(
                name=message.tool_name,
                tool_call_id=message.tool_call_id,
                content=message.content,
                success=message.tool_success,
            ))
            return self._types.Content(role="user", parts=parts)

        if message.content:
            parts.append(self._types.Part.from_text(text=message.content))
        for image_b64 in message.images:
            parts.append(
                self._types.Part.from_bytes(
                    data=base64.b64decode(image_b64),
                    mime_type="image/png",
                )
            )
        if message.tool_calls:
            for tool_call in message.tool_calls:
                parts.append(
                    self._types.Part(
                        function_call=self._types.FunctionCall(
                            id=tool_call.get("id"),
                            name=tool_call["name"],
                            args=tool_call["arguments"],
                        )
                    )
                )
        if not parts:
            parts.append(self._types.Part.from_text(text=""))

        role = "model" if message.role == "assistant" else "user"
        return self._types.Content(role=role, parts=parts)

    def _tool_results_to_content(self, results: list[LLMToolResult]) -> Any:
        parts = [
            self._function_response_part(
                name=result.name,
                tool_call_id=result.tool_call_id,
                content=result.content,
                success=result.success,
            )
            for result in results
        ]
        return self._types.Content(role="user", parts=parts)

    def _tool_messages_to_content(self, messages: list[LLMMessage]) -> Any:
        parts = [
            self._function_response_part(
                name=message.tool_name,
                tool_call_id=message.tool_call_id,
                content=message.content,
                success=message.tool_success,
            )
            for message in messages
        ]
        return self._types.Content(role="user", parts=parts)

    def _function_response_part(
        self,
        *,
        name: Optional[str],
        tool_call_id: Optional[str],
        content: str,
        success: bool,
    ) -> Any:
        if not name:
            raise ValueError("Gemini tool messages require a tool name.")
        return self._types.Part(
            function_response=self._types.FunctionResponse(
                id=tool_call_id,
                name=name,
                response={
                    "result": content,
                    "success": success,
                },
            )
        )

    @staticmethod
    def _extract_candidate_content(response: Any) -> Any:
        if not getattr(response, "candidates", None):
            return None
        return response.candidates[0].content

    def _parse_response(self, response: Any) -> LLMResponse:
        tool_calls: list[dict[str, Any]] = []
        for function_call in response.function_calls or []:
            tool_calls.append(
                {
                    "id": function_call.id or function_call.name,
                    "name": function_call.name,
                    "arguments": dict(function_call.args or {}),
                }
            )

        finish_reason = "stop"
        if getattr(response, "candidates", None):
            candidate = response.candidates[0]
            reason = getattr(candidate, "finish_reason", None)
            finish_reason = getattr(reason, "name", None) or str(reason or "stop")

        usage = {}
        usage_metadata = getattr(response, "usage_metadata", None)
        if usage_metadata is not None:
            usage = {
                "prompt_tokens": getattr(usage_metadata, "prompt_token_count", 0) or 0,
                "completion_tokens": getattr(usage_metadata, "candidates_token_count", 0) or 0,
            }

        return LLMResponse(
            content=self._response_text(response),
            tool_calls=tool_calls,
            finish_reason=finish_reason.lower(),
            usage=usage,
        )

    @staticmethod
    def _response_text(response: Any) -> str:
        if not getattr(response, "candidates", None):
            return ""

        content = getattr(response.candidates[0], "content", None)
        if content is None:
            return ""

        text_parts = []
        for part in getattr(content, "parts", []) or []:
            text = getattr(part, "text", None)
            if text:
                text_parts.append(text)
        return "".join(text_parts)
