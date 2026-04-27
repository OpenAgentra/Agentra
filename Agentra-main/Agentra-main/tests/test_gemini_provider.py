"""Tests for the official Gemini provider implementation."""

from __future__ import annotations

import aiohttp
import asyncio
from types import SimpleNamespace

import pytest
from google.genai import types

import agentra.llm.gemini_provider as gemini_provider_module
from agentra.config import AgentConfig
from agentra.llm.base import LLMMessage, LLMToolResult
from agentra.llm.gemini_provider import GeminiProvider


PNG_1X1_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)


class FakeModels:
    """Capture Gemini generate_content calls and return queued responses."""

    def __init__(self, responses: list[object]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, object]] = []

    async def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


class FakeAioClient:
    """Async facade that looks like the Google GenAI client."""

    def __init__(self, responses: list[object]) -> None:
        self.models = FakeModels(responses)
        self.closed = False

    async def aclose(self) -> None:
        self.closed = True


class FakeClient:
    """Simple fake of google.genai.Client."""

    def __init__(self, responses: list[object], **kwargs) -> None:
        self.init_kwargs = kwargs
        self.aio = FakeAioClient(responses)


def _install_fake_client(monkeypatch, responses: list[object]) -> dict[str, object]:
    import google.genai

    state: dict[str, object] = {}

    def fake_client(*, api_key=None, http_options=None):
        client = FakeClient(responses, api_key=api_key, http_options=http_options)
        state["client"] = client
        return client

    monkeypatch.setattr(google.genai, "Client", fake_client)
    return state


def _make_response(
    *,
    text: str = "",
    candidate_content: types.Content | None = None,
    function_calls: list[types.FunctionCall] | None = None,
    finish_reason: str = "STOP",
    prompt_tokens: int = 11,
    completion_tokens: int = 7,
):
    function_calls = function_calls or []
    if candidate_content is None:
        parts = []
        if text:
            parts.append(types.Part.from_text(text=text))
        for call in function_calls:
            parts.append(types.Part(functionCall=call))
        if not parts:
            parts.append(types.Part.from_text(text=""))
        candidate_content = types.Content(role="model", parts=parts)

    return SimpleNamespace(
        text=text,
        function_calls=function_calls,
        candidates=[
            SimpleNamespace(
                content=candidate_content,
                finish_reason=SimpleNamespace(name=finish_reason),
            )
        ],
        usage_metadata=SimpleNamespace(
            prompt_token_count=prompt_tokens,
            candidates_token_count=completion_tokens,
        ),
    )


def _config(tmp_path) -> AgentConfig:
    return AgentConfig(
        llm_provider="gemini",
        llm_model="gemini-3-flash-preview",
        gemini_api_key="test-key",
        workspace_dir=tmp_path / "workspace",
        memory_dir=tmp_path / "workspace" / ".memory",
    )


@pytest.mark.asyncio
async def test_gemini_session_preserves_native_tool_turns(monkeypatch, tmp_path):
    tool_call = types.FunctionCall(
        id="call-1",
        name="browser",
        args={"action": "navigate", "url": "https://www.python.org"},
    )
    candidate_content = types.Content(
        role="model",
        parts=[
            types.Part(text="Opening python.org", thoughtSignature=b"demo-signature"),
            types.Part(functionCall=tool_call),
        ],
    )
    state = _install_fake_client(
        monkeypatch,
        [
            _make_response(
                text="Opening python.org",
                candidate_content=candidate_content,
                function_calls=[tool_call],
            ),
            _make_response(text="DONE: Python.org opened."),
        ],
    )

    provider = GeminiProvider(_config(tmp_path))
    session = provider.start_session(
        system_message=LLMMessage(role="system", content="Use tools carefully."),
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "browser",
                    "description": "Navigate the browser.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "url": {"type": "string"},
                        },
                    },
                },
            }
        ],
    )
    session.append(LLMMessage(role="user", content="Open python.org"))

    first = await session.complete()
    assert first.content == "Opening python.org"
    assert first.tool_calls == [
        {
            "id": "call-1",
            "name": "browser",
            "arguments": {"action": "navigate", "url": "https://www.python.org"},
        }
    ]

    session.append_tool_results(
        [
            LLMToolResult(
                tool_call_id="call-1",
                name="browser",
                content="Navigated to python.org",
                success=True,
            )
        ]
    )
    second = await session.complete()

    client = state["client"]
    calls = client.aio.models.calls
    assert len(calls) == 2
    assert calls[0]["model"] == "gemini-3-flash-preview"
    assert calls[0]["config"].model_dump()["system_instruction"] == "Use tools carefully."
    assert calls[0]["config"].model_dump()["automatic_function_calling"]["disable"] is True
    assert (
        calls[0]["config"].model_dump()["tools"][0]["function_declarations"][0]["name"]
        == "browser"
    )

    assert calls[1]["contents"][1] is candidate_content
    assert calls[1]["contents"][1].parts[0].thought_signature == b"demo-signature"
    tool_response = calls[1]["contents"][2].parts[0].function_response
    assert tool_response.id == "call-1"
    assert tool_response.name == "browser"
    assert tool_response.response["result"] == "Navigated to python.org"
    assert tool_response.response["success"] is True
    assert second.content == "DONE: Python.org opened."

    await session.aclose()
    assert client.aio.closed is True


@pytest.mark.asyncio
async def test_gemini_provider_serializes_multimodal_user_messages(monkeypatch, tmp_path):
    state = _install_fake_client(
        monkeypatch,
        [_make_response(text="DONE: I can see the screenshot.")],
    )

    provider = GeminiProvider(_config(tmp_path))
    response = await provider.complete(
        [
            LLMMessage(role="system", content="Describe images."),
            LLMMessage(
                role="user",
                content="What is in this screenshot?",
                images=[PNG_1X1_B64],
            ),
        ]
    )

    call = state["client"].aio.models.calls[0]
    assert call["config"].model_dump()["system_instruction"] == "Describe images."
    assert call["contents"][0].role == "user"
    assert call["contents"][0].parts[0].text == "What is in this screenshot?"
    assert call["contents"][0].parts[1].inline_data.mime_type == "image/png"
    assert call["contents"][0].parts[1].inline_data.data.startswith(b"\x89PNG")
    assert response.content == "DONE: I can see the screenshot."


def test_gemini_provider_backfills_missing_aiohttp_dns_error_alias(monkeypatch, tmp_path) -> None:
    import aiohttp

    state = _install_fake_client(monkeypatch, [_make_response(text="DONE: ok")])
    monkeypatch.delattr(aiohttp, "ClientConnectorDNSError", raising=False)

    GeminiProvider(_config(tmp_path))

    assert aiohttp.ClientConnectorDNSError is aiohttp.ClientConnectorError
    assert state["client"].init_kwargs["api_key"] == "test-key"


@pytest.mark.asyncio
async def test_gemini_provider_retries_transient_server_disconnect(monkeypatch, tmp_path) -> None:
    import google.genai

    created_clients: list[FakeClient] = []
    response_batches = [
        [aiohttp.ServerDisconnectedError()],
        [_make_response(text="DONE: recovered after retry")],
    ]

    def fake_client(*, api_key=None, http_options=None):
        responses = response_batches.pop(0)
        client = FakeClient(responses, api_key=api_key, http_options=http_options)
        created_clients.append(client)
        return client

    sleep_delays: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleep_delays.append(delay)

    monkeypatch.setattr(google.genai, "Client", fake_client)
    monkeypatch.setattr(gemini_provider_module.asyncio, "sleep", fake_sleep)

    provider = GeminiProvider(_config(tmp_path))
    response = await provider.complete([LLMMessage(role="user", content="Ping")])

    assert response.content == "DONE: recovered after retry"
    assert len(created_clients) == 2
    assert created_clients[0].aio.closed is True
    assert len(created_clients[0].aio.models.calls) == 1
    assert len(created_clients[1].aio.models.calls) == 1
    assert sleep_delays == [pytest.approx(0.35)]
