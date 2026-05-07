"""Tests for provider sessions and provider registry metadata."""

from __future__ import annotations

import pytest

from agentra.config import AgentConfig
from agentra.llm.base import LLMMessage, LLMProvider, LLMResponse, LLMToolResult
from agentra.llm.registry import get_provider_spec, provider_ids


class RecordingProvider(LLMProvider):
    """Minimal provider used to inspect the stateless session adapter."""

    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, object]] = []

    async def complete(self, messages, tools=None, temperature=0.2, max_tokens=4096):
        self.calls.append(
            {
                "messages": list(messages),
                "tools": tools,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )
        return self._responses.pop(0)

    async def embed(self, text: str) -> list[float]:
        return [0.0, 0.0, 0.0]


@pytest.mark.asyncio
async def test_stateless_session_replays_assistant_tool_calls_and_tool_results():
    provider = RecordingProvider(
        [
            LLMResponse(
                content="Opening python.org",
                tool_calls=[
                    {
                        "id": "call-1",
                        "name": "browser",
                        "arguments": {
                            "action": "navigate",
                            "url": "https://www.python.org",
                        },
                    }
                ],
            ),
            LLMResponse(content="DONE: Loaded python.org"),
        ]
    )
    session = provider.start_session(
        system_message=LLMMessage(role="system", content="You are helpful."),
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "browser",
                    "description": "Navigate the browser.",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
    )

    session.append(LLMMessage(role="user", content="Open python.org"))
    first = await session.complete(
        extra_messages=[LLMMessage(role="user", content="A screenshot is also attached.")]
    )

    assert first.tool_calls[0]["name"] == "browser"
    assert [message.role for message in provider.calls[0]["messages"]] == [
        "system",
        "user",
        "user",
    ]

    session.append_tool_results(
        [
            LLMToolResult(
                tool_call_id="call-1",
                name="browser",
                content="Navigated to python.org",
                success=False,
            )
        ]
    )
    second = await session.complete()

    second_messages = provider.calls[1]["messages"]
    assert [message.role for message in second_messages] == [
        "system",
        "user",
        "assistant",
        "tool",
    ]
    assert second_messages[2].tool_calls[0]["arguments"]["url"] == "https://www.python.org"
    assert second_messages[3].tool_call_id == "call-1"
    assert second_messages[3].tool_name == "browser"
    assert second_messages[3].tool_success is False
    assert second.content == "DONE: Loaded python.org"


def test_provider_registry_exposes_expected_defaults():
    assert provider_ids() == ("openai", "anthropic", "ollama", "gemini")

    gemini = get_provider_spec("gemini")
    assert gemini.default_model == "gemini-3-flash-preview"
    assert gemini.api_key_env == "AGENTRA_GEMINI_API_KEY"
    assert "images" in gemini.capabilities
    assert "tools" in gemini.capabilities

    ollama = get_provider_spec("ollama")
    assert ollama.base_url_env == "AGENTRA_OLLAMA_BASE_URL"


def test_config_applies_provider_default_model(tmp_path):
    config = AgentConfig(
        llm_provider="gemini",
        llm_model="gpt-4o",
        workspace_dir=tmp_path / "workspace",
        memory_dir=tmp_path / "workspace" / ".memory",
    )

    assert config.llm_model == "gemini-3-flash-preview"
    assert config.gemini_base_url == "https://generativelanguage.googleapis.com"
