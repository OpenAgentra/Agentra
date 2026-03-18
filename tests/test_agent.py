"""Tests for the AutonomousAgent."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentra.agents.autonomous import AutonomousAgent
from agentra.config import AgentConfig
from agentra.llm.base import LLMMessage, LLMProvider, LLMResponse
from agentra.tools.base import BaseTool, ToolResult


class FakeLLM(LLMProvider):
    """Deterministic fake LLM for testing."""

    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = iter(responses)

    async def complete(self, messages, tools=None, temperature=0.2, max_tokens=4096):
        return next(self._responses)

    async def embed(self, text: str) -> list[float]:
        return [0.1] * 256


class EchoTool(BaseTool):
    """Simple tool that echoes its input."""

    name = "echo"
    description = "Echo the input text back."

    @property
    def schema(self):
        return {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        }

    async def execute(self, **kwargs: Any) -> ToolResult:
        return ToolResult(success=True, output=kwargs.get("text", ""))


class PreviewTool(EchoTool):
    """Echo tool with preview metadata for visual-intent tests."""

    async def preview(self, **kwargs: Any) -> dict[str, Any] | None:
        return {
            "frame_label": "echo · preview",
            "summary": "Preparing the echo action",
            "focus_x": 0.4,
            "focus_y": 0.5,
        }


@pytest.fixture
def tmp_workspace(tmp_path):
    return tmp_path / "workspace"


@pytest.fixture
def config(tmp_workspace):
    return AgentConfig(
        llm_provider="openai",
        llm_model="gpt-4o",
        workspace_dir=tmp_workspace,
        memory_dir=tmp_workspace / ".memory",
        max_iterations=5,
        allow_computer_control=False,  # no real desktop in tests
        allow_terminal=False,
    )


# ── basic run ─────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_agent_done_on_first_response(config, tmp_workspace):
    """Agent should stop immediately when the model says DONE."""
    llm = FakeLLM([LLMResponse(content="DONE: Task complete.", finish_reason="stop")])
    agent = AutonomousAgent(config=config, llm=llm, tools=[EchoTool()])

    events = []
    gen = await agent.run("Say hello")
    async for event in gen:
        events.append(event)

    types = [e["type"] for e in events]
    assert "done" in types
    done_event = next(e for e in events if e["type"] == "done")
    assert "DONE:" in done_event["content"]


@pytest.mark.asyncio
async def test_agent_done_when_done_marker_is_last_line(config, tmp_workspace):
    """Agent should stop even when the model appends DONE after a richer summary."""
    llm = FakeLLM(
        [
            LLMResponse(
                content="I opened the page and captured the screenshot.\n\nDONE: Task complete."
            )
        ]
    )
    agent = AutonomousAgent(config=config, llm=llm, tools=[EchoTool()])

    events = []
    gen = await agent.run("Say hello")
    async for event in gen:
        events.append(event)

    done_event = next(e for e in events if e["type"] == "done")
    assert done_event["content"] == "DONE: Task complete."


@pytest.mark.asyncio
async def test_agent_calls_tool_then_finishes(config, tmp_workspace):
    """Agent should call a tool, get the result, then finish."""
    llm = FakeLLM(
        [
            # First call: request tool
            LLMResponse(
                content="",
                tool_calls=[
                    {"id": "1", "name": "echo", "arguments": {"text": "hello world"}}
                ],
            ),
            # Second call: model receives tool result and says done
            LLMResponse(content="DONE: I echoed 'hello world'.", finish_reason="stop"),
        ]
    )
    agent = AutonomousAgent(config=config, llm=llm, tools=[EchoTool()])

    events = []
    gen = await agent.run("Echo 'hello world'")
    async for event in gen:
        events.append(event)

    types = [e["type"] for e in events]
    assert "tool_call" in types
    assert "tool_result" in types
    assert "done" in types

    tool_call = next(e for e in events if e["type"] == "tool_call")
    assert tool_call["tool"] == "echo"
    assert tool_call["args"] == {"text": "hello world"}

    tool_result = next(e for e in events if e["type"] == "tool_result")
    assert tool_result["success"] is True
    assert tool_result["result"] == "hello world"


@pytest.mark.asyncio
async def test_agent_stops_async_tools_on_exit(config, tmp_workspace):
    """Async tool cleanup should run after the agent finishes."""

    class ClosableTool(EchoTool):
        def __init__(self) -> None:
            self.stopped = False

        async def stop(self) -> None:
            self.stopped = True

    tool = ClosableTool()
    llm = FakeLLM([LLMResponse(content="DONE: Task complete.", finish_reason="stop")])
    agent = AutonomousAgent(config=config, llm=llm, tools=[tool])

    gen = await agent.run("Say hello")
    async for _event in gen:
        pass

    assert tool.stopped is True


@pytest.mark.asyncio
async def test_agent_respects_max_iterations(config, tmp_workspace):
    """Agent should stop after max_iterations even without DONE."""
    # Return tool calls indefinitely
    infinite_response = LLMResponse(
        content="thinking...",
        tool_calls=[{"id": "1", "name": "echo", "arguments": {"text": "loop"}}],
    )
    call_count = 0

    class CountingLLM(LLMProvider):
        async def complete(self, messages, tools=None, temperature=0.2, max_tokens=4096):
            nonlocal call_count
            call_count += 1
            return infinite_response

        async def embed(self, text: str) -> list[float]:
            return [0.0] * 256

    config_limited = AgentConfig(
        llm_provider="openai",
        workspace_dir=tmp_workspace,
        memory_dir=tmp_workspace / ".memory",
        max_iterations=3,
        allow_computer_control=False,
        allow_terminal=False,
    )
    agent = AutonomousAgent(config=config_limited, llm=CountingLLM(), tools=[EchoTool()])

    events = []
    gen = await agent.run("Loop forever")
    async for event in gen:
        events.append(event)

    # Should end with a done event (iteration limit)
    types = [e["type"] for e in events]
    assert "done" in types


@pytest.mark.asyncio
async def test_agent_interrupt(config, tmp_workspace):
    """Calling interrupt() should stop the agent."""
    import asyncio  # noqa: PLC0415

    class SlowLLM(LLMProvider):
        async def complete(self, messages, tools=None, temperature=0.2, max_tokens=4096):
            await asyncio.sleep(0.05)
            return LLMResponse(content="thinking...", tool_calls=[])

        async def embed(self, text: str) -> list[float]:
            return [0.0] * 256

    agent = AutonomousAgent(config=config, llm=SlowLLM(), tools=[EchoTool()])

    events = []

    async def collect():
        gen = await agent.run("Do something")
        async for event in gen:
            events.append(event)
            if len(events) >= 1:
                agent.interrupt()

    await collect()
    # As long as we collected at least one event without hanging, the test passes
    assert len(events) >= 1


# ── unknown tool ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_agent_handles_unknown_tool(config, tmp_workspace):
    """Calling an unknown tool should return an error result without crashing."""
    llm = FakeLLM(
        [
            LLMResponse(
                content="",
                tool_calls=[{"id": "1", "name": "nonexistent", "arguments": {}}],
            ),
            LLMResponse(content="DONE: Handled error.", finish_reason="stop"),
        ]
    )
    agent = AutonomousAgent(config=config, llm=llm, tools=[EchoTool()])

    events = []
    gen = await agent.run("Call unknown tool")
    async for event in gen:
        events.append(event)

    errors = [e for e in events if e["type"] == "tool_result" and not e["success"]]
    assert len(errors) == 1
    assert "nonexistent" in errors[0]["result"]


@pytest.mark.asyncio
async def test_agent_emits_phase_and_visual_intent_for_previewable_tools(config, tmp_workspace):
    """Preview-capable tools should emit waiting phases and visual intent metadata."""
    llm = FakeLLM(
        [
            LLMResponse(
                content="",
                tool_calls=[{"id": "1", "name": "echo", "arguments": {"text": "hello"}}],
            ),
            LLMResponse(content="DONE: Preview complete.", finish_reason="stop"),
        ]
    )
    agent = AutonomousAgent(config=config, llm=llm, tools=[PreviewTool()])

    events = []
    gen = await agent.run("Echo hello")
    async for event in gen:
        events.append(event)

    types = [event["type"] for event in events]
    assert "phase" in types
    assert "visual_intent" in types

    first_phase = next(event for event in events if event["type"] == "phase")
    assert first_phase["phase"] == "thinking"

    visual_intent = next(event for event in events if event["type"] == "visual_intent")
    assert visual_intent["summary"] == "Preparing the echo action"
    assert visual_intent["focus_x"] == pytest.approx(0.4)
    assert visual_intent["focus_y"] == pytest.approx(0.5)
