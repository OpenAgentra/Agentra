"""Tests for the AutonomousAgent."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from google.genai.errors import ClientError

from agentra.agents.autonomous import (
    AutonomousAgent,
    _goal_has_local_desktop_component,
    _goal_mentions_web_target,
    _goal_is_browser_only,
)
from agentra.config import AgentConfig
from agentra.llm.base import LLMMessage, LLMProvider, LLMResponse
from agentra.task_routing import (
    goal_has_local_desktop_component as routing_goal_has_local_desktop_component,
    goal_mentions_web_target as routing_goal_mentions_web_target,
)
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


class NamedTool(BaseTool):
    """Simple configurable tool used for routing tests."""

    def __init__(self, name: str, description: str) -> None:
        self._name = name
        self._description = description
        self.calls: list[dict[str, Any]] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def schema(self):
        return {
            "type": "object",
            "properties": {
                "action": {"type": "string"},
                "command": {"type": "string"},
                "url": {"type": "string"},
                "text": {"type": "string"},
            },
        }

    async def execute(self, **kwargs: Any) -> ToolResult:
        self.calls.append(kwargs)
        return ToolResult(success=True, output=f"{self._name} ok")


class ScriptedBrowserTool(BaseTool):
    """Browser tool with deterministic per-call results for takeover tests."""

    name = "browser"
    description = "Web browser control."

    def __init__(self, results: list[ToolResult]) -> None:
        self._results = list(results)
        self.calls: list[dict[str, Any]] = []

    @property
    def schema(self):
        return {
            "type": "object",
            "properties": {
                "action": {"type": "string"},
                "url": {"type": "string"},
                "selector": {"type": "string"},
                "text": {"type": "string"},
            },
        }

    async def execute(self, **kwargs: Any) -> ToolResult:
        self.calls.append(kwargs)
        if self._results:
            return self._results.pop(0)
        return ToolResult(success=True, output="browser ok")


class DesktopConflictTool(BaseTool):
    """Computer tool that forces a desktop takeover before succeeding."""

    name = "computer"
    description = "Desktop control."

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    @property
    def schema(self):
        return {
            "type": "object",
            "properties": {"action": {"type": "string"}, "text": {"type": "string"}},
            "required": ["action"],
        }

    async def execute(self, **kwargs: Any) -> ToolResult:
        self.calls.append(kwargs)
        if kwargs.get("action") == "screenshot":
            return ToolResult(
                success=True,
                output="Desktop refreshed.",
                metadata={"frame_label": "computer · screenshot", "summary": "Desktop refreshed."},
            )
        return ToolResult(
            success=False,
            error="Desktop takeover required.",
            metadata={
                "pause_kind": "desktop_control_takeover",
                "summary": "Masaustu gorunumu Agentra penceresinde kaldigi icin kontrol size devredildi.",
                "content": (
                    "Gorunur masaustu otomasyonu icin hedef pencereyi one getirin; "
                    "hazir olunca Finish Control ile devam edin."
                ),
            },
        )


class WindowsDesktopConflictTool(BaseTool):
    """windows_desktop tool that forces a desktop takeover before succeeding."""

    name = "windows_desktop"
    description = "Native Windows desktop control."

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    @property
    def schema(self):
        return {
            "type": "object",
            "properties": {
                "action": {"type": "string"},
                "app": {"type": "string"},
                "profile_id": {"type": "string"},
            },
            "required": ["action"],
        }

    async def execute(self, **kwargs: Any) -> ToolResult:
        self.calls.append(kwargs)
        return ToolResult(
            success=False,
            error="Desktop takeover required.",
            metadata={
                "pause_kind": "desktop_control_takeover",
                "summary": "Gorunur Windows uygulamasi odagi degistirecegi icin kontrol size devredildi.",
                "content": (
                    "Bu gorunur Windows adimi odagi hedef uygulamaya tasiyacak. "
                    "Hazir oldugunuzda Finish Control ile devam edin."
                ),
            },
        )


class QuotaFailingLLM(LLMProvider):
    """Raise a provider quota error so the agent surfaces a friendly message."""

    async def complete(self, messages, tools=None, temperature=0.2, max_tokens=4096):
        raise ClientError(
            429,
            {
                "error": {
                    "code": 429,
                    "message": (
                        "You exceeded your current quota, please check your plan and billing details."
                    ),
                    "status": "RESOURCE_EXHAUSTED",
                }
            },
            None,
        )

    async def embed(self, text: str) -> list[float]:
        return [0.0] * 256


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
        long_term_memory_dir=tmp_workspace / ".memory-global",
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


@pytest.mark.asyncio
async def test_agent_surfaces_friendly_provider_quota_errors(tmp_workspace):
    config = AgentConfig(
        llm_provider="gemini",
        llm_model="gemini-3-flash-preview",
        workspace_dir=tmp_workspace,
        memory_dir=tmp_workspace / ".memory",
        max_iterations=5,
        allow_computer_control=False,
        allow_terminal=False,
    )
    agent = AutonomousAgent(config=config, llm=QuotaFailingLLM(), tools=[EchoTool()])

    events = []
    gen = await agent.run("Open a repository")
    async for event in gen:
        events.append(event)

    error = next(event for event in events if event["type"] == "error")

    assert error["content"] == (
        "Gemini quota exceeded for model gemini-3-flash-preview. "
        "Add billing or wait for quota reset, then retry."
    )
    assert error["summary"] == (
        "Switch the thread to another provider/model, or add Gemini "
        "billing/credits before retrying."
    )
    assert error["details"]["status_code"] == 429
    assert error["details"]["provider_status"] == "RESOURCE_EXHAUSTED"
    assert error["details"]["error_kind"] == "quota_exceeded"


@pytest.mark.asyncio
async def test_agent_blocks_browser_drift_for_local_desktop_goal(config, tmp_workspace):
    browser = NamedTool("browser", "Web browser control.")
    computer = NamedTool("computer", "Desktop control.")
    llm = FakeLLM(
        [
            LLMResponse(
                content="",
                tool_calls=[
                    {
                        "id": "1",
                        "name": "browser",
                        "arguments": {"action": "navigate", "url": "https://github.com/kagankakao"},
                    }
                ],
            ),
            LLMResponse(
                content="",
                tool_calls=[{"id": "2", "name": "computer", "arguments": {"action": "click"}}],
            ),
            LLMResponse(content='DONE: "Second Sun" folder is now open on the desktop.'),
        ]
    )
    agent = AutonomousAgent(config=config, llm=llm, tools=[browser, computer])

    events = []
    gen = await agent.run("masaüstündeki Second Sun klasörüne gir")
    async for event in gen:
        events.append(event)

    browser_result = next(
        event
        for event in events
        if event["type"] == "tool_result" and event["tool"] == "browser"
    )
    assert browser_result["success"] is False
    assert "desktop/folder task" in browser_result["result"]
    assert browser.calls == []
    assert computer.calls == [{"action": "click"}]


@pytest.mark.asyncio
async def test_agent_blocks_computer_drift_for_browser_only_goal(config, tmp_workspace):
    browser = NamedTool("browser", "Web browser control.")
    computer = NamedTool("computer", "Desktop control.")
    llm = FakeLLM(
        [
            LLMResponse(
                content="",
                tool_calls=[{"id": "1", "name": "computer", "arguments": {"action": "click"}}],
            ),
            LLMResponse(
                content="",
                tool_calls=[
                    {
                        "id": "2",
                        "name": "browser",
                        "arguments": {"action": "navigate", "url": "https://github.com/login"},
                    }
                ],
            ),
            LLMResponse(content="DONE: GitHub login page is open."),
        ]
    )
    agent = AutonomousAgent(config=config, llm=llm, tools=[browser, computer])

    events = []
    gen = await agent.run("github.com/login sayfasini ac")
    async for event in gen:
        events.append(event)

    computer_result = next(
        event
        for event in events
        if event["type"] == "tool_result" and event["tool"] == "computer"
    )
    assert computer_result["success"] is False
    assert "browser-only website task" in computer_result["result"]
    assert computer.calls == []
    assert browser.calls == [{"action": "navigate", "url": "https://github.com/login"}]


@pytest.mark.asyncio
async def test_agent_pauses_instead_of_finishing_when_sensitive_browser_goal_reaches_login_page(
    config,
    tmp_workspace,
):
    browser = ScriptedBrowserTool(
        [
            ToolResult(
                success=True,
                output="Opened GitHub login.",
                metadata={
                    "active_url": "https://github.com/login",
                    "active_title": "Sign in to GitHub · GitHub",
                },
            ),
            ToolResult(
                success=True,
                output="Refreshed after manual login.",
                metadata={
                    "active_url": "https://github.com/",
                    "active_title": "GitHub",
                },
            ),
        ]
    )
    llm = FakeLLM(
        [
            LLMResponse(
                content="",
                tool_calls=[
                    {
                        "id": "1",
                        "name": "browser",
                        "arguments": {"action": "navigate", "url": "https://github.com/login"},
                    }
                ],
            ),
            LLMResponse(content="DONE: GitHub login page is open."),
            LLMResponse(content="DONE: GitHub is ready after manual login."),
        ]
    )
    agent = AutonomousAgent(config=config, llm=llm, tools=[browser])
    agent.take_control = AsyncMock(return_value=None)

    events = []
    gen = await agent.run(
        "Chrome profilimi kullanarak github.com/login sayfasini ac; parola, kod veya "
        "benzeri hassas bir adima gelince kontrolu bana devret, ben manuel tamamlayinca "
        "Finish Control ile devam et; masaustumdeki diger pencere veya uygulamalara dokunma."
    )
    async for event in gen:
        events.append(event)

    paused_index = next(
        index
        for index, event in enumerate(events)
        if event["type"] == "paused" and event.get("pause_kind") == "sensitive_browser_takeover"
    )
    done_index = next(index for index, event in enumerate(events) if event["type"] == "done")
    assert paused_index < done_index
    assert events[paused_index]["takeover_kind"] == "secret"
    assert events[done_index]["content"] == "DONE: GitHub is ready after manual login."
    assert browser.calls == [
        {"action": "navigate", "url": "https://github.com/login"},
        {"action": "screenshot"},
    ]
    agent.take_control.assert_awaited_once()


@pytest.mark.asyncio
async def test_desktop_goal_pauses_for_manual_desktop_takeover(config):
    computer = DesktopConflictTool()
    llm = FakeLLM(
        [
            LLMResponse(
                content="",
                tool_calls=[
                    {
                        "id": "1",
                        "name": "computer",
                        "arguments": {"action": "type", "text": "Agentra desktop test"},
                    }
                ],
            ),
            LLMResponse(content="DONE: Desktop is ready after manual stabilization."),
        ]
    )
    agent = AutonomousAgent(config=config, llm=llm, tools=[computer])
    agent.take_control = AsyncMock(return_value=None)

    events = []
    gen = await agent.run("Masaustunde gorunur bir alana metin yaz.")
    async for event in gen:
        events.append(event)

    paused_index = next(
        index
        for index, event in enumerate(events)
        if event["type"] == "paused" and event.get("pause_kind") == "desktop_control_takeover"
    )
    done_index = next(index for index, event in enumerate(events) if event["type"] == "done")
    assert paused_index < done_index
    assert computer.calls == [
        {"action": "type", "text": "Agentra desktop test"},
        {"action": "screenshot"},
    ]
    agent.take_control.assert_awaited_once()


@pytest.mark.asyncio
async def test_windows_desktop_goal_pauses_for_manual_desktop_takeover(config):
    computer = NamedTool("computer", "Desktop control.")
    windows_desktop = WindowsDesktopConflictTool()
    llm = FakeLLM(
        [
            LLMResponse(
                content="",
                tool_calls=[
                    {
                        "id": "1",
                        "name": "windows_desktop",
                        "arguments": {"action": "launch_app", "app": "Notepad"},
                    }
                ],
            ),
            LLMResponse(content="DONE: Native desktop step is ready after manual stabilization."),
        ]
    )
    agent = AutonomousAgent(config=config, llm=llm, tools=[computer, windows_desktop])
    agent.take_control = AsyncMock(return_value=None)

    events = []
    gen = await agent.run("Windows Not Defteri'ni ac ve metni yaz.")
    async for event in gen:
        events.append(event)

    paused_event = next(
        event
        for event in events
        if event["type"] == "paused" and event.get("pause_kind") == "desktop_control_takeover"
    )
    paused_index = events.index(paused_event)
    done_index = next(index for index, event in enumerate(events) if event["type"] == "done")
    assert paused_event["tool"] == "windows_desktop"
    assert paused_index < done_index
    assert windows_desktop.calls == [{"action": "launch_app", "app": "Notepad"}]
    assert computer.calls == [{"action": "screenshot"}]
    agent.take_control.assert_awaited_once()


def test_browser_goal_with_desktop_guardrail_stays_browser_only():
    goal = (
        "Chrome profilimi kullanarak github.com/login sayfasini ac; "
        "masaustumdeki diger pencere veya uygulamalara dokunma."
    )

    assert _goal_has_local_desktop_component(goal) is False
    assert _goal_is_browser_only(goal) is True
    assert routing_goal_has_local_desktop_component(goal) is False


def test_real_local_desktop_step_survives_guardrail_filter():
    goal = (
        "github.com/login sayfasini ac, sonra masaustumdeki RDR2 kisayolunu ac, "
        "ama diger pencere veya uygulamalara dokunma."
    )

    assert _goal_has_local_desktop_component(goal) is True
    assert routing_goal_has_local_desktop_component(goal) is True


def test_browser_negation_does_not_turn_desktop_goal_into_web_goal():
    goal = (
        "Masaustu katmaninda calis: Windows Hesap Makinesi'ni ac ve 482 + 157 islemini yap. "
        "Tarayici kullanma ve diger pencere veya uygulamalara dokunma."
    )

    assert _goal_mentions_web_target(goal) is False
    assert routing_goal_mentions_web_target(goal) is False
    assert _goal_has_local_desktop_component(goal) is True
    assert routing_goal_has_local_desktop_component(goal) is True


@pytest.mark.asyncio
async def test_agent_requires_computer_before_done_for_local_desktop_goal(config, tmp_workspace):
    terminal = NamedTool("terminal", "Terminal access.")
    computer = NamedTool("computer", "Desktop control.")
    llm = FakeLLM(
        [
            LLMResponse(
                content="",
                tool_calls=[
                    {
                        "id": "1",
                        "name": "terminal",
                        "arguments": {"command": 'ls "/mnt/c/Users/ariba/OneDrive/Desktop/Second Sun"'},
                    }
                ],
            ),
            LLMResponse(content='DONE: "Second Sun" folder was listed in the terminal.'),
            LLMResponse(
                content="",
                tool_calls=[{"id": "2", "name": "computer", "arguments": {"action": "double_click"}}],
            ),
            LLMResponse(content='DONE: "Second Sun" folder is now open on screen.'),
        ]
    )
    agent = AutonomousAgent(config=config, llm=llm, tools=[terminal, computer])

    events = []
    gen = await agent.run("masaüstündeki Second Sun klasörüne gir")
    async for event in gen:
        events.append(event)

    done_events = [event for event in events if event["type"] == "done"]
    assert len(done_events) == 1
    assert done_events[0]["content"] == 'DONE: "Second Sun" folder is now open on screen.'
    assert terminal.calls == [{'command': 'ls "/mnt/c/Users/ariba/OneDrive/Desktop/Second Sun"'}]
    assert computer.calls == [{"action": "double_click"}]


@pytest.mark.asyncio
async def test_agent_requires_listing_evidence_before_done_for_local_desktop_contents_goal(
    config, tmp_workspace
):
    computer = NamedTool("computer", "Desktop control.")
    filesystem = NamedTool("filesystem", "Filesystem access.")
    llm = FakeLLM(
        [
            LLMResponse(
                content="",
                tool_calls=[{"id": "1", "name": "computer", "arguments": {"action": "double_click"}}],
            ),
            LLMResponse(content='DONE: "Second Sun" folder is open and its contents are listed.'),
            LLMResponse(
                content="",
                tool_calls=[
                    {
                        "id": "2",
                        "name": "filesystem",
                        "arguments": {"action": "list", "path": "/mnt/c/Users/ariba/Desktop/Second Sun"},
                    }
                ],
            ),
            LLMResponse(content='DONE: "Second Sun" folder is open and its contents are verified.'),
        ]
    )
    agent = AutonomousAgent(config=config, llm=llm, tools=[computer, filesystem])

    events = []
    gen = await agent.run("masaüstüme git ve Second Sun klasörünü aç ve içindekileri bana söyle")
    async for event in gen:
        events.append(event)

    done_events = [event for event in events if event["type"] == "done"]
    assert len(done_events) == 1
    assert done_events[0]["content"] == 'DONE: "Second Sun" folder is open and its contents are verified.'
    assert computer.calls == [{"action": "double_click"}]
    assert filesystem.calls == [{"action": "list", "path": "/mnt/c/Users/ariba/Desktop/Second Sun"}]


@pytest.mark.asyncio
async def test_agent_blocks_repeated_same_desktop_click_guessing(config, tmp_workspace):
    computer = NamedTool("computer", "Desktop control.")
    llm = FakeLLM(
        [
            LLMResponse(
                content="",
                tool_calls=[
                    {
                        "id": "1",
                        "name": "computer",
                        "arguments": {"action": "double_click", "x": 920, "y": 30},
                    }
                ],
            ),
            LLMResponse(
                content="",
                tool_calls=[
                    {
                        "id": "2",
                        "name": "computer",
                        "arguments": {"action": "double_click", "x": 920, "y": 30},
                    }
                ],
            ),
            LLMResponse(
                content="",
                tool_calls=[
                    {
                        "id": "3",
                        "name": "computer",
                        "arguments": {"action": "double_click", "x": 921, "y": 31},
                    }
                ],
            ),
            LLMResponse(
                content="",
                tool_calls=[{"id": "4", "name": "computer", "arguments": {"action": "screenshot"}}],
            ),
            LLMResponse(content='DONE: "Second Sun" folder is now open on screen.'),
        ]
    )
    agent = AutonomousAgent(config=config, llm=llm, tools=[computer])

    events = []
    gen = await agent.run("masaüstündeki Second Sun klasörüne gir")
    async for event in gen:
        events.append(event)

    assert any(
        event["type"] == "tool_result"
        and event["tool"] == "computer"
        and event["success"] is False
        and "Stop guessing coordinates" in event["result"]
        for event in events
    )
    assert computer.calls == [
        {"action": "double_click", "x": 920, "y": 30},
        {"action": "double_click", "x": 920, "y": 30},
        {"action": "screenshot"},
    ]


@pytest.mark.asyncio
async def test_agent_blocks_excessive_desktop_click_guessing_across_different_points(
    config, tmp_workspace
):
    config.max_iterations = 10
    computer = NamedTool("computer", "Desktop control.")
    filesystem = NamedTool("filesystem", "Filesystem access.")
    llm = FakeLLM(
        [
            LLMResponse(
                content="",
                tool_calls=[{"id": "1", "name": "computer", "arguments": {"action": "double_click", "x": 925, "y": 30}}],
            ),
            LLMResponse(
                content="",
                tool_calls=[{"id": "2", "name": "computer", "arguments": {"action": "double_click", "x": 1800, "y": 40}}],
            ),
            LLMResponse(
                content="",
                tool_calls=[{"id": "3", "name": "computer", "arguments": {"action": "double_click", "x": 1820, "y": 35}}],
            ),
            LLMResponse(
                content="",
                tool_calls=[{"id": "4", "name": "computer", "arguments": {"action": "double_click", "x": 500, "y": 590}}],
            ),
            LLMResponse(
                content="",
                tool_calls=[{"id": "5", "name": "computer", "arguments": {"action": "double_click", "x": 530, "y": 335}}],
            ),
            LLMResponse(
                content="",
                tool_calls=[
                    {
                        "id": "6",
                        "name": "filesystem",
                        "arguments": {"action": "list", "path": "/mnt/c/Users/ariba/OneDrive/Desktop/secondsun"},
                    }
                ],
            ),
            LLMResponse(content='DONE: "Second Sun" folder path is confirmed.'),
        ]
    )
    agent = AutonomousAgent(config=config, llm=llm, tools=[computer, filesystem])

    events = []
    gen = await agent.run("masaüstüne git ve oradan secondsun klasörüne gir")
    async for event in gen:
        events.append(event)

    assert any(
        event["type"] == "tool_result"
        and event["tool"] == "computer"
        and event["success"] is False
        and "Stop brute-forcing the UI" in event["result"]
        for event in events
    )
    assert computer.calls == [
        {"action": "double_click", "x": 925, "y": 30},
        {"action": "double_click", "x": 1800, "y": 40},
        {"action": "double_click", "x": 1820, "y": 35},
        {"action": "double_click", "x": 500, "y": 590},
    ]
    assert filesystem.calls == [{"action": "list", "path": "/mnt/c/Users/ariba/OneDrive/Desktop/secondsun"}]


@pytest.mark.asyncio
async def test_agent_blocks_windows_paths_in_terminal_for_local_desktop_goals(config, tmp_workspace):
    terminal = NamedTool("terminal", "Terminal access.")
    filesystem = NamedTool("filesystem", "Filesystem access.")
    llm = FakeLLM(
        [
            LLMResponse(
                content="",
                tool_calls=[
                    {
                        "id": "1",
                        "name": "terminal",
                        "arguments": {"command": 'ls "C:\\\\Users\\\\ariba\\\\Desktop"'},
                    }
                ],
            ),
            LLMResponse(
                content="",
                tool_calls=[
                    {
                        "id": "2",
                        "name": "filesystem",
                        "arguments": {"action": "list", "path": "/mnt/c/Users/ariba/Desktop"},
                    }
                ],
            ),
            LLMResponse(
                content="",
                tool_calls=[{"id": "3", "name": "computer", "arguments": {"action": "double_click"}}],
            ),
            LLMResponse(content='DONE: Desktop path confirmed and folder opened on screen.'),
        ]
    )
    computer = NamedTool("computer", "Desktop control.")
    agent = AutonomousAgent(config=config, llm=llm, tools=[terminal, filesystem, computer])

    events = []
    gen = await agent.run("masaüstüne git ve oradan secondsun klasörüne gir")
    async for event in gen:
        events.append(event)

    terminal_failures = [
        event for event in events if event["type"] == "tool_result" and event["tool"] == "terminal" and not event["success"]
    ]
    assert terminal_failures
    assert "WSL/Linux path space" in terminal_failures[0]["result"]
    assert terminal.calls == []
    assert filesystem.calls == [{"action": "list", "path": "/mnt/c/Users/ariba/Desktop"}]


@pytest.mark.asyncio
async def test_agent_requires_path_resolution_before_done_for_local_document_open_goal(
    config, tmp_workspace
):
    computer = NamedTool("computer", "Desktop control.")
    filesystem = NamedTool("filesystem", "Filesystem access.")
    llm = FakeLLM(
        [
            LLMResponse(
                content="",
                tool_calls=[{"id": "1", "name": "computer", "arguments": {"action": "double_click"}}],
            ),
            LLMResponse(content="DONE: PowerPoint presentation is open."),
            LLMResponse(
                content="",
                tool_calls=[
                    {
                        "id": "2",
                        "name": "filesystem",
                        "arguments": {"action": "list", "path": "/mnt/c/Users/ariba/OneDrive/Desktop/Second Sun"},
                    }
                ],
            ),
            LLMResponse(content="DONE: PowerPoint presentation is open."),
        ]
    )
    agent = AutonomousAgent(config=config, llm=llm, tools=[computer, filesystem])

    events = []
    gen = await agent.run("masaüstüne git ve oradan secondsun klasörüne gir ve karşına çıkan powerpoint sunumunu aç")
    async for event in gen:
        events.append(event)

    done_events = [event for event in events if event["type"] == "done"]
    assert len(done_events) == 1
    assert filesystem.calls == [{"action": "list", "path": "/mnt/c/Users/ariba/OneDrive/Desktop/Second Sun"}]


@pytest.mark.asyncio
async def test_agent_prefers_local_system_for_under_the_hood_document_goals(tmp_workspace):
    config = AgentConfig(
        llm_provider="openai",
        llm_model="gpt-4o",
        workspace_dir=tmp_workspace,
        memory_dir=tmp_workspace / ".memory",
        max_iterations=6,
        local_execution_mode="under_the_hood",
        desktop_fallback_policy="pause_and_ask",
        allow_computer_control=False,
        allow_terminal=False,
    )
    computer = NamedTool("computer", "Desktop control.")
    filesystem = NamedTool("filesystem", "Filesystem access.")
    local_system = NamedTool("local_system", "Native local operations.")
    llm = FakeLLM(
        [
            LLMResponse(
                content="",
                tool_calls=[{"id": "1", "name": "computer", "arguments": {"action": "double_click"}}],
            ),
            LLMResponse(
                content="",
                tool_calls=[
                    {
                        "id": "2",
                        "name": "local_system",
                        "arguments": {"action": "resolve_known_folder", "folder_key": "desktop"},
                    }
                ],
            ),
            LLMResponse(
                content="",
                tool_calls=[
                    {
                        "id": "3",
                        "name": "filesystem",
                        "arguments": {"action": "list", "path": "/mnt/c/Users/ariba/OneDrive/Desktop/secondsun"},
                    }
                ],
            ),
            LLMResponse(
                content="",
                tool_calls=[
                    {
                        "id": "4",
                        "name": "local_system",
                        "arguments": {
                            "action": "open_path",
                            "path": "/mnt/c/Users/ariba/OneDrive/Desktop/secondsun/deck.pptx",
                        },
                    }
                ],
            ),
            LLMResponse(content="DONE: PowerPoint presentation is open."),
        ]
    )
    agent = AutonomousAgent(config=config, llm=llm, tools=[computer, filesystem, local_system])

    events = []
    gen = await agent.run(
        "masaüstüne git ve oradan secondsun klasörüne gir ve karşına çıkan powerpoint sunumunu aç"
    )
    async for event in gen:
        events.append(event)

    computer_failures = [
        event
        for event in events
        if event["type"] == "tool_result" and event["tool"] == "computer" and not event["success"]
    ]
    assert computer_failures
    assert "visible desktop automation is disabled" in computer_failures[0]["result"]
    assert computer.calls == []
    assert filesystem.calls == [
        {"action": "list", "path": "/mnt/c/Users/ariba/OneDrive/Desktop/secondsun"}
    ]


@pytest.mark.asyncio
async def test_agent_prefers_windows_desktop_before_visible_desktop_for_standard_windows_app_goal(
    tmp_workspace,
):
    class NativeWindowsTool(NamedTool):
        async def execute(self, **kwargs: Any) -> ToolResult:
            self.calls.append(kwargs)
            action = str(kwargs.get("action", "")).lower()
            metadata = {"verified": action == "read_status", "frame_label": f"windows_desktop · {action}"}
            output = "639" if action == "read_status" else "windows_desktop ok"
            return ToolResult(success=True, output=output, metadata=metadata)

    config = AgentConfig(
        llm_provider="openai",
        llm_model="gpt-4o",
        workspace_dir=tmp_workspace,
        memory_dir=tmp_workspace / ".memory",
        max_iterations=8,
        local_execution_mode="native",
        desktop_execution_mode="desktop_native",
        desktop_backend_preference="native",
        desktop_fallback_policy="visible_control",
        allow_computer_control=True,
        allow_terminal=False,
    )
    computer = NamedTool("computer", "Desktop control.")
    windows_desktop = NativeWindowsTool("windows_desktop", "Native Windows desktop control.")
    llm = FakeLLM(
        [
            LLMResponse(
                content="",
                tool_calls=[{"id": "1", "name": "computer", "arguments": {"action": "double_click"}}],
            ),
            LLMResponse(
                content="",
                tool_calls=[{"id": "2", "name": "windows_desktop", "arguments": {"action": "launch_app", "app": "Calculator"}}],
            ),
            LLMResponse(content="DONE: Calculator is open."),
            LLMResponse(
                content="",
                tool_calls=[
                    {
                        "id": "3",
                        "name": "windows_desktop",
                        "arguments": {"action": "read_status", "app": "Calculator", "expected_text": "639"},
                    }
                ],
            ),
            LLMResponse(content="DONE: Calculator result 639 verified."),
        ]
    )
    agent = AutonomousAgent(config=config, llm=llm, tools=[computer, windows_desktop])

    events = []
    gen = await agent.run("Windows Hesap Makinesi'ni ac ve 482 + 157 islemini yap.")
    async for event in gen:
        events.append(event)

    done_events = [event for event in events if event["type"] == "done"]
    assert len(done_events) == 1
    assert done_events[0]["content"] == "DONE: Calculator result 639 verified."
    assert computer.calls == []
    assert windows_desktop.calls == [
        {"action": "launch_app", "app": "Calculator"},
        {"action": "read_status", "app": "Calculator", "expected_text": "639"},
    ]


@pytest.mark.asyncio
async def test_agent_blocks_local_system_launch_app_before_windows_desktop_for_standard_windows_app_goal(
    tmp_workspace,
):
    class NativeWindowsTool(NamedTool):
        async def execute(self, **kwargs: Any) -> ToolResult:
            self.calls.append(kwargs)
            return ToolResult(
                success=True,
                output="windows_desktop ok",
                metadata={"verified": True, "frame_label": "windows_desktop · launch_app"},
            )

    config = AgentConfig(
        llm_provider="openai",
        llm_model="gpt-4o",
        workspace_dir=tmp_workspace,
        memory_dir=tmp_workspace / ".memory",
        max_iterations=6,
        local_execution_mode="native",
        desktop_execution_mode="desktop_native",
        desktop_backend_preference="native",
        desktop_fallback_policy="visible_control",
        allow_computer_control=True,
        allow_terminal=False,
    )
    local_system = NamedTool("local_system", "Hidden local system tool.")
    windows_desktop = NativeWindowsTool("windows_desktop", "Native Windows desktop control.")
    llm = FakeLLM(
        [
            LLMResponse(
                content="",
                tool_calls=[
                    {
                        "id": "1",
                        "name": "local_system",
                        "arguments": {"action": "launch_app", "app": "Calculator"},
                    }
                ],
            ),
            LLMResponse(
                content="",
                tool_calls=[
                    {
                        "id": "2",
                        "name": "windows_desktop",
                        "arguments": {"action": "launch_app", "app": "Calculator"},
                    }
                ],
            ),
            LLMResponse(content="DONE: Calculator opened with windows_desktop."),
        ]
    )
    agent = AutonomousAgent(config=config, llm=llm, tools=[local_system, windows_desktop])

    events = []
    gen = await agent.run("Windows Hesap Makinesi'ni ac ve 482 + 157 islemini yap.")
    async for event in gen:
        events.append(event)

    local_system_failures = [
        event
        for event in events
        if event["type"] == "tool_result" and event["tool"] == "local_system" and not event["success"]
    ]
    assert local_system_failures
    assert "Do not use `local_system launch_app` as the primary path" in local_system_failures[0]["result"]
    assert local_system.calls == []
    assert windows_desktop.calls == [{"action": "launch_app", "app": "Calculator"}]


@pytest.mark.asyncio
async def test_agent_blocks_duplicate_windows_desktop_launch_app_for_same_target(tmp_workspace):
    class NativeWindowsTool(NamedTool):
        async def execute(self, **kwargs: Any) -> ToolResult:
            self.calls.append(kwargs)
            action = str(kwargs.get("action", "")).lower()
            metadata = {"frame_label": f"windows_desktop · {action}", "verified": action == "read_window_text"}
            output = "Untitled - Notepad" if action == "read_window_text" else "windows_desktop ok"
            return ToolResult(success=True, output=output, metadata=metadata)

    config = AgentConfig(
        llm_provider="openai",
        llm_model="gpt-4o",
        workspace_dir=tmp_workspace,
        memory_dir=tmp_workspace / ".memory",
        max_iterations=8,
        local_execution_mode="native",
        desktop_execution_mode="desktop_native",
        desktop_backend_preference="native",
        desktop_fallback_policy="visible_control",
        allow_computer_control=True,
        allow_terminal=False,
    )
    windows_desktop = NativeWindowsTool("windows_desktop", "Native Windows desktop control.")
    llm = FakeLLM(
        [
            LLMResponse(
                content="",
                tool_calls=[{"id": "1", "name": "windows_desktop", "arguments": {"action": "launch_app", "profile_id": "notepad"}}],
            ),
            LLMResponse(
                content="",
                tool_calls=[{"id": "2", "name": "windows_desktop", "arguments": {"action": "launch_app", "app": "notepad.exe"}}],
            ),
            LLMResponse(
                content="",
                tool_calls=[{"id": "3", "name": "windows_desktop", "arguments": {"action": "read_window_text", "profile_id": "notepad"}}],
            ),
            LLMResponse(content="DONE: Reused the existing Notepad window."),
        ]
    )
    agent = AutonomousAgent(config=config, llm=llm, tools=[windows_desktop])

    events = []
    gen = await agent.run("Windows'ta Not Defteri'ni ac ve sonucu dogrula.")
    async for event in gen:
        events.append(event)

    duplicate_failures = [
        event
        for event in events
        if event["type"] == "tool_result" and event["tool"] == "windows_desktop" and not event["success"]
    ]
    assert duplicate_failures
    assert "You already launched this Windows app in this run." in duplicate_failures[0]["result"]
    assert windows_desktop.calls == [
        {"action": "launch_app", "profile_id": "notepad"},
        {"action": "read_window_text", "profile_id": "notepad"},
    ]


@pytest.mark.asyncio
async def test_agent_requires_local_completion_after_browser_for_mixed_under_the_hood_goal(
    tmp_workspace,
):
    config = AgentConfig(
        llm_provider="openai",
        llm_model="gpt-4o",
        workspace_dir=tmp_workspace,
        memory_dir=tmp_workspace / ".memory",
        max_iterations=8,
        local_execution_mode="under_the_hood",
        desktop_fallback_policy="pause_and_ask",
        allow_computer_control=False,
        allow_terminal=False,
    )
    browser = NamedTool("browser", "Web browser control.")
    filesystem = NamedTool("filesystem", "Filesystem access.")
    local_system = NamedTool("local_system", "Native local operations.")
    llm = FakeLLM(
        [
            LLMResponse(
                content="",
                tool_calls=[
                    {
                        "id": "1",
                        "name": "browser",
                        "arguments": {"action": "navigate", "url": "https://github.com/dhatfieldai/agentra"},
                    }
                ],
            ),
            LLMResponse(content="DONE: Agentra GitHub repo is open."),
            LLMResponse(
                content="",
                tool_calls=[
                    {
                        "id": "2",
                        "name": "local_system",
                        "arguments": {"action": "resolve_known_folder", "folder_key": "desktop"},
                    }
                ],
            ),
            LLMResponse(
                content="",
                tool_calls=[
                    {
                        "id": "3",
                        "name": "filesystem",
                        "arguments": {"action": "list", "path": "/mnt/c/Users/ariba/OneDrive/Desktop/secondsun"},
                    }
                ],
            ),
            LLMResponse(
                content="",
                tool_calls=[
                    {
                        "id": "4",
                        "name": "local_system",
                        "arguments": {
                            "action": "open_path",
                            "path": "/mnt/c/Users/ariba/OneDrive/Desktop/secondsun/deck.pptx",
                        },
                    }
                ],
            ),
            LLMResponse(content="DONE: Agentra GitHub repo is open and the PowerPoint presentation is open."),
        ]
    )
    agent = AutonomousAgent(config=config, llm=llm, tools=[browser, filesystem, local_system])

    events = []
    gen = await agent.run(
        "Tarayicida google.com'u ac, Agentra GitHub reposunu bul, sonra masaustumdeki "
        "secondsun klasorunu bulup icindeki PowerPoint dosyasini varsayilan uygulamayla ac."
    )
    async for event in gen:
        events.append(event)

    done_events = [event for event in events if event["type"] == "done"]
    assert len(done_events) == 1
    assert "PowerPoint presentation is open" in done_events[0]["content"]
    assert browser.calls == [{"action": "navigate", "url": "https://github.com/dhatfieldai/agentra"}]
    assert local_system.calls == [
        {"action": "resolve_known_folder", "folder_key": "desktop"},
        {"action": "open_path", "path": "/mnt/c/Users/ariba/OneDrive/Desktop/secondsun/deck.pptx"},
    ]
    assert filesystem.calls == [
        {"action": "list", "path": "/mnt/c/Users/ariba/OneDrive/Desktop/secondsun"}
    ]


@pytest.mark.asyncio
async def test_agent_blocks_terminal_navigation_for_under_the_hood_local_goals(tmp_workspace):
    config = AgentConfig(
        llm_provider="openai",
        llm_model="gpt-4o",
        workspace_dir=tmp_workspace,
        memory_dir=tmp_workspace / ".memory",
        max_iterations=6,
        local_execution_mode="under_the_hood",
        desktop_fallback_policy="pause_and_ask",
        allow_computer_control=False,
        allow_terminal=False,
    )
    terminal = NamedTool("terminal", "Terminal access.")
    filesystem = NamedTool("filesystem", "Filesystem access.")
    local_system = NamedTool("local_system", "Native local operations.")
    llm = FakeLLM(
        [
            LLMResponse(
                content="",
                tool_calls=[
                    {
                        "id": "1",
                        "name": "terminal",
                        "arguments": {"command": "ls /mnt/c/Users/ariba/OneDrive/Desktop"},
                    }
                ],
            ),
            LLMResponse(
                content="",
                tool_calls=[
                    {
                        "id": "2",
                        "name": "local_system",
                        "arguments": {"action": "resolve_known_folder", "folder_key": "desktop"},
                    }
                ],
            ),
            LLMResponse(
                content="",
                tool_calls=[
                    {
                        "id": "3",
                        "name": "filesystem",
                        "arguments": {"action": "list", "path": "/mnt/c/Users/ariba/OneDrive/Desktop/secondsun"},
                    }
                ],
            ),
            LLMResponse(content="DONE: Folder contents are verified."),
        ]
    )
    agent = AutonomousAgent(config=config, llm=llm, tools=[terminal, filesystem, local_system])

    events = []
    gen = await agent.run("masaüstüne git ve oradan secondsun klasörüne gir ve içindekileri bana söyle")
    async for event in gen:
        events.append(event)

    terminal_failures = [
        event
        for event in events
        if event["type"] == "tool_result" and event["tool"] == "terminal" and not event["success"]
    ]
    assert terminal_failures
    assert "prefer `filesystem` for path discovery" in terminal_failures[0]["result"]
    assert terminal.calls == []
    assert local_system.calls == [{"action": "resolve_known_folder", "folder_key": "desktop"}]
    assert filesystem.calls == [
        {"action": "list", "path": "/mnt/c/Users/ariba/OneDrive/Desktop/secondsun"}
    ]


@pytest.mark.asyncio
async def test_long_term_memory_is_skipped_for_local_desktop_goals(config, tmp_workspace):
    agent = AutonomousAgent(config=config, llm=FakeLLM([]), tools=[EchoTool()])
    agent._run_id = "run-current"
    await agent.long_term_memory.add(
        "Navigated to https://github.com/kagankakao",
        metadata={"run_id": "run-older", "summary": "Opening github.com/kagankakao"},
    )

    memory_message = await agent._long_term_memory_message("masaüstündeki Second Sun klasörüne gir")

    assert memory_message is None


@pytest.mark.asyncio
async def test_long_term_memory_requires_goal_overlap(config, tmp_workspace):
    agent = AutonomousAgent(config=config, llm=FakeLLM([]), tools=[EchoTool()])
    agent._run_id = "run-current"
    await agent.long_term_memory.add(
        "Navigated to https://github.com/kagankakao",
        metadata={"run_id": "run-older", "summary": "Opening github.com/kagankakao"},
    )
    await agent.long_term_memory.add(
        "Navigated to https://example.com/docs",
        metadata={"run_id": "run-older-2", "summary": "Opening example.com/docs"},
    )

    memory_message = await agent._long_term_memory_message("open example.com")

    assert memory_message is not None
    assert "example.com" in memory_message.content
    assert "github.com/kagankakao" not in memory_message.content
