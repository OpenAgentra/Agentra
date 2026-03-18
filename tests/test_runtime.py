"""Tests for the thread-aware runtime backend."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

import pytest

from agentra.agents.autonomous import AutonomousAgent
from agentra.config import AgentConfig
from agentra.llm.base import LLMProvider, LLMResponse
from agentra.memory.workspace import WorkspaceManager
from agentra.runtime import ExecutionScheduler, ThreadManager
from agentra.tools.base import BaseTool, ToolResult


@dataclass
class RuntimeRequest:
    goal: str
    thread_id: str | None = None
    thread_title: str | None = None
    provider: str | None = None
    model: str | None = None
    headless: bool | None = None
    workspace: str | None = None
    max_iterations: int | None = None


class SlowDoneAgent:
    """Small fake agent used to verify parallel thread execution."""

    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        self.interrupted = False

    async def run(self, goal: str):
        async def generator():
            yield {"type": "phase", "phase": "thinking", "content": "thinking", "summary": "thinking"}
            await asyncio.sleep(0.05)
            if self.interrupted:
                yield {"type": "done", "content": "DONE: interrupted"}
                return
            yield {"type": "done", "content": f"DONE: {goal}"}

        return generator()

    def interrupt(self) -> None:
        self.interrupted = True


class EchoTool(BaseTool):
    name = "echo"
    description = "Echo the input text back."

    @property
    def schema(self):
        return {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        }

    async def execute(self, **kwargs):
        return ToolResult(success=True, output=kwargs.get("text", ""))


class TerminalTool(BaseTool):
    name = "terminal"
    description = "Run a terminal command."

    @property
    def schema(self):
        return {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        }

    async def execute(self, **kwargs):
        return ToolResult(success=True, output=f"ran: {kwargs.get('command', '')}")


class QuestionLLM(LLMProvider):
    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, messages, tools=None, temperature=0.2, max_tokens=4096):
        self.calls += 1
        if self.calls == 1:
            return LLMResponse(content="Hangi dosya adını kullanayım?")
        return LLMResponse(content="DONE: report.md kullanılacak.")

    async def embed(self, text: str) -> list[float]:
        return [0.0] * 256


class ApprovalLLM(LLMProvider):
    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, messages, tools=None, temperature=0.2, max_tokens=4096):
        self.calls += 1
        if self.calls == 1:
            return LLMResponse(
                content="",
                tool_calls=[{"id": "call-1", "name": "terminal", "arguments": {"command": "pip install demo"}}],
            )
        return LLMResponse(content="DONE: terminal action complete.")

    async def embed(self, text: str) -> list[float]:
        return [0.0] * 256


class WorkspaceWritingAgent:
    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        self.workspace = WorkspaceManager(config.workspace_dir)

    async def run(self, goal: str):
        self.workspace.init()
        (self.config.workspace_dir / "artifact.txt").write_text(goal, encoding="utf-8")

        async def generator():
            yield {"type": "done", "content": f"DONE: {goal}"}

        return generator()


async def _wait_for_run(manager: ThreadManager, run_id: str, timeout: float = 3.0):
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        session = manager.get_session(run_id)
        if session.completed.is_set():
            return session
        await asyncio.sleep(0.02)
    raise AssertionError("Run did not complete in time.")


async def _wait_for(predicate, timeout: float = 3.0):
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        value = predicate()
        if value:
            return value
        await asyncio.sleep(0.02)
    raise AssertionError("Condition did not become true in time.")


@pytest.fixture
def base_config(tmp_path: Path) -> AgentConfig:
    return AgentConfig(
        llm_provider="openai",
        workspace_dir=tmp_path / "workspace",
        memory_dir=tmp_path / "workspace" / ".memory",
        allow_terminal=False,
        allow_computer_control=False,
    )


@pytest.mark.asyncio
async def test_thread_manager_supports_parallel_independent_threads(base_config: AgentConfig) -> None:
    manager = ThreadManager(base_config, agent_factory=lambda cfg: SlowDoneAgent(cfg))

    first = await manager.start_run(RuntimeRequest(goal="Goal one"))
    second = await manager.start_run(RuntimeRequest(goal="Goal two"))

    assert first["thread_id"] != second["thread_id"]
    assert len(manager.list_threads_for_http()) == 2

    await _wait_for_run(manager, first["run_id"])
    await _wait_for_run(manager, second["run_id"])


@pytest.mark.asyncio
async def test_execution_scheduler_serializes_computer_capability() -> None:
    scheduler = ExecutionScheduler()
    order: list[str] = []

    async def job(name: str):
        async with scheduler.reserve(("computer",), thread_id=name, tool_name="computer"):
            order.append(f"start:{name}")
            await asyncio.sleep(0.02)
            order.append(f"end:{name}")

    await asyncio.gather(job("a"), job("b"))

    assert order in (
        ["start:a", "end:a", "start:b", "end:b"],
        ["start:b", "end:b", "start:a", "end:a"],
    )


@pytest.mark.asyncio
async def test_thread_manager_unblocks_question_requested_runs(base_config: AgentConfig) -> None:
    def factory(cfg: AgentConfig) -> AutonomousAgent:
        return AutonomousAgent(config=cfg, llm=QuestionLLM(), tools=[EchoTool()])

    manager = ThreadManager(base_config, agent_factory=factory)
    snapshot = await manager.start_run(RuntimeRequest(goal="Bir isim sor"))

    def pending_question():
        thread = manager.get_thread(snapshot["thread_id"])
        return next(iter(thread.question_requests.values()), None)

    request = await _wait_for(pending_question)
    await manager.answer_question(snapshot["thread_id"], request.request_id, "report.md")
    session = await _wait_for_run(manager, snapshot["run_id"])
    final_snapshot = manager.snapshot_for_http(session)

    assert final_snapshot["status"] == "completed"
    assert any(event["type"] == "question_requested" for event in final_snapshot["events"])
    assert any(event["type"] == "user_answer_received" for event in final_snapshot["events"])


@pytest.mark.asyncio
async def test_thread_manager_requires_approval_for_risky_actions(base_config: AgentConfig) -> None:
    def factory(cfg: AgentConfig) -> AutonomousAgent:
        return AutonomousAgent(config=cfg, llm=ApprovalLLM(), tools=[TerminalTool()])

    manager = ThreadManager(base_config, agent_factory=factory)
    snapshot = await manager.start_run(RuntimeRequest(goal="Risky terminal action"))

    def pending_approval():
        thread = manager.get_thread(snapshot["thread_id"])
        return next(iter(thread.approval_requests.values()), None)

    request = await _wait_for(pending_approval)
    await manager.respond_to_approval(snapshot["thread_id"], request.request_id, approved=True, note="approved")
    session = await _wait_for_run(manager, snapshot["run_id"])
    final_snapshot = manager.snapshot_for_http(session)

    assert final_snapshot["status"] == "completed"
    assert any(event["type"] == "approval_requested" for event in final_snapshot["events"])
    assert any(event["type"] == "approved" for event in final_snapshot["events"])
    assert final_snapshot["approval_requests"][0]["rule_id"] == "terminal-install-or-side-effect"
    assert final_snapshot["approval_requests"][0]["risk_level"] == "high"


@pytest.mark.asyncio
async def test_thread_manager_captures_workspace_audit(base_config: AgentConfig) -> None:
    manager = ThreadManager(base_config, agent_factory=lambda cfg: WorkspaceWritingAgent(cfg))

    snapshot = await manager.start_run(RuntimeRequest(goal="write artifact"))
    session = await _wait_for_run(manager, snapshot["run_id"])
    final_snapshot = manager.snapshot_for_http(session)

    entry_types = [entry["entry_type"] for entry in final_snapshot["audit"]]
    assert "workspace_checkpoint" in entry_types
    assert "workspace_diff" in entry_types
