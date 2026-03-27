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
    permission_mode: str | None = None
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


class BrowserTool(BaseTool):
    name = "browser"
    description = "Run a browser action."

    def __init__(self) -> None:
        self.calls: list[dict[str, str]] = []

    @property
    def schema(self):
        return {
            "type": "object",
            "properties": {
                "action": {"type": "string"},
                "selector": {"type": "string"},
            },
            "required": ["action"],
        }

    async def execute(self, **kwargs):
        self.calls.append(kwargs)
        return ToolResult(success=True, output=f"browser:{kwargs.get('action', '')}")


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


class BrowserApprovalLLM(LLMProvider):
    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, messages, tools=None, temperature=0.2, max_tokens=4096):
        self.calls += 1
        if self.calls == 1:
            return LLMResponse(
                content="",
                tool_calls=[{"id": "call-1", "name": "browser", "arguments": {"action": "click", "selector": "button.login"}}],
            )
        return LLMResponse(content="DONE: browser action complete.")

    async def embed(self, text: str) -> list[float]:
        return [0.0] * 256


class BrowserSecretTypeLLM(LLMProvider):
    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, messages, tools=None, temperature=0.2, max_tokens=4096):
        self.calls += 1
        if self.calls == 1:
            return LLMResponse(
                content="",
                tool_calls=[
                    {
                        "id": "call-1",
                        "name": "browser",
                        "arguments": {
                            "action": "type",
                            "selector": "input[name='password']",
                            "text": "",
                        },
                    }
                ],
            )
        return LLMResponse(content="DONE: browser secret step complete.")

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


class HangingAgent:
    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        self.interrupted = False

    async def run(self, goal: str):
        async def generator():
            yield {
                "type": "phase",
                "phase": "thinking",
                "content": "waiting",
                "summary": "waiting",
            }
            while not self.interrupted:
                await asyncio.sleep(0.05)
            yield {"type": "done", "content": f"DONE: {goal}"}

        return generator()

    def interrupt(self) -> None:
        self.interrupted = True


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
    await _wait_for_run(manager, first["run_id"])
    await _wait_for_run(manager, second["run_id"])


@pytest.mark.asyncio
async def test_thread_manager_skips_browser_warmup_for_desktop_only_full_mode_goal(
    base_config: AgentConfig,
    monkeypatch,
) -> None:
    manager = ThreadManager(base_config, agent_factory=lambda cfg: SlowDoneAgent(cfg))
    warmup_calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        manager.browser_sessions,
        "warmup_thread",
        lambda **kwargs: warmup_calls.append(kwargs),
    )

    snapshot = await manager.start_run(
        RuntimeRequest(
            goal="Masaustu katmaninda calis ve Not Defteri'ni ac.",
            permission_mode="full",
        )
    )
    await _wait_for_run(manager, snapshot["run_id"])

    assert warmup_calls == []


@pytest.mark.asyncio
async def test_thread_manager_skips_browser_warmup_when_goal_says_not_to_use_browser(
    base_config: AgentConfig,
    monkeypatch,
) -> None:
    manager = ThreadManager(base_config, agent_factory=lambda cfg: SlowDoneAgent(cfg))
    warmup_calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        manager.browser_sessions,
        "warmup_thread",
        lambda **kwargs: warmup_calls.append(kwargs),
    )

    snapshot = await manager.start_run(
        RuntimeRequest(
            goal=(
                "Masaustu katmaninda calis: Windows Hesap Makinesi'ni ac ve 482 + 157 islemini yap. "
                "Tarayici kullanma ve diger pencere veya uygulamalara dokunma."
            ),
            permission_mode="full",
        )
    )
    await _wait_for_run(manager, snapshot["run_id"])

    assert warmup_calls == []


@pytest.mark.asyncio
async def test_thread_manager_chooses_native_desktop_policy_for_standard_windows_app_goal(
    base_config: AgentConfig,
) -> None:
    manager = ThreadManager(base_config, agent_factory=lambda cfg: SlowDoneAgent(cfg))

    snapshot = await manager.start_run(
        RuntimeRequest(
            goal=(
                "Masaustu katmaninda calis: Windows Hesap Makinesi'ni ac ve 482 + 157 "
                "islemini yap. Tarayici kullanma."
            ),
            permission_mode="full",
        )
    )
    session = await _wait_for_run(manager, snapshot["run_id"])

    assert session.config.local_execution_mode == "native"
    assert session.config.desktop_execution_mode == "desktop_native"
    assert session.config.desktop_backend_preference == "native"
    thread = manager.get_thread(snapshot["thread_id"])
    run_started_entries = [
        entry for entry in thread.ledger.entries(run_id=snapshot["run_id"])
        if entry.get("entry_type") == "run_started"
    ]
    assert run_started_entries
    details = run_started_entries[-1]["details"]
    assert details["local_execution_mode"] == "native"
    assert details["desktop_execution_mode"] == "desktop_native"
    assert details["desktop_backend_preference"] == "native"


@pytest.mark.asyncio
async def test_thread_manager_keeps_browser_warmup_for_web_full_mode_goal(
    base_config: AgentConfig,
    monkeypatch,
) -> None:
    manager = ThreadManager(base_config, agent_factory=lambda cfg: SlowDoneAgent(cfg))
    warmup_calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        manager.browser_sessions,
        "warmup_thread",
        lambda **kwargs: warmup_calls.append(kwargs),
    )

    snapshot = await manager.start_run(
        RuntimeRequest(
            goal="https://example.com sayfasini ac ve incele.",
            permission_mode="full",
        )
    )
    await _wait_for_run(manager, snapshot["run_id"])

    assert len(warmup_calls) == 1
    assert warmup_calls[0]["identity"] == "chrome_profile"


@pytest.mark.asyncio
async def test_thread_manager_restores_threads_and_marks_in_progress_runs_for_restart(
    base_config: AgentConfig,
) -> None:
    manager = ThreadManager(base_config, agent_factory=lambda cfg: HangingAgent(cfg))
    snapshot = await manager.start_run(RuntimeRequest(goal="Recover after restart"))
    await asyncio.sleep(0.08)

    restored_manager = ThreadManager(base_config, agent_factory=lambda cfg: HangingAgent(cfg))
    summaries = restored_manager.list_threads_for_http()

    assert len(summaries) == 1
    summary = summaries[0]
    assert summary["thread_id"] == snapshot["thread_id"]
    assert summary["current_run_id"] == snapshot["run_id"]
    assert summary["recovered_from_disk"] is True
    assert summary["restart_required"] is True
    assert summary["active_run_summary"]["run_id"] == snapshot["run_id"]
    assert "active_run" not in summary
    assert "audit" not in summary

    restored_thread = restored_manager.get_thread(snapshot["thread_id"])
    detail = restored_manager.thread_snapshot_for_http(restored_thread)
    restored_run = restored_manager.run_snapshot_for_http(snapshot["run_id"])

    assert detail["recovered_from_disk"] is True
    assert detail["restart_required"] is True
    assert "Sunucu yeniden" in detail["restart_notice"]
    assert restored_run["run_id"] == snapshot["run_id"]
    assert restored_run["active"] is False
    assert restored_run["restart_required"] is True

    await manager.stop_run(snapshot["run_id"])
    await _wait_for_run(manager, snapshot["run_id"])


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
async def test_thread_manager_requests_browser_takeover_before_sensitive_full_mode_step(
    base_config: AgentConfig,
) -> None:
    browser = BrowserTool()

    def factory(cfg: AgentConfig) -> AutonomousAgent:
        return AutonomousAgent(config=cfg, llm=BrowserApprovalLLM(), tools=[browser])

    manager = ThreadManager(base_config, agent_factory=factory)
    snapshot = await manager.start_run(
        RuntimeRequest(
            goal="GitHub giris sayfasini ac ve sifre alanina parolami yazmayi dene.",
            permission_mode="full",
        )
    )

    def paused_thread():
        thread = manager.get_thread(snapshot["thread_id"])
        return thread if thread.status == "paused_for_user" else None

    thread = await _wait_for(paused_thread)
    paused_snapshot = manager.thread_snapshot_for_http(thread)

    assert thread.handoff_state == "user"
    assert thread.approval_requests == {}
    assert thread.question_requests == {}
    assert browser.calls == []
    assert any(
        event["type"] == "paused" and event.get("pause_kind") == "sensitive_browser_takeover"
        for event in paused_snapshot["active_run"]["events"]
    )

    await manager.resume_thread(snapshot["thread_id"], reason="finished manual control")
    session = await _wait_for_run(manager, snapshot["run_id"])
    final_snapshot = manager.snapshot_for_http(session)

    assert final_snapshot["status"] == "completed"
    assert browser.calls == [{"action": "screenshot"}]
    assert not any(event["type"] == "approval_requested" for event in final_snapshot["events"])
    assert not any(event["type"] == "question_requested" for event in final_snapshot["events"])
    assert any(event["type"] == "paused" for event in final_snapshot["events"])
    assert any(event["type"] == "resumed" for event in final_snapshot["events"])


@pytest.mark.asyncio
async def test_thread_manager_collects_manual_sensitive_browser_input_without_persisting_plaintext(
    base_config: AgentConfig,
) -> None:
    browser = BrowserTool()

    def factory(cfg: AgentConfig) -> AutonomousAgent:
        return AutonomousAgent(config=cfg, llm=BrowserSecretTypeLLM(), tools=[browser])

    manager = ThreadManager(base_config, agent_factory=factory)
    snapshot = await manager.start_run(
        RuntimeRequest(
            goal="GitHub giris sayfasini ac ve sifre alanina parolami yazmayi dene.",
            permission_mode="full",
        )
    )

    def paused_thread():
        thread = manager.get_thread(snapshot["thread_id"])
        return thread if thread.status == "paused_for_user" else None

    await _wait_for(paused_thread)
    await manager.human_action(
        snapshot["thread_id"],
        "browser",
        {"action": "type", "selector": "input[name='password']", "text": "Kaan123"},
    )
    await manager.human_action(
        snapshot["thread_id"],
        "browser",
        {"action": "click", "selector": "button.login"},
    )
    await manager.resume_thread(snapshot["thread_id"], reason="manual login attempt finished")

    session = await _wait_for_run(manager, snapshot["run_id"])
    final_snapshot = manager.snapshot_for_http(session)
    thread = manager.get_thread(snapshot["thread_id"])

    assert final_snapshot["status"] == "completed"
    assert browser.calls == [
        {"action": "type", "selector": "input[name='password']", "text": "Kaan123"},
        {"action": "click", "selector": "button.login"},
        {"action": "screenshot"},
    ]
    assert final_snapshot["approval_requests"] == []
    assert final_snapshot["question_requests"] == []
    assert "Kaan123" not in thread.ledger.path.read_text(encoding="utf-8")
    assert "Kaan123" not in thread.ledger.audit_path.read_text(encoding="utf-8")
    assert "Kaan123" not in session.report.events_path.read_text(encoding="utf-8")

    audit_entries = final_snapshot["audit"]
    human_entries = [entry for entry in audit_entries if entry["entry_type"] == "human_action"]
    assert len(human_entries) == 2
    assert human_entries[0]["details"]["args"]["text"] == "[REDACTED]"
    assert not any(event["type"] == "approval_requested" for event in final_snapshot["events"])
    assert not any(event["type"] == "question_requested" for event in final_snapshot["events"])


@pytest.mark.asyncio
async def test_thread_manager_captures_workspace_audit(base_config: AgentConfig) -> None:
    manager = ThreadManager(base_config, agent_factory=lambda cfg: WorkspaceWritingAgent(cfg))

    snapshot = await manager.start_run(RuntimeRequest(goal="write artifact"))
    session = await _wait_for_run(manager, snapshot["run_id"])
    final_snapshot = manager.snapshot_for_http(session)

    entry_types = [entry["entry_type"] for entry in final_snapshot["audit"]]
    assert "workspace_checkpoint" in entry_types
    assert "workspace_diff" in entry_types
