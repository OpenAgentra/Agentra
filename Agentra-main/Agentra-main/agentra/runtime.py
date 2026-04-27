"""Thread-aware runtime primitives for Agentra."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlencode

from agentra.approval_policy import (
    REDACTED_PLACEHOLDER,
    ApprovalPolicyEngine,
    redact_tool_args_for_storage,
)
from agentra.browser_runtime import BrowserSessionManager, LiveBrowserFrame
from agentra.config import AgentConfig
from agentra.desktop_automation import DesktopLiveFrame, DesktopSessionManager
from agentra.desktop_automation.preview_windows import PreviewWindowManager
from agentra.llm.registry import get_provider_spec
from agentra.logging_utils import exception_details_with_context
from agentra.run_report import RunReport
from agentra.task_routing import (
    choose_live_execution_policy,
    goal_mentions_web_target,
    goal_requests_real_browser_context,
)

RunSnapshot = dict[str, Any]
EventPayload = dict[str, Any]
AgentFactory = Callable[[AgentConfig], Any]
logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _slugify(value: str, *, default: str = "thread") -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or default


def _sanitize_tool_args_for_storage(
    tool_name: str,
    tool_args: dict[str, Any],
    *,
    goal: str = "",
    permission_mode: str = "default",
) -> dict[str, Any]:
    sanitized = redact_tool_args_for_storage(
        tool_name,
        tool_args,
        goal=goal,
        permission_mode=permission_mode,
    )
    if isinstance(sanitized, dict):
        sanitized.pop("capture_result_screenshot", None)
        sanitized.pop("capture_follow_up_screenshots", None)
    return sanitized


def _sanitize_event_for_storage(
    event: dict[str, Any],
    *,
    goal: str,
    permission_mode: str,
) -> dict[str, Any]:
    payload = dict(event)
    tool_name = str(payload.get("tool", ""))
    if isinstance(payload.get("args"), dict):
        payload["args"] = _sanitize_tool_args_for_storage(
            tool_name,
            payload["args"],
            goal=goal,
            permission_mode=permission_mode,
        )
    return payload


@dataclass
class ApprovalRequest:
    """A pending approval for a risky action."""

    request_id: str
    tool: str
    args: dict[str, Any]
    reason: str
    summary: str
    status: str = "pending"
    rule_id: str | None = None
    risk_level: str = "medium"
    created_at: str = field(default_factory=_now_iso)
    responded_at: str | None = None
    decision_note: str | None = None
    approved: bool | None = None


@dataclass
class UserInputRequest:
    """A pending user-facing question that must be answered before continuing."""

    request_id: str
    prompt: str
    summary: str
    response_kind: str = "text"
    status: str = "pending"
    created_at: str = field(default_factory=_now_iso)
    responded_at: str | None = None
    answer: str | None = None
    answer_redacted: bool = False


@dataclass
class HumanAction:
    """A manual user action injected into a paused thread."""

    action_id: str
    tool: str
    args: dict[str, Any]
    created_at: str = field(default_factory=_now_iso)


@dataclass
class RunSession:
    """Single run execution state for one thread."""

    run_id: str
    goal: str
    thread_id: str
    config: AgentConfig
    report: RunReport
    task: asyncio.Task[None] | None = None
    agent: Any | None = None
    status: str = "running"
    subscribers: set[asyncio.Queue[EventPayload]] = field(default_factory=set)
    completed: asyncio.Event = field(default_factory=asyncio.Event)
    started_at: str = field(default_factory=_now_iso)
    finished_at: str | None = None


@dataclass
class ThreadSession:
    """Long-lived user-visible thread with isolated workspace and memory."""

    thread_id: str
    title: str
    thread_dir: Path
    workspace_dir: Path
    memory_dir: Path
    long_term_memory_dir: Path
    config: AgentConfig
    ledger: "WorkspaceLedger"
    status: str = "idle"
    handoff_state: str = "agent"
    created_at: str = field(default_factory=_now_iso)
    current_run_id: str | None = None
    run_ids: list[str] = field(default_factory=list)
    approval_requests: dict[str, ApprovalRequest] = field(default_factory=dict)
    question_requests: dict[str, UserInputRequest] = field(default_factory=dict)
    human_actions: list[HumanAction] = field(default_factory=list)
    controller: "ThreadRuntimeController | None" = None
    stored_runs: list[dict[str, Any]] = field(default_factory=list)
    recovered_from_disk: bool = False
    restart_required: bool = False
    restart_notice: str = ""
    desktop_execution_mode_override: str | None = None


class ExecutionScheduler:
    """Coordinate shared runtime capabilities across concurrent threads."""

    def __init__(self) -> None:
        self._computer_lock = asyncio.Lock()
        self._hidden_session_locks: dict[str, asyncio.Lock] = {}

    @asynccontextmanager
    async def reserve(
        self,
        capabilities: tuple[str, ...] | list[str],
        *,
        thread_id: str | None = None,
        tool_name: str | None = None,
    ):
        del thread_id, tool_name
        normalized = {str(item or "") for item in capabilities}
        needs_computer = "computer" in normalized
        hidden_sessions = sorted(
            capability
            for capability in normalized
            if capability.startswith("desktop_session:")
        )
        if not needs_computer and not hidden_sessions:
            yield
            return

        acquired_hidden: list[asyncio.Lock] = []
        if needs_computer:
            await self._computer_lock.acquire()
        try:
            for capability in hidden_sessions:
                lock = self._hidden_session_locks.setdefault(capability, asyncio.Lock())
                await lock.acquire()
                acquired_hidden.append(lock)
            yield
        finally:
            for lock in reversed(acquired_hidden):
                lock.release()
            if needs_computer:
                self._computer_lock.release()


class WorkspaceLedger:
    """Persist thread-level runtime metadata alongside the isolated workspace."""

    def __init__(self, thread_dir: Path) -> None:
        self.thread_dir = thread_dir
        self.path = thread_dir / "ledger.json"
        self.audit_path = thread_dir / "audit.jsonl"
        self.thread_dir.mkdir(parents=True, exist_ok=True)

    def write_snapshot(
        self,
        thread: ThreadSession,
        runs: list[RunSession] | list[dict[str, Any]],
        *,
        browser: dict[str, Any] | None = None,
        desktop: dict[str, Any] | None = None,
    ) -> None:
        run_entries: list[dict[str, Any]] = []
        for item in runs:
            if isinstance(item, RunSession):
                run_entries.append(
                    {
                        "run_id": item.run_id,
                        "goal": item.goal,
                        "status": item.status,
                        "permission_mode": item.config.permission_mode,
                        "local_execution_mode": item.config.local_execution_mode,
                        "desktop_execution_mode": item.config.desktop_execution_mode,
                        "desktop_backend_preference": item.config.desktop_backend_preference,
                        "started_at": item.started_at,
                        "finished_at": item.finished_at,
                        "report_path": str(item.report.html_path),
                    }
                )
                continue
            record = dict(item)
            run_entries.append(
                {
                    "run_id": str(record.get("run_id", "")),
                    "goal": str(record.get("goal", "")),
                    "status": str(record.get("status", "running") or "running"),
                    "permission_mode": str(record.get("permission_mode", thread.config.permission_mode)),
                    "local_execution_mode": str(record.get("local_execution_mode", "")),
                    "desktop_execution_mode": str(record.get("desktop_execution_mode", "")),
                    "desktop_backend_preference": str(record.get("desktop_backend_preference", "")),
                    "started_at": str(record.get("started_at", "")),
                    "finished_at": record.get("finished_at"),
                    "report_path": str(record.get("report_path", "")),
                }
            )
        payload = {
            "thread_id": thread.thread_id,
            "title": thread.title,
            "status": thread.status,
            "handoff_state": thread.handoff_state,
            "permission_mode": thread.config.permission_mode,
            "desktop_execution_mode_override": thread.desktop_execution_mode_override,
            "browser_identity": thread.config.browser_identity,
            "browser_profile_name": thread.config.browser_profile_name,
            "created_at": thread.created_at,
            "current_run_id": thread.current_run_id,
            "workspace_dir": str(thread.workspace_dir),
            "memory_dir": str(thread.memory_dir),
            "long_term_memory_dir": str(thread.long_term_memory_dir),
            "runs": [
                dict(item)
                for item in run_entries
            ],
            "approvals": [asdict(item) for item in thread.approval_requests.values()],
            "questions": [asdict(item) for item in thread.question_requests.values()],
            "human_actions": [asdict(item) for item in thread.human_actions],
            "browser": browser or {},
            "desktop": desktop or {},
            "audit": self.entries(),
        }
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def append_entry(
        self,
        entry_type: str,
        *,
        run_id: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = {
            "entry_id": f"{entry_type}-{datetime.now().strftime('%Y%m%d%H%M%S%f')}",
            "entry_type": entry_type,
            "timestamp": _now_iso(),
            "run_id": run_id,
            "details": details or {},
        }
        with self.audit_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        return payload

    def entries(self, *, run_id: str | None = None) -> list[dict[str, Any]]:
        if not self.audit_path.exists():
            return []
        entries: list[dict[str, Any]] = []
        for raw_line in self.audit_path.read_text(encoding="utf-8").splitlines():
            if not raw_line.strip():
                continue
            item = json.loads(raw_line)
            if run_id is not None and item.get("run_id") != run_id:
                continue
            entries.append(item)
        return entries


class ThreadRuntimeController:
    """Pause/resume plus approval and question futures for one thread."""

    def __init__(self, thread_id: str) -> None:
        self.thread_id = thread_id
        self._resume_event = asyncio.Event()
        self._resume_event.set()
        self._approval_futures: dict[str, asyncio.Future[bool]] = {}
        self._question_futures: dict[str, asyncio.Future[str]] = {}
        self._preview_window_manager = PreviewWindowManager()

    async def wait_until_resumed(self) -> None:
        await self._resume_event.wait()

    def pause(self) -> None:
        self._resume_event.clear()
        self.restore_preview_windows()

    def resume(self) -> None:
        self._resume_event.set()

    def prepare_for_visible_desktop_action(self) -> dict[str, Any] | None:
        return self._preview_window_manager.prepare_for_visible_desktop_action()

    def restore_preview_windows(self) -> int:
        return self._preview_window_manager.restore_preview_windows()

    def create_approval(
        self,
        tool: str,
        args: dict[str, Any],
        summary: str,
        reason: str,
        *,
        rule_id: str | None = None,
        risk_level: str = "medium",
    ) -> ApprovalRequest:
        request_id = f"approval-{self.thread_id}-{len(self._approval_futures) + 1:03d}"
        future: asyncio.Future[bool] = asyncio.get_running_loop().create_future()
        self._approval_futures[request_id] = future
        return ApprovalRequest(
            request_id=request_id,
            tool=tool,
            args=args,
            summary=summary,
            reason=reason,
            rule_id=rule_id,
            risk_level=risk_level,
        )

    async def wait_for_approval(self, request_id: str) -> bool:
        future = self._approval_futures[request_id]
        return await future

    def resolve_approval(self, request_id: str, approved: bool) -> None:
        future = self._approval_futures.get(request_id)
        if future is None or future.done():
            return
        future.set_result(bool(approved))

    def create_question(self, prompt: str, summary: str, *, response_kind: str = "text") -> UserInputRequest:
        request_id = f"question-{self.thread_id}-{len(self._question_futures) + 1:03d}"
        future: asyncio.Future[str] = asyncio.get_running_loop().create_future()
        self._question_futures[request_id] = future
        return UserInputRequest(
            request_id=request_id,
            prompt=prompt,
            summary=summary,
            response_kind=response_kind,
        )

    async def wait_for_answer(self, request_id: str) -> str:
        future = self._question_futures[request_id]
        return await future

    def resolve_question(self, request_id: str, answer: str) -> None:
        future = self._question_futures.get(request_id)
        if future is None or future.done():
            return
        future.set_result(answer)

    def cancel_pending(self) -> None:
        for future in list(self._approval_futures.values()):
            if not future.done():
                future.cancel()
        for future in list(self._question_futures.values()):
            if not future.done():
                future.cancel()


class ThreadManager:
    """Manage concurrent user-visible threads and their active runs."""

    def __init__(self, base_config: AgentConfig, agent_factory: AgentFactory | None = None) -> None:
        self.base_config = base_config
        self._agent_factory = agent_factory or _default_agent_factory
        self.scheduler = ExecutionScheduler()
        self.approval_engine = ApprovalPolicyEngine.default()
        self.browser_sessions = BrowserSessionManager()
        self.desktop_sessions = DesktopSessionManager()
        self._threads: dict[str, ThreadSession] = {}
        self._runs: dict[str, RunSession] = {}
        self._run_to_thread: dict[str, str] = {}
        self._persisted_run_artifacts: dict[str, dict[str, Any]] = {}
        self._lock = asyncio.Lock()
        self._threads_root = self.base_config.workspace_dir / ".threads"
        self._registry_path = self._threads_root / "registry.json"
        self._threads_root.mkdir(parents=True, exist_ok=True)
        self._restore_threads_from_disk()

    @property
    def active_run_id(self) -> str | None:
        active = [
            run for run in self._runs.values()
            if not run.completed.is_set() and run.status == "running"
        ]
        if not active:
            return None
        active.sort(key=lambda run: run.started_at)
        return active[-1].run_id

    def _restore_threads_from_disk(self) -> None:
        thread_dirs: dict[str, Path] = {}
        if self._registry_path.exists():
            try:
                payload = json.loads(self._registry_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                payload = []
            for item in payload if isinstance(payload, list) else []:
                thread_id = str(item.get("thread_id", "")).strip()
                if thread_id:
                    thread_dirs[thread_id] = self._threads_root / thread_id
        for candidate in sorted(self._threads_root.glob("*/ledger.json")):
            thread_dirs.setdefault(candidate.parent.name, candidate.parent)
        for thread_id, thread_dir in thread_dirs.items():
            ledger_path = thread_dir / "ledger.json"
            if not ledger_path.exists():
                continue
            try:
                payload = json.loads(ledger_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                logger.debug("Skipping unreadable thread ledger thread_id=%s", thread_id, exc_info=True)
                continue
            restored = self._thread_from_ledger(thread_dir, payload)
            if restored is None:
                continue
            self._threads[restored.thread_id] = restored
            self.browser_sessions.note_thread_browser_defaults(
                restored.thread_id,
                identity=restored.config.browser_identity,
                profile_name=restored.config.browser_profile_name,
            )
            self.browser_sessions.restore_thread_snapshot(restored.thread_id, payload.get("browser"))
            self.desktop_sessions.note_thread_config(restored.thread_id, restored.config)
            self.desktop_sessions.restore_thread_snapshot(restored.thread_id, payload.get("desktop"))
            for run_record in restored.stored_runs:
                self._remember_persisted_run(restored.thread_id, run_record)

    def _thread_from_ledger(self, thread_dir: Path, payload: dict[str, Any]) -> ThreadSession | None:
        thread_id = str(payload.get("thread_id", thread_dir.name)).strip()
        if not thread_id:
            return None
        stored_runs = [dict(item) for item in payload.get("runs", []) if isinstance(item, dict)]
        run_ids = [str(item.get("run_id", "")).strip() for item in stored_runs if str(item.get("run_id", "")).strip()]
        raw_current_run_id = payload.get("current_run_id")
        if raw_current_run_id in {None, "", "None"}:
            current_run_id = None
        else:
            current_run_id = str(raw_current_run_id).strip() or None
        permission_mode = str(payload.get("permission_mode", self.base_config.permission_mode) or self.base_config.permission_mode)
        browser_identity = str(payload.get("browser_identity", self.base_config.browser_identity) or self.base_config.browser_identity)
        browser_profile_name = str(payload.get("browser_profile_name", self.base_config.browser_profile_name) or self.base_config.browser_profile_name)

        workspace_dir = Path(str(payload.get("workspace_dir") or (thread_dir / "workspace")))
        memory_dir = Path(str(payload.get("memory_dir") or (workspace_dir / ".memory")))
        long_term_memory_dir = Path(
            str(payload.get("long_term_memory_dir") or (self.base_config.workspace_dir / ".memory-global"))
        )

        overrides = self.base_config.model_dump()
        overrides["workspace_dir"] = workspace_dir
        overrides["memory_dir"] = memory_dir
        overrides["long_term_memory_dir"] = long_term_memory_dir
        overrides["permission_mode"] = permission_mode
        overrides["browser_identity"] = browser_identity
        overrides["browser_profile_name"] = browser_profile_name
        if browser_identity == "chrome_profile":
            overrides["browser_headless"] = False
        config = AgentConfig(**overrides)

        approvals = {
            str(item.get("request_id", "")): ApprovalRequest(**item)
            for item in payload.get("approvals", [])
            if isinstance(item, dict) and str(item.get("request_id", "")).strip()
        }
        questions = {
            str(item.get("request_id", "")): UserInputRequest(**item)
            for item in payload.get("questions", [])
            if isinstance(item, dict) and str(item.get("request_id", "")).strip()
        }
        human_actions = [
            HumanAction(**item)
            for item in payload.get("human_actions", [])
            if isinstance(item, dict) and str(item.get("action_id", "")).strip()
        ]

        thread = ThreadSession(
            thread_id=thread_id,
            title=str(payload.get("title", thread_id)),
            thread_dir=thread_dir,
            workspace_dir=workspace_dir,
            memory_dir=memory_dir,
            long_term_memory_dir=long_term_memory_dir,
            config=config,
            ledger=WorkspaceLedger(thread_dir),
            status=str(payload.get("status", "idle") or "idle"),
            handoff_state=str(payload.get("handoff_state", "agent") or "agent"),
            created_at=str(payload.get("created_at", _now_iso())),
            current_run_id=current_run_id,
            run_ids=run_ids,
            approval_requests=approvals,
            question_requests=questions,
            human_actions=human_actions,
            controller=ThreadRuntimeController(thread_id),
            stored_runs=stored_runs,
            recovered_from_disk=True,
            desktop_execution_mode_override=(
                str(payload.get("desktop_execution_mode_override", "") or "").strip() or None
            ),
        )
        if thread.desktop_execution_mode_override:
            thread.config.desktop_execution_mode = thread.desktop_execution_mode_override
        current_run = next((item for item in stored_runs if str(item.get("run_id")) == current_run_id), None)
        if current_run_id and self._run_record_is_non_terminal(current_run):
            thread.restart_required = True
            thread.restart_notice = (
                "Sunucu yeniden baslatildi. Bu run canli olarak devam etmiyor; yeni bir run baslatip devam edin."
            )
        return thread

    @staticmethod
    def _run_record_is_non_terminal(record: dict[str, Any] | None) -> bool:
        if not record:
            return False
        status = str(record.get("status", "") or "").strip().lower()
        if status in {"completed", "error"}:
            return False
        finished_at = record.get("finished_at")
        return not bool(finished_at)

    def _remember_persisted_run(self, thread_id: str, record: dict[str, Any]) -> None:
        run_id = str(record.get("run_id", "")).strip()
        if not run_id:
            return
        report_path = Path(str(record.get("report_path", ""))) if record.get("report_path") else None
        run_dir = report_path.parent if report_path else (self._threads_root / thread_id / "workspace" / ".runs" / run_id)
        self._persisted_run_artifacts[run_id] = {
            "thread_id": thread_id,
            "run_id": run_id,
            "goal": str(record.get("goal", "")),
            "status": str(record.get("status", "running") or "running"),
            "started_at": str(record.get("started_at", "")),
            "finished_at": record.get("finished_at"),
            "report_path": str(report_path) if report_path else str(run_dir / "index.html"),
            "events_path": str(run_dir / "events.json"),
            "run_dir": str(run_dir),
        }

    async def start_run(self, request: Any) -> RunSnapshot:
        async with self._lock:
            thread = self._thread_for_request(request)
            current = self.current_run(thread.thread_id)
            if current and not current.completed.is_set():
                raise ValueError("This thread already has an active run.")

            config = self._config_for_request(request, thread)
            report = RunReport(
                config.workspace_dir,
                request.goal,
                config.llm_provider,
                config.llm_model,
                thread_id=thread.thread_id,
                thread_title=thread.title,
            )
            run = RunSession(
                run_id=report.store.run_id,
                goal=request.goal,
                thread_id=thread.thread_id,
                config=config,
                report=report,
            )
            self._runs[run.run_id] = run
            self._run_to_thread[run.run_id] = thread.thread_id
            thread.current_run_id = run.run_id
            thread.run_ids.append(run.run_id)
            thread.recovered_from_disk = False
            thread.restart_required = False
            thread.restart_notice = ""
            thread.status = "running"
            thread.handoff_state = "agent"
            thread.config.local_execution_mode = config.local_execution_mode
            thread.config.desktop_fallback_policy = config.desktop_fallback_policy
            thread.config.desktop_execution_mode = config.desktop_execution_mode
            thread.config.desktop_backend_preference = config.desktop_backend_preference
            if thread.controller is None:
                thread.controller = ThreadRuntimeController(thread.thread_id)
            thread.controller.resume()
            self.desktop_sessions.note_thread_config(thread.thread_id, config)
            thread.ledger.append_entry(
                "run_started",
                run_id=run.run_id,
                details={
                    "goal": request.goal,
                    "status": "running",
                    "permission_mode": config.permission_mode,
                    "local_execution_mode": config.local_execution_mode,
                    "desktop_execution_mode": config.desktop_execution_mode,
                    "desktop_backend_preference": config.desktop_backend_preference,
                },
            )
            self._persist_thread(thread)
            logger.info(
                "Thread run started thread_id=%s run_id=%s provider=%s model=%s permission_mode=%s local_execution_mode=%s desktop_execution_mode=%s desktop_backend_preference=%s goal=%r workspace=%s",
                thread.thread_id,
                run.run_id,
                config.llm_provider,
                config.llm_model,
                config.permission_mode,
                config.local_execution_mode,
                config.desktop_execution_mode,
                config.desktop_backend_preference,
                request.goal,
                thread.workspace_dir,
            )
            if config.browser_identity == "chrome_profile" and goal_mentions_web_target(str(request.goal or "")):
                self.browser_sessions.warmup_thread(
                    thread_id=thread.thread_id,
                    browser_type="chromium",
                    headless=config.browser_headless,
                    identity="chrome_profile",
                    profile_name=config.browser_profile_name,
                )
            run.task = asyncio.create_task(self._run_session(thread.thread_id, run.run_id))
            return self.snapshot_for_http(run)

    def list_threads_for_http(self) -> list[dict[str, Any]]:
        threads = sorted(self._threads.values(), key=lambda item: item.created_at, reverse=True)
        return [self.thread_summary_for_http(thread) for thread in threads]

    def get_thread(self, thread_id: str) -> ThreadSession:
        session = self._threads.get(thread_id)
        if session is None:
            raise KeyError(thread_id)
        return session

    def get_session(self, run_id: str) -> RunSession:
        session = self._runs.get(run_id)
        if session is None:
            raise KeyError(run_id)
        return session

    def current_run(self, thread_id: str) -> RunSession | None:
        thread = self.get_thread(thread_id)
        if not thread.current_run_id:
            return None
        return self._runs.get(thread.current_run_id)

    async def stop_run(self, run_id: str) -> RunSnapshot:
        session = self.get_session(run_id)
        agent = session.agent
        if agent is not None:
            interrupter = getattr(agent, "interrupt", None)
            if callable(interrupter):
                interrupter()
        if session.task and session.task.done():
            await asyncio.sleep(0)
        return self.snapshot_for_http(session)

    async def pause_thread(self, thread_id: str, *, reason: str = "User took over control.") -> dict[str, Any]:
        thread = self.get_thread(thread_id)
        run = self.current_run(thread_id)
        if thread.controller is None:
            thread.controller = ThreadRuntimeController(thread.thread_id)
        thread.controller.pause()
        if run is not None and run.agent is not None:
            pauser = getattr(run.agent, "pause", None)
            if callable(pauser):
                pauser()
        thread.status = "paused_for_user"
        thread.handoff_state = "user"
        payload = {
            "type": "paused",
            "content": reason,
            "summary": "Kullanıcı devraldı.",
        }
        if run is not None:
            stored = run.report.record(payload)
            await self._broadcast(run, {"kind": "event", "event": self._event_for_http(run.run_id, stored)})
        self._persist_thread(thread)
        logger.info("Thread paused thread_id=%s run_id=%s reason=%r", thread.thread_id, thread.current_run_id, reason)
        return self.thread_snapshot_for_http(thread)

    async def resume_thread(self, thread_id: str, *, reason: str = "Agent resumed control.") -> dict[str, Any]:
        thread = self.get_thread(thread_id)
        run = self.current_run(thread_id)
        if thread.controller is None:
            thread.controller = ThreadRuntimeController(thread.thread_id)
        thread.controller.resume()
        if run is not None and run.agent is not None:
            resumer = getattr(run.agent, "resume", None)
            if callable(resumer):
                resumer()
        thread.status = "running" if run and not run.completed.is_set() else "idle"
        thread.handoff_state = "agent"
        payload = {
            "type": "resumed",
            "content": reason,
            "summary": "Ajan kontrolü devraldı.",
        }
        if run is not None:
            stored = run.report.record(payload)
            await self._broadcast(run, {"kind": "event", "event": self._event_for_http(run.run_id, stored)})
        self._persist_thread(thread)
        logger.info("Thread resumed thread_id=%s run_id=%s reason=%r", thread.thread_id, thread.current_run_id, reason)
        return self.thread_snapshot_for_http(thread)

    async def respond_to_approval(
        self,
        thread_id: str,
        request_id: str,
        *,
        approved: bool,
        note: str = "",
    ) -> dict[str, Any]:
        thread = self.get_thread(thread_id)
        request = thread.approval_requests[request_id]
        if request.status != "pending":
            return self.thread_snapshot_for_http(thread)
        request.status = "approved" if approved else "rejected"
        request.approved = approved
        request.decision_note = note or None
        request.responded_at = _now_iso()
        if thread.controller is not None:
            thread.controller.resolve_approval(request_id, approved)
        thread.ledger.append_entry(
            "approval_resolved",
            run_id=thread.current_run_id,
            details={
                "request_id": request_id,
                "tool": request.tool,
                "approved": approved,
                "rule_id": request.rule_id,
                "risk_level": request.risk_level,
                "note": note,
            },
        )

        run = self.current_run(thread_id)
        if run is not None:
            payload = {
                "type": "approved" if approved else "rejected",
                "content": note or request.reason,
                "summary": "Kullanıcı onay verdi." if approved else "Kullanıcı işlemi reddetti.",
                "request_id": request_id,
                "tool": request.tool,
            }
            stored = run.report.record(payload)
            await self._broadcast(run, {"kind": "event", "event": self._event_for_http(run.run_id, stored)})
            if thread.status == "blocked_waiting_user":
                thread.status = "running"
        self._persist_thread(thread)
        return self.thread_snapshot_for_http(thread)

    async def answer_question(self, thread_id: str, request_id: str, answer: str) -> dict[str, Any]:
        thread = self.get_thread(thread_id)
        request = thread.question_requests[request_id]
        if request.status != "pending":
            return self.thread_snapshot_for_http(thread)
        request.status = "answered"
        request.responded_at = _now_iso()
        is_secret = request.response_kind == "secret"
        if is_secret:
            request.answer = None
            request.answer_redacted = True
        else:
            request.answer = answer
        if thread.controller is not None:
            thread.controller.resolve_question(request_id, answer)
        thread.ledger.append_entry(
            "question_answered",
            run_id=thread.current_run_id,
            details={
                "request_id": request_id,
                "answer": REDACTED_PLACEHOLDER if is_secret else answer,
                "answer_redacted": is_secret,
            },
        )

        run = self.current_run(thread_id)
        if run is not None:
            payload = {
                "type": "user_answer_received",
                "content": REDACTED_PLACEHOLDER if is_secret else answer,
                "summary": "Güvenli yanıt alındı." if is_secret else "Kullanıcı yanıtı alındı.",
                "request_id": request_id,
                "response_kind": request.response_kind,
                "answer_redacted": is_secret,
            }
            stored = run.report.record(payload)
            await self._broadcast(run, {"kind": "event", "event": self._event_for_http(run.run_id, stored)})
            if thread.status == "blocked_waiting_user":
                thread.status = "running"
        self._persist_thread(thread)
        return self.thread_snapshot_for_http(thread)

    async def human_action(
        self,
        thread_id: str,
        tool: str,
        args: dict[str, Any],
        *,
        return_snapshot: bool = True,
    ) -> dict[str, Any]:
        thread = self.get_thread(thread_id)
        run = self.current_run(thread_id)
        if run is None or run.agent is None:
            raise ValueError("No active agent is available for this thread.")

        stored_args = _sanitize_tool_args_for_storage(
            tool,
            args,
            goal=run.goal,
            permission_mode=thread.config.permission_mode,
        )

        actor_note = {
            "type": "human_action",
            "tool": tool,
            "args": stored_args,
            "content": "User performed a manual action.",
            "summary": f"Kullanıcı {tool} aracını kullandı.",
        }
        stored = run.report.record(actor_note)
        await self._broadcast(run, {"kind": "event", "event": self._event_for_http(run.run_id, stored)})

        performer = getattr(run.agent, "perform_human_action", None)
        if not callable(performer):
            raise ValueError("This agent does not support manual tool actions.")

        result = await performer(tool, args)
        action = HumanAction(
            action_id=f"manual-{len(thread.human_actions) + 1:03d}",
            tool=tool,
            args=stored_args,
        )
        thread.human_actions.append(action)
        thread.ledger.append_entry(
            "human_action",
            run_id=thread.current_run_id,
            details={"action_id": action.action_id, "tool": tool, "args": stored_args},
        )

        if result.screenshot_b64:
            action_name = str(args.get("action", "manual")).strip() or "manual"
            screenshot_event = {
                "type": "screenshot",
                "data": result.screenshot_b64,
                "frame_label": f"{tool} · {action_name}",
                "summary": f"Kullanıcı {tool} aracını kullandı.",
            }
            screenshot_event.update(
                {
                    key: value
                    for key, value in result.metadata.items()
                    if key in {"focus_x", "focus_y", "frame_label", "summary"}
                }
            )
            stored = run.report.record(screenshot_event)
            await self._broadcast(run, {"kind": "event", "event": self._event_for_http(run.run_id, stored)})
            for extra_frame in result.extra_screenshots:
                extra_event = {
                    "type": "screenshot",
                    "data": extra_frame.get("data", ""),
                }
                extra_event.update(
                    {
                        key: value
                        for key, value in extra_frame.items()
                        if key in {"focus_x", "focus_y", "frame_label", "summary"}
                    }
                )
                stored = run.report.record(extra_event)
                await self._broadcast(run, {"kind": "event", "event": self._event_for_http(run.run_id, stored)})

        tool_result_event = {
            "type": "tool_result",
            "tool": tool,
            "result": str(result),
            "success": result.success,
            "actor": "user",
            "metadata": result.metadata,
        }
        stored = run.report.record(tool_result_event)
        await self._broadcast(run, {"kind": "event", "event": self._event_for_http(run.run_id, stored)})
        self._persist_thread(thread)
        logger.info(
            "Human action applied thread_id=%s run_id=%s tool=%s args=%s success=%s",
            thread.thread_id,
            run.run_id,
            tool,
            args,
            result.success,
        )
        if return_snapshot:
            return self.thread_snapshot_for_http(thread)
        return {
            "ok": True,
            "thread_id": thread.thread_id,
            "run_id": run.run_id,
            "action_id": action.action_id,
            "tool": tool,
            "success": result.success,
            "manual_fast_path": True,
        }

    def update_thread_settings(
        self,
        thread_id: str,
        *,
        permission_mode: str | None = None,
        desktop_execution_mode: str | None = None,
    ) -> dict[str, Any]:
        thread = self.get_thread(thread_id)
        changed = False
        if permission_mode is not None:
            next_mode = str(permission_mode or "").strip().lower() or "default"
            if next_mode not in {"default", "full"}:
                raise ValueError("Unsupported permission mode.")
            if thread.config.permission_mode != next_mode:
                thread.config.permission_mode = next_mode
                thread.config.browser_identity = "chrome_profile" if next_mode == "full" else "isolated"
                if next_mode == "full":
                    thread.config.browser_headless = False
                changed = True
        if desktop_execution_mode is not None:
            next_desktop_mode = str(desktop_execution_mode or "").strip().lower()
            if next_desktop_mode not in {"desktop_visible", "desktop_native", "desktop_hidden"}:
                raise ValueError("Unsupported desktop execution mode.")
            if thread.desktop_execution_mode_override != next_desktop_mode:
                thread.desktop_execution_mode_override = next_desktop_mode
                thread.config.desktop_execution_mode = next_desktop_mode
                changed = True
        if changed:
            self.desktop_sessions.note_thread_config(thread.thread_id, thread.config)
            self.browser_sessions.note_thread_browser_defaults(
                thread.thread_id,
                identity=thread.config.browser_identity,
                profile_name=thread.config.browser_profile_name,
            )
            self.thread_ledger_append_settings(
                thread,
                permission_mode=thread.config.permission_mode,
                desktop_execution_mode_override=thread.desktop_execution_mode_override,
            )
            self._persist_thread(thread)
        return self.thread_snapshot_for_http(thread)

    def subscribe(self, run_id: str) -> asyncio.Queue[EventPayload]:
        session = self.get_session(run_id)
        queue: asyncio.Queue[EventPayload] = asyncio.Queue()
        session.subscribers.add(queue)
        return queue

    def unsubscribe(self, run_id: str, queue: asyncio.Queue[EventPayload]) -> None:
        session = self._runs.get(run_id)
        if session is not None:
            session.subscribers.discard(queue)

    def snapshot_for_http(self, session: RunSession) -> RunSnapshot:
        snapshot = session.report.snapshot()
        thread = self.get_thread(session.thread_id)
        return self._decorate_run_snapshot(
            snapshot,
            run_id=session.run_id,
            thread=thread,
            goal=session.goal,
            run_config=session.config,
            permission_mode=session.config.permission_mode,
            browser_identity=session.config.browser_identity,
            browser_profile_name=session.config.browser_profile_name,
            browser_headless=session.config.browser_headless,
            active=not session.completed.is_set() and thread.current_run_id == session.run_id,
        )

    def thread_summary_for_http(self, thread: ThreadSession) -> dict[str, Any]:
        active_summary = self._active_run_summary_for_thread(thread)
        activity = self._activity_for_thread(thread, active_summary)
        return {
            "thread_id": thread.thread_id,
            "title": thread.title,
            "status": thread.status,
            "handoff_state": thread.handoff_state,
            "permission_mode": thread.config.permission_mode,
            "desktop_execution_mode_override": thread.desktop_execution_mode_override,
            "browser_identity": thread.config.browser_identity,
            "browser_profile_name": thread.config.browser_profile_name,
            "created_at": thread.created_at,
            "workspace_dir": str(thread.workspace_dir),
            "memory_dir": str(thread.memory_dir),
            "long_term_memory_dir": str(thread.long_term_memory_dir),
            "current_run_id": thread.current_run_id,
            "logs_url": self._logs_url(thread_id=thread.thread_id, run_id=thread.current_run_id),
            "runs": self._run_summaries_for_http(thread),
            "approval_requests": [asdict(item) for item in thread.approval_requests.values()],
            "question_requests": [asdict(item) for item in thread.question_requests.values()],
            "active_run_summary": active_summary,
            "activity": activity,
            "activity_summary": str((activity or {}).get("summary") or ""),
            "activity_title": str((activity or {}).get("title") or ""),
            "recovered_from_disk": thread.recovered_from_disk,
            "restart_required": thread.restart_required,
            "restart_notice": thread.restart_notice,
            **self.browser_sessions.snapshot_payload(thread.thread_id),
            **self.desktop_sessions.snapshot_payload(thread.thread_id),
        }

    def thread_snapshot_for_http(self, thread: ThreadSession) -> dict[str, Any]:
        active_snapshot = self._active_run_snapshot_for_thread(thread)
        activity = active_snapshot.get("activity") if active_snapshot else None
        return {
            "thread_id": thread.thread_id,
            "title": thread.title,
            "status": thread.status,
            "handoff_state": thread.handoff_state,
            "permission_mode": thread.config.permission_mode,
            "desktop_execution_mode_override": thread.desktop_execution_mode_override,
            "browser_identity": thread.config.browser_identity,
            "browser_profile_name": thread.config.browser_profile_name,
            "created_at": thread.created_at,
            "workspace_dir": str(thread.workspace_dir),
            "memory_dir": str(thread.memory_dir),
            "long_term_memory_dir": str(thread.long_term_memory_dir),
            "current_run_id": thread.current_run_id,
            "logs_url": self._logs_url(thread_id=thread.thread_id, run_id=thread.current_run_id),
            "runs": self._run_summaries_for_http(thread),
            "approval_requests": [asdict(item) for item in thread.approval_requests.values()],
            "question_requests": [asdict(item) for item in thread.question_requests.values()],
            "active_run": active_snapshot,
            "activity": activity,
            "activity_summary": str((activity or {}).get("summary") or ""),
            "activity_title": str((activity or {}).get("title") or ""),
            "audit": thread.ledger.entries(run_id=thread.current_run_id) if thread.current_run_id else [],
            "recovered_from_disk": thread.recovered_from_disk,
            "restart_required": thread.restart_required,
            "restart_notice": thread.restart_notice,
            **self.browser_sessions.snapshot_payload(thread.thread_id),
            **self.desktop_sessions.snapshot_payload(thread.thread_id),
        }

    def run_snapshot_for_http(self, run_id: str) -> RunSnapshot:
        session = self._runs.get(run_id)
        if session is not None:
            return self.snapshot_for_http(session)
        record = self._persisted_run_artifacts.get(run_id)
        if record is None:
            raise KeyError(run_id)
        return self._load_persisted_run_snapshot(record)

    def _decorate_run_snapshot(
        self,
        snapshot: dict[str, Any],
        *,
        run_id: str,
        thread: ThreadSession,
        goal: str,
        run_config: AgentConfig | None,
        permission_mode: str,
        browser_identity: str,
        browser_profile_name: str,
        browser_headless: bool,
        active: bool,
    ) -> RunSnapshot:
        policy = choose_live_execution_policy(goal, requested_headless=browser_headless)
        payload = dict(snapshot)
        payload["active"] = active
        payload["thread_id"] = thread.thread_id
        payload["thread_title"] = thread.title
        payload["thread_status"] = thread.status
        payload["handoff_state"] = thread.handoff_state
        payload["permission_mode"] = permission_mode
        payload["browser_identity"] = browser_identity
        payload["browser_profile_name"] = browser_profile_name
        payload["control_surface_hint"] = policy.control_surface_hint
        payload["local_execution_mode"] = (
            run_config.local_execution_mode if run_config is not None else policy.local_execution_mode
        )
        payload["desktop_execution_mode"] = (
            run_config.desktop_execution_mode if run_config is not None else policy.desktop_execution_mode
        )
        payload["desktop_backend_preference"] = (
            run_config.desktop_backend_preference if run_config is not None else policy.desktop_backend_preference
        )
        payload["report_url"] = f"/runs/{run_id}/report"
        payload["logs_url"] = self._logs_url(thread_id=thread.thread_id, run_id=run_id)
        payload["events"] = [self._event_for_http(run_id, event) for event in payload.get("events", [])]
        payload["frames"] = [self._frame_for_http(run_id, frame) for frame in payload.get("frames", [])]
        payload["steps"] = _build_steps(payload["events"])
        payload["activity"] = _latest_activity(payload["events"])
        payload["approval_requests"] = [asdict(item) for item in thread.approval_requests.values()]
        payload["question_requests"] = [asdict(item) for item in thread.question_requests.values()]
        payload["audit"] = thread.ledger.entries(run_id=run_id)
        payload["recovered_from_disk"] = thread.recovered_from_disk
        payload["restart_required"] = thread.restart_required and thread.current_run_id == run_id
        payload["restart_notice"] = thread.restart_notice if payload["restart_required"] else ""
        payload.update(self.browser_sessions.snapshot_payload(thread.thread_id))
        payload.update(self.desktop_sessions.snapshot_payload(thread.thread_id))
        return payload

    def _run_summaries_for_http(self, thread: ThreadSession) -> list[dict[str, Any]]:
        summaries: list[dict[str, Any]] = []
        seen: set[str] = set()
        for run_id in thread.run_ids:
            summary = self._run_summary_for_thread(thread, run_id)
            if summary is None:
                continue
            normalized_run_id = str(summary.get("run_id", "")).strip()
            if not normalized_run_id or normalized_run_id in seen:
                continue
            seen.add(normalized_run_id)
            summaries.append(summary)
        return summaries

    def _run_summary_for_thread(self, thread: ThreadSession, run_id: str) -> dict[str, Any] | None:
        live_run = self._runs.get(run_id)
        if live_run is not None:
            return self._run_summary_for_http(live_run)
        record = next((item for item in thread.stored_runs if str(item.get("run_id", "")).strip() == run_id), None)
        if record is None:
            record = self._persisted_run_artifacts.get(run_id)
        if record is None:
            return None
        return self._stored_run_summary_for_http(thread.thread_id, record)

    def _stored_run_summary_for_http(self, thread_id: str, record: dict[str, Any]) -> dict[str, Any]:
        run_id = str(record.get("run_id", "")).strip()
        thread = self.get_thread(thread_id)
        return {
            "run_id": run_id,
            "goal": str(record.get("goal", "")),
            "status": str(record.get("status", "running") or "running"),
            "permission_mode": thread.config.permission_mode,
            "started_at": str(record.get("started_at", "")),
            "finished_at": record.get("finished_at"),
            "report_url": f"/runs/{run_id}/report",
            "logs_url": self._logs_url(thread_id=thread_id, run_id=run_id),
        }

    def _active_run_summary_for_thread(self, thread: ThreadSession) -> dict[str, Any] | None:
        if not thread.current_run_id:
            return None
        summary = self._run_summary_for_thread(thread, thread.current_run_id)
        if summary is None:
            return None
        activity = None
        live_run = self._runs.get(thread.current_run_id)
        if live_run is not None:
            activity = self.snapshot_for_http(live_run).get("activity")
        elif thread.restart_required:
            activity = {
                "title": "Devam gerekli",
                "summary": thread.restart_notice,
            }
        compact = dict(summary)
        compact["activity"] = activity
        return compact

    def _activity_for_thread(self, thread: ThreadSession, active_summary: dict[str, Any] | None) -> dict[str, Any] | None:
        activity = active_summary.get("activity") if active_summary else None
        if activity:
            return activity
        if thread.restart_required:
            return {
                "title": "Devam gerekli",
                "summary": thread.restart_notice,
            }
        return None

    def _active_run_snapshot_for_thread(self, thread: ThreadSession) -> dict[str, Any] | None:
        if not thread.current_run_id:
            return None
        try:
            return self.run_snapshot_for_http(thread.current_run_id)
        except KeyError:
            return None

    def _load_persisted_run_snapshot(self, record: dict[str, Any]) -> RunSnapshot:
        run_id = str(record.get("run_id", "")).strip()
        thread_id = str(record.get("thread_id", "")).strip()
        thread = self.get_thread(thread_id)
        events_path = Path(str(record.get("events_path", "")))
        if not events_path.exists():
            raise KeyError(run_id)
        payload = json.loads(events_path.read_text(encoding="utf-8"))
        return self._decorate_run_snapshot(
            payload,
            run_id=run_id,
            thread=thread,
            goal=str(payload.get("goal", record.get("goal", ""))),
            run_config=None,
            permission_mode=thread.config.permission_mode,
            browser_identity=thread.config.browser_identity,
            browser_profile_name=thread.config.browser_profile_name,
            browser_headless=thread.config.browser_headless,
            active=False,
        )

    async def capture_live_browser_frame(self, thread_id: str) -> LiveBrowserFrame | None:
        thread = self.get_thread(thread_id)
        frame = await self.browser_sessions.capture_live_frame(thread_id)
        self._archive_live_debug_frame(thread, frame, source="live-browser")
        return frame

    async def capture_live_computer_frame(
        self,
        thread_id: str,
    ) -> LiveBrowserFrame | DesktopLiveFrame | None:
        thread = self.get_thread(thread_id)
        if not thread.config.allow_computer_control:
            return None
        frame = await self.desktop_sessions.capture_live_frame(thread_id)
        self._archive_live_debug_frame(thread, frame, source="live-desktop")
        return frame

    def _archive_live_debug_frame(
        self,
        thread: ThreadSession,
        frame: LiveBrowserFrame | None,
        *,
        source: str,
    ) -> None:
        if frame is None:
            return
        run = self.current_run(thread.thread_id)
        if run is None:
            return
        saver = getattr(run.report.store, "save_debug_image_bytes", None)
        if not callable(saver):
            return
        try:
            saver(
                frame.data,
                source=source,
                media_type=frame.media_type,
                label=source,
                dedupe=True,
            )
        except Exception:  # noqa: BLE001
            logger.debug(
                "Failed to archive debug frame thread_id=%s run_id=%s source=%s",
                thread.thread_id,
                run.run_id,
                source,
                exc_info=True,
            )

    async def _run_session(self, thread_id: str, run_id: str) -> None:
        thread = self.get_thread(thread_id)
        session = self.get_session(run_id)
        status = "partial"
        try:
            agent = self._agent_factory(session.config)
            session.agent = agent

            binder = getattr(agent, "bind_runtime", None)
            if callable(binder):
                binder(
                    controller=thread.controller,
                    scheduler=self.scheduler,
                    thread_id=thread.thread_id,
                    run_id=session.run_id,
                    browser_sessions=self.browser_sessions,
                    desktop_sessions=self.desktop_sessions,
                    approval_engine=self.approval_engine,
                )

            generator = await agent.run(session.goal)
            async for raw_event in generator:
                stored = session.report.record(
                    _sanitize_event_for_storage(
                        raw_event,
                        goal=session.goal,
                        permission_mode=session.config.permission_mode,
                    )
                )
                if stored["type"] == "done":
                    status = "completed"
                elif stored["type"] == "error":
                    status = "error"
                    logger.error(
                        "Thread run emitted error event thread_id=%s run_id=%s content=%r",
                        thread.thread_id,
                        session.run_id,
                        stored.get("content", ""),
                    )
                elif stored["type"] in {"approval_requested", "question_requested"}:
                    thread.status = "blocked_waiting_user"
                await self._capture_runtime_state(thread, stored)
                await self._broadcast(
                    session,
                    {
                        "kind": "event",
                        "event": self._event_for_http(session.run_id, stored),
                    },
                )
        except Exception as exc:  # noqa: BLE001
            status = "error"
            details = exception_details_with_context(
                exc,
                provider=session.config.llm_provider,
                model=session.config.llm_model,
            )
            message = str(details.get("public_message") or str(exc))
            logger.exception(
                "Thread run crashed thread_id=%s run_id=%s goal=%r",
                thread.thread_id,
                session.run_id,
                session.goal,
            )
            payload: dict[str, Any] = {"type": "error", "content": message, "details": details}
            hint = str(details.get("hint") or "")
            if hint:
                payload["summary"] = hint
            stored = session.report.record(payload)
            thread.ledger.append_entry("run_error", run_id=session.run_id, details=details)
            await self._broadcast(
                session,
                {"kind": "event", "event": self._event_for_http(session.run_id, stored)},
            )
        finally:
            self.browser_sessions.cancel_warmup_thread(thread.thread_id)
            workspace_audit = self._capture_workspace_audit(thread, session)
            for entry in workspace_audit:
                session.report.record_audit(entry)
            session.status = status
            session.finished_at = _now_iso()
            session.report.finalize(status)
            session.completed.set()
            if thread.current_run_id == session.run_id:
                thread.current_run_id = None
            if status == "completed":
                thread.status = "completed"
            elif status == "error":
                thread.status = "error"
            elif thread.status not in {"paused_for_user", "blocked_waiting_user"}:
                thread.status = "idle"
            thread.handoff_state = "agent"
            if thread.controller is not None:
                thread.controller.restore_preview_windows()
                thread.controller.resume()
            thread.ledger.append_entry(
                "run_finished",
                run_id=session.run_id,
                details={"status": status, "workspace_audit": workspace_audit},
            )
            self._persist_thread(thread)
            logger.info(
                "Thread run finished thread_id=%s run_id=%s status=%s",
                thread.thread_id,
                session.run_id,
                status,
            )
            snapshot = self.snapshot_for_http(session)
            await self._broadcast(session, {"kind": "status", "status": status, "snapshot": snapshot})
            await self._broadcast(session, {"kind": "complete", "status": status})

    async def _capture_runtime_state(self, thread: ThreadSession, event: dict[str, Any]) -> None:
        event_type = str(event.get("type", ""))
        if thread.controller is None:
            return
        if event_type == "approval_requested":
            request = ApprovalRequest(
                request_id=str(event["request_id"]),
                tool=str(event.get("tool", "")),
                args=_sanitize_tool_args_for_storage(
                    str(event.get("tool", "")),
                    dict(event.get("args", {})),
                    goal=self.current_run(thread.thread_id).goal if self.current_run(thread.thread_id) else "",
                    permission_mode=thread.config.permission_mode,
                ),
                reason=str(event.get("reason", "")),
                summary=str(event.get("summary", "")),
                rule_id=str(event.get("rule_id", "") or "") or None,
                risk_level=str(event.get("risk_level", "medium")),
            )
            thread.approval_requests[request.request_id] = request
            thread.status = "blocked_waiting_user"
            thread.ledger.append_entry(
                "approval_requested",
                run_id=thread.current_run_id,
                details={
                    "request_id": request.request_id,
                    "tool": request.tool,
                    "reason": request.reason,
                    "rule_id": request.rule_id,
                    "risk_level": request.risk_level,
                },
            )
        elif event_type == "question_requested":
            request = UserInputRequest(
                request_id=str(event["request_id"]),
                prompt=str(event.get("content", "")),
                summary=str(event.get("summary", "")),
                response_kind=str(event.get("response_kind", "text") or "text"),
            )
            thread.question_requests[request.request_id] = request
            thread.status = "blocked_waiting_user"
            thread.ledger.append_entry(
                "question_requested",
                run_id=thread.current_run_id,
                details={"request_id": request.request_id, "prompt": request.prompt},
            )
        elif event_type == "paused":
            thread.status = "paused_for_user"
            thread.handoff_state = "user"
            thread.controller.restore_preview_windows()
        elif event_type == "screenshot":
            thread.ledger.append_entry(
                "artifact_recorded",
                run_id=thread.current_run_id,
                details={
                    "frame_id": event.get("frame_id"),
                    "frame_label": event.get("frame_label"),
                    "image_path": event.get("image_path"),
                },
            )
        elif event_type == "done":
            thread.status = "completed"
        elif event_type == "error":
            thread.status = "error"
            details = event.get("details") if isinstance(event.get("details"), dict) else {}
            thread.ledger.append_entry(
                "run_error",
                run_id=thread.current_run_id,
                details={
                    "message": str(event.get("content", "")),
                    "summary": str(event.get("summary", "")),
                    "exception_type": str(details.get("exception_type", "")),
                    "traceback": str(details.get("traceback", "")),
                },
            )
        elif event_type not in {"paused", "resumed"} and thread.status not in {"paused_for_user", "blocked_waiting_user"}:
            thread.status = "running"
        self._persist_thread(thread)

    def _capture_workspace_audit(self, thread: ThreadSession, session: RunSession) -> list[dict[str, Any]]:
        agent = session.agent
        workspace = getattr(agent, "workspace", None)
        if workspace is None:
            return []
        checkpoint = getattr(workspace, "checkpoint", None)
        if not callable(checkpoint):
            return []
        try:
            summary = checkpoint(f"task: {session.goal[:60]}")
        except Exception:  # noqa: BLE001
            return []
        audit_entries: list[dict[str, Any]] = []
        if summary.before_sha or summary.after_sha:
            audit_entries.append(
                thread.ledger.append_entry(
                    "workspace_checkpoint",
                    run_id=session.run_id,
                    details={
                        "before_sha": summary.before_sha,
                        "after_sha": summary.after_sha,
                        "status": summary.status,
                    },
                )
            )
        if summary.changed_files or summary.diff_stats:
            audit_entries.append(
                thread.ledger.append_entry(
                    "workspace_diff",
                    run_id=session.run_id,
                    details={
                        "changed_files": summary.changed_files,
                        "diff_stats": summary.diff_stats,
                    },
                )
            )
        return audit_entries

    async def _broadcast(self, session: RunSession, payload: EventPayload) -> None:
        stale: list[asyncio.Queue[EventPayload]] = []
        for queue in session.subscribers:
            try:
                queue.put_nowait(payload)
            except asyncio.QueueFull:
                stale.append(queue)
        for queue in stale:
            session.subscribers.discard(queue)

    def _event_for_http(self, run_id: str, event: dict[str, Any]) -> dict[str, Any]:
        from agentra.live_app import _display_label_for_event, _display_summary_for_event

        payload = dict(event)
        image_path = payload.get("image_path")
        if image_path:
            payload["image_url"] = self._asset_url(run_id, str(image_path))
        payload["display_label"] = _display_label_for_event(payload)
        payload["display_summary"] = _display_summary_for_event(payload)
        payload["activity"] = _activity_for_event_payload(payload)
        return payload

    def _frame_for_http(self, run_id: str, frame: dict[str, Any]) -> dict[str, Any]:
        from agentra.live_app import _display_label_for_frame, _display_summary_for_frame

        payload = dict(frame)
        image_path = payload.get("image_path")
        if image_path:
            payload["image_url"] = self._asset_url(run_id, str(image_path))
        payload["display_label"] = _display_label_for_frame(payload)
        payload["display_summary"] = _display_summary_for_frame(payload)
        return payload

    @staticmethod
    def _asset_url(run_id: str, image_path: str) -> str:
        return f"/runs/{run_id}/assets/{Path(image_path).name}"

    @staticmethod
    def _logs_url(*, thread_id: str | None = None, run_id: str | None = None) -> str:
        params: dict[str, str] = {}
        if thread_id:
            params["thread_id"] = thread_id
        if run_id:
            params["run_id"] = run_id
        query = urlencode(params)
        return f"/logs?{query}" if query else "/logs"

    def _thread_for_request(self, request: Any) -> ThreadSession:
        thread_id = getattr(request, "thread_id", None)
        if thread_id:
            return self.get_thread(str(thread_id))

        title = getattr(request, "thread_title", None) or getattr(request, "goal", "Yeni Thread")
        base_root = Path(getattr(request, "workspace", "")).expanduser().resolve() if getattr(request, "workspace", None) else self.base_config.workspace_dir
        threads_root = base_root / ".threads"
        threads_root.mkdir(parents=True, exist_ok=True)
        thread_id = self._generate_thread_id(title)
        thread_dir = threads_root / thread_id
        workspace_dir = thread_dir / "workspace"
        memory_dir = workspace_dir / ".memory"
        long_term_memory_dir = base_root / ".memory-global"
        overrides = self.base_config.model_dump()
        overrides["workspace_dir"] = workspace_dir
        overrides["memory_dir"] = memory_dir
        overrides["long_term_memory_dir"] = long_term_memory_dir
        explicit_permission_mode = getattr(request, "permission_mode", None)
        if explicit_permission_mode:
            overrides["permission_mode"] = request.permission_mode
        elif goal_requests_real_browser_context(str(getattr(request, "goal", "") or "")):
            overrides["permission_mode"] = "full"
            overrides["browser_identity"] = "chrome_profile"
            overrides["browser_headless"] = False
        config = AgentConfig(**overrides)
        thread = ThreadSession(
            thread_id=thread_id,
            title=title,
            thread_dir=thread_dir,
            workspace_dir=workspace_dir,
            memory_dir=memory_dir,
            long_term_memory_dir=long_term_memory_dir,
            config=config,
            ledger=WorkspaceLedger(thread_dir),
            controller=ThreadRuntimeController(thread_id),
        )
        self._threads[thread_id] = thread
        self.browser_sessions.note_thread_browser_defaults(
            thread.thread_id,
            identity=thread.config.browser_identity,
            profile_name=thread.config.browser_profile_name,
        )
        self.desktop_sessions.note_thread_config(thread.thread_id, thread.config)
        self._persist_thread(thread)
        return thread

    def _config_for_request(self, request: Any, thread: ThreadSession) -> AgentConfig:
        overrides: dict[str, Any] = self.base_config.model_dump()
        overrides["workspace_dir"] = thread.workspace_dir
        overrides["memory_dir"] = thread.memory_dir
        overrides["long_term_memory_dir"] = thread.long_term_memory_dir
        overrides["permission_mode"] = thread.config.permission_mode
        overrides["browser_identity"] = thread.config.browser_identity
        overrides["browser_profile_name"] = thread.config.browser_profile_name
        policy = choose_live_execution_policy(
            str(getattr(request, "goal", "") or ""),
            requested_headless=getattr(request, "headless", None),
        )
        effective_local_execution_mode = policy.local_execution_mode
        effective_desktop_fallback_policy = policy.desktop_fallback_policy
        effective_desktop_execution_mode = policy.desktop_execution_mode
        effective_desktop_backend_preference = policy.desktop_backend_preference
        override_mode = str(thread.desktop_execution_mode_override or "").strip().lower()
        if policy.local_execution_mode != "under_the_hood" and override_mode:
            effective_desktop_execution_mode = override_mode
            if override_mode == "desktop_hidden":
                effective_desktop_fallback_policy = "pause_and_ask"
                if policy.desktop_backend_preference == "native":
                    effective_local_execution_mode = "native"
                    effective_desktop_backend_preference = "native"
                else:
                    effective_local_execution_mode = "visible"
                    effective_desktop_backend_preference = "visible"
            elif override_mode == "desktop_native":
                effective_local_execution_mode = "native"
                effective_desktop_fallback_policy = "visible_control"
                effective_desktop_backend_preference = "native"
            elif override_mode == "desktop_visible":
                effective_local_execution_mode = "visible"
                effective_desktop_fallback_policy = "visible_control"
                effective_desktop_backend_preference = "visible"
        overrides["local_execution_mode"] = effective_local_execution_mode
        overrides["desktop_fallback_policy"] = effective_desktop_fallback_policy
        overrides["desktop_execution_mode"] = effective_desktop_execution_mode
        overrides["desktop_backend_preference"] = effective_desktop_backend_preference
        if getattr(request, "provider", None):
            overrides["llm_provider"] = request.provider
            if not getattr(request, "model", None):
                overrides["llm_model"] = get_provider_spec(request.provider).default_model
        if getattr(request, "model", None):
            overrides["llm_model"] = request.model
        overrides["browser_headless"] = policy.browser_headless
        if getattr(request, "max_iterations", None) is not None:
            overrides["max_iterations"] = request.max_iterations
        return AgentConfig(**overrides)

    def _generate_thread_id(self, title: str) -> str:
        base = _slugify(title, default="thread")[:24]
        candidate = f"thread-{base}"
        if candidate not in self._threads:
            return candidate
        index = 2
        while f"{candidate}-{index}" in self._threads:
            index += 1
        return f"{candidate}-{index}"

    def _persist_thread(self, thread: ThreadSession) -> None:
        run_records = self._thread_run_records_for_ledger(thread)
        thread.stored_runs = [dict(item) for item in run_records]
        thread.ledger.write_snapshot(
            thread,
            run_records,
            browser=self.browser_sessions.snapshot_payload(thread.thread_id),
            desktop=self.desktop_sessions.snapshot_payload(thread.thread_id),
        )
        for record in run_records:
            self._remember_persisted_run(thread.thread_id, record)
        registry = []
        for item in sorted(self._threads.values(), key=lambda entry: entry.created_at):
            registry.append(
                {
                    "thread_id": item.thread_id,
                    "title": item.title,
                    "status": item.status,
                    "permission_mode": item.config.permission_mode,
                    "created_at": item.created_at,
                    "current_run_id": item.current_run_id,
                    "workspace_dir": str(item.workspace_dir),
                }
            )
        self._registry_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")

    def _thread_run_records_for_ledger(self, thread: ThreadSession) -> list[dict[str, Any]]:
        records_by_id: dict[str, dict[str, Any]] = {
            str(item.get("run_id", "")).strip(): dict(item)
            for item in thread.stored_runs
            if str(item.get("run_id", "")).strip()
        }
        for run_id in thread.run_ids:
            live_run = self._runs.get(run_id)
            if live_run is not None:
                records_by_id[run_id] = {
                    "run_id": live_run.run_id,
                    "goal": live_run.goal,
                    "status": live_run.status,
                    "permission_mode": live_run.config.permission_mode,
                    "local_execution_mode": live_run.config.local_execution_mode,
                    "desktop_execution_mode": live_run.config.desktop_execution_mode,
                    "desktop_backend_preference": live_run.config.desktop_backend_preference,
                    "started_at": live_run.started_at,
                    "finished_at": live_run.finished_at,
                    "report_path": str(live_run.report.html_path),
                }
        ordered: list[dict[str, Any]] = []
        seen: set[str] = set()
        for run_id in thread.run_ids:
            record = records_by_id.get(run_id)
            if record is None or run_id in seen:
                continue
            ordered.append(record)
            seen.add(run_id)
        for run_id, record in records_by_id.items():
            if run_id in seen:
                continue
            ordered.append(record)
        return ordered

    @staticmethod
    def _run_summary_for_http(run: RunSession) -> dict[str, Any]:
        return {
            "run_id": run.run_id,
            "goal": run.goal,
            "status": run.status,
            "permission_mode": run.config.permission_mode,
            "local_execution_mode": run.config.local_execution_mode,
            "desktop_execution_mode": run.config.desktop_execution_mode,
            "desktop_backend_preference": run.config.desktop_backend_preference,
            "started_at": run.started_at,
            "finished_at": run.finished_at,
            "report_url": f"/runs/{run.run_id}/report",
            "logs_url": ThreadManager._logs_url(thread_id=run.thread_id, run_id=run.run_id),
        }

    @staticmethod
    def thread_ledger_append_settings(
        thread: ThreadSession,
        *,
        permission_mode: str,
        desktop_execution_mode_override: str | None,
    ) -> None:
        thread.ledger.append_entry(
            "thread_settings_updated",
            run_id=thread.current_run_id,
            details={
                "permission_mode": permission_mode,
                "desktop_execution_mode_override": desktop_execution_mode_override,
            },
        )


def _activity_channel_for_tool(tool_name: str) -> str:
    mapping = {
        "browser": "browser",
        "computer": "desktop",
        "windows_desktop": "desktop_native",
        "filesystem": "filesystem",
        "local_system": "local_system",
        "terminal": "terminal",
        "git": "workspace",
    }
    return mapping.get(tool_name, "agent")


def _activity_visibility(tool_name: str) -> str:
    if tool_name in {"browser", "computer", "windows_desktop"}:
        return "visible"
    if tool_name in {"filesystem", "local_system", "terminal", "git"}:
        return "hidden"
    return "background"


def _activity_for_event_payload(event: dict[str, Any]) -> dict[str, Any]:
    event_type = str(event.get("type", "event"))
    tool_name = str(event.get("tool", "") or "")
    metadata = event.get("metadata") if isinstance(event.get("metadata"), dict) else {}
    channel = _activity_channel_for_tool(tool_name)
    visibility = _activity_visibility(tool_name)
    desktop_execution_mode = str(metadata.get("desktop_execution_mode", "") or "")
    if desktop_execution_mode == "desktop_hidden" and tool_name in {"computer", "windows_desktop"}:
        channel = "desktop_hidden"
        visibility = "hidden"
    status = "idle"
    if event_type == "phase":
        channel = "agent"
        visibility = "background"
        status = "thinking" if str(event.get("phase", "")) == "thinking" else "preparing"
    elif event_type == "thought":
        channel = "agent"
        visibility = "background"
        status = "thinking"
    elif event_type == "tool_call":
        status = "running"
    elif event_type == "visual_intent":
        status = "preparing"
    elif event_type == "screenshot":
        frame_label = str(event.get("frame_label") or event.get("label") or "")
        tool_name = frame_label.split("·", 1)[0].strip().lower() if "·" in frame_label else tool_name
        channel = _activity_channel_for_tool(tool_name)
        visibility = _activity_visibility(tool_name)
        if desktop_execution_mode == "desktop_hidden" and tool_name in {"computer", "windows_desktop"}:
            channel = "desktop_hidden"
            visibility = "hidden"
        status = "visible_update"
    elif event_type == "tool_result":
        status = "completed" if event.get("success") else "error"
    elif event_type in {"approval_requested", "question_requested"}:
        channel = "agent"
        visibility = "background"
        status = "waiting"
    elif event_type == "paused":
        channel = "browser" if str(event.get("pause_kind", "")) == "sensitive_browser_takeover" else "agent"
        visibility = "visible" if channel == "browser" else "background"
        status = "waiting"
    elif event_type == "resumed":
        channel = "agent"
        visibility = "background"
        status = "completed"
    elif event_type in {"approved", "user_answer_received", "done"}:
        channel = "agent" if event_type == "done" else channel
        visibility = "background" if event_type == "done" else visibility
        status = "completed"
    elif event_type in {"rejected", "error"}:
        channel = "agent" if event_type == "error" else channel
        visibility = "background" if event_type == "error" else visibility
        status = "error"

    payload = {
        "channel": channel,
        "visibility": visibility,
        "status": status,
        "title": str(event.get("display_label") or event.get("frame_label") or event.get("tool") or "İşlem"),
        "summary": str(event.get("display_summary") or event.get("summary") or event.get("content") or ""),
    }
    if "focus_x" in event:
        payload["focus_x"] = event.get("focus_x")
    if "focus_y" in event:
        payload["focus_y"] = event.get("focus_y")
    if tool_name:
        payload["tool"] = tool_name
    args = event.get("args") if isinstance(event.get("args"), dict) else {}
    if args:
        target = args.get("path") or args.get("url") or args.get("selector") or args.get("folder_key")
        if target:
            payload["target"] = str(target)
    return payload


def _latest_activity(events: list[dict[str, Any]]) -> dict[str, Any]:
    for event in reversed(events):
        activity = event.get("activity")
        if isinstance(activity, dict) and (activity.get("summary") or activity.get("title")):
            return activity
    return {
        "channel": "agent",
        "visibility": "background",
        "status": "idle",
        "title": "Hazır",
        "summary": "",
    }


def _default_agent_factory(config: AgentConfig):
    from agentra.agents.autonomous import AutonomousAgent

    return AutonomousAgent(config=config)


def _generic_tool_label(tool: str) -> str:
    if not tool:
        return "Araç"
    return f"{tool.replace('_', ' ').title()} · İşlem"


def _build_steps(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    current_tool_step: dict[str, Any] | None = None

    for index, event in enumerate(events):
        event_type = str(event.get("type", "event"))
        timestamp = str(event.get("timestamp", ""))

        if event_type == "phase":
            continue
        if event_type == "thought":
            current_tool_step = None
            steps.append(
                {
                    "id": f"step-{index + 1:03d}",
                    "kind": "thought",
                    "tone": "assistant",
                    "title": str(event.get("display_label") or "Düşünce Özeti"),
                    "summary": str(event.get("display_summary") or ""),
                    "status_label": "düşünüyor",
                    "timestamp": timestamp,
                }
            )
            continue
        if event_type == "tool_call":
            current_tool_step = {
                "id": f"step-{index + 1:03d}",
                "kind": "tool",
                "tone": "pending",
                "tool": str(event.get("tool", "")),
                "title": str(event.get("display_label") or _generic_tool_label(str(event.get("tool", "")))),
                "summary": str(event.get("display_summary") or ""),
                "status_label": "uyguluyor",
                "timestamp": timestamp,
                "detail": json.dumps(event.get("args", {}), indent=2, ensure_ascii=False),
                "image_url": None,
            }
            steps.append(current_tool_step)
            continue
        if event_type == "visual_intent":
            if current_tool_step and current_tool_step.get("tool") == event.get("tool"):
                current_tool_step["status_label"] = "hazırlanıyor"
                current_tool_step["summary"] = str(event.get("display_summary") or current_tool_step["summary"])
            continue
        if event_type == "screenshot":
            if current_tool_step is not None:
                current_tool_step["image_url"] = event.get("image_url")
                current_tool_step["frame_id"] = event.get("frame_id")
            continue
        if event_type == "tool_result":
            if current_tool_step and current_tool_step.get("tool") == event.get("tool"):
                current_tool_step["tone"] = "success" if event.get("success") else "error"
                current_tool_step["status_label"] = "tamamlandı" if event.get("success") else "hata"
                current_tool_step["summary"] = str(event.get("display_summary") or current_tool_step["summary"])
                current_tool_step["detail"] = str(event.get("result") or "")
                current_tool_step = None
            continue
        if event_type in {"approval_requested", "question_requested", "approved", "rejected", "user_answer_received", "paused", "resumed", "human_action", "done", "error"}:
            current_tool_step = None
            tone = "neutral"
            if event_type in {"done", "approved", "user_answer_received", "resumed"}:
                tone = "success"
            elif event_type in {"rejected", "error"}:
                tone = "error"
            steps.append(
                {
                    "id": f"step-{index + 1:03d}",
                    "kind": event_type,
                    "tone": tone,
                    "title": str(event.get("display_label") or event_type.replace("_", " ").title()),
                    "summary": str(event.get("display_summary") or event.get("content") or ""),
                    "status_label": event_type.replace("_", " "),
                    "timestamp": timestamp,
                }
            )

    return steps[-24:]
