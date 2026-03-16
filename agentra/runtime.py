"""Thread-aware runtime primitives for Agentra."""

from __future__ import annotations

import asyncio
import json
import re
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from agentra.approval_policy import ApprovalPolicyEngine
from agentra.browser_runtime import BrowserSessionManager
from agentra.config import AgentConfig
from agentra.llm.registry import get_provider_spec
from agentra.run_report import RunReport

RunSnapshot = dict[str, Any]
EventPayload = dict[str, Any]
AgentFactory = Callable[[AgentConfig], Any]


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _slugify(value: str, *, default: str = "thread") -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or default


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
    status: str = "pending"
    created_at: str = field(default_factory=_now_iso)
    responded_at: str | None = None
    answer: str | None = None


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


class ExecutionScheduler:
    """Coordinate shared runtime capabilities across concurrent threads."""

    def __init__(self) -> None:
        self._computer_lock = asyncio.Lock()

    @asynccontextmanager
    async def reserve(
        self,
        capabilities: tuple[str, ...] | list[str],
        *,
        thread_id: str | None = None,
        tool_name: str | None = None,
    ):
        del thread_id, tool_name
        needs_computer = "computer" in set(capabilities)
        if not needs_computer:
            yield
            return

        await self._computer_lock.acquire()
        try:
            yield
        finally:
            self._computer_lock.release()


class WorkspaceLedger:
    """Persist thread-level runtime metadata alongside the isolated workspace."""

    def __init__(self, thread_dir: Path) -> None:
        self.thread_dir = thread_dir
        self.path = thread_dir / "ledger.json"
        self.audit_path = thread_dir / "audit.jsonl"
        self.thread_dir.mkdir(parents=True, exist_ok=True)

    def write_snapshot(self, thread: ThreadSession, runs: list[RunSession], *, browser: dict[str, Any] | None = None) -> None:
        payload = {
            "thread_id": thread.thread_id,
            "title": thread.title,
            "status": thread.status,
            "handoff_state": thread.handoff_state,
            "created_at": thread.created_at,
            "current_run_id": thread.current_run_id,
            "workspace_dir": str(thread.workspace_dir),
            "memory_dir": str(thread.memory_dir),
            "long_term_memory_dir": str(thread.long_term_memory_dir),
            "runs": [
                {
                    "run_id": run.run_id,
                    "goal": run.goal,
                    "status": run.status,
                    "started_at": run.started_at,
                    "finished_at": run.finished_at,
                    "report_path": str(run.report.html_path),
                }
                for run in runs
            ],
            "approvals": [asdict(item) for item in thread.approval_requests.values()],
            "questions": [asdict(item) for item in thread.question_requests.values()],
            "human_actions": [asdict(item) for item in thread.human_actions],
            "browser": browser or {},
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

    async def wait_until_resumed(self) -> None:
        await self._resume_event.wait()

    def pause(self) -> None:
        self._resume_event.clear()

    def resume(self) -> None:
        self._resume_event.set()

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

    def create_question(self, prompt: str, summary: str) -> UserInputRequest:
        request_id = f"question-{self.thread_id}-{len(self._question_futures) + 1:03d}"
        future: asyncio.Future[str] = asyncio.get_running_loop().create_future()
        self._question_futures[request_id] = future
        return UserInputRequest(request_id=request_id, prompt=prompt, summary=summary)

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
        self._threads: dict[str, ThreadSession] = {}
        self._runs: dict[str, RunSession] = {}
        self._run_to_thread: dict[str, str] = {}
        self._lock = asyncio.Lock()
        self._threads_root = self.base_config.workspace_dir / ".threads"
        self._registry_path = self._threads_root / "registry.json"
        self._threads_root.mkdir(parents=True, exist_ok=True)

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
            thread.status = "running"
            thread.handoff_state = "agent"
            if thread.controller is None:
                thread.controller = ThreadRuntimeController(thread.thread_id)
            thread.controller.resume()
            thread.ledger.append_entry(
                "run_started",
                run_id=run.run_id,
                details={"goal": request.goal, "status": "running"},
            )
            self._persist_thread(thread)
            run.task = asyncio.create_task(self._run_session(thread.thread_id, run.run_id))
            return self.snapshot_for_http(run)

    def list_threads_for_http(self) -> list[dict[str, Any]]:
        threads = sorted(self._threads.values(), key=lambda item: item.created_at, reverse=True)
        return [self.thread_snapshot_for_http(thread) for thread in threads]

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
        request.status = "answered"
        request.answer = answer
        request.responded_at = _now_iso()
        if thread.controller is not None:
            thread.controller.resolve_question(request_id, answer)
        thread.ledger.append_entry(
            "question_answered",
            run_id=thread.current_run_id,
            details={"request_id": request_id, "answer": answer},
        )

        run = self.current_run(thread_id)
        if run is not None:
            payload = {
                "type": "user_answer_received",
                "content": answer,
                "summary": "Kullanıcı yanıtı alındı.",
                "request_id": request_id,
            }
            stored = run.report.record(payload)
            await self._broadcast(run, {"kind": "event", "event": self._event_for_http(run.run_id, stored)})
            if thread.status == "blocked_waiting_user":
                thread.status = "running"
        self._persist_thread(thread)
        return self.thread_snapshot_for_http(thread)

    async def human_action(self, thread_id: str, tool: str, args: dict[str, Any]) -> dict[str, Any]:
        thread = self.get_thread(thread_id)
        run = self.current_run(thread_id)
        if run is None or run.agent is None:
            raise ValueError("No active agent is available for this thread.")

        actor_note = {
            "type": "human_action",
            "tool": tool,
            "args": args,
            "content": "User performed a manual action.",
            "summary": f"Kullanıcı {tool} aracını kullandı.",
        }
        stored = run.report.record(actor_note)
        await self._broadcast(run, {"kind": "event", "event": self._event_for_http(run.run_id, stored)})

        performer = getattr(run.agent, "perform_human_action", None)
        if not callable(performer):
            raise ValueError("This agent does not support manual tool actions.")

        result = await performer(tool, args)
        action = HumanAction(action_id=f"manual-{len(thread.human_actions) + 1:03d}", tool=tool, args=args)
        thread.human_actions.append(action)
        thread.ledger.append_entry(
            "human_action",
            run_id=thread.current_run_id,
            details={"action_id": action.action_id, "tool": tool, "args": args},
        )

        if result.screenshot_b64:
            screenshot_event = {"type": "screenshot", "data": result.screenshot_b64}
            screenshot_event.update(
                {
                    key: value
                    for key, value in result.metadata.items()
                    if key in {"focus_x", "focus_y", "frame_label", "summary"}
                }
            )
            stored = run.report.record(screenshot_event)
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
        snapshot["active"] = not session.completed.is_set() and thread.current_run_id == session.run_id
        snapshot["thread_id"] = thread.thread_id
        snapshot["thread_title"] = thread.title
        snapshot["thread_status"] = thread.status
        snapshot["handoff_state"] = thread.handoff_state
        snapshot["report_url"] = f"/runs/{session.run_id}/report"
        snapshot["events"] = [self._event_for_http(session.run_id, event) for event in snapshot["events"]]
        snapshot["frames"] = [self._frame_for_http(session.run_id, frame) for frame in snapshot["frames"]]
        snapshot["steps"] = _build_steps(snapshot["events"])
        snapshot["approval_requests"] = [asdict(item) for item in thread.approval_requests.values()]
        snapshot["question_requests"] = [asdict(item) for item in thread.question_requests.values()]
        snapshot["audit"] = thread.ledger.entries(run_id=session.run_id)
        snapshot.update(self.browser_sessions.snapshot_payload(thread.thread_id))
        return snapshot

    def thread_snapshot_for_http(self, thread: ThreadSession) -> dict[str, Any]:
        current = self.current_run(thread.thread_id)
        return {
            "thread_id": thread.thread_id,
            "title": thread.title,
            "status": thread.status,
            "handoff_state": thread.handoff_state,
            "created_at": thread.created_at,
            "workspace_dir": str(thread.workspace_dir),
            "memory_dir": str(thread.memory_dir),
            "long_term_memory_dir": str(thread.long_term_memory_dir),
            "current_run_id": thread.current_run_id,
            "runs": [
                self._run_summary_for_http(self._runs[run_id])
                for run_id in thread.run_ids
                if run_id in self._runs
            ],
            "approval_requests": [asdict(item) for item in thread.approval_requests.values()],
            "question_requests": [asdict(item) for item in thread.question_requests.values()],
            "active_run": self.snapshot_for_http(current) if current is not None else None,
            "audit": thread.ledger.entries(run_id=thread.current_run_id) if thread.current_run_id else [],
            **self.browser_sessions.snapshot_payload(thread.thread_id),
        }

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
                    approval_engine=self.approval_engine,
                )

            generator = await agent.run(session.goal)
            async for raw_event in generator:
                stored = session.report.record(raw_event)
                if stored["type"] == "done":
                    status = "completed"
                elif stored["type"] == "error":
                    status = "error"
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
            stored = session.report.record({"type": "error", "content": str(exc)})
            await self._broadcast(
                session,
                {"kind": "event", "event": self._event_for_http(session.run_id, stored)},
            )
        finally:
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
                thread.controller.resume()
            thread.ledger.append_entry(
                "run_finished",
                run_id=session.run_id,
                details={"status": status, "workspace_audit": workspace_audit},
            )
            self._persist_thread(thread)
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
                args=dict(event.get("args", {})),
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
            )
            thread.question_requests[request.request_id] = request
            thread.status = "blocked_waiting_user"
            thread.ledger.append_entry(
                "question_requested",
                run_id=thread.current_run_id,
                details={"request_id": request.request_id, "prompt": request.prompt},
            )
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
        self._persist_thread(thread)
        return thread

    def _config_for_request(self, request: Any, thread: ThreadSession) -> AgentConfig:
        overrides: dict[str, Any] = self.base_config.model_dump()
        overrides["workspace_dir"] = thread.workspace_dir
        overrides["memory_dir"] = thread.memory_dir
        overrides["long_term_memory_dir"] = thread.long_term_memory_dir
        if getattr(request, "provider", None):
            overrides["llm_provider"] = request.provider
            if not getattr(request, "model", None):
                overrides["llm_model"] = get_provider_spec(request.provider).default_model
        if getattr(request, "model", None):
            overrides["llm_model"] = request.model
        if getattr(request, "headless", None) is not None:
            overrides["browser_headless"] = request.headless
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
        runs = [self._runs[run_id] for run_id in thread.run_ids if run_id in self._runs]
        thread.ledger.write_snapshot(thread, runs, browser=self.browser_sessions.snapshot_payload(thread.thread_id))
        registry = []
        for item in sorted(self._threads.values(), key=lambda entry: entry.created_at):
            registry.append(
                {
                    "thread_id": item.thread_id,
                    "title": item.title,
                    "status": item.status,
                    "created_at": item.created_at,
                    "current_run_id": item.current_run_id,
                    "workspace_dir": str(item.workspace_dir),
                }
            )
        self._registry_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")

    @staticmethod
    def _run_summary_for_http(run: RunSession) -> dict[str, Any]:
        return {
            "run_id": run.run_id,
            "goal": run.goal,
            "status": run.status,
            "started_at": run.started_at,
            "finished_at": run.finished_at,
            "report_url": f"/runs/{run.run_id}/report",
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
