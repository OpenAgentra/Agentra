"""Local live UI for Agentra runs."""

from __future__ import annotations

import asyncio
import json
import threading
import webbrowser
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, Field

from agentra.config import AgentConfig
from agentra.llm.registry import get_provider_spec
from agentra.runtime import ThreadManager
from agentra.run_report import RunReport

RunSnapshot = dict[str, Any]
EventPayload = dict[str, Any]
AgentFactory = Callable[[AgentConfig], Any]

_DEFAULT_FOCUS = (0.74, 0.2)


class RunCreateRequest(BaseModel):
    """Payload to start a live run."""

    goal: str = Field(min_length=1)
    thread_id: str | None = None
    thread_title: str | None = None
    provider: str | None = None
    model: str | None = None
    headless: bool | None = None
    workspace: str | None = None
    max_iterations: int | None = Field(default=None, ge=1)


class ApprovalDecisionRequest(BaseModel):
    approved: bool
    note: str = ""


class UserAnswerRequest(BaseModel):
    answer: str = Field(min_length=1)


class HumanActionRequest(BaseModel):
    tool: str = Field(min_length=1)
    args: dict[str, Any] = Field(default_factory=dict)


@dataclass
class LiveRunSession:
    """In-memory state for a single live run."""

    run_id: str
    goal: str
    config: AgentConfig
    report: RunReport
    task: asyncio.Task[None] | None = None
    agent: Any | None = None
    status: str = "running"
    subscribers: set[asyncio.Queue[EventPayload]] = field(default_factory=set)
    completed: asyncio.Event = field(default_factory=asyncio.Event)


class LiveRunManager:
    """Manage one active live run plus historical snapshots."""

    def __init__(self, base_config: AgentConfig, agent_factory: AgentFactory | None = None) -> None:
        self.base_config = base_config
        self._agent_factory = agent_factory or _default_agent_factory
        self._active_run_id: str | None = None
        self._sessions: dict[str, LiveRunSession] = {}
        self._lock = asyncio.Lock()

    @property
    def active_run_id(self) -> str | None:
        return self._active_run_id

    async def start_run(self, request: RunCreateRequest) -> RunSnapshot:
        """Create and launch a live run."""
        async with self._lock:
            if self._active_run_id:
                active = self._sessions.get(self._active_run_id)
                if active and not active.completed.is_set():
                    raise ValueError("Another run is already active.")

            config = self._config_for_request(request)
            report = RunReport(config.workspace_dir, request.goal, config.llm_provider, config.llm_model)
            run_id = report.store.run_id
            session = LiveRunSession(
                run_id=run_id,
                goal=request.goal,
                config=config,
                report=report,
            )
            self._sessions[run_id] = session
            self._active_run_id = run_id
            session.task = asyncio.create_task(self._run_session(run_id))
            return self.snapshot_for_http(session)

    def get_session(self, run_id: str) -> LiveRunSession:
        session = self._sessions.get(run_id)
        if session is None:
            raise KeyError(run_id)
        return session

    async def stop_run(self, run_id: str) -> RunSnapshot:
        """Request a running session to stop."""
        session = self.get_session(run_id)
        agent = session.agent
        if agent is not None:
            interrupter = getattr(agent, "interrupt", None)
            if callable(interrupter):
                interrupter()
        if session.task and session.task.done():
            await asyncio.sleep(0)
        return self.snapshot_for_http(session)

    def subscribe(self, run_id: str) -> asyncio.Queue[EventPayload]:
        session = self.get_session(run_id)
        queue: asyncio.Queue[EventPayload] = asyncio.Queue()
        session.subscribers.add(queue)
        return queue

    def unsubscribe(self, run_id: str, queue: asyncio.Queue[EventPayload]) -> None:
        session = self._sessions.get(run_id)
        if session is not None:
            session.subscribers.discard(queue)

    def snapshot_for_http(self, session: LiveRunSession) -> RunSnapshot:
        """Return an API-friendly snapshot with asset URLs and derived steps."""
        snapshot = session.report.snapshot()
        snapshot["active"] = session.run_id == self._active_run_id and not session.completed.is_set()
        snapshot["report_url"] = f"/runs/{session.run_id}/report"
        snapshot["events"] = [self._event_for_http(session.run_id, event) for event in snapshot["events"]]
        snapshot["frames"] = [self._frame_for_http(session.run_id, frame) for frame in snapshot["frames"]]
        snapshot["steps"] = _build_steps(snapshot["events"])
        return snapshot

    async def _run_session(self, run_id: str) -> None:
        session = self.get_session(run_id)
        status = "partial"
        try:
            agent = self._agent_factory(session.config)
            session.agent = agent
            generator = await agent.run(session.goal)
            async for raw_event in generator:
                stored = session.report.record(raw_event)
                if stored["type"] == "done":
                    status = "completed"
                elif stored["type"] == "error":
                    status = "error"
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
            session.status = status
            session.report.finalize(status)
            session.completed.set()
            snapshot = self.snapshot_for_http(session)
            await self._broadcast(session, {"kind": "status", "status": status, "snapshot": snapshot})
            await self._broadcast(session, {"kind": "complete", "status": status})
            async with self._lock:
                if self._active_run_id == session.run_id:
                    self._active_run_id = None

    async def _broadcast(self, session: LiveRunSession, payload: EventPayload) -> None:
        stale: list[asyncio.Queue[EventPayload]] = []
        for queue in session.subscribers:
            try:
                queue.put_nowait(payload)
            except asyncio.QueueFull:
                stale.append(queue)
        for queue in stale:
            session.subscribers.discard(queue)

    def _event_for_http(self, run_id: str, event: dict[str, Any]) -> dict[str, Any]:
        payload = dict(event)
        image_path = payload.get("image_path")
        if image_path:
            payload["image_url"] = self._asset_url(run_id, str(image_path))
        payload["display_label"] = _display_label_for_event(payload)
        payload["display_summary"] = _display_summary_for_event(payload)
        return payload

    def _frame_for_http(self, run_id: str, frame: dict[str, Any]) -> dict[str, Any]:
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

    def _config_for_request(self, request: RunCreateRequest) -> AgentConfig:
        overrides: dict[str, Any] = self.base_config.model_dump()
        if request.provider:
            overrides["llm_provider"] = request.provider
            if not request.model:
                overrides["llm_model"] = get_provider_spec(request.provider).default_model
        if request.model:
            overrides["llm_model"] = request.model
        if request.headless is not None:
            overrides["browser_headless"] = request.headless
        if request.workspace:
            workspace_dir = Path(request.workspace)
            overrides["workspace_dir"] = workspace_dir
            overrides["memory_dir"] = workspace_dir / ".memory"
        if request.max_iterations is not None:
            overrides["max_iterations"] = request.max_iterations
        return AgentConfig(**overrides)


def _generic_tool_label(tool: str) -> str:
    if not tool:
        return "Araç"
    return f"{tool.replace('_', ' ').title()} · İşlem"


def _short_url(url: str) -> str:
    compact = str(url or "").replace("https://", "").replace("http://", "").rstrip("/")
    return compact.removeprefix("www.") or "sayfa"


def _trim_text(value: Any, limit: int = 220) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _parse_action_from_label(label: str | None) -> str | None:
    if not label:
        return None
    if "·" in label:
        return label.split("·", 1)[1].strip().lower()
    return None


def _browser_action_display(action: str, args: dict[str, Any] | None = None) -> tuple[str, str]:
    args = args or {}
    action = (action or "").lower()
    if action == "navigate":
        return ("Tarayıcı · Aç", f"{_short_url(args.get('url'))} açılıyor")
    if action == "click":
        selector = args.get("selector")
        return ("Tarayıcı · Tıkla", f"{selector} tıklanıyor" if selector else "Sayfaya tıklanıyor")
    if action == "type":
        selector = args.get("selector")
        return (
            "Tarayıcı · Yaz",
            f"{selector} alanına yazılıyor" if selector else "Metin yazılıyor",
        )
    if action == "scroll":
        return ("Tarayıcı · Kaydır", "Sayfa kaydırılıyor")
    if action == "screenshot":
        return ("Tarayıcı · Görüntü Al", "Ekran görüntüsü alınıyor")
    if action == "back":
        return ("Tarayıcı · Geri", "Önceki sayfaya dönülüyor")
    if action == "forward":
        return ("Tarayıcı · İleri", "Sonraki sayfaya gidiliyor")
    if action == "new_tab":
        return ("Tarayıcı · Yeni Sekme", "Yeni sekme açılıyor")
    if action == "close_tab":
        return ("Tarayıcı · Sekmeyi Kapat", "Sekme kapatılıyor")
    if action == "get_text":
        return ("Tarayıcı · Metin Al", "Sayfadan metin okunuyor")
    if action == "get_html":
        return ("Tarayıcı · HTML Al", "Sayfa kaynağı alınıyor")
    return ("Tarayıcı · İşlem", "Tarayıcı işlemi hazırlanıyor")


def _tool_result_summary(event: dict[str, Any]) -> str:
    tool = str(event.get("tool", ""))
    metadata = event.get("metadata", {}) if isinstance(event.get("metadata"), dict) else {}
    action = _parse_action_from_label(str(metadata.get("frame_label") or ""))
    if tool == "browser":
        if not event.get("success"):
            return _trim_text(event.get("result") or event.get("error") or "Tarayıcı işlemi başarısız oldu.")
        mapping = {
            "navigate": "Sayfa açıldı",
            "click": "Tıklama tamamlandı",
            "type": "Yazı girişi tamamlandı",
            "scroll": "Kaydırma tamamlandı",
            "screenshot": "Ekran görüntüsü alındı",
            "back": "Geri dönüldü",
            "forward": "İleri gidildi",
            "new_tab": "Yeni sekme açıldı",
            "close_tab": "Sekme kapatıldı",
        }
        return mapping.get(action or "", _trim_text(event.get("result") or "Tarayıcı işlemi tamamlandı"))
    if event.get("success"):
        return _trim_text(event.get("result") or f"{tool} işlemi tamamlandı")
    return _trim_text(event.get("result") or event.get("error") or f"{tool} işlemi başarısız oldu")


def _display_label_for_event(event: dict[str, Any]) -> str:
    event_type = str(event.get("type", "event"))
    if event_type in {"tool_call", "visual_intent"}:
        tool = str(event.get("tool", ""))
        args = event.get("args", {}) if isinstance(event.get("args"), dict) else {}
        if tool == "browser":
            return _browser_action_display(str(args.get("action", "")), args)[0]
        return _generic_tool_label(tool)
    if event_type == "screenshot":
        return _display_label_for_frame(event)
    if event_type == "tool_result":
        return _generic_tool_label(str(event.get("tool", "Araç")))
    if event_type == "phase":
        return "Düşünüyor" if event.get("phase") == "thinking" else "Hazırlanıyor"
    if event_type == "thought":
        return "Düşünce Özeti"
    if event_type == "done":
        return "Görev Tamamlandı"
    if event_type == "error":
        return "Bir Hata Oluştu"
    return event_type.replace("_", " ").title()


def _display_summary_for_event(event: dict[str, Any]) -> str:
    event_type = str(event.get("type", "event"))
    if event_type in {"tool_call", "visual_intent"}:
        tool = str(event.get("tool", ""))
        args = event.get("args", {}) if isinstance(event.get("args"), dict) else {}
        if tool == "browser":
            return _browser_action_display(str(args.get("action", "")), args)[1]
        return _trim_text(event.get("summary") or f"{tool} aracı hazırlanıyor")
    if event_type == "screenshot":
        return _display_summary_for_frame(event)
    if event_type == "tool_result":
        return _tool_result_summary(event)
    if event_type == "phase":
        return str(event.get("summary") or event.get("content") or "İşlem hazırlanıyor...")
    if event_type == "thought":
        return _trim_text(event.get("summary") or event.get("content") or "")
    if event_type in {"done", "error"}:
        return _trim_text(event.get("content") or "")
    return _trim_text(event.get("summary") or event.get("content") or event.get("result") or "")


def _display_label_for_frame(frame: dict[str, Any]) -> str:
    action = _parse_action_from_label(str(frame.get("label") or frame.get("frame_label") or ""))
    if action:
        return _browser_action_display(action)[0]
    return "Görsel Kare"


def _display_summary_for_frame(frame: dict[str, Any]) -> str:
    action = _parse_action_from_label(str(frame.get("label") or frame.get("frame_label") or ""))
    if action:
        return _browser_action_display(action)[1]
    return _trim_text(frame.get("summary") or "Görsel güncelleme")


def _json_detail(value: Any) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False)


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
                "detail": _json_detail(event.get("args", {})),
                "image_url": None,
            }
            steps.append(current_tool_step)
            continue
        if event_type == "visual_intent":
            if current_tool_step and current_tool_step.get("tool") == event.get("tool"):
                current_tool_step["status_label"] = "hazırlanıyor"
                current_tool_step["summary"] = str(event.get("display_summary") or current_tool_step["summary"])
                current_tool_step["focus_x"] = event.get("focus_x")
                current_tool_step["focus_y"] = event.get("focus_y")
            continue
        if event_type == "screenshot":
            if current_tool_step is not None:
                current_tool_step["image_url"] = event.get("image_url")
                current_tool_step["frame_id"] = event.get("frame_id")
            else:
                steps.append(
                    {
                        "id": f"step-{index + 1:03d}",
                        "kind": "visual",
                        "tone": "neutral",
                        "title": str(event.get("display_label") or "Görsel Kare"),
                        "summary": str(event.get("display_summary") or ""),
                        "status_label": "görsel",
                        "timestamp": timestamp,
                        "image_url": event.get("image_url"),
                    }
                )
            continue
        if event_type == "tool_result":
            if current_tool_step and current_tool_step.get("tool") == event.get("tool"):
                current_tool_step["tone"] = "success" if event.get("success") else "error"
                current_tool_step["status_label"] = "tamamlandı" if event.get("success") else "hata"
                current_tool_step["summary"] = str(event.get("display_summary") or current_tool_step["summary"])
                current_tool_step["detail"] = str(event.get("result") or "")
                current_tool_step["finished_at"] = timestamp
                current_tool_step = None
            else:
                steps.append(
                    {
                        "id": f"step-{index + 1:03d}",
                        "kind": "result",
                        "tone": "success" if event.get("success") else "error",
                        "title": str(event.get("display_label") or _generic_tool_label(str(event.get("tool", "")))),
                        "summary": str(event.get("display_summary") or ""),
                        "status_label": "tamamlandı" if event.get("success") else "hata",
                        "timestamp": timestamp,
                        "detail": str(event.get("result") or ""),
                    }
                )
            continue
        if event_type == "done":
            current_tool_step = None
            steps.append(
                {
                    "id": f"step-{index + 1:03d}",
                    "kind": "done",
                    "tone": "success",
                    "title": "Görev Tamamlandı",
                    "summary": str(event.get("content", "")),
                    "status_label": "tamamlandı",
                    "timestamp": timestamp,
                }
            )
            continue
        if event_type == "error":
            current_tool_step = None
            steps.append(
                {
                    "id": f"step-{index + 1:03d}",
                    "kind": "error",
                    "tone": "error",
                    "title": "Bir Hata Oluştu",
                    "summary": str(event.get("content", "")),
                    "status_label": "hata",
                    "timestamp": timestamp,
                }
            )

    return steps[-24:]


def create_live_app(
    base_config: AgentConfig,
    agent_factory: AgentFactory | None = None,
):
    """Create the FastAPI application for the live operator UI."""
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse

    app = FastAPI(title="Agentra App")
    manager = ThreadManager(base_config=base_config, agent_factory=agent_factory)
    app.state.manager = manager

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        boot = {
            "provider": base_config.llm_provider,
            "model": base_config.llm_model,
            "activeRunId": manager.active_run_id,
        }
        return HTMLResponse(_render_app_html(boot))

    @app.post("/runs")
    async def create_run(request: RunCreateRequest) -> JSONResponse:
        try:
            snapshot = await manager.start_run(request)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Thread not found.") from exc
        return JSONResponse(snapshot)

    @app.get("/runs/{run_id}")
    async def get_run(run_id: str) -> JSONResponse:
        try:
            session = manager.get_session(run_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Run not found.") from exc
        return JSONResponse(manager.snapshot_for_http(session))

    @app.post("/runs/{run_id}/stop")
    async def stop_run(run_id: str) -> JSONResponse:
        try:
            snapshot = await manager.stop_run(run_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Run not found.") from exc
        return JSONResponse(snapshot)

    @app.get("/runs/{run_id}/events")
    async def stream_events(run_id: str) -> StreamingResponse:
        try:
            session = manager.get_session(run_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Run not found.") from exc

        async def event_stream():
            yield _sse_payload({"kind": "snapshot", "snapshot": manager.snapshot_for_http(session)})
            if session.completed.is_set():
                yield _sse_payload({"kind": "complete", "status": session.status})
                return

            queue = manager.subscribe(run_id)
            try:
                while True:
                    try:
                        payload = await asyncio.wait_for(queue.get(), timeout=5.0)
                    except asyncio.TimeoutError:
                        if session.completed.is_set():
                            yield _sse_payload({"kind": "complete", "status": session.status})
                            return
                        yield ": keepalive\n\n"
                        continue
                    yield _sse_payload(payload)
                    if payload.get("kind") == "complete":
                        return
            finally:
                manager.unsubscribe(run_id, queue)

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/runs/{run_id}/assets/{asset_name}")
    async def get_asset(run_id: str, asset_name: str) -> FileResponse:
        try:
            session = manager.get_session(run_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Run not found.") from exc
        path = session.report.assets_dir / Path(asset_name).name
        if not path.exists():
            raise HTTPException(status_code=404, detail="Asset not found.")
        return FileResponse(path)

    @app.get("/runs/{run_id}/report")
    async def get_report(run_id: str) -> FileResponse:
        try:
            session = manager.get_session(run_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Run not found.") from exc
        return FileResponse(session.report.html_path)

    @app.get("/threads")
    async def list_threads() -> JSONResponse:
        return JSONResponse({"threads": manager.list_threads_for_http()})

    @app.get("/threads/{thread_id}")
    async def get_thread(thread_id: str) -> JSONResponse:
        try:
            thread = manager.get_thread(thread_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Thread not found.") from exc
        return JSONResponse(manager.thread_snapshot_for_http(thread))

    @app.post("/threads/{thread_id}/pause")
    async def pause_thread(thread_id: str) -> JSONResponse:
        try:
            snapshot = await manager.pause_thread(thread_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Thread not found.") from exc
        return JSONResponse(snapshot)

    @app.post("/threads/{thread_id}/resume")
    async def resume_thread(thread_id: str) -> JSONResponse:
        try:
            snapshot = await manager.resume_thread(thread_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Thread not found.") from exc
        return JSONResponse(snapshot)

    @app.post("/threads/{thread_id}/approvals/{request_id}")
    async def respond_approval(thread_id: str, request_id: str, payload: ApprovalDecisionRequest) -> JSONResponse:
        try:
            snapshot = await manager.respond_to_approval(
                thread_id,
                request_id,
                approved=payload.approved,
                note=payload.note,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Approval request not found.") from exc
        return JSONResponse(snapshot)

    @app.post("/threads/{thread_id}/questions/{request_id}")
    async def answer_question(thread_id: str, request_id: str, payload: UserAnswerRequest) -> JSONResponse:
        try:
            snapshot = await manager.answer_question(thread_id, request_id, payload.answer)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Question request not found.") from exc
        return JSONResponse(snapshot)

    @app.post("/threads/{thread_id}/actions")
    async def human_action(thread_id: str, payload: HumanActionRequest) -> JSONResponse:
        try:
            snapshot = await manager.human_action(thread_id, payload.tool, payload.args)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Thread not found.") from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return JSONResponse(snapshot)

    return app


def open_live_app(url: str) -> None:
    """Open the live app in the user's browser after the server starts."""
    threading.Timer(0.6, lambda: webbrowser.open(url, new=1)).start()


def _default_agent_factory(config: AgentConfig):
    from agentra.agents.autonomous import AutonomousAgent

    return AutonomousAgent(config=config)


def _sse_payload(payload: EventPayload) -> str:
    return f"data: {json.dumps(payload)}\n\n"


def _render_app_html(boot: dict[str, Any]) -> str:
    boot_json = json.dumps(boot).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="tr">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Agentra</title>
</head>
<body>
  <div id="app"></div>
  <script>window.__AGENTRA_BOOT__ = {boot_json};</script>
  <script>{_app_script()}</script>
  <style>{_app_styles()}</style>
</body>
</html>"""


def _app_styles() -> str:
    return _app_styles_base() + _app_styles_tv() + _app_styles_side()


def _app_script() -> str:
    return _app_script_state() + _app_script_render() + _app_script_mount()


def _app_styles_base() -> str:
    return """
:root {
  --text: #111827;
  --muted: rgba(229, 238, 255, 0.76);
  --line: rgba(255, 255, 255, 0.28);
  --panel: rgba(255, 255, 255, 0.16);
  --shadow: 0 22px 56px rgba(32, 66, 132, 0.14);
}
* { box-sizing: border-box; }
body {
  margin: 0;
  min-height: 100vh;
  color: var(--text);
  font-family: "Aptos", "Segoe UI Variable", "Segoe UI", sans-serif;
  background:
    radial-gradient(circle at 12% 14%, rgba(220, 243, 255, 0.95), transparent 22%),
    radial-gradient(circle at 19% 76%, rgba(214, 201, 255, 0.24), transparent 16%),
    radial-gradient(circle at 72% 32%, rgba(134, 208, 255, 0.18), transparent 20%),
    linear-gradient(90deg, #d6ebfb 0%, #bddcff 31%, #4d9af0 63%, #1660d6 100%);
}
button, input, textarea, summary, select { font: inherit; }
a { color: inherit; text-decoration: none; }
.page {
  min-height: 100vh;
  display: grid;
  grid-template-rows: 62px minmax(0, 1fr);
}
.topbar {
  display: flex;
  align-items: center;
  justify-content: flex-start;
  gap: 14px;
  padding: 0 18px;
  background: rgba(255,255,255,0.96);
  border-bottom: 1px solid rgba(117, 148, 209, 0.14);
}
.topbar-left {
  display: flex;
  align-items: center;
  gap: 14px;
}
.nav-button,
.stage-close {
  display: inline-grid;
  place-items: center;
  width: 34px;
  height: 34px;
  border: 0;
  border-radius: 999px;
  background: transparent;
  color: rgba(24, 36, 64, 0.74);
  font-size: 28px;
  line-height: 1;
  cursor: pointer;
}
.title-stack {
  display: grid;
  gap: 2px;
}
.app-title {
  font-size: 16px;
  font-weight: 700;
  color: #101828;
}
.app-subtitle {
  font-size: 12px;
  color: rgba(16, 24, 40, 0.64);
}
.workspace {
  display: grid;
  grid-template-columns: minmax(430px, 40%) minmax(0, 1fr);
  min-height: 0;
}
.left-pane {
  position: relative;
  border-right: 1px solid rgba(255, 255, 255, 0.26);
  background: linear-gradient(180deg, rgba(255,255,255,0.08), rgba(255,255,255,0.04));
  min-height: 0;
}
.left-pane::before {
  content: none;
}
.left-pane-inner {
  position: relative;
  z-index: 1;
  display: flex;
  flex-direction: column;
  gap: 12px;
  height: 100%;
  padding: 18px;
  overflow: auto;
}
.console-section {
  display: grid;
  gap: 10px;
  padding: 14px;
  border-radius: 18px;
  background: rgba(255, 255, 255, 0.16);
  border: 1px solid rgba(255, 255, 255, 0.34);
  box-shadow: 0 10px 32px rgba(34, 70, 148, 0.1);
  backdrop-filter: blur(14px);
}
.section-title {
  font-size: 12px;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: rgba(18, 32, 61, 0.72);
}
.field-label {
  font-size: 12px;
  font-weight: 600;
  color: rgba(19, 31, 57, 0.78);
}
.text-input,
.text-area,
.mini-input,
.json-input {
  width: 100%;
  border: 1px solid rgba(103, 132, 194, 0.22);
  border-radius: 12px;
  background: rgba(255, 255, 255, 0.42);
  color: #132241;
  padding: 10px 12px;
  outline: none;
}
.text-input::placeholder,
.text-area::placeholder,
.mini-input::placeholder,
.json-input::placeholder {
  color: rgba(19, 34, 65, 0.48);
}
.text-area {
  min-height: 92px;
  resize: vertical;
  line-height: 1.45;
}
.mini-input {
  min-height: 38px;
}
.json-input {
  min-height: 110px;
  resize: vertical;
  font-family: "Cascadia Code", "Consolas", monospace;
  font-size: 12px;
}
.action-row {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}
.action-button {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  min-height: 36px;
  padding: 8px 12px;
  border-radius: 11px;
  border: 1px solid rgba(106, 130, 186, 0.2);
  background: rgba(255, 255, 255, 0.34);
  color: #132241;
  cursor: pointer;
  transition: transform 120ms ease, background 120ms ease;
}
.action-button:hover:not(:disabled) {
  transform: translateY(-1px);
  background: rgba(255, 255, 255, 0.46);
}
.action-button:disabled {
  opacity: 0.45;
  cursor: not-allowed;
}
.action-button.primary {
  color: white;
  border-color: rgba(73, 106, 195, 0.6);
  background: linear-gradient(135deg, rgba(45, 86, 189, 0.96), rgba(82, 103, 240, 0.92));
}
.action-button.warn {
  color: #8a3145;
  background: rgba(255, 219, 226, 0.62);
}
.action-button.success {
  color: #136853;
  background: rgba(210, 255, 236, 0.74);
}
.inline-error {
  min-height: 15px;
  font-size: 12px;
  color: #a53a49;
}
.empty-state {
  padding: 10px 12px;
  border-radius: 12px;
  background: rgba(255, 255, 255, 0.2);
  border: 1px dashed rgba(112, 128, 187, 0.2);
  color: rgba(20, 33, 61, 0.65);
  font-size: 13px;
}
.thread-list,
.request-list,
.history-list {
  display: grid;
  gap: 8px;
}
.thread-item {
  width: 100%;
  display: grid;
  gap: 6px;
  padding: 10px 12px;
  border-radius: 14px;
  border: 1px solid rgba(112, 128, 187, 0.2);
  background: rgba(255, 255, 255, 0.28);
  text-align: left;
  cursor: pointer;
}
.thread-item.active {
  border-color: rgba(69, 112, 224, 0.52);
  background: rgba(235, 244, 255, 0.66);
  box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.44);
}
.thread-item-head,
.history-item-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
}
.thread-title,
.request-title {
  font-size: 14px;
  font-weight: 700;
  color: #11203c;
}
.thread-summary,
.history-summary,
.request-text,
.helper-text {
  font-size: 13px;
  line-height: 1.45;
  color: rgba(17, 32, 60, 0.78);
}
.badge-row {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}
.detail-grid {
  display: grid;
  gap: 8px;
}
.detail-row {
  display: grid;
  gap: 4px;
}
.detail-key {
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: rgba(20, 33, 61, 0.56);
}
.path-copy {
  font-size: 12px;
  line-height: 1.4;
  font-family: "Cascadia Code", "Consolas", monospace;
  color: rgba(17, 32, 60, 0.82);
  word-break: break-all;
}
.request-card,
.history-item,
.thread-detail-card {
  display: grid;
  gap: 8px;
  padding: 10px 12px;
  border-radius: 14px;
  border: 1px solid rgba(112, 128, 187, 0.2);
  background: rgba(255, 255, 255, 0.28);
}
.request-meta,
.history-meta {
  font-size: 12px;
  color: rgba(17, 32, 60, 0.6);
}
.inline-form,
.manual-grid {
  display: grid;
  gap: 8px;
}
.inline-form {
  grid-template-columns: minmax(0, 1fr) auto;
}
.manual-row {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 8px;
}
.details-shell {
  display: grid;
  gap: 8px;
  padding-top: 4px;
}
details > summary {
  cursor: pointer;
  color: rgba(20, 33, 61, 0.72);
  font-size: 13px;
  font-weight: 600;
}
.right-pane {
  position: relative;
  padding: 18px 24px 18px 20px;
}
.right-pane::before {
  content: "";
  position: absolute;
  inset: 0;
  background:
    radial-gradient(circle at 26% 14%, rgba(255,255,255,0.16), transparent 14%),
    linear-gradient(135deg, rgba(102, 176, 255, 0.12), rgba(23, 92, 221, 0.12));
  pointer-events: none;
}
.stage-shell {
  position: relative;
  z-index: 1;
  display: grid;
  grid-template-rows: auto 1fr;
  gap: 12px;
  height: 100%;
}
.stage-head {
  display: flex;
  align-items: center;
  justify-content: flex-start;
  color: rgba(255,255,255,0.92);
  padding: 2px 4px 0 4px;
}
.stage-title {
  font-size: 15px;
  font-weight: 700;
}
.tv-stage {
  min-height: 0;
  display: grid;
  place-items: start center;
  padding: 10px 4px 8px;
}
.muted { color: var(--muted); }
@media (max-width: 1080px) {
  .workspace { grid-template-columns: 1fr; }
  .left-pane { border-right: 0; border-bottom: 1px solid rgba(255,255,255,0.28); }
  .tv-stage { padding-top: 0; }
  .left-pane-inner { max-height: 50vh; }
}
@media (max-width: 760px) {
  .manual-row,
  .inline-form {
    grid-template-columns: 1fr;
  }
}
"""


def _app_styles_tv() -> str:
    return """
.tv-shell {
  width: min(100%, 920px);
  background: linear-gradient(135deg, rgba(255,255,255,0.12), rgba(151, 112, 255, 0.18));
  border: 1px solid rgba(255,255,255,0.22);
  border-radius: 26px;
  padding: 0 0 14px;
  box-shadow: 0 26px 56px rgba(19, 43, 108, 0.18);
  backdrop-filter: blur(16px);
}
.tv-screen {
  position: relative;
  aspect-ratio: 16 / 9;
  overflow: hidden;
  border-radius: 26px 26px 20px 20px;
  background:
    linear-gradient(180deg, rgba(255,255,255,0.96), rgba(242,246,252,0.94) 7%, rgba(29,41,69,0.98) 7%, rgba(29,41,69,0.98) 100%);
}
.tv-screen.busy::after {
  content: "";
  position: absolute;
  inset: 42px 0 0;
  background:
    linear-gradient(180deg, rgba(255,255,255,0.0), rgba(255,255,255,0.08), rgba(255,255,255,0.0)),
    repeating-linear-gradient(
      180deg,
      rgba(255,255,255,0.02) 0px,
      rgba(255,255,255,0.02) 18px,
      rgba(255,255,255,0.0) 18px,
      rgba(255,255,255,0.0) 36px
    );
  animation: scan 2.6s linear infinite;
  pointer-events: none;
}
@keyframes scan {
  0% { transform: translateY(-40%); opacity: 0.35; }
  100% { transform: translateY(120%); opacity: 0.15; }
}
.tv-title {
  text-align: center;
  color: #6d7b90;
  font-size: 12px;
  padding: 8px 0 12px;
}
.tv-image {
  width: 100%;
  height: calc(100% - 42px);
  object-fit: contain;
  display: none;
  border-bottom-left-radius: 18px;
  border-bottom-right-radius: 18px;
  transition: opacity 180ms ease, transform 220ms ease;
}
.tv-image.swapping {
  opacity: 0.28;
  transform: scale(0.992);
}
.tv-empty {
  position: absolute;
  inset: 56px 24px 24px;
  display: grid;
  place-items: center;
  text-align: center;
  color: #d6e2fb;
  font-size: 16px;
  line-height: 1.5;
  text-shadow: 0 1px 12px rgba(7, 13, 26, 0.28);
}
.tv-empty .muted {
  color: rgba(221, 232, 255, 0.82);
}
.cursor-dot {
  position: absolute;
  width: 16px;
  height: 16px;
  border-radius: 999px;
  background: white;
  border: 3px solid #151d34;
  box-shadow: 0 0 0 5px rgba(255,255,255,0.14), 0 10px 24px rgba(0, 0, 0, 0.28);
  transform: translate(-50%, -50%);
  display: none;
  transition: left 220ms ease, top 220ms ease, opacity 180ms ease;
}
.cursor-dot.pending {
  animation: pulse 1.25s ease-in-out infinite;
}
@keyframes pulse {
  0%, 100% { box-shadow: 0 0 0 5px rgba(255,255,255,0.12), 0 10px 24px rgba(0,0,0,0.28); }
  50% { box-shadow: 0 0 0 9px rgba(255,255,255,0.22), 0 10px 24px rgba(0,0,0,0.22); }
}
.cursor-bubble {
  position: absolute;
  max-width: min(360px, 58%);
  padding: 13px 15px;
  border-radius: 18px;
  background: rgba(255,255,255,0.94);
  color: #172033;
  line-height: 1.45;
  box-shadow: 0 18px 38px rgba(0,0,0,0.18);
  transform: translate(-14%, 18px);
  display: none;
  transition: left 220ms ease, top 220ms ease, opacity 180ms ease;
}
.cursor-bubble.flip {
  transform: translate(-14%, calc(-100% - 24px));
}
.tv-footer {
  display: grid;
  gap: 0;
  padding: 8px 12px 0;
}
.scrub-row {
  display: grid;
  grid-template-columns: auto 1fr auto auto;
  gap: 8px;
  align-items: center;
  color: white;
}
.scrub-meta {
  font-size: 12px;
  font-weight: 400;
  color: rgba(255,255,255,0.92);
}
.timeline-track {
  position: relative;
  height: 8px;
  border-radius: 999px;
  cursor: pointer;
  background: linear-gradient(180deg, rgba(255,255,255,0.38), rgba(255,255,255,0.14));
  border: 1px solid rgba(255,255,255,0.58);
  box-shadow:
    inset 0 1px 0 rgba(255,255,255,0.42),
    0 0 26px rgba(255,255,255,0.18),
    0 8px 24px rgba(0,0,0,0.16);
}
.timeline-track::after {
  content: "";
  position: absolute;
  inset: 1px;
  border-radius: 999px;
  background: linear-gradient(180deg, rgba(255,255,255,0.44), rgba(255,255,255,0.04));
  opacity: 0.72;
}
.timeline-progress {
  position: absolute;
  left: 1px;
  top: 1px;
  bottom: 1px;
  width: 0%;
  border-radius: 999px;
  background: linear-gradient(90deg, rgba(255,255,255,0.98), rgba(221,239,255,0.98));
  box-shadow: 0 0 18px rgba(255,255,255,0.42), 0 0 34px rgba(190,225,255,0.24);
  z-index: 1;
}
.timeline-handle {
  position: absolute;
  top: 50%;
  left: 0%;
  width: 16px;
  height: 16px;
  border-radius: 999px;
  border: 1px solid rgba(255,255,255,0.98);
  background: radial-gradient(circle at 35% 35%, #ffffff 0%, #f4faff 45%, #d8e9ff 100%);
  box-shadow:
    0 0 0 2px rgba(255,255,255,0.14),
    0 0 12px rgba(255,255,255,0.4),
    0 5px 14px rgba(0, 0, 0, 0.14);
  transform: translate(-50%, -50%);
  pointer-events: none;
  z-index: 2;
}
"""


def _app_styles_side() -> str:
    return """
.status-pill,
.thread-chip,
.count-badge {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-width: 70px;
  padding: 5px 9px;
  border-radius: 999px;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  font-size: 11px;
}
.status-pill.idle,
.status-pill.partial,
.thread-chip.idle,
.thread-chip.partial { color: #537ab7; background: rgba(130, 176, 255, 0.18); }
.status-pill.running,
.thread-chip.running,
.thread-chip.blocked_waiting_user { color: #9f7722; background: rgba(255, 213, 110, 0.22); }
.status-pill.completed,
.thread-chip.completed { color: #1f8e73; background: rgba(101, 220, 176, 0.22); }
.status-pill.error,
.thread-chip.error { color: #b9485b; background: rgba(255, 154, 166, 0.24); }
.thread-chip.paused_for_user { color: #6644ba; background: rgba(212, 194, 255, 0.3); }
.count-badge {
  min-width: 0;
  color: rgba(20, 32, 58, 0.72);
  background: rgba(255,255,255,0.42);
  border: 1px solid rgba(112, 128, 187, 0.16);
}
"""


def _app_script_state() -> str:
    return """
const boot = window.__AGENTRA_BOOT__ || {};
const DEFAULT_FOCUS = { x: 0.74, y: 0.2 };

const state = {
  runId: boot.activeRunId || null,
  goal: "",
  provider: boot.provider || "",
  model: boot.model || "",
  status: "idle",
  events: [],
  frames: [],
  steps: [],
  liveMode: true,
  selectedFrameId: null,
  reportUrl: null,
  source: null,
  scrubPercent: 1,
  scrubDrag: null,
  activity: {
    mode: "idle",
    summary: "",
    focus_x: DEFAULT_FOCUS.x,
    focus_y: DEFAULT_FOCUS.y,
  },
};

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function shortText(value, limit = 220) {
  const text = String(value ?? "").replace(/\\s+/g, " ").trim();
  if (text.length <= limit) return text;
  return `${text.slice(0, limit - 3).trimEnd()}...`;
}

function statusLabel(status) {
  if (status === "running") return "ÇALIŞIYOR";
  if (status === "completed") return "TAMAMLANDI";
  if (status === "error") return "HATA";
  if (status === "partial") return "KISMİ";
  return "HAZIR";
}

function localizedLabel(event) {
  return event.display_label || event.frame_label || event.label || event.tool || "İşlem";
}

function localizedSummary(event) {
  return event.display_summary || event.summary || event.content || event.result || "";
}

function frameFromEvent(event) {
  if (event.type !== "screenshot" || !event.image_url) return null;
  return {
    id: event.frame_id || `frame-${Date.now()}`,
    timestamp: event.timestamp || "",
    label: event.label || event.frame_label || "Görsel Kare",
    display_label: event.display_label || event.frame_label || "Görsel Kare",
    summary: event.summary || "Görsel güncelleme",
    display_summary: event.display_summary || event.summary || "Görsel güncelleme",
    image_path: event.image_path || "",
    image_url: event.image_url,
    focus_x: typeof event.focus_x === "number" ? event.focus_x : null,
    focus_y: typeof event.focus_y === "number" ? event.focus_y : null,
  };
}

function currentFrame() {
  if (!state.frames.length) return null;
  if (state.liveMode || !state.selectedFrameId) return state.frames[state.frames.length - 1];
  return state.frames.find((frame) => frame.id === state.selectedFrameId) || state.frames[state.frames.length - 1];
}

function buildSteps(events) {
  const steps = [];
  let currentTool = null;

  events.forEach((event, index) => {
    const timestamp = event.timestamp || "";
    const type = event.type;

    if (type === "phase") return;

    if (type === "thought") {
      currentTool = null;
      steps.push({
        id: `step-${index + 1}`,
        kind: "thought",
        tone: "assistant",
        title: localizedLabel(event) || "Düşünce özeti",
        summary: shortText(localizedSummary(event), 260),
        status_label: "düşünüyor",
        timestamp,
      });
      return;
    }

    if (type === "tool_call") {
      currentTool = {
        id: `step-${index + 1}`,
        kind: "tool",
        tone: "pending",
        tool: event.tool,
        title: localizedLabel(event),
        summary: shortText(localizedSummary(event), 220),
        status_label: "uyguluyor",
        timestamp,
        detail: JSON.stringify(event.args || {}, null, 2),
        image_url: null,
      };
      steps.push(currentTool);
      return;
    }

    if (type === "visual_intent") {
      if (currentTool && currentTool.tool === event.tool) {
        currentTool.summary = shortText(localizedSummary(event), 220);
        currentTool.status_label = "hazırlanıyor";
        currentTool.focus_x = event.focus_x;
        currentTool.focus_y = event.focus_y;
      }
      return;
    }

    if (type === "screenshot") {
      if (currentTool) {
        currentTool.image_url = event.image_url || currentTool.image_url;
        currentTool.frame_id = event.frame_id || currentTool.frame_id;
      } else {
        steps.push({
          id: `step-${index + 1}`,
          kind: "visual",
          tone: "neutral",
          title: localizedLabel(event),
          summary: shortText(localizedSummary(event), 220),
          status_label: "görsel",
          timestamp,
          image_url: event.image_url,
        });
      }
      return;
    }

    if (type === "tool_result") {
      if (currentTool && currentTool.tool === event.tool) {
        currentTool.tone = event.success ? "success" : "error";
        currentTool.status_label = event.success ? "tamamlandı" : "hata";
        currentTool.summary = shortText(localizedSummary(event), 220);
        currentTool.detail = String(event.result || "");
        currentTool = null;
      } else {
        steps.push({
          id: `step-${index + 1}`,
          kind: "result",
          tone: event.success ? "success" : "error",
          title: localizedLabel(event),
          summary: shortText(localizedSummary(event), 220),
          status_label: event.success ? "tamamlandı" : "hata",
          timestamp,
          detail: String(event.result || ""),
        });
      }
      return;
    }

    if (type === "done") {
      currentTool = null;
      steps.push({
        id: `step-${index + 1}`,
        kind: "done",
        tone: "success",
        title: "Görev Tamamlandı",
        summary: String(event.content || ""),
        status_label: "tamamlandı",
        timestamp,
      });
      return;
    }

    if (type === "error") {
      currentTool = null;
      steps.push({
        id: `step-${index + 1}`,
        kind: "error",
        tone: "error",
        title: "Bir Hata Oluştu",
        summary: shortText(event.content || "", 260),
        status_label: "hata",
        timestamp,
      });
    }
  });

  return steps.slice(-24);
}

function deriveActivity(events, frames) {
  const latestFrame = frames.length ? frames[frames.length - 1] : null;

  for (let index = events.length - 1; index >= 0; index -= 1) {
    const event = events[index];
    if (event.type === "phase") {
      return {
        mode: event.phase || "thinking",
        summary: localizedSummary(event),
        focus_x: latestFrame && typeof latestFrame.focus_x === "number" ? latestFrame.focus_x : DEFAULT_FOCUS.x,
        focus_y: latestFrame && typeof latestFrame.focus_y === "number" ? latestFrame.focus_y : DEFAULT_FOCUS.y,
      };
    }
    if (event.type === "visual_intent") {
      return {
        mode: "acting",
        summary: localizedSummary(event),
        focus_x: typeof event.focus_x === "number" ? event.focus_x : DEFAULT_FOCUS.x,
        focus_y: typeof event.focus_y === "number" ? event.focus_y : DEFAULT_FOCUS.y,
      };
    }
    if (event.type === "tool_call") {
      return {
        mode: "acting",
        summary: localizedSummary(event),
        focus_x: latestFrame && typeof latestFrame.focus_x === "number" ? latestFrame.focus_x : DEFAULT_FOCUS.x,
        focus_y: latestFrame && typeof latestFrame.focus_y === "number" ? latestFrame.focus_y : DEFAULT_FOCUS.y,
      };
    }
    if (event.type === "screenshot") {
      return {
        mode: "acting",
        summary: localizedSummary(event),
        focus_x: typeof event.focus_x === "number" ? event.focus_x : DEFAULT_FOCUS.x,
        focus_y: typeof event.focus_y === "number" ? event.focus_y : DEFAULT_FOCUS.y,
      };
    }
    if (event.type === "tool_result") {
      return {
        mode: "idle",
        summary: localizedSummary(event),
        focus_x: latestFrame && typeof latestFrame.focus_x === "number" ? latestFrame.focus_x : DEFAULT_FOCUS.x,
        focus_y: latestFrame && typeof latestFrame.focus_y === "number" ? latestFrame.focus_y : DEFAULT_FOCUS.y,
      };
    }
    if (event.type === "done" || event.type === "error") {
      return {
        mode: "idle",
        summary: localizedSummary(event),
        focus_x: latestFrame && typeof latestFrame.focus_x === "number" ? latestFrame.focus_x : DEFAULT_FOCUS.x,
        focus_y: latestFrame && typeof latestFrame.focus_y === "number" ? latestFrame.focus_y : DEFAULT_FOCUS.y,
      };
    }
  }

  return {
    mode: "idle",
    summary: "",
    focus_x: DEFAULT_FOCUS.x,
    focus_y: DEFAULT_FOCUS.y,
  };
}

function syncSelectionToFrames() {
  if (!state.frames.length) {
    state.selectedFrameId = null;
    state.scrubPercent = 1;
    return;
  }

  if (state.liveMode || !state.selectedFrameId) {
    state.selectedFrameId = state.frames[state.frames.length - 1].id;
  } else if (!state.frames.find((frame) => frame.id === state.selectedFrameId)) {
    state.selectedFrameId = state.frames[state.frames.length - 1].id;
    state.liveMode = true;
  }

  const index = Math.max(0, state.frames.findIndex((frame) => frame.id === state.selectedFrameId));
  state.scrubPercent = state.frames.length > 1 ? index / (state.frames.length - 1) : 1;
}

function applySnapshot(snapshot) {
  state.runId = snapshot.run_id;
  state.goal = snapshot.goal || state.goal;
  state.provider = snapshot.provider;
  state.model = snapshot.model;
  state.status = snapshot.status || state.status;
  state.events = snapshot.events || [];
  state.frames = snapshot.frames || [];
  state.steps = snapshot.steps || buildSteps(state.events);
  state.reportUrl = snapshot.report_url || null;
  syncSelectionToFrames();
  state.activity = deriveActivity(state.events, state.frames);
  render();
}

function applyEvent(event) {
  state.events = [...state.events, event];
  const frame = frameFromEvent(event);
  if (frame) {
    state.frames = [...state.frames, frame];
  }
  if (event.type === "done" && state.status === "running") state.status = "completed";
  if (event.type === "error") state.status = "error";
  state.steps = buildSteps(state.events);
  syncSelectionToFrames();
  state.activity = deriveActivity(state.events, state.frames);
  render();
}
"""


def _app_script_render() -> str:
    return """
function currentOverlay(frame) {
  if (state.status === "running" && (state.activity.mode === "thinking" || state.activity.mode === "acting")) {
    return {
      busy: true,
      summary: state.activity.summary || (frame ? frame.display_summary : ""),
      focus_x: typeof state.activity.focus_x === "number" ? state.activity.focus_x : (frame ? frame.focus_x : DEFAULT_FOCUS.x),
      focus_y: typeof state.activity.focus_y === "number" ? state.activity.focus_y : (frame ? frame.focus_y : DEFAULT_FOCUS.y),
    };
  }
  if (frame) {
    return {
      busy: false,
      summary: frame.display_summary || "",
      focus_x: typeof frame.focus_x === "number" ? frame.focus_x : DEFAULT_FOCUS.x,
      focus_y: typeof frame.focus_y === "number" ? frame.focus_y : DEFAULT_FOCUS.y,
    };
  }
  return {
    busy: false,
    summary: "",
    focus_x: DEFAULT_FOCUS.x,
    focus_y: DEFAULT_FOCUS.y,
  };
}

function projectStamp() {
  const dateLabel = new Intl.DateTimeFormat("en-US", {
    month: "short",
    day: "numeric",
  }).format(new Date());
  return `${dateLabel} · Otonom Ajan Projesi`;
}

function statusSentence() {
  if (state.status === "running") {
    if (state.activity.mode === "thinking") {
      return "Ajan Düşünüyor: Sonraki adımı planlıyor";
    }
    return "Ajan Çalışıyor: İşlemi sürdürüyor";
  }
  if (state.status === "completed") {
    return "Ajan Tamamlandı: Yeni komut bekleniyor";
  }
  if (state.status === "error") {
    return "Ajan Durduruldu: Bir hata oluştu";
  }
  if (state.status === "partial") {
    return "Ajan Beklemede: Yeni komut bekleniyor";
  }
  return "Ajan Beklemede: Yeni Komut Bekleniyor";
}

function renderTV() {
  const frame = currentFrame();
  const overlay = currentOverlay(frame);
  const screen = document.getElementById("tv-screen");
  const image = document.getElementById("tv-image");
  const empty = document.getElementById("tv-empty");
  const dot = document.getElementById("cursor-dot");
  const bubble = document.getElementById("cursor-bubble");
  const title = document.getElementById("tv-frame-title");

  screen.classList.toggle("busy", Boolean(overlay.busy));

  if (!frame) {
    image.style.display = "none";
    empty.style.display = "grid";
    title.textContent = "Agentra Canlı Kumanda";
    empty.innerHTML = overlay.busy
      ? `<div><strong>${escapeHtml(overlay.summary || "İşlem hazırlanıyor...")}</strong><div class="muted" style="margin-top:8px;">İlk görsel kare gelene kadar son durumu burada göstereceğim.</div></div>`
      : `<div><strong>Bir istem yaz ve çalıştır.</strong><div class="muted" style="margin-top:8px;">Canlı ekran görüntüleri, adım özetleri ve sürüklenebilir timeline burada görünecek.</div></div>`;
  } else {
    title.textContent = frame.display_label || frame.label || "Görsel Kare";
    empty.style.display = "none";
    image.style.display = "block";
    if (image.dataset.frameId !== frame.id) {
      image.classList.add("swapping");
      image.src = frame.image_url;
      image.dataset.frameId = frame.id;
      window.setTimeout(() => image.classList.remove("swapping"), 180);
    }
  }

  if (overlay.summary) {
    const left = `${Math.max(0, Math.min(100, (overlay.focus_x ?? DEFAULT_FOCUS.x) * 100))}%`;
    const top = `${Math.max(0, Math.min(100, (overlay.focus_y ?? DEFAULT_FOCUS.y) * 100))}%`;
    dot.style.display = "block";
    bubble.style.display = "block";
    dot.style.left = left;
    dot.style.top = top;
    bubble.style.left = left;
    bubble.style.top = top;
    dot.classList.toggle("pending", overlay.busy);
    bubble.classList.toggle("flip", (overlay.focus_y ?? DEFAULT_FOCUS.y) > 0.68);
    bubble.textContent = overlay.summary;
  } else {
    dot.style.display = "none";
    bubble.style.display = "none";
  }
}

function renderTimeline() {
  const frame = currentFrame();
  const frameIndex = frame ? Math.max(0, state.frames.findIndex((item) => item.id === frame.id)) : 0;
  const label = document.getElementById("scrubber-label");
  const progress = document.getElementById("timeline-progress");
  const handle = document.getElementById("timeline-handle");
  const percent = state.frames.length > 1 ? state.scrubPercent : 1;
  const percentText = `${Math.max(0, Math.min(100, percent * 100))}%`;

  label.textContent = state.frames.length ? `${frameIndex + 1} / ${state.frames.length}` : "0 / 0";
  progress.style.width = percentText;
  handle.style.left = percentText;
}

function renderSteps() {
  return;
}

function renderStatus() {
  document.getElementById("project-stamp").textContent = projectStamp();
  document.getElementById("status-pill").className = `status-pill ${state.status}`;
  document.getElementById("status-pill").textContent = statusLabel(state.status);
  document.getElementById("status-dot").className = `status-dot ${state.status}`;
  document.getElementById("status-copy").textContent = shortText(statusSentence(), 72);
  document.getElementById("tv-status").textContent = statusLabel(state.status);
  document.getElementById("agent-title").textContent = "Ajan";

  const goalCard = document.getElementById("goal-card");
  const goalText = document.getElementById("goal-preview");
  if (state.goal) {
    goalCard.classList.remove("hidden");
    goalText.textContent = state.goal;
  } else {
    goalCard.classList.add("hidden");
    goalText.textContent = "";
  }
}

function render() {
  renderStatus();
  renderTV();
  renderTimeline();
  renderSteps();
}

async function fetchJson(url, options = {}) {
  const response = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!response.ok) {
    let message = `${response.status}`;
    try {
      const payload = await response.json();
      message = payload.detail || message;
    } catch (error) {
      // Ignore JSON parse failures on empty responses.
    }
    throw new Error(message);
  }
  return response.json();
}

function selectFrameByPercent(percent) {
  if (!state.frames.length) return;
  const safe = Math.max(0, Math.min(1, percent));
  state.scrubPercent = safe;
  state.liveMode = false;
  const index = state.frames.length > 1 ? Math.round(safe * (state.frames.length - 1)) : 0;
  state.selectedFrameId = state.frames[index].id;
  render();
}
"""


def _app_script_mount() -> str:
    return """
function setupScrubber() {
  const track = document.getElementById("timeline-track");

  function updateFromClientX(clientX) {
    const rect = track.getBoundingClientRect();
    if (!rect.width) return;
    selectFrameByPercent((clientX - rect.left) / rect.width);
  }

  track.addEventListener("pointerdown", (event) => {
    if (!state.frames.length) return;
    state.scrubDrag = { pointerId: event.pointerId };
    track.setPointerCapture(event.pointerId);
    updateFromClientX(event.clientX);
  });

  track.addEventListener("pointermove", (event) => {
    if (!state.scrubDrag || state.scrubDrag.pointerId !== event.pointerId) return;
    const clientX = event.clientX;
    if (state.scrubDrag.raf) {
      state.scrubDrag.pendingX = clientX;
      return;
    }
    state.scrubDrag.pendingX = clientX;
    state.scrubDrag.raf = window.requestAnimationFrame(() => {
      updateFromClientX(state.scrubDrag.pendingX);
      state.scrubDrag.raf = null;
    });
  });

  function stopScrub(event) {
    if (!state.scrubDrag || state.scrubDrag.pointerId !== event.pointerId) return;
    if (state.scrubDrag.raf) window.cancelAnimationFrame(state.scrubDrag.raf);
    track.releasePointerCapture?.(event.pointerId);
    state.scrubDrag = null;
  }

  track.addEventListener("pointerup", stopScrub);
  track.addEventListener("pointercancel", stopScrub);
}

function setupRail() {
  return;
}

function connectStream() {
  if (!state.runId) return;
  if (state.source) state.source.close();
  state.source = new EventSource(`/runs/${state.runId}/events`);
  state.source.onmessage = (message) => {
    const payload = JSON.parse(message.data);
    if (payload.kind === "snapshot") {
      applySnapshot(payload.snapshot);
      return;
    }
    if (payload.kind === "event") {
      applyEvent(payload.event);
      return;
    }
    if (payload.kind === "status") {
      state.status = payload.status || state.status;
      applySnapshot(payload.snapshot);
      return;
    }
    if (payload.kind === "complete") {
      state.status = payload.status || state.status;
      state.activity = deriveActivity(state.events, state.frames);
      render();
      state.source.close();
      state.source = null;
    }
  };
}

async function startRun(event) {
  event.preventDefault();
  const input = document.getElementById("goal-input");
  const goal = input.value.trim();
  if (!goal) return;

  try {
    state.liveMode = true;
    const snapshot = await fetchJson("/runs", {
      method: "POST",
      body: JSON.stringify({ goal }),
    });
    applySnapshot(snapshot);
    connectStream();
  } catch (error) {
    state.status = "error";
    state.events = [{
      type: "error",
      timestamp: new Date().toISOString(),
      display_label: "Bir Hata Oluştu",
      display_summary: error.message,
      content: error.message,
    }];
    state.steps = buildSteps(state.events);
    state.activity = deriveActivity(state.events, state.frames);
    render();
  }
}

async function stopRun() {
  if (!state.runId) return;
  await fetchJson(`/runs/${state.runId}/stop`, { method: "POST" });
}

function mount() {
  document.getElementById("app").innerHTML = `
    <main class="page">
      <header class="topbar">
        <div class="topbar-left">
          <button class="nav-button" type="button" aria-label="Geri">←</button>
          <div class="title-stack">
            <div class="app-title">Otonom Ajan - Görev Paneli</div>
            <div class="app-subtitle" id="project-stamp">May 22 · Otonom Ajan Projesi</div>
          </div>
        </div>
      </header>
      <section class="workspace">
        <section class="left-pane">
          <div class="left-pane-inner">
            <div class="goal-card hidden" id="goal-card">
              <div id="goal-preview"></div>
            </div>
            <div class="status-row">
              <div class="status-banner">
                <div class="status-banner-left">
                  <span class="status-dot ${state.status}" id="status-dot"></span>
                  <span class="status-copy" id="status-copy">Ajan beklemede: yeni komut bekleniyor.</span>
                </div>
                <div class="status-actions">
                  <span class="status-pill ${state.status}" id="status-pill">${statusLabel(state.status)}</span>
                  <span class="status-dismiss" aria-hidden="true">×</span>
                </div>
              </div>
            </div>
            <div class="panel-spacer"></div>
            <div class="composer-shell">
              <form id="run-form" class="prompt-form">
                <button class="composer-action" type="button" aria-label="Ek">⌘</button>
                <textarea id="goal-input" class="prompt-field" placeholder="Yeni bir komut veya görev girin"></textarea>
                <button class="composer-action" type="button" aria-label="Mikrofon">◉</button>
                <button class="composer-send" type="submit" aria-label="Gönder" title="Gönder">➤</button>
              </form>
            </div>
          </div>
        </section>
        <section class="right-pane">
          <div class="stage-shell">
            <div class="stage-head">
              <div class="stage-title" id="agent-title">Ajan</div>
              <button class="stage-close" type="button" aria-label="Kapat">×</button>
            </div>
            <div class="tv-stage">
              <div class="tv-shell">
                <div class="tv-screen" id="tv-screen">
                  <div class="tv-title" id="tv-frame-title">Agentra Canlı Kumanda</div>
                  <img class="tv-image" id="tv-image" alt="Agentra canlı kare" />
                  <div class="tv-empty" id="tv-empty"></div>
                  <div class="cursor-dot" id="cursor-dot"></div>
                  <div class="cursor-bubble" id="cursor-bubble"></div>
                </div>
                <div class="tv-footer">
                  <div class="scrub-row">
                    <span class="scrub-meta" id="scrubber-label">0 / 0</span>
                    <div class="timeline-track" id="timeline-track" aria-label="Zaman çizelgesi">
                      <div class="timeline-progress" id="timeline-progress"></div>
                      <div class="timeline-handle" id="timeline-handle"></div>
                    </div>
                    <span id="tv-status">${statusLabel(state.status)}</span>
                    <span>CANLI</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>
      </section>
    </main>
  `;

  document.getElementById("run-form").addEventListener("submit", startRun);

  setupScrubber();
  render();

  if (boot.activeRunId) {
    fetchJson(`/runs/${boot.activeRunId}`)
      .then((snapshot) => {
        applySnapshot(snapshot);
        connectStream();
      })
      .catch(() => {});
  }
}

window.addEventListener("beforeunload", () => {
  if (state.source) state.source.close();
});

mount();
"""


def _app_script_state() -> str:
    return """
const boot = window.__AGENTRA_BOOT__ || {};
const DEFAULT_FOCUS = { x: 0.74, y: 0.2 };

const state = {
  runId: null,
  selectedRunId: null,
  streamRunId: null,
  activeThreadId: null,
  activeThread: null,
  threads: [],
  goal: "",
  provider: boot.provider || "",
  model: boot.model || "",
  status: "idle",
  events: [],
  frames: [],
  steps: [],
  liveMode: true,
  historyMode: false,
  selectedFrameId: null,
  reportUrl: null,
  source: null,
  threadPollTimer: null,
  refreshInFlight: false,
  scrubPercent: 1,
  scrubDrag: null,
  initialRunId: boot.activeRunId || null,
  errors: {
    run: "",
    thread: "",
    approval: "",
    question: "",
    manual: "",
  },
  drafts: {
    goal: "",
    threadTitle: "",
    questionAnswers: {},
    scrollAmount: "600",
    clickSelector: "",
    typeSelector: "",
    typeText: "",
    advancedTool: "browser",
    advancedArgs: "{\\n  \\"action\\": \\"back\\"\\n}",
  },
  activity: {
    mode: "idle",
    summary: "",
    focus_x: DEFAULT_FOCUS.x,
    focus_y: DEFAULT_FOCUS.y,
  },
};

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function shortText(value, limit = 220) {
  const text = String(value ?? "").replace(/\\s+/g, " ").trim();
  if (text.length <= limit) return text;
  return `${text.slice(0, limit - 3).trimEnd()}...`;
}

function statusLabel(status) {
  if (status === "running") return "ÇALIŞIYOR";
  if (status === "completed") return "TAMAMLANDI";
  if (status === "error") return "HATA";
  if (status === "partial") return "KISMİ";
  return "HAZIR";
}

function threadStatusLabel(status) {
  if (status === "running") return "ÇALIŞIYOR";
  if (status === "paused_for_user") return "DEVREDİLDİ";
  if (status === "blocked_waiting_user") return "BEKLİYOR";
  if (status === "completed") return "TAMAMLANDI";
  if (status === "error") return "HATA";
  return "HAZIR";
}

function formatDateTime(value) {
  if (!value) return "—";
  try {
    return new Intl.DateTimeFormat("tr-TR", {
      day: "2-digit",
      month: "short",
      hour: "2-digit",
      minute: "2-digit",
    }).format(new Date(value));
  } catch (error) {
    return String(value);
  }
}

function projectStamp() {
  const dateLabel = new Intl.DateTimeFormat("tr-TR", {
    month: "short",
    day: "numeric",
  }).format(new Date());
  return `${dateLabel} · Otonom Ajan Projesi`;
}

function shortPath(value) {
  const text = String(value || "");
  if (!text) return "—";
  const normalized = text.replaceAll("\\\\", "/");
  const parts = normalized.split("/").filter(Boolean);
  return parts.slice(-4).join("/");
}

function pendingApprovals(thread) {
  return (thread?.approval_requests || []).filter((item) => !item.status || item.status === "pending");
}

function pendingQuestions(thread) {
  return (thread?.question_requests || []).filter((item) => !item.status || item.status === "pending");
}

function activeRunForThread(thread) {
  return thread?.active_run || null;
}

function runSummaries(thread) {
  return Array.isArray(thread?.runs) ? [...thread.runs] : [];
}

function canManualControl(thread) {
  if (!thread || !activeRunForThread(thread)) return false;
  return thread.status === "paused_for_user" || thread.status === "blocked_waiting_user";
}

function displayedRunSummary() {
  const thread = state.activeThread;
  if (!thread || !state.selectedRunId) return null;
  return runSummaries(thread).find((item) => item.run_id === state.selectedRunId) || null;
}

function setError(key, message) {
  state.errors[key] = String(message || "");
}

function localizedLabel(event) {
  return event.display_label || event.frame_label || event.label || event.tool || "İşlem";
}

function localizedSummary(event) {
  return event.display_summary || event.summary || event.content || event.result || "";
}

function frameFromEvent(event) {
  if (event.type !== "screenshot" || !event.image_url) return null;
  return {
    id: event.frame_id || `frame-${Date.now()}`,
    timestamp: event.timestamp || "",
    label: event.label || event.frame_label || "Görsel Kare",
    display_label: event.display_label || event.frame_label || "Görsel Kare",
    summary: event.summary || "Görsel güncelleme",
    display_summary: event.display_summary || event.summary || "Görsel güncelleme",
    image_path: event.image_path || "",
    image_url: event.image_url,
    focus_x: typeof event.focus_x === "number" ? event.focus_x : null,
    focus_y: typeof event.focus_y === "number" ? event.focus_y : null,
  };
}

function currentFrame() {
  if (!state.frames.length) return null;
  if (state.liveMode || !state.selectedFrameId) return state.frames[state.frames.length - 1];
  return state.frames.find((frame) => frame.id === state.selectedFrameId) || state.frames[state.frames.length - 1];
}

function buildSteps(events) {
  const steps = [];
  let currentTool = null;

  events.forEach((event, index) => {
    const timestamp = event.timestamp || "";
    const type = event.type;
    if (type === "phase") return;

    if (type === "thought") {
      currentTool = null;
      steps.push({
        id: `step-${index + 1}`,
        kind: "thought",
        tone: "assistant",
        title: localizedLabel(event) || "Düşünce özeti",
        summary: shortText(localizedSummary(event), 260),
        status_label: "düşünüyor",
        timestamp,
      });
      return;
    }

    if (type === "tool_call") {
      currentTool = {
        id: `step-${index + 1}`,
        kind: "tool",
        tone: "pending",
        tool: event.tool,
        title: localizedLabel(event),
        summary: shortText(localizedSummary(event), 220),
        status_label: "uyguluyor",
        timestamp,
        detail: JSON.stringify(event.args || {}, null, 2),
        image_url: null,
      };
      steps.push(currentTool);
      return;
    }

    if (type === "visual_intent") {
      if (currentTool && currentTool.tool === event.tool) {
        currentTool.summary = shortText(localizedSummary(event), 220);
        currentTool.status_label = "hazırlanıyor";
        currentTool.focus_x = event.focus_x;
        currentTool.focus_y = event.focus_y;
      }
      return;
    }

    if (type === "screenshot") {
      if (currentTool) {
        currentTool.image_url = event.image_url || currentTool.image_url;
        currentTool.frame_id = event.frame_id || currentTool.frame_id;
      } else {
        steps.push({
          id: `step-${index + 1}`,
          kind: "visual",
          tone: "neutral",
          title: localizedLabel(event),
          summary: shortText(localizedSummary(event), 220),
          status_label: "görsel",
          timestamp,
          image_url: event.image_url,
        });
      }
      return;
    }

    if (type === "tool_result") {
      if (currentTool && currentTool.tool === event.tool) {
        currentTool.tone = event.success ? "success" : "error";
        currentTool.status_label = event.success ? "tamamlandı" : "hata";
        currentTool.summary = shortText(localizedSummary(event), 220);
        currentTool.detail = String(event.result || "");
        currentTool = null;
      } else {
        steps.push({
          id: `step-${index + 1}`,
          kind: "result",
          tone: event.success ? "success" : "error",
          title: localizedLabel(event),
          summary: shortText(localizedSummary(event), 220),
          status_label: event.success ? "tamamlandı" : "hata",
          timestamp,
          detail: String(event.result || ""),
        });
      }
      return;
    }

    if (type === "done") {
      currentTool = null;
      steps.push({
        id: `step-${index + 1}`,
        kind: "done",
        tone: "success",
        title: "Görev Tamamlandı",
        summary: String(event.content || ""),
        status_label: "tamamlandı",
        timestamp,
      });
      return;
    }

    if (type === "error") {
      currentTool = null;
      steps.push({
        id: `step-${index + 1}`,
        kind: "error",
        tone: "error",
        title: "Bir Hata Oluştu",
        summary: shortText(event.content || "", 260),
        status_label: "hata",
        timestamp,
      });
    }
  });

  return steps.slice(-24);
}

function deriveActivity(events, frames) {
  const latestFrame = frames.length ? frames[frames.length - 1] : null;
  for (let index = events.length - 1; index >= 0; index -= 1) {
    const event = events[index];
    if (event.type === "phase") {
      return {
        mode: event.phase || "thinking",
        summary: localizedSummary(event),
        focus_x: latestFrame && typeof latestFrame.focus_x === "number" ? latestFrame.focus_x : DEFAULT_FOCUS.x,
        focus_y: latestFrame && typeof latestFrame.focus_y === "number" ? latestFrame.focus_y : DEFAULT_FOCUS.y,
      };
    }
    if (event.type === "visual_intent") {
      return {
        mode: "acting",
        summary: localizedSummary(event),
        focus_x: typeof event.focus_x === "number" ? event.focus_x : DEFAULT_FOCUS.x,
        focus_y: typeof event.focus_y === "number" ? event.focus_y : DEFAULT_FOCUS.y,
      };
    }
    if (event.type === "tool_call") {
      return {
        mode: "acting",
        summary: localizedSummary(event),
        focus_x: latestFrame && typeof latestFrame.focus_x === "number" ? latestFrame.focus_x : DEFAULT_FOCUS.x,
        focus_y: latestFrame && typeof latestFrame.focus_y === "number" ? latestFrame.focus_y : DEFAULT_FOCUS.y,
      };
    }
    if (event.type === "screenshot") {
      return {
        mode: "acting",
        summary: localizedSummary(event),
        focus_x: typeof event.focus_x === "number" ? event.focus_x : DEFAULT_FOCUS.x,
        focus_y: typeof event.focus_y === "number" ? event.focus_y : DEFAULT_FOCUS.y,
      };
    }
    if (event.type === "tool_result") {
      return {
        mode: "idle",
        summary: localizedSummary(event),
        focus_x: latestFrame && typeof latestFrame.focus_x === "number" ? latestFrame.focus_x : DEFAULT_FOCUS.x,
        focus_y: latestFrame && typeof latestFrame.focus_y === "number" ? latestFrame.focus_y : DEFAULT_FOCUS.y,
      };
    }
    if (event.type === "done" || event.type === "error") {
      return {
        mode: "idle",
        summary: localizedSummary(event),
        focus_x: latestFrame && typeof latestFrame.focus_x === "number" ? latestFrame.focus_x : DEFAULT_FOCUS.x,
        focus_y: latestFrame && typeof latestFrame.focus_y === "number" ? latestFrame.focus_y : DEFAULT_FOCUS.y,
      };
    }
  }
  return {
    mode: "idle",
    summary: "",
    focus_x: DEFAULT_FOCUS.x,
    focus_y: DEFAULT_FOCUS.y,
  };
}

function syncSelectionToFrames() {
  if (!state.frames.length) {
    state.selectedFrameId = null;
    state.scrubPercent = 1;
    return;
  }
  if (state.liveMode || !state.selectedFrameId) {
    state.selectedFrameId = state.frames[state.frames.length - 1].id;
  } else if (!state.frames.find((frame) => frame.id === state.selectedFrameId)) {
    state.selectedFrameId = state.frames[state.frames.length - 1].id;
    state.liveMode = true;
  }
  const index = Math.max(0, state.frames.findIndex((frame) => frame.id === state.selectedFrameId));
  state.scrubPercent = state.frames.length > 1 ? index / (state.frames.length - 1) : 1;
}

function disconnectStream() {
  if (state.source) {
    state.source.close();
    state.source = null;
  }
  state.streamRunId = null;
}

function clearRunState(options = {}) {
  if (options.disconnect !== false) disconnectStream();
  state.runId = null;
  state.selectedRunId = null;
  state.goal = "";
  state.status = "idle";
  state.events = [];
  state.frames = [];
  state.steps = [];
  state.reportUrl = null;
  state.historyMode = false;
  state.activity = {
    mode: "idle",
    summary: "",
    focus_x: DEFAULT_FOCUS.x,
    focus_y: DEFAULT_FOCUS.y,
  };
  syncSelectionToFrames();
}

function applySnapshot(snapshot, options = {}) {
  state.runId = snapshot.run_id;
  state.selectedRunId = snapshot.run_id;
  state.goal = snapshot.goal || state.goal;
  state.provider = snapshot.provider;
  state.model = snapshot.model;
  state.status = snapshot.status || state.status;
  state.events = snapshot.events || [];
  state.frames = snapshot.frames || [];
  state.steps = snapshot.steps || buildSteps(state.events);
  state.reportUrl = snapshot.report_url || null;
  if (snapshot.thread_id) state.activeThreadId = snapshot.thread_id;
  state.historyMode = Boolean(options.history);
  syncSelectionToFrames();
  state.activity = deriveActivity(state.events, state.frames);
  if (options.connect) {
    connectStream(snapshot.run_id);
  } else if (options.disconnect !== false) {
    disconnectStream();
  }
  render();
}

function applyEvent(event) {
  state.events = [...state.events, event];
  const frame = frameFromEvent(event);
  if (frame) state.frames = [...state.frames, frame];
  if (event.type === "done" && state.status === "running") state.status = "completed";
  if (event.type === "error") state.status = "error";
  state.steps = buildSteps(state.events);
  syncSelectionToFrames();
  state.activity = deriveActivity(state.events, state.frames);
  render();
}
"""


def _app_script_render() -> str:
    return """
function currentOverlay(frame) {
  if (state.status === "running" && (state.activity.mode === "thinking" || state.activity.mode === "acting")) {
    return {
      busy: true,
      summary: state.activity.summary || (frame ? frame.display_summary : ""),
      focus_x: typeof state.activity.focus_x === "number" ? state.activity.focus_x : (frame ? frame.focus_x : DEFAULT_FOCUS.x),
      focus_y: typeof state.activity.focus_y === "number" ? state.activity.focus_y : (frame ? frame.focus_y : DEFAULT_FOCUS.y),
    };
  }
  if (frame) {
    return {
      busy: false,
      summary: frame.display_summary || "",
      focus_x: typeof frame.focus_x === "number" ? frame.focus_x : DEFAULT_FOCUS.x,
      focus_y: typeof frame.focus_y === "number" ? frame.focus_y : DEFAULT_FOCUS.y,
    };
  }
  return {
    busy: false,
    summary: "",
    focus_x: DEFAULT_FOCUS.x,
    focus_y: DEFAULT_FOCUS.y,
  };
}

function currentThreadTitle() {
  return state.activeThread?.title || "Ajan";
}

function currentRunStatusLabel() {
  return statusLabel(state.status);
}

function setInlineText(id, value) {
  const node = document.getElementById(id);
  if (!node) return;
  node.textContent = value || "";
}

function renderThreadList() {
  const container = document.getElementById("thread-list");
  if (!container) return;
  if (!state.threads.length) {
    container.innerHTML = `<div class="empty-state">Henüz thread yok. Yeni bir run başlat.</div>`;
    return;
  }
  container.innerHTML = state.threads.map((thread) => {
    const activeRun = activeRunForThread(thread);
    const runs = runSummaries(thread);
    const latestRun = runs.length ? runs[runs.length - 1] : null;
    const summary = activeRun?.goal || latestRun?.goal || "Henüz run yok";
    const isActive = thread.thread_id === state.activeThreadId;
    const approvals = pendingApprovals(thread).length;
    const questions = pendingQuestions(thread).length;
    return `
      <button type="button" class="thread-item ${isActive ? "active" : ""}" data-thread-select="${escapeHtml(thread.thread_id)}">
        <div class="thread-item-head">
          <span class="thread-title">${escapeHtml(thread.title || thread.thread_id)}</span>
          <span class="thread-chip ${escapeHtml(thread.status || "idle")}">${escapeHtml(threadStatusLabel(thread.status || "idle"))}</span>
        </div>
        <div class="thread-summary">${escapeHtml(shortText(summary, 84))}</div>
        <div class="badge-row">
          <span class="count-badge">${escapeHtml(activeRun ? "aktif run" : "son run")}</span>
          ${approvals ? `<span class="count-badge">${approvals} onay</span>` : ""}
          ${questions ? `<span class="count-badge">${questions} soru</span>` : ""}
        </div>
      </button>
    `;
  }).join("");
}

function renderSelectedThread() {
  const container = document.getElementById("selected-thread-panel");
  if (!container) return;
  const thread = state.activeThread;
  if (!thread) {
    container.innerHTML = `<div class="empty-state">Detayları görmek için bir thread seç.</div>`;
    return;
  }
  const activeRun = activeRunForThread(thread);
  const selectedRun = displayedRunSummary();
  const canPause = Boolean(activeRun) && thread.status === "running";
  const canResume = Boolean(activeRun) && (thread.status === "paused_for_user" || thread.status === "blocked_waiting_user");
  const canOpenReport = Boolean(state.reportUrl);
  const canReturnLive = Boolean(activeRun && state.historyMode && state.selectedRunId && state.selectedRunId !== activeRun.run_id);
  container.innerHTML = `
    <div class="thread-detail-card">
      <div class="thread-item-head">
        <span class="thread-title">${escapeHtml(thread.title)}</span>
        <span class="thread-chip ${escapeHtml(thread.status || "idle")}">${escapeHtml(threadStatusLabel(thread.status || "idle"))}</span>
      </div>
      <div class="thread-summary">${escapeHtml(shortText((selectedRun || activeRun || {}).goal || "Henüz run yok", 120))}</div>
      <div class="detail-grid">
        <div class="detail-row">
          <span class="detail-key">Aktif run</span>
          <span>${escapeHtml(activeRun?.run_id || "—")}</span>
        </div>
        <div class="detail-row">
          <span class="detail-key">Görünen run</span>
          <span>${escapeHtml(state.selectedRunId || "—")}</span>
        </div>
        <div class="detail-row">
          <span class="detail-key">Workspace</span>
          <span class="path-copy">${escapeHtml(shortPath(thread.workspace_dir))}</span>
        </div>
        <div class="detail-row">
          <span class="detail-key">Memory</span>
          <span class="path-copy">${escapeHtml(shortPath(thread.memory_dir))}</span>
        </div>
      </div>
      <div class="action-row">
        <button type="button" class="action-button" data-thread-action="pause" ${canPause ? "" : "disabled"}>Pause</button>
        <button type="button" class="action-button" data-thread-action="resume" ${canResume ? "" : "disabled"}>Resume</button>
        <button type="button" class="action-button" data-thread-action="report" ${canOpenReport ? "" : "disabled"}>Report aç</button>
        ${canReturnLive ? `<button type="button" class="action-button primary" data-thread-action="return-live">Aktif run'a dön</button>` : ""}
      </div>
    </div>
  `;
}

function renderApprovalList() {
  const container = document.getElementById("approval-list");
  if (!container) return;
  const approvals = pendingApprovals(state.activeThread);
  if (!approvals.length) {
    container.innerHTML = `<div class="empty-state">Bekleyen onay yok.</div>`;
    return;
  }
  container.innerHTML = approvals.map((item) => `
    <div class="request-card">
      <div class="request-title">${escapeHtml(item.summary || item.tool || "Onay isteği")}</div>
      <div class="request-text">${escapeHtml(item.reason || "Bu işlem kullanıcı onayı bekliyor.")}</div>
      <div class="request-meta">${escapeHtml(item.request_id)}</div>
      <div class="action-row">
        <button type="button" class="action-button success" data-approval-id="${escapeHtml(item.request_id)}" data-approved="true">Onayla</button>
        <button type="button" class="action-button warn" data-approval-id="${escapeHtml(item.request_id)}" data-approved="false">Reddet</button>
      </div>
    </div>
  `).join("");
}

function renderQuestionList() {
  const container = document.getElementById("question-list");
  if (!container) return;
  const questions = pendingQuestions(state.activeThread);
  if (!questions.length) {
    container.innerHTML = `<div class="empty-state">Bekleyen soru yok.</div>`;
    return;
  }
  container.innerHTML = questions.map((item) => `
    <div class="request-card">
      <div class="request-title">${escapeHtml(item.summary || "Kullanıcı yanıtı gerekiyor")}</div>
      <div class="request-text">${escapeHtml(item.prompt)}</div>
      <div class="inline-form">
        <input class="mini-input" type="text" value="${escapeHtml(state.drafts.questionAnswers[item.request_id] || "")}" data-question-input="${escapeHtml(item.request_id)}" placeholder="Yanıtını yaz" />
        <button type="button" class="action-button primary" data-question-submit="${escapeHtml(item.request_id)}">Gönder</button>
      </div>
    </div>
  `).join("");
}

function renderManualControls() {
  const container = document.getElementById("manual-controls");
  if (!container) return;
  const enabled = canManualControl(state.activeThread);
  const disabled = enabled ? "" : "disabled";
  container.innerHTML = `
    <div class="manual-grid">
      <div class="helper-text">${enabled ? "Browser tool için hızlı manuel kontroller." : "Manuel kontrol için önce thread'i pause et veya kullanıcı bekleyen duruma gelmesini bekle."}</div>
      <div class="action-row">
        <button type="button" class="action-button" data-manual-action="back" ${disabled}>Geri</button>
        <button type="button" class="action-button" data-manual-action="forward" ${disabled}>İleri</button>
      </div>
      <div class="inline-form">
        <input class="mini-input" type="number" value="${escapeHtml(state.drafts.scrollAmount)}" data-manual-input="scrollAmount" placeholder="Kaydırma miktarı" ${disabled} />
        <button type="button" class="action-button" data-manual-action="scroll" ${disabled}>Kaydır</button>
      </div>
      <div class="inline-form">
        <input class="mini-input" type="text" value="${escapeHtml(state.drafts.clickSelector)}" data-manual-input="clickSelector" placeholder="CSS selector" ${disabled} />
        <button type="button" class="action-button" data-manual-action="click" ${disabled}>Selector'a tıkla</button>
      </div>
      <div class="manual-row">
        <input class="mini-input" type="text" value="${escapeHtml(state.drafts.typeSelector)}" data-manual-input="typeSelector" placeholder="CSS selector" ${disabled} />
        <input class="mini-input" type="text" value="${escapeHtml(state.drafts.typeText)}" data-manual-input="typeText" placeholder="Yazılacak metin" ${disabled} />
      </div>
      <div class="action-row">
        <button type="button" class="action-button" data-manual-action="type" ${disabled}>Selector'a yaz</button>
      </div>
      <details class="details-shell">
        <summary>Gelişmiş</summary>
        <div class="manual-grid">
          <input class="mini-input" type="text" value="${escapeHtml(state.drafts.advancedTool)}" data-manual-input="advancedTool" placeholder="Tool adı" ${disabled} />
          <textarea class="json-input" data-manual-input="advancedArgs" placeholder='{"action":"back"}' ${disabled}>${escapeHtml(state.drafts.advancedArgs)}</textarea>
          <div class="action-row">
            <button type="button" class="action-button primary" data-manual-action="advanced" ${disabled}>Manuel aksiyon gönder</button>
          </div>
        </div>
      </details>
    </div>
  `;
}

function renderRunHistory() {
  const container = document.getElementById("run-history");
  if (!container) return;
  const runs = runSummaries(state.activeThread).slice().reverse();
  if (!runs.length) {
    container.innerHTML = `<div class="empty-state">Bu thread için kayıtlı run yok.</div>`;
    return;
  }
  container.innerHTML = runs.map((run) => {
    const active = run.run_id === state.selectedRunId;
    return `
      <div class="history-item ${active ? "active" : ""}">
        <div class="history-item-head">
          <span class="thread-title">${escapeHtml(shortText(run.goal || run.run_id, 54))}</span>
          <span class="thread-chip ${escapeHtml(run.status || "idle")}">${escapeHtml(statusLabel(run.status || "idle"))}</span>
        </div>
        <div class="history-meta">${escapeHtml(formatDateTime(run.started_at))} · ${escapeHtml(formatDateTime(run.finished_at))}</div>
        <div class="action-row">
          <button type="button" class="action-button" data-history-open="${escapeHtml(run.run_id)}">Aç</button>
          <a class="action-button" href="${escapeHtml(run.report_url)}" target="_blank" rel="noreferrer">Report</a>
        </div>
      </div>
    `;
  }).join("");
}

function renderHeader() {
  setInlineText("project-stamp", projectStamp());
  setInlineText("agent-title", currentThreadTitle());
}

function renderTV() {
  const frame = currentFrame();
  const overlay = currentOverlay(frame);
  const screen = document.getElementById("tv-screen");
  const image = document.getElementById("tv-image");
  const empty = document.getElementById("tv-empty");
  const dot = document.getElementById("cursor-dot");
  const bubble = document.getElementById("cursor-bubble");
  const title = document.getElementById("tv-frame-title");
  const activeLabel = currentThreadTitle();

  screen.classList.toggle("busy", Boolean(overlay.busy));
  title.textContent = `${activeLabel} · ${currentRunStatusLabel()}`;

  if (!frame) {
    image.style.display = "none";
    empty.style.display = "grid";
    empty.innerHTML = overlay.busy
      ? `<div><strong>${escapeHtml(overlay.summary || "İşlem hazırlanıyor...")}</strong><div class="muted" style="margin-top:8px;">Thread canlı olduğunda yeni kareler burada görünür.</div></div>`
      : `<div><strong>Bir thread seç veya yeni run başlat.</strong><div class="muted" style="margin-top:8px;">Canlı TV, timeline ve run replay burada gösterilecek.</div></div>`;
  } else {
    empty.style.display = "none";
    image.style.display = "block";
    if (image.dataset.frameId !== frame.id) {
      image.classList.add("swapping");
      image.src = frame.image_url;
      image.dataset.frameId = frame.id;
      window.setTimeout(() => image.classList.remove("swapping"), 180);
    }
  }

  if (overlay.summary) {
    const left = `${Math.max(0, Math.min(100, (overlay.focus_x ?? DEFAULT_FOCUS.x) * 100))}%`;
    const top = `${Math.max(0, Math.min(100, (overlay.focus_y ?? DEFAULT_FOCUS.y) * 100))}%`;
    dot.style.display = "block";
    bubble.style.display = "block";
    dot.style.left = left;
    dot.style.top = top;
    bubble.style.left = left;
    bubble.style.top = top;
    dot.classList.toggle("pending", overlay.busy);
    bubble.classList.toggle("flip", (overlay.focus_y ?? DEFAULT_FOCUS.y) > 0.68);
    bubble.textContent = overlay.summary;
  } else {
    dot.style.display = "none";
    bubble.style.display = "none";
  }
}

function renderTimeline() {
  const frame = currentFrame();
  const frameIndex = frame ? Math.max(0, state.frames.findIndex((item) => item.id === frame.id)) : 0;
  const label = document.getElementById("scrubber-label");
  const progress = document.getElementById("timeline-progress");
  const handle = document.getElementById("timeline-handle");
  const percent = state.frames.length > 1 ? state.scrubPercent : 1;
  const percentText = `${Math.max(0, Math.min(100, percent * 100))}%`;
  label.textContent = state.frames.length ? `${frameIndex + 1} / ${state.frames.length}` : "0 / 0";
  progress.style.width = percentText;
  handle.style.left = percentText;
  setInlineText("tv-status", currentRunStatusLabel());
}

function renderConsole() {
  const goalInput = document.getElementById("goal-input");
  const titleInput = document.getElementById("thread-title-input");
  const appendThreadButton = document.getElementById("append-thread-button");

  if (goalInput && goalInput.value !== state.drafts.goal) goalInput.value = state.drafts.goal;
  if (titleInput && titleInput.value !== state.drafts.threadTitle) titleInput.value = state.drafts.threadTitle;
  if (appendThreadButton) appendThreadButton.disabled = !state.activeThreadId;

  renderThreadList();
  renderSelectedThread();
  renderApprovalList();
  renderQuestionList();
  renderManualControls();
  renderRunHistory();

  setInlineText("run-error", state.errors.run);
  setInlineText("thread-error", state.errors.thread);
  setInlineText("approval-error", state.errors.approval);
  setInlineText("question-error", state.errors.question);
  setInlineText("manual-error", state.errors.manual);
}

function render() {
  renderHeader();
  renderConsole();
  renderTV();
  renderTimeline();
}

async function fetchJson(url, options = {}) {
  const response = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!response.ok) {
    let message = `${response.status}`;
    try {
      const payload = await response.json();
      message = payload.detail || message;
    } catch (error) {
      // Ignore JSON parse failures on empty responses.
    }
    throw new Error(message);
  }
  return response.json();
}

function selectFrameByPercent(percent) {
  if (!state.frames.length) return;
  const safe = Math.max(0, Math.min(1, percent));
  state.scrubPercent = safe;
  state.liveMode = false;
  const index = state.frames.length > 1 ? Math.round(safe * (state.frames.length - 1)) : 0;
  state.selectedFrameId = state.frames[index].id;
  render();
}
"""


def _app_script_mount() -> str:
    return """
function setupScrubber() {
  const track = document.getElementById("timeline-track");

  function updateFromClientX(clientX) {
    const rect = track.getBoundingClientRect();
    if (!rect.width) return;
    selectFrameByPercent((clientX - rect.left) / rect.width);
  }

  track.addEventListener("pointerdown", (event) => {
    if (!state.frames.length) return;
    state.scrubDrag = { pointerId: event.pointerId };
    track.setPointerCapture(event.pointerId);
    updateFromClientX(event.clientX);
  });

  track.addEventListener("pointermove", (event) => {
    if (!state.scrubDrag || state.scrubDrag.pointerId !== event.pointerId) return;
    const clientX = event.clientX;
    if (state.scrubDrag.raf) {
      state.scrubDrag.pendingX = clientX;
      return;
    }
    state.scrubDrag.pendingX = clientX;
    state.scrubDrag.raf = window.requestAnimationFrame(() => {
      updateFromClientX(state.scrubDrag.pendingX);
      state.scrubDrag.raf = null;
    });
  });

  function stopScrub(event) {
    if (!state.scrubDrag || state.scrubDrag.pointerId !== event.pointerId) return;
    if (state.scrubDrag.raf) window.cancelAnimationFrame(state.scrubDrag.raf);
    track.releasePointerCapture?.(event.pointerId);
    state.scrubDrag = null;
  }

  track.addEventListener("pointerup", stopScrub);
  track.addEventListener("pointercancel", stopScrub);
}

function connectStream(runId) {
  if (!runId) return;
  if (state.streamRunId === runId && state.source) return;
  disconnectStream();
  state.streamRunId = runId;
  state.source = new EventSource(`/runs/${runId}/events`);
  state.source.onmessage = (message) => {
    const payload = JSON.parse(message.data);
    if (payload.kind === "snapshot") {
      applySnapshot(payload.snapshot, { connect: false, disconnect: false, history: false });
      return;
    }
    if (payload.kind === "event") {
      applyEvent(payload.event);
      return;
    }
    if (payload.kind === "status") {
      state.status = payload.status || state.status;
      applySnapshot(payload.snapshot, { connect: false, disconnect: false, history: false });
      return;
    }
    if (payload.kind === "complete") {
      state.status = payload.status || state.status;
      state.activity = deriveActivity(state.events, state.frames);
      if (state.source) {
        state.source.close();
        state.source = null;
      }
      state.streamRunId = null;
      render();
    }
  };
}

async function loadRun(runId, options = {}) {
  const snapshot = await fetchJson(`/runs/${runId}`);
  applySnapshot(snapshot, { connect: Boolean(options.live), history: Boolean(options.history) });
}

async function syncThreadSnapshot(thread, options = {}) {
  state.activeThread = thread;
  state.activeThreadId = thread.thread_id;
  const activeRun = activeRunForThread(thread);
  const runs = runSummaries(thread);
  const runIds = runs.map((item) => item.run_id);

  let targetRunId = null;
  let live = false;

  if (options.preferLive && activeRun) {
    targetRunId = activeRun.run_id;
    live = true;
  } else if (state.historyMode && state.selectedRunId && runIds.includes(state.selectedRunId)) {
    targetRunId = state.selectedRunId;
  } else if (options.preferRunId && runIds.includes(options.preferRunId)) {
    targetRunId = options.preferRunId;
    live = Boolean(activeRun && activeRun.run_id === options.preferRunId && options.preferLive);
  } else if (activeRun) {
    targetRunId = activeRun.run_id;
    live = true;
  } else if (runs.length) {
    targetRunId = runs[runs.length - 1].run_id;
  }

  if (!targetRunId) {
    clearRunState();
    render();
    return;
  }

  if (activeRun && targetRunId === activeRun.run_id) {
    applySnapshot(activeRun, { connect: live, history: false });
    return;
  }

  await loadRun(targetRunId, { live: false, history: Boolean(state.historyMode) });
}

async function loadThreadDetail(threadId, options = {}) {
  const thread = await fetchJson(`/threads/${threadId}`);
  await syncThreadSnapshot(thread, options);
}

function pickInitialThreadId() {
  if (!state.threads.length) return null;
  if (state.activeThreadId && state.threads.some((item) => item.thread_id === state.activeThreadId)) {
    return state.activeThreadId;
  }
  if (state.initialRunId) {
    const matched = state.threads.find((item) => item.current_run_id === state.initialRunId || runSummaries(item).some((run) => run.run_id === state.initialRunId));
    if (matched) return matched.thread_id;
  }
  return state.threads[0].thread_id;
}

async function refreshThreads(options = {}) {
  if (state.refreshInFlight) return;
  state.refreshInFlight = true;
  try {
    const payload = await fetchJson("/threads");
    state.threads = payload.threads || [];
    setError("thread", "");
    const nextThreadId = options.forceThreadId || pickInitialThreadId();
    if (!nextThreadId) {
      state.activeThreadId = null;
      state.activeThread = null;
      clearRunState();
      render();
      return;
    }
    await loadThreadDetail(nextThreadId, {
      preferRunId: options.preferRunId || state.selectedRunId,
      preferLive: Boolean(options.preferLive),
    });
  } catch (error) {
    setError("thread", error.message);
    render();
  } finally {
    state.refreshInFlight = false;
  }
}

function scheduleThreadPolling() {
  if (state.threadPollTimer) window.clearInterval(state.threadPollTimer);
  state.threadPollTimer = window.setInterval(() => {
    refreshThreads({
      forceThreadId: state.activeThreadId,
      preferRunId: state.historyMode ? state.selectedRunId : null,
      preferLive: !state.historyMode,
    });
  }, 2000);
}

async function startRun(mode) {
  const goal = state.drafts.goal.trim();
  if (!goal) {
    setError("run", "Önce bir görev gir.");
    render();
    return;
  }

  const payload = { goal };
  if (mode === "new") {
    const title = state.drafts.threadTitle.trim();
    if (title) payload.thread_title = title;
  } else if (state.activeThreadId) {
    payload.thread_id = state.activeThreadId;
  } else {
    setError("run", "Önce bir thread seç.");
    render();
    return;
  }

  try {
    setError("run", "");
    state.historyMode = false;
    const snapshot = await fetchJson("/runs", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    state.drafts.goal = "";
    if (mode === "new") state.drafts.threadTitle = "";
    await refreshThreads({
      forceThreadId: snapshot.thread_id,
      preferRunId: snapshot.run_id,
      preferLive: true,
    });
  } catch (error) {
    setError("run", error.message);
    render();
  }
}

async function pauseActiveThread() {
  if (!state.activeThreadId) return;
  try {
    setError("thread", "");
    const snapshot = await fetchJson(`/threads/${state.activeThreadId}/pause`, { method: "POST" });
    await syncThreadSnapshot(snapshot, { preferLive: true });
  } catch (error) {
    setError("thread", error.message);
    render();
  }
}

async function resumeActiveThread() {
  if (!state.activeThreadId) return;
  try {
    setError("thread", "");
    const snapshot = await fetchJson(`/threads/${state.activeThreadId}/resume`, { method: "POST" });
    await syncThreadSnapshot(snapshot, { preferLive: true });
  } catch (error) {
    setError("thread", error.message);
    render();
  }
}

async function respondApproval(requestId, approved) {
  if (!state.activeThreadId) return;
  try {
    setError("approval", "");
    const snapshot = await fetchJson(`/threads/${state.activeThreadId}/approvals/${requestId}`, {
      method: "POST",
      body: JSON.stringify({ approved, note: approved ? "UI onayı" : "UI reddi" }),
    });
    await syncThreadSnapshot(snapshot, { preferLive: true });
  } catch (error) {
    setError("approval", error.message);
    render();
  }
}

async function submitQuestionAnswer(requestId) {
  if (!state.activeThreadId) return;
  const answer = (state.drafts.questionAnswers[requestId] || "").trim();
  if (!answer) {
    setError("question", "Önce bir yanıt yaz.");
    render();
    return;
  }
  try {
    setError("question", "");
    const snapshot = await fetchJson(`/threads/${state.activeThreadId}/questions/${requestId}`, {
      method: "POST",
      body: JSON.stringify({ answer }),
    });
    delete state.drafts.questionAnswers[requestId];
    await syncThreadSnapshot(snapshot, { preferLive: true });
  } catch (error) {
    setError("question", error.message);
    render();
  }
}

async function sendManualAction(tool, args) {
  if (!state.activeThreadId) return;
  try {
    setError("manual", "");
    const snapshot = await fetchJson(`/threads/${state.activeThreadId}/actions`, {
      method: "POST",
      body: JSON.stringify({ tool, args }),
    });
    await syncThreadSnapshot(snapshot, { preferLive: true });
  } catch (error) {
    setError("manual", error.message);
    render();
  }
}

async function handleManualAction(kind) {
  if (!canManualControl(state.activeThread)) {
    setError("manual", "Önce pause et veya kullanıcı bekleyen duruma gelmesini bekle.");
    render();
    return;
  }
  if (kind === "back") return sendManualAction("browser", { action: "back" });
  if (kind === "forward") return sendManualAction("browser", { action: "forward" });
  if (kind === "scroll") {
    return sendManualAction("browser", {
      action: "scroll",
      amount: Number.parseInt(state.drafts.scrollAmount || "0", 10) || 0,
    });
  }
  if (kind === "click") {
    if (!state.drafts.clickSelector.trim()) {
      setError("manual", "Önce bir selector yaz.");
      render();
      return;
    }
    return sendManualAction("browser", {
      action: "click",
      selector: state.drafts.clickSelector.trim(),
    });
  }
  if (kind === "type") {
    if (!state.drafts.typeSelector.trim()) {
      setError("manual", "Önce bir selector yaz.");
      render();
      return;
    }
    return sendManualAction("browser", {
      action: "type",
      selector: state.drafts.typeSelector.trim(),
      text: state.drafts.typeText,
    });
  }
  if (kind === "advanced") {
    try {
      const args = JSON.parse(state.drafts.advancedArgs || "{}");
      return sendManualAction(state.drafts.advancedTool.trim() || "browser", args);
    } catch (error) {
      setError("manual", "Geçerli bir JSON gir.");
      render();
    }
  }
}

function openReport() {
  if (!state.reportUrl) return;
  window.open(state.reportUrl, "_blank", "noreferrer");
}

async function openRunHistory(runId) {
  state.historyMode = true;
  await loadRun(runId, { live: false, history: true });
}

async function returnToLiveRun() {
  if (!state.activeThreadId) return;
  state.historyMode = false;
  await loadThreadDetail(state.activeThreadId, { preferLive: true });
}

function handleConsoleInput(event) {
  const target = event.target;
  if (target.id === "goal-input") state.drafts.goal = target.value;
  if (target.id === "thread-title-input") state.drafts.threadTitle = target.value;

  const questionId = target.getAttribute("data-question-input");
  if (questionId) state.drafts.questionAnswers[questionId] = target.value;

  const manualKey = target.getAttribute("data-manual-input");
  if (manualKey) state.drafts[manualKey] = target.value;
}

async function handleConsoleClick(event) {
  const threadSelect = event.target.closest("[data-thread-select]");
  if (threadSelect) {
    state.historyMode = false;
    await loadThreadDetail(threadSelect.getAttribute("data-thread-select"), { preferLive: true });
    return;
  }

  const threadAction = event.target.closest("[data-thread-action]");
  if (threadAction) {
    const action = threadAction.getAttribute("data-thread-action");
    if (action === "pause") return pauseActiveThread();
    if (action === "resume") return resumeActiveThread();
    if (action === "report") return openReport();
    if (action === "return-live") return returnToLiveRun();
  }

  const approvalAction = event.target.closest("[data-approval-id]");
  if (approvalAction) {
    const requestId = approvalAction.getAttribute("data-approval-id");
    const approved = approvalAction.getAttribute("data-approved") === "true";
    return respondApproval(requestId, approved);
  }

  const questionAction = event.target.closest("[data-question-submit]");
  if (questionAction) {
    return submitQuestionAnswer(questionAction.getAttribute("data-question-submit"));
  }

  const historyAction = event.target.closest("[data-history-open]");
  if (historyAction) {
    return openRunHistory(historyAction.getAttribute("data-history-open"));
  }

  const manualAction = event.target.closest("[data-manual-action]");
  if (manualAction) {
    return handleManualAction(manualAction.getAttribute("data-manual-action"));
  }
}

function mount() {
  document.getElementById("app").innerHTML = `
    <main class="page">
      <header class="topbar">
        <div class="topbar-left">
          <button class="nav-button" type="button" aria-label="Geri">←</button>
          <div class="title-stack">
            <div class="app-title">Otonom Ajan - Görev Paneli</div>
            <div class="app-subtitle" id="project-stamp"></div>
          </div>
        </div>
      </header>
      <section class="workspace">
        <section class="left-pane">
          <div class="left-pane-inner">
            <section class="console-section">
              <div class="section-title">Yeni Run</div>
              <label class="field-label" for="goal-input">Görev</label>
              <textarea id="goal-input" class="text-area" placeholder="Yeni bir komut veya görev girin"></textarea>
              <label class="field-label" for="thread-title-input">Thread başlığı (opsiyonel)</label>
              <input id="thread-title-input" class="text-input" type="text" placeholder="Yeni thread başlığı" />
              <div class="action-row">
                <button id="new-thread-button" class="action-button primary" type="button">Yeni thread'de başlat</button>
                <button id="append-thread-button" class="action-button" type="button">Seçili thread'e ekle</button>
              </div>
              <div class="inline-error" id="run-error"></div>
            </section>

            <section class="console-section">
              <div class="section-title">Threadler</div>
              <div id="thread-list" class="thread-list"></div>
            </section>

            <section class="console-section">
              <div class="section-title">Seçili Thread</div>
              <div id="selected-thread-panel"></div>
              <div class="inline-error" id="thread-error"></div>
            </section>

            <section class="console-section">
              <div class="section-title">Bekleyen Onaylar</div>
              <div id="approval-list" class="request-list"></div>
              <div class="inline-error" id="approval-error"></div>
            </section>

            <section class="console-section">
              <div class="section-title">Bekleyen Sorular</div>
              <div id="question-list" class="request-list"></div>
              <div class="inline-error" id="question-error"></div>
            </section>

            <section class="console-section">
              <div class="section-title">Manuel Browser Kontrolleri</div>
              <div id="manual-controls"></div>
              <div class="inline-error" id="manual-error"></div>
            </section>

            <section class="console-section">
              <div class="section-title">Run Geçmişi</div>
              <div id="run-history" class="history-list"></div>
            </section>
          </div>
        </section>

        <section class="right-pane">
          <div class="stage-shell">
            <div class="stage-head">
              <div class="stage-title" id="agent-title">Ajan</div>
            </div>
            <div class="tv-stage">
              <div class="tv-shell">
                <div class="tv-screen" id="tv-screen">
                  <div class="tv-title" id="tv-frame-title">Ajan · HAZIR</div>
                  <img class="tv-image" id="tv-image" alt="Agentra canlı kare" />
                  <div class="tv-empty" id="tv-empty"></div>
                  <div class="cursor-dot" id="cursor-dot"></div>
                  <div class="cursor-bubble" id="cursor-bubble"></div>
                </div>
                <div class="tv-footer">
                  <div class="scrub-row">
                    <span class="scrub-meta" id="scrubber-label">0 / 0</span>
                    <div class="timeline-track" id="timeline-track" aria-label="Zaman çizelgesi">
                      <div class="timeline-progress" id="timeline-progress"></div>
                      <div class="timeline-handle" id="timeline-handle"></div>
                    </div>
                    <span id="tv-status">HAZIR</span>
                    <span>CANLI</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>
      </section>
    </main>
  `;

  document.getElementById("new-thread-button").addEventListener("click", () => startRun("new"));
  document.getElementById("append-thread-button").addEventListener("click", () => startRun("append"));
  document.getElementById("goal-input").addEventListener("input", handleConsoleInput);
  document.getElementById("thread-title-input").addEventListener("input", handleConsoleInput);
  document.getElementById("thread-list").addEventListener("click", handleConsoleClick);
  document.getElementById("selected-thread-panel").addEventListener("click", handleConsoleClick);
  document.getElementById("approval-list").addEventListener("click", handleConsoleClick);
  document.getElementById("question-list").addEventListener("click", handleConsoleClick);
  document.getElementById("question-list").addEventListener("input", handleConsoleInput);
  document.getElementById("manual-controls").addEventListener("click", handleConsoleClick);
  document.getElementById("manual-controls").addEventListener("input", handleConsoleInput);
  document.getElementById("run-history").addEventListener("click", handleConsoleClick);

  setupScrubber();
  render();
  refreshThreads({ preferLive: true });
  scheduleThreadPolling();
}

window.addEventListener("beforeunload", () => {
  disconnectStream();
  if (state.threadPollTimer) window.clearInterval(state.threadPollTimer);
});

mount();
"""
