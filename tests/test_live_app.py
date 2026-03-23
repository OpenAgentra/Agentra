"""Tests for the live FastAPI operator UI."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from agentra.config import AgentConfig
from agentra.live_app import _render_logs_html, create_live_app
from agentra.runtime import ApprovalRequest, UserInputRequest
from agentra.tools.base import ToolResult

PNG_1X1_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)


class FakeAgent:
    """Deterministic async agent used to exercise the live app routes."""

    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        self.interrupted = False
        self.paused = False

    async def run(self, goal: str):
        async def generator():
            yield {
                "type": "phase",
                "phase": "thinking",
                "content": "Sonraki adımı planlıyor...",
                "summary": "Sonraki adımı planlıyor...",
            }
            yield {"type": "thought", "content": f"Planning for {goal}"}
            await asyncio.sleep(0.01)
            yield {
                "type": "phase",
                "phase": "acting",
                "content": "İşlemi hazırlıyor...",
                "summary": "İşlemi hazırlıyor...",
            }
            yield {
                "type": "tool_call",
                "tool": "browser",
                "args": {"action": "navigate", "url": "https://www.python.org"},
            }
            await asyncio.sleep(0.01)
            yield {
                "type": "visual_intent",
                "tool": "browser",
                "args": {"action": "navigate", "url": "https://www.python.org"},
                "focus_x": 0.5,
                "focus_y": 0.4,
                "frame_label": "browser · navigate",
                "summary": "Opening python.org",
            }
            await asyncio.sleep(0.01)
            yield {
                "type": "screenshot",
                "data": PNG_1X1_B64,
                "focus_x": 0.5,
                "focus_y": 0.4,
                "frame_label": "browser · navigate",
                "summary": "Opening python.org",
            }
            await asyncio.sleep(0.01)
            yield {
                "type": "tool_result",
                "tool": "browser",
                "result": "Navigated to python.org",
                "success": True,
                "summary": "Page loaded",
            }
            for _ in range(100):
                if self.interrupted:
                    break
                while self.paused and not self.interrupted:
                    await asyncio.sleep(0.01)
                await asyncio.sleep(0.01)
            done_text = "DONE: interrupted" if self.interrupted else "DONE: complete"
            yield {"type": "done", "content": done_text}

        return generator()

    def interrupt(self) -> None:
        self.interrupted = True

    def pause(self) -> None:
        self.paused = True

    def resume(self) -> None:
        self.paused = False

    async def perform_human_action(self, tool_name: str, args: dict) -> ToolResult:
        action = str(args.get("action", "manual"))
        return ToolResult(
            success=True,
            output=f"{tool_name} {action} OK",
            screenshot_b64=PNG_1X1_B64,
            metadata={
                "frame_label": f"{tool_name} · {action}",
                "summary": f"Kullanıcı {tool_name} aracını kullandı.",
                "focus_x": 0.42,
                "focus_y": 0.34,
            },
        )


def _make_app(tmp_path: Path, created_agents: list[FakeAgent]):
    config = AgentConfig(
        llm_provider="gemini",
        llm_model="gemini-3-flash-preview",
        workspace_dir=tmp_path / "workspace",
        memory_dir=tmp_path / "workspace" / ".memory",
    )

    def factory(cfg: AgentConfig) -> FakeAgent:
        agent = FakeAgent(cfg)
        created_agents.append(agent)
        return agent

    return create_live_app(config, agent_factory=factory)


async def _wait_for_completion(
    client: httpx.AsyncClient,
    run_id: str,
    timeout: float = 3.0,
) -> dict:
    deadline = asyncio.get_running_loop().time() + timeout
    last: dict = {}
    while asyncio.get_running_loop().time() < deadline:
        response = await client.get(f"/runs/{run_id}")
        response.raise_for_status()
        last = response.json()
        if last["status"] != "running":
            return last
        await asyncio.sleep(0.05)
    raise AssertionError("Run did not finish in time.")


async def _wait_for_agent_creation(created_agents: list[FakeAgent], timeout: float = 1.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if created_agents:
            return
        await asyncio.sleep(0.01)
    raise AssertionError("Agent was not created in time.")


@pytest.mark.asyncio
async def test_live_app_root_renders_operator_console(tmp_path: Path) -> None:
    app = _make_app(tmp_path, [])
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.get("/")

    assert response.status_code == 200
    assert "Otonom Ajan - Görev Paneli" in response.text
    assert "Logs" in response.text
    assert "Yeni Run" in response.text
    assert "Threadler" in response.text
    assert "Seçili Thread" in response.text
    assert "Bekleyen Onaylar" in response.text
    assert "Bekleyen Sorular" in response.text
    assert "Manuel Browser Kontrolleri" not in response.text
    assert '<div class="section-title">Audit</div>' not in response.text
    assert "Yeni thread'de başlat" in response.text
    assert "Seçili thread'e ekle" in response.text
    assert "Yeni bir komut veya görev girin" in response.text
    assert "Agentra Canlı Kumanda" not in response.text
    assert "Sağlayıcı" not in response.text
    assert "Model" not in response.text


@pytest.mark.asyncio
async def test_live_app_logs_page_renders(tmp_path: Path) -> None:
    app = _make_app(tmp_path, [])
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.get("/logs")

    assert response.status_code == 200
    assert "Agentra Logs" in response.text
    assert "Server Log Tail" in response.text
    assert "agentra-app.log" in response.text


def test_live_app_logs_page_renders_provider_error_hint(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    run_id = "20260322-174415-demo"
    run_dir = workspace / ".runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "index.html").write_text("", encoding="utf-8")
    (run_dir / "events.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "goal": "Demo failure",
                "provider": "gemini",
                "model": "gemini-3-flash-preview",
                "status": "error",
                "started_at": "2026-03-22T17:44:15",
                "finished_at": "2026-03-22T17:44:17",
                "report_path": str(run_dir / "index.html"),
                "events": [
                    {
                        "type": "error",
                        "timestamp": "2026-03-22T17:44:17",
                        "content": (
                            "Gemini quota exceeded for model gemini-3-flash-preview. "
                            "Add billing or wait for quota reset, then retry."
                        ),
                        "details": {
                            "hint": (
                                "Switch the thread to another provider/model, or add "
                                "Gemini billing/credits before retrying."
                            ),
                            "traceback": "traceback text",
                        },
                    }
                ],
                "frames": [],
                "audit": [],
            }
        ),
        encoding="utf-8",
    )
    log_path = workspace / ".logs" / "agentra-app.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("log line", encoding="utf-8")

    html = _render_logs_html(workspace, log_path, run_id=run_id)

    assert "Latest Run Error" in html
    assert "Suggested fix:" in html
    assert "Gemini billing/credits before retrying." in html


@pytest.mark.asyncio
async def test_live_app_run_state_exposes_frames_assets_and_report(tmp_path: Path) -> None:
    created_agents: list[FakeAgent] = []
    app = _make_app(tmp_path, created_agents)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        create_response = await client.post("/runs", json={"goal": "Open python.org and explain it"})
        assert create_response.status_code == 200
        run_id = create_response.json()["run_id"]

        snapshot = await _wait_for_completion(client, run_id)

        assert snapshot["status"] == "completed"
        assert snapshot["frames"][0]["label"] == "browser · navigate"
        assert snapshot["frames"][0]["image_url"].startswith(f"/runs/{run_id}/assets/")
        assert snapshot["report_url"] == f"/runs/{run_id}/report"
        assert snapshot["events"][-1]["type"] == "done"
        assert snapshot["steps"]
        assert any(step["kind"] == "tool" for step in snapshot["steps"])

        asset_response = await client.get(snapshot["frames"][0]["image_url"])
        report_response = await client.get(snapshot["report_url"])

    assert asset_response.status_code == 200
    assert asset_response.content
    assert report_response.status_code == 200
    assert "Agentra Run Report" in report_response.text


@pytest.mark.asyncio
async def test_live_app_chooses_under_the_hood_policy_for_local_document_goal(tmp_path: Path) -> None:
    created_agents: list[FakeAgent] = []
    app = _make_app(tmp_path, created_agents)
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        create_response = await client.post(
            "/runs",
            json={"goal": "Masaustumdeki secondsun klasorunu bul ve icindeki PowerPoint dosyasini varsayilan uygulamayla ac"},
        )
        assert create_response.status_code == 200
        await _wait_for_agent_creation(created_agents)

    agent_config = created_agents[-1].config
    assert agent_config.browser_headless is True
    assert agent_config.local_execution_mode == "under_the_hood"
    assert agent_config.desktop_fallback_policy == "pause_and_ask"


@pytest.mark.asyncio
async def test_live_app_chooses_visible_desktop_policy_for_visual_local_goal(tmp_path: Path) -> None:
    created_agents: list[FakeAgent] = []
    app = _make_app(tmp_path, created_agents)
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        create_response = await client.post(
            "/runs",
            json={"goal": "Masaustune git ve secondsun klasorune gir ve karsina cikan sunumu ac"},
        )
        assert create_response.status_code == 200
        await _wait_for_agent_creation(created_agents)

    agent_config = created_agents[-1].config
    assert agent_config.browser_headless is True
    assert agent_config.local_execution_mode == "visible"
    assert agent_config.desktop_fallback_policy == "visible_control"


@pytest.mark.asyncio
async def test_live_app_respects_explicit_headless_override(tmp_path: Path) -> None:
    created_agents: list[FakeAgent] = []
    app = _make_app(tmp_path, created_agents)
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        create_response = await client.post(
            "/runs",
            json={"goal": "Open python.org and summarize it", "headless": False},
        )
        assert create_response.status_code == 200
        await _wait_for_agent_creation(created_agents)

    assert created_agents[-1].config.browser_headless is False


@pytest.mark.asyncio
async def test_live_app_event_stream_and_stop_endpoint(tmp_path: Path) -> None:
    created_agents: list[FakeAgent] = []
    app = _make_app(tmp_path, created_agents)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        create_response = await client.post("/runs", json={"goal": "Open python.org and stop when asked"})
        run_id = create_response.json()["run_id"]

        await _wait_for_agent_creation(created_agents)
        stop_response = await client.post(f"/runs/{run_id}/stop")
        assert stop_response.status_code == 200
        assert created_agents[-1].interrupted is True

        await _wait_for_completion(client, run_id)

        payloads = []
        async with client.stream("GET", f"/runs/{run_id}/events") as response:
            async for line in response.aiter_lines():
                if not line or line.startswith(":"):
                    continue
                payloads.append(json.loads(line.removeprefix("data: ")))

    assert payloads[0]["kind"] == "snapshot"
    assert payloads[-1]["kind"] == "complete"
    assert any(event["type"] == "phase" for event in payloads[0]["snapshot"]["events"])


@pytest.mark.asyncio
async def test_live_app_exposes_thread_endpoints_and_parallel_runs(tmp_path: Path) -> None:
    created_agents: list[FakeAgent] = []
    app = _make_app(tmp_path, created_agents)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        first = await client.post("/runs", json={"goal": "First run"})
        second = await client.post("/runs", json={"goal": "Second run"})

        assert first.status_code == 200
        assert second.status_code == 200

        first_payload = first.json()
        second_payload = second.json()
        assert first_payload["thread_id"] != second_payload["thread_id"]

        threads_response = await client.get("/threads")
        assert threads_response.status_code == 200
        threads = threads_response.json()["threads"]
        assert len(threads) == 2

        thread_response = await client.get(f"/threads/{first_payload['thread_id']}")
        assert thread_response.status_code == 200
        assert thread_response.json()["thread_id"] == first_payload["thread_id"]
        assert thread_response.json()["current_run_id"] == first_payload["run_id"]


@pytest.mark.asyncio
async def test_live_app_supports_pause_approval_question_and_manual_action(tmp_path: Path) -> None:
    created_agents: list[FakeAgent] = []
    app = _make_app(tmp_path, created_agents)
    manager = app.state.manager
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        create_response = await client.post("/runs", json={"goal": "Pause and test controls"})
        assert create_response.status_code == 200

        payload = create_response.json()
        thread_id = payload["thread_id"]

        await _wait_for_agent_creation(created_agents)
        thread = manager.get_thread(thread_id)

        thread.approval_requests["approval-001"] = ApprovalRequest(
            request_id="approval-001",
            tool="browser",
            args={"action": "click", "selector": "button.submit"},
            reason="Publish benzeri bir işlem tespit edildi.",
            summary="Gönder düğmesine basmadan önce onay gerekiyor.",
        )
        thread.question_requests["question-001"] = UserInputRequest(
            request_id="question-001",
            prompt="Hangi dosya adı kullanılsın?",
            summary="Devam etmeden önce bir dosya adı gerekli.",
        )
        thread.status = "blocked_waiting_user"

        approval_response = await client.post(
            f"/threads/{thread_id}/approvals/approval-001",
            json={"approved": True, "note": "Devam et"},
        )
        assert approval_response.status_code == 200
        approval_payload = approval_response.json()
        assert any(
            item["request_id"] == "approval-001" and item["status"] == "approved"
            for item in approval_payload["approval_requests"]
        )

        thread.status = "blocked_waiting_user"
        question_response = await client.post(
            f"/threads/{thread_id}/questions/question-001",
            json={"answer": "report.md"},
        )
        assert question_response.status_code == 200
        question_payload = question_response.json()
        assert any(
            item["request_id"] == "question-001" and item["status"] == "answered"
            for item in question_payload["question_requests"]
        )

        pause_response = await client.post(f"/threads/{thread_id}/pause")
        assert pause_response.status_code == 200
        assert pause_response.json()["status"] == "paused_for_user"
        assert created_agents[-1].paused is True

        action_response = await client.post(
            f"/threads/{thread_id}/actions",
            json={"tool": "browser", "args": {"action": "back"}},
        )
        assert action_response.status_code == 200
        action_payload = action_response.json()
        event_types = [event["type"] for event in action_payload["active_run"]["events"]]
        assert "human_action" in event_types
        assert "tool_result" in event_types

        key_response = await client.post(
            f"/threads/{thread_id}/actions",
            json={"tool": "browser", "args": {"action": "key", "key": "Enter"}},
        )
        assert key_response.status_code == 200
        key_payload = key_response.json()
        assert any(
            event["type"] == "tool_result"
            and event["metadata"].get("frame_label") == "browser · key"
            for event in key_payload["active_run"]["events"]
        )

        computer_response = await client.post(
            f"/threads/{thread_id}/actions",
            json={"tool": "computer", "args": {"action": "click", "x": 240, "y": 160}},
        )
        assert computer_response.status_code == 200
        computer_payload = computer_response.json()
        assert any(
            event["type"] == "tool_result"
            and event["tool"] == "computer"
            and event["metadata"].get("frame_label") == "computer · click"
            for event in computer_payload["active_run"]["events"]
        )

        resume_response = await client.post(f"/threads/{thread_id}/resume")
        assert resume_response.status_code == 200
        assert created_agents[-1].paused is False

        await _wait_for_completion(client, payload["run_id"])


@pytest.mark.asyncio
async def test_live_app_exposes_live_browser_frame_route(tmp_path: Path) -> None:
    created_agents: list[FakeAgent] = []
    app = _make_app(tmp_path, created_agents)
    manager = app.state.manager
    transport = httpx.ASGITransport(app=app)

    async def fake_capture(thread_id: str):
        assert thread_id.startswith("thread-")
        return SimpleNamespace(data=b"jpeg-live-frame", media_type="image/jpeg")

    manager.capture_live_browser_frame = fake_capture

    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        create_response = await client.post("/runs", json={"goal": "Open python.org live"})
        assert create_response.status_code == 200
        thread_id = create_response.json()["thread_id"]

        frame_response = await client.get(f"/threads/{thread_id}/live-frame")

    assert frame_response.status_code == 200
    assert frame_response.headers["content-type"] == "image/jpeg"
    assert frame_response.content == b"jpeg-live-frame"


@pytest.mark.asyncio
async def test_live_app_exposes_live_browser_stream_route(tmp_path: Path) -> None:
    created_agents: list[FakeAgent] = []
    app = _make_app(tmp_path, created_agents)
    manager = app.state.manager
    transport = httpx.ASGITransport(app=app)

    async def fake_capture(thread_id: str):
        assert thread_id.startswith("thread-")
        return SimpleNamespace(data=b"jpeg-live-frame", media_type="image/jpeg")

    manager.capture_live_browser_frame = fake_capture

    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        create_response = await client.post("/runs", json={"goal": "Open python.org live"})
        assert create_response.status_code == 200
        thread_id = create_response.json()["thread_id"]

        async with client.stream("GET", f"/threads/{thread_id}/live-stream?stream=1") as stream_response:
            assert stream_response.status_code == 200
            assert stream_response.headers["content-type"].startswith("multipart/x-mixed-replace")
            first_chunk = await stream_response.aiter_bytes().__anext__()

    assert b"Content-Type: image/jpeg" in first_chunk
    assert b"jpeg-live-frame" in first_chunk


@pytest.mark.asyncio
async def test_live_app_exposes_live_desktop_frame_route(tmp_path: Path) -> None:
    created_agents: list[FakeAgent] = []
    app = _make_app(tmp_path, created_agents)
    manager = app.state.manager
    transport = httpx.ASGITransport(app=app)

    async def fake_capture(thread_id: str):
        assert thread_id.startswith("thread-")
        return SimpleNamespace(data=b"desktop-live-frame", media_type="image/png")

    manager.capture_live_computer_frame = fake_capture

    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        create_response = await client.post("/runs", json={"goal": "Open desktop live"})
        assert create_response.status_code == 200
        thread_id = create_response.json()["thread_id"]

        frame_response = await client.get(f"/threads/{thread_id}/desktop-frame")

    assert frame_response.status_code == 200
    assert frame_response.headers["content-type"] == "image/png"
    assert frame_response.content == b"desktop-live-frame"


@pytest.mark.asyncio
async def test_live_app_exposes_live_desktop_stream_route(tmp_path: Path) -> None:
    created_agents: list[FakeAgent] = []
    app = _make_app(tmp_path, created_agents)
    manager = app.state.manager
    transport = httpx.ASGITransport(app=app)

    async def fake_capture(thread_id: str):
        assert thread_id.startswith("thread-")
        return SimpleNamespace(data=b"desktop-live-frame", media_type="image/png")

    manager.capture_live_computer_frame = fake_capture

    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        create_response = await client.post("/runs", json={"goal": "Open desktop stream"})
        assert create_response.status_code == 200
        thread_id = create_response.json()["thread_id"]

        async with client.stream("GET", f"/threads/{thread_id}/desktop-stream?stream=1") as stream_response:
            assert stream_response.status_code == 200
            assert stream_response.headers["content-type"].startswith("multipart/x-mixed-replace")
            first_chunk = await stream_response.aiter_bytes().__anext__()

    assert b"Content-Type: image/png" in first_chunk
    assert b"desktop-live-frame" in first_chunk
