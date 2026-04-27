"""Browser-level regressions for live app thread switching."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from agentra.config import AgentConfig
from agentra.live_app import create_live_app

PNG_1X1_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)
PNG_URL = f"data:image/png;base64,{PNG_1X1_B64}"
PNG_RESPONSE_SCRIPT = """
const binaryFrameResponse = (status = 200) => {
  if (status === 204) {
    return new Response(null, { status: 204 });
  }
  const bytes = Uint8Array.from(atob(PNG_1X1_B64), (char) => char.charCodeAt(0));
  return new Response(bytes, {
    status,
    headers: {
      "Content-Type": "image/png",
      "Cache-Control": "no-store",
    },
  });
};
"""


def _run_snapshot(thread_id: str, thread_title: str, run_id: str, goal: str, summary: str) -> dict:
    return {
        "run_id": run_id,
        "goal": goal,
        "status": "running",
        "provider": "gemini",
        "model": "gemini-3-flash-preview",
        "permission_mode": "default",
        "browser_identity": "isolated",
        "browser_profile_name": "Default",
        "activity": {
            "channel": "browser",
            "visibility": "visible",
            "status": "visible_update",
            "title": "Tarayıcı · Aç",
            "summary": summary,
        },
        "events": [
            {
                "type": "screenshot",
                "timestamp": "2026-03-16T12:00:00",
                "image_url": PNG_URL,
                "frame_id": f"{run_id}-frame-1",
                "frame_label": "browser · navigate",
                "display_label": "browser · navigate",
                "summary": summary,
                "display_summary": summary,
                "focus_x": 0.5,
                "focus_y": 0.4,
            }
        ],
        "frames": [
            {
                "id": f"{run_id}-frame-1",
                "timestamp": "2026-03-16T12:00:00",
                "label": "browser · navigate",
                "display_label": "browser · navigate",
                "summary": summary,
                "display_summary": summary,
                "image_url": PNG_URL,
                "focus_x": 0.5,
                "focus_y": 0.4,
            }
        ],
        "steps": [],
        "report_url": f"/runs/{run_id}/report",
        "thread_id": thread_id,
        "thread_title": thread_title,
        "thread_status": "running",
        "handoff_state": "agent",
        "approval_requests": [],
        "question_requests": [],
        "active": True,
        "browser_session_active": True,
        "active_url": "https://example.com",
        "active_title": "Example",
        "tab_count": 1,
        "browser": {
            "active": True,
            "active_url": "https://example.com",
            "active_title": "Example",
            "tab_count": 1,
            "identity": "isolated",
            "profile_name": "Default",
            "last_error": "",
        },
    }


def _thread_snapshot(thread_id: str, title: str, run_id: str, goal: str, summary: str) -> dict:
    active_run = _run_snapshot(thread_id, title, run_id, goal, summary)
    return {
        "thread_id": thread_id,
        "title": title,
        "status": "running",
        "handoff_state": "agent",
        "permission_mode": "default",
        "browser_identity": "isolated",
        "browser_profile_name": "Default",
        "created_at": "2026-03-16T12:00:00",
        "workspace_dir": f"C:/tmp/{thread_id}/workspace",
        "memory_dir": f"C:/tmp/{thread_id}/workspace/.memory",
        "current_run_id": run_id,
        "runs": [
            {
                "run_id": run_id,
                "goal": goal,
                "status": "running",
                "started_at": "2026-03-16T12:00:00",
                "finished_at": None,
                "report_url": f"/runs/{run_id}/report",
            }
        ],
        "approval_requests": [],
        "question_requests": [],
        "activity": active_run["activity"],
        "activity_summary": summary,
        "activity_title": "Tarayıcı · Aç",
        "active_run": active_run,
        "browser_session_active": True,
        "active_url": "https://example.com",
        "active_title": "Example",
        "tab_count": 1,
        "browser": {
            "active": True,
            "active_url": "https://example.com",
            "active_title": "Example",
            "tab_count": 1,
            "identity": "isolated",
            "profile_name": "Default",
            "last_error": "",
        },
    }


def _paused_thread_snapshot(thread_id: str, title: str, run_id: str, goal: str, summary: str) -> dict:
    payload = _thread_snapshot(thread_id, title, run_id, goal, summary)
    return _paused_snapshot(payload)


def _paused_snapshot(payload: dict) -> dict:
    payload = json.loads(json.dumps(payload))
    payload["status"] = "paused_for_user"
    payload["handoff_state"] = "user"
    if payload.get("active_run"):
        payload["active_run"]["thread_status"] = "paused_for_user"
        payload["active_run"]["handoff_state"] = "user"
    return payload


def _computer_activity_thread_snapshot(thread_id: str, title: str, run_id: str, goal: str, summary: str) -> dict:
    active_run = {
        "run_id": run_id,
        "goal": goal,
        "status": "running",
        "provider": "gemini",
        "model": "gemini-3-flash-preview",
        "permission_mode": "default",
        "browser_identity": "isolated",
        "browser_profile_name": "Default",
        "activity": {
            "channel": "desktop",
            "visibility": "visible",
            "status": "running",
            "title": "Masaüstü · Tuş",
            "summary": summary,
        },
        "events": [
            {
                "type": "tool_call",
                "timestamp": "2026-03-23T12:00:00",
                "tool": "computer",
                "args": {"action": "key", "text": "win+d"},
                "summary": summary,
                "display_summary": summary,
                "display_label": "Masaüstü · Tuş",
            }
        ],
        "frames": [],
        "steps": [],
        "report_url": f"/runs/{run_id}/report",
        "thread_id": thread_id,
        "thread_title": title,
        "thread_status": "running",
        "handoff_state": "agent",
        "approval_requests": [],
        "question_requests": [],
        "active": True,
        "browser_session_active": False,
        "active_url": "",
        "active_title": "",
        "tab_count": 0,
        "browser": {
            "active": False,
            "active_url": "",
            "active_title": "",
            "tab_count": 0,
        },
    }
    return {
        "thread_id": thread_id,
        "title": title,
        "status": "running",
        "handoff_state": "agent",
        "permission_mode": "default",
        "browser_identity": "isolated",
        "browser_profile_name": "Default",
        "created_at": "2026-03-23T12:00:00",
        "workspace_dir": f"C:/tmp/{thread_id}/workspace",
        "memory_dir": f"C:/tmp/{thread_id}/workspace/.memory",
        "current_run_id": run_id,
        "runs": [
            {
                "run_id": run_id,
                "goal": goal,
                "status": "running",
                "started_at": "2026-03-23T12:00:00",
                "finished_at": None,
                "report_url": f"/runs/{run_id}/report",
            }
        ],
        "approval_requests": [],
        "question_requests": [],
        "activity": active_run["activity"],
        "activity_summary": summary,
        "activity_title": "Masaüstü · Tuş",
        "active_run": active_run,
        "browser_session_active": False,
        "active_url": "",
        "active_title": "",
        "tab_count": 0,
        "browser": {
            "active": False,
            "active_url": "",
            "active_title": "",
            "tab_count": 0,
            "identity": "isolated",
            "profile_name": "Default",
            "last_error": "",
        },
    }


def _local_system_activity_thread_snapshot(
    thread_id: str, title: str, run_id: str, goal: str, summary: str
) -> dict:
    active_run = {
        "run_id": run_id,
        "goal": goal,
        "status": "running",
        "provider": "gemini",
        "model": "gemini-3-flash-preview",
        "permission_mode": "default",
        "browser_identity": "isolated",
        "browser_profile_name": "Default",
        "activity": {
            "channel": "local_system",
            "visibility": "hidden",
            "status": "running",
            "title": "Yerel Sistem · Aç",
            "summary": summary,
        },
        "events": [
            {
                "type": "tool_call",
                "timestamp": "2026-03-23T12:00:00",
                "tool": "local_system",
                "args": {"action": "open_path", "path": "/mnt/c/Users/ariba/OneDrive/Desktop/secondsun/deck.pptx"},
                "summary": summary,
                "display_summary": summary,
                "display_label": "Yerel Sistem · Aç",
            }
        ],
        "frames": [],
        "steps": [],
        "report_url": f"/runs/{run_id}/report",
        "thread_id": thread_id,
        "thread_title": title,
        "thread_status": "running",
        "handoff_state": "agent",
        "approval_requests": [],
        "question_requests": [],
        "active": True,
        "browser_session_active": False,
        "active_url": "",
        "active_title": "",
        "tab_count": 0,
        "browser": {
            "active": False,
            "active_url": "",
            "active_title": "",
            "tab_count": 0,
        },
    }
    return {
        "thread_id": thread_id,
        "title": title,
        "status": "running",
        "handoff_state": "agent",
        "permission_mode": "default",
        "browser_identity": "isolated",
        "browser_profile_name": "Default",
        "created_at": "2026-03-23T12:00:00",
        "workspace_dir": f"C:/tmp/{thread_id}/workspace",
        "memory_dir": f"C:/tmp/{thread_id}/workspace/.memory",
        "current_run_id": run_id,
        "runs": [
            {
                "run_id": run_id,
                "goal": goal,
                "status": "running",
                "started_at": "2026-03-23T12:00:00",
                "finished_at": None,
                "report_url": f"/runs/{run_id}/report",
            }
        ],
        "approval_requests": [],
        "question_requests": [],
        "activity": active_run["activity"],
        "activity_summary": summary,
        "activity_title": "Yerel Sistem · Aç",
        "active_run": active_run,
        "browser_session_active": False,
        "active_url": "",
        "active_title": "",
        "tab_count": 0,
        "browser": {
            "active": False,
            "active_url": "",
            "active_title": "",
            "tab_count": 0,
            "identity": "isolated",
            "profile_name": "Default",
            "last_error": "",
        },
    }


def _idle_thread_snapshot(thread_id: str, title: str) -> dict:
    return {
        "thread_id": thread_id,
        "title": title,
        "status": "idle",
        "handoff_state": "agent",
        "permission_mode": "default",
        "browser_identity": "isolated",
        "browser_profile_name": "Default",
        "created_at": "2026-03-23T12:00:00",
        "workspace_dir": f"C:/tmp/{thread_id}/workspace",
        "memory_dir": f"C:/tmp/{thread_id}/workspace/.memory",
        "current_run_id": None,
        "runs": [],
        "approval_requests": [],
        "question_requests": [],
        "active_run": None,
        "activity": {"channel": "agent", "visibility": "background", "status": "idle", "title": "Hazır", "summary": ""},
        "activity_summary": "",
        "activity_title": "Hazır",
        "browser_session_active": False,
        "active_url": "",
        "active_title": "",
        "tab_count": 0,
        "browser": {
            "active": False,
            "active_url": "",
            "active_title": "",
            "tab_count": 0,
            "identity": "isolated",
            "profile_name": "Default",
            "last_error": "",
        },
    }


async def _root_html(tmp_path: Path) -> str:
    config = AgentConfig(
        llm_provider="gemini",
        llm_model="gemini-3-flash-preview",
        workspace_dir=tmp_path / "workspace",
        memory_dir=tmp_path / "workspace" / ".memory",
    )
    app = create_live_app(config)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.get("/")
    response.raise_for_status()
    return response.text


@pytest.mark.asyncio
async def test_live_app_thread_switch_clears_old_tv_and_last_selection_wins(tmp_path: Path) -> None:
    playwright = pytest.importorskip("playwright.async_api")
    from playwright.async_api import Error as PlaywrightError
    from playwright.async_api import async_playwright

    html = await _root_html(tmp_path)

    thread_one = _thread_snapshot(
        "thread-one",
        "Thread One",
        "run-one",
        "First goal",
        "Thread one snapshot",
    )
    thread_two = _thread_snapshot(
        "thread-two",
        "Thread Two",
        "run-two",
        "Second goal",
        "Thread two snapshot",
    )

    responses = {
        "/threads": {
            "body": {
                "threads": [thread_one, thread_two],
            }
        },
        "/threads/thread-one": [
            {"delay": 20, "body": thread_one},
            {"delay": 160, "body": thread_one},
        ],
        "/threads/thread-two": [
            {"delay": 150, "body": thread_two},
            {"delay": 30, "body": thread_two},
        ],
    }

    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            page = await browser.new_page()
            await page.add_init_script(
                script=f"""
                (() => {{
                  const responses = {json.dumps(responses)};
                  const counts = {{}};
                  const wait = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

                  class MockEventSource {{
                    constructor(url) {{
                      this.url = url;
                      this.readyState = 1;
                      this.onmessage = null;
                      this.onerror = null;
                    }}
                    close() {{
                      this.readyState = 2;
                    }}
                  }}

                  window.EventSource = MockEventSource;
                  window.fetch = async (url) => {{
                    const key = typeof url === "string" ? url : (url?.url || String(url));
                    counts[key] = (counts[key] || 0) + 1;
                    const value = responses[key];
                    if (!value) {{
                      throw new Error(`No mock response for ${{key}}`);
                    }}
                    const entry = Array.isArray(value)
                      ? value[Math.min(counts[key] - 1, value.length - 1)]
                      : value;
                    await wait(entry.delay || 0);
                    return {{
                      ok: (entry.status || 200) < 400,
                      status: entry.status || 200,
                      async json() {{
                        return entry.body;
                      }},
                      async text() {{
                        return JSON.stringify(entry.body);
                      }},
                    }};
                  }};
                }})();
                """
            )

            await page.set_content(html, wait_until="load")

            await page.wait_for_function(
                "() => document.getElementById('tv-frame-title').textContent.includes('Thread One')"
            )
            await page.wait_for_function(
                "() => document.getElementById('scrubber-label').textContent.trim() === '1 / 1'"
            )

            await page.locator("[data-thread-select='thread-two']").click()
            await page.wait_for_function(
                "() => document.getElementById('tv-frame-title').textContent.includes('Thread Two')"
            )
            await page.wait_for_function(
                "() => document.getElementById('scrubber-label').textContent.trim() === '0 / 0'"
            )
            loading_text = await page.locator("#tv-screen").inner_text()
            assert "yükleniyor" in loading_text.lower()
            assert "Thread one snapshot" not in loading_text

            await page.wait_for_function(
                "() => document.getElementById('tv-frame-title').textContent.includes('Thread Two') && document.getElementById('scrubber-label').textContent.trim() === '1 / 1'"
            )

            await page.locator("[data-thread-select='thread-one']").click()
            await page.wait_for_function(
                "() => document.getElementById('scrubber-label').textContent.trim() === '0 / 0'"
            )
            await page.locator("[data-thread-select='thread-two']").click()

            await page.wait_for_function(
                "() => document.getElementById('tv-frame-title').textContent.includes('Thread Two') && document.getElementById('scrubber-label').textContent.trim() === '1 / 1'"
            )
            await page.wait_for_timeout(220)

            final_title = await page.locator("#tv-frame-title").text_content()
            final_label = await page.locator("#scrubber-label").text_content()
            assert final_title is not None and "Thread Two" in final_title
            assert final_label == "1 / 1"

            await browser.close()
    except PlaywrightError as exc:
        pytest.skip(f"Playwright browser is unavailable: {exc}")


@pytest.mark.asyncio
async def test_live_app_interact_mode_pauses_and_routes_manual_inputs(tmp_path: Path) -> None:
    playwright = pytest.importorskip("playwright.async_api")
    from playwright.async_api import Error as PlaywrightError
    from playwright.async_api import async_playwright

    html = await _root_html(tmp_path)

    running_thread = _thread_snapshot(
        "thread-live",
        "Live Thread",
        "run-live",
        "Interact goal",
        "Live snapshot",
    )
    paused_thread = _paused_thread_snapshot(
        "thread-live",
        "Live Thread",
        "run-live",
        "Interact goal",
        "Paused snapshot",
    )

    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            page = await browser.new_page()
            await page.add_init_script(
                script=f"""
                (() => {{
                  const runningThread = {json.dumps(running_thread)};
                  const pausedThread = {json.dumps(paused_thread)};
                  window.__agentraBaseUrl = "http://testserver";
                  window.__fetchCalls = [];

                  class MockEventSource {{
                    constructor(url) {{
                      this.url = url;
                      this.readyState = 1;
                      this.onmessage = null;
                      this.onerror = null;
                    }}
                    close() {{
                      this.readyState = 2;
                    }}
                  }}

                  const jsonResponse = (body, status = 200) => ({{
                    ok: status < 400,
                    status,
                    async json() {{
                      return body;
                    }},
                    async text() {{
                      return JSON.stringify(body);
                    }},
                  }});

                  window.EventSource = MockEventSource;
                  window.fetch = async (input, options = {{}}) => {{
                    const rawUrl = typeof input === "string" ? input : (input?.url || String(input));
                    const url = new URL(rawUrl, "http://testserver");
                    const method = String(options.method || "GET").toUpperCase();
                    const body = options.body ? JSON.parse(options.body) : null;
                    const key = `${{method}} ${{url.pathname}}`;
                    window.__fetchCalls.push({{ key, body }});

                    if (key === "GET /threads") {{
                      return jsonResponse({{ threads: [runningThread] }});
                    }}
                    if (key === "GET /threads/thread-live") {{
                      return jsonResponse(runningThread);
                    }}
                    if (key === "POST /threads/thread-live/pause") {{
                      return jsonResponse(pausedThread);
                    }}
                    if (key === "POST /threads/thread-live/resume") {{
                      return jsonResponse(runningThread);
                    }}
                    if (key === "POST /threads/thread-live/actions") {{
                      return jsonResponse(pausedThread);
                    }}
                    throw new Error(`No mock response for ${{key}}`);
                  }};
                }})();
                """
            )

            await page.set_content(html, wait_until="load")
            await page.wait_for_function(
                "() => { const title = document.getElementById('tv-frame-title').textContent; return title.includes('Ajan') || title.includes('Live Thread'); }"
            )
            await page.wait_for_function(
                "() => document.getElementById('scrubber-label').textContent.trim() === '1 / 1'"
            )

            await page.locator("[data-manual-action='interact']").click()
            await page.wait_for_function(
                "() => window.__fetchCalls.some((call) => call.key === 'POST /threads/thread-live/pause')"
            )
            await page.wait_for_function(
                "() => document.activeElement && document.activeElement.id === 'tv-screen'"
            )
            await page.wait_for_function(
                "() => document.getElementById('cursor-bubble').style.display === 'none'"
            )

            await page.locator("#tv-image").click()
            await page.wait_for_function(
                "() => window.__fetchCalls.some((call) => call.key === 'POST /threads/thread-live/actions' && call.body?.args?.action === 'click')"
            )

            box = await page.locator("#tv-image").bounding_box()
            assert box is not None
            await page.mouse.move(box["x"] + 120, box["y"] + 120)
            await page.mouse.down()
            await page.mouse.move(box["x"] + 220, box["y"] + 160, steps=10)
            await page.mouse.up()
            await page.wait_for_function(
                "() => window.__fetchCalls.some((call) => call.key === 'POST /threads/thread-live/actions' && call.body?.args?.action === 'drag')"
            )

            await page.locator("#tv-image").hover()
            await page.mouse.wheel(0, 720)
            await page.wait_for_function(
                "() => window.__fetchCalls.some((call) => call.key === 'POST /threads/thread-live/actions' && call.body?.args?.action === 'scroll')"
            )

            await page.keyboard.type("abc")
            await page.wait_for_function(
                "() => window.__fetchCalls.some((call) => call.key === 'POST /threads/thread-live/actions' && call.body?.args?.action === 'type' && call.body?.args?.text === 'abc')"
            )

            await page.keyboard.press("Enter")
            await page.wait_for_function(
                "() => window.__fetchCalls.some((call) => call.key === 'POST /threads/thread-live/actions' && call.body?.args?.action === 'key' && call.body?.args?.key === 'Enter')"
            )

            await page.keyboard.press("Backspace")
            await page.wait_for_function(
                "() => window.__fetchCalls.some((call) => call.key === 'POST /threads/thread-live/actions' && call.body?.args?.action === 'key' && call.body?.args?.key === 'Backspace')"
            )

            await page.locator("[data-thread-action='resume']").click()
            await page.wait_for_function(
                "() => window.__fetchCalls.some((call) => call.key === 'POST /threads/thread-live/resume')"
            )
            await page.wait_for_function(
                "() => !document.getElementById('tv-screen').classList.contains('manual-interact')"
            )

            fetch_calls = await page.evaluate("window.__fetchCalls")
            action_bodies = [
                item["body"]["args"]
                for item in fetch_calls
                if item["key"] == "POST /threads/thread-live/actions"
            ]
            assert any(body["action"] == "click" and body["x"] > 0 and body["y"] > 0 for body in action_bodies)
            assert any(body["action"] == "drag" and body["start_x"] > 0 and body["end_x"] > body["start_x"] for body in action_bodies)
            assert any(body["action"] == "scroll" and body["amount"] == 720 for body in action_bodies)
            assert any(body["action"] == "type" and body["text"] == "abc" for body in action_bodies)
            assert any(body["action"] == "key" and body["key"] == "Enter" for body in action_bodies)
            assert any(body["action"] == "key" and body["key"] == "Backspace" for body in action_bodies)

            await browser.close()
    except PlaywrightError as exc:
        pytest.skip(f"Playwright browser is unavailable: {exc}")


@pytest.mark.asyncio
async def test_live_app_shows_takeover_banner_and_finish_control_for_sensitive_browser_pause(tmp_path: Path) -> None:
    playwright = pytest.importorskip("playwright.async_api")
    from playwright.async_api import Error as PlaywrightError
    from playwright.async_api import async_playwright

    html = await _root_html(tmp_path)

    running_thread = _thread_snapshot(
        "thread-live",
        "Live Thread",
        "run-live",
        "Login goal",
        "Running snapshot",
    )
    paused_thread = _paused_thread_snapshot(
        "thread-live",
        "Live Thread",
        "run-live",
        "Login goal",
        "Hassas tarayici adimi icin kontrol size devredildi.",
    )
    paused_event = {
        "type": "paused",
        "timestamp": "2026-03-23T12:00:01",
        "content": (
            "Parola, kod veya giris onayi gereken bu adimi tarayicida manuel tamamlayin. "
            "Bitince Finish Control ile ajani devam ettirin."
        ),
        "summary": "Hassas tarayici adimi icin kontrol size devredildi.",
        "pause_kind": "sensitive_browser_takeover",
        "takeover_kind": "secret",
        "display_label": "Kontrol Sende",
        "display_summary": "Hassas tarayici adimi icin kontrol size devredildi.",
        "activity": {
            "channel": "browser",
            "visibility": "visible",
            "status": "waiting",
            "title": "Kontrol Sende",
            "summary": "Hassas tarayici adimi icin kontrol size devredildi.",
        },
    }
    paused_thread["active_run"]["events"].append(paused_event)
    paused_thread["active_run"]["activity"] = paused_event["activity"]
    paused_thread["activity"] = paused_event["activity"]
    paused_thread["activity_summary"] = paused_event["activity"]["summary"]
    paused_thread["activity_title"] = paused_event["activity"]["title"]

    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            page = await browser.new_page()
            await page.add_init_script(
                script=f"""
                (() => {{
                  const pausedThread = {json.dumps(paused_thread)};
                  const runningThread = {json.dumps(running_thread)};
                  const PNG_1X1_B64 = "{PNG_1X1_B64}";
                  window.__agentraBaseUrl = "http://testserver";
                  window.__fetchCalls = [];

                  class MockEventSource {{
                    constructor(url) {{
                      this.url = url;
                      this.readyState = 1;
                      this.onmessage = null;
                      this.onerror = null;
                    }}
                    close() {{
                      this.readyState = 2;
                    }}
                  }}

                  const jsonResponse = (body, status = 200) => ({{
                    ok: status < 400,
                    status,
                    async json() {{
                      return body;
                    }},
                    async text() {{
                      return JSON.stringify(body);
                    }},
                  }});
                  {PNG_RESPONSE_SCRIPT}

                  window.EventSource = MockEventSource;
                  window.fetch = async (input, options = {{}}) => {{
                    const rawUrl = typeof input === "string" ? input : (input?.url || String(input));
                    const url = new URL(rawUrl, "http://testserver");
                    const method = String(options.method || "GET").toUpperCase();
                    const body = options.body ? JSON.parse(options.body) : null;
                    const key = `${{method}} ${{url.pathname}}`;
                    window.__fetchCalls.push({{ key, body }});

                    if (key === "GET /threads") {{
                      return jsonResponse({{ threads: [pausedThread] }});
                    }}
                    if (key === "GET /threads/thread-live") {{
                      return jsonResponse(pausedThread);
                    }}
                    if (key === "GET /threads/thread-live/live-frame") {{
                      return binaryFrameResponse();
                    }}
                    if (key === "POST /threads/thread-live/resume") {{
                      return jsonResponse(runningThread);
                    }}
                    throw new Error(`No mock response for ${{key}}`);
                  }};
                }})();
                """
            )

            await page.set_content(html, wait_until="load")
            await page.wait_for_function(
                "() => document.body.innerText.includes('Kontrol Sende') && document.body.innerText.includes('Finish Control')"
            )
            await page.wait_for_function(
                "() => document.getElementById('tv-screen').classList.contains('manual-interact')"
            )
            await page.wait_for_function(
                "() => window.__fetchCalls.some((call) => call.key === 'GET /threads/thread-live/live-frame')"
            )
            await page.wait_for_function(
                "() => document.getElementById('tv-image').src.startsWith('blob:')"
            )

            await page.evaluate(
                "() => { state.ui.controlLayer = 'desktop'; state.ui.manualLayerOverride = 'desktop'; render(); }"
            )
            await page.wait_for_function(
                "() => document.getElementById('tv-frame-title').textContent.includes('TARAYICI')"
            )
            await page.wait_for_function(
                "() => document.getElementById('tv-image').src.startsWith('blob:')"
            )

            dock_text = await page.locator("#stage-manual-dock").inner_text()
            assert "Kontrol Sende" in dock_text
            assert "Finish Control" in dock_text

            await page.locator("[data-thread-action='resume']").first.click()
            await page.wait_for_function(
                "() => window.__fetchCalls.some((call) => call.key === 'POST /threads/thread-live/resume')"
            )

            await browser.close()
    except PlaywrightError as exc:
        pytest.skip(f"Playwright browser is unavailable: {exc}")


@pytest.mark.asyncio
@pytest.mark.parametrize("takeover_tool", ["computer", "windows_desktop"])
async def test_live_app_shows_takeover_banner_for_desktop_control_pause(
    tmp_path: Path,
    takeover_tool: str,
) -> None:
    playwright = pytest.importorskip("playwright.async_api")
    from playwright.async_api import Error as PlaywrightError
    from playwright.async_api import async_playwright

    html = await _root_html(tmp_path)

    running_thread = _computer_activity_thread_snapshot(
        "thread-live",
        "Live Thread",
        "run-live",
        "Desktop goal",
        "Masaustu hedefini hazirliyor",
    )
    paused_thread = _paused_snapshot(running_thread)
    paused_event = {
        "type": "paused",
        "timestamp": "2026-03-23T12:00:01",
        "tool": takeover_tool,
        "content": (
            "Gorunur masaustu otomasyonu icin hedef pencereyi one getirin; "
            "hazir olunca Finish Control ile devam edin."
        ),
        "summary": "Masaustu gorunumu Agentra penceresinde kaldigi icin kontrol size devredildi.",
        "pause_kind": "desktop_control_takeover",
        "display_label": "Kontrol Sende",
        "display_summary": "Masaustu gorunumu Agentra penceresinde kaldigi icin kontrol size devredildi.",
        "activity": {
            "channel": "desktop",
            "visibility": "visible",
            "status": "waiting",
            "title": "Kontrol Sende",
            "summary": "Masaustu gorunumu Agentra penceresinde kaldigi icin kontrol size devredildi.",
        },
    }
    paused_thread["active_run"]["events"].append(paused_event)
    paused_thread["active_run"]["activity"] = paused_event["activity"]
    paused_thread["activity"] = paused_event["activity"]
    paused_thread["activity_summary"] = paused_event["activity"]["summary"]
    paused_thread["activity_title"] = paused_event["activity"]["title"]

    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            page = await browser.new_page()
            await page.add_init_script(
                script=f"""
                (() => {{
                  const pausedThread = {json.dumps(paused_thread)};
                  const runningThread = {json.dumps(running_thread)};
                  const PNG_1X1_B64 = "{PNG_1X1_B64}";
                  window.__agentraBaseUrl = "http://testserver";
                  window.__fetchCalls = [];

                  class MockEventSource {{
                    constructor(url) {{
                      this.url = url;
                      this.readyState = 1;
                      this.onmessage = null;
                      this.onerror = null;
                    }}
                    close() {{
                      this.readyState = 2;
                    }}
                  }}

                  const jsonResponse = (body, status = 200) => ({{
                    ok: status < 400,
                    status,
                    async json() {{
                      return body;
                    }},
                    async text() {{
                      return JSON.stringify(body);
                    }},
                  }});
                  {PNG_RESPONSE_SCRIPT}

                  window.EventSource = MockEventSource;
                  window.fetch = async (input, options = {{}}) => {{
                    const rawUrl = typeof input === "string" ? input : (input?.url || String(input));
                    const url = new URL(rawUrl, "http://testserver");
                    const method = String(options.method || "GET").toUpperCase();
                    const key = `${{method}} ${{url.pathname}}`;
                    window.__fetchCalls.push({{ key }});

                    if (key === "GET /threads") {{
                      return jsonResponse({{ threads: [pausedThread] }});
                    }}
                    if (key === "GET /threads/thread-live") {{
                      return jsonResponse(pausedThread);
                    }}
                    if (key === "GET /threads/thread-live/desktop-frame") {{
                      return binaryFrameResponse();
                    }}
                    if (key === "POST /threads/thread-live/resume") {{
                      return jsonResponse(runningThread);
                    }}
                    throw new Error(`No mock response for ${{key}}`);
                  }};
                }})();
                """
            )

            await page.set_content(html, wait_until="load")
            await page.wait_for_function(
                "() => document.body.innerText.includes('Kontrol Sende') && document.body.innerText.includes('Finish Control')"
            )
            await page.wait_for_function(
                "() => document.getElementById('tv-frame-title').textContent.includes('MASAÜSTÜ')"
            )
            await page.wait_for_function(
                "() => window.__fetchCalls.some((call) => call.key === 'GET /threads/thread-live/desktop-frame')"
            )

            dock_text = await page.locator("#stage-manual-dock").inner_text()
            assert "Kontrol Sende" in dock_text
            assert "Finish Control" in dock_text

            await browser.close()
    except PlaywrightError as exc:
        pytest.skip(f"Playwright browser is unavailable: {exc}")


@pytest.mark.asyncio
async def test_live_app_new_thread_auto_selects_even_if_initial_refresh_is_in_flight(tmp_path: Path) -> None:
    playwright = pytest.importorskip("playwright.async_api")
    from playwright.async_api import Error as PlaywrightError
    from playwright.async_api import async_playwright

    html = await _root_html(tmp_path)

    existing_thread = _thread_snapshot(
        "thread-existing",
        "Existing Thread",
        "run-existing",
        "Existing goal",
        "Existing snapshot",
    )
    new_thread = _thread_snapshot(
        "thread-new",
        "New Thread",
        "run-new",
        "New goal",
        "New snapshot",
    )

    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            page = await browser.new_page()
            await page.add_init_script(
                script=f"""
                (() => {{
                  const existingThread = {json.dumps(existing_thread)};
                  const newThread = {json.dumps(new_thread)};
                  let threadListCalls = 0;
                  window.__agentraBaseUrl = "http://testserver";

                  class MockEventSource {{
                    constructor(url) {{
                      this.url = url;
                      this.readyState = 1;
                      this.onmessage = null;
                      this.onerror = null;
                    }}
                    close() {{
                      this.readyState = 2;
                    }}
                  }}

                  const sleep = (ms) => new Promise((resolve) => window.setTimeout(resolve, ms));
                  const jsonResponse = (body, status = 200) => ({{
                    ok: status < 400,
                    status,
                    async json() {{
                      return body;
                    }},
                    async text() {{
                      return JSON.stringify(body);
                    }},
                  }});

                  window.EventSource = MockEventSource;
                  window.fetch = async (input, options = {{}}) => {{
                    const rawUrl = typeof input === "string" ? input : (input?.url || String(input));
                    const url = new URL(rawUrl, "http://testserver");
                    const method = String(options.method || "GET").toUpperCase();
                    const key = `${{method}} ${{url.pathname}}`;

                    if (key === "GET /threads") {{
                      threadListCalls += 1;
                      if (threadListCalls === 1) {{
                        await sleep(120);
                        return jsonResponse({{ threads: [existingThread] }});
                      }}
                      return jsonResponse({{ threads: [existingThread, newThread] }});
                    }}
                    if (key === "GET /threads/thread-existing") {{
                      return jsonResponse(existingThread);
                    }}
                    if (key === "GET /threads/thread-new") {{
                      return jsonResponse(newThread);
                    }}
                    if (key === "POST /runs") {{
                      return jsonResponse({{
                        thread_id: "thread-new",
                        run_id: "run-new",
                        status: "running",
                      }});
                    }}
                    throw new Error(`No mock response for ${{key}}`);
                  }};
                }})();
                """
            )

            await page.set_content(html, wait_until="load")
            await page.locator("#goal-input").fill("New goal")
            await page.locator("#new-thread-button").click()

            await page.wait_for_function(
                "() => document.querySelector('.thread-item.active .thread-title')?.textContent.includes('New Thread')"
            )
            await page.wait_for_function(
                "() => document.getElementById('selected-thread-panel').textContent.includes('New Thread')"
            )

            await browser.close()
    except PlaywrightError as exc:
        pytest.skip(f"Playwright browser is unavailable: {exc}")


@pytest.mark.asyncio
async def test_live_app_desktop_layer_routes_manual_inputs_to_computer_tool(tmp_path: Path) -> None:
    playwright = pytest.importorskip("playwright.async_api")
    from playwright.async_api import Error as PlaywrightError
    from playwright.async_api import async_playwright

    html = await _root_html(tmp_path)

    running_thread = _computer_activity_thread_snapshot(
        "thread-live",
        "Live Thread",
        "run-live",
        "Desktop interact goal",
        "Masaüstü hazırlaniyor",
    )
    paused_thread = _paused_snapshot(running_thread)

    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            page = await browser.new_page()
            await page.add_init_script(
                script=f"""
                (() => {{
                  const runningThread = {json.dumps(running_thread)};
                  const pausedThread = {json.dumps(paused_thread)};
                  window.__agentraBaseUrl = "http://testserver";
                  window.__fetchCalls = [];

                  class MockEventSource {{
                    constructor(url) {{
                      this.url = url;
                      this.readyState = 1;
                      this.onmessage = null;
                      this.onerror = null;
                    }}
                    close() {{
                      this.readyState = 2;
                    }}
                  }}

                  const jsonResponse = (body, status = 200) => ({{
                    ok: status < 400,
                    status,
                    async json() {{
                      return body;
                    }},
                    async text() {{
                      return JSON.stringify(body);
                    }},
                  }});

                  window.EventSource = MockEventSource;
                  window.fetch = async (input, options = {{}}) => {{
                    const rawUrl = typeof input === "string" ? input : (input?.url || String(input));
                    const url = new URL(rawUrl, "http://testserver");
                    const method = String(options.method || "GET").toUpperCase();
                    const body = options.body ? JSON.parse(options.body) : null;
                    const key = `${{method}} ${{url.pathname}}`;
                    window.__fetchCalls.push({{ key, body }});

                    if (key === "GET /threads") {{
                      return jsonResponse({{ threads: [runningThread] }});
                    }}
                    if (key === "GET /threads/thread-live") {{
                      return jsonResponse(runningThread);
                    }}
                    if (key === "POST /threads/thread-live/pause") {{
                      return jsonResponse(pausedThread);
                    }}
                    if (key === "POST /threads/thread-live/resume") {{
                      return jsonResponse(runningThread);
                    }}
                    if (key === "POST /threads/thread-live/actions") {{
                      return jsonResponse(pausedThread);
                    }}
                    throw new Error(`No mock response for ${{key}}`);
                  }};
                }})();
                """
            )

            await page.set_content(html, wait_until="load")
            await page.wait_for_function(
                "() => document.getElementById('tv-image').src.includes('/threads/thread-live/desktop-stream?stream=')"
            )

            await page.locator("[data-manual-action='interact']").click()
            await page.wait_for_function(
                "() => window.__fetchCalls.some((call) => call.key === 'POST /threads/thread-live/pause')"
            )

            await page.locator("#tv-image").click()
            await page.wait_for_function(
                "() => window.__fetchCalls.some((call) => call.key === 'POST /threads/thread-live/actions' && call.body?.tool === 'computer' && call.body?.args?.action === 'click')"
            )

            box = await page.locator("#tv-image").bounding_box()
            assert box is not None
            await page.mouse.move(box["x"] + 100, box["y"] + 110)
            await page.mouse.down()
            await page.mouse.move(box["x"] + 180, box["y"] + 160, steps=8)
            await page.mouse.up()
            await page.wait_for_function(
                "() => window.__fetchCalls.some((call) => call.key === 'POST /threads/thread-live/actions' && call.body?.tool === 'computer' && call.body?.args?.action === 'drag')"
            )

            await page.locator("#tv-image").hover()
            await page.mouse.wheel(0, 480)
            await page.wait_for_function(
                "() => window.__fetchCalls.some((call) => call.key === 'POST /threads/thread-live/actions' && call.body?.tool === 'computer' && call.body?.args?.action === 'scroll' && call.body?.args?.delta_y === 480)"
            )

            await page.keyboard.type("abc")
            await page.wait_for_function(
                "() => window.__fetchCalls.some((call) => call.key === 'POST /threads/thread-live/actions' && call.body?.tool === 'computer' && call.body?.args?.action === 'type' && call.body?.args?.text === 'abc')"
            )

            await page.keyboard.press("Enter")
            await page.wait_for_function(
                "() => window.__fetchCalls.some((call) => call.key === 'POST /threads/thread-live/actions' && call.body?.tool === 'computer' && call.body?.args?.action === 'key' && call.body?.args?.text === 'Enter')"
            )

            fetch_calls = await page.evaluate("window.__fetchCalls")
            action_bodies = [
                item["body"]
                for item in fetch_calls
                if item["key"] == "POST /threads/thread-live/actions"
            ]
            assert any(body["tool"] == "computer" and body["args"]["action"] == "click" for body in action_bodies)
            assert any(body["tool"] == "computer" and body["args"]["action"] == "drag" for body in action_bodies)
            assert any(body["tool"] == "computer" and body["args"]["action"] == "scroll" for body in action_bodies)
            assert any(body["tool"] == "computer" and body["args"]["action"] == "type" for body in action_bodies)
            assert any(body["tool"] == "computer" and body["args"]["action"] == "key" for body in action_bodies)

            await browser.close()
    except PlaywrightError as exc:
        pytest.skip(f"Playwright browser is unavailable: {exc}")


@pytest.mark.asyncio
async def test_live_app_interact_ignores_duplicate_preview_clicks_while_action_is_pending(
    tmp_path: Path,
) -> None:
    playwright = pytest.importorskip("playwright.async_api")
    from playwright.async_api import Error as PlaywrightError
    from playwright.async_api import async_playwright

    html = await _root_html(tmp_path)

    running_thread = _thread_snapshot(
        "thread-live",
        "Live Thread",
        "run-live",
        "Interact goal",
        "Live snapshot",
    )
    paused_thread = _paused_thread_snapshot(
        "thread-live",
        "Live Thread",
        "run-live",
        "Interact goal",
        "Paused snapshot",
    )

    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            page = await browser.new_page()
            await page.add_init_script(
                script=f"""
                (() => {{
                  const runningThread = {json.dumps(running_thread)};
                  const pausedThread = {json.dumps(paused_thread)};
                  window.__agentraBaseUrl = "http://testserver";
                  window.__fetchCalls = [];

                  class MockEventSource {{
                    constructor(url) {{
                      this.url = url;
                      this.readyState = 1;
                      this.onmessage = null;
                      this.onerror = null;
                    }}
                    close() {{
                      this.readyState = 2;
                    }}
                  }}

                  const jsonResponse = (body, status = 200) => ({{
                    ok: status < 400,
                    status,
                    async json() {{
                      return body;
                    }},
                    async text() {{
                      return JSON.stringify(body);
                    }},
                  }});

                  window.EventSource = MockEventSource;
                  window.fetch = async (input, options = {{}}) => {{
                    const rawUrl = typeof input === "string" ? input : (input?.url || String(input));
                    const url = new URL(rawUrl, "http://testserver");
                    const method = String(options.method || "GET").toUpperCase();
                    const body = options.body ? JSON.parse(options.body) : null;
                    const key = `${{method}} ${{url.pathname}}`;
                    window.__fetchCalls.push({{ key, body }});

                    if (key === "GET /threads") {{
                      return jsonResponse({{ threads: [runningThread] }});
                    }}
                    if (key === "GET /threads/thread-live") {{
                      return jsonResponse(runningThread);
                    }}
                    if (key === "POST /threads/thread-live/pause") {{
                      return jsonResponse(pausedThread);
                    }}
                    if (key === "POST /threads/thread-live/actions") {{
                      await new Promise((resolve) => window.setTimeout(resolve, 180));
                      return jsonResponse(pausedThread);
                    }}
                    throw new Error(`No mock response for ${{key}}`);
                  }};
                }})();
                """
            )

            await page.set_content(html, wait_until="load")
            await page.wait_for_function(
                "() => document.getElementById('scrubber-label').textContent.trim() === '1 / 1'"
            )

            await page.locator("[data-manual-action='interact']").click()
            await page.wait_for_function(
                "() => document.getElementById('tv-screen').classList.contains('manual-interact')"
            )

            const_box = await page.locator("#tv-image").bounding_box()
            assert const_box is not None
            await page.mouse.click(const_box["x"] + 40, const_box["y"] + 40)
            await page.mouse.click(const_box["x"] + 40, const_box["y"] + 40)
            await page.wait_for_timeout(350)

            action_calls = await page.evaluate(
                "() => window.__fetchCalls.filter((call) => call.key === 'POST /threads/thread-live/actions' && call.body?.args?.action === 'click').length"
            )
            assert action_calls == 1

            await browser.close()
    except PlaywrightError as exc:
        pytest.skip(f"Playwright browser is unavailable: {exc}")


@pytest.mark.asyncio
async def test_live_app_manual_desktop_layer_stays_pinned_across_poll_refresh(tmp_path: Path) -> None:
    playwright = pytest.importorskip("playwright.async_api")
    from playwright.async_api import Error as PlaywrightError
    from playwright.async_api import async_playwright

    html = await _root_html(tmp_path)

    live_thread = _computer_activity_thread_snapshot(
        "thread-live",
        "Live Thread",
        "run-live",
        "Pinned layer goal",
        "Masaüstü hazırlaniyor",
    )

    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            page = await browser.new_page()
            await page.add_init_script(
                script=f"""
                (() => {{
                  const liveThread = {json.dumps(live_thread)};
                  window.__agentraBaseUrl = "http://testserver";

                  class MockEventSource {{
                    constructor(url) {{
                      this.url = url;
                      this.readyState = 1;
                      this.onmessage = null;
                      this.onerror = null;
                    }}
                    close() {{
                      this.readyState = 2;
                    }}
                  }}

                  const jsonResponse = (body, status = 200) => ({{
                    ok: status < 400,
                    status,
                    async json() {{
                      return body;
                    }},
                    async text() {{
                      return JSON.stringify(body);
                    }},
                  }});

                  window.EventSource = MockEventSource;
                  window.fetch = async (input, options = {{}}) => {{
                    const rawUrl = typeof input === "string" ? input : (input?.url || String(input));
                    const url = new URL(rawUrl, "http://testserver");
                    const method = String(options.method || "GET").toUpperCase();
                    const key = `${{method}} ${{url.pathname}}`;

                    if (key === "GET /threads") {{
                      return jsonResponse({{ threads: [liveThread] }});
                    }}
                    if (key === "GET /threads/thread-live") {{
                      return jsonResponse(liveThread);
                    }}
                    throw new Error(`No mock response for ${{key}}`);
                  }};
                }})();
                """
            )

            await page.set_content(html, wait_until="load")
            await page.wait_for_function(
                "() => document.getElementById('tv-image').src.includes('/threads/thread-live/desktop-stream?stream=')"
            )

            await page.wait_for_timeout(2300)

            image_src = await page.locator("#tv-image").get_attribute("src")
            title = await page.locator("#tv-frame-title").text_content()
            assert image_src is not None
            assert "/threads/thread-live/desktop-stream?stream=" in image_src
            assert title is not None and "MASAÜSTÜ" in title

            await browser.close()
    except PlaywrightError as exc:
        pytest.skip(f"Playwright browser is unavailable: {exc}")


@pytest.mark.asyncio
async def test_live_app_manual_desktop_layer_stays_pinned_without_active_run(tmp_path: Path) -> None:
    playwright = pytest.importorskip("playwright.async_api")
    from playwright.async_api import Error as PlaywrightError
    from playwright.async_api import async_playwright

    html = await _root_html(tmp_path)
    idle_thread = _idle_thread_snapshot("thread-idle", "Idle Thread")

    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            page = await browser.new_page()
            await page.add_init_script(
                script=f"""
                (() => {{
                  const idleThread = {json.dumps(idle_thread)};
                  window.__agentraBaseUrl = "http://testserver";

                  class MockEventSource {{
                    constructor(url) {{
                      this.url = url;
                      this.readyState = 1;
                      this.onmessage = null;
                      this.onerror = null;
                    }}
                    close() {{
                      this.readyState = 2;
                    }}
                  }}

                  const jsonResponse = (body, status = 200) => ({{
                    ok: status < 400,
                    status,
                    async json() {{
                      return body;
                    }},
                    async text() {{
                      return JSON.stringify(body);
                    }},
                  }});

                  window.EventSource = MockEventSource;
                  window.fetch = async (input, options = {{}}) => {{
                    const rawUrl = typeof input === "string" ? input : (input?.url || String(input));
                    const url = new URL(rawUrl, "http://testserver");
                    const method = String(options.method || "GET").toUpperCase();
                    const key = `${{method}} ${{url.pathname}}`;

                    if (key === "GET /threads") {{
                      return jsonResponse({{ threads: [idleThread] }});
                    }}
                    if (key === "GET /threads/thread-idle") {{
                      return jsonResponse(idleThread);
                    }}
                    throw new Error(`No mock response for ${{key}}`);
                  }};
                }})();
                """
            )

            await page.set_content(html, wait_until="load")
            await page.wait_for_function(
                "() => document.getElementById('selected-thread-panel').textContent.includes('Idle Thread')"
            )

            assert await page.locator("[data-manual-layer='browser']").count() == 0
            assert await page.locator("[data-manual-layer='desktop']").count() == 0
            assert await page.locator("#permission-mode-input").count() == 1

            await browser.close()
    except PlaywrightError as exc:
        pytest.skip(f"Playwright browser is unavailable: {exc}")


@pytest.mark.asyncio
async def test_live_app_browser_live_mirror_uses_frame_url(tmp_path: Path) -> None:
    playwright = pytest.importorskip("playwright.async_api")
    from playwright.async_api import Error as PlaywrightError
    from playwright.async_api import async_playwright

    html = await _root_html(tmp_path)

    live_thread = _thread_snapshot(
        "thread-live",
        "Live Thread",
        "run-live",
        "Fast mirror goal",
        "Live snapshot",
    )
    live_thread["browser_session_active"] = True
    live_thread["active_url"] = "https://example.com"
    live_thread["active_title"] = "Example"
    live_thread["tab_count"] = 1
    live_thread["browser"] = {
        "active": True,
        "active_url": "https://example.com",
        "active_title": "Example",
        "tab_count": 1,
    }

    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            page = await browser.new_page()
            await page.add_init_script(
                script=f"""
                (() => {{
                  const liveThread = {json.dumps(live_thread)};
                  const PNG_1X1_B64 = "{PNG_1X1_B64}";
                  window.__agentraBaseUrl = "http://testserver";
                  window.__fetchCalls = [];

                  class MockEventSource {{
                    constructor(url) {{
                      this.url = url;
                      this.readyState = 1;
                      this.onmessage = null;
                      this.onerror = null;
                    }}
                    close() {{
                      this.readyState = 2;
                    }}
                  }}

                  const jsonResponse = (body, status = 200) => ({{
                    ok: status < 400,
                    status,
                    async json() {{
                      return body;
                    }},
                    async text() {{
                      return JSON.stringify(body);
                    }},
                  }});
                  {PNG_RESPONSE_SCRIPT}

                  window.EventSource = MockEventSource;
                  window.fetch = async (input, options = {{}}) => {{
                    const rawUrl = typeof input === "string" ? input : (input?.url || String(input));
                    const url = new URL(rawUrl, "http://testserver");
                    const method = String(options.method || "GET").toUpperCase();
                    const key = `${{method}} ${{url.pathname}}`;
                    window.__fetchCalls.push({{ key }});

                    if (key === "GET /threads") {{
                      return jsonResponse({{ threads: [liveThread] }});
                    }}
                    if (key === "GET /threads/thread-live") {{
                      return jsonResponse(liveThread);
                    }}
                    if (key === "GET /threads/thread-live/live-frame") {{
                      return binaryFrameResponse();
                    }}
                    throw new Error(`No mock response for ${{key}}`);
                  }};
                }})();
                """
            )

            await page.set_content(html, wait_until="load")
            await page.wait_for_function(
                "() => document.getElementById('tv-frame-title').textContent.includes('Ajan')"
            )
            await page.wait_for_function(
                "() => window.__fetchCalls.some((call) => call.key === 'GET /threads/thread-live/live-frame')"
            )

            image_src = await page.locator("#tv-image").get_attribute("src")
            assert image_src is not None
            assert image_src.startswith("blob:")

            fetch_calls = await page.evaluate("window.__fetchCalls")
            assert any(call["key"] == "GET /threads/thread-live/live-frame" for call in fetch_calls)

            await browser.close()
    except PlaywrightError as exc:
        pytest.skip(f"Playwright browser is unavailable: {exc}")


@pytest.mark.asyncio
async def test_live_app_auto_switches_to_desktop_stream_for_computer_activity(tmp_path: Path) -> None:
    playwright = pytest.importorskip("playwright.async_api")
    from playwright.async_api import Error as PlaywrightError
    from playwright.async_api import async_playwright

    html = await _root_html(tmp_path)

    live_thread = _computer_activity_thread_snapshot(
        "thread-live",
        "Live Thread",
        "run-live",
        "Desktop key goal",
        "win+d tuşuna basılıyor",
    )

    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            page = await browser.new_page()
            await page.add_init_script(
                script=f"""
                (() => {{
                  const liveThread = {json.dumps(live_thread)};
                  window.__agentraBaseUrl = "http://testserver";

                  class MockEventSource {{
                    constructor(url) {{
                      this.url = url;
                      this.readyState = 1;
                      this.onmessage = null;
                      this.onerror = null;
                    }}
                    close() {{
                      this.readyState = 2;
                    }}
                  }}

                  const jsonResponse = (body, status = 200) => ({{
                    ok: status < 400,
                    status,
                    async json() {{
                      return body;
                    }},
                    async text() {{
                      return JSON.stringify(body);
                    }},
                  }});

                  window.EventSource = MockEventSource;
                  window.fetch = async (input, options = {{}}) => {{
                    const rawUrl = typeof input === "string" ? input : (input?.url || String(input));
                    const url = new URL(rawUrl, "http://testserver");
                    const method = String(options.method || "GET").toUpperCase();
                    const key = `${{method}} ${{url.pathname}}`;

                    if (key === "GET /threads") {{
                      return jsonResponse({{ threads: [liveThread] }});
                    }}
                    if (key === "GET /threads/thread-live") {{
                      return jsonResponse(liveThread);
                    }}
                    throw new Error(`No mock response for ${{key}}`);
                  }};
                }})();
                """
            )

            await page.set_content(html, wait_until="load")
            await page.wait_for_function(
                "() => document.getElementById('tv-image').src.includes('/threads/thread-live/desktop-stream?stream=')"
            )
            await page.wait_for_function(
                "() => document.getElementById('tv-frame-title').textContent.includes('MASAÜSTÜ')"
            )

            image_src = await page.locator("#tv-image").get_attribute("src")
            title = await page.locator("#tv-frame-title").text_content()
            assert image_src is not None
            assert "/threads/thread-live/desktop-stream?stream=" in image_src
            assert title is not None and "MASAÜSTÜ" in title

            await browser.close()
    except PlaywrightError as exc:
        pytest.skip(f"Playwright browser is unavailable: {exc}")


@pytest.mark.asyncio
async def test_live_app_keeps_browser_layer_for_hidden_local_system_activity(tmp_path: Path) -> None:
    playwright = pytest.importorskip("playwright.async_api")
    from playwright.async_api import Error as PlaywrightError
    from playwright.async_api import async_playwright

    html = await _root_html(tmp_path)

    live_thread = _local_system_activity_thread_snapshot(
        "thread-live",
        "Live Thread",
        "run-live",
        "Open the PowerPoint under the hood",
        "Dosya arka planda açılıyor",
    )

    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            page = await browser.new_page()
            await page.add_init_script(
                script=f"""
                (() => {{
                  const liveThread = {json.dumps(live_thread)};
                  window.__agentraBaseUrl = "http://testserver";

                  class MockEventSource {{
                    constructor(url) {{
                      this.url = url;
                      this.readyState = 1;
                      this.onmessage = null;
                      this.onerror = null;
                    }}
                    close() {{
                      this.readyState = 2;
                    }}
                  }}

                  const jsonResponse = (body, status = 200) => ({{
                    ok: status < 400,
                    status,
                    async json() {{
                      return body;
                    }},
                    async text() {{
                      return JSON.stringify(body);
                    }},
                  }});

                  window.EventSource = MockEventSource;
                  window.fetch = async (input, options = {{}}) => {{
                    const rawUrl = typeof input === "string" ? input : (input?.url || String(input));
                    const url = new URL(rawUrl, "http://testserver");
                    const method = String(options.method || "GET").toUpperCase();
                    const key = `${{method}} ${{url.pathname}}`;

                    if (key === "GET /threads") {{
                      return jsonResponse({{ threads: [liveThread] }});
                    }}
                    if (key === "GET /threads/thread-live") {{
                      return jsonResponse(liveThread);
                    }}
                    throw new Error(`No mock response for ${{key}}`);
                  }};
                }})();
                """
            )

            await page.set_content(html, wait_until="load")
            await page.wait_for_function(
                "() => document.getElementById('tv-frame-title').textContent.includes('Ajan') && document.getElementById('tv-frame-title').textContent.includes('YEREL SİSTEM')"
            )

            title = await page.locator("#tv-frame-title").text_content()
            selected_panel = await page.locator("#selected-thread-panel").text_content()
            assert title is not None and "YEREL SİSTEM" in title
            assert "MASAÜSTÜ" not in title
            assert selected_panel is not None and "Dosya arka planda açılıyor" in selected_panel

            await browser.close()
    except PlaywrightError as exc:
        pytest.skip(f"Playwright browser is unavailable: {exc}")


@pytest.mark.asyncio
async def test_live_app_prefers_desktop_layer_from_run_hint_before_computer_events(tmp_path: Path) -> None:
    playwright = pytest.importorskip("playwright.async_api")
    from playwright.async_api import Error as PlaywrightError
    from playwright.async_api import async_playwright

    html = await _root_html(tmp_path)

    live_thread = _thread_snapshot(
        "thread-live",
        "Live Thread",
        "run-live",
        "Masaustune git ve secondsun klasorune gir",
        "Masaustu hazirlaniyor",
    )
    live_thread["active_run"]["events"] = []
    live_thread["active_run"]["frames"] = []
    live_thread["active_run"]["control_surface_hint"] = "desktop"

    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            page = await browser.new_page()
            await page.add_init_script(
                script=f"""
                (() => {{
                  const liveThread = {json.dumps(live_thread)};
                  window.__agentraBaseUrl = "http://testserver";

                  class MockEventSource {{
                    constructor(url) {{
                      this.url = url;
                      this.readyState = 1;
                      this.onmessage = null;
                      this.onerror = null;
                    }}
                    close() {{
                      this.readyState = 2;
                    }}
                  }}

                  const jsonResponse = (body, status = 200) => ({{
                    ok: status < 400,
                    status,
                    async json() {{
                      return body;
                    }},
                    async text() {{
                      return JSON.stringify(body);
                    }},
                  }});

                  window.EventSource = MockEventSource;
                  window.fetch = async (input, options = {{}}) => {{
                    const rawUrl = typeof input === "string" ? input : (input?.url || String(input));
                    const url = new URL(rawUrl, "http://testserver");
                    const method = String(options.method || "GET").toUpperCase();
                    const key = `${{method}} ${{url.pathname}}`;

                    if (key === "GET /threads") {{
                      return jsonResponse({{ threads: [liveThread] }});
                    }}
                    if (key === "GET /threads/thread-live") {{
                      return jsonResponse(liveThread);
                    }}
                    throw new Error(`No mock response for ${{key}}`);
                  }};
                }})();
                """
            )

            await page.set_content(html, wait_until="load")
            await page.wait_for_function(
                "() => document.getElementById('tv-frame-title').textContent.includes('MASAUSTU') || document.getElementById('tv-frame-title').textContent.includes('MASAÜSTÜ')"
            )

            title = await page.locator("#tv-frame-title").text_content()
            assert title is not None
            assert "TARAYICI" not in title

            await browser.close()
    except PlaywrightError as exc:
        pytest.skip(f"Playwright browser is unavailable: {exc}")


@pytest.mark.asyncio
async def test_live_app_live_mirror_starts_for_browser_flow_before_browser_status_arrives(tmp_path: Path) -> None:
    playwright = pytest.importorskip("playwright.async_api")
    from playwright.async_api import Error as PlaywrightError
    from playwright.async_api import async_playwright

    html = await _root_html(tmp_path)

    live_thread = _thread_snapshot(
        "thread-live",
        "Live Thread",
        "run-live",
        "GitHub login goal",
        "Tarayici aciliyor",
    )
    live_thread["browser_session_active"] = False
    live_thread["active_url"] = ""
    live_thread["active_title"] = ""
    live_thread["tab_count"] = 0
    live_thread["browser"] = {
        "active": False,
        "active_url": "",
        "active_title": "",
        "tab_count": 0,
    }
    live_thread["activity"]["channel"] = "browser"
    live_thread["active_run"]["activity"]["channel"] = "browser"

    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            page = await browser.new_page()
            await page.add_init_script(
                script=f"""
                (() => {{
                  const liveThread = {json.dumps(live_thread)};
                  const PNG_1X1_B64 = "{PNG_1X1_B64}";
                  window.__agentraBaseUrl = "http://testserver";
                  window.__fetchCalls = [];

                  class MockEventSource {{
                    constructor(url) {{
                      this.url = url;
                      this.readyState = 1;
                      this.onmessage = null;
                      this.onerror = null;
                    }}
                    close() {{
                      this.readyState = 2;
                    }}
                  }}

                  const jsonResponse = (body, status = 200) => ({{
                    ok: status < 400,
                    status,
                    async json() {{
                      return body;
                    }},
                    async text() {{
                      return JSON.stringify(body);
                    }},
                  }});
                  {PNG_RESPONSE_SCRIPT}

                  window.EventSource = MockEventSource;
                  window.fetch = async (input, options = {{}}) => {{
                    const rawUrl = typeof input === "string" ? input : (input?.url || String(input));
                    const url = new URL(rawUrl, "http://testserver");
                    const method = String(options.method || "GET").toUpperCase();
                    const key = `${{method}} ${{url.pathname}}`;
                    window.__fetchCalls.push({{ key }});

                    if (key === "GET /threads") {{
                      return jsonResponse({{ threads: [liveThread] }});
                    }}
                    if (key === "GET /threads/thread-live") {{
                      return jsonResponse(liveThread);
                    }}
                    if (key === "GET /threads/thread-live/live-frame") {{
                      return binaryFrameResponse();
                    }}
                    throw new Error(`No mock response for ${{key}}`);
                  }};
                }})();
                """
            )

            await page.set_content(html, wait_until="load")
            await page.wait_for_function(
                "() => window.__fetchCalls.some((call) => call.key === 'GET /threads/thread-live/live-frame')"
            )
            await page.wait_for_function(
                "() => document.getElementById('tv-image').src.startsWith('blob:')"
            )
            assert await page.locator("#tv-empty").is_hidden()

            await browser.close()
    except PlaywrightError as exc:
        pytest.skip(f"Playwright browser is unavailable: {exc}")


@pytest.mark.asyncio
async def test_live_app_browser_live_mirror_keeps_last_frame_when_poll_returns_empty(tmp_path: Path) -> None:
    playwright = pytest.importorskip("playwright.async_api")
    from playwright.async_api import Error as PlaywrightError
    from playwright.async_api import async_playwright

    html = await _root_html(tmp_path)

    live_thread = _thread_snapshot(
        "thread-live",
        "Live Thread",
        "run-live",
        "GitHub login goal",
        "Tarayici aciliyor",
    )
    live_thread["browser_session_active"] = True
    live_thread["active_url"] = "https://github.com/login"
    live_thread["active_title"] = "Sign in to GitHub"
    live_thread["tab_count"] = 1
    live_thread["browser"] = {
        "active": True,
        "active_url": "https://github.com/login",
        "active_title": "Sign in to GitHub",
        "tab_count": 1,
    }

    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            page = await browser.new_page()
            await page.add_init_script(
                script=f"""
                (() => {{
                  const liveThread = {json.dumps(live_thread)};
                  const PNG_1X1_B64 = "{PNG_1X1_B64}";
                  window.__agentraBaseUrl = "http://testserver";
                  window.__liveFrameHits = 0;

                  class MockEventSource {{
                    constructor(url) {{
                      this.url = url;
                      this.readyState = 1;
                      this.onmessage = null;
                      this.onerror = null;
                    }}
                    close() {{
                      this.readyState = 2;
                    }}
                  }}

                  const jsonResponse = (body, status = 200) => ({{
                    ok: status < 400,
                    status,
                    async json() {{
                      return body;
                    }},
                    async text() {{
                      return JSON.stringify(body);
                    }},
                  }});
                  {PNG_RESPONSE_SCRIPT}

                  window.EventSource = MockEventSource;
                  window.fetch = async (input, options = {{}}) => {{
                    const rawUrl = typeof input === "string" ? input : (input?.url || String(input));
                    const url = new URL(rawUrl, "http://testserver");
                    const method = String(options.method || "GET").toUpperCase();
                    const key = `${{method}} ${{url.pathname}}`;

                    if (key === "GET /threads") {{
                      return jsonResponse({{ threads: [liveThread] }});
                    }}
                    if (key === "GET /threads/thread-live") {{
                      return jsonResponse(liveThread);
                    }}
                    if (key === "GET /threads/thread-live/live-frame") {{
                      window.__liveFrameHits += 1;
                      if (window.__liveFrameHits === 1) {{
                        return binaryFrameResponse();
                      }}
                      if (window.__liveFrameHits === 2) {{
                        return binaryFrameResponse();
                      }}
                      return binaryFrameResponse(204);
                    }}
                    throw new Error(`No mock response for ${{key}}`);
                  }};
                }})();
                """
            )

            await page.set_content(html, wait_until="load")
            await page.wait_for_function(
                "() => document.getElementById('tv-image').src.startsWith('blob:')"
            )
            initial_src = await page.locator("#tv-image").get_attribute("src")
            assert initial_src is not None and initial_src.startswith("blob:")
            await page.wait_for_function("() => window.__liveFrameHits >= 3")
            await page.wait_for_timeout(180)
            image_src = await page.locator("#tv-image").get_attribute("src")
            assert image_src == initial_src
            assert await page.locator("#tv-empty").is_hidden()

            await browser.close()
    except PlaywrightError as exc:
        pytest.skip(f"Playwright browser is unavailable: {exc}")


@pytest.mark.asyncio
async def test_live_app_desktop_live_mirror_falls_back_to_frame_endpoint_after_stream_error(tmp_path: Path) -> None:
    playwright = pytest.importorskip("playwright.async_api")
    from playwright.async_api import Error as PlaywrightError
    from playwright.async_api import async_playwright

    html = await _root_html(tmp_path)

    live_thread = _thread_snapshot(
        "thread-live",
        "Live Thread",
        "run-live",
        "Fast mirror goal",
        "Live snapshot",
    )
    live_thread["browser_session_active"] = True
    live_thread["active_url"] = "https://example.com"
    live_thread["active_title"] = "Example"
    live_thread["tab_count"] = 1
    live_thread["browser"] = {
        "active": True,
        "active_url": "https://example.com",
        "active_title": "Example",
        "tab_count": 1,
    }
    live_thread["active_run"]["events"] = [
        {
            "type": "tool_call",
            "tool": "computer",
            "label": "Desktop move",
            "summary": "Desktop move",
        }
    ]
    live_thread["activity"]["channel"] = "desktop"
    live_thread["active_run"]["activity"]["channel"] = "desktop"
    live_thread["active_run"]["control_surface_hint"] = "desktop"

    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            page = await browser.new_page()
            await page.add_init_script(
                script=f"""
                (() => {{
                  const liveThread = {json.dumps(live_thread)};
                  window.__agentraBaseUrl = "http://testserver";
                  window.__fetchCalls = [];

                  class MockEventSource {{
                    constructor(url) {{
                      this.url = url;
                      this.readyState = 1;
                      this.onmessage = null;
                      this.onerror = null;
                    }}
                    close() {{
                      this.readyState = 2;
                    }}
                  }}

                  const jsonResponse = (body, status = 200) => ({{
                    ok: status < 400,
                    status,
                    async json() {{
                      return body;
                    }},
                    async text() {{
                      return JSON.stringify(body);
                    }},
                  }});

                  window.EventSource = MockEventSource;
                  window.fetch = async (input, options = {{}}) => {{
                    const rawUrl = typeof input === "string" ? input : (input?.url || String(input));
                    const url = new URL(rawUrl, "http://testserver");
                    const method = String(options.method || "GET").toUpperCase();
                    const key = `${{method}} ${{url.pathname}}`;
                    window.__fetchCalls.push({{ key }});

                    if (key === "GET /threads") {{
                      return jsonResponse({{ threads: [liveThread] }});
                    }}
                    if (key === "GET /threads/thread-live") {{
                      return jsonResponse(liveThread);
                    }}
                    throw new Error(`No mock response for ${{key}}`);
                  }};
                }})();
                """
            )

            await page.set_content(html, wait_until="load")
            await page.wait_for_function(
                "() => document.getElementById('tv-image').src.includes('/threads/thread-live/desktop-stream?stream=')"
            )
            await page.locator("#tv-image").dispatch_event("load")
            await page.locator("#tv-image").dispatch_event("error")
            live_frame_url = await page.evaluate("state.liveFrameUrl")
            assert live_frame_url is not None and "/threads/thread-live/desktop-frame?ts=" in live_frame_url
            await page.wait_for_function(
                "() => document.getElementById('tv-image').src.includes('/threads/thread-live/desktop-frame?ts=')"
            )

            image_src = await page.locator("#tv-image").get_attribute("src")
            assert image_src is not None
            assert "/threads/thread-live/desktop-frame?ts=" in image_src
            assert await page.locator("#tv-empty").is_hidden()

            await browser.close()
    except PlaywrightError as exc:
        pytest.skip(f"Playwright browser is unavailable: {exc}")
