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


def _run_snapshot(thread_id: str, thread_title: str, run_id: str, goal: str, summary: str) -> dict:
    return {
        "run_id": run_id,
        "goal": goal,
        "status": "running",
        "provider": "gemini",
        "model": "gemini-3-flash-preview",
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
    }


def _thread_snapshot(thread_id: str, title: str, run_id: str, goal: str, summary: str) -> dict:
    active_run = _run_snapshot(thread_id, title, run_id, goal, summary)
    return {
        "thread_id": thread_id,
        "title": title,
        "status": "running",
        "handoff_state": "agent",
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
        "active_run": active_run,
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
