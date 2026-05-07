"""Tests for the HTML run report."""

from __future__ import annotations

import json

from agentra.run_report import RunReport


PNG_1X1_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)


def test_run_report_writes_html_and_assets(tmp_path):
    report = RunReport(
        workspace_dir=tmp_path,
        goal="Demo browser task",
        provider="gemini",
        model="gemini-3-flash-preview",
    )

    report.record({"type": "thought", "content": "Opening a page."})
    report.record(
        {
            "type": "tool_call",
            "tool": "browser",
            "args": {"action": "navigate", "url": "https://www.python.org"},
        }
    )
    report.record({"type": "screenshot", "data": PNG_1X1_B64})
    report.finalize("completed")

    html_text = report.html_path.read_text(encoding="utf-8")
    payload = json.loads(report.events_path.read_text(encoding="utf-8"))

    assert report.html_path.exists()
    assert "Demo browser task" in html_text
    assert "gemini-3-flash-preview" in html_text
    assert "Screenshot" in html_text
    assert 'class="hero-shot"' in html_text
    assert payload["status"] == "completed"
    assert any(event["type"] == "screenshot" for event in payload["events"])
    assert payload["frames"][0]["label"] == "browser · navigate"
    assert payload["frames"][0]["summary"] == "Opening python.org"

    screenshot_files = list((report.run_dir / "assets").glob("*.png"))
    debug_files = list((report.run_dir / "debug-images" / "run-screenshots").glob("*.png"))
    assert len(screenshot_files) == 1
    assert len(debug_files) == 1


def test_run_report_keeps_event_history(tmp_path):
    report = RunReport(
        workspace_dir=tmp_path,
        goal="Check terminal output",
        provider="ollama",
        model="qwen3.5:latest",
    )

    report.record({"type": "tool_result", "tool": "terminal", "result": "hello", "success": True})
    report.finalize("partial")

    assert report.events[0]["tool"] == "terminal"
    assert report.events[0]["success"] is True


def test_run_report_renders_error_state_with_wrapped_path(tmp_path):
    deep_workspace = tmp_path / "very" / "deep" / "workspace" / "path"
    report = RunReport(
        workspace_dir=deep_workspace,
        goal="Check browser failure",
        provider="gemini",
        model="gemini-3-flash-preview",
    )

    report.record(
        {
            "type": "error",
            "content": "Function call is missing a thought_signature in functionCall parts.",
        }
    )
    report.finalize("error")

    html_text = report.html_path.read_text(encoding="utf-8")

    assert 'class="status error"' in html_text
    assert "thought_signature" in html_text
    assert "overflow-wrap: anywhere" in html_text


def test_run_report_renders_audit_trail(tmp_path):
    report = RunReport(
        workspace_dir=tmp_path,
        goal="Audit demo",
        provider="gemini",
        model="gemini-3-flash-preview",
    )

    report.record_audit(
        {
            "entry_type": "workspace_checkpoint",
            "timestamp": "2026-03-16T12:00:00",
            "details": {"before_sha": "abc12345", "after_sha": "def67890"},
        }
    )
    report.finalize("completed")

    html_text = report.html_path.read_text(encoding="utf-8")
    payload = json.loads(report.events_path.read_text(encoding="utf-8"))

    assert "Audit Trail" in html_text
    assert "workspace_checkpoint" in payload["audit"][0]["entry_type"]
