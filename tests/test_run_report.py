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
    report.record({"type": "tool_call", "tool": "browser", "args": {"action": "navigate"}})
    report.record({"type": "screenshot", "data": PNG_1X1_B64})
    report.finalize("completed")

    html_text = report.html_path.read_text(encoding="utf-8")
    payload = json.loads(report.events_path.read_text(encoding="utf-8"))

    assert report.html_path.exists()
    assert "Demo browser task" in html_text
    assert "gemini-3-flash-preview" in html_text
    assert "Screenshot" in html_text
    assert payload["status"] == "completed"
    assert any(event["type"] == "screenshot" for event in payload["events"])

    screenshot_files = list((report.run_dir / "assets").glob("*.png"))
    assert len(screenshot_files) == 1


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


def test_run_report_no_double_escaping(tmp_path):
    """Tool names and sub-task labels with special chars must not be double-escaped."""
    report = RunReport(
        workspace_dir=tmp_path,
        goal="Escaping test",
        provider="openai",
        model="gpt-4o",
    )
    report.record({"type": "tool_result", "tool": "a&b", "result": "ok", "success": True})
    report.record({"type": "sub_task", "label": "step <1>", "result": "done", "success": True})
    report.finalize("completed")

    html_text = report.html_path.read_text(encoding="utf-8")

    # Should appear escaped once, not double-escaped (&amp;amp; or &amp;lt;)
    assert "a&amp;b" in html_text
    assert "&amp;amp;" not in html_text
    assert "step &lt;1&gt;" in html_text
    assert "&amp;lt;" not in html_text
