"""Tests for video/GIF recording (A5)."""

from __future__ import annotations

import asyncio
import io
from pathlib import Path

from PIL import Image

from agentra.tools.computer import ComputerTool


def _make_png_bytes(color: tuple[int, int, int] = (100, 150, 200), size: tuple[int, int] = (32, 32)) -> bytes:
    img = Image.new("RGB", size, color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_tool(tmp_path: Path) -> ComputerTool:
    t = ComputerTool()
    t._workspace_dir = tmp_path
    t.capture_screenshot_bytes = lambda fmt=None, quality=None, region=None, monitor=0: _make_png_bytes()  # type: ignore[assignment]
    return t


class TestComputerRecording:
    def test_start_recording(self, tmp_path: Path) -> None:
        tool = _make_tool(tmp_path)

        async def _run() -> None:
            result = await tool._start_recording()
            assert result.success is True
            assert tool._recording is True
            tool._recording = False
            if tool._recording_task:
                tool._recording_task.cancel()
                try:
                    await tool._recording_task
                except asyncio.CancelledError:
                    pass

        asyncio.run(_run())

    def test_start_recording_twice_fails(self, tmp_path: Path) -> None:
        tool = _make_tool(tmp_path)

        async def _run() -> None:
            await tool._start_recording()
            result = await tool._start_recording()
            assert result.success is False
            assert "already" in result.error.lower()
            tool._recording = False
            if tool._recording_task:
                tool._recording_task.cancel()
                try:
                    await tool._recording_task
                except asyncio.CancelledError:
                    pass

        asyncio.run(_run())

    def test_stop_without_start_fails(self, tmp_path: Path) -> None:
        tool = _make_tool(tmp_path)

        async def _run() -> None:
            result = await tool._stop_recording()
            assert result.success is False

        asyncio.run(_run())

    def test_start_stop_produces_gif(self, tmp_path: Path) -> None:
        tool = _make_tool(tmp_path)

        async def _run() -> None:
            await tool._start_recording()
            await asyncio.sleep(0.6)
            result = await tool._stop_recording()
            assert result.success is True
            assert "recording_path" in result.metadata
            gif_path = Path(result.metadata["recording_path"])
            assert gif_path.exists()
            assert gif_path.suffix == ".gif"
            assert gif_path.read_bytes()[:6] in (b"GIF87a", b"GIF89a")

        asyncio.run(_run())

    def test_gif_frame_count(self, tmp_path: Path) -> None:
        tool = _make_tool(tmp_path)

        async def _run() -> None:
            await tool._start_recording()
            await asyncio.sleep(0.6)
            result = await tool._stop_recording()
            assert result.metadata["frame_count"] >= 1

        asyncio.run(_run())


class TestSchemaRecording:
    def test_schema_has_recording_actions(self) -> None:
        tool = ComputerTool()
        actions = tool.schema["properties"]["action"]["enum"]
        assert "start_recording" in actions
        assert "stop_recording" in actions

    def test_browser_schema_has_recording_actions(self) -> None:
        from agentra.tools.browser import BrowserTool
        tool = BrowserTool()
        actions = tool.schema["properties"]["action"]["enum"]
        assert "start_recording" in actions
        assert "stop_recording" in actions
