"""Tests for screenshot format and quality options (A2)."""

from __future__ import annotations

import io

from PIL import Image

from agentra.tools.computer import ComputerTool


def _stub_mss_grab(tool: ComputerTool, color: tuple[int, int, int] = (100, 150, 200)) -> None:
    """Monkey-patch capture_screenshot_bytes so it doesn't need a real display."""
    img = Image.new("RGB", (64, 64), color)

    def _fake_capture(fmt: str | None = None, quality: int | None = None) -> bytes:
        fmt = (fmt or tool._screenshot_format).lower()
        quality = quality if quality is not None else tool._screenshot_quality
        buf = io.BytesIO()
        if fmt == "png":
            img.save(buf, format="PNG")
        elif fmt == "jpeg":
            img.save(buf, format="JPEG", quality=quality)
        elif fmt == "webp":
            img.save(buf, format="WEBP", quality=quality)
        return buf.getvalue()

    tool.capture_screenshot_bytes = _fake_capture  # type: ignore[assignment]


class TestScreenshotFormat:
    def test_png_format_default(self) -> None:
        tool = ComputerTool(screenshot_format="png")
        _stub_mss_grab(tool)
        data = tool.capture_screenshot_bytes()
        assert data[:8] == b"\x89PNG\r\n\x1a\n"

    def test_jpeg_format(self) -> None:
        tool = ComputerTool(screenshot_format="jpeg")
        _stub_mss_grab(tool)
        data = tool.capture_screenshot_bytes()
        assert data[:2] == b"\xff\xd8"  # JPEG magic bytes

    def test_webp_format(self) -> None:
        tool = ComputerTool(screenshot_format="webp")
        _stub_mss_grab(tool)
        data = tool.capture_screenshot_bytes()
        assert data[:4] == b"RIFF"  # WebP magic bytes

    def test_quality_affects_jpeg_size(self) -> None:
        tool_low = ComputerTool(screenshot_format="jpeg", screenshot_quality=10)
        tool_high = ComputerTool(screenshot_format="jpeg", screenshot_quality=95)
        _stub_mss_grab(tool_low)
        _stub_mss_grab(tool_high)
        low_data = tool_low.capture_screenshot_bytes()
        high_data = tool_high.capture_screenshot_bytes()
        assert len(low_data) <= len(high_data)

    def test_capture_png_bytes_backward_compat(self) -> None:
        tool = ComputerTool(screenshot_format="jpeg")
        _stub_mss_grab(tool)
        # capture_png_bytes must always return PNG regardless of configured format
        data = tool.capture_png_bytes()
        assert data[:8] == b"\x89PNG\r\n\x1a\n"


class TestConfigIntegration:
    def test_config_fields_exist(self) -> None:
        from agentra.config import AgentConfig
        config = AgentConfig(llm_provider="ollama")
        assert hasattr(config, "screenshot_format")
        assert hasattr(config, "screenshot_quality")
        assert config.screenshot_format == "png"
        assert config.screenshot_quality == 85
