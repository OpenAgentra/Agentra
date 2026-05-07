"""Tests for C2: visual element detection (find_template)."""

from __future__ import annotations

import importlib.util
import io
from pathlib import Path

import pytest
from PIL import Image

from agentra.tools.computer import ComputerTool


class TestFindTemplateSchema:
    def test_find_template_in_action_enum(self) -> None:
        actions = ComputerTool().schema["properties"]["action"]["enum"]
        assert "find_template" in actions

    def test_template_path_property_present(self) -> None:
        props = ComputerTool().schema["properties"]
        assert "template_path" in props

    def test_confidence_property_present(self) -> None:
        props = ComputerTool().schema["properties"]
        assert "confidence" in props


class TestFindTemplateValidation:
    def test_missing_template_path_fails(self) -> None:
        tool = ComputerTool()
        result = tool._find_template()
        assert result.success is False
        assert "template_path" in result.error

    def test_nonexistent_path_fails(self, tmp_path: Path) -> None:
        tool = ComputerTool()
        result = tool._find_template(template_path=str(tmp_path / "missing.png"))
        assert result.success is False
        assert "not found" in result.error.lower()


class TestFindTemplateNoOpenCV:
    """Without cv2, should return helpful error."""

    def test_no_opencv_returns_helpful_error(self, monkeypatch, tmp_path: Path) -> None:
        # Create a real template file so we get past the existence check
        template = tmp_path / "test.png"
        Image.new("RGB", (10, 10), (255, 0, 0)).save(template, format="PNG")

        import sys
        monkeypatch.setitem(sys.modules, "cv2", None)

        tool = ComputerTool()
        tool.capture_screenshot_bytes = lambda **kwargs: b""  # type: ignore[assignment]
        result = tool._find_template(template_path=str(template))
        assert result.success is False
        assert "OpenCV" in result.error or "cv2" in result.error.lower()


@pytest.mark.skipif(
    importlib.util.find_spec("cv2") is None,
    reason="OpenCV not installed",
)
class TestFindTemplateWithOpenCV:
    """Real template matching, runs when cv2 is present."""

    def test_finds_template_in_screenshot(self, tmp_path: Path) -> None:
        # Build a haystack image with a distinctive gradient patch
        haystack = Image.new("RGB", (200, 200), (255, 255, 255))
        patch = Image.new("RGB", (40, 40))
        for x in range(40):
            for y in range(40):
                patch.putpixel((x, y), (x * 6, y * 6, (x + y) * 3))
        haystack.paste(patch, (50, 60))
        haystack_buf = io.BytesIO()
        haystack.save(haystack_buf, format="PNG")

        template_path = tmp_path / "patch.png"
        patch.save(template_path, format="PNG")

        tool = ComputerTool()
        tool.capture_screenshot_bytes = lambda **kwargs: haystack_buf.getvalue()  # type: ignore[assignment]
        result = tool._find_template(template_path=str(template_path), confidence=0.95)
        assert result.success is True
        assert "matches" in result.metadata
        assert len(result.metadata["matches"]) >= 1
        # Best match should be near (50, 60)
        positions = [(m["x"], m["y"]) for m in result.metadata["matches"]]
        # Allow either exact match or any nearby — check at least one is near origin
        assert any(abs(x - 50) <= 2 and abs(y - 60) <= 2 for x, y in positions), positions
