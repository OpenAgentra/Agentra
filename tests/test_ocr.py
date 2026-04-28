"""Tests for C1: OCR text extraction."""

from __future__ import annotations

import importlib.util

import pytest

from agentra.tools.computer import ComputerTool


class TestExtractTextSchema:
    def test_extract_text_in_action_enum(self) -> None:
        actions = ComputerTool().schema["properties"]["action"]["enum"]
        assert "extract_text" in actions

    def test_lang_property_present(self) -> None:
        props = ComputerTool().schema["properties"]
        assert "lang" in props


class TestExtractTextNoBackend:
    """When no OCR backend is available, return helpful error."""

    def test_no_backend_returns_helpful_error(self, monkeypatch) -> None:
        # Force both imports to fail by removing the modules from the import system
        import sys
        monkeypatch.setitem(sys.modules, "easyocr", None)
        monkeypatch.setitem(sys.modules, "pytesseract", None)

        tool = ComputerTool()
        # Stub capture so we don't need a real display
        tool.capture_screenshot_bytes = lambda **kwargs: b"\x89PNG\r\n\x1a\n"  # type: ignore[assignment]
        result = tool._extract_text()
        assert result.success is False
        assert "No OCR backend" in result.error or "easyocr" in result.error or "pytesseract" in result.error


class TestExtractTextLangNormalization:
    def test_eng_maps_to_en(self) -> None:
        assert ComputerTool._normalize_easyocr_lang("eng") == "en"

    def test_tur_maps_to_tr(self) -> None:
        assert ComputerTool._normalize_easyocr_lang("tur") == "tr"

    def test_unknown_passes_through(self) -> None:
        assert ComputerTool._normalize_easyocr_lang("xyz") == "xyz"

    def test_case_insensitive(self) -> None:
        assert ComputerTool._normalize_easyocr_lang("ENG") == "en"


@pytest.mark.skipif(
    importlib.util.find_spec("easyocr") is None,
    reason="easyocr not installed (pytesseract requires external Tesseract binary, harder to test)",
)
class TestExtractTextWithBackend:
    """Real OCR test, only runs when easyocr is present."""

    def test_extract_text_returns_string(self) -> None:
        import io
        from PIL import Image, ImageDraw

        img = Image.new("RGB", (200, 60), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw.text((10, 10), "HELLO", fill=(0, 0, 0))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        png_bytes = buf.getvalue()

        tool = ComputerTool()
        tool.capture_screenshot_bytes = lambda **kwargs: png_bytes  # type: ignore[assignment]
        result = tool._extract_text()
        assert result.success is True
        assert isinstance(result.output, str)
