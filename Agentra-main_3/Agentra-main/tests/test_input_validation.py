"""Tests for input validation and recovery (C4)."""

from __future__ import annotations

import sys

import pytest

from agentra.tools.computer import ComputerTool


class TestValidateKeySequence:
    """ComputerTool._validate_key_sequence — static helper."""

    def test_empty_string_fails(self) -> None:
        valid, error = ComputerTool._validate_key_sequence("")
        assert valid is False
        assert error and "empty" in error.lower()

    def test_whitespace_only_fails(self) -> None:
        valid, _ = ComputerTool._validate_key_sequence("   ")
        assert valid is False

    def test_valid_combo(self) -> None:
        valid, error = ComputerTool._validate_key_sequence("CTRL+S")
        assert valid is True
        assert error is None

    def test_lowercase_combo_valid(self) -> None:
        valid, _ = ComputerTool._validate_key_sequence("ctrl+a")
        assert valid is True

    def test_single_letter_valid(self) -> None:
        valid, _ = ComputerTool._validate_key_sequence("a")
        assert valid is True

    def test_single_digit_valid(self) -> None:
        valid, _ = ComputerTool._validate_key_sequence("5")
        assert valid is True

    def test_function_key_valid(self) -> None:
        valid, _ = ComputerTool._validate_key_sequence("F5")
        assert valid is True

    def test_three_part_combo_valid(self) -> None:
        valid, _ = ComputerTool._validate_key_sequence("CTRL+SHIFT+T")
        assert valid is True

    def test_unknown_key_fails(self) -> None:
        valid, error = ComputerTool._validate_key_sequence("CTRL+INVALIDXYZ")
        assert valid is False
        assert error and "unknown" in error.lower()

    def test_empty_segment_fails(self) -> None:
        valid, error = ComputerTool._validate_key_sequence("CTRL++")
        assert valid is False
        assert error and "empty" in error.lower()


class TestKeyAction:
    """ComputerTool._key validates before invoking pyautogui."""

    def test_invalid_key_returns_failure(self) -> None:
        tool = ComputerTool()
        result = tool._key("CTRL+NOTAKEY")
        assert result.success is False
        assert "Invalid key sequence" in result.error

    def test_empty_key_returns_failure(self) -> None:
        tool = ComputerTool()
        result = tool._key("")
        assert result.success is False


class TestTypeAction:
    """ComputerTool._type validates input."""

    def test_empty_text_returns_failure(self) -> None:
        tool = ComputerTool()
        result = tool._type("")
        assert result.success is False
        assert "empty" in result.error.lower()

    def test_too_long_text_returns_failure(self) -> None:
        tool = ComputerTool()
        result = tool._type("x" * 10001)
        assert result.success is False
        assert "too long" in result.error.lower()


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-only adapter")
class TestSessionManagerKeyParser:
    """_WindowMessageInputAdapter._parse_key_sequence raises on invalid input."""

    def _adapter(self):
        from agentra.desktop_automation.session_manager import _WindowMessageInputAdapter
        return _WindowMessageInputAdapter.__new__(_WindowMessageInputAdapter)

    def test_unclosed_brace_raises(self) -> None:
        adapter = self._adapter()
        with pytest.raises(ValueError, match="Unclosed brace"):
            adapter._parse_key_sequence("{CTRL+S")

    def test_empty_token_raises(self) -> None:
        adapter = self._adapter()
        with pytest.raises(ValueError, match="Empty key token"):
            adapter._parse_key_sequence("{}")

    def test_unknown_modifier_raises(self) -> None:
        adapter = self._adapter()
        with pytest.raises(ValueError, match="Unknown modifier"):
            adapter._parse_key_sequence("{NOTAMOD+S}")

    def test_unknown_key_raises(self) -> None:
        adapter = self._adapter()
        with pytest.raises(ValueError, match="Unknown key"):
            adapter._parse_key_sequence("{CTRL+UNKNOWNKEY}")

    def test_known_combo_parses(self) -> None:
        adapter = self._adapter()
        events = adapter._parse_key_sequence("{CTRL+S}")
        assert len(events) == 1
        assert events[0]["kind"] == "key"
        assert events[0]["modifiers"]

    def test_plain_text_parses(self) -> None:
        adapter = self._adapter()
        events = adapter._parse_key_sequence("hi")
        assert len(events) == 2
        assert all(e["kind"] == "char" for e in events)


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-only adapter")
class TestStuckModifierTracking:
    """_pressed_keys tracking and release."""

    def test_pressed_keys_initialized(self) -> None:
        from agentra.desktop_automation.session_manager import _WindowMessageInputAdapter
        adapter = _WindowMessageInputAdapter.__new__(_WindowMessageInputAdapter)
        adapter._pressed_keys = {}
        assert adapter._pressed_keys == {}

    def test_release_stuck_modifiers_clears_tracking(self) -> None:
        from agentra.desktop_automation.session_manager import _WindowMessageInputAdapter
        adapter = _WindowMessageInputAdapter.__new__(_WindowMessageInputAdapter)
        adapter._pressed_keys = {123: {17, 65}}
        # Use an invalid handle so PostMessageW silently fails but we exercise the cleanup path.
        adapter._release_stuck_modifiers(123)
        assert 123 not in adapter._pressed_keys

    def test_is_window_responsive_zero_handle(self) -> None:
        from agentra.desktop_automation.session_manager import _WindowMessageInputAdapter
        adapter = _WindowMessageInputAdapter.__new__(_WindowMessageInputAdapter)
        assert adapter._is_window_responsive(0) is False
