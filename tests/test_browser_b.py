"""Tests for Group B: Browser interaction improvements."""

from __future__ import annotations

from agentra.tools.browser import BrowserTool


class TestSchemaGroupB:
    """Verify all new actions and properties are in the schema."""

    def _schema(self) -> dict:
        return BrowserTool().schema

    def test_new_actions_in_enum(self) -> None:
        actions = self._schema()["properties"]["action"]["enum"]
        expected = [
            "hover", "double_click", "right_click", "select_option",
            "file_upload", "evaluate_js", "switch_frame", "switch_main_frame",
            "handle_dialog", "get_cookies", "set_cookie", "clear_cookies",
        ]
        for action in expected:
            assert action in actions, f"Missing action: {action}"

    def test_new_properties_exist(self) -> None:
        props = self._schema()["properties"]
        expected_props = [
            "value", "label", "option_index", "filepath", "expression",
            "frame_selector", "frame_index", "dialog_action", "prompt_text",
            "cookie_name", "cookie_value", "cookie_domain", "cookie_url",
        ]
        for prop in expected_props:
            assert prop in props, f"Missing property: {prop}"

    def test_dialog_action_has_enum(self) -> None:
        dialog_action = self._schema()["properties"]["dialog_action"]
        assert dialog_action["enum"] == ["accept", "dismiss"]

    def test_timeout_description_updated(self) -> None:
        timeout = self._schema()["properties"]["timeout"]
        assert "navigate" in timeout["description"].lower()

    def test_total_action_count(self) -> None:
        actions = self._schema()["properties"]["action"]["enum"]
        assert len(actions) == 28


class TestBrowserToolInit:
    """Verify new state variables are initialized."""

    def test_frame_context_initialized(self) -> None:
        tool = BrowserTool()
        assert tool._frame_context is None

    def test_dialog_state_initialized(self) -> None:
        tool = BrowserTool()
        assert tool._last_dialog is None
        assert tool._pending_dialog is None


class TestActiveFrameProperty:
    """Test the _active_frame property."""

    def test_returns_page_when_no_frame(self) -> None:
        tool = BrowserTool()
        tool._page = "fake_page"
        assert tool._active_frame == "fake_page"

    def test_returns_frame_when_set(self) -> None:
        tool = BrowserTool()
        tool._page = "fake_page"
        tool._frame_context = "fake_frame"
        assert tool._active_frame == "fake_frame"


class TestSwitchFrame:
    """Test iframe switching methods."""

    def test_switch_main_frame_resets_context(self) -> None:
        import asyncio
        tool = BrowserTool()
        tool._frame_context = "some_frame"
        result = asyncio.run(tool._switch_main_frame())
        assert result.success is True
        assert tool._frame_context is None

    def test_switch_frame_requires_params(self) -> None:
        import asyncio
        tool = BrowserTool()
        result = asyncio.run(tool._switch_frame(None, None))
        assert result.success is False
        assert "frame_selector or frame_index" in result.error


class TestHandleDialog:
    """Test dialog handling."""

    def test_no_pending_dialog(self) -> None:
        import asyncio
        tool = BrowserTool()
        result = asyncio.run(tool._handle_dialog("accept", None))
        assert result.success is False
        assert "No pending dialog" in result.error

    def test_accept_pending_dialog(self) -> None:
        import asyncio

        class FakeDialog:
            type = "alert"
            message = "Hello!"
            async def accept(self, text=None):
                pass
            async def dismiss(self):
                pass

        tool = BrowserTool()
        tool._pending_dialog = FakeDialog()
        result = asyncio.run(tool._handle_dialog("accept", None))
        assert result.success is True
        assert "Accepted" in result.output
        assert tool._pending_dialog is None

    def test_dismiss_pending_dialog(self) -> None:
        import asyncio

        class FakeDialog:
            type = "confirm"
            message = "Are you sure?"
            async def accept(self, text=None):
                pass
            async def dismiss(self):
                pass

        tool = BrowserTool()
        tool._pending_dialog = FakeDialog()
        result = asyncio.run(tool._handle_dialog("dismiss", None))
        assert result.success is True
        assert "Dismissed" in result.output


class TestEvaluateJs:
    """Test JavaScript evaluation."""

    def test_empty_expression_fails(self) -> None:
        import asyncio
        tool = BrowserTool()
        result = asyncio.run(tool._evaluate_js(""))
        assert result.success is False
        assert "expression is required" in result.error


class TestSelectOption:
    """Test select_option validation."""

    def test_no_selector_fails(self) -> None:
        import asyncio
        tool = BrowserTool()
        result = asyncio.run(tool._select_option(""))
        assert result.success is False

    def test_no_value_fails(self) -> None:
        import asyncio
        tool = BrowserTool()
        result = asyncio.run(tool._select_option("select#test"))
        assert result.success is False
        assert "value, label, or option_index" in result.error


class TestFileUpload:
    """Test file_upload validation."""

    def test_no_selector_fails(self) -> None:
        import asyncio
        tool = BrowserTool()
        result = asyncio.run(tool._file_upload("", "file.txt"))
        assert result.success is False

    def test_no_filepath_fails(self) -> None:
        import asyncio
        tool = BrowserTool()
        result = asyncio.run(tool._file_upload("input#file", ""))
        assert result.success is False


class TestHoverValidation:
    """Test hover input validation."""

    def test_no_input_fails(self) -> None:
        import asyncio
        tool = BrowserTool()
        result = asyncio.run(tool._hover(None, None, None))
        assert result.success is False
        assert "selector or x,y" in result.error


class TestCookieValidation:
    """Test cookie set validation."""

    def test_no_name_fails(self) -> None:
        import asyncio
        tool = BrowserTool()
        result = asyncio.run(tool._set_cookie("", "val"))
        assert result.success is False
        assert "cookie_name" in result.error
