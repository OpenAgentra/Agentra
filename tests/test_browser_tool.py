"""Tests for browser-tool visual capture metadata."""

from __future__ import annotations

import pytest

from agentra.tools.browser import BrowserTool


class FakeLocator:
    """Minimal Playwright-like locator stub."""

    def __init__(self, box: dict[str, float] | None) -> None:
        self._box = box

    @property
    def first(self) -> "FakeLocator":
        return self

    async def bounding_box(self) -> dict[str, float] | None:
        return self._box


class FakeMouse:
    """Minimal mouse stub."""

    def __init__(self) -> None:
        self.clicked: tuple[float, float] | None = None
        self.wheel_args: tuple[float, float] | None = None
        self.moves: list[tuple[float, float, int | None]] = []
        self.down_calls = 0
        self.up_calls = 0

    async def click(self, x: float, y: float) -> None:
        self.clicked = (x, y)

    async def move(self, x: float, y: float, steps: int | None = None) -> None:
        self.moves.append((x, y, steps))

    async def down(self) -> None:
        self.down_calls += 1

    async def up(self) -> None:
        self.up_calls += 1

    async def wheel(self, *, delta_x: float, delta_y: float) -> None:
        self.wheel_args = (delta_x, delta_y)


class FakeKeyboard:
    """Minimal keyboard stub."""

    def __init__(self) -> None:
        self.typed: str | None = None
        self.pressed: str | None = None

    async def type(self, text: str) -> None:
        self.typed = text

    async def press(self, key: str) -> None:
        self.pressed = key


class FakePage:
    """Minimal page stub covering the tool's visual actions."""

    def __init__(self) -> None:
        self.viewport_size = {"width": 1000, "height": 500}
        self.url = "https://example.com"
        self.mouse = FakeMouse()
        self.keyboard = FakeKeyboard()
        self.clicked_selector: str | None = None
        self.filled: tuple[str, str] | None = None
        self.goto_url: str | None = None
        self.goto_wait_until: str | None = None
        self.screenshot_calls = 0
        self.closed = False

    async def goto(self, url: str, timeout: int, wait_until: str | None = None) -> None:
        self.goto_url = url
        self.goto_wait_until = wait_until
        self.url = url

    async def title(self) -> str:
        return "Example"

    async def screenshot(self, *, type: str) -> bytes:
        assert type == "png"
        if self.closed:
            raise RuntimeError("Target page, context or browser has been closed")
        self.screenshot_calls += 1
        return b"png-bytes"

    async def click(self, selector: str, timeout: int) -> None:
        self.clicked_selector = selector

    async def fill(self, selector: str, text: str) -> None:
        self.filled = (selector, text)

    async def inner_text(self, selector: str) -> str:
        return f"text:{selector}"

    async def inner_html(self, selector: str) -> str:
        return f"<div>{selector}</div>"

    async def content(self) -> str:
        return "<body>hello</body>"

    async def close(self) -> None:
        self.closed = True

    def is_closed(self) -> bool:
        return self.closed

    def locator(self, selector: str) -> FakeLocator:
        assert selector == "#cta"
        return FakeLocator({"x": 100, "y": 50, "width": 200, "height": 100})


class RecoveringBrowser:
    def __init__(self) -> None:
        self.new_page_calls = 0
        self.closed = False

    async def new_page(self) -> FakePage:
        self.new_page_calls += 1
        return FakePage()

    async def close(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_browser_navigate_returns_frame_metadata() -> None:
    tool = BrowserTool(headless=True)
    tool._page = FakePage()

    result = await tool.execute(action="navigate", url="https://www.python.org")

    assert result.success is True
    assert result.screenshot_b64 is not None
    assert result.metadata["frame_label"] == "browser · navigate"
    assert result.metadata["summary"] == "Opening python.org"
    assert result.metadata["focus_x"] == pytest.approx(0.74, rel=1e-3)
    assert result.metadata["focus_y"] == pytest.approx(0.2, rel=1e-3)
    assert tool._page.goto_wait_until == "domcontentloaded"
    assert len(result.extra_screenshots) == 2
    assert result.extra_screenshots[0]["summary"].startswith("Opening python.org")
    assert tool._page.screenshot_calls == 3


@pytest.mark.asyncio
async def test_browser_selector_click_uses_element_center_for_focus() -> None:
    tool = BrowserTool(headless=True)
    tool._page = FakePage()

    result = await tool.execute(action="click", selector="#cta")

    assert result.success is True
    assert result.metadata["frame_label"] == "browser · click"
    assert result.metadata["focus_x"] == pytest.approx(0.2)
    assert result.metadata["focus_y"] == pytest.approx(0.2)


@pytest.mark.asyncio
async def test_browser_preview_returns_visual_intent_metadata() -> None:
    tool = BrowserTool(headless=True)
    tool._page = FakePage()

    preview = await tool.preview(action="click", selector="#cta")

    assert preview is not None
    assert preview["frame_label"] == "browser · click"
    assert preview["summary"] == "Clicking #cta"
    assert preview["focus_x"] == pytest.approx(0.2)
    assert preview["focus_y"] == pytest.approx(0.2)


@pytest.mark.asyncio
async def test_browser_key_action_returns_frame_metadata() -> None:
    tool = BrowserTool(headless=True)
    tool._page = FakePage()

    result = await tool.execute(action="key", key="Enter")

    assert result.success is True
    assert tool._page.keyboard.pressed == "Enter"
    assert result.metadata["frame_label"] == "browser · key"
    assert result.metadata["summary"] == "Pressing Enter"
    assert result.screenshot_b64 is not None


@pytest.mark.asyncio
async def test_browser_key_preview_returns_visual_intent_metadata() -> None:
    tool = BrowserTool(headless=True)
    tool._page = FakePage()

    preview = await tool.preview(action="key", key="Escape")

    assert preview is not None
    assert preview["frame_label"] == "browser · key"
    assert preview["summary"] == "Pressing Escape"
    assert preview["focus_x"] == pytest.approx(0.74)
    assert preview["focus_y"] == pytest.approx(0.56)


@pytest.mark.asyncio
async def test_browser_drag_action_returns_frame_metadata() -> None:
    tool = BrowserTool(headless=True)
    tool._page = FakePage()

    result = await tool.execute(action="drag", start_x=12, start_y=16, end_x=210, end_y=150, steps=7)

    assert result.success is True
    assert tool._page.mouse.moves == [(12.0, 16.0, None), (210.0, 150.0, 7)]
    assert tool._page.mouse.down_calls == 1
    assert tool._page.mouse.up_calls == 1
    assert result.metadata["frame_label"] == "browser · drag"
    assert result.metadata["summary"] == "Dragging on the page"


@pytest.mark.asyncio
async def test_browser_drag_preview_returns_visual_intent_metadata() -> None:
    tool = BrowserTool(headless=True)
    tool._page = FakePage()

    preview = await tool.preview(action="drag", start_x=12, start_y=16, end_x=210, end_y=150)

    assert preview is not None
    assert preview["frame_label"] == "browser · drag"
    assert preview["summary"] == "Dragging on the page"
    assert preview["focus_x"] == pytest.approx(0.21)
    assert preview["focus_y"] == pytest.approx(0.3)


@pytest.mark.asyncio
async def test_browser_text_action_does_not_create_visual_frame() -> None:
    tool = BrowserTool(headless=True)
    tool._page = FakePage()

    result = await tool.execute(action="get_text")

    assert result.success is True
    assert result.output == "text:body"
    assert result.screenshot_b64 is None
    assert result.metadata == {}


@pytest.mark.asyncio
async def test_browser_tool_recovers_after_browser_window_is_closed() -> None:
    tool = BrowserTool(headless=True)
    browser = RecoveringBrowser()
    tool._browser = browser
    tool._page = FakePage()
    original_page = tool._page
    await original_page.close()

    async def fake_start() -> None:
        tool._browser = browser
        tool._page = await browser.new_page()

    tool.start = fake_start  # type: ignore[method-assign]

    result = await tool.execute(action="screenshot")

    assert result.success is True
    assert tool._page is not original_page
    assert browser.new_page_calls == 1
