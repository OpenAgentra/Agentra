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

    async def click(self, x: float, y: float) -> None:
        self.clicked = (x, y)

    async def wheel(self, *, delta_x: float, delta_y: float) -> None:
        self.wheel_args = (delta_x, delta_y)


class FakeKeyboard:
    """Minimal keyboard stub."""

    def __init__(self) -> None:
        self.typed: str | None = None

    async def type(self, text: str) -> None:
        self.typed = text


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

    async def goto(self, url: str, timeout: int, wait_until: str | None = None) -> None:
        self.goto_url = url
        self.goto_wait_until = wait_until
        self.url = url

    async def title(self) -> str:
        return "Example"

    async def screenshot(self, *, type: str) -> bytes:
        assert type == "png"
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

    def locator(self, selector: str) -> FakeLocator:
        assert selector == "#cta"
        return FakeLocator({"x": 100, "y": 50, "width": 200, "height": 100})


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
async def test_browser_text_action_does_not_create_visual_frame() -> None:
    tool = BrowserTool(headless=True)
    tool._page = FakePage()

    result = await tool.execute(action="get_text")

    assert result.success is True
    assert result.output == "text:body"
    assert result.screenshot_b64 is None
    assert result.metadata == {}
