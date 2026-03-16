"""Tests for shared browser runtime sessions."""

from __future__ import annotations

import pytest

import agentra.browser_runtime as browser_runtime


class FakeMouse:
    async def click(self, x: float, y: float) -> None:
        self.clicked = (x, y)

    async def wheel(self, *, delta_x: float, delta_y: float) -> None:
        self.wheel = (delta_x, delta_y)


class FakeKeyboard:
    async def type(self, text: str) -> None:
        self.typed = text


class FakeLocator:
    @property
    def first(self) -> "FakeLocator":
        return self

    async def bounding_box(self):
        return {"x": 20.0, "y": 10.0, "width": 40.0, "height": 20.0}


class FakePage:
    def __init__(self) -> None:
        self.viewport_size = {"width": 100.0, "height": 100.0}
        self.url = "about:blank"
        self._title = "Blank"
        self.mouse = FakeMouse()
        self.keyboard = FakeKeyboard()
        self.closed = False

    async def goto(self, url: str, timeout: int) -> None:
        self.url = url
        self._title = f"title:{url}"

    async def title(self) -> str:
        return self._title

    async def screenshot(self, *, type: str) -> bytes:
        assert type == "png"
        return b"png"

    async def click(self, selector: str, timeout: int) -> None:
        self.clicked_selector = selector

    async def fill(self, selector: str, text: str) -> None:
        self.filled = (selector, text)

    async def inner_text(self, selector: str) -> str:
        return f"text:{selector}"

    async def inner_html(self, selector: str) -> str:
        return f"<div>{selector}</div>"

    async def content(self) -> str:
        return "<body></body>"

    async def go_back(self) -> None:
        self.url = "about:back"
        self._title = "Back"

    async def go_forward(self) -> None:
        self.url = "about:forward"
        self._title = "Forward"

    async def close(self) -> None:
        self.closed = True

    def is_closed(self) -> bool:
        return self.closed

    def locator(self, selector: str) -> FakeLocator:
        return FakeLocator()


class FakeContext:
    def __init__(self) -> None:
        self.pages: list[FakePage] = []

    async def new_page(self) -> FakePage:
        page = FakePage()
        self.pages.append(page)
        return page


class FakeBrowser:
    async def new_context(self) -> FakeContext:
        return FakeContext()


class FakeBrowserRuntime:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

    async def browser(self) -> FakeBrowser:
        return FakeBrowser()

    async def close(self) -> None:
        return None


@pytest.mark.asyncio
async def test_browser_session_manager_reuses_session_per_thread(monkeypatch) -> None:
    monkeypatch.setattr(browser_runtime, "BrowserRuntime", FakeBrowserRuntime)
    manager = browser_runtime.BrowserSessionManager()

    first = await manager.session_for(thread_id="thread-a")
    second = await manager.session_for(thread_id="thread-a")

    assert first is second

    result = await first.execute(action="navigate", url="https://example.com")
    assert result.success is True
    snapshot = manager.snapshot("thread-a")
    assert snapshot.active is True
    assert snapshot.active_url == "https://example.com"


@pytest.mark.asyncio
async def test_browser_session_manager_isolates_threads(monkeypatch) -> None:
    monkeypatch.setattr(browser_runtime, "BrowserRuntime", FakeBrowserRuntime)
    manager = browser_runtime.BrowserSessionManager()

    first = await manager.session_for(thread_id="thread-a")
    second = await manager.session_for(thread_id="thread-b")

    await first.execute(action="navigate", url="https://a.example")
    await second.execute(action="navigate", url="https://b.example")

    assert manager.snapshot("thread-a").active_url == "https://a.example"
    assert manager.snapshot("thread-b").active_url == "https://b.example"
