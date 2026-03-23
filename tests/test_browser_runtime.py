"""Tests for shared browser runtime sessions."""

from __future__ import annotations

import pytest

import agentra.browser_runtime as browser_runtime


class FakeMouse:
    def __init__(self) -> None:
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
        self.wheel = (delta_x, delta_y)


class FakeKeyboard:
    async def type(self, text: str) -> None:
        self.typed = text

    async def press(self, key: str) -> None:
        self.pressed = key


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
        self.fail_closed_once = False
        self.screenshot_calls: list[dict[str, object]] = []

    async def goto(self, url: str, timeout: int) -> None:
        self.url = url
        self._title = f"title:{url}"

    async def title(self) -> str:
        return self._title

    async def screenshot(self, *, type: str, **kwargs) -> bytes:
        self.screenshot_calls.append({"type": type, **kwargs})
        if self.closed or self.fail_closed_once:
            self.fail_closed_once = False
            raise RuntimeError("Target page, context or browser has been closed")
        return b"jpeg" if type == "jpeg" else b"png"

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
    def __init__(self) -> None:
        self.contexts: list[FakeContext] = []

    async def new_context(self) -> FakeContext:
        context = FakeContext()
        self.contexts.append(context)
        return context


class FakeBrowserRuntime:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.browser_instance = FakeBrowser()
        self.close_calls = 0

    async def browser(self) -> FakeBrowser:
        return self.browser_instance

    async def close(self) -> None:
        self.close_calls += 1
        self.browser_instance = FakeBrowser()
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


@pytest.mark.asyncio
async def test_browser_session_supports_key_actions(monkeypatch) -> None:
    monkeypatch.setattr(browser_runtime, "BrowserRuntime", FakeBrowserRuntime)
    manager = browser_runtime.BrowserSessionManager()

    session = await manager.session_for(thread_id="thread-a")
    result = await session.execute(action="key", key="Enter")

    assert result.success is True
    assert session._page.keyboard.pressed == "Enter"
    assert result.metadata["frame_label"] == "browser · key"
    assert result.metadata["summary"] == "Pressing Enter"


@pytest.mark.asyncio
async def test_browser_session_key_preview_reports_visual_intent(monkeypatch) -> None:
    monkeypatch.setattr(browser_runtime, "BrowserRuntime", FakeBrowserRuntime)
    manager = browser_runtime.BrowserSessionManager()

    session = await manager.session_for(thread_id="thread-a")
    await session.execute(action="navigate", url="https://example.com")
    preview = await session.preview(action="key", key="Tab")

    assert preview is not None
    assert preview["frame_label"] == "browser · key"
    assert preview["summary"] == "Pressing Tab"
    assert preview["focus_x"] == pytest.approx(0.74)
    assert preview["focus_y"] == pytest.approx(0.56)


@pytest.mark.asyncio
async def test_browser_session_supports_drag_actions(monkeypatch) -> None:
    monkeypatch.setattr(browser_runtime, "BrowserRuntime", FakeBrowserRuntime)
    manager = browser_runtime.BrowserSessionManager()

    session = await manager.session_for(thread_id="thread-a")
    result = await session.execute(action="drag", start_x=10, start_y=12, end_x=72, end_y=68, steps=9)

    assert result.success is True
    assert session._page.mouse.moves == [(10.0, 12.0, None), (72.0, 68.0, 9)]
    assert session._page.mouse.down_calls == 1
    assert session._page.mouse.up_calls == 1
    assert result.metadata["frame_label"] == "browser · drag"
    assert result.metadata["summary"] == "Dragging on the page"


@pytest.mark.asyncio
async def test_browser_session_recovers_after_browser_page_is_closed(monkeypatch) -> None:
    monkeypatch.setattr(browser_runtime, "BrowserRuntime", FakeBrowserRuntime)
    manager = browser_runtime.BrowserSessionManager()

    session = await manager.session_for(thread_id="thread-a")
    await session.execute(action="navigate", url="https://example.com")
    first_page = session._page
    await first_page.close()

    result = await session.execute(action="screenshot")

    assert result.success is True
    assert session._page is not first_page
    assert session.runtime.close_calls == 1


@pytest.mark.asyncio
async def test_browser_live_capture_recovers_after_browser_is_removed(monkeypatch) -> None:
    monkeypatch.setattr(browser_runtime, "BrowserRuntime", FakeBrowserRuntime)
    manager = browser_runtime.BrowserSessionManager()

    session = await manager.session_for(thread_id="thread-a")
    await session.execute(action="navigate", url="https://example.com")
    session._page.fail_closed_once = True

    png = await manager.capture_live_png("thread-a")

    assert png == b"png"
    assert session.runtime.close_calls == 1
    assert session.snapshot().active is True


@pytest.mark.asyncio
async def test_browser_live_frame_prefers_fast_jpeg_capture(monkeypatch) -> None:
    monkeypatch.setattr(browser_runtime, "BrowserRuntime", FakeBrowserRuntime)
    manager = browser_runtime.BrowserSessionManager()

    session = await manager.session_for(thread_id="thread-a")
    await session.execute(action="navigate", url="https://example.com")

    frame = await manager.capture_live_frame("thread-a")

    assert frame is not None
    assert frame.data == b"jpeg"
    assert frame.media_type == "image/jpeg"
    assert session._page.screenshot_calls[-1] == {"type": "jpeg", "quality": 55, "scale": "css"}
