"""Shared browser runtime and thread-scoped browser sessions."""

from __future__ import annotations

import asyncio
import base64
from dataclasses import asdict, dataclass
from typing import Any, Literal

from agentra.tools.base import ToolResult

_DEFAULT_FOCUS = (0.74, 0.2)
_LIVE_REFRESH_INTERVALS = (0.2, 0.35)


@dataclass
class BrowserSnapshot:
    """Serializable summary of the current browser session."""

    active: bool = False
    active_url: str = ""
    active_title: str = ""
    tab_count: int = 0


class BrowserRuntime:
    """Owns a Playwright runtime plus a launched browser instance."""

    def __init__(
        self,
        *,
        browser_type: Literal["chromium", "firefox", "webkit"] = "chromium",
        headless: bool = False,
    ) -> None:
        self.browser_type = browser_type
        self.headless = headless
        self._playwright: Any = None
        self._browser: Any = None
        self._lock = asyncio.Lock()

    async def browser(self) -> Any:
        async with self._lock:
            if self._browser is not None:
                return self._browser
            from playwright.async_api import async_playwright  # noqa: PLC0415

            self._playwright = await async_playwright().start()
            launcher = getattr(self._playwright, self.browser_type)
            self._browser = await launcher.launch(headless=self.headless)
            return self._browser

    async def close(self) -> None:
        async with self._lock:
            if self._browser is not None:
                await self._browser.close()
            if self._playwright is not None:
                await self._playwright.stop()
            self._browser = None
            self._playwright = None


class BrowserSession:
    """A single thread-scoped browser context shared by agent and user."""

    def __init__(self, thread_id: str, runtime: BrowserRuntime) -> None:
        self.thread_id = thread_id
        self.runtime = runtime
        self._context: Any = None
        self._pages: list[Any] = []
        self._page: Any = None
        self._snapshot = BrowserSnapshot()

    async def execute(self, **kwargs: Any) -> ToolResult:
        await self._ensure_started()
        action: str = kwargs.get("action", "")
        try:
            if action == "navigate":
                return await self._navigate(kwargs.get("url", ""))
            if action == "click":
                return await self._click(kwargs.get("selector"), kwargs.get("x"), kwargs.get("y"))
            if action == "type":
                return await self._type(kwargs.get("selector"), kwargs.get("text", ""))
            if action == "scroll":
                delta_y = kwargs.get("delta_y", kwargs.get("amount", 500))
                return await self._scroll(kwargs.get("x", 0), kwargs.get("y", 0), delta_y)
            if action == "screenshot":
                return await self._screenshot()
            if action == "get_text":
                return await self._get_text(kwargs.get("selector"))
            if action == "get_html":
                return await self._get_html(kwargs.get("selector"))
            if action == "wait":
                await asyncio.sleep(kwargs.get("timeout", 1000) / 1000)
                await self._refresh_snapshot()
                return self._plain_result("Waited.")
            if action == "back":
                await self._page.go_back()
                return await self._capture_visual_state(
                    output="Navigated back.",
                    frame_label="browser · back",
                    summary="Going back",
                    focus=self._default_focus(),
                    burst=True,
                )
            if action == "forward":
                await self._page.go_forward()
                return await self._capture_visual_state(
                    output="Navigated forward.",
                    frame_label="browser · forward",
                    summary="Going forward",
                    focus=self._default_focus(),
                    burst=True,
                )
            if action == "new_tab":
                page = await self._context.new_page()
                self._set_active_page(page)
                return await self._capture_visual_state(
                    output="Opened new tab.",
                    frame_label="browser · new_tab",
                    summary="Opening a new tab",
                    focus=self._default_focus(),
                    burst=True,
                )
            if action == "close_tab":
                if self._page is not None:
                    await self._page.close()
                    self._pages = [page for page in self._pages if not getattr(page, "is_closed", lambda: False)()]
                if not self._pages:
                    page = await self._context.new_page()
                    self._pages = [page]
                self._set_active_page(self._pages[-1])
                return await self._capture_visual_state(
                    output="Closed tab.",
                    frame_label="browser · close_tab",
                    summary="Closing the tab",
                    focus=self._default_focus(),
                    burst=True,
                )
            return ToolResult(success=False, error=f"Unknown action: {action!r}")
        except Exception as exc:  # noqa: BLE001
            return ToolResult(success=False, error=str(exc))

    async def preview(self, **kwargs: Any) -> dict[str, Any] | None:
        action: str = kwargs.get("action", "")
        if action == "navigate":
            return self._preview_payload(
                frame_label="browser · navigate",
                summary=f"Opening {self._short_url(kwargs.get('url', 'the page'))}",
                focus=self._default_focus_normalized(),
            )
        if action == "click":
            selector = kwargs.get("selector")
            if selector:
                return self._preview_payload(
                    frame_label="browser · click",
                    summary=f"Clicking {selector}",
                    focus=await self._selector_focus_normalized(selector),
                )
            x = kwargs.get("x")
            y = kwargs.get("y")
            focus = self._normalize_preview_focus(x, y) if x is not None and y is not None else self._default_focus_normalized()
            return self._preview_payload(frame_label="browser · click", summary="Clicking the page", focus=focus)
        if action == "type":
            selector = kwargs.get("selector")
            if selector:
                return self._preview_payload(
                    frame_label="browser · type",
                    summary=f"Typing into {selector}",
                    focus=await self._selector_focus_normalized(selector),
                )
            return self._preview_payload(
                frame_label="browser · type",
                summary="Typing text",
                focus=self._default_focus_normalized(y=0.56),
            )
        if action == "scroll":
            x = kwargs.get("x")
            y = kwargs.get("y")
            focus = self._normalize_preview_focus(x, y) if x or y else self._default_focus_normalized(y=0.78)
            return self._preview_payload(frame_label="browser · scroll", summary="Scrolling the page", focus=focus)
        if action == "screenshot":
            return self._preview_payload(
                frame_label="browser · screenshot",
                summary="Capturing a screenshot",
                focus=self._default_focus_normalized(),
            )
        if action in {"back", "forward", "new_tab", "close_tab"}:
            return self._preview_payload(
                frame_label=f"browser · {action}",
                summary={
                    "back": "Going back",
                    "forward": "Going forward",
                    "new_tab": "Opening a new tab",
                    "close_tab": "Closing the tab",
                }[action],
                focus=self._default_focus_normalized(),
            )
        return None

    def snapshot(self) -> BrowserSnapshot:
        return BrowserSnapshot(
            active=self._snapshot.active,
            active_url=self._snapshot.active_url,
            active_title=self._snapshot.active_title,
            tab_count=self._snapshot.tab_count,
        )

    async def capture_live_png(self) -> bytes | None:
        if self._page is None:
            return None
        png_bytes = await self._page.screenshot(type="png")
        await self._refresh_snapshot()
        return png_bytes

    async def _ensure_started(self) -> None:
        if self._page is not None:
            return
        browser = await self.runtime.browser()
        self._context = await browser.new_context()
        page = await self._context.new_page()
        self._pages = [page]
        self._set_active_page(page)
        await self._refresh_snapshot()

    def _set_active_page(self, page: Any) -> None:
        if page not in self._pages:
            self._pages.append(page)
        self._page = page

    async def _navigate(self, url: str) -> ToolResult:
        if not url:
            return ToolResult(success=False, error="url is required for 'navigate'")
        await self._page.goto(url, timeout=30000, wait_until="domcontentloaded")
        title = await self._page.title()
        return await self._capture_visual_state(
            output=f"Navigated to {url!r}. Page title: {title!r}",
            frame_label="browser · navigate",
            summary=f"Opening {self._short_url(url)}",
            focus=self._default_focus(),
            burst=True,
        )

    async def _click(self, selector: str | None, x: float | None, y: float | None) -> ToolResult:
        if selector:
            focus = await self._selector_focus(selector)
            await self._page.click(selector, timeout=5000)
            return await self._capture_visual_state(
                output=f"Clicked {selector!r}",
                frame_label="browser · click",
                summary=f"Clicking {selector}",
                focus=focus,
                burst=True,
            )
        if x is not None and y is not None:
            await self._page.mouse.click(x, y)
            return await self._capture_visual_state(
                output=f"Clicked ({x}, {y})",
                frame_label="browser · click",
                summary="Clicking the page",
                focus=(x, y),
                burst=True,
            )
        return ToolResult(success=False, error="Provide selector or x,y coordinates.")

    async def _type(self, selector: str | None, text: str) -> ToolResult:
        if selector:
            focus = await self._selector_focus(selector)
            await self._page.fill(selector, text)
            return await self._capture_visual_state(
                output=f"Typed into {selector!r}",
                frame_label="browser · type",
                summary=f"Typing into {selector}",
                focus=focus,
                burst=True,
            )
        await self._page.keyboard.type(text)
        return await self._capture_visual_state(
            output="Typed text.",
            frame_label="browser · type",
            summary="Typing text",
            focus=self._default_focus(y=0.56),
            burst=True,
        )

    async def _scroll(self, x: float, y: float, delta_y: float) -> ToolResult:
        await self._page.mouse.wheel(delta_x=0, delta_y=delta_y)
        focus = (x, y) if x or y else self._default_focus(y=0.78)
        return await self._capture_visual_state(
            output=f"Scrolled {delta_y}px",
            frame_label="browser · scroll",
            summary="Scrolling the page",
            focus=focus,
            burst=True,
        )

    async def _screenshot(self) -> ToolResult:
        return await self._capture_visual_state(
            output=f"Screenshot captured. Current URL: {self._page.url}",
            frame_label="browser · screenshot",
            summary="Capturing a screenshot",
            focus=self._default_focus(),
        )

    async def _get_text(self, selector: str | None) -> ToolResult:
        text = await self._page.inner_text(selector or "body")
        await self._refresh_snapshot()
        return self._plain_result(text[:8000], extracted_text=text[:8000])

    async def _get_html(self, selector: str | None) -> ToolResult:
        html = await self._page.inner_html(selector) if selector else await self._page.content()
        await self._refresh_snapshot()
        return self._plain_result(html[:8000], extracted_text=html[:8000])

    def _plain_result(self, output: str, *, extracted_text: str | None = None) -> ToolResult:
        metadata = self._metadata(summary="", extracted_text=extracted_text)
        return ToolResult(success=True, output=output, metadata=metadata)

    async def _capture_visual_state(
        self,
        *,
        output: str,
        frame_label: str,
        summary: str,
        focus: tuple[float, float],
        burst: bool = False,
    ) -> ToolResult:
        png_bytes = await self._page.screenshot(type="png")
        await self._refresh_snapshot()
        focus_x, focus_y = self._normalize_focus(*focus)
        return ToolResult(
            success=True,
            output=output,
            screenshot_b64=base64.b64encode(png_bytes).decode(),
            extra_screenshots=await self._capture_follow_up_frames(
                frame_label=frame_label,
                summary=summary,
                focus_x=focus_x,
                focus_y=focus_y,
                enabled=burst,
            ),
            metadata=self._metadata(
                summary=summary,
                frame_label=frame_label,
                focus_x=focus_x,
                focus_y=focus_y,
            ),
        )

    def _metadata(
        self,
        *,
        summary: str,
        frame_label: str | None = None,
        focus_x: float | None = None,
        focus_y: float | None = None,
        extracted_text: str | None = None,
    ) -> dict[str, Any]:
        snapshot = self.snapshot()
        metadata: dict[str, Any] = {
            "summary": summary,
            "active_url": snapshot.active_url,
            "active_title": snapshot.active_title,
            "tab_count": snapshot.tab_count,
        }
        if frame_label:
            metadata["frame_label"] = frame_label
        if focus_x is not None:
            metadata["focus_x"] = focus_x
        if focus_y is not None:
            metadata["focus_y"] = focus_y
        if extracted_text:
            metadata["extracted_text"] = extracted_text
        return metadata

    async def _capture_follow_up_frames(
        self,
        *,
        frame_label: str,
        summary: str,
        focus_x: float,
        focus_y: float,
        enabled: bool,
    ) -> list[dict[str, Any]]:
        if not enabled or self._page is None:
            return []
        frames: list[dict[str, Any]] = []
        for index, delay in enumerate(_LIVE_REFRESH_INTERVALS, start=1):
            await asyncio.sleep(delay)
            png_bytes = await self._page.screenshot(type="png")
            await self._refresh_snapshot()
            frame_summary = f"{summary} (refresh {index})"
            frame: dict[str, Any] = {
                "data": base64.b64encode(png_bytes).decode(),
                "frame_label": frame_label,
                "summary": frame_summary,
                "focus_x": focus_x,
                "focus_y": focus_y,
            }
            frame.update(
                {
                    key: value
                    for key, value in self._metadata(
                        summary=frame_summary,
                        frame_label=frame_label,
                        focus_x=focus_x,
                        focus_y=focus_y,
                    ).items()
                    if key not in {"summary", "frame_label", "focus_x", "focus_y"}
                }
            )
            frames.append(frame)
        return frames

    async def _refresh_snapshot(self) -> None:
        if self._page is None:
            self._snapshot = BrowserSnapshot()
            return
        pages = [page for page in self._pages if not getattr(page, "is_closed", lambda: False)()]
        self._pages = pages or [self._page]
        title = ""
        try:
            title = await self._page.title()
        except Exception:  # noqa: BLE001
            title = ""
        self._snapshot = BrowserSnapshot(
            active=True,
            active_url=str(getattr(self._page, "url", "") or ""),
            active_title=title,
            tab_count=len(self._pages),
        )

    async def _selector_focus(self, selector: str) -> tuple[float, float]:
        locator = self._page.locator(selector).first
        box = await locator.bounding_box()
        if box:
            return (box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
        return self._default_focus()

    async def _selector_focus_normalized(self, selector: str) -> tuple[float, float]:
        if self._page is None:
            return self._default_focus_normalized()
        locator = self._page.locator(selector).first
        box = await locator.bounding_box()
        if box:
            return self._normalize_focus(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
        return self._default_focus_normalized()

    def _normalize_focus(self, x: float, y: float) -> tuple[float, float]:
        size = self._page.viewport_size or {"width": 1280, "height": 720}
        width = max(float(size.get("width", 1280)), 1.0)
        height = max(float(size.get("height", 720)), 1.0)
        return (
            max(0.0, min(1.0, float(x) / width)),
            max(0.0, min(1.0, float(y) / height)),
        )

    def _default_focus(self, *, x: float = _DEFAULT_FOCUS[0], y: float = _DEFAULT_FOCUS[1]) -> tuple[float, float]:
        size = self._page.viewport_size or {"width": 1280, "height": 720}
        width = float(size.get("width", 1280))
        height = float(size.get("height", 720))
        return (x * width, y * height)

    @staticmethod
    def _default_focus_normalized(*, x: float = _DEFAULT_FOCUS[0], y: float = _DEFAULT_FOCUS[1]) -> tuple[float, float]:
        return (
            max(0.0, min(1.0, float(x))),
            max(0.0, min(1.0, float(y))),
        )

    def _normalize_preview_focus(self, x: float | None, y: float | None) -> tuple[float, float]:
        if x is None or y is None or self._page is None:
            return self._default_focus_normalized()
        return self._normalize_focus(float(x), float(y))

    @staticmethod
    def _preview_payload(*, frame_label: str, summary: str, focus: tuple[float, float]) -> dict[str, Any]:
        focus_x, focus_y = focus
        return {
            "frame_label": frame_label,
            "summary": summary,
            "focus_x": max(0.0, min(1.0, float(focus_x))),
            "focus_y": max(0.0, min(1.0, float(focus_y))),
        }

    @staticmethod
    def _short_url(url: str) -> str:
        compact = url.replace("https://", "").replace("http://", "").rstrip("/")
        return compact.removeprefix("www.") or url


class BrowserSessionManager:
    """Creates and reuses thread-scoped browser sessions."""

    def __init__(self) -> None:
        self._runtimes: dict[tuple[str, bool], BrowserRuntime] = {}
        self._sessions: dict[str, BrowserSession] = {}
        self._session_runtime_key: dict[str, tuple[str, bool]] = {}
        self._lock = asyncio.Lock()

    async def session_for(
        self,
        *,
        thread_id: str,
        browser_type: Literal["chromium", "firefox", "webkit"] = "chromium",
        headless: bool = False,
    ) -> BrowserSession:
        async with self._lock:
            existing = self._sessions.get(thread_id)
            if existing is not None:
                return existing
            runtime_key = (browser_type, bool(headless))
            runtime = self._runtimes.get(runtime_key)
            if runtime is None:
                runtime = BrowserRuntime(browser_type=browser_type, headless=headless)
                self._runtimes[runtime_key] = runtime
            session = BrowserSession(thread_id=thread_id, runtime=runtime)
            self._sessions[thread_id] = session
            self._session_runtime_key[thread_id] = runtime_key
            return session

    def snapshot(self, thread_id: str) -> BrowserSnapshot:
        session = self._sessions.get(thread_id)
        if session is None:
            return BrowserSnapshot()
        return session.snapshot()

    async def close(self) -> None:
        runtimes = list(self._runtimes.values())
        self._sessions.clear()
        self._session_runtime_key.clear()
        self._runtimes.clear()
        for runtime in runtimes:
            await runtime.close()

    def snapshot_payload(self, thread_id: str) -> dict[str, Any]:
        snapshot = self.snapshot(thread_id)
        return {
            "browser_session_active": snapshot.active,
            "active_url": snapshot.active_url,
            "active_title": snapshot.active_title,
            "tab_count": snapshot.tab_count,
            "browser": asdict(snapshot),
        }

    async def capture_live_png(self, thread_id: str) -> bytes | None:
        session = self._sessions.get(thread_id)
        if session is None:
            return None
        return await session.capture_live_png()
