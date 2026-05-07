"""Browser tool — autonomous web browsing via Playwright."""

from __future__ import annotations

import asyncio
import base64
import io
import json
import uuid
from pathlib import Path
from typing import Any, Literal, Optional

from agentra.browser_runtime import BrowserSessionManager
from agentra.tools.base import BaseTool, ToolResult
from agentra.tools.visual_diff import compute_image_hash, images_are_similar

_DEFAULT_FOCUS = (0.74, 0.2)
_LIVE_REFRESH_INTERVALS = (0.2, 0.35)
_SELECTOR_FOCUS_TIMEOUT_SECONDS = 0.75


class BrowserTool(BaseTool):
    """
    Lets the agent open a browser, navigate to URLs, click elements,
    fill forms, take screenshots, and extract page content.
    """

    name = "browser"
    tool_capabilities = ("browser",)
    description = (
        "Control a web browser. You can navigate to URLs, click elements, "
        "type text, scroll, extract page content, and take screenshots. "
        "Use this to browse websites, log in to accounts, fill forms, and "
        "perform any web-based task on behalf of the user."
    )

    def __init__(
        self,
        headless: bool = False,
        browser_type: Literal["chromium", "firefox", "webkit"] = "chromium",
        identity: Literal["isolated", "chrome_profile"] = "isolated",
        profile_name: str = "Default",
        screenshot_format: str = "png",
        screenshot_quality: int = 85,
    ) -> None:
        self._headless = headless
        self._browser_type = browser_type
        self._identity = identity
        self._profile_name = profile_name or "Default"
        self._screenshot_format = screenshot_format
        self._screenshot_quality = screenshot_quality
        self._playwright: Any = None
        self._browser: Any = None
        self._page: Any = None
        self._last_screenshot_hash: str | None = None
        self._frame_context: Any = None
        self._last_dialog: dict[str, str] | None = None
        self._pending_dialog: Any = None
        self._recording: bool = False
        self._recording_frames: list[bytes] = []
        self._recording_task: asyncio.Task[None] | None = None
        self._recording_fps: int = 4
        self._workspace_dir: Path | None = None
        self._browser_sessions: BrowserSessionManager | None = None
        self._thread_id: str | None = None

    def bind_runtime(
        self,
        *,
        browser_sessions: BrowserSessionManager | None = None,
        thread_id: str | None = None,
    ) -> None:
        """Attach a shared browser session manager for thread-scoped browsing."""
        self._browser_sessions = browser_sessions
        self._thread_id = thread_id

    # ── lifecycle ──────────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Launch the browser. Called automatically on first use."""
        from playwright.async_api import async_playwright  # noqa: PLC0415

        self._playwright = await async_playwright().start()
        launcher = getattr(self._playwright, self._browser_type)
        self._browser = await launcher.launch(headless=self._headless)
        self._page = await self._browser.new_page()
        self._page.on("dialog", self._on_dialog)

    async def stop(self) -> None:
        """Close the browser and Playwright context."""
        if self._browser_sessions is not None and self._thread_id is not None:
            self._page = None
            self._browser = None
            self._playwright = None
            return
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()
        self._browser = None
        self._page = None
        self._playwright = None

    async def _ensure_started(self) -> None:
        if self._browser_sessions is not None and self._thread_id is not None:
            session = await self._browser_sessions.session_for(
                thread_id=self._thread_id,
                browser_type=self._browser_type,
                headless=self._headless,
                identity=self._identity,
                profile_name=self._profile_name,
            )
            snapshot = session.snapshot()
            if snapshot.active:
                self._page = object()
            return
        if self._page is not None and not self._page_is_closed(self._page):
            return
        if self._page is not None:
            await self._recover_after_browser_loss(None)
        if self._page is None:
            await self.start()

    # ── schema ─────────────────────────────────────────────────────────────────

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "navigate",
                        "click",
                        "type",
                        "key",
                        "drag",
                        "scroll",
                        "screenshot",
                        "get_text",
                        "get_html",
                        "wait",
                        "back",
                        "forward",
                        "new_tab",
                        "close_tab",
                        "start_recording",
                        "stop_recording",
                        "hover",
                        "double_click",
                        "right_click",
                        "select_option",
                        "file_upload",
                        "evaluate_js",
                        "switch_frame",
                        "switch_main_frame",
                        "handle_dialog",
                        "get_cookies",
                        "set_cookie",
                        "clear_cookies",
                    ],
                    "description": "Browser action to perform.",
                },
                "url": {"type": "string", "description": "URL to navigate to."},
                "selector": {
                    "type": "string",
                    "description": "CSS selector or text content (prefixed with 'text=').",
                },
                "text": {"type": "string", "description": "Text to type."},
                "key": {"type": "string", "description": "Keyboard key to press."},
                "x": {"type": "number", "description": "X coordinate for click/scroll."},
                "y": {"type": "number", "description": "Y coordinate for click/scroll."},
                "start_x": {"type": "number", "description": "Drag start X coordinate."},
                "start_y": {"type": "number", "description": "Drag start Y coordinate."},
                "end_x": {"type": "number", "description": "Drag end X coordinate."},
                "end_y": {"type": "number", "description": "Drag end Y coordinate."},
                "steps": {"type": "number", "description": "Interpolation steps for drag movement."},
                "delta_y": {"type": "number", "description": "Pixels to scroll vertically."},
                "timeout": {
                    "type": "number",
                    "description": "Timeout in ms. Defaults: navigate=30000, click/type/get_text=5000.",
                },
                "value": {"type": "string", "description": "Option value for select_option."},
                "label": {"type": "string", "description": "Visible label for select_option."},
                "option_index": {"type": "integer", "description": "Zero-based index for select_option."},
                "filepath": {"type": "string", "description": "File path for file_upload."},
                "expression": {"type": "string", "description": "JavaScript expression for evaluate_js."},
                "frame_selector": {"type": "string", "description": "CSS selector for iframe (switch_frame)."},
                "frame_index": {"type": "integer", "description": "Zero-based frame index (switch_frame)."},
                "dialog_action": {
                    "type": "string",
                    "enum": ["accept", "dismiss"],
                    "description": "Accept or dismiss the dialog (handle_dialog).",
                },
                "prompt_text": {"type": "string", "description": "Text for prompt dialog (handle_dialog)."},
                "cookie_name": {"type": "string", "description": "Cookie name (set_cookie)."},
                "cookie_value": {"type": "string", "description": "Cookie value (set_cookie)."},
                "cookie_domain": {"type": "string", "description": "Cookie domain (set_cookie)."},
                "cookie_url": {"type": "string", "description": "URL scope for cookie (set_cookie)."},
                "region_x": {
                    "type": "number",
                    "description": "Left edge of capture region (pixels). Optional, for screenshot action.",
                },
                "region_y": {
                    "type": "number",
                    "description": "Top edge of capture region (pixels). Optional, for screenshot action.",
                },
                "region_width": {
                    "type": "number",
                    "description": "Width of capture region (pixels). Optional, for screenshot action.",
                },
                "region_height": {
                    "type": "number",
                    "description": "Height of capture region (pixels). Optional, for screenshot action.",
                },
            },
            "required": ["action"],
        }

    # ── execute ────────────────────────────────────────────────────────────────

    async def execute(self, **kwargs: Any) -> ToolResult:
        if self._browser_sessions is not None and self._thread_id is not None:
            session = await self._browser_sessions.session_for(
                thread_id=self._thread_id,
                browser_type=self._browser_type,
                headless=self._headless,
                identity=self._identity,
                profile_name=self._profile_name,
            )
            return await session.execute(**kwargs)
        action: str = kwargs.get("action", "")
        capture_result_screenshot = bool(kwargs.get("capture_result_screenshot", True))
        capture_follow_up_screenshots = bool(
            kwargs.get("capture_follow_up_screenshots", capture_result_screenshot)
        )
        for attempt in range(2):
            try:
                await self._ensure_started()
                timeout = kwargs.get("timeout")
                if action == "navigate":
                    return await self._navigate(kwargs.get("url", ""), timeout=timeout)
                if action == "click":
                    return await self._click(
                        kwargs.get("selector"),
                        kwargs.get("x"),
                        kwargs.get("y"),
                        timeout=timeout,
                        capture_result_screenshot=capture_result_screenshot,
                        capture_follow_up_screenshots=capture_follow_up_screenshots,
                    )
                if action == "type":
                    return await self._type(
                        kwargs.get("selector"),
                        kwargs.get("text", ""),
                        timeout=timeout,
                        capture_result_screenshot=capture_result_screenshot,
                        capture_follow_up_screenshots=capture_follow_up_screenshots,
                    )
                if action == "key":
                    return await self._key(
                        kwargs.get("key", ""),
                        capture_result_screenshot=capture_result_screenshot,
                        capture_follow_up_screenshots=capture_follow_up_screenshots,
                    )
                if action == "drag":
                    return await self._drag(
                        kwargs.get("start_x"),
                        kwargs.get("start_y"),
                        kwargs.get("end_x"),
                        kwargs.get("end_y"),
                        kwargs.get("steps", 14),
                        capture_result_screenshot=capture_result_screenshot,
                        capture_follow_up_screenshots=capture_follow_up_screenshots,
                    )
                if action == "scroll":
                    return await self._scroll(
                        kwargs.get("x", 0),
                        kwargs.get("y", 0),
                        kwargs.get("delta_y", 500),
                        capture_result_screenshot=capture_result_screenshot,
                        capture_follow_up_screenshots=capture_follow_up_screenshots,
                    )
                if action == "screenshot":
                    return await self._screenshot(**kwargs)
                if action == "get_text":
                    return await self._get_text(kwargs.get("selector"), timeout=timeout)
                if action == "get_html":
                    return await self._get_html(kwargs.get("selector"), timeout=timeout)
                if action == "wait":
                    await asyncio.sleep(kwargs.get("timeout", 1000) / 1000)
                    return ToolResult(success=True, output="Waited.")
                if action == "back":
                    await self._page.go_back()
                    return await self._action_result(
                        output="Navigated back.",
                        frame_label="browser · back",
                        summary="Going back",
                        focus=self._default_focus(),
                        burst=True,
                        capture_result_screenshot=capture_result_screenshot,
                        capture_follow_up_screenshots=capture_follow_up_screenshots,
                    )
                if action == "forward":
                    await self._page.go_forward()
                    return await self._action_result(
                        output="Navigated forward.",
                        frame_label="browser · forward",
                        summary="Going forward",
                        focus=self._default_focus(),
                        burst=True,
                        capture_result_screenshot=capture_result_screenshot,
                        capture_follow_up_screenshots=capture_follow_up_screenshots,
                    )
                if action == "new_tab":
                    self._page = await self._browser.new_page()
                    self._page.on("dialog", self._on_dialog)
                    return await self._capture_visual_state(
                        output="Opened new tab.",
                        frame_label="browser · new_tab",
                        summary="Opening a new tab",
                        focus=self._default_focus(),
                        burst=True,
                    )
                if action == "close_tab":
                    await self._page.close()
                    self._page = await self._browser.new_page()
                    return await self._capture_visual_state(
                        output="Closed tab.",
                        frame_label="browser · close_tab",
                        summary="Closing the tab",
                        focus=self._default_focus(),
                        burst=True,
                    )
                if action == "start_recording":
                    return await self._start_recording()
                if action == "stop_recording":
                    return await self._stop_recording()
                if action == "hover":
                    return await self._hover(
                        kwargs.get("selector"), kwargs.get("x"), kwargs.get("y"),
                        timeout=timeout,
                        capture_result_screenshot=capture_result_screenshot,
                        capture_follow_up_screenshots=capture_follow_up_screenshots,
                    )
                if action == "double_click":
                    return await self._double_click(
                        kwargs.get("selector"), kwargs.get("x"), kwargs.get("y"),
                        timeout=timeout,
                        capture_result_screenshot=capture_result_screenshot,
                        capture_follow_up_screenshots=capture_follow_up_screenshots,
                    )
                if action == "right_click":
                    return await self._right_click(
                        kwargs.get("selector"), kwargs.get("x"), kwargs.get("y"),
                        timeout=timeout,
                        capture_result_screenshot=capture_result_screenshot,
                        capture_follow_up_screenshots=capture_follow_up_screenshots,
                    )
                if action == "select_option":
                    return await self._select_option(
                        kwargs.get("selector", ""),
                        value=kwargs.get("value"), label=kwargs.get("label"),
                        option_index=kwargs.get("option_index"),
                        timeout=timeout,
                        capture_result_screenshot=capture_result_screenshot,
                        capture_follow_up_screenshots=capture_follow_up_screenshots,
                    )
                if action == "file_upload":
                    return await self._file_upload(
                        kwargs.get("selector", ""), kwargs.get("filepath", ""),
                        capture_result_screenshot=capture_result_screenshot,
                        capture_follow_up_screenshots=capture_follow_up_screenshots,
                    )
                if action == "evaluate_js":
                    return await self._evaluate_js(kwargs.get("expression", ""))
                if action == "switch_frame":
                    return await self._switch_frame(kwargs.get("frame_selector"), kwargs.get("frame_index"))
                if action == "switch_main_frame":
                    return await self._switch_main_frame()
                if action == "handle_dialog":
                    return await self._handle_dialog(kwargs.get("dialog_action", "accept"), kwargs.get("prompt_text"))
                if action == "get_cookies":
                    return await self._get_cookies()
                if action == "set_cookie":
                    return await self._set_cookie(
                        kwargs.get("cookie_name", ""), kwargs.get("cookie_value", ""),
                        cookie_domain=kwargs.get("cookie_domain"), cookie_url=kwargs.get("cookie_url"),
                    )
                if action == "clear_cookies":
                    return await self._clear_cookies()
                return ToolResult(success=False, error=f"Unknown action: {action!r}")
            except Exception as exc:  # noqa: BLE001
                recovered = await self._recover_after_browser_loss(exc)
                if attempt == 0 and recovered:
                    continue
                return ToolResult(success=False, error=str(exc))
        return ToolResult(success=False, error="Browser is unavailable.")

    async def preview(self, **kwargs: Any) -> dict[str, Any] | None:
        if self._browser_sessions is not None and self._thread_id is not None:
            session = await self._browser_sessions.session_for(
                thread_id=self._thread_id,
                browser_type=self._browser_type,
                headless=self._headless,
                identity=self._identity,
                profile_name=self._profile_name,
            )
            return await session.preview(**kwargs)
        """Return visual metadata before a browser action is executed."""
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
            return self._preview_payload(
                frame_label="browser · click",
                summary="Clicking the page",
                focus=focus,
            )
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
        if action == "key":
            return self._preview_payload(
                frame_label="browser · key",
                summary=self._key_summary(kwargs.get("key")),
                focus=self._default_focus_normalized(y=0.56),
            )
        if action == "drag":
            return self._preview_payload(
                frame_label="browser · drag",
                summary="Dragging on the page",
                focus=self._normalize_drag_focus(
                    kwargs.get("start_x"),
                    kwargs.get("start_y"),
                    kwargs.get("end_x"),
                    kwargs.get("end_y"),
                ),
            )
        if action == "scroll":
            x = kwargs.get("x")
            y = kwargs.get("y")
            focus = self._normalize_preview_focus(x, y) if x or y else self._default_focus_normalized(y=0.78)
            return self._preview_payload(
                frame_label="browser · scroll",
                summary="Scrolling the page",
                focus=focus,
            )
        if action == "screenshot":
            return self._preview_payload(
                frame_label="browser · screenshot",
                summary="Capturing a screenshot",
                focus=self._default_focus_normalized(),
            )
        if action == "back":
            return self._preview_payload(
                frame_label="browser · back",
                summary="Going back",
                focus=self._default_focus_normalized(),
            )
        if action == "forward":
            return self._preview_payload(
                frame_label="browser · forward",
                summary="Going forward",
                focus=self._default_focus_normalized(),
            )
        if action == "new_tab":
            return self._preview_payload(
                frame_label="browser · new_tab",
                summary="Opening a new tab",
                focus=self._default_focus_normalized(),
            )
        if action == "close_tab":
            return self._preview_payload(
                frame_label="browser · close_tab",
                summary="Closing the tab",
                focus=self._default_focus_normalized(),
            )
        return None

    # ── private actions ────────────────────────────────────────────────────────

    @property
    def _active_frame(self) -> Any:
        """Return the current frame context or the page itself."""
        return self._frame_context if self._frame_context is not None else self._page

    async def _navigate(self, url: str, *, timeout: int | None = None) -> ToolResult:
        if not url:
            return ToolResult(success=False, error="url is required for 'navigate'")
        await self._page.goto(url, timeout=timeout or 30000, wait_until="domcontentloaded")
        title = await self._page.title()
        return await self._capture_visual_state(
            output=f"Navigated to {url!r}. Page title: {title!r}",
            frame_label="browser · navigate",
            summary=f"Opening {self._short_url(url)}",
            focus=self._default_focus(),
            burst=True,
        )

    async def _click(
        self,
        selector: Optional[str],
        x: Optional[float],
        y: Optional[float],
        *,
        timeout: int | None = None,
        capture_result_screenshot: bool = True,
        capture_follow_up_screenshots: bool = True,
    ) -> ToolResult:
        if selector:
            focus = await self._selector_focus(selector)
            await self._active_frame.click(selector, timeout=timeout or 5000)
            return await self._action_result(
                output=f"Clicked {selector!r}",
                frame_label="browser · click",
                summary=f"Clicking {selector}",
                focus=focus,
                burst=True,
                capture_result_screenshot=capture_result_screenshot,
                capture_follow_up_screenshots=capture_follow_up_screenshots,
            )
        if x is not None and y is not None:
            await self._page.mouse.click(x, y)
            return await self._action_result(
                output=f"Clicked ({x}, {y})",
                frame_label="browser · click",
                summary="Clicking the page",
                focus=(x, y),
                burst=True,
                capture_result_screenshot=capture_result_screenshot,
                capture_follow_up_screenshots=capture_follow_up_screenshots,
            )
        return ToolResult(success=False, error="Provide selector or x,y coordinates.")

    async def _type(
        self,
        selector: Optional[str],
        text: str,
        *,
        timeout: int | None = None,
        capture_result_screenshot: bool = True,
        capture_follow_up_screenshots: bool = True,
    ) -> ToolResult:
        if selector:
            focus = await self._selector_focus(selector)
            await self._active_frame.fill(selector, text, timeout=timeout or 5000)
            return await self._action_result(
                output=f"Typed into {selector!r}",
                frame_label="browser · type",
                summary=f"Typing into {selector}",
                focus=focus,
                burst=True,
                capture_result_screenshot=capture_result_screenshot,
                capture_follow_up_screenshots=capture_follow_up_screenshots,
            )
        await self._page.keyboard.type(text)
        return await self._action_result(
            output="Typed text.",
            frame_label="browser · type",
            summary="Typing text",
            focus=self._default_focus(y=0.56),
            burst=True,
            capture_result_screenshot=capture_result_screenshot,
            capture_follow_up_screenshots=capture_follow_up_screenshots,
        )

    async def _key(
        self,
        key: str,
        *,
        capture_result_screenshot: bool = True,
        capture_follow_up_screenshots: bool = True,
    ) -> ToolResult:
        if not key:
            return ToolResult(success=False, error="key is required for 'key'")
        await self._page.keyboard.press(key)
        return await self._action_result(
            output=f"Pressed {key!r}.",
            frame_label="browser · key",
            summary=self._key_summary(key),
            focus=self._default_focus(y=0.56),
            burst=True,
            capture_result_screenshot=capture_result_screenshot,
            capture_follow_up_screenshots=capture_follow_up_screenshots,
        )

    async def _drag(
        self,
        start_x: float | None,
        start_y: float | None,
        end_x: float | None,
        end_y: float | None,
        steps: int | None,
        *,
        capture_result_screenshot: bool = True,
        capture_follow_up_screenshots: bool = True,
    ) -> ToolResult:
        if None in {start_x, start_y, end_x, end_y}:
            return ToolResult(success=False, error="start_x, start_y, end_x, and end_y are required for 'drag'")
        drag_steps = max(1, int(steps or 14))
        await self._page.mouse.move(float(start_x), float(start_y))
        await self._page.mouse.down()
        await self._page.mouse.move(float(end_x), float(end_y), steps=drag_steps)
        await self._page.mouse.up()
        return await self._action_result(
            output=f"Dragged ({start_x}, {start_y}) to ({end_x}, {end_y})",
            frame_label="browser · drag",
            summary="Dragging on the page",
            focus=(float(end_x), float(end_y)),
            burst=True,
            capture_result_screenshot=capture_result_screenshot,
            capture_follow_up_screenshots=capture_follow_up_screenshots,
        )

    async def _scroll(
        self,
        x: float,
        y: float,
        delta_y: float,
        *,
        capture_result_screenshot: bool = True,
        capture_follow_up_screenshots: bool = True,
    ) -> ToolResult:
        await self._page.mouse.wheel(delta_x=0, delta_y=delta_y)
        focus = (x, y) if x or y else self._default_focus(y=0.78)
        return await self._action_result(
            output=f"Scrolled {delta_y}px",
            frame_label="browser · scroll",
            summary="Scrolling the page",
            focus=focus,
            burst=True,
            capture_result_screenshot=capture_result_screenshot,
            capture_follow_up_screenshots=capture_follow_up_screenshots,
        )

    async def _screenshot(self, **kwargs: Any) -> ToolResult:
        clip = self._extract_clip(kwargs)
        return await self._capture_visual_state(
            output=f"Screenshot captured. Current URL: {self._page.url}",
            frame_label="browser · screenshot",
            summary="Capturing a screenshot",
            focus=self._default_focus(),
            clip=clip,
        )

    @staticmethod
    def _extract_clip(kwargs: dict[str, Any]) -> dict[str, float] | None:
        rx = kwargs.get("region_x")
        ry = kwargs.get("region_y")
        rw = kwargs.get("region_width")
        rh = kwargs.get("region_height")
        if all(v is not None for v in (rx, ry, rw, rh)):
            return {"x": float(rx), "y": float(ry), "width": float(rw), "height": float(rh)}
        return None

    async def _start_recording(self) -> ToolResult:
        if self._recording:
            return ToolResult(success=False, error="Recording is already in progress.")
        self._recording = True
        self._recording_frames = []

        async def _capture_loop() -> None:
            interval = 1.0 / self._recording_fps
            while self._recording:
                try:
                    frame = await self._page.screenshot(type="png")
                    self._recording_frames.append(frame)
                except Exception:  # noqa: BLE001
                    pass
                await asyncio.sleep(interval)

        self._recording_task = asyncio.create_task(_capture_loop())
        return ToolResult(
            success=True,
            output=f"Browser recording started at {self._recording_fps} FPS.",
            metadata={"frame_label": "browser · start_recording", "summary": "Started browser recording"},
        )

    async def _stop_recording(self) -> ToolResult:
        if not self._recording:
            return ToolResult(success=False, error="No recording in progress.")
        self._recording = False
        if self._recording_task is not None:
            self._recording_task.cancel()
            try:
                await self._recording_task
            except asyncio.CancelledError:
                pass
            self._recording_task = None

        frames = self._recording_frames
        self._recording_frames = []
        if not frames:
            return ToolResult(success=False, error="No frames were captured during recording.")

        from PIL import Image as _Image  # noqa: PLC0415

        images = [_Image.open(io.BytesIO(f)) for f in frames]
        buf = io.BytesIO()
        duration = int(1000 / self._recording_fps)
        images[0].save(
            buf,
            format="GIF",
            save_all=True,
            append_images=images[1:],
            duration=duration,
            loop=0,
        )
        gif_bytes = buf.getvalue()

        save_dir = self._workspace_dir or Path.cwd() / "workspace"
        recordings_dir = save_dir / ".memory" / "recordings"
        recordings_dir.mkdir(parents=True, exist_ok=True)
        gif_path = recordings_dir / f"{uuid.uuid4().hex[:12]}.gif"
        gif_path.write_bytes(gif_bytes)

        return ToolResult(
            success=True,
            output=f"Recording saved: {gif_path} ({len(frames)} frames, {len(gif_bytes)} bytes)",
            metadata={
                "frame_label": "browser · stop_recording",
                "summary": "Stopped browser recording and saved GIF",
                "recording_path": str(gif_path),
                "frame_count": len(frames),
            },
        )

    async def _action_result(
        self,
        *,
        output: str,
        frame_label: str,
        summary: str,
        focus: tuple[float, float],
        burst: bool = False,
        capture_result_screenshot: bool = True,
        capture_follow_up_screenshots: bool = True,
    ) -> ToolResult:
        if capture_result_screenshot:
            return await self._capture_visual_state(
                output=output,
                frame_label=frame_label,
                summary=summary,
                focus=focus,
                burst=burst and capture_follow_up_screenshots,
            )
        focus_x, focus_y = self._normalize_focus(*focus)
        return ToolResult(
            success=True,
            output=output,
            metadata={
                "focus_x": focus_x,
                "focus_y": focus_y,
                "frame_label": frame_label,
                "summary": summary,
            },
        )

    async def _get_text(self, selector: Optional[str], *, timeout: int | None = None) -> ToolResult:
        timeout_s = (timeout or 5000) / 1000
        target = self._active_frame
        if selector:
            coro = target.inner_text(selector)
        else:
            coro = target.inner_text("body")
        text = await asyncio.wait_for(coro, timeout=timeout_s)
        return ToolResult(success=True, output=text[:8000])

    async def _get_html(self, selector: Optional[str], *, timeout: int | None = None) -> ToolResult:
        timeout_s = (timeout or 5000) / 1000
        target = self._active_frame
        if selector:
            coro = target.inner_html(selector)
        else:
            coro = self._page.content()
        html = await asyncio.wait_for(coro, timeout=timeout_s)
        return ToolResult(success=True, output=html[:8000])

    # ── B1: new interaction actions ──────────────────────────────────────────

    async def _hover(
        self, selector: Optional[str], x: Optional[float], y: Optional[float],
        *, timeout: int | None = None,
        capture_result_screenshot: bool = True, capture_follow_up_screenshots: bool = True,
    ) -> ToolResult:
        if selector:
            focus = await self._selector_focus(selector)
            await self._active_frame.hover(selector, timeout=timeout or 5000)
            return await self._action_result(
                output=f"Hovered {selector!r}", frame_label="browser · hover",
                summary=f"Hovering {selector}", focus=focus, burst=True,
                capture_result_screenshot=capture_result_screenshot,
                capture_follow_up_screenshots=capture_follow_up_screenshots,
            )
        if x is not None and y is not None:
            await self._page.mouse.move(x, y)
            return await self._action_result(
                output=f"Hovered ({x}, {y})", frame_label="browser · hover",
                summary="Hovering on the page", focus=(x, y), burst=True,
                capture_result_screenshot=capture_result_screenshot,
                capture_follow_up_screenshots=capture_follow_up_screenshots,
            )
        return ToolResult(success=False, error="Provide selector or x,y coordinates for hover.")

    async def _double_click(
        self, selector: Optional[str], x: Optional[float], y: Optional[float],
        *, timeout: int | None = None,
        capture_result_screenshot: bool = True, capture_follow_up_screenshots: bool = True,
    ) -> ToolResult:
        if selector:
            focus = await self._selector_focus(selector)
            await self._active_frame.dblclick(selector, timeout=timeout or 5000)
            return await self._action_result(
                output=f"Double-clicked {selector!r}", frame_label="browser · double_click",
                summary=f"Double-clicking {selector}", focus=focus, burst=True,
                capture_result_screenshot=capture_result_screenshot,
                capture_follow_up_screenshots=capture_follow_up_screenshots,
            )
        if x is not None and y is not None:
            await self._page.mouse.dblclick(x, y)
            return await self._action_result(
                output=f"Double-clicked ({x}, {y})", frame_label="browser · double_click",
                summary="Double-clicking the page", focus=(x, y), burst=True,
                capture_result_screenshot=capture_result_screenshot,
                capture_follow_up_screenshots=capture_follow_up_screenshots,
            )
        return ToolResult(success=False, error="Provide selector or x,y coordinates for double_click.")

    async def _right_click(
        self, selector: Optional[str], x: Optional[float], y: Optional[float],
        *, timeout: int | None = None,
        capture_result_screenshot: bool = True, capture_follow_up_screenshots: bool = True,
    ) -> ToolResult:
        if selector:
            focus = await self._selector_focus(selector)
            await self._active_frame.click(selector, button="right", timeout=timeout or 5000)
            return await self._action_result(
                output=f"Right-clicked {selector!r}", frame_label="browser · right_click",
                summary=f"Right-clicking {selector}", focus=focus, burst=True,
                capture_result_screenshot=capture_result_screenshot,
                capture_follow_up_screenshots=capture_follow_up_screenshots,
            )
        if x is not None and y is not None:
            await self._page.mouse.click(x, y, button="right")
            return await self._action_result(
                output=f"Right-clicked ({x}, {y})", frame_label="browser · right_click",
                summary="Right-clicking the page", focus=(x, y), burst=True,
                capture_result_screenshot=capture_result_screenshot,
                capture_follow_up_screenshots=capture_follow_up_screenshots,
            )
        return ToolResult(success=False, error="Provide selector or x,y coordinates for right_click.")

    async def _select_option(
        self, selector: str, *, value: str | None = None, label: str | None = None,
        option_index: int | None = None, timeout: int | None = None,
        capture_result_screenshot: bool = True, capture_follow_up_screenshots: bool = True,
    ) -> ToolResult:
        if not selector:
            return ToolResult(success=False, error="selector is required for select_option.")
        option_kwargs: dict[str, Any] = {}
        if value is not None:
            option_kwargs["value"] = value
        elif label is not None:
            option_kwargs["label"] = label
        elif option_index is not None:
            option_kwargs["index"] = option_index
        else:
            return ToolResult(success=False, error="Provide value, label, or option_index for select_option.")
        focus = await self._selector_focus(selector)
        await self._active_frame.select_option(selector, timeout=timeout or 5000, **option_kwargs)
        desc = value or label or str(option_index)
        return await self._action_result(
            output=f"Selected {desc!r} in {selector!r}", frame_label="browser · select_option",
            summary=f"Selecting option in {selector}", focus=focus, burst=True,
            capture_result_screenshot=capture_result_screenshot,
            capture_follow_up_screenshots=capture_follow_up_screenshots,
        )

    async def _file_upload(
        self, selector: str, filepath: str, *,
        capture_result_screenshot: bool = True, capture_follow_up_screenshots: bool = True,
    ) -> ToolResult:
        if not selector:
            return ToolResult(success=False, error="selector is required for file_upload.")
        if not filepath:
            return ToolResult(success=False, error="filepath is required for file_upload.")
        focus = await self._selector_focus(selector)
        await self._active_frame.set_input_files(selector, filepath)
        return await self._action_result(
            output=f"Uploaded {filepath!r} to {selector!r}", frame_label="browser · file_upload",
            summary=f"Uploading file to {selector}", focus=focus, burst=True,
            capture_result_screenshot=capture_result_screenshot,
            capture_follow_up_screenshots=capture_follow_up_screenshots,
        )

    async def _evaluate_js(self, expression: str) -> ToolResult:
        if not expression:
            return ToolResult(success=False, error="expression is required for evaluate_js.")
        result = await self._active_frame.evaluate(expression)
        output = json.dumps(result, default=str) if result is not None else "undefined"
        return ToolResult(success=True, output=output[:8000])

    # ── B2: iframe support ────────────────────────────────────────────────────

    async def _switch_frame(self, frame_selector: str | None, frame_index: int | None) -> ToolResult:
        if frame_selector:
            self._frame_context = self._page.frame_locator(frame_selector)
            return ToolResult(success=True, output=f"Switched to frame: {frame_selector!r}")
        if frame_index is not None:
            frames = self._page.frames
            if frame_index < 0 or frame_index >= len(frames):
                return ToolResult(
                    success=False,
                    error=f"Frame index {frame_index} out of range (0-{len(frames) - 1}).",
                )
            self._frame_context = frames[frame_index]
            return ToolResult(success=True, output=f"Switched to frame index {frame_index}.")
        return ToolResult(success=False, error="Provide frame_selector or frame_index.")

    async def _switch_main_frame(self) -> ToolResult:
        self._frame_context = None
        return ToolResult(success=True, output="Switched back to main frame.")

    # ── B3: dialog handling ───────────────────────────────────────────────────

    def _on_dialog(self, dialog: Any) -> None:
        self._last_dialog = {"type": dialog.type, "message": dialog.message}
        self._pending_dialog = dialog
        asyncio.ensure_future(self._auto_dismiss_dialog(dialog))

    async def _auto_dismiss_dialog(self, dialog: Any) -> None:
        await asyncio.sleep(5.0)
        if self._pending_dialog is dialog:
            try:
                await dialog.dismiss()
            except Exception:  # noqa: BLE001
                pass
            self._pending_dialog = None

    async def _handle_dialog(self, dialog_action: str, prompt_text: str | None) -> ToolResult:
        if self._pending_dialog is None:
            info = self._last_dialog or {}
            return ToolResult(
                success=False, error="No pending dialog to handle.",
                output=f"Last dialog info: {info}" if info else "",
            )
        dialog = self._pending_dialog
        self._pending_dialog = None
        if dialog_action == "dismiss":
            await dialog.dismiss()
            return ToolResult(success=True, output=f"Dismissed {dialog.type} dialog: {dialog.message!r}")
        if prompt_text is not None:
            await dialog.accept(prompt_text)
        else:
            await dialog.accept()
        return ToolResult(success=True, output=f"Accepted {dialog.type} dialog: {dialog.message!r}")

    # ── B4: cookie management ─────────────────────────────────────────────────

    async def _get_cookies(self) -> ToolResult:
        cookies = await self._page.context.cookies()
        output = json.dumps(cookies, default=str)
        return ToolResult(success=True, output=output[:8000])

    async def _set_cookie(
        self, name: str, value: str, *,
        cookie_domain: str | None = None, cookie_url: str | None = None,
    ) -> ToolResult:
        if not name:
            return ToolResult(success=False, error="cookie_name is required.")
        cookie: dict[str, Any] = {"name": name, "value": value}
        if cookie_url:
            cookie["url"] = cookie_url
        elif cookie_domain:
            cookie["domain"] = cookie_domain
            cookie["path"] = "/"
        else:
            cookie["url"] = self._page.url
        await self._page.context.add_cookies([cookie])
        return ToolResult(success=True, output=f"Cookie {name!r} set.")

    async def _clear_cookies(self) -> ToolResult:
        await self._page.context.clear_cookies()
        return ToolResult(success=True, output="All cookies cleared.")

    # ── visual capture ────────────────────────────────────────────────────────

    async def _capture_browser_screenshot(
        self,
        clip: dict[str, float] | None = None,
    ) -> bytes:
        """Capture a browser screenshot in the configured format.

        *clip* is an optional ``{"x": ..., "y": ..., "width": ..., "height": ...}`` dict.
        """
        fmt = self._screenshot_format
        ss_kwargs: dict[str, Any] = {}
        if clip:
            ss_kwargs["clip"] = clip
        if fmt == "jpeg":
            return await self._page.screenshot(type="jpeg", quality=self._screenshot_quality, **ss_kwargs)
        png_bytes = await self._page.screenshot(type="png", **ss_kwargs)
        if fmt == "webp":
            from PIL import Image as _Image  # noqa: PLC0415
            img = _Image.open(io.BytesIO(png_bytes))
            buf = io.BytesIO()
            img.save(buf, format="WEBP", quality=self._screenshot_quality)
            return buf.getvalue()
        return png_bytes

    async def _capture_visual_state(
        self,
        *,
        output: str,
        frame_label: str,
        summary: str,
        focus: tuple[float, float],
        burst: bool = False,
        clip: dict[str, float] | None = None,
    ) -> ToolResult:
        img_bytes = await self._capture_browser_screenshot(clip=clip)
        focus_x, focus_y = self._normalize_focus(*focus)
        current_hash = compute_image_hash(img_bytes)
        no_change = images_are_similar(self._last_screenshot_hash, current_hash)
        self._last_screenshot_hash = current_hash
        return ToolResult(
            success=True,
            output=output,
            screenshot_b64=base64.b64encode(img_bytes).decode(),
            extra_screenshots=await self._capture_follow_up_frames(
                frame_label=frame_label,
                summary=summary,
                focus_x=focus_x,
                focus_y=focus_y,
                enabled=burst,
            ),
            metadata={
                "focus_x": focus_x,
                "focus_y": focus_y,
                "frame_label": frame_label,
                "summary": summary,
                "no_change": no_change,
                "image_format": self._screenshot_format,
            },
        )

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
            frames.append(
                {
                    "data": base64.b64encode(png_bytes).decode(),
                    "frame_label": frame_label,
                    "summary": f"{summary} (refresh {index})",
                    "focus_x": focus_x,
                    "focus_y": focus_y,
                }
            )
        return frames

    async def _recover_after_browser_loss(self, exc: Exception | None) -> bool:
        if exc is not None and not self._looks_like_closed_browser_error(exc):
            return False
        try:
            await self.stop()
        except Exception:  # noqa: BLE001
            self._browser = None
            self._page = None
            self._playwright = None
        return True

    @staticmethod
    def _looks_like_closed_browser_error(exc: Exception) -> bool:
        message = str(exc).lower()
        if not message:
            return False
        if "target page" in message and "closed" in message:
            return True
        if "browser has been closed" in message:
            return True
        if "browser is closed" in message:
            return True
        if "context has been closed" in message:
            return True
        if "page has been closed" in message:
            return True
        if "connection closed" in message:
            return True
        if "disconnected" in message and "browser" in message:
            return True
        return False

    @staticmethod
    def _page_is_closed(page: Any) -> bool:
        checker = getattr(page, "is_closed", None)
        if callable(checker):
            try:
                return bool(checker())
            except Exception:  # noqa: BLE001
                return True
        return False

    async def _selector_focus(self, selector: str) -> tuple[float, float]:
        box = await self._selector_box(selector)
        if box:
            return (box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
        return self._default_focus()

    async def _selector_focus_normalized(self, selector: str) -> tuple[float, float]:
        if self._page is None:
            return self._default_focus_normalized()
        box = await self._selector_box(selector)
        if box:
            return self._normalize_focus(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
        return self._default_focus_normalized()

    async def _selector_box(self, selector: str) -> dict[str, float] | None:
        if self._page is None:
            return None
        locator = self._page.locator(selector).first
        try:
            box = await asyncio.wait_for(
                locator.bounding_box(),
                timeout=_SELECTOR_FOCUS_TIMEOUT_SECONDS,
            )
        except Exception:  # noqa: BLE001
            return None
        return box

    def _normalize_focus(self, x: float, y: float) -> tuple[float, float]:
        size = self._page.viewport_size or {"width": 1280, "height": 720}
        width = max(float(size.get("width", 1280)), 1.0)
        height = max(float(size.get("height", 720)), 1.0)
        return (
            max(0.0, min(1.0, float(x) / width)),
            max(0.0, min(1.0, float(y) / height)),
        )

    def _default_focus(
        self,
        *,
        x: float = _DEFAULT_FOCUS[0],
        y: float = _DEFAULT_FOCUS[1],
    ) -> tuple[float, float]:
        size = self._page.viewport_size or {"width": 1280, "height": 720}
        width = float(size.get("width", 1280))
        height = float(size.get("height", 720))
        return (x * width, y * height)

    @staticmethod
    def _default_focus_normalized(
        *,
        x: float = _DEFAULT_FOCUS[0],
        y: float = _DEFAULT_FOCUS[1],
    ) -> tuple[float, float]:
        return (
            max(0.0, min(1.0, float(x))),
            max(0.0, min(1.0, float(y))),
        )

    def _normalize_preview_focus(self, x: float | None, y: float | None) -> tuple[float, float]:
        if x is None or y is None:
            return self._default_focus_normalized()
        if self._page is None:
            return self._default_focus_normalized()
        return self._normalize_focus(float(x), float(y))

    def _normalize_drag_focus(
        self,
        start_x: float | None,
        start_y: float | None,
        end_x: float | None,
        end_y: float | None,
    ) -> tuple[float, float]:
        if end_x is not None and end_y is not None:
            return self._normalize_preview_focus(end_x, end_y)
        if start_x is not None and start_y is not None:
            return self._normalize_preview_focus(start_x, start_y)
        return self._default_focus_normalized()

    @staticmethod
    def _preview_payload(
        *,
        frame_label: str,
        summary: str,
        focus: tuple[float, float],
    ) -> dict[str, Any]:
        focus_x, focus_y = focus
        return {
            "frame_label": frame_label,
            "summary": summary,
            "focus_x": max(0.0, min(1.0, float(focus_x))),
            "focus_y": max(0.0, min(1.0, float(focus_y))),
        }

    @staticmethod
    def _key_summary(key: Any) -> str:
        value = str(key or "").strip()
        return f"Pressing {value}" if value else "Pressing a key"

    @staticmethod
    def _short_url(url: str) -> str:
        compact = url.replace("https://", "").replace("http://", "").rstrip("/")
        return compact.removeprefix("www.") or url
