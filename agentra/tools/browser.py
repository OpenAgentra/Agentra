"""Browser tool — autonomous web browsing via Playwright."""

from __future__ import annotations

import asyncio
import base64
from typing import Any, Literal, Optional

from agentra.tools.base import BaseTool, ToolResult


class BrowserTool(BaseTool):
    """
    Lets the agent open a browser, navigate to URLs, click elements,
    fill forms, take screenshots, and extract page content.
    """

    name = "browser"
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
    ) -> None:
        self._headless = headless
        self._browser_type = browser_type
        self._playwright: Any = None
        self._browser: Any = None
        self._page: Any = None

    # ── lifecycle ──────────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Launch the browser. Called automatically on first use."""
        from playwright.async_api import async_playwright  # noqa: PLC0415

        self._playwright = await async_playwright().start()
        launcher = getattr(self._playwright, self._browser_type)
        self._browser = await launcher.launch(headless=self._headless)
        self._page = await self._browser.new_page()

    async def stop(self) -> None:
        """Close the browser and Playwright context."""
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()
        self._browser = None
        self._page = None
        self._playwright = None

    async def _ensure_started(self) -> None:
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
                        "scroll",
                        "screenshot",
                        "get_text",
                        "get_html",
                        "wait",
                        "back",
                        "forward",
                        "new_tab",
                        "close_tab",
                    ],
                    "description": "Browser action to perform.",
                },
                "url": {"type": "string", "description": "URL to navigate to."},
                "selector": {
                    "type": "string",
                    "description": "CSS selector or text content (prefixed with 'text=').",
                },
                "text": {"type": "string", "description": "Text to type."},
                "x": {"type": "number", "description": "X coordinate for click/scroll."},
                "y": {"type": "number", "description": "Y coordinate for click/scroll."},
                "delta_y": {"type": "number", "description": "Pixels to scroll vertically."},
                "timeout": {
                    "type": "number",
                    "description": "Timeout in milliseconds (default 5000).",
                },
            },
            "required": ["action"],
        }

    # ── execute ────────────────────────────────────────────────────────────────

    async def execute(self, **kwargs: Any) -> ToolResult:
        await self._ensure_started()
        action: str = kwargs.get("action", "")
        try:
            if action == "navigate":
                return await self._navigate(kwargs.get("url", ""))
            if action == "click":
                return await self._click(
                    kwargs.get("selector"), kwargs.get("x"), kwargs.get("y")
                )
            if action == "type":
                return await self._type(kwargs.get("selector"), kwargs.get("text", ""))
            if action == "scroll":
                return await self._scroll(
                    kwargs.get("x", 0), kwargs.get("y", 0), kwargs.get("delta_y", 500)
                )
            if action == "screenshot":
                return await self._screenshot()
            if action == "get_text":
                return await self._get_text(kwargs.get("selector"))
            if action == "get_html":
                return await self._get_html(kwargs.get("selector"))
            if action == "wait":
                await asyncio.sleep(kwargs.get("timeout", 1000) / 1000)
                return ToolResult(success=True, output="Waited.")
            if action == "back":
                await self._page.go_back()
                return ToolResult(success=True, output="Navigated back.")
            if action == "forward":
                await self._page.go_forward()
                return ToolResult(success=True, output="Navigated forward.")
            if action == "new_tab":
                self._page = await self._browser.new_page()
                return ToolResult(success=True, output="Opened new tab.")
            if action == "close_tab":
                await self._page.close()
                self._page = await self._browser.new_page()
                return ToolResult(success=True, output="Closed tab.")
            return ToolResult(success=False, error=f"Unknown action: {action!r}")
        except Exception as exc:  # noqa: BLE001
            return ToolResult(success=False, error=str(exc))

    # ── private actions ────────────────────────────────────────────────────────

    async def _navigate(self, url: str) -> ToolResult:
        if not url:
            return ToolResult(success=False, error="url is required for 'navigate'")
        await self._page.goto(url, timeout=30000)
        title = await self._page.title()
        return ToolResult(success=True, output=f"Navigated to {url!r}. Page title: {title!r}")

    async def _click(
        self,
        selector: Optional[str],
        x: Optional[float],
        y: Optional[float],
    ) -> ToolResult:
        if selector:
            await self._page.click(selector, timeout=5000)
            return ToolResult(success=True, output=f"Clicked {selector!r}")
        if x is not None and y is not None:
            await self._page.mouse.click(x, y)
            return ToolResult(success=True, output=f"Clicked ({x}, {y})")
        return ToolResult(success=False, error="Provide selector or x,y coordinates.")

    async def _type(self, selector: Optional[str], text: str) -> ToolResult:
        if selector:
            await self._page.fill(selector, text)
            return ToolResult(success=True, output=f"Typed into {selector!r}")
        await self._page.keyboard.type(text)
        return ToolResult(success=True, output="Typed text.")

    async def _scroll(self, x: float, y: float, delta_y: float) -> ToolResult:
        await self._page.mouse.wheel(delta_x=0, delta_y=delta_y)
        return ToolResult(success=True, output=f"Scrolled {delta_y}px")

    async def _screenshot(self) -> ToolResult:
        png_bytes = await self._page.screenshot(type="png")
        b64 = base64.b64encode(png_bytes).decode()
        url = self._page.url
        return ToolResult(
            success=True,
            output=f"Screenshot captured. Current URL: {url}",
            screenshot_b64=b64,
        )

    async def _get_text(self, selector: Optional[str]) -> ToolResult:
        if selector:
            text = await self._page.inner_text(selector)
        else:
            text = await self._page.inner_text("body")
        return ToolResult(success=True, output=text[:8000])

    async def _get_html(self, selector: Optional[str]) -> ToolResult:
        if selector:
            html = await self._page.inner_html(selector)
        else:
            html = await self._page.content()
        return ToolResult(success=True, output=html[:8000])
