"""Computer control tool — mouse, keyboard, and screenshots for the desktop."""

from __future__ import annotations

import base64
import io
from typing import Any, Literal, Optional

from agentra.tools.base import BaseTool, ToolResult


class ComputerTool(BaseTool):
    """
    Control the computer's mouse and keyboard, and capture screenshots.
    This gives the agent full desktop access — it can open applications,
    interact with any GUI element, and observe the screen.
    """

    name = "computer"
    description = (
        "Control the computer: move the mouse, click, type keyboard input, "
        "take a screenshot of the entire screen, and read what is visible. "
        "Use this to interact with any desktop application — file managers, "
        "IDEs, native apps — anything that cannot be reached via a browser."
    )

    def __init__(self, allow: bool = True) -> None:
        self._allow = allow

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "screenshot",
                        "click",
                        "double_click",
                        "right_click",
                        "move",
                        "type",
                        "key",
                        "scroll",
                        "drag",
                    ],
                    "description": "Desktop action to perform.",
                },
                "x": {"type": "number", "description": "X coordinate (pixels from left)."},
                "y": {"type": "number", "description": "Y coordinate (pixels from top)."},
                "text": {"type": "string", "description": "Text to type or key combination."},
                "end_x": {
                    "type": "number",
                    "description": "Destination X for drag action.",
                },
                "end_y": {
                    "type": "number",
                    "description": "Destination Y for drag action.",
                },
                "delta_y": {
                    "type": "number",
                    "description": "Vertical scroll amount (positive = down).",
                },
            },
            "required": ["action"],
        }

    async def execute(self, **kwargs: Any) -> ToolResult:
        if not self._allow:
            return ToolResult(
                success=False,
                error="Computer control is disabled in the current configuration.",
            )

        action: str = kwargs.get("action", "")

        try:
            if action == "screenshot":
                return self._take_screenshot()
            if action == "click":
                return self._click(kwargs.get("x"), kwargs.get("y"), button="left")
            if action == "double_click":
                return self._double_click(kwargs.get("x"), kwargs.get("y"))
            if action == "right_click":
                return self._click(kwargs.get("x"), kwargs.get("y"), button="right")
            if action == "move":
                return self._move(kwargs.get("x"), kwargs.get("y"))
            if action == "type":
                return self._type(kwargs.get("text", ""))
            if action == "key":
                return self._key(kwargs.get("text", ""))
            if action == "scroll":
                return self._scroll(kwargs.get("x"), kwargs.get("y"), kwargs.get("delta_y", 3))
            if action == "drag":
                return self._drag(
                    kwargs.get("x"),
                    kwargs.get("y"),
                    kwargs.get("end_x"),
                    kwargs.get("end_y"),
                )
            return ToolResult(success=False, error=f"Unknown action: {action!r}")
        except ImportError as exc:
            return ToolResult(
                success=False,
                error=f"Required library not available: {exc}. "
                "Install pyautogui and mss for desktop control.",
            )
        except Exception as exc:  # noqa: BLE001
            return ToolResult(success=False, error=str(exc))

    # ── private helpers ────────────────────────────────────────────────────────

    def _take_screenshot(self) -> ToolResult:
        try:
            import mss  # noqa: PLC0415
            import mss.tools  # noqa: PLC0415

            with mss.mss() as sct:
                monitor = sct.monitors[0]  # all monitors combined
                img = sct.grab(monitor)
                png_bytes = mss.tools.to_png(img.rgb, img.size)
        except ImportError:
            # Fallback to pyautogui
            import pyautogui  # noqa: PLC0415

            buf = io.BytesIO()
            pyautogui.screenshot().save(buf, format="PNG")
            png_bytes = buf.getvalue()

        b64 = base64.b64encode(png_bytes).decode()
        return ToolResult(success=True, output="Screenshot taken.", screenshot_b64=b64)

    def _click(
        self,
        x: Optional[float],
        y: Optional[float],
        button: Literal["left", "right", "middle"] = "left",
    ) -> ToolResult:
        if x is None or y is None:
            return ToolResult(success=False, error="x and y are required for click.")
        import pyautogui  # noqa: PLC0415

        pyautogui.click(x=int(x), y=int(y), button=button)
        return ToolResult(success=True, output=f"Clicked ({x}, {y}) with {button} button.")

    def _double_click(self, x: Optional[float], y: Optional[float]) -> ToolResult:
        if x is None or y is None:
            return ToolResult(success=False, error="x and y are required for double_click.")
        import pyautogui  # noqa: PLC0415

        pyautogui.doubleClick(x=int(x), y=int(y))
        return ToolResult(success=True, output=f"Double-clicked ({x}, {y}).")

    def _move(self, x: Optional[float], y: Optional[float]) -> ToolResult:
        if x is None or y is None:
            return ToolResult(success=False, error="x and y are required for move.")
        import pyautogui  # noqa: PLC0415

        pyautogui.moveTo(int(x), int(y))
        return ToolResult(success=True, output=f"Moved cursor to ({x}, {y}).")

    def _type(self, text: str) -> ToolResult:
        import pyautogui  # noqa: PLC0415

        pyautogui.typewrite(text, interval=0.02)
        return ToolResult(success=True, output=f"Typed: {text!r}")

    def _key(self, key: str) -> ToolResult:
        import pyautogui  # noqa: PLC0415

        pyautogui.hotkey(*key.split("+"))
        return ToolResult(success=True, output=f"Pressed key: {key!r}")

    def _scroll(
        self,
        x: Optional[float],
        y: Optional[float],
        delta_y: float,
    ) -> ToolResult:
        import pyautogui  # noqa: PLC0415

        clicks = int(delta_y)
        if x is not None and y is not None:
            pyautogui.scroll(clicks, x=int(x), y=int(y))
        else:
            pyautogui.scroll(clicks)
        return ToolResult(success=True, output=f"Scrolled {clicks} clicks.")

    def _drag(
        self,
        x: Optional[float],
        y: Optional[float],
        end_x: Optional[float],
        end_y: Optional[float],
    ) -> ToolResult:
        if any(v is None for v in [x, y, end_x, end_y]):
            return ToolResult(success=False, error="x, y, end_x, end_y are required for drag.")
        import pyautogui  # noqa: PLC0415

        pyautogui.moveTo(int(x), int(y))
        pyautogui.dragTo(int(end_x), int(end_y), duration=0.5, button="left")
        return ToolResult(success=True, output=f"Dragged from ({x},{y}) to ({end_x},{end_y}).")
