"""Computer control tool — mouse, keyboard, and screenshots for the desktop."""

from __future__ import annotations

import base64
import ctypes
import io
import math
import os
from typing import Any, Literal, Optional

from agentra.tools.base import BaseTool, ToolResult


class ComputerTool(BaseTool):
    """
    Control the computer's mouse and keyboard, and capture screenshots.
    This gives the agent full desktop access — it can open applications,
    interact with any GUI element, and observe the screen.
    """

    name = "computer"
    tool_capabilities = ("computer",)
    description = (
        "Control the computer: move the mouse, click, type keyboard input, "
        "take a screenshot of the entire screen, and read what is visible. "
        "Use this to interact with any desktop application — file managers, "
        "IDEs, native apps — anything that cannot be reached via a browser."
    )

    def __init__(self, allow: bool = True) -> None:
        self._allow = allow
        self._last_observation: dict[str, Any] | None = None
        self._runtime_controller: Any = None

    def bind_runtime(self, *, controller: Any = None, **_: Any) -> None:
        """Attach runtime helpers such as preview-window parking."""
        self._runtime_controller = controller

    async def preview(self, **kwargs: Any) -> dict[str, Any] | None:
        action = str(kwargs.get("action", "")).strip().lower()
        if not action:
            return None
        return {
            "frame_label": f"computer · {action}",
            "summary": self._summary_for_action(action, kwargs),
        }

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
        capture_result_screenshot = bool(kwargs.get("capture_result_screenshot", True))
        preview_adjusted = self._prepare_visible_desktop_action(action)
        if preview_adjusted:
            self._last_observation = self._capture_desktop_observation()
        conflict = self._desktop_control_conflict(action)
        if conflict is not None:
            return ToolResult(
                success=False,
                error=str(conflict.get("content") or conflict.get("summary") or "Desktop control conflict detected."),
                metadata=conflict,
            )

        try:
            if action == "screenshot":
                result = self._take_screenshot()
            elif action == "click":
                result = self._click(
                    kwargs.get("x"),
                    kwargs.get("y"),
                    button="left",
                    capture_result_screenshot=capture_result_screenshot,
                )
            elif action == "double_click":
                result = self._double_click(
                    kwargs.get("x"),
                    kwargs.get("y"),
                    capture_result_screenshot=capture_result_screenshot,
                )
            elif action == "right_click":
                result = self._click(
                    kwargs.get("x"),
                    kwargs.get("y"),
                    button="right",
                    capture_result_screenshot=capture_result_screenshot,
                )
            elif action == "move":
                result = self._move(
                    kwargs.get("x"),
                    kwargs.get("y"),
                    capture_result_screenshot=capture_result_screenshot,
                )
            elif action == "type":
                result = self._type(
                    kwargs.get("text", ""),
                    capture_result_screenshot=capture_result_screenshot,
                )
            elif action == "key":
                result = self._key(
                    kwargs.get("text", ""),
                    capture_result_screenshot=capture_result_screenshot,
                )
            elif action == "scroll":
                result = self._scroll(
                    kwargs.get("x"),
                    kwargs.get("y"),
                    kwargs.get("delta_y", 3),
                    capture_result_screenshot=capture_result_screenshot,
                )
            elif action == "drag":
                result = self._drag(
                    kwargs.get("x"),
                    kwargs.get("y"),
                    kwargs.get("end_x"),
                    kwargs.get("end_y"),
                    capture_result_screenshot=capture_result_screenshot,
                )
            else:
                result = ToolResult(success=False, error=f"Unknown action: {action!r}")
        except ImportError as exc:
            return ToolResult(
                success=False,
                error=f"Required library not available: {exc}. "
                "Install pyautogui and mss for desktop control.",
            )
        except Exception as exc:  # noqa: BLE001
            return ToolResult(success=False, error=str(exc))
        if result.success:
            self._remember_desktop_observation()
        return result

    # ── private helpers ────────────────────────────────────────────────────────

    def _prepare_visible_desktop_action(self, action: str) -> bool:
        normalized_action = str(action or "").strip().lower()
        if normalized_action == "screenshot":
            return False
        if self._runtime_controller is None:
            return False
        prepare = getattr(self._runtime_controller, "prepare_for_visible_desktop_action", None)
        if not callable(prepare):
            return False
        try:
            payload = prepare()
        except Exception:  # noqa: BLE001
            return False
        return bool((payload or {}).get("hidden_preview_window"))

    def _desktop_control_conflict(self, action: str) -> dict[str, Any] | None:
        normalized_action = str(action or "").strip().lower()
        if normalized_action == "screenshot":
            self._remember_desktop_observation()
            return None
        current = self._capture_desktop_observation()
        if current is None:
            return None
        if self._observation_looks_like_agentra(current):
            return {
                "pause_kind": "desktop_control_takeover",
                "summary": "Masaustu gorunumu Agentra penceresinde kaldigi icin kontrol size devredildi.",
                "content": (
                    "Gorunur masaustu otomasyonu icin hedef pencereyi one getirin; Agentra onizlemesini "
                    "ekrandan cekin veya farkli bir yere tasiyin. Hazir olunca Finish Control ile devam edin."
                ),
                "desktop_observation": current,
            }
        if self._desktop_observation_drifted(current, self._last_observation):
            return {
                "pause_kind": "desktop_control_takeover",
                "summary": "Masaustu beklenmedik sekilde degistigi icin kontrol size devredildi.",
                "content": (
                    "Masaustu hedefi degismis gorunuyor. Istediginiz pencereyi yeniden one getirin veya "
                    "manuel duzeltmeyi tamamlayin; sonra Finish Control ile devam edin."
                ),
                "desktop_observation": current,
            }
        return None

    def _remember_desktop_observation(self) -> None:
        observation = self._capture_desktop_observation()
        if observation is not None:
            self._last_observation = observation

    def _capture_desktop_observation(self) -> dict[str, Any] | None:
        cursor = self._cursor_position()
        window = self._foreground_window_snapshot()
        if cursor is None and window is None:
            return None
        return {
            "window_handle": int((window or {}).get("window_handle", 0) or 0),
            "window_title": str((window or {}).get("window_title", "") or ""),
            "cursor_x": int((cursor or {}).get("x", 0) or 0),
            "cursor_y": int((cursor or {}).get("y", 0) or 0),
        }

    @staticmethod
    def _observation_looks_like_agentra(observation: dict[str, Any] | None) -> bool:
        title = str((observation or {}).get("window_title", "") or "").casefold()
        return "agentra" in title

    @staticmethod
    def _desktop_observation_drifted(
        current: dict[str, Any] | None,
        previous: dict[str, Any] | None,
    ) -> bool:
        if current is None or previous is None:
            return False
        current_handle = int(current.get("window_handle", 0) or 0)
        previous_handle = int(previous.get("window_handle", 0) or 0)
        if current_handle and previous_handle and current_handle != previous_handle:
            return True
        current_title = str(current.get("window_title", "") or "").strip()
        previous_title = str(previous.get("window_title", "") or "").strip()
        if not current_handle and not previous_handle and current_title and previous_title and current_title != previous_title:
            return True
        delta_x = int(current.get("cursor_x", 0) or 0) - int(previous.get("cursor_x", 0) or 0)
        delta_y = int(current.get("cursor_y", 0) or 0) - int(previous.get("cursor_y", 0) or 0)
        return math.hypot(delta_x, delta_y) >= 32

    @staticmethod
    def _cursor_position() -> dict[str, int] | None:
        try:
            import pyautogui  # noqa: PLC0415
        except ImportError:
            return None
        try:
            position = pyautogui.position()
        except Exception:  # noqa: BLE001
            return None
        return {"x": int(getattr(position, "x", 0)), "y": int(getattr(position, "y", 0))}

    @staticmethod
    def _foreground_window_snapshot() -> dict[str, Any] | None:
        if os.name != "nt":
            return None
        windll = getattr(ctypes, "windll", None)
        if windll is None:
            return None
        hwnd = windll.user32.GetForegroundWindow()
        if not hwnd:
            return None
        length = windll.user32.GetWindowTextLengthW(hwnd)
        title_buffer = ctypes.create_unicode_buffer(max(int(length) + 1, 1))
        windll.user32.GetWindowTextW(hwnd, title_buffer, len(title_buffer))
        return {
            "window_handle": int(hwnd),
            "window_title": str(title_buffer.value or "").strip(),
        }

    def _take_screenshot(self) -> ToolResult:
        return ToolResult(
            success=True,
            output="Screenshot taken.",
            screenshot_b64=self._capture_screenshot_b64(),
            metadata={
                "frame_label": "computer · screenshot",
                "summary": "Capturing the desktop",
            },
        )

    def _click(
        self,
        x: Optional[float],
        y: Optional[float],
        button: Literal["left", "right", "middle"] = "left",
        *,
        capture_result_screenshot: bool = True,
    ) -> ToolResult:
        if x is None or y is None:
            return ToolResult(success=False, error="x and y are required for click.")
        import pyautogui  # noqa: PLC0415

        pyautogui.click(x=int(x), y=int(y), button=button)
        action = "right_click" if button == "right" else "click"
        return self._desktop_result(
            action=action,
            output=f"Clicked ({x}, {y}) with {button} button.",
            summary="Right-clicking on the desktop" if button == "right" else "Clicking on the desktop",
            capture_result_screenshot=capture_result_screenshot,
        )

    def _double_click(
        self,
        x: Optional[float],
        y: Optional[float],
        *,
        capture_result_screenshot: bool = True,
    ) -> ToolResult:
        if x is None or y is None:
            return ToolResult(success=False, error="x and y are required for double_click.")
        import pyautogui  # noqa: PLC0415

        pyautogui.doubleClick(x=int(x), y=int(y))
        return self._desktop_result(
            action="double_click",
            output=f"Double-clicked ({x}, {y}).",
            summary="Double-clicking on the desktop",
            capture_result_screenshot=capture_result_screenshot,
        )

    def _move(
        self,
        x: Optional[float],
        y: Optional[float],
        *,
        capture_result_screenshot: bool = True,
    ) -> ToolResult:
        if x is None or y is None:
            return ToolResult(success=False, error="x and y are required for move.")
        import pyautogui  # noqa: PLC0415

        pyautogui.moveTo(int(x), int(y))
        return self._desktop_result(
            action="move",
            output=f"Moved cursor to ({x}, {y}).",
            summary="Moving the cursor",
            capture_result_screenshot=capture_result_screenshot,
        )

    def _type(self, text: str, *, capture_result_screenshot: bool = True) -> ToolResult:
        import pyautogui  # noqa: PLC0415

        pyautogui.typewrite(text, interval=0.02)
        return self._desktop_result(
            action="type",
            output=f"Typed: {text!r}",
            summary="Typing text on the desktop",
            capture_result_screenshot=capture_result_screenshot,
        )

    def _key(self, key: str, *, capture_result_screenshot: bool = True) -> ToolResult:
        import pyautogui  # noqa: PLC0415

        pyautogui.hotkey(*key.split("+"))
        return self._desktop_result(
            action="key",
            output=f"Pressed key: {key!r}",
            summary=f"Pressing {key}" if key else "Pressing a key",
            capture_result_screenshot=capture_result_screenshot,
        )

    def _scroll(
        self,
        x: Optional[float],
        y: Optional[float],
        delta_y: float,
        *,
        capture_result_screenshot: bool = True,
    ) -> ToolResult:
        import pyautogui  # noqa: PLC0415

        clicks = int(delta_y)
        if x is not None and y is not None:
            pyautogui.scroll(clicks, x=int(x), y=int(y))
        else:
            pyautogui.scroll(clicks)
        return self._desktop_result(
            action="scroll",
            output=f"Scrolled {clicks} clicks.",
            summary="Scrolling on the desktop",
            capture_result_screenshot=capture_result_screenshot,
        )

    def _drag(
        self,
        x: Optional[float],
        y: Optional[float],
        end_x: Optional[float],
        end_y: Optional[float],
        *,
        capture_result_screenshot: bool = True,
    ) -> ToolResult:
        if any(v is None for v in [x, y, end_x, end_y]):
            return ToolResult(success=False, error="x, y, end_x, end_y are required for drag.")
        import pyautogui  # noqa: PLC0415

        pyautogui.moveTo(int(x), int(y))
        pyautogui.dragTo(int(end_x), int(end_y), duration=0.5, button="left")
        return self._desktop_result(
            action="drag",
            output=f"Dragged from ({x},{y}) to ({end_x},{end_y}).",
            summary="Dragging on the desktop",
            capture_result_screenshot=capture_result_screenshot,
        )

    def capture_png_bytes(self) -> bytes:
        try:
            import mss  # noqa: PLC0415
            import mss.tools  # noqa: PLC0415

            with mss.mss() as sct:
                monitor = sct.monitors[0]  # all monitors combined
                img = sct.grab(monitor)
                return mss.tools.to_png(img.rgb, img.size)
        except ImportError:
            import pyautogui  # noqa: PLC0415

            buf = io.BytesIO()
            pyautogui.screenshot().save(buf, format="PNG")
            return buf.getvalue()

    def _capture_screenshot_b64(self) -> str:
        return base64.b64encode(self.capture_png_bytes()).decode()

    def _desktop_result(
        self,
        *,
        action: str,
        output: str,
        summary: str,
        capture_result_screenshot: bool = True,
    ) -> ToolResult:
        metadata = {
            "frame_label": f"computer · {action}",
            "summary": summary,
        }
        if not capture_result_screenshot:
            return ToolResult(success=True, output=output, metadata=metadata)
        return ToolResult(
            success=True,
            output=output,
            screenshot_b64=self._capture_screenshot_b64(),
            metadata=metadata,
        )

    @staticmethod
    def _summary_for_action(action: str, args: dict[str, Any]) -> str:
        if action == "screenshot":
            return "Capturing the desktop"
        if action == "click":
            return "Clicking on the desktop"
        if action == "double_click":
            return "Double-clicking on the desktop"
        if action == "right_click":
            return "Right-clicking on the desktop"
        if action == "move":
            return "Moving the cursor"
        if action == "type":
            text = str(args.get("text", "") or "")
            return f"Typing {text!r}" if text else "Typing text on the desktop"
        if action == "key":
            text = str(args.get("text", "") or "")
            return f"Pressing {text}" if text else "Pressing a key"
        if action == "scroll":
            return "Scrolling on the desktop"
        if action == "drag":
            return "Dragging on the desktop"
        return "Preparing a desktop action"
