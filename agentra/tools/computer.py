"""Computer control tool — mouse, keyboard, and screenshots for the desktop."""

from __future__ import annotations

import asyncio
import base64
import ctypes
import io
import math
import os
import uuid
from pathlib import Path
from typing import Any, Literal, Optional

from agentra.tools.base import BaseTool, ToolResult
from agentra.tools.visual_diff import (
    compute_image_hash,
    compute_structural_hash,
    images_are_similar,
    images_structurally_similar,
)


_VALID_KEY_NAMES = frozenset({
    "ctrl", "shift", "alt", "win", "command", "cmd", "option", "meta",
    "enter", "return", "escape", "esc", "tab", "backspace", "back", "delete", "del",
    "home", "end", "left", "right", "up", "down",
    "space", "insert", "ins", "pageup", "pgup", "pagedown", "pgdn",
    "capslock", "numlock", "scrolllock", "printscreen", "pause",
    "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10", "f11", "f12",
})


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

    def __init__(
        self,
        allow: bool = True,
        screenshot_format: str = "png",
        screenshot_quality: int = 85,
    ) -> None:
        self._allow = allow
        self._screenshot_format = screenshot_format
        self._screenshot_quality = screenshot_quality
        self._last_observation: dict[str, Any] | None = None
        self._last_screenshot_hash: str | None = None
        self._last_screenshot_raw: bytes = b""
        self._recording: bool = False
        self._recording_frames: list[bytes] = []
        self._recording_task: asyncio.Task[None] | None = None
        self._recording_fps: int = 4
        self._workspace_dir: Path | None = None
        self._runtime_controller: Any = None
        self._desktop_sessions: Any = None
        self._thread_id: str | None = None

    def bind_runtime(
        self,
        *,
        controller: Any = None,
        desktop_sessions: Any = None,
        thread_id: str | None = None,
        **_: Any,
    ) -> None:
        """Attach runtime helpers such as preview-window parking."""
        self._runtime_controller = controller
        self._desktop_sessions = desktop_sessions
        self._thread_id = thread_id

    @property
    def capabilities(self) -> tuple[str, ...]:
        session_capability = self._hidden_session_capability()
        if session_capability:
            return (session_capability,)
        return super().capabilities

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
                        "list_monitors",
                        "start_recording",
                        "stop_recording",
                        "extract_text",
                        "find_template",
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
                "monitor": {
                    "type": "integer",
                    "description": "Monitor index (0=all combined, 1=primary, 2=secondary...). Optional, for screenshot action.",
                },
                "lang": {
                    "type": "string",
                    "description": "OCR language hint (e.g. 'eng', 'tur', 'eng+tur'). Optional, for extract_text action.",
                },
                "template_path": {
                    "type": "string",
                    "description": "Path to template image to find on screen. Required for find_template action.",
                },
                "confidence": {
                    "type": "number",
                    "description": "Match confidence threshold 0-1 (default 0.8). Optional, for find_template action.",
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
        session_capability = self._hidden_session_capability()
        if session_capability and self._desktop_sessions is not None and self._thread_id:
            backend_kwargs = dict(kwargs)
            backend_kwargs.pop("action", None)
            try:
                return await asyncio.to_thread(
                    self._desktop_sessions.execute_computer_action,
                    self._thread_id,
                    action,
                    **backend_kwargs,
                )
            except Exception as exc:  # noqa: BLE001
                return ToolResult(success=False, error=str(exc))
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
            if action == "list_monitors":
                result = self._list_monitors()
            elif action == "extract_text":
                result = self._extract_text(**kwargs)
            elif action == "find_template":
                result = self._find_template(**kwargs)
            elif action == "start_recording":
                return await self._start_recording()
            elif action == "stop_recording":
                return await self._stop_recording()
            elif action == "screenshot":
                result = self._take_screenshot(**kwargs)
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

    def _hidden_session_capability(self) -> str | None:
        if self._desktop_sessions is None or not self._thread_id:
            return None
        capability = getattr(self._desktop_sessions, "hidden_capability", None)
        if not callable(capability):
            return None
        try:
            return capability(self._thread_id)
        except Exception:  # noqa: BLE001
            return None

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
        observation: dict[str, Any] = {
            "window_handle": int((window or {}).get("window_handle", 0) or 0),
            "window_title": str((window or {}).get("window_title", "") or ""),
            "cursor_x": int((cursor or {}).get("x", 0) or 0),
            "cursor_y": int((cursor or {}).get("y", 0) or 0),
        }
        try:
            png_bytes = self.capture_screenshot_bytes(fmt="png")
            observation["structural_hash"] = compute_structural_hash(png_bytes)
        except Exception:  # noqa: BLE001
            # Pixel-level drift detection is best-effort; skip on capture failures.
            pass
        return observation

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
        if math.hypot(delta_x, delta_y) >= 32:
            return True
        current_hash = current.get("structural_hash")
        previous_hash = previous.get("structural_hash")
        if current_hash and previous_hash and not images_structurally_similar(
            str(current_hash), str(previous_hash)
        ):
            return True
        return False

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

    def _take_screenshot(self, **kwargs: Any) -> ToolResult:
        region = self._extract_region(kwargs)
        monitor_idx = int(kwargs.get("monitor", 0) or 0)
        img_bytes = self.capture_screenshot_bytes(region=region, monitor=monitor_idx)
        self._last_screenshot_raw = img_bytes
        screenshot_b64 = base64.b64encode(img_bytes).decode()
        changed = self._check_screenshot_change(img_bytes)
        summary = "Capturing the desktop"
        if region:
            summary = f"Capturing region ({region[0]},{region[1]} {region[2]}x{region[3]})"
        elif monitor_idx > 0:
            summary = f"Capturing monitor {monitor_idx}"
        return ToolResult(
            success=True,
            output="Screenshot taken.",
            screenshot_b64=screenshot_b64,
            metadata={
                "frame_label": "computer · screenshot",
                "summary": summary,
                "no_change": not changed,
                "image_format": self._screenshot_format,
            },
        )

    def _list_monitors(self) -> ToolResult:
        import json  # noqa: PLC0415
        monitors = self.list_monitors()
        return ToolResult(
            success=True,
            output=json.dumps(monitors, indent=2),
            metadata={"frame_label": "computer · list_monitors", "summary": "Listing available monitors"},
        )

    # ── C1: OCR text extraction (optional dependency) ─────────────────────────

    def _extract_text(self, **kwargs: Any) -> ToolResult:
        """Extract visible text from screen using OCR. Tries easyocr → pytesseract."""
        region = self._extract_region(kwargs)
        monitor_idx = int(kwargs.get("monitor", 0) or 0)
        lang = str(kwargs.get("lang", "eng") or "eng")
        try:
            png_bytes = self.capture_screenshot_bytes(fmt="png", region=region, monitor=monitor_idx)
        except Exception as exc:  # noqa: BLE001
            return ToolResult(success=False, error=f"Screenshot capture failed: {exc}")

        # Try easyocr first
        try:
            import easyocr  # type: ignore  # noqa: PLC0415
            import numpy as _np  # noqa: PLC0415
            from PIL import Image as _Image  # noqa: PLC0415

            langs = lang.split("+") if "+" in lang else [lang]
            # easyocr uses different lang codes (e.g. 'en' not 'eng')
            normalized = [self._normalize_easyocr_lang(language) for language in langs]
            reader = easyocr.Reader(normalized, gpu=False, verbose=False)
            img = _Image.open(io.BytesIO(png_bytes)).convert("RGB")
            arr = _np.array(img)
            results = reader.readtext(arr)
            blocks = []
            text_parts = []
            for bbox, text, conf in results:
                xs = [p[0] for p in bbox]
                ys = [p[1] for p in bbox]
                blocks.append({
                    "text": text,
                    "bbox": [int(min(xs)), int(min(ys)), int(max(xs) - min(xs)), int(max(ys) - min(ys))],
                    "confidence": float(conf),
                })
                text_parts.append(text)
            return ToolResult(
                success=True,
                output="\n".join(text_parts),
                metadata={
                    "frame_label": "computer · extract_text",
                    "summary": f"Extracted text via easyocr ({len(blocks)} blocks)",
                    "text_blocks": blocks,
                    "ocr_backend": "easyocr",
                },
            )
        except ImportError:
            pass
        except Exception as exc:  # noqa: BLE001
            return ToolResult(success=False, error=f"easyocr failed: {exc}")

        # Fall back to pytesseract
        try:
            import pytesseract  # type: ignore  # noqa: PLC0415
            from PIL import Image as _Image  # noqa: PLC0415

            img = _Image.open(io.BytesIO(png_bytes))
            text = pytesseract.image_to_string(img, lang=lang)
            return ToolResult(
                success=True,
                output=text.strip(),
                metadata={
                    "frame_label": "computer · extract_text",
                    "summary": "Extracted text via pytesseract",
                    "ocr_backend": "pytesseract",
                },
            )
        except ImportError:
            return ToolResult(
                success=False,
                error=(
                    "No OCR backend available. Install one of: "
                    "'pip install easyocr' or 'pip install pytesseract' "
                    "(pytesseract also requires Tesseract binary)."
                ),
            )
        except Exception as exc:  # noqa: BLE001
            return ToolResult(success=False, error=f"pytesseract failed: {exc}")

    @staticmethod
    def _normalize_easyocr_lang(lang: str) -> str:
        """Map common 3-letter codes to easyocr 2-letter codes."""
        mapping = {"eng": "en", "tur": "tr", "deu": "de", "fra": "fr", "spa": "es"}
        return mapping.get(lang.strip().lower(), lang.strip().lower())

    # ── C2: visual element detection (optional dependency) ────────────────────

    def _find_template(self, **kwargs: Any) -> ToolResult:
        """Find a template image on screen via OpenCV template matching."""
        template_path = str(kwargs.get("template_path", "") or "")
        if not template_path:
            return ToolResult(success=False, error="template_path is required for find_template.")
        if not Path(template_path).exists():
            return ToolResult(success=False, error=f"Template not found: {template_path}")

        confidence_threshold = float(kwargs.get("confidence", 0.8) or 0.8)
        region = self._extract_region(kwargs)
        monitor_idx = int(kwargs.get("monitor", 0) or 0)

        try:
            import cv2  # type: ignore  # noqa: PLC0415
            import numpy as _np  # noqa: PLC0415
        except ImportError:
            return ToolResult(
                success=False,
                error="OpenCV not installed. Install with 'pip install opencv-python'.",
            )

        try:
            png_bytes = self.capture_screenshot_bytes(fmt="png", region=region, monitor=monitor_idx)
            from PIL import Image as _Image  # noqa: PLC0415
            haystack_pil = _Image.open(io.BytesIO(png_bytes)).convert("RGB")
            haystack = cv2.cvtColor(_np.array(haystack_pil), cv2.COLOR_RGB2BGR)
            template = cv2.imread(template_path, cv2.IMREAD_COLOR)
            if template is None:
                return ToolResult(success=False, error=f"Could not load template: {template_path}")

            t_h, t_w = template.shape[:2]
            result = cv2.matchTemplate(haystack, template, cv2.TM_CCOEFF_NORMED)
            locations = _np.where(result >= confidence_threshold)
            matches = []
            seen: set[tuple[int, int]] = set()
            for pt in zip(*locations[::-1]):
                # Deduplicate nearby matches
                key = (int(pt[0]) // max(t_w // 2, 1), int(pt[1]) // max(t_h // 2, 1))
                if key in seen:
                    continue
                seen.add(key)
                matches.append({
                    "x": int(pt[0]),
                    "y": int(pt[1]),
                    "width": int(t_w),
                    "height": int(t_h),
                    "confidence": float(result[pt[1], pt[0]]),
                })
        except Exception as exc:  # noqa: BLE001
            return ToolResult(success=False, error=f"Template matching failed: {exc}")

        import json  # noqa: PLC0415
        return ToolResult(
            success=True,
            output=json.dumps(matches, indent=2),
            metadata={
                "frame_label": "computer · find_template",
                "summary": f"Found {len(matches)} match(es) for template",
                "matches": matches,
                "template_path": template_path,
            },
        )

    async def _start_recording(self) -> ToolResult:
        if self._recording:
            return ToolResult(success=False, error="Recording is already in progress.")
        self._recording = True
        self._recording_frames = []

        async def _capture_loop() -> None:
            interval = 1.0 / self._recording_fps
            while self._recording:
                try:
                    frame = self.capture_screenshot_bytes(fmt="png")
                    self._recording_frames.append(frame)
                except Exception:  # noqa: BLE001
                    pass
                await asyncio.sleep(interval)

        self._recording_task = asyncio.create_task(_capture_loop())
        return ToolResult(
            success=True,
            output=f"Recording started at {self._recording_fps} FPS.",
            metadata={"frame_label": "computer · start_recording", "summary": "Started screen recording"},
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
                "frame_label": "computer · stop_recording",
                "summary": "Stopped screen recording and saved GIF",
                "recording_path": str(gif_path),
                "frame_count": len(frames),
            },
        )

    @staticmethod
    def _extract_region(kwargs: dict[str, Any]) -> tuple[int, int, int, int] | None:
        rx = kwargs.get("region_x")
        ry = kwargs.get("region_y")
        rw = kwargs.get("region_width")
        rh = kwargs.get("region_height")
        if all(v is not None for v in (rx, ry, rw, rh)):
            return (int(rx), int(ry), int(rw), int(rh))
        return None

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
        if not text:
            return ToolResult(success=False, error="text is required for 'type' and cannot be empty.")
        if len(text) > 10000:
            return ToolResult(success=False, error="text too long (>10000 chars). Split into smaller chunks.")
        import pyautogui  # noqa: PLC0415

        pyautogui.typewrite(text, interval=0.02)
        return self._desktop_result(
            action="type",
            output=f"Typed: {text!r}",
            summary="Typing text on the desktop",
            capture_result_screenshot=capture_result_screenshot,
        )

    def _key(self, key: str, *, capture_result_screenshot: bool = True) -> ToolResult:
        valid, error = self._validate_key_sequence(key)
        if not valid:
            return ToolResult(success=False, error=f"Invalid key sequence {key!r}: {error}")
        import pyautogui  # noqa: PLC0415

        try:
            pyautogui.hotkey(*key.split("+"))
        except Exception as exc:  # noqa: BLE001
            return ToolResult(success=False, error=f"Failed to press {key!r}: {exc}")
        return self._desktop_result(
            action="key",
            output=f"Pressed key: {key!r}",
            summary=f"Pressing {key}" if key else "Pressing a key",
            capture_result_screenshot=capture_result_screenshot,
        )

    @staticmethod
    def _validate_key_sequence(key: str) -> tuple[bool, str | None]:
        """Validate a hotkey string like 'CTRL+S' or 'a' or 'F5'.

        Returns ``(True, None)`` if valid, ``(False, reason)`` otherwise.
        """
        if not key or not key.strip():
            return False, "key is empty"
        parts = key.split("+")
        if any(not p.strip() for p in parts):
            return False, "empty segment between '+'"
        for part in parts:
            normalized = part.strip().lower()
            if normalized in _VALID_KEY_NAMES:
                continue
            if len(normalized) == 1 and (normalized.isalnum() or not normalized.isalpha()):
                # Single char (letter, digit, or punctuation)
                continue
            return False, f"unknown key {part!r}"
        return True, None

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

    def capture_screenshot_bytes(
        self,
        fmt: str | None = None,
        quality: int | None = None,
        region: tuple[int, int, int, int] | None = None,
        monitor: int = 0,
    ) -> bytes:
        """Capture the screen and return image bytes in the requested format.

        *fmt* defaults to the configured ``screenshot_format`` (png/jpeg/webp).
        *quality* defaults to the configured ``screenshot_quality``.
        *region* is an optional ``(x, y, width, height)`` crop rectangle.
        *monitor* selects which display (0 = all combined, 1+ = individual).
        """
        fmt = (fmt or self._screenshot_format).lower()
        quality = quality if quality is not None else self._screenshot_quality

        try:
            import mss  # noqa: PLC0415

            with mss.mss() as sct:
                if region is not None:
                    rx, ry, rw, rh = region
                    grab_area = {"left": int(rx), "top": int(ry), "width": int(rw), "height": int(rh)}
                else:
                    if monitor < 0 or monitor >= len(sct.monitors):
                        raise ValueError(
                            f"Monitor index {monitor} out of range. "
                            f"Available: 0-{len(sct.monitors) - 1}"
                        )
                    grab_area = sct.monitors[monitor]
                grab = sct.grab(grab_area)
                if fmt == "png":
                    import mss.tools  # noqa: PLC0415
                    return mss.tools.to_png(grab.rgb, grab.size)
                from PIL import Image as _Image  # noqa: PLC0415
                pil_img = _Image.frombytes("RGB", grab.size, bytes(grab.rgb))
        except ImportError:
            import pyautogui  # noqa: PLC0415
            pil_img = pyautogui.screenshot(region=region)
            if fmt == "png":
                buf = io.BytesIO()
                pil_img.save(buf, format="PNG")
                return buf.getvalue()

        buf = io.BytesIO()
        save_fmt = "JPEG" if fmt == "jpeg" else fmt.upper()
        save_kwargs: dict[str, Any] = {}
        if fmt in ("jpeg", "webp"):
            save_kwargs["quality"] = quality
        pil_img.save(buf, format=save_fmt, **save_kwargs)
        return buf.getvalue()

    def list_monitors(self) -> list[dict[str, Any]]:
        """Return a list of available monitors with their dimensions."""
        try:
            import mss  # noqa: PLC0415
            with mss.mss() as sct:
                monitors = []
                for i, m in enumerate(sct.monitors):
                    monitors.append({
                        "index": i,
                        "left": m["left"],
                        "top": m["top"],
                        "width": m["width"],
                        "height": m["height"],
                        "label": "all combined" if i == 0 else f"monitor {i}",
                    })
                return monitors
        except ImportError:
            return [{"index": 0, "label": "all combined", "left": 0, "top": 0, "width": 0, "height": 0}]

    def capture_png_bytes(self) -> bytes:
        """Backward-compatible helper that always returns PNG bytes."""
        return self.capture_screenshot_bytes(fmt="png")

    def _capture_screenshot_b64(self) -> str:
        img_bytes = self.capture_screenshot_bytes()
        self._last_screenshot_raw = img_bytes
        return base64.b64encode(img_bytes).decode()

    def _check_screenshot_change(self, png_bytes: bytes) -> bool:
        """Compare *png_bytes* against the previous screenshot hash.

        Returns ``True`` when the image changed (or this is the first capture).
        Updates the stored hash as a side-effect.
        """
        current_hash = compute_image_hash(png_bytes)
        no_change = images_are_similar(self._last_screenshot_hash, current_hash)
        self._last_screenshot_hash = current_hash
        return not no_change

    def _desktop_result(
        self,
        *,
        action: str,
        output: str,
        summary: str,
        capture_result_screenshot: bool = True,
    ) -> ToolResult:
        metadata: dict[str, Any] = {
            "frame_label": f"computer · {action}",
            "summary": summary,
        }
        if not capture_result_screenshot:
            return ToolResult(success=True, output=output, metadata=metadata)
        screenshot_b64 = self._capture_screenshot_b64()
        changed = self._check_screenshot_change(self._last_screenshot_raw)
        metadata["no_change"] = not changed
        return ToolResult(
            success=True,
            output=output,
            screenshot_b64=screenshot_b64,
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
