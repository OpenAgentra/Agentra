"""Thread-scoped visible and hidden desktop session management."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import ctypes
import io
import math
import os
import queue
import re
import struct
import subprocess
import threading
import time
import uuid
import zlib
from dataclasses import dataclass
from typing import Any

from agentra.config import AgentConfig
from agentra.desktop_automation.models import DesktopActionVerification, DesktopLiveFrame, DesktopSessionSnapshot, WindowInfo
from agentra.desktop_automation.windows_native import NativeWindowsDesktopBackend
from agentra.tools.base import ToolResult
from agentra.windows_apps import get_windows_app_profile, guess_windows_app_profile

_USER32 = getattr(ctypes, "windll", None)
_POINTER_SIZED = ctypes.c_size_t
_WM_LBUTTONDOWN = 0x0201
_WM_LBUTTONUP = 0x0202
_WM_LBUTTONDBLCLK = 0x0203
_WM_RBUTTONDOWN = 0x0204
_WM_RBUTTONUP = 0x0205
_WM_RBUTTONDBLCLK = 0x0206
_WM_MOUSEMOVE = 0x0200
_WM_MOUSEWHEEL = 0x020A
_WM_KEYDOWN = 0x0100
_WM_KEYUP = 0x0101
_WM_CHAR = 0x0102
_PW_CLIENTONLY = 0x00000001
_VK_SHIFT = 0x10
_VK_CONTROL = 0x11
_VK_MENU = 0x12
_VK_RETURN = 0x0D
_VK_END = 0x23
_VK_HOME = 0x24
_VK_LEFT = 0x25
_VK_UP = 0x26
_VK_RIGHT = 0x27
_VK_DOWN = 0x28
_VK_TAB = 0x09
_VK_BACK = 0x08
_VK_DELETE = 0x2E
_VK_INSERT = 0x2D
_VK_ESCAPE = 0x1B
_VK_PRIOR = 0x21
_VK_NEXT = 0x22
_VK_SPACE = 0x20
_MK_LBUTTON = 0x0001
_MK_RBUTTON = 0x0002
_HIDDEN_SESSION_PREFIX = "desktop_session:"
_HIDDEN_DIALOG_CLASS_NAMES = frozenset({"#32770", "Credential Dialog Xaml Host", "OperationStatusWindow"})
_RISKY_WINDOWS_ACTIONS = frozenset({"invoke_control", "set_text", "type_keys"})
_RISKY_COMPUTER_ACTIONS = frozenset({"click", "double_click", "right_click", "move", "type", "key", "scroll", "drag"})


def _make_lparam(x: int, y: int) -> int:
    return ((int(y) & 0xFFFF) << 16) | (int(x) & 0xFFFF)


def _png_chunk(chunk_type: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + chunk_type
        + payload
        + struct.pack(">I", zlib.crc32(chunk_type + payload) & 0xFFFFFFFF)
    )


def _png_from_bgra(width: int, height: int, pixels: bytes) -> bytes:
    if width <= 0 or height <= 0:
        return b""
    stride = width * 4
    rows: list[bytes] = []
    for row_index in range(height):
        source_index = (height - 1 - row_index) * stride
        row = pixels[source_index : source_index + stride]
        rgb = bytearray()
        for index in range(0, len(row), 4):
            blue, green, red, _alpha = row[index : index + 4]
            rgb.extend((red, green, blue))
        rows.append(b"\x00" + bytes(rgb))
    raw = b"".join(rows)
    header = b"\x89PNG\r\n\x1a\n"
    ihdr = _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    idat = _png_chunk(b"IDAT", zlib.compress(raw, level=6))
    iend = _png_chunk(b"IEND", b"")
    return header + ihdr + idat + iend


def _visible_desktop_png_bytes() -> bytes:
    try:
        import mss  # noqa: PLC0415
        import mss.tools  # noqa: PLC0415

        with mss.mss() as sct:
            monitor = sct.monitors[0]
            img = sct.grab(monitor)
            return mss.tools.to_png(img.rgb, img.size)
    except ImportError:
        import pyautogui  # noqa: PLC0415

        buf = io.BytesIO()
        pyautogui.screenshot().save(buf, format="PNG")
        return buf.getvalue()


def _normalized_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").casefold()).strip()


@dataclass
class _CaptureAttempt:
    data: bytes | None
    backend: str
    compatibility_state: str
    fallback_reason: str = ""
    media_type: str = "image/png"


class _BaseCaptureAdapter:
    name = "unsupported"

    def capture(self, handle: int) -> _CaptureAttempt:
        return _CaptureAttempt(
            data=None,
            backend=self.name,
            compatibility_state="fallback_required",
            fallback_reason="No capture backend is available for this window.",
        )


class _WindowsGraphicsCaptureAdapter(_BaseCaptureAdapter):
    name = "windows_graphics_capture"

    def capture(self, handle: int) -> _CaptureAttempt:
        del handle
        return _CaptureAttempt(
            data=None,
            backend=self.name,
            compatibility_state="fallback_required",
            fallback_reason="Windows Graphics Capture adapter is not configured in this build.",
        )


class _PrintWindowCaptureAdapter(_BaseCaptureAdapter):
    name = "printwindow_client"

    def capture(self, handle: int) -> _CaptureAttempt:
        if os.name != "nt":
            return _CaptureAttempt(
                data=None,
                backend=self.name,
                compatibility_state="fallback_required",
                fallback_reason="PrintWindow capture is only available on Windows.",
            )
        try:
            import win32con  # noqa: PLC0415
            import win32gui  # noqa: PLC0415
            import win32ui  # noqa: PLC0415
        except ImportError as exc:
            return _CaptureAttempt(
                data=None,
                backend=self.name,
                compatibility_state="fallback_required",
                fallback_reason=f"PrintWindow capture requires pywin32: {exc}",
            )

        client_rect = win32gui.GetClientRect(handle)
        width = max(0, int(client_rect[2] - client_rect[0]))
        height = max(0, int(client_rect[3] - client_rect[1]))
        if width <= 0 or height <= 0:
            return _CaptureAttempt(
                data=None,
                backend=self.name,
                compatibility_state="fallback_required",
                fallback_reason="Target window does not expose a client area.",
            )

        hwnd_dc = save_dc = mem_dc = bitmap = None
        try:
            hwnd_dc = win32gui.GetWindowDC(handle)
            save_dc = win32ui.CreateDCFromHandle(hwnd_dc)
            mem_dc = save_dc.CreateCompatibleDC()
            bitmap = win32ui.CreateBitmap()
            bitmap.CreateCompatibleBitmap(save_dc, width, height)
            mem_dc.SelectObject(bitmap)
            ok = int(ctypes.windll.user32.PrintWindow(int(handle), mem_dc.GetSafeHdc(), _PW_CLIENTONLY))
            if not ok:
                return _CaptureAttempt(
                    data=None,
                    backend=self.name,
                    compatibility_state="fallback_required",
                    fallback_reason="PrintWindow could not capture the target client area.",
                )
            pixels = bitmap.GetBitmapBits(True)
            if not pixels or not any(pixels):
                return _CaptureAttempt(
                    data=None,
                    backend=self.name,
                    compatibility_state="fallback_required",
                    fallback_reason="Captured frame was blank, likely due to an unsupported GPU surface.",
                )
            data = _png_from_bgra(width, height, pixels)
            if not data:
                return _CaptureAttempt(
                    data=None,
                    backend=self.name,
                    compatibility_state="fallback_required",
                    fallback_reason="PrintWindow capture returned an empty frame.",
                )
            return _CaptureAttempt(
                data=data,
                backend=self.name,
                compatibility_state="background_capable",
            )
        except Exception as exc:  # noqa: BLE001
            return _CaptureAttempt(
                data=None,
                backend=self.name,
                compatibility_state="fallback_required",
                fallback_reason=str(exc),
            )
        finally:
            with contextlib.suppress(Exception):
                if bitmap is not None:
                    win32gui.DeleteObject(bitmap.GetHandle())
            with contextlib.suppress(Exception):
                if mem_dc is not None:
                    mem_dc.DeleteDC()
            with contextlib.suppress(Exception):
                if save_dc is not None:
                    save_dc.DeleteDC()
            with contextlib.suppress(Exception):
                if hwnd_dc is not None:
                    win32gui.ReleaseDC(handle, hwnd_dc)


class _WindowMessageInputAdapter:
    """Route preview input to a target window without using the visible desktop."""

    _SPECIAL_KEYS = {
        "ALT": _VK_MENU,
        "BACK": _VK_BACK,
        "BACKSPACE": _VK_BACK,
        "CTRL": _VK_CONTROL,
        "CONTROL": _VK_CONTROL,
        "DEL": _VK_DELETE,
        "DELETE": _VK_DELETE,
        "DOWN": _VK_DOWN,
        "END": _VK_END,
        "ENTER": _VK_RETURN,
        "ESC": _VK_ESCAPE,
        "ESCAPE": _VK_ESCAPE,
        "HOME": _VK_HOME,
        "INS": _VK_INSERT,
        "INSERT": _VK_INSERT,
        "LEFT": _VK_LEFT,
        "PAGEDOWN": _VK_NEXT,
        "PAGEUP": _VK_PRIOR,
        "PGDN": _VK_NEXT,
        "PGUP": _VK_PRIOR,
        "RIGHT": _VK_RIGHT,
        "SHIFT": _VK_SHIFT,
        "SPACE": _VK_SPACE,
        "TAB": _VK_TAB,
        "UP": _VK_UP,
    }

    def __init__(self, worker: "_Win32HiddenDesktopWorker") -> None:
        self._worker = worker

    def click(self, handle: int, x: int, y: int, *, button: str = "left", double: bool = False) -> None:
        user32 = ctypes.windll.user32
        target = self._target_window_for_point(handle, x, y)
        message_down = _WM_LBUTTONDOWN if button == "left" else _WM_RBUTTONDOWN
        message_up = _WM_LBUTTONUP if button == "left" else _WM_RBUTTONUP
        message_double = _WM_LBUTTONDBLCLK if button == "left" else _WM_RBUTTONDBLCLK
        flag = _MK_LBUTTON if button == "left" else _MK_RBUTTON
        lparam = _make_lparam(x, y)
        user32.PostMessageW(target, message_down, flag, lparam)
        user32.PostMessageW(target, message_up, 0, lparam)
        if double:
            user32.PostMessageW(target, message_double, flag, lparam)
            user32.PostMessageW(target, message_up, 0, lparam)

    def drag(self, handle: int, start_x: int, start_y: int, end_x: int, end_y: int) -> None:
        user32 = ctypes.windll.user32
        target = self._target_window_for_point(handle, start_x, start_y)
        user32.PostMessageW(target, _WM_LBUTTONDOWN, _MK_LBUTTON, _make_lparam(start_x, start_y))
        steps = max(6, int(math.hypot(end_x - start_x, end_y - start_y) / 18))
        for step in range(1, steps + 1):
            ratio = step / steps
            next_x = round(start_x + (end_x - start_x) * ratio)
            next_y = round(start_y + (end_y - start_y) * ratio)
            user32.PostMessageW(target, _WM_MOUSEMOVE, _MK_LBUTTON, _make_lparam(next_x, next_y))
            time.sleep(0.01)
        user32.PostMessageW(target, _WM_LBUTTONUP, 0, _make_lparam(end_x, end_y))

    def scroll(self, handle: int, x: int, y: int, amount: int) -> None:
        user32 = ctypes.windll.user32
        target = self._target_window_for_point(handle, x, y)
        wheel_delta = int(amount) << 16
        user32.PostMessageW(target, _WM_MOUSEWHEEL, wheel_delta, _make_lparam(x, y))

    def type_text(self, handle: int, text: str) -> None:
        target = self._keyboard_target(handle)
        user32 = ctypes.windll.user32
        for char in str(text or ""):
            user32.PostMessageW(target, _WM_CHAR, ord(char), 1)
            time.sleep(0.005)

    def key_sequence(self, handle: int, text: str) -> None:
        target = self._keyboard_target(handle)
        for entry in self._parse_key_sequence(text):
            kind = str(entry["kind"])
            if kind == "char":
                ctypes.windll.user32.PostMessageW(target, _WM_CHAR, int(entry["char"]), 1)
                time.sleep(0.005)
                continue
            modifiers = list(entry.get("modifiers", []))
            key_vk = int(entry["vk"])
            for modifier in modifiers:
                ctypes.windll.user32.PostMessageW(target, _WM_KEYDOWN, modifier, 1)
            ctypes.windll.user32.PostMessageW(target, _WM_KEYDOWN, key_vk, 1)
            ctypes.windll.user32.PostMessageW(target, _WM_KEYUP, key_vk, 1)
            for modifier in reversed(modifiers):
                ctypes.windll.user32.PostMessageW(target, _WM_KEYUP, modifier, 1)
            time.sleep(0.01)

    def _target_window_for_point(self, handle: int, x: int, y: int) -> int:
        user32 = ctypes.windll.user32
        target = int(handle)
        point = POINT(int(x), int(y))
        candidate = int(user32.ChildWindowFromPointEx(target, point, 0) or 0)
        if candidate:
            target = candidate
        return target

    def _keyboard_target(self, handle: int) -> int:
        focused = self._focused_control(handle)
        return focused or int(handle)

    def _focused_control(self, handle: int) -> int:
        user32 = ctypes.windll.user32
        thread_id = user32.GetWindowThreadProcessId(int(handle), None)
        if not thread_id:
            return 0
        gui = GUITHREADINFO()
        gui.cbSize = ctypes.sizeof(GUITHREADINFO)
        if not user32.GetGUIThreadInfo(int(thread_id), ctypes.byref(gui)):
            return 0
        return int(gui.hwndFocus or 0)

    def _parse_key_sequence(self, text: str) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        raw = str(text or "")
        index = 0
        while index < len(raw):
            if raw[index] != "{":
                events.append({"kind": "char", "char": ord(raw[index])})
                index += 1
                continue
            closing = raw.find("}", index + 1)
            if closing == -1:
                events.append({"kind": "char", "char": ord(raw[index])})
                index += 1
                continue
            token = raw[index + 1 : closing].strip()
            index = closing + 1
            if not token:
                continue
            if "+" in token:
                parts = [part.strip() for part in token.split("+") if part.strip()]
                if len(parts) >= 2:
                    modifiers = [self._SPECIAL_KEYS.get(part.upper()) for part in parts[:-1]]
                    key_vk = self._key_to_vk(parts[-1])
                    if key_vk and all(modifiers):
                        events.append({"kind": "key", "vk": key_vk, "modifiers": modifiers})
                        continue
            key_vk = self._key_to_vk(token)
            if key_vk:
                events.append({"kind": "key", "vk": key_vk, "modifiers": []})
                continue
            if len(token) == 1:
                events.append({"kind": "char", "char": ord(token)})
        return events

    def _key_to_vk(self, token: str) -> int:
        normalized = str(token or "").strip()
        if not normalized:
            return 0
        if len(normalized) == 1:
            character = normalized.upper()
            if "A" <= character <= "Z" or "0" <= character <= "9":
                return ord(character)
            if character == " ":
                return _VK_SPACE
        return int(self._SPECIAL_KEYS.get(normalized.upper(), 0) or 0)


class _HiddenDesktopBackend(NativeWindowsDesktopBackend):
    """Structured Windows backend that operates inside a hidden worker desktop."""

    backend_id = "hidden_windows"
    desktop_execution_mode = "desktop_hidden"

    def __init__(self, worker: "_Win32HiddenDesktopWorker", *, default_timeout: float = 6.0) -> None:
        super().__init__(default_timeout=default_timeout)
        self._worker = worker

    def execute(self, action: str, **kwargs: Any) -> DesktopActionVerification:
        normalized_action = str(action or "").strip().lower()
        if normalized_action in _RISKY_WINDOWS_ACTIONS:
            takeover = self._worker.guard_interaction(
                action=normalized_action,
                window_title=str(kwargs.get("window_title", "") or ""),
                app=str(kwargs.get("app", "") or ""),
                profile_id=str(kwargs.get("profile_id", "") or ""),
                require_background_preview=True,
            )
            if takeover is not None:
                return DesktopActionVerification(
                    target=str(takeover.get("target") or kwargs.get("window_title") or kwargs.get("app") or "hidden desktop"),
                    action=normalized_action,
                    observed_outcome=str(takeover.get("content") or takeover.get("fallback_reason") or "Hidden desktop action paused."),
                    success=False,
                    verified=False,
                    fallback_reason=str(takeover.get("fallback_reason") or takeover.get("content") or "Hidden desktop action paused."),
                    details=takeover,
                )
        result = super().execute(action, **kwargs)
        if result.success:
            self._worker.session_status = "ready"
        return result

    def launch_app(
        self,
        *,
        app: str | None = None,
        profile_id: str | None = None,
        timeout_sec: float | None = None,
        **_: Any,
    ) -> DesktopActionVerification:
        requested_app = str(app or "").strip()
        if not requested_app and profile_id:
            profile = get_windows_app_profile(profile_id)
            if profile is not None:
                requested_app = profile.executable
        if not requested_app:
            return self._failure("launch_app", "desktop", "app is required")
        command = requested_app.strip()
        try:
            self._worker.launch_process(command)
        except Exception as exc:  # noqa: BLE001
            return self._failure("launch_app", command, str(exc))
        inferred_profile = profile_id or guess_windows_app_profile(requested_app)
        wait_result = self.wait_for_window(
            app=requested_app,
            profile_id=inferred_profile,
            timeout_sec=timeout_sec,
        )
        if wait_result.success:
            self._worker.note_active_window(
                profile_id=inferred_profile or "",
                app=requested_app,
                window_title=str(wait_result.details.get("window_title", "") or requested_app),
                window_handle=int(wait_result.details.get("window_handle", 0) or 0),
            )
            self._worker.probe_capture_backend()
        details = dict(wait_result.details)
        details["requested_app"] = requested_app
        details["app"] = requested_app
        return DesktopActionVerification(
            target=requested_app,
            action="launch_app",
            observed_outcome=wait_result.observed_outcome or f"Launched {requested_app}.",
            success=wait_result.success,
            verified=wait_result.verified,
            fallback_reason=wait_result.fallback_reason,
            details=details,
        )

    def focus_window(
        self,
        *,
        window_title: str | None = None,
        app: str | None = None,
        profile_id: str | None = None,
        **_: Any,
    ) -> DesktopActionVerification:
        window = self._resolve_window(window_title=window_title, app=app, profile_id=profile_id)
        if window is None:
            return self._failure("focus_window", window_title or app or "window", "Window not found.")
        self._remember_window(window, profile_id=profile_id, app=app, window_title=window_title)
        self._worker.note_active_window(
            profile_id=profile_id or guess_windows_app_profile(app or window_title or "") or "",
            app=app or "",
            window_title=window.title,
            window_handle=window.handle,
        )
        self._worker.probe_capture_backend()
        return DesktopActionVerification(
            target=window.title or str(window.handle),
            action="focus_window",
            observed_outcome=f"Focused hidden session target {window.title or 'window'}.",
            success=True,
            verified=True,
            details={"window_title": window.title, "window_handle": window.handle},
        )

    def _resolve_window(
        self,
        *,
        window_title: str | None = None,
        app: str | None = None,
        profile_id: str | None = None,
    ) -> WindowInfo | None:
        window = super()._resolve_window(window_title=window_title, app=app, profile_id=profile_id)
        if window is not None:
            return window
        if window_title or app or profile_id:
            return None
        return self._worker.current_target_window()

    def type_keys(
        self,
        *,
        text: str,
        window_title: str | None = None,
        app: str | None = None,
        profile_id: str | None = None,
        **_: Any,
    ) -> DesktopActionVerification:
        if not str(text or ""):
            return self._failure("type_keys", window_title or app or "window", "text is required")
        resolved_profile = profile_id or guess_windows_app_profile(app or window_title or "")
        window = self._resolve_window(window_title=window_title, app=app, profile_id=resolved_profile)
        if window is None:
            return self._failure("type_keys", window_title or app or "window", "Window not found.")
        self._worker.note_active_window(
            profile_id=resolved_profile or "",
            app=app or "",
            window_title=window.title,
            window_handle=window.handle,
        )
        if resolved_profile == "calculator":
            result = self._type_into_calculator(window, text=text)
            self._worker.probe_capture_backend()
            return result
        before = self.read_window_text(window_title=window.title, profile_id=resolved_profile).observed_outcome
        self._worker.input_adapter.key_sequence(window.handle, text)
        time.sleep(0.1)
        after = self.read_window_text(window_title=window.title, profile_id=resolved_profile)
        verified = after.success and after.observed_outcome != before
        self._worker.probe_capture_backend()
        return DesktopActionVerification(
            target=window.title,
            action="type_keys",
            observed_outcome=f"Typed keys into {window.title}.",
            success=True,
            verified=verified,
            fallback_reason="" if verified else "Keystroke effect could not be verified from the hidden session window text.",
            details={
                "pre_state": before,
                "post_state": after.observed_outcome,
                "window_title": window.title,
                "window_handle": window.handle,
                "requested_text": text,
            },
        )

    def _enumerate_windows(self) -> list[WindowInfo]:
        if os.name != "nt" or not self._worker.desktop_handle:
            return []
        try:
            import win32gui  # noqa: PLC0415
            import win32process  # noqa: PLC0415
        except ImportError:
            return []

        windows: list[WindowInfo] = []
        callback = ctypes.WINFUNCTYPE(ctypes.c_bool, _POINTER_SIZED, _POINTER_SIZED)

        @callback
        def _enumerate(handle: int, _extra: int) -> bool:
            try:
                if not win32gui.IsWindowVisible(handle):
                    return True
                title = str(win32gui.GetWindowText(handle) or "").strip()
                if not title:
                    return True
                class_name = str(win32gui.GetClassName(handle) or "").strip()
                try:
                    _thread_id, process_id = win32process.GetWindowThreadProcessId(handle)
                except Exception:  # noqa: BLE001
                    process_id = 0
                windows.append(
                    WindowInfo(
                        handle=int(handle),
                        title=title,
                        class_name=class_name,
                        process_id=int(process_id),
                        visible=True,
                    )
                )
            except Exception:  # noqa: BLE001
                return True
            return True

        ctypes.windll.user32.EnumDesktopWindows(self._worker.desktop_handle, _enumerate, 0)
        return windows

    def _foreground_window_handle(self) -> int:  # type: ignore[override]
        return int(self._worker.active_window_handle or 0)


class _Win32HiddenDesktopWorker:
    """Single-threaded Win32 hidden desktop worker."""

    def __init__(self, thread_id: str) -> None:
        self.thread_id = thread_id
        self.session_id = f"hidden-{thread_id}-{uuid.uuid4().hex[:8]}"
        self.desktop_name = f"AgentraHidden-{uuid.uuid4().hex[:12]}"
        self.desktop_handle = 0
        self.active_window_handle = 0
        self.active_window_title = ""
        self.active_target_app = ""
        self.active_profile_id = ""
        self.capture_backend = ""
        self.compatibility_state = "unknown"
        self.fallback_reason = ""
        self.session_status = "initializing"
        self.desktop_backend = "hidden_windows"
        self._backend: _HiddenDesktopBackend | None = None
        self.input_adapter = _WindowMessageInputAdapter(self)
        self._requests: queue.Queue[tuple[str, dict[str, Any], queue.Queue[Any]]] = queue.Queue()
        self._ready = threading.Event()
        self._closed = threading.Event()
        self._thread = threading.Thread(target=self._thread_main, name=f"agentra-hidden-{thread_id}", daemon=True)

    def start(self, *, timeout_sec: float = 5.0) -> None:
        self._thread.start()
        self._ready.wait(timeout=timeout_sec)

    def close(self) -> None:
        if self._closed.is_set():
            return
        reply: queue.Queue[Any] = queue.Queue(maxsize=1)
        self._requests.put(("close", {}, reply))
        with contextlib.suppress(Exception):
            reply.get(timeout=2.0)
        self._thread.join(timeout=2.0)
        self._closed.set()

    def call(self, method: str, **payload: Any) -> Any:
        if not self._ready.is_set():
            self.start()
        reply: queue.Queue[Any] = queue.Queue(maxsize=1)
        self._requests.put((method, payload, reply))
        result = reply.get(timeout=15.0)
        if isinstance(result, Exception):
            raise result
        return result

    def launch_process(self, command: str) -> None:
        startupinfo = subprocess.STARTUPINFO()
        desktop_value = self.desktop_name
        with contextlib.suppress(Exception):
            startupinfo.lpDesktop = desktop_value
        try:
            subprocess.Popen([command], startupinfo=startupinfo)  # noqa: S603
            return
        except Exception:
            startupinfo = subprocess.STARTUPINFO()
            with contextlib.suppress(Exception):
                startupinfo.lpDesktop = f"WinSta0\\{desktop_value}"
            subprocess.Popen([command], startupinfo=startupinfo)  # noqa: S603

    def note_active_window(
        self,
        *,
        profile_id: str,
        app: str,
        window_title: str,
        window_handle: int,
    ) -> None:
        self.active_profile_id = str(profile_id or "")
        self.active_target_app = str(app or self.active_target_app or "")
        self.active_window_title = str(window_title or "")
        self.active_window_handle = int(window_handle or 0)

    def current_target_window(self) -> WindowInfo | None:
        return self._resolve_target_window(
            window_title=self.active_window_title,
            app=self.active_target_app,
            profile_id=self.active_profile_id,
        )

    def guard_interaction(
        self,
        *,
        action: str,
        window_title: str = "",
        app: str = "",
        profile_id: str = "",
        require_background_preview: bool,
    ) -> dict[str, Any] | None:
        windows = self._enumerated_windows()
        if not windows:
            self.session_status = "blocked_waiting_user"
            self.compatibility_state = "preview_only"
            self.fallback_reason = "No active hidden desktop window is available yet."
            return self._desktop_takeover_payload(
                action=action,
                reason=self.fallback_reason,
                summary="Arka plan masaustu hedefi hazir olmadigi icin kontrol size devredildi.",
            )

        has_target_hint = any(
            str(value or "").strip()
            for value in (window_title, app, profile_id, self.active_window_title, self.active_target_app, self.active_profile_id)
        )
        if not has_target_hint and len(windows) > 1:
            self.session_status = "blocked_waiting_user"
            self.compatibility_state = "preview_only"
            self.fallback_reason = "Multiple hidden desktop windows are open and no target window has been pinned yet."
            return self._desktop_takeover_payload(
                action=action,
                reason=(
                    "Arka plan oturumunda birden fazla pencere acik ve hangi pencerenin kontrol edilecegi belli degil. "
                    "Dogru pencereyi one alin veya manuel secimi tamamlayin; sonra Finish Control ile devam edin."
                ),
                summary="Arka plan masaustu hedefi secilmedigi icin kontrol size devredildi.",
                windows=windows,
            )

        expected_window = self._resolve_target_window(window_title=window_title, app=app, profile_id=profile_id)
        if expected_window is None and len(windows) > 1:
            self.session_status = "blocked_waiting_user"
            self.compatibility_state = "preview_only"
            self.fallback_reason = "Multiple hidden desktop windows are open and the target is ambiguous."
            return self._desktop_takeover_payload(
                action=action,
                reason=(
                    "Arka plan oturumunda birden fazla pencere acik ve hedef pencere net degil. "
                    "Dogru pencereyi secin veya manuel duzeltin; sonra Finish Control ile devam edin."
                ),
                summary="Arka plan masaustu hedefi belirsiz kaldigi icin kontrol size devredildi.",
                windows=windows,
            )
        if expected_window is None:
            expected_window = windows[0]

        self.note_active_window(
            profile_id=profile_id or guess_windows_app_profile(app or window_title or expected_window.title) or "",
            app=app or self.active_target_app,
            window_title=expected_window.title,
            window_handle=expected_window.handle,
        )

        unexpected_dialog = self._unexpected_dialog_window(
            windows=windows,
            window_title=window_title,
            app=app,
            profile_id=profile_id,
        )
        if unexpected_dialog is not None:
            self.session_status = "blocked_waiting_user"
            self.compatibility_state = "fallback_required"
            self.fallback_reason = (
                f"Unexpected hidden desktop dialog detected: {unexpected_dialog.title or unexpected_dialog.class_name}."
            )
            return self._desktop_takeover_payload(
                action=action,
                reason=(
                    "Arka plan oturumunda beklenmeyen bir pencere veya popup acildi. "
                    "Durumu inceleyip gerekirse manuel tamamlayin; sonra Finish Control ile devam edin."
                ),
                summary="Beklenmeyen arka plan popup'i tespit edildigi icin kontrol size devredildi.",
                window=unexpected_dialog,
                windows=windows,
            )

        if require_background_preview:
            attempt = self._capture_window(expected_window.handle)
            self.capture_backend = attempt.backend
            self.compatibility_state = attempt.compatibility_state
            self.fallback_reason = attempt.fallback_reason
            if attempt.data is None:
                self.session_status = "blocked_waiting_user"
                return self._desktop_takeover_payload(
                    action=action,
                    reason=(
                        "Bu pencere arka planda guvenli onizleme ve denetim icin uyumlu degil. "
                        "Gorunur kontrol veya manuel yardim gerektiriyor."
                    ),
                    summary="Arka plan denetimi uyumsuz kaldigi icin kontrol size devredildi.",
                    window=expected_window,
                    windows=windows,
                )

        self.session_status = "ready"
        return None

    def probe_capture_backend(self) -> None:
        attempt = self._capture_current_window()
        self.capture_backend = attempt.backend
        self.compatibility_state = attempt.compatibility_state
        self.fallback_reason = attempt.fallback_reason

    def snapshot(self) -> DesktopSessionSnapshot:
        return DesktopSessionSnapshot(
            mode="desktop_hidden",
            session_status=self.session_status,
            active_target_app=self.active_target_app,
            active_target_window=self.active_window_title,
            capture_backend=self.capture_backend,
            compatibility_state=self.compatibility_state,
            fallback_reason=self.fallback_reason,
            session_id=self.session_id,
            desktop_backend=self.desktop_backend,
            active_window_handle=self.active_window_handle,
        )

    def _thread_main(self) -> None:
        try:
            self._initialize_hidden_desktop()
            self._backend = _HiddenDesktopBackend(self)
            self.session_status = "ready"
        except Exception as exc:  # noqa: BLE001
            self.session_status = "error"
            self.compatibility_state = "fallback_required"
            self.fallback_reason = str(exc)
        finally:
            self._ready.set()

        while True:
            method, payload, reply = self._requests.get()
            try:
                if method == "close":
                    reply.put(True)
                    return
                if self._backend is None:
                    raise RuntimeError(self.fallback_reason or "Hidden desktop backend is unavailable.")
                if method == "windows":
                    result = self._backend.execute(str(payload["action"]), **dict(payload.get("kwargs", {})))
                    reply.put(result)
                    continue
                if method == "computer":
                    result = self._execute_computer_action(str(payload["action"]), **dict(payload.get("kwargs", {})))
                    reply.put(result)
                    continue
                if method == "capture":
                    attempt = self._capture_current_window()
                    self.capture_backend = attempt.backend
                    self.compatibility_state = attempt.compatibility_state
                    self.fallback_reason = attempt.fallback_reason
                    if attempt.data is None:
                        reply.put(None)
                    else:
                        reply.put(DesktopLiveFrame(data=attempt.data, media_type=attempt.media_type))
                    continue
                if method == "snapshot":
                    reply.put(self.snapshot())
                    continue
                raise RuntimeError(f"Unknown hidden desktop worker method: {method}")
            except Exception as exc:  # noqa: BLE001
                self.session_status = "error"
                self.compatibility_state = "fallback_required"
                self.fallback_reason = str(exc)
                reply.put(exc)
        # unreachable

    def _initialize_hidden_desktop(self) -> None:
        if os.name != "nt":
            raise RuntimeError("Hidden desktop sessions are only available on Windows.")
        access = 0x0001 | 0x0002 | 0x0040 | 0x0080 | 0x0100
        handle = int(ctypes.windll.user32.CreateDesktopW(self.desktop_name, None, None, 0, access, None) or 0)
        if not handle:
            raise RuntimeError("CreateDesktopW failed for the hidden desktop worker.")
        self.desktop_handle = handle
        if not ctypes.windll.user32.SetThreadDesktop(handle):
            raise RuntimeError("SetThreadDesktop failed for the hidden desktop worker.")

    def _capture_window(self, handle: int) -> _CaptureAttempt:
        if not handle:
            return _CaptureAttempt(
                data=None,
                backend="none",
                compatibility_state="preview_only",
                fallback_reason="No active hidden desktop window is available yet.",
            )
        for adapter in (_WindowsGraphicsCaptureAdapter(), _PrintWindowCaptureAdapter()):
            attempt = adapter.capture(handle)
            if attempt.data is not None:
                return attempt
        return _CaptureAttempt(
            data=None,
            backend="unsupported",
            compatibility_state="fallback_required",
            fallback_reason="No compatible capture backend succeeded for the hidden desktop window.",
        )

    def _capture_current_window(self) -> _CaptureAttempt:
        return self._capture_window(self._current_window_handle())

    def _enumerated_windows(self) -> list[WindowInfo]:
        if self._backend is None:
            return []
        return self._backend._enumerate_windows()  # noqa: SLF001

    def _resolve_target_window(
        self,
        *,
        window_title: str | None = None,
        app: str | None = None,
        profile_id: str | None = None,
    ) -> WindowInfo | None:
        windows = self._enumerated_windows()
        if not windows:
            return None

        normalized_title = _normalized_text(window_title or "")
        normalized_app = _normalized_text(str(app or "").replace(".exe", ""))
        resolved_profile_id = str(profile_id or self.active_profile_id or guess_windows_app_profile(app or window_title or "") or "").strip()
        profile = get_windows_app_profile(resolved_profile_id)
        title_tokens = [token for token in (normalized_title, normalized_app) if token]
        if profile is not None:
            title_tokens.extend(_normalized_text(token) for token in profile.window_title_tokens)
        if self.active_window_title:
            title_tokens.append(_normalized_text(self.active_window_title))

        active_candidate = next((item for item in windows if item.handle == self.active_window_handle), None)
        if title_tokens:
            scored = sorted(
                windows,
                key=lambda item: self._window_match_score(item, title_tokens=title_tokens, active_handle=self.active_window_handle),
                reverse=True,
            )
            if scored and (
                self._window_matches_tokens(scored[0], title_tokens)
                or (active_candidate is not None and scored[0].handle == active_candidate.handle)
            ):
                return scored[0]
        if active_candidate is not None:
            return active_candidate
        if len(windows) == 1:
            return windows[0]
        scored = sorted(
            windows,
            key=lambda item: self._window_match_score(item, title_tokens=(), active_handle=self.active_window_handle),
            reverse=True,
        )
        return scored[0] if scored else None

    def _window_matches_tokens(self, window: WindowInfo, title_tokens: tuple[str, ...] | list[str]) -> bool:
        normalized_window_title = _normalized_text(window.title)
        return any(token and token in normalized_window_title for token in title_tokens)

    def _window_match_score(
        self,
        window: WindowInfo,
        *,
        title_tokens: tuple[str, ...] | list[str],
        active_handle: int,
    ) -> int:
        normalized_window_title = _normalized_text(window.title)
        score = 0
        if active_handle and window.handle == active_handle:
            score += 80
        if window.class_name not in _HIDDEN_DIALOG_CLASS_NAMES:
            score += 20
        if self.active_window_title and _normalized_text(self.active_window_title) == normalized_window_title:
            score += 60
        for token in title_tokens:
            if token and token in normalized_window_title:
                score += 40
        return score

    def _unexpected_dialog_window(
        self,
        *,
        windows: list[WindowInfo],
        window_title: str,
        app: str,
        profile_id: str,
    ) -> WindowInfo | None:
        if len(windows) <= 1:
            return None
        for window in windows:
            if window.class_name not in _HIDDEN_DIALOG_CLASS_NAMES:
                continue
            if self._window_matches_expected(window, window_title=window_title, app=app, profile_id=profile_id):
                continue
            return window
        return None

    def _window_matches_expected(
        self,
        window: WindowInfo,
        *,
        window_title: str,
        app: str,
        profile_id: str,
    ) -> bool:
        normalized_window_title = _normalized_text(window.title)
        resolved_profile_id = str(profile_id or self.active_profile_id or guess_windows_app_profile(app or window_title or "") or "").strip()
        profile = get_windows_app_profile(resolved_profile_id)
        tokens = [_normalized_text(window_title or ""), _normalized_text(str(app or "").replace(".exe", ""))]
        if profile is not None:
            tokens.extend(_normalized_text(token) for token in profile.window_title_tokens)
        if self.active_window_title:
            tokens.append(_normalized_text(self.active_window_title))
        return any(token and token in normalized_window_title for token in tokens)

    def _desktop_takeover_payload(
        self,
        *,
        action: str,
        reason: str,
        summary: str,
        window: WindowInfo | None = None,
        windows: list[WindowInfo] | None = None,
    ) -> dict[str, Any]:
        observation = {
            "window_title": window.title if window is not None else self.active_window_title,
            "window_handle": window.handle if window is not None else self.active_window_handle,
            "window_class_name": window.class_name if window is not None else "",
            "visible_window_count": len(windows or []),
        }
        if windows:
            observation["windows"] = [
                {
                    "title": item.title,
                    "handle": item.handle,
                    "class_name": item.class_name,
                    "process_id": item.process_id,
                }
                for item in windows
            ]
        return {
            "pause_kind": "desktop_control_takeover",
            "summary": summary,
            "content": reason,
            "fallback_reason": reason,
            "target": observation["window_title"] or self.active_target_app or "hidden desktop",
            "compatibility_state": self.compatibility_state,
            "capture_backend": self.capture_backend,
            "desktop_observation": observation,
            "window_title": observation["window_title"],
            "window_handle": observation["window_handle"],
            "window_class_name": observation["window_class_name"],
            "visible_window_count": observation["visible_window_count"],
            "action": action,
        }

    def _current_window_handle(self) -> int:
        target = self.current_target_window()
        if target is None:
            return 0
        self.note_active_window(
            profile_id=guess_windows_app_profile(target.title) or "",
            app=self.active_target_app,
            window_title=target.title,
            window_handle=target.handle,
        )
        return target.handle

    def _execute_computer_action(self, action: str, **kwargs: Any) -> ToolResult:
        if action == "screenshot":
            self.session_status = "ready"
            return self._computer_result(action="screenshot", output="Hidden desktop frame captured.", summary="Capturing hidden desktop frame")
        if action in _RISKY_COMPUTER_ACTIONS:
            takeover = self.guard_interaction(
                action=action,
                window_title=self.active_window_title,
                app=self.active_target_app,
                profile_id=self.active_profile_id,
                require_background_preview=True,
            )
            if takeover is not None:
                return ToolResult(
                    success=False,
                    error=str(takeover.get("content") or takeover.get("fallback_reason") or "Hidden desktop action paused."),
                    metadata=takeover,
                )
        handle = self._current_window_handle()
        if not handle:
            return ToolResult(
                success=False,
                error="No active hidden desktop window is available.",
                metadata={"fallback_reason": "No active hidden desktop window is available."},
            )

        x = int(kwargs.get("x", 0) or 0)
        y = int(kwargs.get("y", 0) or 0)
        capture_result_screenshot = bool(kwargs.get("capture_result_screenshot", True))
        if action == "click":
            self.input_adapter.click(handle, x, y, button="left", double=False)
            return self._computer_result(action="click", output=f"Clicked ({x}, {y}) in the hidden session.", summary="Clicking inside the hidden desktop worker", capture_result_screenshot=capture_result_screenshot)
        if action == "double_click":
            self.input_adapter.click(handle, x, y, button="left", double=True)
            return self._computer_result(action="double_click", output=f"Double-clicked ({x}, {y}) in the hidden session.", summary="Double-clicking inside the hidden desktop worker", capture_result_screenshot=capture_result_screenshot)
        if action == "right_click":
            self.input_adapter.click(handle, x, y, button="right", double=False)
            return self._computer_result(action="right_click", output=f"Right-clicked ({x}, {y}) in the hidden session.", summary="Right-clicking inside the hidden desktop worker", capture_result_screenshot=capture_result_screenshot)
        if action == "move":
            return self._computer_result(action="move", output=f"Virtual pointer moved to ({x}, {y}) in the hidden session.", summary="Moving inside the hidden desktop worker", capture_result_screenshot=capture_result_screenshot)
        if action == "scroll":
            amount = int(kwargs.get("delta_y", 0) or 0)
            self.input_adapter.scroll(handle, x, y, amount)
            return self._computer_result(action="scroll", output=f"Scrolled {amount} in the hidden session.", summary="Scrolling inside the hidden desktop worker", capture_result_screenshot=capture_result_screenshot)
        if action == "drag":
            end_x = int(kwargs.get("end_x", 0) or 0)
            end_y = int(kwargs.get("end_y", 0) or 0)
            self.input_adapter.drag(handle, x, y, end_x, end_y)
            return self._computer_result(action="drag", output=f"Dragged from ({x}, {y}) to ({end_x}, {end_y}) in the hidden session.", summary="Dragging inside the hidden desktop worker", capture_result_screenshot=capture_result_screenshot)
        if action == "type":
            text = str(kwargs.get("text", "") or "")
            self.input_adapter.type_text(handle, text)
            return self._computer_result(action="type", output=f"Typed {text!r} in the hidden session.", summary="Typing inside the hidden desktop worker", capture_result_screenshot=capture_result_screenshot)
        if action == "key":
            text = str(kwargs.get("text", "") or "")
            self.input_adapter.key_sequence(handle, text)
            return self._computer_result(action="key", output=f"Pressed {text!r} in the hidden session.", summary="Sending keys inside the hidden desktop worker", capture_result_screenshot=capture_result_screenshot)
        return ToolResult(success=False, error=f"Unknown action: {action!r}")

    def _computer_result(
        self,
        *,
        action: str,
        output: str,
        summary: str,
        capture_result_screenshot: bool = True,
    ) -> ToolResult:
        self.session_status = "ready"
        metadata = {
            "frame_label": f"computer · {action}",
            "summary": summary,
            "desktop_execution_mode": "desktop_hidden",
            "capture_backend": self.capture_backend,
            "compatibility_state": self.compatibility_state,
            "fallback_reason": self.fallback_reason,
        }
        if not capture_result_screenshot:
            return ToolResult(success=True, output=output, metadata=metadata)
        attempt = self._capture_current_window()
        self.capture_backend = attempt.backend
        self.compatibility_state = attempt.compatibility_state
        self.fallback_reason = attempt.fallback_reason
        if attempt.data is None:
            return ToolResult(success=True, output=output, metadata=metadata)
        self.session_status = "ready"
        return ToolResult(
            success=True,
            output=output,
            screenshot_b64=base64.b64encode(attempt.data).decode("ascii"),
            metadata=metadata,
        )


class HiddenDesktopSession:
    """High-level wrapper around the Win32 hidden desktop worker."""

    def __init__(self, thread_id: str) -> None:
        self.thread_id = thread_id
        self._worker: _Win32HiddenDesktopWorker | None = None
        self._snapshot = DesktopSessionSnapshot(
            mode="desktop_hidden",
            session_status="not_started",
            compatibility_state="unknown",
            session_id=f"hidden-{thread_id}",
            desktop_backend="hidden_windows",
        )

    def close(self) -> None:
        if self._worker is not None:
            self._worker.close()
            self._worker = None

    def snapshot(self) -> DesktopSessionSnapshot:
        if self._worker is None:
            return self._snapshot
        try:
            self._snapshot = self._worker.call("snapshot")
        except Exception as exc:  # noqa: BLE001
            self._snapshot = DesktopSessionSnapshot(
                mode="desktop_hidden",
                session_status="error",
                compatibility_state="fallback_required",
                fallback_reason=str(exc),
                session_id=self._snapshot.session_id,
                desktop_backend="hidden_windows",
            )
        return self._snapshot

    def execute_windows_action(self, action: str, **kwargs: Any) -> DesktopActionVerification:
        worker = self._ensure_worker()
        result = worker.call("windows", action=action, kwargs=kwargs)
        self._snapshot = worker.call("snapshot")
        return result

    def execute_computer_action(self, action: str, **kwargs: Any) -> ToolResult:
        worker = self._ensure_worker()
        result = worker.call("computer", action=action, kwargs=kwargs)
        self._snapshot = worker.call("snapshot")
        return result

    def capture_live_frame(self) -> DesktopLiveFrame | None:
        worker = self._ensure_worker()
        frame = worker.call("capture")
        self._snapshot = worker.call("snapshot")
        return frame

    def _ensure_worker(self) -> _Win32HiddenDesktopWorker:
        if self._worker is not None:
            return self._worker
        if os.name != "nt":
            self._snapshot = DesktopSessionSnapshot(
                mode="desktop_hidden",
                session_status="error",
                compatibility_state="fallback_required",
                fallback_reason="Hidden desktop workers are only available on Windows.",
                session_id=self._snapshot.session_id,
                desktop_backend="hidden_windows",
            )
            raise RuntimeError(self._snapshot.fallback_reason)
        worker = _Win32HiddenDesktopWorker(self.thread_id)
        worker.start()
        if worker.session_status == "error":
            self._snapshot = worker.snapshot()
            raise RuntimeError(worker.fallback_reason or "Hidden desktop worker startup failed.")
        self._worker = worker
        self._snapshot = worker.snapshot()
        return worker


class DesktopSessionManager:
    """Own visible desktop capture plus per-thread hidden worker sessions."""

    def __init__(self) -> None:
        self._configs: dict[str, dict[str, str]] = {}
        self._hidden_sessions: dict[str, HiddenDesktopSession] = {}
        self._restored_snapshots: dict[str, DesktopSessionSnapshot] = {}
        self._visible_windows_backend = NativeWindowsDesktopBackend()
        self._lock = threading.Lock()

    def note_thread_config(self, thread_id: str, config: AgentConfig) -> None:
        with self._lock:
            self._configs[thread_id] = {
                "local_execution_mode": str(config.local_execution_mode),
                "desktop_execution_mode": str(config.desktop_execution_mode),
                "desktop_backend_preference": str(config.desktop_backend_preference),
            }
            if config.desktop_execution_mode != "desktop_hidden" or config.local_execution_mode == "under_the_hood":
                session = self._hidden_sessions.pop(thread_id, None)
                if session is not None:
                    session.close()

    def restore_thread_snapshot(self, thread_id: str, payload: Any) -> None:
        if not isinstance(payload, dict):
            return
        desktop_payload = payload.get("desktop_session") if isinstance(payload.get("desktop_session"), dict) else payload
        if not isinstance(desktop_payload, dict):
            return
        self._restored_snapshots[thread_id] = DesktopSessionSnapshot(
            mode=str(desktop_payload.get("mode", "desktop_visible") or "desktop_visible"),
            session_status=str(desktop_payload.get("session_status", "not_started") or "not_started"),
            active_target_app=str(desktop_payload.get("active_target_app", "") or ""),
            active_target_window=str(desktop_payload.get("active_target_window", "") or ""),
            capture_backend=str(desktop_payload.get("capture_backend", "") or ""),
            compatibility_state=str(desktop_payload.get("compatibility_state", "unknown") or "unknown"),
            fallback_reason=str(desktop_payload.get("fallback_reason", "") or ""),
            session_id=str(desktop_payload.get("session_id", "") or ""),
            desktop_backend=str(desktop_payload.get("desktop_backend", "") or ""),
            active_window_handle=int(desktop_payload.get("active_window_handle", 0) or 0),
        )

    def thread_mode(self, thread_id: str) -> str:
        return str(self._configs.get(thread_id, {}).get("desktop_execution_mode", "desktop_visible"))

    def thread_local_mode(self, thread_id: str) -> str:
        return str(self._configs.get(thread_id, {}).get("local_execution_mode", "visible"))

    def hidden_capability(self, thread_id: str) -> str | None:
        if self.thread_mode(thread_id) != "desktop_hidden":
            return None
        if self.thread_local_mode(thread_id) == "under_the_hood":
            return None
        return f"{_HIDDEN_SESSION_PREFIX}{thread_id}"

    def snapshot_payload(self, thread_id: str) -> dict[str, Any]:
        mode = self.thread_mode(thread_id)
        if mode == "desktop_hidden" and self.thread_local_mode(thread_id) != "under_the_hood":
            session = self._hidden_sessions.get(thread_id)
            if session is None:
                snapshot = self._restored_snapshots.get(thread_id)
                if snapshot is None:
                    snapshot = DesktopSessionSnapshot(
                        mode="desktop_hidden",
                        session_status="not_started",
                        compatibility_state="unknown",
                        session_id=f"hidden-{thread_id}",
                        desktop_backend="hidden_windows",
                    )
            else:
                snapshot = session.snapshot()
            return {"desktop_session": snapshot.payload()}

        backend = "native_windows" if mode == "desktop_native" else "visible_desktop"
        snapshot = DesktopSessionSnapshot(
            mode=mode,
            session_status="visible",
            compatibility_state="visible_control",
            session_id=f"visible-{thread_id}",
            desktop_backend=backend,
            capture_backend="screen_capture",
        )
        return {"desktop_session": snapshot.payload()}

    def execute_windows_action(self, thread_id: str, action: str, **kwargs: Any) -> DesktopActionVerification:
        if self.hidden_capability(thread_id):
            session = self._hidden_sessions.setdefault(thread_id, HiddenDesktopSession(thread_id))
            return session.execute_windows_action(action, **kwargs)
        return self._visible_windows_backend.execute(action, **kwargs)

    def execute_computer_action(self, thread_id: str, action: str, **kwargs: Any) -> ToolResult:
        if self.hidden_capability(thread_id):
            session = self._hidden_sessions.setdefault(thread_id, HiddenDesktopSession(thread_id))
            return session.execute_computer_action(action, **kwargs)
        raise RuntimeError("Visible desktop computer actions should execute through ComputerTool directly.")

    async def capture_live_frame(self, thread_id: str) -> DesktopLiveFrame | None:
        if self.hidden_capability(thread_id):
            session = self._hidden_sessions.setdefault(thread_id, HiddenDesktopSession(thread_id))
            return await asyncio.to_thread(session.capture_live_frame)
        return await asyncio.to_thread(self._capture_visible_frame)

    def _capture_visible_frame(self) -> DesktopLiveFrame | None:
        try:
            return DesktopLiveFrame(data=_visible_desktop_png_bytes(), media_type="image/png")
        except Exception:  # noqa: BLE001
            return None


class POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]


class RECT(ctypes.Structure):
    _fields_ = [
        ("left", ctypes.c_long),
        ("top", ctypes.c_long),
        ("right", ctypes.c_long),
        ("bottom", ctypes.c_long),
    ]


class GUITHREADINFO(ctypes.Structure):
    _fields_ = [
        ("cbSize", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("hwndActive", _POINTER_SIZED),
        ("hwndFocus", _POINTER_SIZED),
        ("hwndCapture", _POINTER_SIZED),
        ("hwndMenuOwner", _POINTER_SIZED),
        ("hwndMoveSize", _POINTER_SIZED),
        ("hwndCaret", _POINTER_SIZED),
        ("rcCaret", RECT),
    ]
