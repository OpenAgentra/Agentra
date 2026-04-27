"""Native Windows desktop automation tool."""

from __future__ import annotations

import asyncio
import ctypes
import os
from typing import Any

from agentra.desktop_automation import NativeWindowsDesktopBackend
from agentra.tools.base import BaseTool, ToolResult
from agentra.windows_apps import get_windows_app_profile, guess_windows_app_profile


_VISIBLE_DESKTOP_ACTIONS = frozenset({"launch_app", "focus_window", "invoke_control", "set_text", "type_keys"})
_AGENTRA_TITLE_TERMS = ("agentra", "codex", "127.0.0.1:8765", "localhost:8765")


class WindowsDesktopTool(BaseTool):
    """Structured Windows-native desktop automation for standard apps."""

    name = "windows_desktop"
    tool_capabilities = ("computer", "windows_desktop")
    description = (
        "Use native Windows UI Automation for standard desktop apps such as Calculator, "
        "Notepad, Explorer, common dialogs, buttons, and text inputs. Prefer this before "
        "`computer` for basic Windows app tasks because it can launch apps, focus windows, "
        "inspect controls, type keys, set text, and verify outcomes without blind clicking."
    )

    def __init__(self) -> None:
        self._backend = NativeWindowsDesktopBackend()
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
            return (session_capability, "windows_desktop")
        return super().capabilities

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "launch_app",
                        "focus_window",
                        "wait_for_window",
                        "list_windows",
                        "list_controls",
                        "invoke_control",
                        "set_text",
                        "type_keys",
                        "read_window_text",
                        "read_status",
                    ],
                    "description": "Native Windows desktop action to perform.",
                },
                "app": {"type": "string", "description": "Windows app name or executable."},
                "profile_id": {
                    "type": "string",
                    "enum": ["calculator", "notepad", "explorer", "paint", "wordpad"],
                    "description": "Optional standard app profile hint.",
                },
                "window_title": {
                    "type": "string",
                    "description": "Window title or title fragment to target.",
                },
                "text": {
                    "type": "string",
                    "description": "Text or key sequence for set_text / type_keys.",
                },
                "expected_text": {
                    "type": "string",
                    "description": "Expected status text for read_status verification.",
                },
                "control_name": {
                    "type": "string",
                    "description": "UI Automation control name to target.",
                },
                "automation_id": {
                    "type": "string",
                    "description": "UI Automation automation ID to target.",
                },
                "control_type": {
                    "type": "string",
                    "description": "UI Automation control type name, such as ButtonControl.",
                },
                "timeout_sec": {
                    "type": "number",
                    "description": "Timeout for wait_for_window or launch verification.",
                },
                "max_depth": {
                    "type": "integer",
                    "description": "Maximum control tree depth for discovery actions.",
                },
            },
            "required": ["action"],
        }

    async def preview(self, **kwargs: Any) -> dict[str, Any] | None:
        action = str(kwargs.get("action", "") or "").strip().lower()
        if not action:
            return None
        target = str(
            kwargs.get("app")
            or kwargs.get("window_title")
            or kwargs.get("profile_id")
            or kwargs.get("control_name")
            or "Windows app"
        )
        return {
            "frame_label": f"windows_desktop · {action}",
            "summary": f"Native Windows automation: {action} -> {target}",
        }

    async def execute(self, **kwargs: Any) -> ToolResult:
        action = str(kwargs.get("action", "") or "").strip().lower()
        session_capability = self._hidden_session_capability()
        if session_capability and self._desktop_sessions is not None and self._thread_id:
            backend_kwargs = dict(kwargs)
            backend_kwargs.pop("action", None)
            try:
                verification = await asyncio.to_thread(
                    self._desktop_sessions.execute_windows_action,
                    self._thread_id,
                    action,
                    **backend_kwargs,
                )
            except Exception as exc:  # noqa: BLE001
                return ToolResult(success=False, error=str(exc))
            summary = verification.observed_outcome or verification.fallback_reason or "Hidden desktop action completed."
            metadata = verification.metadata(
                backend="hidden_windows",
                desktop_execution_mode="desktop_hidden",
                frame_label=f"windows_desktop · {action or 'action'}",
                summary=summary,
            )
            return ToolResult(
                success=verification.success,
                output=verification.observed_outcome,
                error="" if verification.success else verification.fallback_reason or verification.observed_outcome,
                metadata=metadata,
            )
        desktop_conflict = self._desktop_control_conflict(action, kwargs)
        if desktop_conflict is not None:
            return ToolResult(
                success=False,
                error=str(desktop_conflict.get("content") or desktop_conflict.get("summary") or "Desktop control conflict detected."),
                metadata=desktop_conflict,
            )
        self._prepare_visible_desktop_action(action)
        backend_kwargs = dict(kwargs)
        backend_kwargs.pop("action", None)
        verification = await asyncio.to_thread(self._backend.execute, action, **backend_kwargs)
        summary = verification.observed_outcome or verification.fallback_reason or "Native desktop action completed."
        metadata = verification.metadata(
            backend=self._backend.backend_id,
            desktop_execution_mode=self._backend.desktop_execution_mode,
            frame_label=f"windows_desktop · {action or 'action'}",
            summary=summary,
        )
        return ToolResult(
            success=verification.success,
            output=verification.observed_outcome,
            error="" if verification.success else verification.fallback_reason or verification.observed_outcome,
            metadata=metadata,
        )

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

    def _prepare_visible_desktop_action(self, action: str) -> bool:
        normalized_action = str(action or "").strip().lower()
        if normalized_action not in _VISIBLE_DESKTOP_ACTIONS:
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

    def _desktop_control_conflict(self, action: str, tool_args: dict[str, Any]) -> dict[str, Any] | None:
        normalized_action = str(action or "").strip().lower()
        if normalized_action not in _VISIBLE_DESKTOP_ACTIONS:
            return None
        foreground = self._foreground_window_snapshot()
        if foreground is None:
            return None
        current_title = str(foreground.get("window_title", "") or "").strip()
        if not current_title:
            return None
        normalized_title = current_title.casefold()
        if any(term in normalized_title for term in _AGENTRA_TITLE_TERMS):
            return None
        if self._foreground_matches_target(normalized_title, tool_args):
            return None
        return {
            "pause_kind": "desktop_control_takeover",
            "summary": "Gorunur Windows otomasyonu odagi degistirecegi icin kontrol size devredildi.",
            "content": (
                "Bu adim gorunur bir Windows uygulamasi acacak veya odagi ona tasiyacak. "
                "Baska bir uygulamada calisiyor gorunuyorsunuz. Hazir oldugunuzda Finish Control ile devam edin."
            ),
            "desktop_observation": foreground,
        }

    def _foreground_matches_target(self, normalized_title: str, tool_args: dict[str, Any]) -> bool:
        target_title = str(tool_args.get("window_title", "") or "").strip().casefold()
        if target_title and target_title in normalized_title:
            return True
        profile_id = str(tool_args.get("profile_id", "") or "").strip().lower()
        if not profile_id:
            profile_id = str(guess_windows_app_profile(str(tool_args.get("app", "") or str(tool_args.get("window_title", "") or ""))) or "").strip().lower()
        profile = get_windows_app_profile(profile_id) if profile_id else None
        if profile is not None:
            if any(str(token or "").casefold() in normalized_title for token in profile.window_title_tokens):
                return True
        app = str(tool_args.get("app", "") or "").strip().casefold()
        if app and app.replace(".exe", "") in normalized_title:
            return True
        return False

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
