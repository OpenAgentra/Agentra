"""Desktop automation backends and support models."""

from agentra.desktop_automation.base import DesktopAutomationBackend
from agentra.desktop_automation.models import (
    ControlInfo,
    DesktopActionVerification,
    DesktopLiveFrame,
    DesktopSessionSnapshot,
    WindowInfo,
)
from agentra.desktop_automation.preview_windows import PreviewWindowManager
from agentra.desktop_automation.session_manager import DesktopSessionManager
from agentra.desktop_automation.windows_native import NativeWindowsDesktopBackend

__all__ = [
    "ControlInfo",
    "DesktopActionVerification",
    "DesktopAutomationBackend",
    "DesktopLiveFrame",
    "DesktopSessionManager",
    "DesktopSessionSnapshot",
    "NativeWindowsDesktopBackend",
    "PreviewWindowManager",
    "WindowInfo",
]
