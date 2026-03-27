"""Desktop automation backends and support models."""

from agentra.desktop_automation.base import DesktopAutomationBackend
from agentra.desktop_automation.models import ControlInfo, DesktopActionVerification, WindowInfo
from agentra.desktop_automation.preview_windows import PreviewWindowManager
from agentra.desktop_automation.windows_native import NativeWindowsDesktopBackend

__all__ = [
    "ControlInfo",
    "DesktopActionVerification",
    "DesktopAutomationBackend",
    "NativeWindowsDesktopBackend",
    "PreviewWindowManager",
    "WindowInfo",
]
