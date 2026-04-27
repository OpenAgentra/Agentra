"""Helpers for temporarily moving Agentra's own preview out of the way."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any


_PREVIEW_TITLE_TERMS = ("agentra", "127.0.0.1:8765", "localhost:8765")


@dataclass(frozen=True)
class PreviewWindowSnapshot:
    """Minimal window snapshot used for preview parking."""

    handle: int
    title: str


class PreviewWindowManager:
    """Hide and restore blocking Agentra preview windows on Windows."""

    def __init__(self) -> None:
        self._hidden_handles: list[int] = []

    def prepare_for_visible_desktop_action(self) -> dict[str, Any] | None:
        """Minimize the foreground preview window when it blocks visible desktop control."""

        snapshot = self._foreground_window()
        if snapshot is None or not self._looks_like_preview(snapshot.title):
            return None
        try:
            import win32con  # noqa: PLC0415
            import win32gui  # noqa: PLC0415
        except ImportError:
            return None

        try:
            win32gui.ShowWindow(snapshot.handle, win32con.SW_MINIMIZE)
        except Exception:  # noqa: BLE001
            return None
        if snapshot.handle not in self._hidden_handles:
            self._hidden_handles.append(snapshot.handle)
        time.sleep(0.2)
        return {
            "hidden_preview_window": True,
            "hidden_preview_title": snapshot.title,
            "hidden_preview_handle": snapshot.handle,
        }

    def restore_preview_windows(self) -> int:
        """Restore any preview windows hidden during visible desktop fallback."""

        if os.name != "nt" or not self._hidden_handles:
            self._hidden_handles.clear()
            return 0
        try:
            import win32con  # noqa: PLC0415
            import win32gui  # noqa: PLC0415
        except ImportError:
            self._hidden_handles.clear()
            return 0

        restored = 0
        for handle in reversed(self._hidden_handles):
            try:
                if win32gui.IsWindow(handle):
                    win32gui.ShowWindow(handle, win32con.SW_RESTORE)
                    restored += 1
            except Exception:  # noqa: BLE001
                continue
        self._hidden_handles.clear()
        return restored

    @staticmethod
    def _looks_like_preview(title: str) -> bool:
        normalized = str(title or "").casefold()
        return any(term in normalized for term in _PREVIEW_TITLE_TERMS)

    @staticmethod
    def _foreground_window() -> PreviewWindowSnapshot | None:
        if os.name != "nt":
            return None
        try:
            import win32gui  # noqa: PLC0415
        except ImportError:
            return None
        try:
            handle = int(win32gui.GetForegroundWindow() or 0)
        except Exception:  # noqa: BLE001
            return None
        if not handle:
            return None
        try:
            title = str(win32gui.GetWindowText(handle) or "").strip()
        except Exception:  # noqa: BLE001
            title = ""
        return PreviewWindowSnapshot(handle=handle, title=title)
