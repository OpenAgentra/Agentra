"""Structured desktop automation models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class WindowInfo:
    """Top-level window metadata."""

    handle: int
    title: str
    class_name: str = ""
    process_id: int = 0
    visible: bool = True


@dataclass(frozen=True)
class ControlInfo:
    """Structured UI control metadata."""

    name: str
    control_type: str
    automation_id: str
    class_name: str
    depth: int


@dataclass
class DesktopActionVerification:
    """Outcome of a desktop backend action with explicit verification state."""

    target: str
    action: str
    observed_outcome: str
    success: bool
    verified: bool
    fallback_reason: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    def metadata(
        self,
        *,
        backend: str,
        desktop_execution_mode: str,
        frame_label: str,
        summary: str,
    ) -> dict[str, Any]:
        payload = {
            "target": self.target,
            "action": self.action,
            "observed_outcome": self.observed_outcome,
            "verified": self.verified,
            "fallback_reason": self.fallback_reason,
            "desktop_backend": backend,
            "desktop_execution_mode": desktop_execution_mode,
            "frame_label": frame_label,
            "summary": summary,
        }
        payload.update(self.details)
        return payload


@dataclass(frozen=True)
class DesktopLiveFrame:
    """PNG-like live frame payload for a desktop session."""

    data: bytes
    media_type: str = "image/png"


@dataclass
class DesktopSessionSnapshot:
    """Thread-scoped desktop session state exposed to runtime and HTTP snapshots."""

    mode: str
    session_status: str
    active_target_app: str = ""
    active_target_window: str = ""
    capture_backend: str = ""
    compatibility_state: str = "unknown"
    fallback_reason: str = ""
    session_id: str = ""
    desktop_backend: str = ""
    active_window_handle: int = 0

    def payload(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "session_status": self.session_status,
            "active_target_app": self.active_target_app,
            "active_target_window": self.active_target_window,
            "capture_backend": self.capture_backend,
            "compatibility_state": self.compatibility_state,
            "fallback_reason": self.fallback_reason,
            "session_id": self.session_id,
            "desktop_backend": self.desktop_backend,
            "active_window_handle": self.active_window_handle,
        }
