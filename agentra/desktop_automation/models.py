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
