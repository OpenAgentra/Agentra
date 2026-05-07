"""Abstract base for structured desktop automation backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from agentra.desktop_automation.models import DesktopActionVerification


class DesktopAutomationBackend(ABC):
    """Backend abstraction for native or visible desktop automation."""

    backend_id = "desktop"
    desktop_execution_mode = "desktop_visible"
    supports_structured_controls = False

    @abstractmethod
    def is_available(self) -> tuple[bool, str | None]:
        """Return backend availability and an optional reason when unavailable."""

    @abstractmethod
    def execute(self, action: str, **kwargs: Any) -> DesktopActionVerification:
        """Execute a structured desktop action."""
