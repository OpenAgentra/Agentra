from __future__ import annotations

from agentra.desktop_automation.models import WindowInfo
from agentra.desktop_automation.session_manager import (
    _CaptureAttempt,
    _HiddenDesktopBackend,
    _Win32HiddenDesktopWorker,
)


def test_hidden_desktop_worker_pauses_when_target_window_is_ambiguous(monkeypatch) -> None:
    worker = _Win32HiddenDesktopWorker("thread-hidden")
    windows = [
        WindowInfo(handle=1, title="First App", class_name="AppWindow"),
        WindowInfo(handle=2, title="Second App", class_name="AppWindow"),
    ]

    monkeypatch.setattr(worker, "_enumerated_windows", lambda: list(windows))

    takeover = worker.guard_interaction(action="click", require_background_preview=True)

    assert takeover is not None
    assert takeover["pause_kind"] == "desktop_control_takeover"
    assert takeover["visible_window_count"] == 2
    assert worker.session_status == "blocked_waiting_user"


def test_hidden_desktop_worker_pauses_when_unexpected_dialog_is_present(monkeypatch) -> None:
    worker = _Win32HiddenDesktopWorker("thread-hidden")
    worker.active_profile_id = "notepad"
    worker.active_target_app = "notepad.exe"
    worker.active_window_title = "Untitled - Notepad"
    worker.active_window_handle = 11
    windows = [
        WindowInfo(handle=11, title="Untitled - Notepad", class_name="Notepad"),
        WindowInfo(handle=12, title="Open", class_name="#32770"),
    ]

    monkeypatch.setattr(worker, "_enumerated_windows", lambda: list(windows))
    monkeypatch.setattr(
        worker,
        "_capture_window",
        lambda handle: _CaptureAttempt(data=b"png", backend="printwindow_client", compatibility_state="background_capable"),
    )

    takeover = worker.guard_interaction(action="type_keys", require_background_preview=True)

    assert takeover is not None
    assert takeover["pause_kind"] == "desktop_control_takeover"
    assert takeover["window_class_name"] == "#32770"
    assert worker.compatibility_state == "fallback_required"
    assert worker.session_status == "blocked_waiting_user"


def test_hidden_desktop_worker_pauses_when_background_preview_is_unsupported(monkeypatch) -> None:
    worker = _Win32HiddenDesktopWorker("thread-hidden")
    worker.active_target_app = "paint.exe"
    worker.active_window_title = "Paint"
    worker.active_window_handle = 22
    windows = [WindowInfo(handle=22, title="Paint", class_name="MSPaintApp")]

    monkeypatch.setattr(worker, "_enumerated_windows", lambda: list(windows))
    monkeypatch.setattr(
        worker,
        "_capture_window",
        lambda handle: _CaptureAttempt(
            data=None,
            backend="windows_graphics_capture",
            compatibility_state="fallback_required",
            fallback_reason="Unsupported GPU surface.",
        ),
    )

    takeover = worker.guard_interaction(action="drag", require_background_preview=True)

    assert takeover is not None
    assert takeover["pause_kind"] == "desktop_control_takeover"
    assert takeover["capture_backend"] == "windows_graphics_capture"
    assert takeover["compatibility_state"] == "fallback_required"
    assert worker.session_status == "blocked_waiting_user"


def test_hidden_desktop_backend_uses_current_target_window_when_action_has_no_explicit_target(monkeypatch) -> None:
    worker = _Win32HiddenDesktopWorker("thread-hidden")
    backend = _HiddenDesktopBackend(worker)
    current = WindowInfo(handle=33, title="Agentra Test Window", class_name="AppWindow")

    monkeypatch.setattr(worker, "current_target_window", lambda: current)

    assert backend._resolve_window() == current
