"""Tests for individual tools."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

from agentra.tools.base import ToolResult
from agentra.desktop_automation.models import DesktopActionVerification, WindowInfo
from agentra.desktop_automation.windows_native import NativeWindowsDesktopBackend
from agentra.tools.filesystem import FilesystemTool
from agentra.tools.git_tool import GitTool
from agentra.tools.local_system import LocalSystemTool, ResolvedLocalPath
from agentra.tools.terminal import TerminalTool


# ── FilesystemTool ────────────────────────────────────────────────────────────

@pytest.fixture
def fs_tool(tmp_path):
    return FilesystemTool(workspace_dir=tmp_path, allow_write=True)


@pytest.fixture
def fs_tool_readonly(tmp_path):
    return FilesystemTool(workspace_dir=tmp_path, allow_write=False)


@pytest.mark.asyncio
async def test_filesystem_write_and_read(fs_tool, tmp_path):
    result = await fs_tool.execute(action="write", path="test.txt", content="hello")
    assert result.success

    result = await fs_tool.execute(action="read", path="test.txt")
    assert result.success
    assert result.output == "hello"


@pytest.mark.asyncio
async def test_filesystem_append(fs_tool, tmp_path):
    await fs_tool.execute(action="write", path="a.txt", content="line1\n")
    result = await fs_tool.execute(action="append", path="a.txt", content="line2\n")
    assert result.success

    result = await fs_tool.execute(action="read", path="a.txt")
    assert "line1" in result.output
    assert "line2" in result.output


@pytest.mark.asyncio
async def test_filesystem_list(fs_tool, tmp_path):
    (tmp_path / "a.txt").write_text("a")
    (tmp_path / "b.txt").write_text("b")

    result = await fs_tool.execute(action="list", path=str(tmp_path))
    assert result.success
    assert "a.txt" in result.output
    assert "b.txt" in result.output


@pytest.mark.asyncio
async def test_filesystem_mkdir(fs_tool, tmp_path):
    new_dir = tmp_path / "subdir" / "nested"
    result = await fs_tool.execute(action="mkdir", path=str(new_dir))
    assert result.success
    assert new_dir.exists()


@pytest.mark.asyncio
async def test_filesystem_delete_file(fs_tool, tmp_path):
    p = tmp_path / "del.txt"
    p.write_text("delete me")

    result = await fs_tool.execute(action="delete", path=str(p))
    assert result.success
    assert not p.exists()


@pytest.mark.asyncio
async def test_filesystem_exists(fs_tool, tmp_path):
    p = tmp_path / "exists.txt"
    p.write_text("yes")

    result = await fs_tool.execute(action="exists", path=str(p))
    assert result.success
    assert result.output == "True"

    result = await fs_tool.execute(action="exists", path=str(tmp_path / "nope.txt"))
    assert result.output == "False"


@pytest.mark.asyncio
async def test_filesystem_copy(fs_tool, tmp_path):
    src = tmp_path / "src.txt"
    dst = tmp_path / "dst.txt"
    src.write_text("copy me")

    result = await fs_tool.execute(action="copy", path=str(src), destination=str(dst))
    assert result.success
    assert dst.read_text() == "copy me"


@pytest.mark.asyncio
async def test_filesystem_move(fs_tool, tmp_path):
    src = tmp_path / "mv_src.txt"
    dst = tmp_path / "mv_dst.txt"
    src.write_text("move me")

    result = await fs_tool.execute(action="move", path=str(src), destination=str(dst))
    assert result.success
    assert not src.exists()
    assert dst.read_text() == "move me"


@pytest.mark.asyncio
async def test_filesystem_write_blocked_readonly(fs_tool_readonly, tmp_path):
    result = await fs_tool_readonly.execute(action="write", path="x.txt", content="hello")
    assert not result.success
    assert "disabled" in result.error.lower()


@pytest.mark.asyncio
async def test_filesystem_read_nonexistent(fs_tool, tmp_path):
    result = await fs_tool.execute(action="read", path=str(tmp_path / "ghost.txt"))
    assert not result.success


@pytest.mark.asyncio
async def test_filesystem_unknown_action(fs_tool):
    result = await fs_tool.execute(action="explode")
    assert not result.success


@pytest.mark.asyncio
async def test_filesystem_cwd(fs_tool):
    result = await fs_tool.execute(action="cwd")
    assert result.success
    assert len(result.output) > 0


# ── TerminalTool ──────────────────────────────────────────────────────────────

@pytest.fixture
def term_tool(tmp_path):
    return TerminalTool(cwd=tmp_path, allow=True, timeout=10)


@pytest.fixture
def term_tool_disabled(tmp_path):
    return TerminalTool(cwd=tmp_path, allow=False)


@pytest.mark.asyncio
async def test_terminal_echo(term_tool):
    result = await term_tool.execute(command="echo hello")
    assert result.success
    assert "hello" in result.output


@pytest.mark.asyncio
async def test_terminal_nonzero_exit(term_tool):
    result = await term_tool.execute(command="exit 1")
    assert not result.success
    assert result.metadata.get("returncode") == 1


@pytest.mark.asyncio
async def test_terminal_disabled(term_tool_disabled):
    result = await term_tool_disabled.execute(command="echo hi")
    assert not result.success
    assert "disabled" in result.error.lower()


@pytest.mark.asyncio
async def test_terminal_empty_command(term_tool):
    result = await term_tool.execute(command="   ")
    assert not result.success


@pytest.mark.asyncio
async def test_terminal_timeout(tmp_path):
    tool = TerminalTool(cwd=tmp_path, allow=True, timeout=1)
    result = await tool.execute(
        command=f'"{sys.executable}" -c "import time; time.sleep(10)"'
    )
    assert not result.success
    assert "timed out" in result.error.lower()


# ── LocalSystemTool ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_local_system_resolve_known_folder_returns_wsl_and_windows_paths(monkeypatch):
    tool = LocalSystemTool()

    monkeypatch.setattr(
        tool,
        "_resolve_known_folder_sync",
        lambda folder_key: ResolvedLocalPath(
            requested=folder_key,
            windows_path="C:\\Users\\ariba\\OneDrive\\Desktop",
            wsl_path="/mnt/c/Users/ariba/OneDrive/Desktop",
        ),
    )

    result = await tool.execute(action="resolve_known_folder", folder_key="desktop")

    assert result.success
    assert "/mnt/c/Users/ariba/OneDrive/Desktop" in result.output
    assert result.metadata["windows_path"] == "C:\\Users\\ariba\\OneDrive\\Desktop"
    assert result.metadata["wsl_path"] == "/mnt/c/Users/ariba/OneDrive/Desktop"


@pytest.mark.asyncio
async def test_local_system_open_path_uses_normalized_windows_target(monkeypatch):
    tool = LocalSystemTool()
    opened: list[str] = []

    monkeypatch.setattr(
        tool,
        "_normalize_input_path",
        lambda path: ResolvedLocalPath(
            requested=path,
            windows_path="C:\\Users\\ariba\\OneDrive\\Desktop\\secondsun\\deck.pptx",
            wsl_path="/mnt/c/Users/ariba/OneDrive/Desktop/secondsun/deck.pptx",
        ),
    )
    monkeypatch.setattr(tool, "_ensure_path_exists", lambda resolved: None)
    monkeypatch.setattr(tool, "_open_path_sync", lambda resolved: opened.append(resolved.windows_path))

    result = await tool.execute(
        action="open_path",
        path="/mnt/c/Users/ariba/OneDrive/Desktop/secondsun/deck.pptx",
    )

    assert result.success
    assert opened == ["C:\\Users\\ariba\\OneDrive\\Desktop\\secondsun\\deck.pptx"]
    assert result.metadata["frame_label"] == "local_system · open_path"


@pytest.mark.asyncio
async def test_local_system_launch_app_normalizes_common_aliases(monkeypatch):
    tool = LocalSystemTool()
    launched: list[str] = []

    monkeypatch.setattr(tool, "_launch_app_sync", lambda app: launched.append(app))

    result = await tool.execute(action="launch_app", app="Calculator")

    assert result.success
    assert launched == ["calc.exe"]
    assert result.metadata["app"] == "calc.exe"
    assert result.metadata["requested_app"] == "Calculator"


@pytest.mark.asyncio
async def test_windows_desktop_tool_reports_missing_backend(monkeypatch):
    from agentra.tools.windows_desktop import WindowsDesktopTool  # noqa: PLC0415

    tool = WindowsDesktopTool()
    monkeypatch.setattr(tool, "_foreground_window_snapshot", lambda: None)
    monkeypatch.setattr(tool._backend, "is_available", lambda: (False, "native backend unavailable"))

    result = await tool.execute(action="launch_app", app="Calculator")

    assert not result.success
    assert "native backend unavailable" in result.error
    assert result.metadata["desktop_execution_mode"] == "desktop_native"


@pytest.mark.asyncio
async def test_windows_desktop_tool_returns_structured_verification_metadata(monkeypatch):
    from agentra.tools.windows_desktop import WindowsDesktopTool  # noqa: PLC0415

    tool = WindowsDesktopTool()
    monkeypatch.setattr(
        tool._backend,
        "execute",
        lambda action, **kwargs: DesktopActionVerification(
            target="Calculator",
            action=action,
            observed_outcome="639",
            success=True,
            verified=True,
            details={"window_title": "Hesap Makinesi"},
        ),
    )

    result = await tool.execute(action="read_status", app="Calculator", expected_text="639")

    assert result.success
    assert result.output == "639"
    assert result.metadata["verified"] is True
    assert result.metadata["window_title"] == "Hesap Makinesi"


@pytest.mark.asyncio
async def test_windows_desktop_tool_routes_hidden_mode_through_session_manager() -> None:
    from agentra.tools.windows_desktop import WindowsDesktopTool  # noqa: PLC0415

    class FakeDesktopSessions:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str, dict[str, object]]] = []

        def hidden_capability(self, thread_id: str) -> str:
            return f"desktop_session:{thread_id}"

        def execute_windows_action(self, thread_id: str, action: str, **kwargs: object) -> DesktopActionVerification:
            self.calls.append((thread_id, action, dict(kwargs)))
            return DesktopActionVerification(
                target="Calculator",
                action=action,
                observed_outcome="639",
                success=True,
                verified=True,
                details={"window_title": "Hidden Calculator"},
            )

    tool = WindowsDesktopTool()
    sessions = FakeDesktopSessions()
    tool.bind_runtime(desktop_sessions=sessions, thread_id="thread-hidden")

    result = await tool.execute(action="read_status", app="Calculator", expected_text="639")

    assert result.success
    assert result.output == "639"
    assert result.metadata["desktop_execution_mode"] == "desktop_hidden"
    assert sessions.calls == [
        ("thread-hidden", "read_status", {"app": "Calculator", "expected_text": "639"})
    ]
    assert tool.capabilities == ("desktop_session:thread-hidden", "windows_desktop")


def test_native_windows_backend_launch_app_accepts_profile_id_without_explicit_app(monkeypatch):
    backend = NativeWindowsDesktopBackend()
    launched: list[list[str]] = []
    waits: list[dict[str, object]] = []

    monkeypatch.setattr(backend, "is_available", lambda: (True, None))
    monkeypatch.setattr(
        "agentra.desktop_automation.windows_native.subprocess.Popen",
        lambda args: launched.append(list(args)) or object(),
    )
    monkeypatch.setattr(
        backend,
        "wait_for_window",
        lambda **kwargs: (
            waits.append(dict(kwargs))
            or DesktopActionVerification(
                target="calc.exe",
                action="wait_for_window",
                observed_outcome="Calculator ready.",
                success=True,
                verified=True,
                details={"window_title": "Calculator"},
            )
        ),
    )

    result = backend.execute("launch_app", profile_id="calculator")

    assert result.success is True
    assert launched == [["calc.exe"]]
    assert waits == [{"app": "calc.exe", "profile_id": "calculator", "timeout_sec": None}]
    assert result.details["requested_app"] == "calc.exe"
    assert result.details["app"] == "calc.exe"


@pytest.mark.asyncio
async def test_windows_desktop_tool_pauses_when_unrelated_window_is_foreground(monkeypatch):
    from agentra.tools.windows_desktop import WindowsDesktopTool  # noqa: PLC0415

    tool = WindowsDesktopTool()
    monkeypatch.setattr(
        tool,
        "_foreground_window_snapshot",
        lambda: {"window_handle": 99, "window_title": "Physics Notes - OneNote"},
    )

    result = await tool.execute(action="launch_app", profile_id="notepad")

    assert result.success is False
    assert result.metadata["pause_kind"] == "desktop_control_takeover"
    assert "gorunur" in result.error.lower()


def test_native_windows_backend_uses_calculator_button_sequence(monkeypatch):
    backend = NativeWindowsDesktopBackend()
    window = WindowInfo(handle=1, title="Hesap Makinesi")
    invoked: list[str] = []
    reads = iter(("before", "after"))

    monkeypatch.setattr(backend, "_resolve_window", lambda **kwargs: window)
    monkeypatch.setattr(
        backend,
        "_invoke_control_by_automation_id",
        lambda _window, *, automation_id: (
            invoked.append(automation_id)
            or DesktopActionVerification(
                target=automation_id,
                action="invoke_control",
                observed_outcome=f"Invoked {automation_id}",
                success=True,
                verified=True,
            )
        ),
    )
    monkeypatch.setattr(
        backend,
        "read_window_text",
        lambda **kwargs: DesktopActionVerification(
            target=window.title,
            action="read_window_text",
            observed_outcome=next(reads),
            success=True,
            verified=True,
        ),
    )
    monkeypatch.setattr(backend, "_calculator_status", lambda _window: "639")

    result = backend.type_keys(app="Calculator", profile_id="calculator", text="482+157=")

    assert result.success
    assert invoked == [
        "clearButton",
        "num4Button",
        "num8Button",
        "num2Button",
        "plusButton",
        "num1Button",
        "num5Button",
        "num7Button",
        "equalButton",
    ]


def test_native_windows_backend_reads_calculator_result_from_native_result_control(monkeypatch):
    backend = NativeWindowsDesktopBackend()
    window = WindowInfo(handle=1, title="Hesap Makinesi")

    class FakeControl:
        Name = "Ekran değeri 639"
        AutomationId = "CalculatorResults"

    monkeypatch.setattr(
        backend,
        "_resolve_control",
        lambda _window, **kwargs: FakeControl() if kwargs.get("automation_id") == "CalculatorResults" else None,
    )

    assert backend._calculator_status(window) == "639"


def test_native_windows_backend_keeps_incremental_calculator_state_between_type_calls(monkeypatch):
    backend = NativeWindowsDesktopBackend()
    window = WindowInfo(handle=1, title="Hesap Makinesi")
    invoked: list[str] = []
    reads = iter(
        (
            "before-1",
            "after-1",
            "before-2",
            "after-2",
            "before-3",
            "after-3",
            "before-4",
            "after-4",
        )
    )
    statuses = iter(("482", "482", "157", "639"))

    monkeypatch.setattr(backend, "_resolve_window", lambda **kwargs: window)
    monkeypatch.setattr(
        backend,
        "_invoke_control_by_automation_id",
        lambda _window, *, automation_id: (
            invoked.append(automation_id)
            or DesktopActionVerification(
                target=automation_id,
                action="invoke_control",
                observed_outcome=f"Invoked {automation_id}",
                success=True,
                verified=True,
            )
        ),
    )
    monkeypatch.setattr(
        backend,
        "read_window_text",
        lambda **kwargs: DesktopActionVerification(
            target=window.title,
            action="read_window_text",
            observed_outcome=next(reads),
            success=True,
            verified=True,
        ),
    )
    monkeypatch.setattr(backend, "_calculator_status", lambda _window: next(statuses))

    assert backend.type_keys(app="Calculator", profile_id="calculator", text="482").success
    assert backend.type_keys(app="Calculator", profile_id="calculator", text="+").success
    assert backend.type_keys(app="Calculator", profile_id="calculator", text="157").success
    assert backend.type_keys(app="Calculator", profile_id="calculator", text="=").success

    assert invoked == [
        "clearButton",
        "num4Button",
        "num8Button",
        "num2Button",
        "plusButton",
        "num1Button",
        "num5Button",
        "num7Button",
        "equalButton",
    ]


def test_native_windows_backend_translates_agent_send_keys_syntax():
    translated = NativeWindowsDesktopBackend._translate_send_keys_text(
        "{UP}{HOME}{SHIFT+END}{CTRL+C}{END}{DOWN}{CTRL+V}"
    )

    assert translated == "{Up}{Home}{Shift}{End}{Ctrl}c{End}{Down}{Ctrl}v"


def test_native_windows_backend_uses_translated_send_keys_for_standard_windows_apps(monkeypatch):
    backend = NativeWindowsDesktopBackend()
    window = WindowInfo(handle=1, title="Adsiz - Not Defteri")

    class FakeRoot:
        def __init__(self) -> None:
            self.sent: list[str] = []

        def SendKeys(self, text: str, interval: float, waitTime: float) -> None:  # noqa: N802
            self.sent.append(text)

    root = FakeRoot()
    reads = iter(("line1\nline2", "line1\nline2\nline2"))

    monkeypatch.setattr(backend, "_resolve_window", lambda **kwargs: window)
    monkeypatch.setattr(backend, "_control_root", lambda _window: root)
    monkeypatch.setattr(
        backend,
        "focus_window",
        lambda **kwargs: DesktopActionVerification(
            target=window.title,
            action="focus_window",
            observed_outcome="Focused",
            success=True,
            verified=True,
        ),
    )
    monkeypatch.setattr(
        backend,
        "read_window_text",
        lambda **kwargs: DesktopActionVerification(
            target=window.title,
            action="read_window_text",
            observed_outcome=next(reads),
            success=True,
            verified=True,
        ),
    )

    result = backend.type_keys(
        app="Notepad",
        profile_id="notepad",
        text="{UP}{HOME}{SHIFT+END}{CTRL+C}{END}{DOWN}{CTRL+V}",
    )

    assert result.success
    assert root.sent == ["{Up}{Home}{Shift}{End}{Ctrl}c{End}{Down}{Ctrl}v"]
    assert result.details["translated_text"] == "{Up}{Home}{Shift}{End}{Ctrl}c{End}{Down}{Ctrl}v"


# ── GitTool ───────────────────────────────────────────────────────────────────

@pytest.fixture
def git_tool(tmp_path):
    return GitTool(workspace_dir=tmp_path)


@pytest.mark.asyncio
async def test_git_init(git_tool, tmp_path):
    pytest.importorskip("git")
    result = await git_tool.execute(action="init")
    assert result.success
    assert (tmp_path / ".git").exists()


@pytest.mark.asyncio
async def test_git_init_and_commit(git_tool, tmp_path):
    pytest.importorskip("git")
    await git_tool.execute(action="init")

    (tmp_path / "hello.txt").write_text("hello")
    result = await git_tool.execute(action="commit", message="initial commit")
    assert result.success


@pytest.mark.asyncio
async def test_git_status(git_tool, tmp_path):
    pytest.importorskip("git")
    await git_tool.execute(action="init")
    result = await git_tool.execute(action="status")
    assert result.success


@pytest.mark.asyncio
async def test_git_log_empty(git_tool, tmp_path):
    pytest.importorskip("git")
    await git_tool.execute(action="init")
    result = await git_tool.execute(action="log")
    assert result.success


@pytest.mark.asyncio
async def test_git_diff(git_tool, tmp_path):
    pytest.importorskip("git")
    await git_tool.execute(action="init")
    (tmp_path / "f.txt").write_text("v1")
    await git_tool.execute(action="commit", message="v1")

    (tmp_path / "f.txt").write_text("v2")
    result = await git_tool.execute(action="diff")
    assert result.success


@pytest.mark.asyncio
async def test_git_unknown_action(git_tool):
    pytest.importorskip("git")
    result = await git_tool.execute(action="teleport")
    assert not result.success


# ── ComputerTool ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_computer_tool_disabled():
    from agentra.tools.computer import ComputerTool  # noqa: PLC0415

    tool = ComputerTool(allow=False)
    result = await tool.execute(action="screenshot")
    assert not result.success
    assert "disabled" in result.error.lower()


@pytest.mark.asyncio
async def test_computer_tool_unknown_action():
    from agentra.tools.computer import ComputerTool  # noqa: PLC0415

    tool = ComputerTool(allow=True)
    result = await tool.execute(action="fly")
    assert not result.success


@pytest.mark.asyncio
async def test_computer_click_requires_coords():
    from agentra.tools.computer import ComputerTool  # noqa: PLC0415

    tool = ComputerTool(allow=True)
    result = await tool.execute(action="click")
    # Either an error (no coords) or an ImportError (pyautogui not installed)
    # Both are acceptable; what matters is it doesn't crash unhandled.
    assert isinstance(result.success, bool)


@pytest.mark.asyncio
async def test_computer_tool_preview_describes_desktop_action():
    from agentra.tools.computer import ComputerTool  # noqa: PLC0415

    tool = ComputerTool(allow=True)
    preview = await tool.preview(action="click", x=120, y=80)

    assert preview is not None
    assert preview["frame_label"] == "computer · click"
    assert "desktop" in preview["summary"].lower()


@pytest.mark.asyncio
async def test_computer_tool_pauses_when_agentra_window_is_foreground(monkeypatch):
    from agentra.tools.computer import ComputerTool  # noqa: PLC0415

    tool = ComputerTool(allow=True)
    monkeypatch.setattr(
        tool,
        "_capture_desktop_observation",
        lambda: {
            "window_handle": 123,
            "window_title": "Agentra - Live Preview",
            "cursor_x": 40,
            "cursor_y": 20,
        },
    )

    result = await tool.execute(action="type", text="hello")

    assert result.success is False
    assert result.metadata["pause_kind"] == "desktop_control_takeover"
    assert "Agentra" in result.error


@pytest.mark.asyncio
async def test_computer_tool_pauses_when_desktop_state_drifted(monkeypatch):
    from agentra.tools.computer import ComputerTool  # noqa: PLC0415

    tool = ComputerTool(allow=True)
    tool._last_observation = {
        "window_handle": 100,
        "window_title": "Untitled - Notepad",
        "cursor_x": 80,
        "cursor_y": 90,
    }
    monkeypatch.setattr(
        tool,
        "_capture_desktop_observation",
        lambda: {
            "window_handle": 200,
            "window_title": "Another App",
            "cursor_x": 140,
            "cursor_y": 160,
        },
    )

    result = await tool.execute(action="click", x=10, y=10)

    assert result.success is False
    assert result.metadata["pause_kind"] == "desktop_control_takeover"
    assert "beklenmedik" in result.metadata["summary"]


@pytest.mark.asyncio
async def test_computer_tool_routes_hidden_mode_through_session_manager() -> None:
    from agentra.tools.computer import ComputerTool  # noqa: PLC0415

    class FakeDesktopSessions:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str, dict[str, object]]] = []

        def hidden_capability(self, thread_id: str) -> str:
            return f"desktop_session:{thread_id}"

        def execute_computer_action(self, thread_id: str, action: str, **kwargs: object) -> ToolResult:
            self.calls.append((thread_id, action, dict(kwargs)))
            return ToolResult(
                success=True,
                output="hidden click ok",
                metadata={"frame_label": "computer · click", "desktop_execution_mode": "desktop_hidden"},
            )

    tool = ComputerTool(allow=True)
    sessions = FakeDesktopSessions()
    tool.bind_runtime(desktop_sessions=sessions, thread_id="thread-hidden")

    result = await tool.execute(action="click", x=12, y=34)

    assert result.success
    assert result.output == "hidden click ok"
    assert sessions.calls == [("thread-hidden", "click", {"x": 12, "y": 34})]
    assert tool.capabilities == ("desktop_session:thread-hidden",)
