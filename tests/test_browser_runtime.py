"""Tests for shared browser runtime sessions."""

from __future__ import annotations

import asyncio
import json
import sys
import types

import pytest

import agentra.browser_runtime as browser_runtime


class FakeMouse:
    def __init__(self) -> None:
        self.moves: list[tuple[float, float, int | None]] = []
        self.down_calls = 0
        self.up_calls = 0

    async def click(self, x: float, y: float) -> None:
        self.clicked = (x, y)

    async def move(self, x: float, y: float, steps: int | None = None) -> None:
        self.moves.append((x, y, steps))

    async def down(self) -> None:
        self.down_calls += 1

    async def up(self) -> None:
        self.up_calls += 1

    async def wheel(self, *, delta_x: float, delta_y: float) -> None:
        self.wheel = (delta_x, delta_y)


class FakeKeyboard:
    async def type(self, text: str) -> None:
        self.typed = text

    async def press(self, key: str) -> None:
        self.pressed = key


class FakeLocator:
    @property
    def first(self) -> "FakeLocator":
        return self

    async def bounding_box(self):
        return {"x": 20.0, "y": 10.0, "width": 40.0, "height": 20.0}


class SlowLocator:
    @property
    def first(self) -> "SlowLocator":
        return self

    async def bounding_box(self):
        await asyncio.sleep(0.05)
        return None


class FakePage:
    def __init__(self) -> None:
        self.viewport_size = {"width": 100.0, "height": 100.0}
        self.url = "about:blank"
        self._title = "Blank"
        self.mouse = FakeMouse()
        self.keyboard = FakeKeyboard()
        self.closed = False
        self.fail_closed_once = False
        self.screenshot_calls: list[dict[str, object]] = []

    async def goto(self, url: str, timeout: int, wait_until: str | None = None) -> None:
        self.url = url
        self._title = f"title:{url}"

    async def title(self) -> str:
        return self._title

    async def screenshot(self, *, type: str, **kwargs) -> bytes:
        self.screenshot_calls.append({"type": type, **kwargs})
        if self.closed or self.fail_closed_once:
            self.fail_closed_once = False
            raise RuntimeError("Target page, context or browser has been closed")
        return b"jpeg" if type == "jpeg" else b"png"

    async def click(self, selector: str, timeout: int) -> None:
        self.clicked_selector = selector

    async def fill(self, selector: str, text: str) -> None:
        self.filled = (selector, text)

    async def inner_text(self, selector: str) -> str:
        return f"text:{selector}"

    async def inner_html(self, selector: str) -> str:
        return f"<div>{selector}</div>"

    async def content(self) -> str:
        return "<body></body>"

    async def go_back(self) -> None:
        self.url = "about:back"
        self._title = "Back"

    async def go_forward(self) -> None:
        self.url = "about:forward"
        self._title = "Forward"

    async def close(self) -> None:
        self.closed = True

    def is_closed(self) -> bool:
        return self.closed

    def locator(self, selector: str) -> FakeLocator:
        return FakeLocator()


class FakeContext:
    def __init__(self) -> None:
        self.pages: list[FakePage] = []

    async def new_page(self) -> FakePage:
        page = FakePage()
        self.pages.append(page)
        return page


class FakeBrowser:
    def __init__(self) -> None:
        self.contexts: list[FakeContext] = []

    async def new_context(self) -> FakeContext:
        context = FakeContext()
        self.contexts.append(context)
        return context


class FakeBrowserRuntime:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.browser_instance = FakeBrowser()
        self.close_calls = 0
        self.identity = kwargs.get("identity", "isolated")
        self.profile_name = kwargs.get("profile_name", "Default")

    async def browser(self) -> FakeBrowser:
        return self.browser_instance

    async def create_context(self) -> FakeContext:
        return await self.browser_instance.new_context()

    async def close(self) -> None:
        self.close_calls += 1
        self.browser_instance = FakeBrowser()
        return None


def test_browser_runtime_prefers_per_user_chrome_from_local_app_data(monkeypatch) -> None:
    expected_executable = browser_runtime._platform_path_for_local_process(
        r"C:\Users\Ariba\AppData\Local\Google\Chrome\Application\chrome.exe"
    )
    expected_user_data = browser_runtime._platform_path_for_local_process(
        r"C:\Users\Ariba\AppData\Local\Google\Chrome\User Data"
    )
    monkeypatch.setattr(
        browser_runtime,
        "_windows_env_value",
        lambda name: {"LOCALAPPDATA": r"C:\Users\Ariba\AppData\Local"}.get(name, ""),
    )
    monkeypatch.setattr(
        browser_runtime,
        "_path_exists",
        lambda path: path
        in {
            expected_executable,
            expected_user_data,
        },
    )

    executable_path, user_data_dir = browser_runtime.BrowserRuntime._resolve_chrome_profile_launch_paths()

    assert executable_path == expected_executable
    assert user_data_dir == expected_user_data


def test_browser_runtime_uses_powershell_windows_env_lookup_in_wsl(monkeypatch) -> None:
    monkeypatch.delenv("LOCALAPPDATA", raising=False)
    monkeypatch.delenv("PROGRAMFILES", raising=False)
    monkeypatch.delenv("PROGRAMFILES(X86)", raising=False)
    monkeypatch.delenv("USERPROFILE", raising=False)
    monkeypatch.setattr(browser_runtime, "_running_in_wsl", lambda: True)

    def fake_powershell_stdout(script: str, *args: str) -> str:
        if "LocalApplicationData" in script:
            return r"C:\Users\Ariba\AppData\Local"
        if "UserProfile" in script:
            return r"C:\Users\Ariba"
        if args and args[0] == "PROGRAMFILES":
            return r"C:\Program Files"
        if args and args[0] == "PROGRAMFILES(X86)":
            return r"C:\Program Files (x86)"
        return ""

    monkeypatch.setattr(browser_runtime, "_powershell_stdout", fake_powershell_stdout)
    expected_executable = browser_runtime._platform_path_for_local_process(
        r"C:\Program Files\Google\Chrome\Application\chrome.exe"
    )
    expected_user_data = browser_runtime._platform_path_for_local_process(
        r"C:\Users\Ariba\AppData\Local\Google\Chrome\User Data"
    )
    monkeypatch.setattr(
        browser_runtime,
        "_path_exists",
        lambda path: path
        in {
            expected_executable,
            expected_user_data,
        },
    )

    executable_path, user_data_dir = browser_runtime.BrowserRuntime._resolve_chrome_profile_launch_paths()

    assert executable_path == expected_executable
    assert user_data_dir == expected_user_data


def test_browser_runtime_reports_checked_locations_when_chrome_profile_lookup_fails(monkeypatch) -> None:
    monkeypatch.setattr(browser_runtime, "_windows_env_value", lambda name: "")
    monkeypatch.setattr(browser_runtime, "_path_exists", lambda path: False)

    with pytest.raises(RuntimeError) as exc_info:
        browser_runtime.BrowserRuntime._resolve_chrome_profile_launch_paths()

    message = str(exc_info.value)
    assert "Checked executables:" in message
    assert "Checked user-data directories:" in message
    assert "/mnt/c/Program Files/Google/Chrome/Application/chrome.exe" in message
    assert "User Data" in message


@pytest.mark.asyncio
async def test_browser_runtime_uses_non_default_launch_clone_for_chrome_profile(
    monkeypatch,
    tmp_path,
) -> None:
    source_root = tmp_path / "Chrome" / "User Data"
    profile_dir = source_root / "Default"
    profile_dir.mkdir(parents=True)
    (source_root / "Local State").write_text("{}", encoding="utf-8")
    (profile_dir / "Preferences").write_text("{}", encoding="utf-8")
    local_app_data = tmp_path / "LocalAppData"

    monkeypatch.setattr(
        browser_runtime.BrowserRuntime,
        "_resolve_chrome_profile_launch_paths",
        classmethod(lambda cls: ("chrome.exe", str(source_root))),
    )
    monkeypatch.setattr(
        browser_runtime,
        "_windows_env_value",
        lambda name: str(local_app_data) if name == "LOCALAPPDATA" else "",
    )

    launcher_calls: dict[str, object] = {}

    class FakePersistentContext:
        async def close(self) -> None:
            return None

    class FakeLauncher:
        async def launch_persistent_context(self, **kwargs):
            launcher_calls.update(kwargs)
            return FakePersistentContext()

    fake_async_api = types.ModuleType("playwright.async_api")

    class FakeAsyncPlaywright:
        def __init__(self) -> None:
            self.chromium = FakeLauncher()

        async def start(self):
            return self

        async def stop(self) -> None:
            return None

    fake_async_api.async_playwright = lambda: FakeAsyncPlaywright()
    monkeypatch.setitem(sys.modules, "playwright.async_api", fake_async_api)
    fake_playwright = types.ModuleType("playwright")
    fake_playwright.async_api = fake_async_api
    monkeypatch.setitem(sys.modules, "playwright", fake_playwright)
    parked_windows: list[str] = []

    async def fake_park_window(self, launch_user_data_dir: str) -> None:
        parked_windows.append(launch_user_data_dir)

    monkeypatch.setattr(browser_runtime.BrowserRuntime, "_park_chrome_profile_window", fake_park_window)

    runtime = browser_runtime.BrowserRuntime(
        identity="chrome_profile",
        profile_name="Default",
        profile_runtime_id="thread-browser",
    )

    await runtime.create_context()

    launch_root = browser_runtime.Path(str(launcher_calls["user_data_dir"]))
    assert launch_root != source_root
    assert "Agentra" in str(launch_root)
    assert launcher_calls["args"][0] == "--profile-directory=Default"
    assert "--window-position=-32000,-32000" in launcher_calls["args"]
    assert "--window-size=1280,720" in launcher_calls["args"]
    assert "--hide-crash-restore-bubble" in launcher_calls["args"]
    assert (launch_root / "Local State").read_text(encoding="utf-8") == "{}"
    assert (launch_root / "Default" / "Preferences").read_text(encoding="utf-8") == "{}"
    assert parked_windows == [str(launch_root)]

    await runtime.close()
    assert launch_root.exists()


def test_browser_runtime_sanitized_launch_clone_skips_session_and_encrypted_state(tmp_path) -> None:
    source_root = tmp_path / "Chrome" / "User Data"
    profile_dir = source_root / "Default"
    sessions_dir = profile_dir / "Sessions"
    sessions_dir.mkdir(parents=True)
    (source_root / "Local State").write_text("{}", encoding="utf-8")
    (profile_dir / "Preferences").write_text(
        json.dumps({"profile": {"exit_type": "Crashed"}}),
        encoding="utf-8",
    )
    (profile_dir / "Bookmarks").write_text("{}", encoding="utf-8")
    (profile_dir / "Login Data").write_text("secret", encoding="utf-8")
    (profile_dir / "Web Data-journal").write_text("secret", encoding="utf-8")
    (profile_dir / "Account Web Data").write_text("secret", encoding="utf-8")
    (profile_dir / "trusted_vault.pb").write_text("secret", encoding="utf-8")
    (sessions_dir / "Session_123").write_text("restore", encoding="utf-8")
    local_app_data = tmp_path / "LocalAppData"

    runtime = browser_runtime.BrowserRuntime(
        identity="chrome_profile",
        profile_name="Default",
        profile_runtime_id="thread-browser",
    )

    original_env_lookup = browser_runtime._windows_env_value
    browser_runtime._windows_env_value = lambda name: str(local_app_data) if name == "LOCALAPPDATA" else ""
    try:
        launch_root, profile_directory = runtime._prepare_chrome_profile_launch_dir(str(source_root), sanitized=True)
    finally:
        browser_runtime._windows_env_value = original_env_lookup

    launch_path = browser_runtime.Path(launch_root)
    cloned_profile = launch_path / profile_directory
    assert (cloned_profile / "Preferences").exists()
    assert not (cloned_profile / "Login Data").exists()
    assert not (cloned_profile / "Web Data-journal").exists()
    assert not (cloned_profile / "Account Web Data").exists()
    assert not (cloned_profile / "trusted_vault.pb").exists()
    assert not (cloned_profile / "Sessions").exists()

    preferences = json.loads((cloned_profile / "Preferences").read_text(encoding="utf-8"))
    assert preferences["profile"]["exit_type"] == "Normal"
    assert preferences["profile"]["exited_cleanly"] is True


def test_browser_runtime_launch_clone_keeps_only_session_sync_targets(tmp_path) -> None:
    source_root = tmp_path / "Chrome" / "User Data"
    profile_dir = source_root / "Default"
    profile_dir.mkdir(parents=True)
    (source_root / "Local State").write_text("{}", encoding="utf-8")
    (profile_dir / "Preferences").write_text("{}", encoding="utf-8")
    (profile_dir / "Secure Preferences").write_text("{}", encoding="utf-8")
    (profile_dir / "Bookmarks").write_text("{}", encoding="utf-8")
    (profile_dir / "History").write_text("history", encoding="utf-8")
    (profile_dir / "Network").mkdir()
    (profile_dir / "Network" / "Cookies").write_text("cookies", encoding="utf-8")
    (profile_dir / "Extensions").mkdir()
    (profile_dir / "Extensions" / "extension.txt").write_text("ext", encoding="utf-8")
    (profile_dir / "Cache").mkdir()
    (profile_dir / "Cache" / "cache.bin").write_text("cache", encoding="utf-8")
    local_app_data = tmp_path / "LocalAppData"

    runtime = browser_runtime.BrowserRuntime(
        identity="chrome_profile",
        profile_name="Default",
        profile_runtime_id="thread-browser",
    )

    original_env_lookup = browser_runtime._windows_env_value
    browser_runtime._windows_env_value = lambda name: str(local_app_data) if name == "LOCALAPPDATA" else ""
    try:
        launch_root, profile_directory = runtime._prepare_chrome_profile_launch_dir(str(source_root), sanitized=False)
    finally:
        browser_runtime._windows_env_value = original_env_lookup

    launch_path = browser_runtime.Path(launch_root)
    cloned_profile = launch_path / profile_directory
    assert (cloned_profile / "Preferences").exists()
    assert (cloned_profile / "Secure Preferences").exists()
    assert (cloned_profile / "Network" / "Cookies").exists()
    assert not (cloned_profile / "Bookmarks").exists()
    assert not (cloned_profile / "History").exists()
    assert not (cloned_profile / "Extensions").exists()
    assert not (cloned_profile / "Cache").exists()


@pytest.mark.asyncio
async def test_browser_runtime_retries_chrome_profile_launch_with_sanitized_clone(
    monkeypatch,
    tmp_path,
) -> None:
    source_root = tmp_path / "Chrome" / "User Data"
    profile_dir = source_root / "Default"
    profile_dir.mkdir(parents=True)
    (source_root / "Local State").write_text("{}", encoding="utf-8")
    (profile_dir / "Preferences").write_text(
        json.dumps({"profile": {"exit_type": "Crashed"}}),
        encoding="utf-8",
    )
    (profile_dir / "Login Data").write_text("secret", encoding="utf-8")
    local_app_data = tmp_path / "LocalAppData"

    monkeypatch.setattr(
        browser_runtime.BrowserRuntime,
        "_resolve_chrome_profile_launch_paths",
        classmethod(lambda cls: ("chrome.exe", str(source_root))),
    )
    monkeypatch.setattr(
        browser_runtime,
        "_windows_env_value",
        lambda name: str(local_app_data) if name == "LOCALAPPDATA" else "",
    )

    launch_calls: list[dict[str, object]] = []

    class FakePersistentContext:
        async def close(self) -> None:
            return None

    class FakeLauncher:
        async def launch_persistent_context(self, **kwargs):
            launch_calls.append(kwargs)
            launch_profile = browser_runtime.Path(str(kwargs["user_data_dir"])) / "Default"
            if len(launch_calls) == 1:
                assert (launch_profile / "Preferences").exists()
                raise RuntimeError(
                    "BrowserType.launch_persistent_context: Protocol error "
                    "(Browser.getWindowForTarget): Browser window not found"
                )
            assert (launch_profile / "Preferences").exists()
            assert not (launch_profile / "Login Data").exists()
            return FakePersistentContext()

    fake_async_api = types.ModuleType("playwright.async_api")

    class FakeAsyncPlaywright:
        def __init__(self) -> None:
            self.chromium = FakeLauncher()

        async def start(self):
            return self

        async def stop(self) -> None:
            return None

    fake_async_api.async_playwright = lambda: FakeAsyncPlaywright()
    monkeypatch.setitem(sys.modules, "playwright.async_api", fake_async_api)
    fake_playwright = types.ModuleType("playwright")
    fake_playwright.async_api = fake_async_api
    monkeypatch.setitem(sys.modules, "playwright", fake_playwright)
    parked_windows: list[str] = []

    async def fake_park_window(self, launch_user_data_dir: str) -> None:
        parked_windows.append(launch_user_data_dir)

    monkeypatch.setattr(browser_runtime.BrowserRuntime, "_park_chrome_profile_window", fake_park_window)

    runtime = browser_runtime.BrowserRuntime(
        identity="chrome_profile",
        profile_name="Default",
        profile_runtime_id="thread-browser",
    )

    await runtime.create_context()

    assert len(launch_calls) == 2
    assert launch_calls[0]["timeout"] == browser_runtime._CHROME_PROFILE_LAUNCH_TIMEOUT_MS
    assert launch_calls[1]["timeout"] == browser_runtime._CHROME_PROFILE_LAUNCH_TIMEOUT_MS
    assert "--hide-crash-restore-bubble" in launch_calls[1]["args"]
    assert parked_windows == [str(launch_calls[1]["user_data_dir"])]


@pytest.mark.asyncio
async def test_browser_session_manager_reuses_session_per_thread(monkeypatch) -> None:
    monkeypatch.setattr(browser_runtime, "BrowserRuntime", FakeBrowserRuntime)
    manager = browser_runtime.BrowserSessionManager()

    first = await manager.session_for(thread_id="thread-a")
    second = await manager.session_for(thread_id="thread-a")

    assert first is second

    result = await first.execute(action="navigate", url="https://example.com")
    assert result.success is True
    snapshot = manager.snapshot("thread-a")
    assert snapshot.active is True
    assert snapshot.active_url == "https://example.com"


@pytest.mark.asyncio
async def test_browser_session_manager_isolates_threads(monkeypatch) -> None:
    monkeypatch.setattr(browser_runtime, "BrowserRuntime", FakeBrowserRuntime)
    manager = browser_runtime.BrowserSessionManager()

    first = await manager.session_for(thread_id="thread-a")
    second = await manager.session_for(thread_id="thread-b")

    await first.execute(action="navigate", url="https://a.example")
    await second.execute(action="navigate", url="https://b.example")

    assert manager.snapshot("thread-a").active_url == "https://a.example"
    assert manager.snapshot("thread-b").active_url == "https://b.example"


@pytest.mark.asyncio
async def test_browser_session_supports_key_actions(monkeypatch) -> None:
    monkeypatch.setattr(browser_runtime, "BrowserRuntime", FakeBrowserRuntime)
    manager = browser_runtime.BrowserSessionManager()

    session = await manager.session_for(thread_id="thread-a")
    result = await session.execute(action="key", key="Enter")

    assert result.success is True
    assert session._page.keyboard.pressed == "Enter"
    assert result.metadata["frame_label"] == "browser · key"
    assert result.metadata["summary"] == "Pressing Enter"


@pytest.mark.asyncio
async def test_browser_session_key_preview_reports_visual_intent(monkeypatch) -> None:
    monkeypatch.setattr(browser_runtime, "BrowserRuntime", FakeBrowserRuntime)
    manager = browser_runtime.BrowserSessionManager()

    session = await manager.session_for(thread_id="thread-a")
    await session.execute(action="navigate", url="https://example.com")
    preview = await session.preview(action="key", key="Tab")

    assert preview is not None
    assert preview["frame_label"] == "browser · key"
    assert preview["summary"] == "Pressing Tab"
    assert preview["focus_x"] == pytest.approx(0.74)
    assert preview["focus_y"] == pytest.approx(0.56)


@pytest.mark.asyncio
async def test_browser_session_preview_fast_falls_back_when_selector_lookup_times_out(monkeypatch) -> None:
    monkeypatch.setattr(browser_runtime, "BrowserRuntime", FakeBrowserRuntime)
    monkeypatch.setattr(browser_runtime, "_SELECTOR_FOCUS_TIMEOUT_SECONDS", 0.01)
    manager = browser_runtime.BrowserSessionManager()

    session = await manager.session_for(thread_id="thread-a")
    await session.execute(action="navigate", url="https://example.com")
    session._page = FakePage()
    session._page.locator = lambda selector: SlowLocator()

    started = asyncio.get_running_loop().time()
    preview = await session.preview(action="click", selector="#missing")
    elapsed = asyncio.get_running_loop().time() - started

    assert preview is not None
    assert elapsed < 0.05
    assert preview["focus_x"] == pytest.approx(0.74)
    assert preview["focus_y"] == pytest.approx(0.2)


@pytest.mark.asyncio
async def test_browser_session_supports_drag_actions(monkeypatch) -> None:
    monkeypatch.setattr(browser_runtime, "BrowserRuntime", FakeBrowserRuntime)
    manager = browser_runtime.BrowserSessionManager()

    session = await manager.session_for(thread_id="thread-a")
    result = await session.execute(action="drag", start_x=10, start_y=12, end_x=72, end_y=68, steps=9)

    assert result.success is True
    assert session._page.mouse.moves == [(10.0, 12.0, None), (72.0, 68.0, 9)]
    assert session._page.mouse.down_calls == 1
    assert session._page.mouse.up_calls == 1
    assert result.metadata["frame_label"] == "browser · drag"
    assert result.metadata["summary"] == "Dragging on the page"


@pytest.mark.asyncio
async def test_browser_session_recovers_after_browser_page_is_closed(monkeypatch) -> None:
    monkeypatch.setattr(browser_runtime, "BrowserRuntime", FakeBrowserRuntime)
    manager = browser_runtime.BrowserSessionManager()

    session = await manager.session_for(thread_id="thread-a")
    await session.execute(action="navigate", url="https://example.com")
    first_page = session._page
    await first_page.close()

    result = await session.execute(action="screenshot")

    assert result.success is True
    assert session._page is not first_page
    assert session.runtime.close_calls == 0


@pytest.mark.asyncio
async def test_browser_live_capture_recovers_after_browser_is_removed(monkeypatch) -> None:
    monkeypatch.setattr(browser_runtime, "BrowserRuntime", FakeBrowserRuntime)
    manager = browser_runtime.BrowserSessionManager()

    session = await manager.session_for(thread_id="thread-a")
    await session.execute(action="navigate", url="https://example.com")
    session._page.fail_closed_once = True

    png = await manager.capture_live_png("thread-a")

    assert png == b"png"
    assert session.runtime.close_calls == 0
    assert session.snapshot().active is True


@pytest.mark.asyncio
async def test_browser_live_frame_prefers_fast_jpeg_capture(monkeypatch) -> None:
    monkeypatch.setattr(browser_runtime, "BrowserRuntime", FakeBrowserRuntime)
    manager = browser_runtime.BrowserSessionManager()

    session = await manager.session_for(thread_id="thread-a")
    await session.execute(action="navigate", url="https://example.com")

    frame = await manager.capture_live_frame("thread-a")

    assert frame is not None
    assert frame.data == b"jpeg"
    assert frame.media_type == "image/jpeg"
    assert session._page.screenshot_calls[-1] == {"type": "jpeg", "quality": 55, "scale": "css"}


@pytest.mark.asyncio
async def test_browser_session_manager_can_cancel_pending_warmup_task() -> None:
    manager = browser_runtime.BrowserSessionManager()
    blocker = asyncio.Event()
    task = asyncio.create_task(blocker.wait())
    manager._warmup_tasks["thread-a"] = task

    manager.cancel_warmup_thread("thread-a")
    await asyncio.sleep(0)

    assert task.cancelled() is True
    assert "thread-a" not in manager._warmup_tasks
