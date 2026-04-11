"""Shared browser runtime and thread-scoped browser sessions."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import json
import logging
import os
import re
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from agentra.tools.base import ToolResult
from agentra.tools.visual_diff import compute_image_hash, images_are_similar

_DEFAULT_FOCUS = (0.74, 0.2)
_LIVE_REFRESH_INTERVALS = (0.2, 0.35)
_WINDOWS_DRIVE_PATTERN = re.compile(r"^([a-zA-Z]):[\\/](.*)$")
_SELECTOR_FOCUS_TIMEOUT_SECONDS = 0.75
_CHROME_PROFILE_ROOT_FILES = ("Local State", "First Run", "Last Version")
_CHROME_PROFILE_SKIP_DIRS = {
    "AutofillStrikeDatabase",
    "blob_storage",
    "Cache",
    "CacheStorage",
    "Code Cache",
    "Crashpad",
    "DawnGraphiteCache",
    "Dictionaries",
    "Extension Rules",
    "Extension Scripts",
    "Extensions",
    "GPUCache",
    "GrShaderCache",
    "Local Extension Settings",
    "Media Cache",
    "OptimizationGuidePredictionModels",
    "ScriptCache",
    "ShaderCache",
    "Sessions",
}
_CHROME_PROFILE_SKIP_NAMES = {
    "CrashpadMetrics-active.pma",
    "DevToolsActivePort",
    "lockfile",
}
_CHROME_PROFILE_SKIP_PREFIXES = ("Singleton",)
_CHROME_PROFILE_SANITIZED_SKIP_DIRS = {
    "Accounts",
    "Sync App Settings",
    "Sync Data",
}
_CHROME_PROFILE_SANITIZED_SKIP_NAMES = {
    "trusted_vault.pb",
}
_CHROME_PROFILE_SANITIZED_SKIP_PREFIXES = (
    "Account Web Data",
    "Affiliation Database",
    "Login Data",
    "MediaDeviceSalts",
    "Web Data",
)
_CHROME_PROFILE_LAUNCH_ARGS = (
    "--hide-crash-restore-bubble",
    "--restore-last-session=false",
)
_CHROME_PROFILE_LAUNCH_TIMEOUT_MS = 30_000
_CHROME_PROFILE_HIDDEN_WINDOW_POSITION = (-32_000, -32_000)
_CHROME_PROFILE_HIDDEN_WINDOW_SIZE = (1_280, 720)
_CHROME_PROFILE_WINDOW_PARK_ATTEMPTS = 8
_CHROME_PROFILE_WINDOW_PARK_RETRY_SECONDS = 0.1
_CHROME_PROFILE_STORAGE_DIRNAME = "Chrome Automation"
_CHROME_PROFILE_CACHE_DIRNAME = "Profile Cache"
_CHROME_PROFILE_LAUNCH_DIRNAME = "Launches"
_CHROME_PROFILE_NATIVE_COPY_OK_CODES = set(range(8))
_CHROME_PROFILE_SYNC_SCHEMA_VERSION = "v2"
_CHROME_PROFILE_SESSION_DIRS = (
    "IndexedDB",
    "Local Storage",
    "Network",
    "Service Worker",
    "Session Storage",
    "WebStorage",
    "shared_proto_db",
)
_CHROME_PROFILE_SESSION_FILES = (
    "Preferences",
    "Secure Preferences",
)

logger = logging.getLogger(__name__)


def _running_in_wsl() -> bool:
    if os.name == "nt":
        return False
    if os.environ.get("WSL_DISTRO_NAME") or os.environ.get("WSL_INTEROP"):
        return True
    version_path = Path("/proc/version")
    if version_path.exists():
        try:
            return "microsoft" in version_path.read_text(encoding="utf-8", errors="ignore").casefold()
        except OSError:
            return False
    return False


def _looks_like_windows_path(path: str) -> bool:
    return bool(_WINDOWS_DRIVE_PATTERN.match(str(path or "").strip()))


def _normalize_windows_path(path: str) -> str:
    value = str(path or "").strip().replace("/", "\\")
    match = _WINDOWS_DRIVE_PATTERN.match(value)
    if match is None:
        return value
    drive, remainder = match.groups()
    remainder = remainder.lstrip("\\")
    return f"{drive.upper()}:\\{remainder}"


def _windows_path_to_wsl_path(path: str) -> str:
    normalized = _normalize_windows_path(path)
    match = _WINDOWS_DRIVE_PATTERN.match(normalized)
    if match is None:
        return normalized
    drive, remainder = match.groups()
    tail = remainder.replace("\\", "/").lstrip("/")
    return f"/mnt/{drive.lower()}/{tail}"


def _wsl_path_to_windows_path(path: str) -> str:
    raw = str(path or "").strip()
    match = re.match(r"^/mnt/([a-zA-Z])/(.*)$", raw)
    if match is None:
        return raw
    drive, remainder = match.groups()
    windows_tail = remainder.replace("/", "\\")
    return f"{drive.upper()}:\\{windows_tail}"


def _windows_native_path(path: str) -> str:
    raw = str(path or "").strip()
    if not raw:
        return ""
    if _looks_like_windows_path(raw):
        return _normalize_windows_path(raw)
    if raw.startswith("/mnt/"):
        return _wsl_path_to_windows_path(raw)
    return raw


def _platform_path_for_local_process(path: str) -> str:
    raw = str(path or "").strip()
    if not raw:
        return ""
    if _looks_like_windows_path(raw):
        return _normalize_windows_path(raw) if os.name == "nt" else _windows_path_to_wsl_path(raw)
    return raw


def _path_exists(path: str) -> bool:
    return bool(path) and Path(path).exists()


def _powershell_stdout(script: str, *args: str) -> str:
    try:
        result = subprocess.run(  # noqa: S603
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script, *args],
            capture_output=True,
            text=True,
            check=False,
            encoding="utf-8",
            errors="ignore",
        )
    except OSError:
        return ""
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def _windows_env_value(name: str) -> str:
    direct = os.environ.get(name)
    if direct:
        return direct.strip()
    if not _running_in_wsl():
        return ""
    lowered = str(name or "").strip().casefold()
    if lowered == "localappdata":
        value = _powershell_stdout("[Environment]::GetFolderPath('LocalApplicationData')")
        if value:
            return value
    if lowered == "userprofile":
        value = _powershell_stdout("[Environment]::GetFolderPath('UserProfile')")
        if value:
            return value
    lookup = (
        "$name = $args[0]; "
        "$value = [Environment]::GetEnvironmentVariable($name, 'Process'); "
        "if (-not $value) { $value = [Environment]::GetEnvironmentVariable($name, 'User') }; "
        "if (-not $value) { $value = [Environment]::GetEnvironmentVariable($name, 'Machine') }; "
        "if ($value) { Write-Output $value }"
    )
    return _powershell_stdout(lookup, name)


def _checked_location_label(raw_path: str) -> str:
    platform_path = _platform_path_for_local_process(raw_path)
    if raw_path and platform_path and raw_path != platform_path:
        return f"{raw_path} -> {platform_path}"
    return raw_path or platform_path


@dataclass
class BrowserSnapshot:
    """Serializable summary of the current browser session."""

    active: bool = False
    active_url: str = ""
    active_title: str = ""
    tab_count: int = 0
    identity: str = "isolated"
    profile_name: str = "Default"
    last_error: str = ""


@dataclass(frozen=True)
class LiveBrowserFrame:
    """Encoded browser frame for the live mirror."""

    data: bytes
    media_type: str = "image/jpeg"


class BrowserRuntime:
    """Owns a Playwright runtime plus a launched browser instance."""

    def __init__(
        self,
        *,
        browser_type: Literal["chromium", "firefox", "webkit"] = "chromium",
        headless: bool = False,
        identity: Literal["isolated", "chrome_profile"] = "isolated",
        profile_name: str = "Default",
        profile_runtime_id: str | None = None,
    ) -> None:
        self.browser_type = browser_type
        self.headless = headless
        self.identity = identity
        self.profile_name = profile_name or "Default"
        self.profile_runtime_id = profile_runtime_id or self.profile_name or "Default"
        self._playwright: Any = None
        self._browser: Any = None
        self._persistent_context: Any = None
        self._profile_clone_dir: Path | None = None
        self._profile_cache_dir: Path | None = None
        self._lock = asyncio.Lock()

    async def browser(self) -> Any:
        async with self._lock:
            if self._browser is not None:
                return self._browser
            from playwright.async_api import async_playwright  # noqa: PLC0415

            self._playwright = await async_playwright().start()
            launcher = getattr(self._playwright, self.browser_type)
            self._browser = await launcher.launch(headless=self.headless)
            return self._browser

    async def create_context(self) -> Any:
        async with self._lock:
            from playwright.async_api import async_playwright  # noqa: PLC0415

            if self.identity == "chrome_profile":
                if self._persistent_context is not None:
                    return self._persistent_context
                if self._playwright is None:
                    self._playwright = await async_playwright().start()
                launcher = getattr(self._playwright, self.browser_type)
                executable_path, source_user_data_dir = self._resolve_chrome_profile_launch_paths()
                try:
                    launch_user_data_dir, profile_directory = self._prepare_chrome_profile_launch_dir(
                        source_user_data_dir
                    )
                    self._persistent_context = await self._launch_chrome_profile_context(
                        launcher=launcher,
                        executable_path=executable_path,
                        source_user_data_dir=source_user_data_dir,
                        launch_user_data_dir=launch_user_data_dir,
                        profile_directory=profile_directory,
                    )
                except Exception:
                    self._cleanup_profile_clone_dir()
                    raise
                return self._persistent_context

            if self._browser is None:
                if self._playwright is None:
                    self._playwright = await async_playwright().start()
                launcher = getattr(self._playwright, self.browser_type)
                self._browser = await launcher.launch(headless=self.headless)
            return await self._browser.new_context()

    async def close(self) -> None:
        async with self._lock:
            if self._persistent_context is not None:
                await self._persistent_context.close()
            if self._browser is not None:
                await self._browser.close()
            if self._playwright is not None:
                await self._playwright.stop()
            self._persistent_context = None
            self._browser = None
            self._playwright = None
            self._cleanup_profile_clone_dir()

    @classmethod
    def _resolve_chrome_profile_launch_paths(cls) -> tuple[str, str]:
        executable_path, checked_executables = cls._first_existing_candidate(cls._chrome_executable_candidates())
        user_data_dir, checked_user_data = cls._first_existing_candidate(cls._chrome_user_data_candidates())
        if executable_path and user_data_dir:
            return executable_path, user_data_dir
        raise RuntimeError(cls._chrome_profile_error_message(checked_executables, checked_user_data))

    @classmethod
    def _chrome_executable_candidates(cls) -> list[str]:
        local_app_data = _windows_env_value("LOCALAPPDATA")
        program_files = _windows_env_value("PROGRAMFILES")
        program_files_x86 = _windows_env_value("PROGRAMFILES(X86)")
        return cls._dedupe_candidates(
            [
                str(Path(local_app_data) / "Google" / "Chrome" / "Application" / "chrome.exe")
                if local_app_data
                else "",
                str(Path(program_files) / "Google" / "Chrome" / "Application" / "chrome.exe")
                if program_files
                else "",
                str(Path(program_files_x86) / "Google" / "Chrome" / "Application" / "chrome.exe")
                if program_files_x86
                else "",
                "/mnt/c/Program Files/Google/Chrome/Application/chrome.exe",
                "/mnt/c/Program Files (x86)/Google/Chrome/Application/chrome.exe",
            ]
        )

    @classmethod
    def _chrome_user_data_candidates(cls) -> list[str]:
        local_app_data = _windows_env_value("LOCALAPPDATA")
        user_profile = _windows_env_value("USERPROFILE")
        return cls._dedupe_candidates(
            [
                str(Path(local_app_data) / "Google" / "Chrome" / "User Data") if local_app_data else "",
                str(Path(user_profile) / "AppData" / "Local" / "Google" / "Chrome" / "User Data")
                if user_profile
                else "",
                f"/mnt/c/Users/{Path.home().name}/AppData/Local/Google/Chrome/User Data",
            ]
        )

    @classmethod
    def _first_existing_candidate(cls, candidates: list[str]) -> tuple[str, list[str]]:
        checked_locations: list[str] = []
        for raw_path in candidates:
            checked_locations.append(_checked_location_label(raw_path))
            platform_path = _platform_path_for_local_process(raw_path)
            if _path_exists(platform_path):
                return platform_path, checked_locations
        return "", checked_locations

    @staticmethod
    def _dedupe_candidates(candidates: list[str]) -> list[str]:
        deduped: list[str] = []
        seen: set[str] = set()
        for raw_path in candidates:
            if not raw_path:
                continue
            key = raw_path.casefold()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(raw_path)
        return deduped

    @staticmethod
    def _chrome_profile_error_message(
        checked_executables: list[str],
        checked_user_data: list[str],
    ) -> str:
        executable_text = ", ".join(checked_executables) if checked_executables else "none"
        user_data_text = ", ".join(checked_user_data) if checked_user_data else "none"
        return (
            "Chrome profile mode is unavailable because the Chrome executable or user-data directory "
            f"could not be resolved. Checked executables: {executable_text}. "
            f"Checked user-data directories: {user_data_text}."
        )

    async def _launch_chrome_profile_context(
        self,
        *,
        launcher: Any,
        executable_path: str,
        source_user_data_dir: str,
        launch_user_data_dir: str,
        profile_directory: str,
    ) -> Any:
        launch_started = time.perf_counter()
        launch_kwargs = {
            "user_data_dir": launch_user_data_dir,
            "executable_path": executable_path,
            "headless": False,
            "args": self._chrome_profile_launch_args(profile_directory),
            "timeout": _CHROME_PROFILE_LAUNCH_TIMEOUT_MS,
        }
        try:
            context = await launcher.launch_persistent_context(**launch_kwargs)
            await self._park_chrome_profile_window(launch_user_data_dir)
            logger.info(
                "Chrome profile launch ready runtime=%s profile=%s launch_ms=%.1f",
                self.profile_runtime_id,
                profile_directory,
                (time.perf_counter() - launch_started) * 1000,
            )
            return context
        except Exception as exc:  # noqa: BLE001
            if not self._should_retry_with_sanitized_profile(exc):
                raise
            self._cleanup_profile_clone_dir()
            sanitized_user_data_dir, sanitized_profile_directory = self._prepare_chrome_profile_launch_dir(
                source_user_data_dir,
                sanitized=True,
            )
            context = await launcher.launch_persistent_context(
                user_data_dir=sanitized_user_data_dir,
                executable_path=executable_path,
                headless=False,
                args=self._chrome_profile_launch_args(sanitized_profile_directory),
                timeout=_CHROME_PROFILE_LAUNCH_TIMEOUT_MS,
            )
            await self._park_chrome_profile_window(sanitized_user_data_dir)
            logger.info(
                "Chrome profile launch ready after sanitize runtime=%s profile=%s launch_ms=%.1f",
                self.profile_runtime_id,
                sanitized_profile_directory,
                (time.perf_counter() - launch_started) * 1000,
            )
            return context

    @staticmethod
    def _chrome_profile_launch_args(profile_directory: str) -> list[str]:
        x, y = _CHROME_PROFILE_HIDDEN_WINDOW_POSITION
        width, height = _CHROME_PROFILE_HIDDEN_WINDOW_SIZE
        return [
            f"--profile-directory={profile_directory}",
            f"--window-position={x},{y}",
            f"--window-size={width},{height}",
            *_CHROME_PROFILE_LAUNCH_ARGS,
        ]

    async def _park_chrome_profile_window(self, launch_user_data_dir: str) -> None:
        native_path = _windows_native_path(launch_user_data_dir)
        if not native_path or not (_running_in_wsl() or os.name == "nt"):
            return
        for attempt in range(_CHROME_PROFILE_WINDOW_PARK_ATTEMPTS):
            moved_count = await asyncio.to_thread(self._park_chrome_profile_window_once, native_path)
            if moved_count > 0:
                logger.info(
                    "Chrome profile window parked runtime=%s moved_windows=%s",
                    self.profile_runtime_id,
                    moved_count,
                )
                return
            if attempt + 1 < _CHROME_PROFILE_WINDOW_PARK_ATTEMPTS:
                await asyncio.sleep(_CHROME_PROFILE_WINDOW_PARK_RETRY_SECONDS)
        logger.debug(
            "Chrome profile window parking skipped runtime=%s user_data_dir=%s",
            self.profile_runtime_id,
            native_path,
        )

    @staticmethod
    def _park_chrome_profile_window_once(launch_user_data_dir: str) -> int:
        x, y = _CHROME_PROFILE_HIDDEN_WINDOW_POSITION
        width, height = _CHROME_PROFILE_HIDDEN_WINDOW_SIZE
        script = rf"""
$launchDir = $args[0]
if (-not $launchDir) {{
    Write-Output 0
    exit 0
}}
$escaped = [Management.Automation.WildcardPattern]::Escape($launchDir)
$processes = @(
    Get-CimInstance Win32_Process -Filter "name = 'chrome.exe'" |
        Where-Object {{ $_.CommandLine -and $_.CommandLine -like "*$escaped*" }}
)
if (-not ("Agentra.ChromeWindowParker" -as [type])) {{
    Add-Type -Namespace Agentra -Name ChromeWindowParker -MemberDefinition @"
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;

public static class ChromeWindowParker
{{
    private delegate bool EnumWindowsProc(IntPtr hWnd, IntPtr lParam);

    [DllImport("user32.dll")]
    private static extern bool EnumWindows(EnumWindowsProc lpEnumFunc, IntPtr lParam);

    [DllImport("user32.dll")]
    private static extern uint GetWindowThreadProcessId(IntPtr hWnd, out uint processId);

    [DllImport("user32.dll")]
    private static extern bool IsWindowVisible(IntPtr hWnd);

    [DllImport("user32.dll")]
    private static extern bool SetWindowPos(
        IntPtr hWnd,
        IntPtr hWndInsertAfter,
        int x,
        int y,
        int cx,
        int cy,
        uint flags
    );

    public static int ParkWindows(uint[] processIds, int x, int y, int width, int height)
    {{
        HashSet<uint> targets = new HashSet<uint>(processIds ?? Array.Empty<uint>());
        if (targets.Count == 0)
        {{
            return 0;
        }}
        const uint SWP_NOZORDER = 0x0004;
        const uint SWP_NOACTIVATE = 0x0010;
        const uint SWP_SHOWWINDOW = 0x0040;
        int moved = 0;
        EnumWindows(delegate (IntPtr hWnd, IntPtr lParam)
        {{
            GetWindowThreadProcessId(hWnd, out uint processId);
            if (!targets.Contains(processId) || !IsWindowVisible(hWnd))
            {{
                return true;
            }}
            if (SetWindowPos(hWnd, IntPtr.Zero, x, y, width, height, SWP_NOZORDER | SWP_NOACTIVATE | SWP_SHOWWINDOW))
            {{
                moved++;
            }}
            return true;
        }}, IntPtr.Zero);
        return moved;
    }}
}}
"@
}}
$processIds = @($processes | ForEach-Object {{ [uint32]$_.ProcessId }})
$moved = [Agentra.ChromeWindowParker]::ParkWindows($processIds, {x}, {y}, {width}, {height})
Write-Output $moved
"""
        try:
            result = subprocess.run(  # noqa: S603
                ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script, launch_user_data_dir],
                capture_output=True,
                text=True,
                check=False,
                timeout=5,
                encoding="utf-8",
                errors="ignore",
            )
        except (OSError, subprocess.TimeoutExpired):
            return 0
        if result.returncode != 0:
            return 0
        try:
            return int((result.stdout or "").strip() or "0")
        except ValueError:
            return 0

    @staticmethod
    def _should_retry_with_sanitized_profile(exc: Exception) -> bool:
        message = str(exc or "").casefold()
        if not message:
            return False
        if "browser window not found" in message:
            return True
        if "browser.getwindowfortarget" in message:
            return True
        if "failed to decrypt" in message:
            return True
        if "os_crypt" in message:
            return True
        return False

    def _prepare_chrome_profile_launch_dir(
        self,
        source_user_data_dir: str,
        *,
        sanitized: bool = False,
    ) -> tuple[str, str]:
        source_root = Path(_platform_path_for_local_process(source_user_data_dir))
        if not source_root.exists():
            raise RuntimeError(
                "Chrome profile mode is unavailable because the source user-data directory "
                f"does not exist: {source_root}."
            )
        profile_source_dir = self._resolve_profile_source_dir(source_root)
        storage_root = self._chrome_profile_storage_root(source_root)
        cache_root = self._chrome_profile_cache_root(storage_root)
        launch_root = self._chrome_profile_launch_root(storage_root)
        cache_root.mkdir(parents=True, exist_ok=True)
        launch_root.mkdir(parents=True, exist_ok=True)

        source_to_cache_started = time.perf_counter()
        copied_root_files = self._copy_root_files_incremental(source_root, cache_root)
        copied_profile_files, copy_errors = self._mirror_profile_tree(
            profile_source_dir,
            cache_root / profile_source_dir.name,
            sanitized=sanitized,
        )
        logger.info(
            "Chrome profile cache prepared runtime=%s profile=%s source_to_cache_ms=%.1f copied_root=%s copied_profile=%s sanitized=%s",
            self.profile_runtime_id,
            profile_source_dir.name,
            (time.perf_counter() - source_to_cache_started) * 1000,
            copied_root_files,
            copied_profile_files,
            sanitized,
        )

        launch_copy_started = time.perf_counter()
        launch_root_files = self._copy_root_files_incremental(cache_root, launch_root)
        copied_launch_files, launch_errors = self._mirror_profile_tree(
            cache_root / profile_source_dir.name,
            launch_root / profile_source_dir.name,
            sanitized=sanitized,
        )
        logger.info(
            "Chrome profile launch dir prepared runtime=%s profile=%s cache_to_launch_ms=%.1f copied_root=%s copied_profile=%s sanitized=%s",
            self.profile_runtime_id,
            profile_source_dir.name,
            (time.perf_counter() - launch_copy_started) * 1000,
            launch_root_files,
            copied_launch_files,
            sanitized,
        )

        prepared_profile_dir = launch_root / profile_source_dir.name
        if not prepared_profile_dir.exists() or not (prepared_profile_dir / "Preferences").exists():
            error_details = "; ".join(copy_errors[:5]) if copy_errors else "no readable profile files were copied"
            raise RuntimeError(
                "Chrome profile mode is unavailable because Agentra could not prepare a non-default "
                f"launch profile for {profile_source_dir.name!r}. Source: {profile_source_dir}. "
                f"Launch dir: {launch_root}. Details: {error_details}."
            )

        if launch_errors:
            logger.debug(
                "Chrome profile launch mirror reported recoverable copy errors runtime=%s errors=%s",
                self.profile_runtime_id,
                launch_errors[:8],
            )
        self._clean_transient_launch_state(launch_root)
        self._normalize_chrome_profile_launch_state(launch_root / profile_source_dir.name)
        self._profile_clone_dir = launch_root
        self._profile_cache_dir = cache_root
        return str(launch_root), profile_source_dir.name

    def _resolve_profile_source_dir(self, source_root: Path) -> Path:
        direct = source_root / self.profile_name
        if direct.exists():
            return direct
        for child in source_root.iterdir():
            if child.is_dir() and child.name.casefold() == self.profile_name.casefold():
                return child
        available_profiles = ", ".join(
            child.name
            for child in sorted(source_root.iterdir(), key=lambda item: item.name.casefold())
            if child.is_dir()
        )
        raise RuntimeError(
            "Chrome profile mode is unavailable because the requested profile directory "
            f"{self.profile_name!r} was not found under {source_root}. "
            f"Available profile directories: {available_profiles or 'none'}."
        )

    def _chrome_profile_storage_root(self, source_root: Path) -> Path:
        local_app_data = _windows_env_value("LOCALAPPDATA")
        if local_app_data:
            platform_root = _platform_path_for_local_process(str(Path(local_app_data) / "Agentra" / _CHROME_PROFILE_STORAGE_DIRNAME))
            if platform_root:
                return Path(platform_root)
        fallback_base = source_root.parents[2] if len(source_root.parents) >= 3 else source_root.parent
        return fallback_base / "Agentra" / _CHROME_PROFILE_STORAGE_DIRNAME

    def _chrome_profile_cache_root(self, storage_root: Path) -> Path:
        return storage_root / _CHROME_PROFILE_CACHE_DIRNAME / _CHROME_PROFILE_SYNC_SCHEMA_VERSION / self._profile_cache_slug()

    def _chrome_profile_launch_root(self, storage_root: Path) -> Path:
        return (
            storage_root
            / _CHROME_PROFILE_LAUNCH_DIRNAME
            / _CHROME_PROFILE_SYNC_SCHEMA_VERSION
            / self._profile_runtime_slug()
        )

    def _profile_runtime_slug(self) -> str:
        raw = str(self.profile_runtime_id or self.profile_name or "profile")
        compact = re.sub(r"[^a-z0-9]+", "-", raw.casefold()).strip("-")
        digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10]
        return compact[:48] + f"-{digest}" if compact else f"profile-{digest}"

    def _profile_cache_slug(self) -> str:
        raw = str(self.profile_name or "profile")
        compact = re.sub(r"[^a-z0-9]+", "-", raw.casefold()).strip("-")
        digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10]
        return compact[:48] + f"-{digest}" if compact else f"profile-{digest}"

    def _cleanup_profile_clone_dir(self) -> None:
        if self._profile_clone_dir is None:
            return
        self._profile_clone_dir = None
        self._profile_cache_dir = None

    @staticmethod
    def _copy_file_incremental(source: Path, destination: Path) -> bool:
        if not source.exists() or not source.is_file():
            return False
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            if destination.exists():
                source_stat = source.stat()
                destination_stat = destination.stat()
                if (
                    source_stat.st_size == destination_stat.st_size
                    and int(source_stat.st_mtime) == int(destination_stat.st_mtime)
                ):
                    return False
            shutil.copy2(source, destination)
        except OSError:
            return False
        return True

    def _copy_root_files_incremental(self, source_root: Path, destination_root: Path) -> int:
        copied = 0
        for filename in _CHROME_PROFILE_ROOT_FILES:
            copied += int(self._copy_file_incremental(source_root / filename, destination_root / filename))
        return copied

    def _mirror_profile_tree(
        self,
        source: Path,
        destination: Path,
        *,
        sanitized: bool = False,
    ) -> tuple[int, list[str]]:
        if source.exists():
            destination.mkdir(parents=True, exist_ok=True)
        copied = 0
        errors: list[str] = []
        expected_names = set()

        for filename in _CHROME_PROFILE_SESSION_FILES:
            expected_names.add(filename.casefold())
            source_file = source / filename
            dest_file = destination / filename
            if self._copy_file_incremental(source_file, dest_file):
                copied += 1
            elif dest_file.exists() and not source_file.exists():
                try:
                    dest_file.unlink()
                except OSError as exc:
                    errors.append(f"{dest_file.name}: {exc}")

        for dirname in _CHROME_PROFILE_SESSION_DIRS:
            expected_names.add(dirname.casefold())
            source_dir = source / dirname
            dest_dir = destination / dirname
            if not source_dir.exists():
                if dest_dir.exists():
                    shutil.rmtree(dest_dir, ignore_errors=True)
                continue
            native_errors = self._mirror_with_robocopy(source_dir, dest_dir, sanitized=sanitized)
            if native_errors is None:
                dir_copied, dir_errors = self._copy_profile_tree_incremental(source_dir, dest_dir, sanitized=sanitized)
                copied += dir_copied
                errors.extend(dir_errors)
            else:
                copied += 1
                errors.extend(native_errors)
            self._prune_skipped_entries(dest_dir, sanitized=sanitized)

        self._prune_non_session_entries(destination, expected_names=expected_names)
        return copied, errors

    @staticmethod
    def _prune_non_session_entries(root: Path, *, expected_names: set[str]) -> None:
        if not root.exists():
            return
        for path in root.iterdir():
            if path.name.casefold() in expected_names:
                continue
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            else:
                try:
                    path.unlink()
                except OSError:
                    continue

    def _mirror_with_robocopy(
        self,
        source: Path,
        destination: Path,
        *,
        sanitized: bool,
    ) -> list[str] | None:
        source_path = _windows_native_path(str(source))
        destination_path = _windows_native_path(str(destination))
        if not source_path or not destination_path:
            return None
        executable = shutil.which("robocopy.exe") or shutil.which("robocopy")
        if not executable:
            return None
        command = [
            executable,
            source_path,
            destination_path,
            "/MIR",
            "/XJ",
            "/R:1",
            "/W:1",
            "/NFL",
            "/NDL",
            "/NJH",
            "/NJS",
            "/NP",
        ]
        skip_dirs = sorted(_CHROME_PROFILE_SKIP_DIRS)
        if sanitized:
            skip_dirs.extend(sorted(_CHROME_PROFILE_SANITIZED_SKIP_DIRS))
        if skip_dirs:
            command.append("/XD")
            command.extend(skip_dirs)
        skip_names = sorted(_CHROME_PROFILE_SKIP_NAMES)
        if sanitized:
            skip_names.extend(sorted(_CHROME_PROFILE_SANITIZED_SKIP_NAMES))
        if skip_names:
            command.append("/XF")
            command.extend(skip_names)
        try:
            result = subprocess.run(  # noqa: S603
                command,
                capture_output=True,
                text=True,
                check=False,
                encoding="utf-8",
                errors="ignore",
            )
        except OSError as exc:
            return [str(exc)]
        if result.returncode in _CHROME_PROFILE_NATIVE_COPY_OK_CODES:
            return []
        message = result.stdout.strip() or result.stderr.strip() or f"robocopy exit code {result.returncode}"
        return [message]

    @classmethod
    def _copy_profile_tree_incremental(
        cls,
        source: Path,
        destination: Path,
        *,
        sanitized: bool = False,
    ) -> tuple[int, list[str]]:
        copied_files = 0
        errors: list[str] = []
        expected_files: set[Path] = set()
        expected_dirs: set[Path] = {destination}
        for root, dirs, files in os.walk(source):
            root_path = Path(root)
            dirs[:] = [
                name
                for name in dirs
                if not cls._should_skip_chrome_copy_entry(name, is_dir=True, sanitized=sanitized)
            ]
            relative_root = root_path.relative_to(source)
            target_root = destination / relative_root
            target_root.mkdir(parents=True, exist_ok=True)
            expected_dirs.add(target_root)
            for filename in files:
                if cls._should_skip_chrome_copy_entry(filename, is_dir=False, sanitized=sanitized):
                    continue
                source_file = root_path / filename
                destination_file = target_root / filename
                expected_files.add(destination_file)
                try:
                    copied_files += int(cls._copy_file_incremental(source_file, destination_file))
                except OSError as exc:
                    errors.append(f"{source_file.name}: {exc}")
        for dest_file in sorted(destination.rglob("*"), reverse=True):
            if dest_file.is_file() and dest_file not in expected_files:
                try:
                    dest_file.unlink()
                except OSError:
                    continue
            elif dest_file.is_dir() and dest_file not in expected_dirs:
                shutil.rmtree(dest_file, ignore_errors=True)
        return copied_files, errors

    @staticmethod
    def _copy_file_if_present(source: Path, destination: Path) -> bool:
        if not source.exists() or not source.is_file():
            return False
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(source, destination)
        except OSError:
            return False
        return True

    @classmethod
    def _copy_profile_tree(
        cls,
        source: Path,
        destination: Path,
        *,
        sanitized: bool = False,
    ) -> tuple[int, list[str]]:
        copied_files = 0
        errors: list[str] = []
        for root, dirs, files in os.walk(source):
            root_path = Path(root)
            dirs[:] = [
                name
                for name in dirs
                if not cls._should_skip_chrome_copy_entry(name, is_dir=True, sanitized=sanitized)
            ]
            relative_root = root_path.relative_to(source)
            target_root = destination / relative_root
            target_root.mkdir(parents=True, exist_ok=True)
            for filename in files:
                if cls._should_skip_chrome_copy_entry(filename, is_dir=False, sanitized=sanitized):
                    continue
                source_file = root_path / filename
                destination_file = target_root / filename
                try:
                    shutil.copy2(source_file, destination_file)
                    copied_files += 1
                except OSError as exc:
                    errors.append(f"{source_file.name}: {exc}")
        return copied_files, errors

    @classmethod
    def _should_skip_chrome_copy_entry(cls, name: str, *, is_dir: bool, sanitized: bool = False) -> bool:
        lowered = str(name or "").casefold()
        if not lowered:
            return True
        if lowered in {item.casefold() for item in _CHROME_PROFILE_SKIP_NAMES}:
            return True
        if any(lowered.startswith(prefix.casefold()) for prefix in _CHROME_PROFILE_SKIP_PREFIXES):
            return True
        if is_dir and lowered in {item.casefold() for item in _CHROME_PROFILE_SKIP_DIRS}:
            return True
        if sanitized:
            if lowered in {item.casefold() for item in _CHROME_PROFILE_SANITIZED_SKIP_NAMES}:
                return True
            if any(lowered.startswith(prefix.casefold()) for prefix in _CHROME_PROFILE_SANITIZED_SKIP_PREFIXES):
                return True
            if is_dir and lowered in {item.casefold() for item in _CHROME_PROFILE_SANITIZED_SKIP_DIRS}:
                return True
        return False

    @classmethod
    def _prune_skipped_entries(cls, root: Path, *, sanitized: bool) -> None:
        if not root.exists():
            return
        for path in sorted(root.rglob("*"), reverse=True):
            name = path.name
            should_skip = cls._should_skip_chrome_copy_entry(name, is_dir=path.is_dir(), sanitized=sanitized)
            if should_skip:
                if path.is_dir():
                    shutil.rmtree(path, ignore_errors=True)
                else:
                    try:
                        path.unlink()
                    except OSError:
                        continue

    @staticmethod
    def _clean_transient_launch_state(launch_root: Path) -> None:
        for name in _CHROME_PROFILE_SKIP_NAMES:
            path = launch_root / name
            if path.exists():
                try:
                    path.unlink()
                except OSError:
                    pass
        for prefix in _CHROME_PROFILE_SKIP_PREFIXES:
            for path in launch_root.glob(f"{prefix}*"):
                if path.is_dir():
                    shutil.rmtree(path, ignore_errors=True)
                else:
                    try:
                        path.unlink()
                    except OSError:
                        pass

    @staticmethod
    def _normalize_chrome_profile_launch_state(profile_dir: Path) -> None:
        preferences_path = profile_dir / "Preferences"
        if not preferences_path.exists():
            return
        try:
            preferences = json.loads(preferences_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        changed = False
        profile = preferences.get("profile")
        if isinstance(profile, dict):
            if profile.get("exit_type") != "Normal":
                profile["exit_type"] = "Normal"
                changed = True
            if profile.get("exited_cleanly") is not True:
                profile["exited_cleanly"] = True
                changed = True
        if preferences.get("exit_type") == "Crashed":
            preferences["exit_type"] = "Normal"
            changed = True
        if preferences.get("exited_cleanly") is False:
            preferences["exited_cleanly"] = True
            changed = True
        if not changed:
            return
        try:
            preferences_path.write_text(
                json.dumps(preferences, ensure_ascii=True, separators=(",", ":")),
                encoding="utf-8",
            )
        except OSError:
            return


class BrowserSession:
    """A single thread-scoped browser context shared by agent and user."""

    def __init__(
        self,
        thread_id: str,
        runtime: BrowserRuntime,
        screenshot_format: str = "png",
        screenshot_quality: int = 85,
    ) -> None:
        self.thread_id = thread_id
        self.runtime = runtime
        self._screenshot_format = screenshot_format
        self._screenshot_quality = screenshot_quality
        self._context: Any = None
        self._pages: list[Any] = []
        self._page: Any = None
        self._last_screenshot_hash: str | None = None
        self._frame_context: Any = None
        self._last_dialog: dict[str, str] | None = None
        self._pending_dialog: Any = None
        self._snapshot = BrowserSnapshot(identity=runtime.identity, profile_name=runtime.profile_name)
        self._start_lock = asyncio.Lock()

    async def execute(self, **kwargs: Any) -> ToolResult:
        action: str = kwargs.get("action", "")
        capture_result_screenshot = bool(kwargs.get("capture_result_screenshot", True))
        capture_follow_up_screenshots = bool(
            kwargs.get("capture_follow_up_screenshots", capture_result_screenshot)
        )
        for attempt in range(2):
            try:
                await self._ensure_started()
                timeout = kwargs.get("timeout")
                if action == "navigate":
                    return await self._navigate(kwargs.get("url", ""), timeout=timeout)
                if action == "click":
                    return await self._click(
                        kwargs.get("selector"),
                        kwargs.get("x"),
                        kwargs.get("y"),
                        timeout=timeout,
                        capture_result_screenshot=capture_result_screenshot,
                        capture_follow_up_screenshots=capture_follow_up_screenshots,
                    )
                if action == "type":
                    return await self._type(
                        kwargs.get("selector"),
                        kwargs.get("text", ""),
                        timeout=timeout,
                        capture_result_screenshot=capture_result_screenshot,
                        capture_follow_up_screenshots=capture_follow_up_screenshots,
                    )
                if action == "key":
                    return await self._key(
                        kwargs.get("key", ""),
                        capture_result_screenshot=capture_result_screenshot,
                        capture_follow_up_screenshots=capture_follow_up_screenshots,
                    )
                if action == "drag":
                    return await self._drag(
                        kwargs.get("start_x"),
                        kwargs.get("start_y"),
                        kwargs.get("end_x"),
                        kwargs.get("end_y"),
                        kwargs.get("steps", 14),
                        capture_result_screenshot=capture_result_screenshot,
                        capture_follow_up_screenshots=capture_follow_up_screenshots,
                    )
                if action == "scroll":
                    delta_y = kwargs.get("delta_y", kwargs.get("amount", 500))
                    return await self._scroll(
                        kwargs.get("x", 0),
                        kwargs.get("y", 0),
                        delta_y,
                        capture_result_screenshot=capture_result_screenshot,
                        capture_follow_up_screenshots=capture_follow_up_screenshots,
                    )
                if action == "screenshot":
                    return await self._screenshot()
                if action == "get_text":
                    return await self._get_text(kwargs.get("selector"), timeout=timeout)
                if action == "get_html":
                    return await self._get_html(kwargs.get("selector"), timeout=timeout)
                if action == "wait":
                    await asyncio.sleep(kwargs.get("timeout", 1000) / 1000)
                    await self._refresh_snapshot()
                    return self._plain_result("Waited.")
                if action == "back":
                    await self._page.go_back()
                    return await self._action_result(
                        output="Navigated back.",
                        frame_label="browser · back",
                        summary="Going back",
                        focus=self._default_focus(),
                        burst=True,
                        capture_result_screenshot=capture_result_screenshot,
                        capture_follow_up_screenshots=capture_follow_up_screenshots,
                    )
                if action == "forward":
                    await self._page.go_forward()
                    return await self._action_result(
                        output="Navigated forward.",
                        frame_label="browser · forward",
                        summary="Going forward",
                        focus=self._default_focus(),
                        burst=True,
                        capture_result_screenshot=capture_result_screenshot,
                        capture_follow_up_screenshots=capture_follow_up_screenshots,
                    )
                if action == "new_tab":
                    page = await self._context.new_page()
                    page.on("dialog", self._on_dialog)
                    self._set_active_page(page)
                    return await self._capture_visual_state(
                        output="Opened new tab.",
                        frame_label="browser · new_tab",
                        summary="Opening a new tab",
                        focus=self._default_focus(),
                        burst=True,
                    )
                if action == "close_tab":
                    if self._page is not None:
                        await self._page.close()
                        self._pages = [page for page in self._pages if not self._page_is_closed(page)]
                    if not self._pages:
                        page = await self._context.new_page()
                        self._pages = [page]
                    self._set_active_page(self._pages[-1])
                    return await self._capture_visual_state(
                        output="Closed tab.",
                        frame_label="browser · close_tab",
                        summary="Closing the tab",
                        focus=self._default_focus(),
                        burst=True,
                    )
                if action == "hover":
                    return await self._hover(
                        kwargs.get("selector"), kwargs.get("x"), kwargs.get("y"),
                        timeout=timeout,
                        capture_result_screenshot=capture_result_screenshot,
                        capture_follow_up_screenshots=capture_follow_up_screenshots,
                    )
                if action == "double_click":
                    return await self._double_click(
                        kwargs.get("selector"), kwargs.get("x"), kwargs.get("y"),
                        timeout=timeout,
                        capture_result_screenshot=capture_result_screenshot,
                        capture_follow_up_screenshots=capture_follow_up_screenshots,
                    )
                if action == "right_click":
                    return await self._right_click(
                        kwargs.get("selector"), kwargs.get("x"), kwargs.get("y"),
                        timeout=timeout,
                        capture_result_screenshot=capture_result_screenshot,
                        capture_follow_up_screenshots=capture_follow_up_screenshots,
                    )
                if action == "select_option":
                    return await self._select_option(
                        kwargs.get("selector", ""),
                        value=kwargs.get("value"), label=kwargs.get("label"),
                        option_index=kwargs.get("option_index"), timeout=timeout,
                        capture_result_screenshot=capture_result_screenshot,
                        capture_follow_up_screenshots=capture_follow_up_screenshots,
                    )
                if action == "file_upload":
                    return await self._file_upload(
                        kwargs.get("selector", ""), kwargs.get("filepath", ""),
                        capture_result_screenshot=capture_result_screenshot,
                        capture_follow_up_screenshots=capture_follow_up_screenshots,
                    )
                if action == "evaluate_js":
                    return await self._evaluate_js(kwargs.get("expression", ""))
                if action == "switch_frame":
                    return await self._switch_frame(kwargs.get("frame_selector"), kwargs.get("frame_index"))
                if action == "switch_main_frame":
                    return await self._switch_main_frame()
                if action == "handle_dialog":
                    return await self._handle_dialog(kwargs.get("dialog_action", "accept"), kwargs.get("prompt_text"))
                if action == "get_cookies":
                    return await self._get_cookies()
                if action == "set_cookie":
                    return await self._set_cookie(
                        kwargs.get("cookie_name", ""), kwargs.get("cookie_value", ""),
                        cookie_domain=kwargs.get("cookie_domain"), cookie_url=kwargs.get("cookie_url"),
                    )
                if action == "clear_cookies":
                    return await self._clear_cookies()
                return ToolResult(success=False, error=f"Unknown action: {action!r}")
            except Exception as exc:  # noqa: BLE001
                recovered = await self._recover_after_browser_loss(exc)
                if attempt == 0 and recovered:
                    continue
                self._snapshot.last_error = str(exc)
                return ToolResult(success=False, error=str(exc))
        return ToolResult(success=False, error="Browser session is unavailable.")

    async def preview(self, **kwargs: Any) -> dict[str, Any] | None:
        action: str = kwargs.get("action", "")
        if action == "navigate":
            return self._preview_payload(
                frame_label="browser · navigate",
                summary=f"Opening {self._short_url(kwargs.get('url', 'the page'))}",
                focus=self._default_focus_normalized(),
            )
        if action == "click":
            selector = kwargs.get("selector")
            if selector:
                return self._preview_payload(
                    frame_label="browser · click",
                    summary=f"Clicking {selector}",
                    focus=await self._selector_focus_normalized(selector),
                )
            x = kwargs.get("x")
            y = kwargs.get("y")
            focus = self._normalize_preview_focus(x, y) if x is not None and y is not None else self._default_focus_normalized()
            return self._preview_payload(frame_label="browser · click", summary="Clicking the page", focus=focus)
        if action == "type":
            selector = kwargs.get("selector")
            if selector:
                return self._preview_payload(
                    frame_label="browser · type",
                    summary=f"Typing into {selector}",
                    focus=await self._selector_focus_normalized(selector),
                )
            return self._preview_payload(
                frame_label="browser · type",
                summary="Typing text",
                focus=self._default_focus_normalized(y=0.56),
            )
        if action == "key":
            return self._preview_payload(
                frame_label="browser · key",
                summary=self._key_summary(kwargs.get("key")),
                focus=self._default_focus_normalized(y=0.56),
            )
        if action == "drag":
            return self._preview_payload(
                frame_label="browser · drag",
                summary="Dragging on the page",
                focus=self._normalize_drag_focus(kwargs.get("start_x"), kwargs.get("start_y"), kwargs.get("end_x"), kwargs.get("end_y")),
            )
        if action == "scroll":
            x = kwargs.get("x")
            y = kwargs.get("y")
            focus = self._normalize_preview_focus(x, y) if x or y else self._default_focus_normalized(y=0.78)
            return self._preview_payload(frame_label="browser · scroll", summary="Scrolling the page", focus=focus)
        if action == "screenshot":
            return self._preview_payload(
                frame_label="browser · screenshot",
                summary="Capturing a screenshot",
                focus=self._default_focus_normalized(),
            )
        if action in {"back", "forward", "new_tab", "close_tab"}:
            return self._preview_payload(
                frame_label=f"browser · {action}",
                summary={
                    "back": "Going back",
                    "forward": "Going forward",
                    "new_tab": "Opening a new tab",
                    "close_tab": "Closing the tab",
                }[action],
                focus=self._default_focus_normalized(),
            )
        return None

    def snapshot(self) -> BrowserSnapshot:
        return BrowserSnapshot(
            active=self._snapshot.active,
            active_url=self._snapshot.active_url,
            active_title=self._snapshot.active_title,
            tab_count=self._snapshot.tab_count,
            identity=self._snapshot.identity,
            profile_name=self._snapshot.profile_name,
            last_error=self._snapshot.last_error,
        )

    async def capture_live_frame(self) -> LiveBrowserFrame | None:
        frame_bytes = await self._capture_live_bytes(image_type="jpeg")
        if frame_bytes is None:
            return None
        return LiveBrowserFrame(data=frame_bytes, media_type="image/jpeg")

    async def capture_live_png(self) -> bytes | None:
        return await self._capture_live_bytes(image_type="png")

    async def warmup(self) -> None:
        await self._ensure_started()

    async def _capture_live_bytes(self, *, image_type: Literal["png", "jpeg"]) -> bytes | None:
        for attempt in range(2):
            try:
                await self._ensure_started()
                if self._page is None:
                    return None
                screenshot_kwargs: dict[str, Any] = {"type": image_type}
                if image_type == "jpeg":
                    screenshot_kwargs["quality"] = 55
                    screenshot_kwargs["scale"] = "css"
                png_bytes = await self._page.screenshot(**screenshot_kwargs)
                await self._refresh_snapshot()
                return png_bytes
            except Exception as exc:  # noqa: BLE001
                recovered = await self._recover_after_browser_loss(exc)
                if attempt == 0 and recovered:
                    continue
                return None
        return None

    async def _ensure_started(self) -> None:
        async with self._start_lock:
            if self._page is not None and not self._page_is_closed(self._page):
                return
            startup_started = time.perf_counter()
            if self._page is not None:
                await self._recover_after_browser_loss(None)
            self._context = await self.runtime.create_context()
            self._pages = [page for page in getattr(self._context, "pages", []) if not self._page_is_closed(page)]
            if not self._pages:
                self._pages = [await self._context.new_page()]
            self._set_active_page(self._pages[-1])
            self._snapshot.last_error = ""
            await self._refresh_snapshot()
            logger.info(
                "Browser session warmed thread_id=%s identity=%s startup_ms=%.1f",
                self.thread_id,
                self.runtime.identity,
                (time.perf_counter() - startup_started) * 1000,
            )

    def _set_active_page(self, page: Any) -> None:
        if page not in self._pages:
            self._pages.append(page)
        self._page = page

    async def _recover_after_browser_loss(self, exc: Exception | None) -> bool:
        if exc is not None and not self._looks_like_closed_browser_error(exc):
            self._snapshot.last_error = str(exc)
            return False
        try:
            if self.runtime.identity == "chrome_profile":
                await self.runtime.close()
            elif self._context is not None:
                await self._context.close()
        except Exception:  # noqa: BLE001
            pass
        self._context = None
        self._pages = []
        self._page = None
        self._snapshot = BrowserSnapshot(
            identity=self.runtime.identity,
            profile_name=self.runtime.profile_name,
            last_error="" if exc is None else str(exc),
        )
        return True

    @staticmethod
    def _looks_like_closed_browser_error(exc: Exception) -> bool:
        message = str(exc).lower()
        if not message:
            return False
        if "target page" in message and "closed" in message:
            return True
        if "browser has been closed" in message:
            return True
        if "browser is closed" in message:
            return True
        if "context has been closed" in message:
            return True
        if "page has been closed" in message:
            return True
        if "connection closed" in message:
            return True
        if "disconnected" in message and "browser" in message:
            return True
        return False

    @staticmethod
    def _page_is_closed(page: Any) -> bool:
        checker = getattr(page, "is_closed", None)
        if callable(checker):
            try:
                return bool(checker())
            except Exception:  # noqa: BLE001
                return True
        return False

    @property
    def _active_frame(self) -> Any:
        return self._frame_context if self._frame_context is not None else self._page

    async def _navigate(self, url: str, *, timeout: int | None = None) -> ToolResult:
        if not url:
            return ToolResult(success=False, error="url is required for 'navigate'")
        await self._page.goto(url, timeout=timeout or 30000, wait_until="domcontentloaded")
        title = await self._page.title()
        return await self._capture_visual_state(
            output=f"Navigated to {url!r}. Page title: {title!r}",
            frame_label="browser · navigate",
            summary=f"Opening {self._short_url(url)}",
            focus=self._default_focus(),
            burst=True,
        )

    async def _click(
        self,
        selector: str | None,
        x: float | None,
        y: float | None,
        *,
        timeout: int | None = None,
        capture_result_screenshot: bool = True,
        capture_follow_up_screenshots: bool = True,
    ) -> ToolResult:
        if selector:
            focus = await self._selector_focus(selector)
            await self._active_frame.click(selector, timeout=timeout or 5000)
            return await self._action_result(
                output=f"Clicked {selector!r}",
                frame_label="browser · click",
                summary=f"Clicking {selector}",
                focus=focus,
                burst=True,
                capture_result_screenshot=capture_result_screenshot,
                capture_follow_up_screenshots=capture_follow_up_screenshots,
            )
        if x is not None and y is not None:
            await self._page.mouse.click(x, y)
            return await self._action_result(
                output=f"Clicked ({x}, {y})",
                frame_label="browser · click",
                summary="Clicking the page",
                focus=(x, y),
                burst=True,
                capture_result_screenshot=capture_result_screenshot,
                capture_follow_up_screenshots=capture_follow_up_screenshots,
            )
        return ToolResult(success=False, error="Provide selector or x,y coordinates.")

    async def _type(
        self,
        selector: str | None,
        text: str,
        *,
        timeout: int | None = None,
        capture_result_screenshot: bool = True,
        capture_follow_up_screenshots: bool = True,
    ) -> ToolResult:
        if selector:
            focus = await self._selector_focus(selector)
            await self._active_frame.fill(selector, text, timeout=timeout or 5000)
            return await self._action_result(
                output=f"Typed into {selector!r}",
                frame_label="browser · type",
                summary=f"Typing into {selector}",
                focus=focus,
                burst=True,
                capture_result_screenshot=capture_result_screenshot,
                capture_follow_up_screenshots=capture_follow_up_screenshots,
            )
        await self._page.keyboard.type(text)
        return await self._action_result(
            output="Typed text.",
            frame_label="browser · type",
            summary="Typing text",
            focus=self._default_focus(y=0.56),
            burst=True,
            capture_result_screenshot=capture_result_screenshot,
            capture_follow_up_screenshots=capture_follow_up_screenshots,
        )

    async def _key(
        self,
        key: str,
        *,
        capture_result_screenshot: bool = True,
        capture_follow_up_screenshots: bool = True,
    ) -> ToolResult:
        if not key:
            return ToolResult(success=False, error="key is required for 'key'")
        await self._page.keyboard.press(key)
        return await self._action_result(
            output=f"Pressed {key!r}.",
            frame_label="browser · key",
            summary=self._key_summary(key),
            focus=self._default_focus(y=0.56),
            burst=True,
            capture_result_screenshot=capture_result_screenshot,
            capture_follow_up_screenshots=capture_follow_up_screenshots,
        )

    async def _drag(
        self,
        start_x: float | None,
        start_y: float | None,
        end_x: float | None,
        end_y: float | None,
        steps: int | None,
        *,
        capture_result_screenshot: bool = True,
        capture_follow_up_screenshots: bool = True,
    ) -> ToolResult:
        if None in {start_x, start_y, end_x, end_y}:
            return ToolResult(success=False, error="start_x, start_y, end_x, and end_y are required for 'drag'")
        drag_steps = max(1, int(steps or 14))
        await self._page.mouse.move(float(start_x), float(start_y))
        await self._page.mouse.down()
        await self._page.mouse.move(float(end_x), float(end_y), steps=drag_steps)
        await self._page.mouse.up()
        return await self._action_result(
            output=f"Dragged ({start_x}, {start_y}) to ({end_x}, {end_y})",
            frame_label="browser · drag",
            summary="Dragging on the page",
            focus=(float(end_x), float(end_y)),
            burst=True,
            capture_result_screenshot=capture_result_screenshot,
            capture_follow_up_screenshots=capture_follow_up_screenshots,
        )

    async def _scroll(
        self,
        x: float,
        y: float,
        delta_y: float,
        *,
        capture_result_screenshot: bool = True,
        capture_follow_up_screenshots: bool = True,
    ) -> ToolResult:
        await self._page.mouse.wheel(delta_x=0, delta_y=delta_y)
        focus = (x, y) if x or y else self._default_focus(y=0.78)
        return await self._action_result(
            output=f"Scrolled {delta_y}px",
            frame_label="browser · scroll",
            summary="Scrolling the page",
            focus=focus,
            burst=True,
            capture_result_screenshot=capture_result_screenshot,
            capture_follow_up_screenshots=capture_follow_up_screenshots,
        )

    async def _screenshot(self) -> ToolResult:
        return await self._capture_visual_state(
            output=f"Screenshot captured. Current URL: {self._page.url}",
            frame_label="browser · screenshot",
            summary="Capturing a screenshot",
            focus=self._default_focus(),
        )

    async def _action_result(
        self,
        *,
        output: str,
        frame_label: str,
        summary: str,
        focus: tuple[float, float],
        burst: bool = False,
        capture_result_screenshot: bool = True,
        capture_follow_up_screenshots: bool = True,
    ) -> ToolResult:
        if capture_result_screenshot:
            return await self._capture_visual_state(
                output=output,
                frame_label=frame_label,
                summary=summary,
                focus=focus,
                burst=burst and capture_follow_up_screenshots,
            )
        await self._refresh_snapshot()
        focus_x, focus_y = self._normalize_focus(*focus)
        return ToolResult(
            success=True,
            output=output,
            metadata=self._metadata(
                summary=summary,
                frame_label=frame_label,
                focus_x=focus_x,
                focus_y=focus_y,
            ),
        )

    async def _get_text(self, selector: str | None, *, timeout: int | None = None) -> ToolResult:
        timeout_s = (timeout or 5000) / 1000
        coro = self._active_frame.inner_text(selector or "body")
        text = await asyncio.wait_for(coro, timeout=timeout_s)
        await self._refresh_snapshot()
        return self._plain_result(text[:8000], extracted_text=text[:8000])

    async def _get_html(self, selector: str | None, *, timeout: int | None = None) -> ToolResult:
        timeout_s = (timeout or 5000) / 1000
        if selector:
            coro = self._active_frame.inner_html(selector)
        else:
            coro = self._page.content()
        html = await asyncio.wait_for(coro, timeout=timeout_s)
        await self._refresh_snapshot()
        return self._plain_result(html[:8000], extracted_text=html[:8000])

    # ── B1-B4: new interaction, iframe, dialog, cookie actions ────────────────

    async def _hover(
        self, selector: str | None, x: float | None, y: float | None,
        *, timeout: int | None = None,
        capture_result_screenshot: bool = True, capture_follow_up_screenshots: bool = True,
    ) -> ToolResult:
        if selector:
            focus = await self._selector_focus(selector)
            await self._active_frame.hover(selector, timeout=timeout or 5000)
            return await self._action_result(
                output=f"Hovered {selector!r}", frame_label="browser · hover",
                summary=f"Hovering {selector}", focus=focus, burst=True,
                capture_result_screenshot=capture_result_screenshot,
                capture_follow_up_screenshots=capture_follow_up_screenshots,
            )
        if x is not None and y is not None:
            await self._page.mouse.move(x, y)
            return await self._action_result(
                output=f"Hovered ({x}, {y})", frame_label="browser · hover",
                summary="Hovering on the page", focus=(x, y), burst=True,
                capture_result_screenshot=capture_result_screenshot,
                capture_follow_up_screenshots=capture_follow_up_screenshots,
            )
        return ToolResult(success=False, error="Provide selector or x,y coordinates for hover.")

    async def _double_click(
        self, selector: str | None, x: float | None, y: float | None,
        *, timeout: int | None = None,
        capture_result_screenshot: bool = True, capture_follow_up_screenshots: bool = True,
    ) -> ToolResult:
        if selector:
            focus = await self._selector_focus(selector)
            await self._active_frame.dblclick(selector, timeout=timeout or 5000)
            return await self._action_result(
                output=f"Double-clicked {selector!r}", frame_label="browser · double_click",
                summary=f"Double-clicking {selector}", focus=focus, burst=True,
                capture_result_screenshot=capture_result_screenshot,
                capture_follow_up_screenshots=capture_follow_up_screenshots,
            )
        if x is not None and y is not None:
            await self._page.mouse.dblclick(x, y)
            return await self._action_result(
                output=f"Double-clicked ({x}, {y})", frame_label="browser · double_click",
                summary="Double-clicking the page", focus=(x, y), burst=True,
                capture_result_screenshot=capture_result_screenshot,
                capture_follow_up_screenshots=capture_follow_up_screenshots,
            )
        return ToolResult(success=False, error="Provide selector or x,y coordinates for double_click.")

    async def _right_click(
        self, selector: str | None, x: float | None, y: float | None,
        *, timeout: int | None = None,
        capture_result_screenshot: bool = True, capture_follow_up_screenshots: bool = True,
    ) -> ToolResult:
        if selector:
            focus = await self._selector_focus(selector)
            await self._active_frame.click(selector, button="right", timeout=timeout or 5000)
            return await self._action_result(
                output=f"Right-clicked {selector!r}", frame_label="browser · right_click",
                summary=f"Right-clicking {selector}", focus=focus, burst=True,
                capture_result_screenshot=capture_result_screenshot,
                capture_follow_up_screenshots=capture_follow_up_screenshots,
            )
        if x is not None and y is not None:
            await self._page.mouse.click(x, y, button="right")
            return await self._action_result(
                output=f"Right-clicked ({x}, {y})", frame_label="browser · right_click",
                summary="Right-clicking the page", focus=(x, y), burst=True,
                capture_result_screenshot=capture_result_screenshot,
                capture_follow_up_screenshots=capture_follow_up_screenshots,
            )
        return ToolResult(success=False, error="Provide selector or x,y coordinates for right_click.")

    async def _select_option(
        self, selector: str, *, value: str | None = None, label: str | None = None,
        option_index: int | None = None, timeout: int | None = None,
        capture_result_screenshot: bool = True, capture_follow_up_screenshots: bool = True,
    ) -> ToolResult:
        if not selector:
            return ToolResult(success=False, error="selector is required for select_option.")
        option_kwargs: dict[str, Any] = {}
        if value is not None:
            option_kwargs["value"] = value
        elif label is not None:
            option_kwargs["label"] = label
        elif option_index is not None:
            option_kwargs["index"] = option_index
        else:
            return ToolResult(success=False, error="Provide value, label, or option_index for select_option.")
        focus = await self._selector_focus(selector)
        await self._active_frame.select_option(selector, timeout=timeout or 5000, **option_kwargs)
        desc = value or label or str(option_index)
        return await self._action_result(
            output=f"Selected {desc!r} in {selector!r}", frame_label="browser · select_option",
            summary=f"Selecting option in {selector}", focus=focus, burst=True,
            capture_result_screenshot=capture_result_screenshot,
            capture_follow_up_screenshots=capture_follow_up_screenshots,
        )

    async def _file_upload(
        self, selector: str, filepath: str, *,
        capture_result_screenshot: bool = True, capture_follow_up_screenshots: bool = True,
    ) -> ToolResult:
        if not selector:
            return ToolResult(success=False, error="selector is required for file_upload.")
        if not filepath:
            return ToolResult(success=False, error="filepath is required for file_upload.")
        focus = await self._selector_focus(selector)
        await self._active_frame.set_input_files(selector, filepath)
        return await self._action_result(
            output=f"Uploaded {filepath!r} to {selector!r}", frame_label="browser · file_upload",
            summary=f"Uploading file to {selector}", focus=focus, burst=True,
            capture_result_screenshot=capture_result_screenshot,
            capture_follow_up_screenshots=capture_follow_up_screenshots,
        )

    async def _evaluate_js(self, expression: str) -> ToolResult:
        if not expression:
            return ToolResult(success=False, error="expression is required for evaluate_js.")
        result = await self._active_frame.evaluate(expression)
        output = json.dumps(result, default=str) if result is not None else "undefined"
        return ToolResult(success=True, output=output[:8000])

    async def _switch_frame(self, frame_selector: str | None, frame_index: int | None) -> ToolResult:
        if frame_selector:
            self._frame_context = self._page.frame_locator(frame_selector)
            return ToolResult(success=True, output=f"Switched to frame: {frame_selector!r}")
        if frame_index is not None:
            frames = self._page.frames
            if frame_index < 0 or frame_index >= len(frames):
                return ToolResult(success=False, error=f"Frame index {frame_index} out of range (0-{len(frames) - 1}).")
            self._frame_context = frames[frame_index]
            return ToolResult(success=True, output=f"Switched to frame index {frame_index}.")
        return ToolResult(success=False, error="Provide frame_selector or frame_index.")

    async def _switch_main_frame(self) -> ToolResult:
        self._frame_context = None
        return ToolResult(success=True, output="Switched back to main frame.")

    def _on_dialog(self, dialog: Any) -> None:
        self._last_dialog = {"type": dialog.type, "message": dialog.message}
        self._pending_dialog = dialog
        asyncio.ensure_future(self._auto_dismiss_dialog(dialog))

    async def _auto_dismiss_dialog(self, dialog: Any) -> None:
        await asyncio.sleep(5.0)
        if self._pending_dialog is dialog:
            try:
                await dialog.dismiss()
            except Exception:  # noqa: BLE001
                pass
            self._pending_dialog = None

    async def _handle_dialog(self, dialog_action: str, prompt_text: str | None) -> ToolResult:
        if self._pending_dialog is None:
            info = self._last_dialog or {}
            return ToolResult(
                success=False, error="No pending dialog to handle.",
                output=f"Last dialog info: {info}" if info else "",
            )
        dialog = self._pending_dialog
        self._pending_dialog = None
        if dialog_action == "dismiss":
            await dialog.dismiss()
            return ToolResult(success=True, output=f"Dismissed {dialog.type} dialog: {dialog.message!r}")
        if prompt_text is not None:
            await dialog.accept(prompt_text)
        else:
            await dialog.accept()
        return ToolResult(success=True, output=f"Accepted {dialog.type} dialog: {dialog.message!r}")

    async def _get_cookies(self) -> ToolResult:
        cookies = await self._context.cookies()
        output = json.dumps(cookies, default=str)
        return ToolResult(success=True, output=output[:8000])

    async def _set_cookie(
        self, name: str, value: str, *,
        cookie_domain: str | None = None, cookie_url: str | None = None,
    ) -> ToolResult:
        if not name:
            return ToolResult(success=False, error="cookie_name is required.")
        cookie: dict[str, Any] = {"name": name, "value": value}
        if cookie_url:
            cookie["url"] = cookie_url
        elif cookie_domain:
            cookie["domain"] = cookie_domain
            cookie["path"] = "/"
        else:
            cookie["url"] = self._page.url
        await self._context.add_cookies([cookie])
        return ToolResult(success=True, output=f"Cookie {name!r} set.")

    async def _clear_cookies(self) -> ToolResult:
        await self._context.clear_cookies()
        return ToolResult(success=True, output="All cookies cleared.")

    def _plain_result(self, output: str, *, extracted_text: str | None = None) -> ToolResult:
        metadata = self._metadata(summary="", extracted_text=extracted_text)
        return ToolResult(success=True, output=output, metadata=metadata)

    async def _capture_browser_screenshot(self) -> bytes:
        """Capture a browser screenshot in the configured format."""
        fmt = self._screenshot_format
        if fmt == "jpeg":
            return await self._page.screenshot(type="jpeg", quality=self._screenshot_quality)
        png_bytes = await self._page.screenshot(type="png")
        if fmt == "webp":
            from PIL import Image as _Image  # noqa: PLC0415
            img = _Image.open(io.BytesIO(png_bytes))
            buf = io.BytesIO()
            img.save(buf, format="WEBP", quality=self._screenshot_quality)
            return buf.getvalue()
        return png_bytes

    async def _capture_visual_state(
        self,
        *,
        output: str,
        frame_label: str,
        summary: str,
        focus: tuple[float, float],
        burst: bool = False,
    ) -> ToolResult:
        img_bytes = await self._capture_browser_screenshot()
        await self._refresh_snapshot()
        focus_x, focus_y = self._normalize_focus(*focus)
        current_hash = compute_image_hash(img_bytes)
        no_change = images_are_similar(self._last_screenshot_hash, current_hash)
        self._last_screenshot_hash = current_hash
        metadata = self._metadata(
            summary=summary,
            frame_label=frame_label,
            focus_x=focus_x,
            focus_y=focus_y,
        )
        metadata["no_change"] = no_change
        metadata["image_format"] = self._screenshot_format
        return ToolResult(
            success=True,
            output=output,
            screenshot_b64=base64.b64encode(img_bytes).decode(),
            extra_screenshots=await self._capture_follow_up_frames(
                frame_label=frame_label,
                summary=summary,
                focus_x=focus_x,
                focus_y=focus_y,
                enabled=burst,
            ),
            metadata=metadata,
        )

    def _metadata(
        self,
        *,
        summary: str,
        frame_label: str | None = None,
        focus_x: float | None = None,
        focus_y: float | None = None,
        extracted_text: str | None = None,
    ) -> dict[str, Any]:
        snapshot = self.snapshot()
        metadata: dict[str, Any] = {
            "summary": summary,
            "active_url": snapshot.active_url,
            "active_title": snapshot.active_title,
            "tab_count": snapshot.tab_count,
        }
        if frame_label:
            metadata["frame_label"] = frame_label
        if focus_x is not None:
            metadata["focus_x"] = focus_x
        if focus_y is not None:
            metadata["focus_y"] = focus_y
        if extracted_text:
            metadata["extracted_text"] = extracted_text
        return metadata

    async def _capture_follow_up_frames(
        self,
        *,
        frame_label: str,
        summary: str,
        focus_x: float,
        focus_y: float,
        enabled: bool,
    ) -> list[dict[str, Any]]:
        if not enabled or self._page is None:
            return []
        frames: list[dict[str, Any]] = []
        for index, delay in enumerate(_LIVE_REFRESH_INTERVALS, start=1):
            await asyncio.sleep(delay)
            png_bytes = await self._page.screenshot(type="png")
            await self._refresh_snapshot()
            frame_summary = f"{summary} (refresh {index})"
            frame: dict[str, Any] = {
                "data": base64.b64encode(png_bytes).decode(),
                "frame_label": frame_label,
                "summary": frame_summary,
                "focus_x": focus_x,
                "focus_y": focus_y,
            }
            frame.update(
                {
                    key: value
                    for key, value in self._metadata(
                        summary=frame_summary,
                        frame_label=frame_label,
                        focus_x=focus_x,
                        focus_y=focus_y,
                    ).items()
                    if key not in {"summary", "frame_label", "focus_x", "focus_y"}
                }
            )
            frames.append(frame)
        return frames

    async def _refresh_snapshot(self) -> None:
        if self._page is None:
            self._snapshot = BrowserSnapshot(
                identity=self.runtime.identity,
                profile_name=self.runtime.profile_name,
                last_error=self._snapshot.last_error,
            )
            return
        pages = [page for page in self._pages if not self._page_is_closed(page)]
        if self._page is not None and self._page not in pages and not self._page_is_closed(self._page):
            pages.append(self._page)
        self._pages = pages
        if self._page is None or self._page_is_closed(self._page):
            self._page = self._pages[-1] if self._pages else None
        if self._page is None:
            self._snapshot = BrowserSnapshot(
                identity=self.runtime.identity,
                profile_name=self.runtime.profile_name,
                last_error=self._snapshot.last_error,
            )
            return
        title = ""
        try:
            title = await self._page.title()
        except Exception:  # noqa: BLE001
            title = ""
        self._snapshot = BrowserSnapshot(
            active=True,
            active_url=str(getattr(self._page, "url", "") or ""),
            active_title=title,
            tab_count=len(self._pages),
            identity=self.runtime.identity,
            profile_name=self.runtime.profile_name,
            last_error="",
        )

    async def close(self) -> None:
        try:
            if self.runtime.identity == "chrome_profile":
                await self.runtime.close()
            elif self._context is not None:
                await self._context.close()
        except Exception:  # noqa: BLE001
            pass
        self._context = None
        self._pages = []
        self._page = None
        self._snapshot = BrowserSnapshot(identity=self.runtime.identity, profile_name=self.runtime.profile_name)

    async def _selector_focus(self, selector: str) -> tuple[float, float]:
        box = await self._selector_box(selector)
        if box:
            return (box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
        return self._default_focus()

    async def _selector_focus_normalized(self, selector: str) -> tuple[float, float]:
        if self._page is None:
            return self._default_focus_normalized()
        box = await self._selector_box(selector)
        if box:
            return self._normalize_focus(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
        return self._default_focus_normalized()

    async def _selector_box(self, selector: str) -> dict[str, float] | None:
        if self._page is None:
            return None
        locator = self._page.locator(selector).first
        try:
            box = await asyncio.wait_for(
                locator.bounding_box(),
                timeout=_SELECTOR_FOCUS_TIMEOUT_SECONDS,
            )
        except Exception:  # noqa: BLE001
            return None
        return box

    def _normalize_focus(self, x: float, y: float) -> tuple[float, float]:
        size = self._page.viewport_size or {"width": 1280, "height": 720}
        width = max(float(size.get("width", 1280)), 1.0)
        height = max(float(size.get("height", 720)), 1.0)
        return (
            max(0.0, min(1.0, float(x) / width)),
            max(0.0, min(1.0, float(y) / height)),
        )

    def _default_focus(self, *, x: float = _DEFAULT_FOCUS[0], y: float = _DEFAULT_FOCUS[1]) -> tuple[float, float]:
        size = self._page.viewport_size or {"width": 1280, "height": 720}
        width = float(size.get("width", 1280))
        height = float(size.get("height", 720))
        return (x * width, y * height)

    @staticmethod
    def _default_focus_normalized(*, x: float = _DEFAULT_FOCUS[0], y: float = _DEFAULT_FOCUS[1]) -> tuple[float, float]:
        return (
            max(0.0, min(1.0, float(x))),
            max(0.0, min(1.0, float(y))),
        )

    def _normalize_preview_focus(self, x: float | None, y: float | None) -> tuple[float, float]:
        if x is None or y is None or self._page is None:
            return self._default_focus_normalized()
        return self._normalize_focus(float(x), float(y))

    def _normalize_drag_focus(
        self,
        start_x: float | None,
        start_y: float | None,
        end_x: float | None,
        end_y: float | None,
    ) -> tuple[float, float]:
        if end_x is not None and end_y is not None:
            return self._normalize_preview_focus(end_x, end_y)
        if start_x is not None and start_y is not None:
            return self._normalize_preview_focus(start_x, start_y)
        return self._default_focus_normalized()

    @staticmethod
    def _preview_payload(*, frame_label: str, summary: str, focus: tuple[float, float]) -> dict[str, Any]:
        focus_x, focus_y = focus
        return {
            "frame_label": frame_label,
            "summary": summary,
            "focus_x": max(0.0, min(1.0, float(focus_x))),
            "focus_y": max(0.0, min(1.0, float(focus_y))),
        }

    @staticmethod
    def _key_summary(key: Any) -> str:
        value = str(key or "").strip()
        return f"Pressing {value}" if value else "Pressing a key"

    @staticmethod
    def _short_url(url: str) -> str:
        compact = url.replace("https://", "").replace("http://", "").rstrip("/")
        return compact.removeprefix("www.") or url


class BrowserSessionManager:
    """Creates and reuses thread-scoped browser sessions."""

    def __init__(self) -> None:
        self._runtimes: dict[tuple[Any, ...], BrowserRuntime] = {}
        self._sessions: dict[str, BrowserSession] = {}
        self._session_runtime_key: dict[str, tuple[Any, ...]] = {}
        self._defaults: dict[str, dict[str, str]] = {}
        self._restored_snapshots: dict[str, BrowserSnapshot] = {}
        self._warmup_tasks: dict[str, asyncio.Task[None]] = {}
        self._lock = asyncio.Lock()

    async def session_for(
        self,
        *,
        thread_id: str,
        browser_type: Literal["chromium", "firefox", "webkit"] = "chromium",
        headless: bool = False,
        identity: Literal["isolated", "chrome_profile"] = "isolated",
        profile_name: str = "Default",
    ) -> BrowserSession:
        async with self._lock:
            self.note_thread_browser_defaults(
                thread_id,
                identity=identity,
                profile_name=profile_name,
            )
            runtime_key: tuple[Any, ...] = (
                browser_type,
                bool(headless),
                identity,
                profile_name or "Default",
                thread_id if identity == "chrome_profile" else "__shared__",
            )
            existing = self._sessions.get(thread_id)
            if existing is not None and self._session_runtime_key.get(thread_id) == runtime_key:
                return existing
            if existing is not None:
                await existing.close()
            runtime = self._runtimes.get(runtime_key)
            if runtime is None:
                runtime = BrowserRuntime(
                    browser_type=browser_type,
                    headless=headless,
                    identity=identity,
                    profile_name=profile_name or "Default",
                    profile_runtime_id=thread_id if identity == "chrome_profile" else None,
                )
                self._runtimes[runtime_key] = runtime
            session = BrowserSession(thread_id=thread_id, runtime=runtime)
            self._sessions[thread_id] = session
            self._session_runtime_key[thread_id] = runtime_key
            return session

    def note_thread_browser_defaults(
        self,
        thread_id: str,
        *,
        identity: str,
        profile_name: str,
    ) -> None:
        self._defaults[thread_id] = {
            "identity": identity or "isolated",
            "profile_name": profile_name or "Default",
        }

    def restore_thread_snapshot(self, thread_id: str, payload: Any) -> None:
        if not isinstance(payload, dict):
            return
        browser_payload = payload.get("browser") if isinstance(payload.get("browser"), dict) else payload
        if not isinstance(browser_payload, dict):
            return
        self._restored_snapshots[thread_id] = BrowserSnapshot(
            active=False,
            active_url=str(browser_payload.get("active_url", "") or ""),
            active_title=str(browser_payload.get("active_title", "") or ""),
            tab_count=int(browser_payload.get("tab_count", 0) or 0),
            identity=str(browser_payload.get("identity") or payload.get("browser_identity") or "isolated"),
            profile_name=str(browser_payload.get("profile_name") or payload.get("browser_profile_name") or "Default"),
            last_error=str(browser_payload.get("last_error", "") or payload.get("browser_last_error", "") or ""),
        )

    def snapshot(self, thread_id: str) -> BrowserSnapshot:
        session = self._sessions.get(thread_id)
        if session is None:
            restored = self._restored_snapshots.get(thread_id)
            if restored is not None:
                defaults = self._defaults.get(thread_id, {})
                return BrowserSnapshot(
                    active=False,
                    active_url=restored.active_url,
                    active_title=restored.active_title,
                    tab_count=restored.tab_count,
                    identity=str(restored.identity or defaults.get("identity") or "isolated"),
                    profile_name=str(restored.profile_name or defaults.get("profile_name") or "Default"),
                    last_error=restored.last_error,
                )
            defaults = self._defaults.get(thread_id, {})
            return BrowserSnapshot(
                identity=str(defaults.get("identity") or "isolated"),
                profile_name=str(defaults.get("profile_name") or "Default"),
            )
        return session.snapshot()

    async def close(self) -> None:
        for task in list(self._warmup_tasks.values()):
            task.cancel()
        runtimes = list(self._runtimes.values())
        self._sessions.clear()
        self._session_runtime_key.clear()
        self._runtimes.clear()
        self._defaults.clear()
        self._restored_snapshots.clear()
        self._warmup_tasks.clear()
        for runtime in runtimes:
            await runtime.close()

    def snapshot_payload(self, thread_id: str) -> dict[str, Any]:
        defaults = self._defaults.get(thread_id, {})
        snapshot = self.snapshot(thread_id)
        identity = str(snapshot.identity or defaults.get("identity") or "isolated")
        profile_name = str(snapshot.profile_name or defaults.get("profile_name") or "Default")
        return {
            "browser_session_active": snapshot.active,
            "active_url": snapshot.active_url,
            "active_title": snapshot.active_title,
            "tab_count": snapshot.tab_count,
            "browser_identity": identity,
            "browser_profile_name": profile_name,
            "browser_last_error": snapshot.last_error,
            "browser": {
                **asdict(snapshot),
                "identity": identity,
                "profile_name": profile_name,
            },
        }

    async def capture_live_frame(self, thread_id: str) -> LiveBrowserFrame | None:
        session = self._sessions.get(thread_id)
        if session is None:
            return None
        return await session.capture_live_frame()

    async def capture_live_png(self, thread_id: str) -> bytes | None:
        session = self._sessions.get(thread_id)
        if session is None:
            return None
        return await session.capture_live_png()

    def warmup_thread(
        self,
        *,
        thread_id: str,
        browser_type: Literal["chromium", "firefox", "webkit"] = "chromium",
        headless: bool = False,
        identity: Literal["isolated", "chrome_profile"] = "isolated",
        profile_name: str = "Default",
    ) -> asyncio.Task[None]:
        existing = self._warmup_tasks.get(thread_id)
        if existing is not None and not existing.done():
            return existing

        async def runner() -> None:
            try:
                session = await self.session_for(
                    thread_id=thread_id,
                    browser_type=browser_type,
                    headless=headless,
                    identity=identity,
                    profile_name=profile_name,
                )
                await session.warmup()
            except Exception:
                logger.debug("Browser warmup failed thread_id=%s", thread_id, exc_info=True)
            finally:
                if self._warmup_tasks.get(thread_id) is task:
                    self._warmup_tasks.pop(thread_id, None)

        task = asyncio.create_task(runner())
        self._warmup_tasks[thread_id] = task
        return task

    def cancel_warmup_thread(self, thread_id: str) -> None:
        task = self._warmup_tasks.pop(thread_id, None)
        if task is not None and not task.done():
            task.cancel()
