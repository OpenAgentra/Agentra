"""Native local OS operations for hidden local execution flows."""

from __future__ import annotations

import asyncio
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agentra.tools.base import BaseTool, ToolResult


@dataclass
class ResolvedLocalPath:
    """Normalized path information for Windows-backed local operations."""

    requested: str
    windows_path: str
    wsl_path: str


class LocalSystemTool(BaseTool):
    """Resolve Windows user folders and open local items with the OS default handler."""

    name = "local_system"
    tool_capabilities = ("local_system",)
    description = (
        "Resolve known local folders such as Desktop and open confirmed local files or "
        "folders with the operating system default handler without using visible terminal "
        "windows. Prefer this for under-the-hood local desktop/file tasks."
    )

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["resolve_known_folder", "open_path"],
                    "description": "Native local-system action to perform.",
                },
                "folder_key": {
                    "type": "string",
                    "enum": ["desktop", "onedrive_desktop"],
                    "description": "Known folder to resolve.",
                },
                "path": {
                    "type": "string",
                    "description": "Resolved local file or folder path to open.",
                },
            },
            "required": ["action"],
        }

    async def execute(self, **kwargs: Any) -> ToolResult:
        action = str(kwargs.get("action", "") or "").strip().lower()
        try:
            if action == "resolve_known_folder":
                folder_key = str(kwargs.get("folder_key", "") or "desktop").strip().lower()
                return await self._resolve_known_folder(folder_key)
            if action == "open_path":
                path = str(kwargs.get("path", "") or "").strip()
                return await self._open_path(path)
            return ToolResult(success=False, error=f"Unknown action: {action!r}")
        except Exception as exc:  # noqa: BLE001
            return ToolResult(success=False, error=str(exc), metadata={"summary": str(exc)})

    async def preview(self, **kwargs: Any) -> dict[str, Any] | None:
        action = str(kwargs.get("action", "") or "").strip().lower()
        if action == "resolve_known_folder":
            return {
                "frame_label": "local_system · resolve_known_folder",
                "summary": "Yerel klasor cozuluyor",
            }
        if action == "open_path":
            return {
                "frame_label": "local_system · open_path",
                "summary": "Dosya arka planda aciliyor",
            }
        return None

    async def _resolve_known_folder(self, folder_key: str) -> ToolResult:
        resolved = await asyncio.to_thread(self._resolve_known_folder_sync, folder_key)
        output = (
            f"Resolved {folder_key}.\n"
            f"WSL path: {resolved.wsl_path}\n"
            f"Windows path: {resolved.windows_path}"
        )
        summary = f"Resolved {folder_key}."
        return ToolResult(
            success=True,
            output=output,
            metadata={
                "summary": summary,
                "frame_label": "local_system · resolve_known_folder",
                "folder_key": folder_key,
                "resolved_path": resolved.wsl_path,
                "windows_path": resolved.windows_path,
                "wsl_path": resolved.wsl_path,
            },
        )

    async def _open_path(self, path: str) -> ToolResult:
        if not path:
            return ToolResult(success=False, error="path is required for open_path")
        resolved = await asyncio.to_thread(self._normalize_input_path, path)
        await asyncio.to_thread(self._ensure_path_exists, resolved)
        await asyncio.to_thread(self._open_path_sync, resolved)
        output = f"Opened local path with the OS default handler: {resolved.windows_path}"
        return ToolResult(
            success=True,
            output=output,
            metadata={
                "summary": "Opened local path.",
                "frame_label": "local_system · open_path",
                "resolved_path": resolved.wsl_path,
                "windows_path": resolved.windows_path,
                "wsl_path": resolved.wsl_path,
            },
        )

    def _resolve_known_folder_sync(self, folder_key: str) -> ResolvedLocalPath:
        key = folder_key.strip().lower()
        if key == "desktop":
            windows_path = self._desktop_windows_path()
        elif key == "onedrive_desktop":
            windows_path = self._onedrive_desktop_windows_path()
        else:
            raise ValueError(f"Unknown folder_key: {folder_key!r}")

        resolved = self._normalize_input_path(windows_path)
        if not self._path_exists(resolved):
            raise FileNotFoundError(f"Known folder does not exist: {resolved.windows_path}")
        return resolved

    def _desktop_windows_path(self) -> str:
        desktop = self._powershell_stdout("[Environment]::GetFolderPath('Desktop')")
        if desktop:
            return desktop

        userprofile = os.environ.get("USERPROFILE")
        if userprofile:
            candidate = Path(userprofile) / "Desktop"
            if candidate.exists():
                return str(candidate)

        return str(Path.home() / "Desktop")

    def _onedrive_desktop_windows_path(self) -> str:
        script = (
            "$candidate = Join-Path ([Environment]::GetFolderPath('UserProfile')) 'OneDrive\\Desktop'; "
            "if (Test-Path -LiteralPath $candidate) { $candidate }"
        )
        desktop = self._powershell_stdout(script)
        if desktop:
            return desktop

        onedrive = os.environ.get("OneDrive")
        if onedrive:
            candidate = Path(onedrive) / "Desktop"
            if candidate.exists():
                return str(candidate)

        candidate = Path.home() / "OneDrive" / "Desktop"
        return str(candidate)

    def _normalize_input_path(self, path: str) -> ResolvedLocalPath:
        raw = path.strip()
        if not raw:
            raise ValueError("path is required")

        if self._looks_like_windows_path(raw):
            windows_path = self._normalize_windows_path(raw)
            wsl_path = self._windows_to_wsl_path(windows_path)
            return ResolvedLocalPath(requested=raw, windows_path=windows_path, wsl_path=wsl_path)

        expanded = Path(raw).expanduser()
        if expanded.is_absolute():
            wsl_path = str(expanded.resolve(strict=False))
        else:
            wsl_path = str(expanded.resolve(strict=False))
        windows_path = self._wsl_to_windows_path(wsl_path)
        return ResolvedLocalPath(requested=raw, windows_path=windows_path, wsl_path=wsl_path)

    def _ensure_path_exists(self, resolved: ResolvedLocalPath) -> None:
        if self._path_exists(resolved):
            return
        raise FileNotFoundError(f"Path not found: {resolved.requested}")

    def _path_exists(self, resolved: ResolvedLocalPath) -> bool:
        if resolved.wsl_path:
            try:
                if Path(resolved.wsl_path).exists():
                    return True
            except OSError:
                pass
        if os.name == "nt":
            return Path(resolved.windows_path).exists()
        if self._running_in_wsl() and resolved.windows_path:
            script = (
                "$target = $args[0]; "
                "if (Test-Path -LiteralPath $target) { Write-Output 'True' }"
            )
            return bool(self._powershell_stdout(script, resolved.windows_path))
        return False

    def _open_path_sync(self, resolved: ResolvedLocalPath) -> None:
        if os.name == "nt":
            os.startfile(resolved.windows_path)  # type: ignore[attr-defined]
            return
        if self._running_in_wsl():
            script = (
                "$target = $args[0]; "
                "if (-not (Test-Path -LiteralPath $target)) { throw \"Path not found: $target\" }; "
                "$item = Get-Item -LiteralPath $target; "
                "Start-Process -FilePath $item.FullName | Out-Null"
            )
            self._run_subprocess(
                [
                    "powershell.exe",
                    "-NoProfile",
                    "-WindowStyle",
                    "Hidden",
                    "-Command",
                    script,
                    resolved.windows_path,
                ]
            )
            return

        opener = "open" if sys.platform == "darwin" else "xdg-open"
        with open(os.devnull, "wb") as sink:
            subprocess.Popen(  # noqa: S603
                [opener, resolved.wsl_path],
                stdout=sink,
                stderr=sink,
                start_new_session=True,
            )

    def _powershell_stdout(self, script: str, *args: str) -> str:
        candidates = ["powershell.exe"] if self._running_in_wsl() or os.name == "nt" else []
        for binary in candidates:
            try:
                result = self._run_subprocess(
                    [binary, "-NoProfile", "-Command", script, *args],
                    capture_output=True,
                )
            except FileNotFoundError:
                continue
            output = (result.stdout or "").strip()
            if output:
                return output.splitlines()[-1].strip()
        return ""

    def _run_subprocess(
        self,
        args: list[str],
        *,
        capture_output: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            args,
            check=True,
            text=True,
            capture_output=capture_output,
        )

    @staticmethod
    def _looks_like_windows_path(path: str) -> bool:
        return bool(re.match(r"^[a-zA-Z]:[\\\\/]", path))

    @staticmethod
    def _normalize_windows_path(path: str) -> str:
        return path.replace("/", "\\")

    @staticmethod
    def _windows_to_wsl_path(path: str) -> str:
        match = re.match(r"^([a-zA-Z]):\\(.*)$", path)
        if not match:
            return path
        drive, rest = match.groups()
        rest = rest.replace("\\", "/")
        return f"/mnt/{drive.lower()}/{rest}"

    @staticmethod
    def _wsl_to_windows_path(path: str) -> str:
        match = re.match(r"^/mnt/([a-zA-Z])/(.*)$", path)
        if match:
            drive, rest = match.groups()
            return f"{drive.upper()}:\\{rest.replace('/', '\\')}"
        return path

    @staticmethod
    def _running_in_wsl() -> bool:
        if os.environ.get("WSL_INTEROP"):
            return True
        try:
            version = Path("/proc/version").read_text(encoding="utf-8", errors="ignore")
        except OSError:
            return False
        return "microsoft" in version.casefold()
