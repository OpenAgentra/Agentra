"""Filesystem tool — read and write files on disk."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

from agentra.tools.base import BaseTool, ToolResult


class FilesystemTool(BaseTool):
    """
    Read, write, list, copy, move and delete files and directories.
    Operations are unrestricted when *allow_write* is True; read-only otherwise.
    """

    name = "filesystem"
    tool_capabilities = ("filesystem",)
    description = (
        "Access the local filesystem: read file contents, write or append to files, "
        "list directory contents, create directories, copy/move/delete files. "
        "Use this to work with the user's documents, configuration files, and any "
        "local data."
    )

    def __init__(self, workspace_dir: Optional[Path] = None, allow_write: bool = True) -> None:
        self._workspace_dir = workspace_dir
        self._allow_write = allow_write

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "read",
                        "write",
                        "append",
                        "list",
                        "mkdir",
                        "delete",
                        "exists",
                        "copy",
                        "move",
                        "cwd",
                    ],
                    "description": "Filesystem action to perform.",
                },
                "path": {"type": "string", "description": "File or directory path."},
                "content": {"type": "string", "description": "Content to write/append."},
                "destination": {
                    "type": "string",
                    "description": "Destination path for copy/move.",
                },
                "recursive": {
                    "type": "boolean",
                    "description": "List recursively (default false).",
                },
            },
            "required": ["action"],
        }

    async def execute(self, **kwargs: Any) -> ToolResult:
        action: str = kwargs.get("action", "")
        path_str: Optional[str] = kwargs.get("path")

        path = self._resolve(path_str) if path_str else None

        try:
            if action == "cwd":
                return ToolResult(success=True, output=str(os.getcwd()))
            if action == "read":
                return self._read(path)
            if action == "write":
                return self._write(path, kwargs.get("content", ""), mode="w")
            if action == "append":
                return self._write(path, kwargs.get("content", ""), mode="a")
            if action == "list":
                return self._list(path, kwargs.get("recursive", False))
            if action == "mkdir":
                return self._mkdir(path)
            if action == "delete":
                return self._delete(path)
            if action == "exists":
                return self._exists(path)
            if action == "copy":
                dst = self._resolve(kwargs.get("destination"))
                return self._copy(path, dst)
            if action == "move":
                dst = self._resolve(kwargs.get("destination"))
                return self._move(path, dst)
            return ToolResult(success=False, error=f"Unknown action: {action!r}")
        except Exception as exc:  # noqa: BLE001
            return ToolResult(success=False, error=str(exc))

    # ── private ────────────────────────────────────────────────────────────────

    def _resolve(self, path_str: Optional[str]) -> Path:
        if not path_str:
            raise ValueError("path is required")
        p = Path(path_str).expanduser()
        if not p.is_absolute() and self._workspace_dir:
            p = self._workspace_dir / p
        return p.resolve()

    def _read(self, path: Optional[Path]) -> ToolResult:
        if path is None:
            return ToolResult(success=False, error="path is required for read")
        if not path.exists():
            return ToolResult(success=False, error=f"File not found: {path}")
        content = path.read_text(encoding="utf-8", errors="replace")
        return ToolResult(success=True, output=content)

    def _write(self, path: Optional[Path], content: str, mode: str) -> ToolResult:
        if not self._allow_write:
            return ToolResult(success=False, error="Filesystem write is disabled.")
        if path is None:
            return ToolResult(success=False, error="path is required for write/append")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open(mode, encoding="utf-8") as f:
            f.write(content)
        verb = "Written" if mode == "w" else "Appended"
        return ToolResult(success=True, output=f"{verb} to {path}")

    def _list(self, path: Optional[Path], recursive: bool) -> ToolResult:
        target = path or Path.cwd()
        if not target.exists():
            return ToolResult(success=False, error=f"Path not found: {target}")
        if target.is_file():
            return ToolResult(success=True, output=str(target))
        if recursive:
            entries = sorted(str(p) for p in target.rglob("*"))
        else:
            entries = sorted(str(p) for p in target.iterdir())
        return ToolResult(success=True, output="\n".join(entries))

    def _mkdir(self, path: Optional[Path]) -> ToolResult:
        if not self._allow_write:
            return ToolResult(success=False, error="Filesystem write is disabled.")
        if path is None:
            return ToolResult(success=False, error="path is required for mkdir")
        path.mkdir(parents=True, exist_ok=True)
        return ToolResult(success=True, output=f"Directory created: {path}")

    def _delete(self, path: Optional[Path]) -> ToolResult:
        if not self._allow_write:
            return ToolResult(success=False, error="Filesystem write is disabled.")
        if path is None:
            return ToolResult(success=False, error="path is required for delete")
        if not path.exists():
            return ToolResult(success=False, error=f"Path not found: {path}")
        import shutil  # noqa: PLC0415

        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
        return ToolResult(success=True, output=f"Deleted: {path}")

    def _exists(self, path: Optional[Path]) -> ToolResult:
        if path is None:
            return ToolResult(success=False, error="path is required for exists")
        return ToolResult(success=True, output=str(path.exists()))

    def _copy(self, src: Optional[Path], dst: Optional[Path]) -> ToolResult:
        if not self._allow_write:
            return ToolResult(success=False, error="Filesystem write is disabled.")
        if src is None or dst is None:
            return ToolResult(success=False, error="path and destination are required for copy")
        import shutil  # noqa: PLC0415

        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        return ToolResult(success=True, output=f"Copied {src} → {dst}")

    def _move(self, src: Optional[Path], dst: Optional[Path]) -> ToolResult:
        if not self._allow_write:
            return ToolResult(success=False, error="Filesystem write is disabled.")
        if src is None or dst is None:
            return ToolResult(success=False, error="path and destination are required for move")
        import shutil  # noqa: PLC0415

        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        return ToolResult(success=True, output=f"Moved {src} → {dst}")
