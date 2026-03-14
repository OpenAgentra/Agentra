"""Terminal tool — execute shell commands in the agent's workspace."""

from __future__ import annotations

import asyncio
import os
import shlex
from pathlib import Path
from typing import Any, Optional

from agentra.tools.base import BaseTool, ToolResult

# Hard limit on captured output to avoid flooding the context window
_MAX_OUTPUT_CHARS = 10_000


class TerminalTool(BaseTool):
    """
    Run arbitrary shell commands.  All commands execute inside the agent's
    workspace directory by default and capture both stdout and stderr.
    """

    name = "terminal"
    description = (
        "Execute shell commands on the local computer. "
        "You can run scripts, install packages, compile code, manage files, "
        "start/stop services, or do anything the user's shell can do. "
        "stdout and stderr are both captured and returned."
    )

    def __init__(
        self,
        cwd: Optional[Path] = None,
        allow: bool = True,
        timeout: int = 60,
    ) -> None:
        self._cwd = cwd
        self._allow = allow
        self._timeout = timeout

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Shell command to execute.",
                },
                "cwd": {
                    "type": "string",
                    "description": "Working directory (defaults to agent workspace).",
                },
                "timeout": {
                    "type": "number",
                    "description": "Timeout in seconds (default 60).",
                },
            },
            "required": ["command"],
        }

    async def execute(self, **kwargs: Any) -> ToolResult:
        if not self._allow:
            return ToolResult(success=False, error="Terminal access is disabled.")

        command: str = kwargs.get("command", "")
        if not command.strip():
            return ToolResult(success=False, error="Empty command.")

        cwd_str: Optional[str] = kwargs.get("cwd")
        cwd = Path(cwd_str).expanduser() if cwd_str else self._cwd or Path.cwd()
        timeout: int = int(kwargs.get("timeout", self._timeout))

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(cwd),
                env={**os.environ},
            )
            try:
                stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            except asyncio.TimeoutError:
                proc.kill()
                return ToolResult(
                    success=False,
                    error=f"Command timed out after {timeout}s.",
                )

            stdout = stdout_b.decode(errors="replace")
            stderr = stderr_b.decode(errors="replace")
            combined = ""
            if stdout:
                combined += stdout
            if stderr:
                combined += f"\n[stderr]\n{stderr}"

            # Truncate very long output
            if len(combined) > _MAX_OUTPUT_CHARS:
                combined = combined[:_MAX_OUTPUT_CHARS] + "\n... [truncated]"

            success = proc.returncode == 0
            return ToolResult(
                success=success,
                output=combined.strip(),
                error="" if success else f"Exit code: {proc.returncode}",
                metadata={"returncode": proc.returncode},
            )
        except Exception as exc:  # noqa: BLE001
            return ToolResult(success=False, error=str(exc))
