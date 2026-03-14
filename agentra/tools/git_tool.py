"""Git tool — version-track the agent's workspace."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from agentra.tools.base import BaseTool, ToolResult


class GitTool(BaseTool):
    """
    Run Git operations inside the agent's workspace.
    The agent can commit snapshots of its work, diff changes, roll back
    to previous states, and maintain a full audit trail.
    """

    name = "git"
    description = (
        "Perform Git version-control operations inside the workspace: "
        "init, status, diff, add, commit, log, checkout, and clone. "
        "Use this to track changes, save progress, and recover previous states."
    )

    def __init__(self, workspace_dir: Optional[Path] = None) -> None:
        self._workspace_dir = workspace_dir or Path.cwd()

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "init",
                        "status",
                        "diff",
                        "add",
                        "commit",
                        "log",
                        "checkout",
                        "branch",
                        "clone",
                        "reset",
                    ],
                    "description": "Git action to perform.",
                },
                "path": {
                    "type": "string",
                    "description": "File or directory path (for add/checkout).",
                },
                "message": {
                    "type": "string",
                    "description": "Commit message.",
                },
                "branch": {
                    "type": "string",
                    "description": "Branch name.",
                },
                "url": {
                    "type": "string",
                    "description": "Remote URL (for clone).",
                },
                "n": {
                    "type": "integer",
                    "description": "Number of log entries to return (default 10).",
                },
            },
            "required": ["action"],
        }

    async def execute(self, **kwargs: Any) -> ToolResult:
        action: str = kwargs.get("action", "")
        try:
            import git  # noqa: PLC0415
        except ImportError as exc:
            return ToolResult(
                success=False,
                error=f"gitpython not installed: {exc}",
            )

        try:
            if action == "init":
                return self._init(git)
            if action == "status":
                return self._status(git)
            if action == "diff":
                return self._diff(git)
            if action == "add":
                return self._add(git, kwargs.get("path", "."))
            if action == "commit":
                return self._commit(git, kwargs.get("message", "Agent checkpoint"))
            if action == "log":
                return self._log(git, int(kwargs.get("n", 10)))
            if action == "checkout":
                return self._checkout(git, kwargs.get("branch"), kwargs.get("path"))
            if action == "branch":
                return self._branch(git)
            if action == "clone":
                return self._clone(git, kwargs.get("url", ""))
            if action == "reset":
                return self._reset(git, kwargs.get("path"))
            return ToolResult(success=False, error=f"Unknown action: {action!r}")
        except Exception as exc:  # noqa: BLE001
            return ToolResult(success=False, error=str(exc))

    # ── private ────────────────────────────────────────────────────────────────

    def _repo(self, git_module: Any) -> Any:
        return git_module.Repo(self._workspace_dir)

    def _init(self, git_module: Any) -> ToolResult:
        self._workspace_dir.mkdir(parents=True, exist_ok=True)
        try:
            repo = self._repo(git_module)
            return ToolResult(success=True, output=f"Already a git repo: {repo.git_dir}")
        except Exception:  # noqa: BLE001
            repo = git_module.Repo.init(self._workspace_dir)
            return ToolResult(success=True, output=f"Initialized git repo at {repo.git_dir}")

    def _status(self, git_module: Any) -> ToolResult:
        repo = self._repo(git_module)
        return ToolResult(success=True, output=repo.git.status())

    def _diff(self, git_module: Any) -> ToolResult:
        repo = self._repo(git_module)
        diff = repo.git.diff()
        return ToolResult(success=True, output=diff or "No changes.")

    def _add(self, git_module: Any, path: str) -> ToolResult:
        repo = self._repo(git_module)
        repo.git.add(path)
        return ToolResult(success=True, output=f"Staged: {path}")

    def _commit(self, git_module: Any, message: str) -> ToolResult:
        repo = self._repo(git_module)
        # Ensure a git identity is set locally so commits work in any environment
        with repo.config_writer() as cw:
            if not cw.has_option("user", "email"):
                cw.set_value("user", "email", "agentra@localhost")
            if not cw.has_option("user", "name"):
                cw.set_value("user", "name", "Agentra Agent")
        # Stage all tracked + untracked
        repo.git.add("--all")
        result = repo.git.commit("-m", message, "--allow-empty")
        return ToolResult(success=True, output=result)

    def _log(self, git_module: Any, n: int) -> ToolResult:
        repo = self._repo(git_module)
        try:
            log = repo.git.log("--oneline", f"-{n}")
        except Exception:  # noqa: BLE001
            return ToolResult(success=True, output="No commits yet.")
        return ToolResult(success=True, output=log or "No commits yet.")

    def _checkout(
        self,
        git_module: Any,
        branch: Optional[str],
        path: Optional[str],
    ) -> ToolResult:
        repo = self._repo(git_module)
        if branch:
            repo.git.checkout(branch)
            return ToolResult(success=True, output=f"Checked out branch: {branch}")
        if path:
            repo.git.checkout("--", path)
            return ToolResult(success=True, output=f"Restored: {path}")
        return ToolResult(success=False, error="Provide branch or path for checkout.")

    def _branch(self, git_module: Any) -> ToolResult:
        repo = self._repo(git_module)
        branches = repo.git.branch("--list")
        return ToolResult(success=True, output=branches or "No branches.")

    def _clone(self, git_module: Any, url: str) -> ToolResult:
        if not url:
            return ToolResult(success=False, error="url is required for clone.")
        dest = self._workspace_dir / url.split("/")[-1].replace(".git", "")
        git_module.Repo.clone_from(url, dest)
        return ToolResult(success=True, output=f"Cloned {url!r} to {dest}")

    def _reset(self, git_module: Any, path: Optional[str]) -> ToolResult:
        repo = self._repo(git_module)
        if path:
            repo.git.reset("HEAD", "--", path)
            return ToolResult(success=True, output=f"Reset staged changes for: {path}")
        repo.git.reset("--hard", "HEAD")
        return ToolResult(success=True, output="Hard reset to HEAD.")
