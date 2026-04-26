"""
Workspace manager — the agent's own git-tracked working directory.

Every session is automatically committed so the agent can recover from
any state and the user has a full audit trail of what was done.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class WorkspaceChangeSummary:
    """Serializable summary of a workspace checkpoint."""

    before_sha: str | None = None
    after_sha: str | None = None
    changed_files: list[str] = field(default_factory=list)
    diff_stats: list[dict[str, Any]] = field(default_factory=list)
    status: str = "unchanged"


class WorkspaceManager:
    """
    Manages the agent's dedicated workspace directory.

    * Creates the directory if it does not exist.
    * Initialises a git repository inside it.
    * Provides helpers to commit snapshots after each task.
    """

    def __init__(self, workspace_dir: Path, author: str = "Agentra Agent") -> None:
        self._dir = workspace_dir
        self._author = author
        self._repo: Optional[Any] = None
        self._dir.mkdir(parents=True, exist_ok=True)

    # ── setup ──────────────────────────────────────────────────────────────────

    def init(self) -> None:
        """Initialise (or open) the git repository in *workspace_dir*."""
        try:
            import git  # noqa: PLC0415
        except ImportError:
            return  # gitpython not installed; skip silently

        try:
            self._repo = git.Repo(self._dir)
        except Exception:  # noqa: BLE001
            self._repo = git.Repo.init(self._dir)
            self._write_readme()
            self._commit("chore: initialise Agentra workspace")

    # ── convenience ────────────────────────────────────────────────────────────

    @property
    def path(self) -> Path:
        return self._dir

    def snapshot(self, message: Optional[str] = None) -> bool:
        """
        Stage all changes and create a commit.

        Returns *True* if a commit was made, *False* otherwise
        (e.g. nothing changed or gitpython is not installed).
        """
        if self._repo is None:
            return False
        msg = message or f"chore: agent snapshot {time.strftime('%Y-%m-%d %H:%M:%S')}"
        summary = self.checkpoint(msg)
        return summary.status == "committed"

    def checkpoint(self, message: Optional[str] = None) -> WorkspaceChangeSummary:
        """Create a git checkpoint and return a structured diff summary."""
        if self._repo is None:
            return WorkspaceChangeSummary(status="git_unavailable")
        msg = message or f"chore: agent snapshot {time.strftime('%Y-%m-%d %H:%M:%S')}"
        before_sha = self.current_sha()
        changed_files = self._pending_changed_files()
        committed = self._commit(msg)
        after_sha = self.current_sha()
        diff_stats = self._diff_stats(before_sha, after_sha)
        status = "committed" if committed else "unchanged"
        return WorkspaceChangeSummary(
            before_sha=before_sha,
            after_sha=after_sha,
            changed_files=changed_files,
            diff_stats=diff_stats,
            status=status,
        )

    def history(self, n: int = 20) -> list[dict[str, str]]:
        """Return the last *n* commits as a list of dicts."""
        if self._repo is None:
            return []
        commits = []
        for commit in list(self._repo.iter_commits(max_count=n)):
            commits.append(
                {
                    "sha": commit.hexsha[:8],
                    "message": commit.message.strip(),
                    "author": str(commit.author),
                    "date": time.strftime(
                        "%Y-%m-%d %H:%M:%S", time.gmtime(commit.committed_date)
                    ),
                }
            )
        return commits

    def restore(self, sha: str) -> bool:
        """Hard-reset the workspace to commit *sha*. USE WITH CARE."""
        if self._repo is None:
            return False
        self._repo.git.reset("--hard", sha)
        return True

    def current_sha(self) -> str | None:
        if self._repo is None:
            return None
        try:
            return self._repo.head.commit.hexsha[:8]
        except Exception:  # noqa: BLE001
            return None

    # ── private ────────────────────────────────────────────────────────────────

    def _write_readme(self) -> None:
        readme = self._dir / "README.md"
        if not readme.exists():
            readme.write_text(
                "# Agentra Workspace\n\n"
                "This directory is managed by the Agentra autonomous agent.\n"
                "Agentra initializes and checkpoints this workspace when GitPython is available.\n",
                encoding="utf-8",
            )

    def _commit(self, message: str) -> bool:
        if self._repo is None:
            return False
        # Ensure a local git identity exists so commits work in any environment
        with self._repo.config_writer() as cw:
            if not cw.has_option("user", "email"):
                cw.set_value("user", "email", "agentra@localhost")
            if not cw.has_option("user", "name"):
                cw.set_value("user", "name", "Agentra Agent")
        self._repo.git.add("--all")
        try:
            self._repo.git.commit("-m", message, "--allow-empty")
            return True
        except Exception:  # noqa: BLE001
            return False

    def _pending_changed_files(self) -> list[str]:
        if self._repo is None:
            return []
        try:
            output = self._repo.git.status("--porcelain")
        except Exception:  # noqa: BLE001
            return []
        files: list[str] = []
        for line in output.splitlines():
            if not line.strip():
                continue
            candidate = line[3:].strip() if len(line) > 3 else line.strip()
            if candidate:
                files.append(candidate)
        return sorted(set(files))

    def _diff_stats(self, before_sha: str | None, after_sha: str | None) -> list[dict[str, Any]]:
        if self._repo is None or after_sha is None:
            return []
        try:
            if before_sha and before_sha != after_sha:
                output = self._repo.git.diff("--numstat", before_sha, after_sha)
            else:
                output = self._repo.git.show("--numstat", "--format=", after_sha)
        except Exception:  # noqa: BLE001
            return []
        stats: list[dict[str, Any]] = []
        for line in output.splitlines():
            parts = line.split("\t")
            if len(parts) != 3:
                continue
            added, deleted, path = parts
            stats.append(
                {
                    "path": path,
                    "added": _safe_int(added),
                    "deleted": _safe_int(deleted),
                }
            )
        return stats


def _safe_int(value: str) -> int:
    try:
        return int(value)
    except ValueError:
        return 0
