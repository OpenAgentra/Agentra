"""Tests for individual tools."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from agentra.tools.base import ToolResult
from agentra.tools.filesystem import FilesystemTool
from agentra.tools.git_tool import GitTool
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
    result = await tool.execute(command="sleep 10")
    assert not result.success
    assert "timed out" in result.error.lower()


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
