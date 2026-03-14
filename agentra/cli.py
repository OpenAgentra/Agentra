"""
Agentra CLI — interact with the autonomous agent from your terminal.

Usage:
    agentra run "Apply to 10 Python jobs on LinkedIn"
    agentra run --provider ollama --model llama3 "Summarise my Downloads folder"
    agentra config show
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

console = Console()


def _print_event(event: dict) -> None:
    etype = event.get("type", "")
    if etype == "thought":
        console.print(Panel(event["content"], title="[bold cyan]🤔 Thought[/]", border_style="cyan"))
    elif etype == "tool_call":
        args_str = json.dumps(event.get("args", {}), indent=2)
        console.print(
            Panel(
                f"[yellow]Tool:[/] [bold]{event['tool']}[/]\n[dim]{args_str}[/]",
                title="[bold yellow]🔧 Tool Call[/]",
                border_style="yellow",
            )
        )
    elif etype == "tool_result":
        style = "green" if event.get("success") else "red"
        icon = "✓" if event.get("success") else "✗"
        console.print(
            Panel(
                event.get("result", ""),
                title=f"[bold {style}]{icon} Result: {event['tool']}[/]",
                border_style=style,
            )
        )
    elif etype == "screenshot":
        console.print("[dim]📸 Screenshot captured.[/]")
    elif etype == "done":
        console.print(
            Panel(
                Markdown(event["content"]),
                title="[bold green]✅ Done[/]",
                border_style="green",
            )
        )
    elif etype == "error":
        console.print(
            Panel(event["content"], title="[bold red]❌ Error[/]", border_style="red")
        )


@click.group()
def main() -> None:
    """Agentra — open-source autonomous AI agent with full computer access."""


# ── run ───────────────────────────────────────────────────────────────────────

@main.command()
@click.argument("goal")
@click.option(
    "--provider",
    default=None,
    type=click.Choice(["openai", "anthropic", "ollama"]),
    help="LLM provider (overrides AGENTRA_LLM_PROVIDER).",
)
@click.option("--model", default=None, help="Model name (overrides AGENTRA_LLM_MODEL).")
@click.option(
    "--headless/--no-headless",
    default=None,
    help="Run browser in headless mode.",
)
@click.option(
    "--workspace",
    default=None,
    type=click.Path(),
    help="Workspace directory path.",
)
@click.option(
    "--max-iterations",
    default=None,
    type=int,
    help="Maximum agent iterations.",
)
@click.option(
    "--orchestrate/--no-orchestrate",
    default=False,
    help="Use multi-agent orchestration.",
)
def run(
    goal: str,
    provider: Optional[str],
    model: Optional[str],
    headless: Optional[bool],
    workspace: Optional[str],
    max_iterations: Optional[int],
    orchestrate: bool,
) -> None:
    """Run the agent with a GOAL."""
    from agentra.config import AgentConfig  # noqa: PLC0415

    overrides: dict = {}
    if provider:
        overrides["llm_provider"] = provider
    if model:
        overrides["llm_model"] = model
    if headless is not None:
        overrides["browser_headless"] = headless
    if workspace:
        overrides["workspace_dir"] = Path(workspace)
    if max_iterations:
        overrides["max_iterations"] = max_iterations

    config = AgentConfig(**overrides)

    console.print(
        Panel(
            f"[bold]{goal}[/]",
            title="[bold blue]🤖 Agentra — Starting[/]",
            border_style="blue",
        )
    )
    console.print(
        f"[dim]Provider: {config.llm_provider} / Model: {config.llm_model} / "
        f"Workspace: {config.workspace_dir}[/]"
    )

    asyncio.run(_async_run(goal, config, orchestrate))


async def _async_run(goal: str, config: "AgentConfig", orchestrate: bool) -> None:
    if orchestrate:
        from agentra.agents.orchestrator import Orchestrator  # noqa: PLC0415

        orch = Orchestrator(config=config)
        result = await orch.run(goal)
        console.print(
            Panel(
                result.final_summary,
                title="[bold green]✅ Orchestration Complete[/]",
                border_style="green",
            )
        )
        if result.sub_tasks:
            table = Table(title="Sub-task Results")
            table.add_column("Index", style="dim")
            table.add_column("Agent")
            table.add_column("Task")
            table.add_column("Status")
            for t in result.sub_tasks:
                status = "[green]✓[/]" if t.success else "[red]✗[/]"
                table.add_row(str(t.index), t.agent_name, t.task[:60], status)
            console.print(table)
    else:
        from agentra.agents.autonomous import AutonomousAgent  # noqa: PLC0415

        agent = AutonomousAgent(config=config)
        gen = await agent.run(goal)
        async for event in gen:
            _print_event(event)


# ── config ────────────────────────────────────────────────────────────────────

@main.group()
def config() -> None:
    """Manage Agentra configuration."""


@config.command("show")
def config_show() -> None:
    """Show current configuration."""
    from agentra.config import AgentConfig  # noqa: PLC0415

    cfg = AgentConfig()
    table = Table(title="Agentra Configuration")
    table.add_column("Setting", style="bold")
    table.add_column("Value")
    for key, value in cfg.model_dump().items():
        table.add_row(key, str(value))
    console.print(table)


@config.command("init")
@click.option("--provider", prompt="LLM provider", default="openai",
              type=click.Choice(["openai", "anthropic", "ollama"]))
@click.option("--model", prompt="Model name", default="gpt-4o")
@click.option("--api-key", prompt="API key (leave empty for Ollama)", default="", hide_input=True)
def config_init(provider: str, model: str, api_key: str) -> None:
    """Create a .env file with basic configuration."""
    env_path = Path(".env")
    lines = [
        f"AGENTRA_LLM_PROVIDER={provider}",
        f"AGENTRA_LLM_MODEL={model}",
    ]
    if api_key:
        if provider == "openai":
            lines.append(f"AGENTRA_OPENAI_API_KEY={api_key}")
        elif provider == "anthropic":
            lines.append(f"AGENTRA_ANTHROPIC_API_KEY={api_key}")
    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    console.print(f"[green]Configuration written to {env_path}[/]")


# ── workspace ─────────────────────────────────────────────────────────────────

@main.group()
def workspace() -> None:
    """Manage the agent's workspace."""


@workspace.command("history")
@click.option("--n", default=20, help="Number of commits to show.")
@click.option(
    "--workspace",
    default=None,
    type=click.Path(),
    help="Workspace directory.",
)
def workspace_history(n: int, workspace: Optional[str]) -> None:
    """Show workspace git history."""
    from agentra.config import AgentConfig  # noqa: PLC0415
    from agentra.memory.workspace import WorkspaceManager  # noqa: PLC0415

    cfg = AgentConfig(**({"workspace_dir": Path(workspace)} if workspace else {}))
    mgr = WorkspaceManager(cfg.workspace_dir)
    mgr.init()
    history = mgr.history(n)
    if not history:
        console.print("[yellow]No commits found.[/]")
        return
    table = Table(title="Workspace History")
    table.add_column("SHA", style="dim")
    table.add_column("Date")
    table.add_column("Message")
    for entry in history:
        table.add_row(entry["sha"], entry["date"], entry["message"])
    console.print(table)


@workspace.command("restore")
@click.argument("sha")
@click.option(
    "--workspace",
    default=None,
    type=click.Path(),
    help="Workspace directory.",
)
def workspace_restore(sha: str, workspace: Optional[str]) -> None:
    """Restore workspace to a previous commit SHA."""
    from agentra.config import AgentConfig  # noqa: PLC0415
    from agentra.memory.workspace import WorkspaceManager  # noqa: PLC0415

    cfg = AgentConfig(**({"workspace_dir": Path(workspace)} if workspace else {}))
    mgr = WorkspaceManager(cfg.workspace_dir)
    mgr.init()
    if mgr.restore(sha):
        console.print(f"[green]Workspace restored to {sha}[/]")
    else:
        console.print(f"[red]Failed to restore to {sha}[/]")


if __name__ == "__main__":
    main()
