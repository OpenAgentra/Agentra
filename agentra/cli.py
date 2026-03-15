"""Agentra CLI."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Optional

import click
from agentra.llm.registry import get_provider_spec, provider_ids
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table


def _enable_utf8_console() -> None:
    """Avoid Windows Unicode crashes when Rich renders non-ASCII output."""
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except ValueError:
                pass


_enable_utf8_console()
console = Console()


def _runtime_overrides(
    provider: Optional[str],
    model: Optional[str],
    headless: Optional[bool],
    workspace: Optional[str],
    max_iterations: Optional[int],
) -> dict:
    overrides: dict = {}
    if provider:
        overrides["llm_provider"] = provider
        if not model:
            overrides["llm_model"] = get_provider_spec(provider).default_model
    if model:
        overrides["llm_model"] = model
    if headless is not None:
        overrides["browser_headless"] = headless
    if workspace:
        workspace_path = Path(workspace)
        overrides["workspace_dir"] = workspace_path
        overrides["memory_dir"] = workspace_path / ".memory"
    if max_iterations:
        overrides["max_iterations"] = max_iterations
    return overrides


def _print_event(event: dict) -> None:
    event_type = event.get("type", "")
    if event_type == "thought":
        console.print(
            Panel(event["content"], title="[bold cyan]Thought[/]", border_style="cyan")
        )
    elif event_type == "tool_call":
        args_str = json.dumps(event.get("args", {}), indent=2)
        console.print(
            Panel(
                f"[yellow]Tool:[/] [bold]{event['tool']}[/]\n[dim]{args_str}[/]",
                title="[bold yellow]Tool Call[/]",
                border_style="yellow",
            )
        )
    elif event_type == "tool_result":
        style = "green" if event.get("success") else "red"
        label = "Result" if event.get("success") else "Error Result"
        console.print(
            Panel(
                event.get("result", ""),
                title=f"[bold {style}]{label}: {event['tool']}[/]",
                border_style=style,
            )
        )
    elif event_type == "screenshot":
        path = event.get("image_path")
        if path:
            console.print(f"[dim]Screenshot captured: {path}[/]")
        else:
            console.print("[dim]Screenshot captured.[/]")
    elif event_type == "done":
        console.print(
            Panel(
                Markdown(event["content"]),
                title="[bold green]Done[/]",
                border_style="green",
            )
        )
    elif event_type == "error":
        console.print(
            Panel(event["content"], title="[bold red]Error[/]", border_style="red")
        )


@click.group()
def main() -> None:
    """Agentra - open-source autonomous AI agent with full computer access."""


@main.command()
@click.argument("goal")
@click.option(
    "--provider",
    default=None,
    type=click.Choice(provider_ids()),
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
@click.option(
    "--report/--no-report",
    default=True,
    help="Save a visual HTML report for the run.",
)
@click.option(
    "--open-report/--no-open-report",
    default=False,
    help="Open the generated HTML report in your browser.",
)
def run(
    goal: str,
    provider: Optional[str],
    model: Optional[str],
    headless: Optional[bool],
    workspace: Optional[str],
    max_iterations: Optional[int],
    orchestrate: bool,
    report: bool,
    open_report: bool,
) -> None:
    """Run the agent with a GOAL."""
    from agentra.config import AgentConfig  # noqa: PLC0415
    from agentra.run_report import RunReport  # noqa: PLC0415

    overrides = _runtime_overrides(provider, model, headless, workspace, max_iterations)
    config = AgentConfig(**overrides)
    reporter = RunReport(config.workspace_dir, goal, config.llm_provider, config.llm_model) if report else None

    console.print(
        Panel(
            f"[bold]{goal}[/]",
            title="[bold blue]Agentra Starting[/]",
            border_style="blue",
        )
    )
    console.print(
        f"[dim]Provider: {config.llm_provider} / Model: {config.llm_model} / "
        f"Workspace: {config.workspace_dir}[/]"
    )
    if reporter:
        console.print(f"[dim]Run report: {reporter.html_path}[/]")
        if open_report:
            reporter.open()

    asyncio.run(_async_run(goal, config, orchestrate, reporter))


@main.command()
@click.option("--host", default="127.0.0.1", show_default=True, help="Host for the local app.")
@click.option("--port", default=8765, show_default=True, type=int, help="Port for the local app.")
@click.option(
    "--provider",
    default=None,
    type=click.Choice(provider_ids()),
    help="LLM provider (overrides AGENTRA_LLM_PROVIDER).",
)
@click.option("--model", default=None, help="Model name (overrides AGENTRA_LLM_MODEL).")
@click.option(
    "--headless/--no-headless",
    default=None,
    help="Run browser in headless mode for app-triggered runs.",
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
    help="Maximum agent iterations per run.",
)
@click.option(
    "--open/--no-open",
    "open_browser",
    default=True,
    help="Open the local app in your browser after startup.",
)
def app(
    host: str,
    port: int,
    provider: Optional[str],
    model: Optional[str],
    headless: Optional[bool],
    workspace: Optional[str],
    max_iterations: Optional[int],
    open_browser: bool,
) -> None:
    """Start the local live operator UI."""
    from agentra.config import AgentConfig  # noqa: PLC0415
    from agentra.live_app import create_live_app, open_live_app  # noqa: PLC0415

    overrides = _runtime_overrides(provider, model, headless, workspace, max_iterations)
    config = AgentConfig(**overrides)
    web_app = create_live_app(config)
    url = f"http://{host}:{port}/"

    console.print(
        Panel(
            f"[bold]{url}[/]\n\n"
            f"[dim]Provider: {config.llm_provider} / Model: {config.llm_model}[/]\n"
            f"[dim]Workspace: {config.workspace_dir}[/]",
            title="[bold blue]Agentra App[/]",
            border_style="blue",
        )
    )

    if open_browser:
        open_live_app(url)

    import uvicorn  # noqa: PLC0415

    uvicorn.run(web_app, host=host, port=port, log_level="info")


async def _async_run(goal: str, config: "AgentConfig", orchestrate: bool, reporter: Optional["RunReport"]) -> None:
    status = "partial"
    try:
        if orchestrate:
            from agentra.agents.orchestrator import Orchestrator  # noqa: PLC0415

            orch = Orchestrator(config=config)
            result = await orch.run(goal)
            if reporter:
                reporter.record(
                    {
                        "type": "thought",
                        "content": f"Planned {len(result.sub_tasks)} sub-task(s) for orchestration.",
                    }
                )
                for task in result.sub_tasks:
                    reporter.record(
                        {
                            "type": "sub_task",
                            "label": f"{task.agent_name}: {task.task}",
                            "result": task.result or "No output captured.",
                            "success": task.success,
                        }
                    )
                reporter.record({"type": "done", "content": result.final_summary})

            console.print(
                Panel(
                    result.final_summary,
                    title="[bold green]Orchestration Complete[/]",
                    border_style="green",
                )
            )
            if result.sub_tasks:
                table = Table(title="Sub-task Results")
                table.add_column("Index", style="dim")
                table.add_column("Agent")
                table.add_column("Task")
                table.add_column("Status")
                for task in result.sub_tasks:
                    status_text = "[green]OK[/]" if task.success else "[red]FAIL[/]"
                    table.add_row(str(task.index), task.agent_name, task.task[:60], status_text)
                console.print(table)
            status = "completed" if result.success else "partial"
            return

        from agentra.agents.autonomous import AutonomousAgent  # noqa: PLC0415

        agent = AutonomousAgent(config=config)
        gen = await agent.run(goal)
        async for raw_event in gen:
            event = reporter.record(raw_event) if reporter else raw_event
            _print_event(event)
            if event["type"] == "done":
                status = "completed"
            elif event["type"] == "error":
                status = "error"
    finally:
        if reporter:
            reporter.finalize(status)
            console.print(f"[dim]Saved report to {reporter.html_path}[/]")


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
@click.option(
    "--provider",
    prompt="LLM provider",
    default="openai",
    type=click.Choice(provider_ids()),
)
@click.option("--model", prompt="Model name", default=get_provider_spec("openai").default_model)
@click.option(
    "--api-key",
    prompt="API key (leave empty for local providers)",
    default="",
    hide_input=True,
)
def config_init(provider: str, model: str, api_key: str) -> None:
    """Create a .env file with basic configuration."""
    env_path = Path(".env")
    spec = get_provider_spec(provider)
    openai_default = get_provider_spec("openai").default_model
    if provider != "openai" and model == openai_default:
        model = spec.default_model

    lines = [
        f"AGENTRA_LLM_PROVIDER={provider}",
        f"AGENTRA_LLM_MODEL={model}",
    ]
    if api_key and spec.api_key_env:
        lines.append(f"{spec.api_key_env}={api_key}")
    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    console.print(f"[green]Configuration written to {env_path}[/]")


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
