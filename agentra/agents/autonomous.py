"""
Autonomous agent — the core ReAct (Reasoning + Acting) loop.

The agent receives a goal from the user, reasons about the next action,
calls one of its tools, observes the result, and repeats until the task
is complete or the iteration limit is reached.  At any point the user
can interrupt and take back control.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, AsyncIterator, Optional

from agentra.config import AgentConfig
from agentra.llm.base import LLMMessage, LLMProvider
from agentra.llm.factory import get_provider
from agentra.memory.embedding_memory import EmbeddingMemory
from agentra.memory.workspace import WorkspaceManager
from agentra.tools.base import BaseTool, ToolResult
from agentra.tools.browser import BrowserTool
from agentra.tools.computer import ComputerTool
from agentra.tools.filesystem import FilesystemTool
from agentra.tools.git_tool import GitTool
from agentra.tools.terminal import TerminalTool

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are Agentra, an autonomous AI agent with full access to a computer.

## Available tools
{tool_descriptions}

## How to operate
1. Read the user's goal carefully.
2. Break it into steps and execute them one at a time using the tools above.
3. After each tool call, observe the result and plan the next action.
4. When the goal is fully achieved, output a final summary beginning with "DONE: ".
5. If you are unsure or need clarification, ask the user.

## Safety rules
- Never delete files unless explicitly asked.
- Do not execute commands that could cause irreversible damage without warning.
- Confirm destructive actions with the user before proceeding.
- You operate inside the workspace directory: {workspace_dir}
"""


class AutonomousAgent:
    """
    An autonomous AI agent that can browse the web, control the desktop,
    read/write files, and run terminal commands.

    Usage::

        agent = AutonomousAgent(config)
        async for event in agent.run("Apply to 5 jobs on LinkedIn"):
            print(event)
    """

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        llm: Optional[LLMProvider] = None,
        tools: Optional[list[BaseTool]] = None,
    ) -> None:
        self.config = config or AgentConfig()
        self.llm: LLMProvider = llm or get_provider(self.config)

        # Workspace + memory
        self.workspace = WorkspaceManager(self.config.workspace_dir)
        self.memory = EmbeddingMemory(
            memory_dir=self.config.memory_dir,
            embed_fn=self.llm.embed,
            screenshot_history=self.config.screenshot_history,
        )

        # Default tool set
        self.tools: dict[str, BaseTool] = {}
        if tools is not None:
            for t in tools:
                self.tools[t.name] = t
        else:
            self._register_default_tools()

        self._messages: list[LLMMessage] = []
        self._running = False
        self._interrupt = asyncio.Event()

    # ── public API ─────────────────────────────────────────────────────────────

    async def run(self, goal: str) -> AsyncIterator[dict[str, Any]]:
        """
        Execute *goal* autonomously.

        Yields event dicts with at least a ``"type"`` key:
        - ``{"type": "thought", "content": str}``
        - ``{"type": "tool_call", "tool": str, "args": dict}``
        - ``{"type": "tool_result", "tool": str, "result": str, "success": bool}``
        - ``{"type": "screenshot", "data": str}``   (base64 PNG)
        - ``{"type": "done", "content": str}``
        - ``{"type": "error", "content": str}``
        """
        self.workspace.init()
        self._interrupt.clear()
        self._running = True
        self._messages = [self._system_message()]

        # Add the user's goal
        self._messages.append(LLMMessage(role="user", content=goal))
        await self.memory.add(goal, role="user")

        iteration = 0

        async def _generator() -> AsyncIterator[dict[str, Any]]:
            nonlocal iteration
            try:
                while iteration < self.config.max_iterations and not self._interrupt.is_set():
                    iteration += 1
                    logger.debug("Iteration %d/%d", iteration, self.config.max_iterations)

                    # Optionally attach recent screenshots to give the model visual context
                    screenshots = self.memory.recent_screenshots()
                    if screenshots:
                        self._messages[-1] = LLMMessage(
                            role=self._messages[-1].role,
                            content=self._messages[-1].content,
                            images=screenshots,
                        )

                    # Ask the LLM what to do next
                    response = await self.llm.complete(
                        messages=self._messages,
                        tools=self._tool_schemas(),
                        temperature=self.config.temperature,
                        max_tokens=self.config.max_tokens,
                    )

                    # ── model returned text (thought / done) ───────────────────
                    if response.content:
                        yield {"type": "thought", "content": response.content}
                        await self.memory.add(response.content, role="assistant")
                        self._messages.append(
                            LLMMessage(role="assistant", content=response.content)
                        )
                        if response.content.strip().startswith("DONE:"):
                            self.workspace.snapshot(f"task: {goal[:60]}")
                            yield {"type": "done", "content": response.content}
                            return

                    # ── model wants to call a tool ─────────────────────────────
                    if response.tool_calls:
                        # Record the assistant's tool-call turn
                        self._messages.append(
                            LLMMessage(
                                role="assistant",
                                content=response.content,
                                tool_calls=response.tool_calls,
                            )
                        )
                        for tc in response.tool_calls:
                            tool_name: str = tc["name"]
                            tool_args: dict[str, Any] = tc["arguments"]
                            tc_id: str = tc.get("id", tool_name)

                            yield {"type": "tool_call", "tool": tool_name, "args": tool_args}

                            result: ToolResult = await self._call_tool(tool_name, tool_args)

                            if result.screenshot_b64:
                                await self.memory.add(
                                    f"Screenshot after {tool_name}",
                                    role="observation",
                                    screenshot_b64=result.screenshot_b64,
                                )
                                yield {"type": "screenshot", "data": result.screenshot_b64}

                            result_text = str(result)
                            await self.memory.add(result_text, role="observation")

                            yield {
                                "type": "tool_result",
                                "tool": tool_name,
                                "result": result_text,
                                "success": result.success,
                            }

                            # Feed result back to the conversation
                            self._messages.append(
                                LLMMessage(
                                    role="tool",
                                    content=result_text,
                                    tool_call_id=tc_id,
                                )
                            )
                        continue

                    # ── no tool calls and no "DONE" → model is stuck ───────────
                    if not response.content and not response.tool_calls:
                        yield {
                            "type": "error",
                            "content": "Model returned an empty response. Stopping.",
                        }
                        return

                # Iteration limit reached
                self.workspace.snapshot("chore: iteration limit reached")
                yield {
                    "type": "done",
                    "content": f"Reached iteration limit ({self.config.max_iterations}). "
                    "Partial results saved to workspace.",
                }
            except Exception as exc:  # noqa: BLE001
                logger.exception("Agent error: %s", exc)
                yield {"type": "error", "content": str(exc)}
            finally:
                self._running = False

        # Return the async generator
        return _generator()

    def interrupt(self) -> None:
        """Signal the agent to stop after the current iteration."""
        self._interrupt.set()
        logger.info("Agent interrupted by user.")

    async def take_control(self) -> None:
        """Pause the agent and let the user take over."""
        self.interrupt()
        while self._running:
            await asyncio.sleep(0.1)

    # ── internals ─────────────────────────────────────────────────────────────

    def _register_default_tools(self) -> None:
        tools: list[BaseTool] = [
            BrowserTool(
                headless=self.config.browser_headless,
                browser_type=self.config.browser_type,
            ),
            ComputerTool(allow=self.config.allow_computer_control),
            FilesystemTool(
                workspace_dir=self.config.workspace_dir,
                allow_write=self.config.allow_filesystem_write,
            ),
            TerminalTool(
                cwd=self.config.workspace_dir,
                allow=self.config.allow_terminal,
            ),
            GitTool(workspace_dir=self.config.workspace_dir),
        ]
        for t in tools:
            self.tools[t.name] = t

    async def _call_tool(self, name: str, args: dict[str, Any]) -> ToolResult:
        tool = self.tools.get(name)
        if tool is None:
            return ToolResult(success=False, error=f"Unknown tool: {name!r}")
        return await tool.execute(**args)

    def _tool_schemas(self) -> list[dict[str, Any]]:
        return [t.to_openai_tool() for t in self.tools.values()]

    def _system_message(self) -> LLMMessage:
        tool_descriptions = "\n".join(
            f"- **{t.name}**: {t.description}" for t in self.tools.values()
        )
        content = _SYSTEM_PROMPT.format(
            tool_descriptions=tool_descriptions,
            workspace_dir=self.config.workspace_dir,
        )
        return LLMMessage(role="system", content=content)
