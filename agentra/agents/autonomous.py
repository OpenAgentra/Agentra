"""
Autonomous agent - the core ReAct (Reasoning + Acting) loop.

The agent receives a goal from the user, reasons about the next action,
calls one of its tools, observes the result, and repeats until the task
is complete or the iteration limit is reached.
"""

from __future__ import annotations

import asyncio
import logging
import re
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Optional

from agentra.approval_policy import ApprovalPolicyContext, ApprovalPolicyEngine
from agentra.config import AgentConfig
from agentra.llm.base import LLMMessage, LLMProvider, LLMSession, LLMToolResult
from agentra.llm.factory import get_embedding_provider, get_provider
from agentra.logging_utils import exception_details_with_context
from agentra.memory.embedding_memory import LongTermMemoryStore, ThreadWorkingMemory
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
    """

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        llm: Optional[LLMProvider] = None,
        tools: Optional[list[BaseTool]] = None,
        embedding_provider: Optional[Any] = None,
    ) -> None:
        self.config = config or AgentConfig()
        self.llm: LLMProvider = llm or get_provider(self.config)
        if embedding_provider is not None:
            self.embedding_provider = embedding_provider
        elif llm is not None:
            from agentra.memory.providers import LLMEmbeddingProvider

            self.embedding_provider = LLMEmbeddingProvider(self.llm)
        else:
            self.embedding_provider = get_embedding_provider(self.config)

        self.workspace = WorkspaceManager(self.config.workspace_dir)
        self.working_memory = ThreadWorkingMemory(
            memory_dir=self.config.memory_dir,
            embed_provider=self.embedding_provider,
            screenshot_history=self.config.screenshot_history,
        )
        self.long_term_memory = LongTermMemoryStore(
            memory_dir=self.config.long_term_memory_dir,
            embed_provider=self.embedding_provider,
            screenshot_history=self.config.screenshot_history,
        )
        self.memory = self.working_memory

        self.tools: dict[str, BaseTool] = {}
        if tools is not None:
            for tool in tools:
                self.tools[tool.name] = tool
        else:
            self._register_default_tools()

        self._session: Optional[LLMSession] = None
        self._running = False
        self._interrupt = asyncio.Event()
        self._resume_event = asyncio.Event()
        self._resume_event.set()
        self._runtime_controller: Any = None
        self._execution_scheduler: Any = None
        self._thread_id: str | None = None
        self._run_id: str | None = None
        self._goal: str = ""
        self._approval_engine = ApprovalPolicyEngine.default()

    async def run(self, goal: str) -> AsyncIterator[dict[str, Any]]:
        """
        Execute *goal* autonomously.

        Yields event dicts with at least a ``"type"`` key.
        """
        self.workspace.init()
        self._interrupt.clear()
        self._resume_event.set()
        self._running = True
        self._goal = goal
        self._session = self.llm.start_session(
            system_message=self._system_message(),
            tools=self._tool_schemas(),
        )

        self._session.append(LLMMessage(role="user", content=goal))
        await self._remember(goal, role="user", source_type="goal")

        iteration = 0

        async def _generator() -> AsyncIterator[dict[str, Any]]:
            nonlocal iteration
            try:
                while iteration < self.config.max_iterations and not self._interrupt.is_set():
                    await self._wait_until_resumed()
                    iteration += 1
                    logger.debug("Iteration %d/%d", iteration, self.config.max_iterations)
                    yield {
                        "type": "phase",
                        "phase": "thinking",
                        "content": "Sonraki adımı planlıyor...",
                        "summary": "Sonraki adımı planlıyor...",
                    }

                    extra_messages: list[LLMMessage] = []
                    screenshots = self.working_memory.recent_screenshots()
                    if screenshots:
                        extra_messages.append(
                            LLMMessage(
                                role="user",
                                content="Recent screenshots are attached as visual context for the current step.",
                                images=screenshots,
                            )
                        )
                    memory_message = await self._long_term_memory_message(goal)
                    if memory_message is not None:
                        extra_messages.append(memory_message)

                    response = await self._session.complete(
                        extra_messages=extra_messages or None,
                        temperature=self.config.temperature,
                        max_tokens=self.config.max_tokens,
                    )

                    if response.content:
                        yield {"type": "thought", "content": response.content}
                        await self._remember(
                            response.content,
                            role="assistant",
                            source_type="assistant_thought",
                        )
                        done_content = self._extract_done_content(response.content)
                        if done_content is not None:
                            if self._thread_id is None:
                                self.workspace.snapshot(f"task: {goal[:60]}")
                            yield {"type": "done", "content": done_content}
                            return
                        if not response.tool_calls:
                            question_event = self._prepare_question_event(response.content)
                            if question_event is not None:
                                yield question_event
                                answer = await self._wait_for_user_answer(str(question_event["request_id"]))
                                self._session.append(LLMMessage(role="user", content=answer))
                                await self._remember(answer, role="user", source_type="user_answer")
                                continue

                    if response.tool_calls:
                        tool_results: list[LLMToolResult] = []
                        for tool_call in response.tool_calls:
                            await self._wait_until_resumed()
                            tool_name: str = tool_call["name"]
                            tool_args: dict[str, Any] = tool_call["arguments"]
                            tool_call_id: str = tool_call.get("id", tool_name)
                            preview = await self._preview_tool(tool_name, tool_args)
                            approval_event = self._prepare_approval_event(tool_name, tool_args)

                            yield {
                                "type": "phase",
                                "phase": "acting",
                                "content": "İşlemi hazırlıyor...",
                                "summary": "İşlemi hazırlıyor...",
                            }

                            if approval_event is not None:
                                yield approval_event
                                approved = await self._wait_for_approval(str(approval_event["request_id"]))
                                if not approved:
                                    result = ToolResult(success=False, error="User rejected this action.")
                                    result_text = str(result)
                                    await self._remember(
                                        result_text,
                                        role="observation",
                                        source_type="tool_result",
                                        metadata={"tool": tool_name, "summary": "User rejected this action."},
                                    )
                                    yield {
                                        "type": "tool_result",
                                        "tool": tool_name,
                                        "result": result_text,
                                        "success": False,
                                        "summary": "İşlem kullanıcı tarafından reddedildi.",
                                    }
                                    tool_results.append(
                                        LLMToolResult(
                                            tool_call_id=tool_call_id,
                                            name=tool_name,
                                            content=result_text,
                                            success=False,
                                        )
                                    )
                                    continue

                            yield {"type": "tool_call", "tool": tool_name, "args": tool_args}
                            if preview:
                                yield {
                                    "type": "visual_intent",
                                    "tool": tool_name,
                                    "args": tool_args,
                                    **preview,
                                }

                            result = await self._call_tool(tool_name, tool_args)
                            async for result_event in self._emit_tool_result_events(tool_name, result):
                                yield result_event
                            result_text = str(result)
                            tool_results.append(
                                LLMToolResult(
                                    tool_call_id=tool_call_id,
                                    name=tool_name,
                                    content=result_text,
                                    success=result.success,
                                )
                            )

                        self._session.append_tool_results(tool_results)
                        continue

                    if not response.content and not response.tool_calls:
                        yield {
                            "type": "error",
                            "content": "Model returned an empty response. Stopping.",
                        }
                        return

                if self._thread_id is None:
                    self.workspace.snapshot("chore: iteration limit reached")
                yield {
                    "type": "done",
                    "content": (
                        f"Reached iteration limit ({self.config.max_iterations}). "
                        "Partial results saved to workspace."
                    ),
                }
            except Exception as exc:  # noqa: BLE001
                details = exception_details_with_context(
                    exc,
                    provider=self.config.llm_provider,
                    model=self.config.llm_model,
                )
                message = str(details.get("public_message") or str(exc))
                payload: dict[str, Any] = {"type": "error", "content": message, "details": details}
                hint = str(details.get("hint") or "")
                if hint:
                    payload["summary"] = hint
                logger.exception("Agent error: %s", message)
                yield payload
            finally:
                self._running = False
                if self._session is not None:
                    await self._session.aclose()
                    self._session = None
                await self._shutdown_tools()

        return _generator()

    def interrupt(self) -> None:
        """Signal the agent to stop after the current iteration."""
        self._interrupt.set()
        if self._runtime_controller is not None:
            cancel = getattr(self._runtime_controller, "cancel_pending", None)
            if callable(cancel):
                cancel()
        self._resume_event.set()
        logger.info("Agent interrupted by user.")

    def pause(self) -> None:
        """Pause autonomous execution until resumed."""
        self._resume_event.clear()
        if self._runtime_controller is not None:
            pause = getattr(self._runtime_controller, "pause", None)
            if callable(pause):
                pause()

    def resume(self) -> None:
        """Resume a paused autonomous execution."""
        self._resume_event.set()
        if self._runtime_controller is not None:
            resume = getattr(self._runtime_controller, "resume", None)
            if callable(resume):
                resume()

    def bind_runtime(
        self,
        *,
        controller: Any = None,
        scheduler: Any = None,
        thread_id: str | None = None,
        run_id: str | None = None,
        browser_sessions: Any = None,
        approval_engine: ApprovalPolicyEngine | None = None,
    ) -> None:
        """Attach thread-aware runtime primitives to the agent."""
        self._runtime_controller = controller
        self._execution_scheduler = scheduler
        self._thread_id = thread_id
        self._run_id = run_id
        if approval_engine is not None:
            self._approval_engine = approval_engine
        for tool in self.tools.values():
            binder = getattr(tool, "bind_runtime", None)
            if callable(binder):
                binder(browser_sessions=browser_sessions, thread_id=thread_id)

    async def take_control(self) -> None:
        """Pause the agent and let the user take over."""
        self.pause()
        while self._running and not self._interrupt.is_set():
            if self._resume_event.is_set():
                return
            await asyncio.sleep(0.1)

    async def perform_human_action(self, tool_name: str, args: dict[str, Any]) -> ToolResult:
        """Execute a manual tool action while the thread is paused for the user."""
        return await self._call_tool(tool_name, args, bypass_pause=True)

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
        for tool in tools:
            self.tools[tool.name] = tool

    async def _call_tool(
        self,
        name: str,
        args: dict[str, Any],
        *,
        bypass_pause: bool = False,
    ) -> ToolResult:
        tool = self.tools.get(name)
        if tool is None:
            return ToolResult(success=False, error=f"Unknown tool: {name!r}")
        if not bypass_pause:
            await self._wait_until_resumed()
        async with self._reserve_tool(tool):
            return await tool.execute(**args)

    async def _preview_tool(self, name: str, args: dict[str, Any]) -> dict[str, Any] | None:
        tool = self.tools.get(name)
        if tool is None:
            return None
        previewer = getattr(tool, "preview", None)
        if not callable(previewer):
            return None
        try:
            preview = previewer(**args)
            if asyncio.iscoroutine(preview):
                preview = await preview
        except Exception as exc:  # noqa: BLE001
            logger.debug("Tool preview failed for %s: %s", name, exc)
            return None
        if not isinstance(preview, dict):
            return None
        return preview

    async def _wait_until_resumed(self) -> None:
        await self._resume_event.wait()
        if self._runtime_controller is not None:
            waiter = getattr(self._runtime_controller, "wait_until_resumed", None)
            if callable(waiter):
                await waiter()

    def _prepare_question_event(self, content: str) -> dict[str, Any] | None:
        if self._runtime_controller is None or not self._looks_like_user_question(content):
            return None
        creator = getattr(self._runtime_controller, "create_question", None)
        if not callable(creator):
            return None
        request = creator(content, "Ajan kullanıcıdan ek bilgi bekliyor.")
        return {
            "type": "question_requested",
            "request_id": request.request_id,
            "content": request.prompt,
            "summary": request.summary,
        }

    async def _wait_for_user_answer(self, request_id: str) -> str:
        if self._runtime_controller is None:
            raise RuntimeError("No runtime controller is available for user answers.")
        waiter = getattr(self._runtime_controller, "wait_for_answer", None)
        if not callable(waiter):
            raise RuntimeError("Runtime controller does not support user answers.")
        return await waiter(request_id)

    def _prepare_approval_event(self, tool_name: str, tool_args: dict[str, Any]) -> dict[str, Any] | None:
        reason = self._approval_reason(tool_name, tool_args)
        if reason is None or self._runtime_controller is None:
            return None
        creator = getattr(self._runtime_controller, "create_approval", None)
        if not callable(creator):
            return None
        summary = f"{tool_name} aracı için kullanıcı onayı gerekiyor."
        request = creator(tool_name, tool_args, summary, reason)
        return {
            "type": "approval_requested",
            "request_id": request.request_id,
            "tool": tool_name,
            "args": tool_args,
            "reason": reason,
            "summary": summary,
        }

    async def _wait_for_approval(self, request_id: str) -> bool:
        if self._runtime_controller is None:
            return True
        waiter = getattr(self._runtime_controller, "wait_for_approval", None)
        if not callable(waiter):
            return True
        return await waiter(request_id)

    async def _emit_tool_result_events(self, tool_name: str, result: ToolResult) -> AsyncIterator[dict[str, Any]]:
        if result.screenshot_b64:
            screenshot_event = {"type": "screenshot", "data": result.screenshot_b64}
            screenshot_event.update(
                {
                    key: value
                    for key, value in result.metadata.items()
                    if key in {"focus_x", "focus_y", "frame_label", "summary"}
                }
            )
            await self.memory.add(
                f"Screenshot after {tool_name}",
                role="observation",
                screenshot_b64=result.screenshot_b64,
            )
            yield screenshot_event

        result_text = str(result)
        await self.memory.add(result_text, role="observation")
        tool_result_event = {
            "type": "tool_result",
            "tool": tool_name,
            "result": result_text,
            "success": result.success,
        }
        if result.metadata:
            tool_result_event["metadata"] = result.metadata
            if result.metadata.get("summary"):
                tool_result_event["summary"] = result.metadata["summary"]
        yield tool_result_event

    @asynccontextmanager
    async def _reserve_tool(self, tool: BaseTool):
        if self._execution_scheduler is None:
            yield
            return
        async with self._execution_scheduler.reserve(
            tool.capabilities,
            thread_id=self._thread_id,
            tool_name=tool.name,
        ):
            yield

    def _tool_schemas(self) -> list[dict[str, Any]]:
        return [tool.to_openai_tool() for tool in self.tools.values()]

    def _system_message(self) -> LLMMessage:
        tool_descriptions = "\n".join(
            f"- **{tool.name}**: {tool.description}" for tool in self.tools.values()
        )
        content = _SYSTEM_PROMPT.format(
            tool_descriptions=tool_descriptions,
            workspace_dir=self.config.workspace_dir,
        )
        return LLMMessage(role="system", content=content)

    @staticmethod
    def _extract_done_content(content: str) -> str | None:
        match = re.search(r"(?im)^\s*DONE:", content)
        if match is None:
            return None
        return content[match.start() :].strip()

    @staticmethod
    def _looks_like_user_question(content: str) -> bool:
        lowered = content.lower()
        if "?" in lowered:
            return True
        triggers = (
            "need clarification",
            "which",
            "can you provide",
            "please provide",
            "what should i",
            "hangi",
            "onaylıyor musun",
            "devam etmemi ister misin",
        )
        return any(trigger in lowered for trigger in triggers)

    @staticmethod
    def _approval_reason(tool_name: str, tool_args: dict[str, Any]) -> str | None:
        lowered = jsonless(tool_args)
        if tool_name == "browser":
            action = str(tool_args.get("action", "")).lower()
            if action == "type" and any(token in lowered for token in ("password", "otp", "2fa", "captcha", "secret", "token")):
                return "Secret or authentication data entry requires user approval."
            if action in {"click", "type"} and any(token in lowered for token in ("submit", "post", "send", "share", "publish", "apply", "checkout", "pay", "confirm", "delete", "remove")):
                return "This browser action may publish, submit, send, or otherwise have external effects."
        if tool_name == "terminal":
            if any(token in lowered for token in ("pip install", "npm install", "apt ", "brew ", "rm ", "del ", "git push", "shutdown", "restart")):
                return "This terminal command may install software or perform important side effects."
        if tool_name == "filesystem":
            if str(tool_args.get("action", "")).lower() in {"delete", "move"}:
                return "This filesystem action changes or removes existing files."
        if tool_name == "git":
            if str(tool_args.get("action", "")).lower() in {"reset", "checkout", "clone"}:
                return "This git action can rewrite or materially change workspace state."
        if tool_name == "computer":
            if str(tool_args.get("action", "")).lower() != "screenshot":
                return "Direct desktop control requires explicit user approval."
        return None

    def _prepare_approval_event(self, tool_name: str, tool_args: dict[str, Any]) -> dict[str, Any] | None:
        if self._runtime_controller is None:
            return None
        decision = self._approval_engine.evaluate(
            ApprovalPolicyContext(
                tool_name=tool_name,
                tool_args=tool_args,
                goal=self._goal,
                thread_id=self._thread_id,
                run_id=self._run_id,
            )
        )
        if decision.action != "require_approval":
            return None
        creator = getattr(self._runtime_controller, "create_approval", None)
        if not callable(creator):
            return None
        request = creator(
            tool_name,
            tool_args,
            decision.summary,
            decision.reason,
            rule_id=decision.rule_id,
            risk_level=decision.risk_level,
        )
        return {
            "type": "approval_requested",
            "request_id": request.request_id,
            "tool": tool_name,
            "args": tool_args,
            "reason": decision.reason,
            "summary": decision.summary,
            "rule_id": decision.rule_id,
            "risk_level": decision.risk_level,
        }

    async def _emit_tool_result_events(self, tool_name: str, result: ToolResult) -> AsyncIterator[dict[str, Any]]:
        if result.screenshot_b64:
            screenshot_event = {"type": "screenshot", "data": result.screenshot_b64}
            screenshot_event.update(
                {
                    key: value
                    for key, value in result.metadata.items()
                    if key in {"focus_x", "focus_y", "frame_label", "summary"}
                }
            )
            await self._remember(
                f"Screenshot after {tool_name}",
                role="observation",
                screenshot_b64=result.screenshot_b64,
                source_type="screenshot",
                metadata={
                    "tool": tool_name,
                    "summary": result.metadata.get("summary", ""),
                    "url": result.metadata.get("active_url", ""),
                    "active_title": result.metadata.get("active_title", ""),
                    "extracted_text": result.metadata.get("extracted_text", ""),
                },
            )
            yield screenshot_event
            for extra_frame in result.extra_screenshots:
                extra_event = {
                    "type": "screenshot",
                    "data": extra_frame.get("data", ""),
                }
                extra_event.update(
                    {
                        key: value
                        for key, value in extra_frame.items()
                        if key in {"focus_x", "focus_y", "frame_label", "summary"}
                    }
                )
                yield extra_event

        result_text = str(result)
        await self._remember(
            result_text,
            role="observation",
            source_type="tool_result",
            metadata={
                "tool": tool_name,
                "summary": result.metadata.get("summary", ""),
                "url": result.metadata.get("active_url", ""),
                "active_title": result.metadata.get("active_title", ""),
                "extracted_text": result.metadata.get("extracted_text", ""),
            },
        )
        tool_result_event = {
            "type": "tool_result",
            "tool": tool_name,
            "result": result_text,
            "success": result.success,
        }
        if result.metadata:
            tool_result_event["metadata"] = result.metadata
            if result.metadata.get("summary"):
                tool_result_event["summary"] = result.metadata["summary"]
        yield tool_result_event

    async def _remember(
        self,
        text: str,
        *,
        role: str,
        source_type: str,
        screenshot_b64: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        payload = {
            "thread_id": self._thread_id or "",
            "run_id": self._run_id or "",
            "source_type": source_type,
        }
        if metadata:
            payload.update({key: value for key, value in metadata.items() if value not in (None, "")})
        await self.working_memory.add(text, role=role, screenshot_b64=screenshot_b64, metadata=payload)
        await self.long_term_memory.add(text, role=role, screenshot_b64=screenshot_b64, metadata=payload)

    async def _long_term_memory_message(self, goal: str) -> LLMMessage | None:
        try:
            results = await self.long_term_memory.search(goal, top_k=3)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Long-term memory search failed: %s", exc)
            return None

        lines: list[str] = []
        seen: set[str] = set()
        for item in results:
            if item.metadata.get("run_id") == self._run_id:
                continue
            snippet = item.text.strip()
            if not snippet:
                continue
            compact = snippet[:220]
            if compact in seen:
                continue
            seen.add(compact)
            label = item.metadata.get("summary") or item.metadata.get("source_type") or item.role
            lines.append(f"- {label}: {compact}")
        if not lines:
            return None
        return LLMMessage(role="user", content="Relevant memory from previous runs:\n" + "\n".join(lines))

    async def _shutdown_tools(self) -> None:
        for tool in self.tools.values():
            closer = getattr(tool, "stop", None)
            if not callable(closer):
                closer = getattr(tool, "aclose", None)
            if not callable(closer):
                continue
            try:
                result = closer()
                if asyncio.iscoroutine(result):
                    await result
            except Exception as exc:  # noqa: BLE001
                logger.debug("Tool shutdown failed for %s: %s", tool.name, exc)


def jsonless(value: dict[str, Any]) -> str:
    """Collapse dict values into a lowercase string for lightweight heuristics."""
    return " ".join(str(item).lower() for item in value.values())
