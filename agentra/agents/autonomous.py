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
import unicodedata
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
from agentra.tools.local_system import LocalSystemTool
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

## Tool routing rules
- Use `browser` only for websites, web apps, URLs, and web pages.
- Use `local_system` to resolve known local folders and open confirmed local files or folders with the OS default handler.
- Use `computer` only for visible desktop/UI tasks that truly require on-screen interaction.
- Use `filesystem` to inspect local paths and contents. Use `terminal` only when `filesystem` or `local_system` cannot directly resolve the local task.
- After every `computer` action, inspect the latest screenshot before repeating the same click. If the screen did not visibly change, stop guessing and reassess.
- If the user asks what is inside a local folder, verify the contents with a successful local listing/read step (prefer `filesystem`) after opening it. Never invent contents from failed commands or blind clicks.
- Ignore unrelated remembered context, unrelated open tabs, and unrelated previous-run state. Never substitute a different website, account, or task just because it was used before.
- Do not take unrelated extra actions after the goal is already satisfied.

## Safety rules
- Never delete files unless explicitly asked.
- Do not execute commands that could cause irreversible damage without warning.
- Confirm destructive actions with the user before proceeding.
- You operate inside the workspace directory: {workspace_dir}

## Current goal guidance
{goal_guidance}
"""

_DESKTOP_SURFACE_TERMS = (
    "desktop",
    "masaustu",
    "taskbar",
    "gorev cubugu",
    "start menu",
    "baslat",
    "file explorer",
    "explorer",
    "folder",
    "klasor",
    "directory",
    "window",
    "pencere",
    "icon",
    "kisayol",
    "shortcut",
    "native app",
    "uygulama",
)
_VISUAL_DESKTOP_ACTION_TERMS = (
    "open",
    "launch",
    "click",
    "double click",
    "drag",
    "scroll",
    "type",
    "press",
    "enter",
    "go to",
    "show",
    "select",
    "navigate",
    "ac",
    "tikla",
    "surukle",
    "kaydir",
    "yaz",
    "bas",
    "sec",
    "goster",
    "gir",
)
_WEB_TERMS = (
    "browser",
    "tarayici",
    "web",
    "website",
    "site",
    "sayfa",
    "page",
    "repo",
    "repository",
    "github",
    "gitlab",
    "google",
    "youtube",
    "linkedin",
    "x.com",
    "twitter",
    "login",
    "log in",
    "sign in",
    "account",
    "hesap",
)
_FOLDER_CONTENT_TERMS = (
    "contents",
    "content",
    "inside",
    "what is inside",
    "what's inside",
    "list",
    "listele",
    "icerik",
    "icerigini",
    "icindekileri",
    "neler var",
    "bana soyle",
)
_LOCAL_DOCUMENT_OPEN_TERMS = (
    "powerpoint",
    "ppt",
    "pptx",
    "sunum",
    "presentation",
    "slide",
    "document",
    "dosya",
    "pdf",
    "word",
    "excel",
)
_GOAL_STOPWORDS = {
    "the",
    "and",
    "then",
    "with",
    "into",
    "from",
    "that",
    "this",
    "open",
    "click",
    "launch",
    "show",
    "select",
    "navigate",
    "please",
    "desktop",
    "folder",
    "window",
    "app",
    "taskbar",
    "browser",
    "masaustu",
    "klasor",
    "pencere",
    "uygulama",
    "gorev",
    "cubugu",
    "ac",
    "tikla",
    "goster",
    "sec",
    "gir",
    "ve",
    "ile",
    "bir",
    "icin",
    "sonra",
    "hesap",
    "site",
    "sayfa",
}


def _normalized_text(text: str) -> str:
    folded = unicodedata.normalize("NFKD", text.casefold())
    stripped = "".join(char for char in folded if not unicodedata.combining(char))
    return re.sub(r"\s+", " ", stripped).strip()


def _contains_any_phrase(text: str, phrases: tuple[str, ...]) -> bool:
    return any(phrase in text for phrase in phrases)


def _goal_tokens(text: str) -> set[str]:
    normalized = _normalized_text(text)
    tokens = set(re.findall(r"[a-z0-9]{4,}", normalized))
    return {token for token in tokens if token not in _GOAL_STOPWORDS}


def _goal_mentions_web_target(goal: str) -> bool:
    normalized = _normalized_text(goal)
    if re.search(r"https?://|www\.", normalized):
        return True
    if re.search(r"\b[a-z0-9-]+\.(com|org|net|io|ai|app|dev|co|tr)\b", normalized):
        return True
    return _contains_any_phrase(normalized, _WEB_TERMS)


def _goal_mentions_desktop_surface(goal: str) -> bool:
    normalized = _normalized_text(goal)
    if re.search(r"\b[a-z]:\\", goal, flags=re.IGNORECASE):
        return True
    if "onedrive" in normalized:
        return True
    return _contains_any_phrase(normalized, _DESKTOP_SURFACE_TERMS)


def _goal_requires_visual_desktop_control(goal: str) -> bool:
    normalized = _normalized_text(goal)
    if not _goal_mentions_desktop_surface(goal):
        return False
    return _contains_any_phrase(normalized, _VISUAL_DESKTOP_ACTION_TERMS)


def _goal_has_local_desktop_component(goal: str) -> bool:
    normalized = _normalized_text(goal)
    if not _goal_mentions_desktop_surface(goal):
        return False
    return (
        _contains_any_phrase(normalized, _VISUAL_DESKTOP_ACTION_TERMS)
        or _contains_any_phrase(normalized, _FOLDER_CONTENT_TERMS)
        or _contains_any_phrase(normalized, _LOCAL_DOCUMENT_OPEN_TERMS)
        or _contains_any_phrase(normalized, _PATH_STYLE_LOCAL_TERMS)
    )


def _goal_has_mixed_web_and_local_desktop_components(goal: str) -> bool:
    return _goal_mentions_web_target(goal) and _goal_has_local_desktop_component(goal)


def _goal_is_desktop_local_only(goal: str) -> bool:
    return _goal_has_local_desktop_component(goal) and not _goal_mentions_web_target(goal)


def _goal_requests_folder_contents(goal: str) -> bool:
    if not _goal_has_local_desktop_component(goal):
        return False
    return _contains_any_phrase(_normalized_text(goal), _FOLDER_CONTENT_TERMS)


def _goal_requests_local_document_open(goal: str) -> bool:
    if not _goal_has_local_desktop_component(goal):
        return False
    return _contains_any_phrase(_normalized_text(goal), _LOCAL_DOCUMENT_OPEN_TERMS)


def _memory_entry_matches_goal(goal: str, text: str) -> bool:
    tokens = _goal_tokens(goal)
    if not tokens:
        return True
    haystack = _normalized_text(text)
    return any(token in haystack for token in tokens)


def _recent_repeated_desktop_click(
    tool_args: dict[str, Any],
    tool_history: list[dict[str, Any]],
    *,
    tolerance: float = 24.0,
) -> bool:
    action = str(tool_args.get("action", "")).lower()
    if action not in {"click", "double_click", "right_click"}:
        return False
    x = tool_args.get("x")
    y = tool_args.get("y")
    if x is None or y is None:
        return False

    similar = 0
    for entry in reversed(tool_history[-6:]):
        if entry.get("tool") != "computer" or not entry.get("success"):
            continue
        prior_args = entry.get("args") or {}
        if str(prior_args.get("action", "")).lower() != action:
            continue
        prior_x = prior_args.get("x")
        prior_y = prior_args.get("y")
        if prior_x is None or prior_y is None:
            continue
        if abs(float(prior_x) - float(x)) <= tolerance and abs(float(prior_y) - float(y)) <= tolerance:
            similar += 1
        if similar >= 2:
            return True
    return False


def _has_successful_local_listing(tool_history: list[dict[str, Any]]) -> bool:
    return any(
        entry.get("success") and entry.get("tool") in {"filesystem", "terminal"}
        for entry in tool_history
    )


def _has_successful_local_open(tool_history: list[dict[str, Any]]) -> bool:
    return any(
        entry.get("success")
        and entry.get("tool") == "local_system"
        and str((entry.get("args") or {}).get("action", "")).lower() == "open_path"
        for entry in tool_history
    )


def _has_excessive_desktop_click_guessing(
    tool_args: dict[str, Any],
    tool_history: list[dict[str, Any]],
    *,
    threshold: int = 4,
) -> bool:
    action = str(tool_args.get("action", "")).lower()
    if action not in {"click", "double_click", "right_click"}:
        return False

    click_count = 0
    for entry in reversed(tool_history[-12:]):
        if not entry.get("success"):
            continue
        tool = entry.get("tool")
        if tool in {"filesystem", "terminal", "browser"}:
            return False
        if tool != "computer":
            continue
        prior_action = str((entry.get("args") or {}).get("action", "")).lower()
        if prior_action in {"click", "double_click", "right_click"}:
            click_count += 1
    return click_count >= threshold


def _successful_tools_used(tool_history: list[dict[str, Any]]) -> set[str]:
    return {str(entry.get("tool")) for entry in tool_history if entry.get("success")}


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
            system_message=self._system_message(goal),
            tools=self._tool_schemas(),
        )

        self._session.append(LLMMessage(role="user", content=goal))
        await self._remember(goal, role="user", source_type="goal")

        iteration = 0
        successful_tools_used: set[str] = set()
        tool_history: list[dict[str, Any]] = []

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
                            desktop_guard_message = self._done_guard_message(
                                goal,
                                tool_history,
                            )
                            if desktop_guard_message is not None:
                                self._session.append(LLMMessage(role="user", content=desktop_guard_message))
                                continue
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

                            yield {
                                "type": "phase",
                                "phase": "acting",
                                "content": "İşlemi hazırlıyor...",
                                "summary": "İşlemi hazırlıyor...",
                            }
                            yield {"type": "tool_call", "tool": tool_name, "args": tool_args}

                            guardrail_error = self._tool_guardrail_error(
                                goal,
                                tool_name,
                                tool_args,
                                successful_tools_used,
                                tool_history,
                            )
                            if guardrail_error is not None:
                                result = ToolResult(
                                    success=False,
                                    error=guardrail_error,
                                    metadata={"summary": guardrail_error},
                                )
                                async for result_event in self._emit_tool_result_events(tool_name, result):
                                    yield result_event
                                tool_results.append(
                                    LLMToolResult(
                                        tool_call_id=tool_call_id,
                                        name=tool_name,
                                        content=str(result),
                                        success=False,
                                    )
                                )
                                tool_history.append(
                                    {
                                        "tool": tool_name,
                                        "args": dict(tool_args),
                                        "success": False,
                                        "guardrail": True,
                                    }
                                )
                                continue

                            preview = await self._preview_tool(tool_name, tool_args)
                            approval_event = self._prepare_approval_event(tool_name, tool_args)

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

                            if preview:
                                yield {
                                    "type": "visual_intent",
                                    "tool": tool_name,
                                    "args": tool_args,
                                    **preview,
                                }

                            result = await self._call_tool(tool_name, tool_args)
                            tool_history.append(
                                {
                                    "tool": tool_name,
                                    "args": dict(tool_args),
                                    "success": result.success,
                                    "result": str(result),
                                }
                            )
                            if result.success:
                                successful_tools_used.add(tool_name)
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
            FilesystemTool(
                workspace_dir=self.config.workspace_dir,
                allow_write=self.config.allow_filesystem_write,
            ),
            LocalSystemTool(),
            ComputerTool(allow=self.config.allow_computer_control),
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

    def _system_message(self, goal: str) -> LLMMessage:
        tool_descriptions = "\n".join(
            f"- **{tool.name}**: {tool.description}" for tool in self.tools.values()
        )
        content = _SYSTEM_PROMPT.format(
            tool_descriptions=tool_descriptions,
            workspace_dir=self.config.workspace_dir,
            goal_guidance=self._goal_guidance(goal),
        )
        return LLMMessage(role="system", content=content)

    def _goal_guidance(self, goal: str) -> str:
        if _goal_has_local_desktop_component(goal):
            if self._uses_under_the_hood_local_execution(goal):
                guidance = (
                    "This goal includes a local desktop/file step, and this run is configured for "
                    "under-the-hood local execution. Do not use `computer` automatically. For the "
                    "local step, first use `local_system` to resolve known folders such as Desktop, "
                    "then use `filesystem` to inspect the resolved WSL path, and use "
                    "`local_system` `open_path` to open the confirmed file or folder with the "
                    "native OS. Only use `terminal` if `filesystem` or `local_system` cannot "
                    "resolve the local path. If you cannot resolve the target confidently, ask the "
                    "user instead of switching to visible desktop automation."
                )
                if _goal_has_mixed_web_and_local_desktop_components(goal):
                    guidance = (
                        "This goal has both web and local steps. Use `browser` for the website part "
                        "first, then complete the local desktop/file step with under-the-hood local "
                        "execution. "
                    ) + guidance
                    guidance += (
                        " Do not say DONE after only the website step. The goal is complete only "
                        "after both the requested web step and the requested local open/listing step "
                        "succeed."
                    )
                else:
                    guidance += " Do not use `browser` unless the user explicitly mentions a website or URL."
                guidance += (
                    " `local_system resolve_known_folder` returns the real local Desktop path, including "
                    "OneDrive-backed Desktop locations when present."
                )
                if _goal_requests_folder_contents(goal):
                    guidance += (
                        " The user also wants the folder contents. Verify the actual items with a "
                        "successful `filesystem` listing of the resolved path before you say DONE."
                    )
                if _goal_requests_local_document_open(goal):
                    guidance += (
                        " This goal names a local document or presentation. Follow this order: "
                        "1) resolve the Desktop path with `local_system`, 2) locate the exact folder "
                        "or file with `filesystem`, 3) open the confirmed item with "
                        "`local_system open_path`."
                    )
                return guidance
            if _goal_has_mixed_web_and_local_desktop_components(goal):
                guidance = (
                    "This goal has both web and visible local desktop steps. Use `browser` for the "
                    "website step, then use `computer` for the on-screen local desktop step. Do not "
                    "say DONE after only the browser step; the local step must also be completed."
                )
                guidance += " " + self._desktop_path_guidance()
                if _goal_requests_folder_contents(goal):
                    guidance += (
                        " The user also wants the folder contents. After the local step, verify the "
                        "actual items with a successful local listing/read step, preferably "
                        "`filesystem`, before you say DONE."
                    )
                if _goal_requests_local_document_open(goal):
                    guidance += (
                        " This goal names a local document or presentation. After the web step, "
                        "resolve the exact local folder or file path with `filesystem` or a valid "
                        "`terminal` command, then open the confirmed item on screen. Do not keep "
                        "double-clicking guesses."
                    )
                return guidance
            guidance = (
                "This is a local desktop task. Prefer `computer` for opening folders, apps, or "
                "windows on screen. `terminal` or `filesystem` may help you discover paths, but "
                "they do not count as visually opening the requested folder or app. Do not use "
                "`browser` unless the user explicitly mentions a website or URL."
            )
            guidance += " " + self._desktop_path_guidance()
            if _goal_requests_folder_contents(goal):
                guidance += (
                    " The user also wants the folder contents. After opening the folder on screen, "
                    "verify the actual items with a successful local listing/read step, preferably "
                    "`filesystem`, before you say DONE."
                )
            if _goal_requests_local_document_open(goal):
                guidance += (
                    " This goal names a local document or presentation. First resolve the exact "
                    "folder or file path with `filesystem` or a valid `terminal` command, then "
                    "open the confirmed item on screen. Do not keep double-clicking guesses."
                )
            return guidance
        if _goal_requires_visual_desktop_control(goal):
            return (
                "This goal includes visible desktop interaction. Use `computer` to complete any "
                "desktop/UI step, and only use `browser` for an explicitly requested website."
            )
        if _goal_mentions_web_target(goal):
            return (
                "This goal is web-oriented. Prefer `browser` for website actions and ignore "
                "unrelated desktop or previous-run context."
            )
        return (
            "Pick the tool that directly matches the user's requested environment, and ignore "
            "unrelated memories or stale browser state."
        )

    def _uses_under_the_hood_local_execution(self, goal: str) -> bool:
        return (
            self.config.local_execution_mode == "under_the_hood"
            and _goal_has_local_desktop_component(goal)
        )

    def _desktop_path_guidance(self) -> str:
        workspace_text = str(self.config.workspace_dir).replace("\\", "/")
        match = re.search(r"/mnt/([a-z])/Users/([^/]+)/", workspace_text, re.IGNORECASE)
        if match:
            drive_letter, username = match.groups()
            desktop_path = f"/mnt/{drive_letter.lower()}/Users/{username}/Desktop"
            onedrive_desktop_path = f"/mnt/{drive_letter.lower()}/Users/{username}/OneDrive/Desktop"
        else:
            desktop_path = "/mnt/c/Users/<user>/Desktop"
            onedrive_desktop_path = "/mnt/c/Users/<user>/OneDrive/Desktop"
        return (
            "The `terminal` and `filesystem` tools run in WSL/Linux path space, not raw Windows "
            f"paths. Prefer paths like `{desktop_path}` or `{onedrive_desktop_path}`; if you need "
            "native Windows shell behavior, use `powershell.exe` explicitly."
        )

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

    def _tool_guardrail_error(
        self,
        goal: str,
        tool_name: str,
        tool_args: dict[str, Any],
        successful_tools_used: set[str],
        tool_history: list[dict[str, Any]],
    ) -> str | None:
        if tool_name == "browser":
            if "computer" in successful_tools_used:
                return None
            if not _goal_is_desktop_local_only(goal):
                return None
            action = str(tool_args.get("action", "")).lower()
            if action in {"navigate", "click", "type", "scroll", "drag", "back", "forward", "new_tab"}:
                if self._uses_under_the_hood_local_execution(goal):
                    return (
                        "This is an under-the-hood local task, not a website task. Use "
                        "`local_system` plus `filesystem` to resolve and open the requested local "
                        "folder or file instead of using the browser."
                    )
                return (
                    "This goal is a local desktop/folder task. Use the computer tool to open the "
                    "requested folder or window on screen before using the browser."
                )
            return None
        if tool_name == "local_system" and self._uses_under_the_hood_local_execution(goal):
            action = str(tool_args.get("action", "")).lower()
            if action == "open_path" and not str(tool_args.get("path", "") or "").strip():
                return "Resolve the exact local file or folder path before calling `local_system open_path`."
            return None
        if tool_name == "terminal" and _goal_has_local_desktop_component(goal):
            command = str(tool_args.get("command", "") or "")
            lowered_command = command.casefold()
            if self._uses_under_the_hood_local_execution(goal):
                if re.search(r"\b(ls|dir|find|fd|pwd|cd)\b", lowered_command):
                    return (
                        "For this under-the-hood local task, prefer `filesystem` for path discovery "
                        "and listings. Use `terminal` only if `filesystem` or `local_system` cannot "
                        "resolve the needed local path."
                    )
            if re.search(r"[a-zA-Z]:\\", command) and "powershell.exe" not in lowered_command:
                return (
                    "The terminal tool runs in WSL/Linux path space. Do not pass raw Windows "
                    "paths like `C:\\...` to shell commands here. Use `/mnt/c/Users/...` paths, "
                    "or call `powershell.exe` explicitly if you need Windows-native behavior."
                )
        if tool_name == "filesystem" and _goal_has_local_desktop_component(goal):
            path = str(tool_args.get("path", "") or "")
            if re.search(r"^[a-zA-Z]:\\", path):
                return (
                    "The filesystem tool resolves paths in WSL/Linux space. Use `/mnt/c/Users/...` "
                    "style paths instead of raw `C:\\...` Windows paths."
                )
        if tool_name == "computer" and _goal_has_local_desktop_component(goal):
            if self._uses_under_the_hood_local_execution(goal):
                return (
                    "This run is configured for under-the-hood local execution, so visible desktop "
                    "automation is disabled by default. Resolve the actual local path with "
                    "`local_system` and `filesystem`, then open the confirmed item with "
                    "`local_system open_path`. If you still cannot continue confidently, ask the "
                    "user instead of using the computer tool."
                )
            if _recent_repeated_desktop_click(tool_args, tool_history):
                return (
                    "You have already repeated this same desktop click at nearly the same "
                    "coordinates multiple times. Stop guessing coordinates, inspect the latest "
                    "screenshot, and choose a different target or use a path/listing step before "
                    "continuing."
                )
            if _has_excessive_desktop_click_guessing(tool_args, tool_history):
                return (
                    "You have already tried several desktop click or double-click guesses without "
                    "real progress. Stop brute-forcing the UI. Resolve the local folder/file path "
                    "with `filesystem` or a valid WSL/PowerShell command, then continue from the "
                    "confirmed target."
                )
        return None

    def _done_guard_message(self, goal: str, tool_history: list[dict[str, Any]]) -> str | None:
        successful_tools_used = _successful_tools_used(tool_history)
        if _goal_has_mixed_web_and_local_desktop_components(goal) and "browser" not in successful_tools_used:
            return (
                "This goal includes both a web step and a local step. Complete the requested "
                "browser action first, then continue to the local step before you say DONE."
            )
        if self._uses_under_the_hood_local_execution(goal):
            if _goal_requests_folder_contents(goal) and not _has_successful_local_listing(tool_history):
                return (
                    "The user asked for the contents of a local folder. Do not guess from partial "
                    "path resolution alone. First verify the actual items with a successful "
                    "`filesystem` or `terminal` listing of the resolved local path, then continue."
                )
            if not _goal_requests_folder_contents(goal) and not _has_successful_local_open(tool_history):
                return (
                    "This under-the-hood local task is not complete yet. Resolve the exact local "
                    "file or folder path, then open the confirmed item with `local_system "
                    "open_path` before you say DONE."
                )
            if _goal_requests_local_document_open(goal) and not _has_successful_local_listing(tool_history):
                return (
                    "The user asked you to open a local document or presentation. Confirm the "
                    "actual folder or file path with `filesystem` before you finish."
                )
            return None
        if _goal_has_local_desktop_component(goal) and "computer" not in successful_tools_used:
            if _goal_has_mixed_web_and_local_desktop_components(goal):
                return (
                    "This goal also includes a local desktop/UI step. A completed website step does "
                    "not finish the task. Use the computer tool to complete the requested on-screen "
                    "local action, then continue."
                )
            return (
                "The user asked for a desktop/UI action. A terminal listing or unrelated browser tab "
                "does not count as opening the requested folder or window. Use the computer tool to "
                "visually open it on screen, then continue."
            )
        if _goal_requests_folder_contents(goal) and not _has_successful_local_listing(tool_history):
            return (
                "The user asked you to report the contents of a local desktop folder. Do not guess "
                "from clicks or failed commands. After opening the folder on screen, verify the "
                "actual items with a successful `filesystem` or `terminal` listing of the resolved "
                "local path, then continue."
            )
        if _goal_requests_local_document_open(goal) and not _has_successful_local_listing(tool_history):
            return (
                "The user asked you to open a local desktop document or presentation. Do not guess "
                "which icon or file to open from repeated clicks alone. First confirm the local "
                "folder or file path with a successful `filesystem` or `terminal` step, then "
                "continue."
            )
        return None

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
        if _goal_is_desktop_local_only(goal):
            return None
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
            text_to_match = "\n".join(
                filter(
                    None,
                    (
                        item.text,
                        item.retrieval_text,
                        str(item.metadata.get("summary", "")),
                        str(item.metadata.get("url", "")),
                        str(item.metadata.get("active_url", "")),
                        str(item.metadata.get("active_title", "")),
                    ),
                )
            )
            if not _memory_entry_matches_goal(goal, text_to_match):
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
