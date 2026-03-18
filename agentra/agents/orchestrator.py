"""
Orchestrator — coordinate multiple autonomous agents on a single goal.

The orchestrator breaks a high-level goal into sub-tasks, assigns each
sub-task to a specialist agent, collects the results, and synthesises
a final answer.  Agents may run in parallel or sequentially depending
on their data dependencies.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from agentra.agents.autonomous import AutonomousAgent
from agentra.config import AgentConfig
from agentra.llm.base import LLMMessage, LLMProvider
from agentra.llm.factory import get_provider

logger = logging.getLogger(__name__)

_ORCHESTRATOR_PROMPT = """\
You are the Orchestrator for Agentra — a multi-agent system.

Given a high-level goal, your job is to decompose it into sub-tasks
and assign each sub-task to a specialist agent.

Available agent types:
{agent_types}

Respond with a JSON array of sub-task objects, each having:
  - "agent": one of the agent type names listed above
  - "task": the specific instruction for that agent
  - "depends_on": list of sub-task indices this task must wait for (empty list for independent tasks)

Example:
[
  {{"agent": "web", "task": "Search LinkedIn for job postings", "depends_on": []}},
  {{"agent": "web", "task": "Apply to the top 3 jobs found", "depends_on": [0]}}
]

Output ONLY the JSON array, no other text.
"""


@dataclass
class SubTask:
    """A single sub-task assigned to one agent."""

    index: int
    agent_name: str
    task: str
    depends_on: list[int]
    result: Optional[str] = None
    success: bool = False


@dataclass
class OrchestratorResult:
    """Aggregate result from the orchestrator."""

    goal: str
    sub_tasks: list[SubTask]
    final_summary: str = ""
    success: bool = False


class Orchestrator:
    """
    Decomposes a goal into sub-tasks and runs them across multiple agents.

    Example::

        orch = Orchestrator(config)
        result = await orch.run("Apply to 10 Python jobs on LinkedIn")
        print(result.final_summary)
    """

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        llm: Optional[LLMProvider] = None,
    ) -> None:
        self.config = config or AgentConfig()
        self.llm: LLMProvider = llm or get_provider(self.config, role="planner")
        self._summary_llm: LLMProvider = llm or get_provider(self.config, role="summary")
        # Agent factory — maps type names to agents (created lazily)
        self._agent_pool: dict[str, AutonomousAgent] = {}
        self._agent_types = {
            "web": "Browser-based web agent — navigates websites, fills forms, extracts data.",
            "computer": "Desktop agent — controls mouse/keyboard, reads screen.",
            "file": "Filesystem agent — reads, writes and organises files.",
            "terminal": "Terminal agent — runs shell commands, installs packages.",
            "general": "General-purpose agent — handles tasks that span multiple tools.",
        }

    async def run(self, goal: str) -> OrchestratorResult:
        """Decompose *goal* into sub-tasks and execute them."""
        logger.info("Orchestrating goal: %s", goal)
        sub_tasks = await self._plan(goal)
        await self._execute(sub_tasks)
        summary = await self._summarise(goal, sub_tasks)
        return OrchestratorResult(
            goal=goal,
            sub_tasks=sub_tasks,
            final_summary=summary,
            success=all(t.success for t in sub_tasks),
        )

    # ── planning ──────────────────────────────────────────────────────────────

    async def _plan(self, goal: str) -> list[SubTask]:
        agent_types_text = "\n".join(
            f"- {name}: {desc}" for name, desc in self._agent_types.items()
        )
        system_msg = LLMMessage(
            role="system",
            content=_ORCHESTRATOR_PROMPT.format(agent_types=agent_types_text),
        )
        user_msg = LLMMessage(role="user", content=f"Goal: {goal}")

        response = await self.llm.complete(
            messages=[system_msg, user_msg],
            temperature=0.1,
            max_tokens=2048,
        )

        import json  # noqa: PLC0415

        raw = response.content.strip()
        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        try:
            items = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Orchestrator could not parse plan JSON, running as single task")
            items = [{"agent": "general", "task": goal, "depends_on": []}]

        return [
            SubTask(
                index=i,
                agent_name=item.get("agent", "general"),
                task=item.get("task", ""),
                depends_on=item.get("depends_on", []),
            )
            for i, item in enumerate(items)
        ]

    # ── execution ─────────────────────────────────────────────────────────────

    async def _execute(self, sub_tasks: list[SubTask]) -> None:
        pending = {t.index for t in sub_tasks}
        completed: set[int] = set()

        while pending:
            # Find tasks whose dependencies are all satisfied
            ready = [
                t
                for t in sub_tasks
                if t.index in pending and all(d in completed for d in t.depends_on)
            ]
            if not ready:
                logger.error("Dependency deadlock detected — forcing remaining tasks.")
                ready = [sub_tasks[i] for i in pending]

            # Run ready tasks in parallel
            results = await asyncio.gather(
                *(self._run_sub_task(t) for t in ready), return_exceptions=True
            )

            for task, result in zip(ready, results):
                if isinstance(result, Exception):
                    task.result = str(result)
                    task.success = False
                pending.discard(task.index)
                completed.add(task.index)

    async def _run_sub_task(self, task: SubTask) -> None:
        logger.info("Running sub-task %d [%s]: %s", task.index, task.agent_name, task.task)
        agent = self._get_or_create_agent(task.agent_name)
        output_parts: list[str] = []
        try:
            gen = await agent.run(task.task)
            async for event in gen:
                if event["type"] in ("thought", "done", "tool_result"):
                    output_parts.append(event.get("content", event.get("result", "")))
        except Exception as exc:  # noqa: BLE001
            task.result = str(exc)
            task.success = False
            return

        task.result = "\n".join(output_parts)
        task.success = any("DONE:" in p for p in output_parts)

    # ── summary ───────────────────────────────────────────────────────────────

    async def _summarise(self, goal: str, sub_tasks: list[SubTask]) -> str:
        results_text = "\n\n".join(
            f"Sub-task {t.index} ({t.agent_name}): {t.task}\n"
            f"Result ({'✓' if t.success else '✗'}): {t.result or 'No output'}"
            for t in sub_tasks
        )
        prompt = (
            f"The following sub-tasks were executed to achieve this goal:\n\n"
            f"Goal: {goal}\n\n{results_text}\n\n"
            "Provide a concise final summary for the user."
        )
        response = await self._summary_llm.complete(
            messages=[LLMMessage(role="user", content=prompt)],
            temperature=0.3,
            max_tokens=512,
        )
        return response.content

    # ── agent pool ────────────────────────────────────────────────────────────

    def _get_or_create_agent(self, agent_type: str) -> AutonomousAgent:
        if agent_type not in self._agent_pool:
            self._agent_pool[agent_type] = AutonomousAgent(config=self.config)
        return self._agent_pool[agent_type]
