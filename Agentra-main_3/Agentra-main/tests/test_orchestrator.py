"""Tests for the Orchestrator."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentra.agents.orchestrator import Orchestrator, SubTask
from agentra.config import AgentConfig
from agentra.llm.base import LLMMessage, LLMProvider, LLMResponse


class FakeLLM(LLMProvider):
    """Deterministic fake LLM that returns preset responses."""

    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = list(responses)
        self._index = 0

    async def complete(self, messages, tools=None, temperature=0.2, max_tokens=4096):
        if self._index < len(self._responses):
            resp = self._responses[self._index]
            self._index += 1
            return resp
        # Fallback: done
        return LLMResponse(content="DONE: finished.", finish_reason="stop")

    async def embed(self, text: str) -> list[float]:
        return [0.0] * 256


@pytest.fixture
def config(tmp_path):
    return AgentConfig(
        llm_provider="openai",
        workspace_dir=tmp_path / "ws",
        memory_dir=tmp_path / "ws" / ".memory",
        max_iterations=3,
        allow_computer_control=False,
        allow_terminal=False,
    )


# ── planning ──────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_orchestrator_parses_plan(config):
    plan_json = '[{"agent": "web", "task": "Search jobs", "depends_on": []}]'
    llm = FakeLLM(
        [
            LLMResponse(content=plan_json),              # plan
            LLMResponse(content="DONE: searched."),      # agent run
            LLMResponse(content="Summary of results."),  # summarise
        ]
    )
    orch = Orchestrator(config=config, llm=llm)
    result = await orch.run("Find Python jobs")

    assert len(result.sub_tasks) == 1
    assert result.sub_tasks[0].agent_name == "web"
    assert result.sub_tasks[0].task == "Search jobs"


@pytest.mark.asyncio
async def test_orchestrator_invalid_json_fallback(config):
    """When the model returns invalid JSON, fallback to a single general task."""
    llm = FakeLLM(
        [
            LLMResponse(content="not valid json at all!!"),  # plan (bad)
            LLMResponse(content="DONE: done."),              # agent
            LLMResponse(content="Fallback summary."),        # summarise
        ]
    )
    orch = Orchestrator(config=config, llm=llm)
    result = await orch.run("Some goal")

    assert len(result.sub_tasks) == 1
    assert result.sub_tasks[0].agent_name == "general"


@pytest.mark.asyncio
async def test_orchestrator_parallel_independent_tasks(config):
    """Independent tasks (no depends_on) should all run."""
    plan_json = (
        '[{"agent": "general", "task": "Task A", "depends_on": []}, '
        ' {"agent": "general", "task": "Task B", "depends_on": []}]'
    )
    llm = FakeLLM(
        [
            LLMResponse(content=plan_json),
            # Enough responses for both agents + summary
            LLMResponse(content="DONE: A done."),
            LLMResponse(content="DONE: B done."),
            LLMResponse(content="Both tasks completed."),
        ]
    )
    orch = Orchestrator(config=config, llm=llm)
    result = await orch.run("Do A and B")

    assert len(result.sub_tasks) == 2
    task_names = {t.task for t in result.sub_tasks}
    assert "Task A" in task_names
    assert "Task B" in task_names


@pytest.mark.asyncio
async def test_orchestrator_dependent_tasks_order(config):
    """Task B depends on Task A — both should complete."""
    plan_json = (
        '[{"agent": "general", "task": "Task A", "depends_on": []}, '
        ' {"agent": "general", "task": "Task B", "depends_on": [0]}]'
    )
    llm = FakeLLM(
        [
            LLMResponse(content=plan_json),
            LLMResponse(content="DONE: A done."),
            LLMResponse(content="DONE: B done."),
            LLMResponse(content="Sequential tasks completed."),
        ]
    )
    orch = Orchestrator(config=config, llm=llm)
    result = await orch.run("Do A then B")

    assert len(result.sub_tasks) == 2
    assert result.sub_tasks[0].task == "Task A"
    assert result.sub_tasks[1].task == "Task B"
    assert result.sub_tasks[1].depends_on == [0]


@pytest.mark.asyncio
async def test_orchestrator_result_has_summary(config):
    llm = FakeLLM(
        [
            LLMResponse(content='[{"agent": "general", "task": "do it", "depends_on": []}]'),
            LLMResponse(content="DONE: did it."),
            LLMResponse(content="Everything went well."),
        ]
    )
    orch = Orchestrator(config=config, llm=llm)
    result = await orch.run("do something")

    assert result.final_summary != ""
    assert result.goal == "do something"


@pytest.mark.asyncio
async def test_orchestrator_strips_markdown_fences(config):
    """Plan JSON wrapped in markdown fences should still parse."""
    plan_json = '```json\n[{"agent": "file", "task": "List files", "depends_on": []}]\n```'
    llm = FakeLLM(
        [
            LLMResponse(content=plan_json),
            LLMResponse(content="DONE: listed."),
            LLMResponse(content="Files listed."),
        ]
    )
    orch = Orchestrator(config=config, llm=llm)
    result = await orch.run("list files")

    assert len(result.sub_tasks) == 1
    assert result.sub_tasks[0].agent_name == "file"
