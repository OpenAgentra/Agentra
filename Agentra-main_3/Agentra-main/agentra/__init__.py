"""
Agentra — source-available autonomous AI agent with full computer access.

Gives an LLM the ability to control a web browser, the desktop,
the filesystem and a terminal — all coordinated through a ReAct
(Reasoning + Acting) loop.  Multiple agents can be orchestrated
together and every session is tracked in its own git workspace.
"""

from agentra.agents.autonomous import AutonomousAgent
from agentra.agents.orchestrator import Orchestrator
from agentra.config import AgentConfig
from agentra.runtime import ExecutionScheduler, ThreadManager

__version__ = "0.1.0"
__all__ = ["AutonomousAgent", "Orchestrator", "AgentConfig", "ExecutionScheduler", "ThreadManager"]
