# Agentra Architecture

This page maps the runtime pieces that turn a goal into tool calls, UI updates, persisted artifacts, and workspace checkpoints.

See also [Policies](policies.md) for execution rules, [Interfaces](interfaces.md) for reference surfaces, and [Artifacts](artifacts.md) for on-disk state.

## System Overview

Agentra has two main execution surfaces:

- `agentra run ...` for a direct CLI run.
- `agentra app` for the live operator UI backed by `FastAPI` and `ThreadManager`.

Both surfaces ultimately rely on the same core pieces:

- `AgentConfig` for environment-backed configuration.
- `AutonomousAgent` for the main ReAct loop.
- `BaseTool` implementations for browser, desktop, filesystem, terminal, local-system, and git actions.
- provider implementations under `agentra/llm/` for model access and embeddings.
- memory stores under `agentra/memory/`.
- reporting and persistence primitives in `RunReport`, `RunStore`, and `WorkspaceLedger`.

## Major Components

### `AgentConfig`

`agentra/config.py` is the runtime configuration root. It resolves `AGENTRA_*` environment variables, expands important paths, and applies provider and permission-mode defaults.

Important normalization rules:

- `permission_mode="full"` forces `browser_identity="chrome_profile"` and `browser_headless=False`.
- `browser_identity="chrome_profile"` also implies `permission_mode="full"`.
- workspace, working-memory, and long-term-memory paths are resolved to absolute paths.

### `AutonomousAgent`

`agentra/agents/autonomous.py` owns the single-agent execution loop.

Responsibilities:

- builds the system prompt and goal-specific guidance.
- starts an `LLMSession` with tool schemas.
- emits thought, tool call, tool result, screenshot, approval, question, done, and error events.
- applies tool guardrails before execution.
- writes observations into working memory and long-term memory.
- binds shared runtime services such as the execution scheduler, approval controller, and thread-scoped browser sessions.

### `Orchestrator`

`agentra/agents/orchestrator.py` is the multi-agent layer.

It:

- asks a planner model for a JSON sub-task plan.
- creates dependency-aware `SubTask` records.
- runs ready tasks in parallel with `asyncio.gather`.
- reuses `AutonomousAgent` instances by agent type.
- asks a summary model for the final synthesis.

The orchestrator does not introduce a second tool system; each sub-task still runs through the normal autonomous agent machinery.

### `ThreadManager`

`agentra/runtime.py` is the thread-aware runtime center.

It manages:

- thread creation and per-thread workspace layout.
- run creation, active-run tracking, and status transitions.
- pause and resume handoff state.
- approval requests, question requests, and manual human actions.
- event broadcasting for the live app.
- browser-session snapshots and live frame access.
- thread snapshots, run snapshots, and persisted ledger state.

In the live app, `ThreadManager` is the primary runtime boundary between HTTP routes and agent execution.

### `ExecutionScheduler`

`ExecutionScheduler` serializes the `computer` capability across concurrent threads. Browser, filesystem, terminal, and other non-desktop capabilities do not share the same global lock.

This means Agentra can run multiple threads at once, but only one thread can actively use the desktop-control tool at a time.

### Browser Runtime

The browser stack is split across:

- `agentra/tools/browser.py` for the tool surface exposed to the model.
- `agentra/browser_runtime.py` for shared Playwright runtimes and thread-scoped browser sessions.

Key types:

- `BrowserRuntime` owns Playwright launch state.
- `BrowserSession` owns the active context and page state for one thread.
- `BrowserSessionManager` creates and reuses sessions, tracks browser defaults, and exposes live browser frame capture.

The browser runtime supports two identities:

- `isolated`
- `chrome_profile`

`chrome_profile` mode prepares a non-default launch clone of the user's Chrome profile and is used when permission mode is `full`.

### Memory Stores

`agentra/memory/embedding_memory.py` provides disk-backed embedding memory.

Main layers:

- `ThreadWorkingMemory` stores thread-local observations.
- `LongTermMemoryStore` stores project-wide memories across runs and threads.

Each entry stores text, embedding, timestamp, optional screenshot path, metadata, and retrieval text. Both stores persist to disk and support semantic search.

### Workspace Management

`agentra/memory/workspace.py` owns the git-tracked workspace manager.

It:

- initializes a repository if possible.
- creates checkpoints with commit metadata and diff summaries.
- exposes history and restore helpers used by CLI workspace commands.

Threaded runs call into workspace checkpointing after a run finishes so that ledger audit entries can capture what changed.

### Reporting And Persistence

Reporting is split into two layers:

- `RunStore` writes structured run state to disk.
- `RunReport` wraps the store and renders the HTML timeline.

`RunStore` keeps:

- run metadata
- structured events
- frame metadata
- audit entries
- screenshot asset files

`RunReport` turns the snapshot into a standalone `index.html` page and updates it as new events arrive.

## Execution Paths

### Direct CLI Run

`agentra/cli.py -> run -> _async_run -> AutonomousAgent`

In this path:

- config comes from `AgentConfig` plus CLI overrides.
- `RunReport` writes into the configured workspace's `.runs/` directory.
- events are printed to the console through `_print_event`.
- no thread-aware live HTTP control layer is inserted.

### Live App Run

`agentra/live_app.py -> FastAPI route -> ThreadManager.start_run -> ThreadManager._run_session -> AutonomousAgent`

In this path:

- a thread is created or reused.
- the thread gets isolated workspace and memory directories.
- a runtime controller is attached for pause/resume, approvals, and user questions.
- live routes can subscribe to SSE events or pull browser/desktop frames while the run is active.

## End-To-End Event Flow

A normal tool step looks like this:

1. The user starts a run from the CLI or the live app.
2. `AgentConfig` is resolved and the agent is created.
3. `AutonomousAgent` starts an LLM session with the system prompt, tool schemas, and relevant memory.
4. The model emits a thought or tool call.
5. Tool guardrails inspect the proposed tool call before execution.
6. If needed, `ApprovalPolicyEngine` converts the step into an `approval_requested` event and waits for user input.
7. The tool runs and returns `ToolResult` data, including optional screenshots and metadata.
8. The agent writes observations into working memory and long-term memory.
9. `RunReport` and `RunStore` persist the event stream, frame data, and rendered HTML.
10. In the live app, `ThreadManager` broadcasts the stored event to subscribers and updates the thread ledger.
11. At run end, workspace checkpoint metadata is added to the thread audit trail.

## Runtime State Boundaries

Agentra keeps different kinds of state in different places:

- agent conversation state lives in the active `LLMSession`.
- thread lifecycle state lives in `ThreadSession` and `RunSession` objects.
- browser state lives in `BrowserSessionManager` and its sessions.
- persisted run state lives in `.runs/<run-id>/`.
- persisted thread state lives in `.threads/<thread-id>/ledger.json` and `audit.jsonl`.
- semantic memory lives in `.memory/` and `.memory-global/`.
- workspace file state lives in the git-tracked workspace directory.

## Where To Look When Debugging

- Agent loop behavior: `agentra/agents/autonomous.py`
- Orchestration behavior: `agentra/agents/orchestrator.py`
- Thread lifecycle and HTTP snapshots: `agentra/runtime.py`
- Browser identity and Chrome profile behavior: `agentra/browser_runtime.py`
- Report rendering and run-store persistence: `agentra/run_report.py`, `agentra/run_store.py`
- Memory persistence and retrieval: `agentra/memory/embedding_memory.py`
- Workspace git checkpoints: `agentra/memory/workspace.py`
