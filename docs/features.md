# Implemented Features

This page lists implemented capabilities only. It is intentionally scoped to what the current repository already exposes through code and tests.

Use [Interfaces](interfaces.md) as the source of truth for commands, routes, fields, and tool schemas, [Policies](policies.md) for guardrail behavior, and [Audit & Gaps](audit.md) for known incomplete or risky areas.

## Operator Surfaces

### CLI

`agentra/cli.py` exposes four command groups:

- `run` executes a single goal directly from the terminal.
- `app` starts the local FastAPI operator UI.
- `config` shows the current config or writes a `.env` file.
- `workspace` shows git history or restores a prior workspace commit.

The `run` command supports both single-agent and orchestrated execution, optional HTML run reports, and provider/model overrides.

### Live Operator App

`agentra/live_app.py` provides a local operator console with:

- thread creation and thread listing
- live run status and activity summaries
- browser live-frame and live-stream endpoints
- desktop live-frame and live-stream endpoints
- approval and question handling endpoints
- manual human-action submission
- per-thread permission-mode updates
- report and log views

## Autonomous Execution

### Single-Agent ReAct Loop

`AutonomousAgent` implements a standard thought -> act -> observe loop with tool calling. It can:

- reason over a free-form goal
- select from the registered tool set
- ask the user clarifying questions when needed
- stop on a `DONE:` final answer
- surface structured error events when a run fails

### Multi-Agent Orchestration

The `Orchestrator` adds:

- planner-model task decomposition
- dependency-aware sub-task ordering
- parallel execution of independent sub-tasks
- final summary synthesis over sub-task results

## Human Supervision Features

The live runtime supports explicit human control points:

- pause and resume of an active thread
- approval requests for sensitive actions
- question requests when the agent needs more input
- manual human actions routed through the same tool surfaces used by the agent
- thread handoff state tracking between `agent` and `user`

These behaviors are covered by runtime and live-app tests under `tests/test_runtime.py` and `tests/test_live_app.py`.

## Browser And Desktop Control

### Browser Automation

The browser tool supports implemented actions for:

- navigation
- selector- or coordinate-based clicking
- typing
- key presses
- drag gestures
- scrolling
- screenshots
- text extraction
- link extraction
- HTML extraction
- visible feed-card listing
- marked feed-control clicking
- waits
- tab navigation and tab management

Browser sessions are thread-scoped in the live runtime, and live mirror endpoints can stream or snapshot the active browser surface.

### Desktop Control

The computer tool supports:

- full-screen screenshots
- mouse move, click, double-click, and right-click
- keyboard typing and key combos
- scroll and drag gestures

Desktop live frames are also exposed by the live runtime when computer control is enabled.

### Hidden Desktop Workers

Agentra supports same-machine hidden desktop workers for eligible local GUI threads.

In that flow:

- a thread gets a dedicated hidden desktop session instead of using the real interactive desktop
- the existing desktop preview routes expose the worker frame
- `Interact` routes clicks, keys, drags, and wheel input into the worker session rather than the user's visible desktop
- incompatible or unsafe surfaces pause and ask instead of silently switching to visible control

This is the default execution path for eligible local Windows app goals. Current hidden-session capture is strongest for standard windowed Windows apps; the Windows Graphics Capture path exists as an adapter boundary but is not configured in this build.

### Under-The-Hood Local Execution

Agentra also supports a non-visible local execution path for many local file and document goals.

In that flow the agent prefers:

- `local_system` to resolve Desktop-like folders and open confirmed local files or folders with the OS default handler
- `filesystem` to inspect resolved local paths
- `terminal` only when the other local tools cannot resolve the task directly

This path is separate from hidden desktop workers. Under-the-hood execution is for local tasks that do not need a GUI session at all.

## Filesystem, Terminal, Local System, And Git Tools

Implemented local tool capabilities include:

- filesystem reads, writes, appends, directory listing, mkdir, exists, copy, move, delete, and cwd lookup
- terminal command execution with timeout handling and output truncation
- native local-system folder resolution, file opening, and app launching
- workspace-scoped git init, status, diff, add, commit, log, checkout, branch, clone, and reset operations

The detailed schema surface is documented in [Interfaces](interfaces.md).

## Memory Features

Agentra persists two searchable memory scopes:

- thread-local working memory
- project-wide long-term memory

Memory entries can include:

- plain observation text
- normalized retrieval text
- tool and run metadata
- optional screenshot artifacts

Long-term memory retrieval is filtered by goal relevance and is skipped for desktop-local-only goals.

If the configured provider cannot produce embeddings, memory writes fall back to a trivial local embedding function. That preserves the memory pipeline but gives lower-quality semantic retrieval.

## Persistence And Observability

### HTML Run Reports

Every report-backed run writes:

- structured events
- frame metadata
- screenshot assets
- a rendered HTML timeline

The HTML report is continuously updated while the run is active.

### Thread Ledger And Audit Trail

The live runtime persists:

- thread snapshots in `ledger.json`
- append-only audit entries in `audit.jsonl`
- thread registry summaries in `.threads/registry.json`

Audit entries include run start and finish events, approval and question activity, human actions, and workspace checkpoint data.

### Git-Tracked Workspace

Agentra workspaces are intended to be git-tracked. The workspace manager can initialize repositories, create checkpoints, expose history, and restore previous states. Current checkpoint code can create empty commits, so checkpoint counts should not be treated as proof that files changed.

## Provider Support

The current provider registry includes:

- `openai`
- `anthropic`
- `ollama`
- `gemini`

Role-specific model selection is supported through `executor_model`, `planner_model`, `summary_model`, and `embedding_model`.

## Logging

Agentra configures a rotating application log under `.logs/agentra-app.log` in the active workspace root. The live app exposes a `/logs` HTML view for that log stream.

## Test-Covered Behavior Areas

The repository includes targeted tests for:

- autonomous-agent guardrails and completion rules
- approval policy behavior in default and full modes
- browser session management and Chrome profile handling
- live app routes, thread state, and live-browser/live-desktop endpoints
- memory persistence and retrieval behavior
- run-report persistence and rendering
- tool behavior across browser, desktop, filesystem, terminal, local-system, and git surfaces

Known gaps include direct CLI approval enforcement, CLI command tests, provider tests for OpenAI/Anthropic/Ollama, and static lint cleanup. See [Audit & Gaps](audit.md).
