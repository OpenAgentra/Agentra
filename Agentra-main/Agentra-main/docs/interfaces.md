# Interfaces Reference

This page is the contributor reference for the public and semi-public surfaces that the repository currently exposes.

For lifecycle context, read [Architecture](architecture.md). For behavior constraints, read [Policies](policies.md). For persisted output, read [Artifacts](artifacts.md).

## Python Exports

`agentra/__init__.py` exports these symbols:

| Name | Source | Purpose |
| --- | --- | --- |
| `AutonomousAgent` | `agentra/agents/autonomous.py` | Single-agent ReAct runtime |
| `Orchestrator` | `agentra/agents/orchestrator.py` | Multi-agent planning and execution |
| `AgentConfig` | `agentra/config.py` | Environment-backed runtime configuration |
| `ExecutionScheduler` | `agentra/runtime.py` | Shared capability scheduler across threads |
| `ThreadManager` | `agentra/runtime.py` | Thread-aware live runtime manager |

## `AgentConfig` Fields And Environment Variables

`AgentConfig` is a `BaseSettings` model with the `AGENTRA_` prefix. The table below lists the current fields, defaults, and matching environment variables.

| Field | Env var | Default | Notes |
| --- | --- | --- | --- |
| `llm_provider` | `AGENTRA_LLM_PROVIDER` | `openai` | Must match a provider id from the registry |
| `llm_model` | `AGENTRA_LLM_MODEL` | `gpt-4o` | Rewritten to the provider default when needed |
| `llm_vision_model` | `AGENTRA_LLM_VISION_MODEL` | same as `llm_model` when unset | Separate vision model override |
| `executor_model` | `AGENTRA_EXECUTOR_MODEL` | unset | Model used for interactive execution loops |
| `planner_model` | `AGENTRA_PLANNER_MODEL` | unset | Model used for orchestration planning |
| `summary_model` | `AGENTRA_SUMMARY_MODEL` | unset | Model used for final summaries |
| `embedding_model` | `AGENTRA_EMBEDDING_MODEL` | unset | Model used for embeddings when supported |
| `openai_api_key` | `AGENTRA_OPENAI_API_KEY` | unset | OpenAI credential |
| `anthropic_api_key` | `AGENTRA_ANTHROPIC_API_KEY` | unset | Anthropic credential |
| `ollama_base_url` | `AGENTRA_OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama REST endpoint |
| `gemini_api_key` | `AGENTRA_GEMINI_API_KEY` | unset | Gemini credential |
| `gemini_base_url` | `AGENTRA_GEMINI_BASE_URL` | `https://generativelanguage.googleapis.com` | Gemini base URL |
| `max_iterations` | `AGENTRA_MAX_ITERATIONS` | `50` | Range-limited integer |
| `max_tokens` | `AGENTRA_MAX_TOKENS` | `4096` | Per-request token cap |
| `temperature` | `AGENTRA_TEMPERATURE` | `0.2` | Sampling control |
| `workspace_dir` | `AGENTRA_WORKSPACE_DIR` | `<cwd>/workspace` | Active workspace root |
| `memory_dir` | `AGENTRA_MEMORY_DIR` | `<cwd>/workspace/.memory` | Thread or direct-run memory store |
| `long_term_memory_dir` | `AGENTRA_LONG_TERM_MEMORY_DIR` | `<cwd>/workspace/.memory-global` | Shared cross-run memory store |
| `screenshot_history` | `AGENTRA_SCREENSHOT_HISTORY` | `10` | Recent screenshots kept in context |
| `browser_headless` | `AGENTRA_BROWSER_HEADLESS` | `False` | Forced to `False` in full mode |
| `browser_type` | `AGENTRA_BROWSER_TYPE` | `chromium` | `chromium`, `firefox`, or `webkit` |
| `browser_identity` | `AGENTRA_BROWSER_IDENTITY` | `isolated` | `chrome_profile` implies full mode |
| `browser_profile_name` | `AGENTRA_BROWSER_PROFILE_NAME` | `Default` | Chrome profile directory name |
| `local_execution_mode` | `AGENTRA_LOCAL_EXECUTION_MODE` | `visible` | `visible`, `under_the_hood`, or `native` |
| `desktop_fallback_policy` | `AGENTRA_DESKTOP_FALLBACK_POLICY` | `visible_control` | `visible_control` or `pause_and_ask` |
| `desktop_execution_mode` | `AGENTRA_DESKTOP_EXECUTION_MODE` | `desktop_visible` | `desktop_visible`, `desktop_native`, or `desktop_hidden` |
| `desktop_backend_preference` | `AGENTRA_DESKTOP_BACKEND_PREFERENCE` | `visible` | `visible`, `native`, or `under_the_hood` |
| `permission_mode` | `AGENTRA_PERMISSION_MODE` | `default` | `default` or `full` |
| `allow_terminal` | `AGENTRA_ALLOW_TERMINAL` | `True` | Enables terminal tool |
| `allow_filesystem_write` | `AGENTRA_ALLOW_FILESYSTEM_WRITE` | `True` | Enables write-capable filesystem operations |
| `allow_computer_control` | `AGENTRA_ALLOW_COMPUTER_CONTROL` | `True` | Enables desktop tool usage |

Derived property:

- `threads_dir` resolves to `<workspace_dir>/.threads`

## Provider Registry

`agentra/llm/registry.py` currently registers these providers.

| Provider id | Label | Default model | Capabilities | Credential/base URL env vars |
| --- | --- | --- | --- | --- |
| `openai` | OpenAI | `gpt-4o` | `text`, `images`, `tools`, `embeddings` | `AGENTRA_OPENAI_API_KEY` |
| `anthropic` | Anthropic | `claude-3-5-sonnet-latest` | `text`, `images`, `tools` | `AGENTRA_ANTHROPIC_API_KEY` |
| `ollama` | Ollama | `llava` | `text`, `images`, `tools`, `embeddings` | `AGENTRA_OLLAMA_BASE_URL` |
| `gemini` | Gemini | `gemini-3-flash-preview` | `text`, `images`, `tools` | `AGENTRA_GEMINI_API_KEY`, `AGENTRA_GEMINI_BASE_URL` |

Provider creation flows through `create_provider()` and role-based model selection flows through `get_provider()` in `agentra/llm/factory.py`.

## CLI Commands

`agentra/cli.py` defines the following command groups and options.

### `agentra run GOAL`

Options:

- `--provider`
- `--model`
- `--headless/--no-headless`
- `--workspace`
- `--max-iterations`
- `--orchestrate/--no-orchestrate`
- `--report/--no-report`
- `--open-report/--no-open-report`

Behavior:

- creates an `AgentConfig` from environment plus overrides
- configures file logging
- optionally creates a `RunReport`
- runs either `AutonomousAgent` or `Orchestrator`

### `agentra app`

Options:

- `--host`
- `--port`
- `--provider`
- `--model`
- `--headless/--no-headless`
- `--workspace`
- `--max-iterations`
- `--open/--no-open`

Behavior:

- builds the FastAPI live app with `create_live_app()`
- configures logging
- optionally opens the browser UI
- starts `uvicorn`

### `agentra config`

Subcommands:

- `config show`
- `config init --provider --model --api-key`

`config init` writes a `.env` file in the current directory.

### `agentra workspace`

Subcommands:

- `workspace history --n --workspace`
- `workspace restore SHA --workspace`

## Live App HTTP API

`create_live_app()` currently exposes these routes.

| Method | Route | Purpose |
| --- | --- | --- |
| `GET` | `/` | HTML operator console |
| `POST` | `/runs` | Start a run |
| `GET` | `/runs/{run_id}` | Get a run snapshot |
| `POST` | `/runs/{run_id}/stop` | Interrupt a run |
| `GET` | `/runs/{run_id}/events` | SSE event stream |
| `GET` | `/runs/{run_id}/assets/{asset_name}` | Serve stored report assets |
| `GET` | `/runs/{run_id}/report` | Serve the HTML run report |
| `GET` | `/logs` | HTML log view |
| `GET` | `/threads` | List threads |
| `GET` | `/threads/{thread_id}` | Get a thread snapshot |
| `PATCH` | `/threads/{thread_id}` | Update thread settings |
| `GET` | `/threads/{thread_id}/live-frame` | Single browser frame snapshot |
| `GET` | `/threads/{thread_id}/desktop-frame` | Single desktop frame snapshot |
| `GET` | `/threads/{thread_id}/live-stream` | Multipart browser frame stream |
| `GET` | `/threads/{thread_id}/desktop-stream` | Multipart desktop frame stream |
| `POST` | `/threads/{thread_id}/pause` | Pause a thread |
| `POST` | `/threads/{thread_id}/resume` | Resume a thread |
| `POST` | `/threads/{thread_id}/approvals/{request_id}` | Approve or reject a pending action |
| `POST` | `/threads/{thread_id}/questions/{request_id}` | Answer a pending question |
| `POST` | `/threads/{thread_id}/actions` | Submit a manual human tool action |

## Live App Request Models

`agentra/live_app.py` defines these request payloads.

### `RunCreateRequest`

Fields:

- `goal`
- `thread_id`
- `thread_title`
- `provider`
- `model`
- `headless`
- `workspace`
- `max_iterations`
- `permission_mode`

### `ApprovalDecisionRequest`

Fields:

- `approved`
- `note`

### `UserAnswerRequest`

Fields:

- `answer`

### `HumanActionRequest`

Fields:

- `tool`
- `args`

### `ThreadSettingsUpdateRequest`

Fields:

- `permission_mode`
- `desktop_execution_mode`

## Tool Surface Reference

The model-facing tools are implemented under `agentra/tools/`.

| Tool | Schema shape | Supported actions or fields | Notes |
| --- | --- | --- | --- |
| `browser` | `action` enum plus action-specific args | `navigate`, `click`, `type`, `key`, `drag`, `scroll`, `screenshot`, `get_text`, `get_html`, `wait`, `back`, `forward`, `new_tab`, `close_tab` | Can bind to shared thread browser sessions |
| `computer` | `action` enum plus coordinates/text | `screenshot`, `click`, `double_click`, `right_click`, `move`, `type`, `key`, `scroll`, `drag` | Backend-selected desktop control surface for visible or hidden sessions |
| `windows_desktop` | `action` enum plus Windows app args | `launch_app`, `focus_window`, `wait_for_window`, `list_windows`, `list_controls`, `invoke_control`, `set_text`, `type_keys`, `read_window_text`, `read_status` | Structured Windows app automation; uses the thread desktop session context |
| `filesystem` | `action` enum plus `path`, `content`, `destination`, `recursive` | `read`, `write`, `append`, `list`, `mkdir`, `delete`, `exists`, `copy`, `move`, `cwd` | Path resolution can be workspace-relative |
| `terminal` | structured command request | `command`, optional `cwd`, optional `timeout` | Output is truncated and timed out defensively |
| `local_system` | `action` enum plus local OS args | `resolve_known_folder`, `open_path`, `launch_app` | Bridges WSL paths and Windows-native open behavior |
| `git` | `action` enum plus git args | `init`, `status`, `diff`, `add`, `commit`, `log`, `checkout`, `branch`, `clone`, `reset` | Intended for workspace version tracking |

## Provider And Session Interfaces

`agentra/llm/base.py` defines the shared provider contract.

Main types:

- `LLMMessage`
- `LLMToolResult`
- `LLMResponse`
- `LLMSession`
- `StatelessLLMSession`
- `LLMProvider`

Every provider must implement:

- `complete(messages, tools, temperature, max_tokens)`
- `embed(text)`

Providers can also override `start_session()` when they need a custom session implementation, as Gemini does with `GeminiSession`.
