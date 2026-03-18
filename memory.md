# Agentra Project Memory

This file is the quick handoff for future chats/threads. Read this first before making changes.

## What This Project Is

Agentra is an open-source autonomous AI agent project focused on real computer-use workflows:

- browser automation
- desktop/computer control
- filesystem and terminal access
- git-aware workspace handling
- multimodal and multi-provider LLM support
- live visual playback and HTML run reports

The intent is not to build a fake demo. The guiding product direction is a real, flexible MVP that can grow into a controllable, thread-aware agent runtime.

## Main User-Facing Modes

- `agentra run ...`
  The CLI path for running the agent on a single goal.
- `agentra app ...`
  The local live web UI for watching runs, reviewing frames/events, and demoing the system visually.

## Core Code Areas

- `agentra/cli.py`
  CLI entrypoints for `run` and `app`.
- `agentra/runtime.py`
  Thread-aware runtime, thread sessions, approvals/questions/human actions, and workspace ledger/audit persistence.
- `agentra/approval_policy.py`
  Rule-based approval engine for risky tool actions.
- `agentra/browser_runtime.py`
  Shared Playwright runtime and thread-scoped browser sessions.
- `agentra/live_app.py`
  Local live UI/harness for watching and controlling runs.
- `agentra/run_report.py`
  Stores run events and renders the HTML report/timeline output.
- `agentra/run_store.py`
  Run/event/frame storage that backs reports and live playback.
- `agentra/memory/embedding_memory.py`
  Embedding-based memory storage/retrieval.
- `agentra/memory/providers.py`
  Embedding provider abstractions.
- `agentra/memory/workspace.py`
  Workspace-related memory/state handling.
- `agentra/llm/*.py`
  Provider implementations for OpenAI, Anthropic, Ollama, and Gemini.
- `agentra/tools/*.py`
  Browser, computer, filesystem, terminal, and git tools.

## Product Direction We Have Been Following

The project evolved in roughly this order:

1. Get the project running locally.
2. Make Gemini usable for the MVP.
3. Add a rich HTML report and a live local UI.
4. Avoid hardcoded demo-only behavior and keep the runtime generic.
5. Move toward a more professional backend:
   - approval policy engine
   - shared browser session
   - explicit memory architecture
   - audit/ledger trail

## Important Recent Milestones

- `c166de5`
  Thread runtime and live harness foundation.
- `004d985`
  Merge of `codex/gemini-mvp-foundation`.
- `fcec419`
  Minimal web test harness UI.
- `8ce0ee9`
  Multi-thread TV switching stabilization.
- `25df1bc`
  Approval engine and shared browser runtime.
- `c29c8d2`
  Live refresh improvements for the TV box.
- `ec27c39`
  Advanced button for hiding unnecessary test sections in the UI.
- `b89f465`
  Initial findings document for presentation/storytelling.
- `aac9baf`
  Teacher presentation brief.

## Last Active Focus

The latest meaningful project focus was the backend milestone around:

- stronger approval-policy behavior
- real shared browser session semantics
- thread-local working memory plus project-wide/long-term memory
- workspace ledger and reportable audit trail

In other words: the center of gravity moved from "make the demo look good" to "make the runtime trustworthy and explainable."

## Current High-Level State

The repo already contains:

- a thread-aware runtime
- a rule-based approval engine
- a shared browser runtime/session layer
- a local live web app
- HTML run reports
- provider abstractions for multiple LLM backends
- memory-related modules and tests

This means the project is beyond the "blank prototype" stage. Most future work should extend and harden the existing runtime instead of replacing it.

## Practical Notes For New Threads

- Read this file first, then `README.md`, then any task-specific docs such as `TEACHER_BRIEF.md`.
- Assume the goal is a real MVP, not hardcoded eyewash.
- Prefer improving the runtime and architecture over cosmetic one-off hacks unless the user explicitly asks for UI polish.
- Be careful with generated workspace artifacts and nested repos.

## Very Important Repo Notes

- The top-level repo is the product code.
- `workspace`, `workspace-windows-demo`, `workspace-windows-demo-2`, and `tmp-runtime-debug/.threads/.../workspace` are nested git workspaces / generated runtime artifacts.
- Do not casually commit those nested repos or their generated screenshots/reports/memory files unless the user explicitly wants that.
- When git looks "dirty", first verify whether the dirt is in the top-level product repo or only in those nested workspace repos.

## Environment Notes

- This project has been run in both Windows and WSL contexts.
- The repo path in WSL is:
  `/mnt/c/Users/ariba/OneDrive/Documenti/Software Projects/AI Projects/Agentra/Agentra`
- A common Windows runtime path used in previous work is:
  `C:\Users\ariba\anaconda3\python.exe`
- `PYTHONUTF8=1` has been important on Windows to avoid Unicode console issues.
- Ollama was used for early/local smoke tests.
- Gemini has been the main richer-demo provider path for the MVP.

## What To Do Next If A New Chat Resumes Work

If the user says "continue" and gives no extra direction, treat the likely next step as one of:

1. continue backend milestone work around approval/shared-browser/memory/audit
2. stabilize the live app around that backend
3. prepare a demo or teacher-facing explanation using the existing runtime

If unclear, inspect:

- recent commits
- `agentra/runtime.py`
- `agentra/approval_policy.py`
- `agentra/browser_runtime.py`
- `agentra/live_app.py`
- `agentra/run_report.py`
- `TEACHER_BRIEF.md`

## Working Assumptions To Preserve

- Keep the system generic.
- Do not hardcode a demo flow.
- Preserve user control for risky actions.
- Keep the live UI explainable without exposing hidden chain-of-thought.
- Prefer thread-aware state and auditability over ad hoc shortcuts.
