# Project Memory & Instructions

## 1. Project Overview
* **Project Name:** Agentra
* **Purpose:** Source-available autonomous AI agent runtime for real computer-use workflows: browser automation, desktop control, filesystem/terminal operations, memory, approvals, and live visual playback.
* **Tech Stack:** Python 3.10+, Click CLI, Rich terminal UI, FastAPI live app, Playwright, Pydantic, provider adapters for OpenAI/Anthropic/Gemini/Ollama, pytest, Ruff, Black.
* **Core Architectural Pattern:** Thread-aware Python monolith with pluggable providers, tools, runtime sessions, and report/live UI layers.

## 2. Immutable Rules (NEVER Do This)
* Never hardcode demo-only flows, URLs, or provider-specific hacks that make the MVP look fake.
* Never expose hidden chain-of-thought in the UI; only show concise reasoning summaries, action intent, and observable results.
* Never casually commit nested runtime workspaces such as `workspace/`, `workspace-windows-demo/`, `workspace-windows-demo-2/`, or `tmp-runtime-debug/.threads/.../workspace`.
* Never use destructive git commands like `git reset --hard` or revert user changes without explicit permission.
* Never bypass approval checks for risky browser, terminal, filesystem, git, or computer actions.

## 3. Coding & Style Standards (ALWAYS Do This)
* **Language:** Use Python for new backend/runtime code and keep new files typed where practical.
* **Architecture:** Extend the existing modules instead of creating parallel systems; runtime logic belongs near `agentra/runtime.py`, approvals near `agentra/approval_policy.py`, browser session behavior near `agentra/browser_runtime.py`, and live playback/report behavior near `agentra/live_app.py` and `agentra/run_report.py`.
* **Formatting:** Follow repo tooling: line length `100`, format with `black`, lint with `ruff`.
* **Naming:** `snake_case` for functions/variables/modules, `PascalCase` for classes, clear action-oriented names for runtime events and policy objects.
* **Comments:** Keep comments brief and high-signal; add them only when code flow is not obvious.
* **Behavior:** Prefer generic, reusable runtime improvements over one-off UI hacks or special cases.

## 4. Key Project Files & Directories
* `agentra/cli.py`: Main CLI entrypoints for `agentra run` and `agentra app`.
* `agentra/runtime.py`: Thread-aware runtime, approvals/questions/human actions, thread sessions, ledger persistence.
* `agentra/approval_policy.py`: Ordered approval-rule engine.
* `agentra/browser_runtime.py`: Shared Playwright runtime and thread-scoped browser sessions.
* `agentra/live_app.py`: Local live web app / TV-style interface.
* `agentra/run_report.py`: HTML run report generation and run finalization.
* `agentra/run_store.py`: Stored events/frames used by live playback and reports.
* `agentra/llm/`: Provider implementations and provider abstractions.
* `agentra/tools/`: Browser, computer, filesystem, terminal, and git tools.
* `agentra/memory/`: Embedding memory, providers, and workspace-related memory utilities.
* `tests/`: Runtime, live app, browser, approval, memory, report, and provider test coverage.
* `README.md`: Main product and setup documentation.
* `docs/audit.md`: Current dead-code, missing coverage, wrong-logic, and architecture-debt register.
* `TEACHER_BRIEF.md`: Presentation/demo narrative for explaining the project externally.

## 5. Development Workflow
* **Install Locally:** `pip install -e ".[dev]"`
* **Install Browser Runtime:** `python -m playwright install chromium`
* **Run CLI Locally:** `python -m agentra.cli run "Open python.org and summarize it"`
* **Run Live App:** `python -m agentra.cli app --provider gemini --model gemini-3-flash-preview --no-headless`
* **Running Tests:** `pytest tests -v`
* **Focused Tests:** `pytest tests/test_runtime.py tests/test_live_app.py -q`
* **Linting:** `ruff check .`
* **Formatting:** `black .`

## 6. Known Quirks & Gotchas
* On Windows, `PYTHONUTF8=1` is important to avoid Unicode/Rich console issues.
* The repo is used from both Windows and WSL; be explicit about which environment a command is meant for.
* `workspace*` paths are nested git repos / generated artifacts, so top-level `git status` can look dirty even when product code is clean.
* Gemini has been the main richer MVP/demo provider path; Ollama is useful for local smoke tests but may be limited depending on the installed model.
* `agentra/live_app.py` is large and should be edited carefully; verify you are changing the active UI/render path, not a stale duplicate block.

## 7. Memory & Context Management
* **Memory Depth:** Keep this file short enough to scan quickly; target under 200 lines.
* **Updating:** Update this file whenever architecture, workflow, major constraints, or project priorities change.
* **Resume Context:** If a new chat says `continue`, assume the most likely focus is the backend milestone around approvals, shared browser session, memory, and auditability unless the user redirects.
* **Recent Direction:** The project moved from basic setup/demo work toward a trustworthy runtime: approval engine, shared browser runtime, thread/local memory, and audit ledger/reporting.
* **External Notes:** Read `README.md` for setup and `TEACHER_BRIEF.md` for presentation/demo context.

## 8. Git/Command Guidelines
* Use Conventional Commit style when possible, e.g. `feat: ...`, `fix: ...`, `docs: ...`.
* Check whether changes are in the top-level repo or only inside generated nested workspace repos before committing.
* Keep product-code commits separate from generated run artifacts.
* Prefer non-interactive git commands.
* Before pushing, verify what branch you are on and what exact files are staged.
