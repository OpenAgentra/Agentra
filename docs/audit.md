# Audit & Gaps

This page records the current known gaps, wrong-logic risks, stale-code candidates, and cleanup signals found during the April 26, 2026 documentation audit. It is intentionally direct: items listed here are not endorsed architecture unless another docs page explicitly says so.

## Confirmed High-Risk Logic Gaps

### Direct CLI approval bypass

The approval engine is only actionable when `AutonomousAgent` has a runtime controller. The live app binds that controller through `ThreadManager`, but direct `agentra run ...` creates `AutonomousAgent` without one. In that path, `_prepare_approval_event()` returns `None`, so documented approval rules do not pause the CLI run.

Files:

- `agentra/cli.py`
- `agentra/agents/autonomous.py`
- `agentra/runtime.py`
- `agentra/approval_policy.py`

Correct architecture: approval enforcement should be consistent across direct CLI and live app paths, or the CLI should clearly run in a documented non-interactive policy mode that blocks or rejects approval-required actions instead of silently continuing.

### Orchestrator reuses agents across parallel tasks

`Orchestrator._execute()` can run ready tasks in parallel with `asyncio.gather()`, while `_get_or_create_agent()` reuses one `AutonomousAgent` per agent type. Two parallel `web` tasks can therefore share session state, tool instances, browser state, and memory writes.

Files:

- `agentra/agents/orchestrator.py`
- `agentra/agents/autonomous.py`

Correct architecture: create an isolated `AutonomousAgent` per sub-task, or serialize sub-tasks that share an agent instance. Shared memory/reporting should be explicit and thread-safe.

### Python version mismatch

`pyproject.toml` declares Python `>=3.10` and classifiers for Python 3.10 and 3.11, but `agentra/tools/local_system.py` uses a backslash inside an f-string expression, which is only valid in Python 3.12+.

Files:

- `pyproject.toml`
- `agentra/tools/local_system.py`

Correct architecture: either keep Python 3.10 support and rewrite the f-string expression, or raise `requires-python` and classifiers to the real minimum version.

## Medium-Risk Architecture Debt

### Stale live runtime classes

`agentra/live_app.py` still defines `LiveRunSession` and `LiveRunManager`, but `create_live_app()` uses `ThreadManager` for the active live runtime. These classes look like a superseded single-run runtime.

Correct architecture: keep `ThreadManager` as the single live runtime boundary and remove or migrate stale live-run manager code after confirming no external import depends on it.

### Routing logic drift

Goal routing appears in both `agentra/task_routing.py` and `agentra/agents/autonomous.py`. The routing module handles Turkish dotless `ı` normalization, while the agent-local normalization copy does not. This can make policy selection and guardrail behavior diverge for the same goal text.

Correct architecture: make `task_routing.py` the canonical routing and normalization module, and have agent guardrails call shared functions instead of keeping a parallel phrase table.

### Workspace checkpoints can be empty

`WorkspaceManager` uses `git commit --allow-empty` for checkpoints. That means a checkpoint can exist and report a commit SHA even when no files changed.

Correct architecture: distinguish `unchanged`, `committed_with_changes`, and `committed_empty` in checkpoint metadata, or avoid empty commits when the docs promise that a checkpoint means workspace changes were captured.

### Embedding support is partial

OpenAI and Ollama are the intended embedding-capable providers. Anthropic and Gemini raise `NotImplementedError` for `embed()`, and memory stores fall back to trivial local embeddings when provider embedding fails. That preserves writes but weakens semantic retrieval.

Correct architecture: document degraded retrieval, expose embedding provider choice clearly, and add tests for provider-specific fallback behavior.

## Dead-Code And Dependency Signals

Static checks found no broad unused module tree, but they did identify targeted cleanup candidates:

- `LiveRunSession` and `LiveRunManager` in `agentra/live_app.py` are stale-code candidates.
- Vulture reported unused `shlex` in `agentra/tools/terminal.py`.
- Vulture reported unused `MagicMock` imports in `tests/test_agent.py` and `tests/test_orchestrator.py`.
- Vulture reported unused variables `interval` and `waitTime` in `tests/test_tools.py`.
- `numpy`, `sentence-transformers`, and `chromadb` are declared dependencies but are not imported by package code in the current tree.

Do not delete dependencies solely from this list. First confirm whether they are intended future memory backends, optional transitive requirements, or stale packaging entries.

## Test And Tooling Gaps

The repository has broad unit coverage for runtime, live app, browser, tools, memory, reports, approval policy, and Gemini provider behavior. Known gaps:

- CLI behavior has little direct `click.testing` coverage.
- OpenAI, Anthropic, and Ollama providers have less direct fake-client coverage than Gemini.
- `tests/test_live_app_browser.py` launches real Playwright Chromium and uses timing-sensitive browser checks; it should have explicit marker documentation.
- No `.github`, `tox`, `nox`, `Makefile`, or pre-commit workflow is present.

Current verification signals from this audit:

- `ruff check .` failed with 59 issues, including import ordering, unused imports/variables, undefined forward-reference names in `agentra/cli.py`, and the Python 3.10 f-string syntax incompatibility in `agentra/tools/local_system.py`.
- `vulture agentra tests --min-confidence 80` found the small unused-code list above.
- `pytest --collect-only -q -s` collected 258 tests after adding the missing `aiohttp` dev dependency.

## Documentation Fixes Completed In This Pass

- README now describes Agentra as source-available under BUSL-1.1 instead of currently open source.
- Package metadata, CLI help text, package docstring, and generated workspace README template now use the source-available/checkpoint wording.
- `aiohttp` is now listed in the `dev` extra because provider/logging tests import it directly.
- README architecture now includes `ThreadManager`, `RunStore`, `approval_policy`, `task_routing`, `browser_runtime`, `desktop_automation`, `local_system`, and `windows_desktop`.
- Gemini docs now match the official `google-genai` provider path and `https://generativelanguage.googleapis.com` default base URL.
- Live HTTP docs include `/threads/clear` and `/runs/{run_id}/debug-images/{asset_path:path}`.
- Tool docs include browser `extract_links`, `list_feed_items`, and `click_feed_item_control`.
- Hidden desktop docs now state that Windows Graphics Capture is an unconfigured adapter boundary in this build, while `PrintWindow` is the implemented capture path.
- Artifact docs now include `debug-images/` and tracked generated-artifact caveats.

## Not Yet Fixed

These need code changes, not only docs:

- Add CLI approval enforcement or explicit non-interactive rejection behavior.
- Isolate orchestrator agents per parallel sub-task.
- Fix Python 3.10/3.11 syntax compatibility or update supported Python versions.
- Remove or migrate stale live runtime classes.
- Consolidate duplicated routing/normalization logic.
- Decide whether empty workspace checkpoint commits are intended.
- Add CLI and provider coverage.
- Resolve Ruff and Vulture findings.
