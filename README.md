# Agentra

> **Source-available autonomous AI agent with full computer access** — browse the web, control the desktop, manage files, run terminal commands, and keep auditable run artifacts, all orchestrated by an LLM you choose.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: BUSL-1.1](https://img.shields.io/badge/License-BUSL--1.1-blue.svg)](LICENSE)

---
![Animationlast-ezgif com-speed](https://github.com/user-attachments/assets/c0ae3618-2fc2-4f4c-ae33-4c351994ac2f)
## What is Agentra?

Agentra is a source-available computer-use runtime. It is built for contributor inspection, local operation, and auditable automation across browser, desktop, filesystem, terminal, memory, and live operator surfaces.

| Feature | ChatGPT Operator | Agentra |
|---|---|---|
| Web browsing | ✅ | ✅ |
| Desktop control | ❌ | ✅ |
| Hidden desktop workers | ❌ | ✅ |
| Filesystem access | ❌ | ✅ |
| Terminal / shell | ❌ | ✅ |
| Multi-agent orchestration | ❌ | ✅ |
| Your own LLM (free) | ❌ | ✅ (Ollama) |
| Source available | ❌ | ✅ |
| Git-tracked workspace | ❌ | ✅ |
| Persistent visual memory | ❌ | ✅ |

You give Agentra a natural-language goal — _"Apply to 10 Python jobs on LinkedIn"_, _"Summarise all PDFs in my Downloads folder"_, _"Set up a new Django project"_ — and it works through it step-by-step, using whatever tools it needs. You can watch actions in real time, keep long-running browser or desktop work inside live preview surfaces, and take back control at runtime approval, question, pause, or manual-action boundaries.

---

## Architecture

```
agentra/
├── agents/
│   ├── autonomous.py     ← ReAct loop (Reason → Act → Observe → repeat)
│   └── orchestrator.py   ← Multi-agent task decomposition & coordination
├── runtime.py            ← Thread manager, approvals, questions, runs, ledgers
├── live_app.py           ← FastAPI operator UI and live browser/desktop routes
├── task_routing.py       ← Goal-to-browser/local/desktop execution policy
├── approval_policy.py    ← Approval rules and storage redaction
├── browser_runtime.py    ← Shared Playwright runtime and thread sessions
├── desktop_automation/   ← Visible/native/hidden desktop backends
├── tools/
│   ├── browser.py        ← Playwright web browser
│   ├── computer.py       ← Mouse / keyboard / screenshot
│   ├── filesystem.py     ← Read / write / list / copy / move / delete
│   ├── local_system.py   ← Resolve/open local OS paths without GUI automation
│   ├── windows_desktop.py← Structured Windows UI automation
│   ├── terminal.py       ← Shell command execution
│   └── git_tool.py       ← Git operations on the workspace
├── llm/
│   ├── openai_provider.py
│   ├── anthropic_provider.py
│   ├── gemini_provider.py
│   └── ollama_provider.py
├── memory/
│   ├── embedding_memory.py   ← Working and long-term embedding stores
│   └── workspace.py          ← Git-tracked workspace manager
├── run_store.py          ← Structured events, frames, assets, debug images
├── run_report.py         ← HTML timeline renderer
└── cli.py                    ← Rich terminal interface
```

---

## Documentation

Contributor-first project documentation lives in `docs/`:

- [Docs Hub](docs/README.md)
- [Architecture](docs/architecture.md)
- [Hidden Desktop Workers](docs/hidden-desktop-workers.md)
- [Features](docs/features.md)
- [Policies](docs/policies.md)
- [Interfaces](docs/interfaces.md)
- [Artifacts](docs/artifacts.md)
- [Audit & Gaps](docs/audit.md)

## Quick Start

### 1 — Install

```bash
pip install agentra
# or from source:
git clone https://github.com/OpenAgentra/Agentra.git
cd Agentra
pip install -e .
```

For contributor work:

```bash
pip install -e ".[dev]"
python -m playwright install chromium
```

### 2 — Configure

```bash
agentra config init
```

Or create a `.env` file manually:

```env
AGENTRA_LLM_PROVIDER=openai          # openai | anthropic | ollama | gemini
AGENTRA_LLM_MODEL=gpt-4o
AGENTRA_OPENAI_API_KEY=sk-...
```

For a quick Gemini setup using Google's official GenAI SDK:

```env
AGENTRA_LLM_PROVIDER=gemini
AGENTRA_LLM_MODEL=gemini-3-flash-preview
AGENTRA_GEMINI_API_KEY=your-api-key
```

For a **completely free, local** setup using Ollama:

```env
AGENTRA_LLM_PROVIDER=ollama
AGENTRA_LLM_MODEL=llava              # vision-capable model recommended
AGENTRA_OLLAMA_BASE_URL=http://localhost:11434
```

> Make sure [Ollama](https://ollama.ai) is running: `ollama serve` and `ollama pull llava`

### 3 — Run

```bash
# Single agent
agentra run "Summarise all .txt files in ~/Documents and save a report"

# With multi-agent orchestration
agentra run --orchestrate "Research the top 5 Python web frameworks and write a comparison"

# Control the browser visually
agentra run --no-headless "Search LinkedIn for Python jobs in Istanbul and list them"

# Save a visual HTML timeline for a demo
agentra run --open-report "Open python.org, take a screenshot, and describe the page"

# Use a specific model
agentra run --provider ollama --model llava "Take a screenshot and describe what you see"
```

---

## Features

### 🌐 Browser Automation
Navigate websites, click elements, fill forms, press keys, drag, scroll, extract text/HTML/links, inspect visible feed cards, click marked feed controls, manage tabs, and take screenshots.

```python
from agentra import AutonomousAgent, AgentConfig

config = AgentConfig(llm_provider="openai", browser_headless=False)
agent = AutonomousAgent(config)

async for event in await agent.run("Go to python.org and list the latest news"):
    print(event)
```

### 🖥️ Desktop Control
Full mouse and keyboard control of any GUI application — not just the browser.

```python
# The agent can take screenshots, click, type, scroll, drag on the desktop
config = AgentConfig(allow_computer_control=True)
```

### 🪟 Background Desktop Workers
Eligible Windows app tasks can run inside same-machine hidden desktop workers instead of the visible user desktop. Agentra keeps the existing live preview and `Interact` flow, but routes preview input to the worker session so background runs do not steal focus from the real desktop. See [Hidden Desktop Workers](docs/hidden-desktop-workers.md) for current backend limits.

### 📁 Filesystem Access
Read, write, copy, move and delete files anywhere on disk.

### 💻 Terminal Execution
Run shell commands, install packages, start services — with a configurable timeout and output capture.

### 🧠 Visual Memory
Screenshots and observations can be stored with embeddings and surfaced as context for future steps. Thread working memory and project-wide long-term memory are separate stores.

```python
# Retrieve semantically similar past observations
results = await agent.memory.search("login form error", top_k=5)
```

### 🔀 Multi-Agent Orchestration
Break complex goals into parallel sub-tasks, each handled by a specialist agent:

```python
from agentra import Orchestrator

orch = Orchestrator(config=config)
result = await orch.run("Apply to 10 Python jobs on LinkedIn with my CV")
print(result.final_summary)
```

### 📦 Git-Tracked Workspace
Agentra initializes and checkpoints workspaces when GitPython is available. Live runs mirror checkpoint summaries into thread audit entries; direct workspace history and restore commands operate on the configured workspace repository.

```bash
agentra workspace history
agentra workspace restore abc1234
```

---

## Configuration Reference

All settings can be set via environment variables (prefix `AGENTRA_`) or a `.env` file.

| Variable | Default | Description |
|---|---|---|
| `AGENTRA_LLM_PROVIDER` | `openai` | `openai` \| `anthropic` \| `ollama` \| `gemini` |
| `AGENTRA_LLM_MODEL` | `gpt-4o` | Model name for the chosen provider |
| `AGENTRA_LLM_VISION_MODEL` | _(same as model)_ | Separate vision model if needed |
| `AGENTRA_EXECUTOR_MODEL` | — | Role-specific executor model override |
| `AGENTRA_PLANNER_MODEL` | — | Role-specific orchestrator planning model override |
| `AGENTRA_SUMMARY_MODEL` | — | Role-specific summary model override |
| `AGENTRA_EMBEDDING_MODEL` | — | Embedding model override when supported |
| `AGENTRA_OPENAI_API_KEY` | — | OpenAI API key |
| `AGENTRA_ANTHROPIC_API_KEY` | — | Anthropic API key |
| `AGENTRA_OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `AGENTRA_GEMINI_API_KEY` | — | Google Gemini API key |
| `AGENTRA_GEMINI_BASE_URL` | `https://generativelanguage.googleapis.com` | Gemini API base URL used by `google-genai` |
| `AGENTRA_MAX_ITERATIONS` | `50` | Maximum agent loop iterations |
| `AGENTRA_MAX_TOKENS` | `4096` | Max tokens per LLM call |
| `AGENTRA_TEMPERATURE` | `0.2` | LLM sampling temperature |
| `AGENTRA_WORKSPACE_DIR` | `./workspace` | Agent's working directory |
| `AGENTRA_MEMORY_DIR` | `./workspace/.memory` | Where embeddings & screenshots are stored |
| `AGENTRA_LONG_TERM_MEMORY_DIR` | `./workspace/.memory-global` | Shared long-term memory store |
| `AGENTRA_SCREENSHOT_HISTORY` | `10` | Screenshots kept in context |
| `AGENTRA_BROWSER_HEADLESS` | `false` | Run browser without UI |
| `AGENTRA_BROWSER_TYPE` | `chromium` | `chromium` \| `firefox` \| `webkit` |
| `AGENTRA_BROWSER_IDENTITY` | `isolated` | `isolated` \| `chrome_profile` |
| `AGENTRA_BROWSER_PROFILE_NAME` | `Default` | Chrome profile directory name |
| `AGENTRA_LOCAL_EXECUTION_MODE` | `visible` | `visible` \| `under_the_hood` \| `native` |
| `AGENTRA_DESKTOP_FALLBACK_POLICY` | `visible_control` | `visible_control` \| `pause_and_ask` |
| `AGENTRA_DESKTOP_EXECUTION_MODE` | `desktop_visible` | `desktop_visible` \| `desktop_native` \| `desktop_hidden` |
| `AGENTRA_DESKTOP_BACKEND_PREFERENCE` | `visible` | `visible` \| `native` \| `under_the_hood` |
| `AGENTRA_PERMISSION_MODE` | `default` | `default` \| `full` |
| `AGENTRA_ALLOW_TERMINAL` | `true` | Enable/disable terminal access |
| `AGENTRA_ALLOW_FILESYSTEM_WRITE` | `true` | Enable/disable file writes |
| `AGENTRA_ALLOW_COMPUTER_CONTROL` | `true` | Enable/disable mouse/keyboard |

The full field reference, provider registry, CLI commands, HTTP routes, and tool schemas live in [docs/interfaces.md](docs/interfaces.md).

---

## Python API

```python
import asyncio
from agentra import AutonomousAgent, AgentConfig

async def main():
    config = AgentConfig(
        llm_provider="openai",
        llm_model="gpt-4o",
        max_iterations=30,
    )
    agent = AutonomousAgent(config)

    async for event in await agent.run("Create a Python hello-world script in my workspace"):
        if event["type"] == "thought":
            print(f"💭 {event['content']}")
        elif event["type"] == "tool_call":
            print(f"🔧 {event['tool']}({event['args']})")
        elif event["type"] == "tool_result":
            status = "✓" if event["success"] else "✗"
            print(f"   {status} {event['result'][:200]}")
        elif event["type"] == "done":
            print(f"✅ {event['content']}")

asyncio.run(main())
```

### Taking back control

```python
# Interrupt the agent from another coroutine
agent.interrupt()

# Or wait for it to fully stop
await agent.take_control()
```

---

## Running Tests

```bash
pip install -e ".[dev]"
python -m playwright install chromium
pytest tests -v
ruff check .
```

Focused checks:

```bash
pytest tests/test_runtime.py tests/test_live_app.py -q
pytest tests/test_live_app_browser.py -q
vulture agentra tests --min-confidence 80
```

The current repository audit, including known lint failures, Python-version issues, stale-code candidates, generated artifacts, and test coverage gaps, is tracked in [docs/audit.md](docs/audit.md).

---

## License

Agentra is source-available under the **[Business Source License 1.1 (BUSL-1.1)](LICENSE)**.

**What this means:**

| Use case | Allowed? |
|---|---|
| Reading / studying the source code | ✅ |
| Personal, non-commercial use | ✅ |
| Research & evaluation | ✅ |
| Contributing back to this repo | ✅ |
| Running in production for your own non-commercial projects | ✅ |
| Offering Agentra as a commercial SaaS / managed service | ❌ (requires a commercial license) |
| Embedding in a commercial product for redistribution | ❌ (requires a commercial license) |

**Open-source transition:** On **2030-03-16** (or four years after any given version's first public release, whichever is earlier) the license automatically converts to the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0) for that version.

For commercial licensing inquiries, please open an issue or contact the maintainers.
