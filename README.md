# Agentra

> **Open-source autonomous AI agent with full computer access** — browse the web, control the desktop, manage files, run terminal commands — all orchestrated by an LLM you choose.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## What is Agentra?

Agentra is the open-source alternative to ChatGPT's "computer use" feature — and it goes much further:

| Feature | ChatGPT Operator | Agentra |
|---|---|---|
| Web browsing | ✅ | ✅ |
| Desktop control | ❌ | ✅ |
| Filesystem access | ❌ | ✅ |
| Terminal / shell | ❌ | ✅ |
| Multi-agent orchestration | ❌ | ✅ |
| Your own LLM (free) | ❌ | ✅ (Ollama) |
| Open-source | ❌ | ✅ |
| Git-tracked workspace | ❌ | ✅ |
| Persistent visual memory | ❌ | ✅ |

You give Agentra a natural-language goal — _"Apply to 10 Python jobs on LinkedIn"_, _"Summarise all PDFs in my Downloads folder"_, _"Set up a new Django project"_ — and it works through it step-by-step, using whatever tools it needs.  You can watch every action in real time and take back control at any moment.

---

## Architecture

```
agentra/
├── agents/
│   ├── autonomous.py     ← ReAct loop (Reason → Act → Observe → repeat)
│   └── orchestrator.py   ← Multi-agent task decomposition & coordination
├── tools/
│   ├── browser.py        ← Playwright web browser
│   ├── computer.py       ← Mouse / keyboard / screenshot
│   ├── filesystem.py     ← Read / write / list / copy / move / delete
│   ├── terminal.py       ← Shell command execution
│   └── git_tool.py       ← Git operations on the workspace
├── llm/
│   ├── openai_provider.py    ← GPT-4o, GPT-4-vision …
│   ├── anthropic_provider.py ← Claude 3.5 Sonnet (computer-use model)
│   └── ollama_provider.py    ← Any local model (LLaVA, LLaMA 3, Mistral …)
├── memory/
│   ├── embedding_memory.py   ← Screenshot + text embedding store (no DB needed)
│   └── workspace.py          ← Git-tracked workspace manager
└── cli.py                    ← Rich terminal interface
```

---

## Quick Start

### 1 — Install

```bash
pip install agentra
# or from source:
git clone https://github.com/Kagankakao/Agentra
cd Agentra
pip install -e .
```

### 2 — Configure

```bash
agentra config init
```

Or create a `.env` file manually:

```env
AGENTRA_LLM_PROVIDER=openai          # openai | anthropic | ollama
AGENTRA_LLM_MODEL=gpt-4o
AGENTRA_OPENAI_API_KEY=sk-...
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

# Use a specific model
agentra run --provider ollama --model llava "Take a screenshot and describe what you see"
```

---

## Features

### 🌐 Browser Automation
Navigate websites, click elements, fill forms, extract text and HTML, take screenshots — everything Playwright supports.

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

### 📁 Filesystem Access
Read, write, copy, move and delete files anywhere on disk.

### 💻 Terminal Execution
Run shell commands, install packages, start services — with a configurable timeout and output capture.

### 🧠 Visual Memory
Screenshots are automatically stored with embeddings and surfaced as context for future steps.  The agent "remembers" what it has seen.

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
Every task is automatically committed to a git repository inside the workspace directory, giving you a full audit trail and the ability to roll back.

```bash
agentra workspace history
agentra workspace restore abc1234
```

---

## Configuration Reference

All settings can be set via environment variables (prefix `AGENTRA_`) or a `.env` file.

| Variable | Default | Description |
|---|---|---|
| `AGENTRA_LLM_PROVIDER` | `openai` | `openai` \| `anthropic` \| `ollama` |
| `AGENTRA_LLM_MODEL` | `gpt-4o` | Model name for the chosen provider |
| `AGENTRA_LLM_VISION_MODEL` | _(same as model)_ | Separate vision model if needed |
| `AGENTRA_OPENAI_API_KEY` | — | OpenAI API key |
| `AGENTRA_ANTHROPIC_API_KEY` | — | Anthropic API key |
| `AGENTRA_OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `AGENTRA_MAX_ITERATIONS` | `50` | Maximum agent loop iterations |
| `AGENTRA_MAX_TOKENS` | `4096` | Max tokens per LLM call |
| `AGENTRA_TEMPERATURE` | `0.2` | LLM sampling temperature |
| `AGENTRA_WORKSPACE_DIR` | `./workspace` | Agent's working directory |
| `AGENTRA_MEMORY_DIR` | `./workspace/.memory` | Where embeddings & screenshots are stored |
| `AGENTRA_SCREENSHOT_HISTORY` | `10` | Screenshots kept in context |
| `AGENTRA_BROWSER_HEADLESS` | `false` | Run browser without UI |
| `AGENTRA_BROWSER_TYPE` | `chromium` | `chromium` \| `firefox` \| `webkit` |
| `AGENTRA_ALLOW_TERMINAL` | `true` | Enable/disable terminal access |
| `AGENTRA_ALLOW_FILESYSTEM_WRITE` | `true` | Enable/disable file writes |
| `AGENTRA_ALLOW_COMPUTER_CONTROL` | `true` | Enable/disable mouse/keyboard |

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
pytest tests/ -v
```

---

## License

MIT — see [LICENSE](LICENSE).