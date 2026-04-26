# Persisted Artifacts And Layout

This page documents what Agentra writes to disk and where contributors should look for runtime state.

Read [Architecture](architecture.md) for how these artifacts are produced, [Interfaces](interfaces.md) for the commands and routes that expose them, and [Audit & Gaps](audit.md) for generated artifacts that should not be treated as source.

## Two Common Layouts

Agentra writes slightly different layouts depending on whether you run the direct CLI path or the live thread-aware app.

### Direct CLI Workspace Layout

A direct CLI run uses the configured `workspace_dir` directly.

Typical layout:

```text
<workspace_dir>/
  .git/
  .logs/
    agentra-app.log
  .memory/
    index.json
    screenshots/
  .memory-global/
    index.json
    screenshots/
  .runs/
    <timestamp-slug>/
      events.json
      index.html
      assets/
      debug-images/
```

Notes:

- `.git/` exists when the workspace manager has initialized a repository.
- `.memory/` is the working memory store.
- `.memory-global/` is the long-term memory store.
- `.runs/` contains one folder per report-backed run.

### Live App Thread Layout

The live app creates a thread root under `<base workspace>/.threads/` and nests an isolated workspace inside each thread.

Typical layout:

```text
<base workspace>/
  .logs/
    agentra-app.log
  .memory-global/
    index.json
    screenshots/
  .threads/
    registry.json
    thread-<slug>/
      ledger.json
      audit.jsonl
      workspace/
        .git/
        .memory/
          index.json
          screenshots/
        .runs/
          <timestamp-slug>/
            events.json
            index.html
            assets/
            debug-images/
```

Notes:

- `registry.json` stores a compact list of known threads.
- each thread directory stores both the durable ledger snapshot and the append-only audit trail.
- the thread's own `workspace/` holds the per-thread `.memory/` and `.runs/` directories.
- long-term memory is shared at the base workspace level as `.memory-global/`.

## Run Report Artifacts

`RunStore` and `RunReport` create a run directory named with a timestamp plus a slugified goal prefix.

Each run directory contains:

- `events.json`
  - serialized run snapshot
  - event list
  - frame list
  - audit list
  - run metadata such as provider, model, status, and timestamps
- `index.html`
  - rendered report view for the run
- `assets/`
  - captured screenshot files such as `screenshot-001.png`
- `debug-images/`
  - archived debug/live frames such as `run-screenshots/` and `live-browser/`

`events.json` is the most complete on-disk record for a single run report.

## Thread Ledger Artifacts

`WorkspaceLedger` stores two files in each thread directory.

### `ledger.json`

This is the latest thread snapshot. It includes:

- thread identity and title
- thread status and handoff state
- permission mode and browser identity
- current run id and run summaries
- approval requests and question requests
- human actions
- browser snapshot payload
- embedded audit entries

### `audit.jsonl`

This is an append-only line-delimited JSON stream. Current entry types include items such as:

- `run_started`
- `run_finished`
- `approval_requested`
- `approval_resolved`
- `question_requested`
- `question_answered`
- `human_action`
- `workspace_checkpoint`
- `workspace_diff`
- `run_error`
- `thread_settings_updated`

## Memory Artifacts

`ThreadWorkingMemory` and `LongTermMemoryStore` persist to disk in the same basic shape:

- `index.json`
  - serialized memory records and embeddings
- `screenshots/`
  - screenshot files referenced by memory entries

Each memory record stores:

- `id`
- `text`
- `embedding`
- `timestamp`
- `role`
- `scope`
- optional `screenshot_path`
- `metadata`
- `retrieval_text`

## Logging Artifacts

`configure_app_logging()` writes a rotating file log at:

- `<workspace root>/.logs/agentra-app.log`

The log view served at `/logs` reads from this persistent file.

## Browser And Desktop Live Surfaces

Live browser and desktop frames are served dynamically and are not stored as a dedicated artifact stream by default.

Thread and run snapshots can also expose desktop session metadata such as mode, session status, active target window, capture backend, compatibility state, and fallback reason.

Persisted visual artifacts appear when:

- a run report stores screenshots in `.runs/<run-id>/assets/`
- the live app archives live/debug frames in `.runs/<run-id>/debug-images/`
- memory writes save screenshots into `.memory/screenshots/` or `.memory-global/screenshots/`

## Workspace Git State

`WorkspaceManager` treats the workspace as a git repository when `gitpython` is available.

Persistent effects include:

- repository initialization
- checkpoints that produce commit SHAs and diff summaries
- history visible through `agentra workspace history`
- restore support through `agentra workspace restore <sha>`

In the live runtime, checkpoint summaries are also mirrored into `workspace_checkpoint` and `workspace_diff` audit entries.

Current caveat: checkpoint code uses `git commit --allow-empty`, so a checkpoint commit can exist even when no files changed. Use `changed_files` and `diff_stats` in the run or ledger data before treating a checkpoint as proof of workspace modification.

## Tracked Generated Artifacts

This repository currently has some generated/runtime paths under version control, including `tmp-live-workspace`, `tmp-runtime-debug`, `tmp-smoke-under-the-hood`, `workspace`, `workspace-windows-demo`, `workspace-windows-demo-2`, and `@AutomationLog.txt`. Treat them as existing historical artifacts unless a task explicitly asks to refresh fixtures. New run output should not be added to normal documentation or code commits.

## Where To Inspect Artifacts During Debugging

- run event history: `.runs/<run-id>/events.json`
- rendered timeline: `.runs/<run-id>/index.html`
- archived debug images: `.runs/<run-id>/debug-images/`
- thread snapshot: `.threads/<thread-id>/ledger.json`
- thread audit trail: `.threads/<thread-id>/audit.jsonl`
- hidden desktop compatibility/debug state: thread snapshot desktop session payload
- working memory store: `workspace/.memory/index.json`
- long-term memory store: `.memory-global/index.json`
- application log: `.logs/agentra-app.log`
