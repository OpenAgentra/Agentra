# Runtime Policies

This page documents the current guardrails and routing rules that shape how Agentra executes work.

Use [Features](features.md) for the capability inventory, [Interfaces](interfaces.md) for the exact command, endpoint, and schema surfaces that these policies apply to, and [Audit & Gaps](audit.md) for known policy enforcement gaps.

## Permission Modes

Agentra currently supports two thread-level permission modes.

| Mode | Browser identity | Headless behavior | Approval posture |
| --- | --- | --- | --- |
| `default` | `isolated` | whatever the request/config sets | broader approval set for risky browser, terminal, filesystem, git, and desktop actions |
| `full` | `chrome_profile` | forced to non-headless | narrower approval set, but still blocks secrets, payment data, purchases, money movement, and destructive account actions |

Normalization rules from `AgentConfig`:

- setting `permission_mode="full"` forces `browser_identity="chrome_profile"`
- setting `permission_mode="full"` forces `browser_headless=False`
- setting `browser_identity="chrome_profile"` also implies `permission_mode="full"`

## Approval Policy Rules

The default rule engine lives in `agentra/approval_policy.py`.

### Default Mode Rules

In `default` mode, approval is required for these rule groups:

- `browser-auth-secret-entry`
  - secret, authentication, verification, or similar sensitive browser entry
- `browser-external-side-effect`
  - browser actions that may submit, publish, pay, delete, or create outside effects
- `terminal-install-or-side-effect`
  - commands such as installs, package manager operations, network fetches, git pushes, shutdowns, or destructive shell commands
- `filesystem-destructive`
  - filesystem `delete` and `move`
- `git-rewrite-or-clone`
  - git `reset`, `checkout`, and `clone`
- `computer-direct-control`
  - direct desktop-control actions that require explicit approval

### Full Mode Rules

In `full` mode, approvals are narrower but still required for:

- `sensitive-secret-entry`
- `transaction-or-account-destruction`
- `payment-information-entry`

This is why full mode can use a real browser profile but still pauses for high-risk actions.

Current enforcement boundary: approval requests are only actionable when `AutonomousAgent` is bound to a runtime controller, which happens in the live app path. Direct `agentra run ...` CLI execution currently creates an agent without that controller, so policy decisions do not pause the CLI run for approval.

## Goal Routing Policy

`choose_live_execution_policy()` in `agentra/task_routing.py` derives three runtime choices from the goal text:

- `browser_headless`
- `local_execution_mode`
- `desktop_fallback_policy`
- `desktop_execution_mode`

The current policy model is:

- browser-only goals default to browser-focused execution
- eligible local GUI goals default to `desktop_hidden`
- goals that explicitly request visible on-screen work still use `desktop_visible`
- local folder/document goals can switch to `under_the_hood` execution when the goal wording implies path-driven or background local work
- mixed web + local goals can still use `under_the_hood` local handling if the local portion looks like a file-resolution or document-open task

The policy also sets a `control_surface_hint` of either `browser` or `desktop` for the live UI. Goals requesting real browser context, a user's Chrome profile, or account-specific browser state can infer `full` browser identity in the live runtime unless the request explicitly chooses default mode.

## Local Execution Modes

### `visible`

Visible mode expects real on-screen interaction for the local part of the task.

Agent guidance in this mode:

- use `computer` for visible desktop actions
- use `browser` for explicit website steps
- use `filesystem` or `terminal` only as helpers for path discovery, not as a substitute for visibly opening the requested item

The usual desktop fallback is `visible_control`.

### `desktop_hidden`

Hidden desktop mode is the default GUI path for eligible local Windows app goals.

Agent guidance in this mode:

- prefer `windows_desktop` for structured Windows app tasks
- use `computer` against the hidden session preview only when raw interaction is needed
- do not silently switch to visible desktop control
- if the session is unsafe or incompatible, pause and ask for explicit visible/manual fallback

### `under_the_hood`

Under-the-hood mode is used for local tasks that should be completed without visible desktop automation by default.

Agent guidance in this mode:

- do not use `computer` automatically
- use `local_system resolve_known_folder` to resolve Desktop-like locations
- use `filesystem` to inspect the resolved WSL path
- use `local_system open_path` to open a confirmed local file or folder with the OS default handler
- only use `terminal` when `filesystem` or `local_system` cannot resolve the task directly

The desktop fallback in this mode is `pause_and_ask` rather than visible control.

Under-the-hood mode disables hidden desktop session routing for that local portion. It is intended for path/file/document operations that can be resolved through `local_system` and `filesystem`, not for GUI session automation.

## Path Handling Policy

Agentra intentionally separates WSL/Linux path space from Windows-native path space.

Current rules:

- `filesystem` resolves paths in WSL/Linux space
- `terminal` runs in the shell environment and therefore also expects WSL/Linux paths by default
- raw `C:\...` paths are blocked by guardrails for `filesystem` and discouraged for `terminal`
- if Windows-native shell behavior is required, the agent must call `powershell.exe` explicitly
- `local_system` bridges the gap by resolving Windows known folders and returning both Windows and WSL path forms

In practice, contributors should expect path examples such as `/mnt/c/Users/<user>/Desktop` rather than `C:\Users\<user>\Desktop` in terminal and filesystem flows.

## Tool Guardrail Policy

`AutonomousAgent` applies additional pre-execution guardrails beyond the approval engine.

Main guardrail categories:

- browser-only goals should stay in `browser` and not drift into desktop automation
- local desktop goals should not drift into unrelated browser navigation
- under-the-hood local goals should use `local_system` plus `filesystem` instead of `computer`
- raw Windows paths should not be passed to WSL-oriented terminal or filesystem flows
- repeated desktop click guessing is blocked after repeated near-identical or excessive attempts
- `local_system open_path` should only be called after the path is actually resolved

These rules are heavily exercised in `tests/test_agent.py`.

## Completion Policy

Agentra does not allow every `DONE:` response to end the run immediately. The agent checks whether the required evidence for the goal has been gathered.

Current completion rules include:

- mixed web + local goals require both the requested website step and the requested local step
- local folder-contents goals require a successful local listing before completion
- local document-open goals require path resolution before completion
- under-the-hood local goals require a successful `local_system open_path` or equivalent confirmed completion before `DONE:`
- visible desktop goals do not count as complete if the agent only listed files in a terminal without opening the requested item on screen

## Concurrency And Handoff Policy

Runtime coordination rules from `agentra/runtime.py`:

- only one thread can hold visible real-desktop control at a time
- multiple hidden desktop threads can run concurrently with isolated session locks
- browser and non-desktop tool calls do not use the visible desktop lock
- a thread can be paused into `paused_for_user`
- approval or question events move the thread into `blocked_waiting_user`
- responding to an approval or question returns the thread to `running` when appropriate
- manual human actions are recorded as run events and thread audit entries
- live thread selection can reuse an existing thread for similar browser titles instead of creating a new isolated thread every time

## Hidden Desktop Safety Policy

Background desktop execution has a hard safety guarantee:

- it must not steal focus from the real desktop
- it must not type into the user's current foreground app
- it must not silently downgrade from hidden mode to visible mode
- unknown popups, unsafe targets, and incompatible GPU/input surfaces pause and ask

## Memory Retrieval Policy

The agent writes observations to both working memory and long-term memory, but retrieval is selective.

Current retrieval rules:

- long-term memory lookup is skipped for desktop-local-only goals
- current-run memories are excluded from long-term retrieval prompts
- candidate memories must overlap with the goal, based on text, retrieval text, summary, URL, or title
- only a compact list of relevant prior snippets is inserted into the prompt

This keeps historical context available without letting stale browser or unrelated desktop state dominate the run.
