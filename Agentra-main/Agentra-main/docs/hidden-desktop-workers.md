# Hidden Desktop Workers

This note describes Agentra's same-machine hidden desktop worker architecture for background Windows app execution.

## Goals

The subsystem exists to let multiple threads run local GUI work without disturbing the user's real desktop.

The design targets:

- standard Windows desktop apps
- Electron, WPF, WinForms, Qt, and similar windowed apps
- windowed or editor-style DirectX-heavy apps such as Unreal Editor

The hard safety guarantee is:

- no focus stealing from the real desktop
- no typing into the user's current foreground app
- no silent downgrade from hidden mode to visible mode

## Worker Model

Each eligible thread gets a thread-scoped desktop session managed by `DesktopSessionManager`.

Session modes:

- `desktop_visible`
- `desktop_native`
- `desktop_hidden`

In `desktop_hidden`:

- apps launch on an isolated Win32 desktop object
- the live app still uses the existing desktop preview/frame routes
- preview interaction is routed into the worker session instead of the visible desktop

## Backend Split

Desktop automation is split into these responsibilities:

- session lifecycle
- frame capture
- raw input routing
- structured UI automation
- compatibility probing

This split keeps `computer` and `windows_desktop` stable at the tool surface while allowing the execution backend to vary by thread.

## Capture Adapters

Hidden sessions choose a capture adapter chain per target window/process.

Compatibility tiers:

- `background_capable`: capture and input both supported
- `preview_only`: frames are available, but safe background input is not
- `fallback_required`: safe background execution is not supported

Adapter order:

1. GPU/window capture path for windowed DirectX-heavy surfaces when available
2. standard window capture path for normal desktop apps
3. unsupported/fallback state when neither is safe enough

Exclusive-fullscreen, raw-input-only, or similar incompatible surfaces must report `fallback_required` and pause for explicit visible/manual takeover.

## Input Adapters

Hidden session input must avoid the real desktop as the control surface.

Rules:

- do not use the system clipboard as the primary control path
- do not use global visible-desktop `SendInput` as the primary hidden-session path
- prefer structured control APIs, value patterns, and window-targeted messaging
- raw preview clicks, drags, wheel events, and keys are routed through a session input adapter

## Live Operator UX

The hidden worker architecture keeps the existing UI model:

- `/threads/{thread_id}/desktop-frame`
- `/threads/{thread_id}/desktop-stream`
- `Interact`
- pause/resume / `Finish Control`

The difference is that the desktop preview now represents the thread's desktop session, not necessarily the real desktop.

Thread and run snapshots expose desktop session metadata:

- mode
- session status
- active target app/window
- capture backend
- compatibility state
- fallback reason
