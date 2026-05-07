"""Native Windows desktop backend built on UI Automation."""

from __future__ import annotations

import contextlib
import os
import re
import subprocess
import time
from typing import Any

from agentra.desktop_automation.base import DesktopAutomationBackend
from agentra.desktop_automation.models import ControlInfo, DesktopActionVerification, WindowInfo
from agentra.windows_apps import (
    WINDOWS_APP_PROFILES,
    get_windows_app_profile,
    guess_windows_app_profile,
    normalize_windows_app_command,
)


def _normalized_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").casefold()).strip()


class NativeWindowsDesktopBackend(DesktopAutomationBackend):
    """Structured Windows desktop automation for standard native apps."""

    backend_id = "native_windows"
    desktop_execution_mode = "desktop_native"
    supports_structured_controls = True

    def __init__(self, *, default_timeout: float = 6.0) -> None:
        self._default_timeout = default_timeout
        self._preferred_window_handles: dict[str, int] = {}
        self._initialized_calculator_windows: set[int] = set()

    def is_available(self) -> tuple[bool, str | None]:
        if os.name != "nt":
            return (False, "windows_native backend is only available on Windows.")
        try:
            self._uia()
        except ImportError:
            return (
                False,
                "Windows UI Automation dependency is not installed. Install agentra[windows-desktop].",
            )
        except Exception as exc:  # noqa: BLE001
            return (False, str(exc))
        try:
            import win32gui  # noqa: PLC0415,F401
        except ImportError:
            return (
                False,
                "Windows native desktop support also requires pywin32. Install agentra[windows-desktop].",
            )
        return (True, None)

    def execute(self, action: str, **kwargs: Any) -> DesktopActionVerification:
        available, reason = self.is_available()
        if not available:
            return DesktopActionVerification(
                target=str(kwargs.get("app") or kwargs.get("window_title") or "desktop"),
                action=action,
                observed_outcome=reason or "Backend unavailable.",
                success=False,
                verified=False,
                fallback_reason=reason or "",
            )

        handlers = {
            "launch_app": self.launch_app,
            "focus_window": self.focus_window,
            "wait_for_window": self.wait_for_window,
            "list_windows": self.list_windows,
            "list_controls": self.list_controls,
            "invoke_control": self.invoke_control,
            "set_text": self.set_text,
            "type_keys": self.type_keys,
            "read_window_text": self.read_window_text,
            "read_status": self.read_status,
        }
        handler = handlers.get(action)
        if handler is None:
            return DesktopActionVerification(
                target="desktop",
                action=action,
                observed_outcome=f"Unsupported windows_desktop action: {action}",
                success=False,
                verified=False,
                fallback_reason=f"Unsupported action: {action}",
            )
        return handler(**kwargs)

    def launch_app(
        self,
        *,
        app: str | None = None,
        profile_id: str | None = None,
        timeout_sec: float | None = None,
        **_: Any,
    ) -> DesktopActionVerification:
        requested_app = str(app or "").strip()
        if not requested_app and profile_id:
            profile = get_windows_app_profile(profile_id)
            if profile is not None:
                requested_app = profile.executable
        if not requested_app:
            return self._failure("launch_app", "desktop", "app is required")
        command = normalize_windows_app_command(requested_app)
        try:
            subprocess.Popen([command])  # noqa: S603
        except Exception as exc:  # noqa: BLE001
            return self._failure("launch_app", command, str(exc))

        inferred_profile = profile_id or guess_windows_app_profile(requested_app)
        if inferred_profile:
            wait_result = self.wait_for_window(
                app=requested_app,
                profile_id=inferred_profile,
                timeout_sec=timeout_sec,
            )
            details = dict(wait_result.details)
            details["requested_app"] = requested_app
            details["app"] = command
            return DesktopActionVerification(
                target=command,
                action="launch_app",
                observed_outcome=wait_result.observed_outcome or f"Launched {command}.",
                success=wait_result.success,
                verified=wait_result.verified,
                fallback_reason=wait_result.fallback_reason,
                details=details,
            )

        return DesktopActionVerification(
            target=command,
            action="launch_app",
            observed_outcome=f"Launched {command}.",
            success=True,
            verified=True,
            details={"requested_app": requested_app, "app": command},
        )

    def focus_window(
        self,
        *,
        window_title: str | None = None,
        app: str | None = None,
        profile_id: str | None = None,
        **_: Any,
    ) -> DesktopActionVerification:
        window = self._resolve_window(window_title=window_title, app=app, profile_id=profile_id)
        if window is None:
            return self._failure("focus_window", window_title or app or "window", "Window not found.")
        self._remember_window(window, profile_id=profile_id, app=app, window_title=window_title)
        try:
            import win32con  # noqa: PLC0415
            import win32gui  # noqa: PLC0415

            win32gui.ShowWindow(window.handle, win32con.SW_RESTORE)
            win32gui.SetForegroundWindow(window.handle)
        except Exception as exc:  # noqa: BLE001
            return self._failure("focus_window", window.title or str(window.handle), str(exc))
        return DesktopActionVerification(
            target=window.title or str(window.handle),
            action="focus_window",
            observed_outcome=f"Focused {window.title or 'the target window'}.",
            success=True,
            verified=True,
            details={"window_title": window.title, "window_handle": window.handle},
        )

    def wait_for_window(
        self,
        *,
        window_title: str | None = None,
        app: str | None = None,
        profile_id: str | None = None,
        timeout_sec: float | None = None,
        **_: Any,
    ) -> DesktopActionVerification:
        timeout = float(timeout_sec or self._default_timeout)
        deadline = time.monotonic() + timeout
        window: WindowInfo | None = None
        profile = get_windows_app_profile(profile_id or guess_windows_app_profile(app or window_title or ""))
        while time.monotonic() < deadline:
            window = self._resolve_window(window_title=window_title, app=app, profile_id=profile_id)
            if window is not None:
                if profile is not None and not self._window_looks_actionable(window, profile):
                    time.sleep(0.2)
                    continue
                self._remember_window(window, profile_id=profile_id, app=app, window_title=window_title)
                break
            time.sleep(0.2)
        if window is None:
            return self._failure(
                "wait_for_window",
                window_title or app or profile_id or "window",
                "Window did not appear in time.",
            )
        return DesktopActionVerification(
            target=window.title or str(window.handle),
            action="wait_for_window",
            observed_outcome=f"Window ready: {window.title}.",
            success=True,
            verified=True,
            details={
                "window_title": window.title,
                "window_handle": window.handle,
                "window_class_name": window.class_name,
            },
        )

    def list_windows(self, **_: Any) -> DesktopActionVerification:
        windows = self._enumerate_windows()
        outcome = "\n".join(
            f"- {item.title} (handle={item.handle}, class={item.class_name})"
            for item in windows
            if item.title
        )
        return DesktopActionVerification(
            target="windows",
            action="list_windows",
            observed_outcome=outcome or "No visible top-level windows found.",
            success=True,
            verified=True,
            details={
                "windows": [
                    {
                        "title": item.title,
                        "handle": item.handle,
                        "class_name": item.class_name,
                        "process_id": item.process_id,
                    }
                    for item in windows
                ]
            },
        )

    def list_controls(
        self,
        *,
        window_title: str | None = None,
        app: str | None = None,
        profile_id: str | None = None,
        max_depth: int = 5,
        **_: Any,
    ) -> DesktopActionVerification:
        window = self._resolve_window(window_title=window_title, app=app, profile_id=profile_id)
        if window is None:
            return self._failure("list_controls", window_title or app or "window", "Window not found.")
        root = self._control_root(window)
        if root is None:
            return self._failure("list_controls", window.title, "UI Automation root not found.")
        controls = self._collect_controls(root, max_depth=max_depth)
        observed = "\n".join(
            f"- depth={item.depth} {item.control_type} name={item.name!r} automation_id={item.automation_id!r}"
            for item in controls
        )
        return DesktopActionVerification(
            target=window.title,
            action="list_controls",
            observed_outcome=observed or "No controls found.",
            success=True,
            verified=True,
            details={
                "window_title": window.title,
                "window_handle": window.handle,
                "controls": [item.__dict__ for item in controls],
            },
        )

    def invoke_control(
        self,
        *,
        window_title: str | None = None,
        app: str | None = None,
        profile_id: str | None = None,
        control_name: str | None = None,
        automation_id: str | None = None,
        control_type: str | None = None,
        max_depth: int = 8,
        **_: Any,
    ) -> DesktopActionVerification:
        window = self._resolve_window(window_title=window_title, app=app, profile_id=profile_id)
        if window is None:
            return self._failure("invoke_control", window_title or app or "window", "Window not found.")
        control = self._resolve_control(
            window,
            control_name=control_name,
            automation_id=automation_id,
            control_type=control_type,
            max_depth=max_depth,
        )
        if control is None:
            return self._failure("invoke_control", window.title, "Requested control not found.")
        target = control_name or automation_id or window.title
        invoked = self._invoke_control_object(control, target=target, window=window)
        if invoked.success and not invoked.observed_outcome:
            return DesktopActionVerification(
                target=target,
                action="invoke_control",
                observed_outcome=f"Invoked {control.Name or control.AutomationId or control.ControlTypeName}.",
                success=True,
                verified=True,
                details={"window_title": window.title, "window_handle": window.handle},
            )
        return invoked

    def set_text(
        self,
        *,
        text: str,
        window_title: str | None = None,
        app: str | None = None,
        profile_id: str | None = None,
        control_name: str | None = None,
        automation_id: str | None = None,
        max_depth: int = 8,
        **_: Any,
    ) -> DesktopActionVerification:
        if not str(text or ""):
            return self._failure("set_text", window_title or app or "window", "text is required")
        window = self._resolve_window(window_title=window_title, app=app, profile_id=profile_id)
        if window is None:
            return self._failure("set_text", window_title or app or "window", "Window not found.")
        control = self._resolve_edit_control(
            window,
            control_name=control_name,
            automation_id=automation_id,
            max_depth=max_depth,
        )
        if control is None:
            return self._failure("set_text", window.title, "Editable control not found.")
        before_value = self._control_text(control)
        try:
            pattern = getattr(control, "GetValuePattern", lambda: None)()
            if pattern is not None:
                pattern.SetValue(text)
            else:
                control.Click(simulateMove=False)
                control.SendKeys("{Ctrl}a{Delete}", interval=0.01)
                control.SendKeys(text, interval=0.01)
        except Exception as exc:  # noqa: BLE001
            return self._failure("set_text", window.title, str(exc))
        after_value = self._control_text(control)
        verified = bool(after_value) and text in after_value
        return DesktopActionVerification(
            target=window.title,
            action="set_text",
            observed_outcome=f"Text updated in {window.title}.",
            success=True,
            verified=verified,
            fallback_reason="" if verified else "Text change could not be verified from the control value.",
            details={
                "window_title": window.title,
                "window_handle": window.handle,
                "pre_state": before_value,
                "post_state": after_value,
            },
        )

    def type_keys(
        self,
        *,
        text: str,
        window_title: str | None = None,
        app: str | None = None,
        profile_id: str | None = None,
        **_: Any,
    ) -> DesktopActionVerification:
        if not str(text or ""):
            return self._failure("type_keys", window_title or app or "window", "text is required")
        resolved_profile = profile_id or guess_windows_app_profile(app or window_title or "")
        window = self._resolve_window(window_title=window_title, app=app, profile_id=resolved_profile)
        if window is None:
            return self._failure("type_keys", window_title or app or "window", "Window not found.")
        if resolved_profile == "calculator":
            before_text = self.read_window_text(
                window_title=window.title,
                profile_id=resolved_profile,
            ).observed_outcome
            sent = self._type_into_calculator(window, text=text)
            if not sent.success:
                return sent
            time.sleep(0.15)
            after = self.read_window_text(
                window_title=window.title,
                profile_id=resolved_profile,
            )
            observed_status = self._calculator_status(window)
            verified = after.success and after.observed_outcome != before_text
            return DesktopActionVerification(
                target=window.title,
                action="type_keys",
                observed_outcome=f"Typed keys into {window.title}.",
                success=True,
                verified=verified,
                fallback_reason="" if verified else "Calculator UI did not visibly change after native button input.",
                details={
                    "pre_state": before_text,
                    "post_state": after.observed_outcome,
                    "window_title": window.title,
                    "window_handle": window.handle,
                    "observed_status": observed_status,
                },
            )
        root = self._control_root(window)
        focus_warning = ""
        focus = self.focus_window(window_title=window.title)
        if not focus.success:
            focus_warning = focus.fallback_reason or focus.observed_outcome
        before_text = self.read_window_text(
            window_title=window.title,
            profile_id=profile_id,
        ).observed_outcome
        translated_text = self._translate_send_keys_text(text)
        try:
            if root is not None:
                root.SendKeys(translated_text, interval=0.01, waitTime=0.1)
            else:
                self._uia().SendKeys(translated_text, interval=0.01, waitTime=0.1)
        except Exception as exc:  # noqa: BLE001
            return self._failure("type_keys", window.title, str(exc))
        time.sleep(0.15)
        after = self.read_window_text(
            window_title=window.title,
            profile_id=profile_id,
        )
        verified = after.success and after.observed_outcome != before_text
        return DesktopActionVerification(
            target=window.title,
            action="type_keys",
            observed_outcome=f"Typed keys into {window.title}.",
            success=True,
            verified=verified,
            fallback_reason=(
                focus_warning
                or ("" if verified else "Keystroke effect could not be verified from the window text.")
            ),
            details={
                "pre_state": before_text,
                "post_state": after.observed_outcome,
                "requested_text": text,
                "translated_text": translated_text,
                "window_title": window.title,
                "window_handle": window.handle,
            },
        )

    def read_window_text(
        self,
        *,
        window_title: str | None = None,
        app: str | None = None,
        profile_id: str | None = None,
        max_depth: int = 8,
        **_: Any,
    ) -> DesktopActionVerification:
        window = self._resolve_window(window_title=window_title, app=app, profile_id=profile_id)
        if window is None:
            return self._failure("read_window_text", window_title or app or "window", "Window not found.")
        root = self._control_root(window)
        if root is None:
            return self._failure("read_window_text", window.title, "UI Automation root not found.")
        text = self._collect_window_text(root, max_depth=max_depth)
        return DesktopActionVerification(
            target=window.title,
            action="read_window_text",
            observed_outcome=text,
            success=True,
            verified=bool(text),
            fallback_reason="" if text else "Window text was empty.",
            details={"window_title": window.title, "window_handle": window.handle},
        )

    def read_status(
        self,
        *,
        window_title: str | None = None,
        app: str | None = None,
        profile_id: str | None = None,
        expected_text: str | None = None,
        **_: Any,
    ) -> DesktopActionVerification:
        resolved_profile = profile_id or guess_windows_app_profile(app or window_title or "")
        window = self._resolve_window(window_title=window_title, app=app, profile_id=resolved_profile)
        if window is None:
            return self._failure("read_status", window_title or app or "window", "Window not found.")
        if resolved_profile == "calculator":
            observed = self._calculator_status(window)
        else:
            observed = self.read_window_text(
                window_title=window.title,
                profile_id=resolved_profile,
            ).observed_outcome
        expected = str(expected_text or "").strip()
        verified = bool(observed)
        fallback_reason = ""
        if expected:
            verified = _normalized_text(expected) in _normalized_text(observed)
            if not verified:
                fallback_reason = f"Expected {expected!r} was not observed."
        return DesktopActionVerification(
            target=window.title,
            action="read_status",
            observed_outcome=observed,
            success=bool(observed) if not expected else verified,
            verified=verified,
            fallback_reason=fallback_reason,
            details={"window_title": window.title, "window_handle": window.handle},
        )

    def _resolve_window(
        self,
        *,
        window_title: str | None = None,
        app: str | None = None,
        profile_id: str | None = None,
    ) -> WindowInfo | None:
        profile = get_windows_app_profile(profile_id or guess_windows_app_profile(app or window_title or ""))
        title_tokens = []
        normalized_title = _normalized_text(window_title or "")
        if normalized_title:
            title_tokens.append(normalized_title)
        if profile is not None:
            title_tokens.extend(_normalized_text(item) for item in profile.window_title_tokens)
        candidates: list[WindowInfo] = []
        for item in self._enumerate_windows():
            normalized_window_title = _normalized_text(item.title)
            if not normalized_window_title:
                continue
            if any(token and token in normalized_window_title for token in title_tokens):
                candidates.append(item)
        if not candidates:
            return None
        foreground_handle = self._foreground_window_handle()
        preferred_handle = self._preferred_window_handle(
            profile_id=profile.profile_id if profile is not None else None,
            app=app,
            window_title=window_title,
        )
        actionable_handles: set[int] = set()
        if profile is not None:
            for item in candidates:
                if self._window_looks_actionable(item, profile):
                    actionable_handles.add(item.handle)
        if actionable_handles:
            candidates = [item for item in candidates if item.handle in actionable_handles]
        scored = sorted(
            candidates,
            key=lambda item: self._window_candidate_score(
                item,
                foreground_handle=foreground_handle,
                preferred_handle=preferred_handle,
            ),
            reverse=True,
        )
        return scored[0]

    def _resolve_control(
        self,
        window: WindowInfo,
        *,
        control_name: str | None = None,
        automation_id: str | None = None,
        control_type: str | None = None,
        max_depth: int,
    ) -> Any | None:
        root = self._control_root(window)
        if root is None:
            return None
        normalized_name = _normalized_text(control_name or "")
        normalized_automation_id = str(automation_id or "").strip().casefold()
        normalized_control_type = str(control_type or "").strip().casefold()
        for control, _depth in self._uia().WalkControl(root, includeTop=True, maxDepth=max_depth):
            if normalized_name and normalized_name not in _normalized_text(getattr(control, "Name", "")):
                continue
            if normalized_automation_id and normalized_automation_id != str(getattr(control, "AutomationId", "")).strip().casefold():
                continue
            if normalized_control_type and normalized_control_type != str(getattr(control, "ControlTypeName", "")).strip().casefold():
                continue
            if normalized_name or normalized_automation_id or normalized_control_type:
                return control
        return None

    def _resolve_edit_control(
        self,
        window: WindowInfo,
        *,
        control_name: str | None = None,
        automation_id: str | None = None,
        max_depth: int,
    ) -> Any | None:
        control = self._resolve_control(
            window,
            control_name=control_name,
            automation_id=automation_id,
            control_type="editcontrol",
            max_depth=max_depth,
        )
        if control is not None:
            return control
        root = self._control_root(window)
        if root is None:
            return None
        try:
            return root.EditControl(searchDepth=max_depth)
        except Exception:  # noqa: BLE001
            return None

    def _control_root(self, window: WindowInfo) -> Any | None:
        try:
            return self._uia().ControlFromHandle(window.handle)
        except Exception:  # noqa: BLE001
            return None

    def _collect_controls(self, root: Any, *, max_depth: int) -> list[ControlInfo]:
        controls: list[ControlInfo] = []
        auto = self._uia()
        for control, depth in auto.WalkControl(root, includeTop=True, maxDepth=max_depth):
            controls.append(
                ControlInfo(
                    name=str(getattr(control, "Name", "") or ""),
                    control_type=str(getattr(control, "ControlTypeName", "") or ""),
                    automation_id=str(getattr(control, "AutomationId", "") or ""),
                    class_name=str(getattr(control, "ClassName", "") or ""),
                    depth=int(depth),
                )
            )
        return controls

    def _collect_window_text(self, root: Any, *, max_depth: int) -> str:
        lines: list[str] = []
        seen: set[str] = set()
        auto = self._uia()
        for control, _depth in auto.WalkControl(root, includeTop=True, maxDepth=max_depth):
            for candidate in (
                str(getattr(control, "Name", "") or "").strip(),
                self._control_text(control),
            ):
                normalized = _normalized_text(candidate)
                if not normalized or normalized in seen:
                    continue
                seen.add(normalized)
                lines.append(candidate.strip())
        return "\n".join(lines)

    def _control_text(self, control: Any) -> str:
        value = ""
        with contextlib.suppress(Exception):
            pattern = getattr(control, "GetValuePattern", lambda: None)()
            if pattern is not None:
                value = str(pattern.Value or "").strip()
        if value:
            return value
        with contextlib.suppress(Exception):
            legacy = getattr(control, "GetLegacyIAccessiblePattern", lambda: None)()
            if legacy is not None:
                value = str(legacy.Value or "").strip()
        return value

    def _calculator_status(self, window: WindowInfo) -> str:
        profile = WINDOWS_APP_PROFILES.get("calculator")
        if profile is not None:
            for automation_id in profile.result_automation_ids:
                control = self._resolve_control(
                    window,
                    automation_id=automation_id,
                    max_depth=10,
                )
                if control is None:
                    continue
                for candidate in (
                    str(getattr(control, "Name", "") or "").strip(),
                    self._control_text(control),
                ):
                    parsed = self._extract_calculator_value(candidate)
                    if parsed:
                        return parsed
        fallback = self.read_window_text(window_title=window.title, profile_id="calculator").observed_outcome
        parsed = self._extract_calculator_value(fallback)
        return parsed or fallback

    def _type_into_calculator(self, window: WindowInfo, *, text: str) -> DesktopActionVerification:
        sequence = self._calculator_button_sequence(text)
        if not sequence:
            return self._failure("type_keys", window.title, "Calculator expression did not contain any supported input.")
        current_window = self._resolve_window(app="Calculator", profile_id="calculator") or window
        if current_window.handle not in self._initialized_calculator_windows:
            clear_attempt = self._invoke_control_by_automation_id(current_window, automation_id="clearButton")
            if not clear_attempt.success:
                clear_attempt = self._invoke_control_by_automation_id(current_window, automation_id="clearEntryButton")
            self._initialized_calculator_windows.add(current_window.handle)
        for automation_id in sequence:
            current_window = self._resolve_window(app="Calculator", profile_id="calculator") or current_window
            invoked = self._invoke_control_by_automation_id(current_window, automation_id=automation_id)
            if not invoked.success:
                return self._failure(
                    "type_keys",
                    current_window.title,
                    invoked.fallback_reason or invoked.observed_outcome or f"Could not invoke {automation_id}.",
                )
            time.sleep(0.05)
        return DesktopActionVerification(
            target=current_window.title,
            action="type_keys",
            observed_outcome=f"Entered calculator expression with {len(sequence)} native button presses.",
            success=True,
            verified=True,
            details={"button_sequence": list(sequence)},
        )

    def _invoke_control_by_automation_id(
        self,
        window: WindowInfo,
        *,
        automation_id: str,
        timeout_sec: float = 2.0,
    ) -> DesktopActionVerification:
        deadline = time.monotonic() + max(0.1, float(timeout_sec))
        control = None
        while time.monotonic() < deadline:
            control = self._resolve_control(window, automation_id=automation_id, max_depth=10)
            if control is not None:
                break
            time.sleep(0.1)
        if control is None:
            return self._failure("invoke_control", window.title, f"Control {automation_id!r} not found.")
        return self._invoke_control_object(control, target=automation_id, window=window)

    def _invoke_control_object(self, control: Any, *, target: str, window: WindowInfo) -> DesktopActionVerification:
        pattern = getattr(control, "GetInvokePattern", lambda: None)()
        try:
            if pattern is not None:
                pattern.Invoke()
            else:
                control.Click(simulateMove=False)
        except Exception as exc:  # noqa: BLE001
            return self._failure("invoke_control", target, str(exc))
        return DesktopActionVerification(
            target=target,
            action="invoke_control",
            observed_outcome=f"Invoked {target}.",
            success=True,
            verified=True,
            details={"window_title": window.title, "window_handle": window.handle},
        )

    @staticmethod
    def _calculator_button_sequence(text: str) -> list[str]:
        mapping = {
            "0": "num0Button",
            "1": "num1Button",
            "2": "num2Button",
            "3": "num3Button",
            "4": "num4Button",
            "5": "num5Button",
            "6": "num6Button",
            "7": "num7Button",
            "8": "num8Button",
            "9": "num9Button",
            "+": "plusButton",
            "-": "minusButton",
            "*": "multiplyButton",
            "x": "multiplyButton",
            "X": "multiplyButton",
            "×": "multiplyButton",
            "/": "divideButton",
            "÷": "divideButton",
            "=": "equalButton",
            "\n": "equalButton",
            "\r": "equalButton",
            ".": "decimalSeparatorButton",
            ",": "decimalSeparatorButton",
            "(": "openParenthesisButton",
            ")": "closeParenthesisButton",
        }
        sequence: list[str] = []
        for char in str(text or ""):
            if char.isspace():
                continue
            automation_id = mapping.get(char)
            if automation_id:
                sequence.append(automation_id)
        return sequence

    @staticmethod
    def _extract_calculator_value(text: str) -> str:
        candidate = str(text or "").replace("\xa0", " ").strip()
        if not candidate:
            return ""
        lowered = _normalized_text(candidate)
        for marker in ("ekran degeri", "display is", "sonuc", "result"):
            if lowered.startswith(marker):
                candidate = candidate.split(" ", maxsplit=len(marker.split(" ")))[-1].strip()
                break
        match = re.search(r"([-+]?[\d\s.,]+)$", candidate)
        if match:
            return match.group(1).strip()
        return candidate.strip()

    @classmethod
    def _translate_send_keys_text(cls, text: str) -> str:
        raw = str(text or "")
        if not raw:
            return raw

        translated: list[str] = []
        held_modifiers: list[str] = []
        index = 0
        length = len(raw)
        while index < length:
            if raw[index] != "{":
                translated.append(raw[index])
                index += 1
                continue
            closing_index = raw.find("}", index + 1)
            if closing_index == -1:
                translated.append(raw[index:])
                break
            token = raw[index + 1 : closing_index]
            translated.append(cls._translate_send_keys_token(token, held_modifiers))
            index = closing_index + 1
        if held_modifiers:
            translated.append(")" * len(held_modifiers))
        return "".join(translated)

    @classmethod
    def _translate_send_keys_token(cls, token: str, held_modifiers: list[str]) -> str:
        stripped = token.strip()
        if not stripped:
            return "{}"
        if stripped in {"{", "}"}:
            return "{" + stripped + "}"

        upper = stripped.upper()
        if upper.endswith(" DOWN"):
            modifier = cls._canonical_modifier_name(stripped[:-5])
            if modifier:
                held_modifiers.append(modifier)
                return f"{{{modifier}}}("
            return "{" + stripped + "}"
        if upper.endswith(" UP"):
            modifier = cls._canonical_modifier_name(stripped[:-3])
            if modifier and modifier in held_modifiers:
                held_modifiers.pop()
                return ")"
            return ""

        if "+" in stripped:
            chord_parts = [part.strip() for part in stripped.split("+") if part.strip()]
            if len(chord_parts) >= 2:
                modifiers = [cls._canonical_modifier_name(part) for part in chord_parts[:-1]]
                if all(modifiers):
                    return "".join(f"{{{modifier}}}" for modifier in modifiers) + cls._canonical_send_key(
                        chord_parts[-1],
                        chord=True,
                    )

        count_parts = stripped.split()
        if len(count_parts) == 2 and count_parts[1].isdigit():
            key = cls._canonical_send_key(count_parts[0], chord=False)
            if key.startswith("{") and key.endswith("}"):
                return f"{key[:-1]} {count_parts[1]}}}"
            return f"{{{count_parts[0]} {count_parts[1]}}}"

        return cls._canonical_send_key(stripped, chord=False)

    @staticmethod
    def _canonical_modifier_name(token: str) -> str | None:
        mapping = {
            "ALT": "Alt",
            "CTRL": "Ctrl",
            "CONTROL": "Ctrl",
            "LALT": "LAlt",
            "RALT": "RAlt",
            "LCTRL": "LCtrl",
            "RCTRL": "RCtrl",
            "SHIFT": "Shift",
            "LSHIFT": "LShift",
            "RSHIFT": "RShift",
            "WIN": "Win",
            "LWIN": "LWin",
            "RWIN": "RWin",
        }
        return mapping.get(str(token or "").strip().upper())

    @classmethod
    def _canonical_send_key(cls, token: str, *, chord: bool) -> str:
        stripped = str(token or "").strip()
        if len(stripped) == 1:
            if chord and stripped.isalpha():
                return stripped.lower()
            return stripped

        special_key_map = {
            "ALT": "Alt",
            "BACK": "Back",
            "BACKSPACE": "Back",
            "BKSP": "Back",
            "CTRL": "Ctrl",
            "CONTROL": "Ctrl",
            "DEL": "Delete",
            "DELETE": "Delete",
            "DOWN": "Down",
            "END": "End",
            "ENTER": "Enter",
            "ESC": "Esc",
            "ESCAPE": "Esc",
            "HOME": "Home",
            "INS": "Insert",
            "INSERT": "Insert",
            "LEFT": "Left",
            "PAGEDOWN": "PageDown",
            "PGDN": "PageDown",
            "PAGEUP": "PageUp",
            "PGUP": "PageUp",
            "RIGHT": "Right",
            "SHIFT": "Shift",
            "SPACE": "Space",
            "TAB": "Tab",
            "UP": "Up",
            "WIN": "Win",
        }
        modifier = cls._canonical_modifier_name(stripped)
        if modifier:
            return f"{{{modifier}}}"
        special = special_key_map.get(stripped.upper())
        if special:
            return f"{{{special}}}"
        return "{" + stripped + "}"

    def _enumerate_windows(self) -> list[WindowInfo]:
        if os.name != "nt":
            return []
        try:
            import win32gui  # noqa: PLC0415
            import win32process  # noqa: PLC0415
        except ImportError:
            return []

        windows: list[WindowInfo] = []

        def _callback(handle: int, _extra: Any) -> bool:
            if not win32gui.IsWindowVisible(handle):
                return True
            title = str(win32gui.GetWindowText(handle) or "").strip()
            if not title:
                return True
            class_name = str(win32gui.GetClassName(handle) or "").strip()
            try:
                _thread_id, process_id = win32process.GetWindowThreadProcessId(handle)
            except Exception:  # noqa: BLE001
                process_id = 0
            windows.append(
                WindowInfo(
                    handle=int(handle),
                    title=title,
                    class_name=class_name,
                    process_id=int(process_id),
                    visible=True,
                )
            )
            return True

        win32gui.EnumWindows(_callback, None)
        return windows

    def _window_candidate_score(
        self,
        window: WindowInfo,
        *,
        foreground_handle: int,
        preferred_handle: int,
    ) -> int:
        score = 0
        if window.class_name == "ApplicationFrameWindow":
            score += 40
        elif window.class_name == "Windows.UI.Core.CoreWindow":
            score += 20
        root = self._control_root(window)
        if root is not None and str(getattr(root, "ControlTypeName", "") or "") == "WindowControl":
            score += 30
        if window.handle == foreground_handle:
            score += 25
        if preferred_handle and window.handle == preferred_handle:
            score += 80
        return score

    def _window_looks_actionable(self, window: WindowInfo, profile: Any) -> bool:
        automation_ids = list(getattr(profile, "result_automation_ids", ()) or ())
        automation_ids.extend(getattr(profile, "expression_automation_ids", ()) or ())
        if getattr(profile, "profile_id", "") == "calculator":
            automation_ids.extend(("num0Button", "num1Button", "plusButton", "equalButton"))
        for automation_id in automation_ids:
            control = self._resolve_control(window, automation_id=automation_id, max_depth=8)
            if control is not None:
                return True
        return False

    @staticmethod
    def _foreground_window_handle() -> int:
        if os.name != "nt":
            return 0
        try:
            import win32gui  # noqa: PLC0415
        except ImportError:
            return 0
        try:
            return int(win32gui.GetForegroundWindow() or 0)
        except Exception:  # noqa: BLE001
            return 0

    def _remember_window(
        self,
        window: WindowInfo,
        *,
        profile_id: str | None = None,
        app: str | None = None,
        window_title: str | None = None,
    ) -> None:
        if profile_id == "calculator":
            return
        for key in self._window_preference_keys(profile_id=profile_id, app=app, window_title=window_title):
            self._preferred_window_handles[key] = window.handle

    def _preferred_window_handle(
        self,
        *,
        profile_id: str | None = None,
        app: str | None = None,
        window_title: str | None = None,
    ) -> int:
        if profile_id == "calculator":
            return 0
        for key in self._window_preference_keys(profile_id=profile_id, app=app, window_title=window_title):
            handle = int(self._preferred_window_handles.get(key, 0) or 0)
            if handle:
                return handle
        return 0

    @staticmethod
    def _window_preference_keys(
        *,
        profile_id: str | None = None,
        app: str | None = None,
        window_title: str | None = None,
    ) -> list[str]:
        keys: list[str] = []
        if profile_id:
            keys.append(f"profile:{profile_id}")
        if app:
            keys.append(f"app:{_normalized_text(app)}")
        if window_title:
            keys.append(f"title:{_normalized_text(window_title)}")
        return keys

    @staticmethod
    def _failure(action: str, target: str, reason: str) -> DesktopActionVerification:
        return DesktopActionVerification(
            target=target,
            action=action,
            observed_outcome=reason,
            success=False,
            verified=False,
            fallback_reason=reason,
        )

    @staticmethod
    def _uia() -> Any:
        import uiautomation as auto  # noqa: PLC0415

        return auto

    @contextlib.contextmanager
    def _clipboard_guard(self):
        try:
            import win32clipboard  # noqa: PLC0415
            import win32con  # noqa: PLC0415
        except ImportError:
            yield _ClipboardProxy(None, None)
            return

        previous_text = None
        try:
            win32clipboard.OpenClipboard()
            if win32clipboard.IsClipboardFormatAvailable(win32con.CF_UNICODETEXT):
                previous_text = win32clipboard.GetClipboardData(win32con.CF_UNICODETEXT)
        except Exception:  # noqa: BLE001
            previous_text = None
        finally:
            with contextlib.suppress(Exception):
                win32clipboard.CloseClipboard()

        proxy = _ClipboardProxy(win32clipboard, win32con)
        try:
            yield proxy
        finally:
            if previous_text is None:
                return
            try:
                win32clipboard.OpenClipboard()
                win32clipboard.EmptyClipboard()
                win32clipboard.SetClipboardText(previous_text, win32con.CF_UNICODETEXT)
            except Exception:  # noqa: BLE001
                pass
            finally:
                with contextlib.suppress(Exception):
                    win32clipboard.CloseClipboard()


class _ClipboardProxy:
    """Small clipboard helper used by calculator status verification."""

    def __init__(self, win32clipboard: Any, win32con: Any) -> None:
        self._win32clipboard = win32clipboard
        self._win32con = win32con

    def read_text(self) -> str:
        if self._win32clipboard is None or self._win32con is None:
            return ""
        try:
            self._win32clipboard.OpenClipboard()
            if self._win32clipboard.IsClipboardFormatAvailable(self._win32con.CF_UNICODETEXT):
                return str(self._win32clipboard.GetClipboardData(self._win32con.CF_UNICODETEXT) or "")
            return ""
        finally:
            with contextlib.suppress(Exception):
                self._win32clipboard.CloseClipboard()
