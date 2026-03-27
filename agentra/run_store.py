"""Persistent event and frame storage for agent runs."""

from __future__ import annotations

import base64
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


class RunStore:
    """Store events, screenshots, and timeline frames for a single run."""

    def __init__(
        self,
        workspace_dir: Path,
        goal: str,
        provider: str,
        model: str,
        *,
        thread_id: str | None = None,
        thread_title: str | None = None,
    ) -> None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        slug = _slugify(goal) or "run"
        self.run_dir = (workspace_dir / ".runs" / f"{timestamp}-{slug[:48]}").resolve()
        self.assets_dir = self.run_dir / "assets"
        self.debug_images_dir = self.run_dir / "debug-images"
        self.events_path = self.run_dir / "events.json"
        self.html_path = self.run_dir / "index.html"
        self.run_id = self.run_dir.name
        self.goal = goal
        self.provider = provider
        self.model = model
        self.thread_id = thread_id
        self.thread_title = thread_title
        self.started_at = datetime.now().isoformat(timespec="seconds")
        self.finished_at: str | None = None
        self.status = "running"
        self._events: list[dict[str, Any]] = []
        self._frames: list[dict[str, Any]] = []
        self._audit: list[dict[str, Any]] = []
        self._screenshot_index = 0
        self._debug_image_index = 0
        self._pending_frame_id: str | None = None
        self._last_tool_call: dict[str, Any] | None = None
        self._last_debug_digest_by_source: dict[str, str] = {}

        self.assets_dir.mkdir(parents=True, exist_ok=True)
        self._write_events()

    @property
    def events(self) -> list[dict[str, Any]]:
        return list(self._events)

    @property
    def frames(self) -> list[dict[str, Any]]:
        return list(self._frames)

    @property
    def audit(self) -> list[dict[str, Any]]:
        return list(self._audit)

    def record(self, event: dict[str, Any]) -> dict[str, Any]:
        """Append *event* to the run history and persist the updated state."""
        stored = self._sanitize(event)
        stored["timestamp"] = datetime.now().isoformat(timespec="seconds")
        event_type = str(stored.get("type", "event"))

        if event_type == "tool_call":
            stored.setdefault("summary", self._tool_call_summary(stored))
            self._last_tool_call = stored
        elif event_type == "tool_result":
            stored.setdefault(
                "summary",
                self._plain_summary(
                    str(stored.get("result") or stored.get("error") or stored.get("tool", ""))
                ),
            )
            self._apply_summary_to_pending_frame(str(stored["summary"]))
        elif event_type == "thought":
            stored["summary"] = self._thought_summary(str(stored.get("content", "")))
            self._apply_summary_to_pending_frame(str(stored["summary"]))
        elif event_type in {"done", "error"}:
            stored.setdefault("summary", self._plain_summary(str(stored.get("content", ""))))
        elif event_type == "sub_task":
            stored.setdefault("summary", self._plain_summary(str(stored.get("result", ""))))

        if event_type == "screenshot" and stored.get("data"):
            image_bytes = base64.b64decode(str(stored["data"]))
            stored["image_path"] = self._save_screenshot(image_bytes)
            stored.pop("data", None)

            frame_id = f"frame-{len(self._frames) + 1:03d}"
            frame_label = str(
                stored.get("frame_label")
                or self._frame_label_from_tool_call(self._last_tool_call)
                or "Visual Frame"
            )
            summary = str(
                stored.get("summary")
                or self._frame_summary_from_tool_call(self._last_tool_call)
                or frame_label
            )
            focus_x = self._coerce_focus(stored.get("focus_x"))
            focus_y = self._coerce_focus(stored.get("focus_y"))

            stored.update(
                {
                    "frame_id": frame_id,
                    "frame_label": frame_label,
                    "summary": summary,
                }
            )
            if focus_x is not None:
                stored["focus_x"] = focus_x
            if focus_y is not None:
                stored["focus_y"] = focus_y

            self._frames.append(
                {
                    "id": frame_id,
                    "timestamp": stored["timestamp"],
                    "image_path": stored["image_path"],
                    "label": frame_label,
                    "summary": summary,
                    "focus_x": focus_x,
                    "focus_y": focus_y,
                }
            )
            self._pending_frame_id = frame_id
            self.save_debug_image_bytes(
                image_bytes,
                source="run-screenshots",
                media_type="image/png",
                label=frame_label,
            )

        self._events.append(stored)
        self._write_events()
        return stored

    def finalize(self, status: str) -> None:
        """Mark the run complete and persist the terminal status."""
        self.status = status
        self.finished_at = datetime.now().isoformat(timespec="seconds")
        self._write_events()

    def record_audit(self, entry: dict[str, Any]) -> dict[str, Any]:
        stored = self._sanitize(entry)
        stored.setdefault("timestamp", datetime.now().isoformat(timespec="seconds"))
        self._audit.append(stored)
        self._write_events()
        return stored

    def snapshot(self) -> dict[str, Any]:
        """Return a JSON-serializable snapshot of the run."""
        return {
            "run_id": self.run_id,
            "goal": self.goal,
            "provider": self.provider,
            "model": self.model,
            "thread_id": self.thread_id,
            "thread_title": self.thread_title,
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "report_path": str(self.html_path),
            "events": self.events,
            "frames": self.frames,
            "audit": self.audit,
        }

    def _apply_summary_to_pending_frame(self, summary: str) -> None:
        if not summary or not self._pending_frame_id:
            return
        for frame in reversed(self._frames):
            if frame["id"] == self._pending_frame_id:
                frame["summary"] = summary
                break
        for event in reversed(self._events):
            if event.get("frame_id") == self._pending_frame_id:
                event["summary"] = summary
                break

    def _save_screenshot(self, image_bytes: bytes) -> str:
        self._screenshot_index += 1
        filename = f"screenshot-{self._screenshot_index:03d}.png"
        path = self.assets_dir / filename
        path.write_bytes(image_bytes)
        return f"assets/{filename}"

    def save_debug_image_bytes(
        self,
        image_bytes: bytes,
        *,
        source: str,
        media_type: str,
        label: str | None = None,
        dedupe: bool = False,
    ) -> str | None:
        source_name = _slugify(source) or "debug"
        digest = hashlib.sha256(image_bytes).hexdigest()
        if dedupe and self._last_debug_digest_by_source.get(source_name) == digest:
            return None
        self._last_debug_digest_by_source[source_name] = digest

        self._debug_image_index += 1
        target_dir = self.debug_images_dir / source_name
        target_dir.mkdir(parents=True, exist_ok=True)

        label_name = _slugify(label or "")[:48]
        filename = f"{self._debug_image_index:04d}-{digest[:12]}"
        if label_name:
            filename += f"-{label_name}"
        filename += _image_extension_for_media_type(media_type)

        path = target_dir / filename
        path.write_bytes(image_bytes)
        return str(path.relative_to(self.run_dir)).replace("\\", "/")

    def _write_events(self) -> None:
        payload = self.snapshot()
        self.events_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def _sanitize(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return {str(key): cls._sanitize(item) for key, item in value.items()}
        if isinstance(value, list):
            return [cls._sanitize(item) for item in value]
        if isinstance(value, tuple):
            return [cls._sanitize(item) for item in value]
        if isinstance(value, Path):
            return str(value)
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        return str(value)

    @staticmethod
    def _coerce_focus(value: Any) -> float | None:
        if value is None:
            return None
        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return None

    def _tool_call_summary(self, event: dict[str, Any]) -> str:
        tool = str(event.get("tool", "tool"))
        args = event.get("args", {})
        if tool == "browser" and isinstance(args, dict):
            action = str(args.get("action", "browser")).lower()
            if action == "navigate":
                return f"Opening {self._short_url(str(args.get('url', 'the page')))}"
            if action == "click":
                selector = args.get("selector")
                if selector:
                    return f"Clicking {selector}"
                return "Clicking the page"
            if action == "type":
                selector = args.get("selector")
                if selector:
                    return f"Typing into {selector}"
                return "Typing text"
            if action == "scroll":
                return "Scrolling the page"
            if action == "back":
                return "Going back"
            if action == "forward":
                return "Going forward"
            if action == "new_tab":
                return "Opening a new tab"
            if action == "close_tab":
                return "Closing the tab"
            if action == "screenshot":
                return "Capturing a screenshot"
        if tool == "windows_desktop" and isinstance(args, dict):
            action = str(args.get("action", "windows_desktop")).lower()
            target = str(
                args.get("app")
                or args.get("window_title")
                or args.get("profile_id")
                or args.get("control_name")
                or "the Windows app"
            )
            if action == "launch_app":
                return f"Launching {target}"
            if action == "focus_window":
                return f"Focusing {target}"
            if action == "wait_for_window":
                return f"Waiting for {target}"
            if action == "list_controls":
                return f"Inspecting controls in {target}"
            if action == "set_text":
                return f"Setting text in {target}"
            if action == "type_keys":
                return f"Typing keys in {target}"
            if action == "read_status":
                return f"Verifying {target}"
        return self._plain_summary(f"Running {tool}")

    def _frame_label_from_tool_call(self, event: dict[str, Any] | None) -> str | None:
        if not event:
            return None
        tool = str(event.get("tool", "tool"))
        action = ""
        args = event.get("args", {})
        if isinstance(args, dict):
            action = str(args.get("action", "")).strip()
        if action:
            return f"{tool} · {action}"
        return tool

    def _frame_summary_from_tool_call(self, event: dict[str, Any] | None) -> str | None:
        if not event:
            return None
        return str(event.get("summary") or self._tool_call_summary(event))

    def _thought_summary(self, content: str) -> str:
        content = re.split(r"(?im)^\s*DONE:\s*", content, maxsplit=1)[0]
        lines = []
        for raw_line in content.splitlines():
            cleaned = raw_line.strip().lstrip("-*•").strip()
            cleaned = re.sub(r"\*\*(.*?)\*\*", r"\1", cleaned)
            if cleaned:
                lines.append(cleaned)
        if not lines:
            return self._plain_summary(content)
        first_line = lines[0]
        if first_line.endswith(":") and len(lines) > 1:
            first_line = lines[1]
        return self._plain_summary(first_line)

    @staticmethod
    def _plain_summary(text: str, limit: int = 170) -> str:
        compact = re.sub(r"\s+", " ", text).strip()
        if len(compact) <= limit:
            return compact
        return compact[: limit - 3].rstrip() + "..."

    @staticmethod
    def _short_url(url: str) -> str:
        parsed = urlparse(url)
        if parsed.netloc:
            return parsed.netloc.removeprefix("www.")
        return url or "the page"


def _slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def _image_extension_for_media_type(media_type: str) -> str:
    media = str(media_type or "").lower()
    if "jpeg" in media or "jpg" in media:
        return ".jpg"
    if "webp" in media:
        return ".webp"
    return ".png"
