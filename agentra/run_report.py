"""Generate a lightweight HTML timeline for agent runs."""

from __future__ import annotations

import base64
import html
import json
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any


class RunReport:
    """Persist run events as JSON plus a presentable HTML report."""

    def __init__(self, workspace_dir: Path, goal: str, provider: str, model: str) -> None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        slug = _slugify(goal) or "run"
        self.run_dir = (workspace_dir / ".runs" / f"{timestamp}-{slug[:48]}").resolve()
        self.assets_dir = self.run_dir / "assets"
        self.events_path = self.run_dir / "events.json"
        self.html_path = self.run_dir / "index.html"
        self.goal = goal
        self.provider = provider
        self.model = model
        self.started_at = datetime.now().isoformat(timespec="seconds")
        self.finished_at: str | None = None
        self.status = "running"
        self._events: list[dict[str, Any]] = []
        self._screenshot_index = 0

        self.assets_dir.mkdir(parents=True, exist_ok=True)
        self._write()

    def record(self, event: dict[str, Any]) -> dict[str, Any]:
        """Append an event and refresh the HTML report."""
        stored = dict(event)
        stored["timestamp"] = datetime.now().isoformat(timespec="seconds")
        if stored.get("type") == "screenshot" and stored.get("data"):
            stored["image_path"] = self._save_screenshot(stored["data"])
            stored.pop("data", None)
        self._events.append(stored)
        self._write()
        return stored

    def finalize(self, status: str) -> None:
        """Mark the run complete and write the final report."""
        self.status = status
        self.finished_at = datetime.now().isoformat(timespec="seconds")
        self._write()

    def open(self) -> None:
        """Open the HTML report in the default browser."""
        if os.name == "nt":
            os.startfile(self.html_path)  # type: ignore[attr-defined]
            return
        if os.name == "posix":
            opener = "open" if sys_platform() == "darwin" else "xdg-open"
            subprocess.Popen([opener, str(self.html_path)])

    @property
    def events(self) -> list[dict[str, Any]]:
        return list(self._events)

    def _save_screenshot(self, b64: str) -> str:
        self._screenshot_index += 1
        filename = f"screenshot-{self._screenshot_index:03d}.png"
        path = self.assets_dir / filename
        path.write_bytes(base64.b64decode(b64))
        return f"assets/{filename}"

    def _write(self) -> None:
        payload = {
            "goal": self.goal,
            "provider": self.provider,
            "model": self.model,
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "events": self._events,
        }
        self.events_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self.html_path.write_text(self._render_html(), encoding="utf-8")

    def _render_html(self) -> str:
        refresh = '<meta http-equiv="refresh" content="2" />' if self.status == "running" else ""
        event_cards = "\n".join(self._render_event(event) for event in self._events) or (
            '<section class="card empty"><h2>Waiting for events</h2>'
            "<p>The report will update as the agent produces output.</p></section>"
        )
        return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  {refresh}
  <title>Agentra Run Report</title>
  <style>
    :root {{
      --bg: #0b1220;
      --panel: rgba(12, 22, 42, 0.88);
      --panel-2: rgba(18, 31, 58, 0.92);
      --line: rgba(148, 163, 184, 0.18);
      --text: #e5eefb;
      --muted: #9eb2cf;
      --blue: #68b3ff;
      --cyan: #65e0ff;
      --green: #65f0b5;
      --amber: #ffd166;
      --red: #ff7d7d;
      --shadow: 0 24px 80px rgba(0, 0, 0, 0.35);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      min-height: 100vh;
      color: var(--text);
      font-family: "Segoe UI", "Helvetica Neue", sans-serif;
      background:
        radial-gradient(circle at top left, rgba(104, 179, 255, 0.28), transparent 28%),
        radial-gradient(circle at top right, rgba(101, 240, 181, 0.18), transparent 24%),
        linear-gradient(160deg, #08111f 0%, #0b1220 46%, #12203a 100%);
    }}
    .shell {{
      max-width: 1220px;
      margin: 32px auto;
      padding: 0 20px 32px;
    }}
    .hero {{
      display: grid;
      gap: 18px;
      grid-template-columns: minmax(0, 1.6fr) minmax(280px, 0.9fr);
      margin-bottom: 22px;
    }}
    .frame {{
      background: linear-gradient(135deg, rgba(94, 165, 255, 0.92), rgba(86, 102, 255, 0.7));
      border-radius: 28px;
      padding: 14px;
      box-shadow: var(--shadow);
    }}
    .frame-inner {{
      min-height: 220px;
      border-radius: 20px;
      padding: 24px;
      background:
        linear-gradient(180deg, rgba(255,255,255,0.94), rgba(238,243,251,0.92) 10%, rgba(18,31,58,0.96) 10%, rgba(18,31,58,0.96) 100%);
      position: relative;
      overflow: hidden;
    }}
    .window-bar {{
      color: #4f5d73;
      font-size: 14px;
      text-align: center;
      margin-bottom: 24px;
    }}
    .goal {{
      background: rgba(255, 255, 255, 0.1);
      border: 1px solid rgba(255, 255, 255, 0.14);
      border-radius: 18px;
      padding: 18px;
      font-size: 24px;
      line-height: 1.35;
      max-width: 90%;
    }}
    .live {{
      position: absolute;
      right: 18px;
      bottom: 12px;
      color: white;
      font-size: 13px;
      letter-spacing: 0.18em;
    }}
    .meta {{
      display: grid;
      gap: 14px;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 20px;
      padding: 18px;
      box-shadow: var(--shadow);
      backdrop-filter: blur(14px);
    }}
    .card h2, .card h3 {{
      margin: 0 0 10px;
      font-size: 16px;
    }}
    .status {{
      display: inline-flex;
      align-items: center;
      gap: 10px;
      padding: 8px 12px;
      border-radius: 999px;
      font-weight: 700;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      font-size: 12px;
      background: rgba(104, 179, 255, 0.14);
      color: var(--blue);
    }}
    .status.running {{ color: var(--amber); background: rgba(255, 209, 102, 0.12); }}
    .status.completed {{ color: var(--green); background: rgba(101, 240, 181, 0.12); }}
    .status.partial {{ color: var(--blue); }}
    .status.error {{ color: var(--red); background: rgba(255, 125, 125, 0.12); }}
    .meta-grid {{
      display: grid;
      gap: 12px;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      color: var(--muted);
      font-size: 14px;
    }}
    .meta-grid strong {{
      display: block;
      color: var(--text);
      font-size: 15px;
      margin-top: 4px;
    }}
    .timeline {{
      display: grid;
      gap: 14px;
    }}
    .event {{
      background: var(--panel-2);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 16px;
      position: relative;
      overflow: hidden;
    }}
    .event::before {{
      content: "";
      position: absolute;
      inset: 0 auto 0 0;
      width: 4px;
      background: var(--blue);
    }}
    .event.thought::before {{ background: var(--cyan); }}
    .event.tool_call::before {{ background: var(--amber); }}
    .event.tool_result.success::before,
    .event.done::before,
    .event.sub_task.success::before {{ background: var(--green); }}
    .event.tool_result.error::before,
    .event.error::before,
    .event.sub_task.error::before {{ background: var(--red); }}
    .eyebrow {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 10px;
      color: var(--muted);
      font-size: 12px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }}
    .event h3 {{
      margin: 0 0 10px;
      font-size: 18px;
    }}
    pre {{
      margin: 0;
      padding: 14px;
      overflow: auto;
      border-radius: 14px;
      background: rgba(6, 13, 24, 0.72);
      border: 1px solid rgba(148, 163, 184, 0.12);
      color: #d8e2f2;
      font-family: "Cascadia Code", "Consolas", monospace;
      font-size: 13px;
      line-height: 1.45;
      white-space: pre-wrap;
      word-break: break-word;
    }}
    p {{
      margin: 0;
      color: var(--text);
      line-height: 1.55;
    }}
    img {{
      display: block;
      width: 100%;
      border-radius: 16px;
      border: 1px solid rgba(148, 163, 184, 0.12);
      background: rgba(255, 255, 255, 0.04);
    }}
    .muted {{ color: var(--muted); }}
    .empty {{ text-align: center; color: var(--muted); }}
    @media (max-width: 960px) {{
      .hero {{ grid-template-columns: 1fr; }}
      .meta-grid {{ grid-template-columns: 1fr; }}
      .goal {{ max-width: none; font-size: 20px; }}
    }}
  </style>
</head>
<body>
  <main class="shell">
    <section class="hero">
      <div class="frame">
        <div class="frame-inner">
          <div class="window-bar">Agentra Live Run</div>
          <div class="goal">{html.escape(self.goal)}</div>
          <div class="live">{self._live_badge()}</div>
        </div>
      </div>
      <aside class="meta">
        <section class="card">
          <div class="status {self.status}">{self.status.upper()}</div>
          <div class="meta-grid" style="margin-top: 14px;">
            <div>Provider<strong>{html.escape(self.provider)}</strong></div>
            <div>Model<strong>{html.escape(self.model)}</strong></div>
            <div>Started<strong>{html.escape(self.started_at)}</strong></div>
            <div>Finished<strong>{html.escape(self.finished_at or "Live")}</strong></div>
            <div>Events<strong>{len(self._events)}</strong></div>
            <div>Report<strong>{html.escape(str(self.html_path))}</strong></div>
          </div>
        </section>
        <section class="card">
          <h2>Demo Notes</h2>
          <p class="muted">This standalone report is refreshed as the agent runs and keeps screenshots, tool calls, and model output in one shareable place.</p>
        </section>
      </aside>
    </section>
    <section class="timeline">
      {event_cards}
    </section>
  </main>
</body>
</html>
"""

    def _render_event(self, event: dict[str, Any]) -> str:
        event_type = str(event.get("type", "event"))
        timestamp = html.escape(str(event.get("timestamp", "")))
        title = event_type.replace("_", " ").title()
        if event_type == "thought":
            return self._event_shell(event_type, title, timestamp, self._content_block(event.get("content", "")))
        if event_type == "tool_call":
            tool = html.escape(str(event.get("tool", "tool")))
            args = html.escape(json.dumps(event.get("args", {}), indent=2))
            body = f"<h3>{tool}</h3><pre>{args}</pre>"
            return self._event_shell(event_type, "Tool Call", timestamp, body)
        if event_type == "tool_result":
            success = "success" if event.get("success") else "error"
            tool = html.escape(str(event.get("tool", "tool")))
            result = self._content_block(event.get("result", ""))
            return self._event_shell(f"{event_type} {success}", f"Tool Result: {tool}", timestamp, result)
        if event_type == "screenshot":
            image_path = html.escape(str(event.get("image_path", "")))
            body = (
                f'<h3>Screenshot #{self._find_screenshot_number(image_path)}</h3>'
                f'<img src="{image_path}" alt="Agent screenshot" />'
            )
            return self._event_shell(event_type, "Screenshot", timestamp, body)
        if event_type == "done":
            return self._event_shell(event_type, "Done", timestamp, self._content_block(event.get("content", "")))
        if event_type == "error":
            return self._event_shell(event_type, "Error", timestamp, self._content_block(event.get("content", "")))
        if event_type == "sub_task":
            success = "success" if event.get("success") else "error"
            label = html.escape(str(event.get("label", "Sub-task")))
            result = self._content_block(event.get("result", ""))
            return self._event_shell(f"{event_type} {success}", label, timestamp, result)
        return self._event_shell(event_type, title, timestamp, self._content_block(json.dumps(event, indent=2)))

    def _event_shell(self, kind: str, title: str, timestamp: str, body: str) -> str:
        return (
            f'<section class="event {html.escape(kind)}">'
            f'<div class="eyebrow"><span>{html.escape(title)}</span><span>{timestamp}</span></div>'
            f"{body}</section>"
        )

    @staticmethod
    def _content_block(content: Any) -> str:
        text = html.escape(str(content))
        if "\n" in str(content) or len(str(content)) > 180:
            return f"<pre>{text}</pre>"
        return f"<p>{text}</p>"

    @staticmethod
    def _find_screenshot_number(image_path: str) -> str:
        match = re.search(r"(\d+)\.png$", image_path)
        return match.group(1) if match else "?"

    def _live_badge(self) -> str:
        return "LIVE" if self.status == "running" else self.status.upper()


def _slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def sys_platform() -> str:
    return os.uname().sysname.lower() if hasattr(os, "uname") else ""
