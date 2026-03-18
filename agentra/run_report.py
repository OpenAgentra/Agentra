"""Generate and persist HTML reports for agent runs."""

from __future__ import annotations

import html
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

from agentra.run_store import RunStore


class RunReport:
    """Persist run events plus a presentable HTML report."""

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
        self.store = RunStore(
            workspace_dir,
            goal,
            provider,
            model,
            thread_id=thread_id,
            thread_title=thread_title,
        )
        self.run_dir = self.store.run_dir
        self.assets_dir = self.store.assets_dir
        self.events_path = self.store.events_path
        self.html_path = self.store.html_path
        self.goal = goal
        self.provider = provider
        self.model = model
        self._write_html()

    @property
    def events(self) -> list[dict[str, Any]]:
        return self.store.events

    @property
    def frames(self) -> list[dict[str, Any]]:
        return self.store.frames

    @property
    def audit(self) -> list[dict[str, Any]]:
        return self.store.audit

    def snapshot(self) -> dict[str, Any]:
        return self.store.snapshot()

    def record(self, event: dict[str, Any]) -> dict[str, Any]:
        stored = self.store.record(event)
        self._write_html()
        return stored

    def record_audit(self, entry: dict[str, Any]) -> dict[str, Any]:
        stored = self.store.record_audit(entry)
        self._write_html()
        return stored

    def finalize(self, status: str) -> None:
        self.store.finalize(status)
        self._write_html()

    def open(self) -> None:
        """Open the report in the default browser."""
        if os.name == "nt":
            os.startfile(self.html_path)  # type: ignore[attr-defined]
            return
        if os.name == "posix":
            opener = "open" if _sys_platform() == "darwin" else "xdg-open"
            subprocess.Popen([opener, str(self.html_path)])

    def _write_html(self) -> None:
        self.html_path.write_text(self._render_html(self.snapshot()), encoding="utf-8")

    def _render_html(self, snapshot: dict[str, Any]) -> str:
        status = str(snapshot.get("status", "running"))
        events = snapshot.get("events", [])
        frames = snapshot.get("frames", [])
        audit = snapshot.get("audit", [])
        refresh = '<meta http-equiv="refresh" content="2" />' if status == "running" else ""
        event_cards = "\n".join(self._render_event(event) for event in events) or (
            '<section class="card empty"><h2>Waiting for events</h2>'
            "<p>The report will update as the agent produces output.</p></section>"
        )
        latest_frame = frames[-1] if frames else None
        hero_media = (
            f'<div class="hero-shot"><img src="{html.escape(str(latest_frame["image_path"]))}" alt="Latest run screenshot" /></div>'
            if latest_frame
            else ""
        )
        frame_strip = self._render_frame_strip(frames)
        audit_strip = self._render_audit_strip(audit)

        return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  {refresh}
  <title>Agentra Run Report</title>
  <style>
    :root {{
      --bg: #09111e;
      --panel: rgba(12, 22, 42, 0.9);
      --panel-2: rgba(18, 31, 58, 0.94);
      --line: rgba(148, 163, 184, 0.18);
      --text: #edf3ff;
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
      font-family: "Trebuchet MS", "Aptos", "Segoe UI Variable", sans-serif;
      background:
        radial-gradient(circle at top left, rgba(104, 179, 255, 0.28), transparent 28%),
        radial-gradient(circle at top right, rgba(101, 240, 181, 0.18), transparent 24%),
        linear-gradient(160deg, #08111f 0%, #0b1220 46%, #12203a 100%);
    }}
    .shell {{
      max-width: 1280px;
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
    .hero-shot {{
      margin: 0 0 18px;
      border-radius: 18px;
      overflow: hidden;
      border: 1px solid rgba(148, 163, 184, 0.16);
      background: rgba(255, 255, 255, 0.05);
    }}
    .hero-shot img {{
      aspect-ratio: 16 / 9;
      object-fit: cover;
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
      overflow-wrap: anywhere;
      word-break: break-word;
    }}
    .frame-strip {{
      display: grid;
      gap: 12px;
      grid-auto-flow: column;
      grid-auto-columns: minmax(190px, 220px);
      overflow-x: auto;
      padding-bottom: 6px;
    }}
    .frame-thumb {{
      background: rgba(8, 17, 31, 0.58);
      border: 1px solid rgba(148, 163, 184, 0.16);
      border-radius: 16px;
      padding: 10px;
    }}
    .frame-thumb img {{
      aspect-ratio: 16 / 9;
      object-fit: cover;
      margin-bottom: 10px;
    }}
    .frame-thumb strong {{
      display: block;
      margin-bottom: 6px;
      font-size: 14px;
      color: var(--text);
    }}
    .frame-thumb span {{
      display: block;
      font-size: 12px;
      color: var(--muted);
      margin-bottom: 6px;
    }}
    .frame-thumb p {{
      font-size: 13px;
      color: var(--text);
    }}
    .timeline, .audit-strip {{
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
          {hero_media}
          <div class="goal">{html.escape(str(snapshot["goal"]))}</div>
          <div class="live">{status.upper() if status != "running" else "LIVE"}</div>
        </div>
      </div>
      <aside class="meta">
        <section class="card">
          <div class="status {status}">{status.upper()}</div>
          <div class="meta-grid" style="margin-top: 14px;">
            <div>Provider<strong>{html.escape(str(snapshot["provider"]))}</strong></div>
            <div>Model<strong>{html.escape(str(snapshot["model"]))}</strong></div>
            <div>Started<strong>{html.escape(str(snapshot["started_at"]))}</strong></div>
            <div>Finished<strong>{html.escape(str(snapshot.get("finished_at") or "Live"))}</strong></div>
            <div>Events<strong>{len(events)}</strong></div>
            <div>Frames<strong>{len(frames)}</strong></div>
            <div>Report<strong>{html.escape(str(snapshot["report_path"]))}</strong></div>
          </div>
        </section>
        <section class="card">
          <h2>Demo Notes</h2>
          <p class="muted">This standalone report is refreshed as the agent runs and keeps screenshots, timeline frames, tool calls, and model output in one shareable place.</p>
        </section>
      </aside>
    </section>
    {frame_strip}
    {audit_strip}
    <section class="timeline">
      {event_cards}
    </section>
  </main>
</body>
</html>
"""

    def _render_frame_strip(self, frames: list[dict[str, Any]]) -> str:
        if not frames:
            return ""
        cards = "\n".join(
            (
                '<article class="frame-thumb">'
                f'<img src="{html.escape(str(frame["image_path"]))}" alt="{html.escape(str(frame["label"]))}" />'
                f'<strong>{html.escape(str(frame["label"]))}</strong>'
                f'<span>{html.escape(str(frame["timestamp"]))}</span>'
                f'<p>{html.escape(str(frame.get("summary", "")))}</p>'
                "</article>"
            )
            for frame in frames
        )
        return (
            '<section class="card"><h2>Frame Timeline</h2>'
            f'<div class="frame-strip">{cards}</div></section>'
        )

    def _render_audit_strip(self, audit: list[dict[str, Any]]) -> str:
        if not audit:
            return ""
        cards = "\n".join(self._render_audit_entry(entry) for entry in audit)
        return (
            '<section class="card"><h2>Audit Trail</h2>'
            f'<div class="audit-strip">{cards}</div></section>'
        )

    def _render_event(self, event: dict[str, Any]) -> str:
        event_type = str(event.get("type", "event"))
        timestamp = html.escape(str(event.get("timestamp", "")))
        title = event_type.replace("_", " ").title()
        if event_type == "phase":
            return self._event_shell(
                event_type,
                html.escape(str(event.get("phase", "Phase")).title()),
                timestamp,
                self._content_block(event.get("summary") or event.get("content", "")),
            )
        if event_type == "thought":
            return self._event_shell(
                event_type,
                title,
                timestamp,
                self._content_block(event.get("content", "")),
            )
        if event_type == "visual_intent":
            body = self._content_block(event.get("summary", "Pending visual action."))
            return self._event_shell(event_type, "Visual Intent", timestamp, body)
        if event_type == "tool_call":
            tool = html.escape(str(event.get("tool", "tool")))
            args = html.escape(json.dumps(event.get("args", {}), indent=2))
            body = f"<h3>{tool}</h3><pre>{args}</pre>"
            return self._event_shell(event_type, "Tool Call", timestamp, body)
        if event_type == "tool_result":
            success = "success" if event.get("success") else "error"
            tool = html.escape(str(event.get("tool", "tool")))
            result = self._content_block(event.get("result", ""))
            return self._event_shell(
                f"{event_type} {success}",
                f"Tool Result: {tool}",
                timestamp,
                result,
            )
        if event_type == "screenshot":
            image_path = html.escape(str(event.get("image_path", "")))
            frame_label = html.escape(str(event.get("frame_label", "Screenshot")))
            body = (
                f"<h3>{frame_label}</h3>"
                f'<img src="{image_path}" alt="Agent screenshot" />'
            )
            return self._event_shell(event_type, "Screenshot", timestamp, body)
        if event_type == "done":
            return self._event_shell(
                event_type, "Done", timestamp, self._content_block(event.get("content", ""))
            )
        if event_type == "error":
            return self._event_shell(
                event_type, "Error", timestamp, self._content_block(event.get("content", ""))
            )
        if event_type == "sub_task":
            success = "success" if event.get("success") else "error"
            label = html.escape(str(event.get("label", "Sub-task")))
            result = self._content_block(event.get("result", ""))
            return self._event_shell(f"{event_type} {success}", label, timestamp, result)
        return self._event_shell(
            event_type,
            title,
            timestamp,
            self._content_block(json.dumps(event, indent=2)),
        )

    def _event_shell(self, kind: str, title: str, timestamp: str, body: str) -> str:
        return (
            f'<section class="event {html.escape(kind)}">'
            f'<div class="eyebrow"><span>{html.escape(title)}</span><span>{timestamp}</span></div>'
            f"{body}</section>"
        )

    def _render_audit_entry(self, entry: dict[str, Any]) -> str:
        title = str(entry.get("entry_type", "audit")).replace("_", " ").title()
        timestamp = html.escape(str(entry.get("timestamp", "")))
        details = entry.get("details", {})
        body = self._content_block(json.dumps(details, indent=2, ensure_ascii=False))
        return self._event_shell("audit", title, timestamp, body)

    @staticmethod
    def _content_block(content: Any) -> str:
        text = html.escape(str(content))
        if "\n" in str(content) or len(str(content)) > 180:
            return f"<pre>{text}</pre>"
        return f"<p>{text}</p>"


def _sys_platform() -> str:
    return os.uname().sysname.lower() if hasattr(os, "uname") else ""
