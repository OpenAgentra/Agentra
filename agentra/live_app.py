"""Local live UI for Agentra runs."""

from __future__ import annotations

import asyncio
import html
import json
import logging
import threading
import webbrowser
from pathlib import Path
from typing import Any, Callable, Literal

from fastapi import Request as FastAPIRequest
from pydantic import BaseModel, Field

from agentra.config import AgentConfig
from agentra.logging_utils import (
    app_log_path,
    configure_app_logging,
    read_log_tail,
)
from agentra.runtime import ThreadManager

EventPayload = dict[str, Any]
AgentFactory = Callable[[AgentConfig], Any]
logger = logging.getLogger(__name__)


class RunCreateRequest(BaseModel):
    """Payload to start a live run."""

    goal: str = Field(min_length=1)
    thread_id: str | None = None
    thread_title: str | None = None
    provider: str | None = None
    model: str | None = None
    headless: bool | None = None
    workspace: str | None = None
    max_iterations: int | None = Field(default=None, ge=1)
    permission_mode: Literal["default", "full"] | None = None


class ApprovalDecisionRequest(BaseModel):
    approved: bool
    note: str = ""


class UserAnswerRequest(BaseModel):
    answer: str = Field(min_length=1)


class HumanActionRequest(BaseModel):
    tool: str = Field(min_length=1)
    args: dict[str, Any] = Field(default_factory=dict)
    return_snapshot: bool = True


class ThreadSettingsUpdateRequest(BaseModel):
    permission_mode: Literal["default", "full"] | None = None
    desktop_execution_mode: Literal["desktop_visible", "desktop_native", "desktop_hidden"] | None = None


def _generic_tool_label(tool: str) -> str:
    if not tool:
        return "Araç"
    if tool == "browser":
        return "Tarayıcı · İşlem"
    if tool == "computer":
        return "Masaüstü · İşlem"
    if tool == "windows_desktop":
        return "Yerel Masaüstü · İşlem"
    if tool == "local_system":
        return "Yerel Sistem · İşlem"
    if tool == "filesystem":
        return "Dosya Sistemi · İşlem"
    return f"{tool.replace('_', ' ').title()} · İşlem"


def _short_url(url: str) -> str:
    compact = str(url or "").replace("https://", "").replace("http://", "").rstrip("/")
    return compact.removeprefix("www.") or "sayfa"


def _trim_text(value: Any, limit: int = 220) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _parse_action_from_label(label: str | None) -> str | None:
    if not label:
        return None
    if "·" in label:
        return label.split("·", 1)[1].strip().lower()
    return None


def _parse_tool_action_from_label(label: str | None) -> tuple[str | None, str | None]:
    if not label or "·" not in label:
        return (None, None)
    tool, action = label.split("·", 1)
    return (tool.strip().lower() or None, action.strip().lower() or None)


def _browser_action_display(action: str, args: dict[str, Any] | None = None) -> tuple[str, str]:
    args = args or {}
    action = (action or "").lower()
    if action == "navigate":
        return ("Tarayıcı · Aç", f"{_short_url(args.get('url'))} açılıyor")
    if action == "click":
        selector = args.get("selector")
        return ("Tarayıcı · Tıkla", f"{selector} tıklanıyor" if selector else "Sayfaya tıklanıyor")
    if action == "type":
        selector = args.get("selector")
        return (
            "Tarayıcı · Yaz",
            f"{selector} alanına yazılıyor" if selector else "Metin yazılıyor",
        )
    if action == "drag":
        return ("Tarayıcı · Sürükle", "Sayfada sürükleme yapılıyor")
    if action == "scroll":
        return ("Tarayıcı · Kaydır", "Sayfa kaydırılıyor")
    if action == "key":
        key = str(args.get("key") or "").strip()
        return ("Tarayıcı · Tuş", f"{key} tuşuna basılıyor" if key else "Tuş gönderiliyor")
    if action == "screenshot":
        return ("Tarayıcı · Görüntü Al", "Ekran görüntüsü alınıyor")
    if action == "back":
        return ("Tarayıcı · Geri", "Önceki sayfaya dönülüyor")
    if action == "forward":
        return ("Tarayıcı · İleri", "Sonraki sayfaya gidiliyor")
    if action == "new_tab":
        return ("Tarayıcı · Yeni Sekme", "Yeni sekme açılıyor")
    if action == "close_tab":
        return ("Tarayıcı · Sekmeyi Kapat", "Sekme kapatılıyor")
    if action == "get_text":
        return ("Tarayıcı · Metin Al", "Sayfadan metin okunuyor")
    if action == "get_html":
        return ("Tarayıcı · HTML Al", "Sayfa kaynağı alınıyor")
    return ("Tarayıcı · İşlem", "Tarayıcı işlemi hazırlanıyor")


def _computer_action_display(action: str, args: dict[str, Any] | None = None) -> tuple[str, str]:
    args = args or {}
    action = (action or "").lower()
    if action == "screenshot":
        return ("Masaüstü · Görüntü Al", "Masaüstü görüntüleniyor")
    if action == "click":
        return ("Masaüstü · Tıkla", "Masaüstünde tıklanıyor")
    if action == "double_click":
        return ("Masaüstü · Çift Tıkla", "Masaüstünde çift tıklanıyor")
    if action == "right_click":
        return ("Masaüstü · Sağ Tık", "Masaüstünde sağ tıklanıyor")
    if action == "move":
        return ("Masaüstü · İmleç Taşı", "İmleç masaüstünde taşınıyor")
    if action == "type":
        text = str(args.get("text") or "").strip()
        return ("Masaüstü · Yaz", f"{text!r} yazılıyor" if text else "Masaüstüne yazılıyor")
    if action == "key":
        text = str(args.get("text") or "").strip()
        return ("Masaüstü · Tuş", f"{text} tuşuna basılıyor" if text else "Tuş gönderiliyor")
    if action == "scroll":
        return ("Masaüstü · Kaydır", "Masaüstünde kaydırılıyor")
    if action == "drag":
        return ("Masaüstü · Sürükle", "Masaüstünde sürükleniyor")
    return ("Masaüstü · İşlem", "Masaüstü işlemi hazırlanıyor")


def _windows_desktop_action_display(action: str, args: dict[str, Any] | None = None) -> tuple[str, str]:
    args = args or {}
    action = (action or "").lower()
    target = str(
        args.get("app")
        or args.get("window_title")
        or args.get("profile_id")
        or args.get("control_name")
        or "Windows app"
    ).strip()
    if action == "launch_app":
        return ("Yerel Masaüstü · Uygulama Aç", f"{target} başlatılıyor")
    if action == "focus_window":
        return ("Yerel Masaüstü · Pencere Odakla", f"{target} öne getiriliyor")
    if action == "wait_for_window":
        return ("Yerel Masaüstü · Pencere Bekle", f"{target} bekleniyor")
    if action == "list_windows":
        return ("Yerel Masaüstü · Pencereleri Listele", "Görünür pencereler okunuyor")
    if action == "list_controls":
        return ("Yerel Masaüstü · Kontrolleri Listele", f"{target} kontrol ağacı okunuyor")
    if action == "invoke_control":
        return ("Yerel Masaüstü · Kontrol Çağır", f"{target} denetleniyor")
    if action == "set_text":
        return ("Yerel Masaüstü · Metin Ayarla", f"{target} alanı güncelleniyor")
    if action == "type_keys":
        return ("Yerel Masaüstü · Tuş Yaz", f"{target} için tuş gönderiliyor")
    if action == "read_window_text":
        return ("Yerel Masaüstü · Pencereyi Oku", f"{target} içeriği okunuyor")
    if action == "read_status":
        return ("Yerel Masaüstü · Durum Oku", f"{target} doğrulanıyor")
    return ("Yerel Masaüstü · İşlem", "Yerel masaüstü işlemi hazırlanıyor")


def _local_system_action_display(action: str, args: dict[str, Any] | None = None) -> tuple[str, str]:
    args = args or {}
    action = (action or "").lower()
    if action == "resolve_known_folder":
        return ("Yerel Sistem · Klasör Çöz", "Yerel klasör çözülüyor")
    if action == "open_path":
        target = str(args.get("path") or "").strip()
        return ("Yerel Sistem · Aç", f"{Path(target).name or 'Dosya'} arka planda açılıyor")
    if action == "launch_app":
        app_name = str(args.get("app") or "").strip()
        return ("Yerel Sistem · Uygulama Aç", f"{app_name or 'Uygulama'} başlatılıyor")
    return ("Yerel Sistem · İşlem", "Yerel işlem hazırlanıyor")


def _filesystem_action_display(action: str, args: dict[str, Any] | None = None) -> tuple[str, str]:
    args = args or {}
    action = (action or "").lower()
    target = str(args.get("path") or "").strip()
    target_name = Path(target).name or target or "öğe"
    if action == "list":
        return ("Dosya Sistemi · Listele", f"{target_name} içeriği taranıyor")
    if action == "read":
        return ("Dosya Sistemi · Oku", f"{target_name} okunuyor")
    if action == "exists":
        return ("Dosya Sistemi · Kontrol", f"{target_name} doğrulanıyor")
    if action == "mkdir":
        return ("Dosya Sistemi · Klasör Oluştur", f"{target_name} hazırlanıyor")
    if action == "write":
        return ("Dosya Sistemi · Yaz", f"{target_name} güncelleniyor")
    if action == "append":
        return ("Dosya Sistemi · Ekle", f"{target_name} genişletiliyor")
    if action == "copy":
        return ("Dosya Sistemi · Kopyala", f"{target_name} kopyalanıyor")
    if action == "move":
        return ("Dosya Sistemi · Taşı", f"{target_name} taşınıyor")
    if action == "delete":
        return ("Dosya Sistemi · Sil", f"{target_name} siliniyor")
    if action == "cwd":
        return ("Dosya Sistemi · Konum", "Çalışma konumu okunuyor")
    return ("Dosya Sistemi · İşlem", "Dosya sistemi işlemi hazırlanıyor")


def _tool_action_display(tool: str, action: str, args: dict[str, Any] | None = None) -> tuple[str, str]:
    if tool == "computer":
        return _computer_action_display(action, args)
    if tool == "windows_desktop":
        return _windows_desktop_action_display(action, args)
    if tool == "local_system":
        return _local_system_action_display(action, args)
    if tool == "filesystem":
        return _filesystem_action_display(action, args)
    return _browser_action_display(action, args)


def _tool_result_summary(event: dict[str, Any]) -> str:
    tool = str(event.get("tool", ""))
    metadata = event.get("metadata", {}) if isinstance(event.get("metadata"), dict) else {}
    action = _parse_action_from_label(str(metadata.get("frame_label") or ""))
    if tool == "browser":
        if not event.get("success"):
            return _trim_text(event.get("result") or event.get("error") or "Tarayıcı işlemi başarısız oldu.")
        mapping = {
            "navigate": "Sayfa açıldı",
            "click": "Tıklama tamamlandı",
            "type": "Yazı girişi tamamlandı",
            "drag": "Sürükleme tamamlandı",
            "scroll": "Kaydırma tamamlandı",
            "key": "Tuş gönderildi",
            "screenshot": "Ekran görüntüsü alındı",
            "back": "Geri dönüldü",
            "forward": "İleri gidildi",
            "new_tab": "Yeni sekme açıldı",
            "close_tab": "Sekme kapatıldı",
        }
        return mapping.get(action or "", _trim_text(event.get("result") or "Tarayıcı işlemi tamamlandı"))
    if tool == "computer":
        if not event.get("success"):
            return _trim_text(event.get("result") or event.get("error") or "Masaüstü işlemi başarısız oldu.")
        mapping = {
            "screenshot": "Masaüstü görüntüsü alındı",
            "click": "Masaüstü tıklaması tamamlandı",
            "double_click": "Çift tıklama tamamlandı",
            "right_click": "Sağ tıklama tamamlandı",
            "move": "İmleç taşındı",
            "type": "Yazı girişi tamamlandı",
            "key": "Tuş gönderildi",
            "scroll": "Kaydırma tamamlandı",
            "drag": "Sürükleme tamamlandı",
        }
        return mapping.get(action or "", _trim_text(event.get("result") or "Masaüstü işlemi tamamlandı"))
    if tool == "windows_desktop":
        if not event.get("success"):
            return _trim_text(event.get("result") or event.get("error") or "Yerel masaüstü işlemi başarısız oldu.")
        mapping = {
            "launch_app": "Uygulama başlatıldı",
            "focus_window": "Pencere odaklandı",
            "wait_for_window": "Pencere hazır",
            "list_windows": "Pencereler okundu",
            "list_controls": "Kontrol ağacı okundu",
            "invoke_control": "Kontrol çalıştırıldı",
            "set_text": "Metin güncellendi",
            "type_keys": "Tuşlar gönderildi",
            "read_window_text": "Pencere içeriği okundu",
            "read_status": "Durum doğrulandı",
        }
        return mapping.get(action or "", _trim_text(event.get("result") or "Yerel masaüstü işlemi tamamlandı"))
    if tool == "local_system":
        if not event.get("success"):
            return _trim_text(event.get("result") or event.get("error") or "Yerel işlem başarısız oldu.")
        mapping = {
            "resolve_known_folder": "Yerel klasör çözüldü",
            "open_path": "Yerel öğe açıldı",
            "launch_app": "Yerel uygulama başlatıldı",
        }
        return mapping.get(action or "", _trim_text(event.get("result") or "Yerel işlem tamamlandı"))
    if tool == "filesystem":
        if not event.get("success"):
            return _trim_text(event.get("result") or event.get("error") or "Dosya sistemi işlemi başarısız oldu.")
        mapping = {
            "list": "Klasör içeriği alındı",
            "read": "Dosya okundu",
            "exists": "Yol doğrulandı",
            "mkdir": "Klasör hazırlandı",
            "write": "Dosya yazıldı",
            "append": "Dosya güncellendi",
            "copy": "Dosya kopyalandı",
            "move": "Dosya taşındı",
            "delete": "Dosya silindi",
            "cwd": "Çalışma konumu okundu",
        }
        return mapping.get(action or "", _trim_text(event.get("result") or "Dosya sistemi işlemi tamamlandı"))
    if event.get("success"):
        return _trim_text(event.get("result") or f"{tool} işlemi tamamlandı")
    return _trim_text(event.get("result") or event.get("error") or f"{tool} işlemi başarısız oldu")


def _display_label_for_event(event: dict[str, Any]) -> str:
    event_type = str(event.get("type", "event"))
    if event_type in {"tool_call", "visual_intent"}:
        tool = str(event.get("tool", ""))
        args = event.get("args", {}) if isinstance(event.get("args"), dict) else {}
        if tool in {"browser", "computer", "windows_desktop", "local_system", "filesystem"}:
            return _tool_action_display(tool, str(args.get("action", "")), args)[0]
        return _generic_tool_label(tool)
    if event_type == "screenshot":
        return _display_label_for_frame(event)
    if event_type == "tool_result":
        return _generic_tool_label(str(event.get("tool", "Araç")))
    if event_type == "phase":
        return "Düşünüyor" if event.get("phase") == "thinking" else "Hazırlanıyor"
    if event_type == "thought":
        return "Düşünce Özeti"
    if event_type == "paused":
        if str(event.get("pause_kind", "")) == "sensitive_browser_takeover":
            return "Kontrol Sende"
        return "Duraklatıldı"
    if event_type == "resumed":
        return "Ajan Devam Ediyor"
    if event_type == "done":
        return "Görev Tamamlandı"
    if event_type == "error":
        return "Bir Hata Oluştu"
    return event_type.replace("_", " ").title()


def _display_summary_for_event(event: dict[str, Any]) -> str:
    event_type = str(event.get("type", "event"))
    if event_type in {"tool_call", "visual_intent"}:
        tool = str(event.get("tool", ""))
        args = event.get("args", {}) if isinstance(event.get("args"), dict) else {}
        if tool in {"browser", "computer", "windows_desktop", "local_system", "filesystem"}:
            return _tool_action_display(tool, str(args.get("action", "")), args)[1]
        return _trim_text(event.get("summary") or f"{tool} aracı hazırlanıyor")
    if event_type == "screenshot":
        return _display_summary_for_frame(event)
    if event_type == "tool_result":
        return _tool_result_summary(event)
    if event_type == "phase":
        return str(event.get("summary") or event.get("content") or "İşlem hazırlanıyor...")
    if event_type == "thought":
        return _trim_text(event.get("summary") or event.get("content") or "")
    if event_type in {"paused", "resumed"}:
        return _trim_text(event.get("summary") or event.get("content") or "")
    if event_type in {"done", "error"}:
        return _trim_text(event.get("content") or "")
    return _trim_text(event.get("summary") or event.get("content") or event.get("result") or "")


def _display_label_for_frame(frame: dict[str, Any]) -> str:
    tool, action = _parse_tool_action_from_label(str(frame.get("label") or frame.get("frame_label") or ""))
    if tool and action:
        return _tool_action_display(tool, action)[0]
    return "Görsel Kare"


def _display_summary_for_frame(frame: dict[str, Any]) -> str:
    tool, action = _parse_tool_action_from_label(str(frame.get("label") or frame.get("frame_label") or ""))
    if tool and action:
        return _tool_action_display(tool, action)[1]
    return _trim_text(frame.get("summary") or "Görsel güncelleme")


def _json_detail(value: Any) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False)


def _build_steps(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    current_tool_step: dict[str, Any] | None = None

    for index, event in enumerate(events):
        event_type = str(event.get("type", "event"))
        timestamp = str(event.get("timestamp", ""))

        if event_type == "phase":
            continue
        if event_type == "thought":
            current_tool_step = None
            steps.append(
                {
                    "id": f"step-{index + 1:03d}",
                    "kind": "thought",
                    "tone": "assistant",
                    "title": str(event.get("display_label") or "Düşünce Özeti"),
                    "summary": str(event.get("display_summary") or ""),
                    "status_label": "düşünüyor",
                    "timestamp": timestamp,
                }
            )
            continue
        if event_type == "tool_call":
            current_tool_step = {
                "id": f"step-{index + 1:03d}",
                "kind": "tool",
                "tone": "pending",
                "tool": str(event.get("tool", "")),
                "title": str(event.get("display_label") or _generic_tool_label(str(event.get("tool", "")))),
                "summary": str(event.get("display_summary") or ""),
                "status_label": "uyguluyor",
                "timestamp": timestamp,
                "detail": _json_detail(event.get("args", {})),
                "image_url": None,
            }
            steps.append(current_tool_step)
            continue
        if event_type == "visual_intent":
            if current_tool_step and current_tool_step.get("tool") == event.get("tool"):
                current_tool_step["status_label"] = "hazırlanıyor"
                current_tool_step["summary"] = str(event.get("display_summary") or current_tool_step["summary"])
                current_tool_step["focus_x"] = event.get("focus_x")
                current_tool_step["focus_y"] = event.get("focus_y")
            continue
        if event_type == "screenshot":
            if current_tool_step is not None:
                current_tool_step["image_url"] = event.get("image_url")
                current_tool_step["frame_id"] = event.get("frame_id")
            else:
                steps.append(
                    {
                        "id": f"step-{index + 1:03d}",
                        "kind": "visual",
                        "tone": "neutral",
                        "title": str(event.get("display_label") or "Görsel Kare"),
                        "summary": str(event.get("display_summary") or ""),
                        "status_label": "görsel",
                        "timestamp": timestamp,
                        "image_url": event.get("image_url"),
                    }
                )
            continue
        if event_type == "tool_result":
            if current_tool_step and current_tool_step.get("tool") == event.get("tool"):
                current_tool_step["tone"] = "success" if event.get("success") else "error"
                current_tool_step["status_label"] = "tamamlandı" if event.get("success") else "hata"
                current_tool_step["summary"] = str(event.get("display_summary") or current_tool_step["summary"])
                current_tool_step["detail"] = str(event.get("result") or "")
                current_tool_step["finished_at"] = timestamp
                current_tool_step = None
            else:
                steps.append(
                    {
                        "id": f"step-{index + 1:03d}",
                        "kind": "result",
                        "tone": "success" if event.get("success") else "error",
                        "title": str(event.get("display_label") or _generic_tool_label(str(event.get("tool", "")))),
                        "summary": str(event.get("display_summary") or ""),
                        "status_label": "tamamlandı" if event.get("success") else "hata",
                        "timestamp": timestamp,
                        "detail": str(event.get("result") or ""),
                    }
                )
            continue
        if event_type == "done":
            current_tool_step = None
            steps.append(
                {
                    "id": f"step-{index + 1:03d}",
                    "kind": "done",
                    "tone": "success",
                    "title": "Görev Tamamlandı",
                    "summary": str(event.get("content", "")),
                    "status_label": "tamamlandı",
                    "timestamp": timestamp,
                }
            )
            continue
        if event_type == "error":
            current_tool_step = None
            steps.append(
                {
                    "id": f"step-{index + 1:03d}",
                    "kind": "error",
                    "tone": "error",
                    "title": "Bir Hata Oluştu",
                    "summary": str(event.get("content", "")),
                    "status_label": "hata",
                    "timestamp": timestamp,
                }
            )

    return steps[-24:]


def create_live_app(
    base_config: AgentConfig,
    agent_factory: AgentFactory | None = None,
):
    """Create the FastAPI application for the live operator UI."""
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse

    log_path = configure_app_logging(base_config.workspace_dir)
    app = FastAPI(title="Agentra App")
    manager = ThreadManager(base_config=base_config, agent_factory=agent_factory)
    app.state.manager = manager
    app.state.log_path = log_path
    root_logger = logging.getLogger()
    app.state.log_handler = next(
        (handler for handler in root_logger.handlers if getattr(handler, "baseFilename", "") == str(log_path)),
        None,
    )
    logger.info(
        "Live app initialized workspace=%s provider=%s model=%s log_path=%s",
        base_config.workspace_dir,
        base_config.llm_provider,
        base_config.llm_model,
        log_path,
    )

    @app.on_event("shutdown")
    async def close_app_log_handler() -> None:
        handler = getattr(app.state, "log_handler", None)
        if handler is None:
            return
        root_logger.removeHandler(handler)
        handler.close()
        app.state.log_handler = None

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        boot = {
            "provider": base_config.llm_provider,
            "model": base_config.llm_model,
            "activeRunId": manager.active_run_id,
            "permissionMode": base_config.permission_mode,
        }
        return HTMLResponse(_render_app_html(boot))

    @app.post("/runs")
    async def create_run(request: RunCreateRequest) -> JSONResponse:
        try:
            snapshot = await manager.start_run(request)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Thread not found.") from exc
        return JSONResponse(snapshot)

    @app.get("/runs/{run_id}")
    async def get_run(run_id: str) -> JSONResponse:
        try:
            snapshot = manager.run_snapshot_for_http(run_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Run not found.") from exc
        return JSONResponse(snapshot)

    @app.post("/runs/{run_id}/stop")
    async def stop_run(run_id: str) -> JSONResponse:
        try:
            snapshot = await manager.stop_run(run_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Run not found.") from exc
        return JSONResponse(snapshot)

    @app.get("/runs/{run_id}/events")
    async def stream_events(run_id: str) -> StreamingResponse:
        try:
            session = manager.get_session(run_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Run not found.") from exc

        async def event_stream():
            yield _sse_payload({"kind": "snapshot", "snapshot": manager.snapshot_for_http(session)})
            if session.completed.is_set():
                yield _sse_payload({"kind": "complete", "status": session.status})
                return

            queue = manager.subscribe(run_id)
            try:
                while True:
                    try:
                        payload = await asyncio.wait_for(queue.get(), timeout=5.0)
                    except asyncio.TimeoutError:
                        if session.completed.is_set():
                            yield _sse_payload({"kind": "complete", "status": session.status})
                            return
                        yield ": keepalive\n\n"
                        continue
                    yield _sse_payload(payload)
                    if payload.get("kind") == "complete":
                        return
            finally:
                manager.unsubscribe(run_id, queue)

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/runs/{run_id}/assets/{asset_name}")
    async def get_asset(run_id: str, asset_name: str) -> FileResponse:
        artifacts = _resolve_run_artifacts(base_config.workspace_dir, run_id)
        if not artifacts:
            raise HTTPException(status_code=404, detail="Run not found.")
        path = artifacts["run_dir"] / "assets" / Path(asset_name).name
        if not path.exists():
            raise HTTPException(status_code=404, detail="Asset not found.")
        return FileResponse(path)

    @app.get("/runs/{run_id}/report")
    async def get_report(run_id: str) -> FileResponse:
        artifacts = _resolve_run_artifacts(base_config.workspace_dir, run_id)
        if not artifacts:
            raise HTTPException(status_code=404, detail="Run not found.")
        report_path = artifacts["report_path"]
        if not report_path.exists():
            raise HTTPException(status_code=404, detail="Report not found.")
        return FileResponse(report_path)

    @app.get("/runs/{run_id}/debug-images/{asset_path:path}")
    async def get_debug_image(run_id: str, asset_path: str) -> FileResponse:
        artifacts = _resolve_run_artifacts(base_config.workspace_dir, run_id)
        if not artifacts:
            raise HTTPException(status_code=404, detail="Run not found.")
        debug_root = (artifacts["run_dir"] / "debug-images").resolve()
        path = (debug_root / Path(asset_path)).resolve()
        if not path.is_file() or debug_root not in path.parents:
            raise HTTPException(status_code=404, detail="Debug image not found.")
        return FileResponse(path)

    @app.get("/logs", response_class=HTMLResponse)
    async def get_logs(run_id: str | None = None, thread_id: str | None = None, lines: int = 400) -> HTMLResponse:
        safe_lines = max(50, min(lines, 1500))
        return HTMLResponse(
            _render_logs_html(
                base_config.workspace_dir,
                app_log_path(base_config.workspace_dir),
                run_id=run_id,
                thread_id=thread_id,
                max_lines=safe_lines,
            )
        )

    @app.get("/threads")
    async def list_threads() -> JSONResponse:
        return JSONResponse({"threads": manager.list_threads_for_http()})

    @app.post("/threads/clear")
    async def clear_threads() -> JSONResponse:
        return JSONResponse(await manager.clear_all_threads())

    @app.get("/threads/{thread_id}")
    async def get_thread(thread_id: str) -> JSONResponse:
        try:
            thread = manager.get_thread(thread_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Thread not found.") from exc
        return JSONResponse(manager.thread_snapshot_for_http(thread))

    @app.patch("/threads/{thread_id}")
    async def update_thread(thread_id: str, payload: ThreadSettingsUpdateRequest) -> JSONResponse:
        try:
            snapshot = manager.update_thread_settings(
                thread_id,
                permission_mode=payload.permission_mode,
                desktop_execution_mode=payload.desktop_execution_mode,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Thread not found.") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return JSONResponse(snapshot)

    @app.get("/threads/{thread_id}/live-frame")
    async def get_live_frame(thread_id: str) -> Response:
        try:
            frame = await manager.capture_live_browser_frame(thread_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Thread not found.") from exc
        if not frame:
            return Response(status_code=204)
        return Response(
            content=frame.data,
            media_type=frame.media_type,
            headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"},
        )

    @app.get("/threads/{thread_id}/desktop-frame")
    async def get_desktop_frame(thread_id: str) -> Response:
        try:
            frame = await manager.capture_live_computer_frame(thread_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Thread not found.") from exc
        if not frame:
            return Response(status_code=204)
        return Response(
            content=frame.data,
            media_type=frame.media_type,
            headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"},
        )

    @app.get("/threads/{thread_id}/live-stream")
    async def get_live_stream(thread_id: str, request: FastAPIRequest) -> StreamingResponse:
        try:
            manager.get_thread(thread_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Thread not found.") from exc

        boundary = "agentraframe"
        single_frame_only = request.query_params.get("stream") == "1"

        async def frame_stream():
            try:
                while True:
                    if await request.is_disconnected():
                        return
                    frame = await manager.capture_live_browser_frame(thread_id)
                    if frame is None:
                        await asyncio.sleep(0.04)
                        continue
                    header = (
                        f"--{boundary}\r\n"
                        f"Content-Type: {frame.media_type}\r\n"
                        f"Content-Length: {len(frame.data)}\r\n\r\n"
                    ).encode("ascii")
                    yield header + frame.data + b"\r\n"
                    if single_frame_only:
                        return
                    await asyncio.sleep(0.01)
            except asyncio.CancelledError:
                return

        return StreamingResponse(
            frame_stream(),
            media_type=f"multipart/x-mixed-replace; boundary={boundary}",
            headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0", "X-Accel-Buffering": "no"},
        )

    @app.get("/threads/{thread_id}/desktop-stream")
    async def get_desktop_stream(thread_id: str, request: FastAPIRequest) -> StreamingResponse:
        try:
            manager.get_thread(thread_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Thread not found.") from exc

        boundary = "agentradesktop"
        single_frame_only = request.query_params.get("stream") == "1"

        async def frame_stream():
            try:
                while True:
                    if await request.is_disconnected():
                        return
                    frame = await manager.capture_live_computer_frame(thread_id)
                    if frame is None:
                        await asyncio.sleep(0.08)
                        continue
                    header = (
                        f"--{boundary}\r\n"
                        f"Content-Type: {frame.media_type}\r\n"
                        f"Content-Length: {len(frame.data)}\r\n\r\n"
                    ).encode("ascii")
                    yield header + frame.data + b"\r\n"
                    if single_frame_only:
                        return
                    await asyncio.sleep(0.04)
            except asyncio.CancelledError:
                return

        return StreamingResponse(
            frame_stream(),
            media_type=f"multipart/x-mixed-replace; boundary={boundary}",
            headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0", "X-Accel-Buffering": "no"},
        )

    @app.post("/threads/{thread_id}/pause")
    async def pause_thread(thread_id: str) -> JSONResponse:
        try:
            snapshot = await manager.pause_thread(thread_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Thread not found.") from exc
        return JSONResponse(snapshot)

    @app.post("/threads/{thread_id}/resume")
    async def resume_thread(thread_id: str) -> JSONResponse:
        try:
            snapshot = await manager.resume_thread(thread_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Thread not found.") from exc
        return JSONResponse(snapshot)

    @app.post("/threads/{thread_id}/approvals/{request_id}")
    async def respond_approval(thread_id: str, request_id: str, payload: ApprovalDecisionRequest) -> JSONResponse:
        try:
            snapshot = await manager.respond_to_approval(
                thread_id,
                request_id,
                approved=payload.approved,
                note=payload.note,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Approval request not found.") from exc
        return JSONResponse(snapshot)

    @app.post("/threads/{thread_id}/questions/{request_id}")
    async def answer_question(thread_id: str, request_id: str, payload: UserAnswerRequest) -> JSONResponse:
        try:
            snapshot = await manager.answer_question(thread_id, request_id, payload.answer)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Question request not found.") from exc
        return JSONResponse(snapshot)

    @app.post("/threads/{thread_id}/actions")
    async def human_action(thread_id: str, payload: HumanActionRequest) -> JSONResponse:
        try:
            snapshot = await manager.human_action(
                thread_id,
                payload.tool,
                payload.args,
                return_snapshot=payload.return_snapshot,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Thread not found.") from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return JSONResponse(snapshot)

    return app


def open_live_app(url: str) -> None:
    """Open the live app in the user's browser after the server starts."""
    threading.Timer(0.6, lambda: webbrowser.open(url, new=1)).start()


def _sse_payload(payload: EventPayload) -> str:
    return f"data: {json.dumps(payload)}\n\n"


def _resolve_run_artifacts(workspace_dir: Path, run_id: str | None) -> dict[str, Path] | None:
    if not run_id:
        return None
    candidates = [workspace_dir / ".runs" / run_id]
    candidates.extend(sorted((workspace_dir / ".threads").glob(f"*/workspace/.runs/{run_id}")))
    for run_dir in candidates:
        if run_dir.exists():
            return {
                "run_dir": run_dir,
                "events_path": run_dir / "events.json",
                "report_path": run_dir / "index.html",
            }
    return None


def _resolve_thread_artifacts(workspace_dir: Path, thread_id: str | None) -> dict[str, Path] | None:
    if not thread_id:
        return None
    thread_dir = workspace_dir / ".threads" / thread_id
    if not thread_dir.exists():
        return None
    return {
        "thread_dir": thread_dir,
        "ledger_path": thread_dir / "ledger.json",
        "audit_path": thread_dir / "audit.jsonl",
    }


def _latest_error_from_events(events_path: Path) -> dict[str, str] | None:
    if not events_path.exists():
        return None
    try:
        payload = json.loads(events_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    for event in reversed(payload.get("events", [])):
        if str(event.get("type", "")) != "error":
            continue
        details = event.get("details") if isinstance(event.get("details"), dict) else {}
        return {
            "timestamp": str(event.get("timestamp", "")),
            "message": str(event.get("content", "")),
            "exception_type": str(details.get("exception_type", "")),
            "traceback": str(details.get("traceback", "")),
            "hint": str(details.get("hint", "")),
        }
    return None


def _collect_run_debug_images(run_dir: Path, *, max_images: int = 48) -> list[Path]:
    debug_dir = run_dir / "debug-images"
    if not debug_dir.exists():
        return []
    candidates = [
        path
        for path in debug_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
    ]
    candidates.sort(key=lambda item: (item.stat().st_mtime, str(item)), reverse=True)
    return candidates[:max_images]


def _render_logs_html(
    workspace_dir: Path,
    log_path: Path,
    *,
    run_id: str | None = None,
    thread_id: str | None = None,
    max_lines: int = 400,
) -> str:
    run_artifacts = _resolve_run_artifacts(workspace_dir, run_id)
    thread_artifacts = _resolve_thread_artifacts(workspace_dir, thread_id)
    latest_error = _latest_error_from_events(run_artifacts["events_path"]) if run_artifacts else None
    debug_images = _collect_run_debug_images(run_artifacts["run_dir"]) if run_artifacts else []
    ledger_text = (
        thread_artifacts["ledger_path"].read_text(encoding="utf-8", errors="replace")
        if thread_artifacts and thread_artifacts["ledger_path"].exists()
        else ""
    )
    audit_text = (
        read_log_tail(thread_artifacts["audit_path"], max_lines=80)
        if thread_artifacts and thread_artifacts["audit_path"].exists()
        else ""
    )
    server_log_text = read_log_tail(log_path, max_lines=max_lines) or "No log lines have been written yet."

    latest_error_html = ""
    if latest_error:
        trace_html = ""
        hint_html = ""
        if latest_error["traceback"]:
            trace_html = "<h2>Traceback</h2>" f"<pre>{html.escape(latest_error['traceback'])}</pre>"
        if latest_error["hint"]:
            hint_html = (
                "<div class=\"meta accent\">Suggested fix: "
                f"{html.escape(latest_error['hint'])}</div>"
            )
        latest_error_html = (
            "<section class=\"card\">"
            "<h2>Latest Run Error</h2>"
            f"<div class=\"meta\">{html.escape(latest_error['timestamp'] or 'unknown time')}</div>"
            f"<pre>{html.escape(latest_error['message'] or 'Unknown error')}</pre>"
            f"{hint_html}"
            f"{trace_html}"
            "</section>"
        )

    run_paths_html = ""
    if run_artifacts:
        debug_dir = run_artifacts["run_dir"] / "debug-images"
        run_paths_html = (
            "<section class=\"card\">"
            "<h2>Run Artifacts</h2>"
            f"<div class=\"meta\">Run ID: {html.escape(run_id or '')}</div>"
            f"<div class=\"path\"><strong>Run Dir</strong> {html.escape(str(run_artifacts['run_dir']))}</div>"
            f"<div class=\"path\"><strong>Events</strong> {html.escape(str(run_artifacts['events_path']))}</div>"
            f"<div class=\"path\"><strong>Report</strong> {html.escape(str(run_artifacts['report_path']))}</div>"
            f"<div class=\"path\"><strong>Debug Images</strong> {html.escape(str(debug_dir))}</div>"
            "</section>"
        )

    thread_paths_html = ""
    if thread_artifacts:
        thread_paths_html = (
            "<section class=\"card\">"
            "<h2>Thread Artifacts</h2>"
            f"<div class=\"meta\">Thread ID: {html.escape(thread_id or '')}</div>"
            f"<div class=\"path\"><strong>Thread Dir</strong> {html.escape(str(thread_artifacts['thread_dir']))}</div>"
            f"<div class=\"path\"><strong>Ledger</strong> {html.escape(str(thread_artifacts['ledger_path']))}</div>"
            f"<div class=\"path\"><strong>Audit</strong> {html.escape(str(thread_artifacts['audit_path']))}</div>"
            "</section>"
        )

    ledger_html = ""
    if ledger_text:
        ledger_html = "<section class=\"card\"><h2>Thread Ledger</h2>" f"<pre>{html.escape(ledger_text)}</pre></section>"

    audit_html = ""
    if audit_text:
        audit_html = "<section class=\"card\"><h2>Thread Audit Tail</h2>" f"<pre>{html.escape(audit_text)}</pre></section>"

    debug_images_html = ""
    if run_artifacts:
        if debug_images:
            cards = []
            debug_root = run_artifacts["run_dir"] / "debug-images"
            for image_path in debug_images:
                asset_path = image_path.relative_to(debug_root).as_posix()
                image_url = f"/runs/{run_id}/debug-images/{asset_path}"
                rel_path = image_path.relative_to(run_artifacts["run_dir"]).as_posix()
                cards.append(
                    "<figure class=\"debug-image-card\">"
                    f"<img src=\"{html.escape(image_url)}\" alt=\"{html.escape(rel_path)}\" loading=\"lazy\" />"
                    f"<figcaption>{html.escape(rel_path)}</figcaption>"
                    "</figure>"
                )
            debug_images_html = (
                "<section class=\"card\">"
                "<h2>Debug Images</h2>"
                "<div class=\"meta\">Recent screenshots and live preview frames saved for debugging.</div>"
                f"<div class=\"debug-image-grid\">{''.join(cards)}</div>"
                "</section>"
            )
        else:
            debug_images_html = (
                "<section class=\"card\">"
                "<h2>Debug Images</h2>"
                "<div class=\"meta\">No debug images have been captured for this run yet.</div>"
                "</section>"
            )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <meta http-equiv="refresh" content="5" />
  <title>Agentra Logs</title>
  <style>
    :root {{
      --bg: #0b1424;
      --panel: #13213a;
      --line: rgba(167, 193, 235, 0.24);
      --text: #edf4ff;
      --muted: #9ab0d2;
      --accent: #78b8ff;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Aptos", "Segoe UI Variable", "Segoe UI", sans-serif;
      background:
        radial-gradient(circle at top left, rgba(81, 149, 255, 0.18), transparent 28%),
        linear-gradient(180deg, #09111f 0%, #0f1b31 100%);
      color: var(--text);
    }}
    .page {{
      max-width: 1240px;
      margin: 0 auto;
      padding: 28px 20px 40px;
      display: grid;
      gap: 18px;
    }}
    .hero {{
      display: grid;
      gap: 8px;
    }}
    h1, h2 {{
      margin: 0;
    }}
    h1 {{
      font-size: 28px;
    }}
    h2 {{
      font-size: 16px;
    }}
    .meta {{
      color: var(--muted);
      font-size: 13px;
    }}
    .grid {{
      display: grid;
      gap: 16px;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
    }}
    .card {{
      display: grid;
      gap: 10px;
      padding: 16px;
      border-radius: 18px;
      border: 1px solid var(--line);
      background: rgba(19, 33, 58, 0.94);
      box-shadow: 0 18px 42px rgba(0, 0, 0, 0.22);
    }}
    .path {{
      font-size: 13px;
      line-height: 1.5;
      word-break: break-all;
    }}
    pre {{
      margin: 0;
      padding: 14px;
      border-radius: 14px;
      border: 1px solid rgba(167, 193, 235, 0.16);
      background: rgba(4, 9, 18, 0.84);
      color: #e8f0ff;
      overflow: auto;
      white-space: pre-wrap;
      word-break: break-word;
      font-family: "Cascadia Code", "Consolas", monospace;
      font-size: 12px;
      line-height: 1.5;
    }}
    .accent {{
      color: var(--accent);
    }}
    .debug-image-grid {{
      display: grid;
      gap: 12px;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    }}
    .debug-image-card {{
      margin: 0;
      display: grid;
      gap: 8px;
      padding: 10px;
      border-radius: 14px;
      border: 1px solid rgba(167, 193, 235, 0.16);
      background: rgba(4, 9, 18, 0.42);
    }}
    .debug-image-card img {{
      width: 100%;
      border-radius: 10px;
      border: 1px solid rgba(167, 193, 235, 0.12);
      background: rgba(255, 255, 255, 0.04);
      aspect-ratio: 16 / 10;
      object-fit: contain;
    }}
    .debug-image-card figcaption {{
      color: var(--muted);
      font-size: 12px;
      word-break: break-all;
    }}
  </style>
</head>
<body>
  <main class="page">
    <section class="hero">
      <h1>Agentra Logs</h1>
      <div class="meta">Detailed app and thread diagnostics. This page auto-refreshes every 5 seconds.</div>
      <div class="meta accent">Server Log: {html.escape(str(log_path))}</div>
    </section>
    <section class="grid">
      {run_paths_html}
      {thread_paths_html}
      {latest_error_html}
    </section>
    {ledger_html}
    {audit_html}
    {debug_images_html}
    <section class="card">
      <h2>Server Log Tail</h2>
      <div class="meta">Showing the last {max_lines} lines from {html.escape(str(log_path.name))}</div>
      <pre>{html.escape(server_log_text)}</pre>
    </section>
  </main>
</body>
</html>"""


def _render_app_html(boot: dict[str, Any]) -> str:
    boot_json = json.dumps(boot).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="tr">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Agentra</title>
</head>
<body>
  <div id="app"></div>
  <script>window.__AGENTRA_BOOT__ = {boot_json};</script>
  <script>{_app_script()}</script>
  <style>{_app_styles()}</style>
</body>
</html>"""


def _app_styles() -> str:
    return _app_styles_base() + _app_styles_tv() + _app_styles_side()


def _app_script() -> str:
    return _app_script_state() + _app_script_render() + _app_script_mount()


def _app_styles_base() -> str:
    return """
:root {
  --text: #111827;
  --muted: rgba(229, 238, 255, 0.76);
  --line: rgba(255, 255, 255, 0.28);
  --panel: rgba(255, 255, 255, 0.16);
  --shadow: 0 22px 56px rgba(32, 66, 132, 0.14);
}
* { box-sizing: border-box; }
body {
  margin: 0;
  min-height: 100vh;
  color: var(--text);
  font-family: "Aptos", "Segoe UI Variable", "Segoe UI", sans-serif;
  background:
    radial-gradient(circle at 12% 14%, rgba(220, 243, 255, 0.95), transparent 22%),
    radial-gradient(circle at 19% 76%, rgba(214, 201, 255, 0.24), transparent 16%),
    radial-gradient(circle at 72% 32%, rgba(134, 208, 255, 0.18), transparent 20%),
    linear-gradient(90deg, #d6ebfb 0%, #bddcff 31%, #4d9af0 63%, #1660d6 100%);
}
button, input, textarea, summary, select { font: inherit; }
a { color: inherit; text-decoration: none; }
.page {
  min-height: 100vh;
  display: grid;
  grid-template-rows: 62px minmax(0, 1fr);
}
.topbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 14px;
  padding: 0 18px;
  background: rgba(255,255,255,0.96);
  border-bottom: 1px solid rgba(117, 148, 209, 0.14);
}
.topbar-left {
  display: flex;
  align-items: center;
  gap: 14px;
}
.topbar-actions {
  display: flex;
  align-items: center;
  gap: 10px;
}
.nav-button,
.stage-close {
  display: inline-grid;
  place-items: center;
  width: 34px;
  height: 34px;
  border: 0;
  border-radius: 999px;
  background: transparent;
  color: rgba(24, 36, 64, 0.74);
  font-size: 28px;
  line-height: 1;
  cursor: pointer;
}
.title-stack {
  display: grid;
  gap: 2px;
}
.app-title {
  font-size: 16px;
  font-weight: 700;
  color: #101828;
}
.app-subtitle {
  font-size: 12px;
  color: rgba(16, 24, 40, 0.64);
}
.workspace {
  display: grid;
  grid-template-columns: minmax(430px, 40%) minmax(0, 1fr);
  min-height: 0;
}
.left-pane {
  position: relative;
  border-right: 1px solid rgba(255, 255, 255, 0.26);
  background: linear-gradient(180deg, rgba(255,255,255,0.08), rgba(255,255,255,0.04));
  min-height: 0;
}
.left-pane::before {
  content: none;
}
.left-pane-inner {
  position: relative;
  z-index: 1;
  display: flex;
  flex-direction: column;
  gap: 12px;
  height: 100%;
  padding: 18px;
  overflow: auto;
}
.console-section {
  display: grid;
  gap: 10px;
  padding: 14px;
  border-radius: 18px;
  background: rgba(255, 255, 255, 0.16);
  border: 1px solid rgba(255, 255, 255, 0.34);
  box-shadow: 0 10px 32px rgba(34, 70, 148, 0.1);
  backdrop-filter: blur(14px);
}
.section-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
}
.section-title {
  font-size: 12px;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: rgba(18, 32, 61, 0.72);
}
.ghost-button {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-height: 30px;
  padding: 6px 10px;
  border-radius: 999px;
  border: 1px solid rgba(106, 130, 186, 0.22);
  background: rgba(255, 255, 255, 0.28);
  color: rgba(19, 34, 65, 0.82);
  cursor: pointer;
}
.console-section[hidden],
.advanced-panel[hidden],
[hidden] {
  display: none !important;
}
.field-label {
  font-size: 12px;
  font-weight: 600;
  color: rgba(19, 31, 57, 0.78);
}
.text-input,
.text-area,
.mini-input,
.json-input {
  width: 100%;
  border: 1px solid rgba(103, 132, 194, 0.22);
  border-radius: 12px;
  background: rgba(255, 255, 255, 0.42);
  color: #132241;
  padding: 10px 12px;
  outline: none;
}
.text-input::placeholder,
.text-area::placeholder,
.mini-input::placeholder,
.json-input::placeholder {
  color: rgba(19, 34, 65, 0.48);
}
.text-area {
  min-height: 92px;
  resize: vertical;
  line-height: 1.45;
}
.mini-input {
  min-height: 38px;
}
.json-input {
  min-height: 110px;
  resize: vertical;
  font-family: "Cascadia Code", "Consolas", monospace;
  font-size: 12px;
}
.action-row {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}
.action-button {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  min-height: 36px;
  padding: 8px 12px;
  border-radius: 11px;
  border: 1px solid rgba(106, 130, 186, 0.2);
  background: rgba(255, 255, 255, 0.34);
  color: #132241;
  cursor: pointer;
  transition: transform 120ms ease, background 120ms ease;
}
.action-button:hover:not(:disabled) {
  transform: translateY(-1px);
  background: rgba(255, 255, 255, 0.46);
}
.action-button:disabled {
  opacity: 0.45;
  cursor: not-allowed;
}
.action-button.primary {
  color: white;
  border-color: rgba(73, 106, 195, 0.6);
  background: linear-gradient(135deg, rgba(45, 86, 189, 0.96), rgba(82, 103, 240, 0.92));
}
.action-button.warn {
  color: #8a3145;
  background: rgba(255, 219, 226, 0.62);
}
.action-button.success {
  color: #136853;
  background: rgba(210, 255, 236, 0.74);
}
.inline-error {
  min-height: 15px;
  font-size: 12px;
  color: #a53a49;
}
.empty-state {
  padding: 10px 12px;
  border-radius: 12px;
  background: rgba(255, 255, 255, 0.2);
  border: 1px dashed rgba(112, 128, 187, 0.2);
  color: rgba(20, 33, 61, 0.65);
  font-size: 13px;
}
.thread-list,
.request-list,
.history-list,
.audit-list {
  display: grid;
  gap: 8px;
}
.audit-item {
  padding: 10px 12px;
  border-radius: 12px;
  border: 1px solid rgba(112, 128, 187, 0.2);
  background: rgba(255, 255, 255, 0.26);
  display: grid;
  gap: 6px;
}
.audit-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
}
.audit-type {
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: rgba(35, 55, 96, 0.75);
}
.audit-meta {
  font-size: 11px;
  color: rgba(35, 55, 96, 0.6);
}
.audit-detail {
  font-size: 13px;
  color: rgba(17, 32, 60, 0.84);
  line-height: 1.4;
}
.audit-chip {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  padding: 2px 8px;
  border-radius: 999px;
  border: 1px solid rgba(106, 130, 186, 0.3);
  background: rgba(255, 255, 255, 0.38);
  font-size: 11px;
  color: rgba(28, 44, 79, 0.76);
}
.thread-item {
  width: 100%;
  display: grid;
  gap: 6px;
  padding: 10px 12px;
  border-radius: 14px;
  border: 1px solid rgba(112, 128, 187, 0.2);
  background: rgba(255, 255, 255, 0.28);
  text-align: left;
  cursor: pointer;
}
.thread-item.active {
  border-color: rgba(69, 112, 224, 0.52);
  background: rgba(235, 244, 255, 0.66);
  box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.44);
}
.thread-item-head,
.history-item-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
}
.thread-title,
.request-title {
  font-size: 14px;
  font-weight: 700;
  color: #11203c;
}
.thread-summary,
.history-summary,
.request-text,
.helper-text {
  font-size: 13px;
  line-height: 1.45;
  color: rgba(17, 32, 60, 0.78);
}
.badge-row {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}
.detail-grid {
  display: grid;
  gap: 8px;
}
.detail-row {
  display: grid;
  gap: 4px;
}
.detail-key {
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: rgba(20, 33, 61, 0.56);
}
.detail-value {
  font-size: 12px;
  color: rgba(17, 32, 60, 0.84);
}
.path-copy {
  font-size: 12px;
  line-height: 1.4;
  font-family: "Cascadia Code", "Consolas", monospace;
  color: rgba(17, 32, 60, 0.82);
  word-break: break-all;
}
.request-card,
.history-item,
.thread-detail-card {
  display: grid;
  gap: 8px;
  padding: 10px 12px;
  border-radius: 14px;
  border: 1px solid rgba(112, 128, 187, 0.2);
  background: rgba(255, 255, 255, 0.28);
}
.request-meta,
.history-meta {
  font-size: 12px;
  color: rgba(17, 32, 60, 0.6);
}
.inline-form,
.manual-grid {
  display: grid;
  gap: 8px;
}
.inline-form {
  grid-template-columns: minmax(0, 1fr) auto;
}
.manual-row {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 8px;
}
.details-shell {
  display: grid;
  gap: 8px;
  padding-top: 4px;
}
details > summary {
  cursor: pointer;
  color: rgba(20, 33, 61, 0.72);
  font-size: 13px;
  font-weight: 600;
}
.right-pane {
  position: relative;
  padding: 18px 24px 18px 20px;
}
.right-pane::before {
  content: "";
  position: absolute;
  inset: 0;
  background:
    radial-gradient(circle at 26% 14%, rgba(255,255,255,0.16), transparent 14%),
    linear-gradient(135deg, rgba(102, 176, 255, 0.12), rgba(23, 92, 221, 0.12));
  pointer-events: none;
}
.stage-shell {
  position: relative;
  z-index: 1;
  display: grid;
  grid-template-rows: auto 1fr;
  gap: 12px;
  height: 100%;
}
.stage-head {
  display: flex;
  align-items: center;
  justify-content: flex-start;
  color: rgba(255,255,255,0.92);
  padding: 2px 4px 0 4px;
}
.stage-title {
  font-size: 15px;
  font-weight: 700;
}
.tv-stage {
  min-height: 0;
  display: grid;
  place-items: start center;
  padding: 10px 4px 8px;
}
.muted { color: var(--muted); }
@media (max-width: 1080px) {
  .workspace { grid-template-columns: 1fr; }
  .left-pane { border-right: 0; border-bottom: 1px solid rgba(255,255,255,0.28); }
  .tv-stage { padding-top: 0; }
  .left-pane-inner { max-height: 50vh; }
}
@media (max-width: 760px) {
  .manual-row,
  .inline-form {
    grid-template-columns: 1fr;
  }
}
"""


def _app_styles_tv() -> str:
    return """
.tv-shell {
  width: min(100%, 920px);
  background: linear-gradient(135deg, rgba(255,255,255,0.12), rgba(151, 112, 255, 0.18));
  border: 1px solid rgba(255,255,255,0.22);
  border-radius: 26px;
  padding: 0 0 14px;
  box-shadow: 0 26px 56px rgba(19, 43, 108, 0.18);
  backdrop-filter: blur(16px);
}
.tv-screen {
  position: relative;
  aspect-ratio: 16 / 9;
  overflow: hidden;
  border-radius: 26px 26px 20px 20px;
  background:
    linear-gradient(180deg, rgba(255,255,255,0.96), rgba(242,246,252,0.94) 7%, rgba(29,41,69,0.98) 7%, rgba(29,41,69,0.98) 100%);
}
.tv-screen:focus-visible {
  outline: 2px solid rgba(255,255,255,0.92);
  outline-offset: 2px;
}
.tv-screen.busy::after {
  content: "";
  position: absolute;
  inset: 42px 0 0;
  background:
    linear-gradient(180deg, rgba(255,255,255,0.0), rgba(255,255,255,0.08), rgba(255,255,255,0.0)),
    repeating-linear-gradient(
      180deg,
      rgba(255,255,255,0.02) 0px,
      rgba(255,255,255,0.02) 18px,
      rgba(255,255,255,0.0) 18px,
      rgba(255,255,255,0.0) 36px
    );
  animation: scan 2.6s linear infinite;
  pointer-events: none;
}
@keyframes scan {
  0% { transform: translateY(-40%); opacity: 0.35; }
  100% { transform: translateY(120%); opacity: 0.15; }
}
.tv-title {
  text-align: center;
  color: #6d7b90;
  font-size: 12px;
  padding: 8px 0 12px;
}
.tv-image {
  width: 100%;
  height: calc(100% - 42px);
  object-fit: contain;
  display: none;
  border-bottom-left-radius: 18px;
  border-bottom-right-radius: 18px;
  user-select: none;
  touch-action: none;
  will-change: opacity, transform;
  transition: opacity 180ms ease, transform 220ms ease;
}
.tv-image.swapping {
  opacity: 0.28;
  transform: scale(0.992);
}
.tv-empty {
  position: absolute;
  inset: 56px 24px 24px;
  display: grid;
  place-items: center;
  text-align: center;
  color: #d6e2fb;
  font-size: 16px;
  line-height: 1.5;
  text-shadow: 0 1px 12px rgba(7, 13, 26, 0.28);
}
.tv-empty .muted {
  color: rgba(221, 232, 255, 0.82);
}
.cursor-dot {
  position: absolute;
  width: 16px;
  height: 16px;
  border-radius: 999px;
  background: white;
  border: 3px solid #151d34;
  box-shadow: 0 0 0 5px rgba(255,255,255,0.14), 0 10px 24px rgba(0, 0, 0, 0.28);
  transform: translate(-50%, -50%);
  display: none;
  transition: left 220ms ease, top 220ms ease, opacity 180ms ease;
}
.cursor-dot.pending {
  animation: pulse 1.25s ease-in-out infinite;
}
@keyframes pulse {
  0%, 100% { box-shadow: 0 0 0 5px rgba(255,255,255,0.12), 0 10px 24px rgba(0,0,0,0.28); }
  50% { box-shadow: 0 0 0 9px rgba(255,255,255,0.22), 0 10px 24px rgba(0,0,0,0.22); }
}
.cursor-bubble {
  position: absolute;
  max-width: min(360px, 58%);
  padding: 13px 15px;
  border-radius: 18px;
  background: rgba(255,255,255,0.94);
  color: #172033;
  line-height: 1.45;
  box-shadow: 0 18px 38px rgba(0,0,0,0.18);
  transform: translate(-14%, 18px);
  display: none;
  transition: left 220ms ease, top 220ms ease, opacity 180ms ease;
}
.cursor-bubble.flip {
  transform: translate(-14%, calc(-100% - 24px));
}
.tv-footer {
  display: grid;
  gap: 0;
  padding: 8px 12px 0;
}
.manual-dock {
  display: grid;
  gap: 10px;
  padding: 12px 14px 0;
}
.takeover-banner {
  display: grid;
  gap: 6px;
  padding: 12px 14px;
  border-radius: 18px;
  background: linear-gradient(135deg, rgba(255, 245, 196, 0.22), rgba(255, 193, 92, 0.16));
  border: 1px solid rgba(255, 222, 146, 0.34);
  box-shadow: 0 14px 30px rgba(20, 24, 35, 0.16);
}
.takeover-banner-title {
  font-size: 12px;
  font-weight: 800;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: rgba(255, 248, 219, 0.96);
}
.takeover-banner-text {
  font-size: 13px;
  line-height: 1.5;
  color: rgba(255, 247, 231, 0.94);
}
.manual-dock-head {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  align-items: center;
  flex-wrap: wrap;
}
.manual-dock-copy {
  display: grid;
  gap: 4px;
}
.manual-dock-title {
  font-size: 12px;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: rgba(255,255,255,0.86);
}
.manual-dock-text {
  font-size: 13px;
  line-height: 1.45;
  color: rgba(226, 236, 255, 0.88);
}
.interact-button.active {
  background: rgba(255,255,255,0.92);
  color: #16203a;
  border-color: rgba(255,255,255,0.92);
  box-shadow: 0 8px 18px rgba(255,255,255,0.18);
}
.tv-screen.manual-interact .tv-image {
  cursor: crosshair;
}
.manual-inline-note {
  font-size: 12px;
  color: rgba(226, 236, 255, 0.82);
}
.manual-inline-note strong {
  color: rgba(255,255,255,0.96);
}
.manual-dock .inline-error {
  min-height: 18px;
}
.scrub-row {
  display: grid;
  grid-template-columns: auto 1fr auto auto;
  gap: 8px;
  align-items: center;
  color: white;
}
.scrub-meta {
  font-size: 12px;
  font-weight: 400;
  color: rgba(255,255,255,0.92);
}
.timeline-track {
  position: relative;
  height: 8px;
  border-radius: 999px;
  cursor: pointer;
  background: linear-gradient(180deg, rgba(255,255,255,0.38), rgba(255,255,255,0.14));
  border: 1px solid rgba(255,255,255,0.58);
  box-shadow:
    inset 0 1px 0 rgba(255,255,255,0.42),
    0 0 26px rgba(255,255,255,0.18),
    0 8px 24px rgba(0,0,0,0.16);
}
.timeline-track::after {
  content: "";
  position: absolute;
  inset: 1px;
  border-radius: 999px;
  background: linear-gradient(180deg, rgba(255,255,255,0.44), rgba(255,255,255,0.04));
  opacity: 0.72;
}
.timeline-progress {
  position: absolute;
  left: 1px;
  top: 1px;
  bottom: 1px;
  width: 0%;
  border-radius: 999px;
  background: linear-gradient(90deg, rgba(255,255,255,0.98), rgba(221,239,255,0.98));
  box-shadow: 0 0 18px rgba(255,255,255,0.42), 0 0 34px rgba(190,225,255,0.24);
  z-index: 1;
}
.timeline-handle {
  position: absolute;
  top: 50%;
  left: 0%;
  width: 16px;
  height: 16px;
  border-radius: 999px;
  border: 1px solid rgba(255,255,255,0.98);
  background: radial-gradient(circle at 35% 35%, #ffffff 0%, #f4faff 45%, #d8e9ff 100%);
  box-shadow:
    0 0 0 2px rgba(255,255,255,0.14),
    0 0 12px rgba(255,255,255,0.4),
    0 5px 14px rgba(0, 0, 0, 0.14);
  transform: translate(-50%, -50%);
  pointer-events: none;
  z-index: 2;
}
"""


def _app_styles_side() -> str:
    return """
.status-pill,
.thread-chip,
.count-badge {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-width: 64px;
  max-width: 100%;
  padding: 4px 8px;
  border-radius: 999px;
  font-weight: 700;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  font-size: 10px;
  line-height: 1;
  white-space: nowrap;
}
.status-pill.idle,
.status-pill.partial,
.thread-chip.idle,
.thread-chip.partial { color: #537ab7; background: rgba(130, 176, 255, 0.18); }
.status-pill.running,
.thread-chip.running,
.thread-chip.blocked_waiting_user { color: #9f7722; background: rgba(255, 213, 110, 0.22); }
.status-pill.completed,
.thread-chip.completed { color: #1f8e73; background: rgba(101, 220, 176, 0.22); }
.status-pill.error,
.thread-chip.error { color: #b9485b; background: rgba(255, 154, 166, 0.24); }
.thread-chip.paused_for_user { color: #6644ba; background: rgba(212, 194, 255, 0.3); }
.count-badge {
  min-width: 0;
  color: rgba(20, 32, 58, 0.72);
  background: rgba(255,255,255,0.42);
  border: 1px solid rgba(112, 128, 187, 0.16);
}
"""


def _app_script_state() -> str:
    return """
const boot = window.__AGENTRA_BOOT__ || {};
const DEFAULT_FOCUS = { x: 0.74, y: 0.2 };

const state = {
  runId: null,
  selectedRunId: null,
  streamRunId: null,
  activeThreadId: null,
  activeThread: null,
  threads: [],
  goal: "",
  provider: boot.provider || "",
  model: boot.model || "",
  status: "idle",
  events: [],
  frames: [],
  steps: [],
  audit: [],
  browser: {
    active: false,
    active_url: "",
    active_title: "",
    tab_count: 0,
    identity: "isolated",
    profile_name: "Default",
    last_error: "",
  },
  liveMode: true,
  historyMode: false,
  selectedFrameId: null,
  reportUrl: null,
  source: null,
  liveFrameUrl: null,
  liveFrameObjectUrl: null,
  liveFrameSignature: null,
  liveFrameThreadId: null,
  liveFrameSource: "browser",
  liveFrameMode: "stream",
  liveFrameLoopToken: 0,
  liveFrameStamp: 0,
  liveFrameTimer: null,
  liveFrameLoadedAt: 0,
  liveFrameErrorCount: 0,
  liveFrameAbortController: null,
  liveFramePumpActive: false,
  threadPollTimer: null,
  refreshInFlight: false,
  scrubPercent: 1,
  scrubDrag: null,
  initialRunId: boot.activeRunId || null,
  defaultPermissionMode: boot.permissionMode || "default",
  errors: {
    run: "",
    thread: "",
    approval: "",
    question: "",
    manual: "",
  },
  drafts: {
    goal: "",
    threadTitle: "",
    permissionMode: "",
    questionAnswers: {},
    scrollAmount: "600",
    clickSelector: "",
    typeSelector: "",
    typeText: "",
    advancedTool: "browser",
    advancedArgs: "{\\n  \\"action\\": \\"back\\"\\n}",
  },
  activity: {
    mode: "idle",
    summary: "",
    focus_x: DEFAULT_FOCUS.x,
    focus_y: DEFAULT_FOCUS.y,
  },
  ui: {
    showAdvanced: false,
    manualPending: false,
    manualQueue: null,
    pendingApprovalIds: {},
    pendingQuestionIds: {},
    threadDetailLoading: false,
    threadRefreshRetrying: false,
    activeThreadDetailKey: "",
    queuedThreadRefresh: null,
    preferredThreadId: null,
    preferredRunId: null,
    preferredLive: false,
    controlLayer: "browser",
    manualLayerOverride: null,
    autoFollowLayer: true,
    interactActive: false,
    interactBuffer: "",
    interactFlushTimer: null,
    interactDrag: null,
    interactSuppressClickUntil: 0,
  },
};

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function buildAppUrl(path) {
  const base = window.__agentraBaseUrl || window.location.origin || "";
  try {
    return new URL(path, base || "http://127.0.0.1").toString();
  } catch (error) {
    return path;
  }
}

function shortText(value, limit = 220) {
  const text = String(value ?? "").replace(/\\s+/g, " ").trim();
  if (text.length <= limit) return text;
  return `${text.slice(0, limit - 3).trimEnd()}...`;
}

function statusLabel(status) {
  if (status === "running") return "ÇALIŞIYOR";
  if (status === "completed") return "TAMAMLANDI";
  if (status === "error") return "HATA";
  if (status === "partial") return "KISMİ";
  return "HAZIR";
}

function threadStatusLabel(status) {
  if (status === "running") return "ÇALIŞIYOR";
  if (status === "paused_for_user") return "DEVREDİLDİ";
  if (status === "blocked_waiting_user") return "BEKLİYOR";
  if (status === "completed") return "TAMAMLANDI";
  if (status === "error") return "HATA";
  return "HAZIR";
}

function formatDateTime(value) {
  if (!value) return "—";
  try {
    return new Intl.DateTimeFormat("tr-TR", {
      day: "2-digit",
      month: "short",
      hour: "2-digit",
      minute: "2-digit",
    }).format(new Date(value));
  } catch (error) {
    return String(value);
  }
}

function projectStamp() {
  return "";
}

function shortPath(value) {
  const text = String(value || "");
  if (!text) return "—";
  const normalized = text.replaceAll("\\\\", "/");
  const parts = normalized.split("/").filter(Boolean);
  return parts.slice(-4).join("/");
}

function permissionModeLabel(mode) {
  return String(mode || "").toLowerCase() === "full" ? "Full Yetki" : "Default";
}

function permissionModeSummary(mode) {
  if (String(mode || "").toLowerCase() === "full") {
    return "Gercek Chrome profili, kurulu uygulamalar ve esnek araclar acik. Parola ve benzeri hassas browser adimlarinda kontrol sana devredilir; geri alinmaz islemler yine onay ister.";
  }
  return "Izole tarayici, standart guvenlik kapilari ve daha sinirli ortam erisimi kullanilir.";
}

function browserIdentityLabel(identity, profileName) {
  if (String(identity || "").toLowerCase() === "chrome_profile") {
    const profile = String(profileName || "Default").trim();
    return profile ? `Chrome Profili · ${profile}` : "Chrome Profili";
  }
  return "Izole Tarayici";
}

function activityVisibilityLabel(activity) {
  const visibility = String(activity?.visibility || "").toLowerCase();
  if (visibility === "visible") return "Gorunur";
  if (visibility === "hidden") return "Arka planda";
  if (visibility === "background") return "Arka planda";
  return "Hazir";
}

function browserStatusLabel() {
  const active = Boolean(state.browser?.active);
  const tabCount = Number(state.browser?.tab_count ?? 0);
  const identity = browserIdentityLabel(state.browser?.identity, state.browser?.profile_name);
  return `${active ? "AKTIF" : "KAPALI"} · ${tabCount} sekme · ${identity}`;
}

function updateBrowserState(payload) {
  if (!payload) return;
  const hasDirectFields = Object.prototype.hasOwnProperty.call(payload, "browser_session_active")
    || Object.prototype.hasOwnProperty.call(payload, "active_url")
    || Object.prototype.hasOwnProperty.call(payload, "active_title")
    || Object.prototype.hasOwnProperty.call(payload, "tab_count")
    || Object.prototype.hasOwnProperty.call(payload, "browser_identity")
    || Object.prototype.hasOwnProperty.call(payload, "browser_profile_name")
    || Object.prototype.hasOwnProperty.call(payload, "browser_last_error");
  const browserPayload = payload.browser || {};
  const hasBrowser = Object.keys(browserPayload || {}).length > 0;
  if (!hasDirectFields && !hasBrowser) return;
  state.browser = {
    active: Boolean(payload.browser_session_active ?? browserPayload.active ?? false),
    active_url: String(payload.active_url ?? browserPayload.active_url ?? ""),
    active_title: String(payload.active_title ?? browserPayload.active_title ?? ""),
    tab_count: Number(payload.tab_count ?? browserPayload.tab_count ?? 0) || 0,
    identity: String(payload.browser_identity ?? browserPayload.identity ?? "isolated"),
    profile_name: String(payload.browser_profile_name ?? browserPayload.profile_name ?? "Default"),
    last_error: String(payload.browser_last_error ?? browserPayload.last_error ?? ""),
  };
}

function pendingApprovals(thread) {
  return (thread?.approval_requests || []).filter((item) => !item.status || item.status === "pending");
}

function pendingQuestions(thread) {
  return (thread?.question_requests || []).filter((item) => !item.status || item.status === "pending");
}

function latestControlEvent(thread) {
  const events = activeRunDetailForThread(thread)?.events || [];
  for (let index = events.length - 1; index >= 0; index -= 1) {
    const type = String(events[index]?.type || "");
    if (type === "paused" || type === "resumed") return events[index];
  }
  return null;
}

function pendingTakeoverRequest(thread) {
  if (!thread || thread.status !== "paused_for_user" || thread.handoff_state !== "user") return null;
  const event = latestControlEvent(thread);
  if (!event || String(event.type || "") !== "paused") return null;
  const pauseKind = String(event.pause_kind || "");
  if (pauseKind !== "sensitive_browser_takeover" && pauseKind !== "desktop_control_takeover") return null;
  return event;
}

function activeRunForThread(thread) {
  return thread?.active_run || thread?.active_run_summary || null;
}

function activeRunDetailForThread(thread) {
  return thread?.active_run || null;
}

function runSummaries(thread) {
  return Array.isArray(thread?.runs) ? [...thread.runs] : [];
}

function canManualControl(thread) {
  if (!thread || thread.restart_required || !activeRunForThread(thread)) return false;
  return thread.status === "paused_for_user" || thread.status === "blocked_waiting_user";
}

function canDirectPreviewControl() {
  const thread = state.activeThread;
  const activeRun = activeRunForThread(thread);
  if (!canManualControl(thread) || !activeRun || !state.selectedRunId) return false;
  if (state.historyMode || !state.liveMode) return false;
  return activeRun.run_id === state.selectedRunId;
}

function canStartInteract() {
  const thread = state.activeThread;
  const activeRun = activeRunForThread(thread);
  if (!thread || !activeRun || !state.selectedRunId) return false;
  if (thread.restart_required) return false;
  if (state.historyMode || !state.liveMode) return false;
  if (activeRun.run_id !== state.selectedRunId) return false;
  return thread.status === "running" || canManualControl(thread);
}

function threadSummarySignature(thread) {
  if (!thread) return "";
  const activeRun = activeRunForThread(thread);
  const approvals = pendingApprovals(thread).map((item) => `${item.request_id}:${item.status || "pending"}`).join("|");
  const questions = pendingQuestions(thread).map((item) => `${item.request_id}:${item.status || "pending"}`).join("|");
  const browserPayload = thread.browser || {};
  return JSON.stringify({
    thread_id: thread.thread_id || "",
    current_run_id: thread.current_run_id || "",
    status: thread.status || "",
    handoff_state: thread.handoff_state || "",
    restart_required: Boolean(thread.restart_required),
    recovered_from_disk: Boolean(thread.recovered_from_disk),
    activity_summary: thread.activity_summary || "",
    active_run_status: activeRun?.status || "",
    approvals,
    questions,
    active_url: thread.active_url || browserPayload.active_url || "",
    active_title: thread.active_title || browserPayload.active_title || "",
  });
}

function mergeThreadSummary(existingThread, summaryThread) {
  if (!summaryThread) return existingThread;
  const existingActiveRun = existingThread?.active_run || null;
  const merged = {
    ...(existingThread || {}),
    ...summaryThread,
  };
  if (existingActiveRun && existingActiveRun.run_id === summaryThread.current_run_id) {
    merged.active_run = {
      ...existingActiveRun,
      thread_status: summaryThread.status || existingActiveRun.thread_status,
      handoff_state: summaryThread.handoff_state || existingActiveRun.handoff_state,
      activity: summaryThread.activity || existingActiveRun.activity,
    };
  } else if (summaryThread.active_run) {
    merged.active_run = summaryThread.active_run;
  } else if (Object.prototype.hasOwnProperty.call(merged, "active_run")) {
    delete merged.active_run;
  }
  return merged;
}

function selectedRunIdForThread(thread, options = {}) {
  const runs = runSummaries(thread);
  if (!runs.length) return null;
  const preferredRunId = options.preferRunId || state.selectedRunId;
  if (preferredRunId && runs.some((run) => run.run_id === preferredRunId)) return preferredRunId;
  const activeRun = activeRunForThread(thread);
  if (options.preferLive && activeRun?.run_id) return activeRun.run_id;
  if (thread?.current_run_id && runs.some((run) => run.run_id === thread.current_run_id)) return thread.current_run_id;
  return runs[runs.length - 1].run_id;
}

function mergeRefreshOptions(base = {}, override = {}) {
  const merged = { ...(base || {}) };
  if (override.forceThreadId) merged.forceThreadId = override.forceThreadId;
  if (override.preferRunId) merged.preferRunId = override.preferRunId;
  if (Object.prototype.hasOwnProperty.call(override, "preferLive")) {
    merged.preferLive = Boolean(override.preferLive);
  }
  if (override.forceDetail) merged.forceDetail = true;
  return merged;
}

function threadFocusPreference() {
  if (!state.ui.preferredThreadId) return {};
  return {
    forceThreadId: state.ui.preferredThreadId,
    preferRunId: state.ui.preferredRunId,
    preferLive: Boolean(state.ui.preferredLive),
  };
}

function rememberThreadFocusPreference(options = {}) {
  if (!options.forceThreadId) return;
  state.ui.preferredThreadId = options.forceThreadId;
  state.ui.preferredRunId = options.preferRunId || null;
  state.ui.preferredLive = Boolean(options.preferLive);
}

function clearThreadFocusPreference() {
  state.ui.preferredThreadId = null;
  state.ui.preferredRunId = null;
  state.ui.preferredLive = false;
}

function settleThreadFocusPreference(threadId, runId = null) {
  if (!state.ui.preferredThreadId || state.ui.preferredThreadId !== threadId) return;
  if (state.ui.preferredRunId && runId && state.ui.preferredRunId !== runId) return;
  if (state.ui.preferredRunId && !runId) return;
  clearThreadFocusPreference();
}

function applyThreadSummary(summaryThread, options = {}) {
  if (!summaryThread) return;
  const nextRunId = selectedRunIdForThread(summaryThread, options);
  const threadChanged = state.activeThreadId !== summaryThread.thread_id;
  const runChanged = nextRunId !== state.selectedRunId;
  if (threadChanged || runChanged) {
    clearRunState();
  }
  state.activeThreadId = summaryThread.thread_id;
  state.activeThread = mergeThreadSummary(
    !threadChanged ? state.activeThread : null,
    summaryThread,
  );
  state.selectedRunId = nextRunId;
  state.runId = nextRunId;
  if (!nextRunId) {
    state.liveMode = true;
    state.historyMode = false;
  } else if (options.preferLive && summaryThread.current_run_id === nextRunId) {
    state.liveMode = true;
    state.historyMode = false;
  }
  updateBrowserState(state.activeThread);
  settleThreadFocusPreference(summaryThread.thread_id, nextRunId);
}

function shouldRefreshThreadDetail(summaryThread, options = {}) {
  if (!summaryThread || state.historyMode) return false;
  if (options.forceDetail) return true;
  if (!state.activeThread || state.activeThread.thread_id !== summaryThread.thread_id) return true;
  if (!activeRunDetailForThread(state.activeThread) && summaryThread.current_run_id) return true;
  return threadSummarySignature(summaryThread) !== state.ui.activeThreadDetailKey;
}

function interactReady() {
  return Boolean(state.ui.interactActive) && canDirectPreviewControl();
}

function clearInteractFlushTimer() {
  if (state.ui.interactFlushTimer !== null) {
    window.clearTimeout(state.ui.interactFlushTimer);
    state.ui.interactFlushTimer = null;
  }
}

function resetInteractState() {
  clearInteractFlushTimer();
  state.ui.interactActive = false;
  state.ui.interactBuffer = "";
  state.ui.interactDrag = null;
  state.ui.interactSuppressClickUntil = 0;
}

function ensureInteractState() {
  if (!state.ui.interactActive && !state.ui.interactBuffer) return;
  if (state.status === "completed" || state.status === "error" || !canDirectPreviewControl()) {
    resetInteractState();
  }
}

function interactHint() {
  if (!state.activeThread) return "Bir thread seç ya da yeni run başlat.";
  if (state.historyMode || !state.liveMode) return "Interact yalnızca aktif canlı run üzerinde kullanılabilir.";
  const activeRun = activeRunForThread(state.activeThread);
  if (!activeRun || activeRun.run_id !== state.selectedRunId) return "Önce aktif run'a dön.";
  if (pendingTakeoverRequest(state.activeThread)) {
    return "Kontrol sende. Tarayicida hassas adimi manuel tamamla, sonra Finish Control ile ajani devam ettir.";
  }
  const layerLabel = currentControlLayerLabel();
  if (state.activeThread.status === "running") return `Interact ajanı pause eder, sonra ${layerLabel.toLowerCase()} katmanında tıklayıp sürükleyebilir, yazabilir ve scroll yapabilirsin.`;
  if (interactReady()) return `Interact açık. ${layerLabel} canlı görüntüsüne odaklanıp tıklayabilir, sürükleyebilir, yazabilir ve scroll yapabilirsin.`;
  if (canDirectPreviewControl()) return `Interact'e bas, ardından ${layerLabel.toLowerCase()} katmanı üzerinden doğrudan kontrol et.`;
  return "Interact şu anda kullanılamıyor.";
}

function interactStatusText() {
  const parts = [state.ui.interactActive ? "Interact açık" : "Interact kapalı", `katman: ${currentControlLayerLabel()}`];
  if (state.ui.manualPending) {
    parts.push("işlem gönderiliyor");
  } else if (state.ui.interactBuffer) {
    parts.push(`bekleyen yazı: "${shortText(state.ui.interactBuffer, 24)}"`);
  } else if (state.ui.interactActive) {
    parts.push("tıkla, sürükle, yaz veya scroll yap");
  }
  return parts.join(" · ");
}

function interactSurface() {
  return document.getElementById("tv-screen");
}

function focusInteractSurface() {
  const surface = interactSurface();
  surface?.focus({ preventScroll: true });
}

function isAgentraEditableTarget(target) {
  if (!target || !(target instanceof Element)) return false;
  return Boolean(target.closest("input, textarea, [contenteditable='true']"));
}

function isPrintableKeyEvent(event) {
  return !event.ctrlKey && !event.metaKey && !event.altKey && !event.isComposing && event.key.length === 1;
}

function isInteractSpecialKey(key) {
  return ["Enter", "Tab", "Backspace", "Escape", "ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight"].includes(key);
}

function displayedRunSummary() {
  const thread = state.activeThread;
  if (!thread || !state.selectedRunId) return null;
  return runSummaries(thread).find((item) => item.run_id === state.selectedRunId) || null;
}

function currentControlLayer() {
  const takeover = pendingTakeoverRequest(state.activeThread);
  if (takeover) {
    if (String(takeover.pause_kind || "") === "desktop_control_takeover") return "desktop";
    return String(takeover.tool || "") === "computer" ? "desktop" : "browser";
  }
  const manual = state.ui.manualLayerOverride;
  if (manual === "desktop" || manual === "browser") return manual;
  return state.ui.controlLayer === "desktop" ? "desktop" : "browser";
}

function latestVisualTool() {
  for (let index = state.events.length - 1; index >= 0; index -= 1) {
    const event = state.events[index];
    if (!event || !event.type) continue;
    if ((event.type === "visual_intent" || event.type === "tool_call" || event.type === "tool_result") && event.tool) {
      if (event.tool === "computer") return "computer";
      if (event.tool === "windows_desktop") return "windows_desktop";
      if (event.tool === "browser") return "browser";
      if (event.tool === "local_system") return "local_system";
      continue;
    }
    if (event.type === "screenshot") {
      const label = String(event.frame_label || event.label || "");
      if (label.startsWith("computer ·")) return "computer";
      if (label.startsWith("windows_desktop ·")) return "windows_desktop";
      if (label.startsWith("browser ·")) return "browser";
      if (label.startsWith("local_system ·")) return "local_system";
    }
  }
  return null;
}

function preferredAutoLayer() {
  const visualTool = latestVisualTool();
  const thread = state.activeThread;
  const activeRun = activeRunForThread(thread);
  if (visualTool === "computer") return "desktop";
  if (visualTool === "windows_desktop") return "desktop";
  if (visualTool === "browser") return "browser";
  if (visualTool === "local_system") return "browser";
  if (activeRun && (activeRun.control_surface_hint === "desktop" || activeRun.control_surface_hint === "browser")) {
    return activeRun.control_surface_hint;
  }
  if (state.browser.active) return "browser";
  return "desktop";
}

function syncAutoControlLayer() {
  if (!state.ui.autoFollowLayer || state.ui.interactActive) return;
  if (state.historyMode || !state.liveMode) return;
  const thread = state.activeThread;
  const activeRun = activeRunForThread(thread);
  if (!thread || !activeRun || activeRun.run_id !== state.selectedRunId) return;
  if (thread.status !== "running") return;
  const nextLayer = preferredAutoLayer();
  if (nextLayer) state.ui.controlLayer = nextLayer;
}

function currentControlLayerLabel() {
  return currentControlLayer() === "desktop" ? "Masaüstü" : "Tarayıcı";
}

function currentSurfaceStatusLabel() {
  const activity = state.activity || {};
  const channel = String(activity.channel || "").toLowerCase();
  if (channel === "desktop") return "MASAÜSTÜ";
  if (channel === "desktop_hidden") return "ARKA PLAN MASAÜSTÜ";
  if (channel === "desktop_native") return "YEREL MASAÜSTÜ";
  if (channel === "local_system") return "YEREL SİSTEM";
  if (channel === "filesystem") return "DOSYA SİSTEMİ";
  if (channel === "terminal") return "TERMİNAL";
  if (channel === "workspace") return "ÇALIŞMA ALANI";
  if (channel === "browser") return "TARAYICI";
  return currentControlLayer() === "desktop" ? "MASAÜSTÜ" : "TARAYICI";
}

function currentInteractTool() {
  return currentControlLayer() === "desktop" ? "computer" : "browser";
}

function browserMirrorAvailable() {
  if (pendingTakeoverRequest(state.activeThread)) return true;
  if (state.browser.active) return true;
  const thread = state.activeThread;
  const activeRun = activeRunForThread(thread);
  const channels = [
    String(state.activity?.channel || "").toLowerCase(),
    String(thread?.activity?.channel || "").toLowerCase(),
    String(activeRun?.activity?.channel || "").toLowerCase(),
    String(activeRun?.control_surface_hint || "").toLowerCase(),
  ];
  return channels.includes("browser");
}

function backForwardAvailable() {
  return currentControlLayer() === "browser";
}

function canMirrorCurrentLayer() {
  if (currentControlLayer() === "desktop") return true;
  return browserMirrorAvailable();
}

function shouldAutoEnableInteract() {
  return Boolean(pendingTakeoverRequest(state.activeThread)) && canDirectPreviewControl();
}

function syncAutoInteract() {
  if (shouldAutoEnableInteract()) {
    state.ui.controlLayer = "browser";
    state.ui.interactActive = true;
  }
}

function interactPointerBusy() {
  return Boolean(state.ui.manualPending || state.ui.manualQueue);
}

function liveFramePath(kind, threadId, stamp) {
  const encodedThread = encodeURIComponent(threadId);
  if (kind === "desktop") {
    return {
      stream: buildAppUrl(`/threads/${encodedThread}/desktop-stream?stream=${stamp}`),
      frame: buildAppUrl(`/threads/${encodedThread}/desktop-frame?ts=${stamp}`),
    };
  }
  return {
    stream: buildAppUrl(`/threads/${encodedThread}/live-stream?stream=${stamp}`),
    frame: buildAppUrl(`/threads/${encodedThread}/live-frame?ts=${stamp}`),
  };
}

function liveMirrorUsesStream(source) {
  return source === "desktop";
}

function liveFramePollDelay(source) {
  return source === "desktop" ? 55 : 90;
}

function setError(key, message) {
  state.errors[key] = String(message || "");
}

function localizedLabel(event) {
  return event.display_label || event.frame_label || event.label || event.tool || "İşlem";
}

function localizedSummary(event) {
  return event.display_summary || event.summary || event.content || event.result || "";
}

function frameFromEvent(event) {
  if (event.type !== "screenshot" || !event.image_url) return null;
  return {
    id: event.frame_id || `frame-${Date.now()}`,
    timestamp: event.timestamp || "",
    label: event.label || event.frame_label || "Görsel Kare",
    display_label: event.display_label || event.frame_label || "Görsel Kare",
    summary: event.summary || "Görsel güncelleme",
    display_summary: event.display_summary || event.summary || "Görsel güncelleme",
    image_path: event.image_path || "",
    image_url: event.image_url,
    focus_x: typeof event.focus_x === "number" ? event.focus_x : null,
    focus_y: typeof event.focus_y === "number" ? event.focus_y : null,
  };
}

function currentFrame() {
  if (!state.frames.length) return null;
  if (state.liveMode || !state.selectedFrameId) return state.frames[state.frames.length - 1];
  return state.frames.find((frame) => frame.id === state.selectedFrameId) || state.frames[state.frames.length - 1];
}

function wantsLiveMirror() {
  const thread = state.activeThread;
  const activeRun = activeRunForThread(thread);
  if (!thread || !state.activeThreadId || !canMirrorCurrentLayer()) return false;
  if (state.historyMode || !state.liveMode) return false;
  if (!activeRun || !state.selectedRunId) return false;
  return activeRun.run_id === state.selectedRunId;
}

function releaseLiveFrameUrl() {
  if (state.liveFrameObjectUrl) {
    try {
      window.URL.revokeObjectURL(state.liveFrameObjectUrl);
    } catch (error) {
      console.debug("Failed to revoke live frame URL", error);
    }
  }
  state.liveFrameObjectUrl = null;
  state.liveFrameSignature = null;
  state.liveFrameUrl = null;
  state.liveFrameStamp = 0;
  state.liveFrameLoadedAt = 0;
}

function cancelLiveMirrorRequest() {
  if (state.liveFrameTimer) {
    window.clearTimeout(state.liveFrameTimer);
    state.liveFrameTimer = null;
  }
  if (state.liveFrameAbortController) {
    state.liveFrameAbortController.abort();
  }
  state.liveFrameAbortController = null;
}

function buildSteps(events) {
  const steps = [];
  let currentTool = null;

  events.forEach((event, index) => {
    const timestamp = event.timestamp || "";
    const type = event.type;
    if (type === "phase") return;

    if (type === "thought") {
      currentTool = null;
      steps.push({
        id: `step-${index + 1}`,
        kind: "thought",
        tone: "assistant",
        title: localizedLabel(event) || "Düşünce özeti",
        summary: shortText(localizedSummary(event), 260),
        status_label: "düşünüyor",
        timestamp,
      });
      return;
    }

    if (type === "tool_call") {
      currentTool = {
        id: `step-${index + 1}`,
        kind: "tool",
        tone: "pending",
        tool: event.tool,
        title: localizedLabel(event),
        summary: shortText(localizedSummary(event), 220),
        status_label: "uyguluyor",
        timestamp,
        detail: JSON.stringify(event.args || {}, null, 2),
        image_url: null,
      };
      steps.push(currentTool);
      return;
    }

    if (type === "visual_intent") {
      if (currentTool && currentTool.tool === event.tool) {
        currentTool.summary = shortText(localizedSummary(event), 220);
        currentTool.status_label = "hazırlanıyor";
        currentTool.focus_x = event.focus_x;
        currentTool.focus_y = event.focus_y;
      }
      return;
    }

    if (type === "screenshot") {
      if (currentTool) {
        currentTool.image_url = event.image_url || currentTool.image_url;
        currentTool.frame_id = event.frame_id || currentTool.frame_id;
      } else {
        steps.push({
          id: `step-${index + 1}`,
          kind: "visual",
          tone: "neutral",
          title: localizedLabel(event),
          summary: shortText(localizedSummary(event), 220),
          status_label: "görsel",
          timestamp,
          image_url: event.image_url,
        });
      }
      return;
    }

    if (type === "tool_result") {
      if (currentTool && currentTool.tool === event.tool) {
        currentTool.tone = event.success ? "success" : "error";
        currentTool.status_label = event.success ? "tamamlandı" : "hata";
        currentTool.summary = shortText(localizedSummary(event), 220);
        currentTool.detail = String(event.result || "");
        currentTool = null;
      } else {
        steps.push({
          id: `step-${index + 1}`,
          kind: "result",
          tone: event.success ? "success" : "error",
          title: localizedLabel(event),
          summary: shortText(localizedSummary(event), 220),
          status_label: event.success ? "tamamlandı" : "hata",
          timestamp,
          detail: String(event.result || ""),
        });
      }
      return;
    }

    if (type === "done") {
      currentTool = null;
      steps.push({
        id: `step-${index + 1}`,
        kind: "done",
        tone: "success",
        title: "Görev Tamamlandı",
        summary: String(event.content || ""),
        status_label: "tamamlandı",
        timestamp,
      });
      return;
    }

    if (type === "error") {
      currentTool = null;
      steps.push({
        id: `step-${index + 1}`,
        kind: "error",
        tone: "error",
        title: "Bir Hata Oluştu",
        summary: shortText(event.content || "", 260),
        status_label: "hata",
        timestamp,
      });
    }
  });

  return steps.slice(-24);
}

function deriveActivity(events, frames) {
  const latestFrame = frames.length ? frames[frames.length - 1] : null;
  for (let index = events.length - 1; index >= 0; index -= 1) {
    const event = events[index];
    if (event.type === "phase") {
      return {
        mode: event.phase || "thinking",
        summary: localizedSummary(event),
        focus_x: latestFrame && typeof latestFrame.focus_x === "number" ? latestFrame.focus_x : DEFAULT_FOCUS.x,
        focus_y: latestFrame && typeof latestFrame.focus_y === "number" ? latestFrame.focus_y : DEFAULT_FOCUS.y,
      };
    }
    if (event.type === "visual_intent") {
      return {
        mode: "acting",
        summary: localizedSummary(event),
        focus_x: typeof event.focus_x === "number" ? event.focus_x : DEFAULT_FOCUS.x,
        focus_y: typeof event.focus_y === "number" ? event.focus_y : DEFAULT_FOCUS.y,
      };
    }
    if (event.type === "tool_call") {
      return {
        mode: "acting",
        summary: localizedSummary(event),
        focus_x: latestFrame && typeof latestFrame.focus_x === "number" ? latestFrame.focus_x : DEFAULT_FOCUS.x,
        focus_y: latestFrame && typeof latestFrame.focus_y === "number" ? latestFrame.focus_y : DEFAULT_FOCUS.y,
      };
    }
    if (event.type === "screenshot") {
      return {
        mode: "acting",
        summary: localizedSummary(event),
        focus_x: typeof event.focus_x === "number" ? event.focus_x : DEFAULT_FOCUS.x,
        focus_y: typeof event.focus_y === "number" ? event.focus_y : DEFAULT_FOCUS.y,
      };
    }
    if (event.type === "tool_result") {
      return {
        mode: "idle",
        summary: localizedSummary(event),
        focus_x: latestFrame && typeof latestFrame.focus_x === "number" ? latestFrame.focus_x : DEFAULT_FOCUS.x,
        focus_y: latestFrame && typeof latestFrame.focus_y === "number" ? latestFrame.focus_y : DEFAULT_FOCUS.y,
      };
    }
    if (event.type === "done" || event.type === "error") {
      return {
        mode: "idle",
        summary: localizedSummary(event),
        focus_x: latestFrame && typeof latestFrame.focus_x === "number" ? latestFrame.focus_x : DEFAULT_FOCUS.x,
        focus_y: latestFrame && typeof latestFrame.focus_y === "number" ? latestFrame.focus_y : DEFAULT_FOCUS.y,
      };
    }
  }
  return {
    mode: "idle",
    summary: "",
    focus_x: DEFAULT_FOCUS.x,
    focus_y: DEFAULT_FOCUS.y,
  };
}

function syncSelectionToFrames() {
  if (!state.frames.length) {
    state.selectedFrameId = null;
    state.scrubPercent = 1;
    return;
  }
  if (state.liveMode || !state.selectedFrameId) {
    state.selectedFrameId = state.frames[state.frames.length - 1].id;
  } else if (!state.frames.find((frame) => frame.id === state.selectedFrameId)) {
    state.selectedFrameId = state.frames[state.frames.length - 1].id;
    state.liveMode = true;
  }
  const index = Math.max(0, state.frames.findIndex((frame) => frame.id === state.selectedFrameId));
  state.scrubPercent = state.frames.length > 1 ? index / (state.frames.length - 1) : 1;
}

function disconnectStream() {
  if (state.source) {
    state.source.close();
    state.source = null;
  }
  state.streamRunId = null;
}

function stopLiveMirror() {
  state.liveFrameLoopToken += 1;
  state.liveFrameThreadId = null;
  state.liveFrameSource = "browser";
  state.liveFrameMode = "stream";
  state.liveFramePumpActive = false;
  state.liveFrameErrorCount = 0;
  cancelLiveMirrorRequest();
  releaseLiveFrameUrl();
}

async function fetchLiveFrameBlob(threadId, source, loopToken) {
  if (!threadId || source !== "browser") return;
  const stamp = state.liveFrameStamp + 1;
  const controller = new AbortController();
  state.liveFrameAbortController = controller;
  try {
    const response = await fetch(liveFramePath(source, threadId, stamp).frame, {
      cache: "no-store",
      signal: controller.signal,
    });
    if (state.liveFrameAbortController === controller) {
      state.liveFrameAbortController = null;
    }
    if (
      state.liveFrameLoopToken !== loopToken
      || state.liveFrameThreadId !== threadId
      || state.liveFrameSource !== source
      || !wantsLiveMirror()
    ) {
      return;
    }
    if (response.status === 204) {
      scheduleNextLiveFrame(threadId, source, liveFramePollDelay(source));
      return;
    }
    if (!response.ok) {
      throw new Error(`Live frame request failed (${response.status})`);
    }
    const frameBytes = await response.arrayBuffer();
    if (!frameBytes || !frameBytes.byteLength) {
      scheduleNextLiveFrame(threadId, source, liveFramePollDelay(source));
      return;
    }
    const frameSignature = fingerprintFrameBytes(frameBytes);
    if (frameSignature === state.liveFrameSignature && state.liveFrameUrl) {
      state.liveFrameErrorCount = 0;
      state.liveFrameLoadedAt = Date.now();
      scheduleNextLiveFrame(threadId, source, liveFramePollDelay(source));
      return;
    }
    const frameBlob = new Blob([frameBytes], {
      type: response.headers.get("Content-Type") || "image/png",
    });
    if (!frameBlob.size) {
      scheduleNextLiveFrame(threadId, source, liveFramePollDelay(source));
      return;
    }
    const nextUrl = window.URL.createObjectURL(frameBlob);
    if (
      state.liveFrameLoopToken !== loopToken
      || state.liveFrameThreadId !== threadId
      || state.liveFrameSource !== source
      || !wantsLiveMirror()
    ) {
      window.URL.revokeObjectURL(nextUrl);
      return;
    }
    await preloadLiveFrameUrl(nextUrl);
    if (
      state.liveFrameLoopToken !== loopToken
      || state.liveFrameThreadId !== threadId
      || state.liveFrameSource !== source
      || !wantsLiveMirror()
    ) {
      window.URL.revokeObjectURL(nextUrl);
      return;
    }
    const previousUrl = state.liveFrameObjectUrl;
    state.liveFrameObjectUrl = nextUrl;
    state.liveFrameSignature = frameSignature;
    state.liveFrameUrl = nextUrl;
    state.liveFrameStamp = stamp;
    state.liveFrameErrorCount = 0;
    state.liveFrameLoadedAt = Date.now();
    renderTV();
    if (previousUrl && previousUrl !== nextUrl) {
      window.setTimeout(() => {
        try {
          window.URL.revokeObjectURL(previousUrl);
        } catch (error) {
          console.debug("Failed to revoke previous live frame URL", error);
        }
      }, 1000);
    }
  } catch (error) {
    if (controller.signal.aborted || error?.name === "AbortError") {
      return;
    }
    if (state.liveFrameAbortController === controller) {
      state.liveFrameAbortController = null;
    }
    if (
      state.liveFrameLoopToken !== loopToken
      || state.liveFrameThreadId !== threadId
      || state.liveFrameSource !== source
      || !wantsLiveMirror()
    ) {
      return;
    }
    state.liveFrameErrorCount += 1;
    const retryDelay = Math.min(320, 70 + (state.liveFrameErrorCount * 40));
    scheduleNextLiveFrame(threadId, source, retryDelay);
  }
}

function fingerprintFrameBytes(buffer) {
  const bytes = new Uint8Array(buffer);
  if (!bytes.length) return "0:0";
  let hash = bytes.length >>> 0;
  const step = Math.max(1, Math.floor(bytes.length / 64));
  for (let index = 0; index < bytes.length; index += step) {
    hash = ((hash * 33) ^ bytes[index]) >>> 0;
  }
  hash = ((hash * 33) ^ bytes[bytes.length - 1]) >>> 0;
  return `${bytes.length}:${hash.toString(16)}`;
}

function preloadLiveFrameUrl(url) {
  return new Promise((resolve, reject) => {
    const probe = new window.Image();
    let settled = false;
    const finish = () => {
      if (settled) return;
      settled = true;
      resolve();
    };
    const fail = () => {
      if (settled) return;
      settled = true;
      reject(new Error("Live frame preload failed"));
    };
    probe.onload = finish;
    probe.onerror = fail;
    probe.decoding = "async";
    probe.src = url;
    if (typeof probe.decode === "function") {
      probe.decode().then(finish).catch(() => {});
    }
  });
}

function scheduleNextLiveFrame(threadId, source, delay = 0) {
  if (!threadId) return;
  cancelLiveMirrorRequest();
  const loopToken = state.liveFrameLoopToken;
  state.liveFrameTimer = window.setTimeout(() => {
    state.liveFrameTimer = null;
    if (
      state.liveFrameLoopToken !== loopToken
      || state.liveFrameThreadId !== threadId
      || state.liveFrameSource !== source
      || !wantsLiveMirror()
    ) return;
    if (source === "browser") {
      void fetchLiveFrameBlob(threadId, source, loopToken);
      return;
    }
    state.liveFrameStamp += 1;
    state.liveFrameUrl = liveFramePath(source, threadId, state.liveFrameStamp).frame;
    renderTV();
  }, Math.max(0, delay));
}

function fallbackToLiveFramePolling(threadId, source) {
  if (!threadId) return;
  state.liveFrameMode = "poll";
  state.liveFrameErrorCount = 0;
  cancelLiveMirrorRequest();
  if (source !== "browser") {
    state.liveFrameStamp += 1;
    state.liveFrameUrl = liveFramePath(source, threadId, state.liveFrameStamp).frame;
    renderTV();
    return;
  }
  scheduleNextLiveFrame(threadId, source, 0);
  renderTV();
}

function startLiveMirror(threadId, source) {
  stopLiveMirror();
  if (!threadId) return;
  state.liveFrameThreadId = threadId;
  state.liveFrameSource = source;
  if (!liveMirrorUsesStream(source)) {
    state.liveFrameMode = "poll";
    state.liveFramePumpActive = false;
    scheduleNextLiveFrame(threadId, source, 0);
    renderTV();
    return;
  }
  state.liveFrameMode = "stream";
  state.liveFramePumpActive = true;
  state.liveFrameStamp += 1;
  state.liveFrameUrl = liveFramePath(source, threadId, state.liveFrameStamp).stream;
  renderTV();
  state.liveFrameTimer = window.setTimeout(() => {
    state.liveFrameTimer = null;
    if (
      state.liveFrameThreadId !== threadId
      || state.liveFrameSource !== source
      || state.liveFrameMode !== "stream"
      || state.liveFrameLoadedAt
    ) return;
    fallbackToLiveFramePolling(threadId, source);
  }, 1400);
}

function syncLiveMirror() {
  syncAutoControlLayer();
  const desiredThreadId = wantsLiveMirror() ? state.activeThreadId : null;
  const desiredSource = currentControlLayer();
  if (!desiredThreadId) {
    stopLiveMirror();
    return;
  }
  if (state.liveFrameThreadId === desiredThreadId && state.liveFrameSource === desiredSource) {
    if (
      state.liveFrameUrl
      || state.liveFrameTimer
      || state.liveFrameAbortController
      || state.liveFramePumpActive
    ) {
      return;
    }
  }
  startLiveMirror(desiredThreadId, desiredSource);
}

function handleLiveImageLoad() {
  if (!wantsLiveMirror() || !state.liveFrameThreadId) return;
  state.liveFrameLoadedAt = Date.now();
  state.liveFrameErrorCount = 0;
  if (state.liveFrameMode === "stream") {
    cancelLiveMirrorRequest();
    return;
  }
  scheduleNextLiveFrame(
    state.liveFrameThreadId,
    state.liveFrameSource,
    liveFramePollDelay(state.liveFrameSource)
  );
}

function handleLiveImageError() {
  if (!wantsLiveMirror() || !state.liveFrameThreadId) return;
  if (state.liveFrameMode === "stream") {
    fallbackToLiveFramePolling(state.liveFrameThreadId, state.liveFrameSource);
    return;
  }
  state.liveFrameErrorCount += 1;
  const retryDelay = Math.min(320, 70 + (state.liveFrameErrorCount * 40));
  scheduleNextLiveFrame(state.liveFrameThreadId, state.liveFrameSource, retryDelay);
}

function clearRunState(options = {}) {
  if (options.disconnect !== false) disconnectStream();
  stopLiveMirror();
  resetInteractState();
  state.runId = null;
  state.selectedRunId = null;
  state.goal = "";
  state.status = "idle";
  state.events = [];
  state.frames = [];
  state.steps = [];
  state.audit = [];
  state.reportUrl = null;
  state.liveMode = true;
  state.historyMode = false;
  state.activity = {
    mode: "idle",
    summary: "",
    title: "",
    channel: "agent",
    visibility: "background",
    status: "idle",
    focus_x: DEFAULT_FOCUS.x,
    focus_y: DEFAULT_FOCUS.y,
  };
  state.browser = {
    active: false,
    active_url: "",
    active_title: "",
    tab_count: 0,
  };
  state.ui.activeThreadDetailKey = "";
  syncSelectionToFrames();
}

function applySnapshot(snapshot, options = {}) {
  state.runId = snapshot.run_id;
  state.selectedRunId = snapshot.run_id;
  state.goal = snapshot.goal || state.goal;
  state.provider = snapshot.provider;
  state.model = snapshot.model;
  state.status = snapshot.status || state.status;
  state.events = snapshot.events || [];
  state.frames = snapshot.frames || [];
  state.steps = snapshot.steps || buildSteps(state.events);
  state.audit = Array.isArray(snapshot.audit) ? snapshot.audit : [];
  state.reportUrl = snapshot.report_url || null;
  if (snapshot.thread_id) state.activeThreadId = snapshot.thread_id;
  state.historyMode = Boolean(options.history);
  if (!state.historyMode) state.liveMode = true;
  state.ui.threadDetailLoading = false;
  state.ui.threadRefreshRetrying = false;
  updateBrowserState(snapshot);
  syncAutoControlLayer();
  syncSelectionToFrames();
  state.activity = snapshot.activity || deriveActivity(state.events, state.frames);
  if (options.connect) {
    connectStream(snapshot.run_id);
  } else if (options.disconnect !== false) {
    disconnectStream();
  }
  render();
}

function applyEvent(event) {
  state.events = [...state.events, event];
  const frame = frameFromEvent(event);
  if (frame) state.frames = [...state.frames, frame];
  if (state.activeThread?.active_run) {
    const activeRunEvents = Array.isArray(state.activeThread.active_run.events) ? state.activeThread.active_run.events : [];
    state.activeThread.active_run.events = [...activeRunEvents, event];
    if (frame) {
      const activeRunFrames = Array.isArray(state.activeThread.active_run.frames) ? state.activeThread.active_run.frames : [];
      state.activeThread.active_run.frames = [...activeRunFrames, frame];
    }
  }
  if (state.activeThread) {
    if (event.type === "paused") {
      state.activeThread.status = "paused_for_user";
      state.activeThread.handoff_state = "user";
      if (state.activeThread.active_run) {
        state.activeThread.active_run.thread_status = "paused_for_user";
        state.activeThread.active_run.handoff_state = "user";
      }
    } else if (event.type === "resumed") {
      state.activeThread.status = "running";
      state.activeThread.handoff_state = "agent";
      if (state.activeThread.active_run) {
        state.activeThread.active_run.thread_status = "running";
        state.activeThread.active_run.handoff_state = "agent";
      }
    } else if (event.type === "approval_requested" || event.type === "question_requested") {
      state.activeThread.status = "blocked_waiting_user";
      if (state.activeThread.active_run) state.activeThread.active_run.thread_status = "blocked_waiting_user";
    } else if (event.type === "done") {
      state.activeThread.status = "completed";
      if (state.activeThread.active_run) state.activeThread.active_run.thread_status = "completed";
    } else if (event.type === "error") {
      state.activeThread.status = "error";
      if (state.activeThread.active_run) state.activeThread.active_run.thread_status = "error";
    }
  }
  if (event.type === "done" && state.status === "running") state.status = "completed";
  if (event.type === "error") state.status = "error";
  state.steps = buildSteps(state.events);
  syncAutoControlLayer();
  syncSelectionToFrames();
  state.activity = event.activity || deriveActivity(state.events, state.frames);
  if (state.activeThread) {
    state.activeThread.activity = state.activity;
    state.activeThread.activity_summary = String(state.activity?.summary || "");
    state.activeThread.activity_title = String(state.activity?.title || "");
    if (state.activeThread.active_run) state.activeThread.active_run.activity = state.activity;
    if (state.activeThread.active_run_summary) state.activeThread.active_run_summary.activity = state.activity;
  }
  render();
}
"""


def _app_script_render() -> str:
    return """
function currentOverlay(frame) {
  if (state.status === "running" && (state.activity.mode === "thinking" || state.activity.mode === "acting")) {
    return {
      busy: true,
      summary: state.activity.summary || (frame ? frame.display_summary : ""),
      focus_x: typeof state.activity.focus_x === "number" ? state.activity.focus_x : (frame ? frame.focus_x : DEFAULT_FOCUS.x),
      focus_y: typeof state.activity.focus_y === "number" ? state.activity.focus_y : (frame ? frame.focus_y : DEFAULT_FOCUS.y),
    };
  }
  if (frame) {
    return {
      busy: false,
      summary: frame.display_summary || "",
      focus_x: typeof frame.focus_x === "number" ? frame.focus_x : DEFAULT_FOCUS.x,
      focus_y: typeof frame.focus_y === "number" ? frame.focus_y : DEFAULT_FOCUS.y,
    };
  }
  return {
    busy: false,
    summary: "",
    focus_x: DEFAULT_FOCUS.x,
    focus_y: DEFAULT_FOCUS.y,
  };
}

function currentThreadTitle() {
  return "Ajan";
}

function currentRunStatusLabel() {
  const threadStatus = String(state.activeThread?.status || "");
  if (threadStatus === "paused_for_user" || threadStatus === "blocked_waiting_user") {
    return threadStatusLabel(threadStatus);
  }
  return statusLabel(state.status);
}

function setInlineText(id, value) {
  const node = document.getElementById(id);
  if (!node) return;
  node.textContent = value || "";
}

function renderThreadList() {
  const container = document.getElementById("thread-list");
  if (!container) return;
  if (!state.threads.length) {
    container.innerHTML = `<div class="empty-state">Henüz thread yok. Yeni bir run başlat.</div>`;
    return;
  }
  container.innerHTML = state.threads.map((thread) => {
    const activeRun = activeRunForThread(thread);
    const runs = runSummaries(thread);
    const latestRun = runs.length ? runs[runs.length - 1] : null;
    const summary = thread.activity_summary || activeRun?.activity?.summary || activeRun?.goal || latestRun?.goal || "Henüz run yok";
    const isActive = thread.thread_id === state.activeThreadId;
    const approvals = pendingApprovals(thread).length;
    const questions = pendingQuestions(thread).length;
    const modeLabel = permissionModeLabel(thread.permission_mode || "default");
    const activityTitle = thread.activity_title || activeRun?.activity?.title || "";
    return `
      <button type="button" class="thread-item ${isActive ? "active" : ""}" data-thread-select="${escapeHtml(thread.thread_id)}">
        <div class="thread-item-head">
          <span class="thread-title">${escapeHtml(thread.title || thread.thread_id)}</span>
          <span class="thread-chip ${escapeHtml(thread.status || "idle")}">${escapeHtml(threadStatusLabel(thread.status || "idle"))}</span>
        </div>
        <div class="thread-summary">${escapeHtml(shortText(summary, 84))}</div>
        <div class="badge-row">
          <span class="count-badge">${escapeHtml(modeLabel)}</span>
          ${activityTitle ? `<span class="count-badge">${escapeHtml(shortText(activityTitle, 22))}</span>` : ""}
          <span class="count-badge">${escapeHtml(activeRun ? "aktif run" : "son run")}</span>
          ${approvals ? `<span class="count-badge">${approvals} onay</span>` : ""}
          ${questions ? `<span class="count-badge">${questions} soru</span>` : ""}
        </div>
      </button>
    `;
  }).join("");
}

function renderSelectedThread() {
  const container = document.getElementById("selected-thread-panel");
  if (!container) return;
  const thread = state.activeThread;
  if (!thread) {
    container.innerHTML = state.errors.thread
      ? `<div class="empty-state">${escapeHtml(state.errors.thread)}</div>`
      : `<div class="empty-state">Detayları görmek için bir thread seç.</div>`;
    return;
  }
  const activeRun = activeRunForThread(thread);
  const selectedRun = displayedRunSummary();
  const canPause = Boolean(activeRun) && thread.status === "running";
  const canResume = Boolean(activeRun) && (thread.status === "paused_for_user" || thread.status === "blocked_waiting_user");
  const canOpenReport = Boolean(state.reportUrl);
  const canReturnLive = Boolean(activeRun && state.historyMode && state.selectedRunId && state.selectedRunId !== activeRun.run_id);
  const permissionMode = String(thread.permission_mode || "default");
  const activity = thread.activity || activeRun?.activity || state.activity || {};
  const activityTitle = String(activity.title || "Hazır");
  const activitySummary = String(activity.summary || "Yeni run bekleniyor.");
  const takeover = pendingTakeoverRequest(thread);
  const restartRequired = Boolean(thread.restart_required);
  const resumeLabel = takeover ? "Finish Control" : "Resume";
  container.innerHTML = `
    <div class="thread-detail-card">
      <div class="thread-item-head">
        <span class="thread-title">${escapeHtml(thread.title)}</span>
        <span class="thread-chip ${escapeHtml(thread.status || "idle")}">${escapeHtml(threadStatusLabel(thread.status || "idle"))}</span>
      </div>
      <div class="thread-summary">${escapeHtml(shortText((selectedRun || activeRun || {}).goal || "Henüz run yok", 120))}</div>
      <div class="detail-grid">
        <div class="detail-row">
          <span class="detail-key">Aktif run</span>
          <span>${escapeHtml(activeRun?.run_id || "—")}</span>
        </div>
        <div class="detail-row">
          <span class="detail-key">Görünen run</span>
          <span>${escapeHtml(state.selectedRunId || "—")}</span>
        </div>
        <div class="detail-row">
          <span class="detail-key">Workspace</span>
          <span class="path-copy">${escapeHtml(shortPath(thread.workspace_dir))}</span>
        </div>
        <div class="detail-row">
          <span class="detail-key">Memory</span>
          <span class="path-copy">${escapeHtml(shortPath(thread.memory_dir))}</span>
        </div>
        <div class="detail-row">
          <span class="detail-key">Long-term Memory</span>
          <span class="path-copy">${escapeHtml(shortPath(thread.long_term_memory_dir))}</span>
        </div>
        <div class="detail-row">
          <span class="detail-key">İzin modu</span>
          <span class="detail-value">${escapeHtml(permissionModeLabel(permissionMode))}</span>
        </div>
        <div class="detail-row">
          <span class="detail-key">Browser</span>
          <span class="detail-value">${escapeHtml(browserStatusLabel())}</span>
        </div>
        <div class="detail-row">
          <span class="detail-key">Tarayıcı kimliği</span>
          <span class="detail-value">${escapeHtml(browserIdentityLabel(thread.browser_identity || state.browser.identity, thread.browser_profile_name || state.browser.profile_name))}</span>
        </div>
        <div class="detail-row">
          <span class="detail-key">URL</span>
          <span class="path-copy">${escapeHtml(shortText(state.browser.active_url || "-", 54))}</span>
        </div>
        <div class="detail-row">
          <span class="detail-key">Title</span>
          <span class="path-copy">${escapeHtml(shortText(state.browser.active_title || "-", 54))}</span>
        </div>
        <div class="detail-row">
          <span class="detail-key">Canlı aktivite</span>
          <span class="detail-value">${escapeHtml(activityTitle)}</span>
        </div>
        <div class="detail-row">
          <span class="detail-key">Aktivite özeti</span>
          <span class="path-copy">${escapeHtml(shortText(activitySummary, 72))}</span>
        </div>
        <div class="detail-row">
          <span class="detail-key">Görünürlük</span>
          <span class="detail-value">${escapeHtml(activityVisibilityLabel(activity))}</span>
        </div>
        ${state.browser.last_error ? `
        <div class="detail-row">
          <span class="detail-key">Tarayıcı uyarısı</span>
          <span class="path-copy">${escapeHtml(shortText(state.browser.last_error, 72))}</span>
        </div>` : ""}
      </div>
      <div class="manual-grid" style="margin-top: 14px;">
        <label class="field-label" for="thread-permission-select">İzin modu</label>
        <div class="inline-form">
          <select id="thread-permission-select" class="text-input" data-thread-permission-select>
            <option value="default" ${permissionMode === "default" ? "selected" : ""}>Default</option>
            <option value="full" ${permissionMode === "full" ? "selected" : ""}>Full</option>
          </select>
          <button type="button" class="action-button" data-thread-action="save-permission">İzin modunu kaydet</button>
        </div>
        <div class="helper-text">${escapeHtml(permissionModeSummary(permissionMode))}</div>
      </div>
      ${restartRequired ? `
      <div class="takeover-banner" style="margin-top: 14px;">
        <div class="takeover-banner-title">Yeniden Başlatma Gerekli</div>
        <div class="takeover-banner-text">${escapeHtml(thread.restart_notice || "Sunucu yeniden başlatıldı. Bu thread görünümü geri yüklendi ama canlı run devam etmiyor.")}</div>
      </div>` : ""}
      ${takeover ? `
      <div class="takeover-banner" style="margin-top: 14px;">
        <div class="takeover-banner-title">Kontrol Sende</div>
        <div class="takeover-banner-text">${escapeHtml(takeover.content || "Tarayicida hassas adimi manuel tamamlayip Finish Control ile devam et.")}</div>
      </div>` : ""}
      <div class="action-row">
        <button type="button" class="action-button" data-thread-action="pause" ${canPause && !restartRequired ? "" : "disabled"}>Pause</button>
        <button type="button" class="action-button" data-thread-action="resume" ${canResume && !restartRequired ? "" : "disabled"}>${resumeLabel}</button>
        <button type="button" class="action-button" data-thread-action="report" ${canOpenReport ? "" : "disabled"}>Report aç</button>
        <button type="button" class="action-button" data-thread-action="logs">Logs</button>
        ${canReturnLive && !restartRequired ? `<button type="button" class="action-button primary" data-thread-action="return-live">Aktif run'a dön</button>` : ""}
      </div>
    </div>
  `;
}

function renderApprovalList() {
  const container = document.getElementById("approval-list");
  if (!container) return;
  const approvals = pendingApprovals(state.activeThread);
  if (!approvals.length) {
    container.innerHTML = `<div class="empty-state">Bekleyen onay yok.</div>`;
    return;
  }
  container.innerHTML = approvals.map((item) => `
    <div class="request-card">
      <div class="request-title">${escapeHtml(item.summary || item.tool || "Onay isteği")}</div>
      <div class="request-text">${escapeHtml(item.reason || "Bu işlem kullanıcı onayı bekliyor.")}</div>
      <div class="request-meta">${escapeHtml(item.request_id)}</div>
      <div class="action-row">
        <button type="button" class="action-button success" data-approval-id="${escapeHtml(item.request_id)}" data-approved="true" ${state.ui.pendingApprovalIds[item.request_id] ? "disabled" : ""}>Onayla</button>
        <button type="button" class="action-button warn" data-approval-id="${escapeHtml(item.request_id)}" data-approved="false" ${state.ui.pendingApprovalIds[item.request_id] ? "disabled" : ""}>Reddet</button>
      </div>
    </div>
  `).join("");
}

function renderQuestionList() {
  const container = document.getElementById("question-list");
  if (!container) return;
  const questions = pendingQuestions(state.activeThread);
  if (!questions.length) {
    container.innerHTML = `<div class="empty-state">Bekleyen soru yok.</div>`;
    return;
  }
  container.innerHTML = questions.map((item) => `
    <div class="request-card">
      <div class="request-title">${escapeHtml(item.summary || "Kullanıcı yanıtı gerekiyor")}</div>
      <div class="request-text">${escapeHtml(item.prompt)}</div>
      ${item.response_kind === "secret" ? `<div class="helper-text">Bu yanıt gizli olarak işlenecek ve kaydedilmeyecek.</div>` : ""}
      <div class="inline-form">
        <input class="mini-input" type="${item.response_kind === "secret" ? "password" : "text"}" autocomplete="off" value="${escapeHtml(state.drafts.questionAnswers[item.request_id] || "")}" data-question-input="${escapeHtml(item.request_id)}" placeholder="${item.response_kind === "secret" ? "Gizli yanıtı yaz" : "Yanıtını yaz"}" />
        <button type="button" class="action-button primary" data-question-submit="${escapeHtml(item.request_id)}" ${state.ui.pendingQuestionIds[item.request_id] ? "disabled" : ""}>Gönder</button>
      </div>
    </div>
  `).join("");
}

function renderManualControls() {
  const container = document.getElementById("manual-controls");
  if (!container) return;
  const enabled = canManualControl(state.activeThread);
  const disabled = enabled ? "" : "disabled";
  container.innerHTML = `
    <div class="manual-grid">
      <div class="helper-text">${enabled ? "Hızlı kontrol artık canlı TV üzerinden yapılabiliyor. Buradaki alanlar uzman/fallback kontrolleridir." : "Manuel kontrol için önce thread'i pause et veya kullanıcı bekleyen duruma gelmesini bekle."}</div>
      <div class="action-row">
        <button type="button" class="action-button" data-manual-action="back" ${disabled}>Geri</button>
        <button type="button" class="action-button" data-manual-action="forward" ${disabled}>İleri</button>
      </div>
      <div class="inline-form">
        <input class="mini-input" type="number" value="${escapeHtml(state.drafts.scrollAmount)}" data-manual-input="scrollAmount" placeholder="Kaydırma miktarı" ${disabled} />
        <button type="button" class="action-button" data-manual-action="scroll" ${disabled}>Kaydır</button>
      </div>
      <div class="inline-form">
        <input class="mini-input" type="text" value="${escapeHtml(state.drafts.clickSelector)}" data-manual-input="clickSelector" placeholder="CSS selector" ${disabled} />
        <button type="button" class="action-button" data-manual-action="click" ${disabled}>Selector'a tıkla</button>
      </div>
      <div class="manual-row">
        <input class="mini-input" type="text" value="${escapeHtml(state.drafts.typeSelector)}" data-manual-input="typeSelector" placeholder="CSS selector" ${disabled} />
        <input class="mini-input" type="text" value="${escapeHtml(state.drafts.typeText)}" data-manual-input="typeText" placeholder="Yazılacak metin" ${disabled} />
      </div>
      <div class="action-row">
        <button type="button" class="action-button" data-manual-action="type" ${disabled}>Selector'a yaz</button>
      </div>
      <details class="details-shell">
        <summary>Gelişmiş</summary>
        <div class="manual-grid">
          <input class="mini-input" type="text" value="${escapeHtml(state.drafts.advancedTool)}" data-manual-input="advancedTool" placeholder="Tool adı" ${disabled} />
          <textarea class="json-input" data-manual-input="advancedArgs" placeholder='{"action":"back"}' ${disabled}>${escapeHtml(state.drafts.advancedArgs)}</textarea>
          <div class="action-row">
            <button type="button" class="action-button primary" data-manual-action="advanced" ${disabled}>Manuel aksiyon gönder</button>
          </div>
        </div>
      </details>
    </div>
  `;
}

function renderRunHistory() {
  const container = document.getElementById("run-history");
  if (!container) return;
  const runs = runSummaries(state.activeThread).slice().reverse();
  if (!runs.length) {
    container.innerHTML = `<div class="empty-state">Bu thread için kayıtlı run yok.</div>`;
    return;
  }
  container.innerHTML = runs.map((run) => {
    const active = run.run_id === state.selectedRunId;
    return `
      <div class="history-item ${active ? "active" : ""}">
        <div class="history-item-head">
          <span class="thread-title">${escapeHtml(shortText(run.goal || run.run_id, 54))}</span>
          <span class="thread-chip ${escapeHtml(run.status || "idle")}">${escapeHtml(statusLabel(run.status || "idle"))}</span>
        </div>
        <div class="history-meta">${escapeHtml(formatDateTime(run.started_at))} · ${escapeHtml(formatDateTime(run.finished_at))}</div>
        <div class="action-row">
          <button type="button" class="action-button" data-history-open="${escapeHtml(run.run_id)}">Aç</button>
          <a class="action-button" href="${escapeHtml(run.report_url)}" target="_blank" rel="noreferrer">Report</a>
        </div>
      </div>
    `;
  }).join("");
}

function renderHeader() {
  setInlineText("project-stamp", projectStamp());
  setInlineText("agent-title", currentThreadTitle());
  const logsLink = document.getElementById("global-logs-link");
  if (logsLink) logsLink.href = currentLogsUrl();
}

function renderStageManualDock() {
  const container = document.getElementById("stage-manual-dock");
  if (!container) return;
  const thread = state.activeThread;
  const activeRun = activeRunForThread(thread);
  const restartRequired = Boolean(thread?.restart_required);
  const previewEnabled = canDirectPreviewControl();
  const canInteract = canStartInteract();
  const canPause = Boolean(activeRun) && thread?.status === "running" && !restartRequired;
  const canResume = Boolean(activeRun) && canManualControl(thread) && !restartRequired;
  const canReturnLive = Boolean(activeRun && state.historyMode && state.selectedRunId && state.selectedRunId !== activeRun.run_id);
  const pending = Boolean(state.ui.manualPending);
  const dockDisabled = previewEnabled && !pending && backForwardAvailable() ? "" : "disabled";
  const interactLabel = state.ui.interactActive ? "Interact Açık" : (pending && canPause ? "Hazırlanıyor..." : "Interact");
  const activityTitle = String(state.activity?.title || "Hazır");
  const activitySummary = String(state.activity?.summary || "Yeni run bekleniyor.");
  const activityVisibility = activityVisibilityLabel(state.activity);
  const surfaceLabel = currentSurfaceStatusLabel();
  const takeover = pendingTakeoverRequest(thread);
  const resumeLabel = takeover ? "Finish Control" : "Resume";

  container.innerHTML = `
    <div class="manual-dock">
      ${restartRequired ? `
      <div class="takeover-banner">
        <div class="takeover-banner-title">Yeniden Başlatma Gerekli</div>
        <div class="takeover-banner-text">${escapeHtml(thread?.restart_notice || "Sunucu yeniden başlatıldı. Canlı kontrol için yeni bir run başlat ya da thread'i yeniden çalıştır.")}</div>
      </div>` : ""}
      ${takeover ? `
      <div class="takeover-banner">
        <div class="takeover-banner-title">Kontrol Sende</div>
        <div class="takeover-banner-text">${escapeHtml(takeover.content || "Tarayicida hassas adimi manuel tamamlayip Finish Control ile devam et.")}</div>
      </div>` : ""}
      <div class="manual-dock-head">
        <div class="manual-dock-copy">
          <div class="manual-dock-title">Canli Manuel Kontrol</div>
          <div class="manual-dock-text">${escapeHtml(interactHint())}</div>
        </div>
        <div class="action-row">
          <button type="button" class="action-button primary interact-button ${state.ui.interactActive ? "active" : ""}" data-manual-action="interact" ${canInteract && !pending && !state.ui.interactActive ? "" : "disabled"}>${escapeHtml(interactLabel)}</button>
          <button type="button" class="action-button" data-thread-action="pause" ${canPause && !pending ? "" : "disabled"}>Pause</button>
          <button type="button" class="action-button" data-thread-action="resume" ${canResume && !pending ? "" : "disabled"}>${resumeLabel}</button>
          ${canReturnLive && !restartRequired ? `<button type="button" class="action-button primary" data-thread-action="return-live" ${pending ? "disabled" : ""}>Aktif run'a dön</button>` : ""}
          <button type="button" class="action-button" data-manual-action="back" ${dockDisabled}>Geri</button>
          <button type="button" class="action-button" data-manual-action="forward" ${dockDisabled}>İleri</button>
        </div>
      </div>
      <div class="manual-inline-note"><strong>Yüzey:</strong> ${escapeHtml(surfaceLabel)} · <strong>Akış:</strong> ${escapeHtml(activityTitle)} · ${escapeHtml(activityVisibility)}</div>
      <div class="manual-inline-note">${escapeHtml(shortText(activitySummary, 160))}</div>
      <div class="manual-inline-note"><strong>Durum:</strong> ${escapeHtml(interactStatusText())}</div>
      <div class="inline-error" id="manual-dock-error">${escapeHtml(state.errors.manual || "")}</div>
    </div>
  `;
}

function renderTV() {
  const frame = currentFrame();
  const overlay = currentOverlay(frame);
  const liveMirrorRequested = wantsLiveMirror() && Boolean(state.liveFrameThreadId);
  const liveMirrorActive = liveMirrorRequested && Boolean(state.liveFrameUrl);
  const liveMirrorConnecting = liveMirrorRequested && !liveMirrorActive && !frame;
  const screen = document.getElementById("tv-screen");
  const image = document.getElementById("tv-image");
  const empty = document.getElementById("tv-empty");
  const dot = document.getElementById("cursor-dot");
  const bubble = document.getElementById("cursor-bubble");
  const title = document.getElementById("tv-frame-title");
  const activeLabel = currentThreadTitle();

  screen.classList.toggle("busy", Boolean(overlay.busy));
  screen.classList.toggle("manual-interact", interactReady());
  title.textContent = `${activeLabel} · ${currentRunStatusLabel()} · ${currentSurfaceStatusLabel()}`;

  if (!frame && !liveMirrorActive) {
    image.style.display = "none";
    empty.style.display = "grid";
    if (!state.activeThread && state.ui.threadRefreshRetrying) {
      empty.innerHTML = `<div><strong>Sunucu ısınıyor...</strong><div class="muted" style="margin-top:8px;">Thread listesi yeniden deneniyor. Sayfa boş görünse bile uygulama ayakta.</div></div>`;
    } else if (state.activeThread?.restart_required) {
      empty.innerHTML = `<div><strong>Thread geri yüklendi.</strong><div class="muted" style="margin-top:8px;">${escapeHtml(state.activeThread.restart_notice || "Sunucu yeniden başlatıldığı için bu canlı run otomatik devam etmiyor.")}</div></div>`;
    } else if (state.activeThread && state.ui.threadDetailLoading) {
      empty.innerHTML = `<div><strong>Thread detayları yükleniyor...</strong><div class="muted" style="margin-top:8px;">Özet veriler hazır. Run detayları arka planda alınıyor.</div></div>`;
    } else if (state.activeThread && state.errors.thread) {
      empty.innerHTML = `<div><strong>Thread görünümü hazır değil.</strong><div class="muted" style="margin-top:8px;">${escapeHtml(state.errors.thread)}</div></div>`;
    } else {
      empty.innerHTML = overlay.busy
        ? `<div><strong>${escapeHtml(overlay.summary || "İşlem hazırlanıyor...")}</strong><div class="muted" style="margin-top:8px;">Thread canlı olduğunda yeni kareler burada görünür.</div></div>`
        : `<div><strong>Bir thread seç veya yeni run başlat.</strong><div class="muted" style="margin-top:8px;">Canlı TV, timeline ve run replay burada gösterilecek.</div></div>`;
    }
  } else if (liveMirrorConnecting) {
    image.style.display = "none";
    empty.style.display = "grid";
    empty.innerHTML = `<div><strong>Canlı önizleme bağlanıyor...</strong><div class="muted" style="margin-top:8px;">İlk kare gelir gelmez burada görünecek.</div></div>`;
  } else {
    empty.style.display = "none";
    image.style.display = "block";
    const displayKey = liveMirrorActive ? `live-${state.liveFrameSource}-${state.liveFrameMode}-${state.liveFrameStamp}` : frame.id;
    const displaySrc = liveMirrorActive ? state.liveFrameUrl : frame.image_url;
    if (image.dataset.frameId !== displayKey) {
      if (liveMirrorActive) {
        image.classList.remove("swapping");
      } else {
        image.classList.add("swapping");
        window.setTimeout(() => image.classList.remove("swapping"), 180);
      }
      image.src = displaySrc;
      image.dataset.frameId = displayKey;
    }
  }

  if (state.ui.interactActive) {
    dot.style.display = "none";
    bubble.style.display = "none";
    return;
  }

  if (overlay.summary) {
    const left = `${Math.max(0, Math.min(100, (overlay.focus_x ?? DEFAULT_FOCUS.x) * 100))}%`;
    const top = `${Math.max(0, Math.min(100, (overlay.focus_y ?? DEFAULT_FOCUS.y) * 100))}%`;
    dot.style.display = "block";
    bubble.style.display = "block";
    dot.style.left = left;
    dot.style.top = top;
    bubble.style.left = left;
    bubble.style.top = top;
    dot.classList.toggle("pending", overlay.busy);
    bubble.classList.toggle("flip", (overlay.focus_y ?? DEFAULT_FOCUS.y) > 0.68);
    bubble.textContent = overlay.summary;
  } else {
    dot.style.display = "none";
    bubble.style.display = "none";
  }
}

function renderTimeline() {
  const frame = currentFrame();
  const frameIndex = frame ? Math.max(0, state.frames.findIndex((item) => item.id === frame.id)) : 0;
  const label = document.getElementById("scrubber-label");
  const progress = document.getElementById("timeline-progress");
  const handle = document.getElementById("timeline-handle");
  const percent = state.frames.length > 1 ? state.scrubPercent : 1;
  const percentText = `${Math.max(0, Math.min(100, percent * 100))}%`;
  label.textContent = state.frames.length ? `${frameIndex + 1} / ${state.frames.length}` : "0 / 0";
  progress.style.width = percentText;
  handle.style.left = percentText;
  setInlineText("tv-status", currentRunStatusLabel());
}

function renderConsole() {
  const goalInput = document.getElementById("goal-input");
  const titleInput = document.getElementById("thread-title-input");
  const permissionModeInput = document.getElementById("permission-mode-input");
  const appendThreadButton = document.getElementById("append-thread-button");
  const advancedToggle = document.getElementById("advanced-toggle");
  const advancedPanelIds = [
    "thread-list",
    "selected-thread-panel",
    "approval-list",
    "question-list",
    "run-history",
  ];

  if (goalInput && goalInput.value !== state.drafts.goal) goalInput.value = state.drafts.goal;
  if (titleInput && titleInput.value !== state.drafts.threadTitle) titleInput.value = state.drafts.threadTitle;
  if (permissionModeInput) {
    const nextPermissionMode = state.drafts.permissionMode || state.defaultPermissionMode || "default";
    if (permissionModeInput.value !== nextPermissionMode) permissionModeInput.value = nextPermissionMode;
  }
  if (appendThreadButton) appendThreadButton.disabled = !state.activeThreadId;
  advancedPanelIds.forEach((id) => {
    const panel = document.getElementById(id)?.closest(".console-section");
    if (panel) panel.hidden = !state.ui.showAdvanced;
  });
  if (advancedToggle) advancedToggle.textContent = state.ui.showAdvanced ? "Advanced Gizle" : "Advanced";

  renderThreadList();
  renderSelectedThread();
  renderApprovalList();
  renderQuestionList();
  renderRunHistory();

  setInlineText("run-error", state.errors.run);
  setInlineText("thread-error", state.errors.thread);
  setInlineText("approval-error", state.errors.approval);
  setInlineText("question-error", state.errors.question);
  setInlineText("manual-error", state.errors.manual);
}

function render() {
  ensureInteractState();
  syncAutoInteract();
  renderHeader();
  renderConsole();
  renderStageManualDock();
  renderTV();
  renderTimeline();
  syncLiveMirror();
  if (shouldAutoEnableInteract()) {
    window.requestAnimationFrame(() => focusInteractSurface());
  }
}

async function fetchJson(url, options = {}) {
  const controller = new AbortController();
  const timeoutMs = Number(options.timeoutMs || 0);
  const timeoutMessage = options.timeoutMessage || "İstek zaman aşımına uğradı.";
  const requestOptions = { ...options };
  delete requestOptions.timeoutMs;
  delete requestOptions.timeoutMessage;
  const headers = { "Content-Type": "application/json", ...(requestOptions.headers || {}) };
  const timeoutId = timeoutMs > 0 ? window.setTimeout(() => controller.abort(), timeoutMs) : null;
  try {
    const response = await fetch(url, {
      ...requestOptions,
      headers,
      signal: requestOptions.signal || controller.signal,
    });
    if (!response.ok) {
      let message = `${response.status}`;
      try {
        const payload = await response.json();
        message = payload.detail || message;
      } catch (error) {
        // Ignore JSON parse failures on empty responses.
      }
      throw new Error(message);
    }
    return response.json();
  } catch (error) {
    if (error?.name === "AbortError") {
      throw new Error(timeoutMessage);
    }
    throw error;
  } finally {
    if (timeoutId !== null) window.clearTimeout(timeoutId);
  }
}

function selectFrameByPercent(percent) {
  if (!state.frames.length) return;
  const safe = Math.max(0, Math.min(1, percent));
  state.scrubPercent = safe;
  state.liveMode = false;
  const index = state.frames.length > 1 ? Math.round(safe * (state.frames.length - 1)) : 0;
  state.selectedFrameId = state.frames[index].id;
  render();
}
"""


def _app_script_mount() -> str:
    return """
function setupScrubber() {
  const track = document.getElementById("timeline-track");

  function updateFromClientX(clientX) {
    const rect = track.getBoundingClientRect();
    if (!rect.width) return;
    selectFrameByPercent((clientX - rect.left) / rect.width);
  }

  track.addEventListener("pointerdown", (event) => {
    if (!state.frames.length) return;
    state.scrubDrag = { pointerId: event.pointerId };
    track.setPointerCapture(event.pointerId);
    updateFromClientX(event.clientX);
  });

  track.addEventListener("pointermove", (event) => {
    if (!state.scrubDrag || state.scrubDrag.pointerId !== event.pointerId) return;
    const clientX = event.clientX;
    if (state.scrubDrag.raf) {
      state.scrubDrag.pendingX = clientX;
      return;
    }
    state.scrubDrag.pendingX = clientX;
    state.scrubDrag.raf = window.requestAnimationFrame(() => {
      updateFromClientX(state.scrubDrag.pendingX);
      state.scrubDrag.raf = null;
    });
  });

  function stopScrub(event) {
    if (!state.scrubDrag || state.scrubDrag.pointerId !== event.pointerId) return;
    if (state.scrubDrag.raf) window.cancelAnimationFrame(state.scrubDrag.raf);
    track.releasePointerCapture?.(event.pointerId);
    state.scrubDrag = null;
  }

  track.addEventListener("pointerup", stopScrub);
  track.addEventListener("pointercancel", stopScrub);
}

function connectStream(runId) {
  if (!runId) return;
  if (state.streamRunId === runId && state.source) return;
  disconnectStream();
  state.streamRunId = runId;
  state.source = new EventSource(`/runs/${runId}/events`);
  state.source.onmessage = (message) => {
    const payload = JSON.parse(message.data);
    if (payload.kind === "snapshot") {
      applySnapshot(payload.snapshot, { connect: false, disconnect: false, history: false });
      return;
    }
    if (payload.kind === "event") {
      applyEvent(payload.event);
      return;
    }
    if (payload.kind === "status") {
      state.status = payload.status || state.status;
      applySnapshot(payload.snapshot, { connect: false, disconnect: false, history: false });
      return;
    }
    if (payload.kind === "complete") {
      state.status = payload.status || state.status;
      state.activity = deriveActivity(state.events, state.frames);
      if (state.source) {
        state.source.close();
        state.source = null;
      }
      state.streamRunId = null;
      render();
    }
  };
}

async function loadRun(runId, options = {}) {
  if (options.live) state.liveMode = true;
  if (options.history) state.liveMode = false;
  const snapshot = await fetchJson(`/runs/${runId}`);
  applySnapshot(snapshot, { connect: Boolean(options.live), history: Boolean(options.history) });
}

async function syncThreadSnapshot(thread, options = {}) {
  state.ui.threadDetailLoading = false;
  state.ui.threadRefreshRetrying = false;
  state.ui.activeThreadDetailKey = threadSummarySignature(thread);
  state.activeThread = thread;
  state.activeThreadId = thread.thread_id;
  state.audit = Array.isArray(thread.audit) ? thread.audit : state.audit;
  updateBrowserState(thread);
  settleThreadFocusPreference(thread.thread_id, selectedRunIdForThread(thread, options));
  const activeRun = activeRunForThread(thread);
  const activeRunDetail = activeRunDetailForThread(thread);
  const runs = runSummaries(thread);
  const runIds = runs.map((item) => item.run_id);

  let targetRunId = null;
  let live = false;

  if (options.preferLive && activeRun) {
    targetRunId = activeRun.run_id;
    live = true;
    state.liveMode = true;
  } else if (state.historyMode && state.selectedRunId && runIds.includes(state.selectedRunId)) {
    targetRunId = state.selectedRunId;
  } else if (options.preferRunId && runIds.includes(options.preferRunId)) {
    targetRunId = options.preferRunId;
    live = Boolean(activeRun && activeRun.run_id === options.preferRunId && options.preferLive);
  } else if (activeRun) {
    targetRunId = activeRun.run_id;
    live = true;
  } else if (runs.length) {
    targetRunId = runs[runs.length - 1].run_id;
  }

  if (!targetRunId) {
    clearRunState();
    state.audit = Array.isArray(thread.audit) ? thread.audit : [];
    updateBrowserState(thread);
    render();
    return;
  }

  if (activeRunDetail && targetRunId === activeRunDetail.run_id) {
    if (live) state.liveMode = true;
    applySnapshot(activeRunDetail, { connect: live, history: false });
    return;
  }

  await loadRun(targetRunId, { live: false, history: Boolean(state.historyMode) });
}

async function loadThreadDetail(threadId, options = {}) {
  state.ui.threadDetailLoading = true;
  render();
  try {
    const thread = await fetchJson(`/threads/${threadId}`, {
      timeoutMs: 4500,
      timeoutMessage: "Thread detayları yüklenirken sunucu ısınıyor. Tekrar denenecek.",
    });
    await syncThreadSnapshot(thread, options);
  } catch (error) {
    state.ui.threadDetailLoading = false;
    state.ui.threadRefreshRetrying = true;
    setError("thread", error.message);
    render();
    throw error;
  }
}

function pickInitialThreadId() {
  if (!state.threads.length) return null;
  if (state.ui.preferredThreadId && state.threads.some((item) => item.thread_id === state.ui.preferredThreadId)) {
    return state.ui.preferredThreadId;
  }
  if (state.activeThreadId && state.threads.some((item) => item.thread_id === state.activeThreadId)) {
    return state.activeThreadId;
  }
  if (state.initialRunId) {
    const matched = state.threads.find((item) => item.current_run_id === state.initialRunId || runSummaries(item).some((run) => run.run_id === state.initialRunId));
    if (matched) return matched.thread_id;
  }
  return state.threads[0].thread_id;
}

async function refreshThreads(options = {}) {
  const requestedOptions = mergeRefreshOptions({}, options);
  rememberThreadFocusPreference(requestedOptions);
  if (state.refreshInFlight) {
    state.ui.queuedThreadRefresh = mergeRefreshOptions(state.ui.queuedThreadRefresh, requestedOptions);
    return;
  }
  state.refreshInFlight = true;
  try {
    const effectiveOptions = mergeRefreshOptions(requestedOptions, threadFocusPreference());
    const payload = await fetchJson("/threads", {
      timeoutMs: 3200,
      timeoutMessage: "Sunucu ısınıyor, thread listesi yeniden deneniyor.",
    });
    state.threads = payload.threads || [];
    state.ui.threadRefreshRetrying = false;
    setError("thread", "");
    render();
    const nextThreadId = effectiveOptions.forceThreadId || pickInitialThreadId();
    if (!nextThreadId) {
      state.activeThreadId = null;
      state.activeThread = null;
      clearRunState();
      render();
      return;
    }
    const nextThread = state.threads.find((item) => item.thread_id === nextThreadId) || null;
    if (!nextThread) {
      render();
      return;
    }
    applyThreadSummary(nextThread, {
      preferRunId: effectiveOptions.preferRunId || state.selectedRunId,
      preferLive: Boolean(effectiveOptions.preferLive),
    });
    render();
    if (!shouldRefreshThreadDetail(nextThread, effectiveOptions)) return;
    await loadThreadDetail(nextThreadId, {
      preferRunId: effectiveOptions.preferRunId || state.selectedRunId,
      preferLive: Boolean(effectiveOptions.preferLive),
    });
  } catch (error) {
    state.ui.threadRefreshRetrying = true;
    setError("thread", error.message);
    render();
  } finally {
    state.refreshInFlight = false;
    const queuedRefresh = state.ui.queuedThreadRefresh;
    if (queuedRefresh) {
      state.ui.queuedThreadRefresh = null;
      void refreshThreads(queuedRefresh);
    }
  }
}

async function clearAllThreads() {
  if (!window.confirm("Tüm threadleri silmek istiyor musun? Bu işlem kayıtlı thread geçmişini temizler.")) {
    return;
  }
  try {
    setError("thread", "");
    await fetchJson("/threads/clear", { method: "POST" });
    state.threads = [];
    state.activeThread = null;
    state.activeThreadId = null;
    clearRunState();
    render();
    await refreshThreads();
  } catch (error) {
    setError("thread", error.message);
    render();
  }
}

function scheduleThreadPolling() {
  if (state.threadPollTimer) window.clearInterval(state.threadPollTimer);
  state.threadPollTimer = window.setInterval(() => {
    refreshThreads({
      forceThreadId: state.activeThreadId,
      preferRunId: state.historyMode ? state.selectedRunId : null,
      preferLive: !state.historyMode,
    });
  }, 2000);
}

async function startRun(mode) {
  const goal = state.drafts.goal.trim();
  if (!goal) {
    setError("run", "Önce bir görev gir.");
    render();
    return;
  }

  const payload = { goal };
  if (mode === "new") {
    const title = state.drafts.threadTitle.trim();
    if (title) payload.thread_title = title;
    if (state.drafts.permissionMode) payload.permission_mode = state.drafts.permissionMode;
  } else if (state.activeThreadId) {
    payload.thread_id = state.activeThreadId;
  } else {
    setError("run", "Önce bir thread seç.");
    render();
    return;
  }

  try {
    setError("run", "");
    state.historyMode = false;
    const snapshot = await fetchJson("/runs", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    state.drafts.goal = "";
    if (mode === "new") state.drafts.threadTitle = "";
    if (mode === "new") state.drafts.permissionMode = "";
    await refreshThreads({
      forceThreadId: snapshot.thread_id,
      preferRunId: snapshot.run_id,
      preferLive: true,
    });
  } catch (error) {
    setError("run", error.message);
    render();
  }
}

async function updateThreadPermissionMode() {
  if (!state.activeThreadId) return;
  const select = document.querySelector("[data-thread-permission-select]");
  const permissionMode = String(select?.value || "default").toLowerCase();
  try {
    setError("thread", "");
    const snapshot = await fetchJson(`/threads/${state.activeThreadId}`, {
      method: "PATCH",
      body: JSON.stringify({ permission_mode: permissionMode }),
    });
    await syncThreadSnapshot(snapshot, {
      preferRunId: state.selectedRunId,
      preferLive: !state.historyMode,
    });
  } catch (error) {
    setError("thread", error.message);
    render();
  }
}

async function pauseActiveThread() {
  if (!state.activeThreadId) return;
  try {
    setError("thread", "");
    const snapshot = await fetchJson(`/threads/${state.activeThreadId}/pause`, { method: "POST" });
    await syncThreadSnapshot(snapshot, { preferLive: true });
    return snapshot;
  } catch (error) {
    setError("thread", error.message);
    render();
    return null;
  }
}

async function resumeActiveThread() {
  if (!state.activeThreadId) return;
  resetInteractState();
  state.ui.manualLayerOverride = null;
  state.ui.autoFollowLayer = true;
  try {
    setError("thread", "");
    const snapshot = await fetchJson(`/threads/${state.activeThreadId}/resume`, { method: "POST" });
    await syncThreadSnapshot(snapshot, { preferLive: true });
    return snapshot;
  } catch (error) {
    setError("thread", error.message);
    render();
    return null;
  }
}

async function respondApproval(requestId, approved) {
  if (!state.activeThreadId) return;
  if (state.ui.pendingApprovalIds[requestId]) return;
  try {
    state.ui.pendingApprovalIds[requestId] = true;
    setError("approval", "");
    render();
    const snapshot = await fetchJson(`/threads/${state.activeThreadId}/approvals/${requestId}`, {
      method: "POST",
      body: JSON.stringify({ approved, note: approved ? "UI onayı" : "UI reddi" }),
    });
    await syncThreadSnapshot(snapshot, { preferLive: true });
  } catch (error) {
    setError("approval", error.message);
    render();
  } finally {
    delete state.ui.pendingApprovalIds[requestId];
    render();
  }
}

async function submitQuestionAnswer(requestId) {
  if (!state.activeThreadId) return;
  if (state.ui.pendingQuestionIds[requestId]) return;
  const answer = (state.drafts.questionAnswers[requestId] || "").trim();
  if (!answer) {
    setError("question", "Önce bir yanıt yaz.");
    render();
    return;
  }
  try {
    state.ui.pendingQuestionIds[requestId] = true;
    setError("question", "");
    render();
    const snapshot = await fetchJson(`/threads/${state.activeThreadId}/questions/${requestId}`, {
      method: "POST",
      body: JSON.stringify({ answer }),
    });
    delete state.drafts.questionAnswers[requestId];
    await syncThreadSnapshot(snapshot, { preferLive: true });
  } catch (error) {
    setError("question", error.message);
    render();
  } finally {
    delete state.ui.pendingQuestionIds[requestId];
    render();
  }
}

async function sendManualAction(tool, args) {
  const threadId = state.activeThreadId;
  if (!threadId) return null;
  const manualArgs = {
    ...args,
    capture_result_screenshot: false,
    capture_follow_up_screenshots: false,
  };
  const runner = async () => {
    if (state.activeThreadId !== threadId) return null;
    state.ui.manualPending = true;
    try {
      setError("manual", "");
      render();
      const snapshot = await fetchJson(`/threads/${threadId}/actions`, {
        method: "POST",
        body: JSON.stringify({ tool, args: manualArgs, return_snapshot: false }),
      });
      if (state.activeThreadId === threadId && snapshot && snapshot.active_run) {
        await syncThreadSnapshot(snapshot, { preferLive: true });
      }
      return snapshot;
    } catch (error) {
      setError("manual", error.message);
      render();
      return null;
    } finally {
      state.ui.manualPending = false;
      render();
    }
  };
  const queued = Promise.resolve(state.ui.manualQueue).catch(() => {}).then(runner);
  state.ui.manualQueue = queued;
  try {
    return await queued;
  } finally {
    if (state.ui.manualQueue === queued) state.ui.manualQueue = null;
  }
}

function selectManualLayer(layer) {
  const nextLayer = layer === "desktop" ? "desktop" : "browser";
  state.ui.autoFollowLayer = false;
  state.ui.manualLayerOverride = nextLayer;
  if (currentControlLayer() === nextLayer && state.ui.controlLayer === nextLayer) {
    render();
    return;
  }
  resetInteractState();
  syncLiveMirror();
  render();
}

async function handleManualAction(kind) {
  if (kind === "interact") {
    return startInteract();
  }
  if (!canManualControl(state.activeThread)) {
    setError("manual", "Önce pause et veya kullanıcı bekleyen duruma gelmesini bekle.");
    render();
    return;
  }
  if (kind === "back") {
    if (!backForwardAvailable()) {
      setError("manual", "Geri yalnızca tarayıcı katmanında kullanılabilir.");
      render();
      return;
    }
    return sendManualAction("browser", { action: "back" });
  }
  if (kind === "forward") {
    if (!backForwardAvailable()) {
      setError("manual", "İleri yalnızca tarayıcı katmanında kullanılabilir.");
      render();
      return;
    }
    return sendManualAction("browser", { action: "forward" });
  }
  if (kind === "advanced") {
    try {
      const args = JSON.parse(state.drafts.advancedArgs || "{}");
      return sendManualAction(state.drafts.advancedTool.trim() || "browser", args);
    } catch (error) {
      setError("manual", "Geçerli bir JSON gir.");
      render();
    }
  }
}

function queueInteractText(text) {
  if (!text) return;
  state.ui.interactBuffer += text;
  clearInteractFlushTimer();
  state.ui.interactFlushTimer = window.setTimeout(() => {
    void flushInteractTextBuffer();
  }, 120);
  render();
}

async function flushInteractTextBuffer() {
  clearInteractFlushTimer();
  const text = state.ui.interactBuffer;
  if (!text) return true;
  state.ui.interactBuffer = "";
  render();
  const tool = currentInteractTool();
  const args = tool === "computer"
    ? { action: "type", text }
    : { action: "type", text };
  await sendManualAction(tool, args);
  focusInteractSurface();
  return true;
}

async function sendInteractKey(key) {
  await flushInteractTextBuffer();
  if (!interactReady()) return;
  const tool = currentInteractTool();
  const args = tool === "computer"
    ? { action: "key", text: key }
    : { action: "key", key };
  await sendManualAction(tool, args);
  focusInteractSurface();
}

async function startInteract() {
  if (!canStartInteract()) {
    setError("manual", state.historyMode || !state.liveMode ? "Önce aktif canlı run'a dön." : "Interact şu anda kullanılamıyor.");
    render();
    return;
  }
  setError("manual", "");
  if (!canDirectPreviewControl()) {
    const snapshot = await pauseActiveThread();
    if (!snapshot) return;
  }
  if (!canDirectPreviewControl()) {
    setError("manual", "Interact için kontrol devri tamamlanamadı.");
    render();
    return;
  }
  state.ui.interactActive = true;
  render();
  focusInteractSurface();
}

function previewImageMetrics() {
  const image = document.getElementById("tv-image");
  if (!image || image.style.display === "none" || !image.src) return null;
  const naturalWidth = Number(image.naturalWidth || 0);
  const naturalHeight = Number(image.naturalHeight || 0);
  if (!naturalWidth || !naturalHeight) return null;
  const rect = image.getBoundingClientRect();
  if (!rect.width || !rect.height) return null;
  const scale = Math.min(rect.width / naturalWidth, rect.height / naturalHeight);
  const renderedWidth = naturalWidth * scale;
  const renderedHeight = naturalHeight * scale;
  return {
    left: rect.left + (rect.width - renderedWidth) / 2,
    top: rect.top + (rect.height - renderedHeight) / 2,
    width: renderedWidth,
    height: renderedHeight,
    naturalWidth,
    naturalHeight,
  };
}

function previewPointFromEvent(event) {
  const metrics = previewImageMetrics();
  if (!metrics) return null;
  const relativeX = event.clientX - metrics.left;
  const relativeY = event.clientY - metrics.top;
  if (relativeX < 0 || relativeY < 0 || relativeX > metrics.width || relativeY > metrics.height) {
    return null;
  }
  return {
    x: Number((((relativeX / metrics.width) * metrics.naturalWidth)).toFixed(1)),
    y: Number((((relativeY / metrics.height) * metrics.naturalHeight)).toFixed(1)),
  };
}

function interactDragDistance(fromPoint, toPoint) {
  if (!fromPoint || !toPoint) return 0;
  return Math.hypot(Number(toPoint.x) - Number(fromPoint.x), Number(toPoint.y) - Number(fromPoint.y));
}

function handlePreviewPointerDown(event) {
  if (!interactReady() || interactPointerBusy() || event.button !== 0) return;
  const point = previewPointFromEvent(event);
  if (!point) return;
  event.preventDefault();
  focusInteractSurface();
  event.currentTarget?.setPointerCapture?.(event.pointerId);
  state.ui.interactDrag = {
    pointerId: event.pointerId,
    startPoint: point,
    lastPoint: point,
    moved: false,
  };
}

function handlePreviewPointerMove(event) {
  const drag = state.ui.interactDrag;
  if (!drag || drag.pointerId !== event.pointerId) return;
  const point = previewPointFromEvent(event);
  if (!point) return;
  drag.lastPoint = point;
  if (!drag.moved && interactDragDistance(drag.startPoint, point) >= 12) {
    drag.moved = true;
  }
}

async function handlePreviewPointerUp(event) {
  const drag = state.ui.interactDrag;
  if (!drag || drag.pointerId !== event.pointerId) return;
  const point = previewPointFromEvent(event) || drag.lastPoint || drag.startPoint;
  state.ui.interactDrag = null;
  event.currentTarget?.releasePointerCapture?.(event.pointerId);
  if (!drag.moved || !point || interactDragDistance(drag.startPoint, point) < 12) {
    return;
  }
  event.preventDefault();
  state.ui.interactSuppressClickUntil = Date.now() + 260;
  focusInteractSurface();
  await flushInteractTextBuffer();
  const tool = currentInteractTool();
  const args = tool === "computer"
    ? {
        action: "drag",
        x: drag.startPoint.x,
        y: drag.startPoint.y,
        end_x: point.x,
        end_y: point.y,
      }
    : {
        action: "drag",
        start_x: drag.startPoint.x,
        start_y: drag.startPoint.y,
        end_x: point.x,
        end_y: point.y,
        steps: 18,
      };
  await sendManualAction(tool, args);
  focusInteractSurface();
}

function handlePreviewPointerCancel(event) {
  const drag = state.ui.interactDrag;
  if (!drag || drag.pointerId !== event.pointerId) return;
  state.ui.interactDrag = null;
  event.currentTarget?.releasePointerCapture?.(event.pointerId);
}

async function handlePreviewClick(event) {
  if (!interactReady() || interactPointerBusy()) return;
  if (Date.now() < Number(state.ui.interactSuppressClickUntil || 0)) {
    event.preventDefault();
    return;
  }
  const point = previewPointFromEvent(event);
  if (!point) return;
  event.preventDefault();
  state.ui.interactSuppressClickUntil = Date.now() + 320;
  focusInteractSurface();
  await flushInteractTextBuffer();
  await sendManualAction(currentInteractTool(), {
    action: "click",
    x: point.x,
    y: point.y,
  });
  focusInteractSurface();
}

async function handlePreviewWheel(event) {
  if (!interactReady() || interactPointerBusy()) return;
  const point = previewPointFromEvent(event);
  if (!point) return;
  event.preventDefault();
  focusInteractSurface();
  await flushInteractTextBuffer();
  const amount = Math.round(event.deltaY || 600) || 600;
  const tool = currentInteractTool();
  const args = tool === "computer"
    ? {
        action: "scroll",
        x: point.x,
        y: point.y,
        delta_y: amount,
      }
    : {
        action: "scroll",
        x: point.x,
        y: point.y,
        amount,
      };
  await sendManualAction(tool, args);
  focusInteractSurface();
}

function handlePreviewKeydown(event) {
  if (!interactReady() || isAgentraEditableTarget(event.target)) return;
  if (isPrintableKeyEvent(event)) {
    event.preventDefault();
    queueInteractText(event.key);
    return;
  }
  if (event.ctrlKey || event.metaKey || event.altKey || event.isComposing) return;
  if (!isInteractSpecialKey(event.key)) return;
  event.preventDefault();
  if (event.key === "Backspace" && state.ui.interactBuffer) {
    state.ui.interactBuffer = state.ui.interactBuffer.slice(0, -1);
    if (state.ui.interactBuffer) {
      clearInteractFlushTimer();
      state.ui.interactFlushTimer = window.setTimeout(() => {
        void flushInteractTextBuffer();
      }, 120);
    } else {
      clearInteractFlushTimer();
    }
    render();
    return;
  }
  void sendInteractKey(event.key);
}

function handlePreviewPaste(event) {
  if (!interactReady() || isAgentraEditableTarget(event.target)) return;
  const text = event.clipboardData?.getData("text") || "";
  if (!text) return;
  event.preventDefault();
  queueInteractText(text.replace(/\\r\\n/g, "\\n"));
}

function openReport() {
  if (!state.reportUrl) return;
  window.open(state.reportUrl, "_blank", "noreferrer");
}

function currentLogsUrl() {
  const params = new URLSearchParams();
  if (state.activeThreadId) params.set("thread_id", state.activeThreadId);
  if (state.selectedRunId) params.set("run_id", state.selectedRunId);
  const query = params.toString();
  return query ? `/logs?${query}` : "/logs";
}

function openLogs() {
  window.open(currentLogsUrl(), "_blank", "noreferrer");
}

async function openRunHistory(runId) {
  resetInteractState();
  state.historyMode = true;
  state.liveMode = false;
  await loadRun(runId, { live: false, history: true });
}

async function returnToLiveRun() {
  if (!state.activeThreadId) return;
  resetInteractState();
  state.historyMode = false;
  state.liveMode = true;
  await loadThreadDetail(state.activeThreadId, { preferLive: true });
}

function handleConsoleInput(event) {
  const target = event.target;
  if (target.id === "goal-input") state.drafts.goal = target.value;
  if (target.id === "thread-title-input") state.drafts.threadTitle = target.value;
  if (target.id === "permission-mode-input") state.drafts.permissionMode = target.value;

  const questionId = target.getAttribute("data-question-input");
  if (questionId) state.drafts.questionAnswers[questionId] = target.value;

  const manualKey = target.getAttribute("data-manual-input");
  if (manualKey) state.drafts[manualKey] = target.value;
}

async function handleConsoleClick(event) {
  const uiAction = event.target.closest("[data-ui-action]");
  if (uiAction) {
    const action = uiAction.getAttribute("data-ui-action");
    if (action === "toggle-advanced") {
      state.ui.showAdvanced = !state.ui.showAdvanced;
      render();
    }
    return;
  }

  const threadSelect = event.target.closest("[data-thread-select]");
  if (threadSelect) {
    clearThreadFocusPreference();
    resetInteractState();
    state.historyMode = false;
    state.liveMode = true;
    const threadId = threadSelect.getAttribute("data-thread-select");
    const summaryThread = state.threads.find((item) => item.thread_id === threadId) || null;
    if (summaryThread) {
      applyThreadSummary(summaryThread, { preferLive: true, forceDetail: true });
      render();
    }
    await loadThreadDetail(threadId, { preferLive: true });
    return;
  }

  const threadAction = event.target.closest("[data-thread-action]");
  if (threadAction) {
    const action = threadAction.getAttribute("data-thread-action");
    if (action === "clear-all") return clearAllThreads();
    if (action === "pause") return pauseActiveThread();
    if (action === "resume") return resumeActiveThread();
    if (action === "save-permission") return updateThreadPermissionMode();
    if (action === "report") return openReport();
    if (action === "logs") return openLogs();
    if (action === "return-live") return returnToLiveRun();
  }

  const approvalAction = event.target.closest("[data-approval-id]");
  if (approvalAction) {
    const requestId = approvalAction.getAttribute("data-approval-id");
    const approved = approvalAction.getAttribute("data-approved") === "true";
    return respondApproval(requestId, approved);
  }

  const questionAction = event.target.closest("[data-question-submit]");
  if (questionAction) {
    return submitQuestionAnswer(questionAction.getAttribute("data-question-submit"));
  }

  const historyAction = event.target.closest("[data-history-open]");
  if (historyAction) {
    return openRunHistory(historyAction.getAttribute("data-history-open"));
  }

  const manualAction = event.target.closest("[data-manual-action]");
  if (manualAction) {
    return handleManualAction(manualAction.getAttribute("data-manual-action"));
  }
}

function mount() {
  document.getElementById("app").innerHTML = `
    <main class="page">
      <header class="topbar">
        <div class="topbar-left">
          <button class="nav-button" type="button" aria-label="Geri">←</button>
          <div class="title-stack">
            <div class="app-title">Ajan</div>
            <div class="app-subtitle" id="project-stamp"></div>
          </div>
        </div>
      </header>
      <section class="workspace">
        <section class="left-pane">
          <div class="left-pane-inner">
            <section class="console-section">
              <div class="section-head">
                <div class="section-title">Yeni Run</div>
                <button id="advanced-toggle" class="ghost-button" type="button" data-ui-action="toggle-advanced">Advanced</button>
              </div>
              <label class="field-label" for="goal-input">Görev</label>
              <textarea id="goal-input" class="text-area" placeholder="Yeni bir komut veya görev girin"></textarea>
              <label class="field-label" for="thread-title-input">Thread başlığı (opsiyonel)</label>
              <input id="thread-title-input" class="text-input" type="text" placeholder="Yeni thread başlığı" />
              <label class="field-label" for="permission-mode-input">İzin modu</label>
              <select id="permission-mode-input" class="text-input">
                <option value="default">Default</option>
                <option value="full">Full</option>
              </select>
              <div class="helper-text">Full mod Chrome profilini ve kurulu uygulamalari acabilir; hassas browser adimlarinda kontrol sana devredilir, geri alinmaz islemler yine onay ister.</div>
              <div class="action-row">
                <button id="new-thread-button" class="action-button primary" type="button">Yeni thread'de başlat</button>
                <button id="append-thread-button" class="action-button" type="button">Seçili thread'e ekle</button>
              </div>
              <div class="inline-error" id="run-error"></div>
            </section>

            <section class="console-section" hidden>
              <div class="section-head">
                <div class="section-title">Threadler</div>
                <button type="button" class="ghost-button" data-thread-action="clear-all">Tümünü Temizle</button>
              </div>
              <div id="thread-list" class="thread-list"></div>
            </section>

            <section class="console-section" hidden>
              <div class="section-title">Seçili Thread</div>
              <div id="selected-thread-panel"></div>
              <div class="inline-error" id="thread-error"></div>
            </section>

            <section class="console-section" hidden>
              <div class="section-title">Bekleyen Onaylar</div>
              <div id="approval-list" class="request-list"></div>
              <div class="inline-error" id="approval-error"></div>
            </section>

            <section class="console-section" hidden>
              <div class="section-title">Bekleyen Sorular</div>
              <div id="question-list" class="request-list"></div>
              <div class="inline-error" id="question-error"></div>
            </section>

            <section class="console-section" hidden>
              <div class="section-title">Run Geçmişi</div>
              <div id="run-history" class="history-list"></div>
            </section>
          </div>
        </section>

        <section class="right-pane">
          <div class="stage-shell">
            <div class="stage-head">
              <div class="stage-title" id="agent-title">Ajan</div>
            </div>
            <div class="tv-stage">
              <div class="tv-shell">
                <div class="tv-screen" id="tv-screen">
                  <div class="tv-title" id="tv-frame-title">Ajan · HAZIR</div>
                  <img class="tv-image" id="tv-image" alt="Agentra canlı kare" decoding="async" fetchpriority="high" />
                  <div class="tv-empty" id="tv-empty"></div>
                  <div class="cursor-dot" id="cursor-dot"></div>
                  <div class="cursor-bubble" id="cursor-bubble"></div>
                </div>
                <div class="tv-footer">
                  <div class="scrub-row">
                    <span class="scrub-meta" id="scrubber-label">0 / 0</span>
                    <div class="timeline-track" id="timeline-track" aria-label="Zaman çizelgesi">
                      <div class="timeline-progress" id="timeline-progress"></div>
                      <div class="timeline-handle" id="timeline-handle"></div>
                    </div>
                    <span id="tv-status" style="font-size:11px; letter-spacing:0.04em;">HAZIR</span>
                    <span style="font-size:11px; letter-spacing:0.04em; opacity:0.92;">CANLI</span>
                  </div>
                </div>
                <div id="stage-manual-dock"></div>
              </div>
            </div>
          </div>
        </section>
      </section>
    </main>
  `;

  document.getElementById("new-thread-button").addEventListener("click", () => startRun("new"));
  document.getElementById("append-thread-button").addEventListener("click", () => startRun("append"));
  document.getElementById("advanced-toggle").addEventListener("click", handleConsoleClick);
  document.getElementById("goal-input").addEventListener("input", handleConsoleInput);
  document.getElementById("thread-title-input").addEventListener("input", handleConsoleInput);
  document.getElementById("permission-mode-input").addEventListener("input", handleConsoleInput);
  document.getElementById("permission-mode-input").addEventListener("change", handleConsoleInput);
  document.getElementById("thread-list").addEventListener("click", handleConsoleClick);
  document.getElementById("selected-thread-panel").addEventListener("click", handleConsoleClick);
  document.getElementById("approval-list").addEventListener("click", handleConsoleClick);
  document.getElementById("question-list").addEventListener("click", handleConsoleClick);
  document.getElementById("question-list").addEventListener("input", handleConsoleInput);
  document.getElementById("stage-manual-dock").addEventListener("click", handleConsoleClick);
  document.getElementById("stage-manual-dock").addEventListener("input", handleConsoleInput);
  document.getElementById("run-history").addEventListener("click", handleConsoleClick);
  document.getElementById("tv-image").addEventListener("load", handleLiveImageLoad);
  document.getElementById("tv-image").addEventListener("error", handleLiveImageError);
  document.getElementById("tv-image").addEventListener("pointerdown", handlePreviewPointerDown);
  document.getElementById("tv-image").addEventListener("pointermove", handlePreviewPointerMove);
  document.getElementById("tv-image").addEventListener("pointerup", (event) => { void handlePreviewPointerUp(event); });
  document.getElementById("tv-image").addEventListener("pointercancel", handlePreviewPointerCancel);
  document.getElementById("tv-image").addEventListener("click", handlePreviewClick);
  document.getElementById("tv-image").addEventListener("wheel", handlePreviewWheel, { passive: false });
  window.addEventListener("keydown", handlePreviewKeydown);
  window.addEventListener("paste", handlePreviewPaste);

  setupScrubber();
  render();
  refreshThreads({ preferLive: true });
  scheduleThreadPolling();
}

window.addEventListener("beforeunload", () => {
  disconnectStream();
  stopLiveMirror();
  if (state.threadPollTimer) window.clearInterval(state.threadPollTimer);
});

mount();
"""
