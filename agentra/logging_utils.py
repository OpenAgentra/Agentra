"""Logging helpers for the Agentra CLI and live app."""

from __future__ import annotations

import logging
import traceback
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

APP_LOG_DIRNAME = ".logs"
APP_LOG_FILENAME = "agentra-app.log"
APP_LOG_HANDLER_NAME = "agentra-file-log"


def app_log_dir(workspace_dir: Path) -> Path:
    """Return the directory that stores persistent Agentra logs."""
    return workspace_dir / APP_LOG_DIRNAME


def app_log_path(workspace_dir: Path) -> Path:
    """Return the main rotating application log path."""
    return app_log_dir(workspace_dir) / APP_LOG_FILENAME


def configure_app_logging(workspace_dir: Path) -> Path:
    """Attach a rotating file handler once and return the log file path."""
    log_path = app_log_path(workspace_dir).resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.touch(exist_ok=True)

    root_logger = logging.getLogger()
    target_name = str(log_path)
    for handler in list(root_logger.handlers):
        if getattr(handler, "name", "") != APP_LOG_HANDLER_NAME:
            continue
        if getattr(handler, "baseFilename", "") == target_name:
            return log_path
        root_logger.removeHandler(handler)
        handler.close()

    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=2_000_000,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.set_name(APP_LOG_HANDLER_NAME)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)s [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    root_logger.addHandler(file_handler)

    logging.getLogger("agentra").setLevel(logging.DEBUG)
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("uvicorn.error").setLevel(logging.INFO)
    logging.getLogger("uvicorn.access").setLevel(logging.INFO)
    return log_path


def exception_details(
    exc: BaseException,
    *,
    provider: str | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    """Return traceback details plus provider-specific classification when available."""
    details = {
        "exception_type": type(exc).__name__,
        "message": str(exc),
        "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
    }
    details.update(_provider_error_details(exc, provider=provider, model=model))
    return details


def exception_details_with_context(
    exc: BaseException,
    *,
    provider: str | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    """Backward-compatible alias for provider-aware exception details."""
    return exception_details(exc, provider=provider, model=model)


def _provider_error_details(
    exc: BaseException,
    *,
    provider: str | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    raw_payload = getattr(exc, "details", None)
    payload = raw_payload if isinstance(raw_payload, dict) else {}
    error_payload = payload.get("error") if isinstance(payload.get("error"), dict) else payload

    status_code = _coerce_int(getattr(exc, "status_code", None))
    if status_code is None:
        status_code = _coerce_int(getattr(exc, "code", None))
    if status_code is None and isinstance(error_payload, dict):
        status_code = _coerce_int(error_payload.get("code"))

    provider_status = _compact_text(getattr(exc, "status", None))
    provider_message = _compact_text(getattr(exc, "message", None))
    if isinstance(error_payload, dict):
        provider_status = provider_status or _compact_text(error_payload.get("status"))
        provider_message = provider_message or _compact_text(error_payload.get("message"))

    message_text = " ".join(
        part
        for part in (
            _compact_text(str(exc)),
            provider_status,
            provider_message,
        )
        if part
    ).lower()

    public_message = ""
    hint = ""
    error_kind = ""
    retryable: bool | None = None
    provider_name = _provider_name(provider)
    target = f"model {model}" if model else "the current model"

    quota_markers = ("exceeded your current quota", "current quota", "billing details", "check your plan")
    auth_markers = ("invalid api key", "api key", "unauthorized", "permission denied")
    model_markers = ("model not found", "unknown model", "unsupported model")
    transient_markers = (
        "server disconnected",
        "connection reset",
        "connection closed",
        "temporarily unavailable",
        "upstream connect error",
        "remote host closed",
    )

    if status_code == 429 or "resource_exhausted" in message_text or "rate limit" in message_text:
        if any(marker in message_text for marker in quota_markers):
            error_kind = "quota_exceeded"
            retryable = False
            public_message = (
                f"{provider_name} quota exceeded for {target}. "
                "Add billing or wait for quota reset, then retry."
            )
            hint = (
                "Switch the thread to another provider/model, or add "
                f"{provider_name} billing/credits before retrying."
            )
        else:
            error_kind = "rate_limited"
            retryable = True
            public_message = (
                f"{provider_name} rate limit reached for {target}. "
                "Wait a moment and try again."
            )
            hint = "Retry after a short pause, or reduce concurrent runs sharing the same provider quota."
    elif status_code in {401, 403} or any(marker in message_text for marker in auth_markers):
        error_kind = "authentication_failed"
        retryable = False
        public_message = (
            f"{provider_name} rejected the request for {target}. "
            "Check the API key and permissions, then retry."
        )
        hint = f"Verify the configured {provider_name} credentials and that the selected model is enabled."
    elif status_code == 404 or any(marker in message_text for marker in model_markers):
        error_kind = "model_unavailable"
        retryable = False
        public_message = (
            f"{provider_name} could not find or access {target}. "
            "Choose a different model and retry."
        )
        hint = "Update the configured model name or switch to a model your account can access."
    elif any(marker in message_text for marker in transient_markers):
        error_kind = "provider_unavailable"
        retryable = True
        public_message = (
            f"{provider_name} connection dropped while contacting {target}. "
            "Try again shortly."
        )
        hint = "Retry after a short delay, or switch providers if the connection keeps dropping."
    elif status_code is not None and status_code >= 500:
        error_kind = "provider_unavailable"
        retryable = True
        public_message = (
            f"{provider_name} is temporarily unavailable for {target}. "
            "Try again shortly."
        )
        hint = "Retry after a short delay, or switch providers if the outage persists."

    if not public_message:
        return {}

    details = {
        "provider": provider,
        "model": model,
        "status_code": status_code,
        "provider_status": provider_status,
        "provider_message": provider_message,
        "error_kind": error_kind,
        "retryable": retryable,
        "public_message": public_message,
        "hint": hint,
    }
    return {
        key: value
        for key, value in details.items()
        if value not in (None, "")
    }


def _provider_name(provider: str | None) -> str:
    labels = {
        "openai": "OpenAI",
        "anthropic": "Anthropic",
        "gemini": "Gemini",
        "ollama": "Ollama",
    }
    key = str(provider or "").strip().lower()
    if not key:
        return "The provider"
    return labels.get(key, key.replace("_", " ").title())


def _compact_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def _coerce_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def read_log_tail(path: Path, *, max_lines: int = 400) -> str:
    """Read the tail of a UTF-8 text file safely."""
    if not path.exists():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-max(1, max_lines):])
