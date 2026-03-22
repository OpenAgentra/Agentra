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


def exception_details(exc: BaseException) -> dict[str, Any]:
    """Return a JSON-friendly exception payload with traceback text."""
    return {
        "exception_type": type(exc).__name__,
        "message": str(exc),
        "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
    }


def read_log_tail(path: Path, *, max_lines: int = 400) -> str:
    """Read the tail of a UTF-8 text file safely."""
    if not path.exists():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-max(1, max_lines):])
