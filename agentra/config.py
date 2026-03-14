"""Configuration management for Agentra."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal, Optional

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AgentConfig(BaseSettings):
    """Runtime configuration resolved from environment variables and/or explicit kwargs."""

    model_config = SettingsConfigDict(env_prefix="AGENTRA_", env_file=".env", extra="ignore")

    # ── LLM provider ──────────────────────────────────────────────────────────
    llm_provider: Literal["openai", "anthropic", "ollama", "gemini"] = Field(
        default="openai",
        description="Which LLM back-end to use.",
    )
    llm_model: str = Field(
        default="gpt-4o",
        description="Model name passed to the provider.",
    )
    llm_vision_model: Optional[str] = Field(
        default=None,
        description="Separate vision model (defaults to llm_model when None).",
    )

    # ── Provider credentials ───────────────────────────────────────────────────
    openai_api_key: Optional[str] = Field(default=None)
    anthropic_api_key: Optional[str] = Field(default=None)
    ollama_base_url: str = Field(default="http://localhost:11434")
    gemini_api_key: Optional[str] = Field(default=None)
    gemini_base_url: str = Field(
        default="https://generativelanguage.googleapis.com/v1beta/openai/"
    )

    # ── Agent behaviour ────────────────────────────────────────────────────────
    max_iterations: int = Field(default=50, ge=1, le=500)
    max_tokens: int = Field(default=4096, ge=1)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)

    # ── Workspace / memory ─────────────────────────────────────────────────────
    workspace_dir: Path = Field(
        default_factory=lambda: Path(os.getcwd()) / "workspace",
        description="Directory used as the agent's own git-tracked workspace.",
    )
    memory_dir: Path = Field(
        default_factory=lambda: Path(os.getcwd()) / "workspace" / ".memory",
        description="Directory used to persist embeddings and screenshots.",
    )
    screenshot_history: int = Field(
        default=10,
        ge=0,
        description="Number of recent screenshots kept in memory context.",
    )

    # ── Browser ────────────────────────────────────────────────────────────────
    browser_headless: bool = Field(default=False)
    browser_type: Literal["chromium", "firefox", "webkit"] = Field(default="chromium")

    # ── Safety ────────────────────────────────────────────────────────────────
    allow_terminal: bool = Field(default=True)
    allow_filesystem_write: bool = Field(default=True)
    allow_computer_control: bool = Field(default=True)

    @field_validator("workspace_dir", "memory_dir", mode="before")
    @classmethod
    def _expand_path(cls, v: object) -> Path:
        return Path(str(v)).expanduser().resolve()

    @model_validator(mode="after")
    def _apply_provider_defaults(self) -> "AgentConfig":
        if self.llm_provider == "gemini" and self.llm_model == "gpt-4o":
            self.llm_model = "gemini-3-flash-preview"
        return self

    @property
    def vision_model(self) -> str:
        return self.llm_vision_model or self.llm_model
