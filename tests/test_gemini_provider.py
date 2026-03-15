"""Tests for Gemini provider integration."""

from __future__ import annotations

from unittest.mock import patch

from agentra.config import AgentConfig
from agentra.llm.factory import get_provider
from agentra.llm.gemini_provider import GeminiProvider


def test_gemini_provider_sets_default_model():
    cfg = AgentConfig(_env_file=None, llm_provider="gemini")
    assert cfg.llm_model == "gemini-3-flash-preview"


def test_factory_returns_gemini_provider():
    with patch("openai.AsyncOpenAI") as async_client:
        cfg = AgentConfig(
            _env_file=None,
            llm_provider="gemini",
            gemini_api_key="test-key",
            gemini_base_url="https://example.invalid/openai/",
        )
        provider = get_provider(cfg)

    assert isinstance(provider, GeminiProvider)
    async_client.assert_called_once_with(
        api_key="test-key",
        base_url="https://example.invalid/openai/",
    )
