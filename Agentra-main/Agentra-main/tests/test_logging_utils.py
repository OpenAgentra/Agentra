"""Tests for provider-aware exception formatting."""

from __future__ import annotations

import aiohttp
from google.genai.errors import ClientError

from agentra.logging_utils import exception_details_with_context


def test_exception_details_with_context_normalizes_gemini_quota_errors() -> None:
    exc = ClientError(
        429,
        {
            "error": {
                "code": 429,
                "message": (
                    "You exceeded your current quota, please check your plan and billing details."
                ),
                "status": "RESOURCE_EXHAUSTED",
            }
        },
        None,
    )

    details = exception_details_with_context(
        exc,
        provider="gemini",
        model="gemini-3-flash-preview",
    )

    assert details["status_code"] == 429
    assert details["provider_status"] == "RESOURCE_EXHAUSTED"
    assert details["error_kind"] == "quota_exceeded"
    assert details["retryable"] is False
    assert details["public_message"] == (
        "Gemini quota exceeded for model gemini-3-flash-preview. "
        "Add billing or wait for quota reset, then retry."
    )
    assert details["hint"] == (
        "Switch the thread to another provider/model, or add Gemini "
        "billing/credits before retrying."
    )


def test_exception_details_with_context_marks_server_disconnect_retryable() -> None:
    details = exception_details_with_context(
        aiohttp.ServerDisconnectedError(),
        provider="gemini",
        model="gemini-2.5-flash",
    )

    assert details["error_kind"] == "provider_unavailable"
    assert details["retryable"] is True
    assert details["public_message"] == (
        "Gemini connection dropped while contacting model gemini-2.5-flash. "
        "Try again shortly."
    )
    assert details["hint"] == (
        "Retry after a short delay, or switch providers if the connection keeps dropping."
    )
