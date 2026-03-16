"""Tests for the rule-based approval engine."""

from __future__ import annotations

from agentra.approval_policy import ApprovalPolicyContext, ApprovalPolicyEngine


def test_default_policy_requires_approval_for_secret_browser_entry() -> None:
    engine = ApprovalPolicyEngine.default()

    decision = engine.evaluate(
        ApprovalPolicyContext(
            tool_name="browser",
            tool_args={"action": "type", "selector": "#password", "text": "secret"},
        )
    )

    assert decision.action == "require_approval"
    assert decision.rule_id == "browser-auth-secret-entry"
    assert decision.risk_level == "high"


def test_default_policy_allows_safe_browser_navigation() -> None:
    engine = ApprovalPolicyEngine.default()

    decision = engine.evaluate(
        ApprovalPolicyContext(
            tool_name="browser",
            tool_args={"action": "navigate", "url": "https://www.python.org"},
        )
    )

    assert decision.action == "allow"
