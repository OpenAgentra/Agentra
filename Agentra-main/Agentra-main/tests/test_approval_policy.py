"""Tests for the rule-based approval engine."""

from __future__ import annotations

from agentra.approval_policy import (
    REDACTED_PLACEHOLDER,
    ApprovalPolicyContext,
    ApprovalPolicyEngine,
    browser_sensitive_takeover_kind,
    content_requests_sensitive_input,
    redact_tool_args_for_storage,
)


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


def test_default_policy_allows_safe_desktop_navigation_key_for_local_goal() -> None:
    engine = ApprovalPolicyEngine.default()

    decision = engine.evaluate(
        ApprovalPolicyContext(
            tool_name="computer",
            tool_args={"action": "key", "text": "win+d"},
            goal="masaüstüme git ve secondsun klasörünü aç",
        )
    )

    assert decision.action == "allow"


def test_default_policy_requires_approval_for_risky_desktop_shortcut() -> None:
    engine = ApprovalPolicyEngine.default()

    decision = engine.evaluate(
        ApprovalPolicyContext(
            tool_name="computer",
            tool_args={"action": "key", "text": "alt+f4"},
            goal="masaüstüme git ve secondsun klasörünü aç",
        )
    )

    assert decision.action == "require_approval"
    assert decision.rule_id == "computer-direct-control"


def test_full_mode_allows_non_sensitive_browser_typing() -> None:
    engine = ApprovalPolicyEngine.default()

    decision = engine.evaluate(
        ApprovalPolicyContext(
            tool_name="browser",
            tool_args={"action": "type", "selector": "input[name='q']", "text": "agentra github"},
            permission_mode="full",
        )
    )

    assert decision.action == "allow"


def test_full_mode_still_requires_approval_for_secret_entry() -> None:
    engine = ApprovalPolicyEngine.default()

    decision = engine.evaluate(
        ApprovalPolicyContext(
            tool_name="browser",
            tool_args={"action": "type", "selector": "#password", "text": "secret"},
            permission_mode="full",
        )
    )

    assert decision.action == "require_approval"
    assert decision.rule_id == "sensitive-secret-entry"


def test_full_mode_requires_approval_for_goal_only_secret_click() -> None:
    engine = ApprovalPolicyEngine.default()

    decision = engine.evaluate(
        ApprovalPolicyContext(
            tool_name="browser",
            tool_args={"action": "click", "selector": "button.login"},
            goal="GitHub giris sayfasini ac ve sifre alanina parolami yazmayi dene.",
            permission_mode="full",
        )
    )

    assert decision.action == "require_approval"
    assert decision.rule_id == "sensitive-secret-entry"


def test_browser_sensitive_takeover_kind_detects_auth_submit_click() -> None:
    assert (
        browser_sensitive_takeover_kind(
            {"action": "click", "selector": "button.login"},
            goal="GitHub giris sayfasini ac ve sifre alanina parolami yazmayi dene.",
            permission_mode="full",
        )
        == "secret"
    )


def test_browser_sensitive_takeover_kind_detects_payment_entry() -> None:
    assert (
        browser_sensitive_takeover_kind(
            {"action": "type", "selector": "#card-number", "text": "4242 4242 4242 4242"},
            permission_mode="full",
        )
        == "payment"
    )


def test_full_mode_still_requires_approval_for_payment_entry() -> None:
    engine = ApprovalPolicyEngine.default()

    decision = engine.evaluate(
        ApprovalPolicyContext(
            tool_name="browser",
            tool_args={"action": "type", "selector": "#card-number", "text": "4242 4242 4242 4242"},
            permission_mode="full",
        )
    )

    assert decision.action == "require_approval"
    assert decision.rule_id == "payment-information-entry"


def test_full_mode_requires_approval_for_goal_only_payment_typing() -> None:
    engine = ApprovalPolicyEngine.default()

    decision = engine.evaluate(
        ApprovalPolicyContext(
            tool_name="browser",
            tool_args={"action": "type", "selector": "#card-number", "text": "test data"},
            goal="Bir odeme sayfasinda kredi karti bilgilerimi girmeyi dene.",
            permission_mode="full",
        )
    )

    assert decision.action == "require_approval"
    assert decision.rule_id == "payment-information-entry"


def test_full_mode_still_requires_approval_for_irreversible_transaction_click() -> None:
    engine = ApprovalPolicyEngine.default()

    decision = engine.evaluate(
        ApprovalPolicyContext(
            tool_name="browser",
            tool_args={"action": "click", "selector": "button.checkout"},
            permission_mode="full",
        )
    )

    assert decision.action == "require_approval"
    assert decision.rule_id == "transaction-or-account-destruction"


def test_full_mode_requires_approval_for_goal_only_confirm_purchase_click() -> None:
    engine = ApprovalPolicyEngine.default()

    decision = engine.evaluate(
        ApprovalPolicyContext(
            tool_name="browser",
            tool_args={"action": "click", "selector": "button.confirm"},
            goal="Bu satin alma islemini tamamla ve odemeyi onayla.",
            permission_mode="full",
        )
    )

    assert decision.action == "require_approval"
    assert decision.rule_id == "transaction-or-account-destruction"


def test_full_mode_allows_login_page_navigation_before_sensitive_step() -> None:
    engine = ApprovalPolicyEngine.default()

    decision = engine.evaluate(
        ApprovalPolicyContext(
            tool_name="browser",
            tool_args={"action": "navigate", "url": "https://github.com/login"},
            goal="GitHub giris sayfasini ac ve sifre alanina parolami yazmayi dene.",
            permission_mode="full",
        )
    )

    assert decision.action == "allow"


def test_redact_tool_args_for_storage_masks_sensitive_browser_text() -> None:
    sanitized = redact_tool_args_for_storage(
        "browser",
        {"action": "type", "selector": "input[name='password']", "text": "Kaan123"},
        goal="GitHub giris sayfasini ac ve sifre alanina parolami yazmayi dene.",
        permission_mode="full",
    )

    assert sanitized["text"] == REDACTED_PLACEHOLDER


def test_redact_tool_args_for_storage_keeps_safe_browser_text() -> None:
    sanitized = redact_tool_args_for_storage(
        "browser",
        {"action": "type", "selector": "input[name='q']", "text": "agentra github"},
        permission_mode="full",
    )

    assert sanitized["text"] == "agentra github"


def test_content_requests_sensitive_input_detects_plaintext_secret_prompts() -> None:
    assert content_requests_sensitive_input("Parolanizi girmemi onayliyor musunuz?")
    assert not content_requests_sensitive_input("Hangi repo adini kullanayim?")
