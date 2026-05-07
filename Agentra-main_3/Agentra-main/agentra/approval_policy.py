"""Rule-based approval policy engine for risky tool actions."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Callable

_SECRET_INTENT_TERMS = (
    "password",
    "passcode",
    "parola",
    "sifre",
    "otp",
    "2fa",
    "mfa",
    "verification code",
    "dogrulama kodu",
    "recovery code",
    "kurtarma kodu",
    "authenticator",
    "secret",
    "token",
    "captcha",
)
_SECRET_SURFACE_TERMS = (
    "password",
    "passcode",
    "parola",
    "sifre",
    "otp",
    "2fa",
    "mfa",
    "verification",
    "dogrulama",
    "authenticator",
    "recovery",
    "kurtarma",
    "captcha",
    "secret",
    "token",
)
_SECRET_COMMIT_TERMS = (
    "login",
    "log in",
    "sign in",
    "submit",
    "continue",
    "verify",
    "confirm",
    "giris yap",
    "oturum ac",
    "devam et",
    "dogrula",
    "onayla",
)
_PAYMENT_INTENT_TERMS = (
    "payment",
    "odeme",
    "credit card",
    "debit card",
    "kredi kart",
    "card number",
    "card-number",
    "cardnumber",
    "cardholder",
    "cvv",
    "cvc",
    "expiry",
    "expiration",
    "billing",
    "iban",
    "swift",
)
_PAYMENT_SURFACE_TERMS = (
    "payment",
    "odeme",
    "credit card",
    "debit card",
    "kredi kart",
    "card number",
    "card-number",
    "cardnumber",
    "cardholder",
    "cvv",
    "cvc",
    "expiry",
    "expiration",
    "billing",
    "iban",
    "swift",
)
_PAYMENT_COMMIT_TERMS = (
    "checkout",
    "pay",
    "purchase",
    "buy",
    "order",
    "place order",
    "confirm payment",
    "odeme yap",
    "satin al",
    "siparisi tamamla",
)
_IRREVERSIBLE_INTENT_TERMS = (
    "checkout",
    "purchase",
    "buy",
    "subscribe",
    "order",
    "place order",
    "confirm purchase",
    "confirm payment",
    "send money",
    "transfer",
    "wire",
    "withdraw",
    "delete account",
    "close account",
    "remove account",
    "terminate account",
    "hesabi sil",
    "hesabi kapat",
    "abonelik",
    "satin al",
    "odeme yap",
)
_IRREVERSIBLE_SURFACE_TERMS = (
    "checkout",
    "purchase",
    "buy",
    "subscribe",
    "order",
    "pay",
    "confirm",
    "delete",
    "remove",
    "close account",
    "delete account",
    "terminate account",
    "send money",
    "transfer",
    "wire",
    "withdraw",
    "hesabi sil",
    "hesabi kapat",
    "satin al",
    "odeme yap",
    "onayla",
)
_BROWSER_SIDE_EFFECT_TERMS = (
    "submit",
    "post",
    "send",
    "share",
    "publish",
    "apply",
    "checkout",
    "pay",
    "confirm",
    "delete",
    "remove",
)
_SENSITIVE_COMMIT_KEYS = {
    "enter",
    "return",
    "ctrl+v",
    "control+v",
    "meta+v",
    "cmd+v",
    "command+v",
}
REDACTED_PLACEHOLDER = "[REDACTED]"


@dataclass(frozen=True)
class ApprovalPolicyContext:
    """Context used to evaluate whether a tool call requires approval."""

    tool_name: str
    tool_args: dict[str, Any]
    goal: str = ""
    thread_id: str | None = None
    run_id: str | None = None
    permission_mode: str = "default"


@dataclass(frozen=True)
class ApprovalDecision:
    """Structured result returned by the approval engine."""

    action: str  # "allow" | "require_approval"
    summary: str
    reason: str = ""
    rule_id: str | None = None
    risk_level: str = "low"


@dataclass(frozen=True)
class ApprovalSignals:
    """Normalized context facts reused by ordered approval rules."""

    context: ApprovalPolicyContext
    action: str
    goal: str
    payload: str
    selector: str
    text: str
    key: str


@dataclass(frozen=True)
class ApprovalRule:
    """Single ordered approval rule."""

    rule_id: str
    risk_level: str
    matcher: Callable[[ApprovalSignals], bool]
    reason: str
    summary: str

    def matches(self, signals: ApprovalSignals) -> bool:
        return bool(self.matcher(signals))


class ApprovalPolicyEngine:
    """Ordered rule engine for approval decisions."""

    def __init__(
        self,
        rules: list[ApprovalRule],
        *,
        full_mode_rules: list[ApprovalRule] | None = None,
    ) -> None:
        self._rules = list(rules)
        self._full_mode_rules = list(full_mode_rules or [])

    def evaluate(self, context: ApprovalPolicyContext) -> ApprovalDecision:
        rules = self._full_mode_rules if str(context.permission_mode or "").lower() == "full" else self._rules
        signals = _build_signals(context)
        for rule in rules:
            if rule.matches(signals):
                return ApprovalDecision(
                    action="require_approval",
                    summary=rule.summary,
                    reason=rule.reason,
                    rule_id=rule.rule_id,
                    risk_level=rule.risk_level,
                )
        return ApprovalDecision(action="allow", summary="")

    @classmethod
    def default(cls) -> "ApprovalPolicyEngine":
        return cls(
            [
                ApprovalRule(
                    rule_id="browser-auth-secret-entry",
                    risk_level="high",
                    matcher=lambda signals: signals.context.tool_name == "browser"
                    and _is_secret_or_auth_approval_step(signals),
                    reason="Secret, authentication, or verification data entry requires user approval.",
                    summary="Tarayıcıda gizli veya kimlik doğrulama bilgisi girilecek.",
                ),
                ApprovalRule(
                    rule_id="browser-external-side-effect",
                    risk_level="high",
                    matcher=lambda signals: signals.context.tool_name == "browser"
                    and _is_browser_external_side_effect_step(signals),
                    reason="This browser action may submit, send, publish, pay, or otherwise have external effects.",
                    summary="Tarayıcı eylemi dış etki üretebilir veya geri alınması zor bir işlem başlatabilir.",
                ),
                ApprovalRule(
                    rule_id="terminal-install-or-side-effect",
                    risk_level="high",
                    matcher=lambda signals: signals.context.tool_name == "terminal"
                    and _contains_any(
                        signals.context.tool_args,
                        (
                            "pip install",
                            "pip uninstall",
                            "npm install",
                            "npm publish",
                            "apt ",
                            "apt-get ",
                            "brew ",
                            "choco ",
                            "winget ",
                            "curl ",
                            "wget ",
                            "git push",
                            "shutdown",
                            "restart",
                            "rm ",
                            "rmdir ",
                            "del ",
                        ),
                    ),
                    reason="This terminal command may install software, call external systems, or perform destructive side effects.",
                    summary="Terminal komutu önemli yan etki veya kurulum işlemi içerebilir.",
                ),
                ApprovalRule(
                    rule_id="filesystem-destructive",
                    risk_level="medium",
                    matcher=lambda signals: signals.context.tool_name == "filesystem"
                    and _tool_action(signals.context) in {"delete", "move"},
                    reason="This filesystem action changes or removes existing files.",
                    summary="Dosya sistemi işlemi mevcut dosyaları değiştirebilir veya silebilir.",
                ),
                ApprovalRule(
                    rule_id="git-rewrite-or-clone",
                    risk_level="medium",
                    matcher=lambda signals: signals.context.tool_name == "git"
                    and _tool_action(signals.context) in {"reset", "checkout", "clone"},
                    reason="This git action can rewrite or materially change workspace state.",
                    summary="Git işlemi çalışma alanı durumunu önemli ölçüde değiştirebilir.",
                ),
                ApprovalRule(
                    rule_id="computer-direct-control",
                    risk_level="high",
                    matcher=lambda signals: signals.context.tool_name == "computer"
                    and _computer_action_requires_approval(signals.context),
                    reason="Direct desktop control requires explicit user approval.",
                    summary="Doğrudan masaüstü kontrolü için kullanıcı onayı gerekiyor.",
                ),
            ],
            full_mode_rules=[
                ApprovalRule(
                    rule_id="sensitive-secret-entry",
                    risk_level="high",
                    matcher=_is_secret_or_auth_approval_step,
                    reason="Secret or authentication information still requires explicit user approval in full mode.",
                    summary="Gizli veya kimlik doğrulama verisi girişi için kullanıcı onayı gerekiyor.",
                ),
                ApprovalRule(
                    rule_id="transaction-or-account-destruction",
                    risk_level="high",
                    matcher=_is_irreversible_external_action_step,
                    reason="Purchases, money movement, account deletion, or similarly irreversible external actions still require approval in full mode.",
                    summary="Satın alma, para hareketi veya geri alınamaz dış işlem için kullanıcı onayı gerekiyor.",
                ),
                ApprovalRule(
                    rule_id="payment-information-entry",
                    risk_level="high",
                    matcher=_is_payment_information_approval_step,
                    reason="Payment or credit card information still requires explicit user approval in full mode.",
                    summary="Odeme veya kredi karti bilgisi girişi için kullanıcı onayı gerekiyor.",
                ),
            ],
        )


def browser_sensitive_input_kind(
    tool_args: dict[str, Any],
    *,
    goal: str = "",
    permission_mode: str = "default",
) -> str | None:
    context = ApprovalPolicyContext(
        tool_name="browser",
        tool_args=tool_args,
        goal=goal,
        permission_mode=permission_mode,
    )
    signals = _build_signals(context)
    if signals.action != "type":
        return None
    if _is_secret_or_auth_approval_step(signals):
        return "secret"
    if _is_payment_information_approval_step(signals):
        return "payment"
    return None


def browser_sensitive_takeover_kind(
    tool_args: dict[str, Any],
    *,
    goal: str = "",
    permission_mode: str = "default",
) -> str | None:
    context = ApprovalPolicyContext(
        tool_name="browser",
        tool_args=tool_args,
        goal=goal,
        permission_mode=permission_mode,
    )
    signals = _build_signals(context)
    if _is_secret_or_auth_approval_step(signals):
        return "secret"
    if _is_payment_information_approval_step(signals):
        return "payment"
    return None


def redact_tool_args_for_storage(
    tool_name: str,
    tool_args: dict[str, Any],
    *,
    goal: str = "",
    permission_mode: str = "default",
) -> dict[str, Any]:
    sanitized = dict(tool_args)
    if tool_name != "browser":
        return sanitized
    if browser_sensitive_input_kind(
        sanitized,
        goal=goal,
        permission_mode=permission_mode,
    ) and str(sanitized.get("text", "")).strip():
        sanitized["text"] = REDACTED_PLACEHOLDER
    return sanitized


def content_requests_sensitive_input(text: str) -> bool:
    normalized = _normalized_text(text)
    if not normalized:
        return False
    if "?" not in normalized and not any(
        trigger in normalized
        for trigger in ("please provide", "can you provide", "girmemi onayliyor musun", "girmemi onaylıyor musun")
    ):
        return False
    return _contains_any_phrase(
        normalized,
        _SECRET_INTENT_TERMS + _PAYMENT_INTENT_TERMS,
    )


def _build_signals(context: ApprovalPolicyContext) -> ApprovalSignals:
    key_value = context.tool_args.get("key", context.tool_args.get("text", ""))
    payload = " ".join(_normalized_text(value) for value in context.tool_args.values() if value not in (None, ""))
    return ApprovalSignals(
        context=context,
        action=_tool_action(context),
        goal=_normalized_text(context.goal),
        payload=payload,
        selector=_normalized_text(context.tool_args.get("selector", "")),
        text=_normalized_text(context.tool_args.get("text", "")),
        key=_normalized_text(key_value),
    )


def _contains_any(payload: dict[str, Any], tokens: tuple[str, ...]) -> bool:
    flattened = " ".join(str(item).lower() for item in payload.values())
    return any(token in flattened for token in tokens)


def _contains_any_phrase(text: str, phrases: tuple[str, ...]) -> bool:
    return any(phrase in text for phrase in phrases)


def _tool_action(context: ApprovalPolicyContext) -> str:
    return str(context.tool_args.get("action", "")).lower()


def _tool_name(signals: ApprovalSignals) -> str:
    return str(signals.context.tool_name or "").lower()


def _is_sensitive_tool(signals: ApprovalSignals) -> bool:
    return _tool_name(signals) in {"browser", "computer"} and signals.action in {"click", "type", "key"}


def _goal_has_secret_intent(signals: ApprovalSignals) -> bool:
    return _contains_any_phrase(signals.goal, _SECRET_INTENT_TERMS)


def _goal_has_payment_intent(signals: ApprovalSignals) -> bool:
    return _contains_any_phrase(signals.goal, _PAYMENT_INTENT_TERMS)


def _goal_has_irreversible_intent(signals: ApprovalSignals) -> bool:
    return _contains_any_phrase(signals.goal, _IRREVERSIBLE_INTENT_TERMS)


def _payload_has_secret_indicator(signals: ApprovalSignals) -> bool:
    return _contains_any_phrase(signals.payload, _SECRET_INTENT_TERMS)


def _payload_has_payment_indicator(signals: ApprovalSignals) -> bool:
    return _contains_any_phrase(signals.payload, _PAYMENT_INTENT_TERMS)


def _payload_has_irreversible_indicator(signals: ApprovalSignals) -> bool:
    return _contains_any_phrase(signals.payload, _IRREVERSIBLE_SURFACE_TERMS)


def _target_looks_secret_surface(signals: ApprovalSignals) -> bool:
    return _contains_any_phrase(" ".join((signals.selector, signals.payload)), _SECRET_SURFACE_TERMS)


def _target_looks_secret_commit_surface(signals: ApprovalSignals) -> bool:
    return _contains_any_phrase(" ".join((signals.selector, signals.payload)), _SECRET_COMMIT_TERMS)


def _target_looks_payment_surface(signals: ApprovalSignals) -> bool:
    return _contains_any_phrase(" ".join((signals.selector, signals.payload)), _PAYMENT_SURFACE_TERMS)


def _target_looks_payment_commit_surface(signals: ApprovalSignals) -> bool:
    combined = " ".join((signals.selector, signals.payload))
    return _contains_any_phrase(combined, _PAYMENT_COMMIT_TERMS) or _contains_any_phrase(combined, _IRREVERSIBLE_SURFACE_TERMS)


def _target_looks_irreversible_surface(signals: ApprovalSignals) -> bool:
    return _contains_any_phrase(" ".join((signals.selector, signals.payload)), _IRREVERSIBLE_SURFACE_TERMS)


def _is_sensitive_commit_key(signals: ApprovalSignals) -> bool:
    return signals.key in _SENSITIVE_COMMIT_KEYS


def _is_secret_or_auth_approval_step(signals: ApprovalSignals) -> bool:
    if not _is_sensitive_tool(signals):
        return False
    payload_match = _payload_has_secret_indicator(signals)
    goal_match = _goal_has_secret_intent(signals)
    if signals.action == "type":
        return payload_match or (goal_match and _target_looks_secret_surface(signals))
    if signals.action == "click":
        return (payload_match and (_target_looks_secret_surface(signals) or _target_looks_secret_commit_surface(signals))) or (
            goal_match and (_target_looks_secret_surface(signals) or _target_looks_secret_commit_surface(signals))
        )
    if signals.action == "key":
        return _is_sensitive_commit_key(signals) and (payload_match or goal_match)
    return False


def _is_payment_information_approval_step(signals: ApprovalSignals) -> bool:
    if not _is_sensitive_tool(signals):
        return False
    payload_match = _payload_has_payment_indicator(signals)
    goal_match = _goal_has_payment_intent(signals)
    if signals.action == "type":
        return payload_match or (goal_match and _target_looks_payment_surface(signals))
    if signals.action == "click":
        return (payload_match and (_target_looks_payment_surface(signals) or _target_looks_payment_commit_surface(signals))) or (
            goal_match and (_target_looks_payment_surface(signals) or _target_looks_payment_commit_surface(signals))
        )
    if signals.action == "key":
        return _is_sensitive_commit_key(signals) and (payload_match or goal_match)
    return False


def _is_irreversible_external_action_step(signals: ApprovalSignals) -> bool:
    if not _is_sensitive_tool(signals):
        return False
    payload_match = _payload_has_irreversible_indicator(signals)
    goal_match = _goal_has_irreversible_intent(signals)
    if signals.action == "type":
        return payload_match or (goal_match and _target_looks_irreversible_surface(signals))
    if signals.action == "click":
        return payload_match or (goal_match and _target_looks_irreversible_surface(signals))
    if signals.action == "key":
        return _is_sensitive_commit_key(signals) and (payload_match or goal_match)
    return False


def _is_browser_external_side_effect_step(signals: ApprovalSignals) -> bool:
    if signals.action not in {"click", "type", "key"}:
        return False
    if _contains_any_phrase(signals.payload, _BROWSER_SIDE_EFFECT_TERMS):
        return True
    return _is_irreversible_external_action_step(signals)


def _normalized_text(text: Any) -> str:
    folded = unicodedata.normalize("NFKD", str(text or "").casefold())
    stripped = "".join(char for char in folded if not unicodedata.combining(char))
    stripped = stripped.replace("\u0131", "i")
    return re.sub(r"\s+", " ", stripped).strip()


def _goal_is_local_desktop_navigation(goal: str) -> bool:
    normalized = _normalized_text(goal)
    if not normalized:
        return False
    desktop_terms = (
        "desktop",
        "masaustu",
        "klasor",
        "folder",
        "file explorer",
        "explorer",
        "pencere",
        "window",
        "taskbar",
        "gorev cubugu",
        "baslat",
        "start menu",
        "onedrive",
    )
    web_terms = (
        "http://",
        "https://",
        "www.",
        ".com",
        ".org",
        ".net",
        "github",
        "google",
        "browser",
        "tarayici",
        "website",
        "site",
    )
    return any(term in normalized for term in desktop_terms) and not any(term in normalized for term in web_terms)


def _computer_action_requires_approval(context: ApprovalPolicyContext) -> bool:
    action = str(context.tool_args.get("action", "")).lower()
    if action == "screenshot":
        return False

    goal_is_desktop_nav = _goal_is_local_desktop_navigation(context.goal)
    if goal_is_desktop_nav and action in {"move", "click", "double_click", "right_click", "scroll", "drag"}:
        return False

    if action == "type":
        if goal_is_desktop_nav and not _contains_any(
            context.tool_args,
            ("password", "otp", "2fa", "captcha", "secret", "token", "code"),
        ):
            return False
        return True

    if action == "key":
        key = _normalized_text(context.tool_args.get("text", ""))
        safe_navigation_keys = {
            "win+d",
            "win+e",
            "alt+tab",
            "tab",
            "shift+tab",
            "arrowup",
            "arrowdown",
            "arrowleft",
            "arrowright",
            "escape",
            "esc",
            "enter",
            "home",
            "end",
            "pagedown",
            "pageup",
        }
        risky_keys = {
            "alt+f4",
            "ctrl+w",
            "ctrl+q",
            "ctrl+shift+esc",
            "ctrl+alt+delete",
            "shift+delete",
            "win+r",
            "win+x",
            "delete",
        }
        if key in risky_keys:
            return True
        if goal_is_desktop_nav and key in safe_navigation_keys:
            return False
        return True

    return True
