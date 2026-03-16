"""Rule-based approval policy engine for risky tool actions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class ApprovalPolicyContext:
    """Context used to evaluate whether a tool call requires approval."""

    tool_name: str
    tool_args: dict[str, Any]
    goal: str = ""
    thread_id: str | None = None
    run_id: str | None = None


@dataclass(frozen=True)
class ApprovalDecision:
    """Structured result returned by the approval engine."""

    action: str  # "allow" | "require_approval"
    summary: str
    reason: str = ""
    rule_id: str | None = None
    risk_level: str = "low"


@dataclass(frozen=True)
class ApprovalRule:
    """Single ordered approval rule."""

    rule_id: str
    risk_level: str
    matcher: Callable[[ApprovalPolicyContext], bool]
    reason: str
    summary: str

    def matches(self, context: ApprovalPolicyContext) -> bool:
        return bool(self.matcher(context))


class ApprovalPolicyEngine:
    """Ordered rule engine for approval decisions."""

    def __init__(self, rules: list[ApprovalRule]) -> None:
        self._rules = list(rules)

    def evaluate(self, context: ApprovalPolicyContext) -> ApprovalDecision:
        for rule in self._rules:
            if rule.matches(context):
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
                    matcher=lambda ctx: ctx.tool_name == "browser"
                    and str(ctx.tool_args.get("action", "")).lower() == "type"
                    and _contains_any(ctx.tool_args, ("password", "otp", "2fa", "captcha", "secret", "token", "code")),
                    reason="Secret, authentication, or verification data entry requires user approval.",
                    summary="Tarayıcıda gizli veya kimlik doğrulama bilgisi girilecek.",
                ),
                ApprovalRule(
                    rule_id="browser-external-side-effect",
                    risk_level="high",
                    matcher=lambda ctx: ctx.tool_name == "browser"
                    and str(ctx.tool_args.get("action", "")).lower() in {"click", "type"}
                    and _contains_any(ctx.tool_args, ("submit", "post", "send", "share", "publish", "apply", "checkout", "pay", "confirm", "delete", "remove")),
                    reason="This browser action may submit, send, publish, pay, or otherwise have external effects.",
                    summary="Tarayıcı eylemi dış etki üretebilir veya geri alınması zor bir işlem başlatabilir.",
                ),
                ApprovalRule(
                    rule_id="terminal-install-or-side-effect",
                    risk_level="high",
                    matcher=lambda ctx: ctx.tool_name == "terminal"
                    and _contains_any(
                        ctx.tool_args,
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
                    matcher=lambda ctx: ctx.tool_name == "filesystem"
                    and str(ctx.tool_args.get("action", "")).lower() in {"delete", "move"},
                    reason="This filesystem action changes or removes existing files.",
                    summary="Dosya sistemi işlemi mevcut dosyaları değiştirebilir veya silebilir.",
                ),
                ApprovalRule(
                    rule_id="git-rewrite-or-clone",
                    risk_level="medium",
                    matcher=lambda ctx: ctx.tool_name == "git"
                    and str(ctx.tool_args.get("action", "")).lower() in {"reset", "checkout", "clone"},
                    reason="This git action can rewrite or materially change workspace state.",
                    summary="Git işlemi çalışma alanı durumunu önemli ölçüde değiştirebilir.",
                ),
                ApprovalRule(
                    rule_id="computer-direct-control",
                    risk_level="high",
                    matcher=lambda ctx: ctx.tool_name == "computer"
                    and str(ctx.tool_args.get("action", "")).lower() != "screenshot",
                    reason="Direct desktop control requires explicit user approval.",
                    summary="Doğrudan masaüstü kontrolü için kullanıcı onayı gerekiyor.",
                ),
            ]
        )


def _contains_any(payload: dict[str, Any], tokens: tuple[str, ...]) -> bool:
    flattened = " ".join(str(item).lower() for item in payload.values())
    return any(token in flattened for token in tokens)
