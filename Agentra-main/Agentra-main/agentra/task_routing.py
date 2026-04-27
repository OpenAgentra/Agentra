"""Goal routing helpers for live app execution policy."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Literal

from agentra.windows_apps import guess_windows_app_profile

_DESKTOP_SURFACE_TERMS = (
    "desktop",
    "masaustu",
    "taskbar",
    "gorev cubugu",
    "start menu",
    "baslat",
    "file explorer",
    "explorer",
    "folder",
    "klasor",
    "directory",
    "window",
    "pencere",
    "icon",
    "kisayol",
    "shortcut",
    "native app",
    "uygulama",
)
_VISUAL_DESKTOP_ACTION_TERMS = (
    "open",
    "launch",
    "click",
    "double click",
    "drag",
    "scroll",
    "type",
    "press",
    "enter",
    "go to",
    "show",
    "select",
    "navigate",
    "ac",
    "tikla",
    "surukle",
    "kaydir",
    "yaz",
    "bas",
    "sec",
    "goster",
    "gir",
)
_WEB_TERMS = (
    "browser",
    "tarayici",
    "web",
    "website",
    "site",
    "sayfa",
    "page",
    "repo",
    "repository",
    "github",
    "gitlab",
    "google",
    "youtube",
    "linkedin",
    "x.com",
    "twitter",
    "login",
    "log in",
    "sign in",
    "account",
    "hesap",
)
_FOLDER_CONTENT_TERMS = (
    "contents",
    "content",
    "inside",
    "what is inside",
    "what's inside",
    "list",
    "listele",
    "icerik",
    "icerigini",
    "icindekileri",
    "neler var",
    "bana soyle",
)
_LOCAL_DOCUMENT_OPEN_TERMS = (
    "powerpoint",
    "ppt",
    "pptx",
    "sunum",
    "presentation",
    "slide",
    "document",
    "dosya",
    "pdf",
    "word",
    "excel",
)
_UNDER_THE_HOOD_HINT_TERMS = (
    "under the hood",
    "arka planda",
    "arka planda yap",
    "gorunur masaustu",
    "visible desktop",
    "terminal acmadan",
    "without opening terminal",
    "without a terminal",
    "background",
)
_PATH_STYLE_LOCAL_TERMS = (
    "masaustumdeki",
    "masaustundeki",
    "my desktop",
    "on my desktop",
    "varsayilan uygulama",
    "default app",
    "default handler",
    "bul",
    "find",
)
_VISIBLE_DESKTOP_PREFERENCE_TERMS = (
    "masaustune git",
    "masaustune gir",
    "go to the desktop",
    "switch to the desktop",
    "karsina cikan",
    "gorunen",
    "gordugun",
    "on screen",
    "ekranda",
    "cift tikla",
    "double click",
    "sag tik",
    "right click",
    "surukle",
    "drag",
)
_DESKTOP_GUARDRAIL_PATTERNS = (
    r"(?:^|[;,.]|\band\b|\bbut\b|\bama\b)\s*(?:masaustu(?:mdeki|ndeki)?|desktop|diger|baska|other|any)[^.;,]{0,120}?(?:pencere|uygulama|window|app)[^.;,]{0,120}?(?:dokunma|dokunmadan|elleme|mudahale etme|touch(?:ing)?|interact(?: with)?|mess with|use)",
    r"(?:^|[;,.]|\band\b|\bbut\b|\bama\b)\s*(?:hicbir|diger|baska|other|any)[^.;,]{0,80}?(?:pencere|uygulama|window|app)[^.;,]{0,120}?(?:dokunma|dokunmadan|elleme|mudahale etme|touch(?:ing)?|interact(?: with)?|mess with|use)",
)
_WEB_GUARDRAIL_PATTERNS = (
    r"(?:^|[;,.]|\band\b|\bbut\b|\bama\b)\s*(?:tarayici|browser|web|website|site|sayfa)[^.;,]{0,80}?(?:kullanma|kullanmadan|kullanmayin|elleme|dokunma|do not use|don't use|without using|without a browser|touch(?:ing)?|interact(?: with)?|mess with)",
)
_WEB_FALSE_POSITIVE_PATTERNS = (
    r"\bhesap makinesi\b",
)
_REAL_BROWSER_CONTEXT_TERMS = (
    "chrome profile",
    "browser profile",
    "real environment",
    "real browser",
    "my browser",
    "my profile",
    "my chrome",
    "my account",
    "my repo",
    "my repository",
    "my github",
    "my real environment",
    "chrome profil",
    "chrome profilim",
    "chrome profilimi",
    "gercek ortam",
    "gercek cevre",
    "gercek tarayici",
    "kendi chrome profilim",
    "kendi profilim",
    "kendi hesabim",
    "hesabim",
    "hesabima ait",
    "profilim",
)
_VISIBLE_DESKTOP_ONLY_TERMS = (
    "drag",
    "surukle",
    "canvas",
    "oyun",
    "game",
    "coordinate",
    "pixel",
    "live preview",
)


@dataclass(frozen=True)
class LiveExecutionPolicy:
    """Derived live-app execution policy for a single goal."""

    browser_headless: bool
    local_execution_mode: Literal["visible", "under_the_hood", "native"]
    desktop_fallback_policy: Literal["visible_control", "pause_and_ask"]
    control_surface_hint: Literal["browser", "desktop"]
    desktop_execution_mode: Literal["desktop_native", "desktop_visible", "desktop_hidden"]
    desktop_backend_preference: Literal["native", "visible", "under_the_hood"]


def _normalized_text(text: str) -> str:
    folded = unicodedata.normalize("NFKD", text.casefold())
    stripped = "".join(char for char in folded if not unicodedata.combining(char))
    stripped = stripped.replace("\u0131", "i")
    return re.sub(r"\s+", " ", stripped).strip()


def _contains_any_phrase(text: str, phrases: tuple[str, ...]) -> bool:
    return any(phrase in text for phrase in phrases)


def _desktop_detection_text(goal: str) -> str:
    cleaned = _normalized_text(goal)
    for pattern in _DESKTOP_GUARDRAIL_PATTERNS:
        cleaned = re.sub(pattern, " ", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def _web_detection_text(goal: str) -> str:
    cleaned = _normalized_text(goal)
    for pattern in _WEB_GUARDRAIL_PATTERNS:
        cleaned = re.sub(pattern, " ", cleaned)
    for pattern in _WEB_FALSE_POSITIVE_PATTERNS:
        cleaned = re.sub(pattern, " ", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def goal_mentions_web_target(goal: str) -> bool:
    normalized = _web_detection_text(goal)
    if re.search(r"https?://|www\.", normalized):
        return True
    if re.search(r"\b[a-z0-9-]+\.(com|org|net|io|ai|app|dev|co|tr)\b", normalized):
        return True
    return _contains_any_phrase(normalized, _WEB_TERMS)


def goal_mentions_desktop_surface(goal: str) -> bool:
    normalized = _desktop_detection_text(goal)
    if re.search(r"\b[a-z]:\\", goal, flags=re.IGNORECASE):
        return True
    if "onedrive" in normalized:
        return True
    return _contains_any_phrase(normalized, _DESKTOP_SURFACE_TERMS)


def goal_requires_visual_desktop_control(goal: str) -> bool:
    normalized = _desktop_detection_text(goal)
    if not goal_mentions_desktop_surface(goal):
        return False
    return _contains_any_phrase(normalized, _VISUAL_DESKTOP_ACTION_TERMS)


def goal_has_local_desktop_component(goal: str) -> bool:
    normalized = _desktop_detection_text(goal)
    if not goal_mentions_desktop_surface(goal):
        return False
    return (
        _contains_any_phrase(normalized, _VISUAL_DESKTOP_ACTION_TERMS)
        or _contains_any_phrase(normalized, _FOLDER_CONTENT_TERMS)
        or _contains_any_phrase(normalized, _LOCAL_DOCUMENT_OPEN_TERMS)
        or _contains_any_phrase(normalized, _PATH_STYLE_LOCAL_TERMS)
    )


def goal_has_mixed_web_and_local_desktop_components(goal: str) -> bool:
    return goal_mentions_web_target(goal) and goal_has_local_desktop_component(goal)


def goal_is_desktop_local_only(goal: str) -> bool:
    return goal_has_local_desktop_component(goal) and not goal_mentions_web_target(goal)


def goal_requests_folder_contents(goal: str) -> bool:
    if not goal_has_local_desktop_component(goal):
        return False
    return _contains_any_phrase(_desktop_detection_text(goal), _FOLDER_CONTENT_TERMS)


def goal_requests_local_document_open(goal: str) -> bool:
    if not goal_has_local_desktop_component(goal):
        return False
    return _contains_any_phrase(_desktop_detection_text(goal), _LOCAL_DOCUMENT_OPEN_TERMS)


def goal_prefers_under_the_hood_local_execution(goal: str) -> bool:
    if not goal_has_local_desktop_component(goal):
        return False

    normalized = _desktop_detection_text(goal)
    if _contains_any_phrase(normalized, _UNDER_THE_HOOD_HINT_TERMS):
        return True
    if _contains_any_phrase(normalized, _VISIBLE_DESKTOP_PREFERENCE_TERMS):
        return False
    if not _contains_any_phrase(normalized, _PATH_STYLE_LOCAL_TERMS):
        return False
    return goal_requests_folder_contents(goal) or goal_requests_local_document_open(goal)


def goal_prefers_native_windows_desktop_execution(goal: str) -> bool:
    if not goal_has_local_desktop_component(goal):
        return False
    if goal_prefers_under_the_hood_local_execution(goal):
        return False
    if guess_windows_app_profile(goal) is None:
        return False
    return not _contains_any_phrase(_desktop_detection_text(goal), _VISIBLE_DESKTOP_ONLY_TERMS)


def goal_prefers_visible_desktop_execution(goal: str) -> bool:
    if not goal_has_local_desktop_component(goal):
        return False
    normalized = _desktop_detection_text(goal)
    return _contains_any_phrase(normalized, _VISIBLE_DESKTOP_PREFERENCE_TERMS) or _contains_any_phrase(
        normalized,
        _VISIBLE_DESKTOP_ONLY_TERMS,
    )


def goal_requests_real_browser_context(goal: str) -> bool:
    normalized = _normalized_text(goal)
    if not normalized:
        return False
    if _contains_any_phrase(normalized, _REAL_BROWSER_CONTEXT_TERMS):
        return True
    if not goal_mentions_web_target(goal):
        return False
    personal_web_terms = (
        "my account",
        "my repo",
        "my repository",
        "my profile",
        "my browser",
        "my github",
        "hesabim",
        "hesabima ait",
        "profilim",
        "kendi hesabim",
        "kendi profilim",
    )
    return _contains_any_phrase(normalized, personal_web_terms)


def choose_live_execution_policy(
    goal: str,
    *,
    requested_headless: bool | None = None,
) -> LiveExecutionPolicy:
    """Choose live-app execution defaults based on the user goal."""

    browser_headless = True if requested_headless is None else requested_headless

    if goal_has_local_desktop_component(goal):
        if goal_prefers_under_the_hood_local_execution(goal):
            return LiveExecutionPolicy(
                browser_headless=browser_headless,
                local_execution_mode="under_the_hood",
                desktop_fallback_policy="pause_and_ask",
                control_surface_hint="browser",
                desktop_execution_mode="desktop_hidden",
                desktop_backend_preference="under_the_hood",
            )
        if goal_prefers_visible_desktop_execution(goal):
            if goal_prefers_native_windows_desktop_execution(goal):
                return LiveExecutionPolicy(
                    browser_headless=browser_headless,
                    local_execution_mode="native",
                    desktop_fallback_policy="visible_control",
                    control_surface_hint="browser" if goal_has_mixed_web_and_local_desktop_components(goal) else "desktop",
                    desktop_execution_mode="desktop_native",
                    desktop_backend_preference="native",
                )
            return LiveExecutionPolicy(
                browser_headless=browser_headless,
                local_execution_mode="visible",
                desktop_fallback_policy="visible_control",
                control_surface_hint="browser" if goal_has_mixed_web_and_local_desktop_components(goal) else "desktop",
                desktop_execution_mode="desktop_visible",
                desktop_backend_preference="visible",
            )
        if goal_prefers_native_windows_desktop_execution(goal):
            return LiveExecutionPolicy(
                browser_headless=browser_headless,
                local_execution_mode="native",
                desktop_fallback_policy="pause_and_ask",
                control_surface_hint="browser" if goal_has_mixed_web_and_local_desktop_components(goal) else "desktop",
                desktop_execution_mode="desktop_hidden",
                desktop_backend_preference="native",
            )
        return LiveExecutionPolicy(
            browser_headless=browser_headless,
            local_execution_mode="native" if goal_has_mixed_web_and_local_desktop_components(goal) else "visible",
            desktop_fallback_policy="pause_and_ask",
            control_surface_hint="browser" if goal_has_mixed_web_and_local_desktop_components(goal) else "desktop",
            desktop_execution_mode="desktop_hidden",
            desktop_backend_preference="visible",
        )

    return LiveExecutionPolicy(
        browser_headless=browser_headless,
        local_execution_mode="visible",
        desktop_fallback_policy="visible_control",
        control_surface_hint="browser",
        desktop_execution_mode="desktop_visible",
        desktop_backend_preference="visible",
    )
