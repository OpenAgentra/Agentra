"""Shared Windows app aliases and profile detection."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


def _normalized_text(text: str) -> str:
    folded = unicodedata.normalize("NFKD", text.casefold())
    stripped = "".join(char for char in folded if not unicodedata.combining(char))
    stripped = stripped.replace("\u0131", "i")
    return re.sub(r"\s+", " ", stripped).strip()


@dataclass(frozen=True)
class WindowsAppProfile:
    """Structured hints for standard Windows app automation."""

    profile_id: str
    display_name: str
    aliases: tuple[str, ...]
    executable: str
    window_title_tokens: tuple[str, ...]
    result_automation_ids: tuple[str, ...] = ()
    expression_automation_ids: tuple[str, ...] = ()


WINDOWS_APP_PROFILES: dict[str, WindowsAppProfile] = {
    "calculator": WindowsAppProfile(
        profile_id="calculator",
        display_name="Calculator",
        aliases=("calculator", "calc", "hesap makinesi"),
        executable="calc.exe",
        window_title_tokens=("calculator", "hesap makinesi"),
        result_automation_ids=("NormalOutput", "CalculatorResults", "TextContainer"),
        expression_automation_ids=("CalculatorExpression",),
    ),
    "notepad": WindowsAppProfile(
        profile_id="notepad",
        display_name="Notepad",
        aliases=("notepad", "not defteri"),
        executable="notepad.exe",
        window_title_tokens=("notepad", "not defteri", "adsiz", "untitled"),
    ),
    "explorer": WindowsAppProfile(
        profile_id="explorer",
        display_name="File Explorer",
        aliases=("file explorer", "explorer", "dosya gezgini"),
        executable="explorer.exe",
        window_title_tokens=("file explorer", "explorer", "dosya gezgini"),
    ),
    "paint": WindowsAppProfile(
        profile_id="paint",
        display_name="Paint",
        aliases=("paint", "mspaint"),
        executable="mspaint.exe",
        window_title_tokens=("paint",),
    ),
    "wordpad": WindowsAppProfile(
        profile_id="wordpad",
        display_name="WordPad",
        aliases=("wordpad", "write"),
        executable="write.exe",
        window_title_tokens=("wordpad",),
    ),
}

WINDOWS_APP_ALIASES = {
    alias: profile.executable
    for profile in WINDOWS_APP_PROFILES.values()
    for alias in profile.aliases
}


def normalize_windows_app_command(app: str) -> str:
    """Normalize common Windows app aliases to launchable executables."""

    normalized = _normalized_text(app)
    return WINDOWS_APP_ALIASES.get(normalized, app.strip())


def guess_windows_app_profile(text: str) -> str | None:
    """Infer a standard Windows app profile from freeform text."""

    normalized = _normalized_text(text)
    if not normalized:
        return None
    for profile in WINDOWS_APP_PROFILES.values():
        if any(alias in normalized for alias in profile.aliases):
            return profile.profile_id
    return None


def get_windows_app_profile(profile_id: str | None) -> WindowsAppProfile | None:
    """Return a known Windows app profile when available."""

    if not profile_id:
        return None
    return WINDOWS_APP_PROFILES.get(profile_id)
