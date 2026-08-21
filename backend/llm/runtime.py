"""
Active chat-provider selection.

``LLM_PROVIDER`` is the process default. ``set_active_provider`` lets the
app switch subscription SDKs at runtime without restarting.
"""

from __future__ import annotations

SUPPORTED_PROVIDERS = ("openai", "cursor")

_override: str | None = None


def get_active_provider() -> str:
    """Return the chat provider currently billing LLM calls."""
    if _override:
        return _override
    from backend.config import settings

    return settings.llm_provider


def set_active_provider(provider: str) -> str:
    """Switch the in-process chat provider. Returns the new provider id."""
    normalized = provider.strip().lower()
    if normalized not in SUPPORTED_PROVIDERS:
        raise ValueError(f"Unknown provider '{provider}'. Expected one of: {SUPPORTED_PROVIDERS}")
    if normalized == "cursor" and not cursor_is_configured():
        raise ValueError(
            "Cursor is not configured. Set CURSOR_API_KEY to bill Cursor usage."
        )
    global _override
    _override = normalized
    return normalized


def reset_active_provider() -> None:
    """Clear the runtime override so ``LLM_PROVIDER`` is used again."""
    global _override
    _override = None


def cursor_is_configured() -> bool:
    """True when a Cursor API key is available for the Python SDK."""
    from backend.config import settings

    return bool(settings.cursor_api_key.strip())
