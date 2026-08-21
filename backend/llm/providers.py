"""
Provider SDK registry.

SlideGuide can bill chat to different subscriptions by swapping the SDK:

- ``openai`` — official OpenAI SDK against an OpenAI-compatible endpoint
- ``cursor`` — official ``cursor-sdk`` against the user's Cursor usage

Embeddings stay on the OpenAI-compatible endpoint (Cursor has no embedding API).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from backend.config import settings
from backend.llm.runtime import cursor_is_configured, get_active_provider

# The OpenAI SDK refuses to initialize without a non-empty api_key. Many
# OpenAI-compatible endpoints (local Ollama/vLLM/LocalAI, LM Studio, etc.) do
# not require authentication, so we send a harmless placeholder in that case.
_NO_AUTH_PLACEHOLDER = "sk-no-key-required"

PROVIDER_CATALOG: tuple[dict[str, str], ...] = (
    {
        "id": "cursor",
        "label": "Cursor",
        "sdk": "cursor-sdk",
        "usage": "cursor_subscription",
        "description": "Bills chat to your Cursor plan via the official Cursor SDK.",
    },
    {
        "id": "openai",
        "label": "OpenAI-compatible",
        "sdk": "openai",
        "usage": "openai_compatible_endpoint",
        "description": "Bills chat to the configured OpenAI-compatible endpoint.",
    },
)


@dataclass(frozen=True)
class ProviderConfig:
    """Connection parameters for an OpenAI-compatible API endpoint."""

    name: str
    base_url: str
    api_key: str
    headers: dict[str, str]

    def client_kwargs(self) -> dict[str, Any]:
        """Return kwargs suitable for ``openai.AsyncOpenAI(...)``."""
        kwargs: dict[str, Any] = {
            "base_url": self.base_url,
            "api_key": self.api_key,
        }
        if self.headers:
            kwargs["default_headers"] = self.headers
        return kwargs


def get_provider_config() -> ProviderConfig:
    """Resolve the OpenAI-compatible endpoint used for embeddings (and OpenAI chat)."""
    return get_embedding_config()


def get_embedding_config() -> ProviderConfig:
    """OpenAI-compatible connection used for embeddings regardless of chat SDK."""
    return ProviderConfig(
        name="openai",
        base_url=settings.openai_base_url,
        api_key=settings.openai_api_key or _NO_AUTH_PLACEHOLDER,
        headers={},
    )


def create_chat_provider(name: str | None = None) -> Any:
    """Instantiate the SDK adapter for ``name`` (or the active provider)."""
    provider = name or get_active_provider()
    if provider == "cursor":
        from backend.llm.cursor_provider import CursorChatProvider

        return CursorChatProvider()
    from backend.llm.openai_provider import OpenAIChatProvider

    return OpenAIChatProvider()


def provider_metadata(provider: str | None = None) -> dict[str, Any]:
    """Describe a provider for the settings API / frontend banner."""
    provider = provider or get_active_provider()
    for entry in PROVIDER_CATALOG:
        if entry["id"] == provider:
            return dict(entry)
    return {
        "id": provider,
        "label": provider,
        "sdk": provider,
        "usage": "unknown",
        "description": "",
    }


def available_providers() -> list[dict[str, Any]]:
    """Providers the UI can offer, with whether each is configured."""
    rows: list[dict[str, Any]] = []
    for entry in PROVIDER_CATALOG:
        configured = True
        if entry["id"] == "cursor":
            configured = cursor_is_configured()
        rows.append({**entry, "configured": configured})
    return rows
