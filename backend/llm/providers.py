"""
Provider catalog and embedding (RAG) HTTP config.

``provider`` here means the OpenAI-compatible RAG endpoint. Chat SDK
selection lives in ``backend.llm.runtime``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from backend.config import settings
from backend.llm.runtime import cursor_is_configured, default_provider

_NO_AUTH_PLACEHOLDER = "sk-no-key-required"

PROVIDER_CATALOG: tuple[dict[str, str], ...] = (
    {
        "id": "cursor",
        "label": "Cursor",
        "sdk": "cursor-sdk",
        "usage": "cursor_subscription",
        "description": "Bills this session's chat to your Cursor plan via cursor-sdk.",
    },
    {
        "id": "openai",
        "label": "OpenAI-compatible",
        "sdk": "openai",
        "usage": "openai_compatible_endpoint",
        "description": "Bills this session's chat to the configured OpenAI-compatible endpoint.",
    },
)


@dataclass(frozen=True)
class ProviderConfig:
    """Connection parameters for the OpenAI-compatible embeddings/chat HTTP API."""

    name: str
    base_url: str
    api_key: str
    headers: dict[str, str]

    def client_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "base_url": self.base_url,
            "api_key": self.api_key,
        }
        if self.headers:
            kwargs["default_headers"] = self.headers
        return kwargs


def get_provider_config() -> ProviderConfig:
    """OpenAI-compatible HTTP endpoint used for embeddings (and OpenAI chat)."""
    return get_embedding_config()


def get_embedding_config() -> ProviderConfig:
    return ProviderConfig(
        name="openai",
        base_url=settings.openai_base_url,
        api_key=settings.openai_api_key or _NO_AUTH_PLACEHOLDER,
        headers={},
    )


def provider_metadata(provider: str | None = None) -> dict[str, Any]:
    provider = provider or default_provider()
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
    rows: list[dict[str, Any]] = []
    for entry in PROVIDER_CATALOG:
        configured = True
        if entry["id"] == "cursor":
            configured = cursor_is_configured()
        rows.append({**entry, "configured": configured})
    return rows
