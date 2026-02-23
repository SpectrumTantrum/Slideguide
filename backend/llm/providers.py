"""
Provider configuration resolution for LLM and embedding clients.

Returns connection parameters (base_url, api_key, headers) based on
the active provider setting. Both OpenRouter and LM Studio use the
OpenAI-compatible API, so the SDK is the same — only config differs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from backend.config import settings


@dataclass(frozen=True)
class ProviderConfig:
    """Connection parameters for an OpenAI-compatible API provider."""

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


def get_chat_provider_config() -> ProviderConfig:
    """Resolve the active chat/LLM provider configuration."""
    if settings.llm_provider == "lmstudio":
        return ProviderConfig(
            name="lmstudio",
            base_url=settings.lmstudio_base_url,
            api_key="lm-studio",  # SDK requires non-empty; LM Studio ignores it
            headers={},
        )
    return ProviderConfig(
        name="openrouter",
        base_url=settings.openrouter_base_url,
        api_key=settings.openrouter_api_key,
        headers={
            "HTTP-Referer": settings.app_url,
            "X-Title": settings.app_name,
        },
    )


def get_embedding_provider_config() -> ProviderConfig:
    """Resolve the active embedding provider configuration."""
    if settings.embedding_provider == "lmstudio":
        return ProviderConfig(
            name="lmstudio",
            base_url=settings.lmstudio_base_url,
            api_key="lm-studio",
            headers={},
        )
    return ProviderConfig(
        name="openrouter",
        base_url=settings.openrouter_base_url,
        api_key=settings.openrouter_api_key,
        headers={
            "HTTP-Referer": settings.app_url,
            "X-Title": settings.app_name,
        },
    )


def get_vision_provider_config() -> ProviderConfig:
    """Resolve the vision provider — defaults to OpenRouter for cloud VLMs."""
    if settings.vision_provider == "lmstudio":
        return ProviderConfig(
            name="lmstudio",
            base_url=settings.lmstudio_base_url,
            api_key="lm-studio",
            headers={},
        )
    return ProviderConfig(
        name="openrouter",
        base_url=settings.openrouter_base_url,
        api_key=settings.openrouter_api_key,
        headers={
            "HTTP-Referer": settings.app_url,
            "X-Title": settings.app_name,
        },
    )
