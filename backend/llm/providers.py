"""
Provider configuration resolution for LLM and embedding clients.

Returns connection parameters (base_url, api_key, headers) based on
the active provider setting. OpenRouter, LM Studio, and any generic
OpenAI-compatible endpoint all speak the OpenAI API, so the SDK is the
same — only the base URL, key, and headers differ.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from backend.config import settings

# The OpenAI SDK refuses to initialize without a non-empty api_key. Many
# OpenAI-compatible endpoints (LM Studio, local Ollama/vLLM/LocalAI, etc.) do
# not require authentication, so we send a harmless placeholder in that case.
_NO_AUTH_PLACEHOLDER = "sk-no-key-required"


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


def _config_for(provider: str) -> ProviderConfig:
    """Resolve connection parameters for a named provider."""
    if provider == "lmstudio":
        return ProviderConfig(
            name="lmstudio",
            base_url=settings.lmstudio_base_url,
            api_key="lm-studio",  # SDK requires non-empty; LM Studio ignores it
            headers={},
        )
    if provider == "openai":
        # Generic OpenAI-compatible endpoint chosen by the user.
        return ProviderConfig(
            name="openai",
            base_url=settings.openai_base_url,
            api_key=settings.openai_api_key or _NO_AUTH_PLACEHOLDER,
            headers={},
        )
    # Default: OpenRouter.
    return ProviderConfig(
        name="openrouter",
        base_url=settings.openrouter_base_url,
        api_key=settings.openrouter_api_key or _NO_AUTH_PLACEHOLDER,
        headers={
            "HTTP-Referer": settings.app_url,
            "X-Title": settings.app_name,
        },
    )


def get_chat_provider_config() -> ProviderConfig:
    """Resolve the active chat/LLM provider configuration."""
    return _config_for(settings.llm_provider)


def get_embedding_provider_config() -> ProviderConfig:
    """Resolve the active embedding provider configuration."""
    return _config_for(settings.embedding_provider)


def get_vision_provider_config() -> ProviderConfig:
    """Resolve the vision provider — defaults to OpenRouter for cloud VLMs."""
    return _config_for(settings.vision_provider)
