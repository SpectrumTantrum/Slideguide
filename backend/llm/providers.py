"""
Connection parameters for the OpenAI-compatible endpoint.

SlideGuide talks to a single OpenAI-compatible API (chat, embeddings, and
vision) whose base URL and key are chosen by the user. This can be real
OpenAI, Ollama, vLLM, LocalAI, OpenRouter, LM Studio, or any gateway that
speaks the OpenAI API.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from backend.config import settings

# The OpenAI SDK refuses to initialize without a non-empty api_key. Many
# OpenAI-compatible endpoints (local Ollama/vLLM/LocalAI, LM Studio, etc.) do
# not require authentication, so we send a harmless placeholder in that case.
_NO_AUTH_PLACEHOLDER = "sk-no-key-required"


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
    """Resolve the OpenAI-compatible endpoint configuration."""
    return ProviderConfig(
        name="openai",
        base_url=settings.openai_base_url,
        api_key=settings.openai_api_key or _NO_AUTH_PLACEHOLDER,
        headers={},
    )
