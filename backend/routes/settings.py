"""
Settings and provider information API routes.

Exposes the current provider configuration and available models
so the frontend can adapt its UI and show capability warnings.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from backend.config import settings
from backend.llm.discovery import (
    check_endpoint_health,
    check_lmstudio_health,
    discover_local_models,
    fetch_models,
)
from backend.llm.models import MODELS
from backend.llm.tool_compatibility import ToolCompatibilityLayer

router = APIRouter(prefix="/api/settings", tags=["settings"])

# Shared reference — same instance used by agent nodes
_tool_compat: ToolCompatibilityLayer | None = None


def set_tool_compat(tc: ToolCompatibilityLayer) -> None:
    """Inject the shared ToolCompatibilityLayer for status reporting."""
    global _tool_compat
    _tool_compat = tc


@router.get("/provider")
async def get_provider_config() -> dict[str, Any]:
    """Return the current provider configuration and capabilities."""
    # Vision needs a real cloud key only when routed through OpenRouter.
    # Self-hosted providers (LM Studio / custom OpenAI-compatible) are trusted.
    if settings.vision_provider == "openrouter":
        vision_available = bool(settings.openrouter_api_key)
    else:
        vision_available = True

    tool_mode = "native"
    if _tool_compat is not None:
        tool_mode = _tool_compat.mode

    result: dict[str, Any] = {
        "llm_provider": settings.llm_provider,
        "embedding_provider": settings.embedding_provider,
        "vision_provider": settings.vision_provider,
        "capabilities": {
            "vision": vision_available,
            "tool_mode": tool_mode,
        },
        "models": {
            "primary": settings.active_primary_model,
            "routing": settings.active_routing_model,
            "embedding": settings.active_embedding_model,
            "vision": settings.active_vision_model,
        },
    }

    if settings.llm_provider == "lmstudio":
        result["lmstudio"] = await check_lmstudio_health()
    elif settings.llm_provider == "openai":
        result["openai"] = {
            "base_url": settings.openai_base_url,
            **await check_endpoint_health(settings.openai_base_url),
        }

    return result


@router.get("/models")
async def get_available_models() -> dict[str, Any]:
    """Return available models for the active provider."""
    if settings.llm_provider == "lmstudio":
        local_models = await discover_local_models()
        return {
            "provider": "lmstudio",
            "models": [
                {
                    "id": m.get("id", ""),
                    "object": m.get("object", "model"),
                }
                for m in local_models
            ],
        }

    if settings.llm_provider == "openai":
        endpoint_models = await fetch_models(settings.openai_base_url)
        return {
            "provider": "openai",
            "models": [
                {
                    "id": m.get("id", ""),
                    "object": m.get("object", "model"),
                }
                for m in endpoint_models
            ],
        }

    return {
        "provider": "openrouter",
        "models": [
            {
                "id": cfg.model_id,
                "display_name": cfg.display_name,
                "supports_tools": cfg.supports_tools,
                "supports_vision": cfg.supports_vision,
            }
            for cfg in MODELS.values()
        ],
    }
