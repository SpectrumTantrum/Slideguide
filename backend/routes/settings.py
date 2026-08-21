"""
Settings and provider information API routes.

Exposes the current provider configuration and available models
so the frontend can adapt its UI and show capability warnings.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from backend.config import settings
from backend.llm.discovery import check_endpoint_health, fetch_models
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
    """Return the OpenAI-compatible endpoint configuration and capabilities."""
    tool_mode = "native"
    if _tool_compat is not None:
        tool_mode = _tool_compat.mode

    return {
        "provider": "openai",
        "base_url": settings.openai_base_url,
        "endpoint": await check_endpoint_health(settings.openai_base_url),
        "capabilities": {
            "vision": bool(settings.active_vision_model),
            "tool_mode": tool_mode,
        },
        "models": {
            "primary": settings.active_primary_model,
            "routing": settings.active_routing_model,
            "embedding": settings.active_embedding_model,
            "vision": settings.active_vision_model,
        },
    }


@router.get("/models")
async def get_available_models() -> dict[str, Any]:
    """Return the models served by the configured OpenAI-compatible endpoint."""
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
