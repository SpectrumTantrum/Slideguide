"""
Settings and provider information API routes.

Exposes the current provider SDK, available subscriptions, and models
so the frontend can switch billing (Cursor usage vs OpenAI-compatible).
"""

from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.config import settings
from backend.llm.discovery import (
    check_active_provider_health,
    check_endpoint_health,
    list_active_models,
)
from backend.llm.providers import available_providers, provider_metadata
from backend.llm.runtime import get_active_provider, set_active_provider
from backend.llm.tool_compatibility import ToolCompatibilityLayer

router = APIRouter(prefix="/api/settings", tags=["settings"])

# Shared reference — same instance used by agent nodes
_tool_compat: ToolCompatibilityLayer | None = None


def set_tool_compat(tc: ToolCompatibilityLayer) -> None:
    """Inject the shared ToolCompatibilityLayer for status reporting."""
    global _tool_compat
    _tool_compat = tc


class ProviderSwitchRequest(BaseModel):
    """Select which subscription SDK bills chat usage."""

    provider: Literal["openai", "cursor"] = Field(..., description="Chat provider SDK id")


@router.get("/provider")
async def get_provider_status() -> dict[str, Any]:
    """Return the active provider SDK, capabilities, and switchable subscriptions."""
    provider = get_active_provider()
    meta = provider_metadata(provider)

    tool_mode = "prompt" if provider == "cursor" else "native"
    if provider != "cursor" and _tool_compat is not None:
        tool_mode = _tool_compat.mode

    endpoint = await check_active_provider_health()
    embeddings = await check_endpoint_health(settings.openai_base_url)

    return {
        "provider": provider,
        "sdk": meta["sdk"],
        "usage": meta["usage"],
        "label": meta["label"],
        "base_url": settings.openai_base_url if provider == "openai" else "cursor-sdk",
        "runtime": settings.cursor_runtime if provider == "cursor" else None,
        "endpoint": endpoint,
        "embeddings": embeddings,
        "capabilities": {
            "vision": bool(settings.active_vision_model) or provider == "cursor",
            "tool_mode": tool_mode,
        },
        "models": {
            "primary": settings.active_primary_model,
            "routing": settings.active_routing_model,
            "embedding": settings.active_embedding_model,
            "vision": settings.active_vision_model,
            "effort": settings.cursor_reasoning_effort if provider == "cursor" else None,
        },
        "available_providers": available_providers(),
    }


@router.post("/provider")
async def switch_provider(body: ProviderSwitchRequest) -> dict[str, Any]:
    """Switch the in-process chat SDK (Cursor subscription vs OpenAI-compatible)."""
    try:
        set_active_provider(body.provider)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return await get_provider_status()


@router.get("/models")
async def get_available_models() -> dict[str, Any]:
    """Return models from the active provider SDK (Cursor models first on that route)."""
    provider = get_active_provider()
    models = await list_active_models()
    return {
        "provider": provider,
        "sdk": provider_metadata(provider)["sdk"],
        "models": models,
    }
