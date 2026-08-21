"""
Model discovery.

OpenAI-compatible: ``/v1/models``.
Cursor: ``cursor-sdk`` ``Cursor.models.list``.
The chat SDK is an argument — not a process-wide override.
"""

from __future__ import annotations

from typing import Any

import httpx

from backend.config import settings
from backend.llm.runtime import default_provider, parse_provider
from backend.monitoring.logger import get_logger

logger = get_logger(__name__)

_REQUEST_TIMEOUT = 5.0


async def fetch_models(base_url: str) -> list[dict[str, Any]]:
    url = f"{base_url.rstrip('/')}/models"
    try:
        async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()
    except Exception as exc:
        logger.warning("model_discovery_failed", url=url, error=str(exc))
        return []
    return data.get("data", [])


async def check_endpoint_health(base_url: str) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}/models"
    try:
        async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()
    except Exception:
        return {"status": "unreachable", "models_loaded": 0}
    return {"status": "ok", "models_loaded": len(data.get("data", []))}


async def list_models(provider: str | None = None) -> list[dict[str, Any]]:
    sdk = parse_provider(provider) if provider else default_provider()
    if sdk == "cursor":
        from backend.llm.cursor import CursorTranslator

        return await CursorTranslator().list_models()
    models = await fetch_models(settings.openai_base_url)
    return [
        {
            "id": m.get("id", ""),
            "object": m.get("object", "model"),
            "display_name": m.get("id", ""),
            "preferred": False,
        }
        for m in models
        if m.get("id")
    ]


async def check_chat_health(provider: str | None = None) -> dict[str, Any]:
    sdk = parse_provider(provider) if provider else default_provider()
    if sdk == "cursor":
        from backend.llm.cursor import CursorTranslator

        return await CursorTranslator().health()
    return await check_endpoint_health(settings.openai_base_url)


# Older names used by health/settings before session-scoped switch.
async def list_active_models() -> list[dict[str, Any]]:
    return await list_models()


async def check_active_provider_health() -> dict[str, Any]:
    return await check_chat_health()
