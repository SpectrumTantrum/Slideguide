"""
Model discovery for the active chat provider SDK.

The OpenAI-compatible path queries ``/v1/models``. The Cursor path uses
``cursor-sdk`` (``Cursor.models.list``), preferring Cursor-owned models.
"""

from __future__ import annotations

from typing import Any

import httpx

from backend.monitoring.logger import get_logger

logger = get_logger(__name__)

_REQUEST_TIMEOUT = 5.0  # seconds


async def fetch_models(base_url: str) -> list[dict[str, Any]]:
    """
    Query an OpenAI-compatible ``/models`` endpoint.

    Returns a list of model dicts, or an empty list if the endpoint is
    unreachable.
    """
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
    """
    Check reachability of an OpenAI-compatible endpoint's ``/models`` route.

    Returns ``{"status": "ok"|"unreachable", "models_loaded": int}``.
    """
    url = f"{base_url.rstrip('/')}/models"
    try:
        async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()
    except Exception:
        return {"status": "unreachable", "models_loaded": 0}
    return {"status": "ok", "models_loaded": len(data.get("data", []))}


async def list_active_models() -> list[dict[str, Any]]:
    """List models from the active chat provider SDK."""
    from backend.llm.providers import create_chat_provider

    return await create_chat_provider().list_models()


async def check_active_provider_health() -> dict[str, Any]:
    """Reachability check for the active chat provider SDK."""
    from backend.llm.providers import create_chat_provider

    return await create_chat_provider().health()
