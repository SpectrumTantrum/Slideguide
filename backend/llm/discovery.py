"""
Model discovery for LM Studio.

Queries the local ``/v1/models`` endpoint to find which models the user
has loaded. Results are cached briefly to avoid hammering the endpoint.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from backend.config import settings
from backend.monitoring.logger import get_logger

logger = get_logger(__name__)

CACHE_TTL_SECONDS = 60.0
_REQUEST_TIMEOUT = 5.0  # seconds


@dataclass
class _ModelCache:
    models: list[dict[str, Any]] = field(default_factory=list)
    fetched_at: float = 0.0


_cache = _ModelCache()


async def discover_local_models() -> list[dict[str, Any]]:
    """
    Query LM Studio for currently loaded models.

    Returns a list of dicts, each with at least ``id`` (model name).
    Returns an empty list if LM Studio is unreachable or has no models.
    """
    now = time.time()
    if _cache.models and (now - _cache.fetched_at) < CACHE_TTL_SECONDS:
        return _cache.models

    url = f"{settings.lmstudio_base_url}/models"
    try:
        async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()
    except (httpx.HTTPError, httpx.TimeoutException, Exception) as exc:
        logger.warning("lmstudio_discovery_failed", url=url, error=str(exc))
        return []

    models = data.get("data", [])
    _cache.models = models
    _cache.fetched_at = now

    logger.info("lmstudio_models_discovered", count=len(models))
    return models


async def check_lmstudio_health() -> dict[str, Any]:
    """
    Check if LM Studio is reachable and report loaded model count.

    Returns ``{"status": "ok"|"unreachable", "models_loaded": int}``.
    """
    models = await discover_local_models()
    if models:
        return {"status": "ok", "models_loaded": len(models)}

    # Distinguish between "reachable but empty" and "unreachable"
    url = f"{settings.lmstudio_base_url}/models"
    try:
        async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            return {"status": "ok", "models_loaded": 0}
    except Exception:
        return {"status": "unreachable", "models_loaded": 0}


async def auto_select_model(role: str = "primary") -> str | None:
    """
    Auto-select a loaded model for a given role.

    Returns the model ID string, or None if no models are loaded.
    Logs the auto-selection for visibility.
    """
    models = await discover_local_models()
    if not models:
        return None

    model_id = models[0].get("id", "")
    if model_id:
        logger.info("lmstudio_auto_selected", role=role, model=model_id)
    return model_id or None


def invalidate_cache() -> None:
    """Clear the model cache so the next call re-fetches."""
    _cache.models = []
    _cache.fetched_at = 0.0
