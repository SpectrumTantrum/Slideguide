"""
Health check and metrics endpoints.

GET /health      — Full health check with dependency status
GET /health/live — Kubernetes liveness (is the process alive?)
GET /health/ready — Kubernetes readiness (can we serve traffic?)
GET /metrics     — Aggregated cost, latency, and error metrics
"""

from __future__ import annotations

from datetime import datetime, timezone

import httpx
from fastapi import APIRouter

from backend.config import settings
from backend.monitoring.metrics import metrics

router = APIRouter(tags=["monitoring"])


def _check_supabase() -> str:
    """Check Supabase PostgreSQL connectivity."""
    try:
        from backend.db.client import get_supabase

        supabase = get_supabase()
        result = supabase.table("uploads").select("id").limit(1).execute()
        # If we get here without exception, the connection works
        return "ok"
    except Exception:
        return "unavailable"


async def _check_openrouter() -> str:
    """Check OpenRouter API reachability."""
    if not settings.openrouter_api_key:
        return "not_configured"
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(
                f"{settings.openrouter_base_url}/models",
                headers={"Authorization": f"Bearer {settings.openrouter_api_key}"},
            )
            return "ok" if resp.status_code == 200 else "degraded"
    except Exception:
        return "unavailable"


async def _check_lmstudio() -> dict:
    """Check LM Studio connectivity and loaded models."""
    from backend.llm.discovery import check_lmstudio_health

    return await check_lmstudio_health()


@router.get("/health")
async def health_check() -> dict:
    """Full health check with all dependency statuses."""
    checks: dict = {
        "supabase": _check_supabase(),
    }

    # Check the active LLM provider
    if settings.llm_provider == "lmstudio":
        lms_health = await _check_lmstudio()
        checks["lmstudio"] = lms_health["status"]
        checks["lmstudio_models"] = lms_health["models_loaded"]
        # Still check OpenRouter if vision needs it
        if settings.vision_provider == "openrouter" and settings.openrouter_api_key:
            checks["openrouter_vision"] = await _check_openrouter()
    else:
        checks["openrouter"] = await _check_openrouter()

    all_ok = all(
        v == "ok" for k, v in checks.items()
        if isinstance(v, str) and k != "lmstudio_models"
    )
    # Supabase is always required; LM Studio is critical when it's the LLM provider
    critical_ok = checks["supabase"] == "ok"
    if settings.llm_provider == "lmstudio":
        critical_ok = critical_ok and checks.get("lmstudio") == "ok"

    if all_ok:
        status = "healthy"
    elif critical_ok:
        status = "degraded"
    else:
        status = "unhealthy"

    return {
        "status": status,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": "0.1.0",
        "environment": settings.environment,
        "checks": checks,
    }


@router.get("/health/live")
async def liveness() -> dict:
    """Kubernetes liveness probe — is the process alive?"""
    return {"status": "alive", "timestamp": datetime.now(timezone.utc).isoformat()}


@router.get("/health/ready")
async def readiness() -> dict:
    """
    Kubernetes readiness probe — can we serve traffic?

    Checks that critical dependencies (Supabase) are available.
    """
    supabase_status = _check_supabase()
    is_ready = supabase_status == "ok"

    return {
        "status": "ready" if is_ready else "not_ready",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "checks": {
            "supabase": supabase_status,
        },
    }


@router.get("/metrics")
async def get_metrics() -> dict:
    """Return aggregated application metrics."""
    summary = metrics.get_summary()

    # Add per-model failure tracking
    model_stats = {}
    for model_id, calls in metrics.model_call_counts.items():
        model_stats[model_id] = {
            "calls": calls,
            "tokens": metrics.model_token_counts.get(model_id, 0),
            "errors": metrics.model_error_counts.get(model_id, 0),
        }

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "active_provider": {
            "llm": settings.llm_provider,
            "embedding": settings.embedding_provider,
            "vision": settings.vision_provider,
        },
        **summary,
        "models": model_stats,
        "sessions": {
            "active": metrics.active_sessions,
            "total": metrics.total_sessions,
        },
    }
