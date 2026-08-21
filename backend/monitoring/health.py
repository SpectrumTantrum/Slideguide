"""
Health check and metrics endpoints.

GET /health      — Full health check with dependency status
GET /health/live — Kubernetes liveness (is the process alive?)
GET /health/ready — Kubernetes readiness (can we serve traffic?)
GET /metrics     — Aggregated cost, latency, and error metrics
"""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter

from backend.config import settings
from backend.llm.runtime import get_active_provider
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


async def _check_endpoint() -> dict:
    """Check the active chat provider SDK and the embeddings endpoint."""
    from backend.llm.discovery import check_active_provider_health, check_endpoint_health

    chat = await check_active_provider_health()
    embeddings = await check_endpoint_health(settings.openai_base_url)
    return {"chat": chat, "embeddings": embeddings}


@router.get("/health")
async def health_check() -> dict:
    """Full health check with all dependency statuses."""
    checks: dict = {
        "supabase": _check_supabase(),
    }

    endpoint_health = await _check_endpoint()
    checks["chat_provider"] = get_active_provider()
    checks["llm_endpoint"] = endpoint_health["chat"]["status"]
    checks["llm_endpoint_models"] = endpoint_health["chat"]["models_loaded"]
    checks["embeddings_endpoint"] = endpoint_health["embeddings"]["status"]

    all_ok = all(
        v == "ok" for k, v in checks.items()
        if isinstance(v, str) and k not in {"llm_endpoint_models", "chat_provider"}
    )
    # Supabase and the active chat SDK are required. Embeddings may be down
    # independently when chat is billed to Cursor.
    critical_ok = checks["supabase"] == "ok" and checks.get("llm_endpoint") == "ok"

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
        "endpoint": {
            "provider": get_active_provider(),
            "base_url": settings.openai_base_url,
            "primary_model": settings.active_primary_model,
            "embedding_model": settings.active_embedding_model,
        },
        **summary,
        "models": model_stats,
        "sessions": {
            "active": metrics.active_sessions,
            "total": metrics.total_sessions,
        },
    }
