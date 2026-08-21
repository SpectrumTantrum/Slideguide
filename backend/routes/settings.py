"""
Settings and provider information API routes.

GET is a catalog + the process default (or one session's stored SDK).
POST requires ``session_id`` and only changes that session. It never
flips a process-wide billing cell.
"""

from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from backend.config import settings
from backend.db.repositories.sessions import SessionRepository
from backend.llm.discovery import check_chat_health, check_endpoint_health, list_models
from backend.llm.providers import available_providers, provider_metadata
from backend.llm.runtime import (
    default_provider,
    models_for,
    parse_provider,
    resolve_provider,
    session_chat_sdk,
    tool_mode_for,
)
from backend.monitoring.logger import get_logger

router = APIRouter(prefix="/api/settings", tags=["settings"])
logger = get_logger(__name__)

_tool_compat = None


def set_tool_compat(tc: Any) -> None:
    global _tool_compat
    _tool_compat = tc


class ProviderSwitchRequest(BaseModel):
    """Select which SDK bills one tutoring session."""

    provider: Literal["openai", "cursor"] = Field(..., description="Chat SDK id")
    session_id: str | None = Field(
        default=None,
        description="Session whose bill should change. Required — there is no process override.",
    )


def _provider_payload(sdk: str, *, session_id: str | None = None) -> dict[str, Any]:
    meta = provider_metadata(sdk)
    resolved = models_for(sdk)
    tool_mode = tool_mode_for(sdk)
    if sdk != "cursor" and _tool_compat is not None:
        # Use the learned OpenAI-compat mode, not mode (which reads request ContextVar).
        tool_mode = getattr(_tool_compat, "learned_mode", getattr(_tool_compat, "_mode", tool_mode))
    cloud = settings.cursor_runtime == "cloud"
    return {
        "provider": sdk,
        "sdk": meta["sdk"],
        "usage": meta["usage"],
        "label": meta["label"],
        "scope": "session" if session_id else "process_default",
        "session_id": session_id,
        "base_url": settings.openai_base_url if sdk == "openai" else "cursor-sdk",
        "runtime": settings.cursor_runtime if sdk == "cursor" else None,
        "cursor": {
            "runtime": settings.cursor_runtime,
            "local_tools_disabled": sdk == "cursor" and not cloud,
            # Cloud tutoring is refused in CursorTranslator; flag stays truthful.
            "cloud_loads_team_tools": sdk == "cursor" and cloud,
            "tutoring_requires_local": True,
        },
        "capabilities": {
            "vision": bool(resolved.vision) or sdk == "cursor",
            "tool_mode": tool_mode,
        },
        "models": {
            "primary": resolved.primary,
            "routing": resolved.routing,
            "embedding": settings.embedding_model,
            "vision": resolved.vision,
            "effort": resolved.effort,
        },
        "available_providers": available_providers(),
    }


async def _session_sdk(request: Request, session_id: str) -> str:
    supabase = getattr(request.app.state, "supabase", None)
    if supabase is None:
        raise HTTPException(status_code=503, detail="Session store is unavailable")
    session = SessionRepository(supabase).get_by_id(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    stored = session_chat_sdk(session)
    return parse_provider(stored) if stored else default_provider()


@router.get("/provider")
async def get_provider_status(
    request: Request,
    session_id: str | None = None,
    provider: str | None = None,
) -> dict[str, Any]:
    """
    Return SDK catalog and models.

    ``session_id`` shows that session's stored SDK. ``provider`` previews a
    catalog without applying it. Otherwise the process default is shown.
    """
    try:
        if session_id:
            sdk = await _session_sdk(request, session_id)
        elif provider:
            sdk = parse_provider(provider)
        else:
            sdk = default_provider()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    payload = _provider_payload(sdk, session_id=session_id)
    payload["endpoint"] = await check_chat_health(sdk)
    payload["embeddings"] = await check_endpoint_health(settings.openai_base_url)
    return payload


@router.post("/provider")
async def switch_provider(request: Request, body: ProviderSwitchRequest) -> dict[str, Any]:
    """Change the chat SDK for one session. Resets that SDK's circuit breaker."""
    if not (body.session_id or "").strip():
        raise HTTPException(
            status_code=400,
            detail="session_id is required; the chat SDK switch is per session",
        )

    try:
        sdk = resolve_provider(body.provider)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    supabase = getattr(request.app.state, "supabase", None)
    if supabase is None:
        raise HTTPException(status_code=503, detail="Session store is unavailable")
    session_repo = SessionRepository(supabase)
    session = session_repo.get_by_id(body.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    meta = dict(session.get("metadata") or {})
    meta["chat_sdk"] = sdk
    session_repo.update(body.session_id, metadata=meta)

    try:
        from backend.routes.chat import get_graph

        get_graph().update_state(
            {"configurable": {"thread_id": body.session_id}},
            {"chat_sdk": sdk},
        )
    except Exception as exc:
        logger.warning(
            "provider_switch_graph_state_failed",
            session_id=body.session_id,
            chat_sdk=sdk,
            error=str(exc),
        )

    from backend.agent.nodes import llm

    llm.reset_breaker(sdk)
    return await get_provider_status(request, session_id=body.session_id)


@router.get("/models")
async def get_available_models(
    request: Request,
    session_id: str | None = None,
    provider: str | None = None,
) -> dict[str, Any]:
    try:
        if session_id:
            sdk = await _session_sdk(request, session_id)
        elif provider:
            sdk = parse_provider(provider)
        else:
            sdk = default_provider()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "provider": sdk,
        "sdk": provider_metadata(sdk)["sdk"],
        "models": await list_models(sdk),
    }
