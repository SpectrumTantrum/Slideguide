"""
Chat-SDK selection. This is the only module that decides which SDK bills
a tutoring turn.

``LLM_PROVIDER`` is the process default. A session may override that for
itself (stored on ``sessions.metadata.chat_sdk``). There is no process-wide
override — one tab cannot change another session's bill.
"""

from __future__ import annotations

from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import Any, Literal

from backend.config import settings

SUPPORTED_PROVIDERS = ("openai", "cursor")
ChatSdk = Literal["openai", "cursor"]
ChatPurpose = Literal["chat", "route"]


@dataclass(frozen=True)
class ChatContext:
    """Request-scoped billing target."""

    provider: str
    session_id: str | None = None


_ctx: ContextVar[ChatContext | None] = ContextVar("slideguide_chat_ctx", default=None)


@dataclass(frozen=True)
class ChatModels:
    """Models for one chat SDK. Cursor never inherits OpenAI-only ids."""

    primary: str
    routing: str
    vision: str
    effort: str | None
    fallback: list[str]


def default_provider() -> str:
    """Process default from ``LLM_PROVIDER``."""
    return settings.llm_provider


def cursor_is_configured() -> bool:
    """True when a Cursor API key is available for the Python SDK."""
    return bool(settings.cursor_api_key.strip())


def parse_provider(provider: str) -> str:
    """Validate a provider id. Does not require credentials or change state."""
    normalized = (provider or "").strip().lower()
    if normalized not in SUPPORTED_PROVIDERS:
        raise ValueError(
            f"Unknown provider '{provider}'. Expected one of: {SUPPORTED_PROVIDERS}"
        )
    return normalized


def normalize_provider(provider: str) -> str:
    """Validate a provider id. Cursor also requires ``CURSOR_API_KEY``."""
    normalized = parse_provider(provider)
    if normalized == "cursor" and not cursor_is_configured():
        raise ValueError(
            "Cursor is not configured. Set CURSOR_API_KEY to bill Cursor usage."
        )
    return normalized


def resolve_provider(requested: str | None = None) -> str:
    """Resolve a request/session choice, falling back to ``LLM_PROVIDER``.

    Always runs credential checks (e.g. ``CURSOR_API_KEY`` when the
    resolved SDK is Cursor), including when the caller omitted ``requested``
    and we fall back to the process default.
    """
    return normalize_provider(requested or default_provider())


def clear_chat_context() -> None:
    """Drop the request-scoped bind. Used by tests; never a process override."""
    _ctx.set(None)


def bind_chat_sdk(provider: str, session_id: str | None = None) -> Token[ChatContext | None]:
    """Bind the SDK for this task only. Reset with ``reset_chat_sdk``."""
    return _ctx.set(ChatContext(provider=resolve_provider(provider), session_id=session_id))


def reset_chat_sdk(token: Token[ChatContext | None]) -> None:
    _ctx.reset(token)


def current_context() -> ChatContext | None:
    return _ctx.get()


def current_chat_sdk() -> str:
    """SDK bound for this request, or the process default."""
    ctx = _ctx.get()
    if ctx:
        return ctx.provider
    return default_provider()


def current_session_id() -> str | None:
    ctx = _ctx.get()
    return ctx.session_id if ctx else None


def session_chat_sdk(session: dict[str, Any] | None) -> str | None:
    """Read ``chat_sdk`` from a sessions row (``metadata`` JSON)."""
    if not session:
        return None
    meta = session.get("metadata") or {}
    if not isinstance(meta, dict):
        return None
    raw = meta.get("chat_sdk")
    return raw.strip().lower() if isinstance(raw, str) and raw.strip() else None


def models_for(provider: str | None = None, purpose: ChatPurpose = "chat") -> ChatModels:
    """
    Resolve model ids for a chat SDK.

    Cursor ignores ``ROUTING_MODEL`` / ``VISION_MODEL`` / ``PRIMARY_MODEL``
    unless those ids are Cursor-owned. Routing on Cursor stays on a cheap
    Composer id so JSON classifier nodes do not become Grok-high agent runs.
    """
    from backend.llm.models import (
        CURSOR_DEFAULT_MODEL,
        CURSOR_ROUTING_MODELS,
        cursor_fallback_chain,
        is_cursor_owned_model,
    )

    sdk = parse_provider(provider) if provider else current_chat_sdk()
    if sdk == "cursor":
        primary = (settings.cursor_model or "").strip() or CURSOR_DEFAULT_MODEL
        routing_override = (settings.routing_model or "").strip()
        if routing_override and is_cursor_owned_model(routing_override):
            routing = routing_override
        else:
            routing = CURSOR_ROUTING_MODELS[0]
        vision_override = (settings.vision_model or "").strip()
        vision = (
            vision_override
            if vision_override and is_cursor_owned_model(vision_override)
            else primary
        )
        fallback = (
            list(CURSOR_ROUTING_MODELS)
            if purpose == "route"
            else cursor_fallback_chain(primary)
        )
        return ChatModels(
            primary=primary,
            routing=routing,
            vision=vision,
            effort=settings.cursor_reasoning_effort,
            fallback=fallback,
        )

    primary = settings.primary_model
    routing = settings.routing_model or settings.primary_model
    return ChatModels(
        primary=primary,
        routing=routing,
        vision=settings.vision_model,
        effort=None,
        fallback=[primary] if primary else [],
    )


def tool_mode_for(provider: str | None = None) -> str:
    """Cursor has no OpenAI-format tools; that route is always prompt-based."""
    sdk = parse_provider(provider) if provider else current_chat_sdk()
    return "prompt" if sdk == "cursor" else "native"


# Back-compat aliases used by older tests / health. These never mutate process
# state — they only read the request context or the env default.
def get_active_provider() -> str:
    return current_chat_sdk()
