"""
Model selection helpers.

On the OpenAI-compatible route the configured primary model is used.
On the Cursor SDK route, Cursor first-party models are preferred first
(Grok 4.6 at high effort, then Composer, then Router).
"""

from __future__ import annotations

from dataclasses import dataclass

from backend.config import settings
from backend.llm.runtime import get_active_provider

# Cursor first-party ids. Order is the default try-list on that route.
CURSOR_DEFAULT_MODEL = "grok-4.6"
CURSOR_DEFAULT_EFFORT = "high"
CURSOR_PREFERRED_MODELS: tuple[str, ...] = (
    "grok-4.6",
    "composer-2.5",
    "composer-2",
    "auto-smart",
)

_CURSOR_OWNED_EXACT = frozenset({"auto-smart", "auto"})
_GROK_EFFORT_PARAM = "reasoning_effort"


@dataclass(frozen=True)
class CursorModelRequest:
    """Resolved Cursor model id plus SDK params (e.g. Grok effort=high)."""

    id: str
    params: tuple[tuple[str, str], ...] = ()


def is_grok_model(model_id: str) -> bool:
    """True for Grok ids served through Cursor (``grok-4.6``, …)."""
    return (model_id or "").strip().lower().startswith("grok-")


def is_cursor_owned_model(model_id: str) -> bool:
    """True for Cursor first-party ids (Grok, Composer, Router)."""
    mid = (model_id or "").strip().lower()
    if not mid:
        return False
    return (
        mid.startswith("composer-")
        or mid.startswith("grok-")
        or mid in _CURSOR_OWNED_EXACT
    )


def cursor_model_request(model_id: str = "", effort: str | None = None) -> CursorModelRequest:
    """Build the Cursor SDK model selection, defaulting Grok to high effort."""
    mid = (model_id or "").strip() or CURSOR_DEFAULT_MODEL
    raw_effort = effort if effort is not None else settings.cursor_reasoning_effort
    resolved_effort = raw_effort.strip().lower()
    params: tuple[tuple[str, str], ...] = ()
    if is_grok_model(mid) and resolved_effort:
        params = ((_GROK_EFFORT_PARAM, resolved_effort),)
    return CursorModelRequest(id=mid, params=params)


def cursor_model_chain(primary: str = "") -> list[str]:
    """
    Fallback list for the Cursor route.

    An explicit Cursor-owned primary stays first. Other first-party models
    follow. A non-Cursor primary is appended after those.
    """
    chain: list[str] = []
    primary = (primary or "").strip()
    if primary and is_cursor_owned_model(primary):
        chain.append(primary)
    for model_id in CURSOR_PREFERRED_MODELS:
        if model_id not in chain:
            chain.append(model_id)
    if primary and primary not in chain:
        chain.append(primary)
    return chain


def prefer_cursor_models(model_ids: list[str]) -> list[str]:
    """Stable-sort a catalog so Cursor-owned models come first."""
    rank = {model_id: index for index, model_id in enumerate(CURSOR_PREFERRED_MODELS)}
    seen: set[str] = set()
    unique: list[str] = []
    for model_id in model_ids:
        if model_id and model_id not in seen:
            seen.add(model_id)
            unique.append(model_id)

    cursor_owned = [m for m in unique if is_cursor_owned_model(m)]
    others = [m for m in unique if not is_cursor_owned_model(m)]
    cursor_owned.sort(key=lambda mid: (rank.get(mid, 100), mid.lower()))
    return cursor_owned + others


def get_fallback_chain(provider: str | None = None) -> list[str]:
    """Return the ordered list of models to try for the active (or given) provider."""
    provider = provider or get_active_provider()
    if provider == "cursor":
        chain = cursor_model_chain(settings.cursor_model or CURSOR_DEFAULT_MODEL)
        extra = (settings.primary_model or "").strip()
        if extra and extra not in chain:
            chain.append(extra)
        return chain
    return [settings.primary_model]
