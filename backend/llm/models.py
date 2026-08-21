"""
Model selection helpers.

With a single OpenAI-compatible endpoint there is no multi-model fallback
chain; the configured primary model is used. Cost estimation lives in
``backend.monitoring.metrics`` (unknown models are tracked at $0.00).
"""

from __future__ import annotations

from backend.config import settings


def get_fallback_chain() -> list[str]:
    """Return the ordered list of models to try (just the primary model)."""
    return [settings.active_primary_model]
