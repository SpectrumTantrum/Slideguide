"""
Model configuration, pricing, and fallback chains.

Centralizes model metadata so cost calculations and model selection
are consistent across the application.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from backend.config import settings


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for a single LLM model."""

    model_id: str
    display_name: str
    cost_per_1k_input: float  # USD
    cost_per_1k_output: float  # USD
    max_tokens: int = 4096
    supports_tools: bool = True
    supports_vision: bool = False
    context_window: int = 200_000


# Model registry — all models available via OpenRouter
MODELS: dict[str, ModelConfig] = {
    "anthropic/claude-sonnet-4": ModelConfig(
        model_id="anthropic/claude-sonnet-4",
        display_name="Claude Sonnet 4",
        cost_per_1k_input=0.003,
        cost_per_1k_output=0.015,
        max_tokens=8192,
        supports_tools=True,
        supports_vision=True,
        context_window=200_000,
    ),
    "anthropic/claude-haiku-4": ModelConfig(
        model_id="anthropic/claude-haiku-4",
        display_name="Claude Haiku 4",
        cost_per_1k_input=0.0008,
        cost_per_1k_output=0.004,
        max_tokens=8192,
        supports_tools=True,
        supports_vision=False,
        context_window=200_000,
    ),
    "deepseek/deepseek-chat-v3": ModelConfig(
        model_id="deepseek/deepseek-chat-v3",
        display_name="DeepSeek Chat V3",
        cost_per_1k_input=0.00014,
        cost_per_1k_output=0.00028,
        max_tokens=8192,
        supports_tools=True,
        supports_vision=False,
        context_window=64_000,
    ),
    "text-embedding-3-small": ModelConfig(
        model_id="text-embedding-3-small",
        display_name="OpenAI Embedding 3 Small",
        cost_per_1k_input=0.00002,
        cost_per_1k_output=0.0,
        max_tokens=0,
        supports_tools=False,
        supports_vision=False,
        context_window=8191,
    ),
}

# Fallback chains: if primary fails, try these in order
FALLBACK_CHAIN: list[str] = [
    settings.primary_model,
    settings.fallback_model,
]


def get_model(
    purpose: Literal["reasoning", "routing", "vision", "embedding", "fallback"],
) -> ModelConfig:
    """Get the appropriate model config for a given purpose."""
    model_map = {
        "reasoning": settings.primary_model,
        "routing": settings.routing_model,
        "vision": settings.vision_model,
        "embedding": settings.embedding_model,
        "fallback": settings.fallback_model,
    }
    model_id = model_map[purpose]
    return MODELS.get(model_id, MODELS[settings.primary_model])


def estimate_cost(model_id: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate cost in USD for a model call."""
    config = MODELS.get(model_id)
    if config:
        return (
            input_tokens / 1000 * config.cost_per_1k_input
            + output_tokens / 1000 * config.cost_per_1k_output
        )
    # Fallback estimate
    return (input_tokens + output_tokens) / 1000 * 0.002
