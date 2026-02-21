"""
LLM client wrappers for chat and embeddings.

Supports multiple providers (OpenRouter, LM Studio) via the same
OpenAI-compatible SDK. Provider selection is driven by config.

LLMClient handles:
- Chat completions via configurable provider
- Retry with exponential backoff (3 attempts)
- Circuit breaker (opens after 5 consecutive failures)
- Automatic cost tracking via MetricsCollector
- Model fallback chain

EmbeddingClient handles embeddings via configurable provider.
"""

from __future__ import annotations

import asyncio
import random
import time
from typing import Any, AsyncGenerator

import openai

from backend.config import settings
from backend.llm.models import get_fallback_chain, estimate_cost
from backend.llm.providers import get_chat_provider_config, get_embedding_provider_config
from backend.monitoring.logger import get_logger
from backend.monitoring.metrics import metrics

logger = get_logger(__name__)

# Retry config
MAX_RETRIES = 3
BASE_DELAY = 1.0  # seconds
MAX_DELAY = 30.0
JITTER = 0.5

# Circuit breaker config
CIRCUIT_FAILURE_THRESHOLD = 5
CIRCUIT_RESET_TIMEOUT = 30.0  # seconds


class CircuitBreaker:
    """Simple circuit breaker for external API calls."""

    def __init__(self, failure_threshold: int = CIRCUIT_FAILURE_THRESHOLD,
                 reset_timeout: float = CIRCUIT_RESET_TIMEOUT):
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.failure_count = 0
        self.last_failure_time: float = 0
        self.state: str = "closed"  # closed, open, half-open

    def can_execute(self) -> bool:
        """Check if the circuit allows execution."""
        if self.state == "closed":
            return True
        if self.state == "open":
            if time.time() - self.last_failure_time > self.reset_timeout:
                self.state = "half-open"
                return True
            return False
        # half-open: allow one attempt
        return True

    def record_success(self) -> None:
        self.failure_count = 0
        self.state = "closed"

    def record_failure(self) -> None:
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.state = "open"
            logger.warning(
                "circuit_breaker_opened",
                failure_count=self.failure_count,
                reset_timeout=self.reset_timeout,
            )


class LLMClient:
    """
    LLM client with retry, circuit breaker, and cost tracking.

    Uses the OpenAI SDK pointed at the active provider's base URL
    (OpenRouter or LM Studio).
    """

    def __init__(self) -> None:
        self._provider_config = get_chat_provider_config()
        self._client = openai.AsyncOpenAI(**self._provider_config.client_kwargs())
        self._circuit = CircuitBreaker()

    @property
    def provider(self) -> str:
        """Name of the active provider (e.g., 'openrouter', 'lmstudio')."""
        return self._provider_config.name

    async def chat(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> dict[str, Any]:
        """
        Non-streaming chat completion with retry and fallback.

        Returns the full ChatCompletion response as a dict.
        """
        model = model or settings.active_primary_model
        return await self._call_with_retry(
            model=model,
            messages=messages,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,
        )

    async def stream_chat(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """
        Streaming chat completion yielding delta chunks.

        Each chunk is a dict with: choices[0].delta.content, etc.
        """
        model = model or settings.active_primary_model

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        if tools:
            kwargs["tools"] = tools

        start_time = time.perf_counter()

        try:
            stream = await self._client.chat.completions.create(**kwargs)

            async for chunk in stream:
                yield {
                    "id": chunk.id,
                    "choices": [
                        {
                            "delta": {
                                "content": chunk.choices[0].delta.content if chunk.choices else None,
                                "tool_calls": (
                                    [tc.model_dump() for tc in chunk.choices[0].delta.tool_calls]
                                    if chunk.choices and chunk.choices[0].delta.tool_calls
                                    else None
                                ),
                                "role": chunk.choices[0].delta.role if chunk.choices else None,
                            },
                            "finish_reason": chunk.choices[0].finish_reason if chunk.choices else None,
                        }
                    ],
                }

            self._circuit.record_success()

        except Exception as e:
            self._circuit.record_failure()
            logger.error("stream_chat_failed", model=model, error=str(e))
            raise

    async def _call_with_retry(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        temperature: float,
        max_tokens: int,
        stream: bool,
    ) -> dict[str, Any]:
        """Execute an API call with retry logic and model fallback."""
        models_to_try = [model] + [m for m in get_fallback_chain() if m != model]

        for model_id in models_to_try:
            if not self._circuit.can_execute():
                logger.warning("circuit_breaker_blocking", model=model_id)
                continue

            for attempt in range(MAX_RETRIES):
                try:
                    start_time = time.perf_counter()

                    kwargs: dict[str, Any] = {
                        "model": model_id,
                        "messages": messages,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                    }
                    if tools:
                        kwargs["tools"] = tools

                    response = await self._client.chat.completions.create(**kwargs)
                    elapsed_ms = (time.perf_counter() - start_time) * 1000

                    self._circuit.record_success()

                    # Track cost
                    usage = response.usage
                    if usage:
                        metrics.record_llm_call(
                            model=model_id,
                            input_tokens=usage.prompt_tokens or 0,
                            output_tokens=usage.completion_tokens or 0,
                            latency_ms=elapsed_ms,
                            operation="chat",
                            provider=self.provider,
                        )

                    return response.model_dump()

                except (openai.APITimeoutError, openai.APIConnectionError) as e:
                    delay = min(BASE_DELAY * (2**attempt) + random.uniform(0, JITTER), MAX_DELAY)
                    logger.warning(
                        "llm_retry",
                        model=model_id,
                        attempt=attempt + 1,
                        delay=round(delay, 1),
                        error=str(e),
                    )
                    await asyncio.sleep(delay)

                except openai.APIStatusError as e:
                    if e.status_code in (429, 500, 502, 503):
                        delay = min(BASE_DELAY * (2**attempt) + random.uniform(0, JITTER), MAX_DELAY)
                        logger.warning(
                            "llm_retry",
                            model=model_id,
                            attempt=attempt + 1,
                            status_code=e.status_code,
                            delay=round(delay, 1),
                        )
                        await asyncio.sleep(delay)
                    else:
                        self._circuit.record_failure()
                        raise

            # All retries exhausted for this model
            self._circuit.record_failure()
            logger.error("llm_model_exhausted", model=model_id)

        raise openai.APIConnectionError(
            message="All models in fallback chain exhausted",
            request=None,
        )


class EmbeddingClient:
    """Embedding client using the active provider (OpenAI or LM Studio)."""

    def __init__(self) -> None:
        self._provider_config = get_embedding_provider_config()
        self._client = openai.AsyncOpenAI(**self._provider_config.client_kwargs())

    @property
    def provider(self) -> str:
        """Name of the active embedding provider."""
        return self._provider_config.name

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts."""
        response = await self._client.embeddings.create(
            model=settings.active_embedding_model,
            input=texts,
        )

        if response.usage:
            metrics.record_llm_call(
                model=settings.active_embedding_model,
                input_tokens=response.usage.total_tokens,
                output_tokens=0,
                latency_ms=0,
                operation="embedding",
                provider=self.provider,
            )

        return [item.embedding for item in response.data]

    async def embed_query(self, query: str) -> list[float]:
        """Embed a single query string."""
        result = await self.embed([query])
        return result[0]
