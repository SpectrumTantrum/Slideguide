"""
LLM client wrappers for chat and embeddings.

Chat is dispatched to the active provider SDK (OpenAI or Cursor) so
different subscriptions can bill tutoring usage. Embeddings always use
the OpenAI-compatible endpoint.

LLMClient handles:
- Provider SDK dispatch
- Retry with exponential backoff (3 attempts)
- Circuit breaker (opens after 5 consecutive failures)
- Automatic cost tracking via MetricsCollector
- Cursor-first model fallback when the Cursor route is active
"""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import AsyncGenerator
from typing import Any

import openai

from backend.config import settings
from backend.llm.models import get_fallback_chain
from backend.llm.providers import create_chat_provider, get_embedding_config
from backend.llm.runtime import get_active_provider
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


def _is_retryable(exc: Exception) -> bool:
    """True for transient provider/SDK failures worth retrying."""
    if isinstance(exc, (openai.APITimeoutError, openai.APIConnectionError)):
        return True
    if isinstance(exc, openai.APIStatusError):
        return exc.status_code in (429, 500, 502, 503)
    if getattr(exc, "is_retryable", False):
        return True
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    return status in (429, 500, 502, 503)


class LLMClient:
    """
    LLM client with retry, circuit breaker, and cost tracking.

    Resolves the active provider SDK on each call so the app can switch
    subscriptions (Cursor vs OpenAI-compatible) without restarting.
    """

    def __init__(self) -> None:
        self._circuit = CircuitBreaker()
        self._providers: dict[str, Any] = {}

    def _adapter(self) -> Any:
        name = get_active_provider()
        if name not in self._providers:
            self._providers[name] = create_chat_provider(name)
        return self._providers[name]

    @property
    def provider(self) -> str:
        """Name of the active chat provider SDK."""
        return get_active_provider()

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
        adapter = self._adapter()

        try:
            async for chunk in adapter.stream_chat(
                messages=messages,
                model=model,
                tools=tools,
                temperature=temperature,
                max_tokens=max_tokens,
            ):
                yield chunk
            self._circuit.record_success()
        except Exception as e:
            self._circuit.record_failure()
            logger.error("stream_chat_failed", model=model, provider=self.provider, error=str(e))
            raise

    async def _call_with_retry(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        temperature: float,
        max_tokens: int,
    ) -> dict[str, Any]:
        """Execute an API call with retry logic and model fallback."""
        models_to_try = [model] + [m for m in get_fallback_chain() if m != model]
        adapter = self._adapter()

        for model_id in models_to_try:
            if not self._circuit.can_execute():
                logger.warning("circuit_breaker_blocking", model=model_id)
                continue

            for attempt in range(MAX_RETRIES):
                try:
                    start_time = time.perf_counter()
                    response = await adapter.chat(
                        messages=messages,
                        model=model_id,
                        tools=tools,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    elapsed_ms = (time.perf_counter() - start_time) * 1000
                    self._circuit.record_success()

                    usage = response.get("usage") or {}
                    prompt_tokens = usage.get("prompt_tokens") or 0
                    completion_tokens = usage.get("completion_tokens") or 0
                    if prompt_tokens or completion_tokens:
                        metrics.record_llm_call(
                            model=model_id,
                            input_tokens=prompt_tokens,
                            output_tokens=completion_tokens,
                            latency_ms=elapsed_ms,
                            operation="chat",
                            provider=self.provider,
                        )

                    return response

                except Exception as e:
                    if _is_retryable(e) and attempt < MAX_RETRIES - 1:
                        delay = min(
                            BASE_DELAY * (2**attempt) + random.uniform(0, JITTER),
                            MAX_DELAY,
                        )
                        logger.warning(
                            "llm_retry",
                            model=model_id,
                            provider=self.provider,
                            attempt=attempt + 1,
                            delay=round(delay, 1),
                            error=str(e),
                        )
                        await asyncio.sleep(delay)
                        continue
                    if _is_retryable(e):
                        break
                    self._circuit.record_failure()
                    raise

            self._circuit.record_failure()
            logger.error("llm_model_exhausted", model=model_id, provider=self.provider)

        raise openai.APIConnectionError(
            message="All models in fallback chain exhausted",
            request=None,  # type: ignore[arg-type]
        )


class EmbeddingClient:
    """Embedding client using the configured OpenAI-compatible endpoint."""

    def __init__(self) -> None:
        self._provider_config = get_embedding_config()
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
