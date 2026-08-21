"""
LLM client wrappers for chat and embeddings.

OpenAI-compatible HTTP stays here. Cursor is one translator
(``backend.llm.cursor``). Which SDK bills a turn comes from
``backend.llm.runtime`` (request/session), never a process-wide cell.

Embeddings always use the OpenAI-compatible endpoint.
"""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import AsyncGenerator
from typing import Any

import openai

from backend.config import settings
from backend.llm.cursor import CursorTranslator
from backend.llm.providers import get_embedding_config
from backend.llm.runtime import current_chat_sdk, models_for
from backend.monitoring.logger import get_logger
from backend.monitoring.metrics import metrics

logger = get_logger(__name__)

MAX_RETRIES = 3
BASE_DELAY = 1.0
MAX_DELAY = 30.0
JITTER = 0.5
CIRCUIT_FAILURE_THRESHOLD = 5
CIRCUIT_RESET_TIMEOUT = 30.0


class AllModelsExhaustedError(RuntimeError):
    """Raised when every model in the fallback chain failed."""


class CircuitBreaker:
    """Simple circuit breaker for one chat SDK."""

    def __init__(
        self,
        failure_threshold: int = CIRCUIT_FAILURE_THRESHOLD,
        reset_timeout: float = CIRCUIT_RESET_TIMEOUT,
    ):
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.failure_count = 0
        self.last_failure_time: float = 0
        self.state: str = "closed"

    def can_execute(self) -> bool:
        if self.state == "closed":
            return True
        if self.state == "open":
            if time.time() - self.last_failure_time > self.reset_timeout:
                self.state = "half-open"
                return True
            return False
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

    def reset(self) -> None:
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = "closed"


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


def _is_unknown_model(exc: Exception) -> bool:
    """True when this model id is unusable — try the next fallback id."""
    if isinstance(exc, openai.NotFoundError):
        return True
    if isinstance(exc, openai.APIStatusError) and exc.status_code in (400, 404):
        text = str(exc).lower()
        return "model" in text
    text = str(exc).lower()
    return "model" in text and any(
        token in text for token in ("not found", "unknown", "invalid", "does not exist")
    )


def _completion_dict(
    content: str,
    *,
    model: str,
    completion_id: str = "",
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
) -> dict[str, Any]:
    return {
        "id": completion_id or "chatcmpl-slideguide",
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                    "tool_calls": None,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def _stream_chunk(
    text: str | None,
    *,
    completion_id: str = "",
    finish_reason: str | None = None,
    role: str | None = None,
) -> dict[str, Any]:
    return {
        "id": completion_id or "chatcmpl-slideguide",
        "choices": [
            {
                "delta": {"content": text, "tool_calls": None, "role": role},
                "finish_reason": finish_reason,
            }
        ],
    }


class LLMClient:
    """
    Chat + embeddings with retry, per-SDK circuit breakers, and cost tracking.

    Pass ``purpose="route"`` for cheap JSON classifier nodes so a Cursor
    session does not spend a Grok-high agent run on intent routing.
    """

    def __init__(self) -> None:
        self._breakers: dict[str, CircuitBreaker] = {}
        self._openai = openai.AsyncOpenAI(**get_embedding_config().client_kwargs())
        self._cursor = CursorTranslator()

    def _breaker(self, provider: str) -> CircuitBreaker:
        if provider not in self._breakers:
            self._breakers[provider] = CircuitBreaker()
        return self._breakers[provider]

    def reset_breaker(self, provider: str | None = None) -> None:
        """Close the breaker for one SDK, or every SDK. Called on session switch."""
        if provider:
            self._breaker(provider).reset()
            return
        for breaker in self._breakers.values():
            breaker.reset()

    @property
    def provider(self) -> str:
        return current_chat_sdk()

    async def chat(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        purpose: str = "chat",
    ) -> dict[str, Any]:
        sdk = current_chat_sdk()
        resolved = models_for(sdk, purpose=purpose)  # type: ignore[arg-type]
        model = model or (resolved.routing if purpose == "route" else resolved.primary)
        return await self._call_with_retry(
            provider=sdk,
            model=model,
            messages=messages,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
            purpose=purpose,
        )

    async def stream_chat(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        purpose: str = "chat",
    ) -> AsyncGenerator[dict[str, Any], None]:
        sdk = current_chat_sdk()
        resolved = models_for(sdk, purpose=purpose)  # type: ignore[arg-type]
        model = model or (resolved.routing if purpose == "route" else resolved.primary)
        breaker = self._breaker(sdk)
        if not breaker.can_execute():
            logger.warning("circuit_breaker_blocking", model=model, provider=sdk)
            raise AllModelsExhaustedError(
                f"Circuit breaker open for provider '{sdk}'; streaming unavailable"
            )
        try:
            async for chunk in self._stream(
                sdk, messages, model, tools, temperature, max_tokens
            ):
                yield chunk
            breaker.record_success()
        except Exception as e:
            breaker.record_failure()
            logger.error("stream_chat_failed", model=model, provider=sdk, error=str(e))
            raise

    async def _stream(
        self,
        provider: str,
        messages: list[dict[str, Any]],
        model: str,
        tools: list[dict[str, Any]] | None,
        temperature: float,
        max_tokens: int,
    ) -> AsyncGenerator[dict[str, Any], None]:
        if provider == "cursor":
            completion_id = f"cursor-{model}"
            async for text in self._cursor.stream_text(messages, model):
                yield _stream_chunk(text, completion_id=completion_id, role="assistant")
            yield _stream_chunk(None, completion_id=completion_id, finish_reason="stop")
            return

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        if tools:
            kwargs["tools"] = tools
        stream = await self._openai.chat.completions.create(**kwargs)
        async for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            yield {
                "id": chunk.id,
                "choices": [
                    {
                        "delta": {
                            "content": delta.content if delta else None,
                            "tool_calls": (
                                [tc.model_dump() for tc in delta.tool_calls]
                                if delta and delta.tool_calls
                                else None
                            ),
                            "role": delta.role if delta else None,
                        },
                        "finish_reason": (
                            chunk.choices[0].finish_reason if chunk.choices else None
                        ),
                    }
                ],
            }

    async def _call_with_retry(
        self,
        provider: str,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        temperature: float,
        max_tokens: int,
        purpose: str,
    ) -> dict[str, Any]:
        resolved = models_for(provider, purpose=purpose)  # type: ignore[arg-type]
        models_to_try = [model] + [m for m in resolved.fallback if m != model]
        breaker = self._breaker(provider)

        for model_id in models_to_try:
            if not breaker.can_execute():
                logger.warning("circuit_breaker_blocking", model=model_id, provider=provider)
                continue

            for attempt in range(MAX_RETRIES):
                try:
                    start_time = time.perf_counter()
                    response = await self._once(
                        provider, messages, model_id, tools, temperature, max_tokens
                    )
                    elapsed_ms = (time.perf_counter() - start_time) * 1000
                    breaker.record_success()

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
                            provider=provider,
                        )
                    return response

                except Exception as e:
                    if _is_unknown_model(e):
                        logger.warning(
                            "llm_unknown_model",
                            model=model_id,
                            provider=provider,
                            error=str(e),
                        )
                        break
                    if _is_retryable(e) and attempt < MAX_RETRIES - 1:
                        delay = min(
                            BASE_DELAY * (2**attempt) + random.uniform(0, JITTER),
                            MAX_DELAY,
                        )
                        logger.warning(
                            "llm_retry",
                            model=model_id,
                            provider=provider,
                            attempt=attempt + 1,
                            delay=round(delay, 1),
                            error=str(e),
                        )
                        await asyncio.sleep(delay)
                        continue
                    if _is_retryable(e):
                        break
                    breaker.record_failure()
                    raise

            breaker.record_failure()
            logger.error("llm_model_exhausted", model=model_id, provider=provider)

        raise AllModelsExhaustedError("All models in fallback chain exhausted")

    async def _once(
        self,
        provider: str,
        messages: list[dict[str, Any]],
        model: str,
        tools: list[dict[str, Any]] | None,
        temperature: float,
        max_tokens: int,
    ) -> dict[str, Any]:
        if provider == "cursor":
            reply = await self._cursor.chat_async(messages, model)
            return _completion_dict(
                reply.text,
                model=reply.model,
                completion_id=reply.run_id,
                prompt_tokens=reply.prompt_tokens,
                completion_tokens=reply.completion_tokens,
            )

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            kwargs["tools"] = tools
        response = await self._openai.chat.completions.create(**kwargs)
        return response.model_dump()


class EmbeddingClient:
    """Embedding client using the configured OpenAI-compatible endpoint."""

    def __init__(self) -> None:
        self._provider_config = get_embedding_config()
        self._client = openai.AsyncOpenAI(**self._provider_config.client_kwargs())

    @property
    def provider(self) -> str:
        return self._provider_config.name

    async def embed(self, texts: list[str]) -> list[list[float]]:
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
        result = await self.embed([query])
        return result[0]
