"""
OpenAI Python SDK adapter.

Talks to whatever OpenAI-compatible endpoint the user configured. Usage
is billed to that endpoint's key (OpenAI, OpenRouter, a local server, …).
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

import openai

from backend.config import settings
from backend.llm.discovery import check_endpoint_health, fetch_models
from backend.llm.providers import get_embedding_config


class OpenAIChatProvider:
    """Chat + model discovery via the official OpenAI SDK."""

    name = "openai"
    sdk = "openai"

    def __init__(self) -> None:
        self._config = get_embedding_config()
        self._client = openai.AsyncOpenAI(**self._config.client_kwargs())

    async def chat(
        self,
        messages: list[dict[str, Any]],
        model: str,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            kwargs["tools"] = tools

        response = await self._client.chat.completions.create(**kwargs)
        return response.model_dump()

    async def stream_chat(
        self,
        messages: list[dict[str, Any]],
        model: str,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> AsyncGenerator[dict[str, Any], None]:
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        if tools:
            kwargs["tools"] = tools

        stream = await self._client.chat.completions.create(**kwargs)
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
                        "finish_reason": chunk.choices[0].finish_reason if chunk.choices else None,
                    }
                ],
            }

    async def health(self) -> dict[str, Any]:
        return await check_endpoint_health(settings.openai_base_url)

    async def list_models(self) -> list[dict[str, Any]]:
        models = await fetch_models(settings.openai_base_url)
        return [
            {
                "id": m.get("id", ""),
                "object": m.get("object", "model"),
                "display_name": m.get("id", ""),
                "preferred": False,
            }
            for m in models
            if m.get("id")
        ]
