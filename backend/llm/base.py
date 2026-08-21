"""
Shared chat-provider contract and OpenAI-shaped response helpers.

Each subscription SDK (OpenAI-compatible, Cursor, …) implements
``ChatProvider`` so the rest of the app can stay SDK-agnostic.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any, Protocol


class ChatProvider(Protocol):
    """Minimal chat/vision surface implemented by each provider SDK."""

    name: str
    sdk: str

    async def chat(
        self,
        messages: list[dict[str, Any]],
        model: str,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> dict[str, Any]: ...

    def stream_chat(
        self,
        messages: list[dict[str, Any]],
        model: str,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> AsyncGenerator[dict[str, Any], None]: ...

    async def health(self) -> dict[str, Any]: ...

    async def list_models(self) -> list[dict[str, Any]]: ...


def message_text(content: Any) -> str:
    """Flatten an OpenAI-style message content field to plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and part.get("type") == "text":
                parts.append(str(part.get("text") or ""))
        return "\n".join(p for p in parts if p)
    return str(content)


def extract_image_payloads(messages: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """
    Collect ``(base64_data, mime_type)`` pairs from OpenAI image_url parts.

    Only ``data:`` URLs are returned; remote http(s) URLs are ignored so we
    never fetch arbitrary hosts from the vision path.
    """
    images: list[tuple[str, str]] = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict) or part.get("type") != "image_url":
                continue
            image_url = part.get("image_url") or {}
            url = image_url.get("url") if isinstance(image_url, dict) else ""
            parsed = _parse_data_url(str(url or ""))
            if parsed:
                images.append(parsed)
    return images


def format_messages_as_prompt(messages: list[dict[str, Any]]) -> str:
    """Render a chat transcript as a single prompt for agent-style SDKs."""
    lines = [
        "You are SlideGuide's tutoring model. Reply with the assistant answer only.",
        "Do not edit files, run commands, or call coding-agent tools.",
        "",
    ]
    for message in messages:
        role = str(message.get("role") or "user").upper()
        text = message_text(message.get("content"))
        if not text:
            continue
        lines.append(f"[{role}]")
        lines.append(text)
        lines.append("")
    lines.append("[ASSISTANT]")
    return "\n".join(lines)


def chat_completion_dict(
    content: str,
    *,
    model: str,
    completion_id: str = "",
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    total_tokens: int = 0,
    finish_reason: str = "stop",
) -> dict[str, Any]:
    """Build an OpenAI ChatCompletion-shaped dict from provider output."""
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
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens or (prompt_tokens + completion_tokens),
        },
    }


def stream_delta_chunk(
    text: str | None,
    *,
    completion_id: str = "",
    finish_reason: str | None = None,
    role: str | None = None,
) -> dict[str, Any]:
    """Build an OpenAI streaming chunk from a text delta."""
    return {
        "id": completion_id or "chatcmpl-slideguide",
        "choices": [
            {
                "delta": {
                    "content": text,
                    "tool_calls": None,
                    "role": role,
                },
                "finish_reason": finish_reason,
            }
        ],
    }


def _parse_data_url(url: str) -> tuple[str, str] | None:
    if not url.startswith("data:") or ";base64," not in url:
        return None
    header, _, data = url.partition(";base64,")
    mime = header.removeprefix("data:") or "image/png"
    if not data:
        return None
    return data, mime
