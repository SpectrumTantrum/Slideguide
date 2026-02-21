"""
SSE (Server-Sent Events) streaming handler.

Converts LLM streaming deltas into SSE-formatted events for the frontend.
Handles tool call assembly from partial deltas and provides heartbeat.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, AsyncGenerator

from backend.monitoring.logger import get_logger

logger = get_logger(__name__)

HEARTBEAT_INTERVAL = 15.0  # seconds


class SSEHandler:
    """Convert LLM stream deltas to Server-Sent Events format."""

    def __init__(self) -> None:
        self._tool_call_buffer: dict[int, dict[str, Any]] = {}

    async def stream_response(
        self,
        llm_stream: AsyncGenerator[dict[str, Any], None],
    ) -> AsyncGenerator[str, None]:
        """
        Convert an LLM streaming response to SSE events.

        Yields SSE-formatted strings ready to send to the client.
        Events: stream_start, token, tool_call, stream_end, error
        """
        yield self._format_event("stream_start", {})
        last_heartbeat = time.time()

        try:
            async for chunk in llm_stream:
                choice = chunk.get("choices", [{}])[0]
                delta = choice.get("delta", {})
                finish_reason = choice.get("finish_reason")

                # Content token
                content = delta.get("content")
                if content:
                    yield self._format_event("token", {"text": content})

                # Tool calls (may arrive as partial deltas)
                tool_calls = delta.get("tool_calls")
                if tool_calls:
                    for tc in tool_calls:
                        assembled = self._assemble_tool_call(tc)
                        if assembled:
                            yield self._format_event("tool_call", assembled)

                # Stream finished
                if finish_reason:
                    yield self._format_event("stream_end", {
                        "finish_reason": finish_reason,
                    })

                # Heartbeat to keep connection alive
                now = time.time()
                if now - last_heartbeat > HEARTBEAT_INTERVAL:
                    yield self._format_event("heartbeat", {})
                    last_heartbeat = now

        except Exception as e:
            logger.error("sse_stream_error", error=str(e))
            yield self._format_event("error", {"message": str(e)})

    def _assemble_tool_call(self, partial: dict[str, Any]) -> dict[str, Any] | None:
        """
        Assemble partial tool call deltas into complete tool calls.

        OpenAI streams tool calls as: index + (id, type, function.name, function.arguments)
        where arguments arrive incrementally.
        """
        index = partial.get("index", 0)

        if index not in self._tool_call_buffer:
            self._tool_call_buffer[index] = {
                "id": "",
                "type": "function",
                "function": {"name": "", "arguments": ""},
            }

        buf = self._tool_call_buffer[index]

        if "id" in partial and partial["id"]:
            buf["id"] = partial["id"]

        func = partial.get("function", {})
        if "name" in func and func["name"]:
            buf["function"]["name"] = func["name"]
        if "arguments" in func and func["arguments"]:
            buf["function"]["arguments"] += func["arguments"]

        # Check if we have a complete tool call (has closing brace in arguments)
        args = buf["function"]["arguments"]
        if args and buf["function"]["name"]:
            try:
                json.loads(args)  # Valid JSON = complete
                complete = dict(buf)
                del self._tool_call_buffer[index]
                return complete
            except json.JSONDecodeError:
                pass  # Still accumulating

        return None

    @staticmethod
    def _format_event(event: str, data: dict[str, Any]) -> str:
        """Format a dict as an SSE event string."""
        payload = json.dumps({"event": event, "data": data})
        return f"data: {payload}\n\n"
