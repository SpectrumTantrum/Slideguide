"""
Tool use compatibility layer for models with inconsistent function calling.

Provides two modes:
1. Native — uses OpenAI-format function calling (``tools`` kwarg)
2. Prompt-based — injects tool schemas into the system prompt and parses
   fenced JSON blocks from the model's text response

Starts in native mode and auto-switches to prompt-based after consecutive
parsing failures, adapting to whatever model is loaded.
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Any

from backend.config import settings
from backend.monitoring.logger import get_logger

logger = get_logger(__name__)

NATIVE_FAILURE_THRESHOLD = 3

# Regex to find tool call JSON in fenced code blocks
_TOOL_CALL_PATTERN = re.compile(
    r"```(?:tool_call|json)?\s*\n?\s*(\{.*?\})\s*\n?\s*```",
    re.DOTALL,
)


def _build_tool_prompt(tools: list[dict[str, Any]]) -> str:
    """Build a system prompt section describing available tools."""
    lines = [
        "\n\n--- TOOL USE ---",
        "You have access to tools. To use a tool, include a JSON block in your response:",
        "```tool_call",
        '{"name": "tool_name", "arguments": {"arg1": "value"}}',
        "```",
        "",
        "You may call multiple tools by including multiple blocks.",
        "Always explain your reasoning before and after tool calls.",
        "",
        "Available tools:",
    ]
    for tool in tools:
        func = tool.get("function", {})
        name = func.get("name", "")
        desc = func.get("description", "")
        params = func.get("parameters", {})
        props = params.get("properties", {})
        required = params.get("required", [])

        param_parts = []
        for pname, pinfo in props.items():
            req_marker = "" if pname in required else "?"
            param_parts.append(f"{pname}{req_marker}")

        sig = f"{name}({', '.join(param_parts)})"
        lines.append(f"- {sig}: {desc}")

    lines.append("--- END TOOL USE ---")
    return "\n".join(lines)


def _parse_tool_calls_from_text(content: str) -> list[dict[str, Any]]:
    """Extract tool call dicts from fenced JSON blocks in model output."""
    tool_calls = []
    for match in _TOOL_CALL_PATTERN.finditer(content):
        raw = match.group(1)
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("tool_call_json_invalid", raw=raw[:200])
            continue

        name = parsed.get("name", "")
        arguments = parsed.get("arguments", {})
        if not name:
            continue

        tool_calls.append({
            "id": f"call_{uuid.uuid4().hex[:12]}",
            "function": {
                "name": name,
                "arguments": json.dumps(arguments),
            },
        })

    return tool_calls


class ToolCompatibilityLayer:
    """
    Wraps LLM chat calls with automatic tool-use mode detection.

    Starts in native mode. If native tool call parsing fails repeatedly,
    switches to prompt-based mode for the rest of the session.
    """

    def __init__(self) -> None:
        self._native_failures: int = 0
        self._mode: str = "native"  # "native" or "prompt"

    @property
    def mode(self) -> str:
        return self._mode

    def _should_use_prompt_mode(self) -> bool:
        """Check if we should use prompt-based tool injection."""
        if settings.llm_provider == "openrouter":
            return False  # Cloud models have reliable native tool use
        return self._mode == "prompt"

    async def wrap_chat_call(
        self,
        llm: Any,
        messages: list[dict[str, Any]],
        model: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> dict[str, Any]:
        """
        Call the LLM with tool use support, using native or prompt-based mode.

        Returns a response dict with the same structure regardless of mode:
        ``response["choices"][0]["message"]`` will contain either native
        ``tool_calls`` or synthesized ones parsed from text.
        """
        if not tools:
            return await llm.chat(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            )

        if self._should_use_prompt_mode():
            return await self._prompt_based_call(
                llm, messages, model, tools, temperature, max_tokens,
            )

        # Try native tool use first
        response = await llm.chat(
            messages=messages,
            model=model,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        message = response.get("choices", [{}])[0].get("message", {})
        tool_calls = message.get("tool_calls")

        if tool_calls:
            # Validate that arguments are parseable JSON
            try:
                for tc in tool_calls:
                    json.loads(tc["function"]["arguments"])
                self._native_failures = 0
                return response
            except (json.JSONDecodeError, KeyError, TypeError):
                self._native_failures += 1
                logger.warning(
                    "native_tool_call_parse_failed",
                    failures=self._native_failures,
                    threshold=NATIVE_FAILURE_THRESHOLD,
                )
                if self._native_failures >= NATIVE_FAILURE_THRESHOLD:
                    self._mode = "prompt"
                    logger.info("tool_mode_switched", new_mode="prompt")
                # Fall through to try prompt-based for this call
                return await self._prompt_based_call(
                    llm, messages, model, tools, temperature, max_tokens,
                )

        # No tool calls in response — model chose not to use tools
        return response

    async def _prompt_based_call(
        self,
        llm: Any,
        messages: list[dict[str, Any]],
        model: str | None,
        tools: list[dict[str, Any]],
        temperature: float,
        max_tokens: int,
    ) -> dict[str, Any]:
        """Call LLM with tool schemas injected into the system prompt."""
        tool_prompt = _build_tool_prompt(tools)

        # Inject tool instructions into the system message
        augmented_messages = []
        system_found = False
        for msg in messages:
            if msg.get("role") == "system" and not system_found:
                augmented_messages.append({
                    **msg,
                    "content": msg["content"] + tool_prompt,
                })
                system_found = True
            else:
                augmented_messages.append(msg)

        if not system_found:
            augmented_messages.insert(0, {
                "role": "system",
                "content": tool_prompt.lstrip("\n"),
            })

        # Call without native tools kwarg
        response = await llm.chat(
            messages=augmented_messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # Parse tool calls from the text response
        message = response.get("choices", [{}])[0].get("message", {})
        content = message.get("content", "")
        parsed_calls = _parse_tool_calls_from_text(content)

        if parsed_calls:
            # Inject synthesized tool_calls into the response
            message["tool_calls"] = parsed_calls
            # Strip the tool call blocks from the content
            cleaned = _TOOL_CALL_PATTERN.sub("", content).strip()
            message["content"] = cleaned

            logger.info(
                "prompt_based_tool_calls_parsed",
                count=len(parsed_calls),
                tools=[tc["function"]["name"] for tc in parsed_calls],
            )

        return response
