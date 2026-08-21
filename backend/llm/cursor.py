"""
Cursor SDK translator.

Turns SlideGuide chat messages into a text-only ``cursor-sdk`` agent run
and returns plain text. LLMClient owns HTTP/OpenAI shaping.

Tutoring always uses a local agent with ``tools=[]`` and ``mcp_servers={}``.
``CURSOR_RUNTIME=cloud`` is refused — team MCP/hooks cannot be blocked there.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncGenerator, Iterator
from dataclasses import dataclass
from pathlib import Path
from tempfile import gettempdir
from typing import Any

from backend.config import settings
from backend.llm.models import (
    CURSOR_DEFAULT_MODEL,
    CURSOR_PREFERRED_MODELS,
    cursor_model_request,
    is_cursor_owned_model,
    prefer_cursor_models,
)
from backend.llm.runtime import current_session_id
from backend.monitoring.logger import get_logger

logger = get_logger(__name__)

# Process-wide registry so the SSE event-loop thread can cancel worker runs.
_active_runs_lock = threading.Lock()
_active_runs: dict[str, list[Any]] = {}


class CursorNotConfiguredError(RuntimeError):
    """Raised when the Cursor SDK is selected but no API key is set."""


class CursorCloudRuntimeError(RuntimeError):
    """Raised when tutoring would run under cloud agents that load team tools."""


@dataclass(frozen=True)
class CursorReply:
    """Plain translator output — not an OpenAI payload."""

    text: str
    model: str
    run_id: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0


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
    """Collect ``(base64_data, mime_type)`` pairs from ``data:`` image_url parts."""
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
    """Render a chat transcript as a single prompt for the agent SDK."""
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


def workspace_dir(session_id: str | None = None) -> str:
    """Per-session cwd. Not a shared durable /tmp folder."""
    if settings.cursor_workspace.strip():
        root = Path(settings.cursor_workspace).expanduser()
    else:
        root = Path(gettempdir()) / "slideguide-cursor"
    sid = (session_id or current_session_id() or "unbound").replace("/", "_")
    path = root / sid
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def cancel_active_runs(session_id: str | None = None) -> None:
    """Cancel tracked Cursor runs (SSE disconnect from the event-loop thread)."""
    sid = session_id or current_session_id()
    with _active_runs_lock:
        if sid:
            runs = list(_active_runs.pop(sid, []))
        else:
            runs = []
            for key in list(_active_runs):
                runs.extend(_active_runs.pop(key))
    for run in runs:
        cancel = getattr(run, "cancel", None)
        if cancel:
            try:
                cancel()
            except Exception as exc:
                logger.warning("cursor_run_cancel_failed", error=str(exc))


class CursorTranslator:
    """One Cursor agent run per tutoring call."""

    def chat(
        self,
        messages: list[dict[str, Any]],
        model: str,
        session_id: str | None = None,
    ) -> CursorReply:
        result = self._run_once(messages, model, session_id=session_id)
        usage = getattr(result, "usage", None)
        return CursorReply(
            text=getattr(result, "result", None) or "",
            model=getattr(getattr(result, "model", None), "id", None) or model,
            run_id=getattr(result, "id", "") or "",
            prompt_tokens=getattr(usage, "input_tokens", 0) or 0,
            completion_tokens=getattr(usage, "output_tokens", 0) or 0,
        )

    def iter_text(
        self,
        messages: list[dict[str, Any]],
        model: str,
        session_id: str | None = None,
    ) -> Iterator[str]:
        yield from self._iter_text(messages, model, session_id=session_id)

    async def chat_async(
        self, messages: list[dict[str, Any]], model: str
    ) -> CursorReply:
        session_id = current_session_id()
        return await asyncio.to_thread(self.chat, messages, model, session_id)

    async def stream_text(
        self, messages: list[dict[str, Any]], model: str
    ) -> AsyncGenerator[str, None]:
        # Capture before the worker thread; ContextVar does not cross threads.
        session_id = current_session_id()
        queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def worker() -> None:
            try:
                for chunk in self.iter_text(messages, model, session_id=session_id):
                    loop.call_soon_threadsafe(queue.put_nowait, ("token", chunk))
                loop.call_soon_threadsafe(queue.put_nowait, ("done", None))
            except Exception as exc:
                loop.call_soon_threadsafe(queue.put_nowait, ("error", exc))

        thread = threading.Thread(target=worker, name="cursor-sdk-stream", daemon=True)
        thread.start()
        while True:
            kind, payload = await queue.get()
            if kind == "token":
                yield payload
            elif kind == "done":
                return
            else:
                raise payload

    def list_models_sync(self) -> list[dict[str, Any]]:
        from cursor_sdk import Cursor

        api_key = settings.cursor_api_key.strip() or None
        catalog = Cursor.models.list(api_key=api_key)
        models = [
            {
                "id": item.id,
                "object": "model",
                "display_name": getattr(item, "display_name", None) or item.id,
                "preferred": is_cursor_owned_model(item.id),
            }
            for item in catalog
            if getattr(item, "id", None)
        ]
        return models or _fallback_catalog()

    async def list_models(self) -> list[dict[str, Any]]:
        try:
            catalog = await asyncio.to_thread(self.list_models_sync)
        except Exception as exc:
            logger.warning("cursor_model_list_failed", error=str(exc))
            catalog = _fallback_catalog()
        ordered_ids = prefer_cursor_models([m["id"] for m in catalog if m.get("id")])
        by_id = {m["id"]: m for m in catalog}
        return [by_id[model_id] for model_id in ordered_ids if model_id in by_id]

    async def health(self) -> dict[str, Any]:
        if not settings.cursor_api_key.strip():
            return {"status": "unconfigured", "models_loaded": 0}
        try:
            models = await asyncio.to_thread(self.list_models_sync)
            return {"status": "ok", "models_loaded": len(models)}
        except Exception as exc:
            logger.warning("cursor_health_failed", error=str(exc))
            return {"status": "unreachable", "models_loaded": 0}

    def agent_options(self, model: str, session_id: str | None = None) -> Any:
        from cursor_sdk import AgentOptions, LocalAgentOptions

        api_key = settings.cursor_api_key.strip()
        if not api_key:
            raise CursorNotConfiguredError(
                "CURSOR_API_KEY is required to bill tutoring calls to Cursor usage."
            )
        if settings.cursor_runtime == "cloud":
            # Cloud agents still load team MCP servers and hooks; tools=[] does
            # not remove them. Tutoring must stay on an isolated local agent.
            raise CursorCloudRuntimeError(
                "CURSOR_RUNTIME=cloud is not supported for tutoring: team MCP "
                "servers and hooks still load. Set CURSOR_RUNTIME=local."
            )

        return AgentOptions(
            model=_sdk_model(model),
            api_key=api_key,
            tools=[],
            mcp_servers={},
            local=LocalAgentOptions(
                cwd=workspace_dir(session_id),
                setting_sources=[],
            ),
        )

    def _run_once(
        self,
        messages: list[dict[str, Any]],
        model: str,
        session_id: str | None = None,
    ) -> Any:
        from cursor_sdk import Agent

        sid = session_id or current_session_id()
        options = self.agent_options(model, session_id=sid)
        prompt, images = format_messages_as_prompt(messages), extract_image_payloads(messages)
        with Agent.create(options) as agent:
            run = agent.send(_user_message(prompt, images))
            _track_run(run, sid)
            try:
                result = run.wait()
            finally:
                _untrack_run(run, sid)
        _raise_if_failed(result)
        return result

    def _iter_text(
        self,
        messages: list[dict[str, Any]],
        model: str,
        session_id: str | None = None,
    ) -> Any:
        from cursor_sdk import Agent

        sid = session_id or current_session_id()
        options = self.agent_options(model, session_id=sid)
        prompt, images = format_messages_as_prompt(messages), extract_image_payloads(messages)
        with Agent.create(options) as agent:
            run = agent.send(_user_message(prompt, images))
            _track_run(run, sid)
            try:
                yield from run.iter_text()
                result = run.wait()
            finally:
                _untrack_run(run, sid)
        _raise_if_failed(result)


def _sdk_model(model: str) -> Any:
    from cursor_sdk import ModelParameterValue, ModelSelection

    request = cursor_model_request(model or CURSOR_DEFAULT_MODEL)
    if not request.params:
        return request.id
    return ModelSelection(
        id=request.id,
        params=[ModelParameterValue(id=pid, value=value) for pid, value in request.params],
    )


def _user_message(prompt: str, images: list[tuple[str, str]]) -> Any:
    if not images:
        return prompt
    from cursor_sdk import SDKImage, UserMessage

    return UserMessage(
        text=prompt,
        images=[SDKImage.data_image(data, mime) for data, mime in images],
    )


def _raise_if_failed(result: Any) -> None:
    if getattr(result, "status", "finished") not in {"finished", None}:
        raise RuntimeError(
            f"Cursor run ended with status {result.status}: {getattr(result, 'result', '')}"
        )


def _track_run(run: Any, session_id: str | None = None) -> None:
    sid = session_id or current_session_id() or "unbound"
    with _active_runs_lock:
        _active_runs.setdefault(sid, []).append(run)


def _untrack_run(run: Any, session_id: str | None = None) -> None:
    sid = session_id or current_session_id() or "unbound"
    with _active_runs_lock:
        items = _active_runs.get(sid)
        if not items:
            return
        try:
            items.remove(run)
        except ValueError:
            pass
        if not items:
            _active_runs.pop(sid, None)


def _parse_data_url(url: str) -> tuple[str, str] | None:
    if not url.startswith("data:") or ";base64," not in url:
        return None
    header, _, data = url.partition(";base64,")
    mime = header.removeprefix("data:") or "image/png"
    if not data:
        return None
    return data, mime


def _fallback_catalog() -> list[dict[str, Any]]:
    labels = {
        "grok-4.6": "Grok 4.6",
        "composer-2.5": "Composer 2.5",
        "composer-2": "Composer 2",
        "auto-smart": "Cursor Router",
    }
    return [
        {
            "id": model_id,
            "object": "model",
            "display_name": labels.get(model_id, model_id),
            "preferred": True,
        }
        for model_id in CURSOR_PREFERRED_MODELS
    ]
