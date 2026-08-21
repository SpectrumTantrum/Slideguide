"""
Cursor Python SDK adapter.

Chat and vision calls go through ``cursor-sdk`` so token usage is billed
to the user's Cursor subscription (dashboard usage, SDK tag). The Cursor
agent is run text-only (``tools=[]``) against an isolated workspace so it
cannot edit the SlideGuide repo.

Grok 4.6 at high reasoning effort is the default on this route.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncGenerator
from pathlib import Path
from tempfile import gettempdir
from typing import Any

from backend.config import settings
from backend.llm.base import (
    chat_completion_dict,
    extract_image_payloads,
    format_messages_as_prompt,
    stream_delta_chunk,
)
from backend.llm.models import (
    CURSOR_DEFAULT_MODEL,
    CURSOR_PREFERRED_MODELS,
    cursor_model_request,
    is_cursor_owned_model,
    prefer_cursor_models,
)
from backend.monitoring.logger import get_logger

logger = get_logger(__name__)


class CursorNotConfiguredError(RuntimeError):
    """Raised when the Cursor SDK is selected but no API key is set."""


class CursorChatProvider:
    """Chat + vision via the official ``cursor-sdk`` package."""

    name = "cursor"
    sdk = "cursor-sdk"

    def __init__(self) -> None:
        self._workspace = _workspace_dir()

    async def chat(
        self,
        messages: list[dict[str, Any]],
        model: str,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> dict[str, Any]:
        del tools, temperature, max_tokens  # agent SDK has no OpenAI sampling knobs
        result = await asyncio.to_thread(self._run_once, messages, model)
        return self._completion_from_result(result, model)

    async def stream_chat(
        self,
        messages: list[dict[str, Any]],
        model: str,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> AsyncGenerator[dict[str, Any], None]:
        del tools, temperature, max_tokens
        queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def worker() -> None:
            try:
                for chunk in self._iter_text(messages, model):
                    loop.call_soon_threadsafe(queue.put_nowait, ("token", chunk))
                loop.call_soon_threadsafe(queue.put_nowait, ("done", None))
            except Exception as exc:
                loop.call_soon_threadsafe(queue.put_nowait, ("error", exc))

        thread = threading.Thread(target=worker, name="cursor-sdk-stream", daemon=True)
        thread.start()

        completion_id = f"cursor-{model}"
        while True:
            kind, payload = await queue.get()
            if kind == "token":
                yield stream_delta_chunk(payload, completion_id=completion_id, role="assistant")
            elif kind == "done":
                yield stream_delta_chunk(None, completion_id=completion_id, finish_reason="stop")
                return
            else:
                raise payload

    async def health(self) -> dict[str, Any]:
        if not settings.cursor_api_key.strip():
            return {"status": "unconfigured", "models_loaded": 0}
        try:
            models = await asyncio.to_thread(self._list_models_sync)
            return {"status": "ok", "models_loaded": len(models)}
        except Exception as exc:
            logger.warning("cursor_health_failed", error=str(exc))
            return {"status": "unreachable", "models_loaded": 0}

    async def list_models(self) -> list[dict[str, Any]]:
        try:
            catalog = await asyncio.to_thread(self._list_models_sync)
        except Exception as exc:
            logger.warning("cursor_model_list_failed", error=str(exc))
            catalog = _fallback_catalog()

        ordered_ids = prefer_cursor_models([m["id"] for m in catalog if m.get("id")])
        by_id = {m["id"]: m for m in catalog}
        return [by_id[model_id] for model_id in ordered_ids if model_id in by_id]

    def _run_once(self, messages: list[dict[str, Any]], model: str) -> Any:
        from cursor_sdk import Agent

        options = self._agent_options(model)
        prompt, images = self._prompt_and_images(messages)
        with Agent.create(options) as agent:
            run = agent.send(self._user_message(prompt, images))
            result = run.wait()
        if getattr(result, "status", "finished") not in {"finished", None}:
            raise RuntimeError(
                f"Cursor run ended with status {result.status}: {getattr(result, 'result', '')}"
            )
        return result

    def _iter_text(self, messages: list[dict[str, Any]], model: str) -> Any:
        from cursor_sdk import Agent

        options = self._agent_options(model)
        prompt, images = self._prompt_and_images(messages)
        with Agent.create(options) as agent:
            run = agent.send(self._user_message(prompt, images))
            yield from run.iter_text()
            result = run.wait()
        if getattr(result, "status", "finished") not in {"finished", None}:
            raise RuntimeError(
                f"Cursor run ended with status {result.status}: {getattr(result, 'result', '')}"
            )

    def _agent_options(self, model: str) -> Any:
        from cursor_sdk import AgentOptions, CloudAgentOptions, LocalAgentOptions

        api_key = settings.cursor_api_key.strip()
        if not api_key:
            raise CursorNotConfiguredError(
                "CURSOR_API_KEY is required to bill tutoring calls to Cursor usage."
            )

        kwargs: dict[str, Any] = {
            "model": _sdk_model(model),
            "api_key": api_key,
            "tools": [],
        }
        if settings.cursor_runtime == "cloud":
            kwargs["cloud"] = CloudAgentOptions(repos=[], auto_create_pr=False)
        else:
            kwargs["local"] = LocalAgentOptions(
                cwd=self._workspace,
                setting_sources=[],
            )
        return AgentOptions(**kwargs)

    @staticmethod
    def _prompt_and_images(messages: list[dict[str, Any]]) -> tuple[str, list[tuple[str, str]]]:
        return format_messages_as_prompt(messages), extract_image_payloads(messages)

    @staticmethod
    def _user_message(prompt: str, images: list[tuple[str, str]]) -> Any:
        if not images:
            return prompt
        from cursor_sdk import SDKImage, UserMessage

        return UserMessage(
            text=prompt,
            images=[SDKImage.data_image(data, mime) for data, mime in images],
        )

    @staticmethod
    def _completion_from_result(result: Any, model: str) -> dict[str, Any]:
        usage = getattr(result, "usage", None)
        return chat_completion_dict(
            getattr(result, "result", None) or "",
            model=getattr(getattr(result, "model", None), "id", None) or model,
            completion_id=getattr(result, "id", "") or "",
            prompt_tokens=getattr(usage, "input_tokens", 0) or 0,
            completion_tokens=getattr(usage, "output_tokens", 0) or 0,
            total_tokens=getattr(usage, "total_tokens", 0) or 0,
        )

    def _list_models_sync(self) -> list[dict[str, Any]]:
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
        if models:
            return models
        return _fallback_catalog()


def _sdk_model(model: str) -> Any:
    """Turn a model id into a Cursor ``ModelSelection`` (Grok high effort)."""
    from cursor_sdk import ModelParameterValue, ModelSelection

    request = cursor_model_request(model or CURSOR_DEFAULT_MODEL)
    if not request.params:
        return request.id
    return ModelSelection(
        id=request.id,
        params=[
            ModelParameterValue(id=pid, value=value) for pid, value in request.params
        ],
    )


def _workspace_dir() -> str:
    if settings.cursor_workspace.strip():
        path = Path(settings.cursor_workspace).expanduser()
    else:
        path = Path(gettempdir()) / "slideguide-cursor-workspace"
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def _fallback_catalog() -> list[dict[str, Any]]:
    """Static catalog used when the live Cursor model list is unavailable."""
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
