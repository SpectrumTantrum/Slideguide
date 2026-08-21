"""Process e2e for the Cursor SDK route.

These tests go through ``LLMClient``, ``VisionClient``, and the settings
API with official ``cursor-sdk`` types. Only ``Agent.create`` / model
listing is stubbed so CI does not bill Cursor usage.
"""

from __future__ import annotations

import cursor_sdk
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.config import settings
from backend.llm.client import LLMClient
from backend.llm.cursor_provider import CursorChatProvider
from backend.llm.runtime import get_active_provider, set_active_provider
from backend.llm.vision import VisionClient
from backend.routes.settings import router as settings_router

from .fakes import FakeCursorModels, RecordingAgent, model_id_and_params

pytestmark = pytest.mark.e2e


def _settings_client() -> TestClient:
    app = FastAPI()
    app.include_router(settings_router)
    return TestClient(app)


def test_cursor_sdk_exports_required_types():
    from cursor_sdk import (
        Agent,
        AgentOptions,
        CloudAgentOptions,
        Cursor,
        LocalAgentOptions,
        ModelParameterValue,
        ModelSelection,
        SDKImage,
        SDKModel,
        UserMessage,
    )

    assert Agent.create is not None
    assert Cursor.models.list is not None
    assert AgentOptions is cursor_sdk.AgentOptions
    assert LocalAgentOptions is cursor_sdk.LocalAgentOptions
    assert CloudAgentOptions is cursor_sdk.CloudAgentOptions
    assert ModelSelection is cursor_sdk.ModelSelection
    assert ModelParameterValue is cursor_sdk.ModelParameterValue
    assert SDKImage is cursor_sdk.SDKImage
    assert SDKModel is cursor_sdk.SDKModel
    assert UserMessage is cursor_sdk.UserMessage


def test_agent_options_use_official_sdk_types(cursor_configured):
    from cursor_sdk import AgentOptions, LocalAgentOptions, ModelSelection

    provider = CursorChatProvider()
    options = provider._agent_options("grok-4.6")

    assert isinstance(options, AgentOptions)
    assert isinstance(options.model, ModelSelection)
    assert isinstance(options.local, LocalAgentOptions)
    assert options.api_key == "crsr_e2e_test"
    assert list(options.tools) == []
    assert options.cloud is None

    model_id, params = model_id_and_params(options.model)
    assert model_id == "grok-4.6"
    assert params == (("reasoning_effort", "high"),)
    assert str(options.local.cwd) == settings.cursor_workspace
    assert list(options.local.setting_sources or []) == []


def test_cloud_runtime_uses_no_repo_options(cursor_configured, monkeypatch):
    from cursor_sdk import CloudAgentOptions

    monkeypatch.setattr(settings, "cursor_runtime", "cloud")
    options = CursorChatProvider()._agent_options("composer-2.5")

    assert isinstance(options.cloud, CloudAgentOptions)
    assert list(options.cloud.repos or []) == []
    assert options.cloud.auto_create_pr is False
    assert options.local is None
    assert options.model == "composer-2.5"


@pytest.mark.asyncio
async def test_llm_client_chat_sends_grok_high(cursor_configured, monkeypatch):
    RecordingAgent.last = None
    monkeypatch.setattr(cursor_sdk, "Agent", RecordingAgent)
    set_active_provider("cursor")

    client = LLMClient()
    response = await client.chat(
        messages=[{"role": "user", "content": "Explain osmosis."}],
    )

    assert client.provider == "cursor"
    assert response["choices"][0]["message"]["content"].startswith("Osmosis")
    assert response["usage"]["total_tokens"] == 11
    assert RecordingAgent.last is not None

    model_id, params = model_id_and_params(RecordingAgent.last.options.model)
    assert model_id == "grok-4.6"
    assert params == (("reasoning_effort", "high"),)
    assert RecordingAgent.last.options.tools == []
    assert "Explain osmosis." in RecordingAgent.last.message


@pytest.mark.asyncio
async def test_llm_client_stream_yields_tokens(cursor_configured, monkeypatch):
    monkeypatch.setattr(cursor_sdk, "Agent", RecordingAgent)
    set_active_provider("cursor")

    texts: list[str] = []
    finish = None
    async for chunk in LLMClient().stream_chat(
        messages=[{"role": "user", "content": "Hi"}],
        model="composer-2.5",
    ):
        delta = chunk["choices"][0]["delta"]["content"]
        if delta:
            texts.append(delta)
        finish = chunk["choices"][0]["finish_reason"]

    assert "".join(texts) == "Osmosis works."
    assert finish == "stop"
    assert RecordingAgent.last is not None
    assert RecordingAgent.last.options.model == "composer-2.5"


@pytest.mark.asyncio
async def test_vision_sends_sdk_image(cursor_configured, monkeypatch, tmp_path):
    from cursor_sdk import SDKImage, UserMessage

    RecordingAgent.last = None
    monkeypatch.setattr(cursor_sdk, "Agent", RecordingAgent)
    set_active_provider("cursor")

    png = tmp_path / "slide.png"
    png.write_bytes(
        bytes.fromhex(
            "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4"
            "890000000a49444154789c63000100000500010d0a2db40000000049454e44ae426082"
        )
    )

    description = await VisionClient().describe_image(str(png), context="cell membrane")
    assert "Osmosis" in description
    assert RecordingAgent.last is not None
    assert isinstance(RecordingAgent.last.message, UserMessage)
    images = list(RecordingAgent.last.message.images or [])
    assert len(images) == 1
    assert isinstance(images[0], SDKImage)
    assert images[0].mime_type == "image/png"
    assert images[0].data


def test_settings_switch_lists_cursor_catalog(cursor_configured, monkeypatch):
    monkeypatch.setattr(cursor_sdk.Cursor, "models", FakeCursorModels())

    client = _settings_client()
    switched = client.post("/api/settings/provider", json={"provider": "cursor"})
    assert switched.status_code == 200
    body = switched.json()
    assert body["provider"] == "cursor"
    assert body["sdk"] == "cursor-sdk"
    assert body["usage"] == "cursor_subscription"
    assert body["models"]["primary"] == "grok-4.6"
    assert body["models"]["effort"] == "high"
    assert body["capabilities"]["tool_mode"] == "prompt"
    assert body["runtime"] == "local"
    assert get_active_provider() == "cursor"

    models = client.get("/api/settings/models")
    assert models.status_code == 200
    catalog = models.json()
    assert catalog["provider"] == "cursor"
    assert catalog["sdk"] == "cursor-sdk"
    ids = [row["id"] for row in catalog["models"]]
    assert ids[:3] == ["grok-4.6", "composer-2.5", "auto-smart"]
    assert "gpt-5.5" in ids


def test_settings_models_fallback_when_list_fails(cursor_configured, monkeypatch):
    class BoomModels:
        def list(self, **_kwargs):
            raise RuntimeError("cursor models unavailable")

    monkeypatch.setattr(cursor_sdk.Cursor, "models", BoomModels())
    set_active_provider("cursor")

    catalog = _settings_client().get("/api/settings/models").json()
    ids = [row["id"] for row in catalog["models"]]
    assert ids == ["grok-4.6", "composer-2.5", "composer-2", "auto-smart"]
    assert all(row["preferred"] is True for row in catalog["models"])
