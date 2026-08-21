"""Tests for multi-SDK provider selection and the Cursor route."""

from __future__ import annotations

import pytest

from backend.config import settings
from backend.llm.base import (
    chat_completion_dict,
    extract_image_payloads,
    format_messages_as_prompt,
    message_text,
)
from backend.llm.models import (
    CURSOR_PREFERRED_MODELS,
    cursor_model_chain,
    get_fallback_chain,
    is_cursor_owned_model,
    prefer_cursor_models,
)
from backend.llm.providers import available_providers, provider_metadata
from backend.llm.runtime import (
    get_active_provider,
    reset_active_provider,
    set_active_provider,
)
from backend.llm.tool_compatibility import ToolCompatibilityLayer


@pytest.fixture(autouse=True)
def _reset_provider():
    reset_active_provider()
    yield
    reset_active_provider()


class TestCursorModelPreference:
    def test_composer_is_cursor_owned(self):
        assert is_cursor_owned_model("composer-2.5") is True
        assert is_cursor_owned_model("auto-smart") is True
        assert is_cursor_owned_model("gpt-4o-mini") is False

    def test_chain_starts_with_composer(self):
        chain = cursor_model_chain()
        assert chain[0] == "composer-2.5"
        assert chain[:3] == list(CURSOR_PREFERRED_MODELS)

    def test_explicit_cursor_model_stays_first(self):
        chain = cursor_model_chain("auto-smart")
        assert chain[0] == "auto-smart"
        assert "composer-2.5" in chain

    def test_third_party_primary_is_appended(self):
        chain = cursor_model_chain("gpt-5.5")
        assert chain[0] == "composer-2.5"
        assert chain[-1] == "gpt-5.5"

    def test_catalog_sorts_cursor_first(self):
        ordered = prefer_cursor_models(["gpt-5.5", "auto-smart", "composer-2.5", "claude-4"])
        assert ordered[:2] == ["composer-2.5", "auto-smart"]
        assert ordered[-2:] == ["gpt-5.5", "claude-4"]

    def test_fallback_chain_openai_is_primary(self):
        assert get_fallback_chain("openai") == [settings.primary_model]

    def test_fallback_chain_cursor_prefers_composer(self):
        chain = get_fallback_chain("cursor")
        assert chain[0] == "composer-2.5"
        assert "auto-smart" in chain


class TestRuntimeSwitch:
    def test_default_provider_is_openai(self):
        assert get_active_provider() == "openai"

    def test_cannot_switch_to_cursor_without_key(self, monkeypatch):
        monkeypatch.setattr(settings, "cursor_api_key", "")
        with pytest.raises(ValueError, match="CURSOR_API_KEY"):
            set_active_provider("cursor")

    def test_switch_to_cursor_prefers_composer(self, monkeypatch):
        monkeypatch.setattr(settings, "cursor_api_key", "crsr_test")
        monkeypatch.setattr(settings, "primary_model", "gpt-4o-mini")
        monkeypatch.setattr(settings, "cursor_model", "")
        set_active_provider("cursor")
        assert get_active_provider() == "cursor"
        assert settings.active_primary_model == "composer-2.5"
        assert get_fallback_chain()[0] == "composer-2.5"
        assert "gpt-4o-mini" in get_fallback_chain()

    def test_unknown_provider_rejected(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            set_active_provider("anthropic")


class TestProviderCatalog:
    def test_cursor_listed_first(self):
        rows = available_providers()
        assert rows[0]["id"] == "cursor"
        assert rows[0]["sdk"] == "cursor-sdk"
        assert rows[0]["usage"] == "cursor_subscription"

    def test_cursor_configured_flag(self, monkeypatch):
        monkeypatch.setattr(settings, "cursor_api_key", "")
        assert available_providers()[0]["configured"] is False
        monkeypatch.setattr(settings, "cursor_api_key", "crsr_test")
        assert available_providers()[0]["configured"] is True

    def test_metadata(self):
        meta = provider_metadata("cursor")
        assert meta["sdk"] == "cursor-sdk"
        assert meta["usage"] == "cursor_subscription"


class TestPromptFormatting:
    def test_message_text_flattens_parts(self):
        parts = [{"type": "text", "text": "Hello"}, {"type": "text", "text": "there"}]
        assert message_text(parts) == "Hello\nthere"

    def test_format_includes_roles(self):
        prompt = format_messages_as_prompt([
            {"role": "system", "content": "You are a tutor."},
            {"role": "user", "content": "Explain osmosis."},
        ])
        assert "[SYSTEM]" in prompt
        assert "You are a tutor." in prompt
        assert "[USER]" in prompt
        assert prompt.endswith("[ASSISTANT]")

    def test_extracts_data_images_only(self):
        images = extract_image_payloads([
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "what is this?"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc123"}},
                    {"type": "image_url", "image_url": {"url": "https://example.com/x.png"}},
                ],
            }
        ])
        assert images == [("abc123", "image/png")]

    def test_chat_completion_shape(self):
        payload = chat_completion_dict(
            "hi", model="composer-2.5", prompt_tokens=2, completion_tokens=1
        )
        assert payload["choices"][0]["message"]["content"] == "hi"
        assert payload["usage"]["total_tokens"] == 3


class TestToolMode:
    def test_cursor_forces_prompt_tools(self, monkeypatch):
        monkeypatch.setattr(settings, "cursor_api_key", "crsr_test")
        layer = ToolCompatibilityLayer()
        assert layer.mode == "native"
        set_active_provider("cursor")
        assert layer.mode == "prompt"


class TestCursorChatProvider:
    @pytest.mark.asyncio
    async def test_chat_returns_openai_shape(self, monkeypatch):
        from backend.llm.cursor_provider import CursorChatProvider

        class FakeUsage:
            input_tokens = 10
            output_tokens = 4
            total_tokens = 14

        class FakeResult:
            status = "finished"
            result = "Photosynthesis converts light to chemical energy."
            id = "run-1"
            model = type("M", (), {"id": "composer-2.5"})()
            usage = FakeUsage()

        class FakeRun:
            def wait(self):
                return FakeResult()

            def iter_text(self):
                yield "Photosynthesis"
                return
                yield

        class FakeAgent:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def send(self, message):
                assert "Explain photosynthesis" in message
                return FakeRun()

        class FakeAgentType:
            @staticmethod
            def create(options):
                assert options.model == "composer-2.5"
                assert options.tools == []
                return FakeAgent()

        monkeypatch.setattr(settings, "cursor_api_key", "crsr_test")
        import cursor_sdk

        monkeypatch.setattr(cursor_sdk, "Agent", FakeAgentType)

        provider = CursorChatProvider()
        response = await provider.chat(
            messages=[{"role": "user", "content": "Explain photosynthesis"}],
            model="composer-2.5",
        )
        assert response["choices"][0]["message"]["content"].startswith("Photosynthesis")
        assert response["usage"]["prompt_tokens"] == 10

    @pytest.mark.asyncio
    async def test_stream_chat_yields_deltas(self, monkeypatch):
        from backend.llm.cursor_provider import CursorChatProvider

        class FakeResult:
            status = "finished"
            result = "hello world"
            id = "run-2"
            model = type("M", (), {"id": "composer-2.5"})()
            usage = None

        class FakeRun:
            def wait(self):
                return FakeResult()

            def iter_text(self):
                yield "hello "
                yield "world"

        class FakeAgent:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def send(self, message):
                return FakeRun()

        class FakeAgentType:
            @staticmethod
            def create(options):
                return FakeAgent()

        monkeypatch.setattr(settings, "cursor_api_key", "crsr_test")
        import cursor_sdk

        monkeypatch.setattr(cursor_sdk, "Agent", FakeAgentType)

        provider = CursorChatProvider()
        texts: list[str] = []
        finish = None
        async for chunk in provider.stream_chat(
            messages=[{"role": "user", "content": "Hi"}],
            model="composer-2.5",
        ):
            delta = chunk["choices"][0]["delta"]["content"]
            if delta:
                texts.append(delta)
            finish = chunk["choices"][0]["finish_reason"]
        assert "".join(texts) == "hello world"
        assert finish == "stop"


class TestSettingsRoutes:
    def test_switch_to_cursor_without_key_is_400(self, monkeypatch):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from backend.routes.settings import router

        monkeypatch.setattr(settings, "cursor_api_key", "")
        app = FastAPI()
        app.include_router(router)
        response = TestClient(app).post("/api/settings/provider", json={"provider": "cursor"})
        assert response.status_code == 400
        assert "CURSOR_API_KEY" in response.json()["detail"]

    def test_get_provider_includes_available_sdks(self, monkeypatch):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from backend.routes.settings import router

        async def fake_chat_health():
            return {"status": "ok", "models_loaded": 3}

        async def fake_embed_health(_url: str):
            return {"status": "ok", "models_loaded": 1}

        monkeypatch.setattr(
            "backend.routes.settings.check_active_provider_health", fake_chat_health
        )
        monkeypatch.setattr(
            "backend.routes.settings.check_endpoint_health", fake_embed_health
        )

        app = FastAPI()
        app.include_router(router)
        response = TestClient(app).get("/api/settings/provider")
        assert response.status_code == 200
        body = response.json()
        assert body["provider"] == "openai"
        assert body["sdk"] == "openai"
        assert {row["id"] for row in body["available_providers"]} == {"cursor", "openai"}
