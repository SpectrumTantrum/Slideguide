"""Tests for session-scoped chat SDK selection and the Cursor translator."""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.config import settings
from backend.llm.client import LLMClient
from backend.llm.cursor import (
    CursorTranslator,
    cancel_active_runs,
    extract_image_payloads,
    format_messages_as_prompt,
    message_text,
    workspace_dir,
)
from backend.llm.models import (
    CURSOR_DEFAULT_MODEL,
    CURSOR_FALLBACK_MODELS,
    CURSOR_ROUTING_MODELS,
    cursor_model_chain,
    cursor_model_request,
    get_fallback_chain,
    is_cursor_owned_model,
    prefer_cursor_models,
)
from backend.llm.providers import available_providers, provider_metadata
from backend.llm.runtime import (
    bind_chat_sdk,
    clear_chat_context,
    current_chat_sdk,
    get_active_provider,
    models_for,
    normalize_provider,
    parse_provider,
    reset_chat_sdk,
)
from backend.llm.tool_compatibility import ToolCompatibilityLayer


@pytest.fixture(autouse=True)
def _reset_chat_ctx():
    clear_chat_context()
    yield
    clear_chat_context()


class _MemSessions:
    """In-memory SessionRepository stand-in for settings-route tests."""

    def __init__(self, _client: Any = None) -> None:
        self.rows: dict[str, dict[str, Any]] = {
            "sess-1": {
                "id": "sess-1",
                "upload_id": "u1",
                "phase": "GREETING",
                "metadata": {},
            },
            "sess-2": {
                "id": "sess-2",
                "upload_id": "u2",
                "phase": "GREETING",
                "metadata": {"chat_sdk": "openai"},
            },
        }

    def get_by_id(self, session_id: str) -> dict[str, Any] | None:
        return self.rows.get(session_id)

    def update(self, session_id: str, **data: Any) -> dict[str, Any]:
        self.rows[session_id].update(data)
        return self.rows[session_id]


def _settings_client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    repo = _MemSessions()
    monkeypatch.setattr(
        "backend.routes.settings.SessionRepository",
        lambda _client: repo,
    )
    app = FastAPI()
    app.state.supabase = object()
    from backend.routes.settings import router

    app.include_router(router)
    return TestClient(app)


class TestCursorModelPreference:
    def test_composer_is_cursor_owned(self):
        assert is_cursor_owned_model("composer-2.5") is True
        assert is_cursor_owned_model("grok-4.6") is True
        assert is_cursor_owned_model("auto-smart") is True
        assert is_cursor_owned_model("gpt-4o-mini") is False

    def test_teaching_chain_is_short_and_skips_auto_smart(self):
        chain = cursor_model_chain()
        assert chain == list(CURSOR_FALLBACK_MODELS)
        assert "auto-smart" not in chain
        assert chain[0] == "grok-4.6"

    def test_explicit_cursor_model_stays_first(self):
        chain = cursor_model_chain("auto-smart")
        assert chain[0] == "auto-smart"
        assert "grok-4.6" in chain

    def test_third_party_primary_is_dropped(self):
        chain = cursor_model_chain("gpt-5.5")
        assert "gpt-5.5" not in chain
        assert chain[0] == "grok-4.6"

    def test_catalog_sorts_cursor_first(self):
        ordered = prefer_cursor_models(
            ["gpt-5.5", "auto-smart", "composer-2.5", "grok-4.6", "claude-4"]
        )
        assert ordered[:3] == ["grok-4.6", "composer-2.5", "auto-smart"]
        assert ordered[-2:] == ["gpt-5.5", "claude-4"]

    def test_fallback_chain_openai_is_primary(self):
        expected = [settings.primary_model] if settings.primary_model else []
        assert get_fallback_chain("openai") == expected

    def test_fallback_chain_cursor_prefers_grok(self):
        chain = get_fallback_chain("cursor")
        assert chain[0] == "grok-4.6"
        assert "composer-2.5" in chain
        assert "auto-smart" not in chain

    def test_cursor_ignores_openai_routing_and_vision(self, monkeypatch):
        monkeypatch.setattr(settings, "routing_model", "gpt-4o-mini")
        monkeypatch.setattr(settings, "vision_model", "gpt-4o")
        monkeypatch.setattr(settings, "primary_model", "gpt-4o-mini")
        resolved = models_for("cursor")
        assert resolved.routing == CURSOR_ROUTING_MODELS[0]
        assert resolved.vision == resolved.primary
        assert resolved.primary == "grok-4.6"
        assert models_for("cursor", purpose="route").fallback == list(CURSOR_ROUTING_MODELS)

    def test_cursor_keeps_owned_routing_override(self, monkeypatch):
        monkeypatch.setattr(settings, "routing_model", "composer-2")
        assert models_for("cursor").routing == "composer-2"

    def test_grok_defaults_to_high_effort(self):
        request = cursor_model_request()
        assert request.id == CURSOR_DEFAULT_MODEL
        assert request.params == (("reasoning_effort", "high"),)

    def test_composer_has_no_grok_effort_param(self):
        request = cursor_model_request("composer-2.5")
        assert request.id == "composer-2.5"
        assert request.params == ()


class TestRuntimeSwitch:
    def test_default_provider_is_openai(self):
        assert get_active_provider() == "openai"
        assert current_chat_sdk() == "openai"

    def test_bind_is_request_local(self, monkeypatch):
        monkeypatch.setattr(settings, "cursor_api_key", "crsr_test")
        assert current_chat_sdk() == "openai"
        token = bind_chat_sdk("cursor", "sess-1")
        assert current_chat_sdk() == "cursor"
        reset_chat_sdk(token)
        assert current_chat_sdk() == "openai"

    def test_cannot_bind_cursor_without_key(self, monkeypatch):
        monkeypatch.setattr(settings, "cursor_api_key", "")
        with pytest.raises(ValueError, match="CURSOR_API_KEY"):
            bind_chat_sdk("cursor")

    def test_cannot_normalize_cursor_without_key(self, monkeypatch):
        monkeypatch.setattr(settings, "cursor_api_key", "")
        with pytest.raises(ValueError, match="CURSOR_API_KEY"):
            normalize_provider("cursor")
        assert parse_provider("cursor") == "cursor"

    def test_models_for_cursor_does_not_mutate_env_settings(self, monkeypatch):
        monkeypatch.setattr(settings, "cursor_api_key", "crsr_test")
        monkeypatch.setattr(settings, "primary_model", "gpt-4o-mini")
        monkeypatch.setattr(settings, "cursor_model", "")
        token = bind_chat_sdk("cursor")
        try:
            assert current_chat_sdk() == "cursor"
            assert settings.active_primary_model == "gpt-4o-mini"
            assert models_for().primary == "grok-4.6"
            assert "gpt-4o-mini" not in get_fallback_chain()
        finally:
            reset_chat_sdk(token)

    def test_unknown_provider_rejected(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            parse_provider("anthropic")


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


class TestToolMode:
    def test_cursor_forces_prompt_tools(self, monkeypatch):
        monkeypatch.setattr(settings, "cursor_api_key", "crsr_test")
        layer = ToolCompatibilityLayer()
        assert layer.mode == "native"
        token = bind_chat_sdk("cursor")
        try:
            assert layer.mode == "prompt"
        finally:
            reset_chat_sdk(token)
        assert layer.mode == "native"


class TestCursorTranslator:
    def test_workspace_is_per_session(self, monkeypatch, tmp_path):
        monkeypatch.setattr(settings, "cursor_workspace", str(tmp_path / "root"))
        first = workspace_dir("sess-a")
        second = workspace_dir("sess-b")
        assert first != second
        assert first.endswith("sess-a")
        assert second.endswith("sess-b")

    def test_local_options_disable_tools(self, monkeypatch, tmp_path):
        monkeypatch.setattr(settings, "cursor_api_key", "crsr_test")
        monkeypatch.setattr(settings, "cursor_runtime", "local")
        monkeypatch.setattr(settings, "cursor_workspace", str(tmp_path / "root"))
        options = CursorTranslator().agent_options("grok-4.6", session_id="sess-1")
        assert list(options.tools) == []
        assert options.mcp_servers == {}
        assert str(options.local.cwd).endswith("sess-1")

    def test_cloud_runtime_is_refused_for_tutoring(self, monkeypatch):
        monkeypatch.setattr(settings, "cursor_api_key", "crsr_test")
        monkeypatch.setattr(settings, "cursor_runtime", "cloud")
        from backend.llm.cursor import CursorCloudRuntimeError

        with pytest.raises(CursorCloudRuntimeError, match="CURSOR_RUNTIME=cloud"):
            CursorTranslator().agent_options("composer-2.5", session_id="sess-1")

    def test_cancel_active_runs_from_other_thread(self):
        class FakeRun:
            def __init__(self) -> None:
                self.cancelled = False

            def cancel(self) -> None:
                self.cancelled = True

        from backend.llm import cursor as cursor_mod

        run = FakeRun()
        cursor_mod._track_run(run, "sess-cancel")
        # Simulate SSE disconnect on a different logical caller.
        cancel_active_runs("sess-cancel")
        assert run.cancelled is True
        assert "sess-cancel" not in cursor_mod._active_runs

    def test_cancel_active_runs(self):
        class FakeRun:
            def __init__(self) -> None:
                self.cancelled = False

            def cancel(self) -> None:
                self.cancelled = True

        from backend.llm import cursor as cursor_mod

        run = FakeRun()
        cursor_mod._track_run(run, "sess-1")
        cancel_active_runs("sess-1")
        assert run.cancelled is True

    @pytest.mark.asyncio
    async def test_chat_returns_plain_reply(self, monkeypatch):
        class FakeUsage:
            input_tokens = 10
            output_tokens = 4
            total_tokens = 14

        class FakeResult:
            status = "finished"
            result = "Photosynthesis converts light to chemical energy."
            id = "run-1"
            model = type("M", (), {"id": "grok-4.6"})()
            usage = FakeUsage()

        class FakeRun:
            def wait(self):
                return FakeResult()

            def cancel(self) -> None:
                return None

            def iter_text(self):
                yield "Photosynthesis"

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
                model = options.model
                model_id = model if isinstance(model, str) else model.id
                assert model_id == "grok-4.6"
                params = (
                    ()
                    if isinstance(model, str)
                    else tuple((p.id, p.value) for p in model.params)
                )
                assert params == (("reasoning_effort", "high"),)
                assert list(options.tools) == []
                return FakeAgent()

        monkeypatch.setattr(settings, "cursor_api_key", "crsr_test")
        import cursor_sdk

        monkeypatch.setattr(cursor_sdk, "Agent", FakeAgentType)

        reply = await CursorTranslator().chat_async(
            messages=[{"role": "user", "content": "Explain photosynthesis"}],
            model="grok-4.6",
        )
        assert reply.text.startswith("Photosynthesis")
        assert reply.prompt_tokens == 10

    @pytest.mark.asyncio
    async def test_llm_client_unknown_model_falls_through(self, monkeypatch):
        monkeypatch.setattr(settings, "cursor_api_key", "crsr_test")
        client = LLMClient()
        calls: list[str] = []

        async def once(provider, messages, model, tools, temperature, max_tokens):
            del provider, messages, tools, temperature, max_tokens
            calls.append(model)
            if model == "grok-4.6":
                raise RuntimeError("Unknown model grok-4.6 does not exist")
            return {
                "id": "ok",
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "ok",
                            "tool_calls": None,
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {},
            }

        monkeypatch.setattr(client, "_once", once)
        token = bind_chat_sdk("cursor")
        try:
            response = await client.chat(messages=[{"role": "user", "content": "hi"}])
        finally:
            reset_chat_sdk(token)
        assert calls[0] == "grok-4.6"
        assert calls[1] == "composer-2.5"
        assert response["choices"][0]["message"]["content"] == "ok"

    def test_breakers_are_per_sdk_and_reset(self):
        client = LLMClient()
        cursor = client._breaker("cursor")
        openai = client._breaker("openai")
        for _ in range(5):
            cursor.record_failure()
        assert cursor.can_execute() is False
        assert openai.can_execute() is True
        client.reset_breaker("cursor")
        assert cursor.can_execute() is True


class TestVisionMime:
    def test_encode_image_detects_jpeg(self, tmp_path):
        from backend.llm.vision import VisionClient

        jpeg = tmp_path / "slide.jpeg"
        jpeg.write_bytes(b"\xff\xd8\xff\xd9")
        encoded = VisionClient._encode_image(str(jpeg))
        assert encoded is not None
        data, mime = encoded
        assert data
        assert mime == "image/jpeg"

    def test_encode_image_detects_webp(self, tmp_path):
        from backend.llm.vision import VisionClient

        webp = tmp_path / "slide.webp"
        webp.write_bytes(b"RIFF\x00\x00\x00\x00WEBP")
        encoded = VisionClient._encode_image(str(webp))
        assert encoded is not None
        _, mime = encoded
        assert mime == "image/webp"


class TestSettingsRoutes:
    def test_post_without_session_id_is_400(self, monkeypatch):
        monkeypatch.setattr(settings, "cursor_api_key", "crsr_test")
        response = _settings_client(monkeypatch).post(
            "/api/settings/provider", json={"provider": "cursor"}
        )
        assert response.status_code == 400
        assert "session_id" in response.json()["detail"]

    def test_switch_to_cursor_without_key_is_400(self, monkeypatch):
        monkeypatch.setattr(settings, "cursor_api_key", "")
        response = _settings_client(monkeypatch).post(
            "/api/settings/provider",
            json={"provider": "cursor", "session_id": "sess-1"},
        )
        assert response.status_code == 400
        assert "CURSOR_API_KEY" in response.json()["detail"]

    def test_post_updates_only_that_session(self, monkeypatch):
        monkeypatch.setattr(settings, "cursor_api_key", "crsr_test")

        async def fake_chat_health(provider=None):
            return {"status": "ok", "models_loaded": 3}

        async def fake_embed_health(_url: str):
            return {"status": "ok", "models_loaded": 1}

        monkeypatch.setattr("backend.routes.settings.check_chat_health", fake_chat_health)
        monkeypatch.setattr("backend.routes.settings.check_endpoint_health", fake_embed_health)

        client = _settings_client(monkeypatch)
        response = client.post(
            "/api/settings/provider",
            json={"provider": "cursor", "session_id": "sess-1"},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["provider"] == "cursor"
        assert body["scope"] == "session"
        assert body["session_id"] == "sess-1"
        assert body["cursor"]["local_tools_disabled"] is True
        assert body["cursor"]["cloud_loads_team_tools"] is False

        other = client.get("/api/settings/provider", params={"session_id": "sess-2"})
        assert other.status_code == 200
        assert other.json()["provider"] == "openai"

        default = client.get("/api/settings/provider")
        assert default.status_code == 200
        assert default.json()["provider"] == "openai"
        assert default.json()["scope"] == "process_default"

    def test_get_provider_includes_available_sdks(self, monkeypatch):
        async def fake_chat_health(provider=None):
            return {"status": "ok", "models_loaded": 3}

        async def fake_embed_health(_url: str):
            return {"status": "ok", "models_loaded": 1}

        monkeypatch.setattr("backend.routes.settings.check_chat_health", fake_chat_health)
        monkeypatch.setattr("backend.routes.settings.check_endpoint_health", fake_embed_health)

        app = FastAPI()
        from backend.routes.settings import router

        app.include_router(router)
        response = TestClient(app).get("/api/settings/provider")
        assert response.status_code == 200
        body = response.json()
        assert body["provider"] == "openai"
        assert body["sdk"] == "openai"
        assert {row["id"] for row in body["available_providers"]} == {"cursor", "openai"}
