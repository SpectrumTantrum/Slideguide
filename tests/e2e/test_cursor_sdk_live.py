"""Live Cursor SDK e2e. Skipped unless ``CURSOR_API_KEY`` is set."""

from __future__ import annotations

import asyncio

import pytest

from backend.config import settings
from backend.llm.cursor_provider import CursorChatProvider
from backend.llm.runtime import set_active_provider

from .conftest import cursor_api_key

LIVE_REASON = "CURSOR_API_KEY is not set; live Cursor SDK calls are skipped"
LIVE_TIMEOUT_S = 180

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.live,
    pytest.mark.skipif(not cursor_api_key(), reason=LIVE_REASON),
]


@pytest.fixture
def live_cursor(monkeypatch: pytest.MonkeyPatch, tmp_path) -> str:
    key = cursor_api_key()
    monkeypatch.setattr(settings, "cursor_api_key", key)
    monkeypatch.setattr(settings, "cursor_model", "")
    monkeypatch.setattr(settings, "cursor_reasoning_effort", "high")
    monkeypatch.setattr(settings, "cursor_runtime", "local")
    monkeypatch.setattr(settings, "cursor_workspace", str(tmp_path / "cursor-live"))
    set_active_provider("cursor")
    return key


@pytest.mark.asyncio
async def test_live_cursor_lists_models(live_cursor: str):
    from cursor_sdk import Cursor, SDKModel

    catalog = await asyncio.to_thread(Cursor.models.list, api_key=live_cursor)
    assert catalog
    assert all(isinstance(item, SDKModel) for item in catalog)
    ids = [item.id for item in catalog]
    assert any(item_id.startswith("grok-") or item_id.startswith("composer-") for item_id in ids)

    listed = await CursorChatProvider().list_models()
    assert listed[0]["id"]
    assert listed[0].get("preferred") is True or "grok" in listed[0]["id"]


@pytest.mark.asyncio
async def test_live_cursor_chat_grok_high(live_cursor: str):
    del live_cursor
    provider = CursorChatProvider()
    response = await asyncio.wait_for(
        provider.chat(
            messages=[{"role": "user", "content": "Reply with the single word pong."}],
            model="grok-4.6",
        ),
        timeout=LIVE_TIMEOUT_S,
    )
    text = response["choices"][0]["message"]["content"].strip().lower()
    assert text
    assert "pong" in text or len(text) < 80
    assert response["model"]
