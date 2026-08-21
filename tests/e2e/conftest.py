"""Shared fixtures for Cursor SDK end-to-end tests."""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest

from backend.config import settings
from backend.llm.runtime import reset_active_provider


def cursor_api_key() -> str:
    """Return a Cursor API key from the environment or loaded settings."""
    return (os.environ.get("CURSOR_API_KEY") or settings.cursor_api_key or "").strip()


@pytest.fixture(autouse=True)
def _reset_provider() -> Iterator[None]:
    reset_active_provider()
    yield
    reset_active_provider()


@pytest.fixture
def cursor_configured(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """Configure the Cursor route without calling the live API."""
    monkeypatch.setattr(settings, "cursor_api_key", "crsr_e2e_test")
    monkeypatch.setattr(settings, "cursor_model", "")
    monkeypatch.setattr(settings, "cursor_reasoning_effort", "high")
    monkeypatch.setattr(settings, "cursor_runtime", "local")
    monkeypatch.setattr(settings, "cursor_workspace", str(tmp_path / "cursor-workspace"))
    monkeypatch.setattr(settings, "primary_model", "gpt-4o-mini")
