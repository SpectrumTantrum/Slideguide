"""
Application configuration loaded from environment variables.

Uses pydantic-settings to validate and type all config values at startup.
Fails fast with clear error messages if required variables are missing.
"""

from __future__ import annotations

from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """SlideGuide application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Provider selection
    llm_provider: Literal["openrouter", "lmstudio"] = "openrouter"
    embedding_provider: Literal["openrouter", "lmstudio"] = "openrouter"
    vision_provider: Literal["openrouter", "lmstudio"] = "openrouter"

    # OpenRouter (LLM gateway + embeddings)
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    # LM Studio (local models)
    lmstudio_base_url: str = "http://localhost:1234/v1"
    lmstudio_primary_model: str = ""
    lmstudio_routing_model: str = ""
    lmstudio_embedding_model: str = ""

    # Model IDs (OpenRouter format, used when llm_provider=openrouter)
    primary_model: str = "anthropic/claude-sonnet-4"
    routing_model: str = "anthropic/claude-haiku-4"
    vision_model: str = "anthropic/claude-sonnet-4"
    fallback_model: str = "deepseek/deepseek-chat-v3"
    embedding_model: str = "openai/text-embedding-3-small"

    # Supabase
    supabase_url: str = "http://127.0.0.1:54321"
    supabase_anon_key: str = ""
    supabase_service_role_key: str = ""

    # Tesseract OCR
    tesseract_cmd: str = "tesseract"

    # Application
    app_name: str = "SlideGuide"
    app_url: str = "https://github.com/yourusername/slideguide"
    environment: str = "development"
    log_level: str = "DEBUG"

    # Cost control
    max_tokens_per_session: int = 100_000
    max_upload_size_mb: int = 50

    @property
    def is_local_llm(self) -> bool:
        return self.llm_provider == "lmstudio"

    @property
    def is_local_embeddings(self) -> bool:
        return self.embedding_provider == "lmstudio"

    @property
    def active_primary_model(self) -> str:
        if self.is_local_llm:
            return self.lmstudio_primary_model
        return self.primary_model

    @property
    def active_routing_model(self) -> str:
        if self.is_local_llm:
            return self.lmstudio_routing_model or self.lmstudio_primary_model
        return self.routing_model

    @property
    def active_embedding_model(self) -> str:
        if self.is_local_embeddings:
            return self.lmstudio_embedding_model
        return self.embedding_model

    @property
    def active_vision_model(self) -> str:
        if self.vision_provider == "lmstudio":
            return self.lmstudio_primary_model
        return self.vision_model

    @property
    def is_production(self) -> bool:
        return self.environment == "production"

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024


settings = Settings()
