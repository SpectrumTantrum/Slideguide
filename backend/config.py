"""
Application configuration loaded from environment variables.

Uses pydantic-settings to validate and type all config values at startup.
Fails fast with clear error messages if required variables are missing.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """SlideGuide application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # OpenAI-compatible endpoint. Point openai_base_url at ANY OpenAI-compatible
    # /v1 API you choose — real OpenAI, Ollama, vLLM, LocalAI, OpenRouter,
    # LM Studio, Together, Groq, or a self-hosted gateway. The API key may be
    # left empty for endpoints that don't require auth (a placeholder is sent so
    # the OpenAI SDK still initializes). Model names are whatever your endpoint
    # serves. The embedding model must return 1536-d vectors (pgvector schema).
    openai_base_url: str = "https://api.openai.com/v1"
    openai_api_key: str = ""
    primary_model: str = ""
    routing_model: str = ""  # falls back to primary_model if empty
    embedding_model: str = ""
    vision_model: str = ""  # optional; leave empty to disable vision

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
    def active_primary_model(self) -> str:
        return self.primary_model

    @property
    def active_routing_model(self) -> str:
        return self.routing_model or self.primary_model

    @property
    def active_embedding_model(self) -> str:
        return self.embedding_model

    @property
    def active_vision_model(self) -> str:
        return self.vision_model

    @property
    def is_production(self) -> bool:
        return self.environment == "production"

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024


settings = Settings()
