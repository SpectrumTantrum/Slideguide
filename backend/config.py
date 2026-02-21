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

    # OpenRouter (LLM gateway)
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    # OpenAI (embeddings only)
    openai_api_key: str = ""

    # Model IDs (OpenRouter format)
    primary_model: str = "anthropic/claude-sonnet-4"
    routing_model: str = "anthropic/claude-haiku-4"
    vision_model: str = "anthropic/claude-sonnet-4"
    fallback_model: str = "deepseek/deepseek-chat-v3"
    embedding_model: str = "text-embedding-3-small"

    # Database
    database_url: str = "postgresql://postgres:postgres@localhost:5432/slideguide"

    # ChromaDB
    chromadb_host: str = "localhost"
    chromadb_port: int = 8000

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
    def is_production(self) -> bool:
        return self.environment == "production"

    @property
    def chromadb_url(self) -> str:
        return f"http://{self.chromadb_host}:{self.chromadb_port}"

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024


settings = Settings()
